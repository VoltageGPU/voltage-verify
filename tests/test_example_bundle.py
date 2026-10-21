# SPDX-License-Identifier: MIT
"""The reference bundle captured on a VoltageGPU h100-xlarge Confidential VM on 12 September 2026.

Verified offline against the Intel collateral and NVIDIA JWKS embedded at attestation time,
so this test stays deterministic after Intel re-issues TCB info and NVIDIA rotates keys.
"""

import dataclasses
import json
from pathlib import Path

from voltage_verify import pcs, tdx, verify
from voltage_verify.bundle import Bundle
from voltage_verify.cli import main
from voltage_verify.selftest import run_selftest
from voltage_verify.verify import verify_bundle

EXAMPLE = Path(__file__).parent.parent / "examples" / "hello-workload" / "bundle-8xh100-2026-09-12.json"
CHALLENGE = "ef7f54c160627ee536a02b5e73896ac4b1ac2cdefdb7cfaaf2366a7d33cc8731"


def _debug_bundle() -> Bundle:
    """The reference bundle with bit 0 (DEBUG) of the quote's TD attributes set."""
    bundle = Bundle.load(EXAMPLE)
    raw = bytearray(bundle.quote)
    raw[tdx.TD_ATTRIBUTES_OFFSET] |= 0x01
    return dataclasses.replace(bundle, quote=bytes(raw))


def _check(report, name):
    return next(c for c in report.checks if c.name == name)


def test_reference_bundle_verifies_offline():
    bundle = Bundle.load(EXAMPLE)
    report = verify_bundle(bundle, online=False, expected_challenge=CHALLENGE, allowed_hwmodels={"GH100"}, require_devices=8)
    assert report.ok, report.failed()
    assert report.facts["tdx"]["tcb_status"] == "UpToDate"
    assert report.facts["tdx"]["qe_status"] == "UpToDate"
    assert len(report.facts["nvidia"]["devices"]) == 8
    assert bundle.nvidia_mode == "multi-gpu-ppcie"


def test_reference_bundle_rejects_every_mutation():
    bundle = Bundle.load(EXAMPLE)
    clean, results = run_selftest(bundle, online=False, expected_challenge=CHALLENGE)
    assert clean.ok
    assert all(r.passed for r in results), [(r.name, r.failed_checks) for r in results if not r.passed]


def test_reference_bundle_fails_with_another_challenge():
    bundle = Bundle.load(EXAMPLE)
    report = verify_bundle(bundle, online=False, expected_challenge="00" * 32)
    assert not report.ok and "manifest.challenge" in report.failed()


def test_reference_bundle_is_not_a_debug_td():
    report = verify_bundle(Bundle.load(EXAMPLE), online=False, expected_challenge=CHALLENGE)
    debug = _check(report, "tdx.debug")
    assert debug.label == "PASS"
    assert "0000001000000000" in debug.detail


def test_debug_td_bundle_is_rejected_offline():
    # The edited byte also breaks the quote signature; tdx.debug must be among the failures.
    report = verify_bundle(_debug_bundle(), online=False, expected_challenge=CHALLENGE)
    assert not report.ok and "tdx.debug" in report.failed()


def test_signed_debug_td_bundle_fails_on_the_debug_bit_alone(monkeypatch):
    # A real debug TD gets a validly signed quote: stub the signature step to isolate tdx.debug.
    monkeypatch.setattr(tdx, "verify_signatures", lambda quote, **_: None)
    report = verify_bundle(_debug_bundle(), online=False, expected_challenge=CHALLENGE)
    assert report.failed() == ["tdx.debug"]
    allowed = verify_bundle(_debug_bundle(), online=False, expected_challenge=CHALLENGE, allow_debug=True)
    assert allowed.ok
    assert _check(allowed, "tdx.debug").label == "WARN"


def test_debug_td_bundle_is_rejected_online_too(monkeypatch):
    def no_network(*_args, **_kwargs):
        raise pcs.CollateralError("no network in tests")

    monkeypatch.setattr(pcs, "fetch_collateral", no_network)
    monkeypatch.setattr(verify, "fetch_jwks", no_network)
    report = verify_bundle(_debug_bundle(), online=True, expected_challenge=CHALLENGE)
    assert "tdx.debug" in report.failed()


def test_verify_command_allow_debug(tmp_path, capsys):
    path = tmp_path / "bundle.json"
    _debug_bundle().save(path)
    argv = ["verify", str(path), "--offline", "--json", "--challenge", CHALLENGE]
    assert main(argv) == 1
    strict = {c["name"]: c["result"] for c in json.loads(capsys.readouterr().out)["checks"]}
    assert main([*argv, "--allow-debug"]) == 1  # the quote signature still fails
    allowed = {c["name"]: c["result"] for c in json.loads(capsys.readouterr().out)["checks"]}
    assert strict["tdx.debug"] == "FAIL"
    assert allowed["tdx.debug"] == "WARN"
    assert allowed["tdx.signatures"] == "FAIL"

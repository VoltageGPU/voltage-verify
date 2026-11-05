# SPDX-License-Identifier: MIT
"""Offline, the TCB status and revocation come from the bundle's Intel collateral: a warning says so.

Up to 0.2.3 verify --offline judged the TCB status and the revocation lists on the collateral the
bundle carries, and nothing in the output said that Intel was not asked today. Found in the
audit of 5 November 2026. 0.2.4 says it with a warning, tdx.tcb_offline, next to the age of
that collateral (tdx.collateral_fresh), and keeps everything a script reads from 0.2.3: the
line RESULT: VERIFIED, exit code 0, the --json keys ok, checks and facts. quote --offline keeps
its 0.2.3 warning on tdx.collateral (no TCB status, no revocation at all).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from voltage_verify import pcs
from voltage_verify import verify as verify_mod
from voltage_verify.bundle import Bundle
from voltage_verify.cli import main
from voltage_verify.verify import BUNDLE_TCB_OFFLINE, verify_bundle, verify_quote

EXAMPLE = Path(__file__).parent.parent / "examples" / "hello-workload" / "bundle-8xh100-2026-09-12.json"
CHALLENGE = "ef7f54c160627ee536a02b5e73896ac4b1ac2cdefdb7cfaaf2366a7d33cc8731"
THIRD_PARTY = Path(__file__).parent / "fixtures" / "third-party" / "tdx_prod_quote_SPR_E4.dat"
AT = datetime(2026, 9, 12, tzinfo=timezone.utc)
VERIFY_OFFLINE = ["verify", str(EXAMPLE), "--offline", "--challenge", CHALLENGE]


def _names(report) -> list[str]:
    return [c.name for c in report.checks]


def _check(report, name):
    return next((c for c in report.checks if c.name == name), None)


def test_wording():
    assert BUNDLE_TCB_OFFLINE == "TCB and revocation from the bundle's Intel collateral, not checked against Intel today"


def test_offline_bundle_warns_where_the_tcb_came_from():
    report = verify_bundle(Bundle.load(EXAMPLE), online=False, expected_challenge=CHALLENGE, at=AT)
    assert report.ok, report.failed()
    check = _check(report, "tdx.tcb_offline")
    assert check.label == "WARN" and check.detail == BUNDLE_TCB_OFFLINE
    names = _names(report)  # read in that order: the source, its age, then what was judged on it
    assert names.index("tdx.collateral") < names.index("tdx.tcb_offline") < names.index("tdx.collateral_fresh")
    assert names.index("tdx.collateral_fresh") < names.index("tdx.tcb") < names.index("tdx.revocation")


def test_online_bundle_has_no_offline_warning(monkeypatch):
    bundle = Bundle.load(EXAMPLE)
    monkeypatch.setattr(pcs, "fetch_collateral", lambda *a, **k: pcs.Collateral.from_json(bundle.collateral["intel"]))
    monkeypatch.setattr(verify_mod, "fetch_jwks", lambda: bundle.jwks)
    report = verify_bundle(bundle, online=True, expected_challenge=CHALLENGE, at=AT)
    assert report.ok, report.failed()
    assert _check(report, "tdx.tcb_offline") is None and _check(report, "tdx.collateral_fresh") is None


def test_cli_offline_verify_keeps_the_verdict_and_exit_code_of_023(capsys):
    assert main(VERIFY_OFFLINE) == 0  # the reference collateral is weeks old: a warning, not a failure
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-1] == "RESULT: VERIFIED"
    assert any(line.startswith("[WARN] tdx.tcb_offline ") and line.endswith(f"  {BUNDLE_TCB_OFFLINE}") for line in lines)
    assert any(line.startswith("[WARN] tdx.collateral_fresh  STALE: ") for line in lines)


def test_cli_offline_json_keeps_the_keys_of_023(capsys):
    assert main([*VERIFY_OFFLINE, "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert set(out) == {"ok", "checks", "facts"} and out["ok"] is True
    assert {"name": "tdx.tcb_offline", "result": "WARN", "detail": BUNDLE_TCB_OFFLINE} in out["checks"]


def test_cli_offline_failure_stays_exit_1(capsys):
    argv = ["verify", str(EXAMPLE), "--offline", "--challenge", "00" * 32]
    assert main(argv) == 1
    assert capsys.readouterr().out.strip().splitlines()[-1] == "RESULT: NOT VERIFIED (manifest.challenge)"


def test_offline_quote_keeps_its_023_warning_and_exit_code(capsys):
    report = verify_quote(THIRD_PARTY.read_bytes(), online=False)
    assert report.ok and _check(report, "tdx.tcb_offline") is None
    collateral = _check(report, "tdx.collateral")
    assert collateral.label == "WARN"
    assert collateral.detail == "Intel collateral skipped (--offline): TCB status and revocation not checked"
    assert main(["quote", str(THIRD_PARTY), "--offline"]) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == "RESULT: VERIFIED"


def test_selftest_prints_the_verdict_of_023(capsys):
    assert main(["selftest", str(EXAMPLE), "--offline", "--challenge", CHALLENGE]) == 0
    assert capsys.readouterr().out.splitlines()[0] == "clean bundle: VERIFIED"

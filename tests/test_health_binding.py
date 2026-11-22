# SPDX-License-Identifier: MIT
"""Committing a health report in the manifest, and checking it in ``verify``.

The manifest carries the SHA-256 of the report, so the TDX quote (report_data = SHA-512 of the
manifest) and the NVIDIA tokens (nonce = SHA-256 of the manifest) cover the report too. A
manifest without a report is byte for byte what 0.3.0 wrote, and a bundle without a report
verifies exactly as before: same lines, same RESULT, same exit code, same --json keys.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from voltage_verify import nvidia, tdx
from voltage_verify.bundle import Bundle
from voltage_verify.canonical import commitments
from voltage_verify.cli import main
from voltage_verify.health.report import (
    FAIL,
    PASS,
    REQUIRED_CHECKS,
    SCHEMA,
    SKIP,
    WARN,
    HealthCheck,
    HealthReport,
    bind,
    canonical_report,
    report_sha256,
)
from voltage_verify.manifest import ManifestError, build_manifest, validate_manifest
from voltage_verify.verify import verify_bundle

EXAMPLE = Path(__file__).parent.parent / "examples" / "hello-workload" / "bundle-8xh100-2026-09-12.json"
CHALLENGE = "ef7f54c160627ee536a02b5e73896ac4b1ac2cdefdb7cfaaf2366a7d33cc8731"
AT = datetime(2026, 9, 12, tzinfo=timezone.utc)
REPORT_DATA_OFFSET = 568  # docs/BUNDLE_FORMAT.md: report_data sits at byte 568 of the quote

# The lines 0.3.0 prints for the reference bundle, offline with its challenge (captured on
# origin/main, 7421432, before the health module existed).
LINES_0_3_0 = [
    ("manifest.format", "PASS"),
    ("manifest.commitments", "PASS"),
    ("manifest.challenge", "PASS"),
    ("tdx.structure", "PASS"),
    ("tdx.debug", "PASS"),
    ("tdx.report_data", "PASS"),
    ("tdx.measurements", "WARN"),
    ("tdx.signatures", "PASS"),
    ("tdx.collateral", "PASS"),
    ("tdx.collateral_fresh", "PASS"),
    ("tdx.tcb", "PASS"),
    ("tdx.revocation", "PASS"),
    ("tdx.revocation", "PASS"),
    ("nvidia.signatures", "PASS"),
    ("nvidia.nonce", "PASS"),
    ("nvidia.policy", "PASS"),
    ("nvidia.freshness", "PASS"),
    ("nvidia.mode", "PASS"),
    ("nvidia.scope", "WARN"),
]


def _health(*checks: HealthCheck) -> dict:
    """A complete report (every check ``verify`` expects), healthy unless a check is given."""
    given = {c.name: c for c in checks}
    base = [
        HealthCheck(name, PASS, "8x NVIDIA H100 80GB HBM3" if name == "inventory.gpus" else "fine")
        for name in REQUIRED_CHECKS
    ]
    full = [given.pop(c.name, c) for c in base] + list(given.values())
    return HealthReport(
        checks=full,
        created_at=datetime(2026, 9, 11, 22, 50, tzinfo=timezone.utc),
        duration_ms=312000,
        machine={"hostname": "node-1"},
        options={},
        samples={},
    ).to_json()


def _check(report, name):
    return next(c for c in report.checks if c.name == name)


# --- manifest ----------------------------------------------------------------------------------


def test_a_manifest_without_a_report_has_the_keys_of_0_3_0():
    manifest = build_manifest(image_digest="sha256:" + "ab" * 32, image="ghcr.io/org/app:1")
    assert sorted(manifest) == ["challenge", "created_at", "format", "statement", "workload"]
    assert sorted(manifest["workload"]) == ["artifacts", "extra", "image"]


def test_manifest_commits_to_the_report_by_its_sha256(tmp_path):
    doc = _health()
    path = tmp_path / "health.json"
    path.write_bytes(canonical_report(doc))
    manifest = build_manifest(health=path)
    assert manifest["health"] == {"schema": SCHEMA, "sha256": report_sha256(doc)}
    validate_manifest(manifest)
    other = json.loads(json.dumps(manifest))
    other["health"]["sha256"] = "0" * 64
    assert commitments(manifest).sha512 != commitments(other).sha512
    assert commitments(manifest).sha256 != commitments(other).sha256


def test_the_commitment_does_not_depend_on_how_the_report_file_is_indented(tmp_path):
    doc = _health()
    path = tmp_path / "health.json"
    path.write_text(json.dumps(doc, indent=4), encoding="utf-8")
    assert build_manifest(health=path)["health"]["sha256"] == report_sha256(doc)


def test_manifest_refuses_a_file_that_is_not_a_health_report(tmp_path):
    path = tmp_path / "health.json"
    path.write_text('{"schema": "something/else"}')
    with pytest.raises(ManifestError, match="health"):
        build_manifest(health=path)
    with pytest.raises(ManifestError, match="health"):
        build_manifest(health=tmp_path / "missing.json")


@pytest.mark.parametrize(
    "entry",
    [None, "abc", {}, {"schema": SCHEMA}, {"schema": "x/1", "sha256": "0" * 64}, {"schema": SCHEMA, "sha256": "AB" * 32}],
)
def test_validate_manifest_refuses_a_malformed_health_entry(entry):
    manifest = build_manifest()
    manifest["health"] = entry
    with pytest.raises(ManifestError, match="health"):
        validate_manifest(manifest)


def test_manifest_command_takes_health(tmp_path, capsys):
    doc = _health()
    report = tmp_path / "health.json"
    report.write_bytes(canonical_report(doc))
    out = tmp_path / "manifest.json"
    assert main(["manifest", "--health", str(report), "-o", str(out)]) == 0
    assert json.loads(out.read_text())["health"]["sha256"] == report_sha256(doc)
    assert report_sha256(doc) in capsys.readouterr().out
    report.write_text("{}")
    assert main(["manifest", "--health", str(report), "-o", str(out)]) == 2
    assert "health" in capsys.readouterr().err


# --- bind(): the check itself --------------------------------------------------------------------


def test_no_report_anywhere_adds_nothing():
    assert bind(build_manifest(), None) is None


def test_a_committed_report_that_is_not_supplied_is_a_warning():
    manifest = build_manifest()
    manifest["health"] = {"schema": SCHEMA, "sha256": "ab" * 32}
    result = bind(manifest, None)
    assert result is not None and not result.ok and not result.required
    assert "--health" in result.detail and "abababababababab" in result.detail


def test_a_report_the_manifest_does_not_commit_to_fails():
    result = bind(build_manifest(), canonical_report(_health()))
    assert result is not None and not result.ok and result.required
    assert "commits to no health report" in result.detail


def _committed(doc: dict) -> dict:
    manifest = build_manifest()
    manifest["created_at"] = "2026-09-11T23:10:00+00:00"  # twenty minutes after the report
    manifest["health"] = {"schema": SCHEMA, "sha256": report_sha256(doc)}
    return manifest


def test_the_committed_report_passes_and_its_verdict_is_repeated():
    doc = _health()
    result = bind(_committed(doc), canonical_report(doc))
    assert result is not None and result.ok
    assert "verdict of the report: PASS" in result.detail
    assert report_sha256(doc)[:16] in result.detail
    assert result.facts == {
        "schema": SCHEMA,
        "sha256": report_sha256(doc),
        "result": "PASS",
        "verdict": "PASS",
        "created_at": "2026-09-11T22:50:00+00:00",
    }


def test_a_report_altered_after_the_commitment_fails():
    doc = _health(HealthCheck("errors.ecc", FAIL, "GPU 6: 2 uncorrected ECC errors"))
    manifest = _committed(doc)
    next(c for c in doc["checks"] if c["name"] == "errors.ecc")["result"] = "PASS"  # the failure edited out afterwards
    result = bind(manifest, canonical_report(doc))
    assert result is not None and not result.ok and result.required
    assert "NOT the one the manifest commits to" in result.detail


def test_a_committed_report_that_failed_fails_verify_too():
    doc = _health(HealthCheck("errors.ecc", FAIL, "GPU 6: 2 uncorrected ECC errors"))
    result = bind(_committed(doc), canonical_report(doc))
    assert result is not None and not result.ok and result.required
    assert "verdict of the report: FAIL (errors.ecc)" in result.detail


def test_a_committed_report_with_untested_parts_is_a_warning_that_names_them():
    doc = _health(HealthCheck("inventory.gpus", PASS, "1x"), HealthCheck("compute.matmul", SKIP, "no PyTorch"))
    result = bind(_committed(doc), canonical_report(doc))
    assert result is not None and not result.ok and not result.required
    assert "verdict of the report: INCOMPLETE (not tested: compute.matmul)" in result.detail
    assert result.limit == "health report INCOMPLETE (not tested: compute.matmul)"


def test_only_a_report_that_is_not_a_plain_pass_carries_a_limit():
    doc = _health()
    assert bind(_committed(doc), canonical_report(doc)).limit is None
    assert bind(_committed(doc), None).limit == "health report committed but not checked (no --health)"
    warned = _health(HealthCheck("sensors.clocks", WARN, "GPU 0 at a low clock"))
    assert bind(_committed(warned), canonical_report(warned)).limit == "health report WARN (sensors.clocks)"


def test_a_report_whose_verdict_contradicts_its_own_checks_fails():
    doc = _health(HealthCheck("errors.ecc", FAIL, "GPU 6: 2 uncorrected ECC errors"))
    doc.update(ok=True, result="PASS", verdict="PASS")
    result = bind(_committed(doc), canonical_report(doc))
    assert result is not None and not result.ok and result.required
    assert "contradicts" in result.detail


def test_an_unreadable_report_fails():
    manifest = build_manifest()
    manifest["health"] = {"schema": SCHEMA, "sha256": "ab" * 32}
    result = bind(manifest, b"not json")
    assert result is not None and not result.ok and result.required
    assert "unusable" in result.detail


# --- verify: bundles without a report are untouched ---------------------------------------------


def test_reference_bundle_prints_the_lines_of_0_3_0():
    report = verify_bundle(Bundle.load(EXAMPLE), online=False, expected_challenge=CHALLENGE, at=AT)
    assert [(c.name, c.label) for c in report.checks] == LINES_0_3_0
    assert (
        report.verdict == "VERIFIED (TCB and revocation from the bundle's Intel collateral, not checked against Intel today)"
    )
    assert sorted(report.facts) == ["artifacts", "commitment_sha256", "commitment_sha512", "image", "nvidia", "tdx"]
    assert sorted(report.to_json()) == ["checks", "facts", "limits", "ok", "verdict"]


def test_cli_on_the_reference_bundle_is_unchanged(capsys):
    argv = ["verify", str(EXAMPLE), "--offline", "--challenge", CHALLENGE, "--max-collateral-age", "3650"]
    assert main(argv) == 3
    out = capsys.readouterr().out
    assert "health" not in out
    assert out.strip().splitlines()[-1].startswith("RESULT: VERIFIED (TCB and revocation from the bundle's Intel collateral")
    assert main([*argv, "--accept-offline-tcb"]) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == "RESULT: VERIFIED"


# --- verify: a bundle whose manifest commits to a report ----------------------------------------


@pytest.fixture
def attested(monkeypatch):
    """The reference bundle re-issued for a manifest that commits to a health report.

    Only real hardware can sign a new manifest, so the two signature steps are bridged: the
    quote's report_data is rewritten to the new SHA-512 with the Intel signature check stubbed,
    and NRAS is verified against the nonce it really signed. Every other check runs for real.
    """
    original = Bundle.load(EXAMPLE)
    signed_nonce = commitments(original.manifest).sha256_hex
    real_verify_nras = nvidia.verify_nras
    monkeypatch.setattr(tdx, "verify_signatures", lambda quote, **_: None)
    monkeypatch.setattr(
        nvidia, "verify_nras", lambda nras, jwks, _nonce, **kw: real_verify_nras(nras, jwks, signed_nonce, **kw)
    )

    def make(doc: dict) -> Bundle:
        manifest = json.loads(json.dumps(original.manifest))
        manifest["health"] = {"schema": SCHEMA, "sha256": report_sha256(doc)}
        com = commitments(manifest)
        quote = bytearray(original.quote)
        quote[REPORT_DATA_OFFSET : REPORT_DATA_OFFSET + 64] = com.sha512
        return dataclasses.replace(
            original,
            manifest=manifest,
            commitments={"sha256": com.sha256_hex, "sha512": com.sha512_hex},
            quote=bytes(quote),
            report_data=com.sha512,
        )

    return make


def test_verify_passes_health_report_when_the_report_is_the_committed_one(attested):
    doc = _health()
    report = verify_bundle(
        attested(doc), online=False, expected_challenge=CHALLENGE, at=AT, health_report=canonical_report(doc)
    )
    assert report.ok, report.failed()
    assert [(c.name, c.label) for c in report.checks] == [*LINES_0_3_0, ("health.report", "PASS")]
    assert report.facts["health"]["sha256"] == report_sha256(doc)


def test_verify_fails_on_a_report_altered_after_the_commitment(attested):
    doc = _health()
    bundle = attested(doc)
    doc["checks"][0]["detail"] = "8x NVIDIA H200"
    report = verify_bundle(bundle, online=False, expected_challenge=CHALLENGE, at=AT, health_report=canonical_report(doc))
    assert report.failed() == ["health.report"]
    assert report.verdict == "NOT VERIFIED (health.report)"


def test_verify_names_a_committed_report_that_is_not_supplied(attested, tmp_path, capsys):
    # Not a failure, not a plain VERIFIED either: the report may say FAIL, and nobody looked.
    report = verify_bundle(attested(_health()), online=False, expected_challenge=CHALLENGE, at=AT, accept_offline_tcb=True)
    assert report.ok and _check(report, "health.report").label == "WARN"
    assert report.verdict == "VERIFIED (health report committed but not checked (no --health))"

    bundle_path = tmp_path / "bundle.json"
    attested(_health()).save(bundle_path)
    argv = ["verify", str(bundle_path), "--offline", "--challenge", CHALLENGE, "--max-collateral-age", "3650"]
    assert main([*argv, "--accept-offline-tcb"]) == 3
    assert capsys.readouterr().out.strip().splitlines()[-1] == (
        "RESULT: VERIFIED (health report committed but not checked (no --health))"
    )


def test_an_incomplete_report_is_a_limit_named_in_the_verdict(attested, tmp_path, capsys):
    # A SKIP never becomes a pass, here either: VERIFIED says what the report did not test.
    doc = _health(HealthCheck("inventory.gpus", PASS, "1x"), HealthCheck("compute.matmul", SKIP, "no PyTorch"))
    report = verify_bundle(
        attested(doc),
        online=False,
        expected_challenge=CHALLENGE,
        at=AT,
        accept_offline_tcb=True,
        health_report=canonical_report(doc),
    )
    assert report.ok and _check(report, "health.report").label == "WARN"
    assert report.limits == ["health report INCOMPLETE (not tested: compute.matmul)"]
    assert report.verdict == "VERIFIED (health report INCOMPLETE (not tested: compute.matmul))"

    bundle_path, health_path = tmp_path / "bundle.json", tmp_path / "health.json"
    attested(doc).save(bundle_path)
    health_path.write_bytes(canonical_report(doc))
    argv = ["verify", str(bundle_path), "--offline", "--challenge", CHALLENGE, "--max-collateral-age", "3650"]
    assert main([*argv, "--accept-offline-tcb", "--health", str(health_path)]) == 3
    assert capsys.readouterr().out.strip().splitlines()[-1] == (
        "RESULT: VERIFIED (health report INCOMPLETE (not tested: compute.matmul))"
    )


def test_verify_cli_with_health(attested, tmp_path, capsys):
    doc = _health()
    bundle_path, health_path = tmp_path / "bundle.json", tmp_path / "health.json"
    attested(doc).save(bundle_path)
    health_path.write_bytes(canonical_report(doc))
    argv = ["verify", str(bundle_path), "--offline", "--challenge", CHALLENGE, "--max-collateral-age", "3650"]
    argv += ["--accept-offline-tcb", "--health", str(health_path)]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert any(line.startswith("[PASS] health.report ") for line in out.splitlines())
    assert out.strip().splitlines()[-1] == "RESULT: VERIFIED"

    health_path.write_bytes(canonical_report(doc).replace(b"H100", b"H200"))
    assert main(argv) == 1
    out = capsys.readouterr().out
    assert any(line.startswith("[FAIL] health.report ") for line in out.splitlines())
    assert out.strip().splitlines()[-1] == "RESULT: NOT VERIFIED (health.report)"

    assert main([*argv[:-1], str(tmp_path / "missing.json")]) == 2
    assert "cannot read" in capsys.readouterr().err


def test_verify_cli_refuses_a_report_for_a_bundle_that_commits_to_none(tmp_path, capsys):
    health_path = tmp_path / "health.json"
    health_path.write_bytes(canonical_report(_health()))
    argv = ["verify", str(EXAMPLE), "--offline", "--challenge", CHALLENGE, "--max-collateral-age", "3650"]
    assert main([*argv, "--health", str(health_path)]) == 1
    assert capsys.readouterr().out.strip().splitlines()[-1] == "RESULT: NOT VERIFIED (health.report)"

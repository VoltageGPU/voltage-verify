# SPDX-License-Identifier: MIT
"""Offline, the verdict says what was not checked against Intel today, and the exit code is 3.

Up to 0.2.3 `quote --offline` ended on RESULT: VERIFIED and exit code 0 with no TCB status and
no revocation check at all, and `verify --offline` judged the TCB on the collateral the bundle
carries, without saying so in the verdict. The verdict now names the limit, a script sees exit
code 3 instead of 0, and --accept-offline-tcb is the explicit way to accept it (exit 0, the
warning or the source stays in the checks). Found in the audit of 5 November 2026.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from voltage_verify import pcs
from voltage_verify import verify as verify_mod
from voltage_verify.bundle import Bundle
from voltage_verify.cli import main
from voltage_verify.verify import BUNDLE_TCB_OFFLINE, QUOTE_TCB_OFFLINE, verify_bundle, verify_quote

EXAMPLE = Path(__file__).parent.parent / "examples" / "hello-workload" / "bundle-8xh100-2026-09-12.json"
CHALLENGE = "ef7f54c160627ee536a02b5e73896ac4b1ac2cdefdb7cfaaf2366a7d33cc8731"
THIRD_PARTY = Path(__file__).parent / "fixtures" / "third-party" / "tdx_prod_quote_SPR_E4.dat"
AT = datetime(2026, 9, 12, tzinfo=timezone.utc)
VERIFY_OFFLINE = ["verify", str(EXAMPLE), "--offline", "--challenge", CHALLENGE, "--max-collateral-age", "3650"]


def _check(report, name):
    return next(c for c in report.checks if c.name == name)


def _last_line(capsys) -> str:
    return capsys.readouterr().out.strip().splitlines()[-1]


def test_limit_wording():
    assert QUOTE_TCB_OFFLINE == "signatures only, TCB not checked offline"
    assert BUNDLE_TCB_OFFLINE == "TCB and revocation from the bundle's Intel collateral, not checked against Intel today"


# --- quote --offline ---------------------------------------------------------------------------


def test_offline_quote_is_verified_with_a_limit():
    report = verify_quote(THIRD_PARTY.read_bytes(), online=False)
    assert report.ok and report.limits == [QUOTE_TCB_OFFLINE]
    assert report.verdict == "VERIFIED (signatures only, TCB not checked offline)"
    assert _check(report, "tdx.collateral").label == "WARN"


def test_offline_quote_accepted_explicitly():
    report = verify_quote(THIRD_PARTY.read_bytes(), online=False, accept_offline_tcb=True)
    assert report.ok and report.limits == [] and report.verdict == "VERIFIED"
    assert "accepted with --accept-offline-tcb" in _check(report, "tdx.collateral").detail


def test_cli_offline_quote_exits_3_unless_accepted(capsys):
    assert main(["quote", str(THIRD_PARTY), "--offline"]) == 3
    assert _last_line(capsys) == "RESULT: VERIFIED (signatures only, TCB not checked offline)"
    assert main(["quote", str(THIRD_PARTY), "--offline", "--accept-offline-tcb"]) == 0
    assert _last_line(capsys) == "RESULT: VERIFIED"


def test_cli_offline_quote_json_carries_the_verdict_and_limits(capsys):
    assert main(["quote", str(THIRD_PARTY), "--offline", "--json"]) == 3
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True  # every required check passed, as before
    assert out["verdict"] == "VERIFIED (signatures only, TCB not checked offline)"
    assert out["limits"] == [QUOTE_TCB_OFFLINE]
    assert {"ok", "verdict", "limits", "checks", "facts"} == set(out)


def test_cli_failure_stays_exit_1_offline(capsys):
    assert main(["quote", str(THIRD_PARTY), "--offline", "--report-data", "00" * 64]) == 1
    assert _last_line(capsys).startswith("RESULT: NOT VERIFIED (tdx.report_data")


# --- verify --offline (collateral from the bundle) --------------------------------------------


def test_offline_bundle_names_where_the_tcb_came_from():
    bundle = Bundle.load(EXAMPLE)
    report = verify_bundle(bundle, online=False, expected_challenge=CHALLENGE, at=AT)
    assert report.ok and report.limits == [BUNDLE_TCB_OFFLINE]
    assert report.verdict == f"VERIFIED ({BUNDLE_TCB_OFFLINE})"
    accepted = verify_bundle(bundle, online=False, expected_challenge=CHALLENGE, at=AT, accept_offline_tcb=True)
    assert accepted.verdict == "VERIFIED"
    assert "accepted with --accept-offline-tcb" in _check(accepted, "tdx.collateral").detail


def test_online_bundle_has_no_limit(monkeypatch):
    bundle = Bundle.load(EXAMPLE)
    monkeypatch.setattr(pcs, "fetch_collateral", lambda *a, **k: pcs.Collateral.from_json(bundle.collateral["intel"]))
    monkeypatch.setattr(verify_mod, "fetch_jwks", lambda: bundle.jwks)
    report = verify_bundle(bundle, online=True, expected_challenge=CHALLENGE, at=AT)
    assert report.ok and report.limits == [] and report.verdict == "VERIFIED"


def test_cli_offline_verify_exits_3_unless_accepted(capsys):
    assert main(VERIFY_OFFLINE) == 3
    assert _last_line(capsys) == f"RESULT: VERIFIED ({BUNDLE_TCB_OFFLINE})"
    assert main([*VERIFY_OFFLINE, "--accept-offline-tcb"]) == 0
    assert _last_line(capsys) == "RESULT: VERIFIED"


@pytest.mark.parametrize("flag", [[], ["--accept-offline-tcb"]])
def test_cli_offline_verify_that_fails_exits_1(flag, capsys):
    argv = ["verify", str(EXAMPLE), "--offline", "--challenge", "00" * 32, "--max-collateral-age", "3650", *flag]
    assert main(argv) == 1
    assert _last_line(capsys) == "RESULT: NOT VERIFIED (manifest.challenge)"


def test_selftest_prints_the_verdict_of_the_clean_bundle(capsys):
    assert main(["selftest", str(EXAMPLE), "--offline", "--challenge", CHALLENGE, "--max-collateral-age", "3650"]) == 0
    assert capsys.readouterr().out.splitlines()[0] == f"clean bundle: VERIFIED ({BUNDLE_TCB_OFFLINE})"

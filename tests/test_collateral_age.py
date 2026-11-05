# SPDX-License-Identifier: MIT
"""Offline, how old the Intel collateral carried by the bundle is (tdx.collateral_fresh).

Up to 0.2.3 verify --offline judged the TCB status and revocation on the collateral the bundle
carries and said nothing of its age until it was past Intel's nextUpdate ("refetch online").
Found in the audit of 5 November 2026. The age is read from what Intel signed: the oldest issue
date of the TCB info, the QE identity and the PCK CRL, which Intel re-issues continuously with a
nextUpdate 30 days later; the root CA CRL, re-issued once a year, must be within its nextUpdate.

In 0.2.4 the age is a warning, so that every bundle 0.2.3 verified still verifies (same verdict,
exit code 0); past the 30 days Intel gives the collateral the warning says STALE. A verifier
that wants fresh collateral asks for it with --max-collateral-age DAYS, and the check is then
required.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from voltage_verify import pcs
from voltage_verify import verify as verify_mod
from voltage_verify.bundle import Bundle
from voltage_verify.cli import main
from voltage_verify.verify import verify_bundle

EXAMPLE = Path(__file__).parent.parent / "examples" / "hello-workload" / "bundle-8xh100-2026-09-12.json"
CHALLENGE = "ef7f54c160627ee536a02b5e73896ac4b1ac2cdefdb7cfaaf2366a7d33cc8731"
# The reference collateral: QE identity issued 2026-09-11T22:48:40Z, the oldest of the three
# (TCB info 22:53:50, PCK CRL 23:01:34); root CA CRL 2026-02-26 to 2027-02-26.
OLDEST = datetime(2026, 9, 11, 22, 48, 40, tzinfo=timezone.utc)
NOV5 = datetime(2026, 11, 5, tzinfo=timezone.utc)
THIRTY = timedelta(days=30)


@pytest.fixture(scope="module")
def example() -> Bundle:
    return Bundle.load(EXAMPLE)


def _fresh(report):
    return next((c for c in report.checks if c.name == "tdx.collateral_fresh"), None)


def test_intel_gives_its_collateral_30_days():
    assert pcs.COLLATERAL_VALIDITY == THIRTY


# --- by default: a warning that says the age, the verdict of 0.2.3 --------------------------


def test_collateral_of_the_day_is_a_warning_that_says_its_age(example):
    report = verify_bundle(example, online=False, expected_challenge=CHALLENGE, at=OLDEST + timedelta(hours=2))
    check = _fresh(report)
    assert report.ok and check.label == "WARN"
    assert check.detail == (
        "Intel collateral issued 2026-09-11 (QE identity), 0 day(s) old, within the 30-day validity Intel gives it; "
        "root CA CRL valid until 2027-02-26"
    )


def test_stale_collateral_is_a_louder_warning_and_still_verifies(example):
    report = verify_bundle(example, online=False, expected_challenge=CHALLENGE, at=NOV5)
    check = _fresh(report)
    assert report.ok, report.failed()
    assert check.label == "WARN"
    assert check.detail == (
        "STALE: Intel collateral is 54 days old (QE identity issued 2026-09-11), over the 30-day validity Intel "
        "gives it: TCB changes and revocations since then are not seen; verify online"
    )


@pytest.mark.parametrize(("after", "stale"), [(THIRTY, False), (THIRTY + timedelta(seconds=1), True)])
def test_stale_starts_after_30_days_to_the_second(example, after, stale):
    check = _fresh(verify_bundle(example, online=False, at=OLDEST + after))
    assert check.label == "WARN" and check.detail.startswith("STALE: ") is stale


def test_root_crl_past_its_next_update_is_stale(example):
    at = datetime(2027, 3, 1, tzinfo=timezone.utc)  # root CA CRL next update was 2027-02-26
    check = _fresh(verify_bundle(example, online=False, at=at))
    assert check.label == "WARN" and check.detail.startswith("STALE: ")
    assert "root CA CRL past its nextUpdate (2027-02-26)" in check.detail


def test_collateral_without_crls_is_a_warning_by_default(example):
    # 0.2.3 verified such a bundle (revocation "NOT checked" in tdx.revocation): it still does.
    intel = {**example.collateral["intel"], "pck_crl_pem": None}
    report = verify_bundle(dataclasses.replace(example, collateral={"intel": intel}), online=False, at=OLDEST)
    assert report.ok, report.failed()
    assert _fresh(report).label == "WARN" and "no CRL" in _fresh(report).detail


# --- with --max-collateral-age: a required check ------------------------------------------------


def test_required_limit_passes_on_collateral_of_the_day(example):
    report = verify_bundle(example, online=False, at=OLDEST + timedelta(hours=2), max_collateral_age=THIRTY)
    check = _fresh(report)
    assert report.ok and check.label == "PASS"
    assert check.detail == (
        "Intel collateral issued 2026-09-11 (QE identity), 0 day(s) old, within the 30-day limit; "
        "root CA CRL valid until 2027-02-26"
    )


def test_required_limit_fails_stale_collateral_and_names_the_option(example):
    report = verify_bundle(example, online=False, expected_challenge=CHALLENGE, at=NOV5, max_collateral_age=THIRTY)
    check = _fresh(report)
    assert not report.ok and report.failed() == ["tdx.collateral_fresh"]
    assert "54 days old" in check.detail and "QE identity issued 2026-09-11" in check.detail
    assert "over the 30-day limit" in check.detail and "--max-collateral-age 55" in check.detail


@pytest.mark.parametrize(("days", "ok"), [(54, False), (55, True)])
def test_a_wider_limit_is_an_explicit_choice(example, days, ok):
    wider = timedelta(days=days)
    report = verify_bundle(example, online=False, expected_challenge=CHALLENGE, at=NOV5, max_collateral_age=wider)
    assert _fresh(report).ok is ok and report.ok is ok


@pytest.mark.parametrize(("after", "ok"), [(THIRTY, True), (THIRTY + timedelta(seconds=1), False)])
def test_the_limit_is_exact(example, after, ok):
    assert _fresh(verify_bundle(example, online=False, at=OLDEST + after, max_collateral_age=THIRTY)).ok is ok


def test_root_crl_past_its_next_update_needs_the_wider_limit_too(example):
    at = datetime(2027, 3, 1, tzinfo=timezone.utc)
    narrow = _fresh(verify_bundle(example, online=False, at=at, max_collateral_age=timedelta(days=200)))
    assert not narrow.ok and narrow.required and "root CA CRL past its nextUpdate (2027-02-26)" in narrow.detail
    assert "--max-collateral-age 368" in narrow.detail  # last update 2026-02-26 13:04
    assert _fresh(verify_bundle(example, online=False, at=at, max_collateral_age=timedelta(days=368))).ok


def test_required_limit_refuses_collateral_without_crls(example):
    intel = {**example.collateral["intel"], "pck_crl_pem": None}
    bare = dataclasses.replace(example, collateral={"intel": intel})
    report = verify_bundle(bare, online=False, at=OLDEST, max_collateral_age=THIRTY)
    assert not report.ok and report.failed() == ["tdx.collateral_fresh"] and "no CRL" in _fresh(report).detail


def test_online_collateral_is_not_judged_by_the_bundle_age(example, monkeypatch):
    # Online the collateral is Intel's of the moment; here a stub hands back the September copy.
    monkeypatch.setattr(pcs, "fetch_collateral", lambda *a, **k: pcs.Collateral.from_json(example.collateral["intel"]))
    monkeypatch.setattr(verify_mod, "fetch_jwks", lambda: example.jwks)
    for limit in (None, THIRTY):
        report = verify_bundle(example, online=True, expected_challenge=CHALLENGE, at=NOV5, max_collateral_age=limit)
        assert report.ok, report.failed()
        assert _fresh(report).detail == "Intel collateral is past its nextUpdate; refetch online"  # 0.2.3, unchanged


# --- command line ------------------------------------------------------------------------------


def _run_json(argv, capsys) -> tuple[int, dict[str, str]]:
    code = main([*argv, "--json"])
    return code, {c["name"]: c["result"] for c in json.loads(capsys.readouterr().out)["checks"]}


def test_cli_offline_on_an_archived_bundle(capsys):
    argv = ["verify", str(EXAMPLE), "--offline", "--challenge", CHALLENGE]
    code, checks = _run_json(argv, capsys)
    assert code == 0 and checks["tdx.collateral_fresh"] == "WARN"
    code, checks = _run_json([*argv, "--max-collateral-age", "30"], capsys)
    assert code == 1 and checks["tdx.collateral_fresh"] == "FAIL"
    code, checks = _run_json([*argv, "--max-collateral-age", "3650"], capsys)
    assert code == 0 and checks["tdx.collateral_fresh"] == "PASS"


@pytest.mark.parametrize("value", ["-1", "abc", "1.5", "1234567"])
def test_cli_refuses_a_bad_age(value, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["verify", str(EXAMPLE), "--offline", "--max-collateral-age", value])
    assert exc.value.code == 2 and "--max-collateral-age" in capsys.readouterr().err


@pytest.mark.parametrize(("days", "code"), [(None, 0), ("3650", 0), ("30", 1)])
def test_selftest_takes_the_same_option(days, code, capsys):
    limit = ["--max-collateral-age", days] if days else []
    assert main(["selftest", str(EXAMPLE), "--offline", "--challenge", CHALLENGE, *limit]) == code
    assert capsys.readouterr().out.strip().splitlines()[-1] == ("SELFTEST: PASS" if code == 0 else "SELFTEST: FAIL")

# SPDX-License-Identifier: MIT
"""Offline, the Intel collateral carried by the bundle must be recent (--max-collateral-age).

Up to 0.2.3 a stale collateral was a warning: verify --offline of a bundle captured months ago
said VERIFIED on Intel's TCB evaluation and CRLs of that day, with no limit. The check is now
required offline. Its age is read from what Intel signed: the oldest issue date of the TCB info,
the QE identity and the PCK CRL, which Intel re-issues continuously with a nextUpdate 30 days
later; the root CA CRL, re-issued once a year, must be within its nextUpdate. The default limit
is those 30 days. Found in the audit of 5 November 2026.
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


@pytest.fixture(scope="module")
def example() -> Bundle:
    return Bundle.load(EXAMPLE)


def _fresh(report):
    return next((c for c in report.checks if c.name == "tdx.collateral_fresh"), None)


def test_default_limit_is_intels_30_days():
    assert pcs.DEFAULT_MAX_COLLATERAL_AGE == timedelta(days=30)


def test_collateral_of_the_day_passes_and_says_its_age(example):
    report = verify_bundle(example, online=False, expected_challenge=CHALLENGE, at=OLDEST + timedelta(hours=2))
    check = _fresh(report)
    assert report.ok and check.label == "PASS"
    assert check.detail == (
        "Intel collateral issued 2026-09-11 (QE identity), 0 day(s) old, within the 30-day limit; "
        "root CA CRL valid until 2027-02-26"
    )


def test_stale_collateral_fails_offline_and_names_the_option(example):
    report = verify_bundle(example, online=False, expected_challenge=CHALLENGE, at=NOV5)
    check = _fresh(report)
    assert not report.ok and report.failed() == ["tdx.collateral_fresh"]
    assert "54 days old" in check.detail and "QE identity issued 2026-09-11" in check.detail
    assert "over the 30-day limit" in check.detail and "--max-collateral-age 55" in check.detail


@pytest.mark.parametrize(("days", "ok"), [(54, False), (55, True)])
def test_a_wider_limit_is_an_explicit_choice(example, days, ok):
    wider = timedelta(days=days)
    report = verify_bundle(example, online=False, expected_challenge=CHALLENGE, at=NOV5, max_collateral_age=wider)
    assert _fresh(report).ok is ok and report.ok is ok


@pytest.mark.parametrize(("after", "ok"), [(timedelta(days=30), True), (timedelta(days=30, seconds=1), False)])
def test_the_limit_is_exact(example, after, ok):
    assert _fresh(verify_bundle(example, online=False, at=OLDEST + after)).ok is ok


def test_root_crl_past_its_next_update_needs_the_wider_limit_too(example):
    at = datetime(2027, 3, 1, tzinfo=timezone.utc)  # root CA CRL next update was 2027-02-26
    narrow = _fresh(verify_bundle(example, online=False, at=at, max_collateral_age=timedelta(days=200)))
    assert not narrow.ok and "root CA CRL past its nextUpdate (2027-02-26)" in narrow.detail
    assert "--max-collateral-age 368" in narrow.detail  # last update 2026-02-26 13:04
    assert _fresh(verify_bundle(example, online=False, at=at, max_collateral_age=timedelta(days=368))).ok


def test_collateral_without_crls_is_refused_offline(example):
    intel = {**example.collateral["intel"], "pck_crl_pem": None}
    report = verify_bundle(dataclasses.replace(example, collateral={"intel": intel}), online=False, at=OLDEST)
    assert not _fresh(report).ok and "no CRL" in _fresh(report).detail


def test_online_collateral_is_not_judged_by_the_bundle_age(example, monkeypatch):
    # Online the collateral is Intel's of the moment; here a stub hands back the September copy.
    monkeypatch.setattr(pcs, "fetch_collateral", lambda *a, **k: pcs.Collateral.from_json(example.collateral["intel"]))
    monkeypatch.setattr(verify_mod, "fetch_jwks", lambda: example.jwks)
    report = verify_bundle(example, online=True, expected_challenge=CHALLENGE, at=NOV5)
    assert report.ok, report.failed()
    assert _fresh(report).label == "WARN"  # past Intel's nextUpdate: the warning of 0.2.3, unchanged online


# --- command line ------------------------------------------------------------------------------


def _checks(capsys) -> dict[str, str]:
    return {c["name"]: c["result"] for c in json.loads(capsys.readouterr().out)["checks"]}


def test_cli_offline_on_an_archived_bundle(capsys):
    argv = ["verify", str(EXAMPLE), "--offline", "--json", "--challenge", CHALLENGE]
    assert main(argv) == 1
    assert _checks(capsys)["tdx.collateral_fresh"] == "FAIL"
    main([*argv, "--max-collateral-age", "3650"])
    assert _checks(capsys)["tdx.collateral_fresh"] == "PASS"


@pytest.mark.parametrize("value", ["-1", "abc", "1.5"])
def test_cli_refuses_a_bad_age(value, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["verify", str(EXAMPLE), "--offline", "--max-collateral-age", value])
    assert exc.value.code == 2 and "--max-collateral-age" in capsys.readouterr().err


def test_selftest_takes_the_same_limit(capsys):
    assert main(["selftest", str(EXAMPLE), "--offline", "--challenge", CHALLENGE, "--max-collateral-age", "3650"]) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == "SELFTEST: PASS"

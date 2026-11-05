# SPDX-License-Identifier: MIT
"""MRTD and RTMR0..3 against the values the verifier pins (--expect-mrtd, --expect-rtmr0..3).

Up to 0.2.3 VERIFIED said nothing about the software in the TD: MRTD and the RTMRs were
printed as facts and compared to nothing. The word VERIFIED is kept (changing it would break
every client), but without a pinned value a WARN now says so in the text and JSON output.
Found in the audit of 5 November 2026.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from voltage_verify.bundle import Bundle
from voltage_verify.cli import main
from voltage_verify.verify import TD_SOFTWARE_NOT_PINNED, verify_bundle, verify_quote

EXAMPLE = Path(__file__).parent.parent / "examples" / "hello-workload" / "bundle-8xh100-2026-09-12.json"
CHALLENGE = "ef7f54c160627ee536a02b5e73896ac4b1ac2cdefdb7cfaaf2366a7d33cc8731"
THIRD_PARTY = Path(__file__).parent / "fixtures" / "third-party" / "tdx_prod_quote_SPR_E4.dat"
AT = datetime(2026, 9, 12, tzinfo=timezone.utc)  # the day the reference bundle was captured
WRONG = bytes.fromhex("ff" * 48)  # matches no register here (RTMR3 is all zero in both quotes)


@pytest.fixture(scope="module")
def example() -> Bundle:
    return Bundle.load(EXAMPLE)


@pytest.fixture(scope="module")
def measured(example) -> dict[str, bytes]:
    """The reference quote's own MRTD and RTMR0..3, as a verifier would pin them."""
    report = verify_bundle(example, online=False, expected_challenge=CHALLENGE, at=AT)
    tdx = report.facts["tdx"]
    return {"mrtd": bytes.fromhex(tdx["mr_td"]), **{f"rtmr{i}": bytes.fromhex(r) for i, r in enumerate(tdx["rtmr"])}}


def _check(report, name):
    return next((c for c in report.checks if c.name == name), None)


def test_without_a_pinned_value_verified_stays_but_a_warning_says_so(example):
    report = verify_bundle(example, online=False, expected_challenge=CHALLENGE, at=AT)
    assert report.ok, report.failed()
    check = _check(report, "tdx.measurements")
    assert check.label == "WARN" and check.detail == TD_SOFTWARE_NOT_PINNED
    assert TD_SOFTWARE_NOT_PINNED == "TD software not pinned: pass --expect-mrtd to check it"


def test_pinned_mrtd_passes_and_names_what_is_left_unpinned(example, measured):
    pin = {"mrtd": measured["mrtd"]}
    report = verify_bundle(example, online=False, expected_challenge=CHALLENGE, at=AT, expected_measurements=pin)
    check = _check(report, "tdx.measurements")
    assert report.ok and check.label == "PASS"
    assert check.detail == "MRTD as expected; not pinned: RTMR0, RTMR1, RTMR2, RTMR3"


def test_every_register_pinned(example, measured):
    report = verify_bundle(example, online=False, expected_challenge=CHALLENGE, at=AT, expected_measurements=measured)
    check = _check(report, "tdx.measurements")
    assert report.ok and check.detail == "MRTD, RTMR0, RTMR1, RTMR2, RTMR3 as expected"


@pytest.mark.parametrize("register", ["mrtd", "rtmr0", "rtmr1", "rtmr2", "rtmr3"])
def test_one_register_that_differs_fails_the_bundle(example, measured, register):
    wrong = {**measured, register: WRONG}
    report = verify_bundle(example, online=False, expected_challenge=CHALLENGE, at=AT, expected_measurements=wrong)
    check = _check(report, "tdx.measurements")
    assert not report.ok and report.failed() == ["tdx.measurements"]
    assert check.label == "FAIL" and register.upper() in check.detail and "not the TD software you pinned" in check.detail


def test_unknown_register_name_is_a_programming_error(example):
    with pytest.raises(ValueError, match="mrtdx"):
        verify_bundle(example, online=False, at=AT, expected_measurements={"mrtdx": bytes(48)})


def test_bare_quote_gets_the_same_warning_and_option():
    raw = THIRD_PARTY.read_bytes()
    assert _check(verify_quote(raw, online=False), "tdx.measurements").detail == TD_SOFTWARE_NOT_PINNED
    mrtd = bytes.fromhex(verify_quote(raw, online=False).facts["tdx"]["mr_td"])
    assert _check(verify_quote(raw, online=False, expected_measurements={"mrtd": mrtd}), "tdx.measurements").ok
    assert not verify_quote(raw, online=False, expected_measurements={"mrtd": WRONG}).ok


# --- command line ------------------------------------------------------------------------------


def _json(argv, capsys) -> tuple[int, dict]:
    code = main([*argv, "--json"])
    return code, json.loads(capsys.readouterr().out)


def test_cli_warning_in_text_and_json(capsys):
    main(["quote", str(THIRD_PARTY), "--offline"])
    assert f"[WARN] tdx.measurements  {TD_SOFTWARE_NOT_PINNED}" in capsys.readouterr().out
    _, out = _json(["quote", str(THIRD_PARTY), "--offline"], capsys)
    assert {"name": "tdx.measurements", "result": "WARN", "detail": TD_SOFTWARE_NOT_PINNED} in out["checks"]


def test_cli_expect_options_reach_the_check(measured, capsys):
    argv = ["verify", str(EXAMPLE), "--offline", "--challenge", CHALLENGE]
    pins = [f"--expect-{name}" for name in measured]
    good = [a for name, value in measured.items() for a in (f"--expect-{name}", value.hex().upper())]
    _, out = _json([*argv, *good], capsys)
    assert {c["name"]: c["result"] for c in out["checks"]}["tdx.measurements"] == "PASS"
    code, out = _json([*argv, pins[2], "ff" * 48], capsys)
    assert code == 1 and {c["name"]: c["result"] for c in out["checks"]}["tdx.measurements"] == "FAIL"


@pytest.mark.parametrize("value", ["zz", "00" * 47, "00" * 49, "0x" + "00" * 48])
@pytest.mark.parametrize("command", ["verify", "quote"])
def test_cli_refuses_a_malformed_expected_value(command, value, capsys):
    target = str(EXAMPLE if command == "verify" else THIRD_PARTY)
    assert main([command, target, "--offline", "--expect-rtmr1", value]) == 2
    assert "--expect-rtmr1" in capsys.readouterr().err

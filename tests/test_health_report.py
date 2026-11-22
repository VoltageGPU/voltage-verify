# SPDX-License-Identifier: MIT
"""The health report: five statuses, a verdict that never turns a SKIP into a PASS, and a
canonical JSON form whose SHA-256 a manifest can commit to."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pytest

from voltage_verify.health.report import (
    FAIL,
    INFO,
    MAX_REPORT_BYTES,
    PASS,
    SCHEMA,
    SKIP,
    WARN,
    HealthCheck,
    HealthReport,
    HealthReportError,
    canonical_report,
    exit_code,
    load_report,
    report_sha256,
    summarize,
)


def _report(*checks: HealthCheck) -> HealthReport:
    return HealthReport(
        checks=list(checks),
        created_at=datetime(2026, 11, 22, 14, 0, 0, tzinfo=timezone.utc),
        duration_ms=61234,
        machine={"hostname": "node-1", "system": "Linux"},
        options={"quick": True, "burn_seconds": 60},
        samples={"fields": ["temp"], "rows": [[0, "idle", 0, "44"]]},
    )


def test_schema_name():
    assert SCHEMA == "voltage-verify/health/1"


def test_all_pass_is_pass_and_info_does_not_count():
    s = summarize([("inventory.gpus", PASS), ("topology.links", INFO)])
    assert (s.result, s.verdict) == ("PASS", "PASS")
    assert exit_code(s.result) == 0


def test_a_skip_is_never_a_global_pass_and_the_verdict_names_what_was_not_tested():
    s = summarize([("inventory.gpus", PASS), ("compute.matmul", SKIP), ("vram.pattern", SKIP)])
    assert s.result == "INCOMPLETE"
    assert s.verdict == "INCOMPLETE (not tested: compute.matmul, vram.pattern)"
    assert s.not_tested == ["compute.matmul", "vram.pattern"]
    assert exit_code(s.result) == 3


def test_a_warning_is_named_and_exits_3():
    s = summarize([("inventory.gpus", PASS), ("sensors.thermal", WARN)])
    assert (s.result, s.verdict) == ("WARN", "WARN (sensors.thermal)")
    assert exit_code(s.result) == 3


def test_incomplete_also_lists_the_warnings():
    s = summarize([("sensors.thermal", WARN), ("errors.xid", SKIP)])
    assert s.verdict == "INCOMPLETE (not tested: errors.xid; warnings: sensors.thermal)"


def test_a_failure_wins_and_still_says_what_was_not_tested():
    s = summarize([("errors.ecc", FAIL), ("sensors.thermal", WARN), ("vram.pattern", SKIP), ("compute.matmul", FAIL)])
    assert s.result == "FAIL"
    assert s.verdict == "FAIL (errors.ecc, compute.matmul; not tested: vram.pattern; warnings: sensors.thermal)"
    assert s.failed == ["errors.ecc", "compute.matmul"]
    assert exit_code(s.result) == 1


def test_unknown_status_is_refused():
    with pytest.raises(HealthReportError):
        summarize([("x", "OK")])


def test_report_json_is_versioned_and_carries_machine_time_duration_and_raw_data():
    doc = _report(HealthCheck("inventory.gpus", PASS, "1x GPU", {"gpus": [{"index": "0"}]})).to_json()
    assert doc["schema"] == SCHEMA
    assert doc["created_at"] == "2026-11-22T14:00:00+00:00"
    assert doc["duration_ms"] == 61234
    assert doc["machine"]["hostname"] == "node-1"
    assert doc["tool"]["name"] == "voltage-verify"
    assert doc["ok"] is True and doc["result"] == "PASS" and doc["verdict"] == "PASS"
    assert doc["checks"] == [
        {"name": "inventory.gpus", "result": "PASS", "detail": "1x GPU", "data": {"gpus": [{"index": "0"}]}}
    ]
    assert doc["samples"]["rows"] == [[0, "idle", 0, "44"]]


def test_ok_is_true_only_for_a_plain_pass():
    assert _report(HealthCheck("a", PASS, "")).to_json()["ok"] is True
    for status in (WARN, SKIP, FAIL):
        assert _report(HealthCheck("a", status, "")).to_json()["ok"] is False


def test_canonical_form_is_key_order_independent_and_its_sha256_is_the_file_sha256():
    doc = _report(HealthCheck("a", PASS, "é")).to_json()
    shuffled = json.loads(json.dumps(doc, sort_keys=True, indent=3))
    raw = canonical_report(doc)
    assert raw == canonical_report(shuffled)
    assert b"\n" not in raw and b": " not in raw
    assert report_sha256(doc) == hashlib.sha256(raw).hexdigest()
    assert load_report(raw) == doc


def test_a_float_cannot_enter_a_report():
    with pytest.raises(HealthReportError, match="float"):
        canonical_report({"schema": SCHEMA, "x": 1.5})


def _valid() -> dict:
    return _report(HealthCheck("a", PASS, "fine"), HealthCheck("b", SKIP, "not run")).to_json()


def test_load_report_accepts_what_the_tool_writes_pretty_printed_too():
    doc = _valid()
    assert load_report(json.dumps(doc, indent=2).encode()) == doc


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"not json",
        b"[]",
        b'{"schema": "voltage-verify/health/2"}',
        b"\xff\xfe",
        b'{"schema":"voltage-verify/health/1","schema":"voltage-verify/health/1"}',
        b'{"schema":"voltage-verify/health/1","x":NaN}',
        b'{"schema":"voltage-verify/health/1","x":1.5}',
        b"[" * 100000,
    ],
    ids=["empty", "not-json", "list", "other-schema", "not-utf8", "duplicated-key", "nan", "float", "nested-too-deep"],
)
def test_load_report_refuses_unusable_input(raw):
    with pytest.raises(HealthReportError):
        load_report(raw)


def test_load_report_refuses_an_oversized_file():
    with pytest.raises(HealthReportError, match="larger"):
        load_report(b" " * (MAX_REPORT_BYTES + 1))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.pop("checks"),
        lambda d: d.update(checks="x"),
        lambda d: d["checks"].append("x"),
        lambda d: d["checks"][0].update(result="OK"),
        lambda d: d["checks"][0].update(name="has space"),
        lambda d: d["checks"][0].update(detail=3),
        lambda d: d.update(created_at="yesterday"),
        lambda d: d.update(created_at="2026-11-22T14:00:00"),
        lambda d: d.update(verdict=3),
    ],
)
def test_load_report_refuses_a_malformed_report(mutate):
    doc = _valid()
    mutate(doc)
    with pytest.raises(HealthReportError):
        load_report(json.dumps(doc).encode())

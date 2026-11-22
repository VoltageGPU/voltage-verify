# SPDX-License-Identifier: MIT
"""What a verifier must not be told is a plain pass, and what the producer must not do as root.

A report can be committed and unaltered and still say little: a report with no load test in
it, a report measured days before the attestation, a committed report nobody showed. Each of
those is a limit named in the verdict (exit code 3), never a plain VERIFIED.
"""

from __future__ import annotations

import os
import stat
import sys
from datetime import datetime, timedelta, timezone

import pytest
from health_fakes import FakeHost, FakeWorkload

from voltage_verify.cli import main
from voltage_verify.health import run as run_module
from voltage_verify.health.report import (
    PASS,
    REQUIRED_CHECKS,
    SCHEMA,
    SKIP,
    HealthCheck,
    HealthReport,
    HealthReportError,
    bind,
    canonical_report,
    load_report,
    report_sha256,
)
from voltage_verify.health.run import Options, run_health
from voltage_verify.manifest import build_manifest

MEASURED = datetime(2026, 9, 11, 22, 50, tzinfo=timezone.utc)


def _report(*checks: HealthCheck, created_at: datetime = MEASURED) -> dict:
    return HealthReport(
        checks=list(checks),
        created_at=created_at,
        duration_ms=312000,
        machine={"hostname": "node-1"},
        options={},
        samples={},
    ).to_json()


def _complete(created_at: datetime = MEASURED) -> dict:
    return _report(*(HealthCheck(name, PASS, "fine") for name in REQUIRED_CHECKS), created_at=created_at)


def _committed(doc: dict, after: timedelta = timedelta(minutes=20)) -> dict:
    manifest = build_manifest()
    manifest["created_at"] = (datetime.fromisoformat(doc["created_at"]) + after).isoformat(timespec="seconds")
    manifest["health"] = {"schema": SCHEMA, "sha256": report_sha256(doc)}
    return manifest


# --- a report that leaves checks out -----------------------------------------------------------


def test_the_checks_a_report_must_carry():
    assert REQUIRED_CHECKS == (
        "inventory.gpus",
        "errors.ecc",
        "errors.xid",
        "sensors.thermal",
        "sensors.clocks",
        "compute.matmul",
        "vram.pattern",
    )


def test_a_complete_report_is_a_plain_pass():
    doc = _complete()
    result = bind(_committed(doc), canonical_report(doc))
    assert result is not None and result.ok and result.limit is None


def test_a_report_with_no_check_is_not_a_pass():
    doc = _report()
    assert doc["result"] == "PASS", "the report itself has nothing to contradict"
    result = bind(_committed(doc), canonical_report(doc))
    assert result is not None and not result.ok and not result.required
    assert result.limit == "health report PASS, but without the checks " + ", ".join(REQUIRED_CHECKS)
    assert "without the checks" in result.detail


def test_a_report_without_the_load_tests_names_them():
    kept = [name for name in REQUIRED_CHECKS if name not in ("compute.matmul", "vram.pattern")]
    doc = _report(*(HealthCheck(name, PASS, "fine") for name in kept))
    result = bind(_committed(doc), canonical_report(doc))
    assert result is not None and not result.ok and not result.required
    assert result.limit == "health report PASS, but without the checks compute.matmul, vram.pattern"
    assert result.facts is not None and result.facts["missing"] == "compute.matmul, vram.pattern"


def test_missing_checks_are_named_next_to_the_untested_ones():
    doc = _report(HealthCheck("inventory.gpus", PASS, "1x"), HealthCheck("errors.xid", SKIP, "not root"))
    result = bind(_committed(doc), canonical_report(doc))
    assert result is not None and not result.ok
    assert result.limit is not None
    assert result.limit.startswith("health report INCOMPLETE (not tested: errors.xid), and without the checks errors.ecc, ")


def test_a_check_written_twice_is_refused():
    doc = _complete()
    doc["checks"].append({"name": "errors.ecc", "result": "PASS", "detail": "again", "data": {}})
    with pytest.raises(HealthReportError, match="twice"):
        load_report(canonical_report(doc))
    result = bind(_committed(doc), canonical_report(doc))
    assert result is not None and not result.ok and result.required


# --- a committed report nobody showed ----------------------------------------------------------


def test_a_committed_report_that_is_not_supplied_is_a_limit():
    result = bind(_committed(_complete()), None)
    assert result is not None and not result.ok and not result.required
    assert result.limit == "health report committed but not checked (no --health)"


# --- when the report was measured --------------------------------------------------------------


@pytest.mark.parametrize("after", [timedelta(seconds=0), timedelta(minutes=20), timedelta(hours=24)])
def test_a_report_measured_within_a_day_before_the_manifest_is_fine(after):
    doc = _complete()
    result = bind(_committed(doc, after), canonical_report(doc))
    assert result is not None and result.ok and result.limit is None


def test_a_report_measured_days_before_the_manifest_is_a_limit():
    doc = _complete()
    result = bind(_committed(doc, timedelta(days=3, hours=2)), canonical_report(doc))
    assert result is not None and not result.ok and not result.required
    assert result.limit == "health report measured 3 days before the manifest"


def test_a_report_dated_after_the_manifest_is_a_limit():
    doc = _complete()
    result = bind(_committed(doc, timedelta(hours=-2)), canonical_report(doc))
    assert result is not None and not result.ok and not result.required
    assert result.limit == "health report dated after the manifest that commits to it"


def test_a_manifest_without_a_readable_date_gets_no_date_check():
    doc = _complete()
    manifest = _committed(doc)
    manifest["created_at"] = "yesterday"
    result = bind(manifest, canonical_report(doc))
    assert result is not None and result.ok and result.limit is None


# --- text that has no canonical bytes ----------------------------------------------------------


def test_a_lone_surrogate_is_a_report_error_not_a_crash():
    doc = _complete()
    doc["machine"]["hostname"] = "node\ud800"
    with pytest.raises(HealthReportError):
        canonical_report(doc)
    raw = canonical_report(_complete()).replace(b'"node-1"', b'"node\\ud800"')
    with pytest.raises(HealthReportError):
        load_report(raw)


def test_the_host_name_is_written_as_printable_text(monkeypatch):
    monkeypatch.setattr(run_module.platform, "node", lambda: "node\ud800\x1b[31m-1" + "x" * 400)
    host = FakeHost("rtx3060ti-1x")
    doc = run_health(Options(quick=True), host=host, workload=FakeWorkload(host, torch=None)).to_json()
    assert doc["machine"]["hostname"].startswith("node[31m-1x")
    assert len(doc["machine"]["hostname"]) == 255
    canonical_report(doc)


# --- the workers are stopped whatever happens --------------------------------------------------


class _FailsToStart(FakeWorkload):
    def start(self, gpus, seconds):
        super().start(gpus, seconds)
        raise OSError("fork: resource temporarily unavailable")


def test_workers_are_collected_when_starting_them_fails():
    host = FakeHost("h100-8x")
    workload = _FailsToStart(host)
    with pytest.raises(OSError):
        run_health(Options(quick=True), host=host, workload=workload)
    assert workload.collected, "collect() stops the workers that did start"


# --- writing the report ------------------------------------------------------------------------


@pytest.fixture
def machine(monkeypatch):
    made = {}

    def install(fixture, **workload):
        host = FakeHost(fixture)
        made["host"], made["workload"] = host, FakeWorkload(host, **workload)
        monkeypatch.setattr(run_module, "SystemHost", lambda: made["host"])
        monkeypatch.setattr(run_module, "TorchWorkload", lambda: made["workload"])
        return made

    return install


def test_an_unwritable_report_path_is_refused_before_anything_runs(machine, tmp_path, capsys):
    made = machine("h100-8x")
    assert main(["health", "--quick", "-o", str(tmp_path / "no-such-dir" / "health.json")]) == 2
    assert "cannot write" in capsys.readouterr().err
    assert made["host"].calls == [], "a 5 to 60 minute test must not run for a report that cannot be written"


def test_the_report_replaces_an_older_one_and_leaves_no_temporary_file(machine, tmp_path):
    machine("h100-8x")
    out = tmp_path / "health.json"
    out.write_text("an older report")
    assert main(["health", "--quick", "-o", str(out)]) == 0
    assert load_report(out.read_bytes())["result"] == "PASS"
    assert [p.name for p in tmp_path.iterdir()] == ["health.json"]


def test_an_interrupted_run_leaves_no_temporary_file(machine, tmp_path, monkeypatch):
    machine("h100-8x")

    def interrupted(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(run_module, "run_health", interrupted)
    with pytest.raises(KeyboardInterrupt):
        main(["health", "--quick", "-o", str(tmp_path / "health.json")])
    assert list(tmp_path.iterdir()) == []


def test_the_result_is_still_printed_when_the_report_cannot_be_put_in_place(machine, tmp_path, monkeypatch, capsys):
    machine("h100-8x")

    def refuse(*_args, **_kwargs):
        raise PermissionError("read-only file system")

    monkeypatch.setattr(os, "replace", refuse)
    assert main(["health", "--quick", "-o", str(tmp_path / "health.json")]) == 2
    captured = capsys.readouterr()
    assert captured.out.strip().splitlines()[-1] == "RESULT: PASS", "five minutes of test are not thrown away"
    assert "cannot write" in captured.err
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_the_report_is_readable_by_its_owner_only(machine, tmp_path):
    machine("h100-8x")
    out = tmp_path / "health.json"
    assert main(["health", "--quick", "-o", str(out)]) == 0
    assert stat.S_IMODE(out.stat().st_mode) == 0o600, "it names the machine: host name, GPU UUIDs, serial numbers"


def test_a_symbolic_link_at_the_report_path_is_replaced_not_followed(machine, tmp_path):
    machine("h100-8x")
    target, out = tmp_path / "system-file", tmp_path / "health.json"
    target.write_text("must stay as it is")
    try:
        out.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available here")
    assert main(["health", "--quick", "-o", str(out)]) == 0
    assert target.read_text() == "must stay as it is"
    assert not out.is_symlink() and load_report(out.read_bytes())["result"] == "PASS"

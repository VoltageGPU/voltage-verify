# SPDX-License-Identifier: MIT
"""`voltage-verify health` from end to end, on fake machines: a healthy 8x H100 node, a sick
one, a desktop without a CUDA build of PyTorch, a machine without a GPU."""

from __future__ import annotations

import hashlib
import json
import re
import socket
from pathlib import Path

import pytest
from health_fakes import FakeHost, FakeWorkload, burn_record, healthy, vram_record

from voltage_verify.cli import main
from voltage_verify.health import run as run_module
from voltage_verify.health.report import SCHEMA, canonical_report, load_report, report_sha256
from voltage_verify.health.run import Options, run_health

HEALTH_SOURCES = sorted((Path(__file__).parent.parent / "src" / "voltage_verify" / "health").glob("*.py"))


def _statuses(report) -> dict[str, str]:
    return {c.name: c.status for c in report.checks}


def _sick_records() -> dict:
    records = {gpu: healthy() for gpu in range(8)}
    records[4] = healthy(burn=burn_record(gflops=301200))
    records[6] = healthy(burn=burn_record(mismatch=2), vram=vram_record(wrong_bytes=64))
    return records


@pytest.fixture
def machine(monkeypatch):
    """Put a fake machine behind the command line: machine("h100-8x", torch=None, ...)."""
    made = {}

    def install(fixture, *, kernel_log="dmesg", **workload):
        host = FakeHost(fixture, kernel_log=kernel_log)
        made["host"], made["workload"] = host, FakeWorkload(host, **workload)
        monkeypatch.setattr(run_module, "SystemHost", lambda: made["host"])
        monkeypatch.setattr(run_module, "TorchWorkload", lambda: made["workload"])
        return made

    return install


# --- the run -----------------------------------------------------------------------------------


def test_healthy_node_passes_every_check():
    host = FakeHost("h100-8x")
    workload = FakeWorkload(host)
    report = run_health(Options(quick=True, expect_gpu="H100", expect_gpus=8, expect_vram_gb=80), host, workload)
    assert _statuses(report) == {
        "inventory.gpus": "PASS",
        "inventory.model": "PASS",
        "inventory.count": "PASS",
        "inventory.vram": "PASS",
        "errors.ecc": "PASS",
        "errors.xid": "PASS",
        "sensors.thermal": "PASS",
        "sensors.clocks": "PASS",
        "compute.matmul": "PASS",
        "vram.pattern": "PASS",
        "topology.links": "INFO",
    }
    assert report.summary.verdict == "PASS"
    assert workload.started == ([0, 1, 2, 3, 4, 5, 6, 7], 60)
    assert host.slept == [2.0] * 5  # sensors read every 2 s while the load runs


def test_an_interrupted_run_leaves_no_load_running():
    host = FakeHost("h100-8x")
    workload = FakeWorkload(host)

    def interrupted(_seconds):
        raise KeyboardInterrupt

    host.sleep = interrupted
    with pytest.raises(KeyboardInterrupt):
        run_health(Options(quick=True), host, workload)
    assert workload.collected, "the workers must be stopped and collected"
    assert host.busy is False


def test_default_load_is_300_seconds_and_burn_seconds_overrides_quick():
    assert Options().seconds == 300
    assert Options(quick=True).seconds == 60
    assert Options(quick=True, burn_seconds=45).seconds == 45
    host = FakeHost("h100-8x")
    workload = FakeWorkload(host)
    run_health(Options(), host, workload)
    assert workload.started is not None and workload.started[1] == 300
    assert host.slept == [5.0] * 5  # and every 5 s on a long run


def test_sick_node_fails_and_names_every_failed_check():
    host = FakeHost("h100-8x-sick")
    report = run_health(Options(quick=True), host, FakeWorkload(host, records=_sick_records()))
    statuses = _statuses(report)
    assert statuses["errors.ecc"] == "FAIL"  # uncorrected ECC errors on GPU 6, pending remap on GPU 1
    assert statuses["errors.xid"] == "FAIL"  # Xid 79 during the test
    assert statuses["sensors.thermal"] == "FAIL"  # GPU 5 in hardware thermal slowdown
    assert statuses["sensors.clocks"] == "WARN"  # GPU 0 at a low clock
    assert statuses["compute.matmul"] == "FAIL"  # GPU 6 changes its results, GPU 4 at half speed
    assert statuses["vram.pattern"] == "FAIL"  # 64 wrong bytes on GPU 6
    assert report.summary.verdict == (
        "FAIL (errors.ecc, errors.xid, sensors.thermal, compute.matmul, vram.pattern; warnings: sensors.clocks)"
    )


def test_without_pytorch_the_result_is_incomplete_never_pass():
    host = FakeHost("rtx3060ti-1x")
    workload = FakeWorkload(host, torch=None)
    report = run_health(Options(quick=True), host, workload)
    statuses = _statuses(report)
    assert statuses["inventory.gpus"] == "PASS"
    assert statuses["errors.ecc"] == "INFO" and statuses["errors.xid"] == "SKIP"
    assert statuses["sensors.thermal"] == "INFO" and statuses["sensors.clocks"] == "INFO"
    assert statuses["compute.matmul"] == "SKIP" and statuses["vram.pattern"] == "SKIP"
    assert report.summary.result == "INCOMPLETE"
    assert report.summary.verdict == "INCOMPLETE (not tested: errors.xid, compute.matmul, vram.pattern)"
    assert workload.started is None, "no load may start without a CUDA build of PyTorch"
    assert report.to_json()["ok"] is False


def test_a_cpu_only_pytorch_is_explained():
    host = FakeHost("rtx3060ti-1x")
    report = run_health(Options(quick=True), host, FakeWorkload(host, torch="2.3.1+cpu", cuda=False))
    detail = next(c.detail for c in report.checks if c.name == "compute.matmul")
    assert detail.startswith("PyTorch 2.3.1+cpu sees no CUDA device: the compute test did not run")
    assert "nothing is installed for you" in detail


def test_no_gpu_fails_and_everything_else_is_not_tested():
    host = FakeHost(None)
    workload = FakeWorkload(host)
    report = run_health(Options(quick=True), host, workload)
    assert _statuses(report)["inventory.gpus"] == "FAIL"
    assert report.summary.verdict == (
        "FAIL (inventory.gpus; not tested: errors.ecc, errors.xid, sensors.thermal, sensors.clocks, "
        "compute.matmul, vram.pattern)"
    )
    assert workload.started is None


def test_the_listing_is_checked_before_anything_else():
    host = FakeHost("rtx3060ti-1x")
    report = run_health(Options(quick=True, expect_gpu="RTX 4090", expect_vram_gb=24), host, FakeWorkload(host, torch=None))
    assert report.summary.failed == ["inventory.model", "inventory.vram"]


def test_report_carries_machine_utc_time_duration_options_and_raw_samples():
    host = FakeHost("h100-8x")
    doc = run_health(Options(quick=True, expect_gpus=8), host, FakeWorkload(host)).to_json()
    assert doc["schema"] == SCHEMA
    assert doc["created_at"].endswith("+00:00")
    assert doc["duration_ms"] > 0
    assert sorted(doc["machine"]) == ["arch", "hostname", "python", "release", "system"]
    assert doc["options"] == {
        "quick": True,
        "burn_seconds": 60,
        "expect_gpu": None,
        "expect_gpus": 8,
        "expect_vram_gb": None,
    }
    fields, rows = doc["samples"]["fields"], doc["samples"]["rows"]
    assert fields[:3] == ["t_ms", "phase", "index"] and "temperature" in fields and "power_draw" in fields
    assert len(rows) == 8 * 6  # one idle reading and five under load, for eight GPUs
    assert rows[0][1] == "idle" and rows[-1][1] == "load"
    assert rows[-1][fields.index("power_draw")] == "695.30"  # as nvidia-smi printed it
    assert load_report(canonical_report(doc)) == doc  # canonical, and no float anywhere


def test_only_expected_tools_are_called_each_as_an_argument_list():
    host = FakeHost("h100-8x-sick")
    run_health(Options(quick=True), host, FakeWorkload(host, records=_sick_records()))
    assert {call[0] for call in host.calls} == {"nvidia-smi", "dmesg"}
    assert all("sudo" not in call for call in host.calls)


def test_health_never_touches_the_network(monkeypatch):
    def no_network(*_args, **_kwargs):
        raise AssertionError("health opened a socket")

    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    host = FakeHost("h100-8x")
    assert run_health(Options(quick=True), host, FakeWorkload(host)).summary.result == "PASS"


@pytest.mark.parametrize("source", HEALTH_SOURCES, ids=lambda p: p.name)
def test_health_modules_import_no_network_code_and_never_use_a_shell(source):
    text = source.read_text(encoding="utf-8")
    imported = set(re.findall(r"^\s*(?:from|import)\s+([a-zA-Z_][\w.]*)", text, flags=re.MULTILINE))
    assert not imported & {
        "urllib",
        "urllib.request",
        "http",
        "http.client",
        "socket",
        "ssl",
        "requests",
        "ftplib",
        "smtplib",
    }
    assert "shell=True" not in text and "os.system" not in text and "pip" not in imported


# --- the command line --------------------------------------------------------------------------


def test_cli_prints_one_line_per_check_and_ends_on_the_result(machine, capsys):
    machine("h100-8x")
    assert main(["health", "--quick", "--expect-gpu", "H100", "--expect-gpus", "8", "--expect-vram", "80"]) == 0
    captured = capsys.readouterr()
    lines = captured.out.strip().splitlines()
    assert lines[0].startswith("[PASS] inventory.gpus   8x NVIDIA H100 80GB HBM3, 81559 MiB each")
    assert any(line.startswith("[INFO] topology.links ") for line in lines)
    assert "GPU 0: NVIDIA H100 80GB HBM3  81559 MiB  max 62 C  max 695 W  SM 1755 MHz  652 TFLOPS  0 changed" in lines[12]
    assert lines[-1] == "RESULT: PASS"
    assert "load test: 60 s of variable matmul load on 8 GPU(s)" in captured.err


def test_cli_exit_codes_follow_the_verdict(machine, capsys):
    machine("h100-8x-sick", records=_sick_records())
    assert main(["health", "--quick"]) == 1
    assert capsys.readouterr().out.strip().splitlines()[-1].startswith("RESULT: FAIL (errors.ecc, errors.xid")
    machine("rtx3060ti-1x", torch=None)
    assert main(["health", "--quick"]) == 3
    out = capsys.readouterr().out
    assert out.strip().splitlines()[-1] == "RESULT: INCOMPLETE (not tested: errors.xid, compute.matmul, vram.pattern)"
    assert "[SKIP] compute.matmul " in out and "[SKIP] errors.xid " in out


def test_cli_json_is_the_report(machine, capsys):
    machine("rtx3060ti-1x", torch=None)
    assert main(["health", "--quick", "--json"]) == 3
    doc = json.loads(capsys.readouterr().out)
    assert doc["schema"] == SCHEMA and doc["result"] == "INCOMPLETE" and doc["ok"] is False
    assert doc["not_tested"] == ["errors.xid", "compute.matmul", "vram.pattern"]
    assert sorted(doc) == [
        "checks",
        "created_at",
        "duration_ms",
        "failed",
        "machine",
        "not_tested",
        "ok",
        "options",
        "result",
        "samples",
        "schema",
        "tool",
        "verdict",
        "warnings",
    ]


def test_cli_writes_the_canonical_report_whose_sha256_a_manifest_commits_to(machine, tmp_path, capsys):
    machine("h100-8x")
    out = tmp_path / "health.json"
    assert main(["health", "--quick", "-o", str(out)]) == 0
    raw = out.read_bytes()
    doc = load_report(raw)
    assert raw == canonical_report(doc)
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == report_sha256(doc)
    printed = capsys.readouterr().out
    assert f"REPORT: {out}  sha256 {digest}" in printed
    manifest = tmp_path / "manifest.json"
    assert main(["manifest", "--health", str(out), "-o", str(manifest)]) == 0
    assert json.loads(manifest.read_text())["health"]["sha256"] == digest


@pytest.mark.parametrize(
    "argv",
    [
        ["--burn-seconds", "5"],
        ["--burn-seconds", "3601"],
        ["--burn-seconds", "1e2"],
        ["--expect-gpus", "0"],
        ["--expect-gpus", "-1"],
        ["--expect-vram", "8.5"],
        ["--expect-vram", "99999"],
        ["--expect-gpu", ""],
        ["--expect-gpu", "x" * 81],
        ["--expect-gpu", "RTX\x1b[31m4090"],
    ],
)
def test_cli_refuses_arguments_out_of_range(argv, machine):
    made = machine("h100-8x")
    with pytest.raises(SystemExit) as exit_info:
        main(["health", *argv])
    assert exit_info.value.code == 2
    assert made["host"].calls == [], "nothing may run on bad arguments"


def test_cli_cannot_write_the_report(machine, tmp_path, capsys):
    machine("rtx3060ti-1x", torch=None)
    assert main(["health", "--quick", "-o", str(tmp_path / "no-such-dir" / "health.json")]) == 2
    assert "cannot write" in capsys.readouterr().err

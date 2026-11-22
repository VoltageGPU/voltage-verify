# SPDX-License-Identifier: MIT
"""The PyTorch worker and the process that starts it.

The stages are run here on the CPU build of PyTorch when one is installed (the code path is
the same, the device differs), with a fault injected the way faulty hardware would: a bit of
a result flipped, a byte of memory changed after it was written. Without PyTorch those tests
are skipped; the ones about the worker's protocol and about its absence always run.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from voltage_verify.health import worker
from voltage_verify.health.workload import TorchWorkload, parse_records

# --- stages, on the CPU ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def test_exact_product_is_the_same_on_the_device_and_on_the_cpu(torch):
    assert worker.cross_check(torch, "cpu") == {"stage": "cross_check", "wrong": 0}


def test_exact_product_detects_a_single_wrong_value(torch):
    def one_off(result):
        result[17, 3] += 1

    assert worker.cross_check(torch, "cpu", fault=one_off)["wrong"] == 1


def test_burn_recomputes_and_compares_and_finds_nothing_on_a_sound_device(torch):
    record = worker.burn(torch, "cpu", 0.6, n=128, schedule=((0.2, 0.05), (0.1, 0.05)))
    assert record["stage"] == "burn" and record["mismatch"] == 0 and record["finite"] is True
    assert record["iterations"] >= record["checks"] >= 2
    assert record["gflops"] > 0 and record["dtype"] == "float32" and record["n"] == 128
    assert all(type(v) in (int, str, bool) for v in record.values()), "no float may enter the report"


def test_burn_load_is_variable_bursts_and_pauses(torch):
    record = worker.burn(torch, "cpu", 0.8, n=64, schedule=((0.1, 0.1), (0.05, 0.05)))
    assert record["segments"] >= 3
    assert record["load_ms"] < record["elapsed_ms"]  # the pauses are not counted as load


def test_burn_detects_a_computation_error(torch):
    def flip_one_value(result, check):
        if check in (1, 3):
            result[5, 7] += 1.0

    record = worker.burn(torch, "cpu", 0.6, n=128, schedule=((0.3, 0.02),), fault=flip_one_value)
    assert record["mismatch"] == 2
    assert record["checks"] > 4


def test_vram_patterns_read_back_clean(torch):
    record = worker.vram(torch, "cpu", free_bytes=128 << 20)
    assert record["wrong_bytes"] == 0 and record["patterns"] == 5
    assert record["chunk_bytes"] == 2 << 20
    assert record["tested_bytes"] * 100 >= 90 * record["free_bytes"]  # the 90% the check asks for
    assert record["tested_bytes"] == record["chunks"] * record["chunk_bytes"]


def test_vram_detects_a_wrong_pattern(torch):
    def stuck_bytes(chunks, pattern_number):
        if pattern_number == 0:  # the random pattern: three bytes change after the write
            chunks[2][100] ^= 0xFF
            chunks[2][101] ^= 0x01
            chunks[5][0] ^= 0x80
        if pattern_number == 3:  # the all-zero pattern: one bit stuck at 1
            chunks[0][12345] = 0x10

    record = worker.vram(torch, "cpu", free_bytes=64 << 20, fault=stuck_bytes)
    assert record["wrong_bytes"] == 4


def test_vram_random_pattern_differs_from_chunk_to_chunk(torch):
    # An address fault (two chunks landing on the same memory) only shows if chunks differ.
    seen = []

    def keep(chunks, pattern_number):
        if pattern_number == 0:
            seen.extend(bytes(c[:16].tolist()) for c in chunks[:4])

    worker.vram(torch, "cpu", free_bytes=64 << 20, fault=keep)
    assert len(set(seen)) == 4


# --- the worker's command line -----------------------------------------------------------------


def test_probe_without_pytorch_says_so_and_installs_nothing(monkeypatch, capsys):
    def no_torch(name):
        raise ModuleNotFoundError(f"No module named {name!r}")

    monkeypatch.setattr(importlib, "import_module", no_torch)
    assert worker.main(["probe"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "stage": "probe",
        "torch": None,
        "cuda": False,
        "devices": 0,
        "error": "ModuleNotFoundError",
    }


def test_run_reports_each_stage_and_goes_on_after_one_fails(monkeypatch, capsys):
    monkeypatch.setattr(importlib, "import_module", lambda name: object())
    monkeypatch.setattr(worker, "cross_check", lambda torch, device: {"stage": "cross_check", "wrong": 0})

    def falls_off_the_bus(torch, device, seconds):
        raise RuntimeError("CUDA error: unspecified launch failure " + "x" * 500)

    monkeypatch.setattr(worker, "burn", falls_off_the_bus)
    monkeypatch.setattr(worker, "vram", lambda torch, device: {"stage": "vram", "wrong_bytes": 0})
    assert worker.main(["run", "--seconds", "10"]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["stage"] for line in lines] == ["cross_check", "burn", "vram"]
    assert lines[1]["error"].startswith("RuntimeError: CUDA error: unspecified launch failure")
    assert len(lines[1]["error"]) < 250


@pytest.mark.parametrize("seconds", ["0", "3601", "abc", "-5", "1e3", "٣"])
def test_run_refuses_seconds_out_of_range(seconds):
    with pytest.raises(SystemExit) as exit_info:
        worker.main(["run", "--seconds", seconds])
    assert exit_info.value.code == 2


# --- reading what a worker prints --------------------------------------------------------------


def test_parse_records_keeps_only_known_stages_fields_and_types():
    text = "\n".join(
        [
            "a warning PyTorch printed on stdout",
            json.dumps(
                {"stage": "burn", "mismatch": 0, "gflops": 652400, "finite": True, "dtype": "bfloat16", "extra": "dropped"}
            ),
            json.dumps({"stage": "vram", "wrong_bytes": "0", "tested_bytes": 1.5, "chunks": True, "patterns": 5}),
            json.dumps({"stage": "format_disk", "now": True}),
            json.dumps(["burn"]),
            json.dumps({"stage": "cross_check", "error": "Runtime\x1b[31mError: " + "y" * 500}),
        ]
    )
    records = parse_records(text)
    assert sorted(records) == ["burn", "cross_check", "vram"]
    assert records["burn"] == {"stage": "burn", "mismatch": 0, "gflops": 652400, "finite": True, "dtype": "bfloat16"}
    assert records["vram"] == {"stage": "vram", "patterns": 5}  # a string, a float and a bool are not counts
    assert "\x1b" not in records["cross_check"]["error"] and len(records["cross_check"]["error"]) == 200


# --- the processes -----------------------------------------------------------------------------

STUB = """
import json, os, sys, time
if sys.argv[1] == "probe":
    print(json.dumps({"stage": "probe", "torch": "9.9+cu999", "cuda": True, "devices": 2}))
elif os.environ.get("STUB_MODE") == "hang":
    time.sleep(60)
elif os.environ.get("STUB_MODE") == "crash":
    print(json.dumps({"stage": "cross_check", "wrong": 0}), flush=True)
    os._exit(139)
else:
    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    print(json.dumps({"stage": "cross_check", "wrong": 0}))
    print(json.dumps({"stage": "burn", "mismatch": int(gpu), "checks": int(sys.argv[3])}))
    print("x" * 200000)
    print(json.dumps({"stage": "vram", "wrong_bytes": 0}))
"""


@pytest.fixture
def stub(tmp_path):
    path = tmp_path / "stub_worker.py"
    path.write_text(STUB, encoding="utf-8")
    return [sys.executable, str(path)]


def _finish(workload: TorchWorkload) -> dict:
    import time

    deadline = time.monotonic() + 60
    while workload.poll() and time.monotonic() < deadline:
        time.sleep(0.05)
    return workload.collect()


def test_workload_probe_reads_the_worker_answer(stub):
    probe = TorchWorkload(stub).probe()
    assert (probe.torch, probe.cuda, probe.devices, probe.usable) == ("9.9+cu999", True, 2, True)


def test_workers_do_not_import_from_the_directory_the_user_stands_in(tmp_path, monkeypatch):
    # `python -m` puts the current directory first on sys.path: a torch.py planted there would
    # be imported in place of PyTorch. The workers start in a private, empty directory.
    (tmp_path / "torch.py").write_text("raise SystemExit('planted torch.py was imported')")
    (tmp_path / "voltage_verify").mkdir()
    (tmp_path / "voltage_verify" / "__init__.py").write_text("raise SystemExit('planted package was imported')")
    monkeypatch.chdir(tmp_path)
    probe = TorchWorkload().probe()
    assert "could not be probed" not in probe.reason, probe.reason
    where_am_i = "import os, json; print(json.dumps({'stage': 'burn', 'dtype': os.getcwd()}))"
    workload = TorchWorkload([sys.executable, "-c", where_am_i])
    workload.start([0], 10)
    where = _finish(workload)[0]["burn"]["dtype"]
    assert Path(where).resolve() != tmp_path.resolve()
    assert not Path(where).exists(), "the private directory is removed afterwards"


def test_workload_runs_one_worker_per_gpu_by_its_index(stub):
    workload = TorchWorkload(stub)
    workload.start([0, 3], 42)
    results = _finish(workload)
    assert sorted(results) == [0, 3]
    assert results[3]["burn"] == {"stage": "burn", "mismatch": 3, "checks": 42}
    assert results[0]["vram"] == {"stage": "vram", "wrong_bytes": 0}  # 200 kB of other output did not block it


def test_a_worker_that_dies_keeps_what_it_reported_and_names_the_rest(stub, monkeypatch):
    monkeypatch.setenv("STUB_MODE", "crash")
    workload = TorchWorkload(stub)
    workload.start([0], 10)
    results = _finish(workload)
    assert results[0]["cross_check"] == {"stage": "cross_check", "wrong": 0}
    assert results[0]["burn"] == {"stage": "burn", "error": "its worker ended with exit code 139"}
    assert results[0]["vram"]["error"] == "its worker ended with exit code 139"


def test_a_worker_that_hangs_is_stopped_at_the_deadline(stub, monkeypatch):
    monkeypatch.setenv("STUB_MODE", "hang")
    workload = TorchWorkload(stub, finish_margin=-9.5)  # deadline half a second after the start
    workload.start([0], 10)
    results = _finish(workload)
    assert results[0]["burn"] == {"stage": "burn", "error": "stopped at the deadline"}


def test_real_probe_on_this_machine_never_claims_a_gpu_it_cannot_use():
    probe = TorchWorkload().probe()
    assert probe.usable or probe.reason
    assert not (probe.usable and probe.torch is None)

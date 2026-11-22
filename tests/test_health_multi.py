# SPDX-License-Identifier: MIT
"""`multi.p2p`: copies between every ordered pair of GPUs, on machines with two GPUs or more.

Rules and thresholds are those of the "P2P copies" line of hwcheck's acceptance.sh 1.1,
replayed on its fixtures: what the copy worker would report is given by each test. The worker
itself runs here on the CPU build of PyTorch when one is installed (same code path, other
device); no test of this file drives a real GPU.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import types

import pytest
from health_fakes import FakeHost, FakeWorkload, healthy_copies

from voltage_verify.health import multi_checks, smi, worker
from voltage_verify.health import workload as workload_module
from voltage_verify.health.report import FAIL, INFO, PASS, SKIP, WARN, canonical_report, load_report
from voltage_verify.health.run import Options, run_health
from voltage_verify.health.workload import PeerCopies, Probe, TorchWorkload, peer_copies

TORCH = Probe("2.14.1+cu130", True, 8, "")
EIGHT = list(range(8))
PCIE = 19_073  # MiB/s: 20 GB/s, a pair that goes over PCIe
OLD_NOTE = "read only: P2P copies and NCCL are not measured here (hwcheck's acceptance.sh does)"


def _inventory(fixture: str) -> list[dict[str, str]]:
    gpus = smi.Smi(FakeHost(fixture)).query(smi.INVENTORY_FIELDS)
    assert gpus is not None
    return gpus


def _topo(fixture: str) -> str | None:
    return smi.Smi(FakeHost(fixture)).text("topo", "-m")


def _check(fixture: str, copies: PeerCopies | None, probe: Probe = TORCH, model: str | None = None):
    gpus = _inventory(fixture)
    if model is not None:
        gpus = [{**gpu, "name": model} for gpu in gpus]
    return multi_checks.p2p(probe, gpus, copies, _topo(fixture))


# --- the verdict -------------------------------------------------------------------------------


def test_a_healthy_nvlink_node_passes_with_its_56_pairs():
    check = _check("h100-8x", healthy_copies(EIGHT))
    assert check.name == "multi.p2p" and check.status == PASS
    assert check.detail == (
        "56/56 GPU pairs: 256 MiB copied from each GPU to every other one, 3 warm-up copies then 20 timed ones; "
        "the 3 warm-up copies and the last timed copy compared byte for byte: 0 wrong bytes; "
        "slowest pair 389 GB/s (GPU 0>1), median 389 GB/s "
        "(FAIL under 112, WARN under 337: H100 SXM line rate 450 GB/s per direction)"
    )
    assert len(check.data["pairs"]) == 56
    assert check.data["pairs"][0] == {"src": 0, "dst": 1, "mib_per_s": 371_000, "wrong_bytes": 0, "peer": True}
    assert check.data["nvlink_mesh"] == "full" and check.data["line_rate_gb_per_s"] == 450
    assert check.data["median_mib_per_s"] == 371_000 and check.data["slowest_mib_per_s"] == 371_000


def test_a_pair_whose_bytes_differ_fails_and_is_named():
    check = _check("h100-8x", healthy_copies(EIGHT, {(2, 5): {"wrong_bytes": 4096}}))
    assert check.status == FAIL
    assert check.detail.startswith("bytes differ after the copy on GPU 2>5 (4096 wrong bytes)")


def test_wrong_bytes_fail_over_pcie_too():
    # acceptance.sh only prints the corrupted pairs of a machine without NVLink (N/A line);
    # here a wrong byte fails whatever links the GPUs.
    check = _check("pcie-8x", healthy_copies(EIGHT, {(6, 1): {"mib_per_s": PCIE, "wrong_bytes": 1}}))
    assert check.status == FAIL
    assert check.detail.startswith("bytes differ after the copy on GPU 6>1 (1 wrong byte)")


def test_a_pair_under_90_percent_of_the_median_is_a_warning():
    check = _check("h100-8x", healthy_copies(EIGHT, {(3, 4): {"mib_per_s": 320_000}}))  # 86% of the median
    assert check.status == WARN
    assert check.detail == "GPU 3>4 at 336 GB/s, under 90% of the median pair (389 GB/s)"


def test_a_pair_at_90_percent_of_the_median_passes():
    check = _check("h100-8x", healthy_copies(EIGHT, {(3, 4): {"mib_per_s": 333_900}}))
    assert check.status == PASS
    assert "slowest pair 350 GB/s (GPU 3>4), median 389 GB/s" in check.detail


def test_a_pair_under_a_quarter_of_the_line_rate_is_not_on_nvlink():
    check = _check("h100-8x", healthy_copies(EIGHT, {(0, 7): {"mib_per_s": PCIE}}))
    assert check.status == FAIL
    assert check.detail == (
        "GPU 0>7 at 20.0 GB/s, under 112 GB/s (25% of the 450 GB/s line rate of H100 SXM): not on NVLink"
    )


def test_without_a_complete_mesh_an_nvlink_model_is_warned_on_its_slow_pairs_never_failed():
    # The line rate is known (H100 SXM) but nvidia-smi topo -m does not show every pair on
    # NVLink: the machine may be a VM that got part of a board. Slow pairs are named, as a warning.
    off = {pair: {"mib_per_s": PCIE, "peer": False} for i in range(7) for pair in ((i, 7), (7, i))}
    check = _check("h100-8x-sick", healthy_copies(EIGHT, off))
    assert check.status == WARN and check.data["nvlink_mesh"] == "partial"
    assert check.detail == (
        "GPU 0>7, GPU 1>7, GPU 2>7, GPU 3>7, GPU 4>7, GPU 5>7 and 8 more at 20.0 GB/s or less, under 112 GB/s "
        "(25% of the 450 GB/s line rate of H100 SXM): H100 SXM is an NVLink model, but nvidia-smi topo -m "
        "shows NVLink between some pairs only: the speed is reported, not failed"
    )
    assert "no peer access" not in check.detail  # judged on a complete mesh only, as acceptance.sh does


def test_two_gpus_of_an_nvlink_board_in_a_vm_without_nvlink_are_never_failed_on_speed():
    two = [{**gpu, "name": "NVIDIA H200"} for gpu in _inventory("pcie-8x")[:2]]
    slow = healthy_copies([0, 1], {(0, 1): {"mib_per_s": PCIE}, (1, 0): {"mib_per_s": PCIE}})
    check = multi_checks.p2p(TORCH, two, slow, _topo("pcie-8x"))
    assert check.status == WARN and check.data["nvlink_mesh"] == "none"
    assert check.detail == (
        "GPU 0>1, GPU 1>0 at 20.0 GB/s or less, under 112 GB/s (25% of the 450 GB/s line rate of H200 SXM): "
        "H200 SXM is an NVLink model, but nvidia-smi topo -m shows no NVLink between these GPUs: "
        "the speed is reported, not failed"
    )
    fast = multi_checks.p2p(TORCH, two, healthy_copies([0, 1]), _topo("pcie-8x"))
    assert fast.status == PASS
    assert fast.detail.endswith(
        "(WARN under 112 for a pair, under 337 for the median: H200 SXM line rate 450 GB/s per direction; "
        "never FAIL on speed: nvidia-smi topo -m shows no NVLink between these GPUs)"
    )


def test_without_a_complete_mesh_wrong_bytes_and_missing_pairs_still_fail():
    two = [{**gpu, "name": "NVIDIA H200"} for gpu in _inventory("pcie-8x")[:2]]
    corrupted = healthy_copies([0, 1], {(0, 1): {"mib_per_s": PCIE, "wrong_bytes": 2}})
    assert multi_checks.p2p(TORCH, two, corrupted, _topo("pcie-8x")).status == FAIL
    one_way = healthy_copies([0, 1])
    missing = PeerCopies(one_way.summary, one_way.pairs[:1])
    assert multi_checks.p2p(TORCH, two, missing, _topo("pcie-8x")).status == FAIL


def test_a_median_under_75_percent_of_the_line_rate_is_a_warning():
    slow = {(i, j): {"mib_per_s": 300_000} for i in EIGHT for j in EIGHT if i != j}
    check = _check("h100-8x", healthy_copies(EIGHT, slow))
    assert check.status == WARN
    assert check.detail == "median pair 315 GB/s, under 337 GB/s (75% of the H100 SXM line rate)"


def test_no_peer_access_on_a_complete_mesh_fails():
    check = _check("h100-8x", healthy_copies(EIGHT, {(1, 6): {"peer": False}}))
    assert check.status == FAIL
    assert check.detail.startswith("no peer access on GPU 1>6")


def test_a_model_without_a_known_line_rate_is_judged_against_the_median():
    passed = _check("h100-8x", healthy_copies(EIGHT), model="NVIDIA X900")
    assert passed.status == PASS and passed.data["line_rate_gb_per_s"] == 0
    assert passed.detail.endswith("(no line rate known for this model: each pair is judged against the median)")
    quarter = _check("h100-8x", healthy_copies(EIGHT, {(5, 2): {"mib_per_s": 90_000}}), model="NVIDIA X900")
    assert quarter.status == FAIL
    assert quarter.detail == "GPU 5>2 at 94.4 GB/s, under 25% of the median pair (389 GB/s)"
    spread = _check("h100-8x", healthy_copies(EIGHT, {(5, 2): {"mib_per_s": 100_000}}), model="NVIDIA X900")
    assert spread.status == WARN
    assert spread.detail == "GPU 5>2 at 105 GB/s, under 90% of the median pair (389 GB/s)"


def test_over_pcie_the_speed_is_measured_and_not_judged():
    pairs = {(i, j): {"mib_per_s": 12_000, "peer": False} for i in EIGHT for j in EIGHT if i != j}
    pairs[(0, 4)] = {"mib_per_s": 2_000, "peer": False}  # a sixth of the median: another socket, a busy switch
    check = _check("pcie-8x", healthy_copies(EIGHT, pairs))
    assert check.status == INFO
    assert check.detail == (
        "56/56 GPU pairs: 256 MiB copied from each GPU to every other one, 3 warm-up copies then 20 timed ones; "
        "the 3 warm-up copies and the last timed copy compared byte for byte: 0 wrong bytes; "
        "slowest pair 2.1 GB/s (GPU 0>4), median 12.6 GB/s "
        "(no NVLink mesh: copies over PCIe, speed not judged; WARN under 10% of the median pair)"
    )
    assert check.data["nvlink_mesh"] == "none"


def test_over_pcie_a_pair_almost_at_a_standstill_is_a_warning_that_names_it():
    two = [{**gpu, "name": "NVIDIA A40"} for gpu in _inventory("pcie-8x")[:2]]
    stalled = healthy_copies([0, 1], {(0, 1): {"mib_per_s": 1}, (1, 0): {"mib_per_s": 300_000}})
    check = multi_checks.p2p(TORCH, two, stalled, _topo("pcie-8x"))
    assert check.status == WARN
    assert check.detail == (
        "GPU 0>1 at 0.0 GB/s, under 10% of the median pair (157 GB/s); "
        "no NVLink mesh: copies over PCIe, speed not judged above that"
    )


def test_over_pcie_ten_percent_of_the_median_is_the_limit():
    pairs = {(i, j): {"mib_per_s": 12_000} for i in EIGHT for j in EIGHT if i != j}
    at = _check("pcie-8x", healthy_copies(EIGHT, {**pairs, (0, 4): {"mib_per_s": 1_200}}))
    assert at.status == INFO
    under = _check("pcie-8x", healthy_copies(EIGHT, {**pairs, (0, 4): {"mib_per_s": 1_199}, (4, 0): {"mib_per_s": 600}}))
    assert under.status == WARN
    assert under.detail.startswith("GPU 0>4, GPU 4>0 at 1.3 GB/s or less, under 10% of the median pair (12.6 GB/s); ")


def test_without_a_readable_topology_the_speed_is_not_judged_and_the_line_says_why():
    slow = healthy_copies(EIGHT, {(0, 4): {"mib_per_s": 40_000}})  # 10.8% of the median
    check = multi_checks.p2p(TORCH, _inventory("pcie-8x"), slow, None)
    assert check.status == INFO and check.data["nvlink_mesh"] == "unknown"
    assert check.detail.endswith("(nvidia-smi topo -m not readable, speed not judged; WARN under 10% of the median pair)")
    sxm = multi_checks.p2p(TORCH, _inventory("h100-8x"), slow, None)  # an SXM model: warned, as its mesh is not seen
    assert sxm.status == WARN
    assert sxm.detail.endswith(
        "H100 SXM is an NVLink model, but nvidia-smi topo -m could not be read: the speed is reported, not failed"
    )


@pytest.mark.parametrize(
    ("copies", "reason"),
    [
        (None, "the copy test did not finish (no result from its worker): 0 of 56 pairs measured"),
        (
            PeerCopies({"stage": "p2p", "error": "no result from its worker"}, []),
            "the copy test did not finish (no result from its worker): 0 of 56 pairs measured",
        ),
        (
            PeerCopies({"stage": "p2p", "error": "stopped at the deadline (900 s)"}, healthy_copies(EIGHT).pairs[:12]),
            "the copy test did not finish (stopped at the deadline (900 s)): 12 of 56 pairs measured",
        ),
        (
            PeerCopies({"stage": "p2p", "error": "RuntimeError: CUDA error"}, healthy_copies(EIGHT).pairs),
            "the copy test did not finish (RuntimeError: CUDA error): 56 of 56 pairs measured",
        ),
    ],
)
def test_a_copy_test_that_does_not_finish_fails_with_the_reason(copies, reason):
    check = _check("h100-8x", copies)
    assert check.status == FAIL  # acceptance.sh: "the copy test failed"; never a pass
    assert check.detail == reason


def test_wrong_bytes_are_named_even_when_the_test_did_not_finish():
    partial = healthy_copies(EIGHT, {(0, 3): {"wrong_bytes": 7}}).pairs[:12]
    check = _check("h100-8x", PeerCopies({"stage": "p2p", "error": "stopped at the deadline (900 s)"}, partial))
    assert check.status == FAIL
    assert "bytes differ after the copy on GPU 0>3 (7 wrong bytes)" in check.detail


def test_a_pair_left_out_or_without_a_measure_is_never_a_pass():
    full = healthy_copies(EIGHT)
    missing = PeerCopies(full.summary, [p for p in full.pairs if (p["src"], p["dst"]) != (4, 1)])
    check = _check("h100-8x", missing)
    assert check.status == FAIL and check.detail == "no valid result for 1 of 56 pairs (GPU 4>1)"
    unmeasured = [{k: v for k, v in p.items() if k != "wrong_bytes"} if p["src"] == 6 else p for p in full.pairs]
    check = _check("h100-8x", PeerCopies(full.summary, unmeasured))
    assert check.status == FAIL
    assert check.detail == (
        "no valid result for 7 of 56 pairs (GPU 6>0, GPU 6>1, GPU 6>2, GPU 6>3, GPU 6>4, GPU 6>5 and 1 more)"
    )
    stalled = healthy_copies(EIGHT, {(2, 3): {"mib_per_s": 0}})
    assert _check("h100-8x", stalled).detail == "no valid result for 1 of 56 pairs (GPU 2>3)"


def test_a_pair_reported_twice_fails_a_clean_line_must_not_hide_a_corrupted_one():
    full = healthy_copies(EIGHT)
    corrupted = {"src": 0, "dst": 1, "mib_per_s": 371_000, "wrong_bytes": 5, "peer": True}
    check = _check("h100-8x", PeerCopies(full.summary, [corrupted, *full.pairs]))  # then the clean line of 0>1
    assert check.status == FAIL
    assert check.detail == "reported twice: GPU 0>1"
    assert [p["wrong_bytes"] for p in check.data["pairs"][:2]] == [5, 0]  # both lines are in the report
    same = _check("h100-8x", PeerCopies(full.summary, [full.pairs[0], *full.pairs]))
    assert same.status == FAIL and same.detail == "reported twice: GPU 0>1"


@pytest.mark.parametrize(
    "value",
    [
        {"mib_per_s": 10**9 + 1},  # more than any link carries: not a measure
        {"mib_per_s": -5},
        {"wrong_bytes": 4 * (256 << 20) + 1},  # more bytes than were compared
        {"wrong_bytes": -1},
        {"mib_per_s": True},
        {"wrong_bytes": "0"},
    ],
)
def test_a_value_out_of_bounds_is_not_a_result(value):
    check = _check("h100-8x", healthy_copies(EIGHT, {(3, 6): value}))
    assert check.status == FAIL
    assert check.detail == "no valid result for 1 of 56 pairs (GPU 3>6)"


def test_values_at_the_bounds_are_results():
    fastest = _check("h100-8x", healthy_copies(EIGHT, {(3, 6): {"mib_per_s": 10**9}}))
    assert fastest.status == PASS
    all_wrong = _check("h100-8x", healthy_copies(EIGHT, {(3, 6): {"wrong_bytes": 4 * (256 << 20)}}))
    assert all_wrong.status == FAIL
    assert all_wrong.detail.startswith("bytes differ after the copy on GPU 3>6 (1073741824 wrong bytes)")


@pytest.mark.parametrize(
    ("closing", "said"),
    [
        ({"bytes": 0}, "0 bytes, 3 warm-up and 20 timed copies, 56 pairs for 56 expected"),
        ({"bytes": -1}, "-1 bytes, 3 warm-up and 20 timed copies, 56 pairs for 56 expected"),
        ({"iterations": 0}, "268435456 bytes, 3 warm-up and 0 timed copies, 56 pairs for 56 expected"),
        ({"warmup": -1}, "268435456 bytes, -1 warm-up and 20 timed copies, 56 pairs for 56 expected"),
        ({"warmup": None}, "268435456 bytes, ? warm-up and 20 timed copies, 56 pairs for 56 expected"),
        ({"pairs": 55}, "268435456 bytes, 3 warm-up and 20 timed copies, 55 pairs for 56 expected"),
        ({"pairs": 57}, "268435456 bytes, 3 warm-up and 20 timed copies, 57 pairs for 56 expected"),
    ],
)
def test_a_closing_record_that_does_not_add_up_fails(closing, said):
    check = _check("h100-8x", healthy_copies(EIGHT, **closing))
    assert check.status == FAIL
    assert check.detail == f"the closing record of the copy worker is not valid ({said})"


def test_a_closing_record_without_its_counts_fails():
    full = healthy_copies(EIGHT)
    bare = {k: v for k, v in full.summary.items() if k not in ("bytes", "iterations", "warmup", "pairs")}
    check = _check("h100-8x", PeerCopies(bare, full.pairs))
    assert check.status == FAIL
    assert check.detail == (
        "the closing record of the copy worker is not valid (? bytes, ? warm-up and ? timed copies, ? pairs for 56 expected)"
    )


@pytest.mark.parametrize(
    "ended", ["its worker ended with exit code 139", "stopped at the deadline (900 s), and still there after the kill"]
)
def test_a_worker_that_ends_badly_after_its_closing_record_is_a_warning(ended):
    check = _check("h100-8x", healthy_copies(EIGHT, ended=ended))
    assert check.status == WARN  # every pair is there and judged: not a failure, not a plain pass either
    assert check.detail == f"every pair was reported, but the copy worker did not end well ({ended})"
    corrupted = _check("h100-8x", healthy_copies(EIGHT, {(2, 5): {"wrong_bytes": 1}}, ended=ended))
    assert corrupted.status == FAIL and corrupted.detail.endswith(f"; also {check.detail}")


def test_gpus_nvidia_smi_does_not_tell_apart_are_never_zero_pairs_for_information():
    twins = [{**gpu, "index": "0"} for gpu in _inventory("pcie-8x")[:2]]
    check = multi_checks.p2p(TORCH, twins, healthy_copies([0, 1]), _topo("pcie-8x"))
    assert check.status == FAIL
    assert check.detail == "nvidia-smi does not tell the 2 GPUs apart (indices 0, 0): no pair is judged"
    mixed = [{**gpu, "index": index} for gpu, index in zip(_inventory("pcie-8x")[:3], ("0", "1", "1"), strict=True)]
    assert multi_checks.p2p(TORCH, mixed, healthy_copies([0, 1, 2]), _topo("pcie-8x")).status == FAIL
    alone = multi_checks.p2p(TORCH, _inventory("rtx3060ti-1x"), healthy_copies([0]), None)
    assert alone.status == FAIL and alone.detail == "fewer than two GPUs: no pair to copy between"


def test_pytorch_must_initialise_every_gpu_nvidia_smi_lists():
    check = _check("h100-8x", healthy_copies(EIGHT[:7]))
    assert check.status == FAIL
    assert check.detail == "PyTorch initialises 7 of the 8 GPUs nvidia-smi lists: no pair is judged"
    assert check.data["pairs"] == []  # the worker's device numbers no longer say which GPU is which


def test_without_a_cuda_pytorch_the_copies_are_skipped_with_the_reason():
    probe = Probe("2.3.1+cpu", False, 0, "PyTorch 2.3.1+cpu sees no CUDA device")
    check = _check("pcie-8x", None, probe=probe)
    assert check.status == SKIP
    assert check.detail == (
        "PyTorch 2.3.1+cpu sees no CUDA device: the GPU-to-GPU copy test did not run "
        "(install a CUDA build of PyTorch to run it; nothing is installed for you)"
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("NVIDIA H100 80GB HBM3", ("H100 SXM", 450)),
        ("NVIDIA H100-SXM5-80GB", ("H100 SXM", 450)),
        ("NVIDIA H100 PCIe", ("", 0)),
        ("NVIDIA H100", ("", 0)),
        ("NVIDIA H200", ("H200 SXM", 450)),
        ("NVIDIA A100-SXM4-80GB", ("A100 SXM", 300)),
        ("NVIDIA A100-PCIE-40GB", ("", 0)),
        ("NVIDIA B200", ("B200 SXM", 900)),
        ("NVIDIA B300", ("B300 SXM", 900)),
        ("NVIDIA GB200 NVL72", ("", 0)),
        ("NVIDIA L40S", ("", 0)),
        ("NVIDIA GeForce RTX 3060 Ti", ("", 0)),
    ],
)
def test_nvlink_line_rate_by_model_as_in_acceptance_sh(name, expected):
    assert multi_checks.line_rate(name) == expected


# --- the run -----------------------------------------------------------------------------------


def _names(report) -> list[str]:
    return [c.name for c in report.checks]


def test_a_node_with_several_gpus_gets_the_p2p_line_after_the_load():
    host = FakeHost("h100-8x")
    workload = FakeWorkload(host)
    said: list[str] = []
    report = run_health(Options(quick=True), host, workload, progress=said.append)
    assert _names(report)[-2:] == ["topology.links", "multi.p2p"]
    check = report.checks[-1]
    assert check.status == PASS and len(check.data["pairs"]) == 56
    assert report.summary.verdict == "PASS"
    assert workload.copied == EIGHT, "one worker that sees every GPU, by nvidia-smi index"
    assert "GPU-to-GPU copies between each of the 56 ordered pairs of 8 GPUs" in said
    doc = report.to_json()
    assert load_report(canonical_report(doc)) == doc  # canonical, and no float anywhere


def test_error_counters_and_the_kernel_log_are_read_after_the_copies():
    host = FakeHost("h100-8x")
    workload = FakeWorkload(host)
    run_health(Options(quick=True), host, workload)
    after = host.calls[workload.calls_before_copies :]
    assert any(call[0] == "nvidia-smi" and "ecc.errors" in call[1] for call in after)
    assert any(call[0] == "dmesg" for call in after)


def test_a_corrupted_pair_fails_the_report():
    host = FakeHost("h100-8x")
    workload = FakeWorkload(host, copies=healthy_copies(EIGHT, {(2, 5): {"wrong_bytes": 4096}}))
    report = run_health(Options(quick=True), host, workload)
    assert report.summary.verdict == "FAIL (multi.p2p)"
    assert "GPU 2>5" in report.checks[-1].detail


def test_the_sick_node_with_a_gpu_off_nvlink_is_warned_on_its_copies():
    host = FakeHost("h100-8x-sick")
    off = {pair: {"mib_per_s": PCIE, "peer": False} for i in range(7) for pair in ((i, 7), (7, i))}
    report = run_health(Options(quick=True), host, FakeWorkload(host, copies=healthy_copies(EIGHT, off)))
    assert report.summary.failed == ["errors.ecc", "errors.xid", "sensors.thermal"]
    assert report.summary.warnings[-1] == "multi.p2p"
    assert "the speed is reported, not failed" in report.checks[-1].detail


def test_no_copy_worker_starts_when_nvidia_smi_gives_two_gpus_the_same_index():
    host = FakeHost("pcie-8x")
    host.overrides[1] = {"index": "0"}
    workload = FakeWorkload(host)
    report = run_health(Options(quick=True), host, workload)
    check = report.checks[-1]
    assert (check.name, check.status) == ("multi.p2p", FAIL)
    assert check.detail.startswith("nvidia-smi does not tell the 8 GPUs apart (indices 0, 0, 2, 3, 4, 5, 6, 7)")
    assert workload.copied is None, "CUDA_VISIBLE_DEVICES would name the same GPU twice"


def test_a_copy_worker_that_never_answers_fails_the_report():
    host = FakeHost("pcie-8x")
    silent = PeerCopies({"stage": "p2p", "error": "stopped at the deadline (900 s)"}, [])
    report = run_health(Options(quick=True), host, FakeWorkload(host, copies=silent))
    assert report.checks[-1].status == FAIL
    assert "multi.p2p" in report.summary.failed


def test_several_gpus_without_a_cuda_pytorch_the_line_says_skip():
    host = FakeHost("pcie-8x")
    workload = FakeWorkload(host, torch=None)
    report = run_health(Options(quick=True), host, workload)
    check = report.checks[-1]
    assert (check.name, check.status) == ("multi.p2p", SKIP)
    assert check.detail.startswith("PyTorch is not installed in this Python: the GPU-to-GPU copy test did not run")
    assert report.summary.not_tested[-1] == "multi.p2p"
    assert workload.copied is None, "no copy worker may start without a CUDA build of PyTorch"


@pytest.mark.parametrize("torch", ["2.14.1+cu130", None])
def test_a_single_gpu_machine_gets_no_p2p_line_and_the_same_output_as_before(torch):
    host = FakeHost("rtx3060ti-1x")
    workload = FakeWorkload(host, torch=torch)
    report = run_health(Options(quick=True), host, workload)
    assert _names(report) == [
        "inventory.gpus",
        "errors.ecc",
        "errors.xid",
        "sensors.thermal",
        "sensors.clocks",
        "compute.matmul",
        "vram.pattern",
        "topology.links",
    ]
    assert report.checks[-1].detail.endswith(OLD_NOTE)
    assert workload.copied is None, "no copy worker on a single GPU"


def test_the_topology_line_of_a_multi_gpu_node_no_longer_says_p2p_is_not_measured():
    host = FakeHost("h100-8x")
    report = run_health(Options(quick=True), host, FakeWorkload(host))
    detail = next(c.detail for c in report.checks if c.name == "topology.links")
    assert OLD_NOTE not in detail
    assert detail.endswith(
        "read only: NCCL and the NVLink and PCIe link verdicts are not measured here "
        "(hwcheck's acceptance.sh does); GPU-to-GPU copies: multi.p2p"
    )


def test_no_gpu_no_p2p_line():
    host = FakeHost(None)
    report = run_health(Options(quick=True), host, FakeWorkload(host))
    assert "multi.p2p" not in _names(report)


# --- the copy stage of the worker, on the CPU --------------------------------------------------


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def test_copies_between_every_ordered_pair_arrive_as_sent(torch):
    records = list(worker.p2p(torch, ["cpu", "cpu", "cpu"], size=1 << 20, iterations=2))
    pairs, closing = records[:-1], records[-1]
    assert [(r["src"], r["dst"]) for r in pairs] == [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    assert all(r["stage"] == "p2p_pair" and r["wrong_bytes"] == 0 and r["mib_per_s"] > 0 for r in pairs)
    assert all(r["peer"] is True for r in pairs)
    assert {k: v for k, v in closing.items() if k != "elapsed_ms"} == {
        "stage": "p2p",
        "devices": 3,
        "bytes": 1 << 20,
        "warmup": 3,
        "iterations": 2,
        "pairs": 6,
    }
    assert all(type(v) in (int, str, bool) for r in records for v in r.values()), "no float may enter the report"


def test_each_warm_up_copy_and_the_last_timed_copy_are_compared(torch):
    compared = []

    def watch(arrived, src, dst, check):
        compared.append((src, dst, check))

    list(worker.p2p(torch, ["cpu", "cpu"], size=1 << 20, iterations=5, fault=watch))
    assert compared == [(0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 1, 3), (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 0, 3)]


def test_a_byte_changed_on_the_way_is_counted_on_its_pair_only(torch):
    def flips(arrived, src, dst, check):
        if (src, dst, check) == (0, 2, 3):  # the last timed copy
            arrived[100] ^= 0xFF
            arrived[101] ^= 0x01
            arrived[-1] ^= 0x80

    records = list(worker.p2p(torch, ["cpu", "cpu", "cpu"], size=1 << 20, iterations=1, fault=flips))
    assert {(r["src"], r["dst"]): r["wrong_bytes"] for r in records[:-1]} == {
        (0, 1): 0,
        (0, 2): 3,
        (1, 0): 0,
        (1, 2): 0,
        (2, 0): 0,
        (2, 1): 0,
    }


def test_wrong_bytes_of_the_warm_up_copies_are_added_to_those_of_the_last_copy(torch):
    def flips(arrived, src, dst, check):
        if (src, dst) == (1, 0) and check in (0, 2, 3):  # two warm-up copies and the last timed one
            arrived[check] ^= 0xFF
        if (src, dst, check) == (1, 0, 2):
            arrived[7] ^= 0x01

    records = list(worker.p2p(torch, ["cpu", "cpu"], size=1 << 20, iterations=2, fault=flips))
    assert {(r["src"], r["dst"]): r["wrong_bytes"] for r in records[:-1]} == {(0, 1): 0, (1, 0): 4}


def test_a_copy_that_did_not_happen_is_seen_each_source_holds_other_bytes(torch):
    kept = {}

    def stale(arrived, src, dst, check):
        if (src, dst, check) == (0, 2, 3):
            kept["from 0"] = arrived.clone()
        if (src, dst, check) == (1, 2, 0):  # at its first copy the destination still holds what GPU 0 sent
            arrived.copy_(kept["from 0"])

    records = list(worker.p2p(torch, ["cpu", "cpu", "cpu"], size=1 << 20, iterations=1, fault=stale))
    wrong = {(r["src"], r["dst"]): r["wrong_bytes"] for r in records[:-1]}
    assert wrong[(1, 2)] == 1 << 20 and wrong[(0, 2)] == 0


def test_peer_access_is_asked_from_cuda_for_cuda_devices_only():
    asked = []

    def can_device_access_peer(src, dst):
        asked.append((src, dst))
        return (src, dst) != (0, 1)

    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(can_device_access_peer=can_device_access_peer))
    devices = ["cuda:0", "cuda:1"]
    assert worker.peer_access(fake, devices, 0, 1) is False
    assert worker.peer_access(fake, devices, 1, 0) is True
    assert worker.peer_access(fake, ["cpu", "cpu"], 0, 1) is True
    assert asked == [(0, 1), (1, 0)]


def test_p2p_command_reports_each_pair_then_how_it_ended(monkeypatch, capsys):
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(device_count=lambda: 2))
    monkeypatch.setattr(importlib, "import_module", lambda name: fake)

    def falls_off_the_bus(torch, devices):
        assert devices == ["cuda:0", "cuda:1"]
        yield {"stage": "p2p_pair", "src": 0, "dst": 1, "mib_per_s": 371000, "wrong_bytes": 0, "peer": True}
        raise RuntimeError("CUDA error: unspecified launch failure " + "x" * 500)

    monkeypatch.setattr(worker, "p2p", falls_off_the_bus)
    assert worker.main(["p2p"]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["stage"] for line in lines] == ["p2p_pair", "p2p"]
    assert lines[1]["error"].startswith("RuntimeError: CUDA error: unspecified launch failure")
    assert len(lines[1]["error"]) < 250


# --- reading what the copy worker prints -------------------------------------------------------


def _line(**record) -> str:
    return json.dumps(record)


def test_pairs_are_read_by_nvidia_smi_index_and_only_known_fields_are_kept():
    text = "\n".join(
        [
            "a warning PyTorch printed on stdout",
            _line(stage="p2p_pair", src=0, dst=1, mib_per_s=371000, wrong_bytes=0, peer=True, extra="dropped"),
            _line(stage="p2p_pair", src=1, dst=2, mib_per_s=370000.5, wrong_bytes="0", peer=1),  # no valid measure
            _line(stage="p2p_pair", src=2, dst=2, mib_per_s=1, wrong_bytes=0),  # not a pair
            _line(stage="p2p_pair", src=0, dst=3, mib_per_s=1, wrong_bytes=0),  # no such device
            _line(stage="p2p_pair", src=-1, dst=0, mib_per_s=1, wrong_bytes=0),
            _line(stage="p2p_pair", src=True, dst=0, mib_per_s=1, wrong_bytes=0),  # a bool is not a device
            _line(stage="p2p_pair", src=0, dst=1, mib_per_s=5, wrong_bytes=0, peer=True),  # the first pair, again
            _line(stage="p2p", devices=3, bytes=268435456, warmup=3, iterations=20, pairs=6, elapsed_ms=900, note="no"),
        ]
    )
    copies = peer_copies(text, [0, 3, 5])
    assert copies.summary == {
        "stage": "p2p",
        "devices": 3,
        "bytes": 268435456,
        "warmup": 3,
        "iterations": 20,
        "pairs": 6,
        "elapsed_ms": 900,
    }
    assert copies.pairs == [  # every line that names a pair, in order: a second line never replaces the first
        {"src": 0, "dst": 3, "mib_per_s": 371000, "wrong_bytes": 0, "peer": True},
        {"src": 3, "dst": 5},
        {"src": 0, "dst": 3, "mib_per_s": 5, "wrong_bytes": 0, "peer": True},
    ]
    assert copies.ended == ""


def test_a_worker_that_printed_nothing_says_how_it_ended():
    assert peer_copies("", [0, 1]).summary == {"stage": "p2p", "error": "no result from its worker"}
    ended = peer_copies(_line(stage="p2p_pair", src=0, dst=1, mib_per_s=5, wrong_bytes=0), [0, 1], "stopped at the deadline")
    assert ended.summary == {"stage": "p2p", "error": "stopped at the deadline"}
    assert len(ended.pairs) == 1  # what was measured before is kept


def test_how_the_worker_ended_is_kept_next_to_a_closing_record():
    closing = _line(stage="p2p", devices=2, bytes=1, warmup=3, iterations=1, pairs=0, elapsed_ms=1)
    copies = peer_copies(closing, [0, 1], "its worker ended with exit code 139")
    assert "error" not in copies.summary and copies.ended == "its worker ended with exit code 139"


def test_no_peer_access_reported_by_the_worker_fails_on_the_real_8x_h100_topology():
    # From the worker's lines to the verdict, with the nvidia-smi topo -m of a real 8x H100 SXM node:
    # every pair reads NV18, CUDA says GPU 1 cannot reach GPU 6.
    lines = [
        _line(stage="p2p_pair", src=i, dst=j, mib_per_s=371_000, wrong_bytes=0, peer=(i, j) != (1, 6))
        for i in EIGHT
        for j in EIGHT
        if i != j
    ]
    lines.append(_line(stage="p2p", devices=8, bytes=256 << 20, warmup=3, iterations=20, pairs=56, elapsed_ms=21000))
    check = _check("h100-8x", peer_copies("\n".join(lines), EIGHT))
    assert check.data["nvlink_mesh"] == "full"
    assert check.status == FAIL and check.detail == "no peer access on GPU 1>6"


# --- the process -------------------------------------------------------------------------------

STUB = """
import json, os, sys, time
assert sys.argv[1:] == ["p2p"], sys.argv
assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
seen = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
mode = os.environ.get("STUB_MODE")
print(json.dumps({"stage": "p2p_pair", "src": 0, "dst": 1, "mib_per_s": int(seen[1]), "wrong_bytes": 0,
                  "peer": not os.path.samefile(os.getcwd(), os.environ["STUB_STARTED_IN"])}), flush=True)
if mode == "hang":
    time.sleep(60)
if mode == "crash":
    os._exit(139)
print("x" * 200000)
print(json.dumps({"stage": "p2p", "devices": len(seen), "bytes": 1, "iterations": 1, "pairs": 1, "elapsed_ms": 1}),
      flush=True)
if mode == "late-crash":
    os._exit(139)
"""


@pytest.fixture
def stub(tmp_path, monkeypatch):
    path = tmp_path / "stub_copies.py"
    path.write_text(STUB, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STUB_STARTED_IN", str(tmp_path))
    return [sys.executable, str(path)]


def test_one_worker_sees_every_gpu_in_pci_order_from_a_private_directory(stub):
    copies = TorchWorkload(stub).copies([2, 4, 6])
    assert copies.summary == {"stage": "p2p", "devices": 3, "bytes": 1, "iterations": 1, "pairs": 1, "elapsed_ms": 1}
    # mib_per_s carries the second visible device, peer whether the worker started elsewhere
    assert copies.pairs == [{"src": 2, "dst": 4, "mib_per_s": 4, "wrong_bytes": 0, "peer": True}]
    assert copies.ended == ""


def test_a_copy_worker_that_hangs_is_stopped_at_the_deadline(stub, monkeypatch):
    monkeypatch.setenv("STUB_MODE", "hang")
    copies = TorchWorkload(stub, p2p_timeout=3.0).copies([0, 1])
    assert copies.summary == {"stage": "p2p", "error": "stopped at the deadline (3 s)"}
    assert len(copies.pairs) == 1, "what it reported before is kept"


def test_a_copy_worker_that_dies_says_its_exit_code(stub, monkeypatch):
    monkeypatch.setenv("STUB_MODE", "crash")
    copies = TorchWorkload(stub).copies([0, 1])
    assert copies.summary == {"stage": "p2p", "error": "its worker ended with exit code 139"}


def test_a_copy_worker_that_dies_after_its_closing_record_is_not_taken_for_a_clean_end(stub, monkeypatch):
    monkeypatch.setenv("STUB_MODE", "late-crash")
    copies = TorchWorkload(stub).copies([0, 1])
    assert copies.summary["pairs"] == 1 and "error" not in copies.summary
    assert copies.ended == "its worker ended with exit code 139"


class _Undying:
    """A worker stuck in the driver: it does not end, even killed. Waiting for it without a
    limit would hang the report."""

    made: list[_Undying] = []

    def __init__(self, *_args, **_kwargs) -> None:
        self.killed = 0
        self.waits: list[float | None] = []
        _Undying.made.append(self)

    def poll(self) -> None:
        return None

    def kill(self) -> None:
        self.killed += 1

    def wait(self, timeout: float | None = None) -> int:
        self.waits.append(timeout)
        assert timeout is not None, "an unbounded wait on a worker"
        raise subprocess.TimeoutExpired("worker", timeout)


@pytest.fixture
def undying(monkeypatch):
    _Undying.made = []
    monkeypatch.setattr(workload_module.subprocess, "Popen", _Undying)
    return _Undying.made


def test_the_wait_for_a_killed_copy_worker_is_bounded(undying):
    copies = TorchWorkload(["worker"], p2p_timeout=900.0).copies([0, 1])
    assert copies.summary == {
        "stage": "p2p",
        "error": "stopped at the deadline (900 s), and still there after the kill",
    }
    assert copies.ended == copies.summary["error"]
    (process,) = undying
    assert process.killed == 1 and process.waits == [900.0, workload_module.KILL_WAIT]


def test_the_wait_for_a_killed_load_worker_is_bounded(undying):
    workload = TorchWorkload(["worker"])
    workload.start([0, 3], 10)
    assert workload.poll() is True
    results = workload.collect()
    assert [p.killed for p in undying] == [1, 1]
    assert [p.waits for p in undying] == [[workload_module.KILL_WAIT]] * 2
    assert results[3]["burn"] == {"stage": "burn", "error": "its worker did not end after it was killed"}


def test_a_copy_worker_that_cannot_start_is_an_error_not_a_crash(tmp_path):
    copies = TorchWorkload([str(tmp_path / "no-such-python")]).copies([0, 1])
    assert copies.summary == {"stage": "p2p", "error": "its worker could not be started (FileNotFoundError)"}
    assert copies.pairs == []


def test_the_real_copy_worker_on_this_machine_never_reports_a_pair_it_did_not_copy():
    # No multi-GPU machine here: with no PyTorch, or a CPU build, the worker must say so.
    copies = TorchWorkload().copies([0, 1])
    devices = copies.summary.get("devices", 0)
    if devices < 2:
        assert copies.pairs == []
    check = multi_checks.p2p(TORCH, [{"index": "0"}, {"index": "1"}], copies, None)
    assert check.status != PASS or devices == 2

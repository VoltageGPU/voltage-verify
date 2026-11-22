# SPDX-License-Identifier: MIT
"""How each health verdict is decided. Thresholds and rules are those of hwcheck's
acceptance test (measured on an 8x H100 SXM node on 24 October 2026), replayed on its fixtures."""

from __future__ import annotations

import pytest
from health_fakes import FakeHost, burn_record, healthy, vram_record

from voltage_verify.health import checks, klog, load_checks, smi
from voltage_verify.health.checks import Expect
from voltage_verify.health.load_checks import Sample
from voltage_verify.health.report import FAIL, INFO, PASS, SKIP, WARN
from voltage_verify.health.workload import Probe

NO_EXPECT = Expect(None, None, None)
TORCH = Probe("2.14.1+cu130", True, 8, "")


def _inventory(fixture: str) -> list[dict[str, str]]:
    gpus = smi.Smi(FakeHost(fixture)).query(smi.INVENTORY_FIELDS)
    assert gpus is not None
    return gpus


def _ecc(fixture: str) -> list[dict[str, str]]:
    rows = smi.Smi(FakeHost(fixture)).query(smi.ECC_FIELDS)
    assert rows is not None
    return rows


def _samples(fixture: str, count: int = 12) -> list[Sample]:
    host = FakeHost(fixture)
    reader = smi.Smi(host)
    out = [Sample(0, "idle", row) for row in reader.query(smi.SAMPLE_FIELDS) or []]
    host.busy = True
    for i in range(count):
        out += [Sample(2000 * (i + 1), "load", row) for row in reader.query(smi.SAMPLE_FIELDS) or []]
    return out


def _by_name(found: list) -> dict:
    return {c.name: c for c in found}


# --- inventory ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("listed", "measured", "expected"),
    [
        ("RTX 3060 Ti", "NVIDIA GeForce RTX 3060 Ti", True),
        ("rtx-3060", "NVIDIA GeForce RTX 3060 Ti", True),
        ("RTX 4090", "NVIDIA GeForce RTX 3060 Ti", False),
        ("H100", "NVIDIA H100 80GB HBM3", True),
        ("H100 80GB", "NVIDIA H100 80GB HBM3", True),
        ("H200", "NVIDIA H100 80GB HBM3", False),
        ("A100", "NVIDIA H100 80GB HBM3", False),
        ("L40S", "NVIDIA L40S", True),
        ("L40", "NVIDIA L40S", False),  # whole tokens: an L40 is not an L40S
        ("Titan", "NVIDIA TITAN Xp", True),  # no digit in the listing: every word must be there
        ("Quadro", "NVIDIA TITAN Xp", False),
        ("", "NVIDIA TITAN Xp", False),
    ],
)
def test_model_matches_on_the_tokens_that_carry_a_digit(listed, measured, expected):
    assert checks.model_matches(listed, measured) is expected


def test_inventory_of_one_gpu_names_model_vram_driver_vbios_and_pci_id():
    (check,) = checks.inventory(_inventory("rtx3060ti-1x"), NO_EXPECT)
    assert (check.name, check.status) == ("inventory.gpus", PASS)
    assert check.detail == "1x NVIDIA GeForce RTX 3060 Ti, 8192 MiB, driver 610.88, vbios 94.04.6b.00.4a, PCI 29:00.0"
    assert check.data["gpus"][0]["uuid"].startswith("GPU-")


def test_inventory_shows_the_pci_device_id_when_the_driver_gives_it():
    gpus = [{**_inventory("rtx3060ti-1x")[0], "pci_device_id": "0x248910DE"}]  # as the real card answers
    (check,) = checks.inventory(gpus, NO_EXPECT)
    assert check.detail.endswith("vbios 94.04.6b.00.4a, PCI 29:00.0 (id 10de:2489)")
    garbage = [{**gpus[0], "pci_device_id": "0x2489; rm -rf"}]
    assert checks.inventory(garbage, NO_EXPECT)[0].detail.endswith("PCI 29:00.0")


def test_inventory_of_eight_gpus():
    (check,) = checks.inventory(_inventory("h100-8x"), NO_EXPECT)
    assert check.status == PASS
    assert check.detail.startswith(
        "8x NVIDIA H100 80GB HBM3, 81559 MiB each, driver 580.173.02, vbios 96.00.CF.00.01, PCI 01:00.0"
    )
    assert len(check.data["gpus"]) == 8


def test_no_gpu_fails():
    for gpus in (None, []):
        (check,) = checks.inventory(gpus, NO_EXPECT)
        assert (check.name, check.status) == ("inventory.gpus", FAIL)
        assert "no NVIDIA GPU visible" in check.detail


def test_expectations_met():
    found = _by_name(checks.inventory(_inventory("rtx3060ti-1x"), Expect("RTX 3060 Ti", 1, 8)))
    assert [found[n].status for n in ("inventory.gpus", "inventory.model", "inventory.count", "inventory.vram")] == [
        PASS
    ] * 4


def test_vram_below_the_listing_fails():
    # The classic case: a cheaper card sold under a bigger name.
    found = _by_name(checks.inventory(_inventory("rtx3060ti-1x"), Expect("RTX 4090", 1, 24)))
    assert found["inventory.vram"].status == FAIL
    assert found["inventory.vram"].detail == "8 GB per GPU (8192 MiB on the smallest), the listing says 24 GB"
    assert found["inventory.model"].status == FAIL
    assert "RTX 4090" in found["inventory.model"].detail and "RTX 3060 Ti" in found["inventory.model"].detail


def test_vram_tolerance_is_seven_percent_as_the_driver_reports_less_than_the_spec_sheet():
    gpus = [{**_inventory("h100-8x")[0], "memory_total": "143771"}]  # an H200: 140 GB for the driver
    assert _by_name(checks.inventory(gpus, Expect(None, None, 141)))["inventory.vram"].status == PASS
    assert _by_name(checks.inventory(gpus, Expect(None, None, 151)))["inventory.vram"].status == FAIL


def test_fewer_gpus_than_listed_fails():
    found = _by_name(checks.inventory(_inventory("h100-8x")[:7], Expect(None, 8, None)))
    assert found["inventory.count"].status == FAIL
    assert found["inventory.count"].detail == "7 GPUs visible, the listing says 8"


def test_mixed_models_and_mig_are_warnings():
    gpus = _inventory("h100-8x")
    mixed = [*gpus[:7], {**gpus[7], "name": "NVIDIA A100-SXM4-80GB"}]
    check = _by_name(checks.inventory(mixed, NO_EXPECT))["inventory.gpus"]
    assert check.status == WARN and "mixed models" in check.detail
    mig = [{**gpus[0], "mig_mode": "Enabled"}, *gpus[1:]]
    check = _by_name(checks.inventory(mig, NO_EXPECT))["inventory.gpus"]
    assert check.status == WARN and "MIG enabled" in check.detail
    assert _by_name(checks.inventory(mixed, Expect("H100", None, None)))["inventory.model"].status == FAIL


# --- errors.ecc --------------------------------------------------------------------------------


def test_ecc_clean():
    rows = _ecc("h100-8x")
    check = checks.ecc(rows, rows)
    assert check.status == PASS
    assert check.detail == "ECC on 8/8 GPUs, no uncorrected error, nothing pending, 0 corrected since the driver loaded"
    assert check.data["after"][0]["uncorrected_volatile"] == "0"


def test_uncorrected_ecc_errors_and_a_pending_remap_fail():
    rows = _ecc("h100-8x-sick")
    check = checks.ecc(rows, rows)
    assert check.status == FAIL
    assert "GPU 6: 2 uncorrected ECC errors since the driver loaded" in check.detail
    assert "GPU 1: row remapping pending (the GPU needs a reset)" in check.detail


def test_corrected_errors_during_the_test_and_lifetime_errors_are_warnings():
    before = _ecc("h100-8x")
    after = [{**before[0], "corrected_volatile": "3"}, {**before[1], "uncorrected_aggregate": "1"}, *before[2:]]
    check = checks.ecc(before, after)
    assert check.status == WARN
    assert "GPU 0: 3 corrected errors during the test" in check.detail
    assert "GPU 1: 1 uncorrected errors in its lifetime" in check.detail


def test_retired_pages_failed_remap_and_disabled_ecc():
    base = _ecc("h100-8x")
    cases = [
        ({"retired_pending": "Yes"}, FAIL, "page retirement pending"),
        ({"remap_failure": "Yes"}, FAIL, "row remapping failed"),
        ({"retired_dbe": "2"}, WARN, "2 pages retired after double-bit errors"),
        ({"remapped_uncorrectable": "1"}, WARN, "1 rows remapped after uncorrectable errors"),
        ({"ecc_mode": "Disabled"}, WARN, "ECC disabled"),
    ]
    for change, status, text in cases:
        rows = [{**base[0], **change}, *base[1:]]
        check = checks.ecc(rows, rows)
        assert check.status == status and text in check.detail, change


def test_a_gpu_without_ecc_is_not_judged_and_not_called_a_pass():
    rows = _ecc("rtx3060ti-1x")
    check = checks.ecc(rows, rows)
    assert check.status == INFO
    assert check.detail == "no ECC reported by these GPUs (consumer cards have none): not applicable"


def test_ecc_counters_that_cannot_be_read_are_a_skip():
    assert checks.ecc(None, None).status == SKIP


# --- errors.xid --------------------------------------------------------------------------------


def test_xid_79_during_the_test_fails():
    host = FakeHost("h100-8x-sick")
    check = checks.xid(klog.read(host), host.uptime())
    assert check.status == FAIL
    assert check.detail.startswith("during the test: Xid 79 on PCI:0000:08:00")
    assert check.data["events"][1] == {"uptime_ms": 5030000, "xid": 79, "pci": "PCI:0000:08:00", "during_test": True}
    assert check.data["source"] == "dmesg"


def test_no_xid_passes_and_says_where_the_log_came_from():
    host = FakeHost("h100-8x", kernel_log="journalctl")
    check = checks.xid(klog.read(host), host.uptime())
    assert (check.status, check.detail) == (PASS, "no Xid during the test, read with journalctl")


def test_a_hardware_xid_earlier_in_this_boot_is_a_warning_and_a_software_one_is_not():
    log = klog.KernelLog("dmesg", (klog.XidEvent(7345, 13, "PCI:0000:03:00"), klog.XidEvent(60000, 79, "PCI:0000:08:00")))
    check = checks.xid(log, 5000.0)
    assert check.status == WARN
    assert check.detail == "none during the test; earlier in this boot: Xid 79 on PCI:0000:08:00 at 60 s after boot"
    only_software = klog.KernelLog("dmesg", (klog.XidEvent(7345, 13, "PCI:0000:03:00"),))
    check = checks.xid(only_software, 5000.0)
    assert check.status == PASS and "1 earlier in this boot, not hardware errors" in check.detail


def test_another_xid_during_the_test_is_a_warning():
    log = klog.KernelLog("dmesg", (klog.XidEvent(5_010_000, 31, "PCI:0000:03:00"),))
    check = checks.xid(log, 5000.0)
    assert (check.status, check.detail) == (WARN, "during the test: Xid 31 on PCI:0000:03:00")


def test_an_unreadable_kernel_log_is_a_skip_that_says_so():
    check = checks.xid(None, 5000.0)
    assert check.status == SKIP
    assert check.detail == "kernel log not readable (not root, a container, or not Linux): Xid errors not checked"


# --- sensors -----------------------------------------------------------------------------------


def test_sensors_of_a_healthy_node_under_load():
    found = _by_name(load_checks.sensors(_samples("h100-8x"), _inventory("h100-8x"), under_load=True))
    thermal, clocks = found["sensors.thermal"], found["sensors.clocks"]
    assert thermal.status == PASS
    assert thermal.detail == (
        "max 65 C (GPU 3), max 695 W, no thermal or hardware slowdown; "
        "at the power cap in 100% of the samples under load (normal under full load)"
    )
    assert clocks.status == PASS
    assert clocks.detail == "median SM clock 1769 MHz under load, every GPU between 1755 and 1783 MHz and at least 90% busy"
    gpu0 = thermal.data["gpus"][0]
    assert gpu0 == {
        "gpu": 0,
        "samples": 13,
        "temperature_max_c": 62,
        "power_max_mw": 695300,
        "power_limit_mw": 700000,
        "power_cap_samples": 12,
        "thermal_slowdown_samples": 0,
        "hw_slowdown_samples": 0,
        "power_brake_samples": 0,
    }
    assert clocks.data["gpus"][0] == {"gpu": 0, "sm_clock_median_mhz": 1755, "utilization_max": 100}


def test_overheating_with_hardware_thermal_slowdown_fails():
    found = _by_name(load_checks.sensors(_samples("h100-8x-sick"), _inventory("h100-8x-sick"), under_load=True))
    thermal = found["sensors.thermal"]
    assert thermal.status == FAIL
    assert "GPU 5: hardware thermal slowdown or power brake in 12 of 13 samples" in thermal.detail
    assert "GPU 5 reached 88 C, the median GPU 64 C" in thermal.detail
    assert "GPU 4 power limit 500 W, under its default 700 W" in thermal.detail
    clocks = found["sensors.clocks"]
    assert clocks.status == WARN
    assert clocks.detail == "GPU 0 median SM clock 1400 MHz, under 85% of the median GPU (1769 MHz)"


def test_one_sample_of_hardware_slowdown_in_many_is_a_warning():
    samples = _samples("h100-8x", count=40)
    flagged = [
        Sample(s.t_ms, s.phase, {**s.values, "hw_thermal_slowdown": "Active"})
        if s.values["index"] == "3" and s.t_ms == 4000
        else s
        for s in samples
    ]
    thermal = _by_name(load_checks.sensors(flagged, _inventory("h100-8x"), under_load=True))["sensors.thermal"]
    assert thermal.status == WARN
    assert thermal.detail == "GPU 3: hardware thermal slowdown or power brake in 1 sample(s)"


def test_software_thermal_slowdown_and_a_gpu_that_never_got_busy_are_warnings():
    samples = [
        Sample(s.t_ms, s.phase, {**s.values, "sw_thermal_slowdown": "Active", "utilization": "20"})
        if s.values["index"] == "2"
        else s
        for s in _samples("h100-8x")
    ]
    found = _by_name(load_checks.sensors(samples, _inventory("h100-8x"), under_load=True))
    assert (
        found["sensors.thermal"].status == WARN
        and "GPU 2: thermal slowdown in 13 of 13 samples" in found["sensors.thermal"].detail
    )
    assert found["sensors.clocks"].status == WARN and "GPU 2 never above 20% busy" in found["sensors.clocks"].detail


def test_without_a_load_the_sensors_are_read_but_not_judged():
    host = FakeHost("rtx3060ti-1x")
    samples = [Sample(i * 1000, "idle", row) for i in range(3) for row in smi.Smi(host).query(smi.SAMPLE_FIELDS) or []]
    found = _by_name(load_checks.sensors(samples, _inventory("rtx3060ti-1x"), under_load=False))
    assert found["sensors.thermal"].status == INFO
    assert found["sensors.thermal"].detail == (
        "idle only, not judged under load: max 44 C (GPU 0), max 38 W, no slowdown flag active"
    )
    assert found["sensors.clocks"].status == INFO
    assert found["sensors.clocks"].detail == "idle only, not judged under load: median SM clock 1215 MHz, at most 20% busy"


def test_a_slowdown_flag_at_idle_is_still_reported():
    host = FakeHost("rtx3060ti-1x")
    host.overrides[0] = {"clocks_event_reasons.hw_thermal_slowdown": "Active"}
    samples = [Sample(i * 1000, "idle", row) for i in range(3) for row in smi.Smi(host).query(smi.SAMPLE_FIELDS) or []]
    thermal = _by_name(load_checks.sensors(samples, _inventory("rtx3060ti-1x"), under_load=False))["sensors.thermal"]
    assert thermal.status == FAIL and "hardware thermal slowdown" in thermal.detail


def test_no_sample_is_a_skip():
    found = _by_name(load_checks.sensors([], _inventory("h100-8x"), under_load=True))
    assert found["sensors.thermal"].status == SKIP and found["sensors.clocks"].status == SKIP


# --- compute.matmul ----------------------------------------------------------------------------


def _records(**per_gpu: dict) -> dict[int, dict]:
    records = {gpu: healthy() for gpu in range(8)}
    for key, stages in per_gpu.items():
        records[int(key.removeprefix("gpu"))] = stages
    return records


def test_compute_passes_when_every_gpu_computes_right_at_the_same_speed():
    check = load_checks.compute(TORCH, _inventory("h100-8x"), _records(), 60)
    assert check.status == PASS
    assert check.detail == (
        "8/8 GPUs ran 60 s of variable load, median 652 TFLOPS (bfloat16), 8960 results recomputed and compared: "
        "0 changed; exact product equal to the CPU's on every GPU (PyTorch 2.14.1+cu130)"
    )
    assert check.data["gpus"][0]["gflops"] == 652400 and check.data["gpus"][0]["mismatch"] == 0


def test_a_result_that_changes_is_a_computation_error_and_fails():
    records = _records(gpu6=healthy(burn=burn_record(mismatch=2)))
    check = load_checks.compute(TORCH, _inventory("h100-8x"), records, 60)
    assert check.status == FAIL
    assert check.detail.startswith("GPU 6: the same product changed 2 times (silent data corruption?)")


def test_a_product_that_differs_from_the_cpu_fails():
    records = _records(gpu3=healthy(cross_check={"stage": "cross_check", "wrong": 1}))
    check = load_checks.compute(TORCH, _inventory("h100-8x"), records, 60)
    assert check.status == FAIL and "GPU 3: an exact integer product differs from the CPU's result" in check.detail


def test_non_finite_results_fail():
    records = _records(gpu1=healthy(burn=burn_record(finite=False)))
    check = load_checks.compute(TORCH, _inventory("h100-8x"), records, 60)
    assert check.status == FAIL and "GPU 1: non-finite matmul results" in check.detail


def test_half_the_throughput_fails_and_under_85_percent_warns():
    slow = load_checks.compute(TORCH, _inventory("h100-8x"), _records(gpu4=healthy(burn=burn_record(gflops=301200))), 60)
    assert slow.status == FAIL and "GPU 4 at 301 TFLOPS, under half the median GPU (652)" in slow.detail
    slower = load_checks.compute(TORCH, _inventory("h100-8x"), _records(gpu4=healthy(burn=burn_record(gflops=540000))), 60)
    assert slower.status == WARN and "GPU 4 at 540 TFLOPS, under 85% of the median GPU (652)" in slower.detail


def test_a_gpu_whose_worker_did_not_finish_fails_with_the_reason():
    records = _records(
        gpu7={
            "cross_check": {"stage": "cross_check", "wrong": 0},
            "burn": {"stage": "burn", "error": "RuntimeError: CUDA error: unspecified launch failure"},
        }
    )
    check = load_checks.compute(TORCH, _inventory("h100-8x"), records, 60)
    assert check.status == FAIL
    assert "GPU 7: the load did not finish (RuntimeError: CUDA error: unspecified launch failure)" in check.detail
    del records[7]
    check = load_checks.compute(TORCH, _inventory("h100-8x"), records, 60)
    assert check.status == FAIL and "GPU 7: the load did not finish (no result from its worker)" in check.detail


def test_identical_gpus_that_compute_different_results_warn():
    records = _records(gpu2=healthy(burn=burn_record(checksum=-7654321)))
    check = load_checks.compute(TORCH, _inventory("h100-8x"), records, 60)
    assert check.status == WARN
    assert check.detail == "GPU 2 computed another result than GPU 0 (same model, same inputs)"


def test_without_pytorch_the_compute_test_is_skipped_and_explained():
    absent = load_checks.compute(
        Probe(None, False, 0, "PyTorch is not installed in this Python"), _inventory("h100-8x"), {}, 60
    )
    assert absent.status == SKIP
    assert absent.detail == (
        "PyTorch is not installed in this Python: the compute test did not run "
        "(install a CUDA build of PyTorch to run it; nothing is installed for you)"
    )
    cpu_only = load_checks.vram(
        Probe("2.3.1+cpu", False, 0, "PyTorch 2.3.1+cpu sees no CUDA device"), _inventory("h100-8x"), {}
    )
    assert cpu_only.status == SKIP and cpu_only.detail.startswith(
        "PyTorch 2.3.1+cpu sees no CUDA device: the VRAM test did not run"
    )


# --- vram.pattern ------------------------------------------------------------------------------


def test_vram_passes_on_ninety_percent_of_the_free_memory_with_no_wrong_byte():
    check = load_checks.vram(TORCH, _inventory("h100-8x"), _records())
    assert check.status == PASS
    assert check.detail == (
        "8/8 GPUs: 5 patterns written and read back on 92% of the free VRAM (72.0 GiB per GPU at least), 0 wrong bytes"
    )
    assert check.data["gpus"][0]["tested_bytes"] == 77_280_000_000


def test_a_wrong_byte_fails():
    records = _records(gpu5=healthy(vram=vram_record(wrong_bytes=4096)))
    check = load_checks.vram(TORCH, _inventory("h100-8x"), records)
    assert check.status == FAIL
    assert check.detail.startswith("GPU 5: 4096 wrong bytes read back")


def test_less_than_ninety_percent_of_the_free_vram_is_a_warning():
    records = _records(gpu0=healthy(vram=vram_record(tested_bytes=60_000_000_000)))
    check = load_checks.vram(TORCH, _inventory("h100-8x"), records)
    assert check.status == WARN
    assert check.detail == "GPU 0: only 71% of the free VRAM could be tested (90% expected)"


def test_a_vram_test_that_did_not_finish_fails():
    records = _records(gpu2={"cross_check": {"stage": "cross_check", "wrong": 0}, "burn": burn_record()})
    check = load_checks.vram(TORCH, _inventory("h100-8x"), records)
    assert check.status == FAIL and "GPU 2: the VRAM test did not finish (no result from its worker)" in check.detail


# --- topology.links ----------------------------------------------------------------------------


def test_topology_is_information_only():
    host = FakeHost("h100-8x")
    reader = smi.Smi(host)
    check = checks.topology(_inventory("h100-8x"), reader.text("topo", "-m"), reader.text("nvlink", "-s"), [])
    assert check.status == INFO
    assert check.detail == (
        "28 GPU pairs: NV18 x28; PCIe links at best Gen1 x16 (Gen5 x16 possible) on 8/8 GPUs; "
        "read only: P2P copies and NCCL are not measured here (hwcheck's acceptance.sh does)"
    )


def test_topology_of_one_gpu_and_the_best_pcie_link_seen_under_load():
    host = FakeHost("rtx3060ti-1x")
    reader = smi.Smi(host)
    samples = [Sample(0, "load", {"index": "0", "pcie_gen": "3", "pcie_width": "16"})]
    check = checks.topology(_inventory("rtx3060ti-1x"), reader.text("topo", "-m"), reader.text("nvlink", "-s"), samples)
    assert check.status == INFO
    assert check.detail.startswith("1 GPU, no GPU-to-GPU link; PCIe link at best Gen3 x16 (Gen3 x16 possible)")
    assert check.data["pcie"] == [{"gpu": 0, "gen": 3, "gen_max": 3, "width": 16, "width_max": 16}]

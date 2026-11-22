# SPDX-License-Identifier: MIT
"""Verdicts of the load: the sensors sampled while it ran, the verified compute, the VRAM
pattern test. Rules and thresholds are those of hwcheck's acceptance.sh 1.1; pure functions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .checks import Findings, Gpu, index_of, median
from .report import INFO, SKIP, HealthCheck
from .smi import as_int, as_milli
from .workload import Probe

THROUGHPUT_FAIL_PERCENT = 50  # one GPU under half the median matmul throughput
THROUGHPUT_WARN_PERCENT = 85  # ... or under 85% of it
CLOCK_WARN_PERCENT = 85  # one GPU's median SM clock under load under 85% of the median GPU
TEMPERATURE_SPREAD_C = 10  # one GPU at least 10 C hotter than the median GPU
HW_SLOWDOWN_FAIL_PERCENT = 5  # hardware thermal slowdown or power brake in 5% of the samples or more
BUSY_MIN_PERCENT = 90  # a GPU that never reached 90% utilisation did not really run the load
VRAM_COVERAGE_PERCENT = 90  # share of the free VRAM the pattern test must cover
NOT_INSTALLED = "install a CUDA build of PyTorch to run it; nothing is installed for you"
GIB = 1 << 30
Records = dict[int, dict[str, dict[str, Any]]]  # GPU index -> stage -> what its worker reported


@dataclass(frozen=True)
class Sample:
    """One nvidia-smi reading of one GPU: milliseconds since the start, phase, raw values."""

    t_ms: int
    phase: str  # "idle" or "load"
    values: dict[str, str]


def _active(samples: Sequence[Sample], key: str) -> int:
    return sum(1 for s in samples if s.values.get(key) == "Active")


def _tflops(gflops: int) -> str:
    return str(gflops // 1000) if gflops >= 10_000 else f"{gflops // 1000}.{gflops % 1000 // 100}"


def sensors(samples: Sequence[Sample], gpus: list[Gpu], *, under_load: bool) -> list[HealthCheck]:
    """sensors.thermal (temperature, power, slowdown flags) and sensors.clocks (SM clock, use)."""
    if not samples:
        skipped = "no nvidia-smi sample could be read"
        return [HealthCheck("sensors.thermal", SKIP, skipped, {}), HealthCheck("sensors.clocks", SKIP, skipped, {})]
    thermal, clocks = Findings(), Findings()
    rows_thermal, rows_clocks = [], []
    for row in gpus:
        gpu = index_of(row)
        mine = [s for s in samples if index_of(s.values) == gpu]
        loaded = [s for s in mine if s.phase == "load"]
        busy = [s for s in loaded if (as_int(s.values.get("utilization")) or 0) >= BUSY_MIN_PERCENT]
        hw_thermal, brake = _active(mine, "hw_thermal_slowdown"), _active(mine, "hw_power_brake")
        sw_thermal, hw_slow = _active(mine, "sw_thermal_slowdown"), _active(mine, "hw_slowdown")
        rows_thermal.append(
            {
                "gpu": gpu,
                "samples": len(mine),
                "temperature_max_c": max((as_int(s.values.get("temperature")) or 0 for s in mine), default=0),
                "power_max_mw": max((as_milli(s.values.get("power_draw")) or 0 for s in mine), default=0),
                "power_limit_mw": as_milli(row.get("power_limit")) or 0,
                "power_cap_samples": _active(loaded, "sw_power_cap"),
                "thermal_slowdown_samples": sw_thermal + hw_thermal,
                "hw_slowdown_samples": hw_slow,
                "power_brake_samples": brake,
            }
        )
        rows_clocks.append(
            {
                "gpu": gpu,
                "sm_clock_median_mhz": median([as_int(s.values.get("sm_clock")) or 0 for s in (busy or loaded or mine)]),
                "utilization_max": max((as_int(s.values.get("utilization")) or 0 for s in (loaded or mine)), default=0),
            }
        )
        hard = hw_thermal + brake
        if hard and hard * 100 >= max(100, len(mine) * HW_SLOWDOWN_FAIL_PERCENT):
            thermal.fail(f"GPU {gpu}: hardware thermal slowdown or power brake in {hard} of {len(mine)} samples")
        elif hard:
            thermal.warn(f"GPU {gpu}: hardware thermal slowdown or power brake in {hard} sample(s)")
        if sw_thermal:
            thermal.warn(f"GPU {gpu}: thermal slowdown in {sw_thermal} of {len(mine)} samples")
        if hw_slow and not hard:
            thermal.warn(f"GPU {gpu}: hardware slowdown in {hw_slow} of {len(mine)} samples")
        limit, default = as_milli(row.get("power_limit")), as_milli(row.get("power_default_limit"))
        if limit is not None and default is not None and limit + 1000 < default:
            thermal.warn(f"GPU {gpu} power limit {limit // 1000} W, under its default {default // 1000} W")

    hottest = max(rows_thermal, key=lambda r: r["temperature_max_c"])
    typical = median([r["temperature_max_c"] for r in rows_thermal])
    typical_clock = median([r["sm_clock_median_mhz"] for r in rows_clocks])
    for heat, clock in zip(rows_thermal, rows_clocks, strict=True):
        gpu = heat["gpu"]
        if len(gpus) > 1 and heat["temperature_max_c"] >= typical + TEMPERATURE_SPREAD_C:
            thermal.warn(f"GPU {gpu} reached {heat['temperature_max_c']} C, the median GPU {typical} C")
        if under_load and clock["utilization_max"] < BUSY_MIN_PERCENT:
            clocks.warn(f"GPU {gpu} never above {clock['utilization_max']}% busy")
        if under_load and clock["sm_clock_median_mhz"] * 100 < CLOCK_WARN_PERCENT * typical_clock:
            clocks.warn(
                f"GPU {gpu} median SM clock {clock['sm_clock_median_mhz']} MHz, "
                f"under {CLOCK_WARN_PERCENT}% of the median GPU ({typical_clock} MHz)"
            )

    watts = max(r["power_max_mw"] for r in rows_thermal) // 1000
    peak = f"max {hottest['temperature_max_c']} C (GPU {hottest['gpu']}), max {watts} W"
    speeds = [r["sm_clock_median_mhz"] for r in rows_clocks]
    data_thermal, data_clocks = {"gpus": rows_thermal}, {"gpus": rows_clocks}
    if not under_load:
        idle = "idle only, not judged under load: "
        busiest = max(r["utilization_max"] for r in rows_clocks)
        return [
            thermal.check("sensors.thermal", f"{idle}{peak}, no slowdown flag active", data_thermal, INFO),
            clocks.check(
                "sensors.clocks", f"{idle}median SM clock {typical_clock} MHz, at most {busiest}% busy", data_clocks, INFO
            ),
        ]
    capped = sum(r["power_cap_samples"] for r in rows_thermal)
    loaded_total = sum(1 for s in samples if s.phase == "load")
    cap = (
        f"; at the power cap in {100 * capped // loaded_total}% of the samples under load (normal under full load)"
        if capped and loaded_total
        else ""
    )
    return [
        thermal.check("sensors.thermal", f"{peak}, no thermal or hardware slowdown{cap}", data_thermal),
        clocks.check(
            "sensors.clocks",
            f"median SM clock {typical_clock} MHz under load, every GPU between {min(speeds)} and {max(speeds)} MHz "
            f"and at least {BUSY_MIN_PERCENT}% busy",
            data_clocks,
        ),
    ]


def _stage(records: Records, gpu: int, stage: str) -> tuple[dict[str, Any] | None, str]:
    """The record of one stage of one GPU's worker, or None and why there is none."""
    record = records.get(gpu, {}).get(stage)
    if record is None:
        return None, "no result from its worker"
    if "error" in record:
        return None, str(record["error"])
    return record, ""


def compute(probe: Probe, gpus: list[Gpu], records: Records, seconds: int) -> HealthCheck:
    """compute.matmul: the same product recomputed under a variable load must never change."""
    if not probe.usable:
        return HealthCheck("compute.matmul", SKIP, f"{probe.reason}: the compute test did not run ({NOT_INSTALLED})", {})
    found = Findings()
    rows: list[dict[str, Any]] = []
    for row in gpus:
        gpu = index_of(row)
        burn, why = _stage(records, gpu, "burn")
        if burn is None:
            found.fail(f"GPU {gpu}: the load did not finish ({why})")
            continue
        cross, why = _stage(records, gpu, "cross_check")
        if cross is None:
            found.fail(f"GPU {gpu}: the exact product against the CPU did not finish ({why})")
        elif cross.get("wrong"):
            found.fail(f"GPU {gpu}: an exact integer product differs from the CPU's result")
        if not burn.get("finite"):
            found.fail(f"GPU {gpu}: non-finite matmul results")
        if burn.get("mismatch"):
            found.fail(f"GPU {gpu}: the same product changed {burn['mismatch']} times (silent data corruption?)")
        rows.append({"gpu": gpu, "name": row.get("name", ""), **{k: v for k, v in burn.items() if k != "stage"}})
    speeds = [r["gflops"] for r in rows if r.get("gflops")]
    typical = median(speeds)
    reference: dict[str, dict[str, Any]] = {}
    for r in rows:
        speed = r.get("gflops") or 0
        if speed * 100 < THROUGHPUT_FAIL_PERCENT * typical:
            found.fail(f"GPU {r['gpu']} at {_tflops(speed)} TFLOPS, under half the median GPU ({_tflops(typical)})")
        elif speed * 100 < THROUGHPUT_WARN_PERCENT * typical:
            found.warn(
                f"GPU {r['gpu']} at {_tflops(speed)} TFLOPS, "
                f"under {THROUGHPUT_WARN_PERCENT}% of the median GPU ({_tflops(typical)})"
            )
        first = reference.setdefault(r["name"], r)
        if first.get("checksum") != r.get("checksum"):
            found.warn(f"GPU {r['gpu']} computed another result than GPU {first['gpu']} (same model, same inputs)")
    dtype = rows[0].get("dtype", "?") if rows else "?"
    compared = sum(r.get("checks") or 0 for r in rows)
    passed = (
        f"{len(rows)}/{len(gpus)} GPUs ran {seconds} s of variable load, median {_tflops(typical)} TFLOPS "
        f"({dtype}), {compared} results recomputed and compared: "
        f"0 changed; exact product equal to the CPU's on every GPU (PyTorch {probe.torch})"
    )
    return found.check("compute.matmul", passed, {"torch": probe.torch or "", "seconds": seconds, "gpus": rows})


def vram(probe: Probe, gpus: list[Gpu], records: Records) -> HealthCheck:
    """vram.pattern: patterns written then read back on the free VRAM, no wrong byte allowed."""
    if not probe.usable:
        return HealthCheck("vram.pattern", SKIP, f"{probe.reason}: the VRAM test did not run ({NOT_INSTALLED})", {})
    found = Findings()
    rows: list[dict[str, Any]] = []
    for row in gpus:
        gpu = index_of(row)
        record, why = _stage(records, gpu, "vram")
        if record is None:
            found.fail(f"GPU {gpu}: the VRAM test did not finish ({why})")
            continue
        free, tested = record.get("free_bytes") or 0, record.get("tested_bytes") or 0
        coverage = tested * 100 // free if free else 0
        rows.append({"gpu": gpu, "coverage_percent": coverage, **{k: v for k, v in record.items() if k != "stage"}})
        if record.get("wrong_bytes"):
            found.fail(f"GPU {gpu}: {record['wrong_bytes']} wrong bytes read back")
        elif coverage < VRAM_COVERAGE_PERCENT:
            found.warn(f"GPU {gpu}: only {coverage}% of the free VRAM could be tested ({VRAM_COVERAGE_PERCENT}% expected)")
    least = min((r.get("tested_bytes") or 0 for r in rows), default=0)
    tenths = (least * 10 + GIB // 2) // GIB
    passed = (
        f"{len(rows)}/{len(gpus)} GPUs: {rows[0].get('patterns') if rows else 0} patterns written and read back on "
        f"{min((r['coverage_percent'] for r in rows), default=0)}% of the free VRAM "
        f"({tenths // 10}.{tenths % 10} GiB per GPU at least), 0 wrong bytes"
    )
    return found.check("vram.pattern", passed, {"gpus": rows})

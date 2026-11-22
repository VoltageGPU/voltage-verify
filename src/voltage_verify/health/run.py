# SPDX-License-Identifier: MIT
"""Run the health check from start to end and return its report.

Order: inventory, error counters before, sensors at idle, the load on every GPU at once with
the sensors sampled while it runs, error counters and kernel log after, topology. Without a
PyTorch that sees the GPUs the load is skipped and the report says so: the result is then
INCOMPLETE, never PASS.
"""

from __future__ import annotations

import platform
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from . import checks, klog, load_checks, smi
from .checks import Expect
from .host import Host, SystemHost
from .load_checks import Sample
from .report import SKIP, HealthCheck, HealthReport
from .workload import TorchWorkload, Workload

QUICK_SECONDS = 60
DEFAULT_SECONDS = 300
MIN_SECONDS, MAX_SECONDS = 10, 3600
IDLE_SAMPLES = 3  # one second apart, when no load runs
NO_GPU = "no GPU"
Progress = Callable[[str], None]


@dataclass(frozen=True)
class Options:
    quick: bool = False
    burn_seconds: int | None = None  # None: 60 with quick, else 300
    expect_gpu: str | None = None
    expect_gpus: int | None = None
    expect_vram_gb: int | None = None

    @property
    def seconds(self) -> int:
        if self.burn_seconds is not None:
            return self.burn_seconds
        return QUICK_SECONDS if self.quick else DEFAULT_SECONDS


def _machine() -> dict[str, str]:
    return {
        "hostname": platform.node(),
        "system": platform.system(),
        "release": platform.release(),
        "arch": platform.machine(),
        "python": platform.python_version(),
    }


class _Sampler:
    """Reads the sensors of every GPU, stamps each reading with the time since the start."""

    def __init__(self, reader: smi.Smi, host: Host, started: float) -> None:
        self._reader, self._host, self._started = reader, host, started
        self.samples: list[Sample] = []
        self.fields = [key for key, _ in reader.names(smi.SAMPLE_FIELDS)]

    def take(self, phase: str) -> None:
        rows = self._reader.query(smi.SAMPLE_FIELDS) or []
        t_ms = int((self._host.clock() - self._started) * 1000)
        self.samples.extend(Sample(t_ms, phase, row) for row in rows)

    def raw(self) -> dict[str, Any]:
        """The samples as taken: one row per reading, values as nvidia-smi printed them."""
        rows = [[s.t_ms, s.phase, *(s.values.get(key, "") for key in self.fields)] for s in self.samples]
        return {"fields": ["t_ms", "phase", *self.fields], "rows": rows}


def run_health(
    options: Options,
    host: Host | None = None,
    workload: Workload | None = None,
    progress: Progress | None = None,
) -> HealthReport:
    host = host or SystemHost()
    workload = workload or TorchWorkload()
    say = progress or (lambda _text: None)
    started = host.clock()
    started_uptime = host.uptime()
    reader = smi.Smi(host)
    expect = Expect(options.expect_gpu, options.expect_gpus, options.expect_vram_gb)
    report_options = {
        "quick": options.quick,
        "burn_seconds": options.seconds,
        "expect_gpu": options.expect_gpu,
        "expect_gpus": options.expect_gpus,
        "expect_vram_gb": options.expect_vram_gb,
    }

    def finish(found: list[HealthCheck], samples: dict[str, Any]) -> HealthReport:
        return HealthReport(
            checks=found,
            created_at=datetime.fromtimestamp(started, timezone.utc),
            duration_ms=int((host.clock() - started) * 1000),
            machine=_machine(),
            options=report_options,
            samples=samples,
        )

    gpus = reader.query(smi.INVENTORY_FIELDS)
    found = checks.inventory(gpus, expect)
    if not gpus:
        skipped = ("errors.ecc", "errors.xid", "sensors.thermal", "sensors.clocks", "compute.matmul", "vram.pattern")
        found += [HealthCheck(name, SKIP, NO_GPU, {}) for name in skipped]
        return finish(found, {})

    ecc_before = reader.query(smi.ECC_FIELDS)
    sampler = _Sampler(reader, host, started)
    sampler.take("idle")
    say("looking for a PyTorch that sees the GPUs")
    probe = workload.probe()
    records: load_checks.Records = {}
    seconds = options.seconds
    if probe.usable:
        interval = 2.0 if seconds <= 120 else 5.0
        say(
            f"load test: {seconds} s of variable matmul load on {len(gpus)} GPU(s), then the VRAM pattern test; "
            f"sensors read every {int(interval)} s"
        )
        workload.start([checks.index_of(g) for g in gpus], seconds)
        try:
            while workload.poll():
                sampler.take("load")
                host.sleep(interval)
        finally:  # also on Ctrl-C: collect() stops the workers, no load is left running
            records = workload.collect()
    else:
        say(f"no load test: {probe.reason}")
        for _ in range(IDLE_SAMPLES - 1):
            host.sleep(1.0)
            sampler.take("idle")

    ecc_after = reader.query(smi.ECC_FIELDS)
    found.append(checks.ecc(ecc_before, ecc_after))
    found.append(checks.xid(klog.read(host), started_uptime))
    found += load_checks.sensors(sampler.samples, gpus, under_load=probe.usable)
    found.append(load_checks.compute(probe, gpus, records, seconds))
    found.append(load_checks.vram(probe, gpus, records))
    found.append(checks.topology(gpus, reader.text("topo", "-m"), reader.text("nvlink", "-s"), sampler.samples))
    return finish(found, sampler.raw())

# SPDX-License-Identifier: MIT
"""Verdicts that need no load: inventory against the listing, ECC and retired memory, Xid
events in the kernel log, and the topology (read only).

Rules and thresholds are those of hwcheck (hwcheck.sh and acceptance.sh 1.1). Every function
is pure: it gets what was read from the machine and returns checks, so that each verdict is
tested on fixtures without a GPU.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .klog import KernelLog
from .report import FAIL, INFO, PASS, SKIP, WARN, HealthCheck
from .smi import as_int, parse_nvlink, parse_topology

if TYPE_CHECKING:
    from .load_checks import Sample

VRAM_TOLERANCE_PERCENT = 93  # the driver reports less than the round figure of a spec sheet
# Hardware Xids in NVIDIA's Xid catalog: bus, memory, power and thermal faults.
HARDWARE_XIDS = frozenset({48, 62, 64, 74, 79, 92, 94, 95, 119, 120, 140})
ACCEPTANCE_NOTE = "read only: P2P copies and NCCL are not measured here (hwcheck's acceptance.sh does)"
Gpu = dict[str, str]


@dataclass(frozen=True)
class Expect:
    """What the listing says, when the caller passed it."""

    gpu: str | None
    gpus: int | None
    vram_gb: int | None


class Findings:
    """Failures and warnings of one check, in the order they were found."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def fail(self, text: str) -> None:
        self.failures.append(text)

    def warn(self, text: str) -> None:
        self.warnings.append(text)

    def check(self, name: str, passed: str, data: dict[str, Any], passed_status: str = PASS) -> HealthCheck:
        if self.failures:
            also = f"; also {'; '.join(self.warnings)}" if self.warnings else ""
            return HealthCheck(name, FAIL, "; ".join(self.failures) + also, data)
        if self.warnings:
            return HealthCheck(name, WARN, "; ".join(self.warnings), data)
        return HealthCheck(name, passed_status, passed, data)


def median(values: Sequence[int]) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) // 2


def index_of(row: Gpu) -> int:
    return as_int(row.get("index")) or 0


# --- inventory ---------------------------------------------------------------------------------


def _tokens(name: str) -> list[str]:
    text = re.sub(r"\(r\)|\(tm\)", "", name.lower())
    return re.sub(r"[-_/,]+", " ", text).split()


def model_matches(listed: str, measured: str) -> bool:
    """Every token of the listing that carries a digit (4090, h100, 80gb) must be a token of
    the name the driver reports; a listing without any digit must have all its words there."""
    wanted = _tokens(listed)
    with_digit = [t for t in wanted if any(c.isdigit() for c in t)]
    have = set(_tokens(measured))
    return bool(wanted) and all(t in have for t in (with_digit or wanted))


def _short_bus(bus_id: str) -> str:
    return bus_id.removeprefix("00000000:")


def _pci_id(device_id: str) -> str | None:
    """nvidia-smi prints the PCI device id as 0x248910DE: device then vendor. Here: 10de:2489."""
    match = re.fullmatch(r"0x([0-9A-Fa-f]{4})([0-9A-Fa-f]{4})", device_id)
    return f"{match.group(2)}:{match.group(1)}".lower() if match else None


def inventory(gpus: list[Gpu] | None, expect: Expect) -> list[HealthCheck]:
    """inventory.gpus, plus one check per value of the listing the caller passed."""
    if not gpus:
        return [HealthCheck("inventory.gpus", FAIL, "no NVIDIA GPU visible (nvidia-smi missing or failing)", {"gpus": []})]
    data = {"gpus": gpus}
    names = [g.get("name", "") for g in gpus]
    sizes = [as_int(g.get("memory_total")) or 0 for g in gpus]
    first, smallest = gpus[0], min(sizes)
    found = Findings()
    if len(set(names)) > 1:
        found.warn(f"{len(gpus)} GPUs of mixed models: {', '.join(sorted(set(names)))}")
    if any(g.get("mig_mode") == "Enabled" for g in gpus):
        found.warn("MIG enabled (the GPUs are split)")
    buses = [_short_bus(g.get("pci_bus_id", "?")) for g in gpus]
    devices = {_pci_id(g.get("pci_device_id", "")) for g in gpus}
    described = (
        f"{len(gpus)}x {names[0]}, {smallest} MiB{' each' if len(gpus) > 1 else ''}, driver {first.get('driver', '?')}, "
        f"vbios {first.get('vbios', '?')}, PCI {' '.join(buses[:8])}{' ...' if len(buses) > 8 else ''}"
        + (f" (id {devices.pop()})" if len(devices) == 1 and None not in devices else "")
    )
    out = [found.check("inventory.gpus", described, data)]

    if expect.gpu is not None:
        wrong = sorted({n for n in names if not model_matches(expect.gpu, n)})
        if wrong:
            detail = f'the listing says "{expect.gpu}", the driver reports {", ".join(wrong)}'
            out.append(HealthCheck("inventory.model", FAIL, detail, {"expected": expect.gpu}))
        else:
            out.append(
                HealthCheck("inventory.model", PASS, f'{names[0]}, as listed ("{expect.gpu}")', {"expected": expect.gpu})
            )
    if expect.gpus is not None:
        ok = len(gpus) >= expect.gpus
        said = "the listing says" if not ok else "as listed:" if len(gpus) == expect.gpus else "more than listed:"
        detail = f"{len(gpus)} GPUs visible, {said} {expect.gpus}"
        out.append(
            HealthCheck("inventory.count", PASS if ok else FAIL, detail, {"expected": expect.gpus, "visible": len(gpus)})
        )
    if expect.vram_gb is not None:
        measured_gb = (smallest + 512) // 1024
        ok = measured_gb * 100 >= expect.vram_gb * VRAM_TOLERANCE_PERCENT
        said = f"as listed ({expect.vram_gb} GB)" if ok else f"the listing says {expect.vram_gb} GB"
        detail = f"{measured_gb} GB per GPU ({smallest} MiB on the smallest), {said}"
        out.append(
            HealthCheck(
                "inventory.vram", PASS if ok else FAIL, detail, {"expected_gb": expect.vram_gb, "smallest_mib": smallest}
            )
        )
    return out


# --- errors ------------------------------------------------------------------------------------


def ecc(before: list[Gpu] | None, after: list[Gpu] | None) -> HealthCheck:
    """ECC counters and retired or remapped memory, read before and after the tests."""
    if before is None or after is None:
        return HealthCheck("errors.ecc", SKIP, "ECC counters could not be read from nvidia-smi", {})
    earlier = {index_of(row): row for row in before}
    found = Findings()
    enabled = corrected = 0
    for row in after:
        gpu = index_of(row)
        count = {key: as_int(row.get(key)) for key in row}
        if row.get("ecc_mode") == "Enabled":
            enabled += 1
        elif row.get("ecc_mode") == "Disabled":
            found.warn(f"GPU {gpu} has ECC disabled")
        if count.get("uncorrected_volatile"):
            found.fail(f"GPU {gpu}: {count['uncorrected_volatile']} uncorrected ECC errors since the driver loaded")
        if row.get("remap_pending") == "Yes":
            found.fail(f"GPU {gpu}: row remapping pending (the GPU needs a reset)")
        if row.get("remap_failure") == "Yes":
            found.fail(f"GPU {gpu}: row remapping failed")
        if row.get("retired_pending") == "Yes":
            found.fail(f"GPU {gpu}: page retirement pending (the GPU needs a reset)")
        now, then = count.get("corrected_volatile"), as_int(earlier.get(gpu, {}).get("corrected_volatile"))
        if now is not None and then is not None and now > then:
            found.warn(f"GPU {gpu}: {now - then} corrected errors during the test")
        if count.get("uncorrected_aggregate"):
            found.warn(f"GPU {gpu}: {count['uncorrected_aggregate']} uncorrected errors in its lifetime")
        if count.get("retired_dbe"):
            found.warn(f"GPU {gpu}: {count['retired_dbe']} pages retired after double-bit errors")
        if count.get("remapped_uncorrectable"):
            found.warn(f"GPU {gpu}: {count['remapped_uncorrectable']} rows remapped after uncorrectable errors")
        corrected += now or 0
    data = {"before": before, "after": after}
    if not enabled and not found.failures and not found.warnings:
        detail = "no ECC reported by these GPUs (consumer cards have none): not applicable"
        return HealthCheck("errors.ecc", INFO, detail, data)
    passed = (
        f"ECC on {enabled}/{len(after)} GPUs, no uncorrected error, nothing pending, "
        f"{corrected} corrected since the driver loaded"
    )
    return found.check("errors.ecc", passed, data)


def xid(log: KernelLog | None, started_uptime: float | None) -> HealthCheck:
    """NVIDIA Xid events: a hardware one during the test fails, earlier in this boot it warns."""
    if log is None:
        detail = "kernel log not readable (not root, a container, or not Linux): Xid errors not checked"
        return HealthCheck("errors.xid", SKIP, detail, {})
    start_ms = int(started_uptime * 1000) if started_uptime is not None else None
    hard: list[str] = []
    soft: list[str] = []
    earlier: list[str] = []
    other = 0
    seen: set[tuple[str, bool]] = set()
    events = []
    for event in log.events:
        during = start_ms is not None and event.uptime_ms >= start_ms
        events.append({"uptime_ms": event.uptime_ms, "xid": event.xid, "pci": event.pci, "during_test": during})
        item = f"Xid {event.xid} on {event.pci}"
        if (item, during) in seen:
            continue
        seen.add((item, during))
        hardware = event.xid in HARDWARE_XIDS
        if during:
            (hard if hardware else soft).append(item)
        elif hardware:
            earlier.append(f"{item} at {event.uptime_ms // 1000} s after boot")
        else:
            other += 1
    data = {"source": log.source, "events": events}
    before = f"; earlier in this boot: {', '.join(earlier)}" if earlier else ""
    if hard:
        return HealthCheck("errors.xid", FAIL, f"during the test: {', '.join(hard + soft)}{before}", data)
    if soft:
        return HealthCheck("errors.xid", WARN, f"during the test: {', '.join(soft)}{before}", data)
    if earlier:
        return HealthCheck("errors.xid", WARN, f"none during the test{before}", data)
    note = f" ({other} earlier in this boot, not hardware errors)" if other else ""
    return HealthCheck("errors.xid", PASS, f"no Xid during the test{note}, read with {log.source}", data)


# --- topology ----------------------------------------------------------------------------------


def _link(gen: int, width: int) -> str:
    return f"Gen{gen} x{width}"


def topology(gpus: list[Gpu], topo_text: str | None, nvlink_text: str | None, samples: Sequence[Sample]) -> HealthCheck:
    """What connects the GPUs, and each PCIe link at its best. Information only in this version."""
    topo = parse_topology(topo_text or "")
    nvlink = parse_nvlink(nvlink_text or "")
    pcie = []
    for row in gpus:
        gpu = index_of(row)
        seen = [row, *(s.values for s in samples if index_of(s.values) == gpu)]
        pcie.append(
            {
                "gpu": gpu,
                "gen": max((as_int(r.get("pcie_gen")) or 0 for r in seen), default=0),
                "gen_max": as_int(row.get("pcie_gen_max")) or 0,
                "width": max((as_int(r.get("pcie_width")) or 0 for r in seen), default=0),
                "width_max": as_int(row.get("pcie_width_max")) or 0,
            }
        )
    parts = []
    if len(gpus) == 1:
        parts.append("1 GPU, no GPU-to-GPU link")
    elif topo["pairs"]:
        parts.append(f"{topo['pairs']} GPU pairs: " + ", ".join(f"{kind} x{n}" for kind, n in topo["kinds"].items()))
    else:
        parts.append(f"{len(gpus)} GPUs, topology matrix not readable")
    if nvlink["active"] or nvlink["inactive"]:
        parts.append(f"NVLink: {nvlink['active']} links active, {nvlink['inactive']} inactive")
    shapes = [(p["gen"], p["width"], p["gen_max"], p["width_max"]) for p in pcie]
    common = max(set(shapes), key=shapes.count)
    same = shapes.count(common)
    many = len(gpus) > 1
    text = f"PCIe link{'s' if many else ''} at best {_link(common[0], common[1])} ({_link(common[2], common[3])} possible)"
    parts.append(text + (f" on {same}/{len(gpus)} GPUs" if many else ""))
    for p, shape in zip(pcie, shapes, strict=True):
        if shape != common:
            parts.append(f"GPU {p['gpu']} at best {_link(shape[0], shape[1])} ({_link(shape[2], shape[3])} possible)")
    parts.append(ACCEPTANCE_NOTE)
    return HealthCheck("topology.links", INFO, "; ".join(parts), {"topology": topo, "nvlink": nvlink, "pcie": pcie})

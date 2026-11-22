# SPDX-License-Identifier: MIT
"""Read nvidia-smi: the fields of ``--query-gpu``, the topology matrix, the NVLink status.

nvidia-smi fails a whole query on one field its driver does not know, and field names changed
over the years (``clocks_throttle_reasons`` became ``clocks_event_reasons``): the fields this
driver lists (``--help-query-gpu``) are read first and only those are asked for. What comes
back is text from outside this program: every value is cut and cleaned before it is used.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from .host import Host

TIMEOUT = 30.0  # seconds, per nvidia-smi call
MAX_VALUE_CHARS = 128
MAX_GPUS = 1024

# (our key, the driver's names for it, newest first)
Spec = tuple[str, tuple[str, ...]]


def _specs(*items: str) -> tuple[Spec, ...]:
    return tuple((key, tuple(names.split("|"))) for key, names in (item.split("=", 1) for item in items))


INVENTORY_FIELDS = _specs(
    "index=index",
    "name=name",
    "uuid=uuid",
    "serial=serial",
    "pci_bus_id=pci.bus_id",
    "pci_device_id=pci.device_id",
    "pci_sub_device_id=pci.sub_device_id",
    "vbios=vbios_version",
    "driver=driver_version",
    "memory_total=memory.total",
    "memory_used=memory.used",
    "ecc_mode=ecc.mode.current",
    "pcie_gen=pcie.link.gen.current",
    "pcie_gen_max=pcie.link.gen.max",
    "pcie_width=pcie.link.width.current",
    "pcie_width_max=pcie.link.width.max",
    "power_limit=power.limit",
    "power_default_limit=power.default_limit",
    "mig_mode=mig.mode.current",
)
ECC_FIELDS = _specs(
    "index=index",
    "ecc_mode=ecc.mode.current",
    "corrected_volatile=ecc.errors.corrected.volatile.total",
    "uncorrected_volatile=ecc.errors.uncorrected.volatile.total",
    "corrected_aggregate=ecc.errors.corrected.aggregate.total",
    "uncorrected_aggregate=ecc.errors.uncorrected.aggregate.total",
    "retired_sbe=retired_pages.single_bit_ecc.count|retired_pages.sbe",
    "retired_dbe=retired_pages.double_bit.count|retired_pages.dbe",
    "retired_pending=retired_pages.pending",
    "remapped_correctable=remapped_rows.correctable",
    "remapped_uncorrectable=remapped_rows.uncorrectable",
    "remap_pending=remapped_rows.pending",
    "remap_failure=remapped_rows.failure",
)
SAMPLE_FIELDS = _specs(
    "index=index",
    "temperature=temperature.gpu",
    "temperature_memory=temperature.memory",
    "sm_clock=clocks.current.sm|clocks.sm",
    "sm_clock_max=clocks.max.sm",
    "power_draw=power.draw|power.draw.average",
    "power_limit=power.limit",
    "utilization=utilization.gpu",
    "pcie_gen=pcie.link.gen.current",
    "pcie_width=pcie.link.width.current",
    "sw_power_cap=clocks_event_reasons.sw_power_cap|clocks_throttle_reasons.sw_power_cap",
    "hw_slowdown=clocks_event_reasons.hw_slowdown|clocks_throttle_reasons.hw_slowdown",
    "hw_thermal_slowdown=clocks_event_reasons.hw_thermal_slowdown|clocks_throttle_reasons.hw_thermal_slowdown",
    "hw_power_brake=clocks_event_reasons.hw_power_brake_slowdown|clocks_throttle_reasons.hw_power_brake_slowdown",
    "sw_thermal_slowdown=clocks_event_reasons.sw_thermal_slowdown|clocks_throttle_reasons.sw_thermal_slowdown",
)

_FIELD_NAME_RE = re.compile(r'"([a-z0-9_.]+)"')
_INT_RE = re.compile(r"[0-9]{1,15}")
_DECIMAL_RE = re.compile(r"([0-9]{1,12})(?:\.([0-9]{1,6}))?")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_PCIE_KINDS = ("SYS", "NODE", "PHB", "PXB", "PIX")


def clean(value: str) -> str:
    """One value of a tool's output: escape sequences and control characters out, cut to size."""
    text = _ANSI_RE.sub("", value)
    return "".join(c for c in text if c.isprintable()).strip()[:MAX_VALUE_CHARS]


def as_int(value: str | None) -> int | None:
    """A non-negative whole number as nvidia-smi prints it; None for [N/A] and anything else."""
    return int(value) if value is not None and _INT_RE.fullmatch(value) else None


def as_milli(value: str | None) -> int | None:
    """A decimal such as ``695.30`` in thousandths (695300), so that no float enters a report."""
    match = _DECIMAL_RE.fullmatch(value) if value is not None else None
    if match is None:
        return None
    return int(match.group(1)) * 1000 + int((match.group(2) or "").ljust(3, "0")[:3])


class Smi:
    """nvidia-smi on one host. ``supported`` is read once."""

    def __init__(self, host: Host) -> None:
        self._host = host
        self._supported: frozenset[str] | None = None
        self._asked = False

    def text(self, *args: str) -> str | None:
        done = self._host.run(["nvidia-smi", *args], TIMEOUT)
        return done.stdout if done is not None and done.returncode == 0 else None

    def supported(self) -> frozenset[str]:
        """The field names this driver lists. Empty when it lists none: every field is then tried."""
        if not self._asked:
            self._asked = True
            self._supported = frozenset(_FIELD_NAME_RE.findall(self.text("--help-query-gpu") or ""))
        return self._supported or frozenset()

    def names(self, specs: Sequence[Spec]) -> list[tuple[str, str]]:
        """(our key, the name to ask for) for every field this driver knows."""
        known = self.supported()
        chosen = []
        for key, aliases in specs:
            name = next((a for a in aliases if not known or a in known), None)
            if name is not None:
                chosen.append((key, name))
        return chosen

    def query(self, specs: Sequence[Spec]) -> list[dict[str, str]] | None:
        """One dict per GPU, by our keys. None when nvidia-smi is missing or the query fails."""
        chosen = self.names(specs)
        if not chosen or chosen[0][0] != "index":
            return None
        out = self.text("--query-gpu=" + ",".join(name for _, name in chosen), "--format=csv,noheader,nounits")
        if out is None:
            return None
        rows = []
        for line in out.splitlines()[:MAX_GPUS]:
            values = [clean(v) for v in line.split(",")]
            if len(values) != len(chosen) or as_int(values[0]) is None:
                continue  # an error message, or a line this parser cannot place: never guessed
            rows.append({key: value for (key, _), value in zip(chosen, values, strict=True)})
        return rows


def parse_topology(text: str) -> dict[str, Any]:
    """``nvidia-smi topo -m``: how each pair of GPUs is connected (NV18, SYS, NODE, PIX...)."""
    kinds: dict[str, int] = {}
    gpus = 0
    for raw in text.splitlines():
        cells = [clean(c) for c in raw.split("\t")]
        match = re.fullmatch(r"GPU([0-9]{1,4})", cells[0]) if cells else None
        if match is None:
            continue
        gpus += 1
        row = int(match.group(1))
        for col, cell in enumerate(cells[1:]):
            if col > row and (re.fullmatch(r"NV[0-9]{1,3}", cell) or cell in _PCIE_KINDS):
                kinds[cell] = kinds.get(cell, 0) + 1
    return {"gpus": gpus, "pairs": sum(kinds.values()), "kinds": dict(sorted(kinds.items()))}


def parse_nvlink(text: str) -> dict[str, int]:
    """``nvidia-smi nvlink -s``: links that carry a speed, and links reported inactive."""
    active = inactive = 0
    for line in text.splitlines():
        match = re.fullmatch(r"\s*Link [0-9]{1,3}: (.{1,40})", clean(line))
        if match is None:
            continue
        if "inactive" in match.group(1).lower():
            inactive += 1
        elif re.match(r"[0-9.]+ GB/s", match.group(1)):
            active += 1
    return {"active": active, "inactive": inactive}

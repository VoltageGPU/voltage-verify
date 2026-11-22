# SPDX-License-Identifier: MIT
"""Verdict of the copies between GPUs (``multi.p2p``), on machines with two GPUs or more.

Rules and thresholds are those of the "P2P copies" line of hwcheck's acceptance.sh 1.1, with
one difference: a wrong byte fails here whatever links the GPUs (acceptance.sh only lists the
corrupted pairs of a machine without NVLink). NCCL is not run here. Pure functions. The report
holds no float: a throughput is kept in MiB/s, as the worker counted it, and shown in GB/s
(10**9 bytes per second), the unit of the NVLink line rates.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from .checks import Findings, Gpu, index_of, median
from .load_checks import NOT_INSTALLED
from .report import FAIL, INFO, PASS, SKIP, HealthCheck
from .smi import parse_topology
from .workload import PeerCopies, Probe

NAME = "multi.p2p"
P2P_FAIL_PERCENT = 25  # a pair under 25% of the NVLink line rate is not really on NVLink
P2P_WARN_PERCENT = 75  # median pair under 75% of the line rate (measured: 86% on 8x H100 SXM)
P2P_SPREAD_PERCENT = 90  # one pair under 90% of the median pair (measured: the slowest at 99%)
MAX_NAMED = 6  # pairs named in a line; the data of the check holds every pair
GB = 10**9
MIB = 1 << 20
# NVLink line rate per direction in GB/s, by model: half the two-way figure of NVIDIA's
# datasheets (acceptance.sh, profile()). A name with one of _NOT_SXM has no NVLink mesh.
_NOT_SXM = ("pcie", "nvl", "gh200", "gb200", "gb300")
_LINE_RATES = (
    (r"h100.*(hbm3|sxm)", "H100 SXM", 450),
    (r"h200", "H200 SXM", 450),
    (r"a100.*sxm", "A100 SXM", 300),
    (r"b200", "B200 SXM", 900),
    (r"b300", "B300 SXM", 900),
)
# Without a complete NVLink mesh nor an SXM model, the speed is measured and not judged: pairs
# over PCIe legitimately differ (same switch, same socket, other socket).
_NOT_JUDGED = {
    "none": "(no NVLink mesh: copies over PCIe, speed not judged)",
    "partial": "(NVLink between some pairs only: speed not judged)",
    "unknown": "(nvidia-smi topo -m not readable: speed not judged)",
    "full": "",
}
Pair = dict[str, Any]


def line_rate(name: str) -> tuple[str, int]:
    """(label, GB/s per direction) of the NVLink of this model, ("", 0) when none is known."""
    text = name.lower()
    if any(word in text for word in _NOT_SXM):
        return "", 0
    for pattern, label, rate in _LINE_RATES:
        if re.search(pattern, text):
            return label, rate
    return "", 0


def _mesh(topo: dict[str, Any]) -> str:
    """How ``nvidia-smi topo -m`` links the GPU pairs: every pair on NVLink, some, or none."""
    on_nvlink = sum(count for kind, count in topo["kinds"].items() if kind.startswith("NV"))
    if not topo["pairs"]:
        return "unknown"
    if on_nvlink == topo["pairs"]:
        return "full"
    return "partial" if on_nvlink else "none"


def _gb(mib_per_s: int) -> str:
    rate = mib_per_s * MIB
    if rate >= 100 * GB:
        return str((rate + GB // 2) // GB)
    tenths = (rate * 10 + GB // 2) // GB
    return f"{tenths // 10}.{tenths % 10}"


def _named(pairs: Sequence[tuple[int, int]]) -> str:
    names = ", ".join(f"GPU {src}>{dst}" for src, dst in pairs[:MAX_NAMED])
    return names + (f" and {len(pairs) - MAX_NAMED} more" if len(pairs) > MAX_NAMED else "")


def _slow(pairs: Sequence[Pair]) -> str:
    fastest = _gb(max(p["mib_per_s"] for p in pairs))
    return f"{_named([(p['src'], p['dst']) for p in pairs])} at {fastest} GB/s{' or less' if len(pairs) > 1 else ''}"


def _measured(pair: Pair) -> bool:
    return type(pair.get("mib_per_s")) is int and pair["mib_per_s"] > 0 and type(pair.get("wrong_bytes")) is int


def _judge_arrival(
    found: Findings, pairs: Sequence[Pair], measured: Sequence[Pair], expected: Sequence[tuple[int, int]], error: Any
) -> None:
    """What fails whatever links the GPUs: a test that did not finish, a byte that did not
    arrive as sent, a pair without a result."""
    if error is not None:
        found.fail(f"the copy test did not finish ({error}): {len(measured)} of {len(expected)} pairs measured")
    corrupted = [p for p in pairs if type(p.get("wrong_bytes")) is int and p["wrong_bytes"] != 0]
    if corrupted:
        named = [
            f"GPU {p['src']}>{p['dst']} ({p['wrong_bytes']} wrong byte{'' if p['wrong_bytes'] == 1 else 's'})"
            for p in corrupted[:MAX_NAMED]
        ]
        more = f" and {len(corrupted) - MAX_NAMED} more pairs" if len(corrupted) > MAX_NAMED else ""
        found.fail(f"bytes differ after the copy on {', '.join(named)}{more}")
    done = {(p["src"], p["dst"]) for p in measured}
    missing = [key for key in expected if key not in done]
    if error is None and missing:
        found.fail(f"no result for {len(missing)} of {len(expected)} pairs ({_named(missing)})")


def _judge_speed(found: Findings, measured: Sequence[Pair], mesh: str, label: str, line: int, data: dict[str, Any]) -> str:
    """The speed rules of acceptance.sh, on a machine whose GPUs are on NVLink. Returns what
    the pairs were judged against, for the line of a pass."""
    typical = data["median_mib_per_s"]
    if mesh == "full":
        blind = [(p["src"], p["dst"]) for p in measured if p.get("peer") is False]
        if blind:
            found.fail(f"no peer access on {_named(blind)}")
    judged_on = "(no line rate known for this model: each pair is judged against the median)"
    if line:
        floor = line * GB * P2P_FAIL_PERCENT // (100 * MIB)
        low = line * GB * P2P_WARN_PERCENT // (100 * MIB)
        data["fail_under_mib_per_s"], data["warn_median_under_mib_per_s"] = floor, low
        judged_on = f"(FAIL under {_gb(floor)}, WARN under {_gb(low)}: {label} line rate {line} GB/s per direction)"
        off = [p for p in measured if p["mib_per_s"] < floor]
        if off:
            share = f"{P2P_FAIL_PERCENT}% of the {line} GB/s line rate of {label}"
            found.fail(f"{_slow(off)}, under {_gb(floor)} GB/s ({share}): not on NVLink")
        elif typical < low:
            share = f"{P2P_WARN_PERCENT}% of the {label} line rate"
            found.warn(f"median pair {_gb(typical)} GB/s, under {_gb(low)} GB/s ({share})")
    else:
        off = [p for p in measured if p["mib_per_s"] * 100 < P2P_FAIL_PERCENT * typical]
        if off:
            found.fail(f"{_slow(off)}, under {P2P_FAIL_PERCENT}% of the median pair ({_gb(typical)} GB/s)")
    if not found.failures and not found.warnings:
        uneven = [p for p in measured if p["mib_per_s"] * 100 < P2P_SPREAD_PERCENT * typical]
        if uneven:
            found.warn(f"{_slow(uneven)}, under {P2P_SPREAD_PERCENT}% of the median pair ({_gb(typical)} GB/s)")
    return judged_on


def p2p(probe: Probe, gpus: list[Gpu], copies: PeerCopies | None, topo_text: str | None) -> HealthCheck:
    """multi.p2p: a buffer copied from every GPU to every other one must arrive as sent, and,
    between GPUs on NVLink, at the speed of NVLink."""
    if not probe.usable:
        return HealthCheck(NAME, SKIP, f"{probe.reason}: the GPU-to-GPU copy test did not run ({NOT_INSTALLED})", {})
    summary = copies.summary if copies is not None else {"error": "no result from its worker"}
    reported = copies.pairs if copies is not None else []
    indices = [index_of(row) for row in gpus]
    expected = [(src, dst) for src in indices for dst in indices if src != dst]
    label, line = line_rate(gpus[0].get("name", "")) if gpus else ("", 0)
    mesh = _mesh(parse_topology(topo_text or ""))
    devices = summary.get("devices", 0)
    data: dict[str, Any] = {
        "torch": probe.torch or "",
        "gpus": len(gpus),
        "devices": devices,
        "bytes": summary.get("bytes", 0),
        "iterations": summary.get("iterations", 0),
        "nvlink_mesh": mesh,
        "model": label,
        "line_rate_gb_per_s": line,
        "pairs": [],
    }
    error = summary.get("error")
    if error is None and devices != len(gpus):
        # The worker numbers the GPUs it sees: with one missing, its pairs name other GPUs.
        detail = f"PyTorch initialises {devices} of the {len(gpus)} GPUs nvidia-smi lists: no pair is judged"
        return HealthCheck(NAME, FAIL, detail, data)

    by_pair = {(p.get("src"), p.get("dst")): p for p in reported}
    pairs = [by_pair[key] for key in expected if key in by_pair]
    measured = [p for p in pairs if _measured(p)]
    data["pairs"] = pairs
    found = Findings()
    _judge_arrival(found, pairs, measured, expected, error)

    judged, judged_on, speed = False, "", ""
    if error is None and measured:
        slowest = min(measured, key=lambda p: (p["mib_per_s"], p["src"], p["dst"]))
        typical = median([p["mib_per_s"] for p in measured])
        data["slowest_mib_per_s"], data["median_mib_per_s"] = slowest["mib_per_s"], typical
        speed = (
            f"; slowest pair {_gb(slowest['mib_per_s'])} GB/s (GPU {slowest['src']}>{slowest['dst']}), "
            f"median {_gb(typical)} GB/s "
        )
        judged = mesh == "full" or line > 0
        judged_on = _NOT_JUDGED[mesh]
        if judged:
            judged_on = _judge_speed(found, measured, mesh, label, line, data)
    passed = (
        f"{len(measured)}/{len(expected)} GPU pairs: {data['bytes'] // MIB} MiB copied {data['iterations']} times "
        f"from each GPU to every other one, every byte as sent{speed}{judged_on}"
    )
    return found.check(NAME, passed, data, PASS if judged else INFO)

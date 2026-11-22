# SPDX-License-Identifier: MIT
"""Verdict of the copies between GPUs (``multi.p2p``), on machines with two GPUs or more.

Rules and thresholds are those of the "P2P copies" line of hwcheck's acceptance.sh 1.1, with
three differences. A wrong byte fails here whatever links the GPUs (acceptance.sh only lists
the corrupted pairs of a machine without NVLink). A pair is failed for its speed only when
``nvidia-smi topo -m`` shows every pair on NVLink: an NVLink model whose mesh is not seen (part
of a board in a VM) is warned, since a failure here may end up under an attestation. And
without NVLink, where acceptance.sh judges nothing, a pair almost at a standstill is warned.
NCCL is not run here.

Pure functions. What the worker reported is not trusted: a pair counts only with one line, of
the right types, within bounds. The report holds no float: a throughput is kept in MiB/s, as
the worker counted it, and shown in GB/s (10**9 bytes per second), the unit of the NVLink line
rates.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from .checks import Findings, Gpu, index_of, median
from .load_checks import NOT_INSTALLED
from .report import FAIL, INFO, PASS, SKIP, HealthCheck
from .smi import parse_topology
from .worker import P2P_BYTES, P2P_WARMUP
from .workload import PeerCopies, Probe

NAME = "multi.p2p"
P2P_FAIL_PERCENT = 25  # a pair under 25% of the NVLink line rate is not really on NVLink
P2P_WARN_PERCENT = 75  # median pair under 75% of the line rate (measured: 86% on 8x H100 SXM)
P2P_SPREAD_PERCENT = 90  # one pair under 90% of the median pair (measured: the slowest at 99%)
P2P_STALL_PERCENT = 10  # without NVLink, where pairs legitimately differ: a pair under 10% of the median
MAX_MIB_PER_S = 10**9  # a petabyte per second: no link carries it, such a figure is not a measure
MAX_WRONG_BYTES = P2P_BYTES * (P2P_WARMUP + 1)  # every byte of every copy the worker compares
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
# Without a complete NVLink mesh nor an NVLink model, the speed is measured and not judged:
# pairs over PCIe legitimately differ (same switch, same socket, other socket).
_NOT_JUDGED = {
    "none": "no NVLink mesh: copies over PCIe",
    "partial": "NVLink between some pairs only",
    "unknown": "nvidia-smi topo -m not readable",
}
_MESH_SEEN = {
    "none": "nvidia-smi topo -m shows no NVLink between these GPUs",
    "partial": "nvidia-smi topo -m shows NVLink between some pairs only",
    "unknown": "nvidia-smi topo -m could not be read",
}
Pair = dict[str, Any]
Key = tuple[int, int]


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


def _named(pairs: Sequence[Key]) -> str:
    names = ", ".join(f"GPU {src}>{dst}" for src, dst in pairs[:MAX_NAMED])
    return names + (f" and {len(pairs) - MAX_NAMED} more" if len(pairs) > MAX_NAMED else "")


def _slow(pairs: Sequence[Pair]) -> str:
    fastest = _gb(max(p["mib_per_s"] for p in pairs))
    return f"{_named([(p['src'], p['dst']) for p in pairs])} at {fastest} GB/s{' or less' if len(pairs) > 1 else ''}"


def _count(summary: dict[str, Any], key: str) -> int:
    value = summary.get(key)
    return value if type(value) is int else 0


def _valid(pair: Pair) -> bool:
    """A measure: a throughput and a count of wrong bytes, both integers within bounds."""
    speed, wrong = pair.get("mib_per_s"), pair.get("wrong_bytes")
    return type(speed) is int and type(wrong) is int and 0 < speed <= MAX_MIB_PER_S and 0 <= wrong <= MAX_WRONG_BYTES


def _judge_arrival(
    found: Findings, groups: dict[Any, list[Pair]], measured: Sequence[Pair], expected: Sequence[Key], summary: Pair
) -> None:
    """What fails whatever links the GPUs: a test that did not finish, a closing record that
    does not add up, a pair reported twice, a byte that did not arrive as sent, a pair without
    a valid result."""
    error = summary.get("error")
    # A count that is missing, or not an integer, is no count: it reads "?" and is never valid.
    keys = ("bytes", "warmup", "iterations", "pairs")
    counts: dict[str, int] = {key: summary[key] if type(summary.get(key)) is int else -1 for key in keys}
    said = {key: str(summary[key]) if type(summary.get(key)) is int else "?" for key in keys}
    if error is not None:
        found.fail(f"the copy test did not finish ({error}): {len(measured)} of {len(expected)} pairs measured")
    elif counts["bytes"] <= 0 or counts["iterations"] <= 0 or counts["warmup"] < 0 or counts["pairs"] != len(expected):
        found.fail(
            f"the closing record of the copy worker is not valid ({said['bytes']} bytes, {said['warmup']} warm-up "
            f"and {said['iterations']} timed copies, {said['pairs']} pairs for {len(expected)} expected)"
        )
    repeated = [key for key in expected if len(groups.get(key, [])) > 1]
    if repeated:  # a second line could hide what the first one said
        found.fail(f"reported twice: {_named(repeated)}")
    corrupted = [p for p in measured if p["wrong_bytes"]]
    if corrupted:
        named = [
            f"GPU {p['src']}>{p['dst']} ({p['wrong_bytes']} wrong byte{'' if p['wrong_bytes'] == 1 else 's'})"
            for p in corrupted[:MAX_NAMED]
        ]
        more = f" and {len(corrupted) - MAX_NAMED} more pairs" if len(corrupted) > MAX_NAMED else ""
        found.fail(f"bytes differ after the copy on {', '.join(named)}{more}")
    done = {(p["src"], p["dst"]) for p in measured} | set(repeated)
    missing = [key for key in expected if key not in done]
    if error is None and missing:
        found.fail(f"no valid result for {len(missing)} of {len(expected)} pairs ({_named(missing)})")


def _against_line_rate(
    found: Findings, measured: Sequence[Pair], typical: int, mesh: str, label: str, line: int, data: dict[str, Any]
) -> str:
    """A model whose NVLink line rate is known. A slow pair fails when the topology shows
    every pair on NVLink, and is a warning otherwise: the GPUs may be part of a board, in a VM
    that has no NVLink between them."""
    floor = line * GB * P2P_FAIL_PERCENT // (100 * MIB)
    low = line * GB * P2P_WARN_PERCENT // (100 * MIB)
    data["fail_under_mib_per_s"], data["warn_median_under_mib_per_s"] = floor, low
    off = [p for p in measured if p["mib_per_s"] < floor]
    under = (
        f"{_slow(off)}, under {_gb(floor)} GB/s ({P2P_FAIL_PERCENT}% of the {line} GB/s line rate of {label})" if off else ""
    )
    rate = f"{label} line rate {line} GB/s per direction"
    if off and mesh == "full":
        found.fail(f"{under}: not on NVLink")
    elif off:
        found.warn(f"{under}: {label} is an NVLink model, but {_MESH_SEEN[mesh]}: the speed is reported, not failed")
    elif typical < low:
        share = f"{P2P_WARN_PERCENT}% of the {label} line rate"
        found.warn(f"median pair {_gb(typical)} GB/s, under {_gb(low)} GB/s ({share})")
    if mesh == "full":
        return f"(FAIL under {_gb(floor)}, WARN under {_gb(low)}: {rate})"
    return (
        f"(WARN under {_gb(floor)} for a pair, under {_gb(low)} for the median: {rate}; "
        f"never FAIL on speed: {_MESH_SEEN[mesh]})"
    )


def _judge_speed(found: Findings, measured: Sequence[Pair], mesh: str, label: str, line: int, data: dict[str, Any]) -> str:
    """The speed rules. Returns the end of the line of a pass: the speeds, and what they were
    judged against."""
    slowest = min(measured, key=lambda p: (p["mib_per_s"], p["src"], p["dst"]))
    typical = median([p["mib_per_s"] for p in measured])
    data["slowest_mib_per_s"], data["median_mib_per_s"] = slowest["mib_per_s"], typical
    speed = (
        f"; slowest pair {_gb(slowest['mib_per_s'])} GB/s (GPU {slowest['src']}>{slowest['dst']}), "
        f"median {_gb(typical)} GB/s "
    )
    if not line and mesh != "full":  # not judged, short of a pair almost at a standstill
        stalled = [p for p in measured if p["mib_per_s"] * 100 < P2P_STALL_PERCENT * typical]
        if stalled:
            found.warn(
                f"{_slow(stalled)}, under {P2P_STALL_PERCENT}% of the median pair ({_gb(typical)} GB/s); "
                f"{_NOT_JUDGED[mesh]}, speed not judged above that"
            )
        return f"{speed}({_NOT_JUDGED[mesh]}, speed not judged; WARN under {P2P_STALL_PERCENT}% of the median pair)"
    if mesh == "full":
        blind = [(p["src"], p["dst"]) for p in measured if p.get("peer") is False]
        if blind:
            found.fail(f"no peer access on {_named(blind)}")
    if line:
        judged_on = _against_line_rate(found, measured, typical, mesh, label, line, data)
    else:
        judged_on = "(no line rate known for this model: each pair is judged against the median)"
        off = [p for p in measured if p["mib_per_s"] * 100 < P2P_FAIL_PERCENT * typical]
        if off:
            found.fail(f"{_slow(off)}, under {P2P_FAIL_PERCENT}% of the median pair ({_gb(typical)} GB/s)")
    if not found.failures and not found.warnings:
        uneven = [p for p in measured if p["mib_per_s"] * 100 < P2P_SPREAD_PERCENT * typical]
        if uneven:
            found.warn(f"{_slow(uneven)}, under {P2P_SPREAD_PERCENT}% of the median pair ({_gb(typical)} GB/s)")
    return speed + judged_on


def p2p(probe: Probe, gpus: list[Gpu], copies: PeerCopies | None, topo_text: str | None) -> HealthCheck:
    """multi.p2p: a buffer copied from every GPU to every other one must arrive as sent, and,
    between GPUs on NVLink, at the speed of NVLink."""
    if not probe.usable:
        return HealthCheck(NAME, SKIP, f"{probe.reason}: the GPU-to-GPU copy test did not run ({NOT_INSTALLED})", {})
    summary = copies.summary if copies is not None else {"error": "no result from its worker"}
    ended = copies.ended if copies is not None else ""
    indices = [index_of(row) for row in gpus]
    expected = [(src, dst) for src in indices for dst in indices if src != dst]
    label, line = line_rate(gpus[0].get("name", "")) if gpus else ("", 0)
    mesh = _mesh(parse_topology(topo_text or ""))
    data: dict[str, Any] = {
        "torch": probe.torch or "",
        "gpus": len(gpus),
        "devices": _count(summary, "devices"),
        "bytes": _count(summary, "bytes"),
        "warmup": _count(summary, "warmup"),
        "iterations": _count(summary, "iterations"),
        "nvlink_mesh": mesh,
        "model": label,
        "line_rate_gb_per_s": line,
        "pairs": [],
    }
    error = summary.get("error")
    stop = ""  # what leaves no pair to judge
    if len(gpus) < 2:
        stop = "fewer than two GPUs: no pair to copy between"
    elif len(set(indices)) != len(indices):
        said = ", ".join(str(index) for index in indices)
        stop = f"nvidia-smi does not tell the {len(gpus)} GPUs apart (indices {said}): no pair is judged"
    elif error is None and data["devices"] != len(gpus):
        # The worker numbers the GPUs it sees: with one missing, its pairs name other GPUs.
        stop = f"PyTorch initialises {data['devices']} of the {len(gpus)} GPUs nvidia-smi lists: no pair is judged"
    if stop:
        return HealthCheck(NAME, FAIL, stop, data)

    groups: dict[Any, list[Pair]] = {}
    for pair in copies.pairs if copies is not None else []:
        groups.setdefault((pair.get("src"), pair.get("dst")), []).append(pair)
    data["pairs"] = [pair for key in expected for pair in groups.get(key, [])]
    # A pair counts with exactly one line, and a valid one.
    measured = [groups[key][0] for key in expected if len(groups.get(key, [])) == 1 and _valid(groups[key][0])]
    found = Findings()
    _judge_arrival(found, groups, measured, expected, summary)
    speeds = _judge_speed(found, measured, mesh, label, line, data) if error is None and measured else ""
    if error is None and ended:  # the data is complete, and was judged: not a failure, not a plain pass
        found.warn(f"every pair was reported, but the copy worker did not end well ({ended})")
    passed = (
        f"{len(measured)}/{len(expected)} GPU pairs: {data['bytes'] // MIB} MiB copied from each GPU to every other one, "
        f"{data['warmup']} warm-up copies then {data['iterations']} timed ones; the {data['warmup']} warm-up copies "
        f"and the last timed copy compared byte for byte: 0 wrong bytes{speeds}"
    )
    return found.check(NAME, passed, data, PASS if mesh == "full" or line else INFO)

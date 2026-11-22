# SPDX-License-Identifier: MIT
"""Fakes for the health tests: an nvidia-smi, a kernel log and a PyTorch workload that replay
tests/fixtures/health (see its README), with no GPU and no subprocess.

``FakeHost`` answers like the real tools: ``--query-gpu`` reads gpus.csv, one column per
nvidia-smi field; while the fake workload "runs", the columns of gpus-load.csv replace them;
a field missing from help-query-gpu.txt fails the whole query, as with the real nvidia-smi.
In dmesg.txt, ``@NOW@`` becomes an uptime just after the start of the run: that line falls
during the test.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from voltage_verify.health.host import Completed
from voltage_verify.health.workload import PeerCopies, Probe

FIXTURES = Path(__file__).parent / "fixtures" / "health"
SUPPORTED = set(re.findall(r'"([a-z0-9_.]+)"', (FIXTURES / "help-query-gpu.txt").read_text(encoding="utf-8")))
UPTIME = 5000.0


def _table(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class FakeHost:
    """The machine of one fixture directory. ``fixture=None``: no NVIDIA driver at all."""

    def __init__(self, fixture: str | None, *, kernel_log: str = "dmesg", uptime: float | None = UPTIME) -> None:
        self.dir = FIXTURES / fixture if fixture else None
        self.kernel_log = kernel_log  # "dmesg", "journalctl" or "none"
        self.busy = False
        self.calls: list[list[str]] = []
        self.slept: list[float] = []
        self._uptime = uptime
        self._clock = 1_800_000_000.0
        self.overrides: dict[int, dict[str, str]] = {}  # gpu row -> field -> value, on top of the fixture

    # --- Host protocol -------------------------------------------------------------------------

    def run(self, argv: Sequence[str], timeout: float) -> Completed | None:
        assert isinstance(argv, list) and all(isinstance(a, str) for a in argv), "argv must be a list of strings"
        assert timeout > 0, "every command needs a timeout"
        self.calls.append(list(argv))
        tool, args = argv[0], list(argv[1:])
        if tool == "nvidia-smi":
            return self._smi(args) if self.dir else None
        if tool in ("dmesg", "journalctl"):
            return self._klog(tool)
        return None

    def uptime(self) -> float | None:
        return self._uptime

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self._clock += seconds

    def clock(self) -> float:
        self._clock += 0.25
        return self._clock

    # --- nvidia-smi ----------------------------------------------------------------------------

    def _smi(self, args: list[str]) -> Completed:
        assert self.dir is not None
        if args == ["--help-query-gpu"]:
            return Completed(0, (FIXTURES / "help-query-gpu.txt").read_text(encoding="utf-8"), "")
        if args == ["topo", "-m"]:
            return Completed(0, (self.dir / "topo.txt").read_text(encoding="utf-8"), "")
        if args == ["nvlink", "-s"]:
            path = self.dir / "nvlink.txt"
            return Completed(0, path.read_text(encoding="utf-8") if path.exists() else "", "")
        if len(args) == 2 and args[0].startswith("--query-gpu=") and args[1] == "--format=csv,noheader,nounits":
            return self._query(args[0].removeprefix("--query-gpu=").split(","))
        return Completed(2, "", f"fake nvidia-smi: unsupported arguments: {args}")

    def _query(self, fields: list[str]) -> Completed:
        assert self.dir is not None
        for name in fields:
            if name not in SUPPORTED:
                return Completed(2, f'Field "{name}" is not a valid field to query.\n\n', "")
        rows = _table(self.dir / "gpus.csv")
        load = _table(self.dir / "gpus-load.csv") if self.busy and (self.dir / "gpus-load.csv").exists() else []
        lines = []
        for i, row in enumerate(rows):
            merged = {**row, **(load[i] if i < len(load) else {}), **self.overrides.get(i, {})}
            lines.append(", ".join(merged.get(name, "[N/A]") for name in fields))
        return Completed(0, "\n".join(lines) + "\n", "")

    # --- kernel log ----------------------------------------------------------------------------

    def _klog(self, tool: str) -> Completed | None:
        path = self.dir / "dmesg.txt" if self.dir else None
        if self.kernel_log == "none" or path is None or not path.exists():
            if tool == "dmesg":
                return Completed(1, "", "dmesg: read kernel buffer failed: Operation not permitted")
            return None  # no journalctl on this machine
        if tool != self.kernel_log:
            return Completed(1, "", "dmesg: read kernel buffer failed: Operation not permitted") if tool == "dmesg" else None
        now = f"{(self._uptime or 0.0) + 30.0:12.6f}"
        text = path.read_text(encoding="utf-8").replace("@NOW@", now)
        if tool == "journalctl":  # -o short-monotonic puts the host name after the stamp
            text = re.sub(r"^(\[[^\]]+\]) ", r"\1 node-1 kernel: ", text, flags=re.MULTILINE)
        return Completed(0, text, "")


def burn_record(**changes: Any) -> dict[str, Any]:
    """What a healthy H100 worker reports for the load (typical values, not measurements)."""
    record = {
        "stage": "burn",
        "dtype": "bfloat16",
        "n": 8192,
        "iterations": 28000,
        "checks": 1120,
        "mismatch": 0,
        "finite": True,
        "checksum": -1234567,
        "load_ms": 47200,
        "elapsed_ms": 60000,
        "segments": 13,
        "gflops": 652400,
    }
    return {**record, **changes}


def vram_record(**changes: Any) -> dict[str, Any]:
    record = {
        "stage": "vram",
        "free_bytes": 84_000_000_000,
        "total_bytes": 85_520_000_000,
        "tested_bytes": 77_280_000_000,
        "chunk_bytes": 67_108_864,
        "chunks": 1151,
        "patterns": 5,
        "wrong_bytes": 0,
        "elapsed_ms": 9000,
    }
    return {**record, **changes}


def healthy(**stages: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The three stage records of one GPU's worker, healthy unless a stage is replaced."""
    base = {"cross_check": {"stage": "cross_check", "wrong": 0}, "burn": burn_record(), "vram": vram_record()}
    return {**base, **stages}


def pair_record(src: int, dst: int, **changes: Any) -> dict[str, Any]:
    """What the copy worker reports for one pair on an NVLink mesh: 371000 MiB/s is 389 GB/s, the
    median pair measured by hwcheck on 8x H100 SXM (2026-10-24)."""
    return {"src": src, "dst": dst, "mib_per_s": 371_000, "wrong_bytes": 0, "peer": True, **changes}


def healthy_copies(
    gpus: Sequence[int],
    pairs: dict[tuple[int, int], dict[str, Any]] | None = None,
    /,
    ended: str = "",
    **summary: Any,
) -> PeerCopies:
    """Every ordered pair of these GPUs copied clean and fast, unless ``pairs`` changes one:
    healthy_copies(range(8), {(2, 5): {"wrong_bytes": 4096}}). ``summary`` changes the closing
    record, ``ended`` says that the worker did not end well after it."""
    changed = pairs or {}
    records = [pair_record(i, j, **changed.get((i, j), {})) for i in gpus for j in gpus if i != j]
    closing = {
        "stage": "p2p",
        "devices": len(gpus),
        "bytes": 256 << 20,
        "warmup": 3,
        "iterations": 20,
        "pairs": len(records),
        "elapsed_ms": 21000,
    }
    return PeerCopies({**closing, **summary}, records, ended)


class FakeWorkload:
    """A PyTorch that is absent, CPU-only, or runs and reports the given records per GPU,
    then the given GPU-to-GPU copies (healthy ones by default)."""

    def __init__(
        self,
        host: FakeHost,
        *,
        torch: str | None = "2.14.1+cu130",
        cuda: bool = True,
        records: dict[int, dict[str, dict[str, Any]]] | None = None,
        polls: int = 6,
        copies: PeerCopies | None = None,
    ) -> None:
        self.host = host
        self.torch = torch
        self.cuda = cuda
        self.records = records
        self.polls = polls
        self.started: tuple[list[int], int] | None = None
        self.collected = False
        self.peer_copies = copies
        self.copied: list[int] | None = None
        self.calls_before_copies = 0
        self._left = 0

    def probe(self) -> Probe:
        if self.torch is None:
            return Probe(None, False, 0, "PyTorch is not installed in this Python")
        if not self.cuda:
            return Probe(self.torch, False, 0, f"PyTorch {self.torch} sees no CUDA device")
        return Probe(self.torch, True, 8, "")

    def start(self, gpus: Sequence[int], seconds: int) -> None:
        self.started = (list(gpus), seconds)
        self._left = self.polls
        self.host.busy = True

    def poll(self) -> bool:
        self._left -= 1
        if self._left <= 0:
            self.host.busy = False
        return self._left > 0

    def collect(self) -> dict[int, dict[str, dict[str, Any]]]:
        assert self.started is not None
        self.host.busy = False
        self.collected = True
        if self.records is not None:
            return self.records
        return {gpu: healthy() for gpu in self.started[0]}

    def copies(self, gpus: Sequence[int]) -> PeerCopies:
        assert self.collected, "the copies start after the load workers have ended"
        self.copied = list(gpus)
        self.calls_before_copies = len(self.host.calls)
        return self.peer_copies if self.peer_copies is not None else healthy_copies(gpus)

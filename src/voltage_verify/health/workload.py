# SPDX-License-Identifier: MIT
"""Start the PyTorch workers (``worker.py``), one process per GPU, and read what they report.

The parent never imports PyTorch: it asks a child whether a CUDA build is there (``probe``),
and if so runs one child per GPU with a deadline, then, on a machine with several GPUs, one
child that sees them all for the copies between GPUs (``copies``). What a child prints is read
as untrusted text: only the known stages and fields are kept, each with the type it must have.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Protocol

PROBE_TIMEOUT = 180.0  # importing PyTorch from a cold disk takes a while
FINISH_MARGIN = 900  # seconds allowed after the load for the exact product and the VRAM test
P2P_TIMEOUT = 900.0  # seconds for the copies between GPUs, as in hwcheck's acceptance.sh
KILL_WAIT = 10.0  # seconds a killed worker is waited for
MAX_WORKER_OUTPUT = 1 << 20
MAX_TEXT = 200

# stage -> field -> type: what a worker may report. Anything else is dropped.
_FIELDS: dict[str, dict[str, type]] = {
    "probe": {"torch": str, "cuda": bool, "devices": int, "error": str},
    "cross_check": {"wrong": int, "error": str},
    "burn": {
        "dtype": str,
        "n": int,
        "iterations": int,
        "checks": int,
        "mismatch": int,
        "finite": bool,
        "checksum": int,
        "load_ms": int,
        "elapsed_ms": int,
        "segments": int,
        "gflops": int,
        "error": str,
    },
    "vram": {
        "free_bytes": int,
        "total_bytes": int,
        "tested_bytes": int,
        "chunk_bytes": int,
        "chunks": int,
        "patterns": int,
        "wrong_bytes": int,
        "elapsed_ms": int,
        "error": str,
    },
    # The copies between GPUs: one "p2p_pair" line per ordered pair, then one "p2p" line.
    "p2p_pair": {"src": int, "dst": int, "mib_per_s": int, "wrong_bytes": int, "peer": bool},
    "p2p": {
        "devices": int,
        "bytes": int,
        "warmup": int,
        "iterations": int,
        "pairs": int,
        "elapsed_ms": int,
        "error": str,
    },
}


@dataclass(frozen=True)
class Probe:
    """Whether PyTorch can drive the GPUs from this Python. ``reason`` says why not."""

    torch: str | None
    cuda: bool
    devices: int
    reason: str

    @property
    def usable(self) -> bool:
        return self.cuda and self.devices > 0


@dataclass(frozen=True)
class PeerCopies:
    """What the copy worker reported. ``summary`` is its closing record, or says under
    ``error`` why there is none; ``pairs`` has one record per line that names an ordered pair,
    source and destination by nvidia-smi index, in the order of the lines (a pair reported
    twice is there twice); ``ended`` says how the worker ended when it did not end well, even
    after a closing record."""

    summary: dict[str, Any]
    pairs: list[dict[str, Any]]
    ended: str = ""


class Workload(Protocol):
    def probe(self) -> Probe: ...

    def start(self, gpus: Sequence[int], seconds: int) -> None:
        """Start the load on these GPUs (nvidia-smi indices), all at once."""

    def poll(self) -> bool:
        """True while a worker is still running."""

    def collect(self) -> dict[int, dict[str, dict[str, Any]]]:
        """GPU index -> stage -> record, for every stage a worker reported."""

    def copies(self, gpus: Sequence[int]) -> PeerCopies:
        """Copy a buffer from each of these GPUs (nvidia-smi indices) to every other one, in
        one worker that sees them all. Called once the load has been collected; returns when
        that worker has ended or has been stopped."""


def parse_lines(text: str) -> list[dict[str, Any]]:
    """The JSON lines of one worker, in order. Unknown stages, fields and types are dropped."""
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        try:
            raw = json.loads(line)
        except (ValueError, RecursionError):
            continue
        if not isinstance(raw, dict) or raw.get("stage") not in _FIELDS:
            continue
        stage = str(raw["stage"])
        record: dict[str, Any] = {"stage": stage}
        for key, kind in _FIELDS[stage].items():
            value = raw.get(key)
            if isinstance(value, str) and kind is str:
                record[key] = "".join(c for c in value if c.isprintable())[:MAX_TEXT]
            elif type(value) is kind and kind is not str:  # exact type: a bool is not accepted as an int
                record[key] = value
        records.append(record)
    return records


def parse_records(text: str) -> dict[str, dict[str, Any]]:
    """The records of one worker by stage, the last one when a stage reported twice."""
    return {record["stage"]: record for record in parse_lines(text)}


def peer_copies(text: str, gpus: Sequence[int], ended: str = "") -> PeerCopies:
    """What the copy worker printed, read as untrusted text. The worker numbers the GPUs it
    sees from 0, in the order of ``gpus``: a pair is kept under their nvidia-smi indices, a
    line that names no such pair is dropped. Every other line is kept, in order: a second line
    for a pair never replaces the first one. ``ended`` says how the worker ended when it did
    not end well: it is the error of a worker that never printed its closing record."""
    summary: dict[str, Any] | None = None
    pairs: list[dict[str, Any]] = []
    for record in parse_lines(text):
        if record["stage"] == "p2p":
            summary = record
        elif record["stage"] == "p2p_pair":
            src, dst = record.get("src"), record.get("dst")
            if src is None or dst is None or src == dst or not (0 <= src < len(gpus) and 0 <= dst < len(gpus)):
                continue
            measured = {k: v for k, v in record.items() if k not in ("stage", "src", "dst")}
            pairs.append({"src": gpus[src], "dst": gpus[dst], **measured})
    if summary is None:
        summary = {"stage": "p2p", "error": ended or "no result from its worker"}
    return PeerCopies(summary, pairs, ended)


def _worker_environment(gpus: Sequence[int] = ()) -> dict[str, str]:
    """The environment of a worker: this package importable, and only these GPUs visible, by
    their nvidia-smi indices (CUDA orders GPUs differently unless told to follow the PCI bus).
    No GPU named: the caller's environment decides, as for ``probe``."""
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parents[2])
    # The worker starts in another directory (see _private_directory): relative entries of the
    # caller's PYTHONPATH are made absolute so that they still mean the same place.
    inherited = [os.path.abspath(p) for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    env["PYTHONPATH"] = os.pathsep.join([root, *inherited])
    if gpus:
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(int(gpu)) for gpu in gpus)
    return env


def _private_directory() -> tempfile.TemporaryDirectory[str]:
    """Where a worker starts: an empty directory only this user can write to. ``python -m`` and
    ``python -c`` put the current directory first on sys.path, so a ``torch.py`` lying where
    the user stands would otherwise be imported in place of PyTorch."""
    return tempfile.TemporaryDirectory(prefix="voltage-verify-", ignore_cleanup_errors=True)


def _stop(process: subprocess.Popen[bytes]) -> int | None:
    """Kills a worker and waits for it, but not for ever: a process stuck in the driver may
    not end even killed. Returns its exit code, None when it is still there."""
    process.kill()
    try:
        return process.wait(timeout=KILL_WAIT)
    except subprocess.TimeoutExpired:
        return None


@dataclass
class _Running:
    gpu: int
    process: subprocess.Popen[bytes]
    stdout: IO[bytes]
    stderr: IO[bytes]


class TorchWorkload:
    """The real workload: ``python -m voltage_verify.health.worker`` with this interpreter."""

    def __init__(
        self,
        command: Sequence[str] | None = None,
        finish_margin: float = FINISH_MARGIN,
        p2p_timeout: float = P2P_TIMEOUT,
    ) -> None:
        self._command = list(command) if command is not None else [sys.executable, "-m", "voltage_verify.health.worker"]
        self._finish_margin = finish_margin
        self._p2p_timeout = p2p_timeout
        self._running: list[_Running] = []
        self._home: tempfile.TemporaryDirectory[str] | None = None
        self._deadline = 0.0
        self._timed_out = False

    def probe(self) -> Probe:
        try:
            with _private_directory() as home:
                done = subprocess.run(
                    [*self._command, "probe"],
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    timeout=PROBE_TIMEOUT,
                    check=False,
                    env=_worker_environment(),
                    cwd=home,
                )
        except (OSError, subprocess.SubprocessError) as err:
            return Probe(None, False, 0, f"PyTorch could not be probed ({type(err).__name__})")
        record = parse_records(done.stdout[:MAX_WORKER_OUTPUT].decode("utf-8", "replace")).get("probe")
        if record is None:
            return Probe(None, False, 0, f"PyTorch could not be probed (exit code {done.returncode})")
        version = record.get("torch")
        if version is None:
            error = record.get("error", "")
            if error in ("ModuleNotFoundError", "ImportError", ""):
                return Probe(None, False, 0, "PyTorch is not installed in this Python")
            return Probe(None, False, 0, f"PyTorch is installed but does not load ({error})")
        devices = record.get("devices", 0)
        if not record.get("cuda") or devices < 1:
            return Probe(version, False, 0, f"PyTorch {version} is installed but sees no CUDA device (a CPU-only build?)")
        return Probe(version, True, devices, "")

    def start(self, gpus: Sequence[int], seconds: int) -> None:
        self._deadline = time.monotonic() + seconds + self._finish_margin
        self._timed_out = False
        self._home = _private_directory()
        for gpu in gpus:
            # Outputs go to unnamed temporary files, not pipes: a pipe nobody reads fills up.
            # They are closed in collect().
            stdout, stderr = tempfile.TemporaryFile(), tempfile.TemporaryFile()
            process = subprocess.Popen(  # an argument list, no shell
                [*self._command, "run", "--seconds", str(int(seconds))],
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                env=_worker_environment([gpu]),
                cwd=self._home.name,
            )
            self._running.append(_Running(gpu, process, stdout, stderr))

    def poll(self) -> bool:
        alive = [r for r in self._running if r.process.poll() is None]
        if alive and time.monotonic() > self._deadline:
            self._timed_out = True
            for r in alive:
                r.process.kill()
            return False
        return bool(alive)

    def collect(self) -> dict[int, dict[str, dict[str, Any]]]:
        results: dict[int, dict[str, dict[str, Any]]] = {}
        for r in self._running:
            # Every worker has ended or is killed here: the wait is short, and bounded.
            code = _stop(r.process) if r.process.poll() is None else r.process.returncode
            r.stdout.seek(0)
            records = parse_records(r.stdout.read(MAX_WORKER_OUTPUT).decode("utf-8", "replace"))
            r.stdout.close()
            r.stderr.close()
            why = "its worker did not end after it was killed" if code is None else f"its worker ended with exit code {code}"
            if self._timed_out:
                why = "stopped at the deadline"
            for stage in ("cross_check", "burn", "vram"):  # a stage that never reported: say how the worker ended
                if stage not in records and (self._timed_out or code != 0):
                    records[stage] = {"stage": stage, "error": why}
            results[r.gpu] = records
        self._running = []
        if self._home is not None:
            self._home.cleanup()
            self._home = None
        return results

    def copies(self, gpus: Sequence[int]) -> PeerCopies:
        ended, text = "", ""
        try:
            # The output goes to an unnamed temporary file: what was printed before the deadline
            # is kept, and at most MAX_WORKER_OUTPUT bytes of it are read.
            with _private_directory() as home, tempfile.TemporaryFile() as stdout:
                process = subprocess.Popen(  # an argument list, no shell
                    [*self._command, "p2p"],
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=subprocess.DEVNULL,
                    env=_worker_environment(gpus),
                    cwd=home,
                )
                try:
                    code = process.wait(timeout=self._p2p_timeout)
                    if code != 0:
                        ended = f"its worker ended with exit code {code}"
                except subprocess.TimeoutExpired:
                    ended = f"stopped at the deadline ({int(self._p2p_timeout)} s)"
                finally:  # at the deadline, and on Ctrl-C: the worker is killed, and not waited for ever
                    if process.poll() is None and _stop(process) is None:
                        ended += ", and still there after the kill"
                stdout.seek(0)
                text = stdout.read(MAX_WORKER_OUTPUT).decode("utf-8", "replace")
        except (OSError, subprocess.SubprocessError) as err:
            ended = f"its worker could not be started ({type(err).__name__})"
        return peer_copies(text, gpus, ended)

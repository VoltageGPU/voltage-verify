# SPDX-License-Identifier: MIT
"""Start the PyTorch workers (``worker.py``), one process per GPU, and read what they report.

The parent never imports PyTorch: it asks a child whether a CUDA build is there (``probe``),
and if so runs one child per GPU with a deadline. What a child prints is read as untrusted
text: only the known stages and fields are kept, each with the type it must have.
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


class Workload(Protocol):
    def probe(self) -> Probe: ...

    def start(self, gpus: Sequence[int], seconds: int) -> None:
        """Start the load on these GPUs (nvidia-smi indices), all at once."""

    def poll(self) -> bool:
        """True while a worker is still running."""

    def collect(self) -> dict[int, dict[str, dict[str, Any]]]:
        """GPU index -> stage -> record, for every stage a worker reported."""


def parse_records(text: str) -> dict[str, dict[str, Any]]:
    """The JSON lines of one worker, by stage. Unknown stages, fields and types are dropped."""
    records: dict[str, dict[str, Any]] = {}
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
        records[stage] = record
    return records


def _worker_environment(gpu: int | None = None) -> dict[str, str]:
    """The environment of a worker: this package importable, and at most one GPU visible, by
    its nvidia-smi index (CUDA orders GPUs differently unless told to follow the PCI bus)."""
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parents[2])
    # The worker starts in another directory (see _private_directory): relative entries of the
    # caller's PYTHONPATH are made absolute so that they still mean the same place.
    inherited = [os.path.abspath(p) for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    env["PYTHONPATH"] = os.pathsep.join([root, *inherited])
    if gpu is not None:
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    return env


def _private_directory() -> tempfile.TemporaryDirectory[str]:
    """Where a worker starts: an empty directory only this user can write to. ``python -m`` and
    ``python -c`` put the current directory first on sys.path, so a ``torch.py`` lying where
    the user stands would otherwise be imported in place of PyTorch."""
    return tempfile.TemporaryDirectory(prefix="voltage-verify-", ignore_cleanup_errors=True)


@dataclass
class _Running:
    gpu: int
    process: subprocess.Popen[bytes]
    stdout: IO[bytes]
    stderr: IO[bytes]


class TorchWorkload:
    """The real workload: ``python -m voltage_verify.health.worker`` with this interpreter."""

    def __init__(self, command: Sequence[str] | None = None, finish_margin: float = FINISH_MARGIN) -> None:
        self._command = list(command) if command is not None else [sys.executable, "-m", "voltage_verify.health.worker"]
        self._finish_margin = finish_margin
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
                env=_worker_environment(gpu),
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
            if r.process.poll() is None:
                r.process.kill()
            code = r.process.wait()
            r.stdout.seek(0)
            records = parse_records(r.stdout.read(MAX_WORKER_OUTPUT).decode("utf-8", "replace"))
            r.stdout.close()
            r.stderr.close()
            why = "stopped at the deadline" if self._timed_out else f"its worker ended with exit code {code}"
            for stage in ("cross_check", "burn", "vram"):  # a stage that never reported: say how the worker ended
                if stage not in records and (self._timed_out or code != 0):
                    records[stage] = {"stage": stage, "error": why}
            results[r.gpu] = records
        self._running = []
        if self._home is not None:
            self._home.cleanup()
            self._home = None
        return results

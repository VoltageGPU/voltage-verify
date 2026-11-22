# SPDX-License-Identifier: MIT
"""The machine the health check runs on: the only place a command is started.

Every command is an argument list handed to the tool itself (no shell), with a timeout, as
the current user: nothing asks for root, nothing touches the network.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

MAX_OUTPUT_CHARS = 8 * 1024 * 1024  # of one tool's output: a kernel log can be large


@dataclass(frozen=True)
class Completed:
    returncode: int
    stdout: str
    stderr: str


class Host(Protocol):
    def run(self, argv: Sequence[str], timeout: float) -> Completed | None:
        """Run a tool to its end. None when it is missing, cannot start or runs out of time."""

    def uptime(self) -> float | None:
        """Seconds since boot, the clock of the kernel log; None when the system has none."""

    def sleep(self, seconds: float) -> None: ...

    def clock(self) -> float:
        """Seconds since the epoch."""


def find_tool(name: str) -> str | None:
    """Where a tool is, looking only in the absolute directories of PATH.

    Not ``shutil.which`` alone: on Windows it looks in the current directory first, and a PATH
    may hold ``.`` or an empty entry. A file named nvidia-smi dropped where the user stands
    must never be the one that runs.
    """
    if os.path.dirname(name):
        return name if os.path.isfile(name) else None
    suffixes = [""]
    if os.name == "nt" and not os.path.splitext(name)[1]:
        suffixes = [".exe", ".com"]  # programs only: a .bat or .cmd would need a shell
    for directory in os.get_exec_path():
        if not directory or not os.path.isabs(directory):
            continue
        for suffix in suffixes:
            candidate = os.path.join(directory, name + suffix)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
    return None


class SystemHost:
    def __init__(self, max_output: int = MAX_OUTPUT_CHARS) -> None:
        self._max_output = max_output

    def run(self, argv: Sequence[str], timeout: float) -> Completed | None:
        tool = find_tool(argv[0])
        if tool is None:
            return None
        try:
            done = subprocess.run(
                [tool, *argv[1:]],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return Completed(done.returncode, done.stdout[: self._max_output], done.stderr[: self._max_output])

    def uptime(self) -> float | None:
        try:
            return float(Path("/proc/uptime").read_text(encoding="ascii").split()[0])
        except (OSError, ValueError, IndexError):
            return None

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def clock(self) -> float:
        return time.time()

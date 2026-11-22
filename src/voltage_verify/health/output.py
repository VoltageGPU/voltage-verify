# SPDX-License-Identifier: MIT
"""Where ``health -o`` writes its report.

The report names the machine (host name, GPU UUIDs, serial numbers) and the tool is often run
as root, so the file is written the careful way: into a temporary file next to the target,
readable by its owner only, then moved in place. The move replaces whatever sits at the
target, a symbolic link included, and never writes through it. The temporary file is created
before the test starts: a path that cannot be written is refused then, not after an hour of
load.
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path


def sudo_owner(environ: Mapping[str, str], euid: int) -> tuple[int, int] | None:
    """The user and group that ran ``sudo``, when this process is root because of it.

    ``health`` reads the kernel log (Xid errors) only as root, so it is often run with sudo by
    a user who then copies the report away: the file goes back to that user, still readable by
    its owner only. None when not root, when root ran it directly, or when sudo's variables are
    not two plain numbers."""
    if euid != 0:
        return None
    try:
        uid, gid = int(environ["SUDO_UID"]), int(environ["SUDO_GID"])
    except (KeyError, ValueError):
        return None
    if uid <= 0 or gid < 0:
        return None
    return uid, gid


class ReportFile:
    """A report file being written. ``commit`` puts it in place, ``discard`` removes the draft."""

    def __init__(self, target: str) -> None:
        self._target = Path(target)
        folder = self._target.parent
        # mkstemp: a name nobody can guess, created exclusively, mode 0600.
        handle, draft = tempfile.mkstemp(prefix=f".{self._target.name}.", suffix=".tmp", dir=folder)
        try:
            if sys.platform != "win32":
                owner = sudo_owner(os.environ, os.geteuid())
                if owner is not None:
                    # On the open descriptor, not the path: nothing can be swapped in between.
                    with contextlib.suppress(OSError):  # a file system without owners: root keeps it
                        os.fchown(handle, *owner)
        finally:
            os.close(handle)
        self._draft: str | None = draft

    def commit(self, data: bytes) -> None:
        """Write ``data`` and move it to the target. Raises OSError; the draft is then removed."""
        if self._draft is None:
            raise OSError("report file already closed")
        try:
            with open(self._draft, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(self._draft, self._target)
            self._draft = None
        finally:
            self.discard()

    def discard(self) -> None:
        if self._draft is not None:
            with contextlib.suppress(OSError):
                os.unlink(self._draft)
            self._draft = None

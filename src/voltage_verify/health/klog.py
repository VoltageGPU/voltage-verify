# SPDX-License-Identifier: MIT
"""Read the kernel log for NVIDIA Xid events, as the current user.

``dmesg`` first, then the journal. Both are refused to an unprivileged user on many systems
(kernel.dmesg_restrict, a container): the caller then reports the check as not run. ``sudo``
is never tried: this tool does not ask for root.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .host import Host

TIMEOUT = 20.0
MAX_EVENTS = 1000
# Lines stamped with seconds since boot: "[ 7.345678] ...", the journal adds "host kernel: ".
_STAMP_RE = re.compile(r"^\[ *([0-9]{1,9})\.([0-9]{1,9})\]", re.MULTILINE)
_XID_RE = re.compile(r"^\[ *([0-9]{1,9})\.([0-9]{1,9})\].*NVRM: Xid \(([^)]{1,40})\): *([0-9]{1,4})", re.MULTILINE)
_SOURCES = (
    ("dmesg", ["dmesg"]),
    ("journalctl", ["journalctl", "-k", "-b", "--no-pager", "-o", "short-monotonic"]),
)


@dataclass(frozen=True)
class XidEvent:
    uptime_ms: int
    xid: int
    pci: str


@dataclass(frozen=True)
class KernelLog:
    source: str
    events: tuple[XidEvent, ...]


def parse_xids(text: str) -> tuple[XidEvent, ...]:
    events = []
    for match in _XID_RE.finditer(text):
        seconds, fraction, pci, code = match.groups()
        millis = int(seconds) * 1000 + int(fraction.ljust(3, "0")[:3])
        pci_clean = "".join(c for c in pci if c.isalnum() or c in ":.-")
        events.append(XidEvent(millis, int(code), pci_clean))
        if len(events) >= MAX_EVENTS:
            break
    return tuple(events)


def read(host: Host) -> KernelLog | None:
    """The Xid events of this boot, or None when no kernel log can be read."""
    for source, argv in _SOURCES:
        done = host.run(argv, TIMEOUT)
        if done is not None and done.returncode == 0 and _STAMP_RE.search(done.stdout):
            return KernelLog(source, parse_xids(done.stdout))
    return None

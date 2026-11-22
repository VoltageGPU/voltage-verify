# SPDX-License-Identifier: MIT
"""The health report: its statuses, its verdict, its canonical JSON form, and the commitment
a manifest can carry to it.

Format ``voltage-verify/health/1``. The report holds no float (integers, and the raw strings
nvidia-smi printed), so it has one canonical encoding, the one the manifest uses (keys sorted,
no whitespace, UTF-8). ``health -o`` writes exactly those bytes: the SHA-256 of the file is
the SHA-256 a manifest commits to, and ``sha256sum`` shows it without this tool.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from .. import __version__
from ..canonical import CanonicalError, canonical_bytes

SCHEMA = "voltage-verify/health/1"

# PASS and FAIL are judged against a threshold. WARN is worth reading before a long job. SKIP
# could not run: it never counts as a pass, the verdict names it. INFO is measured, with no
# reference to judge it (or does not apply to this hardware).
PASS, FAIL, WARN, SKIP, INFO = "PASS", "FAIL", "WARN", "SKIP", "INFO"
STATUSES = (PASS, FAIL, WARN, SKIP, INFO)

# What a report must carry to be read as a full pass by ``verify``. A check left out is not a
# check that passed: leaving one out is easier than lying about its result.
REQUIRED_CHECKS = (
    "inventory.gpus",
    "errors.ecc",
    "errors.xid",
    "sensors.thermal",
    "sensors.clocks",
    "compute.matmul",
    "vram.pattern",
)
# A report is written before the manifest that commits to it. Older than this, it describes
# the machine as it was, not as it is attested.
MAX_AGE_AT_MANIFEST = timedelta(hours=24)
CLOCK_SKEW = timedelta(minutes=5)
NOT_SUPPLIED = "health report committed but not checked (no --health)"

MAX_REPORT_BYTES = 16 * 1024 * 1024
_NAME_RE = re.compile(r"[a-z0-9_.]{1,64}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
# Exit codes, as for ``verify``: 0 pass, 1 a check failed, 2 unusable input, 3 a limit the
# verdict names (here: a warning, or something that was not tested).
_EXIT = {"PASS": 0, "FAIL": 1, "WARN": 3, "INCOMPLETE": 3}


class HealthReportError(ValueError):
    """Raised when a health report cannot be read, or is not one."""


@dataclass(frozen=True)
class HealthCheck:
    name: str
    status: str
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Summary:
    result: str  # PASS, WARN, INCOMPLETE or FAIL
    verdict: str  # the RESULT line
    failed: list[str]
    not_tested: list[str]
    warnings: list[str]


def summarize(statuses: Iterable[tuple[str, str]]) -> Summary:
    """The verdict of (name, status) pairs. A SKIP is never a pass: the verdict lists it."""
    by: dict[str, list[str]] = {status: [] for status in STATUSES}
    for name, status in statuses:
        if status not in by:
            raise HealthReportError(f"unknown status {str(status)[:20]!r} for check {str(name)[:40]!r}")
        by[status].append(name)
    failed, skipped, warned = by[FAIL], by[SKIP], by[WARN]
    notes = []
    if skipped:
        notes.append("not tested: " + ", ".join(skipped))
    if warned:
        notes.append("warnings: " + ", ".join(warned))
    if failed:
        return Summary("FAIL", f"FAIL ({'; '.join([', '.join(failed), *notes])})", failed, skipped, warned)
    if skipped:
        return Summary("INCOMPLETE", f"INCOMPLETE ({'; '.join(notes)})", failed, skipped, warned)
    if warned:
        return Summary("WARN", f"WARN ({', '.join(warned)})", failed, skipped, warned)
    return Summary("PASS", "PASS", failed, skipped, warned)


def exit_code(result: str) -> int:
    return _EXIT.get(result, 1)


@dataclass
class HealthReport:
    checks: list[HealthCheck]
    created_at: datetime  # UTC, when the run started
    duration_ms: int
    machine: dict[str, Any]
    options: dict[str, Any]
    samples: dict[str, Any]  # the raw nvidia-smi samples taken during the run

    @property
    def summary(self) -> Summary:
        return summarize((c.name, c.status) for c in self.checks)

    def to_json(self) -> dict[str, Any]:
        summary = self.summary
        return {
            "schema": SCHEMA,
            "tool": {"name": "voltage-verify", "version": __version__},
            "created_at": self.created_at.isoformat(timespec="seconds"),
            "duration_ms": self.duration_ms,
            "machine": self.machine,
            "options": self.options,
            "ok": summary.result == "PASS",
            "result": summary.result,
            "verdict": summary.verdict,
            "failed": summary.failed,
            "not_tested": summary.not_tested,
            "warnings": summary.warnings,
            "checks": [{"name": c.name, "result": c.status, "detail": c.detail, "data": c.data} for c in self.checks],
            "samples": self.samples,
        }


def canonical_report(doc: dict[str, Any]) -> bytes:
    """The canonical UTF-8 bytes of a report: what ``health -o`` writes and what is hashed."""
    try:
        return canonical_bytes(doc)
    except CanonicalError as err:
        raise HealthReportError(str(err).replace("in a manifest", "in a health report")) from err
    except (UnicodeError, RecursionError) as err:  # a lone surrogate has no UTF-8 bytes
        raise HealthReportError(f"health report has no canonical encoding ({type(err).__name__})") from err


def report_sha256(doc: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_report(doc)).hexdigest()


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    # A key written twice reads differently from one parser to the next: refuse it.
    keys = [k for k, _ in pairs]
    if len(set(keys)) != len(keys):
        raise HealthReportError("health report has a duplicated key")
    return dict(pairs)


def _refuse(token: str) -> Any:
    raise HealthReportError(f"health report holds a number that is not an integer ({token[:20]!r}): floats are not allowed")


def load_report(raw: bytes) -> dict[str, Any]:
    """Parse and validate a health report. Raises HealthReportError, never anything else."""
    if len(raw) > MAX_REPORT_BYTES:
        raise HealthReportError(f"health report is larger than {MAX_REPORT_BYTES} bytes")
    try:
        doc = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicates, parse_float=_refuse, parse_constant=_refuse)
    except HealthReportError:
        raise
    except (ValueError, RecursionError) as err:
        raise HealthReportError(f"health report is not JSON ({type(err).__name__})") from err
    if not isinstance(doc, dict):
        raise HealthReportError("health report is not a JSON object")
    if doc.get("schema") != SCHEMA:
        raise HealthReportError(f"unsupported health report schema {repr(doc.get('schema'))[:40]}, expected {SCHEMA!r}")
    checks = doc.get("checks")
    if not isinstance(checks, list) or not all(isinstance(c, dict) for c in checks):
        raise HealthReportError("health report has no list of checks")
    for check in checks:
        name, status = check.get("name"), check.get("result")
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
            raise HealthReportError(f"health report has a malformed check name {repr(name)[:40]}")
        if status not in STATUSES:
            raise HealthReportError(f"check {name} has an unknown result {repr(status)[:20]}")
        if not isinstance(check.get("detail"), str):
            raise HealthReportError(f"check {name} has no detail text")
    names = [check["name"] for check in checks]
    twice = sorted({name for name in names if names.count(name) > 1})
    if twice:
        raise HealthReportError(f"health report holds the check {twice[0]} twice")
    for key in ("result", "verdict", "created_at"):
        if not isinstance(doc.get(key), str):
            raise HealthReportError(f"health report field {key!r} is missing or not a string")
    try:
        created = datetime.fromisoformat(doc["created_at"].replace("Z", "+00:00"))
    except ValueError as err:
        raise HealthReportError("health report created_at is not a date") from err
    if created.tzinfo is None:
        raise HealthReportError("health report created_at has no UTC offset")
    canonical_report(doc)
    return doc


# --- The link with the attestation: a manifest commits to a report by its SHA-256 --------------


def manifest_entry(doc: dict[str, Any]) -> dict[str, str]:
    """What a manifest carries under ``health`` to commit to this report."""
    return {"schema": SCHEMA, "sha256": report_sha256(doc)}


def validate_manifest_entry(entry: Any) -> None:
    if not isinstance(entry, dict):
        raise HealthReportError("manifest health entry is not an object")
    if entry.get("schema") != SCHEMA:
        raise HealthReportError(f"manifest health entry has an unsupported schema {repr(entry.get('schema'))[:40]}")
    if not isinstance(entry.get("sha256"), str) or not _SHA256_RE.fullmatch(entry["sha256"]):
        raise HealthReportError("manifest health entry sha256 is not 32 bytes of lowercase hex")


@dataclass(frozen=True)
class Binding:
    """The outcome of ``health.report`` in ``verify``."""

    ok: bool
    required: bool
    detail: str
    facts: dict[str, str] | None = None
    # What a matching report itself did not establish (warnings, untested parts): ``verify``
    # names it in its verdict, as it does for an offline TCB.
    limit: str | None = None


def bind(manifest: dict[str, Any], raw: bytes | None) -> Binding | None:
    """Check a health report against the manifest that should commit to it.

    None when the manifest commits to no report and none was supplied: such a bundle gets no
    ``health.report`` line at all. The manifest is assumed valid (``validate_manifest``).
    """
    if "health" not in manifest:
        if raw is None:
            return None
        return Binding(
            False,
            True,
            "this manifest commits to no health report: the report supplied is not covered by the quote "
            "or the NVIDIA tokens",
        )
    try:
        validate_manifest_entry(manifest["health"])
    except HealthReportError as err:
        return Binding(False, True, str(err))
    committed = str(manifest["health"]["sha256"])
    if raw is None:
        # Not a failure (the bundle is what it is), not a plain pass either: the report may say
        # FAIL, and nobody looked. The verdict names it.
        return Binding(
            False,
            False,
            f"the manifest commits to a health report (sha256 {committed[:16]}...) that was not supplied: "
            "pass --health FILE to check it",
            limit=NOT_SUPPLIED,
        )
    try:
        doc = load_report(raw)
    except HealthReportError as err:
        return Binding(False, True, f"health report is unusable: {err}")
    digest = report_sha256(doc)
    if digest != committed:
        return Binding(
            False,
            True,
            f"health report sha256 {digest[:16]}... is NOT the one the manifest commits to ({committed[:16]}...): "
            "altered after the commitment, or another report",
        )
    summary = summarize((c["name"], c["result"]) for c in doc["checks"])
    if (summary.result, summary.verdict) != (doc["result"], doc["verdict"]):
        return Binding(
            False,
            True,
            f"health report sha256 {digest[:16]}... is the one the manifest commits to, but its verdict "
            f"({doc['verdict'][:80]!r}) contradicts its own checks ({summary.verdict})",
        )
    present = {c["name"] for c in doc["checks"]}
    missing = [name for name in REQUIRED_CHECKS if name not in present]
    detail = (
        f"health report sha256 {digest[:16]}... is the one the manifest commits to; measured {doc['created_at']} "
        f"from inside the machine, verdict of the report: {summary.verdict}"
    )
    facts = {
        "schema": SCHEMA,
        "sha256": digest,
        "result": summary.result,
        "verdict": summary.verdict,
        "created_at": doc["created_at"],
    }
    if summary.result == "FAIL":  # a failed report fails the verification, whatever else it lacks
        return Binding(False, True, detail, facts)
    # A warning, an untested part, a check left out or a report of another day is a warning here
    # and a limit of the verdict: none of them ever turns into a plain VERIFIED.
    limits, notes = [], []
    if missing:
        facts["missing"] = ", ".join(missing)
        without = f"without the checks {', '.join(missing)}"
        joint = ", but" if summary.result == "PASS" else ", and"
        limits.append(f"health report {summary.verdict}{joint} {without}")
        notes.append(without)
    elif summary.result != "PASS":
        limits.append(f"health report {summary.verdict}")
    stale = _measured_when(doc["created_at"], manifest.get("created_at"))
    if stale:
        limits.append(stale)
        notes.append(stale)
    if not limits:
        return Binding(True, False, detail, facts)
    return Binding(False, False, "; ".join([detail, *notes]), facts, "; ".join(limits))


def _utc(text: Any) -> datetime | None:
    if not isinstance(text, str):
        return None
    try:
        when = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return when if when.tzinfo is not None else None


def _measured_when(report_date: str, manifest_date: Any) -> str | None:
    """A limit when the report was not measured shortly before the manifest that commits to it.
    Both dates are the word of whoever wrote them: this catches an old report left lying
    around, not a forged date."""
    measured, committed = _utc(report_date), _utc(manifest_date)
    if measured is None or committed is None:
        return None
    age = committed - measured
    if age < -CLOCK_SKEW:
        return "health report dated after the manifest that commits to it"
    if age <= MAX_AGE_AT_MANIFEST:
        return None
    hours = int(age.total_seconds() // 3600)
    span = f"{hours} hours" if hours < 48 else f"{age.days} days"
    return f"health report measured {span} before the manifest"

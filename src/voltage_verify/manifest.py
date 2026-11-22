# SPDX-License-Identifier: MIT
"""Build the workload manifest: what the verifier expects to find inside the attested VM.

The verifier writes the manifest, including a fresh random challenge, and hands it to the
VM. The VM never chooses anything: it hashes the manifest and asks the hardware to sign that
hash. A manifest built after the fact, or with a recycled challenge, produces different
commitments and the proofs stop matching.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .bundle import MANIFEST_FORMAT
from .canonical import CanonicalError, canonical_bytes
from .health.report import MAX_REPORT_BYTES, HealthReportError, load_report, manifest_entry, validate_manifest_entry
from .registry import DIGEST_RE, resolve_digest


class ManifestError(ValueError):
    """Raised when a manifest cannot be built or is malformed."""


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def new_challenge() -> str:
    """32 random bytes, hex. Generated on the verifier's machine, never on the VM."""
    return secrets.token_hex(32)


def health_commitment(path: Path) -> dict[str, str]:
    """The ``health`` entry of a manifest: the schema and the SHA-256 of a health report file
    (``voltage-verify health -o``), hashed in its canonical form."""
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_REPORT_BYTES + 1)
        return manifest_entry(load_report(raw))
    except OSError as err:
        raise ManifestError(f"cannot read the health report {path}: {type(err).__name__}") from err
    except HealthReportError as err:
        raise ManifestError(f"{path} is not a usable health report: {err}") from err


def build_manifest(
    image: str | None = None,
    image_digest: str | None = None,
    artifacts: list[Path] | None = None,
    extra: dict[str, str] | None = None,
    statement: str | None = None,
    challenge: str | None = None,
    health: Path | None = None,
) -> dict[str, Any]:
    """Assemble a manifest. Resolves the image digest through the registry when not given.

    ``health``: a health report to commit to. The manifest then carries its SHA-256 under
    ``health``; without one the key is absent and the manifest is what 0.3.0 wrote.
    """
    if challenge is None:
        challenge = new_challenge()
    challenge = challenge.lower()
    if len(challenge) != 64 or any(c not in "0123456789abcdef" for c in challenge):
        raise ManifestError("challenge must be 32 bytes in hex (64 characters)")

    image_entry: dict[str, str] | None = None
    if image or image_digest:
        if image_digest and not DIGEST_RE.match(image_digest):
            raise ManifestError("image digest must look like sha256:<64 hex>")
        digest = image_digest or resolve_digest(str(image))
        image_entry = {"reference": image or "", "digest": digest}

    artifact_entries: list[dict[str, Any]] = []
    for path in artifacts or []:
        if not path.is_file():
            raise ManifestError(f"artifact {path} is not a file")
        sha, size = sha256_file(path)
        artifact_entries.append({"name": path.name, "sha256": sha, "size": size})
    artifact_entries.sort(key=lambda a: a["name"])

    manifest: dict[str, Any] = {
        "format": MANIFEST_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "challenge": challenge,
        "workload": {
            "image": image_entry,
            "artifacts": artifact_entries,
            "extra": dict(sorted((extra or {}).items())),
        },
        "statement": statement,
    }
    if health is not None:
        manifest["health"] = health_commitment(health)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Structural checks used by both the attester and the verifier."""
    if not isinstance(manifest, dict):
        raise ManifestError("manifest is not a JSON object")
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ManifestError(f"unsupported manifest format {repr(manifest.get('format'))[:40]}")
    challenge = str(manifest.get("challenge", ""))
    if len(challenge) != 64 or any(c not in "0123456789abcdef" for c in challenge):
        raise ManifestError("manifest challenge is not 32 bytes of lowercase hex")
    workload = manifest.get("workload")
    if not isinstance(workload, dict):
        raise ManifestError("manifest has no workload object")
    image = workload.get("image")
    if image is not None and not (isinstance(image, dict) and DIGEST_RE.match(str(image.get("digest", "")))):
        raise ManifestError("workload image digest is malformed")
    artifacts = workload.get("artifacts", [])
    if not isinstance(artifacts, list) or not all(isinstance(art, dict) for art in artifacts):
        raise ManifestError("workload artifacts are not a list of objects")
    for art in artifacts:
        if len(str(art.get("sha256", ""))) != 64:
            raise ManifestError(f"artifact {str(art.get('name'))[:40]!r} has a malformed sha256")
    if "health" in manifest:  # optional, since 0.4.0: the commitment to a health report
        try:
            validate_manifest_entry(manifest["health"])
        except HealthReportError as err:
            raise ManifestError(str(err)) from err
    try:
        canonical_bytes(manifest)  # the commitments are computed from it: floats are refused here
    except CanonicalError as err:
        raise ManifestError(str(err)) from err

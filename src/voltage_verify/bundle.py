# SPDX-License-Identifier: MIT
"""The evidence bundle: one JSON file that carries everything a verifier needs.

Format ``voltage-verify-bundle/1``::

    {
      "format": "voltage-verify-bundle/1",
      "tool": {"name": "voltage-verify", "version": "0.1.0"},
      "created_at": "2026-09-12T01:02:03+00:00",
      "manifest": { ... the workload manifest, verbatim ... },
      "commitments": {"sha256": "<hex>", "sha512": "<hex>"},
      "tdx": {"quote_b64": "<base64>", "report_data_hex": "<hex>"},
      "nvidia": {"mode": "single-gpu" | "multi-gpu-ppcie", "nras": [...], "jwks": {...}},
      "collateral": {"intel": { ... pcs.Collateral ... }},
      "environment": { ... free-form facts recorded on the VM ... }
    }

The manifest is the only input a verifier must trust nothing about: every other field is
either derived from it (commitments), signed by Intel (quote, collateral) or signed by NVIDIA
(NRAS tokens). The JWKS and Intel collateral are embedded so that verification can run
offline against the material of the day; an online verification refetches Intel's collateral.
NVIDIA signing keys always come from the embedded JWKS, each through its x5c chain to a
pinned NRAS intermediate (GPU Intermediate 004 or 005).
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__

FORMAT = "voltage-verify-bundle/1"
MANIFEST_FORMAT = "voltage-verify-manifest/1"


class BundleError(ValueError):
    """Raised when a bundle file is not usable."""


@dataclass
class Bundle:
    manifest: dict[str, Any]
    commitments: dict[str, str]
    quote: bytes
    report_data: bytes
    nvidia_mode: str
    nras: Any
    jwks: dict[str, Any] | None
    collateral: dict[str, Any] | None
    environment: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tool: dict[str, str] = field(default_factory=lambda: {"name": "voltage-verify", "version": __version__})

    def to_json(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "tool": self.tool,
            "created_at": self.created_at.isoformat(timespec="seconds"),
            "manifest": self.manifest,
            "commitments": self.commitments,
            "tdx": {"quote_b64": base64.b64encode(self.quote).decode("ascii"), "report_data_hex": self.report_data.hex()},
            "nvidia": {"mode": self.nvidia_mode, "nras": self.nras, "jwks": self.jwks},
            "collateral": self.collateral,
            "environment": self.environment,
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_json(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def from_json(cls, data: Any) -> Bundle:
        if not isinstance(data, dict):
            raise BundleError("bundle is not a JSON object")
        if data.get("format") != FORMAT:
            raise BundleError(f"unsupported bundle format {repr(data.get('format'))[:40]}")
        # The sections the verifier reads as objects must be objects: a list or a string in their
        # place is a malformed bundle, not an exception further down.
        for name in ("manifest", "tdx", "nvidia"):
            if not isinstance(data.get(name), dict):
                raise BundleError(f"bundle field {name!r} is missing or not an object")
        if data.get("collateral") is not None and not isinstance(data["collateral"], dict):
            raise BundleError("bundle field 'collateral' is not an object")
        try:
            created = datetime.fromisoformat(str(data["created_at"]).replace("Z", "+00:00"))
            if created.tzinfo is None:
                raise ValueError("created_at has no UTC offset")
            return cls(
                manifest=data["manifest"],
                commitments=dict(data["commitments"]),
                quote=base64.b64decode(data["tdx"]["quote_b64"]),
                report_data=bytes.fromhex(data["tdx"]["report_data_hex"]),
                nvidia_mode=str(data["nvidia"]["mode"]),
                nras=data["nvidia"]["nras"],
                jwks=data["nvidia"].get("jwks"),
                collateral=data.get("collateral"),
                environment=dict(data.get("environment") or {}),
                created_at=created,
                tool=dict(data.get("tool") or {}),
            )
        except (KeyError, TypeError, ValueError) as err:
            raise BundleError(f"bundle is missing or has a malformed field: {err}") from err

    @classmethod
    def load(cls, path: Path) -> Bundle:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError) as err:
            raise BundleError(f"cannot read {path}: {type(err).__name__}: {str(err)[:200]}") from err
        return cls.from_json(data)

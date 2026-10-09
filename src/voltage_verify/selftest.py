# SPDX-License-Identifier: MIT
"""Negative tests: prove the verifier rejects what it must reject.

Each mutation takes a bundle that verifies cleanly and changes one thing an attacker might
change. Where the manifest is altered, the bundle's commitments are recomputed as a
sophisticated attacker would, so the failure has to come from the hardware-signed evidence
no longer matching, not from a bookkeeping field.
"""

from __future__ import annotations

import base64
import copy
import json
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from . import nvidia
from .bundle import Bundle
from .canonical import commitments
from .verify import Report, verify_bundle


@dataclass
class Mutation:
    name: str
    description: str
    apply: Callable[[Bundle], Bundle]
    must_fail: tuple[str, ...]  # at least one of these checks must fail


def _with_manifest(bundle: Bundle, manifest: dict[str, Any]) -> Bundle:
    com = commitments(manifest)
    clone = copy.deepcopy(bundle)
    clone.manifest = manifest
    clone.commitments = {"sha256": com.sha256_hex, "sha512": com.sha512_hex}
    return clone


def _flip_hex(value: str, position: int = -1) -> str:
    chars = list(value)
    chars[position] = "0" if chars[position] != "0" else "1"
    return "".join(chars)


def mutate_image_digest(bundle: Bundle) -> Bundle:
    manifest = copy.deepcopy(bundle.manifest)
    image = manifest["workload"].get("image") or {"reference": "example", "digest": "sha256:" + "0" * 64}
    image["digest"] = _flip_hex(image["digest"])
    manifest["workload"]["image"] = image
    return _with_manifest(bundle, manifest)


def mutate_artifact(bundle: Bundle) -> Bundle:
    manifest = copy.deepcopy(bundle.manifest)
    arts = manifest["workload"].get("artifacts") or []
    if arts:
        arts[0]["sha256"] = _flip_hex(arts[0]["sha256"])
    else:
        arts.append({"name": "injected.bin", "sha256": "0" * 64, "size": 1})
    manifest["workload"]["artifacts"] = arts
    return _with_manifest(bundle, manifest)


def mutate_manifest_field(bundle: Bundle) -> Bundle:
    manifest = copy.deepcopy(bundle.manifest)
    manifest["workload"]["extra"] = {**manifest["workload"].get("extra", {}), "injected": "value"}
    return _with_manifest(bundle, manifest)


def mutate_challenge(bundle: Bundle) -> Bundle:
    manifest = copy.deepcopy(bundle.manifest)
    manifest["challenge"] = secrets.token_hex(32)
    return _with_manifest(bundle, manifest)


def mutate_quote(bundle: Bundle) -> Bundle:
    clone = copy.deepcopy(bundle)
    raw = bytearray(clone.quote)
    raw[48 + 136] ^= 0x01  # first byte of MRTD inside the TD report
    clone.quote = bytes(raw)
    return clone


def mutate_nras(bundle: Bundle) -> Bundle:
    clone = copy.deepcopy(bundle)
    nras = json.loads(json.dumps(clone.nras))
    devices = nras[1]["REMOTE_GPU_CLAIMS"][1]
    name = sorted(devices)[0]
    header, payload, signature = devices[name].split(".")
    body = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    body["dbgstat"] = "enabled"
    forged = base64.urlsafe_b64encode(json.dumps(body, separators=(",", ":")).encode()).decode().rstrip("=")
    devices[name] = ".".join([header, forged, signature])
    clone.nras = nras
    return clone


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def mutate_nras_own_key(bundle: Bundle) -> Bundle:
    """Re-sign every NRAS token with an attacker key and ship that key in the bundle's JWKS.

    The forged JWKS entry keeps NVIDIA's real x5c chain, so the certificate subjects still
    read "NVIDIA", while its JWK coordinates are the attacker's. Only a verifier that binds
    the JWK to a leaf certificate chained to a pinned NVIDIA anchor rejects it.
    """
    from cryptography.hazmat.primitives.asymmetric import ec
    from jwt.algorithms import ECAlgorithm

    clone = copy.deepcopy(bundle)
    priv = ec.generate_private_key(ec.SECP384R1())
    kid = "forged-" + secrets.token_hex(8)
    jwk = json.loads(ECAlgorithm.to_jwk(priv.public_key()))
    real = next((k for k in (bundle.jwks or {}).get("keys", []) if k.get("x5c")), None)
    jwk.update({"kid": kid, "alg": "ES384", "use": "sig"})
    if real:
        jwk["x5c"] = real["x5c"]

    def resign(token: str) -> str:
        header, payload, _ = token.split(".")
        head = json.loads(base64.urlsafe_b64decode(header + "=" * (-len(header) % 4)))
        head.update({"kid": kid, "alg": "ES384"})
        signing_input = f"{_b64url(json.dumps(head, separators=(',', ':')).encode())}.{payload}"
        sig = ECAlgorithm(ECAlgorithm.SHA384).sign(signing_input.encode(), priv)
        return f"{signing_input}.{_b64url(sig)}"

    nras = json.loads(json.dumps(clone.nras))
    nras[0][1] = resign(nras[0][1])
    claims = nras[1]["REMOTE_GPU_CLAIMS"]
    claims[0][1] = resign(claims[0][1])
    claims[1] = {name: resign(tok) for name, tok in claims[1].items()}
    clone.nras = nras
    clone.jwks = {"keys": [jwk]}
    return clone


def _signing_kid(bundle: Bundle) -> str | None:
    try:
        return nvidia.unverified_header(nvidia.split_tokens(bundle.nras)[1]).get("kid")
    except nvidia.NrasError:
        return None


def mutate_jwks_duplicate_kid(bundle: Bundle) -> Bundle:
    """List the signing key id a second time, with attacker coordinates, after the real entry."""
    from cryptography.hazmat.primitives.asymmetric import ec
    from jwt.algorithms import ECAlgorithm

    clone = copy.deepcopy(bundle)
    keys = list((bundle.jwks or {}).get("keys") or [])
    kid = _signing_kid(bundle)
    real = next((k for k in keys if isinstance(k, dict) and k.get("kid") == kid), {"kid": kid, "kty": "EC"})
    attacker = json.loads(ECAlgorithm.to_jwk(ec.generate_private_key(ec.SECP384R1()).public_key()))
    clone.jwks = {"keys": [*keys, {**real, "x": attacker["x"], "y": attacker["y"]}]}
    return clone


def mutate_jwks_no_chain(bundle: Bundle) -> Bundle:
    """Strip the x5c certificate chain from every key of the bundle's JWKS."""
    clone = copy.deepcopy(bundle)
    keys = (bundle.jwks or {}).get("keys") or []
    clone.jwks = {"keys": [{f: v for f, v in k.items() if f != "x5c"} for k in keys if isinstance(k, dict)]}
    return clone


BOUND = ("tdx.report_data", "nvidia.nonce")
MUTATIONS: list[Mutation] = [
    Mutation("image-digest", "container image digest changed, commitments recomputed", mutate_image_digest, BOUND),
    Mutation("artifact", "model or artifact digest changed, commitments recomputed", mutate_artifact, BOUND),
    Mutation("manifest-field", "another manifest field changed, commitments recomputed", mutate_manifest_field, BOUND),
    Mutation(
        "challenge-replay",
        "challenge replaced by a fresh one (replay of old evidence)",
        mutate_challenge,
        (*BOUND, "manifest.challenge"),
    ),
    Mutation("quote-tamper", "one byte of the TDX quote body altered", mutate_quote, ("tdx.signatures",)),
    Mutation("nras-tamper", "one claim inside a GPU token altered", mutate_nras, ("nvidia.signatures",)),
    Mutation(
        "nras-own-key",
        "every NRAS token re-signed with an attacker key shipped in the bundle JWKS",
        mutate_nras_own_key,
        ("nvidia.signatures",),
    ),
    Mutation(
        "jwks-duplicate-kid",
        "signing key id listed twice in the bundle JWKS, second copy with attacker coordinates",
        mutate_jwks_duplicate_kid,
        ("nvidia.signatures", "nvidia.jwks"),
    ),
    Mutation(
        "jwks-no-chain",
        "x5c certificate chain stripped from every key of the bundle JWKS",
        mutate_jwks_no_chain,
        ("nvidia.signatures", "nvidia.jwks"),
    ),
]


@dataclass
class SelfTestResult:
    name: str
    description: str
    passed: bool
    failed_checks: list[str]


def run_selftest(
    bundle: Bundle, online: bool = False, expected_challenge: str | None = None
) -> tuple[Report, list[SelfTestResult]]:
    """Return the clean report and one result per mutation (passed = verifier rejected it)."""
    clean = verify_bundle(bundle, online=online, expected_challenge=expected_challenge)
    results: list[SelfTestResult] = []
    for mutation in MUTATIONS:
        mutated = mutation.apply(bundle)
        rep = verify_bundle(mutated, online=online, expected_challenge=expected_challenge)
        failed = rep.failed()
        rejected = any(f in failed for f in mutation.must_fail) and not rep.ok
        results.append(SelfTestResult(mutation.name, mutation.description, rejected, failed))
    return clean, results

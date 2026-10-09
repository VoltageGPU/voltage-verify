# SPDX-License-Identifier: MIT
"""NVIDIA Remote Attestation Service (NRAS) token verification.

The nv-attestation-sdk stores the NRAS answer as a JSON array::

    [["JWT", <overall token>],
     {"REMOTE_GPU_CLAIMS": [["JWT", <verifier token>], {"GPU-0": <device token>, ...}]}]

The verifier token and every device token are JWTs signed by NVIDIA (ES384) with a key
published in NRAS's JWKS. This module verifies those signatures, reads the claims, checks
that ``eat_nonce`` equals the workload commitment, and applies a small policy on the
per-GPU claims (measurements succeeded, secure boot on, debug off, nonce matched, report
signature and certificate chain validated by NVIDIA).

Signing keys are taken from the bundle's own JWKS only, and only through their x5c chain:
the leaf must be issued by the pinned NRAS intermediate (or by an intermediate that NVIDIA's
live JWKS itself presents, to survive a rotation), name NVIDIA Corporation, hold exactly the
JWK's key and have been valid when the token was issued. A JWKS listing a key id twice is
refused. These are the rules of the website verifier (voltagegpu.com, lib/verified/nras.ts).

A verification proves that NVIDIA issued these claims for that GPU at that time, against
that nonce. It does not name the operator of the machine; the TDX quote and the fresh
challenge take care of that side.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import jwt
from cryptography import x509
from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from jwt.algorithms import ECAlgorithm

NRAS_ISSUER = "https://nras.attestation.nvidia.com"
NRAS_JWKS_URL = "https://nras.attestation.nvidia.com/.well-known/jwks.json"
# JWT algorithm -> the only curve its key may be on.
ALG_CURVES: dict[str, str] = {"ES384": "secp384r1", "ES256": "secp256r1"}
ALLOWED_ALGS = tuple(ALG_CURVES)
JWK_CURVES = frozenset({"P-384", "P-256"})
NVIDIA_ORG = "NVIDIA Corporation"
MAX_KID_LEN = 200
MAX_TOKEN_LEN = 16 * 1024  # real NRAS tokens are under 2 KiB
_B64URL = re.compile(r"[A-Za-z0-9_-]+")

PINNED_INTERMEDIATE = "pinned-intermediate"
LIVE_INTERMEDIATE = "nvidia-live-intermediate"

# SHA-256 of the SubjectPublicKeyInfo of the NRAS intermediate CA that tops every x5c chain.
# The root above it ("NVIDIA Attestation Service CA 001") is offline and never shipped in the
# chain, so the intermediate is the trust anchor. Pinned from the live NRAS JWKS over HTTPS on
# 23 September 2026 and cross-checked against confidential-dot-ai/attestation-rs.
# CN=NVIDIA Attestation Service GPU Intermediate 004, valid 2025-12-08 to 2029-12-08.
NRAS_TRUSTED_SPKI_SHA256 = frozenset(
    {"fd32837f954e2c45db073105166dfe6985ae0480bb113fba63b091a75affe896"}
)

DEVICE_POLICY: dict[str, Any] = {
    "measres": "success",
    "secboot": True,
    "dbgstat": "disabled",
    "x-nvidia-gpu-attestation-report-nonce-match": True,
    "x-nvidia-gpu-attestation-report-signature-verified": True,
    "x-nvidia-gpu-attestation-report-cert-chain-validated": True,
}


class NrasError(ValueError):
    """Raised when the token bundle is malformed or fails verification."""


@dataclass
class VerifiedToken:
    kind: str  # "overall" | "verifier" | "device"
    name: str
    header: dict[str, Any]
    claims: dict[str, Any]
    signed_by_nvidia: bool
    kid: str | None
    x5c_subjects: list[str] = field(default_factory=list)
    key_source: str = PINNED_INTERMEDIATE  # which NVIDIA intermediate anchored the key


def _b64url_json(segment: str, what: str) -> dict[str, Any]:
    if not _B64URL.fullmatch(segment):
        raise NrasError(f"{what} is not base64url")
    try:
        value = json.loads(base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4)))
    except (ValueError, TypeError, RecursionError) as err:
        raise NrasError(f"{what} is not base64url JSON") from err
    if not isinstance(value, dict):
        raise NrasError(f"{what} is not a JSON object")
    return value


def _segments(token: Any) -> list[str]:
    parts = token.split(".") if isinstance(token, str) else []
    if len(parts) != 3:
        raise NrasError("token is not a compact JWS")
    return parts


def unverified_header(token: str) -> dict[str, Any]:
    return _b64url_json(_segments(token)[0], "JWT header")


def unverified_claims(token: str) -> dict[str, Any]:
    return _b64url_json(_segments(token)[1], "JWT payload")


def split_tokens(payload: Any) -> tuple[str, str, dict[str, str]]:
    """Return (overall, verifier, {device_name: token}) from the SDK's JSON array."""
    try:
        overall = payload[0][1]
        claims = payload[1]["REMOTE_GPU_CLAIMS"]
        verifier = claims[0][1]
        devices = dict(claims[1])
    except (IndexError, KeyError, TypeError, ValueError) as err:
        raise NrasError(f"unexpected NRAS token layout: {err}") from err
    if not isinstance(overall, str) or not isinstance(verifier, str) or not devices:
        raise NrasError("NRAS token layout has the wrong types")
    if not all(isinstance(n, str) and isinstance(t, str) for n, t in devices.items()):
        raise NrasError("NRAS device tokens must be a map of GPU name to token")
    return overall, verifier, devices


def _spki_sha256(cert: x509.Certificate) -> str:
    der = cert.public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    return hashlib.sha256(der).hexdigest()


def _spki_der(key: Any) -> bytes:
    return key.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)


def live_intermediates(live_jwks: Any) -> frozenset[str]:
    """SPKI SHA-256 of the intermediates NVIDIA's live JWKS presents (``x5c[1]`` of each key).

    The live JWKS is fetched from nvidia.com over TLS. It is used for this only: to accept an
    intermediate NVIDIA rotated in after this release. It never supplies a signing key, and a
    malformed entry adds nothing.
    """
    keys = live_jwks.get("keys") if isinstance(live_jwks, dict) else None
    out: set[str] = set()
    for key in keys if isinstance(keys, list) else []:
        x5c = key.get("x5c") if isinstance(key, dict) else None
        if not isinstance(x5c, list) or len(x5c) < 2 or not isinstance(x5c[1], str):
            continue
        try:
            out.add(_spki_sha256(x509.load_der_x509_certificate(base64.b64decode(x5c[1], validate=True))))
        except (ValueError, TypeError, UnsupportedAlgorithm):
            continue
    return frozenset(out)


def _jwk_public_key(key: dict[str, Any], kid: str | None) -> ec.EllipticCurvePublicKey:
    """The EC public key a JWK declares, built from kty/crv/x/y only (never a private part)."""
    crv = key.get("crv")
    if key.get("kty") != "EC" or not isinstance(crv, str) or crv not in JWK_CURVES:
        raise NrasError(f"JWKS key {kid!r} is not an EC P-384/P-256 public key")
    try:
        declared = ECAlgorithm.from_jwk(json.dumps({f: key.get(f) for f in ("kty", "crv", "x", "y")}))
    except (jwt.PyJWTError, ValueError, TypeError) as err:
        raise NrasError(f"JWKS key {kid!r} is not an EC P-384/P-256 public key: {err}") from err
    if not isinstance(declared, ec.EllipticCurvePublicKey):
        raise NrasError(f"JWKS key {kid!r} is not an EC public key")
    return declared


def _load_chain(
    key: dict[str, Any], kid: str | None, live_anchors: frozenset[str] = frozenset()
) -> list[x509.Certificate]:
    """Return ``[leaf, intermediate]`` once the JWKS entry's x5c chain is checked.

    The key a JWKS entry carries is only trusted if ``x5c[0]`` is a leaf certificate issued by
    ``x5c[1]``, that intermediate is a CA whose key is pinned here (or presented by NVIDIA's
    live JWKS, ``live_anchors``), the leaf names NVIDIA Corporation and holds exactly the
    public key the JWK declares. Raw (x, y) coordinates are never accepted on their own.
    """
    x5c = key.get("x5c")
    if not x5c:
        raise NrasError(f"JWKS key {kid!r} has no x5c chain; bare JWK coordinates are not trusted")
    if not isinstance(x5c, list) or not all(isinstance(c, str) for c in x5c):
        raise NrasError(f"JWKS key {kid!r}: x5c is not a list of base64 certificates")
    if len(x5c) < 2:
        raise NrasError(f"JWKS key {kid!r}: x5c chain is incomplete, it needs the leaf and the NVIDIA intermediate")
    try:
        leaf, issuer = (x509.load_der_x509_certificate(base64.b64decode(c, validate=True)) for c in x5c[:2])
        anchor = _spki_sha256(issuer)
    except (ValueError, UnsupportedAlgorithm) as err:
        raise NrasError(f"JWKS key {kid!r}: unreadable x5c certificate: {err}") from err
    if anchor not in NRAS_TRUSTED_SPKI_SHA256 and anchor not in live_anchors:
        raise NrasError(
            f"JWKS key {kid!r}: x5c chain does not lead to the pinned NVIDIA intermediate "
            f"or one NVIDIA's live JWKS presents (SPKI {anchor[:16]}...)"
        )
    try:
        bc = issuer.extensions.get_extension_for_class(x509.BasicConstraints).value
    except x509.ExtensionNotFound:
        bc = None
    except ValueError as err:  # undecodable extensions
        raise NrasError(f"JWKS key {kid!r}: unreadable intermediate certificate extensions: {err}") from err
    if bc is None or not bc.ca:
        raise NrasError(f"JWKS key {kid!r}: {issuer.subject.rfc4514_string()} is not a CA")
    try:
        leaf.verify_directly_issued_by(issuer)
        leaf_spki = _spki_der(leaf.public_key())
    except (ValueError, TypeError, InvalidSignature, UnsupportedAlgorithm) as err:
        raise NrasError(f"JWKS key {kid!r}: leaf certificate is not signed by the NVIDIA intermediate: {err}") from err
    orgs = [a.value for a in leaf.subject.get_attributes_for_oid(NameOID.ORGANIZATION_NAME)]
    if NVIDIA_ORG not in orgs:
        raise NrasError(f"JWKS key {kid!r}: leaf certificate subject is not {NVIDIA_ORG}")
    if _spki_der(_jwk_public_key(key, kid)) != leaf_spki:
        raise NrasError(f"JWKS key {kid!r}: JWK coordinates do not match the leaf certificate")
    return [leaf, issuer]


def _key_for(
    jwks: Any, kid: str, live_anchors: frozenset[str] = frozenset()
) -> tuple[ec.EllipticCurvePublicKey, list[x509.Certificate], str]:
    """Resolve ``kid`` in the bundle's JWKS: exactly one entry, with a valid NVIDIA chain."""
    keys = jwks.get("keys") if isinstance(jwks, dict) else None
    if not isinstance(keys, list):
        raise NrasError("the bundle's JWKS has no list of keys")
    matches = [k for k in keys if isinstance(k, dict) and k.get("kid") == kid]
    if not matches:
        raise NrasError(f"no JWKS key in the bundle matches kid {kid!r}: the bundle must carry every signing key")
    if len(matches) > 1:
        raise NrasError(f"the bundle's JWKS lists key {kid!r} more than once")
    chain = _load_chain(matches[0], kid, live_anchors)
    leaf_key = chain[0].public_key()
    if not isinstance(leaf_key, ec.EllipticCurvePublicKey):
        raise NrasError(f"JWKS key {kid!r}: leaf certificate does not hold an EC key")
    source = PINNED_INTERMEDIATE if _spki_sha256(chain[1]) in NRAS_TRUSTED_SPKI_SHA256 else LIVE_INTERMEDIATE
    return leaf_key, chain, source


def _check_validity(chain: list[x509.Certificate], at: datetime | None, name: str) -> None:
    """Every certificate must have been valid when NVIDIA signed the token.

    NRAS leaf certificates live about two days, so an archived bundle is judged at the token's
    own ``iat``, which the signature covers, not at the time of verification.
    """
    if at is None:
        raise NrasError(f"{name}: token has no iat, cannot place it inside the certificate validity")
    for cert in chain:
        if not cert.not_valid_before_utc <= at <= cert.not_valid_after_utc:
            raise NrasError(
                f"{name}: token issued {at.isoformat()} outside the validity of {cert.subject.rfc4514_string()}"
            )


def verify_token(
    token: str, jwks: dict[str, Any], kind: str, name: str, live_anchors: frozenset[str] = frozenset()
) -> VerifiedToken:
    """Verify one JWT signature against the bundle's JWKS and return its claims.

    ``jwks`` is the JWKS carried by the bundle, the only source of signing keys.
    ``live_anchors`` (see ``live_intermediates``) may only widen the accepted intermediates.
    """
    if isinstance(token, str) and len(token) > MAX_TOKEN_LEN:
        raise NrasError(f"{name}: token is too large")
    header = unverified_header(token)
    alg = header.get("alg")
    # isinstance first: a list or dict "alg" must be refused, not raise TypeError on lookup.
    if not isinstance(alg, str) or alg not in ALG_CURVES:
        raise NrasError(f"{name}: algorithm {str(alg)[:16]!r} is not allowed")
    curve = ALG_CURVES[alg]
    kid = header.get("kid")
    if not isinstance(kid, str) or not kid or len(kid) > MAX_KID_LEN:
        raise NrasError(f"{name}: token has no usable key id (kid)")
    key, chain, source = _key_for(jwks, kid, live_anchors)
    if key.curve.name != curve:
        raise NrasError(f"{name}: key curve {key.curve.name} does not match {alg}")
    try:
        claims = jwt.decode(
            token,
            key=key,
            algorithms=[alg],
            options={"verify_exp": False, "verify_aud": False, "verify_iat": False, "verify_nbf": False},
        )
    except jwt.PyJWTError as err:
        raise NrasError(f"{name}: signature does not verify: {err}") from err
    if claims.get("iss") != NRAS_ISSUER:
        raise NrasError(f"{name}: issuer {claims.get('iss')!r} is not NRAS")
    _check_validity(chain, _ts(claims.get("iat")), name)
    subjects = [c.subject.rfc4514_string() for c in chain]
    return VerifiedToken(kind, name, header, claims, True, kid, subjects, source)


@dataclass
class NrasResult:
    verifier: VerifiedToken
    devices: list[VerifiedToken]
    overall_claims: dict[str, Any]
    nonce: str
    issued_at: datetime | None
    expires_at: datetime | None
    policy_failures: list[str]

    @property
    def overall_ok(self) -> bool:
        return bool(self.verifier.claims.get("x-nvidia-overall-att-result")) and not self.policy_failures


def _ts(value: Any) -> datetime | None:
    try:
        return datetime.fromtimestamp(int(value), tz=timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def verify_nras(
    payload: Any,
    jwks: dict[str, Any],
    expected_nonce_hex: str,
    require_devices: int | None = None,
    allowed_hwmodels: set[str] | None = None,
    live_anchors: frozenset[str] = frozenset(),
) -> NrasResult:
    """Verify every NVIDIA-signed token in ``payload`` against the bundle's ``jwks`` and apply
    the claims policy. ``live_anchors``: intermediates NVIDIA's live JWKS presents, if fetched."""
    overall, verifier_tok, devices = split_tokens(payload)
    failures: list[str] = []
    verifier = verify_token(verifier_tok, jwks, "verifier", "REMOTE_GPU_CLAIMS", live_anchors)
    nonce = str(verifier.claims.get("eat_nonce", "")).lower()
    if nonce != expected_nonce_hex.lower():
        failures.append(f"eat_nonce {nonce[:16]}... does not match the commitment {expected_nonce_hex[:16]}...")
    if verifier.claims.get("x-nvidia-overall-att-result") is not True:
        failures.append("x-nvidia-overall-att-result is not true")

    verified_devices: list[VerifiedToken] = []
    for name, tok in sorted(devices.items()):
        dev = verify_token(tok, jwks, "device", name, live_anchors)
        for claim, wanted in DEVICE_POLICY.items():
            if dev.claims.get(claim) != wanted:
                failures.append(f"{name}: {claim} is {dev.claims.get(claim)!r}, expected {wanted!r}")
        if allowed_hwmodels and dev.claims.get("hwmodel") not in allowed_hwmodels:
            failures.append(f"{name}: hwmodel {dev.claims.get('hwmodel')!r} not in {sorted(allowed_hwmodels)}")
        verified_devices.append(dev)
    if require_devices is not None and len(verified_devices) != require_devices:
        failures.append(f"expected {require_devices} attested GPU(s), got {len(verified_devices)}")

    return NrasResult(
        verifier=verifier,
        devices=verified_devices,
        overall_claims=unverified_claims(overall),
        nonce=nonce,
        issued_at=_ts(verifier.claims.get("iat")),
        expires_at=_ts(verifier.claims.get("exp")),
        policy_failures=failures,
    )


def jwks_fingerprints(jwks: dict[str, Any]) -> dict[str, str]:
    """SHA-256 fingerprint of the leaf certificate of every JWKS key, keyed by kid."""
    out: dict[str, str] = {}
    for key in jwks.get("keys", []):
        x5c = key.get("x5c") or []
        if not x5c:
            continue
        cert = x509.load_der_x509_certificate(base64.b64decode(x5c[0]))
        out[key.get("kid", "")] = cert.fingerprint(hashes.SHA256()).hex()
    return out

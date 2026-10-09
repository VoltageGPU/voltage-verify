# SPDX-License-Identifier: MIT
"""The NRAS key rules of the website verifier (voltagegpu.com, lib/verified/nras.ts), ported.

0.2.0 still accepted a JWKS that lists the signing key id twice (the first entry won), took
signing keys from NVIDIA's live JWKS instead of the bundle when run online, and accepted an
x5c of a single certificate. Signing keys now come from the bundle only; NVIDIA's live JWKS
can only add an intermediate (to survive a rotation), never a key.
"""

from __future__ import annotations

import base64
import copy
import datetime as dt
import hashlib
import hmac
import json
from pathlib import Path

import jwt
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from jwt.algorithms import ECAlgorithm

from voltage_verify import nvidia, pcs
from voltage_verify import verify as verify_mod
from voltage_verify.bundle import Bundle
from voltage_verify.selftest import MUTATIONS
from voltage_verify.verify import verify_bundle

EXAMPLE = Path(__file__).parent.parent / "examples" / "hello-workload" / "bundle-8xh100-2026-09-12.json"


@pytest.fixture(scope="module")
def example() -> Bundle:
    return Bundle.load(EXAMPLE)


def _verifier(bundle: Bundle) -> str:
    return nvidia.split_tokens(bundle.nras)[1]


def _signing_key(bundle: Bundle) -> dict:
    kid = nvidia.unverified_header(_verifier(bundle))["kid"]
    return next(k for k in bundle.jwks["keys"] if k["kid"] == kid)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _with_header(token: str, header: dict) -> str:
    _, payload, sig = token.split(".")
    return ".".join([_b64url(json.dumps(header).encode()), payload, sig])


def _attacker_coordinates(jwk: dict) -> dict:
    pub = json.loads(ECAlgorithm.to_jwk(ec.generate_private_key(ec.SECP384R1()).public_key()))
    return {**jwk, "x": pub["x"], "y": pub["y"]}


# --- look-alike certificate chains, made here, never signed by NVIDIA ---------------------


def _name(cn: str, org: str = "NVIDIA Corporation") -> x509.Name:
    return x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, org),
            x509.NameAttribute(NameOID.COMMON_NAME, cn),
        ]
    )


def _cert(subject: x509.Name, issuer: x509.Name, public_key, signer, ca: bool) -> x509.Certificate:
    now = dt.datetime.now(dt.timezone.utc)
    return (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(public_key)
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(days=1))
        .not_valid_after(now + dt.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
        .sign(signer, hashes.SHA384())
    )


def _der_b64(cert: x509.Certificate) -> str:
    return base64.b64encode(cert.public_bytes(serialization.Encoding.DER)).decode()


KID = "lookalike-kid"


def _lookalike_key(org: str = "NVIDIA Corporation"):
    """A leaf and an intermediate named exactly like NVIDIA's, issued by a key made here."""
    ca_key = ec.generate_private_key(ec.SECP384R1())
    leaf_key = ec.generate_private_key(ec.SECP384R1())
    inter = _cert(
        _name("NVIDIA Attestation Service GPU Intermediate 004"),
        _name("NVIDIA Attestation Service CA 001"),
        ca_key.public_key(),
        ca_key,
        ca=True,
    )
    leaf = _cert(_name("NVIDIA Attestation Service GPU GH100", org), inter.subject, leaf_key.public_key(), ca_key, ca=False)
    jwk = json.loads(ECAlgorithm.to_jwk(leaf_key.public_key()))
    jwk.update({"kid": KID, "alg": "ES384", "use": "sig", "x5c": [_der_b64(leaf), _der_b64(inter)]})
    return leaf_key, jwk


def _lookalike(org: str = "NVIDIA Corporation"):
    leaf_key, jwk = _lookalike_key(org)
    token = jwt.encode(
        {"iss": nvidia.NRAS_ISSUER, "iat": int(dt.datetime.now(dt.timezone.utc).timestamp())},
        leaf_key,
        algorithm="ES384",
        headers={"kid": KID},
    )
    return token, jwk


def _resign_all(bundle: Bundle, priv, jwk: dict) -> Bundle:
    """Every NVIDIA token of the bundle re-signed with ``priv``, the bundle JWKS replaced by ``jwk``."""

    def resign(token: str) -> str:
        signing_input = f"{_b64url(json.dumps({'alg': 'ES384', 'kid': jwk['kid']}).encode())}.{token.split('.')[1]}"
        return f"{signing_input}.{_b64url(ECAlgorithm(ECAlgorithm.SHA384).sign(signing_input.encode(), priv))}"

    clone = copy.deepcopy(bundle)
    nras = json.loads(json.dumps(bundle.nras))
    claims = nras[1]["REMOTE_GPU_CLAIMS"]
    claims[0][1] = resign(claims[0][1])
    claims[1] = {name: resign(tok) for name, tok in claims[1].items()}
    clone.nras = nras
    clone.jwks = {"keys": [jwk]}
    return clone


# --- 1. a JWKS that lists the signing key id twice ----------------------------------------


@pytest.mark.parametrize("order", ["real-real", "real-forged", "forged-real"])
def test_jwks_listing_the_signing_kid_twice_is_refused(example, order):
    real = _signing_key(example)
    forged = _attacker_coordinates(real)
    pair = {"real-real": [real, real], "real-forged": [real, forged], "forged-real": [forged, real]}[order]
    others = [k for k in example.jwks["keys"] if k["kid"] != real["kid"]]
    with pytest.raises(nvidia.NrasError, match="more than once"):
        nvidia.verify_nras(example.nras, {"keys": [*others, *pair]}, example.commitments["sha256"])


def test_bundle_with_a_duplicated_kid_does_not_verify(example):
    doubled = copy.deepcopy(example)
    doubled.jwks = {"keys": [*example.jwks["keys"], _signing_key(example)]}
    report = verify_bundle(doubled, online=False)
    assert not report.ok and "nvidia.signatures" in report.failed()


# --- 2. keys only in NVIDIA's live JWKS, not carried by the bundle ------------------------


@pytest.fixture
def live(monkeypatch, example):
    """Run verify_bundle(online=True) without the network: Intel collateral from the bundle,
    NVIDIA's live JWKS set by each test (a dict, or an exception to simulate an outage)."""
    state: dict = {"jwks": copy.deepcopy(example.jwks)}

    def fake_fetch_jwks():
        if isinstance(state["jwks"], Exception):
            raise state["jwks"]
        return state["jwks"]

    monkeypatch.setattr(verify_mod, "fetch_jwks", fake_fetch_jwks)
    monkeypatch.setattr(pcs, "fetch_collateral", lambda *a, **k: pcs.Collateral.from_json(example.collateral["intel"]))
    return state


def _detail(report, name: str) -> str:
    return " ".join(c.detail for c in report.checks if c.name == name)


def test_bundle_without_jwks_is_refused_even_if_nvidia_publishes_the_key(example, live):
    bare = copy.deepcopy(example)
    bare.jwks = None
    report = verify_bundle(bare, online=True)
    assert not report.ok and "nvidia.jwks" in report.failed()


def test_key_only_in_the_live_jwks_is_refused(example, live):
    empty = copy.deepcopy(example)
    empty.jwks = {"keys": []}
    report = verify_bundle(empty, online=True)
    assert not report.ok and "nvidia.signatures" in report.failed()
    assert "bundle" in _detail(report, "nvidia.signatures")


def test_bundle_keys_without_chain_are_refused_even_if_the_live_jwks_has_them(example, live):
    stripped = copy.deepcopy(example)
    stripped.jwks = {"keys": [{k: v for k, v in key.items() if k != "x5c"} for key in example.jwks["keys"]]}
    report = verify_bundle(stripped, online=True)
    assert not report.ok and "nvidia.signatures" in report.failed()


def test_honest_archived_bundle_verifies_online_after_nvidia_rotated_its_keys(example, live):
    live["jwks"] = {"keys": []}  # none of the bundle's keys is still published
    report = verify_bundle(example, online=True)
    assert report.ok, report.failed()


def test_honest_bundle_verifies_online_when_the_live_jwks_is_unreachable(example, live):
    live["jwks"] = OSError("network down")
    report = verify_bundle(example, online=True)
    assert report.ok, report.failed()


def test_honest_bundle_verifies_online_with_the_live_jwks(example, live):
    report = verify_bundle(example, online=True)
    assert report.ok, report.failed()


# --- 3 and 4. algorithm names --------------------------------------------------------------


@pytest.mark.parametrize(
    "alg",
    [
        "toString", "__proto__", "constructor", "hasOwnProperty", "__class__", "__init__", "get", "keys",
        "", None, 384, ["ES384"], {"ES384": True},
    ],
)
def test_algorithm_names_outside_the_allow_list_are_refused(example, alg):
    tok = _with_header(_verifier(example), {"alg": alg, "kid": _signing_key(example)["kid"]})
    with pytest.raises(nvidia.NrasError, match="not allowed"):
        nvidia.verify_token(tok, example.jwks, "verifier", "x")


@pytest.mark.parametrize("alg", ["none", "None", "NONE", "HS256", "HS384", "RS256", "PS384", "EdDSA"])
def test_alg_none_hmac_and_rsa_are_refused(example, alg):
    real = _signing_key(example)
    header = _b64url(json.dumps({"alg": alg, "kid": real["kid"]}).encode())
    payload = _verifier(example).split(".")[1]
    # HMAC keyed with NVIDIA's public certificate: the classic key-confusion forgery.
    secret = base64.b64decode(real["x5c"][0])
    sig = _b64url(hmac.new(secret, f"{header}.{payload}".encode(), hashlib.sha256).digest())
    with pytest.raises(nvidia.NrasError, match="not allowed"):
        nvidia.verify_token(f"{header}.{payload}.{sig}", example.jwks, "verifier", "x")


def test_es256_header_on_a_p384_key_is_refused(example):
    tok = _with_header(_verifier(example), {"alg": "ES256", "kid": _signing_key(example)["kid"]})
    with pytest.raises(nvidia.NrasError, match="curve"):
        nvidia.verify_token(tok, example.jwks, "verifier", "x")


@pytest.mark.parametrize("kid", [None, "", 123, ["k"], "k" * 201])
def test_token_without_a_usable_key_id_is_refused(example, kid):
    header = {"alg": "ES384"} if kid is None else {"alg": "ES384", "kid": kid}
    with pytest.raises(nvidia.NrasError, match="key id"):
        nvidia.verify_token(_with_header(_verifier(example), header), example.jwks, "verifier", "x")


@pytest.mark.parametrize("token", ["a.b", "!!!.x.y", _b64url(b"[1]") + ".e30.c2ln", _b64url(b"not json") + ".e30.c2ln"])
def test_malformed_token_is_an_nras_error_not_a_crash(example, token):
    with pytest.raises(nvidia.NrasError):
        nvidia.verify_token(token, example.jwks, "verifier", "x")


@pytest.mark.parametrize(
    "devices", ["GPU-0", [[1, "a.b.c"], ["GPU-1", "a.b.c"]], {"GPU-0": 7}, [["GPU-0"]]]
)
def test_malformed_device_map_is_an_nras_error_not_a_crash(example, devices):
    payload = json.loads(json.dumps(example.nras))
    payload[1]["REMOTE_GPU_CLAIMS"][1] = devices
    with pytest.raises(nvidia.NrasError):
        nvidia.verify_nras(payload, example.jwks, example.commitments["sha256"])


@pytest.mark.parametrize("value", [10**20, -(10**20), "x", None, float("nan")])
def test_out_of_range_timestamp_reads_as_missing(value):
    assert nvidia._ts(value) is None


@pytest.mark.parametrize("bad", [None, [], "keys", {"keys": "x"}, {"keys": ["junk", 1]}])
def test_malformed_jwks_is_an_nras_error_not_a_crash(example, bad):
    with pytest.raises(nvidia.NrasError):
        nvidia.verify_token(_verifier(example), bad, "verifier", "x")


# --- 5. a bundle key without its certificate chain -----------------------------------------


def test_x5c_of_a_single_certificate_is_refused(example):
    real = _signing_key(example)
    leaf_only = {**real, "x5c": real["x5c"][:1]}
    with pytest.raises(nvidia.NrasError, match="incomplete"):
        nvidia._load_chain(leaf_only, real["kid"])


@pytest.mark.parametrize("x5c", ["MIIB", [123, 456], [None, None], {"0": "a"}])
def test_malformed_x5c_is_an_nras_error_not_a_crash(example, x5c):
    key = {**_signing_key(example), "x5c": x5c}
    with pytest.raises(nvidia.NrasError):
        nvidia._load_chain(key, key["kid"])


def test_jwk_with_a_list_as_curve_is_an_nras_error_not_a_crash(example):
    key = {**_signing_key(example), "crv": ["P-384"]}
    with pytest.raises(nvidia.NrasError):
        nvidia._load_chain(key, key["kid"])


def test_deeply_nested_header_is_an_nras_error_not_a_crash(example):
    header = _b64url(b"[" * 5000 + b"]" * 5000)
    with pytest.raises(nvidia.NrasError):
        nvidia.verify_token(f"{header}.e30.c2ln", example.jwks, "verifier", "x")


def test_oversized_token_is_refused(example):
    with pytest.raises(nvidia.NrasError, match="too large"):
        nvidia.verify_token(_verifier(example) + "A" * (17 * 1024), example.jwks, "verifier", "x")


def _real_intermediate(example) -> x509.Certificate:
    return x509.load_der_x509_certificate(base64.b64decode(_signing_key(example)["x5c"][1]))


def test_leaf_with_an_unsupported_signature_hash_is_an_nras_error(example):
    """A leaf naming the real intermediate as issuer, signed with SHA-1 by a key made here."""
    inter = _real_intermediate(example)
    attacker = ec.generate_private_key(ec.SECP384R1())
    leaf = _cert(_name("NVIDIA Attestation Service GPU GH100"), inter.subject, attacker.public_key(), attacker, ca=False)
    try:
        leaf = (
            x509.CertificateBuilder()
            .subject_name(leaf.subject)
            .issuer_name(inter.subject)
            .public_key(attacker.public_key())
            .serial_number(1)
            .not_valid_before(leaf.not_valid_before_utc)
            .not_valid_after(leaf.not_valid_after_utc)
            .sign(attacker, hashes.SHA1())  # noqa: S303
        )
    except Exception:  # noqa: BLE001
        pytest.skip("this cryptography build refuses to sign with SHA-1")
    jwk = json.loads(ECAlgorithm.to_jwk(attacker.public_key()))
    jwk.update({"kid": "sha1", "x5c": [_der_b64(leaf), _signing_key(example)["x5c"][1]]})
    with pytest.raises(nvidia.NrasError):
        nvidia._load_chain(jwk, "sha1")


def test_pinned_key_in_a_certificate_with_broken_extensions_is_an_nras_error(example):
    """x5c[1] carries the pinned public key but an undecodable BasicConstraints extension."""
    inter = _real_intermediate(example)
    signer = ec.generate_private_key(ec.SECP384R1())
    now = dt.datetime.now(dt.timezone.utc)
    broken = (
        x509.CertificateBuilder()
        .subject_name(inter.subject)
        .issuer_name(inter.issuer)
        .public_key(inter.public_key())
        .serial_number(1)
        .not_valid_before(now - dt.timedelta(days=1))
        .not_valid_after(now + dt.timedelta(days=1))
        .add_extension(
            x509.UnrecognizedExtension(x509.oid.ExtensionOID.BASIC_CONSTRAINTS, b"\x04\x00"), critical=True
        )
        .sign(signer, hashes.SHA384())
    )
    key = {**_signing_key(example), "x5c": [_signing_key(example)["x5c"][0], _der_b64(broken)]}
    with pytest.raises(nvidia.NrasError):
        nvidia._load_chain(key, key["kid"])


def test_jwk_that_is_not_an_ec_public_key_is_refused(example):
    key = {**_signing_key(example), "kty": "RSA"}
    with pytest.raises(nvidia.NrasError, match="EC"):
        nvidia._load_chain(key, key["kid"])


# --- 6. self-made keys with a look-alike x5c -----------------------------------------------


def test_lookalike_chain_is_refused(example):
    token, jwk = _lookalike()
    with pytest.raises(nvidia.NrasError, match="pinned NVIDIA intermediate"):
        nvidia.verify_token(token, {"keys": [jwk]}, "verifier", "x")


def test_forged_key_under_the_real_intermediate_listed_twice_is_refused(example):
    real = _signing_key(example)
    forged = {**_attacker_coordinates(real), "x5c": [real["x5c"][1], real["x5c"][1]]}
    with pytest.raises(nvidia.NrasError):
        nvidia._load_chain(forged, forged["kid"])


def test_lookalike_chain_is_refused_even_with_nvidia_live_anchors(example):
    token, jwk = _lookalike()
    anchors = nvidia.live_intermediates(example.jwks)
    with pytest.raises(nvidia.NrasError, match="pinned NVIDIA intermediate"):
        nvidia.verify_token(token, {"keys": [jwk]}, "verifier", "x", live_anchors=anchors)


@pytest.mark.parametrize("online", [False, True])
def test_bundle_resigned_under_a_lookalike_chain_does_not_verify(example, live, online):
    """The live anchors come from NVIDIA's endpoint only, never from the bundle under test."""
    forged = _resign_all(example, *_lookalike_key())
    report = verify_bundle(forged, online=online)
    assert not report.ok and "nvidia.signatures" in report.failed()


# --- NVIDIA's live JWKS may add an intermediate (rotation), never a key --------------------


def test_intermediate_served_by_nvidia_live_jwks_is_accepted():
    token, jwk = _lookalike()
    anchors = nvidia.live_intermediates({"keys": [jwk]})
    tok = nvidia.verify_token(token, {"keys": [jwk]}, "verifier", "x", live_anchors=anchors)
    assert tok.key_source == "nvidia-live-intermediate"


def test_live_intermediate_still_requires_an_nvidia_leaf():
    token, jwk = _lookalike(org="Example Corp")
    anchors = nvidia.live_intermediates({"keys": [jwk]})
    with pytest.raises(nvidia.NrasError, match="NVIDIA Corporation"):
        nvidia.verify_token(token, {"keys": [jwk]}, "verifier", "x", live_anchors=anchors)


def test_pinned_intermediate_is_reported_as_such(example):
    tok = nvidia.verify_token(_verifier(example), example.jwks, "verifier", "x")
    assert tok.key_source == "pinned-intermediate"


@pytest.mark.parametrize("live_jwks", [None, [], {"keys": "x"}, {"keys": ["junk", {"x5c": "MIIB"}, {"x5c": ["a", "!!"]}]}])
def test_malformed_live_jwks_adds_no_anchor(live_jwks):
    assert nvidia.live_intermediates(live_jwks) == frozenset()


def test_live_intermediates_of_the_snapshot_is_the_pin(jwks):
    assert nvidia.live_intermediates(jwks) == nvidia.NRAS_TRUSTED_SPKI_SHA256


# --- selftest shows the new refusals to the user -------------------------------------------


def test_selftest_includes_the_jwks_mutations():
    names = {m.name for m in MUTATIONS}
    assert {"nras-own-key", "jwks-duplicate-kid", "jwks-no-chain"} <= names

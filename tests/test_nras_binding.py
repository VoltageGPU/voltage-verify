# SPDX-License-Identifier: MIT
"""Every GPU token must be one the verifier token lists, and carry the same nonce.

NRAS signs one verifier token (overall result, eat_nonce) and one token per GPU. The verifier
token lists each GPU token in ``submods`` as ``["DIGEST", ["SHA-256", <hex>]]``, the SHA-256 of
the compact GPU token, and every GPU token carries its own ``eat_nonce``. Up to 0.2.3 neither
link was checked: a GPU token taken from another attestation, signed by NVIDIA in its own
right, verified next to a fresh verifier token, and could pad ``--gpus`` or replace a GPU whose
claims failed. Found in the audit of 5 November 2026.

The foreign token comes from a second real 8x H100 attestation (10 September 2026) whose key
is in the JWKS snapshot; a GPU token with a mismatched nonce can only be built under a key made
here, pinned for the test.
"""

from __future__ import annotations

import base64
import copy
import dataclasses
import datetime as dt
import hashlib
import json
from pathlib import Path

import jwt
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from jwt.algorithms import ECAlgorithm

from voltage_verify import nvidia
from voltage_verify.bundle import Bundle
from voltage_verify.selftest import MUTATIONS, mutate_nras_extra_gpu
from voltage_verify.verify import verify_bundle

EXAMPLE = Path(__file__).parent.parent / "examples" / "hello-workload" / "bundle-8xh100-2026-09-12.json"
CHALLENGE = "ef7f54c160627ee536a02b5e73896ac4b1ac2cdefdb7cfaaf2366a7d33cc8731"


@pytest.fixture(scope="module")
def example() -> Bundle:
    return Bundle.load(EXAMPLE)


def _devices(payload) -> dict:
    return payload[1]["REMOTE_GPU_CLAIMS"][1]


def _with_devices(bundle: Bundle, devices: dict, extra_keys: list | None = None) -> Bundle:
    nras = json.loads(json.dumps(bundle.nras))
    nras[1]["REMOTE_GPU_CLAIMS"][1] = devices
    known = {k["kid"] for k in bundle.jwks["keys"]}
    keys = [*bundle.jwks["keys"], *[k for k in extra_keys or [] if k["kid"] not in known]]
    return dataclasses.replace(bundle, nras=nras, jwks={"keys": keys})


def _foreign_key(nras_ppcie, jwks) -> dict:
    kid = nvidia.unverified_header(_devices(nras_ppcie)["GPU-3"])["kid"]
    return next(k for k in jwks["keys"] if k["kid"] == kid)


# --- the real bundles still verify ----------------------------------------------------------


def test_every_gpu_token_of_the_reference_bundle_is_listed_by_the_verifier(example):
    res = nvidia.verify_nras(example.nras, example.jwks, example.commitments["sha256"], require_devices=8)
    assert res.overall_ok, res.policy_failures
    assert res.nonce_mismatch == []
    listed = nvidia.unverified_claims(nvidia.split_tokens(example.nras)[1])["submods"]
    for name, token in _devices(example.nras).items():
        assert listed[name] == ["DIGEST", ["SHA-256", hashlib.sha256(token.encode()).hexdigest()]]


# --- a GPU token from another attestation ---------------------------------------------------


def test_gpu_token_from_another_attestation_is_refused(example, nras_ppcie, jwks):
    devices = {**_devices(example.nras), "GPU-3": _devices(nras_ppcie)["GPU-3"]}
    forged = _with_devices(example, devices, [_foreign_key(nras_ppcie, jwks)])
    with pytest.raises(nvidia.NrasError, match="GPU-3.*not the one the verifier token lists"):
        nvidia.verify_nras(forged.nras, forged.jwks, example.commitments["sha256"])


def test_bundle_with_a_gpu_token_from_another_attestation_does_not_verify(example, nras_ppcie, jwks):
    devices = {**_devices(example.nras), "GPU-3": _devices(nras_ppcie)["GPU-3"]}
    forged = _with_devices(example, devices, [_foreign_key(nras_ppcie, jwks)])
    report = verify_bundle(forged, online=False, expected_challenge=CHALLENGE, require_devices=8)
    assert not report.ok and "nvidia.signatures" in report.failed()
    assert "GPU-3" in " ".join(c.detail for c in report.checks if c.name == "nvidia.signatures")


def test_extra_gpu_token_cannot_pad_the_gpu_count(example, nras_ppcie, jwks):
    devices = {**_devices(example.nras), "GPU-8": _devices(nras_ppcie)["GPU-3"]}
    forged = _with_devices(example, devices, [_foreign_key(nras_ppcie, jwks)])
    with pytest.raises(nvidia.NrasError, match="GPU-8"):
        nvidia.verify_nras(forged.nras, forged.jwks, example.commitments["sha256"], require_devices=9)


def test_selftest_pads_the_gpu_count_and_expects_a_refusal(example):
    assert "nras-extra-gpu" in {m.name for m in MUTATIONS}
    padded = mutate_nras_extra_gpu(example)
    assert len(_devices(padded.nras)) == 9 and _devices(padded.nras)["GPU-8"] == _devices(example.nras)["GPU-0"]
    report = verify_bundle(padded, online=False, require_devices=9)
    assert not report.ok and "nvidia.signatures" in report.failed()


def test_missing_gpu_token_is_refused(example):
    devices = {n: t for n, t in _devices(example.nras).items() if n != "GPU-7"}
    with pytest.raises(nvidia.NrasError, match="GPU-7"):
        nvidia.verify_nras(_with_devices(example, devices).nras, example.jwks, example.commitments["sha256"])


def test_gpu_tokens_swapped_between_names_are_refused(example):
    devices = dict(_devices(example.nras))
    devices["GPU-0"], devices["GPU-1"] = devices["GPU-1"], devices["GPU-0"]
    with pytest.raises(nvidia.NrasError, match="not the one the verifier token lists"):
        nvidia.verify_nras(_with_devices(example, devices).nras, example.jwks, example.commitments["sha256"])


@pytest.mark.parametrize(
    "submods",
    [
        None,
        {},
        "GPU-0",
        {"GPU-0": ["DIGEST", ["SHA-384", "00" * 48]]},
        {"GPU-0": ["DIGEST", ["SHA-256", "zz" * 32]]},
        {"GPU-0": ["CLAIMS", {"measres": "success"}]},
        {"GPU-0": ["DIGEST", "SHA-256"]},
    ],
)
def test_submods_that_are_not_sha256_digests_are_refused(pinned, submods):
    payload, jwks = pinned.payload(verifier_claims={"submods": submods})
    with pytest.raises(nvidia.NrasError, match="submods"):
        nvidia.verify_nras(payload, jwks, pinned.nonce)


# --- a GPU token on another nonce (only under a key made here) --------------------------------


def _name(cn: str) -> x509.Name:
    return x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "NVIDIA Corporation"),
            x509.NameAttribute(NameOID.COMMON_NAME, cn),
        ]
    )


def _cert(subject, issuer, public_key, signer, ca: bool) -> x509.Certificate:
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


GOOD_DEVICE = {**nvidia.DEVICE_POLICY, "hwmodel": "GH100"}


@dataclasses.dataclass
class _Pinned:
    """Mints NRAS answers under a test intermediate that the test pins as if it were NVIDIA's."""

    key: ec.EllipticCurvePrivateKey
    jwk: dict
    nonce: str

    def token(self, claims: dict) -> str:
        now = int(dt.datetime.now(dt.timezone.utc).timestamp())
        body = {"iss": nvidia.NRAS_ISSUER, "iat": now, "eat_nonce": self.nonce, **claims}
        return jwt.encode(body, self.key, algorithm="ES384", headers={"kid": self.jwk["kid"]})

    def payload(self, verifier_claims: dict | None = None, devices: dict | None = None) -> tuple[list, dict]:
        gpus = {name: self.token({**GOOD_DEVICE, **claims}) for name, claims in (devices or {"GPU-0": {}}).items()}
        submods = {n: ["DIGEST", ["SHA-256", hashlib.sha256(t.encode()).hexdigest()]] for n, t in gpus.items()}
        verifier = self.token({"x-nvidia-overall-att-result": True, "submods": submods, **(verifier_claims or {})})
        overall = jwt.encode({"iss": "NV-Attestation-SDK"}, "sdk-local-envelope-not-signed-by-nvidia", algorithm="HS256")
        return [["JWT", overall], {"REMOTE_GPU_CLAIMS": [["JWT", verifier], gpus]}], {"keys": [copy.deepcopy(self.jwk)]}


@pytest.fixture
def pinned(monkeypatch) -> _Pinned:
    ca = ec.generate_private_key(ec.SECP384R1())
    leaf_key = ec.generate_private_key(ec.SECP384R1())
    inter = _cert(_name("Test Intermediate"), _name("Test Root"), ca.public_key(), ca, ca=True)
    leaf = _cert(_name("Test GPU signer"), inter.subject, leaf_key.public_key(), ca, ca=False)
    jwk = json.loads(ECAlgorithm.to_jwk(leaf_key.public_key()))
    jwk.update({"kid": "test-kid", "alg": "ES384", "x5c": [_der_b64(leaf), _der_b64(inter)]})
    monkeypatch.setattr(nvidia, "NRAS_TRUSTED_SPKI_SHA256", frozenset({nvidia._spki_sha256(inter)}))
    return _Pinned(leaf_key, jwk, "ab" * 32)


def test_minted_answer_verifies_when_every_link_holds(pinned):
    payload, jwks = pinned.payload(devices={"GPU-0": {}, "GPU-1": {}})
    res = nvidia.verify_nras(payload, jwks, pinned.nonce, require_devices=2)
    assert res.overall_ok, res.policy_failures


def test_gpu_token_on_another_nonce_is_refused_even_when_listed(pinned):
    payload, jwks = pinned.payload(devices={"GPU-0": {}, "GPU-1": {"eat_nonce": "cd" * 32}})
    res = nvidia.verify_nras(payload, jwks, pinned.nonce)
    assert not res.overall_ok
    assert res.nonce_mismatch == ["GPU-1"]
    assert any(f.startswith("GPU-1: eat_nonce") for f in res.policy_failures)


def test_gpu_token_without_eat_nonce_is_refused(pinned):
    payload, jwks = pinned.payload(devices={"GPU-0": {"eat_nonce": None}})
    res = nvidia.verify_nras(payload, jwks, pinned.nonce)
    assert res.nonce_mismatch == ["GPU-0"]


def test_bundle_whose_gpu_token_is_on_another_nonce_fails_nvidia_nonce(example, pinned):
    pinned.nonce = example.commitments["sha256"]
    payload, jwks = pinned.payload(devices={"GPU-0": {}, "GPU-1": {"eat_nonce": "cd" * 32}})
    report = verify_bundle(dataclasses.replace(example, nras=payload, jwks=jwks), online=False)
    by_name = {c.name: c for c in report.checks}
    assert by_name["nvidia.signatures"].ok
    assert not by_name["nvidia.nonce"].ok and "GPU-1" in by_name["nvidia.nonce"].detail
    assert "nvidia.nonce" in report.failed() and "nvidia.policy" not in report.failed()

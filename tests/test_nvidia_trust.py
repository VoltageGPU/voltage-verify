# SPDX-License-Identifier: MIT
"""The JWKS is only trusted through an x5c chain ending on a pinned NRAS intermediate.

Regression tests for a private report of 23 September 2026: in offline mode, and when the
live JWKS was unreachable, the verifier used the JWK coordinates shipped in the bundle
itself, so a bundle could carry its own signing key.
"""

import copy
import datetime as dt
import json

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from jwt.algorithms import ECAlgorithm

from voltage_verify import nvidia


def _attacker_jwk(kid: str) -> tuple[ec.EllipticCurvePrivateKey, dict]:
    priv = ec.generate_private_key(ec.SECP384R1())
    jwk = json.loads(ECAlgorithm.to_jwk(priv.public_key()))
    jwk.update({"kid": kid, "alg": "ES384"})
    return priv, jwk


def _sign(priv, kid: str, claims: dict) -> str:
    import jwt

    return jwt.encode(claims, priv, algorithm="ES384", headers={"kid": kid})


def _self_signed(priv) -> str:
    import base64

    from cryptography.hazmat.primitives import serialization

    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "NVIDIA Attestation Service GPU Intermediate 004")])
    now = dt.datetime.now(dt.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(priv.public_key())
        .serial_number(1)
        .not_valid_before(now - dt.timedelta(days=1))
        .not_valid_after(now + dt.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(priv, hashes.SHA384())
    )
    return base64.b64encode(cert.public_bytes(serialization.Encoding.DER)).decode()


CLAIMS = {"iss": nvidia.NRAS_ISSUER, "iat": 1789000000, "eat_nonce": "00" * 32}


def test_bare_jwk_without_x5c_is_rejected():
    priv, jwk = _attacker_jwk("k1")
    with pytest.raises(nvidia.NrasError, match="no x5c chain"):
        nvidia.verify_token(_sign(priv, "k1", CLAIMS), {"keys": [jwk]}, "verifier", "x")


def test_real_nvidia_chain_with_attacker_coordinates_is_rejected(jwks):
    priv, jwk = _attacker_jwk("k2")
    jwk["x5c"] = jwks["keys"][0]["x5c"]
    with pytest.raises(nvidia.NrasError, match="do not match the leaf"):
        nvidia.verify_token(_sign(priv, "k2", CLAIMS), {"keys": [jwk]}, "verifier", "x")


def test_self_signed_chain_named_like_nvidia_is_rejected():
    priv, jwk = _attacker_jwk("k3")
    cert = _self_signed(priv)
    jwk["x5c"] = [cert, cert]  # leaf and "intermediate": both made here, named like NVIDIA's
    with pytest.raises(nvidia.NrasError, match="pinned NVIDIA intermediate"):
        nvidia.verify_token(_sign(priv, "k3", CLAIMS), {"keys": [jwk]}, "verifier", "x")


def test_every_snapshot_key_chains_to_the_pin(jwks):
    for key in jwks["keys"]:
        chain = nvidia._load_chain(key, key.get("kid"))
        assert chain[-1].subject.rfc4514_string().endswith("CN=NVIDIA Attestation Service GPU Intermediate 004")


def test_keys_under_both_nvidia_intermediates_chain_to_a_pin_offline(jwks_oct22):
    """Since October 2026 NRAS also signs under GPU Intermediate 005 (77 of 213 keys on 22 Oct).

    Offline there is no live JWKS to vouch for an intermediate, so 0.2.2 refused every bundle
    signed under a 005 key. Both must now resolve to a pinned intermediate.
    """
    tops = []
    for key in jwks_oct22["keys"]:
        _, chain, source = nvidia._key_for(jwks_oct22, key["kid"])
        assert source == nvidia.PINNED_INTERMEDIATE
        tops.append(chain[-1].subject.rfc4514_string())
    assert [t.rsplit(" ", 1)[-1] for t in tops] == ["004", "005"]


def test_pins_are_exactly_nvidia_intermediates_004_and_005():
    assert nvidia.NRAS_TRUSTED_SPKI_SHA256 == {
        "fd32837f954e2c45db073105166dfe6985ae0480bb113fba63b091a75affe896",  # GPU Intermediate 004
        "ba14d00377699189b6762c22c744a9dbdc62aa9396097ada7150426c6ff39afc",  # GPU Intermediate 005
    }


def test_real_005_chain_with_attacker_coordinates_is_rejected(jwks_oct22):
    k005 = jwks_oct22["keys"][1]
    _, jwk = _attacker_jwk(k005["kid"])
    jwk["x5c"] = k005["x5c"]
    with pytest.raises(nvidia.NrasError, match="do not match the leaf"):
        nvidia._key_for({"keys": [jwk]}, k005["kid"])


@pytest.mark.parametrize(("leaf_from", "intermediate_from"), [(0, 1), (1, 0)])
def test_leaf_presented_with_the_other_pinned_intermediate_is_rejected(jwks_oct22, leaf_from, intermediate_from):
    """A 004 leaf with the 005 certificate, or the reverse: both are pinned, neither issued it."""
    keys = jwks_oct22["keys"]
    jwk = {**keys[leaf_from], "x5c": [keys[leaf_from]["x5c"][0], keys[intermediate_from]["x5c"][1]]}
    with pytest.raises(nvidia.NrasError, match="not signed by the NVIDIA intermediate"):
        nvidia._key_for({"keys": [jwk]}, jwk["kid"])


def test_token_outside_leaf_validity_is_rejected(jwks):
    chain = nvidia._load_chain(jwks["keys"][0], None)
    late = chain[0].not_valid_after_utc + dt.timedelta(days=1)
    with pytest.raises(nvidia.NrasError, match="outside the validity"):
        nvidia._check_validity(chain, late, "x")
    nvidia._check_validity(chain, chain[0].not_valid_before_utc + dt.timedelta(minutes=1), "x")


def test_forged_bundle_is_rejected_offline():
    from pathlib import Path

    from voltage_verify.bundle import Bundle
    from voltage_verify.selftest import mutate_nras_own_key
    from voltage_verify.verify import verify_bundle

    example = Path(__file__).parent.parent / "examples" / "hello-workload" / "bundle-8xh100-2026-09-12.json"
    forged = mutate_nras_own_key(Bundle.load(example))
    report = verify_bundle(copy.deepcopy(forged), online=False)
    assert not report.ok and "nvidia.signatures" in report.failed()

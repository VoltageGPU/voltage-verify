# SPDX-License-Identifier: MIT
"""Offline checks against a real TDX quote captured on a VoltageGPU H200 VM on 4 September 2026."""

from datetime import datetime, timedelta, timezone

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from voltage_verify import tdx

AT = datetime(2026, 9, 12, tzinfo=timezone.utc)


def test_parse_real_quote(quote_bytes, report_data):
    q = tdx.parse_quote(quote_bytes)
    assert q.version == 4 and q.tee_type == 0x81 and q.att_key_type == 2
    assert q.qe_vendor_id == tdx.INTEL_QE_VENDOR_ID
    assert q.report.report_data == report_data
    assert quote_bytes[tdx.REPORT_DATA_OFFSET : tdx.REPORT_DATA_OFFSET + 64] == report_data
    assert len(q.pck_chain) == 3
    assert q.tdx_module_version == 1
    # trailing zero padding from the TSM outblob is dropped
    assert len(q.raw) < len(quote_bytes)


def test_signatures_and_chain_verify_to_pinned_intel_root(quote_bytes):
    q = tdx.parse_quote(quote_bytes)
    tdx.verify_signatures(q, at=AT)


def test_pinned_root_matches_intel_publication():
    root = tdx.intel_root_ca()
    assert "Intel SGX Root CA" in root.subject.rfc4514_string()


def test_pck_extension_fields(quote_bytes):
    q = tdx.parse_quote(quote_bytes)
    info = tdx.pck_info(q.pck_leaf)
    assert info.fmspc.hex() == "90c06f000000"
    assert len(info.tcb_components) == 16 and len(info.cpu_svn) == 16
    assert info.pce_svn > 0


def test_td_attributes_sit_at_offset_168_and_real_quote_is_not_debug(quote_bytes):
    q = tdx.parse_quote(quote_bytes)
    # 48-byte header, then 120 bytes into the TD report (TEE_TCB_SVN, MRSEAM, MRSIGNERSEAM, SEAMATTRIBUTES)
    assert quote_bytes[168:176] == q.report.td_attributes == bytes.fromhex("0000001000000000")
    assert tdx.TD_ATTRIBUTES_OFFSET == 168
    assert q.report.debug is False  # only SEPT_VE_DISABLE (bit 28) is set


def test_debug_is_bit_0_of_td_attributes(quote_bytes):
    raw = bytearray(quote_bytes)
    raw[tdx.TD_ATTRIBUTES_OFFSET] |= 0x01
    assert tdx.parse_quote(bytes(raw)).report.debug is True
    raw[tdx.TD_ATTRIBUTES_OFFSET] ^= 0x03  # bit 0 back to 0, bit 1 (reserved) set
    assert tdx.parse_quote(bytes(raw)).report.debug is False


def test_tampered_body_fails_signature(quote_bytes):
    raw = bytearray(quote_bytes)
    raw[48 + 136] ^= 1  # MRTD
    q = tdx.parse_quote(bytes(raw))
    with pytest.raises(tdx.QuoteError, match="quote signature"):
        tdx.verify_signatures(q, at=AT)


def test_tampered_attestation_key_fails_binding(quote_bytes):
    raw = bytearray(quote_bytes)
    raw[636 + 64] ^= 1  # first byte of the attestation public key
    q = tdx.parse_quote(bytes(raw))
    with pytest.raises(tdx.QuoteError):
        tdx.verify_signatures(q, at=AT)


def test_wrong_version_rejected(quote_bytes):
    raw = bytearray(quote_bytes)
    raw[0] = 5
    with pytest.raises(tdx.QuoteError, match="version"):
        tdx.parse_quote(bytes(raw))


def test_garbage_after_signature_rejected(quote_bytes):
    with pytest.raises(tdx.QuoteError, match="non-zero"):
        tdx.parse_quote(quote_bytes + b"\x01")


# --- every issuer in the chain must be a CA (audit of 5 November 2026) -------------------------
# verify_chain checked each signature up to the pinned root but not that the issuers were CAs:
# a certificate Intel issued as an end entity could have signed a "PCK" leaf of its own.


def _name(cn: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])


def _issue(subject: str, issuer: x509.Certificate | None, key, signer, basic: x509.BasicConstraints | None):
    now = datetime.now(timezone.utc)
    builder = (
        x509.CertificateBuilder()
        .subject_name(_name(subject))
        .issuer_name(issuer.subject if issuer is not None else _name(subject))
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=1))
    )
    if basic is not None:
        builder = builder.add_extension(basic, critical=True)
    return builder.sign(signer, hashes.SHA256())


def _chain(intermediate_constraints: x509.BasicConstraints | None):
    root_key, inter_key, leaf_key = (ec.generate_private_key(ec.SECP256R1()) for _ in range(3))
    root = _issue("Test Root CA", None, root_key, root_key, x509.BasicConstraints(ca=True, path_length=1))
    inter = _issue("Test Platform CA", root, inter_key, root_key, intermediate_constraints)
    leaf = _issue("Test PCK Certificate", inter, leaf_key, inter_key, x509.BasicConstraints(ca=False, path_length=None))
    return [leaf, inter, root], root


def test_chain_whose_issuers_are_cas_verifies():
    chain, root = _chain(x509.BasicConstraints(ca=True, path_length=0))
    tdx.verify_chain(chain, root)


@pytest.mark.parametrize("constraints", [x509.BasicConstraints(ca=False, path_length=None), None])
def test_issuer_that_is_not_a_ca_is_refused(constraints):
    chain, root = _chain(constraints)
    with pytest.raises(tdx.QuoteError, match="certificate 1 .*is not a CA"):
        tdx.verify_chain(chain, root)


def test_real_chains_have_ca_issuers(quote_bytes):
    q = tdx.parse_quote(quote_bytes)
    for cert in q.pck_chain[1:]:
        assert cert.extensions.get_extension_for_class(x509.BasicConstraints).value.ca

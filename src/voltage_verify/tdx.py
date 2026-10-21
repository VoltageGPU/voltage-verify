# SPDX-License-Identifier: MIT
"""Intel TDX quote (version 4, TD report 1.0) parsing and cryptographic verification.

What this module checks, in order, and what each step proves:

1. Structure: version 4, TEE type 0x81, ECDSA-P256 attestation key, certification data of
   type 6 (QE report) wrapping type 5 (PCK certificate chain). Trailing zero padding, as
   returned by the configfs TSM ``outblob``, is tolerated.
2. Quote signature: the attestation key signs the 632-byte header + TD report. Proves the
   report was not altered after the Quoting Enclave signed it.
3. Attestation key binding: the Quoting Enclave report carries SHA-256(attestation key ||
   QE authentication data) in its first 32 bytes of report_data. Proves the key belongs to
   that QE.
4. QE report signature: signed by the platform's PCK certificate. Proves the QE ran on that
   platform.
5. PCK chain: leaf -> Intel SGX PCK Platform/Processor CA -> Intel SGX Root CA, signatures,
   validity dates, and the root pinned to the certificate Intel publishes. Proves the platform
   is genuine Intel silicon with a provisioning identity. VoltageGPU is nowhere in the chain.

TCB status and Quoting Enclave identity need Intel's published collateral and live in
``pcs.py``. Binding to the workload (report_data) and the refusal of a TD in debug mode
(``TdReport.debug``) live in ``verify.py``.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import resources

from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
from cryptography.x509.oid import ObjectIdentifier

from . import der

HEADER_LEN = 48
TD_REPORT_LEN = 584
SIGNED_LEN = HEADER_LEN + TD_REPORT_LEN  # 632, the bytes the attestation key signs
REPORT_DATA_OFFSET = 568  # inside the quote, 48 + 520
TD_ATTRIBUTES_OFFSET = 168  # inside the quote, 48 + 120
# TD attributes bit 0 (ATTRIBUTES.DEBUG in Intel's TDX module ABI, TDX_TD_ATTR_DEBUG in Linux):
# the TD runs in debug mode and the host can read and modify its private memory and CPU state.
TD_ATTR_DEBUG = 1 << 0
QE_REPORT_LEN = 384
INTEL_QE_VENDOR_ID = bytes.fromhex("939a7233f79c4ca9940a0db3957f0607")
SGX_EXTENSION_OID = ObjectIdentifier("1.2.840.113741.1.13.1")
INTEL_ROOT_CA_SHA256 = "44a0196b2b99f889b8e149e95b807a350e7424964399e885a7cbb8ccfab674d3"
INTEL_ROOT_CA_URL = "https://certificates.trustedservices.intel.com/Intel_SGX_Provisioning_Certification_RootCA.pem"

CERT_DATA_QE_REPORT = 6
CERT_DATA_PCK_CHAIN = 5


class QuoteError(ValueError):
    """Raised when a quote is malformed or fails a cryptographic check."""


@dataclass(frozen=True)
class TdReport:
    tee_tcb_svn: bytes
    mr_seam: bytes
    mr_signer_seam: bytes
    seam_attributes: bytes
    td_attributes: bytes
    xfam: bytes
    mr_td: bytes
    mr_config_id: bytes
    mr_owner: bytes
    mr_owner_config: bytes
    rtmr: tuple[bytes, bytes, bytes, bytes]
    report_data: bytes

    @classmethod
    def parse(cls, body: bytes) -> TdReport:
        if len(body) != TD_REPORT_LEN:
            raise QuoteError(f"TD report must be {TD_REPORT_LEN} bytes, got {len(body)}")
        off = 0

        def take(n: int) -> bytes:
            nonlocal off
            chunk = body[off : off + n]
            off += n
            return chunk

        return cls(
            tee_tcb_svn=take(16),
            mr_seam=take(48),
            mr_signer_seam=take(48),
            seam_attributes=take(8),
            td_attributes=take(8),
            xfam=take(8),
            mr_td=take(48),
            mr_config_id=take(48),
            mr_owner=take(48),
            mr_owner_config=take(48),
            rtmr=(take(48), take(48), take(48), take(48)),
            report_data=take(64),
        )

    @property
    def debug(self) -> bool:
        """True when the TD runs in debug mode: whatever else verifies, it keeps nothing from the host."""
        return bool(int.from_bytes(self.td_attributes, "little") & TD_ATTR_DEBUG)


@dataclass(frozen=True)
class QeReport:
    """The SGX report body (384 bytes) of the Quoting Enclave that signed the quote."""

    raw: bytes
    cpu_svn: bytes
    misc_select: bytes
    attributes: bytes
    mr_enclave: bytes
    mr_signer: bytes
    isv_prod_id: int
    isv_svn: int
    report_data: bytes

    @classmethod
    def parse(cls, raw: bytes) -> QeReport:
        if len(raw) != QE_REPORT_LEN:
            raise QuoteError(f"QE report must be {QE_REPORT_LEN} bytes, got {len(raw)}")
        return cls(
            raw=raw,
            cpu_svn=raw[0:16],
            misc_select=raw[16:20],
            attributes=raw[48:64],
            mr_enclave=raw[64:96],
            mr_signer=raw[128:160],
            isv_prod_id=struct.unpack_from("<H", raw, 256)[0],
            isv_svn=struct.unpack_from("<H", raw, 258)[0],
            report_data=raw[320:384],
        )


@dataclass(frozen=True)
class PckInfo:
    """Fields of the Intel SGX extension carried by the PCK leaf certificate."""

    fmspc: bytes
    pce_id: bytes
    cpu_svn: bytes
    pce_svn: int
    tcb_components: tuple[int, ...]  # 16 SGX TCB component SVNs
    sgx_type: int


@dataclass
class Quote:
    raw: bytes
    version: int
    att_key_type: int
    tee_type: int
    qe_vendor_id: bytes
    user_data: bytes
    report: TdReport
    signature: bytes
    attestation_key: bytes
    qe_report: QeReport
    qe_report_signature: bytes
    qe_auth_data: bytes
    pck_chain: list[x509.Certificate] = field(default_factory=list)
    # Bytes after the signature data that were ignored (only when parse_quote was
    # called with allow_trailing=True). Zero for anything this tool produced.
    trailing: int = 0

    @property
    def signed_bytes(self) -> bytes:
        return self.raw[:SIGNED_LEN]

    @property
    def pck_leaf(self) -> x509.Certificate:
        return self.pck_chain[0]

    @property
    def tdx_module_version(self) -> int:
        return self.report.tee_tcb_svn[1]


def _p256_verify(public_key: ec.EllipticCurvePublicKey, raw_sig: bytes, message: bytes) -> None:
    if len(raw_sig) != 64:
        raise QuoteError("expected a 64-byte raw ECDSA signature")
    der_sig = encode_dss_signature(int.from_bytes(raw_sig[:32], "big"), int.from_bytes(raw_sig[32:], "big"))
    public_key.verify(der_sig, message, ec.ECDSA(hashes.SHA256()))


def parse_quote(raw: bytes, *, allow_trailing: bool = False) -> Quote:
    """Parse a TDX v4 quote. Raises QuoteError on any structural problem.

    Bundles produced by this tool never carry bytes after the signature data, so by default
    any non-zero trailing byte is a structural error. Quotes captured elsewhere sometimes do
    (Google's go-tdx-guest test vector deliberately appends a text marker); ``allow_trailing``
    keeps those bytes out of the parsed quote and reports how many were ignored in ``trailing``.
    """
    if len(raw) < SIGNED_LEN + 4:
        raise QuoteError("quote is too short to hold a header and a TD report")
    version, att_key_type, tee_type = struct.unpack_from("<HHI", raw, 0)
    if version != 4:
        raise QuoteError(f"unsupported quote version {version} (expected 4)")
    if tee_type != 0x81:
        raise QuoteError(f"TEE type 0x{tee_type:x} is not TDX (0x81)")
    if att_key_type != 2:
        raise QuoteError(f"attestation key type {att_key_type} is not ECDSA-256-with-P256 (2)")
    qe_vendor_id = raw[12:28]
    user_data = raw[28:48]
    report = TdReport.parse(raw[HEADER_LEN:SIGNED_LEN])

    sig_len = struct.unpack_from("<I", raw, SIGNED_LEN)[0]
    sig_start = SIGNED_LEN + 4
    sig_end = sig_start + sig_len
    if sig_end > len(raw):
        raise QuoteError("signature data runs past the end of the quote")
    trailing = len(raw) - sig_end if any(raw[sig_end:]) else 0
    if trailing and not allow_trailing:
        raise QuoteError(f"{trailing} non-zero bytes follow the signature data")
    sd = raw[sig_start:sig_end]
    if len(sd) < 134:
        raise QuoteError("signature data is too short")
    signature, attestation_key = sd[0:64], sd[64:128]
    cert_type, cert_size = struct.unpack_from("<HI", sd, 128)
    if cert_type != CERT_DATA_QE_REPORT:
        raise QuoteError(f"certification data type {cert_type} is not QE report certification data (6)")
    cert = sd[134 : 134 + cert_size]
    if len(cert) != cert_size or cert_size < QE_REPORT_LEN + 64 + 2 + 6:
        raise QuoteError("QE report certification data is truncated")
    qe_report = QeReport.parse(cert[0:QE_REPORT_LEN])
    qe_sig = cert[QE_REPORT_LEN : QE_REPORT_LEN + 64]
    auth_len = struct.unpack_from("<H", cert, QE_REPORT_LEN + 64)[0]
    auth_start = QE_REPORT_LEN + 66
    qe_auth = cert[auth_start : auth_start + auth_len]
    inner_type, inner_size = struct.unpack_from("<HI", cert, auth_start + auth_len)
    if inner_type != CERT_DATA_PCK_CHAIN:
        raise QuoteError(f"inner certification data type {inner_type} is not a PCK certificate chain (5)")
    pem_start = auth_start + auth_len + 6
    pem = cert[pem_start : pem_start + inner_size].rstrip(b"\x00")
    try:
        chain = x509.load_pem_x509_certificates(pem)
    except ValueError as err:
        raise QuoteError(f"PCK certificate chain does not parse: {err}") from err
    if len(chain) != 3:
        raise QuoteError(f"expected a 3-certificate PCK chain, got {len(chain)}")
    return Quote(
        raw=raw[:sig_end],
        version=version,
        att_key_type=att_key_type,
        tee_type=tee_type,
        qe_vendor_id=qe_vendor_id,
        user_data=user_data,
        report=report,
        signature=signature,
        attestation_key=attestation_key,
        qe_report=qe_report,
        qe_report_signature=qe_sig,
        qe_auth_data=qe_auth,
        pck_chain=list(chain),
        trailing=trailing,
    )


def intel_root_ca() -> x509.Certificate:
    """The Intel SGX Root CA shipped with the package (pinned by SHA-256 fingerprint)."""
    pem = resources.files("voltage_verify.data").joinpath("intel_sgx_root_ca.pem").read_bytes()
    cert = x509.load_pem_x509_certificate(pem)
    if cert.fingerprint(hashes.SHA256()).hex() != INTEL_ROOT_CA_SHA256:
        raise QuoteError("the packaged Intel SGX Root CA does not match its pinned fingerprint")
    return cert


def verify_chain(chain: list[x509.Certificate], trusted_root: x509.Certificate, at: datetime | None = None) -> None:
    """Verify signatures and validity of leaf -> intermediate -> root, and that root is trusted."""
    now = at or datetime.now(timezone.utc)
    for idx, cert in enumerate(chain):
        if not (cert.not_valid_before_utc <= now <= cert.not_valid_after_utc):
            raise QuoteError(f"certificate {idx} ({cert.subject.rfc4514_string()}) is not valid at {now.isoformat()}")
        issuer = chain[idx + 1] if idx + 1 < len(chain) else cert
        key = issuer.public_key()
        if not isinstance(key, ec.EllipticCurvePublicKey):
            raise QuoteError("PCK chain certificates must use EC keys")
        try:
            key.verify(cert.signature, cert.tbs_certificate_bytes, ec.ECDSA(cert.signature_hash_algorithm))
        except Exception as err:  # noqa: BLE001
            raise QuoteError(f"certificate {idx} is not signed by certificate {idx + 1}: {err}") from err
    root = chain[-1]
    if root.fingerprint(hashes.SHA256()) != trusted_root.fingerprint(hashes.SHA256()):
        raise QuoteError("the quote's root certificate is not the pinned Intel SGX Root CA")


def verify_signatures(quote: Quote, trusted_root: x509.Certificate | None = None, at: datetime | None = None) -> None:
    """Run checks 2 to 5 of the module docstring. Raises QuoteError on the first failure."""
    if quote.qe_vendor_id != INTEL_QE_VENDOR_ID:
        raise QuoteError(f"QE vendor id {quote.qe_vendor_id.hex()} is not Intel's")
    x = int.from_bytes(quote.attestation_key[:32], "big")
    y = int.from_bytes(quote.attestation_key[32:], "big")
    try:
        att_key = ec.EllipticCurvePublicNumbers(x, y, ec.SECP256R1()).public_key()
    except ValueError as err:
        raise QuoteError(f"attestation key is not a valid P-256 point: {err}") from err
    try:
        _p256_verify(att_key, quote.signature, quote.signed_bytes)
    except Exception as err:  # noqa: BLE001
        raise QuoteError(f"quote signature does not verify with the attestation key: {err}") from err

    expected = hashlib.sha256(quote.attestation_key + quote.qe_auth_data).digest()
    if quote.qe_report.report_data[:32] != expected:
        raise QuoteError("QE report does not bind the attestation key (report_data mismatch)")

    leaf_key = quote.pck_leaf.public_key()
    if not isinstance(leaf_key, ec.EllipticCurvePublicKey):
        raise QuoteError("PCK leaf certificate must carry an EC key")
    try:
        _p256_verify(leaf_key, quote.qe_report_signature, quote.qe_report.raw)
    except Exception as err:  # noqa: BLE001
        raise QuoteError(f"QE report signature does not verify with the PCK certificate: {err}") from err

    verify_chain(quote.pck_chain, trusted_root or intel_root_ca(), at=at)


def pck_info(leaf: x509.Certificate) -> PckInfo:
    """Read FMSPC, PCE id, TCB components and CPU SVN from the PCK leaf certificate."""
    try:
        ext = leaf.extensions.get_extension_for_oid(SGX_EXTENSION_OID).value
    except x509.ExtensionNotFound as err:
        raise QuoteError("PCK certificate has no Intel SGX extension") from err
    raw = ext.value if isinstance(ext, x509.UnrecognizedExtension) else bytes(ext.public_bytes())
    nodes = der.decode(raw)
    if len(nodes) != 1:
        raise QuoteError("unexpected SGX extension layout")
    top = der.pairs(nodes[0])
    base = "1.2.840.113741.1.13.1"
    try:
        tcb = der.pairs(top[f"{base}.2"])
        comps = tuple(int(tcb[f"{base}.2.{i}"].value) for i in range(1, 17))
        pce_svn = int(tcb[f"{base}.2.17"].value)
        cpu_svn = bytes(tcb[f"{base}.2.18"].value)
        pce_id = bytes(top[f"{base}.3"].value)
        fmspc = bytes(top[f"{base}.4"].value)
        sgx_type = int(top[f"{base}.5"].value)
    except (KeyError, TypeError, ValueError) as err:
        raise QuoteError(f"SGX extension is missing a required field: {err}") from err
    if len(fmspc) != 6 or len(cpu_svn) != 16:
        raise QuoteError("SGX extension field sizes are wrong")
    return PckInfo(fmspc=fmspc, pce_id=pce_id, cpu_svn=cpu_svn, pce_svn=pce_svn, tcb_components=comps, sgx_type=sgx_type)

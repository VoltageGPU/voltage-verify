# SPDX-License-Identifier: MIT
"""Intel Provisioning Certification Service (PCS) collateral: fetch, verify, evaluate.

Three documents complete a TDX quote verification, all published by Intel and signed by
Intel's TCB Signing certificate, which chains to the same pinned Intel SGX Root CA as the
PCK certificates in the quote:

* TCB info (``/tdx/certification/v4/tcb?fmspc=...``): the security version numbers a
  platform must reach for each TCB status (UpToDate, SWHardeningNeeded, OutOfDate, ...).
* QE identity (``/tdx/certification/v4/qe/identity``): what the genuine TDX Quoting Enclave
  looks like (MRSIGNER, product id, attributes) and its minimum ISV SVN.
* Certificate revocation lists for the PCK CA and the root CA.

Everything fetched here is stored in the bundle so that a later verification can run
offline against the exact collateral of the day, and a fresh verification can fetch it
again and compare.
"""

from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

from .tdx import PckInfo, Quote, QuoteError, intel_root_ca, verify_chain

PCS_BASE = "https://api.trustedservices.intel.com"
ROOT_CRL_URL = "https://certificates.trustedservices.intel.com/IntelSGXRootCA.der"
USER_AGENT = "voltage-verify/0.1 (+https://voltagegpu.com)"
# Intel issues TCB info, QE identity and the PCK CRL with a nextUpdate 30 days after their issue
# date (every bundle since September 2026): Intel's own validity for them. Offline, collateral
# from a bundle older than that is reported as stale; it is refused only when the verifier asks
# for a limit (--max-collateral-age).
COLLATERAL_VALIDITY = timedelta(days=30)

STATUS_ORDER = [
    "UpToDate",
    "SWHardeningNeeded",
    "ConfigurationNeeded",
    "ConfigurationAndSWHardeningNeeded",
    "OutOfDate",
    "OutOfDateConfigurationNeeded",
    "Revoked",
]


class CollateralError(ValueError):
    """Raised when collateral cannot be fetched, does not verify, or does not match the quote."""


# What follows the signed part of a PCS document, to the last byte: the signature and nothing else.
_SIGNATURE_TAIL = re.compile(r',"signature":"([0-9a-fA-F]{128})"\}')


@dataclass
class SignedDocument:
    """A PCS JSON document with its signature and issuer chain, kept as raw text.

    The body must be exactly ``{"<inner key>":<signed JSON>,"signature":"<128 hex>"}``, the way
    PCS serves it. The inner document is always read from the signed bytes: a second copy of
    it after the signature, which a JSON parser resolves to the last one, is refused, not read.
    """

    kind: str  # "tcb_info" | "qe_identity"
    body: str  # exact response body
    issuer_chain_pem: str

    @property
    def inner_key(self) -> str:
        return "tcbInfo" if self.kind == "tcb_info" else "enclaveIdentity"

    def _split(self) -> tuple[bytes, bytes]:
        """(the bytes Intel signed, the raw 64-byte signature), once the layout is checked."""
        prefix = f'{{"{self.inner_key}":'
        if not isinstance(self.body, str) or not self.body.startswith(prefix):
            raise CollateralError(f"{self.kind}: unexpected document layout")
        end = self.body.rfind(',"signature":')
        tail = _SIGNATURE_TAIL.fullmatch(self.body, end) if end >= 0 else None
        if tail is None:
            raise CollateralError(f"{self.kind}: unexpected document layout after the signed part")
        return self.body[len(prefix) : end].encode("utf-8"), bytes.fromhex(tail.group(1))

    def signed_bytes(self) -> bytes:
        return self._split()[0]

    @property
    def inner(self) -> dict[str, Any]:
        """The document Intel signed, parsed from the signed bytes only."""
        try:
            value = json.loads(self.signed_bytes())
        except (ValueError, RecursionError) as err:
            raise CollateralError(f"{self.kind}: signed part is not JSON: {err}") from err
        if not isinstance(value, dict):
            raise CollateralError(f"{self.kind}: signed part is not a JSON object")
        return value

    @property
    def payload(self) -> dict[str, Any]:
        return {self.inner_key: self.inner, "signature": self._split()[1].hex()}

    def to_json(self) -> dict[str, str]:
        return {"kind": self.kind, "body": self.body, "issuer_chain_pem": self.issuer_chain_pem}

    @classmethod
    def from_json(cls, data: dict[str, str]) -> SignedDocument:
        return cls(kind=data["kind"], body=data["body"], issuer_chain_pem=data["issuer_chain_pem"])


@dataclass
class Collateral:
    tcb_info: SignedDocument
    qe_identity: SignedDocument
    pck_crl_pem: str | None = None
    root_crl_der_b64: str | None = None
    fetched_at: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "tcb_info": self.tcb_info.to_json(),
            "qe_identity": self.qe_identity.to_json(),
            "pck_crl_pem": self.pck_crl_pem,
            "root_crl_der_b64": self.root_crl_der_b64,
            "fetched_at": self.fetched_at,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Collateral:
        return cls(
            tcb_info=SignedDocument.from_json(data["tcb_info"]),
            qe_identity=SignedDocument.from_json(data["qe_identity"]),
            pck_crl_pem=data.get("pck_crl_pem"),
            root_crl_der_b64=data.get("root_crl_der_b64"),
            fetched_at=data.get("fetched_at"),
        )


def _get(url: str) -> tuple[bytes, dict[str, str]]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            return resp.read(), {k.lower(): v for k, v in resp.headers.items()}
    except urllib.error.HTTPError as err:
        raise CollateralError(f"HTTP {err.code} from {url}") from err
    except urllib.error.URLError as err:
        raise CollateralError(f"cannot reach {url}: {err.reason}") from err


def fetch_collateral(fmspc: bytes, pck_ca: str = "platform") -> Collateral:
    """Fetch TCB info, QE identity and both CRLs from Intel for the platform's FMSPC."""
    body, headers = _get(f"{PCS_BASE}/tdx/certification/v4/tcb?fmspc={fmspc.hex()}")
    chain = urllib.parse.unquote(headers.get("tcb-info-issuer-chain", ""))
    if not chain:
        raise CollateralError("PCS did not return TCB-Info-Issuer-Chain")
    tcb = SignedDocument("tcb_info", body.decode("utf-8"), chain)

    body, headers = _get(f"{PCS_BASE}/tdx/certification/v4/qe/identity")
    chain = urllib.parse.unquote(headers.get("sgx-enclave-identity-issuer-chain", ""))
    if not chain:
        raise CollateralError("PCS did not return SGX-Enclave-Identity-Issuer-Chain")
    qe = SignedDocument("qe_identity", body.decode("utf-8"), chain)

    crl_pem, _ = _get(f"{PCS_BASE}/sgx/certification/v4/pckcrl?ca={pck_ca}&encoding=pem")
    root_crl, _ = _get(ROOT_CRL_URL)
    return Collateral(
        tcb_info=tcb,
        qe_identity=qe,
        pck_crl_pem=crl_pem.decode("utf-8"),
        root_crl_der_b64=base64.b64encode(root_crl).decode("ascii"),
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def pck_ca_kind(quote: Quote) -> str:
    """'platform' or 'processor', from the PCK intermediate certificate's common name."""
    cn = quote.pck_chain[1].subject.rfc4514_string().lower()
    return "processor" if "processor" in cn else "platform"


def verify_document(doc: SignedDocument, trusted_root: x509.Certificate | None = None, at: datetime | None = None) -> None:
    """Verify the issuer chain to the pinned Intel root and the ECDSA signature over the body."""
    try:
        chain = x509.load_pem_x509_certificates(doc.issuer_chain_pem.encode("utf-8"))
    except ValueError as err:
        raise CollateralError(f"{doc.kind}: issuer chain does not parse: {err}") from err
    if len(chain) < 2:
        raise CollateralError(f"{doc.kind}: issuer chain is too short")
    try:
        verify_chain(chain, trusted_root or intel_root_ca(), at=at)
    except QuoteError as err:
        raise CollateralError(f"{doc.kind}: {err}") from err
    signer = chain[0].public_key()
    if not isinstance(signer, ec.EllipticCurvePublicKey):
        raise CollateralError(f"{doc.kind}: signing certificate does not carry an EC key")
    signed, sig = doc._split()  # 64 raw bytes: the layout check allows exactly 128 hex characters
    der_sig = encode_dss_signature(int.from_bytes(sig[:32], "big"), int.from_bytes(sig[32:], "big"))
    try:
        signer.verify(der_sig, signed, ec.ECDSA(hashes.SHA256()))
    except Exception as err:  # noqa: BLE001
        raise CollateralError(f"{doc.kind}: signature does not verify: {err}") from err


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _worse(a: str, b: str) -> str:
    return max(a, b, key=lambda s: STATUS_ORDER.index(s) if s in STATUS_ORDER else len(STATUS_ORDER))


@dataclass
class TcbEvaluation:
    status: str
    level_date: str | None
    advisory_ids: list[str]
    tdx_module_id: str | None
    tdx_module_status: str | None
    qe_status: str
    tcb_info_expired: bool
    qe_identity_expired: bool
    notes: list[str] = field(default_factory=list)

    @property
    def acceptable(self) -> bool:
        return self.status in ("UpToDate", "SWHardeningNeeded") and self.qe_status == "UpToDate"


def evaluate(quote: Quote, pck: PckInfo, collateral: Collateral, at: datetime | None = None) -> TcbEvaluation:
    """Apply Intel's TCB level matching rules to this quote and platform."""
    now = at or datetime.now(timezone.utc)
    notes: list[str] = []
    info = collateral.tcb_info.inner
    if info.get("id") != "TDX":
        raise CollateralError(f"TCB info id {info.get('id')!r} is not TDX")
    if info.get("version") != 3:
        raise CollateralError(f"unsupported TCB info version {info.get('version')}")
    if str(info.get("fmspc", "")).lower() != pck.fmspc.hex():
        raise CollateralError(f"TCB info FMSPC {info.get('fmspc')} does not match the platform's {pck.fmspc.hex()}")
    if str(info.get("pceId", "")).lower() != pck.pce_id.hex():
        raise CollateralError("TCB info PCE id does not match the PCK certificate")
    tcb_expired = _parse_iso(info["nextUpdate"]) < now

    tee_svn = list(quote.report.tee_tcb_svn)
    matched: dict[str, Any] | None = None
    for level in info["tcbLevels"]:
        tcb = level["tcb"]
        sgx_ok = all(
            pck.tcb_components[i] >= int(tcb["sgxtcbcomponents"][i]["svn"]) for i in range(16)
        ) and pck.pce_svn >= int(tcb["pcesvn"])
        tdx_ok = all(tee_svn[i] >= int(tcb["tdxtcbcomponents"][i]["svn"]) for i in range(16))
        if sgx_ok and tdx_ok:
            matched = level
            break
    if matched is None:
        raise CollateralError("no TCB level in Intel's TCB info matches this platform (treat as OutOfDate)")
    status = str(matched["tcbStatus"])
    advisories = list(matched.get("advisoryIDs", []))

    module_id: str | None = None
    module_status: str | None = None
    if quote.tdx_module_version > 0:
        module_id = f"TDX_{quote.tdx_module_version:02d}"
        identities = {m["id"]: m for m in info.get("tdxModuleIdentities", [])}
        module = identities.get(module_id)
        if module is None:
            raise CollateralError(f"TCB info lists no TDX module identity {module_id}")
        if str(module["mrsigner"]).lower() != quote.report.mr_signer_seam.hex():
            raise CollateralError(f"{module_id}: MRSIGNERSEAM does not match Intel's TDX module identity")
        attrs = int.from_bytes(quote.report.seam_attributes, "little")
        mask = int(module["attributesMask"], 16)
        want = int(module["attributes"], 16)
        if attrs & mask != want:
            raise CollateralError(f"{module_id}: SEAM attributes do not match Intel's TDX module identity")
        module_status = "OutOfDate"
        for level in module["tcbLevels"]:
            if tee_svn[0] >= int(level["tcb"]["isvsvn"]):
                module_status = str(level["tcbStatus"])
                break
        status = _worse(status, module_status)
    else:
        notes.append("TDX module version 0: module identity check not applicable")

    qe = collateral.qe_identity.inner
    if qe.get("id") != "TD_QE":
        raise CollateralError(f"QE identity id {qe.get('id')!r} is not TD_QE")
    qe_expired = _parse_iso(qe["nextUpdate"]) < now
    rep = quote.qe_report
    if str(qe["mrsigner"]).lower() != rep.mr_signer.hex():
        raise CollateralError("Quoting Enclave MRSIGNER does not match Intel's QE identity")
    if int(qe["isvprodid"]) != rep.isv_prod_id:
        raise CollateralError("Quoting Enclave ISV product id does not match Intel's QE identity")
    attrs = int.from_bytes(rep.attributes, "big")
    if attrs & int(qe["attributesMask"], 16) != int(qe["attributes"], 16):
        raise CollateralError("Quoting Enclave attributes do not match Intel's QE identity")
    misc = int.from_bytes(rep.misc_select, "little")
    if misc & int(qe["miscselectMask"], 16) != int(qe["miscselect"], 16):
        raise CollateralError("Quoting Enclave MISCSELECT does not match Intel's QE identity")
    qe_status = "OutOfDate"
    for level in qe["tcbLevels"]:
        if rep.isv_svn >= int(level["tcb"]["isvsvn"]):
            qe_status = str(level["tcbStatus"])
            break

    return TcbEvaluation(
        status=status,
        level_date=matched.get("tcbDate"),
        advisory_ids=advisories,
        tdx_module_id=module_id,
        tdx_module_status=module_status,
        qe_status=qe_status,
        tcb_info_expired=tcb_expired,
        qe_identity_expired=qe_expired,
        notes=notes,
    )


@dataclass(frozen=True)
class SignedDates:
    """The dates that tell how old a collateral is, each read from a document Intel signed."""

    oldest: str  # "TCB info", "QE identity" or "PCK CRL": the oldest of the three
    oldest_issued: datetime
    root_crl_issued: datetime
    root_crl_next: datetime | None


def signed_dates(collateral: Collateral) -> SignedDates:
    """Issue dates of the collateral, from the signed bytes. TCB info, QE identity and the PCK CRL
    are re-issued by Intel continuously (issued within hours of any fetch), so the collateral is
    as old as the oldest of them; the root CA CRL is re-issued once a year and kept apart.

    Raises CollateralError without both CRLs: revocation could not be checked from it."""
    if not collateral.pck_crl_pem or not collateral.root_crl_der_b64:
        raise CollateralError("the collateral carries no CRL: revocation cannot be checked from it")
    try:
        pck_crl = x509.load_pem_x509_crl(collateral.pck_crl_pem.encode("utf-8"))
        root_crl = x509.load_der_x509_crl(base64.b64decode(collateral.root_crl_der_b64, validate=True))
        issued = {
            "TCB info": _parse_iso(collateral.tcb_info.inner["issueDate"]),
            "QE identity": _parse_iso(collateral.qe_identity.inner["issueDate"]),
            "PCK CRL": pck_crl.last_update_utc,
        }
        if any(d.tzinfo is None for d in issued.values()):
            raise ValueError("an issue date has no UTC offset")
        oldest = min(issued, key=lambda name: issued[name])
    except (AttributeError, KeyError, TypeError, ValueError) as err:
        raise CollateralError(f"collateral dates are unreadable: {err}") from err
    return SignedDates(oldest, issued[oldest], root_crl.last_update_utc, root_crl.next_update_utc)


def check_revocation(quote: Quote, collateral: Collateral, at: datetime | None = None) -> list[str]:
    """Check the PCK leaf and intermediate against Intel's CRLs. Returns human-readable notes."""
    now = at or datetime.now(timezone.utc)
    notes: list[str] = []
    if not collateral.pck_crl_pem or not collateral.root_crl_der_b64:
        return ["revocation lists not present in collateral: revocation NOT checked"]
    leaf, intermediate, root = quote.pck_chain
    pck_crl = x509.load_pem_x509_crl(collateral.pck_crl_pem.encode("utf-8"))
    root_crl = x509.load_der_x509_crl(base64.b64decode(collateral.root_crl_der_b64))
    for crl, issuer, subject, label in (
        (pck_crl, intermediate, leaf, "PCK certificate"),
        (root_crl, root, intermediate, "PCK CA certificate"),
    ):
        if not crl.is_signature_valid(issuer.public_key()):
            raise CollateralError(f"CRL for the {label} is not signed by its issuer")
        if crl.next_update_utc and crl.next_update_utc < now:
            notes.append(f"CRL for the {label} is past its next update ({crl.next_update_utc.date()})")
        if crl.get_revoked_certificate_by_serial_number(subject.serial_number) is not None:
            raise CollateralError(f"the {label} is REVOKED")
        notes.append(f"{label} not revoked ({len(list(crl))} entries in CRL)")
    return notes

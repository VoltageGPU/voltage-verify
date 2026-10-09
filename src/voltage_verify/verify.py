# SPDX-License-Identifier: MIT
"""Verify an evidence bundle end to end and say, check by check, what held and what did not."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from . import nvidia, pcs, tdx
from .bundle import Bundle
from .canonical import commitments
from .manifest import ManifestError, validate_manifest

NRAS_FRESHNESS = timedelta(minutes=15)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True

    @property
    def label(self) -> str:
        if self.ok:
            return "PASS"
        return "FAIL" if self.required else "WARN"


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks if c.required)

    def add(self, name: str, ok: bool, detail: str, required: bool = True) -> None:
        self.checks.append(Check(name, ok, detail, required))

    def failed(self) -> list[str]:
        return [c.name for c in self.checks if c.required and not c.ok]

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": [{"name": c.name, "result": c.label, "detail": c.detail} for c in self.checks],
            "facts": self.facts,
        }


def fetch_jwks() -> dict[str, Any]:
    req = urllib.request.Request(nvidia.NRAS_JWKS_URL, headers={"User-Agent": pcs.USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def verify_bundle(
    bundle: Bundle,
    *,
    expected_challenge: str | None = None,
    expected_image_digest: str | None = None,
    online: bool = True,
    allowed_hwmodels: set[str] | None = None,
    require_devices: int | None = None,
    at: datetime | None = None,
) -> Report:
    """Run every check. Never raises for a verification failure; returns a Report instead."""
    report = Report()
    now = at or datetime.now(timezone.utc)

    # 1. Manifest and commitments.
    try:
        validate_manifest(bundle.manifest)
        report.add("manifest.format", True, "manifest is well formed")
    except ManifestError as err:
        report.add("manifest.format", False, str(err))
        return report
    com = commitments(bundle.manifest)
    same = bundle.commitments.get("sha256") == com.sha256_hex and bundle.commitments.get("sha512") == com.sha512_hex
    report.add(
        "manifest.commitments",
        same,
        "bundle commitments match the manifest" if same else "bundle commitments do not match the manifest",
    )
    report.facts["commitment_sha256"] = com.sha256_hex
    report.facts["commitment_sha512"] = com.sha512_hex

    if expected_challenge is not None:
        match = str(bundle.manifest.get("challenge", "")).lower() == expected_challenge.lower()
        report.add(
            "manifest.challenge",
            match,
            "challenge is the one this verifier issued"
            if match
            else "challenge differs from the one this verifier issued (replay or foreign bundle)",
        )
    image = (bundle.manifest.get("workload") or {}).get("image")
    if expected_image_digest is not None:
        got = (image or {}).get("digest")
        match = got == expected_image_digest
        report.add(
            "manifest.image",
            match,
            f"image digest {got}" if match else f"image digest {got} differs from expected {expected_image_digest}",
        )
    if image:
        report.facts["image"] = image
    report.facts["artifacts"] = (bundle.manifest.get("workload") or {}).get("artifacts", [])

    # 2. Intel TDX quote: structure, binding, signatures.
    quote: tdx.Quote | None = None
    try:
        quote = tdx.parse_quote(bundle.quote)
        report.add("tdx.structure", True, f"TDX quote v{quote.version}, TEE 0x{quote.tee_type:x}, {len(bundle.quote)} bytes")
    except tdx.QuoteError as err:
        report.add("tdx.structure", False, str(err))
    if quote is not None:
        bound = quote.report.report_data == com.sha512
        report.add(
            "tdx.report_data",
            bound,
            "quote report_data equals SHA-512 of the manifest"
            if bound
            else "quote report_data does NOT equal SHA-512 of the manifest: this quote was not issued for this workload",
        )
        report.facts["tdx"] = {
            "mr_td": quote.report.mr_td.hex(),
            "mr_seam": quote.report.mr_seam.hex(),
            "rtmr": [r.hex() for r in quote.report.rtmr],
            "tee_tcb_svn": quote.report.tee_tcb_svn.hex(),
            "td_attributes": quote.report.td_attributes.hex(),
            "xfam": quote.report.xfam.hex(),
        }
        try:
            tdx.verify_signatures(quote, at=now)
            report.add(
                "tdx.signatures",
                True,
                "attestation key, QE binding, QE report and PCK chain verify up to the pinned Intel SGX Root CA",
            )
        except tdx.QuoteError as err:
            report.add("tdx.signatures", False, str(err))
            quote_ok = False
        else:
            quote_ok = True

        # 3. Intel collateral: TCB status, QE identity, revocation.
        if quote_ok:
            try:
                pck = tdx.pck_info(quote.pck_leaf)
                report.facts["tdx"]["fmspc"] = pck.fmspc.hex()
                collateral: pcs.Collateral | None = None
                source = ""
                if online:
                    collateral = pcs.fetch_collateral(pck.fmspc, pcs.pck_ca_kind(quote))
                    source = "Intel PCS, fetched now"
                elif bundle.collateral and bundle.collateral.get("intel"):
                    collateral = pcs.Collateral.from_json(bundle.collateral["intel"])
                    source = f"bundle (fetched {collateral.fetched_at})"
                if collateral is None:
                    report.add("tdx.collateral", False, "no Intel collateral: run online or use a bundle that embeds it")
                else:
                    pcs.verify_document(collateral.tcb_info, at=now)
                    pcs.verify_document(collateral.qe_identity, at=now)
                    report.add("tdx.collateral", True, f"TCB info and QE identity signed by Intel ({source})")
                    ev = pcs.evaluate(quote, pck, collateral, at=now)
                    report.facts["tdx"]["tcb_status"] = ev.status
                    report.facts["tdx"]["tcb_level_date"] = ev.level_date
                    report.facts["tdx"]["advisory_ids"] = ev.advisory_ids
                    report.facts["tdx"]["tdx_module"] = {"id": ev.tdx_module_id, "status": ev.tdx_module_status}
                    report.facts["tdx"]["qe_status"] = ev.qe_status
                    detail = f"platform TCB {ev.status}, QE {ev.qe_status}"
                    if ev.tdx_module_id:
                        detail += f", {ev.tdx_module_id} {ev.tdx_module_status}"
                    if ev.advisory_ids:
                        detail += f", advisories {', '.join(ev.advisory_ids)}"
                    report.add("tdx.tcb", ev.acceptable, detail)
                    if ev.tcb_info_expired or ev.qe_identity_expired:
                        report.add(
                            "tdx.collateral_fresh",
                            False,
                            "Intel collateral is past its nextUpdate; refetch online",
                            required=False,
                        )
                    try:
                        for note in pcs.check_revocation(quote, collateral, at=now):
                            report.add("tdx.revocation", True, note, required=False)
                    except pcs.CollateralError as err:
                        report.add("tdx.revocation", False, str(err))
            except (tdx.QuoteError, pcs.CollateralError) as err:
                report.add("tdx.tcb", False, str(err))

    # 4. NVIDIA NRAS tokens. Signing keys come from the bundle's JWKS only, each through its x5c
    # chain to the pinned NRAS intermediate (nvidia._load_chain), so a bundle cannot bring its
    # own key. Online, NVIDIA's live JWKS is fetched only to learn the intermediates it
    # presents (rotation); a key that is only in the live JWKS is not used.
    live_anchors: frozenset[str] = frozenset()
    live_note = "offline"
    if online:
        try:
            live_anchors = nvidia.live_intermediates(fetch_jwks())
            live_note = "NVIDIA's live JWKS fetched for its intermediates"
        except Exception as err:  # noqa: BLE001
            live_note = f"live NRAS JWKS unreachable ({err}), pinned intermediate only"
    if not bundle.jwks:
        report.add(
            "nvidia.jwks",
            False,
            "the bundle carries no NVIDIA JWKS; signing keys are only taken from the bundle",
        )
    else:
        try:
            res = nvidia.verify_nras(
                bundle.nras,
                bundle.jwks,
                com.sha256_hex,
                require_devices=require_devices,
                allowed_hwmodels=allowed_hwmodels,
                live_anchors=live_anchors,
            )
            sources = {t.key_source for t in (res.verifier, *res.devices)}
            anchor = (
                "the pinned NRAS intermediate"
                if sources == {nvidia.PINNED_INTERMEDIATE}
                else "an NRAS intermediate presented by NVIDIA's live JWKS"
            )
            report.add(
                "nvidia.signatures",
                True,
                f"{1 + len(res.devices)} NVIDIA-signed tokens verify, "
                f"keys from the bundle chained to {anchor} ({live_note})",
            )
            report.facts["nvidia"] = {
                "mode": bundle.nvidia_mode,
                "issued_at": res.issued_at.isoformat() if res.issued_at else None,
                "expires_at": res.expires_at.isoformat() if res.expires_at else None,
                "devices": [
                    {
                        "name": d.name,
                        "hwmodel": d.claims.get("hwmodel"),
                        "driver": d.claims.get("x-nvidia-gpu-driver-version"),
                        "vbios": d.claims.get("x-nvidia-gpu-vbios-version"),
                        "measres": d.claims.get("measres"),
                    }
                    for d in res.devices
                ],
            }
            nonce_ok = res.nonce == com.sha256_hex
            report.add(
                "nvidia.nonce",
                nonce_ok,
                "NRAS eat_nonce equals SHA-256 of the manifest"
                if nonce_ok
                else "NRAS eat_nonce does NOT equal SHA-256 of the manifest: not issued for this workload",
            )
            policy = [f for f in res.policy_failures if "eat_nonce" not in f]
            policy_ok_text = (
                "overall result true, every GPU: measurements ok, secure boot on, debug off, "
                "nonce matched, report signature and chain validated"
            )
            report.add("nvidia.policy", not policy, policy_ok_text if not policy else "; ".join(policy))
            if res.issued_at:
                drift = abs(res.issued_at - bundle.created_at)
                fresh = drift <= NRAS_FRESHNESS
                detail = f"token issued {int(drift.total_seconds())}s from bundle creation"
                if not fresh:
                    detail += f", more than {int(NRAS_FRESHNESS.total_seconds())}s apart"
                report.add("nvidia.freshness", fresh, detail)
        except nvidia.NrasError as err:
            report.add("nvidia.signatures", False, str(err))

    return report


def verify_quote(
    raw: bytes,
    *,
    expected_report_data: bytes | None = None,
    online: bool = True,
    accept_out_of_date: bool = False,
    at: datetime | None = None,
) -> Report:
    """Verify a bare Intel TDX quote from any provider: structure, signature chain up to the
    pinned Intel root, then TCB status, QE identity and revocation from Intel PCS.

    No manifest, no NVIDIA token, no VoltageGPU anywhere in the path. The only thing the
    caller can bind is ``expected_report_data``: the 64 bytes the guest put in the report
    (typically the hash of a nonce the verifier issued). Never raises for a verification
    failure; returns a Report.
    """
    report = Report()
    now = at or datetime.now(timezone.utc)
    try:
        quote = tdx.parse_quote(raw, allow_trailing=True)
    except tdx.QuoteError as err:
        report.add("tdx.structure", False, str(err))
        return report
    report.add("tdx.structure", True, f"TDX quote v{quote.version}, TEE 0x{quote.tee_type:x}, {len(quote.raw)} bytes")
    if quote.trailing:
        report.add(
            "tdx.trailing",
            False,
            f"{quote.trailing} bytes after the signature data were ignored (not covered by any signature)",
            required=False,
        )
    report.facts["tdx"] = {
        "mr_td": quote.report.mr_td.hex(),
        "mr_seam": quote.report.mr_seam.hex(),
        "rtmr": [r.hex() for r in quote.report.rtmr],
        "tee_tcb_svn": quote.report.tee_tcb_svn.hex(),
        "td_attributes": quote.report.td_attributes.hex(),
        "xfam": quote.report.xfam.hex(),
        "report_data": quote.report.report_data.hex(),
    }
    if expected_report_data is not None:
        bound = quote.report.report_data == expected_report_data
        report.add(
            "tdx.report_data",
            bound,
            "quote report_data equals the expected value"
            if bound
            else "quote report_data does NOT equal the expected value: not issued for your nonce",
        )
    try:
        tdx.verify_signatures(quote, at=now)
        report.add(
            "tdx.signatures",
            True,
            "attestation key, QE binding, QE report and PCK chain verify up to the pinned Intel SGX Root CA",
        )
    except tdx.QuoteError as err:
        report.add("tdx.signatures", False, str(err))
        return report
    try:
        pck = tdx.pck_info(quote.pck_leaf)
        report.facts["tdx"]["fmspc"] = pck.fmspc.hex()
        if not online:
            report.add(
                "tdx.collateral",
                False,
                "Intel collateral skipped (--offline): TCB status and revocation not checked",
                required=False,
            )
            return report
        collateral = pcs.fetch_collateral(pck.fmspc, pcs.pck_ca_kind(quote))
        pcs.verify_document(collateral.tcb_info, at=now)
        pcs.verify_document(collateral.qe_identity, at=now)
        report.add("tdx.collateral", True, "TCB info and QE identity signed by Intel (Intel PCS, fetched now)")
        ev = pcs.evaluate(quote, pck, collateral, at=now)
        report.facts["tdx"]["tcb_status"] = ev.status
        report.facts["tdx"]["tcb_level_date"] = ev.level_date
        report.facts["tdx"]["advisory_ids"] = ev.advisory_ids
        report.facts["tdx"]["tdx_module"] = {"id": ev.tdx_module_id, "status": ev.tdx_module_status}
        report.facts["tdx"]["qe_status"] = ev.qe_status
        detail = f"platform TCB {ev.status}, QE {ev.qe_status}"
        if ev.tdx_module_id:
            detail += f", {ev.tdx_module_id} {ev.tdx_module_status}"
        if ev.advisory_ids:
            detail += f", advisories {', '.join(ev.advisory_ids)}"
        report.add("tdx.tcb", ev.acceptable, detail, required=not accept_out_of_date)
        try:
            for note in pcs.check_revocation(quote, collateral, at=now):
                report.add("tdx.revocation", True, note, required=False)
        except pcs.CollateralError as err:
            report.add("tdx.revocation", False, str(err))
    except pcs.CollateralError as err:
        # Intel's own rule: a platform whose TCB components match no published level is
        # OutOfDate. With --accept-out-of-date that is a warning like any other stale TCB.
        stale = "OutOfDate" in str(err)
        if stale:
            report.facts["tdx"]["tcb_status"] = "OutOfDate"
        report.add("tdx.tcb", False, str(err), required=not (stale and accept_out_of_date))
    except tdx.QuoteError as err:
        report.add("tdx.tcb", False, str(err))
    return report

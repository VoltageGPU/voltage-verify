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

# The NVIDIA attestation modes this tool understands. Each GPU is always attested on its own by
# NRAS; on a multi-GPU node the fabric between the GPUs is a separate property these per-GPU
# tokens do NOT cover, so it is named here and surfaced as a WARN rather than left implicit.
#   single-gpu       one GPU, nothing else to say about a fabric.
#   multi-gpu-ppcie  NVIDIA Protected PCIe (Hopper HGX): NVSwitch fabric not attested.
#   multi-gpu-nvle   NVIDIA NVLink encryption (Blackwell): the encrypted NVLink fabric not attested.
KNOWN_NVIDIA_MODES = ("single-gpu", "multi-gpu-ppcie", "multi-gpu-nvle")
_FABRIC_NOT_ATTESTED: dict[str, str] = {
    "multi-gpu-ppcie": (
        "NVIDIA Protected PCIe mode: every GPU is attested on its own, but the NVSwitch fabric "
        "between them is NOT attested by these per-GPU tokens"
    ),
    "multi-gpu-nvle": (
        "NVIDIA NVLink-encryption (NVLE) mode: every GPU is attested on its own, but the "
        "encrypted NVLink fabric between them is NOT attested by these per-GPU tokens"
    ),
}


# The TD measurement registers, in quote order. On a TDVF guest MRTD is the initial firmware
# image, then the UEFI event log is extended into RTMR0 (PCR 1 and 7: firmware configuration),
# RTMR1 (PCR 2 to 6: boot loader stage) and RTMR2 (PCR 8 to 15: kernel command line and what
# the boot loader loads); RTMR3 is left to the OS and applications. Compared only to values the
# caller pins; without one, VERIFIED says nothing about the software in the TD, and a WARN
# says so rather than leaving it implicit.
MEASUREMENTS = ("mrtd", "rtmr0", "rtmr1", "rtmr2", "rtmr3")
TD_SOFTWARE_NOT_PINNED = "TD software not pinned: pass --expect-mrtd to check it"
# Without the challenge the verifier issued, a bundle replayed from another session verifies.
NO_CHALLENGE = "no challenge supplied: freshness and replay not checked"

# Offline, what a VERIFIED does not cover. Named in the verdict (and the CLI exits 3, not 0)
# unless the caller accepts it with --accept-offline-tcb.
QUOTE_TCB_OFFLINE = "signatures only, TCB not checked offline"
BUNDLE_TCB_OFFLINE = "TCB and revocation from the bundle's Intel collateral, not checked against Intel today"
ACCEPTED_OFFLINE = " (accepted with --accept-offline-tcb)"


def nvidia_mode_scope(mode: str) -> tuple[bool, str | None]:
    """For an NVIDIA attestation mode, return (is it a mode this tool knows, what the per-GPU
    tokens do not cover). Pure, so the scope wording is tested without a bundle."""
    return mode in KNOWN_NVIDIA_MODES, _FABRIC_NOT_ATTESTED.get(mode)


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
    # What a passing report did not check (offline TCB). ``ok`` stays true; the verdict names
    # each limit and the CLI exits 3 instead of 0.
    limits: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks if c.required)

    @property
    def verdict(self) -> str:
        """The RESULT line: VERIFIED, VERIFIED (limits), or NOT VERIFIED (failed checks)."""
        if not self.ok:
            return "NOT VERIFIED (" + ", ".join(self.failed()) + ")"
        return f"VERIFIED ({'; '.join(self.limits)})" if self.limits else "VERIFIED"

    def add(self, name: str, ok: bool, detail: str, required: bool = True) -> None:
        self.checks.append(Check(name, ok, detail, required))

    def failed(self) -> list[str]:
        return [c.name for c in self.checks if c.required and not c.ok]

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "verdict": self.verdict,
            "limits": list(self.limits),
            "checks": [{"name": c.name, "result": c.label, "detail": c.detail} for c in self.checks],
            "facts": self.facts,
        }


def fetch_jwks() -> dict[str, Any]:
    req = urllib.request.Request(nvidia.NRAS_JWKS_URL, headers={"User-Agent": pcs.USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def _check_debug(report: Report, quote: tdx.Quote, allow_debug: bool) -> None:
    """TD attributes bit 0. Read from the TD report itself, so it is reported even when the
    signatures fail; with ``allow_debug`` a debug TD is a warning instead of a failure."""
    raw = quote.report.td_attributes
    attrs = f"{raw.hex()} (0x{int.from_bytes(raw, 'little'):x})"  # bytes as in the quote, then the 64-bit value
    if not quote.report.debug:
        report.add("tdx.debug", True, f"TD attributes {attrs}, DEBUG (bit 0) off: not a debug TD")
        return
    detail = (
        f"TD attributes {attrs}, DEBUG (bit 0) ON: a debug TD gives no confidentiality, "
        "the host can read and modify its memory"
    )
    if allow_debug:
        detail += " (accepted with --allow-debug)"
    report.add("tdx.debug", False, detail, required=not allow_debug)


def _days(delta: timedelta) -> int:
    """Whole days, rounded up: the smallest --max-collateral-age that accepts ``delta``."""
    return -(-delta // timedelta(days=1))


def _check_collateral_age(report: Report, collateral: pcs.Collateral, now: datetime, max_age: timedelta) -> None:
    """Offline, the collateral comes from the bundle: it must not be older than ``max_age``, judged
    on the dates Intel signed (pcs.signed_dates). Required, unlike the online warning."""
    try:
        dates = pcs.signed_dates(collateral)
    except pcs.CollateralError as err:
        report.add("tdx.collateral_fresh", False, f"{err}: refetch online")
        return
    age, root_age = now - dates.oldest_issued, now - dates.root_crl_issued
    limit = f"the {max_age.days}-day limit" if max_age % timedelta(days=1) == timedelta(0) else f"the {max_age} limit"
    # The root CA CRL lives a year: within its nextUpdate, or no older than the limit itself.
    root_current = dates.root_crl_next is not None and now <= dates.root_crl_next
    root_until = f"{dates.root_crl_next:%Y-%m-%d}" if dates.root_crl_next else "none"
    problems = []
    if age > max_age:
        problems.append(
            f"Intel collateral is {age.days} days old ({dates.oldest} issued {dates.oldest_issued:%Y-%m-%d}), over {limit}"
        )
    if not root_current and root_age > max_age:
        problems.append(f"root CA CRL past its nextUpdate ({root_until})")
    if problems:
        needed = _days(age if root_current else max(age, root_age))
        hint = f": refetch online, or accept it with --max-collateral-age {needed}"
        report.add("tdx.collateral_fresh", False, "; ".join(problems) + hint)
        return
    root = f"root CA CRL valid until {root_until}"
    if not root_current:
        root = f"root CA CRL past its nextUpdate ({root_until}), within {limit}"
    report.add(
        "tdx.collateral_fresh",
        True,
        f"Intel collateral issued {dates.oldest_issued:%Y-%m-%d} ({dates.oldest}), {max(age.days, 0)} day(s) old, "
        f"within {limit}; {root}",
    )


def _pinned(expected: dict[str, bytes] | None) -> dict[str, bytes]:
    """The caller's pinned registers; an unknown name is a programming error, not a failed check."""
    unknown = sorted(set(expected or {}) - set(MEASUREMENTS))
    if unknown:
        raise ValueError(f"unknown measurement register(s) {unknown}, expected some of {MEASUREMENTS}")
    return dict(expected or {})


def _check_measurements(report: Report, quote: tdx.Quote, expected: dict[str, bytes]) -> None:
    """MRTD and RTMR0..3 against the values the caller pinned (``expected``, by register name).
    Read from the TD report itself, like the debug bit; only Intel's word once the signatures pass."""
    if not expected:
        report.add("tdx.measurements", False, TD_SOFTWARE_NOT_PINNED, required=False)
        return
    got = dict(zip(MEASUREMENTS, (quote.report.mr_td, *quote.report.rtmr), strict=True))
    wrong = [m for m in MEASUREMENTS if m in expected and expected[m] != got[m]]
    if wrong:
        differs = "; ".join(f"{m.upper()} is {got[m].hex()[:16]}..., expected {expected[m].hex()[:16]}..." for m in wrong)
        report.add("tdx.measurements", False, f"{differs}: not the TD software you pinned")
        return
    loose = [m.upper() for m in MEASUREMENTS if m not in expected]
    detail = ", ".join(m.upper() for m in MEASUREMENTS if m in expected) + " as expected"
    report.add("tdx.measurements", True, detail + (f"; not pinned: {', '.join(loose)}" if loose else ""))


def verify_bundle(
    bundle: Bundle,
    *,
    expected_challenge: str | None = None,
    expected_image_digest: str | None = None,
    online: bool = True,
    allowed_hwmodels: set[str] | None = None,
    require_devices: int | None = None,
    allow_debug: bool = False,
    expected_measurements: dict[str, bytes] | None = None,
    max_collateral_age: timedelta = pcs.DEFAULT_MAX_COLLATERAL_AGE,
    accept_offline_tcb: bool = False,
    at: datetime | None = None,
) -> Report:
    """Run every check. Never raises for a verification failure; returns a Report instead.

    ``expected_measurements``: MRTD and RTMR0..3 to pin, by name (``MEASUREMENTS``), 48 bytes each.
    ``max_collateral_age``: offline, the oldest Intel collateral accepted from the bundle.
    ``accept_offline_tcb``: offline, do not list BUNDLE_TCB_OFFLINE in ``limits``.
    """
    pinned = _pinned(expected_measurements)
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
    else:
        report.add("manifest.challenge", False, NO_CHALLENGE, required=False)
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
        _check_debug(report, quote, allow_debug)
        bound = quote.report.report_data == com.sha512
        report.add(
            "tdx.report_data",
            bound,
            "quote report_data equals SHA-512 of the manifest"
            if bound
            else "quote report_data does NOT equal SHA-512 of the manifest: this quote was not issued for this workload",
        )
        _check_measurements(report, quote, pinned)
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
                    if accept_offline_tcb:
                        source += ", accepted with --accept-offline-tcb"
                    else:
                        report.limits.append(BUNDLE_TCB_OFFLINE)
                if collateral is None:
                    report.add("tdx.collateral", False, "no Intel collateral: run online or use a bundle that embeds it")
                else:
                    pcs.verify_document(collateral.tcb_info, at=now)
                    pcs.verify_document(collateral.qe_identity, at=now)
                    report.add("tdx.collateral", True, f"TCB info and QE identity signed by Intel ({source})")
                    if not online:
                        _check_collateral_age(report, collateral, now, max_collateral_age)
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
                    if online and (ev.tcb_info_expired or ev.qe_identity_expired):  # offline: tdx.collateral_fresh above
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
    # chain to a pinned NRAS intermediate (nvidia._load_chain), so a bundle cannot bring its
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
                "a pinned NRAS intermediate"
                if sources == {nvidia.PINNED_INTERMEDIATE}
                else "an NRAS intermediate presented by NVIDIA's live JWKS"
            )
            report.add(
                "nvidia.signatures",
                True,
                f"{1 + len(res.devices)} NVIDIA-signed tokens verify, each GPU token listed by its digest in the "
                f"verifier token, keys from the bundle chained to {anchor} ({live_note})",
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
            # The verifier token and every GPU token carry eat_nonce; all must be the manifest's.
            report.add(
                "nvidia.nonce",
                not res.nonce_mismatch,
                f"NRAS eat_nonce equals SHA-256 of the manifest (verifier token and {len(res.devices)} GPU token(s))"
                if not res.nonce_mismatch
                else f"NRAS eat_nonce does NOT equal SHA-256 of the manifest in {', '.join(res.nonce_mismatch)}: "
                "not issued for this workload",
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

    # 5. Mode and scope. The mode is recorded in the bundle by the attester; an unknown value is a
    # failure (it must not pass silently), and for a multi-GPU mode the fabric the tokens do not
    # cover is stated outright as a WARN.
    known, fabric = nvidia_mode_scope(bundle.nvidia_mode)
    report.add(
        "nvidia.mode",
        known,
        f"attestation mode {bundle.nvidia_mode!r}"
        if known
        else f"attestation mode {bundle.nvidia_mode!r} is not one this tool knows ({', '.join(KNOWN_NVIDIA_MODES)})",
    )
    if fabric:
        report.add("nvidia.scope", False, f"not verified: {fabric}", required=False)

    # 6. Measured composition. When the TD runs a provider-measured software stack (e.g. dstack,
    # whose compose hash lands in RTMR3), the trust boundary is the whole measured VM, not a VM
    # the tenant alone controls. This tool records the RTMRs but compares them to no published
    # reference, so the composition is trusted, not proven: said plainly rather than left out.
    env = bundle.environment if isinstance(bundle.environment, dict) else {}
    dstack = env.get("dstack") if isinstance(env.get("dstack"), dict) else None
    if dstack is not None:
        app_id = dstack.get("app_id")
        report.add(
            "tdx.measured_compose",
            False,
            "not verified: this TD runs a provider-measured composition (dstack"
            + (f", app_id {app_id}" if app_id else "")
            + "); RTMR3 records it but this tool compares it to no published reference, so the "
            "measured composition is trusted, not proven",
            required=False,
        )

    return report


def verify_quote(
    raw: bytes,
    *,
    expected_report_data: bytes | None = None,
    online: bool = True,
    accept_out_of_date: bool = False,
    allow_debug: bool = False,
    expected_measurements: dict[str, bytes] | None = None,
    accept_offline_tcb: bool = False,
    at: datetime | None = None,
) -> Report:
    """Verify a bare Intel TDX quote from any provider: structure, debug bit, signature chain up
    to the pinned Intel root, then TCB status, QE identity and revocation from Intel PCS.

    No manifest, no NVIDIA token, no VoltageGPU anywhere in the path. The caller can bind
    ``expected_report_data``, the 64 bytes the guest put in the report (typically the hash of a
    nonce the verifier issued), and pin ``expected_measurements`` (MRTD, RTMR0..3). Never
    raises for a verification failure; returns a Report.
    """
    pinned = _pinned(expected_measurements)
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
    _check_debug(report, quote, allow_debug)
    if expected_report_data is not None:
        bound = quote.report.report_data == expected_report_data
        report.add(
            "tdx.report_data",
            bound,
            "quote report_data equals the expected value"
            if bound
            else "quote report_data does NOT equal the expected value: not issued for your nonce",
        )
    _check_measurements(report, quote, pinned)
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
                "Intel collateral skipped (--offline): TCB status and revocation not checked"
                + (ACCEPTED_OFFLINE if accept_offline_tcb else ""),
                required=False,
            )
            if not accept_offline_tcb:
                report.limits.append(QUOTE_TCB_OFFLINE)
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

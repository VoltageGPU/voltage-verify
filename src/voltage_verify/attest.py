# SPDX-License-Identifier: MIT
"""The part that runs inside the attested VM, as root.

It reads the manifest the verifier prepared, derives the two commitments, asks the Intel TDX
guest driver for a quote over ``commitment512`` (through the kernel's configfs TSM
interface), asks NVIDIA's attestation service for GPU claims over ``commitment256`` (through
the nv-attestation-sdk), fetches the public collateral needed for offline verification, and
writes one bundle. It chooses nothing: every hash comes from the manifest.
"""

from __future__ import annotations

import errno
import json
import platform
import secrets
import shutil
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import pcs, tdx
from .bundle import Bundle
from .canonical import commitments
from .manifest import validate_manifest

TSM_REPORT_DIR = Path("/sys/kernel/config/tsm/report")
NRAS_GPU_URL = "https://nras.attestation.nvidia.com/v3/attest/gpu"
NRAS_OCSP_URL = "https://ocsp.ndis.nvidia.com/"
NRAS_RIM_URL = "https://rim.attestation.nvidia.com/v1/rim/"
NRAS_JWKS_URL = "https://nras.attestation.nvidia.com/.well-known/jwks.json"


class AttestError(RuntimeError):
    """Raised when evidence cannot be produced on this machine."""


def tsm_quote(report_data: bytes, attempts: int = 5) -> bytes:
    """Request a TDX quote through configfs TSM. Retries only on EINVAL (concurrent request)."""
    if len(report_data) != 64:
        raise AttestError("report_data must be exactly 64 bytes")
    if not TSM_REPORT_DIR.is_dir():
        raise AttestError(f"{TSM_REPORT_DIR} is missing: not a TDX guest, or configfs TSM not mounted (run as root)")
    for attempt in range(attempts):
        entry = TSM_REPORT_DIR / f"voltage-verify-{secrets.token_hex(4)}"
        try:
            entry.mkdir()
            (entry / "inblob").write_bytes(report_data)
            quote = (entry / "outblob").read_bytes()
            provider = (entry / "provider").read_text().strip() if (entry / "provider").exists() else ""
            if provider and provider != "tdx_guest":
                raise AttestError(f"TSM provider is {provider!r}, not tdx_guest")
            return quote
        except OSError as err:
            if err.errno != errno.EINVAL or attempt == attempts - 1:
                raise AttestError(f"TSM quote request failed: {err}") from err
            time.sleep(0.05 * (attempt + 1))
        finally:
            try:
                entry.rmdir()
            except OSError:
                pass
    raise AttestError("TSM quote request failed after retries")


def gpu_count() -> int:
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        return int(pynvml.nvmlDeviceGetCount())
    except Exception:  # noqa: BLE001
        smi = shutil.which("nvidia-smi")
        if not smi:
            return 0
        out = subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=30, check=False).stdout
        return sum(1 for line in out.splitlines() if line.startswith("GPU "))


def nras_attest(nonce_hex: str, mode: str) -> Any:
    """Run the nv-attestation-sdk against NRAS with our nonce. Returns the parsed token array."""
    try:
        from nv_attestation_sdk import attestation  # type: ignore
    except ImportError as err:
        raise AttestError("nv-attestation-sdk is not installed: pip install 'voltage-verify[attest]'") from err
    client = attestation.Attestation()
    client.set_name("voltage-verify")
    client.set_nonce(nonce_hex)
    client.set_claims_version("2.0")
    client.add_verifier(
        attestation.Devices.GPU,
        attestation.Environment.REMOTE,
        NRAS_GPU_URL,
        "",
        ocsp_url=NRAS_OCSP_URL,
        rim_url=NRAS_RIM_URL,
    )
    if mode == "multi-gpu-ppcie":
        # On a multi-GPU node in NVIDIA Protected PCIe mode the SDK refuses "standalone"
        # attestation; ppcie_mode=False means "not standalone" for this SDK version.
        options = {"ppcie_mode": False}
        try:
            evidence = client.get_evidence(options=options)
        except TypeError:
            evidence = client.get_evidence(options)
        try:
            ok = client.attest(evidence, options=options)
        except TypeError:
            ok = client.attest(evidence)
    else:
        evidence = client.get_evidence()
        ok = client.attest(evidence)
    token = client.get_token()
    if not token:
        raise AttestError("NRAS returned no token")
    parsed = json.loads(token) if isinstance(token, str) else token
    if not ok:
        raise AttestError("NRAS attestation did not succeed (token kept for diagnosis): " + json.dumps(parsed)[:300])
    return parsed


def fetch_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": pcs.USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def environment_facts() -> dict[str, Any]:
    facts: dict[str, Any] = {"kernel": platform.release(), "python": platform.python_version()}
    smi = shutil.which("nvidia-smi")
    if smi:
        for key, args in (("gpus", ["-L"]), ("conf_compute", ["conf-compute", "-q"])):
            try:
                run = subprocess.run([smi, *args], capture_output=True, text=True, timeout=30, check=False)
                facts[key] = run.stdout.strip()
            except (OSError, subprocess.SubprocessError):
                pass
    return facts


def run_attest(manifest_path: Path, out_path: Path, mode: str = "auto", with_collateral: bool = True) -> Bundle:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    com = commitments(manifest)

    quote_raw = tsm_quote(com.sha512)
    quote = tdx.parse_quote(quote_raw)
    if quote.report.report_data != com.sha512:
        raise AttestError("the TDX quote does not carry the requested report_data")

    if mode == "auto":
        mode = "multi-gpu-ppcie" if gpu_count() > 1 else "single-gpu"
    nras = nras_attest(com.sha256_hex, mode)

    # The NRAS JWKS is always embedded, even with --no-collateral: since 0.2.1 the verifier
    # takes NVIDIA signing keys from the bundle only, so a bundle without it cannot verify.
    jwks: dict[str, Any] | None = None
    try:
        jwks = fetch_json(NRAS_JWKS_URL)
    except Exception as err:  # noqa: BLE001
        print(f"warning: could not fetch NRAS JWKS, this bundle will NOT verify (keys come from the bundle): {err}")
    collateral: dict[str, Any] | None = None
    if with_collateral:
        try:
            pck = tdx.pck_info(quote.pck_leaf)
            collateral = {"intel": pcs.fetch_collateral(pck.fmspc, pcs.pck_ca_kind(quote)).to_json()}
        except Exception as err:  # noqa: BLE001
            print(f"warning: could not fetch Intel collateral for offline verification: {err}")

    bundle = Bundle(
        manifest=manifest,
        commitments={"sha256": com.sha256_hex, "sha512": com.sha512_hex},
        quote=quote.raw,
        report_data=com.sha512,
        nvidia_mode=mode,
        nras=nras,
        jwks=jwks,
        collateral=collateral,
        environment=environment_facts(),
        created_at=datetime.now(timezone.utc),
    )
    bundle.save(out_path)
    return bundle

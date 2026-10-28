# SPDX-License-Identifier: MIT
"""Produce a `voltage-verify-bundle/1` from inside a dstack-based confidential TD.

Some providers run the GPU workload in a container inside an Intel TDX guest managed by
dstack (https://github.com/Dstack-TEE/dstack). There, the tenant has no `/dev/tdx_guest` and no
configfs TSM: the TDX quote is served by the dstack guest agent over a unix socket
(`GET /GetQuote?report_data=<hex>`), and `GET /Info` describes the measured application. The
stock `voltage-verify attest` talks to configfs TSM, so it cannot run here; this collector
takes the same manifest, gets the quote from the socket, runs the NVIDIA attestation for every
GPU, and writes the identical bundle format.

It chooses nothing. The 64-byte report_data it asks dstack for is the SHA-512 of the manifest
the verifier prepared; the NVIDIA nonce is the SHA-256 of the same manifest. The quote dstack
returns is then parsed and its report_data is checked to equal that SHA-512 byte for byte: if
dstack hashed or transformed the input instead of using it raw, the bundle would not bind, so
this collector refuses to write one and says exactly what it observed (see `classify_report_data`).

What it does NOT establish, and the bundle/verifier say so: the encrypted NVLink fabric between
the GPUs (per-GPU NRAS tokens do not cover it) and the dstack-measured composition recorded in
RTMR3 (no published reference to compare it against; the whole measured VM is the trust boundary).

Usage, inside the container, as the workload user:
    python3 collect.py --manifest manifest.json --socket /var/run/dstack.sock \
        --mode multi-gpu-nvle --out bundle.json
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_SOCKETS = ("/var/run/dstack.sock", "/run/dstack.sock")
MAX_HTTP_BODY = 4 * 1024 * 1024  # a quote + event log is tens of KB; cap defensively
# Keys of dstack /Info and /GetQuote worth keeping in the bundle environment so a reader knows
# what else runs in this TD. The bulky/opaque ones (the raw quote, the app certificate) are left out.
INFO_KEYS = ("app_id", "instance_id", "app_name", "tcb_info", "event_log", "app_compose", "compose_hash", "rtmr3")


class CollectError(RuntimeError):
    """Raised when evidence cannot be produced honestly on this machine."""


def http_body(raw: bytes, path: str) -> bytes:
    """The body of an HTTP/1.0 response, once a 200 status line is confirmed. Pure (no socket)."""
    sep = raw.find(b"\r\n\r\n")
    if sep < 0:
        raise CollectError(f"dstack returned a malformed HTTP response on {path}")
    status_line = raw[: raw.find(b"\r\n")].decode("latin-1", "replace")
    if " 200" not in status_line:
        raise CollectError(f"dstack answered {status_line.strip()!r} on {path}")
    return raw[sep + 4 :]


def unix_http_get(socket_path: str, path: str, timeout: float = 15.0) -> bytes:
    """Minimal HTTP/1.0 GET over a unix-domain socket (no third-party dependency)."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(socket_path)
        request = f"GET {path} HTTP/1.0\r\nHost: localhost\r\nConnection: close\r\n\r\n"
        sock.sendall(request.encode("ascii"))
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_HTTP_BODY:
                raise CollectError(f"dstack response on {path} exceeds {MAX_HTTP_BODY} bytes")
    finally:
        sock.close()
    return http_body(b"".join(chunks), path)


def dstack_get_json(socket_path: str, path: str) -> dict[str, Any]:
    body = unix_http_get(socket_path, path)
    try:
        value = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as err:
        raise CollectError(f"dstack response on {path} is not JSON: {err}") from err
    if not isinstance(value, dict):
        raise CollectError(f"dstack response on {path} is not a JSON object")
    return value


def parse_quote_hex(value: Any) -> bytes:
    """The quote field dstack returns, as bytes. Accepts hex with optional 0x/whitespace."""
    if not isinstance(value, str):
        raise CollectError("dstack GetQuote returned no 'quote' string")
    text = value.strip()
    if text[:2].lower() == "0x":
        text = text[2:]
    text = "".join(text.split())
    try:
        return bytes.fromhex(text)
    except ValueError as err:
        raise CollectError(f"dstack GetQuote 'quote' is not hex: {err}") from err


def classify_report_data(observed: bytes, commitment512: bytes, commitment256: bytes, sent_hex: str) -> str:
    """Name how dstack placed our report_data, so a mismatch is explained rather than guessed.

    ``observed`` is the 64-byte report_data read back from the quote; ``sent_hex`` is the exact
    query value we sent. Only ``raw`` lets the standard bundle bind; everything else is a reason
    to refuse to write one.
    """
    if observed == commitment512:
        return "raw"
    candidates = {
        "sha512(report_data bytes)": hashlib.sha512(commitment512).digest(),
        "sha512(report_data hex-ascii)": hashlib.sha512(sent_hex.encode()).digest(),
        "sha256(report_data bytes), zero-padded": hashlib.sha256(commitment512).digest() + b"\x00" * 32,
        "sha256(report_data hex-ascii), zero-padded": hashlib.sha256(sent_hex.encode()).digest() + b"\x00" * 32,
        "the NVIDIA nonce (sha256 of manifest), zero-padded": commitment256 + b"\x00" * 32,
        "all zero (empty/ignored report_data)": b"\x00" * 64,
    }
    for label, value in candidates.items():
        if observed == value:
            return label
    return "unknown"


def environment_facts(info: dict[str, Any], quote_response: dict[str, Any]) -> dict[str, Any]:
    """Kernel/GPU facts plus a dstack block: what the reader needs to know runs in this TD."""
    from voltage_verify.attest import environment_facts as base_facts

    facts = base_facts()
    # Record the exact NVIDIA SDK that produced the tokens: Blackwell support depends on it.
    with contextlib.suppress(Exception):
        from importlib.metadata import version

        facts["nv_attestation_sdk"] = version("nv-attestation-sdk")
    dstack: dict[str, Any] = {}
    for source in (info, quote_response):
        for key in INFO_KEYS:
            if key in source and key not in dstack:
                dstack[key] = source[key]
    facts["dstack"] = dstack
    return facts


def collect(manifest_path: Path, socket_path: str, out_path: Path, mode: str, with_collateral: bool = True) -> Any:
    from voltage_verify import pcs, tdx
    from voltage_verify.attest import fetch_json, nras_attest
    from voltage_verify.bundle import Bundle
    from voltage_verify.canonical import commitments
    from voltage_verify.manifest import validate_manifest

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    com = commitments(manifest)

    sent_hex = com.sha512_hex  # 128 hex chars = the 64-byte report_data we want in the quote
    info = dstack_get_json(socket_path, "/Info")
    quote_response = dstack_get_json(socket_path, f"/GetQuote?report_data={sent_hex}")
    quote_raw = parse_quote_hex(quote_response.get("quote"))
    quote = tdx.parse_quote(quote_raw, allow_trailing=True)

    binding = classify_report_data(quote.report.report_data, com.sha512, com.sha256, sent_hex)
    if binding != "raw":
        raise CollectError(
            "the dstack quote does NOT carry our manifest hash as report_data; refusing to write a "
            f"bundle that cannot bind.\n  expected (SHA-512 of manifest): {com.sha512_hex}\n"
            f"  observed in quote:              {quote.report.report_data.hex()}\n"
            f"  dstack appears to apply:        {binding}\n"
            "If dstack hashes report_data, the standard bundle format cannot represent this path; "
            "report this before publishing anything."
        )

    nras = nras_attest(com.sha256_hex, mode)

    jwks: dict[str, Any] | None = None
    try:
        jwks = fetch_json("https://nras.attestation.nvidia.com/.well-known/jwks.json")
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
        environment=environment_facts(info, quote_response),
        created_at=datetime.now(timezone.utc),
    )
    bundle.save(out_path)
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="collect.py", description="Build a voltage-verify bundle inside a dstack TD.")
    parser.add_argument("--manifest", required=True, help="manifest.json prepared on the verifier's machine")
    parser.add_argument("-o", "--output", default="bundle.json")
    parser.add_argument("--socket", default=None, help="dstack guest-agent unix socket (default: autodetect)")
    parser.add_argument(
        "--mode",
        choices=["single-gpu", "multi-gpu-ppcie", "multi-gpu-nvle"],
        default="multi-gpu-nvle",
        help="NVIDIA attestation mode; multi-gpu-nvle for a Blackwell NVLink-encryption node",
    )
    parser.add_argument("--no-collateral", action="store_true", help="do not embed Intel collateral")
    args = parser.parse_args(argv)

    sock = args.socket
    if sock is None:
        sock = next((s for s in DEFAULT_SOCKETS if Path(s).exists()), None)
        if sock is None:
            print(f"error: no dstack socket found at {DEFAULT_SOCKETS}; pass --socket", file=sys.stderr)
            return 2
    try:
        bundle = collect(Path(args.manifest), sock, Path(args.output), args.mode, with_collateral=not args.no_collateral)
    except Exception as err:  # noqa: BLE001
        print(f"error: {err}", file=sys.stderr)
        return 2
    print(f"bundle written to {args.output} ({bundle.nvidia_mode}, {len(bundle.quote)}-byte quote)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

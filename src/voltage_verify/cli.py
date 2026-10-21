# SPDX-License-Identifier: MIT
"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .bundle import Bundle, BundleError
from .canonical import commitments
from .manifest import ManifestError, build_manifest
from .registry import RegistryError
from .verify import Report, verify_bundle, verify_quote

ALLOW_DEBUG_HELP = "report a TD in debug mode as a warning, not a failure (its host can read and modify its memory)"


def _print_report(report: Report) -> None:
    width = max(len(c.name) for c in report.checks) if report.checks else 20
    for check in report.checks:
        print(f"[{check.label}] {check.name.ljust(width)}  {check.detail}")
    facts = report.facts
    if "tdx" in facts:
        t = facts["tdx"]
        print(f"\nTDX: MRTD {t['mr_td'][:32]}...  TCB {t.get('tcb_status', '?')}  QE {t.get('qe_status', '?')}")
    if "nvidia" in facts:
        for d in facts["nvidia"]["devices"]:
            print(f"GPU: {d['name']} {d['hwmodel']} driver {d['driver']} vbios {d['vbios']} measurements {d['measres']}")
    print("\nRESULT:", "VERIFIED" if report.ok else "NOT VERIFIED (" + ", ".join(report.failed()) + ")")


def cmd_manifest(args: argparse.Namespace) -> int:
    extra: dict[str, str] = {}
    for item in args.extra or []:
        if "=" not in item:
            print(f"--extra expects key=value, got {item!r}", file=sys.stderr)
            return 2
        key, value = item.split("=", 1)
        extra[key] = value
    try:
        manifest = build_manifest(
            image=args.image,
            image_digest=args.image_digest,
            artifacts=[Path(p) for p in (args.artifact or [])],
            extra=extra,
            statement=args.statement,
            challenge=None if args.challenge in (None, "auto") else args.challenge,
        )
    except (ManifestError, RegistryError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    Path(args.output).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    com = commitments(manifest)
    print(f"manifest written to {args.output}")
    print(f"challenge        {manifest['challenge']}")
    print(f"commitment256    {com.sha256_hex}   (NVIDIA nonce)")
    print(f"commitment512    {com.sha512_hex[:64]}...   (TDX report_data)")
    print("keep the challenge: pass it to `verify --challenge` to reject replayed bundles")
    return 0


def cmd_attest(args: argparse.Namespace) -> int:
    from .attest import AttestError, run_attest  # imported late: VM-only dependencies

    try:
        bundle = run_attest(Path(args.manifest), Path(args.output), mode=args.mode, with_collateral=not args.no_collateral)
    except AttestError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    print(f"bundle written to {args.output} ({bundle.nvidia_mode}, {len(bundle.quote)}-byte quote)")
    return 0


def _load(path: str) -> Bundle | None:
    try:
        return Bundle.load(Path(path))
    except BundleError as err:
        print(f"error: {err}", file=sys.stderr)
        return None


def cmd_verify(args: argparse.Namespace) -> int:
    bundle = _load(args.bundle)
    if bundle is None:
        return 2
    report = verify_bundle(
        bundle,
        expected_challenge=args.challenge,
        expected_image_digest=args.image_digest,
        online=not args.offline,
        allowed_hwmodels=set(args.hwmodel) if args.hwmodel else None,
        require_devices=args.gpus,
        allow_debug=args.allow_debug,
    )
    if args.json:
        print(json.dumps(report.to_json(), indent=2))
    else:
        _print_report(report)
    return 0 if report.ok else 1


def _read_quote_bytes(path: str) -> bytes:
    """A quote file as written by configfs TSM (raw), or as people paste it (hex, base64)."""
    raw = Path(path).read_bytes()
    text = raw.strip()
    if text and all(c in b"0123456789abcdefABCDEF\r\n" for c in text):
        return bytes.fromhex(text.decode("ascii").replace("\n", "").replace("\r", ""))
    if text and all(c in b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\r\n" for c in text):
        import base64

        try:
            return base64.b64decode(text, validate=False)
        except Exception:  # noqa: BLE001
            return raw
    return raw


def cmd_quote(args: argparse.Namespace) -> int:
    try:
        raw = _read_quote_bytes(args.quote)
    except OSError as err:
        print(f"cannot read {args.quote}: {err}", file=sys.stderr)
        return 2
    expected = None
    if args.report_data:
        try:
            expected = bytes.fromhex(args.report_data)
        except ValueError:
            print("--report-data must be hex", file=sys.stderr)
            return 2
        if len(expected) != 64:
            print(f"--report-data must be 64 bytes (got {len(expected)})", file=sys.stderr)
            return 2
    report = verify_quote(
        raw,
        expected_report_data=expected,
        online=not args.offline,
        accept_out_of_date=args.accept_out_of_date,
        allow_debug=args.allow_debug,
    )
    if args.json:
        print(json.dumps(report.to_json(), indent=2))
    else:
        _print_report(report)
    return 0 if report.ok else 1


def cmd_inspect(args: argparse.Namespace) -> int:
    bundle = _load(args.bundle)
    if bundle is None:
        return 2
    data = bundle.to_json()
    data["tdx"]["quote_b64"] = f"<{len(bundle.quote)} bytes>"
    data["nvidia"]["jwks"] = "<embedded>" if bundle.jwks else None
    data["collateral"] = "<embedded>" if bundle.collateral else None
    print(json.dumps(data, indent=2, sort_keys=True))
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    from .selftest import run_selftest

    bundle = _load(args.bundle)
    if bundle is None:
        return 2
    clean, results = run_selftest(bundle, online=not args.offline, expected_challenge=args.challenge)
    print("clean bundle:", "VERIFIED" if clean.ok else "NOT VERIFIED (" + ", ".join(clean.failed()) + ")")
    all_ok = clean.ok
    for r in results:
        all_ok = all_ok and r.passed
        rejected = ", ".join(r.failed_checks) or "nothing"
        print(f"[{'PASS' if r.passed else 'FAIL'}] {r.name.ljust(18)} {r.description}  ->  rejected on {rejected}")
    print("\nSELFTEST:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="voltage-verify",
        description="Bind TDX and NVIDIA attestations to a workload and verify them.",
    )
    p.add_argument("--version", action="version", version=f"voltage-verify {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    m = sub.add_parser("manifest", help="build the workload manifest on the verifier's machine")
    m.add_argument("--image", help="container image reference, e.g. ghcr.io/org/app:1.2.3")
    m.add_argument("--image-digest", help="sha256:... digest, when the registry cannot be queried")
    m.add_argument("--artifact", action="append", help="file to hash (model weights, config); repeatable")
    m.add_argument("--extra", action="append", help="key=value to include in the manifest; repeatable")
    m.add_argument("--statement", help="free text describing the run")
    m.add_argument("--challenge", default="auto", help="32-byte hex challenge, or 'auto' to generate one")
    m.add_argument("-o", "--output", default="manifest.json")
    m.set_defaults(func=cmd_manifest)

    a = sub.add_parser("attest", help="produce the evidence bundle; run as root inside the TDX VM")
    a.add_argument("--manifest", required=True)
    a.add_argument("-o", "--output", default="bundle.json")
    a.add_argument("--mode", choices=["auto", "single-gpu", "multi-gpu-ppcie"], default="auto")
    a.add_argument(
        "--no-collateral",
        action="store_true",
        help="do not embed Intel collateral (NVIDIA's JWKS is always embedded: keys are taken from the bundle)",
    )
    a.set_defaults(func=cmd_attest)

    v = sub.add_parser("verify", help="verify a bundle")
    v.add_argument("bundle")
    v.add_argument("--challenge", help="the challenge you issued; rejects replayed bundles")
    v.add_argument("--image-digest", help="expected sha256:... of the workload image")
    v.add_argument("--offline", action="store_true", help="use the collateral embedded in the bundle")
    v.add_argument("--hwmodel", action="append", help="allowed GPU model claim, e.g. GH100; repeatable")
    v.add_argument("--gpus", type=int, help="exact number of attested GPUs expected")
    v.add_argument("--allow-debug", action="store_true", help=ALLOW_DEBUG_HELP)
    v.add_argument("--json", action="store_true")
    v.set_defaults(func=cmd_verify)

    q = sub.add_parser("quote", help="verify a bare Intel TDX quote from any provider (no manifest, no NVIDIA)")
    q.add_argument("quote", help="quote file: raw bytes as written by /sys/kernel/config/tsm/report, or hex, or base64")
    q.add_argument("--report-data", help="expected 64-byte report_data as hex, e.g. the SHA-512 of the nonce you issued")
    q.add_argument("--accept-out-of-date", action="store_true", help="report a non-UpToDate TCB as a warning, not a failure")
    q.add_argument("--allow-debug", action="store_true", help=ALLOW_DEBUG_HELP)
    q.add_argument("--offline", action="store_true", help="structure and signatures only, no Intel PCS call")
    q.add_argument("--json", action="store_true")
    q.set_defaults(func=cmd_quote)

    i = sub.add_parser("inspect", help="print a bundle without the bulky fields")
    i.add_argument("bundle")
    i.set_defaults(func=cmd_inspect)

    s = sub.add_parser("selftest", help="verify a bundle, then check that six mutations are rejected")
    s.add_argument("bundle")
    s.add_argument("--challenge")
    s.add_argument("--offline", action="store_true")
    s.set_defaults(func=cmd_selftest)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

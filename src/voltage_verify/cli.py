# SPDX-License-Identifier: MIT
"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

from . import __version__
from .bundle import Bundle, BundleError
from .canonical import commitments
from .health.report import MAX_REPORT_BYTES
from .manifest import ManifestError, build_manifest
from .pcs import DEFAULT_MAX_COLLATERAL_AGE
from .registry import RegistryError
from .verify import MEASUREMENTS, Report, malformed_report, verify_bundle, verify_quote

ALLOW_DEBUG_HELP = "report a TD in debug mode as a warning, not a failure (its host can read and modify its memory)"
MEASUREMENT_HELP = {
    "mrtd": "expected MRTD, 48 bytes as hex: the TD's initial firmware image",
    "rtmr0": "expected RTMR0 (48-byte hex): firmware configuration",
    "rtmr1": "expected RTMR1 (48-byte hex): boot loader stage",
    "rtmr2": "expected RTMR2 (48-byte hex): kernel command line and what the boot loader loads",
    "rtmr3": "expected RTMR3 (48-byte hex): left to the OS and applications",
}
MAX_AGE_HELP = (
    "offline: refuse Intel collateral from the bundle older than DAYS days "
    f"(default {DEFAULT_MAX_COLLATERAL_AGE.days}, the validity Intel gives TCB info, QE identity and the PCK CRL)"
)
ACCEPT_OFFLINE_HELP = (
    "with --offline: accept a result whose TCB status and revocation were not checked against Intel's "
    "collateral of today (exit code 0 instead of 3)"
)
TRUST_LIVE_HELP = (
    "online: also accept an NRAS intermediate that NVIDIA's live JWKS presents but this release does not pin "
    "(after a rotation by NVIDIA); off by default, only NVIDIA GPU Intermediate 004 and 005 are trusted"
)
HEALTH_HELP = (
    "the health report the manifest commits to (`manifest --health`): checked against that commitment, "
    "and its verdict repeated (health.report)"
)
# Exit codes: 0 verified, 1 a required check failed, 2 unusable input, 3 verified with a limit
# named in the verdict (offline TCB), which --accept-offline-tcb accepts.
EXIT_LIMITED = 3


def _days(value: str) -> timedelta:
    """--max-collateral-age: a whole number of days, 0 to 999999."""
    if not re.fullmatch(r"[0-9]{1,6}", value):
        raise argparse.ArgumentTypeError(f"expected a whole number of days, got {value[:20]!r}")
    return timedelta(days=int(value))


def _expected_measurements(args: argparse.Namespace) -> dict[str, bytes]:
    """The --expect-* values, each 48 bytes of hex. Raises ValueError naming the bad option."""
    pinned: dict[str, bytes] = {}
    for name in MEASUREMENTS:
        value = getattr(args, f"expect_{name}")
        if value is None:
            continue
        try:
            raw = bytes.fromhex(value)
        except ValueError:
            raise ValueError(f"--expect-{name} must be hex") from None
        if len(raw) != 48:
            raise ValueError(f"--expect-{name} must be 48 bytes (96 hex characters), got {len(raw)}")
        pinned[name] = raw
    return pinned


def _exit_code(report: Report) -> int:
    if not report.ok:
        return 1
    return EXIT_LIMITED if report.limits else 0


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
    print("\nRESULT:", report.verdict)


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
            health=Path(args.health) if args.health else None,
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
    if "health" in manifest:
        print(f"health report    {manifest['health']['sha256']}   (committed: pass the file to `verify --health`)")
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


def _show(report: Report, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report.to_json(), indent=2))
    else:
        _print_report(report)


def cmd_verify(args: argparse.Namespace) -> int:
    try:
        pinned = _expected_measurements(args)
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    try:
        bundle = Bundle.load(Path(args.bundle))
    except BundleError as err:
        print(f"error: {err}", file=sys.stderr)
        _show(malformed_report(str(err)), args.json)
        return 2
    health_report: bytes | None = None
    if args.health:
        try:
            with Path(args.health).open("rb") as handle:
                health_report = handle.read(MAX_REPORT_BYTES + 1)  # an oversized file fails health.report
        except OSError as err:
            print(f"error: cannot read {args.health}: {type(err).__name__}", file=sys.stderr)
            return 2
    try:
        report = verify_bundle(
            bundle,
            expected_challenge=args.challenge,
            expected_image_digest=args.image_digest,
            online=not args.offline,
            allowed_hwmodels=set(args.hwmodel) if args.hwmodel else None,
            require_devices=args.gpus,
            allow_debug=args.allow_debug,
            expected_measurements=pinned,
            max_collateral_age=args.max_collateral_age,
            accept_offline_tcb=args.accept_offline_tcb,
            trust_live_intermediates=args.trust_live_intermediates,
            health_report=health_report,
        )
    except Exception as err:  # noqa: BLE001  fail closed on a malformed field no check names yet
        detail = f"verification stopped on {type(err).__name__}: {str(err)[:200]}"
        print(f"error: {detail}", file=sys.stderr)
        _show(malformed_report(detail), args.json)
        return 2
    _show(report, args.json)
    return _exit_code(report)


def _read_quote_bytes(path: str) -> bytes:
    """A quote file as written by configfs TSM (raw), or as people paste it (hex, base64)."""
    raw = Path(path).read_bytes()
    text = raw.strip()
    if text and all(c in b"0123456789abcdefABCDEF\r\n" for c in text):
        try:
            return bytes.fromhex(text.decode("ascii").replace("\n", "").replace("\r", ""))
        except ValueError:  # odd number of hex digits: not hex after all, let the parser say so
            return raw
    if text and all(c in b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\r\n" for c in text):
        import base64

        try:
            return base64.b64decode(text, validate=False)
        except Exception:  # noqa: BLE001
            return raw
    return raw


def cmd_quote(args: argparse.Namespace) -> int:
    try:
        pinned = _expected_measurements(args)
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
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
        expected_measurements=pinned,
        accept_offline_tcb=args.accept_offline_tcb,
    )
    _show(report, args.json)
    return _exit_code(report)


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
    clean, results = run_selftest(
        bundle, online=not args.offline, expected_challenge=args.challenge, max_collateral_age=args.max_collateral_age
    )
    print("clean bundle:", clean.verdict)
    all_ok = clean.ok
    for r in results:
        all_ok = all_ok and r.passed
        rejected = ", ".join(r.failed_checks) or "nothing"
        print(f"[{'PASS' if r.passed else 'FAIL'}] {r.name.ljust(18)} {r.description}  ->  rejected on {rejected}")
    print("\nSELFTEST:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


def _whole(low: int, high: int) -> Callable[[str], int]:
    """An argparse type: a whole number between ``low`` and ``high``."""

    def parse(value: str) -> int:
        if not re.fullmatch(r"[0-9]{1,6}", value) or not low <= int(value) <= high:
            raise argparse.ArgumentTypeError(f"expected a whole number from {low} to {high}, got {value[:20]!r}")
        return int(value)

    return parse


def _model_name(value: str) -> str:
    """--expect-gpu: a short printable name, as in a listing."""
    if not 1 <= len(value) <= 80 or not value.isprintable():
        raise argparse.ArgumentTypeError("expected a GPU model name of 1 to 80 printable characters")
    return value


def _print_health(doc: dict[str, Any], written: str | None) -> None:
    checks = doc["checks"]
    width = max(len(c["name"]) for c in checks) if checks else 20
    for check in checks:
        print(f"[{check['result']}] {check['name'].ljust(width)}  {check['detail']}")
    data = {c["name"]: c["data"] for c in checks}
    rows: dict[int, list[str]] = {}
    for gpu in data.get("inventory.gpus", {}).get("gpus", []):
        rows[int(gpu["index"])] = [f"{gpu.get('name', '?')}  {gpu.get('memory_total', '?')} MiB"]
    for heat in data.get("sensors.thermal", {}).get("gpus", []):
        rows[heat["gpu"]].append(f"max {heat['temperature_max_c']} C  max {heat['power_max_mw'] // 1000} W")
    for clock in data.get("sensors.clocks", {}).get("gpus", []):
        rows[clock["gpu"]].append(f"SM {clock['sm_clock_median_mhz']} MHz")
    for load in data.get("compute.matmul", {}).get("gpus", []):
        rows[load["gpu"]].append(f"{load.get('gflops', 0) // 1000} TFLOPS  {load.get('mismatch', 0)} changed")
    for memory in data.get("vram.pattern", {}).get("gpus", []):
        rows[memory["gpu"]].append(f"VRAM {memory['coverage_percent']}% tested  {memory.get('wrong_bytes', 0)} wrong bytes")
    if rows:
        print()
    for index, parts in sorted(rows.items()):
        print(f"GPU {index}: {'  '.join(parts)}")
    if written:
        print(f"\nREPORT: {written}")
    print("\nRESULT:", doc["verdict"])


def cmd_health(args: argparse.Namespace) -> int:
    from .health.report import canonical_report, exit_code, report_sha256
    from .health.run import Options, run_health

    options = Options(
        quick=args.quick,
        burn_seconds=args.burn_seconds,
        expect_gpu=args.expect_gpu,
        expect_gpus=args.expect_gpus,
        expect_vram_gb=args.expect_vram,
    )
    doc = run_health(options, progress=lambda text: print(f". {text}", file=sys.stderr, flush=True)).to_json()
    written = None
    if args.output:
        try:
            Path(args.output).write_bytes(canonical_report(doc))
        except OSError as err:
            print(f"error: cannot write {args.output}: {type(err).__name__}", file=sys.stderr)
            return 2
        written = f"{args.output}  sha256 {report_sha256(doc)}"
    if args.json:
        print(json.dumps(doc, indent=2, sort_keys=True))
        if written:
            print(f"report written to {written}", file=sys.stderr)
    else:
        _print_health(doc, written)
    return exit_code(doc["result"])


def _add_measurement_args(parser: argparse.ArgumentParser) -> None:
    for name, text in MEASUREMENT_HELP.items():
        parser.add_argument(f"--expect-{name}", metavar="HEX", help=text)


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
    m.add_argument("--health", metavar="FILE", help="health report (`health -o`) to commit to: the attestation covers it")
    m.add_argument("-o", "--output", default="manifest.json")
    m.set_defaults(func=cmd_manifest)

    a = sub.add_parser("attest", help="produce the evidence bundle; run as root inside the TDX VM")
    a.add_argument("--manifest", required=True)
    a.add_argument("-o", "--output", default="bundle.json")
    a.add_argument("--mode", choices=["auto", "single-gpu", "multi-gpu-ppcie", "multi-gpu-nvle"], default="auto")
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
    v.add_argument("--max-collateral-age", type=_days, default=DEFAULT_MAX_COLLATERAL_AGE, metavar="DAYS", help=MAX_AGE_HELP)
    v.add_argument("--accept-offline-tcb", action="store_true", help=ACCEPT_OFFLINE_HELP)
    v.add_argument("--trust-live-intermediates", action="store_true", help=TRUST_LIVE_HELP)
    v.add_argument("--hwmodel", action="append", help="allowed GPU model claim, e.g. GH100; repeatable")
    v.add_argument("--gpus", type=int, help="exact number of attested GPUs expected")
    v.add_argument("--allow-debug", action="store_true", help=ALLOW_DEBUG_HELP)
    _add_measurement_args(v)
    v.add_argument("--health", metavar="FILE", help=HEALTH_HELP)
    v.add_argument("--json", action="store_true")
    v.set_defaults(func=cmd_verify)

    q = sub.add_parser("quote", help="verify a bare Intel TDX quote from any provider (no manifest, no NVIDIA)")
    q.add_argument("quote", help="quote file: raw bytes as written by /sys/kernel/config/tsm/report, or hex, or base64")
    q.add_argument("--report-data", help="expected 64-byte report_data as hex, e.g. the SHA-512 of the nonce you issued")
    q.add_argument("--accept-out-of-date", action="store_true", help="report a non-UpToDate TCB as a warning, not a failure")
    q.add_argument("--allow-debug", action="store_true", help=ALLOW_DEBUG_HELP)
    q.add_argument("--offline", action="store_true", help="structure and signatures only, no Intel PCS call (exit code 3)")
    q.add_argument("--accept-offline-tcb", action="store_true", help=ACCEPT_OFFLINE_HELP)
    _add_measurement_args(q)
    q.add_argument("--json", action="store_true")
    q.set_defaults(func=cmd_quote)

    i = sub.add_parser("inspect", help="print a bundle without the bulky fields")
    i.add_argument("bundle")
    i.set_defaults(func=cmd_inspect)

    s = sub.add_parser("selftest", help="verify a bundle, then check that six mutations are rejected")
    s.add_argument("bundle")
    s.add_argument("--challenge")
    s.add_argument("--offline", action="store_true")
    s.add_argument("--max-collateral-age", type=_days, default=DEFAULT_MAX_COLLATERAL_AGE, metavar="DAYS", help=MAX_AGE_HELP)
    s.set_defaults(func=cmd_selftest)

    h = sub.add_parser(
        "health",
        help="check the NVIDIA GPUs of this machine: inventory, errors, sensors, compute and VRAM under load",
        description=(
            "Hardware health check of the NVIDIA GPUs of this machine. Local and read-only: no root, no network, "
            "nothing is sent anywhere, nothing is installed. Exit code 0 pass, 1 a check failed, 2 unusable "
            "arguments, 3 a warning or a part that was not tested (the RESULT line names it)."
        ),
    )
    h.add_argument("--quick", action="store_true", help="60 seconds of load instead of 300")
    h.add_argument("--burn-seconds", type=_whole(10, 3600), metavar="N", help="length of the load, 10 to 3600 seconds")
    h.add_argument("--expect-gpu", type=_model_name, metavar="NAME", help='GPU model of the listing, e.g. "RTX 4090"')
    h.add_argument("--expect-gpus", type=_whole(1, 1024), metavar="N", help="GPU count of the listing: fewer is a failure")
    h.add_argument("--expect-vram", type=_whole(1, 4096), metavar="GB", help="VRAM per GPU of the listing, in GB")
    h.add_argument("--json", action="store_true", help="print the report as JSON instead of one line per check")
    h.add_argument("-o", "--output", metavar="FILE", help="also write the report, in the form `manifest --health` reads")
    h.set_defaults(func=cmd_health)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

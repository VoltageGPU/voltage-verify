# Verify a public bundle in five minutes

A walkthrough on a real bundle, with no account and no GPU rental: download it, verify it
offline, then make it fail on purpose. Every output below was produced on 19 October 2026 with
voltage-verify 0.2.1.

The bundle was captured on 9 October 2026 inside an Intel TDX guest with one NVIDIA RTX PRO 6000
Blackwell GPU (VoltageGPU SKU `rtx6000b-small`, rented through the public API by the weekly proof
run). Its folder also holds what the script printed inside the VM, `nvidia-smi` output and kernel
details: [README.txt](https://voltagegpu.com/blog/two-proofs/evidence/rtx6000b-2026-10-09/README.txt).

## 1. Download and check the checksums

```bash
mkdir bundle && cd bundle
curl -sO https://voltagegpu.com/blog/two-proofs/evidence/rtx6000b-2026-10-09/bundle.json
curl -sO https://voltagegpu.com/blog/two-proofs/evidence/rtx6000b-2026-10-09/manifest.json
curl -sO https://voltagegpu.com/blog/two-proofs/evidence/rtx6000b-2026-10-09/SHA256SUMS
sha256sum -c SHA256SUMS --ignore-missing
# manifest.json: OK
# bundle.json: OK
```

## 2. Install

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install "voltage-verify>=0.2.1"
```

## 3. Verify, offline

```bash
voltage-verify verify bundle.json --offline \
  --challenge 627ba9c51d4719038e917c7431675b183ac576c819d712178dc8a100f4409a45 \
  --hwmodel GB20X --gpus 1
```

```text
[PASS] manifest.format       manifest is well formed
[PASS] manifest.commitments  bundle commitments match the manifest
[PASS] manifest.challenge    challenge is the one this verifier issued
[PASS] tdx.structure         TDX quote v4, TEE 0x81, 4940 bytes
[PASS] tdx.report_data       quote report_data equals SHA-512 of the manifest
[PASS] tdx.signatures        attestation key, QE binding, QE report and PCK chain verify up to the pinned Intel SGX Root CA
[PASS] tdx.collateral        TCB info and QE identity signed by Intel (bundle (fetched 2026-10-09T21:54:01+00:00))
[PASS] tdx.tcb               platform TCB UpToDate, QE UpToDate, TDX_03 UpToDate
[PASS] tdx.revocation        PCK certificate not revoked (57 entries in CRL)
[PASS] tdx.revocation        PCK CA certificate not revoked (0 entries in CRL)
[PASS] nvidia.signatures     2 NVIDIA-signed tokens verify, keys from the bundle chained to the pinned NRAS intermediate (offline)
[PASS] nvidia.nonce          NRAS eat_nonce equals SHA-256 of the manifest
[PASS] nvidia.policy         overall result true, every GPU: measurements ok, secure boot on, debug off, nonce matched, report signature and chain validated
[PASS] nvidia.freshness      token issued 0s from bundle creation

TDX: MRTD eea8b6a814569a52bd1e12f6b869bb2d...  TCB UpToDate  QE UpToDate
GPU: GPU-0 GB20X driver 595.71.05 vbios 98.02.9E.00.01 measurements success

RESULT: VERIFIED
```

`--offline` reads the Intel collateral embedded in the bundle (itself checked against the pinned
Intel root) instead of fetching it from Intel PCS.

## 4. Make it fail

A replayed bundle answers a challenge you did not issue:

```bash
voltage-verify verify bundle.json --offline \
  --challenge 0000000000000000000000000000000000000000000000000000000000000000 \
  --hwmodel GB20X --gpus 1
# [FAIL] manifest.challenge    challenge differs from the one this verifier issued (replay or foreign bundle)
# RESULT: NOT VERIFIED (manifest.challenge)
```

One changed character inside the signed part of the quote:

```bash
python3 - <<'PY'
import json
b = json.load(open("bundle.json"))
q = b["tdx"]["quote_b64"]
b["tdx"]["quote_b64"] = q[:300] + ("A" if q[300] != "A" else "B") + q[301:]
json.dump(b, open("tampered.json", "w"))
PY

voltage-verify verify tampered.json --offline \
  --challenge 627ba9c51d4719038e917c7431675b183ac576c819d712178dc8a100f4409a45 \
  --hwmodel GB20X --gpus 1
# [FAIL] tdx.signatures        quote signature does not verify with the attestation key
# RESULT: NOT VERIFIED (tdx.signatures)
```

## What this shows, and what it does not

It shows that this manifest was bound by a genuine Intel TDX guest on a platform Intel rates up to
date, with an NVIDIA GPU in confidential-computing mode whose measurements NVIDIA accepted, and
that the bundle was not altered afterwards.

It does not show which software ran in the guest (compare the printed MRTD and RTMR values with a
reference you trust, see [WHAT_IT_PROVES.md](WHAT_IT_PROVES.md)), nor freshness relative to you:
the challenge of this bundle was drawn by the publisher's script. For your own guarantee, draw the
challenge yourself on your own VM ([ATTEST_ON_VM.md](ATTEST_ON_VM.md)).

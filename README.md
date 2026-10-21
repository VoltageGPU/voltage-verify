# voltage-verify

[![PyPI](https://img.shields.io/pypi/v/voltage-verify.svg)](https://pypi.org/project/voltage-verify/) [![CI](https://github.com/VoltageGPU/voltage-verify/actions/workflows/ci.yml/badge.svg)](https://github.com/VoltageGPU/voltage-verify/actions) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Evidence](https://img.shields.io/badge/evidence-dated%20bundles-4ade80.svg)](https://github.com/VoltageGPU/confidential-gpu-attestation-evidence)

Bind an Intel TDX quote and an NVIDIA GPU attestation to the workload you meant to run, then
verify the bundle on your own machine, without trusting the cloud provider.

`voltage-verify` is a small command line tool with three roles:

1. **On your machine**, it writes a *manifest*: the container image digest, the digests of the
   model or files you care about, a free-text statement, and a fresh random challenge. Two
   hashes of that manifest become the *commitments*.
2. **Inside the confidential VM**, it asks the hardware to sign those commitments: the Intel
   TDX quote carries `SHA-512(manifest)` as its 64-byte `report_data`, and NVIDIA's Remote
   Attestation Service (NRAS) signs GPU claims over the nonce `SHA-256(manifest)`. It writes
   one JSON *bundle* with the quote, the NVIDIA tokens and the public collateral needed to
   check them later.
3. **Back on your machine**, it verifies the bundle: Intel's signature chain to the pinned
   Intel SGX Root CA, that the Trust Domain does not run in debug mode, the platform's TCB
   status against Intel's published TCB info, the Quoting Enclave identity, revocation lists,
   NVIDIA's ES384 signatures against the NRAS JWKS carried by the bundle (each key trusted
   only through its x5c chain to the pinned NVIDIA NRAS intermediate, listed once, so a bundle
   cannot bring its own signing key), the per-GPU claims (measurements ok, secure boot on,
   debug off), and above all that both proofs carry *your* commitments. Then it runs nine
   negative tests to show that any change to the manifest, the challenge, the quote or a GPU
   claim is rejected, and so are tokens re-signed with a key of the attacker's choosing and a
   JWKS that lists the signing key twice or drops its certificate chain.

It was written by [VoltageGPU](https://voltagegpu.com) so that tenants of its Confidential VMs
can verify more than "this is a TDX VM with a confidential GPU": they can verify that the two
hardware proofs were produced *for the workload described in a manifest they wrote*, on a
challenge they chose. The trust chain ends at Intel and NVIDIA, never at VoltageGPU.

## What it proves, and what it does not

Verified means:

* the TDX quote was produced inside a genuine Intel TDX Trust Domain that does not run in
  debug mode (the host can read and modify the memory of a debug TD), on a platform whose
  provisioning certificate chains to Intel's root, at an acceptable TCB level, by a genuine
  Quoting Enclave, and it embeds `SHA-512` of your manifest;
* NVIDIA's service signed, over `SHA-256` of your manifest, that the GPU(s) in that VM passed
  attestation with secure boot on and debugging off;
* both proofs were made after you issued the challenge, so they cannot be replayed from an
  earlier session or copied from another tenant.

Verified does **not** mean:

* that the GPU *executed* the image or model named in the manifest. Binding a manifest to a
  quote proves the hardware signed your description of the workload; proving that this exact
  code ran needs a measured launcher (for instance the Confidential Containers project with
  a key broker), which is outside this tool;
* anything about the VM's own software stack beyond what the TDX measurements (`MRTD`,
  `RTMR0..3`) say. The tool prints them; comparing them against a reference image is your
  policy, not the tool's;
* anything about a VM you did not attest yourself.

`docs/WHAT_IT_PROVES.md` goes through every check and its limit.

## Install

```
# verifier machine: Python 3.10+, cryptography, PyJWT
pip install voltage-verify
# inside the VM: the NVIDIA SDK first (it pins cryptography and PyJWT), then the tool
pip install nv-attestation-sdk nvidia-ml-py
pip install --no-deps voltage-verify
```

Source and issues: https://github.com/VoltageGPU/voltage-verify. The same wheel and source
archive are mirrored with SHA-256 sums at https://voltagegpu.com/blog/two-proofs/voltage-verify/
for installs that must not touch PyPI (`pip install <wheel URL>`).

## Reference run

`examples/hello-workload/` holds a real bundle captured on 12 September 2026 on a VoltageGPU
`h100-xlarge` Confidential VM (8x H100, NVIDIA Protected PCIe mode), with its manifest and the
unedited run log. Replay the verification on your own machine:

```
voltage-verify verify examples/hello-workload/bundle-8xh100-2026-09-12.json \
    --challenge ef7f54c160627ee536a02b5e73896ac4b1ac2cdefdb7cfaaf2366a7d33cc8731 --hwmodel GH100 --gpus 8
voltage-verify selftest examples/hello-workload/bundle-8xh100-2026-09-12.json
```

Expected: every check `PASS`, platform TCB `UpToDate`, Quoting Enclave `UpToDate`, eight
`GH100` devices with `measres success`, then six mutations rejected. Read those tokens for what
they are: in Protected PCIe mode `nvidia-smi` reports `CC State: OFF` next to `Multi-GPU Mode:
Protected PCIe` (the normal reading for that mode, kept in the bundle's `environment` section),
NVIDIA attested each of the eight GPUs individually, the NVSwitch fabric is not attested, and
this is not the single-GPU `CC State: ON` mode of an H200 VM. `docs/WHAT_IT_PROVES.md` spells
out the difference.

## Use

On your machine, describe the workload and generate a challenge:

```
voltage-verify manifest --image ghcr.io/you/app:1.4.2 --artifact model.safetensors \
    --statement "inference run for customer X" -o manifest.json
```

Copy `manifest.json` into the VM (scp), then, as root inside the VM:

```
sudo -E python -m voltage_verify attest --manifest manifest.json -o bundle.json
```

Copy `bundle.json` back, then:

```
voltage-verify verify bundle.json --challenge <the challenge printed by `manifest`>
voltage-verify selftest bundle.json --challenge <same>
voltage-verify verify bundle.json --offline      # same checks, from the collateral in the bundle
```

Exit code 0 means verified, 1 means a required check failed, 2 means the bundle or arguments
are unusable. `--json` prints the machine-readable report.

## Works with any Intel TDX host, any provider

Nothing here is VoltageGPU-specific, and the tool does not call VoltageGPU's API to attest or
to verify. `manifest` and `verify` run entirely on your own machine. `attest` reads the kernel's
own confidential-computing interface, `/sys/kernel/config/tsm/report` (configfs TSM, upstream
since Linux 6.7, present on any distribution with a recent kernel), and calls NVIDIA's own SDK,
`nv-attestation-sdk`, to talk to NVIDIA's Remote Attestation Service. Neither of those is a
VoltageGPU endpoint. So `attest` runs the same way inside a TDX guest on Azure, on GCP, on a bare
metal TDX host with a passthrough NVIDIA GPU, or on VoltageGPU: the interface is the Linux kernel
and NVIDIA's own service, not us. `verify` checks the bundle against Intel's and NVIDIA's public
collateral, also fetched from Intel and NVIDIA, never from VoltageGPU.

What this means in practice: if your workload runs on a TDX VM with an NVIDIA H100, H200, or
Blackwell GPU in confidential compute mode, anywhere, `voltage-verify attest` inside that VM and
`voltage-verify verify` on your laptop give you the same tenant-side proof this tool was built to
give VoltageGPU's own customers. We have not run this against every cloud ourselves (the
[reference run](#reference-run) above is on our own fleet, because that is the hardware we have),
so if you verify a bundle produced elsewhere, an issue or a pull request with what you saw, good
or bad, is exactly the kind of report `SECURITY.md` and this repository want.

## Verify a quote from anywhere

`quote` takes a bare Intel TDX quote, from any provider, any tool, and runs the Intel half of
the checks on it: structure, the TD's debug bit, signature chain up to the pinned Intel root,
then TCB status, QE identity and revocation from Intel PCS. No manifest, no NVIDIA token, no
VoltageGPU anywhere in the path. The file can be the raw bytes written by
`/sys/kernel/config/tsm/report`, or hex, or base64.

```
voltage-verify quote quote.bin
voltage-verify quote quote.bin --report-data <64 bytes hex>   # bind it to the nonce you issued
voltage-verify quote quote.bin --offline                       # structure and signatures only
voltage-verify quote quote.bin --accept-out-of-date --json
voltage-verify quote quote.bin --allow-debug                   # a debug TD as a warning, development only
```

Tested on Google's production Sapphire Rapids quote from
[go-tdx-guest](https://github.com/google/go-tdx-guest/blob/main/testing/testdata/tdx_prod_quote_SPR_E4.dat)
(vendored in `tests/fixtures/third-party/`, Apache 2.0). Output with `--accept-out-of-date` on
21 October 2026 (0.2.2):

```
[PASS] tdx.structure   TDX quote v4, TEE 0x81, 4935 bytes
[WARN] tdx.trailing    39 bytes after the signature data were ignored (not covered by any signature)
[PASS] tdx.debug       TD attributes 0000004000000000 (0x40000000), DEBUG (bit 0) off: not a debug TD
[PASS] tdx.signatures  attestation key, QE binding, QE report and PCK chain verify up to the pinned Intel SGX Root CA
[PASS] tdx.collateral  TCB info and QE identity signed by Intel (Intel PCS, fetched now)
[WARN] tdx.tcb         no TCB level in Intel's TCB info matches this platform (treat as OutOfDate)
```

Two things to read in that output. The trailing bytes are a marker Google appends on purpose
in its test vector; they sit outside the signed region and are reported, not silently dropped.
The TCB warning is real: that 2023 quote comes from a platform whose firmware level Intel no
longer lists, so a verifier must call it OutOfDate. Without `--accept-out-of-date` the exit
code is 1, which is the right answer for a quote you would rely on today.

`tdx.debug` reads bit 0 of the TD attributes in the TD report. A TD in debug mode gives no
confidentiality: the host can read and modify its memory and CPU state, yet Intel's Quoting
Enclave signs its quote like any other, so the signature chain alone does not catch it. `quote`
and `verify` fail on that bit; `--allow-debug` (both commands) turns the failure into a visible
warning, for development VMs only. Versions up to 0.2.1 reported the TD attributes in the
`--json` facts without checking them.

## Bundle format

Documented in `docs/BUNDLE_FORMAT.md`. In short: the manifest verbatim, both commitments, the
raw TDX quote (base64), the NRAS token array as returned by the NVIDIA SDK, NRAS's JWKS and
Intel's TCB info, QE identity and CRLs as fetched at attestation time, plus a few facts about
the VM (kernel, driver, `nvidia-smi conf-compute -q`). The verifier trusts none of it: it
recomputes the commitments from the manifest and checks every signature.

## Development

```
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check src tests
```

The tests run offline against real evidence: a TDX quote captured on a VoltageGPU H200 VM on
4 September 2026, NRAS tokens from 4 and 10 September 2026 (single H200, 8x H100 node in
NVIDIA Protected PCIe mode), and a snapshot of Intel's collateral and NVIDIA's JWKS.

## See also

- [docs/VERIFY_A_PUBLIC_BUNDLE.md](docs/VERIFY_A_PUBLIC_BUNDLE.md): a five-minute walkthrough on a
  public bundle, no account needed, including two ways to make the verification fail.
- [confidential-gpu-attestation-evidence](https://github.com/VoltageGPU/confidential-gpu-attestation-evidence):
  every dated bundle this tool has verified on VoltageGPU's fleet since 4 September 2026,
  including the runs that fail (NVSwitch fabric from a TDX guest), re-verifiable with
  `voltage-verify verify`.
- [tdx-guest-probe](https://github.com/VoltageGPU/tdx-guest-probe): one script that lists what a
  TDX guest can observe and the host setting it cannot, useful before trusting any quote.

## Security

See `SECURITY.md`. In one line: report anything that would make `verify` say yes when it
should say no to contact@voltagegpu.com, and expect an answer within two working days.

## License

MIT. Copyright 2026 VOLTAGE EI (VoltageGPU).

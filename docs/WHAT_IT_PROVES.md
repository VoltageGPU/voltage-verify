# What each check proves, and what it does not

`voltage-verify verify` prints one line per check. This page is the honest reading of each.

## Manifest

| Check | Passes when | Proves | Does not prove |
|---|---|---|---|
| `manifest.format` | the manifest has the expected shape | the bundle is parseable | anything |
| `manifest.commitments` | SHA-256 and SHA-512 of the canonical manifest equal the stored values | bookkeeping is consistent | anything: an attacker can recompute them |
| `manifest.challenge` | the challenge equals the one you passed with `--challenge` | the bundle was made for this session | anything without the TDX and NVIDIA checks below |
| `manifest.image` | the image digest equals `--image-digest` | the manifest names the image you expect | that the image ran |

## Intel TDX

| Check | Passes when | Proves | Does not prove |
|---|---|---|---|
| `tdx.structure` | version 4 quote, TEE type TDX, ECDSA-P256 key, QE report and PCK chain present | it is a TDX quote | anything about its origin |
| `tdx.debug` | bit 0 (DEBUG) of the TD attributes is 0; with `--allow-debug` a set bit is a warning, not a failure | the TD does not run in debug mode, in which the host can read and modify its memory and CPU state | anything on its own: the TD attributes are only Intel's word once `tdx.signatures` passes |
| `tdx.report_data` | the quote's 64-byte `report_data` equals `SHA-512(manifest)` | the quote was requested with your manifest hash, so after your challenge | who requested it |
| `tdx.signatures` | attestation key signs header+report; QE report binds the key; PCK certificate signs the QE report; PCK chain verifies to the pinned Intel SGX Root CA and is within validity | the quote was produced by a genuine Intel TDX Quoting Enclave on Intel-provisioned hardware | that the platform's firmware is current |
| `tdx.collateral` | TCB info and QE identity carry Intel's signature and chain to the root | the collateral is Intel's | freshness (see `tdx.collateral_fresh`) |
| `tdx.tcb` | the platform's SVNs (from the PCK certificate and the quote's `tee_tcb_svn`) reach a level Intel rates `UpToDate` or `SWHardeningNeeded`; the TDX module identity matches; the Quoting Enclave identity matches and is `UpToDate` | the platform is patched to a level Intel accepts | that no unknown vulnerability exists |
| `tdx.revocation` | the PCK certificate and its CA are absent from Intel's CRLs | Intel has not revoked this platform's provisioning identity | anything if CRLs are missing (then the line says NOT checked) |

The quote's `MRTD` and `RTMR0..3` are printed as facts. They measure the VM's initial state
and boot chain. This tool does not compare them to a reference: whether a given `MRTD` is the
image you expect is a policy decision that belongs to you.

## NVIDIA

| Check | Passes when | Proves | Does not prove |
|---|---|---|---|
| `nvidia.jwks` | the bundle carries a JWKS (reported only when it does not) | keys can be looked up | anything |
| `nvidia.signatures` | the verifier token and every device token verify (ES384/ES256 on the matching curve) with a key from the bundle's JWKS, listed once, whose x5c leaf names NVIDIA Corporation, holds that key, is issued by a pinned NRAS intermediate, GPU Intermediate 004 or 005 (or one NVIDIA's live JWKS presents) and was valid at the token's `iat`; issuer is NRAS | NVIDIA issued these claims | when, until `nvidia.freshness` |
| `nvidia.nonce` | `eat_nonce` equals `SHA-256(manifest)` | the claims were issued for your manifest hash, after your challenge | which VM: the TDX side does that |
| `nvidia.policy` | overall result true; every GPU: `measres` success, `secboot` true, `dbgstat` disabled, nonce matched, report signature verified, certificate chain validated; optional `--hwmodel` and `--gpus` | NVIDIA's service accepted the GPU's measurements and configuration | that the GPU executed your workload |
| `nvidia.freshness` | the verifier token's `iat` is within 15 minutes of the bundle's `created_at` | the token belongs to this attestation session | anything if the VM clock is wrong (the check then fails, which is the safe direction) |

On a multi-GPU node in NVIDIA Protected PCIe mode (`nvidia.mode = multi-gpu-ppcie`), each GPU
is attested individually through the same flow: one NRAS token per GPU, on the same nonce.
`nvidia-smi conf-compute -q` reports `CC State: OFF` next to `Multi-GPU Mode: Protected PCIe`
on such a node; that is the normal reading for that mode, not a failure, and the bundle keeps
it verbatim in `environment.conf_compute`. NVSwitch attestation is not part of the bundle. This
is a different mode from the single-GPU `CC State: ON` of an H200 VM, and the tool never claims
one when it saw the other: the policy checks the per-GPU claims listed above and nothing else.
The reference bundle in `examples/hello-workload/` is of the Protected PCIe kind (8x H100).

## The gap this tool does not close

Both proofs are bound to a description of the workload. Neither proof says that the
described image or model was what the GPU computed on. Closing that gap requires a measured
launch: a component inside the VM that measures the container image and the model before
starting them and extends a TDX runtime measurement (`RTMR`) with the result, so that the
quote itself carries the measurement. The Confidential Containers project (Kata + Trustee)
and similar stacks do this. `voltage-verify` stops one step earlier on purpose, and says so.

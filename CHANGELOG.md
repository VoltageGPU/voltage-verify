# Changelog

## 0.2.3 (2026-10-22)

NVIDIA key rotation. Upgrade with `pip install -U voltage-verify` before verifying a bundle
offline. Nothing that 0.2.2 reported as verified needs to be verified again.

* What changed at NVIDIA: since October 2026, NRAS also signs tokens with keys issued by a
  second intermediate, `CN=NVIDIA Attestation Service GPU Intermediate 005` (valid 2026-10-08
  to 2030-10-08). On 22 October 2026, NRAS's JWKS presented 77 of its 213 keys under 005.
  Both 004 and 005 verify under NVIDIA's root `NVIDIA Attestation Service CA 001`, as served
  by NVIDIA's PKI at
  `https://prod.vault.nvidia.com/v1/dgxc-attestation/pki/nras-prod/ca_chain` (checked on
  22 October 2026). The tool itself does not fetch or check that root.
* What 0.2.2 did with such a bundle: offline, `nvidia.signatures` failed, since only 004 was
  pinned and there is no live JWKS to vouch for another intermediate. Online it passed,
  through the intermediates NVIDIA's live JWKS presents. Failing closed, it never accepted a
  key it should not have.
* What 0.2.3 does: pins 005 next to 004 (SPKI SHA-256 `ba14d003...`, read from the live NRAS
  JWKS over HTTPS and cross-checked against confidential-dot-ai/attestation-rs). Everything
  else in the chain check is unchanged: a leaf issued by the intermediate, named NVIDIA
  Corporation, holding exactly the JWK's key, valid when the token was issued, listed once.
* The `nvidia.signatures` line now says "a pinned NRAS intermediate".

## 0.2.2 (2026-10-21)

Security release for the Intel TDX side of the verifier. Upgrade with
`pip install -U voltage-verify`. The NVIDIA checks were not affected: the GPU policy has always
required `dbgstat` disabled.

* What 0.2.1 and earlier accepted: a quote from a Trust Domain running in debug mode. Bit 0
  (DEBUG) of the TD attributes was listed in the `--json` facts (`td_attributes`) but never
  checked. A debug TD gives no confidentiality: the host can read and modify its memory and CPU
  state, and Intel's Quoting Enclave still signs its quote, so `quote` and `verify` reported
  it as verified. Found in an internal review on 21 October 2026.
* What 0.2.2 refuses: a new check, `tdx.debug`, in `quote`, `verify` and `selftest`, online
  and offline, fails when bit 0 of the TD attributes is set, and prints the TD attributes
  either way. It reads the TD report itself, so it is reported even when the signatures fail.
* `--allow-debug` (`quote` and `verify`) turns that failure into a warning, for development
  VMs only, the way `--accept-out-of-date` does for the TCB status.
* To re-read a result from 0.2.1 or earlier without running it again: if the first byte of
  `td_attributes` in the `--json` facts is odd (`01`, `03`, ...), the TD was in debug mode.
  The reference bundle in `examples/hello-workload/` and the test quotes read
  `0000001000000000` or `0000004000000000`: debug off.
* Library: `tdx.TdReport.debug`, `tdx.TD_ATTRIBUTES_OFFSET` (168) and the `allow_debug`
  argument of `verify_bundle` and `verify_quote`.

## 0.2.1 (2026-10-09)

Security release for the NVIDIA side of the verifier. Upgrade with
`pip install -U voltage-verify`. A bundle that passed with 0.2.0 or earlier proves nothing
about NVIDIA until it is verified again with 0.2.1. The Intel TDX checks were not affected.

* What 0.2.0 and earlier accepted:
  * With `--offline`, and online whenever NVIDIA's live JWKS could not be fetched, the JWK
    coordinates shipped in the bundle were trusted as they were; x5c was only read to display
    certificate subjects. A bundle could carry its own signing key, and NRAS tokens re-signed
    with it verified. This case was reported privately on 23 September 2026 by Satyawan Singh,
    who confirmed on 23 October 2026 that 0.2.3 refuses his original reproduction (bundle key
    without an x5c chain) and still verifies a genuine bundle. Thank you.
  * A JWKS listing the signing key id twice: the first entry was used.
  * Online, signing keys came from NVIDIA's live JWKS, not from the bundle: a bundle whose keys
    were missing or stripped of their chain still verified while NVIDIA published the key.
    The other way round, honest archived bundles failed online once NVIDIA had rotated the key
    (about two days later).
  * A token without `kid` was checked against the first key of the JWKS.
* What 0.2.1 refuses. These are the rules of the website verifier (voltagegpu.com,
  `lib/verified/nras.ts`):
  * Signing keys are taken from the bundle's JWKS only. A key that is only in NVIDIA's live
    JWKS is not used. Online, the live JWKS only contributes the intermediates it presents,
    so that a rotation of NVIDIA's intermediate does not break verification.
  * Each key needs an x5c of at least two certificates. The leaf must be issued by the pinned
    NRAS intermediate (SPKI SHA-256 `fd32837f...e896`, "NVIDIA Attestation Service GPU
    Intermediate 004") or by one that NVIDIA's live JWKS presents. It must name NVIDIA
    Corporation and hold exactly the key the JWK declares. Both certificates must have been
    valid at the token's `iat`.
  * Also refused: a key id listed more than once, a token without a usable `kid`, any
    algorithm other than ES384/ES256 (`none`, HMAC, RSA, any other name), and an algorithm
    that does not match the key's curve.
  * Malformed tokens, JWKS entries or x5c chains now fail the `nvidia.signatures` check
    instead of stopping the verifier with an exception. Tokens over 16 KiB are refused.
* Honest bundles are unaffected: the published evidence bundles still verify, offline and
  now also online.
* `attest` always embeds NRAS's JWKS, also with `--no-collateral`.
* `selftest` runs three more negative tests: `nras-own-key`, `jwks-duplicate-kid` and
  `jwks-no-chain`.

## 0.2.0 (2026-09-18)

* New command `voltage-verify quote <file>`: verifies a bare Intel TDX quote from any provider,
  raw, hex or base64, with no manifest and no NVIDIA token. Structure, signature chain up to
  the pinned Intel root, TCB status, QE identity and revocation from Intel PCS. Options
  `--report-data` (bind to the 64 bytes you expect), `--offline`, `--accept-out-of-date`,
  `--json`. Tested against Google's production Sapphire Rapids quote from go-tdx-guest, now
  vendored under `tests/fixtures/third-party/`.
* `tdx.parse_quote(raw, allow_trailing=True)` tolerates bytes after the signature data and
  reports how many were ignored (`Quote.trailing`); the default stays strict, so bundles
  produced by `attest` are checked exactly as before.
* A platform whose TCB components match no level in Intel's TCB info is reported as OutOfDate
  (Intel's own rule) and, with `--accept-out-of-date`, becomes a warning instead of a failure.

## 0.1.2 (2026-09-18)

* Docs only, no code change. Spells out what the Protected PCIe reference tokens in
  `examples/hello-workload/` actually cover (CC State OFF reading, one token per GPU, no NVSwitch
  fabric attestation). Adds a "Works with any Intel TDX host, any provider" section: `attest` and
  `verify` never call a VoltageGPU endpoint, they read the kernel's own configfs TSM interface and
  NVIDIA's own SDK and collateral, so the tool works the same on any TDX host with a confidential
  NVIDIA GPU, on any cloud.

## 0.1.1 (2026-09-12)

* Dependencies: the upper bound `cryptography<47` is removed. It existed only so that the
  `[attest]` extra could co-install with nv-attestation-sdk (which pins `cryptography==43.0.1`),
  but it also stopped verifier machines from getting current cryptography releases and their
  security fixes. The lower bound is unchanged; the test suite passes on 43.0.1 and 50.0.1.
  No code change.

## 0.1.0 (2026-09-12)

First release.

* `manifest`: image digest resolution through the OCI registry API (no pull), artifact
  hashing, free-form extras, verifier-side random challenge.
* `attest`: TDX quote through configfs TSM with `report_data = SHA-512(manifest)`, NVIDIA
  NRAS attestation with `nonce = SHA-256(manifest)`, single-GPU and multi-GPU Protected PCIe
  modes, Intel and NVIDIA collateral embedded for offline verification.
* `verify`: quote structure, attestation key signature, Quoting Enclave binding and signature,
  PCK chain to the pinned Intel SGX Root CA, TCB level and TDX module identity against Intel's
  signed TCB info, Quoting Enclave identity, CRLs, NRAS ES384 signatures, nonce, claims policy,
  token freshness, challenge and image expectations.
* `selftest`: six negative tests (image digest, artifact, manifest field, challenge replay,
  quote tampering, GPU claim tampering).
* Tests against real evidence captured on VoltageGPU Confidential VMs.

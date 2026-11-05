# Changelog

## 0.3.0 (2026-11-05)

Security release for both sides of the verifier, from an audit of 5 November 2026. Upgrade
with `pip install -U voltage-verify`. Scripts that run `verify --offline` or `quote --offline`
must handle the new exit code 3 (see Compatibility). The published bundles still verify online;
offline, see the age of the Intel collateral below.

What 0.2.3 and earlier accepted, and what is refused now:

* NVIDIA GPU tokens taken from another attestation. NRAS lists every GPU token in the verifier
  token (`submods`, the SHA-256 of the compact token) and each GPU token carries `eat_nonce`;
  neither was checked, so a GPU token from another NVIDIA attestation verified next to a fresh
  verifier token and could pad `--gpus`. The GPU tokens must now be exactly the ones the
  verifier token lists (`nvidia.signatures`) and carry the manifest's nonce (`nvidia.nonce`).
  `selftest` adds `nras-extra-gpu` (ten mutations).
* Offline, Intel collateral with a second copy after its signature. A TCB info or QE identity
  body `{"tcbInfo":<signed>,"signature":"...","tcbInfo":<forged>}` verified, and the forged copy
  was the one read: a bundle could carry any TCB status and any issue date. The body must now
  be exactly the signed document followed by its signature, and is read from the signed bytes.
  Found while working on the audit. Collateral fetched online from Intel was not exposed.
* Offline, stale Intel collateral, which was only a warning. `tdx.collateral_fresh` is now
  required offline: it fails when the oldest issue date Intel signed into the TCB info, the QE
  identity and the PCK CRL is more than 30 days old (the validity Intel gives these documents),
  when the root CA CRL is past its nextUpdate, or when the CRLs are missing.
  `--max-collateral-age DAYS` (`verify`, `selftest`) accepts older collateral explicitly. Online
  behaviour is unchanged.
* `quote --offline` ended on `RESULT: VERIFIED` and exit code 0 with no TCB status and no
  revocation check, and `verify --offline` judged the TCB on the bundle's collateral without
  saying so. The verdict now names the limit, `RESULT: VERIFIED (signatures only, TCB not
  checked offline)` or `RESULT: VERIFIED (TCB and revocation from the bundle's Intel
  collateral, not checked against Intel today)`, and the exit code is 3. `--accept-offline-tcb`
  (`verify`, `quote`) accepts it: plain `VERIFIED`, exit code 0.
* Online, every intermediate NVIDIA's live JWKS presented became a trust anchor. Only the pinned
  GPU Intermediate 004 and 005 are now, online and offline, and the live JWKS is not fetched;
  `verify --trust-live-intermediates` restores the previous behaviour after a rotation by
  NVIDIA. All published bundles chain to 004.
* A PCK chain whose issuers were not CAs. Every issuer must now carry BasicConstraints with
  `ca=True`.
* Malformed input stopped the verifier with a traceback, and an unreadable bundle printed no JSON
  with `--json`. An unusable bundle is now the failed check `bundle.malformed` (exit code 2,
  valid `--json` report); a malformed field fails the check that owns it.

New:

* `--expect-mrtd` and `--expect-rtmr0` to `--expect-rtmr3` (`verify`, `quote`) pin the TD's
  measurements, checked by `tdx.measurements`. Without them `VERIFIED` is unchanged, but a
  warning says `TD software not pinned: pass --expect-mrtd to check it`. `docs/WHAT_IT_PROVES.md`
  also says that NVIDIA's tokens name no TD.
* Without `--challenge`, `manifest.challenge` is a warning: `no challenge supplied: freshness and
  replay not checked`.
* `attest --mode multi-gpu-nvle`, for a Blackwell node in NVIDIA NVLink-encryption (NVLE) mode:
  each GPU is attested on its own, through the same standalone path as `single-gpu`.
* `nvidia.mode`: the NVIDIA attestation mode recorded in the bundle must be one the tool knows
  (`single-gpu`, `multi-gpu-ppcie`, `multi-gpu-nvle`); 0.2.3 did not check it. On a multi-GPU
  mode, `nvidia.scope` warns that the per-GPU tokens do not attest the fabric between the GPUs:
  the NVSwitch fabric in Protected PCIe mode, the encrypted NVLink fabric in NVLE mode.
* `tdx.measured_compose`: when the bundle's environment carries a `dstack` block, a warning says
  that the TD runs a composition measured by the provider (its compose hash lands in RTMR3),
  which the tool compares to no published reference: trusted, not proven.
* `contrib/dstack-b200/`: a collector for a confidential GPU container run by dstack, where the
  tenant has no configfs TSM. It asks the dstack guest agent for a quote over the manifest's
  SHA-512, runs NRAS for every GPU and writes the same `voltage-verify-bundle/1`, with the
  `dstack` block (app id, event log) in its environment. It refuses to write a bundle unless the
  quote's report_data is the manifest hash byte for byte. Not installed with the package.

Compatibility:

* Exit codes: 0 verified, 1 a required check failed, 2 unusable input, 3 verified with a limit
  named in the verdict (offline TCB). A script that treats anything but 0 as a failure fails
  offline runs until it passes `--accept-offline-tcb` or verifies online.
* `--json`: new keys `verdict` and `limits`; `ok`, `checks` and `facts` are unchanged. New check
  names `tdx.measurements`, `bundle.malformed` and `nvidia.mode`, and the warnings `nvidia.scope`
  (every multi-GPU bundle) and `tdx.measured_compose` (a bundle with a `dstack` block); offline,
  `tdx.collateral_fresh` now always appears; `manifest.challenge` may be a warning. A script that
  treats any warning as a failure fails on every run that does not pin the TD measurements.
* Library: `verify_bundle` takes `expected_measurements`, `max_collateral_age`,
  `accept_offline_tcb` and `trust_live_intermediates`, `verify_quote` takes
  `expected_measurements` and `accept_offline_tcb`; `verify_bundle(online=True)` no longer uses
  NVIDIA's live intermediates by default. New: `Report.limits`, `Report.verdict`,
  `NrasResult.nonce_mismatch`. `pcs.SignedDocument.payload` is rebuilt from the signed bytes.
* Dependencies: `cryptography>=43.0.1`, whose earlier wheels ship a vulnerable OpenSSL
  (GHSA-h4gh-qq45-vh27).

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

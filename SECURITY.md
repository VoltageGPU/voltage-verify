# Security policy

`voltage-verify` is a verifier. The bugs that matter are the ones that make it accept
evidence it should reject: a signature check that can be skipped, a commitment that can be
recomputed without the hardware noticing, a policy claim that is read but not enforced, a
collateral document accepted without its signature.

## Reporting

Write to contact@voltagegpu.com with the subject `voltage-verify security`. Include the bundle
or the minimal input that triggers the behaviour, the version (`voltage-verify --version`) and
what you expected. You will get a human answer within two working days, a fix or a documented
limitation within thirty, and credit in the changelog if you want it.

Please do not open a public issue for a bypass before it is fixed.

## Scope

In scope: everything under `src/voltage_verify`, the bundle format, the documented checks in
`docs/WHAT_IT_PROVES.md`.

Out of scope: the security of Intel TDX or NVIDIA confidential computing themselves, the
availability of Intel's PCS or NVIDIA's NRAS, and the `attest` command running on a machine
that is already compromised at the hypervisor level (that is what the hardware proofs detect).

## What the tool pins

* Intel SGX Root CA, by SHA-256 fingerprint, shipped in the package
  (`src/voltage_verify/data/intel_sgx_root_ca.pem`).
* NVIDIA NRAS issuer `https://nras.attestation.nvidia.com`.
* The NRAS intermediate "NVIDIA Attestation Service GPU Intermediate 004", by SHA-256 of its
  public key (`src/voltage_verify/nvidia.py`, `NRAS_TRUSTED_SPKI_SHA256`). Signing keys are
  taken from the JWKS embedded in the bundle at attestation time, and only through their x5c
  chain: the leaf must be issued by this intermediate, name NVIDIA Corporation, hold exactly
  the JWK's key and have been valid when the token was issued. A key id listed twice is
  refused. Online, NVIDIA's live JWKS (fetched over TLS) can add the intermediate it presents,
  to survive a rotation; it never supplies a signing key.
* Accepted JWT algorithms: ES384 and ES256 only, on the matching curve. `none`, HMAC and RSA
  are rejected.

## Supported versions

Only the latest release gets security fixes. 0.2.1 and earlier accept a quote from a TD in
debug mode, and 0.2.0 and earlier trust NVIDIA signing keys that the bundle itself supplies
(see `CHANGELOG.md`, 0.2.2 and 0.2.1): upgrade.

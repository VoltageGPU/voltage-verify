# Bundle format `voltage-verify-bundle/1`

A bundle is one JSON document. Keys are sorted when written; readers must not rely on order.

```json
{
  "format": "voltage-verify-bundle/1",
  "tool": {"name": "voltage-verify", "version": "0.1.0"},
  "created_at": "2026-09-12T01:02:03+00:00",
  "manifest": {
    "format": "voltage-verify-manifest/1",
    "created_at": "2026-09-12T00:58:11+00:00",
    "challenge": "<64 hex chars, chosen by the verifier>",
    "workload": {
      "image": {"reference": "ghcr.io/you/app:1.4.2", "digest": "sha256:<64 hex>"},
      "artifacts": [{"name": "model.safetensors", "sha256": "<64 hex>", "size": 123456}],
      "extra": {"any": "string"}
    },
    "statement": "free text or null"
  },
  "commitments": {"sha256": "<64 hex>", "sha512": "<128 hex>"},
  "tdx": {"quote_b64": "<base64 of the raw quote, padding removed>", "report_data_hex": "<128 hex>"},
  "nvidia": {
    "mode": "single-gpu | multi-gpu-ppcie",
    "nras": [["JWT", "<overall>"], {"REMOTE_GPU_CLAIMS": [["JWT", "<verifier>"], {"GPU-0": "<jwt>"}]}],
    "jwks": {"keys": [ ... ]}
  },
  "collateral": {
    "intel": {
      "tcb_info": {"kind": "tcb_info", "body": "<exact PCS response>", "issuer_chain_pem": "<PEM chain>"},
      "qe_identity": {"kind": "qe_identity", "body": "<exact PCS response>", "issuer_chain_pem": "<PEM chain>"},
      "pck_crl_pem": "<PEM CRL or null>",
      "root_crl_der_b64": "<base64 DER CRL or null>",
      "fetched_at": "2026-09-12T01:02:00+00:00"
    }
  },
  "environment": {"kernel": "...", "gpus": "...", "conf_compute": "..."}
}
```

## Commitments

`canonical = JSON(manifest)` with keys sorted by code point, separators `,` and `:` with no
spaces, UTF-8, non-ASCII characters kept as is, no floats anywhere. This matches RFC 8785 for
the value types a manifest contains.

* `commitments.sha256 = SHA-256(canonical)`, hex. This is the NVIDIA nonce: the SDK passes it
  to NRAS, which returns it in the verifier token as `eat_nonce`.
* `commitments.sha512 = SHA-512(canonical)`, hex. This is the TDX `report_data`: written to
  the configfs TSM `inblob`, returned verbatim at byte offset 568 of the quote.

A verifier recomputes both from `manifest` and ignores the stored values except to flag a
mismatch. Everything binding is inside the signed evidence.

## Why the collateral travels with the bundle

Intel's TCB info and QE identity are re-issued regularly and NVIDIA rotates its JWKS keys
within days. A bundle verified six months later would otherwise fail for reasons unrelated
to the evidence. The embedded copies are signed by Intel (ECDSA P-256 over the exact response
body, issuer chain to the pinned Intel SGX Root CA) and by NVIDIA (x5c chain in the JWKS),
so `verify --offline` still ends at a hardware vendor's key. `verify` without `--offline`
refetches Intel's collateral and reports which source it used. NVIDIA signing keys always come
from the embedded JWKS (since 0.2.1): online, NVIDIA's live JWKS is fetched only to learn the
intermediates it presents, never to supply a key.

## Compatibility rules

* A reader must reject any `format` it does not know.
* Fields may be added in later minor formats; unknown fields are ignored.
* `nras` is stored exactly as the NVIDIA SDK returns it, so any change in the SDK layout
  shows up as a verification failure rather than a silent reinterpretation.

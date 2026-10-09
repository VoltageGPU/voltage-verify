# Running `attest` inside a Confidential VM

`attest` needs three things the VM must expose: the TDX guest device and configfs TSM
(`/sys/kernel/config/tsm/report`), an NVIDIA GPU in confidential-computing mode with the
driver's attestation support, and outbound HTTPS to NVIDIA (NRAS, OCSP, RIM) and Intel (PCS).
On a VoltageGPU Confidential VM all three are there at boot.

## One-time setup in the VM

```
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip
python3 -m venv ~/vv
. ~/vv/bin/activate
pip install nv-attestation-sdk nvidia-ml-py
pip install --no-deps voltage-verify
```

Install the NVIDIA SDK first: it pins `cryptography==43.0.1` and `PyJWT 2.7`, both inside the
ranges this tool accepts, and pulls the NVIDIA local verifier and `nvidia-ml-py`. The first
install takes a minute. `pip install "voltage-verify[attest]"` in one step also works, but
letting pip resolve both at once is slower on the VM than the two commands above. Without
PyPI access, install the wheel mirrored at
https://voltagegpu.com/blog/two-proofs/voltage-verify/ with `--no-deps`.

## Multi-GPU nodes (NVIDIA Protected PCIe mode)

On an 8-GPU Hopper node the GPUs run in NVIDIA's multi-GPU *Protected PCIe* mode. `nvidia-smi
conf-compute -q` shows `CC State: OFF` next to `Multi-GPU Mode: Protected PCIe`; that is the
normal reading for that mode, not a failure. Two things differ from a single-GPU VM:

* the ready state has to be set once after boot: `sudo nvidia-smi conf-compute -srs 1`;
* the NVIDIA SDK refuses "standalone" attestation on such a node, so `attest` calls it with
  `ppcie_mode=False` (this SDK flag means "not standalone"). `--mode auto` picks this when
  more than one GPU is present.

## Produce the bundle

Copy the manifest your verifier produced, then run as root (TSM needs it):

```
sudo -E ~/vv/bin/python -m voltage_verify attest --manifest manifest.json -o bundle.json
```

What happens, in order:

1. the manifest is validated and both commitments are computed;
2. `report_data = SHA-512(manifest)` is written to a fresh entry under
   `/sys/kernel/config/tsm/report/`, the quote is read from `outblob` (retry only on
   `EINVAL`, which signals a concurrent request; any other error stops);
3. the NVIDIA SDK is called with `nonce = SHA-256(manifest)` against NRAS;
4. NRAS's JWKS (always: the verifier takes NVIDIA's signing keys from the bundle only) and
   Intel's TCB info, QE identity and CRLs (unless `--no-collateral`) are fetched and embedded;
5. `bundle.json` is written. Nothing secret is in it: quote, public tokens, public collateral,
   and a few `nvidia-smi` lines.

Copy `bundle.json` back to your machine and verify it there. The VM's own opinion of the
bundle is worth nothing; that is the point.

## Timing and cost

On a single-H200 VM the whole `attest` step takes about one minute after the SDK is installed.
Release the VM when you have the bundle; the evidence does not depend on the VM staying up.

# dstack collector (confidential GPU container)

Some providers run a GPU workload in a container inside an Intel TDX guest managed by
[dstack](https://github.com/Dstack-TEE/dstack). Inside that container the tenant has **no**
`/dev/tdx_guest` and **no** configfs TSM, so `voltage-verify attest` (which talks to configfs
TSM) cannot run. The TDX quote is instead served by the dstack guest agent over a unix socket:

```
GET /Info                          -> app_id, instance_id, the measured application
GET /GetQuote?report_data=<hex>    -> a TDX v4 quote carrying those 64 bytes
```

`collect.py` takes the same manifest the verifier prepared, asks the socket for a quote over the
SHA-512 of that manifest, runs the NVIDIA attestation for every GPU, and writes the identical
[`voltage-verify-bundle/1`](../../docs/BUNDLE_FORMAT.md). You then copy the bundle off the VM and
verify it on your own machine with `voltage-verify verify`.

## Run it

On your machine (the challenge is chosen here, never in the VM):

```bash
voltage-verify manifest --challenge auto -o manifest.json   # keep the challenge it prints
```

Copy `manifest.json`, `collect.py` and `collect.sh` into the container, then inside it:

```bash
./collect.sh manifest.json bundle.json multi-gpu-nvle
```

Copy `bundle.json` back and verify offline (pass the GPU model NRAS actually returned and the
GPU count; read both from the collector's output or `voltage-verify inspect bundle.json`):

```bash
voltage-verify verify bundle.json --offline --challenge <the one you kept> --hwmodel <MODEL> --gpus <N>
```

## What this proves, and what it does not

* **Binds the quote to your manifest, or refuses.** The collector reads the report_data back out
  of the quote and checks it equals the SHA-512 of your manifest byte for byte. If dstack hashed
  or transformed the value instead of using it raw, it writes **no** bundle and prints what it
  observed — the standard format cannot represent a transformed report_data, and shipping one
  anyway would mislead.
* **Each GPU is attested by NVIDIA (NRAS).** On a multi-GPU node the fabric between the GPUs is a
  separate property these per-GPU tokens do not cover. `verify` prints it as a WARN
  (`nvidia.scope`): NVSwitch for `multi-gpu-ppcie`, the encrypted NVLink fabric for
  `multi-gpu-nvle`.
* **The TD runs a provider-measured composition.** dstack's compose hash lands in RTMR3. This
  tool records the RTMRs and the dstack `/Info` but compares them to no published reference, so
  `verify` prints `tdx.measured_compose` as a WARN: the measured composition is trusted, not
  proven. The trust boundary is the whole measured VM, not a VM the tenant alone controls.
* **The NVIDIA SDK version is recorded** in the bundle environment. Blackwell (B200) support
  depends on it; confirm the working version on the first real run.

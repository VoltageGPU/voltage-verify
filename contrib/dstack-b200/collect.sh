#!/usr/bin/env bash
# Build a voltage-verify bundle from inside a dstack-based confidential TD (GPU container with
# /var/run/dstack.sock, no /dev/tdx_guest). Pinned installs, then collect.py does the work.
#
#   ./collect.sh <manifest.json> [out.json] [mode]
#     mode: single-gpu | multi-gpu-ppcie | multi-gpu-nvle   (default: multi-gpu-nvle)
#
# The manifest is prepared on the verifier's machine (the challenge is chosen there) and copied
# in. Nothing here chooses a secret. Versions are pinned; the installed nv-attestation-sdk is
# recorded in the bundle environment, because Blackwell (B200) support depends on it and the
# exact working version must be confirmed on the first real run.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

MANIFEST="${1:?usage: collect.sh <manifest.json> [out.json] [mode]}"
OUT="${2:-bundle.json}"
MODE="${3:-multi-gpu-nvle}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# Overridable only to confirm the Blackwell-capable SDK version on a real run; keep it pinned.
VOLTAGE_VERIFY_SPEC="${VOLTAGE_VERIFY_SPEC:-voltage-verify[attest]==0.2.3}"
NVTRUST_SDK_SPEC="${NVTRUST_SDK_SPEC:-nv-attestation-sdk==2.7.0}"

echo "== pip (dstack images often ship without it)"
if ! python3 -m pip --version >/dev/null 2>&1; then
  curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
  python3 /tmp/get-pip.py --user --break-system-packages -q
fi
python3 -m pip install --user --break-system-packages -q "$VOLTAGE_VERIFY_SPEC" "$NVTRUST_SDK_SPEC" "nvidia-ml-py>=12"

echo "== collect"
python3 "$HERE/collect.py" --manifest "$MANIFEST" --out "$OUT" --mode "$MODE"
echo "== done: $OUT (copy it off the VM and verify it on your own machine)"

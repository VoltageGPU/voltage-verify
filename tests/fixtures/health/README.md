# Health test fixtures

What the fake `nvidia-smi` and kernel log of `tests/health_fakes.py` replay. Copied unchanged
from the test fixtures of [hwcheck](https://github.com/VoltageGPU/hwcheck) 1.1 (MIT, same
author), where `tests/fixtures/README.md` and `build.sh` say how each file was made. Real
captures are kept byte for byte (tabs and escape codes included); the constructed files follow
the format of the real outputs, with typical values that are not measurements.

| File | Origin |
|---|---|
| `help-query-gpu.txt` | real: `nvidia-smi --help-query-gpu`, driver 610.88, RTX 3060 Ti, WSL2 Ubuntu 24.04, 24 October 2026. The fake rejects any field it does not list, as the real tool does |
| `rtx3060ti-1x/topo.txt` | real, same machine and day |
| `rtx3060ti-1x/gpus.csv` | real values of `nvidia-smi --query-gpu` on that machine, one column per field (UUID zeroed) |
| `h100-8x/topo.txt` | real: `nvidia-smi topo -m` of an 8x H100 SXM virtual machine, 10 September 2026 |
| `h100-8x/*` (the rest) | constructed: a healthy node; driver 580.173.02 as seen on a real H100 on 17 October 2026 |
| `h100-8x-sick/*` | constructed from `h100-8x`: GPU 7 off NVLink, GPU 2 at x8, GPU 5 at Gen4 and hot with hardware thermal slowdown, GPU 0 at a low clock, GPU 4 capped at 500 W, GPU 6 with uncorrected ECC errors, GPU 1 with a pending row remap, an Xid 13 at boot and an Xid 79 during the test |
| `pcie-8x/*` | constructed: 8 PCIe GPUs on two sockets, no NVLink |

`gpus.csv` has one column per `nvidia-smi` field; while the fake workload "runs", the columns
of `gpus-load.csv` replace them (GPUs under load). In `dmesg.txt`, `@NOW@` becomes an uptime
just after the start of the run: that line falls during the test. A fixture without `dmesg.txt`
is a machine whose kernel log cannot be read. What a PyTorch worker would report (throughput,
changed results, wrong bytes) is given by each test, not by a file.

# SPDX-License-Identifier: MIT
"""The PyTorch side of the health check, run as a separate process, one per GPU.

A GPU that falls off the bus can hang or kill the process that drives it: here it takes this
worker down, not the report. The worker prints one JSON line per stage as soon as the stage
ends, so that what finished is kept if a later stage does not.

    python -m voltage_verify.health.worker probe
    CUDA_VISIBLE_DEVICES=3 python -m voltage_verify.health.worker run --seconds 60
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m voltage_verify.health.worker p2p

Stages of ``run``, on the one GPU it sees:

* ``cross_check``  a product of integer-valued float64 matrices, on the GPU and on the CPU.
  Every partial sum is an integer far below 2**53, so both results are exact in any order of
  operations: one differing bit is a computation error.
* ``burn``  the same half-precision 8192x8192 product, again and again, under a variable load
  (bursts and pauses of changing length, for the power and thermal transients a constant load
  never causes). The result is compared with the first one after every batch: a healthy GPU
  returns the same bits every time.
* ``vram``  patterns written to at least 90% of the free memory, then read back.

``p2p`` is one more process, started after the others have ended, that sees every GPU: it
copies 256 MiB from each GPU to every other one (one ``p2p_pair`` line per ordered pair, then
one ``p2p`` line): 3 warm-up copies, each compared byte for byte with what was sent, then 20
timed copies, the last of which is compared too. It is the "P2P copies" test of hwcheck's
acceptance.sh; NCCL is not run here. It also holds 256 MiB of host memory, the bytes it sends.

PyTorch is imported, never installed. No network, and the worker writes no file (PyTorch and
the CUDA runtime may fill their own caches under the home directory).
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any

SEED = 20261122
BURN_N = 8192
# (seconds of load, seconds of pause), repeated until the time is up: about 80% under load.
SCHEDULE: tuple[tuple[float, float], ...] = ((8.0, 2.0), (4.0, 1.0), (2.0, 0.5), (1.0, 0.5))
BATCH_SECONDS = 0.25  # the result is compared about four times a second
MAX_BATCH = 25
CROSS_N = 512
CROSS_RANGE = 1000  # entries in [-1000, 1000]: |sum| <= 512 * 10**6, exact in float64
VRAM_TARGET_PERCENT = 92  # of the free memory: 90% is the pass mark, the rest is working room
MAX_CHUNK = 64 << 20
MIN_CHUNK = 1 << 20
PATTERNS: tuple[int | None, ...] = (None, 0x55, 0xAA, 0x00, 0xFF)  # None: random bytes, different in every chunk
P2P_BYTES = 256 << 20  # copied between two GPUs, as in hwcheck's acceptance.sh
P2P_ITERATIONS = 20  # timed copies per pair, after P2P_WARMUP untimed ones
P2P_WARMUP = 3
Fault = Callable[..., None]  # tests only: corrupts a result or a chunk, as faulty hardware would


def _sync(torch: Any, device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _sync_pair(torch: Any, src: str, dst: str) -> None:
    """Waits for both ends of a copy between two devices."""
    for device in (src, dst):
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)


def _half(torch: Any, device: str) -> Any:
    if not device.startswith("cuda"):
        return torch.float32  # the CPU, in the tests: half-precision matmul is a GPU matter
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def probe(torch: Any) -> dict[str, Any]:
    available = bool(torch.cuda.is_available())
    return {
        "stage": "probe",
        "torch": str(torch.__version__),
        "cuda": available,
        "devices": int(torch.cuda.device_count()) if available else 0,
    }


def cross_check(torch: Any, device: str, fault: Fault | None = None) -> dict[str, Any]:
    gen = torch.Generator().manual_seed(SEED + 1)
    a = torch.randint(-CROSS_RANGE, CROSS_RANGE + 1, (CROSS_N, CROSS_N), generator=gen).to(torch.float64)
    b = torch.randint(-CROSS_RANGE, CROSS_RANGE + 1, (CROSS_N, CROSS_N), generator=gen).to(torch.float64)
    on_device = torch.matmul(a.to(device), b.to(device))
    if fault is not None:
        fault(on_device)
    return {"stage": "cross_check", "wrong": 0 if torch.equal(on_device.cpu(), torch.matmul(a, b)) else 1}


def burn(
    torch: Any,
    device: str,
    seconds: float,
    *,
    n: int = BURN_N,
    schedule: Sequence[tuple[float, float]] = SCHEDULE,
    fault: Fault | None = None,
) -> dict[str, Any]:
    dtype = _half(torch, device)
    gen = torch.Generator().manual_seed(SEED)
    a = torch.randn(n, n, generator=gen).to(device, dtype)
    b = torch.randn(n, n, generator=gen).to(device, dtype)
    c = torch.empty(n, n, device=device, dtype=dtype)
    torch.matmul(a, b, out=c)
    _sync(torch, device)
    reference = c.clone()
    started = time.perf_counter()
    torch.matmul(a, b, out=c)
    _sync(torch, device)
    batch = max(1, min(MAX_BATCH, int(BATCH_SECONDS / max(time.perf_counter() - started, 1e-6))))

    iterations = checks = mismatch = segments = 0
    load_seconds = 0.0
    started = time.perf_counter()
    end = started + seconds
    while time.perf_counter() < end:
        load, pause = schedule[segments % len(schedule)]
        segment_start = time.perf_counter()
        segment_end = min(segment_start + load, end)
        while time.perf_counter() < segment_end:
            for _ in range(batch):
                torch.matmul(a, b, out=c)
            iterations += batch
            if fault is not None:
                fault(c, checks)
            checks += 1
            if not torch.equal(c, reference):  # reads the result back: also waits for the GPU
                mismatch += 1
        load_seconds += time.perf_counter() - segment_start
        segments += 1
        left = end - time.perf_counter()
        if left > 0:
            time.sleep(min(pause, left))
    elapsed = time.perf_counter() - started
    width = torch.int32 if dtype == torch.float32 else torch.int16
    return {
        "stage": "burn",
        "dtype": str(dtype).replace("torch.", ""),
        "n": n,
        "iterations": iterations,
        "checks": checks,
        "mismatch": mismatch,
        "finite": bool(torch.isfinite(reference).all()),
        "checksum": int(reference.view(width).to(torch.int64).sum().item()),
        "load_ms": int(load_seconds * 1000),
        "elapsed_ms": int(elapsed * 1000),
        "segments": segments,
        "gflops": int(2 * n**3 * iterations / max(load_seconds, 1e-6) / 1e9),
    }


def vram(torch: Any, device: str, *, free_bytes: int | None = None, fault: Fault | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    total = 0
    if free_bytes is None:
        torch.cuda.empty_cache()
        free_bytes, total = (int(v) for v in torch.cuda.mem_get_info(device))
    chunk = max(MIN_CHUNK, min(MAX_CHUNK, free_bytes // 64))
    goal = free_bytes * VRAM_TARGET_PERCENT // 100
    gen = torch.Generator().manual_seed(SEED + 2)
    base = torch.randint(0, 256, (chunk,), dtype=torch.uint8, generator=gen).to(device)  # counted in the room left
    chunks: list[Any] = []
    try:
        while (len(chunks) + 1) * chunk <= goal:
            chunks.append(torch.empty(chunk, dtype=torch.uint8, device=device))
    except RuntimeError:  # out of memory before the goal: test what was obtained, less working room
        del chunks[-3:]
    wrong = 0
    for number, pattern in enumerate(PATTERNS):
        for i, block in enumerate(chunks):
            if pattern is None:
                torch.bitwise_xor(base, i % 251 + 1, out=block)
            else:
                block.fill_(pattern)
        _sync(torch, device)
        if fault is not None:
            fault(chunks, number)
        for i, block in enumerate(chunks):
            expected = torch.bitwise_xor(base, i % 251 + 1) if pattern is None else pattern
            wrong += int((block != expected).sum().item())
    return {
        "stage": "vram",
        "free_bytes": free_bytes,
        "total_bytes": total,
        "tested_bytes": len(chunks) * chunk,
        "chunk_bytes": chunk,
        "chunks": len(chunks),
        "patterns": len(PATTERNS),
        "wrong_bytes": wrong,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


def peer_access(torch: Any, devices: Sequence[str], src: int, dst: int) -> bool:
    """What CUDA says of direct access between two of the visible GPUs. Devices that are not
    CUDA ones (the CPU, in the tests) have nothing to ask."""
    if devices[src].startswith("cuda") and devices[dst].startswith("cuda"):
        return bool(torch.cuda.can_device_access_peer(src, dst))
    return True


def p2p(
    torch: Any,
    devices: Sequence[str],
    *,
    size: int = P2P_BYTES,
    iterations: int = P2P_ITERATIONS,
    fault: Fault | None = None,
) -> Iterator[dict[str, Any]]:
    """Copies ``size`` bytes from every device to every other one. One record per ordered pair
    as soon as it is measured (devices numbered as they are listed), then the closing record.

    The bytes are random, drawn once on the CPU and sent to every device, then combined on
    each device with its own number: every source holds other bytes, and what must arrive is
    computed on the destination without going through the copy under test. (A byte that
    changes on its way from the host shows on every pair of that device.)

    Per pair, P2P_WARMUP + 1 copies are compared byte for byte with what must arrive: each
    warm-up copy, outside the timing, and the last of the timed copies. ``wrong_bytes`` is
    their sum. The timed copies before the last one are overwritten unread.
    """
    started = time.perf_counter()
    count = len(devices)
    closing = {"stage": "p2p", "devices": count, "bytes": size, "warmup": P2P_WARMUP, "iterations": iterations}
    if count < 2:
        yield {**closing, "pairs": 0, "elapsed_ms": 0}
        return
    gen = torch.Generator().manual_seed(SEED + 3)
    pattern = torch.randint(0, 256, (size,), dtype=torch.uint8, generator=gen)
    base = [pattern.to(device) for device in devices]
    marks = [number % 255 + 1 for number in range(count)]
    sources = [torch.bitwise_xor(block, mark) for block, mark in zip(base, marks, strict=True)]
    arrived = [torch.empty(size, dtype=torch.uint8, device=device) for device in devices]

    def wrong_bytes(src: int, dst: int, expected: Any, check: int) -> int:
        if fault is not None:
            fault(arrived[dst], src, dst, check)
        return int((arrived[dst] != expected).sum().item())

    pairs = 0
    for src in range(count):
        for dst in range(count):
            if src == dst:
                continue
            expected = torch.bitwise_xor(base[dst], marks[src])
            wrong = 0
            for check in range(P2P_WARMUP):
                arrived[dst].copy_(sources[src], non_blocking=True)
                _sync_pair(torch, devices[src], devices[dst])
                wrong += wrong_bytes(src, dst, expected, check)
            timed = time.perf_counter()
            for _ in range(iterations):
                arrived[dst].copy_(sources[src], non_blocking=True)
            _sync_pair(torch, devices[src], devices[dst])
            elapsed = max(time.perf_counter() - timed, 1e-9)
            wrong += wrong_bytes(src, dst, expected, P2P_WARMUP)
            pairs += 1
            yield {
                "stage": "p2p_pair",
                "src": src,
                "dst": dst,
                "mib_per_s": int(size * iterations / elapsed) >> 20,
                "wrong_bytes": wrong,
                "peer": peer_access(torch, devices, src, dst),
            }
    yield {**closing, "pairs": pairs, "elapsed_ms": int((time.perf_counter() - started) * 1000)}


def _emit(record: dict[str, Any]) -> None:
    print(json.dumps(record, sort_keys=True), flush=True)


def _seconds(value: str) -> int:
    if not value.isascii() or not value.isdigit() or not 1 <= int(value) <= 3600:
        raise argparse.ArgumentTypeError("expected 1 to 3600 seconds")
    return int(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="voltage_verify.health.worker")
    parser.add_argument("command", choices=["probe", "run", "p2p"])
    parser.add_argument("--seconds", type=_seconds, default=60)
    args = parser.parse_args(argv)
    try:
        torch = importlib.import_module("torch")
    except Exception as err:  # noqa: BLE001  not installed, or a build that does not load here
        _emit({"stage": "probe", "torch": None, "cuda": False, "devices": 0, "error": type(err).__name__})
        return 0
    if args.command == "probe":
        _emit(probe(torch))
        return 0
    if args.command == "p2p":
        try:  # every GPU CUDA_VISIBLE_DEVICES leaves visible, in that order
            for record in p2p(torch, [f"cuda:{number}" for number in range(int(torch.cuda.device_count()))]):
                _emit(record)
        except Exception as err:  # noqa: BLE001  the pairs already printed are kept, the rest is named
            _emit({"stage": "p2p", "error": f"{type(err).__name__}: {str(err)[:200]}"})
        return 0
    device = "cuda:0"  # the one GPU CUDA_VISIBLE_DEVICES leaves visible
    stages: tuple[tuple[str, Callable[[], dict[str, Any]]], ...] = (
        ("cross_check", lambda: cross_check(torch, device)),
        ("burn", lambda: burn(torch, device, args.seconds)),
        ("vram", lambda: vram(torch, device)),
    )
    for name, stage in stages:
        try:
            _emit(stage())
        except Exception as err:  # noqa: BLE001  reported as that stage's failure, the next one still runs
            _emit({"stage": name, "error": f"{type(err).__name__}: {str(err)[:200]}"})
    return 0


if __name__ == "__main__":
    sys.exit(main())

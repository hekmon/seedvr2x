#!/usr/bin/env python3
"""Over-commit probe for seedvr2x: what a card does when a window passes it, under a driver that
spills into system memory instead of refusing (Windows' sysmem fallback, WSL included). No
BlockSwap here: the models loaded whole, as a run without a plan loads them; the windows are
placement_bench.py's.

  A  windows of --small latents at 1080p: the reference digests, the peak, the least free memory
  T  the same with the allocator's pool trimmed before each (torch.cuda.empty_cache)
  B  one window of --large latents, which passes the card: its time, the least free memory
  C  A again, right after: the same digests?
  D  A again after a trim

Measured 2026-10-11 on an RTX 5090 under WSL (30.20 GiB free, cudaMallocAsync, SDPA), --small 6:
A's peak 22.535 GiB with 4.1 to 4.7 GiB free at the least, the same trimmed (T); --large 11
(27.88 GiB) left 0.00 free, took 9.5 s, and C and D kept A's digests; --large 12 (28.93 GiB)
aborted the process: "CUDA error: out of memory ... Sticky error detected ... from
cuMemFreeAsync", raised in a tensor's destructor (SIGABRT), no OutOfMemoryError.

Usage (seedvr2x's venv python, under the GPU's lock; it fills the card on purpose):
  python spill_probe.py --dit DIT.safetensors --vae VAE.safetensors --small 6 --large 11
"""

import argparse
import os
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dit", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--bench-dir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--small", type=int, default=6)
    parser.add_argument("--large", type=int, default=12)
    parser.add_argument("--skip-large", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, args.bench_dir)
    import placement_bench as bench  # sets the allocator's default before torch is imported
    from pathlib import Path

    import torch

    from seedvr2x.runtime import job, model

    device = torch.device("cuda:0")
    free, total = torch.cuda.mem_get_info(device)
    print(f"free {free / bench.GIB:.2f} GiB of {total / bench.GIB:.2f}", flush=True)
    models = model.load_models(Path(args.dit), Path(args.vae), device)
    target = (1080, 1920)
    height, width = job.padded_size(target)

    def windows(label: str, length: int, count: int, trim: bool = False) -> list[str]:
        latent = bench.make_latent(torch, model.COMPUTE_DTYPE, length, height // 8, width // 8)
        digests = []
        for index in range(count):
            if trim:
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
            before = torch.cuda.mem_get_info(device)[0] / bench.GIB
            with bench.FreeSampler(torch, device) as sampler:
                started = time.perf_counter()
                run = bench.one_forward(torch, model, models, latent, 42 + index)
                took = time.perf_counter() - started
            digests.append(run["sha256"])
            print(
                f"{label} L={length} #{index}: peak {run['peak_gib']:.3f} GiB, wall"
                f" {run['wall_s']:.2f} s ({took:.2f} with the copies), free before {before:.2f},"
                f" least free {sampler.least:.2f}, allocated after"
                f" {torch.cuda.memory_allocated(device) / bench.GIB:.3f}, {run['sha256'][:12]}",
                flush=True,
            )
        return digests

    a = windows("A", args.small, 3)
    t = windows("T (trimmed)", args.small, 3, trim=True)
    print(f"T equals A: {t == a}", flush=True)
    if not args.skip_large:
        windows("B", args.large, 1)
        c = windows("C", args.small, 3)
        print(f"C equals A: {c == a}", flush=True)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    d = windows("D", args.small, 3)
    print(f"D equals A: {d == a}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

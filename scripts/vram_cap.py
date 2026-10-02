#!/usr/bin/env python3
"""Emulate a smaller GPU: cap the device memory SeedVR2 can use, then run the CLI in-process.

The card is described by its nominal size N ("a 12 GB card" -> 12, in GiB, which is what
GDDR cards really have). What a process can use on such a card is:

  capacity = N - hidden        what cudaMemGetInfo reports as total (the driver keeps some
                               memory out of sight: 0.95 GiB on a 96 GB RTX PRO 6000,
                               ≈ 0.25-0.65 GiB on consumer cards; default 0.35)
  usable   = capacity - other  minus what other processes hold (desktop, compositor,
                               browser; default 0.3, 0 for a headless card)

`usable` includes the process's own CUDA context (≈ 0.7-0.8 GiB, measured, not assumed), so
torch itself gets ≈ N - 1.4 GiB with the defaults.

Modes (VRAM_CAP_MODE):
  ballast   (default) after creating the CUDA context, allocate `total - capacity + other`
            bytes with the driver API (cuMemAlloc, outside torch's allocator, so torch's
            allocated / reserved / peak figures are untouched) and keep them until exit. The
            device is then really full where the emulated card would be: works with every
            allocator backend (cudaMallocAsync, native, expandable segments), counts the
            context, cuBLAS/cuDNN workspaces and allocator overhead, and makes the free memory
            SeedVR2 reads (torch.cuda.mem_get_info) realistic. mem_get_info's total is patched
            to `capacity`, so SeedVR2's "< 5% free" check (clear_memory(force=False), called
            after every swapped block) uses the emulated card's size.
  fraction  torch.cuda.set_per_process_memory_fraction((usable - context) / total): caps
            torch's own allocations only (honoured by the native and cudaMallocAsync backends
            on torch 2.14); free memory and memory outside torch's allocator are not
            emulated. For cross-checks.

bench.py reads the "vram_cap: ballast" line from the log and subtracts the ballast from its
NVML figures, so they read as the emulated card's device memory used.

Environment:
  VRAM_CAP_GIB         nominal card size, GiB (required)
  VRAM_CAP_HIDDEN_GIB  default 0.35
  VRAM_CAP_OTHER_GIB   default 0.3
  VRAM_CAP_MODE        ballast | fraction (default ballast)

The CLI sets PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync unless it is already set; this
wrapper imports torch first, so it applies the same default itself.

Usage (cwd = the SeedVR2 checkout, its venv's python):
  VRAM_CAP_GIB=12 python vram_cap.py inference_cli.py <CLI args>
  python3 bench.py run NAME --wrap vram_cap.py --env VRAM_CAP_GIB=12 -- <CLI args>
"""
import ctypes
import os
import runpy
import sys

GIB = 1024 ** 3


def log(msg):
    print(f"vram_cap: {msg}", flush=True)


def setup():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")  # the CLI's default
    n = float(os.environ["VRAM_CAP_GIB"])
    hidden = float(os.environ.get("VRAM_CAP_HIDDEN_GIB", "0.35"))
    other = float(os.environ.get("VRAM_CAP_OTHER_GIB", "0.3"))
    mode = os.environ.get("VRAM_CAP_MODE", "ballast")
    import torch

    torch.cuda.init()
    torch.zeros(1, device="cuda")  # creates the primary context, current on this thread
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    used = total - free           # this process's context (+ whatever else is on the device)
    capacity = int((n - hidden) * GIB)
    usable = capacity - int(other * GIB)
    if usable - used < GIB:
        sys.exit(f"vram_cap: {n} GiB card leaves {(usable - used) / GIB:.2f} GiB, too little")
    log(f"card {n:g} GiB, hidden {hidden:g}, other {other:g} -> capacity {capacity / GIB:.2f} GiB, "
        f"usable {usable / GIB:.2f} GiB; device total {total / GIB:.2f}, already used {used / GIB:.2f} "
        f"(context), left for torch {(usable - used) / GIB:.2f} GiB; mode {mode}, "
        f"alloc conf {os.environ['PYTORCH_CUDA_ALLOC_CONF']}")

    if mode == "fraction":
        torch.cuda.set_per_process_memory_fraction((usable - used) / total)
        log(f"memory fraction {(usable - used) / total:.4f}")
        return
    if mode != "ballast":
        sys.exit(f"vram_cap: unknown VRAM_CAP_MODE {mode!r}")
    ballast = total - capacity + int(other * GIB)
    if ballast > free:
        sys.exit(f"vram_cap: device too small to emulate a {n:g} GiB card")
    cuda = ctypes.CDLL("libcuda.so.1")
    ptr = ctypes.c_uint64()
    rc = cuda.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(ballast))
    if rc != 0:
        sys.exit(f"vram_cap: cuMemAlloc({ballast}) failed with CUresult {rc}")
    setup.keep = ptr  # never freed: the driver releases it at exit
    free2, _ = torch.cuda.mem_get_info()
    log(f"ballast {ballast} bytes ({ballast / GIB:.3f} GiB); free now {free2 / GIB:.2f} GiB")

    real = torch.cuda.mem_get_info

    def mem_get_info(device=None):
        f, _t = real(device)
        return f, capacity

    torch.cuda.mem_get_info = mem_get_info
    torch.cuda.memory.mem_get_info = mem_get_info


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help") or "VRAM_CAP_GIB" not in os.environ:
        sys.exit(__doc__)
    script = os.path.abspath(sys.argv[1])
    sys.argv = [script, *sys.argv[2:]]
    sys.path[0] = os.path.dirname(script)
    setup()
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()

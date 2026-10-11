#!/usr/bin/env python3
"""Pinned-memory probe for seedvr2x's BlockSwap (runtime/blockswap.py, HostCopy): what torch's
pinned allocator takes for a buffer the size of a 7B block, whether a buffer of the exact size,
mapped and registered with cudaHostRegister, copies to the device as fast, and what a driver does
with a request for more device memory than is free.

Prints, for a block's 0.4220 GiB: the process's memory after each allocation; the time a
non-blocking copy to the device takes to return and to finish, and its rate, from pageable
memory, from torch's pinned memory and from registered memory; then asks for 2 GiB more than the
device has free, in one request, freed at once if given: a driver that refuses says so in an
instant, one that spills into system memory (Windows' sysmem fallback, WSL included) gives it.

Measured 2026-10-10 on an RTX 5090 under WSL (torch 2.14.1): torch's pinned allocator took
536,870,912 bytes for 453,126,144 asked (a power of two); 18.2 GB/s from pageable memory, the
call blocking 24.9 ms; 53.4 GB/s from torch's pinned memory and 54.6 from registered memory,
the call returning in 0.11 and 0.12 ms; 31.75 GiB asked with 29.75 free: given in 0.49 s.

Usage (seedvr2x's venv python, under the GPU's lock; holds about 1 GiB of device memory for a few
seconds, and all of it for an instant if the driver spills):
  PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync python pin_probe.py
"""

import mmap
import sys
import time

import torch

GIB = 1 << 30
BLOCK = 453_126_144  # about one 7B fp16 block (0.4220 GiB)


def meminfo(key: str) -> float:
    with open("/proc/meminfo") as file:
        for line in file:
            if line.startswith(key + ":"):
                return int(line.split()[1]) / (1 << 20)
    raise KeyError(key)


def status(key: str) -> float:
    with open("/proc/self/status") as file:
        for line in file:
            if line.startswith(key + ":"):
                return int(line.split()[1]) / (1 << 20)
    raise KeyError(key)


def state(label: str) -> None:
    print(
        f"{label}: RSS {status('VmRSS'):.3f} GiB, locked {status('VmLck'):.3f},"
        f" pinned {status('VmPin'):.3f}, MemAvailable {meminfo('MemAvailable'):.2f}"
    )


def bandwidth(label: str, host: torch.Tensor, device: torch.device) -> None:
    slot = torch.empty(host.numel(), dtype=torch.uint8, device=device)
    torch.cuda.synchronize()
    for _ in range(2):  # the second, warm
        started = time.perf_counter()
        slot.copy_(host, non_blocking=True)
        returned = time.perf_counter()
        torch.cuda.synchronize()
        done = time.perf_counter()
    rate = host.numel() / (done - started) / 1e9
    print(
        f"{label}: is_pinned {host.is_pinned()}, the call returned in"
        f" {1e3 * (returned - started):.2f} ms, done in {1e3 * (done - started):.2f} ms,"
        f" {rate:.1f} GB/s"
    )
    back = slot.cpu()
    assert torch.equal(back, host)
    del slot, back


def main() -> None:
    print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))
    device = torch.device("cuda:0")
    free, total = torch.cuda.mem_get_info(device)
    print(f"free {free / GIB:.2f} GiB of {total / GIB:.2f}")
    state("start")

    pageable = torch.empty(BLOCK, dtype=torch.uint8).random_()
    state("pageable block")
    bandwidth("pageable", pageable, device)

    pinned = torch.empty(BLOCK, dtype=torch.uint8, pin_memory=True)
    pinned.copy_(pageable)
    state("torch-pinned block")
    stats = torch.cuda.host_memory_stats()
    print({k: v for k, v in stats.items() if "current" in k and ("bytes" in k or "segment" in k)})
    bandwidth("torch-pinned", pinned, device)
    del pinned
    state("torch-pinned block deleted (cached by the allocator?)")
    torch._C._host_emptyCache()
    state("host cache emptied")

    area = mmap.mmap(-1, BLOCK, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    registered = torch.frombuffer(area, dtype=torch.uint8)
    registered.copy_(pageable)
    cudart = torch.cuda.cudart()
    code = cudart.cudaHostRegister(registered.data_ptr(), BLOCK, 0)
    print("cudaHostRegister:", code)
    state("registered block")
    bandwidth("registered", registered, device)
    print("cudaHostUnregister:", cudart.cudaHostUnregister(registered.data_ptr()))
    del registered
    area.close()
    state("unregistered")

    # An over-allocation: 2 GiB more than what is free, in one request, freed at once if given.
    free, total = torch.cuda.mem_get_info(device)
    want = free + 2 * GIB
    try:
        started = time.perf_counter()
        big = torch.empty(want, dtype=torch.uint8, device=device)
        took = time.perf_counter() - started
        after, _ = torch.cuda.mem_get_info(device)
        print(
            f"OVER-ALLOCATION GIVEN: {want / GIB:.2f} GiB asked with {free / GIB:.2f} free, in"
            f" {took:.2f} s; free after {after / GIB:.2f}: the driver spills into system memory"
        )
        del big
    except torch.OutOfMemoryError as error:
        took = time.perf_counter() - started
        print(
            f"over-allocation refused in {took:.2f} s: {want / GIB:.2f} GiB asked with"
            f" {free / GIB:.2f} free: {str(error).splitlines()[0][:200]}"
        )
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    print(f"free at the end {free / GIB:.2f} GiB")


if __name__ == "__main__":
    sys.exit(main())

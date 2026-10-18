"""What torch's cudaMallocAsync allocator lets a run see of its refused allocations (the frames'
floor, floor.py).

One process, the GPU's lock held by the caller. In order:
1. the allocator in force and memory_stats()'s keys and values at the start;
2. an OOM observer attached (torch._C._cuda_attach_out_of_memory_observer), or why not;
3. a request refused outright (the card filled by the driver API, outside torch): what changed
   in memory_stats(), whether the observer was called, what torch said;
4. a request that fails first and is given after the pool is trimmed (a block freed into the
   pool, the card filled around it): the same readings, and torch's warning;
5. one convolution as a VAE decode runs its pieces (S1's `conv` probe: 3x3x3, 256 channels,
   bfloat16, 1x256x6x160x640) with 2.0 GiB left, then 0.5 GiB left, then the memory free again:
   per call, its bits against the first call's, its seconds, and every memory_stats() key that
   moved, the observer's calls.
Nothing here is seedvr2x's; the readings are printed as `probe:` lines. Run alone on the GPU,
under its lock."""

import ctypes
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

import torch

GIB = 1024**3
calls = []


def say(text):
    print(f"probe: {text}", flush=True)


def stats():
    return dict(torch.cuda.memory_stats())


def moved(before, after):
    keys = sorted(set(before) | set(after))
    return {k: (before.get(k), after.get(k)) for k in keys if before.get(k) != after.get(k)}


def observer(*given):
    calls.append(given)


cuda = ctypes.CDLL("libcuda.so.1")


def driver_take(size):
    ptr = ctypes.c_uint64()
    rc = cuda.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(size))
    if rc != 0:
        raise RuntimeError(f"cuMemAlloc({size}) failed: {rc}")
    return ptr


def driver_give(ptr):
    cuda.cuMemFree_v2(ptr)


def leave(left_gib):
    """Fill the card by the driver API so that `left_gib` GiB stay free."""
    torch.cuda.synchronize()
    free = torch.cuda.mem_get_info()[0]
    fill = free - round(left_gib * GIB)
    return driver_take(fill), fill


torch.cuda.init()
torch.zeros(1, device="cuda")
say(
    f"torch {torch.__version__}, allocator backend {torch.cuda.get_allocator_backend()},"
    f" conf {os.environ['PYTORCH_CUDA_ALLOC_CONF']}, cuDNN {torch.backends.cudnn.version()},"
    f" benchmark {torch.backends.cudnn.benchmark},"
    f" deterministic {torch.backends.cudnn.deterministic}"
)
start = stats()
say(f"memory_stats at the start: {len(start)} keys: {start}")
try:
    torch._C._cuda_attach_out_of_memory_observer(observer)
    say("OOM observer attached")
except Exception as error:
    say(f"OOM observer NOT attached: {error!r}")

# 3. A request refused outright.
ballast, fill = leave(0.3)
before = stats()
try:
    big = torch.empty(GIB, dtype=torch.uint8, device="cuda")
    say("3: 1 GiB with 0.3 left: GIVEN (unexpected)")
    del big
except torch.OutOfMemoryError as error:
    say(f"3: 1 GiB with 0.3 left: REFUSED: {str(error).splitlines()[0]}")
after = stats()
say(f"3: memory_stats moved: {moved(before, after)}; observer calls {len(calls)}: {calls[-1:]}")
driver_give(ballast)
torch.cuda.synchronize()

# 4. A request given after the pool is trimmed.
torch.cuda.empty_cache()
block = torch.empty(2 * GIB, dtype=torch.uint8, device="cuda")
del block  # back to the pool, reserved
torch.cuda.synchronize()
say(
    f"4: a 2 GiB block freed into the pool: allocated {torch.cuda.memory_allocated()} reserved"
    f" {torch.cuda.memory_reserved()} driver free {torch.cuda.mem_get_info()[0]}"
)
ballast, fill = leave(0.1)
before, seen = stats(), len(calls)
try:
    asked = torch.empty(round(2.05 * GIB), dtype=torch.uint8, device="cuda")
    say("4: 2.05 GiB with the pool's 2 free and 0.1 on the card: GIVEN")
    del asked
except torch.OutOfMemoryError as error:
    say(f"4: 2.05 GiB: REFUSED: {str(error).splitlines()[0]}")
after = stats()
say(f"4: memory_stats moved: {moved(before, after)}; observer calls {len(calls) - seen}")
driver_give(ballast)
torch.cuda.synchronize()
torch.cuda.empty_cache()

# 5. One convolution, short of memory.
torch.manual_seed(0)
layer = (
    torch.nn.Conv3d(256, 256, 3, padding=(0, 1, 1)).to("cuda", torch.bfloat16).requires_grad_(False)
)
x = torch.randn((1, 256, 6, 160, 640), generator=torch.Generator().manual_seed(1)).to(
    "cuda", torch.bfloat16
)


def conv():
    torch.cuda.synchronize()
    started = time.monotonic()
    with torch.no_grad():
        out = layer(x)
    torch.cuda.synchronize()
    return out, time.monotonic() - started


reference, _ = conv()
reference = reference.cpu()
for label, left in (("2.0 left", 2.0), ("0.5 left", 0.5), ("free again", None)):
    torch.cuda.empty_cache()
    ballast = None
    if left is not None:
        ballast, fill = leave(left)
    for repeat in range(2):
        before, seen = stats(), len(calls)
        try:
            out, seconds = conv()
            same = bool(torch.equal(out.cpu(), reference))
            del out
            result = f"{'the same bits' if same else 'OTHER BITS'} in {seconds:.3f} s"
        except torch.OutOfMemoryError as error:
            result = f"REFUSED: {str(error).splitlines()[0]}"
        after = stats()
        say(
            f"5: {label}, call {repeat + 1}: {result}; memory_stats moved:"
            f" {moved(before, after)}; observer calls {len(calls) - seen}"
        )
    if ballast is not None:
        driver_give(ballast)
        torch.cuda.synchronize()
say(f"end: memory_stats: {stats()}; observer calls in all {len(calls)}")
sys.exit(0)

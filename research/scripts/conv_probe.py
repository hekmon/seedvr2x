"""One convolution short of memory, as alloc_probe.py's step 5, alone: a first call with the memory
free, then two with 0.5 GiB left and two with the memory free again, each one's bits against the
first call's and its seconds. Run under an environment variable whose output may say a refused
cuDNN workspace: TORCH_CUDNN_V8_API_DEBUG=1, CUDNN_LOGLEVEL_DBG=2, or CUDNN_FRONTEND_LOG_INFO=1
with CUDNN_FRONTEND_LOG_FILE=stderr, the cuDNN frontend's log, which names every execution plan
built, its engine and its workspace. Run alone on the GPU, under its lock. Prints `probe:` lines."""

import ctypes
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

import torch

GIB = 1024**3
cuda = ctypes.CDLL("libcuda.so.1")


def say(text):
    print(f"probe: {text}", flush=True)


def leave(left_gib):
    torch.cuda.synchronize()
    free = torch.cuda.mem_get_info()[0]
    ptr = ctypes.c_uint64()
    rc = cuda.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(free - round(left_gib * GIB)))
    assert rc == 0, rc
    return ptr


say(
    f"env: TORCH_CUDNN_V8_API_DEBUG={os.environ.get('TORCH_CUDNN_V8_API_DEBUG')}"
    f" CUDNN_LOGLEVEL_DBG={os.environ.get('CUDNN_LOGLEVEL_DBG')}"
    f" CUDNN_FRONTEND_LOG_INFO={os.environ.get('CUDNN_FRONTEND_LOG_INFO')}"
)
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
say("first call done (memory free)")
for label, left in (("0.5 left", 0.5), ("free again", None)):
    torch.cuda.empty_cache()
    ballast = leave(left) if left is not None else None
    for repeat in range(2):
        say(f"--- {label}, call {repeat + 1} begins")
        out, seconds = conv()
        same = bool(torch.equal(out.cpu(), reference))
        del out
        say(
            f"{label}, call {repeat + 1}: {'the same bits' if same else 'OTHER BITS'}"
            f" in {seconds:.3f} s"
        )
    if ballast is not None:
        cuda.cuMemFree_v2(ballast)
        torch.cuda.synchronize()

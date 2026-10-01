"""Probe a SeedVR2 Python environment: versions, and whether each attention
backend actually runs on the current GPU (not just imports).

Calls SeedVR2's own varlen wrappers (src/optimization/compatibility.py), so it
exercises the exact code path the DiT uses, and compares each against a
per-sequence PyTorch SDPA reference.

Usage: <venv>/bin/python probe_env.py [SEEDVR2_DIR]   (default: $SEEDVR2_DIR or .)
"""
import importlib.metadata as md
import os
import platform
import sys
import time
import traceback

seedvr2_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SEEDVR2_DIR", ".")
sys.path.insert(0, os.path.abspath(seedvr2_dir))

import torch
import torch.nn.functional as F


def section(title):
    print(f"\n== {title}", flush=True)


section("interpreter / torch")
print("python", sys.version.split()[0], platform.python_implementation(), sys.executable)
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| cudnn", torch.backends.cudnn.version())
print("device", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0))
print("torch arch list", torch.cuda.get_arch_list())
print("PYTORCH_CUDA_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))

section("attention / compile related distributions")
keys = ("sage", "flash", "triton", "xformers", "torch", "nvidia-cudnn", "nvidia-cuda-runtime")
for d in sorted(md.distributions(), key=lambda d: d.metadata["Name"].lower()):
    name = d.metadata["Name"]
    if any(k in name.lower() for k in keys):
        print(f"  {name}=={d.version}")

section("other key distributions")
for p in ["numpy", "diffusers", "peft", "safetensors", "gguf", "opencv-python", "transformers",
          "accelerate", "einops", "omegaconf", "rotary-embedding-torch", "bitsandbytes"]:
    try:
        print(f"  {p}=={md.version(p)}")
    except md.PackageNotFoundError:
        print(f"  {p}: not installed")

section("SeedVR2 compatibility layer")
from src.optimization import compatibility as C  # noqa: E402

for flag in ["FLASH_ATTN_2_AVAILABLE", "FLASH_ATTN_3_AVAILABLE", "SAGE_ATTN_2_AVAILABLE",
             "SAGE_ATTN_3_AVAILABLE", "TRITON_AVAILABLE", "GGUF_AVAILABLE"]:
    print(f"  {flag} = {getattr(C, flag, '?')}")
for fn in ["flash_attn_2_varlen_func", "flash_attn_3_varlen_func", "sageattn_varlen", "sageattn_blackwell"]:
    f = getattr(C, fn, None)
    print(f"  {fn}: {getattr(f, '__module__', None)}")
for mode in ["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"]:
    print(f"  validate_attention_mode({mode!r}) -> {C.validate_attention_mode(mode)!r}")

try:
    import triton
    print("  triton", triton.__version__)
except Exception as e:  # noqa: BLE001
    print("  triton import failed:", e)

section("kernel runs (vs SDPA reference)")
H, D = 24, 128  # 7B DiT: heads=24, head_dim=128 (3B: 20 heads)
dev, dt = "cuda", torch.bfloat16


def make(lens):
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32, device=dev)
    total = int(cu[-1])
    q, k, v = (torch.randn(total, H, D, device=dev, dtype=dt) for _ in range(3))
    return q, k, v, cu, max(lens)


def reference(q, k, v, cu):
    outs = []
    for i in range(len(cu) - 1):
        s, e = int(cu[i]), int(cu[i + 1])
        qi, ki, vi = (t[s:e].transpose(0, 1)[None] for t in (q, k, v))
        outs.append(F.scaled_dot_product_attention(qi, ki, vi)[0].transpose(0, 1))
    return torch.cat(outs)


backends = {
    "flash_attn_2": C.call_flash_attn_2_varlen,
    "flash_attn_3": C.call_flash_attn_3_varlen,
    "sageattn_2": C.call_sage_attn_2_varlen,
    "sageattn_3": C.call_sage_attn_3_varlen,
}
cases = {
    "uniform 4x2048": [2048] * 4,
    "varlen 1024/3000/2048": [1024, 3000, 2048],
}
for case, lens in cases.items():
    q, k, v, cu, mx = make(lens)
    ref = reference(q, k, v, cu).float()
    print(f"  [{case}]")
    for name, fn in backends.items():
        try:
            out = fn(q, k, v, cu, cu, mx, mx).float()
            torch.cuda.synchronize()
            err = (out - ref).abs()
            cos = F.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
            print(f"    {name:13s} OK   max_abs={err.max().item():.4f} mean_abs={err.mean().item():.5f} cos={cos:.6f}")
        except Exception as e:  # noqa: BLE001
            msg = str(e).strip().splitlines()[0] if str(e).strip() else ""
            print(f"    {name:13s} FAIL {type(e).__name__}: {msg[:160]}")


def bench(fn, n=20):
    for _ in range(5):  # warmup (Triton JIT / autotune happens on first calls)
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t0) / n


for nseq, seqlen in [(8, 4096), (4, 16384)]:
    section(f"throughput: {nseq} x {seqlen} tokens, {H} heads x {D}, bf16 (ms per call, TFLOPS)")
    q, k, v, cu, mx = make([seqlen] * nseq)
    flops = 4 * nseq * H * seqlen * seqlen * D
    qb, kb, vb = (t.view(nseq, seqlen, H, D) for t in (q, k, v))  # batched NHD view, same data
    timed = {
        # what SeedVR2 calls, per --attention_mode
        "sdpa (SeedVR2 loop)": lambda: reference(q, k, v, cu),
        **{f"{n} (SeedVR2)": (lambda f=f: f(q, k, v, cu, cu, mx, mx)) for n, f in backends.items()},
        # reference points SeedVR2 does not use
        "sdpa batched": lambda: F.scaled_dot_product_attention(qb.transpose(1, 2), kb.transpose(1, 2), vb.transpose(1, 2)),
    }
    try:
        from sageattention import sageattn  # SA2 batched API: CUDA int8 QK + fp8 PV kernels on sm89/sm120

        timed["sageattn batched (SA2 CUDA)"] = lambda: sageattn(qb, kb, vb, tensor_layout="NHD")
    except Exception:  # noqa: BLE001
        pass
    for name, fn in timed.items():
        try:
            ms = bench(fn)
            print(f"  {name:30s} {ms:8.3f} ms  {flops / ms / 1e9:7.1f} TFLOPS")
        except Exception as e:  # noqa: BLE001
            print(f"  {name:30s} FAIL {type(e).__name__}")
    del q, k, v, qb, kb, vb
    torch.cuda.empty_cache()

if os.environ.get("PROBE_TRACEBACK"):
    traceback.print_exc()

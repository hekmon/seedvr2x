# /// script
# requires-python = "==3.12.*"
# dependencies = ["torch==2.14.1", "numpy==2.5.2", "gguf==0.19.0"]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""What each format does to the weights: its error against ByteDance's fp32 master, tensor by
tensor. No file is made: the figures of models/FORMATS.md.

    uv run models/formats_study.py MASTER.pth OUT.json [--numz-q4 FILE.gguf]

- The 288 attention and MLP matrices of the blocks (99% of the weights), each format simulated
  from the master: float16 (nearest, ties to even); fp8 E4M3 with no scale, one scale per tensor
  (max|W| / 448), one per output row; int8 with one scale per row (max / 127); GGUF Q8_0
  (gguf-py's quantizer); NVFP4 (E2M1 values, an E4M3 scale per 16 values along the input, a
  float32 scale per tensor, max|W| / (448 x 6)), its block scales from each block's max as
  comfy-kitchen sets them, and chosen among the max's scale and the 7 E4M3 values below it to
  minimise each block's squared error; with --numz-q4, numz's Q4_K_M file as it is (gguf-py's
  decoder).
- Each matrix's error: ||W_hat - W|| / ||W||. Per format: the median over the matrices, the
  pooled error (over all their weights at once), the worst matrix. And, for fp8 with no scale,
  the share of each matrix's nonzero weights in E4M3's subnormal range (below 2^-6).
- The 840 other tensors (the 6 matrices outside the blocks, biases, norms, modulation, RoPE's
  frequencies) with fp8 with no scale, as numz's fp8 file stores them, and with one scale per
  tensor.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFReader
from gguf.quants import dequantize, quantize

E4 = torch.float8_e4m3fn
BLOCK = re.compile(
    r"^blocks\.\d+\.(attn\.proj_(qkv|out)\.(txt|vid)|mlp\.(txt|vid)\.proj_(in|out))\.weight$"
)
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
MID = (E2M1[1:] + E2M1[:-1]) / 2
MIN_NORMAL = 2.0**-6


def rel(a: torch.Tensor, w: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(a - w, dtype=torch.float64)
        / torch.linalg.vector_norm(w, dtype=torch.float64)
    )


def fp8_scaled(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return (w * (1.0 / s)).clamp(-448.0, 448.0).to(E4).float() * s


def fp8_per_row(w: torch.Tensor) -> torch.Tensor:
    s = w.abs().amax(dim=1, keepdim=True) / 448.0
    return fp8_scaled(w, torch.where(s == 0, torch.ones_like(s), s))


def int8_per_row(w: torch.Tensor) -> torch.Tensor:
    s = w.abs().amax(dim=1, keepdim=True) / 127.0
    s = torch.where(s == 0, torch.ones_like(s), s)
    return torch.round(w / s).clamp(-127, 127) * s


def q8_0(w: torch.Tensor) -> torch.Tensor:
    a = w.numpy()
    q = quantize(a, GGMLQuantizationType.Q8_0)
    return torch.from_numpy(dequantize(q, GGMLQuantizationType.Q8_0).reshape(a.shape))


def e2m1(x: torch.Tensor) -> torch.Tensor:
    return E2M1[torch.bucketize(x.abs().clamp(max=6.0), MID)] * x.sign()


def nvfp4(w: torch.Tensor, below: int) -> torch.Tensor:
    """NVFP4, each block's scale the max's E4M3 code or, if it gives a smaller squared error,
    one of the `below` codes under it."""
    s2 = w.abs().max() / (448.0 * 6.0)
    b = w.reshape(w.shape[0], -1, 16)
    code = (b.abs().amax(dim=2, keepdim=True) / 6.0 / s2).clamp(max=448.0).to(E4).view(torch.uint8)
    best, best_err = None, None
    for j in range(below + 1):
        c = (code.to(torch.int16) - j).clamp(min=0).to(torch.uint8)
        eff = c.view(E4).float() * s2
        x = torch.where(eff > 0, b / torch.where(eff > 0, eff, torch.ones_like(eff)), 0.0)
        hat = e2m1(x) * eff
        err = ((hat - b) ** 2).sum(dim=2, keepdim=True)
        if best is None:
            best, best_err = hat, err
        else:
            better = err < best_err
            best = torch.where(better, hat, best)
            best_err = torch.where(better, err, best_err)
    return best.reshape(w.shape)


FORMATS = {
    "fp16": lambda w: w.to(torch.float16).float(),
    "fp8_no_scale": lambda w: w.to(E4).float(),
    "fp8_per_tensor": lambda w: fp8_scaled(w, w.abs().max() / 448.0),
    "fp8_per_row": fp8_per_row,
    "int8_per_row": int8_per_row,
    "q8_0": q8_0,
    "nvfp4_max_scales": lambda w: nvfp4(w, 0),
    "nvfp4_best_of_8_scales": lambda w: nvfp4(w, 7),
}


def subnormal(x: torch.Tensor) -> float:
    nz = x != 0
    return float(((x.abs() < MIN_NORMAL) & nz).sum() / max(int(nz.sum()), 1))


def summary(errors: dict[str, float]) -> dict[str, object]:
    v = list(errors.values())
    worst = max(errors, key=errors.__getitem__)
    return {"median": statistics.median(v), "min": min(v), "max": max(v), "worst": worst}


def group(k: str, shape: list[int]) -> str:
    if len(shape) == 2:
        return "outer matrix"
    if "rope" in k:
        return "rope"
    if k.endswith(".bias"):
        return "bias"
    if ".ada" in k or k.startswith("ada"):
        return "modulation"
    return "norm" if "norm" in k else "other"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("master", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--numz-q4", type=Path, help="numz's Q4_K_M file of the same model")
    a = ap.parse_args()
    t0 = time.time()
    sd = torch.load(a.master, map_location="cpu", mmap=True, weights_only=True)
    blocks = [k for k in sd if BLOCK.match(k)]
    assert len(blocks) == 288, len(blocks)
    q4 = {}
    if a.numz_q4:
        q4 = {
            t.name.removeprefix("model.diffusion_model."): t for t in GGUFReader(a.numz_q4).tensors
        }
    err: dict[str, dict[str, float]] = {f: {} for f in FORMATS}
    pooled: dict[str, list[float]] = {f: [0.0, 0.0] for f in FORMATS}
    if q4:
        err["numz_q4_k_m"], pooled["numz_q4_k_m"] = {}, [0.0, 0.0]
    sub_plain, sub_tensor = {}, {}
    for i, k in enumerate(blocks):
        w = sd[k].float().contiguous()
        nw = float(torch.linalg.vector_norm(w, dtype=torch.float64))
        hats = {f: fn(w) for f, fn in FORMATS.items()}
        if q4:
            t = q4[k]
            d = np.asarray(dequantize(t.data, t.tensor_type), np.float32).reshape(w.shape)
            hats["numz_q4_k_m"] = torch.from_numpy(d)
        for f, hat in hats.items():
            e = float(torch.linalg.vector_norm(hat - w, dtype=torch.float64))
            err[f][k] = e / nw
            pooled[f][0] += e * e
            pooled[f][1] += nw * nw
        sub_plain[k] = subnormal(w)
        sub_tensor[k] = subnormal(w / (w.abs().max() / 448.0))
        if i % 36 == 35:
            print(f"{i + 1}/288 matrices, {time.time() - t0:.0f} s", flush=True)
    blocks_out = {
        f: {**summary(e), "pooled": (pooled[f][0] / pooled[f][1]) ** 0.5} for f, e in err.items()
    }
    plain = err["fp8_no_scale"]
    for t in (0.03, 0.05, 0.1):
        blocks_out["fp8_no_scale"][f"matrices_over_{t}"] = sum(v > t for v in plain.values())
    worst_plain = sorted(plain, key=plain.__getitem__, reverse=True)[:12]
    others: dict[str, dict[str, float | list[int]]] = {}
    for k, w in sd.items():
        if k in blocks:
            continue
        w = w.float()
        m = w.abs().max()
        others[k] = {
            "shape": list(w.shape),
            "fp8_no_scale": rel(w.to(E4).float(), w) if m > 0 else 0.0,
            "fp8_per_tensor": rel(fp8_scaled(w, m / 448.0), w) if m > 0 else 0.0,
            "subnormal_no_scale": subnormal(w),
        }
    groups: dict[str, dict[str, object]] = {}
    for k, v in others.items():
        g = groups.setdefault(
            group(k, v["shape"]), {"count": 0, "no_scale": {}, "per_tensor": {}, "subnormal": {}}
        )
        g["count"] += 1
        g["no_scale"][k] = v["fp8_no_scale"]
        g["per_tensor"][k] = v["fp8_per_tensor"]
        g["subnormal"][k] = v["subnormal_no_scale"]
    others_out = {
        g: {
            "count": d["count"],
            "fp8_no_scale": summary(d["no_scale"]),
            "fp8_per_tensor": summary(d["per_tensor"]),
            "subnormal_no_scale": summary(d["subnormal"]),
        }
        for g, d in groups.items()
    }
    result = {
        "master": str(a.master),
        "blocks": blocks_out,
        "subnormal_no_scale": summary(sub_plain),
        "subnormal_per_tensor": summary(sub_tensor),
        "worst_no_scale": [
            {
                "matrix": k,
                "no_scale": plain[k],
                "per_tensor": err["fp8_per_tensor"][k],
                "subnormal_no_scale": sub_plain[k],
            }
            for k in worst_plain
        ],
        "others": others_out,
        "worst_others_no_scale": sorted(
            ({"tensor": k, **v} for k, v in others.items()), key=lambda d: -d["fp8_no_scale"]
        )[:10],
        "per_matrix": err,
    }
    a.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_matrix"}, indent=1))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

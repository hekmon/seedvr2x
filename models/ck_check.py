# /// script
# requires-python = "==3.12.*"
# dependencies = ["torch==2.14.1", "numpy==2.5.2", "safetensors==0.8.0", "comfy-kitchen==0.2.37",
#                 "packaging"]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""Our quantized safetensors files through comfy-kitchen, on the CPU: the load test of DESIGN.md's
phase 2 for the layouts seedvr2x's runtime will run.

    CUDA_VISIBLE_DEVICES= uv run models/ck_check.py FILE.safetensors...

comfy-kitchen (Comfy-Org, Apache-2.0) at 0.2.37 (its eager backend: no GPU needed; it imports
`packaging` without declaring it). For every layer marked by `<layer>.comfy_quant`:
- the marker's JSON gives the layout: {"format": "float8_e4m3fn"} for FP8 (TensorCoreFP8Layout,
  one float32 scale), {"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}
  for INT8 with a rotation (TensorWiseINT8Layout, a scale per row), {"format": "nvfp4"} for NVFP4
  (TensorCoreNVFP4Layout: two E2M1 codes per byte, an E4M3 scale per 16 values in cuBLAS's 128x4
  tiles, and `<layer>.weight_scale_2`, one float32 scale);
- comfy-kitchen's QuantizedTensor built from the file's raw tensors, as a loader would, dequantized
  to float32: for fp8, equal bit for bit to our own reconstruction q x s; for int8, (q x s) x H per
  256 columns, H comfy-kitchen's rotation (its own inverse), equal to float32's precision, the
  rotation's sums possibly running in another order; for NVFP4, each code's E2M1 value (the even
  value's code in the high nibble) times its block's scale, un-swizzled (comfy-kitchen's
  from_blocked), times the float32 scale, equal bit for bit;
- for one layer, F.linear with a float input through comfy-kitchen's dispatch, against F.linear
  with the dequantized weight (NVFP4 multiplies in 4 bits only an input quantized to NVFP4, on a
  GPU: a float input runs on the dequantized weight).
Every other tensor must be float16. Prints a JSON report; exits 1 if a check fails.
"""

from __future__ import annotations

import json
import os
import sys
from importlib.metadata import version

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn.functional as F
from comfy_kitchen.float_utils import from_blocked
from comfy_kitchen.tensor import (
    QuantizedTensor,
    TensorCoreFP8Layout,
    TensorCoreNVFP4Layout,
    TensorWiseINT8Layout,
)
from safetensors import safe_open

# NVFP4: each 4-bit E2M1 code's value, the sign in bit 3.
E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)


def weight(f, layer: str, fmt: dict) -> tuple[QuantizedTensor, torch.Tensor]:
    """comfy-kitchen's tensor for a layer, and our own float32 reconstruction of it."""
    q, s = f.get_tensor(f"{layer}.weight"), f.get_tensor(f"{layer}.weight_scale").float()
    if fmt == {"format": "float8_e4m3fn"}:
        qt = QuantizedTensor(
            q,
            "TensorCoreFP8Layout",
            TensorCoreFP8Layout.Params(
                scale=s, orig_dtype=torch.float32, orig_shape=tuple(q.shape)
            ),
        )
        return qt, q.float() * s
    if fmt.get("format") == "int8_tensorwise" and fmt.get("convrot"):
        g = int(fmt["convrot_groupsize"])
        qt = QuantizedTensor(
            q,
            "TensorWiseINT8Layout",
            TensorWiseINT8Layout.Params(
                scale=s,
                orig_dtype=torch.float32,
                orig_shape=tuple(q.shape),
                convrot=True,
                convrot_groupsize=g,
            ),
        )
        h = hadamard(g)
        w = (q.float() * s).reshape(q.shape[0], -1, g) @ h
        return qt, w.reshape(q.shape)
    if fmt == {"format": "nvfp4"}:
        s2 = f.get_tensor(f"{layer}.weight_scale_2")
        n, k = q.shape[0], q.shape[1] * 2
        qt = QuantizedTensor(
            q,
            "TensorCoreNVFP4Layout",
            TensorCoreNVFP4Layout.Params(
                scale=s2,
                orig_dtype=torch.float32,
                orig_shape=(n, k),
                block_scale=f.get_tensor(f"{layer}.weight_scale"),
            ),
        )
        codes = torch.stack([q >> 4, q & 15], dim=-1).reshape(n, -1, 16)  # the even value high
        w = E2M1[codes.long()] * (from_blocked(s, n, k // 16) * s2).unsqueeze(2)
        return qt, w.reshape(n, k)
    raise SystemExit(f"{layer}: comfy_quant {fmt}: not a layout this check knows")


def hadamard(n: int) -> torch.Tensor:
    """H4 (x) H4 (x) ... / sqrt(n), H4 the regular 4x4 Hadamard: comfy-kitchen's rotation
    (comfy_kitchen/tensor/int8_utils.py), symmetric and its own inverse."""
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32
    )
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.kron(h, h4)
    if h.shape[0] != n:
        raise SystemExit(f"rotation size {n}: not a power of 4")
    return h / n**0.5


def check(path: str) -> dict:
    r: dict = {"file": path, "comfy_kitchen": version("comfy-kitchen"), "torch": torch.__version__}
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        layers = [k.removesuffix(".comfy_quant") for k in keys if k.endswith(".comfy_quant")]
        formats, equal, first = {}, 0, None
        for layer in layers:
            fmt = json.loads(bytes(f.get_tensor(f"{layer}.comfy_quant").tolist()).decode())
            formats[json.dumps(fmt, sort_keys=True)] = (
                formats.get(json.dumps(fmt, sort_keys=True), 0) + 1
            )
            qt, ours = weight(f, layer, fmt)
            deq = qt.dequantize()
            if fmt["format"] == "float8_e4m3fn":
                equal += bool(torch.equal(deq.float(), ours))
            elif fmt["format"] == "nvfp4":  # bit for bit, the zeros' signs too
                equal += bool(torch.equal(deq.view(torch.int32), ours.view(torch.int32)))
            else:  # the rotation's sums may run in another order: equal to float32's precision
                equal += float((deq.float() - ours).abs().max() / ours.abs().max()) < 1e-6
            if first is None:
                first = layer
                b = f.get_tensor(f"{layer}.bias").float() if f"{layer}.bias" in keys else None
                x = torch.randn(4, ours.shape[1], generator=torch.Generator().manual_seed(0))
                y = F.linear(x, qt, b)
                y_ref = F.linear(x, deq.float(), b)
                r["linear"] = {
                    "layer": layer,
                    "rel_diff": float((y.float() - y_ref).norm() / y_ref.norm()),
                }
        quantized = {
            f"{layer}.{s}"
            for layer in layers
            for s in ("weight", "weight_scale", "weight_scale_2", "comfy_quant")
        }
        others = {f.get_tensor(k).dtype for k in keys if k not in quantized}
    r.update(
        {
            "layers": len(layers),
            "formats": formats,
            "dequantize_equal_ours": equal,
            "other_dtypes": sorted(str(d) for d in others),
        }
    )
    # fp8 and NVFP4 run F.linear on the dequantized weight: equal; int8 multiplies quantized
    # activations (per token, after the rotation): close, not equal.
    exact = ("float8_e4m3fn", "nvfp4")
    tol = 0.0 if all(json.loads(k)["format"] in exact for k in formats) else 0.05
    r["pass"] = (
        len(layers) == 288
        and equal == 288
        and r["other_dtypes"] == ["torch.float16"]
        and r["linear"]["rel_diff"] <= tol
    )
    return r


results = [check(p) for p in sys.argv[1:]]
print(json.dumps(results, indent=1))
sys.exit(0 if all(r["pass"] for r in results) else 1)

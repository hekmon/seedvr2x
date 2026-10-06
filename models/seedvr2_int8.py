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
"""SeedVR2's INT8 files for seedvr2x, from ByteDance's fp32 masters, rotated: int8 weights and
activations (W8A8), the fast path of the RTX 20 and 30 generations, where fp8 can't multiply.

The 288 attention and MLP matrices of the blocks (99% of the weights) are stored in comfy-kitchen's
INT8 layout with its rotation ("convrot", Comfy-Org, Apache-2.0, 0.2.37): each 256 columns of W are
multiplied by H, the regular Hadamard H4 (x) H4 (x) H4 (x) H4 / 16 (symmetric, its own inverse:
comfy_kitchen/tensor/int8_utils.py), which spreads each row's outliers; then each row gets one
float32 scale, s = max|row| / 127, and its values round(x / s), to nearest, ties to even, clamped
to [-128, 127]. `<layer>.weight` (int8), `<layer>.weight_scale` (float32 [rows, 1]) and the marker
`<layer>.comfy_quant`, {"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256},
as Comfy-Org's SeedVR2 files mark theirs. At run time comfy-kitchen rotates each input the same way
before quantizing it to int8 per token, so (x H)(W H)^T = x W^T needs no un-rotation. Every other
tensor is our fp16 file's: the master rounded to the nearest float16.

    uv run models/seedvr2_int8.py [--masters DIR]... [--cache DIR] [--fp16 DIR] [--out DIR]
                                  [FILE...]

- The checks: each matrix's error against the master once un-rotated (relative, ||W_hat - W|| /
  ||W||), median and worst; our codes and scales against comfy-kitchen's own quantizer on the CPU
  (torch.ops.comfy_kitchen.quantize_int8_convrot_weight), share equal; every tensor read back with
  the safetensors library; with --fp16, every 16-bit tensor byte for byte our fp16 file's; the
  output's SHA-256 pinned in OUTPUTS. A file failing a check is deleted.
- The output, in --out (default models/dist): not uploaded before GPU runs validate it (DESIGN.md,
  Weights).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from importlib.metadata import version
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import comfy_kitchen  # noqa: F401 (registers torch.ops.comfy_kitchen)
import numpy as np
import safetensors
import torch
from common import (
    CACHE,
    DIST,
    Entry,
    check_output,
    find_or_fetch,
    log,
    read_header,
    write_safetensors,
)
from safetensors import safe_open
from seedvr2_fp8 import BLOCK
from seedvr2_fp16 import COPYRIGHT, FILES, REPO, REVISION, hf_url, load_master

GROUP = 256
MARKER = json.dumps(
    {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP}
).encode()
SOURCES = {  # our int8 file: the fp16 file of ours it shares its master and its 16-bit tensors with
    "seedvr2x_ema_7b_int8_convrot.safetensors": "seedvr2x_ema_7b_fp16.safetensors",
    "seedvr2x_ema_7b_sharp_int8_convrot.safetensors": "seedvr2x_ema_7b_sharp_fp16.safetensors",
}

# The outputs of the pinned inputs and versions: a run must give these bytes.
OUTPUTS = {
    "seedvr2x_ema_7b_int8_convrot.safetensors": (
        "7eb2c7841b1c480d3302d83c8583d345cb9bb80dc169ec83e9a72f3585ab33da"
    ),
    "seedvr2x_ema_7b_sharp_int8_convrot.safetensors": (
        "d83aeaa4687b43755658d85eadcf2e28bb10b70c0196798b303171cf9a16de43"
    ),
}


def hadamard(n: int) -> torch.Tensor:
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float64
    )
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.kron(h, h4)
    if h.shape[0] != n:
        raise SystemExit(f"rotation size {n}: not a power of 4")
    return h / n**0.5


H = hadamard(GROUP)


def rotate(w: torch.Tensor) -> torch.Tensor:
    """Each GROUP columns of a float32 matrix times H, computed in float64, rounded once to
    float32."""
    rows, cols = w.shape
    if cols % GROUP:
        raise SystemExit(f"{cols} columns: not a multiple of {GROUP}")
    return (w.double().reshape(rows, -1, GROUP) @ H).reshape(rows, cols).float()


def quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(int8 codes, float32 scales [rows, 1]) of the rotated matrix: s = max|row| / 127, codes
    round(x / s) to nearest, ties to even, clamped to [-128, 127]."""
    r = rotate(w)
    s = (r.abs().amax(dim=1, keepdim=True) / 127.0).to(torch.float32)
    s = torch.where(s == 0, torch.ones_like(s), s)
    return torch.round(r / s).clamp(-128, 127).to(torch.int8), s


def dequantize(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """The weight back: (q x s) x H per GROUP columns (H its own inverse)."""
    rows, cols = q.shape
    x = (q.double() * s.double()).reshape(rows, -1, GROUP) @ H
    return x.reshape(rows, cols).float()


def metadata(fp16_name: str) -> dict[str, str]:
    f = FILES[fp16_name]
    return {
        "format": "pt",
        "license": "apache-2.0",
        "copyright": COPYRIGHT,
        "source": f"{f.master} of {REPO}, revision {REVISION}",
        "source_url": hf_url(REPO, REVISION, f.master),
        "source_sha256": f.master_sha256,
        "change": "the 288 attention and MLP matrices of the blocks in int8 with comfy-kitchen's "
        "rotation (each 256 columns times the regular Hadamard / 16) and one float32 scale per row "
        "(max|row| / 127), rounded to nearest, ties to even (comfy-kitchen's INT8 layout, "
        "<layer>.weight and <layer>.weight_scale, marked by <layer>.comfy_quant); every other "
        "tensor rounded to the nearest float16",
        "conversion": "unofficial, not ByteDance's: made by seedvr2x's models/seedvr2_int8.py",
    }


def build(name: str, a: argparse.Namespace) -> None:
    t0 = time.time()
    fp16_name = SOURCES[name]
    f = FILES[fp16_name]
    master = find_or_fetch(
        f.master,
        hf_url(REPO, REVISION, f.master),
        f.master_sha256,
        f.master_size,
        a.masters,
        a.cache,
    )
    sd = load_master(master)
    blocks = [k for k in sd if BLOCK.match(k)]
    if len(blocks) != 288:
        raise SystemExit(f"{master}: {len(blocks)} block matrices, 288 expected")
    done: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    errors: dict[str, float] = {}
    ck_codes, ck_scales, n_codes = 0, 0, 0

    def quantized(k: str) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal ck_codes, ck_scales, n_codes
        if k not in done:
            w = sd[k].contiguous()
            q, s = quantize(w)
            hat = dequantize(q, s)
            errors[k] = float(
                torch.linalg.vector_norm(hat - w, dtype=torch.float64)
                / torch.linalg.vector_norm(w, dtype=torch.float64)
            )
            cq, cs = torch.ops.comfy_kitchen.quantize_int8_convrot_weight(w, GROUP)
            ck_codes += int((cq == q).sum())
            ck_scales += int((cs.reshape(s.shape) == s).sum())
            n_codes += q.numel()
            done[k] = (q, s)  # the writer takes every scale before any weight: keep them all
        return done[k]

    def fp16(k: str) -> memoryview:
        return memoryview(sd[k].contiguous().to(torch.float16).reshape(-1).numpy().view(np.uint8))

    entries = []
    for k, v in sd.items():
        if k in blocks:
            layer = k.removesuffix(".weight")
            rows = v.shape[0]
            entries += [
                Entry(
                    k,
                    "I8",
                    tuple(v.shape),
                    lambda k=k: memoryview(quantized(k)[0].reshape(-1).numpy().view(np.uint8)),
                ),
                Entry(
                    f"{layer}.weight_scale",
                    "F32",
                    (rows, 1),
                    lambda k=k: memoryview(quantized(k)[1].reshape(-1).numpy().view(np.uint8)),
                ),
                Entry(f"{layer}.comfy_quant", "U8", (len(MARKER),), lambda: memoryview(MARKER)),
            ]
        else:
            entries.append(Entry(k, "F16", tuple(v.shape), lambda k=k: fp16(k)))
    md = metadata(fp16_name)
    out = a.out / name
    try:
        write_safetensors(out, entries, md)
        e = list(errors.values())
        worst = max(errors, key=errors.__getitem__)
        log(
            f"{name}: 288 matrices in int8 with the rotation, error against the master: median "
            f"{statistics.median(e):.5f}, worst {max(e):.5f} ({worst}); comfy-kitchen's quantizer "
            f"gives the same codes on {ck_codes / n_codes:.6%} of values; "
            f"{len(sd) - 288} tensors in fp16; {out.stat().st_size} bytes"
        )
        with safe_open(out, framework="pt") as r:
            if r.metadata() != md:
                raise SystemExit(f"{out}: metadata read back differs")
            for k in blocks:
                q, s = done[k]
                layer = k.removesuffix(".weight")
                if not (
                    torch.equal(r.get_tensor(k), q)
                    and torch.equal(r.get_tensor(f"{layer}.weight_scale"), s)
                    and bytes(r.get_tensor(f"{layer}.comfy_quant").numpy()) == MARKER
                ):
                    raise SystemExit(f"{out}: {k} read back differs")
        log(f"{name}: read back by safetensors {safetensors.__version__}: every int8 tensor equal")
        if a.fp16:
            ref = a.fp16 / fp16_name
            hr, sr = read_header(ref)
            ho, so = read_header(out)
            mr, mo = np.memmap(ref, np.uint8, "r"), np.memmap(out, np.uint8, "r")
            n = 0
            for k in sd:
                if k in blocks:
                    continue
                x = mo[so + ho[k]["data_offsets"][0] : so + ho[k]["data_offsets"][1]]
                y = mr[sr + hr[k]["data_offsets"][0] : sr + hr[k]["data_offsets"][1]]
                if ho[k]["dtype"] != "F16" or not np.array_equal(x, y):
                    raise SystemExit(f"{out}: {k} differs from {ref}")
                n += 1
            log(f"{name}: its {n} fp16 tensors are {fp16_name}'s, byte for byte")
        check_output(out, OUTPUTS.get(name))
    except BaseException:
        out.unlink(missing_ok=True)
        raise
    log(f"{name}: done in {time.time() - t0:.0f} s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "files", nargs="*", metavar="FILE", help=f"of {', '.join(SOURCES)} (default: all)"
    )
    ap.add_argument("--masters", type=Path, action="append", default=[])
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--fp16", type=Path, help="a directory holding our fp16 files, to compare with")
    ap.add_argument("--out", type=Path, default=DIST)
    a = ap.parse_args()
    for name in a.files:
        if name not in SOURCES:
            ap.error(f"{name}: not one of {', '.join(SOURCES)}")
    log(
        f"torch {torch.__version__}, comfy-kitchen {version('comfy-kitchen')}, "
        f"safetensors {safetensors.__version__}, {torch.get_num_threads()} threads"
    )
    for name in a.files or SOURCES:
        build(name, a)


if __name__ == "__main__":
    main()

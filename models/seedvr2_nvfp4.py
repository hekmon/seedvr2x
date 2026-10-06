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
"""SeedVR2's NVFP4 files for seedvr2x, from ByteDance's fp32 masters, each block's scale searched:
4-bit weights, multiplied with 4-bit activations on Blackwell (W4A4, sm_100 and sm_120), widened
to 16 bits before (W4A16).

The 288 attention and MLP matrices of the blocks (99% of the weights) are stored in comfy-kitchen's
NVFP4 layout (TensorCoreNVFP4Layout, Comfy-Org, Apache-2.0, 0.2.37): each weight an E2M1 value
(0, 0.5, 1, 1.5, 2, 3, 4 or 6, signed) times the E4M3 scale of its block, 16 values along the
input, times one float32 scale per tensor, s2 = max|W| / (448 x 6). comfy-kitchen sets a block's
scale from the block's max: max / 6 / s2, to the nearest E4M3, clamped to 448. Here it is the one,
of that scale and the 7 E4M3 values below it (never below 0), that gives the block the smallest
squared error, its values W / (scale x s2) rounded to the nearest E2M1 (ties to the even code,
saturated at +-6): formats_study.py's search. comfy-kitchen decodes any block scale, so it loads
the file as it is: `<layer>.weight` (uint8 [N, K/2], two codes per byte, the even value's in the
high nibble, the sign in bit 3), `<layer>.weight_scale` (E4M3 [N, K/16], in cuBLAS's 128x4 tiles),
`<layer>.weight_scale_2` (float32, s2) and the marker `<layer>.comfy_quant`, {"format": "nvfp4"},
as Comfy-Org's SeedVR2 files mark theirs. Every other tensor is our fp16 file's: the master rounded
to the nearest float16.

    uv run models/seedvr2_nvfp4.py [--masters DIR]... [--cache DIR] [--fp16 DIR] [--out DIR]
                                   [FILE...]

- The checks: each matrix's error against the master (relative, ||W_hat - W|| / ||W||), median and
  worst, beside that of comfy-kitchen's own quantizer (TensorCoreNVFP4Layout.quantize, the max's
  scales), whose codes and scales are counted against the search's first candidate; every tensor
  read back with the safetensors library, and comfy-kitchen's QuantizedTensor, built from the
  file's tensors as a loader would, dequantized bit for bit to our reconstruction; with --fp16,
  every 16-bit tensor byte for byte our fp16 file's; the output's SHA-256 pinned in OUTPUTS. A file
  failing a check is deleted.
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

import numpy as np
import safetensors
import torch
from comfy_kitchen.float_utils import from_blocked, pack_uint4, to_blocked, unpack_uint4
from comfy_kitchen.tensor import QuantizedTensor, TensorCoreNVFP4Layout
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

E4 = torch.float8_e4m3fn
E4M3_MAX, E2M1_MAX = 448.0, 6.0
GROUP = 16  # the values sharing a block scale, along the input
BELOW = 7  # the E4M3 codes tried under the max's
# Each E2M1 code's value: the magnitude in bits 0-2, the sign in bit 3 (8 is -0).
E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)
# The midpoints between neighbouring magnitudes, and whether a tie goes up there (to the even code).
MIDPOINTS = (
    (0.25, False),
    (0.75, True),
    (1.25, False),
    (1.75, True),
    (2.5, False),
    (3.5, True),
    (5.0, False),
)
MARKER = json.dumps({"format": "nvfp4"}).encode()
SOURCES = {  # our NVFP4 file: the fp16 file of ours it shares its master and 16-bit tensors with
    "seedvr2x_ema_7b_nvfp4.safetensors": "seedvr2x_ema_7b_fp16.safetensors",
    "seedvr2x_ema_7b_sharp_nvfp4.safetensors": "seedvr2x_ema_7b_sharp_fp16.safetensors",
}

# The outputs of the pinned inputs and versions: a run must give these bytes.
OUTPUTS = {
    "seedvr2x_ema_7b_nvfp4.safetensors": (
        "9cd143590bcced4ff1261fd77df4e61117c5c9a83be2fa4f3ca8340abf4828d5"
    ),
    "seedvr2x_ema_7b_sharp_nvfp4.safetensors": (
        "d0a1d5a40ae87e2428b17659fabcd0676b48b171dd48b6cb5ba072cd4a3cc9cb"
    ),
}


def e2m1(x: torch.Tensor) -> torch.Tensor:
    """The E2M1 codes of a float32 tensor, uint8: the nearest of 0, 0.5, 1, 1.5, 2, 3, 4, 6 (codes
    0 to 7), a tie to the even code, saturated at 6; the sign in bit 3."""
    a = x.abs()
    q = torch.zeros(a.shape, dtype=torch.uint8)
    for m, up in MIDPOINTS:
        q += (a >= m) if up else (a > m)
    return q | (torch.signbit(x).to(torch.uint8) << 3)


def encode(w: torch.Tensor, c: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
    """The E2M1 codes [N, K] of a float32 matrix whose blocks' scales are the E4M3 codes c
    [N, K/16] (uint8): W / (scale x s2), 0 in a block whose scale is 0."""
    eff = (c.view(E4).float() * s2).unsqueeze(2)
    b = w.reshape(*c.shape, GROUP)
    x = torch.where(eff > 0, b / torch.where(eff > 0, eff, 1.0), 0.0)
    return e2m1(x).reshape(w.shape)


def decode(q: torch.Tensor, c: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
    """The float32 matrix of E2M1 codes q [N, K] and block scales c [N, K/16]: each code's value
    times (its block's scale x s2), in float32, in comfy-kitchen's order."""
    eff = (c.view(E4).float() * s2).unsqueeze(2)
    return (E2M1[q.int()].reshape(*c.shape, GROUP) * eff).reshape(q.shape)


def quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(E2M1 codes [N, K], block scales [N, K/16] as E4M3 codes in uint8, s2, the max's scales) of
    a float32 matrix: s2 = max|W| / (448 x 6); each block's scale, of the max's code (max / 6 / s2,
    clamped to 448) and the BELOW codes under it, the one giving the block the smallest squared
    error, the first of equals."""
    n, k = w.shape
    if n % 128 or k % (4 * GROUP):
        raise SystemExit(f"{n}x{k}: not whole 128x4 tiles of block scales")
    s2 = w.abs().max() / (E4M3_MAX * E2M1_MAX)
    top = (w.reshape(n, -1, GROUP).abs().amax(dim=2) / E2M1_MAX / s2).clamp(max=E4M3_MAX)
    top = top.to(E4).view(torch.uint8)
    best, best_err = top, None
    for j in range(BELOW + 1):
        c = (top.to(torch.int16) - j).clamp(min=0).to(torch.uint8)
        err = ((decode(encode(w, c, s2), c, s2) - w).reshape(n, -1, GROUP) ** 2).sum(dim=2)
        if best_err is None:
            best_err = err
        else:
            better = err < best_err
            best = torch.where(better, c, best)
            best_err = torch.where(better, err, best_err)
    return encode(w, best, s2), best, s2, top


def swizzled(c: torch.Tensor) -> torch.Tensor:
    """Block scales [N, K/16], E4M3 codes in uint8, as comfy-kitchen stores them: E4M3, in
    cuBLAS's 128x4 tiles."""
    return to_blocked(c.view(E4), flatten=False)


def rel(hat: torch.Tensor, w: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(hat - w, dtype=torch.float64)
        / torch.linalg.vector_norm(w, dtype=torch.float64)
    )


def metadata(fp16_name: str) -> dict[str, str]:
    f = FILES[fp16_name]
    return {
        "format": "pt",
        "license": "apache-2.0",
        "copyright": COPYRIGHT,
        "source": f"{f.master} of {REPO}, revision {REVISION}",
        "source_url": hf_url(REPO, REVISION, f.master),
        "source_sha256": f.master_sha256,
        "change": "the 288 attention and MLP matrices of the blocks in NVFP4: E2M1 values, an E4M3 "
        "scale per 16 values along the input and a float32 scale per tensor (max|W| / (448 x 6)), "
        "each block's scale the one, of its max's (max / 6, to the nearest E4M3) and the 7 E4M3 "
        "values below it, that minimises the block's squared error, values rounded to nearest, "
        "ties to even (comfy-kitchen's NVFP4 layout, <layer>.weight, <layer>.weight_scale and "
        "<layer>.weight_scale_2, marked by <layer>.comfy_quant); every other tensor rounded to the "
        "nearest float16",
        "conversion": "unofficial, not ByteDance's: made by seedvr2x's models/seedvr2_nvfp4.py",
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
    done: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    errors: dict[str, float] = {}
    ck_errors: dict[str, float] = {}
    ck_codes, ck_scales, n_codes, n_scales = 0, 0, 0, 0

    def quantized(k: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nonlocal ck_codes, ck_scales, n_codes, n_scales
        if k not in done:
            w = sd[k].contiguous()
            q, c, s2, top = quantize(w)
            errors[k] = rel(decode(q, c, s2), w)
            # comfy-kitchen's own quantizer, each block's scale from its max: our first candidate
            cq, cp = TensorCoreNVFP4Layout.quantize(w)
            if not torch.equal(cp.scale, s2):
                raise SystemExit(f"{k}: comfy-kitchen's s2 {float(cp.scale)}, ours {float(s2)}")
            ck_codes += int((unpack_uint4(cq) == encode(w, top, s2)).sum())
            ck_scales += int(
                (from_blocked(cp.block_scale, *top.shape).view(torch.uint8) == top).sum()
            )
            n_codes += q.numel()
            n_scales += c.numel()
            ck_errors[k] = rel(QuantizedTensor(cq, "TensorCoreNVFP4Layout", cp).dequantize(), w)
            done[k] = (q, c, s2)  # the writer takes every s2 before any weight: keep them all
            if len(done) % 36 == 0:
                log(f"{name}: {len(done)}/288 matrices, {time.time() - t0:.0f} s")
        return done[k]

    def fp16(k: str) -> memoryview:
        return memoryview(sd[k].contiguous().to(torch.float16).reshape(-1).numpy().view(np.uint8))

    entries = []
    for k, v in sd.items():
        if k in blocks:
            layer = k.removesuffix(".weight")
            rows, cols = v.shape
            entries += [
                Entry(
                    k,
                    "U8",
                    (rows, cols // 2),
                    lambda k=k: memoryview(pack_uint4(quantized(k)[0]).reshape(-1).numpy()),
                ),
                Entry(
                    f"{layer}.weight_scale",
                    "F8_E4M3",
                    (rows, cols // GROUP),
                    lambda k=k: memoryview(
                        swizzled(quantized(k)[1]).view(torch.uint8).reshape(-1).numpy()
                    ),
                ),
                Entry(
                    f"{layer}.weight_scale_2",
                    "F32",
                    (),
                    lambda k=k: memoryview(quantized(k)[2].reshape(1).numpy().view(np.uint8)),
                ),
                Entry(f"{layer}.comfy_quant", "U8", (len(MARKER),), lambda: memoryview(MARKER)),
            ]
        else:
            entries.append(Entry(k, "F16", tuple(v.shape), lambda k=k: fp16(k)))
    md = metadata(fp16_name)
    out = a.out / name
    try:
        write_safetensors(out, entries, md)
        e, ck = list(errors.values()), list(ck_errors.values())
        worst = max(errors, key=errors.__getitem__)
        lower = sum(errors[k] < ck_errors[k] for k in errors)
        log(
            f"{name}: 288 matrices in NVFP4, each block's scale the best of {BELOW + 1}, error "
            f"against the master: median {statistics.median(e):.5f}, worst {max(e):.5f} "
            f"({worst}); comfy-kitchen's own quantizer: median {statistics.median(ck):.5f}, worst "
            f"{max(ck):.5f}, ours lower on {lower} of 288, its codes other than our first "
            f"candidate's on {n_codes - ck_codes} of {n_codes} values, its block scales on "
            f"{n_scales - ck_scales} of {n_scales}; {len(sd) - 288} tensors in fp16; "
            f"{out.stat().st_size} bytes"
        )
        with safe_open(out, framework="pt") as r:
            if r.metadata() != md:
                raise SystemExit(f"{out}: metadata read back differs")
            for k in blocks:
                q, c, s2 = done[k]
                layer = k.removesuffix(".weight")
                weight = r.get_tensor(k)
                scale = r.get_tensor(f"{layer}.weight_scale")
                scale_2 = r.get_tensor(f"{layer}.weight_scale_2")
                if not (
                    torch.equal(weight, pack_uint4(q))
                    and torch.equal(scale.view(torch.uint8), swizzled(c).view(torch.uint8))
                    and torch.equal(scale_2, s2)
                    and bytes(r.get_tensor(f"{layer}.comfy_quant").numpy()) == MARKER
                ):
                    raise SystemExit(f"{out}: {k} read back differs")
                qt = QuantizedTensor(
                    weight,
                    "TensorCoreNVFP4Layout",
                    TensorCoreNVFP4Layout.Params(
                        scale=scale_2,
                        orig_dtype=torch.float32,
                        orig_shape=tuple(q.shape),
                        block_scale=scale,
                    ),
                )
                ours = decode(q, c, s2)
                if not torch.equal(qt.dequantize().view(torch.int32), ours.view(torch.int32)):
                    raise SystemExit(f"{out}: {k}: comfy-kitchen decodes it otherwise")
        log(
            f"{name}: read back by safetensors {safetensors.__version__}: every NVFP4 tensor "
            "equal; comfy-kitchen decodes all 288 from them bit for bit as our reconstruction"
        )
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

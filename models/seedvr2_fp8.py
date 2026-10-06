# /// script
# requires-python = "==3.12.*"
# dependencies = ["torch==2.14.1", "numpy==2.5.2", "safetensors==0.8.0"]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""SeedVR2's fp8 files for seedvr2x, from ByteDance's fp32 masters, with one scale per tensor.

The 288 attention and MLP matrices of the blocks (99% of the weights) are stored in
float8_e4m3fn with one float32 scale each, in comfy-kitchen's FP8 layout: `<layer>.weight`, the
values W x (1 / s) rounded to the nearest E4M3 (ties to even, clamped to +-448), and
`<layer>.weight_scale`, s = max|W| / 448, so that the weight is weight x weight_scale. Each such
layer is marked by `<layer>.comfy_quant`, the UTF-8 JSON {"format": "float8_e4m3fn"} as uint8, as
Comfy-Org's SeedVR2 files mark theirs. Every other tensor is our fp16 file's: the master rounded
to the nearest float16. Why one scale per tensor, and what a cast without scale loses:
models/FORMATS.md.

    uv run models/seedvr2_fp8.py [--masters DIR]... [--cache DIR] [--fp16 DIR] [--out DIR]
                                 [FILE...]

- The masters: as seedvr2_fp16.py's (ByteDance-Seed/SeedVR2-7B at its REVISION), read where a
  --masters directory has them, else downloaded into --cache.
- The checks: each fp8 matrix's error against the master (relative, ||W_hat - W|| / ||W||), its
  median and worst printed; every tensor read back with the safetensors library; with --fp16,
  every 16-bit tensor byte for byte our fp16 file's; the output's SHA-256 pinned in OUTPUTS. A file
  failing a check is deleted.
- The output, in --out (default models/dist): not uploaded before GPU runs validate it (DESIGN.md,
  Weights).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

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
from seedvr2_fp16 import COPYRIGHT, FILES, REPO, REVISION, hf_url, load_master

BLOCK = re.compile(
    r"^blocks\.\d+\.(attn\.proj_(qkv|out)\.(txt|vid)|mlp\.(txt|vid)\.proj_(in|out))\.weight$"
)
E4M3_MAX = 448.0
MARKER = json.dumps({"format": "float8_e4m3fn"}).encode()
SOURCES = {  # our fp8 file: the fp16 file of ours it shares its master and its 16-bit tensors with
    "seedvr2x_ema_7b_fp8_scaled.safetensors": "seedvr2x_ema_7b_fp16.safetensors",
    "seedvr2x_ema_7b_sharp_fp8_scaled.safetensors": "seedvr2x_ema_7b_sharp_fp16.safetensors",
}

# The outputs of the pinned inputs and versions: a run must give these bytes.
OUTPUTS = {
    "seedvr2x_ema_7b_fp8_scaled.safetensors": (
        "3222ce3ea311c82ed152ac818888fa7f7436920a2e9940eb062a382f95c5a6ef"
    ),
    "seedvr2x_ema_7b_sharp_fp8_scaled.safetensors": (
        "6cdf191b8f63a9b74795060c1b505a4c9f3ac10619f36ccd00cfdb5a4c3e130d"
    ),
}


def quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(E4M3 values, float32 scale) of a float32 matrix: s = max|W| / 448, values W x (1 / s)
    rounded to the nearest E4M3, ties to even, clamped to +-448."""
    s = (w.abs().max() / E4M3_MAX).to(torch.float32)
    q = (w * (1.0 / s)).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return q, s


def metadata(fp16_name: str) -> dict[str, str]:
    f = FILES[fp16_name]
    return {
        "format": "pt",
        "license": "apache-2.0",
        "copyright": COPYRIGHT,
        "source": f"{f.master} of {REPO}, revision {REVISION}",
        "source_url": hf_url(REPO, REVISION, f.master),
        "source_sha256": f.master_sha256,
        "change": "the 288 attention and MLP matrices of the blocks in float8_e4m3fn with one "
        "float32 scale per tensor (max|W| / 448; comfy-kitchen's FP8 layout, <layer>.weight and "
        "<layer>.weight_scale, marked by <layer>.comfy_quant), rounded to nearest, ties to even; "
        "every other tensor rounded to the nearest float16",
        "conversion": "unofficial, not ByteDance's: made by seedvr2x's models/seedvr2_fp8.py",
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
    errors: dict[str, float] = {}

    def fp8(k: str) -> memoryview:
        w = sd[k].contiguous()
        q, s = quantize(w)
        errors[k] = float(
            torch.linalg.vector_norm(q.float() * s - w, dtype=torch.float64)
            / torch.linalg.vector_norm(w, dtype=torch.float64)
        )
        return memoryview(q.reshape(-1).view(torch.uint8).numpy())

    def scale(k: str) -> memoryview:
        return memoryview(quantize(sd[k])[1].reshape(1).numpy().view(np.uint8))

    def fp16(k: str) -> memoryview:
        return memoryview(sd[k].contiguous().to(torch.float16).reshape(-1).numpy().view(np.uint8))

    entries = []
    for k, v in sd.items():
        if k in blocks:
            layer = k.removesuffix(".weight")
            entries += [
                Entry(k, "F8_E4M3", tuple(v.shape), lambda k=k: fp8(k)),
                Entry(f"{layer}.weight_scale", "F32", (), lambda k=k: scale(k)),
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
            f"{name}: 288 matrices in fp8, error against the master: median "
            f"{statistics.median(e):.5f}, worst {max(e):.5f} ({worst}); "
            f"{len(sd) - 288} tensors in fp16; {out.stat().st_size} bytes"
        )
        with safe_open(out, framework="pt") as r:
            if r.metadata() != md:
                raise SystemExit(f"{out}: metadata read back differs")
            for k in blocks:
                q, s = quantize(sd[k])
                layer = k.removesuffix(".weight")
                if not (
                    torch.equal(r.get_tensor(k).view(torch.uint8), q.view(torch.uint8))
                    and torch.equal(r.get_tensor(f"{layer}.weight_scale"), s)
                    and bytes(r.get_tensor(f"{layer}.comfy_quant").numpy()) == MARKER
                ):
                    raise SystemExit(f"{out}: {k} read back differs")
        log(f"{name}: read back by safetensors {safetensors.__version__}: every fp8 tensor equal")
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
        f"torch {torch.__version__}, numpy {np.__version__}, "
        f"safetensors {safetensors.__version__}, {torch.get_num_threads()} threads"
    )
    for name in a.files or SOURCES:
        build(name, a)


if __name__ == "__main__":
    main()

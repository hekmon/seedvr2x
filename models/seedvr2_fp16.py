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
"""SeedVR2's fp16 files for seedvr2x, rounded from ByteDance's fp32 masters.

Every tensor of a master is rounded to the nearest float16, ties to even, under the same name and
shape; nothing else changes. numz's fp16 files hold the same values (DESIGN.md, Weights), and the
check holds ours equal to them, element for element.

    uv run models/seedvr2_fp16.py [--masters DIR]... [--reference DIR]... [--cache DIR]
                                  [--out DIR] [--no-reference] [FILE...]

- The masters: ByteDance-Seed/SeedVR2-7B on Hugging Face at REVISION, each checked by its SHA-256
  (its LFS object id). Read where a --masters directory has them, never moved; else downloaded
  into --cache (default ~/.cache/seedvr2x-models).
- The rounding: torch's float32 to float16 conversion, checked bit for bit against numpy's, a
  separate implementation. A value that would overflow float16 is refused.
- The check: numz's fp16 file with the same values (numz/SeedVR2_comfyUI at NUMZ_REVISION,
  checked by SHA-256), found in a --reference directory or downloaded into --cache: the same
  tensor names, dtypes and shapes, and every tensor's bytes equal. --no-reference skips it.
- The output, in --out (default models/dist): read back with the safetensors library and compared
  with the rounding, bit for bit; its SHA-256 must be the one pinned in OUTPUTS. A file failing a
  check is deleted.

The files are named seedvr2x_*, not as numz's: numz's downloader deletes a file named like one of
its own whose SHA-256 differs (src/utils/downloads.py:216-235 at 4490bd1), and ours differ from
numz's by their header, which holds the metadata.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

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

REPO = "ByteDance-Seed/SeedVR2-7B"
REVISION = "eb0c4281d41ba3767d4f14370f0e37e9e9180c16"  # main since 2025-07-14, "add sharp ver."
NUMZ_REPO = "numz/SeedVR2_comfyUI"
NUMZ_REVISION = "09ced71023636e9bc8cdf9cdecfb2625d1e691e8"  # main since 2025-11-09
COPYRIGHT = "SeedVR2: Copyright (c) 2025 Bytedance Ltd. and/or its affiliates"


class File(NamedTuple):
    master: str  # ByteDance's file
    master_size: int
    master_sha256: str
    numz: str  # numz's fp16 file, the same values
    numz_size: int
    numz_sha256: str  # numz's registry (src/utils/model_registry.py at 4490bd1), its LFS object id


FILES = {
    "seedvr2x_ema_7b_fp16.safetensors": File(
        "seedvr2_ema_7b.pth",
        32958774606,
        "e1b2ae25505607e61f2a7dc7967ba778aaf3e3626d9969ce6e24c52d9ddebfcd",
        "seedvr2_ema_7b_fp16.safetensors",
        16479334424,
        "7b8241aa957606ab6cfb66edabc96d43234f9819c5392b44d2492d9f0b0bbe4a",
    ),
    "seedvr2x_ema_7b_sharp_fp16.safetensors": File(
        "seedvr2_ema_7b_sharp.pth",
        32958774606,
        "ced5706c976d5879efcab9e108349d67abcbd8a9b36a1f48bf0f19c24164a264",
        "seedvr2_ema_7b_sharp_fp16.safetensors",
        16479334424,
        "20a93e01ff24beaeebc5de4e4e5be924359606c356c9c51509fba245bd2d77dd",
    ),
    "seedvr2x_ema_vae_fp16.safetensors": File(
        "ema_vae.pth",
        1002691902,
        "c7df8a67e68b7f9aca3d5d2153d2ce8ab4373687741a0f9ce87cb356ace51cac",
        "ema_vae_fp16.safetensors",
        501324814,
        "20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1",
    ),
}

# The outputs of the pinned inputs and versions: a run must give these bytes.
OUTPUTS = {
    "seedvr2x_ema_7b_fp16.safetensors": (
        "071cab5e5ef7a4471e1df0023c26cc16deeb14e58f8ad5c9196d2a08f96da5f2"
    ),
    "seedvr2x_ema_7b_sharp_fp16.safetensors": (
        "5eb47fdee4b620765573a697b6f82442234e7dc7aa0beeb3ba72fc63b917a817"
    ),
    "seedvr2x_ema_vae_fp16.safetensors": (
        "b9c6ebf0b14107be595825f476b9f89029a067d5b13e9db39c1351a608265468"
    ),
}


def hf_url(repo: str, revision: str, name: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{name}"


def metadata(f: File) -> dict[str, str]:
    return {
        "format": "pt",
        "license": "apache-2.0",
        "copyright": COPYRIGHT,
        "source": f"{f.master} of {REPO}, revision {REVISION}",
        "source_url": hf_url(REPO, REVISION, f.master),
        "source_sha256": f.master_sha256,
        "change": "every tensor rounded from float32 to the nearest float16, ties to even; "
        "names and shapes unchanged",
        "conversion": "unofficial, not ByteDance's: made by seedvr2x's models/seedvr2_fp16.py",
    }


@dataclass
class Rounding:
    tensors: int = 0
    elements: int = 0
    exact: int = 0  # values float16 holds exactly
    subnormal: int = 0  # float16 subnormals
    to_zero: int = 0  # nonzero values rounded to zero

    def round(self, t: torch.Tensor) -> np.ndarray:
        """t rounded to float16 by torch, checked bit for bit against numpy; finite."""
        t = t.contiguous()
        h = t.to(torch.float16)
        n = t.numpy().astype(np.float16)
        if not np.array_equal(h.numpy().view(np.uint16), n.view(np.uint16)):
            raise SystemExit("torch and numpy round a tensor differently")
        if not torch.isfinite(h).all():
            raise SystemExit("a value overflows float16")
        self.tensors += 1
        self.elements += t.numel()
        self.exact += int((h.float() == t).sum())
        self.subnormal += int(((h != 0) & (h.abs() < 2.0**-14)).sum())
        self.to_zero += int(((h == 0) & (t != 0)).sum())
        return n


def load_master(path: Path) -> dict[str, torch.Tensor]:
    """The master's state dict, memory-mapped: names to float32 tensors, nothing else."""
    try:
        sd = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except RuntimeError as e:
        log(f"{path.name}: no memory map ({e}); loading it whole")
        sd = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(sd, dict) or not all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in sd.items()
    ):
        raise SystemExit(f"{path}: not a flat dict of tensors")
    dtypes = {str(v.dtype) for v in sd.values()}
    if dtypes != {"torch.float32"}:
        raise SystemExit(f"{path}: dtypes {sorted(dtypes)}, float32 only expected")
    return sd


def read_back(path: Path, sd: dict[str, torch.Tensor], md: dict[str, str]) -> None:
    """The file as the safetensors library reads it: the metadata, and every tensor equal, bit
    for bit, to its master's rounding."""
    with safe_open(path, framework="pt") as f:
        if f.metadata() != md:
            raise SystemExit(f"{path}: metadata read back differs")
        if set(f.keys()) != set(sd):
            raise SystemExit(f"{path}: tensor names read back differ")
        for k in f.keys():
            got, want = f.get_tensor(k), sd[k].to(torch.float16)
            if (
                got.dtype != torch.float16
                or got.shape != want.shape
                or not torch.equal(got.view(torch.int16), want.view(torch.int16))
            ):
                raise SystemExit(f"{path}: {k} read back differs")


def compare(ours: Path, ref: Path) -> dict[str, object]:
    """Ours against numz's file: names, dtypes and shapes, and each tensor's bytes."""
    ho, so = read_header(ours)
    hr, sr = read_header(ref)
    ho.pop("__metadata__", None)
    ref_metadata = hr.pop("__metadata__", None)
    if set(ho) != set(hr):
        raise SystemExit(
            f"{ours.name} and {ref.name}: tensor names differ "
            f"(only ours: {sorted(set(ho) - set(hr))[:5]}, "
            f"only numz's: {sorted(set(hr) - set(ho))[:5]})"
        )
    mo, mr = np.memmap(ours, np.uint8, "r"), np.memmap(ref, np.uint8, "r")
    elements, differ = 0, {}
    for k, a in ho.items():
        b = hr[k]
        if a["dtype"] != b["dtype"] or a["shape"] != b["shape"]:
            raise SystemExit(
                f"{k}: {a['dtype']} {a['shape']} here, {b['dtype']} {b['shape']} in numz's"
            )
        x = mo[so + a["data_offsets"][0] : so + a["data_offsets"][1]]
        y = mr[sr + b["data_offsets"][0] : sr + b["data_offsets"][1]]
        elements += math.prod(a["shape"])
        if not np.array_equal(x, y):
            differ[k] = int((x.view(np.uint16) != y.view(np.uint16)).sum())
    return {
        "tensors": len(ho),
        "elements": elements,
        "tensors_differing": len(differ),
        "elements_differing": sum(differ.values()),
        "differing_head": dict(list(differ.items())[:5]),
        "same_order_and_offsets": list(ho) == list(hr)
        and all(ho[k]["data_offsets"] == hr[k]["data_offsets"] for k in ho),
        "numz_metadata": ref_metadata,
    }


def build(name: str, f: File, a: argparse.Namespace) -> None:
    t0 = time.time()
    master = find_or_fetch(
        f.master,
        hf_url(REPO, REVISION, f.master),
        f.master_sha256,
        f.master_size,
        a.masters,
        a.cache,
    )
    ref = None
    if not a.no_reference:
        ref = find_or_fetch(
            f.numz,
            hf_url(NUMZ_REPO, NUMZ_REVISION, f.numz),
            f.numz_sha256,
            f.numz_size,
            a.reference,
            a.cache,
        )
    log(f"{name}: from {master} ({time.time() - t0:.0f} s for the SHA-256s)")
    sd = load_master(master)
    md = metadata(f)
    r = Rounding()
    entries = [
        Entry(
            k, "F16", tuple(v.shape), lambda v=v: memoryview(r.round(v).reshape(-1).view(np.uint8))
        )
        for k, v in sd.items()
    ]
    out = a.out / name
    t1 = time.time()
    try:
        write_safetensors(out, entries, md)
        log(
            f"{name}: {r.tensors} tensors, {r.elements} values rounded in "
            f"{time.time() - t1:.0f} s: "
            f"{r.exact} exact ({100 * r.exact / r.elements:.2f}%), {r.subnormal} subnormal, "
            f"{r.to_zero} rounded to zero; torch = numpy bit for bit; {out.stat().st_size} bytes"
        )
        t1 = time.time()
        read_back(out, sd, md)
        log(
            f"{name}: read back by safetensors {safetensors.__version__}: metadata and every "
            f"tensor equal ({time.time() - t1:.0f} s)"
        )
        if ref is not None:
            c = compare(out, ref)
            log(f"{name} against numz's {f.numz}: {c}")
            if c["tensors_differing"]:
                raise SystemExit(f"{name}: {c['elements_differing']} values differ from numz's")
            log(
                f"{name}: equal to numz's {f.numz}, element for element ({c['tensors']} tensors, "
                f"{c['elements']} values)"
            )
        check_output(out, OUTPUTS.get(name))
    except BaseException:
        out.unlink(missing_ok=True)
        raise
    log(f"{name}: done in {time.time() - t0:.0f} s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "files", nargs="*", metavar="FILE", help=f"of {', '.join(FILES)} (default: all)"
    )
    ap.add_argument(
        "--masters",
        type=Path,
        action="append",
        default=[],
        help="a directory holding ByteDance's .pth files (read in place)",
    )
    ap.add_argument(
        "--reference",
        type=Path,
        action="append",
        default=[],
        help="a directory holding numz's fp16 files (read in place)",
    )
    ap.add_argument("--cache", type=Path, default=CACHE, help="where missing files are downloaded")
    ap.add_argument("--out", type=Path, default=DIST)
    ap.add_argument(
        "--no-reference", action="store_true", help="skip the check against numz's files"
    )
    a = ap.parse_args()
    for name in a.files:
        if name not in FILES:
            ap.error(f"{name}: not one of {', '.join(FILES)}")
    log(
        f"torch {torch.__version__}, numpy {np.__version__}, "
        f"safetensors {safetensors.__version__}, {torch.get_num_threads()} threads"
    )
    for name in a.files or FILES:
        build(name, FILES[name], a)


if __name__ == "__main__":
    main()

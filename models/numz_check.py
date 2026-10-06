# /// script
# requires-python = "==3.12.*"
# dependencies = ["torch==2.14.1", "numpy==2.5.2"]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""How numz's SeedVR2 files were made: each tensor against ByteDance's fp32 master.

A tensor stored in float16 is checked against the master rounded to the nearest float16, ties
to even; one stored in float8_e4m3fn against the master cast straight to it (torch's cast, nearest,
ties to even), and against the same cast of numz's float16 file, the other possible source. Every
file is checked by its SHA-256 first. No file is made.

    uv run models/numz_check.py --numz DIR [--masters DIR]... [--cache DIR] [FILE...]

- --numz: the directory holding numz's files (read in place).
- The masters: ByteDance's, at the pinned revisions, read where a --masters directory has them,
  else downloaded into --cache (default ~/.cache/seedvr2x-models).
- FILE: of CHECKS (default: all). The 7B's fp8 file is the control: measurement found it to be
  the master cast straight to fp8 (research/docs/models.md), and this check must find the same.
- The 3B's files are checked against both of ByteDance's 3B masters: its repository replaced the
  3B's weights on 2025-06-22 ("update ckpt"); the earlier file is cached apart, by revision.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from common import CACHE, check_file, find_or_fetch, log, read_header


class Master(NamedTuple):
    repo: str
    revision: str
    name: str
    size: int
    sha256: str
    current: bool = True  # main's file; an earlier one shares its name: cached apart, by revision


class Check(NamedTuple):
    masters: tuple[str, ...]  # keys of MASTERS, each checked
    size: int
    sha256: str  # numz's registry (src/utils/model_registry.py at 4490bd1)
    fp16: str | None  # numz's float16 file of the same model, the other possible source of fp8


MASTERS = {
    "7b": Master(
        "ByteDance-Seed/SeedVR2-7B",
        "eb0c4281d41ba3767d4f14370f0e37e9e9180c16",
        "seedvr2_ema_7b.pth",
        32958774606,
        "e1b2ae25505607e61f2a7dc7967ba778aaf3e3626d9969ce6e24c52d9ddebfcd",
    ),
    # main since 2025-06-22, "update ckpt", which replaced the 3B's weights
    "3b": Master(
        "ByteDance-Seed/SeedVR2-3B",
        "37255ff8cccfb01071b87f635a5948ca8d53117c",
        "seedvr2_ema_3b.pth",
        13566090228,
        "6bcc5ac59447e97b100477480aebb01be2ec724c8340bb83faae21f64848604b",
    ),
    # the 3B's first weights, from 2025-06-11 to 2025-06-20
    "3b-2025-06-20": Master(
        "ByteDance-Seed/SeedVR2-3B",
        "e2bc8d4d0845482db0e7ebd2164d501bfd1c1697",
        "seedvr2_ema_3b.pth",
        13566090228,
        "91627bba45175e1f5913378bc16151dc9c28c78fd608913546238ac13041d97d",
        current=False,
    ),
}
CHECKS = {
    "seedvr2_ema_3b_fp16.safetensors": Check(
        ("3b", "3b-2025-06-20"),
        6783018808,
        "2fd0e03a3dad24e07086750360727ca437de4ecd456f769856e960ae93e2b304",
        None,
    ),
    "seedvr2_ema_3b_fp8_e4m3fn.safetensors": Check(
        ("3b", "3b-2025-06-20"),
        3391544696,
        "3bf1e43ebedd570e7e7a0b1b60d6a02e105978f505c8128a241cde99a8240cff",
        "seedvr2_ema_3b_fp16.safetensors",
    ),
    "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors": Check(
        ("7b",),
        8466296338,
        "3d68b5ec0b295ae28092e355c8cad870edd00b817b26587d0cb8f9dd2df19bb2",
        "seedvr2_ema_7b_fp16.safetensors",
    ),
}
FP16_SHA256 = {
    "seedvr2_ema_7b_fp16.safetensors": (
        "7b8241aa957606ab6cfb66edabc96d43234f9819c5392b44d2492d9f0b0bbe4a"
    ),
}
DTYPES = {
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F32": torch.float32,
}


def raw(t: torch.Tensor) -> np.ndarray:
    return t.contiguous().reshape(-1).view(torch.uint8).numpy()


def tensors(path: Path):
    """name -> (dtype, shape, the tensor's bytes) of a safetensors file, memory-mapped."""
    header, start = read_header(path)
    header.pop("__metadata__", None)
    mm = np.memmap(path, np.uint8, "r")
    return {
        k: (v["dtype"], v["shape"], mm[start + v["data_offsets"][0] : start + v["data_offsets"][1]])
        for k, v in header.items()
    }


def equal_elements(a: np.ndarray, b: np.ndarray, itemsize: int) -> int:
    view = {1: np.uint8, 2: np.uint16, 4: np.uint32}[itemsize]
    return int((a.view(view) == b.view(view)).sum())


def fetch_master(m: Master, a: argparse.Namespace) -> Path:
    url = f"https://huggingface.co/{m.repo}/resolve/{m.revision}/{m.name}"
    if m.current:
        return find_or_fetch(m.name, url, m.sha256, m.size, a.masters, a.cache)
    cache = a.cache / m.revision[:7]  # an earlier file of the same name: never mixed with main's
    return find_or_fetch(m.name, url, m.sha256, m.size, [cache], cache)


def check(name: str, c: Check, key: str, a: argparse.Namespace) -> dict[str, object]:
    m = MASTERS[key]
    master = fetch_master(m, a)
    path = a.numz / name
    log(f"{name}: checking {path}")
    check_file(path, c.sha256, c.size)
    sd = torch.load(master, map_location="cpu", mmap=True, weights_only=True)
    ours = tensors(path)
    fp16 = None
    if c.fp16:
        p16 = a.numz / c.fp16
        if c.fp16 in FP16_SHA256:
            check_file(p16, FP16_SHA256[c.fp16])
        elif c.fp16 in CHECKS:
            check_file(p16, CHECKS[c.fp16].sha256, CHECKS[c.fp16].size)
        fp16 = tensors(p16)
    if set(ours) != set(sd):
        log(
            f"{name}: names differ from the master: only numz's {sorted(set(ours) - set(sd))[:5]}, "
            f"only the master's {sorted(set(sd) - set(ours))[:5]}"
        )
    groups: dict[str, dict[str, int]] = {}
    for k, (dtype, shape, data) in ours.items():
        if k not in sd:
            continue
        w = sd[k]
        if list(w.shape) != shape:
            raise SystemExit(f"{k}: shape {shape} in numz's file, {list(w.shape)} in the master")
        g = groups.setdefault(
            dtype,
            {
                "tensors": 0,
                "elements": 0,
                "equal_master": 0,
                "tensors_equal_master": 0,
                "equal_fp16_recast": 0,
            },
        )
        n = math.prod(shape)
        size = DTYPES[dtype].itemsize
        eq = equal_elements(data, raw(w.to(DTYPES[dtype])), size)
        g["tensors"] += 1
        g["elements"] += n
        g["equal_master"] += eq
        g["tensors_equal_master"] += eq == n
        if dtype == "F8_E4M3" and fp16 is not None:
            _, s16, b16 = fp16[k]
            h = torch.from_numpy(np.asarray(b16).view(np.float16).copy()).reshape(s16)
            g["equal_fp16_recast"] += equal_elements(data, raw(h.to(torch.float8_e4m3fn)), size)
    for g in groups.values():
        g["share_equal_master"] = g["equal_master"] / g["elements"]
        g["share_equal_fp16_recast"] = g["equal_fp16_recast"] / g["elements"]
    return {
        "file": name,
        "master": f"{m.repo}@{m.revision[:7]} {m.name}",
        "tensors": len(ours),
        "groups": groups,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "files", nargs="*", metavar="FILE", help=f"of {', '.join(CHECKS)} (default: all)"
    )
    ap.add_argument("--numz", type=Path, required=True)
    ap.add_argument("--masters", type=Path, action="append", default=[])
    ap.add_argument("--cache", type=Path, default=CACHE)
    a = ap.parse_args()
    for name in a.files:
        if name not in CHECKS:
            ap.error(f"{name}: not one of {', '.join(CHECKS)}")
    log(f"torch {torch.__version__}, numpy {np.__version__}")
    results = []
    for name in a.files or CHECKS:
        for key in CHECKS[name].masters:
            t0 = time.time()
            r = check(name, CHECKS[name], key, a)
            results.append(r)
            log(
                f"{name} against {r['master']}: {json.dumps(r['groups'])} "
                f"({time.time() - t0:.0f} s)"
            )
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

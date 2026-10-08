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
"""SeedVR2's 3B in float16, for research: rounded from ByteDance's current 3B master.

Not a file of the upload. numz's 3B fp16 file holds the 3B's FIRST weights (numz_check.py), which
ByteDance replaced on 2025-06-22 ("update ckpt"); this file holds the current ones, for a GPU run
of both. seedvr2_fp16.py's method, its functions imported: every tensor of the master rounded to
the nearest float16, ties to even (torch, checked bit for bit against numpy), under the same name
and shape; written by common.py in the library's layout, with seedvr2_fp16.py's metadata keys;
read back with the safetensors library, bit for bit.

    uv run models/seedvr2_fp16_3b.py --out DIR [--masters DIR]... [--reference DIR]...
                                     [--cache DIR] [--no-reference] [--first]

- The master: seedvr2_ema_3b.pth of ByteDance-Seed/SeedVR2-3B at REVISION, checked by its
  SHA-256, read where a --masters directory has it, never moved; else downloaded into --cache.
- numz's 3B fp16 file (numz/SeedVR2_comfyUI at seedvr2_fp16.py's NUMZ_REVISION, checked by
  SHA-256), from a --reference directory or downloaded into --cache: ours must hold its tensor
  names, dtypes and shapes, in its order at its offsets; the share of values equal to it is
  printed, not required (other weights). --no-reference skips it.
- --first, the method's control, instead: the same rounding of the 3B's first master (FIRST, at
  FIRST_REVISION; read from DIR/<its first 7 hex digits> of a --masters DIR or of --cache, as
  numz_check.py caches it) must give numz's file's data section byte for byte. Not pinned.
- The output, in --out (not models/dist: the upload): its size and SHA-256 must be the ones
  pinned in OUTPUTS. A file failing a check is deleted.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import safetensors
import torch
from common import (
    CACHE,
    CHUNK,
    DIST,
    Entry,
    check_output,
    find_or_fetch,
    log,
    read_header,
    write_safetensors,
)
from seedvr2_fp16 import (
    NUMZ_REPO,
    NUMZ_REVISION,
    File,
    Rounding,
    compare,
    hf_url,
    load_master,
    read_back,
)
from seedvr2_fp16 import metadata as metadata_7b

REPO = "ByteDance-Seed/SeedVR2-3B"
REVISION = "37255ff8cccfb01071b87f635a5948ca8d53117c"  # main since 2025-06-22, "update ckpt"
FILE = File(
    "seedvr2_ema_3b.pth",
    13566090228,
    "6bcc5ac59447e97b100477480aebb01be2ec724c8340bb83faae21f64848604b",
    "seedvr2_ema_3b_fp16.safetensors",  # numz's: the first weights, rounded the same way
    6783018808,
    "2fd0e03a3dad24e07086750360727ca437de4ecd456f769856e960ae93e2b304",
)
NAME = "seedvr2x_ema_3b_fp16.safetensors"

# The control: the 3B's first weights, main from 2025-06-11 to 2025-06-20 ("Add task tag (#1)").
FIRST_REVISION = "e2bc8d4d0845482db0e7ebd2164d501bfd1c1697"
FIRST = FILE._replace(
    master_sha256="91627bba45175e1f5913378bc16151dc9c28c78fd608913546238ac13041d97d"
)
FIRST_NAME = "seedvr2x_ema_3b_first_fp16.safetensors"

# The output of the pinned inputs and versions, (size, SHA-256): a run must give these bytes.
OUTPUTS = {
    "seedvr2x_ema_3b_fp16.safetensors": (
        6783019464,
        "20bbc89f6db4e39675dada01116f181679f2f35124fc9d85b470ca9173eff8dc",
    ),
}


def metadata(f: File, revision: str) -> dict[str, str]:
    """seedvr2_fp16.py's metadata, the same keys and wording, for the 3B's master."""
    md = metadata_7b(f)
    md.update(
        source=f"{f.master} of {REPO}, revision {revision}",
        source_url=hf_url(REPO, revision, f.master),
        source_sha256=f.master_sha256,
        conversion="unofficial, not ByteDance's: made by seedvr2x's models/seedvr2_fp16_3b.py",
    )
    return md


def data_sha256(path: Path) -> tuple[int, str]:
    """The size and SHA-256 of a safetensors file's data section: every byte after the header."""
    _, start = read_header(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(start)
        while b := f.read(CHUNK):
            h.update(b)
    return path.stat().st_size - start, h.hexdigest()


def build(name: str, f: File, revision: str, a: argparse.Namespace, control: bool) -> None:
    t0 = time.time()
    url = hf_url(REPO, revision, f.master)
    if control:  # the same file name as main's: kept apart, by revision
        dirs = [d / revision[:7] for d in [*a.masters, a.cache]]
        master = find_or_fetch(f.master, url, f.master_sha256, f.master_size, dirs, dirs[-1])
    else:
        master = find_or_fetch(f.master, url, f.master_sha256, f.master_size, a.masters, a.cache)
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
    md = metadata(f, revision)
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
            if not c["same_order_and_offsets"]:
                raise SystemExit(f"{name}: tensors not in the order and at the offsets of numz's")
            equal = c["elements"] - c["elements_differing"]
            log(
                f"{name}: numz's {f.numz}'s names, dtypes, shapes, order and offsets; "
                f"{c['tensors'] - c['tensors_differing']} of {c['tensors']} tensors and "
                f"{equal} of {c['elements']} values ({100 * equal / c['elements']:.4f}%) equal"
            )
            if control:
                if c["tensors_differing"]:
                    raise SystemExit(f"{name}: {c['elements_differing']} values differ from numz's")
                ours, theirs = data_sha256(out), data_sha256(ref)
                if ours != theirs:
                    raise SystemExit(f"{name}: data section {ours}, numz's {theirs}")
                log(
                    f"{name}: numz's {f.numz}'s data section, byte for byte ({ours[0]} bytes, "
                    f"SHA-256 {ours[1]}): the method gives numz's 3B fp16 from the first master"
                )
        if not control:
            pin = OUTPUTS.get(name)
            if pin is not None and out.stat().st_size != pin[0]:
                raise SystemExit(f"{out}: {out.stat().st_size} bytes, the pinned output {pin[0]}")
            check_output(out, pin[1] if pin else None)
    except BaseException:
        out.unlink(missing_ok=True)
        raise
    log(f"{name}: done in {time.time() - t0:.0f} s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--out", type=Path, required=True, help="not models/dist: this file is not uploaded"
    )
    ap.add_argument(
        "--masters",
        type=Path,
        action="append",
        default=[],
        help="a directory holding ByteDance's 3B .pth (read in place; the first one in "
        f"DIR/{FIRST_REVISION[:7]})",
    )
    ap.add_argument(
        "--reference",
        type=Path,
        action="append",
        default=[],
        help=f"a directory holding numz's {FILE.numz} (read in place)",
    )
    ap.add_argument("--cache", type=Path, default=CACHE, help="where missing files are downloaded")
    ap.add_argument(
        "--no-reference", action="store_true", help="skip the comparison with numz's file"
    )
    ap.add_argument(
        "--first",
        action="store_true",
        help="the control instead: the first 3B master must give numz's data section",
    )
    a = ap.parse_args()
    if a.out.resolve() == DIST.resolve():
        ap.error(f"--out {a.out}: models/dist is the upload, and this file is not uploaded")
    if a.first and a.no_reference:
        ap.error("--first compares with numz's file: no --no-reference")
    log(
        f"torch {torch.__version__}, numpy {np.__version__}, "
        f"safetensors {safetensors.__version__}, {torch.get_num_threads()} threads"
    )
    if a.first:
        build(FIRST_NAME, FIRST, FIRST_REVISION, a, control=True)
    else:
        build(NAME, FILE, REVISION, a, control=False)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""How far a compiled seedvr2x run's output is from the plain one, against the spread between seeds.

seedvr2x's output depends on the seed of its noise; torch.compile changes how the models compute,
not what, but its fused kernels round differently. This script puts the two on one scale, on the
full-reference clips: for each clip, outputs of runs that differ in one thing (another seed,
compile on, another process compiling) are compared with each other, and each with the clip's
ground truth. CPU only.

Each run is a directory seedvr2x wrote (-o DIR: FFV1 segments seg_*.mkv, here in gbrp16le) or one
video file. Frames are decoded with ffmpeg as 16-bit RGB. Per pair of runs:
  equal         every frame bit for bit (the decoded frames, not the files)
  max_levels    the largest absolute difference of a sample, in 8-bit levels (16-bit / 257)
  mean_levels   the mean absolute difference, in 8-bit levels
  psnr_rgb      PSNR over the RGB samples, 16-bit peak
  psnr_y        PSNR of BT.709 luma (full-range RGB), 8-bit scale, as fr_metrics.py's psnr_y
Per run against the ground truth (the clip's .gt.mkv): psnr_y the same way.

Usage:
  compile_bits.py --clips NAME,NAME,... --gt-dir DIR --run LABEL=DIR_PATTERN ... \\
      --pair A:B ... --out RESULTS.json
DIR_PATTERN holds {clip} (say runs/p42/{clip}); --pair names two labels; a run missing for a
clip is skipped. ffmpeg on PATH.
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

# BT.709 luma weights, as fr_metrics.py computes Y.
LUMA = np.array([0.2126, 0.7152, 0.0722])


def inputs(path):
    """The files ffmpeg reads for a run: its segments in order, or the file itself."""
    path = Path(path)
    if path.is_dir():
        return sorted(path.glob("seg_*.mkv"))
    return [path]


def size(file):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", str(file)],
        check=True, capture_output=True, text=True,
    ).stdout.strip().split(",")
    return int(out[0]), int(out[1])


def frames(path):
    """Every frame of a run as uint16 RGB, (N, H, W, 3), its segments concatenated."""
    parts = []
    for file in inputs(path):
        width, height = size(file)
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(file), "-map", "0:v:0", "-f", "rawvideo",
             "-pix_fmt", "rgb48le", "-"],
            check=True, capture_output=True,
        ).stdout
        parts.append(np.frombuffer(raw, dtype="<u2").reshape(-1, height, width, 3))
    if not parts:
        raise FileNotFoundError(f"no frames in {path}")
    return np.concatenate(parts)


def luma8(rgb16):
    return (rgb16.astype(np.float64) / 257.0) @ LUMA


def psnr(mse, peak):
    return round(10 * math.log10(peak**2 / mse), 3) if mse > 0 else None


def compare(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if np.array_equal(a, b):
        return {"frames": n, "equal": True}
    diff = np.abs(a.astype(np.int32) - b.astype(np.int32))
    mse_rgb = float((diff.astype(np.float64) ** 2).mean())
    mse_y = float(((luma8(a) - luma8(b)) ** 2).mean())
    return {
        "frames": n,
        "equal": False,
        "max_levels": round(float(diff.max()) / 257.0, 3),
        "mean_levels": round(float(diff.mean()) / 257.0, 4),
        "values_differing": round(float((diff > 0).mean()), 4),
        "psnr_rgb": psnr(mse_rgb, 65535.0),
        "psnr_y": psnr(mse_y, 255.0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--clips", required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=DIR_PATTERN")
    parser.add_argument("--pair", action="append", default=[], metavar="A:B")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runs = dict(item.split("=", 1) for item in args.run)
    results = {"runs": runs, "pairs": args.pair, "clips": {}}
    for clip in args.clips.split(","):
        loaded = {}
        for label, pattern in runs.items():
            path = Path(pattern.format(clip=clip))
            if path.exists():
                loaded[label] = frames(path)
        gt = frames(args.gt_dir / f"{clip}.gt.mkv")
        row = {"against_gt": {}, "pairs": {}}
        for label, data in loaded.items():
            n = min(len(data), len(gt))
            row["against_gt"][label] = psnr(
                float(((luma8(data[:n]) - luma8(gt[:n])) ** 2).mean()), 255.0)
        for pair in args.pair:
            first, second = pair.split(":")
            if first in loaded and second in loaded:
                row["pairs"][pair] = compare(loaded[first], loaded[second])
        results["clips"][clip] = row
        print(clip, json.dumps(row), flush=True)
    args.out.write_text(json.dumps(results, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

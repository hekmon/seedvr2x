#!/usr/bin/env python3
"""Compare two PNG sequences (SeedVR2 --output_format png): PSNR and where the differences are.

  python frame_diff.py REF_DIR TEST_DIR [--tile T --overlap O [--band B]] [--json OUT]

Frames are matched by sorted file name. Prints per-sequence PSNR (8-bit RGB, all frames), the
worst frame, and a seam check against the tile grid implied by --tile/--overlap (SeedVR2's
tiling grid, in output pixels), per axis: the mean absolute difference and the gradient
excess (|test step| - |reference step| between neighbouring pixels) near the tile boundaries
vs. the rest of the image, and the mean signed difference of each tile's core (tile without
its blended margins). A seam shows up as boundary values well above the interior ones; a
per-tile brightness shift as spread tile offsets.

Needs numpy and Pillow (both are in the SeedVR2 venv).
"""
import argparse
import json
import math
import os

import numpy as np
from PIL import Image


def frames(d):
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(".png"))


def tile_edges(size, tile, overlap, scale=8):
    """Output-pixel spans [start, end) of SeedVR2's tiles along one axis (tiled_decode's grid)."""
    lat = math.ceil(size / (2 * scale)) * 2  # SeedVR2 pads the frame to a multiple of 16
    lt = max(1, tile // scale)
    lo = max(0, min(overlap // scale, lt - 1))
    stride = max(1, lt - lo)
    spans, y = [], 0
    while True:
        end = min(y + lt, lat)
        spans.append((y * scale, end * scale))
        if end >= lat:
            break
        y += stride
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--tile", type=int, default=0)
    ap.add_argument("--overlap", type=int, default=0)
    ap.add_argument("--band", type=int, default=8, help="pixels on each side of a boundary")
    ap.add_argument("--json")
    a = ap.parse_args()

    fa, fb = frames(a.ref), frames(a.test)
    n = min(len(fa), len(fb))
    if n == 0:
        raise SystemExit("no PNG frames")
    se_total, count, per_frame = 0.0, 0, []
    col = row = sd = gx = gy = None
    for pa, pb in zip(fa[:n], fb[:n]):
        x = np.asarray(Image.open(pa).convert("RGB"), dtype=np.float64)
        y = np.asarray(Image.open(pb).convert("RGB"), dtype=np.float64)
        d = y - x
        se = float((d ** 2).sum())
        se_total += se
        count += d.size
        mse = se / d.size
        per_frame.append(round(10 * math.log10(255 ** 2 / mse), 2) if mse > 0 else float("inf"))
        ad = np.abs(d).mean(axis=2)
        # gradient excess: how much sharper the test's pixel-to-pixel steps are than the reference's
        ex = (np.abs(np.diff(y, axis=1)) - np.abs(np.diff(x, axis=1))).mean(axis=(0, 2))
        ey = (np.abs(np.diff(y, axis=0)) - np.abs(np.diff(x, axis=0))).mean(axis=(1, 2))
        acc = (ad.mean(axis=0), ad.mean(axis=1), d.mean(axis=2), ex, ey)
        if col is None:
            col, row, sd, gx, gy = acc
        else:
            col, row, sd, gx, gy = (u + v for u, v in zip((col, row, sd, gx, gy), acc))
    col, row, sd, gx, gy = (v / n for v in (col, row, sd, gx, gy))
    mse = se_total / count
    out = {"ref": a.ref, "test": a.test, "frames": n,
           "psnr_db": round(10 * math.log10(255 ** 2 / mse), 2) if mse > 0 else float("inf"),
           "psnr_min_frame_db": min(per_frame), "psnr_per_frame": per_frame,
           "mean_abs_diff": round(float(col.mean()), 4), "mean_signed_diff": round(float(sd.mean()), 4)}

    if a.tile:
        h, w = len(row), len(col)
        res, cores = {}, {}
        for axis, prof, grad, size in (("x", col, gx, w), ("y", row, gy, h)):
            spans = tile_edges(size, a.tile, a.overlap)
            mask = np.zeros(size, bool)
            # interior tile edges: the start of each tile but the first, the end of each but the last
            edges = [s for s, _ in spans[1:]] + [e for _, e in spans[:-1]]
            for e in edges:
                mask[max(0, e - a.band):min(size, e + a.band)] = True
            gmask = mask[:-1]
            res[axis] = {"tiles": len(spans), "edges": sorted(set(edges)),
                         "boundary_mad": round(float(prof[mask].mean()), 4) if mask.any() else None,
                         "interior_mad": round(float(prof[~mask].mean()), 4),
                         "max_mad": round(float(prof.max()), 4), "argmax": int(prof.argmax()),
                         "boundary_grad_excess": round(float(grad[gmask].mean()), 4) if gmask.any() else None,
                         "interior_grad_excess": round(float(grad[~gmask].mean()), 4),
                         "max_grad_excess": round(float(grad.max()), 4), "grad_argmax": int(grad.argmax())}
            # tile cores: each tile without its blended margins
            ov = a.overlap // 2
            cores[axis] = [(s + (ov if i else 0), min(size, e - (ov if i < len(spans) - 1 else 0)))
                           for i, (s, e) in enumerate(spans) if s < size]
        offs = [float(sd[y0:y1, x0:x1].mean()) for y0, y1 in cores["y"] for x0, x1 in cores["x"] if y1 > y0 and x1 > x0]
        res["tile_offsets"] = {"min": round(min(offs), 3), "max": round(max(offs), 3),
                               "std": round(float(np.std(offs)), 3)}
        out["seams"] = res
    print(json.dumps({k: v for k, v in out.items() if k != "psnr_per_frame"}, indent=1))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f)


if __name__ == "__main__":
    main()

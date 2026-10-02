#!/usr/bin/env python3
"""No-reference quality proxies for a SeedVR2 PNG sequence, against its input and a reference run.

  quality_metrics.py OUT_DIR --input VIDEO [--skip N] [--ref REF_DIR] [--batch B]
                     [--temporal-overlap O] [--prepend P [--drop-first P]] [--tile T --tile-overlap O]
                     [--frames N] [--json F]
  quality_metrics.py --input VIDEO --skip N --frames N        # describe the input only
  quality_metrics.py --summary A.json B.json ...              # Markdown table of saved results

There is no ground truth: the input is the low-quality source and the output an upscale of it.
Every figure is a proxy and only means something compared with another configuration.

Input frames are read like SeedVR2 reads them (cv2.VideoCapture, seek to --skip, BGR -> RGB),
or from a PNG directory (--input-dir). The output is resized to the input size (INTER_AREA)
when they differ, so "vs input" figures are at the input resolution.

Per run:
- fidelity: PSNR (RGB) and SSIM (luma, 11x11 Gaussian, sigma 1.5) vs the input
- reference: PSNR vs another run's frames (--ref), e.g. the default configuration or another seed
- colour: per-channel RGB and CIELAB mean/std of output and input, and the mean CIEDE76 between
  the two after a Gaussian blur (sigma 4 px): low-frequency colour error, insensitive to detail
- sharpness: variance of the luma Laplacian (output and input), and the luma standard
  deviation inside flat areas of the input (local std < --flat-std levels): grain/noise added
- temporal: for each pair of consecutive frames, mean |dY| of the output and of the input, and
  the "added change" mean |d(Y_out - Y_in)|: what changed in the output that did not change in
  the input, at full resolution and on 16x16-pixel block means (low-frequency flicker:
  brightness/colour patches). Transitions are split into batch boundaries (where the output
  switches from one DiT batch to the next, from --batch/--temporal-overlap/--prepend, as
  SeedVR2's encode loop and phase 3 blending place them) and the rest. On held frames (input
  mean |dY| < --hold, e.g. anime animated on threes) the output's |dY| is pure added flicker
- tiles (--ref with --tile): mean signed luma offset of each VAE tile core vs the reference,
  over flat input areas only, where a shift is most visible

Needs numpy and opencv-python (both in the SeedVR2 venv).
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frame_diff import tile_edges  # noqa: E402


# ------------------------------------------------------------------ I/O

def png_frames(d):
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(".png"))


def read_png(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def video_frames(path, skip, count):
    """Frames as SeedVR2's CLI reads them: cv2.VideoCapture, CAP_PROP_POS_FRAMES seek, BGR->RGB."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    if skip:
        cap.set(cv2.CAP_PROP_POS_FRAMES, skip)
    out = []
    while len(out) < count:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


# ------------------------------------------------------------------ metrics

def luma(rgb):
    rgb = rgb.astype(np.float32)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def ssim(y1, y2):
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)  # noqa: E731
    m1, m2 = blur(y1), blur(y2)
    s11, s22, s12 = blur(y1 * y1) - m1 * m1, blur(y2 * y2) - m2 * m2, blur(y1 * y2) - m1 * m2
    v = ((2 * m1 * m2 + c1) * (2 * s12 + c2)) / ((m1 * m1 + m2 * m2 + c1) * (s11 + s22 + c2))
    return float(v.mean())


def lab(rgb):
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)


def local_std(y, k=9):
    m = cv2.blur(y, (k, k))
    return np.sqrt(np.maximum(cv2.blur(y * y, (k, k)) - m * m, 0))


def block_mean(x, b=16):
    h, w = x.shape[:2]
    return cv2.resize(x, (max(1, w // b), max(1, h // b)), interpolation=cv2.INTER_AREA)


def psnr_from_mse(mse):
    return round(10 * math.log10(255 ** 2 / mse), 2) if mse > 0 else float("inf")


def batch_boundaries(n, batch, overlap=0, prepend=0):
    """Transitions t (between output frames t-1 and t) where the output switches DiT batch.

    Mirrors encode_all_batches (start indices, step = batch - overlap) on the prepended
    sequence, and decode_all_batches (overlap frames [s, s+overlap) cross-faded), then removes
    the prepended frames. Returns (transitions, batch starts in output frames)."""
    total = n + prepend
    step = batch - overlap if overlap > 0 else batch
    if step <= 0:
        step, overlap = batch, 0
    starts, s = [], 0
    while s < total:
        end = min(s + batch, total)
        if s > 0 and end - s <= overlap:
            break
        starts.append(s)
        s += step
    # weight of the previous batch on frames s..s+overlap-1 (blend_overlapping_frames): linear
    # 1 -> 0 below 3 frames, Hann over the middle third from 3. A transition is a boundary
    # where that weight changes, i.e. where the frame source really switches.
    if overlap >= 3:
        u = np.clip((np.linspace(0, 1, overlap) - 1 / 3) * 3, 0, 1)
        w = 0.5 + 0.5 * np.cos(np.pi * u)
    else:
        w = np.linspace(1, 0, overlap)
    w = np.r_[1.0, w, 0.0]  # frame s-1 is pure previous batch, s+overlap pure current
    trans = set()
    for s in starts[1:]:
        for j in range(overlap + 1):
            t = s + j - prepend
            if abs(w[j + 1] - w[j]) > 1e-6 and 1 <= t <= n - 1:
                trans.add(t)
    return sorted(trans), [s - prepend for s in starts]


def split_stats(series, boundary):
    """Mean of a per-transition series inside batches and at boundaries, and the max at boundaries."""
    idx = np.arange(1, len(series) + 1)
    b = np.isin(idx, boundary)
    s = np.asarray(series, dtype=float)
    ok = np.isfinite(s)  # NaN = transition excluded
    bi, bb = ok & ~b, ok & b
    inner = float(s[bi].mean()) if bi.any() else None
    at = float(s[bb].mean()) if bb.any() else None
    return {"inner": r4(inner), "boundary": r4(at), "boundary_max": r4(float(s[bb].max())) if bb.any() else None,
            "ratio": r4(at / inner) if at is not None and inner else None, "n_boundary": int(bb.sum())}


def r4(v):
    return None if v is None else round(v, 4)


def analyse(a):
    outs = png_frames(a.out)[a.drop_first:] if a.out else []
    n = (min(len(outs), a.frames) if a.frames else len(outs)) if outs else a.frames
    if a.input_dir:
        ins = [read_png(p) for p in png_frames(a.input_dir)[:n]]
    else:
        ins = video_frames(a.input, a.skip, n)
    if not ins:
        raise SystemExit("no input frames")
    n = min(n, len(ins)) if outs else len(ins)
    refs = png_frames(a.ref)[:n] if a.ref else []
    if a.ref and len(refs) < n:
        n = len(refs)
    h, w = ins[0].shape[:2]
    res = {"out": a.out, "ref": a.ref, "input": a.input or a.input_dir, "skip": a.skip, "frames": n,
           "input_size": [w, h], "label": a.label}

    acc = {k: [] for k in ("psnr_in", "ssim_in", "psnr_ref", "lap_out", "lap_in", "flat_out", "flat_in",
                           "de_lf", "shift", "rgb_out", "rgb_in", "lab_out", "lab_in")}
    se_in = se_ref = 0.0
    cnt = 0
    tdo, tdi, add, add_lf = [], [], [], []
    prev = None
    tile_acc = None
    for i in range(n):
        x = ins[i]
        yi = luma(x)
        flat = local_std(yi) < a.flat_std
        if outs:
            o = read_png(outs[i])
            if o.shape[:2] != (h, w):
                res["output_size"] = [o.shape[1], o.shape[0]]
                o = cv2.resize(o, (w, h), interpolation=cv2.INTER_AREA)
        else:
            o = x
        yo = luma(o)
        d = o.astype(np.float32) - x.astype(np.float32)
        se_in += float((d * d).sum())
        cnt += d.size
        acc["psnr_in"].append(psnr_from_mse(float((d * d).mean())))
        acc["ssim_in"].append(ssim(yo, yi))
        if refs:
            r = read_png(refs[i])
            if r.shape[:2] != (h, w):
                r = cv2.resize(r, (w, h), interpolation=cv2.INTER_AREA)
            dr = o.astype(np.float32) - r.astype(np.float32)
            se_ref += float((dr * dr).sum())
            acc["psnr_ref"].append(psnr_from_mse(float((dr * dr).mean())))
            if a.tile:
                sd = (yo - luma(r))
                if tile_acc is None:
                    tile_acc = (np.zeros_like(sd), np.zeros_like(sd))
                tile_acc[0][flat] += sd[flat]
                tile_acc[1][flat] += 1
        acc["lap_out"].append(float(cv2.Laplacian(yo, cv2.CV_32F).var()))
        acc["lap_in"].append(float(cv2.Laplacian(yi, cv2.CV_32F).var()))
        if flat.any():
            acc["flat_out"].append(float(local_std(yo)[flat].mean()))
            acc["flat_in"].append(float(local_std(yi)[flat].mean()))
        lo, li = lab(o), lab(x)
        blo = cv2.GaussianBlur(lo, (0, 0), 4)
        bli = cv2.GaussianBlur(li, (0, 0), 4)
        acc["de_lf"].append(float(np.sqrt(((blo - bli) ** 2).sum(axis=2)).mean()))
        acc["shift"].append(float(yo.mean() - yi.mean()))
        of, xf = o.reshape(-1, 3).astype(np.float32), x.reshape(-1, 3).astype(np.float32)
        acc["rgb_out"].append(np.r_[of.mean(0), of.std(0)])
        acc["rgb_in"].append(np.r_[xf.mean(0), xf.std(0)])
        lof, lif = lo.reshape(-1, 3), li.reshape(-1, 3)
        acc["lab_out"].append(np.r_[lof.mean(0), lof.std(0)])
        acc["lab_in"].append(np.r_[lif.mean(0), lif.std(0)])
        resid = yo - yi
        resid_lf = block_mean(d)
        if prev is not None:
            pyo, pyi, presid, presid_lf = prev
            tdo.append(float(np.abs(yo - pyo).mean()))
            tdi.append(float(np.abs(yi - pyi).mean()))
            add.append(float(np.abs(resid - presid).mean()))
            add_lf.append(float(np.abs(resid_lf - presid_lf).mean()))
        prev = (yo, yi, resid, resid_lf)

    mean = lambda k: float(np.mean(acc[k])) if acc[k] else None  # noqa: E731
    vec = lambda k: [round(float(v), 2) for v in np.mean(acc[k], axis=0)]  # noqa: E731
    res.update({
        "psnr_in": psnr_from_mse(se_in / cnt), "psnr_in_min": min(acc["psnr_in"]),
        "ssim_in": r4(mean("ssim_in")),
        "psnr_ref": psnr_from_mse(se_ref / cnt) if refs else None,
        "psnr_ref_min": min(acc["psnr_ref"]) if refs else None,
        "lap_var_out": round(mean("lap_out"), 1), "lap_var_in": round(mean("lap_in"), 1),
        "flat_std_out": r4(mean("flat_out")), "flat_std_in": r4(mean("flat_in")),
        "de76_lowfreq": r4(mean("de_lf")), "luma_shift": r4(mean("shift")),
        "rgb_mean_std_out": vec("rgb_out"), "rgb_mean_std_in": vec("rgb_in"),
        "lab_mean_std_out": vec("lab_out"), "lab_mean_std_in": vec("lab_in"),
        "per_frame": {k: [round(float(v), 4) for v in acc[k]] for k in ("psnr_in", "psnr_ref", "shift", "de_lf")},
    })
    if n > 1:
        boundary, starts = batch_boundaries(n, a.batch, a.temporal_overlap, a.prepend) if a.batch else ([], [0])
        res["batch_starts"], res["boundaries"] = starts, boundary
        res["tdiff_out"], res["tdiff_in"] = r4(float(np.mean(tdo))), r4(float(np.mean(tdi)))
        res["added_change"] = split_stats(add, boundary)
        res["added_change_lf"] = split_stats(add_lf, boundary)
        res["tdiff_out_split"] = split_stats(tdo, boundary)
        # held frames (input nearly unchanged, e.g. anime animated on twos/threes): any output
        # change there is flicker added by the model
        hold = np.asarray(tdi) < a.hold
        res["hold_transitions"] = int(hold.sum())
        if hold.any():
            res["hold_tdiff_out"] = split_stats(np.where(hold, tdo, np.nan), boundary)
            res["hold_tdiff_in"] = r4(float(np.asarray(tdi)[hold].mean()))
        res["per_transition"] = {"tdiff_out": [round(v, 3) for v in tdo], "tdiff_in": [round(v, 3) for v in tdi],
                                 "added": [round(v, 3) for v in add], "added_lf": [round(v, 3) for v in add_lf]}
    if tile_acc is not None:
        mean_sd = np.where(tile_acc[1] > 0, tile_acc[0] / np.maximum(tile_acc[1], 1), np.nan)
        ov = a.tile_overlap // 2
        cores = {}
        for axis, size in (("x", w), ("y", h)):
            spans = tile_edges(size, a.tile, a.tile_overlap)
            cores[axis] = [(s + (ov if i else 0), min(size, e - (ov if i < len(spans) - 1 else 0)))
                           for i, (s, e) in enumerate(spans) if s < size]
        offs = []
        for y0, y1 in cores["y"]:
            for x0, x1 in cores["x"]:
                blk = mean_sd[y0:y1, x0:x1]
                if np.isfinite(blk).sum() > 1000:
                    offs.append(float(np.nanmean(blk)))
        res["tile_flat_offsets"] = {"tiles": len(offs), "min": r4(min(offs)) if offs else None,
                                    "max": r4(max(offs)) if offs else None,
                                    "std": r4(float(np.std(offs))) if offs else None,
                                    "flat_fraction": r4(float((tile_acc[1] > 0).mean()))}
    return res


# ------------------------------------------------------------------ summary table

COLUMNS = [
    ("PSNR in", lambda r: r.get("psnr_in")),
    ("SSIM in", lambda r: r.get("ssim_in")),
    ("PSNR ref", lambda r: r.get("psnr_ref")),
    ("ΔE lf", lambda r: r.get("de76_lowfreq")),
    ("Lap var", lambda r: r.get("lap_var_out")),
    ("flat std", lambda r: r.get("flat_std_out")),
    ("tdiff", lambda r: r.get("tdiff_out")),
    ("added in/bnd", lambda r: _pair(r.get("added_change"))),
    ("added lf in/bnd", lambda r: _pair(r.get("added_change_lf"))),
    ("hold tdiff in/bnd", lambda r: _pair(r.get("hold_tdiff_out"))),
]


def _pair(s):
    if not s:
        return None
    return f"{s['inner']} / {s['boundary']}" if s.get("boundary") is not None else f"{s['inner']} / –"


def summary(paths):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        name = r.get("label") or os.path.basename(p).rsplit(".", 1)[0]
        rows.append("| " + " | ".join([name] + ["–" if (v := fn(r)) is None else str(v) for _, fn in COLUMNS]) + " |")
    print("| Run | " + " | ".join(c for c, _ in COLUMNS) + " |")
    print("|---" * (len(COLUMNS) + 1) + "|")
    print("\n".join(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", nargs="?", help="output PNG directory")
    ap.add_argument("--input", help="input video (read with cv2 like the CLI)")
    ap.add_argument("--input-dir", help="input PNG directory instead of a video")
    ap.add_argument("--skip", type=int, default=0, help="--skip_first_frames of the run")
    ap.add_argument("--frames", type=int, default=0, help="analyse the first N frames only (required without an output directory)")
    ap.add_argument("--ref", help="reference PNG directory (another run)")
    ap.add_argument("--batch", type=int, default=0, help="--batch_size of the run (boundary split)")
    ap.add_argument("--temporal-overlap", type=int, default=0)
    ap.add_argument("--prepend", type=int, default=0, help="--prepend_frames of the run")
    ap.add_argument("--drop-first", type=int, default=0,
                    help="ignore the first N output frames (single-GPU runs keep the prepended frames)")
    ap.add_argument("--tile", type=int, default=0, help="VAE tile size (with --ref: per-tile offsets)")
    ap.add_argument("--tile-overlap", type=int, default=0)
    ap.add_argument("--flat-std", type=float, default=1.5, help="flat-area threshold (input local luma std)")
    ap.add_argument("--hold", type=float, default=1.0,
                    help="held-frame threshold: input mean |dY| below it (levels)")
    ap.add_argument("--label")
    ap.add_argument("--json", help="write the full result (per-frame series included)")
    ap.add_argument("--summary", nargs="+", metavar="JSON", help="print a Markdown table of saved results")
    a = ap.parse_args()
    if a.summary:
        return summary(a.summary)
    if not (a.input or a.input_dir):
        ap.error("--input or --input-dir is required")
    if not a.out and not a.frames:
        ap.error("give an output directory, or --frames for the input alone")
    res = analyse(a)
    brief = {k: v for k, v in res.items() if k not in ("per_frame", "per_transition")}
    print(json.dumps(brief, indent=1, ensure_ascii=False))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(res, f)


if __name__ == "__main__":
    main()

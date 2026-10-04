#!/usr/bin/env python3
"""Where the colour error of numz's `lab` sits: step 0 of the colour study (docs/colour.md).

  colour_diag.py scan --clip NAME --gt GT.mkv --bicubic BIC.mkv --run SEED NONE.mkv LAB.mkv
                 [--run ...] --json OUT.json [--oracle-seed 42] [--frames N] [--threads 2]
  colour_diag.py check OUT.json --fr STREAM=FR.json [--fr ...]  # sigma 4 against fr_metrics.py
  colour_diag.py report OUT.json ... > tables.md
  colour_diag.py histtest --clip NAME --gt GT.mkv --none NONE.mkv --bicubic BIC.mkv  # numz's matching
  colour_diag.py selftest

Why: on the full-reference clips (docs/numerics.md), `lab` leaves a low-frequency colour error
(ΔE00 1.07-1.58 after a 4 px blur) about twice the input's own (a Catmull-Rom upscale of it:
0.6-0.7). Before any variant is built, this measures where that error sits: at which scales, in
lightness or in colour, in which parts of the picture, and how much of it the input could still
correct. It reads 16-bit RGB masters only, on the CPU: the ground truth, a run with
--color_correction none, its `lab` rendering (numerics_patch.py NUM_CC_EXTRA=lab), and the
bicubic baseline.

Per frame, each output against the ground truth, in CIELAB as OpenCV converts float RGB (sRGB,
D65), as fr_metrics.py does:
- ΔE00 (CIEDE2000, Sharma, Wu and Dalal 2005) after a Gaussian blur of sigma 0 (none), 1, 2, 4,
  8 and 16 px of both CIELAB images, on every s-th row and column (s = 2 up to sigma 4, sigma / 2
  above: a mean over a regular subset of pixels). Sigma 4 is fr_metrics.py's de00_lf exactly.
  With its two parts: lightness |ΔL'/S_L|, and colour, the chroma and hue terms with their
  rotation, sqrt((ΔC'/S_C)² + (ΔH'/S_H)² + R_T (ΔC'/S_C)(ΔH'/S_H)): ΔE00² is the sum of their
  squares. Over the whole frame, and over the picture: the letterbox bars and the bottom 16
  rows left out (numz's padding damages those: numerics.md).
- Bands: the CIELAB difference split by differences of Gaussians (sigma < 1, 1-2, 2-4, 4-8,
  8-16, 16-32 and > 32 px); the mean square of ΔL* and of Δa*² + Δb*² in each, over the
  picture. The bands telescope: they add up to the difference itself.
- Classes of the ground truth's pixels, from its CIELAB blurred at sigma 1: the gradient of L*
  (flat: the frame's lower half; edges: its top tenth), lightness (L* < 30, 30-70, >= 70),
  chroma (C* < 10, 10-40, >= 40), hue in 60° sectors where C* >= 10, and rows (picture, bars,
  bottom 16). ΔE00 and its parts summed per class at sigma 0, 1, 2 and 4.
- Offsets: per frame, the mean ΔL*, Δa*, Δb* and ΔC* over the picture. The static share at
  sigma 4: the energy of the difference's mean over the clip, over the difference's energy.
- Detail: the luma Laplacian variance, as fr_clips.py computes it (8-bit scale).
- Oracle (one seed): what a split with lightness and colour at different scales could reach.
  The `none` output takes the low bands of the bicubic baseline, which stands in for the
  encoder's input. In Y'CbCr (BT.709 matrix on the gamma-encoded RGB, full range: linear, so
  equal scales are the plain RGB split), Y' takes the low band of sL stages, Cb and Cr that of
  sC stages. The low band is runtime/colour.py's: à-trous stages of the 3x3 binomial kernel,
  taps 2^stage px apart, edges replicated at each stage: 5 stages are a sigma of 13.1 px, 4:
  6.5, 3: 3.2, 2: 1.6, 1: 0.7. (5, 5) is numz's `wavelet` (lab without its histogram step), on
  the clamped bfloat16 master. No histogram step here.

Needs numpy and opencv (the metrics venv), skimage for the self-test, and ffmpeg.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

FFMPEG = os.environ.get("COLOUR_FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = os.environ.get("COLOUR_FFPROBE") or shutil.which("ffprobe") or "ffprobe"
RGB_BITS = {"gbrp": 8, "gbrp10le": 10, "gbrp12le": 12, "gbrp16le": 16}

SIGMAS = (0, 1, 2, 4, 8, 16)
STRIDE = {0: 2, 1: 2, 2: 2, 4: 2, 8: 4, 16: 8}
CLASS_SIGMAS = (0, 1, 2, 4)  # on the stride-2 grid
BAND_SIGMAS = (1, 2, 4, 8, 16, 32)
BAND_NAMES = ("<1", "1-2", "2-4", "4-8", "8-16", "16-32", ">32")
BOTTOM_ROWS = 16
BAR_LSTAR = 5.0  # a bar row: L* <= 5 everywhere in every frame of the ground truth (RGB <= 18/255)
ORACLE = ((5, 5), (5, 4), (5, 3), (5, 2), (5, 1), (4, 4), (4, 2), (3, 3), (3, 2), (2, 2))
STAGE_SIGMA = {s: math.sqrt(sum(4 ** k for k in range(s)) / 2) for s in range(1, 6)}

FAMILIES = {
    "rows": ("picture", "bars", "bottom16"),
    "gradient": ("flat", "mid", "edge"),
    "lightness": ("L<30", "L30-70", "L>=70"),
    "chroma": ("C<10", "C10-40", "C>=40"),
    "hue": ("neutral", "h0-60", "h60-120", "h120-180", "h180-240", "h240-300", "h300-360"),
}

# BT.709 Y'CbCr on gamma-encoded RGB, full range
YCC = np.array([[0.2126, 0.7152, 0.0722],
                [-0.2126 / 1.8556, -0.7152 / 1.8556, 0.9278 / 1.8556],
                [0.7874 / 1.5748, -0.7152 / 1.5748, -0.0722 / 1.5748]], dtype=np.float64)
YCC_INV = np.linalg.inv(YCC)


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ------------------------------------------------------------------ reading

class Master:
    """An RGB master decoded by ffmpeg as stored: float32 RGB (H, W, 3) in [0, 1]."""

    def __init__(self, path, threads=2):
        r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=pix_fmt,width,height", "-of", "json", path],
                           capture_output=True, text=True)
        streams = json.loads(r.stdout or "{}").get("streams") if not r.returncode else None
        if not streams:
            raise SystemExit(f"ffprobe {path}: {r.stderr.strip() or 'no video stream'}")
        st = streams[0]
        self.path, self.fmt = path, st["pix_fmt"]
        if self.fmt not in RGB_BITS:
            raise SystemExit(f"{path}: {self.fmt}, a planar RGB master (gbrp*) is expected")
        self.W, self.H, self.bits = st["width"], st["height"], RGB_BITS[self.fmt]
        self.dtype = np.dtype(np.uint8) if self.bits == 8 else np.dtype("<u2")
        self.size = 3 * self.W * self.H * self.dtype.itemsize
        self.proc = subprocess.Popen([FFMPEG, "-v", "error", "-nostdin", "-threads", str(threads),
                                      "-i", path, "-map", "0:v:0", "-fps_mode", "passthrough",
                                      "-f", "rawvideo", "-pix_fmt", self.fmt, "-"],
                                     stdout=subprocess.PIPE)

    def read(self):
        buf = self.proc.stdout.read(self.size)
        if len(buf) < self.size:
            return None
        g, b, r = np.frombuffer(buf, self.dtype).reshape(3, self.H, self.W)
        x = np.stack([r, g, b], axis=-1).astype(np.float32)
        return x * np.float32(1.0 / ((1 << self.bits) - 1))

    def close(self):
        if self.proc:
            self.proc.kill()
            self.proc.stdout.close()
            self.proc.wait()
            self.proc = None


# ------------------------------------------------------------------ colour maths

def to_lab(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)


def blur(x, sigma):
    return x if sigma == 0 else cv2.GaussianBlur(x, (0, 0), sigma)


def ciede2000(lab1, lab2):
    """CIEDE2000 of lab1 (reference) and lab2, (..., 3) float32, kL = kC = kH = 1, from Sharma,
    Wu and Dalal (2005): (dE, tl, tc, th, rt), with tl = ΔL'/S_L, tc = ΔC'/S_C, th = ΔH'/S_H
    (lab2 minus lab1) and rt = R_T, so that dE² = tl² + tc² + th² + rt·tc·th."""
    f = np.float32
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    cbar = f(0.5) * (np.hypot(a1, b1) + np.hypot(a2, b2))
    c7 = cbar ** 7
    g1 = f(1.5) - f(0.5) * np.sqrt(c7 / (c7 + f(25.0 ** 7)))  # 1 + G
    a1p, a2p = a1 * g1, a2 * g1
    c1, c2 = np.hypot(a1p, b1), np.hypot(a2p, b2)
    twopi = f(2 * math.pi)
    h1 = np.arctan2(b1, a1p) % twopi
    h2 = np.arctan2(b2, a2p) % twopi
    cc = c1 * c2
    zero = cc == 0
    dh = h2 - h1
    dh = np.where(dh > f(math.pi), dh - twopi, np.where(dh < f(-math.pi), dh + twopi, dh))
    dh[zero] = 0
    dhh = f(2) * np.sqrt(cc) * np.sin(dh * f(0.5))
    hsum = h1 + h2
    far = (np.abs(h1 - h2) > f(math.pi)) & ~zero
    hbar = np.where(far, np.where(hsum < twopi, hsum + twopi, hsum - twopi), hsum)
    hbar = np.where(zero, hbar, hbar * f(0.5))
    deg = f(math.pi / 180)
    t = (f(1) - f(0.17) * np.cos(hbar - f(30) * deg) + f(0.24) * np.cos(f(2) * hbar)
         + f(0.32) * np.cos(f(3) * hbar + f(6) * deg) - f(0.20) * np.cos(f(4) * hbar - f(63) * deg))
    lbar = f(0.5) * (L1 + L2) - f(50)
    sl = f(1) + f(0.015) * lbar * lbar / np.sqrt(f(20) + lbar * lbar)
    cbp = f(0.5) * (c1 + c2)
    sc = f(1) + f(0.045) * cbp
    sh = f(1) + f(0.015) * cbp * t
    cb7 = cbp ** 7
    rc = f(2) * np.sqrt(cb7 / (cb7 + f(25.0 ** 7)))
    dtheta = f(30) * deg * np.exp(-((hbar / deg - f(275)) / f(25)) ** 2)
    rt = -np.sin(f(2) * dtheta) * rc
    tl, tc, th = (L2 - L1) / sl, (c2 - c1) / sc, dhh / sh
    de = np.sqrt(np.maximum(tl * tl + tc * tc + th * th + rt * tc * th, 0))
    return de, tl, tc, th, rt


def colour_part(tc, th, rt):
    return np.sqrt(np.maximum(tc * tc + th * th + rt * tc * th, 0))


def lap_var(y):
    """fr_clips.py's: the 4-neighbour Laplacian of the 8-bit-scale luma, inner pixels."""
    lap = y[1:-1, :-2] + y[1:-1, 2:] + y[:-2, 1:-1] + y[2:, 1:-1] - 4 * y[1:-1, 1:-1]
    return float(lap.var())


def luma8(rgb):
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]) * np.float32(255)


def low_bands(x, stages=5):
    """runtime/colour.py's low band of each channel of x (H, W, C) float32 after 1..stages à-trous
    stages: the 3x3 binomial kernel, taps 2^stage px apart (at most min(H, W) // 8), edges
    replicated at each stage. {stage: (H, W, C)}."""
    out = {}
    cap = max(1, min(x.shape[:2]) // 8)
    for s in range(stages):
        d = min(2 ** s, cap)
        k = np.zeros(2 * d + 1, np.float32)
        k[0], k[d], k[2 * d] = 0.25, 0.5, 0.25
        x = cv2.sepFilter2D(x, -1, k, k, borderType=cv2.BORDER_REPLICATE)
        out[s + 1] = x
    return out


# ------------------------------------------------------------------ scan

def find_bars(path, frames, threads):
    """Letterbox rows of the ground truth: from the top and from the bottom, the rows whose L* stays
    <= BAR_LSTAR in every frame."""
    src = Master(path, threads)
    rowmax, n = None, 0
    while not frames or n < frames:
        x = src.read()
        if x is None:
            break
        m = to_lab(x)[..., 0].max(axis=1)
        rowmax = m if rowmax is None else np.maximum(rowmax, m)
        n += 1
    src.close()
    if rowmax is None:
        raise SystemExit(f"{path}: no frame")
    dark = rowmax <= BAR_LSTAR
    top = int(np.argmin(dark)) if not dark.all() else len(dark)
    bottom = int(np.argmin(dark[::-1])) if not dark.all() else 0
    return top, bottom, n


class Acc:
    """One output's accumulators."""

    def __init__(self, grid2, classes):
        self.pf = {}
        self.cls = {s: {fam: np.zeros((len(names) + 1, 4)) for fam, names in classes.items()}
                    for s in CLASS_SIGMAS}
        self.s1 = np.zeros(grid2 + (3,))
        self.s2 = np.zeros(grid2 + (3,))
        self.frames = 0

    def add(self, key, v):
        self.pf.setdefault(key, []).append(round(float(v), 6))


def frame_classes(gtb1, pic2, top, bottom, H):
    """Class index per pixel of the stride-2 grid, per family, from the ground truth's CIELAB
    blurred at sigma 1 (full resolution); the last index of each family = outside the picture."""
    lab2 = gtb1[::2, ::2]
    L, a, b = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    gx = cv2.Sobel(gtb1[..., 0], cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(gtb1[..., 0], cv2.CV_32F, 0, 1, ksize=3) / 8
    grad = np.hypot(gx, gy)[::2, ::2]
    rows = np.arange(0, H, 2)[: lab2.shape[0]]
    out = {}
    r = np.zeros(L.shape, np.int8)
    bars = (rows < top) | (rows >= H - bottom)
    r[(rows >= H - BOTTOM_ROWS) & ~bars, :] = 2
    r[bars, :] = 1
    out["rows"] = r
    p50, p90 = np.percentile(grad[pic2], (50, 90))
    out["gradient"] = np.where(grad <= p50, 0, np.where(grad > p90, 2, 1)).astype(np.int8)
    out["lightness"] = np.where(L < 30, 0, np.where(L >= 70, 2, 1)).astype(np.int8)
    c = np.hypot(a, b)
    out["chroma"] = np.where(c < 10, 0, np.where(c >= 40, 2, 1)).astype(np.int8)
    hue = (np.degrees(np.arctan2(b, a)) % 360) // 60 + 1
    out["hue"] = np.where(c < 10, 0, hue).astype(np.int8)
    for fam in ("gradient", "lightness", "chroma", "hue"):
        out[fam][~pic2] = len(FAMILIES[fam])
    return out


def scan(a):
    cv2.setNumThreads(a.threads)
    t0 = time.perf_counter()
    top, bottom, n_gt = find_bars(a.gt, a.frames, a.threads)
    log(f"{a.clip}: bars {top} rows at the top, {bottom} at the bottom ({n_gt} frames)")
    outputs = [("bicubic", a.bicubic)]
    for seed, none, lab in a.run:
        outputs += [(f"none.s{seed}", none), (f"lab.s{seed}", lab)]
    oracle_none = f"none.s{a.oracle_seed}" if a.oracle_seed is not None else None
    if oracle_none and oracle_none not in dict(outputs):
        raise SystemExit(f"--oracle-seed {a.oracle_seed}: no --run with that seed")
    gt = Master(a.gt, a.threads)
    srcs = {name: Master(path, a.threads) for name, path in outputs}
    for name, s in srcs.items():
        if (s.W, s.H) != (gt.W, gt.H):
            raise SystemExit(f"{s.path}: {s.W}x{s.H}, the ground truth is {gt.W}x{gt.H}")
    H, W = gt.H, gt.W
    rows_full = np.arange(H)
    pic_rows = (rows_full >= top) & (rows_full < H - max(bottom, BOTTOM_ROWS))
    pic = {s: np.broadcast_to(pic_rows[::st, None], (len(range(0, H, st)), len(range(0, W, st))))
           for s, st in STRIDE.items()}
    pic2 = pic[0]
    grid2 = pic2.shape
    accs = {name: Acc(grid2, FAMILIES) for name, _ in outputs}
    oracles = {f"oracle.s{a.oracle_seed}.L{sl}C{sc}": (sl, sc) for sl, sc in ORACLE} if oracle_none else {}
    for name in oracles:
        accs[name] = Acc(grid2, {})
    gt_lap = []
    n = 0
    while not a.frames or n < a.frames:
        g = gt.read()
        xs = {name: s.read() for name, s in srcs.items()}
        if g is None or any(x is None for x in xs.values()):
            break
        lab_g = to_lab(g)
        gb = {s: blur(lab_g, s) for s in (0,) + BAND_SIGMAS}
        classes = frame_classes(gb[1], pic2, top, bottom, H)
        gt_lap.append(lap_var(luma8(g)))
        gs = {s: np.ascontiguousarray(gb[s][::STRIDE[s], ::STRIDE[s]]) for s in SIGMAS}
        for name, x in xs.items():
            measure(accs[name], x, gb, gs, pic, classes, bands=True)
        if oracles:
            none, bic = xs[oracle_none], xs["bicubic"]
            delta = (bic - none) @ YCC.T.astype(np.float32)
            lbs = low_bands(np.ascontiguousarray(delta))
            for name, (sl, sc) in oracles.items():
                d = np.concatenate([lbs[sl][..., :1], lbs[sc][..., 1:]], axis=-1)
                out = np.clip(none + d @ YCC_INV.T.astype(np.float32), 0, 1)
                measure(accs[name], out, gb, gs, pic, None, bands=False)
        n += 1
        if n % 5 == 0:
            log(f"{a.clip}: {n} frames, {time.perf_counter() - t0:.0f} s")
    gt.close()
    for s in srcs.values():
        s.close()
    if n == 0:
        raise SystemExit("no frame compared")
    res = {"clip": a.clip, "gt": os.path.abspath(a.gt), "frames": n, "size": [W, H],
           "bars": [top, bottom], "sigmas": SIGMAS, "stride": STRIDE, "band_names": BAND_NAMES,
           "families": FAMILIES, "oracle": {k: list(v) for k, v in oracles.items()},
           "stage_sigma": STAGE_SIGMA, "gt_lap_var": [round(v, 3) for v in gt_lap],
           "outputs": {}, "seconds": round(time.perf_counter() - t0, 1)}
    paths = dict(outputs)
    for name, acc in accs.items():
        m2 = pic2.sum()
        static = {}
        if acc.frames:
            mean = acc.s1 / acc.frames
            sq = acc.s2 / acc.frames
            static = {"L": [float((mean[..., 0] ** 2)[pic2].sum() / m2), float(sq[..., 0][pic2].sum() / m2)],
                      "C": [float((mean[..., 1:] ** 2).sum(-1)[pic2].sum() / m2),
                            float(sq[..., 1:].sum(-1)[pic2].sum() / m2)]}
        res["outputs"][name] = {
            "path": os.path.abspath(paths[name]) if name in paths else None,
            "per_frame": acc.pf,
            "classes": {str(s): {fam: v.tolist() for fam, v in d.items()} for s, d in acc.cls.items()}
            if acc.frames else {},
            "static_sigma4": static,
        }
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(res, f)
    log(f"{a.clip}: {n} frames, {len(accs)} outputs in {res['seconds']} s -> {a.json}")


def measure(acc, x, gb, gs, pic, classes, bands):
    lab_x = to_lab(x)
    acc.add("lap_var", lap_var(luma8(x)))
    xb = {0: lab_x}
    for s in SIGMAS[1:] + ((32,) if bands else ()):
        xb[s] = blur(lab_x, s)
    for s in SIGMAS:
        st = STRIDE[s]
        xs = np.ascontiguousarray(xb[s][::st, ::st])
        de, tl, tc, th, rt = ciede2000(gs[s], xs)
        cp = colour_part(tc, th, rt)
        p = pic[s]
        acc.add(f"de{s}", de.mean())
        acc.add(f"de{s}_pic", de[p].mean())
        acc.add(f"tl{s}_pic", np.abs(tl[p]).mean())
        acc.add(f"tc{s}_pic", cp[p].mean())
        acc.add(f"tl_sq{s}_pic", (tl[p] ** 2).mean())
        acc.add(f"tc_sq{s}_pic", (cp[p] ** 2).mean())
        if classes is not None and s in CLASS_SIGMAS:
            for fam, idx in classes.items():
                k = len(FAMILIES[fam]) + 1
                ii = idx.ravel()
                acc.cls[s][fam] += np.stack([
                    np.bincount(ii, weights=de.ravel(), minlength=k),
                    np.bincount(ii, weights=(tl * tl).ravel(), minlength=k),
                    np.bincount(ii, weights=(cp * cp).ravel(), minlength=k),
                    np.bincount(ii, minlength=k)], axis=1)
        if s == 0:
            d = xs - gs[0]
            acc.add("off_L", d[..., 0][p].mean())
            acc.add("off_a", d[..., 1][p].mean())
            acc.add("off_b", d[..., 2][p].mean())
            acc.add("off_C", (np.hypot(xs[..., 1], xs[..., 2]) - np.hypot(gs[0][..., 1], gs[0][..., 2]))[p].mean())
    if not bands:
        return
    p = pic[0]
    ds = [xb[s][::2, ::2] - gb[s][::2, ::2] for s in (0,) + BAND_SIGMAS]
    el, ec = [], []
    for k in range(len(ds)):
        band = ds[k] - ds[k + 1] if k + 1 < len(ds) else ds[k]
        el.append(float((band[..., 0] ** 2)[p].mean()))
        ec.append(float((band[..., 1:] ** 2).sum(-1)[p].mean()))
    acc.pf.setdefault("band_L", []).append([round(v, 6) for v in el])
    acc.pf.setdefault("band_C", []).append([round(v, 6) for v in ec])
    d4 = ds[BAND_SIGMAS.index(4) + 1].astype(np.float64)
    acc.s1 += d4
    acc.s2 += d4 * d4
    acc.frames += 1


# ------------------------------------------------------------------ the histogram step alone

def histtest(a):
    """numz's histogram step on top of the oracle's split, the clip pooled as one batch: does it
    alone account for what `lab` leaves at coarse scales? numz's matching (color_fix.py:330-343,
    477-521, Apache-2.0): a* and b* get the reference's value of the same rank, L* = 0.8 L* + 0.2
    matched; here in OpenCV's CIELAB, the reference being the bicubic baseline."""
    cv2.setNumThreads(a.threads)
    srcs = [Master(p, a.threads) for p in (a.gt, a.none, a.bicubic)]
    frames = []
    while not a.frames or len(frames) < a.frames:
        xs = [s.read() for s in srcs]
        if any(x is None for x in xs):
            break
        frames.append(xs)
    for s in srcs:
        s.close()
    t = len(frames)
    shape = frames[0][0].shape
    gts = {s: np.stack([np.ascontiguousarray(blur(to_lab(f[0]), s)[::STRIDE[s], ::STRIDE[s]]) for f in frames])
           for s in SIGMAS}
    ref = np.stack([to_lab(f[2]) for f in frames])
    print(f"{a.clip}: {t} frames, ΔE00 after a blur of sigma (px), whole frame")
    print("| Clip | Split | Histogram step | " + " | ".join(f"σ {s}" for s in SIGMAS) + " |")
    print("|---|---|---|" + "---|" * len(SIGMAS))
    for sl, sc in ((5, 5), (5, 3)):
        lab = np.empty((t,) + shape, np.float32)
        for i, (_, none, bic) in enumerate(frames):
            lbs = low_bands(np.ascontiguousarray((bic - none) @ YCC.T.astype(np.float32)))
            d = np.concatenate([lbs[sl][..., :1], lbs[sc][..., 1:]], axis=-1)
            lab[i] = to_lab(np.clip(none + d @ YCC_INV.T.astype(np.float32), 0, 1))
        matched = {}
        for ch in (0, 1, 2):
            order = np.argsort(lab[..., ch].ravel(), kind="stable")
            m = np.empty(order.size, np.float32)
            m[order] = np.sort(ref[..., ch].ravel())
            matched[ch] = m.reshape(lab.shape[:3])
        variants = {
            "none": lab,
            "a*b*": np.stack([lab[..., 0], matched[1], matched[2]], axis=-1),
            "numz (a*b*, L* 0.2)": np.stack([0.8 * lab[..., 0] + 0.2 * matched[0], matched[1], matched[2]], axis=-1),
        }
        for name, v in variants.items():
            de = {s: [] for s in SIGMAS}
            for i in range(t):
                rgb = np.clip(cv2.cvtColor(np.ascontiguousarray(v[i]), cv2.COLOR_Lab2RGB), 0, 1)
                x = to_lab(rgb)
                for s in SIGMAS:
                    xs = np.ascontiguousarray(blur(x, s)[::STRIDE[s], ::STRIDE[s]])
                    de[s].append(float(ciede2000(gts[s][i], xs)[0].mean()))
            print(f"| {a.clip} | sL {sl} sC {sc} | {name} | " + " | ".join(f"{np.mean(de[s]):.3f}" for s in SIGMAS) + " |",
                  flush=True)


# ------------------------------------------------------------------ check against fr_metrics.py

def check(a):
    with open(a.json, encoding="utf-8") as f:
        res = json.load(f)
    worst = 0.0
    for spec in a.fr:
        name, path = spec.split("=", 1)
        with open(path, encoding="utf-8") as f:
            fr = json.load(f)
        ref = np.array(fr["per_frame"]["de00_lf"], float)
        mine = np.array(res["outputs"][name]["per_frame"]["de4"], float)
        n = min(len(ref), len(mine))
        d = np.abs(ref[:n] - mine[:n])
        worst = max(worst, float(d.max()))
        print(f"{res['clip']} {name}: {n} frames, mean {mine[:n].mean():.4f} vs fr_metrics {ref[:n].mean():.4f}, "
              f"largest per-frame difference {d.max():.2e}")
    print(f"largest difference {worst:.2e}")
    if worst > a.tolerance:
        raise SystemExit(f"beyond the tolerance {a.tolerance:g}")


# ------------------------------------------------------------------ report

def load(paths):
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def group(res, kind):
    """The outputs of one kind (none, lab) per seed."""
    return {k.split(".s", 1)[1]: v for k, v in res["outputs"].items() if k.startswith(kind + ".s")}


def mean_of(v):
    return float(np.mean(v))


def seed_stats(res, kind, key):
    per_seed = [mean_of(o["per_frame"][key]) for o in group(res, kind).values()]
    return float(np.mean(per_seed)), float(max(per_seed) - min(per_seed)), len(per_seed)


def band_rms(o, which):
    return np.sqrt(np.mean(np.array(o["per_frame"][which]), axis=0))


def report(a):
    runs = load(a.json)
    f2 = lambda v: f"{v:.2f}"  # noqa: E731
    f3 = lambda v: f"{v:.3f}"  # noqa: E731
    print("## ΔE00 after a blur of sigma (px), whole frame\n")
    print("none and lab: mean of the seeds (their spread in brackets for lab); lab / bicubic: the excess "
          "over the input's own error.\n")
    print("| Clip | Output | " + " | ".join(f"σ {s}" for s in SIGMAS) + " |")
    print("|---|---|" + "---|" * len(SIGMAS))
    for r in runs:
        bic = r["outputs"]["bicubic"]["per_frame"]
        rows = [("bicubic", [mean_of(bic[f"de{s}"]) for s in SIGMAS], None)]
        for kind in ("none", "lab"):
            st = [seed_stats(r, kind, f"de{s}") for s in SIGMAS]
            rows.append((f"{kind} ({st[0][2]} seeds)", [m for m, _, _ in st], [sp for _, sp, _ in st]))
        for label, vals, spread in rows:
            cells = [f2(v) + (f" ({sp:.2f})" if spread and label.startswith("lab") else "") for v, sp in
                     zip(vals, spread or [0] * len(vals))]
            print(f"| {r['clip']} | {label} | " + " | ".join(cells) + " |")
        lab = rows[2][1]
        print(f"| {r['clip']} | lab / bicubic | " + " | ".join(f"{l / b:.1f}×" for l, b in zip(lab, rows[0][1])) + " |")
    print("\n## Lightness and colour parts of ΔE00 (picture: bars and bottom 16 rows left out)\n")
    print("Mean |ΔL'/S_L| (lightness) and mean colour part; lab: mean of the seeds.\n")
    print("| Clip | Output | Part | " + " | ".join(f"σ {s}" for s in SIGMAS) + " |")
    print("|---|---|---|" + "---|" * len(SIGMAS))
    for r in runs:
        bic = r["outputs"]["bicubic"]["per_frame"]
        for label, src in (("bicubic", None), ("none", "none"), ("lab", "lab")):
            for part, key in (("lightness", "tl"), ("colour", "tc")):
                if src is None:
                    vals = [mean_of(bic[f"{key}{s}_pic"]) for s in SIGMAS]
                else:
                    vals = [seed_stats(r, src, f"{key}{s}_pic")[0] for s in SIGMAS]
                print(f"| {r['clip']} | {label} | {part} | " + " | ".join(f2(v) for v in vals) + " |")
    print("\n## Error per band: RMS of ΔL* and of the a*b* distance (picture)\n")
    print("Bands of the CIELAB difference by differences of Gaussians (sigma in px). Bold: the input "
          "(bicubic) closer to the ground truth than the model (none) in that band.\n")
    print("| Clip | Channel | Output | " + " | ".join(BAND_NAMES) + " |")
    print("|---|---|---|" + "---|" * len(BAND_NAMES))
    for r in runs:
        bic = r["outputs"]["bicubic"]
        for ch, key in (("L*", "band_L"), ("a*b*", "band_C")):
            bvals = band_rms(bic, key)
            nvals = np.mean([band_rms(o, key) for o in group(r, "none").values()], axis=0)
            lvals = np.mean([band_rms(o, key) for o in group(r, "lab").values()], axis=0)
            for label, vals in (("bicubic", bvals), ("none", nvals), ("lab", lvals)):
                cells = [(f"**{v:.2f}**" if label == "bicubic" and v < nv else f"{v:.2f}") for v, nv in zip(vals, nvals)]
                print(f"| {r['clip']} | {ch} | {label} | " + " | ".join(cells) + " |")
    print("\n## Where lab's error sits in the picture (seed mean)\n")
    print("Per class of ground-truth pixels: share of the picture's pixels, share of lab's ΔE00 sum, "
          "lab's mean ΔE00 and the bicubic's, at sigma 1 and 4.\n")
    for s in ("1", "4"):
        print(f"\n### sigma {s}\n")
        print("| Clip | Family | Class | Pixels | lab's error | lab ΔE00 | bicubic ΔE00 | lab lightness² share |")
        print("|---|---|---|---|---|---|---|---|")
        for r in runs:
            labs = list(group(r, "lab").values())
            bic = r["outputs"]["bicubic"]["classes"][s]
            for fam, names in r["families"].items():
                lab_sum = np.sum([np.array(o["classes"][s][fam]) for o in labs], axis=0)
                b = np.array(bic[fam])
                if fam == "rows":
                    tot_de, tot_n = lab_sum[:, 0].sum(), lab_sum[:, 3].sum()
                    rng = range(len(names))
                else:
                    tot_de, tot_n = lab_sum[:-1, 0].sum(), lab_sum[:-1, 3].sum()
                    rng = range(len(names))
                for k in rng:
                    cnt = lab_sum[k, 3]
                    if cnt == 0:
                        continue
                    mde = lab_sum[k, 0] / cnt
                    bde = b[k, 0] / max(b[k, 3], 1)
                    lsh = lab_sum[k, 1] / max(lab_sum[k, 1] + lab_sum[k, 2], 1e-12)
                    print(f"| {r['clip']} | {fam} | {names[k]} | {100 * cnt / tot_n:.1f}% | "
                          f"{100 * lab_sum[k, 0] / tot_de:.1f}% | {mde:.2f} | {bde:.2f} | {100 * lsh:.0f}% |")
    print("\n## Offsets and the static share (picture)\n")
    print("Per-frame mean differences (output − ground truth, CIELAB units): mean over frames, and "
          "mean |·|; static: share of the sigma-4 difference's energy in its mean over the clip.\n")
    print("| Clip | Output | ΔL* | Δa* | Δb* | ΔC* | mean \\|ΔL*\\| | static L* | static a*b* |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        for label in ("bicubic", "none", "lab"):
            outs = [r["outputs"]["bicubic"]] if label == "bicubic" else list(group(r, label).values())
            off = {k: np.mean([mean_of(o["per_frame"][f"off_{k}"]) for o in outs]) for k in "LabC"}
            absl = np.mean([np.mean(np.abs(o["per_frame"]["off_L"])) for o in outs])
            stl = np.mean([o["static_sigma4"]["L"][0] / o["static_sigma4"]["L"][1] for o in outs])
            stc = np.mean([o["static_sigma4"]["C"][0] / o["static_sigma4"]["C"][1] for o in outs])
            print(f"| {r['clip']} | {label} | {off['L']:+.2f} | {off['a']:+.2f} | {off['b']:+.2f} | {off['C']:+.2f} | "
                  f"{absl:.2f} | {100 * stl:.0f}% | {100 * stc:.0f}% |")
    print("\n## Oracle: the split with lightness and colour at different scales (one seed)\n")
    print("`none` with the bicubic's low bands: Y' below sL stages, Cb and Cr below sC stages "
          "(5 = sigma 13.1 px, 4 = 6.5, 3 = 3.2, 2 = 1.6, 1 = 0.7). ΔE00 at blurs of sigma (whole frame); "
          "lightness / colour parts at sigma 2 (picture); detail: Laplacian variance over none's.\n")
    print("| Clip | Output | " + " | ".join(f"σ {s}" for s in SIGMAS) + " | σ 2 lightness | σ 2 colour | Detail |")
    print("|---|---|" + "---|" * len(SIGMAS) + "---|---|---|")
    for r in runs:
        if not r["oracle"]:
            continue
        seed = next(iter(r["oracle"])).split(".")[1][1:]
        none = r["outputs"][f"none.s{seed}"]["per_frame"]
        lap0 = mean_of(none["lap_var"])
        names = ["bicubic", f"none.s{seed}", f"lab.s{seed}"] + list(r["oracle"])
        for name in names:
            pf = r["outputs"][name]["per_frame"]
            label = name if not name.startswith("oracle") else "sL {} sC {}".format(*r["oracle"][name])
            print(f"| {r['clip']} | {label} | " + " | ".join(f3(mean_of(pf[f'de{s}'])) for s in SIGMAS)
                  + f" | {mean_of(pf['tl2_pic']):.3f} | {mean_of(pf['tc2_pic']):.3f}"
                  + f" | {mean_of(pf['lap_var']) / lap0:.3f} |")


# ------------------------------------------------------------------ self-test

def selftest(_a):
    from skimage.color import deltaE_ciede2000
    # Sharma, Wu and Dalal (2005), table 1: pairs 1, 2, 3, 7 and 17
    pairs = [((50, 2.6772, -79.7751), (50, 0, -82.7485), 2.0425),
             ((50, 3.1571, -77.2803), (50, 0, -82.7485), 2.8615),
             ((50, 2.8361, -74.0200), (50, 0, -82.7485), 3.4412),
             ((50, 0, 0), (50, -1, 2), 2.3669),
             ((50, 2.5, 0), (73, 25, -18), 27.1492)]
    l1 = np.array([p[0] for p in pairs], np.float32)
    l2 = np.array([p[1] for p in pairs], np.float32)
    de = ciede2000(l1, l2)[0]
    sk = deltaE_ciede2000(l1.astype(np.float64), l2.astype(np.float64), channel_axis=-1)
    for (p1, p2, want), got, s in zip(pairs, de, sk):
        print(f"{p1} {p2}: {got:.4f} (skimage {s:.4f}, published {want:.4f})")
        assert abs(got - want) < 2e-4 and abs(s - want) < 2e-4
    rng = np.random.default_rng(1)
    rgb1 = rng.random((256, 256, 3), dtype=np.float32)
    rgb2 = np.clip(rgb1 + rng.normal(0, 0.03, rgb1.shape).astype(np.float32), 0, 1)
    rgb2[:8] = rgb1[:8]  # identical pixels, greys included
    rgb1[8:16] = rgb2[8:16] = 0.5
    a1, a2 = to_lab(rgb1), to_lab(rgb2)
    de, tl, tc, th, rt = ciede2000(a1, a2)
    sk = deltaE_ciede2000(a1.astype(np.float64), a2.astype(np.float64), channel_axis=-1)
    err = float(np.abs(de - sk).max())
    print(f"random pairs: largest difference to skimage {err:.2e} (mean ΔE00 {de.mean():.3f})")
    assert err < 1e-3
    parts = np.sqrt(tl * tl + colour_part(tc, th, rt) ** 2)
    assert float(np.abs(parts - de).max()) < 1e-4
    # bands telescope to the difference
    d = (a2 - a1)
    ds = [blur(d, s) for s in (0,) + BAND_SIGMAS]
    total = sum(ds[k] - ds[k + 1] for k in range(len(ds) - 1)) + ds[-1]
    assert float(np.abs(total - d).max()) < 1e-3
    # the oracle at (5, 5) is the RGB split: content + low(reference) - low(content)
    x, r = rgb1, rgb2
    lb = low_bands(np.ascontiguousarray((r - x) @ YCC.T.astype(np.float32)))[5]
    out = x + lb @ YCC_INV.T.astype(np.float32)
    rgb_split = x + (low_bands(r)[5] - low_bands(x)[5])
    err = float(np.abs(out - rgb_split).max())
    print(f"oracle (5, 5) against the RGB split: largest difference {err:.2e}")
    assert err < 1e-5
    # stage sigmas: 1 stage of the binomial kernel has variance 1/2; dilation d multiplies it by d²
    imp = np.zeros((257, 257), np.float32)
    imp[128, 128] = 1
    for s in range(1, 6):
        k = low_bands(imp, s)[s]
        yy = np.arange(257) - 128
        var = float((k.sum(1) * yy ** 2).sum())
        print(f"{s} stages: sigma {math.sqrt(var):.3f} px (expected {STAGE_SIGMA[s]:.3f})")
        assert abs(math.sqrt(var) - STAGE_SIGMA[s]) < 1e-3
    print("selftest passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--clip", required=True)
    s.add_argument("--gt", required=True)
    s.add_argument("--bicubic", required=True)
    s.add_argument("--run", nargs=3, action="append", metavar=("SEED", "NONE", "LAB"), required=True)
    s.add_argument("--json", required=True)
    s.add_argument("--oracle-seed")
    s.add_argument("--frames", type=int, default=0)
    s.add_argument("--threads", type=int, default=2)
    c = sub.add_parser("check")
    c.add_argument("json")
    c.add_argument("--fr", action="append", required=True, metavar="OUTPUT=FR.json")
    c.add_argument("--tolerance", type=float, default=2e-4)
    r = sub.add_parser("report")
    r.add_argument("json", nargs="+")
    h = sub.add_parser("histtest")
    h.add_argument("--clip", required=True)
    h.add_argument("--gt", required=True)
    h.add_argument("--none", required=True)
    h.add_argument("--bicubic", required=True)
    h.add_argument("--frames", type=int, default=0)
    h.add_argument("--threads", type=int, default=2)
    sub.add_parser("selftest")
    a = ap.parse_args()
    {"scan": scan, "check": check, "report": report, "histtest": histtest, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()

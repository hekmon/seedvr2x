#!/usr/bin/env python3
"""Where the model's lightness texture sits, band by band, against the ground truth, and what a
detail strength on top of the split would do: the colour study's 4K relay (docs/colour.md). CPU.

  colour_bands.py scan --clip NAME --gt GT.mkv --ref REF.pt --content DECODE.pt [--bicubic BIC.mkv]
                  [--rows A:B] [--every 3] [--frames N] [--threads 8] --out OUT.json
  colour_bands.py report OUT.json [OUT.json ...]
  colour_bands.py selftest

Why: at 4K the model trails bicubic on every metric but banding, and the close-up's skin comes out
over-textured. The split (colour_variants.py's split:ycc:4:3) takes the reference's lightness below
4 à-trous stages (σ 6.5 px) and leaves everything finer to the model. A detail strength a would
blend the model's K finest lightness bands toward the reference's:

  Y_DS(K, a) = s + (1 - a) D_K,   D_K = hp_K(r - c),   hp_K(x) = x - low_K(x),   K = 1..4

with s = c + low_4(r - c) the split's lightness (a = 1); K = 4, a = 0 is the reference's lightness.
This measures where the model's excess over the ground truth sits, and which K and a the ground
truth itself points to.

Everything on Y', the BT.709 luma of the gamma-encoded RGB in [0, 1] (colour_variants.YCC's first
row), float32, reported in 8-bit levels (x255): c from the model's raw decode (colour_dump.py's
decode.pt) and r from the reference (ref_f32.pt, the input through the model's transform), both
brought to [0, 1] and left unclamped; g the ground truth, b the bicubic upscale, s unclamped. Read
and aligned as colour_eval.py's score reads them (imported: its GT and master readers, its dump
loaders; frame i of each). s and D_K are computed on the whole frame, as seedvr2x would, then cut
to --rows (inside a letterbox); bands and high passes are measured on the cut frames, edges
replicated as colour_variants.low_band replicates them (every low band here is low_band itself).

scan, on frames 0, N, 2N, ... (--every N), sums pooled over those frames:
- bands: band_k(x) = low_{k-1}(x) - low_k(x), k = 1..5 (low_0 = x; low_k at σ 0.71, 1.58, 3.24,
  6.52, 13.06 px): the rms of g, s, r, b, c in each, and the correlation of s, r, b, c with g, on
  the whole cut frame and on the GT's flattest third of 60x60 blocks (per frame, the third with the
  lowest std of g; rows and columns past the last whole block left out)
- per K: PSNR-Y (8-bit peak, Y' clamped to [0, 1], the mean of per-frame values as colour_eval.py's
  psnr_y) of s, b, r, c and of Y_DS at a = 0.75, 0.5, 0.25, 0; the least-squares
  a*_K = 1 - <g - s, D_K> / <D_K, D_K> (raw, and clipped to [0, 1]; whole and flat third) and the
  PSNR at the clipped whole-frame value; the energy-matching a^E_K, the a in [0, 1] where
  rms(hp_K(Y_DS)) meets rms(hp_K(g)), on a grid of step 0.05 interpolated linearly, the first
  crossing from a = 1 down, whole and flat third, and the PSNR there; without a crossing, "none↓"
  when the output is already at or below the GT's energy at a = 1, "none↑" when it is still above
  at a = 0 (the excess sits in coarser bands, which leak into hp_K: the à-trous bands overlap). hp_K
  is linear, so n rms(hp_K(Y_DS))^2 = |hp_K s|^2 + 2 (1 - a) <hp_K s, hp_K D_K> + (1 - a)^2 |hp_K D_K|^2:
  the grid is evaluated exactly from those three sums.
- a check: the luma of colour_variants.apply("split:ycc:4:3")'s output (clamped RGB) against s
  where no channel clipped, and its PSNR-Y.
Writes OUT.json (everything, per-frame values included) and prints the clip's tables.

report: the tables over several scans' JSONs; the band ratios also over the rest of the picture,
the other two thirds of the blocks (and the partial blocks), derived exactly from the pooled sums:
n_whole rms_whole^2 = n_flat rms_flat^2 + n_rest rms_rest^2.

Needs torch (CPU), numpy, opencv (colour_eval.py's imports) and ffmpeg on the PATH;
COLOUR_BASELINE as colour_variants.py wants it.
"""
import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import colour_eval as E  # noqa: E402  its readers and loaders; colour_diag as E.D
import colour_variants as V  # noqa: E402  low_band, apply

LUMA = (0.2126, 0.7152, 0.0722)  # BT.709: colour_variants.YCC's first row
SPLIT, SL = "split:ycc:4:3", 4  # the split and its lightness stages
NB = 5  # bands
KS = (1, 2, 3, 4)
GRID = (0.75, 0.5, 0.25, 0.0)
STEP = 0.05
BLOCK = E.BLOCK  # 60
FLAT = 1 / 3
REGIONS = ("whole", "flat")
NAMES = ("g", "s", "r", "b", "c")
LV = 255.0
SIGMA = [0.0] + [math.sqrt(sum(4 ** j for j in range(k)) / 2) for k in range(1, NB + 1)]
assert max(KS) <= SL


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- the maths

def luma(rgb):
    """Y' of RGB (..., 3, H, W), its range kept: (..., H, W) float32."""
    rgb = rgb.to(torch.float32)
    return rgb[..., 0, :, :] * LUMA[0] + rgb[..., 1, :, :] * LUMA[1] + rgb[..., 2, :, :] * LUMA[2]


def luma_hw3(x):
    """Y' of a reader's (H, W, 3) float32 frame: (H, W) float32 tensor."""
    y = x[..., 0] * np.float32(LUMA[0]) + x[..., 1] * np.float32(LUMA[1]) + x[..., 2] * np.float32(LUMA[2])
    return torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))


def unit(x):
    """[-1, 1] to [0, 1], unclamped."""
    return (x.to(torch.float32) + 1.0) * 0.5


def lows(x, n):
    """[x, low_1(x), ..., low_n(x)]: colour_variants.low_band at each stage count."""
    return [x] + [V.low_band(x, k) for k in range(1, n + 1)]


def bands(x, n=NB):
    """[band_1(x), ..., band_n(x)] and low_n(x): they add up to x."""
    lo = lows(x, n)
    return [lo[k - 1] - lo[k] for k in range(1, n + 1)], lo[n]


def hp(x, k):
    return x - V.low_band(x, k)


def detail(s, d, a):
    """Y_DS = s + (1 - a) D_K."""
    return s + (1 - a) * d


def frame_planes(content, reference, rows):
    """One frame of content and reference, (3, H, W) in [-1, 1]: c, r, s and [D_1, ..., D_4] (Y' in
    [0, 1], unclamped), computed on the whole frame, then cut to rows (A, B)."""
    c = luma(unit(content))
    r = luma(unit(reference))
    d = r - c
    lo = lows(d, SL)
    s = c + lo[SL]
    a, b = rows

    def cut(x):
        return x[a:b].clone()

    return cut(c), cut(r), cut(s), [cut(d - lo[k]) for k in KS]


def flat_mask(g, block=BLOCK, share=FLAT):
    """The GT's flattest `share` of block x block blocks (the lowest std of g within the block; the
    blocks at or below the share's percentile) as a pixel mask of g's shape; rows and columns past
    the last whole block are left out."""
    h, w = g.shape
    hb, wb = h // block, w // block
    std = g[: hb * block, : wb * block].to(torch.float64).reshape(hb, block, wb, block)
    std = std.permute(0, 2, 1, 3).reshape(hb, wb, -1).std(dim=-1).numpy()
    m = torch.from_numpy(std <= np.percentile(std, 100 * share))
    mask = torch.zeros(h, w, dtype=torch.bool)
    mask[: hb * block, : wb * block] = m.repeat_interleave(block, 0).repeat_interleave(block, 1)
    return mask


def psnr(y, g):
    """PSNR (8-bit peak) of Y' clamped to [0, 1] against g in [0, 1]."""
    e = y.clamp(0, 1) - g
    return 10 * math.log10(1.0 / max(float((e * e).mean(dtype=torch.float64)), 1e-20))


def flat_view(x, m):
    """x (n, H, W) or (H, W) as (n, N) or (N,): every pixel, or those of the mask m."""
    if x.dim() == 3:
        return x.reshape(x.shape[0], -1) if m is None else x[:, m]
    return x.reshape(-1) if m is None else x[m]


def total(x):
    return x.sum(dim=-1, dtype=torch.float64).numpy()


def grid():
    return [round(i * STEP, 10) for i in range(int(round(1 / STEP)) + 1)]


def crossings(av, vals, target):
    """Where vals (on the ascending grid av) meets target, linearly interpolated, from a = 1 down."""
    f = [v - target for v in vals]
    out = []
    for j in range(len(av) - 1, 0, -1):
        if f[j] == 0:
            out.append(av[j])
        elif f[j] * f[j - 1] < 0:
            out.append(av[j] + (av[j - 1] - av[j]) * f[j] / (f[j] - f[j - 1]))
    if f[0] == 0:
        out.append(av[0])
    return out


class Acc:
    """Sums pooled over the frames, in float64, per region (whole, flat)."""

    def __init__(self, names):
        self.names = list(names)  # the stack's signals, g and s first
        assert self.names[:2] == ["g", "s"]
        n = len(self.names)
        self.n = dict.fromkeys(REGIONS, 0)
        self.band = {R: np.zeros((NB, 3, n)) for R in REGIONS}  # per band and signal: Σx, Σx², Σx·g
        self.hp = {R: np.zeros((len(KS), n)) for R in REGIONS}  # Σ hp_K(x)²
        # per K: Σ(g - s)·D_K, Σ D_K², Σ hp_K(s)·hp_K(D_K), Σ hp_K(D_K)²
        self.ds = {R: np.zeros((len(KS), 4)) for R in REGIONS}
        self.frame_astar = [[] for _ in KS]  # a*_K of each frame alone, whole

    def add(self, stack, ds, mask):
        """stack (n, H, W): the cut planes in self.names' order; ds: [D_1, ..., D_4] cut; mask: the
        GT's flattest third."""
        regions = (("whole", None), ("flat", mask))
        for R, m in regions:
            self.n[R] += stack[0].numel() if m is None else int(m.sum())
        lo = lows(stack, NB)
        for k in range(1, NB + 1):
            band = lo[k - 1] - lo[k]
            for R, m in regions:
                v = flat_view(band, m)
                self.band[R][k - 1] += np.stack([total(v), total(v * v), total(v * v[0])])
            del band
        e = stack[0] - stack[1]
        for j, K in enumerate(KS):
            h = stack - lo[K]  # hp_K of every signal
            d = ds[j]
            hd = hp(d, K)
            for R, m in regions:
                v = flat_view(h, m)
                self.hp[R][j] += total(v * v)
                ev, dv, hsv, hdv = (flat_view(x, m) for x in (e, d, h[1], hd))
                row = np.array([total(ev * dv), total(dv * dv), total(hsv * hdv), total(hdv * hdv)])
                self.ds[R][j] += row
                if R == "whole":
                    self.frame_astar[j].append(float(1 - row[0] / row[1]) if row[1] > 0 else None)
            del h, hd

    def result(self):
        out = {"pixels": dict(self.n), "bands": {}, "ds": {}}
        for R in REGIONS:
            n = self.n[R]
            rows = {}
            for k in range(NB):
                sx, sxx, sxg = self.band[R][k]
                mean = sx / n
                var = sxx / n - mean ** 2
                cov = sxg / n - mean * mean[0]
                rows[str(k + 1)] = {
                    "sigma": [SIGMA[k], SIGMA[k + 1]],
                    "rms": {nm: float(math.sqrt(sxx[i] / n) * LV) for i, nm in enumerate(self.names)},
                    "mean": {nm: float(mean[i] * LV) for i, nm in enumerate(self.names)},
                    "corr": {nm: float(cov[i] / math.sqrt(var[i] * var[0])) if var[i] * var[0] > 0 else None
                             for i, nm in enumerate(self.names) if i}}
            out["bands"][R] = rows
        av = grid()
        for j, K in enumerate(KS):
            res = {}
            for R in REGIONS:
                n = self.n[R]
                eD, DD, sD, hDD = self.ds[R][j]
                hs2, hg2 = self.hp[R][j][1], self.hp[R][j][0]
                raw = float(1 - eD / DD) if DD > 0 else None
                vals = [math.sqrt(max(hs2 + 2 * (1 - a) * sD + (1 - a) ** 2 * hDD, 0.0) / n) * LV for a in av]
                target = math.sqrt(hg2 / n) * LV
                cr = crossings(av, vals, target)
                res[R] = {"a_star_raw": raw, "a_star": None if raw is None else min(max(raw, 0.0), 1.0),
                          "hp_rms": {nm: float(math.sqrt(self.hp[R][j][i] / n) * LV) for i, nm in enumerate(self.names)},
                          "energy": {"target": target, "grid": av, "rms": vals, "crossings": cr,
                                     "a_E": cr[0] if cr else None},
                          "sums": {"g_minus_s_dot_D": eD, "D_dot_D": DD, "hp_s_dot_hp_D": sD, "hp_D_dot_hp_D": hDD,
                                   "hp_s_sq": hs2, "hp_g_sq": hg2}}
            res["frame_a_star"] = self.frame_astar[j]
            out["ds"][str(K)] = res
        return out


# ---------------------------------------------------------------- scan

def read_lumas(src, every, limit):
    """Y' of frames 0, every, 2 every, ... of a colour_eval reader, at most `limit` frames read (0: all):
    ({index: (H, W) float32}, frames read)."""
    out, n = {}, 0
    try:
        while not limit or n < limit:
            x = src.read()
            if x is None:
                break
            if n % every == 0:
                out[n] = luma_hw3(x)
            n += 1
    finally:
        src.close()
    return out, n


def scan(a):
    torch.set_num_threads(a.threads)
    t_start = time.perf_counter()
    gts, tg = read_lumas(E.D.Master(a.gt, a.ffmpeg_threads), a.every, a.frames)
    if not gts:
        raise SystemExit(f"{a.gt}: no frame")
    h, w = next(iter(gts.values())).shape
    bics, tb = None, tg
    if a.bicubic:
        bics, tb = read_lumas(E.MasterFrames(a.bicubic, a.ffmpeg_threads), a.every, a.frames)
        if next(iter(bics.values())).shape != (h, w):
            raise SystemExit(f"{a.bicubic}: not {w}x{h}")
    rows = (0, h)
    if a.rows:
        rows = tuple(int(v) for v in a.rows.split(":"))
        if not 0 <= rows[0] < rows[1] <= h:
            raise SystemExit(f"--rows {a.rows}: outside 0:{h}")
    log(f"{a.clip}: GT {tg} frames {w}x{h}" + (f", bicubic {tb}" if bics else "")
        + f" ({time.perf_counter() - t_start:.0f} s)")
    content = E.load_content(a.content)
    if tuple(content.shape[-2:]) != (h, w):
        raise SystemExit(f"{a.content}: {tuple(content.shape)}, the GT is {h}x{w}")
    t = min(tg, tb, content.shape[0])
    idx = [i for i in sorted(gts) if i < t]
    csel = content[idx]  # a copy
    del content
    ref = E.load_reference(a.ref, t, h, w)
    if ref.shape[0] < t or tuple(ref.shape[-2:]) != (h, w):
        raise SystemExit(f"{a.ref}: {tuple(ref.shape)}, {t} frames of {h}x{w} wanted")
    rsel = ref[idx]
    del ref
    log(f"{a.clip}: {t} frames aligned, {len(idx)} scanned (every {a.every}), rows {rows[0]}:{rows[1]} "
        f"({time.perf_counter() - t_start:.0f} s)")
    names = ["g", "s", "r", "b", "c"] if bics else ["g", "s", "r", "c"]
    acc = Acc(names)
    pf = defaultdict(list)
    kept = []
    check = {"max_diff_levels": 0.0, "unclipped": [], "psnr": []}
    for j, i in enumerate(idx):
        t0 = time.perf_counter()
        c, r, s, ds = frame_planes(csel[j], rsel[j], rows)
        g = gts[i][rows[0]:rows[1]].clone()
        b = bics[i][rows[0]:rows[1]].clone() if bics else None
        mask = flat_mask(g)
        stack = torch.stack([g, s, r, b, c] if bics else [g, s, r, c])
        acc.add(stack, ds, mask)
        del stack
        for nm, y in (("s", s), ("b", b), ("r", r), ("c", c)):
            if y is not None:
                pf[f"psnr_{nm}"].append(psnr(y, g))
        for K, d in zip(KS, ds):
            for av in GRID:
                pf[f"psnr_ds{K}_{av}"].append(psnr(detail(s, d, av), g))
        out = V.apply(SPLIT, csel[j:j + 1], rsel[j:j + 1])[0]  # (3, H, W) in [0, 1]
        y = luma(out)[rows[0]:rows[1]]
        ok = ((out > 0) & (out < 1)).all(dim=0)[rows[0]:rows[1]]
        if ok.any():
            check["max_diff_levels"] = max(check["max_diff_levels"], float((y - s)[ok].abs().max()) * LV)
        check["unclipped"].append(float(ok.float().mean()))
        check["psnr"].append(psnr(y, g))
        del out, y, ok
        kept.append((g, s, ds))
        log(f"{a.clip} frame {i}: PSNR-Y s {pf['psnr_s'][-1]:.2f}" + (f", b {pf['psnr_b'][-1]:.2f}" if bics else "")
            + f", flat {int(mask.sum())} px ({time.perf_counter() - t0:.1f} s)")
    del csel, rsel
    res = acc.result()
    for j, K in enumerate(KS):
        wr = res["ds"][str(K)]["whole"]
        ast, ae = wr["a_star"], wr["energy"]["a_E"]
        for g, s, ds in kept:
            if ast is not None:
                pf[f"psnr_ds{K}_astar"].append(psnr(detail(s, ds[j], ast), g))
            if ae is not None:
                pf[f"psnr_ds{K}_aE"].append(psnr(detail(s, ds[j], ae), g))
    mean = {k: float(np.mean(v)) for k, v in pf.items()}
    for K in KS:
        d = res["ds"][str(K)]
        d["psnr"] = {str(av): mean[f"psnr_ds{K}_{av}"] for av in GRID}
        d["psnr"]["a_star"] = mean.get(f"psnr_ds{K}_astar")
        d["psnr"]["a_E"] = mean.get(f"psnr_ds{K}_aE")
    result = {"clip": a.clip, "gt": os.path.abspath(a.gt), "ref": os.path.abspath(a.ref),
              "content": os.path.abspath(a.content), "bicubic": os.path.abspath(a.bicubic) if a.bicubic else None,
              "split": SPLIT, "size": [w, h], "rows": list(rows), "t": t, "every": a.every, "frames": idx,
              "block": BLOCK, "flat_share": FLAT, "flat_pixels": res["pixels"]["flat"] / res["pixels"]["whole"],
              "sigma": SIGMA[1:], "bands": res["bands"], "ds": res["ds"],
              "psnr": {nm: mean[f"psnr_{nm}"] for nm in ("s", "b", "r", "c") if f"psnr_{nm}" in mean},
              "check": {"max_diff_levels": check["max_diff_levels"], "unclipped": float(np.mean(check["unclipped"])),
                        "psnr": float(np.mean(check["psnr"])), "per_frame_psnr": check["psnr"]},
              "per_frame": dict(pf), "seconds": round(time.perf_counter() - t_start, 1),
              "threads": a.threads}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(result, f)
    print_clip(result)
    log(f"{a.clip}: {a.out} ({result['seconds']} s)")


# ---------------------------------------------------------------- tables

def fmt(x, spec=".2f"):
    return "none" if x is None else format(x, spec)


def a_e(e):
    """a^E, or why there is none: none↓ already at or below the GT's energy at a = 1, none↑ still above at a = 0."""
    if e["a_E"] is not None:
        return f"{e['a_E']:.2f}"
    return "none↓" if e["rms"][-1] <= e["target"] else "none↑" if e["rms"][0] > e["target"] else "none"


def present(res):
    return [n for n in NAMES if n in res["bands"]["whole"]["1"]["rms"]]


def rest_rms(res, k):
    """Band k's rms per signal outside the GT's flattest third: n_w rms_w² = n_f rms_f² + n_rest rms_rest²."""
    f = res["flat_pixels"]
    w, fl = res["bands"]["whole"][str(k)]["rms"], res["bands"]["flat"][str(k)]["rms"]
    return {n: math.sqrt(max(w[n] ** 2 - f * fl[n] ** 2, 0.0) / (1 - f)) for n in w}


def print_clip(res):
    a, b = res["rows"]
    fr = res["frames"]
    names = present(res)
    others = [n for n in names if n != "g"]
    print(f"\n## {res['clip']}: rows {a}:{b} of {res['size'][1]}, {len(fr)} frames ({fr[0]}..{fr[-1]} every "
          f"{res['every']}); the flattest third holds {100 * res['flat_pixels']:.1f}% of the pixels\n")
    for R in REGIONS:
        print(f"Bands, {'whole cut frame' if R == 'whole' else 'the GT flattest third'}: rms in 8-bit levels, "
              "ratio to g, correlation with g\n")
        print("| Band (σ px) | " + " | ".join(names) + " | " + " | ".join(f"{n}/g" for n in others) + " | "
              + " | ".join(f"ρ {n}" for n in others) + " |")
        print("|---|" + "---|" * (len(names) + 2 * len(others)))
        for k in range(1, NB + 1):
            row = res["bands"][R][str(k)]
            rms = row["rms"]
            print(f"| {k} ({row['sigma'][0]:.1f}–{row['sigma'][1]:.1f}) | " + " | ".join(f"{rms[n]:.3f}" for n in names)
                  + " | " + " | ".join(f"{rms[n] / rms['g']:.2f}" for n in others) + " | "
                  + " | ".join(fmt(row["corr"][n], "+.3f") for n in others) + " |")
        print()
    print("Bands, the rest (outside the flattest third, derived): rms in 8-bit levels, ratio to g\n")
    print("| Band (σ px) | " + " | ".join(names) + " | " + " | ".join(f"{n}/g" for n in others) + " |")
    print("|---|" + "---|" * (len(names) + len(others)))
    for k in range(1, NB + 1):
        rms = rest_rms(res, k)
        print(f"| {k} ({SIGMA[k - 1]:.1f}–{SIGMA[k]:.1f}) | " + " | ".join(f"{rms[n]:.3f}" for n in names) + " | "
              + " | ".join(f"{rms[n] / rms['g']:.2f}" for n in others) + " |")
    print()
    ps, ck = res["psnr"], res["check"]
    print("PSNR-Y (dB; Y' clamped; mean over the frames): " + ", ".join(f"{n} {ps[n]:.2f}" for n in ("s", "b", "r", "c")
                                                                        if n in ps)
          + f"; {res['split']} through colour_variants.apply {ck['psnr']:.2f} (|Y' - s| <= {ck['max_diff_levels']:.1e} "
          f"levels where no channel clipped: {100 * ck['unclipped']:.2f}% of the pixels)\n")
    print("Detail strength Y_DS = s + (1 - a) hp_K(r - c): a* least squares (raw; clipped), a^E energy match "
          "(first crossing from a = 1; none↓: below the GT at a = 1, none↑: above it at a = 0); rms of hp_K "
          "(levels) of g, of Y_DS at a = 1 (s) and a = 0; PSNR-Y at a\n")
    print("| K | a* raw | a* | a* flat raw | a^E | a^E flat | hp g | hp a=1 | hp a=0 | hp g flat | hp a=1 flat "
          "| hp a=0 flat | a=1 | 0.75 | 0.5 | 0.25 | 0 | at a* | at a^E |")
    print("|---|" + "---|" * 18)
    for K in KS:
        d = res["ds"][str(K)]
        w, fl = d["whole"], d["flat"]
        ew, ef = w["energy"], fl["energy"]
        p = d["psnr"]
        print(f"| {K} | {fmt(w['a_star_raw'], '.3f')} | {fmt(w['a_star'])} | {fmt(fl['a_star_raw'], '.3f')} | "
              f"{a_e(ew)} | {a_e(ef)} | {ew['target']:.3f} | {ew['rms'][-1]:.3f} | {ew['rms'][0]:.3f} | "
              f"{ef['target']:.3f} | {ef['rms'][-1]:.3f} | {ef['rms'][0]:.3f} | {ps['s']:.2f} | "
              + " | ".join(f"{p[str(av)]:.2f}" for av in GRID) + f" | {fmt(p['a_star'])} | {fmt(p['a_E'])} |")
    print()


def report(a):
    rs = []
    for p in a.jsons:
        with open(p, encoding="utf-8") as f:
            rs.append(json.load(f))
    print("## Bands: s/g, b/g and r/g per band (rms ratios; g in 8-bit levels; ρ: correlation of s with g); "
          "whole cut frame, the GT's flattest third, the rest (derived)\n")
    print("| Clip | Region | " + " | ".join(f"band {k} ({SIGMA[k - 1]:.1f}–{SIGMA[k]:.1f} px): g · s/g · b/g · r/g · ρs"
                                         for k in range(1, NB + 1)) + " |")
    print("|---|---|" + "---|" * NB)
    for r in rs:
        for R in (*REGIONS, "rest"):
            cells = []
            for k in range(1, NB + 1):
                rms = rest_rms(r, k) if R == "rest" else r["bands"][R][str(k)]["rms"]
                rho = "–" if R == "rest" else fmt(r["bands"][R][str(k)]["corr"]["s"], "+.2f")
                cells.append(f"{rms['g']:.2f} · {rms['s'] / rms['g']:.2f} · "
                             + (f"{rms['b'] / rms['g']:.2f}" if "b" in rms else "–")
                             + f" · {rms['r'] / rms['g']:.2f} · {rho}")
            print(f"| {r['clip']} | {R} | " + " | ".join(cells) + " |")
    print("\n## Detail strength: a*_K (least squares; raw → clipped; flat raw), a^E_K (energy match; whole, flat; "
          "none↓: below the GT at a = 1, none↑: above it at a = 0), PSNR-Y (dB) of s, b and Y_DS\n")
    print("| Clip | K | a* raw | a* | a* flat | a^E | a^E flat | s | b | a=0.75 | 0.5 | 0.25 | 0 | at a* | at a^E |")
    print("|---|---|" + "---|" * 13)
    for r in rs:
        ps = r["psnr"]
        for K in KS:
            d = r["ds"][str(K)]
            w, fl, p = d["whole"], d["flat"], d["psnr"]
            print(f"| {r['clip']} | {K} | {fmt(w['a_star_raw'], '.3f')} | {fmt(w['a_star'])} | "
                  f"{fmt(fl['a_star_raw'], '.3f')} | {a_e(w['energy'])} | {a_e(fl['energy'])} | "
                  f"{ps['s']:.2f} | {fmt(ps.get('b'))} | " + " | ".join(f"{p[str(av)]:.2f}" for av in GRID)
                  + f" | {fmt(p['a_star'])} | {fmt(p['a_E'])} |")
    print("\n## Checks\n")
    print("| Clip | rows | frames | flat share | split via apply: PSNR-Y | vs clamp(s) | max \\|Y' - s\\| (levels) "
          "| unclipped | seconds |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rs:
        ck = r["check"]
        print(f"| {r['clip']} | {r['rows'][0]}:{r['rows'][1]} | {len(r['frames'])} | {100 * r['flat_pixels']:.1f}% | "
              f"{ck['psnr']:.2f} | {r['psnr']['s']:.2f} | {ck['max_diff_levels']:.1e} | {100 * ck['unclipped']:.2f}% | "
              f"{r['seconds']} |")
    if a.detail:
        for r in rs:
            print_clip(r)


# ---------------------------------------------------------------- self-test

def selftest():
    import torch.nn.functional as F
    torch.manual_seed(0)
    # bands of a known pattern: cos(ωx), constant along y. Stage j (taps 2^j apart) passes cos(ωx)
    # times (1 + cos(2^j ω)) / 2, so band_k's gain is L_{k-1} - L_k with L_k the product over j < k;
    # exact past the 31 px the 5 stages reach from the edges
    h, w = 128, 512
    x = torch.arange(w, dtype=torch.float64)
    inner = slice(32, w - 32)
    print("band rms of 0.25 cos(2πx / P), measured (analytic), interior")
    for period in (2, 3, 4, 6, 8, 16, 32, 64):
        om = 2 * math.pi / period
        ref = 0.25 * torch.cos(om * x + 0.3)
        pat = ref.to(torch.float32).expand(h, w).contiguous()
        bs, low = bands(pat)
        tel = (sum(bs) + low - pat).abs().max().item()
        assert tel < 1e-6, f"bands don't add up: {tel}"
        gain = [1.0]
        for j in range(NB):
            gain.append(gain[-1] * (1 + math.cos(2 ** j * om)) / 2)
        cells = []
        for k in range(1, NB + 1):
            want = (gain[k - 1] - gain[k]) * ref[inner]
            got = bs[k - 1][:, inner].double()
            err = (got - want).abs().max().item()
            assert err < 2e-6, f"period {period} band {k}: {err}"
            cells.append(f"{got.pow(2).mean().sqrt().item():.4f} ({want.pow(2).mean().sqrt().item():.4f})")
        print(f"  P {period:2d}: " + "  ".join(cells))
    # a checkerboard at Nyquist sits in band 1 alone
    cb = ((torch.arange(64)[:, None] + torch.arange(64)[None, :]) % 2).to(torch.float32) * 2 - 1
    bs, _ = bands(cb)
    assert (bs[0] - cb)[16:-16, 16:-16].abs().max().item() < 1e-6
    # synthetic frames: a blurry reference and the model's decode as it plus texture, out of range in places
    t, h, w = 3, 192, 256
    base = torch.rand(t, 3, h // 16, w // 16) * 2 - 1
    ref = F.interpolate(base, size=(h, w), mode="bicubic", align_corners=False).clamp(-1, 1)
    content = (ref * 1.05 + 0.08 * torch.randn(t, 3, h, w)).clamp(-1.1, 1.1)
    rows = (16, h - 16)
    acc_star, acc_e, acc_none = Acc("gsrc"), Acc("gsrc"), Acc("gsrc")
    for i in range(t):
        c, r, s, ds = frame_planes(content[i], ref[i], rows)
        for d in ds:  # a = 1: the split, bit for bit
            assert torch.equal(detail(s, d, 1.0), s)
        z = (detail(s, ds[-1], 0.0) - r).abs().max().item()  # K = 4, a = 0: the reference
        assert z < 1e-6, f"Y_DS(4, 0) vs r: {z}"
        out = V.apply(SPLIT, content[i:i + 1], ref[i:i + 1])[0]  # s: the split's lightness
        ok = ((out > 0) & (out < 1)).all(dim=0)[rows[0]:rows[1]]
        dd = (luma(out)[rows[0]:rows[1]] - s)[ok].abs().max().item()
        assert dd < 1e-5 and ok.float().mean().item() > 0.5, f"s vs {SPLIT}: {dd}"
        g = detail(s, ds[1], 0.6)  # a ground truth at K = 2, a = 0.6
        acc_star.add(torch.stack([g, s, r, c]), ds, flat_mask(g, 16))
        g = detail(s, ds[2], 0.3)  # K = 3, a = 0.3
        acc_e.add(torch.stack([g, s, r, c]), ds, flat_mask(g, 16))
        g = detail(s, ds[2], 1.5)  # more texture than the model's
        acc_none.add(torch.stack([g, s, r, c]), ds, flat_mask(g, 16))
    res = acc_star.result()["ds"]["2"]
    for R in REGIONS:
        assert abs(res[R]["a_star_raw"] - 0.6) < 1e-5, (R, res[R]["a_star_raw"])
    assert all(abs(v - 0.6) < 1e-5 for v in res["frame_a_star"])
    res = acc_e.result()["ds"]["3"]
    for R in REGIONS:
        ae = res[R]["energy"]["a_E"]
        assert ae is not None and abs(ae - 0.3) < 0.02, (R, res[R]["energy"])
        assert abs(res[R]["a_star_raw"] - 0.3) < 1e-5
    res = acc_none.result()["ds"]["3"]["whole"]
    assert res["energy"]["a_E"] is None and abs(res["a_star_raw"] - 1.5) < 1e-5 and res["a_star"] == 1.0, res
    # the band rms of a signal pooled over frames: g = s gives s/g = 1 and ρ = 1
    acc = Acc("gsrc")
    acc.add(torch.stack([s, s, r, c]), ds, flat_mask(s, 16))
    bres = acc.result()["bands"]
    for R in REGIONS:
        for k in range(1, NB + 1):
            row = bres[R][str(k)]
            assert abs(row["rms"]["s"] - row["rms"]["g"]) < 1e-9 and abs(row["corr"]["s"] - 1) < 1e-6
    # the flattest third: noise whose amplitude doubles with each block column
    blk, hb, wb = 16, 6, 12
    amp = (2.0 ** torch.arange(wb, dtype=torch.float32)).repeat_interleave(blk)
    g = torch.randn(hb * blk, wb * blk) * amp[None, :] * 1e-3
    m = flat_mask(g, blk)
    assert m[:, : 4 * blk].all() and not m[:, 4 * blk:].any(), "flattest third"
    # PSNR: a 1-level offset is 48.13 dB
    g = torch.rand(64, 64) * 0.8 + 0.1
    assert abs(psnr(g + 1 / 255, g) - 20 * math.log10(255)) < 1e-3
    # crossings: from a = 1 down, interpolated
    av = grid()
    assert crossings(av, [1 - a for a in av], 0.5) == [0.5]
    assert crossings(av, [2.0] * len(av), 1.0) == []
    assert a_e({"a_E": None, "rms": [0.5] * len(av), "target": 1.0}) == "none↓"
    assert a_e({"a_E": None, "rms": [2.0] * len(av), "target": 1.0}) == "none↑"
    assert a_e({"a_E": 0.4, "rms": [], "target": 1.0}) == "0.40"
    # the rest of the picture from the whole frame and the flat third: a quarter at rms 1, the rest at 2
    fake = {"flat_pixels": 0.25, "bands": {"whole": {"1": {"rms": {"g": math.sqrt(0.25 + 0.75 * 4)}}},
                                           "flat": {"1": {"rms": {"g": 1.0}}}}}
    assert abs(rest_rms(fake, 1)["g"] - 2) < 1e-12
    print("baseline", os.path.abspath(V.BASELINE))
    print("selftest passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--clip", required=True)
    s.add_argument("--gt", required=True, help="16-bit RGB ground truth")
    s.add_argument("--ref", required=True, help="colour_dump.py's ref_f32.pt")
    s.add_argument("--content", required=True, help="colour_dump.py's decode.pt")
    s.add_argument("--bicubic", help="the bicubic upscale (an RGB or seedvr2x YUV master)")
    s.add_argument("--rows", metavar="A:B", help="rows kept (e.g. inside the letterbox); all by default")
    s.add_argument("--every", type=int, default=3, help="frames 0, N, 2N, ...")
    s.add_argument("--frames", type=int, default=0, help="at most the first N frames (0: all)")
    s.add_argument("--threads", type=int, default=8)
    s.add_argument("--ffmpeg-threads", type=int, default=4)
    s.add_argument("--out", required=True, help="the JSON")
    r = sub.add_parser("report")
    r.add_argument("jsons", nargs="+")
    r.add_argument("--detail", action="store_true", help="each clip's full tables too")
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "scan":
        scan(a)
    elif a.cmd == "report":
        report(a)
    else:
        selftest()


if __name__ == "__main__":
    main()

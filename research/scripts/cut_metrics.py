#!/usr/bin/env python3
"""What a scene cut costs inside one SeedVR2 processing unit, measured against the ground truth.

  cut_metrics.py analyze RUN.json --ref REF.json --cut C [--noise JSON ...] [--q 0.95] [--stable 4]
                 [--smooth N] [--thr METRIC=V ...] [--blocks 1,8] [--no-ghost] [--label L] [--json OUT]
  cut_metrics.py compare --row LABEL:JSON:FIRST ... --frames N      # per-frame values side by side
  cut_metrics.py first JSON[:FIRST] ... [--frames 8]                 # a shot's first frame vs its next ones
  cut_metrics.py join A.mkv B.mkv ... --out AB.mkv                   # lossless concatenation
  cut_metrics.py selftest GT.mkv --cut C [--out OUT.mkv]             # ghost coefficient on known mixes
  cut_metrics.py phase RUN.json ... --base BASE.json ... [--metrics psnr_y,lpips,vmaf]  # latent grid
  cut_metrics.py summary JSON ...                                     # Markdown table of analyses

Why: SeedVR2 packs 4 frames per latent and runs causal VAE passes and windowed DiT attention, so
a cut inside one processing unit (a batch, or a latent window) may carry one shot into the other.
A run with the cut inside is compared, frame by frame, with the aligned reference: the same
frames processed as two separate shots, A's output followed by B's (`join`), both scored against
the same ground truth by fr_metrics.py (its per-output JSONs are the inputs here).

Conventions: C is the clip index of the first frame of shot B (the frame after the cut). Frame t
is at signed distance d = t - C: d < 0 is shot A (-1 its last frame), d >= 0 shot B (0 its first
frame). A per-transition series (the temporal errors, frames t-1 -> t) is placed at the distance
of its second frame: d = 0 is the cut itself.

`analyze`:
- deficit per frame and metric: how much worse the run is than the reference, in the metric's
  unit, positive = worse: reference - run for PSNR-Y, SSIM-Y, VMAF (higher is better), run -
  reference for LPIPS, DISTS, CIEDE2000 lf and the temporal errors.
- noise band: the CLI is deterministic (a same-seed rerun is bit-identical, so it measures
  nothing); two seeds are two equally valid renderings of the same frames. --noise takes
  fr_metrics JSONs of renderings that differ only by their seed (the aligned reference with
  seeds 42 and 43, or question 1's seed runs): grouped by clip, variant and frame count, every
  pair of seeds in a group gives per-frame |difference|s, pooled; the threshold of a metric is
  their --q quantile (default 0.95). The run and reference JSONs join the groups they match.
  --thr METRIC=V sets a threshold by hand. A deficit at or below the threshold is within the band.
- --smooth N (default 1: off): the deficits, and the seed-to-seed differences of the noise band,
  are first averaged over N consecutive frames (centred; never across the cut). Two runs whose
  4-frame latent groups are out of phase (a shot run alone starts its own grid at its first frame)
  differ by a period-4 pattern unrelated to the cut; N = 4 averages it out. The reach, the counts
  and the * marks then use the smoothed deficits; the per-frame table still prints the raw ones.
- reach after the cut: the smallest r >= 0 such that the --stable (default 4, one latent)
  frames from C + r on are all within the band (frames past the clip's end count as within):
  frames C..C+r-1 hold the measurably worse stretch that starts at the cut, 0 = none. Before the
  cut, the same going back from C - 1: frames C-r..C-1. Also: the farthest exceedance on each side
  and the count of exceedances, which an isolated outlier far from the cut shows apart.
- ghost coefficient alpha_t = argmin_a ||out_t - ((1-a) GT_t + a GT_other)||^2 = <e, g> / <g, g>,
  with e = out_t - GT_t and g = GT_other - GT_t, on luma (BT.709 Y, 8-bit scale), at full
  resolution (block 1) and on B x B block means (--blocks, default 1,8: the means drop the
  detail the model invents, which the projection would otherwise pick up). GT_other is the other
  shot's frame nearest the cut: A's last frame for t >= C, B's first for t < C. The reference's
  alpha is the bias of an output without ghost (its errors correlate with the shots' difference,
  softness for instance); the excess, run - reference, is the ghost. The mean |g| per frame is
  reported: when the two shots barely differ, alpha is ill-conditioned (use --no-ghost for a
  false cut inside one shot).
- binned means of the deficits and alphas by distance, and a per-frame table (exceedances
  marked *).

`compare` prints, for N frames, the per-frame values of several outputs side by side, each row
taken from its own JSON from frame FIRST on (e.g. a short shot run alone, the same frames inside
the aligned reference, and inside a one-batch run).

`first` compares a shot's first frame (index FIRST of each JSON, default 0) with its next N
frames (--frames, default 8): per metric frame 0, the mean of frames 1..N and frame 0's deficit
d0 (positive = frame 0 worse, as in `analyze`); for the temporal errors, transition 0->1 against
the mean of transitions 2..N; and the sharpness, fr_clips' Laplacian variance of the luma
relative to the ground truth's, of frame 0 against frames 1..N and its jump from frame 0 to
frame 1 (read from the masters the JSONs name). The outputs of the same clip and variant that
differ by their seed give the seed noise of d0: |d0(seed a) - d0(seed b)| over every pair, its
--q quantile (default 0.95) and maximum. Made for --prepend_frames, which moves a shot's first
frame out of the lone first latent.

`phase` measures fidelity by a frame's place in the causal VAE's latent grid: a unit's first
frame (--first) is a latent of its own, then each latent holds 4 frames, places 0 to 3 (3 = the
group's last). Each run is compared frame by frame with its clip's baseline (--base, matched on
the JSONs' clip; a bicubic upscale, whose own values show whether the input has a pattern of its
own), oriented as the run's gain (> 0 = closer to the GT), then averaged per place over the frames
after the first and over the runs; plus the gap between places 3 and 1, per run.

`join` concatenates RGB masters (FFV1, same size and pixel format) into one FFV1 master of the
same format, without any pixel conversion, and checks it by framemd5 against its parts.

`selftest` checks the ghost coefficient on the ground truth's own frames: synthetic mixes
(1-a) GT_t + a GT_other quantised to 16 bits (a = 0, 0.1, 0.3, 0.5, 0.9) must give alpha = a,
also with Gaussian noise (2 levels); it also shows the bias of a blurred frame (no ghost) at
each block size and, with --out, alpha of mixes of the output's own frames (excess vs a).

Needs numpy and fr_clips.py (next to this script); scipy for selftest's blur; ffmpeg with ffv1.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fr_clips  # noqa: E402  (numpy-only helpers: RGB decode, luma, framemd5)

# key, title, higher is better, per transition, format of a value, format of a deficit
METRICS = [
    ("psnr_y", "PSNR-Y", True, False, "{:.2f}", "{:+.3f}"),
    ("ssim_y", "SSIM-Y", True, False, "{:.4f}", "{:+.4f}"),
    ("lpips", "LPIPS", False, False, "{:.4f}", "{:+.4f}"),
    ("dists", "DISTS", False, False, "{:.4f}", "{:+.4f}"),
    ("de00_lf", "ΔE00 lf", False, False, "{:.3f}", "{:+.3f}"),
    ("vmaf", "VMAF", True, False, "{:.2f}", "{:+.2f}"),
    ("vmaf_neg", "VMAF NEG", True, False, "{:.2f}", "{:+.2f}"),
    ("temporal_lf", "T-err lf", False, True, "{:.3f}", "{:+.3f}"),
    ("temporal_full", "T-err", False, True, "{:.3f}", "{:+.3f}"),
]
BY_KEY = {m[0]: m for m in METRICS}
BINS = [(-10**6, -17), (-16, -9), (-8, -5), (-4, -2), (-1, -1), (0, 0), (1, 1), (2, 3), (4, 7), (8, 15), (16, 31),
        (32, 10**6)]
RGB_TAGS = ["-colorspace", "rgb", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "pc"]
SETPARAMS = "setparams=color_primaries=bt709:color_trc=bt709:colorspace=gbr:range=pc"


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def load(path):
    with open(path, encoding="utf-8") as f:
        r = json.load(f)
    if "per_frame" not in r:
        raise SystemExit(f"{path}: not an fr_metrics.py result (no per_frame)")
    r["_file"] = path
    return r


def series(r, key):
    m = BY_KEY[key]
    v = (r.get("per_transition") if m[3] else r.get("per_frame") or {}).get(key)
    if not v:
        return None
    return np.array([np.nan if x is None else x for x in v], dtype=float)


def deficit(key, run, ref):
    return (ref - run) if BY_KEY[key][2] else (run - ref)


def bin_label(lo, hi):
    if lo == hi:
        return f"{lo:+d}"
    if lo <= -10**5:
        return f"≤ {hi:+d}"
    if hi >= 10**5:
        return f"≥ {lo:+d}"
    return f"{lo:+d}…{hi:+d}"


def fmt(v, f):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "–"
    return f.format(v)


# ------------------------------------------------------------------ luma and the ghost coefficient

def luma_frames(path):
    """Yield the BT.709 luma (8-bit scale, float32) of every frame of an RGB master."""
    for rgb, bits in fr_clips.rgb_frames(path):
        yield fr_clips.luma709(rgb, bits)


def blocks(y, b):
    if b == 1:
        return y
    h, w = y.shape[0] // b * b, y.shape[1] // b * b
    return y[:h, :w].reshape(h // b, b, w // b, b).mean(axis=(1, 3), dtype=np.float64).astype(np.float32)


def alpha(out_y, gt_y, other_y, b):
    """argmin_a ||out - ((1-a) gt + a other)||^2 on b x b block means: <e, g> / <g, g>."""
    e = blocks(out_y, b).astype(np.float64) - blocks(gt_y, b)
    g = blocks(other_y, b).astype(np.float64) - blocks(gt_y, b)
    gg = float(np.sum(g * g))
    return float(np.sum(e * g)) / gg if gg > 0 else float("nan")


def ghost(gt_path, outs, c, bsizes):
    """alpha per frame for each output (dict label -> path), and the mean |GT_other - GT_t|."""
    head = []
    for i, y in enumerate(luma_frames(gt_path)):  # the two frames next to the cut
        if i >= c - 1:
            head.append(y)
        if i >= c:
            break
    if len(head) != 2:
        raise SystemExit(f"{gt_path}: no frames {c - 1} and {c} around the cut")
    a_last, b_first = head
    res = {k: {b: [] for b in bsizes} for k in outs}
    shot_diff = []
    gens = {k: luma_frames(p) for k, p in outs.items()}
    for t, gy in enumerate(luma_frames(gt_path)):
        other = a_last if t >= c else b_first
        shot_diff.append(float(np.mean(np.abs(other - gy))))
        for k, gen in gens.items():
            oy = next(gen, None)
            if oy is None:
                raise SystemExit(f"{outs[k]}: fewer frames than the ground truth ({t})")
            for b in bsizes:
                res[k][b].append(alpha(oy, gy, other, b))
    for gen in gens.values():
        gen.close()
    return res, shot_diff


# ------------------------------------------------------------------ noise band

def smooth(v, n, cut=None):
    """Centred moving mean over n frames (NaN-aware), never across the cut (clip index `cut`)."""
    v = np.asarray(v, dtype=float)
    if n <= 1:
        return v
    out = np.full_like(v, np.nan)
    lo, hi = (n - 1) // 2, n // 2
    for a, b in ([(0, cut), (cut, len(v))] if cut else [(0, len(v))]):
        for t in range(a, b):
            w = v[max(a, t - lo):min(b, t + hi + 1)]
            w = w[np.isfinite(w)]
            if w.size:
                out[t] = w.mean()
    return out


def noise_band(jsons, keys, q, n_smooth=1):
    """Per metric: the q quantile of per-frame |differences| between renderings that differ only by
    their seed, pooled over every pair of seeds within each (clip, variant, frames) group."""
    groups = {}
    for r in jsons:
        groups.setdefault((r.get("clip"), r.get("variant"), r.get("frames")), {})[str(r.get("seed"))] = r
    pooled = {k: [] for k in keys}
    pairs = []
    for (clip, variant, n), by_seed in sorted(groups.items(), key=lambda kv: str(kv[0])):
        seeds = sorted(by_seed)
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                pairs.append(f"{clip}/{variant} s{seeds[i]}-s{seeds[j]}")
                for k in keys:
                    x, y = series(by_seed[seeds[i]], k), series(by_seed[seeds[j]], k)
                    if x is None or y is None:
                        continue
                    m = min(len(x), len(y))
                    d = np.abs(smooth(x[:m] - y[:m], n_smooth))
                    pooled[k].append(d[np.isfinite(d)])
    thr, count = {}, {}
    for k in keys:
        if pooled[k]:
            allv = np.concatenate(pooled[k])
            if allv.size:
                thr[k] = float(np.quantile(allv, q))
                count[k] = int(allv.size)
    return thr, count, pairs


def reach(defi, c, thr, stable):
    """Measurably worse stretches next to the cut (see the module docstring)."""
    n = len(defi)
    bad = [bool(np.isfinite(v) and v > thr) for v in defi]

    def calm(idx):  # `stable` frames from idx on (direction given by the caller) are within the band
        return all(not (0 <= i < n and bad[i]) for i in idx)

    after = next(r for r in range(0, n - c + 1) if calm(range(c + r, c + r + stable)))
    before = next(r for r in range(0, c + 1) if calm(range(c - 1 - r, c - 1 - r - stable, -1)))
    ex_a = [t - c for t in range(c, n) if bad[t]]
    ex_b = [c - t for t in range(0, c) if bad[t]]
    fin = [(v, t) for t, v in enumerate(defi) if np.isfinite(v)]
    peak, at = max(fin) if fin else (float("nan"), None)
    return {"thr": thr, "after": after, "before": before, "last_after": max(ex_a) if ex_a else None,
            "last_before": max(ex_b) if ex_b else None, "n_after": len(ex_a), "n_before": len(ex_b),
            "peak": peak, "peak_at": None if at is None else at - c}


# ------------------------------------------------------------------ analyze

def cmd_analyze(a):
    run, ref = load(a.run), load(a.ref)
    c = a.cut
    n = min(run["frames"], ref["frames"])
    if run["frames"] != ref["frames"]:
        log(f"warning: {run['frames']} frames in the run, {ref['frames']} in the reference: comparing {n}")
    if run.get("gt") != ref.get("gt"):
        log(f"warning: different ground truths: {run.get('gt')} / {ref.get('gt')}")
    if not 1 <= c < n:
        raise SystemExit(f"--cut {c}: must be inside 1..{n - 1}")
    keys = [k for k, *_ in METRICS if series(run, k) is not None and series(ref, k) is not None]
    dist = list(range(-c, n - c))
    res = {"label": a.label or f"{run.get('clip')}/{run.get('variant')}", "cut": c, "frames": n,
           "run": {k: run.get(k) for k in ("clip", "variant", "seed", "out", "gt", "_file")},
           "ref": {k: ref.get(k) for k in ("clip", "variant", "seed", "out", "gt", "_file")},
           "distance": dist, "metrics": {}, "reach": {}, "bins": {}}
    for k in keys:
        x, y = series(run, k), series(ref, k)
        if BY_KEY[k][3]:  # transitions 1..n-1 -> frames; frame 0 has none
            x, y = np.concatenate([[np.nan], x[:n - 1]]), np.concatenate([[np.nan], y[:n - 1]])
        x, y = x[:n], y[:n]
        d = deficit(k, x, y)
        res["metrics"][k] = {"run": x.tolist(), "ref": y.tolist(), "deficit": d.tolist(),
                             "deficit_smoothed": smooth(d, a.smooth, c).tolist(),
                             "higher_better": BY_KEY[k][2], "per_transition": BY_KEY[k][3]}
    # noise band
    thr, count, pairs = {}, {}, []
    if a.noise:
        thr, count, pairs = noise_band([load(p) for p in a.noise] + [run, ref], keys, a.q, a.smooth)
    for kv in a.thr or []:
        k, _, v = kv.partition("=")
        if k not in BY_KEY:
            raise SystemExit(f"--thr {kv}: unknown metric {k!r} ({', '.join(BY_KEY)})")
        thr[k] = float(v)
        count[k] = "set by hand"
    res["noise"] = {"q": a.q, "pairs": pairs, "thr": thr, "values": count, "stable": a.stable, "smooth": a.smooth}
    for k in keys:
        if k in thr:
            res["reach"][k] = reach(res["metrics"][k]["deficit_smoothed"], c, thr[k], a.stable)
    for lo, hi in BINS:
        idx = [t for t, d in enumerate(dist) if lo <= d <= hi]
        if not idx:
            continue
        res["bins"][bin_label(lo, hi)] = {k: float(np.nanmean([res["metrics"][k]["deficit"][t] for t in idx]))
                                          if any(np.isfinite(res["metrics"][k]["deficit"][t]) for t in idx)
                                          else None for k in keys}
    # ghost coefficient
    bsizes = [int(b) for b in a.blocks.split(",") if b.strip()]
    if not a.no_ghost:
        gt = a.gt or run.get("gt")
        outs = {"run": a.out or run.get("out"), "ref": a.ref_out or ref.get("out")}
        for k, p in [("gt", gt), *outs.items()]:
            if not p or not os.path.exists(p):
                raise SystemExit(f"ghost: {k} file {p!r} not found (give --gt/--out/--ref-out, or --no-ghost)")
        if run.get("rows") or ref.get("rows"):
            raise SystemExit("ghost: the JSONs were scored on a row range (--rows): not supported, use --no-ghost")
        al, shot_diff = ghost(gt, outs, c, bsizes)
        res["ghost"] = {"blocks": bsizes, "shot_diff": shot_diff[:n],
                        "run": {str(b): al["run"][b][:n] for b in bsizes},
                        "ref": {str(b): al["ref"][b][:n] for b in bsizes},
                        "excess": {str(b): (np.array(al["run"][b][:n]) - np.array(al["ref"][b][:n])).tolist()
                                   for b in bsizes}}
        for lo, hi in BINS:
            idx = [t for t, d in enumerate(dist) if lo <= d <= hi]
            if idx:
                for b in bsizes:
                    res["bins"][bin_label(lo, hi)][f"alpha{b}_excess"] = float(
                        np.nanmean([res["ghost"]["excess"][str(b)][t] for t in idx]))
    report(res, keys, bsizes)
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(res, f, allow_nan=True)
        print(f"\n-> {a.json}")


def report(res, keys, bsizes):
    c, n = res["cut"], res["frames"]
    print(f"## {res['label']}: cut before frame {c} ({c} frames of A, {n - c} of B)\n")
    print(f"Run `{res['run']['variant']}` s{res['run']['seed']} ({res['run']['out']}) vs reference "
          f"`{res['ref']['variant']}` s{res['ref']['seed']} ({res['ref']['out']}); deficit = how much worse the "
          f"run is (positive = worse).\n")
    nz = res["noise"]
    if nz["thr"]:
        sm = nz.get("smooth", 1)
        print(f"Noise band: {nz['q']:.2f} quantile of per-frame |seed-to-seed differences| "
              f"({len(nz['pairs'])} pairs: {', '.join(nz['pairs']) or 'none'}); stable = {nz['stable']} frames"
              + (f"; deficits and differences averaged over {sm} frames (not across the cut)" if sm > 1 else "")
              + ".\n")
        print("| Metric | Threshold (values) | Worse before the cut: reach / farthest / count | "
              "Worse after the cut: reach / farthest / count | Peak deficit (at d) |")
        print("|---|---|---|---|---|")
        for k in keys:
            r = res["reach"].get(k)
            if not r:
                continue
            f = BY_KEY[k][5]
            last_b = r["last_before"] if r["last_before"] is not None else "–"
            last_a = r["last_after"] if r["last_after"] is not None else "–"
            peak = f"{fmt(r['peak'], f)} ({r['peak_at']:+d})" if r["peak_at"] is not None else "–"
            print(f"| {BY_KEY[k][1]} | {fmt(r['thr'], f.replace('+', ''))} ({nz['values'].get(k)}) | "
                  f"{r['before']} / {last_b} / {r['n_before']} | {r['after']} / {last_a} / {r['n_after']} | {peak} |")
        print()
    else:
        print("No noise band (no --noise, no --thr): reach not computed.\n")
    gh = res.get("ghost")
    print("Mean deficit by distance to the cut" + (", and the ghost coefficient's excess (run − reference)"
                                                  if gh else "") + ":\n")
    cols = [BY_KEY[k][1] for k in keys] + ([f"α{b} excess" for b in bsizes] if gh else [])
    print("| Distance | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for lab, v in res["bins"].items():
        cells = [fmt(v.get(k), BY_KEY[k][5]) for k in keys]
        if gh:
            cells += [fmt(v.get(f"alpha{b}_excess"), "{:+.3f}") for b in bsizes]
        print(f"| {lab} | " + " | ".join(cells) + " |")
    sm = res["noise"].get("smooth", 1)
    print("\nPer frame (raw deficits; * = beyond the noise band" + (f", on the {sm}-frame average" if sm > 1 else "")
          + "):\n")
    show = [k for k in ("psnr_y", "ssim_y", "lpips", "dists", "de00_lf", "vmaf", "temporal_lf") if k in keys]
    head = ["d", "PSNR-Y run"] + [f"Δ {BY_KEY[k][1]}" for k in show]
    if gh:
        head += [f"α{b} run / ref" for b in bsizes] + ["mean |GT_o − GT|"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    m = res["metrics"]
    for t in range(n):
        cells = [f"{t - c:+d}", fmt(m["psnr_y"]["run"][t], "{:.2f}") if "psnr_y" in m else "–"]
        for k in show:
            v = m[k]["deficit"][t]
            vs = m[k].get("deficit_smoothed", m[k]["deficit"])[t]
            star = "*" if k in res["reach"] and np.isfinite(vs) and vs > res["reach"][k]["thr"] else ""
            cells.append(fmt(v, BY_KEY[k][5]) + star)
        if gh:
            cells += [f"{fmt(gh['run'][str(b)][t], '{:+.3f}')} / {fmt(gh['ref'][str(b)][t], '{:+.3f}')}" for b in bsizes]
            cells.append(fmt(gh["shot_diff"][t], "{:.1f}"))
        print("| " + " | ".join(cells) + " |")


# ------------------------------------------------------------------ compare

def cmd_compare(a):
    rows = []
    for spec in a.row:
        label, path, first = spec.rsplit(":", 2) if spec.count(":") >= 2 else (None, None, None)
        if label is None:
            raise SystemExit(f"--row {spec}: LABEL:JSON:FIRST expected")
        rows.append((label, load(path), int(first)))
    n = a.frames
    keys = [k for k, *_ in METRICS if not BY_KEY[k][3] and all(series(r, k) is not None for _, r, _ in rows)]
    print(f"Per-frame values over {n} frames (each row from its JSON's frame FIRST on):\n")
    for k in keys:
        f = BY_KEY[k][4]
        print(f"**{BY_KEY[k][1]}** ({'higher' if BY_KEY[k][2] else 'lower'} is better)\n")
        print("| Output | First | " + " | ".join(f"+{i}" for i in range(n)) + " | Mean |")
        print("|---|---|" + "---|" * (n + 1))
        for label, r, first in rows:
            s = series(r, k)
            v = s[first:first + n]
            if len(v) < n:
                raise SystemExit(f"{r['_file']}: {len(s)} frames, {first}+{n} needed")
            print(f"| {label} | {first} | " + " | ".join(fmt(x, f) for x in v) + f" | {fmt(float(np.nanmean(v)), f)} |")
        print()


# ------------------------------------------------------------------ join

def cmd_join(a):
    infos = [fr_clips.probe(p) for p in a.parts]
    fmts = {(i["pix_fmt"], i["width"], i["height"]) for i in infos}
    if len(fmts) != 1:
        raise SystemExit(f"the parts differ in pixel format or size: {sorted(fmts)}")
    fmt_, w, h = fmts.pop()
    if fmt_ not in fr_clips.RGB_BITS:
        raise SystemExit(f"{fmt_}: RGB masters expected")
    k = len(a.parts)
    ins = [x for p in a.parts for x in ("-i", p)]
    graph = "".join(f"[{i}:v:0]" for i in range(k)) + f"concat=n={k}:v=1:a=0,{SETPARAMS}[v]"
    cmd = [fr_clips.FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", *ins, "-filter_complex", graph,
           "-map", "[v]", "-fps_mode", "passthrough", *fr_clips.FFV1, "-threads", str(a.threads), "-pix_fmt", fmt_,
           *RGB_TAGS, a.out]
    fr_clips.run(cmd, "join")
    parts = [fr_clips.framemd5(p, None, a.threads) for p in a.parts]
    want = [h_ for hs in parts for h_ in hs]
    got = fr_clips.framemd5(a.out, None, a.threads)
    same = sum(x == y for x, y in zip(want, got))
    ok = len(want) == len(got) == same
    print(f"{a.out}: {len(got)} frames ({' + '.join(str(len(hs)) for hs in parts)}), {fmt_} {w}x{h}; "
          f"framemd5 {same}/{len(want)} equal to the parts: {'OK' if ok else 'DIFFERS'}")
    sys.exit(0 if ok else 1)


# ------------------------------------------------------------------ selftest

def rgb_float(path, want):
    """The RGB frames whose indices are in `want`, as float64 [H, W, 3] in [0, 1]."""
    out = {}
    for i, (rgb, bits) in enumerate(fr_clips.rgb_frames(path)):
        if i in want:
            out[i] = rgb.astype(np.float64) / ((1 << bits) - 1)
        if i >= max(want):
            break
    return out


def luma_of(rgb01, quant=16):
    q = np.rint(np.clip(rgb01, 0, 1) * ((1 << quant) - 1))
    return fr_clips.luma709(q.astype(np.uint16), quant)


def cmd_selftest(a):
    c = a.cut
    n = fr_clips.count_frames(a.gt)
    test = sorted({t for t in (c - 3, c - 1, c, c + 1, c + 3, c + 10) if 0 <= t < n})
    gt = rgb_float(a.gt, set(test) | {c - 1, c})
    bsizes = [int(b) for b in a.blocks.split(",")]
    rng = np.random.default_rng(1)
    ok = True
    worst = {"mix": 0.0, "noise": 0.0}
    print(f"{a.gt}: cut before frame {c}; test frames {test}; GT_other = frame {c - 1} for t >= {c}, {c} below\n")
    print("| Frame t | a | " + " | ".join(f"α{b} mix" for b in bsizes) + " | "
          + " | ".join(f"α{b} mix + noise 2" for b in bsizes) + " |")
    print("|---|---|" + "---|" * (2 * len(bsizes)))
    for t in test:
        o = c - 1 if t >= c else c
        gy = luma_of(gt[t])
        oy = luma_of(gt[o])
        for av in (0.0, 0.1, 0.3, 0.5, 0.9):
            mix = (1 - av) * gt[t] + av * gt[o]
            mixn = mix + rng.normal(0, 2 / 255, size=mix.shape)
            al = [alpha(luma_of(mix), gy, oy, b) for b in bsizes]
            aln = [alpha(luma_of(mixn), gy, oy, b) for b in bsizes]
            worst["mix"] = max(worst["mix"], *(abs(x - av) for x in al))
            worst["noise"] = max(worst["noise"], *(abs(x - av) for x in aln))
            print(f"| {t} ({t - c:+d}) | {av} | " + " | ".join(f"{x:.4f}" for x in al) + " | "
                  + " | ".join(f"{x:.4f}" for x in aln) + " |")
    good = worst["mix"] <= 0.002 and worst["noise"] <= 0.01
    ok &= good
    print(f"\nlargest |alpha - a|: {worst['mix']:.5f} on the mixes (limit 0.002), {worst['noise']:.5f} with noise "
          f"(limit 0.01): {'OK' if good else 'FAILED'}")
    try:
        from scipy.ndimage import gaussian_filter
        print("\nNo ghost, a blurred ground truth (Gaussian sigma 1.5 px): the bias block means remove\n")
        print("| Frame t | " + " | ".join(f"α{b}" for b in bsizes) + " |")
        print("|---|" + "---|" * len(bsizes))
        for t in test:
            o = c - 1 if t >= c else c
            bl = np.stack([gaussian_filter(gt[t][..., k], 1.5) for k in range(3)], axis=-1)
            print(f"| {t} ({t - c:+d}) | " + " | ".join(
                f"{alpha(luma_of(bl), luma_of(gt[t]), luma_of(gt[o]), b):+.4f}" for b in bsizes) + " |")
    except ImportError:
        print("\n(scipy missing: blur bias not shown)")
    if a.out:
        out = rgb_float(a.out, set(test) | {c - 1, c})
        print(f"\nMixes of the output's own frames ({a.out}): (1-a) out_t + a out_other; excess = alpha(mix) - "
              f"alpha(out_t), the ghost the coefficient sees when the ghost is made of rendered frames\n")
        print("| Frame t | α of out_t | " + " | ".join(f"a = {av}: " + " / ".join(f"α{b}" for b in bsizes)
                                                    for av in (0.1, 0.3, 0.5)) + " |")
        print("|---|---|" + "---|" * 3)
        worst_o = 0.0
        for t in test:
            o = c - 1 if t >= c else c
            gy, oy = luma_of(gt[t]), luma_of(gt[o])
            base = [alpha(luma_of(out[t]), gy, oy, b) for b in bsizes]
            cells = []
            for av in (0.1, 0.3, 0.5):
                mix = (1 - av) * out[t] + av * out[o]
                ex = [alpha(luma_of(mix), gy, oy, b) - b0 for b, b0 in zip(bsizes, base)]
                worst_o = max(worst_o, abs(ex[-1] - av))
                cells.append(" / ".join(f"{x:.3f}" for x in ex))
            print(f"| {t} ({t - c:+d}) | " + " / ".join(f"{x:+.3f}" for x in base) + " | " + " | ".join(cells) + " |")
        print(f"\nlargest |excess - a| at block {bsizes[-1]}: {worst_o:.3f} (information: the rendered other frame "
              f"differs from its ground truth)")
    print("\nselftest", "ok" if ok else "FAILED")
    sys.exit(0 if ok else 1)


# ------------------------------------------------------------------ first

FIRST_KEYS = ("psnr_y", "lpips", "dists", "vmaf", "temporal_full", "temporal_lf")


def lap_vars(path, first, n):
    """fr_clips.lap_var of the luma of frames first..first+n of a master."""
    out, gen = [], luma_frames(path)
    try:
        for i, y in enumerate(gen):
            if i >= first:
                out.append(fr_clips.lap_var(y))
            if i >= first + n:
                break
    finally:
        gen.close()
    return np.array(out)


def first_frame(r, first, n):
    """A shot's first frame (index `first` of an fr_metrics result) against its next n frames."""
    res = {}
    for k in FIRST_KEYS:
        s = series(r, k)
        if s is None or len(s) < first + n + 1:
            continue
        s = s[first:]
        if BY_KEY[k][3]:  # per transition: 0->1 against 2..n
            f0, rest = float(s[1]), float(np.nanmean(s[2:n + 1]))
            res[k] = {"f0": f0, "rest": rest, "d0": f0 - rest}
        else:
            f0, rest = float(s[0]), float(np.nanmean(s[1:n + 1]))
            res[k] = {"f0": f0, "rest": rest, "d0": (rest - f0) if BY_KEY[k][2] else (f0 - rest)}
    ratio = lap_vars(r["out"], first, n) / lap_vars(r["gt"], first, n)
    res["sharp"] = {"f0": float(ratio[0]), "rest": float(ratio[1:].mean()),
                    "d0": float(ratio[0] / ratio[1:].mean() - 1), "step": float(ratio[1] / ratio[0] - 1)}
    return res


def cmd_first(a):
    n, rows = a.frames, []
    for item in a.runs:
        path, _, first = item.partition(":")
        r = load(path)
        rows.append((r, int(first or 0), first_frame(r, int(first or 0), n)))
    keys = [k for k in FIRST_KEYS if all(k in x for _, _, x in rows)]
    head = [BY_KEY[k][1] + (f": 0->1 / 2-{n} (excess)" if BY_KEY[k][3] else f": frame 0 / 1-{n} (d0)") for k in keys]
    print(f"### A shot's first frame against its next {n} frames (d0 > 0: frame 0 worse)\n")
    print("| Output | first | " + " | ".join(head)
          + f" | Sharpness vs the GT's: frame 0 / 1-{n} (frame 0 vs 1-{n}; jump 0->1) |")
    print("|---|---|" + "---|" * (len(keys) + 1))
    for r, first, x in rows:
        cells = [f"{fmt(x[k]['f0'], BY_KEY[k][4])} / {fmt(x[k]['rest'], BY_KEY[k][4])} ({fmt(x[k]['d0'], BY_KEY[k][5])})"
                 for k in keys]
        s = x["sharp"]
        cells.append(f"{s['f0']:.3f} / {s['rest']:.3f} ({s['d0']:+.0%}; {s['step']:+.0%})")
        print(f"| {r.get('clip')} {r.get('variant')} s{r.get('seed')} | {first} | " + " | ".join(cells) + " |")
    groups = {}
    for r, first, x in rows:
        groups.setdefault((r.get("clip"), r.get("variant"), first), {})[str(r.get("seed"))] = x
    diffs = {k: [] for k in keys + ["sharp", "step"]}
    for g in groups.values():
        seeds = sorted(g)
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                x, y = g[seeds[i]], g[seeds[j]]
                for k in keys:
                    diffs[k].append(abs(x[k]["d0"] - y[k]["d0"]))
                diffs["sharp"].append(abs(x["sharp"]["d0"] - y["sharp"]["d0"]))
                diffs["step"].append(abs(x["sharp"]["step"] - y["sharp"]["step"]))
    if diffs["sharp"]:
        names = {"sharp": "sharpness of frame 0", "step": "sharpness jump"}
        print(f"\nSeed noise of d0, |d0(seed a) - d0(seed b)| over {len(diffs['sharp'])} pairs (q{a.q:g} / max):\n")
        for k, v in diffs.items():
            print(f"- {BY_KEY[k][1] if k in BY_KEY else names[k]}: {np.quantile(v, a.q):.4g} / {max(v):.4g}")


# ------------------------------------------------------------------ phase

PHASE_KEYS = "psnr_y,lpips,vmaf"


def latent_pos(t, first=0):
    """Frame t's place in the causal VAE's latent grid of a unit starting at `first`: the unit's
    first frame is a latent of its own (-1), then 0…3 in each group of 4 (3 = the group's last)."""
    d = t - first
    return -1 if d == 0 else (d - 1) % 4


def cmd_phase(a):
    bases = {}
    for p in a.base:
        b = load(p)
        bases[b.get("clip")] = b
    groups = {}
    for p in a.runs:
        r = load(p)
        if r.get("clip") not in bases:
            raise SystemExit(f"{p}: no --base for clip {r.get('clip')!r}")
        groups.setdefault(r.get("clip"), []).append(r)
    for k in [k for k in a.metrics.split(",") if k in BY_KEY and not BY_KEY[k][3]]:
        name, up, f, df = BY_KEY[k][1], BY_KEY[k][2], BY_KEY[k][4], BY_KEY[k][5]
        print(f"\n### {name} by place in the 4-frame latent grid (frames after the first; 3 = a group's last)\n")
        print("Run − baseline, oriented (> 0: the run closer to the GT), mean over frames and runs; the gap "
              "between places 3 and 1 per run (its range over the runs); the baseline's own values.\n")
        print("| clip | runs | 0 | 1 | 2 | 3 | 3 − 1 (range) | baseline 0 / 1 / 2 / 3 |")
        print("|---|---:|---:|---:|---:|---:|---|---|")
        for clip, runs in groups.items():
            b = series(bases[clip], k)
            n = len(b)
            pos = [latent_pos(t, a.first) for t in range(n)]
            per = {p: [] for p in range(4)}
            gaps = []
            for r in runs:
                s = series(r, k)[:n]
                d = (s - b) if up else (b - s)
                m = {p: float(np.nanmean([d[t] for t in range(n) if pos[t] == p])) for p in range(4)}
                for p in range(4):
                    per[p] += [d[t] for t in range(n) if pos[t] == p]
                gaps.append(m[3] - m[1])
            bp = [float(np.nanmean([b[t] for t in range(n) if pos[t] == p])) for p in range(4)]
            print(f"| {clip} | {len(runs)} | " + " | ".join(df.format(float(np.nanmean(per[p]))) for p in range(4)) +
                  f" | {df.format(float(np.mean(gaps)))} ({df.format(min(gaps))}…{df.format(max(gaps))}) | " +
                  " / ".join(f.format(x) for x in bp) + " |")


# ------------------------------------------------------------------ summary

def cmd_summary(a):
    files = []
    for p in a.files:
        files += sorted(os.path.join(p, f) for f in os.listdir(p) if f.endswith(".json")) if os.path.isdir(p) else [p]
    res = []
    for p in files:
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        if "cut" in r and "metrics" in r:
            res.append(r)
    if not res:
        raise SystemExit("no analysis JSON")
    keys = [k for k in ("psnr_y", "ssim_y", "lpips", "dists", "de00_lf", "vmaf") if any(k in r["metrics"] for r in res)]
    print("Reach of the measurably worse stretch next to the cut, frames before / after (– = no noise band), "
          "mean deficit over B's first 4 frames (d = 0…3), and the ghost coefficient's excess at d = −1, 0 and "
          "its largest value on B (8×8 block means):\n")
    head = ["Analysis", "Cut"] + [f"{BY_KEY[k][1]} reach" for k in keys] + [f"{BY_KEY[k][1]} Δ d0…3" for k in keys]
    head += ["α8 excess d −1 / 0", "α8 excess max on B (d)"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in res:
        c = r["cut"]
        cells = [r["label"], str(c)]
        for k in keys:
            rr = r["reach"].get(k)
            cells.append(f"{rr['before']} / {rr['after']}" if rr else "–")
        for k in keys:
            d = r["metrics"].get(k, {}).get("deficit")
            cells.append(fmt(float(np.nanmean(d[c:c + 4])), BY_KEY[k][5]) if d else "–")
        g = r.get("ghost")
        if g and "8" in g["excess"]:
            ex = g["excess"]["8"]
            b = [(v, t - c) for t, v in enumerate(ex) if t >= c and v is not None and np.isfinite(v)]
            mx = max(b) if b else (float("nan"), 0)
            cells += [f"{fmt(ex[c - 1], '{:+.3f}')} / {fmt(ex[c], '{:+.3f}')}", f"{fmt(mx[0], '{:+.3f}')} ({mx[1]:+d})"]
        else:
            cells += ["–", "–"]
        print("| " + " | ".join(cells) + " |")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("analyze", help="a run with the cut inside vs the aligned reference")
    s.add_argument("run", help="fr_metrics JSON of the run with the cut inside")
    s.add_argument("--ref", required=True, help="fr_metrics JSON of the reference (the aligned shots, joined)")
    s.add_argument("--cut", type=int, required=True, help="clip index of shot B's first frame")
    s.add_argument("--noise", nargs="+", metavar="JSON", help="fr_metrics JSONs of seed variants (noise band)")
    s.add_argument("--q", type=float, default=0.95, help="noise band quantile (default 0.95)")
    s.add_argument("--thr", action="append", metavar="METRIC=V", help="threshold by hand (repeatable)")
    s.add_argument("--stable", type=int, default=4, help="frames within the band that end a stretch (default 4)")
    s.add_argument("--smooth", type=int, default=1,
                   help="average deficits and seed differences over N frames first, not across the cut (4: one "
                        "latent group, against latent-phase patterns; default 1: off)")
    s.add_argument("--blocks", default="1,8", help="ghost coefficient block sizes (default 1,8)")
    s.add_argument("--no-ghost", action="store_true", help="skip the ghost coefficient (e.g. a false cut)")
    s.add_argument("--gt", help="ground truth (default: the run JSON's)")
    s.add_argument("--out", help="the run's master (default: the run JSON's)")
    s.add_argument("--ref-out", help="the reference's master (default: the reference JSON's)")
    s.add_argument("--label")
    s.add_argument("--json", help="write the analysis here")
    s = sub.add_parser("compare", help="per-frame values of several outputs side by side")
    s.add_argument("--row", action="append", required=True, metavar="LABEL:JSON:FIRST")
    s.add_argument("--frames", type=int, required=True)
    s = sub.add_parser("first", help="a shot's first frame against its next frames, with the seed noise of that")
    s.add_argument("runs", nargs="+", metavar="JSON[:FIRST]",
                   help="fr_metrics JSONs; FIRST = index of the shot's first frame in it (default 0)")
    s.add_argument("--frames", type=int, default=8, help="next frames compared (default 8)")
    s.add_argument("--q", type=float, default=0.95, help="seed-noise quantile (default 0.95)")
    s = sub.add_parser("join", help="lossless concatenation of RGB masters, checked by framemd5")
    s.add_argument("parts", nargs="+")
    s.add_argument("--out", required=True)
    s.add_argument("--threads", type=int, default=16)
    s = sub.add_parser("selftest", help="the ghost coefficient on synthetic mixes of ground-truth frames")
    s.add_argument("gt")
    s.add_argument("--cut", type=int, required=True)
    s.add_argument("--out", help="an output of the same clip: also mixes of its own frames")
    s.add_argument("--blocks", default="1,8")
    s = sub.add_parser("phase", help="fidelity by place in the 4-frame latent grid, against a baseline")
    s.add_argument("runs", nargs="+", help="fr_metrics JSONs of the runs (e.g. a variant's seeds), any clips")
    s.add_argument("--base", action="append", required=True,
                   help="fr_metrics JSON of a clip's baseline (e.g. bicubic), matched on its clip; repeatable")
    s.add_argument("--metrics", default=PHASE_KEYS, help=f"per-frame metrics (default {PHASE_KEYS})")
    s.add_argument("--first", type=int, default=0, help="index of the unit's first frame (default 0)")
    s = sub.add_parser("summary", help="Markdown table of analysis JSONs (files or directories)")
    s.add_argument("files", nargs="+")
    a = ap.parse_args()
    {"analyze": cmd_analyze, "compare": cmd_compare, "first": cmd_first, "join": cmd_join,
     "selftest": cmd_selftest, "phase": cmd_phase, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Model conversation, slice A (1080p x2 from d1, 45 frames, seed 42): each phase-2 file's runs against the 7B fp16,
per clip and metric, then PASS / FAIL per guard and kind of source; one Markdown file per label, an overview, the
7B fp16's own band, and (label valsharp) the validation against colour's baton-2 T4.

  ms_sum.py [--out $VAL_STATE/sum] [--t4 $COLOUR_OUT/sum-b2/T4-1080p.md]

Sources: the 7B fp16 at seed 42 = colour's own scores (eval-b2/<cd>, vmaf-b2/<cd>, bands-b2/<cd>-7b.json: what
colour's T4 paired the sharp 7B with); its seeds 43 and 1234 = ours, the same commands (gpu/eval/7b, gpu/vmaf/7b,
gpu/bands/7b); a label's = gpu/eval/<label>, gpu/vmaf/<label>, gpu/bands/<label> (ref tag = the label, content
s42); fr_metrics.py's JSONs gpu/fr/<label>/<cd> (7B: gpu/fr/7b). JSONs of one clip merged as colour_eval.py
summary merges them (per variant@ref and content, score's and vmaf's series together, the later file's on a
series both hold).
Pairing: colour_eval.py summary --baseline <variant>@f32 on a view holding one clip's label and 7B JSONs,
replicated in-process with colour_eval's own series() and boot(): the label's frames minus the 7B's at the seeds
both hold (42), mean, 95% moving-block bootstrap interval (blocks of 8, 2000 draws, rng seed 1 drawn in colour_eval's
METRICS order; SSIM-Y with its own rng seed 1, as colour's sum_b2.py). B / W when the interval excludes 0 AND |mean|
exceeds the band: the 7B fp16's 3-seed spread (max - min of its per-seed means, seeds 42, 43, 1234; fewer seeds
present: said in the cell's file). Detail (luma Laplacian variance): the same rule, up / down. "Colour mode" (the
band from seed 42 alone, 0: colour's T4) is used only by the validation.
Guards per kind of source: FAIL when a guard is worse on at least one clip of that kind: banding (CAMBI added W),
flicker (T-err or T-err lf W), colour drift (dE00 lf W), detail (Laplacian beyond, either way), band energy
(colour_bands.py's finest band, rms / the GT's: no interval, so the spread alone decides, as DISTS 5f: |the
file's - the 7B's at seed 42| beyond the 7B's seed spread, either way; until 2026-10-07 05:30 'outside the 7B's
3-seed range', which fails a 4th fp16 seed half the time), LPIPS W, DISTS W
(fr_metrics.py's, every frame, where present; else colour_eval's every 9th frame, whose 5-frame interval is
degenerate: the band alone decides). Fidelity (PSNR-Y, SSIM-Y, VMAF) reported, not a guard.
Two rules, both reported (since 2026-10-07 07:15): strict (the above) and calibrated = a strict failure whose |difference|
also exceeds K x the spread, K = KCAL[number of seeds in the spread] (2.7 for 3, 10.9 for 2): the multiple a further
fp16 seed exceeds with 5% probability per clip-variant (S4's Monte Carlo, Gaussian seed scatter, both directions).
Every W / up / down cell shows its multiple of the spread.
The sharp's files (sh-*, since S8 2026-10-07 11:40): paired with the sharp 7B fp16 at seed 42; their band = the
sharp 7B fp16's OWN 3-seed spread (seed 42 = colour's sh42, 43 and 1234 = ours: gpu/eval/sharp, gpu/vmaf/sharp,
gpu/bands/sharp/<cd>-s<seed>.json, gpu/fr/sharp, gpu/diff/sharp), per clip-variant and per source (scores + VMAF,
DISTS 45f, band energy) where all 3 seeds are scored, the 7B fp16's 3-seed spread elsewhere (column Band; each
sh-* page and the overview say which); the sharp's own band: sharp-band.md.
The 3B (labels 3b-*, S16 2026-10-08: 3b-cur = our seedvr2x_ema_3b_fp16 from ByteDance's current 3B weights, 3b-first =
numz's seedvr2_ema_3b_fp16, the first weights): scored and paired as any file, against the 7B fp16 and its band; the
3B has no seed band of its own: its page and overview row say so (informative only); the 3B against the sharp's 4 GB
pick: pair3b.py -> 3b-00-overview.md. Nothing else changed (verdicts, other pages: as before S16).
S23 (2026-10-08, design's decision of 10:20): the floor at 1080p too, one rule for both resolutions (ms_floor.py, as
ms_sum4k.py and ms_sum4ksh.py since S16): the spread used = max(the 3-seed spread, one unit of the resolution the
metric is printed at), in both rules and every printed multiple (a multiple of the floor carries a †); main() applies
it to every label's label_data() (3b-* and valsharp included) and to our Q4_K against numz's Q4_K_M (x_md); not
label_data() itself (imported by the 4K summaries, the pairings and floor_report.py), nor the validation's colour
mode (band 0) or the band pages (the seeds' own spreads).
"""

import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np
from glue_env import env  # noqa: E402  the paths: glue.env (models/gpu/validation/glue.env.example)

SCRIPTS = env("COLOUR_SCRIPTS")
sys.path.insert(0, SCRIPTS)
import colour_eval as E  # noqa: E402  series, boot, HIGHER, METRICS, diff_fmt; colour_diag as E.D
import ms_floor as FL  # noqa: E402  S23: the floor at 1080p too

O = env("COLOUR_OUT")
G = env("VAL_DATA") + "/gpu"
SC = env("VAL_STATE") + "/score"
CLIPS = env("MEAS_CLIPS")
GLUE = os.path.dirname(os.path.abspath(__file__))  # this folder: glue.env.example
CLIPS_A = (
    "anime-clean",
    "anime-grain",
    "anime-sky",
    "anime-bright",
    "cartoon-bright",
    "live-vfx",
    "live-slow",
    "anime-dark",
)
KIND = {
    "anime-clean": "anime",
    "anime-grain": "anime",
    "anime-sky": "anime",
    "anime-bright": "anime",
    "cartoon-bright": "cartoon",
    "live-vfx": "live action",
    "live-slow": "live action",
    "anime-dark": "dark",
}
KINDS = ("anime", "cartoon", "live action", "dark")
SP = "split:ycc:4:3"
VARS = (("none", "none"), (SP, "split"))
SEEDS = ("s42", "s43", "s1234")
NQ4KM = ("anime-sky", "anime-bright", "live-vfx", "live-slow")
PARIS = timezone(timedelta(hours=2))
FR_KEYS = ("dists", "lpips", "psnr_y", "psnr_y_bottom16", "psnr_y_rest")
FR_HIGHER = {"psnr_y", "psnr_y_bottom16", "psnr_y_rest"}
COLS = (
    ("psnr_y", "PSNR-Y"),
    ("ssim_y", "SSIM-Y"),
    ("vmaf", "VMAF"),
    ("lpips", "LPIPS"),
    ("dists", "DISTS 5f"),
    ("dists_all", "DISTS 45f"),
    ("cambi_added", "CAMBI+"),
    ("de4", "ΔE00 lf"),
    ("t_full", "T-err"),
    ("t_lf", "T-err lf"),
)
T2M = (
    ("psnr_y", "PSNR-Y (dB)"),
    ("ssim_y", "SSIM-Y"),
    ("lpips", "LPIPS"),
    ("dists", "DISTS"),
    ("vmaf", "VMAF"),
    ("cambi_added", "CAMBI added"),
    ("de0", "ΔE00 σ0"),
    ("de4", "ΔE00 lf (σ4)"),
    ("de16", "ΔE00 σ16"),
    ("t_full", "T-err"),
    ("t_lf", "T-err lf"),
    ("fringe", "Fringes"),
)
GUARDS = (
    ("banding", "Banding: CAMBI added"),
    ("flicker", "Flicker: T-err, T-err lf"),
    ("drift", "Colour drift: ΔE00 lf"),
    ("detail", "Detail: Laplacian ÷ GT's"),
    ("bands", "Band energy: finest ÷ GT's"),
    ("lpips", "LPIPS"),
    ("dists", "DISTS"),
)
X = {"skipped": [], "cache_hits": 0, "cache_miss": 0}
# The calibrated rule's K by the number of seeds in the 7B fp16's spread: |X_new - X_42| / (max - min of n seeds) of a
# further fp16 seed exceeds K with 5% probability (S4's Monte Carlo, Gaussian seed scatter, both directions).
KCAL = {3: 2.7, 2: 10.9}


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def minus(s):
    return s.replace("-", "−")


def fmt(k, v):
    if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
        return "–"
    return (
        minus(E.diff_fmt("ssim_y" if k == "dists_all" else k).format(v))
        if k != "dists_all"
        else minus("{:+.4f}".format(v))
    )


def f2(v, spec="{:.2f}"):
    if v is not None and v == math.inf:
        return "inf"  # ffv1_out.py --diff of two identical masters
    return "–" if v is None or not math.isfinite(v) else minus(spec.format(v))


# ---------------------------------------------------------------- loading


def stat_key(paths):
    h = hashlib.sha1()
    for p in sorted(paths):
        try:
            st = os.stat(p)
            h.update(f"{p}|{st.st_size}|{int(st.st_mtime)}\n".encode())
        except OSError:
            h.update(f"{p}|missing\n".encode())
    return h.hexdigest()


def load_dir(d):
    runs = []
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            with open(p, encoding="utf-8") as f:
                r = json.load(f)
        except (OSError, ValueError):
            X["skipped"].append(p)
            continue
        if "block_gradient" in r or "per_frame" not in r:
            continue
        r["_path"] = p
        runs.append(r)
    return runs


def merged(runs):
    """colour_eval.summary's merging for one clip: variant@ref -> content -> run (series of later files win)."""
    by = defaultdict(dict)
    for r in runs:
        k = f"{r['variant']}@{r['ref']}"
        prev = by[k].get(r["content"])
        r["_paths"] = [r["_path"]]
        if prev is not None:
            for part in ("per_frame", "per_transition"):
                r[part] = {**prev.get(part, {}), **r.get(part, {})}
            r["_paths"] = prev["_paths"] + r["_paths"]
        by[k][r["content"]] = r
    return by


def clip_sources(cd, labels):
    srcs = [
        f"{O}/eval-b2/{cd}",
        f"{O}/vmaf-b2/{cd}",
        f"{G}/eval/7b/{cd}",
        f"{G}/vmaf/7b/{cd}",
        f"{G}/eval/sharp/{cd}",
        f"{G}/vmaf/sharp/{cd}",
    ]  # the sharp fp16's own seeds 43, 1234 (S8)
    for L in labels:
        srcs += [f"{G}/eval/{L}/{cd}", f"{G}/vmaf/{L}/{cd}"]
    return srcs


def fr_runs(L, cd):
    """fr_metrics.py's JSONs of a label (7b included) on clip cd: variant@ref -> s<seed> -> run."""
    out = defaultdict(dict)
    for p in sorted(glob.glob(f"{G}/fr/{L}/{cd}/*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                r = json.load(f)
        except (OSError, ValueError):
            X["skipped"].append(p)
            continue
        if "per_frame" in r:
            r["_paths"] = [p]
            out[r["variant"]][f"s{r['seed']}"] = r
    return out


def fr_series(r, k):
    v = (
        r.get("per_transition", {}).get(k)
        if k.startswith("temporal")
        else r.get("per_frame", {}).get(k)
    )
    return None if not v else np.asarray(v, dtype=float)


# ---------------------------------------------------------------- pairing (colour_eval's pairs_tables, one view)


def tag_of(k, mean, lo, hi, band, higher):
    if not ((lo > 0 or hi < 0) and abs(mean) > band):
        return ""
    if k == "lap":
        return "↓" if mean < 0 else "↑"
    return "B" if (mean > 0) == (k in higher) else "W"


def mult_of(mean, band):
    """|difference| as a multiple of the spread (inf for a nonzero difference over a zero spread)."""
    if mean is None:
        return None
    if band and band > 0:
        return abs(mean) / band
    return math.inf if mean else 0.0


def mult_txt(m):
    return "–" if m is None else "∞×" if m == math.inf else f"{m:.2f}×" if m < 10 else f"{m:.1f}×"


def verdict(e, mode):
    """One pair entry's verdict in a mode: tag (strict), tag0 (colour mode: band 0), tagc (calibrated: a strict W, ↑
    or ↓ whose |mean| also exceeds K x the band, K = KCAL[number of the band's seeds])."""
    if mode != "tagc":
        return e[mode]
    t = e["tag"]
    if t not in ("W", "↑", "↓"):
        return ""
    k = KCAL.get(len(e.get("bseeds", [])))
    return t if k is not None and abs(e["mean"]) > k * e["band"] else ""


def pair_core(vr, b, bs, keys, series, higher, rng_per_key=()):
    out = {}
    rng = np.random.default_rng(1)
    for k in keys:
        if k in rng_per_key:
            rng = np.random.default_rng(1)
        common = [s for s in SEEDS if s in vr and s in b]
        diffs = []
        for s in common:
            x, y = series(vr[s], k), series(b[s], k)
            if x is None or y is None or len(x) != len(y):
                continue
            diffs.append(x - y)
        if not diffs:
            continue
        mean = float(np.mean(np.concatenate(diffs)))
        lo, hi = E.boot(diffs, 8, 2000, rng)
        sv = {}
        for s in SEEDS:
            if s in bs and series(bs[s], k) is not None:
                sv[s] = float(series(bs[s], k).mean())
        band = (max(sv.values()) - min(sv.values())) if len(sv) > 1 else 0.0
        out[k] = {
            "mean": mean,
            "lo": lo,
            "hi": hi,
            "band": band,
            "bseeds": sorted(sv),
            "seeds": common,
            "tag": tag_of(k, mean, lo, hi, band, higher),
            "tag0": tag_of(k, mean, lo, hi, 0.0, higher),
            "value": float(np.mean([series(vr[s], k).mean() for s in common])),
        }
    return out


def cached(kind, paths, fn):
    key = hashlib.sha1((kind + stat_key(paths)).encode()).hexdigest()
    cp = os.path.join(X["out"], "cache", key + ".json")
    try:
        with open(cp, encoding="utf-8") as f:
            X["cache_hits"] += 1
            return json.load(f)
    except (OSError, ValueError):
        pass
    res = fn()
    X["cache_miss"] += 1
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    with open(cp + ".tmp", "w", encoding="utf-8") as f:
        json.dump(res, f)
    os.replace(cp + ".tmp", cp)
    return res


def run_paths(*groups):
    return [p for g in groups for r in g.values() for p in r.get("_paths", [])]


def pair(by, v, base, bandkey):
    vr, b, bs = by.get(v, {}), by.get(base, {}), by.get(bandkey, {})
    if not any(s in b for s in vr):
        return {}
    return cached(
        f"pair|{v}|{base}|{bandkey}|",
        run_paths(vr, b, bs),
        lambda: pair_core(vr, b, bs, list(E.METRICS) + ["ssim_y"], E.series, E.HIGHER, ("ssim_y",)),
    )


def fr_pair(frv, frb, fr7, v, base, bandkey):
    vr, b, bs = frv.get(v, {}), frb.get(base, {}), fr7.get(bandkey, {})
    if not any(s in b for s in vr):
        return {}
    return cached(
        f"frpair|{v}|{base}|{bandkey}|",
        run_paths(vr, b, bs),
        lambda: pair_core(vr, b, bs, list(FR_KEYS), fr_series, FR_HIGHER),
    )


# ---------------------------------------------------------------- detail, bands, distances


class GtLap:
    def __init__(self, path):
        self.path, self.c, self.new = path, {}, 0
        try:
            with open(path, encoding="utf-8") as f:
                self.c = json.load(f)
        except (OSError, ValueError):
            pass

    def get(self, gt, n):
        st = os.stat(gt)
        k = f"{gt}|{n}|{st.st_size}|{int(st.st_mtime)}"
        if (
            k not in self.c
        ):  # as colour_eval score reads the GT (colour_diag's Master), fr_clips.py's Laplacian
            m = E.D.Master(gt, 4)
            vals = []
            try:
                while len(vals) < n:
                    x = m.read()
                    if x is None:
                        break
                    vals.append(E.D.lap_var(E.D.luma8(x)))
            finally:
                m.close()
            self.c[k] = float(np.mean(vals))
            self.new += 1
        return self.c[k]

    def save(self):
        if self.new:
            with open(self.path + ".tmp", "w", encoding="utf-8") as f:
                json.dump(self.c, f)
            os.replace(self.path + ".tmp", self.path)


def lap_of(runs, seeds):
    vals = [E.run_mean(runs[s], "lap") for s in seeds if s in runs]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def load_json(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def band1(j, sig):
    """colour_bands.py's finest band (k = 1, below 0.7 px), whole picture: (rms of sig / the GT's, corr with the GT)."""
    try:
        b = j["bands"]["whole"]["1"]
        g = b["rms"]["g"]
        return (b["rms"][sig] / g if g else None), b["corr"].get(sig)
    except (KeyError, TypeError):
        return None, None


def bands_7b(cd):
    return {
        "s42": load_json(f"{O}/bands-b2/{cd}-7b.json"),
        "s43": load_json(f"{G}/bands/7b/{cd}-s43.json"),
        "s1234": load_json(f"{G}/bands/7b/{cd}-s1234.json"),
    }


def bands_sharp(cd):
    """The sharp 7B fp16's own scans (S8): seed 42 colour's (bands-b2), 43 and 1234 ours."""
    return {
        "s42": load_json(f"{O}/bands-b2/{cd}-sharp.json"),
        "s43": load_json(f"{G}/bands/sharp/{cd}-s43.json"),
        "s1234": load_json(f"{G}/bands/sharp/{cd}-s1234.json"),
    }


def full_band(runs, keys, series):
    """True when runs (s<seed> -> run) hold the 3 seeds, each with every series of keys (S8: the sharp's band)."""
    return all(s in runs and all(series(runs[s], k) is not None for k in keys) for s in SEEDS)


DB_RE = re.compile(r"PSNR \(RGB, all frames\) (inf|[\d.]+) dB")


def db_of(path):
    try:
        with open(path, encoding="utf-8") as f:
            m = DB_RE.search(f.read())
        return float(m.group(1)) if m else None
    except OSError:
        return None


# ---------------------------------------------------------------- one label


def cell(k, e, mode="tag"):
    """A metric's cell: the difference and its strict verdict; a W (or ↑ / ↓) with its multiple of the spread, in
    bold when the calibrated rule calls it worse too."""
    if not e or k not in e:
        return "–"
    t = e[k][mode]
    s = fmt(k, e[k]["mean"]) + (f" {t}" if t else "")
    if mode == "tag" and t in ("W", "↑", "↓"):
        s += " " + mult_txt(mult_of(e[k]["mean"], e[k]["band"]))
        if verdict(e[k], "tagc"):
            s = f"**{s}**"
    return s


def ref_of(L):
    """The reference model a label is paired with: (ref tag, name, fr/masters label). The sharp 7B's files (sh-*)
    against the sharp 7B fp16 at seed 42 (colour's sh42 scores: eval-b2 ~sharp, vmaf-b2), every other file against
    the 7B fp16. The band: the 7B fp16's 3-seed spread; for the sharp's files the sharp 7B fp16's own 3-seed
    spread (colour's sh42, our s43 and s1234; S8) per clip-variant and source where its 3 seeds are scored, the
    7B's until then (label_data's bandsrc)."""
    return ("sharp", "sharp 7B fp16", "sharp") if L.startswith("sh-") else ("f32", "7B fp16", "7b")


def label_data(L, gl):
    """Per clip: {variant: {pairs, fr, detail, bands, db}} for label L."""
    data = {}
    rt, rname, rlab = ref_of(L)
    for c in CLIPS_A:
        cd = f"{c}-d1"
        by = merged([r for s in clip_sources(cd, [L]) for r in load_dir(s)])
        if not any(k.endswith(f"@{L}") for k in by):
            continue
        frL, frB, fr7 = fr_runs(L, cd), fr_runs(rlab, cd), fr_runs("7b", cd)
        b7 = bands_7b(cd)
        if (
            rt == "sharp"
        ):  # the sharp's s42 scan, its range = the 7B's seed deviations around the 7B's s42
            b7 = {"s42": load_json(f"{O}/bands-b2/{cd}-sharp.json"), "_7b": b7}
        sb = (
            bands_sharp(cd) if rt == "sharp" else {}
        )  # the sharp fp16's own 3 scans, once ours exist (S8)
        bL = load_json(f"{G}/bands/{L}/{cd}.json")
        gt_lap = None
        try:
            gt_lap = gl.get(f"{CLIPS}/{c}.gt.mkv", 45)
        except Exception as ex:  # noqa: BLE001
            log(f"GT Laplacian {cd}: {ex}")
        d = {
            "db": db_of(f"{G}/diff/{L}/{cd}.s42.txt"),
            "db7": {s: db_of(f"{G}/diff/7b/{cd}.{s}.txt") for s in ("s43", "s1234")},
        }
        d["dbsh"] = (
            {s: db_of(f"{G}/diff/sharp/{cd}.{s}.txt") for s in ("s43", "s1234")}
            if rt == "sharp"
            else {}
        )
        for v, short in VARS:
            # the band (S8): a sharp file takes the sharp fp16's own 3-seed spread where its 3 seeds are scored
            own_sc = rt == "sharp" and full_band(
                by.get(f"{v}@sharp", {}), ("psnr_y", "vmaf"), E.series
            )
            own_fr = rt == "sharp" and full_band(frB.get(f"{v}@sharp", {}), ("dists",), fr_series)
            P = pair(by, f"{v}@{L}", f"{v}@{rt}", f"{v}@sharp" if own_sc else f"{v}@f32")
            F = fr_pair(
                frL,
                frB,
                frB if own_fr else fr7,
                f"{v}@{L}",
                f"{v}@{rt}",
                f"{v}@sharp" if own_fr else f"{v}@f32",
            )
            if "dists" in F:
                P["dists_all"] = F["dists"]
            for k in ("psnr_y_bottom16", "psnr_y_rest"):
                if k in F:
                    P[k] = F[k]
            r7 = by.get(f"{v}@{rt}", {})
            lapL = lap_of(by.get(f"{v}@{L}", {}), ("s42",))
            lap7 = {
                s: E.run_mean(r7[s], "lap")
                for s in (SEEDS if rt != "sharp" or own_sc else ("s42",))
                if s in r7
            }
            sig = "c" if v == "none" else "s"
            bl = band1(bL, sig) if bL else (None, None)
            sbr = {s: band1(j, sig)[0] for s, j in sb.items() if j}
            own_bd = len(sbr) == 3 and None not in sbr.values()
            if own_bd:  # the sharp fp16's own 3 scans (S8): its spread, around its s42
                b7r, r0 = sbr, sbr["s42"]
                b7c = {s: band1(j, sig)[1] for s, j in sb.items()}
            elif rt == "sharp":
                own = {s: band1(j, sig)[0] for s, j in b7["_7b"].items() if j}
                r0 = band1(b7["s42"], sig)[0] if b7["s42"] else None
                b7r = (
                    {f"d{s}": r0 + x - own["s42"] for s, x in own.items()}
                    if r0 is not None and "s42" in own and None not in own.values()
                    else {}
                )
                b7c = {"s42": band1(b7["s42"], sig)[1]} if b7["s42"] else {}
            else:
                b7r = {s: band1(j, sig)[0] for s, j in b7.items() if j}
                b7c = {s: band1(j, sig)[1] for s, j in b7.items() if j}
            # band energy: no interval, so the spread alone decides (as DISTS 5f): |file - reference at s42| beyond
            # the 7B's seed spread (max - min of its per-seed values), either way. The reference at s42: the 7B's own
            # scan, or the sharp's for the sharp's files (b7r's "ds42" = r0).
            bref = None
            if b7r:
                bref = r0 if rt == "sharp" else b7r.get("s42")
            btag = btagc = ""
            bmult = None
            if bl[0] is not None and len(b7r) > 1 and bref is not None:
                spread = max(b7r.values()) - min(b7r.values())
                btag = "↑" if bl[0] - bref > spread else "↓" if bref - bl[0] > spread else ""
                bmult = mult_of(bl[0] - bref, spread)
                kc = KCAL.get(len(b7r))
                btagc = btag if btag and kc is not None and abs(bl[0] - bref) > kc * spread else ""
            d[short] = {
                "pairs": P,
                "lap": lapL,
                "lap7": lap7,
                "gtlap": gt_lap,
                "band": bl,
                "band7": b7r,
                "band7ref": bref,
                "band7c": b7c,
                "btag": btag,
                "btagc": btagc,
                "bmult": bmult,
                "rname": "sharp" if rt == "sharp" else "7B",
                "bandsrc": {
                    "scores": "sharp" if own_sc else "7B",
                    "fr": "sharp" if own_fr else "7B",
                    "bands": "sharp" if own_bd else "7B",
                },
            }
        data[c] = d
    return data


def guard_fail(g, e, mode="tag"):
    """(failed?, text) of guard g on one clip's variant entry e."""
    P = e["pairs"]

    def w(k):
        return k in P and verdict(P[k], mode) == "W"

    if g == "banding":
        ks = [k for k in ("cambi_added",) if w(k)]
    elif g == "flicker":
        ks = [k for k in ("t_full", "t_lf") if w(k)]
    elif g == "drift":
        ks = [k for k in ("de4",) if w(k)]
    elif g == "lpips":
        ks = [k for k in ("lpips",) if w(k)]
    elif g == "dists":
        ks = [k for k in (("dists_all",) if "dists_all" in P else ("dists",)) if w(k)]
    elif g == "detail":
        t = verdict(P["lap"], mode) if "lap" in P else ""
        if t in ("↑", "↓"):
            return True, f"{t} {mult_txt(mult_of(P['lap']['mean'], P['lap']['band']))} {det_txt(e)}"
        return False, ""
    elif g == "bands":
        t = e.get("btagc", "") if mode == "tagc" else e["btag"]
        if t:
            return True, f"{t} {mult_txt(e.get('bmult'))} {band_txt(e)}"
        return False, ""
    else:
        return False, ""
    if ks:
        return True, ", ".join(
            f"{fmt(k, P[k]['mean'])} {mult_txt(mult_of(P[k]['mean'], P[k]['band']))}" for k in ks
        )
    return False, ""


def guard_scored(g, e):
    P = e["pairs"]
    need = {
        "banding": "cambi_added",
        "flicker": "t_full",
        "drift": "de4",
        "lpips": "lpips",
        "dists": "dists",
        "detail": "lap",
    }.get(g)
    if g == "bands":
        return e["band"][0] is not None and len(e["band7"]) > 1 and e.get("band7ref") is not None
    return need in P


def det_txt(e):
    gl, l7 = e["gtlap"], [v for v in e["lap7"].values() if v is not None]
    if not gl or e["lap"] is None:
        return "–"
    s = f"{e['lap'] / gl:.2f}"
    rn = e.get("rname", "7B")
    r42 = e["lap7"].get("s42")
    if l7:
        s += (
            f" ({rn} s42 {r42 / gl:.2f}; {min(l7) / gl:.2f}–{max(l7) / gl:.2f})"
            if len(l7) > 1 and r42 is not None
            else f" ({rn} {min(l7) / gl:.2f}–{max(l7) / gl:.2f})"
            if len(l7) > 1
            else f" ({rn} {l7[0] / gl:.2f})"
        )
    return s


def band_txt(e):
    r, c = e["band"]
    if r is None:
        return "–"
    s = f"{r:.2f} · {c:.2f}" if c is not None else f"{r:.2f}"
    v7 = list(e["band7"].values())
    rn = e.get("rname", "7B")
    ref = e.get("band7ref")
    if v7:
        s += (
            f" ({rn} s42 {ref:.2f}; {min(v7):.2f}–{max(v7):.2f})"
            if len(v7) > 1 and ref is not None
            else f" ({rn} {min(v7):.2f}–{max(v7):.2f})"
            if len(v7) > 1
            else f" ({rn} {v7[0]:.2f})"
        )
    return s


def det_cell(e):
    """Detail's per-clip cell: the value, the reference's at s42, the 7B's range; ↑ / ↓ with the multiple (bold when
    calibrated)."""
    P, s = e["pairs"], det_txt(e)
    if "lap" in P and P["lap"]["tag"]:
        s += f" {P['lap']['tag']} {mult_txt(mult_of(P['lap']['mean'], P['lap']['band']))}"
        if verdict(P["lap"], "tagc"):
            s = f"**{s}**"
    return s


def band_cell(e):
    s = band_txt(e)
    if e["btag"]:
        s += f" {e['btag']} {mult_txt(e.get('bmult'))}"
        if e.get("btagc"):
            s = f"**{s}**"
    return s


def band_src(e):
    """A clip-variant's band (S8): sharp (the sharp fp16's own 3 seeds), 7B, or per source."""
    b = e.get("bandsrc") or {}
    if b and set(b.values()) == {"sharp"}:
        return "sharp"
    if set(b.values()) <= {"7B"}:
        return "7B"
    return ", ".join(f"{k} {v}" for k, v in b.items())


def band_counts(data):
    """({source: clip-variants on the sharp fp16's own band}, clip-variants) of a label's data (S8)."""
    es = [data[c][sh] for c in data for _, sh in VARS if sh in data[c]]
    return {
        k: sum((e.get("bandsrc") or {}).get(k) == "sharp" for e in es)
        for k in ("scores", "fr", "bands")
    }, len(es)


def band_count(data):
    n, tot = band_counts(data)
    if tot and all(v == tot for v in n.values()):
        return f"the sharp's own on {tot}/{tot} clip-variants"
    if not any(n.values()):
        return "the 7B's (the sharp's own seeds not scored yet)"
    return (
        f"the sharp's own on scores {n['scores']}/{tot}, DISTS 45f {n['fr']}/{tot}, band energy "
        f"{n['bands']}/{tot}, the 7B's elsewhere"
    )


def sharp_band_note(data):
    """Which band a sh-* page uses (S8)."""
    n, tot = band_counts(data)
    if tot and all(v == tot for v in n.values()):
        return (
            "**Band: the sharp 7B fp16's own 3-seed spread** (seed 42 = colour's sh42, 43 and 1234 = ours, the "
            f"same commands) on every clip-variant ({tot}/{tot}) for the scores, DISTS 45f and band energy: read "
            '"the sharp 7B fp16" for "the 7B fp16" in the rule above (sharp-band.md).'
        )
    if not any(n.values()):
        return (
            "**Band: the 7B fp16's 3-seed spread** (the sharp's own seeds 43 and 1234 not scored yet), the "
            "band-energy range the sharp's s42 value moved by the 7B's seed deviations."
        )
    return (
        "**Band: the sharp 7B fp16's own 3-seed spread** (seed 42 = colour's sh42, 43 and 1234 = ours) where "
        f"its 3 seeds are scored: scores {n['scores']}/{tot}, DISTS 45f {n['fr']}/{tot}, band energy "
        f"{n['bands']}/{tot} clip-variants; **the 7B fp16's 3-seed spread elsewhere** (column Band, per "
        "clip; the band-energy range there the sharp's s42 value moved by the 7B's seed deviations)."
    )


GUARD_KEYS = ("cambi_added", "t_full", "t_lf", "de4", "lpips", "lap")


def worst_mult(data):
    """The label's largest multiple of the spread among its strict guard failures: (multiple, where), or (None, '')."""
    best = (None, "")
    for c in data:
        for short in ("none", "split"):
            if short not in data[c]:
                continue
            e, P = data[c][short], data[c][short]["pairs"]
            dk = "dists_all" if "dists_all" in P else "dists"
            for k in GUARD_KEYS + (dk,):
                if k in P and P[k]["tag"] in ("W", "↑", "↓"):
                    m = mult_of(P[k]["mean"], P[k]["band"])
                    if best[0] is None or m > best[0]:
                        name = dict(COLS, lap="Laplacian").get(k, k)
                        best = (m, f"{c} {short} {name} {fmt(k, P[k]['mean'])} {P[k]['tag']}")
            if (
                e["btag"]
                and e.get("bmult") is not None
                and (best[0] is None or e["bmult"] > best[0])
            ):
                dv = minus(f"{e['band'][0] - e['band7ref']:+.2f}")
                best = (e["bmult"], f"{c} {short} finest band {dv} {e['btag']}")
    return best


def kind_cells(data, short, mode="tag"):
    """{guard: {kind: text}} over the clips of each kind."""
    out = {}
    for g, _ in GUARDS:
        out[g] = {}
        for kd in KINDS:
            clips = [c for c in CLIPS_A if KIND[c] == kd]
            have = [
                c
                for c in clips
                if c in data and short in data[c] and guard_scored(g, data[c][short])
            ]
            if not have:
                out[g][kd] = "–"
                continue
            fails = []
            for c in have:
                ok, txt = guard_fail(g, data[c][short], mode)
                if ok:
                    fails.append(f"{c} {txt}")
            cov = "" if len(have) == len(clips) else f" ({len(have)}/{len(clips)})"
            out[g][kd] = ("FAIL " + "; ".join(fails) if fails else "PASS") + cov
    return out


def table(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + [
        "| " + " | ".join(r) + " |" for r in rows
    ]


def meta_of(L):
    for c in CLIPS_A:
        for p in (f"{G}/dumps/{c}-d1/{L}-s42/meta.json",):
            j = load_json(p)
            if j:
                cmd = j.get("command", [])
                try:
                    return cmd[cmd.index("--dit_model") + 1]
                except (ValueError, IndexError):
                    return "?"
    return "colour's sharp 7B sh42 decodes (validation)" if L == "valsharp" else "?"


def stamp():
    return datetime.now(PARIS).strftime("%Y-%m-%d %H:%M Paris")


THREEB = (
    "**The 3B has no seed band of its own tonight** (S16, 2026-10-08): a different model (3B parameters; 3b-cur = "
    "our seedvr2x_ema_3b_fp16 from ByteDance's current 3B weights, 3b-first = numz's seedvr2_ema_3b_fp16, the "
    "first 3B weights), one seed (42); its cells are paired with the 7B fp16 at seed 42 and set against the 7B "
    "fp16's 3-seed spread: informative only, not a verdict on the 3B (a guard's FAIL here reads \"unlike the 7B "
    'beyond the 7B\'s seed scatter", not "worse"). The 3B against the sharp 7B\'s 4 GB pick (sh-dyn), the sharp '
    "and the 7B float16, against the GT: 3b-00-overview.md."
)


RULE = (
    "Cells: the file's output minus the 7B fp16's at seed 42, frame by frame (colour_eval.py's scores, colour's "
    "baton-2 commands); **B / W** = better / worse when colour_eval's 95% moving-block bootstrap interval (blocks "
    "of 8, 2000 draws) excludes 0 AND |difference| exceeds the 7B fp16's 3-seed spread (seeds 42, 43, 1234); "
    "Laplacian ↑ / ↓ the same; no tag = within. DISTS 5f = colour_eval's every 9th frame (5 frames: its interval "
    "is degenerate, the spread alone decides); DISTS 45f = fr_metrics.py's every frame. Detail = the luma "
    "Laplacian variance ÷ the GT's; band = colour_bands.py's finest band (below 0.7 px), rms ÷ the GT's · its "
    "correlation with the GT (none: the raw decode c; split: s), ↑ / ↓ when |the file's − the 7B's at seed 42| "
    "exceeds the 7B's 3-seed spread (no interval: the spread alone decides, as DISTS 5f); in brackets after "
    "Detail and band: the 7B's seed-42 value; its 3-seed range; dB = RGB PSNR "
    "between the 16-bit none masters (ffv1_out.py --diff) of the file and the 7B fp16 at seed 42. Each W, ↑ or ↓ "
    "shows its multiple of the spread (|difference| ÷ the 7B's seed spread). Two rules: **strict** (the above) and "
    "**calibrated** = strict AND |difference| > K × the spread, K = 2.7 with 3 seeds (10.9 with 2): the multiple "
    "a further fp16 seed would exceed with 5% probability per clip-variant (S4's Monte Carlo, Gaussian seed "
    "scatter, both directions), where the strict rule fails such a seed 29% of the time (50% with 2 seeds); "
    "**bold** cell = worse by the calibrated rule too. " + FL.FLOOR_TXT_1080
)


def label_md(L, data, spread_note):
    n = len(data)
    rname = ref_of(L)[1]
    lines = [
        f"# {L} − {rname}: slice A tier 1 (1080p ×2 from d1, seed 42)",
        "",
        f"Generated {stamp()} by ms_sum.py. File: `{meta_of(L)}`. Clips scored: {n}/{len(CLIPS_A)}"
        + (
            f" (missing: {', '.join(c for c in CLIPS_A if c not in data)})"
            if n < len(CLIPS_A)
            else ""
        )
        + f". {spread_note}",
        "",
        RULE,
        "",
    ]
    if rname != "7B fp16":
        lines += [
            f"**This file is paired with the {rname} at seed 42** (colour's sh42 scores and scan; dB to its "
            "none master), not with the 7B fp16. " + sharp_band_note(data),
            "",
        ]
    if L.startswith("3b-"):  # S16: the 3B, informative only
        lines += [THREEB, ""]
    nk = {kd: sum(KIND[c] == kd for c in CLIPS_A) for kd in KINDS}
    for mode, head in (
        ("tag", "strict rule"),
        ("tagc", f"calibrated rule (strict AND beyond {KCAL[3]} × the 3-seed spread or its floor)"),
    ):
        lines += [f"## Guards per kind of source, {head} (none · split:ycc:4:3)", ""]
        kn, ks = kind_cells(data, "none", mode), kind_cells(data, "split", mode)
        rows = [[title] + [f"{kn[g][kd]} · {ks[g][kd]}" for kd in KINDS] for g, title in GUARDS]
        lines += table(["Guard"] + [f"{kd} ({nk[kd]})" for kd in KINDS], rows) + [""]
    wm = worst_mult(data)
    lines += [
        f"Worst multiple of the spread among the strict failures: {mult_txt(wm[0])}"
        + (f" ({wm[1]})" if wm[1] else "")
        + ".",
        "",
    ]
    lines += [
        "## Fidelity (reported, not a guard): mean of the per-clip differences (B/W/within)",
        "",
    ]
    rows = []
    for v, short in VARS:
        cells = []
        for k in ("psnr_y", "ssim_y", "vmaf"):
            es = [data[c][short]["pairs"][k] for c in data if k in data[c][short]["pairs"]]
            if not es:
                cells.append("–")
                continue
            b = sum(e["tag"] == "B" for e in es)
            w = sum(e["tag"] == "W" for e in es)
            cells.append(
                f"{fmt(k, float(np.mean([e['mean'] for e in es])))} ({b}/{w}/{len(es) - b - w})"
            )
        dbs = [data[c]["db"] for c in data if data[c]["db"] is not None]
        rows.append([v if v == "none" else "ycc:4:3", str(n)] + cells)
    lines += table(["Variant", "Clips", "PSNR-Y", "SSIM-Y", "VMAF"], rows)
    dbs = [data[c]["db"] for c in data if data[c]["db"] is not None]
    d7 = [v for c in data for v in data[c]["db7"].values() if v is not None]
    lines += [
        "",
        f"Distance to the {rname} s42 (none masters): {f2(np.mean(dbs)) if dbs else '–'} dB mean, "
        f"{f2(min(dbs)) if dbs else '–'} worst; the 7B's own seeds 43 and 1234 against its 42: "
        f"{f2(np.mean(d7)) if d7 else '–'} dB mean ({f2(min(d7)) if d7 else '–'}–{f2(max(d7)) if d7 else '–'}).",
    ]
    dsh = [x for c in data for x in (data[c].get("dbsh") or {}).values() if x is not None]
    if dsh:
        lines[-1] = lines[-1][:-1] + (
            f"; the sharp's own seeds 43 and 1234 against its 42: {f2(np.mean(dsh))} dB mean "
            f"({f2(min(dsh))}–{f2(max(dsh))}, {len(dsh)} pairs)."
        )
    lines += ["", "## Per clip", ""]
    sh = rname != "7B fp16"
    head = (
        ["Clip", "Var"]
        + [h for _, h in COLS]
        + [
            "Detail ÷ GT's",
            "Finest band ÷ GT's · corr",
            f"dB to {rname.split(' fp16')[0]} s42 (7B s43, s1234 vs 7B s42)"
            if not sh
            else f"dB to {rname.split(' fp16')[0]} s42 (its own "
            "s43, s1234 vs its s42; the 7B's until scored)",
        ]
    )
    head += ["Band"] if sh else []
    rows = []
    for c in CLIPS_A:
        if c not in data:
            continue
        for v, short in VARS:
            e = data[c][short]
            dsh = data[c].get("dbsh") or {}
            if sh and all(dsh.get(s) is not None for s in ("s43", "s1234")):
                db = (
                    f2(data[c]["db"])
                    + " (sharp "
                    + ", ".join(f2(dsh[s]) for s in ("s43", "s1234"))
                    + ")"
                )
            else:
                db = (
                    f2(data[c]["db"])
                    + (" (7B " if sh else " (")
                    + ", ".join(
                        f2(data[c]["db7"][s]) if data[c]["db7"][s] else "–"
                        for s in ("s43", "s1234")
                    )
                    + ")"
                )
            rows.append(
                [c, "none" if v == "none" else "4:3"]
                + [cell(k, e["pairs"]) for k, _ in COLS]
                + [det_cell(e), band_cell(e), db if short == "none" else ""]
                + ([band_src(e)] if sh else [])
            )
    lines += table(head, rows)
    return lines


# ---------------------------------------------------------------- the 7B fp16's own band, q4k vs nq4km, validation


def band_md(clip_bys, who="7b"):
    """The 7B fp16's band (who 7b), or the sharp 7B fp16's own (who sharp, S8) with its spreads against the 7B's."""
    rt = "f32" if who == "7b" else "sharp"
    if who == "7b":
        lines = [
            "# The 7B fp16's band: its 3-seed spread per clip (seeds 42, 43, 1234)",
            "",
            f"Generated {stamp()}. Seed 42: colour's eval-b2 / vmaf-b2 / bands-b2; 43 and 1234: ours (same commands). "
            "Cells: max − min of the per-seed means (seeds present). dB: seeds 43 and 1234 against 42 (none masters).",
            "",
        ]
    else:
        lines = [
            "# The sharp 7B fp16's own band: its 3-seed spread per clip (seeds 42, 43, 1234)",
            "",
            f"Generated {stamp()}. Seed 42: colour's sh42 (eval-b2 / vmaf-b2 / bands-b2 ~sharp); 43 and 1234: ours "
            "(S8: GPU runs a-<clip>-d1-sharp-s<seed>, colour's sh42 flags but --seed; scored as the 7B's seeds). "
            "Cells: max − min of the per-seed means (seeds present). dB: seeds 43 and 1234 against 42 (none masters). "
            "The sharp's files (sh-*) take this band where all 3 seeds are scored, the 7B's elsewhere.",
            "",
        ]
    ratios = defaultdict(list)
    ks = (
        "psnr_y",
        "ssim_y",
        "vmaf",
        "lpips",
        "dists",
        "cambi_added",
        "de4",
        "t_full",
        "t_lf",
        "lap",
    )
    rows = []
    for c in CLIPS_A:
        cd = f"{c}-d1"
        by = clip_bys[c]
        for v, short in VARS:
            r7, rb = by.get(f"{v}@{rt}", {}), by.get(f"{v}@f32", {})
            cells = []
            for k in ks:
                vals = [
                    float(E.series(r7[s], k).mean())
                    for s in SEEDS
                    if s in r7 and E.series(r7[s], k) is not None
                ]
                vb = [
                    float(E.series(rb[s], k).mean())
                    for s in SEEDS
                    if s in rb and E.series(rb[s], k) is not None
                ]
                if who != "7b" and len(vals) == 3 and len(vb) == 3 and max(vb) > min(vb):
                    ratios[k].append((max(vals) - min(vals)) / (max(vb) - min(vb)))
                cells.append(
                    (
                        "–"
                        if len(vals) < 2
                        else minus(E.diff_fmt(k).format(max(vals) - min(vals)).lstrip("+"))
                    )
                    + ("" if len(vals) == 3 else f" ({len(vals)})")
                )
            d = [db_of(f"{G}/diff/{who}/{cd}.{s}.txt") for s in ("s43", "s1234")]
            b7 = [
                band1(j, "c" if v == "none" else "s")[0]
                for j in (bands_7b(cd) if who == "7b" else bands_sharp(cd)).values()
                if j
            ]
            b7 = [x for x in b7 if x is not None]
            fr7 = fr_runs(who, cd).get(f"{v}@{rt}", {})
            fd = [
                float(fr_series(fr7[s], "dists").mean())
                for s in SEEDS
                if s in fr7 and fr_series(fr7[s], "dists") is not None
            ]
            if who != "7b":
                bb = [band1(j, "c" if v == "none" else "s")[0] for j in bands_7b(cd).values() if j]
                bb = [x for x in bb if x is not None]
                if len(b7) == 3 and len(bb) == 3 and max(bb) > min(bb):
                    ratios["finest band"].append((max(b7) - min(b7)) / (max(bb) - min(bb)))
                fb = fr_runs("7b", cd).get(f"{v}@f32", {})
                fbd = [
                    float(fr_series(fb[s], "dists").mean())
                    for s in SEEDS
                    if s in fb and fr_series(fb[s], "dists") is not None
                ]
                if len(fd) == 3 and len(fbd) == 3 and max(fbd) > min(fbd):
                    ratios["DISTS 45f"].append((max(fd) - min(fd)) / (max(fbd) - min(fbd)))
            rows.append(
                [c, short, ",".join(s[1:] for s in SEEDS if s in r7)]
                + cells
                + [
                    (f"{min(b7):.2f}–{max(b7):.2f}" if len(b7) > 1 else f2(b7[0]) if b7 else "–")
                    + f" ({len(b7)})",
                    (f"{max(fd) - min(fd):.4f}" if len(fd) > 1 else "–") + f" ({len(fd)})",
                    ", ".join(f2(x) for x in d) if short == "none" else "",
                ]
            )
    lines += table(
        ["Clip", "Var", "Seeds"]
        + list(ks)
        + ["finest band ÷ GT's: range", "DISTS 45f", "dB s43, s1234 vs s42"],
        rows,
    )
    if who != "7b":
        lines += [
            "",
            "The sharp's spread ÷ the 7B fp16's, per metric: median (min–max) over the clip-variants where "
            "both have 3 seeds (n): "
            + (
                "; ".join(
                    f"{k} {np.median(r):.2f}× ({min(r):.2f}–{max(r):.2f}, {len(r)})"
                    for k, r in ratios.items()
                    if r
                )
                or "none yet"
            )
            + ".",
        ]
    return lines


def x_md(clip_bys, gl=None):
    """Our Q4_K against numz's Q4_K_M, paired at seed 42, the 7B fp16's 3-seed spread as the band (S23: or its
    floor; detail's from the GT's Laplacian, gl)."""
    lines = [
        "## Our Q4_K − numz's Q4_K_M (paired at seed 42; band: the 7B fp16's 3-seed spread or its floor)",
        "",
    ]
    rows = []
    for c in NQ4KM:
        by = clip_bys.get(c) or {}
        for v, short in VARS:
            P = (
                pair(by, f"{v}@q4k", f"{v}@nq4km", f"{v}@f32")
                if f"{v}@q4k" in by and f"{v}@nq4km" in by
                else {}
            )
            FL.apply_pairs(
                sys.modules[__name__], P, gl.get(f"{CLIPS}/{c}.gt.mkv", 45) if gl and P else None
            )
            rows.append(
                [c, short]
                + [
                    cell(k, P)
                    for k in (
                        "psnr_y",
                        "ssim_y",
                        "vmaf",
                        "lpips",
                        "dists",
                        "cambi_added",
                        "de4",
                        "t_full",
                        "t_lf",
                        "lap",
                    )
                ]
                + [f2(db_of(f"{G}/diff/x/{c}-d1.q4k-nq4km.txt")) if short == "none" else ""]
            )
    lines += table(
        [
            "Clip",
            "Var",
            "PSNR-Y",
            "SSIM-Y",
            "VMAF",
            "LPIPS",
            "DISTS 5f",
            "CAMBI+",
            "ΔE00 lf",
            "T-err",
            "T-err lf",
            "Laplacian",
            "dB q4k–nq4km",
        ],
        rows,
    )
    return lines


def series_diff(a, b):
    """max |a - b| over every series both JSONs hold; (max, n series, series only one holds)."""
    mx, n, only = 0.0, 0, []
    for part in ("per_frame", "per_transition"):
        pa, pb = a.get(part, {}), b.get(part, {})
        for k in sorted(set(pa) | set(pb)):
            if k not in pa or k not in pb:
                only.append(k)
                continue
            x, y = np.asarray(pa[k], dtype=float), np.asarray(pb[k], dtype=float)
            if x.shape != y.shape:
                only.append(f"{k}(shape)")
                continue
            if x.size:
                mx = max(mx, float(np.nanmax(np.abs(x - y))) if np.isfinite(x - y).any() else 0.0)
            n += 1
    return mx, n, only


def parse_t4(path):
    rows = {}
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return rows
    sec = text.split("## Per clip: the sharp − the 7B", 1)
    if len(sec) < 2:
        return rows
    head = None
    for line in sec[1].splitlines():
        s = line.strip()
        if s.startswith("## "):
            break
        if not s.startswith("|") or set(s) <= set("|-"):
            continue
        cells = [x.strip() for x in s.strip("|").split("|")]
        if head is None:
            head = cells
            continue
        if cells[0] != "d1":
            continue
        rows[(cells[1], "split" if cells[2].startswith("ycc") else "none")] = dict(zip(head, cells))
    return rows


def validation_md(t4path, gl):
    L = "valsharp"
    lines = [
        "# Validation: valsharp (colour's sharp 7B sh42 decodes, scored by this pipeline) against colour's baton 2",
        "",
        f"Generated {stamp()}.",
        "",
    ]
    # 1. our JSONs against colour's: same decode, reference and commands -> the same series
    rows, bad = [], 0
    for c in CLIPS_A:
        cd = f"{c}-d1"
        for v, short in VARS:
            sv = E.safe(v)
            pairs = (
                (
                    "score",
                    f"{G}/eval/{L}/{cd}/{cd}.s42.{sv}~{L}.json",
                    f"{O}/eval-b2/{cd}/{cd}.s42.{sv}~sharp.json",
                ),
                (
                    "vmaf",
                    f"{G}/vmaf/{L}/{cd}/{cd}.s42.{sv}~{L}~.json",
                    f"{O}/vmaf-b2/{cd}/{cd}.s42.{sv}~sharp~.json",
                ),
                (
                    "7B s42 vmaf (our render)",
                    f"{G}/vmaf/7b-check/{cd}/{cd}.s42.{sv}~f32~.json",
                    f"{O}/vmaf-b2/{cd}/{cd}.s42.{sv}~f32~.json",
                ),
            )
            for what, a, b in pairs:
                ja, jb = load_json(a), load_json(b)
                if not ja or not jb:
                    rows.append([cd, short, what, "–", "missing" if not ja else "colour's missing"])
                    continue
                mx, n, only = series_diff(ja, jb)
                ok = mx == 0.0 and not only
                bad += not ok
                rows.append(
                    [
                        cd,
                        short,
                        what,
                        str(n),
                        "identical"
                        if ok
                        else f"max |diff| {mx:.3g}"
                        + (f", only one: {','.join(only)}" if only else ""),
                    ]
                )
        jb7 = load_json(f"{O}/bands-b2/{cd}-sharp.json")
        ja7 = load_json(f"{G}/bands/{L}/{cd}.json")
        if ja7 and jb7:
            mx = 0.0
            for k in range(1, 6):
                for sig in ("s", "c", "r", "b", "g"):
                    try:
                        mx = max(
                            mx,
                            abs(
                                ja7["bands"]["whole"][str(k)]["rms"][sig]
                                - jb7["bands"]["whole"][str(k)]["rms"][sig]
                            ),
                        )
                    except (KeyError, TypeError):
                        pass
            bad += mx != 0.0
            rows.append(
                [
                    cd,
                    "-",
                    "bands (rms, bands 1-5)",
                    "25",
                    "identical" if mx == 0 else f"max |diff| {mx:.3g}",
                ]
            )
        else:
            rows.append([cd, "-", "bands", "–", "missing"])
    lines += [
        "## 1. The same series as colour's JSONs (valsharp vs colour's ~sharp; our 7B s42 render's VMAF vs colour's)",
        "",
        f"Not identical: {bad}.",
        "",
    ]
    lines += table(["Clip", "Var", "JSON", "Series", "Result"], rows)
    # 2. colour's T4 cells (sharp - 7B, d1) against ours in colour mode (band 0: the 7B at seed 42 only)
    t4 = parse_t4(t4path)
    agree = total = 0
    diffs = []
    for c in CLIPS_A:
        cd = f"{c}-d1"
        by = merged([r for s in clip_sources(cd, [L]) for r in load_dir(s)])
        gt = gl.get(f"{CLIPS}/{c}.gt.mkv", 45)
        for v, short in VARS:
            row = t4.get((cd, short))
            if not row:
                continue
            P = pair(by, f"{v}@{L}", f"{v}@f32", f"{v}@f32")
            for k, h in T2M:
                if h not in row:
                    continue
                ours = (
                    (fmt(k, P[k]["mean"]) + (f" {P[k]['tag0']}" if P[k]["tag0"] else ""))
                    if k in P
                    else "–"
                )
                total += 1
                if ours == row[h]:
                    agree += 1
                else:
                    diffs.append(f"{cd} {short} {h}: colour {row[h]}, ours {ours}")
            ls = lap_of(by.get(f"{v}@{L}", {}), ("s42",))
            lb = lap_of(by.get(f"{v}@f32", {}), ("s42",))
            for h, val in (
                ("Detail sharp ÷ 7B", ls / lb if ls and lb else None),
                ("sharp ÷ GT's", ls / gt if ls else None),
                ("7B ÷ GT's", lb / gt if lb else None),
            ):
                if h in row:
                    total += 1
                    if f2(val) == row[h]:
                        agree += 1
                    else:
                        diffs.append(f"{cd} {short} {h}: colour {row[h]}, ours {f2(val)}")
    lines += [
        "",
        "## 2. colour's T4 (per clip: the sharp − the 7B, d1) against this pipeline's pairing in colour mode",
        "",
        f"Colour mode = the 7B at seed 42 alone (band 0), as colour's T4 views. Cells compared as printed "
        f"(value and tag): **{agree} of {total} agree**." + (" Disagreements:" if diffs else ""),
        "",
    ]
    lines += [f"- {d}" for d in diffs[:60]]
    X["validation"] = (agree, total, bad)
    return lines


# ---------------------------------------------------------------- main


def write(path, lines):
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(path + ".tmp", path)


def labels_present():
    out = []
    for d in sorted(glob.glob(f"{G}/eval/*/")):
        L = os.path.basename(d.rstrip("/"))
        if L not in ("7b", "sharp") and any(
            os.path.isdir(f"{d}{c}-d1") for c in CLIPS_A
        ):  # slice A's (1080p) labels
            # only; 7b and sharp (S8) hold the fp16 models' own seeds: bands, not labels
            out.append(L)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=env("VAL_STATE") + "/sum")
    ap.add_argument("--t4", default=f"{O}/sum-b2/T4-1080p.md")
    a = ap.parse_args()
    t0 = time.time()
    me = sys.modules[__name__]
    FL.install(me)  # S23: a multiple of a floored spread prints a †
    X["out"] = a.out
    os.makedirs(a.out, exist_ok=True)
    gl = GtLap(os.path.join(a.out, "gt_lap.json"))
    labels = labels_present()
    clip_bys = {
        c: merged([r for s in clip_sources(f"{c}-d1", labels) for r in load_dir(s)])
        for c in CLIPS_A
    }
    seeds7 = {c: {v: sorted(clip_bys[c].get(f"{v}@f32", {})) for v, _ in VARS} for c in CLIPS_A}
    full = sum(len(seeds7[c][v]) == 3 for c in CLIPS_A for v, _ in VARS)
    spread_note = f"The 7B fp16's band from 3 seeds on {full}/{2 * len(CLIPS_A)} clip-variants" + (
        ""
        if full == 2 * len(CLIPS_A)
        else " (fewer seeds elsewhere: a narrower band, verdicts more eager)"
    )
    overview = {}
    for L in labels:
        try:  # one label's trouble never stops the others' files
            data = label_data(L, gl)
            FL.apply(me, data)  # S23: the floor at 1080p too
            write(
                os.path.join(a.out, f"{L}.md"),
                label_md(L, data, spread_note) + (["", *x_md(clip_bys, gl)] if L == "q4k" else []),
            )
            overview[L] = data
        except Exception as ex:  # noqa: BLE001
            import traceback

            log(traceback.format_exc())
            write(
                os.path.join(a.out, f"{L}.md"),
                [
                    f"# {L}",
                    "",
                    f"**ms_sum.py failed on this label at {stamp()}:** "
                    f"{type(ex).__name__}: {ex} (see {SC}/logs/sum.log)",
                ],
            )
    write(os.path.join(a.out, "7b-band.md"), band_md(clip_bys))
    if any(len(clip_bys[c].get(f"{v}@sharp", {})) > 1 for c in CLIPS_A for v, _ in VARS):
        write(os.path.join(a.out, "sharp-band.md"), band_md(clip_bys, "sharp"))
    if "valsharp" in labels:
        write(os.path.join(a.out, "validation.md"), validation_md(a.t4, gl))
    # overview
    lines = [
        "# Slice A tier 1: every file against the 7B fp16 (1080p ×2 from d1, seed 42): overview",
        "",
        f"Generated {stamp()} by ms_sum.py ({GLUE}/). {spread_note}. Per label: <label>.md; "
        "the 7B's band: 7b-band.md" + ("; validation.md" if "valsharp" in labels else "") + ".",
        "",
        RULE,
        "",
        "## Guards failed (kinds of source where a guard is worse on at least one clip), strict and calibrated "
        f"rules (calibrated: beyond {KCAL[3]} × the 3-seed spread or its floor; floors: {FL.FLOOR_LIST})",
        "",
    ]
    rows = []
    for L, data in overview.items():
        cells = []
        for mode in ("tag", "tagc"):
            for short in ("none", "split"):
                kc = kind_cells(data, short, mode)
                fails = [
                    f"{title.split(':')[0]}: "
                    + ", ".join(kd for kd in KINDS if kc[g][kd].startswith("FAIL"))
                    for g, title in GUARDS
                    if any(kc[g][kd].startswith("FAIL") for kd in KINDS)
                ]
                cells.append("; ".join(fails) if fails else "none")
        wm = worst_mult(data)
        cells.append(mult_txt(wm[0]) + f" ({wm[1]})" if wm[0] is not None else "–")
        sp = {
            k: [
                data[c]["split"]["pairs"][k]["mean"] for c in data if k in data[c]["split"]["pairs"]
            ]
            for k in ("psnr_y", "vmaf", "lpips")
        }
        dbs = [data[c]["db"] for c in data if data[c]["db"] is not None]
        rows.append(
            [
                L + (" (the 3B: informative, no band of its own)" if L.startswith("3b-") else ""),
                f"{len(data)}/{len(CLIPS_A)}",
            ]
            + cells
            + [fmt(k, float(np.mean(sp[k]))) if sp[k] else "–" for k in ("psnr_y", "vmaf", "lpips")]
            + [f"{f2(np.mean(dbs))} ({f2(min(dbs))})" if dbs else "–"]
        )
    lines += table(
        [
            "Label",
            "Clips",
            "strict, none: guards failed",
            "strict, split",
            "calibrated, none",
            "calibrated, split",
            "worst multiple (strict failures)",
            "PSNR-Y (split)",
            "VMAF (split)",
            "LPIPS (split)",
            "dB to its reference s42: mean (worst)",
        ],
        rows,
    )
    d7 = [db_of(f"{G}/diff/7b/{c}-d1.{s}.txt") for c in CLIPS_A for s in ("s43", "s1234")]
    d7 = [x for x in d7 if x is not None]
    lines += [
        "",
        f"For scale: the 7B fp16's seeds 43 and 1234 against its seed 42: {f2(np.mean(d7)) if d7 else '–'} dB mean "
        f"({f2(min(d7)) if d7 else '–'}–{f2(max(d7)) if d7 else '–'}, {len(d7)} pairs).",
    ]
    dsh = [db_of(f"{G}/diff/sharp/{c}-d1.{s}.txt") for c in CLIPS_A for s in ("s43", "s1234")]
    dsh = [x for x in dsh if x is not None]
    if dsh:
        lines += [
            "",
            f"The sharp 7B fp16's own seeds 43 and 1234 against its seed 42: {f2(np.mean(dsh))} dB mean "
            f"({f2(min(dsh))}–{f2(max(dsh))}, {len(dsh)} pairs; sharp-band.md).",
        ]
    shl = [L for L in overview if L.startswith("sh-")]
    if shl:
        lines += [
            "",
            "Bands of the sharp's files (sh-*, paired with the sharp 7B fp16 at seed 42): "
            + "; ".join(f"{L}: {band_count(overview[L])}" for L in shl)
            + " (the sharp 7B fp16's own 3-seed spread where its seeds 43 and 1234 are scored, the 7B "
            "fp16's 3-seed spread elsewhere).",
        ]
    if any(L.startswith("3b-") for L in overview):  # S16
        lines += ["", THREEB]
    if "q4k" in labels or "nq4km" in labels:
        lines += ["", *x_md(clip_bys, gl)]
    if "valsharp" in labels and "validation" in X:
        ag, tot, bad = X["validation"]
        lines += [
            "",
            f"Validation (valsharp = colour's sharp decodes through this pipeline): {ag}/{tot} of colour's T4 "
            f"cells agree; series not identical to colour's: {bad} (validation.md).",
        ]
    if X["skipped"]:
        lines += ["", f"JSONs being written, left out this run: {len(X['skipped'])}"]
    write(os.path.join(a.out, "00-overview.md"), lines)
    gl.save()
    print(
        f"ms_sum: {len(labels)} labels, cache {X['cache_hits']} hits / {X['cache_miss']} computed, "
        f"{len(X['skipped'])} skipped, {time.time() - t0:.1f} s"
    )


if __name__ == "__main__":
    main()

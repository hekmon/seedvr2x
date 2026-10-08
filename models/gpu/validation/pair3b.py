#!/usr/bin/env python3
"""S16 (model conversation, 2026-10-08): the 3B against the sharp 7B's 4 GB pick, slice A tier 1 (1080p ×2 from d1,
seed 42, the 8 d1 clips).

  $METRICS_PY $VAL_GLUE/pair3b.py [--cur 3b-cur] [--first 3b-first]
      [--dyn sh-dyn] [--out $VAL_STATE/sum] [--name 3b-00-overview] [--work $VAL_STATE/s16/pair3b-work]
      [--test]

Labels (the pool's): 3b-cur = our seedvr2x_ema_3b_fp16.safetensors (ByteDance's current 3B weights), 3b-first =
numz's seedvr2_ema_3b_fp16.safetensors (the first 3B weights), sh-dyn = the sharp 7B's dynamic GGUF (the 4 GB tier's
pick); for scale the sharp 7B fp16 (colour's sh42) and the 7B fp16 (colour's s42). --test: stand-ins (e.g. --cur
sh-q4k --first sh-q4ki), said on the page.
Read only on the pool's outputs (gpu/eval, gpu/vmaf, gpu/fr, gpu/bands, gpu/masters) and colour's (eval-b2, vmaf-b2,
bands-b2); ms_sum.py's own functions (imported, unchanged: loading, merging, pairing, bootstrap, bands):
 1. against the GT, per clip and variant (none, split:ycc:4:3): every guard and fidelity metric of the five (the run
    means; detail and the finest band ÷ the GT's), and their means over the clips;
 2. paired differences, A's frames minus B's at seed 42 (ms_sum.pair: colour_eval's 95% moving-block bootstrap
    interval, blocks of 8, 2000 draws; none for the finest band, degenerate for DISTS 5f): 3b-cur − 3b-first,
    3b-cur − sh-dyn, 3b-first − sh-dyn; B / W = A better / worse beyond the sharp 7B fp16's 3-seed spread with the
    interval excluding 0 (ms_sum's strict rule; detail ↑ / ↓), the difference's multiples of the sharp's 3-seed
    spread (colour's sh42, our s43, s1234: its own band) and of the 7B fp16's (for scale);
 3. per pair and variant, the clips where each side is closer to the GT (PSNR-Y, LPIPS, DISTS 45f), bands less
    (CAMBI added), flickers less (T-err, T-err lf), has more detail;
 4. decode-to-decode distances: RGB PSNR between the 16-bit none masters (ffv1_out.py --diff, cached in WORK/diff).
The 3B has no seed band of its own tonight (one seed): said on the page. Writes OUT/NAME.md and OUT/NAME.csv; its
pairing cache, GT Laplacian copy and distances in WORK (never the pool's sum/cache).
"""

import argparse
import concurrent.futures as cf
import csv
import hashlib
import math
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ms_sum as S  # noqa: E402

E, O, G = S.E, S.O, S.G
PY = S.env("METRICS_PY")
FFD = S.env("MEAS_SCRIPTS") + "/ffv1_out.py"
SUM = S.env("VAL_STATE") + "/sum"
MODELDIRS = (S.env("VAL_MODELDIR"), S.env("NUMZ_DIR") + "/models/SEEDVR2", S.env("NUMZ_MODELS"))
# key, header, closer to the GT when higher (None: a change, not a closeness)
MET = (
    ("psnr_y", "PSNR-Y", True),
    ("ssim_y", "SSIM-Y", True),
    ("vmaf", "VMAF", True),
    ("lpips", "LPIPS", False),
    ("dists", "DISTS 5f", False),
    ("dists_all", "DISTS 45f", False),
    ("cambi_added", "CAMBI+", False),
    ("de4", "ΔE00 lf", False),
    ("t_full", "T-err", False),
    ("t_lf", "T-err lf", False),
    ("lap", "Detail ÷ GT's", None),
    ("band", "Finest band ÷ GT's", None),
)
HEAD = dict((k, h) for k, h, _ in MET)
HIGH = dict((k, h) for k, _, h in MET)
DEC = {
    "psnr_y": 2,
    "vmaf": 2,
    "ssim_y": 4,
    "lpips": 4,
    "dists": 4,
    "dists_all": 4,
    "cambi_added": 3,
    "de4": 3,
    "t_full": 3,
    "t_lf": 3,
    "lap": 2,
    "band": 2,
}
SCORE_KEYS = ("psnr_y", "ssim_y", "vmaf", "lpips", "dists", "cambi_added", "de4", "t_full", "t_lf")
COUNTS = (
    ("psnr_y", "closer to the GT"),
    ("lpips", "closer to the GT"),
    ("dists_all", "closer to the GT"),
    ("cambi_added", "bands less"),
    ("t_full", "flickers less"),
    ("t_lf", "flickers less"),
    ("lap", "more detail"),
)


class Model:
    def __init__(self, name, tag, frlab, kind):
        self.name, self.tag, self.frlab, self.kind = (
            name,
            tag,
            frlab,
            kind,
        )  # kind: label, sharp, 7b

    def bands(self, cd):
        if self.kind == "sharp":
            return f"{O}/bands-b2/{cd}-sharp.json"
        if self.kind == "7b":
            return f"{O}/bands-b2/{cd}-7b.json"
        return f"{G}/bands/{self.tag}/{cd}.json"

    def master(self, cd):
        lab = {"sharp": "sharp", "7b": "7b"}.get(self.kind, self.tag)
        return f"{G}/masters/{lab}/{cd}.s42.none~{self.tag}.gbrp16le.mkv"


def fv(k, v):
    return "–" if v is None or not math.isfinite(v) else S.minus(f"{v:.{DEC[k]}f}")


def fd(k, v):
    return "–" if v is None or not math.isfinite(v) else S.minus(f"{v:+.{DEC[k]}f}")


def mt(m):
    return S.mult_txt(m) if m is not None else "–"


def spread(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return (max(vals) - min(vals)) if len(vals) > 1 else None


def seed_means(runs, k, series):
    return [
        float(np.mean(series(runs[s], k)))
        for s in S.SEEDS
        if s in runs and series(runs[s], k) is not None
    ]


def ffv1_db(a, b, work):
    """RGB PSNR (dB) between two 16-bit masters, ffv1_out.py --diff, cached by the masters' paths, sizes and mtimes."""
    try:
        sa, sb = os.stat(a), os.stat(b)
    except OSError:
        return None
    key = hashlib.sha1(
        f"{a}|{sa.st_size}|{int(sa.st_mtime)}|{b}|{sb.st_size}|{int(sb.st_mtime)}".encode()
    ).hexdigest()
    base = [os.path.basename(x).replace(".gbrp16le.mkv", "") for x in (a, b)]
    p = os.path.join(work, "diff", f"{base[0]}--{base[1]}.{key[:12]}.txt")
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        env = dict(
            os.environ,
            PATH=S.env("FFMPEG_BIN") + ":" + os.environ.get("PATH", ""),
            CUDA_VISIBLE_DEVICES="",
        )
        r = subprocess.run(
            ["taskset", "-c", "0-31", "nice", "-n", "19", PY, FFD, "--diff", a, b],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        if r.returncode:
            S.log(f"ffv1_out --diff {a} {b}: exit {r.returncode}: {r.stdout[-400:]}")
            return None
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            f.write(r.stdout)
        os.replace(p + ".tmp", p)
    return S.db_of(p)


def file_size(L):
    name = S.meta_of(L)
    for d in MODELDIRS:
        p = os.path.join(d, name)
        if name not in ("?", "") and os.path.exists(p):
            return name, os.path.getsize(os.path.realpath(p))
    return name, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cur", default="3b-cur")
    ap.add_argument("--first", default="3b-first")
    ap.add_argument("--dyn", default="sh-dyn")
    ap.add_argument("--out", default=SUM)
    ap.add_argument("--name", default="3b-00-overview")
    ap.add_argument("--work", default=S.env("VAL_STATE") + "/s16/pair3b-work")
    ap.add_argument("--test", action="store_true", help="stand-in labels: said on the page")
    ap.add_argument("--jobs", type=int, default=4, help="ffv1_out.py --diff processes at once")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.work, exist_ok=True)
    S.X["out"] = a.work  # the pairing cache: WORK/cache
    gp = os.path.join(a.work, "gt_lap.json")
    if not os.path.exists(gp) and os.path.exists(f"{SUM}/gt_lap.json"):
        shutil.copy(f"{SUM}/gt_lap.json", gp)
    gl = S.GtLap(gp)
    cur, first, dyn = (Model(x, x, x, "label") for x in (a.cur, a.first, a.dyn))
    sharp, b7 = Model("sharp fp16", "sharp", "sharp", "sharp"), Model("7B fp16", "f32", "7b", "7b")
    models = (cur, first, dyn, sharp, b7)
    names = {
        cur.tag: "3b-cur" if not a.test else f"3b-cur := {a.cur}",
        first.tag: "3b-first" if not a.test else f"3b-first := {a.first}",
    }
    pairs = ((cur, first), (cur, dyn), (first, dyn))
    rows_csv = []
    vals = {}  # (model tag, clip, short) -> {k: value}
    pdat = {}  # (A tag, B tag, clip, short) -> {k: {mean, lo, hi, ssh, s7, msh, m7, tag}}
    have = {m.tag: 0 for m in models}
    for c in S.CLIPS_A:
        cd = f"{c}-d1"
        by = S.merged(
            [r for s in S.clip_sources(cd, [cur.tag, first.tag, dyn.tag]) for r in S.load_dir(s)]
        )
        fr = {m.frlab: S.fr_runs(m.frlab, cd) for m in models}
        gtlap = None
        try:
            gtlap = gl.get(f"{S.CLIPS}/{c}.gt.mkv", 45)
        except Exception as ex:  # noqa: BLE001
            S.log(f"GT Laplacian {cd}: {ex}")
        bsh = S.bands_sharp(cd)
        bb7 = S.bands_7b(cd)
        for m in models:
            have[m.tag] += f"none@{m.tag}" in by and "s42" in by[f"none@{m.tag}"]
        for v, short in S.VARS:
            sig = "c" if v == "none" else "s"
            for m in models:
                run = by.get(f"{v}@{m.tag}", {}).get("s42")
                d = {}
                if run:
                    for k in SCORE_KEYS:
                        d[k] = E.run_mean(run, k)
                    lp = E.run_mean(run, "lap")
                    d["lap"] = lp / gtlap if lp is not None and gtlap else None
                frr = fr[m.frlab].get(f"{v}@{m.tag}", {}).get("s42")
                x = S.fr_series(frr, "dists") if frr else None
                d["dists_all"] = float(np.mean(x)) if x is not None else None
                bj = S.load_json(m.bands(cd))
                d["band"], d["band_corr"] = S.band1(bj, sig) if bj else (None, None)
                if run or frr or bj:
                    vals[(m.tag, c, short)] = d
                    for k, _, _ in MET:
                        rows_csv.append(
                            ["value", m.tag, "", c, short, k, d.get(k), "", "", "", "", "", "", ""]
                        )
            # the two bands for scale: the sharp's 3 seeds (its own), the 7B's 3 seeds
            rsh, r7 = by.get(f"{v}@sharp", {}), by.get(f"{v}@f32", {})
            frsh, fr7 = fr["sharp"].get(f"{v}@sharp", {}), fr["7b"].get(f"{v}@f32", {})
            ssh = {k: spread(seed_means(rsh, k, E.series)) for k in SCORE_KEYS + ("lap",)}
            s7 = {k: spread(seed_means(r7, k, E.series)) for k in SCORE_KEYS + ("lap",)}
            ssh["dists_all"], s7["dists_all"] = (
                spread(seed_means(frsh, "dists", S.fr_series)),
                spread(seed_means(fr7, "dists", S.fr_series)),
            )
            ssh["band"] = spread([S.band1(j, sig)[0] for j in bsh.values() if j])
            s7["band"] = spread([S.band1(j, sig)[0] for j in bb7.values() if j])
            for A, B in pairs:
                if f"{v}@{A.tag}" not in by or f"{v}@{B.tag}" not in by:
                    continue
                P = dict(S.pair(by, f"{v}@{A.tag}", f"{v}@{B.tag}", f"{v}@sharp"))
                F = S.fr_pair(
                    fr[A.frlab],
                    fr[B.frlab],
                    fr["sharp"],
                    f"{v}@{A.tag}",
                    f"{v}@{B.tag}",
                    f"{v}@sharp",
                )
                if "dists" in F:
                    P["dists_all"] = F["dists"]
                out = {}
                for k, _, _ in MET:
                    if k == "band":
                        va, vb = (
                            vals.get((A.tag, c, short), {}).get("band"),
                            vals.get((B.tag, c, short), {}).get("band"),
                        )
                        if va is None or vb is None:
                            continue
                        mean, lo, hi = va - vb, None, None
                        sp = ssh["band"]
                        tag = (
                            ("↑" if mean > 0 else "↓") if sp is not None and abs(mean) > sp else ""
                        )
                    elif k in P:
                        e = P[k]
                        mean, lo, hi, tag = e["mean"], e["lo"], e["hi"], e["tag"]
                        if k == "lap":
                            if not gtlap:
                                continue
                            mean, lo, hi = mean / gtlap, lo / gtlap, hi / gtlap
                    else:
                        continue
                    shs = (
                        ssh.get(k)
                        if k != "lap"
                        else (ssh["lap"] / gtlap if ssh["lap"] is not None and gtlap else None)
                    )
                    s7k = (
                        s7.get(k)
                        if k != "lap"
                        else (s7["lap"] / gtlap if s7["lap"] is not None and gtlap else None)
                    )
                    msh = S.mult_of(mean, shs) if shs is not None else None
                    m7 = S.mult_of(mean, s7k) if s7k is not None else None
                    out[k] = {
                        "mean": mean,
                        "lo": lo,
                        "hi": hi,
                        "ssh": shs,
                        "s7": s7k,
                        "msh": msh,
                        "m7": m7,
                        "tag": tag,
                    }
                    rows_csv.append(
                        ["pair", A.tag, B.tag, c, short, k, mean, lo, hi, shs, s7k, msh, m7, tag]
                    )
                pdat[(A.tag, B.tag, c, short)] = out
    # 4. distances between decodes (none masters)
    dpairs = (
        (cur, first),
        (cur, sharp),
        (first, sharp),
        (cur, dyn),
        (first, dyn),
        (dyn, sharp),
        (b7, sharp),
        (cur, b7),
        (first, b7),
    )
    todo = [(A, B, c) for A, B in dpairs for c in S.CLIPS_A]
    dist = {}
    with cf.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        futs = {
            ex.submit(ffv1_db, A.master(f"{c}-d1"), B.master(f"{c}-d1"), a.work): (A.tag, B.tag, c)
            for A, B, c in todo
        }
        for fu in cf.as_completed(futs):
            dist[futs[fu]] = fu.result()
    for (x, y, c), db in sorted(dist.items()):
        rows_csv.append(
            ["distance_db", x, y, c, "none", "rgb_psnr_db", db, "", "", "", "", "", "", ""]
        )

    def nm(m):
        return names.get(m.tag, {"sharp": "sharp fp16 s42", "f32": "7B fp16 s42"}.get(m.tag, m.tag))

    # ---------------------------------------------------------------- the page
    sz = {m.tag: file_size(m.tag) for m in (cur, first, dyn)}
    szt = "; ".join(f"{names.get(t, t)}: `{n}` {s / 1e9:.2f} GB" for t, (n, s) in sz.items() if s)
    inp = []
    for m in (cur, first):
        p = f"{SUM}/inputs-{m.tag}-anime-clean-d1.txt"
        try:
            with open(p, encoding="utf-8") as f:
                lines_ = [x.strip() for x in f.read().splitlines() if x.strip()]
            inp.append(f"{names.get(m.tag, m.tag)}: " + "; ".join(lines_[1:]))
        except OSError:
            inp.append(f"{names.get(m.tag, m.tag)}: not checked yet ({p})")
    d7 = [S.db_of(f"{G}/diff/7b/{c}-d1.{s}.txt") for c in S.CLIPS_A for s in ("s43", "s1234")]
    d7 = [x for x in d7 if x is not None]
    dsh = [S.db_of(f"{G}/diff/sharp/{c}-d1.{s}.txt") for c in S.CLIPS_A for s in ("s43", "s1234")]
    dsh = [x for x in dsh if x is not None]
    lines = [
        "# The 3B against the sharp 7B's 4 GB pick: slice A tier 1 (1080p ×2 from d1, seed 42, the 8 d1 clips)",
        "",
    ]
    if a.test:
        lines += [
            f"**TEST with stand-ins, not the 3B:** 3b-cur := {a.cur}, 3b-first := {a.first}, sh-dyn := {a.dyn}.",
            "",
        ]
    lines += [
        f"Generated {S.stamp()} by {os.path.abspath(__file__)} (ms_sum.py's loading, merging and pairing, "
        "imported). Labels: 3b-cur = our seedvr2x_ema_3b_fp16.safetensors (ByteDance's current 3B weights), "
        "3b-first = numz's seedvr2_ema_3b_fp16.safetensors (the first 3B weights), sh-dyn = the sharp 7B's "
        "dynamic GGUF (the 4 GB tier's pick); for scale the sharp 7B fp16 (colour's sh42) and the 7B fp16 "
        "(colour's s42). Scored as every file so far (colour_eval.py with colour's baton-2 commands, against "
        "the GT; split reference = colour's 7B s42 ref_f32.pt). Clips scored: "
        + ", ".join(f"{nm(m)} {have[m.tag]}/{len(S.CLIPS_A)}" for m in models)
        + ".",
        "",
        "- **The 3B has no seed band of its own tonight:** each 3B file ran seed 42 only, so every B / W and "
        "multiple below is set against the sharp 7B fp16's own 3-seed spread (colour's sh42, our s43 and s1234), "
        "with the 7B fp16's 3-seed spread for scale, not against the 3B's own seed scatter, unmeasured.",
        "- **Size:** the 3B float16 file 6.8 GB against 4.8 GB for the sharp 7B's dynamic GGUF (sh-dyn)"
        + (f" (on disk: {szt})" if szt else "")
        + ".",
        "- **The 3B's inputs** (its anime-clean runs' encoder input and reference against colour's 7B s42 ones, "
        "md5): " + " — ".join(inp) + ".",
        f"- For scale: the 7B fp16's seeds 43 and 1234 against its 42: {S.f2(np.mean(d7)) if d7 else '–'} dB "
        f"mean ({S.f2(min(d7)) if d7 else '–'}–{S.f2(max(d7)) if d7 else '–'}); the sharp 7B fp16's own: "
        f"{S.f2(np.mean(dsh)) if dsh else '–'} dB mean ({S.f2(min(dsh)) if dsh else '–'}–"
        f"{S.f2(max(dsh)) if dsh else '–'}).",
        "",
    ]
    # 1. means over the clips
    lines += [
        "## 1. Against the GT: means over the clips each has (per-clip values: section 5)",
        "",
    ]
    rows = []
    for v, short in S.VARS:
        for m in models:
            ds = [vals[(m.tag, c, short)] for c in S.CLIPS_A if (m.tag, c, short) in vals]
            if not ds:
                continue
            cells = []
            for k, _, _ in MET:
                xs = [d.get(k) for d in ds if d.get(k) is not None]
                cells.append(
                    fv(k, float(np.mean(xs)))
                    + ("" if len(xs) == len(S.CLIPS_A) else f" ({len(xs)})")
                    if xs
                    else "–"
                )
            rows.append([nm(m), short] + cells)
    lines += S.table(["Model", "Var"] + [HEAD[k] for k, _, _ in MET], rows) + [""]
    lines += [
        "Higher is closer to the GT for PSNR-Y, SSIM-Y, VMAF; lower for LPIPS, DISTS, ΔE00 lf; lower CAMBI+ = "
        "bands less, lower T-err = flickers less; detail and the finest band: the redraw's texture against the "
        "GT's (1.00 = the GT's), not a closeness.",
        "",
    ]
    # 2. counts per pair
    lines += [
        "## 2. Per pair: clips where each side is closer to the GT, bands less, flickers less, has more detail",
        "",
        "A n · B n · = n (ties: the same value at the printed resolution); in brackets the clips where the "
        "difference is beyond the sharp 7B fp16's 3-seed spread with the interval excluding 0 (ms_sum's strict "
        "rule): A n · B n.",
        "",
    ]
    rows = []
    for A, B in pairs:
        for v, short in S.VARS:
            cells = []
            for k, what in COUNTS:
                na = nb = nt = sa = sb = 0
                for c in S.CLIPS_A:
                    e = pdat.get((A.tag, B.tag, c, short), {}).get(k)
                    if not e:
                        continue
                    if abs(e["mean"]) < 0.5 * 10 ** -DEC[k]:
                        nt += 1
                    elif k == "lap":
                        na += e["mean"] > 0
                        nb += e["mean"] < 0
                    else:
                        better = (e["mean"] > 0) == bool(HIGH[k])
                        na += better
                        nb += not better
                    t = e["tag"]
                    sa += t in ("B", "↑")
                    sb += t in ("W", "↓")
                cells.append(f"A {na} · B {nb} · = {nt} ({sa} · {sb})" if na + nb + nt else "–")
            rows.append([f"{nm(A)} − {nm(B)}", short] + cells)
    lines += S.table(["A − B", "Var"] + [f"{HEAD[k]}: {what}" for k, what in COUNTS], rows) + [""]
    # 3. distances
    lines += [
        "## 3. Distances between the decodes (dB: RGB PSNR between the 16-bit none masters, seed 42)",
        "",
    ]
    rows = []
    for c in S.CLIPS_A + ("mean",):
        cells = []
        for A, B in dpairs:
            if c == "mean":
                xs = [dist.get((A.tag, B.tag, x)) for x in S.CLIPS_A]
                xs = [x for x in xs if x is not None and math.isfinite(x)]
                cells.append(
                    f"{S.f2(np.mean(xs))} ({S.f2(min(xs))})"
                    + ("" if len(xs) == len(S.CLIPS_A) else f" [{len(xs)}]")
                    if xs
                    else "–"
                )
            else:
                cells.append(S.f2(dist.get((A.tag, B.tag, c))))
        rows.append([c if c != "mean" else "mean (worst)"] + cells)
    lines += S.table(["Clip"] + [f"{nm(A)} – {nm(B)}" for A, B in dpairs], rows) + [""]
    # 4. paired differences
    lines += [
        "## 4. Paired differences per clip: A − B frames at seed 42",
        "",
        "Cell: the difference [colour_eval's 95% interval]; **B / W** = A better / worse than B beyond the sharp "
        "7B fp16's 3-seed spread with the interval excluding 0, detail / finest band ↑ ↓ = A more / less; then "
        "the difference's multiples of the sharp's 3-seed spread / of the 7B fp16's (for scale). Detail and "
        "finest band in units of the GT's; the finest band has no interval; DISTS 5f's is degenerate (5 frames).",
        "",
    ]
    for A, B in pairs:
        for v, short in S.VARS:
            rows = []
            for c in S.CLIPS_A:
                out = pdat.get((A.tag, B.tag, c, short))
                if not out:
                    continue
                cells = []
                for k, _, _ in MET:
                    e = out.get(k)
                    if not e:
                        cells.append("–")
                        continue
                    s = fd(k, e["mean"]) + (f" **{e['tag']}**" if e["tag"] else "")
                    if e["lo"] is not None:
                        s += f" [{fd(k, e['lo'])}, {fd(k, e['hi'])}]"
                    s += f" {mt(e['msh'])}/{mt(e['m7'])}"
                    cells.append(s)
                rows.append([c] + cells)
            if rows:
                lines += [f"### {nm(A)} − {nm(B)}, {short}", ""]
                lines += S.table(["Clip"] + [HEAD[k] for k, _, _ in MET], rows) + [""]
    # 5. values per clip
    lines += [
        "## 5. Against the GT, per clip (seed 42; finest band: rms ÷ the GT's · its correlation with the GT)",
        "",
    ]
    for c in S.CLIPS_A:
        rows = []
        for v, short in S.VARS:
            for m in models:
                d = vals.get((m.tag, c, short))
                if not d:
                    continue
                cells = [
                    fv(k, d.get(k))
                    if k != "band"
                    else (
                        fv("band", d.get("band"))
                        + (f" · {d['band_corr']:.2f}" if d.get("band_corr") is not None else "")
                    )
                    for k, _, _ in MET
                ]
                rows.append([nm(m), short] + cells)
        if rows:
            lines += [f"### {c} ({S.KIND[c]})", ""]
            lines += S.table(["Model", "Var"] + [HEAD[k] for k, _, _ in MET], rows) + [""]
    if S.X["skipped"]:
        lines += [f"JSONs being written, left out this run: {len(S.X['skipped'])}", ""]
    S.write(os.path.join(a.out, f"{a.name}.md"), lines)
    cp = os.path.join(a.out, f"{a.name}.csv")
    with open(cp + ".tmp", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "section",
                "a",
                "b",
                "clip",
                "variant",
                "metric",
                "value",
                "lo",
                "hi",
                "spread_sharp3",
                "spread_7b3",
                "mult_sharp",
                "mult_7b",
                "tag",
            ]
        )
        for r in rows_csv:
            w.writerow(
                ["" if x is None else (f"{x:.6g}" if isinstance(x, float) else x) for x in r]
            )
    os.replace(cp + ".tmp", cp)
    gl.save()
    print(
        f"pair3b: {', '.join(f'{nm(m)} {have[m.tag]}/8' for m in models)}; distances {sum(x is not None for x in dist.values())}"
        f"/{len(dist)}; cache {S.X['cache_hits']} hits / {S.X['cache_miss']} computed; {os.path.join(a.out, a.name)}.md"
    )


if __name__ == "__main__":
    main()

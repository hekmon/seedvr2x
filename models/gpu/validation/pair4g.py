#!/usr/bin/env python3
"""S6 (model conversation, 2026-10-07): the 4 GB files against our static Q4_K (label q4k), slice A tier 1.

  $METRICS_PY $VAL_GLUE/pair4g.py [--labels dyn,q4ki,nv4a16,nv4a4,nq4km]
      [--out $VAL_STATE/s6/sum]

Read only on the pool's outputs (gpu/eval, gpu/vmaf, gpu/fr, gpu/bands, gpu/diff, gpu/masters); ms_sum.py's own
functions (imported, unchanged): for each label L and clip, L's frames minus q4k's at seed 42 (pair(), the 7B
fp16's 3-seed spread as the band, colour_eval's bootstrap; B = L better than q4k, W = worse; Laplacian up = L
sharper), DISTS 45f from fr_metrics.py's JSONs (fr_pair()); each label's own detail, band energy and dB to the 7B
fp16 s42 (label_data()); and the dB between L's none master and q4k's (ffv1_out.py --diff, written in OUT/diff).
Writes OUT/pair4g.md; its pairing cache in OUT/cache (never the pool's).
S23 (2026-10-08, design's decision of 10:20: one rule at 1080p and 4K): the band is the 7B fp16's 3-seed spread or its
floor (ms_floor.py), in both rules and every printed multiple (a multiple of the floor carries a †), on the pairs and
on each label's label_data().
"""

import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ms_sum as S  # noqa: E402
import ms_floor as FL  # noqa: E402  S23: the floor at 1080p too

FL.install(S)

KEYS = (
    "psnr_y",
    "ssim_y",
    "vmaf",
    "lpips",
    "dists",
    "dists_all",
    "cambi_added",
    "de4",
    "t_full",
    "t_lf",
    "lap",
)
HEAD = (
    "PSNR-Y",
    "SSIM-Y",
    "VMAF",
    "LPIPS",
    "DISTS 5f",
    "DISTS 45f",
    "CAMBI+",
    "ΔE00 lf",
    "T-err",
    "T-err lf",
    "Laplacian",
)
PY = S.env("METRICS_PY")
FFD = S.env("MEAS_SCRIPTS") + "/ffv1_out.py"


def master(L, cd):
    return f"{S.G}/masters/{L}/{cd}.s42.none~{L}.gbrp16le.mkv"


def db_to_q4k(L, cd, out, base="q4k"):
    p = os.path.join(out, "diff", f"{cd}.{base}-{L}.txt")
    if not os.path.exists(p):
        a, b = master(base, cd), master(L, cd)
        if not (os.path.exists(a) and os.path.exists(b)):
            return None
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
            S.log(f"ffv1_out --diff {cd} q4k-{L}: exit {r.returncode}: {r.stdout[-400:]}")
            return None
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            f.write(r.stdout)
        os.replace(p + ".tmp", p)
    return S.db_of(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="dyn,q4ki,nv4a16,nv4a4,nq4km")
    ap.add_argument("--out", default=S.env("VAL_STATE") + "/s6/sum")
    ap.add_argument(
        "--base", default="q4k", help="the label every other is paired with (default q4k)"
    )
    ap.add_argument("--name", default="pair4g.md")
    a = ap.parse_args()
    labels = [x for x in a.labels.split(",") if x]
    os.makedirs(a.out, exist_ok=True)
    S.X["out"] = a.out
    gp = os.path.join(a.out, "gt_lap.json")
    if not os.path.exists(gp) and os.path.exists(S.env("VAL_STATE") + "/sum/gt_lap.json"):
        shutil.copy(S.env("VAL_STATE") + "/sum/gt_lap.json", gp)
    gl = S.GtLap(gp)
    B = a.base
    data = {L: S.label_data(L, gl) for L in [B] + labels}
    for d in data.values():
        FL.apply(S, d)  # S23: the floor at 1080p too
    lines = [
        "# The 4 GB files against our static Q4_K (q4k), slice A tier 1 (1080p ×2 from d1, seed 42)",
        "",
        f"Generated {S.stamp()} by {os.path.abspath(__file__)} (ms_sum.py's pairing, imported). Cells: the "
        "label's frames minus q4k's at seed 42; **B / W** = the label better / worse than q4k (bootstrap interval "
        "excludes 0 AND |difference| > the 7B fp16's 3-seed spread); Laplacian ↑ = sharper than q4k; multiples "
        "of the spread; bold = beyond 2.7 × the spread (calibrated). dB L–q4k = RGB PSNR between the two none "
        "masters (for scale: the 7B fp16's seeds 43, 1234 vs 42: 40.62 dB mean). "
        + FL.FLOOR_TXT_1080,
        "",
    ]
    summary = []
    for L in labels:
        rows, tally = [], {k: [0, 0, 0] for k in KEYS}
        for c in S.CLIPS_A:
            cd = f"{c}-d1"
            by = S.merged([r for s in S.clip_sources(cd, [B, L]) for r in S.load_dir(s)])
            frL, frQ, fr7 = S.fr_runs(L, cd), S.fr_runs(B, cd), S.fr_runs("7b", cd)
            for v, short in S.VARS:
                if f"{v}@{L}" not in by or f"{v}@{B}" not in by:
                    continue
                P = dict(S.pair(by, f"{v}@{L}", f"{v}@{B}", f"{v}@f32"))
                F = S.fr_pair(frL, frQ, fr7, f"{v}@{L}", f"{v}@{B}", f"{v}@f32")
                if "dists" in F:
                    P["dists_all"] = F["dists"]
                FL.apply_pairs(S, P, data[L][c]["none"]["gtlap"])  # S23: the floor at 1080p too
                for k in KEYS:
                    if k in P:
                        t = P[k]["tag"]
                        tally[k][0 if t in ("B", "↑") else 1 if t in ("W", "↓") else 2] += 1
                db = db_to_q4k(L, cd, a.out, B) if short == "none" else None
                rows.append(
                    [c, short]
                    + [S.cell(k, P) for k in KEYS]
                    + [S.f2(db) if short == "none" else ""]
                )
        lines += [f"## {L} − {B}", ""]
        lines += S.table(["Clip", "Var", *HEAD, f"dB {L}–{B}"], rows) + [""]
        lines += [
            "Tally over the clip-variants (B or ↑ / W or ↓ / within): "
            + "; ".join(
                f"{h} {t[0]}/{t[1]}/{t[2]}" for h, t in zip(HEAD, tally.values()) if sum(t)
            ),
            "",
        ]
        summary.append((L, tally))
    # each label's own figures against the 7B fp16 (as ms_sum.py's pages), side by side with q4k's
    lines += [
        "## Each file against the 7B fp16 s42: dB, detail and band energy (ms_sum.py's label_data)",
        "",
    ]
    rows = []
    for L in [B] + labels:
        d = data[L]
        dbs = [d[c]["db"] for c in d if d[c]["db"] is not None]
        rows.append(
            [L, f"{len(d)}/8", f"{S.f2(np.mean(dbs))} ({S.f2(min(dbs))})" if dbs else "–"]
            + [S.f2(d[c]["db"]) if c in d else "–" for c in S.CLIPS_A]
        )
    lines += S.table(["Label", "Clips", "dB to 7B s42: mean (worst)", *S.CLIPS_A], rows) + [""]
    for c in (
        "anime-sky",
        "live-slow",
        "live-vfx",
        "anime-bright",
        "anime-grain",
        "anime-clean",
        "cartoon-bright",
        "anime-dark",
    ):
        rows = []
        for L in [B] + labels:
            if c not in data[L]:
                continue
            for v, short in S.VARS:
                e = data[L][c][short]
                rows.append([L, short, S.det_cell(e), S.band_cell(e)])
        lines += [
            f"### {c}: detail and band energy (value ÷ GT's; in brackets the 7B s42 value; the 7B's 3-seed "
            "range; ↑ / ↓ against the 7B s42 with the multiple of the spread)",
            "",
        ]
        lines += S.table(["Label", "Var", "Detail ÷ GT's", "Finest band ÷ GT's · corr"], rows) + [
            ""
        ]
    lines[0] = lines[0].replace(
        "our static Q4_K (q4k)", f"{B}" if B != "q4k" else "our static Q4_K (q4k)"
    )
    lines[2] = (
        lines[2]
        .replace("q4k's", f"{B}'s")
        .replace("than q4k", f"than {B}")
        .replace("L–q4k", f"L–{B}")
    )
    S.write(os.path.join(a.out, a.name), lines)
    gl.save()
    print(
        f"pair4g: {len(labels)} labels, cache {S.X['cache_hits']} hits / {S.X['cache_miss']} computed; "
        f"{os.path.join(a.out, a.name)}"
    )


if __name__ == "__main__":
    main()

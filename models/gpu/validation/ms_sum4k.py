#!/usr/bin/env python3
"""Model conversation, slice B (4K x2 from d1, seed 42, the 6 batch-A shots that have two 7B fp16 seeds): each
phase-2 file's runs against the 7B fp16, per shot and metric, then PASS / FAIL per guard and kind of source (Sol
Levante = painted anime, 4 shots; digital, 2 shots); one Markdown file per label (4k-<label>.md), an overview
(4k-00-overview.md), the 7B fp16's own 2-seed band (4k-7b-band.md) and, with the test labels val4k / val4k43,
the validation against colour's JSONs (4k-validation.md). ms_sum.py's code, imported (its loading, merging,
pairing, verdicts, guards and cells: the same rules), on slice B's shots and sources.

  ms_sum4k.py [--out $VAL_STATE/sum]

Sources: the 7B fp16 at seeds 42 and 43 = colour's own 4K scores where colour has them (eval-b1/<cd>: score, s42
and s43; vmaf-b1/<cd>: split VMAF + CAMBI, s42 and s43; vmaf-b2/<cd>: none VMAF + CAMBI, s42; bands/<cd>.json: s42),
ours for the rest, the same commands (gpu/vmaf/7b/<cd>: s43 none and split; gpu/bands/7b/<cd>-s43.json;
gpu/diff/7b/<cd>.s43.txt); a label's = gpu/eval/<label>/<cd>, gpu/vmaf/<label>/<cd>, gpu/bands/<label>/<cd>.json,
gpu/diff/<label>/<cd>.s42.txt, gpu/fr/<label>/<cd> (fr_metrics.py, only while score/FR4_ON exists). The sharp 7B's
files (sh-*) are not this script's since S9 (2026-10-07): their slice B runs on the sharp's own 5 two-seed shots,
against the sharp fp16 with its own 2-seed band: ms_sum4ksh.py (4k-sh-*.md).
Verdicts (ms_sum.py's): the label's frames minus the 7B's at seed 42; B / W when colour_eval's 95% moving-block
bootstrap interval excludes 0 AND |mean| exceeds the band = the 7B fp16's spread over the seeds it has (42, 43);
Laplacian up / down the same; band energy (no interval): |file - 7B at s42| beyond the 7B's 2-seed spread, either
way, the spread alone deciding (as DISTS 5f; since 2026-10-07 05:30). A guard FAILs on a kind of
source when it is worse on at least one of its shots. Two rules, both reported (since 07:15): strict (the above) and
calibrated = a strict failure whose |difference| also exceeds K x the spread, K = 10.9 with 2 seeds (S4's Monte
Carlo: a further fp16 seed exceeds it with 5% probability per shot-variant).
S16 (2026-10-08, design's decision of 2026-10-07 23:55): the 4K floor (ms_floor.py): the spread used = max(the 2-seed
spread, one unit of the resolution the metric is printed at), in both rules and every printed multiple (a multiple of
the floor carries a †); applied to label_data()'s entries before the pages are made.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ms_sum as S  # noqa: E402
import ms_floor as F  # noqa: E402  S16: the 4K floor

F.install(S)

E, O, G = S.E, S.O, S.G
C4 = S.env("MEAS_SHOTS")
SHOTS = (
    "sollevante-painted",
    "sollevante-line",
    "sollevante-action",
    "sollevante-dark",
    "digital-sunrise",
    "digital-space",
)
KIND = {c: ("Sol Levante" if c.startswith("sollevante") else "digital") for c in SHOTS}
KINDS = ("Sol Levante", "digital")
KTITLE = {"Sol Levante": "Sol Levante, painted anime", "digital": "digital"}
NFR = {"sollevante-dark": 41}
VAL = {
    "val4k": "colour's sharp 7B sh42 decode (validation)",
    "val4k43": "colour's 7B fp16 s43 decode (validation)",
}


def clip_sources4(cd, labels):
    srcs = [
        f"{O}/eval-b1/{cd}",
        f"{O}/vmaf-b1/{cd}",
        f"{O}/eval-b2/{cd}",
        f"{O}/vmaf-b2/{cd}",
        f"{G}/eval/7b/{cd}",
        f"{G}/vmaf/7b/{cd}",
    ]
    for L in labels:
        srcs += [f"{G}/eval/{L}/{cd}", f"{G}/vmaf/{L}/{cd}"]
    return srcs


def bands_7b4(cd):
    return {
        "s42": S.load_json(f"{O}/bands/{cd}.json"),
        "s43": S.load_json(f"{G}/bands/7b/{cd}-s43.json"),
    }


# ms_sum.py's functions read these module names at call time: slice B's shots, kinds, GT dir and sources
S.CLIPS_A, S.KIND, S.KINDS, S.CLIPS = SHOTS, KIND, KINDS, C4
S.clip_sources, S.bands_7b = clip_sources4, bands_7b4
_meta_of = S.meta_of
S.meta_of = lambda L: VAL.get(L) or _meta_of(L)

RULE4 = (
    "Cells: the file's output minus the 7B fp16's at seed 42, frame by frame (colour_eval.py's scores with colour's "
    "batch-A / baton-2 4K commands: whole frames (--bars 0:0), --lpips, --dists-every 9, 8 threads; VMAF v1's 2160 "
    "model, CAMBI); **B / W** = better / worse when colour_eval's 95% moving-block bootstrap interval (blocks of 8, "
    "2000 draws) excludes 0 AND |difference| exceeds the 7B fp16's 2-seed spread (|s42 − s43| of its per-seed "
    "means); Laplacian ↑ / ↓ the same; no tag = within. DISTS 5f = colour_eval's every 9th frame (5 frames: its "
    "interval is degenerate, the spread alone decides); DISTS 45f = fr_metrics.py's every frame, where present. "
    "Detail = the luma Laplacian variance ÷ the GT's; band = colour_bands.py's finest band (below 0.7 px), rms ÷ the "
    "GT's · its correlation with the GT (none: the raw decode c; split: s), ↑ / ↓ when |the file's − the 7B's at "
    "seed 42| exceeds the 7B's 2-seed spread (no interval: the spread alone decides, as DISTS 5f); in brackets after "
    "Detail and band: the 7B's seed-42 value; its 2-seed range; dB = RGB PSNR between the 16-bit none masters "
    "(ffv1_out.py --diff) of the file and the 7B fp16 at seed 42. Each W, ↑ or ↓ shows its multiple of the spread "
    "(|difference| ÷ the 7B's 2-seed spread). Two rules: **strict** (the above) and **calibrated** = strict AND "
    "|difference| > K × the spread, K = 10.9 with 2 seeds: the multiple a further fp16 seed would exceed with 5% "
    "probability per shot-variant (S4's Monte Carlo, Gaussian seed scatter, both directions), where the strict "
    "rule fails such a seed 50% of the time; **bold** cell = worse by the calibrated rule too. "
    + F.FLOOR_TXT
)


def band_note(clip_bys):
    full = sum(
        len(clip_bys[c].get(f"{v}@f32", {}).keys() & {"s42", "s43"}) == 2
        for c in SHOTS
        for v, _ in S.VARS
    )
    return f"The 7B fp16's band from 2 seeds (42, 43) on {full}/{2 * len(SHOTS)} shot-variants" + (
        "" if full == 2 * len(SHOTS) else " (one seed elsewhere: band 0, verdicts more eager)"
    )


def mean_cell(k, es):
    if not es:
        return "–"
    b = sum(e["tag"] == "B" for e in es)
    w = sum(e["tag"] == "W" for e in es)
    return f"{S.fmt(k, float(np.mean([e['mean'] for e in es])))} ({b}/{w}/{len(es) - b - w})"


def label_md4(L, data, note):
    n = len(data)
    rname = S.ref_of(L)[1]
    miss = [c for c in SHOTS if c not in data]
    lines = [
        f"# {L} − {rname}: slice B (4K ×2 from d1, seed 42; the 6 batch-A shots with two 7B seeds)",
        "",
        f"Generated {S.stamp()} by ms_sum4k.py. File: `{S.meta_of(L)}`. Shots scored: {n}/{len(SHOTS)}"
        + (f" (missing: {', '.join(miss)})" if miss else "")
        + f". {note}.",
        "",
        RULE4,
        "",
    ]
    if rname != "7B fp16":
        lines += [
            f"**This file is paired with the {rname} at seed 42** (colour's 4K sh42 scores and scan; dB to its "
            "none master), not with the 7B fp16; the band is still the 7B fp16's 2-seed spread, the band-energy "
            "range the sharp's s42 value moved by the 7B's seed deviations.",
            "",
        ]
    nk = {kd: sum(KIND[c] == kd for c in SHOTS) for kd in KINDS}
    for mode, head in (
        ("tag", "strict rule"),
        (
            "tagc",
            f"calibrated rule (strict AND beyond {S.KCAL[2]} × the 2-seed spread or its floor)",
        ),
    ):
        lines += [f"## Guards per kind of source, {head} (none · split:ycc:4:3)", ""]
        kn, ks = S.kind_cells(data, "none", mode), S.kind_cells(data, "split", mode)
        rows = [[title] + [f"{kn[g][kd]} · {ks[g][kd]}" for kd in KINDS] for g, title in S.GUARDS]
        lines += S.table(["Guard"] + [f"{KTITLE[kd]} ({nk[kd]})" for kd in KINDS], rows) + [""]
    wm = S.worst_mult(data)
    lines += [
        f"Worst multiple of the spread among the strict failures: {S.mult_txt(wm[0])}"
        + (f" ({wm[1]})" if wm[1] else "")
        + ".",
        "",
    ]
    lines += [
        "## Fidelity (reported, not a guard): mean of the per-shot differences (B/W/within)",
        "",
    ]
    rows = []
    for v, short in S.VARS:
        for kd in KINDS + ("all",):
            shots = [c for c in data if kd == "all" or KIND[c] == kd]
            if not shots:
                continue
            rows.append(
                [v if v == "none" else "ycc:4:3", KTITLE.get(kd, kd), str(len(shots))]
                + [
                    mean_cell(
                        k,
                        [data[c][short]["pairs"][k] for c in shots if k in data[c][short]["pairs"]],
                    )
                    for k in ("psnr_y", "ssim_y", "vmaf", "lpips", "dists")
                ]
            )
    lines += S.table(
        ["Variant", "Kind", "Shots", "PSNR-Y", "SSIM-Y", "VMAF", "LPIPS", "DISTS 5f"], rows
    )
    dbs = [data[c]["db"] for c in data if data[c]["db"] is not None]
    d7 = [data[c]["db7"]["s43"] for c in data if data[c]["db7"]["s43"] is not None]
    lines += [
        "",
        f"Distance to the {rname} s42 (none masters): {S.f2(np.mean(dbs)) if dbs else '–'} dB mean, "
        f"{S.f2(min(dbs)) if dbs else '–'} worst; the 7B's own seed 43 against its 42: "
        f"{S.f2(np.mean(d7)) if d7 else '–'} dB mean ({S.f2(min(d7)) if d7 else '–'}–{S.f2(max(d7)) if d7 else '–'}).",
    ]
    lines += ["", "## Per shot", ""]
    head = (
        ["Shot", "Var"]
        + [h for _, h in S.COLS]
        + [
            "Detail ÷ GT's",
            "Finest band ÷ GT's · corr",
            f"dB to {rname.split(' fp16')[0]} s42 (7B s43 vs 7B s42)",
        ]
    )
    rows = []
    for c in SHOTS:
        if c not in data:
            continue
        for v, short in S.VARS:
            e = data[c][short]
            db = (
                S.f2(data[c]["db"])
                + " ("
                + (S.f2(data[c]["db7"]["s43"]) if data[c]["db7"]["s43"] else "–")
                + ")"
            )
            rows.append(
                [c, "none" if v == "none" else "4:3"]
                + [S.cell(k, e["pairs"]) for k, _ in S.COLS]
                + [S.det_cell(e), S.band_cell(e), db if short == "none" else ""]
            )
    lines += S.table(head, rows)
    return lines


def band_md4(clip_bys):
    lines = [
        "# The 7B fp16's band at 4K: its 2-seed spread per shot (seeds 42, 43)",
        "",
        f"Generated {S.stamp()} by ms_sum4k.py. Seeds 42 and 43: colour's eval-b1 (score), vmaf-b1 (split VMAF), "
        "vmaf-b2 (s42 none VMAF), bands/<shot>-d1.json (s42 bands); ours, the same commands: the s43 none VMAF, "
        "the s43 bands, the s43 − s42 distance. Cells: |s42 − s43| of the per-seed means.",
        "",
    ]
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
    for c in SHOTS:
        cd = f"{c}-d1"
        by = clip_bys[c]
        for v, short in S.VARS:
            r7 = by.get(f"{v}@f32", {})
            cells = []
            for k in ks:
                vals = [
                    float(E.series(r7[s], k).mean())
                    for s in ("s42", "s43")
                    if s in r7 and E.series(r7[s], k) is not None
                ]
                cells.append(
                    (
                        "–"
                        if len(vals) < 2
                        else S.minus(E.diff_fmt(k).format(max(vals) - min(vals)).lstrip("+"))
                    )
                    + ("" if len(vals) == 2 else f" ({len(vals)})")
                )
            b7 = [S.band1(j, "c" if v == "none" else "s")[0] for j in bands_7b4(cd).values() if j]
            b7 = [x for x in b7 if x is not None]
            fr7 = S.fr_runs("7b", cd).get(f"{v}@f32", {})
            fd = [
                float(S.fr_series(fr7[s], "dists").mean())
                for s in ("s42", "s43")
                if s in fr7 and S.fr_series(fr7[s], "dists") is not None
            ]
            rows.append(
                [c, short, ",".join(s[1:] for s in ("s42", "s43") if s in r7)]
                + cells
                + [
                    (f"{min(b7):.2f}–{max(b7):.2f}" if len(b7) > 1 else S.f2(b7[0]) if b7 else "–")
                    + f" ({len(b7)})",
                    (f"{max(fd) - min(fd):.4f}" if len(fd) > 1 else "–") + f" ({len(fd)})",
                    S.f2(S.db_of(f"{G}/diff/7b/{cd}.s43.txt")) if short == "none" else "",
                ]
            )
    lines += S.table(
        ["Shot", "Var", "Seeds"]
        + list(ks)
        + ["finest band ÷ GT's: range", "DISTS 45f", "dB s43 vs s42"],
        rows,
    )
    return lines


def cmp_rows(pairs):
    rows, bad = [], 0
    for cd, short, what, a, b in pairs:
        ja, jb = S.load_json(a), S.load_json(b)
        if not ja or not jb:
            rows.append([cd, short, what, "–", "ours missing" if not ja else "reference missing"])
            continue
        mx, n, only = S.series_diff(ja, jb)
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
                else f"max |diff| {mx:.3g}" + (f", only one: {','.join(only)}" if only else ""),
            ]
        )
    return rows, bad


def bands_cmp(cd, a, b, what):
    ja, jb = S.load_json(a), S.load_json(b)
    if not ja or not jb:
        return [cd, "-", what, "–", "ours missing" if not ja else "reference missing"], 0
    mx = 0.0
    for k in range(1, 6):
        for sig in ("s", "c", "r", "b", "g"):
            try:
                mx = max(
                    mx,
                    abs(
                        ja["bands"]["whole"][str(k)]["rms"][sig]
                        - jb["bands"]["whole"][str(k)]["rms"][sig]
                    ),
                )
            except (KeyError, TypeError):
                pass
    return [cd, "-", what, "25", "identical" if mx == 0 else f"max |diff| {mx:.3g}"], int(mx != 0.0)


def validation_md4(labels):
    lines = [
        "# Validation of the 4K pipeline (slice B) against colour's 4K JSONs",
        "",
        f"Generated {S.stamp()}.",
        "",
        "val4k = colour's sharp 7B sh42 decode scored as a file (ref tag val4k) against colour's own scores of it "
        "(eval-b2 / vmaf-b2 ~sharp, bands-b2 -sharp); val4k43 = colour's 7B fp16 s43 decode scored as a file "
        "against colour's eval-b1 s43 (a 7-variant call: the 2-variant call must give the same series) and "
        "vmaf-b1 s43 split; then our 7B renders (r7m: s42 none + split, r7v: s43) against colour's VMAF JSONs.",
        "",
    ]
    pairs, brow, bad = [], [], 0
    for L, (sref, eref, vref) in (
        ("val4k", ("s42", "sharp", "sharp")),
        ("val4k43", ("s43", "f32", "f32")),
    ):
        if L not in labels:
            continue
        for c in SHOTS:
            cd = f"{c}-d1"
            if not os.path.isdir(f"{G}/eval/{L}/{cd}"):
                continue
            ev = "eval-b2" if L == "val4k" else "eval-b1"
            for v, short in S.VARS:
                sv = E.safe(v)
                pairs.append(
                    (
                        cd,
                        short,
                        f"{L} score",
                        f"{G}/eval/{L}/{cd}/{cd}.s42.{sv}~{L}.json",
                        f"{O}/{ev}/{cd}/{cd}.{sref}.{sv}~{eref}.json",
                    )
                )
                if L == "val4k":
                    pairs.append(
                        (
                            cd,
                            short,
                            f"{L} vmaf",
                            f"{G}/vmaf/{L}/{cd}/{cd}.s42.{sv}~{L}~.json",
                            f"{O}/vmaf-b2/{cd}/{cd}.s42.{sv}~sharp~.json",
                        )
                    )
                elif short == "split":
                    pairs.append(
                        (
                            cd,
                            short,
                            f"{L} vmaf",
                            f"{G}/vmaf/{L}/{cd}/{cd}.s42.{sv}~{L}~.json",
                            f"{O}/vmaf-b1/{cd}/{cd}.s43.{sv}~f32~.json",
                        )
                    )
                else:
                    pairs.append(
                        (
                            cd,
                            short,
                            f"{L} vmaf vs our r7v s43",
                            f"{G}/vmaf/{L}/{cd}/{cd}.s42.{sv}~{L}~.json",
                            f"{G}/vmaf/7b/{cd}/{cd}.s43.{sv}~f32~.json",
                        )
                    )
            if L == "val4k":
                r, b = bands_cmp(
                    cd, f"{G}/bands/{L}/{cd}.json", f"{O}/bands-b2/{cd}-sharp.json", "val4k bands"
                )
            else:
                r, b = bands_cmp(
                    cd,
                    f"{G}/bands/{L}/{cd}.json",
                    f"{G}/bands/7b/{cd}-s43.json",
                    "val4k43 bands vs our r7b s43",
                )
            brow.append(r)
            bad += b
    for c in SHOTS:
        cd = f"{c}-d1"
        pairs.append(
            (
                cd,
                "none",
                "7B s42 vmaf (our r7m render)",
                f"{G}/vmaf/7b-check/{cd}/{cd}.s42.none~f32~.json",
                f"{O}/vmaf-b2/{cd}/{cd}.s42.none~f32~.json",
            )
        )
        pairs.append(
            (
                cd,
                "split",
                "7B s42 vmaf (our r7m render)",
                f"{G}/vmaf/7b-check/{cd}/{cd}.s42.split_ycc_4_3~f32~.json",
                f"{O}/vmaf-b1/{cd}/{cd}.s42.split_ycc_4_3~f32~.json",
            )
        )
        pairs.append(
            (
                cd,
                "split",
                "7B s43 vmaf (our r7v render)",
                f"{G}/vmaf/7b/{cd}/{cd}.s43.split_ycc_4_3~f32~.json",
                f"{O}/vmaf-b1/{cd}/{cd}.s43.split_ycc_4_3~f32~.json",
            )
        )
    rows, b = cmp_rows(pairs)
    bad += b
    have = sum(r[4] == "identical" for r in rows + brow)
    lines += [
        f"**Identical: {have}; not identical: {bad}; missing: {sum(r[3] == '–' for r in rows + brow)}.**",
        "",
    ]
    lines += S.table(["Shot", "Var", "JSON", "Series", "Result"], rows + brow)
    S.X["validation4k"] = (have, bad)
    return lines


def labels_present4():
    out = []
    for d in sorted(os.listdir(f"{G}/eval")) if os.path.isdir(f"{G}/eval") else []:
        # S9: the sharp's files (sh-*) and the sharp fp16's own seeds (sharp) are ms_sum4ksh.py's (its 5 shots, its band)
        if (
            d != "7b"
            and d != "sharp"
            and not d.startswith("sh-")
            and any(os.path.isdir(f"{G}/eval/{d}/{c}-d1") for c in SHOTS)
        ):
            out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=S.env("VAL_STATE") + "/sum")
    a = ap.parse_args()
    t0 = time.time()
    S.X["out"] = a.out
    os.makedirs(a.out, exist_ok=True)
    gl = S.GtLap(os.path.join(a.out, "gt_lap4k.json"))
    labels = labels_present4()
    clip_bys = {
        c: S.merged([r for s in clip_sources4(f"{c}-d1", labels) for r in S.load_dir(s)])
        for c in SHOTS
    }
    note = band_note(clip_bys)
    overview = {}
    for L in labels:
        try:  # one label's trouble never stops the others' files
            data = S.label_data(L, gl)
            F.apply(S, data)  # S16: the 4K floor
            S.write(os.path.join(a.out, f"4k-{L}.md"), label_md4(L, data, note))
            overview[L] = data
        except Exception as ex:  # noqa: BLE001
            import traceback

            S.log(traceback.format_exc())
            S.write(
                os.path.join(a.out, f"4k-{L}.md"),
                [
                    f"# {L} (4K)",
                    "",
                    f"**ms_sum4k.py failed on this label at "
                    f"{S.stamp()}:** {type(ex).__name__}: {ex} (see {S.SC}/logs/ms_sum4k.log)",
                ],
            )
    S.write(os.path.join(a.out, "4k-7b-band.md"), band_md4(clip_bys))
    vpath = os.path.join(a.out, "4k-validation.md")
    if any(L in VAL for L in labels):  # the record stays once the test labels' outputs are deleted
        S.write(vpath, validation_md4(labels))
    lines = [
        "# Slice B (4K): every file against the 7B fp16 (4K ×2 from d1, seed 42, 6 batch-A shots): overview",
        "",
        f"Generated {S.stamp()} by ms_sum4k.py ({S.GLUE}/). {note}. Per label: 4k-<label>.md; "
        "the 7B's band: 4k-7b-band.md; the pipeline's validation: 4k-validation.md. Kinds: Sol Levante (painted "
        "anime: painted, line, action, dark), digital (sunrise, space).",
        "",
        RULE4,
        "",
        "## Guards failed (kinds of source where a guard is worse on at least one shot), strict and calibrated "
        f"rules (calibrated: beyond {S.KCAL[2]} × the 2-seed spread or its floor; floors: {F.FLOOR_LIST})",
        "",
    ]
    rows = []
    for L, data in overview.items():
        cells = []
        for mode in ("tag", "tagc"):
            for short in ("none", "split"):
                kc = S.kind_cells(data, short, mode)
                fails = [
                    f"{title.split(':')[0]}: "
                    + ", ".join(kd for kd in KINDS if kc[g][kd].startswith("FAIL"))
                    for g, title in S.GUARDS
                    if any(kc[g][kd].startswith("FAIL") for kd in KINDS)
                ]
                cells.append("; ".join(fails) if fails else "none")
        wm = S.worst_mult(data)
        cells.append(S.mult_txt(wm[0]) + f" ({wm[1]})" if wm[0] is not None else "–")
        sp = {
            k: [
                data[c]["split"]["pairs"][k]["mean"] for c in data if k in data[c]["split"]["pairs"]
            ]
            for k in ("psnr_y", "vmaf", "lpips")
        }
        dbs = [data[c]["db"] for c in data if data[c]["db"] is not None]
        rows.append(
            [L + (" (test)" if L in VAL else ""), f"{len(data)}/{len(SHOTS)}"]
            + cells
            + [
                S.fmt(k, float(np.mean(sp[k]))) if sp[k] else "–"
                for k in ("psnr_y", "vmaf", "lpips")
            ]
            + [f"{S.f2(np.mean(dbs))} ({S.f2(min(dbs))})" if dbs else "–"]
        )
    lines += S.table(
        [
            "Label",
            "Shots",
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
    d7 = [S.db_of(f"{G}/diff/7b/{c}-d1.s43.txt") for c in SHOTS]
    d7 = [x for x in d7 if x is not None]
    lines += [
        "",
        f"For scale: the 7B fp16's seed 43 against its seed 42 at 4K: {S.f2(np.mean(d7)) if d7 else '–'} dB mean "
        f"({S.f2(min(d7)) if d7 else '–'}–{S.f2(max(d7)) if d7 else '–'}, {len(d7)} shots).",
    ]
    if "validation4k" in S.X:
        have, bad = S.X["validation4k"]
        lines += [
            "",
            f"Validation (4k-validation.md): {have} JSONs identical to colour's, {bad} not.",
        ]
    elif os.path.exists(vpath):
        with open(vpath, encoding="utf-8") as f:
            got = [x.strip("*") for x in f.read().splitlines() if x.startswith("**Identical:")]
        lines += [
            "",
            f"Validation (4k-validation.md, the test labels' outputs since deleted): {got[0] if got else '?'}",
        ]
    if S.X["skipped"]:
        lines += ["", f"JSONs being written, left out this run: {len(S.X['skipped'])}"]
    S.write(os.path.join(a.out, "4k-00-overview.md"), lines)
    gl.save()
    print(
        f"ms_sum4k: {len(labels)} labels, cache {S.X['cache_hits']} hits / {S.X['cache_miss']} computed, "
        f"{len(S.X['skipped'])} skipped, {time.time() - t0:.1f} s"
    )


if __name__ == "__main__":
    main()

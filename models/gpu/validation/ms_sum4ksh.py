#!/usr/bin/env python3
"""Model conversation, slice B of the sharp 7B's files (S9, 2026-10-07): each sh-* file's 4K runs (x2 from d1, seed
42) on the 5 shots that have two sharp 7B fp16 seeds (colour's dumps/<shot>-d1/sh42 and sh43), against the sharp 7B
fp16 at seed 42, per shot and metric, then PASS / FAIL per guard and kind of source; one Markdown file per label
(4k-<label>.md, e.g. 4k-sh-int8.md), an overview (4k-sh-00-overview.md), the sharp fp16's own 2-seed band at 4K
(4k-sharp-band.md) and, with the test label sh-val43, the validation against colour's JSONs (4k-sh-validation.md).
ms_sum.py's code, imported (loading, merging, pairing, verdicts, guards, cells: the same rules), on the sharp's
shots, kinds, sources and band.

  ms_sum4ksh.py [--out $VAL_STATE/sum]

Shots and kinds: digital (digital-cockpit, digital-space), first film (ouatia-face), Sol Levante
(sollevante-painted), cel (cel4k-detail).
Sources: the sharp 7B fp16 at seeds 42 and 43 = colour's own 4K scores (eval-b2/<cd>: score, sh42 and sh43, ref tag
sharp; vmaf-b2/<cd>: VMAF + CAMBI, s42 none and split, s43 split; bands-b2/<cd>-sharp.json, <cd>-sharp-s43.json),
ours for the rest, the same commands (gpu/vmaf/sharp/<cd>: s43 none and split, rshv-; gpu/diff/sharp/<cd>.s43.txt:
s43 against s42; gpu/fr/sharp/<cd>: fr_metrics.py of both seeds, rshf-); a label's = gpu/eval/<label>/<cd>,
gpu/vmaf/<label>/<cd>, gpu/bands/<label>/<cd>.json, gpu/diff/<label>/<cd>.s42.txt (to the sharp's s42 none
master, rshm-), gpu/fr/<label>/<cd> (only while score/FR4_ON exists).
Verdicts (ms_sum.py's): the label's frames minus the sharp's at seed 42; B / W when colour_eval's 95% moving-block
bootstrap interval excludes 0 AND |mean| exceeds the band = the sharp 7B fp16's spread over its 2 seeds (42, 43; one
seed scored: band 0, said per cell by the seeds behind it); Laplacian up / down the same; band energy (no interval):
|file - sharp at s42| beyond the sharp's 2-seed spread, either way. A guard FAILs on a kind of source when it is
worse on at least one of its shots. Two rules, both reported: strict (the above) and calibrated = a strict failure
whose |difference| also exceeds K x the spread, K = 10.9 with 2 seeds (S4's Monte Carlo).
S16 (2026-10-08, design's decision of 2026-10-07 23:55): the 4K floor (ms_floor.py): the spread used = max(the 2-seed
spread, one unit of the resolution the metric is printed at), in both rules and every printed multiple (a multiple of
the floor carries a †); applied to label_data()'s entries before the pages are made.
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ms_sum as S  # noqa: E402
import ms_floor as F  # noqa: E402  S16: the 4K floor

F.install(S)

E, O, G = S.E, S.O, S.G
C4 = S.env("MEAS_SHOTS")
SHOTS = ("digital-cockpit", "ouatia-face", "digital-space", "sollevante-painted", "cel4k-detail")
KIND = {
    "digital-cockpit": "digital",
    "digital-space": "digital",
    "ouatia-face": "first film",
    "sollevante-painted": "Sol Levante",
    "cel4k-detail": "cel",
}
KINDS = ("digital", "first film", "Sol Levante", "cel")
KTITLE = {
    "digital": "digital: cockpit, space",
    "first film": "first film: face",
    "Sol Levante": "Sol Levante: painted",
    "cel": "cel: detail",
}
SEEDS = ("s42", "s43")
NF = 45  # frames of every one of the 5 shots
ORDER = (
    "sh-int8",
    "sh-dyn",
    "sh-q4k",
    "sh-fp8a8",
    "sh-q80",
    "sh-fp8a16",
    "sh-nv4a4",
    "sh-q4ki",
    "sh-nv4a16",
)
VAL = {"sh-val43": "TEST: colour's sharp 7B fp16 sh43 decode scored as a file (validation)"}

# ms_sum.py's functions read these module names at call time: the sharp's B shots, kinds, GT dir and seeds
S.CLIPS_A, S.KIND, S.KINDS, S.CLIPS, S.SEEDS = SHOTS, KIND, KINDS, C4, SEEDS
_meta_of = S.meta_of
S.meta_of = lambda L: VAL.get(L) or _meta_of(L)


def sources(cd, labels):
    srcs = [f"{O}/eval-b2/{cd}", f"{O}/vmaf-b2/{cd}", f"{G}/vmaf/sharp/{cd}"]
    for L in labels:
        srcs += [f"{G}/eval/{L}/{cd}", f"{G}/vmaf/{L}/{cd}"]
    return srcs


S.clip_sources = sources


def bands_sharp(cd):
    """The sharp 7B fp16's own 4K scans: colour's (bands-b2), seeds 42 and 43."""
    return {
        "s42": S.load_json(f"{O}/bands-b2/{cd}-sharp.json"),
        "s43": S.load_json(f"{O}/bands-b2/{cd}-sharp-s43.json"),
    }


S.bands_sharp = bands_sharp


def label_data(L, gl):
    """Per shot: {variant: {pairs, lap, lap7, band, ...}, db, dbsh} for label L, against the sharp at seed 42, band =
    the sharp's own 2-seed spread (ms_sum.label_data's entries, its keys)."""
    data = {}
    for c in SHOTS:
        cd = f"{c}-d1"
        by = S.merged([r for s in sources(cd, [L]) for r in S.load_dir(s)])
        if not any(k.endswith(f"@{L}") for k in by):
            continue
        frL, frB = S.fr_runs(L, cd), S.fr_runs("sharp", cd)
        sb = bands_sharp(cd)
        bL = S.load_json(f"{G}/bands/{L}/{cd}.json")
        gt_lap = None
        try:
            gt_lap = gl.get(f"{C4}/{c}.gt.mkv", NF)
        except Exception as ex:  # noqa: BLE001
            S.log(f"GT Laplacian {cd}: {ex}")
        d = {
            "db": S.db_of(f"{G}/diff/{L}/{cd}.s42.txt"),
            "dbsh": S.db_of(f"{G}/diff/sharp/{cd}.s43.txt"),
        }
        for v, short in S.VARS:
            P = S.pair(by, f"{v}@{L}", f"{v}@sharp", f"{v}@sharp")
            F = S.fr_pair(frL, frB, frB, f"{v}@{L}", f"{v}@sharp", f"{v}@sharp")
            if "dists" in F:
                P["dists_all"] = F["dists"]
            for k in ("psnr_y_bottom16", "psnr_y_rest"):
                if k in F:
                    P[k] = F[k]
            r7 = by.get(f"{v}@sharp", {})
            lapL = S.lap_of(by.get(f"{v}@{L}", {}), ("s42",))
            lap7 = {s: E.run_mean(r7[s], "lap") for s in SEEDS if s in r7}
            sig = "c" if v == "none" else "s"
            bl = S.band1(bL, sig) if bL else (None, None)
            b7r = {s: S.band1(j, sig)[0] for s, j in sb.items() if j}
            b7r = {s: x for s, x in b7r.items() if x is not None}
            b7c = {s: S.band1(j, sig)[1] for s, j in sb.items() if j}
            bref = b7r.get("s42")
            btag = btagc = ""
            bmult = None
            if bl[0] is not None and len(b7r) > 1 and bref is not None:
                spread = max(b7r.values()) - min(b7r.values())
                btag = "↑" if bl[0] - bref > spread else "↓" if bref - bl[0] > spread else ""
                bmult = S.mult_of(bl[0] - bref, spread)
                kc = S.KCAL.get(len(b7r))
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
                "rname": "sharp",
                "bandsrc": {"scores": "sharp", "fr": "sharp", "bands": "sharp"},
            }
        data[c] = d
    return data


RULE = (
    "Cells: the file's output minus the sharp 7B fp16's at seed 42, frame by frame (colour_eval.py's scores with "
    "colour's baton-2 4K commands: whole frames, the first film --bars 42:42, --lpips, --dists-every 9, 8 threads; "
    "VMAF v1's 2160 model, CAMBI); **B / W** = better / worse when colour_eval's 95% moving-block bootstrap interval "
    "(blocks of 8, 2000 draws) excludes 0 AND |difference| exceeds the sharp 7B fp16's own 2-seed spread (|s42 − "
    "s43| of its per-seed means: colour's sh42 and sh43); Laplacian ↑ / ↓ the same; no tag = within. DISTS 5f = "
    "colour_eval's every 9th frame (5 frames: its interval is degenerate, the spread alone decides); DISTS 45f = "
    "fr_metrics.py's every frame, where present. Detail = the luma Laplacian variance ÷ the GT's; band = "
    "colour_bands.py's finest band (below 0.7 px), rms ÷ the GT's · its correlation with the GT (none: the raw "
    "decode c; split: s), ↑ / ↓ when |the file's − the sharp's at seed 42| exceeds the sharp's 2-seed spread (no "
    "interval: the spread alone decides, as DISTS 5f); in brackets after Detail and band: the sharp's seed-42 "
    "value; its 2-seed range; dB = RGB PSNR between the 16-bit none masters (ffv1_out.py --diff) of the file and "
    "the sharp 7B fp16 at seed 42. Each W, ↑ or ↓ shows its multiple of the spread (|difference| ÷ the sharp's "
    "2-seed spread). Two rules: **strict** (the above) and **calibrated** = strict AND |difference| > K × the "
    "spread, K = 10.9 with 2 seeds: the multiple a further fp16 seed would exceed with 5% probability per "
    "shot-variant (S4's Monte Carlo, Gaussian seed scatter, both directions), where the strict rule fails such a "
    "seed 50% of the time; **bold** cell = worse by the calibrated rule too. " + F.FLOOR_TXT
)


def band_note(clip_bys):
    """How many shot-variants have the sharp's 2 seeds, per source."""
    sc = fr = bd = 0
    for c in SHOTS:
        cd = f"{c}-d1"
        frB = S.fr_runs("sharp", cd)
        sb = bands_sharp(cd)
        for v, short in S.VARS:
            r = clip_bys[c].get(f"{v}@sharp", {})
            sc += all(
                s in r
                and E.series(r[s], "vmaf") is not None
                and E.series(r[s], "psnr_y") is not None
                for s in SEEDS
            )
            fr += all(s in frB.get(f"{v}@sharp", {}) for s in SEEDS)
            bd += all(
                sb.get(s) and S.band1(sb[s], "c" if v == "none" else "s")[0] is not None
                for s in SEEDS
            )
    n = 2 * len(SHOTS)
    full = sc == fr == bd == n
    return (
        f"The sharp 7B fp16's own band from 2 seeds (42, 43) on {sc}/{n} shot-variants for the scores and VMAF, "
        f"{fr}/{n} for DISTS 45f, {bd}/{n} for band energy"
        + ("" if full else " (one seed elsewhere: band 0, verdicts more eager)")
    )


def mean_cell(k, es):
    if not es:
        return "–"
    b = sum(e["tag"] == "B" for e in es)
    w = sum(e["tag"] == "W" for e in es)
    return f"{S.fmt(k, float(np.mean([e['mean'] for e in es])))} ({b}/{w}/{len(es) - b - w})"


def label_md(L, data, note):
    n = len(data)
    miss = [c for c in SHOTS if c not in data]
    lines = [
        f"# {L} − sharp 7B fp16: slice B (4K ×2 from d1, seed 42; the sharp's 5 shots with two sharp seeds)",
        "",
        f"Generated {S.stamp()} by ms_sum4ksh.py. File: `{S.meta_of(L)}`. Shots scored: {n}/{len(SHOTS)}"
        + (f" (missing: {', '.join(miss)})" if miss else "")
        + f". {note}.",
        "",
        RULE,
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
    dsh = [data[c]["dbsh"] for c in data if data[c]["dbsh"] is not None]
    lines += [
        "",
        f"Distance to the sharp 7B fp16 s42 (none masters): {S.f2(np.mean(dbs)) if dbs else '–'} dB mean, "
        f"{S.f2(min(dbs)) if dbs else '–'} worst; the sharp's own seed 43 against its 42: "
        f"{S.f2(np.mean(dsh)) if dsh else '–'} dB mean ({S.f2(min(dsh)) if dsh else '–'}–"
        f"{S.f2(max(dsh)) if dsh else '–'}).",
    ]
    lines += ["", "## Per shot", ""]
    head = (
        ["Shot", "Var"]
        + [h for _, h in S.COLS]
        + ["Detail ÷ GT's", "Finest band ÷ GT's · corr", "dB to sharp s42 (sharp s43 vs sharp s42)"]
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
                + (S.f2(data[c]["dbsh"]) if data[c]["dbsh"] else "–")
                + ")"
            )
            rows.append(
                [c, "none" if v == "none" else "4:3"]
                + [S.cell(k, e["pairs"]) for k, _ in S.COLS]
                + [S.det_cell(e), S.band_cell(e), db if short == "none" else ""]
            )
    lines += S.table(head, rows)
    return lines


def spread(r, k):
    vals = [
        float(E.series(r[s], k).mean()) for s in SEEDS if s in r and E.series(r[s], k) is not None
    ]
    return (max(vals) - min(vals)) if len(vals) == 2 else None, len(vals)


def band_md(clip_bys):
    """The sharp fp16's own 2-seed spread per shot; on the 2 shots the 7B has 2 seeds at 4K too (digital-space,
    sollevante-painted: colour's eval-b1 / vmaf-b1 / vmaf-b2, ours gpu/vmaf/7b), the sharp's spread over the 7B's."""
    lines = [
        "# The sharp 7B fp16's own band at 4K: its 2-seed spread per shot (seeds 42, 43)",
        "",
        f"Generated {S.stamp()} by ms_sum4ksh.py. Seeds 42 and 43: colour's eval-b2 (score ~sharp), vmaf-b2 (s42 "
        "none + split, s43 split VMAF + CAMBI), bands-b2 (<shot>-d1-sharp.json, -sharp-s43.json); ours, the same "
        "commands: the s43 none VMAF + CAMBI (and its split again), the s43 − s42 distance, fr_metrics of both "
        "seeds (DISTS 45f). Cells: |s42 − s43| of the per-seed means; in brackets the number of seeds when not 2. "
        "Last columns: the sharp's spread ÷ the 7B fp16's 2-seed spread, on the 2 shots both have at 4K.",
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
    ratios = defaultdict(list)
    for c in SHOTS:
        cd = f"{c}-d1"
        by = clip_bys[c]
        by7 = None
        if c in ("digital-space", "sollevante-painted"):
            by7 = S.merged(
                [
                    r
                    for s in (
                        f"{O}/eval-b1/{cd}",
                        f"{O}/vmaf-b1/{cd}",
                        f"{O}/vmaf-b2/{cd}",
                        f"{G}/vmaf/7b/{cd}",
                    )
                    for r in S.load_dir(s)
                ]
            )
        frB = S.fr_runs("sharp", cd)
        sb = bands_sharp(cd)
        for v, short in S.VARS:
            r = by.get(f"{v}@sharp", {})
            cells, rc = [], []
            for k in ks:
                sp, n = spread(r, k)
                cells.append(
                    ("–" if sp is None else S.minus(E.diff_fmt(k).format(sp).lstrip("+")))
                    + ("" if n == 2 else f" ({n})")
                )
                if by7 is not None and sp is not None:
                    sp7, _ = spread(by7.get(f"{v}@f32", {}), k)
                    if sp7:
                        ratios[k].append(sp / sp7)
                        if k in ("psnr_y", "lpips", "vmaf", "lap"):
                            rc.append(f"{k} {sp / sp7:.2f}×")
            b7 = [S.band1(j, "c" if v == "none" else "s")[0] for j in sb.values() if j]
            b7 = [x for x in b7 if x is not None]
            fr7 = frB.get(f"{v}@sharp", {})
            fd = [
                float(S.fr_series(fr7[s], "dists").mean())
                for s in SEEDS
                if s in fr7 and S.fr_series(fr7[s], "dists") is not None
            ]
            rows.append(
                [c, short, ",".join(s[1:] for s in SEEDS if s in r)]
                + cells
                + [
                    (f"{min(b7):.2f}–{max(b7):.2f}" if len(b7) > 1 else S.f2(b7[0]) if b7 else "–")
                    + f" ({len(b7)})",
                    (f"{max(fd) - min(fd):.4f}" if len(fd) > 1 else "–") + f" ({len(fd)})",
                    S.f2(S.db_of(f"{G}/diff/sharp/{cd}.s43.txt")) if short == "none" else "",
                    "; ".join(rc) if rc else "",
                ]
            )
    lines += S.table(
        ["Shot", "Var", "Seeds"]
        + list(ks)
        + ["finest band ÷ GT's: range", "DISTS 45f", "dB s43 vs s42", "sharp ÷ 7B spread"],
        rows,
    )
    if ratios:
        lines += [
            "",
            "The sharp's 2-seed spread ÷ the 7B fp16's, per metric: median (min–max) over the shot-variants "
            "both have (n): "
            + "; ".join(
                f"{k} {np.median(x):.2f}× ({min(x):.2f}–{max(x):.2f}, {len(x)})"
                for k, x in ratios.items()
                if x
            )
            + ".",
        ]
    return lines


def cmp_json(a, b):
    ja, jb = S.load_json(a), S.load_json(b)
    if not ja or not jb:
        return "–", "ours missing" if not ja else "reference missing", 0
    mx, n, only = S.series_diff(ja, jb)
    ok = mx == 0.0 and not only
    return (
        str(n),
        "identical"
        if ok
        else (f"max |diff| {mx:.3g}" + (f", only one: {','.join(only)}" if only else "")),
        int(not ok),
    )


def cmp_bands(a, b):
    ja, jb = S.load_json(a), S.load_json(b)
    if not ja or not jb:
        return "–", "ours missing" if not ja else "reference missing", 0
    mx, n = 0.0, 0
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
                n += 1
            except (KeyError, TypeError):
                pass
    return (
        str(n),
        "identical" if mx == 0 and n == 25 else f"max |diff| {mx:.3g} ({n} values)",
        int(mx != 0 or n != 25),
    )


def validation_md(labels):
    """sh-val43 = colour's sharp sh43 decode of sollevante-painted scored as a file, against colour's own JSONs of it;
    then our sharp references' VMAF (rshm s42 renders, rshv s43 renders) against colour's, on the 5 shots."""
    lines = [
        "# Validation of the sharp's 4K pipeline (slice B of the sh-* files) against colour's 4K JSONs",
        "",
        f"Generated {S.stamp()} by ms_sum4ksh.py.",
        "",
        "sh-val43 = colour's sharp 7B fp16 sh43 decode of sollevante-painted, scored by this pipeline as a file "
        "(the sh-* commands: ref tag sh-val43, content s42), against colour's own JSONs of that decode (eval-b2 "
        "s43 ~sharp: score; vmaf-b2 s43 split ~sharp~; bands-b2 sollevante-painted-d1-sharp-s43.json), its none "
        "VMAF against our rshv s43 render's, its distance to the sharp's s42 none master against rshv's; then, "
        "on the 5 shots, our sharp s42 renders (rshm: VMAF into gpu/vmaf/sharp-check) and s43 renders (rshv) "
        "against colour's vmaf-b2 JSONs.",
        "",
    ]
    rows, bad = [], 0
    if "sh-val43" in labels:
        L, cd = "sh-val43", "sollevante-painted-d1"
        for v, short in S.VARS:
            sv = E.safe(v)
            n, res, b = cmp_json(
                f"{G}/eval/{L}/{cd}/{cd}.s42.{sv}~{L}.json",
                f"{O}/eval-b2/{cd}/{cd}.s43.{sv}~sharp.json",
            )
            rows.append([cd, short, "sh-val43 score vs colour's eval-b2 s43", n, res])
            bad += b
            if short == "split":
                n, res, b = cmp_json(
                    f"{G}/vmaf/{L}/{cd}/{cd}.s42.{sv}~{L}~.json",
                    f"{O}/vmaf-b2/{cd}/{cd}.s43.{sv}~sharp~.json",
                )
                rows.append([cd, short, "sh-val43 VMAF vs colour's vmaf-b2 s43", n, res])
                bad += b
            n, res, b = cmp_json(
                f"{G}/vmaf/{L}/{cd}/{cd}.s42.{sv}~{L}~.json",
                f"{G}/vmaf/sharp/{cd}/{cd}.s43.{sv}~sharp~.json",
            )
            rows.append([cd, short, "sh-val43 VMAF vs our rshv s43", n, res])
            bad += b
        n, res, b = cmp_bands(f"{G}/bands/{L}/{cd}.json", f"{O}/bands-b2/{cd}-sharp-s43.json")
        rows.append(
            [cd, "-", "sh-val43 bands vs colour's bands-b2 -sharp-s43 (rms, bands 1-5)", n, res]
        )
        bad += b
        da, db = S.db_of(f"{G}/diff/{L}/{cd}.s42.txt"), S.db_of(f"{G}/diff/sharp/{cd}.s43.txt")
        ok = da is not None and da == db
        rows.append(
            [
                cd,
                "none",
                "sh-val43 dB to the sharp s42 vs rshv s43's",
                "1",
                f"{'identical' if ok else 'DIFFERENT'} ({S.f2(da)} / {S.f2(db)} dB)"
                if da is not None and db is not None
                else "missing",
            ]
        )
        bad += da is not None and db is not None and not ok
    for c in SHOTS:
        cd = f"{c}-d1"
        for v, short in S.VARS:
            sv = E.safe(v)
            n, res, b = cmp_json(
                f"{G}/vmaf/sharp-check/{cd}/{cd}.s42.{sv}~sharp~.json",
                f"{O}/vmaf-b2/{cd}/{cd}.s42.{sv}~sharp~.json",
            )
            rows.append([cd, short, "sharp s42 VMAF (our rshm render) vs colour's vmaf-b2", n, res])
            bad += b
        n, res, b = cmp_json(
            f"{G}/vmaf/sharp/{cd}/{cd}.s43.split_ycc_4_3~sharp~.json",
            f"{O}/vmaf-b2/{cd}/{cd}.s43.split_ycc_4_3~sharp~.json",
        )
        rows.append([cd, "split", "sharp s43 VMAF (our rshv render) vs colour's vmaf-b2", n, res])
        bad += b
    have = sum(r[4].startswith("identical") for r in rows)
    miss = sum(r[3] == "–" or r[4] == "missing" for r in rows)
    lines += [f"**Identical: {have}; not identical: {bad}; missing: {miss}.**", ""]
    lines += S.table(["Shot", "Var", "JSON", "Series", "Result"], rows)
    S.X["validation4ksh"] = (have, bad, miss, "sh-val43" in labels)
    return lines


def labels_present():
    out = []
    for d in sorted(os.listdir(f"{G}/eval")) if os.path.isdir(f"{G}/eval") else []:
        if d.startswith("sh-") and any(os.path.isdir(f"{G}/eval/{d}/{c}-d1") for c in SHOTS):
            out.append(d)
    return sorted(out, key=lambda L: (ORDER.index(L) if L in ORDER else len(ORDER), L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=S.env("VAL_STATE") + "/sum")
    a = ap.parse_args()
    t0 = time.time()
    S.X["out"] = a.out
    os.makedirs(a.out, exist_ok=True)
    gl = S.GtLap(os.path.join(a.out, "gt_lap4ksh.json"))
    labels = labels_present()
    clip_bys = {
        c: S.merged([r for s in sources(f"{c}-d1", labels) for r in S.load_dir(s)]) for c in SHOTS
    }
    note = band_note(clip_bys)
    overview = {}
    for L in labels:
        try:  # one label's trouble never stops the others' files
            data = label_data(L, gl)
            F.apply(S, data)  # S16: the 4K floor
            S.write(os.path.join(a.out, f"4k-{L}.md"), label_md(L, data, note))
            overview[L] = data
        except Exception as ex:  # noqa: BLE001
            import traceback

            S.log(traceback.format_exc())
            S.write(
                os.path.join(a.out, f"4k-{L}.md"),
                [
                    f"# {L} (4K)",
                    "",
                    f"**ms_sum4ksh.py failed on this label at "
                    f"{S.stamp()}:** {type(ex).__name__}: {ex} (see {S.SC}/logs/ms_sum4ksh.log)",
                ],
            )
    S.write(os.path.join(a.out, "4k-sharp-band.md"), band_md(clip_bys))
    vpath = os.path.join(a.out, "4k-sh-validation.md")
    if "sh-val43" in labels or not os.path.exists(
        vpath
    ):  # the record stays once the test label's outputs are deleted
        S.write(vpath, validation_md(labels))
    lines = [
        "# Slice B of the sharp 7B's files (4K ×2 from d1, seed 42, the sharp's 5 two-seed shots): every sh-* "
        "file against the sharp 7B fp16: overview",
        "",
        f"Generated {S.stamp()} by ms_sum4ksh.py ({S.GLUE}/). {note}. Per label: 4k-<label>.md; "
        "the sharp's band: 4k-sharp-band.md; the pipeline's validation: 4k-sh-validation.md. Kinds: digital "
        "(digital-cockpit, digital-space), first film (ouatia-face), Sol Levante (sollevante-painted), cel "
        "(cel4k-detail).",
        "",
        RULE,
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
            "dB to the sharp s42: mean (worst)",
        ],
        rows,
    )
    dsh = [S.db_of(f"{G}/diff/sharp/{c}-d1.s43.txt") for c in SHOTS]
    dsh = [x for x in dsh if x is not None]
    lines += [
        "",
        f"For scale: the sharp 7B fp16's seed 43 against its seed 42 at 4K: {S.f2(np.mean(dsh)) if dsh else '–'} "
        f"dB mean ({S.f2(min(dsh)) if dsh else '–'}–{S.f2(max(dsh)) if dsh else '–'}, {len(dsh)} shots).",
    ]
    if "validation4ksh" in S.X and S.X["validation4ksh"][3]:
        have, bad, miss, _ = S.X["validation4ksh"]
        lines += [
            "",
            f"Validation (4k-sh-validation.md): {have} identical to colour's (or to ours), {bad} not, {miss} missing.",
        ]
    elif os.path.exists(vpath):
        with open(vpath, encoding="utf-8") as f:
            got = [x.strip("*") for x in f.read().splitlines() if x.startswith("**Identical:")]
        lines += [
            "",
            f"Validation (4k-sh-validation.md{'' if 'sh-val43' in labels else ', the test label since deleted'}): "
            f"{got[0] if got else '?'}",
        ]
    if S.X["skipped"]:
        lines += ["", f"JSONs being written, left out this run: {len(S.X['skipped'])}"]
    S.write(os.path.join(a.out, "4k-sh-00-overview.md"), lines)
    gl.save()
    print(
        f"ms_sum4ksh: {len(labels)} labels, cache {S.X['cache_hits']} hits / {S.X['cache_miss']} computed, "
        f"{len(S.X['skipped'])} skipped, {time.time() - t0:.1f} s"
    )


if __name__ == "__main__":
    main()

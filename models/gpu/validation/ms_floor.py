#!/usr/bin/env python3
"""Model conversation, S16 (2026-10-08): the 4K floor (design's decision of 2026-10-07 23:55), for ms_sum4k.py (the
7B's slice B) and ms_sum4ksh.py (the sharp's slice B); ms_sum.py (1080p, 3-seed bands) is left as it was.

At 4K the band is two seeds; where the two agree to the resolution a metric is printed at, their spread is ~0 and any
difference becomes a huge multiple of it (the sharp's ouatia-face ΔE00 lf: a spread near 0.00001, fp8 W8A8's +0.001
87.5 times it). One floor per metric = one unit of the resolution at which the summaries print it:
  - colour_eval.diff_fmt: PSNR-Y, VMAF (and the Laplacian) "{:+.2f}"; SSIM-Y, LPIPS, DISTS "{:+.4f}"; every other
    metric "{:+.3f}" (CAMBI added, ΔE00 lf and the other ΔE00 scales, T-err, T-err lf, fringes);
  - ms_sum.fmt: DISTS 45f (fr_metrics.py's every frame) "{:+.4f}"; fr_metrics' PSNR-Y parts as PSNR-Y;
  - detail and band energy are printed as ratios to the GT's with 2 decimals (ms_sum.det_txt, band_txt): 0.01 of the
    ratio; for detail that is 0.01 x the GT's Laplacian variance in the units of the paired difference.
Effective spread = max(the seeds' spread, the floor), in both rules (strict: colour_eval's interval excludes 0 where it
gives one AND |difference| > the effective spread; calibrated: AND |difference| > K x the effective spread) and in
every printed multiple; a multiple of the floor (the seeds agreeing closer than the printed resolution) carries a †.
"""

import math

# one unit of the printed resolution, per metric key of ms_sum's pair entries
FLOOR = {
    "psnr_y": 0.01,
    "psnr_y_bottom16": 0.01,
    "psnr_y_rest": 0.01,
    "vmaf": 0.01,
    "ssim_y": 0.0001,
    "lpips": 0.0001,
    "dists": 0.0001,
    "dists_all": 0.0001,
    "cambi_added": 0.001,
    "de0": 0.001,
    "de1": 0.001,
    "de2": 0.001,
    "de4": 0.001,
    "de8": 0.001,
    "de16": 0.001,
    "tl2": 0.001,
    "tc2": 0.001,
    "fringe": 0.001,
    "t_full": 0.001,
    "t_lf": 0.001,
}
LAP_RATIO = 0.01  # detail: Laplacian variance ÷ the GT's, 2 decimals
BAND_RATIO = 0.01  # band energy: the finest band's rms ÷ the GT's, 2 decimals
FR_KEYS = (
    "dists_all",
    "psnr_y_bottom16",
    "psnr_y_rest",
)  # fr_metrics.py's: ms_sum.FR_HIGHER decides their direction
FLOOR_TXT = (
    "**Floor (S16, 2026-10-08, design's decision):** with two seeds the band can be ~0 where they agree to "
    "the printed resolution; the spread used is max(the seeds' spread, one unit of the resolution the metric "
    "is printed at): PSNR-Y, VMAF 0.01; SSIM-Y, LPIPS, DISTS 5f and 45f 0.0001; CAMBI+, ΔE00 lf, T-err, "
    "T-err lf 0.001; detail and the finest band 0.01 of the GT's: in both rules and in every multiple; "
    "**†** = a multiple of the floor (the two seeds closer than the printed resolution)."
)
FLOOR_LIST = (
    "PSNR-Y 0.01 dB, VMAF 0.01, SSIM-Y 0.0001, LPIPS 0.0001, DISTS 5f and 45f 0.0001, CAMBI+ 0.001, "
    "ΔE00 lf 0.001, T-err 0.001, T-err lf 0.001, detail 0.01 (÷ GT's), finest band 0.01 (÷ GT's)"
)


class Floored(float):
    """An effective spread set by the floor (the seeds' own spread below it)."""


class FMult(float):
    """A multiple of a floored spread."""


def install(S):
    """ms_sum's mult_of / mult_txt (module attributes, read at call time by cell, det_cell, band_cell, guard_fail,
    worst_mult): a multiple of a floored spread is an FMult, printed with a †. Idempotent."""
    if getattr(S, "_floor_installed", False):
        return
    mo, mt = S.mult_of, S.mult_txt

    def mult_of(mean, band):
        m = mo(mean, band)
        return FMult(m) if isinstance(band, Floored) and m is not None and math.isfinite(m) else m

    def mult_txt(m):
        return mt(m) + ("†" if isinstance(m, FMult) else "")

    S.mult_of, S.mult_txt, S._floor_installed = mult_of, mult_txt, True


def floor_of(k, gtlap):
    """The floor of metric key k in the units of its paired difference (None: no floor known)."""
    if k == "lap":
        return LAP_RATIO * gtlap if gtlap else None
    return FLOOR.get(k)


def apply(S, data):
    """In place, on ms_sum.label_data()-shaped data {shot: {"none" / "split": entry, ...}}: each pair entry's band ->
    max(the seeds' spread, the floor) (band_raw keeps the seeds'), its strict tag recomputed from the same interval
    (ms_sum.tag_of; the calibrated verdict follows, ms_sum.verdict reading the band); band energy's btag, btagc and
    bmult recomputed the same (bspread_raw keeps the seeds' spread). Returns the counts the floor raised."""
    E = S.E
    n = {"pairs": 0, "bands": 0}
    for d in data.values():
        if not isinstance(d, dict):
            continue
        for short in ("none", "split"):
            e = d.get(short)
            if not isinstance(e, dict) or "pairs" not in e:
                continue
            for k, p in e["pairs"].items():
                fl = floor_of(k, e.get("gtlap"))
                if fl is None or not isinstance(p, dict) or "band" not in p:
                    continue
                if "band_raw" not in p:
                    p["band_raw"] = float(p["band"])
                if p["band_raw"] < fl:
                    p["band"] = Floored(fl)
                    hi = S.FR_HIGHER if k in FR_KEYS else E.HIGHER
                    p["tag"] = S.tag_of(k, p["mean"], p["lo"], p["hi"], p["band"], hi)
                    n["pairs"] += 1
            bl = e["band"][0] if e.get("band") else None
            b7r = [x for x in (e.get("band7") or {}).values() if x is not None]
            bref = e.get("band7ref")
            if bl is not None and len(b7r) > 1 and bref is not None:
                raw = max(b7r) - min(b7r)
                e["bspread_raw"] = raw
                if raw < BAND_RATIO:
                    sp = Floored(BAND_RATIO)
                    e["btag"] = "↑" if bl - bref > sp else "↓" if bref - bl > sp else ""
                    e["bmult"] = S.mult_of(bl - bref, sp)
                    kc = S.KCAL.get(len(b7r))
                    e["btagc"] = (
                        e["btag"]
                        if e["btag"] and kc is not None and abs(bl - bref) > kc * sp
                        else ""
                    )
                    n["bands"] += 1
    return n

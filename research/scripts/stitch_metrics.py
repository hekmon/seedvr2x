#!/usr/bin/env python3
"""Batch-boundary stitching metrics for SeedVR2 PNG sequences, against a single-batch reference.

  stitch_metrics.py OUT_DIR --input VIDEO --skip N --ref REF_DIR --batch B [--overlap K]
                    [--curve numz|linear|cosine|prev|cur] [--frames N] [--label L] [--json F]
  stitch_metrics.py --summary A.json B.json ...       # Markdown tables of saved results
  stitch_metrics.py --profile A.json B.json ...       # PSNR vs reference by offset to the boundary

The reference is the same clip computed as one batch (no boundary at all): what a stitched
output should look like if stitching were perfect. Batch starts are placed as SeedVR2's encode
loop places them (step = batch - overlap, a last batch no longer than the overlap is dropped);
the overlap frames [s, s + K) of each batch start s > 0 are mixed with the weights of --curve
(blend_patch.py's curves; numz = the CLI's own). The "zone" of a boundary is the K + 1 frame
transitions s .. s + K (between frames t - 1 and t), where the frame source changes.

Per transition (from quality_metrics.py's definitions):
- hold dY: mean |dY| of the output on held input frames (input mean |dY| < --hold): flicker
- added lf: mean |d(out - in)| on 16x16-pixel block means, RGB: low-frequency brightness and
  colour change the input doesn't have, on every transition (held or not)
For each, the mean inside batches (transitions outside every zone), the mean of each boundary's
largest zone transition ("step", what the eye catches), the zone mean, and the same figures for
the reference at the same transitions (what the clip itself does there without any boundary).
"Excess" = output minus reference, transition by transition: the change the boundary adds. Per
boundary, its largest zone value ("excess step") and its sum over the zone ("excess total": the
whole jump, however it is spread), averaged over the boundaries; inside batches it is ~0.

Per frame: PSNR vs the reference and vs the input; the vs-reference PSNR is also averaged by
offset to the boundary (frames s - 4 .. s + K + 4), and per batch over its interior (frames
whose source is that batch alone, at least 4 frames from any zone): whether frames near a
boundary drift further from the single-batch result than the rest of their batch.

Sharpness: luma Laplacian variance of each frame over the reference's same frame, averaged over
the mixed frames (a mix of two renderings is softer) and over the frames away from any zone.

Cost: frames computed (each batch padded to 4n + 1, as the CLI does) / frames output.

--latent W:M analyses blend_patch.py's latent-space stitching (one batch for the whole clip,
DiT on windows of W latents, M shared): a window starting at latent s > 0 starts at frame
4s - 3, its first 4M frames are cross-faded, and the zone is widened by 2 transitions on each
side because the causal decoder spreads a latent change over its neighbours. The cost is then
the DiT's (latents computed / latents); the VAE encodes and decodes every frame once.

Needs numpy and opencv-python (both in the SeedVR2 venv); quality_metrics.py next to it.
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quality_metrics import block_mean, luma, png_frames, read_png, video_frames  # noqa: E402
from blend_patch import latent_windows, weights  # noqa: E402


def layout(n, batch, overlap):
    """Batch starts and lengths as encode_all_batches computes them (uniform_batch_size off)."""
    step = batch - overlap if overlap > 0 else batch
    if step <= 0:
        step, overlap = batch, 0
    starts, lens, s = [], [], 0
    while s < n:
        end = min(s + batch, n)
        if s > 0 and end - s <= overlap:
            break
        starts.append(s)
        lens.append(end - s)
        s += step
    computed = sum(((L - 1 + 3) // 4) * 4 + 1 if L % 4 != 1 else L for L in lens)
    return starts, lens, overlap, computed


def series(out_dir, ins, ref_dir=None):
    """Per-frame and per-transition series of one PNG sequence."""
    outs = png_frames(out_dir)[:len(ins)]
    refs = png_frames(ref_dir)[:len(ins)] if ref_dir else []
    n = min(len(outs), len(ins), len(refs) if ref_dir else len(ins))
    tdo, tdi, add_lf, psnr_ref, psnr_in, lap = [], [], [], [], [], []
    prev = None
    for i in range(n):
        x = ins[i].astype(np.float32)
        o = read_png(outs[i]).astype(np.float32)
        yo, yi = luma(o), luma(x)
        lap.append(float(cv2.Laplacian(yo, cv2.CV_32F).var()))
        d = o - x
        psnr_in.append(10 * math.log10(255 ** 2 / max(float((d * d).mean()), 1e-12)))
        if refs:
            r = read_png(refs[i]).astype(np.float32)
            dr = o - r
            psnr_ref.append(10 * math.log10(255 ** 2 / max(float((dr * dr).mean()), 1e-12)))
        lf = block_mean(d)
        if prev is not None:
            pyo, pyi, plf = prev
            tdo.append(float(np.abs(yo - pyo).mean()))
            tdi.append(float(np.abs(yi - pyi).mean()))
            add_lf.append(float(np.abs(lf - plf).mean()))
        prev = (yo, yi, lf)
    return {"frames": n, "tdiff_out": tdo, "tdiff_in": tdi, "added_lf": add_lf,
            "psnr_ref": psnr_ref, "psnr_in": psnr_in, "lap": lap}


def r3(v):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(float(v), 3)


def split(vals, zones, valid):
    """vals[t-1] for transition t; zones: list of transition lists; valid: bool mask or None."""
    vals = np.asarray(vals, dtype=float)
    ok = np.ones(len(vals), bool) if valid is None else np.asarray(valid)
    in_zone = np.zeros(len(vals), bool)
    for z in zones:
        for t in z:
            in_zone[t - 1] = True
    inner = vals[ok & ~in_zone]
    steps, zone_vals = [], []
    for z in zones:
        v = [vals[t - 1] for t in z if ok[t - 1]]
        if v:
            steps.append(max(v))
            zone_vals += v
    return {"inner": r3(inner.mean()) if inner.size else None,
            "step": r3(np.mean(steps)) if steps else None,
            "step_max": r3(max(steps)) if steps else None,
            "zone": r3(np.mean(zone_vals)) if zone_vals else None,
            "n_boundaries": len(steps), "n_zone": len(zone_vals)}


def analyse(a):
    ins = video_frames(a.input, a.skip, a.frames or len(png_frames(a.out)))
    s_out = series(a.out, ins, a.ref)
    s_ref = series(a.ref, ins)
    n = min(s_out["frames"], s_ref["frames"])
    if a.latent:
        W, M = (int(x) for x in a.latent.split(":"))
        T = (n - 1) // 4 + 1
        wins = latent_windows(T, W, M)
        starts = [0] + [4 * s - 3 for s, _ in wins[1:]]
        lens = [min(4 * (e - 1), n - 1) - (4 * s - 3 if s else 0) + 1 for s, e in wins]
        K, computed = 4 * M, n
        zones = [[t for t in range(s - 2, s + K + 3) if 1 <= t <= n - 1] for s in starts[1:]]
        dit_cost = sum(e - s for s, e in wins) / T
    else:
        starts, lens, overlap, computed = layout(n, a.batch, a.overlap)
        K = overlap
        zones = [[t for t in range(s, s + K + 1) if 1 <= t <= n - 1] for s in starts[1:]]
        dit_cost = None
    hold = np.asarray(s_out["tdiff_in"][:n - 1]) < a.hold
    nw = int(a.latent.split(":")[1]) if a.latent else K  # weights per latent, or per frame
    w = weights(a.curve, nw) if nw else []
    res = {"label": a.label, "out": a.out, "ref": a.ref, "frames": n, "batch": a.batch, "overlap": K,
           "curve": a.curve if K else "-", "weights": [round(x, 3) for x in w], "batch_starts": starts,
           "batch_lengths": lens, "zones": zones, "computed": computed,
           "cost": round(computed / n, 3), "hold_transitions": int(hold.sum()), "latent": a.latent,
           "dit_cost": round(dit_cost, 3) if dit_cost else None}
    for key, mask in (("hold_tdiff", hold), ("added_lf", None)):
        res[key] = split(s_out["tdiff_out" if key == "hold_tdiff" else "added_lf"][:n - 1], zones, mask)
        res[key + "_ref"] = split(s_ref["tdiff_out" if key == "hold_tdiff" else "added_lf"][:n - 1], zones, mask)
    for key, so, sr, mask in (("hold_excess", "tdiff_out", "tdiff_out", hold), ("lf_excess", "added_lf", "added_lf", None)):
        e = np.asarray(s_out[so][:n - 1]) - np.asarray(s_ref[sr][:n - 1])
        ok = np.ones(n - 1, bool) if mask is None else mask
        steps, tots = [], []
        for z in zones:
            v = [e[t - 1] for t in z if ok[t - 1]]
            if v:
                steps.append(max(v))
                tots.append(sum(v))
        in_zone = np.zeros(n - 1, bool)
        for z in zones:
            in_zone[[t - 1 for t in z]] = True
        res[key] = {"inner": r3(e[ok & ~in_zone].mean()) if (ok & ~in_zone).any() else None,
                    "step": r3(np.mean(steps)) if steps else None, "total": r3(np.mean(tots)) if tots else None}
    pr = np.asarray(s_out["psnr_ref"][:n])
    zone_frames = set()
    for s in starts[1:]:
        zone_frames.update(range(s - 4, s + K + 4))
    interiors = []
    for b, (s, L) in enumerate(zip(starts, lens)):
        lo = s + (K if b else 0)
        hi = starts[b + 1] if b + 1 < len(starts) else n
        fr = [f for f in range(lo, min(hi, n)) if f not in zone_frames]
        interiors.append(r3(np.mean(pr[fr])) if fr else None)
    res["psnr_ref_batch_interior"] = interiors
    # sharpness of the mixed frames vs the reference's same frames (a mix of two renderings is softer)
    ratio = np.asarray(s_out["lap"][:n]) / np.maximum(np.asarray(s_ref["lap"][:n]), 1e-6)
    mixed = sorted({f for s in starts[1:] for f in range(s, s + K) if f < n})
    inner_f = [f for f in range(n) if f not in zone_frames]
    res["lap_ratio_mixed"] = r3(ratio[mixed].mean()) if mixed else None
    res["lap_ratio_inner"] = r3(ratio[inner_f].mean()) if inner_f else None
    res["psnr_ref_mean"] = r3(10 * math.log10(255 ** 2 / np.mean(255 ** 2 / 10 ** (pr / 10))))
    res["psnr_ref_min"] = r3(pr.min())
    res["psnr_in_mean"] = r3(np.mean(s_out["psnr_in"][:n]))
    prof = {}
    for s in starts[1:]:
        for off in range(-4, K + 5):
            f = s + off
            if 0 <= f < n:
                prof.setdefault(off, []).append(pr[f])
    res["psnr_ref_by_offset"] = {str(k): r3(np.mean(v)) for k, v in sorted(prof.items())}
    res["per_transition"] = {"hold": hold.astype(int).tolist(),
                             "tdiff_out": [r3(v) for v in s_out["tdiff_out"][:n - 1]],
                             "tdiff_ref": [r3(v) for v in s_ref["tdiff_out"][:n - 1]],
                             "tdiff_in": [r3(v) for v in s_out["tdiff_in"][:n - 1]],
                             "added_lf": [r3(v) for v in s_out["added_lf"][:n - 1]],
                             "added_lf_ref": [r3(v) for v in s_ref["added_lf"][:n - 1]]}
    res["per_frame"] = {"psnr_ref": [r3(v) for v in pr], "psnr_in": [r3(v) for v in s_out["psnr_in"][:n]]}
    return res


def _trip(d):
    if not d or d.get("step") is None:
        return f"{d['inner']} / – / –" if d else "–"
    return f"{d['inner']} / {d['step']} / {d['zone']}"


def _ex(d):
    return "–" if not d or d.get("step") is None else f"{d['step']} / {d['total']}"


def summary(paths):
    rows = [json.load(open(p, encoding="utf-8")) for p in paths]
    print("| Run | Overlap, curve | Cost | Hold dY inside / step / zone | ref step | Hold excess step / total | "
          "Added lf inside / step / zone | ref step | Lf excess step / total | PSNR vs ref (min) | PSNR in | "
          "Lap var vs ref, mixed / inner |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        ov = f"latent {r['latent']} {r['curve']}" if r.get("latent") else f"{r['overlap']} {r['curve']}"
        cost = f"{r['cost']} (DiT {r['dit_cost']})" if r.get("dit_cost") else r["cost"]
        print(f"| {r.get('label') or r['out']} | {ov} | {cost} | "
              f"{_trip(r['hold_tdiff'])} | {r['hold_tdiff_ref'].get('step')} | {_ex(r.get('hold_excess'))} | "
              f"{_trip(r['added_lf'])} | {r['added_lf_ref'].get('step')} | {_ex(r.get('lf_excess'))} | "
              f"{r['psnr_ref_mean']} ({r['psnr_ref_min']}) | {r['psnr_in_mean']} | "
              f"{r.get('lap_ratio_mixed') or '–'} / {r.get('lap_ratio_inner')} |")


def profile(paths):
    rows = [json.load(open(p, encoding="utf-8")) for p in paths]
    offs = sorted({int(k) for r in rows for k in r["psnr_ref_by_offset"]})
    print("| Run | " + " | ".join(f"{o:+d}" for o in offs) + " | batch interiors |")
    print("|---" * (len(offs) + 2) + "|")
    for r in rows:
        p = r["psnr_ref_by_offset"]
        print(f"| {r.get('label') or r['out']} | " + " | ".join(str(p.get(str(o), "")) for o in offs)
              + f" | {', '.join(str(v) for v in r.get('psnr_ref_batch_interior', []))} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", nargs="?")
    ap.add_argument("--input")
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--ref")
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--overlap", type=int, default=0)
    ap.add_argument("--curve", default="numz")
    ap.add_argument("--hold", type=float, default=1.0)
    ap.add_argument("--latent", help="W:M, latent-space stitching (blend_patch.py STITCH_LATENT)")
    ap.add_argument("--label")
    ap.add_argument("--json")
    ap.add_argument("--summary", nargs="+")
    ap.add_argument("--profile", nargs="+")
    a = ap.parse_args()
    if a.summary:
        return summary(a.summary)
    if a.profile:
        return profile(a.profile)
    if not (a.out and a.input and a.ref and (a.batch or a.latent)):
        ap.error("OUT_DIR, --input, --ref and --batch are required")
    res = analyse(a)
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_transition", "per_frame")}, ensure_ascii=False))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(res, f)


if __name__ == "__main__":
    main()

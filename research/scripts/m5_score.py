#!/usr/bin/env python3
"""Milestone 5's scoring: the research scripts' metrics, frames read from 16-bit RGB masters.

quality_metrics.py and stitch_metrics.py read 8-bit PNGs (cv2.IMREAD_COLOR). Milestone 5 scores
both sides alike from their 16-bit FFV1 masters (seedvr2x/DESIGN.md, Validation milestones 5), so
this wrapper hands those scripts the masters' frames instead, float32 on their 8-bit scale
(v / 257, rounded nowhere), and runs their metric code unchanged. The input is read as they read
it (cv2, the 8-bit RGB file both runs read).

  m5_score.py quality OUT.mkv --input IN.mkv [--ref REF.mkv] [any quality_metrics.py option]
  m5_score.py stitch OUT.mkv --input IN.mkv --ref REF.mkv --latent W:M [stitch_metrics.py opts]
  m5_score.py psnr A.mkv B.mkv [--json F]     PSNR of A against B, 8-bit scale, from the masters
  m5_score.py verdict NAME --ours Q.json --numz Q.json [--fr-ours F.json --fr-numz F.json]
                      [--stitch-ours S.json --stitch-numz S.json] [--psnr P.json]
                      one Markdown row per metric: ours against numz, with the milestone's
                      tolerances

Historical: the record of milestone 5's first scoring (2026-10-04), kept to re-run it: what
m5_score.sh ran, numz's masters against ours as first run (seedvr2x 27ce6ba, lab's bf16
reference; 4 of 11 materials passed). Its verdict has the rules as first written: an a*/b*
spread is its percent off the input's, within 1%; all four boundary steps decide, the raw ones
too. m5_score3b.py is this file with the rules as sharpened on 2026-10-04, for step 3b; the
JSONs of quality, stitch and psnr are the same code. seedvr2x's `--color-correction lab` went
with 68b1529, when split replaced lab. It runs research/scripts' quality_metrics.py and
stitch_metrics.py as they were at eca0ff1.

Environment: M5_SCRIPTS, required by quality and stitch: the directory of quality_metrics.py and
stitch_metrics.py (research/scripts). Needs numpy and opencv (numz's venv), ffmpeg/ffprobe on
PATH.
"""
import json
import math
import os
import subprocess
import sys

import numpy as np

SCRIPTS = os.environ.get("M5_SCRIPTS")
SEP = "::"
SCALE = 65535 / 255  # 16-bit code -> the scripts' 8-bit scale

_masters = {}


def master(path):
    """The frames of a 16-bit RGB FFV1 master, (n, H, W, 3) uint16 RGB, decoded as stored."""
    if path not in _masters:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height,pix_fmt", "-of", "json", path],
            check=True, capture_output=True, text=True)
        stream = json.loads(probe.stdout)["streams"][0]
        if stream["pix_fmt"] != "gbrp16le":
            raise SystemExit(f"{path}: {stream['pix_fmt']}, not a gbrp16le master")
        w, h = stream["width"], stream["height"]
        # gbrp16le out as stored: no conversion at all, planes G, B, R.
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-map", "0:v:0", "-f", "rawvideo",
             "-pix_fmt", "gbrp16le", "-"],
            check=True, capture_output=True).stdout
        planes = np.frombuffer(raw, "<u2").reshape(-1, 3, h, w)
        _masters[path] = np.ascontiguousarray(planes[:, [2, 0, 1]].transpose(0, 2, 3, 1))
    return _masters[path]


def install():
    """quality_metrics and stitch_metrics, their PNG readers taking masters too."""
    if not SCRIPTS:
        sys.exit(f"{os.path.basename(sys.argv[0])}: set M5_SCRIPTS to the directory of"
                 " quality_metrics.py and stitch_metrics.py")
    sys.path.insert(0, SCRIPTS)
    import quality_metrics as qm
    import stitch_metrics as sm

    png_frames, read_png = qm.png_frames, qm.read_png

    def frames(d):
        if d and d.endswith(".mkv"):
            return [f"{d}{SEP}{i}" for i in range(master(d).shape[0])]
        return png_frames(d)

    def read(p):
        if SEP in p:
            path, index = p.rsplit(SEP, 1)
            return master(path)[int(index)].astype(np.float32) / np.float32(SCALE)
        return read_png(p)

    for module in (qm, sm):
        module.png_frames, module.read_png = frames, read

    analyse = qm.analyse

    def analyse_unrounded(a):
        # analyse rounds lab_mean_std to 2 decimals, steps of 0.5% on an a* spread of 2: the
        # spreads again, unrounded, from the same frames, resize and CIELAB as analyse's.
        res = analyse(a)
        n = res["frames"]
        outs = frames(a.out)[a.drop_first:]
        ins = qm.video_frames(a.input, a.skip, n)
        h, w = ins[0].shape[:2]
        acc = []
        for i in range(n):
            o = read(outs[i])
            if o.shape[:2] != (h, w):
                o = qm.cv2.resize(o, (w, h), interpolation=qm.cv2.INTER_AREA)
            lo, li = qm.lab(o).reshape(-1, 3), qm.lab(ins[i]).reshape(-1, 3)
            acc.append(np.r_[lo.std(0), li.std(0)])
        m = np.mean(acc, axis=0)
        res["lab_std_out"], res["lab_std_in"] = m[:3].tolist(), m[3:].tolist()
        return res

    qm.analyse = analyse_unrounded
    return qm, sm


def psnr(a_path, b_path, out=None):
    a, b = master(a_path), master(b_path)
    if a.shape != b.shape:
        raise SystemExit(f"{a.shape} against {b.shape}")
    per_frame, total = [], 0.0
    for i in range(a.shape[0]):
        d = (a[i].astype(np.float64) - b[i].astype(np.float64)) / SCALE
        mse = float((d * d).mean())
        total += mse
        per_frame.append(10 * math.log10(255**2 / mse) if mse else float("inf"))
    diff = int(max(np.abs(a[i].astype(np.int32) - b[i].astype(np.int32)).max()
                   for i in range(a.shape[0])))
    mse = total / a.shape[0]
    res = {"a": a_path, "b": b_path, "frames": a.shape[0],
           "psnr": round(10 * math.log10(255**2 / mse), 3) if mse else float("inf"),
           "psnr_frame_mean": round(float(np.mean(per_frame)), 3),
           "psnr_frame_min": round(float(np.min(per_frame)), 3),
           "max_abs_diff_16bit": diff,
           "identical": bool(all(np.array_equal(a[i], b[i]) for i in range(a.shape[0]))),
           "per_frame": [round(v, 3) for v in per_frame]}
    print(json.dumps({k: v for k, v in res.items() if k != "per_frame"}))
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(res, f)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def verdict(argv):
    """Milestone 5's rule: ours at least as good as numz on every metric, a difference below the
    tolerance counting as equal (DESIGN.md, Validation milestones 5)."""
    import argparse

    ap = argparse.ArgumentParser(prog="m5_score.py verdict")
    ap.add_argument("name")
    for k in ("ours", "numz", "fr-ours", "fr-numz", "stitch-ours", "stitch-numz", "psnr"):
        ap.add_argument(f"--{k}")
    a = ap.parse_args(argv)
    rows = []

    def row(metric, ours, numz, better, tol, unit=""):
        if ours is None or numz is None:  # e.g. no held transition at any boundary
            rows.append(f"| {a.name} | {metric} | {ours} | {numz} | | {tol}{unit} | n/a |")
            return True
        if better == "lower":
            ok = ours <= numz + tol
        else:  # closer to 0: the values given are already distances
            ok = abs(ours) <= abs(numz) + tol
        rows.append(f"| {a.name} | {metric} | {ours:.4g}{unit} | {numz:.4g}{unit} | "
                    f"{ours - numz:+.4g} | {tol}{unit} | {'yes' if ok else 'NO'} |")
        return ok

    ok = True
    if a.ours and a.numz:
        q, n = load(a.ours), load(a.numz)
        ok &= row("ΔE lf to input", q["de76_lowfreq"], n["de76_lowfreq"], "lower", 0.1)
        for i, ch in ((1, "a*"), (2, "b*")):
            spread = lambda r: 100 * (r["lab_std_out"][i] / r["lab_std_in"][i] - 1)  # noqa: E731
            ok &= row(f"{ch} spread vs input's", spread(q), spread(n), "closer", 1.0, "%")
        ok &= row("Y shift", q["luma_shift"], n["luma_shift"], "closer", 0.1)
    if a.fr_ours and a.fr_numz:
        f, g = load(a.fr_ours), load(a.fr_numz)
        ok &= row("ΔE00 lf to GT", f["means"]["de00_lf"], g["means"]["de00_lf"], "lower", 0.1)
    if a.stitch_ours and a.stitch_numz:
        s, t = load(a.stitch_ours), load(a.stitch_numz)
        for key, title in (("hold_excess", "hold excess step"), ("lf_excess", "lf excess step"),
                           ("hold_tdiff", "hold step"), ("added_lf", "lf step")):
            ok &= row(title, s[key]["step"], t[key]["step"], "lower", 0.02)
    if a.psnr:
        p = load(a.psnr)
        rows.append(f"| {a.name} | PSNR ours vs numz (information) | {p['psnr']} dB | | | | |")
    print("\n".join(rows))
    print(f"{a.name}: {'ACCEPTED' if ok else 'NOT ACCEPTED'}")
    return 0 if ok else 1


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    command, rest = sys.argv[1], sys.argv[2:]
    if command == "psnr" and len(rest) in (2, 4):
        return psnr(rest[0], rest[1], rest[3] if len(rest) == 4 and rest[2] == "--json" else None)
    if command == "verdict":
        return verdict(rest)
    if command in ("quality", "stitch"):
        qm, sm = install()
        sys.argv = [f"{command}_metrics.py", *rest]
        return (qm if command == "quality" else sm).main()
    sys.exit(__doc__)


if __name__ == "__main__":
    sys.exit(main())

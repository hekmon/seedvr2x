#!/usr/bin/env python3
"""Step 7's blind set: the sharp 7B against the 7B at x1.5, x3 and x4, each after its split, for the eyes. CPU only.

  colour_factor_set.py DUMPS CLIPS OUT KEYDIR [--threads 16] [--runs RUN,...]

DUMPS/<clip>-d1x<F>/{s42/decode.pt, s42/ref_f32.pt, sh42/decode.pt} (colour_dump.py's, the 7B's run with the
encoder's reference and the sharp's on the same input), CLIPS/<clip>.gt.mkv (fr_clips.py's ground truth). Each run
gets seedvr2x's split for its factor (VARIANT), applied to both decodes with the same reference by colour_crops.py's
own code.

Places (stills), per run:
- the hand windows step 6's eyes judged at x2 (HAND: same window, same frame, so that each reads against its x2
  verdict);
- the window where the two corrected pictures differ most (two on live-vfx at x4): the 384x216 window on a 24-px
  grid, inside the picture rows, overlapping no hand window by more than 25% of its area, of the largest mean
  CIEDE2000 between the two over frames 4, 13, 22, 31 and 40 taken one by one (step 7's first crops' rule).
A window must hold something to recognise: at least REC (10%) of the GT window's pixels with a luma gradient of 4
levels/px or more (Sobel 3x3 / 8, 8-bit scale: measurement's rule for its blind sets, quality_set.py, calibrated on
the places its eyes couldn't judge). Where a window falls short, the nearest window on the same frame (centre
distance, same size, 24-px grid, inside the picture rows, same overlap rule) that reaches REC is taken, and the
set's notes say so; where the frame has none (a hand window on a blurred frame), the nearest frame that has one
(the window nearest the hand window's position there). A run with a recognisable window on fewer than MIN_FRAMES (8)
of its frames 4-44 is blurred all over: no places, no loop, its figures only (said in the notes).
Each place: A, B and the GT at 1:1 and x3 (nearest), 8-bit PNG from cv2 (no text chunks).
Loops: per run the whole 1920x1080 frame, A left, B right (3840x1080), the 45 frames three times, x264 CRF 12,
8-bit 4:2:0 BT.709 limited range, metadata stripped, bitexact (measurement's loops).
Sides: drawn with os.urandom, balanced within each factor's places (half each way, the odd one at random) and over
the loops; the key (which side is the sharp) goes to KEYDIR only, never to OUT.
"""
import argparse
import hashlib
import os
import random
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

VARIANT = {"1.5": "split:ycc:4:3", "3": "split:ycc:4:4", "4": "split:ycc:4:4"}  # the scale follows the factor
RUNS = [("anime-clean", "1.5"), ("cartoon-bright", "1.5"), ("live-slow", "1.5"),
        ("anime-clean", "3"), ("cartoon-bright", "3"), ("live-slow", "3"),
        ("anime-clean", "4"), ("anime-sky", "4"), ("cartoon-bright", "4"), ("live-slow", "4"), ("live-vfx", "4")]
PICTURE = {"live-slow": (131, 947), "live-vfx": (140, 939)}  # a letterboxed clip's picture rows (first, last)
N_MDIFF = {("live-vfx", "4"): 2}
MDIFF_FRAMES = (4, 13, 22, 31, 40)
WW, WH, STEP, MAX_OVERLAP, REC = 384, 216, 24, 0.25, 0.10
MIN_FRAMES = 8  # a run with a recognisable window on fewer of its frames 4-44 is blurred all over
# clip -> [(x, y, w, h, frame, what it shows)]: the windows step 6's eyes judged at x2 (anime-sky's: step 4's)
HAND = {
    "anime-clean": [(800, 864, 384, 216, 22, "three small cel characters before an arcade"),
                    (400, 480, 384, 216, 22, "a facade: arched windows with fine lattice lines, columns, foliage")],
    "cartoon-bright": [(784, 232, 384, 216, 22, "the dog character's face, wind chimes, a sunflower"),
                       (536, 620, 384, 216, 22, "flowers in a bucket, the table cloth")],
    "live-slow": [(1326, 236, 384, 216, 44, "two faces in profile; a palm and a sunlit window grille behind"),
                  (0, 440, 384, 216, 44, "a seated man's face, a bow tie; the window grille behind")],
    "live-vfx": [(300, 232, 384, 216, 44, "the soldier's face, eyes to mouth, film grain"),
                 (200, 560, 384, 216, 44, "an exosuit's shoulder with dirt specks, film grain")],
    "anime-sky": [(150, 210, 480, 270, 44, "cumulus clouds, a dark cloud band"),
                  (1440, 810, 480, 270, 2, "a blurred dark shape against a pale sky")],
}
ZY = "zscale=threads=1:dither=none:min=gbr:m=709:rin=full:r=limited:pin=709:p=709:tin=709:t=709:c=left,format=yuv420p"
RNG = random.Random(int.from_bytes(os.urandom(8), "big"))


def say(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))


def rec_map(g):
    """1 where the GT's luma gradient reaches 4 levels/px (Sobel 3x3 / 8, 8-bit scale), else 0."""
    y = (0.2126 * g[..., 0] + 0.7152 * g[..., 1] + 0.0722 * g[..., 2]) * 255.0
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    return (np.sqrt(gx * gx + gy * gy) / 8 >= 4).astype(np.float32)


def grid(K, m, h, w, ww, wh, top, bottom, avoid):
    """[(mean of m, x, y)] over the ww x wh windows on the STEP grid inside rows top..bottom, none overlapping a
    window of avoid by more than MAX_OVERLAP of its area."""
    ys, xs, s = K.window_scores(m, h, w, wh, ww, STEP)
    out = []
    for iy, y in enumerate(ys):
        if y < top or y + wh - 1 > bottom:
            continue
        for ix, x in enumerate(xs):
            if avoid and max(overlap((x, y, ww, wh), a) for a in avoid) > MAX_OVERLAP * ww * wh:
                continue
            out.append((float(s[iy, ix]), int(x), int(y)))
    return out


def nearest_rec(K, rm, h, w, x, y, ww, wh, top, bottom, avoid):
    """The recognisable window (x, y, rec) nearest to (x, y) by centre distance, or None."""
    ok = [(r, xx, yy) for r, xx, yy in grid(K, rm, h, w, ww, wh, top, bottom, avoid) if r >= REC]
    if not ok:
        return None
    r, xx, yy = min(ok, key=lambda c: ((c[1] - x) ** 2 + (c[2] - y) ** 2, c[2], c[1]))
    return xx, yy, r


def balanced(n):
    sides = [True] * (n // 2) + [False] * (n // 2) + ([RNG.random() < 0.5] if n % 2 else [])
    RNG.shuffle(sides)
    return sides


def png(path, rgb):
    """float RGB [0, 1] -> 8-bit PNG."""
    cv2.imwrite(path, np.ascontiguousarray(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1]))


def loop(path, left, right, t):
    """left | right (dict frame -> (H, W, 3) float RGB), frames 0..t-1 three times, x264 CRF 12, 4:2:0."""
    h, w = left[0].shape[:2]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                          f"{2 * w}x{h}", "-r", "24000/1001", "-i", "-", "-vf", "format=gbrp," + ZY, "-c:v", "libx264",
                          "-crf", "12", "-preset", "slow", "-colorspace", "bt709", "-color_primaries", "bt709",
                          "-color_trc", "bt709", "-color_range", "tv", "-map_metadata", "-1", "-fflags", "+bitexact",
                          "-flags:v", "+bitexact", "-movflags", "+faststart", path], stdin=subprocess.PIPE)
    frames = [np.rint(np.clip(np.concatenate([left[i], right[i]], axis=1), 0, 1) * 255).astype(np.uint8).tobytes()
              for i in range(t)]
    for _ in range(3):
        for b in frames:
            p.stdin.write(b)
    p.stdin.close()
    if p.wait() != 0:
        raise SystemExit(f"ffmpeg failed on {path}")


def one_run(K, D, clip, factor, dumps, clips):
    spec = VARIANT[factor]
    d = f"{dumps}/{clip}-d1x{factor}"
    outs = {}
    for tag, path in (("7B", f"{d}/s42/decode.pt"), ("sharp", f"{d}/sh42/decode.pt")):
        fv = K.load_decode(path)
        t, h, w = fv.shape[:3]
        c = K.to_content(fv)
        del fv
        outs[tag] = K.variant_frames(c, f"{d}/s42/ref_f32.pt", [spec])[spec]
        del c
    gts = K.read_frames(f"{clips}/{clip}.gt.mkv", t, h, w)
    top, bottom = PICTURE.get(clip, (0, h - 1))
    rms = {f: rec_map(gts[f]) for f in range(t)}
    # blurred all over? frames 4-44 with a recognisable 384x216 window
    has = sum(1 for f in range(4, t) if any(r >= REC for r, _, _ in grid(K, rms[f], h, w, WW, WH, top, bottom, [])))
    run = dict(clip=clip, factor=factor, spec=spec, t=t, h=h, w=w, rec_frames=has, places=[], notes=[], outs=outs,
               gts=gts)
    if has < MIN_FRAMES:
        run["notes"].append(f"{clip} x{factor}: only {has} of its frames 4-{t - 1} hold a recognisable window "
                            "(blurred all over): no places, no loop; its figures only.")
        return run
    hand = [(x, y, hw, hh) for x, y, hw, hh, *_ in HAND[clip]]
    flat = []
    for x, y, hw, hh, f, what in HAND[clip]:
        r = float(rms[f][y:y + hh, x:x + hw].mean())
        pl = dict(kind="hand", frame=f, x=x, y=y, w=hw, h=hh, rec=r, shows=what, moved=None)
        (flat if r < REC else run["places"]).append(pl)
    cands = []
    des = {}
    for f in MDIFF_FRAMES:
        des[f] = D.ciede2000(D.to_lab(outs["7B"][f]), D.to_lab(outs["sharp"][f]))[0]
        cands += [(m, f, x, y) for m, x, y in grid(K, des[f], h, w, WW, WH, top, bottom, hand)]
    cands.sort(key=lambda c: (-c[0], c[1], c[2], c[3]))
    picked = []
    for m, f, x, y in cands:
        if len(picked) >= N_MDIFF.get((clip, factor), 1):
            break
        if any(overlap((x, y, WW, WH), (p["x"], p["y"], WW, WH)) > 0 for p in picked):
            continue
        r = float(rms[f][y:y + WH, x:x + WW].mean())
        pl = dict(kind="mdiff", frame=f, x=x, y=y, w=WW, h=WH, rec=r, de=m, moved=None,
                  frame_de=float(des[f][top:bottom + 1].mean()))
        if r < REC:
            nr = nearest_rec(K, rms[f], h, w, x, y, WW, WH, top, bottom,
                             hand + [(p["x"], p["y"], WW, WH) for p in picked])
            if not nr:
                continue
            pl.update(x=nr[0], y=nr[1], rec=nr[2], moved=(x, y, r),
                      de=float(des[f][nr[1]:nr[1] + WH, nr[0]:nr[0] + WW].mean()))
            run["notes"].append(f"{clip} x{factor}: the largest difference on frame {f} sits on a flat window "
                                f"(x {x} y {y}, {100 * r:.1f}% pixels with detail); the nearest window with detail "
                                "is taken.")
        picked.append(pl)
    run["places"] += picked
    # a flat hand window: the nearest window with detail on its frame, else on the nearest frame that has one,
    # overlapping none of the run's other places (on any frame) by more than MAX_OVERLAP of its area
    for pl in flat:
        x, y, hw, hh, f, r = pl["x"], pl["y"], pl["w"], pl["h"], pl["frame"], pl["rec"]
        avoid = [(p["x"], p["y"], p["w"], p["h"]) for p in run["places"]]
        for g in sorted(range(t), key=lambda k: (abs(k - f), k)):
            nr = nearest_rec(K, rms[g], h, w, x, y, hw, hh, top, bottom, avoid)
            if nr:
                break
        if not nr:
            run["notes"].append(f"{clip} x{factor}: the x2-judged window at x {x} y {y} (frame {f}) is flat and no "
                                "frame has a window with detail apart from the other places: left out.")
            continue
        pl.update(frame=g, x=nr[0], y=nr[1], rec=nr[2], moved=(x, y, r), shows="")
        run["notes"].append(f"{clip} x{factor}: the x2-judged window at x {x} y {y} on frame {f} holds "
                            f"{100 * r:.1f}% pixels with detail; the nearest window with detail is taken"
                            + (f", on frame {g}, the nearest frame that has one." if g != f else "."))
        run["places"].append(pl)
    say(f"{clip} x{factor}: {spec}, {t} frames {w}x{h}, rows {top}-{bottom}, {has} frames with detail; places "
        + "; ".join(f"{p['kind']} f{p['frame']} x{p['x']} y{p['y']} rec {100 * p['rec']:.1f}%"
                    + (f" de {p['de']:.3f}" if 'de' in p else "") + (" MOVED" if p["moved"] else "")
                    for p in run["places"]))
    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dumps")
    ap.add_argument("clips")
    ap.add_argument("out")
    ap.add_argument("keydir")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--runs", default="", help="clip-xF,... (default: all eleven)")
    a = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        sys.exit('CPU only: export CUDA_VISIBLE_DEVICES="" first')
    import torch
    torch.set_num_threads(a.threads)
    import colour_crops as K
    import colour_diag as D
    runs = [r for r in RUNS if not a.runs or f"{r[0]}-x{r[1]}" in a.runs.split(",")]
    if os.path.exists(a.out) and os.listdir(a.out):
        sys.exit(f"{a.out} is not empty")
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.keydir, exist_ok=True)
    # sides drawn before the runs are read (each run's files are written once it is read, then freed): balanced
    # over each factor's expected places and over the loops; a place left out shifts the balance by one at most
    sides = {}
    for factor in VARIANT:
        items = [(clip, j) for clip, f in runs if f == factor
                 for j in range(len(HAND[clip]) + N_MDIFF.get((clip, f), 1))]
        sides.update({(factor,) + it: s for it, s in zip(items, balanced(len(items)))})
    loop_side = dict(zip(runs, balanced(len(runs))))
    index, key, notes, verdicts, loops, loop_verdicts = [], [], [], [], [], []
    n = 0
    for clip, factor in runs:
        r = one_run(K, D, clip, factor, a.dumps, a.clips)
        notes += r["notes"]
        for j, p in enumerate(r["places"]):
            n += 1
            a_, b_ = ("sharp", "7B") if sides[(factor, clip, j)] else ("7B", "sharp")
            f, x, y, w, h = p["frame"], p["x"], p["y"], p["w"], p["h"]
            for lab, img in (("A", r["outs"][a_][f]), ("B", r["outs"][b_][f]), ("GT", r["gts"][f])):
                win = img[y:y + h, x:x + w]
                png(f"{a.out}/place{n:02d}_{lab}.png", win)
                png(f"{a.out}/place{n:02d}_{lab}_x3.png",
                    cv2.resize(win, (3 * w, 3 * h), interpolation=cv2.INTER_NEAREST))
            index.append(f"place {n:02d}: x{factor}, clip {clip}, frame {f}, window x {x} y {y} ({w}x{h})")
            verdicts.append(f"place {n:02d} (x{factor} {clip}): ")
            key.append("\t".join([f"place{n:02d}", clip, factor, r["spec"], p["kind"], str(f), str(x), str(y), str(w),
                                  str(h), f"{p['rec']:.4f}", f"{p.get('de', float('nan')):.4f}",
                                  f"{p.get('frame_de', float('nan')):.4f}",
                                  "" if not p["moved"] else "moved from x%d y%d (rec %.4f)" % p["moved"],
                                  p.get("shows", ""), a_, b_]))
        if r["places"]:
            m = len(loops) + 1
            a_, b_ = ("sharp", "7B") if loop_side[(clip, factor)] else ("7B", "sharp")
            loop(f"{a.out}/loop{m:02d}.mp4", r["outs"][a_], r["outs"][b_], r["t"])
            loops.append(f"loop {m:02d}: x{factor}, clip {clip}, frames 0-{r['t'] - 1}, the whole {r['w']}x{r['h']} "
                         "frame, A | B, three times")
            loop_verdicts.append(f"loop {m:02d} (x{factor} {clip}): ")
            key.append("\t".join([f"loop{m:02d}", clip, factor, r["spec"], "loop"] + [""] * 10 + [a_, b_]))
            say(f"loop {m:02d} written")
        del r
    index += loops
    verdicts += loop_verdicts
    m = len(loops)
    head = [
        "1080p. The SeedVR2 7B from smaller inputs: x1.5 (1280x720), x3 (640x360) and x4 (480x270), seed 42.",
        "A and B are two models' outputs of the same input, each with seedvr2x's colour correction (the key says "
        "which).",
        "Each place: the same frame and window of the 1920x1080 frame, at 1:1 and x3 (nearest); GT = the original, "
        "apart:",
        "judge A against B first, open the GT after. Every window holds something to recognise (see the notes).",
        "For each place and loop, answer one of:",
        "  A or B   = that one looks better (a word on why helps: sharper, cleaner, more natural...);",
        "  same     = you see no difference;",
        "  ?        = they differ but neither is better (e.g. one sharper, the other smoother); say how if you can.",
        "Loops: the whole frame, A left, B right (3840x1080), the 45 frames three times (VLC: loop on). Judge them "
        "in motion;",
        "a difference you only see once paused: say so (it counts with the stills).",
        "View at 100% (one image pixel on one screen pixel).",
        "",
    ]
    with open(f"{a.out}/index.txt", "w") as fh:
        fh.write("\n".join(head + index + ["", "notes:"] + (notes or ["none: every window held detail as placed."]))
                 + "\n")
    with open(f"{a.out}/verdicts.txt", "w") as fh:
        fh.write("Write after each line: A, B, same or ?, then a comment if you have one.\n\n" + "\n".join(verdicts)
                 + "\n")
    kp = f"{a.keydir}/key.tsv"
    with open(kp, "w") as fh:
        fh.write("\t".join(["item", "clip", "factor", "variant", "kind", "frame", "x", "y", "w", "h", "rec",
                            "models_de_window", "models_de_frame", "moved", "shows", "A", "B"]) + "\n")
        fh.write("\n".join(key) + "\n")
    with open(kp, "rb") as fh:
        say(f"set: {n} places, {m} loops in {a.out}; key {kp} md5 {hashlib.md5(fh.read()).hexdigest()}")


if __name__ == "__main__":
    main()

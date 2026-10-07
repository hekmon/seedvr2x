#!/usr/bin/env python3
"""Side-by-side crops of colour-correction variants for the eyes (docs/colour.md, step 4).

  colour_crops.py --clip NAME --gt GT.mkv --ref REF.pt --content [LABEL=]DECODE.pt [--content LABEL=DECODE.pt ...]
                  --variants BASE,V1[,V2...] --out DIR [--bicubic BICUBIC.mkv] [--per-kind 2]
                  [--at X,Y[,W,H] ...] [--at-frame F[,F...]] [--size 480x270] [--no-stretch] [--threads 8]

Why: the scores rank the variants against a ground truth; the user's eyes decide whether a variant
that scores better also looks right (DESIGN.md, Beyond numz's lab: crops of edges, skin, skies and
flat areas).

The variants (colour_variants.py specs, the first being the baseline; ds<K>_<A> ones included) are
applied to one dump (colour_dump.py's decode and reference), or to several decodes of one input
(two models', two seeds'...): --content LABEL=DECODE.pt, repeated, with the one --ref; every variant
then runs on every decode, in the order given, each panel labelled "LABEL VARIANT" (a plain
--content DECODE.pt labels the panels by their variant alone). --bicubic adds the bicubic baseline
(an RGB master as the GT, e.g. CLIP.d1.bicubic.mkv, its size the GT's) as a panel right after the
GT's. Any frame size works (4K: 3840x2160, and cropped 3840x2016 or 3840x2048); --size 640x360 suits
4K. --at X,Y[,W,H] (repeatable) cuts a given window (top-left corner X, Y; size W x H, --size by
default) on each frame of --at-frame (the middle frame by default), named <clip>-at-f<frame>-x<x>-y<y>.
Windows of --size are also picked automatically on the first decode, --per-kind per kind (0: none,
the frames then not scanned), at most one per frame, not overlapping, ranked by:
- diff: the largest mean colour difference (CIEDE2000, unblurred) between the last variant and the
  baseline
- edge: the most strong edges of the GT (the top 5% of its luma gradient)
- flat: the flattest GT windows (lowest mean luma gradient) that are not near black
- skin: the most pixels of skin tones in the GT (CIELAB hue 20-60°, chroma 10-45, L* 40-90)
- sky: the most pixels of sky (hue 190-270°, L* > 55, low gradient)
Each crop is written as DIR/<clip>-<kind>-f<frame>-x<x>-y<y>.png: GT, bicubic (with --bicubic),
then each decode's variants, side by side at 1:1 with a label strip (a label wider than its panel
on several lines), and, without --no-stretch, .stretch.png: the same with every panel's luma
stretched by the GT window's 1st-99th percentiles (small colour and level shifts become visible).
DIR/crops.tsv lists them with each panel's mean ΔE00 to the GT over the window.
Memory: the decodes are converted to float one at a time, and when every variant treats each frame
alone (colour_variants.per_frame), on the frames cut only: the --at frames, and the picked frames
(the first decode's every frame then, for the scan).
"""
import argparse
import os
import re
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import colour_diag as D  # noqa: E402
import colour_eval as E  # noqa: E402

FONT, SCALE = cv2.FONT_HERSHEY_SIMPLEX, 0.5
LABEL_H = 22  # a label strip of one line
LINE_H = 18  # each further line of a label wrapped to its panel's width


def window_scores(score_map, h, w, wh, ww, step):
    """Mean of score_map over every window (wh, ww) on a grid of `step` px: (ys, xs, means)."""
    ii = cv2.integral(score_map.astype(np.float64))
    ys = np.arange(0, h - wh + 1, step)
    xs = np.arange(0, w - ww + 1, step)
    s = ii[ys[:, None] + wh, xs[None, :] + ww] - ii[ys[:, None], xs[None, :] + ww] \
        - ii[ys[:, None] + wh, xs[None, :]] + ii[ys[:, None], xs[None, :]]
    return ys, xs, s / (wh * ww)


def stretch(rgb, lo, hi):
    """Luma stretched from [lo, hi] to [0, 1], colour kept (RGB scaled around the luma)."""
    y = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    ys = np.clip((y - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.clip(rgb + (ys - y)[..., None], 0, 1)


def label_lines(head, tail, w):
    """A panel's label as the lines that fit its width w at 1:1: "HEAD  TAIL" (TAIL the ΔE, or None)
    when it fits, else HEAD's words line after line (a word wider than a line cut after its ':'s),
    then TAIL where it fits."""
    fits = lambda s: cv2.getTextSize(s, FONT, SCALE, 1)[0][0] <= w - 12  # noqa: E731
    one = head if tail is None else f"{head}  {tail}"
    if fits(one):
        return [one]
    lines, line = [], ""
    for word in head.split():
        for k, piece in enumerate([word] if fits(word) else re.findall(r"[^:]*:|[^:]+", word)):
            more = line + (" " if line and k == 0 else "") + piece
            if line and not fits(more):
                lines.append(line)
                more = piece
            line = more
    if tail is not None:
        if fits(f"{line}  {tail}"):
            line = f"{line}  {tail}"
        else:
            lines, line = lines + [line], tail
    return lines + [line]


def panel(img, lines, w, n):
    """img under a label strip of n lines, lines (at most n) written on it."""
    strip = np.zeros((LABEL_H + (n - 1) * LINE_H, w, 3), np.uint8)
    for k, s in enumerate(lines):
        cv2.putText(strip, s, (6, 16 + k * LINE_H), FONT, SCALE, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([strip, img])


def read_frames(path, t, h, w, keep=None):
    """The first t frames of an RGB master (colour_diag.Master), which must be w x h: float32 RGB
    (h, w, 3) in [0, 1]; with keep, a set of frame indices, the others None (read, not kept)."""
    src = D.Master(path, 4)
    frames, short = [], False
    for i in range(t):
        x = src.read()
        short |= x is None
        frames.append(x if keep is None or i in keep else None)
    src.close()
    if short or (src.H, src.W) != (h, w):
        raise SystemExit(f"{path}: {src.W}x{src.H}, fewer than {t} frames or not the dump's {w}x{h}")
    return frames


def parse_content(values):
    """--content values -> [(label, path)]: LABEL=DECODE.pt, or a plain DECODE.pt (label None: the
    panels then labelled by their variant alone), one only."""
    out = [tuple(v.split("=", 1)) if "=" in v and not os.path.exists(v) else (None, v) for v in values]
    labels = [label for label, _ in out]
    if len(out) > 1 and not all(labels):
        raise SystemExit("--content: several decodes, each as LABEL=DECODE.pt")
    if len(set(labels)) < len(labels):
        raise SystemExit(f"--content: the labels must differ ({', '.join(map(str, labels))})")
    return out


def load_decode(path):
    """colour_dump.py's decode as stored (colour_eval.load_content's input): (T, H, W, 3), bf16, in [-1, 1]."""
    import torch
    return torch.load(path)["final_video"]


def to_content(fv, frames=None):
    """colour_eval.load_content's conversion of a decode (load_decode's), of the given frames only (None:
    all): float32 (n, 3, H, W) in [-1, 1]."""
    import torch
    return (fv if frames is None else fv[frames]).permute(0, 3, 1, 2).to(torch.float32).contiguous()


def variant_frames(c, ref_path, specs, frames=None):
    """Every variant of specs on c (to_content's, of the given frames, None: all) and the reference at
    ref_path (colour_eval.load_reference's, of the same frames): {spec: {frame: (H, W, 3) float32 RGB
    in [0, 1]}}. Only variants that treat each frame alone (colour_variants.per_frame) may run on
    some frames."""
    import torch
    import colour_variants as V
    n, _, h, w = c.shape
    if frames is None:
        ref = E.load_reference(ref_path, n, h, w)
    else:
        batches = torch.load(ref_path)
        if len(batches) != 1:
            raise SystemExit(f"{ref_path}: {len(batches)} batches, one expected")
        ref = batches[0].permute(1, 0, 2, 3)[frames, :, :h, :w].to(torch.float32).contiguous()
        del batches
    outs = {}
    for spec in specs:
        with torch.inference_mode():
            o = V.apply(spec, c, ref).permute(0, 2, 3, 1).contiguous().numpy()
        outs[spec] = dict(zip(range(n) if frames is None else frames, o))
    return outs


def parse_at(spec, size, h, w):
    """--at X,Y[,W,H] -> (x, y, ww, wh), inside the w x h frame."""
    v = [int(s) for s in spec.split(",")]
    if len(v) not in (2, 4):
        raise SystemExit(f"--at {spec}: X,Y or X,Y,W,H")
    x, y, ww, wh = v if len(v) == 4 else v + list(size)
    if x < 0 or y < 0 or ww <= 0 or wh <= 0 or x + ww > w or y + wh > h:
        raise SystemExit(f"--at {spec}: the window {ww}x{wh} at ({x}, {y}) is not inside the {w}x{h} frame")
    return x, y, ww, wh


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clip", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--ref", required=True, help="colour_dump.py's reference: the one of every --content")
    ap.add_argument("--content", required=True, action="append", metavar="[LABEL=]DECODE.pt",
                    help="colour_dump.py's decode: a plain DECODE.pt (panels labelled by their variant), or "
                         "LABEL=DECODE.pt, repeatable (decodes of one input: every variant on each, in order, "
                         "panels labelled 'LABEL VARIANT')")
    ap.add_argument("--variants", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bicubic", help="the bicubic baseline, an RGB master as the GT: a panel after the GT's")
    ap.add_argument("--per-kind", type=int, default=2,
                    help="automatic windows per kind (0: none), picked on the first --content")
    ap.add_argument("--at", action="append", default=[], metavar="X,Y[,W,H]",
                    help="a given window (top-left corner, size: --size by default), on each --at-frame; repeatable")
    ap.add_argument("--at-frame", default="", metavar="F[,F...]",
                    help="the frames the --at windows are cut on (default: the middle frame)")
    ap.add_argument("--size", default="480x270")
    ap.add_argument("--no-stretch", action="store_true", help="no luma-stretched .stretch.png twins")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    import torch
    import colour_variants as V
    torch.set_num_threads(a.threads)
    cv2.setNumThreads(a.threads)
    ww, wh = (int(v) for v in a.size.split("x"))
    specs = [v for v in a.variants.split(",") if v]
    contents = parse_content(a.content)
    names = [s if label is None else f"{label} {s}" for label, _ in contents for s in specs]  # the panels'
    fv = load_decode(contents[0][1])
    t, h, w = fv.shape[:3]
    ats = [parse_at(s, (ww, wh), h, w) for s in a.at]
    at_frames = [int(f) for f in a.at_frame.split(",") if f] or [t // 2]
    if any(not 0 <= f < t for f in at_frames):
        raise SystemExit(f"--at-frame {a.at_frame}: the dump has frames 0-{t - 1}")
    if not ats and a.per_kind <= 0:
        raise SystemExit("no window: give --at, or --per-kind above 0")
    # every frame for the scan, or for a variant pooled over the shot (a histogram step); else the --at frames
    whole = not all(V.per_frame(s) for s in specs)
    frames = None if whole or a.per_kind > 0 else sorted(set(at_frames))
    c = to_content(fv, frames)
    del fv
    outs = variant_frames(c, a.ref, specs, frames)
    del c
    need = t if a.per_kind > 0 else max(at_frames) + 1  # without the scan, up to the --at frames
    keep = None if a.per_kind > 0 else set(at_frames)
    gts = read_frames(a.gt, need, h, w, keep)
    bic = read_frames(a.bicubic, need, h, w, keep) if a.bicubic else None
    os.makedirs(a.out, exist_ok=True)
    step = 30
    cands = {k: [] for k in ("diff", "edge", "flat", "skin", "sky")}
    for i in range(t if a.per_kind > 0 else 0):
        g = gts[i]
        lab = D.to_lab(g)
        y = D.luma8(g)
        grad = np.hypot(cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3))
        edges = (grad > np.percentile(grad, 95)).astype(np.float32)
        chroma = np.hypot(lab[..., 1], lab[..., 2])
        hue = np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360
        skin = ((hue >= 20) & (hue <= 60) & (chroma >= 10) & (chroma <= 45) & (lab[..., 0] >= 40)
                & (lab[..., 0] <= 90)).astype(np.float32)
        sky = ((hue >= 190) & (hue <= 270) & (lab[..., 0] > 55)
               & (grad < np.percentile(grad, 50))).astype(np.float32)
        base, last = D.to_lab(outs[specs[0]][i]), D.to_lab(outs[specs[-1]][i])
        diff = D.ciede2000(base, last)[0]
        for kind, m, sign in (("diff", diff, 1), ("edge", edges, 1), ("skin", skin, 1), ("sky", sky, 1),
                              ("flat", grad + 1000.0 * (y < 20), -1)):
            ys, xs, s = window_scores(m, h, w, wh, ww, step)
            k = np.argmax(s * sign)
            yy, xx = ys[k // len(xs)], xs[k % len(xs)]
            val = float(s.flat[k])
            if kind in ("skin", "sky") and val < 0.15:  # not enough of it in this frame
                continue
            cands[kind].append((sign * val, i, int(yy), int(xx)))
    windows = []  # (kind, score, frame, y, x, height, width)
    for kind, lst in cands.items():
        lst.sort(reverse=True)
        picked = []
        for val, i, yy, xx in lst:
            if len(picked) >= a.per_kind:
                break
            if any(abs(i - j) < 6 or (abs(yy - y2) < wh and abs(xx - x2) < ww) for _, j, y2, x2 in picked):
                continue
            picked.append((val, i, yy, xx))
        windows += [(kind, val, i, yy, xx, wh, ww) for val, i, yy, xx in picked]
    windows += [("at", None, i, y, x, ah, aw) for i in at_frames for x, y, aw, ah in ats]
    # every decode's variants cut to the windows, one decode in memory at a time (the first one's: outs)
    cuts = [[] for _ in windows]  # per window, the variants' crops in panel order
    cut_frames = None if whole else sorted({i for _, _, i, *_ in windows})
    for k, (label, path) in enumerate(contents):
        if k and windows:
            fv = load_decode(path)
            if tuple(fv.shape[:3]) != (t, h, w):
                raise SystemExit(f"{path}: {fv.shape[0]} frames of {fv.shape[2]}x{fv.shape[1]}, not the "
                                 f"first decode's {t} of {w}x{h}")
            c = to_content(fv, cut_frames)
            del fv
            outs = variant_frames(c, a.ref, specs, cut_frames)
            del c
        for (kind, val, i, yy, xx, ph, pw), cut in zip(windows, cuts):
            cut += [np.ascontiguousarray(outs[s][i][yy:yy + ph, xx:xx + pw]) for s in specs]
        outs = None
    labels = (["bicubic"] if bic else []) + names
    to8 = lambda x: np.rint(np.clip(x, 0, 1) * 255).astype(np.uint8)[..., ::-1]  # noqa: E731
    rows = []
    for (kind, val, i, yy, xx, ph, pw), cut in zip(windows, cuts):
        sl = (slice(yy, yy + ph), slice(xx, xx + pw))
        g = gts[i][sl]
        yg = 0.2126 * g[..., 0] + 0.7152 * g[..., 1] + 0.0722 * g[..., 2]
        lo, hi = np.percentile(yg, (1, 99))
        labg = D.to_lab(gts[i])[sl]
        imgs = [g] + ([bic[i][sl]] if bic else []) + cut
        des = [0.0] + [float(D.ciede2000(labg, D.to_lab(np.ascontiguousarray(img)))[0].mean()) for img in imgs[1:]]
        texts = [label_lines("GT", None, pw)] + [label_lines(s, f"dE {d:.2f}", pw) for s, d in zip(labels, des[1:])]
        n = max(len(x) for x in texts)
        name = f"{a.clip}-{kind}-f{i}-x{xx}-y{yy}" + (f"-{pw}x{ph}" if (pw, ph) != (ww, wh) else "")
        cv2.imwrite(os.path.join(a.out, name + ".png"),
                    np.hstack([panel(to8(img), x, pw, n) for img, x in zip(imgs, texts)]))
        if not a.no_stretch:
            cv2.imwrite(os.path.join(a.out, name + ".stretch.png"),
                        np.hstack([panel(to8(stretch(img, lo, hi)), x, pw, n) for img, x in zip(imgs, texts)]))
        score = "" if val is None else f"{val:.3f}"
        rows.append([name, kind, str(i), str(xx), str(yy), score] + [f"{d:.3f}" for d in des[1:]])
        print(f"{name}: {kind}{' ' + score if score else ''}; ΔE00 to the GT "
              + ", ".join(f"{s} {d:.2f}" for s, d in zip(labels, des[1:])))
    with open(os.path.join(a.out, "crops.tsv"), "a", encoding="utf-8") as f:
        f.write("\t".join(["name", "kind", "frame", "x", "y", "score"] + labels) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")


if __name__ == "__main__":
    main()

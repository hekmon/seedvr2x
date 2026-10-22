#!/usr/bin/env python3
"""A blind set at 4K for --quality: standard against maximum, per shot, with recognisable windows.

Stills: 2 places per shot, frames 4-44 (frames 0-3 of maximum are the shifted runs' own first latents), the 384x384
window (32-px grid) where the two differ most (mean |dY'|); the second place on a frame 8+ away. A window must hold
something recognisable: at least REC (10%) of the GT window's pixels with a luma gradient of 4 levels/px or more
(Sobel 3x3 / 8, 8-bit scale; set_detail_calib.py on session 9's places: of the 9 flat-sky places, the 7 the eyes
couldn't judge among them, 8 had under 1%, the ninth 10.2%; a judged dark place 0%; every judged 4K place 26.8% or
more: a conservative rule, not a line between the two). Where the largest difference sits on a flat window, the
nearest recognisable window on the same frame is taken, and the set's notes say so. Frames with no recognisable
window (motion blur all over) are passed over for the first place while other frames have one, and for the second
unless no frame 8+ away has one; on such a frame, or for a loop whose shot has none, the nearest of the 10% most
detailed windows is taken, noted too. A and B at 1:1 and x3 nearest, GT apart.
Loops: per shot the 1920x1080 window (64-px grid) of largest summed difference over the shot, or, where it is flat
by the GT's mean structure over the shot, the nearest recognisable one (as for the places); A | B side by side
(3840x1080), the 45 frames three times, x264 CRF 12, 8-bit 4:2:0.
A/B counterbalanced (a shot's two places on opposite sides, the shots half each way; loops half each way), drawn
with os.urandom; PNGs from cv2 (no text chunks), MP4 metadata stripped, bitexact.

  quality_set.py DATA GTDIR OUT KEYDIR SHOT:SEED...      (DATA/<shot>-{standard,maximum}/seg_*.mkv, one segment each)
"""
import glob, os, random, subprocess, sys
import numpy as np
import cv2

DATA, GTD, OUT, KEY = sys.argv[1:5]
SHOTS = [a.split(":") for a in sys.argv[5:]]
S, STEP, REC = 384, 32, 0.10
RNG = random.Random(int.from_bytes(os.urandom(8), "big"))
Z8 = "zscale=threads=1:filter=point:dither=none:min=gbr:m=gbr:rin=full:r=full:pin=709:p=709:tin=709:t=709,format=gbrp"
ZY = "zscale=threads=1:dither=none:min=gbr:m=709:rin=full:r=limited:pin=709:p=709:tin=709:t=709:c=left,format=yuv420p"


def probe(p, what):
    return subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", f"stream={what}", "-of",
                           "csv=p=0", p], capture_output=True, text=True, check=True).stdout.strip()


def lumas(p, w, h):
    """Every frame's BT.709 luma (8-bit scale, float32), streamed."""
    pr = subprocess.Popen(["ffmpeg", "-v", "error", "-nostdin", "-threads", "8", "-i", p, "-f", "rawvideo", "-pix_fmt",
                           "gbrp16le", "-"], stdout=subprocess.PIPE)
    out, n = [], w * h * 6
    while True:
        b = pr.stdout.read(n)
        if len(b) < n:
            break
        g, bb, r = np.frombuffer(b, np.uint16).reshape(3, h, w).astype(np.float32) * (255.0 / 65535.0)
        out.append(0.2126 * r + 0.7152 * g + 0.0722 * bb)
    assert pr.wait() == 0
    return np.stack(out)


def edges(y):
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    return (np.sqrt(gx * gx + gy * gy) / 8 >= 4).astype(np.float32)


def boxes(m, sw, sh, step):
    """{(x, y): mean of m over the sw x sh window} on a step grid."""
    ii = cv2.integral(m.astype(np.float64)); h, w = m.shape
    ys = np.arange(0, h - sh + 1, step); xs = np.arange(0, w - sw + 1, step)
    v = (ii[ys[:, None] + sh, xs[None, :] + sw] - ii[ys[:, None], xs[None, :] + sw] - ii[ys[:, None] + sh, xs[None, :]]
         + ii[ys[:, None], xs[None, :]]) / (sw * sh)
    return xs, ys, v


def pick(dm, em, sw, sh, step):
    """(diff, x, y, rec, moved_from): the window of largest diff; if flat, the nearest recognisable one."""
    xs, ys, v = boxes(dm, sw, sh, step); _, _, r = boxes(em, sw, sh, step)
    j, i = np.unravel_index(np.argmax(v), v.shape)
    if r[j, i] >= REC:
        return float(v[j, i]), int(xs[i]), int(ys[j]), float(r[j, i]), None
    ok = np.argwhere(r >= REC)
    tag = (int(xs[i]), int(ys[j]), float(v[j, i]), float(r[j, i]))
    if len(ok) == 0:
        # motion blur all over (no window reaches REC): the frame's 10% of windows with the most detail
        ok = np.argwhere(r >= np.quantile(r, 0.9)); tag = ("blur",) + tag
        if r[j, i] >= np.quantile(r, 0.9):
            return float(v[j, i]), int(xs[i]), int(ys[j]), float(r[j, i]), ("blur-kept",) + tag[1:]
    d2 = (ok[:, 0] - j) ** 2 + (ok[:, 1] - i) ** 2
    jj, ii = ok[np.argmin(d2)]
    return float(v[jj, ii]), int(xs[ii]), int(ys[jj]), float(r[jj, ii]), tag


def f8(p, t, w, h):
    b = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-threads", "4", "-i", p, "-vf", f"select=eq(n\\,{t})," + Z8,
                        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gbrp", "-"], capture_output=True, check=True).stdout
    g, bb, r = np.frombuffer(b, np.uint8).reshape(3, h, w)
    return np.dstack([bb, g, r])


def note(m, where):
    if m[0] == "blur-kept":
        return (f"motion blur all over {where} (no window as detailed as the other shots'); the largest difference "
                "sits among its most detailed windows and is kept.")
    if m[0] == "blur":
        return (f"motion blur all over {where} (no window as detailed as the other shots'); the largest difference "
                f"sits at x {m[1]} y {m[2]}, where nothing is recognisable; this is the nearest of its most detailed "
                "windows.")
    return (f"the largest difference {where} sits on a flat window (x {m[0]} y {m[1]}, nothing to recognise); this is "
            "the nearest window with detail.")


def balanced(n):
    f = [True] * (n // 2) + [False] * (n - n // 2); RNG.shuffle(f); return f


os.makedirs(OUT, exist_ok=False)
info = []
blurnotes = []
for sh, seed in SHOTS:
    seg = {}
    for q in ("standard", "maximum"):
        ss = sorted(glob.glob(f"{DATA}/{sh}-{q}/seg_*.mkv")); assert len(ss) == 1, (sh, q, ss); seg[q] = ss[0]
    gt = f"{GTD}/{sh}.gt.mkv"
    w, h = map(int, probe(seg["standard"], "width,height").split(","))
    assert probe(gt, "width,height") == f"{w},{h}", (sh, probe(gt, "width,height"), w, h)
    ya = lumas(seg["standard"], w, h); yb = lumas(seg["maximum"], w, h)
    assert len(ya) == len(yb) == 45, (sh, len(ya), len(yb))
    d = np.abs(ya - yb); del yb
    motion = float(np.mean(np.abs(np.diff(ya, axis=0)))); del ya
    yg = lumas(gt, w, h); assert len(yg) == 45
    e = np.stack([edges(f) for f in yg]); del yg
    per = [pick(d[t], e[t], S, S, STEP) + (t,) for t in range(4, 45)]
    # frames with no window reaching REC (motion blur all over) only when no other frame has one
    good = [p for p in per if not (p[4] and p[4][0] in ("blur", "blur-kept"))]
    skipped = len(per) - len(good)
    good = good or per
    p1 = max(good, key=lambda p: p[0])
    far = [p for p in good if abs(p[5] - p1[5]) >= 8] or [p for p in per if abs(p[5] - p1[5]) >= 8]
    p2 = max(far, key=lambda p: p[0])
    if skipped:
        blurnotes.append(f"{sh}: {skipped} of its frames 4-44 have no window with detail (motion blur all over); "
                         "its places are on the others.")
    lp = pick(d.sum(axis=0), e.mean(axis=0), 1920, 1080, 64)
    info.append((motion, sh, seed, seg, gt, w, h, p1, p2, lp))
    print(f"{sh} {w}x{h} seed {seed}: motion {motion:.2f}, mean |dY'| {d.mean():.3f}, places f{p1[5]} f{p2[5]}, "
          f"loop x{lp[1]} y{lp[2]} (rec {lp[3]:.2f})", flush=True)
    del d, e
info.sort(key=lambda r: -r[0])  # the most motion first
units = balanced(len(info)); lflags = balanced(len(info))
index, key, notes, n = [], [], list(blurnotes), 0
for k, (motion, sh, seed, seg, gt, w, h, p1, p2, lp) in enumerate(info):
    for f in (units[k], not units[k]):
        p = p1 if f == units[k] else p2
        v, x, y, rec, moved, t = p
        n += 1; o = ["standard", "maximum"] if f else ["maximum", "standard"]
        key.append(f"place {n:02d}: A = {o[0]}, B = {o[1]} (mean |dY'| {v:.2f}, structure {rec:.2f})")
        index.append(f"place {n:02d}: shot {sh}, frame {t}, window x {x} y {y}")
        if moved:
            notes.append(f"place {n:02d}: " + note(moved, f"on frame {t}"))
        for lab, pth in (("A", seg[o[0]]), ("B", seg[o[1]]), ("GT", gt)):
            img = np.ascontiguousarray(f8(pth, t, w, h)[y:y + S, x:x + S])
            cv2.imwrite(f"{OUT}/place{n:02d}_{lab}.png", img)
            cv2.imwrite(f"{OUT}/place{n:02d}_{lab}_x3.png", cv2.resize(img, (S * 3, S * 3), interpolation=cv2.INTER_NEAREST))
        print(index[-1], flush=True)
for i, (motion, sh, seed, seg, gt, w, h, p1, p2, lp) in enumerate(info, 1):
    _, lx, ly, rec, moved = lp
    o = ["standard", "maximum"] if lflags[i - 1] else ["maximum", "standard"]
    key.append(f"loop {i:02d}: left (A) = {o[0]}, right (B) = {o[1]}")
    index.append(f"loop {i:02d}: shot {sh}, frames 0-44, the 1920x1080 window at x {lx} y {ly}, three times, A | B")
    if moved:
        notes.append(f"loop {i:02d}: " + note(moved, "the shot"))
    fps = probe(seg[o[0]], "r_frame_rate"); num, den = fps.split("/")
    fc = (f"[0:v]crop=1920:1080:{lx}:{ly}[L];[1:v]crop=1920:1080:{lx}:{ly}[R];[L][R]hstack=inputs=2,"
          f"loop=loop=2:size=45:start=0,settb={den}/{num},setpts=N,fps={fps},{ZY}[v]")
    out = f"{OUT}/loop{i:02d}.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", seg[o[0]], "-i", seg[o[1]], "-filter_complex", fc,
                    "-map", "[v]", "-c:v", "libx264", "-preset", "slow", "-crf", "12", "-threads", "16",
                    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
                    "-map_metadata", "-1", "-metadata", "title=", "-fflags", "+bitexact", "-flags:v", "+bitexact", out],
                   check=True)
    cnt = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_packets", "-of", "csv=p=0", out], capture_output=True, text=True).stdout.strip()
    assert cnt == "135", (out, cnt)
    print(index[-1], flush=True)
seeds = ", ".join(f"{sh} {seed}" for _, sh, seed, *_ in info)
hdr = f"""4K. seedvr2x, 4K from half-size input (x2). A and B are two settings of the same run (the key says which).
Seeds: {seeds}.
Each place: the same frame and 384x384 window of the 4K frame, at 1:1 and x3 (nearest); GT = the original,
apart. Every window holds something to recognise (where the largest difference was on a flat area, the nearest
window with detail was taken: see the notes at the end).
For each place and loop, answer one of:
  A or B   = that one looks better (a word on why helps: sharper, cleaner, more natural...);
  same     = you see no difference;
  ?        = they differ but neither is better (e.g. one sharper, the other smoother); say how if you can.
Loops: a 1920x1080 window of the 4K frame, A left, B right, the 45 frames three times (VLC: loop on)."""
os.makedirs(KEY, exist_ok=True)
with open(f"{OUT}/index.txt", "w") as f:
    f.write(hdr + "\n\n" + "\n".join(index) + "\n\nnotes:\n" + ("\n".join(notes) if notes else "none: every largest "
            "difference sat on a window with detail.") + "\n")
with open(f"{KEY}/key.txt", "w") as f:
    f.write("\n".join(key) + "\n")
print("done", flush=True)

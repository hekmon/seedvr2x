#!/usr/bin/env python3
"""Chroma downsampling kernels for the default yuv420p10le master, scored against the ground truth.

  chroma_kernels.py siting --json OUT.json [--kernels K,...]            # synthetic siting and step test
  chroma_kernels.py threads --src SRC.mkv --json OUT.json [--kernels bilinear,lanczos] [--threads 2,4,...]
  chroma_kernels.py gtcheck --gt GT.mkv --src SRC.mkv --json OUT.json [--threads 1,16,24,48]
  chroma_kernels.py upcheck --yuv MASTER.yuv.mkv --json OUT.json [--threads 1,2,3,4,8,16]
  chroma_kernels.py convert --clip NAME --model MODEL.mkv --gt GT.mkv --dir DIR [--kernels K,...]
                    [--threads 4] [--filter-threads 1]
  taskset -c CPU chroma_kernels.py proof --src SRC.mkv --yuv DIR/NAME/TAG.bilinear.yuv.mkv --work WORK
                    --json OUT.json
  chroma_kernels.py score --clip NAME --model MODEL.mkv --gt GT.mkv --dir DIR --json OUT.json
                    [--frames N] [--threads 1]
  chroma_kernels.py srcchroma --clip NAME --src SRC.mkv --dir DIR --json OUT.json
  chroma_kernels.py review --clip NAME --model MODEL.mkv --gt GT.mkv --dir DIR --out REVIEW_DIR
                    [--crops 3] [--win 96] [--zoom 4]
  chroma_kernels.py timing --src SRC.mkv --json OUT.json [--frames 24] [--loops 10] [--threads 1,8]
                    [--rounds 3]
  chroma_kernels.py summary SCORE.json ... [--frm FRM_DIR] [--q1 Q1_JSON_DIR] [--proof P.json ...]
                    [--siting S.json] [--threads-json T.json] [--gtcheck G.json ...] [--upcheck U.json ...]
                    [--srcchroma C.json ...] [--timing T.json] [--md OUT.md]

Why: seedvr2x's default master is FFV1 yuv420p10le, converted from the 16-bit RGB frames exactly
as ffv1_out.py does it (zscale: BT.709 matrix, limited range, chroma sited left, no dither, zscale's
default chroma kernel, bilinear; DESIGN.md, Output). Every default output goes through that chroma
downsampling. This measures what it costs and whether another kernel does better: each RGB master
goes to yuv420p10le with each kernel, comes back to 16-bit RGB with one fixed upsampler, and the
round trip is scored against the ground truth of the full-reference clips (fr_clips.py).

Kernels (zscale's filter for the chroma downsampling; luma is never resampled):
  bilinear  zscale's default: ffv1_out.py's chain as it is (its FORMATS["yuv420p10le"] options)
  catrom    bicubic b = 0, c = 0.5 (f=bicubic:param_a=0:param_b=0.5)
  spline16, spline36
  lanczos   3 taps (f=lanczos:param_a=3)
  point     nearest sample, the aliasing control
  yuv444    control: ffv1_out.py's yuv444p10le chain, the 10-bit quantisation alone, no subsampling
zimg widens the kernel by the ratio when it downsamples (a low-pass, not an interpolator).

The upsampler, fixed, is seedvr2x's reader (media/conversion.py, Conversion.filters() for a
yuv420p10le BT.709 limited-range source sited left): Catmull-Rom, then the RGB tags set:
  format=yuv420p10le,zscale=min=709:rin=limited:cin=left:pin=unspecified:p=unspecified:
  tin=unspecified:t=unspecified:m=gbr:r=full:d=none:f=bicubic:param_a=0:param_b=0.5,format=gbrp16le
No dither anywhere (d=none: rounded to nearest), as ffv1_out.py.

zscale's slice threading changes its output. ffmpeg runs zscale in -filter_threads slices, by
default as many as the CPUs the process may use (av_cpu_count, which honours the affinity), and
each slice is a separate zimg graph on its own rows, so the vertical chroma filter near a slice
boundary sees the slice's edge instead of the next rows: every slice count gives other bytes.
`threads` measures it (each count against 1, down and up, which planes and rows differ, by how
much); `gtcheck` checks a GT clip against its source re-converted with fr_clips.py's chain;
`upcheck` compares zscale's Catmull-Rom read of a 4:2:0 source (yuv420p10le or yuv420p, frame 0)
with a float64 reference (Keys a = -0.5, sited left horizontally and centred vertically, edges
mirrored, border excluded), per -filter_threads, and with the same reference whose upsampled
chroma is rounded to the source's bit depth. The kernels are compared with --filter-threads 1
(whole frames, no seams) on both sides.

`convert` writes, for the model output (TAG model) and the GT itself (TAG gt, the format's cost on
perfect content), DIR/NAME/TAG.VARIANT.yuv.mkv (the master: FFV1, yuv420p10le or yuv444p10le,
ffv1_out.py's tags) and DIR/NAME/TAG.VARIANT.rgb.mkv (its round trip: FFV1 gbrp16le, the GT's tags).

`proof`: the bilinear master equals ffv1_out.py's own conversion of the same frames. The source is
decoded to raw gbrp16le planes and fed to ffv1_out.Writer (its exact command), and the two files'
framemd5 must agree frame for frame; zscale run with an explicit f=bilinear must agree too (the
default is bilinear), and so must the ffprobe tags. ffv1_out.py pins zscale to one thread since
2026-10-05; the proofs were run before, with ffv1_out.Writer at ffmpeg's default (one slice per CPU),
on one CPU (taskset -c N) to compare with a --filter-threads 1 master. Either way works now.

`siting`: synthetic 64x32 frames (grey, one coloured column at x = 32 or 33, one row at y = 16,
rows 16-17, columns 32-33, a step at x = 32, a step at y = 16), through every kernel and back.
With chroma sited left, chroma sample k sits on luma column 2k: a column at x = 32 must give 4:2:0
chroma symmetric around sample 16, one at x = 33 symmetric around 16.5. Sited at the centre
vertically, chroma row k sits at y = 2k + 0.5: rows 16-17 must give chroma symmetric around row 8,
one row at y = 16 an asymmetric one (it would be symmetric with top siting). The round trips must be
symmetric around the column (x = 32, 33) and around y = 16.5. The steps give each kernel's
overshoot (8-bit levels beyond the two plateaus) and its 10-90% rise (px) after the round trip.

`score` (all frames; float32 RGB decoded as fr_metrics.py does). Y', Cb, Cr: BT.709 from the
full-range RGB at full resolution, 8-bit scale (Y' in [0, 255], Cb and Cr in [-127.5, 127.5]):
- psnr_y, psnr_cb, psnr_cr: PSNR against the GT (peak 255), the mean of the per-frame PSNRs
- de00_s0, de00_s1, de00_s2, de00_s4: mean CIEDE2000 against the GT after a Gaussian blur of
  sigma 0 (none), 1, 2, 4 px, in CIELAB as fr_metrics.py computes it (OpenCV RGB -> Lab on float
  RGB, the blur on Lab, skimage's CIEDE2000). sigma 4 is fr_metrics.lab_lf itself (every second
  pixel), so de00_s4 of the unconverted output is fr_metrics' de00_lf; sigma 0, 1, 2 use every
  pixel (4:2:0 errors have a 2-pixel period that subsampling would alias)
- de00_edge: de00_s0 over the GT's chroma edges, the 10% of pixels with the largest chroma
  gradient (3x3 Sobel of Cb and Cr: sqrt(gx_cb^2 + gy_cb^2 + gx_cr^2 + gy_cr^2)), per frame
- ringing: e = how far Cb or Cr leaves the GT's local 5x5 [min, max] (8-bit levels, the larger
  of the two channels, 0 inside). ring_share: % of pixels with e > 2 (ring05_share: e > 0.5);
  ring_p999, ring_p9999: the 99.9th and 99.99th percentiles of e over every pixel of every frame
  (0.01-level histogram, pooled), ring_max its largest value. ring_src_*: the same against the
  round trip's own source (the unconverted model output): the kernel alone
- *_src: the round trip against its own source (psnr_y_src, psnr_cb_src, psnr_cr_src,
  de00_s0_src), and the constant shift: dy, dcb, dcr, the mean of round trip - source (levels),
  which a range or matrix error would move
The unconverted model output is scored as variant `none`. For TAG gt the source is the GT.

`srcchroma`: each master's 4:2:0 chroma planes against the source's own (fr_clips.py's src.mkv:
the clip's original 4:2:0 frames, sited left), sample for sample: PSNR-Cb, PSNR-Cr on the 8-bit
scale, and the mean shifts. No upsampling is involved, so neither the GT's own chroma upsampling
(fr_clips.py converted the source with zscale's default, bilinear) nor the reader's Catmull-Rom
weighs on it.

`review`: per clip, the --crops windows (--win px) with the strongest GT chroma gradient (window
mean, best frame, not overlapping), at --zoom x nearest neighbour, side by side: GT, the
unconverted model output, its bilinear, spline36 and lanczos round trips, each labelled with its
mean CIEDE2000 to the GT over the window. REVIEW_DIR/NAME.png, REVIEW_DIR/crops.tsv.

`timing`: zscale alone, per 1080p frame and kernel: --frames frames decoded once to raw gbrp16le in
--tmp (default /dev/shm), then `ffmpeg -f rawvideo -stream_loop LOOPS-1 ... -vf CHAIN -f null -` per
kernel (the frames read --loops times), decoder single-threaded, -filter_threads T, kernels
interleaved in each of --rounds rounds; ms/frame = median over rounds of (time - the same round's
copy pass) / frames, for the wall time and for the child's CPU time (user + system, getrusage: it
holds better than the wall time on a loaded box). Also the upsampler (yuv420p10le in, from the
bilinear chain).

`summary`: Markdown tables of the score JSONs: per clip and variant (model output, then GT round
trips), each kernel - bilinear paired by frame (mean, 95% CI from fr_metrics' moving-block
bootstrap over frames, blocks of 8; better / worse when the CI excludes 0), wins out of the clips,
the means over clips (all, anime, live action), the headroom (bilinear - yuv444) and the 10-bit
floor (yuv444 against its source), the constant shifts, the checks (proof, siting, sigma 4 =
fr_metrics' de00_lf), fr_metrics' standard set on the round-tripped model outputs (--frm: the
JSONs of one fr_metrics.py call per clip, variants src, bilinear, ... seed 42), and the timing.

fr_metrics on the round trips (one call per clip, the GT decoded once):
  fr_metrics.py GT.mkv --clip NAME --json-dir FRM --threads 16 --vmaf-threads 16 \\
    --out src 42 MODEL.mkv --out bilinear 42 DIR/NAME/model.bilinear.rgb.mkv ...  (every variant)

Needs the metrics venv (numpy, opencv, scikit-image; fr_metrics.py and ffv1_out.py next to this
file) and ffmpeg/ffprobe with zimg (n9.0.2 measured).
"""
import argparse
import json
import math
import os
import resource
import shlex
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from fractions import Fraction

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ffv1_out as O  # noqa: E402  the chain under test
import fr_metrics as F  # noqa: E402  decoding, luma, CIELAB and CIEDE2000 as fr_metrics computes them

FFMPEG, FFPROBE = F.FFMPEG, F.FFPROBE

# zscale's chroma downsampling filter, appended to ffv1_out.py's yuv420p10le chain
KERNELS = {
    "bilinear": "",  # zscale's default: ffv1_out.py's chain as it is
    "catrom": ":f=bicubic:param_a=0:param_b=0.5",
    "spline16": ":f=spline16",
    "spline36": ":f=spline36",
    "lanczos": ":f=lanczos:param_a=3",
    "point": ":f=point",
}
VARIANTS = list(KERNELS) + ["yuv444"]
# seedvr2x's reader for a yuv420p10le BT.709 limited-range source sited left (media/conversion.py)
UPSAMPLE = ("zscale=min=709:rin=limited:cin=left:pin=unspecified:p=unspecified:tin=unspecified:"
            "t=unspecified:m=gbr:r=full:d=none:f=bicubic:param_a=0:param_b=0.5")
FFV1 = ["-c:v", "ffv1", "-level", "3", "-g", "1", "-slices", "16", "-slicecrc", "1"]  # ffv1_out.Writer's
RING_T = 2.0      # ringing threshold, 8-bit levels
HMAX, NB = 128.0, 12800  # excursion histogram: 0.01-level bins
K5 = np.ones((5, 5), np.uint8)
TAGS_PROBE = "stream=pix_fmt,width,height,color_space,color_range,color_primaries,color_transfer,chroma_location"


def log(msg):
    print(f"[chroma_kernels {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def kernels_arg(text):
    ks = [k.strip() for k in text.split(",") if k.strip()] if text else list(VARIANTS)
    bad = [k for k in ks if k not in VARIANTS]
    if bad:
        raise SystemExit(f"unknown kernel(s) {', '.join(bad)} (choose from {', '.join(VARIANTS)})")
    return ks


def down(variant):
    """(pix_fmt, ffmpeg output options) of the RGB -> YUV step: ffv1_out.py's own for bilinear and yuv444."""
    if variant == "yuv444":
        return "yuv444p10le", list(O.FORMATS["yuv444p10le"][1])
    opts = list(O.FORMATS["yuv420p10le"][1])
    i = opts.index("-vf") + 1
    if opts[i] != f"{O.ZSCALE}:c=left,format=yuv420p10le":
        raise SystemExit(f"ffv1_out.py's yuv420p10le chain changed: {opts[i]}")
    opts[i] = f"{O.ZSCALE}:c=left{KERNELS[variant]},format=yuv420p10le"
    return "yuv420p10le", opts


def down_vf(variant):
    _, opts = down(variant)
    return opts[opts.index("-vf") + 1]


def up_vf(yuv_fmt):
    return f"format={yuv_fmt},{UPSAMPLE},format=gbrp16le,{O.SETPARAMS}"


def rt_path(d, clip, tag, variant, kind):
    return os.path.join(d, clip, f"{tag}.{variant}.{kind}.mkv")


def probe_tags(path):
    r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries", TAGS_PROBE,
                        "-of", "compact=p=0", path], capture_output=True, text=True)
    return r.stdout.strip()


def run(cmd, what):
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if r.returncode or r.stderr.strip():
        log(f"{what}: exit {r.returncode}: {r.stderr.strip()}\n  $ {shlex.join(cmd)}")
        if r.returncode:
            raise SystemExit(f"{what} failed")
    return dt


# ------------------------------------------------------------------ colour helpers

def ycc(rgb):
    """BT.709 Y', Cb, Cr of full-range float RGB, 8-bit scale (Y' as fr_metrics.luma)."""
    y = F.luma(rgb)
    cb = (rgb[..., 2] * np.float32(255) - y) * np.float32(1 / 1.8556)
    cr = (rgb[..., 0] * np.float32(255) - y) * np.float32(1 / 1.5748)
    return y, cb, cr


def mse(a, b):
    d = a - b
    return float(np.mean(d * d, dtype=np.float64))


def lab_of(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)


def blur(lab, s):
    return cv2.GaussianBlur(lab, (0, 0), s)


def de00(a, b):
    return F.deltaE_ciede2000(a, b)


# ------------------------------------------------------------------ siting

def cmd_siting(a):
    W, H = 64, 32
    grey, col = np.array([0.5, 0.5, 0.5]), np.array([0.3, 0.55, 0.75])  # luma ~0.51, Cb +33, Cr -34 levels
    shapes = {
        "col32": (slice(None), slice(32, 33)), "col33": (slice(None), slice(33, 34)),
        "row16": (slice(16, 17), slice(None)), "rows16-17": (slice(16, 18), slice(None)),
        "cols32-33": (slice(None), slice(32, 34)),
        "step-x32": (slice(None), slice(32, None)), "step-y16": (slice(16, None), slice(None)),
    }
    names = list(shapes)
    frames = []
    for n in names:
        x = np.tile(grey, (H, W, 1))
        x[shapes[n]] = col
        frames.append(x)
    q = np.rint(np.stack(frames) * 65535).astype("<u2")  # T, H, W, 3 (R, G, B)
    raw = np.ascontiguousarray(q.transpose(0, 3, 1, 2)[:, [1, 2, 0]]).tobytes()  # planar G, B, R
    base = [FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-threads", "1", "-filter_threads", "1",
            "-f", "rawvideo", "-s", f"{W}x{H}", "-framerate", "24"]

    def pipe(data, in_fmt, vf, out_fmt):
        cmd = base + ["-pix_fmt", in_fmt, "-i", "-", "-vf", vf, "-f", "rawvideo", "-pix_fmt", out_fmt, "-"]
        r = subprocess.run(cmd, input=data, capture_output=True)
        if r.returncode:
            raise SystemExit(f"$ {shlex.join(cmd)}\n{r.stderr.decode(errors='replace')}")
        return r.stdout

    res = {"frames": names, "colour": col.tolist(), "grey": grey.tolist(), "size": [W, H], "kernels": {}}
    T = len(names)
    for k in kernels_arg(a.kernels):
        if k == "yuv444":
            continue
        yuv = pipe(raw, "gbrp16le", down_vf(k), "yuv420p10le")
        rt = pipe(yuv, "yuv420p10le", up_vf("yuv420p10le"), "gbrp16le")
        fs = W * H + 2 * (W // 2) * (H // 2)
        yv = np.frombuffer(yuv, "<u2").reshape(T, fs)
        U = yv[:, W * H:W * H + (W // 2) * (H // 2)].reshape(T, H // 2, W // 2).astype(np.int64)  # Cb codes
        g, b, r = np.frombuffer(rt, "<u2").reshape(T, 3, H, W).transpose(1, 0, 2, 3).astype(np.float32) / 65535
        _, cb, cr = ycc(np.stack([r, g, b], axis=-1))
        f = {n: i for i, n in enumerate(names)}
        u_row = lambda n: U[f[n], H // 4]  # noqa: E731  a chroma row
        u_col = lambda n: U[f[n], :, W // 4]  # noqa: E731  a chroma column
        c_row = lambda n, c=cb: c[f[n], H // 2]  # noqa: E731
        c_col = lambda n, c=cb: c[f[n], :, W // 2]  # noqa: E731

        def sym(p, lo, hi, n):  # max |p[lo - d] - p[hi + d]|, d = 0..n-1
            return float(max(abs(float(p[lo - d]) - float(p[hi + d])) for d in range(n)))
        chk = {
            # 4:2:0 chroma codes (10-bit)
            "u_col32_sym16": sym(u_row("col32"), 15, 17, 6),     # left siting: symmetric around sample 16
            "u_col33_sym16.5": sym(u_row("col33"), 16, 17, 6),   # and around 16.5
            "u_rows16-17_sym8": sym(u_col("rows16-17"), 7, 9, 6),  # centre siting: symmetric around row 8
            "u_row16_asym": sym(u_col("row16"), 7, 9, 1),        # |U[7] - U[9]|: > 0 with centre siting
            "u_cols32-33_asym": sym(u_row("cols32-33"), 15, 17, 1),  # |U[15] - U[17]|: > 0 with left siting
            # round trips, Cb and Cr in 8-bit levels
            "rt_col32_sym32": max(sym(c_row("col32", c), 31, 33, 12) for c in (cb, cr)),
            "rt_col33_sym33": max(sym(c_row("col33", c), 32, 34, 12) for c in (cb, cr)),
            "rt_rows16-17_sym16.5": max(sym(c_col("rows16-17", c), 16, 17, 10) for c in (cb, cr)),
        }
        prof = {"u_col32": u_row("col32")[12:21].tolist(), "u_col33": u_row("col33")[12:22].tolist(),
                "u_rows16-17": u_col("rows16-17")[4:13].tolist(), "u_row16": u_col("row16")[4:13].tolist(),
                "rt_col32_cb": [round(float(v), 3) for v in c_row("col32")[26:39]],
                "rt_rows16-17_cb": [round(float(v), 3) for v in c_col("rows16-17")[10:24]]}
        steps = {}
        for n, prof_of in (("step-x32", c_row), ("step-y16", c_col)):
            for cname, c in (("cb", cb), ("cr", cr)):
                p = prof_of(n, c).astype(np.float64)
                a0, a1 = p[2], p[-3]  # the plateaus, far from the step
                lo, hi = min(a0, a1), max(a0, a1)
                over = float(max(p.max() - hi, lo - p.min(), 0))
                t = (p - a0) / (a1 - a0)  # 0 -> 1 across the step
                inner = slice(8, len(t) - 8)
                tt, xs = t[inner], np.arange(len(t))[inner]

                def cross(v):  # first crossing of v, linearly interpolated
                    i = int(np.argmax(tt >= v))
                    if i == 0:
                        return float(xs[0])
                    return float(xs[i - 1] + (v - tt[i - 1]) / (tt[i] - tt[i - 1]))
                steps[f"{n}_{cname}"] = {"overshoot": round(over, 3), "rise_10_90": round(cross(0.9) - cross(0.1), 3),
                                         "mid": round(cross(0.5), 3)}
        res["kernels"][k] = {"checks": chk, "profiles": prof, "steps": steps}
        log(f"{k}: " + ", ".join(f"{n} {v:g}" for n, v in chk.items()))
        log(f"{k}: steps " + ", ".join(f"{n} over {s['overshoot']:.2f} rise {s['rise_10_90']:.2f} mid {s['mid']:.2f}"
                                        for n, s in steps.items()))
    ok = all(v["checks"][c] <= 1 for v in res["kernels"].values() for c in
             ("u_col32_sym16", "u_col33_sym16.5", "u_rows16-17_sym8"))
    ok_rt = all(v["checks"][c] < 0.02 for v in res["kernels"].values() for c in
                ("rt_col32_sym32", "rt_col33_sym33", "rt_rows16-17_sym16.5"))
    asym = all(v["checks"]["u_row16_asym"] > 1 and v["checks"]["u_cols32-33_asym"] > 1
               for k, v in res["kernels"].items() if k != "point")
    res["verdict"] = {"chroma_symmetric": ok, "round_trip_symmetric": ok_rt, "controls_asymmetric": asym}
    print(f"siting: 4:2:0 chroma symmetric as left/centre siting predicts: {ok}; round trips symmetric: {ok_rt}; "
          f"controls (row 16, columns 32-33) asymmetric: {asym}")
    with open(a.json, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


# ------------------------------------------------------------------ convert

def cmd_convert(a):
    T = str(a.threads)
    os.makedirs(os.path.join(a.dir, a.clip), exist_ok=True)
    for tag, src in (("model", a.model), ("gt", a.gt)):
        for v in kernels_arg(a.kernels):
            fmt, opts = down(v)
            yuv, rgb = rt_path(a.dir, a.clip, tag, v, "yuv"), rt_path(a.dir, a.clip, tag, v, "rgb")
            pre = [FFMPEG, "-hide_banner", "-nostdin", "-nostats", "-loglevel", "error", "-y",
                   "-filter_threads", str(a.filter_threads), "-threads", T]
            cmd = pre + ["-i", src, "-map", "0:v:0", "-fps_mode", "passthrough", *opts, *FFV1, "-threads", T,
                         "-pix_fmt", fmt, yuv]
            t_down = run(cmd, f"{a.clip} {tag} {v} down")
            cmd_up = pre + ["-i", yuv, "-map", "0:v:0", "-fps_mode", "passthrough", "-vf", up_vf(fmt), *O.RGB_TAGS,
                            *FFV1, "-threads", T, "-pix_fmt", "gbrp16le", rgb]
            t_up = run(cmd_up, f"{a.clip} {tag} {v} up")
            log(f"{a.clip} {tag} {v}: {yuv} {os.path.getsize(yuv) / 2**20:.1f} MiB ({t_down:.1f} s), "
                f"{rgb} {os.path.getsize(rgb) / 2**20:.1f} MiB ({t_up:.1f} s); {probe_tags(yuv)}")
            if a.verbose:
                log(f"  $ {shlex.join(cmd)}\n  $ {shlex.join(cmd_up)}")


# ------------------------------------------------------------------ threads

def planes420(buf, T, W, H):
    """Raw yuv420p10le frames -> Y [T, H, W], Cb, Cr [T, H/2, W/2] as int32."""
    c = (W // 2) * (H // 2)
    v = np.frombuffer(buf, "<u2").reshape(T, W * H + 2 * c).astype(np.int32)
    return v[:, :W * H].reshape(T, H, W), v[:, W * H:W * H + c].reshape(T, H // 2, W // 2), \
        v[:, W * H + c:].reshape(T, H // 2, W // 2)


def diff_stats(d):
    """|difference| [T, h, w]: samples that differ, their share, the max, the RMS over all samples, and the rows
    of the first frame that hold a difference."""
    n = int(np.count_nonzero(d))
    return {"samples": n, "share": n / d.size, "max": int(d.max()),
            "rms": float(np.sqrt(np.mean(d.astype(np.float64) ** 2))), "rows": np.nonzero(d[0].max(axis=1))[0].tolist()}


def cmd_threads(a):
    st = F.probe(a.src)
    W, H = st["width"], st["height"]
    r = subprocess.run([FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-threads", "4", "-i", a.src,
                        "-map", "0:v:0", "-frames:v", str(a.frames), "-f", "rawvideo", "-pix_fmt", "gbrp16le", "-"],
                       capture_output=True)
    if r.returncode:
        raise SystemExit(r.stderr.decode(errors="replace"))
    raw = r.stdout
    T = len(raw) // (3 * W * H * 2)

    def conv(data, in_fmt, vf, out_fmt, ft):
        cmd = [FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-threads", "1", "-filter_threads", str(ft),
               "-f", "rawvideo", "-pix_fmt", in_fmt, "-s", f"{W}x{H}", "-framerate", "24", "-i", "-", "-vf", vf,
               "-f", "rawvideo", "-pix_fmt", out_fmt, "-"]
        p = subprocess.run(cmd, input=data, capture_output=True)
        if p.returncode:
            raise SystemExit(f"$ {shlex.join(cmd)}\n{p.stderr.decode(errors='replace')}")
        return p.stdout

    res = {"src": a.src, "frames": T, "size": [W, H], "kernels": {}}
    rgb_of = lambda b: np.frombuffer(b, "<u2").reshape(T, 3, H, W).astype(np.int32)  # noqa: E731
    for k in kernels_arg(a.kernels):
        y1 = conv(raw, "gbrp16le", down_vf(k), "yuv420p10le", 1)
        p1 = planes420(y1, T, W, H)
        r1 = rgb_of(conv(y1, "yuv420p10le", up_vf("yuv420p10le"), "gbrp16le", 1))
        out = {}
        for ft in [int(x) for x in a.threads.split(",")]:
            pt = planes420(conv(raw, "gbrp16le", down_vf(k), "yuv420p10le", ft), T, W, H)
            dn = {name: diff_stats(np.abs(x - y)) for name, x, y in zip(("Y", "Cb", "Cr"), p1, pt)}
            up = diff_stats(np.abs(r1 - rgb_of(conv(y1, "yuv420p10le", up_vf("yuv420p10le"), "gbrp16le", ft))).max(axis=1))
            out[str(ft)] = {"down": dn, "up": up}
            print(f"{k}, filter_threads {ft} vs 1: down (10-bit codes) " + "; ".join(
                f"{nm} {s['samples']} samples ({s['share']:.3%}) max {s['max']} rms {s['rms']:.4f}"
                f"{' rows ' + str(s['rows'][:16]) + ('…' if len(s['rows']) > 16 else '') if s['samples'] else ''}"
                for nm, s in dn.items()))
            print(f"{k}, filter_threads {ft} vs 1: up (RGB 16-bit codes, the single-threaded master read back) "
                  f"{up['samples']} samples ({up['share']:.3%}) max {up['max']} rms {up['rms']:.3f}"
                  f"{' rows ' + str(up['rows'][:16]) + ('…' if len(up['rows']) > 16 else '') if up['samples'] else ''}")
        res["kernels"][k] = out
    with open(a.json, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


def catrom_up(c, axis, phase):
    """Catmull-Rom 2x upsampling of chroma c along axis (float64, edges mirrored): phase 0 = sited on the
    even output samples (left, horizontally), 0.25 = centred (output 2m sits at m - 0.25, 2m + 1 at m + 0.25)."""
    c = np.moveaxis(c, axis, 0)
    n = c.shape[0]
    pad = np.concatenate([c[2:0:-1], c, c[-2:-4:-1]])  # mirror 2 samples on each side
    out = np.empty((2 * n,) + c.shape[1:])

    def interp(i0, t):  # samples i0-1..i0+2 (pad index + 2), fraction t
        w = [(-t ** 3 + 2 * t ** 2 - t) / 2, (3 * t ** 3 - 5 * t ** 2 + 2) / 2, (-3 * t ** 3 + 4 * t ** 2 + t) / 2,
             (t ** 3 - t ** 2) / 2]
        return sum(wk * pad[i0 + 1 + k:i0 + 1 + k + n] for k, wk in enumerate(w))
    if phase == 0:
        out[0::2] = c
        out[1::2] = interp(0, 0.5)
    else:
        out[0::2] = interp(-1, 0.75)  # m - 0.25: floor m - 1, t = 0.75
        out[1::2] = interp(0, 0.25)   # m + 0.25: floor m, t = 0.25
    return np.moveaxis(out, 0, axis)


def cmd_upcheck(a):
    """zscale's Catmull-Rom read of a 4:2:0 BT.709 limited-range source sited left (yuv420p10le or yuv420p)
    against a float64 reference, per -filter_threads."""
    st = F.probe(a.yuv)
    W, H, fmt = st["width"], st["height"], st["pix_fmt"]
    if fmt not in ("yuv420p10le", "yuv420p"):
        raise SystemExit(f"{a.yuv}: {fmt}, yuv420p10le or yuv420p expected")
    s = 4 if fmt == "yuv420p10le" else 1  # code scale against 8 bits
    dec = [FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-threads", "1"]
    r = subprocess.run(dec + ["-i", a.yuv, "-map", "0:v:0", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", fmt, "-"],
                       capture_output=True)
    buf = r.stdout if s == 4 else np.frombuffer(r.stdout, np.uint8).astype("<u2").tobytes()
    Y, U, V = (p[0].astype(np.float64) for p in planes420(buf, 1, W, H))
    y = (Y - 16 * s) / (219 * s)
    refs = {}
    for name, q in (("float", False), (f"chroma rounded to {8 if s == 1 else 10} bits", True)):
        cb, cr = (catrom_up(catrom_up((p - 128 * s) / (224 * s), 1, 0), 0, 0.25) for p in (U, V))
        if q:
            cb, cr = np.rint(cb * 224 * s) / (224 * s), np.rint(cr * 224 * s) / (224 * s)
        R = y + 1.5748 * cr
        B = y + 1.8556 * cb
        G = (y - 0.2126 * R - 0.0722 * B) / 0.7152
        refs[name] = np.stack([np.rint(np.clip(x, 0, 1) * 65535) for x in (G, B, R)])  # planar G, B, R
    m = a.margin
    inner = (slice(None), slice(m, H - m), slice(m, W - m))
    res = {"yuv": a.yuv, "pix_fmt": fmt, "margin": m, "threads": {}}
    for ft in [int(x) for x in a.threads.split(",")]:
        p = subprocess.run(dec + ["-filter_threads", str(ft), "-i", a.yuv, "-map", "0:v:0", "-frames:v", "1", "-vf",
                                  up_vf(fmt), "-f", "rawvideo", "-pix_fmt", "gbrp16le", "-"], capture_output=True)
        z = np.frombuffer(p.stdout, "<u2").reshape(3, H, W).astype(np.float64)
        out = {}
        for name, ref in refs.items():
            d = np.abs(z - ref)[inner]
            out[name] = {"max": float(d.max()), "rms": float(np.sqrt(np.mean(d ** 2))),
                         "share_over_2": float(np.mean(d > 2))}
        res["threads"][str(ft)] = out
        print(f"filter_threads {ft}: zscale vs float64 Catmull-Rom (16-bit codes, {m}-px border excluded): " + "; ".join(
            f"{n}: max {s['max']:.0f}, rms {s['rms']:.2f}, > 2 codes {s['share_over_2']:.2%}" for n, s in out.items()))
    with open(a.json, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


def chroma_planes(path, threads=2):
    """Yield the Cb, Cr planes of a 4:2:0 file (yuv420p10le or yuv420p) as float64 on the 8-bit scale, centred
    (Cb, Cr in [-127.5, 127.5]: (code - 128 s) / (224 s) * 255)."""
    st = F.probe(path)
    W, H, fmt = st["width"], st["height"], st["pix_fmt"]
    s = {"yuv420p10le": 4, "yuv420p": 1}[fmt]
    dt = np.dtype("<u2") if s == 4 else np.dtype(np.uint8)
    c = (W // 2) * (H // 2)
    size = (W * H + 2 * c) * dt.itemsize
    p = subprocess.Popen([FFMPEG, "-v", "error", "-nostdin", "-threads", str(threads), "-i", path, "-map", "0:v:0",
                          "-f", "rawvideo", "-pix_fmt", fmt, "-"], stdout=subprocess.PIPE)
    try:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            v = np.frombuffer(buf, dt).astype(np.float64)
            yield tuple((v[W * H + i * c:W * H + (i + 1) * c].reshape(H // 2, W // 2) - 128 * s) / (224 * s) * 255
                        for i in (0, 1))
    finally:
        p.kill()
        p.stdout.close()
        p.wait()


def cmd_srcchroma(a):
    """Each kernel's 4:2:0 chroma against the source's own 4:2:0 chroma (fr_clips.py's src.mkv)."""
    res = {"clip": a.clip, "src": a.src, "series": {}, "means": {}}
    for tag in ("model", "gt"):
        for v in [k for k in kernels_arg(a.kernels) if k != "yuv444"]:
            s = defaultdict(list)
            for (cb, cr), (sb, sr) in zip(chroma_planes(rt_path(a.dir, a.clip, tag, v, "yuv")), chroma_planes(a.src)):
                s["psnr_cb"].append(F.psnr(float(np.mean((cb - sb) ** 2))))
                s["psnr_cr"].append(F.psnr(float(np.mean((cr - sr) ** 2))))
                s["dcb"].append(float(np.mean(cb - sb)))
                s["dcr"].append(float(np.mean(cr - sr)))
            res["series"].setdefault(tag, {})[v] = s
            res["means"].setdefault(tag, {})[v] = {k: float(np.mean(x)) for k, x in s.items()}
            m = res["means"][tag][v]
            print(f"{a.clip} {tag} {v}: 4:2:0 chroma vs the source's, {len(s['psnr_cb'])} frames: PSNR-Cb "
                  f"{m['psnr_cb']:.2f}, PSNR-Cr {m['psnr_cr']:.2f}, mean ΔCb {m['dcb']:+.3f}, ΔCr {m['dcr']:+.3f}")
    with open(a.json, "w", encoding="utf-8") as fh:
        json.dump(res, fh)


def cmd_gtcheck(a):
    """A GT against its YUV source re-converted by fr_clips.py's chain, with each -filter_threads."""
    st = F.probe(a.gt)
    W, H = st["width"], st["height"]
    dec = [FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-threads", "1"]
    tail = ["-map", "0:v:0", "-frames:v", str(a.frames), "-f", "rawvideo", "-pix_fmt", "gbrp16le", "-"]
    r = subprocess.run(dec + ["-i", a.gt] + tail, capture_output=True)
    if r.returncode:
        raise SystemExit(r.stderr.decode(errors="replace"))
    T = len(r.stdout) // (3 * W * H * 2)
    gt = np.frombuffer(r.stdout, "<u2").reshape(T, 3, H, W).astype(np.int32)
    res = {"gt": a.gt, "src": a.src, "vf": a.vf, "frames": T, "threads": {}}
    for ft in [int(x) for x in a.threads.split(",")]:
        cmd = dec + ["-filter_threads", str(ft), "-i", a.src, "-vf", f"{a.vf},format=gbrp16le"] + tail
        p = subprocess.run(cmd, capture_output=True)
        if p.returncode:
            raise SystemExit(f"$ {shlex.join(cmd)}\n{p.stderr.decode(errors='replace')}")
        s = diff_stats(np.abs(gt - np.frombuffer(p.stdout, "<u2").reshape(T, 3, H, W).astype(np.int32)).max(axis=1))
        res["threads"][str(ft)] = s
        print(f"GT vs its source re-converted with -filter_threads {ft}: {s['samples']} samples differ ({s['share']:.3%}), "
              f"max {s['max']} (16-bit codes), rms {s['rms']:.3f}"
              + (f", rows {s['rows'][:24]}{'…' if len(s['rows']) > 24 else ''}" if s['samples'] else ""))
    with open(a.json, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


# ------------------------------------------------------------------ proof

def framemd5(path=None, cmd=None, threads=4):
    if cmd is None:
        cmd = [FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-threads", str(threads), "-i", path,
               "-map", "0:v:0", "-f", "framemd5", "-"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"framemd5 failed: {r.stderr}\n$ {shlex.join(cmd)}")
    return [ln.split(",")[-1].strip() for ln in r.stdout.splitlines() if ln.strip() and not ln.startswith("#")]


def cmd_proof(a):
    st = F.probe(a.src)
    W, H = st["width"], st["height"]
    fps = Fraction(st.get("r_frame_rate") or "24000/1001")
    ncpu = len(os.sched_getaffinity(0))  # ffmpeg's default -filter_threads (av_cpu_count honours the affinity)
    pinned = "threads=1" in O.ZSCALE  # ffv1_out.py pins zscale to one slice since 2026-10-05
    slices = 1 if pinned else ncpu
    if slices != a.filter_threads:
        log(f"warning: ffv1_out.Writer's zscale runs {slices} slices here, and --yuv was made with "
            f"{a.filter_threads}: run this under `taskset -c CPU` (one CPU) to compare like with like")
    os.makedirs(a.work, exist_ok=True)
    ref = os.path.join(a.work, os.path.basename(a.src).rsplit(".mkv", 1)[0] + ".ffv1_out.yuv420p10le.mkv")
    w = O.Writer(ref, "yuv420p10le", W, H, fps, 16)
    dec = subprocess.Popen([FFMPEG, "-v", "error", "-nostdin", "-threads", str(a.threads), "-i", a.src, "-map", "0:v:0",
                            "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "gbrp16le", "-"],
                           stdout=subprocess.PIPE)
    size = 3 * W * H * 2
    n = 0
    while True:
        buf = dec.stdout.read(size)
        if len(buf) < size:
            break
        w.write(np.frombuffer(buf, "<u2"))  # already planar G, B, R
        n += 1
    dec.wait()
    rc = w.close()
    if rc:
        raise SystemExit("ffv1_out.Writer failed")
    log(f"ffv1_out.Writer: {n} frames -> {ref}\n  $ {shlex.join(w.cmd)}")
    h_ref, h_mine = framemd5(ref, threads=a.threads), framemd5(a.yuv, threads=a.threads)
    explicit = [FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-threads", str(a.threads), "-i", a.src,
                "-map", "0:v:0", "-fps_mode", "passthrough", "-vf",
                f"{O.ZSCALE}:c=left:f=bilinear,format=yuv420p10le", "-f", "framemd5", "-"]
    h_exp = framemd5(cmd=explicit)
    same = sum(x == y for x, y in zip(h_ref, h_mine))
    same_exp = sum(x == y for x, y in zip(h_ref, h_exp))
    t_ref, t_mine = probe_tags(ref), probe_tags(a.yuv)
    res = {"src": a.src, "ffv1_out": ref, "mine": a.yuv, "cpus": ncpu, "ffv1_out_slices": slices,
           "mine_filter_threads": a.filter_threads,
           "frames": [len(h_ref), len(h_mine), len(h_exp)],
           "identical": same, "explicit_bilinear_identical": same_exp, "tags_ffv1_out": t_ref, "tags_mine": t_mine,
           "tags_equal": t_ref == t_mine, "writer_cmd": w.cmd, "explicit_cmd": explicit}
    ok = len(h_ref) == len(h_mine) == len(h_exp) == same == same_exp and t_ref == t_mine and n == len(h_ref)
    res["ok"] = ok
    print(f"proof {os.path.basename(a.src)} ({ncpu} CPU(s): ffv1_out.Writer's zscale in {slices} slice(s); mine "
          f"made with -filter_threads {a.filter_threads}): {same}/{len(h_ref)} frames identical to ffv1_out.Writer's "
          f"(framemd5), explicit f=bilinear {same_exp}/{len(h_exp)}, tags {'equal' if t_ref == t_mine else 'DIFFER'}"
          f" -> {'OK' if ok else 'FAILED'}")
    with open(a.json, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    if not a.keep:
        os.remove(ref)


# ------------------------------------------------------------------ score

class Ref:
    """Per-frame quantities of a reference: the GT (blurred Lab, chroma edges) or a round trip's source."""

    def __init__(self, rgb, gt=False):
        self.rgb = rgb
        self.y, self.cb, self.cr = ycc(rgb)
        self.lab = lab_of(rgb)
        self.lo_cb, self.hi_cb = cv2.erode(self.cb, K5), cv2.dilate(self.cb, K5)
        self.lo_cr, self.hi_cr = cv2.erode(self.cr, K5), cv2.dilate(self.cr, K5)
        if gt:
            self.lab1, self.lab2, self.lab4 = blur(self.lab, 1), blur(self.lab, 2), F.lab_lf(rgb)
            g2 = sum(cv2.Sobel(c, cv2.CV_32F, dx, dy, ksize=3) ** 2
                     for c in (self.cb, self.cr) for dx, dy in ((1, 0), (0, 1)))
            k = g2.size // 10
            self.edge_idx = np.argpartition(g2.ravel(), g2.size - k)[g2.size - k:]


def excursion(cb, cr, ref):
    e = np.maximum(np.maximum(cb - ref.hi_cb, ref.lo_cb - cb), np.maximum(cr - ref.hi_cr, ref.lo_cr - cr))
    return np.maximum(e, 0, out=e)


class Acc:
    def __init__(self):
        self.s = defaultdict(list)
        self.hist = defaultdict(lambda: np.zeros(NB, np.int64))
        self.npx = defaultdict(int)
        self.mx = defaultdict(float)

    def ring(self, key, e):
        h, _ = np.histogram(e, bins=NB, range=(0.0, HMAX))  # e > HMAX: counted in npx, above every percentile
        self.hist[key] += h
        self.npx[key] += e.size
        self.mx[key] = max(self.mx[key], float(e.max()))
        self.s[key + "_share"].append(100.0 * np.count_nonzero(e > RING_T) / e.size)
        self.s[key + "05_share"].append(100.0 * np.count_nonzero(e > 0.5) / e.size)

    def pct(self, key, q):
        cum = np.cumsum(self.hist[key])
        i = int(np.searchsorted(cum, q * self.npx[key]))
        return (i + 1) * HMAX / NB if i < NB else float("inf")

    def pooled(self):
        out = {}
        for k in self.hist:
            out.update({f"{k}_p999": self.pct(k, 0.999), f"{k}_p9999": self.pct(k, 0.9999), f"{k}_max": self.mx[k]})
        return out


def compare(acc, rgb, gt, src=None, same=False):
    """Scores of one frame against the GT (and against its own source)."""
    s = acc.s
    y, cb, cr = ycc(rgb)
    s["psnr_y"].append(F.psnr(mse(y, gt.y)))
    s["psnr_cb"].append(F.psnr(mse(cb, gt.cb)))
    s["psnr_cr"].append(F.psnr(mse(cr, gt.cr)))
    lab = lab_of(rgb)
    d0 = de00(lab, gt.lab)
    s["de00_s0"].append(float(d0.mean()))
    s["de00_edge"].append(float(d0.ravel()[gt.edge_idx].mean()))
    s["de00_s1"].append(float(de00(blur(lab, 1), gt.lab1).mean()))
    s["de00_s2"].append(float(de00(blur(lab, 2), gt.lab2).mean()))
    s["de00_s4"].append(float(de00(F.lab_lf(rgb), gt.lab4).mean()))
    acc.ring("ring", excursion(cb, cr, gt))
    if src is None:
        return
    if same:  # the source is the GT
        for k in ("psnr_y", "psnr_cb", "psnr_cr", "de00_s0"):
            s[k + "_src"].append(s[k][-1])
        acc.ring("ring_src", excursion(cb, cr, gt))
    else:
        s["psnr_y_src"].append(F.psnr(mse(y, src.y)))
        s["psnr_cb_src"].append(F.psnr(mse(cb, src.cb)))
        s["psnr_cr_src"].append(F.psnr(mse(cr, src.cr)))
        s["de00_s0_src"].append(float(de00(lab, src.lab).mean()))
        acc.ring("ring_src", excursion(cb, cr, src))
    s["dy"].append(float(np.mean(y - src.y, dtype=np.float64)))
    s["dcb"].append(float(np.mean(cb - src.cb, dtype=np.float64)))
    s["dcr"].append(float(np.mean(cr - src.cr, dtype=np.float64)))


def cmd_score(a):
    cv2.setNumThreads(a.threads)
    variants = kernels_arg(a.kernels)
    streams = {("model", "none"): a.model}
    for tag in ("model", "gt"):
        for v in variants:
            streams[(tag, v)] = rt_path(a.dir, a.clip, tag, v, "rgb")
    gsrc = F.Source(a.gt, threads=a.threads)
    srcs = {k: F.Source(p, threads=a.threads) for k, p in streams.items()}
    for k, s in srcs.items():
        if (s.W, s.H) != (gsrc.W, gsrc.H):
            raise SystemExit(f"{s.path}: {s.W}x{s.H}, the GT is {gsrc.W}x{gsrc.H}")
    accs = {k: Acc() for k in streams}
    n, t0 = 0, time.perf_counter()
    while not (a.frames and n >= a.frames):
        g = gsrc.read()
        fr = {k: s.read() for k, s in srcs.items()}
        ended = [k for k, f in fr.items() if f is None]
        if g is None or ended:
            if g is not None or len(ended) != len(fr):
                raise SystemExit(f"frame counts differ at frame {n}: GT {'ended' if g is None else 'goes on'}, "
                                 f"ended: {', '.join('.'.join(k) for k in ended) or 'none'}")
            break
        gt = Ref(g, gt=True)
        m = Ref(fr[("model", "none")])
        compare(accs[("model", "none")], fr[("model", "none")], gt)
        for v in variants:
            compare(accs[("model", v)], fr[("model", v)], gt, src=m)
            compare(accs[("gt", v)], fr[("gt", v)], gt, src=gt, same=True)
        n += 1
        el = time.perf_counter() - t0
        log(f"{a.clip}: frame {n} ({el / n:.1f} s/frame)")
    for s in [gsrc, *srcs.values()]:
        s.close()
    res = {"clip": a.clip, "gt": os.path.abspath(a.gt), "model": os.path.abspath(a.model), "dir": os.path.abspath(a.dir),
           "frames": n, "size": [gsrc.W, gsrc.H], "ring_threshold": RING_T,
           "chains": {"down": {v: down(v)[1] for v in variants}, "up": {f: up_vf(f) for f in ("yuv420p10le", "yuv444p10le")}},
           "series": defaultdict(dict), "pooled": defaultdict(dict), "means": defaultdict(dict),
           "seconds": round(time.perf_counter() - t0, 1)}
    for (tag, v), acc in accs.items():
        res["series"][tag][v] = {k: [round(x, 6) for x in vals] for k, vals in acc.s.items()}
        pooled = acc.pooled()
        res["pooled"][tag][v] = pooled
        res["means"][tag][v] = {**{k: float(np.mean(vals)) for k, vals in acc.s.items()}, **pooled}
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w", encoding="utf-8") as fh:
        json.dump(res, fh)
    for tag in ("model", "gt"):
        for v, mm in res["means"][tag].items():
            print(f"{a.clip} {tag} {v}: PSNR-Cb {mm['psnr_cb']:.2f} Cr {mm['psnr_cr']:.2f}, ΔE00 σ0 {mm['de00_s0']:.4f} "
                  f"σ1 {mm['de00_s1']:.4f} σ4 {mm['de00_s4']:.4f} edge {mm['de00_edge']:.4f}, ring "
                  f"{mm['ring_share']:.3f}% (>0.5: {mm['ring05_share']:.3f}%) p99.9 {mm['ring_p999']:.2f} max "
                  f"{mm['ring_max']:.2f}" + (f"; vs own source: ring {mm['ring_src_share']:.4f}% (>0.5: "
                                             f"{mm['ring_src05_share']:.3f}%) p99.9 {mm['ring_src_p999']:.2f} max "
                                             f"{mm['ring_src_max']:.2f}" if "ring_src_share" in mm else ""))
    log(f"{a.clip}: {n} frames, {len(streams)} streams in {res['seconds']} s -> {a.json}")


# ------------------------------------------------------------------ review

LABEL_H = 22


def window_scores(score_map, h, w, wh, ww, step):
    """Mean of score_map over every window (wh, ww) on a grid of `step` px: (ys, xs, means)."""
    ii = cv2.integral(score_map.astype(np.float64))
    ys = np.arange(0, h - wh + 1, step)
    xs = np.arange(0, w - ww + 1, step)
    s = ii[ys[:, None] + wh, xs[None, :] + ww] - ii[ys[:, None], xs[None, :] + ww] \
        - ii[ys[:, None] + wh, xs[None, :]] + ii[ys[:, None], xs[None, :]]
    return ys, xs, s / (wh * ww)


def cmd_review(a):
    cv2.setNumThreads(a.threads)
    gsrc = F.Source(a.gt, threads=a.threads)
    H, W, win = gsrc.H, gsrc.W, a.win
    best = bestf = None
    t = 0
    while True:
        g = gsrc.read()
        if g is None:
            break
        _, cb, cr = ycc(g)
        mag = np.sqrt(sum(cv2.Sobel(c, cv2.CV_32F, dx, dy, ksize=3) ** 2 for c in (cb, cr)
                          for dx, dy in ((1, 0), (0, 1))))
        ys, xs, s = window_scores(mag, H, W, win, win, 8)
        if best is None:
            best, bestf = s.copy(), np.zeros(s.shape, int)
        else:
            up = s > best
            best[up], bestf[up] = s[up], t
        t += 1
    gsrc.close()
    picked = []
    for k in np.argsort(best.ravel())[::-1]:
        iy, ix = divmod(int(k), len(xs))
        yy, xx = int(ys[iy]), int(xs[ix])
        if any(abs(yy - y2) < 1.5 * win and abs(xx - x2) < 1.5 * win for _, y2, x2, _ in picked):
            continue
        picked.append((int(bestf.flat[k]), yy, xx, float(best.flat[k])))
        if len(picked) == a.crops:
            break
    panels = [("GT", a.gt), ("model (no conversion)", a.model)] + \
        [(f"{v} round trip", rt_path(a.dir, a.clip, "model", v, "rgb")) for v in a.panels.split(",")]
    need = {f for f, *_ in picked}
    crops = {}  # (panel, frame) -> float RGB window
    for label, path in panels:
        src = F.Source(path, threads=a.threads)
        i = 0
        while i <= max(need):
            x = src.read()
            if x is None:
                raise SystemExit(f"{path}: only {i} frames")
            if i in need:
                for f, yy, xx, _ in picked:
                    if f == i:
                        crops[(label, f, yy, xx)] = x[yy:yy + win, xx:xx + win].copy()
            i += 1
        src.close()
    os.makedirs(a.out, exist_ok=True)
    rows, tsv = [], []
    z = a.zoom
    for f, yy, xx, score in picked:
        g = crops[("GT", f, yy, xx)]
        lab_g = lab_of(g)
        tiles, des = [], []
        for label, _ in panels:
            img = crops[(label, f, yy, xx)]
            de = 0.0 if label == "GT" else float(de00(lab_of(img), lab_g).mean())
            des.append(de)
            big = np.repeat(np.repeat(np.rint(np.clip(img, 0, 1) * 255).astype(np.uint8), z, 0), z, 1)[..., ::-1]
            strip = np.zeros((LABEL_H, win * z, 3), np.uint8)
            text = label if label == "GT" else f"{label}  dE00 {de:.2f}"
            cv2.putText(strip, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(np.vstack([strip, big]))
            tiles.append(np.full((LABEL_H + win * z, 4, 3), 64, np.uint8))
        row = np.hstack(tiles[:-1])
        head = np.zeros((LABEL_H, row.shape[1], 3), np.uint8)
        cv2.putText(head, f"{a.clip}  frame {f}  x {xx}  y {yy}  ({win}x{win} px, {z}x nearest; chroma gradient "
                    f"{score:.1f})", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1, cv2.LINE_AA)
        rows += [head, row, np.full((8, row.shape[1], 3), 32, np.uint8)]
        tsv.append([a.clip, str(f), str(xx), str(yy), f"{score:.2f}"] + [f"{d:.3f}" for d in des[1:]])
        log(f"{a.clip}: frame {f} x {xx} y {yy} score {score:.1f}; ΔE00 " +
            ", ".join(f"{lb} {d:.2f}" for (lb, _), d in zip(panels[1:], des[1:])))
    path = os.path.join(a.out, f"{a.clip}.png")
    cv2.imwrite(path, np.vstack(rows))
    tp = os.path.join(a.out, "crops.tsv")
    new = not os.path.exists(tp)
    with open(tp, "a", encoding="utf-8") as fh:
        if new:
            fh.write("\t".join(["clip", "frame", "x", "y", "chroma_gradient"] + [lb for lb, _ in panels[1:]]) + "\n")
        for r in tsv:
            fh.write("\t".join(r) + "\n")
    print(f"{path}: {len(picked)} crops")


# ------------------------------------------------------------------ timing

def cmd_timing(a):
    st = F.probe(a.src)
    W, H = st["width"], st["height"]
    raw = os.path.join(a.tmp, f"chroma_timing.{os.getpid()}.gbrp16le.raw")
    yraw = os.path.join(a.tmp, f"chroma_timing.{os.getpid()}.yuv420p10le.raw")
    try:
        run([FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-threads", "4", "-i", a.src,
             "-map", "0:v:0", "-frames:v", str(a.frames), "-f", "rawvideo", "-pix_fmt", "gbrp16le", raw], "decode")
        rawin = ["-f", "rawvideo", "-s", f"{W}x{H}", "-framerate", "24"]
        run([FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", *rawin, "-pix_fmt", "gbrp16le", "-i", raw,
             "-vf", down_vf("bilinear"), "-f", "rawvideo", "-pix_fmt", "yuv420p10le", yraw], "yuv")
        jobs = [("copy", raw, "gbrp16le", "null")] + [(v, raw, "gbrp16le", down_vf(v)) for v in kernels_arg(a.kernels)] \
            + [("copy-yuv", yraw, "yuv420p10le", "null"), ("up-catrom", yraw, "yuv420p10le", up_vf("yuv420p10le"))]
        n = a.frames * a.loops
        times, cpus = defaultdict(list), defaultdict(list)
        for T in [int(x) for x in a.threads.split(",")]:
            for r in range(a.rounds):
                for name, inp, fmt, vf in jobs:
                    cmd = [FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-threads", "1",
                           "-filter_threads", str(T), *rawin, "-pix_fmt", fmt, "-stream_loop", str(a.loops - 1),
                           "-i", inp, "-vf", vf, "-f", "null", "-"]
                    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
                    times[(T, name)].append(run(cmd, f"timing {name}"))
                    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
                    cpus[(T, name)].append(r1.ru_utime - r0.ru_utime + r1.ru_stime - r0.ru_stime)
                log(f"threads {T}, round {r + 1}: " + ", ".join(
                    f"{nm} {times[(T, nm)][-1]:.2f} s ({cpus[(T, nm)][-1]:.2f} s CPU)" for nm, *_ in jobs))
        res = {"src": a.src, "frames": n, "size": [W, H], "rounds": a.rounds, "results": {}}
        for T in [int(x) for x in a.threads.split(",")]:
            out = {}
            for name, inp, fmt, vf in jobs:
                if name.startswith("copy"):
                    continue
                base = "copy-yuv" if name == "up-catrom" else "copy"
                per = [(t - c) / n * 1000 for t, c in zip(times[(T, name)], times[(T, base)])]
                cpu = [(t - c) / n * 1000 for t, c in zip(cpus[(T, name)], cpus[(T, base)])]
                out[name] = {"ms_per_frame": float(np.median(per)), "min": float(min(per)), "max": float(max(per)),
                             "cpu_ms_per_frame": float(np.median(cpu)), "raw_s": times[(T, name)],
                             "copy_s": times[(T, base)], "cpu_s": cpus[(T, name)], "copy_cpu_s": cpus[(T, base)], "vf": vf}
                print(f"threads {T}: {name}: {np.median(per):.2f} ms/frame wall ({min(per):.2f}-{max(per):.2f}), "
                      f"{np.median(cpu):.2f} ms/frame CPU")
            res["results"][str(T)] = out
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
    finally:
        for p in (raw, yraw):
            if os.path.exists(p):
                os.remove(p)


# ------------------------------------------------------------------ summary

COLS = [  # key, title, higher is better, format
    ("psnr_y", "PSNR-Y", True, "{:.2f}"),
    ("psnr_cb", "PSNR-Cb", True, "{:.2f}"),
    ("psnr_cr", "PSNR-Cr", True, "{:.2f}"),
    ("de00_s0", "ΔE00 σ0", False, "{:.4f}"),
    ("de00_s1", "ΔE00 σ1", False, "{:.4f}"),
    ("de00_s2", "ΔE00 σ2", False, "{:.4f}"),
    ("de00_s4", "ΔE00 σ4", False, "{:.4f}"),
    ("de00_edge", "edge ΔE00", False, "{:.4f}"),
    ("ring_share", "ring %", False, "{:.3f}"),
    ("ring05_share", "ring>0.5 %", False, "{:.3f}"),
    ("ring_p999", "ring p99.9", False, "{:.2f}"),
    ("ring_max", "ring max", False, "{:.2f}"),
]
SRC_COLS = [
    ("psnr_y_src", "PSNR-Y", True, "{:.2f}"),
    ("psnr_cb_src", "PSNR-Cb", True, "{:.2f}"),
    ("psnr_cr_src", "PSNR-Cr", True, "{:.2f}"),
    ("de00_s0_src", "ΔE00 σ0", False, "{:.4f}"),
    ("ring_src_share", "ring %", False, "{:.4f}"),
    ("ring_src05_share", "ring>0.5 %", False, "{:.3f}"),
    ("ring_src_p999", "ring p99.9", False, "{:.2f}"),
    ("ring_src_p9999", "ring p99.99", False, "{:.2f}"),
    ("ring_src_max", "ring max", False, "{:.2f}"),
    ("dy", "mean ΔY'", None, "{:+.4f}"),
    ("dcb", "mean ΔCb", None, "{:+.4f}"),
    ("dcr", "mean ΔCr", None, "{:+.4f}"),
]
PAIRED = ["psnr_cb", "psnr_cr", "de00_s0", "de00_s1", "de00_s4", "de00_edge", "ring_share", "ring05_share"]
FRM = [("psnr_y", "PSNR-Y", "{:.3f}"), ("ssim_y", "SSIM-Y", "{:.5f}"), ("lpips", "LPIPS", "{:.5f}"),
       ("dists", "DISTS", "{:.5f}"), ("vmaf", "VMAF", "{:.3f}"), ("vmaf_neg", "VMAF NEG", "{:.3f}"),
       ("de00_lf", "ΔE00 lf", "{:.4f}"), ("temporal_lf", "T-err lf", "{:.4f}"), ("temporal_full", "T-err", "{:.4f}"),
       ("cambi_added", "CAMBI added", "{:.4f}")]


def col(key):
    return next(c for c in COLS + SRC_COLS if c[0] == key)


def fmtv(v, f):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if isinstance(v, float) and math.isinf(v):
        return "inf"
    return f.format(v)


def signed(f):
    return "{:+" + f[2:]


def table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]


def is_live(clip):
    return clip.startswith("live")


def cmd_summary(a):
    runs = [json.load(open(p, encoding="utf-8")) for p in a.json]
    clips = [r["clip"] for r in runs]
    R = {r["clip"]: r for r in runs}
    variants = [v for v in VARIANTS if all(v in r["means"]["gt"] for r in runs)]
    rng = np.random.default_rng(0)
    L = []
    P = L.append
    P(f"# Chroma downsampling kernels for the yuv420p10le master\n")
    P(f"{len(clips)} clips ({', '.join(clips)}), {', '.join(str(r['frames']) for r in runs)} frames. Model output = "
      f"numz default, seed 42, with lab (q1 `def` `.cc-lab.mkv`). Round trip = RGB 16-bit → yuv420p10le with the "
      f"kernel → 16-bit RGB with Catmull-Rom (seedvr2x's reader). Scores against the GT unless said; Y'CbCr BT.709 "
      f"full range on the 8-bit scale; ring = Cb or Cr beyond the GT's local 5×5 range by > {RING_T:g} levels.\n")

    # ---- checks
    P("## Checks\n")
    for p in a.proof or []:
        pr = json.load(open(p, encoding="utf-8"))
        P(f"- proof {os.path.basename(pr['src'])}: {pr['identical']}/{pr['frames'][0]} frames identical to "
          f"ffv1_out.Writer's (framemd5); explicit `f=bilinear` {pr['explicit_bilinear_identical']}/{pr['frames'][2]}; "
          f"tags {'equal' if pr['tags_equal'] else 'DIFFER'} → **{'OK' if pr['ok'] else 'FAILED'}**")
    if a.siting:
        si = json.load(open(a.siting, encoding="utf-8"))
        P(f"- siting (synthetic): {si['verdict']}")
        rows = []
        for k, v in si["kernels"].items():
            c, st = v["checks"], v["steps"]
            rows.append([k] + [fmtv(c[n], "{:g}") for n in c] +
                        [f"{st['step-x32_cb']['overshoot']:.2f} / {st['step-y16_cb']['overshoot']:.2f}",
                         f"{st['step-x32_cb']['rise_10_90']:.2f} / {st['step-y16_cb']['rise_10_90']:.2f}"])
        first = next(iter(si["kernels"].values()))
        L += [""] + table(["kernel"] + list(first["checks"]) + ["step Cb overshoot x / y", "step Cb rise 10-90 x / y"], rows)
    if a.threads_json:
        th = json.load(open(a.threads_json, encoding="utf-8"))
        P(f"\nzscale's slice threading: each -filter_threads against 1, first {th['frames']} frames of "
          f"{os.path.basename(th['src'])}: samples that differ (share), max |difference|; the up chain reads the "
          f"single-threaded master.\n")
        rows = []
        for k, out in th["kernels"].items():
            for ft, d in out.items():
                cell = lambda s: f"{s['samples']} ({s['share']:.3%}), max {s['max']}"  # noqa: E731
                rows.append([k, ft] + [cell(d["down"][p]) for p in ("Y", "Cb", "Cr")] + [cell(d["up"])])
        L += table(["kernel", "filter_threads", "Y' (10-bit codes)", "Cb", "Cr", "round trip RGB (16-bit codes)"], rows)
    for p in a.gtcheck or []:
        g = json.load(open(p, encoding="utf-8"))
        P(f"\n- GT {os.path.basename(g['gt'])} against its source re-converted by fr_clips.py's chain, first "
          f"{g['frames']} frames: " + "; ".join(
              f"filter_threads {ft}: {s['samples']} samples differ ({s['share']:.3%}), max {s['max']}"
              for ft, s in g["threads"].items()))
    for p in a.upcheck or []:
        u = json.load(open(p, encoding="utf-8"))
        P(f"\n- Catmull-Rom read of {os.path.basename(u['yuv'])} ({u['pix_fmt']}, frame 0) against a float64 "
          f"reference, 16-bit RGB codes, {u['margin']}-px border excluded: " + "; ".join(
              f"filter_threads {ft}: " + ", ".join(f"vs {n} max {s['max']:.0f} rms {s['rms']:.2f}" for n, s in r.items())
              for ft, r in u["threads"].items()))
    P("\nConstant shift, round trip − its source, mean over frames (8-bit levels): the largest |mean| over clips.\n")
    rows = []
    for tag in ("model", "gt"):
        for v in variants:
            rows.append([tag, v] + [fmtv(max(abs(R[c]["means"][tag][v][k]) for c in clips), "{:.4f}")
                                    for k in ("dy", "dcb", "dcr")])
    L += table(["source", "variant", "largest mean ΔY'", "largest mean ΔCb", "largest mean ΔCr"], rows)
    frm = None
    if a.frm:
        frm = defaultdict(dict)
        for r in F.load([a.frm]):
            frm[r["clip"]][r["variant"]] = r
        P("\nσ4 check: this script's de00_s4 of the unconverted model output against fr_metrics' de00_lf "
          "(`src` in --frm; q1's `def+lab` seed 42 JSON with --q1).\n")
        rows = []
        for c in clips:
            mine = R[c]["series"]["model"]["none"]["de00_s4"]
            fs = frm[c].get("src", {}).get("per_frame", {}).get("de00_lf")
            row = [c, f"{np.mean(mine):.6f}"]
            row.append(f"{np.mean(fs):.6f} (largest per-frame Δ {np.max(np.abs(np.array(mine) - np.array(fs))):.1e})"
                       if fs else "–")
            if a.q1:
                p = os.path.join(a.q1, f"{c}-d1.def+lab.s42.json")
                q = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None
                qs = q["per_frame"]["de00_lf"] if q else None
                row.append(f"{np.mean(qs):.6f} (largest per-frame Δ {np.max(np.abs(np.array(mine) - np.array(qs))):.1e})"
                           if qs else "–")
            rows.append(row)
        L += table(["clip", "de00_s4 (this script)", "fr_metrics de00_lf (src)"] + (["q1 def+lab s42 de00_lf"] if a.q1 else []),
                   rows)

    # ---- per clip
    for tag, title in (("model", "Model output: unconverted (`none`) and its round trips, against the GT"),
                       ("gt", "GT round trips (the format's cost on perfect content), against the GT")):
        P(f"\n## {title}\n")
        rows = []
        for c in clips:
            for v in (["none"] if tag == "model" else []) + variants:
                m = R[c]["means"][tag][v]
                rows.append([c, v] + [fmtv(m.get(k), f) for k, _, _, f in COLS])
        L += table(["clip", "variant"] + [t for _, t, _, _ in COLS], rows)
    P("\n## Model output: round trip against its own source (the conversion alone)\n")
    rows = []
    for c in clips:
        for v in variants:
            m = R[c]["means"]["model"][v]
            rows.append([c, v] + [fmtv(m.get(k), f) for k, _, _, f in SRC_COLS])
    L += table(["clip", "variant"] + [t for _, t, _, _ in SRC_COLS], rows)

    # ---- paired
    for tag in ("model", "gt"):
        for v in variants:
            if v == "bilinear":
                continue
            P(f"\n## {tag}: `{v}` − `bilinear`, paired by frame\n")
            P("Mean difference per frame [95% CI, moving-block bootstrap over frames, blocks of 8]; ✓ better / ✗ worse "
              "when the CI excludes 0, = within.\n")
            rows, tally = [], defaultdict(lambda: defaultdict(int))
            for c in clips:
                row = [c]
                for k in PAIRED:
                    _, _, higher, f = col(k)
                    x = np.array(R[c]["series"][tag][v][k]) - np.array(R[c]["series"][tag]["bilinear"][k])
                    lo, hi = F.block_bootstrap([x], 8, 2000, rng)
                    mean = float(x.mean())
                    verdict = ("better" if (mean > 0) == higher else "worse") if (lo > 0 or hi < 0) else "within"
                    tally[k][verdict] += 1
                    mark = {"better": "✓", "worse": "✗", "within": "="}[verdict]
                    row.append(f"{fmtv(mean, signed(f))} [{fmtv(lo, signed(f))}, {fmtv(hi, signed(f))}] {mark}")
                rows.append(row)
            rows.append(["**better / worse / within**"] + [f"{tally[k]['better']} / {tally[k]['worse']} / {tally[k]['within']}"
                                                           for k in PAIRED])
            L += table(["clip"] + [col(k)[1] for k in PAIRED], rows)

    # ---- means over clips
    groups = [("all", clips), ("anime", [c for c in clips if not is_live(c)]), ("live", [c for c in clips if is_live(c)])]
    for tag in ("model", "gt"):
        P(f"\n## {tag}: means over clips, and − bilinear (wins = clips where the kernel is better)\n")
        keys = [k for k, *_ in COLS]
        rows = []
        for gname, cl in groups:
            if not cl:
                continue
            for v in (["none"] if tag == "model" else []) + variants:
                row = [f"{gname} ({len(cl)})", v]
                for k in keys:
                    _, _, higher, f = col(k)
                    val = float(np.mean([R[c]["means"][tag][v][k] for c in cl]))
                    if v in ("bilinear", "none"):
                        row.append(fmtv(val, f))
                        continue
                    ds = [R[c]["means"][tag][v][k] - R[c]["means"][tag]["bilinear"][k] for c in cl]
                    wins = sum((d > 0) == higher and d != 0 for d in ds)
                    row.append(f"{fmtv(val, f)} ({fmtv(float(np.mean(ds)), signed(f))}, {wins}/{len(cl)})")
                rows.append(row)
        L += table(["clips", "variant"] + [t for _, t, _, _ in COLS], rows)

    # ---- headroom and floor
    P("\n## Headroom and the 10-bit floor\n")
    P("Headroom = what a perfect 4:2:0 kernel could at most win over bilinear: bilinear − yuv444 (oriented, > 0 = "
      "yuv444 better), mean over clips. Gain = kernel − bilinear (oriented, > 0 = kernel better), mean over clips; "
      "share = gain / headroom, only where the headroom is positive (a negative one means no 4:4:4 master would do "
      "better than bilinear's 4:2:0). Floor = yuv444 against its own source (10-bit quantisation alone).\n")
    for tag in ("model", "gt"):
        rows = []
        for k in ["psnr_y", "psnr_cb", "psnr_cr", "de00_s0", "de00_s1", "de00_s4", "de00_edge"]:
            _, title, higher, f = col(k)
            sgn = 1 if higher else -1
            head = float(np.mean([sgn * (R[c]["means"][tag]["yuv444"][k] - R[c]["means"][tag]["bilinear"][k]) for c in clips]))
            row = [title, fmtv(head, signed(f))]
            for v in variants:
                if v in ("bilinear", "yuv444"):
                    continue
                g = float(np.mean([sgn * (R[c]["means"][tag][v][k] - R[c]["means"][tag]["bilinear"][k]) for c in clips]))
                row.append(f"{fmtv(g, signed(f))} ({g / head * 100:+.0f}%)" if head > 0 else fmtv(g, signed(f)))
            sk = k + "_src" if k in ("psnr_y", "psnr_cb", "psnr_cr", "de00_s0") else None
            row.append(fmtv(float(np.mean([R[c]["means"][tag]["yuv444"][sk] for c in clips])), f) if sk else "–")
            rows.append(row)
        P(f"\n{tag}:\n")
        L += table(["metric", "headroom"] + [f"{v} gain (share)" for v in variants if v not in ("bilinear", "yuv444")]
                   + ["floor (yuv444 vs its source)"], rows)

    # ---- against the source's own 4:2:0 chroma
    if a.srcchroma:
        sc = {}
        for p in a.srcchroma:
            j = json.load(open(p, encoding="utf-8"))
            sc[j["clip"]] = j
        cl = [c for c in clips if c in sc]
        P("\n## Each kernel's 4:2:0 chroma against the source's own 4:2:0 chroma (src.mkv, 8-bit scale)\n")
        P("No upsampling involved: the master's Cb and Cr planes against the original's, sample for sample "
          "(both sited left). PSNR-Cb / PSNR-Cr; in brackets − bilinear.\n")
        for tag in ("model", "gt"):
            ks = [k for k in variants if k != "yuv444" and all(k in sc[c]["means"][tag] for c in cl)]
            rows = []
            for c in cl:
                m = sc[c]["means"][tag]
                rows.append([c] + [f"{m[k]['psnr_cb']:.2f} / {m[k]['psnr_cr']:.2f}" + (
                    "" if k == "bilinear" else f" ({m[k]['psnr_cb'] - m['bilinear']['psnr_cb']:+.2f} / "
                                               f"{m[k]['psnr_cr'] - m['bilinear']['psnr_cr']:+.2f})") for k in ks])
            wins = []
            for k in ks:
                if k == "bilinear":
                    wins.append("–")
                    continue
                w = sum(sc[c]["means"][tag][k][q] > sc[c]["means"][tag]["bilinear"][q] for c in cl for q in ("psnr_cb", "psnr_cr"))
                wins.append(f"better on {w}/{2 * len(cl)} (clip, plane)")
            rows.append(["**wins**"] + wins)
            P(f"\n{tag}:\n")
            L += table(["clip"] + ks, rows)

    # ---- fr_metrics
    if frm:
        P("\n## fr_metrics' standard set on the round-tripped model outputs\n")
        fv = ["src"] + [v for v in variants if all(v in frm[c] for c in clips)]
        rows = []
        for c in clips:
            for v in fv:
                if v not in frm[c]:
                    continue
                m = frm[c][v]["means"]
                rows.append([c, v] + [fmtv(m.get(k), f) for k, _, f in FRM])
        L += table(["clip", "variant"] + [t for _, t, _ in FRM], rows)
        P("\nDifference to `bilinear`, mean over clips [min, max over clips]; `src` = the unconverted output "
          "(src − bilinear: what the default conversion itself costs, sign as the metric).\n")
        rows = []
        for v in fv:
            if v == "bilinear":
                continue
            row = [v]
            for k, _, f in FRM:
                ds = [frm[c][v]["means"].get(k) - frm[c]["bilinear"]["means"].get(k) for c in clips
                      if frm[c].get(v) and frm[c][v]["means"].get(k) is not None
                      and frm[c]["bilinear"]["means"].get(k) is not None]
                row.append(f"{fmtv(float(np.mean(ds)), signed(f))} [{fmtv(min(ds), signed(f))}, {fmtv(max(ds), signed(f))}]"
                           if ds else "–")
            rows.append(row)
        L += table(["variant"] + [t for _, t, _ in FRM], rows)
        if a.q1:
            bands = defaultdict(list)
            for c in clips:
                ms = [json.load(open(p, encoding="utf-8"))["means"] for p in
                      (os.path.join(a.q1, f"{c}-d1.def+lab.s{s}.json") for s in (42, 43, 1234)) if os.path.exists(p)]
                for k, _, _ in FRM:
                    vals = [m[k] for m in ms if m.get(k) is not None]
                    if len(vals) >= 2:
                        bands[k].append(max(vals) - min(vals))
            P("\nFor scale, the seed band: max − min over q1's `def+lab` seeds 42, 43 and 1234 (same clip, same "
              "settings), mean over the clips [min, max].\n")
            L += table(["seed band"] + [t for _, t, _ in FRM], [["def+lab"] + [
                f"{fmtv(float(np.mean(bands[k])), f)} [{fmtv(min(bands[k]), f)}, {fmtv(max(bands[k]), f)}]"
                if bands[k] else "–" for k, _, f in FRM]])

    # ---- timing
    if a.timing:
        ti = json.load(open(a.timing, encoding="utf-8"))
        P(f"\n## Cost: zscale alone, ms per {ti['size'][0]}x{ti['size'][1]} frame (median of {ti['rounds']} rounds, "
          f"copy pass subtracted, {ti['frames']} frames): wall [min–max] / CPU (user + system)\n")
        ths = list(ti["results"])
        names = list(ti["results"][ths[0]])
        rows = [[n] + [f"{r['ms_per_frame']:.2f} [{r['min']:.2f}–{r['max']:.2f}] / "
                       + (f"{r['cpu_ms_per_frame']:.2f}" if "cpu_ms_per_frame" in r else "–")
                       for r in (ti["results"][t][n] for t in ths)] for n in names]
        L += table(["chain"] + [f"filter_threads {t}" for t in ths], rows)
    text = "\n".join(L) + "\n"
    if a.md:
        with open(a.md, "w", encoding="utf-8") as fh:
            fh.write(text)
        log(f"-> {a.md}")
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("siting", help="synthetic frames: chroma siting and step response of every kernel")
    s.add_argument("--json", required=True)
    s.add_argument("--kernels")
    s.set_defaults(func=cmd_siting)
    s = sub.add_parser("convert", help="masters and round trips of a clip's model output and GT")
    s.add_argument("--clip", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--gt", required=True)
    s.add_argument("--dir", required=True)
    s.add_argument("--kernels")
    s.add_argument("--threads", type=int, default=4, help="FFV1 decoder and encoder threads")
    s.add_argument("--filter-threads", type=int, default=1,
                   help="zscale's slices (default 1: zscale's output depends on it, see `threads`)")
    s.add_argument("--verbose", action="store_true", help="log the ffmpeg commands")
    s.set_defaults(func=cmd_convert)
    s = sub.add_parser("threads", help="zscale's output for each -filter_threads value against 1")
    s.add_argument("--src", required=True)
    s.add_argument("--json", required=True)
    s.add_argument("--kernels", default="bilinear,lanczos")
    s.add_argument("--frames", type=int, default=3)
    s.add_argument("--threads", default="2,4,8,16,24,48", help="comma list of -filter_threads, each against 1")
    s.set_defaults(func=cmd_threads)
    s = sub.add_parser("upcheck", help="zscale's Catmull-Rom read against a float64 reference, per -filter_threads")
    s.add_argument("--yuv", required=True, help="a yuv420p10le or yuv420p source (BT.709 limited, sited left)")
    s.add_argument("--json", required=True)
    s.add_argument("--threads", default="1,2,3,4,8,16")
    s.add_argument("--margin", type=int, default=8)
    s.set_defaults(func=cmd_upcheck)
    s = sub.add_parser("srcchroma", help="each kernel's 4:2:0 chroma against the source's own (src.mkv)")
    s.add_argument("--clip", required=True)
    s.add_argument("--src", required=True, help="the clip's src.mkv (fr_clips.py: 4:2:0, sited left)")
    s.add_argument("--dir", required=True)
    s.add_argument("--json", required=True)
    s.add_argument("--kernels")
    s.set_defaults(func=cmd_srcchroma)
    s = sub.add_parser("gtcheck", help="a GT against its YUV source re-converted with each -filter_threads")
    s.add_argument("--gt", required=True)
    s.add_argument("--src", required=True, help="the clip's src.mkv (fr_clips.py)")
    s.add_argument("--json", required=True)
    s.add_argument("--vf", default="zscale=matrixin=709:rangein=limited:chromalin=left:dither=none",
                   help="fr_clips.py's YUV -> RGB chain for the clip (BT.709, limited, left by default)")
    s.add_argument("--frames", type=int, default=3)
    s.add_argument("--threads", default="1,16,24,48")
    s.set_defaults(func=cmd_gtcheck)
    s = sub.add_parser("proof", help="the bilinear master against ffv1_out.Writer's conversion (framemd5)")
    s.add_argument("--src", required=True)
    s.add_argument("--yuv", required=True)
    s.add_argument("--work", required=True)
    s.add_argument("--json", required=True)
    s.add_argument("--threads", type=int, default=4)
    s.add_argument("--filter-threads", type=int, default=1, help="the -filter_threads --yuv was made with")
    s.add_argument("--keep", action="store_true", help="keep ffv1_out.Writer's file")
    s.set_defaults(func=cmd_proof)
    s = sub.add_parser("score", help="round trips of a clip against the GT")
    s.add_argument("--clip", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--gt", required=True)
    s.add_argument("--dir", required=True)
    s.add_argument("--json", required=True)
    s.add_argument("--kernels")
    s.add_argument("--frames", type=int, default=0)
    s.add_argument("--threads", type=int, default=1, help="decoder and OpenCV threads")
    s.set_defaults(func=cmd_score)
    s = sub.add_parser("review", help="4x crops of the sharpest chroma edges, side by side")
    s.add_argument("--clip", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--gt", required=True)
    s.add_argument("--dir", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--panels", default="bilinear,spline36,lanczos")
    s.add_argument("--crops", type=int, default=3)
    s.add_argument("--win", type=int, default=96)
    s.add_argument("--zoom", type=int, default=4)
    s.add_argument("--threads", type=int, default=2)
    s.set_defaults(func=cmd_review)
    s = sub.add_parser("timing", help="zscale's time per frame and kernel")
    s.add_argument("--src", required=True)
    s.add_argument("--json", required=True)
    s.add_argument("--kernels")
    s.add_argument("--frames", type=int, default=24)
    s.add_argument("--loops", type=int, default=10, help="the raw frames are read this many times (-stream_loop)")
    s.add_argument("--threads", default="1,8", help="comma list of -filter_threads")
    s.add_argument("--rounds", type=int, default=3)
    s.add_argument("--tmp", default="/dev/shm")
    s.set_defaults(func=cmd_timing)
    s = sub.add_parser("summary", help="Markdown tables of score JSONs")
    s.add_argument("json", nargs="+")
    s.add_argument("--frm", help="fr_metrics JSON directory of the round-tripped model outputs")
    s.add_argument("--q1", help="q1 metrics JSON directory (d1-lab) for the σ4 check")
    s.add_argument("--proof", nargs="+")
    s.add_argument("--siting")
    s.add_argument("--threads-json", help="the `threads` JSON")
    s.add_argument("--gtcheck", nargs="+", help="`gtcheck` JSONs")
    s.add_argument("--upcheck", nargs="+", help="`upcheck` JSONs")
    s.add_argument("--srcchroma", nargs="+", help="`srcchroma` JSONs")
    s.add_argument("--timing")
    s.add_argument("--md")
    s.set_defaults(func=cmd_summary)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()

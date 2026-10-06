#!/usr/bin/env python3
"""Full-reference inputs at another upscale factor, for the colour study (docs/colour.md).

  colour_clips.py make CLIPS_DIR/NAME --factor F --out DIR [--degrade d1|d2] [--threads 8]   # F: 4, 3, 1.5...

Why: fr_clips.py makes a clip's ground truth and its x1/2 inputs (d1, d2). The colour study also
needs x4 inputs, since the scale of a colour correction's split may follow the upscale factor
(DESIGN.md, Beyond numz's lab: 13 px of output is 6.5 source pixels at x2, 3.3 at x4). This makes
fr_clips.py's d1 at x1/F instead of x1/2, from the clip's own source slice and manifest
(NAME.src.mkv, NAME.json, read only), with the recipe otherwise unchanged, into DIR. --degrade d2
makes fr_clips.py's d2 instead (swscale area downscale, CRF 26), which it made for two clips only,
for the other clips (at x1/2 the files are NAME.d2.*, fr_clips.py's names, in DIR):
- NAME.d1xF.x264.mkv: the source downscaled x1/F in YUV with zscale bicubic b = c = 1/3
  (Mitchell), limited range, chroma left, x264 -preset slow -crf 20, keyint 48
- NAME.d1xF.lr.mkv: the CLI's input: that file decoded, zscale to full-range 8-bit RGB, FFV1
  bgr0, which the CLI's reader (cv2) gets bit-exactly
- NAME.d1xF.bicubic.mkv: the baseline: the input upscaled xF with zscale bicubic b = 0, c = 1/2
  (Catmull-Rom) to 16-bit RGB at the ground truth's size
The ground truth stays CLIPS_DIR/NAME.gt.mkv. Needs ffmpeg with zimg and libx264. Every zscale runs
on one slice (threads=1), as fr_clips.py's: ffmpeg's slice threading changes a 10-bit 4:2:0
source's RGB (8-bit sources and the resizes were measured unaffected).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

FFMPEG = os.environ.get("COLOUR_FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = os.environ.get("COLOUR_FFPROBE") or shutil.which("ffprobe") or "ffprobe"
# fr_clips.py's constants (its d1 recipe)
FFV1 = ["-c:v", "ffv1", "-level", "3", "-g", "1", "-slices", "16", "-slicecrc", "1"]
THIRD = "0.3333333333333333"
MITCHELL = f"filter=bicubic:param_a={THIRD}:param_b={THIRD}"  # zimg: param_a = b, param_b = c
CATROM = "filter=bicubic:param_a=0:param_b=0.5"
KEYINT = 48
CRF = {"d1": 20, "d2": 26}  # fr_clips.py's DEGRADATIONS
ZMATRIX = {"bt709": "709", "smpte170m": "170m", "bt470bg": "470bg", "bt2020nc": "2020_ncl"}


def run(cmd, what):
    print(f"{what}: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode:
        sys.exit(f"{what}: ffmpeg exited with {r.returncode}")


def probe(path):
    r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                        "stream=width,height,pix_fmt,nb_read_packets", "-of", "json", path],
                       capture_output=True, text=True)
    st = json.loads(r.stdout or "{}").get("streams") if not r.returncode else None
    if not st:
        sys.exit(f"ffprobe {path}: {r.stderr.strip() or 'no video stream'}")
    s = st[0]
    return s["width"], s["height"], s["pix_fmt"], int(s["nb_read_packets"])


def rgb_params(c):
    return f"setparams=colorspace=gbr:range=pc:color_primaries={c['primaries']}:color_trc={c['transfer']}"


def rgb_tags(c):
    return ["-colorspace", "rgb", "-color_primaries", c["primaries"], "-color_trc", c["transfer"],
            "-color_range", "pc"]


def yuv_params(c):
    return (f"setparams=colorspace={c['matrix']}:range={c['range']}:color_primaries={c['primaries']}"
            f":color_trc={c['transfer']}:chroma_location={c['chroma_location']}")


def yuv_tags(c):
    return ["-colorspace", c["matrix"], "-color_primaries", c["primaries"], "-color_trc", c["transfer"],
            "-color_range", c["range"], "-chroma_sample_location", c["chroma_location"]]


def make(a):
    base = a.clip
    name = os.path.basename(base)
    with open(base + ".json", encoding="utf-8") as f:
        man = json.load(f)
    c = man["colours"]
    W, H, _, n_gt = probe(base + ".gt.mkv")
    F = a.factor  # 1.5 too: 1920x1080 / 1.5 = 1280x720
    w, h = round(W / F), round(H / F)
    if abs(w * F - W) > 1e-6 or abs(h * F - H) > 1e-6 or w % 2 or h % 2:
        sys.exit(f"{W}x{H} / {F:g}: not an exact even size for 4:2:0")
    os.makedirs(a.out, exist_ok=True)
    deg = a.degrade
    tag = deg if F == 2 else f"{deg}x{F:g}"
    p = lambda suffix: os.path.join(a.out, f"{name}.{tag}.{suffix}")  # noqa: E731
    zm = ZMATRIX[c["matrix"]]
    zr = "full" if c["range"] == "pc" else "limited"
    lr = {"matrix": c["matrix"], "range": "tv", "chroma_location": "left", "primaries": c["primaries"],
          "transfer": c["transfer"]}
    T = str(a.threads)
    if deg == "d1":
        down = (f"zscale=threads=1:w={w}:h={h}:{MITCHELL}:matrixin={zm}:matrix={zm}:rangein={zr}:range=limited"
                f":chromalin={c['chroma_location']}:chromal=left:dither=none")
    else:  # fr_clips.py's d2: swscale's area filter (zimg has none)
        down = f"scale={w}:{h}:flags=area+accurate_rnd:out_range=tv"
    run([FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-i", base + ".src.mkv", "-map", "0:v:0",
         "-vf", f"{down},format=yuv420p,{yuv_params(lr)}", "-fps_mode", "passthrough",
         "-c:v", "libx264", "-preset", "slow", "-crf", str(CRF[deg]), "-x264-params", f"keyint={KEYINT}",
         "-threads", T, "-pix_fmt", "yuv420p", *yuv_tags(lr), p("x264.mkv")], f"{tag} encode")
    to_rgb = f"zscale=threads=1:matrixin={ZMATRIX[lr['matrix']]}:rangein=limited:chromalin=left:dither=none"
    run([FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-threads", T, "-i", p("x264.mkv"),
         "-map", "0:v:0", "-vf", f"{to_rgb},format=gbrp,{rgb_params(c)}", "-fps_mode", "passthrough",
         *FFV1, "-threads", T, "-pix_fmt", "bgr0", *rgb_tags(c), p("lr.mkv")], f"{tag} input")
    run([FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-i", p("lr.mkv"), "-map", "0:v:0",
         "-vf", f"zscale=threads=1:w={W}:h={H}:{CATROM}:dither=none,format=gbrp16le,{rgb_params(c)}",
         "-fps_mode", "passthrough", *FFV1, "-threads", T, "-pix_fmt", "gbrp16le", *rgb_tags(c),
         p("bicubic.mkv")], f"{tag} bicubic baseline")
    for suffix, want in (("x264.mkv", (w, h)), ("lr.mkv", (w, h)), ("bicubic.mkv", (W, H))):
        fw, fh, fmt, n = probe(p(suffix))
        ok = (fw, fh) == want and n == n_gt
        print(f"{os.path.basename(p(suffix))}: {fmt} {fw}x{fh}, {n} frames"
              + ("" if ok else f" -- expected {want[0]}x{want[1]}, {n_gt} frames"))
        if not ok:
            sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("clip", help="CLIPS_DIR/NAME: NAME.json, NAME.src.mkv and NAME.gt.mkv there")
    m.add_argument("--factor", type=float, default=4, help="the upscale factor: x1/F inputs (4, 3, 1.5...)")
    m.add_argument("--out", required=True)
    m.add_argument("--degrade", choices=tuple(CRF), default="d1", help="fr_clips.py's recipe: d1 or d2")
    m.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    make(a)


if __name__ == "__main__":
    main()

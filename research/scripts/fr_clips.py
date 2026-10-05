#!/usr/bin/env python3
"""Full-reference test clips for SeedVR2: a frame-exact ground truth and the degraded inputs.

  fr_clips.py scan SRC [--ss T] [--duration D] [--frames N] [--json F]   # candidate shots (seek)
  fr_clips.py scores SRC --first A --last B [--json F]                    # frame-exact scene scores
  fr_clips.py make SRC --start S --frames N --name NAME --out DIR [--degrade d1,d2] [--crop]
  fr_clips.py slice DIR/NAME --first A --frames N --name NEW [--out DIR2] [--files gt,d1.lr]
  fr_clips.py verify DIR/NAME [--cv2-python PY] [--source]                # bit- and frame-exactness
  fr_clips.py sheet VIDEO --frames 0,22,44 --out PNG [--width 640]        # contact sheet
  fr_clips.py grain FILE [--frames N | --every] [--sigma 2] [--share 30] [--json F]  # film grain (levels)
  fr_clips.py selftest                                                    # colour chain, extraction

Why: scoring SeedVR2 against a ground truth (GT) needs a GT that is exactly the frames the
degraded input was made from, and an input the CLI reads without a conversion of its own. The
CLI reads its input with cv2.VideoCapture (8-bit BGR, OpenCV picking the YUV matrix itself,
colour tags dropped) and seeks with CAP_PROP_POS_FRAMES, which is not frame-exact in long-GOP
files. So every input is a short all-intra file starting at the clip's first frame, stored as
8-bit RGB: OpenCV then has no matrix to apply (`verify --cv2-python` checks it bit-exactly).

Frame index: 0-based, in the order ffmpeg decodes the file from its start (-fps_mode
passthrough), as framemd5 lists the frames. `make` picks a clip by decoding from the start and
counting (select='between(n,S,S+N-1)'), never by seeking; `scan` seeks, which is fine to find
candidates, and its indices (from the timestamps) are approximate until `scores` confirms them.
SRC may be an ffconcat list (*.ffconcat) of files of one codec, e.g. pre-split segments, read with
the concat demuxer: the index then counts across the files.

Files written by `make` (DIR/NAME.*):
  src.mkv          frames S..S+N-1 of the source as decoded: YUV, FFV1, lossless (the reference of
                   the frame-exactness check, and what everything below is made from)
  gt.mkv           ground truth: src in RGB, FFV1 gbrp16le, tagged RGB / BT.709 / full range.
                   zscale, the source's matrix (BT.709 for untagged HD), range (limited if
                   untagged) and chroma siting (left if untagged), no primaries/transfer change
  d1.x264.mkv      degradation D1: src downscaled x1/2 in YUV with zscale bicubic b = c = 1/3
                   (Mitchell-Netravali), x264 -preset slow -crf 20, keyint 48, tagged BT.709
  d1.lr.mkv        the CLI input: d1.x264 decoded, zscale BT.709 limited -> full range 8-bit RGB,
                   FFV1 bgr0 (FFV1 stores 8-bit RGB packed only: it takes no 8-bit gbrp; lossless)
  d1.bicubic.mkv   baseline: d1.lr upscaled x2 with zscale bicubic b = 0, c = 1/2 (Catmull-Rom)
                   to 16-bit RGB, the format of the CLI's masters (ffv1_out.py): a reference row
  d2.*             degradation D2: area downscale (swscale flags=area: zimg has no area filter;
                   swscale ignores the chroma siting, a quarter-pixel chroma shift at most), CRF 26
  crop.gt.mkv, d1.crop.lr.mkv, d1.crop.bicubic.mkv
                   --crop: the multiple-of-16 control, rows 4..1075 of the GT (1920x1072) and the
                   matching rows 2..537 of the input (960x536), cropped from the files above (same
                   pixels, no padding needed by the CLI); the baseline upscaled from the cropped input
  json             manifest: source, start, frames, colours (and what was assumed), the source's
                   scene scores around and inside the clip, motion and brightness statistics,
                   every file (frames, size, md5), every command, and the checks of `verify`
Every file is all-intra FFV1 but the x264 one, starts at timestamp 0 at the source's exact frame
rate, and holds exactly N frames (checked).

`slice` cuts frames A..A+N-1 out of a clip made by `make` (its src, gt, <deg>.lr and
<deg>.bicubic files, or those named by --files; never the x264 file, which is long-GOP, nor the
crop files) into DIR2/NEW.*, re-encoded as the same all-intra FFV1 without any pixel
conversion. Several slices of one degraded clip thus share bit-identical inputs (a degradation
re-run on a shorter range would not: x264 decides per GOP). Every slice is checked frame by
frame: its framemd5 must equal frames A..A+N-1 of the parent file's. Its manifest NEW.json
names the parent and keeps the source and the absolute start (parent start + A), so `verify`
works on it (`--source` needs the src slice), and carries the parent's scene scores for its own
frames (its entry, inside and exit scores) and its statistics.

Statistics (from gt.mkv, Y = BT.709 luma of the full-range RGB, 8-bit scale): mean Y, the
fraction of near-black pixels (Y < 10), mean |dY| between consecutive frames (motion) and the
number of held transitions (mean |dY| < 0.5: anime drawn on twos/threes), luma Laplacian variance
(detail). Scene scores: ffmpeg's scdet, as sptenc runs it (on the decoded source format; score
= min(mafd, |mafd - previous mafd|), %), for the transitions into the clip, inside it and out.

`verify` checks (a) that cv2.VideoCapture, run by the given Python (the SeedVR2 venv's: cv2 only,
no torch), reads every *.lr.mkv bit-exactly as ffmpeg decodes it (decoded as stored, repacked by
numpy: no swscale on our side), (b) with --source, that frames S..S+N-1 of the source (framemd5 of a
plain decode, frames counted by the muxer, no filter) are src.mkv's frames, and (c) the frame
count of every file. `selftest` checks the colour chain against the BT.709 equations, the FFV1
round trips, and the extraction and a slice on a synthetic long-GOP H.264 file.

`grain` measures film grain (docs/numerics.md, Grain): Y = the BT.709 luma of the RGB in full-range
8-bit levels (a YUV file goes through its colour plan's zscale to 16-bit RGB first; frames taller
than 1080 rows are area-downscaled to 1080), the residual Y - GaussianBlur(Y, sigma 2 px), and its
standard deviation over the mask: the 30% of the picture where the blurred image's gradient (Sobel
magnitude) is lowest, among the pixels with 16 < blurred < 235 (none if 1000 pixels or fewer). Left
out of the mask: black bars, the rows (and columns) from each edge whose mean Y over the measured
frames is below 20, and the 8 rows (columns) next to them, or next to the frame's edge where there
is no bar. Blur and gradient are taken on the whole frame, whose reflected border shows no gradient
across it: a dark picture edge there (OSS 117 fades to black over its 3-4 outermost columns, several
Blu-rays carry a darker outermost line) would otherwise dominate the residual. Per frame, then the
median over the frames (25-75%). Frames: an RGB file (a clip's GT, a baseline, a master) every one,
or with --frames N, N spread over its middle 80%; a YUV source N (12 by default) at even times over
the middle 80% of its duration (clear of logos and end credits), each decoded with an input seek
(-ss before -i): a survey, not frame-exact. --every decodes every frame in order (frame-exact); the
measured frames' luma stays in memory until the bars are known: for clips.

Needs numpy (and OpenCV for `grain`), and ffmpeg/ffprobe with zimg (zscale), libx264 and ffv1
(FR_FFMPEG, FR_FFPROBE override the binaries found on PATH).
"""
import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from fractions import Fraction

import numpy as np

FFMPEG = os.environ.get("FR_FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = os.environ.get("FR_FFPROBE") or shutil.which("ffprobe") or "ffprobe"

FFV1 = ["-c:v", "ffv1", "-level", "3", "-g", "1", "-slices", "16", "-slicecrc", "1"]
THIRD = "0.3333333333333333"
MITCHELL = f"filter=bicubic:param_a={THIRD}:param_b={THIRD}"   # zimg: param_a = b, param_b = c
CATROM = "filter=bicubic:param_a=0:param_b=0.5"
KEYINT = 48
DEGRADATIONS = {
    "d1": {"crf": 20, "what": "zscale bicubic b=c=1/3 (Mitchell) x1/2 in YUV, x264 slow CRF 20"},
    "d2": {"crf": 26, "what": "swscale area x1/2 in YUV, x264 slow CRF 26"},
}
ZMATRIX = {"bt709": "709", "smpte170m": "170m", "bt470bg": "470bg", "bt2020nc": "2020_ncl"}
LOCATIONS = ("left", "center", "topleft", "top", "bottomleft", "bottom")
RGB_BITS = {"bgr0": 8, "gbrp": 8, "gbrp10le": 10, "gbrp12le": 12, "gbrp16le": 16}


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def run(cmd, what=None, record=None):
    log(f"$ {shlex.join(cmd)}")
    t0 = time.perf_counter()
    r = subprocess.run(cmd)
    if r.returncode:
        raise SystemExit(f"{what or cmd[0]} failed with exit status {r.returncode}")
    if record is not None:
        record.append({"cmd": shlex.join(cmd), "s": round(time.perf_counter() - t0, 1)})


def concat_opts(path):
    """A *.ffconcat source (a list of files of one codec, e.g. pre-split segments) is read with the
    concat demuxer: its frame index counts across the files."""
    return ["-f", "concat", "-safe", "0"] if path.endswith(".ffconcat") else []


def probe(path):
    r = subprocess.run([FFPROBE, "-v", "error", *concat_opts(path), "-select_streams", "v:0", "-show_entries",
                        "stream=codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,start_time,"
                        "color_space,color_primaries,color_transfer,color_range,chroma_location,field_order"
                        ":format=duration", "-of", "json", path], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"ffprobe {path}: {r.stderr.strip()}")
    j = json.loads(r.stdout)
    if not j.get("streams"):
        raise SystemExit(f"{path}: no video stream")
    st = j["streams"][0]
    st["duration"] = float(j.get("format", {}).get("duration") or 0)
    return st


def count_frames(path):
    r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_frames",
                        "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    try:
        return int(r.stdout.strip().split(",")[0])
    except ValueError:
        return None


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def ffmpeg_version():
    r = subprocess.run([FFMPEG, "-hide_banner", "-version"], capture_output=True, text=True)
    return r.stdout.splitlines()[0] if r.stdout else "?"


# ------------------------------------------------------------------ colours

def colour_plan(st):
    """What the source's YUV means: from its tags, else the usual assumptions (BT.709 for HD, BT.601
    below; limited range; chroma sited left, MPEG-2/H.264's default)."""
    assumed = []
    matrix = st.get("color_space")
    if matrix not in ZMATRIX:
        matrix = "bt709" if st["height"] > 576 else "smpte170m"
        assumed.append(f"matrix {matrix}")
    rng = st.get("color_range")
    if rng not in ("tv", "pc"):
        rng = "tv"
        assumed.append("limited range")
    loc = st.get("chroma_location")
    if loc not in LOCATIONS:
        loc = "left"
        assumed.append("chroma sited left")
    default = "bt709" if matrix == "bt709" else "smpte170m"
    prim, trc = st.get("color_primaries"), st.get("color_transfer")
    if prim in (None, "unknown", "unspecified", "reserved"):
        prim = default
        assumed.append(f"primaries {prim}")
    if trc in (None, "unknown", "unspecified", "reserved"):
        trc = default
        assumed.append(f"transfer {trc}")
    return {"matrix": matrix, "range": rng, "chroma_location": loc, "primaries": prim, "transfer": trc,
            "assumed": assumed}


def yuv_params(c):
    """setparams: tag YUV frames with the plan (an untagged source gets what was assumed)."""
    return (f"setparams=colorspace={c['matrix']}:range={c['range']}:color_primaries={c['primaries']}"
            f":color_trc={c['transfer']}:chroma_location={c['chroma_location']}")


def rgb_params(c):
    """setparams for RGB frames: identity matrix, full range, the plan's primaries and transfer."""
    return f"setparams=colorspace=gbr:range=pc:color_primaries={c['primaries']}:color_trc={c['transfer']}"


def rgb_tags(c):
    return ["-colorspace", "rgb", "-color_primaries", c["primaries"], "-color_trc", c["transfer"],
            "-color_range", "pc"]


def yuv_tags(c):
    return ["-colorspace", c["matrix"], "-color_primaries", c["primaries"], "-color_trc", c["transfer"],
            "-color_range", c["range"], "-chroma_sample_location", c["chroma_location"]]


def zscale_to_rgb(c, extra=""):
    """zscale: YUV of the plan -> RGB (full range), no primaries/transfer conversion. threads=1,
    libavfilter's generic per-filter option, here and in every zscale below: ffmpeg's slice
    threading (one slice per CPU by default) changes a 10-bit 4:2:0 source's RGB from 4 slices on
    (chroma_kernels.py upcheck); 8-bit sources and the resizes were measured unaffected, and are
    pinned alike."""
    z = "full" if c["range"] == "pc" else "limited"
    return (f"zscale=threads=1:matrixin={ZMATRIX[c['matrix']]}:rangein={z}:chromalin={c['chroma_location']}"
            ":dither=none" + (f":{extra}" if extra else ""))


# ------------------------------------------------------------------ scene scores

def parse_metadata(path):
    """metadata=mode=print file: one dict per frame, in order (pts_time, scd score/mafd, Y avg)."""
    frames = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("frame:"):
                d = {}
                for tok in line.split():
                    k, _, v = tok.partition(":")
                    if k == "pts_time":
                        d["pts_time"] = float(v)
                frames.append(d)
            elif "=" in line and frames:
                k, _, v = line.partition("=")
                key = {"lavfi.scd.score": "score", "lavfi.scd.mafd": "mafd",
                       "lavfi.signalstats.YAVG": "yavg"}.get(k)
                if key:
                    try:
                        frames[-1][key] = float(v)
                    except ValueError:
                        pass
    return frames


def y_full(yavg, st):
    """signalstats' YAVG (native range, 8-bit scale) -> full-range 8-bit levels."""
    if yavg is None:
        return None
    if st.get("color_range") == "pc":
        return yavg
    return (yavg - 16) * 255 / 219


def scene_scores(src, first, last, threads, yavg=True):
    """Frame-exact scdet scores of frames first..last (decode from the start, count). scdet needs
    two frames before to give the score a full scan gives (score = min(mafd, |mafd - previous|))."""
    a = max(0, first - 2)
    with tempfile.TemporaryDirectory() as d:
        meta = os.path.join(d, "scd.txt")
        chain = f"select='between(n,{a},{last})',scdet=t=10" + (",signalstats" if yavg else "") + \
                f",metadata=mode=print:file={meta}"
        run([FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-threads", str(threads), *concat_opts(src),
             "-i", src, "-map", "0:v:0", "-vf", chain, "-fps_mode", "passthrough", "-frames:v", str(last - a + 1),
             "-f", "null", "-"], "scdet")
        fr = parse_metadata(meta)
    out = []
    for k, f in enumerate(fr):
        i = a + k
        if i < first:
            continue
        out.append({"i": i, "pts_time": f.get("pts_time"), "score": f.get("score"), "mafd": f.get("mafd"),
                    "yavg": f.get("yavg"), "valid": i - a >= 2 or a == 0})
    return out


# ------------------------------------------------------------------ RGB decode and statistics

def rgb_frames(path, threads=16, ss=None, count=None):
    """Yield (RGB array [H, W, 3], bits) of a file. An RGB file as stored: planar gbrp* (16-bit masters)
    or packed bgr0 (FFV1's only 8-bit RGB layout: it takes no 8-bit gbrp); a YUV file in 16-bit RGB
    through its colour plan (tagged first, as `sheet` does). ss: input seek in seconds (-ss before -i,
    not frame-exact); count: at most that many frames."""
    st = probe(path)
    fmt = st["pix_fmt"]
    vf = []
    if fmt not in RGB_BITS:
        if not fmt.startswith(("yuv", "yuvj")):
            raise SystemExit(f"{path}: {fmt}, neither RGB nor planar YUV")
        plan = colour_plan(st)
        fmt = "gbrp16le"
        vf = ["-vf", f"{yuv_params(plan)},{zscale_to_rgb(plan)},format={fmt}"]
    W, H, bits = st["width"], st["height"], RGB_BITS[fmt]
    dtype = np.dtype(np.uint8) if bits == 8 else np.dtype("<u2")
    size = (4 if fmt == "bgr0" else 3) * W * H * dtype.itemsize
    proc = subprocess.Popen([FFMPEG, "-v", "error", "-nostdin", "-threads", str(threads),
                             *(["-ss", f"{ss:.3f}"] if ss is not None else []), *concat_opts(path), "-i", path,
                             "-map", "0:v:0", *vf, "-fps_mode", "passthrough",
                             *(["-frames:v", str(count)] if count else []), "-f", "rawvideo", "-pix_fmt", fmt, "-"],
                            stdout=subprocess.PIPE)
    done = False
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                done = True
                break
            x = np.frombuffer(buf, dtype)
            if fmt == "bgr0":
                yield x.reshape(H, W, 4)[..., 2::-1], bits
            else:
                g, b, r = x.reshape(3, H, W)
                yield np.stack([r, g, b], axis=-1), bits
    finally:
        if not done:
            proc.kill()  # closed early: a read-only decode, stop it without a broken pipe
        proc.stdout.close()
        proc.wait()


def luma709(rgb, bits):
    r, g, b = (rgb[..., k].astype(np.float32) for k in range(3))
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) * np.float32(255.0 / ((1 << bits) - 1))


def lap_var(y):
    lap = y[1:-1, :-2] + y[1:-1, 2:] + y[:-2, 1:-1] + y[2:, 1:-1] - 4 * y[1:-1, 1:-1]
    return float(lap.var())


def clip_stats(gt):
    ymean, dark, lapv, dy = [], [], [], []
    prev = None
    for rgb, bits in rgb_frames(gt):
        y = luma709(rgb, bits)
        ymean.append(float(y.mean()))
        dark.append(float((y < 10).mean()))
        lapv.append(lap_var(y))
        if prev is not None:
            dy.append(float(np.abs(y - prev).mean()))
        prev = y
    r = lambda v, n=3: round(float(v), n)  # noqa: E731
    return {"frames": len(ymean), "mean_y": r(np.mean(ymean), 2), "min_frame_y": r(min(ymean), 2),
            "max_frame_y": r(max(ymean), 2), "dark_fraction": r(np.mean(dark)),
            "mean_abs_dy": r(np.mean(dy)) if dy else None, "max_abs_dy": r(max(dy)) if dy else None,
            "held_transitions": int(sum(v < 0.5 for v in dy)), "lap_var": r(np.mean(lapv), 1),
            "per_transition_abs_dy": [r(v) for v in dy], "per_frame_y": [r(v, 2) for v in ymean]}


def file_info(path, expect=None):
    st = probe(path)
    n = count_frames(path)
    info = {"path": path, "pix_fmt": st["pix_fmt"], "size": [st["width"], st["height"]], "frames": n,
            "r_frame_rate": st.get("r_frame_rate"), "bytes": os.path.getsize(path), "md5": md5_file(path)}
    if expect is not None and n != expect:
        raise SystemExit(f"{path}: {n} frames, {expect} expected")
    return info


# ------------------------------------------------------------------ make

def manifest_path(out, name):
    return os.path.join(out, f"{name}.json")


def cmd_make(a):
    src, S, N, name, out, T = os.path.abspath(a.src), a.start, a.frames, a.name, a.out, a.threads
    os.makedirs(out, exist_ok=True)
    st = probe(src)
    if st["pix_fmt"] in RGB_BITS or not st["pix_fmt"].startswith(("yuv", "yuvj")):
        raise SystemExit(f"{src}: {st['pix_fmt']}, a planar YUV source is expected")
    c = colour_plan(st)
    fps = Fraction(st["r_frame_rate"])
    p = lambda suffix: os.path.join(out, f"{name}.{suffix}")  # noqa: E731
    mpath = manifest_path(out, name)
    man = {}
    if os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as f:
            man = json.load(f)
    same = man.get("source") == src and man.get("start") == S and man.get("frames") == N
    if not same or a.force:
        man = {"name": name, "source": src, "start": S, "frames": N, "fps": str(fps), "source_stream": st,
               "colours": c, "ffmpeg": ffmpeg_version(), "files": {}, "commands": []}
    cmds = man["commands"]
    files = man["files"]

    # 1. the clip, by decoding and counting; the scene scores of frames S-2 .. S+N in the same pass
    if "src" not in files or not os.path.exists(p("src.mkv")):
        a0 = max(0, S - 2)
        lead, end = S - a0, S + N
        with tempfile.TemporaryDirectory() as d:
            meta = os.path.join(d, "scd.txt")
            fc = (f"[0:v:0]select='between(n,{a0},{end})',split=2[c][s];"
                  f"[c]select='between(n,{lead},{lead + N - 1})',setpts=N/FRAME_RATE/TB,{yuv_params(c)}[clip];"
                  f"[s]scdet=t=10,metadata=mode=print:file={meta}[sc]")
            run([FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-threads", str(T), *concat_opts(src),
                 "-i", src, "-filter_complex", fc,
                 "-map", "[clip]", "-fps_mode", "passthrough", "-frames:v", str(N), *FFV1, "-threads", str(T),
                 "-pix_fmt", st["pix_fmt"], *yuv_tags(c), p("src.mkv"),
                 "-map", "[sc]", "-fps_mode", "passthrough", "-frames:v", str(end - a0 + 1), "-f", "null", "-"],
                "extraction", cmds)
            fr = parse_metadata(meta)
        idx = {a0 + k: f for k, f in enumerate(fr)}
        inside = [idx[i]["score"] for i in range(S + 1, S + N) if i in idx and "score" in idx[i]]
        man["scdet"] = {
            "entry": idx.get(S, {}).get("score") if S - a0 >= 2 or S == 0 else None,
            "inside_max": round(max(inside), 3) if inside else None,
            "inside_mean": round(float(np.mean(inside)), 3) if inside else None,
            "exit": idx.get(end, {}).get("score"),
            "inside": [round(v, 3) for v in inside],
        }
        man["pts_time_first"] = idx.get(S, {}).get("pts_time")
        man["pts_time_last"] = idx.get(S + N - 1, {}).get("pts_time")
        expect = round(S / fps, 3) if fps else None
        man["pts_check"] = {"expected_first": float(expect) if expect is not None else None,
                            "index_matches_time": (man["pts_time_first"] is not None and expect is not None
                                                   and abs(man["pts_time_first"] - float(S / fps)) < 0.5 / float(fps))}
        files["src"] = file_info(p("src.mkv"), N)
        # 2. the ground truth
        run([FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-i", p("src.mkv"), "-map", "0:v:0",
             "-vf", f"{zscale_to_rgb(c)},format=gbrp16le,{rgb_params(c)}", "-fps_mode", "passthrough",
             *FFV1, "-threads", str(T), "-pix_fmt", "gbrp16le", *rgb_tags(c), p("gt.mkv")], "ground truth", cmds)
        files["gt"] = file_info(p("gt.mkv"), N)
        man["stats"] = clip_stats(p("gt.mkv"))

    W, H = st["width"], st["height"]
    w, h = W // 2, H // 2
    H16, W16 = H // 16 * 16, W // 16 * 16
    top, left = (H - H16) // 2, (W - W16) // 2
    if a.crop and (top % 2 or left % 2):
        raise SystemExit("--crop: the crop offsets must be even to map onto the half-size input")
    if a.crop and "crop.gt" not in files:
        run([FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-i", p("gt.mkv"), "-map", "0:v:0",
             "-vf", f"crop={W16}:{H16}:{left}:{top},{rgb_params(c)}", "-fps_mode", "passthrough",
             *FFV1, "-threads", str(T), "-pix_fmt", "gbrp16le", *rgb_tags(c), p("crop.gt.mkv")], "crop GT", cmds)
        files["crop.gt"] = file_info(p("crop.gt.mkv"), N)
        man["crop"] = {"gt_rows": [top, top + H16], "gt_cols": [left, left + W16],
                       "lr_rows": [top // 2, (top + H16) // 2], "lr_cols": [left // 2, (left + W16) // 2]}

    zm = ZMATRIX[c["matrix"]]
    for deg in [d.strip() for d in a.degrade.split(",") if d.strip()]:
        if deg not in DEGRADATIONS:
            raise SystemExit(f"unknown degradation {deg!r} (choose from {', '.join(DEGRADATIONS)})")
        spec = DEGRADATIONS[deg]
        zr = "full" if c["range"] == "pc" else "limited"
        if deg == "d1":
            down = (f"zscale=threads=1:w={w}:h={h}:{MITCHELL}:matrixin={zm}:matrix={zm}:rangein={zr}:range=limited"
                    f":chromalin={c['chroma_location']}:chromal=left:dither=none")
        else:
            down = f"scale={w}:{h}:flags=area+accurate_rnd:out_range=tv"
        lr_tags = {"matrix": c["matrix"], "range": "tv", "chroma_location": "left", "primaries": c["primaries"],
                   "transfer": c["transfer"]}
        if deg not in files or a.force:
            run([FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-i", p("src.mkv"), "-map", "0:v:0",
                 "-vf", f"{down},format=yuv420p,{yuv_params(lr_tags)}", "-fps_mode", "passthrough",
                 "-c:v", "libx264", "-preset", "slow", "-crf", str(spec["crf"]), "-x264-params", f"keyint={KEYINT}",
                 "-threads", str(T), "-pix_fmt", "yuv420p", *yuv_tags(lr_tags), p(f"{deg}.x264.mkv")],
                f"{deg} encode", cmds)
            run([FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-threads", str(T),
                 "-i", p(f"{deg}.x264.mkv"), "-map", "0:v:0",
                 "-vf", f"{zscale_to_rgb(lr_tags)},format=gbrp,{rgb_params(c)}", "-fps_mode", "passthrough",
                 *FFV1, "-threads", str(T), "-pix_fmt", "bgr0", *rgb_tags(c), p(f"{deg}.lr.mkv")], f"{deg} input", cmds)
            run([FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-i", p(f"{deg}.lr.mkv"),
                 "-map", "0:v:0", "-vf", f"zscale=threads=1:w={W}:h={H}:{CATROM}:dither=none,format=gbrp16le,{rgb_params(c)}",
                 "-fps_mode", "passthrough", *FFV1, "-threads", str(T), "-pix_fmt", "gbrp16le", *rgb_tags(c),
                 p(f"{deg}.bicubic.mkv")], f"{deg} bicubic baseline", cmds)
            files[deg] = {"what": spec["what"], "keyint": KEYINT,
                          "x264": file_info(p(f"{deg}.x264.mkv"), N), "lr": file_info(p(f"{deg}.lr.mkv"), N),
                          "bicubic": file_info(p(f"{deg}.bicubic.mkv"), N)}
        if a.crop and f"{deg}.crop" not in files:
            run([FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-i", p(f"{deg}.lr.mkv"),
                 "-map", "0:v:0", "-vf", f"crop={W16 // 2}:{H16 // 2}:{left // 2}:{top // 2},{rgb_params(c)}",
                 "-fps_mode", "passthrough", *FFV1, "-threads", str(T), "-pix_fmt", "bgr0", *rgb_tags(c),
                 p(f"{deg}.crop.lr.mkv")], f"{deg} crop input", cmds)
            run([FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-i", p(f"{deg}.crop.lr.mkv"),
                 "-map", "0:v:0", "-vf", f"zscale=threads=1:w={W16}:h={H16}:{CATROM}:dither=none,format=gbrp16le,{rgb_params(c)}",
                 "-fps_mode", "passthrough", *FFV1, "-threads", str(T), "-pix_fmt", "gbrp16le", *rgb_tags(c),
                 p(f"{deg}.crop.bicubic.mkv")], f"{deg} crop bicubic baseline", cmds)
            files[f"{deg}.crop"] = {"lr": file_info(p(f"{deg}.crop.lr.mkv"), N),
                                    "bicubic": file_info(p(f"{deg}.crop.bicubic.mkv"), N)}
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1)
    s = man.get("stats", {})
    sc = man.get("scdet", {})
    print(f"{name}: frames {S}..{S + N - 1} of {src} ({man.get('pts_time_first')} s), {N} frames at {fps}")
    print(f"  colours {c['matrix']} {c['range']} chroma {c['chroma_location']}"
          + (f" (assumed: {', '.join(c['assumed'])})" if c["assumed"] else ""))
    print(f"  scdet: entry {sc.get('entry')}, inside max {sc.get('inside_max')} mean {sc.get('inside_mean')}, "
          f"exit {sc.get('exit')}; index/time check {man.get('pts_check', {}).get('index_matches_time')}")
    print(f"  mean Y {s.get('mean_y')} (frames {s.get('min_frame_y')}..{s.get('max_frame_y')}), dark {s.get('dark_fraction')}, "
          f"mean |dY| {s.get('mean_abs_dy')} (max {s.get('max_abs_dy')}, held {s.get('held_transitions')}), "
          f"Laplacian var {s.get('lap_var')}")
    for k, v in files.items():
        for sub in ([v] if "path" in v else [x for x in v.values() if isinstance(x, dict) and "path" in x]):
            print(f"  {os.path.basename(sub['path'])}: {sub['pix_fmt']} {sub['size'][0]}x{sub['size'][1]}, "
                  f"{sub['frames']} frames, {sub['bytes'] / 2**20:.1f} MiB")
    print(f"  manifest {mpath}")


# ------------------------------------------------------------------ slice

def intra_files(man):
    """(manifest key path, file suffix, info) of a clip's all-intra FFV1 files that a slice can cut:
    src, gt, and every degradation's lr and bicubic (not the x264 file, not the crop files)."""
    out = []
    for k, v in man["files"].items():
        if "path" in v:
            if k in ("src", "gt"):
                out.append(((k,), k, v))
        elif k in DEGRADATIONS:
            for sub in ("lr", "bicubic"):
                if isinstance(v.get(sub), dict) and "path" in v[sub]:
                    out.append(((k, sub), f"{k}.{sub}", v[sub]))
    return out


def cmd_slice(a):
    base = a.clip[:-5] if a.clip.endswith(".json") else a.clip
    pdir, pname = os.path.dirname(base) or ".", os.path.basename(base)
    ppath = manifest_path(pdir, pname)
    with open(ppath, encoding="utf-8") as f:
        pman = json.load(f)
    N0, A, N, T = pman["frames"], a.first, a.frames, a.threads
    if A < 0 or N < 1 or A + N > N0:
        raise SystemExit(f"slice {A}..{A + N - 1} is not inside {pname}'s {N0} frames")
    out = a.out or pdir
    os.makedirs(out, exist_ok=True)
    if os.path.abspath(out) == os.path.abspath(pdir) and a.name == pname:
        raise SystemExit("the slice would overwrite its parent: give another --name")
    c = pman["colours"]
    jobs = intra_files(pman)
    if a.files:
        want = [x.strip() for x in a.files.split(",") if x.strip()]
        missing = [w for w in want if w not in {j[1] for j in jobs}]
        if missing:
            raise SystemExit(f"{pname} has no all-intra file {', '.join(missing)} (it has {', '.join(j[1] for j in jobs)})")
        jobs = [j for j in jobs if j[1] in want]
    p = lambda suffix: os.path.join(out, f"{a.name}.{suffix}")  # noqa: E731
    man = {"name": a.name, "source": pman["source"], "start": pman["start"] + A, "frames": N, "fps": pman["fps"],
           "source_stream": pman.get("source_stream"), "colours": c, "ffmpeg": ffmpeg_version(),
           "slice_of": {"name": pname, "manifest": os.path.abspath(ppath), "first": A, "parent_start": pman["start"],
                        "parent_frames": N0},
           "files": {}, "commands": [], "slice_check": {}}
    for key, suffix, info in jobs:
        src = info["path"]
        fmt = probe(src)["pix_fmt"]
        rgb = fmt in RGB_BITS
        run([FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-threads", str(T), "-i", src,
             "-map", "0:v:0", "-vf", f"select='between(n,{A},{A + N - 1})',setpts=N/FRAME_RATE/TB,"
                                     f"{rgb_params(c) if rgb else yuv_params(c)}",
             "-fps_mode", "passthrough", "-frames:v", str(N), *FFV1, "-threads", str(T), "-pix_fmt", fmt,
             *(rgb_tags(c) if rgb else yuv_tags(c)), p(f"{suffix}.mkv")], f"slice {suffix}", man["commands"])
        parent_h = framemd5(src, A + N, T)[A:A + N]
        slice_h = framemd5(p(f"{suffix}.mkv"), None, T)
        same = sum(x == y for x, y in zip(parent_h, slice_h))
        exact = len(parent_h) == len(slice_h) == same == N
        man["slice_check"][suffix] = {"equal_frames": same, "frames": len(slice_h), "exact": exact,
                                      "first_md5": slice_h[0] if slice_h else None}
        print(f"  {a.name}.{suffix}.mkv: frames {A}..{A + N - 1} of {os.path.basename(src)}, framemd5 {same}/{N} "
              f"equal: {'OK' if exact else 'DIFFERS'}")
        if not exact:
            raise SystemExit(f"slice {suffix}: not frame-exact ({same}/{N} frames equal)")
        fi = file_info(p(f"{suffix}.mkv"), N)
        if len(key) == 1:
            man["files"][key[0]] = fi
        else:
            d = man["files"].setdefault(key[0], {k: v for k, v in pman["files"][key[0]].items()
                                                 if not isinstance(v, dict)})
            d[key[1]] = fi
    # the parent's scene scores of the slice's own frames: entry (into frame A), inside, exit
    sc = pman.get("scdet") or {}
    inside = sc.get("inside") or []
    if len(inside) == N0 - 1:
        entry = sc.get("entry") if A == 0 else inside[A - 1]
        ins = inside[A:A + N - 1]
        man["scdet"] = {"entry": entry, "inside_max": round(max(ins), 3) if ins else None,
                        "inside_mean": round(float(np.mean(ins)), 3) if ins else None,
                        "exit": sc.get("exit") if A + N == N0 else inside[A + N - 1], "inside": ins,
                        "inside_max_at": 1 + int(np.argmax(ins)) if ins else None}  # slice frame index
    if "gt" in man["files"]:
        man["stats"] = clip_stats(p("gt.mkv"))
    with open(manifest_path(out, a.name), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1)
    sc = man.get("scdet", {})
    print(f"{a.name}: frames {A}..{A + N - 1} of {pname} = source frames {man['start']}..{man['start'] + N - 1}, "
          f"{N} frames; scdet entry {sc.get('entry')}, inside max {sc.get('inside_max')} (slice frame "
          f"{sc.get('inside_max_at')}), exit {sc.get('exit')}; manifest {manifest_path(out, a.name)}")


# ------------------------------------------------------------------ verify

CV2_READ = r"""
import hashlib, json, sys
import cv2
cap = cv2.VideoCapture(sys.argv[1])
info = {"cv2": cv2.__version__, "backend": cap.getBackendName() if cap.isOpened() else None,
        "frame_count_prop": cap.get(cv2.CAP_PROP_FRAME_COUNT), "fps_prop": cap.get(cv2.CAP_PROP_FPS), "md5": []}
while True:
    ok, f = cap.read()
    if not ok:
        break
    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)  # what the CLI does
    info["shape"] = list(rgb.shape)
    info["md5"].append(hashlib.md5(rgb.tobytes()).hexdigest())
print(json.dumps(info))
"""


def rgb24_md5s(path):
    """md5 of every frame as packed RGB24, decoded as stored and repacked by numpy (no swscale)."""
    out = []
    for rgb, bits in rgb_frames(path):
        if bits != 8:
            raise SystemExit(f"{path}: {bits}-bit, the CLI inputs are 8-bit")
        out.append(hashlib.md5(np.ascontiguousarray(rgb).tobytes()).hexdigest())
    return out


def framemd5(path, count=None, threads=16):
    cmd = [FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-threads", str(threads), *concat_opts(path),
           "-i", path, "-map", "0:v:0", "-fps_mode", "passthrough"]
    if count:
        cmd += ["-frames:v", str(count)]
    cmd += ["-f", "framemd5", "-"]
    log(f"$ {shlex.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"framemd5 {path}: {r.stderr.strip()}")
    return [ln.split(",")[-1].strip() for ln in r.stdout.splitlines() if ln and not ln.startswith("#")]


def cmd_verify(a):
    base = a.clip[:-5] if a.clip.endswith(".json") else a.clip
    out, name = os.path.dirname(base) or ".", os.path.basename(base)
    mpath = manifest_path(out, name)
    with open(mpath, encoding="utf-8") as f:
        man = json.load(f)
    N, S = man["frames"], man["start"]
    ver = man.setdefault("verify", {})
    ok = True
    # frame counts
    counts = {}
    for k, v in man["files"].items():
        for sub in ([v] if "path" in v else [x for x in v.values() if isinstance(x, dict) and "path" in x]):
            n = count_frames(sub["path"])
            counts[os.path.basename(sub["path"])] = n
            ok &= n == N
    ver["frame_counts"] = counts
    print(f"{name}: frame counts {'all ' + str(N) if all(n == N for n in counts.values()) else counts}")
    # cv2 reads the CLI inputs bit-exactly
    if a.cv2_python:
        lrs = [sub["path"] for v in man["files"].values() if isinstance(v, dict)
               for key, sub in v.items() if key == "lr" and isinstance(sub, dict)]
        ver["cv2"] = {}
        for lr in lrs:
            r = subprocess.run([a.cv2_python, "-c", CV2_READ, lr], capture_output=True, text=True)
            if r.returncode:
                raise SystemExit(f"cv2 read failed: {r.stderr.strip()}")
            info = json.loads(r.stdout.strip().splitlines()[-1])
            ref = rgb24_md5s(lr)
            same = sum(x == y for x, y in zip(info["md5"], ref))
            exact = same == len(ref) == len(info["md5"]) == N
            ok &= exact
            ver["cv2"][os.path.basename(lr)] = {"bit_exact": exact, "frames_cv2": len(info["md5"]),
                                                "frames_ffmpeg": len(ref), "equal_frames": same,
                                                **{k: info.get(k) for k in ("cv2", "backend", "frame_count_prop",
                                                                            "fps_prop", "shape")}}
            print(f"  cv2 {info.get('cv2')} ({info.get('backend')}) reads {os.path.basename(lr)}: {same}/{len(ref)} "
                  f"frames bit-exact vs ffmpeg's planar decode ({len(info['md5'])} read, "
                  f"CAP_PROP_FRAME_COUNT {info.get('frame_count_prop')}, shape {info.get('shape')}): "
                  f"{'OK' if exact else 'DIFFERS'}")
    # the clip is frames S..S+N-1 of the source
    if a.source:
        t0 = time.perf_counter()
        src_h = framemd5(man["source"], S + N)
        clip_h = framemd5(os.path.join(out, f"{name}.src.mkv"))
        pick = src_h[S:S + N]
        same = sum(x == y for x, y in zip(pick, clip_h))
        exact = len(pick) == N and len(clip_h) == N and same == N
        ok &= exact
        # where else do the first and last frames appear in the decoded source (off-by-one guard)
        first_at = [i for i, hsh in enumerate(src_h) if hsh == clip_h[0]] if clip_h else []
        last_at = [i for i, hsh in enumerate(src_h) if hsh == clip_h[-1]] if clip_h else []
        ver["source"] = {"frames_hashed": len(src_h), "equal_frames": same, "exact": exact,
                         "first_frame_found_at": first_at[:10], "last_frame_found_at": last_at[:10],
                         "first_md5": clip_h[0] if clip_h else None, "last_md5": clip_h[-1] if clip_h else None,
                         "s": round(time.perf_counter() - t0, 1)}
        print(f"  source framemd5 frames {S}..{S + N - 1} vs {name}.src.mkv: {same}/{N} equal; first frame found at "
              f"source index {first_at[:5]}, last at {last_at[:5]}: {'OK' if exact else 'DIFFERS'}")
    ver["ok"] = bool(ok)
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1)
    print(f"  verify {'OK' if ok else 'FAILED'}")
    sys.exit(0 if ok else 1)


# ------------------------------------------------------------------ scan / scores

def shots_from(frames, threshold):
    """Split a scored frame list at the cuts (score >= threshold): [(start pos, end pos)]."""
    cuts = [k for k, f in enumerate(frames) if k >= 2 and f.get("score", 0) >= threshold]
    bounds = [0] + cuts + [len(frames)]
    return [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)], cuts


def cmd_scan(a):
    src = a.src
    st = probe(src)
    fps = Fraction(st["r_frame_rate"])
    start = float(st.get("start_time") or 0)
    with tempfile.TemporaryDirectory() as d:
        meta = os.path.join(d, "scd.txt")
        cmd = [FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-threads", str(a.threads)]
        if a.ss:
            cmd += ["-ss", str(a.ss)]
        cmd += ["-copyts"]
        if a.duration:
            cmd += ["-t", str(a.duration)]
        cmd += [*concat_opts(src), "-i", src, "-map", "0:v:0",
                "-vf", f"scdet=t=10,signalstats,metadata=mode=print:file={meta}",
                "-fps_mode", "passthrough", "-f", "null", "-"]
        run(cmd, "scan")
        fr = parse_metadata(meta)
    for f in fr:
        f["i"] = round((f.get("pts_time", 0) - start) * fps)
        f["y"] = y_full(f.get("yavg"), st)
    N, soft, hard, margin = a.frames, a.soft, a.hard, a.margin
    shots, cuts = shots_from(fr, a.cut)
    rows = []
    for s, e in shots:
        L = e - s
        sc = [fr[k].get("score", 0) for k in range(s + 1, e) if k >= 2]
        best = None
        for w0 in range(s + margin, e - margin - N + 1):
            inside = [fr[k].get("score", 0) for k in range(w0 + 1, w0 + N)]
            if max(inside) >= soft:
                continue
            mot = float(np.mean([fr[k].get("mafd", 0) for k in range(w0 + 1, w0 + N)]))
            ys = [fr[k]["y"] for k in range(w0, w0 + N) if fr[k].get("y") is not None]
            cand = (mot, w0, max(inside), float(np.mean(ys)) if ys else None, min(ys) if ys else None)
            if best is None or cand[0] > best[0]:
                best = cand
        rows.append({"start": fr[s]["i"], "time": round(fr[s].get("pts_time", 0), 3), "len": L,
                     "max_inside": round(max(sc), 2) if sc else None,
                     "mafd": round(float(np.mean([fr[k].get("mafd", 0) for k in range(s + 1, e)])), 3) if L > 1 else None,
                     "y": round(float(np.mean([fr[k]["y"] for k in range(s, e) if fr[k].get("y") is not None])), 1)
                     if any(fr[k].get("y") is not None for k in range(s, e)) else None,
                     "window": None if best is None else {"start": fr[best[1]]["i"], "mafd": round(best[0], 3),
                                                          "max_score": round(best[2], 2),
                                                          "y": round(best[3], 1) if best[3] is not None else None,
                                                          "min_y": round(best[4], 1) if best[4] is not None else None}})
    cut_rows = []
    for k, (s, e) in enumerate(shots[1:], 1):
        before, after = shots[k - 1][1] - shots[k - 1][0], e - s
        sc = fr[s].get("score", 0)
        if sc < hard:
            continue
        cut_rows.append({"cut": fr[s]["i"], "time": round(fr[s].get("pts_time", 0), 3), "score": round(sc, 2),
                         "before": before, "after": after,
                         "y_before": round(fr[s - 1]["y"], 1) if fr[s - 1].get("y") is not None else None,
                         "y_after": round(fr[s]["y"], 1) if fr[s].get("y") is not None else None,
                         "max_other": round(max([fr[j].get("score", 0) for j in range(max(2, s - a.before), min(len(fr), s + a.after))
                                                 if j != s] or [0]), 2)})
    res = {"source": src, "ss": a.ss, "duration": a.duration, "frames_scanned": len(fr), "fps": str(fps),
           "cut_threshold": a.cut, "shots": rows, "hard_cuts": cut_rows}
    print(f"{src}: {len(fr)} frames scanned from {a.ss or 0} s, {len(shots)} shots (cuts at score >= {a.cut}); "
          f"indices from timestamps (approximate: confirm with `scores`)")
    print(f"\n| Shot start | Time (s) | Frames | Max score inside | Mean mafd | Mean Y | Best {N}-frame window "
          f"(start, mafd, max score, mean Y, min Y) |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        wdw = r["window"]
        wtxt = "–" if wdw is None else f"{wdw['start']}, {wdw['mafd']}, {wdw['max_score']}, {wdw['y']}, {wdw['min_y']}"
        print(f"| {r['start']} | {r['time']} | {r['len']} | {r['max_inside']} | {r['mafd']} | {r['y']} | {wtxt} |")
    print(f"\nHard cuts (score >= {hard}), shot lengths around them, max other score in [cut-{a.before}, cut+{a.after}):\n")
    print("| Cut (first frame of the new shot) | Time (s) | Score | Frames before | Frames after | Y before / after | Max other score |")
    print("|---|---|---|---|---|---|---|")
    for r in cut_rows:
        print(f"| {r['cut']} | {r['time']} | {r['score']} | {r['before']} | {r['after']} | {r['y_before']} / {r['y_after']} "
              f"| {r['max_other']} |")
    if a.json:
        res["per_frame"] = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.items()} for f in fr]
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(res, f)


def cmd_scores(a):
    st = probe(a.src)
    sc = scene_scores(a.src, a.first, a.last, a.threads)
    print(f"{a.src}: frame-exact scdet scores (decoded from the start), frames {a.first}..{a.last}")
    print("\n| Frame | pts_time | Score | mafd | Y (full range) |")
    print("|---|---|---|---|---|")
    for f in sc:
        y = y_full(f.get("yavg"), st)
        print(f"| {f['i']} | {f.get('pts_time')} | {f.get('score')} | {f.get('mafd')} | "
              f"{round(y, 1) if y is not None else '–'} |")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"source": a.src, "first": a.first, "last": a.last, "frames": sc}, f)


# ------------------------------------------------------------------ sheet

def cmd_sheet(a):
    idx = [int(x) for x in a.frames.split(",") if x.strip()]
    st = probe(a.video)
    W = a.width
    H = round(W * st["height"] / st["width"] / 2) * 2
    if st["pix_fmt"] in RGB_BITS:
        conv = f"zscale=threads=1:w={W}:h={H}:filter=spline36:dither=none"
    else:  # tag the frames first: zimg finds no path from an untagged source's unknown transfer
        plan = colour_plan(st)
        conv = f"{yuv_params(plan)},{zscale_to_rgb(plan, f'w={W}:h={H}:filter=spline36')}"
    sel = "+".join(f"eq(n,{i})" for i in idx)
    cols = a.columns or len(idx)
    rows = -(-len(idx) // cols)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    run([FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-threads", str(a.threads),
         *concat_opts(a.video), "-i", a.video, "-map", "0:v:0", "-vf", f"select='{sel}',{conv},format=rgb24,tile=layout={cols}x{rows}:padding=6:color=white",
         "-fps_mode", "passthrough", "-frames:v", "1", "-update", "1", a.out], "sheet")
    print(f"{a.out}: frames {idx} of {a.video}, {W}x{H} each")


# ------------------------------------------------------------------ grain

GRAIN = {"sigma_px": 2.0, "smoothest_pct": 30, "blurred_range": [16, 235], "min_mask_px": 1000,
         "bar_mean_y_below": 20, "margin_px": 8, "max_rows": 1080}


def spread(n):
    """Where `grain` measures n frames, as fractions of the file: evenly over its middle 80%."""
    return [0.5] if n == 1 else [0.1 + 0.8 * k / (n - 1) for k in range(n)]


def probe_luma(rgb, bits):
    """Full-range BT.709 luma, 8-bit levels, area-downscaled to GRAIN['max_rows'] rows when taller."""
    import cv2
    y = luma709(rgb, bits)
    h, w, m = *y.shape, GRAIN["max_rows"]
    return y if h <= m else cv2.resize(y, (round(w * m / h / 2) * 2, m), interpolation=cv2.INTER_AREA)


def picture_window(lumas):
    """(bars top, bottom, left, right; mask window r0, r1, c0, c1): the bars are the rows (columns) from
    each edge whose mean Y over the frames is below GRAIN['bar_mean_y_below']; the window leaves them out
    and the GRAIN['margin_px'] rows (columns) next to them, or next to the frame's edge where there is no
    bar."""
    lo, m = GRAIN["bar_mean_y_below"], GRAIN["margin_px"]

    def edges(mean):
        pic = np.nonzero(mean >= lo)[0]
        return (int(pic[0]), int(len(mean) - 1 - pic[-1])) if len(pic) else (0, 0)
    top, bottom = edges(np.mean([y.mean(axis=1) for y in lumas], axis=0))
    left, right = edges(np.mean([y.mean(axis=0) for y in lumas], axis=0))
    H, W = lumas[0].shape
    return (top, bottom, left, right), (top + m, H - bottom - m, left + m, W - right - m)


def grain_frame(y, win):
    """(grain, mask pixels, mean blurred Y over the valid pixels) of one frame's luma; grain None when the
    mask has GRAIN['min_mask_px'] pixels or fewer."""
    import cv2
    b = cv2.GaussianBlur(y, (0, 0), GRAIN["sigma_px"])
    g = np.hypot(cv2.Sobel(b, cv2.CV_32F, 1, 0), cv2.Sobel(b, cv2.CV_32F, 0, 1))
    r0, r1, c0, c1 = win
    ok = np.zeros(y.shape, bool)
    ok[r0:r1, c0:c1] = True
    lo, hi = GRAIN["blurred_range"]
    ok &= (b > lo) & (b < hi)
    if not ok.any():
        return None, 0, None
    m = ok & (g <= np.percentile(g[ok], GRAIN["smoothest_pct"]))
    n = int(m.sum())
    return (float((y - b)[m].std(dtype=np.float64)) if n > GRAIN["min_mask_px"] else None), n, float(b[ok].mean())


def cmd_grain(a):
    import cv2
    cv2.setNumThreads(a.threads)
    if a.sigma is not None:  # other scales of the same measure (e.g. 1 px: the finest grain)
        GRAIN["sigma_px"] = a.sigma
    if a.share is not None:  # a smaller share reads the very smoothest areas only
        GRAIN["smoothest_pct"] = a.share
    t0 = time.perf_counter()
    st = probe(a.file)
    rgb = st["pix_fmt"] in RGB_BITS
    frames = []  # (frame index, seek time, luma)
    if rgb or a.every:
        n = count_frames(a.file) if a.frames else None
        want = None if n is None else sorted({round((n - 1) * f) for f in spread(a.frames)})
        how = "every frame" if want is None else f"frames {want} of {n}"
        for i, (x, bits) in enumerate(rgb_frames(a.file, a.threads)):
            if want is None or i in want:
                frames.append((i, None, probe_luma(x, bits)))
    else:
        if not st["duration"]:
            raise SystemExit(f"{a.file}: no duration to spread the frames over (--every decodes every frame)")
        N = a.frames or 12
        how = f"{N} frames over the middle 80% of {st['duration']:.1f} s, input seeks (a survey, not frame-exact)"
        for f in spread(N):
            got = list(rgb_frames(a.file, a.threads, ss=st["duration"] * f, count=1))
            frames.append((None, st["duration"] * f, probe_luma(*got[0]) if got else None))
    lumas = [y for _, _, y in frames if y is not None]
    if not lumas:
        raise SystemExit(f"{a.file}: no frame decoded")
    (top, bottom, left, right), win = picture_window(lumas)
    H, W = lumas[0].shape
    plan = None if rgb else colour_plan(st)
    col = "" if plan is None else f"; colours {plan['matrix']} {plan['range']}" + (
        f" (assumed: {', '.join(plan['assumed'])})" if plan["assumed"] else "")
    print(f"{a.file}: {st['pix_fmt']} {st['width']}x{st['height']}, measured at {W}x{H}; {how}{col}")
    print(f"  bars: top {top}, bottom {bottom}, left {left}, right {right}; mask window rows {win[0]}..{win[1] - 1},"
          f" columns {win[2]}..{win[3] - 1}")
    rows, vals, mys = [], [], []
    for i, t, y in frames:
        g, npx, my = grain_frame(y, win) if y is not None else (None, 0, None)
        vals += [] if g is None else [g]
        mys += [] if my is None else [my]
        rows.append({"frame": i, "t": None if t is None else round(t, 3), "grain": None if g is None else round(g, 4),
                     "mask_px": npx, "mean_y": None if my is None else round(my, 2)})
        where = f"frame {i}" if i is not None else f"t {t:.3f} s"
        print(f"  {where}: " + ("no frame decoded" if y is None else
                                f"grain {g:.3f}" if g is not None else f"no grain (mask {npx} px)")
              + (f", mask {npx} px, mean Y {my:.1f}" if g is not None else ""))
    v = np.array(vals)
    res = {"file": a.file, "pix_fmt": st["pix_fmt"], "size": [st["width"], st["height"]], "measured_size": [W, H],
           "colours": plan, "frames_how": how, "params": GRAIN,
           "bars": {"top": top, "bottom": bottom, "left": left, "right": right},
           "window": {"rows": [win[0], win[1]], "columns": [win[2], win[3]]},
           "frames": rows, "n": int(len(v)), "median": None, "q25": None, "q75": None,
           "mean_y": round(float(np.mean(mys)), 2) if mys else None, "ffmpeg": ffmpeg_version()}
    if len(v):
        res.update(median=round(float(np.median(v)), 4), q25=round(float(np.percentile(v, 25)), 4),
                   q75=round(float(np.percentile(v, 75)), 4))
    res["seconds"] = round(time.perf_counter() - t0, 1)
    print(f"  grain {res['median']:.2f} ({res['q25']:.2f}-{res['q75']:.2f}) over {len(v)} of {len(rows)} frames, "
          f"mean Y {res['mean_y']}, {res['seconds']} s" if len(v) else f"  no grain measured ({res['seconds']} s)")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)


# ------------------------------------------------------------------ selftest

def bt709_rgb(y, u, v):
    """BT.709 limited-range 8-bit YUV -> full-range RGB in [0, 1] (the equations)."""
    yn, cb, cr = (y - 16) / 219, (u - 128) / 224, (v - 128) / 224
    r = yn + 1.5748 * cr
    b = yn + 1.8556 * cb
    g = (yn - 0.2126 * r - 0.0722 * b) / 0.7152
    return np.clip(np.stack([r, g, b], -1), 0, 1)


def cmd_selftest(a):
    ok = True
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as d:
        # 1. colour chain: flat 32x32 patches of known BT.709 limited YUV, 4:2:0, read at patch centres
        P, cols, rws = 32, 8, 4
        vals = [(16, 128, 128), (235, 128, 128), (126, 128, 128), (81, 90, 240), (145, 54, 34), (41, 240, 110),
                (210, 16, 146), (170, 166, 16)] + [tuple(int(x) for x in rng.integers([16, 16, 16], [236, 241, 241]))
                                                   for _ in range(cols * rws - 8)]
        Y = np.zeros((rws * P, cols * P), np.uint8)
        U = np.zeros((rws * P // 2, cols * P // 2), np.uint8)
        V = np.zeros_like(U)
        for k, (y, u, v) in enumerate(vals):
            r, c = divmod(k, cols)
            Y[r * P:(r + 1) * P, c * P:(c + 1) * P] = y
            U[r * P // 2:(r + 1) * P // 2, c * P // 2:(c + 1) * P // 2] = u
            V[r * P // 2:(r + 1) * P // 2, c * P // 2:(c + 1) * P // 2] = v
        raw = os.path.join(d, "patches.yuv")
        with open(raw, "wb") as f:
            for _ in range(2):
                f.write(Y.tobytes() + U.tobytes() + V.tobytes())
        src = os.path.join(d, "patches.mkv")
        run([FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
             "-s", f"{cols * P}x{rws * P}", "-framerate", "24000/1001", "-i", raw, "-c:v", "ffv1", "-pix_fmt", "yuv420p",
             src], "patches")  # untagged on purpose: the plan must assume BT.709 / limited / left
        st = probe(src)
        st["height"] = 1080  # treat it as HD for the matrix assumption
        c = colour_plan(st)
        want = np.array([bt709_rgb(*map(float, v)) for v in vals])
        for fmt, bits, store in (("gbrp16le", 16, "gbrp16le"), ("gbrp", 8, "bgr0")):
            out = os.path.join(d, f"rgb.{fmt}.mkv")
            run([FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-i", src, "-vf",
                 f"{zscale_to_rgb(c)},format={fmt},{rgb_params(c)}", *FFV1, "-pix_fmt", store, *rgb_tags(c), out], "to RGB")
            rgb, _ = next(rgb_frames(out))
            got = np.array([rgb[k // cols * P + P // 2, k % cols * P + P // 2] for k in range(len(vals))], float)
            scale = (1 << bits) - 1
            err = np.abs(got - want * scale)
            good = err.max() <= 1.0
            ok &= good
            print(f"YUV 4:2:0 BT.709 limited -> {fmt} (zscale, untagged input, assumed {', '.join(c['assumed'])}): "
                  f"max |error| vs the BT.709 equations {err.max():.3f} LSB over {len(vals)} patches: "
                  f"{'OK' if good else 'FAILED'}")
        # 2. FFV1 RGB round trips (8-bit: gbrp in, stored bgr0, as the CLI inputs are)
        for fmt, bits, store in (("gbrp", 8, "bgr0"), ("gbrp16le", 16, "gbrp16le")):
            x = rng.integers(0, 1 << bits, size=(3, 3, 48, 64), dtype=np.uint16 if bits > 8 else np.uint8)
            rawp = os.path.join(d, f"rand.{fmt}")
            x.astype("<u2" if bits > 8 else np.uint8).tofile(rawp)
            out = os.path.join(d, f"rand.{fmt}.mkv")
            run([FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", fmt,
                 "-s", "64x48", "-framerate", "24", "-i", rawp, *FFV1, "-pix_fmt", store, out], "round trip")
            back = np.stack([f for f, _ in rgb_frames(out)])  # [T, H, W, (R, G, B)]
            want = np.stack([x[:, 2], x[:, 0], x[:, 1]], axis=-1)  # gbrp planes G, B, R -> R, G, B
            good = back.shape == want.shape and np.array_equal(back, want)
            ok &= good
            print(f"FFV1 {fmt} -> {store} round trip: {'bit-exact' if good else 'DIFFERS'}")
        # 3. extraction on a synthetic long-GOP H.264 file with B-frames: decode-and-count vs plain framemd5
        syn = os.path.join(d, "syn.mkv")
        run([FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
             "testsrc2=size=320x180:rate=24000/1001:duration=12", "-c:v", "libx264", "-preset", "fast", "-g", "120",
             "-bf", "3", "-pix_fmt", "yuv420p", syn], "synthetic source")
        all_h = framemd5(syn)
        S, N = 133, 45
        sa = argparse.Namespace(src=syn, start=S, frames=N, name="syn", out=d, threads=4, degrade="d1", crop=True,
                                force=False)
        cmd_make(sa)
        clip_h = framemd5(os.path.join(d, "syn.src.mkv"))
        good = clip_h == all_h[S:S + N]
        ok &= good
        print(f"extraction of frames {S}..{S + N - 1} of a {len(all_h)}-frame long-GOP H.264 file: "
              f"{'identical to its plain decode (framemd5)' if good else 'DIFFERS'}")
        # 4. slices of the made clip: every all-intra file, frame-exact against the parent's frames
        A, M = 7, 11
        cmd_slice(argparse.Namespace(clip=os.path.join(d, "syn"), first=A, frames=M, name="syn-slice", out=d,
                                     files="", threads=4))
        with open(manifest_path(d, "syn-slice"), encoding="utf-8") as f:
            sm = json.load(f)
        good = (all(v["exact"] for v in sm["slice_check"].values()) and len(sm["slice_check"]) == 4
                and sm["start"] == S + A and framemd5(os.path.join(d, "syn-slice.src.mkv")) == all_h[S + A:S + A + M])
        ok &= good
        print(f"slice of frames {A}..{A + M - 1} of the clip ({', '.join(sm['slice_check'])}): "
              f"{'frame-exact, and src = source frames ' + str(S + A) + '..' + str(S + A + M - 1) if good else 'DIFFERS'}")
    print("selftest", "ok" if ok else "FAILED")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="candidate shots and hard cuts in a time range (seek: approximate indices)")
    s.add_argument("src")
    s.add_argument("--ss", type=float, default=0, help="start time (s), input seek")
    s.add_argument("--duration", type=float, default=0, help="seconds to scan (default: to the end)")
    s.add_argument("--frames", type=int, default=45, help="clip length the best window is searched for")
    s.add_argument("--cut", type=float, default=10, help="scene score that splits shots (sptenc: 10)")
    s.add_argument("--soft", type=float, default=5, help="max score allowed inside a window (fades, flashes)")
    s.add_argument("--hard", type=float, default=30, help="score of the hard cuts listed")
    s.add_argument("--margin", type=int, default=2, help="frames kept away from the shot's ends")
    s.add_argument("--before", type=int, default=30, help="hard cuts: frames before the cut to check")
    s.add_argument("--after", type=int, default=60, help="hard cuts: frames after the cut to check")
    s.add_argument("--threads", type=int, default=16)
    s.add_argument("--json")
    s = sub.add_parser("scores", help="frame-exact scene scores of frames FIRST..LAST")
    s.add_argument("src")
    s.add_argument("--first", type=int, required=True)
    s.add_argument("--last", type=int, required=True)
    s.add_argument("--threads", type=int, default=16)
    s.add_argument("--json")
    s = sub.add_parser("make", help="ground truth, degraded inputs and baselines of one clip")
    s.add_argument("src")
    s.add_argument("--start", type=int, required=True, help="first frame (0-based decode order)")
    s.add_argument("--frames", type=int, default=45)
    s.add_argument("--name", required=True, help="file name prefix (content type, not a title)")
    s.add_argument("--out", required=True, help="output directory")
    s.add_argument("--degrade", default="d1", help="comma list: d1 (Mitchell, CRF 20), d2 (area, CRF 26)")
    s.add_argument("--crop", action="store_true", help="also the multiple-of-16 crop control")
    s.add_argument("--force", action="store_true", help="redo what the manifest says is done")
    s.add_argument("--threads", type=int, default=16)
    s = sub.add_parser("slice", help="frames A..A+N-1 of a made clip's all-intra files, checked by framemd5")
    s.add_argument("clip", help="DIR/NAME of a clip made by `make` (or its manifest DIR/NAME.json)")
    s.add_argument("--first", type=int, required=True, help="first frame, 0-based in the clip")
    s.add_argument("--frames", type=int, required=True)
    s.add_argument("--name", required=True, help="file name prefix of the slice")
    s.add_argument("--out", help="output directory (default: the clip's)")
    s.add_argument("--files", default="", help="comma list among src, gt, <deg>.lr, <deg>.bicubic (default: all)")
    s.add_argument("--threads", type=int, default=16)
    s = sub.add_parser("verify", help="frame counts, cv2 bit-exactness, frame-exactness vs the source")
    s.add_argument("clip", help="DIR/NAME (or its manifest DIR/NAME.json)")
    s.add_argument("--cv2-python", help="Python with cv2 to read the inputs as the CLI does")
    s.add_argument("--source", action="store_true", help="framemd5 of the source's frames 0..S+N-1")
    s = sub.add_parser("sheet", help="contact sheet of some frames (frame-exact)")
    s.add_argument("video")
    s.add_argument("--frames", required=True, help="comma list of frame indices")
    s.add_argument("--out", required=True, help="PNG")
    s.add_argument("--width", type=int, default=640, help="width of each frame")
    s.add_argument("--columns", type=int, default=0)
    s.add_argument("--threads", type=int, default=16)
    s = sub.add_parser("grain", help="film grain: luma high-pass residual over the smoothest 30%%, per frame")
    s.add_argument("file", help="an RGB file (GT, baseline, master) or a YUV source")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--frames", type=int, help="N frames over the middle 80%% (YUV sources: 12 by default, seeking)")
    g.add_argument("--every", action="store_true", help="every frame, decoded in order (RGB files: the default)")
    s.add_argument("--sigma", type=float, help="Gaussian blur of the high-pass, px (default 2)")
    s.add_argument("--share", type=float, help="smoothest share of the picture measured, %% (default 30)")
    s.add_argument("--threads", type=int, default=16)
    s.add_argument("--json")
    sub.add_parser("selftest", help="colour chain vs the equations, FFV1 round trips, extraction")
    a = ap.parse_args()
    {"scan": cmd_scan, "scores": cmd_scores, "make": cmd_make, "slice": cmd_slice, "verify": cmd_verify,
     "sheet": cmd_sheet, "grain": cmd_grain, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    main()

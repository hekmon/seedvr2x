#!/usr/bin/env python3
"""Scene-cut detection measurements: every frame's ffmpeg scdet score, computed as sptenc runs it,
a second opinion from PySceneDetect, review candidates and statistics per threshold.

  scd_scores.py check SOURCE --out DIR [--synthetic]  # sptenc's own command vs the all-score pass
  scd_scores.py score SOURCE --out DIR [--until SEC]  # one scdet pass: every frame's score
  scd_scores.py pysd --out DIR [--range A:B]          # AdaptiveDetector + ContentDetector
  scd_scores.py pysd-check FILE --out DIR             # our PySceneDetect loop vs its own pipeline
  scd_scores.py analyse --out DIR [--name NAME]       # candidates, auto-classes, stats per threshold
  scd_scores.py activity --out DIR [--window 25]      # where a film is busiest (to pick a chunk)
  scd_scores.py summary DIR... [--labels INDEX.csv]   # tables across episodes (and label stats)

SOURCE is a video file, a directory of segment files played in name order (an episode kept as
the segments of an earlier split), or @LIST (one file per line). Frames are numbered from 0 in
decode order over the whole source: frame n of a segment list is the n-th frame of the
concatenation. A cut at frame n means a new shot starts at n (scdet and PySceneDetect agree on
this convention). Every decode is a plain sequential one (no seeking), video stream only,
-fps_mode passthrough (no frame dropped or repeated).

check: runs sptenc's own detection command on a single file (its filter chain, threshold and
metadata key, the log parsed as sptenc parses it), then the all-score pass, and verifies that
(a) every frame carries lavfi.scd.score, (b) the frames sptenc reports are exactly the frames
whose printed score is >= the threshold, (c) the metadata frame counter runs 0, 1, 2... with
pts_time = n / fps. --synthetic first makes SOURCE a clip with hard cuts at known frames
(30, 60, 90; libx264 with B-frames) and checks that they are found there.

score: one scdet pass with sptenc's filter chain (setpts=PTS-STARTPTS, scdet=t=10, metadata
print), every key printed for every frame: lavfi.scd.mafd, lavfi.scd.score, and lavfi.scd.time
where the score reaches 10 (sptenc's detections). A segment list is decoded segment by segment
into one raw pipe read by a single scdet, so the joins are scored like any frame (the first
frame of a segment against the last one of the previous segment). Writes DIR/source.json,
DIR/scdet.txt (the raw print) and DIR/scores.tsv (n, pts_time, mafd, score, t10).

pysd: PySceneDetect's AdaptiveDetector and ContentDetector with their defaults, fed the same
decoded frames as BGR, converted as its OpenCV backend converts them (swscale, bicubic flags,
the matrix and range the stream declares, BT.601 when it declares none: pysd-check found
OpenCV's frames and ours identical), then downscaled as its SceneManager does by default (to
max(width, height) = 256 px, cv2.INTER_LINEAR). --range restricts the analysis to a chunk
(frames, or [H:]MM:SS[.s] / s times); the detectors start --warmup frames before it. Writes
DIR/pysd.json (cut frames per detector) and DIR/pysd_content_val.npy (ContentDetector's
content_val per frame).

pysd-check: on one file, our PySceneDetect loop against PySceneDetect's own pipeline
(open_video, OpenCV backend, SceneManager): cut lists per detector, plus the pixel difference
between OpenCV's decoded frames and ours.

analyse: candidates = scdet local maxima (score >= 4, above the previous frame and not below the
next) + PySceneDetect detections + segment joins, merged within +-1 frame. "sure" = scdet >= 30
and both PySceneDetect detectors within +-1 frame, otherwise "doubtful". A candidate's burst is
the number of other frames scoring >= 6 within +-6 frames (new drawings of limited animation in
motion come in bursts, every 2 or 3 frames; a cut stands alone). Per threshold T: detections
(every frame scoring >= T is a cut, as in sptenc), their gaps to the previous detection, sure
cuts below T, possible misses (PySceneDetect detections whose scdet score is below T), shot
lengths (median, and counts under 0.5, 1 and 2 s; the partial first and last shots of a chunk
are left out). Writes DIR/candidates.csv, DIR/stats.json, DIR/stats.md.

activity: scdet activity along a scored source (local maxima per block of minutes), and the
windows richest in doubtful-range local maxima: to pick a chunk of a long film.

summary: Markdown tables across episodes. --labels takes the review index (scd_review.py) with
its label column filled (cut / flash / pan / fade / dissolve / other / not-a-cut) and adds,
per threshold, estimated cuts detected, false positives by label and missed cuts (each labelled
row weighted by the candidates of its review stratum it stands for).

Needs ffmpeg and ffprobe on PATH (a build with scdet; FFMPEG/FFPROBE override) and numpy;
pysd and pysd-check need scenedetect and OpenCV.
"""
import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from fractions import Fraction

import numpy as np

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE", "ffprobe")
VIDEO_EXT = (".mkv", ".mp4", ".mov", ".m2ts", ".mts", ".ts", ".webm", ".avi", ".mxf", ".y4m", ".nut")

SPTENC_T = 10.0  # sptenc's default threshold
THRESHOLDS = (6, 8, 9, 10, 11, 12, 13, 14, 16, 20)
CAND_MIN = 4.0  # scdet local maxima from this score on are candidates
SURE_MIN = 30.0  # scdet score of a sure cut (when both PySceneDetect detectors agree)
SHORT = (0.5, 1.0, 2.0)  # shot length bins, seconds
LABELS = ("cut", "flash", "pan", "fade", "dissolve", "other", "not-a-cut")

BANDS = (4, 6, 8, 10, 12, 14, 17, 20, 30)  # scdet score bands of the review strata
BURST = (6, 6.0)  # burst: other frames within +-6 frames scoring >= 6

FRAME_RE = re.compile(r"frame:(\d+)\s+pts:(\S+)\s+pts_time:(\S+)")
SCDET_LOG_RE = re.compile(r"lavfi\.scd\.score: ([\d.]+), lavfi\.scd\.time: (\S+)")
SYNTH_CUTS = (30, 60, 90)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fmt_num(x):
    """A number as Go's FormatFloat(x, 'f', -1) writes it (sptenc's threshold): 10 -> "10"."""
    return ("%.6f" % x).rstrip("0").rstrip(".")


def timecode(frame, fps):
    s = float(Fraction(frame) / Fraction(fps))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{sec:06.3f}"


def score_band(x):
    if x < BANDS[0]:
        return f"<{BANDS[0]:g}"
    for lo, hi in zip(BANDS, BANDS[1:]):
        if lo <= x < hi:
            return f"{lo:g}-{hi:g}"
    return f">={BANDS[-1]:g}"


def pysd_agreement(c):
    """both / one / none: PySceneDetect detectors cutting within +-1 frame of a candidate."""
    a, b = c.get("adaptive") not in (None, ""), c.get("content") not in (None, "")
    return "both" if a and b else ("one" if a or b else "none")


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")


def read_full(stream, buf):
    """Fill buf from a raw pipe; returns the bytes read (less than len(buf) only at the end)."""
    view = memoryview(buf)
    got = 0
    while got < len(buf):
        n = stream.readinto(view[got:])
        if not n:
            break
        got += n
    return got


# --- sources -------------------------------------------------------------------------------------

def source_files(spec):
    if spec.startswith("@"):
        with open(spec[1:], encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if os.path.isdir(spec):
        files = sorted(os.path.join(spec, f) for f in os.listdir(spec) if f.lower().endswith(VIDEO_EXT))
        if not files:
            sys.exit(f"no video file in {spec}")
        return files
    if not os.path.exists(spec):
        sys.exit(f"{spec}: no such file")
    return [spec]


def probe(path):
    r = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_frames,"
         "color_space,color_range,field_order:format=duration", "-of", "json", path],
        capture_output=True, text=True, check=True)
    d = json.loads(r.stdout)
    if not d.get("streams"):
        sys.exit(f"{path}: no video stream")
    s = d["streams"][0]
    s["duration"] = d.get("format", {}).get("duration")
    return s


def describe_source(spec, pix_fmt=None):
    files = source_files(spec)
    kind = "file" if len(files) == 1 and os.path.isfile(spec) else "segments"
    probes = [probe(f) for f in files]
    p0 = probes[0]
    for f, p in zip(files, probes):
        for k in ("width", "height"):
            if p[k] != p0[k]:
                sys.exit(f"{f}: {k} {p[k]} differs from {files[0]}'s {p0[k]}")
    # the rate only times the print (frames are counted): the most common one; a merged segment
    # with uneven timestamps may report another (24/1 among 24000/1001 ones)
    rates = [p["r_frame_rate"] for p in probes]
    fps = max(set(rates), key=rates.count)
    pix_fmts = sorted({p["pix_fmt"] for p in probes})
    if kind == "segments" and len(pix_fmts) > 1 and not pix_fmt:
        sys.exit(f"segments in several pixel formats {pix_fmts}: choose the pipe's with --pix-fmt")
    nb = [int(p["nb_frames"]) if str(p.get("nb_frames", "")).isdigit() else None for p in probes]
    return {
        "spec": spec, "kind": kind, "files": files,
        "width": p0["width"], "height": p0["height"], "fps": fps,
        "other_rates": [(os.path.basename(f), r) for f, r in zip(files, rates) if r != fps],
        "avg_fps": p0.get("avg_frame_rate"), "codecs": sorted({p["codec_name"] for p in probes}),
        "pix_fmts": pix_fmts, "pipe_pix_fmt": pix_fmt or p0["pix_fmt"],
        "color_space": p0.get("color_space"), "color_range": p0.get("color_range"),
        "field_order": p0.get("field_order"), "container_frames": nb,
    }


def segments_of(src):
    """(file, first frame index, frame count or None) for each file of a scored source."""
    if src["kind"] == "file":
        return [(src["files"][0], 0, src.get("frames"))]
    return [(g["file"], g["start"], g["frames"]) for g in src["segments"]]


def decode_cmd(path, pix_fmt, threads, vf=None, frames=None, progress=None):
    """Plain sequential decode of the video stream to raw frames on stdout."""
    cmd = [FFMPEG, "-hide_banner", "-nostats", "-v", "error", "-threads", str(threads), "-i", path,
           "-map", "0:v:0", "-an", "-sn", "-dn"]
    if vf:
        cmd += ["-vf", vf]
    if frames:
        cmd += ["-frames:v", str(frames)]
    cmd += ["-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", pix_fmt]
    if progress:
        cmd += ["-progress", progress]
    return cmd + ["pipe:1"]


def parse_range(spec, fps, n):
    """'A..B': frames (integers) or times ([H:]MM:SS[.s], or seconds written 2400s / 2400.0);
    a time t maps to frame round(t * fps). Either end may be left empty."""
    if not spec:
        return 0, n
    if ".." not in spec:
        sys.exit("--range: write A..B (frames, or times such as 0:40:00..1:05:00)")

    def one(x, default):
        x = x.strip()
        if not x:
            return default
        if x.isdigit():
            return int(x)
        secs = 0.0
        for part in x.rstrip("s").split(":"):
            secs = secs * 60 + float(part)
        return int(round(secs * Fraction(fps)))

    a, b = spec.split("..", 1)
    return one(a, 0), min(one(b, n), n)


# --- the scdet pass ------------------------------------------------------------------------------

def scdet_vf(t, key=None, file=None):
    """sptenc's filter chain; key=None prints every key of every frame."""
    vf = f"setpts=PTS-STARTPTS,scdet=t={fmt_num(t)},metadata=mode=print"
    if key:
        vf += f":key={key}"
    if file:
        if re.search(r"[\\':,;\[\]\s]", file):
            sys.exit(f"{file!r}: no blank or filtergraph special character allowed in an output path")
        vf += f":file={file}"
    return vf


def progress_frames(path):
    frames = None
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.startswith("frame="):
                frames = int(ln.split("=", 1)[1])
    return frames


def score_pass(src, out_txt, threads, until=None, t=SPTENC_T):
    """Run scdet over the source; returns the frame count of each segment (None for a file)."""
    if src["kind"] == "file":
        cmd = [FFMPEG, "-y", "-hide_banner", "-nostats", "-threads", str(threads), "-i", src["files"][0]]
        if until:
            cmd += ["-t", str(until)]
        cmd += ["-map", "0:v:0", "-an", "-sn", "-dn", "-vf", scdet_vf(t, file=out_txt), "-f", "null", "-"]
        log("scdet: " + " ".join(cmd))
        subprocess.run(cmd, check=True)
        return None
    pix = src["pipe_pix_fmt"]
    cmd = [FFMPEG, "-y", "-hide_banner", "-nostats", "-f", "rawvideo", "-pix_fmt", pix,
           "-video_size", f"{src['width']}x{src['height']}", "-framerate", src["fps"], "-i", "pipe:0",
           "-vf", scdet_vf(t, file=out_txt), "-f", "null", "-"]
    log("scdet: " + " ".join(cmd))
    log("decode: " + " ".join(decode_cmd("SEGMENT", pix, threads, progress="PROGRESS")))
    scd = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    prog = out_txt + ".progress"
    counts = []
    try:
        for i, f in enumerate(src["files"]):
            if os.path.exists(prog):
                os.remove(prog)
            subprocess.run(decode_cmd(f, pix, threads, progress=prog), stdout=scd.stdin, check=True)
            counts.append(progress_frames(prog))
            if (i + 1) % 25 == 0 or i + 1 == len(src["files"]):
                log(f"{i + 1}/{len(src['files'])} segments, {sum(counts)} frames")
    finally:
        scd.stdin.close()
        rc = scd.wait()
        if os.path.exists(prog):
            os.remove(prog)
    if rc:
        sys.exit(f"scdet ffmpeg exited with {rc}")
    return counts


def parse_print(path):
    """The metadata filter's print (its file, or log lines): per frame n, pts_time and its keys."""
    rows = []
    cur = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            m = FRAME_RE.search(ln)
            if m:
                cur = {"n": int(m.group(1)), "pts_time": m.group(3)}
                rows.append(cur)
            elif cur is not None and "=" in ln:
                k, v = ln.strip().split("=", 1)
                cur[k.split("] ")[-1]] = v  # log lines start with "[Parsed_metadata_2 @ 0x...] "
    return rows


def write_scores(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("n\tpts_time\tmafd\tscore\tt10\n")
        for r in rows:
            f.write(f"{r['n']}\t{r['pts_time']}\t{r.get('lavfi.scd.mafd', 'nan')}\t"
                    f"{r.get('lavfi.scd.score', 'nan')}\t{1 if 'lavfi.scd.time' in r else 0}\n")


def load_scores(d):
    n, pt, mafd, score, t10 = [], [], [], [], []
    with open(os.path.join(d, "scores.tsv"), encoding="utf-8") as f:
        next(f)
        for ln in f:
            a, b, c, e, g = ln.rstrip("\n").split("\t")
            n.append(int(a))
            try:
                pt.append(float(b))
            except ValueError:
                pt.append(math.nan)
            mafd.append(float(c))
            score.append(float(e))
            t10.append(g == "1")
    return {"n": np.array(n), "pts_time": np.array(pt), "mafd": np.array(mafd),
            "score": np.array(score), "t10": np.array(t10, bool)}


def validate(rows, fps, t=SPTENC_T):
    """Checks of an all-key print: every frame scored, counter 0..N-1, pts = n / fps,
    lavfi.scd.time (scdet's own decision) <=> printed score >= t."""
    n = np.array([r["n"] for r in rows])
    no_score = [r["n"] for r in rows if "lavfi.scd.score" not in r]
    sc = np.array([float(r.get("lavfi.scd.score", "nan")) for r in rows])
    timed = np.array(["lavfi.scd.time" in r for r in rows])
    by_score = sc >= t
    pts = []
    for r in rows:
        try:
            pts.append(float(r["pts_time"]))
        except ValueError:
            pts.append(math.nan)
    dev = np.abs(np.array(pts) * float(Fraction(fps)) - n)
    return {
        "frames": len(rows),
        "counter_is_0_to_n": bool(len(n) and np.array_equal(n, np.arange(len(n)))),
        "frames_without_score": len(no_score),
        "first_frames_without_score": no_score[:10],
        "detections_t10": int(timed.sum()),
        "score_ge_t10": int(by_score.sum()),
        "t10_mismatch_frames": [int(x) for x in n[timed != by_score]][:20],
        "max_pts_dev_frames": float(np.nanmax(dev)) if len(dev) else None,
    }


def cmd_score(a):
    os.makedirs(a.out, exist_ok=True)
    src = describe_source(a.source, a.pix_fmt)
    log(f"{len(src['files'])} file(s), {src['width']}x{src['height']} {src['fps']} {src['codecs']} "
        f"{src['pix_fmts']}")
    txt = os.path.join(a.out, "scdet.txt")
    t0 = time.time()
    counts = score_pass(src, txt, a.threads, a.until)
    secs = time.time() - t0
    rows = parse_print(txt)
    write_scores(rows, os.path.join(a.out, "scores.tsv"))
    src["frames"] = len(rows)
    src["until"] = a.until
    if counts is not None:
        start = 0
        src["segments"] = []
        for f, c in zip(src["files"], counts):
            src["segments"].append({"file": f, "start": start, "frames": c})
            start += c
        src["segment_frames_total"] = start
    chk = validate(rows, src["fps"])
    if counts is not None:
        chk["segments_sum_equals_frames"] = src["segment_frames_total"] == len(rows)
        cf = src["container_frames"]
        chk["segments_differing_from_container_count"] = [
            (os.path.basename(f), c, n) for f, c, n in zip(src["files"], counts, cf) if n is not None and c != n]
    src["checks"] = chk
    src["score_seconds"] = round(secs, 1)
    save_json(src, os.path.join(a.out, "source.json"))
    log(f"{len(rows)} frames in {secs:.0f} s ({len(rows) / max(secs, 1e-9):.0f} fps); checks: {json.dumps(chk)}")


# --- check: sptenc's own command against the all-score pass ---------------------------------------

def make_synthetic(path, threads):
    """Four 30-frame shots, hard cuts at 30, 60, 90; H.264 with B-frames, 23.976 fps."""
    r = "24000/1001"
    g = (f"testsrc2=s=1280x720:r={r},trim=end_frame=30,setpts=PTS-STARTPTS[a];"
         f"smptehdbars=s=1280x720:r={r},noise=alls=12:allf=t,trim=end_frame=30,setpts=PTS-STARTPTS[b];"
         f"testsrc2=s=1280x720:r={r},hue=h=140,trim=start_frame=300:end_frame=330,setpts=PTS-STARTPTS[c];"
         f"mandelbrot=s=1280x720:r={r},trim=end_frame=30,setpts=PTS-STARTPTS[d];"
         "[a][b][c][d]concat=n=4:v=1:a=0,format=yuv420p[v]")
    cmd = [FFMPEG, "-y", "-hide_banner", "-v", "error", "-filter_complex", g, "-map", "[v]",
           "-c:v", "libx264", "-threads", str(threads), "-bf", "3", "-g", "48", "-crf", "18", path]
    subprocess.run(cmd, check=True)


def cmd_check(a):
    os.makedirs(a.out, exist_ok=True)
    path = a.source
    if a.synthetic:
        make_synthetic(path, a.threads)
        log(f"synthetic clip {path}: hard cuts at frames {list(SYNTH_CUTS)}")
    p = probe(path)
    fps = p["r_frame_rate"]
    # 1. sptenc's command: its chain, threshold and key; scenes read from the log (stderr)
    cmd = [FFMPEG, "-y", "-hide_banner", "-nostats", "-threads", str(a.threads), "-i", path,
           "-vf", scdet_vf(a.threshold, key="lavfi.scd.time"), "-f", "null", "-"]
    log("sptenc's command: " + " ".join(cmd))
    sp_log = os.path.join(a.out, "sptenc_cmd.log")
    with open(sp_log, "w", encoding="utf-8") as lf:
        subprocess.run(cmd, stderr=lf, check=True)
    sp_rows = parse_print(sp_log)
    sp_frames = [r["n"] for r in sp_rows if "lavfi.scd.time" in r]
    with open(sp_log, encoding="utf-8", errors="replace") as f:
        sp_scores = [m.group(1) for m in SCDET_LOG_RE.finditer(f.read())]
    # 2. the all-score pass (production command: video stream only, every key printed)
    src = {"kind": "file", "files": [path]}
    txt = os.path.join(a.out, "scdet.txt")
    score_pass(src, txt, a.threads, t=a.threshold)
    rows = parse_print(txt)
    chk = validate(rows, fps, a.threshold)
    sc = {r["n"]: r.get("lavfi.scd.score") for r in rows}
    ge = [r["n"] for r in rows if float(r.get("lavfi.scd.score", "nan")) >= a.threshold]
    res = {
        "file": path, "fps": fps, "threshold": a.threshold, "all_score_pass": chk,
        "sptenc_detections": len(sp_frames),
        "sptenc_frames_equal_score_ge_t": sp_frames == ge,
        "only_sptenc": sorted(set(sp_frames) - set(ge))[:20],
        "only_scores": sorted(set(ge) - set(sp_frames))[:20],
        "sptenc_log_scores_equal_printed": sp_scores == [sc.get(n) for n in sp_frames],
        "sptenc_frames_head": sp_frames[:12],
    }
    if a.synthetic:
        res["synthetic_expected"] = list(SYNTH_CUTS)
        res["synthetic_found"] = sp_frames
        res["synthetic_ok"] = sp_frames == list(SYNTH_CUTS)
        res["synthetic_scores_at_cuts"] = [sc.get(c) for c in SYNTH_CUTS]
        res["synthetic_max_score_elsewhere"] = max(
            float(v) for n, v in sc.items() if n not in SYNTH_CUTS and v is not None)
    save_json(res, os.path.join(a.out, "check.json"))
    print(json.dumps(res, indent=1))


# --- PySceneDetect -----------------------------------------------------------------------------

def bgr_frames(src, start, end, threads, matrix):
    """(index, BGR frame) for frames start..end-1 (end None = to the end), in decode order.
    The frame buffer is reused: copy what you keep."""
    w, h = src["width"], src["height"]
    size = w * h * 3
    conv = f"scale=in_color_matrix={matrix}:in_range=auto:out_range=pc:flags=bicubic,format=bgr24"
    buf = bytearray(size)
    for path, off, cnt in segments_of(src):
        if end is not None and off >= end:
            break
        if cnt is not None and start >= off + cnt:
            continue
        lo = max(start - off, 0)
        hi = None if end is None or (cnt is not None and end >= off + cnt) else end - off
        trim = []
        if lo:
            trim.append(f"start_frame={lo}")
        if hi is not None:
            trim.append(f"end_frame={hi}")
        vf = ("trim=" + ":".join(trim) + "," + conv) if trim else conv
        p = subprocess.Popen(decode_cmd(path, "bgr24", threads, vf=vf), stdout=subprocess.PIPE, bufsize=0)
        i = off + lo
        try:
            while True:
                got = read_full(p.stdout, buf)
                if got == 0:
                    break
                if got != size:
                    raise RuntimeError(f"{path}: truncated frame ({got} of {size} bytes)")
                yield i, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
                i += 1
        finally:
            p.stdout.close()
            rc = p.wait()
        if rc:
            raise RuntimeError(f"{path}: decoder exited with {rc}")
        want = hi if hi is not None else cnt  # local index the read should have reached
        if want is not None and i - off != want:
            log(f"WARNING {path}: {i - off - lo} frames read, expected {want - lo}")


def run_pysd(src, start, end, threads, matrix, progress_every=20000):
    import cv2
    from scenedetect.common import FrameTimecode
    from scenedetect.detectors import AdaptiveDetector, ContentDetector
    from scenedetect.scene_manager import compute_downscale_factor

    w, h = src["width"], src["height"]
    factor = compute_downscale_factor(max(w, h))  # what SceneManager.detect_scenes does by default
    dsize = (max(1, round(w / factor)), max(1, round(h / factor))) if factor > 1 else None
    fps = Fraction(src["fps"])
    n = src.get("frames") or (end or 0)
    cv = np.full(max(n, end or 0), np.nan, np.float32)
    ad, cd = AdaptiveDetector(), ContentDetector()
    cuts = {"adaptive": [], "content": []}
    last, count, t0 = None, 0, time.time()
    for i, img in bgr_frames(src, start, end, threads, matrix):
        small = cv2.resize(img, dsize, interpolation=cv2.INTER_LINEAR) if dsize else img.copy()
        tc = FrameTimecode(i, fps=fps)
        cuts["adaptive"] += [c.frame_num for c in ad.process_frame(tc, small)]
        cuts["content"] += [c.frame_num for c in cd.process_frame(tc, small)]
        if i < len(cv):
            cv[i] = cd._frame_score
        last, count = tc, count + 1
        if count % progress_every == 0:
            log(f"pysd: {count} frames (at {i}), {count / (time.time() - t0):.0f} fps, "
                f"cuts adaptive {len(cuts['adaptive'])} content {len(cuts['content'])}")
    if last is not None:
        cuts["adaptive"] += [c.frame_num for c in ad.post_process(last)]
        cuts["content"] += [c.frame_num for c in cd.post_process(last)]
    info = {"downscale": factor, "size": list(dsize) if dsize else [w, h], "frames_read": count,
            "seconds": round(time.time() - t0, 1), "matrix": matrix}
    return {k: sorted(set(v)) for k, v in cuts.items()}, cv, info


def cmd_pysd(a):
    import cv2
    import scenedetect
    src = load_json(os.path.join(a.out, "source.json"))
    n = src["frames"]
    s, e = parse_range(a.range, src["fps"], n)
    s0 = max(0, s - a.warmup)
    log(f"pysd: frames {s}..{e} (detectors from {s0}) of {n}")
    cuts, cv, info = run_pysd(src, s0, e, a.threads, a.matrix)
    expected = e - s0
    if info["frames_read"] != expected:
        log(f"WARNING: {info['frames_read']} frames read, the score pass has {expected} in that range")
    res = {"scenedetect": scenedetect.__version__, "opencv": cv2.__version__,
           "detectors": {"adaptive": "AdaptiveDetector() defaults: adaptive_threshold 3.0, "
                                     "min_scene_len 15, window_width 2, min_content_val 15.0",
                         "content": "ContentDetector() defaults: threshold 27.0, min_scene_len 15, "
                                    "weights (1, 1, 1, 0), flash filter MERGE"},
           "range": [s, e], "start": s0, "frames_expected": expected, **info,
           "cuts": {k: [c for c in v if s <= c < e] for k, v in cuts.items()}}
    save_json(res, os.path.join(a.out, "pysd.json"))
    np.save(os.path.join(a.out, "pysd_content_val.npy"), cv)
    log(f"pysd: {info['frames_read']} frames in {info['seconds']} s; cuts adaptive "
        f"{len(res['cuts']['adaptive'])}, content {len(res['cuts']['content'])}")


def cmd_pysd_check(a):
    import cv2
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import AdaptiveDetector, ContentDetector
    os.makedirs(a.out, exist_ok=True)
    src = describe_source(a.source)
    src["frames"] = None
    res = {"file": a.source}
    # frames: OpenCV's decode vs ours (matrix from the stream's tags, or forced)
    cap = cv2.VideoCapture(a.source)
    ref = []
    while len(ref) < a.frames:
        ok, fr = cap.read()
        if not ok:
            break
        ref.append(fr.astype(np.int16))
    cap.release()
    for matrix in ("auto", "bt601", "bt709"):
        diffs = []
        for (i, img), r in zip(bgr_frames(src, 0, len(ref), a.threads, matrix), ref):
            diffs.append(np.abs(img.astype(np.int16) - r))
        res[f"pixels_vs_opencv_{matrix}"] = {
            "frames": len(diffs), "mean_abs": float(np.mean([d.mean() for d in diffs])),
            "max_abs": int(max(d.max() for d in diffs)),
            "share_differing": float(np.mean([(d > 0).mean() for d in diffs]))}
    # cut lists: our loop vs PySceneDetect's own pipeline
    ours, _, info = run_pysd(src, 0, None, a.threads, a.matrix)
    res["ours"] = {**info, "cuts": ours}
    own = {}
    for name, mk in (("adaptive", AdaptiveDetector), ("content", ContentDetector)):
        t0 = time.time()
        video = open_video(a.source)
        sm = SceneManager()
        sm.add_detector(mk())
        sm.detect_scenes(video)
        own[name] = sorted(c.frame_num for c in sm.get_cut_list(show_warning=False))
        res[f"own_{name}_seconds"] = round(time.time() - t0, 1)
    res["own"] = own
    for k in own:
        res[f"{k}_identical"] = own[k] == ours[k]
        res[f"{k}_only_own"] = sorted(set(own[k]) - set(ours[k]))
        res[f"{k}_only_ours"] = sorted(set(ours[k]) - set(own[k]))
    save_json(res, os.path.join(a.out, "pysd_check.json"))
    print(json.dumps({k: v for k, v in res.items() if k not in ("ours", "own")}, indent=1))
    print("counts: ours", {k: len(v) for k, v in ours.items()}, "own", {k: len(v) for k, v in own.items()})


# --- candidates and statistics -------------------------------------------------------------------

def adaptive_ratio(cv, f, w=2, min_content_val=15.0):
    """AdaptiveDetector's adaptive_ratio of frame f, recomputed from content_val."""
    if f - w < 0 or f + w >= len(cv):
        return math.nan
    win = np.concatenate((cv[f - w:f], cv[f + 1:f + w + 1])).astype(np.float64)
    if np.isnan(win).any() or np.isnan(cv[f]):
        return math.nan
    avg = win.sum() / (2.0 * w)
    if abs(avg) < 0.00001:
        return 255.0 if cv[f] >= min_content_val else 0.0
    return min(float(cv[f]) / avg, 255.0)


def build_candidates(s, rng, cuts, joins):
    """Candidates within rng = (S, E): scdet local maxima >= CAND_MIN, PySceneDetect cuts and
    segment joins, merged within +-1 frame."""
    S, E = rng
    n = len(s)
    left = np.concatenate(([np.inf], s[:-1]))
    right = np.concatenate((s[1:], [-np.inf]))
    lm = (s > left) & (s >= right)
    cands = {}

    def new(f, **kw):
        c = {"frame": int(f), "scdet_lm": False, "adaptive": None, "content": None, "join": None}
        c.update(kw)
        cands[int(f)] = c
        return c

    for f in np.nonzero(lm & (s >= CAND_MIN))[0]:
        if S <= f < E:
            new(f, scdet_lm=True)

    def near(f):
        hits = [g for g in (f, f - 1, f + 1) if g in cands]
        if not hits:
            return None
        return hits[0] if hits[0] == f else max(hits, key=lambda g: s[g] if 0 <= g < n else -1)

    for key in ("adaptive", "content"):
        for d in cuts.get(key, []):
            if not S <= d < E:
                continue
            g = near(d)
            if g is None:
                new(d, **{key: int(d)})
            elif cands[g][key] is None:
                cands[g][key] = int(d)
    for j in joins:
        if not S <= j < E:
            continue
        g = near(j)
        if g is None:
            new(j, join=int(j))
        else:
            cands[g]["join"] = int(j)
    return [cands[f] for f in sorted(cands)], lm


def shot_stats(det, rng, n, fps):
    S, E = rng
    bounds = [S] + [int(x) for x in det if S < x < E] + [E]
    lengths = np.diff(bounds)
    if S > 0:
        lengths = lengths[1:]  # the chunk starts within a shot
    if E < n:
        lengths = lengths[:-1]  # and ends within one
    secs = lengths / float(Fraction(fps))
    out = {"shots": int(len(lengths)),
           "median_s": round(float(np.median(secs)), 3) if len(secs) else None,
           "min_frames": int(lengths.min()) if len(lengths) else None}
    for b in SHORT:
        out[f"under_{b:g}s"] = int((secs < b).sum())
    out["one_frame"] = int((lengths == 1).sum())
    return out


def cmd_analyse(a):
    d = a.out
    src = load_json(os.path.join(d, "source.json"))
    sc = load_scores(d)
    s, mafd = sc["score"], sc["mafd"]
    n = len(s)
    fps = src["fps"]
    pysd_path = os.path.join(d, "pysd.json")
    pysd = load_json(pysd_path) if os.path.exists(pysd_path) else None
    S, E = pysd["range"] if pysd else (0, n)
    E = min(E, n)
    cv_path = os.path.join(d, "pysd_content_val.npy")
    cv = np.load(cv_path) if os.path.exists(cv_path) else np.full(n, np.nan, np.float32)
    joins = [g["start"] for g in src.get("segments", [])[1:]]
    cands, lm = build_candidates(s, (S, E), pysd["cuts"] if pysd else {}, joins)
    for c in cands:
        f = c["frame"]
        c["scdet"] = float(s[f])
        c["mafd"] = float(mafd[f])
        c["prev"] = float(s[f - 1]) if f > 0 else math.nan
        c["next"] = float(s[f + 1]) if f + 1 < n else math.nan
        c["local_max"] = bool(lm[f])
        c["content_val"] = float(cv[f]) if f < len(cv) else math.nan
        c["ad_ratio"] = adaptive_ratio(cv, f)
        lo, hi = max(0, f - BURST[0]), min(n, f + BURST[0] + 1)
        c["burst"] = int((s[lo:hi] >= BURST[1]).sum() - (s[f] >= BURST[1]))
        sure = c["scdet"] >= SURE_MIN and c["adaptive"] is not None and c["content"] is not None
        c["class"] = "sure" if sure else "doubtful"
        c["timecode"] = timecode(f, fps)
    name = a.name or os.path.basename(os.path.normpath(d))
    cols = ["frame", "timecode", "class", "scdet", "mafd", "prev", "next", "local_max", "scdet_lm",
            "adaptive", "content", "ad_ratio", "content_val", "join", "burst"]
    with open(os.path.join(d, "candidates.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for c in cands:
            w.writerow([("" if c[k] is None else (f"{c[k]:.3f}" if isinstance(c[k], float) else c[k]))
                        for k in cols])

    sure = [c for c in cands if c["class"] == "sure"]
    doubt = [c for c in cands if c["class"] == "doubtful"]
    rs = s[S:E]
    bins = [(4, 6), (6, 10), (10, 14), (14, 20), (20, 30), (30, 1e9)]
    st = {
        "name": name, "source": src["spec"], "kind": src["kind"], "files": len(src["files"]),
        "fps": fps, "size": f"{src['width']}x{src['height']}", "frames_scored": n,
        "range": [S, E], "range_tc": [timecode(S, fps), timecode(E, fps)], "frames": E - S,
        "duration_s": round((E - S) / float(Fraction(fps)), 1),
        "joins_in_range": sum(1 for j in joins if S <= j < E),
        "pysd": bool(pysd),
        "checks": src.get("checks"),
        "candidates": len(cands), "sure": len(sure), "doubtful": len(doubt),
        "candidates_by_source": {
            "scdet_only": sum(1 for c in cands if c["scdet_lm"] and c["adaptive"] is None and c["content"] is None),
            "scdet_and_pysd": sum(1 for c in cands if c["scdet_lm"] and (c["adaptive"] is not None or c["content"] is not None)),
            "pysd_only": sum(1 for c in cands if not c["scdet_lm"] and (c["adaptive"] is not None or c["content"] is not None)),
            "join_only": sum(1 for c in cands if not c["scdet_lm"] and c["adaptive"] is None and c["content"] is None),
        },
        "doubtful_by_scdet": {f"{lo:g}-{hi:g}" if hi < 1e9 else f">={lo:g}":
                              sum(1 for c in doubt if c["scdet_lm"] and lo <= c["scdet"] < hi) for lo, hi in bins},
        "doubtful_not_scdet_candidate": sum(1 for c in doubt if not c["scdet_lm"]),
        "doubtful_with_both_pysd": sum(1 for c in doubt if c["adaptive"] is not None and c["content"] is not None),
        "doubtful_with_one_pysd": sum(1 for c in doubt if (c["adaptive"] is None) != (c["content"] is None)),
        "pysd_cuts": {k: len(v) for k, v in (pysd["cuts"].items() if pysd else [])},
        "frames_score_ge": {f"{t:g}": int((rs >= t).sum()) for t in (4, 30)},
    }
    # index alignment: PySceneDetect's frame against scdet's on strong cuts
    off = {"adaptive": {}, "content": {}}
    for c in cands:
        if c["scdet_lm"] and c["scdet"] >= SURE_MIN:
            for k in off:
                if c[k] is not None:
                    o = str(c[k] - c["frame"])
                    off[k][o] = off[k].get(o, 0) + 1
    st["pysd_offset_on_scdet_ge30"] = off
    per_t = {}
    for t in THRESHOLDS:
        det = np.nonzero(rs >= t)[0] + S
        detset = set(int(x) for x in det)
        lmset = {c["frame"] for c in cands if c["scdet_lm"]}
        pm_both = [c["frame"] for c in cands
                   if c["adaptive"] is not None and c["content"] is not None and c["scdet"] < t]
        pm_one = [c["frame"] for c in cands
                  if (c["adaptive"] is None) != (c["content"] is None) and c["scdet"] < t]
        gaps = np.diff(det)
        per_t[f"{t:g}"] = {
            "detections": len(det),
            "gaps": {"1": int((gaps == 1).sum()), "2": int((gaps == 2).sum()), "3": int((gaps == 3).sum()),
                     "4-11": int(((gaps >= 4) & (gaps <= 11)).sum()), ">=12": int((gaps >= 12).sum())},
            "after_hold": round(float((mafd[det[det >= 1] - 1] < 0.5).mean()), 3) if len(det) else None,
            "local_maxima": len(detset & lmset),
            "not_local_max": len(detset - lmset),
            "sure_detected": sum(1 for c in sure if c["frame"] in detset),
            "sure_below_t": sum(1 for c in sure if c["scdet"] < t),
            "doubtful_detected": sum(1 for c in doubt if c["frame"] in detset),
            "possible_misses_both": len(pm_both), "possible_misses_one": len(pm_one),
            "possible_misses_both_frames": pm_both[:40],
            "joins_below_t": sum(1 for j in joins if S <= j < E and s[j] < t),
            **shot_stats(det, (S, E), n, fps),
        }
    st["per_threshold"] = per_t
    save_json(st, os.path.join(d, "stats.json"))
    with open(os.path.join(d, "stats.md"), "w", encoding="utf-8") as f:
        f.write(stats_markdown([st]))
    log(f"{name}: {E - S} frames, {len(cands)} candidates ({len(sure)} sure, {len(doubt)} doubtful)")
    print(stats_markdown([st]))


def stats_markdown(sts, labelled=None):
    out = []
    out.append("| episode | frames | duration | range | files | candidates | sure | doubtful | "
               "scdet only | scdet+pysd | pysd only | joins |")
    out.append("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for st in sts:
        cs = st["candidates_by_source"]
        out.append(f"| {st['name']} | {st['frames']} | {st['duration_s'] / 60:.1f} min | "
                   f"{st['range_tc'][0]}–{st['range_tc'][1]} | {st['files']} | {st['candidates']} | "
                   f"{st['sure']} | {st['doubtful']} | {cs['scdet_only']} | {cs['scdet_and_pysd']} | "
                   f"{cs['pysd_only']} | {st['joins_in_range']} |")
    out.append("")
    ts = list(sts[0]["per_threshold"])
    out.append("Detections per threshold (frames scoring >= T; in brackets those that are not a local "
               "maximum):\n")
    out.append("| episode | " + " | ".join(f"T={t}" for t in ts) + " |")
    out.append("|---|" + "---:|" * len(ts))
    for st in sts:
        pt = st["per_threshold"]
        out.append(f"| {st['name']} | " + " | ".join(
            f"{pt[t]['detections']}" + (f" ({pt[t]['not_local_max']})" if pt[t]['not_local_max'] else "")
            for t in ts) + " |")
    out.append("")
    out.append("Detections by gap to the previous detection, frames 1 / 2 / 3 / 4-11 / >=12 (bursts every "
               "2 or 3 frames: new drawings of limited animation in motion), and the share right after a "
               "held frame (previous mafd < 0.5):\n")
    out.append("| episode | " + " | ".join(f"T={t}" for t in ts) + " |")
    out.append("|---|" + "---|" * len(ts))
    for st in sts:
        pt = st["per_threshold"]
        out.append(f"| {st['name']} | " + " | ".join(
            "/".join(str(pt[t]["gaps"][k]) for k in ("1", "2", "3", "4-11", ">=12")) +
            f", {pt[t]['after_hold']}" for t in ts) + " |")
    out.append("")
    out.append("Doubtful candidates by scdet score (local maxima; then the others):\n")
    keys = list(sts[0]["doubtful_by_scdet"])
    out.append("| episode | " + " | ".join(keys) + " | not a scdet candidate | both PySD | one PySD |")
    out.append("|---|" + "---:|" * (len(keys) + 3))
    for st in sts:
        out.append(f"| {st['name']} | " + " | ".join(str(st["doubtful_by_scdet"][k]) for k in keys) +
                   f" | {st['doubtful_not_scdet_candidate']} | {st['doubtful_with_both_pysd']} | "
                   f"{st['doubtful_with_one_pysd']} |")
    out.append("")
    out.append("Possible misses (PySceneDetect detects, scdet below T): both detectors / one:\n")
    out.append("| episode | " + " | ".join(f"T={t}" for t in ts) + " |")
    out.append("|---|" + "---:|" * len(ts))
    for st in sts:
        pt = st["per_threshold"]
        out.append(f"| {st['name']} | " + " | ".join(
            f"{pt[t]['possible_misses_both']} / {pt[t]['possible_misses_one']}" for t in ts) + " |")
    out.append("")
    out.append("Shots per threshold: count, median length (s), shots under 0.5 / 1 / 2 s:\n")
    out.append("| episode | " + " | ".join(f"T={t}" for t in ts) + " |")
    out.append("|---|" + "---|" * len(ts))
    for st in sts:
        pt = st["per_threshold"]
        out.append(f"| {st['name']} | " + " | ".join(
            f"{pt[t]['shots']}, {pt[t]['median_s']}, {pt[t]['under_0.5s']}/{pt[t]['under_1s']}/"
            f"{pt[t]['under_2s']}" for t in ts) + " |")
    out.append("")
    out.append("PySceneDetect frame minus scdet frame, on scdet local maxima >= 30 (index alignment):\n")
    for st in sts:
        o = st["pysd_offset_on_scdet_ge30"]
        out.append(f"- {st['name']}: adaptive {o['adaptive']}, content {o['content']}")
    out.append("")
    if labelled:
        out.append(labelled)
    return "\n".join(out) + "\n"


def label_stats(index_path):
    """Per episode and threshold, from the review index with labels: estimated cuts detected,
    false positives by label and missed cuts. The review samples within strata (scd_review.py):
    each labelled row stands for stratum_size / (labelled rows of its stratum) candidates, so a
    partly labelled index still gives estimates; strata without any label are counted apart."""
    with open(index_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    eps = {}
    for r in rows:
        eps.setdefault(r["episode"], {}).setdefault(r["stratum"], []).append(r)
    ts = [f"{t:g}" for t in THRESHOLDS]
    out = ["Labelled review: estimated candidates per threshold (detected = scdet >= T at the "
           "candidate; FP = detected and not labelled cut; missed = labelled cut, scdet < T)\n"]
    for ep, strata in eps.items():
        est, unlabelled, n_lab = [], 0, 0
        for g in strata.values():
            lab = [r for r in g if (r.get("label") or "").strip()]
            if not lab:
                unlabelled += int(g[0]["stratum_size"])
                continue
            n_lab += len(lab)
            w = int(g[0]["stratum_size"]) / len(lab)
            for r in lab:
                k = r["label"].strip().lower()
                est.append((float(r["scdet"]), k if k in LABELS else "other", w))
        out.append(f"{ep}: {n_lab} labelled rows; {unlabelled} candidates in strata without a label\n")
        out.append("| T | cuts detected | " + " | ".join(f"FP {lab}" for lab in LABELS[1:]) + " | missed cuts |")
        out.append("|---:|" + "---:|" * (len(LABELS) + 1))
        for t in ts:
            tv = float(t)
            tp = sum(w for x, k, w in est if k == "cut" and x >= tv)
            miss = sum(w for x, k, w in est if k == "cut" and x < tv)
            fp = [sum(w for x, k, w in est if k == lab and x >= tv) for lab in LABELS[1:]]
            out.append(f"| {t} | {tp:.0f} | " + " | ".join(f"{v:.0f}" for v in fp) + f" | {miss:.0f} |")
        out.append("")
    return "\n".join(out)


def cmd_activity(a):
    src = load_json(os.path.join(a.out, "source.json"))
    s = load_scores(a.out)["score"]
    n = len(s)
    fps = Fraction(src["fps"])
    left = np.concatenate(([np.inf], s[:-1]))
    right = np.concatenate((s[1:], [-np.inf]))
    lm = (s > left) & (s >= right)
    print(f"{n} frames; local maxima per {a.block:g} min block: >=6, >=10, [6, 30), >=30")
    blk = int(round(a.block * 60 * fps))
    for b0 in range(0, n, blk):
        z, q = lm[b0:b0 + blk], s[b0:b0 + blk]
        print(f"  {timecode(b0, fps)}  {int((z & (q >= 6)).sum()):4d} {int((z & (q >= 10)).sum()):4d} "
              f"{int((z & (q >= 6) & (q < 30)).sum()):4d} {int((z & (q >= 30)).sum()):4d}")
    win, step = int(round(a.window * 60 * fps)), int(round(a.step * 60 * fps))
    head, tail = int(round(a.skip_head * 60 * fps)), int(round(a.skip_tail * 60 * fps))
    doubt = np.cumsum(np.concatenate(([0], lm & (s >= 6) & (s < 30))))
    strong = np.cumsum(np.concatenate(([0], lm & (s >= 30))))
    wins = [(int(doubt[w0 + win] - doubt[w0]), w0) for w0 in range(head, n - win - tail + 1, step)]
    print(f"{a.window:g} min windows (from {a.skip_head:g} min, up to {a.skip_tail:g} min before the end), "
          f"by local maxima in [6, 30):")
    for c, w0 in sorted(wins, reverse=True)[:a.top]:
        print(f"  {timecode(w0, fps)}..{timecode(w0 + win, fps)}  [6, 30): {c}  >=30: "
              f"{int(strong[w0 + win] - strong[w0])}  (frames {w0}..{w0 + win})")


def cmd_summary(a):
    sts = [load_json(os.path.join(d, "stats.json")) for d in a.dirs]
    lab = label_stats(a.labels) if a.labels else None
    md = stats_markdown(sts, lab)
    if a.md:
        with open(a.md, "w", encoding="utf-8") as f:
            f.write(md)
    print(md)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    p.add_argument("source")
    p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float, default=SPTENC_T)
    p.add_argument("--synthetic", action="store_true", help="first write a test clip with known cuts to SOURCE")
    p.add_argument("--threads", type=int, default=16)
    p = sub.add_parser("score")
    p.add_argument("source")
    p.add_argument("--out", required=True)
    p.add_argument("--until", type=float, help="stop after this many seconds")
    p.add_argument("--pix-fmt", help="raw pipe pixel format for segment lists (default: the segments')")
    p.add_argument("--threads", type=int, default=16)
    p = sub.add_parser("pysd")
    p.add_argument("--out", required=True)
    p.add_argument("--range", help="A..B, frames or times (e.g. 0:40:00..1:05:00)")
    p.add_argument("--warmup", type=int, default=48, help="frames fed to the detectors before the range")
    p.add_argument("--matrix", default="auto", help="YUV->BGR matrix (auto: the stream's, as OpenCV)")
    p.add_argument("--threads", type=int, default=16)
    p = sub.add_parser("pysd-check")
    p.add_argument("source")
    p.add_argument("--out", required=True)
    p.add_argument("--frames", type=int, default=48, help="frames compared pixel by pixel with OpenCV's")
    p.add_argument("--matrix", default="auto")
    p.add_argument("--threads", type=int, default=16)
    p = sub.add_parser("analyse")
    p.add_argument("--out", required=True)
    p.add_argument("--name")
    p = sub.add_parser("activity")
    p.add_argument("--out", required=True)
    p.add_argument("--block", type=float, default=5.0, help="minutes")
    p.add_argument("--window", type=float, default=25.0, help="minutes")
    p.add_argument("--step", type=float, default=1.0, help="minutes")
    p.add_argument("--skip-head", type=float, default=5.0, help="minutes left out at the start")
    p.add_argument("--skip-tail", type=float, default=10.0, help="minutes left out at the end (credits)")
    p.add_argument("--top", type=int, default=8)
    p = sub.add_parser("summary")
    p.add_argument("dirs", nargs="+")
    p.add_argument("--labels", help="review index CSV with the label column filled")
    p.add_argument("--md", help="also write the tables to this file")
    a = ap.parse_args()
    {"check": cmd_check, "score": cmd_score, "pysd": cmd_pysd, "pysd-check": cmd_pysd_check,
     "analyse": cmd_analyse, "activity": cmd_activity, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()

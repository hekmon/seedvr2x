#!/usr/bin/env python3
"""Scene-cut detection measurements: every frame's ffmpeg scdet score, computed as sptenc runs it,
second opinions from PySceneDetect and TransNetV2, review candidates and statistics per threshold.

  scd_scores.py check SOURCE --out DIR [--synthetic]  # sptenc's own command vs the all-score pass
  scd_scores.py score SOURCE --out DIR [--no-reinit]  # one scdet pass: every frame's score
  scd_scores.py pysd --out DIR [--range A:B]          # AdaptiveDetector + ContentDetector
  scd_scores.py pysd-check FILE --out DIR             # our PySceneDetect loop vs its own pipeline
  scd_scores.py tnet --out DIR [--threads 16]         # TransNetV2's probabilities, every frame
  scd_scores.py tnet-check FILE --out DIR             # our TransNetV2 run vs the official TF one
  scd_scores.py analyse --out DIR [--name NAME]       # candidates, auto-classes, stats per threshold
  scd_scores.py activity --out DIR [--window 25]      # where a film is busiest (to pick a chunk)
  scd_scores.py compare REF OTHER [--md OUT.md]       # one video scored twice (original vs segments)
  scd_scores.py summary DIR... [--labels INDEX.csv]   # tables across episodes (and label stats)
  scd_scores.py round ROUND... [--dirs DIR...]        # estimates from rounds of labels

SOURCE is a video file, a directory of segment files played in name order (an episode kept as
the segments of an earlier split), or @LIST (one file per line). Frames are numbered from 0 in
decode order over the whole source: frame n of a segment list is the n-th frame of the
concatenation. A cut at frame n means a new shot starts at n (scdet and PySceneDetect agree on
this convention). Every decode is a plain sequential one (no seeking), video stream only,
-fps_mode passthrough (no frame dropped or repeated). Timecodes are the frames' presentation
times relative to the first frame, as a player shows them.

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
DIR/scdet.txt (the raw print) and DIR/scores.tsv (n, pts_time, mafd, score, t10). The print is
checked against ffmpeg's own count of the frames it passed: when the stream's parameters change
mid-way (a DVD whose colour description appears after a few frames), ffmpeg rebuilds the
filtergraph, which restarts the metadata filter's frame counter, setpts' start and scdet, and
reopens the print file; sptenc's command then numbers every later frame from 0 again.
--no-reinit adds -reinit_filter 0 so that one filtergraph sees every frame.

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

tnet: TransNetV2 (github.com/soCzech/TransNetV2, MIT), a network trained to find shot
boundaries, on CPU: the official PyTorch model, imported from a checkout of the official
repository (--tnet-dir or TNET_DIR: inference-pytorch/transnetv2_pytorch.py), with the weights
its convert_weights.py converts from the official TF ones (--weights or TNET_WEIGHTS). Frames 0
to the end of the scored range plus TNET_MARGIN are decoded as the score pass decodes them (same
files, decode order, segments concatenated, -fps_mode passthrough, where the official extraction
leaves ffmpeg's rawvideo default free to drop or repeat frames by timestamp) and scaled as the
official extraction scales them: ffmpeg's own output scaler (-s 48x27, its default flags) to
rgb24, frames as decoded (an anamorphic or interlaced source is neither unsqueezed nor
deinterlaced). The official predict_frames is reproduced: 25 copies of the first frame before
the frames, 25 to 74 copies of the last after, windows of 100 frames every 50, each keeping
its frames 25-74, then the sigmoid of the single-frame and all-frames outputs. A frame's window
reaches 74 frames ahead at most, so the scored range gets the values of a run on the whole
source. Writes DIR/tnet.npz: single and all (float32, one per decoded frame), frames,
weights_sha256, decode_seconds, inference_seconds, threads, and meta (JSON: commit, versions,
CPU seconds, the decode command).

tnet-check: the same frames through the official TF model (inference/transnetv2.py's
TransNetV2.predict_frames with the TF weights of --tf-weights) and through ours: first --random
random frames, then --frames frames of FILE from --start, decoded as tnet decodes them (and
compared byte for byte with the official extraction's own command); per run, the largest
absolute differences of both outputs, and whether the official predictions_to_scenes at 0.5 and
our detections at 0.5 are identical. Also our model with --batch windows per forward against
one at a time (the official way).

analyse: candidates = scdet local maxima (score >= 4, above the previous frame and not below the
next) + PySceneDetect detections + segment joins, merged within +-1 frame. "sure" = scdet >= 30
and both PySceneDetect detectors within +-1 frame, otherwise "doubtful". A candidate's burst is
the number of other frames scoring >= 6 within +-6 frames (new drawings of limited animation in
motion come in bursts, every 2 or 3 frames; a cut stands alone). Per threshold T: detections
(every frame scoring >= T is a cut, as in sptenc), their gaps to the previous detection, sure
cuts below T, possible misses (PySceneDetect detections whose scdet score is below T), shot
lengths (median, and counts under 0.5, 1 and 2 s; the partial first and last shots of a chunk
are left out). Also PySceneDetect's detectors alone: detections, gaps, shots
(pysd_detectors). Writes DIR/candidates.csv, DIR/stats.json, DIR/stats.md.

analyse with DIR/tnet.npz adds TransNetV2. Alignment: on the sure cuts that are scdet local
maxima, the frame of TransNetV2's single-frame peak within +-3 frames minus the cut's frame
(its official predictions_to_scenes ends a shot on its positive frame, so -1 is expected: it
marks the last frame of the outgoing shot); the modal offset (or --tnet-offset) aligns its
values: an aligned value at frame f is TransNetV2's for a cut at f in our convention. Its
local peaks >= 0.1 (aligned) become candidates where no candidate of the old population lies
within +-1 frame; a candidate belongs to the old population iff scdet_lm, adaptive, content or
join is set, and those keep their frame and columns byte for byte. Three columns end
candidates.csv: tnet and tnet_all, the highest aligned single-frame and all-frames
probabilities within +-1 frame, and tnet_peak, the aligned frame of the first when it reaches
0.1. Per threshold p (TNET_P): detections (runs of aligned frames >= p, one per run at its
peak), their gaps, shots, sure cuts with tnet >= p, TransNetV2-only candidates and those 2-3
frames from an old candidate (an alignment check: a wrong offset puts them there), PySceneDetect's
possible misses at T=10 that TransNetV2 gets, and the scdet T x TransNetV2 p agreement on
candidates. The statistics of the old population keep their keys and values; TransNetV2's go
under "tnet".

activity: scdet activity along a scored source (local maxima per block of minutes), and the
windows richest in doubtful-range local maxima: to pick a chunk of a long film.

compare: the same video scored twice, e.g. an original and the re-encoded segments of an earlier
split: the frame offset between them (found on the strong peaks), detections per threshold on
the same frame, one frame apart or in one version only, score and mafd differences, held frames,
bursts, and the reference's scores at the other version's segment joins (each join was a cut the
splitter detected).

summary: Markdown tables across episodes. --labels takes the review index (scd_review.py) with
its label column filled (cut / flash / pan / fade / dissolve / other / not-a-cut) and joins each
labelled row on (episode, frame) with the candidates.csv of the DIRS given. Per detector
(scdet >= T for each threshold, PySceneDetect's adaptive and content detectors, TransNetV2 >= p
for each p): estimated cuts detected and missed, recall, false positives by label, precision
with every other label counted as an error, and with fades and dissolves left out (gradual
transitions, not hard cuts: a detector may rightly cut there); per episode, then pooled by kind
(animation / live action / DVD: KINDS, --kind NAME=KIND; any other episode is animation). Each
labelled row stands for the candidates of its review stratum (stratum size / labelled rows of
the stratum), so a partly labelled index still gives estimates. Recall is relative to the cuts
among the candidates, the union of every detector's (a cut no detector comes near is never
shown). Rows of episodes missing from DIRS are counted apart.

round: estimates from rounds of labels made by scd_review.py round (ROUND/rows.csv: each row's
agreement group, kind, cell size and detector values; ROUND/labels.txt: one letter per row). Per
group and kind, the labels and the share of cuts with its 90% Wilson interval; then summary's table
per detector, each labelled row standing for its cell's candidates / the cell's labelled rows, per
kind and for every kind, with bootstrap intervals within the cells and paired differences to scdet
at 10. Several rounds (or --dirs) pool their rows in refined cells (refined_group: TransNetV2's
band, and the picture's change where it alone fires), sized in the episodes' candidates.csv.

Needs ffmpeg and ffprobe on PATH (a build with scdet; FFMPEG/FFPROBE override) and numpy;
pysd and pysd-check need scenedetect and OpenCV; tnet needs torch and the TransNetV2 checkout,
tnet-check TensorFlow as well.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import resource
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

TNET_P = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)  # TransNetV2 thresholds of the statistics
TNET_MIN = 0.1  # TransNetV2 local peaks from this probability on are candidates
TNET_BANDS = (0.1, 0.3, 0.5, 0.7, 0.9)  # TransNetV2 bands of the review strata
TNET_SIZE = (48, 27)  # TransNetV2's input frames, width x height
TNET_WINDOW, TNET_STEP, TNET_PAD = 100, 50, 25  # predict_frames: 100 frames every 50, keeps 25-74
TNET_MARGIN = 100  # frames decoded past the scored range: a frame's window reaches 74 frames ahead
TNET_SEARCH = 3  # alignment: TransNetV2's peak searched within +-3 frames of a sure cut
TNET_OFFSET = -1  # the official convention (last frame of the outgoing shot), when nothing is measured
KINDS = {"live-1": "live action", "live-2": "live action", "dvd-sitcom": "DVD"}

FRAME_RE = re.compile(r"frame:(\d+)\s+pts:(\S+)\s+pts_time:(\S+)")
SCDET_LOG_RE = re.compile(r"lavfi\.scd\.score: ([\d.]+), lavfi\.scd\.time: (\S+)")
SYNTH_CUTS = (30, 60, 90)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fmt_num(x):
    """A number as Go's FormatFloat(x, 'f', -1) writes it (sptenc's threshold): 10 -> "10"."""
    return ("%.6f" % x).rstrip("0").rstrip(".")


def hms(s):
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{sec:06.3f}"


def timecode(frame, fps, pts=None):
    """Frame time as a player shows it: its presentation time (relative to the first frame) when
    pts (the score pass's pts_time array) knows it, else frame / fps (they part when a stream's
    declared rate differs from its timestamps, or frames are missing)."""
    if pts is not None and 0 <= frame < len(pts) and not math.isnan(pts[frame]):
        return hms(float(pts[frame]))
    return hms(float(Fraction(frame) / Fraction(fps)))


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


def tnet_band(x):
    """TransNetV2 band of the review strata: 0.1-0.3 ... >=0.9."""
    if x < TNET_BANDS[0]:
        return f"<{TNET_BANDS[0]:g}"
    for lo, hi in zip(TNET_BANDS, TNET_BANDS[1:]):
        if lo <= x < hi:
            return f"{lo:g}-{hi:g}"
    return f">={TNET_BANDS[-1]:g}"


def old_population(c):
    """A candidate of scdet, PySceneDetect or a segment join, not of TransNetV2 alone (an analyse
    candidate or a candidates.csv row)."""
    return c.get("scdet_lm") in (True, "True") or any(c.get(k) not in (None, "")
                                                      for k in ("adaptive", "content", "join"))


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
         "stream=codec_name,pix_fmt,width,height,sample_aspect_ratio,r_frame_rate,avg_frame_rate,"
         "nb_frames,color_space,color_range,field_order:format=duration", "-of", "json", path],
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
        "field_order": p0.get("field_order"), "sar": p0.get("sample_aspect_ratio"),
        "container_frames": nb,
    }


def segments_of(src):
    """(file, first frame index, frame count or None) for each file of a scored source."""
    if src["kind"] == "file":
        return [(src["files"][0], 0, src.get("frames"))]
    return [(g["file"], g["start"], g["frames"]) for g in src["segments"]]


def decode_cmd(path, pix_fmt, threads, vf=None, frames=None, progress=None, size=None, filter_threads=None,
               reinit=True):
    """Plain sequential decode of the video stream to raw frames on stdout; size (WxH) scales them
    with ffmpeg's own output scaler (-s), as TransNetV2's extraction does. reinit=False keeps one
    filter graph for every frame (-reinit_filter 0): a graph rebuilt mid-stream restarts the frame
    counts of its filters (trim, select), as it restarts scdet's (see score)."""
    cmd = [FFMPEG, "-hide_banner", "-nostats", "-v", "error", "-threads", str(threads)]
    if filter_threads:
        cmd += ["-filter_threads", str(filter_threads)]
    if not reinit:
        cmd += ["-reinit_filter", "0"]
    cmd += ["-i", path, "-map", "0:v:0", "-an", "-sn", "-dn"]
    if vf:
        cmd += ["-vf", vf]
    if frames:
        cmd += ["-frames:v", str(frames)]
    cmd += ["-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", pix_fmt]
    if size:
        cmd += ["-s", size]
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


def score_pass(src, out_txt, threads, until=None, t=SPTENC_T, no_reinit=False):
    """Run scdet over the source; returns the frame count of each segment (None for a file)
    and ffmpeg's own count of the frames the filtergraph passed."""
    if src["kind"] == "file":
        prog = out_txt + ".progress"
        cmd = [FFMPEG, "-y", "-hide_banner", "-nostats", "-threads", str(threads)]
        if no_reinit:
            cmd += ["-reinit_filter", "0"]
        cmd += ["-i", src["files"][0]]
        if until:
            cmd += ["-t", str(until)]
        cmd += ["-map", "0:v:0", "-an", "-sn", "-dn", "-vf", scdet_vf(t, file=out_txt), "-progress", prog,
                "-f", "null", "-"]
        log("scdet: " + " ".join(cmd))
        subprocess.run(cmd, check=True)
        frames = progress_frames(prog)
        os.remove(prog)
        return None, frames
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
    return counts, sum(counts)


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
    counts, ff_frames = score_pass(src, txt, a.threads, a.until, no_reinit=a.no_reinit)
    secs = time.time() - t0
    rows = parse_print(txt)
    write_scores(rows, os.path.join(a.out, "scores.tsv"))
    src["frames"] = len(rows)
    src["until"] = a.until
    src["no_reinit"] = a.no_reinit
    if counts is not None:
        start = 0
        src["segments"] = []
        for f, c in zip(src["files"], counts):
            src["segments"].append({"file": f, "start": start, "frames": c})
            start += c
        src["segment_frames_total"] = start
    chk = validate(rows, src["fps"])
    chk["ffmpeg_frames"] = ff_frames
    chk["print_has_every_frame"] = ff_frames == len(rows)
    if ff_frames != len(rows):
        log(f"WARNING: the print holds {len(rows)} frames, ffmpeg passed {ff_frames}: the filtergraph was "
            "rebuilt mid-stream (its counter restarted); score again with --no-reinit")
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


# --- TransNetV2 ----------------------------------------------------------------------------------

def scored_range(d, src):
    """The analysed range (S, E): PySceneDetect's chunk, else the whole source."""
    n = src["frames"]
    p = os.path.join(d, "pysd.json")
    if os.path.exists(p):
        s, e = load_json(p)["range"]
        return s, min(e, n)
    return 0, n


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cpu_seconds(who):
    r = resource.getrusage(who)
    return r.ru_utime + r.ru_stime


def tnet_model(tnet_dir, weights, threads):
    """The official PyTorch TransNetV2 (imported from the checkout) with converted weights, on CPU."""
    import torch
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    sys.path.insert(0, os.path.join(tnet_dir, "inference-pytorch"))
    from transnetv2_pytorch import TransNetV2
    model = TransNetV2()
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()
    try:
        commit = subprocess.run(["git", "-C", tnet_dir, "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return model, {"tnet_dir": os.path.abspath(tnet_dir), "commit": commit, "weights": os.path.abspath(weights),
                   "weights_sha256": sha256(weights), "torch": torch.__version__, "threads": threads}


def tnet_frames(src, end, threads, every=20000):
    """Frames 0..end-1 of a scored source as TransNetV2 takes them: decoded as the score pass
    decodes them, scaled by ffmpeg's output scaler to 48x27 rgb24 (the official extraction's
    -s 48x27 -pix_fmt rgb24). Returns a [n, 27, 48, 3] uint8 array and the first decode command."""
    w, h = TNET_SIZE
    size = w * h * 3
    out = np.empty((end, h, w, 3), np.uint8)
    flat = out.reshape(-1)
    got, first, t0 = 0, None, time.time()
    for path, off, cnt in segments_of(src):
        if off >= end:
            break
        if off != got:
            raise RuntimeError(f"{path}: starts at frame {off}, but {got} frames were decoded before it")
        want = end - off if cnt is None or end < off + cnt else None
        cmd = decode_cmd(path, "rgb24", threads, frames=want, size=f"{w}x{h}", filter_threads=threads)
        first = first or cmd
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=0)
        try:
            while got < end:
                k = min(end - got, 4096)
                r = read_full(p.stdout, flat[got * size:(got + k) * size])
                if r % size:
                    raise RuntimeError(f"{path}: truncated frame ({r % size} of {size} bytes)")
                if (got + r // size) // every > got // every:
                    log(f"decode: {got + r // size} frames, {(got + r // size) / (time.time() - t0):.0f} fps")
                got += r // size
                if r < k * size:
                    break
        finally:
            p.stdout.close()
            rc = p.wait()
        if rc:
            raise RuntimeError(f"{path}: decoder exited with {rc}")
        if want is None and cnt is not None and got - off != cnt:
            log(f"WARNING {path}: {got - off} frames decoded, the score pass counted {cnt}")
    return out[:got], first


def tnet_predict(model, frames, batch=1, every=20000):
    """TransNetV2's predict_frames with the PyTorch model: 25 copies of the first frame before, 25 to
    74 of the last after, windows of 100 frames every 50 (batch windows per forward; officially
    one), frames 25-74 of each kept; sigmoid of the single-frame and all-frames logits."""
    import torch
    n = len(frames)
    pad_end = TNET_PAD + TNET_STEP - (n % TNET_STEP or TNET_STEP)
    idx = np.concatenate((np.zeros(TNET_PAD, np.int64), np.arange(n), np.full(pad_end, n - 1)))
    starts = list(range(0, len(idx) - TNET_WINDOW + 1, TNET_STEP))
    one = np.empty(len(starts) * TNET_STEP, np.float32)
    many = np.empty_like(one)
    t0 = time.time()
    with torch.inference_mode():
        for b in range(0, len(starts), batch):
            sl = starts[b:b + batch]
            x = torch.from_numpy(np.stack([frames[idx[s:s + TNET_WINDOW]] for s in sl]))
            logits, d = model(x)
            keep = slice(TNET_PAD, TNET_PAD + TNET_STEP)
            one[b * TNET_STEP:(b + len(sl)) * TNET_STEP] = torch.sigmoid(logits)[:, keep, 0].reshape(-1).numpy()
            many[b * TNET_STEP:(b + len(sl)) * TNET_STEP] = torch.sigmoid(d["many_hot"])[:, keep, 0].reshape(-1).numpy()
            done = min((b + len(sl)) * TNET_STEP, n)
            if done // every > min(b * TNET_STEP, n) // every:
                log(f"inference: {done}/{n} frames, {done / (time.time() - t0):.0f} fps")
    return one[:n], many[:n]


def tnet_detections(x, p):
    """One detection per run of frames >= p, at the run's highest frame (the first if tied)."""
    m = np.concatenate(([False], np.nan_to_num(x, nan=-1.0) >= p, [False]))
    d = np.diff(m.astype(np.int8))
    return [int(a + np.argmax(x[a:b])) for a, b in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0])]


def cmd_tnet(a):
    src = load_json(os.path.join(a.out, "source.json"))
    n = src["frames"]
    S, E = scored_range(a.out, src)
    end = min(n, E + TNET_MARGIN)
    model, info = tnet_model(a.tnet_dir, a.weights, a.threads)
    log(f"tnet: frames 0..{end} of {n} (scored range {S}..{E}); TransNetV2 {info['commit']}, weights "
        f"{info['weights']} (sha256 {info['weights_sha256']}), torch {info['torch']}, {a.threads} threads, "
        f"batch {a.batch}")
    c0, t0 = cpu_seconds(resource.RUSAGE_CHILDREN), time.time()
    frames, cmd = tnet_frames(src, end, a.threads)
    dec_s, dec_cpu = time.time() - t0, cpu_seconds(resource.RUSAGE_CHILDREN) - c0
    log(f"decode: {len(frames)} frames in {dec_s:.1f} s ({len(frames) / max(dec_s, 1e-9):.0f} fps), "
        f"ffmpeg CPU {dec_cpu:.0f} s; {' '.join(cmd)}")
    if len(frames) != end:
        log(f"WARNING: {len(frames)} frames decoded, {end} expected")
    c0, t0 = cpu_seconds(resource.RUSAGE_SELF), time.time()
    one, many = tnet_predict(model, frames, a.batch)
    inf_s, inf_cpu = time.time() - t0, cpu_seconds(resource.RUSAGE_SELF) - c0
    log(f"inference: {len(frames)} frames in {inf_s:.1f} s ({len(frames) / max(inf_s, 1e-9):.0f} fps), "
        f"CPU {inf_cpu:.0f} s; frames >= 0.5: single {int((one >= 0.5).sum())}, all {int((many >= 0.5).sum())}")
    meta = {**info, "range": [S, E], "source_frames": n, "decoded": len(frames), "margin": TNET_MARGIN,
            "batch": a.batch, "size": "x".join(str(v) for v in TNET_SIZE), "fps": src["fps"],
            "decode_cpu_seconds": round(dec_cpu, 1), "inference_cpu_seconds": round(inf_cpu, 1),
            "decode": " ".join(cmd)}
    np.savez(os.path.join(a.out, "tnet.npz"), single=one, all=many, frames=len(frames),
             weights_sha256=info["weights_sha256"], decode_seconds=round(dec_s, 1),
             inference_seconds=round(inf_s, 1), threads=a.threads, meta=json.dumps(meta))
    log(f"wrote {os.path.join(a.out, 'tnet.npz')}")


def tnet_official_extraction(path, n, threads):
    """The official extraction's command (ffmpeg-python's -i FILE -f rawvideo -pix_fmt rgb24 -s 48x27
    pipe:), its first n frames; -threads and quiet logging added."""
    w, h = TNET_SIZE
    cmd = [FFMPEG, "-hide_banner", "-nostats", "-v", "error", "-threads", str(threads), "-i", path,
           "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "pipe:"]
    r = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.uint8).reshape(-1, h, w, 3), cmd


def tnet_compare(official, model, frames, batch):
    """The same frames through the official TF predict_frames and ours."""
    t0 = time.time()
    tf_one, tf_all = official.predict_frames(frames)
    t_tf = time.time() - t0
    t0 = time.time()
    one, many = tnet_predict(model, frames, 1)
    t_pt = time.time() - t0
    sc_tf = official.predictions_to_scenes(tf_one, 0.5)
    sc_pt = official.predictions_to_scenes(one, 0.5)
    d_tf, d_pt = tnet_detections(tf_one, 0.5), tnet_detections(one, 0.5)
    res = {"frames": len(frames), "seconds_tf": round(t_tf, 1), "seconds_torch": round(t_pt, 1),
           "max_abs_single": float(np.abs(tf_one - one).max()), "max_abs_all": float(np.abs(tf_all - many).max()),
           "mean_abs_single": float(np.abs(tf_one - one).mean()),
           "frames_ge_0.5": {"tf": int((tf_one >= 0.5).sum()), "torch": int((one >= 0.5).sum())},
           "scenes_0.5": len(sc_tf), "scenes_0.5_identical": bool(np.array_equal(sc_tf, sc_pt)),
           "detections_0.5": len(d_tf), "detections_0.5_identical": d_tf == d_pt,
           "detections_0.5_head": d_tf[:20]}
    if batch > 1:
        t0 = time.time()
        b_one, b_all = tnet_predict(model, frames, batch)
        res["batch"] = {"windows": batch, "seconds_torch": round(time.time() - t0, 1),
                        "max_abs_single_vs_one": float(np.abs(b_one - one).max()),
                        "max_abs_all_vs_one": float(np.abs(b_all - many).max())}
    return res


def cmd_tnet_check(a):
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(a.threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    os.makedirs(a.out, exist_ok=True)
    model, info = tnet_model(a.tnet_dir, a.weights, a.threads)
    sys.path.insert(0, os.path.join(a.tnet_dir, "inference"))
    from transnetv2 import TransNetV2 as Official
    tf_weights = a.tf_weights or os.path.join(a.tnet_dir, "inference", "transnetv2-weights")
    official = Official(tf_weights)
    res = {"file": a.source, **info, "tensorflow": tf.__version__, "tf_weights": os.path.abspath(tf_weights),
           "tf_variables_sha256": sha256(os.path.join(tf_weights, "variables", "variables.data-00000-of-00001"))}
    if a.random:
        rnd = np.random.default_rng(a.seed).integers(0, 256, (a.random, TNET_SIZE[1], TNET_SIZE[0], 3), np.uint8)
        log(f"random frames: {a.random}")
        res["random"] = tnet_compare(official, model, rnd, a.batch)
        log(json.dumps(res["random"]))
    end = a.start + a.frames
    src = {"kind": "file", "files": [a.source], "frames": None}
    frames, cmd = tnet_frames(src, end, a.threads)
    ref, ref_cmd = tnet_official_extraction(a.source, end, a.threads)
    m = min(len(frames), len(ref))
    diff = np.abs(frames[:m].astype(np.int16) - ref[:m].astype(np.int16))
    res["extraction"] = {"ours": " ".join(cmd), "official": " ".join(ref_cmd), "frames_ours": len(frames),
                         "frames_official": len(ref), "identical": len(frames) == len(ref) and not diff.any(),
                         "frames_differing": int(diff.reshape(m, -1).any(1).sum()), "max_abs": int(diff.max())}
    log(json.dumps(res["extraction"]))
    ex = frames[a.start:end]
    log(f"excerpt: frames {a.start}..{a.start + len(ex)} of {a.source}")
    res["excerpt"] = {"start": a.start, **tnet_compare(official, model, ex, a.batch)}
    log(json.dumps(res["excerpt"]))
    save_json(res, os.path.join(a.out, "tnet_check.json"))
    print(json.dumps(res, indent=1))


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

    def describe(c):
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
        c["timecode"] = timecode(f, fps, sc["pts_time"])

    for c in cands:
        describe(c)
    old = list(cands)  # the old population: scdet, PySceneDetect and joins
    tn = load_tnet(d)
    if tn:
        hist = tnet_offsets(tn["single"], old)
        peaks = {int(k): v for k, v in hist.items() if k != "none"}
        if a.tnet_offset is not None:
            toff, how = a.tnet_offset, "forced"
        elif peaks:
            toff, how = max(peaks, key=lambda k: (peaks[k], -abs(k))), "modal"
        else:
            toff, how = TNET_OFFSET, "official convention (no sure cut with a peak)"
        one, many = tnet_align(tn["single"], toff, n), tnet_align(tn["all"], toff, n)
        have = {c["frame"] for c in old}
        for f in local_peaks(np.nan_to_num(one, nan=-1.0), TNET_MIN):
            if S <= f < E and not have & {f - 1, f, f + 1}:
                c = {"frame": int(f), "scdet_lm": False, "adaptive": None, "content": None, "join": None}
                describe(c)
                cands.append(c)
        cands.sort(key=lambda c: c["frame"])
        for c in cands:
            lo, hi = max(0, c["frame"] - 1), min(n, c["frame"] + 2)
            if np.isnan(one[lo:hi]).all():
                c["tnet"] = c["tnet_all"] = c["tnet_peak"] = None
                continue
            c["tnet"] = round(float(np.nanmax(one[lo:hi])), 6)
            c["tnet_all"] = round(float(np.nanmax(many[lo:hi])), 6)
            c["tnet_peak"] = int(lo + np.nanargmax(one[lo:hi])) if c["tnet"] >= TNET_MIN else None
        log(f"TransNetV2: offset {toff} ({how}; its peak minus the sure cut's frame: {hist}), "
            f"{len(cands) - len(old)} TransNetV2-only candidates")
    name = a.name or os.path.basename(os.path.normpath(d))
    cols = ["frame", "timecode", "class", "scdet", "mafd", "prev", "next", "local_max", "scdet_lm",
            "adaptive", "content", "ad_ratio", "content_val", "join", "burst"]
    tcols = ["tnet", "tnet_all", "tnet_peak"] if tn else []
    with open(os.path.join(d, "candidates.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols + tcols)
        for c in cands:
            w.writerow([("" if c[k] is None else (f"{c[k]:.3f}" if isinstance(c[k], float) else c[k]))
                        for k in cols] +
                       [("" if c[k] is None else (f"{c[k]:.6f}" if isinstance(c[k], float) else c[k]))
                        for k in tcols])

    sure = [c for c in old if c["class"] == "sure"]
    doubt = [c for c in old if c["class"] == "doubtful"]
    rs = s[S:E]
    bins = [(4, 6), (6, 10), (10, 14), (14, 20), (20, 30), (30, 1e9)]
    st = {
        "name": name, "source": src["spec"], "kind": src["kind"], "files": len(src["files"]),
        "fps": fps, "size": f"{src['width']}x{src['height']}", "frames_scored": n,
        "range": [S, E], "range_tc": [timecode(S, fps, sc["pts_time"]), timecode(E, fps, sc["pts_time"])],
        "frames": E - S, "duration_s": round((E - S) / float(Fraction(fps)), 1),
        "joins_in_range": sum(1 for j in joins if S <= j < E),
        "pysd": bool(pysd),
        "checks": src.get("checks"),
        "candidates": len(old), "sure": len(sure), "doubtful": len(doubt),
        "candidates_by_source": {
            "scdet_only": sum(1 for c in old if c["scdet_lm"] and c["adaptive"] is None and c["content"] is None),
            "scdet_and_pysd": sum(1 for c in old if c["scdet_lm"] and (c["adaptive"] is not None or c["content"] is not None)),
            "pysd_only": sum(1 for c in old if not c["scdet_lm"] and (c["adaptive"] is not None or c["content"] is not None)),
            "join_only": sum(1 for c in old if not c["scdet_lm"] and c["adaptive"] is None and c["content"] is None),
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
    for c in old:
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
        lmset = {c["frame"] for c in old if c["scdet_lm"]}
        pm_both = [c["frame"] for c in old
                   if c["adaptive"] is not None and c["content"] is not None and c["scdet"] < t]
        pm_one = [c["frame"] for c in old
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
    if pysd:  # each PySceneDetect detector alone
        st["pysd_detectors"] = {}
        for k, v in pysd["cuts"].items():
            det = np.array(sorted(x for x in v if S <= x < E), int)
            st["pysd_detectors"][k] = {"detections": len(det), "gaps": gap_hist(det), **shot_stats(det, (S, E), n, fps)}
    if tn:
        st["tnet"] = tnet_stats(tn, hist, toff, how, one, cands, old, sure, (S, E), n, fps)
    save_json(st, os.path.join(d, "stats.json"))
    with open(os.path.join(d, "stats.md"), "w", encoding="utf-8") as f:
        f.write(stats_markdown([st]))
    log(f"{name}: {E - S} frames, {len(old)} candidates ({len(sure)} sure, {len(doubt)} doubtful)" +
        (f", {len(cands) - len(old)} more of TransNetV2 alone" if tn else ""))
    print(stats_markdown([st]))


def load_tnet(d):
    p = os.path.join(d, "tnet.npz")
    if not os.path.exists(p):
        return None
    with np.load(p) as z:
        return {"single": z["single"].astype(np.float64), "all": z["all"].astype(np.float64),
                **{k: z[k].item() for k in ("frames", "weights_sha256", "decode_seconds", "inference_seconds",
                                            "threads")},
                "meta": json.loads(str(z["meta"]))}


def tnet_align(x, off, n):
    """TransNetV2's values in our convention (n frames): index f holds its value of frame f + off,
    NaN where it has none."""
    y = np.full(n, np.nan)
    lo, hi = max(0, -off), min(n, len(x) - off)
    if hi > lo:
        y[lo:hi] = x[lo + off:hi + off]
    return y


def tnet_offsets(single, cands):
    """Histogram of the frame of TransNetV2's single-frame peak within +-TNET_SEARCH frames minus the
    frame of each sure cut that is a scdet local maximum; "none" where no frame reaches TNET_MIN."""
    hist = {}
    for c in cands:
        if c["class"] != "sure" or not c["scdet_lm"]:
            continue
        f = c["frame"]
        lo, hi = max(0, f - TNET_SEARCH), min(len(single), f + TNET_SEARCH + 1)
        w = single[lo:hi]
        k = str(int(lo + np.argmax(w) - f)) if len(w) and w.max() >= TNET_MIN else "none"
        hist[k] = hist.get(k, 0) + 1
    return {k: hist[k] for k in sorted(hist, key=lambda k: (k == "none", int(k) if k != "none" else 0))}


def tnet_stats(tn, hist, toff, how, one, cands, old, sure, rng, n, fps):
    """TransNetV2's statistics: alignment, its own candidates, and per threshold p."""
    S, E = rng
    peaks = sum(v for k, v in hist.items() if k != "none")
    tonly = [c for c in cands if not old_population(c)]
    oldf = np.array(sorted(c["frame"] for c in old), int)

    def gap_to_old(f):
        i = np.searchsorted(oldf, f)
        return min([abs(f - int(oldf[j])) for j in (i - 1, i) if 0 <= j < len(oldf)] or [10 ** 9])

    near = {c["frame"] for c in tonly if 2 <= gap_to_old(c["frame"]) <= 3}
    pm = {k: [c for c in old if pysd_agreement(c) == k and c["scdet"] < SPTENC_T] for k in ("both", "one")}
    out = {"offset": toff, "offset_how": how, "offset_on_sure": hist,
           "offset_share": round(hist.get(str(toff), 0) / peaks, 4) if peaks else None,
           "sure_with_peak": peaks, "sure_without_peak": hist.get("none", 0),
           "decoded": tn["frames"], "commit": tn["meta"].get("commit"), "weights_sha256": tn["weights_sha256"],
           "threads": tn["threads"], "decode_seconds": tn["decode_seconds"],
           "inference_seconds": tn["inference_seconds"],
           "tnet_only": len(tonly),
           "tnet_only_by_band": {tnet_band(x): sum(1 for c in tonly if tnet_band(c["tnet"]) == tnet_band(x))
                                 for x in TNET_BANDS},
           "tnet_only_2_3_from_old": len(near),
           "possible_misses_t10": {k: len(v) for k, v in pm.items()}}
    per_p = {}
    for p in TNET_P:
        det = np.array([f for f in tnet_detections(one, p) if S <= f < E], int)
        hit = {c["frame"] for c in cands if c["tnet"] is not None and c["tnet"] >= p}
        per_p[f"{p:g}"] = {
            "detections": len(det), "gaps": gap_hist(det), **shot_stats(det, (S, E), n, fps),
            "sure_detected": sum(1 for c in sure if c["frame"] in hit),
            "candidates": len(hit),
            "old_candidates": sum(1 for c in old if c["frame"] in hit),
            "tnet_only": sum(1 for c in tonly if c["frame"] in hit),
            "tnet_only_2_3_from_old": len(near & hit),
            "possible_misses_t10_detected": {k: sum(1 for c in v if c["frame"] in hit) for k, v in pm.items()},
            "agreement": {f"{t:g}": {"both": sum(1 for c in cands if c["scdet"] >= t and c["frame"] in hit),
                                     "scdet_only": sum(1 for c in cands if c["scdet"] >= t and c["frame"] not in hit),
                                     "tnet_only": sum(1 for c in cands if c["scdet"] < t and c["frame"] in hit)}
                          for t in THRESHOLDS},
        }
    out["per_p"] = per_p
    return out


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
    more = detectors_markdown(sts)
    if more:
        out.append(more)
    if labelled:
        out.append(labelled)
    return "\n".join(out) + "\n"


def detector_cell(x):
    """detections (gaps 1/2/3/4-11/>=12), shots, median s, under 0.5/1/2 s"""
    return (f"{x['detections']} ({'/'.join(str(x['gaps'][k]) for k in ('1', '2', '3', '4-11', '>=12'))}), "
            f"{x['shots']}, {x['median_s']}, {x['under_0.5s']}/{x['under_1s']}/{x['under_2s']}")


def detectors_markdown(sts):
    """PySceneDetect's detectors alone and TransNetV2, for the episodes that have them."""
    out = []
    pd = [st for st in sts if st.get("pysd_detectors")]
    if pd:
        out.append("PySceneDetect's detectors alone: detections (gaps 1/2/3/4-11/>=12), shots, median length (s), "
                   "shots under 0.5 / 1 / 2 s:\n")
        out.append("| episode | adaptive | content |")
        out.append("|---|---|---|")
        for st in pd:
            cells = [detector_cell(st["pysd_detectors"][k]) for k in ("adaptive", "content")]
            out.append(f"| {st['name']} | " + " | ".join(cells) + " |")
        out.append("")
    tn = [st for st in sts if st.get("tnet")]
    if not tn:
        return "\n".join(out)
    ps = list(tn[0]["tnet"]["per_p"])
    bands = list(tn[0]["tnet"]["tnet_only_by_band"])
    out.append("TransNetV2: frame of its single-frame peak (within +-3) minus the sure cut's (scdet local maxima), "
               "the offset that aligns it (modal), its share of the sure cuts with a peak >= 0.1; candidates of "
               "TransNetV2 alone (aligned peaks >= 0.1, no old candidate within +-1 frame) by band "
               f"({' / '.join(bands)}), and those 2-3 frames from an old candidate:\n")
    out.append("| episode | peak offsets on sure cuts | offset | share | TransNetV2-only | by band | 2-3 frames "
               "from an old one |")
    out.append("|---|---|---:|---:|---:|---|---:|")
    for st in tn:
        t = st["tnet"]
        out.append(f"| {st['name']} | {t['offset_on_sure']} | {t['offset']} | {t['offset_share']} | "
                   f"{t['tnet_only']} | {' / '.join(str(v) for v in t['tnet_only_by_band'].values())} | "
                   f"{t['tnet_only_2_3_from_old']} |")
    out.append("")
    out.append("TransNetV2 per threshold p (one detection per run of aligned frames >= p, at its peak): "
               "detections (gaps 1/2/3/4-11/>=12), shots, median length (s), shots under 0.5 / 1 / 2 s:\n")
    out.append("| episode | " + " | ".join(f"p={p}" for p in ps) + " |")
    out.append("|---|" + "---|" * len(ps))
    for st in tn:
        out.append(f"| {st['name']} | " + " | ".join(detector_cell(st["tnet"]["per_p"][p]) for p in ps) + " |")
    out.append("")
    out.append("TransNetV2 >= p on candidates: sure cuts detected / sure cuts; PySceneDetect's possible misses at "
               "T=10 (scdet < 10) it detects, both detectors / one; TransNetV2-only candidates (2-3 frames from an "
               "old one):\n")
    out.append("| episode | " + " | ".join(f"p={p}" for p in ps) + " |")
    out.append("|---|" + "---|" * len(ps))
    for st in tn:
        t = st["tnet"]
        cells = []
        for p in ps:
            x = t["per_p"][p]
            pm = x["possible_misses_t10_detected"]
            cells.append(f"{x['sure_detected']}/{st['sure']}; {pm['both']}/{t['possible_misses_t10']['both']} / "
                         f"{pm['one']}/{t['possible_misses_t10']['one']}; {x['tnet_only']} "
                         f"({x['tnet_only_2_3_from_old']})")
        out.append(f"| {st['name']} | " + " | ".join(cells) + " |")
    out.append("")
    t = f"{SPTENC_T:g}"  # the other thresholds are in stats.json
    out.append(f"Candidates by scdet T={t} x TransNetV2 p: both / scdet only / TransNetV2 only:\n")
    out.append("| episode | " + " | ".join(f"p={p}" for p in ps) + " |")
    out.append("|---|" + "---|" * len(ps))
    for st in tn:
        ag = [st["tnet"]["per_p"][p]["agreement"][t] for p in ps]
        cells = [f"{g['both']} / {g['scdet_only']} / {g['tnet_only']}" for g in ag]
        out.append(f"| {st['name']} | " + " | ".join(cells) + " |")
    out.append("")
    return "\n".join(out)


def read_index(path):
    """Rows of a review index, in the delimiter its header uses (a spreadsheet may save ';')."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        text = f.read()
    first = text.splitlines()[0] if text else ""
    delim = ";" if first.count(";") > first.count(",") else ","
    return list(csv.DictReader(io.StringIO(text), delimiter=delim))


def norm_label(x):
    return "-".join(x.strip().lower().replace("_", " ").split())


def label_detectors(tnet):
    """(name, rule) per detector; a rule takes a candidates.csv row."""
    ds = [(f"scdet >= {t:g}", lambda c, t=t: float(c["scdet"]) >= t) for t in THRESHOLDS]
    ds += [(f"PySceneDetect {k}", lambda c, k=k: c[k] != "") for k in ("adaptive", "content")]
    if tnet:
        ds += [(f"TransNetV2 >= {p:g}", lambda c, p=p: c["tnet"] != "" and float(c["tnet"]) >= p) for p in TNET_P]
    return ds


def ratio(a, b):
    return f"{a / b:.3f}" if b > 0 else "-"


def label_table(est, dets):
    """est: (candidates.csv row, label, weight) of the labelled rows. Per detector: cuts detected and
    missed, recall, false positives by label, precision with fades and dissolves counted as
    errors, then without them (gradual transitions, where a detector may rightly cut)."""
    hard = [lab for lab in LABELS[1:] if lab not in ("fade", "dissolve")]
    out = ["| detector | cuts detected | missed | recall | " + " | ".join(f"FP {lab}" for lab in hard) +
           " | FP without fades/dissolves | fade | dissolve | precision | precision, fades/dissolves not "
           "counted |", "|---|" + "---:|" * (len(hard) + 8)]
    for name, rule in dets:
        hit = [(k, w) for c, k, w in est if rule(c)]
        tp = sum(w for k, w in hit if k == "cut")
        miss = sum(w for c, k, w in est if k == "cut") - tp
        fp = {lab: sum(w for k, w in hit if k == lab) for lab in LABELS[1:]}
        fp_hard = sum(fp[lab] for lab in hard)
        grad = fp["fade"] + fp["dissolve"]
        out.append(f"| {name} | {tp:.0f} | {miss:.0f} | {ratio(tp, tp + miss)} | " +
                   " | ".join(f"{fp[lab]:.0f}" for lab in hard) + f" | {fp_hard:.0f} | {fp['fade']:.0f} | "
                   f"{fp['dissolve']:.0f} | {ratio(tp, tp + fp_hard + grad)} | {ratio(tp, tp + fp_hard)} |")
    return out


def label_stats(index_path, dirs, kinds=None):
    """Per detector, from the review index with labels joined on (episode, frame) with the
    episodes' candidates.csv: estimated cuts detected and missed, false positives by label, recall
    and precision; per episode, then pooled by kind. The review samples within strata
    (scd_review.py): each labelled row stands for stratum_size / (labelled rows of its stratum)
    candidates, so a partly labelled index still gives estimates; strata without any label are
    counted apart."""
    kinds = {**KINDS, **(kinds or {})}
    cands = {}
    for d in dirs:
        name = load_json(os.path.join(d, "stats.json"))["name"]
        with open(os.path.join(d, "candidates.csv"), newline="", encoding="utf-8") as f:
            cands[name] = {int(r["frame"]): r for r in csv.DictReader(f)}
    tnet = bool(cands) and all("tnet" in next(iter(c.values())) for c in cands.values() if c)
    dets = label_detectors(tnet)
    eps = {}
    for r in read_index(index_path):
        eps.setdefault(r["episode"], {}).setdefault(r["stratum"], []).append(r)
    out = ["Labelled review: estimated candidates per detector (detected: scdet >= T at the candidate, a "
           "PySceneDetect detector within +-1 frame, TransNetV2's aligned probability >= p within +-1 frame; FP: "
           "detected and labelled other than cut; missed: labelled cut, not detected; recall among the cuts that "
           "are candidates, of any detector)\n"]
    est = {}
    for ep, strata in eps.items():
        n_rows = sum(len(g) for g in strata.values())
        if ep not in cands:
            out.append(f"- {ep}: not among the episode directories given, its {n_rows} rows left out")
            continue
        e, unlabelled, n_lab, lost, odd = [], 0, 0, 0, 0
        for g in strata.values():
            lab = [r for r in g if (r.get("label") or "").strip()]
            if not lab:
                unlabelled += int(float(g[0]["stratum_size"]))
                continue
            n_lab += len(lab)
            w = int(float(g[0]["stratum_size"])) / len(lab)
            for r in lab:
                k = norm_label(r["label"])
                if k not in LABELS:
                    odd += 1
                    k = "other"
                c = cands[ep].get(int(float(r["frame"])))
                if c is None:
                    lost += 1
                    continue
                e.append((c, k, w))
        est[ep] = e
        out.append(f"- {ep} ({kinds.get(ep, 'animation')}): {n_lab} labelled rows of {n_rows}; {unlabelled} "
                   "candidates in strata without a label" + (f"; {odd} labels read as other" if odd else "") +
                   (f"; {lost} rows missing from its candidates.csv" if lost else ""))
    out.append("")
    for ep, e in est.items():
        out += [f"{ep}:\n"] + label_table(e, dets) + [""]
    pools = {}
    for ep in est:
        pools.setdefault(kinds.get(ep, "animation"), []).append(ep)
    if len(est) > 1:
        for kind, members in sorted(pools.items()):
            out += [f"Pooled, {kind} ({', '.join(members)}):\n"]
            out += label_table([x for ep in members for x in est[ep]], dets) + [""]
        if len(pools) > 1:
            out += [f"Pooled, every episode ({', '.join(est)}):\n"]
            out += label_table([x for e in est.values() for x in e], dets) + [""]
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


def local_peaks(s, floor):
    left = np.concatenate(([np.inf], s[:-1]))
    right = np.concatenate((s[1:], [-np.inf]))
    return np.nonzero((s > left) & (s >= right) & (s >= floor))[0]


def gap_hist(det):
    g = np.diff(det)
    return {"1": int((g == 1).sum()), "2": int((g == 2).sum()), "3": int((g == 3).sum()),
            "4-11": int(((g >= 4) & (g <= 11)).sum()), ">=12": int((g >= 12).sum())}


def cmd_compare(a):
    ra, rb = load_scores(a.ref), load_scores(a.other)
    sa, sb, ma, mb = ra["score"], rb["score"], ra["mafd"], rb["mafd"]
    na, nb = len(sa), len(sb)
    pa, pb = set(local_peaks(sa, a.peak).tolist()), local_peaks(sb, a.peak)
    best, off = -1, 0
    for d in range(-a.max_offset, a.max_offset + 1):  # other frame i is reference frame i + off
        c = sum(1 for x in pb if x + d in pa)
        if c > best:
            best, off = c, d
    lo, hi = max(0, -off), min(nb, na - off)
    ib = np.arange(lo, hi)
    ia = ib + off
    xa, xb = sa[ia], sb[ib]
    res = {"ref": a.ref, "other": a.other, "frames_ref": na, "frames_other": nb, "offset": off,
           "overlap": len(ib), "peaks": {"floor": a.peak, "ref": len(pa), "other": len(pb), "matched": best}}

    def dist(v):
        return {"median": round(float(np.median(v)), 3), "p95": round(float(np.percentile(v, 95)), 3),
                "p99": round(float(np.percentile(v, 99)), 3), "max": round(float(v.max()), 3)}

    res["score_absdiff"] = dist(np.abs(xa - xb))
    act = (xa >= 6) | (xb >= 6)
    res["score_absdiff_where_either_ge6"] = dist(np.abs(xa - xb)[act]) if act.any() else None
    res["mafd_absdiff"] = dist(np.abs(ma[ia] - mb[ib]))
    res["held_frames_share"] = {"ref": round(float((ma[ia] < 0.5).mean()), 3),
                                "other": round(float((mb[ib] < 0.5).mean()), 3)}
    per_t = {}
    for t in THRESHOLDS:
        da, db = set(np.nonzero(xa >= t)[0].tolist()), set(np.nonzero(xb >= t)[0].tolist())
        same = da & db
        rest_a, rest_b = da - same, db - same
        near = 0
        for x in sorted(rest_b):
            for y in (x - 1, x + 1):
                if y in rest_a:
                    rest_a.discard(y)
                    near += 1
                    break
        per_t[f"{t:g}"] = {"ref": len(da), "other": len(db), "same_frame": len(same), "one_frame_apart": near,
                           "only_ref": len(da) - len(same) - near, "only_other": len(db) - len(same) - near,
                           "gaps_ref": gap_hist(np.array(sorted(da))), "gaps_other": gap_hist(np.array(sorted(db)))}
    res["per_threshold"] = per_t
    src = load_json(os.path.join(a.other, "source.json"))
    joins = np.array([g["start"] for g in src.get("segments", [])[1:] if lo <= g["start"] < hi], int)
    if len(joins):
        ja, jb = sa[joins + off], sb[joins]
        order = np.argsort(ja)
        res["joins"] = {
            "n": len(joins), "ref_min": round(float(ja.min()), 3), "other_min": round(float(jb.min()), 3),
            "ref_below": {f"{t:g}": int((ja < t).sum()) for t in (4, 6, 8, 10, 14)},
            "other_below": {f"{t:g}": int((jb < t).sum()) for t in (4, 6, 8, 10, 14)},
            "absdiff": dist(np.abs(ja - jb)),
            "other_higher_by_1": int((jb > ja + 1).sum()), "ref_higher_by_1": int((ja > jb + 1).sum()),
            "lowest_in_ref": [(int(joins[i]), round(float(ja[i]), 3), round(float(jb[i]), 3)) for i in order[:12]],
        }
    lines = [f"# {os.path.basename(os.path.normpath(a.other))} against {os.path.basename(os.path.normpath(a.ref))}",
             "", f"Frames: reference {na}, other {nb}; other frame i = reference frame i {off:+d}; "
             f"{best} of the other's {len(pb)} peaks >= {a.peak:g} on a reference peak (reference: {len(pa)}).",
             f"Score |difference| per frame: {res['score_absdiff']}; where either scores >= 6: "
             f"{res['score_absdiff_where_either_ge6']}; mafd: {res['mafd_absdiff']}; held frames (mafd < 0.5): "
             f"{res['held_frames_share']}.", "",
             "| T | reference | other | same frame | 1 frame apart | only reference | only other | "
             "gaps 1/2/3/4-11/>=12 reference | other |", "|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for t, v in per_t.items():
        g1 = "/".join(str(x) for x in v["gaps_ref"].values())
        g2 = "/".join(str(x) for x in v["gaps_other"].values())
        lines.append(f"| {t} | {v['ref']} | {v['other']} | {v['same_frame']} | {v['one_frame_apart']} | "
                     f"{v['only_ref']} | {v['only_other']} | {g1} | {g2} |")
    if "joins" in res:
        j = res["joins"]
        lines += ["", f"Segment joins ({j['n']}): reference scores below 4/6/8/10/14: "
                  f"{'/'.join(str(x) for x in j['ref_below'].values())} (min {j['ref_min']}); the other's: "
                  f"{'/'.join(str(x) for x in j['other_below'].values())} (min {j['other_min']}); |difference| "
                  f"{j['absdiff']}; other higher by > 1: {j['other_higher_by_1']}, reference higher by > 1: "
                  f"{j['ref_higher_by_1']}. Lowest in the reference (join, reference, other): {j['lowest_in_ref']}"]
    md = "\n".join(lines) + "\n"
    print(md)
    if a.md:
        with open(a.md, "w", encoding="utf-8") as f:
            f.write(md)
        save_json(res, os.path.splitext(a.md)[0] + ".json")


ROUND_LETTER = {"c": "cut", "f": "flash", "p": "pan", "d": "fade", "o": "other", "n": "not-a-cut"}


def read_round_labels(path):
    """{row: label} from a round's labels.txt: a row number then one letter (scd_review.py round);
    '?' and empty rows are not labelled, an unknown letter reads as other."""
    labels, odd = {}, 0
    with open(path, encoding="utf-8-sig") as f:
        for ln in f:
            m = re.match(r"\s*(\d+)\s*(\S?)", ln)
            if ln.lstrip().startswith("#") or not m or not m.group(2) or m.group(2) == "?":
                continue
            k = ROUND_LETTER.get(m.group(2).lower())
            if k is None:
                odd += 1
                k = "other"
            labels[int(m.group(1))] = k
    return labels, odd


def wilson(k, n, z=1.645):
    """90% Wilson interval of a share k/n."""
    if n == 0:
        return math.nan, math.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def cval(c, k):
    """A candidate's numeric value (candidates.csv or a round's rows.csv), 0 when empty or NaN."""
    try:
        v = float(c.get(k) or 0.0)
    except ValueError:
        return 0.0
    return 0.0 if math.isnan(v) else v


REFINED = ("R1", "R2", "R3m", "R3z", "R4h", "R4l", "R5", "R5b", "R6hm", "R6hz", "R6l", "R6z")
STILL_MAFD = 4.0  # MAFD under this: the picture hardly changes at the candidate (round 2's gate question)


def refined_group(c):
    """Round 2's cells, a refinement of round 1's agreement groups by TransNetV2's band (>= 0.5,
    0.3-0.5, 0.1-0.3, under 0.1) and, where it alone fires, the picture's change (MAFD): R1 all three
    (both PySceneDetect detectors), R2 TransNetV2 >= 0.5 with scdet >= 10 or PySceneDetect, R3m/R3z
    TransNetV2 >= 0.5 alone, the picture moving / still (MAFD < STILL_MAFD), R4h/R4l TransNetV2 at
    0.3-0.5 / 0.1-0.3 with another detector, R5 the others with TransNetV2 under 0.1, R5b lone scdet
    hits inside a burst, R6hm/R6hz TransNetV2 at 0.3-0.5 alone (moving / still), R6l at 0.1-0.3
    alone, R6z none of them."""
    tn, n_p = cval(c, "tnet"), (c["adaptive"] != "") + (c["content"] != "")
    S, P = cval(c, "scdet") >= 10, n_p > 0
    still = cval(c, "mafd") < STILL_MAFD
    if tn >= 0.5:
        return "R1" if S and n_p == 2 else "R2" if S or P else "R3z" if still else "R3m"
    if tn >= 0.3:
        return "R4h" if S or P else "R6hz" if still else "R6hm"
    if tn >= 0.1:
        return "R4l" if S or P else "R6l"
    if S and not P and int(c.get("burst") or 0) >= 2:
        return "R5b"
    return "R5" if S or P else "R6z"


def round_detectors():
    """summary's detectors, then combinations the labels can weigh: scdet without its bursts,
    TransNetV2 with a second detector, and TransNetV2 gated on a picture change (MAFD)."""
    return label_detectors(True) + [
        ("scdet >= 10 outside bursts", lambda c: cval(c, "scdet") >= 10 and int(c["burst"] or 0) < 2),
        ("scdet >= 8 outside bursts", lambda c: cval(c, "scdet") >= 8 and int(c["burst"] or 0) < 2),
        ("TransNetV2 >= 0.5 or scdet >= 10", lambda c: cval(c, "tnet") >= 0.5 or cval(c, "scdet") >= 10),
        ("TransNetV2 >= 0.5 or PySceneDetect content", lambda c: cval(c, "tnet") >= 0.5 or c["content"] != ""),
        ("TransNetV2 >= 0.3 or PySceneDetect content", lambda c: cval(c, "tnet") >= 0.3 or c["content"] != ""),
        ("TransNetV2 >= 0.3, MAFD >= 1", lambda c: cval(c, "tnet") >= 0.3 and cval(c, "mafd") >= 1),
        ("TransNetV2 >= 0.3, MAFD >= 1.5", lambda c: cval(c, "tnet") >= 0.3 and cval(c, "mafd") >= 1.5),
        ("TransNetV2 >= 0.3, MAFD >= 2", lambda c: cval(c, "tnet") >= 0.3 and cval(c, "mafd") >= 2),
        ("TransNetV2 >= 0.3, MAFD >= 3", lambda c: cval(c, "tnet") >= 0.3 and cval(c, "mafd") >= 3),
        ("TransNetV2 >= 0.5, MAFD >= 3", lambda c: cval(c, "tnet") >= 0.5 and cval(c, "mafd") >= 3),
    ]


def round_rates(est, rule):
    """Recall and precision (every other label an error) of a rule over weighted labelled rows."""
    cuts = sum(w for c, k, w in est if k == "cut")
    det = sum(w for c, k, w in est if rule(c))
    tp = sum(w for c, k, w in est if k == "cut" and rule(c))
    return (tp / cuts if cuts else math.nan), (tp / det if det else math.nan)


def round_boot(cells, dets, n, seed):
    """Per detector: recall and precision with 90% intervals from a bootstrap of the labelled rows
    within each cell (the round's strata), and the paired differences to scdet >= 10."""
    rng = np.random.default_rng(seed)
    full = [(c, k, size / len(lab)) for size, lab in cells for c, k in lab]
    ref = [name for name, _ in dets].index("scdet >= 10")
    point = [round_rates(full, rule) for _, rule in dets]
    draws = np.full((n, len(dets), 2), np.nan)
    for b in range(n):
        est = []
        for size, lab in cells:
            idx = rng.integers(0, len(lab), len(lab))
            est += [(lab[i][0], lab[i][1], size / len(lab)) for i in idx]
        draws[b] = [round_rates(est, rule) for _, rule in dets]
    diff = draws - draws[:, ref:ref + 1, :]
    ci = lambda x: tuple(np.nanpercentile(x, (5, 95))) if np.isfinite(x).any() else (math.nan, math.nan)  # noqa: E731
    return [(name, point[i], [ci(draws[:, i, j]) for j in (0, 1)],
             [(point[i][j] - point[ref][j], ci(diff[:, i, j])) for j in (0, 1)]) for i, (name, _) in enumerate(dets)]


def cmd_round(a):
    """Estimates from rounds of labels (scd_review.py round): each labelled row stands for its
    cell's candidates / the cell's labelled rows; cells without a label are counted apart. One
    round: its own cells (rows.csv's group and cell_size). Several, or --dirs: every row in its
    refined cell (refined_group), the cells' sizes counted in the episodes' candidates.csv."""
    if len(a.round) > 1 and not a.dirs:
        sys.exit("several rounds are pooled in refined cells: --dirs (the episode directories) is needed")
    rows, labels, odd = [], {}, 0
    for i, rd in enumerate(a.round):
        with open(os.path.join(rd, "rows.csv"), newline="", encoding="utf-8") as f:
            rs = [r for r in csv.DictReader(f) if r["episode"] not in a.skip]
        lab, o = read_round_labels(os.path.join(rd, "labels.txt"))
        odd += o
        for r in rs:
            r["round_row"] = f"{i + 1}.{int(r['round_row']):03d}"  # unique across rounds
            k = int(r["round_row"].split(".")[1])
            if k in lab:
                labels[r["round_row"]] = lab[k]
        rows += rs
    if a.dirs:
        sizes = {}
        for d in a.dirs:
            name = load_json(os.path.join(d, "stats.json"))["name"]
            if name in a.skip:
                continue
            kind = KINDS.get(name, "animation")
            with open(os.path.join(d, "candidates.csv"), newline="", encoding="utf-8") as f:
                for c in csv.DictReader(f):
                    sizes[(refined_group(c), kind)] = sizes.get((refined_group(c), kind), 0) + 1
        for r in rows:
            r["group"] = refined_group(r)
            r["cell_size"] = sizes.get((r["group"], r["kind"]), 0)
    cells = {}
    for r in rows:
        cells.setdefault((r["group"], r["kind"]), []).append(r)
    empty = {}  # refined cells with candidates but no row in any round
    if a.dirs:
        for (g, kind), size in sorted(sizes.items()):
            if (g, kind) not in cells:
                empty.setdefault(kind, []).append(f"{g} {size}")
    out = [f"Round {', '.join(a.round)}: {len(labels)} of {len(rows)} rows labelled" +
           (f"; {odd} letters read as other" if odd else "") + (f"; left out: {', '.join(a.skip)}" if a.skip else "") +
           (" (refined cells)" if a.dirs else "") + ". Labels: c cut, f flash, p pan or motion, d fade or dissolve "
           "(the tables' fade), o other, n nothing (not-a-cut).\n",
           "Per agreement group and kind (scd_review.py round): candidates, rows labelled, labels, the share of "
           "cuts with its 90% Wilson interval:\n",
           "| group | kind | candidates | labelled | " + " | ".join(LABELS) + " | cuts | 90% interval |",
           "|---|---|---:|---:|" + "---:|" * (len(LABELS) + 2)]
    est, unlabelled, strata = {}, {}, {}
    for (g, kind), rs in sorted(cells.items()):
        size = int(rs[0]["cell_size"])
        lab = [(r, labels[r["round_row"]]) for r in rs if r["round_row"] in labels]
        n = len(lab)
        cnt = {k: sum(1 for _, x in lab if x == k) for k in LABELS}
        lo, hi = wilson(cnt["cut"], n)
        out.append(f"| {g} | {kind} | {size} | {n} of {len(rs)} | " + " | ".join(str(cnt[k]) for k in LABELS) +
                   f" | {ratio(cnt['cut'], n)} | {lo:.2f}-{hi:.2f} |" if n else
                   f"| {g} | {kind} | {size} | 0 of {len(rs)} | " + " | ".join("" for _ in LABELS) + " | - | - |")
        if not n:
            unlabelled[kind] = unlabelled.get(kind, 0) + size
            continue
        strata.setdefault(kind, []).append((size, lab))
        for r, k in lab:
            est.setdefault(kind, []).append((r, k, size / n))
    for kind, cs in sorted(empty.items()):
        out.append(f"- {kind}: cells without any row, left out of the estimates: {', '.join(cs)} candidates")
        unlabelled[kind] = unlabelled.get(kind, 0) + sum(int(c.split()[1]) for c in cs)
    out.append("")
    dets = round_detectors()
    for kind in sorted(est):
        miss = f"; {unlabelled[kind]} candidates in cells without a label" if kind in unlabelled else ""
        out += [f"Estimated per detector, {kind} (each labelled row stands for its cell's candidates / the "
                f"cell's labelled rows{miss}):\n"] + label_table(est[kind], dets) + [""]
        out += [f"{kind}: recall and precision (every other label an error) with 90% intervals from {a.boot} "
                "bootstraps of the labelled rows within their cells, and the paired differences to scdet >= 10:\n",
                "| detector | recall | 90% | precision | 90% | recall - scdet 10 | 90% | precision - scdet 10 | 90% |",
                "|---|---:|---|---:|---|---:|---|---:|---|"]
        fmt2 = lambda x: f"{x:.2f}" if x == x else "-"  # noqa: E731
        for name, (rc, pr), (rci, pci), ((dr, drci), (dp, dpci)) in round_boot(strata[kind], dets, a.boot, a.seed):
            out.append(f"| {name} | {fmt2(rc)} | {fmt2(rci[0])}-{fmt2(rci[1])} | {fmt2(pr)} | "
                       f"{fmt2(pci[0])}-{fmt2(pci[1])} | {dr:+.2f} | {fmt2(drci[0])}-{fmt2(drci[1])} | {dp:+.2f} | "
                       f"{fmt2(dpci[0])}-{fmt2(dpci[1])} |" if dr == dr and dp == dp else
                       f"| {name} | {fmt2(rc)} | {fmt2(rci[0])}-{fmt2(rci[1])} | {fmt2(pr)} | "
                       f"{fmt2(pci[0])}-{fmt2(pci[1])} | - | | - | |")
        out.append("")
    if len(est) > 1:
        out += ["Estimated per detector, every kind:\n"] + label_table([x for e in est.values() for x in e], dets)
    md = "\n".join(out) + "\n"
    if a.md:
        with open(a.md, "w", encoding="utf-8") as f:
            f.write(md)
    print(md)


def cmd_summary(a):
    sts = [load_json(os.path.join(d, "stats.json")) for d in a.dirs]
    kinds = dict(k.split("=", 1) for k in a.kind or [])
    lab = label_stats(a.labels, a.dirs, kinds) if a.labels else None
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
    p.add_argument("--no-reinit", action="store_true",
                   help="-reinit_filter 0: one filtergraph for every frame even if the stream's parameters change")
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
    tnet_dir = os.environ.get("TNET_DIR")
    tnet_weights = os.environ.get("TNET_WEIGHTS") or (
        tnet_dir and os.path.join(tnet_dir, "inference-pytorch", "transnetv2-pytorch-weights.pth"))
    for name in ("tnet", "tnet-check"):
        p = sub.add_parser(name)
        if name == "tnet-check":
            p.add_argument("source", help="a video file")
        p.add_argument("--out", required=True)
        p.add_argument("--tnet-dir", default=tnet_dir, required=not tnet_dir,
                       help="checkout of github.com/soCzech/TransNetV2 (default: TNET_DIR)")
        p.add_argument("--weights", default=tnet_weights, required=not tnet_weights,
                       help="converted PyTorch weights (default: TNET_WEIGHTS, else convert_weights.py's output "
                            "in the checkout)")
        p.add_argument("--threads", type=int, default=16, help="torch threads, and ffmpeg's")
        p.add_argument("--batch", type=int, default=1, help="windows per forward (officially 1)")
    p.add_argument("--tf-weights", help="the official TF SavedModel directory (default: the checkout's)")
    p.add_argument("--frames", type=int, default=3000, help="frames of the excerpt")
    p.add_argument("--start", type=int, default=0, help="first frame of the excerpt (decoded from 0)")
    p.add_argument("--random", type=int, default=1000, help="random frames compared first")
    p.add_argument("--seed", type=int, default=1)
    p = sub.add_parser("analyse")
    p.add_argument("--out", required=True)
    p.add_argument("--name")
    p.add_argument("--tnet-offset", type=int,
                   help="TransNetV2's frame offset to ours (default: the modal one on sure cuts)")
    p = sub.add_parser("activity")
    p.add_argument("--out", required=True)
    p.add_argument("--block", type=float, default=5.0, help="minutes")
    p.add_argument("--window", type=float, default=25.0, help="minutes")
    p.add_argument("--step", type=float, default=1.0, help="minutes")
    p.add_argument("--skip-head", type=float, default=5.0, help="minutes left out at the start")
    p.add_argument("--skip-tail", type=float, default=10.0, help="minutes left out at the end (credits)")
    p.add_argument("--top", type=int, default=8)
    p = sub.add_parser("compare")
    p.add_argument("ref", help="scored directory of the reference (the original)")
    p.add_argument("other", help="scored directory of the other version (e.g. its segments)")
    p.add_argument("--peak", type=float, default=20.0, help="peak floor for the offset search")
    p.add_argument("--max-offset", type=int, default=500)
    p.add_argument("--md", help="write the tables there (and the figures to the same name .json)")
    p = sub.add_parser("summary")
    p.add_argument("dirs", nargs="+")
    p.add_argument("--labels", help="review index CSV with the label column filled")
    p.add_argument("--kind", action="append", metavar="NAME=KIND",
                   help="pool an episode with this kind (default: KINDS, else animation)")
    p.add_argument("--md", help="also write the tables to this file")
    p = sub.add_parser("round")
    p.add_argument("round", nargs="+", help="rounds' directories (scd_review.py round): rows.csv, labels.txt")
    p.add_argument("--dirs", nargs="*", default=[], help="episode directories: pool the rows in refined cells")
    p.add_argument("--skip", nargs="*", default=[], metavar="NAME", help="episodes left out")
    p.add_argument("--boot", type=int, default=2000, help="bootstrap draws for the intervals")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--md", help="also write the tables to this file")
    a = ap.parse_args()
    {"check": cmd_check, "score": cmd_score, "pysd": cmd_pysd, "pysd-check": cmd_pysd_check, "tnet": cmd_tnet,
     "tnet-check": cmd_tnet_check, "analyse": cmd_analyse, "activity": cmd_activity, "compare": cmd_compare,
     "summary": cmd_summary, "round": cmd_round}[a.cmd](a)


if __name__ == "__main__":
    main()

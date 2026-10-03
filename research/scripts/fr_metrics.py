#!/usr/bin/env python3
"""Full-reference metrics of SeedVR2 outputs against a ground truth, and their summary tables.

  fr_metrics.py GT.mkv --clip NAME --out VARIANT SEED OUT.mkv [--out ...] --json-dir DIR
                [--rows Y0:Y1] [--frames N] [--no-vmaf] [--no-deep] [--batch B] [--threads T]
  fr_metrics.py --summary DIR_OR_JSON ... [--default VARIANT] [--reference bicubic] [--block L]
  fr_metrics.py --make-test GT.mkv --work DIR [--blur 1.0] [--noise 2.0] [--seed S]  # test inputs

Why: with a ground truth (fr_clips.py), a change of SeedVR2's numerics or input preparation can
be scored as closer to or further from the truth, instead of only "different". One call scores
any number of outputs of one clip in lockstep (the ground truth is decoded, and its deep
features computed, once), and writes one JSON per output: CLIP.VARIANT.sSEED.json.

Inputs: RGB masters (FFV1 gbrp16le from ffv1_out.py, or any RGB: gbrp10le, 8-bit bgr0...), the
same size as the ground truth, or taller with --rows Y0:Y1 (rows Y0..Y1-1 of every input taller
than Y1-Y0 are compared, e.g. --rows 4:1076 for a 1080-row output against the 1072-row crop).
Only the first N frames common to every input are compared.

Per frame (Y = BT.709 luma of the full-range RGB, 8-bit scale):
- psnr_y: PSNR of Y. psnr_y_bottom16 / psnr_y_rest: the same over the bottom 16 rows and over the
  others (the CLI pads the bottom to a multiple of 16: an edge effect shows as a gap)
- ssim_y: SSIM of Y, 11x11 Gaussian window, sigma 1.5, as quality_metrics.py computes it
- lpips: LPIPS v0.1, AlexNet (the lpips package), RGB in [-1, 1], at full resolution
- dists: DISTS (the authors' DISTS_pytorch, weights included), RGB in [0, 1], at full resolution:
  no resize to 256 px as in the paper's protocol (piq's DISTS resizes; it agrees with this one to
  6 decimals at 256x256), since 1080p detail is what is measured
- de00_lf: mean CIEDE2000 between output and ground truth after a Gaussian blur (sigma 4 px) in
  CIELAB (OpenCV, sRGB/D65): low-frequency colour and brightness error, blind to detail; computed
  on every second pixel of the blurred images (the blur leaves nothing above that sampling rate)
- vmaf: VMAF as sptenc measures it: model vmaf_v1.0.16_3d0h (1080p, 3 picture heights) with its
  CAMBI feature clipped to 0 ("fidelity", what `sptenc vmaf` reports first); vmaf_neg: the v0
  model vmaf_v0.6.1neg (no enhancement gain); cambi_added: banding the output adds to the ground
  truth (libvmaf CAMBI full-reference, sptenc's options). Both RGB inputs go through sptenc's own
  RGB -> YUV conversion (--vmaf-convert sptenc: swscale BT.709, limited range, chroma left,
  yuv420p10le) or zscale's (--vmaf-convert zscale), then libvmaf (distorted first, reference
  second, frames paired by index), as `sptenc vmaf` runs it
Per transition t (frames t-1 -> t), with D = Y_out - Y_gt:
- temporal_full: mean |D_t - D_t-1| = |(out_t - out_t-1) - (gt_t - gt_t-1)|: the change between
  frames that the ground truth does not have (flicker, crawling detail), full resolution
- temporal_lf: the same on 16x16-pixel block means: low-frequency flicker (brightness/colour)
A run's figure is the mean of its per-frame (per-transition) series; PSNR is the mean of the
per-frame PSNRs. The JSON keeps every series, frame-aligned, for the paired statistics.

--summary reads the JSONs (files, or directories of them), groups them by clip, variant and
seed, and prints Markdown tables:
- the mean of each metric per clip and variant (over its seeds), reference variants (the bicubic
  baseline...) included as rows
- the seed band: per clip, the spread (max - min) of the default variant's per-seed means
- per variant, per clip, the paired difference to the default variant run with the same seed,
  frame by frame (transition by transition), pooled over the common seeds: mean, 95% CI from a
  moving-block bootstrap over frames (blocks of --block frames: neighbouring frames are
  correlated), and a verdict: "better" / "worse" when the CI excludes 0 and the difference
  exceeds the seed band, "within" otherwise; then, per variant and metric, the count of clips
  better / worse / within

--make-test writes the ground truth blurred (Gaussian, sigma --blur px) and with Gaussian noise
(sigma --noise 8-bit levels, --seed) as 16-bit RGB masters, to check the metrics on known cases.

Needs numpy, scipy, scikit-image, opencv-python(-headless), torch + torchvision (CPU is fine),
lpips, DISTS_pytorch (the metrics venv), and ffmpeg with libvmaf 3.2+ (the v1 models) and zimg.
"""
import argparse
import glob
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from fractions import Fraction

import cv2
import numpy as np

FFMPEG = os.environ.get("FR_FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = os.environ.get("FR_FFPROBE") or shutil.which("ffprobe") or "ffprobe"
RGB_BITS = {"bgr0": 8, "gbrp": 8, "gbrp10le": 10, "gbrp12le": 12, "gbrp16le": 16}

SEP = r"\\:"  # ':' escaped for the option value, then for the filtergraph (sptenc's libvmafParamSeparator)
VMAF_MODELS = (f"version=vmaf_v1.0.16_3d0h{SEP}cambi.cambi_max_val=0{SEP}name=vmaf|"
               f"version=vmaf_v0.6.1neg{SEP}name=vmaf_neg")
CAMBI_FEATURE = (f"name=cambi{SEP}full_ref=true{SEP}cambi_high_res_speedup=1080"
                 f"{SEP}cambi_vis_lum_threshold=0.06")
VMAF_CONVERT = {
    # sptenc's RGBToYUVFilter (ffmpeg/rgb.go) with the BT.709 matrix, then vmafSameMatrix
    "sptenc": "scale=out_color_matrix=bt709:out_range=tv:out_chroma_loc=left,format=yuv420p10le",
    "zscale": "zscale=matrix=709:range=limited:chromal=left:dither=none,format=yuv420p10le",
}

# key, column title, higher is better, format
METRICS = [
    ("psnr_y", "PSNR-Y", True, "{:.2f}"),
    ("ssim_y", "SSIM-Y", True, "{:.4f}"),
    ("lpips", "LPIPS", False, "{:.4f}"),
    ("dists", "DISTS", False, "{:.4f}"),
    ("vmaf", "VMAF", True, "{:.2f}"),
    ("vmaf_neg", "VMAF NEG", True, "{:.2f}"),
    ("de00_lf", "ΔE00 lf", False, "{:.3f}"),
    ("temporal_lf", "T-err lf", False, "{:.3f}"),
    ("temporal_full", "T-err", False, "{:.3f}"),
    ("psnr_y_bottom16", "PSNR-Y bottom 16", True, "{:.2f}"),
    ("psnr_y_rest", "PSNR-Y rest", True, "{:.2f}"),
    ("cambi_added", "CAMBI added", False, "{:.3f}"),
]
PER_TRANSITION = ("temporal_lf", "temporal_full")


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def probe(path):
    r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=pix_fmt,width,height,r_frame_rate", "-of", "json", path],
                       capture_output=True, text=True)
    if r.returncode or not json.loads(r.stdout or "{}").get("streams"):
        raise SystemExit(f"ffprobe {path}: {r.stderr.strip() or 'no video stream'}")
    return json.loads(r.stdout)["streams"][0]


def parse_rows(spec):
    if not spec:
        return None
    y0, y1 = (int(v) for v in spec.split(":"))
    if y1 <= y0:
        raise SystemExit(f"--rows {spec}: empty")
    return y0, y1


class Source:
    """An RGB master decoded by ffmpeg as stored, yielding float32 RGB [H, W, 3] in [0, 1]."""

    def __init__(self, path, rows=None, threads=8):
        st = probe(path)
        self.path, self.fmt = path, st["pix_fmt"]
        if self.fmt not in RGB_BITS:
            raise SystemExit(f"{path}: {self.fmt}, an RGB master (gbrp*, or FFV1's 8-bit bgr0) is expected")
        self.W, self.H, self.bits = st["width"], st["height"], RGB_BITS[self.fmt]
        self.fps = st.get("r_frame_rate")
        self.rows = rows if rows and self.H != rows[1] - rows[0] else None
        if self.rows and self.rows[1] > self.H:
            raise SystemExit(f"{path}: {self.H} rows, --rows {self.rows[0]}:{self.rows[1]} does not fit")
        self.h = self.rows[1] - self.rows[0] if self.rows else self.H
        self.dtype = np.dtype(np.uint8) if self.bits == 8 else np.dtype("<u2")
        self.size = (4 if self.fmt == "bgr0" else 3) * self.W * self.H * self.dtype.itemsize
        self.proc = subprocess.Popen([FFMPEG, "-v", "error", "-nostdin", "-threads", str(threads), "-i", path,
                                      "-map", "0:v:0", "-fps_mode", "passthrough", "-f", "rawvideo",
                                      "-pix_fmt", self.fmt, "-"], stdout=subprocess.PIPE)

    def read(self):
        buf = self.proc.stdout.read(self.size)
        if len(buf) < self.size:
            return None
        if self.fmt == "bgr0":
            x = np.frombuffer(buf, self.dtype).reshape(self.H, self.W, 4)[..., 2::-1]
        else:
            g, b, r = np.frombuffer(buf, self.dtype).reshape(3, self.H, self.W)
            x = np.stack([r, g, b], axis=-1)
        x = x.astype(np.float32) * np.float32(1.0 / ((1 << self.bits) - 1))
        if self.rows:
            x = x[self.rows[0]:self.rows[1]]
        return np.ascontiguousarray(x)

    def close(self):
        if self.proc:
            self.proc.kill()  # read-only decode: no broken-pipe messages
            self.proc.stdout.close()
            self.proc.wait()
            self.proc = None


# ------------------------------------------------------------------ pixel metrics

def luma(rgb):
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]) * np.float32(255)


def ssim(y1, y2):
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)  # noqa: E731
    m1, m2 = blur(y1), blur(y2)
    s11, s22, s12 = blur(y1 * y1) - m1 * m1, blur(y2 * y2) - m2 * m2, blur(y1 * y2) - m1 * m2
    v = ((2 * m1 * m2 + c1) * (2 * s12 + c2)) / ((m1 * m1 + m2 * m2 + c1) * (s11 + s22 + c2))
    return float(v.mean())


def lab_lf(rgb):
    """CIELAB, blurred (sigma 4 px), every second pixel."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
    return cv2.GaussianBlur(lab, (0, 0), 4)[::2, ::2]


def block_mean(x, b=16):
    h, w = x.shape[:2]
    return cv2.resize(x, (max(1, w // b), max(1, h // b)), interpolation=cv2.INTER_AREA)


def psnr(mse):
    return 10 * math.log10(255.0 ** 2 / mse) if mse > 0 else float("inf")


# ------------------------------------------------------------------ deep metrics

class Deep:
    """LPIPS and DISTS on the CPU, channels-last (oneDNN's layout: VGG16 features 1.3-1.8x faster on
    the measurement box, scores equal to 6e-8)."""

    def __init__(self, threads):
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")  # pretrained= in lpips/DISTS
        import torch
        import lpips
        import DISTS_pytorch
        torch.set_num_threads(threads)
        self.torch = torch
        cl = torch.channels_last
        self.lpips = lpips.LPIPS(net="alex", version="0.1", verbose=False).eval().to(memory_format=cl)
        self.dists = DISTS_pytorch.DISTS(load_weights=False).eval().to(memory_format=cl)
        w = torch.load(os.path.join(os.path.dirname(DISTS_pytorch.__file__), "weights.pt"), map_location="cpu")
        self.dists.alpha.data, self.dists.beta.data = w["alpha"], w["beta"]
        self.versions = {"torch": torch.__version__, "lpips": getattr(lpips, "__version__", "?"),
                         "dists": "DISTS_pytorch " + getattr(DISTS_pytorch, "__version__", "0.1"), "threads": threads}
        self.t_lpips = self.t_dists = 0.0
        self.checked = False

    def tensor(self, frames):
        return self.torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous(
            memory_format=self.torch.channels_last)

    def dists_score(self, f0, f1):
        m = self.dists
        c1 = c2 = 1e-6
        w_sum = m.alpha.sum() + m.beta.sum()
        alpha = self.torch.split(m.alpha / w_sum, m.chns, dim=1)
        beta = self.torch.split(m.beta / w_sum, m.chns, dim=1)
        d1 = d2 = 0
        for k in range(len(m.chns)):
            xm, ym = f0[k].mean([2, 3], keepdim=True), f1[k].mean([2, 3], keepdim=True)
            d1 = d1 + (alpha[k] * (2 * xm * ym + c1) / (xm ** 2 + ym ** 2 + c1)).sum(1, keepdim=True)
            xv, yv = ((f0[k] - xm) ** 2).mean([2, 3], keepdim=True), ((f1[k] - ym) ** 2).mean([2, 3], keepdim=True)
            cov = (f0[k] * f1[k]).mean([2, 3], keepdim=True) - xm * ym
            d2 = d2 + (beta[k] * (2 * cov + c2) / (xv + yv + c2)).sum(1, keepdim=True)
        return (1 - (d1 + d2)).flatten()

    def run(self, gt_frames, outs_frames):
        """LPIPS and DISTS of each output batch vs the ground-truth batch: lists of per-frame values."""
        torch = self.torch
        with torch.inference_mode():
            g = self.tensor(gt_frames)
            t0 = time.perf_counter()
            gf = self.dists.forward_once(g)
            self.t_dists += time.perf_counter() - t0
            res = []
            for frames in outs_frames:
                o = self.tensor(frames)
                t0 = time.perf_counter()
                lp = self.lpips(o * 2 - 1, g * 2 - 1).flatten()
                t1 = time.perf_counter()
                of = self.dists.forward_once(o)
                ds = self.dists_score(of, gf)
                t2 = time.perf_counter()
                self.t_lpips += t1 - t0
                self.t_dists += t2 - t1
                if not self.checked:  # the shared-feature score must be DISTS_pytorch's own
                    ref = self.dists(o[:1], g[:1]).flatten()
                    diff = float((ref - ds[:1]).abs().max())
                    if diff > 1e-5:
                        raise SystemExit(f"DISTS from shared features differs from DISTS_pytorch by {diff}")
                    log(f"DISTS from shared ground-truth features = DISTS_pytorch's forward (max |diff| {diff:.1e})")
                    self.checked = True
                del of
                res.append((lp.tolist(), ds.tolist()))
            return res


# ------------------------------------------------------------------ VMAF

def vmaf(gt, out, n, fps, rows_gt, rows_out, convert, threads):
    """Per-frame VMAF (v1 fidelity, v0.6.1neg) and CAMBI added, as sptenc computes them."""
    def chain(rows):
        crop = f",crop=iw:{rows[1] - rows[0]}:0:{rows[0]}" if rows else ""
        return f"setpts=PTS-STARTPTS,trim=end_frame={n}{crop},{VMAF_CONVERT[convert]},setparams=colorspace=unknown"
    with tempfile.TemporaryDirectory() as d:
        report = os.path.join(d, "vmaf.json")
        lav = (f"libvmaf=model={VMAF_MODELS}:feature={CAMBI_FEATURE}:log_fmt=json:log_path={report}"
               f":n_threads={threads}")
        cmd = [FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-r", fps, "-i", out, "-r", fps, "-i", gt,
               "-filter_complex", f"[0:v]{chain(rows_out)}[distorted];[1:v]{chain(rows_gt)}[reference];"
                                  f"[distorted][reference]{lav}", "-f", "null", "-"]
        log(f"$ {shlex.join(cmd)}")
        t0 = time.perf_counter()
        r = subprocess.run(cmd)
        if r.returncode:
            raise SystemExit(f"libvmaf failed with exit status {r.returncode}")
        with open(report, encoding="utf-8") as f:
            rep = json.load(f)
    frames = rep.get("frames", [])
    get = lambda k: [fr["metrics"].get(k) for fr in frames]  # noqa: E731
    pooled = rep.get("pooled_metrics", {})
    return {"vmaf": get("vmaf"), "vmaf_neg": get("vmaf_neg"), "cambi_added": get("cambi_full_reference"),
            "cambi_out": get("cambi_hrs_1080_vlt_0.06"), "cambi_gt": get("cambi_source")}, \
        {k: pooled.get(k) for k in ("vmaf", "vmaf_neg", "cambi_full_reference")}, \
        {"libvmaf": rep.get("version"), "s": round(time.perf_counter() - t0, 1), "convert": convert}


# ------------------------------------------------------------------ scoring

def score(a):
    rows = parse_rows(a.rows)
    gt_src = Source(a.gt, rows)
    outs = []
    for variant, seed, path in a.out:
        s = Source(path, rows)
        if (s.W, s.h) != (gt_src.W, gt_src.h):
            raise SystemExit(f"{path}: {s.W}x{s.h} after --rows, the ground truth is {gt_src.W}x{gt_src.h}")
        outs.append({"variant": variant, "seed": seed, "path": path, "src": s,
                     "pf": defaultdict(list), "pt": defaultdict(list), "prev": None})
    deep = None if a.no_deep else Deep(a.threads)
    n = 0
    t_classic = 0.0
    t_start = time.perf_counter()
    done = False
    while not done:
        gts, batch = [], [[] for _ in outs]
        while len(gts) < a.batch:
            if a.frames and n + len(gts) >= a.frames:
                break
            g = gt_src.read()
            fr = [o["src"].read() for o in outs]
            if g is None or any(f is None for f in fr):
                break
            gts.append(g)
            for k, f in enumerate(fr):
                batch[k].append(f)
        if len(gts) < a.batch:
            done = True
        if not gts:
            break
        t0 = time.perf_counter()
        for i, g in enumerate(gts):
            yg = luma(g)
            lab_g = lab_lf(g)
            for o, frames in zip(outs, batch):
                f = frames[i]
                yo = luma(f)
                d = yo - yg
                pf = o["pf"]
                pf["psnr_y"].append(psnr(float(np.mean(d * d, dtype=np.float64))))
                pf["psnr_y_bottom16"].append(psnr(float(np.mean(d[-16:] * d[-16:], dtype=np.float64))))
                pf["psnr_y_rest"].append(psnr(float(np.mean(d[:-16] * d[:-16], dtype=np.float64))))
                pf["ssim_y"].append(ssim(yo, yg))
                pf["de00_lf"].append(float(deltaE_ciede2000(lab_lf(f), lab_g).mean()))
                dlf = block_mean(d)
                if o["prev"] is not None:
                    pd, pdlf = o["prev"]
                    o["pt"]["temporal_full"].append(float(np.abs(d - pd).mean()))
                    o["pt"]["temporal_lf"].append(float(np.abs(dlf - pdlf).mean()))
                o["prev"] = (d, dlf)
        t_classic += time.perf_counter() - t0
        if deep:
            for o, (lp, ds) in zip(outs, deep.run(gts, batch)):
                o["pf"]["lpips"] += lp
                o["pf"]["dists"] += ds
        n += len(gts)
        log(f"{n} frames")
    gt_src.close()
    for o in outs:
        o["src"].close()
    if n == 0:
        raise SystemExit("no frame compared")
    fps = gt_src.fps or "24000/1001"
    os.makedirs(a.json_dir, exist_ok=True)
    for o in outs:
        res = {"clip": a.clip, "variant": o["variant"], "seed": o["seed"], "gt": os.path.abspath(a.gt),
               "out": os.path.abspath(o["path"]), "frames": n, "size": [gt_src.W, gt_src.h], "rows": a.rows,
               "per_frame": {k: [round(v, 6) for v in vals] for k, vals in o["pf"].items()},
               "per_transition": {k: [round(v, 6) for v in vals] for k, vals in o["pt"].items()},
               "timing": {"classic_frames_per_s": round(n * len(outs) / max(t_classic, 1e-9), 2)}}
        if deep:
            m = n * len(outs)
            res["timing"].update({"lpips_frames_per_s": round(m / max(deep.t_lpips, 1e-9), 2),
                                  "dists_frames_per_s": round(m / max(deep.t_dists, 1e-9), 2),
                                  "outputs_in_lockstep": len(outs)})
            res["versions"] = deep.versions
        if not a.no_vmaf:
            series, pooled, info = vmaf(os.path.abspath(a.gt), os.path.abspath(o["path"]), n, fps,
                                        gt_src.rows, o["src"].rows, a.vmaf_convert, a.vmaf_threads)
            for k, v in series.items():
                res["per_frame"][k] = [None if x is None else round(x, 6) for x in v]
            res["vmaf_pooled"] = pooled
            res["vmaf_info"] = info
        res["means"] = means(res)
        path = os.path.join(a.json_dir, f"{a.clip}.{o['variant']}.s{o['seed']}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(res, f)
        print(f"{a.clip} {o['variant']} seed {o['seed']}: "
              + ", ".join(f"{title} {fmt.format(res['means'][k])}" for k, title, _, fmt in METRICS
                          if res["means"].get(k) is not None) + f" -> {path}")
    t = res["timing"]
    print(f"{n} frames x {len(outs)} outputs in {time.perf_counter() - t_start:.1f} s; frames/s: classic "
          f"{t['classic_frames_per_s']}" + (f", LPIPS {t['lpips_frames_per_s']}, DISTS {t['dists_frames_per_s']}"
                                           if deep else ""))


def deltaE_ciede2000(lab1, lab2):
    from skimage.color import deltaE_ciede2000 as de
    return de(lab1, lab2, channel_axis=-1)


def series_of(res, k):
    v = res.get("per_transition", {}).get(k) if k in PER_TRANSITION else res.get("per_frame", {}).get(k)
    if not v:
        return None
    return np.array([np.nan if x is None else x for x in v], dtype=float)


def means(res):
    out = {}
    for k, *_ in METRICS:
        s = series_of(res, k)
        if s is None:
            continue
        fin = s[~np.isnan(s)]
        out[k] = float(fin.mean()) if fin.size else None
    return out


# ------------------------------------------------------------------ summary

def load(paths):
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "*.json"))) if os.path.isdir(p) else [p]
    runs = []
    for p in files:
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        if "clip" in r and "per_frame" in r:
            r["_file"] = p
            r.setdefault("means", means(r))
            runs.append(r)
    return runs


def fmt(v, f):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if isinstance(v, float) and math.isinf(v):
        return "inf"
    return f.format(v)


def block_bootstrap(diffs, block, n_boot, rng):
    """95% CI of the pooled mean of several per-frame difference series (one per seed)."""
    stats = np.empty(n_boot)
    for b in range(n_boot):
        tot = cnt = 0.0
        for d in diffs:
            n = len(d)
            L = min(block, n)
            k = -(-n // L)
            starts = rng.integers(0, n - L + 1, size=k)
            s = np.concatenate([d[i:i + L] for i in starts])[:n]
            tot += s.sum()
            cnt += n
        stats[b] = tot / cnt
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def summary(a):
    runs = load(a.summary)
    if not runs:
        raise SystemExit("no result JSON")
    refs = {v.strip() for v in a.reference.split(",") if v.strip()}
    by = defaultdict(lambda: defaultdict(dict))  # clip -> variant -> seed -> run
    for r in runs:
        by[r["clip"]][r["variant"]][str(r["seed"])] = r
    keys = [m for m in METRICS if any(r["means"].get(m[0]) is not None for r in runs)]
    rng = np.random.default_rng(0)

    def order(variants):
        return sorted(variants, key=lambda v: (v in refs, v != a.default, v))

    print("## Means per clip and variant\n")
    print("Mean over the seeds of each run's mean (per-frame PSNR averaged); `n` = seeds.\n")
    print("| Clip | Variant | n | " + " | ".join(t for _, t, _, _ in keys) + " |")
    print("|---|---|---|" + "---|" * len(keys))
    for clip in sorted(by):
        for v in order(by[clip]):
            seeds = by[clip][v]
            cells = []
            for k, _, _, f in keys:
                vals = [s["means"].get(k) for s in seeds.values() if s["means"].get(k) is not None]
                cells.append(fmt(float(np.mean(vals)) if vals else None, f))
            print(f"| {clip} | {v} | {len(seeds)} | " + " | ".join(cells) + " |")

    band = {}
    print(f"\n## Seed band of `{a.default}`\n")
    print("Spread (max − min) of the per-seed means, and the range.\n")
    print("| Clip | Seeds | " + " | ".join(t for _, t, _, _ in keys) + " |")
    print("|---|---|" + "---|" * len(keys))
    for clip in sorted(by):
        seeds = by[clip].get(a.default, {})
        cells = []
        for k, _, _, f in keys:
            vals = [s["means"].get(k) for s in seeds.values() if s["means"].get(k) is not None]
            vals = [v for v in vals if math.isfinite(v)]
            if len(vals) >= 1:
                band[(clip, k)] = max(vals) - min(vals)
            cells.append(f"{fmt(max(vals) - min(vals), f)} ({fmt(min(vals), f)}–{fmt(max(vals), f)})"
                         if len(vals) > 1 else "–")
        print(f"| {clip} | {', '.join(sorted(seeds)) or '–'} | " + " | ".join(cells) + " |")

    variants = sorted({v for clip in by for v in by[clip] if v != a.default and v not in refs})
    for v in variants:
        print(f"\n## `{v}` − `{a.default}`, paired by seed\n")
        print(f"Mean difference per frame (per transition for the temporal errors) [95% CI, block bootstrap, "
              f"blocks of {a.block}]: better / worse when the CI excludes 0 and |Δ| > the seed band.\n")
        print("| Clip | Seeds | " + " | ".join(t for _, t, _, _ in keys) + " |")
        print("|---|---|" + "---|" * len(keys))
        tally = defaultdict(lambda: defaultdict(int))
        for clip in sorted(by):
            if v not in by[clip] or a.default not in by[clip]:
                continue
            common = sorted(set(by[clip][v]) & set(by[clip][a.default]))
            cells = []
            for k, _, higher, f in keys:
                diffs = []
                for s in common:
                    x, y = series_of(by[clip][v][s], k), series_of(by[clip][a.default][s], k)
                    if x is None or y is None:
                        continue
                    m = min(len(x), len(y))
                    d = x[:m] - y[:m]
                    d = d[np.isfinite(d)]
                    if d.size:
                        diffs.append(d)
                if not diffs:
                    cells.append("–")
                    continue
                mean = float(np.concatenate(diffs).mean())
                lo, hi = block_bootstrap(diffs, a.block, a.boot, rng)
                b = band.get((clip, k), 0.0)
                if (lo > 0 or hi < 0) and abs(mean) > b:
                    verdict = "better" if (mean > 0) == higher else "worse"
                else:
                    verdict = "within"
                tally[k][verdict] += 1
                cells.append(f"{fmt(mean, '{:+' + f[2:])} [{fmt(lo, '{:+' + f[2:])}, {fmt(hi, '{:+' + f[2:])}] {verdict}")
            print(f"| {clip} | {', '.join(common) or '–'} | " + " | ".join(cells) + " |")
        print("| **clips better / worse / within** | | " + " | ".join(
            f"{tally[k]['better']} / {tally[k]['worse']} / {tally[k]['within']}" for k, *_ in keys) + " |")


# ------------------------------------------------------------------ validation inputs

def make_test(a):
    src = Source(a.make_test)
    os.makedirs(a.work, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.make_test))[0]
    paths = {"blur": os.path.join(a.work, f"{stem}.blur{a.blur:g}.mkv"),
             "noise": os.path.join(a.work, f"{stem}.noise{a.noise:g}.s{a.seed}.mkv")}
    procs = {}
    for k, p in paths.items():
        procs[k] = subprocess.Popen([FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
                                     "-f", "rawvideo", "-pix_fmt", "gbrp16le", "-s", f"{src.W}x{src.H}",
                                     "-framerate", src.fps or "24000/1001", "-i", "-", "-fps_mode", "passthrough",
                                     "-vf", "setparams=colorspace=gbr:range=pc:color_primaries=bt709:color_trc=bt709",
                                     "-c:v", "ffv1", "-level", "3", "-g", "1", "-slices", "16", "-slicecrc", "1",
                                     "-pix_fmt", "gbrp16le", "-colorspace", "rgb", "-color_primaries", "bt709",
                                     "-color_trc", "bt709", "-color_range", "pc", p], stdin=subprocess.PIPE)
    rng = np.random.default_rng(a.seed)
    n = 0
    while True:
        x = src.read()
        if x is None:
            break
        for k, p in paths.items():
            if k == "blur":
                y = cv2.GaussianBlur(x, (0, 0), a.blur)
            else:
                y = x + rng.normal(0, a.noise / 255.0, size=x.shape).astype(np.float32)
            q = np.rint(np.clip(y, 0, 1) * 65535).astype("<u2")
            procs[k].stdin.write(np.ascontiguousarray(q.transpose(2, 0, 1)[[1, 2, 0]]).tobytes())
        n += 1
    src.close()
    for k, pr in procs.items():
        pr.stdin.close()
        if pr.wait():
            raise SystemExit(f"ffmpeg failed writing {paths[k]}")
    for k, p in paths.items():
        print(f"{p}: {n} frames, ground truth {'blurred, sigma ' + format(a.blur, 'g') + ' px' if k == 'blur' else 'plus Gaussian noise, sigma ' + format(a.noise, 'g') + ' levels'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("gt", nargs="?", help="ground-truth RGB master")
    ap.add_argument("--clip", help="clip name (content type)")
    ap.add_argument("--out", nargs=3, action="append", metavar=("VARIANT", "SEED", "PATH"),
                    help="an output to score (repeat): its variant, seed and RGB master")
    ap.add_argument("--json-dir", help="where the per-output JSONs go")
    ap.add_argument("--rows", help="Y0:Y1, rows compared for inputs taller than Y1-Y0")
    ap.add_argument("--frames", type=int, default=0, help="compare the first N frames only")
    ap.add_argument("--batch", type=int, default=4, help="frames per deep-metric batch")
    ap.add_argument("--threads", type=int, default=24, help="torch CPU threads (physical cores: SMT threads slow it down)")
    ap.add_argument("--no-deep", action="store_true", help="skip LPIPS and DISTS")
    ap.add_argument("--no-vmaf", action="store_true")
    ap.add_argument("--vmaf-convert", choices=sorted(VMAF_CONVERT), default="sptenc",
                    help="RGB -> YUV for libvmaf: sptenc's swscale conversion (default) or zscale")
    ap.add_argument("--vmaf-threads", type=int, default=16)
    ap.add_argument("--summary", nargs="+", metavar="JSON_OR_DIR", help="Markdown tables of saved results")
    ap.add_argument("--default", default="default", help="--summary: the variant the others are paired with")
    ap.add_argument("--reference", default="bicubic", help="--summary: comma list of reference-only variants")
    ap.add_argument("--block", type=int, default=8, help="--summary: bootstrap block length (frames)")
    ap.add_argument("--boot", type=int, default=2000, help="--summary: bootstrap resamples")
    ap.add_argument("--make-test", metavar="GT", help="write blurred and noisy copies of GT into --work")
    ap.add_argument("--work", help="--make-test output directory")
    ap.add_argument("--blur", type=float, default=1.0, help="--make-test: Gaussian blur sigma (px)")
    ap.add_argument("--noise", type=float, default=2.0, help="--make-test: noise sigma (8-bit levels)")
    ap.add_argument("--seed", type=int, default=1, help="--make-test: noise seed")
    a = ap.parse_args()
    if a.summary:
        return summary(a)
    if a.make_test:
        if not a.work:
            ap.error("--make-test needs --work")
        return make_test(a)
    if not (a.gt and a.out and a.clip and a.json_dir):
        ap.error("give GT, --clip, at least one --out and --json-dir (or --summary / --make-test)")
    score(a)


if __name__ == "__main__":
    main()

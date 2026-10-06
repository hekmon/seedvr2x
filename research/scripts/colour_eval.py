#!/usr/bin/env python3
"""Colour-correction variants scored against a ground truth, for the colour study (docs/colour.md,
steps 2 and 3).

  colour_eval.py score --clip NAME --gt GT.mkv --ref TAG=REF.pt [--ref ...] --content TAG=DECODE.pt
                 [--content ...] --variants SPEC[@REF],... --out DIR [--untiled TAG] [--lpips]
                 [--dists-every N] [--threads T] [--frames N]
  colour_eval.py score --clip NAME --gt GT.mkv --master TAG=MASTER.mkv [--master ...] --out DIR [...]
  colour_eval.py render --clip NAME --gt GT.mkv --ref TAG=REF.pt --content TAG=DECODE.pt
                 --variants SPEC[@REF],... --out DIR [--pix-fmt yuv420p10le[,gbrp16le]]
  colour_eval.py vmaf --clip NAME --gt GT.mkv --master TAG=MASTER.mkv [--master ...] --out DIR
                 [--master-content s42[,s43]] [--frames N]
  colour_eval.py summary DIR [DIR ...] --baseline SPEC@REF [--block 8] [--boot 2000]

Why: every variant post-processes the same dumped decode and reference (colour_dump.py), so a
variant's score differs from the baseline's by the variant alone.

render: the variants' outputs written as masters through ffv1_out.py's own chain (imported: its
Writer and to_planar), by default seedvr2x's default master, yuv420p10le (DESIGN.md, Output).
score --master: masters scored as they are, each under its tag; a YUV master is read back to
full-range RGB with zscale (BT.709, limited range, chroma sited left), as a player shows it.
--vmaf adds VMAF and CAMBI (fr_metrics.py's libvmaf models and options): an RGB master through
fr_metrics.py's own chain, a YUV master as stored against the ground truth through ffv1_out.py's
yuv420p10le chain, at 10 bits where the master quantises.
vmaf: VMAF and CAMBI alone, the masters scored as score --master --vmaf scores them (an RGB master
through fr_metrics.py's chain, as measurement scores its baselines; a YUV master as stored), one JSON
per master and content tag (a baseline that has no seed, e.g. the bicubic upscale, under each seed's
tag so that it pairs with each): what the other metrics would repeat from the dumps is not computed.
stitch: clip B's boundary steps after each variant (milestone 2's measure): a windowed run and its
one-batch run (colour_dump.py decodes), both corrected by the variant, written as rounded 8-bit
PNGs and compared by stitch_metrics.py (measurement's) with its --latent analysis.

score: per content (a decode dump, colour_dump.py's decode.pt or decode-<tile>-<overlap>.pt: the
VAE's raw output) and variant (colour_variants.py's specs, @TAG picks the reference, the first
--ref by default), the corrected frames against the ground truth (GT), frame by frame:
- de<s>: ΔE00 after a Gaussian blur of s = 0, 1, 2, 4, 8, 16 px, whole frame (colour_diag.py's,
  fr_metrics.py's ΔE00 lf at 4); tl<s>, tc<s>: its lightness and colour parts over the picture
  (the letterbox bars and bottom 16 rows left out)
- psnr_y: PSNR of the BT.709 luma, 8-bit scale; lap: luma Laplacian variance (fr_clips.py's)
- t_full, t_lf (per transition): |Δout - ΔGT| of luma between consecutive frames, at full
  resolution and on 16x16 block means (fr_metrics.py's temporal errors)
- fringe: the colour part of the unblurred ΔE00 on the GT's strong edges: the top 5% of its luma
  gradient, widened by 2 px
- lpips (--lpips): LPIPS AlexNet, full resolution; dists (--dists-every N): DISTS on frames 0, N,
  2N... (fr_metrics.py's Deep, its ground-truth features recomputed per frame)
- with --untiled TAG, for every other content: psnr_tile and de_tile, the output against TAG's
  output of the same variant (what tiling changed after correction), and dy_blocks: the mean luma
  difference (8-bit levels) of 60x60 blocks, tiled minus untiled, per frame
Writes DIR/<clip>.<content>.<variant>.json, and DIR/<clip>.gt.json (the GT's block flatness: the
mean luma gradient of each 60x60 block, per frame).

summary: tables over the JSONs of DIR: means per clip, content and variant (seeds: contents s42,
s43, s1234, untiled); paired differences to --baseline per clip, frame by frame pooled over the
seeds, a 95% interval from a moving-block bootstrap, and a verdict: better or worse when the interval
excludes 0 and the difference exceeds the baseline's seed spread (max - min of its per-seed means),
within otherwise, as numerics.md's protocol; per tiling, the block offsets left after correction
(largest |block mean| per frame, and over the GT's flattest third of blocks); with the table groups,
the latent grid (numerics.md, every fourth frame): after a shot's first frame the causal VAE packs 4
frames per latent, and the last of each group (place 3) comes out closest to the GT; per variant,
place 3 minus place 1 (the group's second frame), whether a correction narrows that gap.
"""
import argparse
import glob
import json
import math
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# fr_metrics.py (measurement's, imported for LPIPS and DISTS): next to this script, or MEAS_SCRIPTS
sys.path.append(os.environ.get("MEAS_SCRIPTS", HERE))
import colour_diag as D  # noqa: E402

SIGMAS = D.SIGMAS
BLOCK = 60
EDGE_SHARE = 0.05
PER_TRANSITION = ("t_full", "t_lf")
HIGHER = {"psnr_y", "lap", "psnr_tile", "vmaf"}  # higher is better (lap: reported as a change)


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def safe(spec):
    return spec.replace(":", "_").replace("@", "~")


def luma8(rgb):
    return D.luma8(rgb)


def block_means(x, block=BLOCK):
    h, w = x.shape[:2]
    hb, wb = h // block, w // block
    return x[: hb * block, : wb * block].reshape(hb, block, wb, block, *x.shape[2:]).mean(axis=(1, 3))


def edge_mask(y_gt):
    gx = cv2.Sobel(y_gt, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y_gt, cv2.CV_32F, 0, 1, ksize=3)
    g = np.hypot(gx, gy)
    m = (g > np.percentile(g, 100 * (1 - EDGE_SHARE))).astype(np.uint8)
    return cv2.dilate(m, np.ones((5, 5), np.uint8)).astype(bool), g


def load_content(path):
    import torch
    fv = torch.load(path)["final_video"]
    return fv.permute(0, 3, 1, 2).to(torch.float32).contiguous()  # (T, 3, H, W) in [-1, 1]


def load_reference(path, t, h, w):
    """colour_dump.py's enc_bf16.pt / ref_f32.pt: one batch [C, T, Hp, Wp] -> (t, 3, h, w) float32,
    cut as numz's Phase 4 cuts it (generation_phases.py: its first t frames, rows :h, columns :w)."""
    import torch
    batches = torch.load(path)
    if len(batches) != 1:
        raise SystemExit(f"{path}: {len(batches)} batches, one expected")
    r = batches[0].permute(1, 0, 2, 3)
    return r[:t, :, :h, :w].to(torch.float32).contiguous()


class MasterFrames:
    """A master's frames as RGB float32 [0, 1] (H, W, 3): RGB as stored; YUV (seedvr2x's default
    yuv420p10le: BT.709, limited range, chroma sited left) back to full-range RGB with zscale, its
    chroma upsampled with zscale's default filter, as a player would show it. zscale on one slice
    (threads=1, as ffv1_out.py's chain): in slices, each would filter the chroma up to its own edge,
    and the frames read back would depend on the machine's CPU count."""

    def __init__(self, path, threads=4):
        import subprocess
        r = subprocess.run([D.FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=pix_fmt,width,height", "-of", "json", path], capture_output=True, text=True)
        s = json.loads(r.stdout)["streams"][0]
        self.W, self.H, fmt = s["width"], s["height"], s["pix_fmt"]
        self.fmt = fmt
        vf = ["-vf", "zscale=threads=1:min=709:rin=limited:chromal=left:m=709:r=full:d=none,format=gbrp16le"] \
            if fmt.startswith("yuv") else []
        self.size = 3 * self.W * self.H * 2
        self.proc = subprocess.Popen([D.FFMPEG, "-v", "error", "-nostdin", "-threads", str(threads), "-i", path,
                                      "-map", "0:v:0", *vf, "-fps_mode", "passthrough", "-f", "rawvideo",
                                      "-pix_fmt", "gbrp16le", "-"], stdout=subprocess.PIPE)

    def read(self):
        buf = self.proc.stdout.read(self.size)
        if len(buf) < self.size:
            return None
        g, b, r = np.frombuffer(buf, "<u2").reshape(3, self.H, self.W)
        return np.stack([r, g, b], axis=-1).astype(np.float32) * np.float32(1 / 65535)

    def close(self):
        self.proc.kill()
        self.proc.stdout.close()
        self.proc.wait()


def vmaf_master(gt, master, n, fps, threads, height):
    """VMAF (v1 fidelity) and CAMBI as fr_metrics.py runs libvmaf (its CAMBI options, and its model
    for a ground truth of this height, as sptenc picks it: the 2160p model from 2160 rows up), but on
    a YUV master as stored: the master is the distorted input as it is, and the ground truth goes
    through ffv1_out.py's yuv420p10le chain (zscale, BT.709, limited range, chroma sited left), where
    seedvr2x's default master quantises."""
    import subprocess
    import tempfile
    import fr_metrics
    import ffv1_out
    chain = ffv1_out.FORMATS["yuv420p10le"][1][1]  # "-vf" value: zscale ...,format=yuv420p10le
    with tempfile.TemporaryDirectory() as d:
        report = os.path.join(d, "vmaf.json")
        lav = (f"libvmaf=model={fr_metrics.vmaf_models(height)}:feature={fr_metrics.CAMBI_FEATURE}:log_fmt=json"
               f":log_path={report}:n_threads={threads}")
        fc = (f"[0:v]setpts=PTS-STARTPTS,trim=end_frame={n},setparams=colorspace=unknown[distorted];"
              f"[1:v]setpts=PTS-STARTPTS,trim=end_frame={n},{chain},setparams=colorspace=unknown[reference];"
              f"[distorted][reference]{lav}")
        cmd = [D.FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-r", str(fps), "-i", master,
               "-r", str(fps), "-i", gt, "-filter_complex", fc, "-f", "null", "-"]
        if subprocess.run(cmd).returncode:
            raise SystemExit(f"libvmaf failed on {master}")
        with open(report, encoding="utf-8") as f:
            rep = json.load(f)
    get = lambda k: [fr["metrics"].get(k) for fr in rep.get("frames", [])]  # noqa: E731
    return {"vmaf": get("vmaf"), "cambi_added": get("cambi_full_reference"),
            "cambi_out": get("cambi_hrs_1080_vlt_0.06")}


def stitch(a):
    """Clip B's boundary steps after each variant: the windowed run's output against the one-batch
    run's, both corrected by the same variant, through stitch_metrics.py (measurement's, unchanged:
    8-bit PNGs, here rounded, and its --latent analysis of blend_patch.py's windows)."""
    import subprocess
    import torch
    import colour_variants as V
    torch.set_num_threads(a.threads)
    refs = dict(r.split("=", 1) for r in a.ref)
    ref_order = [r.split("=", 1)[0] for r in a.ref]
    one, win = load_content(a.onebatch), load_content(a.windows)
    t, _, h, w = one.shape
    meas = os.environ.get("MEAS_SCRIPTS", HERE)
    for spec in (v.strip() for v in a.variants.split(",") if v.strip()):
        vspec, _, rtag = spec.partition("@")
        rtag = rtag or ref_order[0]
        ref = load_reference(refs[rtag], t, h, w)
        dirs = {}
        for tag, c in (("onebatch", one), ("windows", win)):
            with torch.inference_mode():
                o = V.apply(vspec, c, ref).permute(0, 2, 3, 1).numpy()
            d = os.path.join(a.out, f"{safe(vspec)}~{rtag}", tag)
            os.makedirs(d, exist_ok=True)
            for i in range(t):
                cv2.imwrite(os.path.join(d, f"{i:05d}.png"),
                            np.ascontiguousarray(np.rint(o[i] * 255).clip(0, 255).astype(np.uint8)[..., ::-1]))
            dirs[tag] = d
        out_json = os.path.join(a.out, f"{safe(vspec)}~{rtag}.json")
        cmd = [sys.executable, os.path.join(meas, "stitch_metrics.py"), dirs["windows"], "--input", a.input,
               "--skip", str(a.skip), "--ref", dirs["onebatch"], "--batch", str(t), "--latent", a.latent,
               "--curve", a.curve, "--label", spec, "--json", out_json]
        log("$ " + " ".join(cmd))
        if subprocess.run(cmd).returncode:
            raise SystemExit(f"stitch_metrics.py failed for {spec}")


def render(a):
    """Variants' outputs written as masters through ffv1_out.py's own chain (its Writer and
    to_planar: round(x * 65535), then the pixel format's ffmpeg conversion)."""
    import torch
    import colour_variants as V
    import ffv1_out
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    refs = dict(r.split("=", 1) for r in a.ref)
    ref_order = [r.split("=", 1)[0] for r in a.ref]
    fps = ffv1_out.probe_fps(a.gt) if hasattr(ffv1_out, "probe_fps") else "24000/1001"
    for ctag, cpath in (c.split("=", 1) for c in a.content):
        c = load_content(cpath)
        t, _, h, w = c.shape
        for spec in (v.strip() for v in a.variants.split(",") if v.strip()):
            vspec, _, rtag = spec.partition("@")
            rtag = rtag or ref_order[0]
            with torch.inference_mode():
                out = V.apply(vspec, c, load_reference(refs[rtag], t, h, w))
            o = out.permute(0, 2, 3, 1).contiguous().numpy()
            for fmt in a.pix_fmt.split(","):
                path = os.path.join(a.out, f"{a.clip}.{ctag}.{safe(vspec)}~{rtag}.{fmt}.mkv")
                wr = ffv1_out.Writer(path, fmt, w, h, fps, 16)
                for i in range(t):
                    wr.write(ffv1_out.to_planar(o[i].copy(), wr.bits))
                if wr.close():
                    raise SystemExit(f"{path}: ffmpeg failed")
                log(f"{path}: {t} frames")
            del out, o


def score(a):
    import torch
    import colour_variants as V
    torch.set_num_threads(a.threads)
    cv2.setNumThreads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    refs = dict(r.split("=", 1) for r in (a.ref or []))
    ref_order = [r.split("=", 1)[0] for r in (a.ref or [])]
    contents = [c.split("=", 1) for c in (a.content or [])]
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    if a.master:  # already-corrected outputs: each master is scored as it is, one "variant" per file
        # a comma list of tags: scored under the first, the same JSON written under the others
        contents = [(a.master_content.split(",")[0], None)]
        variants = [m.split("=", 1)[0] for m in a.master]
        masters = dict(m.split("=", 1) for m in a.master)
    if a.untiled and a.untiled != contents[0][0]:
        raise SystemExit("--untiled must name the first --content")
    deep = None
    if a.lpips or a.dists_every:
        import fr_metrics
        deep = fr_metrics.Deep(a.threads)
    # the ground truth, read once
    src = D.Master(a.gt, 4)
    gts = []
    while not a.frames or len(gts) < a.frames:
        x = src.read()
        if x is None:
            break
        gts.append(x)
    src.close()
    t = len(gts)
    h, w = gts[0].shape[:2]
    if a.bars:  # given: a ground truth whose bars aren't dark enough for find_bars
        top, bottom = (int(v) for v in a.bars.split(":"))
    else:
        top, bottom, _ = D.find_bars(a.gt, t, 4)
    rows = np.arange(h)
    pic_rows = (rows >= top) & (rows < h - max(bottom, D.BOTTOM_ROWS))
    pic = {s: np.broadcast_to(pic_rows[::st, None], (len(range(0, h, st)), len(range(0, w, st))))
           for s, st in D.STRIDE.items()}
    gb = []  # per frame: {sigma: strided blurred CIELAB}
    gy, gedges, gflat = [], [], []
    for g in gts:
        lab = D.to_lab(g)
        gb.append({s: np.ascontiguousarray(D.blur(lab, s)[::D.STRIDE[s], ::D.STRIDE[s]]) for s in SIGMAS})
        y = luma8(g)
        gy.append(y)
        m, grad = edge_mask(y)
        gedges.append((m, lab))
        gflat.append(block_means(grad))
    with open(os.path.join(a.out, f"{a.clip}.gt.json"), "w", encoding="utf-8") as f:
        json.dump({"clip": a.clip, "gt": os.path.abspath(a.gt), "frames": t, "size": [w, h], "bars": [top, bottom],
                   "block": BLOCK, "block_gradient": [b.round(3).tolist() for b in gflat]}, f)
    log(f"{a.clip}: GT {t} frames {w}x{h}, bars {top}/{bottom}")
    loaded = {}

    def content(tag):
        if tag not in loaded:
            loaded.clear()
            loaded[tag] = load_content(dict(contents)[tag])[:t]
        return loaded[tag]

    reference_cache = {}

    def reference(tag):
        if tag not in reference_cache:
            reference_cache[tag] = load_reference(refs[tag], t, h, w)
        return reference_cache[tag]

    for spec in variants:
        vspec, _, rtag = spec.partition("@")
        rtag = rtag or (ref_order[0] if ref_order else "")
        untiled_out = None
        for ctag, cpath in contents:
            t0 = time.perf_counter()
            if a.master:
                mf = MasterFrames(masters[spec], 4)
                frames = [mf.read() for _ in range(t)]
                mf.close()
                master_fmt = mf.fmt
                if any(x is None or x.shape[:2] != (h, w) for x in frames):
                    raise SystemExit(f"{masters[spec]}: fewer than {t} frames of {w}x{h}")
                o = np.stack(frames)
                cpath = masters[spec]
            else:
                c = content(ctag)
                if tuple(c.shape[-2:]) != (h, w):
                    raise SystemExit(f"{cpath}: {tuple(c.shape)}, the GT is {h}x{w}")
                with torch.inference_mode():
                    out = V.apply(vspec, c, reference(rtag))
                o = out.permute(0, 2, 3, 1).contiguous().numpy()  # (T, H, W, 3) float32 in [0, 1]
                del out
            t_apply = time.perf_counter() - t0
            pf, pt = defaultdict(list), defaultdict(list)
            prev = None
            for i in range(t):
                x = o[i]
                lab = D.to_lab(x)
                for s in SIGMAS:
                    st = D.STRIDE[s]
                    xs = np.ascontiguousarray((lab if s == 0 else D.blur(lab, s))[::st, ::st])
                    de, tl, tc, th, rt = D.ciede2000(gb[i][s], xs)
                    cp = D.colour_part(tc, th, rt)
                    p = pic[s]
                    pf[f"de{s}"].append(float(de.mean()))
                    pf[f"tl{s}"].append(float(np.abs(tl[p]).mean()))
                    pf[f"tc{s}"].append(float(cp[p].mean()))
                m, glab = gedges[i]
                _, _, tc, th, rt = D.ciede2000(glab[m], lab[m])
                pf["fringe"].append(float(D.colour_part(tc, th, rt).mean()))
                y = luma8(x)
                d = y - gy[i]
                pf["psnr_y"].append(10 * math.log10(255.0 ** 2 / max(float(np.mean(d * d, dtype=np.float64)), 1e-12)))
                pf["lap"].append(D.lap_var(y))
                dlf = cv2.resize(d, (w // 16, h // 16), interpolation=cv2.INTER_AREA)
                if prev is not None:
                    pt["t_full"].append(float(np.abs(d - prev[0]).mean()))
                    pt["t_lf"].append(float(np.abs(dlf - prev[1]).mean()))
                prev = (d, dlf)
                if untiled_out is not None:
                    u = untiled_out[i]
                    e = (x - u).astype(np.float64)
                    mse = float(np.mean(e * e))
                    pf["psnr_tile"].append(10 * math.log10(1.0 / max(mse, 1e-20)))
                    ul = D.to_lab(u)
                    pf["de_tile"].append(float(D.ciede2000(ul[::2, ::2], lab[::2, ::2])[0].mean()))
                    pf["dy_blocks"].append(block_means(y - luma8(u)).round(4).tolist())
            if deep is not None:
                tt = deep.torch
                with tt.inference_mode():
                    for i in range(t):
                        g = deep.tensor([gts[i]])
                        xo = deep.tensor([o[i]])
                        if a.lpips:
                            pf["lpips"].append(float(deep.lpips(xo * 2 - 1, g * 2 - 1).flatten()[0]))
                        if a.dists_every and i % a.dists_every == 0:
                            ds = deep.dists_score(deep.dists.forward_once(xo), deep.dists.forward_once(g))
                            pf["dists"].append(float(ds[0]))
            if a.master and a.vmaf:
                import ffv1_out
                fps = ffv1_out.probe_fps(a.gt)
                if master_fmt.startswith("yuv"):
                    pf.update(vmaf_master(a.gt, masters[spec], t, fps, a.threads, h))
                else:
                    import fr_metrics
                    pf.update(fr_metrics.vmaf(a.gt, masters[spec], t, str(fps), None, None, "sptenc", a.threads)[0])
            res = {"clip": a.clip, "content": ctag, "content_path": os.path.abspath(cpath), "variant": vspec,
                   "ref": rtag, "ref_path": os.path.abspath(refs[rtag]) if rtag in refs else None,
                   "frames": t, "size": [w, h],
                   "dists_every": a.dists_every, "per_frame": pf, "per_transition": pt,
                   "seconds": {"apply": round(t_apply, 1), "total": round(time.perf_counter() - t0, 1)}}
            for tag in (a.master_content.split(",") if a.master else [ctag]):
                res["content"] = tag
                name = f"{a.clip}.{tag}.{safe(vspec)}~{rtag}.json"
                with open(os.path.join(a.out, name), "w", encoding="utf-8") as f:
                    json.dump(res, f)
            log(f"{a.clip} {ctag} {spec}: ΔE00 σ4 {np.mean(pf['de4']):.3f}, PSNR-Y {np.mean(pf['psnr_y']):.2f}, "
                f"lap {np.mean(pf['lap']):.1f}" + (f", LPIPS {np.mean(pf['lpips']):.4f}" if pf.get("lpips") else "")
                + (f", vs untiled {np.mean(pf['psnr_tile']):.2f} dB" if pf.get("psnr_tile") else "")
                + f" ({res['seconds']['apply']} + {res['seconds']['total'] - res['seconds']['apply']:.1f} s)")
            if a.untiled and ctag == a.untiled:
                untiled_out = o
            del o


def vmaf(a):
    """VMAF (v1, the model for the ground truth's height) and CAMBI of masters against the GT."""
    import subprocess
    import ffv1_out
    import fr_metrics
    os.makedirs(a.out, exist_ok=True)
    fps = ffv1_out.probe_fps(a.gt)
    height = int(fr_metrics.probe(a.gt)["height"])
    for tag, path in (m.split("=", 1) for m in a.master):
        r = subprocess.run([D.FFPROBE, "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                            "stream=pix_fmt,nb_read_frames", "-of", "json", path], capture_output=True, text=True)
        st = json.loads(r.stdout)["streams"][0]
        n = min(int(st["nb_read_frames"]), a.frames or 1 << 30)
        t0 = time.perf_counter()
        if st["pix_fmt"].startswith("yuv"):
            pf, how = vmaf_master(a.gt, path, n, fps, a.threads, height), "yuv as stored, GT through ffv1_out.py's chain"
        else:
            pf, _, info = fr_metrics.vmaf(a.gt, path, n, str(fps), None, None, "sptenc", a.threads)
            how = f"fr_metrics.py's chain ({info['convert']}, {info['models'][0]})"
        for ctag in a.master_content.split(","):
            res = {"clip": a.clip, "content": ctag, "content_path": os.path.abspath(path), "variant": tag, "ref": "",
                   "ref_path": None, "frames": n, "size": None, "vmaf_chain": how, "per_frame": pf,
                   "per_transition": {}, "seconds": {"total": round(time.perf_counter() - t0, 1)}}
            with open(os.path.join(a.out, f"{a.clip}.{ctag}.{safe(tag)}~.json"), "w", encoding="utf-8") as f:
                json.dump(res, f)
        log(f"{a.clip} {tag}: VMAF {np.mean(pf['vmaf']):.2f}, CAMBI added {np.mean(pf['cambi_added']):.4f} "
            f"({how}; {time.perf_counter() - t0:.0f} s)")


# ---------------------------------------------------------------- summary

def load(dirs):
    runs, gts = [], {}
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "*.json"))):
            with open(p, encoding="utf-8") as f:
                r = json.load(f)
            if "block_gradient" in r:
                gts[r["clip"]] = r
            elif "per_frame" in r:
                runs.append(r)
    return runs, gts


def series(r, k):
    v = r["per_transition"].get(k) if k in PER_TRANSITION else r["per_frame"].get(k)
    return None if not v else np.asarray(v, dtype=float)


def boot(diffs, block, n, rng):
    stats = np.empty(n)
    for b in range(n):
        tot = cnt = 0.0
        for d in diffs:
            L = min(block, len(d))
            k = -(-len(d) // L)
            s = np.concatenate([d[i:i + L] for i in rng.integers(0, len(d) - L + 1, size=k)])[:len(d)]
            tot += s.sum()
            cnt += len(d)
        stats[b] = tot / cnt
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


METRICS = ["de0", "de1", "de2", "de4", "de8", "de16", "tl2", "tc2", "psnr_y", "lpips", "dists", "lap", "fringe",
           "t_full", "t_lf", "vmaf", "cambi_added"]


def summary(a):
    runs, gts = load(a.dirs)
    rng = np.random.default_rng(1)
    key = lambda r: f"{r['variant']}@{r['ref']}"  # noqa: E731
    by = defaultdict(lambda: defaultdict(dict))  # clip -> variant@ref -> content -> run
    for r in runs:
        by[r["clip"]][key(r)][r["content"]] = r
    seeds = ("s42", "s43", "s1234")
    tables = set(a.tables.split(","))
    across = defaultdict(lambda: defaultdict(list))  # variant -> metric -> [(clip mean diff, tag)]
    if "means" in tables:
        means_table(by, seeds)
    if a.baseline:
        pairs_tables(a, by, seeds, rng, tables, across)
    if "across" in tables and across:
        print(f"\n## Across clips: mean of the per-clip paired differences to {a.baseline} (B / W / within)\n")
        ks = [k for k in METRICS if any(across[v].get(k) for v in across)]
        print("| Variant | Clips | " + " | ".join(ks) + " |")
        print("|---|---|" + "---|" * len(ks))
        for v in sorted(across):
            cells = []
            n = max(len(across[v][k]) for k in ks if across[v].get(k))
            for k in ks:
                vals = across[v].get(k)
                if not vals:
                    cells.append("–")
                    continue
                m = np.mean([x for x, _ in vals])
                b = sum(t == "B" for _, t in vals)
                w = sum(t == "W" for _, t in vals)
                up = sum(t == "↑" for _, t in vals)
                dn = sum(t == "↓" for _, t in vals)
                fmt = "{:+.2f}" if k in ("psnr_y", "lap", "vmaf") else "{:+.4f}" if k in ("lpips", "dists") else "{:+.3f}"
                cnt = f"{up}↑ {dn}↓" if k == "lap" else f"{b}/{w}/{len(vals) - b - w}"
                cells.append(f"{fmt.format(m)} ({cnt})")
            print(f"| {v} | {n} | " + " | ".join(cells) + " |")
    if "tiles" in tables:
        tiles_table(by, gts)
    if "groups" in tables:
        groups_table(by, seeds)


GROUP_METRICS = (("psnr_y", "{:+.2f}"), ("de4", "{:+.3f}"), ("lpips", "{:+.4f}"))


def groups_table(by, seeds):
    """Per clip and variant, the mean over the seeds of (place 3 - place 1) within the latent groups:
    frames 1, 2, 3, 4 are places 0-3 of the first group (frame 0 is encoded alone), and so on."""
    print("\n## The latent grid: place 3 of each 4-frame group minus place 1 (means over the seeds)\n")
    print("| Clip | Variant | Seeds | " + " | ".join(k for k, _ in GROUP_METRICS) + " |")
    print("|---|---|---|" + "---|" * len(GROUP_METRICS))
    across = defaultdict(lambda: defaultdict(list))
    for clip in sorted(by):
        for v in sorted(by[clip]):
            rs = [by[clip][v][sd] for sd in seeds if sd in by[clip][v]]
            if not rs:
                continue
            cells = []
            for k, fmt in GROUP_METRICS:
                gaps = []
                for r in rs:
                    x = series(r, k)
                    if x is None or len(x) < 9:
                        continue
                    place = (np.arange(1, len(x)) - 1) % 4
                    gaps.append(x[1:][place == 3].mean() - x[1:][place == 1].mean())
                if gaps:
                    across[v][k].append(float(np.mean(gaps)))
                cells.append(fmt.format(np.mean(gaps)) if gaps else "–")
            print(f"| {clip} | {v} | {len(rs)} | " + " | ".join(cells) + " |")
    print("\nAcross clips (mean of the per-clip gaps, clips):\n")
    print("| Variant | " + " | ".join(k for k, _ in GROUP_METRICS) + " |")
    print("|---|" + "---|" * len(GROUP_METRICS))
    for v in sorted(across):
        print(f"| {v} | " + " | ".join(
            f"{fmt.format(np.mean(across[v][k]))} ({len(across[v][k])})" if across[v].get(k) else "–"
            for k, fmt in GROUP_METRICS) + " |")


def means_table(by, seeds):
    print("## Means over the seeds (untiled)\n")
    print("| Clip | Variant | Seeds | " + " | ".join(METRICS) + " |")
    print("|---|---|---|" + "---|" * len(METRICS))
    for clip in sorted(by):
        for v in sorted(by[clip]):
            rs = [by[clip][v][s] for s in seeds if s in by[clip][v]]
            if not rs:
                continue
            cells = []
            for k in METRICS:
                vals = [series(r, k) for r in rs]
                vals = [x.mean() for x in vals if x is not None]
                cells.append("–" if not vals else (f"{np.mean(vals):.2f}" if k in ("psnr_y", "lap") else f"{np.mean(vals):.4f}"
                                                    if k in ("lpips", "dists") else f"{np.mean(vals):.3f}"))
            print(f"| {clip} | {v} | {len(rs)} | " + " | ".join(cells) + " |")


def pairs_tables(a, by, seeds, rng, tables, across):
    """Paired differences to the baseline per clip (table "pairs"), the verdict counts ("verdicts");
    every clip's mean difference and verdict recorded into across[variant][metric]."""
    if a.baseline:
        if "pairs" in tables:
            print(f"\n## Paired differences to {a.baseline} (variant − baseline; B/W: beyond the interval and the seed spread)\n")
            print("| Clip | Variant | " + " | ".join(METRICS) + " |")
            print("|---|---|" + "---|" * len(METRICS))
        tally = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
        for clip in sorted(by):
            base = by[clip].get(a.baseline)
            if not base:
                continue
            for v in sorted(by[clip]):
                if v == a.baseline:
                    continue
                cells = []
                for k in METRICS:
                    common = [s for s in seeds if s in by[clip][v] and s in base]
                    diffs, spread_vals = [], []
                    for s in common:
                        x, y = series(by[clip][v][s], k), series(base[s], k)
                        if x is None or y is None or len(x) != len(y):
                            continue
                        diffs.append(x - y)
                    for s in seeds:
                        if s in base and series(base[s], k) is not None:
                            spread_vals.append(series(base[s], k).mean())
                    if not diffs:
                        cells.append("–")
                        continue
                    mean = float(np.mean(np.concatenate(diffs)))
                    lo, hi = boot(diffs, a.block, a.boot, rng)
                    spread = (max(spread_vals) - min(spread_vals)) if len(spread_vals) > 1 else 0.0
                    sig = (lo > 0 or hi < 0) and abs(mean) > spread
                    better = (mean > 0) == (k in HIGHER)
                    if k == "lap":  # detail: neither better nor worse by sign alone; report the change
                        tag = "↓" if sig and mean < 0 else ("↑" if sig else "")
                    else:
                        tag = ("B" if better else "W") if sig else ""
                        tally[v][k][0 if tag == "B" else 1 if tag == "W" else 2] += 1
                    fmt = "{:+.2f}" if k in ("psnr_y", "lap") else "{:+.4f}" if k in ("lpips", "dists") else "{:+.3f}"
                    cells.append(fmt.format(mean) + (f" {tag}" if tag else ""))
                    across[v][k].append((mean, tag))
                if "pairs" in tables:
                    print(f"| {clip} | {v} | " + " | ".join(cells) + " |")
        if "verdicts" in tables:
            print("\n### Verdicts over the clips (better / worse / within)\n")
            ks = [k for k in METRICS if k != "lap"]
            print("| Variant | " + " | ".join(ks) + " |")
            print("|---|" + "---|" * len(ks))
            for v in sorted(tally):
                print(f"| {v} | " + " | ".join("{}/{}/{}".format(*tally[v][k]) for k in ks) + " |")


def tiles_table(by, gts):
    print("\n## Tiles: what is left after correction (tiled output against the untiled one, same variant)\n")
    print("Per frame: the largest |mean luma difference| of a 60x60 block (8-bit levels), all blocks and the GT's "
          "flattest third; PSNR and ΔE00 (every second pixel) against the untiled output. Means over frames.\n")
    print("| Clip | Content | Variant | max \\|block ΔY\\| | flat third | block ΔY std | PSNR vs untiled | ΔE00 vs untiled | ΔE00 σ4 vs GT |")
    print("|---|---|---|---|---|---|---|---|---|")
    for clip in sorted(by):
        g = gts.get(clip)
        for v in sorted(by[clip]):
            for ctag, r in sorted(by[clip][v].items()):
                if not r["per_frame"].get("dy_blocks"):
                    continue
                db = np.asarray(r["per_frame"]["dy_blocks"])  # (T, hb, wb)
                mx = np.abs(db).max(axis=(1, 2)).mean()
                sd = db.std(axis=(1, 2)).mean()
                flat = "–"
                if g:
                    gb = np.asarray(g["block_gradient"])[: len(db)]
                    thr = np.percentile(gb.reshape(len(gb), -1), 33, axis=1)[:, None, None]
                    fm = gb <= thr
                    flat = f"{np.mean([np.abs(db[i][fm[i]]).max() for i in range(len(db))]):.2f}"
                print(f"| {clip} | {ctag} | {v} | {mx:.2f} | {flat} | {sd:.2f} | "
                      f"{np.mean(r['per_frame']['psnr_tile']):.2f} | {np.mean(r['per_frame']['de_tile']):.3f} | "
                      f"{np.mean(r['per_frame']['de4']):.3f} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--clip", required=True)
    r.add_argument("--gt", required=True, help="for the frame rate")
    r.add_argument("--ref", action="append", required=True, metavar="TAG=REF.pt")
    r.add_argument("--content", action="append", required=True, metavar="TAG=DECODE.pt")
    r.add_argument("--variants", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--pix-fmt", default="yuv420p10le", help="comma list of ffv1_out.py's pixel formats")
    r.add_argument("--threads", type=int, default=4)
    s = sub.add_parser("score")
    s.add_argument("--clip", required=True)
    s.add_argument("--gt", required=True)
    s.add_argument("--ref", action="append", metavar="TAG=REF.pt")
    s.add_argument("--content", action="append", metavar="TAG=DECODE.pt")
    s.add_argument("--master", action="append", metavar="TAG=MASTER.mkv",
                   help="score already-corrected masters instead (RGB or seedvr2x's YUV), one per tag")
    s.add_argument("--master-content", default="master",
                   help="with --master: the content tag the masters are scored under (e.g. s42, so seeds pair); "
                        "a comma list writes the same scores under each tag")
    s.add_argument("--variants", default="")
    s.add_argument("--out", required=True)
    s.add_argument("--untiled")
    s.add_argument("--vmaf", action="store_true", help="with --master: VMAF and CAMBI (libvmaf)")
    s.add_argument("--lpips", action="store_true")
    s.add_argument("--dists-every", type=int, default=0)
    s.add_argument("--threads", type=int, default=4)
    s.add_argument("--frames", type=int, default=0)
    s.add_argument("--bars", metavar="TOP:BOTTOM",
                   help="the letterbox rows at the top and bottom of the GT, instead of colour_diag's find_bars")
    v = sub.add_parser("vmaf")
    v.add_argument("--clip", required=True)
    v.add_argument("--gt", required=True)
    v.add_argument("--master", action="append", required=True, metavar="TAG=MASTER.mkv")
    v.add_argument("--master-content", default="master", help="content tag(s), comma list")
    v.add_argument("--out", required=True)
    v.add_argument("--frames", type=int, default=0)
    v.add_argument("--threads", type=int, default=8)
    b = sub.add_parser("stitch")
    b.add_argument("--ref", action="append", required=True, metavar="TAG=REF.pt")
    b.add_argument("--onebatch", required=True, help="the one-batch run's decode.pt")
    b.add_argument("--windows", required=True, help="the windowed run's decode.pt")
    b.add_argument("--input", required=True, help="the clip's source video, for stitch_metrics.py")
    b.add_argument("--skip", type=int, default=0)
    b.add_argument("--latent", required=True, help="W:M, as blend_patch.py's STITCH_LATENT")
    b.add_argument("--curve", default="linear")
    b.add_argument("--variants", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--threads", type=int, default=4)
    m = sub.add_parser("summary")
    m.add_argument("dirs", nargs="+")
    m.add_argument("--baseline")
    m.add_argument("--block", type=int, default=8)
    m.add_argument("--boot", type=int, default=2000)
    m.add_argument("--tables", default="means,pairs,verdicts,across,tiles",
                   help="comma list of the tables to print: means, pairs, verdicts, across, tiles, groups")
    a = ap.parse_args()
    {"render": render, "score": score, "vmaf": vmaf, "stitch": stitch, "summary": summary}[a.cmd](a)


if __name__ == "__main__":
    main()

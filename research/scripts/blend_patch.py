#!/usr/bin/env python3
"""Batch-boundary cross-fade weights for the SeedVR2 CLI, patched in at import (no checkout change).

SeedVR2 cross-fades the `--temporal_overlap` frames of two consecutive batches in Phase 3 with
`blend_overlapping_frames`, whose weights start at exactly 1 and end at exactly 0 and, from 3
frames up, only ramp over the middle third (bug 06): overlap 1, 2 and 4 never mix, 3 and 5 mix
one frame. This wrapper replaces the weights, and can render the same run with several weight
curves at once: the curve only changes how the decoded overlap frames are mixed, so every
variant shares the expensive phases (encode, DiT, decode).

Curves (weight of the previous batch on overlap frame i = 1..K, K = --temporal_overlap):
  numz     the CLI's own weights (unchanged behaviour)
  linear   1 - i/(K+1): a straight ramp over every overlap frame, end points excluded
  cosine   0.5 + 0.5 cos(pi i/(K+1)): raised cosine over every overlap frame, end points excluded
  prev     1: keep the previous batch, hard switch K frames into the new batch (the new
           batch's first K frames are context only, like --chunk_size, bug 07)
  cur      0: hard switch at the new batch's first frame (the previous batch's last K frames
           are context only)

Environment:
  STITCH_CURVE       curve of the CLI's own output (default numz)
  STITCH_EXTRA       comma list of more curves to render from the same run: Phase 4 (colour
                     correction) runs again on a copy of the Phase 3 frames with the overlap
                     frames re-mixed, and the result is saved as a PNG sequence (8-bit, truncated
                     like the CLI's PNGs) in <STITCH_EXTRA_DIR>/<curve>/
  STITCH_EXTRA_DIR   default: <--output>/extra
  STITCH_EXTRA_FFV1  comma list of FFV1 pixel formats (as ffv1_out.py: gbrp16le, yuv420p10le, ...)
                     to also write each extra curve as <STITCH_EXTRA_DIR>/<curve>.<pix_fmt>.mkv,
                     from the float frames (needs ffv1_out.py next to this script)

The extras reuse the decoded frames of the run's own batches: with STITCH_CURVE=cosine and
STITCH_EXTRA=numz,linear,prev,cur, one run gives five outputs, bit-identical outside the overlap
frames, and identical to what five separate runs would give (the CLI is deterministic). Phase 4
runs on a shallow copy of the context; its cleanup clears the resize transforms in place, so they
are deep-copied (its tensor releases are no-ops, bug 13). Not for multi-GPU runs (their GPU-join
blend is patched, the extras are not rendered).

Latent-space stitching (experimental, STITCH_LATENT=W:M): run the CLI with --batch_size >= the
clip length and --temporal_overlap 0, so that Phase 1 encodes the whole clip in one causal pass.
Phase 2 is then run on overlapping windows of W latent frames (M shared between consecutive
windows), the windows' DiT outputs are cross-faded over the M shared latents with STITCH_CURVE
(linear or cosine; prev/cur = hard switch), and Phase 3 decodes the merged latent sequence in one
causal pass, as a single batch. Latent j > 0 covers frames 4j-3..4j, latent 0 frame 0 alone, so
W = 6 is the DiT load of a 21-frame batch and M = 1 shares 4 frames. Every window gets the same
noise by position (the CLI seeds each batch alike), and windows after the first start with a
4-frame latent where the DiT always sees a 1-frame latent 0 in a normal batch. W:0 gives hard
DiT-window boundaries with a continuous VAE (the control).

Usage (cwd = the SeedVR2 checkout, its venv's python):
  python blend_patch.py inference_cli.py <CLI args> --temporal_overlap 4
  python3 bench.py run NAME --wrap blend_patch.py --env STITCH_CURVE=cosine -- <CLI args>
  python3 bench.py run NAME --wrap blend_patch.py --wrap ffv1_out.py -- <CLI args>   # chained
  python blend_patch.py --weights 8                    # print every curve's weights for K = 8
  STITCH_LATENT=6:1 STITCH_CURVE=cosine python blend_patch.py inference_cli.py <CLI args> \
      --load_cap 81 --batch_size 81 --temporal_overlap 0   # latent windows of 6, 1 shared
"""
import argparse
import copy
import math
import os
import runpy
import sys
import time
from pathlib import Path

CURVES = ("numz", "linear", "cosine", "prev", "cur")


def log(msg):
    print(f"blend_patch: {msg}", file=sys.stderr, flush=True)


def weights(curve, k):
    """Weight of the previous batch on the k overlap frames, as a list of floats."""
    if curve == "numz":  # src/core/generation_utils.py blend_overlapping_frames
        if k >= 3:
            out = []
            for j in range(k):
                t = j / (k - 1)
                u = min(max((t - 1 / 3) / (1 / 3), 0.0), 1.0)
                out.append(0.5 + 0.5 * math.cos(math.pi * u))
            return out
        return [1.0] if k == 1 else [1.0 - j / (k - 1) for j in range(k)]
    if curve == "linear":
        return [1.0 - i / (k + 1) for i in range(1, k + 1)]
    if curve == "cosine":
        return [0.5 + 0.5 * math.cos(math.pi * i / (k + 1)) for i in range(1, k + 1)]
    if curve == "prev":
        return [1.0] * k
    if curve == "cur":
        return [0.0] * k
    raise SystemExit(f"blend_patch: unknown curve {curve!r} (choose from {', '.join(CURVES)})")


def blend(prev_tail, cur_head, overlap, curve):
    import torch
    w = torch.tensor(weights(curve, overlap), dtype=torch.float32, device=prev_tail.device)
    w_prev = w.to(prev_tail.dtype).view(overlap, 1, 1, 1)
    return prev_tail * w_prev + cur_head * (1.0 - w_prev)


class State:
    def __init__(self):
        self.curve = os.environ.get("STITCH_CURVE", "numz").strip()
        self.extra = [c.strip() for c in os.environ.get("STITCH_EXTRA", "").split(",") if c.strip()]
        self.extra_dir = os.environ.get("STITCH_EXTRA_DIR")
        self.extra_ffv1 = [f.strip() for f in os.environ.get("STITCH_EXTRA_FFV1", "").split(",") if f.strip()]
        for c in [self.curve, *self.extra]:
            weights(c, 3)  # validates the name
        lat = os.environ.get("STITCH_LATENT", "").strip()
        self.latent = tuple(int(x) for x in lat.split(":")) if lat else None
        self.args = None
        self.pairs = []    # (prev_tail, cur_head) of every Phase 3 blend, in batch order
        self.patched = False


S = State()


def patched_blend(prev_tail, cur_head, overlap):
    if S.extra:
        S.pairs.append((prev_tail.clone(), cur_head.clone()))
    return blend(prev_tail, cur_head, overlap, S.curve)


def save_pngs(frames, out_dir):
    """As the CLI's save_frames_to_image: float [0, 1] -> uint8 by truncation, RGB -> BGR."""
    import cv2
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    arr = (frames.float().cpu().numpy() * 255.0).astype(np.uint8)
    for i, f in enumerate(arr):
        cv2.imwrite(os.path.join(out_dir, f"frame_{i:06d}.png"), cv2.cvtColor(f[..., :3], cv2.COLOR_RGB2BGR))
    return len(arr)


def save_ffv1(frames, base):
    from fractions import Fraction
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ffv1_out  # noqa: E402  (module import: only defines functions)
    arr = frames.float().cpu().numpy()
    T, H, W, _ = arr.shape
    fps = ffv1_out.probe_fps(S.args.input) or Fraction(24000, 1001)
    for fmt in S.extra_ffv1:
        path = f"{base}.{fmt}.mkv"
        w = ffv1_out.Writer(path, fmt, W, H, fps, 16)
        for f in arr:
            w.write(ffv1_out.to_planar(f, w.bits))
        rc = w.close()
        log(f"{'FAILED ' if rc else ''}{path}: {T} frames, {fmt}, {fps} fps")


def render_extras(orig_pp, a, k, ctx):
    info = ctx.get("decode_batch_info") or []
    overlap = ctx.get("actual_temporal_overlap", 0)
    pairs, S.pairs = S.pairs, []
    if not overlap:
        log("no temporal overlap in this run: the extra curves equal the main output, not rendered")
        return
    if len(pairs) != len(info) - 1:
        log(f"{len(pairs)} blends recorded for {len(info)} batches: extras not rendered")
        return
    base_dir = S.extra_dir or os.path.join(S.args.output or "output", "extra")
    for curve in S.extra:
        t0 = time.perf_counter()
        c = dict(ctx)
        c["final_video"] = ctx["final_video"].clone()
        c["decode_batch_info"] = list(info)
        # Phase 4's cleanup clears the transforms' __dict__ in place: give the copy its own
        if c.get("video_transform") is not None:
            c["video_transform"] = copy.deepcopy(ctx["video_transform"])
        for (write_start, _, _, _), (prev_tail, cur_head) in zip(info[1:], pairs):
            s = write_start - overlap
            c["final_video"][s:write_start] = blend(prev_tail.to(c["final_video"].device),
                                                    cur_head.to(c["final_video"].device), overlap, curve)
        kk = dict(k)
        kk["ctx"] = c
        c = orig_pp(c, *a[1:], **k) if a else orig_pp(**kk)
        frames = c["final_video"]
        n = save_pngs(frames, os.path.join(base_dir, curve))
        if S.extra_ffv1:
            save_ffv1(frames, os.path.join(base_dir, curve))
        log(f"extra curve {curve} {[round(x, 3) for x in weights(curve, overlap)]}: {n} frames in "
            f"{os.path.join(base_dir, curve)} ({time.perf_counter() - t0:.1f} s)")
        del c, frames


def latent_windows(t, w, m):
    """[start, end) of overlapping windows of w latents (m shared) covering t latents."""
    step = w - m
    if step <= 0:
        raise SystemExit(f"blend_patch: STITCH_LATENT overlap {m} >= window {w}")
    wins, s = [], 0
    while True:
        e = min(s + w, t)
        wins.append((s, e))
        if e >= t:
            return wins
        s += step


def latent_upscale(orig_up, a, k):
    """Phase 2 on overlapping windows of the single batch's latents, merged back into one."""
    import torch
    ctx = k["ctx"] if "ctx" in k else a[1]
    lats = [x for x in ctx["all_latents"] if x is not None]
    if len(lats) != 1:
        log(f"STITCH_LATENT needs one encoded batch (--batch_size >= clip length), got {len(lats)}: not applied")
        return orig_up(*a, **k)
    lat = lats[0]  # [T, H, W, C], channels last
    w, m = S.latent
    wins = latent_windows(lat.shape[0], w, m)
    log(f"latent windows {wins} over {lat.shape[0]} latents ({tuple(lat.shape)}), {m} shared, curve {S.curve}")
    ctx["all_latents"] = [lat[s:e].clone() for s, e in wins]
    del lat, lats
    ctx = orig_up(*a, **k)
    ups = ctx["all_upscaled_latents"]
    merged = ups[0].clone()
    for (s0, e0), (s, e), up in zip(wins, wins[1:], ups[1:]):
        shared = e0 - s  # latents s .. e0-1 are in both windows
        if shared > 0:
            wp = torch.tensor(weights(S.curve, shared), dtype=torch.float32, device=up.device)
            wp = wp.to(up.dtype).view(shared, *([1] * (up.dim() - 1)))
            prev = merged[s:e0].to(up.device)
            merged_part = prev * wp + up[:shared] * (1.0 - wp)
            merged = torch.cat([merged[:s].to(up.device), merged_part, up[shared:]], dim=0)
        else:
            merged = torch.cat([merged.to(up.device), up], dim=0)
    ctx["all_upscaled_latents"] = [merged]
    log(f"merged {len(ups)} windows into {tuple(merged.shape)}")
    return ctx


def patch_cli(g):
    """g: the globals of the running inference_cli.py."""
    gp = sys.modules.get("src.core.generation_phases")
    gu = sys.modules.get("src.core.generation_utils")
    for mod in (gp, gu):
        if mod is not None and hasattr(mod, "blend_overlapping_frames"):
            mod.blend_overlapping_frames = patched_blend
    g["blend_overlapping_frames"] = patched_blend  # multi-GPU join
    orig_pp = g["postprocess_all_batches"]

    def postprocess_all_batches(*a, **k):
        ctx = k["ctx"] if "ctx" in k else a[0]
        if S.extra:
            render_extras(orig_pp, a, k, ctx)
        return orig_pp(*a, **k)

    g["postprocess_all_batches"] = postprocess_all_batches
    if S.latent:
        orig_up = g["upscale_all_batches"]
        g["upscale_all_batches"] = lambda *a, **k: latent_upscale(orig_up, a, k)
    S.patched = True
    log(f"curve {S.curve}" + (f", extra curves {', '.join(S.extra)}" if S.extra else ""))


def install():
    """Patch when the CLI parses its arguments (every function and import of inference_cli.py
    exists by then), wherever this wrapper sits in a chain of wrappers."""
    orig = argparse.ArgumentParser.parse_args

    def parse_args(self, *a, **k):
        res = orig(self, *a, **k)
        if not S.patched:
            fr = sys._getframe(1)
            while fr is not None:
                g = fr.f_globals
                if "postprocess_all_batches" in g and "process_single_file" in g:
                    S.args = res
                    patch_cli(g)
                    k_ = getattr(res, "temporal_overlap", 0)
                    if k_:
                        for c in dict.fromkeys([S.curve, *S.extra]):
                            log(f"K={k_} {c}: previous-batch weights {[round(x, 3) for x in weights(c, k_)]}")
                    break
                fr = fr.f_back
        return res

    argparse.ArgumentParser.parse_args = parse_args


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        sys.exit(__doc__)
    if sys.argv[1] == "--weights" and len(sys.argv) == 3:
        k = int(sys.argv[2])
        for c in CURVES:
            print(f"{c:7s} {[round(x, 3) for x in weights(c, k)]}")
        return
    if sys.argv[1].startswith("--"):
        sys.exit(__doc__)
    script = os.path.abspath(sys.argv[1])
    sys.argv = [script, *sys.argv[2:]]
    sys.path[0] = os.path.dirname(script)
    install()
    runpy.run_path(script, run_name="__main__")
    if not S.patched:
        log("the CLI's functions were never found: nothing patched")


if __name__ == "__main__":
    main()

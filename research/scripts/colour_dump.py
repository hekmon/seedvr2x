#!/usr/bin/env python3
"""Dumps for the colour study (docs/colour.md, step 1): what a colour correction post-processes.

  COLOUR_DUMP=DIR python colour_dump.py [WRAPPER.py ...] inference_cli.py ARGS   # a numz run
  bench.py run NAME --wrap colour_dump.py --env COLOUR_DUMP=DIR -- ARGS          # the same, recorded
  python colour_dump.py decode DIR --out OUT [--untiled] [--tile T:O ...] [--check] [--vae-dtype fp16]
                                                                                  # GPU, under the lock
  python colour_dump.py verify DIR --master MASTER.mkv                            # CPU

Why: colour corrections can be compared without a model run each when they all start from the
same decoded frames and reference (DESIGN.md, Beyond numz's lab: the protocol). This saves what
numz's Phase 4 receives, and the decode again with other VAE tiles from the same latents.

As a wrapper of numz's CLI (hooks set when the CLI parses its arguments, as numerics_patch.py
does; nothing changes the numerics), with COLOUR_DUMP=DIR, into DIR:
- enc_bf16.pt: the encoder's exact input, the tensors runner.vae_encode receives, one per batch:
  numz's frames as k/255 in float16 on the CPU, bfloat16 on the VAE's device, then its
  video_transform (bicubic resize with antialias, clamp to [0, 1], padding to multiples of 16,
  normalisation to [-1, 1]) in bfloat16. [C, T, H, W] bfloat16.
- ref_f32.pt: the same frames through the same video_transform, on the same device, in float32:
  k/255 exactly (k recovered from the float16 frames, which hold every 8-bit code exactly), no
  float16 or bfloat16 step. Design asked for both references (2026-10-04): the bfloat16 one
  reads brighter after a resize. [C, T, H, W] float32.
- latents.pt: what runner.vae_decode receives, one per batch: the DiT's output, bfloat16; and
  what numz's Phase 3 needs to decode it again (the batches' frame counts, the true output size,
  the frame count, the temporal overlap).
- decode.pt: the raw decode, numz's final_video as Phase 4 receives it: [T, H, W, C] bfloat16 in
  [-1, 1], unclamped, trimmed to the true output size and the batches' frames; with
  decode_batch_info.
- meta.json: the command, numz's setup_generation_context and prepare_runner arguments (decode
  rebuilds the runner from them), shapes, dtypes, sha256 and size of every file, the mean of
  enc_bf16 - ref_f32 in 8-bit levels, timings.
COLOUR_DUMP_INPUTS=0 skips enc_bf16.pt and ref_f32.pt: they depend on the clip and the
resolution only, so they are dumped once per clip. With --color_correction none, Phase 1 is told
lab only so that it keeps the batch indices the float32 reference needs (numerics_patch.py does
the same); Phase 4 stays none.

decode: numz's runner rebuilt from DIR/meta.json with other decode tiles; DIR's latents decoded
through numz's own Phase 3 (decode_all_batches), once per tiling, the VAE kept loaded between
them: OUT/decode-<tile>-<overlap>.pt or OUT/decode-untiled.pt (decode.pt's layout), and
OUT/decode.json (time and torch peak per decode). --check decodes untiled first and stops unless
it equals DIR/decode.pt bit for bit. Run from the numz checkout (or SEEDVR2_DIR), with its venv.
--vae-dtype fp16 decodes in float16 instead (numerics.md's dec16; DESIGN.md asks for it scored after
the colour study's winner): the VAE file's own float16 weights, unconverted, and the latents cast to
float16 (bfloat16 values, exact unless out of float16's range: counted), so numz's vae_decode runs
without autocast, as decode_resume.py's --vae-dtype fp16 builds it. Its output stays float32 after
the decoder: numz's decode_all_batches would cast it into a bfloat16 final_video, so the decode
runs numz's Phase 3 steps here (vae_decode, rearrange, trims to the batch's frames and the true
size, channels last) without that cast, for one batch or batches without temporal overlap:
OUT/decode-untiled-fp16.pt or OUT/decode-<tile>-<overlap>-fp16.pt, final_video float32, with the
non-finite values of the decoder's output counted (float16 ends at 65504).

verify: DIR/decode.pt brought to [0, 1] as numz's Phase 4 does with --color_correction none
(clamp to [-1, 1], times 0.5, plus 0.5, in bfloat16) and quantised as ffv1_out.py does
(rint(x * 65535) in float32), against the 16-bit FFV1 master of a run with the same settings:
equal sample for sample, or the differences counted. Needs ffmpeg.
"""
import argparse
import atexit
import hashlib
import math
import json
import os
import runpy
import shutil
import subprocess
import sys
import time


class State:
    dir = None
    inputs = True
    patched = False
    meta = {"command": None, "files": {}, "events": []}


S = State()


def log(msg):
    print(f"colour_dump: {msg}", file=sys.stderr, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def jsonable(v):
    """Arguments recorded as JSON: tuples as lists, devices and dtypes as strings, anything else
    that isn't plain data dropped (None)."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    name = type(v).__name__
    if name in ("device", "dtype"):
        return str(v)
    return None


def describe(t):
    return {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")}


def save(name, obj, info):
    import torch
    path = os.path.join(S.dir, name)
    t0 = time.perf_counter()
    torch.save(obj, path)
    S.meta["files"][name] = {**info, "bytes": os.path.getsize(path), "sha256": sha256(path),
                             "seconds": round(time.perf_counter() - t0, 1)}
    log(f"{name}: {S.meta['files'][name]['bytes'] / 2**20:.0f} MiB")


def write_meta():
    if S.dir:
        with open(os.path.join(S.dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(S.meta, f, indent=1)


def phases_module():
    for name, m in list(sys.modules.items()):
        if name.endswith("core.generation_phases") and hasattr(m, "_prepare_video_batch"):
            return m
    raise RuntimeError("colour_dump: numz's generation_phases module is not loaded")


# ---------------------------------------------------------------- the wrapper

def swap_method(obj, name, fn):
    """Set fn as obj's own attribute name; returns the function that restores what was there."""
    own = name in obj.__dict__
    prev = obj.__dict__.get(name)
    setattr(obj, name, fn)

    def restore():
        if own:
            setattr(obj, name, prev)
        else:
            delattr(obj, name)
    return restore


def dump_inputs(ctx, images, captured):
    """enc_bf16.pt from the tensors vae_encode received; ref_f32.pt by Phase 1's own steps in
    float32 (generation_phases.py:369-413): batch slice, 4n+1 padding, video_transform."""
    import torch
    gp = phases_module()
    batches = ctx.get("batch_metadata")
    if not batches or len(batches) != len(captured):
        raise RuntimeError(f"colour_dump: {len(captured)} encodes, batch metadata {batches!r}")
    codes = images.to(torch.float32).mul(255)
    k = codes.round()
    off = float((codes - k).abs().max()) if codes.numel() else 0.0
    if off > 0.25:  # float16 holds k/255 within 0.07 of a code; anything else isn't 8-bit input
        raise RuntimeError(f"colour_dump: the input frames are not k/255 (off by {off:.3f} of a code)")
    exact = k.div_(255)
    enc, ref, bias = [], [], []
    for (start, end, upad), tensors in zip(batches, captured):
        x = tensors[0]
        video = gp._prepare_video_batch(images=exact, start_idx=start, end_idx=end, uniform_padding=upad)
        video = video.to(ctx["vae_device"], dtype=torch.float32)
        video = gp._apply_4n1_padding(video)
        rgb = video[:, :3, :, :] if ctx.get("is_rgba", False) else video
        r = ctx["video_transform"](rgb).to("cpu")
        if tuple(r.shape) != tuple(x.shape):
            raise RuntimeError(f"colour_dump: float32 reference {tuple(r.shape)}, encoder input {tuple(x.shape)}")
        bias.append(float((x.to(torch.float32) - r).mean()) * 127.5)
        enc.append(x)
        ref.append(r)
        del video, rgb
    S.meta["enc_minus_ref_levels"] = [round(b, 4) for b in bias]
    log(f"encoder input minus float32 reference, mean per batch: {', '.join(f'{b:+.3f}' for b in bias)} 8-bit levels")
    save("enc_bf16.pt", enc, {"what": "runner.vae_encode inputs", "batches": [describe(t) for t in enc]})
    save("ref_f32.pt", ref, {"what": "video_transform of k/255 in float32", "batches": [describe(t) for t in ref]})


def patch_cli(g):
    """g: the globals of the running inference_cli.py."""
    names = ("setup_generation_context", "prepare_runner", "encode_all_batches", "decode_all_batches",
             "postprocess_all_batches")
    missing = [n for n in names if n not in g]
    if missing:
        log(f"the CLI has no {', '.join(missing)}: nothing dumped")
        return
    S.patched = True
    orig = {n: g[n] for n in names}

    def setup_generation_context(*a, **k):
        S.meta["setup_generation_context"] = {n: jsonable(v) for n, v in k.items() if n != "debug"}
        return orig["setup_generation_context"](*a, **k)

    def prepare_runner(*a, **k):
        S.meta["prepare_runner"] = {n: jsonable(v) for n, v in k.items() if n not in ("debug", "ctx")}
        return orig["prepare_runner"](*a, **k)

    def encode_all_batches(runner, *a, **k):
        if k.get("color_correction") == "none":
            k = dict(k, color_correction="lab")  # keeps ctx['batch_metadata']; encodes the same
        captured = []
        enc = runner.vae_encode

        def vae_encode(videos, *aa, **kk):
            if S.inputs:
                captured.append([v.detach().to("cpu", copy=True) for v in videos])
            return enc(videos, *aa, **kk)
        restore = swap_method(runner, "vae_encode", vae_encode)
        t0 = time.perf_counter()
        try:
            out = orig["encode_all_batches"](runner, *a, **k)
        finally:
            restore()
        S.meta["events"].append({"phase1_s": round(time.perf_counter() - t0, 2)})
        if S.inputs:
            dump_inputs(k["ctx"], k["images"], captured)
        return out

    def decode_all_batches(runner, *a, **k):
        ctx = k["ctx"]
        phase3 = {"all_ori_lengths": jsonable(ctx["all_ori_lengths"]),
                  "true_target_dims": jsonable(ctx["true_target_dims"]),
                  "total_frames": ctx.get("total_frames"),
                  "actual_temporal_overlap": ctx.get("actual_temporal_overlap", 0),
                  "is_rgba": bool(ctx.get("is_rgba", False))}
        latents = []
        dec = runner.vae_decode

        def vae_decode(zs, *aa, **kk):
            latents.append(zs[0].detach().to("cpu", copy=True))
            return dec(zs, *aa, **kk)
        restore = swap_method(runner, "vae_decode", vae_decode)
        t0 = time.perf_counter()
        try:
            out = orig["decode_all_batches"](runner, *a, **k)
        finally:
            restore()
        S.meta["events"].append({"phase3_s": round(time.perf_counter() - t0, 2)})
        S.meta["vae_tiles"] = {n: jsonable(getattr(runner, n, None)) for n in (
            "encode_tiled", "encode_tile_size", "encode_tile_overlap",
            "decode_tiled", "decode_tile_size", "decode_tile_overlap")}
        save("latents.pt", {"latents": latents, "phase3": phase3},
             {"what": "runner.vae_decode inputs", "batches": [describe(z) for z in latents], "phase3": phase3})
        return out

    def postprocess_all_batches(*a, **k):
        ctx = k["ctx"] if "ctx" in k else a[0]
        fv = ctx["final_video"]
        info = list(ctx.get("decode_batch_info") or [])
        save("decode.pt", {"final_video": fv, "decode_batch_info": info},
             {"what": "final_video at Phase 4's start", **describe(fv), "decode_batch_info": jsonable(info)})
        write_meta()
        return orig["postprocess_all_batches"](*a, **k)

    g.update(setup_generation_context=setup_generation_context, prepare_runner=prepare_runner,
             encode_all_batches=encode_all_batches, decode_all_batches=decode_all_batches,
             postprocess_all_batches=postprocess_all_batches)
    log(f"dumping into {S.dir}" + ("" if S.inputs else " (without the encoder input and reference)"))


def install():
    """The CLI's own globals when it parses its arguments (every function of inference_cli.py
    exists by then), wherever this wrapper sits in a chain."""
    import argparse
    orig = argparse.ArgumentParser.parse_args

    def parse_args(self, *a, **k):
        res = orig(self, *a, **k)
        if not S.patched:
            fr = sys._getframe(1)
            while fr is not None:
                g = fr.f_globals
                if "postprocess_all_batches" in g and "encode_all_batches" in g:
                    patch_cli(g)
                    break
                fr = fr.f_back
        return res

    argparse.ArgumentParser.parse_args = parse_args


def run_wrapped():
    S.dir = os.environ.get("COLOUR_DUMP")
    if not S.dir:
        sys.exit("colour_dump: set COLOUR_DUMP=DIR")
    os.makedirs(S.dir, exist_ok=True)
    S.inputs = os.environ.get("COLOUR_DUMP_INPUTS", "1") != "0"
    S.meta["command"] = [os.path.abspath(sys.argv[0]), *sys.argv[1:]]
    S.meta["cwd"] = os.getcwd()
    script = os.path.abspath(sys.argv[1])
    sys.argv = [script, *sys.argv[2:]]
    sys.path[0] = os.path.dirname(script)
    install()
    atexit.register(write_meta)
    runpy.run_path(script, run_name="__main__")
    if not S.patched:
        log("the CLI's functions were never found: nothing dumped")


# ---------------------------------------------------------------- decode

def decode_fp16(runner, ctx, latents, p3):
    """numz's Phase 3 (generation_phases.py:913-1020) for a float16 VAE, its output kept float32:
    per batch, the latent cast to float16 on the VAE's device (numz casts it to its compute dtype,
    :923-932), runner.vae_decode, optimized_video_rearrange, the batch's frames (ori_length, :950-958)
    and the true size (:961-966), channels last (:970); written into a float32 final_video on the
    CPU, where numz's is bfloat16 (:879, :1002-1017). Returns (final_video (T, H, W, C) float32 in
    [-1, 1] unclamped, decode_batch_info as numz's, counts)."""
    import torch
    from src.optimization.performance import optimized_video_rearrange, optimized_sample_to_image_format
    if len(latents) > 1 and p3["actual_temporal_overlap"]:
        sys.exit("colour_dump: --vae-dtype fp16 blends no temporal overlap between batches")
    true_h, true_w = p3["true_target_dims"]
    lengths = [n for n in p3["all_ori_lengths"] if n is not None]
    fv = torch.empty((p3["total_frames"], true_h, true_w, 3), dtype=torch.float32)
    info, at, nonfinite, inexact, lo, hi = [], 0, 0, 0, math.inf, -math.inf
    for i, z in enumerate(latents):
        z = z.to(ctx["vae_device"])
        z16 = z.to(torch.float16)
        inexact += int((z16.to(torch.float32) != z.to(torch.float32)).sum())
        sample = optimized_video_rearrange(runner.vae_decode([z16]))[0]  # (T, C, H, W) float16
        n = lengths[i] if i < len(lengths) else sample.shape[0]
        sample = optimized_sample_to_image_format(sample[:n, :, :true_h, :true_w])
        nonfinite += int(torch.isfinite(sample).logical_not_().sum())
        mn, mx = torch.aminmax(sample)
        lo, hi = min(lo, float(mn)), max(hi, float(mx))
        if sample.dtype != torch.float16:
            sys.exit(f"colour_dump: the float16 VAE returned {sample.dtype}")
        fv[at:at + sample.shape[0]] = sample.to("cpu", torch.float32)
        info.append((at, at + sample.shape[0], i, n))
        at += sample.shape[0]
        del sample, z, z16
    if at != fv.shape[0]:
        sys.exit(f"colour_dump: {at} frames decoded, {fv.shape[0]} expected")
    log(f"float16 decode: {nonfinite} non-finite values, range [{lo:.6g}, {hi:.6g}], "
        f"{inexact} latent values inexact in float16")
    return fv, info, {"nonfinite": nonfinite, "decoder_range": [lo, hi], "latent_values_inexact_fp16": inexact}


def cmd_decode(a):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")  # the CLI's default
    root = os.path.abspath(os.environ.get("SEEDVR2_DIR", os.getcwd()))
    if root not in sys.path:
        sys.path.insert(0, root)
    import torch
    from src.utils.debug import Debug
    from src.core.generation_utils import setup_generation_context, prepare_runner
    from src.core.generation_phases import decode_all_batches

    with open(os.path.join(a.dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    data = torch.load(os.path.join(a.dir, "latents.pt"))
    p3 = data["phase3"]
    os.makedirs(a.out, exist_ok=True)
    debug = Debug(enabled=False)
    ctx = setup_generation_context(**meta["setup_generation_context"], debug=debug)
    kw = dict(meta["prepare_runner"])
    for n in ("encode_tile_size", "encode_tile_overlap", "decode_tile_size", "decode_tile_overlap"):
        kw[n] = tuple(kw[n])
    kw.update(dit_cache=False, vae_cache=False, dit_id=None, vae_id=None)
    runner, cache_context = prepare_runner(**kw, debug=debug, ctx=ctx)
    ctx["cache_context"] = cache_context
    fp16 = a.vae_dtype == "fp16"
    if fp16:
        if a.check:
            sys.exit("colour_dump: --check compares with the CLI's bfloat16 decode: not with --vae-dtype fp16")
        from src.core.model_loader import materialize_model
        runner._vae_dtype_override = torch.float16  # numz sets its compute dtype (bfloat16) here
        runner.config.vae.dtype = "float16"
        materialize_model(runner, "vae", ctx["vae_device"], runner.config, debug)
        wdt = {p.dtype for p in runner.vae.parameters()}
        if wdt != {torch.float16}:
            sys.exit(f"colour_dump: the float16 VAE's weights are {sorted(str(d) for d in wdt)}")
    suffix = "-fp16" if fp16 else ""
    plan = ([None] if a.untiled or a.check else []) + [tuple(int(v) for v in t.split(":")) for t in a.tile]
    report = {"dir": os.path.abspath(a.dir), "vae_dtype": a.vae_dtype, "decodes": []}
    for tiles in plan:
        runner.decode_tiled = tiles is not None
        if tiles:
            runner.decode_tile_size, runner.decode_tile_overlap = (tiles[0],) * 2, (tiles[1],) * 2
        ctx.update(all_upscaled_latents=[z.clone() for z in data["latents"]],
                   all_ori_lengths=list(p3["all_ori_lengths"]), true_target_dims=tuple(p3["true_target_dims"]),
                   total_frames=p3["total_frames"], actual_temporal_overlap=p3["actual_temporal_overlap"],
                   is_rgba=p3["is_rgba"])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        extra = {}
        if fp16:
            ctx["final_video"], ctx["decode_batch_info"], extra = decode_fp16(runner, ctx, data["latents"], p3)
        else:
            decode_all_batches(runner, ctx=ctx, debug=debug, progress_callback=None, cache_model=True)
        torch.cuda.synchronize()
        secs = time.perf_counter() - t0
        fv = ctx.pop("final_video")
        name = (f"decode-untiled{suffix}.pt" if tiles is None else f"decode-{tiles[0]}-{tiles[1]}{suffix}.pt")
        path = os.path.join(a.out, name)
        torch.save({"final_video": fv, "decode_batch_info": list(ctx.get("decode_batch_info") or [])}, path)
        row = {"file": name, "tiles": list(tiles) if tiles else None, "vae_dtype": a.vae_dtype,
               "seconds": round(secs, 2), "torch_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
               **describe(fv), **extra, "sha256": sha256(path)}
        if tiles is None and a.check:
            ref = torch.load(os.path.join(a.dir, "decode.pt"))["final_video"]
            same = tuple(ref.shape) == tuple(fv.shape) and bool(torch.equal(ref, fv))
            row["equals_cli_decode"] = same
            if not same:
                d = (ref.to(torch.float32) - fv.to(torch.float32)).abs() if ref.shape == fv.shape else None
                row["max_abs_diff"] = None if d is None else float(d.max())
                report["decodes"].append(row)
                print(json.dumps(row))
                sys.exit("colour_dump: the untiled decode differs from the CLI's: stopping")
        report["decodes"].append(row)
        print(json.dumps(row), flush=True)
        del fv
    with open(os.path.join(a.out, "decode.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)


# ---------------------------------------------------------------- verify

def cmd_verify(a):
    import numpy as np
    import torch
    ffmpeg = os.environ.get("COLOUR_FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"
    fv = torch.load(os.path.join(a.dir, "decode.pt"))["final_video"]
    if fv.dtype != torch.bfloat16:
        sys.exit(f"colour_dump: decode.pt is {fv.dtype}, bfloat16 expected")
    t, h, w, c = fv.shape
    unit = fv.clone().clamp_(-1, 1).mul_(0.5).add_(0.5)  # numz's Phase 4 with none, in bfloat16
    proc = subprocess.Popen([ffmpeg, "-v", "error", "-nostdin", "-i", a.master, "-map", "0:v:0",
                             "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "gbrp16le", "-"],
                            stdout=subprocess.PIPE)
    size = 3 * h * w * 2
    diff_frames, diff_samples, worst = 0, 0, 0
    for i in range(t):
        buf = proc.stdout.read(size)
        if len(buf) < size:
            sys.exit(f"colour_dump: {a.master} ends at frame {i}, the dump has {t}")
        g_, b_, r_ = np.frombuffer(buf, "<u2").reshape(3, h, w)
        master = np.stack([r_, g_, b_], axis=-1).astype(np.int64)
        q = np.clip(unit[i].to(torch.float32).numpy(), 0.0, 1.0)  # ffv1_out.py's to_planar
        q *= 65535
        np.rint(q, out=q)
        d = np.abs(q.astype(np.int64) - master)
        n = int((d > 0).sum())
        if n:
            diff_frames += 1
            diff_samples += n
            worst = max(worst, int(d.max()))
    proc.kill()
    proc.stdout.close()
    proc.wait()
    print(f"{a.dir}: {t} frames against {a.master}: "
          + ("equal, sample for sample" if not diff_frames else
             f"{diff_frames} frames differ, {diff_samples} samples, largest {worst} codes"))
    if diff_frames:
        sys.exit(1)


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("decode", "verify"):
        ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
        sub = ap.add_subparsers(dest="cmd", required=True)
        d = sub.add_parser("decode")
        d.add_argument("dir")
        d.add_argument("--out", required=True)
        d.add_argument("--untiled", action="store_true")
        d.add_argument("--tile", action="append", default=[], metavar="TILE:OVERLAP")
        d.add_argument("--check", action="store_true")
        d.add_argument("--vae-dtype", choices=("bf16", "fp16"), default="bf16",
                       help="bf16: numz's (the fp16 file converted on load); fp16: the file's own weights, "
                            "float16 latents, no autocast, float32 after the decoder")
        v = sub.add_parser("verify")
        v.add_argument("dir")
        v.add_argument("--master", required=True)
        a = ap.parse_args()
        return cmd_decode(a) if a.cmd == "decode" else cmd_verify(a)
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        sys.exit(__doc__)
    run_wrapped()


if __name__ == "__main__":
    main()

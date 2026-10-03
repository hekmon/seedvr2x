#!/usr/bin/env python3
"""Decode resume for SeedVR2's VAE: is a decode resumed mid-shot bit-identical to one pass?

The VAE decoder is causal in time and decodes in temporal slices: the first slice holds latents
0-1 (state INITIALIZING, 5 frames), then one latent per slice (state ACTIVE, 4 frames). Every
InflatedCausalConv3d keeps the last k_t - s_t input frames of the previous slice (`memory`); a
fresh decode starts from replicated first frames instead. This script measures, on real latents:

- determinism: the one-pass decode of all latents, repeated in one process, and against a
  reference saved by another process (--ref-load);
- warm-up resume: a fresh decode of latents [s - w, s + K) whose first 4w - 3 frames are dropped,
  against the one-pass frames of latents s .. s + K - 1, for each warm-up w: bf16 bit-identity,
  max |diff| (bf16 units, [-1, 1]), PSNR on [0, 1], identity after 16-bit (round(x * 65535)) and
  10-bit (round(x * 1023)) quantisation of x = clamp((y + 1) / 2, 0, 1);
- profile: one fresh decode from latent a to the end, every later latent compared (error against
  the distance d = j - a, for every d at once);
- snapshot resume: during a one-pass decode, every conv cache deep-copied at the boundary before
  latent s; then a decode of latents s .. s + K - 1 in state ACTIVE from the restored caches,
  restored once from the GPU copy and once after a round trip through a file (written, fsynced,
  dropped from the page cache, read back); size, and copy / save / load times;
- time of every temporal slice (one latent each in state ACTIVE), so a warm-up's cost is known.

Everything goes through numz's own code, as inference_cli.py runs it: the runner and the VAE
come from prepare_runner + materialize_model (CLI defaults: 7B config, bf16 compute dtype,
caches on the GPU, VAE memory limits 0.5 / 0.5 GiB, tensor offload to the CPU), frames go
through prepare_video_transforms (pre-sized input: no resize; padded to multiples of 16) and
VideoDiffusionInfer.vae_encode, latents through VideoDiffusionInfer.vae_decode, from contiguous
[T, H, W, C] bf16 tensors (the layout of the DiT's output). --tiled = the CLI's
--vae_decode_tiled with --vae_decode_tile_size / --vae_decode_tile_overlap (tiles accumulate on
the CPU, as with the CLI's default --tensor_offload_device cpu). A wrapper on the VAE instance's
_decode (one temporal slice) times each slice and takes the snapshot; it changes no numerics.

Subcommands:
  encode --input VIDEO --frames N [--start F] --size WxH --out LAT.pt
         Frames decoded by ffmpeg to RGB24 at WxH (BT.709 matrix, limited range in; Lanczos
         when resizing), fed as the CLI feeds its cv2 frames (float32 / 255, then bf16 on the
         GPU), encoded once. N must be 4n + 1. LAT.pt keeps the latents (bf16 [T, H/8, W/8, 16],
         scaled by the CLI's 0.9152) and the input frames of the --keep latents (sanity check).
  run    --latents LAT.pt --out RESULTS.json [options below]
         --ref-runs N        one-pass decodes in this process (default 1; all compared)
         --ref-save PATH     save the first one-pass decode (bf16 [C, F, H, W])
         --ref-load PATH     use a saved one-pass decode as the reference
         --tiled TILE:OVERLAP  tiled decode (output pixels), e.g. 512:128
         --resume S[,S]      resume points (latent index)
         --warmup W[,W...]   warm-up latents for each resume point
         --k K               latents compared after each resume point (default 3)
         --profile A[,A]     fresh decodes from latent A to the end
         --snapshot S        snapshot test at latent S (untiled only; needs a decode here)
         --snapshot-dir DIR  where the snapshot file is written (then deleted)
  dtype-ab --latents LAT.pt --input VIDEO --out AB.json [--n-latents 9] [--pairs 3]
         [--phases decode,tiled,encode] [--tiled 1024:128] [--budget-min 19]
         Back to back in one process, alternating bf16, fp16, bf16, fp16…: a decode of the first N
         latents, the same tiled, and an encode of their 4N - 3 source frames. Per run: time
         (synchronized), torch peak, SM clock / power / temperature / power-cap share (NVML),
         non-finite outputs, identity with the first run of the same dtype; fp16 vs bf16 max and
         mean |difference| of the outputs. Stops before a run that would overrun the budget.
  table  RESULTS.json [...]  Markdown tables

Options for encode, run and dtype-ab: --model-dir (the CLI's --model_dir), --dit-model (selects
the config, default seedvr2_ema_7b_fp16.safetensors), --vae-dtype bf16|fp16 (encode and run):
bf16 (default) = numz, the VAE file's fp16 weights converted to bfloat16 on load; fp16 = the
file's own float16 weights, unconverted, and float16 inputs (latents cast exactly or nearly,
numz's bfloat16 frames cast exactly), so numz's vae_encode / vae_decode run without autocast.
Environment: SEEDVR2_DIR (the numz checkout, default: cwd), FFMPEG (default: ffmpeg).

Usage (cwd = the SeedVR2 checkout, its venv's python; GPU commands under the shared GPU lock):
  python decode_resume.py encode --input clip.mp4 --frames 201 --size 1280x720 \\
      --model-dir MODELS --out lat720.pt
  python decode_resume.py run --latents lat720.pt --model-dir MODELS --ref-runs 2 \\
      --ref-save ref720.pt --snapshot 45 --snapshot-dir DIR --profile 3 \\
      --resume 45 --warmup 1,2,4,8,16,24,32,36,37,38,40 --out r720.json
  python decode_resume.py run --latents lat720.pt --model-dir MODELS --ref-load ref720.pt \\
      --resume 41 --warmup 1,2,4,8,16,24,32,36,37,38,40 --out r720b.json
  python decode_resume.py table r720.json r720b.json
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

QUANT = {"q16": 65535, "q10": 1023}


def log(msg):
    print(f"[decode_resume {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ints(text):
    return [int(v) for v in text.split(",") if v.strip()] if text else []


# ---------------------------------------------------------------- numz setup

def setup_numz():
    """Make numz importable as inference_cli.py does (allocator set before torch is imported)."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
    root = os.path.abspath(os.environ.get("SEEDVR2_DIR", os.getcwd()))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def build_runner(args, tiled=None, vae_dtype=None):
    """The CLI's runner and VAE: prepare_runner with the CLI defaults, then materialize_model.

    vae_dtype "bf16" (default, numz): the VAE file's fp16 weights converted to bfloat16 on load.
    "fp16": the file's own float16 weights, unconverted; ctx["vae_in_dtype"] then makes the
    encode / decode inputs float16 too, so numz's vae_encode / vae_decode run without autocast
    (their autocast only kicks in when the input's dtype differs from the VAE's)."""
    import torch
    from src.utils.debug import Debug
    from src.utils.model_registry import DEFAULT_VAE
    from src.core.generation_utils import setup_generation_context, prepare_runner
    from src.core.model_loader import materialize_model

    debug = Debug(enabled=False)
    ctx = setup_generation_context(dit_device="cuda:0", vae_device="cuda:0", dit_offload_device=None,
                                   vae_offload_device=None, tensor_offload_device="cpu", debug=debug)
    tile, overlap = tiled if tiled else (1024, 128)  # CLI defaults when untiled
    runner, _ = prepare_runner(
        dit_model=args.dit_model, vae_model=DEFAULT_VAE, model_dir=args.model_dir, debug=debug, ctx=ctx,
        dit_cache=False, vae_cache=False, dit_id=None, vae_id=None,
        block_swap_config={"blocks_to_swap": 0, "swap_io_components": False, "offload_device": None},
        encode_tiled=False, encode_tile_size=(1024, 1024), encode_tile_overlap=(128, 128),
        decode_tiled=bool(tiled), decode_tile_size=(tile, tile), decode_tile_overlap=(overlap, overlap),
        tile_debug="false", attention_mode="sdpa", torch_compile_args_dit=None, torch_compile_args_vae=None)
    vae_dtype = vae_dtype or getattr(args, "vae_dtype", None) or "bf16"
    if vae_dtype == "fp16":
        runner._vae_dtype_override = torch.float16  # numz sets its compute dtype (bfloat16) here
        runner.config.vae.dtype = "float16"
        ctx["vae_in_dtype"] = torch.float16
    elif vae_dtype != "bf16":
        sys.exit(f"unknown VAE dtype {vae_dtype}")
    materialize_model(runner, "vae", ctx["vae_device"], runner.config, debug)
    return runner, ctx


def in_dtype(ctx):
    """The dtype of the VAE's inputs: numz's compute dtype, or the fp16 VAE's own."""
    return ctx.get("vae_in_dtype", ctx["compute_dtype"])


def env_info(runner):
    import torch
    cil = sys.modules.get("src.models.video_vae_v3.modules.causal_inflation_lib")
    vae = runner.vae
    return {
        "torch": str(torch.__version__), "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0), "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "conv3d_workaround": getattr(cil, "NVIDIA_CONV3D_MEMORY_BUG_WORKAROUND", None),
        "vae_dtype": str(next(vae.parameters()).dtype), "vae_training": vae.training,
        "slicing_latent_min_size": getattr(vae, "slicing_latent_min_size", None),
        "conv_memory_limit_gib": next((m.memory_limit for m in vae.modules()
                                       if type(m).__name__ == "InflatedCausalConv3d"), None),
        "memory_device": next((m.memory_device for m in vae.modules()
                               if type(m).__name__ == "InflatedCausalConv3d"), None),
        "tensor_offload_device": str(getattr(vae, "tensor_offload_device", None)),
        "decode_tiled": runner.decode_tiled, "decode_tile_size": list(runner.decode_tile_size),
        "decode_tile_overlap": list(runner.decode_tile_overlap),
        "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    }


# ---------------------------------------------------------------- slice hook (timing + snapshot)

class SliceHook:
    """Replaces vae._decode (one temporal slice) on the instance: times every slice, and can copy
    every conv cache right before a given slice. Calls the original method unchanged."""

    def __init__(self, vae):
        self.vae = vae
        self.orig = vae._decode
        self.convs = [(n, m) for n, m in vae.named_modules() if type(m).__name__ == "InflatedCausalConv3d"]
        self.reset()
        vae._decode = self

    def reset(self, snapshot_at=None):
        self.count = 0
        self.slices = []
        self.snapshot_at = snapshot_at
        self.snapshot = None
        self.snapshot_s = None

    def take_snapshot(self):
        return {n: m.memory.clone() for n, m in self.convs if m.memory is not None}

    def __call__(self, z, *a, **k):
        import torch
        if self.snapshot_at is not None and self.count == self.snapshot_at:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            self.snapshot = self.take_snapshot()
            torch.cuda.synchronize()
            self.snapshot_s = time.perf_counter() - t0
        state = k.get("memory_state", a[0] if a else None)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.orig(z, *a, **k)
        torch.cuda.synchronize()
        self.slices.append({"state": getattr(state, "name", "DISABLED"), "latents": int(z.shape[2]),
                            "s": round(time.perf_counter() - t0, 4)})
        self.count += 1
        return out

    def clear_memory(self):
        for _, m in self.convs:
            m.memory = None


def slice_stats(slices):
    act = sorted(s["s"] for s in slices if s["state"] == "ACTIVE")
    init = [s["s"] for s in slices if s["state"] == "INITIALIZING"]
    return {"n_slices": len(slices), "init_s": round(sum(init), 4) if init else None,
            "active_n": len(act), "active_mean_s": round(sum(act) / len(act), 4) if act else None,
            "active_median_s": act[len(act) // 2] if act else None,
            "active_min_s": act[0] if act else None, "active_max_s": act[-1] if act else None,
            "total_s": round(sum(s["s"] for s in slices), 4)}


# ---------------------------------------------------------------- decode paths

def decode(runner, ctx, lat_cpu):
    """The CLI's Phase 3 for one batch: latent to the VAE device, then vae_decode."""
    import torch
    from src.optimization.memory_manager import manage_tensor
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    z = manage_tensor(tensor=lat_cpu, target_device=ctx["vae_device"], dtype=in_dtype(ctx))
    out = runner.vae_decode([z])[0]  # [C, F, H, W], the VAE's dtype
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def decode_from_caches(runner, ctx, hook, lat_cpu, s, e, caches):
    """Latents s .. e-1 as numz's slicing_decode would run them after latent s-1: caches
    restored, state ACTIVE, one latent per slice; input prepared as vae_decode prepares it."""
    import torch
    from src.optimization.memory_manager import manage_tensor
    from src.optimization.performance import optimized_channels_to_second
    state = sys.modules[type(runner.vae).__module__].MemoryState
    cfg = runner.config.vae
    scale, shift = cfg.scaling_factor, cfg.get("shifting_factor", 0.0)
    for n, m in hook.convs:
        m.memory = caches.get(n)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        z = manage_tensor(tensor=lat_cpu[s:e], target_device=ctx["vae_device"], dtype=in_dtype(ctx))
        z = z.unsqueeze(0)
        z = z / scale + shift
        z = optimized_channels_to_second(z)
        outs = [hook(z[:, :, j:j + 1], memory_state=state.ACTIVE) for j in range(e - s)]
        out = torch.cat(outs, dim=2).squeeze(0)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    hook.clear_memory()
    return out, dt


# ---------------------------------------------------------------- comparisons

def frames_of(j, a=0):
    """Output frame range of latent j in a decode starting at latent a (j > a, or j == a == 0)."""
    d = j - a
    return (0, 1) if d == 0 else (4 * d - 3, 4 * d + 1)


def metrics(ref, out):
    """ref, out: [C, F, H, W] bf16 decoded frames (y in [-1, 1])."""
    import torch
    if ref.shape != out.shape:
        raise RuntimeError(f"shape mismatch {tuple(ref.shape)} vs {tuple(out.shape)}")
    if torch.equal(ref, out):
        r = {"identical": True, "max_abs": 0.0, "n_diff": 0, "psnr": None}
        for q in QUANT:
            r.update({q: True, f"{q}_ndiff": 0, f"{q}_maxcode": 0})
        return r
    rf, of = ref.float(), out.float()
    r = {"identical": False, "max_abs": float((rf - of).abs().max()), "n_diff": int((ref != out).sum()),
         "n": ref.numel()}
    r01 = ((rf + 1) * 0.5).clamp_(0, 1)
    o01 = ((of + 1) * 0.5).clamp_(0, 1)
    mse = float((r01 - o01).pow(2).mean())
    r["psnr"] = None if mse == 0 else round(10 * math.log10(1 / mse), 3)
    for q, full in QUANT.items():
        qr, qo = torch.round(r01 * full), torch.round(o01 * full)
        diff = (qr - qo).abs()
        r.update({q: bool(torch.equal(qr, qo)), f"{q}_ndiff": int((diff > 0).sum()),
                  f"{q}_maxcode": int(diff.max())})
    return r


def compare_latents(ref, out, a, latents, active_from=None):
    """Per latent j: a decode starting at latent a against the one-pass frames. A fresh decode
    gives latent a one frame (remove_head); a decode resumed from caches before latent
    active_from gives it 4 frames from the first one."""
    rows = []
    for j in latents:
        r0, r1 = frames_of(j)
        if active_from is None:
            o0, o1 = frames_of(j, a)
        else:
            o0, o1 = 4 * (j - active_from), 4 * (j - active_from) + 4
        m = metrics(ref[:, r0:r1], out[:, o0:o1])
        m.update({"latent": j, "distance": None if active_from is not None else j - a})
        if not m["identical"]:
            m["frame_identical"] = [bool((ref[:, r0 + i] == out[:, o0 + i]).all()) for i in range(r1 - r0)]
        rows.append(m)
    return rows


def recon_psnr(ref, frames, height, width):
    """Sanity check: PSNR (dB, RGB on [0, 1]) of the decoded frames of the kept latents
    against the input frames."""
    out = {}
    for j, f in frames.items():
        r0, r1 = frames_of(int(j))
        y = ((ref[:, r0:r1, :height, :width].float() + 1) * 0.5).clamp(0, 1).permute(1, 2, 3, 0)
        x = f.to(y.device).float() / 255.0
        mse = float((y - x).pow(2).mean())
        out[j] = round(10 * math.log10(1 / mse), 2) if mse > 0 else None
    return out


def summarize(rows):
    keys = ("identical", "q16", "q10")
    s = {k: all(r[k] for r in rows) for k in keys}
    s["max_abs"] = max(r["max_abs"] for r in rows)
    ps = [r["psnr"] for r in rows if r["psnr"] is not None]
    s["psnr_min"] = min(ps) if ps else None
    return s


# ---------------------------------------------------------------- encode

def read_frames(path, start, count, width, height):
    import numpy as np
    vf = [f"trim=start_frame={start},setpts=PTS-STARTPTS"] if start else []
    vf += [f"scale={width}:{height}:flags=lanczos+accurate_rnd+full_chroma_int:"
           f"in_color_matrix=bt709:in_range=tv:out_range=pc", "format=rgb24"]
    cmd = [os.environ.get("FFMPEG", "ffmpeg"), "-v", "error", "-nostdin", "-i", path, "-vf", ",".join(vf),
           "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, check=True)
    frame = width * height * 3
    if len(p.stdout) != count * frame:
        sys.exit(f"ffmpeg gave {len(p.stdout) / frame:.2f} frames, wanted {count}")
    return np.frombuffer(p.stdout, np.uint8).reshape(count, height, width, 3)


def cmd_encode(args):
    setup_numz()
    width, height = (int(v) for v in args.size.lower().split("x"))
    if args.frames % 4 != 1:
        sys.exit("--frames must be 4n + 1 (no temporal padding here)")
    log(f"reading {args.frames} frames from frame {args.start} at {width}x{height}")
    frames = read_frames(args.input, args.start, args.frames, width, height)
    import numpy as np
    import torch
    from src.core.generation_utils import prepare_video_transforms
    from src.optimization.memory_manager import manage_tensor
    runner, ctx = build_runner(args)
    images = torch.from_numpy(frames.astype(np.float32) / 255.0)  # CLI: cv2 frame .astype(float32) / 255
    video = images.permute(0, 3, 1, 2)  # _prepare_video_batch: TCHW view
    video = manage_tensor(tensor=video, target_device=ctx["vae_device"], dtype=ctx["compute_dtype"])
    transform = prepare_video_transforms(min(width, height), 0, None)
    x = transform(video)  # [C, T, Hp, Wp], bf16, [-1, 1]
    # pre-sized input: the transform must only pad and normalise (no resize)
    no_resize = bool(torch.equal(x[:, :, :height, :width], ((video - 0.5) / 0.5).permute(1, 0, 2, 3)))
    pad_black = bool((x[:, :, height:] == -1).all() and (x[:, :, :, width:] == -1).all())
    log(f"transform: {tuple(x.shape)} {x.dtype}, pad + normalise only: {no_resize}, padding = -1: {pad_black}")
    if x.dtype != in_dtype(ctx):  # fp16 VAE: numz's bfloat16 frames, cast (exactly) to float16
        x = x.to(in_dtype(ctx))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    lat = runner.vae_encode([x])[0]
    torch.cuda.synchronize()
    enc_s = time.perf_counter() - t0
    lat = lat.contiguous().cpu()
    keep = {}
    for j in ints(args.keep):
        f0, f1 = frames_of(j)
        if f1 <= args.frames:
            keep[str(j)] = torch.from_numpy(frames[f0:f1].copy())
    meta = {"input": os.path.basename(args.input), "start": args.start, "frames": args.frames,
            "width": width, "height": height, "padded": list(x.shape[-2:]), "latent_shape": list(lat.shape),
            "no_resize": no_resize, "pad_black": pad_black, "encode_s": round(enc_s, 3), "env": env_info(runner)}
    torch.save({"latent": lat, "meta": meta, "frames": keep}, args.out)
    log(f"latents {tuple(lat.shape)} {lat.dtype}, encode {enc_s:.1f} s -> {args.out}")
    print(json.dumps(meta, indent=1))


# ---------------------------------------------------------------- run

def save_snapshot(caches, path):
    """GPU copy -> CPU -> file (fsynced, then dropped from the page cache) -> CPU -> GPU."""
    import torch
    t = {}
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    cpu = {n: v.to("cpu") for n, v in caches.items()}
    t["d2h_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    with open(path, "wb") as f:
        torch.save(cpu, f)
        f.flush()
        os.fsync(f.fileno())
        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    t["save_s"] = time.perf_counter() - t0
    t["file_bytes"] = os.path.getsize(path)
    del cpu
    t0 = time.perf_counter()
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    t["load_s"] = time.perf_counter() - t0
    dev = next(iter(caches.values())).device
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gpu = {n: v.to(dev) for n, v in loaded.items()}
    torch.cuda.synchronize()
    t["h2d_s"] = time.perf_counter() - t0
    t["layout_kept"] = all(gpu[n].stride() == v.stride() and gpu[n].dtype == v.dtype for n, v in caches.items())
    os.remove(path)
    return gpu, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in t.items()}


def cmd_run(args):
    setup_numz()
    import torch
    tiled = tuple(int(v) for v in args.tiled.split(":")) if args.tiled else None
    if args.snapshot is not None and tiled:
        sys.exit("--snapshot needs an untiled decode")
    data = torch.load(args.latents, map_location="cpu", weights_only=True)
    lat, meta = data["latent"], data["meta"]
    T = lat.shape[0]
    log(f"latents {tuple(lat.shape)} from {meta['input']} {meta['width']}x{meta['height']} "
        f"(padded {meta['padded']}), tiled={tiled}")
    runner, ctx = build_runner(args, tiled)
    hook = SliceHook(runner.vae)
    res = {"latents": os.path.basename(args.latents), "meta": meta, "tiled": list(tiled) if tiled else None,
           "env": env_info(runner), "k": args.k, "reference": {}, "profile": [], "resume": [], "snapshot": None}
    hp, wp = meta["padded"]
    mpx = hp * wp / 1e6

    def write():
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)

    # one-pass reference(s)
    ref = None
    if args.ref_load:
        ref = torch.load(args.ref_load, map_location="cpu", weights_only=True).to(ctx["vae_device"])
        res["reference"]["loaded"] = os.path.basename(args.ref_load)
        log(f"reference loaded {tuple(ref.shape)}")
    runs = []
    snap = None
    for i in range(args.ref_runs):
        snap_at = args.snapshot - 1 if (args.snapshot is not None and i == 0) else None
        hook.reset(snapshot_at=snap_at)
        torch.cuda.reset_peak_memory_stats()
        out, dt = decode(runner, ctx, lat)
        run = {"run": i, "wall_s": round(dt, 3), "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
               "slices": slice_stats(hook.slices), "frames": int(out.shape[1])}
        if ref is None:
            ref = out
            run["identical_to_ref"] = None
        else:
            m = metrics(ref, out)
            run["identical_to_ref"] = m["identical"]
            if not m["identical"]:
                run["diff"] = m
        runs.append(run)
        log(f"one-pass {i}: {dt:.1f} s, {run['frames']} frames, ACTIVE slice "
            f"{run['slices']['active_mean_s']} s, identical to reference: {run['identical_to_ref']}")
        if snap_at is not None:
            snap = hook.snapshot
            if snap is None:
                sys.exit(f"no snapshot taken: latent {args.snapshot} not reached")
            nbytes = sum(v.numel() * v.element_size() for v in snap.values())
            res["snapshot"] = {"s": args.snapshot, "n_tensors": len(snap), "bytes": nbytes,
                               "gib": round(nbytes / 2**30, 3), "gib_per_mpx": round(nbytes / 2**30 / mpx, 3),
                               "gpu_copy_s": round(hook.snapshot_s, 3)}
            log(f"snapshot before latent {args.snapshot}: {len(snap)} tensors, {nbytes / 2**30:.2f} GiB "
                f"({nbytes / 2**30 / mpx:.2f} GiB per output Mpx), GPU copy {hook.snapshot_s:.2f} s")
        if i == 0 and args.ref_save and not args.ref_load:
            t0 = time.perf_counter()
            torch.save(out.cpu(), args.ref_save)
            log(f"reference saved -> {args.ref_save} ({time.perf_counter() - t0:.1f} s)")
        if out is not ref:
            del out
        res["reference"]["runs"] = runs
        write()
    if ref is None:
        sys.exit("no reference: give --ref-runs >= 1 or --ref-load")
    hook.reset()
    res["reference"]["recon_psnr"] = recon_psnr(ref, data.get("frames", {}), meta["height"], meta["width"])
    res["reference"]["per_latent_s"] = [round(x["wall_s"] / T, 3) for x in runs]
    log(f"reconstruction PSNR vs input (kept latents): {res['reference']['recon_psnr']}")
    write()

    # snapshot resume
    if args.snapshot is not None:
        if snap is None:
            sys.exit("--snapshot needs --ref-runs >= 1")
        s, e = args.snapshot, min(args.snapshot + args.k, T)
        out, dt = decode_from_caches(runner, ctx, hook, lat, s, e, snap)
        rows = compare_latents(ref, out, s, range(s, e), active_from=s)
        res["snapshot"].update({"gpu_resume": summarize(rows), "gpu_resume_rows": rows, "resume_s": round(dt, 3),
                                "resume_slices": slice_stats(hook.slices)})
        log(f"snapshot (GPU copy) resume at {s}: {summarize(rows)}")
        del out
        hook.reset()
        path = os.path.join(args.snapshot_dir or os.path.dirname(os.path.abspath(args.out)),
                            f"snapshot_{os.getpid()}.pt")
        loaded, t = save_snapshot(snap, path)
        snap = None
        out, dt = decode_from_caches(runner, ctx, hook, lat, s, e, loaded)
        rows = compare_latents(ref, out, s, range(s, e), active_from=s)
        res["snapshot"].update({"file": t, "file_resume": summarize(rows), "file_resume_rows": rows})
        log(f"snapshot (file round trip) resume at {s}: {summarize(rows)}, {t}")
        del out, loaded
        hook.reset()
        write()

    # profiles: one fresh decode from latent a to the end
    for a in ints(args.profile):
        hook.reset()
        out, dt = decode(runner, ctx, lat[a:])
        rows = compare_latents(ref, out, a, range(a + 1, T))
        clean = [r["distance"] for r in rows if r["identical"]]
        first = next((r["distance"] for r in rows if all(x["identical"] for x in rows if x["distance"] >= r["distance"])), None)
        res["profile"].append({"a": a, "wall_s": round(dt, 3), "slices": slice_stats(hook.slices),
                               "first_identical_distance": first, "n_identical": len(clean), "rows": rows})
        log(f"profile from {a}: {dt:.1f} s; identical from distance {first} on")
        for r in rows:
            log(f"  d={r['distance']:3d} latent {r['latent']:3d} identical={r['identical']} "
                f"max_abs={r['max_abs']:.6g} psnr={r['psnr']} q16={r['q16']} q10={r['q10']}")
        del out
        write()

    # warm-up resumes
    for s in ints(args.resume):
        e = min(s + args.k, T)
        for w in ints(args.warmup):
            a = s - w
            if a < 1:
                log(f"skip s={s} w={w}: the decode would start at latent {a}")
                continue
            hook.reset()
            out, dt = decode(runner, ctx, lat[a:e])
            rows = compare_latents(ref, out, a, range(s, e))
            summ = summarize(rows)
            res["resume"].append({"s": s, "w": w, "a": a, "e": e, "wall_s": round(dt, 3),
                                  "slices": slice_stats(hook.slices), "summary": summ, "rows": rows})
            log(f"s={s} w={w}: {dt:.1f} s, identical={summ['identical']} max_abs={summ['max_abs']:.6g} "
                f"psnr_min={summ['psnr_min']} q16={summ['q16']} q10={summ['q10']} "
                f"per latent: {[(r['latent'], r['identical']) for r in rows]}")
            del out
            write()
    write()
    log(f"done -> {args.out}")


# ---------------------------------------------------------------- dtype A/B (bf16 vs fp16 VAE)

class GpuSampler:
    """NVML through ctypes (as bench.py): SM clock, power, temperature and clock event reasons of
    GPU 0 every `interval` s, in a thread. Creates no CUDA context."""

    def __init__(self, interval=0.1):
        import ctypes
        import threading
        self.samples, self.interval, self.lib, self.power_limit_w = [], interval, None, None
        try:
            lib = ctypes.CDLL("libnvidia-ml.so.1")
            h = ctypes.c_void_p()
            if lib.nvmlInit_v2() != 0 or lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(h)) != 0:
                raise OSError("NVML init failed")
            self.lib, self.h = lib, h
            self.reasons = (getattr(lib, "nvmlDeviceGetCurrentClocksEventReasons", None)
                            or getattr(lib, "nvmlDeviceGetCurrentClocksThrottleReasons", None))
            v = ctypes.c_uint()
            if lib.nvmlDeviceGetEnforcedPowerLimit(h, ctypes.byref(v)) == 0:
                self.power_limit_w = v.value / 1000
        except (OSError, AttributeError) as e:
            log(f"NVML unavailable: {e}")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def read(self):
        import ctypes
        clk, pw, tmp, rs = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint(), ctypes.c_ulonglong()
        self.lib.nvmlDeviceGetClockInfo(self.h, 1, ctypes.byref(clk))  # NVML_CLOCK_SM
        self.lib.nvmlDeviceGetPowerUsage(self.h, ctypes.byref(pw))  # mW
        self.lib.nvmlDeviceGetTemperature(self.h, 0, ctypes.byref(tmp))  # NVML_TEMPERATURE_GPU
        r = rs.value if self.reasons and self.reasons(self.h, ctypes.byref(rs)) == 0 else 0
        return time.time(), clk.value, pw.value / 1000, tmp.value, r

    def _loop(self):
        while not self._stop.is_set():
            self.samples.append(self.read())
            self._stop.wait(self.interval)

    def start(self):
        if self.lib:
            self._thread.start()

    def stop(self):
        self._stop.set()
        if self.lib:
            self._thread.join(timeout=5)

    def stats(self, t0, t1):
        """Over [t0, t1]: SM clock (MHz), power (W), temperature (C), and the share of samples whose
        clock event reasons include the software power cap (0x4) or a thermal slowdown (0x20, 0x40)."""
        s = [x for x in self.samples if t0 <= x[0] <= t1]
        if not s:
            return None
        clk, pw = [x[1] for x in s], [x[2] for x in s]
        return {"n": len(s), "sm_mhz_mean": round(sum(clk) / len(clk)), "sm_mhz_min": min(clk),
                "sm_mhz_max": max(clk), "power_w_mean": round(sum(pw) / len(pw), 1), "power_w_max": round(max(pw), 1),
                "temp_c_max": max(x[3] for x in s),
                "power_cap_frac": round(sum(1 for x in s if x[4] & 0x4) / len(s), 3),
                "thermal_frac": round(sum(1 for x in s if x[4] & 0x60) / len(s), 3)}


def max_abs_diff(a, b, chunk=8):
    """max and mean |a - b| in float32, along dim 1 in chunks (bounded memory)."""
    mx, tot = 0.0, 0.0
    for i in range(0, a.shape[1], chunk):
        d = (a[:, i:i + chunk].float() - b[:, i:i + chunk].float()).abs()
        mx, tot = max(mx, float(d.max())), tot + float(d.sum())
    return mx, tot / a.numel()


def cmd_dtype_ab(args):
    """Back to back, alternating bf16 / fp16 VAEs in one process: untiled decode, tiled decode and
    encode, through numz's vae_decode / vae_encode, with time, peak, clocks and power per run."""
    t_start = time.time()
    setup_numz()
    import numpy as np
    import torch
    from src.core.generation_utils import prepare_video_transforms
    from src.optimization.memory_manager import manage_tensor
    data = torch.load(args.latents, map_location="cpu", weights_only=True)
    meta = data["meta"]
    lat = data["latent"][:args.n_latents].contiguous()
    n_frames = 4 * (lat.shape[0] - 1) + 1
    inexact = int((lat.to(torch.float16).float() != lat.float()).sum())
    phases = [p for p in args.phases.split(",") if p]
    tile, overlap = (int(v) for v in args.tiled.split(":"))
    log(f"{lat.shape[0]} latents {tuple(lat.shape)} ({n_frames} frames, {meta['width']}x{meta['height']} padded "
        f"{meta['padded']}); bf16 latent values not exact in float16: {inexact}; phases {phases}")
    frames = None
    if "encode" in phases:
        if not args.input:
            sys.exit("the encode phase needs --input (the latents' source video)")
        frames = read_frames(args.input, args.start, n_frames, meta["width"], meta["height"])
    sampler = GpuSampler()
    sampler.start()
    runners = {dt: build_runner(args, None, dt) for dt in ("bf16", "fp16")}
    hooks = {dt: SliceHook(r.vae) for dt, (r, _) in runners.items()}
    wdt = {dt: str(next(r.vae.parameters()).dtype) for dt, (r, _) in runners.items()}
    log(f"VAEs: {wdt}, distinct objects: {runners['bf16'][0].vae is not runners['fp16'][0].vae}")
    res = {"latents": os.path.basename(args.latents), "meta": meta, "n_latents": lat.shape[0], "frames": n_frames,
           "latent_values_inexact_fp16": inexact, "tiled": [tile, overlap], "pairs": args.pairs,
           "vae_weights": wdt, "env": env_info(runners["bf16"][0]), "power_limit_w": sampler.power_limit_w,
           "runs": [], "diff": {}, "skipped": []}

    def write():
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)

    x_bf16 = None
    if frames is not None:  # numz's input preparation (bfloat16), once; the fp16 VAE gets it cast exactly
        _, ctx_b = runners["bf16"]
        video = manage_tensor(tensor=torch.from_numpy(frames.astype(np.float32) / 255.0).permute(0, 3, 1, 2),
                              target_device=ctx_b["vae_device"], dtype=ctx_b["compute_dtype"])
        x_bf16 = prepare_video_transforms(min(meta["width"], meta["height"]), 0, None)(video)
        del video
    budget = args.budget_min * 60
    stop, last = False, 0.0  # last: the previous run's duration, to stop before overrunning the budget
    for phase in phases:
        if stop:
            res["skipped"].append(phase)
            continue
        firsts = {}
        for i, dt in enumerate(["bf16", "fp16"] * args.pairs):
            if time.time() - t_start + last > budget:
                log(f"time budget ({args.budget_min} min) reached: {phase} stops at run {i}")
                res["skipped"].append(f"{phase} from run {i}")
                stop = True
                break
            runner, ctx = runners[dt]
            runner.decode_tiled = phase == "tiled"
            runner.decode_tile_size, runner.decode_tile_overlap = (tile, tile), (overlap, overlap)
            hooks[dt].reset()
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            if phase == "encode":
                x = x_bf16 if x_bf16.dtype == in_dtype(ctx) else x_bf16.to(in_dtype(ctx))
                torch.cuda.synchronize()
                tp = time.perf_counter()
                out = runner.vae_encode([x])[0]
                torch.cuda.synchronize()
                sec = time.perf_counter() - tp
                del x
            else:
                out, sec = decode(runner, ctx, lat)
            t1 = time.time()
            last = t1 - t0
            peak = torch.cuda.max_memory_allocated()
            run = {"phase": phase, "dtype": dt, "i": i, "s": round(sec, 3), "peak_gib": round(peak / 2**30, 2),
                   "peak_above_gib": round((peak - base) / 2**30, 2), "out_dtype": str(out.dtype),
                   "out_shape": list(out.shape), "nonfinite": int(torch.isfinite(out).logical_not_().sum()),
                   "absmax": float(out.abs().max()), "gpu": sampler.stats(t0, t1)}
            if phase != "encode":
                run["slices"] = slice_stats(hooks[dt].slices)
            if dt in firsts:
                run["identical_to_first"] = bool(torch.equal(firsts[dt], out))
            else:
                firsts[dt] = out
            res["runs"].append(run)
            g = run["gpu"] or {}
            log(f"{phase} {dt} #{i}: {sec:.2f} s, peak {run['peak_gib']} GiB (+{run['peak_above_gib']}), "
                f"non-finite {run['nonfinite']}, |max| {run['absmax']:.4g}, SM {g.get('sm_mhz_mean')} MHz "
                f"({g.get('sm_mhz_min')}-{g.get('sm_mhz_max')}), {g.get('power_w_mean')} W (max "
                f"{g.get('power_w_max')}), power cap {g.get('power_cap_frac')}, "
                f"ACTIVE slice {run.get('slices', {}).get('active_mean_s')}, same as first: "
                f"{run.get('identical_to_first')}")
            if out is not firsts.get(dt):
                del out
            write()
        if len(firsts) == 2:
            mx, mean = max_abs_diff(firsts["bf16"], firsts["fp16"])
            res["diff"][phase] = {"max_abs": mx, "mean_abs": mean}
            log(f"{phase}: fp16 vs bf16 output max |diff| {mx:.6g}, mean |diff| {mean:.6g}")
        del firsts
        write()
    sampler.stop()
    for phase in phases:
        for dt in ("bf16", "fp16"):
            rr = [r for r in res["runs"] if r["phase"] == phase and r["dtype"] == dt]
            steady = [r["s"] for r in rr[1:]]
            if steady:
                res.setdefault("steady", {}).setdefault(phase, {})[dt] = round(sum(steady) / len(steady), 3)
    write()
    log(f"steady (first run of each dtype excluded): {res.get('steady')}; total {time.time() - t_start:.0f} s "
        f"-> {args.out}")


# ---------------------------------------------------------------- tables

def fmt_psnr(p):
    return "∞" if p is None else f"{p:.1f}"


def yn(v):
    return "yes" if v else "no"


def cmd_table(args):
    for path in args.results:
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        m = r["meta"]
        label = f"{m['width']}x{m['height']}" + (f" tiled {r['tiled'][0]}:{r['tiled'][1]}" if r["tiled"] else "")
        print(f"\n### {os.path.basename(path)}: {label}, {m['latent_shape'][0]} latents\n")
        runs = r["reference"].get("runs", [])
        if runs or r["reference"].get("loaded"):
            print("| One-pass run | Wall (s) | ACTIVE slice mean (s) | INIT slice (s) | Peak (GiB) | Identical to reference |")
            print("|---|---|---|---|---|---|")
            if r["reference"].get("loaded"):
                print(f"| loaded: {r['reference']['loaded']} | – | – | – | – | reference |")
            for x in runs:
                ident = "reference" if x["identical_to_ref"] is None else yn(x["identical_to_ref"])
                print(f"| {x['run']} | {x['wall_s']:.1f} | {x['slices']['active_mean_s']} | {x['slices']['init_s']} | "
                      f"{x['peak_gib']} | {ident} |")
        if r["resume"]:
            print("\n| s | w | Latents decoded | Wall (s) | bf16 identical | Max abs diff | PSNR min (dB) | "
                  "16-bit identical | 10-bit identical | Identical per latent |")
            print("|---|---|---|---|---|---|---|---|---|---|")
            for x in r["resume"]:
                s = x["summary"]
                per = " ".join(f"{q['latent']}:{'=' if q['identical'] else '≠'}" for q in x["rows"])
                print(f"| {x['s']} | {x['w']} | {x['e'] - x['a']} | {x['wall_s']:.1f} | {yn(s['identical'])} | "
                      f"{s['max_abs']:.3g} | {fmt_psnr(s['psnr_min'])} | {yn(s['q16'])} | {yn(s['q10'])} | {per} |")
        for p in r["profile"]:
            print(f"\nProfile from latent {p['a']}: identical from distance {p['first_identical_distance']} on "
                  f"({p['wall_s']:.1f} s)\n")
            print("| Distance | Latent | bf16 identical | Max abs diff | PSNR (dB) | 16-bit codes ≠ | 10-bit codes ≠ |")
            print("|---|---|---|---|---|---|---|")
            for q in p["rows"]:
                print(f"| {q['distance']} | {q['latent']} | {yn(q['identical'])} | {q['max_abs']:.3g} | "
                      f"{fmt_psnr(q['psnr'])} | {q['q16_ndiff']} | {q['q10_ndiff']} |")
        sn = r.get("snapshot")
        if sn:
            print(f"\nSnapshot before latent {sn['s']}: {sn['n_tensors']} tensors, {sn['bytes']} bytes "
                  f"({sn['gib']} GiB, {sn['gib_per_mpx']} GiB per output Mpx), GPU copy {sn['gpu_copy_s']} s")
            if "gpu_resume" in sn:
                print(f"- resume from the GPU copy: bf16 identical {yn(sn['gpu_resume']['identical'])}, "
                      f"{sn['resume_s']} s for {len(sn['gpu_resume_rows'])} latents")
            if "file" in sn:
                t = sn["file"]
                print(f"- resume after the file round trip: bf16 identical {yn(sn['file_resume']['identical'])}; "
                      f"GPU→CPU {t['d2h_s']} s, save+fsync {t['save_s']} s ({t['file_bytes']} bytes), "
                      f"load (cold) {t['load_s']} s, CPU→GPU {t['h2d_s']} s, layout kept {t['layout_kept']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--model-dir", required=True)
        p.add_argument("--dit-model", default="seedvr2_ema_7b_fp16.safetensors")
        p.add_argument("--vae-dtype", choices=("bf16", "fp16"), default="bf16",
                       help="bf16: numz's (fp16 file converted on load); fp16: the file's own weights, "
                            "float16 inputs, no autocast")

    p = sub.add_parser("encode")
    common(p)
    p.add_argument("--input", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--frames", type=int, default=201)
    p.add_argument("--size", required=True, help="WxH fed to the VAE (no resize in numz)")
    p.add_argument("--keep", default="41,45", help="latents whose input frames are kept")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_encode)

    p = sub.add_parser("run")
    common(p)
    p.add_argument("--latents", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ref-runs", type=int, default=1)
    p.add_argument("--ref-save")
    p.add_argument("--ref-load")
    p.add_argument("--tiled", help="TILE:OVERLAP in output pixels")
    p.add_argument("--resume", default="")
    p.add_argument("--warmup", default="")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--profile", default="")
    p.add_argument("--snapshot", type=int)
    p.add_argument("--snapshot-dir")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("dtype-ab")
    common(p)
    p.add_argument("--latents", required=True)
    p.add_argument("--input", help="the source video of the latents (encode phase)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--n-latents", type=int, default=9, help="first N latents (4N - 3 frames)")
    p.add_argument("--phases", default="decode,tiled,encode")
    p.add_argument("--pairs", type=int, default=3, help="bf16, fp16 alternations per phase")
    p.add_argument("--tiled", default="1024:128", help="tiled phase: TILE:OVERLAP (the CLI's defaults)")
    p.add_argument("--budget-min", type=float, default=19, help="stop before a run that would end later")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_dtype_ab)

    p = sub.add_parser("table")
    p.add_argument("results", nargs="+")
    p.set_defaults(func=cmd_table)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

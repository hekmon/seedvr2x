#!/usr/bin/env python3
"""DiT probe for SeedVR2: torch peak and GPU time of one DiT forward, per output size and window length.

One forward is what inference_cli.py's Phase 2 (upscale_all_batches) does for one batch, on
random latents of a given output size and L latent frames: the latent goes to the GPU in the
compute dtype, the seed is set, the noise and the augmentation noise are drawn, the "sr"
condition is built, then runner.inference runs one step (cfg 1.0) under bf16 autocast when the
DiT weights are not in the compute dtype, with the text embeddings of pos_emb.pt / neg_emb.pt;
the output latent goes back to the CPU. The runner, the DiT and the VAE come from numz's own
code, as the CLI builds them: setup_generation_context + prepare_runner (CLI defaults, no
BlockSwap, no compile, tensor offload to the CPU), materialize_model for the VAE (resident on
the GPU during Phase 2, as in the CLI: --no-vae leaves it out) and for the DiT, the text
embeddings, configure_diffusion with the CLI's one-step settings.

Per forward: torch peak allocated (reset before) and allocated before; GPU time of
runner.inference (CUDA events) and of the whole batch body; device memory used (NVML, sampled
every 0.1 s, and mem_get_info after); SM clock and board power during the forward (NVML);
allocator counters. Every point runs --repeats times (the first may warm up: the steady time is
the median of the others). A reference point (--ref, default 1080:6) is measured at the start,
after every --ref-every points and at the end: `table` divides each time by the reference time
interpolated at that moment (the GPU is power-capped and its speed drifts within and between
sessions), and reports raw and normalised ms per token.

Tokens = L x (H/16) x (W/16) on the padded output (latents H/8 x W/8, patch 1x2x2); the probe
checks it on the DiT's patch-in output. Attention windows are computed with numz's own window
functions and the model config (7B: window (4,3,3), methods 720pwin / 720pswin alternating per
block): at most 15 x 27 tokens in space, ceil(min(L, 30) / 4) latents in time.

A sweep stops at its first OOM (a longer window would fail too); --refine then bisects between
the last passing and the first failing L (--refine-repeats forwards each). Any other CUDA error
ends the run (the context may be unusable); results are written after every point.

Subcommands:
  run    --model-dir DIR --sweep SIZE:L1,L2,... [--sweep ...] [--ref SIZE:L] [--ref-every N]
         [--repeats N] [--ref-repeats N] [--refine] [--refine-repeats N] [--out RESULTS.json] [--dit-model F]
         [--attention-mode M] [--seed S] [--no-vae]
  table  RESULTS.json [...] [--merge]   Markdown tables against the per-token model, and fits; --merge
         makes one table of several files (a sweep split into one invocation per point, the GPU
         lock released in between): points in time order, normalised by every file's references
  plan   --sweep ... [--ref ...] [--ref-every N] [--repeats N] [--room GIB]
         the points, their tokens, model peak and time, and the sweep's GPU time (no GPU used)

SIZE = 720 | 1080 | 1440 | 2160 (16:9 output) or WxH, padded up to multiples of 16 as numz pads
(1080 -> 1920x1088). Model (vram.md, 7B fp16, flash_attn_2): peak = 16.05 GiB + 128.5 KiB x
tokens (VAE resident; 0.47 GiB less with --no-vae), time = 0.231 ms x tokens (a fast session).

Environment: SEEDVR2_DIR (the numz checkout, default: cwd). The allocator default of the CLI
(PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync) is applied before torch is imported.

Usage (cwd = the SeedVR2 checkout, its venv's python; GPU commands under the shared GPU lock):
  python dit_probe.py plan --sweep 1080:1,2,3,6,12,21 --sweep 2160:1,2,4
  python dit_probe.py run --model-dir MODELS --attention-mode flash_attn_2 \\
      --sweep 1080:1,6 --sweep 2160:1,2 --ref 1080:6 --out smoke.json
  python dit_probe.py table smoke.json
"""
import argparse
import ctypes
import gc
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
import warnings

GIB = 1024 ** 3
# vram.md, 7B fp16, flash_attn_2, VAE resident: peak = const + slope x tokens; time per token
MODEL = {"const_gib": 16.05, "kib_per_token": 128.5, "ms_per_token": 0.231, "vae_gib": 0.47}
SIZES = {"720": (1280, 720), "1080": (1920, 1080), "1440": (2560, 1440), "2160": (3840, 2160), "4k": (3840, 2160)}
ALLOC_RETRY = ("recovered from an allocation failure", "memory mapping failed")


def log(msg):
    print(f"[dit_probe {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pad16(v):
    return -(-v // 16) * 16


def parse_size(text):
    t = text.strip().lower()
    w, h = SIZES[t] if t in SIZES else (int(v) for v in t.split("x"))
    return pad16(w), pad16(h)


def parse_points(text):
    """'1080:1,2,3' -> (w, h, [1, 2, 3]), w and h padded to multiples of 16."""
    size, sep, ls = text.partition(":")
    if not sep:
        sys.exit(f"dit_probe: expected SIZE:L1,L2,..., got {text!r}")
    w, h = parse_size(size)
    return w, h, [int(v) for v in ls.split(",") if v.strip()]


def tokens_of(w, h, L):
    return L * (h // 16) * (w // 16)


def model_peak_gib(tokens, vae=True):
    return MODEL["const_gib"] - (0.0 if vae else MODEL["vae_gib"]) + MODEL["kib_per_token"] * tokens / 2 ** 20


def model_time_s(tokens):
    return MODEL["ms_per_token"] * tokens / 1000


def numz_root():
    return os.path.abspath(os.environ.get("SEEDVR2_DIR", os.getcwd()))


# ---------------------------------------------------------------- windows (numz's own functions)

def load_window_module(root):
    """numz's dit_7b/window.py loaded by path: importing the package would create a CUDA context."""
    import importlib.util
    path = os.path.join(root, "src", "models", "dit_7b", "window.py")
    spec = importlib.util.spec_from_file_location("dit_probe_numz_window", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def window_info(win, w, h, L, window=(4, 3, 3), methods=("720pwin_by_size_bysize", "720pswin_by_size_bysize")):
    grid = (L, h // 16, w // 16)
    out = {"grid": list(grid), "window": list(window)}
    for m in methods:
        ws = win.get_window_op(m)(grid, tuple(window))
        n = lambda s: s.stop - s.start  # noqa: E731
        out[m] = {"n": len(ws), "max_tokens": max(n(a) * n(b) * n(c) for a, b, c in ws),
                  "t": max(n(a) for a, _, _ in ws), "h": max(n(b) for _, b, _ in ws),
                  "w": max(n(c) for _, _, c in ws)}
    return out


# ---------------------------------------------------------------- NVML (memory, SM clock, power)

class _MemV2(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint), ("total", ctypes.c_ulonglong), ("reserved", ctypes.c_ulonglong),
                ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


class _MemV1(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


class Nvml:
    """Device memory used (as nvidia-smi), SM clock and power, sampled in a thread during a forward."""

    def __init__(self, index=0, interval=0.1):
        self.interval, self.ok = interval, False
        self.samples, self._stop, self._thread = [], None, None
        self._clk, self._pw = ctypes.c_uint(), ctypes.c_uint()
        try:
            lib = ctypes.CDLL("libnvidia-ml.so.1")
            if lib.nvmlInit_v2() != 0:
                raise OSError("nvmlInit_v2 failed")
            h = ctypes.c_void_p()
            if lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(h)) != 0:
                raise OSError(f"no NVML device {index}")
            m2 = _MemV2(version=ctypes.sizeof(_MemV2) | (2 << 24))
            if lib.nvmlDeviceGetMemoryInfo_v2(h, ctypes.byref(m2)) == 0:
                self._mem = lambda: (lib.nvmlDeviceGetMemoryInfo_v2(h, ctypes.byref(m2)), m2.used)[1]
            else:
                m1 = _MemV1()
                self._mem = lambda: (lib.nvmlDeviceGetMemoryInfo(h, ctypes.byref(m1)), m1.used)[1]
            self.lib, self.h, self.ok = lib, h, True
        except (OSError, AttributeError) as e:
            log(f"NVML unavailable ({e}): no device memory, clock or power samples")

    def read(self):
        sm = self._clk.value if self.lib.nvmlDeviceGetClockInfo(self.h, 1, ctypes.byref(self._clk)) == 0 else None
        pw = self._pw.value / 1000 if self.lib.nvmlDeviceGetPowerUsage(self.h, ctypes.byref(self._pw)) == 0 else None
        return self._mem(), sm, pw  # NVML_CLOCK_SM = 1; power in mW

    def start(self):
        self.samples = []
        if not self.ok:
            return
        self._stop = threading.Event()

        def loop():
            while not self._stop.is_set():
                self.samples.append(self.read())
                self._stop.wait(self.interval)
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        if not self.ok or self._thread is None:
            return None
        self._stop.set()
        self._thread.join(timeout=5)
        self._thread = None
        s = self.samples + [self.read()]
        sm = [x[1] for x in s if x[1]]
        pw = [x[2] for x in s if x[2]]
        return {"n": len(s), "used_max_gib": round(max(x[0] for x in s) / GIB, 3),
                "sm_mhz_mean": round(statistics.mean(sm)) if sm else None, "sm_mhz_min": min(sm) if sm else None,
                "power_w_mean": round(statistics.mean(pw), 1) if pw else None}


class RetryCounter:
    """Counts the allocator's recovered-failure warnings that reach Python's warnings module."""

    def __init__(self):
        self.n = 0
        self._orig = warnings.showwarning
        for text in ALLOC_RETRY:
            warnings.filterwarnings("always", message=".*" + text)
        warnings.showwarning = self._show

    def _show(self, message, category, filename, lineno, file=None, line=None):
        if any(t in str(message) for t in ALLOC_RETRY):
            self.n += 1
        return self._orig(message, category, filename, lineno, file, line)


# ---------------------------------------------------------------- numz setup

def setup_numz():
    """Make numz importable as inference_cli.py does (allocator default set before torch is imported)."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
    root = numz_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def build(args):
    """The CLI's runner: VAE materialized as Phase 1 leaves it, text embeddings, then Phase 2's setup."""
    from src.utils.debug import Debug
    from src.utils.model_registry import DEFAULT_VAE
    from src.core.generation_utils import (setup_generation_context, prepare_runner, load_text_embeddings,
                                           script_directory)
    from src.core.model_loader import materialize_model
    from src.optimization.memory_manager import manage_model_device

    debug = Debug(enabled=False)
    ctx = setup_generation_context(dit_device="cuda:0", vae_device="cuda:0", dit_offload_device=None,
                                   vae_offload_device=None, tensor_offload_device="cpu", debug=debug)
    runner, _ = prepare_runner(
        dit_model=args.dit_model, vae_model=DEFAULT_VAE, model_dir=args.model_dir, debug=debug, ctx=ctx,
        dit_cache=False, vae_cache=False, dit_id=None, vae_id=None,
        block_swap_config={"blocks_to_swap": 0, "swap_io_components": False, "offload_device": None},
        encode_tiled=False, encode_tile_size=(1024, 1024), encode_tile_overlap=(128, 128),
        decode_tiled=False, decode_tile_size=(1024, 1024), decode_tile_overlap=(128, 128),
        tile_debug="false", attention_mode=args.attention_mode, torch_compile_args_dit=None,
        torch_compile_args_vae=None)
    if not args.no_vae:
        materialize_model(runner, "vae", ctx["vae_device"], runner.config, debug)
    ctx["text_embeds"] = load_text_embeddings(script_directory, ctx["dit_device"], ctx["compute_dtype"], debug)
    # upscale_all_batches: one-step distilled model settings, then the DiT on the GPU
    runner.config.diffusion.cfg.scale = 1.0
    runner.config.diffusion.cfg.rescale = 0.0
    runner.config.diffusion.timesteps.sampling.steps = 1
    runner.configure_diffusion(device=ctx["dit_device"], dtype=ctx["compute_dtype"])
    materialize_model(runner, "dit", ctx["dit_device"], runner.config, debug)
    manage_model_device(model=runner.dit, target_device=ctx["dit_device"], model_name="DiT", debug=debug,
                        runner=runner)
    return runner, ctx


def inner_dit(runner):
    return runner.dit.dit_model if hasattr(runner.dit, "dit_model") else runner.dit


class PatchHook:
    """Records the token count and the patch grid on the DiT's patch-in output (vid [tokens, dim], vid_shape)."""

    def __init__(self, runner):
        self.tokens = self.grid = None
        inner_dit(runner).vid_in.register_forward_hook(self._hook)

    def _hook(self, module, inputs, out):
        vid, shape = out
        self.tokens, self.grid = int(vid.shape[0]), [int(v) for v in shape[0].tolist()]


def env_info(runner, ctx, root):
    import torch
    dit = inner_dit(runner)
    p = next(dit.parameters())
    modes = sorted({getattr(m, "attention_mode", None) for m in dit.modules()
                    if type(m).__name__ == "FlashAttentionVarlen"} - {None})
    try:
        import flash_attn
        fa = flash_attn.__version__
    except Exception:  # noqa: BLE001
        fa = None
    try:
        rev = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        rev = None
    vae = getattr(runner, "vae", None)
    vae_on_gpu = vae is not None and next(vae.parameters()).device.type == "cuda"
    return {"torch": str(torch.__version__), "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0), "flash_attn": fa, "numz_rev": rev,
            "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), "compute_dtype": str(ctx["compute_dtype"]),
            "dit_param_dtype": str(p.dtype), "autocast": p.dtype != ctx["compute_dtype"],
            "attention_mode_effective": modes, "dit_class": type(dit).__name__,
            "dit_param_gib": round(sum(x.numel() * x.element_size() for x in dit.parameters()) / GIB, 3),
            "dit_buffer_gib": round(sum(x.numel() * x.element_size() for x in dit.buffers()) / GIB, 3),
            "vae_on_gpu": vae_on_gpu,
            "vae_param_gib": round(sum(x.numel() * x.element_size() for x in vae.parameters()) / GIB, 3)
            if vae_on_gpu else 0.0,
            "text_pos": list(ctx["text_embeds"]["texts_pos"][0].shape),
            "text_neg": list(ctx["text_embeds"]["texts_neg"][0].shape)}


# ---------------------------------------------------------------- one forward

def is_oom(e):
    import torch
    return isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()


def one_forward(runner, ctx, lat_cpu, seed, nvml, retries, hook):
    """upscale_all_batches' body for one batch; returns the measurements."""
    import torch
    from src.common.seed import set_seed
    from src.optimization.memory_manager import manage_tensor

    r = {}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    st0 = torch.cuda.memory_stats()
    r["alloc_before_gib"] = torch.cuda.memory_allocated() / GIB
    n0 = retries.n
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    hook.tokens = hook.grid = None
    nvml.start()
    t0 = time.perf_counter()
    try:
        ev[0].record()
        latent = manage_tensor(tensor=lat_cpu, target_device=ctx["dit_device"], tensor_name="latent",
                               dtype=ctx["compute_dtype"])
        set_seed(seed)
        base_noise = torch.randn_like(latent, dtype=ctx["compute_dtype"])
        noises = [base_noise]
        aug_noises = [base_noise * 0.1 + torch.randn_like(base_noise) * 0.05]  # drawn by the CLI, unused at scale 0
        condition = runner.get_condition(noises[0], task="sr", latent_blur=latent)  # latent_noise_scale 0
        conditions = [condition]
        dit_dtype = next(inner_dit(runner).parameters()).dtype
        ev[1].record()
        with torch.no_grad():
            if dit_dtype != ctx["compute_dtype"] and ctx["dit_device"].type != "mps":
                with torch.autocast(ctx["dit_device"].type, ctx["compute_dtype"], enabled=True):
                    out = runner.inference(noises=noises, conditions=conditions, **ctx["text_embeds"])
            else:
                out = runner.inference(noises=noises, conditions=conditions, **ctx["text_embeds"])
        ev[2].record()
        res = manage_tensor(tensor=out[0], target_device=ctx["tensor_offload_device"], tensor_name="upscaled")
        ev[3].record()
        torch.cuda.synchronize()
        r.update(status="ok", gpu_s=round(ev[1].elapsed_time(ev[2]) / 1000, 4),
                 batch_gpu_s=round(ev[0].elapsed_time(ev[3]) / 1000, 4), out_shape=list(res.shape),
                 out_finite=bool(torch.isfinite(res.float()).all()))
        del latent, base_noise, noises, aug_noises, condition, conditions, out, res
    except (torch.OutOfMemoryError, RuntimeError) as e:
        if not is_oom(e):
            raise
        r.update(status="oom", error=str(e).strip().splitlines()[0][:400])
    finally:
        r["wall_s"] = round(time.perf_counter() - t0, 3)
        r["nvml"] = nvml.stop()
    torch.cuda.synchronize()
    st1 = torch.cuda.memory_stats()
    r["peak_gib"] = torch.cuda.max_memory_allocated() / GIB
    r["peak_above_gib"] = r["peak_gib"] - r["alloc_before_gib"]
    try:
        r["reserved_max_gib"] = torch.cuda.max_memory_reserved() / GIB
    except RuntimeError:
        r["reserved_max_gib"] = None
    r["reserved_after_gib"] = torch.cuda.memory_reserved() / GIB
    free, total = torch.cuda.mem_get_info()
    r["dev_used_after_gib"] = (total - free) / GIB
    r["alloc_retry_warnings"] = retries.n - n0
    for k in ("num_alloc_retries", "num_ooms", "num_sync_all_streams", "num_device_alloc", "num_device_free"):
        if k in st1:
            r[k] = st1[k] - st0.get(k, 0)
    r["tokens_seen"], r["grid_seen"] = hook.tokens, hook.grid
    for k in ("alloc_before_gib", "peak_gib", "peak_above_gib", "reserved_max_gib", "reserved_after_gib",
              "dev_used_after_gib"):
        if isinstance(r.get(k), float):
            r[k] = round(r[k], 4)
    return r


def run_point(runner, ctx, nvml, retries, hook, win, wcfg, kind, w, h, L, repeats, seed):
    import torch
    rec = {"kind": kind, "width": w, "height": h, "L": L, "tokens": tokens_of(w, h, L),
           "latent_shape": [L, h // 8, w // 8, 16], "windows": window_info(win, w, h, L, *wcfg), "repeats": []}
    g = torch.Generator().manual_seed(1000003 * L + 1009 * h + w)  # the same latents for the same point
    lat = torch.randn(rec["latent_shape"], generator=g, dtype=torch.float32).to(ctx["compute_dtype"])
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()  # each point starts from an empty pool: NVML figures are its own
    rec["t_start"] = time.time()
    for _ in range(repeats):
        r = one_forward(runner, ctx, lat, seed, nvml, retries, hook)
        rec["repeats"].append(r)
        gc.collect()
        if r["status"] != "ok":
            torch.cuda.empty_cache()
            break
    rec["t_end"] = time.time()
    reps = rec["repeats"]
    ok = [r for r in reps if r["status"] == "ok"]
    rec["status"] = "ok" if len(ok) == len(reps) else "oom"
    rec["peak_gib"] = round(max(r["peak_gib"] for r in reps), 4)
    rec["alloc_before_gib"] = reps[0]["alloc_before_gib"]
    times = [r["gpu_s"] for r in ok]
    rec["gpu_s"] = times
    rec["steady_s"] = (statistics.median(times[1:]) if len(times) > 1 else times[0]) if times else None
    nv = [r["nvml"] for r in reps if r.get("nvml")]
    rec["nvml_used_max_gib"] = max((x["used_max_gib"] for x in nv), default=None)
    rec["sm_mhz_mean"] = round(statistics.mean(x["sm_mhz_mean"] for x in nv if x["sm_mhz_mean"])) \
        if any(x["sm_mhz_mean"] for x in nv) else None
    rec["power_w_mean"] = round(statistics.mean(x["power_w_mean"] for x in nv if x["power_w_mean"]), 1) \
        if any(x["power_w_mean"] for x in nv) else None
    rec["tokens_seen"] = next((r["tokens_seen"] for r in reps if r.get("tokens_seen")), None)
    rec["alloc_retries"] = sum(r.get("alloc_retry_warnings", 0) for r in reps)
    return rec


def cmd_run(args):
    root = setup_numz()
    import torch
    win = load_window_module(root)
    retries = RetryCounter()
    sweeps = [parse_points(s) for s in args.sweep]
    ref = None
    if args.ref:
        rw, rh, rl = parse_points(args.ref)
        if len(rl) != 1:
            sys.exit("dit_probe: --ref takes one window length (SIZE:L)")
        ref = (rw, rh, rl[0])
    t_load = time.perf_counter()
    import src.core.generation_utils  # noqa: F401  (numz's imports create the CUDA context)
    free0, total = torch.cuda.mem_get_info()
    runner, ctx = build(args)
    torch.cuda.synchronize()
    hook = PatchHook(runner)
    nvml = Nvml()
    cfg = runner.config.dit.model
    wcfg = (tuple(int(v) for v in cfg.window[0]), tuple(dict.fromkeys(str(m) for m in cfg.window_method)))
    res = {"argv": sys.argv, "dit_model": args.dit_model, "attention_mode": args.attention_mode, "seed": args.seed,
           "repeats": args.repeats, "ref_repeats": args.ref_repeats or args.repeats,
           "ref": list(ref) if ref else None, "ref_every": args.ref_every,
           "model": MODEL, "vae_resident": not args.no_vae, "window_config": [list(wcfg[0]), list(wcfg[1])],
           "initial_free_gib": round(free0 / GIB, 3), "total_gib": round(total / GIB, 3),
           "load_s": round(time.perf_counter() - t_load, 1),
           "loaded_alloc_gib": round(torch.cuda.memory_allocated() / GIB, 4),
           "env": env_info(runner, ctx, root), "points": [], "edges": [], "error": None}
    log(f"loaded in {res['load_s']} s: {res['loaded_alloc_gib']} GiB allocated, initial free "
        f"{res['initial_free_gib']} / {res['total_gib']} GiB; {res['env']}")
    out = os.path.abspath(args.out)

    def write():
        with open(out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)

    write()
    state = {"since_ref": 0}

    def do(kind, w, h, L, reps):
        rec = run_point(runner, ctx, nvml, retries, hook, win, wcfg, kind, w, h, L, reps, args.seed)
        rec["seq"] = len(res["points"])
        res["points"].append(rec)
        write()
        wi = rec["windows"][wcfg[1][0]]
        log(f"{kind} {w}x{h} L={L}: {rec['tokens']} tokens (seen {rec['tokens_seen']}), window "
            f"{wi['t']}x{wi['h']}x{wi['w']} ({wi['n']} windows), {rec['status']}, peak {rec['peak_gib']:.2f} GiB "
            f"(model {model_peak_gib(rec['tokens'], not args.no_vae):.2f}), GPU s {rec['gpu_s']} "
            f"(model {model_time_s(rec['tokens']):.2f}), NVML max {rec['nvml_used_max_gib']} GiB, "
            f"SM {rec['sm_mhz_mean']} MHz, {rec['power_w_mean']} W")
        if kind == "ref":
            state["since_ref"] = 0
        else:
            state["since_ref"] += 1
            if ref and args.ref_every and state["since_ref"] >= args.ref_every:
                do("ref", *ref, args.ref_repeats or args.repeats)
        return rec

    try:
        if ref:
            do("ref", *ref, args.ref_repeats or args.repeats)
        for w, h, ls in sweeps:
            last_ok = first_oom = None
            for L in ls:
                if do("sweep", w, h, L, args.repeats)["status"] == "ok":
                    last_ok = L
                else:
                    first_oom = L
                    log(f"{w}x{h}: OOM at L={L}, longer windows skipped")
                    break
            if args.refine and last_ok is not None and first_oom is not None:
                while first_oom - last_ok > 1:
                    mid = (last_ok + first_oom) // 2
                    if do("refine", w, h, mid, args.refine_repeats)["status"] == "ok":
                        last_ok = mid
                    else:
                        first_oom = mid
            res["edges"].append({"width": w, "height": h, "last_ok": last_ok, "first_oom": first_oom})
            write()
        if ref and state["since_ref"]:
            do("ref", *ref, args.ref_repeats or args.repeats)
    except Exception as e:  # noqa: BLE001  (not an OOM: record it, stop; the context may be unusable)
        res["error"] = f"{type(e).__name__}: {str(e)[:500]}"
        write()
        log(f"error, run stopped: {res['error']}")
        raise
    write()
    log(f"done -> {out}")
    print_tables([out])


# ---------------------------------------------------------------- tables

def normalise(points):
    """Per point: the reference's steady time interpolated at its midpoint, and base / that."""
    refs = sorted(((p["t_start"] + p["t_end"]) / 2, p["steady_s"]) for p in points
                  if p["kind"] == "ref" and p["status"] == "ok" and p.get("steady_s"))
    if not refs:
        return None
    base = statistics.median(r[1] for r in refs)
    for p in points:
        t = (p["t_start"] + p["t_end"]) / 2
        before = [r for r in refs if r[0] <= t]
        after = [r for r in refs if r[0] >= t]
        if before and after:
            (t0, s0), (t1, s1) = before[-1], after[0]
            ref_s = s0 if t1 == t0 else s0 + (s1 - s0) * (t - t0) / (t1 - t0)
        else:
            ref_s = (before or after)[-1 if before else 0][1]
        p["ref_s"] = ref_s
        p["norm"] = base / ref_s
    return base


def fit(xs, ys):
    """Least squares y = a + b x; returns (a, b, max |residual|)."""
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    return a, b, max(abs(y - a - b * x) for x, y in zip(xs, ys))


def fit2(x1, x2, ys):
    """Least squares y = a + b x1 + c x2 (Cramer's rule); returns (a, b, c, max |residual|) or None."""
    n = len(ys)
    if n < 3 or not max(x1) or not max(x2):
        return None
    k1, k2 = max(x1), max(x2)  # scaled to [0, 1] for the conditioning, coefficients scaled back below
    x1, x2 = [v / k1 for v in x1], [v / k2 for v in x2]
    s1, s2, s11, s22, s12 = sum(x1), sum(x2), sum(v * v for v in x1), sum(v * v for v in x2), \
        sum(u * v for u, v in zip(x1, x2))
    t0, t1, t2 = sum(ys), sum(u * y for u, y in zip(x1, ys)), sum(v * y for v, y in zip(x2, ys))
    m = [[n, s1, s2], [s1, s11, s12], [s2, s12, s22]]

    def det(a):
        return (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
    d = det(m)
    if abs(d) < 1e-9:
        return None  # the two regressors are (nearly) proportional over these points
    sol = []
    for col in range(3):
        mc = [row[:] for row in m]
        for i, t in enumerate((t0, t1, t2)):
            mc[i][col] = t
        sol.append(det(mc) / d)
    a, b, c = sol
    res = max(abs(y - a - b * u - c * v) for u, v, y in zip(x1, x2, ys))
    return a, b / k1, c / k2, res


def n_windows(p, methods):
    """Attention windows of the block type with the most (the 58 text tokens are repeated in each)."""
    return max(p["windows"][m]["n"] for m in methods)


def fmt(v, nd=2):
    return "–" if v is None else f"{v:.{nd}f}"


def load_results(paths, merge=False):
    """(title, results) per file; with merge, one view of all the files' points in time order, so the reference
    points of every file normalise every point (runs split into one invocation per point, other jobs between)."""
    rs = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            rs.append((os.path.basename(path), json.load(f)))
    if not merge or len(rs) < 2:
        return rs
    first = rs[0][1]
    for name, r in rs[1:]:
        for k in ("dit_model", "ref", "vae_resident", "window_config"):
            if r.get(k) != first.get(k):
                print(f"dit_probe: warning: {name} has another {k} ({r.get(k)} vs {first.get(k)})", file=sys.stderr)
    m = dict(first)
    m["points"] = sorted((p for _, r in rs for p in r["points"]), key=lambda p: p["t_start"])
    m["error"] = "; ".join(f"{n}: {r['error']}" for n, r in rs if r.get("error")) or None
    reps = sorted({r["repeats"] for _, r in rs})
    m["repeats"] = reps[0] if len(reps) == 1 else "/".join(map(str, reps))
    return [(f"{len(rs)} files merged ({', '.join(n for n, _ in rs)})", m)]


def edges_of(points):
    """Per output size: the longest window that passed and the shortest that ran out of memory."""
    out = {}
    for p in points:
        if p["kind"] == "ref":
            continue
        e = out.setdefault((p["width"], p["height"]), {"last_ok": None, "first_oom": None})
        if p["status"] == "ok":
            e["last_ok"] = max(e["last_ok"] or 0, p["L"])
        elif p["status"] == "oom":
            e["first_oom"] = p["L"] if e["first_oom"] is None else min(e["first_oom"], p["L"])
    return out


def print_tables(paths, merge=False):
    for title, r in load_results(paths, merge):
        env = r["env"]
        pts = r["points"]
        vae = r.get("vae_resident", True)
        base = normalise(pts)
        print(f"\n### {title}: {r['dit_model']}, {', '.join(env['attention_mode_effective'])}, "
              f"{env['gpu']}, torch {env['torch']}, flash_attn {env['flash_attn']}\n")
        print(f"Loaded: {r['loaded_alloc_gib']:.2f} GiB allocated (DiT {env['dit_param_gib']} + buffers "
              f"{env['dit_buffer_gib']} + VAE {env['vae_param_gib']}), initial free {r['initial_free_gib']} / "
              f"{r['total_gib']} GiB; {r['repeats']} forwards per point; steady = median after the first.")
        if base:
            rw, rh, rl = r["ref"]
            print(f"Reference {rw}x{rh} L={rl}: median steady {base:.3f} s "
                  f"({1000 * base / tokens_of(rw, rh, rl):.4f} ms/token; model {MODEL['ms_per_token']}).")
        if r.get("error"):
            print(f"**Run stopped by an error:** {r['error']}")
        print()
        print("| Output | L | Tokens | Window t×h×w (windows per block type) | Peak (GiB) | Model (GiB) | Peak − model | "
              "KiB/token above loaded | DiT s (steady) | ms/token | ms/token norm. | Model s | NVML max (GiB) | "
              "SM MHz | W | Status |")
        print("|" + "---|" * 16)
        for p in pts:
            if p["kind"] == "ref":
                continue
            wi = p["windows"][r["window_config"][1][0]]
            nw = "/".join(str(p["windows"][m]["n"]) for m in r["window_config"][1])
            mp = model_peak_gib(p["tokens"], vae)
            above = (p["peak_gib"] - p["alloc_before_gib"]) * 2 ** 20 / p["tokens"]
            st = p.get("steady_s")
            ms = 1000 * st / p["tokens"] if st else None
            msn = ms * p["norm"] if (ms and p.get("norm")) else None
            tag = "" if p["kind"] == "sweep" else f" ({p['kind']})"
            status = p["status"] + ("" if p.get("tokens_seen") in (None, p["tokens"]) else " TOKENS?")
            print(f"| {p['width']}×{p['height']} | {p['L']}{tag} | {p['tokens']} | {wi['t']}×{wi['h']}×{wi['w']} "
                  f"({nw}) | {p['peak_gib']:.2f} | {mp:.2f} | {p['peak_gib'] - mp:+.2f} | {above:.1f} | "
                  f"{fmt(st)} | {fmt(ms, 4)} | {fmt(msn, 4)} | {model_time_s(p['tokens']):.2f} | "
                  f"{fmt(p.get('nvml_used_max_gib'))} | {p.get('sm_mhz_mean') or '–'} | "
                  f"{p.get('power_w_mean') or '–'} | {status} |")
        refs = [p for p in pts if p["kind"] == "ref"]
        if refs:
            print("\n| Reference # | UTC | DiT s (each forward) | Steady s | Peak (GiB) | SM MHz | W |")
            print("|---|---|---|---|---|---|---|")
            for i, p in enumerate(refs):
                print(f"| {i} | {time.strftime('%H:%M:%S', time.gmtime(p['t_start']))} | "
                      f"{', '.join(f'{x:.2f}' for x in p['gpu_s'])} | {fmt(p.get('steady_s'), 3)} | "
                      f"{p['peak_gib']:.2f} | {p.get('sm_mhz_mean') or '–'} | {p.get('power_w_mean') or '–'} |")
        print()
        groups = {}
        for p in pts:
            if p["kind"] != "ref" and p["status"] == "ok":
                groups.setdefault(f"{p['width']}x{p['height']}", []).append(p)
        allp = [p for g in groups.values() for p in g]
        for name, g in list(groups.items()) + ([("all", allp)] if len(groups) > 1 else []):
            f_peak = fit([p["tokens"] for p in g], [p["peak_gib"] for p in g])
            f_time = fit([p["tokens"] for p in g], [p["steady_s"] * p.get("norm", 1) for p in g])
            line = f"- {name}: {len(g)} points"
            if f_peak:
                line += (f"; peak = {f_peak[0]:.2f} GiB + {f_peak[1] * 2 ** 20:.1f} KiB × tokens "
                         f"(max residual {f_peak[2]:.2f} GiB)")
            if f_time:
                line += (f"; normalised time = {f_time[0]:.2f} s + {1000 * f_time[1]:.4f} ms × tokens "
                         f"(max residual {f_time[2]:.2f} s)")
            print(line)
        # tokens + attention windows: each window repeats the text tokens in q, k, v and the output
        methods = r["window_config"][1]
        txt = env["text_pos"][0]
        f2 = fit2([p["tokens"] for p in allp], [n_windows(p, methods) for p in allp], [p["peak_gib"] for p in allp])
        t2 = fit2([p["tokens"] for p in allp], [n_windows(p, methods) for p in allp],
                  [p["steady_s"] * p.get("norm", 1) for p in allp])
        if f2:
            print(f"- tokens + windows, all points: peak = {f2[0]:.2f} GiB + {f2[1] * 2 ** 20:.1f} KiB × tokens + "
                  f"{f2[2] * 2 ** 10:.2f} MiB × windows ({f2[2] * 2 ** 20 / txt:.1f} KiB per repeated text token) "
                  f"(max residual {f2[3]:.2f} GiB)")
        if t2:
            print(f"- tokens + windows, all points: normalised time = {t2[0]:.2f} s + {1000 * t2[1]:.4f} ms × tokens + "
                  f"{1000 * t2[2]:.2f} ms × windows (max residual {t2[3]:.2f} s)")
        for (w, h), e in edges_of(pts).items():
            print(f"- edge {w}x{h}: last passing L = {e['last_ok']}, first OOM L = {e['first_oom']}")
        ooms = [p for p in pts if p["status"] == "oom"]
        for p in ooms:
            last = p["repeats"][-1]
            print(f"- OOM {p['width']}x{p['height']} L={p['L']}: peak before failing {last['peak_gib']:.2f} GiB, "
                  f"NVML max {fmt((last.get('nvml') or {}).get('used_max_gib'))} GiB, after {last['wall_s']} s: "
                  f"{last.get('error')}")


def cmd_table(args):
    print_tables(args.results, args.merge)


# ---------------------------------------------------------------- plan (no GPU)

def cmd_plan(args):
    win = load_window_module(numz_root())
    seq = []
    ref = None
    if args.ref:
        rw, rh, rl = parse_points(args.ref)
        ref = (rw, rh, rl[0])
    since = 0

    def add(kind, w, h, L, reps):
        nonlocal since
        seq.append((kind, w, h, L, reps))
        if kind == "ref":
            since = 0
            return
        since += 1
        if ref and args.ref_every and since >= args.ref_every:
            add("ref", *ref, args.ref_repeats or args.repeats)

    if ref:
        add("ref", *ref, args.ref_repeats or args.repeats)
    for w, h, ls in (parse_points(s) for s in args.sweep):
        last_ok = first_oom = None
        for L in ls:
            add("sweep", w, h, L, args.repeats)
            if model_peak_gib(tokens_of(w, h, L)) > args.room:
                first_oom = L
                break
            last_ok = L
        if args.refine and last_ok is not None and first_oom is not None:
            while first_oom - last_ok > 1:
                mid = (last_ok + first_oom) // 2
                add("refine", w, h, mid, args.refine_repeats)
                if model_peak_gib(tokens_of(w, h, mid)) > args.room:
                    first_oom = mid
                else:
                    last_ok = mid
    if ref and since:
        add("ref", *ref, args.ref_repeats or args.repeats)
    print(f"Model: peak {MODEL['const_gib']} GiB + {MODEL['kib_per_token']} KiB/token, "
          f"{MODEL['ms_per_token']} ms/token; room {args.room} GiB (OOM predicted above it, the forward then fails "
          f"early and costs ~nothing)\n")
    print("| # | Kind | Output | L | Tokens | Window t×h×w (n) | Model peak (GiB) | Model s × forwards |")
    print("|---|---|---|---|---|---|---|---|")
    total = 0.0
    for i, (kind, w, h, L, reps) in enumerate(seq):
        tk = tokens_of(w, h, L)
        wi = window_info(win, w, h, L)["720pwin_by_size_bysize"]
        oom = model_peak_gib(tk) > args.room
        cost = 0.0 if oom else model_time_s(tk) * reps
        total += cost
        print(f"| {i} | {kind} | {w}×{h} | {L} | {tk} | {wi['t']}×{wi['h']}×{wi['w']} ({wi['n']}) | "
              f"{model_peak_gib(tk):.2f}{' OOM' if oom else ''} | {model_time_s(tk):.1f} × {reps} |")
    print(f"\nDiT GPU time at the model's speed: {total / 60:.1f} min; ×1.6 for a slow session: "
          f"{total * 1.6 / 60:.1f} min (+ ≈ 1 min of loading and per-point overhead)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def sweep_opts(p):
        p.add_argument("--sweep", action="append", required=True, metavar="SIZE:L1,L2,...")
        p.add_argument("--ref", default="1080:6", help="reference point SIZE:L ('' for none)")
        p.add_argument("--ref-every", type=int, default=3, help="reference after every N points (0: start/end only)")
        p.add_argument("--repeats", type=int, default=2)
        p.add_argument("--ref-repeats", type=int, help="forwards per reference point (default: --repeats)")
        p.add_argument("--refine", action="store_true", help="bisect the OOM edge after a sweep's first OOM")
        p.add_argument("--refine-repeats", type=int, default=1)

    p = sub.add_parser("run")
    sweep_opts(p)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--dit-model", default="seedvr2_ema_7b_fp16.safetensors")
    p.add_argument("--attention-mode", default="flash_attn_2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-vae", action="store_true", help="leave the VAE off the GPU (the CLI keeps it there)")
    p.add_argument("--out", default="dit_probe.json")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("table")
    p.add_argument("results", nargs="+")
    p.add_argument("--merge", action="store_true",
                   help="one table for all files (points in time order, every file's references normalise them)")
    p.set_defaults(func=cmd_table)

    p = sub.add_parser("plan")
    sweep_opts(p)
    p.add_argument("--room", type=float, default=94.2, help="GiB a forward may use (model peak above it: OOM)")
    p.set_defaults(func=cmd_plan)

    args = ap.parse_args()
    if getattr(args, "repeats", 1) < 1:
        sys.exit("dit_probe: --repeats must be >= 1")
    args.func(args)


if __name__ == "__main__":
    main()

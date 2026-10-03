#!/usr/bin/env python3
"""Numerics and input-preparation switches for the SeedVR2 CLI, patched in at import (no checkout change).

numz (the CLI) and ByteDance's reference inference differ in a few numerical details, and numz's
input preparation makes choices of its own (resize kernel, padding). This wrapper turns each of
them into a switch, so that one CLI run measures one variant against numz's own output. Every
switch defaults to numz's behaviour, and a switch left unset or set to its default patches
nothing: the default run is bit-identical to the plain CLI.

Switches (environment):
  NUM_ROPE=fp16|fp32
      fp16 (numz): the 7B's RoPE frequencies are stored in float16 in numz's fp16 file and loaded
      as they are, so rotary_embedding_torch computes the angle table in float16 (128 pi = 402.12
      is stored as 402.0). The 7B's forward then casts the table to q's dtype, float32 there (q
      leaves the RMS norm in float32 under autocast): the float16 rounding stays.
      fp32 (ByteDance): once the weights are loaded, every RoPE `freqs` buffer is replaced by the
      float32 values RotaryEmbedding computes from its constructor arguments (recorded when the
      model is created); a buffer that is already float32 (ByteDance's .pth) is kept as loaded.
      The get_axial_freqs caches are cleared, and the 7B's forward casts the table to float32.
      The 3B ("lang" frequencies) needs no forward change: its forward keeps the table's dtype.
  NUM_VAE=mode|sample
      mode (numz): the VAE encoder returns the posterior mean. sample (ByteDance): it returns
      posterior.sample() (diffusers: mean + exp(0.5 clamp(logvar, -30, 20)) * randn, in bfloat16,
      from the global CUDA generator, which numz seeds with seed + 1000000 before Phase 1). The
      patch sits above the tiling: a tiled encode samples once, from the blended parameters.
  NUM_PREP=bf16|fp32
      bf16 (numz): the frames (float16 since the CLI read them) are cast to bfloat16 before the
      transform chain (resize, clamp, pad, normalise to [-1, 1]); torchvision's resize computes in
      float32 but reads and writes bfloat16. Phase 4 rebuilds its colour reference in float16.
      fp32 (ByteDance): the chain runs in float32 from the exact 8-bit values (k / 255, recovered
      from the float16 frames and checked; other inputs get a plain float32 cast), and Phase 1
      casts its output to bfloat16 for the encoder, as numz does. Phase 4's colour reference stays
      float32. With --input_noise_scale > 0 the input noise is drawn in float32.
  NUM_RESIZE=tv-aa|tv-noaa|zimg-spline36|zimg-lanczos
      The resize to --resolution (short side), to the size numz computes.
      tv-aa (numz): torchvision bicubic, antialias=True (PyTorch's antialiased kernel, a = -0.5).
      tv-noaa: torchvision bicubic, antialias=False (classic bicubic, a = -0.75).
      zimg-*: ffmpeg's zscale (zimg), filter=spline36 or filter=lanczos (zimg's default 3 taps),
      float in and float out: the RGB frames go to ffmpeg as gbrpf32le rawvideo and come back as
      gbrpf32le, with no 8-bit step. RGB has no chroma planes: each plane is resized alone, with
      the same kernel. No colour property is set, so zimg converts no matrix, transfer or range:
      float RGB is full range as it is, values outside [0, 1] (ringing) are kept, and numz's clamp
      to [0, 1] follows, as after bicubic. Runs on the CPU (the frames go through ffmpeg).
      Like torchvision, every kernel returns the frames unchanged when the size already matches.
  NUM_PAD=zero|reflect|replicate|grey|black+N|reflect+black+N|reflect>=R+black+N
      Padding to multiples of 16, bottom and right only, trimmed after decode as numz does (numz
      crops every decoded frame to the resized size, whatever the padding).
      zero (numz): zeros, black after normalisation. reflect / replicate: torch's pad modes.
      grey: mid grey, 0.5 before normalisation (0 in [-1, 1]), where numz pads.
      black+N: zeros to the multiple of 16, then N more rows of zeros, and N more columns where
      numz pads columns (none at 1920 wide); N a positive multiple of 16.
      reflect+black+N: reflect to the multiple of 16, then the same N rows (and columns) of zeros.
      reflect>=R+black+N: reflect at least R rows (the fewest r >= R that reach a multiple of 16;
      columns the same way, only where numz pads columns), then the same N rows (and columns) of
      zeros: at 1080p reflect>=8+black+16 is reflect+black+16 (r = 8), at 720p it reflects 16.
  NUM_ATTN=auto|fp16
      auto (numz): FlashAttentionVarlen casts q/k/v to the pipeline's compute dtype (bfloat16 when
      the GPU runs a bfloat16 cuBLAS matmul). fp16: to float16 instead (7B and 3B), with autocast
      off around the kernel (SDPA would be cast back to bfloat16). This emulates the attention of
      GPUs without bfloat16 only: there, numz also runs autocast, the VAE and the text embeddings
      in float16.
  NUM_NORM=fp32|bf16
      fp32 (numz): the DiT's norms (numz's CustomRMSNorm / CustomLayerNorm, which replace Apex's
      fused norms; 7B and 3B) compute under the DiT's bfloat16 autocast with autocast's float32
      rules, and return float32. bf16 (ByteDance's Apex FusedRMSNorm / FusedLayerNorm under
      autocast): input and weights cast to the autocast dtype (bfloat16), normalised in float32
      and rounded to bfloat16, then scaled by the weight (and shifted by the bias) in bfloat16
      arithmetic, in the order of Apex's kernel: bfloat16 out. The 7B's RoPE then gets
      bfloat16 q and k: its angle table is cast to float32 (as NUM_ROPE=fp32's forward does; the
      values stay numz's unless NUM_ROPE=fp32) instead of to q's dtype, which would round the
      angles to bfloat16 in numz's code and not in ByteDance's.
  NUM_VAE_AUTOCAST=0|1
      0 (numz): the VAE decodes without autocast when the latent is already in the VAE's dtype
      (bfloat16), and causal_norm_wrapper returns the norms' output dtype. 1 (ByteDance, whose
      decode runs inside the DiT's autocast block): VideoAutoencoderKLWrapper.decode runs under
      bfloat16 autocast, and causal_norm_wrapper casts every norm output back to its input dtype
      (ByteDance's causal_inflation_lib), wherever it is imported.
  NUM_DECODE=bf16|fp16|fp32
      The VAE decode's dtype, weights and activations. bf16 (numz): the VAE encodes and decodes in
      bfloat16 (numz converts the VAE file's weights to bfloat16 when it loads them). fp16 / fp32:
      the encode stays numz's; the VAE file is read again when numz loads it (numz's loader: the
      file's own values, float16 for the CLI's VAE file), and at the first decode the VAE's weights
      are set from it in that dtype, the latent is cast to it (exactly), and the decoder runs
      without autocast. fp32 runs with TF32 off, cuDNN's convolutions included (torch allows TF32
      there by default), unless NUM_TF32=1. Numz's bfloat16 VAE is restored after Phase 3.
      Implies NUM_OUT32=1; excludes NUM_VAE_AUTOCAST=1.
  NUM_OUT32=0|1
      0 (numz): Phase 3 writes the decoded frames into a bfloat16 final_video ([-1, 1]), Phase 4
      colour-corrects them and normalises them to [0, 1] in bfloat16 and writes them back, and the
      CLI casts the result to float32 for its writers. 1: final_video is float32 and Phase 4 works
      in float32 (its colour reference stays numz's): the writers get the decoder's output with
      no bfloat16 step after it.
  NUM_DECODE_HOOKS=auto|0|1
      Forward hooks on every module of the VAE decoder count the non-finite values of its outputs
      and keep their range, logged after the decode (with the decoder output's own count): the
      float16 overflow check (float16 ends at 65504). auto: with NUM_DECODE=fp16 only. Reads only,
      but the extra passes cost decode time and some memory.
  NUM_TF32=0|1
      0 (numz): torch's defaults (matmul TF32 off, cuDNN TF32 on). 1 (ByteDance's init_torch):
      torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32 set to True
      (with NUM_DECODE=fp32, TF32 is on during the decode too; off otherwise).
  NUM_CONV3D_WA=auto|0|1
      numz's NVIDIA Conv3d workaround (causal_inflation_lib, imported by value): auto keeps numz's
      detection, 0 / 1 force it off / on. The workaround calls cudnn_convolution and adds the bias
      separately.
  NUM_DIT_WEIGHTS=<path>, NUM_VAE_WEIGHTS=<path>
      Load these weight files (e.g. ByteDance's fp32 .pth) instead of --dit_model's file and the
      CLI's VAE file: the path is swapped where the loader opens the checkpoint, so --dit_model
      still selects the model config (keep the 7B fp16 name for the 7B) and passes the CLI's
      checks. The DiT keeps the file's dtype (float32 weights run under bfloat16 autocast, as
      ByteDance's; autocast then also caches bfloat16 copies of them for the forward, since the
      parameters require grad: 7B float32 = 30.7 GiB of weights, about 15 GiB more at the DiT's
      peak); the VAE is converted to bfloat16 on load, as numz always does. The weights' size in
      memory is logged, and a tensor left unloaded stops the run.
  NUM_CC_EXTRA=<methods>
      Comma list of the CLI's colour corrections (lab, wavelet, wavelet_adaptive, hsv, adain), and
      of the variants below. Before the CLI's own Phase 4, Phase 4 runs on a copy of the Phase 3
      frames with each method, and the result is written as an extra FFV1 master next to
      ffv1_out.py's, named <master stem>.cc-<method>.mkv (pixel formats from FFV1_OUT_PIXFMT; needs
      ffv1_out.py next to this script). The CLI's output keeps its --color_correction (e.g. none).
      Phase 1 is told about the extra method only so that it keeps the batch indices Phase 4 needs.
      One chunk per input (no --chunk_size).
      labties: numz's lab with tied values matched alike. numz's histogram matching (color_fix.py
      _histogram_matching_channel, for L*, a* and b*) gives the source value of rank r the
      reference's value of rank r; equal source values get consecutive ranks in the order the sort
      leaves them (memory order with CUDA's stable sort), so pixels of one colour can get different
      reference values. labties gives every pixel of a source value the mean of the reference's
      values over that value's ranks (computed in float64; a value that occurs once keeps numz's
      exact result); everything else is numz's lab, run by numz's own Phase 4 with that one
      function swapped while it runs. Per batch and channel, the log tells the share of pixels whose
      value is tied and how far numz's own mapping spreads them: the spread of a value is the range
      of the reference's values over its ranks.
  NUM_CHECK=1
      Logs once each: the dtypes of q/k/v entering FlashAttentionVarlen and the attention kernel
      (with the autocast state), the RoPE buffers and angle table, the resize input and output,
      the padding, the transform chain's output, the VAE encoder's input and whether it returned
      the mode or a sample, the Conv3d workaround and TF32 flags, the DiT norms' input and output
      dtypes, the VAE decode's autocast state and the VAE norms' input and output dtypes, the
      decode's latent and weight dtypes and TF32 flags with the decoder output's range, and the
      frames' dtype after Phase 3 and Phase 4. Probes that never fired are listed at exit. Reads
      only: no numerical change and no random draw.
  NUM_FFMPEG       ffmpeg binary for the zimg resize (default: ffmpeg on PATH; it needs zscale)

The active settings are logged in one line at startup, the patches applied at exit. Single GPU
only: numz's multi-GPU workers are new processes that this wrapper does not reach.

Usage (cwd = the SeedVR2 checkout, its venv's python):
  python numerics_patch.py inference_cli.py <CLI args>
  python3 bench.py run NAME --wrap numerics_patch.py --wrap ffv1_out.py --env NUM_ROPE=fp32 \\
      --env NUM_CHECK=1 -- <CLI args> --color_correction none        # chained with ffv1_out.py
  python numerics_patch.py --selftest    # zimg resize, 8-bit recovery, padding, labties (CPU, needs ffmpeg)
"""
import argparse
import atexit
import contextlib
import copy
import importlib.abc
import inspect
import math
import os
import re
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

CHOICES = {  # the first value is numz's behaviour, the default
    "NUM_ROPE": ("fp16", "fp32"),
    "NUM_VAE": ("mode", "sample"),
    "NUM_PREP": ("bf16", "fp32"),
    "NUM_RESIZE": ("tv-aa", "tv-noaa", "zimg-spline36", "zimg-lanczos"),
    "NUM_PAD": ("zero", "reflect", "replicate", "grey"),  # and black+N, reflect[>=R]+black+N: pad_extra()
    "NUM_ATTN": ("auto", "fp16"),
    "NUM_NORM": ("fp32", "bf16"),
    "NUM_VAE_AUTOCAST": ("0", "1"),
    "NUM_DECODE": ("bf16", "fp16", "fp32"),
    "NUM_OUT32": ("0", "1"),
    "NUM_DECODE_HOOKS": ("auto", "0", "1"),
    "NUM_TF32": ("0", "1"),
    "NUM_CONV3D_WA": ("auto", "0", "1"),
    "NUM_CHECK": ("0", "1"),
}
CC_METHODS = ("lab", "wavelet", "wavelet_adaptive", "hsv", "adain")
CC_VARIANTS = {"labties": "lab"}  # NUM_CC_EXTRA variants: the numz method each one runs, patched
ATTN_KERNELS = ("call_flash_attn_2_varlen", "call_flash_attn_3_varlen", "call_sage_attn_2_varlen",
                "call_sage_attn_3_varlen", "pytorch_varlen_attention")
PROBES = ("attention", "attention kernel", "RoPE buffers", "RoPE table", "resize", "pad", "chain", "VAE encode",
          "norm", "VAE decode", "VAE norm", "decode", "output")


def log(msg):
    print(f"numerics_patch: {msg}", file=sys.stderr, flush=True)


def pad_extra(v):
    """NUM_PAD=black+N, reflect+black+N or reflect>=R+black+N -> (torch mode before the black rows,
    N, R: the fewest rows that mode adds, rounded up to a multiple of 16 in total); None otherwise."""
    m = re.fullmatch(r"black\+(\d+)", v)
    if m:
        mode, rmin, n = "constant", 0, int(m.group(1))
    else:
        m = re.fullmatch(r"reflect(?:>=(\d+))?\+black\+(\d+)", v)
        if not m:
            return None
        mode, rmin, n = "reflect", int(m.group(1) or 0), int(m.group(2))
    if n <= 0 or n % 16:
        sys.exit(f"numerics_patch: NUM_PAD={v!r}: N must be a positive multiple of 16")
    return mode, n, rmin


class State:
    def __init__(self):
        o = {}
        for k, choices in CHOICES.items():
            v = os.environ.get(k, "").strip() or choices[0]
            if v not in choices and not (k == "NUM_PAD" and pad_extra(v)):
                more = ", black+N, reflect[>=R]+black+N (N a multiple of 16)" if k == "NUM_PAD" else ""
                sys.exit(f"numerics_patch: {k}={v!r}: choose from {', '.join(choices)}{more}")
            o[k] = v
        self.rope, self.vae, self.prep = o["NUM_ROPE"], o["NUM_VAE"], o["NUM_PREP"]
        self.resize, self.pad, self.attn = o["NUM_RESIZE"], o["NUM_PAD"], o["NUM_ATTN"]
        self.tf32, self.conv3d, self.check = o["NUM_TF32"], o["NUM_CONV3D_WA"], o["NUM_CHECK"] == "1"
        self.norm, self.vae_ac = o["NUM_NORM"], o["NUM_VAE_AUTOCAST"] == "1"
        self.decode = o["NUM_DECODE"]
        self.out32 = o["NUM_OUT32"] == "1" or self.decode != "bf16"
        self.dec_hooks = o["NUM_DECODE_HOOKS"] == "1" or (o["NUM_DECODE_HOOKS"] == "auto" and self.decode == "fp16")
        if self.vae_ac and self.decode != "bf16":
            sys.exit("numerics_patch: NUM_VAE_AUTOCAST=1 decodes under bfloat16 autocast: it needs NUM_DECODE=bf16")
        self.dit_weights = os.environ.get("NUM_DIT_WEIGHTS", "").strip() or None
        self.vae_weights = os.environ.get("NUM_VAE_WEIGHTS", "").strip() or None
        for k, p in (("NUM_DIT_WEIGHTS", self.dit_weights), ("NUM_VAE_WEIGHTS", self.vae_weights)):
            if p and not os.path.isfile(p):
                sys.exit(f"numerics_patch: {k}={p}: no such file")
        cc = [c.strip() for c in os.environ.get("NUM_CC_EXTRA", "").split(",")]
        self.cc_extra = list(dict.fromkeys(c for c in cc if c and c != "none"))
        for c in self.cc_extra:
            if c not in CC_METHODS and c not in CC_VARIANTS:
                sys.exit(f"numerics_patch: NUM_CC_EXTRA {c!r}: choose from {', '.join(CC_METHODS + tuple(CC_VARIANTS))}")
        self.seen = set()           # NUM_CHECK (kind, key) already logged
        self.probed = set()         # probe kinds that fired, for the summary at exit
        self.once = set()
        self.patched = []
        self.rope_mods = []         # the imported rope modules (their lru caches)
        self.cli_patched = False
        self.ties_calls = 0         # NUM_CC_EXTRA=labties: matchings done in the current render (3 per batch)
        self.file = None            # NUM_CC_EXTRA: current input, CLI output path, writers
        self.files_done = 0
        self.vae_src = None         # NUM_DECODE: (numz's state-dict loader, VAE checkpoint, debug), seen at load
        self.vae_raw = None         # the VAE file's state dict as stored (CPU), read when the VAE is loaded
        self.vae_numz_dtype = None  # the VAE's dtype before the decode swapped its weights
        self.in_decode = None       # NUM_OUT32: numz's compute dtype while decode_all_batches runs

    def summary(self):
        items = [("rope", self.rope, "fp16"), ("vae", self.vae, "mode"), ("prep", self.prep, "bf16"),
                 ("resize", self.resize, "tv-aa"), ("pad", self.pad, "zero"), ("attn", self.attn, "auto"),
                 ("norm", self.norm, "fp32"), ("vae_autocast", "1" if self.vae_ac else "0", "0"),
                 ("decode", self.decode, "bf16"), ("out32", "1" if self.out32 else "0", "0"),
                 ("decode_hooks", "1" if self.dec_hooks else "0", "0"),
                 ("tf32", self.tf32, "0"), ("conv3d_wa", self.conv3d, "auto"),
                 ("dit_weights", self.dit_weights or "numz", "numz"),
                 ("vae_weights", self.vae_weights or "numz", "numz"),
                 ("cc_extra", ",".join(self.cc_extra) or "none", "none"), ("check", "1" if self.check else "0", "0")]
        changed = [k for k, v, d in items if v != d and k not in ("cc_extra", "check", "decode_hooks")]
        return (" ".join(f"{k}={v}" for k, v, _ in items)
                + (f" | changed from numz: {', '.join(changed)}" if changed else " | numz defaults"))


S = State()


def want(kind, key):
    """NUM_CHECK: True the first time (kind, key) comes up."""
    if not S.check:
        return False
    S.probed.add(kind)
    if (kind, key) in S.seen:
        return False
    S.seen.add((kind, key))
    return True


def check(kind, key, msg):
    if want(kind, key):
        log(f"check: {kind}: {msg}")


def once(key, msg):
    if key not in S.once:
        S.once.add(key)
        log(msg)


def dt(x):
    return str(x).replace("torch.", "")


def autocast_state(device_type):
    import torch
    try:
        if torch.is_autocast_enabled(device_type):
            return f"autocast {dt(torch.get_autocast_dtype(device_type))}"
    except (TypeError, AttributeError, RuntimeError):
        pass
    return "autocast off"


def call_with(fn, a, k, **changes):
    """fn(*a, **k) with some arguments replaced (by name)."""
    if not a:
        return fn(**dict(k, **changes))
    b = inspect.signature(fn).bind(*a, **k)
    b.arguments.update(changes)
    return fn(*b.args, **b.kwargs)


# ---------------------------------------------------------------- RoPE (NUM_ROPE)

def patch_rope(mod):
    variant = "7b" if "dit_7b" in mod.__name__ else "3b"
    S.rope_mods.append(mod)
    if S.rope == "fp32":
        orig_cls = mod.RotaryEmbedding

        class RotaryEmbedding(orig_cls):
            """rotary_embedding_torch's class, recording its constructor arguments."""
            _num_orig = orig_cls

            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self._num_rope_args = (a, dict(k))

        RotaryEmbedding.__name__ = RotaryEmbedding.__qualname__ = orig_cls.__name__
        mod.RotaryEmbedding = RotaryEmbedding
        S.patched.append(f"{mod.__name__}.RotaryEmbedding (records its arguments)")
    cls = getattr(mod, "NaRotaryEmbedding3d", None)
    if (S.rope == "fp32" or S.norm == "bf16") and variant == "7b" and cls is not None:
        # NUM_NORM=bf16 needs it too: q leaves bfloat16 norms in bfloat16, and numz's cast of the table to
        # q's dtype would round the angles to bfloat16 (ByteDance rotates with its float32 table)
        def forward(self, q, k, shape, cache):  # numz's NaRotaryEmbedding3d.forward, float32 table
            import torch
            freqs = cache("rope_freqs_3d", lambda: self.get_freqs(shape))
            freqs = freqs.to(device=q.device, dtype=torch.float32)  # numz: dtype=q.dtype
            q = mod.rearrange(q, "L h d -> h L d")
            k = mod.rearrange(k, "L h d -> h L d")
            q = mod.apply_rotary_emb(freqs, q.float()).to(q.dtype)
            k = mod.apply_rotary_emb(freqs, k.float()).to(k.dtype)
            q = mod.rearrange(q, "h L d -> L h d")
            k = mod.rearrange(k, "h L d -> L h d")
            return q, k
        cls.forward = forward
        S.patched.append(f"{mod.__name__}.NaRotaryEmbedding3d.forward (float32 table)")
    if S.check:
        orig_apply = mod.apply_rotary_emb

        def apply_rotary_emb(freqs, t, *a, **k):
            if want("RoPE table", variant):
                log(f"check: RoPE table: {variant}: angle table {dt(freqs.dtype)} {tuple(freqs.shape)}, "
                    f"max |angle| {freqs.detach().abs().max().item():.7g}, rotates {dt(t.dtype)}")
            return orig_apply(freqs, t, *a, **k)
        mod.apply_rotary_emb = apply_rotary_emb


def rope_modules(model):
    """(name, rotary_embedding_torch module) of every RoPE of a DiT."""
    return [(f"{name}.rope", m.rope) for name, m in model.named_modules()
            if hasattr(getattr(m, "rope", None), "freqs") and hasattr(m.rope, "get_axial_freqs")]


def fix_rope(model):
    """NUM_ROPE=fp32: float32 frequencies, recomputed as rotary_embedding_torch builds them."""
    import torch
    refs, n, kept, worst, missing = {}, 0, 0, 0.0, 0
    for name, r in rope_modules(model):
        args = getattr(r, "_num_rope_args", None)
        if args is None:
            missing += 1
            continue
        a, k = args
        key = repr((a, sorted(k.items())))
        if key not in refs:
            with torch.device("cpu"):
                refs[key] = type(r)._num_orig(*a, **k).freqs.detach().float()
        ref, old = refs[key], r.freqs
        worst = max(worst, (old.detach().float().cpu() - ref).abs().max().item())
        if old.dtype == torch.float32:
            kept += 1
        else:
            r.freqs = ref.to(old.device)
        if hasattr(r, "cached_freqs_seq_len"):  # rotary_embedding_torch's own cache ("lang" frequencies)
            r.cached_freqs_seq_len = 0
        n += 1
    cleared = 0
    for mod in S.rope_mods:  # the class-level lru_cache of RotaryEmbeddingBase.get_axial_freqs
        for obj in list(vars(mod).values()):
            fn = obj.__dict__.get("get_axial_freqs") if isinstance(obj, type) else None
            if hasattr(fn, "cache_clear"):
                fn.cache_clear()
                cleared += 1
    for _, m in model.named_modules():  # CompatibleDiT's per-module wrappers, if already installed
        orig = getattr(m, "_original_get_axial_freqs", None)
        if hasattr(orig, "cache_clear"):
            orig.cache_clear()
    if missing:
        log(f"RoPE fp32: {missing} RoPE modules without recorded arguments, NOT changed")
    log(f"RoPE fp32: {n - kept} freqs buffers set to float32 recomputed from scratch, {kept} already float32 "
        f"kept as loaded; max |loaded - recomputed| {worst:.6g}; {cleared} get_axial_freqs caches cleared")


# ---------------------------------------------------------------- weights (loader)

def after_load(model, model_type, path):
    import torch
    left = [n for n, p in model.named_parameters() if p.is_meta] + [n for n, b in model.named_buffers() if b.is_meta]
    if left:
        raise RuntimeError(f"numerics_patch: {len(left)} {model_type} tensors not loaded from {path} "
                           f"(still on meta): {', '.join(left[:8])}")
    if (model_type == "DiT" and S.dit_weights) or (model_type == "VAE" and S.vae_weights) or S.check:
        sizes = {}
        for p in model.parameters():
            c, b = sizes.get(p.dtype, (0, 0))
            sizes[p.dtype] = (c + p.numel(), b + p.numel() * p.element_size())
        alloc = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        log(f"{model_type} loaded from {Path(path).name}: parameters "
            + ", ".join(f"{dt(d)} {c / 1e9:.3f} G" for d, (c, _) in sizes.items())
            + f", {sum(b for _, b in sizes.values()) / 2**30:.2f} GiB; CUDA allocated {alloc:.2f} GiB")
    if model_type != "DiT":
        return
    if S.rope == "fp32":
        fix_rope(model)
    ropes = rope_modules(model)
    if ropes and want("RoPE buffers", "loaded"):
        kinds = {}
        for _, r in ropes:
            kinds[dt(r.freqs.dtype)] = kinds.get(dt(r.freqs.dtype), 0) + 1
        name, r = ropes[0]
        log(f"check: RoPE buffers: {len(ropes)} freqs buffers {kinds}; {name}.freqs = "
            f"{[round(x, 5) for x in r.freqs.detach().float().cpu().tolist()]}")


def patch_loader(mod):
    if not (S.dit_weights or S.vae_weights or S.rope == "fp32" or S.decode != "bf16" or S.check):
        return
    orig = mod._load_model_weights

    def _load_model_weights(model, checkpoint_path, target_device, used_meta, model_type, *a, **k):
        alt = {"DiT": S.dit_weights, "VAE": S.vae_weights}.get(model_type)
        if alt:
            log(f"{model_type} weights: {alt} instead of {checkpoint_path}")
            checkpoint_path = alt
        model = orig(model, checkpoint_path, target_device, used_meta, model_type, *a, **k)
        after_load(model, model_type, checkpoint_path)
        if model_type == "VAE" and S.decode != "bf16":  # (cpu_reason, debug, override_dtype) follow model_type
            S.vae_src = (mod.load_quantized_state_dict, checkpoint_path, a[1] if len(a) > 1 else k.get("debug"))
            S.vae_raw = None
            set_vae_weights(None, None)  # reads the file now, not in the decode's time
        return model

    mod._load_model_weights = _load_model_weights
    S.patched.append(f"{mod.__name__}._load_model_weights")


# ---------------------------------------------------------------- attention (NUM_ATTN)

def patch_attention(mod):
    cls = getattr(mod, "FlashAttentionVarlen", None)
    if cls is None or not (S.attn == "fp16" or S.check):
        return
    variant = "7b" if "dit_7b" in mod.__name__ else "3b"
    orig = cls.forward

    def forward(self, q, k, v, *a, **kw):
        import torch
        if S.attn == "fp16":
            self.compute_dtype = torch.float16
        if want("attention", (variant, q.dtype, self.compute_dtype)):
            log(f"check: attention: {variant} {self.attention_mode}: q/k/v enter FlashAttentionVarlen as "
                f"{dt(q.dtype)}/{dt(k.dtype)}/{dt(v.dtype)}, its compute_dtype {dt(self.compute_dtype)}, "
                f"{autocast_state(q.device.type)}")
        if S.attn == "fp16":
            with torch.autocast(q.device.type, enabled=False):
                return orig(self, q, k, v, *a, **kw)
        return orig(self, q, k, v, *a, **kw)

    cls.forward = forward
    S.patched.append(f"{mod.__name__}.FlashAttentionVarlen.forward" + (" (float16 q/k/v)" if S.attn == "fp16" else ""))
    if S.check:
        for name in ATTN_KERNELS:
            fn = getattr(mod, name, None)
            if fn is None:
                continue

            def wrapped(q, k, v, *a, __fn=fn, __name=name, **kw):
                if want("attention kernel", (variant, __name, q.dtype)):
                    log(f"check: attention kernel: {variant} {__name} gets q/k/v "
                        f"{dt(q.dtype)}/{dt(k.dtype)}/{dt(v.dtype)}, {autocast_state(q.device.type)}")
                return __fn(q, k, v, *a, **kw)
            setattr(mod, name, wrapped)


# ---------------------------------------------------------------- DiT norms (NUM_NORM)

def apex_norm(self, input, adt, rms):
    """Apex's FusedRMSNorm / FusedLayerNorm under autocast: input and weights cast to adt
    (_cast_if_autocast_enabled), the kernel normalising in float32, rounding to adt, then
    gamma * y (+ beta) in adt arithmetic (cuApplyLayerNorm_: gamma[i] * static_cast<V>(...) + beta[i])."""
    import torch
    dims = tuple(range(-len(self.normalized_shape), 0))
    w, b = getattr(self, "weight", None), getattr(self, "bias", None)
    with torch.autocast(input.device.type, enabled=False):
        x = input.to(adt).float()
        if rms:
            y = x * torch.rsqrt(x.pow(2).mean(dim=dims, keepdim=True) + self.eps)
        else:
            mu = x.mean(dim=dims, keepdim=True)
            y = (x - mu) * torch.rsqrt((x - mu).pow(2).mean(dim=dims, keepdim=True) + self.eps)
        y = y.to(adt)
        if w is not None:
            y = y * w.to(adt)
            if b is not None and not rms:
                y = y + b.to(adt)
    return y


def patch_norm(mod):
    if not (S.norm == "bf16" or S.check):
        return
    variant = "7b" if "dit_7b" in mod.__name__ else "3b"
    done = []
    for cname in ("CustomRMSNorm", "CustomLayerNorm"):
        cls = getattr(mod, cname, None)
        if cls is None:
            continue

        def forward(self, input, __orig=cls.forward, __name=cname):
            import torch
            dev = input.device.type
            if S.norm == "bf16" and torch.is_autocast_enabled(dev):  # Apex's fused norms under autocast
                out = apex_norm(self, input, torch.get_autocast_dtype(dev), __name == "CustomRMSNorm")
            else:
                out = __orig(self, input)
            if want("norm", (variant, __name, input.dtype, out.dtype)):
                log(f"check: norm: {variant} {__name} gets {dt(input.dtype)}, returns {dt(out.dtype)}, "
                    f"{autocast_state(dev)}")
            return out

        cls.forward = forward
        done.append(cname)
    if done:
        S.patched.append(f"{mod.__name__}.{', '.join(done)}.forward" + (" (bfloat16 in and out)" if S.norm == "bf16"
                                                                          else ""))


# ---------------------------------------------------------------- VAE (NUM_VAE, NUM_CONV3D_WA, NUM_VAE_AUTOCAST)

def patch_vae(mod):
    cls = getattr(mod, "VideoAutoencoderKLWrapper", None)
    if cls is not None and (S.vae_ac or S.check):
        orig_dec = cls.decode

        def decode(self, z, *a, **k):
            import torch
            dev = z.device.type
            if S.vae_ac:
                with torch.autocast(dev, dtype=torch.bfloat16):
                    check("VAE decode", "autocast", f"latent {dt(z.dtype)} {tuple(z.shape)}, {autocast_state(dev)} "
                          f"(forced, ByteDance)")
                    return orig_dec(self, z, *a, **k)
            check("VAE decode", "numz", f"latent {dt(z.dtype)} {tuple(z.shape)}, {autocast_state(dev)} (numz)")
            return orig_dec(self, z, *a, **k)

        cls.decode = decode
        S.patched.append(f"{mod.__name__}.VideoAutoencoderKLWrapper.decode"
                         + (" (bfloat16 autocast)" if S.vae_ac else ""))
    if cls is None or not (S.vae == "sample" or S.check):
        return
    orig = cls.encode

    def encode(self, x, *a, **k):
        out = orig(self, x, *a, **k)  # numz: CausalEncoderOutput(posterior.mode().squeeze(2), posterior)
        p = out.posterior
        if S.vae == "sample":
            z = p.sample().squeeze(2)  # ByteDance's VideoAutoencoderKLWrapper.encode
            if want("VAE encode", "sample"):
                d = (z.float() - out.latent.float()).abs().mean().item()
                log(f"check: VAE encode: posterior sample (input {dt(x.dtype)} {tuple(x.shape)}, tiled "
                    f"{bool(k.get('tiled'))}): latent {dt(z.dtype)} {tuple(z.shape)}, mean |sample - mode| {d:.4g}, "
                    f"mean std {p.std.float().mean().item():.4g}")
            return type(out)(z, p)
        if want("VAE encode", "mode"):
            log(f"check: VAE encode: posterior mode (input {dt(x.dtype)} {tuple(x.shape)}, tiled "
                f"{bool(k.get('tiled'))}): latent {dt(out.latent.dtype)} {tuple(out.latent.shape)}")
        return out

    cls.encode = encode
    S.patched.append(f"{mod.__name__}.VideoAutoencoderKLWrapper.encode"
                     + (" (posterior sample)" if S.vae == "sample" else ""))


def patch_causal_lib(mod):
    patch_conv3d(mod)
    orig = getattr(mod, "causal_norm_wrapper", None)
    if orig is None or not (S.vae_ac or S.check):
        return

    def causal_norm_wrapper(norm_layer, x):  # imported by name: patched before video_vae / attn_video_vae bind it
        in_dt = x.dtype
        y = orig(norm_layer, x)
        if want("VAE norm", (type(norm_layer).__name__, in_dt, y.dtype)):
            log(f"check: VAE norm: {type(norm_layer).__name__} gets {dt(in_dt)}, returns {dt(y.dtype)}"
                + (f", cast back to {dt(in_dt)}" if S.vae_ac and y.dtype != in_dt else "")
                + f", {autocast_state(x.device.type)}")
        return y.to(in_dt) if S.vae_ac and y.dtype != in_dt else y  # ByteDance: x.to(input_dtype)

    mod.causal_norm_wrapper = causal_norm_wrapper
    S.patched.append(f"{mod.__name__}.causal_norm_wrapper" + (" (output cast to the input dtype)" if S.vae_ac else ""))


def patch_conv3d(mod):
    if not hasattr(mod, "NVIDIA_CONV3D_MEMORY_BUG_WORKAROUND"):
        return
    det = mod.NVIDIA_CONV3D_MEMORY_BUG_WORKAROUND
    if S.conv3d in ("0", "1"):
        mod.NVIDIA_CONV3D_MEMORY_BUG_WORKAROUND = S.conv3d == "1"
        log(f"Conv3d workaround: detected {det}, forced {mod.NVIDIA_CONV3D_MEMORY_BUG_WORKAROUND}")
        S.patched.append(f"{mod.__name__}.NVIDIA_CONV3D_MEMORY_BUG_WORKAROUND={S.conv3d == '1'}")
    else:
        check("Conv3d", "auto", f"workaround {det} (numz's detection)")


# ---------------------------------------------------------------- decode precision (NUM_DECODE, NUM_OUT32)

def decode_dtype():
    import torch
    return {"fp16": torch.float16, "fp32": torch.float32}.get(S.decode)


def set_vae_weights(vae, dtype):
    """The VAE's weights from its file as stored (numz's loader), converted to dtype (numz converts them
    to bfloat16 on load): (tensors loaded, model keys missing from the file, file keys not in the model).
    With vae None, only reads the file (CPU)."""
    import torch
    if S.vae_src is None:
        raise RuntimeError("numerics_patch: NUM_DECODE: the VAE checkpoint was not seen when it was loaded")
    if S.vae_raw is None:
        fn, path, debug = S.vae_src
        S.vae_raw = fn(path, torch.device("cpu"), debug)
    if vae is None:
        return 0, [], []
    vae.to(dtype)
    state = {k: v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v for k, v in S.vae_raw.items()}
    res = vae.load_state_dict(state, strict=False)
    return len(state), list(res.missing_keys), list(res.unexpected_keys)


def install_decode_hooks(vae):
    """Forward hooks on the decoder's modules: per module, the order of its first call, its calls,
    the non-finite values of its outputs and their min / max (tensors on the GPU: no sync per call)."""
    import torch
    stats, handles, order = {}, [], [0]
    for root_name in ("post_quant_conv", "decoder"):
        root = getattr(vae, root_name, None)
        if not isinstance(root, torch.nn.Module):
            continue
        for name, m in root.named_modules():
            def hook(mod, inp, out, __name=f"{root_name}.{name}" if name else root_name):
                for t in out if isinstance(out, (tuple, list)) else (out,):
                    if not (torch.is_tensor(t) and t.is_floating_point() and t.numel()):
                        continue
                    t = t.detach()
                    nf = torch.isfinite(t).logical_not_().sum()
                    lo, hi = torch.aminmax(t)
                    st = stats.get(__name)
                    if st is None:
                        stats[__name] = [order[0], 1, nf, lo.float(), hi.float()]
                    else:
                        st[1] += 1
                        st[2] = st[2] + nf
                        st[3], st[4] = torch.minimum(st[3], lo.float()), torch.maximum(st[4], hi.float())
                    order[0] += 1
            handles.append(m.register_forward_hook(hook))
    return handles, stats


def report_decode_hooks(handles, stats):
    for h in handles:
        h.remove()
    rows = sorted((first, name, calls, int(nf), float(lo), float(hi)) for name, (first, calls, nf, lo, hi) in stats.items())
    if not rows:
        log("decode hooks: no decoder output seen")
        return
    bad = [r for r in rows if r[3]]
    fin = [r for r in rows if math.isfinite(r[4]) and math.isfinite(r[5])]
    big = sorted(fin, key=lambda r: -max(-r[4], r[5]))[:5]
    log(f"decode hooks ({S.decode}): {len(rows)} modules, {sum(r[2] for r in rows)} outputs; non-finite values "
        f"{sum(r[3] for r in rows)} in {len(bad)} modules" + (f", first {bad[0][1]} ({bad[0][3]})" if bad else "")
        + "; largest |x|: " + ", ".join(f"{r[1]} {max(-r[4], r[5]):.6g}" for r in big)
        + (" (float16 ends at 65504)" if S.decode == "fp16" else ""))


def patch_infer(mod):
    cls = getattr(mod, "VideoDiffusionInfer", None)
    if cls is None or not (S.decode != "bf16" or S.dec_hooks or S.check):
        return
    orig = cls.vae_decode

    def vae_decode(self, latents, *a, **k):
        import torch
        dec, flags, hooks = decode_dtype(), None, None
        if dec is not None:
            p = next(self.vae.parameters())
            if p.dtype != dec:
                if S.vae_numz_dtype is None:
                    S.vae_numz_dtype = p.dtype
                n, missing, unexpected = set_vae_weights(self.vae, dec)
                log(f"decode {S.decode}: the VAE's weights loaded again from {Path(S.vae_src[1]).name} in "
                    f"{dt(dec)} ({n} tensors; model keys not in the file: {len(missing)}"
                    f"{' ' + ', '.join(missing[:4]) if missing else ''}; file keys not in the model: {len(unexpected)})")
            latents = [x.to(dec) for x in latents]  # exact: bfloat16 values fit float16 and float32
            if S.decode == "fp32":
                m, c = torch.backends.cuda.matmul, torch.backends.cudnn
                flags = (m.allow_tf32, c.allow_tf32)
                m.allow_tf32 = c.allow_tf32 = S.tf32 == "1"
        if S.dec_hooks:
            hooks = install_decode_hooks(self.vae)
        if S.check and latents:
            wd = {dt(p.dtype) for p in self.vae.parameters()}
            check("decode", (S.decode, latents[0].dtype),
                  f"vae_decode gets a {dt(latents[0].dtype)} latent; VAE weights {'/'.join(sorted(wd))}; TF32 matmul "
                  f"{torch.backends.cuda.matmul.allow_tf32}, cuDNN {torch.backends.cudnn.allow_tf32}; "
                  f"{autocast_state(latents[0].device.type)}")
        try:
            out = orig(self, latents, *a, **k)
        finally:
            if flags is not None:
                torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags
            if hooks is not None:
                report_decode_hooks(*hooks)
        ts = [s for s in out if torch.is_tensor(s) and s.numel()]
        if ts:
            nf = sum(int(torch.isfinite(s).logical_not_().sum()) for s in ts)
            rng = [torch.aminmax(s.detach()) for s in ts]
            log(f"decode {S.decode}: decoder output {dt(ts[0].dtype)} {tuple(ts[0].shape)}: {nf} non-finite values, "
                f"range [{min(float(r[0]) for r in rng):.6g}, {max(float(r[1]) for r in rng):.6g}]")
        return out

    cls.vae_decode = vae_decode
    S.patched.append(f"{mod.__name__}.VideoDiffusionInfer.vae_decode" + (f" ({S.decode} decode)" if S.decode != "bf16"
                                                                         else ""))


def restore_vae(runner):
    """After Phase 3: numz's bfloat16 VAE back (from its file), for any later encode."""
    vae = getattr(runner, "vae", None)
    if S.vae_numz_dtype is None or vae is None:
        return
    try:
        p = next(vae.parameters())
    except StopIteration:
        return
    if p.device.type != "meta" and p.dtype != S.vae_numz_dtype:
        set_vae_weights(vae, S.vae_numz_dtype)
        log(f"decode {S.decode}: numz's {dt(S.vae_numz_dtype)} VAE weights restored after Phase 3")


# ---------------------------------------------------------------- input preparation (NUM_PREP / RESIZE / PAD)

def exact_fp32(x):
    """float16/bfloat16 frames -> float32. 8-bit frames get their exact values back: round(x * 255)
    / 255 is the CLI's own float32 uint8 / 255 (float16 keeps 11 significant bits, bfloat16 8: both
    move k / 255 by less than half a step of 1 / 255), checked against x."""
    import torch
    if x.dtype == torch.float32:
        return x
    y = x.float()
    if x.dtype in (torch.float16, torch.bfloat16):
        q = torch.round(y * 255.0) / 255.0
        if torch.equal(q.to(x.dtype), x):
            once("exact", f"prep fp32: input frames restored to their exact 8-bit values (k / 255) in float32 "
                 f"from {dt(x.dtype)}")
            return q
        once("inexact", f"prep fp32: input frames ({dt(x.dtype)}) are not 8-bit values: plain float32 cast")
    return y


def patch_gen_phases(mod):
    if not (S.prep == "fp32" or S.out32 or S.check):
        return
    import torch
    if S.prep == "fp32":
        orig_prep = mod._prepare_video_batch

        def _prepare_video_batch(*a, **k):  # Phase 1 and Phase 4's colour reference
            return exact_fp32(orig_prep(*a, **k))

        mod._prepare_video_batch = _prepare_video_batch
        S.patched.append(f"{mod.__name__}._prepare_video_batch (float32 transform chain)")
    if S.prep == "fp32" or S.out32:
        orig_mt = mod.manage_tensor

        def manage_tensor(*a, **k):
            name = str(k.get("tensor_name", ""))
            if k.get("dtype") is not None:
                if S.prep == "fp32" and name.startswith("video_batch_"):  # Phase 1's frames to the GPU: keep float32
                    k["dtype"] = torch.float32
                elif S.out32 and S.in_decode is not None and name.startswith("upscaled_latent_"):
                    k["dtype"] = S.in_decode  # Phase 3's latent goes to vae_decode in numz's dtype
                elif S.out32 and name.startswith("sample_"):  # Phase 3's write, Phase 4's work and write-back
                    k["dtype"] = torch.float32
            return orig_mt(*a, **k)

        mod.manage_tensor = manage_tensor
        S.patched.append(f"{mod.__name__}.manage_tensor (" + ", ".join(
            x for x in ("float32 transform chain" if S.prep == "fp32" else "",
                        "float32 frames after the decode" if S.out32 else "") if x) + ")")
    orig_dab, orig_pp = mod.decode_all_batches, mod.postprocess_all_batches

    def decode_all_batches(runner, *a, **k):
        ctx = k["ctx"] if "ctx" in k else (a[0] if a else None)
        numz_dt = ctx.get("compute_dtype") if isinstance(ctx, dict) else None
        swap = S.out32 and numz_dt is not None
        if swap:  # final_video is allocated in compute_dtype
            S.in_decode, ctx["compute_dtype"] = numz_dt, torch.float32
        try:
            res = orig_dab(runner, *a, **k)
        finally:
            if swap:
                ctx["compute_dtype"], S.in_decode = numz_dt, None
        if S.decode != "bf16":
            restore_vae(runner)
        fv = res.get("final_video") if isinstance(res, dict) else None
        if fv is not None:
            check("output", "phase 3", f"Phase 3 wrote final_video as {dt(fv.dtype)} {tuple(fv.shape)} on {fv.device}")
        return res

    def postprocess_all_batches(*a, **k):
        res = orig_pp(*a, **k)
        fv = res.get("final_video") if isinstance(res, dict) else None
        if fv is not None and fv.numel() and want("output", ("phase 4", fv.dtype)):
            log(f"check: output: Phase 4 left final_video as {dt(fv.dtype)} {tuple(fv.shape)}, range "
                f"[{float(fv.min()):.6g}, {float(fv.max()):.6g}]")
        return res

    mod.decode_all_batches, mod.postprocess_all_batches = decode_all_batches, postprocess_all_batches
    if S.out32:
        S.patched.append(f"{mod.__name__}.decode_all_batches (float32 final_video)")


def ffmpeg_bin():
    return os.environ.get("NUM_FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"


def side_size(h, w, size, max_size=0, downsample_only=False):
    """Output (h, w) of numz's SideResize: torchvision's short side = size, then the max_size step."""
    if downsample_only and min(w, h) < size:
        size = min(w, h)
    try:
        from torchvision.transforms.functional import _compute_resized_output_size
        oh, ow = _compute_resized_output_size((h, w), [size], None)
    except ImportError:
        short, long = (w, h) if w <= h else (h, w)
        ns, nl = size, int(size * long / short)
        ow, oh = (ns, nl) if w <= h else (nl, ns)
    if max_size > 0 and max(oh, ow) > max_size:
        s = max_size / max(oh, ow)
        oh, ow = round(oh * s), round(ow * s)
    return oh, ow


def zimg_resize(img, oh, ow, filt):
    """[T, 3, H, W] RGB float tensor -> [T, 3, oh, ow] through ffmpeg's zscale, float32 in and out."""
    import numpy as np
    import torch
    t, c, h, w = img.shape
    if c != 3:
        raise RuntimeError(f"numerics_patch: the zimg resize expects RGB frames, got {c} channels")
    planes = np.ascontiguousarray(img.detach().to("cpu", torch.float32).numpy()[:, [1, 2, 0]], dtype="<f4")
    cmd = [ffmpeg_bin(), "-hide_banner", "-nostdin", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gbrpf32le",
           "-s", f"{w}x{h}", "-framerate", "1", "-i", "-", "-map", "0:v:0", "-fps_mode", "passthrough",
           "-vf", f"zscale=w={ow}:h={oh}:filter={filt}", "-f", "rawvideo", "-pix_fmt", "gbrpf32le", "-"]
    r = subprocess.run(cmd, input=planes.tobytes(), capture_output=True)
    want_bytes = t * 3 * oh * ow * 4
    if r.returncode or len(r.stdout) != want_bytes:
        raise RuntimeError(f"numerics_patch: zscale failed (exit {r.returncode}, {len(r.stdout)} of {want_bytes} "
                           f"bytes): {r.stderr.decode(errors='replace').strip()} -- {' '.join(cmd)}")
    y = np.frombuffer(r.stdout, dtype="<f4").reshape(t, 3, oh, ow)[:, [2, 0, 1]]  # G, B, R -> R, G, B
    return torch.from_numpy(np.ascontiguousarray(y)).to(device=img.device, dtype=img.dtype)


class Resize:
    """numz's SideResize with another kernel, to the same size."""

    def __init__(self, side, kind):
        self.size, self.max_size = side.size, side.max_size
        self.downsample_only, self.interpolation = side.downsample_only, side.interpolation
        self.kind = kind

    def __call__(self, img):
        h, w = img.shape[-2:]
        if self.kind == "tv-noaa":  # SideResize.__call__ with antialias=False
            from torchvision.transforms import functional as TVF
            size = min(w, h) if self.downsample_only and min(w, h) < self.size else self.size
            out = TVF.resize(img, size, self.interpolation, antialias=False)
            if self.max_size > 0:
                rh, rw = out.shape[-2:]
                if max(rh, rw) > self.max_size:
                    s = self.max_size / max(rh, rw)
                    out = TVF.resize(out, (round(rh * s), round(rw * s)), self.interpolation, antialias=False)
            return out
        oh, ow = side_size(h, w, self.size, self.max_size, self.downsample_only)
        if (oh, ow) == (h, w):
            return img
        return zimg_resize(img, oh, ow, self.kind.split("-", 1)[1])


class Pad:
    """numz's DivisiblePad (bottom and right) with another torch padding mode, a grey fill, or N
    more rows (and columns where it pads columns) of zeros after it (NUM_PAD=black+N,
    reflect[>=R]+black+N: at least R reflected rows before the zeros)."""

    def __init__(self, div, mode):
        self.hf, self.wf, self.mode = div.height_factor, div.width_factor, mode
        self.extra = pad_extra(mode)

    def __call__(self, img):
        import torch
        h, w = img.shape[-2:]
        ph, pw = (self.hf - h % self.hf) % self.hf, (self.wf - w % self.wf) % self.wf
        if self.extra:
            mode, n, rmin = self.extra
            ph += 16 * max(0, -(-(rmin - ph) // 16))  # the fewest rows >= rmin reaching a multiple of 16
            if pw:
                pw += 16 * max(0, -(-(rmin - pw) // 16))
            if ph or pw:
                img = torch.nn.functional.pad(img, (0, pw, 0, ph), mode=mode)  # constant: zeros
            return torch.nn.functional.pad(img, (0, n if pw else 0, 0, n), mode="constant", value=0.0)
        if ph == 0 and pw == 0:
            return img
        if self.mode == "grey":
            return torch.nn.functional.pad(img, (0, pw, 0, ph), mode="constant", value=0.5)
        return torch.nn.functional.pad(img, (0, pw, 0, ph), mode=self.mode)


class Probe:
    """NUM_CHECK: logs what a step of the transform chain received and returned."""

    def __init__(self, kind, inner, label):
        self.kind, self.inner, self.label = kind, inner, label

    def __call__(self, x):
        y = self.inner(x) if self.inner is not None else x
        where = f"{dt(x.dtype)} on {x.device.type}"
        if not want(self.kind, where):
            return y
        if self.kind == "resize":
            msg = (f"{self.label}: {tuple(x.shape)} {where} -> {tuple(y.shape)} {dt(y.dtype)}"
                   + (" (size already right: unchanged)" if y is x else ""))
        elif self.kind == "pad":
            (h, w), (ph, pw) = x.shape[-2:], y.shape[-2:]
            msg = f"{self.label}: {h}x{w} -> {ph}x{pw} (bottom {ph - h}, right {pw - w}) {dt(y.dtype)}"
            if ph > h:
                msg += (f"; padded rows mean {y[..., h:, :].float().mean().item():.4g}, the rows above "
                        f"{y[..., 2 * h - ph:h, :].float().mean().item():.4g}")
        else:
            msg = (f"output {tuple(y.shape)} {dt(y.dtype)} on {y.device.type}, "
                   f"range [{y.min().item():.4g}, {y.max().item():.4g}]")
        log(f"check: {self.kind}: {msg}")
        return y


def patch_gen_utils(mod):
    if S.resize == "tv-aa" and S.pad == "zero" and not S.check:
        return
    orig = mod.prepare_video_transforms

    def prepare_video_transforms(resolution, max_resolution=0, debug=None):
        comp = orig(resolution, max_resolution, debug)
        ts = comp.transforms
        names = [type(t).__name__ for t in ts]
        if len(ts) != 5 or names[0] != "SideResize" or names[2] != "DivisiblePad":
            raise RuntimeError(f"numerics_patch: unexpected transform chain {names}")
        if S.resize != "tv-aa":
            ts[0] = Resize(ts[0], S.resize)
        if S.pad != "zero":
            ts[2] = Pad(ts[2], S.pad)
        if S.check:
            ts[0] = Probe("resize", ts[0], S.resize)
            ts[2] = Probe("pad", ts[2], S.pad)
            ts.append(Probe("chain", None, ""))
        return comp

    mod.prepare_video_transforms = prepare_video_transforms
    S.patched.append(f"{mod.__name__}.prepare_video_transforms (resize {S.resize}, pad {S.pad})")


# ---------------------------------------------------------------- import hooks

HOOKS = (
    (".core.model_loader", patch_loader),
    (".core.generation_utils", patch_gen_utils),
    (".core.generation_phases", patch_gen_phases),
    (".core.infer", patch_infer),
    (".dit_7b.rope", patch_rope), (".dit_3b.rope", patch_rope),
    (".dit_7b.attention", patch_attention), (".dit_3b.attention", patch_attention),
    (".video_vae_v3.modules.attn_video_vae", patch_vae),
    (".video_vae_v3.modules.causal_inflation_lib", patch_causal_lib),
    (".dit_7b.normalization", patch_norm), (".dit_3b.normalization", patch_norm),
)


class Finder(importlib.abc.MetaPathFinder):
    """Lets the module load normally, then applies the patch right after it executes."""

    def find_spec(self, name, path, target=None):
        hook = next((h for suffix, h in HOOKS if ("." + name).endswith(suffix)), None)
        if hook is None:
            return None
        for f in sys.meta_path:
            if f is self or not hasattr(f, "find_spec"):
                continue
            spec = f.find_spec(name, path, target)
            if spec is not None:
                break
        else:
            return None
        loader = spec.loader
        orig_exec = loader.exec_module

        def exec_module(module):
            orig_exec(module)
            hook(module)
        loader.exec_module = exec_module
        return spec


# ---------------------------------------------------------------- the CLI (TF32, NUM_CC_EXTRA)

def apply_tf32():
    import torch
    m, c = torch.backends.cuda.matmul, torch.backends.cudnn
    if S.tf32 == "1":
        m.allow_tf32 = True
        c.allow_tf32 = True
        log(f"TF32: matmul {m.allow_tf32}, cuDNN {c.allow_tf32} (forced on)")
    else:
        check("TF32", "flags", f"matmul {m.allow_tf32}, cuDNN {c.allow_tf32} (torch's defaults)")


def write_extra(cc, frames):
    from fractions import Fraction
    f = S.file
    if f is None:
        log(f"extra {cc}: no current input file, not written")
        return
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ffv1_out  # noqa: E402  (a module import only defines functions and reads FFV1_OUT_*)
    arr = frames.float().cpu().numpy()
    T, H, W, _ = arr.shape
    ws = f["writers"].get(cc)
    if ws is None:
        fps = ffv1_out.probe_fps(f["input"]) or Fraction(24000, 1001)
        ffv1_out.S.files_done = S.files_done  # ffv1_out.py names the files after the first by input stem
        paths = [(fmt, p.with_name(p.name[:-4] + f".cc-{cc}.mkv"))
                 for fmt, p in ffv1_out.out_paths(f["input"], f["cli_output"])]
        ws = f["writers"][cc] = [ffv1_out.Writer(p, fmt, W, H, fps, 16) for fmt, p in paths]
        f["received"][cc] = 0
    for frame in arr:
        f["received"][cc] += 1
        if f["received"][cc] <= f["drop"]:
            continue
        planes = {}
        for w in ws:
            if w.bits not in planes:
                planes[w.bits] = ffv1_out.to_planar(frame, w.bits)
            w.write(planes[w.bits])
    log(f"extra colour correction {cc}: {T} frames ({f['drop']} dropped at the start) -> "
        f"{', '.join(w.path for w in ws)}")


def close_extras():
    f, S.file = S.file, None
    if f is None:
        return
    S.files_done += 1
    for cc, ws in f["writers"].items():
        for w in ws:
            rc = w.close()
            log(f"{'FAILED ' if rc else ''}{w.path}: {w.frames} frames ({cc}, {w.pix_fmt})")


def match_ties(source, reference, device, stats=True):
    """NUM_CC_EXTRA=labties: numz's _histogram_matching_channel (color_fix.py:477-521), the same sorts
    and quantile mapping, except that every run of equal source values gets the mean of the reference
    values over its ranks (float64 sums). A value whose ranks all hold one reference value (a value
    that occurs once, for one) keeps numz's exact result."""
    import torch
    shape = source.shape
    src_sorted, src_idx = torch.sort(source.flatten())
    ref_sorted, _ = torch.sort(reference.flatten())
    n, m = src_sorted.numel(), ref_sorted.numel()
    if n == m:
        matched = ref_sorted
    else:  # numz's quantile mapping between unequal sizes
        idx = (torch.linspace(0, 1, n, device=device) * (m - 1)).long()
        idx.clamp_(0, m - 1)
        matched = ref_sorted[idx]
        del idx
    del ref_sorted
    _, counts = torch.unique_consecutive(src_sorted, return_counts=True)
    del src_sorted
    ends = counts.cumsum(0)
    starts = ends - counts
    csum = torch.zeros(n + 1, dtype=torch.float64, device=matched.device)
    torch.cumsum(matched.double(), 0, out=csum[1:])
    means = ((csum[ends] - csum[starts]) / counts.double()).to(matched.dtype)
    del csum
    lo, hi = matched[starts], matched[ends - 1]  # matched is sorted: a value's lowest and highest
    one = hi == lo
    means[one] = lo[one]
    out = torch.repeat_interleave(means, counts, output_size=n)
    if stats:
        tie_stats(counts, hi - lo, (matched - out).abs(), n)
    del matched, means, lo, hi, one, starts, ends
    return out[torch.argsort(src_idx)].reshape(shape)


def tie_stats(counts, spread, dev, n):
    """Log for one matching (numz's calls: a*, b*, then L*, once per batch): the share of values tied,
    the spread numz's mapping gives a tied value, and how far numz's values are from labties' ones."""
    tied = counts > 1
    w = counts.double() / n
    sp = spread.double()
    share = lambda c: float(w[tied & (sp > c)].sum()) * 100  # noqa: E731
    chan = ("a*", "b*", "L*")[S.ties_calls % 3]
    log(f"labties batch {S.ties_calls // 3 + 1} {chan}: {n:,} values, {int(counts.numel()):,} distinct, "
        f"{float(w[tied].sum()) * 100:.1f}% in tied values (largest {float(counts.max()) / n * 100:.2f}%); "
        f"numz spreads a tied pixel's value over {float((w * sp).sum()):.3f} on average, max {float(sp.max()):.3f}; "
        f"pixels in values spread over > 0.5 / 1 / 2 / 4: {share(0.5):.2f}% / {share(1):.2f}% / {share(2):.3f}% / "
        f"{share(4):.4f}%; numz - labties: mean |d| {float(dev.double().mean()):.4f}, max {float(dev.max()):.3f}")
    S.ties_calls += 1


@contextlib.contextmanager
def cc_variant(cc):
    """A NUM_CC_EXTRA variant: numz's Phase 4 runs its method with one function swapped, restored after."""
    if cc not in CC_VARIANTS:
        yield
        return
    mod = next((m for name, m in list(sys.modules.items())
                if name.endswith("utils.color_fix") and hasattr(m, "_histogram_matching_channel")), None)
    if mod is None:
        raise RuntimeError("numerics_patch: labties: numz's color_fix module is not loaded")
    orig, mod._histogram_matching_channel = mod._histogram_matching_channel, match_ties
    S.ties_calls = 0
    try:
        yield
    finally:
        mod._histogram_matching_channel = orig


def render_extras(orig_pp, a, k):
    ctx = k["ctx"] if "ctx" in k else a[0]
    if not ctx.get("decode_batch_info") or ctx.get("final_video") is None:
        return
    for cc in S.cc_extra:
        c = dict(ctx)
        c["final_video"] = ctx["final_video"].clone()
        c["decode_batch_info"] = list(ctx["decode_batch_info"])
        if c.get("video_transform") is not None:  # Phase 4's cleanup clears the transforms in place
            c["video_transform"] = copy.deepcopy(ctx["video_transform"])
        with cc_variant(cc):
            c = call_with(orig_pp, a, k, ctx=c, color_correction=CC_VARIANTS.get(cc, cc))
        write_extra(cc, c["final_video"])
        del c


def patch_cli(g):
    """g: the globals of the running inference_cli.py."""
    S.cli_patched = True
    apply_tf32()
    if not S.cc_extra:
        return
    orig_enc, orig_pp, orig_psf = g["encode_all_batches"], g["postprocess_all_batches"], g["process_single_file"]

    def encode_all_batches(*a, **k):
        if k.get("color_correction") == "none":  # Phase 1 then keeps the batch indices Phase 4 needs
            k = dict(k, color_correction=CC_VARIANTS.get(S.cc_extra[0], S.cc_extra[0]))
        return orig_enc(*a, **k)

    def postprocess_all_batches(*a, **k):
        render_extras(orig_pp, a, k)
        return orig_pp(*a, **k)

    def process_single_file(input_path, args, device_list, output_path=None, *a, **k):
        input_type = g["get_input_type"](input_path)
        out = output_path
        if out is None:  # the output path as process_single_file (and ffv1_out.py) resolve it
            out = g["generate_output_path"](input_path, args.output_format, input_type=input_type)
        elif not Path(out).suffix or (args.output_format == "png" and input_type != "image"):
            out = g["generate_output_path"](input_path, args.output_format, output_dir=out, input_type=input_type)
        if len(device_list) > 1:
            log("NUM_CC_EXTRA: multi-GPU run, the extra colour corrections are not rendered")
        drop = int(os.environ.get("FFV1_OUT_DROP", (args.prepend_frames or 0) if len(device_list) <= 1 else 0))
        S.file = {"input": input_path, "cli_output": out, "writers": {}, "received": {}, "drop": drop}
        try:
            return orig_psf(input_path, args, device_list, output_path, *a, **k)
        finally:
            close_extras()

    g["encode_all_batches"], g["postprocess_all_batches"] = encode_all_batches, postprocess_all_batches
    g["process_single_file"] = process_single_file
    S.patched.append(f"inference_cli: extra colour corrections {', '.join(S.cc_extra)}"
                     + ("; labties: color_fix._histogram_matching_channel swapped while its Phase 4 runs"
                        if "labties" in S.cc_extra else ""))


def install():
    """Module patches at import (Finder); the CLI's own globals when it parses its arguments
    (every function of inference_cli.py exists by then), wherever this wrapper sits in a chain."""
    sys.meta_path.insert(0, Finder())
    orig = argparse.ArgumentParser.parse_args

    def parse_args(self, *a, **k):
        res = orig(self, *a, **k)
        if not S.cli_patched:
            fr = sys._getframe(1)
            while fr is not None:
                g = fr.f_globals
                if "postprocess_all_batches" in g and "process_single_file" in g:
                    patch_cli(g)
                    break
                fr = fr.f_back
        return res

    argparse.ArgumentParser.parse_args = parse_args


def at_exit():
    close_extras()
    log("patched: " + ("; ".join(S.patched) if S.patched else "nothing (numz defaults)"))
    if S.check:
        missing = [p for p in PROBES if p not in S.probed]
        log("check: no call seen for: " + ", ".join(missing) if missing else "check: every probe fired")


# ---------------------------------------------------------------- self test (CPU)

def kernel(name, x):
    """zimg's spline36 and 3-tap lanczos."""
    x = abs(x)
    if name == "spline36":
        if x < 1:
            return ((13 / 11 * x - 453 / 209) * x - 3 / 209) * x + 1
        if x < 2:
            x -= 1
            return ((-6 / 11 * x + 270 / 209) * x - 156 / 209) * x
        if x < 3:
            x -= 2
            return ((1 / 11 * x - 45 / 209) * x + 26 / 209) * x
        return 0.0
    if x == 0:
        return 1.0
    return 3 * math.sin(math.pi * x) * math.sin(math.pi * x / 3) / (math.pi * x) ** 2 if x < 3 else 0.0


def selftest():
    import numpy as np
    import torch
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TVF
    ok = True

    def report(name, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        print(f"{'ok  ' if good else 'FAIL'} {name}{': ' + detail if detail else ''}")

    # output sizes as numz computes them
    report("size 960x540 -> 1080", side_size(540, 960, 1080) == (1080, 1920), str(side_size(540, 960, 1080)))
    report("size 1920x1080 -> 1080", side_size(1080, 1920, 1080) == (1080, 1920))
    report("size 1920x1080 -> 1080, max 1280", side_size(1080, 1920, 1080, 1280) == (720, 1280))
    # exact 8-bit values back from float16 and bfloat16
    rng = np.random.default_rng(0)
    k8 = rng.integers(0, 256, size=(2, 3, 32, 48), dtype=np.uint8)
    k8[0, 0, 0, :] = np.arange(48) * 5 + 15  # covers the top of the range
    ref = torch.from_numpy(k8.astype(np.float32) / 255.0)  # the CLI's frames before its float16 cast
    for d in (torch.float16, torch.bfloat16):
        got = exact_fp32(ref.to(d))
        report(f"8-bit values back from {dt(d)}", torch.equal(got, ref), f"max |diff| {(got - ref).abs().max().item():.3g}")
    noisy = (ref + 1e-3).to(torch.float16)
    report("non-8-bit input: plain float32 cast", torch.equal(exact_fp32(noisy), noisy.float()))
    # zimg: kernel, plane order, size, overshoot kept, same size exact
    for filt in ("spline36", "lanczos"):
        W, OW = 40, 80
        row = rng.random(W).astype(np.float32)
        img = torch.from_numpy(np.ascontiguousarray(np.broadcast_to(row, (1, 3, 8, W))))
        out = zimg_resize(img, 16, OW, filt)[0, 0, 8].double().numpy()
        worst = 0.0
        for j in range(OW):
            x = (j + 0.5) * W / OW - 0.5  # centre alignment, as torchvision's align_corners=False
            lo = math.floor(x) - 2
            if lo < 0 or lo + 5 >= W:
                continue
            ws = [kernel(filt, x - i) for i in range(lo, lo + 6)]
            want_v = sum(wi * float(row[i]) for wi, i in zip(ws, range(lo, lo + 6))) / sum(ws)
            worst = max(worst, abs(out[j] - want_v))
        report(f"zimg {filt}: matches the {filt} kernel, centre-aligned (interior)", worst < 1e-5, f"max |diff| {worst:.2e}")
        const = torch.empty(2, 3, 18, 32)
        const[:, 0], const[:, 1], const[:, 2] = 0.75, 0.25, 0.5
        out = zimg_resize(const, 36, 64, filt)
        report(f"zimg {filt}: size and plane order", tuple(out.shape) == (2, 3, 36, 64) and
               (out - const[:, :, :1, :1]).abs().max().item() < 1e-5,
               f"channel means {[round(out[:, i].mean().item(), 6) for i in range(3)]}")
        step = torch.zeros(1, 3, 16, 32)
        step[..., 16:] = 1.0
        out = zimg_resize(step, 32, 64, filt)
        report(f"zimg {filt}: ringing kept outside [0, 1]", out.max().item() > 1.0 and out.min().item() < 0.0,
               f"range [{out.min().item():.4f}, {out.max().item():.4f}]")
        rnd = torch.rand(2, 3, 16, 32)
        same = zimg_resize(rnd, 16, 32, filt)
        report(f"zimg {filt}: same size gives the float32 input back (no integer step)", torch.equal(same, rnd),
               f"max |diff| {(same - rnd).abs().max().item():.3g}")
    # torchvision's two bicubic kernels differ when upsampling (a = -0.5 vs -0.75)
    img = torch.rand(1, 3, 27, 48)
    aa = TVF.resize(img, 54, InterpolationMode.BICUBIC, antialias=True)
    noaa = TVF.resize(img, 54, InterpolationMode.BICUBIC, antialias=False)
    report("tv-aa and tv-noaa differ when upsampling", (aa - noaa).abs().max().item() > 1e-3,
           f"max |diff| {(aa - noaa).abs().max().item():.4f}")

    class Div:
        height_factor = width_factor = 16
    x = torch.rand(2, 3, 1080, 1912)
    for mode in ("reflect", "replicate"):
        for d in (torch.float32, torch.bfloat16, torch.float16):
            xd = x.to(d)
            y = Pad(Div, mode)(xd)
            row = y[..., 1078, :1912] if mode == "reflect" else y[..., 1079, :1912]
            good = (tuple(y.shape) == (2, 3, 1088, 1920) and torch.equal(y[..., :1080, :1912], xd)
                    and torch.equal(y[..., 1080, :1912], row))
            report(f"pad {mode} {dt(d)}: bottom and right, same amounts as DivisiblePad", good, str(tuple(y.shape)))
    # grey where DivisiblePad pads; N more rows of zeros (columns too only where it pads columns)
    x2 = torch.rand(2, 3, 1080, 1920)
    for d in (torch.float32, torch.bfloat16, torch.float16):
        xd, x2d = x.to(d), x2.to(d)
        y = Pad(Div, "grey")(xd)
        good = (tuple(y.shape) == (2, 3, 1088, 1920) and torch.equal(y[..., :1080, :1912], xd)
                and bool((y[..., 1080:, :] == 0.5).all()) and bool((y[..., 1912:] == 0.5).all()))
        report(f"pad grey {dt(d)}: 0.5 where DivisiblePad pads", good, str(tuple(y.shape)))
        for mode in ("black+16", "reflect+black+16"):
            base = torch.nn.functional.pad(xd, (0, 8, 0, 8), mode="constant" if mode == "black+16" else "reflect")
            y = Pad(Div, mode)(xd)
            good = (tuple(y.shape) == (2, 3, 1104, 1936) and torch.equal(y[..., :1088, :1920], base)
                    and bool((y[..., 1088:, :] == 0).all()) and bool((y[..., 1920:] == 0).all()))
            y2 = Pad(Div, mode)(x2d)
            good2 = (tuple(y2.shape) == (2, 3, 1104, 1920) and torch.equal(y2[..., :1080, :], x2d)
                     and torch.equal(y2[..., 1080:1088, :], torch.nn.functional.pad(
                         x2d, (0, 0, 0, 8), mode="constant" if mode == "black+16" else "reflect")[..., 1080:, :])
                     and bool((y2[..., 1088:, :] == 0).all()))
            report(f"pad {mode} {dt(d)}: DivisiblePad's amounts by {'zeros' if mode == 'black+16' else 'reflection'}, "
                   f"then 16 rows of zeros, 16 columns only where it pads columns", good and good2,
                   f"{tuple(y.shape)}, {tuple(y2.shape)}")
    try:
        pad_extra("black+8")
        report("NUM_PAD=black+8 rejected (N not a multiple of 16)", False)
    except SystemExit:
        report("NUM_PAD=black+8 rejected (N not a multiple of 16)", True)
    report("NUM_PAD parsing", pad_extra("black+16") == ("constant", 16, 0) and pad_extra("reflect+black+32") ==
           ("reflect", 32, 0) and pad_extra("reflect>=8+black+16") == ("reflect", 16, 8)
           and pad_extra("reflect") is None and pad_extra("grey") is None)
    # reflect>=8+black+16: 1080 rows -> 8 reflected (= reflect+black+16), 720 -> 16 reflected; then 16 zeros
    for d in (torch.float32, torch.bfloat16, torch.float16):
        xd = x.to(d)
        same = torch.equal(Pad(Div, "reflect>=8+black+16")(xd), Pad(Div, "reflect+black+16")(xd))
        x7 = torch.rand(2, 3, 720, 1280).to(d)
        y7 = Pad(Div, "reflect>=8+black+16")(x7)
        good7 = (tuple(y7.shape) == (2, 3, 752, 1280) and torch.equal(y7[..., :720, :], x7)
                 and torch.equal(y7[..., 720:736, :],
                                 torch.nn.functional.pad(x7, (0, 0, 0, 16), mode="reflect")[..., 720:, :])
                 and bool((y7[..., 736:, :] == 0).all()))
        report(f"pad reflect>=8+black+16 {dt(d)}: = reflect+black+16 at 1080 rows; 16 reflected, then 16 zeros, "
               f"at 720", same and good7, str(tuple(y7.shape)))
    # NUM_NORM=bf16 on stand-ins for numz's CustomRMSNorm / CustomLayerNorm, under CPU autocast
    import types

    class RMS(torch.nn.Module):  # numz's CustomRMSNorm.forward
        def __init__(self):
            super().__init__()
            self.normalized_shape, self.eps = torch.Size([64]), 1e-5
            self.weight = torch.nn.Parameter(1 + 0.1 * torch.randn(64))

        def forward(self, input):
            return input / torch.sqrt(input.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight

    class LN(torch.nn.Module):  # numz's CustomLayerNorm.forward
        def __init__(self):
            super().__init__()
            self.normalized_shape, self.eps = torch.Size([64]), 1e-5
            self.weight = torch.nn.Parameter(1 + 0.1 * torch.randn(64))
            self.bias = torch.nn.Parameter(0.1 * torch.randn(64))

        def forward(self, input):
            return torch.nn.functional.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)

    saved = S.norm, S.check, list(S.patched)
    S.norm, S.check = "bf16", False
    torch.manual_seed(0)
    patch_norm(types.SimpleNamespace(__name__="selftest.dit_7b.normalization", CustomRMSNorm=RMS, CustomLayerNorm=LN))
    xin = 3 * torch.randn(4, 64)
    bf = torch.bfloat16
    for cls, rms in ((RMS, True), (LN, False)):
        m = cls()
        with torch.no_grad():
            ref32 = m(xin)  # autocast off: numz's own forward, float32
            with torch.autocast("cpu", dtype=bf):
                y = m(xin)
            xb = xin.to(bf).float()  # Apex written out: bfloat16 in, float32 math, rounded, bfloat16 affine
            c = xb if rms else xb - xb.mean(dim=-1, keepdim=True)
            want_y = (c * torch.rsqrt(c.pow(2).mean(dim=-1, keepdim=True) + 1e-5)).to(bf) * m.weight.to(bf)
            if not rms:
                want_y = want_y + m.bias.to(bf)
        err = (y.float() - ref32).abs().max().item()
        report(f"norm bf16 {cls.__name__}: numz's float32 without autocast; under autocast bfloat16 out, "
               f"Apex's rounding order, close to float32",
               ref32.dtype == torch.float32 and y.dtype == bf and torch.equal(y, want_y)
               and err <= 2 ** -6 * ref32.abs().max().item(), f"max |diff| to float32 {err:.3g}")
    S.norm, S.check, S.patched[:] = saved
    # NUM_DECODE: the VAE's weights from the file's own values; the decode hooks (float16 overflow)
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
    raw = {k: (3 * torch.randn_like(v)).to(torch.float16) for k, v in model.state_dict().items()}
    saved = S.vae_src, S.vae_raw
    S.vae_src, S.vae_raw = (lambda path, dev, dbg: dict(raw), "selftest.safetensors", None), None
    model.to(torch.bfloat16)
    n, missing, unexpected = set_vae_weights(model, torch.float32)
    sd = model.state_dict()
    good = (n == len(raw) and not missing and not unexpected
            and all(sd[k].dtype == torch.float32 and torch.equal(sd[k], v.float()) for k, v in raw.items()))
    set_vae_weights(model, torch.bfloat16)
    sd = model.state_dict()
    good2 = all(sd[k].dtype == bf and torch.equal(sd[k], v.to(bf)) for k, v in raw.items())
    report("decode weights: the file's values in float32 (exact), then in bfloat16 as numz loads them", good and good2)
    S.vae_src, S.vae_raw = saved

    class Gain(torch.nn.Module):
        def forward(self, x):
            return x * 1e4

    vae = torch.nn.Module()
    vae.decoder = torch.nn.Sequential(Gain(), Gain())
    handles, stats = install_decode_hooks(vae)
    with torch.no_grad():
        vae.decoder(torch.full((2, 3), 2.0, dtype=torch.float16))  # 2e4, then 2e8: inf in float16
    rows = {name: (calls, int(nf), float(hi)) for name, (first, calls, nf, lo, hi) in stats.items()}
    report_decode_hooks(handles, stats)
    report("decode hooks: non-finite outputs counted per module (float16 overflow)",
           rows.get("decoder.0") == (1, 0, 20000.0) and rows.get("decoder.1", (0, 0))[:2] == (1, 6)
           and rows.get("decoder", (0, 0))[:2] == (1, 6) and not vae.decoder[0]._forward_hooks, str(rows))
    # NUM_CC_EXTRA=labties: numz's rank matching where no value is tied; one value per tied value
    def numz_match(source, reference):  # color_fix.py:491-521, written out (sorts, quantiles, inverse)
        src_sorted, src_idx = torch.sort(source.flatten())
        ref_sorted, _ = torch.sort(reference.flatten())
        n, m = src_sorted.numel(), ref_sorted.numel()
        if n != m:
            idx = (torch.linspace(0, 1, n) * (m - 1)).long()
            ref_sorted = ref_sorted[idx.clamp_(0, m - 1)]
        return ref_sorted[torch.argsort(src_idx)].reshape(source.shape)

    g = torch.Generator().manual_seed(0)
    cpu = torch.device("cpu")
    src = torch.randn(3, 40, 50, generator=g, dtype=torch.float64).float()
    for m in (6000, 9000):
        ref = 20 * torch.randn(m, generator=g)
        report(f"labties: no tied value -> numz's matching exactly ({src.numel()} vs {m} reference values)",
               src.unique().numel() == src.numel() and torch.equal(match_ties(src, ref, cpu, stats=False),
                                                                   numz_match(src, ref)))
    src = torch.randint(0, 25, (3, 40, 50), generator=g).float() * 0.37
    ref = 20 * torch.randn(src.numel(), generator=g)
    got = match_ties(src, ref, cpu, stats=False)
    flat, out = src.flatten().numpy(), got.flatten().double().numpy()
    rs = np.sort(ref.double().numpy())
    good, worst, r0 = True, 0.0, 0
    for v in np.unique(flat):
        sel = flat == v
        want_v = rs[r0:r0 + int(sel.sum())].mean()
        r0 += int(sel.sum())
        good &= bool(np.unique(out[sel]).size == 1)
        worst = max(worst, abs(out[sel][0] - want_v))
    report("labties: each tied value gets one value, the mean of the reference over its ranks",
           good and worst < 1e-5 and r0 == flat.size, f"max |diff| {worst:.2e}")
    numz = numz_match(src, ref).flatten().numpy()
    spread = max(float(numz[flat == v].max() - numz[flat == v].min()) for v in np.unique(flat))
    report("labties: numz's matching spreads the same tied values (written-out numz)", spread > 1.0,
           f"widest spread {spread:.2f}")
    ref2 = ref.clone()
    ref2[torch.argsort(ref)[:300]] = -50.0  # the 300 lowest ranks hold one value
    src2 = src.clone().flatten()
    src2[torch.argsort(src2)[:250]] = -1.0  # the lowest source value, 250 times: inside that run
    got2 = match_ties(src2, ref2, cpu, stats=True)  # logs one tie line (a*)
    report("labties: a tied value whose ranks hold one reference value keeps it exactly",
           bool((got2[src2 == -1.0] == -50.0).all()))
    cwd_path = os.getcwd()
    sys.path.insert(0, cwd_path)
    try:
        from src.utils import color_fix  # numz's own function, when run from the checkout
    except Exception as e:  # noqa: BLE001
        report("labties vs numz's own _histogram_matching_channel: skipped (run from the checkout)", True,
               f"{type(e).__name__}")
    else:
        fn = color_fix._histogram_matching_channel
        src3 = torch.randn(2, 30, 40, generator=g, dtype=torch.float64).float()
        ref3 = 20 * torch.randn(2, 30, 40, generator=g)
        same = torch.equal(fn(src3, ref3, cpu), match_ties(src3, ref3, cpu, stats=False))
        same2 = torch.equal(fn(src3, ref3, cpu), numz_match(src3, ref3))
        nz = fn(src, ref, cpu).flatten().double().numpy()
        means_ok = max(abs(nz[flat == v].mean() - out[flat == v][0]) for v in np.unique(flat)) < 1e-4
        report("labties vs numz's own _histogram_matching_channel: equal without ties (as the written-out "
               "copy), same mean per tied value", same and same2 and means_ok)
    finally:
        sys.path.remove(cwd_path)
    print("selftest", "ok" if ok else "FAILED")
    sys.exit(0 if ok else 1)


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        sys.exit(__doc__)
    if sys.argv[1] == "--selftest":
        return selftest()
    if sys.argv[1].startswith("--"):
        sys.exit(__doc__)
    log(S.summary())
    script = os.path.abspath(sys.argv[1])
    sys.argv = [script, *sys.argv[2:]]
    sys.path[0] = os.path.dirname(script)
    install()
    atexit.register(at_exit)
    runpy.run_path(script, run_name="__main__")
    if not S.cli_patched:
        log("the CLI's functions were never found: TF32 and NUM_CC_EXTRA not applied")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""VAE probe for SeedVR2: memory and time of every VAE encode/decode call, slice and tile.

Runs inference_cli.py in-process after installing an import hook that wraps, without
touching the checkout, the methods of VideoAutoencoderKL (attn_video_vae.py):

- encode / decode: one call per batch (outermost record)
- tiled_encode / tiled_decode: the spatial tiling loop
- slicing_encode / slicing_decode: the temporal slicing loop (one per tile when tiled)
- _encode / _decode: one causal slice (the first slice, then 4 frames / 1 latent frame each)

Each record has the input and output shapes, allocated memory before and after, the call's
own torch peak (max_memory_allocated while it ran), the GPU time (synchronized), and for
slices the bytes held by the causal-convolution caches (InflatedCausalConv3d.memory) at
the end of the slice.

Per-call peaks need torch.cuda.reset_peak_memory_stats. To keep SeedVR2's own "Peak:"
figures (and so bench.py's per-phase torch peak) right, torch.cuda.max_memory_allocated
and reset_peak_memory_stats are wrapped: SeedVR2 sees the maximum since *its* last reset,
the probe's resets included.

Options (environment):
  VAE_PROBE_CONV3D_WORKAROUND=0|1  force SeedVR2's NVIDIA Conv3d workaround off or on
                                   (default: SeedVR2's own detection)
  VAE_PROBE_CACHE_DEVICE=cpu|same  where the causal-conv caches live between temporal slices
                                   (config vae.slicing.memory_device, default "same" = GPU)
  VAE_PROBE_SPLIT_SIZE=N|none      temporal slice length in frames (vae.slicing.split_size,
                                   default 4); "none" disables temporal slicing
  VAE_PROBE_CONV_MAX_MEM=G|none    per-conv input size (GiB) above which a conv is split
  VAE_PROBE_NORM_MAX_MEM=G|none    spatially (vae.memory_limit, default 0.5 / 0.5)
  VAE_PROBE_OUT                    output path (default <$BENCH_LOG without .log>.vae.json)

Usage (cwd = the SeedVR2 checkout, its venv's python):
  python vae_probe.py inference_cli.py <CLI args>
  python3 bench.py run NAME --wrap vae_probe.py -- <CLI args>   # writes <runs-dir>/NAME.vae.json
  python3 vae_probe.py --summary runs/*.vae.json                  # Markdown table
"""
import atexit
import importlib.abc
import json
import os
import runpy
import sys
import time

GIB = 1024 ** 3


class State:
    def __init__(self):
        self.records = []
        self.stack = []          # open records: each keeps its running peak
        self.carry = 0           # max peak seen by the probe since SeedVR2's last reset
        self.patched = []
        self.written = False
        self.conv3d_workaround = None
        self.config = {}         # slicing / memory-limit overrides actually applied


S = State()
_real = {}


def _torch():
    import torch
    return torch


# ---------------------------------------------------------------- peak bookkeeping

def _install_peak_wrappers():
    torch = _torch()
    if _real:
        return
    _real["max"] = torch.cuda.max_memory_allocated
    _real["reset"] = torch.cuda.reset_peak_memory_stats

    def max_memory_allocated(device=None):
        return max(_real["max"](device), S.carry)

    def reset_peak_memory_stats(device=None):
        S.carry = 0
        for r in S.stack:  # an open probe record keeps what it saw so far
            r["_peak"] = max(r["_peak"], _real["max"](device))
        return _real["reset"](device)

    torch.cuda.max_memory_allocated = max_memory_allocated
    torch.cuda.reset_peak_memory_stats = reset_peak_memory_stats


def _fold_peak():
    """Fold the allocator's peak into every open record and the carry, then reset it."""
    m = _real["max"]()
    S.carry = max(S.carry, m)
    for r in S.stack:
        r["_peak"] = max(r["_peak"], m)
    _real["reset"]()


def _shape(x):
    if hasattr(x, "shape"):
        return list(x.shape)
    for attr in ("sample", "latent_dist"):
        v = getattr(x, attr, None)
        if v is not None:
            if hasattr(v, "parameters"):  # DiagonalGaussianDistribution
                v = v.parameters
            return _shape(v)
    if isinstance(x, (list, tuple)) and x:
        return _shape(x[0])
    return None


def _cache_bytes(vae):
    total = 0
    for m in vae.modules():
        mem = getattr(m, "memory", None)
        if mem is not None and hasattr(mem, "numel") and getattr(mem, "is_cuda", False):
            total += mem.numel() * mem.element_size()
    return total


def wrap_method(cls, name, kind):
    orig = getattr(cls, name)

    def wrapped(self, x, *a, __orig=orig, **k):
        torch = _torch()
        if not torch.cuda.is_available():
            return __orig(self, x, *a, **k)
        _install_peak_wrappers()
        torch.cuda.synchronize()
        _fold_peak()
        rec = {"kind": kind, "depth": len(S.stack), "in": _shape(x),
               "alloc_before_gib": torch.cuda.memory_allocated() / GIB, "_peak": 0}
        if "memory_state" in k:
            rec["memory_state"] = getattr(k["memory_state"], "name", str(k["memory_state"]))
        if kind in ("encode", "decode") and k.get("tiled"):
            rec["tiled"] = True
            rec["tile_size"] = list(k.get("tile_size") or ())
            rec["tile_overlap"] = list(k.get("tile_overlap") or ())
        S.stack.append(rec)
        t0 = time.perf_counter()
        try:
            out = __orig(self, x, *a, **k)
        except BaseException as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            raise
        finally:
            try:
                torch.cuda.synchronize()
                _fold_peak()
            except Exception:
                pass
            rec["time_s"] = round(time.perf_counter() - t0, 4)
            S.stack.pop()
            rec["peak_gib"] = rec.pop("_peak") / GIB
            rec["alloc_after_gib"] = torch.cuda.memory_allocated() / GIB
            rec["peak_above_before_gib"] = rec["peak_gib"] - rec["alloc_before_gib"]
            rec["seq"] = len(S.records)
            S.records.append(rec)
        rec["out"] = _shape(out)
        if kind in ("_encode", "_decode"):
            rec["cache_gib"] = _cache_bytes(self) / GIB
        return out

    setattr(cls, name, wrapped)
    S.patched.append(f"{cls.__name__}.{name}")


def patch_config(cls):
    """Override the VAE's slicing / memory-limit config (configs_*/main.yaml) from the environment."""
    dev = os.environ.get("VAE_PROBE_CACHE_DEVICE")
    split = os.environ.get("VAE_PROBE_SPLIT_SIZE")
    conv = os.environ.get("VAE_PROBE_CONV_MAX_MEM")
    norm = os.environ.get("VAE_PROBE_NORM_MAX_MEM")
    S.config = {}
    if (dev or split) and hasattr(cls, "set_causal_slicing"):
        orig = cls.set_causal_slicing

        def set_causal_slicing(self, *, split_size, memory_device, __orig=orig):
            if split:
                split_size = None if split == "none" else int(split)
            if dev:
                memory_device = dev
            S.config["slicing"] = {"split_size": split_size, "memory_device": memory_device}
            return __orig(self, split_size=split_size, memory_device=memory_device)
        cls.set_causal_slicing = set_causal_slicing
        S.patched.append(f"{cls.__name__}.set_causal_slicing")
    if (conv or norm) and hasattr(cls, "set_memory_limit"):
        orig = cls.set_memory_limit

        def set_memory_limit(self, conv_max_mem, norm_max_mem, __orig=orig):
            if conv:
                conv_max_mem = float("inf") if conv == "none" else float(conv)
            if norm:
                norm_max_mem = float("inf") if norm == "none" else float(norm)
            S.config["memory_limit"] = {"conv_max_mem": conv_max_mem, "norm_max_mem": norm_max_mem}
            return __orig(self, conv_max_mem=conv_max_mem, norm_max_mem=norm_max_mem)
        cls.set_memory_limit = set_memory_limit
        S.patched.append(f"{cls.__name__}.set_memory_limit")


def patch_vae(mod):
    cls = getattr(mod, "VideoAutoencoderKL", None)
    if cls is None:
        return
    for name in ("encode", "decode", "tiled_encode", "tiled_decode",
                 "slicing_encode", "slicing_decode", "_encode", "_decode"):
        if hasattr(cls, name):
            wrap_method(cls, name, name)
    wrapper = getattr(mod, "VideoAutoencoderKLWrapper", None)
    if wrapper is not None:
        patch_config(wrapper)


def patch_conv(mod):
    v = os.environ.get("VAE_PROBE_CONV3D_WORKAROUND")
    S.conv3d_workaround = {"detected": getattr(mod, "NVIDIA_CONV3D_MEMORY_BUG_WORKAROUND", None)}
    if v in ("0", "1") and hasattr(mod, "NVIDIA_CONV3D_MEMORY_BUG_WORKAROUND"):
        mod.NVIDIA_CONV3D_MEMORY_BUG_WORKAROUND = v == "1"
        S.patched.append(f"{mod.__name__}.NVIDIA_CONV3D_MEMORY_BUG_WORKAROUND={v == '1'}")
    S.conv3d_workaround["effective"] = getattr(mod, "NVIDIA_CONV3D_MEMORY_BUG_WORKAROUND", None)
    print(f"vae_probe: Conv3d workaround {S.conv3d_workaround}", file=sys.stderr)


HOOKS = (
    (".video_vae_v3.modules.attn_video_vae", patch_vae),
    (".video_vae_v3.modules.causal_inflation_lib", patch_conv),
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


# ---------------------------------------------------------------- summary

def summarize(records):
    """One row per outermost encode/decode call, with its slices and tiles."""
    rows = []
    for i, r in enumerate(records):
        if r["depth"] != 0:
            continue
        # children were recorded (finished) before their parent: walk back to the previous top-level call
        kids = []
        j = i - 1
        while j >= 0 and records[j]["depth"] > 0:
            kids.append(records[j])
            j -= 1
        kids.reverse()
        slices = [c for c in kids if c["kind"] in ("_encode", "_decode")]
        tiles = [c for c in kids if c["kind"] in ("slicing_encode", "slicing_decode")
                 and any(k["kind"].startswith("tiled_") for k in kids)]
        rows.append({
            "kind": r["kind"], "in": r["in"], "out": r.get("out"), "tiled": r.get("tiled", False),
            "tile_size": r.get("tile_size"), "tile_overlap": r.get("tile_overlap"),
            "time_s": r["time_s"], "peak_gib": round(r["peak_gib"], 3),
            "alloc_before_gib": round(r["alloc_before_gib"], 3),
            "peak_above_before_gib": round(r["peak_above_before_gib"], 3),
            "n_tiles": len(tiles), "n_slices": len(slices),
            "slice_in": [s["in"] for s in slices[:2]],
            "slice_peaks_gib": [round(s["peak_gib"], 3) for s in slices[:3]],
            "slice_peak_max_gib": round(max((s["peak_gib"] for s in slices), default=0), 3),
            "cache_max_gib": round(max((s.get("cache_gib", 0) for s in slices), default=0), 3),
            "error": r.get("error"),
        })
    return rows


def write_out():
    if S.written:
        return
    S.written = True
    out = os.environ.get("VAE_PROBE_OUT")
    if not out:
        log = os.environ.get("BENCH_LOG")
        out = (log[:-4] if log and log.endswith(".log") else log or "vae_probe") + ".vae.json"
    summary = summarize(S.records)
    data = {"argv": sys.argv, "patched": S.patched, "conv3d_workaround": S.conv3d_workaround,
            "config_overrides": S.config, "summary": summary, "records": S.records}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    for s in summary:
        print(f"vae_probe: {s['kind']} {s['in']} -> {s['out']} tiled={s['tiled']} tiles={s['n_tiles']} "
              f"slices={s['n_slices']} peak {s['peak_gib']:.2f} GiB (+{s['peak_above_before_gib']:.2f}) "
              f"cache {s['cache_max_gib']:.2f} GiB {s['time_s']:.2f} s", file=sys.stderr)
    print(f"vae_probe: {len(S.records)} records -> {out}", file=sys.stderr)


def report(paths):
    print("| Run | Call | In (b c t h w) | Out | Tiles | Slices | Peak (GiB) | Peak above start | "
          "Max slice peak | Conv caches | Time (s) |")
    print("|" + "---|" * 11)
    for p in paths:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        name = os.path.basename(p).replace(".vae.json", "")
        for s in summarize(data["records"]):
            fmt = lambda v: "×".join(map(str, v)) if v else "–"
            print(f"| {name} | {s['kind']}{' (tiled)' if s['tiled'] else ''} | {fmt(s['in'])} | {fmt(s['out'])} | "
                  f"{s['n_tiles'] or '–'} | {s['n_slices']} | {s['peak_gib']:.2f} | {s['peak_above_before_gib']:.2f} | "
                  f"{s['slice_peak_max_gib']:.2f} | {s['cache_max_gib']:.2f} | {s['time_s']:.2f} |")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        report(sys.argv[2:])
        return
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        sys.exit(__doc__)
    script = os.path.abspath(sys.argv[1])
    sys.argv = [script, *sys.argv[2:]]
    sys.path[0] = os.path.dirname(script)
    sys.meta_path.insert(0, Finder())
    atexit.register(write_out)
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        write_out()


if __name__ == "__main__":
    main()

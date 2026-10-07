#!/usr/bin/env python3
"""The importance matrix of numz's 7B DiT (llama.cpp's imatrix: each linear layer's input second
moments), collected during numz's own SeedVR2 runs, for models/seedvr2_gguf_dyn.py. numz's checkout
is unchanged: the wrapper patches its loader at import, as ck_patch.py does.

    bench.py run NAME --seedvr2-dir NUMZ [--wrap colour_dump.py] --wrap imatrix_hook.py \\
        --env IMATRIX_OUT=FILE ... -- LR ... --model_dir DIR --dit_model seedvr2_ema_7b_fp16.safetensors
    cd NUMZ && IMATRIX_OUT=FILE python [WRAPPER.py ...] imatrix_hook.py inference_cli.py ARGS
    python imatrix_hook.py merge OUT FILE...     # several runs' files: sums of sums and of counts
    python imatrix_hook.py show FILE             # what a file holds

NUMZ is numz's ComfyUI-SeedVR2_VideoUpscaler at 4490bd1 with its environment (Apache-2.0: the hook
point is its src/core/model_loader._load_model_weights). As research/scripts/numerics_patch.py and
ck_patch.py do, the wrapper rewrites argv, runs the next script with runpy, and patches numz's
loader right after its module executes (a sys.meta_path finder that hands the module to the finders
after it: chainable with colour_dump.py and ck_patch.py, in any order).

What it collects. Once numz has loaded the DiT (_load_model_weights, model type "DiT"), a forward
pre-hook on each of the 288 attention and MLP matrices of the blocks (blocks.N.attn.proj_qkv.{vid,txt},
blocks.N.attn.proj_out.{vid,txt}, blocks.N.mlp.{vid,txt}.proj_in, blocks.N.mlp.{vid,txt}.proj_out,
N = 0..35: the matrices our GGUF files quantize) adds, at every DiT forward of the run, the input's
sum of squares per input channel, sum over tokens of x[t, k]^2 (float64, on the input's device), and
the input's token count. The importance of channel k is then in_sum2[k] / counts, the mean of x_k^2,
llama.cpp's imatrix (tools/imatrix, MIT), which ggml's quantizers take per column of a row.
- Only the fp16 DiT through numz's own path: the DiT file must be a .safetensors whose 288 block
  matrices are F16, without comfy-kitchen markers, and each hooked module must be a plain
  torch.nn.Linear with a float16 weight (not numz's GGUFQuantizedLinear, not ck_patch's CKLinear):
  otherwise the run stops (exit 3) before any forward.
- The hooks only read: no value of the run changes (the input is neither modified nor replaced,
  autocast is off inside the hook, the sums are new tensors, never in-place on numz's). They cost
  one read of each input and a float64 square-and-sum per chunk of rows (at most 2^24 values at once).
- Refused: --compile_dit (torch.compile would trace the hooks), several GPUs (--cuda_device 0,1:
  numz's workers would run without the hooks), IMATRIX_OUT unset.

IMATRIX_OUT=FILE   written when numz's CLI returns normally (FILE.part, then renamed): a safetensors
                   file, per layer <layer>.weight.in_sum2 (float64 [in_features]) and
                   <layer>.weight.counts (int64 [1]), keyed by the weight's name in the state dict;
                   metadata: format, the DiT file (name, path, real path, size), and one record per
                   run (numz's arguments, the input, numz's commit, torch, the device, the DiT's
                   forwards with their vid/txt shapes, the tokens per stream, the hook's md5, times).
                   Nothing is written if numz fails, or if a hooked layer saw no input (exit 4).
IMATRIX_DRYRUN=1   stop once numz has parsed its arguments (the guards run): prints the plan.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import json
import os
import re
import runpy
import struct
import subprocess
import sys
import threading
import time

FORMAT = "seedvr2x-imatrix-1"
LAYER = re.compile(r"^blocks\.(\d+)\.(attn\.proj_(qkv|out)\.(txt|vid)|mlp\.(txt|vid)\.proj_(in|out))$")
N_LAYERS = 288
CHUNK = 1 << 24  # values per float64 square-and-sum


def log(msg: str) -> None:
    print(f"imatrix_hook: {msg}", file=sys.stderr, flush=True)


class State:
    def __init__(self):
        self.out = os.environ.get("IMATRIX_OUT") or None
        self.dryrun = os.environ.get("IMATRIX_DRYRUN", "0") not in ("", "0")
        self.args = None  # numz's parsed arguments
        self.dit = None  # {"name", "path", "realpath", "size"}
        self.sums = {}  # layer -> float64 tensor [in_features] (on the input's device)
        self.counts = {}  # layer -> tokens (python int)
        self.calls = {}  # layer -> forwards seen
        self.in_dtypes = {}  # input dtype -> calls
        self.forwards = []  # per DiT forward: (vid_shape, txt_shape) tensors, cloned
        self.hooked = {}  # id(module) -> (name, handle)
        self.names = set()  # the hooked layers' names
        self.handles = []
        self.root_handles = []
        self.models = 0
        self.enabled = True
        self.started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.aborted = None
        self.finder_busy = threading.local()


S = State()


def abort(msg: str):
    S.aborted = msg
    log("ABORT: " + msg)
    raise SystemExit(3)


# ---------------------------------------------------------------- files

def st_header(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def check_dit_file(path: str) -> None:
    """The fp16 DiT: a safetensors file, its 288 block matrices F16, no comfy-kitchen marker."""
    if not path.endswith(".safetensors"):
        abort(f"the DiT {path} is not a .safetensors file: the importance matrix is collected on the fp16 file")
    hdr = st_header(path)
    marked = [k for k in hdr if k.endswith(".comfy_quant")]
    if marked:
        abort(f"the DiT {path} carries {len(marked)} comfy-kitchen markers: the fp16 file only")
    blocks = {k[: -len(".weight")]: v for k, v in hdr.items()
              if k.endswith(".weight") and LAYER.match(k[: -len(".weight")])}
    bad = sorted(k for k, v in blocks.items() if v.get("dtype") != "F16")
    if len(blocks) != N_LAYERS or bad:
        abort(f"the DiT {path}: {len(blocks)} block matrices ({N_LAYERS} expected), not F16: {bad[:4]}")


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def numz_commit(d: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
        return r.stdout.strip() or None
    except Exception:  # noqa: BLE001 (a metadata field only)
        return None


# ---------------------------------------------------------------- hooks

def _pre_hook(name: str, in_features: int):
    import torch

    def hook(mod, args):
        if not S.enabled or not args:
            return None
        x = args[0]
        if not isinstance(x, torch.Tensor) or x.shape[-1] != in_features:
            abort(f"{name}: input {type(x).__name__} {tuple(getattr(x, 'shape', ()))}, "
                  f"last dimension {in_features} expected")
        with torch.no_grad(), torch.autocast(x.device.type, enabled=False):
            x2 = x.detach().reshape(-1, in_features)
            rows = x2.shape[0]
            step = max(1, CHUNK // in_features)
            s = None
            for i in range(0, rows, step):
                part = x2[i:i + step].to(torch.float64).square().sum(0)
                s = part if s is None else s + part
            if s is None:
                s = torch.zeros(in_features, dtype=torch.float64, device=x.device)
            prev = S.sums.get(name)
            if prev is not None and prev.device != s.device:
                prev = prev.to(s.device)
            S.sums[name] = s if prev is None else prev + s
        S.counts[name] = S.counts.get(name, 0) + rows
        S.calls[name] = S.calls.get(name, 0) + 1
        k = str(x.dtype)
        S.in_dtypes[k] = S.in_dtypes.get(k, 0) + 1
        return None

    return hook


def _root_hook(mod, args, kwargs):
    """Each DiT forward's vid_shape and txt_shape (NaDiT.forward(vid, txt, vid_shape, txt_shape, ...))."""
    if not S.enabled:
        return None
    names = ("vid", "txt", "vid_shape", "txt_shape")
    vals = dict(zip(names, args))
    vals.update({k: kwargs[k] for k in names if k in kwargs})
    S.forwards.append(tuple(v.detach().clone() if hasattr(v, "detach") else v
                            for v in (vals.get("vid_shape"), vals.get("txt_shape"))))
    return None


def targets(model) -> list:
    """The 288 block matrices of a 7B DiT: [(name, module)], checked."""
    import torch

    found = [(n, m) for n, m in model.named_modules() if LAYER.match(n)]
    blocks = {int(LAYER.match(n).group(1)) for n, _ in found}
    if len(found) != N_LAYERS or blocks != set(range(36)):
        abort(f"{len(found)} block matrices in {len(blocks)} blocks: {N_LAYERS} in 36 expected (numz's 7B)")
    for n, m in found:
        if type(m) is not torch.nn.Linear:
            abort(f"{n} is a {type(m).__module__}.{type(m).__name__}, not a torch.nn.Linear: "
                  "the importance matrix is collected on the fp16 file through numz's own layers")
        if m.weight.dtype != torch.float16 or m.weight.is_meta:
            abort(f"{n}: weight {m.weight.dtype}{' on meta' if m.weight.is_meta else ''}, float16 expected")
    return found


def install_hooks(model) -> int:
    """Hooks on the model's 288 block matrices (each module once) and one on the model's forward."""
    n_new = 0
    found = targets(model)
    S.names |= {n for n, _ in found}
    for name, m in found:
        if id(m) in S.hooked:
            continue
        h = m.register_forward_pre_hook(_pre_hook(name, m.in_features))
        S.hooked[id(m)] = (name, h)
        S.handles.append(h)
        n_new += 1
    S.root_handles.append(model.register_forward_pre_hook(_root_hook, with_kwargs=True))
    S.models += 1
    log(f"hooks on {n_new} block matrices of the DiT (model {S.models}); inputs' sum of squares per channel")
    return n_new


def remove_hooks() -> None:
    for h in S.handles + S.root_handles:
        h.remove()
    S.handles, S.root_handles, S.hooked = [], [], {}


def reset() -> None:
    """Forget what the hooks collected (the hooks stay)."""
    S.sums, S.counts, S.calls, S.in_dtypes, S.forwards = {}, {}, {}, {}, []


# ---------------------------------------------------------------- the file

def tensors() -> dict:
    import torch

    out = {}
    for name in sorted(S.sums):
        # clone: the sums may be inference tensors (numz's forward under torch.inference_mode)
        out[f"{name}.weight.in_sum2"] = S.sums[name].detach().to("cpu", torch.float64).clone().contiguous()
        out[f"{name}.weight.counts"] = torch.tensor([S.counts[name]], dtype=torch.int64)
    return out


def run_record() -> dict:
    fw = []
    for shapes in S.forwards:
        fw.append([v.tolist() if hasattr(v, "tolist") else v for v in shapes])
    vid = S.counts.get("blocks.0.attn.proj_qkv.vid", 0)
    txt = S.counts.get("blocks.0.attn.proj_qkv.txt", 0)
    args = {k: v for k, v in vars(S.args).items()} if S.args is not None else None
    import torch

    dev = next((str(t.device) for t in S.sums.values()), None)
    return {
        "input": args.get("input") if args else None,
        "numz_args": args,
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "numz_commit": numz_commit(os.getcwd()),
        "torch": torch.__version__,
        "device": dev,
        "input_dtypes": S.in_dtypes,
        "dit_forwards": len(S.forwards),
        "forward_shapes": fw,
        "tokens": {"vid": vid, "txt": txt},
        "hook": os.path.abspath(__file__),
        "hook_md5": md5(os.path.abspath(__file__)),
        "started": S.started,
        "ended": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def write(path: str, tens: dict, meta: dict) -> None:
    from safetensors.torch import save_file

    part = path + ".part"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    save_file(tens, part, metadata={k: v if isinstance(v, str) else json.dumps(v, sort_keys=True)
                                    for k, v in meta.items()})
    os.replace(part, path)


def write_run() -> None:
    missing = [n for n in sorted(S.names) if S.counts.get(n, 0) == 0] if S.names else ["(no DiT loaded)"]
    if missing:
        log(f"ERROR: {len(missing)} hooked layers saw no input ({missing[:3]}): {S.out} not written")
        raise SystemExit(4)
    meta = {
        "format": FORMAT,
        "what": "per linear layer of the DiT's blocks: in_sum2 = sum over every input token of x^2 per input "
                "channel (float64), counts = tokens; importance = in_sum2 / counts (llama.cpp's imatrix)",
        "model": S.dit,
        "runs": [run_record()],
    }
    write(S.out, tensors(), meta)
    c = S.counts
    log(f"{S.out}: {len(S.sums)} layers, {len(S.forwards)} DiT forwards, tokens per layer "
        f"{min(c.values())}..{max(c.values())}, input dtypes {S.in_dtypes}")


# ---------------------------------------------------------------- numz's modules

def patch_loader(mod) -> None:
    if not hasattr(mod, "_load_model_weights"):
        return
    orig = mod._load_model_weights

    def _load_model_weights(model, checkpoint_path, target_device, used_meta, model_type, *a, **k):
        if model_type == "DiT":
            check_dit_file(checkpoint_path)
        res = orig(model, checkpoint_path, target_device, used_meta, model_type, *a, **k)
        if model_type == "DiT":
            S.dit = {"name": os.path.basename(checkpoint_path), "path": checkpoint_path,
                     "realpath": os.path.realpath(checkpoint_path), "size": os.path.getsize(checkpoint_path)}
            log(f"DiT weights: {checkpoint_path} (-> {S.dit['realpath']}), {S.dit['size']} bytes")
            install_hooks(res)
        return res

    mod._load_model_weights = _load_model_weights
    log(f"{mod.__name__}._load_model_weights patched")


HOOKS = ((".src.core.model_loader", patch_loader),)


class Finder(importlib.abc.MetaPathFinder):
    """Lets a module load through the finders after this one, then patches it once it executed."""

    def find_spec(self, name, path, target=None):
        hook = next((h for suffix, h in HOOKS if ("." + name).endswith(suffix)), None)
        busy = S.finder_busy.__dict__.setdefault("names", set())
        if hook is None or name in busy:
            return None
        busy.add(name)
        try:
            after = sys.meta_path[sys.meta_path.index(self) + 1:] if self in sys.meta_path else []
            spec = next((s for f in after if hasattr(f, "find_spec")
                         for s in [f.find_spec(name, path, target)] if s is not None), None)
        finally:
            busy.discard(name)
        if spec is None or spec.loader is None:
            return spec
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            hook(module)

        spec.loader.exec_module = exec_module
        return spec


def preflight(args) -> None:
    S.args = args
    if getattr(args, "compile_dit", False):
        abort("--compile_dit: torch.compile would trace the hooks; run without it")
    dev = str(getattr(args, "cuda_device", "") or "")
    if "," in dev:
        abort(f"--cuda_device {dev}: numz's workers on several GPUs would run without the hooks")
    dit = str(getattr(args, "dit_model", "") or "")
    if not dit.endswith(".safetensors"):
        abort(f"--dit_model {dit}: the fp16 .safetensors file only")
    log(f"numz's arguments: input {getattr(args, 'input', None)}, --dit_model {dit}, "
        f"--model_dir {getattr(args, 'model_dir', None)}; output {S.out}")
    if S.dryrun:
        log("dry run: stopping after the guards")
        raise SystemExit(0)


def install() -> None:
    sys.meta_path.insert(0, Finder())
    orig = argparse.ArgumentParser.parse_args

    def parse_args(self, *a, **k):
        res = orig(self, *a, **k)
        if S.args is None:
            fr = sys._getframe(1)
            while fr is not None:
                if "process_single_file" in fr.f_globals and "download_weight" in fr.f_globals:
                    preflight(res)
                    break
                fr = fr.f_back
        return res

    argparse.ArgumentParser.parse_args = parse_args


# ---------------------------------------------------------------- merge, show

def read(path: str):
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        if meta.get("format") != FORMAT:
            sys.exit(f"{path}: format {meta.get('format')!r}, {FORMAT!r} expected")
        t = {k: f.get_tensor(k) for k in f.keys()}
    return t, {k: (json.loads(v) if k in ("model", "runs", "merged_from") else v) for k, v in meta.items()}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def merge(out: str, paths: list) -> None:
    import torch

    total, runs, model, sources = None, [], None, []
    for p in paths:
        t, meta = read(p)
        m = meta.get("model") or {}
        key = (m.get("name"), m.get("size"))
        if model is None:
            model = m
        elif key != (model.get("name"), model.get("size")):
            sys.exit(f"{p}: model {key}, {(model.get('name'), model.get('size'))} expected: one model per file")
        if total is None:
            total = {k: v.clone() for k, v in t.items()}
        else:
            if set(t) != set(total):
                sys.exit(f"{p}: other layers than {paths[0]}'s")
            for k, v in t.items():
                if v.shape != total[k].shape or v.dtype != total[k].dtype:
                    sys.exit(f"{p}: {k} {tuple(v.shape)} {v.dtype}, {tuple(total[k].shape)} {total[k].dtype} expected")
                total[k] = total[k] + v
        runs += meta.get("runs") or []
        sources.append({"file": os.path.abspath(p), "sha256": sha256(p)})
    assert all(v.dtype in (torch.float64, torch.int64) for v in total.values())
    write(out, total, {"format": FORMAT, "what": read(paths[0])[1]["what"], "model": model, "runs": runs,
                       "merged_from": sources})
    print(f"{out}: {len(paths)} files, {len(runs)} runs, {len(total) // 2} layers")


def show(path: str) -> None:
    import torch

    t, meta = read(path)
    layers = sorted(k[: -len(".in_sum2")] for k in t if k.endswith(".in_sum2"))
    imp = {n: t[n + ".in_sum2"] / float(t[n + ".counts"][0]) for n in layers}
    zeros = sum(int((v <= 0).sum()) for v in imp.values())
    finite = all(bool(torch.isfinite(v).all()) for v in imp.values())
    counts = [int(t[n + ".counts"][0]) for n in layers]
    print(json.dumps({"file": path, "layers": len(layers), "model": meta.get("model"),
                      "runs": [{"input": r.get("input"), "dit_forwards": r.get("dit_forwards"),
                                "tokens": r.get("tokens"), "numz_commit": r.get("numz_commit")}
                               for r in meta.get("runs", [])],
                      "merged_from": meta.get("merged_from"), "tokens_per_layer": [min(counts), max(counts)],
                      "channels_with_zero_importance": zeros, "finite": finite,
                      "mean_importance_range": [float(min(v.mean() for v in imp.values())),
                                                float(max(v.mean() for v in imp.values()))]}, indent=1))


# ---------------------------------------------------------------- main

def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] in ("merge", "show"):
        if sys.argv[1] == "merge" and len(sys.argv) >= 4:
            return merge(sys.argv[2], sys.argv[3:])
        if sys.argv[1] == "show" and len(sys.argv) == 3:
            return show(sys.argv[2])
        sys.exit(__doc__)
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        sys.exit(__doc__)
    if not S.out:
        sys.exit("imatrix_hook: IMATRIX_OUT=FILE is required")
    script = os.path.abspath(sys.argv[1])
    sys.argv = [script, *sys.argv[2:]]
    sys.path[0] = os.path.dirname(script)
    install()
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as e:
        if e.code not in (None, 0):
            log(f"numz exited with {e.code}: {S.out} not written")
            raise
        if S.dryrun:
            raise
    write_run()


if __name__ == "__main__":
    main()

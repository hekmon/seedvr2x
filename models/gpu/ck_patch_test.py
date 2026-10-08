#!/usr/bin/env python3
"""ck_patch.py on the CPU: the model directory through the whole wrapper chain, each comfy-kitchen
format loaded into numz's real 7B by numz's own loading code, each mode's multiply against float32
references. Re-runnable; nothing is written but OUT and the scratch directory.

    cd NUMZ && CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=PYLIB nice -n 19 \\
        taskset -c 0-31 .venv/bin/python /path/to/models/gpu/ck_patch_test.py --out OUT.json \\
        --model-dir DIR --phase2 DIR --fp16 FILE --numz-models DIR --colour-dump FILE --clip FILE \\
        --scratch DIR [--parts cli,load,forward,dit,xchunk]

NUMZ is numz's checkout at 4490bd1 with its environment, PYLIB the directory holding comfy-kitchen
0.2.37. The inputs, every path absolute (numz runs in NUMZ, and the paths it resolves are compared
with them): --model-dir our model directory (links to phase 2's files and to numz's VAE), --phase2
the directory of phase 2's files it links to, --fp16 our seedvr2x_ema_7b_fp16.safetensors,
--numz-models numz's model directory (its ema_vae_fp16.safetensors, the VAE the model directory
links to), --colour-dump research/scripts/colour_dump.py, --clip a clip for numz's CLI
(anime-clean.d1.lr.mkv; the runs stop before inference), --scratch the runs' logs and outputs.
- cli: python colour_dump.py ck_patch.py inference_cli.py ARGS in a subprocess, CK_PATCH_DRYRUN=1
  (it stops once numz has built the DiT's and the VAE's structures, before any weight is read): our
  fp8, NVFP4 and GGUF names in the model directory pass numz's argument parser and resolve to our
  files, the 7B's configuration chosen, the VAE our directory's (numz's file); then the guards, each
  a run that must stop with exit status 3: no mode, a wrong mode, a mode for a GGUF file,
  --compile_dit, two GPUs, CK_PATCH_EXPECT wrong.
- load: in this process, the finder installed before numz's modules are imported: the registry's
  names resolve as without the patch; then each comfy-kitchen file (fp8, INT8, NVFP4) into numz's 7B
  on the meta device through numz's _load_model_weights on the CPU, against the fp16 file through the
  same code: 288 CKLinear holding the file's tensors bit for bit, every other parameter and buffer
  (non-persistent ones too) equal to the fp16 model's, nothing on meta, vid_in a float16 Linear and
  the model's first parameter, numz's CompatibleDiT taking the model for a float16 one; .half(),
  .float(), .to(bfloat16) and a move to the meta device leave CKLinear's dtypes and values alone.
- forward: per mode comfy-kitchen can run on the CPU, a few layers' first forward through the check
  mode's own code (CK_PATCH_CHECK's measures and proof of the path), on synthetic activations under
  bfloat16 autocast as numz's Phase 2 runs; the same input in 3-D gives the same output; a scaled_mm
  forced to fail must abort w8a8 rather than fall back; w4a4 is tried and its failure recorded (the
  GPU only).
- dit: one whole-DiT forward on a tiny input (1 x 16 x 16 latent, 20 text tokens), w8a16 against
  the fp16 model, under bfloat16 autocast: the output's relative difference. RoPE's float16 freqs
  go to float32 in both models first: CPU autocast can't cat float16 tensors (CUDA's can).
- xchunk: quantize_x, the activations' quantizer of w8a8 and w4a4 (row chunks above 2^31 elements:
  comfy-kitchen's CUDA quantizers index in 32 bits), on comfy-kitchen's CPU backend with a small
  limit forcing chunks: fp8 and NVFP4 (with and without padding rows), the global maximum in the
  last chunk, against from_float bit for bit (data, scale, block scales, shapes, dtypes); below the
  limit from_float's own result; a w8a8 CKLinear forward chunked equal to the unchunked one.
Exits 1 if a check fails.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("--out", required=True)
ap.add_argument("--parts", default="cli,load,forward,dit,xchunk")
ap.add_argument("--numz", default=os.getcwd())
ap.add_argument("--model-dir", required=True, help="our model directory: links to phase 2's files and numz's VAE")
ap.add_argument("--phase2", required=True, help="the directory of phase 2's files the model directory links to")
ap.add_argument("--fp16", required=True, help="our seedvr2x_ema_7b_fp16.safetensors")
ap.add_argument("--numz-models", required=True, help="numz's model directory: its ema_vae_fp16.safetensors")
ap.add_argument("--colour-dump", required=True, help="research/scripts/colour_dump.py")
ap.add_argument("--clip", required=True, help="a clip for numz's CLI (the runs stop before inference)")
ap.add_argument("--scratch", required=True, help="the runs' logs and outputs")
A = ap.parse_args()
PARTS = set(A.parts.split(","))
NUMZ = os.path.abspath(A.numz)
PATCH = os.path.join(HERE, "ck_patch.py")
FILES = {"fp8": "seedvr2x_ema_7b_fp8_scaled.safetensors", "int8": "seedvr2x_ema_7b_int8_convrot.safetensors",
         "nvfp4": "seedvr2x_ema_7b_nvfp4.safetensors", "q4k": "seedvr2x_ema_7b_Q4_K.gguf"}
R: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "numz": NUMZ, "patch": PATCH}
FAILS: list = []


def ok(name: str, cond: bool, detail=None) -> bool:
    R.setdefault("checks", []).append({"check": name, "ok": bool(cond), **({"detail": detail} if detail else {})})
    print(("PASS " if cond else "FAIL ") + name + (f": {detail}" if detail and not cond else ""), flush=True)
    if not cond:
        FAILS.append(name)
    return bool(cond)


def save() -> None:
    with open(A.out, "w") as f:
        json.dump(R, f, indent=1, default=str)


# ---------------------------------------------------------------- cli: the wrapper chain

def cli_run(tag: str, dit: str, env_extra: dict, extra_args=()) -> tuple[int, str, dict]:
    os.makedirs(A.scratch, exist_ok=True)
    log = os.path.join(A.scratch, f"cli-{tag}.json")
    if os.path.exists(log):
        os.remove(log)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", CK_PATCH_DRYRUN="1",
               CK_PATCH_LOG=log, COLOUR_DUMP=os.path.join(A.scratch, f"dump-{tag}"), COLOUR_DUMP_INPUTS="0")
    for k in ("CK_PATCH_MODE", "CK_PATCH_EXPECT", "CK_PATCH_CHECK", "CK_PATCH_FP16"):
        env.pop(k, None)
    env.update(env_extra)
    cmd = [sys.executable, A.colour_dump, PATCH, "inference_cli.py", A.clip,
           "--output", os.path.join(A.scratch, f"out-{tag}") + "/", "--model_dir", A.model_dir,
           "--dit_model", dit, "--resolution", "1080", "--batch_size", "5", "--load_cap", "5",
           "--color_correction", "none", "--debug", *extra_args]
    p = subprocess.run(cmd, cwd=NUMZ, env=env, capture_output=True, text=True, timeout=600)
    rep = json.load(open(log)) if os.path.exists(log) else {}
    tail = "\n".join([ln for ln in (p.stdout + p.stderr).splitlines() if "ck_patch" in ln][-12:])
    return p.returncode, tail, rep


def part_cli() -> None:
    res = R["cli"] = {}
    for key, mode in (("fp8", "w8a8"), ("nvfp4", "w4a4"), ("q4k", None)):
        name = FILES[key]
        env = {"CK_PATCH_MODE": mode, "CK_PATCH_EXPECT": "288"} if mode else {}
        code, tail, rep = cli_run(key, name, env)
        st = rep.get("structures", {})
        dit, vae = st.get("dit", {}), st.get("vae", {})
        res[key] = {"exit": code, "structures": st, "dit": rep.get("dit"), "ck": rep.get("comfy_kitchen"),
                    "log": tail}
        ok(f"cli {key}: dry run exit 0", code == 0, tail)
        ok(f"cli {key}: DiT {name} resolved to the model directory's file",
           dit.get("path") == os.path.join(A.model_dir, name)
           and dit.get("realpath") == os.path.join(A.phase2, name), dit)
        ok(f"cli {key}: the 7B's configuration", dit.get("config") == "dit_7b.nadit", dit.get("config"))
        ok(f"cli {key}: the VAE our directory's (numz's fp16 VAE)",
           vae.get("path") == os.path.join(A.model_dir, "ema_vae_fp16.safetensors")
           and vae.get("realpath") == os.path.realpath(os.path.join(A.numz_models, "ema_vae_fp16.safetensors")), vae)
        if mode:
            ok(f"cli {key}: format read in preflight", (rep.get("dit") or {}).get("layers_marked") == 288,
               rep.get("dit"))
    guards = {
        "no mode": (FILES["fp8"], {}, ()),
        "wrong mode": (FILES["fp8"], {"CK_PATCH_MODE": "w4a4"}, ()),
        "mode on GGUF": (FILES["q4k"], {"CK_PATCH_MODE": "w8a8"}, ()),
        "compile_dit": (FILES["fp8"], {"CK_PATCH_MODE": "w8a8"}, ("--compile_dit",)),
        "two GPUs": (FILES["nvfp4"], {"CK_PATCH_MODE": "w4a16"}, ("--cuda_device", "0,1")),
        "expect wrong": (FILES["int8"], {"CK_PATCH_MODE": "int8", "CK_PATCH_EXPECT": "287"}, ()),
        "check without fp16": (FILES["int8"], {"CK_PATCH_MODE": "int8", "CK_PATCH_CHECK": "1"}, ()),
    }
    res["guards"] = {}
    for g, (name, env, extra) in guards.items():
        code, tail, rep = cli_run("guard-" + g.replace(" ", "-"), name, env, extra)
        res["guards"][g] = {"exit": code, "aborted": rep.get("aborted"), "log": tail}
        ok(f"cli guard {g}: exit 3 with a message", code == 3 and bool(rep.get("aborted")), tail)


# ---------------------------------------------------------------- in-process: numz's modules, patched

def numz_modules():
    os.chdir(NUMZ)
    sys.path.insert(0, NUMZ)
    sys.path.insert(1, HERE)
    import ck_patch as cp

    cp.install()
    import torch
    from src.common.config import create_object, load_config
    from src.core import model_loader as ml
    from src.optimization.compatibility import CompatibleDiT
    from src.utils import constants as C
    from src.utils import model_registry as mr
    from src.utils.debug import Debug

    return cp, torch, create_object, load_config, ml, CompatibleDiT, C, mr, Debug(enabled=False)


def part_registry(cp, C, mr) -> None:
    orig = getattr(C.get_all_model_files, "ck_patch_orig", None)
    ok("the constants module patched by the finder", orig is not None, cp.S.patched)
    cp.S.model_dir = None
    ok("without --model_dir the patch adds nothing", C.get_all_model_files() == orig())
    choices0 = mr.get_available_dit_models()
    cp.S.model_dir = A.model_dir
    choices = mr.get_available_dit_models()
    ours = sorted(f for f in os.listdir(A.model_dir)
                  if f.endswith((".safetensors", ".gguf")) and f not in mr.MODEL_REGISTRY)
    need = sorted(FILES.values())  # the files this test runs; the directory may hold more
    R["registry"] = {"choices_without": choices0, "choices_with": choices, "ours": ours, "need": need}
    ok("our DiT files join --dit_model's choices, this test's among them",
       all(f in ours for f in need) and all(f in choices for f in ours) and choices[: len(choices0)] == choices0,
       {"ours": ours, "need": need})
    ok("the registry's names keep numz's resolution",
       C.find_model_file("seedvr2_ema_7b_fp16.safetensors", A.numz_models)
       == os.path.join(A.numz_models, "seedvr2_ema_7b_fp16.safetensors")
       and C.find_model_file("ema_vae_fp16.safetensors", A.model_dir) == os.path.join(A.model_dir, "ema_vae_fp16.safetensors")
       and "ema_vae_fp16.safetensors" not in C.get_all_model_files())
    ok("our names resolve to the model directory", all(
        C.find_model_file(f, A.model_dir) == os.path.join(A.model_dir, f) for f in ours))
    cp.S.model_dir = None


def tensors_equal(a, b, torch) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype.itemsize == 1:
        return torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    return torch.equal(a, b)


def load(cp, torch, create_object, cfg, ml, dbg, path):
    with torch.device("meta"):
        model = create_object(cfg.dit.model)
    return ml._load_model_weights(model, path, torch.device("cpu"), True, "DiT", "", dbg, None)


def part_load(env) -> dict:
    cp, torch, create_object, load_config, ml, CompatibleDiT, C, mr, dbg = env
    from safetensors import safe_open

    cfg = load_config(os.path.join(NUMZ, "configs_7b", "main.yaml"))
    t0 = time.time()
    cp.S.mode = None
    cp.S.dit = None
    m16 = load(cp, torch, create_object, cfg, ml, dbg, A.fp16)
    ref = {**dict(m16.named_parameters()), **dict(m16.named_buffers())}
    R["load"] = {"fp16_seconds": round(time.time() - t0, 1), "fp16_tensors": len(ref)}
    models = {}
    for key, mode in (("fp8", "w8a16"), ("int8", "int8"), ("nvfp4", "w4a16")):
        path = os.path.join(A.phase2, FILES[key])
        cp.S.mode, cp.S.expect = mode, "288"
        cp.S.check, cp.S.fp16 = key == "fp8", A.fp16  # the check mode's load-time comparison, once
        t0 = time.time()
        cp.expect_file(path)
        model = load(cp, torch, create_object, cfg, ml, dbg, path)
        rec = {"seconds": round(time.time() - t0, 1), "report": cp.S.report.get("load")}
        if key == "fp8":
            fc = cp.S.report.get("fp16_compare", {})
            ok("load fp8, check mode: the 840 tensors kept in 16 bits equal the fp16 file's",
               fc.get("tensors") == 840 and fc.get("n_different") == 0, fc)
            cp.S.check = False
        cks = {n: m for n, m in model.named_modules() if isinstance(m, cp.CKLinear)}
        with safe_open(path, framework="pt", device="cpu") as f:
            marked = sorted(k[: -len(cp.MARK)] for k in f.keys() if k.endswith(cp.MARK))
            same_q = all(
                tensors_equal(getattr(m, s), f.get_tensor(f"{n}.{s}"), torch)
                for n, m in cks.items() for s in ("weight", "weight_scale", "weight_scale_2")
                if getattr(m, s, None) is not None
            )
        ok(f"load {key}: 288 CKLinear, exactly the marked layers", len(cks) == 288 and sorted(cks) == marked,
           {"n": len(cks)})
        ok(f"load {key}: CKLinear's tensors are the file's, bit for bit", same_q)
        left = [n for n, p in model.named_parameters() if p.is_meta] + [n for n, b in model.named_buffers() if b.is_meta]
        ok(f"load {key}: nothing left on meta", not left, left[:5])
        mine = {**dict(model.named_parameters()), **dict(model.named_buffers())}
        outside = {n: t for n, t in mine.items() if n.rsplit(".", 1)[0] not in cks}
        want = {n: t for n, t in ref.items() if n.rsplit(".", 1)[0] not in cks}
        diff = [n for n in want if n not in outside or not tensors_equal(outside[n], want[n], torch)]
        extra = [n for n in outside if n not in want]
        ok(f"load {key}: every other parameter and buffer equal to the fp16 model's",
           not diff and not extra, {"diff": diff[:5], "extra": extra[:5]})
        biases = [n for n, m in cks.items() if m.bias is not None]
        ok(f"load {key}: the kept biases equal the fp16 model's",
           len(biases) == 216 and all(tensors_equal(cks[n].bias, ref[f"{n}.bias"], torch) for n in biases),
           {"biases": len(biases)})
        first = next(model.named_parameters())
        vid_in = model.get_submodule("vid_in.proj")
        ok(f"load {key}: vid_in an fp16 Linear, the first parameter",
           type(vid_in) is torch.nn.Linear and vid_in.weight.dtype == torch.float16
           and first[0] == "vid_in.proj.weight", [first[0], str(first[1].dtype)])
        wrapped = CompatibleDiT(model, dbg, compute_dtype=torch.bfloat16)
        ok(f"load {key}: numz's CompatibleDiT sees a float16 model",
           wrapped.model_dtype == torch.float16 and not wrapped.is_fp8_model, str(wrapped.model_dtype))
        rec["counts"] = {"tensors": len(mine), "outside": len(outside), "fp16_reference": len(want)}
        R["load"][key] = rec
        models[key] = model
        if key == "fp8":
            casts(cp, torch, model, cks, ref)
        save()
    del m16, ref
    gc.collect()
    return models


def casts(cp, torch, model, cks, ref) -> None:
    """dtype conversions and moves against CKLinear's buffers."""
    def snap():
        return {f"{n}.{k}": (b.dtype, b.data_ptr(), b.shape) for n, m in cks.items() for k, b in m._buffers.items()
                if b is not None}

    before = snap()
    refused0 = cp.S.refused_casts
    model.half()
    model.float()
    ok("casts: .half(), .float() on the model leave CKLinear's buffers (dtype, storage)", snap() == before)
    others = [p for n, p in model.named_parameters()]
    ok("casts: the other parameters do convert (float32 now)", all(p.dtype == torch.float32 for p in others))
    model.half()
    diff = [n for n, p in model.named_parameters() if not tensors_equal(p, ref[n], torch)]
    ok("casts: .half() back: every other parameter the fp16 file's again", not diff, diff[:5])
    m = cks["blocks.0.mlp.vid.proj_in"]
    one = {k: (b.dtype, b.data_ptr()) for k, b in m._buffers.items() if b is not None}
    m.to(torch.bfloat16)
    m.bfloat16()
    m.to("cpu", torch.float16)
    m.float()
    ok("casts: .to(bfloat16), .bfloat16(), .to(cpu, float16), .float() on a CKLinear leave it",
       one == {k: (b.dtype, b.data_ptr()) for k, b in m._buffers.items() if b is not None}, one)
    ok("casts: refused casts counted", cp.S.refused_casts > refused0, cp.S.refused_casts - refused0)
    m = cks["blocks.0.attn.proj_qkv.vid"]
    mc = copy.deepcopy(m)
    want = {k: (b.dtype, tuple(b.shape)) for k, b in mc._buffers.items() if b is not None}
    mc.to("meta")
    got = {k: (b.dtype, tuple(b.shape)) for k, b in mc._buffers.items() if b is not None}
    ok("casts: a move (to meta) moves CKLinear's buffers, dtypes kept",
       got == want and all(b.is_meta for b in mc._buffers.values() if b is not None), got)
    mc2, lin = copy.deepcopy(m), torch.nn.Linear(4, 4)
    lin.register_buffer("b", torch.ones(3))
    for t in [*mc2.buffers(), *lin.parameters(), *lin.buffers()]:
        t.data.set_()  # numz's release_model_memory (memory_manager.py:563-573)
    same = [b.numel() for b in mc2.buffers()] == [b.numel() for b in m.buffers()]
    ok("casts: numz's release (.data.set_()) runs on CKLinear's buffers as on any module's tensors",
       same and lin.weight.numel() == 16 and lin.b.numel() == 3,
       "torch 2.14: .data.set_() empties a detached alias, the module's own tensors keep their size")
    R["casts"] = {"refused": cp.S.refused_casts - refused0,
                  "release_note": "torch 2.14: .data.set_() leaves the module's tensors as they are (any module)"}


# ---------------------------------------------------------------- forward: each mode's multiply

LAYERS = ("blocks.0.attn.proj_qkv.vid", "blocks.0.mlp.vid.proj_in", "blocks.17.attn.proj_out.txt",
          "blocks.35.mlp.txt.proj_out")


def activations(torch, m, rows: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    scale = torch.exp(torch.randn(m.in_features, generator=g) * 0.8)  # channels of unequal size
    scale[torch.randperm(m.in_features, generator=g)[:8]] *= 20.0  # a few outlier channels
    return (torch.randn(rows, m.in_features, generator=g) * scale).float()


def part_forward(env, models) -> None:
    cp, torch = env[0], env[1]
    import torch.nn.functional as F

    cp.S.check, cp.S.fp16, cp.S.rows = True, A.fp16, 192
    out = R["forward"] = {}
    for key, modes in (("fp8", ("w8a16", "w8a8")), ("int8", ("int8",)), ("nvfp4", ("w4a16", "w4a4"))):
        model = models[key]
        for mode in modes:
            cp.S.mode = mode
            rows = []
            for i, L in enumerate(LAYERS):
                m = model.get_submodule(L)
                m.ck_mode, m.ck_checked = mode, False
                x = activations(torch, m, 256, i)
                try:
                    with torch.autocast("cpu", torch.bfloat16):
                        y = m(x)
                        y3 = m(x.reshape(2, 128, -1))
                except Exception as e:  # noqa: BLE001 (recorded: the GPU-only mode)
                    rows.append({"layer": L, "error": f"{type(e).__name__}: {e}"[:300]})
                    continue
                r = dict(cp.S.checks[-1])
                r["out"] = [str(y.dtype), list(y.shape)]
                r["same_3d"] = bool(torch.equal(y3.reshape(y.shape), y))
                rows.append(r)
            out[mode] = rows
            done = [r for r in rows if "error" not in r]
            if mode == "w4a4":
                R["forward_gpu_only"] = {"w4a4": rows[0].get("error") if rows else None}
                ok("forward w4a4: the CPU can't run it (recorded)", not done and all("error" in r for r in rows),
                   rows[0] if rows else None)
                continue
            ok(f"forward {mode}: {len(LAYERS)} layers, bfloat16 out, 3-D same as 2-D",
               len(done) == len(LAYERS) and all(r["out"][0] == "torch.bfloat16" and r["same_3d"] for r in done),
               [r.get("error") for r in rows if "error" in r])
            ok(f"forward {mode}: the path proven (comfy-kitchen's ops, backends, torch's calls)",
               all(r["proof_ok"] for r in done), [r["proof"] for r in done if not r["proof_ok"]][:1])
            lim = {"w8a16": 0.01, "w8a8": 0.06, "int8": 0.03, "w4a16": 0.01}[mode]
            ok(f"forward {mode}: kernel error under {lim:.0%} of y_deq, all finite",
               all(r["kernel"] < lim and r["finite"] for r in done), [(r["layer"], r["kernel"]) for r in done])
            for r in done:
                print(f"  {mode} {r['layer']}: kernel {r['kernel']:.4%} total {r['total']:.4%} weights "
                      f"{r['weights']:.4%} floor {r['floor']:.4%}; {r['proof']['ck']} {r['proof']['torch']} "
                      f"deq {r['proof']['dequantize']}", flush=True)
        save()
    # a failing scaled_mm must abort w8a8, not fall back to dequantizing
    cp.S.check, cp.S.mode = False, "w8a8"
    m = models["fp8"].get_submodule(LAYERS[1])
    m.ck_mode = "w8a8"
    real = F.scaled_mm

    def broken(*a, **k):
        raise RuntimeError("scaled_mm made to fail by the test")

    F.scaled_mm = broken
    try:
        with torch.autocast("cpu", torch.bfloat16):
            m(activations(torch, m, 64, 9))
        caught = None
    except RuntimeError as e:
        caught = str(e)
    finally:
        F.scaled_mm = real
    ok("forward w8a8: a failing scaled_mm aborts instead of a silent dequantization",
       caught is not None and "fell back to dequantizing" in caught, caught)
    # the exit report, as a run's atexit writes it
    cp.S.check, cp.S.log_path = True, os.path.join(A.scratch, "at_exit.json")
    cp.at_exit()
    rep = json.load(open(cp.S.log_path))
    c = rep.get("check", {})
    ok("the exit report: the check summary, per kind and per path", c.get("layers") == len(cp.S.checks) == 16
       and c.get("proof_ok") == 16 and "all" in c.get("per_kind", {}) and len(c.get("paths", [])) >= 5,
       {k: v for k, v in c.items() if k not in ("per_layer", "per_kind")})
    R["exit_report"] = {k: v for k, v in c.items() if k != "per_layer"}
    cp.S.check, cp.S.log_path = False, None


# ---------------------------------------------------------------- dit: one whole forward

def part_dit(env, models) -> None:
    cp, torch, create_object, load_config, ml, CompatibleDiT, C, mr, dbg = env
    cfg = load_config(os.path.join(NUMZ, "configs_7b", "main.yaml"))
    cp.S.mode, cp.S.check = "w8a16", False
    q = models["fp8"]
    for m in q.modules():
        if isinstance(m, cp.CKLinear):
            m.ck_mode = "w8a16"
    cp.S.dit = None
    m16 = load(cp, torch, create_object, cfg, ml, dbg, A.fp16)
    # CPU autocast can't cat float16 tensors (rotary_embedding_torch's get_axial_freqs: "Unexpected
    # floating ScalarType in at::autocast::prioritize"): RoPE's float16 freqs in float32, in both models
    rope = 0
    for model in (m16, q):
        for mod in model.modules():
            f = getattr(mod, "freqs", None)
            if isinstance(f, torch.Tensor) and f.dtype == torch.float16:
                f.data = f.data.float()
                rope += 1
    R["dit"] = {"rope_freqs_to_float32": rope}
    g = torch.Generator().manual_seed(0)
    vid = torch.randn(1 * 16 * 16, 33, generator=g)
    txt = torch.randn(20, 5120, generator=g) * 0.1
    args = dict(vid_shape=torch.tensor([[1, 16, 16]]), txt_shape=torch.tensor([[20]]),
                timestep=torch.tensor([500.0]))
    outs = {}
    for name, model in (("fp16", m16), ("w8a16", q)):
        t0 = time.time()
        with torch.no_grad(), torch.autocast("cpu", torch.bfloat16):
            outs[name] = model(vid.clone(), txt.clone(), **args).vid_sample.float()
        R.setdefault("dit", {})[f"{name}_seconds"] = round(time.time() - t0, 1)
    a, b = outs["w8a16"], outs["fp16"]
    rel = float((a - b).norm() / b.norm())
    R["dit"].update({"rel_w8a16_vs_fp16": rel, "shape": list(a.shape), "finite": bool(torch.isfinite(a).all()),
                     "ck_calls": dict(cp.S.cdt_seen)})
    ok("dit: a whole forward in w8a16, finite, against the fp16 model", bool(torch.isfinite(a).all()) and rel < 0.2,
       rel)
    print(f"  dit: w8a16 vs fp16 relative difference {rel:.4%} ({a.shape})", flush=True)


# ---------------------------------------------------------------- xchunk: the activations' quantizer in row chunks

def same_qt(torch, a, b) -> dict:
    """Two comfy-kitchen QuantizedTensors, field by field (bytes for the data and block scales)."""
    r = {"layout": a._layout_cls == b._layout_cls, "shape": tuple(a.shape) == tuple(b.shape),
         "dtype": a.dtype == b.dtype, "storage": (a._qdata.dtype, tuple(a._qdata.shape)) == (b._qdata.dtype,
                                                                                            tuple(b._qdata.shape)),
         "data": a._qdata.shape == b._qdata.shape and torch.equal(a._qdata.view(torch.uint8), b._qdata.view(torch.uint8)),
         "scale": a._params.scale.dtype == b._params.scale.dtype and torch.equal(a._params.scale, b._params.scale)}
    if hasattr(a._params, "block_scale"):
        sa, sb = a._params.block_scale, b._params.block_scale
        r["block_scale"] = sa.shape == sb.shape and torch.equal(sa.view(torch.uint8), sb.view(torch.uint8))
    return r


def part_xchunk(cp, torch) -> None:
    from comfy_kitchen.tensor import QuantizedTensor

    cp.import_ck()
    out = R["xchunk"] = {"backends": cp.S.ck["available"]}
    g = torch.Generator().manual_seed(7)
    for layout, m, k in (("TensorCoreFP8Layout", 1000, 256), ("TensorCoreNVFP4Layout", 1000, 256),
                         ("TensorCoreNVFP4Layout", 1003, 256)):
        x = (torch.randn(m, k, generator=g) * torch.exp(torch.randn(k, generator=g) * 0.8)).to(torch.bfloat16)
        x[m - 5, 17] = 300.0  # the global maximum, in the last chunk
        limit = 3 * 128 * k - 1  # chunks of 256 rows: 4 chunks, the last one shorter
        ref = QuantizedTensor.from_float(x, layout)
        calls = cp.S.x_chunked["calls"]
        got = cp.quantize_x(x, layout, limit)
        r = same_qt(torch, got, ref)
        r["chunked"] = cp.S.x_chunked["calls"] == calls + 1
        out[f"{layout} {m}x{k}"] = r
        ok(f"xchunk {layout} {m}x{k}: 4 chunks (256 rows, limit {limit}) = from_float bit for bit "
           f"({', '.join(r)})", all(r.values()), r)
    x = torch.randn(64, 256, generator=g).to(torch.bfloat16)
    calls = cp.S.x_chunked["calls"]
    r = same_qt(torch, cp.quantize_x(x, "TensorCoreFP8Layout"), QuantizedTensor.from_float(x, "TensorCoreFP8Layout"))
    r["not_chunked"] = cp.S.x_chunked["calls"] == calls
    out["below_limit"] = r
    ok("xchunk: below the limit, from_float as it is (no chunk)", all(r.values()), r)
    # a w8a8 CKLinear forward: chunked (X_LIMIT lowered) = unchunked, bit for bit
    n, k = 96, 512
    w = (torch.randn(n, k, generator=g) * 0.05).to(torch.bfloat16)
    qw = QuantizedTensor.from_float(w, "TensorCoreFP8Layout")
    lin = cp.ck_linear_class()("t.proj_out", {"format": "float8_e4m3fn"}, "w8a8",
                               {"weight": qw._qdata, "weight_scale": qw._params.scale},
                               (torch.randn(n, generator=g) * 0.1).to(torch.float16), k, n)
    x = torch.randn(2, 700, k, generator=g) * 2.0
    saved, saved_check = cp.X_LIMIT, cp.S.check
    cp.S.check = False
    try:
        with torch.autocast("cpu", torch.bfloat16):
            y1 = lin(x)
            calls = cp.S.x_chunked["calls"]
            cp.X_LIMIT = 128 * k * 3 - 1  # chunks of 256 rows
            y2 = lin(x)
    finally:
        cp.X_LIMIT, cp.S.check = saved, saved_check
    r = {"chunked": cp.S.x_chunked["calls"] == calls + 1, "equal": bool(torch.equal(y1, y2)),
         "dtype": str(y2.dtype), "finite": bool(torch.isfinite(y2).all())}
    out["ck_linear_w8a8"] = r
    ok("xchunk: a w8a8 CKLinear forward, chunked = unchunked bit for bit", r["chunked"] and r["equal"]
       and r["dtype"] == "torch.bfloat16" and r["finite"], r)


def main() -> None:
    env = None
    try:
        if "cli" in PARTS:
            part_cli()
            save()
        if PARTS & {"load", "forward", "dit"}:
            env = numz_modules()
            part_registry(env[0], env[6], env[7])
            models = part_load(env)
            if "forward" in PARTS:
                part_forward(env, models)
            if "dit" in PARTS:
                part_dit(env, models)
        if "xchunk" in PARTS:
            if env is None:
                sys.path.insert(0, HERE)
                import ck_patch as cp
                import torch
            else:
                cp, torch = env[0], env[1]
            part_xchunk(cp, torch)
    finally:
        R["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        R["failed"] = FAILS
        R["pass"] = not FAILS
        save()
    print(("PASS" if not FAILS else f"FAIL ({len(FAILS)}): " + "; ".join(FAILS)), flush=True)
    sys.exit(0 if not FAILS else 1)


if __name__ == "__main__":
    main()

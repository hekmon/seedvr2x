#!/usr/bin/env python3
"""imatrix_hook.py on the CPU: the files' metadata, bytes and read-back check on small files; the
wrapper chain and its guards through numz's CLI; then numz's real 7B DiT loaded by numz's own code
with the hooks, a whole forward on a tiny input with and without them (bit for bit), the sums
against an independent computation, the file written, read back and merged.
Re-runnable; nothing is written but OUT, IMATRIX and the scratch directory.

    cd NUMZ && CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 nice -n 19 taskset -c 0-31 \\
        .venv/bin/python /path/to/models/gpu/imatrix_hook_test.py --out OUT.json --imatrix FILE \\
        --fp16 FILE --model-dir DIR --colour-dump FILE --clip FILE --scratch DIR \\
        [--parts meta,cli,dit] [--gguf NAME] [--threads 8]

NUMZ is numz's checkout at 4490bd1 with its environment. The inputs, every path absolute (numz runs
in NUMZ): --fp16 numz's seedvr2_ema_7b_fp16.safetensors, --model-dir numz's model directory holding
it (the runs' --model_dir; --gguf a GGUF name of numz's registry for a guard, by default
seedvr2_ema_7b-Q4_K_M.gguf), --colour-dump research/scripts/colour_dump.py, --clip a clip for
numz's CLI (anime-clean.d1.lr.mkv; the runs stop before inference), --scratch the runs' logs and
outputs.
- meta: no numz. A run's record as imatrix_hook.py wrote it before the metadata rule (absolute and
  home paths, times, argv, cwd, the hook's path) kept as RUN_KEYS: base names, frames from the
  latent shapes, numz's other arguments unchanged, a kept record kept again equal to itself; a file
  written twice, the same bytes, read back by the library equal, its data section the library's
  own (save_file's) byte for byte, metadata first with sorted keys; refused, no file left: a path
  (absolute, home, Windows) or a date-time in the metadata, a key outside META_KEYS, a run key
  outside RUN_KEYS, and a file whose bytes read back otherwise than checked (deleted); merge of two
  such old files: the sums and counts added exactly, the runs' records kept as RUN_KEYS, merged = 2,
  no path or time in the header, the same bytes twice; merged alone, an old file gives its own data
  section byte for byte; two models refused.
- cli: python colour_dump.py [ck_patch.py] imatrix_hook.py inference_cli.py ARGS in a subprocess
  with IMATRIX_DRYRUN=1 (it stops once numz parsed its arguments): the chains pass, the patch of
  numz's loader is installed; then the guards, each a run that must stop with exit status 3:
  --compile_dit, two GPUs, a GGUF DiT, an argument the file would keep with a date-time (an
  output directory's name); and IMATRIX_OUT unset must refuse to start.
- dit: in this process, the finder installed before numz's modules are imported; numz's 7B on the
  meta device, the fp16 file loaded by numz's _load_model_weights on the CPU: the hooks on exactly
  the 288 block matrices, each a float16 torch.nn.Linear; a whole DiT forward under bfloat16 autocast
  as numz's Phase 2 runs (1 x 16 x 16 latent, 20 text tokens; RoPE's float16 freqs in float32 first,
  as ck_patch_test.py does: CPU autocast can't cat float16 tensors) without hooks, with them, and
  without again: the three outputs equal bit for bit; the counts (tokens) per stream; the sums of
  4 layers against x.double().square().sum(0) of their inputs captured by a plain hook; the file
  (IMATRIX) written by the hook's own writer, read back, merged with itself (twice the sums and
  counts); the last block's three text matrices whose output nothing reads (its attention output
  projection and MLP, NaDiT.forward keeps only vid after the blocks) zeroed: the output unchanged,
  bit for bit, while zeroing a live one (its text qkv) changes it; and the forward under
  torch.inference_mode (the sums then inference tensors): the same output, sums and file.
Exits 1 if a check fails.
"""

from __future__ import annotations

import argparse
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
ap.add_argument("--imatrix", required=True, help="where the tiny forward's importance matrix goes")
ap.add_argument("--parts", default="meta,cli,dit")
ap.add_argument("--numz", default=os.getcwd())
ap.add_argument("--fp16", required=True, help="numz's seedvr2_ema_7b_fp16.safetensors")
ap.add_argument("--model-dir", required=True, help="numz's model directory, holding --fp16")
ap.add_argument("--gguf", default="seedvr2_ema_7b-Q4_K_M.gguf")
ap.add_argument("--colour-dump", required=True, help="research/scripts/colour_dump.py")
ap.add_argument("--clip", required=True, help="a clip for numz's CLI (the runs stop before inference)")
ap.add_argument("--scratch", required=True, help="the runs' logs and outputs")
ap.add_argument("--threads", type=int, default=8)
A = ap.parse_args()
PARTS = set(A.parts.split(","))
NUMZ = os.path.abspath(A.numz)
HOOK = os.path.join(HERE, "imatrix_hook.py")
CK = os.path.join(HERE, "ck_patch.py")
R: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "numz": NUMZ, "hook": HOOK}
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


# ---------------------------------------------------------------- meta

def part_meta() -> None:
    import hashlib
    import re
    import struct

    sys.path.insert(1, HERE)
    import imatrix_hook as ih
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    res = R["meta"] = {}
    d = os.path.join(A.scratch, "meta")
    os.makedirs(d, exist_ok=True)
    g = torch.Generator().manual_seed(1)
    layers = ["blocks.0.attn.proj_qkv.vid", "blocks.0.attn.proj_qkv.txt", "blocks.35.mlp.txt.proj_out"]

    def sums(c0: int) -> dict:
        t = {}
        for i, n in enumerate(layers):
            t[f"{n}.weight.in_sum2"] = torch.rand(16 + 8 * i, generator=g, dtype=torch.float64) * 1e6
            t[f"{n}.weight.counts"] = torch.tensor([c0 + i], dtype=torch.int64)
        return t

    def split(path: str) -> tuple[dict, bytes, bytes]:
        with open(path, "rb") as f:
            b = f.read()
        (n,) = struct.unpack("<Q", b[:8])
        return json.loads(b[8:8 + n]), b[8:8 + n], b[8 + n:]

    def sha(path: str) -> str:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def refused(tag: str, path: str, fn) -> bool:
        try:
            fn()
        except SystemExit as e:
            res.setdefault("refusals", {})[tag] = str(e)[-160:]
            return not os.path.exists(path) and not os.path.exists(path + ".part")
        return False

    def raw_run(clip: str, out: str, T: int, vid: int) -> dict:
        """A run's record as imatrix_hook.py wrote it before the metadata rule (md5 65f07109)."""
        return {
            "argv": ["/srv/x/numz/inference_cli.py", f"/home/u/clips/{clip}", "--output", f"/home/u/out/{out}/"],
            "cwd": "/srv/x/numz", "device": "cuda:0", "dit_forwards": 1,
            "started": "2026-10-07T07:04:51+0000", "ended": "2026-10-07T07:08:43+0000",
            "forward_shapes": [[[[T, 136, 254]], [[58]]]],
            "hook": "/home/u/models/gpu/imatrix_hook.py", "hook_md5": "65f0710965e894b9360f674c26dd8d62",
            "input": f"/home/u/clips/{clip}", "input_dtypes": {"torch.bfloat16": 72, "torch.float32": 216},
            "numz_args": {"attention_mode": "flash_attn_2", "batch_size": 45, "compile_dit": False,
                          "cuda_device": None, "dit_model": "seedvr2_ema_7b_fp16.safetensors",
                          "input": f"/home/u/clips/{clip}", "input_noise_scale": 0.0, "model_dir": "~/models",
                          "output": f"/home/u/out/{out}/", "resolution": 1080, "seed": 42, "tile_debug": "false",
                          "video_backend": "opencv"},
            "numz_commit": "4490bd1f482e026674543386bb2a4d176da245b9",
            "tokens": {"txt": 58, "vid": vid}, "torch": "2.14.1+cu130"}

    model_old = {"name": "seedvr2_ema_7b_fp16.safetensors", "path": "/srv/x/models/seedvr2_ema_7b_fp16.safetensors",
                 "realpath": "/srv/x/models/seedvr2_ema_7b_fp16.safetensors", "size": 16479334424}
    model = {"name": "seedvr2_ema_7b_fp16.safetensors", "size": 16479334424}
    r1 = raw_run("cel4k-detail.d1x4.lr.mkv", "c-imx-cel4k-detail", 12, 103632)
    r2 = raw_run("ouatia-dark.d1x4.lr.mkv", "c-imx-ouatia-dark", 11, 94996)
    p1 = ih.public_run(r1)
    res["kept_run"] = p1
    path_args = ("input", "output", "model_dir")
    ok("meta: a run's record kept as exactly RUN_KEYS", sorted(p1) == sorted(ih.RUN_KEYS), sorted(p1))
    ok("meta: the clip and numz's path arguments by base name, numz's other arguments unchanged",
       p1["input"] == "cel4k-detail.d1x4.lr.mkv" and p1["numz_args"]["input"] == "cel4k-detail.d1x4.lr.mkv"
       and p1["numz_args"]["output"] == "c-imx-cel4k-detail" and p1["numz_args"]["model_dir"] == "models"
       and {k: v for k, v in p1["numz_args"].items() if k not in path_args}
       == {k: v for k, v in r1["numz_args"].items() if k not in path_args}, p1["numz_args"])
    ok("meta: frames 45 from 12 latent frames (41 from 11), resolution 1080, seed 42; tokens, commit, torch, "
       "forwards, shapes and input dtypes kept",
       p1["frames"] == 45 and ih.public_run(r2)["frames"] == 41 and p1["resolution"] == 1080 and p1["seed"] == 42
       and all(p1[k] == r1[k] for k in ("tokens", "numz_commit", "torch", "dit_forwards", "forward_shapes",
                                        "input_dtypes")), p1)
    ok("meta: a kept record kept again is itself", ih.public_run(p1) == p1)
    meta = {"format": ih.FORMAT, "what": ih.WHAT, "model": model, "runs": [p1]}
    t1 = sums(1000)
    a, a2 = os.path.join(d, "a.imatrix.safetensors"), os.path.join(d, "a2.imatrix.safetensors")
    ih.write(a, t1, meta)
    ih.write(a2, t1, meta)
    hdr, raw, data = split(a)
    with safe_open(a, framework="pt", device="cpu") as f:
        back_meta = f.metadata()
        back = {k: f.get_tensor(k) for k in f.keys()}
    want_meta = {k: v if isinstance(v, str) else json.dumps(v, sort_keys=True) for k, v in meta.items()}
    ok("meta: a file written twice: the same bytes", sha(a) == sha(a2), [sha(a), sha(a2)])
    ok("meta: read back by the library: every tensor (dtype, shape, values) and the metadata as written",
       set(back) == set(t1) and all(back[k].dtype == v.dtype and torch.equal(back[k], v) for k, v in t1.items())
       and back_meta == want_meta)
    lib = os.path.join(d, "lib.safetensors")
    save_file(t1, lib)
    lhdr, _, ldata = split(lib)
    order = [k for k in hdr if k != "__metadata__"]
    lorder = sorted((k for k in lhdr if k != "__metadata__"), key=lambda k: lhdr[k]["data_offsets"][0])
    ok("meta: the data section the library's (save_file) byte for byte, the tensors in its order, each with "
       "its dtype, shape and offsets",
       data == ldata and order == lorder and all(hdr[k] == lhdr[k] for k in order), {"order": order[:3]})
    ok("meta: the header: metadata first, keys sorted, padded with spaces to 8 bytes; no path, no time",
       list(hdr)[0] == "__metadata__" and list(hdr["__metadata__"]) == sorted(want_meta) and len(raw) % 8 == 0
       and not re.search(rb"/home/|/srv/|~/", raw) and not ih.TIME.search(raw.decode()))
    bad = {
        "absolute path in a run": {**meta, "runs": [{**p1, "input": "/home/u/clips/x.mkv"}]},
        "home path in numz's arguments": {**meta, "runs": [{**p1, "numz_args": {**p1["numz_args"],
                                                                                "model_dir": "~/models"}}]},
        "Windows path in a run": {**meta, "runs": [{**p1, "input": "C:\\clips\\x.mkv"}]},
        "absolute path in what": {**meta, "what": ih.WHAT + " (from /srv/x/y)"},
        "date-time in what": {**meta, "what": ih.WHAT + " 2026-10-07T07:04"},
        "date-time in a run": {**meta, "runs": [{**p1, "torch": "2026-10-07 07:04"}]},
        "path in the model": {**meta, "model": {**model, "name": "/srv/x/models/seedvr2_ema_7b_fp16.safetensors"}},
        "key outside META_KEYS": {**meta, "merged_from": [{"file": "a", "sha256": "0" * 64}]},
        "run key outside RUN_KEYS": {**meta, "runs": [{**p1, "started": "early"}]},
        "model key outside MODEL_KEYS": {**meta, "model": {**model, "realpath": "x"}},
        "old-style record": {**meta, "runs": [r1]},
    }
    x = os.path.join(d, "x.imatrix.safetensors")
    for tag, m in bad.items():
        ok(f"meta: refused, no file: {tag}", refused(tag, x, lambda m=m: ih.write(x, t1, m)))
    orig = ih.serialize
    ih.serialize = lambda tens, m: orig(tens, {**m, "what": m["what"] + " /home/u"})
    try:
        ok("meta: a file reading back otherwise than checked: refused and deleted",
           refused("read back", x, lambda: ih.write(x, t1, meta)))
    finally:
        ih.serialize = orig
    # files as the hook wrote them before the rule (the library's save_file, records with paths and times)
    o1, o2 = os.path.join(d, "o1.imatrix.safetensors"), os.path.join(d, "o2.imatrix.safetensors")
    t2 = sums(2000)
    for p, t, r in ((o1, t1, r1), (o2, t2, r2)):
        save_file(t, p, metadata={"format": ih.FORMAT, "what": ih.WHAT, "model": json.dumps(model_old, sort_keys=True),
                                  "runs": json.dumps([r], sort_keys=True)})
    m1, m2 = os.path.join(d, "m1.imatrix.safetensors"), os.path.join(d, "m2.imatrix.safetensors")
    ih.merge(m1, [o1, o2])
    ih.merge(m2, [o1, o2])
    tm, mm = ih.read(m1)
    mhdr, mraw, _ = split(m1)
    ok("meta: merge of two old files: sums and counts added exactly",
       set(tm) == set(t1) and all(torch.equal(tm[k], t1[k] + t2[k]) for k in t1))
    ok("meta: ... the runs' records kept as RUN_KEYS, in order, the model as MODEL_KEYS, merged 2, nothing else",
       sorted(mm) == sorted(ih.META_KEYS) and mm["runs"] == [ih.public_run(r1), ih.public_run(r2)]
       and mm["model"] == model and mm["merged"] == 2 and mm["what"] == ih.WHAT, sorted(mm))
    ok("meta: ... no path or time in the header; merged twice, the same bytes",
       not re.search(rb"/home/|/srv/|~/", mraw) and not ih.TIME.search(mraw.decode()) and sha(m1) == sha(m2))
    s1 = os.path.join(d, "s1.imatrix.safetensors")
    ih.merge(s1, [o1])
    ok("meta: an old file merged alone: its own data section byte for byte (only the header changes)",
       split(s1)[2] == split(o1)[2] and split(s1)[1] != split(o1)[1])
    o3 = os.path.join(d, "o3.imatrix.safetensors")
    save_file(t2, o3, metadata={"format": ih.FORMAT, "what": ih.WHAT,
                                "model": json.dumps({**model_old, "size": 1}, sort_keys=True),
                                "runs": json.dumps([r2], sort_keys=True)})
    m3 = os.path.join(d, "m3.imatrix.safetensors")
    ok("meta: merge refuses files of two models", refused("two models", m3, lambda: ih.merge(m3, [o1, o3])))
    for p in (a, a2, lib, o1, o2, o3, m1, m2, s1):
        os.remove(p)


# ---------------------------------------------------------------- cli

def cli_run(tag: str, wrappers: list, dit: str, env_extra: dict, extra_args=()) -> tuple[int, str]:
    os.makedirs(A.scratch, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", IMATRIX_DRYRUN="1",
               IMATRIX_OUT=os.path.join(A.scratch, f"cli-{tag}.imatrix.safetensors"),
               COLOUR_DUMP=os.path.join(A.scratch, f"dump-{tag}"), COLOUR_DUMP_INPUTS="0")
    for k in ("CK_PATCH_MODE", "CK_PATCH_EXPECT", "CK_PATCH_CHECK", "CK_PATCH_DRYRUN"):
        env.pop(k, None)
    env.update(env_extra)
    env = {k: v for k, v in env.items() if v is not None}
    cmd = [sys.executable, *wrappers, "inference_cli.py", A.clip,
           "--output", os.path.join(A.scratch, f"out-{tag}") + "/", "--model_dir", A.model_dir,
           "--dit_model", dit, "--resolution", "1080", "--batch_size", "5", "--load_cap", "5",
           "--color_correction", "none", "--debug", *extra_args]
    p = subprocess.run(cmd, cwd=NUMZ, env=env, capture_output=True, text=True, timeout=600)
    tail = "\n".join([ln for ln in (p.stdout + p.stderr).splitlines()
                      if "imatrix_hook" in ln or "ck_patch" in ln][-10:])
    return p.returncode, tail


def part_cli() -> None:
    res = R["cli"] = {}
    fp16 = os.path.basename(A.fp16)
    for tag, wr in (("chain", [A.colour_dump, HOOK]), ("chain-ck", [A.colour_dump, CK, HOOK]),
                    ("chain-ck-rev", [A.colour_dump, HOOK, CK])):
        code, tail = cli_run(tag, wr, fp16, {})
        res[tag] = {"exit": code, "log": tail}
        ok(f"cli {tag}: dry run exit 0, numz's loader patched, arguments parsed",
           code == 0 and "_load_model_weights patched" in tail and "dry run" in tail, tail)
    guards = {
        "compile_dit": ([A.colour_dump, HOOK], fp16, {}, ("--compile_dit",), 3),
        "two GPUs": ([A.colour_dump, HOOK], fp16, {}, ("--cuda_device", "0,1"), 3),
        "GGUF DiT": ([A.colour_dump, HOOK], A.gguf, {}, (), 3),
        "date-time in an argument": ([A.colour_dump, HOOK], fp16, {},
                                     ("--output", os.path.join(A.scratch, "out-2026-10-08T01:00") + "/"), 3),
        "no IMATRIX_OUT": ([HOOK], fp16, {"IMATRIX_OUT": None}, (), 1),
    }
    res["guards"] = {}
    for g, (wr, dit, env, extra, want) in guards.items():
        env = dict(env)
        if "IMATRIX_OUT" in env and env["IMATRIX_OUT"] is None:
            env = {"IMATRIX_OUT": ""}
        code, tail = cli_run("guard-" + g.replace(" ", "-"), wr, dit, env, extra)
        res["guards"][g] = {"exit": code, "log": tail}
        ok(f"cli guard {g}: exit {want}", code == want, {"exit": code, "log": tail})


# ---------------------------------------------------------------- dit

def part_dit() -> None:
    os.chdir(NUMZ)
    sys.path.insert(0, NUMZ)
    sys.path.insert(1, HERE)
    import imatrix_hook as ih

    ih.install()
    import torch

    torch.set_num_threads(A.threads)
    from src.common.config import create_object, load_config
    from src.core import model_loader as ml
    from src.utils.debug import Debug

    dbg = Debug(enabled=False)
    ok("numz's loader patched by the finder", getattr(ml._load_model_weights, "__name__", "") == "_load_model_weights"
       and ml._load_model_weights.__module__ == "imatrix_hook")
    cfg = load_config(os.path.join(NUMZ, "configs_7b", "main.yaml"))
    t0 = time.time()
    with torch.device("meta"):
        model = create_object(cfg.dit.model)
    model = ml._load_model_weights(model, A.fp16, torch.device("cpu"), True, "DiT", "", dbg, None)
    R["dit"] = {"load_seconds": round(time.time() - t0, 1)}
    hooked = sorted(n for n, _ in ih.S.hooked.values())
    lin = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    ok("hooks on exactly the 288 block matrices", len(hooked) == 288 and all(ih.LAYER.match(n) for n in hooked)
       and len(set(lin) - set(hooked)) == len(lin) - 288, {"hooked": len(hooked), "linears": len(lin)})
    ok("each hooked module a float16 torch.nn.Linear", all(
        type(model.get_submodule(n)) is torch.nn.Linear and model.get_submodule(n).weight.dtype == torch.float16
        for n in hooked))
    R["dit"]["outside"] = sorted(set(lin) - set(hooked))
    rope = 0
    for mod in model.modules():
        f = getattr(mod, "freqs", None)
        if isinstance(f, torch.Tensor) and f.dtype == torch.float16:
            f.data = f.data.float()
            rope += 1
    R["dit"]["rope_freqs_to_float32"] = rope
    g = torch.Generator().manual_seed(0)
    vid = torch.randn(1 * 16 * 16, 33, generator=g)
    txt = torch.randn(20, 5120, generator=g) * 0.1
    kw = dict(vid_shape=torch.tensor([[1, 16, 16]]), txt_shape=torch.tensor([[20]]), timestep=torch.tensor([500.0]))

    def fwd():
        t = time.time()
        with torch.no_grad(), torch.autocast("cpu", torch.bfloat16):
            y = model(vid.clone(), txt.clone(), **kw).vid_sample
        return y, round(time.time() - t, 1)

    ih.remove_hooks()
    ih.reset()
    y0, s0 = fwd()
    ok("no hook left after remove_hooks: nothing collected", not ih.S.sums and not ih.S.forwards)
    ih.install_hooks(model)
    captured = {}
    probe = ["blocks.0.attn.proj_qkv.vid", "blocks.0.attn.proj_qkv.txt", "blocks.17.mlp.vid.proj_out",
             "blocks.35.mlp.txt.proj_in"]
    hs = [model.get_submodule(n).register_forward_pre_hook(
        lambda m, a, n=n: captured.__setitem__(n, a[0].detach().clone())) for n in probe]
    y1, s1 = fwd()
    for h in hs:
        h.remove()
    snap = {n: (ih.S.sums[n].clone(), ih.S.counts[n]) for n in ih.S.sums}
    fw = list(ih.S.forwards)
    ih.remove_hooks()
    y2, s2 = fwd()
    R["dit"].update({"forward_seconds": [s0, s1, s2], "out_shape": list(y0.shape), "out_dtype": str(y0.dtype)})
    ok("forward finite", bool(torch.isfinite(y0.float()).all()))
    ok("with the hooks: the output bit for bit the output without", torch.equal(y0, y1) and torch.equal(y0, y2),
       {"max_abs_diff": float((y0.float() - y1.float()).abs().max())})
    vid_tokens, txt_tokens = 1 * 8 * 8, 20
    cnt_ok = all(c == (vid_tokens if (".vid" in n) else txt_tokens) for n, (_, c) in snap.items())
    ok(f"counts: {vid_tokens} video tokens, {txt_tokens} text tokens per layer, 288 layers, one forward",
       len(snap) == 288 and cnt_ok and len(fw) == 1,
       {"counts": sorted({c for _, c in snap.values()}), "forwards": len(fw)})
    diffs = {}
    for n in probe:
        x = captured[n].reshape(-1, captured[n].shape[-1])
        want = x.double().square().sum(0)
        got = snap[n][0]
        diffs[n] = {"input_dtype": str(captured[n].dtype), "rows": x.shape[0],
                    "max_rel": float(((got - want).abs() / want.abs().clamp_min(1e-300)).max()),
                    "equal": bool(torch.equal(got, want))}
    R["dit"]["sums_vs_independent"] = diffs
    ok("sums of 4 layers equal x.double().square().sum(0) of their inputs",
       all(d["max_rel"] < 1e-12 for d in diffs.values()), diffs)
    # the file
    ih.S.sums = {n: v for n, (v, _) in snap.items()}
    ih.S.counts = {n: c for n, (_, c) in snap.items()}
    ih.S.forwards = fw
    ih.S.dit = {"name": os.path.basename(A.fp16), "path": A.fp16, "realpath": os.path.realpath(A.fp16),
                "size": os.path.getsize(A.fp16)}
    ih.S.out = A.imatrix
    ih.write_run()
    t, meta = ih.read(A.imatrix)
    ok("file read back: 288 x (in_sum2 float64, counts int64), values as collected",
       len(t) == 576 and all(torch.equal(t[f"{n}.weight.in_sum2"], v) and int(t[f"{n}.weight.counts"][0]) == c
                             for n, (v, c) in snap.items())
       and all(t[k].dtype == (torch.float64 if k.endswith("in_sum2") else torch.int64) for k in t))
    run = meta.get("runs", [{}])[0]
    ok("metadata: format, what, model (name, size), one run kept as RUN_KEYS with its forward shapes, 1 frame",
       sorted(meta) == ["format", "model", "runs", "what"] and meta["format"] == ih.FORMAT
       and meta["model"] == {"name": os.path.basename(A.fp16), "size": os.path.getsize(A.fp16)}
       and len(meta["runs"]) == 1 and sorted(run) == sorted(ih.RUN_KEYS) and run["dit_forwards"] == 1
       and run["forward_shapes"] == [[[[1, 16, 16]], [[20]]]] and run["frames"] == 1
       and run["tokens"] == {"vid": vid_tokens, "txt": txt_tokens}, run)
    merged = A.imatrix.replace(".safetensors", "") + ".merge2.safetensors"
    ih.merge(merged, [A.imatrix, A.imatrix])
    t2, meta2 = ih.read(merged)
    ok("merge of the file with itself: twice the sums and the counts, the run twice, merged 2",
       all(torch.equal(t2[k], t[k] * 2) for k in t) and meta2["runs"] == [run, run] and meta2["merged"] == 2)
    os.remove(merged)
    # dead and live matrices of the last block
    dead = ["blocks.35.attn.proj_out.txt", "blocks.35.mlp.txt.proj_in", "blocks.35.mlp.txt.proj_out"]
    saved = {}
    for n in dead:
        m = model.get_submodule(n)
        saved[n] = (m.weight.data.clone(), None if m.bias is None else m.bias.data.clone())
        m.weight.data.zero_()
        if m.bias is not None:
            m.bias.data.zero_()
    y3, _ = fwd()
    ok("the last block's text attention output and MLP zeroed: the output unchanged, bit for bit",
       torch.equal(y0, y3), {"max_abs_diff": float((y0.float() - y3.float()).abs().max())})
    for n, (w, b) in saved.items():
        m = model.get_submodule(n)
        m.weight.data.copy_(w)
        if b is not None:
            m.bias.data.copy_(b)
    live = model.get_submodule("blocks.35.attn.proj_qkv.txt")
    w = live.weight.data.clone()
    live.weight.data.zero_()
    y4, _ = fwd()
    live.weight.data.copy_(w)
    d4 = float((y0.float() - y4.float()).norm() / y0.float().norm())
    R["dit"]["live_txt_qkv_zeroed_rel_change"] = d4
    ok("zeroing the last block's text qkv (read by the video's attention) changes the output", d4 > 0, d4)
    y5, _ = fwd()
    ok("weights restored: the output as at first", torch.equal(y0, y5))
    # numz may run its forward under torch.inference_mode: the sums are then inference tensors
    ih.reset()
    ih.install_hooks(model)
    with torch.inference_mode(), torch.autocast("cpu", torch.bfloat16):
        y6 = model(vid.clone(), txt.clone(), **kw).vid_sample
    ih.remove_hooks()
    inf = A.imatrix.replace(".safetensors", "") + ".inference.safetensors"
    ih.S.out = inf
    ih.write_run()
    t3, _ = ih.read(inf)
    ok("under torch.inference_mode: the same output, the same sums and counts, the file written",
       torch.equal(y0, y6) and all(torch.equal(t3[k], t[k]) for k in t), {"max_abs_diff": float((y0.float() - y6.float()).abs().max())})
    os.remove(inf)


def main() -> None:
    try:
        if "meta" in PARTS:
            part_meta()
            save()
        if "cli" in PARTS:
            part_cli()
            save()
        if "dit" in PARTS:
            part_dit()
    finally:
        R["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        R["failed"] = FAILS
        R["pass"] = not FAILS
        save()
    print(("PASS" if not FAILS else f"FAIL ({len(FAILS)}): " + "; ".join(FAILS)), flush=True)
    sys.exit(0 if not FAILS else 1)


if __name__ == "__main__":
    main()

# /// script
# requires-python = "==3.12.*"
# dependencies = ["torch==2.14.1", "numpy==2.5.2", "safetensors==0.8.0", "gguf==0.19.0"]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""SeedVR2's static GGUF files for seedvr2x, from ByteDance's fp32 masters: Q4_K and Q8_0.

The 288 attention and MLP matrices of the blocks (99% of the weights) are quantized straight from
the float32 master by ggml's own function, ggml_quantize_chunk (llama.cpp at LLAMA_CPP, MIT, built
here from source, no importance matrix): Q4_K, the type of every quantized tensor in numz's Q4_K_M
file, or Q8_0. Every other tensor is our fp16 file's: the master rounded to the nearest float16.
gguf-py writes the file in city96's conventions, which numz's loader reads: the state dict's
names, each shape reversed (GGUF's order), general.architecture "seedvr" as in numz's own SeedVR2
GGUF files.

    uv run models/seedvr2_gguf.py [--masters DIR]... [--cache DIR] [--work DIR] [--numz DIR]
                                  [--out DIR] [FILE...]

- ggml: llama.cpp cloned at LLAMA_CPP into --work (default ~/.cache/seedvr2x-models/ggml), its
  commit and tree checked by hash; its ggml-base library built by cmake without native CPU
  flags, so that the quantization runs the same code on every x86-64 CPU; called through ctypes.
- The checks: every quantized tensor decoded by ggml (dequantize_row_*) and by gguf-py's own
  decoder, which must agree bit for bit; its error against the master (relative,
  ||W_hat - W|| / ||W||), median and worst; for Q4_K, numz's Q4_K_M file's error on the same
  tensors beside ours (the file found in --numz, else downloaded into --cache); the file read
  back by gguf-py's reader (names, shapes, types, every byte as written); the output's SHA-256
  pinned in OUTPUTS. A file failing a check is deleted.
- The output, in --out (default models/dist): not uploaded before GPU runs validate it (DESIGN.md,
  Weights).
"""

from __future__ import annotations

import argparse
import ctypes
import os
import re
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path
from typing import NamedTuple

import gguf
import numpy as np
import torch
from common import CACHE, DIST, check_output, find_or_fetch, log
from gguf import GGMLQuantizationType as Q
from seedvr2_fp16 import COPYRIGHT, FILES, REPO, REVISION, hf_url, load_master

LLAMA_CPP_URL = "https://github.com/ggml-org/llama.cpp"
LLAMA_CPP = "abeada335e2e78bd3fe63febafab7e900ce75810"  # master on 2026-10-06
LLAMA_CPP_TREE = "f1e6d0c78e58a87a3ec9a63b2bf7239d80403f84"
BLOCK = re.compile(
    r"^blocks\.\d+\.(attn\.proj_(qkv|out)\.(txt|vid)|mlp\.(txt|vid)\.proj_(in|out))\.weight$"
)
NUMZ_REPO = "AInVFX/SeedVR2_comfyUI"  # numz's registry: the 7B GGUF files' repository
NUMZ_REVISION = "ac66d6d98fa49975d893b58c55bff7677191c862"  # main since 2025-11-12


class Output(NamedTuple):
    fp16: str  # our fp16 file of the same master (seedvr2_fp16.py's FILES)
    qtype: Q
    file_type: gguf.LlamaFileType
    numz: str | None  # numz's GGUF file of the same model and type, to compare errors with
    numz_size: int
    numz_sha256: str


OUTPUTS_SPEC = {
    "seedvr2x_ema_7b_Q4_K.gguf": Output(
        "seedvr2x_ema_7b_fp16.safetensors",
        Q.Q4_K,
        gguf.LlamaFileType.MOSTLY_Q4_K_S,
        "seedvr2_ema_7b-Q4_K_M.gguf",
        4758306592,
        "db9cb2ad90ebd40d2e8c29da2b3fc6fd03ba87cd58cbadceccca13ad27162789",
    ),
    "seedvr2x_ema_7b_Q8_0.gguf": Output(
        "seedvr2x_ema_7b_fp16.safetensors", Q.Q8_0, gguf.LlamaFileType.MOSTLY_Q8_0, None, 0, ""
    ),
    "seedvr2x_ema_7b_sharp_Q4_K.gguf": Output(
        "seedvr2x_ema_7b_sharp_fp16.safetensors",
        Q.Q4_K,
        gguf.LlamaFileType.MOSTLY_Q4_K_S,
        "seedvr2_ema_7b_sharp-Q4_K_M.gguf",
        4758306592,
        "7aed800ac4eb8e0d18569a954c0ff35f5a1caa3ed5d920e66cc31405f75b6e69",
    ),
    "seedvr2x_ema_7b_sharp_Q8_0.gguf": Output(
        "seedvr2x_ema_7b_sharp_fp16.safetensors",
        Q.Q8_0,
        gguf.LlamaFileType.MOSTLY_Q8_0,
        None,
        0,
        "",
    ),
}

# The outputs of the pinned inputs and versions: a run must give these bytes.
OUTPUTS = {
    "seedvr2x_ema_7b_Q4_K.gguf": (
        "7f4642d0701c343a33fbd0a72518ccdb08c5b02d6fbb5f58624b9957b42ecde5"
    ),
    "seedvr2x_ema_7b_Q8_0.gguf": (
        "67ea572a82bea623da75d7b275ace0d64af4f1af67454fae4fd9478cb03d8b52"
    ),
    "seedvr2x_ema_7b_sharp_Q4_K.gguf": (
        "a5e423a50bdb0b0b491a0c981a5357fa1dba82819791ed0d7cf241153eeee023"
    ),
    "seedvr2x_ema_7b_sharp_Q8_0.gguf": (
        "03fad523334f2721ae4396fb41f9d273042980dea2a0f16071da65c9aad8ca1a"
    ),
}


def run(cmd: list[str]) -> str:
    """cmd's output; on failure, what it printed."""
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(
            f"{' '.join(cmd)}: exit {r.returncode}\n{r.stdout[-4000:]}{r.stderr[-4000:]}"
        )
    return r.stdout.strip()


def ggml(work: Path, threads: int) -> ctypes.CDLL:
    """ggml-base at LLAMA_CPP, cloned and built in work once. Built through llama.cpp's own
    CMake project, everything but ggml switched off: its ggml directory, configured on its own,
    asks for a file only ggml's own repository has (ggml.pc.in)."""
    src = work / f"llama.cpp-{LLAMA_CPP[:7]}"
    if not (src / ".git").exists():
        log(f"ggml: cloning {LLAMA_CPP_URL} at {LLAMA_CPP} into {src}")
        run(
            [
                "git",
                "clone",
                "--quiet",
                "--filter=blob:none",
                "--no-checkout",
                LLAMA_CPP_URL,
                str(src),
            ]
        )
    run(["git", "-C", str(src), "sparse-checkout", "disable"])
    run(["git", "-C", str(src), "checkout", "--quiet", LLAMA_CPP])
    head = run(["git", "-C", str(src), "rev-parse", "HEAD"])
    tree = run(["git", "-C", str(src), "rev-parse", "HEAD^{tree}"])
    dirty = run(["git", "-C", str(src), "status", "--porcelain", "--untracked-files=no"])
    if (head, tree, dirty) != (LLAMA_CPP, LLAMA_CPP_TREE, ""):
        raise SystemExit(
            f"{src}: commit {head}, tree {tree}, changes {dirty!r}; "
            f"{LLAMA_CPP}, {LLAMA_CPP_TREE} and none expected"
        )
    build = work / f"build-{LLAMA_CPP[:7]}"
    libs = sorted(build.rglob("libggml-base.so"))
    if not libs:
        log(f"ggml: building ggml-base in {build}")
        off = ["COMMON", "TESTS", "TOOLS", "EXAMPLES", "SERVER", "APP"]
        run(
            [
                "cmake",
                "-S",
                str(src),
                "-B",
                str(build),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DBUILD_SHARED_LIBS=ON",
                "-DGGML_NATIVE=OFF",
                "-DLLAMA_OPENSSL=OFF",
                *(f"-DLLAMA_BUILD_{o}=OFF" for o in off),
            ]
        )
        run(["cmake", "--build", str(build), "--target", "ggml-base", "-j", str(threads)])
        libs = sorted(build.rglob("libggml-base.so"))
    lib = ctypes.CDLL(str(libs[0]))
    lib.ggml_quantize_chunk.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
    lib.ggml_row_size.restype = ctypes.c_size_t
    for f in ("dequantize_row_q4_K", "dequantize_row_q8_0"):
        getattr(lib, f).argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        getattr(lib, f).restype = None
    log(f"ggml: {libs[0]} (llama.cpp {LLAMA_CPP[:7]}, {run(['cc', '--version']).splitlines()[0]})")
    return lib


def quantize(lib: ctypes.CDLL, w: np.ndarray, qtype: Q, threads: int) -> np.ndarray:
    """A float32 matrix [rows, n] quantized by ggml_quantize_chunk, in chunks of rows on threads
    (each row's blocks depend on that row alone): uint8 [rows, row size]."""
    rows, n = w.shape
    block = {Q.Q4_K: 256, Q.Q8_0: 32}[qtype]
    if n % block:  # ggml would abort the process
        raise SystemExit(f"rows of {n} values: not a multiple of {qtype.name}'s {block}")
    size = lib.ggml_row_size(int(qtype), n)
    out = np.empty((rows, size), np.uint8)
    step = max(1, -(-rows // (threads * 4)))

    def chunk(r: int) -> None:
        k = min(step, rows - r)
        got = lib.ggml_quantize_chunk(int(qtype), w.ctypes.data, out.ctypes.data, r * n, k, n, None)
        if got != k * size:
            raise RuntimeError(f"ggml_quantize_chunk wrote {got} bytes, {k * size} expected")

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(chunk, range(0, rows, step)))
    return out


def ggml_decode(lib: ctypes.CDLL, q: np.ndarray, qtype: Q, shape: tuple[int, ...]) -> np.ndarray:
    out = np.empty(shape, np.float32)
    fn = lib.dequantize_row_q4_K if qtype == Q.Q4_K else lib.dequantize_row_q8_0
    fn(q.ctypes.data, out.ctypes.data, out.size)
    return out


def rel(a: np.ndarray, w: np.ndarray) -> float:
    ta, tw = torch.from_numpy(a), torch.from_numpy(w)
    return float(
        torch.linalg.vector_norm(ta - tw, dtype=torch.float64)
        / torch.linalg.vector_norm(tw, dtype=torch.float64)
    )


def build(name: str, spec: Output, lib: ctypes.CDLL, a: argparse.Namespace) -> None:
    t0 = time.time()
    f = FILES[spec.fp16]
    master = find_or_fetch(
        f.master,
        hf_url(REPO, REVISION, f.master),
        f.master_sha256,
        f.master_size,
        a.masters,
        a.cache,
    )
    numz = None
    if spec.numz:
        numz = find_or_fetch(
            spec.numz,
            hf_url(NUMZ_REPO, NUMZ_REVISION, spec.numz),
            spec.numz_sha256,
            spec.numz_size,
            a.numz,
            a.cache,
        )
    sd = load_master(master)
    blocks = [k for k in sd if BLOCK.match(k)]
    if len(blocks) != 288:
        raise SystemExit(f"{master}: {len(blocks)} block matrices, 288 expected")
    numz_tensors = {}
    if numz is not None:
        numz_tensors = {
            t.name.removeprefix("model.diffusion_model."): t for t in gguf.GGUFReader(numz).tensors
        }
    out = a.out / name
    part = out.with_name(out.name + ".part")
    out.parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(str(part), arch="seedvr", use_temp_file=False)
    w.add_name(name.removesuffix(".gguf"))
    w.add_license("apache-2.0")
    w.add_file_type(int(spec.file_type))
    w.add_quantization_version(gguf.GGML_QUANT_VERSION)
    w.add_source_url(hf_url(REPO, REVISION, f.master))
    w.add_string("seedvr2x.source", f"{f.master} of {REPO}, revision {REVISION}")
    w.add_string("seedvr2x.source_sha256", f.master_sha256)
    w.add_string("seedvr2x.copyright", COPYRIGHT)
    w.add_string(
        "seedvr2x.change",
        f"the 288 attention and MLP matrices of the blocks quantized to {spec.qtype.name} "
        f"by ggml_quantize_chunk (llama.cpp {LLAMA_CPP}, no importance matrix) from the "
        "float32 master; every other tensor rounded to the nearest float16",
    )
    w.add_string(
        "seedvr2x.conversion",
        "unofficial, not ByteDance's: made by seedvr2x's models/seedvr2_gguf.py",
    )
    written: dict[str, np.ndarray] = {}
    errors, numz_errors, differ = {}, {}, 0
    for k, v in sd.items():
        x = np.ascontiguousarray(v.numpy())
        if x.ndim == 0:
            raise SystemExit(f"{k}: a scalar, which GGUF can't hold")
        if k in blocks:
            q = quantize(lib, x, spec.qtype, a.threads)
            d1 = ggml_decode(lib, q, spec.qtype, x.shape)
            d2 = gguf.quants.dequantize(q, spec.qtype).reshape(x.shape)
            if not np.array_equal(d1.view(np.uint32), d2.view(np.uint32)):
                differ += 1
                log(
                    f"{k}: ggml and gguf-py decode differently: max |diff| "
                    f"{float(np.abs(d1 - d2).max()):.3g}"
                )
            errors[k] = rel(d1, x)
            if k in numz_tensors:
                t = numz_tensors[k]
                numz_errors[k] = rel(
                    np.asarray(gguf.quants.dequantize(t.data, t.tensor_type), np.float32).reshape(
                        x.shape
                    ),
                    x,
                )
            w.add_tensor(k, q, raw_dtype=spec.qtype)
            written[k] = q
        else:
            h = v.contiguous().to(torch.float16).numpy()
            w.add_tensor(k, h)
            written[k] = h
    try:
        if differ:
            raise SystemExit(f"{name}: {differ} tensors decoded differently by ggml and gguf-py")
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        os.replace(part, out)
        e = list(errors.values())
        log(
            f"{name}: 288 matrices in {spec.qtype.name} (ggml = gguf-py, bit for bit), error "
            f"against the master: median {statistics.median(e):.5f}, worst {max(e):.5f}; "
            f"{len(sd) - 288} tensors in F16; {out.stat().st_size} bytes"
        )
        if numz_errors:
            ne = list(numz_errors.values())
            better = sum(errors[k] < numz_errors[k] for k in numz_errors)
            log(
                f"{name}: numz's {spec.numz} on the same {len(ne)} tensors: median "
                f"{statistics.median(ne):.5f}, worst {max(ne):.5f}; ours lower on {better}"
            )
        r = gguf.GGUFReader(out)
        if {t.name for t in r.tensors} != set(sd):
            raise SystemExit(f"{out}: tensor names read back differ")
        for t in r.tensors:
            want = Q.F16 if t.name not in blocks else spec.qtype
            shape = tuple(int(s) for s in reversed(t.shape.tolist()))
            if (
                t.tensor_type != want
                or shape != tuple(sd[t.name].shape)
                or not np.array_equal(
                    np.asarray(t.data).view(np.uint8).reshape(-1),
                    written[t.name].view(np.uint8).reshape(-1),
                )
            ):
                raise SystemExit(f"{out}: {t.name} read back differs")
        log(f"{name}: read back by gguf-py's reader: every tensor's name, type, shape and bytes")
        check_output(out, OUTPUTS.get(name))
    except BaseException:
        part.unlink(missing_ok=True)
        out.unlink(missing_ok=True)
        raise
    log(f"{name}: done in {time.time() - t0:.0f} s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "files", nargs="*", metavar="FILE", help=f"of {', '.join(OUTPUTS_SPEC)} (default: all)"
    )
    ap.add_argument("--masters", type=Path, action="append", default=[])
    ap.add_argument(
        "--numz",
        type=Path,
        action="append",
        default=[],
        help="a directory holding numz's GGUF files (read in place)",
    )
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--work", type=Path, default=CACHE / "ggml")
    ap.add_argument("--threads", type=int, default=len(os.sched_getaffinity(0)))
    ap.add_argument("--out", type=Path, default=DIST)
    a = ap.parse_args()
    for name in a.files:
        if name not in OUTPUTS_SPEC:
            ap.error(f"{name}: not one of {', '.join(OUTPUTS_SPEC)}")
    lib = ggml(a.work, a.threads)
    log(
        f"torch {torch.__version__}, numpy {np.__version__}, gguf {version('gguf')}, "
        f"{a.threads} threads"
    )
    for name in a.files or OUTPUTS_SPEC:
        build(name, OUTPUTS_SPEC[name], lib, a)


if __name__ == "__main__":
    main()

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
"""SeedVR2's dynamic GGUF file for seedvr2x: one ggml type per matrix, chosen and quantized with an
importance matrix, under the size of our uniform Q4_K file; and that Q4_K quantized with the same
importance, the control that tells the importance's own effect from the mix's.

    uv run models/seedvr2_gguf_dyn.py [--model 7b|sharp] [--imatrix FILE|ones] [--imatrices DIR]...
        [--types Q3_K,...] [--masters DIR]... [--cache DIR] [--work DIR] [--static FILE] [--out DIR]
        [--only REGEX] [--probe] [--table FILE] [--margin BYTES] [--threads N]

- The importance: models/gpu/imatrix_hook.py's file (several runs' merged by it): per matrix and
  input channel k, imp_k = in_sum2_k / counts, the mean of x_k^2 over the calibration runs' tokens
  (llama.cpp's imatrix), collected on numz's fp16 file of the same model. By default --model's own,
  pinned in IMATRICES: ours, uploaded to hekmon/seedvr2x on Hugging Face at IMATRIX_REVISION with
  the files made from it; read where an --imatrices directory has it, never moved, else downloaded
  into --cache (default ~/.cache/seedvr2x-models), and refused unless its size and SHA-256 are the
  pinned ones, before the master is read. It keeps its name: the outputs' metadata name it by name
  and SHA-256. --imatrix FILE: another importance file, read as it is (OUTPUTS pins the outputs of
  the pinned ones only). `ones` (every imp_k = 1) is a placeholder for tests: ggml does NOT treat
  it as no importance (below).
- Each of the 288 attention and MLP matrices of the blocks, from the float32 master, quantized by
  ggml's own ggml_quantize_chunk WITH the importance (llama.cpp at seedvr2_gguf.py's LLAMA_CPP, MIT,
  the same build), to each candidate type (default Q3_K, Q4_K, Q5_K, Q6_K, Q8_0: types numz's GGUF
  loader dequantizes, src/optimization/gguf_dequant.py at 4490bd1), decoded by ggml, and measured:
  the plain error ||W_hat - W|| / ||W|| and the activation-weighted one
      e = sqrt(sum_k imp_k ||dW[:, k]||^2 / sum_k imp_k ||W[:, k]||^2),
  dW = W_hat - W, column k the weights input channel k meets: with uncorrelated channels, e^2 is
  E||dW x||^2 / E||W x||^2, the layer's output error relative to its output on the calibration's
  inputs. Also, as the reference, Q4_K without importance: our static file (with --static, its bytes
  are checked equal, tensor by tensor).
- The choice: one type per matrix minimising sum_i e_i^2 over the 288 under a byte budget, the
  288 matrices' bytes in our uniform Q4_K less --margin (1 MiB, for the metadata: the file stays
  within our Q4_K's size): a greedy over each matrix's lower convex hull of (bytes, e^2), steepest
  e^2 saved per byte first, ties broken by name; deterministic. The last block's text attention
  output projection and MLP (blocks.35.attn.proj_out.txt, blocks.35.mlp.txt.proj_in, .proj_out)
  count for nothing: NaDiT.forward keeps only the video tokens after the blocks, so nothing reads
  their output (proven bit for bit by models/gpu/imatrix_hook_test.py); they get the cheapest type.
  Limits: per-layer statistics; every layer's relative error weighs the same whatever its effect
  on the video; errors taken as additive, with no interaction through the network; channels taken
  as uncorrelated; the calibration's clips only. No end-to-end sensitivity: GPU runs judge the file.
- The files, in --out: seedvr2x_ema_7b_dyn.gguf (the choice) and seedvr2x_ema_7b_Q4_K_imatrix.gguf
  (every matrix Q4_K, with the importance), written as seedvr2_gguf.py writes ours: gguf-py's
  writer, city96's conventions (the state dict's names, shapes reversed, general.architecture
  "seedvr"), every other tensor our fp16 file's value (the master rounded to the nearest float16),
  the metadata saying what was done and from what: the importance file by its name and SHA-256,
  its runs, the calibration clips by name; no path and no time, so that the same inputs give the
  same bytes wherever they are read from.
- The checks, as for our static files: every quantized tensor decoded by ggml and by gguf-py, bit
  for bit; its errors against the master (median, worst), beside our static Q4_K's; the file read
  back by gguf-py's reader (names, types, shapes, every byte; no metadata value holding a path or
  a time); its size against our Q4_K's; the second quantization of each chosen matrix equal to
  the first; the SHA-256 against the one OUTPUTS pins for that importance file (both outputs of
  each pinned importance file; another importance's printed, not checked). numz's own loader:
  models/numz_gguf_check.py, in numz's environment. A file failing a check is deleted.
- --only REGEX: only the matching block matrices (the choice under their own Q4_K bytes), nothing
  written; --probe: on them, each type quantized with no importance, with ones and with the
  importance, bytes compared and errors given. --table FILE: the per-matrix measures of an earlier
  run with the same inputs (master, importance, types, llama.cpp), not measured again.
- Why ones isn't "no importance": with no importance ggml quantizes Q3_K..Q6_K by its reference
  functions (quantize_row_q*_K_ref: their own weights, x^2 or av_x + |x|), with one by the
  importance-aware ones (quantize_row_q*_K_impl: weights imp_k * sqrt(sigma^2 + x^2), or imp_k for
  Q6_K, and a longer search); Q8_0 ignores the importance (ggml-quants.c at LLAMA_CPP).
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import heapq
import json
import os
import re
import statistics
import time
from importlib.metadata import version
from pathlib import Path

import gguf
import numpy as np
import torch
from common import CACHE, DIST, find_or_fetch, log
from gguf import GGMLQuantizationType as Q
from seedvr2_fp16 import COPYRIGHT, FILES, REPO, REVISION, hf_url, load_master
from seedvr2_gguf import BLOCK, LLAMA_CPP, ggml

# numz's GGUF loader dequantizes these (src/optimization/gguf_dequant.py, dequantize_functions, at
# 4490bd1), F16 and F32 natively; the candidates must be among them
NUMZ_TYPES = ("Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "BF16")
DEFAULT_TYPES = ("Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0")
DECODE = {Q.Q2_K: "q2_K", Q.Q3_K: "q3_K", Q.Q4_K: "q4_K", Q.Q5_K: "q5_K", Q.Q6_K: "q6_K", Q.Q8_0: "q8_0",
          Q.Q4_0: "q4_0", Q.Q4_1: "q4_1", Q.Q5_0: "q5_0", Q.Q5_1: "q5_1"}
IMATRIX_FORMAT = "seedvr2x-imatrix-1"
# the importance's description in the files' metadata (seedvr2x.imatrix): names, counts and hashes, never a path
IMATRIX_META = ("file", "inputs", "kind", "model", "numz_commit", "runs", "sha256", "zero_channels")
HOST_PATH = re.compile(r"(?:^|[\s\"'\[(,=])~?/(?!/)")  # an absolute or home path; not a URL's // nor a relative path
TIME = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")
MODELS = {  # --model: our fp16 file (its master), the fp16 file the importance must come from
    "7b": ("seedvr2x_ema_7b_fp16.safetensors", "seedvr2_ema_7b_fp16.safetensors", "seedvr2x_ema_7b"),
    "sharp": ("seedvr2x_ema_7b_sharp_fp16.safetensors", "seedvr2_ema_7b_sharp_fp16.safetensors",
              "seedvr2x_ema_7b_sharp"),
}
DEAD = re.compile(r"^blocks\.35\.(attn\.proj_out\.txt|mlp\.txt\.proj_(in|out))\.weight$")
STATIC_Q4K = {"7b": ("seedvr2x_ema_7b_Q4_K.gguf", 4758307552), "sharp": ("seedvr2x_ema_7b_sharp_Q4_K.gguf", 4758307584)}

# The importance files, by --model: models/gpu/imatrix_hook.py's runs of numz's fp16 file on 4 calibration clips,
# merged, with no path and no time in their metadata; uploaded to our Hugging Face repo, IMATRIX_REPO, with the files
# made from them. Without --imatrix, --model's is read where an --imatrices directory has it, else downloaded from
# IMATRIX_REPO at IMATRIX_REVISION into --cache; refused unless it has this size and SHA-256.
IMATRIX_REPO = "hekmon/seedvr2x"
IMATRIX_REVISION = "c14a2bc4aab04cf38ad9b0d324014c4213f07048"  # the upload of 2026-10-08
IMATRICES = {  # --model: (name, size, SHA-256)
    "7b": ("seedvr2_ema_7b_fp16.imatrix.safetensors", 12454888,
           "f2283e03e507c5db9234434bfe89fb79a554377c7dd79894c8bcb35a557ac6ab"),
    "sharp": ("seedvr2_ema_7b_sharp_fp16.imatrix.safetensors", 12454944,
              "2c0dedc3a67931d413dbe0a3ad74f3271a70a79947e39edcbd1bdba8477d5e20"),
}

# The outputs of the pinned inputs and versions, by (file name, the importance file's SHA-256): a run must give these
# bytes, and a second run gave them. The importance files are IMATRICES's, from hekmon/seedvr2x at revision c14a2bc4:
# seedvr2_ema_7b_fp16.imatrix.safetensors (12,454,888 bytes, SHA-256 f2283e03...) and
# seedvr2_ema_7b_sharp_fp16.imatrix.safetensors (12,454,944 bytes, SHA-256 2c0dedc3...). The metadata name the
# importance file by its name and SHA-256, the clips by name, and hold no path: read from anywhere under that name, it
# gives these bytes.
OUTPUTS: dict[tuple[str, str], str] = {
    ("seedvr2x_ema_7b_dyn.gguf", "f2283e03e507c5db9234434bfe89fb79a554377c7dd79894c8bcb35a557ac6ab"): (
        "f8c0c50d233dad8a26fb82d5eda4072c2ad935100e842b3fee953994849a487d"
    ),
    ("seedvr2x_ema_7b_Q4_K_imatrix.gguf", "f2283e03e507c5db9234434bfe89fb79a554377c7dd79894c8bcb35a557ac6ab"): (
        "2d9f7e607d9cdb1dd86debf9b89c2e3dbd34d19d99016454f048d827ca3f1385"
    ),
    ("seedvr2x_ema_7b_sharp_dyn.gguf", "2c0dedc3a67931d413dbe0a3ad74f3271a70a79947e39edcbd1bdba8477d5e20"): (
        "90ba80c85e25ee08162882be58c0e05f1b4fc19c8dd9837b6f87c0efc4fad779"
    ),
    ("seedvr2x_ema_7b_sharp_Q4_K_imatrix.gguf", "2c0dedc3a67931d413dbe0a3ad74f3271a70a79947e39edcbd1bdba8477d5e20"): (
        "486a7da980d83a5fa475d623cf9d29675355975f0e57bacb390f30008163fab1"
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def setup(lib: ctypes.CDLL) -> None:
    for f in DECODE.values():
        fn = getattr(lib, f"dequantize_row_{f}")
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        fn.restype = None


def quantize(lib, w: np.ndarray, qtype: Q, imp: np.ndarray | None, threads: int) -> np.ndarray:
    """A float32 matrix [rows, n] quantized by ggml_quantize_chunk with the importance imp (float32 [n],
    one weight per column; None: no importance), in chunks of rows on threads: uint8 [rows, row size]."""
    from concurrent.futures import ThreadPoolExecutor

    rows, n = w.shape
    block = gguf.GGML_QUANT_SIZES[qtype][0]
    if n % block:
        raise SystemExit(f"rows of {n} values: not a multiple of {qtype.name}'s {block}")
    if imp is not None and (imp.dtype != np.float32 or imp.shape != (n,) or not imp.flags.c_contiguous):
        raise SystemExit(f"importance {imp.dtype} {imp.shape}: float32 [{n}] expected")
    size = lib.ggml_row_size(int(qtype), n)
    out = np.empty((rows, size), np.uint8)
    step = max(1, -(-rows // (threads * 4)))
    ip = None if imp is None else imp.ctypes.data

    def chunk(r: int) -> None:
        k = min(step, rows - r)
        got = lib.ggml_quantize_chunk(int(qtype), w.ctypes.data, out.ctypes.data, r * n, k, n, ip)
        if got != k * size:
            raise RuntimeError(f"ggml_quantize_chunk wrote {got} bytes, {k * size} expected")

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(chunk, range(0, rows, step)))
    return out


def decode(lib, q: np.ndarray, qtype: Q, shape) -> np.ndarray:
    out = np.empty(shape, np.float32)
    getattr(lib, f"dequantize_row_{DECODE[qtype]}")(q.ctypes.data, out.ctypes.data, out.size)
    return out


def measure(wd: torch.Tensor, d: np.ndarray, imp64: torch.Tensor | None, colw: torch.Tensor) -> tuple[float, float]:
    """(plain error, activation-weighted error) of the decoded d against the master wd (float64)."""
    dd = torch.from_numpy(d).double() - wd
    cold = dd.square().sum(0)
    plain = float(cold.sum().sqrt() / colw.sum().sqrt())
    if imp64 is None:
        return plain, plain
    return plain, float(((imp64 @ cold) / (imp64 @ colw)).sqrt())


def pinned_imatrix(a) -> Path:
    """--model's importance file (IMATRICES), under its own name: read where an --imatrices directory has it, else
    downloaded into --cache; refused unless its size and SHA-256 are the pinned ones."""
    name, size, digest = IMATRICES[a.model]
    p = find_or_fetch(name, hf_url(IMATRIX_REPO, IMATRIX_REVISION, name), digest, size, a.imatrices, a.cache)
    log(f"importance: {name}, {size} bytes, SHA-256 {digest}: the pinned file ({IMATRIX_REPO} at "
        f"{IMATRIX_REVISION[:8]})")
    return p


def load_imatrix(spec: str, a, blocks: list, sd: dict) -> tuple[dict, dict]:
    """{weight name: float32 importance [in]} and a description of the source."""
    if spec == "ones":
        return {k: np.ones(sd[k].shape[1], np.float32) for k in blocks}, {"kind": "ones", "sha256": "ones"}
    from safetensors import safe_open

    p = Path(spec)
    with safe_open(str(p), framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        if meta.get("format") != IMATRIX_FORMAT:
            raise SystemExit(f"{p}: format {meta.get('format')!r}, {IMATRIX_FORMAT!r} expected")
        model = json.loads(meta.get("model") or "{}")
        want = MODELS[a.model][1]
        if model.get("name") != want:
            raise SystemExit(f"{p}: collected on {model.get('name')}, {want} expected for --model {a.model}")
        imp, zeros = {}, 0
        for k in blocks:
            s = f.get_tensor(f"{k}.in_sum2").double()
            c = int(f.get_tensor(f"{k}.counts")[0])
            if s.shape != (sd[k].shape[1],) or c <= 0:
                raise SystemExit(f"{p}: {k}: sums {tuple(s.shape)}, count {c}")
            v = s / c
            if not bool(torch.isfinite(v).all()) or bool((v < 0).any()):
                raise SystemExit(f"{p}: {k}: non-finite or negative importance")
            zeros += int((v == 0).sum())
            imp[k] = np.ascontiguousarray(v.float().numpy())
    runs = json.loads(meta.get("runs") or "[]")
    # files by name: the importance file's, the fp16 file's and the clips' (their paths stay in the report, "path")
    src = {"kind": "file", "file": p.name, "sha256": sha256(p),
           "model": {k: model[k] for k in ("name", "size") if k in model},
           "inputs": [Path(r["input"]).name if r.get("input") else None for r in runs], "runs": len(runs),
           "tokens": [r.get("tokens") for r in runs], "numz_commit": sorted({str(r.get("numz_commit")) for r in runs}),
           "zero_channels": zeros, "path": str(p.resolve())}
    log(f"importance: {src['path']} ({src['sha256'][:12]}), {len(runs)} runs on {src['inputs']}, {zeros} channels "
        "at 0")
    return imp, src


def hull(points: list) -> list:
    """The lower convex hull of [(bytes, d, type)], by increasing bytes and decreasing d."""
    h = []
    for b, d, t in sorted(points, key=lambda p: (p[0], p[1])):
        if h and d >= h[-1][1]:
            continue  # more bytes, no less error
        while len(h) >= 2:
            (b1, d1, _), (b2, d2, _) = h[-2], h[-1]
            if (d2 - d1) * (b - b1) >= (d - d1) * (b2 - b1):
                h.pop()  # on or above the chord: never the best at any price
            else:
                break
        h.append((b, d, t))
    return h


def choose(rows: dict, types: list, budget: int) -> tuple[dict, int, float]:
    """One type per matrix: the greedy over hulls under budget bytes. ({name: type}, bytes, last price)."""
    hulls, at = {}, {}
    for n in sorted(rows):
        pts = [(rows[n][t]["bytes"], 0.0 if rows[n]["dead"] else rows[n][t]["werr"] ** 2, t) for t in types]
        hulls[n] = hull(pts)
        at[n] = 0
    used = sum(hulls[n][0][0] for n in hulls)
    if used > budget:
        raise SystemExit(f"the cheapest types take {used} bytes, over the budget of {budget}")
    heap = []

    def push(n):
        i = at[n]
        if i + 1 < len(hulls[n]):
            (b0, d0, _), (b1, d1, _) = hulls[n][i], hulls[n][i + 1]
            heapq.heappush(heap, (-(d0 - d1) / (b1 - b0), n))

    for n in sorted(hulls):
        push(n)
    price = 0.0
    while heap:
        g, n = heapq.heappop(heap)
        i = at[n]
        step = hulls[n][i + 1][0] - hulls[n][i][0]
        if used + step <= budget:
            used += step
            at[n] = i + 1
            price = -g
            push(n)
    return {n: hulls[n][at[n]][2] for n in hulls}, used, price


def stats(v: list) -> dict:
    return {"median": statistics.median(v), "worst": max(v), "mean": statistics.fmean(v)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--imatrix", help="imatrix_hook.py's file, or 'ones' (a test placeholder); default: --model's "
                    "pinned file (IMATRICES)")
    ap.add_argument("--imatrices", type=Path, action="append", default=[],
                    help="a directory holding our importance files (read in place)")
    ap.add_argument("--model", choices=tuple(MODELS), default="7b")
    ap.add_argument("--types", default=",".join(DEFAULT_TYPES))
    ap.add_argument("--masters", type=Path, action="append", default=[])
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--work", type=Path, default=CACHE / "ggml")
    ap.add_argument("--static", type=Path, help="our static Q4_K file: its bytes checked equal to the reference")
    ap.add_argument("--out", type=Path, default=DIST)
    ap.add_argument("--only", help="a regex: only these block matrices, nothing written")
    ap.add_argument("--probe", action="store_true", help="with --only: no importance, ones, the importance, per type")
    ap.add_argument("--table", type=Path, help="an earlier run's measures with the same inputs")
    ap.add_argument("--margin", type=int, default=1 << 20)
    ap.add_argument("--threads", type=int, default=len(os.sched_getaffinity(0)))
    a = ap.parse_args()
    types = [t.strip() for t in a.types.split(",") if t.strip()]
    for t in types:
        if t not in NUMZ_TYPES or t == "BF16":
            ap.error(f"--types {t}: not one of numz's quantized types {', '.join(NUMZ_TYPES[:-1])}")
    if "Q4_K" not in types:
        ap.error("--types: Q4_K is needed (the uniform file, the budget's reference)")
    if a.probe and not a.only:
        ap.error("--probe needs --only")
    torch.set_num_threads(a.threads)
    t0 = time.time()
    lib = ggml(a.work, a.threads)
    setup(lib)
    log(f"torch {torch.__version__}, numpy {np.__version__}, gguf {version('gguf')}, {a.threads} threads; "
        f"candidates {', '.join(types)}")
    fp16_name, imatrix_model, stem = MODELS[a.model]
    f = FILES[fp16_name]
    imatrix = str(pinned_imatrix(a)) if a.imatrix is None else a.imatrix  # checked before the master is read
    master = find_or_fetch(f.master, hf_url(REPO, REVISION, f.master), f.master_sha256, f.master_size,
                           a.masters, a.cache)
    sd = load_master(master)
    blocks = [k for k in sd if BLOCK.match(k)]
    if len(blocks) != 288:
        raise SystemExit(f"{master}: {len(blocks)} block matrices, 288 expected")
    sel = [k for k in blocks if not a.only or re.search(a.only, k)]
    imp, src = load_imatrix(imatrix, a, blocks, sd)
    static = {}
    if a.static:
        static = {t.name: t for t in gguf.GGUFReader(a.static).tensors}
    a.out.mkdir(parents=True, exist_ok=True)
    key = {"master": f.master_sha256, "imatrix": src["sha256"], "types": types, "llama.cpp": LLAMA_CPP,
           "matrices": sel}
    rows = {}
    if a.table and a.table.exists():
        tab = json.loads(a.table.read_text())
        if tab.get("key") != key:
            raise SystemExit(f"{a.table}: measured from other inputs")
        rows = tab["rows"]
        log(f"measures read from {a.table}: {len(rows)} matrices")
    # ---------------------------------------------------------------- measures
    t1 = time.time()
    probe = {}
    for i, k in enumerate(sel):
        if k in rows and not a.probe:
            continue
        w = sd[k].contiguous()
        x = np.ascontiguousarray(w.numpy())
        wd = w.double()
        colw = wd.square().sum(0)
        imp64 = torch.from_numpy(imp[k]).double()
        row = {"shape": list(w.shape), "dead": bool(DEAD.match(k))}
        for t in types:
            qt = Q[t]
            q = quantize(lib, x, qt, imp[k], a.threads)
            e, we = measure(wd, decode(lib, q, qt, x.shape), imp64, colw)
            row[t] = {"bytes": int(q.nbytes), "err": e, "werr": we,
                      "blake2b": hashlib.blake2b(q.tobytes(), digest_size=16).hexdigest()}
        q = quantize(lib, x, Q.Q4_K, None, a.threads)
        e, we = measure(wd, decode(lib, q, Q.Q4_K, x.shape), imp64, colw)
        row["ref"] = {"bytes": int(q.nbytes), "err": e, "werr": we}
        if static:
            st = static[k]
            row["ref"]["equals_static"] = bool(st.tensor_type == Q.Q4_K and np.array_equal(
                np.asarray(st.data).view(np.uint8).reshape(-1), q.reshape(-1)))
        if a.probe:
            pr = {}
            for t in types:
                qt = Q[t]
                qn = quantize(lib, x, qt, None, a.threads)
                q1 = quantize(lib, x, qt, np.ones(x.shape[1], np.float32), a.threads)
                qi = quantize(lib, x, qt, imp[k], a.threads)
                m3 = [measure(wd, decode(lib, qq, qt, x.shape), imp64, colw) for qq in (qn, q1, qi)]
                pr[t] = {
                    "ones_equals_none": bool(np.array_equal(qn, q1)),
                    "imatrix_equals_none": bool(np.array_equal(qn, qi)),
                    "err_none_ones_imatrix": [m[0] for m in m3],
                    "werr_none_ones_imatrix": [m[1] for m in m3],
                }
            probe[k] = pr
        rows[k] = row
        if (i + 1) % 24 == 0 or i + 1 == len(sel):
            log(f"measured {i + 1}/{len(sel)} matrices, {time.time() - t1:.0f} s")
        del wd, colw
    table = a.out / f"{stem}_dyn.table.json"
    table.write_text(json.dumps({"key": key, "rows": rows}, indent=0, sort_keys=True))
    if static:
        same = sum(rows[k]["ref"].get("equals_static", False) for k in sel)
        log(f"reference Q4_K without importance = {a.static.name}'s bytes on {same} of {len(sel)} matrices")
        if same != len(sel):
            raise SystemExit("the reference is not our static Q4_K")
    # ---------------------------------------------------------------- the choice
    q4k_bytes = sum(rows[k]["Q4_K"]["bytes"] for k in sel)
    budget = q4k_bytes - (a.margin if not a.only else 0)
    pick, used, price = choose({k: rows[k] for k in sel}, types, budget)
    hist = {t: sum(1 for k in sel if pick[k] == t) for t in types}
    hist_bytes = {t: sum(rows[k][t]["bytes"] for k in sel if pick[k] == t) for t in types}
    summary = {"matrices": len(sel), "budget": budget, "q4k_bytes": q4k_bytes, "used": used,
               "slack": budget - used, "last_price": price, "types": hist, "type_bytes": hist_bytes,
               "dead": [k for k in sel if rows[k]["dead"]]}
    for label, get in (("dyn", lambda k: rows[k][pick[k]]), ("q4k_imatrix", lambda k: rows[k]["Q4_K"]),
                       ("q4k_static", lambda k: rows[k]["ref"])):
        live = [k for k in sel if not rows[k]["dead"]]
        summary[label] = {"err": stats([get(k)["err"] for k in sel]), "werr": stats([get(k)["werr"] for k in sel]),
                          "objective_sum_werr2_live": sum(get(k)["werr"] ** 2 for k in live)}
    for t in types:
        summary[f"all_{t}"] = {"err": stats([rows[k][t]["err"] for k in sel]),
                               "werr": stats([rows[k][t]["werr"] for k in sel]),
                               "bytes": sum(rows[k][t]["bytes"] for k in sel)}
    better = sum(rows[k][pick[k]]["werr"] < rows[k]["ref"]["werr"] for k in sel)
    summary["dyn_lower_werr_than_static_on"] = better
    summary["q4k_imatrix_lower_werr_than_static_on"] = sum(rows[k]["Q4_K"]["werr"] < rows[k]["ref"]["werr"]
                                                         for k in sel)
    summary["q4k_imatrix_lower_err_than_static_on"] = sum(rows[k]["Q4_K"]["err"] < rows[k]["ref"]["err"] for k in sel)
    report = {"key": key, "imatrix": src, "summary": summary, "choice": {k: pick[k] for k in sel}, "probe": probe,
              "seconds_measures": round(time.time() - t1, 1)}
    log(f"choice: {hist} in {used} bytes of {budget} (Q4_K's {q4k_bytes}); objective sum e^2 "
        f"{summary['dyn']['objective_sum_werr2_live']:.5f} (uniform Q4_K with importance "
        f"{summary['q4k_imatrix']['objective_sum_werr2_live']:.5f}, our static Q4_K "
        f"{summary['q4k_static']['objective_sum_werr2_live']:.5f})")
    for label in ("dyn", "q4k_imatrix", "q4k_static"):
        s = summary[label]
        log(f"  {label}: plain error median {s['err']['median']:.5f} worst {s['err']['worst']:.5f}; "
            f"weighted median {s['werr']['median']:.5f} worst {s['werr']['worst']:.5f}")
    rep = a.out / f"{stem}_dyn.report.json"
    if a.only:
        rep.write_text(json.dumps(report, indent=1, sort_keys=True))
        log(f"--only: nothing written but {rep} and {table}; {time.time() - t0:.0f} s")
        return
    # ---------------------------------------------------------------- the files
    outs = {f"{stem}_dyn.gguf": pick, f"{stem}_Q4_K_imatrix.gguf": {k: "Q4_K" for k in blocks}}
    files = {}
    for name, types_of in outs.items():
        out = a.out / name
        part = out.with_name(out.name + ".part")
        w = gguf.GGUFWriter(str(part), arch="seedvr", use_temp_file=False)
        w.add_name(name.removesuffix(".gguf"))
        w.add_license("apache-2.0")
        dyn = name.endswith("_dyn.gguf")
        w.add_file_type(int(gguf.LlamaFileType.GUESSED if dyn else gguf.LlamaFileType.MOSTLY_Q4_K_S))
        w.add_quantization_version(gguf.GGML_QUANT_VERSION)
        w.add_source_url(hf_url(REPO, REVISION, f.master))
        w.add_string("seedvr2x.source", f"{f.master} of {REPO}, revision {REVISION}")
        w.add_string("seedvr2x.source_sha256", f.master_sha256)
        w.add_string("seedvr2x.copyright", COPYRIGHT)
        how = (f"each to the type among {', '.join(types)} that minimises the summed activation-weighted "
               "squared relative error of the 288 under the bytes of the uniform Q4_K"
               if dyn else "to Q4_K")
        w.add_string(
            "seedvr2x.change",
            f"the 288 attention and MLP matrices of the blocks quantized by ggml_quantize_chunk (llama.cpp "
            f"{LLAMA_CPP}) with an importance matrix (each input channel's mean square over the calibration "
            f"runs of {imatrix_model} in numz's SeedVR2) from the float32 master, {how}; every other tensor "
            "rounded to the nearest float16")
        w.add_string("seedvr2x.conversion", "unofficial, not ByteDance's: made by seedvr2x's models/seedvr2_gguf_dyn.py")
        w.add_string("seedvr2x.imatrix", json.dumps({k: src[k] for k in IMATRIX_META if k in src}, sort_keys=True))
        w.add_string("quantize.imatrix.file", src.get("file", "ones"))
        w.add_string("quantize.imatrix.dataset", json.dumps(src.get("inputs", [])))
        w.add_uint32("quantize.imatrix.entries_count", 288)
        w.add_uint32("quantize.imatrix.chunks_count", int(src.get("runs", 0)))
        files[name] = (out, part, w, types_of, {}, {})
    t2 = time.time()
    differ = 0
    for k, v in sd.items():
        x = np.ascontiguousarray(v.numpy())
        if x.ndim == 0:
            raise SystemExit(f"{k}: a scalar, which GGUF can't hold")
        if k in blocks:
            done = {}
            for name, (out, part, w, types_of, written, errs) in files.items():
                t = types_of[k]
                if t not in done:
                    qt = Q[t]
                    q = quantize(lib, x, qt, imp[k], a.threads)
                    if hashlib.blake2b(q.tobytes(), digest_size=16).hexdigest() != rows[k][t]["blake2b"]:
                        raise SystemExit(f"{k} {t}: quantized differently the second time")
                    d1 = decode(lib, q, qt, x.shape)
                    d2 = gguf.quants.dequantize(q, qt).reshape(x.shape)
                    if not np.array_equal(d1.view(np.uint32), d2.view(np.uint32)):
                        differ += 1
                        log(f"{k} {t}: ggml and gguf-py decode differently")
                    done[t] = q
                w.add_tensor(k, done[t], raw_dtype=Q[t])
                written[k] = done[t]
                errs[k] = (rows[k][t]["err"], rows[k][t]["werr"])
        else:
            h = v.contiguous().to(torch.float16).numpy()
            for name, (out, part, w, types_of, written, errs) in files.items():
                w.add_tensor(k, h)
                written[k] = h
    st_name, st_size = STATIC_Q4K[a.model]
    st_size = a.static.stat().st_size if a.static else st_size
    results = {}
    for name, (out, part, w, types_of, written, errs) in files.items():
        try:
            if differ:
                raise SystemExit(f"{differ} tensors decoded differently by ggml and gguf-py")
            w.write_header_to_file()
            w.write_kv_data_to_file()
            w.write_tensors_to_file()
            w.close()
            os.replace(part, out)
            r = gguf.GGUFReader(out)
            for fld in r.fields.values():
                if fld.types and fld.types[0] == gguf.GGUFValueType.STRING:
                    v = fld.contents()
                    if HOST_PATH.search(v) or TIME.search(v):
                        raise SystemExit(f"{out}: {fld.name} holds a path or a time: {v[:160]!r}")
            if {t.name for t in r.tensors} != set(sd):
                raise SystemExit(f"{out}: tensor names read back differ")
            for t in r.tensors:
                want = Q[types_of[t.name]] if t.name in blocks else Q.F16
                shape = tuple(int(s) for s in reversed(t.shape.tolist()))
                if (t.tensor_type != want or shape != tuple(sd[t.name].shape)
                        or not np.array_equal(np.asarray(t.data).view(np.uint8).reshape(-1),
                                              written[t.name].view(np.uint8).reshape(-1))):
                    raise SystemExit(f"{out}: {t.name} read back differs")
            size = out.stat().st_size
            if name.endswith("_dyn.gguf") and size > st_size:
                raise SystemExit(f"{out}: {size} bytes, over our Q4_K's {st_size}")
            digest = sha256(out)
            pin = OUTPUTS.get((name, src["sha256"]))
            if pin and pin != digest:
                raise SystemExit(f"{out}: SHA-256 {digest}, {pin} pinned")
            e = [v[0] for v in errs.values()]
            we = [v[1] for v in errs.values()]
            results[name] = {"bytes": size, "q4k_static_bytes": st_size, "sha256": digest, "pinned": bool(pin),
                             "err": stats(e), "werr": stats(we),
                             "types": {t: sum(1 for kk in blocks if types_of[kk] == t) for t in types}}
            log(f"{name}: {size} bytes ({size - st_size:+d} against our Q4_K's {st_size}); ggml = gguf-py on every "
                f"quantized tensor; read back: every name, type, shape and byte; error median "
                f"{statistics.median(e):.5f} worst {max(e):.5f}, weighted median {statistics.median(we):.5f} "
                f"worst {max(we):.5f}; SHA-256 {digest}{' (pinned)' if pin else ''}")
        except BaseException:
            part.unlink(missing_ok=True)
            out.unlink(missing_ok=True)
            raise
    report["files"] = results
    report["seconds_files"] = round(time.time() - t2, 1)
    report["seconds"] = round(time.time() - t0, 1)
    rep.write_text(json.dumps(report, indent=1, sort_keys=True))
    log(f"report {rep}; done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

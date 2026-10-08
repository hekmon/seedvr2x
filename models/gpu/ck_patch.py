#!/usr/bin/env python3
"""Our quantized DiT files (models/FORMATS.md) in numz's SeedVR2 CLI, through comfy-kitchen's
layers, and our own model directory, patched in at import (numz's checkout unchanged).

    cd NUMZ && PYTHONPATH=PYLIB CK_PATCH_MODE=MODE python [WRAPPER.py ...] ck_patch.py \\
        inference_cli.py ARGS --model_dir DIR --dit_model FILE
    bench.py run NAME --seedvr2-dir NUMZ --wrap colour_dump.py --wrap ck_patch.py \\
        --env PYTHONPATH=PYLIB --env CK_PATCH_MODE=MODE ... -- LR ... --model_dir DIR --dit_model FILE

NUMZ is numz's ComfyUI-SeedVR2_VideoUpscaler at 4490bd1 with its environment (Apache-2.0: the hook
points below are its functions); PYLIB a directory holding comfy-kitchen 0.2.37 (Comfy-Org,
Apache-2.0, be003b7), the run-time dependency that multiplies, installed for numz's interpreter
outside numz's venv (`uv pip install --python NUMZ/.venv/bin/python --target PYLIB --no-deps
comfy-kitchen==0.2.37`). As research/scripts/numerics_patch.py does, the wrapper rewrites argv,
runs the next script with runpy, and patches numz's modules right after they execute (a
sys.meta_path finder that hands each module to the finders after it, so the wrappers of a chain
all apply): nothing when the DiT file carries no comfy-kitchen marker, but the model directory.

Why: numz loads a safetensors DiT with load_state_dict(strict=False, assign=True)
(src/core/model_loader.py, _load_standard_weights): a comfy-kitchen layer's scales and markers
would be dropped without a word, and its float8 values read as weights 4,000 times too large;
and numz's --dit_model takes a registry name or a file of ./models/SEEDVR2 under its cwd only
(inference_cli.py, the argument's choices), --model_dir being a download target.

1. The model directory. With --model_dir DIR, every model file of DIR whose name is not in numz's
   registry joins numz's list (src/utils/constants.get_all_model_files), ahead of ./models/SEEDVR2:
   --dit_model accepts it, and numz loads exactly DIR/NAME, safetensors or GGUF (a GGUF file goes
   through numz's own GGUF loader, untouched). Registry names resolve as numz resolves them. The
   7B's configuration is numz's choice, by "7b" in the name. The DiT's and the VAE's paths are
   logged at load; the DiT's must be DIR/NAME when NAME is one of DIR's.
2. The layers. numz's _load_standard_weights receives the whole state on the target device. Every
   layer L with an `L.comfy_quant` marker loses its `L.*` tensors from it (L.weight, L.weight_scale,
   L.weight_scale_2, the marker; L.bias kept for the new module), numz's function loads the rest,
   and L becomes a CKLinear holding the file's tensors as they are. No marker: numz's function
   runs unchanged. Markers name the layout: {"format": "float8_e4m3fn"} (TensorCoreFP8Layout),
   {"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": G} (TensorWiseINT8Layout,
   rotated), {"format": "nvfp4"} (TensorCoreNVFP4Layout); one format per file. Only marked layers
   change (in our files the 288 block matrices): vid_in, whose weight is the model's first
   parameter and so drives numz's CompatibleDiT and Phase 2's autocast, the norms and RoPE stay
   numz's.
3. CKLinear keeps its tensors as plain buffers and builds comfy-kitchen's QuantizedTensor at each
   forward, so that numz's moves (Module.to, BlockSwap's .data reassignments, the release's
   .data.set_()) work as for any buffer; its _apply moves the buffers but never changes their
   dtype (a model.to(dtype) or .half() would cast fp8 values and float32 scales otherwise). Its
   forward runs with autocast off: the compute dtype is the one numz's autocast gives a Linear at
   that call (bfloat16 in numz's Phase 2), or the input's when autocast is off; the input (any
   leading dims, flattened to 2-D) and the bias are cast to it, and the output is in it.

CK_PATCH_MODE (required when the file carries markers, refused when it carries none):
  w8a16  fp8 file: F.linear(x, W, b): comfy-kitchen dequantizes W (its dequantize_per_tensor_fp8),
         a 16-bit matmul (torch.addmm, torch.mm without a bias).
  w8a8   fp8 file: x quantized per tensor (QuantizedTensor.from_float, scale max|x| / 448, at run
         time; in row chunks above 2^31 elements, see below), then F.linear: comfy-kitchen's addmm
         and mm handlers call torch's scaled_mm, float32 accumulation, the bias in the epilogue.
  int8   rotated INT8 file: F.linear(x, W, b): comfy-kitchen's int8_linear rotates x as the
         weights were rotated, quantizes it per token (max / 127) and multiplies in int8.
  w4a4   NVFP4 file: x quantized to NVFP4 (from_float, scale max|x| / (448 x 6); in row chunks
         above 2^31 elements, see below), then
         F.linear(xq, W) WITHOUT the bias, and the bias added after, in the compute dtype:
         comfy-kitchen has no NVFP4 addmm handler, and F.linear with a bias reaches aten.addmm,
         where its generic fallback dequantizes both operands (its test_nvfp4_addmm_fallback); its
         mm handler calls scaled_mm_nvfp4 (cuBLASLt FP4 on Blackwell). The GPU only.
  w4a16  NVFP4 file: F.linear(x, W, b): comfy-kitchen dequantizes W (dequantize_nvfp4).
  In w8a8, int8 and w4a4, any dequantization of a comfy-kitchen tensor during the layer's multiply
  aborts the run: comfy-kitchen's handlers fall back to dequantizing when its kernel fails, some
  without a word.
  Row chunks (w8a8, w4a4): comfy-kitchen 0.2.37's CUDA activation quantizers index in 32 bits, so
  from 2^32 elements on they go wrong without a word: the fp8 one (per_tensor_quantize.cu) takes the
  element count as a uint32_t and its index is one, and leaves the last 2^32 elements of its output
  unwritten (whatever memory torch.empty gave: NaN bytes among it, and one NaN makes the next
  layer's per-tensor scale NaN: the whole output NaN); the NVFP4 one (quantize_nvfp4.cu's aligned
  path: rows and columns multiples of 16) reads input + kValsPerThread * idx in unsigned int, and
  quantizes, from 2^32 elements on, the values found 2^32 elements earlier. A 4K window's
  mlp.proj_out input holds about 350,000 to 390,000 tokens of 12288 values (4.3e9 to 4.8e9); at
  1080p, 97,920 tokens (1.2e9).
  Above 2^31 - 1 elements, x goes to comfy-kitchen in chunks of a multiple of 128 rows under 2^31
  elements, each through comfy-kitchen's own quantize with the scale comfy-kitchen's formula gives
  for the WHOLE tensor (quantize_x): its values bit for bit wherever its kernel is right. Below,
  from_float as it is.
CK_PATCH_EXPECT=N    abort unless exactly N layers become CKLinear (288 for our 7B files).
CK_PATCH_LOG=PATH    a JSON report at exit: files, format, mode, layers, comfy-kitchen's version
                     and backends, the compute dtypes met, the check mode's results.
CK_PATCH_CHECK=1     the check mode (a smoke test's, nothing otherwise): at each CKLinear's first
                     forward, on up to CK_PATCH_CHECK_ROWS rows of its input (16384, evenly
                     spaced), in float32 (TF32 off): y_deq = x W_deq^T + b, W_deq the file's
                     tensors decoded here (not by comfy-kitchen), and y_16 = x W16^T + b, W16 the
                     tensor of the fp16 file CK_PATCH_FP16 (read one tensor at a time); per layer
                     ||y - y_deq|| / ||y_deq|| (the kernel and the activations' rounding),
                     ||y - y_16|| / ||y_16|| (the whole format against fp16, on real activations),
                     ||y_deq - y_16|| / ||y_16|| (the weights alone) and the floor, the fp16 file's
                     layer as numz computes it (x, W16, b in the compute dtype) against y_16; and
                     the proof of the path: the aten ops comfy-kitchen's tensor saw, the backend
                     comfy-kitchen's registry chose for each of its functions, the calls to torch's
                     scaled_mm, _scaled_mm, _int_mm, addmm, mm, F.linear made inside, and the
                     dequantizations. Median and worst per kind of layer in the JSON. At load, every
                     tensor numz's loader receives and every kept bias must equal the fp16 file's.
CK_PATCH_DRYRUN=1    stop once numz has resolved the DiT's and the VAE's files and built their
                     structures on the meta device, before any weight is read: prints both paths.

Refused with markers (the run aborts before Phase 1): an unknown or mixed format, a missing or
wrong mode, comfy-kitchen missing or not 0.2.37, --compile_dit (torch.compile wraps numz's
CompatibleDiT), several GPUs (--cuda_device 0,1: numz's workers would load unpatched), a
dtype override of the DiT's state (numz's _convert_state_dtype would cast fp8 values and scales).
"""

from __future__ import annotations

import argparse
import atexit
import importlib.abc
import json
import os
import re
import runpy
import statistics
import struct
import sys
import threading
import time

MARK = ".comfy_quant"
FORMATS = {  # marker format -> (comfy-kitchen layout, modes)
    "float8_e4m3fn": ("TensorCoreFP8Layout", ("w8a16", "w8a8")),
    "int8_tensorwise": ("TensorWiseINT8Layout", ("int8",)),
    "nvfp4": ("TensorCoreNVFP4Layout", ("w4a4", "w4a16")),
}
QUANT_X = {"w8a8": "TensorCoreFP8Layout", "w4a4": "TensorCoreNVFP4Layout"}
X_LIMIT = 2**31 - 1  # elements per call of comfy-kitchen's activation quantizers, at most (quantize_x)
NO_DEQUANT = ("w8a8", "int8", "w4a4")
CK_VERSION = "0.2.37"
SUFFIXES = ("weight", "weight_scale", "weight_scale_2", "comfy_quant")
DTYPES = {"F16": 2, "BF16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1, "I8": 1, "F64": 8, "I64": 8}


class State:
    def __init__(self):
        self.mode = os.environ.get("CK_PATCH_MODE") or None
        self.expect = os.environ.get("CK_PATCH_EXPECT") or None
        self.log_path = os.environ.get("CK_PATCH_LOG") or None
        self.check = os.environ.get("CK_PATCH_CHECK", "0") not in ("", "0")
        self.fp16 = os.environ.get("CK_PATCH_FP16") or None
        self.rows = int(os.environ.get("CK_PATCH_CHECK_ROWS", "16384"))
        self.dryrun = os.environ.get("CK_PATCH_DRYRUN", "0") not in ("", "0")
        self.model_dir = None  # --model_dir, absolute
        self.args = None  # numz's parsed arguments
        self.dit = None  # what preflight found: {"name", "path", "markers", "format"}
        self.paths = {}  # model type -> path numz loads
        self.report = {"patch": os.path.abspath(__file__), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        self.patched = []
        self.aborted = None
        self.no_dequant = False  # inside a CKLinear multiply in a mode that must not dequantize
        self.override_dtype = None  # numz's dtype override of the DiT's state, if any
        self.rec = None  # the check mode's recorder while a first forward runs
        self.cdt_seen = {}
        self.refused_casts = 0
        self.checks = []
        self.x_chunked = {"calls": 0, "chunks": 0, "largest_elements": 0}  # quantize_x's chunked calls
        self.fp16_file = None
        self.ck = None
        self.finder_busy = threading.local()


S = State()


def log(msg: str) -> None:
    print(f"ck_patch: {msg}", file=sys.stderr, flush=True)


def abort(msg: str):
    """Stop the run: SystemExit passes numz's `except Exception` handlers."""
    S.aborted = msg
    log("ABORT: " + msg)
    raise SystemExit(3)


# ---------------------------------------------------------------- files

def st_header(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def st_markers(path: str) -> dict:
    """{layer: marker JSON} of a safetensors file, read with the standard library only."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        out = {}
        for k, v in sorted(hdr.items()):
            if k.endswith(MARK) and isinstance(v, dict) and v.get("dtype") == "U8":
                a, b = v["data_offsets"]
                f.seek(8 + n + a)
                out[k[: -len(MARK)]] = json.loads(f.read(b - a).decode())
        return out


def file_format(markers: dict, where: str) -> str | None:
    """The one layout of a file's markers, or None without markers; aborts on anything else."""
    kinds = {json.dumps(m, sort_keys=True) for m in markers.values()}
    if not kinds:
        return None
    if len(kinds) > 1:
        abort(f"{where}: {len(kinds)} different comfy_quant formats ({', '.join(sorted(kinds))}): one per run")
    m = json.loads(kinds.pop())
    fmt = m.get("format")
    ok = (
        m == {"format": "float8_e4m3fn"}
        or m == {"format": "nvfp4"}
        or (
            fmt == "int8_tensorwise"
            and set(m) == {"format", "convrot", "convrot_groupsize"}
            and m["convrot"] is True
            and isinstance(m["convrot_groupsize"], int)
            and m["convrot_groupsize"] > 0
        )
    )
    if not ok:
        abort(f"{where}: comfy_quant {m}: not a format this patch knows")
    return fmt


# ---------------------------------------------------------------- preflight (numz's arguments)

def preparse_model_dir(argv: list[str]) -> None:
    """--model_dir, needed before numz builds its parser (the choices of --dit_model)."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--model_dir", default=None)
    a, _ = p.parse_known_args(argv)
    if a.model_dir:
        S.model_dir = os.path.abspath(a.model_dir)


def ck_backends() -> dict:
    import comfy_kitchen as ck

    import comfy_kitchen.backends.cuda as ckc

    b = ck.list_backends()
    return {
        "available": sorted(k for k, v in b.items() if v["available"] and not v["disabled"]),
        "unavailable": {k: v["unavailable_reason"] for k, v in sorted(b.items()) if not v["available"]},
        "cuda_extension_loaded": bool(ckc._EXT_AVAILABLE),
        "cuda_extension_error": ckc._EXT_ERROR,
        "cublaslt": bool(getattr(ckc, "_CUBLASLT_AVAILABLE", False)),
    }


def import_ck() -> None:
    """comfy-kitchen, once: its version, its backends, the dequantization guard."""
    if S.ck is not None:
        return
    from importlib.metadata import PackageNotFoundError, version

    try:
        v = version("comfy-kitchen")
    except PackageNotFoundError:
        abort(f"comfy-kitchen is not importable: install comfy-kitchen {CK_VERSION} for numz's interpreter, outside "
              f"its venv (uv pip install --python NUMZ/.venv/bin/python --target PYLIB --no-deps "
              f"comfy-kitchen=={CK_VERSION}), and run with PYTHONPATH=PYLIB")
    if v != CK_VERSION:
        abort(f"comfy-kitchen {v}: this patch's paths are checked for {CK_VERSION} only")
    import comfy_kitchen
    from comfy_kitchen.tensor import QuantizedTensor

    orig = QuantizedTensor.dequantize

    def dequantize(self):
        if S.rec is not None:
            S.rec["dequantize"] += 1
        if S.no_dequant:
            raise RuntimeError(
                f"ck_patch: comfy-kitchen fell back to dequantizing a {self._layout_cls} tensor in "
                f"{S.mode} mode: its {S.mode} kernel did not run"
            )
        return orig(self)

    QuantizedTensor.dequantize = dequantize
    S.ck = {"version": v, "path": os.path.dirname(comfy_kitchen.__file__), **ck_backends()}
    S.report["comfy_kitchen"] = S.ck


def registry_names() -> set | None:
    for name, mod in list(sys.modules.items()):
        if ("." + name).endswith(".src.utils.model_registry") and hasattr(mod, "MODEL_REGISTRY"):
            return set(mod.MODEL_REGISTRY)
    return None


def resolve(name: str) -> str:
    """The path numz will load for --dit_model NAME (after our model directory patch)."""
    reg = registry_names() or set()
    if S.model_dir and name not in reg and os.path.isfile(os.path.join(S.model_dir, name)):
        return os.path.join(S.model_dir, name)
    local = os.path.join(".", "models", "SEEDVR2", name)
    if os.path.isfile(local):
        return os.path.abspath(local)
    return os.path.join(S.model_dir or os.path.abspath(os.path.join(".", "models", "SEEDVR2")), name)


def preflight(args) -> None:
    """Every check that needs no weight, once numz has parsed its arguments."""
    S.args = args
    md = os.path.abspath(args.model_dir) if args.model_dir else None
    if md != S.model_dir:
        abort(f"--model_dir read as {S.model_dir} before numz's parser, {md} after: write it in full")
    fmt = expect_file(resolve(args.dit_model), args.dit_model)
    if fmt is None:
        return
    if getattr(args, "compile_dit", False):
        abort("--compile_dit: torch.compile would wrap numz's CompatibleDiT around comfy-kitchen's layers")
    cuda = getattr(args, "cuda_device", None)
    if cuda and len([d for d in str(cuda).split(",") if d.strip()]) > 1:
        abort(f"--cuda_device {cuda}: one GPU only (numz's workers would load the file unpatched)")


def expect_file(path: str, name: str | None = None) -> str | None:
    """The DiT file numz is to load: its markers' format against CK_PATCH_MODE and the other
    switches; comfy-kitchen imported when the file needs it. Returns the format (None: no marker)."""
    name = name or os.path.basename(path)
    markers = st_markers(path) if path.endswith(".safetensors") and os.path.isfile(path) else {}
    fmt = file_format(markers, path)
    S.dit = {"name": name, "path": path, "realpath": os.path.realpath(path), "layers_marked": len(markers),
             "format": fmt}
    S.report["dit"] = dict(S.dit)
    S.report["mode"] = S.mode
    if fmt is None:
        if S.mode:
            abort(f"CK_PATCH_MODE={S.mode} but {path} carries no comfy_quant marker "
                  f"({'GGUF: numz loads it' if path.endswith('.gguf') else 'numz loads it as it is'})")
        if S.check:
            abort(f"CK_PATCH_CHECK=1 but {path} carries no comfy_quant marker")
        log(f"DiT {name} -> {path}: no comfy_quant marker, numz's loader unchanged")
        return None
    modes = FORMATS[fmt][1]
    if S.mode not in modes:
        abort(f"{path}: format {fmt} ({len(markers)} layers): CK_PATCH_MODE must be one of {', '.join(modes)}"
              f" (got {S.mode!r})")
    if S.expect is not None and int(S.expect) != len(markers):
        abort(f"{path}: {len(markers)} marked layers, CK_PATCH_EXPECT={S.expect}")
    if S.check:
        if not S.fp16 or not os.path.isfile(S.fp16):
            abort(f"CK_PATCH_CHECK=1 needs CK_PATCH_FP16=<the fp16 file> (got {S.fp16!r})")
        hdr = st_header(S.fp16)
        bad = [L for L in markers if hdr.get(f"{L}.weight", {}).get("dtype") != "F16"]
        if bad:
            abort(f"CK_PATCH_FP16={S.fp16}: no float16 {bad[0]}.weight ({len(bad)} layers missing)")
    import_ck()
    log(f"DiT {name} -> {path} ({S.dit['realpath']}): {len(markers)} layers marked {fmt}, mode {S.mode}; "
        f"comfy-kitchen {S.ck['version']}, backends {','.join(S.ck['available'])}, CUDA extension "
        f"{'loaded' if S.ck['cuda_extension_loaded'] else 'NOT loaded: ' + str(S.ck['cuda_extension_error'])}"
        + (f"; check mode on, fp16 reference {S.fp16}" if S.check else ""))
    return fmt


# ---------------------------------------------------------------- the activations' quantizer

def quantize_x(xc, layout: str, limit: int | None = None):
    """QuantizedTensor.from_float(xc, layout) for the 2-D input xc, in row chunks when xc holds more
    than `limit` elements (X_LIMIT): comfy-kitchen 0.2.37's CUDA quantizers index in 32 bits and go
    wrong from 2^32 elements on, without a word (the module's docstring, "Row chunks"). The
    per-tensor scale is comfy-kitchen's own formula over the WHOLE tensor (TensorCoreFP8Layout.quantize,
    TensorCoreNVFP4Layout.quantize): the max of |x| in x's dtype (here the max of the chunks' maxima:
    the same value), divided by 448 (fp8) or 448 x 6 (NVFP4) in x's dtype, then float32; no epsilon.
    Each chunk, a multiple of 128 rows under `limit` elements (the last one shorter), goes through
    comfy-kitchen's own quantize with that scale, and the chunks are joined: fp8 data rows in order;
    NVFP4 data rows, and block scales, whose swizzled layout holds 128-row slabs one after the other
    (the last chunk's padding rows are the whole tensor's). The values are comfy-kitchen's wherever
    its kernel is right, bit for bit (ck_patch_test.py's xchunk part on the CPU; on the GPU, S12's
    proof against from_float)."""
    import torch
    from comfy_kitchen.float_utils import F4_E2M1_MAX, F8_E4M3_MAX
    from comfy_kitchen.tensor import QuantizedTensor, TensorCoreFP8Layout, TensorCoreNVFP4Layout

    limit = X_LIMIT if limit is None else limit
    if xc.numel() <= limit:
        return QuantizedTensor.from_float(xc, layout)
    m, k = xc.shape
    rows = limit // k // 128 * 128
    if rows < 128:
        abort(f"{layout}: rows of {k} values, too long for chunks of 128 rows under {limit} elements")
    spans = [(a, min(m, a + rows)) for a in range(0, m, rows)]
    amax = torch.stack([xc[a:b].abs().amax() for a, b in spans]).amax()  # torch.amax(xc.abs()), exactly
    if layout == "TensorCoreFP8Layout":
        scale = amax / torch.finfo(torch.float8_e4m3fn).max
        parts = [TensorCoreFP8Layout.quantize(xc[a:b], scale=scale) for a, b in spans]
        q = torch.cat([d.view(torch.uint8) for d, _ in parts]).view(torch.float8_e4m3fn)
        p = TensorCoreFP8Layout.Params(scale=parts[0][1].scale, orig_dtype=xc.dtype, orig_shape=(m, k))
    elif layout == "TensorCoreNVFP4Layout":
        scale = amax / (F8_E4M3_MAX * F4_E2M1_MAX)
        parts = [TensorCoreNVFP4Layout.quantize(xc[a:b], scale=scale) for a, b in spans]
        q = torch.cat([d for d, _ in parts])
        blocks = torch.cat([pp.block_scale.view(torch.uint8) for _, pp in parts]).view(torch.float8_e4m3fn)
        p = TensorCoreNVFP4Layout.Params(scale=parts[0][1].scale, orig_dtype=xc.dtype, orig_shape=(m, k),
                                         block_scale=blocks)
    else:
        abort(f"no chunked quantizer for {layout}")
    del parts
    S.x_chunked["calls"] += 1
    S.x_chunked["chunks"] += len(spans)
    S.x_chunked["largest_elements"] = max(S.x_chunked["largest_elements"], xc.numel())
    return QuantizedTensor(q, layout, p)


# ---------------------------------------------------------------- CKLinear

CKLinear = None
LINEAR = None


def ck_linear_class():
    """CKLinear, defined once torch is imported (by numz: never before it)."""
    global CKLinear, LINEAR
    if CKLinear is not None:
        return CKLinear
    import torch
    from comfy_kitchen.tensor import (
        QuantizedTensor,
        TensorCoreFP8Layout,
        TensorCoreNVFP4Layout,
        TensorWiseINT8Layout,
    )

    LINEAR = torch.nn.functional.linear  # captured: the check mode's counters see comfy-kitchen's calls only

    class _CKLinear(torch.nn.Module):
        """A comfy-kitchen layer: the file's tensors as buffers, the QuantizedTensor built per call."""

        def __init__(self, name, fmt, mode, tensors, bias, in_features, out_features):
            super().__init__()
            self.ck_name, self.ck_format, self.ck_mode = name, fmt["format"], mode
            self.ck_group = int(fmt.get("convrot_groupsize", 0))
            self.in_features, self.out_features = in_features, out_features
            for k in ("weight", "weight_scale", "weight_scale_2"):
                if k in tensors:
                    self.register_buffer(k, tensors[k])
            self.register_buffer("bias", bias)
            self.ck_checked = False

        def extra_repr(self):
            return (f"{self.in_features}, {self.out_features}, format={self.ck_format}, mode={self.ck_mode}, "
                    f"bias={self.bias is not None}")

        def _apply(self, fn, recurse=True):
            """Moves, never casts: fn runs on an empty tensor of each buffer's dtype for its target."""
            for key, buf in self._buffers.items():
                if buf is None:
                    continue
                probe = fn(buf.new_empty(0))
                if probe.dtype != buf.dtype:
                    S.refused_casts += 1
                if probe.device != buf.device:
                    self._buffers[key] = buf.to(device=probe.device)
            return self

        def qtensor(self, dtype):
            n, k = self.out_features, self.in_features
            if self.ck_format == "float8_e4m3fn":
                p = TensorCoreFP8Layout.Params(scale=self.weight_scale, orig_dtype=dtype, orig_shape=(n, k))
                return QuantizedTensor(self.weight, "TensorCoreFP8Layout", p)
            if self.ck_format == "int8_tensorwise":
                p = TensorWiseINT8Layout.Params(scale=self.weight_scale, orig_dtype=dtype, orig_shape=(n, k),
                                                is_weight=True, convrot=True, convrot_groupsize=self.ck_group)
                return QuantizedTensor(self.weight, "TensorWiseINT8Layout", p)
            p = TensorCoreNVFP4Layout.Params(scale=self.weight_scale_2, orig_dtype=dtype, orig_shape=(n, k),
                                             block_scale=self.weight_scale)
            return QuantizedTensor(self.weight, "TensorCoreNVFP4Layout", p)

        def multiply(self, xc, b, dtype):
            qt = self.qtensor(dtype)
            S.no_dequant = self.ck_mode in NO_DEQUANT
            try:
                if self.ck_mode in QUANT_X:
                    xq = quantize_x(xc, QUANT_X[self.ck_mode])
                    if self.ck_mode == "w4a4":
                        y = LINEAR(xq, qt)
                        return y if b is None else y + b
                    return LINEAR(xq, qt, b)
                return LINEAR(xc, qt, b)
            finally:
                S.no_dequant = False

        def forward(self, x):
            dev = x.device.type
            auto = torch.is_autocast_enabled(dev)
            dtype = torch.get_autocast_dtype(dev) if auto else x.dtype
            key = f"{dtype}{' (autocast)' if auto else ''}"
            S.cdt_seen[key] = S.cdt_seen.get(key, 0) + 1
            with torch.autocast(dev, enabled=False):
                x2 = x.reshape(-1, x.shape[-1])
                xc = x2.to(dtype)
                b = None if self.bias is None else self.bias.to(dtype)
                if S.check and not self.ck_checked:
                    self.ck_checked = True
                    y = first_call_check(self, x2, xc, b, dtype, key)
                else:
                    y = self.multiply(xc, b, dtype)
            return y.reshape(*x.shape[:-1], self.out_features)

    CKLinear = _CKLinear
    return CKLinear


# ---------------------------------------------------------------- the check mode

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def hadamard(n: int, device):
    """H4 (x) H4 (x) ... / sqrt(n), H4 the regular 4x4 Hadamard: comfy-kitchen's rotation, its own
    inverse (models/ck_check.py)."""
    import torch

    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.kron(h, h4)
    if h.shape[0] != n:
        abort(f"rotation size {n}: not a power of 4")
    return (h / n**0.5).to(device)


def decode(m):
    """The file's weight as float32, decoded here (models/ck_check.py's reconstruction)."""
    import torch

    q, s = m.weight, m.weight_scale.float()
    n, k = m.out_features, m.in_features
    if m.ck_format == "float8_e4m3fn":
        return q.float() * s
    if m.ck_format == "int8_tensorwise":
        g = m.ck_group
        return ((q.float() * s).reshape(n, k // g, g) @ hadamard(g, q.device)).reshape(n, k)
    from comfy_kitchen.float_utils import from_blocked

    lut = torch.tensor(E2M1, dtype=torch.float32, device=q.device)
    codes = torch.stack([q >> 4, q & 15], dim=-1).reshape(n, -1, 16)  # the even value's code high
    blocks = from_blocked(m.weight_scale, n, k // 16).float() * m.weight_scale_2.float()
    return (lut[codes.long()] * blocks.unsqueeze(2)).reshape(n, k)


class Recorder:
    """What serves a first forward: aten ops at comfy-kitchen's tensor, its registry's backends,
    torch's matmul entry points, dequantizations."""

    WRAPPED = (("torch.nn.functional", "scaled_mm"), ("torch", "_scaled_mm"), ("torch", "_int_mm"),
               ("torch", "addmm"), ("torch", "mm"), ("torch.nn.functional", "linear"))

    def __enter__(self):
        import logging

        import torch.nn.functional  # noqa: F401 (the modules wrapped below, in sys.modules)
        from comfy_kitchen.tensor import QuantizedTensor

        rec = {"aten": [], "ck": [], "torch": {}, "dequantize": 0}
        self.rec, self.saved = rec, []
        for modname, attr in self.WRAPPED:
            mod = sys.modules[modname]
            orig = getattr(mod, attr, None)
            if orig is None:
                continue

            def counted(*a, _orig=orig, _name=f"{modname.replace('torch.nn.functional', 'F')}.{attr}", **k):
                rec["torch"][_name] = rec["torch"].get(_name, 0) + 1
                return _orig(*a, **k)

            setattr(mod, attr, counted)
            self.saved.append((mod, attr, orig))
        td = QuantizedTensor.__dict__["__torch_dispatch__"]
        self.td = td

        def spy(cls, func, types, args=(), kwargs=None):
            rec["aten"].append(str(func))
            return td.__func__(cls, func, types, args, kwargs)

        QuantizedTensor.__torch_dispatch__ = classmethod(spy)
        lg = logging.getLogger("comfy_kitchen.dispatch")
        self.lg, self.lg_state = lg, (lg.level, lg.propagate)

        class H(logging.Handler):
            def emit(self, r):
                if r.args and len(r.args) >= 2:
                    rec["ck"].append(f"{r.args[0]}:{r.args[1]}")

        self.h = H(logging.DEBUG)
        lg.addHandler(self.h)
        lg.setLevel(logging.DEBUG)
        lg.propagate = False
        S.rec = rec
        return rec

    def __exit__(self, *exc):
        from comfy_kitchen.tensor import QuantizedTensor

        S.rec = None
        for mod, attr, orig in self.saved:
            setattr(mod, attr, orig)
        QuantizedTensor.__torch_dispatch__ = self.td
        self.lg.removeHandler(self.h)
        self.lg.setLevel(self.lg_state[0])
        self.lg.propagate = self.lg_state[1]
        return False


def proof_ok(mode: str, rec: dict) -> bool:
    t, ck, deq = rec["torch"], rec["ck"], rec["dequantize"]
    scaled = t.get("F.scaled_mm", 0) + t.get("torch._scaled_mm", 0)
    if mode in ("w8a16", "w4a16"):
        return deq >= 1 and scaled == 0 and t.get("torch._int_mm", 0) == 0
    if mode == "w8a8":
        return deq == 0 and scaled >= 1 and any(c.endswith(":quantize_per_tensor_fp8") for c in ck)
    if mode == "int8":
        eager = "eager:int8_linear" in ck
        return deq == 0 and any(c.endswith(":int8_linear") for c in ck) and (not eager or t.get("torch._int_mm", 0) >= 1)
    if mode == "w4a4":
        eager = "eager:scaled_mm_nvfp4" in ck
        return (deq == 0 and any(c.endswith(":scaled_mm_nvfp4") for c in ck)
                and any(c.endswith(":quantize_nvfp4") for c in ck) and (not eager or scaled >= 1))
    return False


def fp16_tensor(name: str):
    from safetensors import safe_open

    if S.fp16_file is None:
        S.fp16_file = safe_open(S.fp16, framework="pt", device="cpu")
    return S.fp16_file.get_tensor(name)


def rel(a, b) -> float:
    return float((a - b).norm() / b.norm()) if float(b.norm()) > 0 else float((a - b).norm())


def first_call_check(m, x2, xc, b, dtype, key):
    import torch

    with Recorder() as rec:
        y = m.multiply(xc, b, dtype)
    if x2.device.type == "cuda":
        torch.cuda.synchronize()
    rows = x2.shape[0]
    idx = torch.linspace(0, rows - 1, min(rows, S.rows), device=x2.device).round().long().unique()
    mm = torch.backends.cuda.matmul
    new_api = hasattr(mm, "fp32_precision")
    saved = mm.fp32_precision if new_api else (mm.allow_tf32, torch.get_float32_matmul_precision())
    if new_api:
        mm.fp32_precision = "ieee"
    else:
        mm.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    try:
        xs = x2[idx].float()
        ys = y[idx].float()
        bf = None if m.bias is None else m.bias.float()
        w_deq = decode(m)
        y_deq = LINEAR(xs, w_deq, bf)
        del w_deq
        w16 = fp16_tensor(m.ck_name + ".weight").to(x2.device)
        y_16 = LINEAR(xs, w16.float(), bf)
        floor = LINEAR(xs.to(dtype), w16.to(dtype), None if bf is None else bf.to(dtype)).float()
        r = {
            "layer": m.ck_name,
            "rows": int(idx.numel()),
            "of_rows": int(rows),
            "input_dtype": str(x2.dtype),
            "compute": key,
            "kernel": rel(ys, y_deq),
            "total": rel(ys, y_16),
            "weights": rel(y_deq, y_16),
            "floor": rel(floor, y_16),
            "finite": bool(torch.isfinite(ys).all()),
            "fp32_matmul": f"fp32_precision={mm.fp32_precision}" if new_api else f"allow_tf32={mm.allow_tf32}",
            "proof": rec,
        }
        r["proof_ok"] = proof_ok(m.ck_mode, rec)
        del y_deq, y_16, floor, w16, xs, ys
    finally:
        if new_api:
            mm.fp32_precision = saved
        else:
            mm.allow_tf32 = saved[0]
            torch.set_float32_matmul_precision(saved[1])
    S.checks.append(r)
    if not r["proof_ok"]:
        log(f"check: {m.ck_name}: the {m.ck_mode} path NOT proven: {rec}")
    return y


def kind(name: str) -> str:
    return re.sub(r"^blocks\.\d+\.", "", name)


def check_summary() -> dict:
    out = {"layers": len(S.checks), "proof_ok": sum(r["proof_ok"] for r in S.checks),
           "finite": sum(r["finite"] for r in S.checks)}
    if not S.checks:
        return out
    groups = {}
    for r in S.checks:
        groups.setdefault(kind(r["layer"]), []).append(r)
    groups["all"] = S.checks
    out["per_kind"] = {
        k: {m: {"median": statistics.median(x[m] for x in rs), "worst": max(x[m] for x in rs),
                "worst_layer": max(rs, key=lambda x: x[m])["layer"]}
            for m in ("kernel", "total", "weights", "floor")} | {"layers": len(rs)}
        for k, rs in sorted(groups.items())
    }
    sig = {}
    for r in S.checks:
        p = r["proof"]
        s = json.dumps({"aten": sorted(set(p["aten"])), "ck": sorted(set(p["ck"])), "torch": p["torch"],
                        "dequantize": p["dequantize"]}, sort_keys=True)
        sig[s] = sig.get(s, 0) + 1
    out["paths"] = [{"layers": n, **json.loads(s)} for s, n in sorted(sig.items(), key=lambda t: -t[1])]
    out["per_layer"] = [{k: v for k, v in r.items() if k != "proof"} for r in S.checks]
    return out


# ---------------------------------------------------------------- numz's loader

def load_ck_layers(model, state, orig, a, k):
    """_load_standard_weights with comfy-kitchen layers: the marked layers out of the state."""
    import torch

    markers = sorted(key[: -len(MARK)] for key in state if key.endswith(MARK))
    fmts = {L: json.loads(bytes(state[L + MARK].cpu().tolist()).decode()) for L in markers}
    fmt = file_format(fmts, "the DiT's state")
    if S.dit is None or S.dit.get("format") != fmt:
        abort(f"the DiT's state carries {fmt} markers, preflight found {S.dit and S.dit.get('format')}")
    if S.override_dtype is not None:
        abort(f"numz converted the DiT's state to {S.override_dtype} before its loader: fp8 values and scales cast")
    if S.mode not in FORMATS[fmt][1]:
        abort(f"format {fmt}: mode {S.mode!r}")
    expected = set(model.state_dict().keys())
    layers = []
    for L in markers:
        m = model.get_submodule(L)
        if type(m) is not torch.nn.Linear:
            abort(f"{L}: marked {fmt} but numz's module is {type(m).__name__}, not a Linear")
        t = {s: state.pop(f"{L}.{s}") for s in SUFFIXES if f"{L}.{s}" in state}
        bias = state.pop(f"{L}.bias", None)
        extra = [key for key in state if key.startswith(L + ".")]
        if extra:
            abort(f"{L}: tensors this patch doesn't know: {extra}")
        n, kk = m.out_features, m.in_features
        want = {
            "float8_e4m3fn": {"weight": (torch.float8_e4m3fn, (n, kk)), "weight_scale": (torch.float32, ())},
            "int8_tensorwise": {"weight": (torch.int8, (n, kk)), "weight_scale": (torch.float32, (n, 1))},
            "nvfp4": {"weight": (torch.uint8, (n, kk // 2)),
                      "weight_scale": (torch.float8_e4m3fn, (-(-n // 128) * 128, -(-(kk // 16) // 4) * 4)),
                      "weight_scale_2": (torch.float32, ())},
        }[fmt]
        got = {s: (v.dtype, tuple(v.shape)) for s, v in t.items() if s != "comfy_quant"}
        if got != want:
            abort(f"{L}: {got}, expected {want}")
        if (bias is None) != (m.bias is None) or (bias is not None and tuple(bias.shape) != (n,)):
            abort(f"{L}: bias {None if bias is None else tuple(bias.shape)}, numz's Linear "
                  f"{None if m.bias is None else tuple(m.bias.shape)}")
        if fmt == "int8_tensorwise" and kk % fmts[L]["convrot_groupsize"]:
            abort(f"{L}: {kk} columns, rotation groups of {fmts[L]['convrot_groupsize']}")
        layers.append((L, m, t, bias))
    replaced = {f"{L}.{s}" for L, *_ in layers for s in ("weight", "bias")}
    missing = sorted(expected - replaced - set(state))
    unexpected = sorted(set(state) - expected)
    if missing or unexpected:
        abort(f"the state against numz's model: {len(missing)} missing ({missing[:4]}), "
              f"{len(unexpected)} unexpected ({unexpected[:4]})")
    if S.check:
        compare_fp16(state, layers)
    kept = len(state)
    model = orig(model, state, *a, **k)
    cls = ck_linear_class()
    for L, m, t, bias in layers:
        parent, _, child = L.rpartition(".")
        setattr(model.get_submodule(parent), child,
                cls(L, fmts[L], S.mode, t, bias, m.in_features, m.out_features))
    if S.expect is not None and int(S.expect) != len(layers):
        abort(f"{len(layers)} layers replaced, CK_PATCH_EXPECT={S.expect}")
    first = next(model.named_parameters())
    S.report["load"] = {"layers": len(layers), "kept_keys": kept, "format": fmt, "mode": S.mode,
                        "first_parameter": [first[0], str(first[1].dtype)],
                        "device": str(layers[0][2]["weight"].device) if layers else None}
    if first[1].dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        abort(f"the model's first parameter {first[0]} is {first[1].dtype}: numz would take the model for fp8")
    log(f"DiT {S.paths.get('DiT')}: {len(layers)} layers -> CKLinear ({fmt}, mode {S.mode}), {kept} kept keys "
        f"to numz's loader, missing 0, unexpected 0; first parameter {first[0]} {first[1].dtype}; comfy-kitchen "
        f"{S.ck['version']}, backends {','.join(S.ck['available'])}")
    return model


def compare_fp16(state, layers) -> None:
    """Check mode: the tensors numz's loader receives, and the kept biases, are the fp16 file's."""
    import torch

    diff = []
    names = list(state) + [f"{L}.bias" for L, _, _, b in layers if b is not None]
    for key in names:
        mine = state[key] if key in state else next(b for L, _, _, b in layers if f"{L}.bias" == key)
        ref = fp16_tensor(key).to(mine.device)
        if mine.dtype != ref.dtype or mine.shape != ref.shape or not torch.equal(mine, ref):
            diff.append(key)
    S.report["fp16_compare"] = {"tensors": len(names), "different": diff[:20], "n_different": len(diff)}
    if diff:
        abort(f"CK_PATCH_FP16={S.fp16}: {len(diff)} of {len(names)} tensors differ from the DiT file's "
              f"({diff[:4]}): not this file's fp16 reference")
    log(f"check: the {len(names)} tensors kept in 16 bits equal {S.fp16}'s, bit for bit")


def patch_loader(mod) -> None:
    if not hasattr(mod, "_load_standard_weights") or not hasattr(mod, "_load_model_weights"):
        return
    orig_std, orig_lmw, orig_pms = mod._load_standard_weights, mod._load_model_weights, mod.prepare_model_structure

    def _load_standard_weights(model, state, *a, **k):
        if not any(key.endswith(MARK) for key in state):
            return orig_std(model, state, *a, **k)
        return load_ck_layers(model, state, orig_std, a, k)

    def _load_model_weights(model, checkpoint_path, target_device, used_meta, model_type, *a, **k):
        S.paths[model_type] = checkpoint_path
        S.report.setdefault("loaded", {})[model_type] = {
            "path": checkpoint_path, "realpath": os.path.realpath(checkpoint_path), "device": str(target_device)}
        log(f"{model_type} weights: {checkpoint_path} (-> {os.path.realpath(checkpoint_path)}) on {target_device}")
        S.override_dtype = (a[2] if len(a) > 2 else k.get("override_dtype")) if model_type == "DiT" else None
        if model_type == "DiT" and S.dit is not None:
            want = os.path.abspath(S.dit["path"])
            if os.path.abspath(checkpoint_path) != want:
                abort(f"numz loads the DiT from {checkpoint_path}, not {want}")
        return orig_lmw(model, checkpoint_path, target_device, used_meta, model_type, *a, **k)

    def prepare_model_structure(runner, model_type, checkpoint_path, config, *a, **k):
        r = orig_pms(runner, model_type, checkpoint_path, config, *a, **k)
        info = {"path": checkpoint_path, "realpath": os.path.realpath(checkpoint_path)}
        if model_type == "dit":
            try:
                info["config"] = str(config.dit.model.__object__.path)
            except Exception:  # noqa: BLE001 (a log line only)
                pass
        S.report.setdefault("structures", {})[model_type] = info
        log(f"{model_type} structure: {checkpoint_path} -> {info['realpath']}"
            + (f", model {info['config']}" if "config" in info else ""))
        if S.dryrun and model_type == "vae":
            log("dry run: stopping before any weight is read")
            raise SystemExit(0)
        return r

    mod._load_standard_weights = _load_standard_weights
    mod._load_model_weights = _load_model_weights
    mod.prepare_model_structure = prepare_model_structure
    S.override_dtype = None
    S.patched.append(f"{mod.__name__}: _load_standard_weights, _load_model_weights, prepare_model_structure")


def patch_constants(mod) -> None:
    if not hasattr(mod, "get_all_model_files") or not hasattr(mod, "is_supported_model_file"):
        return
    orig = mod.get_all_model_files

    def get_all_model_files():
        files = orig()
        reg = registry_names()
        if S.model_dir and reg is not None and os.path.isdir(S.model_dir):
            for f in sorted(os.listdir(S.model_dir)):
                if mod.is_supported_model_file(f) and f not in reg:
                    files[f] = os.path.join(S.model_dir, f)
        return files

    get_all_model_files.ck_patch_orig = orig
    mod.get_all_model_files = get_all_model_files
    S.patched.append(f"{mod.__name__}.get_all_model_files: --model_dir's files")


HOOKS = ((".src.utils.constants", patch_constants), (".src.core.model_loader", patch_loader))


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


def at_exit() -> None:
    S.report.update({
        "patched": S.patched,
        "compute_dtypes": S.cdt_seen,
        "refused_dtype_casts": S.refused_casts,
        "x_chunked": S.x_chunked,
        "aborted": S.aborted,
        "ended": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    if S.check:
        S.report["check"] = c = check_summary()
        if c["layers"]:
            a = c["per_kind"]["all"]
            log(f"check: {c['layers']} layers, path proven on {c['proof_ok']}, finite {c['finite']}; "
                + "; ".join(f"{m} median {a[m]['median']:.4%} worst {a[m]['worst']:.4%}"
                            for m in ("kernel", "total", "weights", "floor")))
    if S.cdt_seen:
        log(f"CKLinear calls by compute dtype: {S.cdt_seen}; dtype casts refused: {S.refused_casts}")
    if S.x_chunked["calls"]:
        log(f"activations quantized in row chunks (comfy-kitchen's 32-bit kernels): {S.x_chunked['calls']} "
            f"calls, {S.x_chunked['chunks']} chunks, largest input {S.x_chunked['largest_elements']} elements")
    if S.log_path:
        os.makedirs(os.path.dirname(os.path.abspath(S.log_path)), exist_ok=True)
        with open(S.log_path, "w") as f:
            json.dump(S.report, f, indent=1)
        log(f"report: {S.log_path}")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        sys.exit(__doc__)
    if S.mode is not None and not any(S.mode in m for _, m in FORMATS.values()):
        sys.exit(f"ck_patch: CK_PATCH_MODE={S.mode}: one of w8a16, w8a8, int8, w4a4, w4a16")
    script = os.path.abspath(sys.argv[1])
    sys.argv = [script, *sys.argv[2:]]
    sys.path[0] = os.path.dirname(script)
    preparse_model_dir(sys.argv[1:])
    install()
    atexit.register(at_exit)
    runpy.run_path(script, run_name="__main__")
    if S.args is None:
        log("numz's arguments were never parsed: nothing checked")


if __name__ == "__main__":
    main()

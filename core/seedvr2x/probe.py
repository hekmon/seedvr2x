"""The environment probe (DESIGN.md, Repository and apps, "ComfyUI's process is not seedvr2x's"):
before the first run in an environment, each kernel the chosen DiT file needs is run on the device
on a small seeded input and held to the CPU's result, so that an environment whose torch, CUDA or
driver gives the wrong numbers is refused, saying why, instead of producing pictures that are
wrong. No list of figures made ahead holds: ComfyUI's Desktop app offers torch 2.9.1 to 2.13.0
on CUDA 12.6, 12.8 or 13.0.

What is checked (`kernels`), by what the file needs:
- always, a bfloat16 matrix product (model.COMPUTE_DTYPE) and the attention, PyTorch's SDPA, the
  one kernel, through the vendored FlashAttentionVarlen as the DiT calls it;
- an int8 file (weights.INT8): int8.py's W8A8 linear, in three steps, the integer product, the
  codes and the output;
- a GGUF file (weights.GGUF): kquant.py's dequantisation of each type the file holds (Q3_K,
  Q4_K, Q5_K), to float32 and to bfloat16.
Each input is a few MiB at most and seeded, so that a check takes seconds and gives the same
numbers every time on the same environment. Each result holds a measured error and the tolerance
it is held to (`Result`); where the tolerance comes from is said at each check. A kernel that
raises is a failed check too, its message kept, except an out-of-memory error, which says nothing
of the kernel and is not cached.

The cache (`cache_path`): one small JSON file in the platform's user cache directory, holding for
each environment seen, by the digest of its fingerprint, the fingerprint's parts and each check's
result, and the place for the allocator's margin. A fingerprint is the parts of `fingerprint`:
seedvr2x's version and the library's code, torch and its CUDA, the NVIDIA driver, the GPU and the
allocator's setting. When a part changes the digest is another and
the checks are run again; a part that can't be read is recorded as "unknown" and logged, never
guessed. A check the file lacks (another DiT layout) is run and added;
a failure is cached too, so that a second run refuses at once, saying it was measured before and
when. A file unreadable or of another format is ignored, logged, and written again.

The allocator's margin: DESIGN.md has the probe measure it "on a few small steps", which waits for
design's rule (after a study of the allocator). The measurement is not built: each environment's
record has a field `allocator_margin`, null today, and `Probed.allocator_margin` is None. Its
reading side is built: a margin there, a share of a step's P, is carried by the app into the
plan's reading (memory.Reading.margin) and peaks.reserve counts the larger of it and the rule's
share for that run (SEGMENTS and POOL), the nodes doing it (apps/comfyui/models.py). Whoever builds
the measurement fills that field through `check`, bumps FORMAT only if the record changes shape
(a value in a field that is null today needs no new format), and decides whether a margin
measured is part of "every needed check passed" for the cache to vouch.

Importing this module imports no torch (the apps' neighbours, model.py and the kernels' modules,
do): torch and the kernels are imported inside the checks and the fingerprint's readings.
"""

import contextlib
import ctypes
import functools
import hashlib
import importlib.metadata
import itertools
import json
import logging
import os
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from seedvr2x import memory, waiting, weights
from seedvr2x.errors import Refusal

if TYPE_CHECKING:
    import torch
    from torch import Tensor

logger = logging.getLogger(__name__)

# The cache file's format: a file of another one is ignored and written again.
FORMAT = 1
NAME = "probe.json"
# The environments the file keeps: one per fingerprint seen (a user with two environments on
# one machine, ComfyUI's and seedvr2x's own, keeps both), the oldest dropped past this many.
ENVIRONMENTS = 8
UNKNOWN = "unknown"
# The kind of the refusal (errors.Refusal.kind).
KIND = "probe"

# The NVML libraries a platform loads, for the driver's version. torch reads the device count
# through the first of them on Linux (torch/cuda/__init__.py, _raw_device_count_nvml); memory.NVML
# is that name, the tests' to replace. On Windows the driver installs nvml.dll in System32, and
# older ones in NVSMI's directory under Program Files (memory.NVML_WINDOWS, which reads the card
# through them too): read so by the nodes' first Windows runs (driver 621.01, 2026-11-07). A library
# not found is "unknown", logged.
NVML_WINDOWS = memory.NVML_WINDOWS

# --- the tolerances, and where they come from ---------------------------------------------------
# PROVISIONAL (impl-nodes lane 05, 2026-11-05; DESIGN.md gives none, no test sets one for these
# two): bfloat16 keeps 8 significant bits, so a result rounded to it is within half a spacing
# (2^-8 of its leading bit's value); a float32 accumulation of a few hundred terms adds under
# 2^-20 of the sum of the terms' magnitudes (the int8 tests' own figure, tests/test_int8.py,
# `explained`). A matrix product is held to 2 spacings of the larger result plus 2^-12 of the sum
# of the terms' magnitudes. What a broken kernel does (garbage, zeros, NaN, a transposed layout)
# is orders over that; a kernel splitting its sum across blocks and reducing in bfloat16 (torch's
# default, allow_bf16_reduced_precision_reduction) could exceed it at a large K, which is
# why the input is small, and the GPU test prints each ratio to refine it.
MATMUL_SPACINGS = 2.0
MATMUL_TERMS = 2.0**-12
# Attention: the largest difference from a float64 softmax over the largest output value. Outputs
# are rounded to bfloat16 (2^-9 of a value), the probabilities enter the second product rounded to
# it, and the sum is float32: a few spacings of the largest value at most; 2^-6 is 8 half
# spacings. PROVISIONAL, as above.
ATTENTION_RELATIVE = 2.0**-6
# The int8 linear (tests/test_int8.py): the integer product exact; the codes the CPU's for all but
# 1 in 100,000 and two at the least (codes_agree: the rotation's float32 sums rounding otherwise,
# int8.py's docstring); the output within the codes' and scales' own difference, one bfloat16
# spacing and 2^-20 of the terms (explained, whose "same bits" clause is not held here: it needs
# the rows the codes agree on, and the cost is a second run).
CODES_SHARE = 100_000
CODES_LEAST = 2
INT8_TERMS = 2.0**-20


@dataclass(frozen=True)
class Result:
    """One check's result: the kernel's name, the error measured in `unit`, the tolerance it is
    held to (the same unit), whether it passed, when it was measured (UTC, ISO 8601, seconds),
    and, for a kernel that raised, the message (error None)."""

    kernel: str
    error: float | None
    tolerance: float
    unit: str
    passed: bool
    at: str
    detail: str = ""

    def said(self) -> str:
        """What was measured, in words."""
        if self.error is None:
            return f"{self.kernel} did not run: {self.detail}"
        return f"{self.kernel}: {self.error:.4g} {self.unit}, {self.tolerance:.4g} allowed"


@dataclass(frozen=True)
class Probed:
    """What `check` found, all passed: the results of the kernels the file needs, in the order
    `kernels` gives them; the fingerprint's parts and their digest; whether the cache vouched
    (cached) or the checks ran; the cache file; and the allocator's margin, None until it is
    built (the module's docstring)."""

    results: tuple[Result, ...]
    fingerprint: dict[str, str]
    digest: str
    cached: bool
    path: Path
    allocator_margin: float | None = None


class ProbeRefusal(Refusal):
    """An environment whose kernel gives the wrong numbers, or doesn't run: kind "probe". Carries
    the failed `result` (kernel, error measured, tolerance), the environment's `fingerprint`
    parts, whether the failure came from the cache (`cached`: measured before, at result.at) and
    the cache's `path`. The message names no option of an application; `way` is the file to delete
    to measure again, which an application may word its own way."""

    def __init__(
        self, result: Result, fingerprint: dict[str, str], path: Path, *, cached: bool
    ) -> None:
        self.result = result
        self.fingerprint = fingerprint
        self.cached = cached
        self.path = path
        self.kernel = result.kernel
        self.error = result.error
        self.tolerance = result.tolerance
        environment = ", ".join(f"{name} {value}" for name, value in fingerprint.items())
        reason = (
            f"this environment's {result.kernel} gives the wrong numbers on {fingerprint['gpu']}"
            if result.error is not None
            else f"this environment's {result.kernel} does not run on {fingerprint['gpu']}"
        )
        reason += f" ({result.said()}), so that a picture made here would be wrong"
        way = (
            f"the result is kept in {path}; delete that file to measure again, after a change to"
            " the environment (torch, its CUDA, the driver)"
        )
        when = f" It was measured before, at {result.at} UTC." if cached else ""
        message = f"{reason}.{when} The environment: {environment}. {way[0].upper()}{way[1:]}."
        super().__init__(message, kind=KIND, reason=reason, way=way)


# --- the fingerprint ----------------------------------------------------------------------------


@functools.cache
def library_code_sha256() -> str:
    """The SHA-256 of the library's files, the library's half of the CLI's job code hash
    (apps/seedvr2x/manifest.code_sha256, which takes the same files under the same keys,
    `seedvr2x/<path inside the package>`, the sorted paths and bytes, and adds the CLI's own
    files): a changed line of the library is another environment, as it is another job. About
    100 files and 3 MB, a few milliseconds; made once per process."""
    package = Path(__file__).resolve().parent
    files = {
        f"seedvr2x/{path.relative_to(package).as_posix()}": path
        for path in package.rglob("*")
        if path.is_file()
    }
    digest = hashlib.sha256()
    for key in sorted(files):
        if "__pycache__" in key.split("/"):
            continue
        digest.update(key.encode() + b"\0" + files[key].read_bytes())
    return digest.hexdigest()


def driver_version() -> str | None:
    """The NVIDIA driver's version, as nvidia-smi says it, read from NVML, the driver's own
    library (Linux's libnvidia-ml.so.1, Windows' nvml.dll); None, logged, where NVML can't say,
    and silently on macOS, which has no NVIDIA driver. torch has no call for it. The same reading
    as model.nvidia_driver, which is Linux's alone and imports the whole model layer."""
    if sys.platform == "darwin":
        return None
    names = NVML_WINDOWS if sys.platform == "win32" else (memory.NVML,)
    problem = "no NVML library found"
    for name in names:
        try:
            nvml = ctypes.CDLL(name)
        except OSError as error:
            problem = str(error)
            continue
        status = nvml.nvmlInit_v2()
        if status != 0:
            problem = f"NVML's initialisation: {status}"
            continue
        try:
            version = ctypes.create_string_buffer(80)  # NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE
            status = nvml.nvmlSystemGetDriverVersion(version, ctypes.c_uint(len(version)))
            if status == 0:
                return version.value.decode()
            problem = f"NVML's answer: {status}"
        finally:
            nvml.nvmlShutdown()
    logger.warning("the NVIDIA driver's version is unknown: %s", problem)
    return None


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def fingerprint(device: "torch.device") -> dict[str, str]:
    """The environment's parts, each text, "unknown" where a part can't be read (logged):
    - seedvr2x: the installed seedvr2x-core's version; code: the library's SHA-256
      (library_code_sha256), since a version does not change with the code;
    - torch: torch.__version__; cuda: torch.version.cuda ("none" for a build without);
    - driver: the NVIDIA driver (driver_version); on a device that is not CUDA, "none";
    - gpu: the device's name (torch.cuda.get_device_name), the device type for another;
      capability: its compute capability "major.minor", "none" for another device;
    - allocator: torch.cuda.get_allocator_backend() ("native" or "cudaMallocAsync"), "none"
      without CUDA; alloc_conf: PYTORCH_CUDA_ALLOC_CONF and PYTORCH_ALLOC_CONF as set
      (memory.VARIABLES, the two torch reads the allocator's setting from), "unset" for neither.
    No part names an attention kernel: PyTorch's SDPA is the one (FlashAttention 2's version was a
    part until it went, 2026-11-08), which torch's version says."""
    import torch

    parts: dict[str, str] = {}
    version = _version("seedvr2x-core")
    if version is None:
        logger.warning("seedvr2x-core's version is unknown: not an installed distribution")
    parts["seedvr2x"] = version or UNKNOWN
    parts["code"] = library_code_sha256()
    parts["torch"] = torch.__version__
    parts["cuda"] = torch.version.cuda or "none"
    if device.type == "cuda":
        parts["driver"] = driver_version() or UNKNOWN
        parts["gpu"] = torch.cuda.get_device_name(device)
        major, minor = torch.cuda.get_device_capability(device)
        parts["capability"] = f"{major}.{minor}"
        parts["allocator"] = torch.cuda.get_allocator_backend()
    else:
        parts["driver"] = parts["capability"] = parts["allocator"] = "none"
        parts["gpu"] = device.type
    set_up = [f"{name}={os.environ[name]}" for name in memory.VARIABLES if name in os.environ]
    parts["alloc_conf"] = ";".join(set_up) or "unset"
    for name, value in parts.items():
        if value == UNKNOWN:
            logger.warning("the environment's %s is unknown: recorded as such", name)
    return parts


def digest(parts: dict[str, str]) -> str:
    """The SHA-256 of a fingerprint's parts, their names sorted, as JSON."""
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


# --- the cache ----------------------------------------------------------------------------------


def cache_directory() -> Path:
    """seedvr2x's directory in the platform's user cache directory: Linux $XDG_CACHE_HOME (an
    absolute path, as the XDG specification has it) or ~/.cache, macOS ~/Library/Caches, Windows
    %LOCALAPPDATA% or ~/AppData/Local. No new dependency: the three paths are the platforms'
    conventions."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        given = os.environ.get("XDG_CACHE_HOME", "")
        base = Path(given) if given and Path(given).is_absolute() else Path.home() / ".cache"
    return base / "seedvr2x"


def cache_path() -> Path:
    """The cache file: probe.json in cache_directory()."""
    return cache_directory() / NAME


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").removesuffix("+00:00") + "Z"


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _result(kernel: str, raw: object) -> Result | None:
    """A result as the file holds it, None where it is not one (a field missing or of the wrong
    type)."""
    if not isinstance(raw, dict):
        return None
    fields = cast("dict[str, Any]", raw)
    error, tolerance, unit = fields.get("error"), fields.get("tolerance"), fields.get("unit")
    passed, at, detail = fields.get("passed"), fields.get("at"), fields.get("detail", "")
    if (
        not (error is None or _number(error))
        or tolerance is None
        or not _number(tolerance)
        or not isinstance(unit, str)
        or not isinstance(passed, bool)
        or not isinstance(at, str)
        or not isinstance(detail, str)
    ):
        return None
    return Result(
        kernel, None if error is None else float(error), float(tolerance), unit, passed, at, detail
    )


def _entry(raw: object) -> dict[str, Any] | None:
    """An environment as the file holds it, its checks parsed (Result by kernel); None where it
    is not as written."""
    if not isinstance(raw, dict):
        return None
    fields = cast("dict[str, Any]", raw)
    parts, stored = fields.get("fingerprint"), fields.get("checks")
    if not isinstance(parts, dict) or not isinstance(stored, dict):
        return None
    checks: dict[str, Result] = {}
    for kernel, each in cast("dict[str, Any]", stored).items():
        result = _result(kernel, each)
        if result is None:
            return None
        checks[kernel] = result
    return {
        "fingerprint": cast("dict[str, str]", parts),
        "checks": checks,
        "allocator_margin": fields.get("allocator_margin"),
    }


def _environments(path: Path) -> dict[str, dict[str, Any]]:
    """The cache's environments by digest, each {"fingerprint", "checks", "allocator_margin"}.
    A missing file is no environment; a file that can't be read, isn't JSON of this FORMAT, or an
    entry that is not as written is ignored, logged, and written over on the next save."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeDecodeError) as error:
        logger.warning("the probe's cache %s is unreadable, ignored: %s", path, error)
        return {}
    try:
        document: object = json.loads(text)
    except ValueError as error:
        logger.warning("the probe's cache %s is not JSON, ignored: %s", path, error)
        return {}
    if not isinstance(document, dict) or cast("dict[str, Any]", document).get("format") != FORMAT:
        logger.warning("the probe's cache %s is of another format, ignored", path)
        return {}
    stored = cast("dict[str, Any]", document).get("environments")
    if not isinstance(stored, dict):
        logger.warning("the probe's cache %s holds no environments, ignored", path)
        return {}
    found: dict[str, dict[str, Any]] = {}
    for key, raw in cast("dict[str, Any]", stored).items():
        entry = _entry(raw)
        if entry is None:
            logger.warning("an environment of the probe's cache %s is damaged, ignored", path)
            continue
        found[key] = entry
    return found


def _save(path: Path, environments: dict[str, dict[str, Any]]) -> None:
    """Write the cache: a temporary file in its directory, then replaced over it, so that a
    reader sees the old file or the new, never half of one. The oldest environments beyond
    ENVIRONMENTS are dropped. A directory that can't be written is logged, not fatal: the checks
    ran, and the next run measures again."""
    while len(environments) > ENVIRONMENTS:
        del environments[next(iter(environments))]
    document = {
        "format": FORMAT,
        "environments": {
            key: {
                "fingerprint": entry["fingerprint"],
                "checks": {
                    kernel: {
                        "error": result.error,
                        "tolerance": result.tolerance,
                        "unit": result.unit,
                        "passed": result.passed,
                        "at": result.at,
                        "detail": result.detail,
                    }
                    for kernel, result in entry["checks"].items()
                },
                "allocator_margin": entry["allocator_margin"],
            }
            for key, entry in environments.items()
        },
    }
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(document, handle, indent=2)
        os.replace(temporary, path)
    except OSError as error:
        logger.warning("the probe's cache %s could not be written: %s", path, error)
        if temporary is not None:
            with contextlib.suppress(OSError):
                temporary.unlink()


# --- the checks ---------------------------------------------------------------------------------

MATMUL = "matmul bfloat16"
INT8_PRODUCT, INT8_CODES, INT8_LINEAR = "int8 product", "int8 codes", "int8 linear"


# The attention's check, by the name its results are cached under: PyTorch's SDPA, the one kernel.
ATTENTION = "attention sdpa"


def dequantisation(qtype: str) -> str:
    return f"dequantise {qtype}"


def kernels(architecture: weights.Architecture) -> tuple[str, ...]:
    """The kernels the DiT file of this architecture needs, by name, in the order they are
    checked: the matrix product and the attention always; an int8 file's linear in three
    steps; a GGUF file's dequantisation of each type it holds."""
    names = [MATMUL, ATTENTION]
    if architecture.layout == weights.INT8:
        names += [INT8_PRODUCT, INT8_CODES, INT8_LINEAR]
    elif architecture.layout == weights.GGUF:
        names += [dequantisation(qtype) for qtype in architecture.matrix_types()]
    return tuple(names)


def _spacing(values: "Tensor") -> "Tensor":
    """bfloat16's spacing at each value's magnitude, float64: 2^(e - 7) for a magnitude in
    [2^e, 2^(e + 1)), the smallest normal's spacing below it (tests/test_int8.py, bf16_ulp)."""
    import torch

    magnitude = values.double().abs().clamp(min=torch.finfo(torch.bfloat16).tiny)
    return torch.exp2(torch.floor(torch.log2(magnitude)) - 7)


def _made(kernel: str, error: float, tolerance: float, unit: str) -> Result:
    # A NaN error is a failure: every comparison with it is false.
    return Result(kernel, error, tolerance, unit, error <= tolerance, _now())


def _matmul(device: "torch.device") -> list[Result]:
    """A (128, 512) times a (512, 256) bfloat16 product on the device, against the float64
    product of the same bfloat16 values on the CPU: within MATMUL_SPACINGS bfloat16 spacings of
    the larger result and MATMUL_TERMS of the sum of the terms' magnitudes, elementwise; the error
    is the largest ratio to that bound, held to 1. PROVISIONAL (above)."""
    import torch

    generator = torch.Generator().manual_seed(11)
    a = torch.randn(128, 512, generator=generator).to(torch.bfloat16)
    b = torch.randn(512, 256, generator=generator).to(torch.bfloat16)
    with torch.no_grad():
        out = (a.to(device) @ b.to(device)).cpu()
    expected = a.double() @ b.double()
    terms = a.double().abs() @ b.double().abs()
    allowed = (
        MATMUL_SPACINGS * _spacing(torch.maximum(out.double().abs(), expected.abs()))
        + MATMUL_TERMS * terms
    )
    ratio = float(((out.double() - expected).abs() / allowed).max())
    return [_made(MATMUL, ratio, 1.0, "times the bound allowed")]


def _attention(device: "torch.device") -> list[Result]:
    """The vendored FlashAttentionVarlen, PyTorch's SDPA, as the DiT calls it
    (vendor/models/dit_7b/attention.py: q, k, v (tokens, heads, head_dim) bfloat16, the cumulative
    lengths int32 on the device), two sequences of 96 and 64 tokens, 4 heads of 128 (the 7B's
    head_dim, vendor/configs_7b/main.yaml), against a float64 softmax attention of each sequence
    alone on the CPU. The error is the largest difference over the largest value of the
    reference, held to ATTENTION_RELATIVE. PROVISIONAL (above)."""
    import torch

    from seedvr2x.model import COMPUTE_DTYPE
    from seedvr2x.vendor.models.dit_7b.attention import FlashAttentionVarlen

    lengths, heads, head_dim = (96, 64), 4, 128
    tokens = sum(lengths)
    generator = torch.Generator().manual_seed(12)
    q, k, v = (
        torch.randn(tokens, heads, head_dim, generator=generator).to(COMPUTE_DTYPE)
        for _ in range(3)
    )
    starts = [0, lengths[0], tokens]
    cumulative = torch.tensor(starts, dtype=torch.int32, device=device)
    module = FlashAttentionVarlen(compute_dtype=COMPUTE_DTYPE)
    with torch.no_grad():
        out = module(
            q.to(device),
            k.to(device),
            v.to(device),
            cu_seqlens_q=cumulative,
            cu_seqlens_k=cumulative,
            max_seqlen_q=max(lengths),
            max_seqlen_k=max(lengths),
        ).cpu()
    expected = torch.empty(tokens, heads, head_dim, dtype=torch.float64)
    for begin, end in itertools.pairwise(starts):
        # (heads, sequence, head_dim) each; no mask, no causality: the DiT's attention is neither.
        sq, sk, sv = (x[begin:end].double().permute(1, 0, 2) for x in (q, k, v))
        weights_ = torch.softmax(sq @ sk.transpose(1, 2) / head_dim**0.5, dim=-1)
        expected[begin:end] = (weights_ @ sv).permute(1, 0, 2)
    error = float((out.double() - expected).abs().max() / expected.abs().max())
    return [_made(ATTENTION, error, ATTENTION_RELATIVE, "of the largest value")]


def _int8_activations(rows: int, inputs: int, outputs: int, seed: int) -> tuple["Tensor", ...]:
    """A layer of this shape and its bfloat16 input, as tests/test_int8.py's `activations` makes
    them: normal inputs of standard deviation 3, four channels 50 times that, weights over the
    whole int8 range, a scale a row, a float16 bias."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, inputs, generator=generator) * 3
    x[:, :4] *= 50
    weight = torch.randint(-128, 128, (outputs, inputs), generator=generator, dtype=torch.int8)
    scale = torch.rand(outputs, 1, generator=generator) * 0.01 + 1e-4
    bias = (torch.randn(outputs, generator=generator) * 0.5).to(torch.float16)
    return x.to(torch.bfloat16), weight, scale, bias


def _int8(device: "torch.device") -> list[Result]:
    """int8.py's W8A8 linear on the device against the CPU's, with 8 rows (torch's int8 multiply
    pads a first matrix of 16 rows or fewer on CUDA, int8.product) and 300, through a layer of
    1,536 inputs and outputs (2.25 MiB of weights; the 7B's matrices are 3,072 to 12,288), in
    three steps:
    - INT8_PRODUCT: the int32 product of the device's codes and the weight, against the exact
      product in float64 (every sum is under 2^53); elements apart, 0 allowed
      (tests/test_int8.py, test_the_w8a8_linear_on_cuda_is_the_cpu_s asserts it equal);
    - INT8_CODES: the device's codes against the CPU's: apart by at most 1 in CODES_SHARE of
      them, CODES_LEAST at the least (that test's codes_agree), the cases' counts summed;
    - INT8_LINEAR: the outputs apart by no more than the codes and the scales that differ make
      them (that test's `explained` bound: the float64 rescalings' difference, one bfloat16
      spacing of the larger and 2^-20 of the terms), the largest ratio to it, held to 1."""
    import torch

    from seedvr2x import int8

    dtype = torch.bfloat16
    product_apart = codes_apart = codes_allowed = 0
    ratio = 0.0
    for rows in (8, 300):
        x, weight, scale, bias = _int8_activations(rows, 1536, 1536, seed=13 + rows)
        on = (x.to(device), weight.to(device), scale.to(device), bias.to(device))
        with torch.no_grad():
            codes, scales = int8.quantize(int8.rotate(on[0].float()), dtype)
            product = int8.product(codes, int8.transposed(on[1])).cpu()
            exact = codes.cpu().double() @ weight.double().T
            product_apart += int((product.double() != exact).sum())
            cpu_codes, cpu_scales = int8.quantize(int8.rotate(x.float()), dtype)
            codes_apart += int((codes.cpu() != cpu_codes).sum())
            codes_allowed += max(CODES_LEAST, cpu_codes.numel() // CODES_SHARE)
            out = int8.linear(*on, dtype).cpu()
            expected = int8.linear(x, weight, scale, bias, dtype)
        row_scale = scale.reshape(1, -1).double()
        sides = [(codes.cpu(), scales.cpu()), (cpu_codes, cpu_scales)]
        rescaled = [(c.double() @ weight.double().T) * s.double() * row_scale for c, s in sides]
        larger = torch.maximum(out.double().abs(), expected.double().abs())
        terms = torch.maximum(rescaled[0].abs(), rescaled[1].abs()) + larger
        allowed = (rescaled[0] - rescaled[1]).abs() + _spacing(larger) + terms * INT8_TERMS
        ratio = max(ratio, float(((out.double() - expected.double()).abs() / allowed).max()))
    return [
        _made(INT8_PRODUCT, float(product_apart), 0.0, "elements apart"),
        _made(INT8_CODES, float(codes_apart), float(codes_allowed), "codes apart"),
        _made(INT8_LINEAR, ratio, 1.0, "times the bound allowed"),
    ]


# Where each type's fp16 fields sit in a block (ggml-common.h; tests/test_kquant.py, HALVES): the
# scale d, and for Q4_K and Q5_K the minimum's dmin.
_HALVES = {"Q3_K": (108,), "Q4_K": (0, 2), "Q5_K": (0, 2)}


def _blocks(qtype: str, count: int, seed: int) -> "Tensor":
    """count blocks (count, bytes) uint8 of qtype: random bytes, the fp16 fields of each block
    one of several magnitudes with a random sign (tests/test_kquant.py's `blocks`: random fp16
    fields would be NaN or infinite one time in 32)."""
    import torch

    from seedvr2x import kquant

    magnitudes = torch.tensor(
        [0.0, 2.0**-24, 1023 * 2.0**-24, 2.0**-14, 3e-4, 0.0123, 0.5, 1.0, 7.3, 1000.0, 65504.0],
        dtype=torch.float16,
    )
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randint(
        0, 256, (count, kquant.BLOCK_BYTES[qtype]), generator=generator, dtype=torch.uint8
    )
    for at in _HALVES[qtype]:
        picked = magnitudes[torch.randint(0, len(magnitudes), (count,), generator=generator)]
        signs = torch.where(torch.rand(count, generator=generator) < 0.5, -1.0, 1.0)
        raw[:, at : at + 2] = (picked * signs.to(torch.float16)).view(torch.uint8).reshape(-1, 2)
    return raw


def _dequantise(device: "torch.device", qtype: str) -> list[Result]:
    """kquant.py's dequantisation of 2,048 synthetic blocks of qtype (0.5 MiB at the most) on the
    device against the CPU's, to float32 and to bfloat16 (the dtype the run dequantises to), bit
    for bit: values apart, 0 allowed (tests/test_kquant.py asserts the CUDA path equal to the
    CPU's bit for bit, and the CPU's to the gguf package's)."""
    import torch

    from seedvr2x import kquant

    count = 2048
    raw = _blocks(qtype, count, seed=14)
    apart = 0
    for dtype in (torch.float32, torch.bfloat16):
        out = kquant.dequantise(raw.to(device), qtype, (count, kquant.VALUES), dtype).cpu()
        expected = kquant.dequantise(raw, qtype, (count, kquant.VALUES), dtype)
        # Compared by bits: a NaN or a signed zero is a difference too.
        width = torch.int32 if dtype == torch.float32 else torch.int16
        apart += int((out.view(width) != expected.view(width)).sum())
    return [_made(dequantisation(qtype), float(apart), 0.0, "values apart")]


def _runners(
    device: "torch.device", architecture: weights.Architecture
) -> list[tuple[tuple[str, ...], Callable[[], list[Result]]]]:
    """The checks the file needs, each as the kernel names it makes results for and the call that
    makes them, in the order of `kernels`."""
    runners: list[tuple[tuple[str, ...], Callable[[], list[Result]]]] = [
        ((MATMUL,), lambda: _matmul(device)),
        ((ATTENTION,), lambda: _attention(device)),
    ]
    if architecture.layout == weights.INT8:
        runners.append(((INT8_PRODUCT, INT8_CODES, INT8_LINEAR), lambda: _int8(device)))
    elif architecture.layout == weights.GGUF:
        for qtype in architecture.matrix_types():
            runners.append(
                ((dequantisation(qtype),), lambda qtype=qtype: _dequantise(device, qtype))
            )
    return runners


def _run(names: tuple[str, ...], run: Callable[[], list[Result]]) -> list[Result]:
    """One check's results; a kernel that raises is a failed check with the message kept, but an
    out-of-memory error, which says nothing of the kernel, goes on up uncached, and so does a CUDA
    error other than out of memory (waiting.cuda_failed), which says the card failed: cached, it
    would refuse the environment for good, and the checks after it would run on that card."""
    import torch

    try:
        return run()
    except torch.OutOfMemoryError:
        raise
    except (RuntimeError, ValueError, AssertionError, ImportError) as error:
        if waiting.cuda_failed(error):
            raise
        logger.warning("the probe's check of %s raised: %s", ", ".join(names), error)
        return [
            Result(name, None, 0.0, "", False, _now(), f"{type(error).__name__}: {error}")
            for name in names
        ]


def check(
    device: "torch.device",
    architecture: weights.Architecture,
    *,
    path: Path | None = None,
) -> Probed:
    """Make sure that each kernel the DiT file of `architecture` needs (`kernels`) gives
    the right numbers on `device` in this environment, and say so. The cache (`path`, by default
    cache_path()) vouches where its record of this environment's fingerprint has every needed
    check passed; a check it lacks is run and added; a failure, cached or just measured, raises a
    ProbeRefusal. One INFO line says what was checked or that the cache vouched, with the file's
    path. For `device` on the library's own: the device the models run on."""
    started = time.monotonic()
    path = cache_path() if path is None else path
    parts = fingerprint(device)
    key = digest(parts)
    needed = kernels(architecture)
    environments = _environments(path)
    entry = environments.get(key)
    known: dict[str, Result] = {} if entry is None else entry["checks"]
    for name in needed:
        if name in known and not known[name].passed:
            logger.info("the probe's cache %s holds a failure of %s", path, name)
            raise ProbeRefusal(known[name], parts, path, cached=True)
    missing = [name for name in needed if name not in known]
    if not missing:
        measured = max(known[name].at for name in needed)
        logger.info(
            "the environment probe: the cache vouches for %s (measured at %s UTC): %s",
            ", ".join(needed),
            measured,
            path,
        )
        margin: float | None = entry["allocator_margin"] if entry is not None else None
        return Probed(tuple(known[name] for name in needed), parts, key, True, path, margin)
    fresh: dict[str, Result] = {}
    for names, run in _runners(device, architecture):
        if any(name in missing for name in names):
            fresh.update((result.kernel, result) for result in _run(names, run))
    # The environment goes last in the file's order: the oldest are the ones dropped.
    margin = entry["allocator_margin"] if entry is not None else None
    environments.pop(key, None)
    environments[key] = {
        "fingerprint": parts,
        "checks": {**known, **fresh},
        "allocator_margin": margin,
    }
    _save(path, environments)
    for name in needed:
        if name in fresh and not fresh[name].passed:
            logger.error("the environment probe: %s", fresh[name].said())
            raise ProbeRefusal(fresh[name], parts, path, cached=False)
    logger.info(
        "the environment probe: checked %s in %.1f s (%s on %s, torch %s, CUDA %s, driver %s),"
        " kept in %s",
        ", ".join(missing),
        time.monotonic() - started,
        digest(parts)[:8],
        parts["gpu"],
        parts["torch"],
        parts["cuda"],
        parts["driver"],
        path,
    )
    results = {**known, **fresh}
    return Probed(tuple(results[name] for name in needed), parts, key, False, path, margin)

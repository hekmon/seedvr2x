"""The environment a job's output depends on besides its settings and inputs, which a resume must
find again to stay bit-identical (DESIGN.md, Pause and resume): the stack, the GPU, ffmpeg and its
conversions, and the version of every distribution the run is made of, derived rather than
listed, so that a new dependency can't be missed."""

import functools
import importlib
import platform
import re
import sys
from collections.abc import Iterable, Mapping
from importlib import metadata
from pathlib import Path

import torch

# seedvr2x's own modules, which the run is made of: their files are hashed (settings.code), and
# the distributions they import are versioned (versions).
PACKAGE = Path(__file__).resolve().parents[1]

# A requirement's distribution and the extras it asks of it, and the extras a marker names
# (PEP 508): `cuda-toolkit[cublas,cudart]==13.0.3; platform_system == "Linux"`.
REQUIREMENT = re.compile(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[([^\]]*)\])?")
EXTRA = re.compile(r"""\bextra\s*==\s*["']([^"']+)["']""")
# A shared library (`libcublas.so.13`), and a Python extension module, which only an import loads
# (PEP 3149: `_speedups.cpython-313-x86_64-linux-gnu.so`, `.abi3.so`). One named with a bare
# `.so`, as triton's `libtriton.so`, passes for a library: that only errs toward recording.
SHARED_LIBRARY = re.compile(r"\.so(\.\d+)*$")
EXTENSION_MODULE = re.compile(r"\.(cpython-[^.]+|abi3)\.so$")
# Used when installed, though not declared: PyPI only has its source (AGENTS.md, Environment).
UNDECLARED = ("flash-attn",)
# setuptools' hook, which site imports at every start from distutils-precedence.pth, unless
# SETUPTOOLS_USE_DISTUTILS=stdlib, whatever the process runs: the run imports no setuptools, and
# a resume mustn't depend on that variable.
START_HOOKS = ("_distutils_hack",)
# Imported by a download alone, huggingface_hub's of a Xet file (1.33.0: file_download.py:554,
# utils/_xet.py:284), which leaves the output alone: what it fetches is checked against its pinned
# SHA-256 before it loads (runtime/pull.py). Recorded, it would set a job started with a download
# apart from its resume, the files then in the cache. huggingface_hub itself is the run's: the
# vendored model imports diffusers, which imports it. Provisional (implementation, 2026-10-09):
# DESIGN.md's environment doesn't say.
DOWNLOAD_ONLY = frozenset({"hf-xet"})


def current(device: torch.device, ffmpeg_version: str, conversions: str) -> dict[str, object]:
    """The environment of a run on device: torch, CUDA, cuDNN, the GPU, the attention backend and
    FlashAttention, ffmpeg and its conversions' fingerprint (media/fingerprint.py), Python and
    the versions of the distributions (versions); and the NVIDIA driver, for information only,
    since the math kernels ship with torch."""
    from seedvr2x.runtime import model

    try:
        flash_attn = metadata.version("flash_attn")
    except metadata.PackageNotFoundError:
        flash_attn = None
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(device),
        "driver": model.nvidia_driver(),
        "attention": model.attention_backend(),
        "flash_attn": flash_attn,
        "ffmpeg": ffmpeg_version,
        "conversions": conversions,
        "python": f"{platform.python_implementation()} {platform.python_version()}",
        "packages": versions(),
    }


def versions() -> dict[str, str]:
    """The version of every distribution the run is made of, by its normalised name (PEP 503):
    those of the run's it imports (imported) once every module of seedvr2x is, the vendored
    models' too, which the run imports only when it builds them; and the shared libraries torch
    requires, its runtime (cuBLAS, cuDNN…), loaded without being imported. The same, so, whether
    a job starts or resumes, before or after its models load."""
    for path in sorted(PACKAGE.rglob("*.py")):
        parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
        # The entry point, or no module at all: a stray file, such as an editor's backup,
        # which settings.code takes in all the same.
        if parts[-1] == "__main__" or not all(part.isidentifier() for part in parts):
            continue
        importlib.import_module(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
    names = imported() | _libraries("torch")
    return {name: metadata.version(name) for name in sorted(names)}


def imported() -> set[str]:
    """The run's distributions this process has imported, by their normalised names: those within
    the closure of seedvr2x's declared requirements, and flash-attn. What else a process imports
    is no part of the run, and a resume mustn't depend on it: pytest, pygments when a dev install
    has it (httpx 0.28.1 imports it for its command line, httpx/__init__.py:15), a profiler, and
    what a model file's download alone imports (DOWNLOAD_ONLY)."""
    providers = _providers()
    tops = {module.partition(".")[0] for module in list(sys.modules)} - set(START_HOOKS)
    loaded = {_name(distribution) for top in tops for distribution in providers.get(top, ())}
    return (loaded & _declared()) - DOWNLOAD_ONLY


@functools.cache
def _providers() -> Mapping[str, list[str]]:
    """The distributions providing each top-level module, read once: reading every installed
    distribution's files takes a few tenths of a second, and they don't change while a process
    runs."""
    return metadata.packages_distributions()


@functools.cache
def _declared() -> frozenset[str]:
    """The distributions seedvr2x requires and flash-attn, and their requirements, through the
    extras each asks: read once, as _providers."""
    return frozenset(_required([*(metadata.requires("seedvr2x") or []), *UNDECLARED]))


def _libraries(root: str) -> set[str]:
    """The distributions root requires, its requirements' requirements and so on, through the
    extras each asks, that ship a shared library other than Python's extension modules: one
    native code loads, which no import shows. Extension modules are seen when imported, as
    torch imports cuda-bindings' (torch 2.14.1, torch/cuda/_utils.py:9), or aren't loaded, as no
    run imports MarkupSafe's (jinja2's speed-ups)."""
    return {
        name
        for name in _required(metadata.requires(root) or [])
        if any(
            SHARED_LIBRARY.search(file.name) and not EXTENSION_MODULE.search(file.name)
            for file in metadata.files(name) or []
        )
    }


def _required(requirements: Iterable[str]) -> set[str]:
    """The installed distributions requirements name (PEP 508), their requirements and so on,
    through the extras each asks. Markers other than extras aren't evaluated: another platform's
    or Python's requirement is rarely installed, and taken then."""
    found: set[str] = set()
    # Each requirement with the extras asked of the distribution declaring it.
    pending: list[tuple[str, frozenset[str]]] = [(each, frozenset()) for each in requirements]
    seen: set[tuple[str, frozenset[str]]] = set()
    while pending:
        requirement, extras = pending.pop()
        match = REQUIREMENT.match(requirement)
        if match is None:
            continue
        asked = _names(EXTRA.findall(requirement.partition(";")[2]))
        if asked and not asked & extras:
            continue  # an option not taken
        name, wanted = _name(match[1]), frozenset(_names((match[2] or "").split(",")))
        if (name, wanted) in seen:
            continue
        seen.add((name, wanted))
        try:
            requires = metadata.requires(name) or []
        except metadata.PackageNotFoundError:
            continue  # not installed: another platform's
        found.add(name)
        pending += [(each, wanted) for each in requires]
    return found


def _names(names: Iterable[str]) -> set[str]:
    return {_name(name) for name in names if name.strip()}


def _name(name: str) -> str:
    """A distribution's name as pip and uv compare it (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name.strip()).lower()

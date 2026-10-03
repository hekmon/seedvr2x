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
SHARED_LIBRARY = re.compile(r"\.so(\.\d+)*$")


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
    those whose modules are imported once every module of seedvr2x is, the vendored models' too,
    which the run imports only when it builds them; and the shared libraries torch requires, its
    runtime (cuBLAS, cuDNN…), loaded without being imported. The same, so, whether a job starts
    or resumes, before or after its models load."""
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
    """The distributions whose modules this process has imported, by their normalised names."""
    providers = _providers()
    return {
        _name(distribution)
        for module in list(sys.modules)
        for distribution in providers.get(module.partition(".")[0], ())
    }


@functools.cache
def _providers() -> Mapping[str, list[str]]:
    """The distributions providing each top-level module, read once: reading every installed
    distribution's files takes a few tenths of a second, and they don't change while a process
    runs."""
    return metadata.packages_distributions()


def _libraries(root: str) -> set[str]:
    """The distributions root requires, its requirements' requirements and so on, through the
    extras each asks, that are installed and ship a shared library."""
    found: set[str] = set()
    pending: list[tuple[str, frozenset[str]]] = [(_name(root), frozenset())]
    seen: set[tuple[str, frozenset[str]]] = set()
    while pending:
        name, extras = pending.pop()
        if (name, extras) in seen:
            continue
        seen.add((name, extras))
        try:
            requirements = metadata.requires(name) or []
        except metadata.PackageNotFoundError:
            continue  # not installed: another platform's
        for requirement in requirements:
            match = REQUIREMENT.match(requirement)
            if match is None:
                continue
            marker = requirement.partition(";")[2]
            asked = {_name(extra) for extra in EXTRA.findall(marker)}
            if asked and not asked & extras:
                continue  # an option not taken
            distribution = _name(match[1])
            try:
                files = metadata.files(distribution) or []
            except metadata.PackageNotFoundError:
                continue
            if any(SHARED_LIBRARY.search(file.name) for file in files):
                found.add(distribution)
            pending.append((distribution, frozenset(_names((match[2] or "").split(",")))))
    return found


def _names(names: Iterable[str]) -> set[str]:
    return {_name(name) for name in names if name.strip()}


def _name(name: str) -> str:
    """A distribution's name as pip and uv compare it (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name.strip()).lower()

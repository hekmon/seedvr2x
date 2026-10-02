"""The ffmpeg build seedvr2x runs with, checked once at startup (DESIGN.md, Input)."""

import shutil
import subprocess
from pathlib import Path

# What seedvr2x asks of the build: zscale for every colour conversion (swscale's 8 → 16-bit
# expansion is inexact, 255 → 65283: DESIGN.md, Input), scdet for the cut detection, FFV1 to
# write the masters and to read lossless segments back. The other filters are in any build, and
# checked for a clear message all the same.
FILTERS = ("zscale", "scdet", "format", "settb", "metadata", "setparams")
ENCODERS = ("ffv1",)
DECODERS = ("ffv1",)

HOW_TO_GET = (
    "zscale comes with libzimg (ffmpeg configured with --enable-libzimg): Ubuntu's ffmpeg package"
    " has it, and so do the builds of https://github.com/BtbN/FFmpeg-Builds"
)


class MediaError(RuntimeError):
    """A source seedvr2x refuses, or a failure of ffmpeg or ffprobe; the message says which."""


def input_args(path: Path) -> list[str]:
    """ffmpeg's input options for a source, the same for every pass over it. Nothing rotates the
    frames behind our back: a rotated source is refused (DESIGN.md, Not in the first version),
    and ffmpeg's rotation would swap the width and height the reader expects."""
    return ["-noautorotate", "-i", str(path)]


def check() -> str:
    """Check the ffmpeg and ffprobe on PATH for what seedvr2x needs and return ffmpeg's version.

    Raises MediaError naming what is missing."""
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise MediaError(f"{tool} not found on PATH: seedvr2x needs ffmpeg and ffprobe")
    filters, encoders, decoders = _listed("-filters"), _listed("-encoders"), _listed("-decoders")
    missing = [
        *(f"the {name} filter" for name in FILTERS if name not in filters),
        *(f"the {name} encoder" for name in ENCODERS if name not in encoders),
        *(f"the {name} decoder" for name in DECODERS if name not in decoders),
    ]
    if missing:
        raise MediaError(
            f"{shutil.which('ffmpeg')} lacks {', '.join(missing)}."
            + (f" {HOW_TO_GET}." if "the zscale filter" in missing else "")
        )
    version = _run("-version").splitlines()
    return version[0].split()[2] if version and len(version[0].split()) > 2 else "unknown"


def _listed(option: str) -> set[str]:
    """The names ffmpeg lists for -filters, -encoders or -decoders: the second column of each
    line, after the flags."""
    return {fields[1] for fields in map(str.split, _run(option).splitlines()) if len(fields) > 1}


def _run(option: str) -> str:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", option], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise MediaError(f"ffmpeg {option} failed: {result.stderr.strip()}")
    return result.stdout

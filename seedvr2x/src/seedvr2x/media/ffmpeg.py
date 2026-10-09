"""The ffmpeg build seedvr2x runs with, checked once at startup (DESIGN.md, Input)."""

import shutil
import subprocess
from pathlib import Path

# What seedvr2x asks of the build: zscale for every colour conversion (swscale's 8 → 16-bit
# expansion is inexact, 255 → 65283: DESIGN.md, Input), scdet for the cut detection, FFV1 to
# write the masters and to read lossless segments back. The other filters are in any build, and
# checked for a clear message all the same.
FILTERS = ("zscale", "scdet", "format", "settb", "metadata", "setparams", "split")
ENCODERS = ("ffv1",)
DECODERS = ("ffv1",)
# Options of those filters that older builds lack: (filter, option, the ffmpeg that brought it).
# setparams' chroma_location tags the yuv420p10le master's frames with its chroma siting
# (media/writer.py:247, master_filters), and so the conversions' fingerprint runs it at every
# start, whatever the output (media/fingerprint.py:90-96, _conversions). It came with ffmpeg
# 7.1: libavfilter/vf_setparams.c:124 at n7.1, absent at n6.1 and n7.0. A build without it passed
# the check of names, then failed the fingerprint with ffmpeg's own error (Ubuntu 24.04's 6.1.1:
# "Error applying option 'chroma_location' to filter 'setparams': Option not found"). Checked by
# the option, as the filter's help lists it, never by the version string: a release's git build
# says n9.0.2-22-g46d8f462ee, a build of master N-<commits>-g<hash>, with no version at all
# (ffbuild/version.sh at n9.0.2), a distribution's 7.1.1-1ubuntu1.
OPTIONS = (("setparams", "chroma_location", "7.1"),)

# BtbN's builds are of ffmpeg's master and latest release branches (8.1 and 9.0 on 2026-10-08),
# with libzimg. Ubuntu's ffmpeg package is built with libzimg, at 7.1.1 in 25.04 and 25.10, 8.0.1
# in 26.04, 6.1.1 in 24.04 (Launchpad, 2026-10-09).
HOW_TO_GET = (
    "seedvr2x needs ffmpeg 7.1 or later with zscale, which comes with libzimg (ffmpeg configured"
    " with --enable-libzimg): the builds of https://github.com/BtbN/FFmpeg-Builds will do, and so"
    " will Ubuntu's ffmpeg package from 25.04 on (24.04's is 6.1)"
)


class MediaError(RuntimeError):
    """A source seedvr2x refuses, or a failure of ffmpeg or ffprobe; the message says which."""


def input_args(path: Path) -> list[str]:
    """ffmpeg's input options for a source, the same for every pass over it. Nothing rotates the
    frames behind our back: a rotated source is refused (DESIGN.md, Not in the first version),
    and ffmpeg's rotation would swap the width and height the reader expects."""
    return ["-noautorotate", "-i", str(path)]


# Every zscale runs on one slice (DESIGN.md, Input; research/docs/numerics.md, The master's
# chroma: 4:2:0 kernels and zscale's slices). ffmpeg cuts zscale into slices by default, one per
# CPU the process may use, each a zimg graph of its own whose vertical chroma filter stops at the
# slice's edge. On the first 3 frames of anime-clean's output, a yuv420p10le master then changes
# with the machine on the slices' edge rows: up to 1.7% of its chroma samples, by up to 13 ten-bit
# codes, other bytes at every count from 1 to 16. A 10-bit 4:2:0 source also reads off the exact
# conversion from 4 slices on, on every row, which the edges don't explain: on anime-clean nearly
# every sample, by up to 0.57 level; on tests/test_zscale.py's random 1080p frame 66% of the
# samples, by up to 12,703 sixteen-bit codes. An 8-bit one reads exactly at any count: why 10-bit
# and not 8-bit is not known. One slice costs 2.3-2.6 ms per 1080p frame, against about 4 s of
# GPU time. `threads` is libavfilter's generic per-filter option, not one of zscale's own:
# `ffmpeg -h filter=zscale` doesn't list it, `ffmpeg -h full` does (AVFilter AVOptions:
# thread_type, enable, threads), and ffmpeg refuses any option a filter lacks (exit 8).
def zscale(*options: str) -> str:
    """The zscale filter with zscale's own options, key=value strings, on one slice. Every zscale
    seedvr2x builds comes from here (tests/test_zscale.py)."""
    return ":".join(("zscale=threads=1", *options))


def check(output_encoders: tuple[str, ...] = (), output_muxers: tuple[str, ...] = ()) -> str:
    """Check the ffmpeg and ffprobe on PATH for what seedvr2x needs, the options of OPTIONS
    included, and for the output's own encoders and muxers (png for PNG output; framehash, which
    hashes a yuv420p10le master's frames), and return ffmpeg's version.

    Raises MediaError naming what is missing, and which ffmpeg to get when it is zscale or an
    option of a newer ffmpeg (HOW_TO_GET)."""
    found()
    filters, encoders, decoders = _listed("-filters"), _listed("-encoders"), _listed("-decoders")
    muxers = _listed("-muxers") if output_muxers else set[str]()
    options = [
        f"the {option} option of its {name} filter (ffmpeg {since} and later have it)"
        for name, option, since in OPTIONS
        if name in filters and option not in _options(name)
    ]
    missing = [
        *(f"the {name} filter" for name in FILTERS if name not in filters),
        *options,
        *(f"the {name} encoder" for name in ENCODERS + output_encoders if name not in encoders),
        *(f"the {name} decoder" for name in DECODERS if name not in decoders),
        *(f"the {name} muxer" for name in output_muxers if name not in muxers),
    ]
    if missing:
        raise MediaError(
            f"{shutil.which('ffmpeg')} lacks {', '.join(missing)}."
            + (f" {HOW_TO_GET}." if options or "the zscale filter" in missing else "")
        )
    version = _run("-version").splitlines()
    return version[0].split()[2] if version and len(version[0].split()) > 2 else "unknown"


def found() -> None:
    """Raise MediaError unless ffmpeg and ffprobe are on PATH."""
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise MediaError(f"{tool} not found on PATH: seedvr2x needs ffmpeg and ffprobe")


def _listed(option: str) -> set[str]:
    """The names ffmpeg lists for -filters, -encoders or -decoders: the second column of each
    line, after the flags."""
    return {fields[1] for fields in map(str.split, _run(option).splitlines()) if len(fields) > 1}


def _options(name: str) -> set[str]:
    """The options ffmpeg lists for the filter `name` (-h filter=name): the first column of each
    line whose second is a type in angle brackets, <int>, where a named value's is the value."""
    return {
        fields[0]
        for fields in map(str.split, _run("-h", f"filter={name}").splitlines())
        if len(fields) > 1 and fields[1].startswith("<") and fields[1].endswith(">")
    }


def _run(*options: str) -> str:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", *options], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise MediaError(f"ffmpeg {' '.join(options)} failed: {result.stderr.strip()}")
    return result.stdout

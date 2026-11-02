"""The ffmpeg build seedvr2x runs with, checked once at startup (DESIGN.md, Input), and what the
assembly starts its children with, to end with seedvr2x (ends_with_us)."""

import ctypes
import functools
import os
import shutil
import signal
import subprocess
from collections.abc import Callable
from pathlib import Path

# What seedvr2x's own ffmpeg commands use, every one checked, so that a build lacking one is
# refused at startup, saying which, rather than failing a pass hours in. zscale for every colour
# conversion (swscale's 8 → 16-bit expansion is inexact, 255 → 65283: DESIGN.md, Input), idet for
# combed frames in the first pass (media/scan.py), FFV1 to write the masters and read them back
# are the ones a build may well lack; the others are in any build, and checked for a clear
# message all the same. scdet is no longer asked: TransNetV2 detects the cuts (DESIGN.md, Shot
# detection), and nothing runs it; nor settb and metadata, the first pass's before it read the
# decoder's timestamps (media/scan.py).
FILTERS = (
    "zscale",  # every colour conversion: the decode's (media/conversion.py), the writers'
    "format",  # the formats the conversions are fed and give
    # Inserted by ffmpeg where a format filter meets a packed format, to repack it (DESIGN.md,
    # Not in the first version: 26 packed or semi-planar formats, repacked exactly), and before
    # idet for a format it doesn't take (libavfilter/vf_idet.c:385-417 at n9.0.2): RGB.
    "scale",
    "setparams",  # the writers' colour tags, its chroma_location (OPTIONS)
    "split",  # the yuv420p10le writer's two branches, its file and its hashes (media/writer.py)
    # The reads' frames from the pts of the first asked (media/reader.py), in an output of their
    # own, no split; alone, no trim: the reads seek without accurate seek (-noaccurate_seek),
    # whose trim fftools would put in front of each graph (fftools/ffmpeg_filter.c:1945-1947).
    "select",
    "idet",  # the first pass's combing detection (media/scan.py)
    # The graph fftools gives an output without filters (fftools/ffmpeg_mux_init.c:436): the
    # first pass's hashes, the checksums' decode (media/checksums.py).
    "null",
)
# rawvideo: the frames given raw, on a pipe (-f rawvideo) or to framehash; wrapped_avframe, the
# null muxer's, the first pass's idet output (libavformat/nullenc.c:34).
ENCODERS = ("ffv1", "rawvideo", "wrapped_avframe")
# rawvideo: the writers' input and the fingerprint's frames (-f rawvideo -i).
DECODERS = ("ffv1", "rawvideo")
# framehash: the first pass's and the reads' CRC-32s (media/scan.py, media/reader.py); null: the
# first pass's idet output; rawvideo: the frames on a pipe; matroska: the FFV1 masters and the
# input copies, whatever the output's format.
MUXERS = ("framehash", "null", "rawvideo", "matroska")
# What an assembly's commands use besides (media/assembly.py), asked only of a job that assembles
# its segments into one file (check's `assembly`), as png and image2 are of a PNG output: the
# concat demuxer, which reads the segments as one stream; the Matroska demuxer, which reads them
# and the file assembled; the setts bitstream filter, which puts the video's timestamps on the
# frame grid, its pts, dts and duration expressions in every ffmpeg since 7.1
# (libavcodec/bsf/setts.c at n7.1). The Matroska and framehash muxers it writes with are MUXERS'.
# All in any build but one configured without them, and checked for a clear message before the
# upscale, where the join comes after it.
ASSEMBLY_DEMUXERS = ("concat", "matroska")
ASSEMBLY_BITSTREAM_FILTERS = ("setts",)
# Options of those filters that older builds lack: (filter, option, the ffmpeg that brought it).
# setparams' chroma_location tags the yuv420p10le master's frames with its chroma siting
# (media/writer.py:285, master_filters), and so the conversions' fingerprint runs it at every
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


# prctl's option that has the kernel send the calling process a signal when its parent ends
# (linux/prctl.h).
PR_SET_PDEATHSIG = 1


@functools.cache
def _libc() -> ctypes.CDLL:
    """The C library, opened once and before any fork, so that a child only calls into it."""
    return ctypes.CDLL(None, use_errno=True)


@functools.cache
def _prctl() -> Callable[..., int]:
    """libc's prctl, looked up and typed once, in the parent. ctypes resolves a library's
    attribute lazily (dlsym, a new function object, a cache set on the library): all of it
    allocates and takes locks, which a child of a multi-threaded process (seedvr2x forks with
    other threads alive) should not do between its fork and its exec, where a lock another
    thread held at the fork stays held for ever. Resolving it here, in the parent, removes from
    the child the dlsym lookup, the loader's lock and the CDLL attribute cache; the call itself
    still converts its arguments into new objects, which no ctypes call avoids, and Python's
    preexec_fn runs Python code that allocates anyway. Design's ruling, baton 3 of v0.1.0
    (2026-10-30): resolve it in the parent, before any fork."""
    function = _libc().prctl
    function.argtypes = (ctypes.c_int, ctypes.c_ulong)
    function.restype = ctypes.c_int
    return function


def ends_with_us() -> Callable[[], None]:
    """A `preexec_fn` for every ffmpeg and ffprobe seedvr2x starts, which has it killed (SIGKILL)
    when seedvr2x dies: DESIGN.md, Output, Assembly (decided on 2026-10-12 for the assembly's:
    "ffmpeg's children of the assembly end with seedvr2x, where a killed seedvr2x leaves its join
    writing"; on 2026-10-29 for all of them). Their own process groups keep them out of the
    terminal's reach (DESIGN.md, Pause and resume, Stopping), so nothing but this ends them when
    seedvr2x is killed. Not the user's `--segment-cmd` (media/writer.py, CommandWriter), which
    ends at its input's end (DESIGN.md, same place).

    prctl(PR_SET_PDEATHSIG) fires when the thread that forked ends, not the process: every
    caller forks from seedvr2x's main thread, which is the process's life, or from a thread that
    waits for its child (media/scan.py, packets, run beside the decode). It is set after the
    fork, so a parent that died in between is not seen by the kernel: the child checks, and ends
    itself. Linux only (DESIGN.md: Linux first); no other system is run, so no fallback.

    The function is resolved here, in the parent, and `prepare` only calls the function object
    it closes over: that removes from the child the dlsym lookup, the loader's lock and the
    CDLL attribute cache, which ctypes' lazy attribute lookup would use there (_prctl; design,
    baton 3, 2026-10-30); not all allocation, which a Python preexec_fn cannot avoid. The
    signal's number is taken here too, for the same reason."""
    prctl = _prctl()
    kill = int(signal.SIGKILL)
    parent = os.getpid()

    def prepare() -> None:
        if prctl(PR_SET_PDEATHSIG, kill) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
        if os.getppid() != parent:
            os._exit(1)  # the parent ended before the signal was set

    return prepare


# No output takes the source's metadata or chapters, which fftools copies to every output by
# default: global metadata, each stream's, the chapters with theirs. -map_metadata -1 stops the
# three copies at once (fftools/ffmpeg_mux_init.c:2911-2916, 3029-3046 at n9.0.2: an output
# without a specifier and no input sets every one manual), -map_chapters -1 the chapters
# themselves (:3024-3026). A muxer's header is printed at info level once its streams are ready,
# while the frames are decoded (fftools/ffmpeg_mux.c:566), a title among it, and each stream's
# language: text of the source's own among the lines the first pass reads (media/scan.py,
# MAPPING). Checked on 2026-10-10: with them, the outputs' headers print their "encoder" tags
# alone.
NO_METADATA = ("-map_metadata", "-1", "-map_chapters", "-1")


def argument(path: Path) -> str:
    """path as one word of an ffmpeg or ffprobe command line, a name they take as a file's
    whatever it holds: a relative name gets `./`. They take what comes before a name's first colon
    for a protocol when it is made of letters, digits, `+`, `-` and `.`: a relative `x: y/a.mkv` is
    refused, "Protocol not found" (ffmpeg and ffprobe n9.0.2, 2026-10-28; media/assembly.py,
    _absolute, found the same of `Film: 2.mkv` on 2026-10-12), where the same name after `./` is a
    file's. A name beginning with `-` is an option to ffmpeg as its last word, "Unrecognized
    option". An absolute name begins with a slash: neither. Only the name on the command line
    changes, so the bytes written are the same, and the messages say the path as the user gave it
    (DESIGN.md, Output, Assembly, decided on 2026-10-12: "a relative `-o` whose name holds ": " is
    taken by ffmpeg for a protocol" goes)."""
    return str(path) if path.is_absolute() else f"./{path}"


def input_args(path: Path) -> list[str]:
    """ffmpeg's input options for a source, the same for every pass over it. Nothing rotates the
    frames behind our back: a rotated source is refused (DESIGN.md, Not in the first version),
    and ffmpeg's rotation would swap the width and height the reader expects."""
    return ["-noautorotate", "-i", argument(path)]


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


def check(
    output_encoders: tuple[str, ...] = (),
    output_muxers: tuple[str, ...] = (),
    assembly: bool = False,
) -> str:
    """Check the ffmpeg and ffprobe on PATH for what seedvr2x needs (FILTERS, ENCODERS, DECODERS,
    MUXERS), the options of OPTIONS included, for the output's own encoders and muxers (png
    and image2 for PNG output), and, when `assembly`, for what joining the segments into one
    file needs (ASSEMBLY_DEMUXERS, ASSEMBLY_BITSTREAM_FILTERS), and return ffmpeg's version.

    Raises MediaError naming what is missing, and which ffmpeg to get when it is zscale or an
    option of a newer ffmpeg (HOW_TO_GET)."""
    found()
    filters, encoders, decoders = _listed("-filters"), _listed("-encoders"), _listed("-decoders")
    muxers = _listed("-muxers")
    # Asked of ffmpeg only for a job that assembles: two more of its listings.
    demuxers = _demuxers() if assembly else set[str]()
    bitstream_filters = _bitstream_filters() if assembly else set[str]()
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
        *(f"the {name} muxer" for name in MUXERS + output_muxers if name not in muxers),
        *(f"the {name} demuxer" for name in ASSEMBLY_DEMUXERS if assembly and name not in demuxers),
        *(
            f"the {name} bitstream filter"
            for name in ASSEMBLY_BITSTREAM_FILTERS
            if assembly and name not in bitstream_filters
        ),
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
    """The names ffmpeg lists for -filters, -encoders, -decoders or -muxers: the second column
    of each line, after the flags."""
    return {fields[1] for fields in map(str.split, _run(option).splitlines()) if len(fields) > 1}


def _demuxers() -> set[str]:
    """The names ffmpeg lists for -demuxers, each of a demuxer of several: "matroska,webm" is
    one line, and both names."""
    return {name for names in _listed("-demuxers") for name in names.split(",")}


def _bitstream_filters() -> set[str]:
    """The names ffmpeg lists for -bsfs: one a line, under a title line ending in a colon."""
    lines = (line.strip() for line in _run("-bsfs").splitlines())
    return {line for line in lines if line and not line.endswith(":")}


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
        ["ffmpeg", "-hide_banner", *options],
        capture_output=True,
        text=True,
        check=False,
        preexec_fn=ends_with_us(),
    )
    if result.returncode != 0:
        raise MediaError(f"ffmpeg {' '.join(options)} failed: {result.stderr.strip()}")
    return result.stdout

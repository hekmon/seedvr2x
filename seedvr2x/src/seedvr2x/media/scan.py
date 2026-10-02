"""The first pass over a source: every frame decoded, counted and timed, before any GPU work
(DESIGN.md, Input). Automatic scene detection will join it."""

import logging
import re
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from seedvr2x.media.ffmpeg import MediaError, input_args

logger = logging.getLogger(__name__)

# sptenc's CountFrames (ffmpeg/count.go): each frame printed with its pts in microseconds,
# whatever the container's time base (settb; pts_time has 6 significant digits only), by a
# metadata filter that prints the frames carrying a key, which the first one gives to all.
FILTERS = "settb=AVTB,metadata=mode=add:key=seedvr2x:value=1,metadata=mode=print:key=seedvr2x"
FRAME = re.compile(r"\] \[info\] frame:(\d+)\s+pts:(\S+)")
ERROR = re.compile(r"\[(error|fatal|panic)\]")

# How much the time between two frames may vary in a constant frame rate stream: sptenc's
# tolerance (ffmpeg/probe.go, IsConstantFrameRate). Matroska rounds timestamps to the
# millisecond, so 23.976 fps lasts 41 and 42 ms; 2 ms would let a mix of 24 and 25 fps through.
TOLERANCE_US = 1000


@dataclass(frozen=True)
class Scan:
    """What the first pass measured."""

    frames: int  # decoded and counted, never the container's count (bug 11)
    durations: int  # times measured between consecutive frames that have a timestamp
    shortest: int  # the shortest of them, in microseconds; 0 without any
    longest: int  # the longest, in microseconds; 0 without any


def scan(path: Path) -> Scan:
    """Decode every frame of the first video stream of path, counting and timing them."""
    command = [
        *("ffmpeg", "-hide_banner", "-nostdin", "-nostats", "-loglevel", "level+info"),
        *input_args(path),
        *("-map", "0:v:0", "-vf", FILTERS, "-fps_mode", "passthrough", "-f", "null", "-"),
    ]
    process = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, errors="replace"
    )
    assert process.stderr is not None
    frames = durations = shortest = longest = 0
    previous: int | None = None
    errors: list[str] = []
    for line in process.stderr:
        match = FRAME.search(line)
        if match is None:
            if ERROR.search(line):
                errors.append(line.strip())
            continue
        frames += 1
        # A frame without a timestamp breaks the chain: the next time measured is between the
        # two frames after it (sptenc).
        pts = None if match[2] == "NOPTS" else int(match[2])
        if pts is not None and previous is not None:
            duration = pts - previous
            shortest = duration if durations == 0 else min(shortest, duration)
            longest = duration if durations == 0 else max(longest, duration)
            durations += 1
        previous = pts
    if process.wait() != 0:
        raise MediaError(f"{path}: ffmpeg failed decoding it: {' / '.join(errors[-5:])}")
    if errors:
        logger.warning("%s: ffmpeg reported errors decoding it: %s", path, " / ".join(errors[:5]))
    return Scan(frames, durations, shortest, longest)


def timing_error(scanned: Scan, frame_rate: Fraction, avg_frame_rate: Fraction | None) -> str:
    """Why the frames' timing is refused, or "" when they have the constant rate declared.

    With frame times measured (sptenc's rule), the shortest and longest must be within
    TOLERANCE_US, and the declared rate within them, since the output is written at that rate.
    Without (a single frame, no timestamps), the declared rates are compared, as sptenc does."""
    if scanned.frames == 0:
        return "no frame decoded"
    if scanned.durations:
        if scanned.longest - scanned.shortest > TOLERANCE_US:
            return (
                f"variable frame rate: frames last from {scanned.shortest / 1000:g} to"
                f" {scanned.longest / 1000:g} ms, and only a constant frame rate is supported"
            )
        period = Fraction(1_000_000) / frame_rate
        if not scanned.shortest - TOLERANCE_US <= period <= scanned.longest + TOLERANCE_US:
            return (
                f"the frame rate declared, {frame_rate} fps, is not the frames' own: they last"
                f" {scanned.shortest / 1000:g} to {scanned.longest / 1000:g} ms"
            )
        return ""
    if avg_frame_rate is None or abs(frame_rate - avg_frame_rate) > Fraction(1, 1000):
        return (
            f"the frame rate can't be checked (no frame times) and the declared ones differ:"
            f" {frame_rate} and {avg_frame_rate or 'none'} fps"
        )
    return ""

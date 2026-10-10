"""A source's frames against the timeline of a constant rate (DESIGN.md, Input: exact rational
frame rate): each frame's deviation from that timeline, anchored at the first frame; the rate
their timestamps are exactly (exact), which a refusal of a contradicted rate gives with two fixes
(media/source.py, timing_refusal), and the warning of a file leaving its declared rate's timeline
names; the constant rate the frames follow within half a frame (follows), that refusal's when
they are no rate's exactly; and --frame-rate R, accepted when every frame lies within half a
frame of R's timeline. Exact throughout: the frames' pts in their stream's time base, as the
frame index holds them, and rates as fractions."""

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
import numpy.typing as npt

from seedvr2x.media.index import NOPTS, FrameIndex

# Beyond it, a deviation's numerator could overflow numpy's int64 (_numerators): Python's exact
# integers then, slower.
INT64_SAFE = 2**62

# PROVISIONAL (DESIGN.md says "that rate" and not how it is found): the rates a constant rate is
# looked for among, the rates ffmpeg itself guesses a stream's r_frame_rate from
# (libavformat/demux.c:2287-2303 at n9.0.2, get_std_framerate): every multiple of 1/12 fps, and the
# NTSC forms N * 1000/1001; here without ffmpeg's bounds (multiples of 1/12 up to 30 fps, integers
# up to 60, then 80, 120 and 240, six NTSC forms), so that 100 and 120000/1001 are found too. Not
# the simplest fraction the frames allow: for S9 (142,477 frames, research/docs/seeking.md
# mechanism 7) that is 2997/125, 23.976 exactly, a rate no file declares, where its timestamps run
# at 24000/1001, its audio's.
TWELFTHS = 12
NTSC = Fraction(1000, 1001)
# The most rates follows tries: more means a span too short to tell one rate from the next (two
# frames 1 ms apart allow any from 500 to 1,500 fps, some 13,000 of them), so none is given.
MOST_RATES = 1000

# The FFV1 master the remake makes, as seedvr2x writes its own (media/writer.py, FFV1Writer):
# every frame a keyframe, each slice with its CRC.
FFV1 = ("-c:v:0", "ffv1", "-level:v:0", "3", "-g:v:0", "1", "-slicecrc:v:0", "1")


@dataclass(frozen=True)
class Timeline:
    """How a source's frames lie on the timeline of `rate`, anchored at its first frame that has a
    pts: each frame's deviation, its time less its place's on that timeline, in seconds, exact;
    frames without a pts have none."""

    rate: Fraction  # frames per second
    lowest: Fraction  # the lowest deviation, in seconds, 0 or below
    highest: Fraction  # the highest deviation, in seconds, 0 or above
    first_off: int | None  # the first frame half a frame or more off, None when none is
    first_off_by: Fraction  # that frame's deviation, in seconds; 0 when none is off
    off: int  # how many frames are half a frame or more off

    @property
    def largest(self) -> Fraction:
        """The largest deviation either way, in seconds."""
        return max(-self.lowest, self.highest)

    @property
    def holds(self) -> bool:
        """Whether every frame lies strictly within half a frame of its place on the timeline: the
        nearest place is then each frame's own, so that, taken at that rate, no frame is dropped
        or doubled."""
        return self.first_off is None


def timeline(index: FrameIndex, rate: Fraction) -> Timeline:
    """The frames of index on the timeline of `rate` (frames per second, positive), anchored at
    the first frame that has a pts."""
    places, pts = _timed(index)
    if not len(places):
        return Timeline(rate, Fraction(0), Fraction(0), None, Fraction(0), 0)
    numerators, scale = _numerators(index.time_base, places, pts, rate)
    # A deviation is numerator / scale seconds, scale q N for a time base p/q and a rate N/D, and
    # half a frame D / 2N seconds: a frame is half a frame or more off when |2 numerator| >= q D,
    # integers.
    off = np.abs(2 * numerators) >= index.time_base.denominator * rate.denominator
    found = np.flatnonzero(off)
    first = int(found[0]) if len(found) else None
    return Timeline(
        rate,
        Fraction(int(numerators.min()), scale),
        Fraction(int(numerators.max()), scale),
        None if first is None else int(places[first]),
        Fraction(0) if first is None else Fraction(int(numerators[first]), scale),
        len(found),
    )


def follows(index: FrameIndex) -> Timeline | None:
    """The constant rate the frames follow: of the rates looked for (TWELFTHS, NTSC), those whose
    timeline holds every frame within half a frame, the one the frames lie steadiest on, by the
    spread of their deviations in frames, the highest less the lowest, whatever the anchor; the
    simplest fraction on a tie. None when none does, fewer than two frames have a pts, or their
    span allows too many rates (MOST_RATES). Not the largest deviation from the first frame's
    place: frames 0 and 10 ms after theirs in turn, around 25 fps, lie within 10 ms of 25's
    timeline and 9.96 ms of 25000/1001's, which drifts 0.04 ms a frame, their spread 10 and 13.9.

    Only rates the frames' span allows are tried: within half a frame of its place, the last frame,
    k frames after the first and t seconds later, bounds the rate to ((k - 1/2) / t, (k + 1/2) /
    t)."""
    places, pts = _timed(index)
    if len(places) < 2:
        return None
    span = int(pts[-1] - pts[0]) * index.time_base
    steps = int(places[-1] - places[0])
    if span <= 0:
        return None
    low, high = (steps - Fraction(1, 2)) / span, (steps + Fraction(1, 2)) / span
    if (high - low) * TWELFTHS > MOST_RATES:
        return None
    best: Timeline | None = None
    for rate in _rates(low, high):
        found = timeline(index, rate)
        if found.holds and (best is None or _spread(found) < _spread(best)):
            best = found
    return best


def exact(index: FrameIndex) -> Timeline | None:
    """The constant rate the frames' timestamps are exactly, to their time base's rounding, or
    None: a rate whose timeline holds every frame within half a frame, their deviations from it
    spread over one tick of the time base at most, the highest less the lowest.

    PROVISIONAL (implementation, 2026-10-10, a question for design, with the warning that names
    the rate: media/source.py, _strays). A constant rate's times, rounded to the time base's
    ticks, each lie within half a tick of a timeline of that rate (rounded to the nearest tick,
    a tie either way; rounded down, of one half a tick earlier): whatever the timeline's anchor,
    their deviations from it spread over one tick at most. Measured in Matroska, whose tick is
    1 ms (2026-10-10): frames at 24000/1001 and at 48 fps, declared 24/1 and 50/1, spread 0.96
    and 0.83 of a tick on their own rate's timeline, within 0.5 ms of it, 600 and 3,000 frames
    alike; 33 ms a frame, 0 on 1000/33's; 2,400 frames joined by ffmpeg's concat demuxer 1.67
    to 2 ticks on the rate looked for nearest them (30000/1001 joined every 10 or 12 frames on
    30/1's, within 1.0 ms; 24000/1001 every 24 or 20 on 24/1's, within 1.3 and 1.7 ms), and 2
    to 2.6 on their own mean rate's. Half a frame, the bound a rate holds the frames within
    (follows), took those joins for files at 30/1 and 24/1; 33 and 42 ms a frame for 91/3 and
    143/6 fps over 120 to 300 frames, within 3.9 to 12.5 ms, rates that 3,000 frames of each
    leave; and 17 ms a frame for 353/6 fps, within 0.85 ms over 300 frames and 8.5 over 3,000
    (a review's findings, 2026-10-10).

    The rates tried: the frames' own mean rate, their steps over their span, given when every
    frame lies on its timeline, a constant step in ticks, whatever the rate (1000/33 fps for 33
    ms a frame, at any length; 17 ms a frame, 1000/17, lies within 0.85 ms of 353/6's timeline
    too over 300 frames); then the rates looked for (TWELFTHS, NTSC) that the span allows, its
    last frame within a tick of its place, when one alone holds the frames so; then the mean
    rate again, when none of them does and its timeline holds the frames so. A rate that is
    none of those, rounded to ticks, isn't always found: 24.03 fps over 1,000 frames spread
    1.03 ticks on its mean rate's timeline, which the rounding of the first and last frames
    tilts; no rate is then given.

    Nor is one when several of the rates looked for hold the frames so, a file too short to
    tell them apart: the one the frames lay steadiest on was given, and of 12,872 synthetic
    files 222, all under 83 frames, were named a neighbour of their rate (a review's finding,
    2026-10-10): 120000/1001 as 120/1, 479/4, 719/6 or 1439/12 over 4 to 41 frames, 48000/1001
    as 48/1 or 575/12 over 12 to 13. In milliseconds, several hold the first 12 frames of
    24000/1001, 18 of 48000/1001 and 60 of 120000/1001, 24/1, 575/12 and 120/1 among them, and
    one alone any more of them, up to 1,500."""
    places, pts = _timed(index)
    if len(places) < 2:
        return None
    ticks = int(pts[-1] - pts[0])
    steps = int(places[-1] - places[0])
    if ticks <= 0:
        return None
    tick = index.time_base
    mean = timeline(index, steps / (ticks * tick))
    if mean.largest == 0:
        return mean
    if ticks <= 2:
        return None  # so short a span allows any rate
    # The rates that put the last frame within two ticks of its place: a tick more than a rate
    # that holds the frames so allows, _rates leaving its bounds out.
    low, high = steps / ((ticks + 2) * tick), steps / ((ticks - 2) * tick)
    if (high - low) * TWELFTHS > MOST_RATES:
        return None
    fitting = [
        found for rate in _rates(low, high) if _rounded(found := timeline(index, rate), tick)
    ]
    if len(fitting) > 1:
        return None  # too short to tell them apart
    if fitting:
        return fitting[0]
    return mean if _rounded(mean, tick) else None


def _rounded(found: Timeline, tick: Fraction) -> bool:
    """Whether the frames lie on a timeline as that rate's times rounded to `tick`, seconds, do
    (exact): within half a frame, their deviations spread over one tick at most."""
    return found.holds and found.highest - found.lowest <= tick


def _spread(found: Timeline) -> Fraction:
    """The spread of the frames' deviations from a timeline, in frames."""
    return (found.highest - found.lowest) * found.rate


def _rates(low: Fraction, high: Fraction) -> list[Fraction]:
    """The rates looked for (TWELFTHS, NTSC) strictly between low and high, the simplest fractions
    first."""
    found = {
        Fraction(k, TWELFTHS)
        for k in range(math.floor(low * TWELFTHS) + 1, math.ceil(high * TWELFTHS))
    }
    found.update(n * NTSC for n in range(math.floor(low / NTSC) + 1, math.ceil(high / NTSC)))
    return sorted((rate for rate in found if low < rate < high), key=lambda r: (r.denominator, r))


def _timed(index: FrameIndex) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """The places of the frames that have a pts, and their pts."""
    places = np.flatnonzero(index.pts != NOPTS).astype(np.int64)
    return places, index.pts[places]


def _numerators(
    time_base: Fraction,
    places: npt.NDArray[np.int64],
    pts: npt.NDArray[np.int64],
    rate: Fraction,
) -> tuple[npt.NDArray[Any], int]:
    """Each frame's deviation from the timeline of `rate` anchored at the first, as integer
    numerators over the scale returned: (pts - pts_0) time_base - (place - place_0) / rate seconds,
    time_base p/q and rate N/D, times q N: (pts - pts_0) p N - (place - place_0) q D; int64, or
    Python's integers where twice a numerator could overflow it (INT64_SAFE)."""
    p, q = time_base.numerator, time_base.denominator
    n, d = rate.numerator, rate.denominator
    ticks = pts - pts[0]
    steps = places - places[0]
    widest = int(np.abs(ticks).max()) * p * n + int(steps.max()) * q * d
    if 2 * widest >= INT64_SAFE:
        return ticks.astype(object) * (p * n) - steps.astype(object) * (q * d), q * n
    return ticks * (p * n) - steps * (q * d), q * n


def said(rate: Fraction) -> str:
    """A rate as messages give it, N/D and its decimal to 3 places, its zeros dropped: 24000/1001
    fps (23.976), 25/1 fps (25)."""
    decimal = f"{float(rate):.3f}".rstrip("0").removesuffix(".")
    return f"{fraction(rate)} fps ({decimal})"


def fraction(rate: Fraction) -> str:
    """A rate written N/D, as ffprobe and --frame-rate write it: 24/1, 24000/1001."""
    return f"{rate.numerator}/{rate.denominator}"


def remake(rate: Fraction, start: Fraction) -> str:
    """The ffmpeg command remaking a source at `rate` (DESIGN.md, Input: "Remake the file"): its
    first video stream, the one seedvr2x reads, an FFV1 master numbering its frames at that rate,
    its other streams copied; `start`, its first frame's time after the file's start, in seconds,
    where that frame stays, the other streams' sync with it kept.

    PROVISIONAL (DESIGN.md names no command): the input option -r:v:0, which gives every frame
    decoded the next place at that rate, from 0 (fftools/ffmpeg_dec.c:398-409 at n9.0.2), so that
    no frame can be dropped or doubled, whatever its timestamp. Of the three retimings
    research/docs/seeking.md measured on S9: fps=R anchors its timeline at 0, not at the first
    frame, and on a fixture whose video starts 23 ms after its audio, its frames within a quarter
    of a frame of 24000/1001's timeline, it dropped one frame and doubled another; -fps_mode cfr
    -r R doubled the first frame there, filling the 23 ms (fftools/ffmpeg_filter.c:2554-2570);
    settb and setpts=N kept every frame but declared the stream's 24/1 again, the rate a filter
    graph takes from its input. Neither -r:v:0 nor setpts=N keeps the first frame's time: 0 for
    the video, its timestamps then shifted back by `start`, in microseconds (settb=AVTB), the
    encoder taking the stream's own time base, so that the shift isn't rounded to the rate's."""
    words = ["ffmpeg", "-r:v:0", fraction(rate), "-i", "SOURCE"]
    words += ["-map", "0", "-map", "-0:d", "-c", "copy", *FFV1, "-fps_mode:v:0", "passthrough"]
    microseconds = round(start * 1_000_000)
    if microseconds:
        words += ["-filter:v:0", f"settb=AVTB,setpts=PTS{microseconds:+d}"]
        words += ["-enc_time_base:v:0", "demux"]
    return " ".join([*words, "remade.mkv"])


def first_time(index: FrameIndex) -> Fraction:
    """The first frame's time after the file's start, in seconds: where ffmpeg puts it when it
    reads the file without -copyts (fftools/ffmpeg_demux.c, ts_offset); 0 without a pts."""
    pts = int(index.pts[0]) if index.frames else NOPTS
    if pts == NOPTS:
        return Fraction(0)
    return pts * index.time_base - Fraction(index.start_time or 0, 1_000_000)

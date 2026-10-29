"""Missing frames filled (DESIGN.md, Input: "Missing frames are filled, not refused"). A source
whose frames lie on its rate's grid with places left empty, as when a decoder drops a keyframe's
leading pictures (research/docs/seeking.md, mechanism 1: S8's second keyframe), has one reading,
a player's: the frame before the hole, held. seedvr2x reads it so by its own rule, from the frame
index, never by ffmpeg's constant-rate mode, which drops and doubles frames elsewhere on its own
judgement. Each decoded frame takes a place on the rate's grid, a place left empty shows the
frame before it again, and everything after the first pass counts in places: the frame index it
reads is the one `filled` builds, a row a place (media/index.py).

This module is the machinery, and judges nothing: each decoded frame's nearest place on a rate's
timeline and how far from it the frame lies, exact (places); the index in places, given each
frame's place (filled), its held rows' timestamps (_ticks) and the shot detector's probabilities
spread over the places so that the cuts fall where they fell (spread); and what was filled, for
the messages and the record (holes, filling). Which frame takes which place, and what is refused,
is media/source.py's to say."""

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
import numpy.typing as npt

from seedvr2x.media.index import NOPTS, FrameIndex

# Beyond it, a product could overflow numpy's int64: Python's exact integers then, slower. The
# bound media/rate.py takes for its own (INT64_SAFE), copied: nothing here calls into rate.py.
INT64_SAFE = 2**62

# The place of a frame without a pts (Places.place): it has none.
NONE = -1


@dataclass(frozen=True, eq=False)
class Places:
    """Each decoded frame's nearest place on the timeline of `rate` (places), and how far from
    it the frame lies. Its arrays are (decoded frames,)."""

    rate: Fraction  # frames per second
    # Each frame's nearest place, counted from the first frame's, 0; NONE without a pts.
    place: npt.NDArray[np.int64]
    timed: npt.NDArray[np.bool_]  # whether the frame has a pts, and so a place
    # Whether the frame lies exactly half a frame from two places: `place` is then the earlier
    # of the two, the frame half a frame after it, and place + 1 is as near.
    tie: npt.NDArray[np.bool_]
    # Each frame's deviation, its time less its place's, in seconds, as integer numerators over
    # `scale`: int64, or Python's integers where int64 could overflow; 0 without a pts. Above
    # minus half a frame, and at most half a frame, which a tie alone reaches.
    numerators: npt.NDArray[Any]
    scale: int

    def deviation(self, frame: int) -> Fraction | None:
        """Frame `frame`'s deviation from its place, in seconds, exact; None without a pts."""
        if not bool(self.timed[frame]):
            return None
        return Fraction(int(self.numerators[frame]), self.scale)

    @property
    def largest(self) -> Fraction:
        """The largest deviation either way, in seconds; 0 when no frame has a pts."""
        if not len(self.numerators):
            return Fraction(0)
        return Fraction(int(np.abs(self.numerators).max()), self.scale)


def places(index: FrameIndex, rate: Fraction) -> Places:
    """Each decoded frame's nearest place on the timeline of `rate` (frames per second,
    positive), counted from the first frame, whose place is 0, with its deviation from that
    place. Exact: the pts in their time base, integers, and the rate a fraction, as
    media/rate.py's deviations are; nothing is judged, and no frame moved: two frames may share
    a place, or go back.

    A frame exactly half a frame from two places is given the earlier, and said (Places.tie): a
    tie is the judge's to settle, never rounded away here.

    Frames without a pts (index.NOPTS, which ffmpeg n9.0.2 never gives) are handled as
    media/rate.py's timeline handles them: left out, without a place (NONE) or a deviation
    (Places.timed says which), each still counted, the timeline anchored at the first frame that
    has a pts, at the place of its own count.

    `index` is the first pass's, of the decoded frames (ValueError with held rows: their places
    are given already)."""
    if index.held is not None:
        raise ValueError("places asked of a frame index with held rows: a seedvr2x bug")
    if rate <= 0:
        raise ValueError(f"rate {rate}, not positive")
    frames = index.frames
    timed = index.pts != NOPTS
    place = np.full(frames, NONE, dtype=np.int64)
    tie = np.zeros(frames, dtype=np.bool_)
    p, q = index.time_base.numerator, index.time_base.denominator
    n, d = rate.numerator, rate.denominator
    found = np.flatnonzero(timed)
    if not len(found):
        return Places(rate, place, timed, tie, np.zeros(frames, dtype=np.int64), q * n)
    anchor = int(found[0])
    ticks: npt.NDArray[Any] = index.pts[found] - index.pts[anchor]
    # A frame's time after the anchor's is ticks p / q seconds, ticks p n / (q d) frames: `after`
    # over `period`, integers.
    period = q * d
    if 2 * int(np.abs(ticks).max()) * p * n + period >= INT64_SAFE:
        ticks = ticks.astype(object)
    after = ticks * (p * n)
    # The nearest whole number of frames, a tie to the earlier: ceil((2 after - period) / (2
    # period)), in floor divisions.
    steps = -((period - 2 * after) // (2 * period))
    # The deviation: after / period - steps frames, of d / n seconds each: over q n, seconds.
    off = after - steps * period
    numerators = np.zeros(frames, dtype=off.dtype)
    numerators[found] = off
    place[found] = anchor + steps.astype(np.int64)
    tie[found] = (2 * off == period).astype(np.bool_)
    return Places(rate, place, timed, tie, numerators, q * n)


def filled(index: FrameIndex, placed: npt.ArrayLike, rate: Fraction) -> FrameIndex:
    """The frame index in places (media/index.py): the first pass's `index`, of the decoded
    frames, each given its own place by `placed`, (decoded frames,) integers, strictly
    increasing from 0 for the first frame; every place between two frames' a held row, showing
    the frame before it again. As many rows as places, the last frame's place plus one.

    A held row's CRC-32 and error flag are those of the frame it shows. Its pts is its place's
    own time, in the stream's time base: the pts of the frame it shows plus as many frame periods
    of `rate` (frames per second) as it lies places after that frame's, rounded to the tick
    (_ticks); none (NOPTS) when that frame has none. It only serves the times shown to a user
    (runtime/cuts.py, time_of): no read looks for a held row's pts, the reads being of the
    decoded frames (media/reader.py). The shot probabilities, when the index has them, are
    spread over the places (spread).

    With no place left empty, `placed` counting 0, 1, 2..., the index returned is `index`
    itself: the first pass's, its file the bytes of before the places. Which place a frame
    takes is the caller's judgement (media/source.py): ValueError for places that aren't one a
    frame, from 0, strictly increasing, or for an index that has held rows already."""
    if index.held is not None:
        raise ValueError("a frame index with held rows filled again: a seedvr2x bug")
    if rate <= 0:
        raise ValueError(f"rate {rate}, not positive")
    own = np.asarray(placed)
    if own.shape != (index.frames,) or (own.size and own.dtype.kind not in "iu"):
        raise ValueError(
            f"places of shape {own.shape} and type {own.dtype} for {index.frames} frames"
        )
    own = own.astype(np.int64)
    if not index.frames:
        return index
    if int(own[0]) != 0:
        raise ValueError(f"the first frame's place is {int(own[0])}, not 0")
    back = np.flatnonzero(np.diff(own) <= 0)
    if len(back):
        frame = int(back[0]) + 1
        raise ValueError(
            f"frame {frame}'s place, {int(own[frame])}, isn't after frame {frame - 1}'s,"
            f" {int(own[frame - 1])}"
        )
    total = int(own[-1]) + 1
    if total == index.frames:
        return index
    held = np.ones(total, dtype=np.bool_)
    held[own] = False
    shown = np.cumsum(~held, dtype=np.int64) - 1
    # How many places after its frame's own each row lies: 0 for a frame's own row.
    after = np.arange(total, dtype=np.int64) - own[shown]
    pts = index.pts[shown]
    pts = np.where(pts == NOPTS, NOPTS, pts + _ticks(after, index.time_base, rate))
    probabilities = None
    if index.probabilities is not None:
        probabilities = spread(index.probabilities, own)
    return FrameIndex(
        *(index.time_base, index.start_time, pts, index.crc32[shown], index.error[shown]),
        index.keyframes,
        probabilities,
        held,
    )


def _ticks(
    periods: npt.NDArray[np.int64], time_base: Fraction, rate: Fraction
) -> npt.NDArray[np.int64]:
    """`periods` frame periods of `rate` (each 0 or more), in ticks of time_base, exact, then
    rounded to the nearest tick, half a tick up.

    Kept as built by design on 2026-10-28 (the questions of 2026-10-12; DESIGN.md, Input, the frame
    rate, has the frame before the hole held and gives a filled place no time): ffmpeg's own
    rounding of a timestamp to a time base, to the nearest and halfway away from zero (av_rescale_q,
    AV_ROUND_NEAR_INF: libavutil/mathematics.c:142-144, mathematics.h:135 at n9.0.2), which its
    muxers' packets get (fftools/ffmpeg_mux.c:158, libavcodec/packet.c:538-546): the pts the missing
    frame would have in a file ffmpeg wrote at that rate, were the frame before it on its own tick.
    Counted from the frame held, not from the first frame: the place's time then follows the frames
    around it, as a player holding that frame has it, whatever the file's rounding or drift since
    its start. At 24000/1001 in Matroska's milliseconds, the places after a frame lie 42, 83,
    125 ms after it, and the twelfth 501, from 500.5."""
    # A period is `numerator` / `denominator` ticks: (d / n) / (p / q).
    numerator = time_base.denominator * rate.denominator
    denominator = time_base.numerator * rate.numerator
    counts: npt.NDArray[Any] = periods
    if 2 * int(periods.max(initial=0)) * numerator + denominator >= INT64_SAFE:
        counts = periods.astype(object)
    return ((2 * counts * numerator + denominator) // (2 * denominator)).astype(np.int64)


def spread(
    probabilities: npt.NDArray[np.float32], placed: npt.NDArray[np.int64]
) -> npt.NDArray[np.float32]:
    """The decoded frames' shot probabilities over the places, (places,) float32, given each
    frame's own place (`placed`, strictly increasing from 0): the cuts drawn from them
    (runtime/cuts.py: detected, possible) are those drawn from the decoded frames', each at its
    frame's own place, with the same probability.

    TransNetV2 saw the decoded frames alone. A frame's probability speaks of the change from it
    to the next frame it saw, and a cut is the frame after a run's peak (runtime/cuts.py). So,
    of the places that show frame k, its own and the held ones after it: the last takes k's
    probability, the change to frame k + 1 coming after it; the others take the lower of k's
    and of frame k - 1's (0 before the first frame). A frame followed by no hole keeps its own.

    Why the cuts stay: a run of frames reaching a threshold is one run of places, from the last
    place of its first frame, the lower value being under the threshold on that frame's other
    places (the frame before it is), and at or above it on every later frame's; the run's first
    highest frame keeps its value on its last place alone, its other places lower (the frame
    before it is lower, or under the threshold), so the peak is that place and the cut the place
    after it, the next frame's own: no shot starts on a held copy of the shot before's last
    frame. Held to it on random probabilities, holes and thresholds (tests/test_grid.py).

    The frame before's probability alone on those other places gives the same cuts (measured:
    that test passed with it, 2026-10-10); the lower of the two also keeps every place at or
    under the frame it shows, so that a run's places are its own frames' and no others'.

    The decoded frames' own probabilities are read back from the last place that shows each
    (FrameIndex.decoded), bit for bit: what is compared with measurement's arrays."""
    if len(probabilities) != len(placed):
        raise ValueError(f"{len(probabilities)} shot probabilities for {len(placed)} frames")
    values = probabilities.astype(np.float32)
    if not len(placed):
        return values
    total = int(placed[-1]) + 1
    lower = np.minimum(values, np.concatenate((np.zeros(1, np.float32), values[:-1])))
    ends = np.append(placed[1:], total)  # the place after the last that shows each frame
    over = np.repeat(lower, ends - placed)
    over[ends - 1] = values
    return over


@dataclass(frozen=True)
class Hole:
    """A run of places left empty, each a held row of the index: `first`, its first place, the
    number of the first missing frame as a user counts frames; `length`, its places, in frames;
    `frame`, the decoded frame held through it, its number among the decoded frames (the rows
    of FrameIndex.decoded)."""

    first: int
    length: int
    frame: int

    @property
    def held(self) -> int:
        """The place of the frame held through the hole, the one before it: that frame's number
        as a user counts frames."""
        return self.first - 1


@dataclass(frozen=True)
class Filling:
    """What was filled in a frame index (filling), for the messages and the record."""

    places: int  # the index's rows: the frames everything after the first pass counts
    filled: int  # the places left empty among them, each showing the frame before it again
    holes: int  # the runs of such places
    first: Hole | None  # the first hole, None without one
    longest: int  # the longest hole's places, in frames; 0 without one

    @property
    def decoded(self) -> int:
        """The frames ffmpeg decodes: the places that aren't filled."""
        return self.places - self.filled

    @property
    def share(self) -> Fraction:
        """The share of the places that are filled, exact, 0 to under 1; 0 without a place."""
        return Fraction(self.filled, self.places) if self.places else Fraction(0)


def holes(index: FrameIndex) -> list[Hole]:
    """The holes of a frame index, in order: each run of held rows. None without a held row."""
    if index.held is None:
        return []
    edges = np.diff(np.concatenate(([0], index.held.astype(np.int8), [0])))
    firsts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    shown = index.shown
    return [
        Hole(int(first), int(end - first), int(shown[first]))
        for first, end in zip(firsts, ends, strict=True)
    ]


def filling(index: FrameIndex) -> Filling:
    """What was filled in a frame index: its places, the filled ones, the holes, the first and
    the longest's length."""
    found = holes(index)
    return Filling(
        index.frames,
        sum(hole.length for hole in found),
        len(found),
        found[0] if found else None,
        max((hole.length for hole in found), default=0),
    )

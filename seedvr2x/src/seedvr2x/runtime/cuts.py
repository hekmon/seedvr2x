"""A source's cuts (DESIGN.md, Shot detection): where its shots start, detected from the shot
detector's per-frame probabilities, the first pass's record (media/index.py), at a threshold, so
that another threshold gives its cuts without a second decode or detection; and the cut list
--plan writes, the cuts and the possible ones, for editing and --cuts (DESIGN.md, Input: the cut
list format, read by job.read_cuts).

A detection is a run of frames whose single-frame probability reaches the threshold, one per run,
at its peak. The cut is the frame after the peak: TransNetV2 marks the outgoing shot's last frame
(offset -1 on 98% of measurement's sure cuts). Every detection starts a shot, however short: no
burst filter and no minimum shot length, a short shot being better run alone than merged into its
neighbour, from 1 frame on, and real cuts coming 1-3 frames apart in action anime."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from fractions import Fraction

import numpy as np
import numpy.typing as npt

from seedvr2x.media.index import NOPTS, FrameIndex

# DESIGN.md, Shot detection: 0.3 by default, confirmed by the user's second round of labels; below
# it each cut found costs 6 to 14 false cuts on animation, 14 to 21 on live action
# (research/docs/scene-detection.md, The threshold below 0.3).
THRESHOLD = 0.3
# DESIGN.md, Shot detection: the possible cuts are the runs peaking from 0.1 up to the threshold,
# about 250 an hour of animation at 0.3, one real cut in 7 to 18 of them; most misses score under
# 0.1, beyond any threshold (77 to 81 of animation's 92 an hour).
POSSIBLE = 0.1


@dataclass(frozen=True)
class Cut:
    """A cut: `frame`, the first frame of the shot it starts, counted from 0 in the source; and
    `probability`, the peak of the run that detected it, on the frame before (float32 in [0, 1],
    as recorded), None for a cut list's."""

    frame: int
    probability: float | None = None


def detections(probabilities: npt.NDArray[np.floating], threshold: float) -> list[int]:
    """The detections at threshold, each the frame of its peak: one per run of frames whose
    probability reaches the threshold, at the first of its highest frames, as measurement detects
    (research/scripts/scd_scores.py's tnet_detections).

    probabilities: (frames,), each frame's, float32 as the first pass records them (others are
    rounded to it); threshold in (0, 1]. Compared exactly, each float32 value against the
    threshold as Python reads it, a double (_exact): "reaches" taken at its word, and
    measurement's own comparison, whose analysis reads its recorded float32 values as float64
    before it detects (scd_scores.py's load_tnet and tnet_align). So float32's 0.7, 0.699999988,
    doesn't reach 0.7, where float32's 0.3, 0.300000012, reaches 0.3. ValueError on a NaN, which
    no sigmoid gives: a seedvr2x bug."""
    return [peak for _, _, peak in _runs(_exact(probabilities), threshold)]


def detected(probabilities: npt.NDArray[np.floating], threshold: float) -> list[Cut]:
    """The cuts detected at threshold (detections): the frame after each peak, with the peak's
    probability. A peak on the last frame gives none: no frame comes after it to start a shot."""
    values = _exact(probabilities)
    return [
        Cut(peak + 1, float(values[peak]))
        for _, _, peak in _runs(values, threshold)
        if peak + 1 < len(values)
    ]


def possible(probabilities: npt.NDArray[np.floating], threshold: float) -> list[Cut]:
    """The possible cuts at threshold, which --plan's cut list gives commented, to be checked: the
    cut of each run of frames from POSSIBLE up whose peak stays under the threshold, the frame
    after its peak, with the peak's probability; none on the last frame, as detected. Never a
    detection's: a run peaking at the threshold or above holds one. None when the threshold is
    POSSIBLE or under. Both bounds compared exactly, as detections compares."""
    values = _exact(probabilities)
    return [
        Cut(peak + 1, float(values[peak]))
        for _, _, peak in _runs(values, POSSIBLE)
        if values[peak] < threshold and peak + 1 < len(values)
    ]


def _exact(probabilities: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    """The probabilities as recorded, float32, each widened to the double of its exact value, so
    that a comparison with a threshold, a double, is exact. Compared in float32, the threshold
    was rounded to float32 first: a probability of float32's 0.7, 0.699999988, then reached 0.7,
    as it does for 0.35, 0.45, 0.65 and 0.9, where measurement's analysis, in float64, finds no
    detection; and a threshold of 1e-50 became 0 (a review's finding, 2026-10-10)."""
    values = np.asarray(probabilities, dtype=np.float32).astype(np.float64)
    nan = np.flatnonzero(np.isnan(values))
    if len(nan):
        raise ValueError(f"a NaN shot probability at frame {int(nan[0])}: a seedvr2x bug")
    return values


def _runs(values: npt.NDArray[np.float64], threshold: float) -> list[tuple[int, int, int]]:
    """Each run of frames whose probability (_exact) reaches threshold, in order: (first, end,
    peak), its frames [first, end), its peak the first of its highest frames (np.argmax)."""
    reached = np.concatenate(([0], (values >= threshold).astype(np.int8), [0]))
    edges = np.diff(reached)
    firsts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [
        (int(first), int(end), int(first) + int(np.argmax(values[first:end])))
        for first, end in zip(firsts, ends, strict=True)
    ]


def time_of(index: FrameIndex, frame: int) -> Fraction | None:
    """Frame `frame`'s time in seconds, as a player shows it, from its timestamp in the frame
    index: from the file's start, the start time ffprobe reads, which mpv takes for 0:00
    (--rebase-start-time, on by default); from the first frame's when the file declares none.
    None for a frame without a timestamp (index.NOPTS, which ffmpeg n9.0.2 never gives); from
    the stream's own zero when the file declares no start and its first frame has none either,
    as rate.first_time counts.

    PROVISIONAL (implementation, 2026-10-09; DESIGN.md asks for the timecode a player jumps to):
    the file's start, not the first frame's, so that a source whose video starts after its audio
    (by 23 ms, say, on a fixture of S9's mechanism 7: research/docs/seeking.md) gets the times a
    player shows; from the first frame, every time would be early by that much, more than half a
    frame at 24 fps, and the jump would land on the outgoing shot's last frame."""
    pts = int(index.pts[frame])
    if pts == NOPTS:
        return None
    if index.start_time is not None:
        origin = Fraction(index.start_time, 1_000_000)
    else:
        first = int(index.pts[0])
        # NOPTS is the lowest int64: as an origin, it gave "102481911520608:37:12.400".
        origin = 0 if first == NOPTS else first * index.time_base
    return pts * index.time_base - origin


def timecode(seconds: Fraction | None) -> str:
    """H:MM:SS.mmm, rounded up to the millisecond; "-" without a time.

    PROVISIONAL (implementation, 2026-10-10, a question for design and the user: which player
    checks the cuts). Rounded up, the time lies in its frame's own display, at its start or
    within a millisecond after: a player showing the frame displayed at that time shows that
    frame, where a time rounded down, before the frame's own, would show the one before. A seek
    to the first frame at or after the time lands one frame late instead, wherever the frame's
    time isn't a whole millisecond: ffmpeg -ss T -i FILE gave the next frame for 45 of the 48
    frames of a 24000/1001 MP4 (time base 1/24000), and the frame itself for all 48 with the time
    rounded down; on Matroska, whose times are milliseconds, the frame itself either way
    (measured 2026-10-10, n9.0.2). mpv's seek to an absolute time is precise by default
    (DOCS/man/options.rst:337-351 at v0.41.0, 41f6a64), and a precise seek skips the frames more
    than 5 ms before its target, showing the first one left (player/video.c:471, 521-525): by
    its code, the frame itself either way; not run here."""
    if seconds is None:
        return "-"
    milliseconds = math.ceil(seconds * 1000)
    sign, value = ("-" if milliseconds < 0 else ""), abs(milliseconds)
    hours, minutes = value // 3_600_000, value // 60_000 % 60
    return f"{sign}{hours}:{minutes:02d}:{value // 1000 % 60:02d}.{value % 1000:03d}"


def probability(value: float) -> str:
    """A probability to 3 decimals: the float32 value's exact decimal, cut, never rounded up, so
    that a possible cut never shows at the threshold it stays under, nor a cut under the one it
    reaches (a threshold of 3 decimals or fewer), as they are compared (_exact): 0.29999998 is
    0.299; float32's 0.7, 0.699999988, which doesn't reach 0.7, is 0.699; float32's 0.3,
    0.300000012, which reaches 0.3, is 0.300. PROVISIONAL (implementation, 2026-10-09; DESIGN.md
    asks for "its probability"): a peak scoring 0.9939 shows 0.993, where a rounded figure would
    say 0.994."""
    exact = Decimal(float(np.float32(value)))
    return str(exact.quantize(Decimal("0.001"), rounding=ROUND_DOWN))


def cut_list(
    header: Sequence[str], index: FrameIndex, cuts: Sequence[Cut], possible: Sequence[Cut] = ()
) -> str:
    """The cut list --plan writes (DESIGN.md, Shot detection), in the cut list format, so that
    --cuts reads it back (job.read_cuts: a frame number per line, # comments, the fields after the
    number ignored): the header's lines as comments, then a line per cut and per possible cut, in
    frame order: a cut `<frame> <probability> <time>`, a possible cut commented, `# <frame>
    <probability> <time>`, so that keeping one is uncommenting its line; a cut list's own cut
    `<frame> <time>`. The probability is its run's peak, on the frame before (probability); the
    time the cut frame's, as a player shows it (time_of, timecode), a jump to it in a player
    checking the cut.

    The header's lines hold names, the source's and a cut list's: whatever in them isn't
    printable is written escaped (a line break, which would start a line of the name's own, read
    back as a cut when it begins with a number: job.read_cuts splits as str.splitlines does). No
    header line may begin with a digit: uncommenting every "# <digit>" line, as an editor's
    search does to keep every possible cut, would make a cut of it (ValueError: a seedvr2x
    bug)."""
    lines: list[str] = []
    for line in header:
        escaped = "".join(
            char if char.isprintable() else char.encode("unicode_escape").decode() for char in line
        )
        if escaped[:1].isdigit():
            raise ValueError(f"a cut list's header line begins with a digit: {escaped!r}")
        lines.append(f"# {escaped}" if escaped else "#")
    found = [(cut, False) for cut in cuts] + [(cut, True) for cut in possible]
    for cut, commented in sorted(found, key=lambda item: item[0].frame):
        fields = [str(cut.frame)]
        if cut.probability is not None:
            fields.append(probability(cut.probability))
        fields.append(timecode(time_of(index, cut.frame)))
        lines.append(("# " if commented else "") + " ".join(fields))
    return "\n".join(lines) + "\n"

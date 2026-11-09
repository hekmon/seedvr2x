"""A source's cuts drawn from the shot detector's probabilities (DESIGN.md, Shot detection): where
its shots start, at a threshold, so that another threshold gives its cuts without a second decode
or detection. A brick of the library, plain Python and NumPy: the CLI's `cuts.py` keeps the cut
list, the times from the frame index and the playlists, and any app draws its cuts here with the
same code, and writes a cut's time in the same form (timecode: the CLI's cut list and playlists,
the ComfyUI nodes' cut list).

A detection is a run of frames whose single-frame probability reaches the threshold, one per run,
at its peak. The cut is the frame after the peak: TransNetV2 marks the outgoing shot's last frame
(offset -1 on 98% of measurement's sure cuts). Every detection starts a shot, however short: no
burst filter and no minimum shot length, a short shot being better run alone than merged into its
neighbour, from 1 frame on, and real cuts coming 1-3 frames apart in action anime."""

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import numpy.typing as npt

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


def timecode(seconds: Fraction | None) -> str:
    """H:MM:SS.mmm, rounded up to the millisecond; "-" without a time.

    DESIGN.md, Shot detection: the cut's "time from the file's start, rounded up to the
    millisecond", for the player that checks the cuts: "The player is VLC, the user's choice
    (2026-10-10)", which "lands within 3 frames of the cut on a file with B-frames, mostly before
    it, which a time rounded up serves better than one rounded down; mpv lands on the cut frame".
    Rounded up, the time lies in its frame's own display, at its start or within a millisecond
    after. A seek to the first frame at or after the time lands one frame late instead, wherever
    the frame's time isn't a whole millisecond: ffmpeg -ss T -i FILE gave the next frame for 45
    of the 48 frames of a 24000/1001 MP4 (time base 1/24000), and the frame itself for all 48
    with the time rounded down; on Matroska, whose times are milliseconds, the frame itself
    either way (measured 2026-10-10, n9.0.2)."""
    if seconds is None:
        return "-"
    milliseconds = math.ceil(seconds * 1000)
    sign, value = ("-" if milliseconds < 0 else ""), abs(milliseconds)
    hours, minutes = value // 3_600_000, value // 60_000 % 60
    return f"{sign}{hours}:{minutes:02d}:{value // 1000 % 60:02d}.{value % 1000:03d}"


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

"""The cuts drawn from the shot detector's probabilities (seedvr2x/shotcuts.py; DESIGN.md, Shot
detection): the detections at a threshold, one per run of frames reaching it, at its peak, the
first of its highest frames, held to measurement's tnet_detections; the cut the frame after the
peak, none after the last frame; the possible cuts, runs from 0.1 up peaking under the threshold.
A time's form (timecode) is held here; the cut list, the playlists and the times they hold, from
the frame index, are the CLI's, tests/test_cuts.py, and the ComfyUI nodes' cut list's,
apps/comfyui/tests/test_cuts_node.py."""

from fractions import Fraction

import numpy as np
import numpy.typing as npt
import pytest

from seedvr2x.shotcuts import (
    POSSIBLE,
    THRESHOLD,
    Cut,
    detected,
    detections,
    possible,
    timecode,
)


# measurement's detection, research/scripts/scd_scores.py's tnet_detections (lines 886-890), the
# reference seedvr2x's is held to: one detection per run of frames >= p, at the run's highest frame
# (the first if tied). Its analysis gives it the recorded float32 values read as float64
# (load_tnet's astype, line 1231; tnet_align's array): the comparison with p, a Python float, is
# then exact, as it is here on the values widened (measured).
def measured(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
    return x.astype(np.float64)


def tnet_detections(x: npt.NDArray[np.float64], p: float) -> list[int]:
    assert x.dtype == np.float64
    m = np.concatenate(([False], np.nan_to_num(x, nan=-1.0) >= p, [False]))
    d = np.diff(m.astype(np.int8))
    starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    return [int(a + np.argmax(x[a:b])) for a, b in zip(starts, ends, strict=True)]


def f32(*values: float | np.floating) -> npt.NDArray[np.float32]:
    return np.array(values, np.float32)


def frames(cuts: list[Cut]) -> list[int]:
    return [cut.frame for cut in cuts]


def test_the_defaults() -> None:
    # DESIGN.md, Shot detection: 0.3 by default, the possible cuts from 0.1 up.
    assert (THRESHOLD, POSSIBLE) == (0.3, 0.1)


def test_one_detection_per_run_at_its_peak() -> None:
    # Runs at both edges, a tie (the first of the highest), one frame long, two runs one frame
    # apart: each a detection at its peak; the cut the frame after it, none after the last frame.
    x = f32(0.9, 0.5, 0.1, 0.4, 0.6, 0.6, 0.2, 0.35, 0.29, 0.5, 0.05, 0.7)
    assert detections(x, 0.3) == [0, 4, 7, 9, 11]
    assert detections(x, 0.3) == tnet_detections(measured(x), 0.3)
    found = detected(x, 0.3)
    assert frames(found) == [1, 5, 8, 10]
    assert [cut.probability for cut in found] == [float(x[k]) for k in (0, 4, 7, 9)]
    # Two runs merged into one when the frame between them reaches the threshold too, detected at
    # the higher peak, 9's 0.5.
    assert detections(x, 0.28) == [0, 4, 9, 11]
    # Every frame reaching it: one run, one detection.
    assert frames(detected(f32(0.4, 0.8, 0.5), 0.3)) == [2]
    # None reaching it: no cut, the source one shot.
    assert detected(f32(0.1, 0.2), 0.3) == [] and detected(f32(), 0.3) == []


def test_the_threshold_reached_exactly() -> None:
    # Compared exactly, the float32 probability against the threshold as Python reads it, as
    # measurement's analysis compares (float64): the float32 nearest a threshold reaches it when
    # it is the threshold or above, 0.3's (0.300000012) and 1's, and not when it is under,
    # 0.7's (0.699999988), 0.35's, 0.45's, 0.65's and 0.9's, where the next float32 up does. In
    # float32, the threshold rounded too, each of them reached its own.
    for p, reached in (
        *((0.3, True), (1.0, True), (0.1, True), (0.5, True)),
        *((0.7, False), (0.35, False), (0.45, False), (0.65, False), (0.9, False)),
    ):
        at = np.float32(p)
        assert (float(at) >= p) == reached
        under, over = np.nextafter(at, np.float32(0)), np.nextafter(at, np.float32(2))
        x = f32(0.0, at, 0.0, under, 0.0, over, 0.0)
        expected = [*([1] if reached else []), 5]
        assert detections(x, p) == tnet_detections(measured(x), p) == expected
        assert frames(detected(x, p)) == [frame + 1 for frame in expected]
    # A threshold under float32's smallest values is no threshold of 0, which every frame
    # reaches, one run: two runs, of the frames above it.
    x = f32(0.0, 0.0, 1e-30, 0.0, 1e-30)
    assert detections(x, 1e-50) == tnet_detections(measured(x), 1e-50) == [2, 4]
    # The possible cuts' bounds likewise: float32's 0.1, 0.100000001, reaches 0.1; float32's 0.7
    # stays under a threshold of 0.7, a possible cut there, never a detection.
    x = f32(0.0, np.float32(0.1), 0.0, np.nextafter(np.float32(0.1), np.float32(0)), 0.0)
    assert frames(possible(x, 0.3)) == [2]
    x = f32(0.0, np.float32(0.7), 0.0)
    assert frames(possible(x, 0.7)) == [2] and detected(x, 0.7) == []


def test_as_measurement_detects() -> None:
    # On random probabilities, plateaus included (values rounded to a few levels), at every
    # threshold: measurement's detections, and the cuts the frames after them.
    generator = np.random.default_rng(7)
    for count in (1, 2, 50, 1000):
        for rounding in (4, 20, 0):
            x = generator.random(count, np.float32)
            if rounding:
                x = (np.round(x * rounding) / rounding).astype(np.float32)
            for p in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7, 1.0):
                found = detections(x, p)
                assert found == tnet_detections(measured(x), p)
                assert frames(detected(x, p)) == [k + 1 for k in found if k + 1 < count]


def test_a_nan_refused() -> None:
    # No sigmoid gives one, and the frame index refuses one: a bug, never a cut or none.
    x = f32(0.1, 0.5, 0.2)
    x[2] = np.nan
    for find in (detections, detected, possible):
        with pytest.raises(ValueError, match="a NaN shot probability at frame 2: a seedvr2x bug"):
            find(x, 0.3)


def test_possible_cuts() -> None:
    # Runs from 0.1 up peaking under the threshold, their cut the frame after the peak: a run
    # peaking at 0.25, one at 0.29999998, float32's under 0.3; never a detection's run, nor one
    # holding two; none after the last frame.
    x = f32(0.05, 0.15, 0.25, 0.12, 0.0, 0.5, 0.2, 0.0, 0.29999998, 0.0, 0.5, 0.2, 0.5, 0.0, 0.2)
    assert frames(detected(x, 0.3)) == [6, 11, 13]
    found = possible(x, 0.3)
    assert frames(found) == [3, 9]
    assert [cut.probability for cut in found] == [float(x[2]), float(x[8])]
    # At 0.5: the runs peaking at 0.25 and 0.29999998 still, none of 0.5's; at 0.26, the one
    # peaking at 0.25 alone, the other a detection.
    assert frames(possible(x, 0.5)) == [3, 9]
    assert frames(possible(x, 0.26)) == [3] and frames(detected(x, 0.26)) == [6, 9, 11, 13]
    # At 0.1 or under: no run from 0.1 up peaks under it.
    assert possible(x, POSSIBLE) == [] and possible(x, 0.05) == []
    # Each run from 0.1 up is one of measurement's detections at 0.1, at its peak: the possible
    # cuts are those peaking under the threshold, never a cut at it.
    generator = np.random.default_rng(11)
    x = (generator.random(2000, np.float32) ** 6).astype(np.float32)
    for p in (0.15, 0.2, 0.3, 0.5):
        runs = tnet_detections(measured(x), POSSIBLE)
        expected = [k + 1 for k in runs if float(x[k]) < p and k + 1 < len(x)]
        assert frames(possible(x, p)) == expected and expected
        assert not set(expected) & set(frames(detected(x, p)))


def test_a_time_is_h_mm_ss_mmm_rounded_up_to_the_millisecond() -> None:
    # Rounded up, so that a jump lands in the frame's own display: 24000/1001's frame 1, 41.708 ms,
    # is 0:00:00.042; a whole millisecond stays as it is; hours not padded, minutes and seconds
    # to two digits.
    assert timecode(Fraction(1001, 24000)) == "0:00:00.042"
    assert timecode(Fraction(36, 25)) == "0:00:01.440"
    assert timecode(Fraction(0)) == "0:00:00.000"
    assert timecode(Fraction(40_001, 1_000_000)) == "0:00:00.041"
    assert timecode(Fraction(172_800 * 1001, 24000)) == "2:00:07.200"
    assert timecode(Fraction(100 * 3600 + 59 * 60 + 59)) == "100:59:59.000"
    # Before the start, signed; no time, "-".
    assert timecode(Fraction(-1, 100)) == "-0:00:00.010"
    assert timecode(None) == "-"

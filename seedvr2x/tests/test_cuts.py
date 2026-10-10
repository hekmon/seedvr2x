"""A source's cuts (runtime/cuts.py; DESIGN.md, Shot detection): the detections at a threshold, one
per run of frames reaching it, at its peak, the first of its highest frames, held to measurement's
tnet_detections; the cut the frame after the peak, none after the last frame; the possible cuts,
runs from 0.1 up peaking under the threshold; the cut list --plan writes, its lines, its times as a
player shows them, and its round trip through --cuts' reader (job.read_cuts)."""

import re
from fractions import Fraction
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from seedvr2x.media.index import NOPTS, FrameIndex
from seedvr2x.runtime.cuts import (
    POSSIBLE,
    THRESHOLD,
    Cut,
    cut_list,
    detected,
    detections,
    possible,
    probability,
    time_of,
    timecode,
)
from seedvr2x.runtime.job import read_cuts


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
        for levels in (4, 20, 0):
            x = generator.random(count, np.float32)
            if levels:
                x = (np.round(x * levels) / levels).astype(np.float32)
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


def index_of(
    pts: list[int], time_base: Fraction = Fraction(1, 25), start_time: int | None = 0
) -> FrameIndex:
    """A frame index of these timestamps, the rest left empty."""
    count = len(pts)
    return FrameIndex(
        time_base,
        start_time,
        np.array(pts, np.int64),
        np.zeros(count, np.uint32),
        np.zeros(count, np.bool_),
        np.array(pts[:1], np.int64),
    )


def test_times_as_a_player_shows_them() -> None:
    # From the file's start (its start time, in microseconds), rounded up to the millisecond so
    # that a jump lands on the frame, never the one before: 24000/1001's frame 1 at 41.708 ms is
    # 0:00:00.042, frame 24 at 1.001 s exactly; two hours on, 2:00:07.200.
    ntsc = index_of([0, 1001, 24 * 1001, 172_800 * 1001], Fraction(1, 24000))
    assert [timecode(time_of(ntsc, k)) for k in range(4)] == [
        "0:00:00.000",
        "0:00:00.042",
        "0:00:01.001",
        "2:00:07.200",
    ]
    # A video starting 23 ms after the file, its audio first (a millisecond time base, as
    # Matroska's): the times a player shows, the first frame at 0:00:00.023.
    late = index_of([23, 65, 106], Fraction(1, 1000), start_time=0)
    assert [timecode(time_of(late, k)) for k in range(3)] == [
        "0:00:00.023",
        "0:00:00.065",
        "0:00:00.106",
    ]
    # A transport stream's start, 1.4 s: 0:00 there.
    stream = index_of([126_000, 129_600], Fraction(1, 90000), start_time=1_400_000)
    assert [timecode(time_of(stream, k)) for k in range(2)] == ["0:00:00.000", "0:00:00.040"]
    # No start time declared: from the first frame's.
    unstarted = index_of([3600, 7200], Fraction(1, 90000), start_time=None)
    assert [timecode(time_of(unstarted, k)) for k in range(2)] == ["0:00:00.000", "0:00:00.040"]
    # Before the start; between milliseconds, rounded up; no timestamp.
    assert timecode(Fraction(-1, 100)) == "-0:00:00.010"
    assert timecode(Fraction(40_001, 1_000_000)) == "0:00:00.041"
    assert timecode(Fraction(1, 25)) == "0:00:00.040"
    assert time_of(index_of([0, NOPTS]), 1) is None and timecode(None) == "-"
    # No start time declared and a first frame without a timestamp: from the stream's own zero,
    # never from the lowest int64 that stands for none ("102481911520608:37:12.400").
    untimed = index_of([NOPTS, 3600, 7200], Fraction(1, 90000), start_time=None)
    assert [timecode(time_of(untimed, k)) for k in (1, 2)] == ["0:00:00.040", "0:00:00.080"]


def test_probabilities_to_three_decimals() -> None:
    # The float32 value's exact decimal, cut: under a threshold of 3 decimals stays under it,
    # reaching it shows it, as they are compared: float32's 0.7, 0.699999988, doesn't reach 0.7.
    assert probability(float(np.float32(0.29999998))) == "0.299"
    assert probability(float(np.float32(0.3))) == "0.300"
    assert probability(float(np.float32(0.7))) == "0.699"
    assert probability(float(np.nextafter(np.float32(0.7), np.float32(1)))) == "0.700"
    # float32's 0.954 is 0.953999996: under a threshold of 0.954, and shown so.
    assert probability(float(np.float32(0.954))) == "0.953"
    assert probability(float(np.float32(0.9636))) == "0.963"
    assert probability(1.0) == "1.000" and probability(float(np.float32(1e-5))) == "0.000"


def test_cut_list_read_back(tmp_path: Path) -> None:
    # The cuts and the possible ones in frame order, a possible cut commented, each with its
    # peak's probability and its time; the header's lines comments. --cuts reads the cuts back;
    # uncommented, the possible ones are cuts too.
    x = f32(0.05, 0.15, 0.25, 0.12, 0.0, 0.5, 0.2, 0.0, 0.29999998, 0.0)
    index = index_of(list(range(10)))
    text = cut_list(["the header", "", "said"], index, detected(x, 0.3), possible(x, 0.3))
    assert text.splitlines() == [
        "# the header",
        "#",
        "# said",
        "# 3 0.250 0:00:00.120",
        "6 0.500 0:00:00.240",
        "# 9 0.299 0:00:00.360",
    ]
    assert text.endswith("\n")
    path = tmp_path / "plan.txt"
    path.write_text(text)
    assert read_cuts(path) == [6]
    path.write_text(text.replace("# 3 ", "3 ").replace("# 9 ", "9 "))
    assert read_cuts(path) == [3, 6, 9]
    # A cut list's own cuts: no probability.
    listed = cut_list([], index, [Cut(4), Cut(7)])
    assert listed == "4 0:00:00.160\n7 0:00:00.280\n"
    path.write_text(listed)
    assert read_cuts(path) == [4, 7]


def test_header_lines_stay_comments(tmp_path: Path) -> None:
    # A name in the header holding what ends a line (as read_cuts splits them: str.splitlines)
    # is written escaped: raw, "a\n7 b.mkv" gave a second line read back as a cut at frame 7.
    # Whatever is printable stays, accents and all.
    x = f32(0.05, 0.15, 0.25, 0.12, 0.0, 0.5, 0.2, 0.0, 0.29999998, 0.0)
    index = index_of(list(range(10)))
    names = "of a\n7 b.mkv, c\r8, d\x0b9, e\x1c1, f\x852, g\u20283, h\t4 and été.mkv"
    text = cut_list([names, "", "Found: 1 cut"], index, detected(x, 0.3), possible(x, 0.3))
    assert text.splitlines() == [
        "# of a\\n7 b.mkv, c\\r8, d\\x0b9, e\\x1c1, f\\x852, g\\u20283, h\\t4 and été.mkv",
        "#",
        "# Found: 1 cut",
        "# 3 0.250 0:00:00.120",
        "6 0.500 0:00:00.240",
        "# 9 0.299 0:00:00.360",
    ]
    path = tmp_path / "plan.txt"
    path.write_text(text)
    assert read_cuts(path) == [6]
    # Every "# <digit>" line uncommented, as a search over the file keeps every possible cut:
    # the cuts and the possible cuts, and nothing of the header's.
    path.write_text(re.sub(r"(?m)^# (?=\d)", "", text))
    assert read_cuts(path) == [3, 6, 9]
    # So no header line may begin with a digit: a seedvr2x bug, refused.
    for line in ("61 cuts at 0.3 (62 shots)", "0; the time"):
        with pytest.raises(ValueError, match="a cut list's header line begins with a digit"):
            cut_list(["fine", line], index, [])

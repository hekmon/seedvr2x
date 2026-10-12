"""A source's cuts (runtime/cuts.py; DESIGN.md, Shot detection): the detections at a threshold, one
per run of frames reaching it, at its peak, the first of its highest frames, held to measurement's
tnet_detections; the cut the frame after the peak, none after the last frame; the possible cuts,
runs from 0.1 up peaking under the threshold; the cut list --plan writes, its lines, its times as a
player shows them, and its round trip through --cuts' reader (job.read_cuts); the playlist of the
possible cuts written beside it for VLC, its entries, its name and what is said of it."""

import errno
import itertools
import logging
import os
import re
from fractions import Fraction
from pathlib import Path
from urllib.parse import unquote_to_bytes

import numpy as np
import numpy.typing as npt
import pytest

from seedvr2x.media import files
from seedvr2x.media.grid import Filling, Hole, filled
from seedvr2x.media.index import NOPTS, FrameIndex
from seedvr2x.runtime import cuts
from seedvr2x.runtime.cuts import (
    POSSIBLE,
    THRESHOLD,
    Cut,
    cut_list,
    detected,
    detections,
    places_note,
    playlist,
    playlist_path,
    possible,
    probability,
    time_of,
    timecode,
    write_playlist,
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


def test_times_of_filled_places() -> None:
    # In an index in places (media/grid.py: missing frames filled), a frame's number is its
    # place and its time its row's: a decoded frame's own timestamp, and for a filled place,
    # which shows the frame before it again, the place's own time, that frame's plus the frame
    # periods since, to the tick. Frames 0, 1, 2, 5 and 6 of 24000/1001 in milliseconds: places
    # 3 and 4 lie 42 and 83 ms after frame 2's 83 ms. The cut list's times are the places'.
    first = index_of([0, 42, 83, 209, 250], Fraction(1, 1000))
    index = filled(first, [0, 1, 2, 5, 6], Fraction(24000, 1001))
    assert index.frames == 7
    assert [timecode(time_of(index, k)) for k in range(7)] == [
        "0:00:00.000",
        "0:00:00.042",
        "0:00:00.083",
        "0:00:00.125",
        "0:00:00.166",
        "0:00:00.209",
        "0:00:00.250",
    ]
    assert cut_list([], index, [Cut(3), Cut(5)]) == "3 0:00:00.125\n5 0:00:00.209\n"
    # From the first frame's time when the file declares no start: the first row is never held.
    late = filled(index_of([3600, 7200, 18000], Fraction(1, 90000), None), [0, 1, 4], Fraction(25))
    assert [timecode(time_of(late, k)) for k in (1, 2, 3, 4)] == [
        "0:00:00.040",
        "0:00:00.080",
        "0:00:00.120",
        "0:00:00.160",
    ]
    # A filled place after a frame without a timestamp has none either.
    untimed = filled(index_of([0, NOPTS, 200], Fraction(1, 1000)), [0, 1, 4], Fraction(25))
    assert [time_of(untimed, k) for k in (1, 2, 3)] == [None, None, None]
    assert time_of(untimed, 4) == Fraction(1, 5)


def test_places_said_in_the_header(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A cut list of a source with missing frames numbers its cuts by place (media/grid.py), and
    # its header says so, with what was filled: a list numbered by a tool counting the frames it
    # decodes is off by the holes before each cut. Nothing for a source without a hole.
    whole = index_of(list(range(27)))
    assert places_note(whole) == []
    assert cut_list([*places_note(whole)], whole, [Cut(7)]) == "7 0:00:00.280\n"
    kept = [k for k in range(27) if k not in (5, 6)]
    one = filled(index_of(kept), kept, Fraction(25))
    assert places_note(one) == [
        "Frame numbers count places, the missing frames' filled ones among them: 2 frames",
        "missing after frame 4, 27 places for 25 frames decoded. A list numbered by a tool that",
        "counts the decoded frames is off by the holes before each cut, its number lower by the",
        "frames missing before it.",
    ]
    kept = [k for k in range(3600) if k not in (5, 100, 101, 102, 2000)]
    several = filled(index_of(kept), kept, Fraction(25))
    assert places_note(several) == [
        "Frame numbers count places, the missing frames' filled ones among them: 5 frames",
        "missing in 3 holes, the first after frame 4, 3600 places for 3595 frames decoded. A",
        "list numbered by a tool that counts the decoded frames is off by the holes before each",
        "cut, its number lower by the frames missing before it.",
    ]
    kept = [k for k in range(27) if k != 9]
    single = " ".join(places_note(filled(index_of(kept), kept, Fraction(25))))
    assert ": 1 frame missing after frame 8, 27 places for 26 frames decoded. " in single
    path = tmp_path / "plan.txt"
    path.write_text(cut_list(["of in.mkv", *places_note(several)], several, [Cut(7), Cut(2001)]))
    assert read_cuts(path) == [7, 2001]
    path.write_text(re.sub(r"(?m)^# (?=\d)", "", path.read_text()))
    assert read_cuts(path) == [7, 2001]
    # No line of it begins with a digit, whatever its figures' lengths bring to the wrap: a
    # header line that does is refused, since uncommented with the possible cuts it would be a
    # cut (cut_list). Figures of 1 to 9 digits each, in place of the index's own: wrapped at any
    # space, 730 of these 6,561 notes had such a line.
    for widths in itertools.product(range(1, 10), repeat=4):
        missing, holes, held, places = (10 ** (width - 1) + 1 for width in widths)
        figures = Filling(missing + places, missing, holes, Hole(held + 1, 1, held), 1)
        monkeypatch.setattr(cuts, "filling", lambda index, figures=figures: figures)
        note = places_note(whole)
        assert all(len(line) <= 86 and not line[:1].isdigit() for line in note), note
        assert f": {missing} frames missing in {holes} holes, the first after" in " ".join(note)


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


# The comments a playlist begins with (playlist).
PLAYLIST_HEADER = [
    "#EXTM3U",
    "# seedvr2x: the possible cuts of the cut list beside this file, to check in VLC, which",
    "# plays an entry per possible cut, from 1 s before its time to 3 s after.",
    "# It names the source by its file name alone: it plays from the source's directory.",
]


def ntsc(count: int) -> FrameIndex:
    """A frame index of `count` frames at 24000/1001, as an MP4 file times them."""
    return index_of([k * 1001 for k in range(count)], Fraction(1, 24000))


def test_playlist_of_the_possible_cuts() -> None:
    # An entry per possible cut, in frame order: its title, the frame, time and probability of
    # its line in the cut list; VLC's start, the list's time less a second, never before 0, and
    # stop, 3 seconds after it, never past the last frame's time (200.159 s); the source's name.
    index = ntsc(4800)
    found = [Cut(4521, float(np.float32(0.213))), Cut(4790, 0.25), Cut(12, 0.15)]
    text = playlist(index, found, "in.mkv")
    assert text.splitlines() == [
        *PLAYLIST_HEADER,
        "#EXTINF:-1,possible cut 12 at 0:00:00.501 (0.150)",
        "#EXTVLCOPT:start-time=0.000",
        "#EXTVLCOPT:stop-time=3.501",
        "in.mkv",
        "#EXTINF:-1,possible cut 4521 at 0:03:08.564 (0.212)",
        "#EXTVLCOPT:start-time=187.564",
        "#EXTVLCOPT:stop-time=191.564",
        "in.mkv",
        "#EXTINF:-1,possible cut 4790 at 0:03:19.783 (0.250)",
        "#EXTVLCOPT:start-time=198.783",
        "#EXTVLCOPT:stop-time=200.159",
        "in.mkv",
    ]
    assert text.endswith("\n") and text.isascii()
    # The times are the cut list's own, its line's frame and probability with them.
    listed = cut_list([], index, [], found).splitlines()
    assert listed == [
        "# 12 0.150 0:00:00.501",
        "# 4521 0.212 0:03:08.564",
        "# 4790 0.250 0:03:19.783",
    ]
    # No comment begins with a tag VLC reads, whatever its case (m3u.c:250-289); no title holds
    # what VLC splits an artist from it by (m3u.c:388-417).
    tags = ("extinf:", "extvlcopt:", "playlist:", "extalbumarturl:", "ext-x-")
    assert not any(line.lstrip("# \t").lower().startswith(tags) for line in PLAYLIST_HEADER)
    titles = [line for line in text.splitlines() if line.startswith("#EXTINF:")]
    assert len(titles) == 3 and not any("," in t[11:] or " - " in t for t in titles)
    # Two hours on: seconds, never H:MM:SS, which VLC's options don't read.
    late = playlist(ntsc(172_900), [Cut(172_800, 0.2)], "in.mkv").splitlines()
    assert late[4:7] == [
        "#EXTINF:-1,possible cut 172800 at 2:00:07.200 (0.200)",
        "#EXTVLCOPT:start-time=7206.200",
        "#EXTVLCOPT:stop-time=7210.200",
    ]
    # A cut whose frame has no time has no entry; a stop at or before its start, which VLC
    # ignores, playing to the file's end, is left out: times before the file's start.
    untimed = index_of([0, 1, NOPTS, 3])
    assert playlist(untimed, [Cut(2, 0.2)], "in.mkv").splitlines() == PLAYLIST_HEADER
    early = index_of([0, 1, 2], start_time=1_000_000)
    assert playlist(early, [Cut(1, 0.2)], "in.mkv").splitlines() == [
        *PLAYLIST_HEADER,
        "#EXTINF:-1,possible cut 1 at -0:00:00.960 (0.200)",
        "#EXTVLCOPT:start-time=0.000",
        "in.mkv",
    ]
    # None: the comments alone.
    assert playlist(index, [], "in.mkv").splitlines() == PLAYLIST_HEADER


def test_playlist_names_the_source_as_a_url() -> None:
    # VLC reads an entry as a URL relative to the playlist: the name percent-encoded, every byte
    # but a letter, a digit or one of -._~, so that a #, a %, a leading space or an accent is
    # the name's own (measured in VLC 3.0.24: raw, "a#b.mkv" opened "a", "#start.mkv" was a
    # comment, "50%20off.mkv" opened "50 off.mkv" and " lead.mkv" "lead.mkv").
    index = ntsc(100)
    for name, written in (
        ("plain-1_2.v~3.mkv", "plain-1_2.v~3.mkv"),
        ("with space.mkv", "with%20space.mkv"),
        ("it's.mkv", "it%27s.mkv"),
        ("été ñ 日本語.mkv", "%C3%A9t%C3%A9%20%C3%B1%20%E6%97%A5%E6%9C%AC%E8%AA%9E.mkv"),
        ("a#b.mkv", "a%23b.mkv"),
        ("#start.mkv", "%23start.mkv"),
        ("100%.mkv", "100%25.mkv"),
        ("50%20off.mkv", "50%2520off.mkv"),
        (" lead.mkv", "%20lead.mkv"),
        (
            "[Group] Title - 01 (BD) [ABCD1234].mkv",
            "%5BGroup%5D%20Title%20-%2001%20%28BD%29%20%5BABCD1234%5D.mkv",
        ),
        ("a&b=c+d;e,f!g$h@i:j?k.mkv", "a%26b%3Dc%2Bd%3Be%2Cf%21g%24h%40i%3Aj%3Fk.mkv"),
        ("a\nb\r.mkv", "a%0Ab%0D.mkv"),
        # A name that isn't UTF-8, as Python reads it from Linux: its own bytes.
        ("caf\udce9.mkv", "caf%E9.mkv"),
    ):
        lines = playlist(index, [Cut(50, 0.2)], name).splitlines()
        assert lines[-1] == written and len(lines) == len(PLAYLIST_HEADER) + 4
        assert unquote_to_bytes(written) == os.fsencode(name)


def test_playlist_path() -> None:
    # Beside FILE, its name with .m3u8 added, whatever its suffix: never FILE itself.
    for plan, expected in (
        ("plan.txt", "plan.txt.m3u8"),
        ("x.cuts", "x.cuts.m3u8"),
        ("plan", "plan.m3u8"),
        ("plan.m3u8", "plan.m3u8.m3u8"),
        ("some/where/plan.txt", "some/where/plan.txt.m3u8"),
        ("/an/absolute/plan.txt", "/an/absolute/plan.txt.m3u8"),
        (".hidden", ".hidden.m3u8"),
    ):
        assert playlist_path(Path(plan)) == Path(expected) != Path(plan)
    # A FILE without a name is refused before any playlist (cli._plan_file): no error here.
    assert playlist_path(Path(".")) == Path(".m3u8")


def test_playlist_written_and_said(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Written beside FILE, whole, an earlier one replaced, and said with what to do with it:
    # beside the source, opened; elsewhere, moved there first.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    index = ntsc(4800)
    found = [Cut(4521, 0.2), Cut(12, 0.15)]
    source = tmp_path / "films" / "in 1.mkv"
    (tmp_path / "films").mkdir()
    (tmp_path / "plans").mkdir()
    beside, elsewhere = tmp_path / "films" / "plan.txt", tmp_path / "plans" / "plan.txt"
    (tmp_path / "films" / "plan.txt.m3u8").write_text("an earlier playlist")
    said = (
        "a playlist of the 2 possible cuts for VLC, an entry each, from 1 second before its cut"
        " to 3 seconds after"
    )
    assert write_playlist(beside, index, found, source)
    written = tmp_path / "films" / "plan.txt.m3u8"
    assert written.read_text() == playlist(index, found, "in 1.mkv")
    assert written.read_text().splitlines()[-1] == "in%201.mkv"
    assert sorted(path.name for path in written.parent.iterdir()) == ["plan.txt.m3u8"]
    assert [record.getMessage() for record in caplog.records] == [
        f"{written}: {said}; beside the source, which it names by its file name alone: open it"
        " in VLC"
    ]
    caplog.clear()
    assert write_playlist(elsewhere, index, found[:1], source)
    written = tmp_path / "plans" / "plan.txt.m3u8"
    assert written.read_text() == playlist(index, found[:1], "in 1.mkv")
    assert [record.getMessage() for record in caplog.records] == [
        f"{written}: a playlist of the 1 possible cut for VLC, an entry each, from 1 second"
        " before its cut to 3 seconds after; it names the source by its file name alone,"
        " in 1.mkv: move it beside the source, then open it in VLC"
    ]
    # The source's directory by another path, a link's: beside it all the same.
    caplog.clear()
    (tmp_path / "link").symlink_to("films")
    assert write_playlist(tmp_path / "link" / "other.txt", index, found, source)
    assert "; beside the source, which it names by its file name alone" in caplog.text
    # No possible cut: no playlist; one from before left as it is, and said.
    caplog.clear()
    assert write_playlist(tmp_path / "plans" / "none.txt", index, [], source)
    assert not (tmp_path / "plans" / "none.txt.m3u8").exists() and caplog.records == []
    assert write_playlist(elsewhere, index, [], source)
    assert written.read_text() == playlist(index, found[:1], "in 1.mkv")
    assert [(record.levelname, record.getMessage()) for record in caplog.records] == [
        (
            "WARNING",
            f"{written}: left as it is: this plan has no possible cut, so no playlist; a playlist"
            " there is an earlier plan's, its entries not this plan's",
        )
    ]
    # Not written, the disk full: said, and False, the earlier one as it was.
    caplog.clear()

    def full(path: Path, data: bytes) -> None:
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    monkeypatch.setattr(files, "write_whole", full)
    assert not write_playlist(elsewhere, index, found, source)
    assert [(record.levelname, record.getMessage()) for record in caplog.records] == [
        (
            "ERROR",
            f"--plan {elsewhere}: its playlist, {written}, not written: [Errno 28] No space left"
            " on device",
        )
    ]
    assert written.read_text() == playlist(index, found[:1], "in 1.mkv")


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

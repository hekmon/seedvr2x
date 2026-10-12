"""Missing frames filled (media/grid.py, media/index.py, media/reader.py; DESIGN.md, Input:
"Missing frames are filled, not refused"): each decoded frame's nearest place on a rate's
timeline, exact; the frame index in places, its held rows, its maps and what was filled; the shot
detector's probabilities spread over the places, the cuts drawn from them those of the decoded
frames, each at its frame's place; and the reads through such an index, on real files with holes
made on purpose, held to the frames of the file they were made from. No judgement here, which is
media/source.py's: these give each frame the place its timestamp says.

A file with holes is a stream copy of a constant-rate encode with some packets dropped, the
others' timestamps kept (holed), two commands, each one line:

    ffmpeg -f lavfi -i testsrc2=s=160x96:r=24000/1001 -frames:v 300 -c:v libx264
      -preset veryfast -x264-params
      keyint=48:min-keyint=24:open-gop=1:bframes=3:b-pyramid=none:b-adapt=0:scenecut=0 full.mkv
    ffmpeg -i full.mkv -map 0 -c copy -bsf:v
      "noise=drop='st(0,round(pts*tb*24000/1001));eq(ld(0),46)+eq(ld(0),47)'" holes.mkv

The noise bitstream filter drops the packets its expression is positive for (libavcodec/bsf/
noise.c at n9.0.2), here by the frame number their pts gives, frames 46 and 47; a remainder
picks the frames of a repeating pattern, st(0,mod(round(pts*tb*24000/1001),12)) (a sum of 100
terms is beyond ffmpeg's expression parser). Only B-frames are dropped, which the encode makes
references of nothing (b-pyramid=none), at fixed places, three after each I or P frame
(b-adapt=0, scenecut=0): every frame kept decodes as in the full file, which the tests check,
their pts and MD5s the full file's (framemd5) and the dropped packets absent (ffprobe). So it is
with the source this is for, S8, whose decoder gives neither of its second keyframe's two
leading pictures (research/docs/seeking.md, mechanism 1): the frames around the hole are whole,
and the stream declares its rate still. sptenc's timing rule refuses such a file ("variable
frame rate: frames last from 41 to 125 ms"), so the machinery's tests go through the first
pass's scan, not its judge; the judge's (media/source.py, _on_grid, in_places) then read the
same files as a source is read (examine), in places, with what its warning says.

Skipped without an ffmpeg seedvr2x accepts, or without libx264."""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from test_index import bounded, needs_ffmpeg, packet_list, reference, run
from test_probe import has_encoder

from seedvr2x.media.conversion import Conversion, conversion_for
from seedvr2x.media.decode import Decoder
from seedvr2x.media.ffmpeg import MediaError, input_args
from seedvr2x.media.grid import NONE, Filling, Hole, filled, filling, holes, places, spread
from seedvr2x.media.index import NOPTS, FrameIndex
from seedvr2x.media.probe import probe
from seedvr2x.media.reader import Reader, read_command, seek_time
from seedvr2x.media.scan import scan
from seedvr2x.media.source import examine, resolved
from seedvr2x.runtime.cuts import Cut, detected, detections, possible

NTSC = Fraction(24000, 1001)


def index_of(
    pts: list[int] | npt.NDArray[np.int64],
    time_base: Fraction = Fraction(1, 1000),
    probabilities: npt.NDArray[np.float32] | None = None,
) -> FrameIndex:
    """A first pass's index of frames at these timestamps: each frame's CRC-32 its number plus
    100, every fifth flagged with an error."""
    count = len(pts)
    return FrameIndex(
        time_base,
        0,
        np.array(pts, np.int64),
        np.arange(count, dtype=np.uint32) + 100,
        np.arange(count) % 5 == 3,
        np.array(pts[:1], np.int64),
        probabilities,
    )


def milliseconds(frames: int, rate: Fraction = NTSC) -> npt.NDArray[np.int64]:
    """The timestamps of `frames` frames at `rate`, in milliseconds, each rounded to the nearest,
    as Matroska holds them."""
    return np.array([round(k * 1000 / rate) for k in range(frames)], np.int64)


def test_places_exact_on_a_long_timeline() -> None:
    # 200,000 frames at 24000/1001 in Matroska's milliseconds, 2 h 19 min: every frame its own
    # place, within half a millisecond of it, no tie; n / fps in floating point is what
    # seeking.md's mechanism 1 warns of. At 25 fps in 90 kHz ticks, every frame on its place.
    pts = milliseconds(200_000)
    found = places(index_of(pts), NTSC)
    assert np.array_equal(found.place, np.arange(200_000))
    assert found.timed.all() and not found.tie.any()
    assert found.largest == Fraction(1, 2000) and found.numerators.dtype == np.int64
    for frame in (1, 12, 199_999):
        expected = Fraction(int(pts[frame]), 1000) - frame / NTSC
        assert found.deviation(frame) == expected
    pal = places(index_of(np.arange(200_000) * 3600, Fraction(1, 90000)), Fraction(25))
    assert np.array_equal(pal.place, np.arange(200_000)) and pal.largest == 0
    # Every third frame missing: the others on their own places still.
    kept = np.flatnonzero(np.arange(200_000) % 3 != 1)
    assert np.array_equal(places(index_of(pts[kept]), NTSC).place, kept)
    # Past int64, in Python's integers, exact all the same: nanoseconds, eleven days in.
    far = np.array([0, 10**15, 10**15 + 16_683_333, 10**15 + 2 * 16_683_333], np.int64)
    rate = Fraction(60000, 1001)
    late = places(index_of(far, Fraction(1, 10**9)), rate)
    assert late.numerators.dtype == object
    for frame in range(4):
        seconds = Fraction(int(far[frame]), 10**9)
        nearest = round(seconds * rate)
        assert int(late.place[frame]) == nearest
        assert late.deviation(frame) == seconds - nearest / rate
    assert late.largest == max(abs(late.deviation(frame) or 0) for frame in range(4)) > 0


def test_a_tie_is_reported_never_rounded_away() -> None:
    # A frame exactly half a frame from two places: given the earlier, half a frame after it,
    # and said; before the first frame as after it. Every other frame's deviation is within
    # half a frame.
    found = places(index_of([0, 40, 100, 120, 139, 141]), Fraction(25))
    assert found.place.tolist() == [0, 1, 2, 3, 3, 4]  # two frames on one place: not judged
    assert found.tie.tolist() == [False, False, True, False, False, False]
    assert found.deviation(2) == Fraction(1, 50) and found.deviation(4) == Fraction(19, 1000)
    assert found.deviation(5) == Fraction(-19, 1000) and found.largest == Fraction(1, 50)
    back = places(index_of([100, 80, 0, 140]), Fraction(25))
    assert back.place.tolist() == [0, -1, -3, 1] and back.tie.tolist() == [False, True, True, False]
    assert back.deviation(1) == back.deviation(2) == Fraction(1, 50)
    # Frames without a pts: no place, no deviation, each counted; the timeline anchored at the
    # first frame that has one, at the place of its own count (as media/rate.py's timeline).
    untimed = places(index_of([NOPTS, 0, 40, NOPTS, 120]), Fraction(25))
    assert untimed.place.tolist() == [NONE, 1, 2, NONE, 4]
    assert untimed.timed.tolist() == [False, True, True, False, True]
    assert untimed.deviation(0) is None and untimed.deviation(4) == 0
    none = places(index_of([NOPTS, NOPTS]), Fraction(25))
    assert none.place.tolist() == [NONE, NONE] and none.largest == 0
    assert places(index_of([]), Fraction(25)).place.tolist() == []
    with pytest.raises(ValueError, match="rate 0, not positive"):
        places(index_of([0, 40]), Fraction(0))


def test_the_index_in_places() -> None:
    # Seven frames of 24000/1001 in milliseconds, frames 3, 4 and 7 to 18 missing: 21 places.
    # A row a place: a frame's own, as the first pass recorded it, or a held one, the frame
    # before it again, its CRC-32 and error flag that frame's; the maps both ways; the decoded
    # frames' index the first pass's exactly.
    kept = [0, 1, 2, 5, 6, 19, 20]
    probabilities = np.array([0.1, 0.2, 0.9, 0.4, 0.5, 0.6, 0.7], np.float32)
    first = index_of(milliseconds(21)[kept], probabilities=probabilities)
    found = places(first, NTSC)
    assert found.place.tolist() == kept
    index = filled(first, found.place, NTSC)
    assert index.frames == 21 and index.held is not None
    assert np.flatnonzero(index.held).tolist() == [3, 4, *range(7, 19)]
    assert index.shown.tolist() == [0, 1, 2, 2, 2, 3, 4, *([4] * 12), 5, 6]
    assert index.placed.tolist() == kept
    assert index.crc32.tolist() == [100 + frame for frame in index.shown.tolist()]
    assert index.error.tolist() == [frame == 3 for frame in index.shown.tolist()]
    assert np.array_equal(index.pts[kept], first.pts)
    assert (index.time_base, index.start_time) == (first.time_base, first.start_time)
    assert np.array_equal(index.keyframes, first.keyframes)
    decoded = index.decoded()
    assert decoded.held is None and decoded.frames == 7
    for name in ("pts", "crc32", "error", "keyframes"):
        assert np.array_equal(getattr(decoded, name), getattr(first, name)), name
    assert decoded.probabilities is not None
    assert decoded.probabilities.tobytes() == probabilities.tobytes()
    assert decoded.to_bytes() == first.to_bytes()
    # No place left empty: the first pass's index itself, whatever the rate; its maps count.
    assert filled(first, list(range(7)), Fraction(24)) is first
    assert first.decoded() is first
    assert first.shown.tolist() == first.placed.tolist() == list(range(7))
    assert replace(first, held=np.zeros(7, np.bool_)).held is None
    # What is asked of ffmpeg's frames is asked of the decoded frames' index: by place, refused.
    for ask in (
        lambda: index.own_keyframe(5),
        lambda: index.first_frame_from(0),
        lambda: index.selects(5),
        lambda: index.lowest_from(5),
        lambda: index.first_error(0, 20),
    ):
        with pytest.raises(ValueError, match="held rows, by place: a seedvr2x bug"):
            ask()
    assert decoded.own_keyframe(5) == 0 and decoded.first_error(0, 6) == 3
    # The places are the caller's judgement: one a frame, from 0, strictly increasing.
    refused = {
        (1, 2, 3, 5, 6, 19, 20): "the first frame's place is 1, not 0",
        (0, 1, 2, 2, 6, 19, 20): "frame 3's place, 2, isn't after frame 2's, 2",
        (0, 1, 2, 5, 4, 19, 20): "frame 4's place, 4, isn't after frame 3's, 5",
        (0, 1, 2): "places of shape (3,) and type int64 for 7 frames",
    }
    for given, said in refused.items():
        with pytest.raises(ValueError, match=re.escape(said)):
            filled(first, np.array(given), NTSC)
    with pytest.raises(ValueError, match="type float64 for 7 frames"):
        filled(first, np.array(kept, np.float64), NTSC)
    with pytest.raises(ValueError, match="rate -24, not positive"):
        filled(first, kept, Fraction(-24))
    with pytest.raises(ValueError, match="held rows filled again"):
        filled(index, np.arange(21), NTSC)
    with pytest.raises(ValueError, match="places asked of a frame index with held rows"):
        places(index, NTSC)
    with pytest.raises(ValueError, match="its first row held"):
        replace(index, held=~index.held)
    with pytest.raises(ValueError, match="20 held flags for 21 rows"):
        replace(index, held=index.held[:20])
    empty = index_of([])
    assert filled(empty, [], NTSC) is empty


def test_a_held_rows_time_is_its_places() -> None:
    # A held row's pts: the pts of the frame it shows plus the frame periods since, rounded to
    # the nearest tick, half a tick up, ffmpeg's rounding of a timestamp. At 24000/1001 in
    # milliseconds, a period is 41.708 ms: 42, 83, 125 ms after the frame, the twelfth place
    # 501 from 500.5, a tie; counted from the frame held, at 83 ms and at 250 ms here.
    kept = [0, 1, 2, 5, 6, 19, 20]
    stamps = milliseconds(21)
    index = filled(index_of(stamps[kept]), kept, NTSC)
    assert index.pts[2:5].tolist() == [83, 83 + 42, 83 + 83]
    after = [42, 83, 125, 167, 209, 250, 292, 334, 375, 417, 459, 501]
    assert index.pts[6:19].tolist() == [250, *(250 + ticks for ticks in after)]
    assert index.pts[19:].tolist() == [792, 834]
    # Within a millisecond of the missing frames' own timestamps in the full file.
    assert np.abs(index.pts - stamps).max() == 1
    # At 25 fps in 90 kHz ticks, a period is 3,600 ticks exactly, from wherever the frame is.
    pal = index_of([900_000, 903_600, 914_400, 918_000], Fraction(1, 90000))
    at = filled(pal, [0, 1, 4, 5], Fraction(25))
    assert at.pts.tolist() == [900_000 + 3600 * place for place in range(6)]
    # Rounded down under half a tick, and up from it: 2 frames at 30000/1001 are 66.73 ms, 3 at
    # 1000/3 fps 9 ms; a frame without a pts holds none.
    assert filled(index_of([0, 100]), [0, 3], Fraction(30000, 1001)).pts.tolist() == [
        0,
        33,
        67,
        100,
    ]
    assert filled(index_of([0, 14]), [0, 4], Fraction(1000, 3)).pts.tolist() == [0, 3, 6, 9, 14]
    assert filled(index_of([0, 5]), [0, 2], Fraction(400)).pts.tolist() == [0, 3, 5]  # 2.5: up
    untimed = filled(index_of([0, NOPTS, 200]), [0, 1, 4], Fraction(25))
    assert untimed.pts.tolist() == [0, NOPTS, NOPTS, NOPTS, 200]
    # A long hole in a fine time base, past int64's products: Python's integers, exact.
    far = filled(index_of([0, 10**17], Fraction(1, 10**12)), [0, 2 * 10**6], Fraction(24))
    assert far.frames == 2 * 10**6 + 1 and int(far.pts[-2]) == round(Fraction(10**12, 24) * 1999999)


def test_what_was_filled() -> None:
    # The summary for the messages and the record: the places, the filled ones, the holes, the
    # first (its first place, the number of the first missing frame; its length; the frame held
    # through it), the longest, the share.
    kept = [0, 1, 2, 5, 6, 19, 20]
    index = filled(index_of(milliseconds(21)[kept]), kept, NTSC)
    assert holes(index) == [Hole(3, 2, 2), Hole(7, 12, 4)]
    assert [hole.held for hole in holes(index)] == [2, 6]  # the places of the frames held
    found = filling(index)
    assert found == Filling(places=21, filled=14, holes=2, first=Hole(3, 2, 2), longest=12)
    assert found.decoded == 7 and found.share == Fraction(2, 3)
    whole = index.decoded()
    assert holes(whole) == []
    assert filling(whole) == Filling(places=7, filled=0, holes=0, first=None, longest=0)
    assert filling(whole).share == 0 and filling(index_of([])).share == 0
    # A third of the places empty, 100,000 holes: counted, not listed one by one in the summary.
    third = np.flatnonzero(np.arange(300_000) % 3 != 1)
    large = filling(filled(index_of(third * 40), third, Fraction(25)))
    assert (large.places, large.filled, large.holes, large.longest) == (
        300_000,
        100_000,
        100_000,
        1,
    )
    assert large.first == Hole(1, 1, 0) and large.share == Fraction(1, 3)


def moved(cuts: list[Cut], placed: npt.NDArray[np.int64]) -> list[Cut]:
    """The decoded frames' cuts, each at its frame's own place, with its probability."""
    return [Cut(int(placed[cut.frame]), cut.probability) for cut in cuts]


def test_the_cuts_stay_where_they_were() -> None:
    # The shot detector saw the decoded frames alone. Spread over the places, its probabilities
    # give the very cuts the decoded frames' give, detected and possible, each at its frame's
    # own place with its probability: no shot starts on a held copy of the shot before's last
    # frame, a run of frames reaching the threshold stays one run, its first highest frame its
    # peak. On random probabilities, plateaus included (values rounded to a few levels, where
    # frames tie), random holes, sparse to two places in three, at 0.1, 0.3, 0.5 and at
    # thresholds some frames reach exactly. And the decoded frames' own come back bit for bit.
    generator = np.random.default_rng(23)
    checked = 0
    for count in (1, 2, 3, 40, 1500):
        for levels in (4, 20, 0):
            for share in (0.02, 0.3, 0.7):
                values = generator.random(count, np.float32)
                if levels:
                    values = (np.round(values * levels) / levels).astype(np.float32)
                # After each frame, a hole with probability `share`, of 1 frame or more.
                lengths = generator.geometric(0.5, count - 1)
                steps = np.where(generator.random(count - 1) < share, 1 + lengths, 1)
                placed = np.concatenate(([0], np.cumsum(steps))).astype(np.int64)
                over = spread(values, placed)
                assert over.dtype == np.float32 and len(over) == placed[-1] + 1
                last = np.append(placed[1:], len(over)) - 1
                assert over[last].tobytes() == values.tobytes()
                # No place scores above the frame it shows: a run's places are its frames'.
                shown = np.repeat(np.arange(count), last + 1 - placed)
                assert bool(np.all(over <= values[shown]))
                ties = [float(values[k]) for k in generator.integers(0, count, 3)]
                for threshold in (0.1, 0.3, 0.5, 0.05, 1.0, *(t for t in ties if t > 0)):
                    assert detected(over, threshold) == moved(detected(values, threshold), placed)
                    assert possible(over, threshold) == moved(possible(values, threshold), placed)
                    peaks = detections(values, threshold)
                    assert detections(over, threshold) == [int(last[peak]) for peak in peaks]
                    checked += len(peaks)
    assert checked > 5000
    # The rule itself, on a case by hand: frame 2 (0.9) held through places 3 and 4, frame 4
    # (0.5) through 7 to 18. A frame's own probability on the last place that shows it, the
    # lower of its own and the frame before's on its other places.
    kept = [0, 1, 2, 5, 6, 19, 20]
    values = np.array([0.1, 0.2, 0.9, 0.4, 0.5, 0.6, 0.7], np.float32)
    index = filled(index_of(milliseconds(21)[kept], probabilities=values), kept, NTSC)
    low, high = np.float32(0.2), np.float32(0.9)
    expected = [0.1, 0.2, low, low, high, 0.4, *([0.4] * 12), 0.5, 0.6, 0.7]
    assert index.probabilities is not None
    assert index.probabilities.tolist() == np.array(expected, np.float32).tolist()
    # At 0.8, the cut after frame 2's peak is frame 3's own place, 5, never the held place 3.
    assert detected(index.probabilities, 0.8) == [Cut(5, float(high))]
    assert detected(values, 0.8) == [Cut(3, float(high))]
    # A frame lower than the one before it, held: its other places take its own, the lower,
    # never the higher one before, which would carry that frame's run over them.
    falling = spread(np.array([0.8, 0.3, 0.1], np.float32), np.array([0, 1, 4]))
    assert falling.tolist() == [np.float32(0.8), *([np.float32(0.3)] * 3), np.float32(0.1)]
    # The first frame held: 0 stands for the frame before it.
    first = spread(np.array([0.6, 0.2], np.float32), np.array([0, 3]))
    assert first.tolist() == [0.0, 0.0, np.float32(0.6), np.float32(0.2)]
    assert detected(first, 0.3) == [Cut(3, float(np.float32(0.6)))]
    with pytest.raises(ValueError, match="6 shot probabilities for 7 frames"):
        spread(values[:6], np.array(kept))
    with pytest.raises(ValueError, match="6 shot probabilities for 7 frames"):
        filled(index_of(milliseconds(21)[kept], probabilities=values[:6]), kept, NTSC)


FRAMES = 300
WIDTH, HEIGHT = 160, 96
# B-frames referenced by nothing, three after each I or P frame, a keyframe every 48 frames with
# open GOPs: any B-frame's packet dropped leaves the others' decode as it was (the module's
# docstring).
X264 = "keyint=48:min-keyint=24:open-gop=1:bframes=3:b-pyramid=none:b-adapt=0:scenecut=0"
# Each file with holes: its container, and its missing frames, those whose number modulo
# `period` is one of `residues` (a period past the file's length for single frames).
ONCE = 10**6
MIDDLE = (201, 202, 203)
HOLED: dict[str, tuple[str, int, tuple[int, ...]]] = {
    # The two frames after the first, as the source this is for has them missing (label
    # anime-dark: 99,583 frames decoded, the second 125 ms after the first).
    "right after the first frame": ("mkv", ONCE, (1, 2)),
    # The second keyframe's last two leading pictures, displayed before it and decoded after
    # it, as S8's decoder gives neither of its two: one hole of 2 frames, before the third
    # keyframe, where reads decode from the start and count.
    "the second keyframe's leading pictures": ("mkv", ONCE, (46, 47)),
    # Holes of 1 to 3 frames: after the first frame; right after the second keyframe (48);
    # the third keyframe (96) between a hole of its three leading pictures and one of the
    # frame after it; the fourth's (144) leading pictures; in the middle of GOPs; and the
    # place before the last frame.
    "scattered": ("mkv", ONCE, (1, 49, 50, 93, 94, 95, 97, 141, 142, 143, 150, *MIDDLE, 294, 298)),
    "scattered, MP4": ("mp4", ONCE, (49, 50, 141, 142, 143, 150, *MIDDLE, 298)),
    # A third of the places empty: of every 12 frames, 4 missing, holes of 2, 1 and 1.
    "a third of the places": ("mkv", 12, (1, 2, 6, 10)),
}


@dataclass(frozen=True)
class Holed:
    """A file with holes, and what its reads are held to, made independently of the index: the
    frames of the full file it was made from, decoded with the same conversion."""

    path: Path
    full: Path  # the file it was made from, every frame there
    missing: list[int]  # the frames dropped, by their number in the full file
    showing: list[int]  # for each place, the full file's frame a player shows there
    rgb: npt.NDArray[np.uint16]  # the full file's frames, (frames, H, W, 3)
    md5: list[str]  # the full file's frames as decoded
    conversion: Conversion
    first_pass: FrameIndex  # the file's own index, of its decoded frames (media/scan.py)
    index: FrameIndex  # in places


def holed(directory: Path, name: str, suffix: str, period: int, residues: tuple[int, ...]) -> Path:
    """The file `name` with holes, in directory: a stream copy of the full file of that suffix,
    made once, the frames whose number modulo period is in residues dropped (the module's
    docstring)."""
    base = directory / "full.mkv"
    if not base.exists():
        source = ("-f", "lavfi", "-i", f"testsrc2=s={WIDTH}x{HEIGHT}:r={NTSC}")
        encode = (
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-x264-params",
            f"{X264}:log-level=error",
        )
        run(*source, "-frames:v", str(FRAMES), *encode, str(base))
    full = directory / f"full.{suffix}"
    if not full.exists():
        run("-i", str(base), "-map", "0", "-c", "copy", str(full))
    number = f"round(pts*tb*{NTSC})"
    frame = f"mod({number},{period})" if period < FRAMES else number
    drop = f"st(0,{frame});" + "+".join(f"eq(ld(0),{residue})" for residue in residues)
    path = directory / f"{name}.{suffix}"
    run("-i", str(full), "-map", "0", "-c", "copy", "-bsf:v", f"noise=drop='{drop}'", str(path))
    return path


@pytest.fixture(scope="module")
def made(tmp_path_factory: pytest.TempPathFactory) -> Callable[[str], Holed]:
    """kind -> the file of HOLED, made, scanned and its index filled, once for the module."""
    directory = tmp_path_factory.mktemp("holed")
    found: dict[str, Holed] = {}

    def get(kind: str) -> Holed:
        if not has_encoder("libx264"):
            pytest.skip("needs ffmpeg with libx264")
        if kind not in found:
            suffix, period, residues = HOLED[kind]
            path = holed(directory, f"holed{len(found)}", suffix, period, residues)
            full = directory / f"full.{suffix}"
            missing = [frame for frame in range(FRAMES) if frame % period in residues]
            showing: list[int] = []
            for frame in range(FRAMES):
                showing.append(showing[-1] if frame in missing else frame)
            stream = resolved(probe(path))
            assert (stream.width, stream.height, stream.frame_rate) == (WIDTH, HEIGHT, NTSC)
            conversion = conversion_for(stream)
            assert conversion == conversion_for(resolved(probe(full)))
            with Decoder(input_args(full), conversion, WIDTH, HEIGHT, FRAMES) as decoder:
                rgb = decoder.read(FRAMES)
            assert rgb.shape[0] == FRAMES
            scanned = scan(path)
            first_pass = scanned.index
            assert first_pass is not None and not first_pass.error.any()
            index = filled(first_pass, places(first_pass, stream.frame_rate).place, NTSC)
            found[kind] = Holed(
                *(path, full, missing, showing, rgb, reference(full).md5, conversion),
                first_pass,
                index,
            )
        return found[kind]

    return get


def pipes(command: tuple[str, ...]) -> tuple[int, ...]:
    """The file descriptors of a read's hash pipes, in its command's order: the CRC-32s', then
    the MD5s' when asked."""
    return tuple(
        int(part.removeprefix("pipe:"))
        for part in command
        if part.startswith("pipe:") and part != "pipe:1"
    )


def read(case: Holed, first: int, counts: list[int], frames: int | None = None) -> Reader:
    """A read of the file's places from `first`, `frames` in all, in as many reads as counts,
    each held to the full file's frames a player shows at those places, RGB and MD5; fewer than
    asked at the file's end alone; stopped after."""
    reader = Reader(
        *(case.path, case.conversion, WIDTH, HEIGHT, case.index, first, frames), md5=True
    )
    try:
        place = first
        for count in counts:
            given = bounded(reader, lambda reading, count=count: reading.read(count))
            end = min(place + count, FRAMES)
            assert given.shape[0] == end - place, (first, place, count)
            wrong = [
                at
                for at in range(place, end)
                if not np.array_equal(given[at - place], case.rgb[case.showing[at]])
            ]
            assert not wrong, f"read from {first}: places {wrong} aren't the frames shown there"
            place = end
            assert reader.position == place
        assert reader.md5s == [case.md5[case.showing[at]] for at in range(first, place)]
    finally:
        reader.stop()
    return reader


@needs_ffmpeg
@pytest.mark.parametrize("kind", HOLED)
def test_a_file_with_holes_is_the_full_files_frames(
    made: Callable[[str], Holed], kind: str
) -> None:
    # The recipe: the frames kept are the full file's, their pts and pictures unchanged, the
    # dropped ones absent, packets and frames; the stream declares its rate still. The first
    # pass's index is of the frames decoded; each one's nearest place at that rate is its number
    # in the full file, within Matroska's millisecond; filled, the index has the full file's
    # places, the missing frames' held.
    case = made(kind)
    full, got = reference(case.full), reference(case.path)
    kept = [frame for frame in range(FRAMES) if frame not in case.missing]
    assert len(full.pts) == FRAMES and case.missing and kept[-1] == FRAMES - 1
    assert got.pts == [full.pts[frame] for frame in kept]
    assert got.md5 == [full.md5[frame] for frame in kept]
    left = {int(packet["pts"]) for packet in packet_list(case.path)}
    assert left == {int(packet["pts"]) for packet in packet_list(case.full)} - {
        full.pts[frame] for frame in case.missing
    }
    first_pass, index = case.first_pass, case.index
    assert first_pass.pts.tolist() == got.pts and first_pass.held is None
    found = places(first_pass, NTSC)
    assert found.place.tolist() == kept and not found.tie.any()
    assert found.largest <= Fraction(1, 1000)
    assert index.frames == FRAMES and index.held is not None
    assert np.flatnonzero(index.held).tolist() == case.missing
    assert index.shown.tolist() == [kept.index(frame) for frame in case.showing]
    assert index.decoded().to_bytes() == first_pass.to_bytes()
    summary = filling(index)
    assert (summary.places, summary.filled) == (FRAMES, len(case.missing))
    assert summary.first is not None and summary.first.first == case.missing[0]
    if kind == "a third of the places":
        assert summary.share == Fraction(1, 3) and summary.holes == 75
    # A held row's time is its place's, within a millisecond and a half of the missing frame's own
    # (two roundings to the millisecond, the file's, and one to the tick).
    apart = max(abs(int(index.pts[place]) - full.pts[place]) for place in range(FRAMES))
    assert apart * full.time_base * 1000 < Fraction(3, 2)


# What a source's warning says of its missing frames (media/source.py, _missing), after its path.
FILLED_AS = "read as a player shows it, the frame before each hole held through it"
SAID = {
    "right after the first frame": (
        "2 frames missing after frame 0: its timestamps step from 0.000 to 0.125 s there, 3"
        " frames of the rate it declares, 24000/1001, leaving places 1 and 2 empty on that rate's"
        " grid; read as a player shows it, frame 0 held in their place: 300 frames, 2 of them"
        " filled (0.67%), 298 decoded"
    ),
    "the second keyframe's leading pictures": (
        "2 frames missing after frame 45: its timestamps step from 1.877 to 2.002 s there, 3"
        " frames of the rate it declares, 24000/1001, leaving places 46 and 47 empty on that"
        " rate's grid; read as a player shows it, frame 45 held in their place: 300 frames, 2 of"
        " them filled (0.67%), 298 decoded"
    ),
    "scattered": (
        "16 frames missing in 9 holes, the first after frame 0, 1 frame long (its timestamps"
        " step from 0.000 to 0.083 s there, 2 frames of the rate it declares, 24000/1001), the"
        " longest 3 frames: its timestamps leave those places empty on the grid of the rate it"
        f" declares, 24000/1001; {FILLED_AS}: 300 frames, 16 of them filled (5.3%), 284 decoded"
    ),
    "scattered, MP4": (
        "10 frames missing in 5 holes, the first after frame 48, 2 frames long (its timestamps"
        " step from 2.002 to 2.127 s there, 3 frames of the rate it declares, 24000/1001), the"
        " longest 3 frames: its timestamps leave those places empty on the grid of the rate it"
        f" declares, 24000/1001; {FILLED_AS}: 300 frames, 10 of them filled (3.3%), 290 decoded"
    ),
    "a third of the places": (
        "100 frames missing in 75 holes, the first after frame 0, 2 frames long (its timestamps"
        " step from 0.000 to 0.125 s there, 3 frames of the rate it declares, 24000/1001), the"
        " longest 2 frames: its timestamps leave those places empty on the grid of the rate it"
        f" declares, 24000/1001; {FILLED_AS}: 300 frames, 100 of them filled (33.3%), 200"
        " decoded; so large a share is what an encoder leaves that drops repeated frames, each"
        " picture held as it was until the next"
    ),
}


def missing_said(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The warnings of missing frames logged since caplog was cleared."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and " missing " in record.getMessage()
    ]


@needs_ffmpeg
@pytest.mark.parametrize("kind", HOLED)
def test_a_source_with_holes_is_read_in_places(
    made: Callable[[str], Holed], kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    # The judge (media/source.py): a file whose frames sit on its rate's grid with places left
    # empty is read, where sptenc's rule refused it as a variable frame rate: its frames are the
    # places, as many as the full file's frames, its index the one in places, and its warning
    # says the first hole, how many, the longest and the share. The same with --frame-rate at
    # its rate. Its reads give the full file's frames, the frame before a hole in each of its
    # places.
    case = made(kind)
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source = examine(case.path)
    assert source.frames == FRAMES and source.index is not None
    assert source.index.to_bytes() == case.index.to_bytes()
    summary = filling(case.index)
    assert source.filled == summary and summary.first is not None
    assert missing_said(caplog) == [f"{case.path}: {SAID[kind]}"]
    decoded = FRAMES - len(case.missing)
    counts = f"{FRAMES} frames ({decoded} decoded, {len(case.missing)} filled), {WIDTH}x{HEIGHT}"
    assert f"{case.path}: {counts}" in caplog.text
    forced = examine(case.path, frame_rate=NTSC)
    assert forced.index is not None and forced.index.to_bytes() == case.index.to_bytes()
    start = max(summary.first.held - 2, 0)
    reader = source.reader(start, md5=True)
    try:
        given = bounded(reader, lambda reading: reading.read(8))
    finally:
        reader.stop()
    shown = [case.showing[place] for place in range(start, start + 8)]
    hole = summary.first
    assert shown[hole.first - start :][: hole.length] == [hole.held] * hole.length
    assert reader.md5s == [case.md5[frame] for frame in shown]
    assert all(np.array_equal(given[k], case.rgb[frame]) for k, frame in enumerate(shown))


@needs_ffmpeg
def test_a_source_without_holes_is_read_as_before(
    made: Callable[[str], Holed], caplog: pytest.LogCaptureFixture
) -> None:
    # The full file: no place filled, its index the first pass's own, nothing said of missing
    # frames, its count as before the places.
    case = made("right after the first frame")
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source = examine(case.full)
    assert source.frames == FRAMES and source.filled is None
    assert source.index is not None and source.index.held is None
    scanned = scan(case.full).index
    assert scanned is not None and source.index.to_bytes() == scanned.to_bytes()
    assert not missing_said(caplog) and "filled" not in caplog.text
    assert (
        f"{case.full}: {FRAMES} frames, {WIDTH}x{HEIGHT} yuv420p at 24000/1001 fps" in caplog.text
    )


@needs_ffmpeg
@pytest.mark.parametrize("kind", HOLED)
def test_every_place_read_from_the_first(made: Callable[[str], Holed], kind: str) -> None:
    # Every place from 0, in one decode from the start: a frame's own place gives it, a filled
    # one the frame before its hole again, as a player shows the file; one MD5 a place; the
    # end checked, every frame decoded checked once.
    case = made(kind)
    reader = Reader(case.path, case.conversion, WIDTH, HEIGHT, case.index, md5=True)
    with reader:
        given = bounded(reader, lambda reading: reading.read(FRAMES + 5))
    assert given.shape[0] == FRAMES and reader.position == FRAMES
    wrong = [
        place
        for place in range(FRAMES)
        if not np.array_equal(given[place], case.rgb[case.showing[place]])
    ]
    assert not wrong, f"places {wrong[:10]} aren't the frames a player shows there"
    assert reader.md5s == [case.md5[frame] for frame in case.showing]
    assert [attempt.keyframe for attempt in reader.attempts] == [None] and not reader.otherwise
    # In several reads, the first ending inside the first hole: the same frames.
    read(case, 0, [case.missing[0] + 1, 1, 7, FRAMES])


def chosen(case: Holed) -> list[Hole]:
    """The holes read around: every one of a file with few, else the first three, those either
    side of the third and fourth keyframes (96, 144), and the last two."""
    found = holes(case.index)
    if len(found) <= 12:
        return found
    near = [hole for hole in found if 90 <= hole.first <= 100 or 138 <= hole.first <= 148]
    return [*found[:3], *near, *found[-2:]]


@needs_ffmpeg
@pytest.mark.parametrize("kind", HOLED)
def test_reads_around_each_hole(made: Callable[[str], Holed], kind: str) -> None:
    # A read starting before a hole, on the frame it holds, on each of its places and just
    # after it gives the frames a player shows there. One starting inside a hole first reads
    # the frame the hole holds, reached as any frame is, by a seek past the third keyframe; a
    # resume opened past a hole seeks past it, to the decoded frame its place says.
    case = made(kind)
    seeks = 0
    for hole in chosen(case):
        end = hole.first + hole.length
        starts = [hole.first - 2, hole.held, *range(hole.first, end), end]
        for first in (start for start in starts if start >= 0):
            reader = read(case, first, [6])
            assert len(reader.attempts) == 1, (first, reader.attempts)
            seeks += reader.attempts[0].keyframe is not None
            # The frame the read was opened at: the one its first place shows, by its own place.
            assert reader.attempts[0].target == case.showing[first]
        # A read ending inside the hole, on its first place, stops there, and the next goes on
        # through it.
        before = max(hole.held - 1, 0)
        read(case, before, [hole.first + 1 - before, 5])
        # A read counted to end inside the hole gives its places and no more: asked for
        # another, it fails as a read past the first pass's count does, ffmpeg having more.
        counted = hole.first + 1
        reader = Reader(
            *(case.path, case.conversion, WIDTH, HEIGHT, case.index, before, counted), md5=True
        )
        try:
            given = bounded(reader, lambda reading: reading.read(reading.frames - reading.position))
            assert given.shape[0] == counted - before == reader.position - before
            assert np.array_equal(given[-1], case.rgb[hole.held])
            decoded = int(case.index.shown[hole.held]) + 1
            said = (
                rf"more frames than the {decoded} the first pass decoded, {counted} with the"
                r" missing frames' places, filled"
            )
            with pytest.raises(MediaError, match=said):
                bounded(reader, lambda reading: reading.read(1))
        finally:
            reader.stop()
    # But where the one hole lies before the third keyframe: its reads decode from the start.
    early = ("right after the first frame", "the second keyframe's leading pictures")
    assert seeks or kind in early


@needs_ffmpeg
def test_a_read_starting_inside_a_hole_says_so(
    made: Callable[[str], Holed], caplog: pytest.LogCaptureFixture
) -> None:
    # The frames a message names are places, as a user counts them: place 202, a missing
    # frame's, shows frame 200, the frame before its hole, decoded frame 189 counted from 0;
    # read through the index from the keyframe one before its own (192): 144, decoded frame 134.
    case = made("scattered")
    caplog.set_level(logging.INFO)
    reader = read(case, 202, [4])
    assert case.index.shown[202] == 189 and case.index.shown[144] == 134
    path = re.escape(str(case.path))
    assert re.search(
        rf"{path}: frame 202 is a missing frame's place, filled by frame 200, the frame before it,"
        r" held: that frame is read for it\n",
        caplog.text,
    )
    assert re.search(
        rf"{path}: frame 200 read through the frame index, decoded from frame 144, the keyframe"
        r" one before its own\n",
        caplog.text,
    )
    [attempt] = reader.attempts
    assert (attempt.target, attempt.keyframe, attempt.back) == (200, 3, 1)
    # The command is the one a read of that decoded frame takes: the seek to the keyframe one
    # before its own, the select on its pts.
    first_pass = case.first_pass
    assert int(first_pass.pts[189]) == 8342 and seek_time(first_pass, 3) == "6.006000"
    assert list(attempt.command) == read_command(
        case.path, case.conversion, "6.006000", 8342, *pipes(attempt.command)
    )


@needs_ffmpeg
def test_a_frame_decoded_otherwise_counts_once(
    made: Callable[[str], Holed], caplog: pytest.LogCaptureFixture
) -> None:
    # The CRC-32 check is of the decoded frames, each once, named by its own place: an index
    # holding another CRC-32 for the frame before the hole (place 45, held through 46 and 47)
    # and for the one after it (place 48, decoded frame 46) takes both as decoded, from the
    # start, counted; the filled places give the frame again, checked no second time.
    case = made("the second keyframe's leading pictures")
    crc32 = case.first_pass.crc32.copy()
    crc32[[45, 46]] ^= 1
    index = filled(replace(case.first_pass, crc32=crc32), case.index.placed, NTSC)
    caplog.set_level(logging.WARNING)
    reader = Reader(case.path, case.conversion, WIDTH, HEIGHT, index, 40, md5=True)
    try:
        given = bounded(reader, lambda reading: reading.read(12))
    finally:
        reader.stop()
    assert all(np.array_equal(given[k], case.rgb[case.showing[40 + k]]) for k in range(12))
    assert reader.otherwise == [45, 48] and len(reader.md5s) == 12
    assert caplog.text.count("frame 45: CRC-32") == 1 and "frame 46" not in caplog.text
    assert "2 frames of this read decoded otherwise than in the first pass" in caplog.text
    assert caplog.text.rstrip().endswith("taken as decoded: 45, 48")


@needs_ffmpeg
def test_without_holes_the_reads_are_as_before(
    made: Callable[[str], Holed], caplog: pytest.LogCaptureFixture
) -> None:
    # A source without an empty place: its index in places is the first pass's own, and a read
    # through it is the read of before the places, its command, its log and its frames.
    case = made("scattered")
    stream = resolved(probe(case.full))
    scanned = scan(case.full)
    first_pass = scanned.index
    assert first_pass is not None
    found = places(first_pass, stream.frame_rate)
    assert found.place.tolist() == list(range(FRAMES))
    index = filled(first_pass, found.place, stream.frame_rate)
    assert index is first_pass and index.held is None
    caplog.set_level(logging.INFO)
    caplog.clear()
    reader = Reader(case.full, case.conversion, WIDTH, HEIGHT, index, 150, md5=True)
    try:
        given = bounded(reader, lambda reading: reading.read(8))
    finally:
        reader.stop()
    assert np.array_equal(given, case.rgb[150:158]) and reader.md5s == case.md5[150:158]
    [attempt] = reader.attempts
    assert (attempt.target, attempt.keyframe, attempt.back) == (150, 2, 1)
    # The command of before the places (0db0a72), spelled out: the seek to the third keyframe,
    # at 4.004 s, the select on frame 150's pts, 6,256 ms.
    assert int(first_pass.pts[150]) == 6256
    hashes, md5 = pipes(attempt.command)
    select = "select=gte(pts\\,6256)"
    one, bare = ("-threads", "1"), ("-map_metadata", "-1", "-map_chapters", "-1")
    hashed = ("-map", "0:v:0", "-fps_mode", "passthrough", "-enc_time_base:v", "demux", *bare)
    assert list(attempt.command) == [
        *("ffmpeg", "-hide_banner", "-nostdin", "-nostats", "-loglevel", "error", "-copyts"),
        *("-noaccurate_seek", "-ss", "4.004000", "-noautorotate", "-i", str(case.full)),
        *("-map", "0:v:0", "-vf", f"{select},{case.conversion.filters()}"),
        *("-fps_mode", "passthrough", "-noautoscale", *one, *bare),
        *("-f", "rawvideo", "-pix_fmt", "gbrp16le", "pipe:1"),
        *(*hashed, "-c:v", "rawvideo", *one, "-f", "framehash", "-hash", "crc32"),
        f"pipe:{hashes}",
        *("-map", "0:v:0", "-vf", select, *hashed[2:]),
        *("-c:v", "rawvideo", *one, "-f", "framemd5", f"pipe:{md5}"),
    ]
    assert list(attempt.command) == read_command(
        case.full, case.conversion, "4.004000", 6256, hashes, md5
    )
    assert caplog.messages == [
        f"{case.full}: frame 150 read through the frame index, decoded from frame 96, the"
        " keyframe one before its own"
    ]

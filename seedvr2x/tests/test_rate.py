"""A source's frame rate against its frames' own timestamps (DESIGN.md, Input: exact rational frame
rate; media/rate.py, media/source.py's timing_refusal): the timeline of a constant rate and its
half-frame bound, the declared rate's NTSC twin the frames follow, the refusal's guidance and its
remake command, the files a refusal names no rate for, and --frame-rate, on indexes made here and
on research/docs/seeking.md's mechanism 7 made on purpose (mechanism7); and seedvr2x's own test
of the files sptenc's rule passes, every frame within half a frame of its place on the declared
rate's timeline: other rates declared, refused naming the rate when the timestamps are its
exactly, joins by ffmpeg's concat demuxer that drift half a frame, refused as no rate's, with the
remake that puts them back; and what stays read, constant rates to their time base's rounding at
any length, joins that keep the frame grid or drift less, a video starting after its audio. The
command line's run with --frame-rate is test_cli_run.py's. Matroska's reading of an NTSC rate
(rate.matroska_reading, media/probe.py's parse): each reading taken, a stream probed read at the
NTSC rate, its timestamps held for hours, and seedvr2x's own master and finished file read back
at the rate they were written at.
Missing frames (DESIGN.md, Input: "Missing frames are filled, not refused"; media/source.py's
_on_grid and in_places): timestamps made here that sit on a rate's grid with places left empty,
read in places, with what the source's warning says, and what stays refused, each with its
message; real files with holes are tests/test_grid.py's, their runs test_cli_run.py's.

Opt-in, a real source whose timestamps contradict its declared rate (S9, on the GPU box): its path
in SEEDVR2X_RATE_SOURCE, and the rate expected, N/D, in SEEDVR2X_RATE_EXPECTED (optional). One
first pass, then the timing rule twice, as the command line takes it: without --frame-rate,
refused with the guidance; with the rate it gives, accepted. Its figures are printed (pytest -rP):

    CUDA_VISIBLE_DEVICES= SEEDVR2X_RATE_SOURCE=<file> SEEDVR2X_RATE_EXPECTED=24000/1001 \\
        uv run pytest tests/test_rate.py -k real_source -rP"""

import json
import logging
import os
import random
import shlex
import subprocess
import time
from collections.abc import Sequence
from dataclasses import replace
from fractions import Fraction
from itertools import accumulate, pairwise
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from test_conversion import UNTAGGED_HD
from test_index import joined_ts
from test_probe import has_encoder

from seedvr2x.media import ffmpeg, rate
from seedvr2x.media.assembly import assemble
from seedvr2x.media.conversion import conversion_for
from seedvr2x.media.ffmpeg import MediaError
from seedvr2x.media.grid import Filling, Hole, filling, places
from seedvr2x.media.index import NOPTS, FrameIndex
from seedvr2x.media.probe import VideoStream, parse
from seedvr2x.media.scan import Discontinuity, Scan, scan, timing_error
from seedvr2x.media.source import (
    Declared,
    FirstPass,
    counted,
    declare,
    examine,
    in_places,
    timing_refusal,
)
from seedvr2x.media.writer import FFV1Writer, Tags


def _usable() -> bool:
    try:
        ffmpeg.check()
    except MediaError:
        return False
    return True


needs_ffmpeg = pytest.mark.skipif(
    not _usable(), reason="needs ffmpeg 7.1 or later with zscale, idet and ffv1"
)
needs_x264 = pytest.mark.skipif(not has_encoder("libx264"), reason="needs ffmpeg with libx264")

NTSC_FILM = Fraction(24000, 1001)


def indexed(pts: list[int], time_base: Fraction = Fraction(1, 1000)) -> FrameIndex:
    """An index of frames at these pts, its first frame a keyframe."""
    times = np.array(pts, dtype=np.int64)
    count = len(times)
    keys = times[:1] if count and times[0] != NOPTS else times[:0]
    return FrameIndex(
        time_base, 0, times, np.zeros(count, np.uint32), np.zeros(count, np.bool_), keys
    )


def live2(frames: int) -> list[int]:
    """S9's pts, in ms, as the prep found them on live-2, its first 2,000 frames (DESIGN.md,
    Input; seeking.md, mechanism 7): pts_n = floor(n * 1000/24 + k * 1000/48 + 1/2), k the long
    steps passed, into frames 251, 753, 1251 and 1752: steps of 41 and 42 ms at 24 fps, and one of
    62-63 ms every 500 frames or so, which pulls the timeline back onto 24000/1001."""
    longs = (251, 753, 1251, 1752)
    return [(2000 * n + 1000 * sum(n >= k for k in longs) + 24) // 48 for n in range(frames)]


def test_half_a_frame_is_off() -> None:
    # 25 fps, half a frame 20 ms: a frame 19 ms off has its own place, one 20 ms off, either way,
    # is off; exact integers, and the first frame the anchor, wherever it starts.
    on = rate.timeline(indexed([1000, 1040, 1080 + 19, 1120 - 19, 1160]), Fraction(25))
    assert on.holds and on.off == 0
    assert (on.lowest, on.highest) == (Fraction(-19, 1000), Fraction(19, 1000))
    for by in (20, -20):
        on = rate.timeline(indexed([1000, 1040, 1080 + by, 1120, 1160]), Fraction(25))
        assert not on.holds and (on.first_off, on.off) == (2, 1)
        assert on.first_off_by == Fraction(by, 1000) and on.largest == Fraction(20, 1000)
    # A frame without a pts has no deviation; the anchor is the first frame that has one.
    on = rate.timeline(indexed([NOPTS, 40, 80, NOPTS, 160]), Fraction(25))
    assert on.holds and on.largest == 0


def test_s9s_timestamps_follow_ntsc_film() -> None:
    # live-2's first 2,000 frames: within -10.71 ... +10.88 ms of 24000/1001's timeline (the
    # prep's figures), a quarter of a frame; at 24/1, frame 252 is the first beyond half a frame,
    # 0.504 of one, and they drift 83.7 ms, 2 frames, by frame 1,999. 24000/1001 is the twin of
    # the 24/1 they declare, the rate a refusal names (rate.twin).
    found = rate.twin(indexed(live2(2000)), Fraction(24))
    assert found is not None and found.rate == NTSC_FILM
    assert round(float(found.lowest) * 1000, 2) == -10.71
    assert round(float(found.highest) * 1000, 2) == 10.88
    assert round(float(found.largest * found.rate), 2) == 0.26
    at24 = rate.timeline(indexed(live2(2000)), Fraction(24))
    assert at24.first_off == 252 and at24.first_off_by == Fraction(21, 1000)
    assert at24.highest == Fraction(251, 3000)
    # The simplest fraction within the bounds the span allows is 2997/125, 23.976 exactly, which
    # holds the frames too; not a rate looked for (rate.TWELFTHS, rate.NTSC), nor a twin.
    assert rate.timeline(indexed(live2(2000)), Fraction(2997, 125)).holds
    assert Fraction(2997, 125) not in rate._rates(Fraction(23), Fraction(25))
    # No rate's exactly, their long steps catching up; and the twin of no other rate declared.
    assert rate.exact(indexed(live2(2000))) is None
    assert rate.twin(indexed(live2(2000)), Fraction(25)) is None
    assert rate.twin(indexed(live2(2000)), NTSC_FILM) is None


def test_the_twin_is_named_once_a_frame_from_the_declared_timeline() -> None:
    # Design's decision (2026-10-12): the declared rate's NTSC twin is named when it holds
    # every frame within half a frame "AND the two timelines have parted by a frame or more
    # over the file, so that one late frame of a short file is never read as a rate". The last
    # frame, k frames after the first, has its two places k/1000 of a frame apart for R x
    # 1000/1001: S9's shape is named from 1,001 frames on, and not over 1,000, nor over its
    # first 260, which 24000/1001's timeline holds as well, within 10.8 ms.
    for frames, named in ((260, False), (1000, False), (1001, True), (1002, True)):
        index = indexed(live2(frames))
        on = rate.timeline(index, NTSC_FILM)
        assert on.holds and round(float(on.largest) * 1000, 1) == 10.8
        found = rate.twin(index, Fraction(24))
        assert (found is not None and found.rate == NTSC_FILM) is named
    # k/1001 of a frame for R x 1001/1000, a frame from 1,002 frames on: frames at 30 fps in
    # milliseconds, every other one 5 ms late, declared 30000/1001.
    late = [(2000 * n + 30) // 60 + (5 if n % 2 else 0) for n in range(1002)]
    declares = Fraction(30000, 1001)
    assert rate.timeline(indexed(late[:1001]), Fraction(30)).holds
    assert rate.twin(indexed(late[:1001]), declares) is None
    found = rate.twin(indexed(late), declares)
    assert found is not None and (found.rate, found.largest) == (30, Fraction(16, 3000))
    # One late frame of a short file: 20 ms late in 10 at 25 fps, within 19.8 ms of
    # 25000/1001's timeline, 0.49 of a frame, which half a frame's rule named.
    one_late = indexed([40 * n + (20 if n == 5 else 0) for n in range(10)])
    on = rate.timeline(one_late, Fraction(25000, 1001))
    assert on.holds and round(float(on.largest) * 1000, 1) == 19.8
    assert rate.twin(one_late, Fraction(25)) is None
    # Too few frames to part at all, or none with a pts.
    for pts in ([0], [0, 42], [NOPTS, NOPTS]):
        assert rate.twin(indexed(pts), Fraction(24)) is None


def test_rates_looked_for() -> None:
    # The rates the one a file's timestamps are exactly is looked for among (rate.exact):
    # ffmpeg's own guesses, unbounded: every twelfth of a frame per second, the NTSC forms; the
    # simplest fractions first, the bounds excluded.
    found = rate._rates(Fraction(23), Fraction(25))
    assert found[:3] == [Fraction(24), Fraction(47, 2), Fraction(49, 2)]
    assert found[-2:] == [NTSC_FILM, Fraction(25000, 1001)] and Fraction(25) not in found
    assert Fraction(120000, 1001) in rate._rates(Fraction(119), Fraction(120))
    assert Fraction(100) in rate._rates(Fraction(99), Fraction(101))
    assert Fraction(25, 2) in rate._rates(Fraction(12), Fraction(13))


def test_no_constant_rate_in_a_variable_one() -> None:
    # 12 frames at 24 fps, then 12 at 30, sptenc's mix (test_decode.py's), or random steps: no
    # rate's exactly, nor the twin of the rate declared, whichever it is: sptenc's refusal, as
    # it was.
    mixed = [round(1000 * (n / 24 if n < 12 else 0.5 + (n - 12) / 30)) for n in range(24)]
    steps = random.Random(9).choices([33, 42, 50], k=200)
    for pts, lasting in ((mixed, (33, 42)), ([sum(steps[:n]) for n in range(200)], (33, 50))):
        assert rate.exact(indexed(pts)) is None
        scanned = scanned_at(np.array(pts, dtype=np.int64), MS)
        for declares in (Fraction(24), Fraction(25), Fraction(30), NTSC_FILM):
            assert rate.twin(indexed(pts), declares) is None
            assert timing_refusal(declaring(declares), scanned) == variable(*lasting)
    # Too few frames to tell one rate from the next, or a single frame: none.
    assert rate.exact(indexed([0, 1, 3])) is None and rate.twin(indexed([0, 1]), NTSC_FILM) is None
    assert rate.exact(indexed([0])) is None and rate.twin(indexed([0]), NTSC_FILM) is None


def test_jitter_around_the_rate_declared_is_named_no_rate() -> None:
    # Steps of 50 and 30 ms around 25 fps, every other frame 10 ms late: 10 ms off 25's timeline,
    # a quarter of a frame, and 9.96 off 25000/1001's, which drifts 0.04 ms a frame. Half a
    # frame's rule named them the rate they declare, the steadier of the two, with --frame-rate
    # 25 as a fix; no rate's exactly, and their twin's timeline 0.1 of a frame from the declared
    # one's at their end: sptenc's refusal as it is, no rate named, no --frame-rate offered
    # (design's decision, 2026-10-12). The option still takes them at 25, R's timeline holding
    # each within half a frame (DESIGN.md, Input).
    late = [40 * n + (10 if n % 2 else 0) for n in range(100)]
    on = rate.timeline(indexed(late), Fraction(25))
    assert on.holds and on.largest == Fraction(1, 100)
    other = rate.timeline(indexed(late), Fraction(25000, 1001))
    assert other.holds and other.largest < on.largest
    assert rate.exact(indexed(late)) is None and rate.twin(indexed(late), Fraction(25)) is None
    scanned = scanned_at(np.array(late, dtype=np.int64), MS)
    assert timing_refusal(declaring(Fraction(25)), scanned) == variable(30, 50)
    assert timing_refusal(declaring(Fraction(25), Fraction(25)), scanned) == ""
    # Over 1,100 frames the twin's timeline is a frame away and more, and no longer holds them.
    longer = indexed([40 * n + (10 if n % 2 else 0) for n in range(1100)])
    assert not rate.timeline(longer, Fraction(25000, 1001)).holds
    assert rate.twin(longer, Fraction(25)) is None


def test_exact_beyond_int64() -> None:
    # A time base whose numerators would overflow int64: Python's integers, the same verdicts.
    base = Fraction(1, 2**40)
    ticks = [round(Fraction(n * 1001, 24000) / base) for n in range(0, 200_000, 997)]
    places = list(range(0, 200_000, 997))
    times = np.full(200_000, NOPTS, dtype=np.int64)
    times[places] = ticks
    index = FrameIndex(
        base, 0, times, np.zeros(200_000, np.uint32), np.zeros(200_000, np.bool_), times[:1]
    )
    on = rate.timeline(index, NTSC_FILM)
    assert on.holds and on.largest < Fraction(1, 2**40)


def test_remake_command() -> None:
    # The first video stream renumbered at the rate from 0 (-r:v:0), the other streams copied, the
    # data streams left out (Matroska refuses them); its first frame's time kept when it isn't 0.
    command = rate.remake(NTSC_FILM, Fraction(0))
    assert command == (
        "ffmpeg -r:v:0 24000/1001 -i SOURCE -map 0 -map -0:d -c copy -c:v:0 ffv1 -level:v:0 3"
        " -g:v:0 1 -slicecrc:v:0 1 -fps_mode:v:0 passthrough remade.mkv"
    )
    later = rate.remake(NTSC_FILM, Fraction(23, 1000))
    assert later.endswith(
        " -filter:v:0 settb=AVTB,setpts=PTS+23000 -enc_time_base:v:0 demux remade.mkv"
    )
    assert rate.said(NTSC_FILM) == "24000/1001 fps (23.976)"
    assert rate.said(Fraction(25)) == "25/1 fps (25)"
    assert rate.said(Fraction(30000, 1001)) == "30000/1001 fps (29.97)"


def run(*arguments: str) -> bytes:
    result = subprocess.run(["ffmpeg", "-v", "error", "-y", *arguments], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()
    return result.stdout


def mechanism7(
    path: Path, size: str = "160x90", frames: int = 260, late: int = 0, audio: bool = False
) -> Path:
    """seeking.md's mechanism 7 made on purpose, as the prep reproduced S9's timestamps with
    ffmpeg n9.0.2 and libx264 (DESIGN.md, Input): declared 24/1 (Matroska's default duration,
    41.666 ms), its pts live-2's (live2), in ms, as settb=1/1000 keeps them (without it they round
    to 1/24 s and the long steps are lost); `late` ms more on each, its video then starting after
    its audio; with `audio`, a FLAC sine from 0, longer than the video."""
    longs = "+".join(f"gte(N\\,{n})" for n in (251, 753, 1251, 1752))
    times = f"settb=1/1000,setpts='floor((2000*N+1000*({longs})+24)/48)+{late}'"
    sound = (
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:sample_rate=48000:duration={frames // 24 + 1}",
    )
    run(
        *("-f", "lavfi", "-i", f"testsrc2=s={size}:r=24", *(sound if audio else ())),
        *("-frames:v", str(frames), "-filter:v", times, "-map", "0:v"),
        *(("-map", "1:a", "-c:a", "flac") if audio else ()),
        *("-fps_mode:v", "passthrough", "-enc_time_base:v", "1/1000"),
        *("-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-x264-params"),
        "keyint=18:min-keyint=18:bframes=2:scenecut=0:log-level=error",
        str(path),
    )
    return path


def streams(path: Path) -> list[dict[str, str]]:
    probed = subprocess.run(
        [
            *("ffprobe", "-v", "error", "-show_entries"),
            *("stream=codec_type,r_frame_rate,avg_frame_rate,start_time", "-of", "json", str(path)),
        ],
        capture_output=True,
        check=True,
    ).stdout
    return json.loads(probed)["streams"]


def pictures(path: Path) -> list[str]:
    """The MD5 of each frame of path's first video stream, decoded, in order (framemd5)."""
    output = run("-i", str(path), "-map", "0:v:0", "-f", "framemd5", "-").decode()
    return [line.split(",")[5].strip() for line in output.splitlines() if line[:1] == "0"]


# The frames from which S9's shape is guided: its last frame's places on 24/1's timeline and on
# 24000/1001's a frame apart (rate.twin). mechanism7's own 260 are a quarter of one apart.
GUIDED = 1001

GUIDANCE = (
    "its timestamps run at 24000/1001 fps (23.976), within 10.8 ms of that rate's timeline (0.26"
    " of a frame), where it declares 24/1, its frames lasting from 41 to 63 ms: a bad file, not"
    " read on a guess: remake it at that rate, an FFV1 master with its other streams copied, a"
    " clean file for every tool (lossless, so large: 0.53-0.55 MB a frame on a 1080p film; and a"
    " full encode), with ffmpeg -r:v:0 24000/1001 -i SOURCE -map 0 -map -0:d -c copy -c:v:0 ffv1"
    " -level:v:0 3 -g:v:0 1 -slicecrc:v:0 1 -fps_mode:v:0 passthrough remade.mkv; or give"
    " --frame-rate 24000/1001, which takes its frames at that rate, each checked to lie within"
    " half a frame of its timeline: the same frames on the same timeline, without the"
    " intermediate file"
)


@needs_ffmpeg
@needs_x264
def test_contradicted_rate_refused_with_guidance(tmp_path: Path) -> None:
    # Mechanism 7: refused, as sptenc's rule refuses its 62 and 63 ms steps, the message giving
    # the rate its timestamps follow, how far, and the two fixes (DESIGN.md, Input), whole: over
    # 1,001 frames, two long steps in them, the twin of the rate declared a frame away.
    path = mechanism7(tmp_path / "s9.mkv", frames=GUIDED)
    with pytest.raises(MediaError) as refused:
        examine(path)
    assert str(refused.value) == f"{path}: {GUIDANCE}"
    # Its first 260 frames, one long step in them, 24000/1001's timeline holding them as well
    # and 24/1's a quarter of a frame from it: sptenc's refusal, no rate named, no fix (design's
    # decision, 2026-10-12: a twin that has not parted by a frame is not named).
    short = mechanism7(tmp_path / "short.mkv")
    with pytest.raises(MediaError) as refused:
        examine(short)
    assert str(refused.value) == f"{short}: {variable(41, 62)}"


@needs_ffmpeg
@needs_x264
def test_frame_rate_override(tmp_path: Path) -> None:
    # --frame-rate 24000/1001: every frame within half a frame of its timeline, taken at that
    # rate, the job's from then on; 24/1: frame 252 named, 0.504 of a frame after its place, and
    # the rate its timestamps run at given with the fixes.
    path = mechanism7(tmp_path / "s9.mkv", frames=GUIDED)
    source = examine(path, frame_rate=NTSC_FILM)
    assert (source.frames, source.frame_rate, source.stream.frame_rate) == (GUIDED, NTSC_FILM, 24)
    with pytest.raises(MediaError) as refused:
        examine(path, frame_rate=Fraction(24))
    said = str(refused.value)
    assert said.startswith(
        f"{path}: --frame-rate 24/1: frame 252 lies 21.0 ms after its place on that rate's"
        " timeline (0.504 of a frame, half a frame being 20.8 ms), and 581 more frames, up to"
        " 42 ms: taken at that rate, frames would be dropped or doubled; its timestamps run at"
        " 24000/1001 fps (23.976), within 10.8 ms of that rate's timeline (0.26 of a frame):"
        " remake it at that rate"
    )
    assert said.endswith(
        "or give --frame-rate 24000/1001, which takes its frames at that rate,"
        " each checked to lie within half a frame of its timeline: the same"
        " frames on the same timeline, without the intermediate file"
    )
    # Its first 260 frames: taken at 24000/1001 alike; 24/1 refused, and no rate named after,
    # none being named without the option (test_contradicted_rate_refused_with_guidance).
    short = mechanism7(tmp_path / "short.mkv")
    assert examine(short, frame_rate=NTSC_FILM).frames == 260
    with pytest.raises(MediaError) as refused:
        examine(short, frame_rate=Fraction(24))
    assert str(refused.value) == (
        f"{short}: --frame-rate 24/1: frame 252 lies 21.0 ms after its place on that rate's"
        " timeline (0.504 of a frame, half a frame being 20.8 ms), and 5 more frames, up to"
        f" 21.3 ms: {DROPPED_OR_DOUBLED}"
    )


def test_no_rate_named_by_half_a_frame() -> None:
    # Design's decision (2026-10-12): "a refusal names another rate only (a) when the timestamps
    # are that rate's EXACTLY, to the time base's rounding, or (b) where sptenc's rule fails,
    # when it is the DECLARED RATE'S NTSC TWIN ... Everything else that sptenc's rule refuses
    # and that isn't frames missing on the grid gets the plain variable-frame-rate refusal, no
    # rate named, no --frame-rate offered." Each file below was named one of ffmpeg's standard
    # rates, the one that held its frames within half a frame, by its length, with --frame-rate
    # at it as a fix, which then read it (a review's finding and design's check of it,
    # 2026-10-12); each with its refusal now, whole, and --frame-rate's at the rate it declares,
    # whole, no rate after it.

    def at(pts: Sequence[int], time_base: Fraction = MS) -> Scan:
        return scanned_at(np.array(pts, dtype=np.int64), time_base)

    segment = [(2002 * n + 24) // 48 for n in range(24)]
    cases: list[tuple[str, Fraction, Scan, str, str]] = [
        # 8 segments of 24 frames at 24000/1001 joined by ffmpeg's concat demuxer, their audio
        # outlasting them: the video steps 63 ms at each of the 7 joins. "47/2 fps (23.5),
        # within 19.7 ms".
        (
            "47/2",
            NTSC_FILM,
            at([1022 * k + pts for k in range(8) for pts in segment]),
            variable(41, 63),
            "--frame-rate 24000/1001: frame 24 lies 21.0 ms after its place on that rate's"
            " timeline (0.503 of a frame, half a frame being 20.9 ms), and 159 more frames, up"
            " to 147.5 ms",
        ),
        # 7 frames at 24 fps, one step of 1.5 frames. "85/4 fps (21.25)".
        (
            "85/4",
            Fraction(24),
            stepping([42, 41, 63, 42, 41, 42]),
            variable(41, 63),
            "--frame-rate 24/1: frame 3 lies 21.0 ms after its place on that rate's timeline"
            " (0.504 of a frame, half a frame being 20.8 ms), and 2 more frames, up to 21.3 ms",
        ),
        # 5 segments of 50 frames at 25 fps, each 64 ms after the last frame of the one before.
        # "99/4 fps (24.75), within 19.8 ms".
        (
            "99/4",
            Fraction(25),
            stepping(([40] * 49 + [64]) * 4 + [40] * 49),
            variable(40, 64),
            "--frame-rate 25/1: frame 50 lies 24.0 ms after its place on that rate's timeline"
            " (0.600 of a frame, half a frame being 20.0 ms), and 199 more frames, up to 96 ms",
        ),
        # 100 frames at 24 fps in milliseconds, the 50 last 21 ms late, S9's catch-up step
        # once. "143/6 fps (23.833), within 14 ms".
        (
            "143/6",
            Fraction(24),
            at([(2000 * n + 24) // 48 + (21 if n >= 50 else 0) for n in range(100)]),
            variable(41, 62),
            "--frame-rate 24/1: frame 51 lies 21.0 ms after its place on that rate's timeline"
            " (0.504 of a frame, half a frame being 20.8 ms), and 32 more frames, up to 21.3 ms",
        ),
        # A short file: 6 frames at 12 fps declared 24/1, one missing, in ticks of 1/24 s.
        # "109/12 fps (9.083), within 53.5 ms".
        (
            "109/12",
            Fraction(24),
            at([0, 2, 4, 8, 10, 12], Fraction(1, 24)),
            "variable frame rate: frames last from 83.3333 to 166.667 ms, and only a constant"
            " frame rate is supported",
            "--frame-rate 24/1: frame 1 lies 41.7 ms after its place on that rate's timeline"
            " (1.000 of a frame, half a frame being 20.8 ms), and 4 more frames, up to 291.7 ms",
        ),
        # One frame 20 ms late in 10 at 25 fps. "25000/1001 fps (24.975), within 19.8 ms".
        (
            "25000/1001",
            Fraction(25),
            at([40 * n + (20 if n == 5 else 0) for n in range(10)]),
            variable(20, 60),
            "--frame-rate 25/1: frame 5 lies 20.0 ms after its place on that rate's timeline"
            " (0.500 of a frame, half a frame being 20.0 ms)",
        ),
    ]
    for was, declares, scanned, plain, given in cases:
        assert unfilled(declaring(declares), scanned) == plain, was
        said = unfilled(declaring(declares, declares), scanned)
        assert said == f"{given}: {DROPPED_OR_DOUBLED}", was
        # The rate once advised still holds every frame within half a frame, and --frame-rate
        # takes what it is given by that bound (DESIGN.md, Input): no refusal sends there now.
        index = scanned.index
        assert index is not None and rate.timeline(index, Fraction(was)).holds
        assert timing_refusal(declaring(declares, Fraction(was)), scanned) == ""
    # S9's shape is still guided, word for word, from the frame where 24000/1001's timeline is
    # a frame from 24/1's; a frame less, a twin that holds but has not parted by a frame, gets
    # sptenc's refusal.
    s9 = live2(GUIDED)
    film = declaring(Fraction(24))
    assert unfilled(film, at(s9)) == GUIDANCE
    assert unfilled(film, at(s9[:-1])) == variable(41, 63)
    assert unfilled(film, at(s9[:260])) == variable(41, 62)
    # --frame-rate's refusal names the twin of the rate DECLARED, what the file is, never R's:
    # S9's shape given 25 is told 24000/1001; declaring 25/1 and given 24/1, whose twin holds
    # it, no rate, as without the option.
    said = unfilled(declaring(Fraction(24), Fraction(25)), at(s9))
    assert said.startswith("--frame-rate 25/1: frame 12 lies 20.0 ms after its place on")
    assert said.endswith(
        f"{DROPPED_OR_DOUBLED}; its timestamps run at 24000/1001 fps (23.976), within 10.8 ms of"
        f" that rate's timeline (0.26 of a frame): {fixes(NTSC_FILM)}"
    )
    pal = declaring(Fraction(25))
    assert unfilled(pal, at(s9)) == variable(41, 63)
    assert unfilled(replace(pal, rate_override=Fraction(24)), at(s9)) == (
        "--frame-rate 24/1: frame 252 lies 21.0 ms after its place on that rate's timeline (0.504"
        " of a frame, half a frame being 20.8 ms), and 581 more frames, up to 42 ms:"
        f" {DROPPED_OR_DOUBLED}"
    )
    # What (a) still names where sptenc's rule fails, a rate the timestamps are exactly: 12 fps
    # declared 24/1, in ticks of 1/24 s; and the rate declared itself, in a time base too coarse
    # for the rule's 1 ms, 24 fps in ticks of 10 ms, which --frame-rate 24/1 reads.
    said = unfilled(film, at(range(0, 14, 2), Fraction(1, 24)))
    assert said == (
        "its timestamps run at 12/1 fps (12), within 0 ms of that rate's timeline (0.00 of a"
        " frame), where it declares 24/1, its frames lasting from 83.3 to 83.3 ms: a bad file,"
        f" not read on a guess: {fixes(Fraction(12))}"
    )
    coarse = at([(100 * n + 12) // 24 for n in range(300)], Fraction(1, 100))
    assert unfilled(film, coarse) == (
        "its timestamps follow 24/1 fps (24), the rate it declares, within 5 ms of that rate's"
        " timeline (0.12 of a frame), but its frames last from 40 to 50 ms, where a constant"
        " rate's lie within 1 ms of each other: a bad file, not read on a guess:"
        f" {fixes(Fraction(24))}"
    )
    assert timing_refusal(declaring(Fraction(24), Fraction(24)), coarse) == ""


@needs_ffmpeg
def test_frame_rate_on_a_file_already_fine(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # 48 frames at 24000/1001, declared so: --frame-rate 24000/1001 accepted, said; 24 too, its
    # frames 2 ms off 24's timeline at most, taken at 24 as asked, warned of, the output 0.1%
    # shorter (media/source.py: accepted by design, 2026-10-12); 25, 1.67 ms a frame off, refused
    # from frame 12.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    path = tmp_path / "ntsc.mkv"
    run(
        "-f",
        "lavfi",
        "-i",
        "testsrc2=s=64x48:r=24000/1001",
        "-frames:v",
        "48",
        "-c:v",
        "ffv1",
        str(path),
    )
    assert examine(path, frame_rate=NTSC_FILM).frame_rate == NTSC_FILM
    said = "--frame-rate 24000/1001: every frame within 0.5 ms of its timeline (0.01 of a frame),"
    assert f"{said} the rate it declares: taken at that rate" in caplog.text
    assert "too: taken" not in caplog.text
    assert examine(path, frame_rate=Fraction(24)).frame_rate == 24
    assert (
        "ntsc.mkv: its frames are at the rate it declares, 24000/1001, too: taken at 24/1 as asked,"
        " the output lasting 0.100% shorter than the source, off its other streams"
    ) in caplog.text
    with pytest.raises(MediaError) as refused:
        examine(path, frame_rate=Fraction(25))
    # Told that it is read without the option, its frames on the declared rate's timeline, and
    # no other rate named: its timestamps are the declared one's.
    assert str(refused.value) == (
        f"{path}: --frame-rate 25/1: frame 12 lies 21.0 ms after its place on that rate's"
        " timeline (0.525 of a frame, half a frame being 20.0 ms), and 35 more frames, up to 80"
        f" ms: {read_without('24000/1001')}"
    )


@needs_ffmpeg
@needs_x264
@pytest.mark.parametrize("late", [0, 23])
def test_remake_keeps_every_frame(tmp_path: Path, late: int) -> None:
    # The refusal's command, run as given: every frame, in order, none dropped or doubled (their
    # decoded pictures' MD5s), declared 24000/1001, the audio where it was, the first frame too,
    # 23 ms after the audio's start (where fps=24000/1001 alone dropped a frame and doubled
    # another, rate.remake); and seedvr2x reads the result.
    path = mechanism7(tmp_path / "s9.mkv", frames=GUIDED, late=late, audio=True)
    with pytest.raises(MediaError) as refused:
        examine(path)
    said = str(refused.value)
    command = said[said.index("ffmpeg -r:v:0") : said.index(" remade.mkv") + len(" remade.mkv")]
    words = shlex.split(command)
    words[words.index("SOURCE")] = str(path)
    words[-1] = str(tmp_path / "remade.mkv")
    run(*words[1:])
    remade = tmp_path / "remade.mkv"
    assert pictures(remade) == pictures(path) and len(pictures(path)) == GUIDED
    before, after = streams(path), streams(remade)
    assert [s["codec_type"] for s in after] == ["video", "audio"]
    assert after[0]["r_frame_rate"] == after[0]["avg_frame_rate"] == "24000/1001"
    assert after[0]["start_time"] == before[0]["start_time"] == f"0.{late:03d}000"
    assert after[1]["start_time"] == before[1]["start_time"] == "0.000000"
    source = examine(remade)
    assert (source.frames, source.frame_rate) == (GUIDED, NTSC_FILM)


def read_without(declares: str) -> str:
    """The end of --frame-rate's refusal for a file read without it: one sptenc's rule passes at
    the rate it declares, its frames on that rate's timeline."""
    return (
        "taken at that rate, frames would be dropped or doubled; without --frame-rate, the file"
        f" is read at the rate it declares, {declares}, whose timeline holds every frame within"
        " half a frame"
    )


REFUSED_WITHOUT = (
    "taken at that rate, frames would be dropped or doubled; without --frame-rate, the file is"
    " refused too, that rate being the one it declares, and"
)


def fixes(found: Fraction) -> str:
    """The two fixes a refusal gives for timestamps that run at `found`, the first frame at the
    file's start."""
    return (
        "remake it at that rate, an FFV1 master with its other streams copied, a clean file for"
        " every tool (lossless, so large: 0.53-0.55 MB a frame on a 1080p film; and a full"
        f" encode), with {rate.remake(found, Fraction(0))}; or give --frame-rate"
        f" {rate.fraction(found)}, which takes its frames at that rate, each checked to lie within"
        " half a frame of its timeline: the same frames on the same timeline, without the"
        " intermediate file"
    )


def retagged(path: Path, declared: int, times: str, frames: int) -> Path:
    """`frames` frames declared `declared` fps (Matroska's default duration), their pts, in ms,
    the expression `times` of the frame's number N (setpts)."""
    run(
        *("-f", "lavfi", "-i", f"testsrc2=s=64x48:r={declared}", "-frames:v", str(frames)),
        *("-vf", f"settb=1/1000,setpts='{times}'", "-fps_mode", "passthrough"),
        *("-enc_time_base", "1/1000", "-c:v", "ffv1", str(path)),
    )
    return path


def test_the_rate_timestamps_are_exactly() -> None:
    # Timestamps in milliseconds, as Matroska holds them. A rate's times rounded to them spread
    # over one tick at most on its timeline: 24000/1001 and 48 fps, whatever the length, though
    # 24/1 and 50/1 hold the first 492 and 12 frames within half a frame. Rounded as ffmpeg
    # rounds, a tie up, 0.96 of a tick; a tie to the even tick, as Python's round, the whole
    # tick, still that rate's.
    for frames in (600, 3000):
        found = rate.exact(indexed([(2002 * n + 24) // 48 for n in range(frames)]))
        assert found is not None and found.rate == NTSC_FILM
        assert found.highest - found.lowest == Fraction(23, 24000)
        found = rate.exact(indexed([round(n * 1001 / 24) for n in range(frames)]))
        assert found is not None and found.rate == NTSC_FILM
        assert found.highest - found.lowest == Fraction(1, 1000)
        found = rate.exact(indexed([(2000 * n + 48) // 96 for n in range(frames)]))
        assert found is not None and found.rate == 48
    # A constant step is its own mean rate's, whatever the rate and the length: 33 ms a frame,
    # 1000/33 fps, where half a frame gave 91/3 over 120 and 300 frames and none over 3,000; 17 ms
    # a frame, 1000/17, though 353/6, a rate looked for, holds 300 frames within 0.85 ms too.
    for frames in (120, 300, 3000):
        steady = indexed([33 * n for n in range(frames)])
        found = rate.exact(steady)
        assert found is not None and (found.rate, found.largest) == (Fraction(1000, 33), 0)
        assert rate.timeline(steady, Fraction(91, 3)).holds is (frames < 3000)
        assert rate.twin(steady, Fraction(30)) is None
    seventeen = indexed([17 * n for n in range(300)])
    found = rate.exact(seventeen)
    assert found is not None and found.rate == Fraction(1000, 17)
    looked_for = rate.timeline(seventeen, Fraction(353, 6))
    assert looked_for.holds and looked_for.highest - looked_for.lowest < Fraction(1, 1000)
    # In a time base that holds the rate, every frame on its tick.
    ticks = indexed([1001 * n for n in range(100)], Fraction(1, 24000))
    found = rate.exact(ticks)
    assert found is not None and (found.rate, found.largest) == (NTSC_FILM, 0)
    # Segments of 24 frames at 24000/1001 joined as ffmpeg's concat demuxer joins them, each
    # starting 1,000 ms after the one before, 1 ms early: 24/1 holds them within 1.33 ms, the
    # rate half a frame's rule named, their deviations spread over 1.67 ticks; no rate's exactly.
    # 24/1 is the twin of the 24000/1001 they declare, 2.4 frames away at their end: named
    # nowhere, sptenc's rule passing their steps of 41 and 42 ms, the declared timeline's rule
    # refusing them as no rate's (source._leaves; design's decision, 2026-10-12: the twin is
    # named "where sptenc's rule fails").
    segment = [(2002 * n + 24) // 48 for n in range(24)]
    joined = indexed([1000 * k + pts for k in range(100) for pts in segment])
    half = rate.timeline(joined, Fraction(24))
    assert half.holds and half.largest == Fraction(4, 3000)
    assert half.highest - half.lowest == Fraction(5, 3000)
    assert rate.exact(joined) is None
    found = rate.twin(joined, NTSC_FILM)
    assert found is not None and found.rate == 24
    said = timing_refusal(declaring(NTSC_FILM), scanned_at(joined.pts, MS))
    assert said.endswith(f", and {no_rate(NTSC_FILM)}") and "24/1" not in said
    # A rate that is none of those looked for, rounded to ticks: found when its mean rate's
    # timeline holds the frames so, 41.5 ms a frame, 2000/83 fps, over 101 frames, each within
    # half a tick; not always (rate.exact): 24.03 fps over 1,000 frames, 1.03 ticks on its mean
    # rate's timeline.
    found = rate.exact(indexed([(83 * n + 1) // 2 for n in range(101)]))
    assert found is not None and (found.rate, found.largest) == (
        Fraction(2000, 83),
        Fraction(1, 2000),
    )
    other = indexed([round(n * 1000 / 24.03) for n in range(1000)])
    assert rate.exact(other) is None
    mean = rate.timeline(other, Fraction(999_000, 41_573))
    assert 1 < (mean.highest - mean.lowest) * 1000 < 1.03
    # Too little to tell: one frame, frames without a pts or on one tick, a span of a few ticks
    # that isn't a constant step, which two frames always are; of two ticks, where the rates
    # that put the last frame within two of its place have no bound (a division by zero).
    for pts in ([0], [NOPTS, NOPTS], [7, 7, 7], [0, 1, 3], [0, 0, 2]):
        assert rate.exact(indexed(pts)) is None
    found = rate.exact(indexed([5, 6]))
    assert found is not None and found.rate == 1000


def test_no_rate_named_when_several_looked_for_hold_the_frames() -> None:
    # A short file's timestamps are several rates' exactly: 24/1 holds the first 12 frames of
    # 24000/1001 within a tick's spread too, in milliseconds, 48/1 or 575/12 the first 18 of
    # 48000/1001, 120/1 the first 60 of 120000/1001. None is named then, where the one the
    # frames lay steadiest on was, a neighbour of their rate for 222 of a review's 12,872
    # synthetic files, all under 83 frames (here 24/1 for 5 frames of 24000/1001, 719/6 for 41
    # of 120000/1001); a frame more, one rate alone holds them, named.
    tick = Fraction(1, 1000)
    cases = (
        (NTSC_FILM, 12, [Fraction(24), NTSC_FILM]),
        (Fraction(48000, 1001), 18, [Fraction(575, 12), Fraction(48000, 1001)]),
        (Fraction(120000, 1001), 60, [Fraction(120), Fraction(1439, 12), Fraction(120000, 1001)]),
    )
    for true, short, several in cases:
        times = [int(n * 1000 / true + Fraction(1, 2)) for n in range(short + 1)]
        index = indexed(times[:short])
        fitting = [
            found.rate
            for looked_for in rate._rates(true - 2, true + 2)
            if rate._rounded(found := rate.timeline(index, looked_for), tick)
        ]
        assert fitting == several and rate.exact(index) is None
        found = rate.exact(indexed(times))
        assert found is not None and found.rate == true
    # Nor is the frames' mean rate, tried last, given in their place: 48/1 for 13 frames of
    # 48000/1001, which it holds so as well.
    thirteen = indexed([int(n * 1001 / 48 + Fraction(1, 2)) for n in range(13)])
    assert rate._rounded(rate.timeline(thirteen, Fraction(48)), tick)
    assert rate.exact(thirteen) is None
    # A constant step is still its own rate's, however short: 3 frames 21 ms apart.
    found = rate.exact(indexed([0, 21, 42]))
    assert found is not None and found.rate == Fraction(1000, 21)


def strayed(caplog: pytest.LogCaptureFixture) -> list[str]:
    """What is logged, from a warning up, of frames half a frame off a rate's timeline: nothing,
    since a source whose frames leave its declared rate's timeline is refused (media/source.py,
    _leaves), where it was read and warned of."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING and "half a frame" in record.getMessage()
    ]


def leaves(count: str, first_off: str, declares: str) -> str:
    """How a refusal says a file's frames leave the timeline of the rate it declares: `count` of
    its frames, "100 of its 600", from `first_off`, the first one and the furthest, "492, up to
    25.3 ms (0.61"."""
    return (
        f"{count} frames lie half a frame or more from their place on the timeline of the rate it"
        f" declares, {declares}, from frame {first_off} of a frame)"
    )


def another_rate(found: Fraction, change: str, tick: str = "1/1000") -> str:
    """The end of the refusal of a file whose timestamps are the rate `found`'s exactly, in
    `tick` s, and not the one it declares: what the declared one would have cost, and the two
    fixes."""
    return (
        f"its output, written at that rate, would be {change} than its timestamps' own time, off"
        f" the source's other streams: its timestamps follow {rate.said(found)} exactly, to the"
        f" rounding of its time base, {tick} s: a bad file, not read on a guess: {fixes(found)}"
    )


def no_rate(declares: Fraction) -> str:
    """The end of the refusal of a file whose timestamps are no one rate's exactly, as a join by
    ffmpeg's concat demuxer is: its ways out a join on the frame grid (rate.grid_join) and the
    remake at the rate it declares, the first frame at the file's start; never --frame-rate."""
    return (
        "its timestamps follow no one rate exactly: a file joined by ffmpeg's concat demuxer"
        " alone drifts so, up to 1 ms a join, and timestamps alone can't tell a drifting join"
        " from a retagged file: a bad file, not read on a guess: join its segments again so that"
        f" every frame keeps its place on the frame grid, {rate.grid_join(declares=declares)};"
        " or remake it at the rate it declares, an FFV1 master numbering its frames at that rate"
        " as they come, every one kept, its other streams copied (lossless, so large: 0.53-0.55"
        " MB a frame on a 1080p film; and a full encode), with"
        f" {rate.remake(declares, Fraction(0))}"
    )


def test_the_join_that_keeps_the_frame_grid() -> None:
    # The advice of a join on the frame grid, in one place (rate.grid_join), whole: by ffmpeg
    # alone first, seedvr2x never needing sptenc, a shell's lines listing each segment with its
    # exact duration and one stream copy snapping every timestamp; then sptenc's concat; then,
    # where no stream copy holds, the remake of the segments' pictures. With the segments'
    # directory and their rate when the refusal knows them, DIR and 24000/1001, said to be an
    # example, when it doesn't. After each command that lists the segments by the shell's
    # *.mkv, what that asks of their names: name order their playing order, and no apostrophe
    # (test_the_join_advised_takes_the_segments_in_name_order). What the commands give is
    # test_the_join_advised_keeps_every_frame_on_the_grid's.
    snapped = "'if(eq({0},NOPTS),NOPTS,round(round({0}*TB*$N/$D)*$D/TB/$N))'"
    listed = (
        'for f in *.mkv; do echo "file \'$f\'"; echo "duration $(ffprobe -v error'
        " -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0"
        ' "$f" | awk -v n=$N -v d=$D \'NR==1{printf "%.6f", $1*d/n}\')"; done > list.txt'
    )
    names = (
        "*.mkv the segments' names, taken in name order, which must be the order they play in:"
        " numbers with leading zeros, as seg_000012.mkv, seg_10.mkv coming before seg_2.mkv"
        " otherwise; check list.txt, or write it by hand; a name must hold no apostrophe"
    )
    assert rate.grid_join() == (
        "with ffmpeg alone, each segment listed with its exact duration, its frames over the"
        " rate, and every timestamp snapped to the grid in one stream copy: cd DIR && N=24000"
        f" D=1001 && {listed} && ffmpeg -f concat -safe 0 -i list.txt -c copy -bsf:v"
        f' "setts=pts={snapped.format("PTS")}:dts={snapped.format("DTS")}" ../joined.mkv'
        f" ({names}; N/D their frame rate, 24000/1001 here), or with sptenc,"
        " for segments of its split, sptenc concat DIR joined.mkv, which snaps every timestamp"
        " to the grid, or, where no stream copy holds, for segments that carry audio, MPEG"
        " program streams, or transport streams of a single frame, their pictures remade as an"
        " FFV1 master, numbered at that rate as they come, every one kept: cd DIR && for f in"
        " *.mkv; do echo \"file '$f'\"; done > list.txt && ffmpeg -f concat -safe 0 -r"
        " 24000/1001 -i list.txt -map 0:v:0 -c:v:0 ffv1 -level:v:0 3 -g:v:0 1 -slicecrc:v:0 1"
        f" -fps_mode:v:0 passthrough ../remade.mkv ({names})"
    )
    known = rate.grid_join(shlex.quote("my segments"), Fraction(30000, 1001))
    assert "sptenc concat 'my segments' joined.mkv" in known
    assert known.count("cd 'my segments' && ") == 2 and "DIR" not in known
    assert "cd 'my segments' && N=30000 D=1001 && for f in" in known
    assert "N/D their frame rate), or with sptenc" in known and " here" not in known
    assert known.count(f"({names}") == 2 and known.endswith(f"../remade.mkv ({names})")
    assert "-safe 0 -r 30000/1001 -i list.txt -map 0:v:0" in known and "24000" not in known
    assert "N=25 D=1 &&" in rate.grid_join(declares=Fraction(25))


@needs_ffmpeg
@pytest.mark.parametrize(
    ("declared", "times", "frames", "found", "strays", "first_off", "change"),
    [
        # 24000/1001 declared 24/1: each frame lasts 41 or 42 ms, within 1 ms of 24's 41.667, and
        # lies 0.042 ms further from 24's timeline than the one before, half a frame at 492.
        (
            24,
            "round(N*1001/24)",
            600,
            NTSC_FILM,
            "100 of its 600",
            "492, up to 25.3 ms (0.61",
            "0.100% shorter",
        ),
        # 48 fps declared 50/1: 20 or 21 ms for 20, 4%, half a frame off by frame 12.
        (
            50,
            "round(N*1000/48)",
            100,
            Fraction(48),
            "88 of its 100",
            "12, up to 83 ms (4.15",
            "4.000% shorter",
        ),
        # 33 ms a frame declared 30/1, 1000/33 fps, 1% off, a rate not looked for: half a frame
        # off by frame 50, whatever the length (refused over 120 and 300 frames as running at
        # 91/3 fps, which held them within 3.9 and 9.9 ms, and read over 3,000, said to follow
        # no constant rate).
        (
            30,
            "33*N",
            120,
            Fraction(1000, 33),
            "70 of its 120",
            "50, up to 39.7 ms (1.19",
            "1.010% longer",
        ),
        (
            30,
            "33*N",
            300,
            Fraction(1000, 33),
            "250 of its 300",
            "50, up to 99.7 ms (2.99",
            "1.010% longer",
        ),
        (
            30,
            "33*N",
            3000,
            Fraction(1000, 33),
            "2950 of its 3000",
            "50, up to 999.7 ms (29.99",
            "1.010% longer",
        ),
        # 17 ms a frame declared 60/1: 1000/17, not the 353/6 that holds 300 frames too.
        (
            60,
            "17*N",
            300,
            Fraction(1000, 17),
            "275 of its 300",
            "25, up to 99.7 ms (5.98",
            "1.961% shorter",
        ),
    ],
)
def test_a_rate_contradicted_within_a_millisecond_refused_naming_the_rate(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    declared: int,
    times: str,
    frames: int,
    found: Fraction,
    strays: str,
    first_off: str,
    change: str,
) -> None:
    # Timestamps at one rate, another declared, each frame lasting within 1 ms of the declared
    # rate's: sptenc's rule passes, and the file was read, its output written at the declared
    # rate, 0.1% to 4% off its frames' own time. Its frames leave the declared rate's timeline:
    # refused (source._leaves; DESIGN.md, Input), the message whole: how many frames, from
    # which, how far, what the declared rate would have cost, the rate its timestamps are
    # exactly and both fixes. Given that rate, read at it, nothing warned, nor of a file whose
    # frames lie on its declared timeline too; given the one declared, refused, --frame-rate's
    # own rule, told that it is refused without the option too, with the same rate and fixes.
    path = retagged(tmp_path / "retagged.mkv", declared, times, frames)
    caplog.set_level(logging.INFO, logger="seedvr2x")
    with pytest.raises(MediaError) as refused:
        examine(path)
    way_out = another_rate(found, change)
    assert str(refused.value) == (
        f"{path}: {leaves(strays, first_off, f'{declared}/1')}, and {way_out}"
    )
    assert f"; or give --frame-rate {rate.fraction(found)}, which takes" in way_out
    source = examine(path, frame_rate=found)
    assert (source.frames, source.frame_rate, source.stream.frame_rate) == (frames, found, declared)
    assert f"where it declares {declared}/1: taken at that rate" in caplog.text
    assert not strayed(caplog) and "too: taken" not in caplog.text
    first = first_off.split(",")[0]
    with pytest.raises(MediaError) as refused:
        examine(path, frame_rate=Fraction(declared))
    said = str(refused.value)
    assert said.startswith(f"{path}: --frame-rate {declared}/1: frame {first} lies")
    assert said.endswith(f": {REFUSED_WITHOUT} {way_out}")
    # Another rate that doesn't hold them either: refused without the option too, said whole.
    with pytest.raises(MediaError) as refused:
        examine(path, frame_rate=Fraction(declared + 7))
    assert str(refused.value).endswith(
        ": taken at that rate, frames would be dropped or doubled; without --frame-rate, the"
        f" file is refused too: {leaves(strays, first_off, f'{declared}/1')}, and {way_out}"
    )


@needs_ffmpeg
def test_a_short_file_on_its_declared_timeline_is_not_warned_of(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # 100 frames at 24000/1001 declared 24/1: within 4.4 ms of 24's timeline, every frame at its
    # place there. Read, nothing said: the output's 0.1% is under half a frame. Given a rate
    # that doesn't hold them, told that it is read without the option, and the rate its
    # timestamps are exactly, which the option takes.
    path = retagged(tmp_path / "short.mkv", 24, "round(N*1001/24)", 100)
    source = examine(path)
    assert (source.frames, source.frame_rate) == (100, 24)
    index = source.index
    assert index is not None and rate.timeline(index, Fraction(24)).holds
    assert not strayed(caplog)
    with pytest.raises(MediaError) as refused:
        examine(path, frame_rate=Fraction(25))
    assert str(refused.value).endswith(
        f": {read_without('24/1')}; its timestamps follow 24000/1001 fps (23.976) exactly, to the"
        " rounding of its time base, 1/1000 s: --frame-rate 24000/1001 takes its frames at that"
        " rate"
    )


@needs_ffmpeg
def test_a_first_pass_record_is_not_judged_again(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # The timing is the first pass's to judge (source.timing_refusal), which refuses 600 frames
    # at 24000/1001 declared 24/1. A resume that trusts the pass's record builds its source from
    # it (source.counted), judging nothing again, as before: the same frames from a record are
    # read at the rate recorded, and nothing is said of their timeline, the warning gone.
    path = retagged(tmp_path / "retagged.mkv", 24, "round(N*1001/24)", 600)
    declared = declare(path)
    scanned = scan(path)
    assert timing_refusal(declared, scanned).startswith("100 of its 600 frames lie half a frame")
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source = counted(declared, FirstPass(scanned.frames, index=scanned.index))
    assert (source.frames, source.frame_rate) == (600, 24)
    assert "timeline" not in caplog.text and "600 frames, 64x48" in caplog.text


def declaring(frame_rate: Fraction, override: Fraction | None = None) -> Declared:
    """A source declaring `frame_rate`, as the timing rule takes one; `override` --frame-rate."""
    stream = replace(UNTAGGED_HD, frame_rate=frame_rate, avg_frame_rate=frame_rate)
    return Declared(Path("made.mkv"), stream, conversion_for(stream), Fraction(1), override)


def scanned_at(pts: npt.NDArray[np.int64], time_base: Fraction) -> Scan:
    """A first pass's findings for frames at these pts, as scan times them: its index, and the
    shortest and longest of their durations, in microseconds."""
    steps = np.diff(pts)
    microseconds = time_base * 1_000_000
    index = FrameIndex(
        time_base, 0, pts, np.zeros(len(pts), np.uint32), np.zeros(len(pts), np.bool_), pts[:1]
    )
    return Scan(
        len(pts),
        len(steps),
        int(steps.min()) * microseconds,
        int(steps.max()) * microseconds,
        index,
    )


def rounded(frames: int, frame_rate: Fraction, time_base: Fraction, down: bool = False) -> Scan:
    """`frames` frames at `frame_rate`, their times rounded to `time_base`'s ticks as ffmpeg's
    muxers round them, to the nearest, a half up, or `down`: exact integers."""
    numerators = np.arange(frames, dtype=np.int64) * (
        frame_rate.denominator * time_base.denominator
    )
    divisor = frame_rate.numerator * time_base.numerator
    pts = numerators // divisor if down else (2 * numerators + divisor) // (2 * divisor)
    return scanned_at(pts, time_base)


@pytest.mark.parametrize(
    ("true", "declared", "time_base"),
    [
        # Matroska's milliseconds.
        (NTSC_FILM, NTSC_FILM, Fraction(1, 1000)),
        (Fraction(25), Fraction(25), Fraction(1, 1000)),
        (Fraction(30000, 1001), Fraction(30000, 1001), Fraction(1, 1000)),
        # 60000/1001 in Matroska, which declares it 19001/317, its frame duration in whole
        # nanoseconds read back by ffprobe: 0.9 ns a frame off, 0.18 ms over these frames.
        (Fraction(60000, 1001), Fraction(19001, 317), Fraction(1, 1000)),
        # MP4's time bases, which hold the rate, each frame on its tick.
        (NTSC_FILM, NTSC_FILM, Fraction(1, 24000)),
        (Fraction(25), Fraction(25), Fraction(1, 12800)),
        (Fraction(60000, 1001), Fraction(60000, 1001), Fraction(1, 60000)),
        # MPEG-TS's 90 kHz: 3,753.75 ticks a frame at 24000/1001, whole ones at the others.
        (NTSC_FILM, NTSC_FILM, Fraction(1, 90000)),
        (Fraction(25), Fraction(25), Fraction(1, 90000)),
        (Fraction(30000, 1001), Fraction(30000, 1001), Fraction(1, 90000)),
    ],
)
@pytest.mark.parametrize("down", [False, True], ids=["nearest", "down"])
def test_a_constant_rate_to_its_time_bases_rounding_read_at_any_length(
    true: Fraction, declared: Fraction, time_base: Fraction, down: bool
) -> None:
    # What sptenc and seedvr2x build (DESIGN.md, Input: "a file either of them builds is never
    # refused"): a constant rate's times, each rounded once to the time base. 200,000 frames, 2
    # h 19 min at 24000/1001: every frame within a tick of its place on the declared rate's
    # timeline, however far in, where a drift would add up; read, with or without --frame-rate.
    scanned = rounded(200_000, true, time_base, down)
    assert timing_refusal(declaring(declared), scanned) == ""
    assert timing_refusal(declaring(declared, declared), scanned) == ""
    index = scanned.index
    assert index is not None
    on = rate.timeline(index, true)
    assert on.holds and on.largest <= time_base


def test_half_a_frame_off_the_declared_timeline_refused_a_tick_less_read() -> None:
    # The bound itself, on a file sptenc's rule passes, 25 fps declared, its frames 40 or 41 ms
    # apart: 20 steps of 41 ms put frame 20 and those after 20 ms after their place, half a
    # frame exactly, refused; 19 of them, 19 ms, a tick less, read. Early alike, 39 ms steps.
    # In 90 kHz ticks: 1,800 of them off, refused, 1,799, read.
    for step, tick, half in ((41, 1000, 20), (39, 1000, 20), (3690, 90000, 1800)):
        on_time = 40 * tick // 1000
        for off, refused in ((half, True), (half - 1, False)):
            steps = [step] * 20 + [on_time] * 40
            steps[19] -= (half - off) * (1 if step > on_time else -1)
            pts = np.cumsum([0, *steps], dtype=np.int64)
            scanned = scanned_at(pts, Fraction(1, tick))
            index = scanned.index
            assert index is not None
            on = rate.timeline(index, Fraction(25))
            assert on.largest == Fraction(off, tick) and on.holds is not refused
            said = timing_refusal(declaring(Fraction(25)), scanned)
            if not refused:
                assert said == ""
                continue
            assert said.startswith(
                "41 of its 61 frames lie half a frame or more from their place on the timeline"
                " of the rate it declares, 25/1, from frame 20, up to 20 ms (0.50 of a frame),"
                " and its timestamps follow no one rate exactly: "
            )


MATROSKA = "matroska,webm"  # ffprobe's name for the container (media/probe.py, CONTAINER)
MP4 = "mov,mp4,m4a,3gp,3g2,mj2"


def probed(said: str, container: str | None = MATROSKA) -> VideoStream:
    """A stream as probe reads it from a file whose stream ffprobe gives the rate `said`, in the
    container it names so, or naming none (media/probe.py, parse): no frame read."""
    stream = {"codec_name": "ffv1", "width": 64, "height": 48, "pix_fmt": "gbrp16le"}
    rates = {"r_frame_rate": said, "avg_frame_rate": said}
    named = {} if container is None else {"format": {"format_name": container}}
    return parse(
        Path("made.mkv"), json.dumps(named | {"streams": [stream | rates]}), first_frame=False
    )


def test_matroskas_readings_of_the_ntsc_rates() -> None:
    # What ffprobe reads back from a Matroska file at N * 1000/1001 fps, as measured with ffmpeg
    # n9.0.2 and mkvmerge v82.0 (rate.matroska_reading): the frame's duration in whole
    # nanoseconds, truncated, read as the nearest fraction of terms up to 30000; each read as
    # its NTSC rate, the rounded duration's reading too, and a stream copy's, which ffmpeg
    # writes from the rate it read (240000/1001: 29011/121, then 23017/96).
    readings = {
        48: ("7001/146", "26997/563"),
        50: ("29021/581",),
        60: ("19001/317",),
        100: ("999/10",),
        120: ("29011/242",),
        240: ("29011/121", "23017/96"),
    }
    for n, read in readings.items():
        for said in read:
            assert rate.matroska_reading(Fraction(said)) == Fraction(n * 1000, 1001), said
    # The durations themselves, ffmpeg's av_reduce line for line: 60000/1001's 16,683,333 ns,
    # 48000/1001's truncated and rounded, 240000/1001's after a stream copy; a fraction whose
    # terms fit is itself.
    for nanoseconds, read in (
        (16_683_333, "19001/317"),
        (20_854_166, "7001/146"),
        (20_854_167, "26997/563"),
        (4_170_831, "23017/96"),
        (41_708_333, "24000/1001"),
        (33_366_666, "30000/1001"),
        (40_000_000, "25"),
    ):
        assert rate._av_reduce(1_000_000_000, nanoseconds, 30000) == Fraction(read)
    # A rate read back as itself, or no NTSC rate's reading, is left: the NTSC rates up to
    # 30000/1001 and those whose fraction reduces, whole rates, a neighbour of a reading, the
    # decimal of 23.976 and of 47.952.
    for said in ("24000/1001", "30000/1001", "3000/91", "25", "50", "60", "120", "1000/33"):
        assert rate.matroska_reading(Fraction(said)) is None, said
    for said in ("19001/318", "19002/317", "2997/125", "5994/125", "7001/147"):
        assert rate.matroska_reading(Fraction(said)) is None, said
    # A reading that is itself a rate files declare is left as declared: 167000/1001's, 1001/6
    # fps, a multiple of 1/12, the one such reading of the 240. And N beyond 240, the highest
    # measured: 241000/1001's reading, 13001/54.
    assert rate._av_reduce(1_000_000_000, 1_001_000_000 // 167, 30000) == Fraction(1001, 6)
    assert rate.matroska_reading(Fraction(1001, 6)) is None
    beyond = rate._av_reduce(1_000_000_000, 1_001_000_000 // 241, 30000)
    assert beyond == Fraction(13001, 54) and rate.matroska_reading(beyond) is None
    # Every N of the range: its two durations' readings are read as it, or are itself, but
    # 167's; no reading is another rate's.
    seen: dict[Fraction, int] = {}
    for n in range(1, rate.MOST_NTSC + 1):
        true = Fraction(n * 1000, 1001)
        for nanoseconds in (1_001_000_000 // n, (2_002_000_000 + n) // (2 * n)):
            reading = rate._av_reduce(1_000_000_000, nanoseconds, 30000)
            found = rate.matroska_reading(reading)
            assert found == (None if reading == true or n == 167 else true), n
            assert seen.setdefault(reading, n) == n
    assert sum(1 for reading, n in seen.items() if reading != Fraction(n * 1000, 1001)) == 172


def test_a_matroska_streams_reading_is_probed_as_its_ntsc_rate() -> None:
    # Where the rate enters (media/probe.py, parse): a Matroska stream ffprobe gives 19001/317
    # for is 60000/1001's, what ffprobe gave kept beside it; the same rate from MP4, which holds
    # 60000/1001 as it is, or from a probe that names no container, is left as given; so is a
    # Matroska stream's own rate.
    stream = probed("19001/317")
    assert (stream.frame_rate, stream.ffprobe_rate) == (Fraction(60000, 1001), Fraction(19001, 317))
    assert stream.avg_frame_rate == Fraction(19001, 317)
    for container in (MP4, "mpegts", None):
        stream = probed("19001/317", container)
        assert (stream.frame_rate, stream.ffprobe_rate) == (Fraction(19001, 317), None)
    for said in ("24000/1001", "25/1", "60/1"):
        stream = probed(said)
        assert (stream.frame_rate, stream.ffprobe_rate) == (Fraction(said), None)


@pytest.mark.parametrize(
    ("true", "said", "left_at"),
    [
        ("120000/1001", "29011/242", 1_161_780),
        ("60000/1001", "19001/317", 8_939_970),
        ("48000/1001", "7001/146", 3_335_976),
        ("48000/1001", "26997/563", 4_297_039),
        ("240000/1001", "23017/96", 515_400),
    ],
)
def test_matroskas_reading_of_an_ntsc_rate_is_read_as_that_rate(
    true: str, said: str, left_at: int
) -> None:
    # A Matroska file declares 120000/1001 as 29011/242, ffprobe's reading of its frame duration
    # in whole nanoseconds, 3.2 ns a frame off: timestamps at 120000/1001, in milliseconds, hold
    # that timeline for 1,161,780 frames, 2 h 41 min, and were refused a frame further, naming
    # the rate they are exactly. Design's decision (2026-10-12): the stream is read as the NTSC
    # rate, which its frames hold however far in, DESIGN.md's "never refused" of a file sptenc
    # or seedvr2x builds; that rate is the source's, the output's. The same file as ffprobe
    # gives it from another container is refused from that frame, as before; MP4 holds the rate
    # itself.
    rate_read, rate_said = Fraction(true), Fraction(said)
    scanned = rounded(left_at + 1, rate_read, MS)
    stream = probed(said)
    assert (stream.frame_rate, stream.ffprobe_rate) == (rate_read, rate_said)
    in_matroska = declaring(stream.frame_rate)
    assert timing_refusal(in_matroska, scanned) == ""
    if left_at < 2_000_000:
        source = counted(in_matroska, FirstPass(scanned.frames, index=scanned.index))
        assert (source.frame_rate, source.stream.frame_rate) == (rate_read, rate_read)
    as_given = declaring(probed(said, MP4).frame_rate)
    refusal = timing_refusal(as_given, scanned)
    assert refusal.startswith(
        f"1 of its {left_at + 1} frames lies half a frame or more from its place on the timeline"
        f" of the rate it declares, {said}, from frame {left_at}, up to "
    )
    if said == "29011/242":
        # Whole: the rate its timestamps are exactly, and what the declared one costs, said to
        # its first digits.
        assert refusal.endswith(
            f" up to 4.2 ms (0.50 of a frame), and {another_rate(rate_read, '0.000038% shorter')}"
        )
    index = scanned.index
    assert index is not None
    assert timing_refusal(as_given, scanned_at(index.pts[:-1], MS)) == ""


def test_a_file_truly_at_a_readings_rate() -> None:
    # A Matroska file whose frames truly run at 29011/242 fps, their timestamps that rate's to
    # the millisecond: read as 120000/1001, it is read while every frame holds that timeline,
    # 1,163,181 frames, 2 h 41 min, the two rates 3.8e-7 apart; a frame more, refused by the
    # declared timeline's rule, as no rate's exactly: 29011/242 is no rate files declare, and
    # their mean rate, rounded, holds them to no tick's spread (rate.exact). Its ways out the
    # remake at 120000/1001, which numbers its frames as they come, every one kept; and
    # --frame-rate 29011/242, which holds them, for who knows the file.
    truly = Fraction(29011, 242)
    stream = probed("29011/242")
    assert stream.frame_rate == Fraction(120000, 1001)
    declared = declaring(stream.frame_rate)
    scanned = rounded(1_163_182, truly, MS)
    index = scanned.index
    assert index is not None
    assert timing_refusal(declared, scanned_at(index.pts[:-1], MS)) == ""
    assert timing_refusal(declared, scanned) == (
        "1 of its 1163182 frames lies half a frame or more from its place on the timeline of the"
        " rate it declares, 120000/1001, from frame 1163181, up to 4.2 ms (0.50 of a frame), and"
        f" {no_rate(Fraction(120000, 1001))}"
    )
    assert timing_refusal(replace(declared, rate_override=truly), scanned) == ""


@needs_ffmpeg
@pytest.mark.parametrize(("fps", "said"), [("60000/1001", "19001/317"), ("48000/1001", "7001/146")])
def test_seedvr2xs_own_matroska_files_read_back_at_their_rate(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, fps: str, said: str
) -> None:
    # seedvr2x's FFV1 master and its finished file are Matroska files: written at 60000/1001,
    # they declare 19001/317 to ffprobe, and were read back at it (DESIGN.md, Open questions).
    # Read back through seedvr2x itself, each is at the rate it was written at, said in an info
    # line with what ffprobe gave: two segments by the master's writer (media/writer.py,
    # FFV1Writer), then their assembly with a source's other streams (media/assembly.py,
    # assemble), a stream copy.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    written = Fraction(fps)
    frames = np.full((30, 48, 64, 3), 0.5, np.float32)
    segments = [(tmp_path / "seg_000000.mkv", 12), (tmp_path / "seg_000001.mkv", 18)]
    first = 0
    for path, count in segments:
        with FFV1Writer(path, "gbrp16le", 64, 48, written, Tags()) as writer:
            writer.write(frames[first : first + count])
        first += count
        declared = declare(path)
        assert (declared.stream.frame_rate, declared.stream.ffprobe_rate) == (
            written,
            Fraction(said),
        )
        assert (
            f"{path}: declares {said} fps, Matroska's reading of {fps} (its frame duration in"
            f" whole nanoseconds): read as {fps}"
        ) in caplog.text
    source_path = tmp_path / "source.mkv"
    source_frames = ("-f", "lavfi", "-i", f"testsrc2=s=64x48:r={fps}", "-frames:v", "30")
    run(*source_frames, "-c:v", "ffv1", str(source_path))
    source = examine(source_path)
    assert source.index is not None and (source.frames, source.frame_rate) == (30, written)
    work = tmp_path / "final.mkv.work"
    work.mkdir()
    final = tmp_path / "final.mkv"
    assemble(final, segments, source.frame_rate, source_path, source.index, work)
    caplog.clear()
    back = examine(final)
    assert streams(final)[0]["r_frame_rate"] == said
    assert (back.frames, back.frame_rate, back.stream.ffprobe_rate) == (30, written, Fraction(said))
    assert (
        f"{final}: declares {said} fps, Matroska's reading of {fps} (its frame duration in whole"
        f" nanoseconds): read as {fps}"
    ) in caplog.text
    # --frame-rate still overrides, the rate declared said as it is read: the whole rate
    # beside it, which holds 30 frames within half a frame.
    whole = Fraction(round(written))
    forced = examine(final, frame_rate=whole)
    assert (forced.frame_rate, forced.stream.frame_rate) == (whole, written)
    assert f"at {whole} fps (--frame-rate; it declares {fps})" in caplog.text


@needs_ffmpeg
@pytest.mark.parametrize(
    ("fps", "written", "copied"),
    [("120000/1001", "29011/242", "29011/242"), ("240000/1001", "29011/121", "23017/96")],
)
def test_a_matroska_file_and_its_stream_copy_read_at_their_ntsc_rate(
    tmp_path: Path, fps: str, written: str, copied: str
) -> None:
    # A Matroska file at 120000/1001 or 240000/1001, as ffmpeg writes it from frames that hold
    # the rate, and a stream copy of it, whose frame duration ffmpeg writes again from the rate
    # it read: 240000/1001 declares 29011/121, its copy 23017/96 and every copy after; each
    # read at the NTSC rate. Not media/writer.py's own master at these two rates, which isn't
    # at them: ffmpeg guesses its rawvideo input's rate among its standard rates, 120/1 for
    # 120000/1001, and the master declares 120/1, a place of that grid skipped every 1,000
    # frames (reported to design, 2026-10-12).
    first = tmp_path / "written.mkv"
    made = ("-f", "lavfi", "-i", f"testsrc2=s=64x48:r={fps}", "-frames:v", "30")
    run(*made, "-c:v", "ffv1", str(first))
    files = [first]
    for name in ("copied.mkv", "copied_again.mkv"):
        run("-i", str(files[-1]), "-c", "copy", str(tmp_path / name))
        files.append(tmp_path / name)
    for path, said in zip(files, (written, copied, copied), strict=True):
        assert streams(path)[0]["r_frame_rate"] == said
        source = examine(path)
        assert (source.frames, source.frame_rate) == (30, Fraction(fps))
        assert source.stream.ffprobe_rate == Fraction(said)


MS = Fraction(1, 1000)


def missing(
    frames: int, frame_rate: Fraction, absent: Sequence[int], time_base: Fraction = MS
) -> Scan:
    """A first pass's findings for `frames` places at `frame_rate`, their times rounded to
    `time_base` (rounded), without the frames `absent`, by their number: the others keep their
    times, as the frames of a file with holes do."""
    index = rounded(frames, frame_rate, time_base).index
    assert index is not None
    return scanned_at(np.delete(index.pts, list(absent)), time_base)


def stepping(steps: Sequence[int], time_base: Fraction = MS) -> Scan:
    """A first pass's findings for frames these steps apart, in ticks of `time_base`."""
    return scanned_at(np.cumsum([0, *steps], dtype=np.int64), time_base)


def filled_in(declared: Declared, scanned: Scan) -> FrameIndex:
    """The index in places of a source the timing rule reads (source.in_places)."""
    assert timing_refusal(declared, scanned) == ""
    index = in_places(declared, scanned)
    assert index is not None
    return index


def held_rows(index: FrameIndex) -> list[int]:
    """The places of an index that are held rows, missing frames' filled."""
    assert index.held is not None
    return np.flatnonzero(index.held).tolist()


def unfilled(declared: Declared, scanned: Scan) -> str:
    """Why the timing rule refuses a source, which no reading in places saves: the index a
    first pass would get of it is the scan's own."""
    said = timing_refusal(declared, scanned)
    assert said and in_places(declared, scanned) is scanned.index
    return said


def variable(shortest: int, longest: int) -> str:
    """sptenc's refusal of frames lasting from `shortest` to `longest` ms (scan.timing_error)."""
    return (
        f"variable frame rate: frames last from {shortest} to {longest} ms, and only a constant"
        " frame rate is supported"
    )


def test_the_frames_missing_after_the_first_are_filled(caplog: pytest.LogCaptureFixture) -> None:
    # The source this was built for (label anime-dark; research/docs/seeking.md, mechanism 1):
    # 99,583 frames decoded at 24000/1001 in Matroska's milliseconds, their steps 41 and 42 ms
    # and one of 125, after the first frame: places 1 and 2 of 99,585 are empty. Refused until
    # now as a variable frame rate; read, its index in places, the first frame held in both, and
    # said so by the source's warning, with the figures, from a first pass and from the record
    # of one alike (source.counted).
    declared = declaring(NTSC_FILM)
    scanned = missing(99_585, NTSC_FILM, [1, 2])
    first_pass = scanned.index
    assert first_pass is not None and scanned.frames == 99_583
    steps = np.diff(first_pass.pts)
    assert {step: int((steps == step).sum()) for step in (41, 42, 125)} == {
        41: 29_044,
        42: 70_537,
        125: 1,
    }
    assert timing_error(scanned, NTSC_FILM, NTSC_FILM) == variable(41, 125)
    assert places(first_pass, NTSC_FILM).largest < MS
    index = filled_in(declared, scanned)
    assert index.frames == 99_585 and held_rows(index) == [1, 2]
    assert index.shown[:4].tolist() == [0, 0, 0, 1]
    assert filling(index) == Filling(99_585, 2, 1, Hole(1, 2, 0), 2)
    assert index.decoded().to_bytes() == first_pass.to_bytes()
    warning = (
        "made.mkv: 2 frames missing after frame 0: its timestamps step from 0.000 to 0.125 s"
        " there, 3 frames of the rate it declares, 24000/1001, leaving places 1 and 2 empty on"
        " that rate's grid; read as a player shows it, frame 0 held in their place: 99585 frames,"
        " 2 of them filled (0.0020%), 99583 decoded"
    )
    recorded = FrameIndex.from_bytes(index.to_bytes())
    for found in (index, recorded):
        caplog.clear()
        caplog.set_level(logging.INFO, logger="seedvr2x")
        source = counted(declared, FirstPass(found.frames, index=found))
        assert source.frames == 99_585 and source.filled == filling(index)
        assert [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING] == [warning]
        assert "made.mkv: 99585 frames (99583 decoded, 2 filled), 1920x1080" in caplog.text
    # With --frame-rate at the rate it declares: the same places, the grid --frame-rate's.
    caplog.clear()
    forced = declaring(NTSC_FILM, NTSC_FILM)
    assert filled_in(forced, scanned).to_bytes() == index.to_bytes()
    assert (
        "made.mkv: --frame-rate 24000/1001: every frame within 0.5 ms of its place on that"
        " rate's timeline (0.01 of a frame), places left empty between them, the rate it"
        " declares: taken at that rate"
    ) in caplog.text
    counted(forced, FirstPass(index.frames, index=index))
    assert (
        "3 frames of the rate it is taken at, 24000/1001 (--frame-rate), leaving places 1 and 2"
        " empty on that rate's grid; read as a player shows it, frame 0 held in their place"
    ) in caplog.text


THIRD = (1, 2, 6, 10)  # of every 12 frames, the missing ones: holes of 2, 1 and 1


@pytest.mark.parametrize(
    ("frames", "frame_rate", "time_base", "absent", "said"),
    [
        # A hole later, of one frame.
        (
            2000,
            NTSC_FILM,
            MS,
            [1000],
            "1 frame missing after frame 999: its timestamps step from 41.667 to 41.750 s there,"
            " 2 frames of the rate it declares, 24000/1001, leaving place 1000 empty on that"
            " rate's grid; read as a player shows it, frame 999 held in its place: 2000 frames, 1"
            " of them filled (0.050%), 1999 decoded",
        ),
        # One of five frames, in MP4's ticks.
        (
            2000,
            NTSC_FILM,
            Fraction(1, 24000),
            [501, 502, 503, 504, 505],
            "5 frames missing after frame 500: its timestamps step from 20.854 to 21.104 s there,"
            " 6 frames of the rate it declares, 24000/1001, leaving places 501 to 505 empty on"
            " that rate's grid; read as a player shows it, frame 500 held in their place: 2000"
            " frames, 5 of them filled (0.25%), 1995 decoded",
        ),
        # Scattered holes of 1 to 3 frames.
        (
            2000,
            NTSC_FILM,
            MS,
            [49, 50, 93, 94, 95, 97, 1500, 1998],
            "8 frames missing in 5 holes, the first after frame 48, 2 frames long (its timestamps"
            " step from 2.002 to 2.127 s there, 3 frames of the rate it declares, 24000/1001),"
            " the longest 3 frames: its timestamps leave those places empty on the grid of the"
            " rate it declares, 24000/1001; read as a player shows it, the frame before each hole"
            " held through it: 2000 frames, 8 of them filled (0.40%), 1992 decoded",
        ),
        # A third of the places empty, as an encoder that drops repeated frames leaves them
        # (DESIGN.md, Input): read the same way, the share said, and what such a file is.
        (
            3000,
            NTSC_FILM,
            MS,
            [frame for frame in range(3000) if frame % 12 in THIRD],
            "1000 frames missing in 750 holes, the first after frame 0, 2 frames long (its"
            " timestamps step from 0.000 to 0.125 s there, 3 frames of the rate it declares,"
            " 24000/1001), the longest 2 frames: its timestamps leave those places empty on the"
            " grid of the rate it declares, 24000/1001; read as a player shows it, the frame"
            " before each hole held through it: 3000 frames, 1000 of them filled (33.3%), 2000"
            " decoded; so large a share is what an encoder leaves that drops repeated frames,"
            " each picture held as it was until the next",
        ),
        # Steps of 2, 1, 3, 2 and 1 frames, at 25 fps in 90 kHz ticks: 4 places in 9 empty.
        (
            2700,
            Fraction(25),
            Fraction(1, 90000),
            [frame for frame in range(2700) if frame % 9 in (1, 4, 5, 7)],
            "1200 frames missing in 900 holes, the first after frame 0, 1 frame long (its"
            " timestamps step from 0.000 to 0.080 s there, 2 frames of the rate it declares,"
            " 25/1), the longest 2 frames: its timestamps leave those places empty on the grid of"
            " the rate it declares, 25/1; read as a player shows it, the frame before each hole"
            " held through it: 2700 frames, 1200 of them filled (44.4%), 1500 decoded; so large a"
            " share is what an encoder leaves that drops repeated frames, each picture held as it"
            " was until the next",
        ),
    ],
    ids=["one later", "five, MP4", "scattered", "a third", "four in nine, 90 kHz"],
)
def test_frames_on_the_grid_with_places_left_empty_are_read_in_places(
    caplog: pytest.LogCaptureFixture,
    frames: int,
    frame_rate: Fraction,
    time_base: Fraction,
    absent: list[int],
    said: str,
) -> None:
    # Each frame decoded takes the place it had among all of them, the frames as many as the
    # places, the frame before a hole shown through it; the same with --frame-rate at that rate;
    # the warning gives the first hole, how many, the longest and the share.
    scanned = missing(frames, frame_rate, absent, time_base)
    first_pass = scanned.index
    assert first_pass is not None and scanned.frames == frames - len(absent)
    assert timing_error(scanned, frame_rate, frame_rate).startswith("variable frame rate: ")
    index = filled_in(declaring(frame_rate), scanned)
    kept = [frame for frame in range(frames) if frame not in set(absent)]
    assert index.frames == frames and index.placed.tolist() == kept
    assert held_rows(index) == absent
    assert index.decoded().to_bytes() == first_pass.to_bytes()
    assert filled_in(declaring(frame_rate, frame_rate), scanned).to_bytes() == index.to_bytes()
    caplog.set_level(logging.WARNING, logger="seedvr2x")
    caplog.clear()
    source = counted(declaring(frame_rate), FirstPass(index.frames, index=index))
    assert source.frames == frames
    assert [record.getMessage() for record in caplog.records] == [f"made.mkv: {said}"]


def test_what_stays_refused_with_places_left_empty() -> None:
    # "A contradicted rate has two readings, which is why it is refused; so is a frame off the
    # grid, a variable frame rate" (DESIGN.md, Input): each refused with the message it had
    # before the places, and no index in places made of it.
    film = declaring(Fraction(24))
    # 24000/1001 declared 24/1: no step is long, so no hole is looked for, and the strict rule
    # refuses it from the frame half a frame off, naming its rate.
    said = unfilled(film, missing(600, NTSC_FILM, []))
    assert said.startswith(
        "100 of its 600 frames lie half a frame or more from their place on the timeline of the"
        " rate it declares, 24/1, from frame 492, up to 25.3 ms (0.61 of a frame), and its"
        " output, written at that rate, would be 0.100% shorter than its timestamps' own time,"
        " off the source's other streams: its timestamps follow 24000/1001 fps (23.976) exactly"
    )
    # The same with two frames really missing: nearest places would fill the drift too, the
    # place 24/1's grid has more by frame 492, which a step of one frame spans: 41 ms for two
    # places, where the others last 41 to 42 a place. Refused, as a variable frame rate.
    assert unfilled(film, missing(600, NTSC_FILM, [200, 201])) == variable(41, 125)
    # And with 69 frames missing from the frame where the drift comes to half a frame, 492: the
    # frames after them are a place further on 24/1's grid, 71 places for 70 frames' time, 41.1
    # ms a place, within 1 ms of the others: refused all the same, a hole's step lasting the
    # frames it skips to within half a frame.
    assert unfilled(film, missing(700, NTSC_FILM, range(492, 561))) == variable(41, 2919)
    # A jump in the timestamps is refused first, as before, places left empty or not
    # (scan.Discontinuity): frames on the grid, two missing, in a scan that found one.
    jump = Discontinuity(1000, Fraction(41), Fraction(1041), False)
    said = unfilled(
        declaring(NTSC_FILM), replace(missing(2000, NTSC_FILM, [46, 47]), discontinuity=jump)
    )
    assert said.startswith("its timestamps jump at frame 1000, from 41.000 to 1041.000 s: a")
    # A short file of frames at 23 fps declared 24/1, two missing: each frame within half a
    # frame of a place of 24's grid, and its steps 43 to 44 ms a place, within 1 ms of each
    # other; but 24's period, 41.7 ms, isn't within them, sptenc's second test: refused.
    assert unfilled(film, missing(10, Fraction(23), [4, 5])) == variable(43, 131)
    # S9's timestamps (live-2; research/docs/seeking.md, mechanism 7): its steps of 62-63 ms
    # are 1.5 frames of the 24/1 it declares, no whole number of them: refused, and guided to
    # 24000/1001, as before.
    said = unfilled(film, scanned_at(np.array(live2(2000), dtype=np.int64), MS))
    assert said.startswith(
        "its timestamps run at 24000/1001 fps (23.976), within 10.9 ms of that rate's timeline"
        " (0.26 of a frame), where it declares 24/1, its frames lasting from 41 to 63 ms: a bad"
        " file, not read on a guess: remake it at that rate"
    )
    # A true variable rate, 24 then 30 fps: two frames in one place of 24's grid.
    assert unfilled(film, stepping([42, 41, 42] * 30 + [33, 34, 33] * 30)) == variable(33, 42)
    # Frames at a constant 12 fps declared 24/1: their steps within 1 ms, no hole is looked
    # for: a contradicted rate, named, with its two fixes; --frame-rate 12/1 reads them as they
    # are, 300 frames, no place filled.
    twelve = missing(300, Fraction(12), [])
    said = unfilled(film, twelve)
    assert said.startswith(
        "its timestamps run at 12/1 fps (12), within 0.3 ms of that rate's timeline (0.00 of a"
        " frame), where it declares 24/1, its frames lasting from 83 to 84 ms: a bad file, not"
        " read on a guess: remake it at that rate"
    )
    assert "; or give --frame-rate 12/1, which takes its frames at that rate" in said
    assert filled_in(declaring(Fraction(24), Fraction(12)), twelve) is twelve.index
    # The same with one frame missing: every step spans 2 places of 24's grid, or 4: the
    # frames sit on 12's grid, the contradicted rate still, never read with half its places
    # filled (media/source.py, _on_grid, PROVISIONAL).
    assert unfilled(film, missing(300, Fraction(12), [100])) == variable(83, 167)
    # Steps that are no whole number of frames, at 25 fps: 1.4 then 1.6 frames, which span 1
    # and 2 places, 56 and 32 ms a place, beside a real hole; and 1.5 frames, exactly between
    # two places. With --frame-rate 25 too: the step of 1.6 frames is within half a frame of
    # its 2 places, and was read as a missing frame while the option waived the time a place.
    pal, forced = declaring(Fraction(25)), declaring(Fraction(25), Fraction(25))
    off = stepping([40] * 10 + [56, 64] + [40] * 10 + [120] + [40] * 10)
    assert unfilled(pal, off) == variable(40, 120)
    assert unfilled(forced, off) == (
        "--frame-rate 25/1: frame 12 lies 40.0 ms after its place on that rate's timeline (1.000"
        " of a frame, half a frame being 20.0 ms), and 21 more frames, up to 120 ms: taken at"
        " that rate, frames would be dropped or doubled"
    )
    assert unfilled(pal, stepping([40] * 10 + [60] + [40] * 10)) == variable(40, 60)
    # Two frames in one place, 8 ms either side of it, beside a real hole: refused, and with
    # --frame-rate too.
    doubled = stepping([40] * 5 + [32, 16, 32] + [40] * 5 + [120] + [40] * 5)
    assert unfilled(pal, doubled) == variable(16, 120)
    assert unfilled(forced, doubled).startswith(
        "--frame-rate 25/1: frame 7 lies 32.0 ms before its place on that rate's timeline (0.800"
        " of a frame, half a frame being 20.0 ms), and "
    )
    # A frame late beside a real hole, by half a frame exactly, 20 ms, or a tick less: refused
    # either way, sptenc's 1 ms holding the time a place, and with --frame-rate 25 as without:
    # its waiver of that bound stops where places are skipped (media/source.py, _on_grid,
    # PROVISIONAL).
    for late, shortest in ((20, 20), (19, 21)):
        scanned = stepping([40] * 10 + [120] + [40] * 4 + [40 + late, 40 - late] + [40] * 10)
        assert unfilled(pal, scanned) == variable(shortest, 120)
        assert unfilled(forced, scanned) == (
            "--frame-rate 25/1: frame 11 lies 80.0 ms after its place on that rate's timeline"
            " (2.000 of a frame, half a frame being 20.0 ms), and 16 more frames, up to"
            f" {80 + late} ms: taken at that rate, frames would be dropped or doubled"
        )
    # A hole's step lasts the frames it skips to within half a frame, strictly, with --frame-rate
    # or without. At 25 fps in ticks of 0.1 ms: 20 steps of 39.5 ms leave a frame 10 ms before
    # its place, and a step of 40.5 frames from it the next one 10 ms after its own, 40 places
    # on, each within half a frame of its place and the step 40.5 ms a place, within 1 ms of the
    # others: refused, by the half frame alone; a tick less, read, its 39 places filled.
    tenth = Fraction(1, 10_000)
    for declared in (pal, forced):
        assert unfilled(declared, stepping([395] * 20 + [16_200] + [400] * 10, tenth))
        index = filled_in(declared, stepping([395] * 20 + [16_199] + [400] * 10, tenth))
        assert index.frames == 71 and held_rows(index) == list(range(21, 60))
    # A frame exactly between two places, half a frame after one, where every step is within
    # 1 ms a place of the others (20 steps of 40.5 ms, then one of 410 ms for 10 places):
    # refused either way, no place being the nearest; a tick less, read, its 9 places filled.
    for declared in (pal, forced):
        assert unfilled(declared, stepping([405] * 20 + [4100] + [400] * 10, tenth))
        index = filled_in(declared, stepping([405] * 20 + [4099] + [400] * 10, tenth))
        assert index.frames == 41 and held_rows(index) == list(range(21, 30))


def test_frame_rate_with_places_left_empty(caplog: pytest.LogCaptureFixture) -> None:
    # --frame-rate R: frames within half a frame of R's timeline counted frame after frame are
    # read as before, no place filled, whatever their steps: S9's, 41 to 63 ms, at 24000/1001.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    ntsc = declaring(Fraction(24), NTSC_FILM)
    s9 = np.array(live2(2000), dtype=np.int64)
    scanned = scanned_at(s9, MS)
    assert filled_in(ntsc, scanned) is scanned.index
    # A file with two frames missing that declares another rate than its timestamps', 24/1 for
    # 24000/1001's to the millisecond: refused without the option, and read with it, on R's
    # grid with its two places filled: the case --frame-rate reads places for.
    holes = missing(600, NTSC_FILM, [200, 201])
    assert unfilled(declaring(Fraction(24)), holes) == variable(41, 125)
    index = filled_in(ntsc, holes)
    assert index.frames == 600 and held_rows(index) == [200, 201]
    assert (
        "made.mkv: --frame-rate 24000/1001: every frame within 0.5 ms of its place on that"
        " rate's timeline (0.01 of a frame), places left empty between them, where it declares"
        " 24/1: taken at that rate"
    ) in caplog.text
    # S9's with two frames missing: refused. The option's waiver of the 1 ms bound stops where
    # places are skipped, and S9's steps are 41 to 63 ms a place (media/source.py, _on_grid,
    # PROVISIONAL): refused as before the places, the first frame off named.
    assert unfilled(ntsc, scanned_at(np.delete(s9, [300, 301]), MS)) == (
        "--frame-rate 24000/1001: frame 300 lies 91.5 ms after its place on that rate's timeline"
        " (2.194 of a frame, half a frame being 20.9 ms), and 1697 more frames, up to 94.3 ms:"
        " taken at that rate, frames would be dropped or doubled"
    )
    # 24000/1001 material given 24: refused as before, from the frame half a frame off; and
    # with two frames really missing, refused too: the place 24's grid has more is spanned by a
    # step of one frame, where a hole's step lasts the frames it skips; told that without the
    # option it is read, its missing frames filled.
    twenty_four = declaring(NTSC_FILM, Fraction(24))
    assert unfilled(twenty_four, missing(600, NTSC_FILM, [])).startswith(
        "--frame-rate 24/1: frame 492 lies 21.0 ms after its place on that rate's timeline"
        " (0.504 of a frame, half a frame being 20.8 ms), and 99 more frames, up to 25.3 ms:"
        " taken at that rate, frames would be dropped or doubled; without --frame-rate, the"
        " file is read at the rate it declares, 24000/1001"
    )
    said = unfilled(twenty_four, holes)
    assert said.startswith("--frame-rate 24/1: frame 200 lies 91.7 ms after its place")
    assert said.endswith(
        ": taken at that rate, frames would be dropped or doubled; without --frame-rate, the file"
        " is read at the rate it declares, 24000/1001, on whose grid its frames sit, the 2 frames"
        " missing there filled"
    )
    # A wrong rate on a file with one frame missing, 25 fps given 30: the same words, where the
    # refusal ended at "dropped or doubled".
    assert unfilled(declaring(Fraction(25), Fraction(30)), missing(250, Fraction(25), [125])) == (
        "--frame-rate 30/1: frame 3 lies 20.0 ms after its place on that rate's timeline (0.600"
        " of a frame, half a frame being 16.7 ms), and 245 more frames, up to 1693.3 ms: taken"
        " at that rate, frames would be dropped or doubled; without --frame-rate, the file is"
        " read at the rate it declares, 25/1, on whose grid its frames sit, the 1 frame missing"
        " there filled"
    )
    # And with 69 frames missing from frame 492, where the drift on 24's grid comes to half a
    # frame: 71 places for 70 frames' time, 41.1 ms a place, within 1 ms of the others: refused
    # with the option as without, by the half frame alone.
    said = unfilled(twenty_four, missing(700, NTSC_FILM, range(492, 561)))
    assert said.startswith("--frame-rate 24/1: frame 492 lies 2898.0 ms after its place on that")
    # A 24 fps file given 48: every other place of 48's grid would be empty, every frame
    # doubled; its frames sit on 24's grid, the rate it declares: refused as before, and told
    # so (media/source.py, _on_grid, PROVISIONAL).
    said = unfilled(declaring(Fraction(24), Fraction(48)), missing(300, Fraction(24), []))
    assert said.startswith("--frame-rate 48/1: frame 1 lies 21.2 ms after its place on that")
    assert said.endswith(
        "taken at that rate, frames would be dropped or doubled; without --frame-rate, the file"
        " is read at the rate it declares, 24/1, whose timeline holds every frame within half a"
        " frame"
    )
    # A short file whose frames, two missing, sit on both grids, the declared rate's and 24's:
    # taken at 24 as asked, and warned of as such a file is without a hole.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    index = filled_in(twenty_four, missing(300, NTSC_FILM, [200, 201]))
    assert index.frames == 300
    assert (
        "made.mkv: --frame-rate 24/1: every frame within 12.7 ms of its place on that rate's"
        " timeline (0.30 of a frame), places left empty between them, where it declares"
        " 24000/1001: taken at that rate"
    ) in caplog.text
    assert (
        "made.mkv: its frames are at the rate it declares, 24000/1001, too: taken at 24/1 as"
        " asked, the output lasting 0.100% shorter than the source, off its other streams"
    ) in caplog.text


DROPPED_OR_DOUBLED = "taken at that rate, frames would be dropped or doubled"


def test_a_joins_step_is_no_missing_frame_with_frame_rate() -> None:
    # Segments joined by ffmpeg's concat demuxer alone, each starting where the one before ends
    # by its declared duration, its audio's when that outlasts the video: 5 segments of 50
    # frames at 25 fps, in milliseconds, each 64 ms after the last frame of the one before, 1.6
    # frames, within half a frame of 2 places. No frame is missing there: refused without
    # --frame-rate, and with --frame-rate 25, which read it with a frame doubled at two joins
    # of four and none at the others (252 places, 50 and 151 filled) while the option waived
    # the time a place where places are skipped (a review's finding, 2026-10-12): 32 ms a
    # place at a join, 40 elsewhere.
    join = stepping(([40] * 49 + [64]) * 4 + [40] * 49)
    assert join.frames == 250
    assert unfilled(declaring(Fraction(25)), join) == variable(40, 64)
    assert unfilled(declaring(Fraction(25), Fraction(25)), join).startswith(
        "--frame-rate 25/1: frame 50 lies 24.0 ms after its place on that rate's timeline (0.600"
        " of a frame, half a frame being 20.0 ms), and 199 more frames, up to 96 ms:"
        f" {DROPPED_OR_DOUBLED}"
    )
    # One long step among steps of 24 fps in milliseconds, S9's catch-up step, with --frame-rate
    # 24: of 63 ms, 1.512 frames, it was read as a frame missing; refused. Of 62 ms, the frames
    # hold 24's timeline counted frame after frame, within half a frame: read as before the
    # places, none filled (DESIGN.md's waiver, S9's).
    film = declaring(Fraction(24), Fraction(24))
    assert unfilled(film, stepping([42, 41, 42] * 20 + [63] + [42, 41, 42] * 20)).startswith(
        "--frame-rate 24/1: frame 61 lies 21.3 ms after its place on that rate's timeline (0.512"
        " of a frame, half a frame being 20.8 ms), and 60 more frames, up to 21.7 ms:"
        f" {DROPPED_OR_DOUBLED}"
    )
    caught_up = stepping([42, 41, 42] * 20 + [62] + [42, 41, 42] * 20)
    assert filled_in(film, caught_up) is caught_up.index


@needs_ffmpeg
@needs_x264
def test_a_join_of_segments_whose_audio_outlasts_them_refused_with_frame_rate(
    tmp_path: Path,
) -> None:
    # The real file of that finding, made here: 8 segments of 24 frames at 24000/1001, each
    # encoded on its own with audio that lasts 1.033 s where its video lasts 1.001 (the review's
    # AAC of 1.012 s lasts so in Matroska; FLAC here, whose length is the samples'), joined by
    # ffmpeg's concat demuxer alone: the video steps 74 ms at each of the 7 joins, 1.77 frames,
    # and no frame is missing. Refused as a variable frame rate without --frame-rate, and with
    # the rate it declares, which read it as 197 frames for 192, a frame doubled at 5 joins of
    # 7: refused, the first frame off named.
    listed = tmp_path / "plain.txt"
    for k in range(8):
        run(
            *("-f", "lavfi", "-i", "testsrc2=s=160x90:r=24000/1001:d=1"),
            *("-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1.033"),
            *("-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p"),
            *("-x264-params", "log-level=error", "-c:a", "flac", str(tmp_path / f"seg_{k}.mkv")),
        )
    listed.write_text("".join(f"file 'seg_{k}.mkv'\n" for k in range(8)))
    path = tmp_path / "joined.mkv"
    run("-f", "concat", "-safe", "0", "-i", str(listed), "-c", "copy", str(path))
    index = scan(path).index
    assert index is not None and index.frames == 192
    steps = np.diff(np.sort(index.pts))
    assert sorted(set(steps.tolist())) == [41, 42, 74] and int((steps == 74).sum()) == 7
    with pytest.raises(MediaError) as refused:
        examine(path)
    assert str(refused.value) == f"{path}: {variable(41, 74)}"
    with pytest.raises(MediaError) as refused:
        examine(path, frame_rate=NTSC_FILM)
    assert str(refused.value) == (
        f"{path}: --frame-rate 24000/1001: frame 24 lies 32.0 ms after its place on that rate's"
        " timeline (0.767 of a frame, half a frame being 20.9 ms), and 167 more frames, up to"
        f" 224.5 ms: {DROPPED_OR_DOUBLED}"
    )


def test_one_odd_step_shows_the_grid(caplog: pytest.LogCaptureFixture) -> None:
    # PROVISIONAL (media/source.py, _on_grid; a question for design). The common divisor
    # refuses frames whose every step spans a multiple of it, and no others: frames every 2
    # places of the 24/1 they declare are a contradicted rate, 12/1 named; with one step of 3
    # places among them, their grid is 24's, and they are read as an encoder that drops
    # repeated frames leaves a file (DESIGN.md, Input: "however many places are filled"): 14
    # places, 7 filled, the share said.
    film, twenty_fourth = declaring(Fraction(24)), Fraction(1, 24)
    even = scanned_at(np.arange(0, 14, 2, dtype=np.int64), twenty_fourth)
    assert unfilled(film, even).startswith(
        "its timestamps run at 12/1 fps (12), within 0 ms of that rate's timeline (0.00 of a"
        " frame), where it declares 24/1, its frames lasting from 83.3 to 83.3 ms: a bad file,"
        " not read on a guess: remake it at that rate"
    )
    odd = scanned_at(np.array([0, 2, 4, 6, 9, 11, 13], dtype=np.int64), twenty_fourth)
    index = filled_in(film, odd)
    assert index.frames == 14 and held_rows(index) == [1, 3, 5, 7, 8, 10, 12]
    assert filling(index) == Filling(14, 7, 6, Hole(1, 1, 0), 2)
    caplog.set_level(logging.WARNING, logger="seedvr2x")
    counted(film, FirstPass(index.frames, index=index))
    (warning,) = [record.getMessage() for record in caplog.records]
    assert warning.endswith(
        "read as a player shows it, the frame before each hole held through it: 14 frames, 7 of"
        " them filled (50.0%), 7 decoded; so large a share is what an encoder leaves that drops"
        " repeated frames, each picture held as it was until the next"
    )
    # The same of a 24 fps file in milliseconds given --frame-rate 48, one of its steps 1.5
    # frames, 3 places of 48's grid: refused without that step (test_frame_rate_with_places_
    # left_empty), read with it, 7 places of 14 filled.
    half = scanned_at(np.array([0, 42, 83, 125, 188, 229, 271], dtype=np.int64), MS)
    index = filled_in(declaring(Fraction(24), Fraction(48)), half)
    assert index.frames == 14 and held_rows(index) == [1, 3, 5, 7, 8, 10, 12]


def concat_joined(master: Path, directory: Path, cuts: list[int]) -> Path:
    """`master` split losslessly before each frame of `cuts` (the segment muxer, timestamps
    reset) and joined by ffmpeg's concat demuxer, the join a directory's refusal gives
    (cli._directory_refused), in `directory`."""
    directory.mkdir()
    run(
        *("-i", str(master), "-map", "0:v", "-c", "copy", "-f", "segment"),
        *("-reset_timestamps", "1", "-segment_frames", ",".join(str(cut) for cut in cuts)),
        str(directory / "seg_%06d.mkv"),
    )
    listed = directory / "list"
    listed.write_text("".join(f"file 'seg_{k:06d}.mkv'\n" for k in range(len(cuts) + 1)))
    path = directory / "joined.mkv"
    run("-f", "concat", "-i", str(listed), "-c", "copy", str(path))
    return path


def grid_joined(directory: Path, cuts: list[int], frames: int) -> Path:
    """The segments concat_joined left in `directory`, of `frames` frames at 24000/1001 in all,
    joined on the frame grid as sptenc's concat joins them (sptenc's ffmpeg/concat.go): a
    duration line a segment, its frames over the rate, and every timestamp snapped to the grid,
    the nearest frame's time rounded to the time base, by the setts bitstream filter."""
    listed = directory / "grid.ffconcat"
    lines = ["ffconcat version 1.0\n"]
    for k, (first, end) in enumerate(pairwise([0, *cuts, frames])):
        seconds = float((end - first) / NTSC_FILM)
        lines.append(f"file 'seg_{k:06d}.mkv'\nduration {seconds:.6f}\n")
    listed.write_text("".join(lines))
    snapped = "'if(eq({0},NOPTS),NOPTS,round(round({0}*TB*24000/1001)*1001/TB/24000))'"
    snap = f"setts=pts={snapped.format('PTS')}:dts={snapped.format('DTS')}"
    path = directory / "grid.mkv"
    run("-f", "concat", "-i", str(listed), "-c", "copy", "-bsf:v", snap, str(path))
    return path


@pytest.fixture(scope="module")
def master(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """2,400 frames at 24000/1001, FFV1, every frame a keyframe: cli._directory_refused's."""
    path = tmp_path_factory.mktemp("master") / "master.mkv"
    run(
        *("-f", "lavfi", "-i", "testsrc2=s=64x48:r=24000/1001", "-frames:v", "2400"),
        *("-c:v", "ffv1", "-g", "1", str(path)),
    )
    return path


def shot_like(seed: int) -> list[int]:
    """Cuts of 2,400 frames into segments of 5 to 40, drawn with `seed`."""
    drawn = random.Random(seed)
    cuts: list[int] = []
    while (cut := (cuts[-1] if cuts else 0) + drawn.randint(5, 40)) < 2400:
        cuts.append(cut)
    return cuts


@needs_ffmpeg
@pytest.mark.parametrize(
    ("cuts", "half", "strays", "off"),
    [
        # Every 20 and every 24 frames, and cut as shots are, in 117 segments: half a frame's
        # rule took them for files at 24/1, which holds their frames within 1.7, 1.3 and 20 ms,
        # and refused them so (a review's finding for the first two); every 25, for no rate's.
        (
            list(range(20, 2400, 20)),
            24,
            "1885 of its 2400 frames@500, up to 99.5 ms (2.38",
            "21.2@0.507",
        ),
        (
            list(range(24, 2400, 24)),
            24,
            "1887 of its 2400 frames@504, up to 99.5 ms (2.38",
            "21.0@0.503",
        ),
        (
            list(range(25, 2400, 25)),
            None,
            "1656 of its 2400 frames@726, up to 67.5 ms (1.62",
            "21.2@0.509",
        ),
        (shot_like(9), 24, "1749 of its 2400 frames@637, up to 81.5 ms (1.95", "21.2@0.508"),
    ],
    ids=["every 20 frames", "every 24", "every 25", "as shots"],
)
def test_concat_joins_that_drift_refused_without_a_rate(
    tmp_path: Path,
    master: Path,
    cuts: list[int],
    half: int | None,
    strays: str,
    off: str,
) -> None:
    # The join by ffmpeg's concat demuxer: each join about a millisecond early, the frames
    # drifting off the declared rate's timeline in steps, half a frame and more. Refused
    # (source._leaves), the message whole: how many frames, from which, how far; no rate named,
    # their timestamps being none's exactly, though the twin of the rate they declare holds some
    # within half a frame, 24/1, which --frame-rate would take them at, 0.1% off, a rate named
    # only where sptenc's rule fails (source._guided), which passes these; its ways out the join on
    # the frame grid and the remake at the rate declared, never --frame-rate. Given the rate it
    # declares, refused, --frame-rate's own rule, whole: told that it is refused without the
    # option too, with the same ways out. The same segments joined on the grid, as sptenc's
    # concat joins them, give the master's timestamps back: read.
    path = concat_joined(master, tmp_path / "join", cuts)
    index = scan(path).index
    assert index is not None and rate.exact(index) is None
    found = rate.twin(index, NTSC_FILM)
    assert (found and found.rate) == half
    count, first_off = strays.split("@")
    way_out = no_rate(NTSC_FILM)
    with pytest.raises(MediaError) as refused:
        examine(path)
    said = str(refused.value)
    assert said == (
        f"{path}: {leaves(count.removesuffix(' frames'), first_off, '24000/1001')}, and {way_out}"
    )
    assert "--frame-rate" not in said and "fps (" not in said and "24/1" not in said
    with pytest.raises(MediaError) as refused:
        examine(path, frame_rate=NTSC_FILM)
    first, furthest = first_off.split(" (")[0].split(", up to ")
    by, share = off.split("@")
    assert str(refused.value) == (
        f"{path}: --frame-rate 24000/1001: frame {first} lies {by} ms before its place on that"
        f" rate's timeline ({share} of a frame, half a frame being 20.9 ms), and"
        f" {int(count.split()[0]) - 1} more frames, up to {furthest}: {REFUSED_WITHOUT} {way_out}"
    )
    on_the_grid = examine(grid_joined(tmp_path / "join", cuts, 2400))
    assert (on_the_grid.frames, on_the_grid.frame_rate) == (2400, NTSC_FILM)
    kept, original = on_the_grid.index, scan(master).index
    assert kept is not None and original is not None
    assert kept.pts.tolist() == original.pts.tolist()


@needs_ffmpeg
def test_a_drifting_join_refused_and_its_remake_read(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # 600 frames at 24000/1001 split losslessly every 20 and joined by ffmpeg's concat demuxer,
    # 30 segments: each join 0.85 ms early, frame 500, after 25 of them, 21.2 ms before its
    # place, half a frame, the last frames 24.5 ms. Refused, whole: the frame, the drift, no
    # rate named, no --frame-rate. Its remake at the rate it declares, run as given, numbers
    # the frames as they come: every picture in order, at the timestamps of the master they
    # were split from, read. The same frames in 18 segments, drifting 14.5 ms, or in 4, 2.5 ms,
    # under half a frame: read, nothing said; in 4 joined on the frame grid, as sptenc's concat
    # joins them, the master's timestamps.
    master = tmp_path / "master.mkv"
    run(
        *("-f", "lavfi", "-i", "testsrc2=s=64x48:r=24000/1001", "-frames:v", "600"),
        *("-c:v", "ffv1", "-g", "1", str(master)),
    )
    original = scan(master).index
    assert original is not None
    path = concat_joined(master, tmp_path / "join", list(range(20, 600, 20)))
    with pytest.raises(MediaError) as refused:
        examine(path)
    said = str(refused.value)
    drifted = leaves("85 of its 600", "500, up to 24.5 ms (0.59", "24000/1001")
    assert said == f"{path}: {drifted}, and {no_rate(NTSC_FILM)}"
    assert "--frame-rate" not in said and "fps (" not in said
    command = said[said.index("ffmpeg -r:v:0") :]
    assert command.startswith("ffmpeg -r:v:0 24000/1001 -i SOURCE ")
    words = shlex.split(command)
    words[words.index("SOURCE")] = str(path)
    words[-1] = str(tmp_path / "remade.mkv")
    run(*words[1:])
    remade = tmp_path / "remade.mkv"
    assert pictures(remade) == pictures(master) and len(pictures(master)) == 600
    source = examine(remade)
    assert (source.frames, source.frame_rate) == (600, NTSC_FILM)
    assert source.index is not None and source.index.pts.tolist() == original.pts.tolist()
    for name, cuts, drift in (
        ("in18", list(range(20, 360, 20)), "14.5"),
        ("in4", [150, 300, 450], "2.5"),
    ):
        source = examine(concat_joined(master, tmp_path / name, cuts))
        assert (source.frames, source.frame_rate) == (600, NTSC_FILM)
        assert source.index is not None
        on = rate.timeline(source.index, NTSC_FILM)
        assert on.holds and f"{float(on.largest) * 1000:.1f}" == drift
    source = examine(grid_joined(tmp_path / "in4", [150, 300, 450], 600))
    assert (source.frames, source.frame_rate) == (600, NTSC_FILM)
    assert source.index is not None and source.index.pts.tolist() == original.pts.tolist()
    assert not strayed(caplog)


@needs_ffmpeg
def test_a_join_at_every_frame_off_the_grid_refused(tmp_path: Path) -> None:
    # 1,530 frames at 24000/1001 split losslessly after each of the first 30 and joined by
    # ffmpeg's concat demuxer: each join 0.71 ms early, the frames from 30 on half a frame or
    # more off the declared timeline, 21.5 ms, and no other rate holding them (24/1 holds the
    # first 996). Refused, the drift said, no rate named (source._leaves; DESIGN.md, Input).
    master = tmp_path / "master.mkv"
    run(
        *("-f", "lavfi", "-i", "testsrc2=s=64x48:r=24000/1001", "-frames:v", "1530"),
        *("-c:v", "ffv1", "-g", "1", str(master)),
    )
    path = concat_joined(master, tmp_path / "join", list(range(1, 31)))
    index = scan(path).index
    assert index is not None and rate.twin(index, NTSC_FILM) is None and rate.exact(index) is None
    assert rate.timeline(index, Fraction(24)).first_off == 996
    with pytest.raises(MediaError) as refused:
        examine(path)
    drifted = leaves("938 of its 1530", "30, up to 21.5 ms (0.51", "24000/1001")
    assert str(refused.value) == f"{path}: {drifted}, and {no_rate(NTSC_FILM)}"


# 30 segments of unequal lengths, 813 frames: of 24 and 48 frames first, the lengths a plain join
# drifts fastest on at 24000/1001, where they last whole milliseconds and their lossless split
# declares one less, 1 ms early a join and half a frame after 21 (30 segments of 1 to 47 frames
# drifted 19.4 ms, under it); then short ones, down to a single frame.
LENGTHS = [24, 48, 24, 24, 48, 24, 48, 24, 24, 24, 48, 24, 24, 48, 24, 24, 24, 48, 24, 48]
LENGTHS += [24, 24, 48, 24, 1, 2, 3, 20, 7, 12]

# What encodes a segment on its own, by codec, as an encoder run per segment does: two B-frames
# between the others, always, so that every segment long enough stores its frames out of order.
# FFV1's segments are the lossless split's own.
ENCODERS: dict[str, tuple[str, ...]] = {
    "ffv1": (),
    "h264": ("-c:v", "libx264", "-crf", "18", "-x264-params", "bframes=2:b-adapt=0"),
    "hevc": ("-c:v", "libx265", "-x265-params", "log-level=error:bframes=2:b-adapt=0"),
    "mpeg2": ("-c:v", "mpeg2video", "-bf", "2", "-q:v", "3"),
}


def segments(
    directory: Path,
    fps: str,
    codec: str,
    suffix: str = "mkv",
    lengths: Sequence[int] = LENGTHS,
    audio: bool = False,
) -> tuple[Path, list[str]]:
    """A directory of segments as a refusal finds one: an FFV1 master of sum(lengths) frames at
    `fps`, beside `directory`, split losslessly into segments of `lengths` frames (the segment
    muxer, timestamps reset), in `directory`, each then encoded on its own with ENCODERS[codec]
    where it has an encoder, in a `suffix` file, with a tone in AAC for `audio`. The master's
    path, and the pictures of every segment decoded on its own, in order (pictures)."""
    master = directory.parent / "master.mkv"
    run(
        *("-f", "lavfi", "-i", f"testsrc2=s=64x48:r={fps}", "-frames:v", str(sum(lengths))),
        *("-c:v", "ffv1", "-g", "1", str(master)),
    )
    encoder = ENCODERS[codec]
    split = directory if not encoder else directory.parent / "split"
    split.mkdir()
    run(
        *("-i", str(master), "-map", "0:v", "-c", "copy", "-f", "segment"),
        *("-reset_timestamps", "1", "-segment_frames"),
        ",".join(str(cut) for cut in list(accumulate(lengths))[:-1]),
        str(split / "seg_%06d.mkv"),
    )
    if not encoder:
        return master, pictures(master)
    directory.mkdir()
    tone = ("-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000") * audio
    sound = ("-map", "1:a:0", "-shortest", "-c:a", "aac", "-b:a", "96k") * audio
    seen: list[str] = []
    for k in range(len(lengths)):
        encoded = directory / f"seg_{k:06d}.{suffix}"
        run(
            *("-i", str(split / f"seg_{k:06d}.mkv"), *tone, "-map", "0:v:0", *sound),
            *(*encoder, "-threads", "1", str(encoded)),
        )
        seen += pictures(encoded)
    return master, seen


def typed(said: str, made: str) -> None:
    """The shell's lines a refusal `said` gives to make `made`, from their cd on, typed as
    they are."""
    end = said.index(made) + len(made)
    lines = said[said.rindex("cd ", 0, end) : end]
    result = subprocess.run(["sh", "-c", lines], stdin=subprocess.DEVNULL, capture_output=True)
    assert result.returncode == 0, result.stderr.decode()


def stored(path: Path) -> list[int]:
    """The pts of the packets of path's first video stream, in the order stored."""
    probe = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts"]
    probed = subprocess.run(
        [*probe, "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True
    )
    return [int(line.split(",")[0]) for line in probed.stdout.split()]


def placed(index: FrameIndex | None) -> list[int]:
    """Each frame's pts counted from the first frame's, as the frame grid is (DESIGN.md, Input:
    "counted from the first frame")."""
    assert index is not None
    return (index.pts - index.pts[0]).tolist()


@needs_ffmpeg
@pytest.mark.parametrize(
    ("codec", "fps", "suffix", "plain"),
    [
        ("ffv1", "24000/1001", "mkv", "refused"),
        ("ffv1", "25", "mkv", "holds"),
        ("ffv1", "30000/1001", "mkv", "drifts"),
        ("h264", "24000/1001", "mkv", "drifts"),
        ("hevc", "24000/1001", "mkv", "drifts"),
        ("mpeg2", "24000/1001", "mkv", "drifts"),
        ("mpeg2", "24000/1001", "ts", "variable frame rate: frames last from 21 to 42 ms"),
    ],
)
def test_the_join_advised_keeps_every_frame_on_the_grid(
    tmp_path: Path, codec: str, fps: str, suffix: str, plain: str
) -> None:
    # The join by ffmpeg alone that a refusal advises (rate.grid_join), typed as the message
    # gives it, the directory quoted for a shell, an apostrophe in it, and the rate filled in:
    # on 30 segments of unequal lengths, some of one, two and three frames, FFV1 split
    # losslessly, and H.264, HEVC and MPEG-2 each encoded on its own with two B-frames, their
    # frames stored out of order, in Matroska; and MPEG-2 in transport streams, none of a single
    # frame, the message's *.mkv their names. Every picture of every segment, in order, the
    # file read (examine), each frame at the place it has in the master the segments were cut
    # from, to the tick. The plain join of the same segments, ffmpeg's concat demuxer and a
    # stream copy alone: at 24000/1001, split segments 26.5 ms early by the end, refused, its
    # refusal naming this join at the rate it declares, the same lines but for the directory;
    # those encoded on their own 1.25 ms late, read, and transport streams 21.6 ms early, their
    # frames 21 to 42 ms apart, refused as a variable rate; at 30000/1001, 14.4 ms early, under
    # half a frame, read; at 25, a frame lasting whole milliseconds, on the grid.
    encoder = ENCODERS[codec]
    if encoder and not has_encoder(encoder[1]):
        pytest.skip(f"needs ffmpeg with {encoder[1]}")
    declares = Fraction(fps)
    lengths = [length for length in LENGTHS if length > 1 or suffix != "ts"]
    directory = tmp_path / "the film's segments"
    master, seen = segments(directory, fps, codec, suffix, lengths)
    original = scan(master).index
    assert original is not None and len(seen) == original.frames == sum(lengths)
    quoted = shlex.quote(str(directory))
    said = rate.grid_join(quoted, declares)
    listed = directory / "plain.txt"
    listed.write_text("".join(f"file 'seg_{k:06d}.{suffix}'\n" for k in range(len(lengths))))
    alone = tmp_path / "plain.mkv"
    run("-f", "concat", "-safe", "0", "-i", str(listed), "-c", "copy", str(alone))
    if plain in ("holds", "drifts"):
        assert (placed(examine(alone).index) == placed(original)) == (plain == "holds")
    else:
        with pytest.raises(MediaError) as refused:
            examine(alone)
        if plain != "refused":
            only = "and only a constant frame rate is supported"
            assert str(refused.value) == f"{alone}: {plain}, {only}"
        else:
            drifted = leaves("132 of its 813", "672, up to 26.5 ms (0.63", fps)
            assert str(refused.value) == f"{alone}: {drifted}, and {no_rate(declares)}"
            assert rate.grid_join(declares=declares).replace("DIR", quoted) == said
    typed(said.replace("*.mkv", f"*.{suffix}"), "../joined.mkv")
    joined = tmp_path / "joined.mkv"
    assert pictures(joined) == seen
    source = examine(joined)
    assert (source.frames, source.frame_rate) == (sum(lengths), declares)
    assert placed(source.index) == placed(original)
    order = stored(joined)
    assert (order != sorted(order)) == bool(encoder)


@needs_ffmpeg
def test_the_join_advised_takes_the_segments_in_name_order(tmp_path: Path) -> None:
    # Why the message says what its commands ask of the segments' names. 12 segments of 20
    # frames named seg_1.mkv to seg_12.mkv, the join and the remake typed as the message gives
    # them: the shell lists them by name, seg_10, seg_11 and seg_12 before seg_2, and so are
    # they joined, every frame on the grid, read without a word, the pictures out of order (a
    # review's finding, 2026-10-12). Named with leading zeros, as the message says: the
    # master's pictures, in order.
    directory = tmp_path / "segments"
    master, seen = segments(directory, "24000/1001", "ffv1", lengths=[20] * 12)
    original = scan(master).index
    assert original is not None and len(seen) == original.frames == 240
    for k in range(12):
        (directory / f"seg_{k:06d}.mkv").rename(directory / f"seg_{k + 1}.mkv")
    said = rate.grid_join(shlex.quote(str(directory)), NTSC_FILM)
    by_name = [0, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8]
    shuffled = [seen[20 * k + frame] for k in by_name for frame in range(20)]
    assert shuffled != seen and sorted(shuffled) == sorted(seen)
    for made in ("joined.mkv", "remade.mkv"):
        typed(said, f"../{made}")
        assert pictures(tmp_path / made) == shuffled
        source = examine(tmp_path / made)
        assert (source.frames, source.frame_rate) == (240, NTSC_FILM)
        assert placed(source.index) == placed(original)
        (tmp_path / made).unlink()
    for k in range(12):
        (directory / f"seg_{k + 1}.mkv").rename(directory / f"seg_{k + 1:06d}.mkv")
    for made in ("joined.mkv", "remade.mkv"):
        typed(said, f"../{made}")
        assert pictures(tmp_path / made) == seen
        assert placed(examine(tmp_path / made).index) == placed(original)


@needs_ffmpeg
@pytest.mark.parametrize(
    ("codec", "suffix", "audio", "copied"),
    [
        ("h264", "mkv", True, "variable frame rate: frames last from 0 to 84 ms"),
        ("mpeg2", "mpg", False, "Can't write packet with unknown timestamp"),
        ("mpeg2", "ts", False, "its timestamps jump at frame 769, from 33.450 to 32.074 s"),
    ],
    ids=["audio in Matroska", "program streams", "a transport stream of one frame"],
)
def test_segments_no_stream_copy_holds_are_remade(
    tmp_path: Path, codec: str, suffix: str, audio: bool, copied: str
) -> None:
    # The segments a refusal says no stream copy holds (rate.grid_join), the same 30, at
    # 24000/1001: H.264 with an AAC stream in Matroska, the join advised for the others giving
    # frames 0 to 84 ms apart, refused; MPEG-2 in program streams, whose packets without a
    # timestamp stop Matroska's muxer, ffmpeg failing; and in transport streams, one of them a
    # single frame, where ffmpeg finds no timestamp, the join jumping back there, refused. Their
    # remake, typed as the message gives it, *.mkv the segments' names: every picture of every
    # segment in order, read (examine), each frame at its place in the master they were cut
    # from, to the tick.
    encoder = ENCODERS[codec]
    if not has_encoder(encoder[1]):
        pytest.skip(f"needs ffmpeg with {encoder[1]}")
    directory = tmp_path / "the film's segments"
    master, seen = segments(directory, "24000/1001", codec, suffix, audio=audio)
    original = scan(master).index
    assert original is not None and len(seen) == original.frames == sum(LENGTHS)
    said = rate.grid_join(shlex.quote(str(directory)), NTSC_FILM).replace("*.mkv", f"*.{suffix}")
    joined = tmp_path / "joined.mkv"
    if suffix == "mpg":
        with pytest.raises(AssertionError, match=copied):
            typed(said, "../joined.mkv")
    else:
        typed(said, "../joined.mkv")
        with pytest.raises(MediaError) as refused:
            examine(joined)
        assert str(refused.value).startswith(f"{joined}: {copied}")
    typed(said, "../remade.mkv")
    remade = tmp_path / "remade.mkv"
    assert pictures(remade) == seen
    source = examine(remade)
    assert (source.frames, source.frame_rate) == (sum(LENGTHS), NTSC_FILM)
    assert placed(source.index) == placed(original)


@needs_ffmpeg
@pytest.mark.parametrize("container", ["mkv", "mp4"])
@pytest.mark.parametrize("fps", ["24000/1001", "24", "25", "30000/1001", "50", "60000/1001"])
def test_constant_rates_read(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, fps: str, container: str
) -> None:
    # 3,000 frames at each rate, FFV1: in Matroska, their timestamps rounded to the millisecond,
    # within half of one of their place on the declared rate's timeline; in MP4, whose time base
    # holds the rate, each on its tick. Read, nothing said. Matroska declares 60000/1001 as
    # 19001/317 to ffprobe, 2.6 microseconds off over these frames: read as 60000/1001
    # (rate.matroska_reading), within half a millisecond as the others, which an info line says.
    path = tmp_path / f"constant.{container}"
    run(
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=s=64x48:r={fps}",
        "-frames:v",
        "3000",
        "-c:v",
        "ffv1",
        str(path),
    )
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source = examine(path)
    read_from_19001_317 = (fps, container) == ("60000/1001", "mkv")
    assert (source.frames, source.frame_rate) == (3000, Fraction(fps))
    assert source.stream.ffprobe_rate == (Fraction(19001, 317) if read_from_19001_317 else None)
    assert ("Matroska's reading of 60000/1001" in caplog.text) is read_from_19001_317
    index = source.index
    assert index is not None
    on = rate.timeline(index, Fraction(fps))
    assert on.holds and on.largest <= (Fraction(1, 2000) if container == "mkv" else 0)
    assert not strayed(caplog)


@needs_ffmpeg
def test_a_video_starting_after_its_audio_read(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # 600 frames at 24000/1001 whose first is 23 ms after the file's start, where its audio
    # begins: more than half a frame, 20.9 ms. The timeline is counted from the first frame:
    # every frame within 0.5 ms of its place, read, nothing said.
    path = tmp_path / "late.mkv"
    run(
        *("-f", "lavfi", "-i", "testsrc2=s=64x48:r=24000/1001"),
        *("-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=26"),
        *("-frames:v", "600", "-filter:v", "settb=1/1000,setpts='floor(N*1001/24+0.5)+23'"),
        *("-map", "0:v", "-map", "1:a", "-c:a", "flac", "-fps_mode:v", "passthrough"),
        *("-enc_time_base:v", "1/1000", "-c:v", "ffv1", str(path)),
    )
    assert [s["start_time"] for s in streams(path)] == ["0.023000", "0.000000"]
    source = examine(path)
    assert (source.frames, source.frame_rate) == (600, NTSC_FILM)
    index = source.index
    assert index is not None and int(index.pts[0]) == 23
    assert rate.timeline(index, NTSC_FILM).largest == Fraction(1, 2000)
    assert not strayed(caplog)


@needs_ffmpeg
def test_a_timestamp_jump_refused_whatever_the_rate(tmp_path: Path) -> None:
    # Two MPEG-TS files joined, the second's timestamps 1,000 s after the first's
    # (tests/test_index.py): the jump is refused as what it is with --frame-rate too, before any
    # rate's timeline is looked at, which would only say its frame 50 lies 1,000 s from its place.
    path, _ = joined_ts(tmp_path, (0, 1000))
    for asked in (None, Fraction(25), NTSC_FILM):
        with pytest.raises(MediaError) as refused:
            examine(path, frame_rate=asked)
        said = str(refused.value)
        assert said.startswith(f"{path}: its timestamps jump at frame 50, from ")
        assert said.endswith("ffmpeg -i SOURCE -map 0 -map -0:d -c copy fixed.mkv")
        assert "--frame-rate" not in said


@needs_ffmpeg
@pytest.mark.parametrize("period", [12, 11])
def test_frame_rate_on_a_rate_under_ffmpegs_jump_bound(tmp_path: Path, period: int) -> None:
    # A frame every 12 s in Matroska, which declares 1000/1 for it: refused for the rate it
    # declares, with the one its timestamps run at, and read with --frame-rate 1/12. No step of
    # it is a jump, though each is past ffmpeg's 10 s (media/scan.py, JUMP_FORWARD), which
    # refused it as jumping "at frame 1, from 0.000 to 12.000 s", whatever the rate given.
    # Every 11 s: 1/11 fps, the rate its timestamps are exactly, where half a frame's was 1/12,
    # a rate looked for, "within 5000 ms of that rate's timeline (0.42 of a frame)" (a review's
    # finding).
    path = tmp_path / "slow.mkv"
    slow = f"testsrc2=s=64x48:r=1/{period}"
    run("-f", "lavfi", "-i", slow, "-frames:v", "6", "-c:v", "ffv1", str(path))
    with pytest.raises(MediaError) as refused:
        examine(path)
    exactly = Fraction(1, period)
    assert str(refused.value) == (
        f"{path}: its timestamps run at {rate.said(exactly)}, within 0 ms of that rate's"
        " timeline (0.00 of a frame), where it declares 1000/1, its frames lasting from"
        f" {period}000 to {period}000 ms: a bad file, not read on a guess: {fixes(exactly)}"
    )
    assert rate.said(exactly) == f"1/{period} fps ({'0.083' if period == 12 else '0.091'})"
    source = examine(path, frame_rate=exactly)
    assert (source.frames, source.frame_rate) == (6, exactly)
    index = source.index
    assert index is not None
    found = rate.exact(index)
    assert found is not None and found.rate == exactly
    assert rate.timeline(index, Fraction(1, 12)).holds


@needs_ffmpeg
@pytest.mark.parametrize("frames", [120, 3000])
def test_the_refusal_names_the_rate_the_timestamps_are_exactly(tmp_path: Path, frames: int) -> None:
    # 33 ms a frame declared 25/1: refused by sptenc's rule, 33 ms being no 40, with the rate
    # its timestamps are exactly, 1000/33 fps, whatever its length (media/source.py, _guided).
    # Half a frame's rate named it by its length: 91/3 fps over 120 frames, "within 3.9 ms of
    # that rate's timeline", which --frame-rate 91/3 then took, and none over 3,000, "the frame
    # rate declared, 25 fps, is not the frames' own" (a review's finding). Read with
    # --frame-rate 1000/33; another rate refused with the same one, and the fixes at it.
    path = retagged(tmp_path / "retagged.mkv", 25, "33*N", frames)
    exactly = Fraction(1000, 33)
    with pytest.raises(MediaError) as refused:
        examine(path)
    runs_at = "its timestamps run at 1000/33 fps (30.303), within 0 ms of that rate's timeline"
    assert str(refused.value) == (
        f"{path}: {runs_at} (0.00 of a frame), where it declares 25/1, its frames lasting from 33"
        f" to 33 ms: a bad file, not read on a guess: {fixes(exactly)}"
    )
    source = examine(path, frame_rate=exactly)
    assert (source.frames, source.frame_rate) == (frames, exactly)
    index = source.index
    assert index is not None
    assert rate.timeline(index, Fraction(91, 3)).holds is (frames == 120)
    assert rate.twin(index, Fraction(25)) is None
    with pytest.raises(MediaError) as refused:
        examine(path, frame_rate=Fraction(30))
    assert str(refused.value) == (
        f"{path}: --frame-rate 30/1: frame 50 lies 16.7 ms before its place on that rate's"
        f" timeline (0.500 of a frame, half a frame being 16.7 ms), and {frames - 51} more"
        f" frames, up to {'39.7' if frames == 120 else '999.7'} ms: taken at that rate, frames"
        f" would be dropped or doubled; {runs_at} (0.00 of a frame): {fixes(exactly)}"
    )


@needs_ffmpeg
def test_variable_rate_refused_without_guidance(tmp_path: Path) -> None:
    # Random steps of 33, 42 and 50 ms: no constant rate, the refusal as it was.
    path = tmp_path / "random.mkv"
    steps = random.Random(9).choices([33, 42, 50], k=60)
    times = "+".join(f"{step}*gte(N\\,{n + 1})" for n, step in enumerate(steps))
    run(
        *("-f", "lavfi", "-i", "testsrc2=s=64x48:r=25", "-frames:v", "60"),
        *("-vf", f"settb=1/1000,setpts='{times}'", "-fps_mode", "passthrough"),
        *("-enc_time_base", "1/1000", "-c:v", "ffv1", str(path)),
    )
    with pytest.raises(MediaError) as refused:
        examine(path)
    said = str(refused.value)
    assert said.startswith(f"{path}: variable frame rate: frames last from 33 to 50 ms")
    assert "--frame-rate" not in said and "remake" not in said


# The opt-in real source (the module's docstring).
RATE_SOURCE = os.environ.get("SEEDVR2X_RATE_SOURCE")
RATE_EXPECTED = os.environ.get("SEEDVR2X_RATE_EXPECTED")


@pytest.mark.skipif(
    not RATE_SOURCE,
    reason="needs SEEDVR2X_RATE_SOURCE, a source whose timestamps contradict its declared rate",
)
def test_real_source_refused_then_taken_at_its_rate() -> None:
    assert RATE_SOURCE is not None
    path = Path(RATE_SOURCE)
    declared = declare(path)
    started = time.monotonic()
    scanned = scan(path)
    seconds = time.monotonic() - started
    assert scanned.index is not None
    refusal = timing_refusal(declared, scanned)
    print(f"{path.name}: first pass {scanned.frames} frames in {seconds:.1f} s; refused: {refusal}")
    found = rate.twin(scanned.index, declared.stream.frame_rate)
    assert found is not None, "its declared rate's NTSC twin doesn't hold its frames a frame away"
    assert refusal.startswith(f"its timestamps run at {rate.said(found.rate)}, within")
    if RATE_EXPECTED:
        assert found.rate == Fraction(RATE_EXPECTED)
    accepted = timing_refusal(replace(declared, rate_override=found.rate), scanned)
    print(
        f"--frame-rate {rate.fraction(found.rate)}: {accepted or 'accepted'}; every frame within"
        f" {float(found.lowest) * 1000:+.2f} ... {float(found.highest) * 1000:+.2f} ms of its"
        f" timeline, {float(found.largest * found.rate):.3f} of a frame"
    )
    assert accepted == ""

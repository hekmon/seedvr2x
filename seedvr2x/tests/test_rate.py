"""A source's frame rate against its frames' own timestamps (DESIGN.md, Input: exact rational frame
rate; media/rate.py, media/source.py's timing_refusal): the timeline of a constant rate and its
half-frame bound, the rate the frames follow, the refusal's guidance and its remake command, and
--frame-rate, on indexes made here and on research/docs/seeking.md's mechanism 7 made on purpose
(mechanism7); and seedvr2x's own test of the files sptenc's rule passes, every frame within half
a frame of its place on the declared rate's timeline: other rates declared, refused naming the
rate when the timestamps are its exactly, joins by ffmpeg's concat demuxer that drift half a
frame, refused as no rate's, with the remake that puts them back; and what stays read, constant
rates to their time base's rounding at any length, joins that keep the frame grid or drift less,
a video starting after its audio. The command line's run with --frame-rate is test_cli_run.py's.

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
from dataclasses import replace
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from test_conversion import UNTAGGED_HD
from test_index import joined_ts
from test_probe import has_encoder

from seedvr2x.media import ffmpeg, rate
from seedvr2x.media.conversion import conversion_for
from seedvr2x.media.ffmpeg import MediaError
from seedvr2x.media.index import NOPTS, FrameIndex
from seedvr2x.media.scan import Scan, scan
from seedvr2x.media.source import (
    Declared,
    FirstPass,
    counted,
    declare,
    examine,
    timing_refusal,
)


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
    # 0.504 of one, and they drift 83.7 ms, 2 frames, by frame 1,999.
    found = rate.follows(indexed(live2(2000)))
    assert found is not None and found.rate == NTSC_FILM
    assert round(float(found.lowest) * 1000, 2) == -10.71
    assert round(float(found.highest) * 1000, 2) == 10.88
    assert round(float(found.largest * found.rate), 2) == 0.26
    at24 = rate.timeline(indexed(live2(2000)), Fraction(24))
    assert at24.first_off == 252 and at24.first_off_by == Fraction(21, 1000)
    assert at24.highest == Fraction(251, 3000)
    # The simplest fraction within the bounds the span allows is 2997/125, 23.976 exactly, which
    # holds the frames too; not a rate looked for (rate.TWELFTHS, rate.NTSC).
    assert rate.timeline(indexed(live2(2000)), Fraction(2997, 125)).holds
    assert Fraction(2997, 125) not in rate._rates(Fraction(23), Fraction(25))


def test_rates_looked_for() -> None:
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
    # constant rate holds them within half a frame.
    mixed = [round(1000 * (n / 24 if n < 12 else 0.5 + (n - 12) / 30)) for n in range(24)]
    assert rate.follows(indexed(mixed)) is None
    steps = random.Random(9).choices([33, 42, 50], k=200)
    assert rate.follows(indexed([sum(steps[:n]) for n in range(200)])) is None
    # Too few frames to tell one rate from the next, or a single frame: none.
    assert rate.follows(indexed([0, 1])) is None
    assert rate.follows(indexed([0])) is None


def test_jitter_follows_the_rate_declared() -> None:
    # Steps of 50 and 30 ms around 25 fps, every other frame 10 ms late: 10 ms off 25's timeline,
    # a quarter of a frame, and 9.96 off 25000/1001's, which drifts 0.04 ms a frame, the frames'
    # spread there 13.9 ms against 10: 25.
    late = [40 * n + (10 if n % 2 else 0) for n in range(100)]
    found = rate.follows(indexed(late))
    assert found is not None and found.rate == 25 and found.largest == Fraction(1, 100)
    other = rate.timeline(indexed(late), Fraction(25000, 1001))
    assert other.holds and other.largest < found.largest


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


GUIDANCE = (
    "its timestamps run at 24000/1001 fps (23.976), within 10.8 ms of that rate's timeline (0.26"
    " of a frame), where it declares 24/1, its frames lasting from 41 to 62 ms: a bad file, not"
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
    # Mechanism 7: refused, as sptenc's rule refuses its 62 ms steps, the message giving the rate
    # its timestamps follow, how far, and the two fixes (DESIGN.md, Input), whole.
    path = mechanism7(tmp_path / "s9.mkv")
    with pytest.raises(MediaError) as refused:
        examine(path)
    assert str(refused.value) == f"{path}: {GUIDANCE}"


@needs_ffmpeg
@needs_x264
def test_frame_rate_override(tmp_path: Path) -> None:
    # --frame-rate 24000/1001: every frame within half a frame of its timeline, taken at that
    # rate, the job's from then on; 24/1: frame 252 named, 0.504 of a frame after its place, and
    # the rate its timestamps run at given with the fixes.
    path = mechanism7(tmp_path / "s9.mkv")
    source = examine(path, frame_rate=NTSC_FILM)
    assert (source.frames, source.frame_rate, source.stream.frame_rate) == (260, NTSC_FILM, 24)
    with pytest.raises(MediaError) as refused:
        examine(path, frame_rate=Fraction(24))
    said = str(refused.value)
    assert said.startswith(
        f"{path}: --frame-rate 24/1: frame 252 lies 21.0 ms after its place on that rate's"
        " timeline (0.504 of a frame, half a frame being 20.8 ms), and 5 more frames, up to"
        " 21.3 ms: taken at that rate, frames would be dropped or doubled; its timestamps run at"
        " 24000/1001 fps (23.976), within 10.8 ms of that rate's timeline (0.26 of a frame):"
        " remake it at that rate"
    )
    assert said.endswith(
        "or give --frame-rate 24000/1001, which takes its frames at that rate,"
        " each checked to lie within half a frame of its timeline: the same"
        " frames on the same timeline, without the intermediate file"
    )


@needs_ffmpeg
def test_frame_rate_on_a_file_already_fine(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # 48 frames at 24000/1001, declared so: --frame-rate 24000/1001 accepted, said; 24 too, its
    # frames 2 ms off 24's timeline at most, taken at 24 as asked, warned of, the output 0.1%
    # shorter (media/source.py, PROVISIONAL); 25, 1.67 ms a frame off, refused from frame 12.
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
    path = mechanism7(tmp_path / "s9.mkv", late=late, audio=True)
    with pytest.raises(MediaError) as refused:
        examine(path)
    said = str(refused.value)
    command = said[said.index("ffmpeg -r:v:0") : said.index(" remade.mkv") + len(" remade.mkv")]
    words = shlex.split(command)
    words[words.index("SOURCE")] = str(path)
    words[-1] = str(tmp_path / "remade.mkv")
    run(*words[1:])
    remade = tmp_path / "remade.mkv"
    assert pictures(remade) == pictures(path) and len(pictures(path)) == 260
    before, after = streams(path), streams(remade)
    assert [s["codec_type"] for s in after] == ["video", "audio"]
    assert after[0]["r_frame_rate"] == after[0]["avg_frame_rate"] == "24000/1001"
    assert after[0]["start_time"] == before[0]["start_time"] == f"0.{late:03d}000"
    assert after[1]["start_time"] == before[1]["start_time"] == "0.000000"
    source = examine(remade)
    assert (source.frames, source.frame_rate) == (260, NTSC_FILM)


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
        half = rate.follows(steady)
        assert (half and half.rate) == (Fraction(91, 3) if frames < 3000 else None)
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
    # starting 1,000 ms after the one before, 1 ms early: 24/1 holds them within 1.33 ms, half a
    # frame's rate (follows), their deviations spread over 1.67 ticks; no rate's exactly.
    segment = [(2002 * n + 24) // 48 for n in range(24)]
    joined = indexed([1000 * k + pts for k in range(100) for pts in segment])
    half = rate.follows(joined)
    assert half is not None and (half.rate, half.largest) == (24, Fraction(4, 3000))
    assert half.highest - half.lowest == Fraction(5, 3000)
    assert rate.exact(joined) is None
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
        " drifts so, about 1 ms a join, and timestamps alone can't tell a drifting join from a"
        " retagged file: a bad file, not read on a guess: join its segments again so that every"
        f" frame keeps its place on the frame grid, {rate.grid_join()}; or remake it at the rate"
        " it declares, an FFV1 master numbering its frames at that rate as they come, every one"
        " kept, its other streams copied (lossless, so large: 0.53-0.55 MB a frame on a 1080p"
        f" film; and a full encode), with {rate.remake(declares, Fraction(0))}"
    )


def test_the_join_that_keeps_the_frame_grid() -> None:
    # The advice of a join on the frame grid, in one place (rate.grid_join): sptenc's concat,
    # with the segments' directory when the refusal knows it.
    assert rate.grid_join() == (
        "with sptenc, for segments of its split, sptenc concat DIR joined.mkv, which snaps every"
        " timestamp to the grid"
    )
    assert "sptenc concat 'my segments' joined.mkv" in rate.grid_join(shlex.quote("my segments"))


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


def test_matroskas_reading_of_a_rate_holds_for_hours() -> None:
    # PROVISIONAL (media/source.py, _leaves; a question for design). A Matroska file declares
    # 120000/1001 as 29011/242, ffprobe's reading of its frame duration in whole nanoseconds,
    # 3.2 ns a frame off: timestamps at 120000/1001, in milliseconds, hold that timeline for
    # 1,161,780 frames, 2 h 41 min, read; a frame more, refused by the rule as written, naming
    # the rate they are exactly, with what the declared one costs, said to its first digits.
    true, declared = Fraction(120000, 1001), Fraction(29011, 242)
    scanned = rounded(1_161_781, true, Fraction(1, 1000))
    index = scanned.index
    assert index is not None
    held = scanned_at(index.pts[:-1], Fraction(1, 1000))
    assert timing_refusal(declaring(declared), held) == ""
    assert timing_refusal(declaring(declared), scanned) == (
        "1 of its 1161781 frames lies half a frame or more from its place on the timeline of the"
        " rate it declares, 29011/242, from frame 1161780, up to 4.2 ms (0.50 of a frame), and"
        f" {another_rate(true, '0.000038% shorter')}"
    )
    assert timing_refusal(declaring(declared, true), scanned) == ""


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
    # their timestamps being none's exactly, though a rate looked for holds some within half a
    # frame, 24/1, which --frame-rate would take them at, 0.1% off; its ways out the join on
    # the frame grid and the remake at the rate declared, never --frame-rate. Given the rate it
    # declares, refused, --frame-rate's own rule, whole: told that it is refused without the
    # option too, with the same ways out. The same segments joined on the grid, as sptenc's
    # concat joins them, give the master's timestamps back: read.
    path = concat_joined(master, tmp_path / "join", cuts)
    index = scan(path).index
    assert index is not None and rate.exact(index) is None
    found = rate.follows(index)
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
    assert index is not None and rate.follows(index) is None and rate.exact(index) is None
    assert rate.timeline(index, Fraction(24)).first_off == 996
    with pytest.raises(MediaError) as refused:
        examine(path)
    drifted = leaves("938 of its 1530", "30, up to 21.5 ms (0.51", "24000/1001")
    assert str(refused.value) == f"{path}: {drifted}, and {no_rate(NTSC_FILM)}"


@needs_ffmpeg
@pytest.mark.parametrize("container", ["mkv", "mp4"])
@pytest.mark.parametrize("fps", ["24000/1001", "24", "25", "30000/1001", "50", "60000/1001"])
def test_constant_rates_read(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, fps: str, container: str
) -> None:
    # 3,000 frames at each rate, FFV1: in Matroska, their timestamps rounded to the millisecond,
    # within half of one of their place on the declared rate's timeline; in MP4, whose time base
    # holds the rate, each on its tick. Read, nothing said. Matroska declares 60000/1001 as
    # 19001/317 (media/source.py, _leaves), 2.6 microseconds off over these frames.
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
    source = examine(path)
    read_as_19001_317 = (fps, container) == ("60000/1001", "mkv")
    declares = Fraction(19001, 317) if read_as_19001_317 else Fraction(fps)
    assert (source.frames, source.frame_rate) == (3000, declares)
    index = source.index
    assert index is not None
    on = rate.timeline(index, declares)
    assert on.holds and on.largest <= (Fraction(503, 1_000_000) if container == "mkv" else 0)
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
    half = rate.follows(index)
    assert half is not None and half.rate == Fraction(1, 12)


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
    half = rate.follows(index)
    assert (half and half.rate) == (Fraction(91, 3) if frames == 120 else None)
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
    found = rate.follows(scanned.index)
    assert found is not None, "no constant rate holds its frames within half a frame"
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

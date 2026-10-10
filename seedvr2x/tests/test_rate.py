"""A source's frame rate against its frames' own timestamps (DESIGN.md, Input: exact rational frame
rate; media/rate.py, media/source.py's timing_refusal): the timeline of a constant rate and its
half-frame bound, the rate the frames follow, the refusal's guidance and its remake command, and
--frame-rate, on indexes made here and on research/docs/seeking.md's mechanism 7 made on purpose
(mechanism7); and the files sptenc's rule passes whose frames leave their declared rate's
timeline, read and warned of: other rates declared, named when the timestamps are theirs exactly,
and joins by ffmpeg's concat demuxer, which are no rate's. The command line's run with
--frame-rate is test_cli_run.py's.

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
from pathlib import Path

import numpy as np
import pytest
from test_index import joined_ts
from test_probe import has_encoder

from seedvr2x.media import ffmpeg, rate
from seedvr2x.media.ffmpeg import MediaError
from seedvr2x.media.index import NOPTS, FrameIndex
from seedvr2x.media.scan import scan
from seedvr2x.media.source import declare, examine, timing_refusal


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
    # Told that it is read without the option, and no other rate named: its timestamps are
    # the declared one's.
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
    """The end of --frame-rate's refusal for a file sptenc's rule passes at the rate it declares."""
    return (
        "taken at that rate, frames would be dropped or doubled; without --frame-rate, the file"
        f" is read at the rate it declares, {declares}, frame after frame, whatever their place"
        " on that rate's timeline"
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
    # half a tick; not always (rate.exact, PROVISIONAL): 24.03 fps over 1,000 frames, 1.03 ticks
    # on its mean rate's timeline.
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
    """The warnings logged of a source whose frames leave its declared rate's timeline."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and "half a frame or more" in record.getMessage()
    ]


READ_ON = (
    "read at the rate it declares all the same, frame after frame, as a file joined by ffmpeg's"
    " concat demuxer is"
)


def follows(found: Fraction, change: str) -> str:
    """The warning's end for timestamps that are the rate `found`'s exactly, in milliseconds."""
    return (
        f"its timestamps follow {rate.said(found)} exactly, to the rounding of its time base,"
        f" 1/1000 s: --frame-rate {rate.fraction(found)} takes its frames at that rate, where the"
        f" rate it declares makes its output {change} than its timestamps' own time, off the"
        " source's other streams"
    )


@needs_ffmpeg
@pytest.mark.parametrize(
    ("declared", "times", "frames", "found", "strays", "first_off"),
    [
        # 24000/1001 declared 24/1: each frame lasts 41 or 42 ms, within 1 ms of 24's 41.667, and
        # lies 0.042 ms further from 24's timeline than the one before, half a frame at 492.
        (24, "round(N*1001/24)", 600, NTSC_FILM, "100 of its 600", "492, up to 25.3 ms (0.61"),
        # 48 fps declared 50/1: 20 or 21 ms for 20, 4%, half a frame off by frame 12.
        (50, "round(N*1000/48)", 100, Fraction(48), "88 of its 100", "12, up to 83 ms (4.15"),
        # 33 ms a frame declared 30/1, 1000/33 fps, 1% off, a rate not looked for: half a frame
        # off by frame 50, whatever the length (refused over 120 and 300 frames as running at
        # 91/3 fps, which held them within 3.9 and 9.9 ms, and read over 3,000, said to follow
        # no constant rate).
        (30, "33*N", 120, Fraction(1000, 33), "70 of its 120", "50, up to 39.7 ms (1.19"),
        (30, "33*N", 300, Fraction(1000, 33), "250 of its 300", "50, up to 99.7 ms (2.99"),
        (30, "33*N", 3000, Fraction(1000, 33), "2950 of its 3000", "50, up to 999.7 ms (29.99"),
        # 17 ms a frame declared 60/1: 1000/17, not the 353/6 that holds 300 frames too.
        (60, "17*N", 300, Fraction(1000, 17), "275 of its 300", "25, up to 99.7 ms (5.98"),
    ],
)
def test_a_rate_contradicted_within_a_millisecond_read_and_warned_of(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    declared: int,
    times: str,
    frames: int,
    found: Fraction,
    strays: str,
    first_off: str,
) -> None:
    # Timestamps at one rate, another declared, each frame lasting within 1 ms of the declared
    # rate's: sptenc's rule passes, and the file is read, as before the frame index, its output
    # written at the declared rate, 0.1% to 4% off its frames' own time. Its frames leave the
    # declared rate's timeline, though: warned of, whole, with the rate its timestamps are
    # exactly and what the declared one does to the output (source._strays). Given that rate,
    # read at it without the warning, nor that of a file whose frames lie on its declared
    # timeline too; given the one declared, refused, --frame-rate's own rule.
    path = retagged(tmp_path / "retagged.mkv", declared, times, frames)
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source = examine(path)
    assert (source.frames, source.frame_rate) == (frames, declared)
    change = float(found / declared - 1) * 100
    changed = f"{abs(change):.3f}% {'shorter' if change < 0 else 'longer'}"
    assert strayed(caplog) == [
        f"{path}: {strays} frames half a frame or more from their place on the timeline of the"
        f" rate it declares, {declared}/1, from frame {first_off} of a frame): {READ_ON};"
        f" {follows(found, changed)}"
    ]
    caplog.clear()
    source = examine(path, frame_rate=found)
    assert (source.frames, source.frame_rate, source.stream.frame_rate) == (frames, found, declared)
    assert f"where it declares {declared}/1: taken at that rate" in caplog.text
    assert not strayed(caplog) and "too: taken" not in caplog.text
    first = first_off.split(",")[0]
    with pytest.raises(MediaError) as refused:
        examine(path, frame_rate=Fraction(declared))
    said = str(refused.value)
    assert said.startswith(f"{path}: --frame-rate {declared}/1: frame {first} lies")
    # Told that it is read without the option, and the rate its timestamps are exactly.
    assert said.endswith(
        f": {read_without(f'{declared}/1')}; its timestamps follow {rate.said(found)} exactly, to"
        f" the rounding of its time base, 1/1000 s: --frame-rate {rate.fraction(found)} takes its"
        " frames at that rate"
    )


@needs_ffmpeg
def test_a_short_file_on_its_declared_timeline_is_not_warned_of(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # 100 frames at 24000/1001 declared 24/1: within 4.4 ms of 24's timeline, every frame at its
    # place there. Read, nothing said: the output's 0.1% is under half a frame.
    path = retagged(tmp_path / "short.mkv", 24, "round(N*1001/24)", 100)
    source = examine(path)
    assert (source.frames, source.frame_rate) == (100, 24)
    index = source.index
    assert index is not None and rate.timeline(index, Fraction(24)).holds
    assert not strayed(caplog)


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
        # and refused them (a review's finding for the first two); every 25, for no rate's.
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
def test_concat_joins_read_and_warned_of_without_a_rate(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    master: Path,
    cuts: list[int],
    half: int | None,
    strays: str,
    off: str,
) -> None:
    # The join a directory's refusal gives: each join about a millisecond early, the frames
    # drifting off the declared rate's timeline in steps, half a frame and more. Read at the rate
    # declared, frame after frame, as before the frame index, and warned of, whole; no rate
    # named, their timestamps being none's exactly, though a rate looked for holds some within
    # half a frame. Given the rate it declares, refused, --frame-rate's own rule, whole: told
    # that it is read without the option, and no rate offered, where half a frame's was, 24/1
    # "within 1.7 ms of that rate's timeline" for the first, which --frame-rate 24/1 then read
    # at 24, 0.1% off, unwarned (a review's finding).
    path = concat_joined(master, tmp_path / "join", cuts)
    index = scan(path).index
    assert index is not None and rate.exact(index) is None
    found = rate.follows(index)
    assert (found and found.rate) == half
    source = examine(path)
    assert (source.frames, source.frame_rate) == (2400, NTSC_FILM)
    count, first_off = strays.split("@")
    assert strayed(caplog) == [
        f"{path}: {count} half a frame or more from their place on the timeline of the rate it"
        f" declares, 24000/1001, from frame {first_off} of a frame): {READ_ON}"
    ]
    with pytest.raises(MediaError) as refused:
        examine(path, frame_rate=NTSC_FILM)
    first, furthest = first_off.split(" (")[0].split(", up to ")
    by, share = off.split("@")
    assert str(refused.value) == (
        f"{path}: --frame-rate 24000/1001: frame {first} lies {by} ms before its place on that"
        f" rate's timeline ({share} of a frame, half a frame being 20.9 ms), and"
        f" {int(count.split()[0]) - 1} more frames, up to {furthest}:"
        f" {read_without('24000/1001')}"
    )


@needs_ffmpeg
def test_a_join_off_the_grid_read_at_the_rate_declared(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # 1,530 frames at 24000/1001 split losslessly after each of the first 30 and joined by
    # ffmpeg's concat demuxer: each join 0.71 ms early, the frames from 30 on half a frame or
    # more off the declared timeline, 21.5 ms, and no other rate holding them (24/1 holds the
    # first 996). Read at the rate declared, the drift warned of (source._strays; DESIGN.md,
    # Input).
    master = tmp_path / "master.mkv"
    run(
        *("-f", "lavfi", "-i", "testsrc2=s=64x48:r=24000/1001", "-frames:v", "1530"),
        *("-c:v", "ffv1", "-g", "1", str(master)),
    )
    path = concat_joined(master, tmp_path / "join", list(range(1, 31)))
    index = scan(path).index
    assert index is not None and rate.follows(index) is None and rate.exact(index) is None
    assert rate.timeline(index, Fraction(24)).first_off == 996
    source = examine(path)
    assert (source.frames, source.frame_rate) == (1530, NTSC_FILM)
    assert strayed(caplog) == [
        f"{path}: 938 of its 1530 frames half a frame or more from their place on the timeline of"
        f" the rate it declares, 24000/1001, from frame 30, up to 21.5 ms (0.51 of a frame):"
        f" {READ_ON}"
    ]


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

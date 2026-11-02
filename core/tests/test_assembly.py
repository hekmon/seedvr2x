"""Assembly (DESIGN.md, Output: Assembly; media/assembly.py): a job's segments joined into one
Matroska file with the source's other streams, held to the same frames written as one master
(its timestamps, packet for packet), to ffmpeg's own remux of the source (every other stream's
packets, the chapters, the records), to the source's own packets, and to seedvr2x's own reading
of the result as a source. Segments are FFV1 masters of FFV1Writer's, a frame counter in each
picture, and the same re-encoded with B-frames as a user's command would; sources are made with
ffmpeg. The streams Matroska refuses, the check of a damaged result, the room on the disk and a
stop are made on purpose."""

import json
import logging
import os
import random
import re
import shlex
import signal
import subprocess
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_probe import has_encoder

from seedvr2x.media import assembly, ffmpeg, files, rate
from seedvr2x.media.assembly import (
    LIST,
    WORK_FILES,
    Signature,
    Unmuxable,
    assemble,
    concat_list,
    grid,
    snap,
)
from seedvr2x.media.ffmpeg import MediaError
from seedvr2x.media.files import partial_path
from seedvr2x.media.index import FrameIndex
from seedvr2x.media.scan import scan
from seedvr2x.media.source import counted, declare, first_pass
from seedvr2x.media.writer import FFV1Writer, Tags
from seedvr2x.runtime import stop


def _usable() -> bool:
    try:
        ffmpeg.check(assembly=True)
    except MediaError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _usable(),
    reason="needs ffmpeg 7.1 or later with zscale, idet, ffv1, the concat demuxer and setts",
)

WIDTH, HEIGHT = 64, 48
TAGS = Tags("bt709", "bt709")
RATES = {
    "24000/1001": Fraction(24000, 1001),
    "25": Fraction(25),
    "30000/1001": Fraction(30000, 1001),
}
# The segments of a job here: more than sptenc's 163 of an episode would need to show a drift.
PARTS = 131
MILLISECOND = Fraction(1, 1000)


# What ffmpeg says, at the warning level, opening a Matroska file that holds a font whose MIME
# type it maps to no codec, the rich source's and any file assembled from it.
FONT = (
    b"Could not find codec parameters for stream 5 (Attachment: none): unknown codec",
    b"Consider increasing the value for the 'analyzeduration' (0) and 'probesize' (5000000)"
    b" options",
)


# What it says of an audio stream that declares no channel layout, PCM in Matroska.
GUESS = b"Guessed Channel Layout"


def run(*arguments: str, data: bytes | None = None) -> bytes:
    """ffmpeg run strictly: it must succeed and print nothing, but FONT and GUESS."""
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "warning", "-y", *arguments],
        input=data,
        capture_output=True,
        check=False,
    )
    said = [
        line for line in result.stderr.splitlines() if not line.endswith(FONT) and GUESS not in line
    ]
    assert result.returncode == 0 and not said, result.stderr.decode()
    return result.stdout


def loose(*arguments: str) -> None:
    """ffmpeg run for its file alone, whatever it warns of."""
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y", *arguments],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode()


def flat(first: int, count: int) -> np.ndarray:
    """count frames, every one the same grey: what a lossy encoder gives back as flat as it got
    it, where idet reads the noise it leaves on the numbered pictures as combing."""
    return np.full((count, HEIGHT, WIDTH, 3), 0.5, np.float32)


def pictures(first: int, count: int) -> np.ndarray:
    """Frames first to first + count - 1, (count, H, W, 3) float32: flat, so that idet finds no
    comb in them, the frame's number in its green plane, which a gbrp16le master holds as it is,
    round(g * 65535)."""
    frames = np.empty((count, HEIGHT, WIDTH, 3), np.float32)
    frames[..., 0], frames[..., 2] = 0.5, 0.25
    frames[..., 1] = (np.arange(first, first + count, dtype=np.float32) / 65535)[:, None, None]
    return frames


def write_master(
    path: Path,
    first: int,
    count: int,
    frame_rate: Fraction,
    drawn: Callable[[int, int], np.ndarray] = pictures,
) -> None:
    with FFV1Writer(path, "gbrp16le", WIDTH, HEIGHT, frame_rate, TAGS) as writer:
        writer.write(drawn(first, count))


@dataclass(frozen=True)
class Job:
    """A job's output: its segments, and the same frames as one master."""

    frame_rate: Fraction
    segments: list[tuple[Path, int]]
    master: Path
    directory: Path

    @property
    def frames(self) -> int:
        return sum(count for _, count in self.segments)


def make_job(
    directory: Path,
    frame_rate: Fraction,
    parts: int,
    seed: int = 1,
    drawn: Callable[[int, int], np.ndarray] = pictures,
) -> Job:
    """A job of `parts` segments of 1 to 6 frames, FFV1Writer's masters, written 4 at a time."""
    directory.mkdir(parents=True)
    chosen = random.Random(seed)
    counts = [chosen.randint(1, 6) for _ in range(parts)]
    firsts = [sum(counts[:place]) for place in range(parts)]
    segments = [(directory / f"seg_{place:06d}.mkv", count) for place, count in enumerate(counts)]
    with ThreadPoolExecutor(4) as pool:
        list(
            pool.map(
                lambda made: write_master(made[0][0], made[1], made[0][1], frame_rate, drawn),
                zip(segments, firsts, strict=True),
            )
        )
    master = directory / "master.mkv"
    write_master(master, 0, sum(counts), frame_rate, drawn)
    return Job(frame_rate, segments, master, directory)


@pytest.fixture(scope="module")
def jobs(tmp_path_factory: pytest.TempPathFactory) -> Callable[..., Job]:
    """Jobs by rate and number of segments, each made once."""
    made: dict[tuple[str, int], Job] = {}

    def get(name: str = "24000/1001", parts: int = PARTS) -> Job:
        if (name, parts) not in made:
            directory = tmp_path_factory.mktemp("job") / "segments"
            made[name, parts] = make_job(directory, RATES[name], parts)
        return made[name, parts]

    return get


def packets(path: Path) -> dict[int, list[tuple[str, ...]]]:
    """Each stream's packets, in the file's order: pts, dts, duration, size, flags, data hash."""
    result = subprocess.run(
        [
            *("ffprobe", "-v", "error", "-show_entries"),
            "packet=stream_index,pts,dts,duration,size,flags,data_hash",
            *("-show_data_hash", "md5", "-of", "csv=p=0", str(path)),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0 and not result.stderr, result.stderr
    found: dict[int, list[tuple[str, ...]]] = {}
    for line in result.stdout.split():
        fields = line.split(",")
        found.setdefault(int(fields[0]), []).append(tuple(fields[1:7]))
    return found


def shown(path: Path) -> list[int]:
    """The pts of the first video stream's packets, in presentation order."""
    return sorted(int(packet[0]) for packet in packets(path)[first_video(path)])


def ahead(path: Path) -> int:
    """How far, in milliseconds, another stream's packets run ahead of the first video stream's
    in the file's order, at most, by their timestamps, up to the last video packet."""
    result = subprocess.run(
        [
            *("ffprobe", "-v", "error", "-show_entries", "packet=stream_index,pts"),
            *("-of", "csv=p=0", str(path)),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    video = first_video(path)
    shown, worst, since = None, 0, 0
    for line in result.stdout.split():
        stream, pts = line.split(",")[:2]
        if int(stream) == video:
            shown, worst, since = int(pts), max(worst, since), 0
        elif pts != "N/A":
            since = max(since, int(pts) - (shown or 0))
    return worst


def probed(path: Path, entries: str) -> dict[str, Any]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", entries, "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def streams(path: Path) -> list[dict[str, Any]]:
    """Each stream's records: what it is, its tags, its dispositions set."""
    entries = (
        "stream=index,codec_type,codec_name,r_frame_rate,avg_frame_rate,sample_aspect_ratio"
        ",color_range,color_space,color_transfer,color_primaries,pix_fmt,width,height"
        ",channels,sample_rate:stream_tags:stream_disposition"
    )
    found = probed(path, entries)["streams"]
    for stream in found:
        stream["disposition"] = sorted(k for k, v in stream.get("disposition", {}).items() if v)
        stream.setdefault("tags", {})
    return found


def first_video(path: Path) -> int:
    return next(s["index"] for s in streams(path) if s["codec_type"] == "video")


def chapters(path: Path) -> list[tuple[str, str, dict[str, str]]]:
    found = probed(path, "chapter=start_time,end_time:chapter_tags")["chapters"]
    return [(c["start_time"], c["end_time"], c.get("tags", {})) for c in found]


def numbers(path: Path) -> list[int]:
    """The frame numbers of a gbrp16le file's pictures, decoded in order (pictures)."""
    # Each frame once: ffmpeg's constant-rate mode would double the first one of a video that
    # starts after its file does (DESIGN.md, Input).
    raw = run(
        *("-i", str(path), "-map", "0:v:0", "-fps_mode", "passthrough"),
        *("-f", "rawvideo", "-pix_fmt", "gbrp16le", "-"),
    )
    frames = np.frombuffer(raw, "<u2").reshape(-1, 3, HEIGHT, WIDTH)
    assert bool((frames[:, 0] == frames[:, 0, :1, :1]).all()), "a picture isn't flat"
    return [int(number) for number in frames[:, 0, 0, 0]]


def decoded(path: Path) -> list[str]:
    """The MD5 of each picture of the first video stream, decoded in order."""
    command = ("-i", str(path), "-map", "0:v:0", "-fps_mode", "passthrough", "-f", "framemd5", "-")
    lines = run(*command).decode().splitlines()
    return [line.split(",")[-1].strip() for line in lines if not line.startswith("#")]


def indexed(path: Path) -> FrameIndex:
    """The frame index of a source, its first pass's."""
    found = scan(path).index
    assert found is not None
    return found


def workdir(directory: Path) -> Path:
    work = directory / "out.mkv.work"
    work.mkdir(parents=True, exist_ok=True)
    return work


def assembled(directory: Path, job: Job, source: Path) -> Path:
    """The job assembled with the source's other streams, its work directory left empty."""
    output = directory / "out.mkv"
    work = workdir(directory)
    done = assemble(output, job.segments, job.frame_rate, source, indexed(source), work)
    assert done.frames == job.frames and done.size == output.stat().st_size
    assert not list(work.iterdir()) and not partial_path(output).exists()
    return output


# What the rich source holds besides its video: subtitles, chapters, a title, a font.
SUBTITLES = (
    "1\n00:00:01,000 --> 00:00:02,500\nHello\n\n2\n00:00:04,000 --> 00:00:05,000\n"
    "World <i>two</i>\n\n3\n00:00:08,000 --> 00:00:09,000\nEnd\n"
)
METADATA = (
    ";FFMETADATA1\ntitle=The Film\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=1000\ntitle=Logo\n"
    "[CHAPTER]\nTIMEBASE=1/1000\nSTART=1000\nEND=6000\ntitle=One\n"
    "[CHAPTER]\nTIMEBASE=1/1000\nSTART=6000\nEND=12000\ntitle=Two\n"
)


def sine(frequency: int, seconds: float) -> tuple[str, ...]:
    return ("-f", "lavfi", "-i", f"sine=f={frequency}:r=48000:d={seconds:.3f}")


def rich(directory: Path, job: Job) -> Path:
    """A Matroska source for the job: its video the job's own master, 23 ms after the audio, so
    that its frames' times are the grid's in the file's own milliseconds; FLAC and AAC tracks
    with languages, a title and a default flag; SubRip and ASS subtitles, one forced; three
    chapters; a title; a font whose MIME type ffmpeg maps to no codec, as mkvmerge's is."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "subs.srt").write_text(SUBTITLES)
    (directory / "meta.txt").write_text(METADATA)
    (directory / "font.ttf").write_bytes(random.Random(7).randbytes(3000))
    seconds = float(job.frames / job.frame_rate)
    source = directory / "rich.mkv"
    loose(
        *("-itsoffset", "0.023", "-i", str(job.master)),
        *(*sine(440, seconds + 0.1), *sine(880, seconds - 0.2)),
        *("-i", str(directory / "subs.srt"), "-i", str(directory / "subs.srt")),
        *("-i", str(directory / "meta.txt"), "-map_metadata", "5", "-map_chapters", "5"),
        *("-map", "0:v", "-map", "1:a", "-map", "2:a", "-map", "3:s", "-map", "4:s"),
        *("-c:v", "copy", "-c:a:0", "flac", "-c:a:1", "aac", "-c:s:0", "srt", "-c:s:1", "ass"),
        *("-metadata:s:v:0", "language=jpn", "-metadata:s:v:0", "title=Main video"),
        *("-metadata:s:v:0", "BPS=12345", "-metadata:s:v:0", "NUMBER_OF_FRAMES=9"),
        *("-metadata:s:a:0", "language=eng", "-metadata:s:a:0", "title=Stereo FLAC"),
        *("-metadata:s:a:1", "language=fra", "-metadata:s:s:0", "language=eng"),
        *("-metadata:s:s:1", "language=fra", "-metadata:s:s:1", "title=Full subs"),
        *("-disposition:v:0", "original", "-disposition:a:0", "default"),
        *("-disposition:a:1", "0", "-disposition:s:0", "0", "-disposition:s:1", "forced"),
        *("-attach", str(directory / "font.ttf"), "-metadata:s:t:0", "mimetype=font/ttf"),
        *("-metadata:s:t:0", "filename=font.ttf", str(source)),
    )
    return source


def transport(directory: Path, job: Job) -> Path:
    """An MPEG-TS source for the job, as many frames at its rate: the whole file starts at 1.4 s,
    its timestamps in 1/90000 s, MP2 and AC-3 beside the video, each starting at its own time;
    the video 45 ticks, half a millisecond, later than its encode put it, so that its start
    after the file's is far from a whole number of milliseconds."""
    directory.mkdir(parents=True, exist_ok=True)
    encode, source = directory / "encode.ts", directory / "source.ts"
    seconds = float(job.frames / job.frame_rate)
    loose(
        *("-f", "lavfi", "-i", f"color=c=gray:s={WIDTH}x{HEIGHT}:r={job.frame_rate}"),
        *(*sine(440, seconds + 0.1), *sine(880, seconds - 0.2)),
        *("-map", "0:v", "-map", "1:a", "-map", "2:a", "-frames:v", str(job.frames)),
        *("-c:v", "mpeg2video", "-c:a:0", "mp2", "-c:a:1", "ac3"),
        *("-metadata:s:a:0", "language=eng", "-metadata:s:a:1", "language=fra"),
        *("-f", "mpegts", str(encode)),
    )
    loose(
        *("-itsoffset", "0.0005", "-i", str(encode), "-i", str(encode)),
        *("-map", "0:v", "-map", "1:a", "-c", "copy", "-f", "mpegts", str(source)),
    )
    return source


def quicktime(directory: Path, job: Job) -> Path:
    """An MP4 source for the job: MPEG-4 video with B-frames, so an edit list, and AAC, whose
    first packet is its priming, before 0."""
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "source.mp4"
    seconds = float(job.frames / job.frame_rate)
    loose(
        *("-f", "lavfi", "-i", f"color=c=gray:s={WIDTH}x{HEIGHT}:r={job.frame_rate}"),
        *(*sine(440, seconds + 0.1), "-map", "0:v", "-map", "1:a"),
        *("-frames:v", str(job.frames), "-c:v", "mpeg4", "-bf", "2", "-c:a", "aac"),
        *("-metadata:s:a:0", "language=eng", "-map_chapters", "-1", str(source)),
    )
    return source


def remuxed(source: Path, *options: str) -> Path:
    """ffmpeg's own remux of a source into Matroska, every stream mapped."""
    remux = source.with_name(f"{source.name}.remux.mkv")
    loose("-i", str(source), "-map", "0", "-c", "copy", *options, str(remux))
    return remux


# --- the arithmetic


def test_the_grid_is_rounded_once_halves_away_from_zero() -> None:
    # Frame 12 at 24000/1001 is at 500.5 ms exactly, 501 as ffmpeg's rescaling writes it; after
    # a start that is no whole number of milliseconds, the start is inside the one rounding.
    film = Fraction(24000, 1001)
    places = grid(2000, film, Fraction(0))
    assert int(places[12]) == 501 and int(places[1]) == 42
    exact = [Fraction(1000 * k) / film for k in range(2000)]
    assert all(abs(int(places[k]) - exact[k]) <= Fraction(1, 2) for k in range(2000))
    late = Fraction(902, 90000)
    shifted = grid(2000, film, late)
    assert [int(t) for t in shifted[:4]] == [10, 52, 93, 135]
    assert all(abs(int(shifted[k]) - exact[k] - 1000 * late) <= Fraction(1, 2) for k in range(2000))
    # Not the same as the grid from 0 shifted by the start's own rounding: 57 ticks of 1/90000 s,
    # 0.63 ms, round to 1 ms, and a frame in three would be 1 ms late.
    ticks = grid(2000, film, Fraction(57, 90000))
    assert (
        int(ticks[0]) == 1 and sum(int(ticks[k]) != int(places[k]) + 1 for k in range(2000)) > 600
    )


def test_where_the_grid_starts() -> None:
    def index(pts: int, time_base: Fraction, start_time: int | None) -> FrameIndex:
        none = np.zeros(1, np.uint32)
        times = np.array([pts], np.int64)
        return FrameIndex(time_base, start_time, times, none, none.astype(np.bool_), times)

    # A Matroska file whose video starts 23 ms after its start; an MPEG-TS file's, 902 ticks of
    # 1/90000 s after the 132,606 its start time rounds to; no start time declared.
    assert assembly.start(index(23, MILLISECOND, 0)) == Fraction(23, 1000)
    assert assembly.start(index(133508, Fraction(1, 90000), 1473400)) == Fraction(902, 90000)
    assert assembly.start(index(40, MILLISECOND, None)) == Fraction(40, 1000)


def test_durations_start_every_segment_at_its_exact_time() -> None:
    # 800 segments of 121 frames at 24000/1001: each lasts 5.046708333... s. Whatever the
    # number of segments before it, a segment starts at its exact time rounded to the
    # microsecond, where durations rounded each on its own would start the last one 266
    # microseconds early.
    film = Fraction(24000, 1001)
    listed = concat_list([(Path(f"/s/{place}.mkv"), 121) for place in range(800)], film)
    lines = listed.splitlines()
    assert lines[0] == "ffconcat version 1.0" and lines[1] == "file 'file:/s/0.mkv'"
    durations = [round(Fraction(line.split()[1]) * 1_000_000) for line in lines[2::2]]
    assert all(
        line.startswith("duration ") and len(line.split(".")[1]) == 6 for line in lines[2::2]
    )
    begins = 0
    for place, duration in enumerate(durations):
        assert begins == round(Fraction(121 * place) / film * 1_000_000)
        begins += duration
    alone = round(Fraction(121) / film * 1_000_000)
    assert 799 * alone - round(Fraction(121 * 799) / film * 1_000_000) == -266


def test_a_name_is_escaped_by_the_demuxers_rules(tmp_path: Path, jobs: Callable[..., Job]) -> None:
    # A space, a quote, a backslash, and what a URL would read as its own: the demuxer opens the
    # very files, and the job assembles from them.
    job = jobs()
    odd = tmp_path / "we ird'q\\b#x?y%20z:1"
    odd.mkdir()
    names = ["a 'b'.mkv", "c\\d.mkv", "e#f?g%41:h.mkv"]
    segments: list[tuple[Path, int]] = []
    for name, (path, count) in zip(names, job.segments, strict=False):
        (odd / name).write_bytes(path.read_bytes())
        segments.append((odd / name, count))
    listed = concat_list(segments, job.frame_rate)
    assert (
        f"file 'file:{odd}/a '\\''b'\\''.mkv'".replace(str(odd), str(odd).replace("'", "'\\''"))
        in listed
    )
    few = Job(job.frame_rate, segments, job.master, odd)
    source = tmp_path / "source.mkv"
    write_master(source, 0, few.frames, job.frame_rate)
    output = assembled(tmp_path, few, source)
    assert numbers(output) == list(range(few.frames))
    with pytest.raises(MediaError, match="a line break in its name"):
        concat_list([(tmp_path / "a\nb.mkv", 1)], job.frame_rate)


# --- the video's timestamps


@pytest.mark.parametrize("name", list(RATES))
def test_the_timestamps_are_the_one_masters(
    tmp_path: Path, jobs: Callable[..., Job], name: str
) -> None:
    # The same frames as one master: every packet's pts, dts, duration, size, flags and data,
    # whatever the number of segments, and the pictures in order by decoding them.
    job = jobs(name)
    assert len(job.segments) >= 130
    output = assembled(tmp_path, job, job.master)
    assert packets(output) == packets(job.master)
    places = [int(place) for place in grid(job.frames, job.frame_rate, Fraction(0))]
    assert shown(output) == places
    assert numbers(output) == list(range(job.frames))
    # What the file says of itself: the rate, the segments' colour tags and sample aspect.
    [written], [master] = streams(output), streams(job.master)
    assert (
        written["r_frame_rate"]
        == written["avg_frame_rate"]
        == name.partition("/1")[0] + ("/1" if "/" not in name else "/1001")
    )
    for record in ("codec_name", "pix_fmt", "width", "height", "sample_aspect_ratio"):
        assert written.get(record) == master.get(record), record
    for record in ("color_range", "color_space", "color_transfer", "color_primaries"):
        assert written.get(record) == master.get(record), record
    assert written["tags"].get("ENCODER") == master["tags"].get("ENCODER")
    # The file ends where the grid does.
    ends = Fraction(probed(output, "format=duration")["format"]["duration"])
    assert abs(ends - job.frames / job.frame_rate) <= MILLISECOND / 2


def test_several_hundred_segments(tmp_path: Path, jobs: Callable[..., Job]) -> None:
    job = jobs("24000/1001", 420)
    output = assembled(tmp_path, job, job.master)
    assert packets(output) == packets(job.master)


def test_the_final_line_counts_in_the_singular(
    tmp_path: Path, jobs: Callable[..., Job], caplog: pytest.LogCaptureFixture
) -> None:
    # One segment, one stream, no chapter: "assembled, 6 frames of 1 segment with no other stream
    # and 0 chapters", not "1 segments" (next-batch item 14, wording; the other streams named by
    # kind, 2026-10-29).
    job = jobs("24000/1001", 1)
    with caplog.at_level(logging.INFO, logger="seedvr2x.media.assembly"):
        assembled(tmp_path, job, job.master)
    [done] = [r.getMessage() for r in caplog.records if ": assembled, " in r.getMessage()]
    assert f"assembled, {job.frames} frame" in done
    assert f" of 1 segment with no other stream, and 0 chapters of {job.master}," in done
    assert "1 segments" not in done and "1 frames" not in done

    assert assembly._counted(1, "chapter") == "1 chapter"  # pyright: ignore[reportPrivateUsage]
    assert assembly._counted(2, "other stream") == "2 other streams"  # pyright: ignore[reportPrivateUsage]


def test_without_the_snap_the_timestamps_would_be_off(
    tmp_path: Path, jobs: Callable[..., Job], monkeypatch: pytest.MonkeyPatch
) -> None:
    # What the snap is for: the durations alone leave a frame in four or five 1 ms off the
    # master's, each rounded twice, and the check refuses the file, naming the first.
    job = jobs()
    monkeypatch.setattr(assembly, "snap", lambda *_: "setts=ts=TS")
    with pytest.raises(MediaError, match=r"its video frame \d+, frame \d+ of .*seg_\d+\.mkv, is"):
        assembled(tmp_path, job, job.master)
    assert not (tmp_path / "out.mkv").exists() and not partial_path(tmp_path / "out.mkv").exists()


def test_the_snap_is_refused_beyond_what_it_computes_exactly() -> None:
    assert snap(Fraction(24000, 1001), Fraction(902, 90000), 10_000_000).startswith("setts=pts='")
    with pytest.raises(MediaError, match="too fine a rate and start"):
        snap(Fraction(2997003, 125000), Fraction(1, 90000), 10_000_000)


# --- the source's other streams


@pytest.mark.parametrize("name", ["24000/1001", "30000/1001"])
def test_with_a_matroska_source(tmp_path: Path, jobs: Callable[..., Job], name: str) -> None:
    job = jobs(name)
    source = rich(tmp_path / "source", job)
    output = assembled(tmp_path, job, source)
    ours, theirs, own = packets(output), packets(remuxed(source)), packets(source)
    # The video: the source's own timestamps, frame for frame, the master's 23 ms later.
    assert shown(output) == shown(source) == [23 + pts for pts in shown(job.master)]
    assert [p[3:] for p in ours[0]] == [p[3:] for p in packets(job.master)[0]]
    assert numbers(output) == list(range(job.frames))
    # Every other stream: the packets of ffmpeg's remux, and the source's own.
    assert set(ours) == set(theirs) == set(own) == {0, 1, 2, 3, 4}
    # A remux counts a stream's last packet a millisecond shorter or longer than the file it
    # reads, some times: every field but that one duration is the source's own.
    for stream in (1, 2, 3, 4):
        assert ours[stream] == theirs[stream], stream
        assert ours[stream][:-1] == own[stream][:-1], stream
        assert (
            ours[stream][-1][:2] + ours[stream][-1][3:] == own[stream][-1][:2] + own[stream][-1][3:]
        )
        assert len(ours[stream]) >= 3
    # The records, as ffmpeg's remux holds them; but the dispositions, the source's own, where a
    # plain remux marks the first subtitle stream default, none being.
    written, plain, sources = streams(output), streams(remuxed(source)), streams(source)
    names = [s.get("codec_name") for s in sources]
    assert names == ["ffv1", "flac", "aac", "subrip", "ass", None]
    assert [s.get("codec_name") for s in written] == names and written[0]["pix_fmt"] == "gbrp16le"
    # Interleaved as ffmpeg's own remux of the source: its inputs kept within 0.1 s of each other,
    # where the audio ran 10 s ahead of the video with the font mapped from the source, and 1.3 s
    # with the subtitles mapped from the audio's input (assembly._join).
    assert ahead(output) <= 250 and ahead(remuxed(source)) <= 250
    for ours_, theirs_, own_ in zip(written[1:], plain[1:], sources[1:], strict=True):
        assert ours_["tags"] == theirs_["tags"]
        assert ours_["disposition"] == own_["disposition"]
        assert {**ours_, "disposition": None} == {**theirs_, "disposition": None}
    assert plain[3]["disposition"] == ["default"] and written[3]["disposition"] == []
    assert written[4]["disposition"] == ["forced"] and written[1]["disposition"] == ["default"]
    assert written[5]["codec_type"] == "attachment" and written[5]["tags"] == {
        "filename": "font.ttf",
        "mimetype": "font/ttf",
    }
    assert (
        assembly.layout(output).streams[5].extradata == assembly.layout(source).streams[5].extradata
    )
    # The upscaled video's own: the source's video's language, title and dispositions, the
    # segments' encoder, and none of the source's statistics.
    assert sources[0]["tags"]["BPS"] == "12345"
    assert {k: v for k, v in written[0]["tags"].items() if k != "DURATION"} == {
        "language": "jpn",
        "title": "Main video",
        "ENCODER": streams(job.master)[0]["tags"]["ENCODER"],
    }
    assert written[0]["disposition"] == sources[0]["disposition"] == ["original"]
    assert written[0]["r_frame_rate"] == name
    # The chapters and the file's own tags.
    assert chapters(output) == chapters(remuxed(source)) == chapters(source)
    assert len(chapters(output)) == 3
    tags = "format_tags"
    assert probed(output, tags)["format"]["tags"]["title"] == "The Film"
    assert probed(output, tags) == probed(remuxed(source), tags)


@pytest.mark.parametrize("name", list(RATES))
def test_with_an_mpegts_source(tmp_path: Path, jobs: Callable[..., Job], name: str) -> None:
    # A whole-file start, 1.4 s, and a video that starts a few ticks of 1/90000 s after it: the
    # output's times are the source's less its start time, the video's those of ffmpeg's own
    # remux of the source's video, every one, the start inside the one rounding.
    job = jobs(name)
    source = transport(tmp_path / "source", job)
    index = indexed(source)
    assert index.start_time is not None and index.start_time > 1_000_000
    first = assembly.start(index)
    assert Fraction(1, 4) < first / MILLISECOND % 1 < Fraction(3, 4)
    output = assembled(tmp_path, job, source)
    remux = remuxed(source)
    assert shown(output) == shown(remux)
    assert shown(output) == [int(place) for place in grid(job.frames, job.frame_rate, first)]
    # Not the grid from 0 moved by the start's own rounding, which is 1 ms off on some frames,
    # but at 25 fps, where a frame lasts a whole number of milliseconds.
    places = [int(place) for place in grid(job.frames, job.frame_rate, Fraction(0))]
    written = shown(output)
    assert (written == [written[0] + place for place in places]) == (name == "25")
    ours, theirs = packets(output), packets(remux)
    assert ours[1] == theirs[1] and ours[2] == theirs[2] and len(ours[1]) > 100
    # The source's own packets, by their data, their times less the start, in milliseconds.
    own = packets(source)
    for stream in (1, 2):
        assert [p[3:] for p in ours[stream]] == [p[3:] for p in own[stream]]
        ticks = [int(p[0]) - round(index.start_time * Fraction(9, 100)) for p in own[stream]]
        assert all(
            abs(int(p[0]) - Fraction(t, 90)) <= 1 for p, t in zip(ours[stream], ticks, strict=True)
        )
    assert Fraction(probed(output, "format=start_time")["format"]["start_time"]) == 0
    assert numbers(output) == list(range(job.frames))


def test_with_an_mp4_source(tmp_path: Path, jobs: Callable[..., Job]) -> None:
    # An edit list and AAC's priming: the audio's first packet stays before 0, as ffmpeg's own
    # remux writes it, and the video is where that remux has the source's.
    job = jobs()
    source = quicktime(tmp_path / "source", job)
    output = assembled(tmp_path, job, source)
    remux = remuxed(source)
    assert (
        shown(output)
        == shown(remux)
        == [int(p) for p in grid(job.frames, job.frame_rate, Fraction(0))]
    )
    ours, theirs = packets(output), packets(remux)
    assert ours[1] == theirs[1]
    assert int(ours[1][0][0]) == -21 and int(packets(source)[1][0][0]) == -1024
    assert [(p[3], p[5]) for p in ours[1]] == [(p[3], p[5]) for p in packets(source)[1]]


def test_chapters_follow_the_sources_start(tmp_path: Path, jobs: Callable[..., Job]) -> None:
    # A Matroska file whose streams all start 2.5 s late: ffmpeg's remux takes the start off the
    # chapters too, leaves out the one that ends before it and holds the next one's start at 0,
    # and the assembled file's chapters are that remux's.
    job = jobs()
    late = tmp_path / "late.mkv"
    loose(
        *("-i", str(rich(tmp_path / "source", job)), "-map", "0", "-c", "copy"),
        *("-output_ts_offset", "2.5", str(late)),
    )
    assert assembly.layout(late).start_time == 2_500_000
    output = assembled(tmp_path, job, late)
    assert chapters(output) == chapters(remuxed(late))
    assert [c[:2] for c in chapters(output)] == [("0.000000", "3.500000"), ("3.500000", "9.500000")]
    assert shown(output) == shown(remuxed(late))


def test_twice_the_same(tmp_path: Path, jobs: Callable[..., Job]) -> None:
    # Assembled twice from the same segments: the same packets and timestamps on every stream,
    # the same records and chapters; not the same bytes, Matroska's muxer drawing its segment's
    # and its tracks' UIDs at random.
    job = jobs()
    source = rich(tmp_path / "source", job)
    first = assembled(tmp_path / "a", job, source)
    second = assembled(tmp_path / "b", job, source)
    assert packets(first) == packets(second)
    assert streams(first) == streams(second) and chapters(first) == chapters(second)
    one, other = first.read_bytes(), second.read_bytes()
    assert len(one) == len(other)
    differing = sum(a != b for a, b in zip(one, other, strict=True))
    assert 0 < differing <= 16 + 2 * 8 * 6 + 8, differing


# --- segments a user's command wrote


def encoded(directory: Path, job: Job, codec: str) -> Job:
    """The job's segments as a user's command would write them: each encoded on its own, with
    B-frames, tagged; one remuxed to start at 0.5 s, and another's timestamps jittered by a
    millisecond."""
    options = {
        "mpeg4": ("-c:v", "mpeg4", "-bf", "2", "-q:v", "4"),
        "libx264": ("-c:v", "libx264", "-preset", "ultrafast", "-bf", "3", "-threads", "1"),
        "libx265": (
            *("-c:v", "libx265", "-preset", "ultrafast", "-x265-params"),
            "log-level=error:pools=1:frame-threads=1",
        ),
    }[codec]
    tagged = (
        *("-vf", "format=yuv420p", "-colorspace", "smpte170m", "-color_range", "tv"),
        *("-color_primaries", "bt709", "-color_trc", "bt709"),
    )
    directory.mkdir(parents=True)
    segments = [(directory / path.name, count) for path, count in job.segments]

    def encode(made: tuple[tuple[Path, int], tuple[Path, int]]) -> None:
        loose("-i", str(made[0][0]), *tagged, *options, str(made[1][0]))

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(encode, zip(job.segments, segments, strict=True)))
    late, jittered = segments[3][0], segments[5][0]
    moved = late.with_name("moved.mkv")
    loose("-i", str(late), "-c", "copy", "-output_ts_offset", "0.5", str(moved))
    moved.replace(late)
    loose("-i", str(jittered), "-c", "copy", "-bsf:v", "setts=pts='PTS+mod(N,2)'", str(moved))
    moved.replace(jittered)
    return Job(job.frame_rate, segments, job.master, directory)


CODECS = [
    "mpeg4",
    *(
        pytest.param(codec, marks=pytest.mark.skipif(not has_encoder(codec), reason=f"no {codec}"))
        for codec in ("libx264", "libx265")
    ),
]


@pytest.mark.parametrize("codec", CODECS)
def test_segments_of_a_users_command(
    tmp_path: Path, jobs: Callable[..., Job], monkeypatch: pytest.MonkeyPatch, codec: str
) -> None:
    job = encoded(tmp_path / "encoded", jobs(), codec)
    late, jittered = job.segments[3], job.segments[5]
    assert min(int(p[0]) for p in packets(late[0])[0]) == 500
    jitter = [
        pts - place
        for pts, place in zip(
            shown(jittered[0]), grid(jittered[1], job.frame_rate, Fraction(0)), strict=True
        )
    ]
    assert max(jitter) - min(jitter) >= 1
    # Each such segment passes the check a run asks as it is written.
    for path, count in job.segments[:8]:
        assert assembly.segment_refusal(path, count, job.frame_rate) == "", path
    said: list[list[str]] = []
    ran = assembly._run  # pyright: ignore[reportPrivateUsage]

    def recorded(command: list[str]) -> tuple[int, list[str]]:
        status, lines = ran(command)
        said.append(lines)
        return status, lines

    monkeypatch.setattr(assembly, "_run", recorded)
    source = transport(tmp_path / "source", job)
    first = assembly.start(indexed(source))
    output = assembled(tmp_path, job, source)
    # ffmpeg printed nothing, at the warning level: no dts out of order, no pts before its dts.
    assert said and not any(said)
    # Every frame once, on the grid, in presentation order; the decode order's timestamps in
    # order, none after its packet's presentation.
    video = packets(output)[0]
    assert len(video) == job.frames
    assert shown(output) == [int(place) for place in grid(job.frames, job.frame_rate, first)]
    assert [int(p[0]) for p in video] != shown(output), "no B-frame in the test's segments"
    dts = [int(p[1]) for p in video if p[1] != "N/A"]
    assert dts == sorted(dts) and len(dts) >= job.frames - 2
    assert all(int(p[1]) <= int(p[0]) for p in video if p[1] != "N/A")
    # The packets are the segments', and the pictures the segments' own, decoded in order.
    data = [p[3:] for path, _ in job.segments for p in packets(path)[0]]
    assert [p[3:] for p in video] == data
    assert decoded(output) == [md5 for path, _ in job.segments for md5 in decoded(path)]


def test_segments_that_differ_are_refused(tmp_path: Path, jobs: Callable[..., Job]) -> None:
    # The concat demuxer gives every packet the first segment's codec headers: a segment of
    # another codec, size or settings is refused by name, before anything is written.
    job = jobs()
    odd = tmp_path / "seg_000002.mkv"
    loose("-i", str(job.segments[2][0]), "-c:v", "ffv1", "-level", "3", "-slices", "4", str(odd))
    segments = [*job.segments[:2], (odd, job.segments[2][1]), *job.segments[3:6]]
    with pytest.raises(MediaError, match=r"seg_000002\.mkv: can't be joined .*codec headers"):
        assemble(
            tmp_path / "out.mkv",
            segments,
            job.frame_rate,
            job.master,
            indexed(job.master),
            workdir(tmp_path),
        )
    assert not list(workdir(tmp_path).iterdir())
    first = assembly.signature(job.segments[0][0])
    assert first == Signature(0, "ffv1", WIDTH, HEIGHT, "gbrp16le", MILLISECOND, first.extradata)
    assert assembly.mismatch(first, assembly.signature(job.segments[1][0])) == ""
    small = tmp_path / "small.mkv"
    loose("-i", str(job.segments[2][0]), "-vf", "scale=32:24", "-c:v", "mpeg4", str(small))
    assert assembly.mismatch(first, assembly.signature(small)).startswith(
        "codec mpeg4, not ffv1, width 32, not 64, height 24, not 48, pixel format yuv420p, not"
    )


def two_encoded(directory: Path, encoder: tuple[str, ...], other: tuple[str, ...]) -> Job:
    """A job of two segments of 12 frames at 25 fps and 128x96, other pictures in each, the
    first written by `encoder`'s options and the second by them and `other`'s; its master,
    their 24 frames in FFV1, is a source for it."""
    directory.mkdir(parents=True)
    paths = [directory / "seg_000000.mkv", directory / "seg_000001.mkv"]
    drawn = ("-f", "lavfi", "-i", "testsrc2=s=128x96:r=25", "-frames:v", "12")
    loose(*drawn, "-pix_fmt", "yuv420p", *encoder, str(paths[0]))
    loose(*drawn, "-vf", "hue=h=90", "-pix_fmt", "yuv420p", *encoder, *other, str(paths[1]))
    master = directory / "master.mkv"
    loose(*drawn[:4], "-frames:v", "24", "-c:v", "ffv1", str(master))
    return Job(Fraction(25), [(paths[0], 12), (paths[1], 12)], master, directory)


@pytest.mark.skipif(not has_encoder("libx265"), reason="needs ffmpeg with libx265")
def test_headers_that_differ_and_decode_alike_are_joined(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # libx265 writes its build and its options in the codec headers, its threads among them: two
    # segments of one command on other machines, or across an update, have other headers and
    # the same parameters. The first's decode the second bit for bit, which is asked of ffmpeg:
    # joined, said once, and the file's frames are the segments'.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    x265 = ("-c:v", "libx265", "-crf", "16", "-x265-params")
    job = two_encoded(tmp_path / "job", (*x265, "frame-threads=1:log-level=error"), ())
    second = job.segments[1][0]
    loose(
        *("-f", "lavfi", "-i", "testsrc2=s=128x96:r=25", "-frames:v", "12", "-vf", "hue=h=90"),
        *("-pix_fmt", "yuv420p", *x265, "frame-threads=4:log-level=error", str(second)),
    )
    one, other = (assembly.signature(path) for path, _ in job.segments)
    assert one.extradata != other.extradata
    assert assembly.mismatch(one, other).startswith("codec headers (their hash) murmur3:")
    work = workdir(tmp_path)
    assert assembly.joins(job.segments[0], job.segments[1], job.frame_rate, work) == ""
    assert not list(work.iterdir())
    said = f"{second}: its codec headers aren't those of seg_000000.mkv, the job's first segment"
    assert caplog.text.count(said) == 1
    output = tmp_path / "out.mkv"
    assemble(output, job.segments, job.frame_rate, job.master, indexed(job.master), work)
    assert decoded(output) == [md5 for path, _ in job.segments for md5 in decoded(path)]
    assert len(decoded(output)) == 24 and not list(work.iterdir())
    assert assembly.signature(output).extradata == one.extradata


def test_headers_that_decode_otherwise_are_refused(tmp_path: Path) -> None:
    # MPEG-4 with and without quarter-pixel motion: other codec headers, and the second
    # segment's frames decode to other pictures under the first's, the decoder saying nothing.
    # Refused, by the frames alone, as each segment is asked and by the assembly.
    mpeg4 = ("-c:v", "mpeg4", "-bf", "2", "-q:v", "2")
    job = two_encoded(tmp_path / "job", mpeg4, ("-flags", "+qpel"))
    work = workdir(tmp_path)
    refusal = assembly.joins(job.segments[0], job.segments[1], job.frame_rate, work)
    assert refusal.startswith("codec headers (their hash) murmur3:")
    assert re.search(
        r", and under those of seg_000000\.mkv \d+ of its 12 frames decode to other pictures,"
        r" from frame \d+$",
        refusal,
    )
    assert not list(work.iterdir())
    with pytest.raises(MediaError, match=r"seg_000001\.mkv: can't be joined .* other pictures"):
        assemble(
            tmp_path / "out.mkv",
            job.segments,
            job.frame_rate,
            job.master,
            indexed(job.master),
            work,
        )
    assert not (tmp_path / "out.mkv").exists() and not list(work.iterdir())
    # The same settings: the same headers, nothing decoded to tell.
    same = two_encoded(tmp_path / "same", mpeg4, ())
    assert assembly.joins(same.segments[0], same.segments[1], same.frame_rate, work) == ""


def test_a_segment_is_refused_as_soon_as_written(tmp_path: Path, jobs: Callable[..., Job]) -> None:
    job = jobs()
    path, count = max(job.segments, key=lambda segment: segment[1])
    assert assembly.segment_refusal(path, count, job.frame_rate) == ""
    assert assembly.segment_refusal(path, count + 1, job.frame_rate) == (
        f"{count} video packets, where the segment has {count + 1} frames"
    )
    # Frames at 25 fps in a job at 24000/1001: half a frame off within 13 frames.
    other = tmp_path / "other.mkv"
    write_master(other, 0, 14, Fraction(25))
    assert "its frame 12 is at 480 ms, -20.5 ms from its place at 24000/1001 fps" in (
        assembly.segment_refusal(other, 14, job.frame_rate)
    )
    # A file named .mkv that isn't Matroska: its timestamps in another unit.
    nut = tmp_path / "nut.mkv"
    loose("-i", str(path), "-c", "copy", "-f", "nut", str(nut))
    refusal = assembly.segment_refusal(nut, count, job.frame_rate)
    assert re.match(
        r"its video stream's time base is 1/\d+ s, where a Matroska file's is 1/1000", refusal
    )
    with pytest.raises(MediaError, match=r"nut\.mkv: its video stream's time base is 1/\d+ s"):
        assemble(
            tmp_path / "out.mkv",
            [(nut, count)],
            job.frame_rate,
            job.master,
            indexed(job.master),
            workdir(tmp_path),
        )


# --- read back as a source


def read_back(output: Path, caplog: pytest.LogCaptureFixture) -> tuple[int, Fraction, FrameIndex]:
    """The file given to seedvr2x as a source: its frames, its rate and its frame index, nothing
    logged at the warning level."""
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        declared = declare(output)
        found = first_pass(declared)
        source = counted(declared, found)
    assert not caplog.records, [record.getMessage() for record in caplog.records]
    assert source.index is not None
    return source.frames, source.frame_rate, source.index


@pytest.mark.parametrize("kind", ["masters with a Matroska source", "B-frames with an MPEG-TS one"])
@pytest.mark.parametrize("name", list(RATES))
def test_read_back_as_a_source(
    tmp_path: Path,
    jobs: Callable[..., Job],
    caplog: pytest.LogCaptureFixture,
    name: str,
    kind: str,
) -> None:
    # The user's condition: a file seedvr2x builds is never refused by seedvr2x. Accepted, as
    # many frames, at the job's rate, every frame at its place on that rate's timeline.
    job = jobs(name)
    if kind.startswith("masters"):
        source = rich(tmp_path / "source", job)
    else:
        grey = make_job(tmp_path / "grey", job.frame_rate, PARTS, drawn=flat)
        job = encoded(tmp_path / "encoded", grey, "mpeg4")
        source = transport(tmp_path / "source", job)
    first = assembly.start(indexed(source))
    output = assembled(tmp_path, job, source)
    frames, frame_rate, index = read_back(output, caplog)
    assert (frames, frame_rate, index.time_base) == (job.frames, job.frame_rate, MILLISECOND)
    # Half a millisecond from its place on the grid, the one rounding.
    times = [int(pts) * MILLISECOND for pts in index.pts]
    assert all(
        abs(time - first - place / job.frame_rate) <= MILLISECOND / 2
        for place, time in enumerate(times)
    )
    # The stricter reading: less than half a frame from its place, counted from the first
    # frame's own time, which the millisecond has rounded too: within 1 ms, and within half of
    # one after a start that is a whole number of milliseconds.
    on = rate.timeline(index, job.frame_rate)
    whole = (first / MILLISECOND).denominator == 1
    assert on.holds and on.largest <= (MILLISECOND / 2 if whole else MILLISECOND)
    assert whole == kind.startswith("masters")


# --- what Matroska can't take


def made(directory: Path, kind: str) -> Path:
    """A source holding a stream of `kind` beside its MPEG-4 video and its audio: its name, the
    inputs after the video's and the audio's, and the output's options."""
    directory.mkdir(parents=True, exist_ok=True)
    subs, meta, cover, captions = (
        directory / name for name in ("subs.srt", "meta.txt", "cover.png", "cc.scc")
    )
    subs.write_text(SUBTITLES)
    meta.write_text(METADATA)
    caption = "9420 9420 94ae 94ae 9470 9470 c8e5 ecec ef80 942f 942f"
    captions.write_text(f"Scenarist_SCC V1.0\n\n00:00:00:10\t{caption}\n\n")
    loose("-f", "lavfi", "-i", "color=red:s=32x32:d=1", "-frames:v", "1", str(cover))
    both = ("-map", "0:v", "-map", "1:a", "-c:v", "mpeg4")
    mpeg2 = ("-map", "0:v", "-map", "1:a", "-c:v", "mpeg2video")
    third = ("-map", "2", "-t", "2")
    kinds: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
        "mov_text": (
            "a.mp4",
            ("-i", str(subs)),
            (*both, *third, "-c:a", "aac", "-c:s", "mov_text"),
        ),
        "timecode": ("a.mov", (), (*both, "-c:a", "pcm_s16be", "-timecode", "01:00:00:00")),
        "chapter track": ("a.mp4", ("-i", str(meta)), (*both, "-map_chapters", "2", "-c:a", "aac")),
        "captions": (
            "a.mov",
            ("-i", str(captions)),
            (*both, *third, "-c:a", "aac", "-c:s", "copy"),
        ),
        "pcm_bluray": (
            "a.m2ts",
            (),
            (*mpeg2, "-c:a", "pcm_bluray", "-f", "mpegts", "-mpegts_m2ts_mode", "1"),
        ),
        "pcm_dvd": ("a.vob", (), (*mpeg2, "-c:a", "pcm_dvd", "-f", "vob")),
        "cover in MP4": (
            "a.mp4",
            ("-i", str(cover)),
            (
                *both,
                "-map",
                "2:v",
                "-c:a",
                "aac",
                "-c:v:1",
                "png",
                "-disposition:v:1",
                "attached_pic",
            ),
        ),
        "cover in Matroska": (
            "a.mkv",
            (),
            (
                *(*both, "-c:a", "flac", "-attach", str(cover)),
                *("-metadata:s:t:0", "mimetype=image/png", "-metadata:s:t:0", "filename=cover.png"),
            ),
        ),
        "WebVTT": ("a.mkv", ("-i", str(subs)), (*both, *third, "-c:a", "flac", "-c:s", "webvtt")),
        "PCM": ("a.mov", (), (*both, "-map", "1:a", "-c:a:0", "pcm_s16be", "-c:a:1", "pcm_s24le")),
    }
    name, inputs, options = kinds[kind]
    video = ("-f", "lavfi", "-i", f"color=c=gray:s={WIDTH}x{HEIGHT}:r=25:d=2")
    loose(*video, *sine(440, 2), *inputs, *options, str(directory / name))
    return directory / name


@pytest.mark.parametrize(
    ("kind", "refused", "advised"),
    [
        (
            "mov_text",
            [
                Unmuxable(
                    2, "subtitle", "mov_text", "Subtitle codec mov_text (94213) is not supported."
                )
            ],
            "ffmpeg -i SOURCE -map 0 -c copy -c:2 srt remuxed.mkv\n(stream 2: -c:2 srt converts it"
            " to SubRip, its text and times kept, its styling and placement lost)",
        ),
        (
            "timecode",
            [
                Unmuxable(
                    2, "data", "", "Only audio, video, and subtitles are supported for Matroska."
                )
            ],
            "",
        ),
        (
            "chapter track",
            [
                Unmuxable(
                    2,
                    "data",
                    "bin_data",
                    "Only audio, video, and subtitles are supported for Matroska.",
                )
            ],
            "",
        ),
        (
            "captions",
            [
                Unmuxable(
                    2, "subtitle", "eia_608", "Subtitle codec eia_608 (94218) is not supported."
                )
            ],
            "ffmpeg -i SOURCE -map 0 -c copy -map -0:2 remuxed.mkv\n(stream 2 left out)",
        ),
        (
            "pcm_bluray",
            [Unmuxable(1, "audio", "pcm_bluray", "No wav codec tag found for codec pcm_bluray")],
            "ffmpeg -i SOURCE -map 0 -c copy -c:1 pcm_s24le remuxed.mkv\n(stream 1: -c:1 pcm_s24le"
            " rewraps its samples as they are, nothing lost)",
        ),
        (
            "pcm_dvd",
            [Unmuxable(1, "audio", "pcm_dvd", "No wav codec tag found for codec pcm_dvd")],
            "ffmpeg -i SOURCE -map 0 -c copy -map -0:1 remuxed.mkv\n(stream 1 left out)",
        ),
        ("cover in MP4", [], ""),
        ("cover in Matroska", [], ""),
        ("WebVTT", [], ""),
        ("PCM", [], ""),
    ],
)
def test_what_matroska_cant_take(
    tmp_path: Path,
    kind: str,
    refused: list[Unmuxable],
    advised: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # By ffmpeg's own verdict, each stream alone: the stream, its kind, its codec and ffmpeg's
    # words; and the remux the refusal gives is one Matroska takes whole. A data stream is no
    # refusal (DESIGN.md, Output, Assembly, decided on 2026-10-12): left out, and warned of.
    source = made(tmp_path, kind)
    work = workdir(tmp_path)
    found = assembly.unmuxable(source)
    assert found == refused
    if not refused:
        assert assembly.preflight(source, work) == ""
        return
    if refused[0].kind == "data":
        with caplog.at_level(logging.WARNING, logger="seedvr2x.media.assembly"):
            assert assembly.preflight(source, work) == ""
        assert assembly.unmuxable_refusal(source, found) == "" and advised == ""
        [warned] = [record.getMessage() for record in caplog.records]
        assert warned.startswith(f"{source}: Matroska can't hold one of its data streams, which")
        assert f"stream 2 (data{', bin_data' if kind == 'chapter track' else ''}): ffmpeg says" in (
            warned
        )
        assert "is left out of the finished file" in warned
        # The chapters of the source are copied all the same, said where it has some.
        assert ("its 3 chapters are copied all the same" in warned) == (kind == "chapter track")
        return
    said = assembly.preflight(source, work)
    assert said == assembly.unmuxable_refusal(source, found)
    assert said.startswith(f"{source}: Matroska can't hold one of its streams as it is: stream ")
    assert f'ffmpeg says "{refused[0].reason}"' in said
    assert advised in said and "or give -o a directory" in said
    if kind == "pcm_dvd":
        # The remux of this source, MPEG-PS, fails whatever it leaves out: its video packets
        # don't all carry a timestamp ("Can't write packet with unknown timestamp"), which ffmpeg
        # makes up when asked (-fflags +genpts): another matter than the stream refused.
        return
    command = advised.partition("\n")[0].replace("SOURCE", shlex.quote(str(source)))
    remux = tmp_path / "remuxed.mkv"
    loose(*shlex.split(command.replace("remuxed.mkv", shlex.quote(str(remux))))[1:])
    assert assembly.preflight(remux, work) == ""
    kept = [s["codec_type"] for s in streams(remux)]
    assert len(kept) == len(streams(source)) - ("left out" in advised)
    if kind == "pcm_bluray":
        # The samples as they are: both decode to the same.
        pcm = ("-map", "0:a:0", "-f", "s32le", "-")
        assert run("-i", str(source), *pcm) == run("-i", str(remux), *pcm) != b""
    if kind == "mov_text":
        text = run("-i", str(remux), "-map", "0:s:0", "-f", "srt", "-").decode()
        assert "Hello" in text and "00:00:01,000 --> 00:00:02,500" in text


def test_several_streams_refused_at_once(tmp_path: Path) -> None:
    # A stream left out before one converted moves it up: the option names its place in the
    # remux, and the remux is one Matroska takes.
    (tmp_path / "subs.srt").write_text(SUBTITLES)
    source = tmp_path / "a.mov"
    loose(
        *("-f", "lavfi", "-i", f"color=c=gray:s={WIDTH}x{HEIGHT}:r=25:d=2", *sine(440, 2)),
        *("-i", str(tmp_path / "subs.srt"), "-map", "0:v", "-map", "1:a", "-map", "2:s"),
        *("-c:v", "mpeg4", "-c:a", "aac", "-c:s", "mov_text", "-timecode", "01:00:00:00"),
        *("-t", "2", str(source)),
    )
    found = assembly.unmuxable(source)
    kinds = {stream.codec or stream.kind: stream.index for stream in found}
    assert set(kinds) == {"mov_text", "data"}
    said = assembly.unmuxable_refusal(source, found)
    # The refusal names the stream it refuses alone, the data stream being left out (PROVISIONAL
    # in unmuxable_refusal: the remux it advises leaves that one out too).
    assert "Matroska can't hold one of its streams as it is" in said
    assert f"stream {kinds['mov_text']} (subtitle, mov_text)" in said
    assert f"stream {kinds['data']} (data" not in said.partition("ffmpeg -i")[0]
    command = said.partition("without it:\n")[2].partition("\n")[0]
    assert f"-map -0:{kinds['data']}" in command and f"-c:{kinds['mov_text']} srt" in command
    remux = tmp_path / "remuxed.mkv"
    loose(
        *shlex.split(
            command.replace("SOURCE", shlex.quote(str(source))).replace(
                "remuxed.mkv", shlex.quote(str(remux))
            )
        )[1:]
    )
    assert [s["codec_name"] for s in streams(remux)] == ["mpeg4", "aac", "subrip"]
    assert assembly.preflight(remux, workdir(tmp_path)) == ""


def data_source(directory: Path, job: Job, kind: str) -> Path:
    """A source for the job holding a data stream Matroska can't take, after its MPEG-4 video and
    its AAC audio: "timecode", a MOV's timecode track; "chapters", an MP4 with three chapters,
    whose chapter track ffmpeg writes as a data stream (made's "chapter track")."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "meta.txt").write_text(METADATA)
    seconds = float(job.frames / job.frame_rate)
    source = directory / ("a.mov" if kind == "timecode" else "a.mp4")
    loose(
        *("-f", "lavfi", "-i", f"color=c=gray:s={WIDTH}x{HEIGHT}:r={job.frame_rate}"),
        *(*sine(440, seconds + 0.1), "-i", str(directory / "meta.txt")),
        *("-map", "0:v", "-map", "1:a", "-frames:v", str(job.frames), "-c:v", "mpeg4"),
        *("-c:a", "aac", *(("-timecode", "01:00:00:00") if kind == "timecode" else ())),
        *("-map_chapters", "-1" if kind == "timecode" else "2", str(source)),
    )
    return source


@pytest.mark.parametrize("kind", ["timecode", "chapters"])
def test_a_data_stream_is_left_out_with_a_warning(
    tmp_path: Path, jobs: Callable[..., Job], caplog: pytest.LogCaptureFixture, kind: str
) -> None:
    # DESIGN.md (Output, Assembly, decided on 2026-10-12): a MOV's timecode track and an MP4's
    # chapter track, which Matroska can't hold, are left out of the finished file with a warning
    # at the start, said once; every other stream, the video's place and the chapters are what a
    # remux of the source without it holds.
    job = jobs()
    source = data_source(tmp_path / "source", job, kind)
    own = streams(source)
    [gone] = [place for place, stream in enumerate(own) if stream["codec_type"] == "data"]
    assert [s["codec_type"] for s in own].count("data") == 1
    work = workdir(tmp_path)
    with caplog.at_level(logging.INFO, logger="seedvr2x.media.assembly"):
        assert assembly.preflight(source, work) == ""
        output = assembled(tmp_path, job, source)
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warned) == 1
    assert f"stream {gone} (data" in warned[0] and "left out of the finished file" in warned[0]
    # The final line counts the other streams written, and says which it left out.
    [done] = [r.getMessage() for r in caplog.records if ": assembled, " in r.getMessage()]
    assert f"with its audio (its data streams {gone} left out), and " in done
    written = streams(output)
    kept = [stream["codec_type"] for stream in own if stream["codec_type"] != "data"]
    assert [stream["codec_type"] for stream in written] == kept == ["video", "audio"]
    remux = remuxed(source, "-map", f"-0:{gone}")
    assert [s["codec_type"] for s in streams(remux)] == kept
    assert shown(output) == [int(p) for p in grid(job.frames, job.frame_rate, Fraction(0))]
    assert shown(output) == shown(remux)
    assert packets(output)[1] == packets(remux)[1]
    assert chapters(output) == chapters(remux) == chapters(source)
    assert len(chapters(output)) == (3 if kind == "chapters" else 0)
    assert written[1]["tags"] == streams(remux)[1]["tags"]
    assert written[1]["disposition"] == streams(remux)[1]["disposition"]
    # The assembly rehearses and compares against the same remux: nothing left in the work
    # directory, and the question asked of the finished file (no data stream) is a clean one.
    assert assembly.preflight(output, work) == ""
    assert assembly.unmuxable(output) == []


def test_a_data_stream_beside_one_refused_is_not_named(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # A mov_text subtitle and a timecode track: refused for the subtitle alone; the data stream
    # isn't warned of (the job stops), and the remux the refusal advises leaves it out.
    (tmp_path / "subs.srt").write_text(SUBTITLES)
    source = tmp_path / "a.mov"
    loose(
        *("-f", "lavfi", "-i", f"color=c=gray:s={WIDTH}x{HEIGHT}:r=25:d=2", *sine(440, 2)),
        *("-i", str(tmp_path / "subs.srt"), "-map", "0:v", "-map", "1:a", "-map", "2:s"),
        *("-c:v", "mpeg4", "-c:a", "aac", "-c:s", "mov_text", "-timecode", "01:00:00:00"),
        *("-t", "2", str(source)),
    )
    work = workdir(tmp_path)
    with caplog.at_level(logging.WARNING, logger="seedvr2x.media.assembly"):
        said = assembly.preflight(source, work)
    assert said.startswith(f"{source}: Matroska can't hold one of its streams as it is: stream 2 (")
    assert "mov_text" in said and "(data" not in said.partition("ffmpeg -i")[0]
    assert "-map -0:3" in said and "-c:2 srt" in said and not caplog.records


@pytest.mark.parametrize("kind", ["cover in MP4", "cover in Matroska"])
def test_a_cover_is_copied(tmp_path: Path, jobs: Callable[..., Job], kind: str) -> None:
    # A cover, and what ffmpeg says of it opening the source, which fails nothing: copied, by
    # its data, as ffmpeg's remux copies it, a video track of one frame after the upscale.
    source = made(tmp_path / "source", kind)
    job = make_job(tmp_path / "job", Fraction(25), 5)
    output = assembled(tmp_path, job, source)
    written, own = streams(output), streams(source)
    assert [s["codec_type"] for s in written] == ["video", "audio", "video"]
    assert (written[2]["codec_name"], own[2]["codec_name"]) == ("png", "png")
    assert own[2]["disposition"] == ["attached_pic"]
    [cover], [theirs] = packets(output)[2], packets(source)[2]
    assert cover[3:] == theirs[3:]
    assert first_video(output) == 0 and numbers(output) == list(range(job.frames))
    # As an attachment, not a video track: ffprobe lists an attached picture as a video stream
    # with the disposition attached_pic, where the remux's track has none (DESIGN.md, Output,
    # Assembly, decided on 2026-10-28). One video track, the upscaled one.
    assert (
        written[2]["disposition"] == ["attached_pic"]
        and "attached_pic" not in own[0]["disposition"]
    )
    assert written[2]["tags"] == {"filename": "cover.png", "mimetype": "image/png"}


def picture_bytes(path: Path, stream: int) -> bytes:
    """The bytes of the picture that is stream `stream` of path: its one packet, whole."""
    result = subprocess.run(
        [
            *("ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(path)),
            *("-map", f"0:{stream}", "-c", "copy", "-f", "data", "-"),
        ],
        capture_output=True,
        check=True,
    )
    return result.stdout


def covered(directory: Path, job: Job) -> tuple[Path, bytes]:
    """A Matroska source for the job as MakeMKV writes one: its video the job's master, FLAC
    audio, a cover.jpg attached BEFORE a font, the source's order of attachments. With the
    cover's bytes."""
    directory.mkdir(parents=True, exist_ok=True)
    cover, font = directory / "cover.jpg", directory / "font.ttf"
    loose("-f", "lavfi", "-i", "testsrc2=s=64x64", "-frames:v", "1", str(cover))
    font.write_bytes(random.Random(9).randbytes(2000))
    source = directory / "covered.mkv"
    seconds = float(job.frames / job.frame_rate)
    loose(
        *("-i", str(job.master), *sine(440, seconds), "-map", "0:v", "-map", "1:a"),
        *("-c:v", "copy", "-c:a", "flac"),
        *("-attach", str(cover), "-metadata:s:t:0", "mimetype=image/jpeg"),
        *("-metadata:s:t:0", "filename=cover.jpg", "-attach", str(font)),
        *("-metadata:s:t:1", "mimetype=application/x-truetype-font"),
        *("-metadata:s:t:1", "filename=font.ttf", str(source)),
    )
    return source, cover.read_bytes()


def test_a_matroska_cover_is_attached_not_a_second_video(
    tmp_path: Path, jobs: Callable[..., Job]
) -> None:
    # MakeMKV adds a cover.jpg to every file it writes. ffmpeg's remux makes it a one-frame
    # video track (V_MJPEG, no disposition), which players list as a second video: the
    # finished file attaches it as it came, with the source's name, type and bytes, after the
    # source's own attachment, the font (DESIGN.md, Output, Assembly, decided on 2026-10-28).
    job = jobs()
    source, jpeg = covered(tmp_path / "source", job)
    own = streams(source)
    assert [s["codec_name"] for s in own] == ["ffv1", "flac", "mjpeg", "ttf"]
    output = assembled(tmp_path, job, source)
    written = streams(output)
    tracks = [s for s in written if s["codec_type"] == "video" and not s["disposition"]]
    assert len(tracks) == 1 and tracks[0]["index"] == 0
    assert [s["codec_name"] for s in written] == ["ffv1", "flac", "ttf", "mjpeg"]
    assert written[3]["disposition"] == ["attached_pic"]
    assert written[3]["tags"] == {"filename": "cover.jpg", "mimetype": "image/jpeg"}
    assert picture_bytes(output, 3) == jpeg == picture_bytes(source, 2)
    # The font as before: its name, type and content.
    assert written[2]["tags"] == {"filename": "font.ttf", "mimetype": "application/x-truetype-font"}
    kept, wrote = assembly.layout(source).streams, assembly.layout(output).streams
    assert wrote[2].extradata == kept[3].extradata
    assert numbers(output) == list(range(job.frames))


def test_an_mp4_cover_is_attached_under_a_name_of_ours(
    tmp_path: Path, jobs: Callable[..., Job]
) -> None:
    # An MP4's cover (a covr atom) has no file name or type: PROVISIONAL, cover.jpg and
    # image/jpeg for a JPEG (impl-v010, 2026-10-28: DESIGN.md names none).
    job = jobs()
    directory = tmp_path / "source"
    directory.mkdir()
    cover = directory / "cover.jpg"
    loose("-f", "lavfi", "-i", "testsrc2=s=64x64", "-frames:v", "1", str(cover))
    source = directory / "a.mp4"
    loose(
        *("-f", "lavfi", "-i", f"color=c=gray:s={WIDTH}x{HEIGHT}:r=25:d=2", *sine(440, 2)),
        *("-i", str(cover), "-map", "0:v", "-map", "1:a", "-map", "2:v", "-c:v:0", "mpeg4"),
        *("-c:a", "aac", "-c:v:1", "copy", "-disposition:v:1", "attached_pic", str(source)),
    )
    own = streams(source)
    assert own[2]["disposition"] == ["attached_pic"] and "filename" not in own[2]["tags"]
    output = assembled(tmp_path, job, source)
    written = streams(output)
    assert [s["codec_name"] for s in written] == ["ffv1", "aac", "mjpeg"]
    assert written[2]["disposition"] == ["attached_pic"]
    assert "attached_pic" not in written[0]["disposition"]
    assert written[2]["tags"] == {"filename": "cover.jpg", "mimetype": "image/jpeg"}
    assert picture_bytes(output, 2) == cover.read_bytes() == picture_bytes(source, 2)


def stream_of(name: str, index: int, codec: str, tags: dict[str, str] | None = None) -> Any:
    """A cover picture's stream as ffprobe declares it, or a video's (`name` "video")."""
    flags = () if name == "video" else ("attached_pic",)
    return assembly.Stream(index, "video", codec, tags or {}, flags, "")


def test_covers_are_named_in_order_and_others_refused() -> None:
    # Pictures that give a name keep it; the others are cover.jpg, cover_2.png: PROVISIONAL.
    found = assembly.Layout(
        (
            stream_of("video", 0, "h264"),
            stream_of("cover", 1, "mjpeg"),
            stream_of("cover", 2, "png"),
            stream_of("cover", 3, "mjpeg", {"filename": "front.jpg", "mimetype": "image/jpeg"}),
        ),
        (),
        None,
    )
    named = [(s.index, s.tags, s.kind, s.picture) for s in assembly._pictured(found).streams[1:]]  # pyright: ignore[reportPrivateUsage]
    assert named == [
        (1, {"filename": "cover.jpg", "mimetype": "image/jpeg"}, "attachment", True),
        (2, {"filename": "cover_2.png", "mimetype": "image/png"}, "attachment", True),
        (3, {"filename": "front.jpg", "mimetype": "image/jpeg"}, "attachment", True),
    ]
    # A codec seedvr2x names no type for, in a source that names none: refused.
    odd = assembly.Layout((stream_of("video", 0, "h264"), stream_of("cover", 1, "bmp")), (), None)
    refused = assembly._unnamed(odd.streams[1])  # pyright: ignore[reportPrivateUsage]
    assert refused is not None and "a cover picture of codec bmp" in refused.reason
    with pytest.raises(MediaError, match="a cover picture of codec bmp"):
        assembly._pictured(odd)  # pyright: ignore[reportPrivateUsage]


def test_an_mp4_cover_of_a_type_we_cant_name_is_refused(tmp_path: Path) -> None:
    # A BMP in an MP4's covr atom: no file name or type in the source, none of ours for the
    # codec (PROVISIONAL): refused before any work, with the remux that leaves it out.
    cover = tmp_path / "cover.bmp"
    loose("-f", "lavfi", "-i", "testsrc2=s=64x64", "-frames:v", "1", str(cover))
    source = tmp_path / "a.mp4"
    loose(
        *("-f", "lavfi", "-i", f"color=c=gray:s={WIDTH}x{HEIGHT}:r=25:d=1", "-i", str(cover)),
        *("-map", "0", "-map", "1", "-c:v:0", "mpeg4", "-c:v:1", "copy"),
        *("-disposition:v:1", "attached_pic", str(source)),
    )
    found = assembly.unmuxable(source)
    assert [(s.index, s.codec) for s in found] == [(1, "bmp")]
    said = assembly.preflight(source, workdir(tmp_path))
    assert "a cover picture of codec bmp with no file name or type of its own" in said
    assert "ffmpeg -i SOURCE -map 0 -c copy -map -0:1 remuxed.mkv\n(stream 1 left out)" in said


def test_the_first_video_is_no_cover() -> None:
    # A source whose first video stream is flagged attached_pic is still the video seedvr2x
    # reads: only the others are covers.
    found = assembly.Layout((stream_of("cover", 0, "mjpeg"),), (), None)
    assert assembly._pictured(found).streams == found.streams  # pyright: ignore[reportPrivateUsage]


def unrated(directory: Path) -> Path:
    """A Matroska source for which ffmpeg says of its cover picture that it can't estimate its
    rate: ffmpeg probes further only while some stream lacks its codec parameters, here a second
    MP3 track whose first packet comes at 0.65 s, past the first 5 MB of video, where the probe
    gives up at its probesize (5 MB), the cover's rate not estimated. The real case is a Blu-ray
    rip's two PGS tracks with no packet in the window (DESIGN.md, Output, Assembly, "Met on
    2026-10-23"). About 7 MB, 0.7 s of noise. (A late MP3 track alone would be refused by the
    rehearsal for moving the others: assembly.moved.)"""
    directory.mkdir(parents=True, exist_ok=True)
    cover = directory / "cover.jpg"
    loose("-f", "lavfi", "-i", "testsrc2=s=64x64", "-frames:v", "1", str(cover))
    source = directory / "unrated.mkv"
    loose(
        *("-f", "lavfi", "-i", "testsrc2=s=640x480:r=25:d=0.7,noise=alls=100:allf=t+u"),
        *("-f", "lavfi", "-i", "sine=d=0.7"),
        *("-itsoffset", "0.65", "-f", "lavfi", "-i", "sine=f=880:d=0.05"),
        *("-map", "0", "-map", "1", "-map", "2", "-c:v", "mpeg4", "-q:v", "2"),
        *("-c:a", "libmp3lame"),
        *("-attach", str(cover), "-metadata:s:t:0", "mimetype=image/jpeg"),
        *("-metadata:s:t:0", "filename=cover.jpg", str(source)),
    )
    return source


NO_RATE = re.compile(r"\[in#0\S* @ 0x[0-9a-f]+\] \[warning\] Stream #(\d+): not enough frames")


def test_a_covers_unestimated_rate_is_tolerated_and_no_other_streams(
    tmp_path: Path, jobs: Callable[..., Job]
) -> None:
    # ffmpeg's note that a stream of one frame has no rate to estimate (DESIGN.md, Output,
    # Assembly, "Met on 2026-10-23", decided on 2026-10-28): tolerated for the source's cover
    # picture, the stream the message numbers, and for no other stream.
    source = unrated(tmp_path / "source")
    plain = subprocess.run(
        [
            *("ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "level+warning", "-y"),
            *("-i", str(source), "-map", "0", "-c", "copy", "-f", "matroska", os.devnull),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    # The fixture proves itself: ffmpeg does print it, for the cover.
    noted = [NO_RATE.search(line) for line in plain.stderr.splitlines()]
    [cover] = [int(found.group(1)) for found in noted if found]
    kinds = {s.index: s for s in assembly.layout(source).streams}
    assert kinds[cover].kind == "video" and kinds[cover].dispositions == ("attached_pic",)
    assert size_of(source) > 5_000_000
    # The preflight, which rehearses the very remux the assembly makes, accepts it...
    work = workdir(tmp_path)
    assert assembly.preflight(source, work) == ""
    assert not list(work.iterdir())
    # ... and the assembly too, the cover attached.
    job = jobs()
    output = tmp_path / "out.mkv"
    assemble(output, job.segments, job.frame_rate, source, indexed(source), work)
    written = streams(output)
    assert [s["codec_name"] for s in written] == ["ffv1", "mp3", "mp3", "mjpeg"]
    assert written[3]["disposition"] == ["attached_pic"]
    assert picture_bytes(output, 3) == (tmp_path / "source" / "cover.jpg").read_bytes()
    # The same note for another stream, or from another input than the source's, stays a
    # complaint, as does anything else.
    line = "[in#0/matroska,webm @ 0x5fd716b70f40] [warning] Stream #{}: not enough frames to"
    line += " estimate rate; consider increasing probesize"
    complaints = assembly._complaints  # pyright: ignore[reportPrivateUsage]
    assert complaints([line.format(cover)], {cover}) == []
    assert complaints([line.format(cover)], set()) == [line.format(cover)]
    assert complaints([line.format(cover + 1)], {cover}) == [line.format(cover + 1)]
    assert complaints([line.format(cover).replace("in#0", "in#1")], {cover}) != []
    assert complaints([line.format(cover).replace("in#0", "in#2")], {cover}) == []
    assert complaints([line.format(cover) + " now"], {cover}) != []


def size_of(path: Path) -> int:
    return path.stat().st_size


COVER_DAMAGE = {
    "the picture's bytes": (
        lambda data, jpeg: data.replace(jpeg, jpeg[:-40] + b"\x00" + jpeg[-39:]),
        "by its bytes",
    ),
    "its file name": (
        lambda data, jpeg: data.replace(b"cover.jpg", b"cover.jpx"),
        "the filename 'cover.jpx', where",
    ),
}


@pytest.mark.parametrize("damage", list(COVER_DAMAGE))
def test_a_cover_that_differs_is_never_put_in_place(
    tmp_path: Path, jobs: Callable[..., Job], monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    # The file the join wrote, its attached picture changed before the check: refused naming
    # the stream, the segments kept (they are the caller's), the output as it was.
    job = jobs()
    source, jpeg = covered(tmp_path / "source", job)
    output = tmp_path / "out.mkv"
    output.write_bytes(b"the file before")
    change, said = COVER_DAMAGE[damage]
    strict = assembly._strict  # pyright: ignore[reportPrivateUsage]

    def then_damaged(command: list[str], doing: str, *pictures: Any) -> None:
        strict(command, doing, *pictures)
        if "joining" in doing:
            path = partial_path(output)
            data = path.read_bytes()
            assert jpeg in data
            path.write_bytes(change(data, jpeg))

    monkeypatch.setattr(assembly, "_strict", then_damaged)
    work = workdir(tmp_path)
    with pytest.raises(
        MediaError, match=rf"stream 3 \(the source's stream \d, attachment, mjpeg\).*{said}"
    ):
        assemble(output, job.segments, job.frame_rate, source, indexed(source), work)
    assert output.read_bytes() == b"the file before"
    assert not partial_path(output).exists() and not list(work.iterdir())
    assert all(path.exists() for path, _ in job.segments)


def earlier(path: Path, stream: int, packet: int, by: int) -> None:
    """Put a packet of a stream of the Matroska file at path `by` ms earlier: ffmpeg itself
    writes no timestamp that goes back. ffprobe's position of a packet is its block's content:
    its track's number, one byte under 127, then its time after its cluster's, 16 bits, signed."""
    found = probed(path, "packet=stream_index,pos")["packets"]
    place = int([p for p in found if p["stream_index"] == stream][packet]["pos"])
    with path.open("r+b") as file:
        file.seek(place)
        head = file.read(3)
        assert head[0] == 0x80 | (stream + 1), head
        moved = int.from_bytes(head[1:], "big", signed=True) - by
        file.seek(place + 1)
        file.write(moved.to_bytes(2, "big", signed=True))


def test_a_remux_ffmpeg_warns_of_is_said_before_any_work(
    tmp_path: Path, jobs: Callable[..., Job]
) -> None:
    # A subtitle that comes before the one before it: ffmpeg's remux warns and writes another
    # time, so the assembly, strict, would refuse at the end. Asked at the start, it is said
    # with the way out, ffmpeg's own remux, which the same question then passes.
    job = jobs()
    source = rich(tmp_path / "source", job)
    earlier(source, 3, 2, 5000)
    assert [int(p[0]) for p in packets(source)[3]] == [1000, 4000, 3000]
    work = workdir(tmp_path)
    assert assembly.unmuxable(source) == []
    said = assembly.preflight(source, work)
    assert said.startswith(
        f"{source}: remuxing its other streams into Matroska with ffmpeg: it reported: "
    )
    assert "Non-monotonic DTS; previous: 4000, current: 3000; changing to 4000" in said
    assert (
        "ffmpeg -i SOURCE -map 0 -c copy remuxed.mkv" in said and "or give -o a directory" in said
    )
    assert not list(work.iterdir())
    with pytest.raises(MediaError, match="Non-monotonic DTS"):
        assemble(tmp_path / "out.mkv", job.segments, job.frame_rate, source, indexed(source), work)
    assert not (tmp_path / "out.mkv").exists() and not list(work.iterdir())
    remux = tmp_path / "remuxed.mkv"
    loose("-i", str(source), "-map", "0", "-c", "copy", str(remux))
    assert assembly.preflight(remux, work) == ""


def early_subtitle(directory: Path, job: Job) -> Path:
    """A Matroska source for the job whose video and audio, FLAC and AAC, start at 5.000 s and
    whose SubRip stream has a line at 1.000 s: ffprobe reads the file's start time 5.000 s, the
    subtitle stream's 1.000 s. The AAC track's first packet is 21 ms before the start ffprobe
    declares for it, as its encoder's priming has it; a file is attached, whose start ffprobe
    declares the file's."""
    directory.mkdir(parents=True, exist_ok=True)
    lines = "1\n00:00:01,000 --> 00:00:02,000\nearly\n\n2\n00:00:05,500 --> 00:00:05,900\nlate\n"
    (directory / "early.srt").write_text(lines)
    source = directory / "early.mkv"
    seconds = float(job.frames / job.frame_rate)
    loose(
        *("-itsoffset", "5", "-i", str(job.master), "-itsoffset", "5", *sine(440, seconds)),
        *("-itsoffset", "5", *sine(880, seconds), "-i", str(directory / "early.srt")),
        *("-map", "0:v", "-map", "1:a", "-map", "2:a", "-map", "3:s"),
        *("-c:v", "copy", "-c:a:0", "flac", "-c:a:1", "aac", "-c:s", "srt"),
        *("-attach", str(directory / "early.srt"), "-metadata:s:t:0", "mimetype=text/plain"),
        str(source),
    )
    return source


def test_a_stream_before_the_files_start_is_said_before_any_work(
    tmp_path: Path, jobs: Callable[..., Job]
) -> None:
    # A subtitle 4 s before the file's start time: ffmpeg's muxer moves every stream 4 s later,
    # Matroska's timestamps starting at 0, and the assembly, which counts the video's frames
    # from the file's start, refuses its own result at the end. Asked at the start, it is said,
    # naming the stream and the move, with the way out that holds: a remux without that
    # stream, which the same question then passes, where a plain remux is moved in its turn.
    job = jobs()
    source = early_subtitle(tmp_path / "source", job)
    found = assembly.layout(source)
    assert found.start_time == 5_000_000
    assert [stream.start for stream in found.streams] == [5, 5, 5, 1, 5]
    assert int(packets(source)[2][0][0]) == 5000 - 21
    work = workdir(tmp_path)
    said = assembly.preflight(source, work)
    assert said.startswith(
        f"{source}: its stream 3 (subtitle, subrip) starts at 1.000 s, 4.000 s before the file's"
        " start time, 5.000 s: "
    )
    assert "ffmpeg's muxer moves every stream 4.000 s later" in said
    assert "ffmpeg -i SOURCE -map 0 -map -0:3 -c copy without.mkv" in said
    assert "or give -o a directory" in said and not list(work.iterdir())
    # What it spares: the join's own refusal, every segment made.
    with pytest.raises(MediaError, match="is at 4000 ms, where the frame grid has it at 0 ms"):
        assemble(tmp_path / "out.mkv", job.segments, job.frame_rate, source, indexed(source), work)
    plain, without = tmp_path / "plain.mkv", tmp_path / "without.mkv"
    loose("-i", str(source), "-map", "0", "-c", "copy", str(plain))
    assert "its stream 3 (subtitle, subrip) starts at 0.000 s, 4.000 s before" in (
        assembly.preflight(plain, work)
    )
    loose("-i", str(source), "-map", "0", "-map", "-0:3", "-c", "copy", str(without))
    assert assembly.preflight(without, work) == ""
    output = tmp_path / "out.mkv"
    assemble(output, job.segments, job.frame_rate, without, indexed(without), work)
    assert shown(output) == shown(job.master)


def test_one_stream_off_its_place_is_no_move(tmp_path: Path, jobs: Callable[..., Job]) -> None:
    # A move is every stream's: one alone later than its own start says, the others at their
    # places, is left to the assembly's own check, and a stream without a declared start is no
    # witness. Starts as ffprobe declares them, not first packets: the rich source's AAC track
    # starts at 0 by both files' records, its first packet 21 ms before.
    job = jobs()
    source = rich(tmp_path / "source", job)
    found = assembly.layout(source)
    _, _, starts = assembly._reference(source, found, workdir(tmp_path))  # pyright: ignore[reportPrivateUsage]
    assert starts == {1: 0, 2: 0, 3: 1, 4: 1}
    assert int(packets(source)[2][0][0]) == -21
    assert assembly.moved(source, found, starts) == ""
    quarter = Fraction(1, 4)
    assert assembly.moved(source, found, {**starts, 1: quarter}) == ""
    assert assembly.moved(source, found, {1: quarter}).startswith(
        f"{source}: remuxed into Matroska, each of its other streams is 0.250 s later than"
    )
    everywhere = {index: start + quarter for index, start in starts.items()}
    assert "each of its other streams is 0.250 s later" in assembly.moved(source, found, everywhere)


# --- whole or absent


def flip(path: Path, stream: int, packet: int) -> None:
    """Change one byte in the middle of a packet of a stream of path."""
    found = probed(path, "packet=stream_index,pos,size")["packets"]
    chosen = [p for p in found if p["stream_index"] == stream][packet]
    # The block's header comes first: a few bytes, where the packets here hold hundreds.
    place = int(chosen["pos"]) + 16 + int(chosen["size"]) // 2
    with path.open("r+b") as file:
        file.seek(place)
        byte = file.read(1)
        file.seek(place)
        file.write(bytes([byte[0] ^ 0x55]))


def dropped(path: Path) -> None:
    """Remux path without its video's last packet."""
    kept = len(packets(path)[0]) - 1
    short = path.with_name("short.mkv")
    loose("-i", str(path), "-map", "0", "-c", "copy", "-frames:v:0", str(kept), str(short))
    short.replace(path)


def unchaptered(path: Path) -> None:
    """Remux path without its chapters."""
    bare = path.with_name("bare.mkv")
    loose("-i", str(path), "-map", "0", "-c", "copy", "-map_chapters", "-1", str(bare))
    bare.replace(path)


def chapter_moved(path: Path) -> None:
    """Remux path with its second chapter half a second longer, the three still there."""
    changed = path.with_name("moved.txt")
    changed.write_text(METADATA.replace("START=1000\nEND=6000", "START=1000\nEND=6500"))
    moved = path.with_name("moved.mkv")
    loose(
        *("-i", str(path), "-i", str(changed), "-map", "0", "-map_chapters", "1"),
        *("-map_metadata", "0", "-c", "copy", str(moved)),
    )
    changed.unlink()
    moved.replace(path)


def attachment_flipped(path: Path) -> None:
    """Change one byte of the font attached to path (rich's), in place."""
    data = bytearray(path.read_bytes())
    place = data.index(random.Random(7).randbytes(3000)) + 1500
    data[place] ^= 0x55
    path.write_bytes(data)


def swapped(path: Path) -> None:
    """Remux path (rich's streams) with its AAC track and its SubRip one in each other's place:
    as many streams, a subtitle where the source has audio."""
    order = [word for index in (0, 1, 3, 2, 4, 5) for word in ("-map", f"0:{index}")]
    other = path.with_name("swapped.mkv")
    loose("-i", str(path), *order, "-c", "copy", str(other))
    other.replace(path)


def private(data: bytes) -> bytes:
    """The codec headers of the first track of a Matroska file whose first bytes are data: its
    CodecPrivate element (ID 0x63A2), after its size, an EBML number whose first byte's leading
    1 gives its length."""
    at = data.index(b"\x63\xa2") + 2
    width = 9 - data[at].bit_length()
    size = int.from_bytes(bytes([data[at] & (0xFF >> width)]) + data[at + 1 : at + width], "big")
    return data[at + width : at + width + size]


def reheaded(path: Path) -> None:
    """Give path's video the codec headers of another FFV1 stream, valid ones, in place: those of
    the same frames in 24 slices where the segments have 16, as long (201 bytes for gbrp16le at
    this size with ffmpeg n9.0.2, whatever the slices), so that every size around them stands
    and no packet changes; and more slices, not fewer: under headers of 4, ffprobe refuses the
    frames as it reads the file ("slice count 16 is invalid (max=4)"), which the read of its
    packets fails on before the headers are compared."""
    other = path.with_name("slices.mkv")
    with FFV1Writer(other, "gbrp16le", WIDTH, HEIGHT, Fraction(25), TAGS, slices=24) as writer:
        writer.write(pictures(0, 1))
    data = bytearray(path.read_bytes())
    ours, theirs = private(bytes(data[:65536])), private(other.read_bytes())
    other.unlink()
    assert len(ours) == len(theirs) and ours != theirs
    place = data.index(ours)
    data[place : place + len(ours)] = theirs
    path.write_bytes(data)


DAMAGE: dict[str, tuple[Callable[[Path], None], str]] = {
    "its chapters missing": (unchaptered, "0 chapters, where the source's are 3"),
    "a chapter's end moved": (
        chapter_moved,
        r"its chapter 1 runs from 1000000000 to 6500000000 ns, where the source's, less its"
        " start time, runs from 1000000000 to 6000000000 ns",
    ),
    "an attachment's byte": (
        attachment_flipped,
        r"stream 5 \(the source's stream 5, attachment, .*\): not the source's attachment, by"
        " its content's hash",
    ),
    "two streams in each other's place": (
        swapped,
        r"stream 2 \(the source's stream 2, audio, aac\): subtitle in the file",
    ),
    "other codec headers for the video": (
        reheaded,
        r"its video stream is ffv1 with codec headers murmur3:[0-9a-f]{32}, where the segments"
        " are ffv1 with murmur3:[0-9a-f]{32}",
    ),
    "a video packet missing": (dropped, r"\d+ video packets in the file, where the segments hold"),
    "a video packet's data": (
        lambda path: flip(path, 0, 40),
        r"its video packet 40 isn't the one read from the segments, packet \d+ of .*seg_0000\d\d"
        r"\.mkv: \d+ bytes, hash [0-9a-f]{32}, where the segment's is \d+ bytes, hash [0-9a-f]{32}",
    ),
    "another stream's packet": (
        lambda path: flip(path, 1, 7),
        r"stream 1 \(the source's stream 1, audio, flac\): its packet 7 isn't the one a remux of"
        " the source holds",
    ),
}


@pytest.mark.parametrize("damage", list(DAMAGE))
def test_a_damaged_result_is_never_put_in_place(
    tmp_path: Path, jobs: Callable[..., Job], monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    # The file the join wrote, damaged before its check: refused, naming the stream and the
    # packet, the partial file removed, what the output held before left as it was.
    job = jobs()
    source = rich(tmp_path / "source", job)
    output = tmp_path / "out.mkv"
    output.write_bytes(b"the file before")
    change, said = DAMAGE[damage]
    strict = assembly._strict  # pyright: ignore[reportPrivateUsage]

    def then_damaged(command: list[str], doing: str, *pictures: Any) -> None:
        strict(command, doing, *pictures)
        if "joining" in doing:
            change(partial_path(output))

    monkeypatch.setattr(assembly, "_strict", then_damaged)
    work = workdir(tmp_path)
    with pytest.raises(MediaError, match=said):
        assemble(output, job.segments, job.frame_rate, source, indexed(source), work)
    assert output.read_bytes() == b"the file before"
    assert not partial_path(output).exists() and not list(work.iterdir())


def test_a_join_that_reads_fewer_packets_is_refused(
    tmp_path: Path, jobs: Callable[..., Job], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A segment the list leaves out: as many packets are missing from what the join read.
    job = jobs()
    listed = assembly.concat_list
    monkeypatch.setattr(assembly, "concat_list", lambda segments, r, _: listed(segments[:-1], r))
    with pytest.raises(
        MediaError, match=r"ffmpeg read \d+ video packets from the segments, which hold"
    ):
        assembled(tmp_path, job, job.master)
    assert not (tmp_path / "out.mkv").exists()


def test_a_misplaced_segment_is_refused(
    tmp_path: Path, jobs: Callable[..., Job], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A segment said one frame longer than it is: the next one's frames are a frame late, off
    # their places of the grid.
    job = jobs()
    listed = assembly.concat_list
    longer = [*job.segments[:9], (job.segments[9][0], job.segments[9][1] + 1), *job.segments[10:]]
    monkeypatch.setattr(assembly, "concat_list", lambda segments, r, _: listed(longer, r))
    with pytest.raises(
        MediaError, match=r"frame 0 of .*seg_000010\.mkv, is at \d+ ms, where the frame grid"
    ):
        assembled(tmp_path, job, job.master)


def test_the_stale_file_goes_as_the_output_is_replaced(
    tmp_path: Path, jobs: Callable[..., Job], monkeypatch: pytest.MonkeyPatch
) -> None:
    job = jobs()
    output, stale, work = tmp_path / "out.mkv", tmp_path / "out.mkv.crc32", workdir(tmp_path)
    stale.write_text("old\n")
    monkeypatch.setattr(assembly, "snap", lambda *_: "setts=ts=TS")
    with pytest.raises(MediaError):
        assemble(output, job.segments, job.frame_rate, job.master, indexed(job.master), work, stale)
    assert stale.read_text() == "old\n"
    monkeypatch.undo()
    assemble(output, job.segments, job.frame_rate, job.master, indexed(job.master), work, stale)
    assert output.exists() and not stale.exists()


def test_the_stale_file_stays_while_the_new_file_is_synced(
    tmp_path: Path, jobs: Callable[..., Job], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The old file's checksums go only once the new file's data is on the disk, just before the
    # rename: a stop during that sync, which lasts as long as the file is large, leaves the old
    # file with its own (DESIGN.md, Output, Checksums).
    from seedvr2x.media import files

    job = jobs()
    output, stale, work = tmp_path / "out.mkv", tmp_path / "out.mkv.crc32", workdir(tmp_path)
    output.write_bytes(b"the file before")
    stale.write_text("old\n")
    sync = files._sync  # pyright: ignore[reportPrivateUsage]
    seen: list[bool] = []

    def stopped(path: Path) -> None:
        if path == partial_path(output):
            seen.append(stale.exists())
            raise KeyboardInterrupt
        sync(path)

    monkeypatch.setattr(files, "_sync", stopped)
    with pytest.raises(KeyboardInterrupt):
        assemble(output, job.segments, job.frame_rate, job.master, indexed(job.master), work, stale)
    assert seen == [True] and stale.read_text() == "old\n"
    assert output.read_bytes() == b"the file before" and not partial_path(output).exists()
    monkeypatch.undo()
    assemble(output, job.segments, job.frame_rate, job.master, indexed(job.master), work, stale)
    assert shown(output) == shown(job.master) and not stale.exists()


def test_a_partial_file_left_goes_before_the_room_is_read(
    tmp_path: Path, jobs: Callable[..., Job], monkeypatch: pytest.MonkeyPatch
) -> None:
    # What a join killed there left, up to the job's size, isn't counted against the room the
    # next one needs: removed first.
    job = jobs()
    output, work = tmp_path / "out.mkv", workdir(tmp_path)
    partial_path(output).write_bytes(b"left by a join that was killed")
    room = assembly._room  # pyright: ignore[reportPrivateUsage]
    there: list[bool] = []

    def looked(*arguments: Any) -> None:
        there.append(partial_path(output).exists())
        room(*arguments)

    monkeypatch.setattr(assembly, "_room", looked)
    assemble(output, job.segments, job.frame_rate, job.master, indexed(job.master), work)
    assert there == [False, False] and shown(output) == shown(job.master)


def test_names_ffmpeg_would_take_for_a_protocol(
    tmp_path: Path, jobs: Callable[..., Job], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Relative names with a colon, which ffmpeg takes for a protocol's ("Film: 2.mkv": Protocol
    # not found): every path the assembly gives ffmpeg and ffprobe is absolute, the output's,
    # the work directory's, the source's and the segments'.
    job = jobs()
    there = tmp_path / "A film: 2"
    there.mkdir()
    for path, _ in job.segments:
        (there / path.name).write_bytes(path.read_bytes())
    source = rich(tmp_path / "source", job)
    (there / "The source: 1.mkv").write_bytes(source.read_bytes())
    monkeypatch.chdir(there)
    source, output, work = Path("The source: 1.mkv"), Path("Film: 2.mkv"), Path("Film: 2.mkv.work")
    work.mkdir()
    segments = [(Path(path.name), frames) for path, frames in job.segments]
    assert assembly.preflight(source, work) == ""
    assert assembly.unmuxable(source) == [] and assembly.rehearse(source, work) == ""
    assert assembly.segment_refusal(segments[0][0], segments[0][1], job.frame_rate) == ""
    assert assembly.signature(segments[0][0]) == assembly.signature(segments[1][0])
    done = assemble(output, segments, job.frame_rate, source, indexed(there / source), work)
    assert done.frames == job.frames and shown(there / output) == [
        23 + pts for pts in shown(job.master)
    ]
    assert not list(work.iterdir())


def test_a_name_the_locale_cant_hold(tmp_path: Path) -> None:
    # A directory whose name isn't valid in the locale's encoding, as a file system holds any
    # bytes: the list of the segments is written as the file system's bytes, no
    # UnicodeEncodeError, and the job is assembled.
    there = tmp_path / os.fsdecode(b"caf\xe9")
    job = make_job(there / "segments", RATES["25"], 4)
    work = workdir(there)
    done = assemble(
        there / "out.mkv", job.segments, job.frame_rate, job.master, indexed(job.master), work
    )
    assert done.frames == job.frames and shown(there / "out.mkv") == shown(job.master)


def test_refused_without_room(
    tmp_path: Path, jobs: Callable[..., Job], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The file takes the segments' size again, and the other streams': refused before anything
    # is written, saying how much is missing, where, and that the same command assembles.
    job = jobs()
    source = rich(tmp_path / "source", job)
    video = sum(path.stat().st_size for path, _ in job.segments)
    statvfs = os.statvfs
    free = {"bytes": video - 4096}

    def little(path: Any) -> Any:
        real = statvfs(path)
        if Path(path) != tmp_path:
            return real
        return os.statvfs_result((*real[:4], free["bytes"] // real.f_frsize, *real[5:]))

    monkeypatch.setattr(assembly.os, "statvfs", little)
    work = workdir(tmp_path)
    refusal = (
        rf"{tmp_path}/out\.mkv: 0\.00 GiB more free space needed in {tmp_path} to assemble it:"
        rf" .* its {len(job.segments)} segments' .* the segments are kept, and the same command"
        " assembles them once there is room"
    )
    with pytest.raises(MediaError, match=refusal):
        assemble(tmp_path / "out.mkv", job.segments, job.frame_rate, source, indexed(source), work)
    assert not list(work.iterdir()) and not partial_path(tmp_path / "out.mkv").exists()
    # Room for the segments, not for the other streams with them.
    free["bytes"] = video + 4096
    with pytest.raises(MediaError, match="more free space needed"):
        assemble(tmp_path / "out.mkv", job.segments, job.frame_rate, source, indexed(source), work)
    # The file is no larger than what was asked for.
    monkeypatch.undo()
    output = assembled(tmp_path, job, source)
    others = remuxed(source, "-map", "-0:v:0").stat().st_size
    assert output.stat().st_size <= video + others


@pytest.fixture
def alarm() -> Iterator[Callable[[Callable[[], None]], None]]:
    """Run something in this thread 0.3 s from now, as a signal's handler runs."""
    before = signal.getsignal(signal.SIGALRM)

    def arm(action: Callable[[], None]) -> None:
        signal.signal(signal.SIGALRM, lambda *_: action())
        signal.setitimer(signal.ITIMER_REAL, 0.3)

    yield arm
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, before)


def interrupt() -> None:
    raise KeyboardInterrupt


def terminate() -> None:
    os.kill(os.getpid(), signal.SIGTERM)


@pytest.mark.parametrize(
    ("action", "raised"), [(interrupt, KeyboardInterrupt), (terminate, stop.Terminated)]
)
def test_a_stop_kills_the_join_and_removes_the_partial(
    tmp_path: Path,
    jobs: Callable[..., Job],
    monkeypatch: pytest.MonkeyPatch,
    alarm: Callable[[Callable[[], None]], None],
    action: Callable[[], None],
    raised: type[BaseException],
) -> None:
    # A second Ctrl-C, or SIGTERM, while the join runs, here a process that would run a minute
    # in the join's place, its partial file begun: the process is killed, the partial and the
    # work files removed, the file before left.
    job = jobs()
    output, work = tmp_path / "out.mkv", workdir(tmp_path)
    output.write_bytes(b"the file before")
    index = indexed(job.master)
    started: list[Any] = []
    popen = subprocess.Popen

    def slow(command: list[str], **options: Any) -> Any:
        if LIST in " ".join(command):
            partial_path(output).write_bytes(b"begun")
            assert (work / LIST).exists() and options["process_group"] == 0
            command = ["sleep", "60"]
            alarm(action)
        started.append(popen(command, **options))
        return started[-1]

    monkeypatch.setattr(assembly.subprocess, "Popen", slow)
    with stop.terminable(), pytest.raises(raised):
        assemble(output, job.segments, job.frame_rate, job.master, index, work)
    monkeypatch.undo()
    assert started and started[-1].args == ["sleep", "60"]
    assert started[-1].returncode == -signal.SIGKILL
    assert output.read_bytes() == b"the file before"
    assert not partial_path(output).exists() and not list(work.iterdir())
    # What a killed run may leave in the work directory, by name.
    assert set(WORK_FILES) == {
        *("assembly.ffconcat", "assembly.framehash", "assembly.others.mkv"),
        *("assembly.pair.mkv", "assembly.pair.ffconcat"),
    }
    assert all(
        map(assembly.work_file, [*WORK_FILES, "assembly.attachment.5", "assembly.picture.3"])
    )
    assert not any(
        map(
            assembly.work_file,
            ["seg_000000.mkv", "assembly.attachment.x", "assembly.picture.x", "assembly.picture."],
        )
    )


def test_no_segment_is_an_error(tmp_path: Path, jobs: Callable[..., Job]) -> None:
    job = jobs()
    index = indexed(job.master)
    with pytest.raises(ValueError, match="needs segments"):
        assemble(tmp_path / "out.mkv", [], job.frame_rate, job.master, index, tmp_path)
    with pytest.raises(ValueError, match="needs segments"):
        assemble(
            tmp_path / "out.mkv",
            [(job.segments[0][0], 0)],
            job.frame_rate,
            job.master,
            index,
            tmp_path,
        )


def kind_stream(kind: str, picture: bool = False) -> assembly.Stream:
    return assembly.Stream(0, kind, "x", {}, (), "", picture=picture)


def test_the_other_streams_are_named_by_kind() -> None:
    # The final lines of an assembly counted "2 other streams" for an audio and a cover picture
    # together (2026-10-29's list from a real job): named by kind. Every kind one stream: "its
    # audio and its cover picture"; else each counted, a cover picture alone "a cover picture".
    said = assembly.others_said
    assert said([]) == "no other stream"
    assert said([kind_stream("audio")]) == "its audio"
    assert said([kind_stream("audio"), kind_stream("attachment", True)]) == (
        "its audio and its cover picture"
    )
    assert said(
        [kind_stream("audio"), kind_stream("subtitle"), kind_stream("attachment", True)]
    ) == ("its audio, its subtitles and its cover picture")
    assert (
        said(
            [
                *(kind_stream("audio"), kind_stream("audio"), kind_stream("subtitle")),
                kind_stream("attachment", True),
            ]
        )
        == "2 audio streams, 1 subtitle stream and a cover picture"
    )
    assert said([kind_stream("attachment"), kind_stream("attachment")]) == "2 attachments"
    assert said([kind_stream("attachment"), kind_stream("attachment", True)]) == (
        "its attachment and its cover picture"
    )
    assert said([kind_stream("attachment", True)] * 2 + [kind_stream("audio")]) == (
        "1 audio stream and 2 cover pictures"
    )


def test_sizes_are_said_in_mib_under_a_tenth_of_a_gib() -> None:
    # "reading its 0.0 GiB" and "checking the bytes of 1 segment, 0.04 GiB" said nothing
    # (2026-10-29's list): MiB under 0.1 GiB, KiB under 1 MiB.
    size = files.size_said
    assert size(16_479_335_080) == "15.3 GiB"
    assert size(16_479_335_080, 2) == "15.35 GiB"
    assert size(int(0.04 * 2**30), 2) == "41 MiB"
    assert size(int(0.1 * 2**30) + 1, 2) == "0.10 GiB"
    assert size(3 * 2**20) == "3 MiB"
    assert size(12 * 2**10) == "12 KiB"
    assert size(0) == "0 KiB"


def test_the_line_before_the_rehearsal_says_what_it_reads_and_for_about_how_long(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # DESIGN.md, Output, Assembly, decided on 2026-10-28: "a line says what the rehearsal reads
    # before it does". The duration is the source's size over a timed read of a sample of it
    # (assembly.reading_said): "under a second" under one (decided on 2026-10-29), at least a few
    # seconds under ten, at least about seconds to the nearest five up to ninety, minutes after (a
    # floor: the remux writes too, and a sample read from the cache is fast; review, 2026-10-28);
    # a read that fails is said with its reason, no duration invented.
    source = tmp_path / "in.mkv"
    source.write_bytes(bytes(3 * 2**20))
    head = (
        f"{source}: the source's other streams (audio, subtitles, a cover) copied once ahead, so"
        f" that one the finished file can't take is found now, not after the night's work;"
        f" reading"
    )
    assert assembly.reading_said(source) == f"{head} its 3 MiB, under a second"
    seconds = iter([0.4, 9.9, 10.0, 66.0, 89.0, 372.0, 5400.0])
    monkeypatch.setattr(assembly, "_read_seconds", lambda *given: next(seconds))
    said = [assembly.reading_said(source).removeprefix(f"{head} its 3 MiB, ") for _ in range(7)]
    assert said == [
        "under a second",
        *("at least a few seconds", "at least about 10 s", "at least about 65 s"),
        *("at least about 90 s", "at least about 6 min", "at least about 90 min"),
    ]

    def unreadable(*given: object) -> float:
        raise OSError("Input/output error")

    monkeypatch.setattr(assembly, "_read_seconds", unreadable)
    assert assembly.reading_said(source).endswith(
        "its 3 MiB, for a time not estimated (Input/output error)"
    )


def test_the_estimate_reads_a_sample_of_the_middle_and_scales_to_the_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The sample is read from the middle of the source, SAMPLE bytes at most; the time it took is
    # scaled to the whole size. A clock that counts 2 s for the sample of a file 10 times larger
    # gives 20 s; the whole of a smaller file is its own sample.
    source = tmp_path / "in.bin"
    size = 10 * assembly.SAMPLE
    with source.open("wb") as handle:
        handle.truncate(size)
    ticks = iter([100.0, 102.0])
    monkeypatch.setattr(assembly.time, "monotonic", lambda: next(ticks))
    assert assembly._read_seconds(source, size) == pytest.approx(20.0)  # pyright: ignore[reportPrivateUsage]
    small = tmp_path / "small.bin"
    small.write_bytes(bytes(1000))
    ticks = iter([5.0, 6.5])
    monkeypatch.setattr(assembly.time, "monotonic", lambda: next(ticks))
    assert assembly._read_seconds(small, 1000) == pytest.approx(1.5)  # pyright: ignore[reportPrivateUsage]

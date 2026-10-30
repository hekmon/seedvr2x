"""A .mkv path's job on the CPU, the model replaced by tests/test_cli_run.py's stand-in: run as a
directory job in its work directory, then assembled into the finished file with the source's other
streams, the work directory removed once the file is whole (DESIGN.md, Output, Assembly;
Validation milestones, 6). The join itself is tests/test_assembly.py's; here, what the command
line does with it: when it assembles, what a stop or a failure keeps, what it refuses, and the
finished file held to milestone 6's acceptance against the source."""

import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import test_stop
from test_cli_run import (
    CUT,
    HOLED_SEGMENTS,
    JOB,
    MISSING_SAID,
    SEEKING,
    SPLIT,
    ScriptedDetector,
    Steps,
    checksums,
    encoded,
    first_passes,
    holed_source,
    indexes,
    job,
    needs_x264,
    rate_of,
    refused,
    repeating,
    shots_of,
    stand_in_model,
    stored,
    upscale,
)
from test_rate import mechanism7

from seedvr2x import cli
from seedvr2x.media import ffmpeg
from seedvr2x.media.ffmpeg import MediaError
from seedvr2x.runtime import resume
from seedvr2x.runtime.manifest import NAME


def _usable() -> bool:
    try:
        ffmpeg.check(assembly=True)
    except MediaError:
        return False
    return True


needs_ffmpeg = pytest.mark.skipif(
    not _usable(), reason="needs ffmpeg 7.1 or later with zscale, idet, ffv1, concat and setts"
)


@pytest.fixture
def steps(monkeypatch: pytest.MonkeyPatch) -> Steps:
    """tests/test_cli_run.py's stand-in model, its steps recorded (its own `steps`)."""
    from seedvr2x.runtime import run

    stand_in_model(monkeypatch.setattr)
    recorded = Steps()
    monkeypatch.setattr(run, "encode_shot", recorded.encode)
    monkeypatch.setattr(run, "sample_windows", recorded.windows)
    monkeypatch.setattr(run, "decode_shot", recorded.decode)
    return recorded


def film(path: Path, rate: str = "24000/1001", frames: int = 96, late: int = 23) -> Path:
    """A source with other streams, as a film's file has them: Matroska, 64x48 FFV1 at `rate`,
    its first frame `late` ms after the file's start; FLAC in English, the default track, and AAC
    in French, both from 0; SRT subtitles; two chapters; a font attached, with its type; a
    title."""
    scratch = path.parent
    (scratch / "subs.srt").write_text(
        "1\n00:00:00,500 --> 00:00:01,500\nfirst line\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\nsecond line\n"
    )
    (scratch / "meta.txt").write_text(
        ";FFMETADATA1\ntitle=The test film\n"
        "[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=1500\ntitle=One\n"
        "[CHAPTER]\nTIMEBASE=1/1000\nSTART=1500\nEND=4000\ntitle=Two\n"
    )
    (scratch / "font.ttf").write_bytes(bytes(range(256)) * 12)
    video = scratch / "video.mkv"
    seconds = str(float(Fraction(frames) / Fraction(rate)))
    sines = [
        argument
        for pitch in (440, 880)
        for argument in ("-f", "lavfi", "-t", seconds, "-i", f"sine=f={pitch}:r=48000")
    ]
    quiet = ("ffmpeg", "-v", "error", "-y")
    subprocess.run(
        [
            *(*quiet, "-f", "lavfi", "-i", f"testsrc2=s=64x48:r={rate}"),
            *("-frames:v", str(frames), "-c:v", "ffv1", "-pix_fmt", "yuv420p", str(video)),
        ],
        check=True,
    )
    subprocess.run(
        [
            *(*quiet, "-itsoffset", str(late / 1000), "-i", str(video), *sines),
            *("-i", str(scratch / "subs.srt"), "-i", str(scratch / "meta.txt")),
            *("-map", "0:v", "-map", "1:a", "-map", "2:a", "-map", "3:s"),
            *("-map_metadata", "4", "-map_chapters", "4", "-c:v", "copy"),
            *("-c:a:0", "flac", "-c:a:1", "aac", "-b:a:1", "64k", "-c:s", "srt"),
            *("-metadata:s:a:0", "language=eng", "-metadata:s:a:1", "language=fra"),
            *("-metadata:s:s:0", "language=eng", "-disposition:a:0", "default"),
            *("-attach", str(scratch / "font.ttf")),
            *("-metadata:s:t", "mimetype=application/x-truetype-font", str(path)),
        ],
        check=True,
    )
    return path


def probed(path: Path, *entries: str) -> dict[str, Any]:
    """ffprobe's records of path, as JSON, for the entries asked."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", *entries, "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def packets(path: Path, stream: int, fields: str = "pts,dts,duration,size,flags") -> list[Any]:
    """The packets of stream `stream` of path, in file order: the fields asked and the data's
    hash."""
    select = ("-select_streams", str(stream), "-show_data_hash", "md5")
    found = probed(path, *select, "-show_entries", f"packet={fields},data_hash")["packets"]
    return [
        tuple(packet.get(field) for field in (*fields.split(","), "data_hash")) for packet in found
    ]


def others(path: Path) -> list[dict[str, Any]]:
    """Every stream of path but its video, as ffprobe records it: its kind and codec, its time
    base and first timestamp, its dispositions and its tags, statistics apart (an attachment's
    content by its hash)."""
    entries = "stream=index,codec_type,codec_name,time_base,start_pts,extradata_hash"
    found = probed(
        path, "-show_data_hash", "md5", "-show_entries", f"{entries}:stream_disposition:stream_tags"
    )["streams"]
    return [stream for stream in found if stream["codec_type"] != "video"]


def chapters(path: Path) -> list[tuple[str, str, str]]:
    found = probed(path, "-show_chapters")["chapters"]
    return [(c["start_time"], c["end_time"], c["tags"]["title"]) for c in found]


def frames_of(path: Path) -> list[int]:
    """The CRC-32 of each frame of path's video at the stand-in's size, decoded as stored, each
    once (-fps_mode passthrough): ffmpeg's own rate mode fills the 23 ms before a video that
    starts after its file with a copy of its first frame, 97 frames read of the film's 96."""
    return stored(["-i", str(path), "-map", "0:v:0", "-fps_mode", "passthrough"], "gbrp16le")


def video_times(path: Path) -> list[int]:
    """The video's timestamps in presentation order, in its time base's units."""
    return sorted(int(pts) for (pts, _hash) in packets(path, 0, "pts"))


# Shots of 10, 20, 20, 27 and 19 frames; half a second is 12 frames at 24000/1001, so the first
# shot merges into the second: segments of 30, 20, 27 and 19 frames.
CUTS = "10\n30\n50\n77\n"
SEGMENTS = [30, 20, 27, 19]
FILM = ("--cuts", "cuts.txt", "--min-segment", "0.5")


def film_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cuts.txt").write_text(CUTS)
    return film(tmp_path / "film.mkv")


def held_to_the_source(final: Path, source: Path, tmp_path: Path, frames: int = 96) -> None:
    """Milestone 6's acceptance (DESIGN.md, Validation milestones): the finished file's video
    timestamps are the source's, frame for frame, and every other stream is copied, as a plain
    ffmpeg remux of the source holds it."""
    # Both files are Matroska, their timestamps milliseconds, and the source's first frame is at
    # a whole millisecond: frame k is the nearest millisecond to that time plus k / R in both, so
    # the bound asserted is none. A source in another time base would differ by its rounding to
    # the millisecond, half of one at most (tests/test_assembly.py holds those).
    assert video_times(final) == video_times(source)
    assert len(video_times(final)) == frames
    reference = tmp_path / "remux.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0",
            "-c",
            "copy",
            str(reference),
        ],
        check=True,
    )
    assert others(final) == others(reference)
    kinds = [(s["codec_type"], s["codec_name"], s["tags"].get("language")) for s in others(final)]
    assert kinds == [
        ("audio", "flac", "eng"),
        ("audio", "aac", "fra"),
        ("subtitle", "subrip", "eng"),
        ("attachment", "ttf", None),
    ]
    attached = others(final)[3]
    assert attached["tags"]["filename"] == "font.ttf"
    assert attached["tags"]["mimetype"] == "application/x-truetype-font"
    assert others(final)[0]["disposition"]["default"] == 1
    for stream in (1, 2, 3):
        assert packets(final, stream) == packets(reference, stream), stream
        # Against the source itself, durations apart: a remux reads the last FLAC packet's as 10
        # ms where the source's block says 11 (ffmpeg n9.0.2, 2026-10-10), the one difference.
        fields = "pts,dts,size,flags"
        assert packets(final, stream, fields) == packets(source, stream, fields), stream
    assert (
        chapters(final)
        == chapters(source)
        == [
            ("0.000000", "1.500000", "One"),
            ("1.500000", "4.000000", "Two"),
        ]
    )
    title = "-show_entries", "format_tags=title"
    assert probed(final, *title) == probed(source, *title)
    assert probed(final, *title)["format"]["tags"]["title"] == "The test film"


@needs_ffmpeg
def test_assembled_with_the_sources_other_streams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # -o x.mkv, end to end: the job run in x.mkv.work, four segments, then assembled: the file
    # there with its checksums, the work directory gone; milestone 6's acceptance against the
    # source; its frames those a directory run of the same job wrote, which is left as it is,
    # never assembled.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source = film_job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source, "x.mkv", *FILM) == 0
    final = tmp_path / "x.mkv"
    listed = sorted(path.name for path in tmp_path.iterdir() if path.name.startswith("x.mkv"))
    assert listed == ["x.mkv", "x.mkv.crc32", "x.mkv.manifest.json"]
    # The other streams named by kind (2026-10-29): two audio, a subtitle, the font attached.
    assert (
        f"wrote {final}: 96 frames from 4 segments, with 2 audio streams, 1 subtitle stream and"
        " 1 attachment, and 2"
    ) in caplog.text
    assert f"{tmp_path / 'x.mkv.work'}: removed, its job assembled" in caplog.text
    held_to_the_source(final, source, tmp_path)
    # The same job to a directory: its segments, never assembled, hold the same frames, and the
    # file's checksums are theirs, one after the other.
    assert upscale(tmp_path, source, "segments", *FILM) == 0
    out = tmp_path / "segments"
    names = [f"seg_{index:06d}" for index in range(4)]
    assert sorted(path.name for path in out.iterdir()) == sorted(
        ["checksums", "frame_index.bin", "manifest.json", *(f"{name}.mkv" for name in names)]
    )
    assert not (tmp_path / "segments.mkv").exists()
    frames = [stored(["-i", str(out / f"{name}.mkv")], "gbrp16le") for name in names]
    assert [len(segment) for segment in frames] == SEGMENTS
    # Each frame says where it comes from, the stand-in's way.
    assert [i for name in names for i in indexes(out / f"{name}.mkv")] == list(range(96))
    joined = [checksum for segment in frames for checksum in segment]
    assert frames_of(final) == joined
    assert checksums(tmp_path / "x.mkv.crc32") == joined
    assert joined == [c for name in names for c in checksums(out / "checksums" / f"{name}.crc32")]
    # verify checks the file's video among its other streams, and names a frame not as written:
    # a byte of the 41st frame's data flipped, in place.
    assert cli.main(["verify", str(final)]) == 0
    listed_packets = probed(final, "-select_streams", "0", "-show_entries", "packet=pos,size")
    packet = listed_packets["packets"][40]
    data = bytearray(final.read_bytes())
    data[int(packet["pos"]) + int(packet["size"]) // 2] ^= 1
    final.write_bytes(data)
    caplog.clear()
    assert cli.main(["verify", str(final)]) == 1
    assert f"{final}: " in caplog.text and "not as written" in caplog.text


def job_content(segments: int = 2, shots: tuple[int, ...] = (0, 4)) -> dict[str, Any]:
    """What resume.clear reads of a manifest's content: the segments' names, the shots' first
    frames, the output's format."""
    return {
        "segments": [{"name": f"seg_{index:06d}.mkv"} for index in range(segments)],
        "shots": [{"start": start} for start in shots],
        "output": {"format": "yuv420p10le"},
    }


def work_directory(tmp_path: Path) -> Path:
    """A work directory holding every name a job of job_content's can leave there."""
    work = tmp_path / "x.mkv.work"
    for name in ("checksums", "resume/shot_000000", "resume/shot_000004"):
        (work / name).mkdir(parents=True)
    names = [NAME, f"{NAME}.partial", "frame_index.bin", "frame_index.bin.partial"]
    names += ["seg_000000.mkv", "seg_000001.mkv", "seg_000001.mkv.partial"]
    # A segment command's partial file, a tiled pass's scratch, and what an assembly killed
    # there left (files.command_partial, tiling.SCRATCH, assembly.work_file).
    names += ["seg_000001.partial.mkv", "assembly.ffconcat", "assembly.attachment.4"]
    names += ["checksums/seg_000000.crc32", "checksums/seg_000001.crc32.partial"]
    names += [f"resume/shot_000000/{name}" for name in ("latent.pt", "window_0000.pt")]
    names += [f"resume/shot_000004/{name}" for name in ("input.mkv", "input.crc32.partial")]
    names += ["resume/shot_000004/scratch.bf16"]
    for name in names:
        (work / name).write_bytes(b"x")
    return work


def test_work_directory_cleared(tmp_path: Path) -> None:
    # This job's names, every one, then the directory itself; names missing are fine.
    work = work_directory(tmp_path)
    assert resume.clear(work, job_content()) == []
    assert not work.exists()
    work = work_directory(tmp_path)
    for name in ("seg_000000.mkv", "frame_index.bin", "checksums/seg_000000.crc32"):
        (work / name).unlink()
    assert resume.clear(work, job_content()) == []
    assert not work.exists()


def test_work_directory_keeps_what_is_not_the_jobs(tmp_path: Path) -> None:
    # Anything else there is left, never deleted, and named; the manifest stays with it, saying
    # whose the directory is, and the job's own files go all the same.
    work = work_directory(tmp_path)
    foreign = ["notes.txt", "seg_000002.mkv", "checksums/mine.crc32", "resume/shot_000004/a.txt"]
    foreign += ["resume/shot_000009/latent.pt", "sub/deep.txt"]
    for name in foreign:
        (work / name).parent.mkdir(parents=True, exist_ok=True)
        (work / name).write_bytes(b"mine")
    # A shot's directory not this job's is named whole, as a directory of anyone's is.
    left = sorted([*foreign[:4], "resume/shot_000009", "sub"])
    assert resume.clear(work, job_content()) == left
    found = sorted(str(path.relative_to(work)) for path in work.rglob("*") if path.is_file())
    assert found == sorted([NAME, *foreign])
    assert all((work / name).read_bytes() == b"mine" for name in foreign)


def test_work_directory_follows_no_link(tmp_path: Path) -> None:
    # A link under one of the job's names isn't the job's file: left, and what it names with it.
    work = work_directory(tmp_path)
    target = tmp_path / "elsewhere.mkv"
    target.write_bytes(b"mine")
    (work / "seg_000000.mkv").unlink()
    (work / "seg_000000.mkv").symlink_to(target)
    kept = tmp_path / "kept"
    kept.mkdir()
    (kept / "seg_000000.crc32").write_bytes(b"mine")
    for name in ("seg_000000.crc32", "seg_000001.crc32.partial"):
        (work / "checksums" / name).unlink(missing_ok=True)
    (work / "checksums").rmdir()
    (work / "checksums").symlink_to(kept, target_is_directory=True)
    assert resume.clear(work, job_content()) == ["checksums", "seg_000000.mkv"]
    assert target.read_bytes() == b"mine" and (kept / "seg_000000.crc32").read_bytes() == b"mine"
    assert (work / NAME).is_file()


def same_file(one: Path, other: Path) -> None:
    """Two assemblies of one job are the same file: every stream's packets, with their
    timestamps, and ffprobe's records of the streams, the chapters and the file."""
    for stream in range(4):
        assert packets(one, stream) == packets(other, stream), stream
    assert others(one) == others(other)
    assert chapters(one) == chapters(other)
    records = ("-show_entries", "format=nb_streams,start_time,duration,size:format_tags")
    assert probed(one, *records) == probed(other, *records)


def listing(directory: Path) -> list[str]:
    return sorted(str(path.relative_to(directory)) for path in directory.rglob("*"))


# What the film's work directory holds once its four segments are finished, before its assembly.
FINISHED = sorted(
    [
        *("checksums", "frame_index.bin", "manifest.json"),
        *(f"checksums/seg_{index:06d}.crc32" for index in range(4)),
        *(f"seg_{index:06d}.mkv" for index in range(4)),
    ]
)


@needs_ffmpeg
def test_stopped_then_resumed_assembles_the_same_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps
) -> None:
    # Stopped at once after its first segment: no file yet, the work directory kept; the same
    # command resumes and assembles the file an uninterrupted run assembles.
    source = film_job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source, "whole.mkv", *FILM) == 0
    uninterrupted = list(steps.calls)
    steps.calls.clear()
    steps.stop = "encode 30"
    assert upscale(tmp_path, source, "x.mkv", *FILM) == 130
    final, work = tmp_path / "x.mkv", tmp_path / "x.mkv.work"
    assert not final.exists() and not (tmp_path / "x.mkv.crc32").exists()
    content = json.loads((work / NAME).read_text())
    assert [segment["finished"] for segment in content["segments"]] == [True, False, False, False]
    assert "assembled" not in content
    steps.stop = None
    steps.calls.clear()
    assert upscale(tmp_path, source, "x.mkv", *FILM) == 0
    assert steps.calls == uninterrupted[uninterrupted.index("encode 30") :]
    assert not work.exists()
    # The same packets, timestamps and records, in a file of the same size; not the same bytes:
    # Matroska's muxer draws its identifiers at random, 104 bytes of this file's 181,044, in its
    # first 4.4 kB (measured on 2026-10-10, ffmpeg n9.0.2).
    same_file(final, tmp_path / "whole.mkv")
    assert final.stat().st_size == (tmp_path / "whole.mkv").stat().st_size
    assert (tmp_path / "x.mkv.crc32").read_bytes() == (tmp_path / "whole.mkv.crc32").read_bytes()
    held_to_the_source(final, source, tmp_path)


def flip_middle_byte(path: Path) -> None:
    """One byte of the file's middle changed, its size the same."""
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))


@needs_ffmpeg
def test_a_damaged_segment_is_made_again_then_the_file_assembled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # DESIGN.md, Output, Assembly (decided on 2026-10-12): each segment's bytes are hashed when
    # it is finished and checked at the assembly. Every segment finished and the assembly refused
    # (no room), one byte of the second segment's middle flipped, its size the same: the same
    # command finds it damaged, assembles nothing (exit 1), names it and its frames, records it
    # unfinished with its file and checksums gone, the others untouched; the next run makes it
    # again from its shots (encode, windows and decode of the shots it holds), then assembles a
    # file with the frames of the uninterrupted run's.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source = film_job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source, "whole.mkv", *FILM) == 0
    steps.calls.clear()
    final, work = tmp_path / "x.mkv", tmp_path / "x.mkv.work"
    with monkeypatch.context() as full:
        full.setattr(os, "statvfs", lambda path: SimpleNamespace(f_bavail=0, f_frsize=4096))
        assert upscale(tmp_path, source, "x.mkv", *FILM) == 1
    recorded = json.loads((work / NAME).read_text())["segments"]
    segment = work / "seg_000001.mkv"
    assert all(re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) for entry in recorded)
    assert len({entry["sha256"] for entry in recorded}) == 4
    assert recorded[1]["sha256"] == hashlib.sha256(segment.read_bytes()).hexdigest()
    size = segment.stat().st_size
    flip_middle_byte(segment)
    assert segment.stat().st_size == size
    steps.calls.clear()
    caplog.clear()
    assert upscale(tmp_path, source, "x.mkv", *FILM) == 1
    assert steps.calls == [] and not final.exists()
    # The hashing reads every master once, silent otherwise: one line says it first (review of
    # the assembly, 2026-10-28), with the bytes alone where no --until measured the disk's rate.
    [checking] = [m for m in caplog.messages if m.startswith("checking the bytes of")]
    assert re.fullmatch(r"checking the bytes of 4 segments, \d+ (K|M)iB", checking)
    assert caplog.messages.index(checking) < next(
        i for i, m in enumerate(caplog.messages) if "damaged, its bytes" in m
    )
    assert f"{segment}: damaged, its bytes are not those recorded when it was finished" in (
        caplog.text
    )
    assert "(frames 30 to 49 of the job); nothing assembled" in caplog.text
    assert "1 segment damaged, none assembled" in caplog.text
    assert "the same command makes it again, its shots encoded, sampled and decoded again" in (
        caplog.text
    )
    assert [r.levelname for r in caplog.records if "segment damaged" in r.getMessage()] == ["ERROR"]
    after = json.loads((work / NAME).read_text())["segments"]
    assert [entry["finished"] for entry in after] == [True, False, True, True]
    assert [(entry["bytes"], entry["sha256"]) for entry in after][1] == (None, None)
    assert [entry for index, entry in enumerate(after) if index != 1] == [
        entry for index, entry in enumerate(recorded) if index != 1
    ]
    assert not segment.exists() and not (work / "checksums" / "seg_000001.crc32").exists()
    assert "resume" not in listing(work)
    # The next run: the segment's shots again, nothing else, then the join.
    caplog.clear()
    assert upscale(tmp_path, source, "x.mkv", *FILM) == 0
    assert "segment damaged" not in caplog.text and not work.exists()
    assert steps.calls == ["encode 30", "window 30:0", "decode 30"]
    same_file(final, tmp_path / "whole.mkv")
    assert (tmp_path / "x.mkv.crc32").read_bytes() == (tmp_path / "whole.mkv.crc32").read_bytes()
    held_to_the_source(final, source, tmp_path)
    assert cli.main(["verify", str(final)]) == 0


@pytest.mark.parametrize("how", ["deleted", "truncated"])
@needs_ffmpeg
def test_a_finished_segment_missing_or_changed_at_a_start_is_made_again_then_the_file_assembled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
    how: str,
) -> None:
    # DESIGN.md, Output, Assembly (decided on 2026-10-29): every segment finished and the assembly
    # refused (no room), one segment then deleted or cut short: the same command, which used to
    # be refused at its start, says it in the assembly's warning, makes the segment's shots again
    # (the job loads its models for them) and assembles a file of the uninterrupted run's frames.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source = film_job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source, "whole.mkv", *FILM) == 0
    steps.calls.clear()
    final, work = tmp_path / "x.mkv", tmp_path / "x.mkv.work"
    with monkeypatch.context() as full:
        full.setattr(os, "statvfs", lambda path: SimpleNamespace(f_bavail=0, f_frsize=4096))
        assert upscale(tmp_path, source, "x.mkv", *FILM) == 1
    segment = work / "seg_000001.mkv"
    recorded = segment.stat().st_size
    if how == "deleted":
        segment.unlink()
        found = "missing"
    else:
        segment.write_bytes(segment.read_bytes()[:-1])
        found = f"{recorded - 1} bytes, where {recorded} were recorded"
    steps.calls.clear()
    caplog.clear()
    assert upscale(tmp_path, source, "x.mkv", *FILM) == 0
    assert [m for m in caplog.messages if "its bytes changed" in m] == [
        f"{segment}: its bytes changed since it was finished (frames 30 to 49 of the job: {found}):"
        " the disk or another program is to suspect; made again now"
    ]
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert steps.calls == ["encode 30", "window 30:0", "decode 30"] and not work.exists()
    same_file(final, tmp_path / "whole.mkv")
    assert (tmp_path / "x.mkv.crc32").read_bytes() == (tmp_path / "whole.mkv.crc32").read_bytes()
    held_to_the_source(final, source, tmp_path)
    assert cli.main(["verify", str(final)]) == 0


def flip_during_the_run(
    monkeypatch: pytest.MonkeyPatch, work: Path, flips: dict[tuple[int, int], int]
) -> list[tuple[int, int]]:
    """Have a byte of a finished segment changed while the run is still going, as a disk or
    another program would: `flips` maps (segment, n) to the segment damaged just after the n-th
    time that segment is recorded finished. Returns the (segment, n) finished so far, in order."""
    from seedvr2x.runtime.units import Units

    finished = Units.segment_finished
    seen: list[tuple[int, int]] = []

    def flipping(self: Units, segment: int) -> None:
        finished(self, segment)
        seen.append((segment, sum(1 for done, _ in seen if done == segment) + 1))
        if seen[-1] in flips:
            flip_middle_byte(work / f"seg_{flips[seen[-1]]:06d}.mkv")

    monkeypatch.setattr(Units, "segment_finished", flipping)
    return seen


@needs_ffmpeg
def test_a_segment_damaged_during_the_run_is_made_again_by_it_then_the_file_assembled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # DESIGN.md, Output, Assembly (decided on 2026-10-28): the run that made the last segment, its
    # models loaded, makes a segment it finds damaged at the assembly again at once, then
    # assembles, and says so in a warning (the disk or another program is to suspect). One byte of
    # the second segment flipped while the fourth is being made: the same run encodes, samples and
    # decodes its shot again, assembles, exit 0, the file of the uninterrupted run's frames.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source = film_job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source, "whole.mkv", *FILM) == 0
    uninterrupted = list(steps.calls)
    steps.calls.clear()
    work = tmp_path / "x.mkv.work"
    seen = flip_during_the_run(monkeypatch, work, {(2, 1): 1})
    caplog.clear()
    assert upscale(tmp_path, source, "x.mkv", *FILM) == 0
    # The segments finished in order, the second again after the fourth.
    assert seen == [(0, 1), (1, 1), (2, 1), (3, 1), (1, 2)]
    assert steps.calls == [*uninterrupted, "encode 30", "window 30:0", "decode 30"]
    segment = work / "seg_000001.mkv"
    [warned] = [r for r in caplog.records if "its bytes changed since it was finished" in r.message]
    assert warned.levelname == "WARNING"
    assert warned.getMessage() == (
        f"{segment}: its bytes changed since it was finished (frames 30 to 49 of the job): the"
        " disk or another program is to suspect; made again now"
    )
    # Said once, and no stop: nothing at error level, no "none assembled".
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert "none assembled" not in caplog.text and not work.exists()
    same_file(tmp_path / "x.mkv", tmp_path / "whole.mkv")
    assert (tmp_path / "x.mkv.crc32").read_bytes() == (tmp_path / "whole.mkv.crc32").read_bytes()
    held_to_the_source(tmp_path / "x.mkv", source, tmp_path)
    assert cli.main(["verify", str(tmp_path / "x.mkv")]) == 0


@needs_ffmpeg
def test_a_segment_damaged_again_after_it_was_made_again_stops_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # Only once: the segment made again by the run is damaged again before the assembly: exit 1,
    # naming it, nothing assembled, the segment recorded unfinished with its file gone, the others
    # kept; the next start (a run with models loaded, the file unchanged) makes it again and
    # assembles.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source = film_job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source, "whole.mkv", *FILM) == 0
    uninterrupted = list(steps.calls)
    steps.calls.clear()
    work = tmp_path / "x.mkv.work"
    flips = {(2, 1): 1, (1, 2): 1}
    with monkeypatch.context() as damaging:
        flip_during_the_run(damaging, work, flips)
        caplog.clear()
        assert upscale(tmp_path, source, "x.mkv", *FILM) == 1
    # Made again once, not twice.
    assert steps.calls == [*uninterrupted, "encode 30", "window 30:0", "decode 30"]
    segment = work / "seg_000001.mkv"
    assert len([m for m in caplog.messages if "made again now" in m]) == 1
    again, summary = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert "the same command makes it again" in summary.getMessage()
    assert again.getMessage().startswith(f"{segment}: damaged again, its bytes changed once more")
    assert "(frames 30 to 49 of the job); nothing assembled" in again.getMessage()
    assert not (tmp_path / "x.mkv").exists() and not segment.exists()
    after = json.loads((work / NAME).read_text())["segments"]
    assert [entry["finished"] for entry in after] == [True, False, True, True]
    steps.calls.clear()
    caplog.clear()
    assert upscale(tmp_path, source, "x.mkv", *FILM) == 0
    assert steps.calls == ["encode 30", "window 30:0", "decode 30"] and not work.exists()
    assert "damaged, its bytes" not in caplog.text and "damaged again" not in caplog.text
    same_file(tmp_path / "x.mkv", tmp_path / "whole.mkv")


@pytest.mark.parametrize("how", ["ctrl-c twice", "ctrl-c once"])
@needs_ffmpeg
def test_a_stop_during_the_segments_made_again_keeps_the_job_for_the_next_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
    how: str,
) -> None:
    # The segments made again are units like any: stopped at once (a second Ctrl-C, here the
    # stand-in's) inside the decode, or between two units (Ctrl-C once, the stop checked before
    # each), the run leaves the job as a stop does, nothing assembled, the shot's windows kept; the
    # next start finishes the segment and assembles, the damage not found again.
    from seedvr2x.runtime.stop import Stop, Stopped

    caplog.set_level(logging.INFO, logger="seedvr2x")
    source = film_job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source, "whole.mkv", *FILM) == 0
    steps.calls.clear()
    work = tmp_path / "x.mkv.work"
    from seedvr2x.runtime.units import Units

    flipped = False
    last: list[int] = []
    finished = Units.segment_finished

    def damaging(self: Units, segment: int) -> None:
        nonlocal flipped
        finished(self, segment)
        if segment == 2 and not flipped:
            flipped = True
            flip_middle_byte(work / "seg_000001.mkv")
            if how == "ctrl-c twice":
                steps.stop = "decode 30"
        if segment == 3:
            last.append(segment)

    def check(self: Stop, cost: float | None = None, unit: str = "the next unit") -> None:
        # Ctrl-C once, asked after the last segment: the assembly's own check passes (the stop
        # asked is held against a unit begun), the first unit made again is stopped before.
        if last and "assembly" not in unit:
            raise Stopped

    with monkeypatch.context() as stopping:
        stopping.setattr(Units, "segment_finished", damaging)
        if how == "ctrl-c once":
            stopping.setattr(Stop, "check", check)
        caplog.clear()
        status = upscale(tmp_path, source, "x.mkv", *FILM)
    assert status == 130 and not (tmp_path / "x.mkv").exists()
    assert len([m for m in caplog.messages if "made again now" in m]) == 1
    assert "none assembled" not in caplog.text
    segments = json.loads((work / NAME).read_text())["segments"]
    assert [entry["finished"] for entry in segments] == [True, False, True, True]
    if how == "ctrl-c twice":
        assert steps.calls[-3:] == ["encode 30", "window 30:0", "decode 30"]
        assert (work / "resume" / "shot_000030" / "window_0000.pt").is_file()
    steps.stop = None
    steps.calls.clear()
    caplog.clear()
    assert upscale(tmp_path, source, "x.mkv", *FILM) == 0
    assert "damaged, its bytes" not in caplog.text and "made again now" not in caplog.text
    assert steps.calls == (
        ["decode 30"] if how == "ctrl-c twice" else ["encode 30", "window 30:0", "decode 30"]
    )
    same_file(tmp_path / "x.mkv", tmp_path / "whole.mkv")
    assert not work.exists()


@needs_ffmpeg
def test_a_ctrl_c_during_the_hash_check_stops_before_the_first_unit_made_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # Review of 2026-10-28: a Ctrl-C once, the real flag, asked while the segments' bytes are read
    # again (the assembly's own Stop gets it) reaches the segments made again: stopped before the
    # first unit, none made, exit 130, the damaged segment recorded unfinished and the job kept;
    # the next start makes the segment and assembles.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source = film_job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source, "whole.mkv", *FILM) == 0
    uninterrupted = list(steps.calls)
    steps.calls.clear()
    work = tmp_path / "x.mkv.work"
    checked = cli._damaged  # pyright: ignore[reportPrivateUsage]
    sent: list[bool] = []

    def interrupted(*arguments: Any) -> Any:
        if not sent:
            sent.append(True)
            os.kill(os.getpid(), signal.SIGINT)
        return checked(*arguments)

    with monkeypatch.context() as stopping:
        flip_during_the_run(stopping, work, {(2, 1): 1})
        stopping.setattr(cli, "_damaged", interrupted)
        caplog.clear()
        assert upscale(tmp_path, source, "x.mkv", *FILM) == 130
    assert steps.calls == uninterrupted and not (tmp_path / "x.mkv").exists()
    [stopped] = [m for m in caplog.messages if m.startswith("stopped after")]
    assert stopped.startswith("stopped after the check of the segments' bytes, as asked; ")
    assert "none assembled" not in caplog.text and "stopped before the assembly" not in caplog.text
    segments = json.loads((work / NAME).read_text())["segments"]
    assert [entry["finished"] for entry in segments] == [True, False, True, True]
    steps.calls.clear()
    caplog.clear()
    assert upscale(tmp_path, source, "x.mkv", *FILM) == 0
    assert steps.calls == ["encode 30", "window 30:0", "decode 30"] and not work.exists()
    same_file(tmp_path / "x.mkv", tmp_path / "whole.mkv")


@needs_ffmpeg
def test_assembly_failed_keeps_everything_then_assembles_without_the_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # Every segment finished, and no room for the file: exit 1, the work directory whole, an old
    # file at the path and its checksums untouched. The same command, once there is room,
    # assembles: no first pass, no model loaded, no unit made.
    import os
    from types import SimpleNamespace

    from seedvr2x.runtime import model

    source = film_job(tmp_path, monkeypatch)
    final, sums, work = (tmp_path / name for name in ("x.mkv", "x.mkv.crc32", "x.mkv.work"))
    with monkeypatch.context() as full:
        full.setattr(os, "statvfs", lambda path: SimpleNamespace(f_bavail=0, f_frsize=4096))
        assert upscale(tmp_path, source, "x.mkv", *FILM) == 1
        # An older file at the path with its checksums, which this job's work directory, its
        # manifest there, doesn't refuse (DESIGN.md, Output, Assembly, decided on 2026-10-12: a
        # file with no manifest at all is refused, at a new job's start): a first run having
        # failed, the second finds them and fails alike. Until the first run, none were there.
        final.write_bytes(b"an older file")
        sums.write_bytes(b"its checksums")
        caplog.clear()
        assert upscale(tmp_path, source, "x.mkv", *FILM) == 1
    assert f"{final}: 0.00 GiB more free space needed in {tmp_path} to assemble it" in caplog.text
    assert "the same command assembles them once there is room; not assembled: every segment" in (
        caplog.text
    )
    assert f"finished, 4 in {work}: the same command assembles {final}" in caplog.text
    # What that run does first, said: its checks of the job, no model.
    assert "once it has checked the job again (its model files, its environment, the" in caplog.text
    assert "source's other streams), no model loaded" in caplog.text
    assert listing(work) == FINISHED
    assert (final.read_bytes(), sums.read_bytes()) == (b"an older file", b"its checksums")
    assert not (tmp_path / "x.mkv.partial").exists()
    passes = first_passes(monkeypatch)
    monkeypatch.setattr(model, "load_models", lambda *a: pytest.fail("the models loaded"))
    steps.calls.clear()
    assert upscale(tmp_path, source, "x.mkv", *FILM) == 0
    assert passes == [] and steps.calls == []
    assert not work.exists()
    held_to_the_source(final, source, tmp_path)
    assert cli.main(["verify", str(final)]) == 0


@needs_ffmpeg
def test_removal_finished_by_the_next_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # A run killed between its file put in place and its work directory removed, the manifest
    # saying assembled: the next run finishes the removal and starts nothing. With the file gone,
    # or of another size, the record is dropped and the job assembled again.
    from seedvr2x.runtime import model

    caplog.set_level(logging.INFO, logger="seedvr2x")
    source = film_job(tmp_path, monkeypatch)
    final, work = tmp_path / "x.mkv", tmp_path / "x.mkv.work"
    kept: list[dict[str, bytes | None]] = []

    def killed(directory: Path, content: dict[str, Any]) -> list[str]:
        """The removal, killed after its first segment went."""
        kept.append({name: None for name in listing(directory)})
        (directory / "seg_000000.mkv").unlink()
        raise KeyboardInterrupt

    with monkeypatch.context() as stopping:
        stopping.setattr(resume, "clear", killed)
        assert upscale(tmp_path, source, "x.mkv", *FILM) == 130
    assert list(kept[0]) == FINISHED
    content = json.loads((work / NAME).read_text())
    assert content["assembled"] == {"name": "x.mkv", "bytes": final.stat().st_size}
    assembled = final.read_bytes()
    passes = first_passes(monkeypatch)

    def unloaded(context: pytest.MonkeyPatch) -> None:
        context.setattr(model, "load_models", lambda *a: pytest.fail("the models loaded"))

    def stopped_at_once(*arguments: object) -> list[str]:
        raise KeyboardInterrupt

    steps.calls.clear()
    caplog.clear()
    with monkeypatch.context() as context:
        unloaded(context)
        assert upscale(tmp_path, source, "x.mkv", *FILM) == 0
    assert f"{final}: assembled already, {len(assembled)} bytes" in caplog.text
    assert f"{work}: removed, its job assembled" in caplog.text
    assert passes == [] and steps.calls == [] and not work.exists()
    assert final.read_bytes() == assembled
    # The record with another file at the path, or none: dropped, and the job assembled again
    # from its segments, the models not loaded. The job is made again first, whole: the file and
    # the manifest beside it removed, as the refusal of a file with no manifest says to do.
    for other in (b"another file", None):
        final.unlink()
        (tmp_path / "x.mkv.manifest.json").unlink()
        with monkeypatch.context() as context:
            context.setattr(resume, "clear", stopped_at_once)
            assert upscale(tmp_path, source, "x.mkv", *FILM) == 130
        assert json.loads((work / NAME).read_text())["assembled"]["name"] == "x.mkv"
        assert listing(work) == FINISHED
        final.unlink()
        if other is not None:
            final.write_bytes(other)
        steps.calls.clear()
        caplog.clear()
        with monkeypatch.context() as context:
            unloaded(context)
            assert upscale(tmp_path, source, "x.mkv", *FILM) == 0
        why = "is missing" if other is None else f"holds {len(other)}"
        assert f"its manifest says, but {final} {why}: assembled again" in caplog.text
        assert steps.calls == [] and not work.exists()
        held_to_the_source(final, source, tmp_path)


@needs_ffmpeg
def test_plan_kept_then_planned_again_then_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # --plan FILE -o x.mkv leaves the job's plan in x.mkv.work, its manifest and frame index, no
    # unit and no file; another --cut-threshold plans it again from its record; the same command
    # without --plan then runs it, with no second first pass, and assembles the file.
    from seedvr2x.runtime import detector

    caplog.set_level(logging.INFO, logger="seedvr2x")
    monkeypatch.setattr(detector, "Detector", ScriptedDetector)
    source = job(tmp_path, monkeypatch)
    passes = first_passes(monkeypatch)
    final, work, plan = (tmp_path / name for name in ("x.mkv", "x.mkv.work", "plan.txt"))
    with monkeypatch.context():
        # The models load, for the time estimates' calibration (DESIGN.md, Memory planner).
        assert upscale(tmp_path, source, "x.mkv", *CUT, "--plan", str(plan)) == 0
        assert listing(work) == ["frame_index.bin", "manifest.json"]
        assert shots_of(work) == [(0, 5), (5, 15), (15, 22), (22, 25)]
        again = (*CUT, "--cut-threshold", "0.5")
        assert upscale(tmp_path, source, "x.mkv", *again, "--plan", str(plan)) == 0
        assert f"{work}: re-planned, its job having made no unit yet" in caplog.text
        assert shots_of(work) == [(0, 5), (5, 15), (15, 25)]
        assert listing(work) == ["frame_index.bin", "manifest.json"]
    assert len(passes) == 1 and steps.calls == [] and not final.exists()
    assert upscale(tmp_path, source, "x.mkv", *again) == 0
    assert len(passes) == 1 and encoded(steps) == [0, 5, 15]
    assert indexes(final) == list(range(25)) and not work.exists()
    assert cli.main(["verify", str(final)]) == 0


def beside_manifest(directory: Path) -> dict[str, Any]:
    return json.loads((directory / "x.mkv.manifest.json").read_text())


def untouched(directory: Path, *names: str) -> dict[str, tuple[bytes, int]]:
    """The bytes and modification time of each file named, to compare after a run that must
    touch none."""
    return {
        name: ((directory / name).read_bytes(), (directory / name).stat().st_mtime_ns)
        for name in names
    }


@needs_ffmpeg
def test_the_finished_files_manifest_stays_beside_it_and_the_same_command_makes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # DESIGN.md, Output, Assembly (decided on 2026-10-12): the job's manifest stays beside the
    # finished file, the record of its assembly in it, whole; the same command again makes
    # nothing: no first pass, no model, no rehearsal, the file, its checksums and its manifest
    # untouched to the byte and the nanosecond.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source = film_job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source, "x.mkv", *FILM) == 0
    final = tmp_path / "x.mkv"
    recorded = beside_manifest(tmp_path)
    assert recorded["assembled"] == {"name": "x.mkv", "bytes": final.stat().st_size}
    assert all(segment["finished"] for segment in recorded["segments"])
    assert recorded["input"]["frames"] == 96
    names = ("x.mkv", "x.mkv.crc32", "x.mkv.manifest.json")
    before = untouched(tmp_path, *names)
    there = listing(tmp_path)
    passes = first_passes(monkeypatch)
    from seedvr2x.media import assembly
    from seedvr2x.runtime import model

    monkeypatch.setattr(model, "load_models", lambda *a: pytest.fail("the models loaded"))
    monkeypatch.setattr(assembly, "preflight", lambda *a: pytest.fail("the rehearsal ran"))
    steps.calls.clear()
    caplog.clear()
    assert upscale(tmp_path, source, "x.mkv", *FILM) == 0
    assert f"{final}: finished already, its 96 frames" in caplog.text
    assert "environment differs" not in caplog.text
    assert passes == [] and steps.calls == []
    assert untouched(tmp_path, *names) == before and listing(tmp_path) == there
    # Under another ffmpeg: said in one line, never refused, the manifest as it was.
    edited = beside_manifest(tmp_path)
    edited["environment"]["ffmpeg"] = "n0.0"
    (tmp_path / "x.mkv.manifest.json").write_text(json.dumps(edited, indent=2) + "\n")
    before = untouched(tmp_path, *names)
    caplog.clear()
    assert upscale(tmp_path, source, "x.mkv", *FILM) == 0
    assert f"{final}: its environment differs from this one and stays its manifest's" in (
        caplog.text
    )
    assert "ffmpeg" in caplog.text and f"{final}: finished already" in caplog.text
    assert untouched(tmp_path, *names) == before and passes == []
    # verify reads the checksums beside the file, as before.
    assert cli.main(["verify", str(final)]) == 0


@needs_ffmpeg
def test_another_job_to_a_finished_file_refused_and_nothing_touched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    source = film_job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source, "x.mkv", *FILM) == 0
    names = ("x.mkv", "x.mkv.crc32", "x.mkv.manifest.json")
    before, there = untouched(tmp_path, *names), listing(tmp_path)
    passes = first_passes(monkeypatch)
    steps.calls.clear()
    # Another seed; another source, the same size of command line.
    text = refused(tmp_path, source, caplog, *FILM, "--seed", "7", output="x.mkv")
    manifest = f"{tmp_path / 'x.mkv.manifest.json'}: another job than the one asked"
    assert manifest in text and "settings.seed: 42 -> 7" in text
    other = film(tmp_path / "other.mkv", rate="25/1")
    there = listing(tmp_path)
    text = refused(tmp_path, other, caplog, *FILM, output="x.mkv")
    assert manifest in text and "input." in text
    assert passes == [] and steps.calls == []
    assert untouched(tmp_path, *names) == before and listing(tmp_path) == there


@needs_ffmpeg
def test_a_file_with_no_manifest_beside_it_refused_before_any_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # DESIGN.md, Output, Assembly: "an x.mkv with no manifest beside it is refused, where today a
    # new job replaces it without a word": a file, then a link to one; neither is touched, no
    # work directory is made, no first pass run.
    source = film_job(tmp_path, monkeypatch)
    passes = first_passes(monkeypatch)
    final = tmp_path / "x.mkv"
    final.write_bytes(b"the user's own file")
    before = untouched(tmp_path, "x.mkv")
    text = refused(tmp_path, source, caplog, *FILM, output="x.mkv")
    assert f"{final}: a file with no x.mkv.manifest.json beside it: not replaced; remove it" in (
        text
    )
    there = listing(tmp_path)  # the models' stand-in files, written by the command's helper
    text = refused(tmp_path, source, caplog, *FILM, output="x.mkv")
    assert untouched(tmp_path, "x.mkv") == before and listing(tmp_path) == there
    final.unlink()
    target = tmp_path / "elsewhere.bin"
    target.write_bytes(b"the user's file, linked")
    final.symlink_to(target)
    there = listing(tmp_path)
    text = refused(tmp_path, source, caplog, *FILM, output="x.mkv")
    assert "a file with no x.mkv.manifest.json beside it: not replaced" in text
    assert final.is_symlink() and final.resolve() == target.resolve()
    assert target.read_bytes() == b"the user's file, linked" and listing(tmp_path) == there
    # A link whose target is missing is a file's name taken all the same.
    target.unlink()
    text = refused(tmp_path, source, caplog, *FILM, output="x.mkv")
    assert "a file with no x.mkv.manifest.json beside it: not replaced" in text
    assert passes == [] and steps.calls == [] and not (tmp_path / "x.mkv.work").exists()


@needs_ffmpeg
def test_a_finished_file_missing_or_changed_is_not_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # The manifest beside the file says its bytes: the file missing or of another size is
    # refused, neither the file nor the record removed; the way out is the record's removal.
    source = film_job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source, "x.mkv", *FILM) == 0
    final, record = tmp_path / "x.mkv", tmp_path / "x.mkv.manifest.json"
    whole = final.read_bytes()
    kept = record.read_bytes()
    passes = first_passes(monkeypatch)
    final.write_bytes(whole[:-1])
    text = refused(tmp_path, source, caplog, *FILM, output="x.mkv")
    assert f"{final} holds {len(whole) - 1} bytes, its manifest x.mkv.manifest.json beside it" in (
        text
    )
    assert f"says {len(whole)}: not replaced; remove x.mkv and x.mkv.manifest.json to make" in text
    assert final.read_bytes() == whole[:-1] and record.read_bytes() == kept
    final.unlink()
    text = refused(tmp_path, source, caplog, *FILM, output="x.mkv")
    assert f"{final} is missing, its manifest x.mkv.manifest.json beside it says" in text
    assert "not replaced; remove x.mkv.manifest.json to make the file" in text  # the record alone
    assert not final.exists() and record.read_bytes() == kept and passes == []
    # The record alone removed, the truncated file in its place, the next run refuses a file
    # with no manifest: the advice with the file in place names both, and followed to the letter
    # the next run makes a new job and the file (review, 2026-10-28).
    final.write_bytes(whole[:-1])
    record.unlink()
    text = refused(tmp_path, source, caplog, *FILM, output="x.mkv")
    assert "a file with no x.mkv.manifest.json beside it: not replaced" in text
    record.write_bytes(kept)
    text = refused(tmp_path, source, caplog, *FILM, output="x.mkv")
    assert "remove x.mkv and x.mkv.manifest.json to make the file" in text
    final.unlink()
    record.unlink()
    assert upscale(tmp_path, source, "x.mkv", *FILM) == 0
    assert final.stat().st_size == len(whole) and record.is_file()


@needs_ffmpeg
def test_plan_on_a_finished_file_writes_nothing_and_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # No frame index is left to draw a cut list from: FILE not written, exit 1, said. FILE naming
    # the manifest beside the file is refused as the file and its checksums are.
    source = film_job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source, "x.mkv", *FILM) == 0
    names = ("x.mkv", "x.mkv.crc32", "x.mkv.manifest.json")
    before, there = untouched(tmp_path, *names), listing(tmp_path)
    plan = tmp_path / "plan.txt"
    passes = first_passes(monkeypatch)
    text = refused(tmp_path, source, caplog, *FILM, "--plan", str(plan), output="x.mkv")
    assert f"--plan {plan}: not written: {tmp_path / 'x.mkv'} is finished already" in text
    assert not plan.exists() and listing(tmp_path) == there and passes == []
    assert untouched(tmp_path, *names) == before
    record = tmp_path / "x.mkv.manifest.json"
    text = refused(tmp_path, source, caplog, *FILM, "--plan", str(record), output="x.mkv")
    assert f"--plan {record}: the output's manifest; it names the cut list's file" in text
    assert untouched(tmp_path, *names) == before


KILLED = ("a", "b", "c", "d", "e")


@needs_ffmpeg
@pytest.mark.parametrize("kill", KILLED)
def test_every_stop_between_the_end_steps_is_finished_by_the_next_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
    kill: str,
) -> None:
    # The end steps (cli._assemble): (a) the file in place, (b) its checksums, (c) the record of
    # the assembly in the work manifest, (d) the manifest beside the file, (e) the work directory
    # removed, its manifest last. A run killed after each, the step that follows made to raise;
    # the same command then ends with the file whole, its checksums, the manifest beside it and
    # no work directory, the file's frames the uninterrupted run's, no model loaded.
    from seedvr2x.media import checksums
    from seedvr2x.runtime import manifest, model

    caplog.set_level(logging.INFO, logger="seedvr2x")
    source = film_job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source, "whole.mkv", *FILM) == 0
    whole = tmp_path / "whole.mkv"
    final, work = tmp_path / "x.mkv", tmp_path / "x.mkv.work"

    def killed(*arguments: object, **options: object) -> None:
        raise KeyboardInterrupt

    def killed_removal(directory: Path, content: dict[str, Any]) -> list[str]:
        # After the work manifest went, and a segment with it.
        (directory / NAME).unlink()
        (directory / "seg_000000.mkv").unlink()
        raise KeyboardInterrupt

    with monkeypatch.context() as stopping:
        if kill == "a":
            # The file's own, the segments' having been written before.
            write = checksums.write_checksums

            def killed_sums(path: Path, sums: Any) -> None:
                if path.name == "x.mkv.crc32":
                    raise KeyboardInterrupt
                write(path, sums)

            stopping.setattr(checksums, "write_checksums", killed_sums)
        elif kill == "b":
            stopping.setattr(manifest.Manifest, "job_assembled", killed)
        elif kill == "c":
            stopping.setattr(cli, "_write_beside", killed)
        elif kill == "d":
            stopping.setattr(cli, "_clear", killed)
        else:
            stopping.setattr(resume, "clear", killed_removal)
        assert upscale(tmp_path, source, "x.mkv", *FILM) == 130
    record = tmp_path / "x.mkv.manifest.json"
    # What the killed run left: the file in place from (a); its checksums from (b); the record of
    # the assembly in the work manifest from (c), the manifest beside the file from (d).
    assert final.is_file()
    assert (tmp_path / "x.mkv.crc32").exists() == (kill != "a")
    assert work.is_dir()
    assert (NAME in listing(work)) == (kill != "e")
    if kill != "e":
        assert ("assembled" in json.loads((work / NAME).read_text())) == (kill in "cd")
    assert record.exists() == (kill in "de")
    passes = first_passes(monkeypatch)
    monkeypatch.setattr(model, "load_models", lambda *a: pytest.fail("the models loaded"))
    steps.calls.clear()
    caplog.clear()
    assert upscale(tmp_path, source, "x.mkv", *FILM) == 0
    text = caplog.text
    # What the second run did: assembled again after (a) and (b); finished the removal after
    # (c) and (d), writing the manifest beside the file first after (c); said the job finished
    # after (e), its work directory's leftovers removed from the manifest beside the file.
    assembled_again = f"wrote {final}: 96 frames" in text
    assert assembled_again == (kill in "ab")
    assert (f"{final}: assembled already" in text) == (kill in "cd")
    assert (f"{record}: written, from the manifest in {work}" in text) == (kill == "c")
    assert (f"{final}: finished already, its 96 frames" in text) == (kill == "e")
    assert f"{work}: removed, its job assembled" in text
    assert passes == [] and steps.calls == []
    assert not work.exists()
    assert sorted(path.name for path in tmp_path.glob("x.mkv*")) == [
        *("x.mkv", "x.mkv.crc32", "x.mkv.manifest.json")
    ]
    assert frames_of(final) == frames_of(whole)
    assert (tmp_path / "x.mkv.crc32").read_bytes() == (tmp_path / "whole.mkv.crc32").read_bytes()
    assert beside_manifest(tmp_path)["assembled"] == {
        "name": "x.mkv",
        "bytes": final.stat().st_size,
    }
    held_to_the_source(final, source, tmp_path)
    assert cli.main(["verify", str(final)]) == 0


@needs_ffmpeg
def test_both_manifests_with_no_record_of_the_assembly_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # PROVISIONAL (impl-v010, 2026-10-28): the manifest beside the file together with a work
    # manifest that doesn't record the assembly: no run makes it; refused, naming both.
    source = film_job(tmp_path, monkeypatch)
    steps.stop = "encode 30"
    assert upscale(tmp_path, source, "x.mkv", *FILM) == 130
    work = tmp_path / "x.mkv.work"
    (tmp_path / "x.mkv.manifest.json").write_text((work / NAME).read_text())
    before = listing(tmp_path)
    steps.stop = None
    text = refused(tmp_path, source, caplog, *FILM, output="x.mkv")
    assert "x.mkv.manifest.json beside" in text and "with no record of the assembly" in text
    assert listing(tmp_path) == before


def mov_text(path: Path) -> Path:
    """An MP4 file with subtitles as MP4 holds them, mov_text, which Matroska's muxer refuses."""
    (path.parent / "text.srt").write_text("1\n00:00:00,100 --> 00:00:00,900\na line\n")
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=64x48:r=25"),
            *("-i", str(path.parent / "text.srt"), "-frames:v", "25", "-c:v", "mpeg4"),
            *("-c:s", "mov_text", str(path)),
        ],
        check=True,
    )
    return path


@needs_ffmpeg
def test_stream_matroska_cannot_take_refused_before_any_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # A source one of whose other streams Matroska can't hold as it is: refused for a .mkv path
    # before the first pass, naming the stream, nothing written and no model loaded; the same
    # source accepted with a directory, whose segments are muxed by the user's own tools.
    from seedvr2x.runtime import model

    monkeypatch.chdir(tmp_path)
    source = mov_text(tmp_path / "in.mp4")
    passes = first_passes(monkeypatch)
    with monkeypatch.context() as context:
        context.setattr(model, "load_models", lambda *a: pytest.fail("the models loaded"))
        text = refused(tmp_path, source, caplog, output="x.mkv")
    assert f"{source}: " in text and "mov_text" in text and "-o" in text
    assert passes == [] and steps.calls == []
    assert not [path.name for path in tmp_path.iterdir() if path.name.startswith("x.mkv")]
    assert upscale(tmp_path, source, "out") == 0
    assert indexes(tmp_path / "out" / "seg_000000.mkv") == list(range(25))


@needs_ffmpeg
def test_a_data_stream_matroska_cannot_take_is_left_out_and_warned_of(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # A MOV with a timecode track, a data stream Matroska can't hold (DESIGN.md, Output,
    # Assembly, decided on 2026-10-12): the job runs, the warning is said once at its start,
    # and the finished file holds the video alone, every frame.
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source = tmp_path / "in.mov"
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=64x48:r=25"),
            *("-frames:v", "25", "-c:v", "mpeg4", "-timecode", "01:00:00:00", str(source)),
        ],
        check=True,
    )
    assert upscale(tmp_path, source, "x.mkv") == 0
    final = tmp_path / "x.mkv"
    warned = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name == "seedvr2x.media.assembly"
    ]
    assert len(warned) == 1 and "left out of the finished file" in warned[0]
    assert warned[0].startswith(f"{source}: Matroska can't hold one of its data streams, which")
    assert indexes(final) == list(range(25))
    assert [s["codec_type"] for s in probed(final, "-show_streams")["streams"]] == ["video"]
    assert "with no other stream (its data streams 1 left out), and 0 chapters" in caplog.text


@needs_ffmpeg
def test_a_cover_picture_is_attached_to_the_finished_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps
) -> None:
    # A Matroska source with a cover.jpg attached, as MakeMKV writes one (DESIGN.md, Output,
    # Assembly, decided on 2026-10-28): the job runs, and the finished file holds the one video,
    # the cover attached under its name and type with its bytes, not a second video track.
    monkeypatch.chdir(tmp_path)
    cover = tmp_path / "cover.jpg"
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=64x64"),
            *("-frames:v", "1", str(cover)),
        ],
        check=True,
    )
    source = tmp_path / "in.mkv"
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=64x48:r=25"),
            *("-frames:v", "25", "-c:v", "mpeg4", "-attach", str(cover)),
            *("-metadata:s:t:0", "mimetype=image/jpeg", "-metadata:s:t:0", "filename=cover.jpg"),
            str(source),
        ],
        check=True,
    )
    assert upscale(tmp_path, source, "x.mkv") == 0
    final = tmp_path / "x.mkv"
    found = probed(
        final,
        "-show_entries",
        "stream=index,codec_type,codec_name:stream_disposition=attached_pic:stream_tags",
    )["streams"]
    assert [(s["codec_type"], s["disposition"]["attached_pic"]) for s in found] == [
        ("video", 0),
        ("video", 1),
    ]
    assert found[1]["tags"] == {"filename": "cover.jpg", "mimetype": "image/jpeg"}
    kept = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(final), "-map", "0:1", "-c", "copy", "-f", "data", "-"],
        capture_output=True,
        check=True,
    )
    assert kept.stdout == cover.read_bytes()
    assert indexes(final) == list(range(25))
    assert not (tmp_path / "x.mkv.work").exists()


@needs_ffmpeg
def test_work_directory_refusals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # What stands where a .mkv path's work directory goes is refused, never deleted: a file, a
    # link, a directory holding anything with no manifest; and --dump-frames inside it.
    source = job(tmp_path, monkeypatch)
    passes = first_passes(monkeypatch)
    final, work = tmp_path / "x.mkv", tmp_path / "x.mkv.work"
    work.write_text("mine")
    text = refused(tmp_path, source, caplog, *JOB, output="x.mkv")
    assert f"{work}: not a directory; {final}'s job keeps its files there" in text
    assert work.read_text() == "mine"
    work.unlink()
    (tmp_path / "elsewhere").mkdir()
    work.symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    text = refused(tmp_path, source, caplog, *JOB, output="x.mkv")
    assert f"{work}: not a directory; {final}'s job keeps its files there" in text
    work.unlink()
    work.mkdir()
    (work / "notes.txt").write_text("mine")
    text = refused(tmp_path, source, caplog, *JOB, output="x.mkv")
    assert f"{work}: not empty, and no manifest.json to resume from; {final}'s job keeps" in text
    assert (work / "notes.txt").read_text() == "mine"
    (work / "notes.txt").unlink()
    dumps = str(work / "dumps")
    text = refused(tmp_path, source, caplog, *JOB, "--dump-frames", dumps, output="x.mkv")
    assert f"--dump-frames {dumps}: in the work directory of {final}, which holds" in text
    assert passes == [] and steps.calls == [] and not final.exists()
    # What an assembly or its rehearsal left when killed is no obstacle, in a new job's
    # directory as in one resumed: its own names (assembly.WORK_FILES), removed at the start.
    from seedvr2x.media.assembly import WORK_FILES

    (work / "assembly.others.mkv").write_bytes(b"left")
    steps.stop = "encode 4"
    assert upscale(tmp_path, source, "x.mkv", *JOB) == 130
    assert listing(work)[:2] == ["checksums", "checksums/seg_000000.crc32"]
    assert WORK_FILES[:3] == ("assembly.ffconcat", "assembly.framehash", "assembly.others.mkv")
    assert WORK_FILES[3:] == ("assembly.pair.mkv", "assembly.pair.ffconcat")
    for name in WORK_FILES:
        (work / name).write_bytes(b"left")
    steps.stop = None
    assert upscale(tmp_path, source, "x.mkv", *JOB) == 0
    assert indexes(final) == list(range(25)) and not work.exists()


@needs_ffmpeg
def test_one_writer_at_a_time_from_the_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # A second seedvr2x to the same file while the first runs: refused by the work directory's
    # lock, before its first pass, on a new job as on one resumed.
    import fcntl
    import os

    source = job(tmp_path, monkeypatch)
    passes = first_passes(monkeypatch)
    work = tmp_path / "x.mkv.work"
    for resumed in (False, True):
        if resumed:
            steps.stop = "encode 4"
            assert upscale(tmp_path, source, "x.mkv", *JOB) == 130
            steps.stop = None
            passes.clear()
        else:
            work.mkdir()
        descriptor = os.open(work, os.O_RDONLY)
        fcntl.flock(descriptor, fcntl.LOCK_EX)  # another seedvr2x's
        try:
            text = refused(tmp_path, source, caplog, *JOB, output="x.mkv")
        finally:
            os.close(descriptor)
        assert f"{work}: another seedvr2x is writing it" in text and passes == []
    assert upscale(tmp_path, source, "x.mkv", *JOB) == 0
    assert indexes(tmp_path / "x.mkv") == list(range(25)) and not work.exists()


@needs_ffmpeg
@pytest.mark.parametrize("rate", ["24000/1001", "25", "30000/1001"])
def test_read_back_as_a_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
    rate: str,
) -> None:
    # A file seedvr2x builds is never refused by seedvr2x (DESIGN.md, Input; milestone 6): 366
    # frames assembled from 122 segments of 3, at each rate, then given back as a source, whose
    # first pass and every check --plan runs: accepted, its frames counted, at its rate, with no
    # warning, each frame within half a millisecond of its place on that rate's timeline.
    from seedvr2x.media.index import FrameIndex

    frames, each = 366, 3
    caplog.set_level(logging.INFO, logger="seedvr2x")
    monkeypatch.chdir(tmp_path)
    source = film(tmp_path / "film.mkv", rate, frames)
    (tmp_path / "cuts.txt").write_text("".join(f"{cut}\n" for cut in range(each, frames, each)))
    assert upscale(tmp_path, source, "x.mkv", "--cuts", "cuts.txt", "--min-segment", "0") == 0
    final = tmp_path / "x.mkv"
    assert f"wrote {final}: {frames} frames from {frames // each} segments" in caplog.text
    assert video_times(final) == video_times(source)
    caplog.clear()
    assert upscale(tmp_path, final, "back", "--cuts", "cuts.txt", "--plan", "plan.txt") == 0
    warned = [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert [record.getMessage() for record in warned] == []
    content = json.loads((tmp_path / "back" / NAME).read_text())
    assert (content["input"]["frames"], content["input"]["frame_rate"]) == (frames, rate)
    index = FrameIndex.read(tmp_path / "back" / "frame_index.bin")
    times = [int(pts) * index.time_base for pts in index.pts]
    off = [abs(time - times[0] - place / Fraction(rate)) for place, time in enumerate(times)]
    assert max(off) <= Fraction(1, 2000)


@needs_ffmpeg
@needs_x264
def test_frame_rate_override_assembled_on_its_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps
) -> None:
    # A source whose timestamps contradict the rate it declares, taken at --frame-rate R
    # (tests/test_rate.py's mechanism 7: declared 24/1, its frames within 10.8 ms of
    # 24000/1001's timeline, its video starting 23 ms after its audio): the finished file is on
    # R's grid from the first frame's time, frame k at the millisecond nearest 23 + k * 1001 /
    # 24, a half going up, as ffmpeg rounds; it declares R; its audio is the source's.
    monkeypatch.chdir(tmp_path)
    source = mechanism7(tmp_path / "s9.mkv", size="64x48", late=23, audio=True)
    (tmp_path / "cuts.txt").write_text("12\n100\n")
    options = ("--cuts", "cuts.txt", "--min-segment", "0", "--frame-rate", "24000/1001")
    assert upscale(tmp_path, source, "x.mkv", *options) == 0
    final = tmp_path / "x.mkv"
    assert rate_of(final) == "24000/1001"
    assert video_times(source)[0] == 23
    assert video_times(final) == [23 + (2002 * place + 24) // 48 for place in range(260)]
    assert video_times(final) != video_times(source)
    reference = tmp_path / "remux.mkv"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(source), "-map", "0", "-c", "copy", str(reference)],
        check=True,
    )
    assert packets(final, 1) == packets(reference, 1)
    assert len(frames_of(final)) == 260


def holed_film(directory: Path) -> tuple[Path, Path]:
    """tests/test_cli_run.py's source with missing frames (holed_source: 40 places at 24000/1001,
    frames 17 and 18 missing) with a second stream, as a film has them, holes.mkv in directory:
    its video's packets copied, their timestamps kept, and FLAC audio from 0 for the 40 places'
    time. Returns it and the file with every frame it was made from."""
    made = directory / "video"
    made.mkdir()
    video, full = holed_source(made, 40, (17, 18))
    path = directory / "holes.mkv"
    seconds = str(float(Fraction(40) / Fraction(24000, 1001)))
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-i", str(video)),
            *("-f", "lavfi", "-t", seconds, "-i", "sine=f=440:r=48000"),
            *("-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "flac", str(path)),
        ],
        check=True,
    )
    assert video_times(path) == video_times(video)
    return path, full


# Each of the 40 places' time in the finished file, in milliseconds: the grid of 24000/1001 from
# the source's first frame, at 0, a half going up, as ffmpeg rounds.
HOLED_TIMES = [(2002 * place + 24) // 48 for place in range(40)]


@needs_ffmpeg
@needs_x264
def test_missing_frames_filled_into_the_finished_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # A source with two frames missing, their places filled (DESIGN.md, Input: "Missing frames
    # are filled, not refused"), to a finished file: the job counts in places, and so does its
    # assembly. The file holds a frame a place, 40 for the 38 the source's video has, each at
    # its place on the rate's grid from the source's first timestamp: the 38 at the source's own
    # times, the hole's two at the grid's, between them; a checksum a place, which verify
    # checks; the audio as a remux of the source holds it, packet for packet, so in step. The
    # hole's places hold the upscale of the frame held there: every frame is the one a file
    # whose frame 16 really comes three times gives. Given back as a source, the file is read
    # with no place to fill: its frames are the places.
    from seedvr2x.media.index import FrameIndex

    caplog.set_level(logging.INFO, logger="seedvr2x")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cuts.txt").write_text("12\n18\n24\n")
    source, full = holed_film(tmp_path)
    options = (*SEEKING, *SPLIT)
    assert upscale(tmp_path, source, "x.mkv", *options) == 0
    final = tmp_path / "x.mkv"
    assert f"{tmp_path / MISSING_SAID}" in caplog.text
    wrote = f"wrote {final}: 40 frames from 4 segments, with its audio, and 0"
    assert wrote in caplog.text
    assert sorted(path.name for path in tmp_path.glob("x.mkv*")) == [
        *("x.mkv", "x.mkv.crc32", "x.mkv.manifest.json")
    ]
    own, times = video_times(source), video_times(final)
    assert len(own) == 38 and own[0] == 0
    assert times == HOLED_TIMES
    assert [time for place, time in enumerate(times) if place not in (17, 18)] == own
    assert own[16:18] == [667, 792] and times[16:20] == [667, 709, 751, 792]
    sums = checksums(tmp_path / "x.mkv.crc32")
    assert len(sums) == 40 and frames_of(final) == sums
    assert cli.main(["verify", str(final)]) == 0
    reference = tmp_path / "remux.mkv"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(source), "-map", "0", "-c", "copy", str(reference)],
        check=True,
    )
    assert others(final) == others(reference) and len(others(final)) == 1
    assert packets(final, 1) == packets(reference, 1)
    # The file whose frame 16 really comes three times, to a directory: the same 40 frames.
    shown = [16 if place in (17, 18) else place for place in range(40)]
    repeated = repeating(full, tmp_path / "repeated.mkv", shown)
    assert upscale(tmp_path, repeated, "repeated", *options) == 0
    segments = [tmp_path / "repeated" / name for name in HOLED_SEGMENTS]
    assert sums == [crc for path in segments for crc in stored(["-i", str(path)], "gbrp16le")]
    assert len(set(sums[16:19])) == 3  # the stand-in's frames differ, each saying its place
    # Read back as a source: 40 frames, none held, nothing said of a missing frame.
    caplog.clear()
    assert upscale(tmp_path, final, "back", "--cuts", "cuts.txt", "--plan", "plan.txt") == 0
    warned = [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert [record.getMessage() for record in warned] == []
    content = json.loads((tmp_path / "back" / NAME).read_text())["input"]
    assert (content["frames"], content["frame_rate"]) == (40, "24000/1001")
    assert "filled" not in content
    index = FrameIndex.read(tmp_path / "back" / "frame_index.bin")
    assert index.frames == 40 and index.held is None


# tests/test_stop.py's run of the stand-in in a process of its own, the join made slow, or the
# rehearsal of the other streams' remux at the job's start, whichever its `slow` names: that
# ffmpeg reads the source at the source's own pace (-re before its first input: 4 s for the
# film), and says when it starts, so that a signal finds it at work.
STOPPING = test_stop.DRIVER.replace(
    "status = cli.main(sys.argv[3:])",
    """
import subprocess
popen = subprocess.Popen

def slow_join(command, *arguments, **options):
    joins = "concat" in command
    rehearses = not joins and str(command[-1]).endswith("assembly.others.mkv")
    if not (joins and slow == "assembly") and not (rehearses and slow == "rehearsal"):
        return popen(command, *arguments, **options)
    first = list(command).index("-i")
    process = popen([*command[:first], "-re", *command[first:]], *arguments, **options)
    print(f"IN {slow}", file=sys.stderr, flush=True)
    return process

subprocess.Popen = slow_join
status = cli.main(sys.argv[3:])""",
)


def stopping(directory: Path, slow: str = "-", signals: tuple[int, ...] = ()) -> tuple[int, str]:
    """Run the film's job to x.mkv in a process of its own (STOPPING), the signals sent to its
    process group, as a terminal sends Ctrl-C, once it says it is in `slow`, a step of the
    stand-in's or the assembly, 0.3 s apart; its status and what it logged."""
    process = subprocess.Popen(
        [
            *(sys.executable, "-c", STOPPING, str(test_stop.TESTS), slow),
            *("film.mkv", "-o", "x.mkv", "--model-dir", ".", "--dit-model", "w.safetensors"),
            *("--vae-model", "w.safetensors", "--resolution", "96", "--seed", "42", *FILM),
            *("--color-correction", "none", "--format", "gbrp16le"),
        ],
        stderr=subprocess.PIPE,
        text=True,
        cwd=directory,
        start_new_session=True,
        env={**os.environ, "HF_HOME": str(directory / "hf"), "HF_HUB_OFFLINE": "1"},
    )
    assert process.stderr is not None
    lines: list[str] = []
    for line in process.stderr:
        lines.append(line)
        if line.startswith(f"IN {slow}"):
            for number in signals:
                os.killpg(process.pid, number)
                time.sleep(0.3)
    return process.wait(timeout=120), "".join(lines)


def joining(directory: Path) -> list[str]:
    """The processes still at work on the job's files in directory, by their command lines
    (Linux's /proc): ffmpeg's, which a stop must not leave behind."""
    found: list[str] = []
    for entry in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = entry.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:  # gone meanwhile
            continue
        if str(directory / "x.mkv") in command and "ffmpeg" in command:
            found.append(command)
    return found


def children_of(parent: int) -> dict[int, str]:
    """The processes whose parent is `parent`, by pid, with their command lines (Linux's /proc)."""
    found: dict[int, str] = {}
    for entry in Path("/proc").glob("[0-9]*/stat"):
        try:
            # After the command's name, in brackets: state, then the parent's pid.
            fields = entry.read_text().rsplit(")", 1)[1].split()
            if int(fields[1]) == parent:
                command = (entry.parent / "cmdline").read_bytes().replace(b"\0", b" ")
                found[int(entry.parent.name)] = command.decode(errors="replace")
        except (OSError, IndexError, ValueError):  # gone meanwhile
            continue
    return found


def alive(pid: int) -> bool:
    """Whether the process `pid` runs: a zombie, ended and not yet reaped, does not."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


@needs_ffmpeg
@pytest.mark.parametrize("ends_with_us", [True, False], ids=["with_us", "control_without"])
def test_a_killed_run_leaves_none_of_its_assemblys_ffmpeg(
    film_directory: Path, ends_with_us: bool
) -> None:
    # DESIGN.md, Output, Assembly (decided on 2026-10-12): ffmpeg's children of the assembly end
    # with seedvr2x. The join (slowed to the source's pace, STOPPING) running, seedvr2x is
    # SIGKILLed, alone, not its group: its ffmpeg, in a group of its own, is gone within seconds.
    # The control, the helper replaced by none, shows the test would see one left.
    driver = STOPPING.replace(
        "subprocess.Popen = slow_join",
        "subprocess.Popen = slow_join"
        + (
            ""
            if ends_with_us
            else "\nimport seedvr2x.media.assembly as a\na.ends_with_us = lambda: None"
        ),
    )
    process = subprocess.Popen(
        [
            *(sys.executable, "-c", driver, str(test_stop.TESTS), "assembly"),
            *("film.mkv", "-o", "x.mkv", "--model-dir", ".", "--dit-model", "w.safetensors"),
            *("--vae-model", "w.safetensors", "--resolution", "96", "--seed", "42", *FILM),
            *("--color-correction", "none", "--format", "gbrp16le"),
        ],
        stderr=subprocess.PIPE,
        text=True,
        cwd=film_directory,
        start_new_session=True,
        env={**os.environ, "HF_HOME": str(film_directory / "hf"), "HF_HUB_OFFLINE": "1"},
    )
    assert process.stderr is not None
    left: dict[int, str] = {}
    try:
        for line in process.stderr:
            if line.startswith("IN assembly"):
                left = {pid: c for pid, c in children_of(process.pid).items() if "concat" in c}
                break
        assert len(left) == 1, left
        process.kill()
        process.wait(timeout=30)
        time.sleep(1.5)
        assert [pid for pid in left if alive(pid)] == ([] if ends_with_us else list(left))
    finally:
        for pid in left:
            if alive(pid):
                os.kill(pid, signal.SIGKILL)
        process.stderr.close()
    assert not (film_directory / "x.mkv").exists()


@pytest.fixture
def film_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    film_job(tmp_path, monkeypatch)
    (tmp_path / "w.safetensors").write_bytes(b"")
    return tmp_path


@needs_ffmpeg
def test_ctrl_c_once_stops_before_the_assembly_or_lets_it_finish(film_directory: Path) -> None:
    # The assembly is a unit, the job's last. Ctrl-C once while the last segment is decoded and
    # written: that unit finished, the run stops before the assembly, exit 130, every segment
    # kept, and the same command assembles. Ctrl-C once while ffmpeg joins: the assembly
    # finishes, exit 0.
    final, work = film_directory / "x.mkv", film_directory / "x.mkv.work"
    status, text = stopping(film_directory, "decode 77", (signal.SIGINT,))
    assert status == 130, text
    assert re.search(r"^\d\d:\d\d:\d\d stopping after segment 4/4's decode and write", text, re.M)
    assert "stopped before the assembly, as asked; every segment finished, 4 in" in text
    assert "x.mkv.work: the same command assembles x.mkv" in text
    assert listing(work) == FINISHED and not final.exists()
    status, text = stopping(film_directory, "assembly", (signal.SIGINT,))
    assert status == 0, text
    assert re.search(
        r"^\d\d:\d\d:\d\d stopping after the assembly of x.mkv; Ctrl-C again", text, re.M
    )
    # Its segments all made, the file not yet in place: not "finished already" (DESIGN.md, Pause
    # and resume: "A finished file's job is finished once the file is in place").
    assert "every segment made, its 4 segments: assembling" in text and "CALLS \n" in text
    assert "finished already" not in text
    assert not work.exists()
    held_to_the_source(final, film_directory / "film.mkv", film_directory)


@needs_ffmpeg
@pytest.mark.parametrize(
    ("signals", "status", "said"),
    [
        ((signal.SIGINT, signal.SIGINT), 130, "stopped at once (Ctrl-C)"),
        ((signal.SIGTERM,), 143, "stopped at once (SIGTERM)"),
    ],
)
def test_stopped_at_once_during_the_assembly(
    film_directory: Path, signals: tuple[int, ...], status: int, said: str
) -> None:
    # Ctrl-C twice, or SIGTERM, while ffmpeg joins: the run stops at once, the join's ffmpeg
    # with it, no partial file left, an older file at the path as it was, the work directory
    # whole; the same command then assembles.
    final, work = film_directory / "x.mkv", film_directory / "x.mkv.work"
    # The job made first and stopped before its assembly, so that the older file at the path has
    # a job's work directory beside it, not a new job to refuse (DESIGN.md, Output, Assembly,
    # decided on 2026-10-12: a file with no manifest is refused at a new job's start).
    assert stopping(film_directory, "decode 77", (signal.SIGINT,))[0] == 130
    final.write_bytes(b"an older file")
    started = time.monotonic()
    found, text = stopping(film_directory, "assembly", signals)
    assert found == status, text
    assert said in text and time.monotonic() - started < 30
    assert listing(work) == FINISHED
    assert final.read_bytes() == b"an older file"
    assert not (film_directory / "x.mkv.partial").exists() and joining(film_directory) == []
    found, text = stopping(film_directory)
    assert found == 0, text
    assert not work.exists()
    held_to_the_source(final, film_directory / "film.mkv", film_directory)


@needs_ffmpeg
def test_first_frame_before_the_files_start_refused_before_any_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # A source whose first video frame comes before its file's start time, which the assembly
    # can't place and refuses: said once the first pass has timed the frames, before the models
    # load and any unit is made, nothing left; the same source accepted with a directory, never
    # assembled. No such file was found to make: the join's start is given one.
    import numpy as np

    from seedvr2x.media import assembly
    from seedvr2x.media.index import FrameIndex
    from seedvr2x.runtime import model
    from seedvr2x.runtime.job import JobError

    def index(start_time: int) -> FrameIndex:
        none, times = np.zeros(1, np.uint32), np.array([23], np.int64)
        return FrameIndex(Fraction(1, 1000), start_time, times, none, none.astype(bool), times)

    check = cli._grid_start  # pyright: ignore[reportPrivateUsage]
    late: Any = SimpleNamespace(path=Path("in.mkv"), index=index(0))
    early: Any = SimpleNamespace(path=Path("in.mkv"), index=index(50_000))
    check(late)
    said = "in.mkv: its first video frame is 27.000 ms before the file's start time"
    with pytest.raises(JobError, match=said):
        check(early)
    source = job(tmp_path, monkeypatch)
    monkeypatch.setattr(assembly, "start", lambda index: Fraction(-27, 1000))
    with monkeypatch.context() as context:
        context.setattr(model, "load_models", lambda *a: pytest.fail("the models loaded"))
        text = refused(tmp_path, source, caplog, *JOB, output="x.mkv")
    assert f"{source}: its first video frame is 27.000 ms before the file's start time" in text
    assert "give -o a directory" in text and steps.calls == []
    assert not [path.name for path in tmp_path.iterdir() if path.name.startswith("x.mkv")]
    assert upscale(tmp_path, source, "out", *JOB) == 0


@needs_ffmpeg
def test_output_that_is_the_source_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # -o naming the source itself, however the two are spelled (a relative name and an absolute
    # one, through a linked directory, through ..): the assembled file would replace the source.
    # Refused before the first pass, nothing made; and its checksums' file likewise.
    source = job(tmp_path, monkeypatch)
    before = source.read_bytes()
    passes = first_passes(monkeypatch)
    (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
    (tmp_path / "sub").mkdir()
    there = sorted(path.name for path in tmp_path.iterdir())
    spelled = ((Path("in.mkv"), "in.mkv"), (source, "link/in.mkv"), (source, "sub/../in.mkv"))
    for given, output in spelled:
        text = refused(tmp_path, given, caplog, *JOB, output=output)
        assert f"{tmp_path / output}: the output is the source itself; choose another -o" in text
    sums = tmp_path / "x.mkv.crc32"
    sums.write_bytes(before)
    text = refused(tmp_path, sums, caplog, *JOB, output="x.mkv")
    assert f"{tmp_path / 'x.mkv'}: the output's checksums, {sums}, is the source itself" in text
    sums.unlink()
    assert passes == [] and steps.calls == [] and source.read_bytes() == before
    models = ["transnetv2.safetensors", "w.safetensors"]
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted([*there, *models])
    # Another name beside it is a job like any other.
    assert upscale(tmp_path, source, "out.mkv", *JOB) == 0 and source.read_bytes() == before


@needs_ffmpeg
def test_work_directory_left_is_only_the_job_askeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # A work directory left with its record of an assembly, here for a file of someone else's
    # in it: the same command says the job assembled already, exit 0, and what keeps the
    # directory, once; another job asked to the same -o, another source and seed, is refused,
    # exit 1, its differences listed, nothing removed: the file there isn't its own. With the
    # file gone, the same command finishes the removal.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source = film_job(tmp_path, monkeypatch)
    final, work = tmp_path / "x.mkv", tmp_path / "x.mkv.work"
    clear = resume.clear

    def with_notes(directory: Path, content: dict[str, Any]) -> list[str]:
        (directory / "notes.txt").write_text("mine")
        return clear(directory, content)

    with monkeypatch.context() as context:
        context.setattr(resume, "clear", with_notes)
        assert upscale(tmp_path, source, "x.mkv", *FILM) == 0
    kept = f"{work}: not removed, holding what isn't this job's: notes.txt"
    assert kept in caplog.text and listing(work) == ["manifest.json", "notes.txt"]
    assembled = final.read_bytes()
    passes = first_passes(monkeypatch)
    caplog.clear()
    assert upscale(tmp_path, source, "x.mkv", *FILM) == 0
    assert f"{final}: assembled already, {len(assembled)} bytes" in caplog.text
    assert caplog.text.count(kept) == 1 and "removed, its job assembled" not in caplog.text
    other = film(tmp_path / "other.mkv", frames=48)
    text = refused(tmp_path, other, caplog, *FILM, "--seed", "7", output="x.mkv")
    assert f"{work}: the work directory of another job than the one asked, which {final}" in text
    assert "settings.seed: 42 -> 7" in text
    assert f"input.bytes: {source.stat().st_size} -> {other.stat().st_size}" in text
    assert f"Nothing was removed: remove {work}, or choose another -o" in text
    assert "assembled already" not in text
    assert listing(work) == ["manifest.json", "notes.txt"] and final.read_bytes() == assembled
    assert passes == [] and (work / "notes.txt").read_text() == "mine"
    (work / "notes.txt").unlink()
    caplog.clear()
    assert upscale(tmp_path, source, "x.mkv", *FILM) == 0
    assert f"{work}: removed, its job assembled" in caplog.text and not work.exists()
    assert passes == [] and final.read_bytes() == assembled


@needs_ffmpeg
def test_removal_finished_for_a_job_that_detects_its_shots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps
) -> None:
    # The job asked is told from the one recorded by what the command line gives, the records
    # that take a hash left out (cli.HASHED): the shot detector's among them, which a job
    # without a cut list records. The same command is the same job.
    source = job(tmp_path, monkeypatch)
    options = ("--min-segment", "0.12")

    def stopped_at_once(*arguments: object) -> list[str]:
        raise KeyboardInterrupt

    with monkeypatch.context() as context:
        context.setattr(resume, "clear", stopped_at_once)
        assert upscale(tmp_path, source, "x.mkv", *options) == 130
    work = tmp_path / "x.mkv.work"
    recorded = json.loads((work / NAME).read_text())["settings"]
    assert set(cli.HASHED) <= set(recorded) and "detector_model" in cli.HASHED
    steps.calls.clear()
    assert upscale(tmp_path, source, "x.mkv", *options) == 0
    assert steps.calls == [] and not work.exists()


@needs_ffmpeg
def test_stream_before_the_files_start_refused_before_any_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # A source whose subtitles start 4 s before its file's start time (tests/test_assembly.py's):
    # ffmpeg's muxer would move every stream, and the assembly refuse its own result once every
    # segment is made. Refused for a .mkv path before the first pass, naming the stream and the
    # remux that leaves it out; accepted with a directory, never assembled.
    from test_assembly import RATES, early_subtitle, make_job

    monkeypatch.chdir(tmp_path)
    made = make_job(tmp_path / "made", RATES["25"], 3)
    source = early_subtitle(tmp_path / "source", made)
    passes = first_passes(monkeypatch)
    text = refused(tmp_path, source, caplog, output="x.mkv")
    assert f"{source}: its stream 3 (subtitle, subrip) starts at 1.000 s, 4.000 s before" in text
    assert "ffmpeg -i SOURCE -map 0 -map -0:3 -c copy without.mkv" in text
    assert passes == [] and steps.calls == []
    assert not [path.name for path in tmp_path.iterdir() if path.name.startswith("x.mkv")]
    assert upscale(tmp_path, source, "out") == 0
    assert (tmp_path / "out" / "seg_000000.mkv").is_file()


@needs_ffmpeg
def test_work_directory_another_holds_is_not_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # A work directory another seedvr2x made and locked between this run's look and its lock:
    # refused, and not this run's to remove as it leaves, empty as it is.
    from seedvr2x.runtime.job import JobError

    source = job(tmp_path, monkeypatch)
    work = tmp_path / "x.mkv.work"

    def taken(directory: Path) -> None:
        directory.mkdir()
        raise JobError(f"{directory}: another seedvr2x is writing it")

    monkeypatch.setattr(cli, "_lock", taken)
    text = refused(tmp_path, source, caplog, *JOB, output="x.mkv")
    assert f"{work}: another seedvr2x is writing it" in text and work.is_dir()


@needs_ffmpeg
def test_removal_the_file_system_refuses_is_said(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # The work directory's removal failing, once the file is in place: said as not removed,
    # with the reason, and the run ends as the job did, exit 0; no traceback.
    import errno

    source = job(tmp_path, monkeypatch)
    final, work = tmp_path / "x.mkv", tmp_path / "x.mkv.work"
    rmdir = Path.rmdir

    def busy(path: Path) -> None:
        if path == work:
            raise OSError(errno.EBUSY, os.strerror(errno.EBUSY), str(path))
        rmdir(path)

    monkeypatch.setattr(Path, "rmdir", busy)
    assert upscale(tmp_path, source, "x.mkv", *JOB) == 0
    assert f"{work}: not removed: [Errno 16] Device or resource busy" in caplog.text
    assert "its job is assembled" in caplog.text and "Traceback" not in caplog.text
    assert indexes(final) == list(range(25)) and cli.main(["verify", str(final)]) == 0


@needs_ffmpeg
def test_sigterm_during_the_rehearsal_stops_its_ffmpeg(film_directory: Path) -> None:
    # SIGTERM while ffmpeg rehearses the other streams' remux, at the job's start, before the
    # run's handlers: unwound as during the run, exit 143, its ffmpeg stopped, nothing left.
    started = time.monotonic()
    status, text = stopping(film_directory, "rehearsal", (signal.SIGTERM,))
    assert status == 143, text
    assert "stopped at once (SIGTERM)" in text and time.monotonic() - started < 30
    assert joining(film_directory) == []
    assert not [path.name for path in film_directory.iterdir() if path.name.startswith("x.mkv")]


# The remux rehearsal said, recorded with the first pass's, and trusted by a resume (DESIGN.md,
# Output, Assembly, "Met on 2026-10-23", decided on 2026-10-28).
def rehearsals(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> list[str]:
    """The lines logged at each rehearsal (assembly.rehearse) from now on, as it starts: the log
    so far, one entry a rehearsal, so that a test sees what was said before it read the source."""
    from seedvr2x.media import assembly

    seen: list[str] = []
    rehearse = assembly.rehearse

    def spy(source: Path, work: Path) -> str:
        seen.append(caplog.text)
        return rehearse(source, work)

    monkeypatch.setattr(assembly, "rehearse", spy)
    return seen


@needs_ffmpeg
def test_a_new_jobs_rehearsal_is_said_before_it_reads_and_recorded_with_the_first_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source = film_job(tmp_path, monkeypatch)
    seen = rehearsals(monkeypatch, caplog)
    steps.stop = "encode 30"
    assert upscale(tmp_path, source, "x.mkv", *FILM) == 130
    work = tmp_path / "x.mkv.work"
    [before] = seen
    line = f"{source}: the assembly rehearsed: the source's other streams remuxed once, reading its"
    assert line in before and before.index(line) > before.index("ffmpeg ")
    assert "the first pass" not in before
    # Recorded in the input's record, with the first pass's: the ffmpeg and conversions it ran
    # under; the manifest is that of a stopped run, units made.
    content = json.loads((work / NAME).read_text())
    ffmpeg_version = ffmpeg.check(assembly=True)
    [conversions] = {content["environment"]["conversions"]}
    assert content["input"]["rehearsal"] == {
        "passed": True,
        "ffmpeg": ffmpeg_version,
        "conversions": conversions,
    }
    assert content["input"]["rehearsal"]["ffmpeg"] == content["environment"]["ffmpeg"]


@needs_ffmpeg
def test_a_resume_trusts_the_recorded_rehearsal_and_runs_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source = film_job(tmp_path, monkeypatch)
    seen = rehearsals(monkeypatch, caplog)
    steps.stop = "encode 30"
    assert upscale(tmp_path, source, "x.mkv", *FILM) == 130
    assert len(seen) == 1
    recorded = (tmp_path / "x.mkv.work" / NAME).read_text()
    steps.stop = None
    caplog.clear()
    assert upscale(tmp_path, source, "x.mkv", *FILM) == 0
    assert len(seen) == 1
    assert "the assembly's rehearsal trusted from the job's record" in caplog.text
    assert "the assembly rehearsed" not in caplog.text
    # The same content compared before: the trusting line comes after the input's SHA-256.
    assert caplog.text.index("SHA-256") < caplog.text.index("rehearsal trusted")
    assert json.loads(recorded)["input"]["rehearsal"]["passed"] is True
    held_to_the_source(tmp_path / "x.mkv", source, tmp_path)
    # The manifest left beside the finished file holds the rehearsal too.
    beside = json.loads((tmp_path / "x.mkv.manifest.json").read_text())
    assert beside["input"]["rehearsal"]["passed"] is True


@needs_ffmpeg
def test_a_resume_under_another_ffmpeg_accepted_rehearses_again_and_records_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source = film_job(tmp_path, monkeypatch)
    seen = rehearsals(monkeypatch, caplog)
    steps.stop = "encode 30"
    assert upscale(tmp_path, source, "x.mkv", *FILM) == 130
    check = ffmpeg.check
    monkeypatch.setattr(ffmpeg, "check", lambda *options, **needs: "n0.0-another")
    steps.stop = "encode 50"
    caplog.clear()
    assert upscale(tmp_path, source, "x.mkv", *FILM, "--accept-env-change") == 130
    assert len(seen) == 2
    assert "not one of this ffmpeg: run again" in caplog.text
    assert "the assembly rehearsal trusted" not in caplog.text
    content = json.loads((tmp_path / "x.mkv.work" / NAME).read_text())
    assert content["input"]["rehearsal"]["ffmpeg"] == "n0.0-another"
    # Back on that ffmpeg, the record is its own: trusted.
    steps.stop = None
    caplog.clear()
    assert upscale(tmp_path, source, "x.mkv", *FILM, "--accept-env-change") == 0
    assert len(seen) == 2 and "rehearsal trusted from the job's record" in caplog.text
    ffmpeg.check = check  # monkeypatch restores it


@needs_ffmpeg
def test_a_resume_with_no_rehearsal_recorded_is_refused_by_it_after_the_content_is_compared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    from seedvr2x.media import assembly

    caplog.set_level(logging.INFO, logger="seedvr2x")
    source = film_job(tmp_path, monkeypatch)
    steps.stop = "encode 30"
    assert upscale(tmp_path, source, "x.mkv", *FILM) == 130
    path = tmp_path / "x.mkv.work" / NAME
    content = json.loads(path.read_text())
    del content["input"]["rehearsal"]
    path.write_text(json.dumps(content))
    kept = path.read_bytes()
    # A rehearsal that would refuse the source now (a stream made unmuxable is not feasible
    # without changing its content, which is another job): replaced by one that refuses.
    monkeypatch.setattr(assembly, "rehearse", lambda *given: "the remux says no")
    steps.stop = None
    text = refused(tmp_path, source, caplog, *FILM, output="x.mkv")
    assert "no rehearsal of the assembly in the job's record: run now" in text
    assert "the remux says no" in text
    # After the input's content was compared (its SHA-256 is hashed by the comparison), not
    # before the model files and the comparison, as it was.
    assert text.index("SHA-256") < text.index("the remux says no")
    assert path.read_bytes() == kept
    # Another content under the same record: another job, refused by the comparison, the
    # rehearsal not run.
    ran: list[Path] = []
    monkeypatch.setattr(assembly, "rehearse", lambda *given: ran.append(given[0]) or "")
    other = film(tmp_path / "other.mkv", frames=97)
    text = refused(tmp_path, other, caplog, *FILM, output="x.mkv")
    assert "another job than the one asked" in text and ran == []


@needs_ffmpeg
def test_a_trusting_resume_still_says_the_data_streams_left_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The warning of a start (assembly.preflight) is recomputed, a header a stream, where the
    # rehearsal is trusted (cli._rehearsed): a MOV with a timecode track.
    from seedvr2x.media import assembly

    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source = tmp_path / "in.mov"
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=64x48:r=25"),
            *("-frames:v", "25", "-c:v", "mpeg4", "-timecode", "01:00:00:00", str(source)),
        ],
        check=True,
    )
    version, conversions = "n1", "abc"
    recorded = {"input": {"rehearsal": {"passed": True, "ffmpeg": version, "conversions": "abc"}}}
    monkeypatch.setattr(assembly, "rehearse", lambda *given: pytest.fail("the rehearsal ran"))
    cli._rehearsed(source, tmp_path, recorded, version, conversions)  # pyright: ignore[reportPrivateUsage]
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warned) == 1 and "left out of the finished file" in warned[0]
    assert "rehearsal trusted from the job's record" in caplog.text

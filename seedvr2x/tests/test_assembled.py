"""A .mkv path's job on the CPU, the model replaced by tests/test_cli_run.py's stand-in: run as a
directory job in its work directory, then assembled into the finished file with the source's other
streams, the work directory removed once the file is whole (DESIGN.md, Output, Assembly;
Validation milestones, 6). The join itself is tests/test_assembly.py's; here, what the command
line does with it: when it assembles, what a stop or a failure keeps, what it refuses, and the
finished file held to milestone 6's acceptance against the source."""

import json
import logging
import os
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
    JOB,
    ScriptedDetector,
    Steps,
    checksums,
    encoded,
    first_passes,
    indexes,
    job,
    needs_x264,
    rate_of,
    refused,
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
    assert listed == ["x.mkv", "x.mkv.crc32"]
    assert f"wrote {final}: 96 frames from 4 segments, with the source's 4 other streams and 2" in (
        caplog.text
    )
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
    final.write_bytes(b"an older file")
    sums.write_bytes(b"its checksums")
    with monkeypatch.context() as full:
        full.setattr(os, "statvfs", lambda path: SimpleNamespace(f_bavail=0, f_frsize=4096))
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
    # from its segments, the models not loaded. The job is made again first, whole, a file at
    # the path with no work directory being no obstacle to a new one.
    for other in (b"another file", None):
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
    from seedvr2x.runtime import detector, model

    caplog.set_level(logging.INFO, logger="seedvr2x")
    monkeypatch.setattr(detector, "Detector", ScriptedDetector)
    source = job(tmp_path, monkeypatch)
    passes = first_passes(monkeypatch)
    final, work, plan = (tmp_path / name for name in ("x.mkv", "x.mkv.work", "plan.txt"))
    with monkeypatch.context() as context:
        context.setattr(model, "load_models", lambda *a: pytest.fail("the models loaded"))
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
    assert "seedvr2x: stopping after segment 4/4's decode and write" in text
    assert "stopped before the assembly, as asked; every segment finished, 4 in" in text
    assert "x.mkv.work: the same command assembles x.mkv" in text
    assert listing(work) == FINISHED and not final.exists()
    status, text = stopping(film_directory, "assembly", (signal.SIGINT,))
    assert status == 0, text
    assert "seedvr2x: stopping after the assembly of x.mkv; Ctrl-C again" in text
    assert "finished already, its 4 segments" in text and "CALLS \n" in text
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

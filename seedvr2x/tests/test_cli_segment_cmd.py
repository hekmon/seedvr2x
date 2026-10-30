"""--segment-cmd through the command line, on the CPU, the model replaced by tests/test_cli_run.py's
stand-in (DESIGN.md, Output, Writer): the user's command writing each output segment, a setting
the manifest records, tried before any work, its segments without checksums, what a failing
command or a stop keeps and how the same command resumes, and `seedvr2x verify` on such an output.
The writer itself, and the stream the command reads, are tests/test_segment_cmd.py's."""

import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_assembled import (
    FILM,
    HOLED_TIMES,
    film_job,
    held_to_the_source,
    holed_film,
    needs_ffmpeg,
    packets,
    video_times,
)
from test_cli_run import (
    JOB,
    SEEKING,
    UNINTERRUPTED,
    Steps,
    decoded,
    indexes,
    job,
    needs_x264,
    refused,
    stand_in_model,
    stopped,
    upscale,
)
from test_probe import has_encoder

from seedvr2x import cli
from seedvr2x.media import ffmpeg
from seedvr2x.media.ffmpeg import MediaError
from seedvr2x.media.writer import count_frames


def _usable() -> bool:
    try:
        ffmpeg.check()
    except MediaError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _usable(), reason="needs ffmpeg 7.1 or later with zscale, idet and ffv1"
)

# Always there: MPEG-4 with B-frames, which a packet count or a timestamp read in file order
# would get wrong; and FFV1, lossless, whose frames are the stand-in's.
MPEG4 = "ffmpeg -i - -c:v mpeg4 -bf 2 -q:v 2 {out}"
FFV1 = "ffmpeg -i - -c:v ffv1 {out}"
# DESIGN.md's own example (Output, Writer).
X265 = "ffmpeg -i - -c:v libx265 -crf 16 {out}"
SEGMENTS = ("seg_000000.mkv", "seg_000001.mkv")


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


def codec(path: Path) -> str:
    """The codec of path's first video stream, and whether it reorders frames (B-frames)."""
    probed = subprocess.run(
        [
            *("ffprobe", "-v", "error", "-select_streams", "v:0"),
            *("-show_entries", "stream=codec_name,has_b_frames", "-of", "csv=p=0", str(path)),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return probed.stdout.strip()


@pytest.mark.parametrize(
    ("command", "found"),
    [
        pytest.param(MPEG4, "mpeg4,1", id="mpeg4"),
        pytest.param(
            X265,
            "hevc,2",
            id="libx265",
            marks=pytest.mark.skipif(
                not has_encoder("libx265"), reason="needs ffmpeg with libx265"
            ),
        ),
    ],
)
def test_directory_of_command_segments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
    command: str,
    found: str,
) -> None:
    # Each segment written by the command, in the default format: every one there with its
    # frames, in the command's codec, no checksums; the command a setting, the sizes recorded.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    options = (*JOB, "--format", "yuv420p10le", "--segment-cmd", command)
    assert upscale(tmp_path, source_path, "out", *options) == 0
    assert steps.calls == UNINTERRUPTED
    out = tmp_path / "out"
    listed = ["frame_index.bin", "manifest.json", *SEGMENTS]
    assert sorted(path.name for path in out.iterdir()) == listed
    assert [count_frames(out / name) for name in SEGMENTS] == [4, 21]
    assert [codec(out / name) for name in SEGMENTS] == [found] * 2
    content = json.loads((out / "manifest.json").read_text())
    assert content["settings"]["segment_cmd"] == command
    assert [(s["finished"], s["bytes"]) for s in content["segments"]] == [
        (True, (out / name).stat().st_size) for name in SEGMENTS
    ]
    # The hash of each file the command wrote, recorded beside its bytes.
    assert [s["sha256"] for s in content["segments"]] == [
        hashlib.sha256((out / name).read_bytes()).hexdigest() for name in SEGMENTS
    ]
    assert "--segment-cmd: tried on 5 black frames, 128x96 yuv420p10le at 25 fps" in caplog.text
    # verify counts each segment's frames, and says that it checked nothing else.
    caplog.clear()
    assert cli.main(["verify", str(out)]) == 0
    assert f"{out / 'seg_000001.mkv'}: its 21 frames there, counted" in caplog.text
    assert f"{out}: 2 checked, by their frame counts alone: a --segment-cmd output" in caplog.text
    assert cli.main(["verify", str(out / "seg_000000.mkv")]) == 0
    # A segment holding other frames than its own: named, and the check fails.
    (out / "seg_000001.mkv").write_bytes((out / "seg_000000.mkv").read_bytes())
    caplog.clear()
    assert cli.main(["verify", str(out)]) == 1
    assert f"{out / 'seg_000001.mkv'}: 4 frames, where 21 were written" in caplog.text
    assert f"{out}: of 2, 1 not as written" in caplog.text
    # The resume takes its segments without their checksums, by their sizes: one of another size
    # is made again (DESIGN.md, Output, Assembly, decided on 2026-10-29), by the command, where it
    # used to be refused, and the segment is the command's output once more, recorded as such.
    caplog.clear()
    steps.calls.clear()
    assert upscale(tmp_path, source_path, "out", *options) == 0
    size = (out / "seg_000000.mkv").stat().st_size
    assert (
        f"{out / 'seg_000001.mkv'}: its bytes changed since it was finished (frames 4 to 24 of"
        f" the job: {size} bytes, where {content['segments'][1]['bytes']} were recorded): the disk"
        " or another program is to suspect; made again now"
    ) in caplog.text
    assert steps.calls == ["encode 4", "window 4:0", "window 4:1", "decode 4"]
    assert [count_frames(out / name) for name in SEGMENTS] == [4, 21]
    assert [codec(out / name) for name in SEGMENTS] == [found] * 2
    assert not (out / "checksums").exists()
    again = json.loads((out / "manifest.json").read_text())["segments"]
    assert [(s["finished"], s["bytes"]) for s in again] == [
        (True, (out / name).stat().st_size) for name in SEGMENTS
    ]
    assert cli.main(["verify", str(out)]) == 0


def test_without_a_command_the_setting_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps
) -> None:
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    content = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert content["settings"]["segment_cmd"] is None
    assert (tmp_path / "out" / "checksums").is_dir()


def failing(flag: Path) -> str:
    """A lossless command that fails on the second segment, exit status 3, while flag exists:
    the same command, a setting, before and after its cause is removed."""
    refuse = f"case {{out}} in *seg_000001*) test ! -e '{flag}' || exit 3;; esac"
    return f"{refuse}; {FFV1}"


def test_failing_command_stops_at_its_segment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # The command failing on the second segment: the run stops there, exit 1, with the command's
    # status and what is kept: the first segment, finished, and the second's shots' windows. The
    # same command, its cause removed, decodes and writes that segment again, no shot encoded or
    # sampled again, and the output is an uninterrupted run's, frame for frame.
    source_path = job(tmp_path, monkeypatch)
    flag = tmp_path / "fail"
    options = (*JOB, "--segment-cmd", failing(flag))
    assert upscale(tmp_path, source_path, "whole", *options) == 0
    flag.touch()
    steps.calls.clear()
    assert upscale(tmp_path, source_path, "out", *options) == 1
    out = tmp_path / "out"
    assert f"{out / 'seg_000001.mkv'}: the segment command exited with status 3" in caplog.text
    assert "; stopped: 1 of 2 segments finished, and the units of the next ones kept" in caplog.text
    assert steps.calls == UNINTERRUPTED
    content = json.loads((out / "manifest.json").read_text())
    assert [segment["finished"] for segment in content["segments"]] == [True, False]
    assert [shot["windows_done"] for shot in content["shots"]] == [1, 1, 2]
    listed = ["frame_index.bin", "manifest.json", "resume", "seg_000000.mkv"]
    assert sorted(path.name for path in out.iterdir()) == listed
    windows = ["window_0000.pt", "window_0001.pt"]
    assert sorted(path.name for path in (out / "resume" / "shot_000004").iterdir()) == windows
    first = (out / "seg_000000.mkv").stat()
    flag.unlink()
    steps.calls.clear()
    assert upscale(tmp_path, source_path, "out", *options) == 0
    assert steps.calls == ["decode 4"]
    after = (out / "seg_000000.mkv").stat()
    assert (after.st_ino, after.st_mtime_ns) == (first.st_ino, first.st_mtime_ns)
    for name in SEGMENTS:
        assert decoded(out / name) == decoded(tmp_path / "whole" / name)
    assert sorted(path.name for path in out.iterdir()) == [listed[0], listed[1], *SEGMENTS]


def test_stopped_once_then_resumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # Ctrl-C once while the first segment is decoded and written: the command finishes it, the
    # run stops before the next unit, and the same command resumes, the finished segment kept as
    # it is; a partial file a killed command left is discarded on the way.
    import os
    import signal

    from seedvr2x.media.files import command_partial
    from seedvr2x.runtime import run

    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    options = (*JOB, "--segment-cmd", FFV1)
    assert upscale(tmp_path, source_path, "whole", *options) == 0
    decode = steps.decode

    def interrupted(*arguments: object, **tiled: object) -> None:
        if not any(call.startswith("decode") for call in steps.calls):
            os.kill(os.getpid(), signal.SIGINT)
        decode(*arguments, **tiled)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(run, "decode_shot", interrupted)
    steps.calls.clear()
    assert upscale(tmp_path, source_path, "out", *options) == 130
    assert steps.calls == UNINTERRUPTED[:6]
    assert "stopped after segment 1/2's decode and write, as asked; 1 of 2 segments" in caplog.text
    out = tmp_path / "out"
    first = (out / "seg_000000.mkv").stat()
    left = command_partial(out / "seg_000001.mkv")
    assert left.name == "seg_000001.partial.mkv"
    left.write_bytes(b"left by a command that was killed")
    monkeypatch.setattr(run, "decode_shot", decode)
    steps.calls.clear()
    assert upscale(tmp_path, source_path, "out", *options) == 0
    assert steps.calls == UNINTERRUPTED[6:]
    assert "1 leftover discarded" in caplog.text and not left.exists()
    after = (out / "seg_000000.mkv").stat()
    assert (after.st_ino, after.st_mtime_ns) == (first.st_ino, first.st_mtime_ns)
    for name in SEGMENTS:
        assert decoded(out / name) == decoded(tmp_path / "whole" / name)
    # A partial file of the command's is no one's in a job without one: refused, never deleted.
    assert upscale(tmp_path, source_path, "plain", *JOB) == 0
    stray = tmp_path / "plain" / "seg_000001.partial.mkv"
    stray.write_bytes(b"mine")
    assert upscale(tmp_path, source_path, "plain", *JOB) == 1
    assert "not this job's: seg_000001.partial.mkv" in caplog.text and stray.exists()


def test_another_command_is_another_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # The command is a setting: a resume with another, or with none, or a command given to a job
    # that had none, is refused, the difference listed, whatever the job has made.
    source_path = job(tmp_path, monkeypatch)
    steps.stop = "encode 4"
    stopped(tmp_path, source_path, "out", *JOB, "--segment-cmd", FFV1)
    steps.stop = None
    text = refused(tmp_path, source_path, caplog, *JOB, "--segment-cmd", MPEG4)
    assert "another job than the one asked, which differs in:" in text
    assert f'settings.segment_cmd: "{FFV1}" -> "{MPEG4}"' in text
    text = refused(tmp_path, source_path, caplog, *JOB)
    assert f'settings.segment_cmd: "{FFV1}" -> null' in text
    assert upscale(tmp_path, source_path, "plain", *JOB) == 0
    text = refused(tmp_path, source_path, caplog, *JOB, "--segment-cmd", FFV1, output="plain")
    assert f'settings.segment_cmd: null -> "{FFV1}"' in text
    assert steps.calls.count("decode 4") == 1  # the plain job's, no other


@pytest.mark.parametrize(
    ("command", "said"),
    [
        ("ffmpeg -i - -c:v no_such_encoder {out}", "Unknown encoder 'no_such_encoder'"),
        ("exit 3 # {out}", "the segment command exited with status 3"),
        ("cat > /dev/null; true {out}", "seg_000000.mkv"),
        ("ffmpeg -i - -frames:v 3 -c:v ffv1 {out}", "3 frames in the file"),
    ],
)
def test_trial_refuses_before_any_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
    command: str,
    said: str,
) -> None:
    # A command that can't work is found by its trial, before the first pass and the models:
    # an encoder ffmpeg doesn't have, a command that fails, one that writes no file, one that
    # drops frames. Nothing is written, the output directory not even made.
    from test_cli_run import first_passes

    source_path = job(tmp_path, monkeypatch)
    passes = first_passes(monkeypatch)
    before = sorted(path.name for path in tmp_path.iterdir())
    text = refused(tmp_path, source_path, caplog, *JOB, "--segment-cmd", command)
    assert "--segment-cmd, tried on 5 black frames before any work: " in text
    assert said in text
    assert passes == [] and steps.calls == []
    # But the stand-in's model files, which the test's run puts beside the source (upscale).
    models = ["transnetv2.safetensors", "w.safetensors"]
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted([*before, *models])


@pytest.mark.parametrize(
    ("options", "said"),
    [
        (
            ("--format", "png", "--segment-cmd", FFV1),
            "--segment-cmd: refused with --format png",
        ),
        (("--segment-cmd", "ffmpeg -i - -c:v ffv1 out.mkv"), "doesn't name {out}"),
        (("--segment-cmd", " "), "the segment command is empty"),
    ],
)
def test_command_refused_by_its_form(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
    options: tuple[str, ...],
    said: str,
) -> None:
    # Refused as given, before the build's check even: with PNG output, which has no stream to
    # give a command; without {out}, the file it must write; empty.
    source_path = job(tmp_path, monkeypatch)
    monkeypatch.setattr(ffmpeg, "check", lambda *a, **k: pytest.fail("the build checked"))
    assert said in refused(tmp_path, source_path, caplog, *JOB, *options)
    assert not (tmp_path / "out").exists()


def test_plan_runs_the_command_for_its_trial_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps
) -> None:
    # --plan records the command with the plan's other settings, and runs it once, its trial;
    # the same command without --plan then runs the plan: the trial again, then each segment.
    source_path = job(tmp_path, monkeypatch)
    runs = tmp_path / "runs.txt"
    command = f"echo {{out}} >> '{runs}'; {FFV1}"
    options = (*JOB, "--segment-cmd", command)
    assert upscale(tmp_path, source_path, "out", *options, "--plan", "plan.txt") == 0
    out = tmp_path / "out"
    assert sorted(path.name for path in out.iterdir()) == ["frame_index.bin", "manifest.json"]
    content = json.loads((out / "manifest.json").read_text())
    assert content["settings"]["segment_cmd"] == command
    assert [Path(line).name for line in runs.read_text().splitlines()] == ["seg_000000.partial.mkv"]
    assert steps.calls == []
    assert upscale(tmp_path, source_path, "out", *options) == 0
    written = [Path(line) for line in runs.read_text().splitlines()]
    names = ["seg_000000.partial.mkv"] * 3 + ["seg_000001.partial.mkv"]
    assert [path.name for path in written] == names
    # The trials wrote elsewhere, the segments in the output directory.
    assert [path.parent == out for path in written] == [False, False, True, True]
    assert [count_frames(out / name) for name in SEGMENTS] == [4, 21]


@needs_ffmpeg
def test_assembled_from_command_segments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # -o x.mkv with a command writing B-frames, on the film with its other streams: assembled
    # from what the command wrote, milestone 6's acceptance as an FFV1 job's (the video's
    # timestamps the source's, frame for frame in presentation order, every other stream
    # copied); no checksums beside it, those of an older file there gone as it is replaced; its
    # manifest beside it, from which verify counts its frames (DESIGN.md, Output, Assembly,
    # decided on 2026-10-12: "a `--segment-cmd` file's frames can be counted"), exit 1 where
    # they differ from the manifest's; the same command again makes nothing, the command not
    # even tried; given back as a source, accepted.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source = film_job(tmp_path, monkeypatch)
    final, sums = tmp_path / "x.mkv", tmp_path / "x.mkv.crc32"
    record = tmp_path / "x.mkv.manifest.json"
    # The job stopped in its last segment, so that an older file and its checksums at the path
    # have a work directory beside them (a file with no manifest at all is refused).
    steps.stop = "encode 77"
    assert upscale(tmp_path, source, "x.mkv", *FILM, "--segment-cmd", MPEG4) == 130
    steps.stop = None
    final.write_bytes(b"an older file")
    sums.write_bytes(b"its checksums")
    assert upscale(tmp_path, source, "x.mkv", *FILM, "--segment-cmd", MPEG4) == 0
    assert sorted(path.name for path in tmp_path.glob("x.mkv*")) == ["x.mkv", record.name]
    assert codec(final) == "mpeg4,1" and count_frames(final) == 96
    held_to_the_source(final, source, tmp_path)
    caplog.clear()
    assert cli.main(["verify", str(final)]) == 0
    assert f"{final}: its 96 frames there, counted" in caplog.text
    assert f"{final}: 1 checked, by their frame counts alone: a --segment-cmd output has no" in (
        caplog.text
    )
    # The manifest's count differs from the file's: exit 1, saying so; on a copy of the pair.
    other = tmp_path / "other"
    other.mkdir()
    (other / "y.mkv").write_bytes(final.read_bytes())
    edited = json.loads(record.read_text())
    edited["segments"][-1]["end"] += 1
    (other / "y.mkv.manifest.json").write_text(json.dumps(edited))
    caplog.clear()
    assert cli.main(["verify", str(other / "y.mkv")]) == 1
    assert f"{other / 'y.mkv'}: 96 frames, where 97 were written" in caplog.text
    # The same command again: finished, the command not tried, nothing touched.
    monkeypatch.setattr(cli, "_trial", lambda *a: pytest.fail("the command was tried"))
    before = (final.read_bytes(), record.read_bytes(), final.stat().st_mtime_ns)
    caplog.clear()
    assert upscale(tmp_path, source, "x.mkv", *FILM, "--segment-cmd", MPEG4) == 0
    assert f"{final}: finished already, its 96 frames" in caplog.text
    assert (final.read_bytes(), record.read_bytes(), final.stat().st_mtime_ns) == before
    caplog.clear()
    assert upscale(tmp_path, final, "back", "--cuts", "cuts.txt", "--plan", "plan.txt") == 0
    # No warning of its timestamps: the one there is of its matrix, which this command's encoder,
    # fed RGB, leaves untagged.
    warned = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warned) == 1 and "untagged, guessed: matrix smpte170m" in warned[0]
    content = json.loads((tmp_path / "back" / "manifest.json").read_text())
    assert (content["input"]["frames"], content["input"]["frame_rate"]) == (96, "24000/1001")


@needs_ffmpeg
def test_a_damaged_command_segment_is_made_again_then_assembled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # A segment the command wrote, damaged between the last segment and the assembly (one byte
    # flipped, same size): the assembly refused as an FFV1 job's is (test_assembled.py), the
    # command run again for that segment alone by the next run, which assembles.
    source = film_job(tmp_path, monkeypatch)
    options = (*FILM, "--segment-cmd", MPEG4)
    work = tmp_path / "x.mkv.work"
    with monkeypatch.context() as full:
        full.setattr(os, "statvfs", lambda path: SimpleNamespace(f_bavail=0, f_frsize=4096))
        assert upscale(tmp_path, source, "x.mkv", *options) == 1
    recorded = json.loads((work / "manifest.json").read_text())["segments"]
    segment = work / "seg_000002.mkv"
    assert recorded[2]["sha256"] == hashlib.sha256(segment.read_bytes()).hexdigest()
    kept = segment.read_bytes()
    damaged = bytearray(kept)
    damaged[len(damaged) // 2] ^= 0xFF
    segment.write_bytes(bytes(damaged))
    steps.calls.clear()
    caplog.clear()
    assert upscale(tmp_path, source, "x.mkv", *options) == 1
    assert "(frames 50 to 76 of the job); nothing assembled" in caplog.text
    assert not segment.exists() and not (tmp_path / "x.mkv").exists()
    content = json.loads((work / "manifest.json").read_text())["segments"]
    assert [s["finished"] for s in content] == [True, True, False, True]
    assert steps.calls == []
    assert upscale(tmp_path, source, "x.mkv", *options) == 0
    assert steps.calls == ["encode 50", "window 50:0", "decode 50"]
    assert not work.exists() and count_frames(tmp_path / "x.mkv") == 96
    held_to_the_source(tmp_path / "x.mkv", source, tmp_path)


# What a command's file holds, said by the command itself: its frames counted by ffprobe, a line
# a file, in fed.txt where seedvr2x runs.
COUNTED = (
    "ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames"
    " -of csv=p=0 {out} >> fed.txt"
)


@needs_ffmpeg
@needs_x264
def test_missing_frames_filled_through_the_command_to_the_finished_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # The source with two frames missing (tests/test_assembled.py's, with its audio), each
    # segment written by the command, then assembled: the command is fed a frame a place, the
    # hole's among them, its files holding 12, 6, 6 and 16 frames for the segments' places (the
    # trial's 5 first), which seedvr2x counts them by; the finished file holds the 40, each
    # saying its place, the stand-in's way, on the grid of the source's rate from its first
    # timestamp; the audio in step, a remux's packets.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cuts.txt").write_text("12\n18\n24\n")
    source, _ = holed_film(tmp_path)
    command = f"{FFV1} && {COUNTED}"
    assert upscale(tmp_path, source, "x.mkv", *SEEKING, "--segment-cmd", command) == 0
    final = tmp_path / "x.mkv"
    assert (tmp_path / "fed.txt").read_text().split() == ["5", "12", "6", "6", "16"]
    assert f"wrote {final}: 40 frames from 4 segments, with the source's 1 other" in caplog.text
    assert sorted(path.name for path in tmp_path.glob("x.mkv*")) == ["x.mkv", "x.mkv.manifest.json"]
    assert count_frames(final) == 40 and indexes(final) == list(range(40))
    assert video_times(final) == HOLED_TIMES
    assert len(video_times(source)) == 38
    reference = tmp_path / "remux.mkv"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(source), "-map", "0", "-c", "copy", str(reference)],
        check=True,
    )
    assert packets(final, 1) == packets(reference, 1)


@needs_ffmpeg
def test_files_the_assembly_cant_join_refused_at_the_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # A command whose file holds every frame, but off the rate's grid (each twice as long): its
    # count passes, the assembly couldn't place its frames. For a .mkv path, refused at the
    # trial, before the first pass; the same command accepted with a directory, never assembled.
    from test_cli_run import first_passes

    stretched = "ffmpeg -i - -vf setpts=2*PTS -fps_mode passthrough -c:v ffv1 {out}"
    source = job(tmp_path, monkeypatch)
    passes = first_passes(monkeypatch)
    text = refused(tmp_path, source, caplog, *JOB, "--segment-cmd", stretched, output="x.mkv")
    assert "--segment-cmd, tried on 5 black frames before any work: " in text
    assert "seg_000000.mkv: what the segment command wrote can't be assembled into" in text
    assert "its frame 1 is at 80 ms, +40.0 ms from its place at 25 fps" in text
    assert passes == [] and steps.calls == []
    assert not [path.name for path in tmp_path.iterdir() if path.name.startswith("x.mkv")]
    assert upscale(tmp_path, source, "out", *JOB, "--segment-cmd", stretched) == 0
    assert [count_frames(tmp_path / "out" / name) for name in SEGMENTS] == [4, 21]


@needs_ffmpeg
def test_segment_unlike_the_one_before_refused_as_it_is_finished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # A command writing its second segment in another codec than its first: each file is right
    # on its own, and the two can't make one stream. For a .mkv path, the run stops at that
    # segment, exit 1, naming what differs, the first segment kept; not at the assembly.
    mixed = "case {out} in *seg_000001*) codec=mpeg4;; *) codec=ffv1;; esac; "
    mixed += "ffmpeg -i - -c:v $codec {out}"
    source = job(tmp_path, monkeypatch)
    text = refused(tmp_path, source, caplog, *JOB, "--segment-cmd", mixed, output="x.mkv")
    work = tmp_path / "x.mkv.work"
    said = f"{work / 'seg_000001.mkv'}: what the segment command wrote can't be assembled into"
    assert said in text and "codec mpeg4, not ffv1" in text and "as seg_000000.mkv has" in text
    # What is kept, and what the same command would do: not "resumes", as if it would pass.
    assert "; stopped: 1 of 2 segments finished, and the units of the next ones kept: the" in text
    assert "same command decodes that segment again, and meets the same refusal unless" in text
    assert "the same command resumes" not in text
    assert steps.calls == UNINTERRUPTED and not (tmp_path / "x.mkv").exists()
    content = json.loads((work / "manifest.json").read_text())
    assert [segment["finished"] for segment in content["segments"]] == [True, False]


@pytest.mark.parametrize(
    ("command", "accepted"),
    [("cat > {out}", True), ('cat > "{out}"', False), ("cat > '{out}'", False)],
)
def test_out_quoted_by_the_command_refused_at_the_trial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
    command: str,
    accepted: bool,
) -> None:
    # {out} is written bare, seedvr2x quoting the name it stands for: a command that quotes it
    # too would write its first segments, in a directory of a plain name, and fail at the first
    # path with a space or an accent, its GPU work done. The trial's own path has a space, a
    # quote and a dollar: such a command fails there, before the first pass, and is told.
    from test_cli_run import first_passes

    assert cli.TRIAL_PREFIX == "seedvr2x trial '$ "
    source = job(tmp_path, monkeypatch)
    passes = first_passes(monkeypatch)
    status = upscale(tmp_path, source, "out", *JOB, "--segment-cmd", command)
    assert status == (0 if accepted else 1)
    if accepted:
        assert [count_frames(tmp_path / "out" / name) for name in SEGMENTS] == [4, 21]
        return
    assert "--segment-cmd, tried on 5 black frames before any work: " in caplog.text
    assert "{out} is to be written bare, seedvr2x quoting the name it stands for" in caplog.text
    assert passes == [] and steps.calls == [] and not (tmp_path / "out").exists()


def test_verify_fails_on_what_ffprobe_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # A segment of the command's damaged since it was written, a byte of one frame: it still
    # decodes its count, and ffprobe says what it met: verify fails that segment, as it fails a
    # master on anything ffmpeg reports while decoding (DESIGN.md, Output, Checksums).
    source = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source, "out", *JOB, "--segment-cmd", MPEG4) == 0
    out = tmp_path / "out"
    assert cli.main(["verify", str(out)]) == 0
    damaged = out / "seg_000001.mkv"
    listed = subprocess.run(
        [
            *("ffprobe", "-v", "error", "-select_streams", "v:0"),
            *("-show_entries", "packet=pos,size", "-of", "json", str(damaged)),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    packet = json.loads(listed.stdout)["packets"][0]
    data = bytearray(damaged.read_bytes())
    for place in range(int(packet["pos"]) + 40, int(packet["pos"]) + int(packet["size"]), 7):
        data[place] ^= 0xFF
    damaged.write_bytes(data)
    assert count_frames(damaged) == 21
    caplog.clear()
    assert cli.main(["verify", str(out)]) == 1
    assert f"{damaged}: ffprobe, decoding it to count its frames, reported: " in caplog.text
    assert f"{out}: of 2, 1 not as written" in caplog.text
    assert cli.main(["verify", str(out / "seg_000000.mkv")]) == 0


@needs_ffmpeg
@pytest.mark.skipif(not has_encoder("libx265"), reason="needs ffmpeg with libx265")
def test_segments_with_other_headers_that_decode_alike_are_assembled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # A command whose second segment has other codec headers than its first, libx265 run with
    # other threads, as on another machine or after an update: the first's headers decode it
    # all the same, which is asked as the segment is finished and by the assembly, and the job
    # ends in its file, where the headers' difference alone refused it at that segment.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    threads = "case {out} in *seg_000001*) threads=4;; *) threads=1;; esac; "
    threads += "ffmpeg -i - -c:v libx265 -crf 16 -x265-params frame-threads=$threads {out}"
    source = job(tmp_path, monkeypatch)
    options = (*JOB, "--format", "yuv420p10le", "--segment-cmd", threads)
    assert upscale(tmp_path, source, "x.mkv", *options) == 0
    final, work = tmp_path / "x.mkv", tmp_path / "x.mkv.work"
    said = f"{work / 'seg_000001.mkv'}: its codec headers aren't those of seg_000000.mkv"
    assert caplog.text.count(said) == 2  # as it is finished, and by the assembly
    assert count_frames(final) == 25 and codec(final) == "hevc,2" and not work.exists()

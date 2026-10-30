"""The disk a job takes beside its output (runtime/disk.py; DESIGN.md, Memory planner, --plan's
list: "the disk the job takes beside its output: its masters, about 100-135 GiB an hour of 1080p
in yuv420p10le, twice that while a file is assembled, against what its disk has free; a run says
it too, warning where they won't fit", decided on 2026-10-28): the estimate of the masters from
DESIGN's figure at 1080p, at 4K and at an odd size, from the segments a resume has written, a
segment command's unknown and then measured; the line among the plan's lines and the warning,
never a refusal or a status. The expected figures are DESIGN's, written out here again."""

import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_cli_run import (
    JOB,
    Steps,
    job,
    plan_lines,
    stand_in,  # noqa: F401
    steps,  # noqa: F401
    stopped,
    upscale,
)

from seedvr2x.runtime import disk
from seedvr2x.runtime.disk import Disk, Masters, masters

GIB = 2**30
# DESIGN.md, Output, Master sizes: an hour of 1080p at 24000/1001 is 3600 x 24000/1001 frames. In
# frames, 1001 hours of it is a whole number.
FRAMES = 3600 * 24000
HOURS = 1001


def test_figure_at_1080p() -> None:
    # 1001 hours of 1080p in yuv420p10le at the upper end of DESIGN's 100-135 GiB an hour: 135
    # GiB each; gbrp16le's 540-690 GiB an hour at its upper end, 690.
    one = masters([FRAMES], [None], "yuv420p10le", (1080, 1920), False)
    assert one.written == 0 and one.bound and one.total == pytest.approx(HOURS * 135 * GIB)
    assert masters([FRAMES], [None], "gbrp16le", (1080, 1920), False).total == pytest.approx(
        HOURS * 690 * GIB
    )


def test_figure_scales_with_pixels_and_frames() -> None:
    # Four times the pixels at 4K; half the frames, half the bytes; an odd size (even sides, as
    # output_size makes them, not multiples of 16) by its pixels.
    hd = masters([FRAMES], [None], "yuv420p10le", (1080, 1920), False).total
    assert hd is not None
    assert masters([FRAMES], [None], "yuv420p10le", (2160, 3840), False).total == pytest.approx(
        4 * hd
    )
    assert masters([FRAMES // 2], [None], "yuv420p10le", (1080, 1920), False).total == (
        pytest.approx(hd / 2)
    )
    assert masters([FRAMES], [None], "yuv420p10le", (562, 1000), False).total == pytest.approx(
        hd * (562 * 1000) / (1080 * 1920)
    )
    # The same in segments: the sum of their frames.
    assert masters(
        [FRAMES // 4, FRAMES // 4, FRAMES // 2], [None] * 3, "yuv420p10le", (1080, 1920), False
    ).total == pytest.approx(hd)


def test_png_is_bounded_by_its_raw_size() -> None:
    # 16-bit RGB, 6 bytes a pixel, PNG compressing it little: at most its raw size.
    got = masters([10, 5], [None, None], "png", (96, 128), False)
    assert got == Masters(0, 15 * 96 * 128 * 6, True)


def test_a_resume_measures_the_segments_it_has() -> None:
    # Two of three segments written: 100 + 100 frames in 4,000 bytes, 20 a frame, the 50 frames
    # left at that rate in place of the figure; the total counts what is on the disk.
    got = masters([100, 100, 50], [1000, 3000, None], "yuv420p10le", (1080, 1920), False)
    assert got == Masters(4000, 1000, False) and got.total == 5000
    # Every segment made: nothing left to write.
    assert masters([10, 10], [7, 9], "png", (96, 128), False) == Masters(16, 0, False)


def test_a_segment_command_is_unknown_then_measured() -> None:
    # Its encoder decides the size, the content less than the format: unknown before a segment
    # is written; then the written segments' bytes a frame, as for any job.
    before = masters([100, 100], [None, None], "yuv420p10le", (1080, 1920), True)
    assert before == Masters(0, None, True) and before.total is None
    assert masters([100, 100], [500, None], "yuv420p10le", (1080, 1920), True) == Masters(
        500, 500, False
    )


def test_the_need_and_the_words(tmp_path: Path) -> None:
    # A one-file output needs the masters left and the file, as many bytes as the masters in all;
    # a directory the masters left. 186 GiB of masters in 85 GiB free: 287 GiB short with the file.
    new = Masters(0, 186 * GIB, True)
    file = Disk(new, 85 * GIB, True)
    assert file.need == 372 * GIB and file.short == 287 * GIB
    assert file.line() == (
        "disk: the masters at most 186 GiB, twice that, 372 GiB, while the file is assembled, and"
        " the source's other streams; 85 GiB free on its disk, 287 GiB short"
    )
    assert file.warning(Path("film.mkv")) == (
        "film.mkv: its disk has 85 GiB free for about 372 GiB this job still writes: free some"
        " before the assembly, or the run stops there, everything kept"
    )
    directory = Disk(new, 85 * GIB, False)
    assert directory.need == 186 * GIB and directory.short == 101 * GIB
    assert directory.line() == (
        "disk: the masters at most 186 GiB in the output directory; 85 GiB free on its disk,"
        " 101 GiB short"
    )
    warned = directory.warning(Path("out"))
    assert warned is not None and "free some as the run goes" in warned
    # Where it fits: enough, no warning.
    fits = Disk(new, 800 * GIB, True)
    assert fits.short == 0 and fits.line().endswith("800 GiB free on its disk, enough")
    assert fits.warning(Path("film.mkv")) is None
    # Small jobs in smaller units.
    assert (
        Disk(Masters(0, 3 * GIB // 2, True), 2 * GIB, False)
        .line()
        .startswith("disk: the masters at most 1.5 GiB in")
    )
    assert "256 MiB" in Disk(Masters(0, 256 * 2**20, True), 2 * GIB, False).line()


def test_a_resumed_job_says_what_is_written_and_what_it_still_takes() -> None:
    got = Disk(Masters(60 * GIB, 126 * GIB, False), 100 * GIB, True)
    # Still to write: 126 GiB of masters and the file of 186; 312 GiB against 100 GiB free.
    assert got.need == 312 * GIB
    assert got.line() == (
        "disk: the masters about 186 GiB, 60 GiB of them written; the job still takes 312 GiB on"
        " its disk, the file assembled beside them, and the source's other streams; 100 GiB free"
        " on its disk, 212 GiB short"
    )


def test_an_unknown_size_is_said_and_never_warned_of() -> None:
    unknown = Disk(Masters(0, None, True), 1, True)
    assert unknown.need is None and unknown.short == 0
    assert unknown.line() == (
        "disk: the masters' size is not known before one is written, the segment command's"
        " encoder deciding it; 0 MiB free on its disk"
    )
    assert unknown.warning(Path("film.mkv")) is None


def test_free_room_is_the_blocks_free_to_a_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # As the assembly's check reads it (os.statvfs), of the nearest parent that exists.
    seen: list[Path] = []

    def fake(path: Path) -> SimpleNamespace:
        seen.append(path)
        return SimpleNamespace(f_bavail=1000, f_frsize=4096)

    monkeypatch.setattr(os, "statvfs", fake)
    assert disk.free_on(tmp_path / "not" / "yet" / "there") == 4_096_000
    assert seen == [tmp_path]


def lines_of(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [line for line in plan_lines(caplog) if line.startswith("plan: disk: ")]


def warnings_of(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and "this job still writes" in record.getMessage()
    ]


@pytest.mark.parametrize("output", ["out", "one.mkv"])
def test_plan_says_the_disk_and_warns_where_it_is_short(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
    output: str,
) -> None:
    # The line is among --plan's, after the scratch's place and before the predicted peaks; the
    # warning after the plan's lines, where the room is short, and --plan still ends with 0.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    options = JOB if output == "out" else JOB[:4]
    monkeypatch.setattr(disk, "free_on", lambda directory: 1000)
    assert upscale(tmp_path, source_path, output, *options, "--plan", "plan.txt") == 0
    (line,) = lines_of(caplog)
    plan = plan_lines(caplog)
    assert plan[plan.index(line) + 1].startswith("plan: predicted peaks: ")
    assert plan[plan.index(line) - 1].startswith("plan: the VAE's room: ")
    # 25 frames of 128x96, at DESIGN's figure of a pixel, at most (the upper end).
    assert line.startswith("plan: disk: the masters at most ")
    assert line.endswith(" short")
    (warning,) = warnings_of(caplog)
    assert warning.startswith(f"{tmp_path / output}: its disk has ")
    assert caplog.messages.index(warning) > caplog.messages.index(plan[-1])
    # Once the room is there, the same command says it fits and warns of nothing.
    caplog.clear()
    monkeypatch.setattr(disk, "free_on", lambda directory: 500 * GIB)
    assert upscale(tmp_path, source_path, output, *options, "--plan", "plan.txt") == 0
    (line,) = lines_of(caplog)
    assert line.endswith("500 GiB free on its disk, enough")
    assert warnings_of(caplog) == []


def test_a_run_is_warned_never_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The disk may be cleared as the job runs (DESIGN.md, decided on 2026-10-28): a run whose
    # disk reads short says it and goes on, status 0, every unit made.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    monkeypatch.setattr(disk, "free_on", lambda directory: 1000)
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert len(lines_of(caplog)) == 1 and len(warnings_of(caplog)) == 1
    assert steps.calls != []


def test_a_resume_says_what_it_has_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Stopped inside the second segment (the first finished): the resume measures the first
    # segment's bytes a frame and counts the frames left at that rate, the figure no more.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    steps.stop = "encode 4"
    stopped(tmp_path, source_path, "out", *JOB)
    caplog.clear()
    steps.stop = None
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    (line,) = lines_of(caplog)
    assert "of them written; the job still takes " in line and "at most" not in line
    assert line.startswith("plan: disk: the masters about ")


def test_a_segment_command_job_is_unknown_then_measured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    options = (*JOB, "--segment-cmd", "ffmpeg -i - -c:v ffv1 {out}")
    monkeypatch.setattr(disk, "free_on", lambda directory: 1000)
    assert upscale(tmp_path, source_path, "out", *options, "--plan", "plan.txt") == 0
    (line,) = lines_of(caplog)
    assert "is not known before one is written" in line and line.endswith("free on its disk")
    assert warnings_of(caplog) == []
    # A segment written, the next run measures it.
    caplog.clear()
    steps.stop = "encode 4"
    stopped(tmp_path, source_path, "out", *options)
    caplog.clear()
    steps.stop = None
    assert upscale(tmp_path, source_path, "out", *options) == 0
    (line,) = lines_of(caplog)
    assert "of them written" in line and "not known" not in line

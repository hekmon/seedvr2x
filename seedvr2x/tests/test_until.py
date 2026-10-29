"""Stopping by the clock (--until; DESIGN.md, Pause and resume, Stopping), the time estimates
--plan prints and the progress lines (DESIGN.md, Memory planner, Time estimates), through the
command line on the CPU, test_cli_run.py's stand-in model under a wall clock of the test's own.

The clock is stop.now, the one function the code reads the time through, and moves only by the
stand-in's steps: each encode, window or shot's decode takes STEP on it. The calibration is
replaced by a clock saying every kind of unit takes STEP, whatever its size (Minutes), so that
the expectations are written in whole steps, independently of the code under test: a unit's
estimate is STEP times the parts it is made of, and a stop falls where the clock plus
MARGIN times the estimate passes --until's time."""

import json
import logging
import math
import re
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from test_cli_run import (
    ASKED,
    ENTERED,
    JOB,
    STAND_IN_MODELS,
    UNINTERRUPTED,
    Steps,
    contents,
    decoded,
    job,
    stand_in,  # noqa: F401
    steps,  # noqa: F401
    upscale,
)

from seedvr2x import cli
from seedvr2x.runtime import calibration, run, timing
from seedvr2x.runtime import stop as stop_module
from seedvr2x.runtime.plan import Plan
from seedvr2x.runtime.shot import shot_layout
from seedvr2x.runtime.timing import Clock, Kind, Model, spoken

# A step of the stand-in, an encode, a window or a shot's decode, on the test's clock.
STEP = 60.0

# When the test's clock starts: a time of day whose minutes --until names.
START = datetime(2026, 10, 10, 22, 0).astimezone()

# What a unit's estimate is multiplied by before it is held against --until's time (DESIGN.md).
MARGIN = 1.15

# The units of JOB, by the call that begins each, and their estimates in steps: a segment's decode
# is one unit of one step for each shot it decodes (3 shots of 3, 1 and 21 frames, in segments of
# the first two and of the third).
UNITS = {
    "encode 0": 1,
    "window 0:0": 1,
    "encode 3": 1,
    "window 3:0": 1,
    "decode 0": 2,
    "encode 4": 1,
    "window 4:0": 1,
    "window 4:1": 1,
    "decode 4": 1,
}
# The tokens of a latent of the stand-in's frames: 96x128 padded to 128x128, 16 pixels a token.
PER_LATENT = 8 * 8

# The disk's rate for a one-file output's assembly, in bytes per second: slow enough for the
# assembly of these few kilobytes to take minutes, several steps.
RATE = 500.0

# The stand-in's shots (frames), and the segments they make under JOB, as indexes of the shots.
SHOTS = (3, 1, 21)
SEGMENTS = ((0, 1), (2,))
# The first frames of a shot's runs for each --quality level (job.RUN_OFFSETS).
OFFSETS = {"standard": (0,), "enhanced": (0, 2), "maximum": (0, 1, 2, 3)}
# The window cap of JOB, in latents.
WINDOW = 5


class Minutes(Clock):
    """A clock by which every encode, window and shot's decode takes STEP, whatever its size, and
    which no unit's time corrects: the stand-in's take milliseconds. `observed` lists what the run
    gave it."""

    def __init__(self) -> None:
        super().__init__(dict.fromkeys(timing.KINDS, Model(STEP, 0.0)))
        self.observed: list[tuple[Kind, int]] = []

    def observe(self, kind: Kind, size: int, seconds: float) -> None:
        self.observed.append((kind, size))


@pytest.fixture
def clocked(steps: Steps, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:  # noqa: F811
    """The stand-in's run on the test's clock: the time is START plus `elapsed`, which STEP is
    added to by each step the stand-in makes. `begins` lists each step's name with the time it
    began at; `calibrated` each calibration asked, by its target, plan and longest window, and
    `clocks` the Minutes each gave."""
    state = SimpleNamespace(
        elapsed=0.0,
        begins=[],
        calibrated=[],
        clocks=[],
        steps=steps,
        rate=RATE,
        # What a step takes on the clock, by its kind and size (frames for an encode and a
        # decode, tokens for a window): STEP for any, unless a test says otherwise.
        seconds=lambda kind, size: STEP,
        disk_rate=cli._disk_rate,  # pyright: ignore[reportPrivateUsage]
        clock=Minutes,
    )

    def now() -> datetime:
        return START + timedelta(seconds=state.elapsed)

    def took(begun: datetime, kind: Kind, size: int) -> None:
        state.begins.append((steps.calls[-1], begun))
        state.elapsed += state.seconds(kind, size)

    encode, windows, decode = run.encode_shot, run.sample_windows, run.decode_shot

    def encode_step(*arguments: Any, **options: Any) -> Any:
        begun = now()
        latent = encode(*arguments, **options)
        took(begun, "encode", arguments[2])
        return latent

    def window_steps(*arguments: Any, **options: Any) -> Any:
        sampled = iter(windows(*arguments, **options))
        layout: list[tuple[int, int]] = arguments[2]
        for start, end in layout[arguments[4] if len(arguments) > 4 else 0 :]:
            begun = now()
            try:
                output = next(sampled)
            except StopIteration:
                return
            took(begun, "window", (end - start) * PER_LATENT)
            yield output

    def decode_step(*arguments: Any, **options: Any) -> None:
        begun = now()
        decode(*arguments, **options)
        took(begun, "decode", arguments[2])

    def calibrate(
        models: object, target: tuple[int, int], plan: Plan, longest: int, **options: Any
    ) -> calibration.Calibrated:
        state.calibrated.append((target, plan, longest))
        state.clocks.append(state.clock())
        return calibration.Calibrated(state.clocks[-1], 0.0)

    state.now = now
    # Everything reads the wall clock through this one function (stop.now).
    monkeypatch.setattr(stop_module, "now", now)
    monkeypatch.setattr(run, "encode_shot", encode_step)
    monkeypatch.setattr(run, "sample_windows", window_steps)
    monkeypatch.setattr(run, "decode_shot", decode_step)
    monkeypatch.setattr(calibration, "calibrate", calibrate)
    monkeypatch.setattr(cli, "_disk_rate", lambda directory: state.rate)
    return state


def until(clocked: SimpleNamespace, minutes: int) -> tuple[str, str]:
    """--until, `minutes` minutes from the time it is on the test's clock."""
    return "--until", f"{clocked.now() + timedelta(minutes=minutes):%H:%M}"


def date(moment: datetime) -> str:
    """A time as the logs say --until's: a full date in its zone."""
    return f"{moment:%Y-%m-%d %H:%M %Z}"


def manifest(directory: Path) -> dict[str, Any]:
    return json.loads((directory / "manifest.json").read_text())


def progress(directory: Path) -> tuple[list[bool], list[int]]:
    """The segments finished and each shot's windows done, as the manifest records them."""
    content = manifest(directory)
    finished = [segment["finished"] for segment in content["segments"]]
    return finished, [shot["windows_done"] for shot in content["shots"]]


def shot_runs(quality: str) -> list[int]:
    """The frames of every run of the stand-in's shots at a --quality level, in the job's order:
    a run is fed its shot from its first frame, and a shot too short for it has none."""
    return [frames - offset for frames in SHOTS for offset in OFFSETS[quality] if offset < frames]


def steps_of(quality: str) -> tuple[int, int, int]:
    """The encodes, windows and decode parts of the stand-in's job at a level, a step each."""
    runs = shot_runs(quality)
    return len(runs), sum(len(shot_layout(frames, WINDOW)) for frames in runs), len(runs)


# What a unit's progress line says of the job's time left, whichever it is.
SAID = (" left, the job ending about ", "time left known once", "the job's last unit made")


def lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Each unit's progress line: the encodes', the windows' and each shot's decode's."""
    return [r.getMessage() for r in caplog.records if any(said in r.getMessage() for said in SAID)]


def test_stops_before_the_first_unit_that_would_end_late(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clocked: SimpleNamespace,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Five minutes from START (22:00 to 22:05): the units, a step each but a segment's decode,
    # one for each shot it decodes, begin at 0, 1, 2 and 3 steps; the segment's decode at 4,
    # whose 2 steps with 15% more end at 6.3. It is not begun, and the third shot's encode, which
    # 1.15 steps would fit after it, is not made in its place: units are never reordered.
    caplog.set_level(logging.INFO)
    made: Steps = clocked.steps
    source_path = job(tmp_path, monkeypatch)
    deadline = START + timedelta(minutes=5)
    assert (
        upscale(tmp_path, source_path, "out", *JOB, "--until", f"{deadline:%H:%M}")
        == cli.EX_TEMPFAIL
    )
    assert made.calls == UNINTERRUPTED[:4]
    assert (
        f"--until {deadline:%H:%M}: the run stops before the first unit that wouldn't finish by"
        f" {date(deadline)}"
    ) in caplog.text
    assert (
        f"stopped before segment 1/2's decode and write, which wouldn't finish by"
        f" {date(deadline)} (--until): its estimate 2 min 0 s; 0 of 2 segments finished, and"
        " the units of the next ones kept: the same command resumes"
    ) in caplog.text
    # Every unit begun fitted, by its estimate with 15% more; the one named would not have.
    assert [name for name, _ in clocked.begins] == UNINTERRUPTED[:4]
    for name, begun in clocked.begins:
        assert begun + timedelta(seconds=MARGIN * STEP * UNITS[name]) <= deadline
    assert clocked.now() + timedelta(seconds=MARGIN * STEP * UNITS["decode 0"]) > deadline
    # What was made is kept, and nothing after it began.
    assert clocked.elapsed == 4 * STEP
    assert progress(tmp_path / "out") == ([False, False], [1, 1, 0])
    assert [shot["encoded"] for shot in manifest(tmp_path / "out")["shots"]] == [True, True, False]


def test_night_after_night(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clocked: SimpleNamespace
) -> None:
    # The same command, run again with --until three minutes ahead of the clock each time (22:13,
    # then later): each run makes the units that fit and stops before the first that doesn't. None
    # is made twice, none out of order, and the job is the uninterrupted run's.
    made: Steps = clocked.steps
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "whole", *JOB) == 0
    made.calls.clear()
    runs: list[list[str]] = []
    statuses: list[int] = []
    times: list[datetime] = []
    while not statuses or statuses[-1] == cli.EX_TEMPFAIL:
        times.append(clocked.now() + timedelta(minutes=3))
        statuses.append(
            upscale(tmp_path, source_path, "out", *JOB, "--until", f"{times[-1]:%H:%M}")
        )
        runs.append(list(made.calls))
        made.calls.clear()
        assert len(runs) < 10
    assert times == sorted(set(times))  # moved later each time
    assert statuses == [75, 75, 75, 75, 0]
    assert runs == [
        ["encode 0", "window 0:0"],  # the next, 1.15 steps, would end at 3.15
        ["encode 3", "window 3:0"],  # then the first segment's decode, 2.3 steps from 4
        ["decode 0", "decode 3"],  # 2 steps of the 3, the next encode 1.15 more wouldn't fit
        ["encode 4", "window 4:0"],
        ["window 4:1", "decode 4"],
    ]
    # No unit made twice, none out of order.
    assert [call for each in runs for call in each] == UNINTERRUPTED
    out = tmp_path / "out"
    for name in ("seg_000000.mkv", "seg_000001.mkv"):
        assert decoded(out / name) == decoded(tmp_path / "whole" / name)
    assert contents(out / "checksums") == contents(tmp_path / "whole" / "checksums")
    assert progress(out) == ([True, True], [1, 1, 2])


def test_past_its_time_after_the_first_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clocked: SimpleNamespace,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # --until's time is set as the command starts, and held again once the first pass is done:
    # a pass that lasts past it leaves nothing to begin, and no model is loaded for nothing; the
    # new job's record is written, as --plan writes it, so that the same command takes up its
    # first pass (a review's finding: it was lost, the message saying the job resumes).
    made: Steps = clocked.steps
    source_path = job(tmp_path, monkeypatch)
    first_pass = cli._first_pass  # pyright: ignore[reportPrivateUsage]

    def slow(*arguments: Any) -> Any:
        clocked.elapsed += 120.0
        return first_pass(*arguments)

    monkeypatch.setattr(cli, "_first_pass", slow)
    deadline = START + timedelta(minutes=1)
    assert (
        upscale(tmp_path, source_path, "out", *JOB, "--until", f"{deadline:%H:%M}")
        == cli.EX_TEMPFAIL
    )
    assert f"it is past {date(deadline)} (--until) already: nothing begun" in caplog.text
    assert "models" not in ASKED
    assert made.calls == [] and clocked.calibrated == []
    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == [
        "frame_index.bin",
        "manifest.json",
    ]
    assert progress(tmp_path / "out") == ([False] * 2, [0] * 3)


def test_plan_says_whether_the_job_ends_by_then(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clocked: SimpleNamespace,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # --plan loads the models and calibrates, and says the job's time, a step each unit: 10 steps
    # in all, 3 encodes, 4 windows and 3 decodes in 2 segments, the first segment's decode, of 2
    # shots, the longest unit. Given --until it says whether the job, run now, ends by then.
    caplog.set_level(logging.INFO)
    made: Steps = clocked.steps
    source_path = job(tmp_path, monkeypatch)
    far = START + timedelta(minutes=30)
    options = (*JOB, "--plan", "plan.txt")
    assert upscale(tmp_path, source_path, "out", *options, "--until", f"{far:%H:%M}") == 0
    assert f"--until {far:%H:%M}: --plan says whether the job, run now, ends by" in caplog.text
    assert f"--until: by its estimate the job ends by {date(far)}, at about 22:10" in caplog.text
    assert made.calls == [] and "models" in ASKED
    assert clocked.calibrated == [((96, 128), Plan(WINDOW), 4)]
    assert (
        "time: about 10 min 0 s on this GPU as it runs now: 3 min 0 s for 3 encodes, 4 min 0 s"
        " for 4 windows, 3 min 0 s for the decode and write of 2 segments"
    ) in caplog.text
    assert (
        "time: the longest unit, segment 1/2's decode and write, about 2 min 0 s: --until may"
        " stop up to about 1.15 times that before its time"
    ) in caplog.text
    levels = {level: sum(steps_of(level)) for level in OFFSETS}
    assert levels == {"standard": 10, "enhanced": 17, "maximum": 28}
    assert (
        "time at each --quality: standard about 10 min 0 s (this job's), enhanced about"
        " 17 min 0 s, maximum about 28 min 0 s"
    ) in caplog.text
    # A new job's plan records the models, for information.
    assert set(manifest(tmp_path / "out")["plan"]["timings"]) == {"encode", "window", "decode"}
    assert progress(tmp_path / "out") == ([False, False], [0, 0, 0])
    # A deadline the job doesn't reach: the first unit that wouldn't end by it, and how far the
    # job gets (the segment's decode, begun at 4 steps, would end at 6.3).
    caplog.clear()
    near = START + timedelta(minutes=5)
    assert upscale(tmp_path, source_path, "out", *options, "--until", f"{near:%H:%M}") == 0
    assert (
        f"--until: by its estimate the job doesn't end by {date(near)}: it would stop before"
        " segment 1/2's decode and write, at about 22:04, 4 min 0 s of its 10 min 0 s done; the"
        " same command then resumes"
    ) in caplog.text
    assert made.calls == []


def test_plan_of_maximum_counts_the_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clocked: SimpleNamespace,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # --quality maximum runs each shot 4 times, from its frames 0 to 3 (a shot too short for one
    # has fewer): 8 encodes, 12 windows and 8 decodes, whose steps are the estimate's.
    caplog.set_level(logging.INFO)
    source_path = job(tmp_path, monkeypatch)
    assert shot_runs("maximum") == [3, 2, 1, 1, 21, 20, 19, 18]
    encodes, windows, decodes = steps_of("maximum")
    assert (encodes, windows, decodes) == (8, 12, 8)
    options = (*JOB, "--quality", "maximum", "--plan", "plan.txt")
    assert upscale(tmp_path, source_path, "out", *options) == 0
    assert (
        f"time: about {spoken(STEP * 28)} on this GPU as it runs now: {spoken(STEP * encodes)}"
        f" for {encodes} encodes, {spoken(STEP * windows)} for {windows} windows,"
        f" {spoken(STEP * decodes)} for the decode and write of 2 segments"
    ) in caplog.text
    assert "time: about 28 min 0 s on this GPU" in caplog.text
    # The first segment's decode, of 4 runs, ties the second's: the first is named.
    assert "time: the longest unit, segment 1/2's decode and write, about 4 min 0 s:" in caplog.text
    assert "maximum about 28 min 0 s (this job's)" in caplog.text
    # The calibration is of the runs' windows: the longest, 4 latents.
    assert clocked.calibrated == [((96, 128), Plan(WINDOW), 4)]


def test_the_deadline_passing_during_the_hash_check_stops_the_segments_made_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clocked: SimpleNamespace,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Review of 2026-10-28: a segment found damaged by the assembly's hash check is made again
    # under the same --until: here the check ends past it (the clock moved), so the run stops
    # before the first unit made again, exit 75, nothing made, the segment recorded unfinished and
    # the job kept; the next start makes it again and assembles.
    from seedvr2x.runtime.units import Units

    caplog.set_level(logging.INFO)
    made: Steps = clocked.steps
    source_path = job(tmp_path, monkeypatch)
    work = tmp_path / "out.mkv.work"
    finished, checked = Units.segment_finished, cli._damaged  # pyright: ignore[reportPrivateUsage]

    def damaging(self: Units, segment: int) -> None:
        finished(self, segment)
        if segment == 1:  # the last: the first one's bytes change
            [first] = work.glob("seg_000000.*")
            data = bytearray(first.read_bytes())
            data[len(data) // 2] ^= 0xFF
            first.write_bytes(bytes(data))

    def late(*arguments: Any) -> Any:
        clocked.elapsed += 3 * 3600  # the hash check takes hours: --until's time is past
        return checked(*arguments)

    monkeypatch.setattr(Units, "segment_finished", damaging)
    monkeypatch.setattr(cli, "_damaged", late)
    assert upscale(tmp_path, source_path, "out.mkv", *JOB, *until(clocked, 120)) == cli.EX_TEMPFAIL
    assert made.calls == UNINTERRUPTED and not (tmp_path / "out.mkv").exists()
    assert "(--until)" in caplog.text and "made again now" in caplog.text
    assert progress(work)[0] == [False, True]
    monkeypatch.setattr(cli, "_damaged", checked)
    made.calls.clear()
    assert upscale(tmp_path, source_path, "out.mkv", *JOB) == 0
    assert made.calls == UNINTERRUPTED[:6] and (tmp_path / "out.mkv").exists()


def test_assembly_is_the_job_s_last_unit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clocked: SimpleNamespace,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A one-file output: the segments' 10 steps end at 22:10, within an --until of 22:11, and the
    # assembly, which its masters' bytes and the source's at the disk's rate make minutes,
    # wouldn't: the run stops before it, every segment kept. The same command given a later
    # --until assembles.
    caplog.set_level(logging.INFO)
    made: Steps = clocked.steps
    source_path = job(tmp_path, monkeypatch)
    estimates: list[tuple[int, float]] = []
    estimate = cli._assembly_estimate  # pyright: ignore[reportPrivateUsage]

    def spy(record: Any, rate: float) -> float:
        seconds = estimate(record, rate)
        estimates.append((len(caplog.records), seconds))
        return seconds

    monkeypatch.setattr(cli, "_assembly_estimate", spy)
    deadline = START + timedelta(minutes=11)
    assert (
        upscale(tmp_path, source_path, "out.mkv", *JOB, "--until", f"{deadline:%H:%M}")
        == cli.EX_TEMPFAIL
    )
    assert made.calls == UNINTERRUPTED
    work = tmp_path / "out.mkv.work"
    assert progress(work)[0] == [True, True]
    assert not (tmp_path / "out.mkv").exists()
    # Its estimate: the masters' bytes four times and the source's twice, at the disk's rate.
    masters = sum(segment["bytes"] for segment in manifest(work)["segments"])
    cost = (4 * masters + 2 * source_path.stat().st_size) / RATE
    assert cost * MARGIN > 60  # more than the minute the segments left
    assert (
        f"stopped before the assembly, which wouldn't finish by {date(deadline)} (--until):"
        f" its estimate {spoken(cost)}; every segment finished"
    ) in caplog.text
    # Each unit's time left is its units' steps left and then the assembly, its masters at the
    # size they were counted by then (_assembly_estimate), the end rounded up to the minute.
    records = [r for r in caplog.records if " left, the job ending about " in r.getMessage()]
    assert len(records) == len(UNINTERRUPTED)
    for count, record in enumerate(records, 1):
        position = caplog.records.index(record)
        extra = [seconds for index, seconds in estimates if index <= position][-1]
        left = STEP * (len(UNINTERRUPTED) - count) + extra
        finish = START + timedelta(seconds=60 * math.ceil((STEP * count + left) / 60))
        assert f"about {spoken(left)} left, the job ending about {finish:%H:%M}" in (
            record.getMessage()
        )
    # Later: the segments are kept, so no model is loaded and nothing is made but the assembly.
    caplog.clear()
    made.calls.clear()
    ASKED.clear()
    later = clocked.now() + timedelta(minutes=60)
    assert upscale(tmp_path, source_path, "out.mkv", *JOB, "--until", f"{later:%H:%M}") == 0
    # The masters' hash check, which reads them once more, is said first with its time, from the
    # rate the estimate uses (cli._checking_said; review of the assembly, 2026-10-28).
    assert (
        f"checking the bytes of 2 segments, {masters / 1024**3:.2f} GiB,"
        f" about {spoken(masters / RATE)}"
    ) in caplog.messages
    assert made.calls == [] and "models" not in ASKED
    assert (tmp_path / "out.mkv").exists() and not work.exists()


def test_progress_lines_with_a_calibration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clocked: SimpleNamespace,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given --until the run is calibrated, and each unit's line says the job's time left, a step
    # for each part not yet made, and the time of day it should end.
    caplog.set_level(logging.INFO)
    source_path = job(tmp_path, monkeypatch)
    far = START + timedelta(minutes=60)
    assert upscale(tmp_path, source_path, "out", *JOB, "--until", f"{far:%H:%M}") == 0
    said = lines(caplog)
    assert len(said) == len(UNINTERRUPTED)
    assert said[0].endswith("about 9 min 0 s left, the job ending about 22:10")
    for made, line in enumerate(said[:-1], 1):
        left = STEP * (len(UNINTERRUPTED) - made)
        finish = START + timedelta(seconds=STEP * made + left)
        assert line.endswith(f"about {spoken(left)} left, the job ending about {finish:%H:%M}"), (
            line
        )
    assert said[-1].endswith("the job's last unit made")
    assert len(clocked.calibrated) == 1


def test_every_run_calibrates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clocked: SimpleNamespace,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A run given no --until is calibrated too (DESIGN.md, Memory planner, Time estimates, decided
    # on 2026-10-21): once, and its first unit's line already gives the job's time left.
    caplog.set_level(logging.INFO)
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert len(clocked.calibrated) == 1
    said = lines(caplog)
    assert len(said) == len(UNINTERRUPTED)
    assert said[0].endswith("about 9 min 0 s left, the job ending about 22:10")
    assert all("time left known once" not in line for line in said)
    assert said[-1].endswith("the job's last unit made")


def made_of(clock: Minutes) -> list[tuple[Kind, int]]:
    """The units' parts a Minutes clock was given, those of the three kinds: its other terms, the
    changes of phase (run._enter), are no part of the sizes asserted here."""
    return [each for each in clock.observed if each[0] in timing.KINDS]


def test_a_resumed_run_s_first_seek_is_not_counted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clocked: SimpleNamespace,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Stopped before the third shot's encode, the first segment done: the resumed process opens
    # its reader at that shot's frame 4 through the frame index, a seek the encode's time is
    # held against and the clock isn't given. The windows and the decode that follow are.
    caplog.set_level(logging.DEBUG)
    made: Steps = clocked.steps
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB, *until(clocked, 7)) == cli.EX_TEMPFAIL
    assert made.calls == UNINTERRUPTED[:6]
    assert "not counted" not in caplog.text
    first = clocked.clocks[-1]
    # The first run's own encodes: the first at shot 1's frame 0 opens no seek, the second's
    # reader is at the end of the first's.
    assert [kind for kind, _ in made_of(first)].count("encode") == 2
    made.calls.clear()
    assert upscale(tmp_path, source_path, "out", *JOB, *until(clocked, 60)) == 0
    assert made.calls == UNINTERRUPTED[6:]
    resumed = clocked.clocks[-1]
    assert resumed is not first
    # 21 frames in windows of 4 latents (96x128 is padded to 128x128, 64 tokens a latent).
    assert made_of(resumed) == [("window", 4 * 64), ("window", 4 * 64), ("decode", 21)]
    assert caplog.text.count("not counted: its reader sought its first frame") == 1
    assert "shot 3/3's encode:" in caplog.text.split("not counted")[0].splitlines()[-1]


def test_the_encode_after_a_resumed_run_s_first_is_counted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clocked: SimpleNamespace,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Stopped after the first shot's units, the resumed process's first encode is the second
    # shot's (frame 3), a seek: not counted. The third shot's comes with the reader at its first
    # frame, no seek: counted, as the units around it.
    caplog.set_level(logging.DEBUG)
    made: Steps = clocked.steps
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB, *until(clocked, 3)) == cli.EX_TEMPFAIL
    assert made.calls == UNINTERRUPTED[:2]
    assert upscale(tmp_path, source_path, "out", *JOB, *until(clocked, 60)) == 0
    observed = made_of(clocked.clocks[-1])
    assert ("encode", 1) not in observed
    assert observed == [
        *(("window", 64), ("decode", 3), ("decode", 1)),
        *(("encode", 21), ("window", 4 * 64), ("window", 4 * 64), ("decode", 21)),
    ]
    assert caplog.text.count("not counted") == 1


def test_until_is_no_part_of_the_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clocked: SimpleNamespace
) -> None:
    # A job stopped by one --until resumes under another, or none: the time to stop at is no
    # setting of the job, and the manifest records none.
    made: Steps = clocked.steps
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB, *until(clocked, 3)) == cli.EX_TEMPFAIL
    assert "until" not in json.dumps(manifest(tmp_path / "out")["settings"])
    made.calls.clear()
    assert upscale(tmp_path, source_path, "out", *JOB, *until(clocked, 4)) == cli.EX_TEMPFAIL
    assert made.calls == UNINTERRUPTED[2:4]
    made.calls.clear()
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert made.calls == UNINTERRUPTED[4:]
    assert progress(tmp_path / "out") == ([True, True], [1, 1, 2])


# calibration.calibrate itself, before the stand-in replaces it (test_cli_run.stand_in_model).
CALIBRATE = calibration.calibrate


def test_plan_runs_the_calibration_through_the_job_s_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The calibration itself, through the command line: --plan runs its steps on the stand-in,
    # the run's own steps (calibration.calibrate takes run.encode_shot, run.sample_windows and
    # run.decode_shot): the encodes' warm-up and two lengths, each after a window of 2 latents as
    # a run's encodes follow one; the windows' warm-up, 2 latents and the longest (4 here); the
    # decodes' warm-up and two lengths; none of them a unit of the job, which --plan leaves as a
    # plan, nothing made.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    monkeypatch.setattr(calibration, "calibrate", CALIBRATE)
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB, "--plan", "plan.txt") == 0
    # Shot "-42": the calibration's seed is 0, which the stand-in names by its offset from SEED.
    assert steps.calls == [
        *("encode -42", "window -42:0", "encode -42", "window -42:0", "encode -42"),
        *("window -42:0", "window -42:0", "window -42:0"),
        *("decode -42", "decode -42", "decode -42"),
    ]
    # The encode's second length capped by the source's frames left after the first two encodes,
    # 25 - 13 - 5 = 7 (SHOTS), under the job's longest run, 21 frames, and under the thousands of
    # frames 3 s of a stand-in's millisecond encode would be: the floor, 13 frames.
    assert any("encodes of 5 and 13 frames" in line for line in caplog.messages)
    assert any(line.startswith("time estimates calibrated in ") for line in caplog.messages)
    assert any(line.startswith("time: about ") for line in caplog.messages)
    assert progress(tmp_path / "out") == ([False] * 2, [0] * 3)


def units_begun(begins: list[tuple[str, datetime]], parts: int) -> list[tuple[str, datetime, int]]:
    """The units a list of steps' begins holds, with their size in steps: an encode or a window
    is one step, and a segment's decode as many as its consecutive decodes (`parts`: the runs it
    decodes, as the job's runs make them)."""
    units: list[tuple[str, datetime, int]] = []
    decoding = False
    for name, begun in begins:
        if name.startswith("decode"):
            if not decoding:
                units.append((name, begun, parts))
            decoding = True
        else:
            units.append((name, begun, 1))
            decoding = False
    return units


def test_maximum_night_after_night(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clocked: SimpleNamespace
) -> None:
    # --quality maximum: 8 runs, whose segments' decodes are 4 steps each (the first segment's 3
    # + 1 runs, the second's 4). Run again with --until 5 minutes ahead, each night makes the
    # units that fit by their estimate and 15% more, and the nights' steps are the uninterrupted
    # run's, none made twice, its segments' frames the same.
    made: Steps = clocked.steps
    source_path = job(tmp_path, monkeypatch)
    options = (*JOB, "--quality", "maximum")
    assert upscale(tmp_path, source_path, "whole", *options) == 0
    uninterrupted = list(made.calls)
    assert len(uninterrupted) == sum(steps_of("maximum"))
    made.calls.clear()
    clocked.begins.clear()
    runs: list[list[str]] = []
    statuses: list[int] = []
    deadlines: list[datetime] = []
    marks: list[int] = []
    while not statuses or statuses[-1] == cli.EX_TEMPFAIL:
        deadlines.append(clocked.now() + timedelta(minutes=5))
        marks.append(len(clocked.begins))
        statuses.append(
            upscale(tmp_path, source_path, "out", *options, "--until", f"{deadlines[-1]:%H:%M}")
        )
        runs.append(list(made.calls))
        made.calls.clear()
        assert len(runs) < 20
    assert statuses == [75] * (len(runs) - 1) + [0] and len(runs) > 2
    assert [call for each in runs for call in each] == uninterrupted
    # Every unit begun obeyed the rule of the night it was begun in.
    marks.append(len(clocked.begins))
    for deadline, first, last in zip(deadlines, marks, marks[1:], strict=False):
        for name, begun, parts in units_begun(clocked.begins[first:last], 4):
            assert begun + timedelta(seconds=MARGIN * STEP * parts) <= deadline, name
    out = tmp_path / "out"
    for name in ("seg_000000.mkv", "seg_000001.mkv"):
        assert decoded(out / name) == decoded(tmp_path / "whole" / name)
    assert contents(out / "checksums") == contents(tmp_path / "whole" / "checksums")


# The models of a clock whose estimates are by the size, in seconds: a constant and a rate by
# frame, padded to 4n + 1, or by token.
SIZED: dict[Kind, Model] = {
    "encode": Model(2.0, 0.5),
    "window": Model(1.0, 0.01),
    "decode": Model(3.0, 0.25),
}
# How much longer than its model the steps of a drifting card take.
DRIFT = 1.3


class Sized(Clock):
    """A calibrated clock, by SIZED's models, which corrects them by the units it is given as a
    run's clock does."""

    def __init__(self) -> None:
        super().__init__(SIZED)


def test_estimates_follow_a_drifting_card(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clocked: SimpleNamespace,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The steps take 1.3 times their model, by the size. A kind's first unit is estimated by the
    # model as it was calibrated; from its fourth, the median of the last ratios, 1 (the
    # calibration's) and three of 1.3, is 1.3 and the estimates are the steps' times.
    from seedvr2x.runtime.shot import padded_length

    caplog.set_level(logging.DEBUG, logger="seedvr2x.runtime.run")
    source_path = job(tmp_path, monkeypatch)

    def seconds(kind: Kind, size: int) -> float:
        size = size if kind == "window" else padded_length(size)
        return DRIFT * SIZED[kind].at(size)

    clocked.seconds = seconds
    clocked.clock = Sized
    # The run's own measure of a unit is the time on the test's clock.
    monkeypatch.setattr(run, "time", SimpleNamespace(monotonic=lambda: clocked.elapsed))
    far = until(clocked, 120)
    assert upscale(tmp_path, source_path, "out", *JOB, "--quality", "maximum", *far) == 0
    said: dict[Kind, list[tuple[float, float]]] = {"encode": [], "window": [], "decode": []}
    for record in caplog.records:
        found = re.fullmatch(
            r".*'s (encode|window|decode)\b[^:]*: ([\d.]+) s, estimated at ([\d.]+) s"
            r"; with its terms ([\d.]+) s, estimated at ([\d.]+) s",
            record.getMessage(),
        )
        if record.name == "seedvr2x.runtime.run" and found:
            said[cast(Kind, found[1])].append((float(found[2]), float(found[3])))
            # The stand-in's phases cost nothing and Sized estimates no term: the unit's, as is.
            assert (found[4], found[5]) == (found[2], found[3])
    assert {kind: len(each) for kind, each in said.items()} == {
        "encode": 8,
        "window": 12,
        "decode": 8,
    }
    # The first of each kind, uncorrected: the shot's first run, 3 frames (padded to 5), a window
    # of 2 latents.
    first: dict[Kind, int] = {"encode": 5, "window": 2 * PER_LATENT, "decode": 5}
    for kind, each in said.items():
        assert each[0][1] == pytest.approx(SIZED[kind].at(first[kind]), abs=1e-3)
        assert each[0][0] == pytest.approx(DRIFT * SIZED[kind].at(first[kind]), abs=1e-3)
        for took, estimated in each[3:]:
            assert estimated == pytest.approx(took, rel=0.02)


def test_plan_of_the_assembly_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clocked: SimpleNamespace,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A one-file job stopped before its assembly, every segment made: --plan says that the
    # assembly alone is left, how long it takes, and given --until whether it ends by then, with
    # no model loaded and nothing calibrated.
    caplog.set_level(logging.INFO)
    source_path = job(tmp_path, monkeypatch)
    deadline = START + timedelta(minutes=11)
    assert (
        upscale(tmp_path, source_path, "out.mkv", *JOB, "--until", f"{deadline:%H:%M}")
        == cli.EX_TEMPFAIL
    )
    work = tmp_path / "out.mkv.work"
    masters = sum(segment["bytes"] for segment in manifest(work)["segments"])
    cost = (4 * masters + 2 * source_path.stat().st_size) / RATE
    ASKED.clear()
    clocked.calibrated.clear()
    clocked.steps.calls.clear()
    caplog.clear()
    options = (*JOB, "--plan", "plan.txt")
    # Time enough: the assembly ends 4 min 28 s after the clock's 22:10, at 22:14:28.
    far = clocked.now() + timedelta(minutes=60)
    assert upscale(tmp_path, source_path, "out.mkv", *options, "--until", f"{far:%H:%M}") == 0
    assert (
        f"time: the assembly alone is left, about {spoken(cost)} at the rate its disk writes at"
    ) in caplog.text
    assert f"--until: by its estimate the job ends by {date(far)}, at about 22:14" in caplog.text
    # Not enough: begun now, it would end at 22:15:08 with the 15% more, after 22:13.
    caplog.clear()
    near = clocked.now() + timedelta(minutes=3)
    assert upscale(tmp_path, source_path, "out.mkv", *options, "--until", f"{near:%H:%M}") == 0
    assert (
        f"--until: by its estimate the job doesn't end by {date(near)}: it would stop before"
        f" the assembly of {tmp_path / 'out.mkv'}, at about 22:10, 0 s of its {spoken(cost)}"
        " done; the same command then resumes"
    ) in caplog.text
    assert "models" not in ASKED and clocked.calibrated == []
    assert clocked.steps.calls == [] and not (tmp_path / "out.mkv").exists()
    assert progress(work)[0] == [True, True]


def test_a_disk_rate_not_measured_does_not_stop_the_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clocked: SimpleNamespace,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The probe can't write beside the output: the assembly isn't estimated, said once as a
    # warning, and --until holds it to its time alone. The job runs and assembles.
    def refused(directory: Path) -> float:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(cli, "_disk_rate", clocked.disk_rate)
    monkeypatch.setattr(timing, "disk_rate", refused)
    source_path = job(tmp_path, monkeypatch)
    far = until(clocked, 120)
    assert upscale(tmp_path, source_path, "out.mkv", *JOB, *far) == 0
    assert caplog.text.count("its disk's rate not measured (") == 1
    assert "No space left on device" in caplog.text
    assert "the assembly isn't estimated, and --until holds it to its time alone" in caplog.text
    assert (tmp_path / "out.mkv").exists() and not (tmp_path / "out.mkv.work").exists()
    assert clocked.steps.calls == UNINTERRUPTED


# What the terms of a unit take, on a clock whose phases and noise cost: a change of phase, by the
# phase entered, in seconds, and the noise's draw per token of a run's latent.
ENTERING = {"encode": 5.0, "dit": 7.0, "decode": 3.0}
DRAW = 0.01


class Termed(Clock):
    """A calibrated clock by which every encode, window and shot's decode takes STEP, as Minutes',
    and the terms a unit pays besides take ENTERING's and DRAW's: the changes of phase into the
    DiT's and the decode's and a run's noise; an encode's change, which is in its own model, takes
    ENTERING's too. Nothing corrects it; `observed` lists what the run gave it."""

    def __init__(self) -> None:
        models = dict.fromkeys(timing.KINDS, Model(STEP, 0.0))
        # An encode's change of phase is in its own model: a run's encodes always follow one
        # (timing.ENTER), so that every encode takes its estimate on the clock.
        models["encode"] = Model(STEP + ENTERING["encode"], 0.0)
        models["enter-dit"] = Model(ENTERING["dit"], 0.0)
        models["enter-decode"] = Model(ENTERING["decode"], 0.0)
        models["draw"] = Model(0.0, DRAW)
        super().__init__(models)
        self.observed: list[tuple[Kind, int]] = []
        self.seconds: list[tuple[Kind, float]] = []

    def observe(self, kind: Kind, size: int, seconds: float) -> None:
        self.observed.append((kind, size))
        self.seconds.append((kind, seconds))


@pytest.fixture
def termed(monkeypatch: pytest.MonkeyPatch, clocked: SimpleNamespace) -> SimpleNamespace:
    """`clocked` with a calibration that estimates the terms (Termed), and a stand-in whose
    changes of phase (Models.enter) and first window of a run (its noise's draw) take them on the
    test's clock. The models start in the decode's phase, as the calibration leaves them
    (cli: units_of's start)."""
    phase = ["decode"]

    def enter(entered: str) -> None:
        ENTERED.append(entered)
        if entered != phase[0]:
            clocked.elapsed += ENTERING[entered]
        phase[0] = entered

    sampled = run.sample_windows

    def drawing(*arguments: Any, **options: Any) -> Any:
        layout: list[tuple[int, int]] = arguments[2]
        windows = iter(sampled(*arguments, **options))
        # The run's noise, drawn for all its windows with the first (shot.sample_windows).
        clocked.elapsed += DRAW * layout[-1][1] * PER_LATENT
        yield from windows

    clocked.clock = Termed
    monkeypatch.setattr(STAND_IN_MODELS, "enter", enter)
    monkeypatch.setattr(run, "sample_windows", drawing)
    # The run's own measure of a unit is the time on the test's clock.
    monkeypatch.setattr(run, "time", SimpleNamespace(monotonic=lambda: clocked.elapsed))
    return clocked


def test_a_unit_s_terms_are_in_the_time_left(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    termed: SimpleNamespace,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # JOB's calls, each with the terms it opens: a change of phase (7 or 3 s) into the DiT's or
    # the decode's where the unit before is of another phase, the first against the decode's; the
    # encode's 5 s are in its model, no term (every encode follows a window or a decode); and a
    # run's first window, its noise by the tokens of the run's latents (2, 1 and 6 latents of 64
    # tokens). A line's time left is the cost of the calls after it, their terms included, and
    # the last unit's says so.
    caplog.set_level(logging.INFO)
    source_path = job(tmp_path, monkeypatch)
    draws = [DRAW * layout[-1][1] * PER_LATENT for layout in map(shot_layout_of, SHOTS)]
    assert draws == pytest.approx([1.28, 0.64, 3.84])
    costs = [
        STEP + ENTERING["encode"],  # encode 0
        STEP + ENTERING["dit"] + draws[0],  # window 0:0
        STEP + ENTERING["encode"],  # encode 3
        STEP + ENTERING["dit"] + draws[1],  # window 3:0
        STEP + ENTERING["decode"],  # decode 0, the segment's decode
        STEP,  # decode 3, its second part
        STEP + ENTERING["encode"],  # encode 4
        STEP + ENTERING["dit"] + draws[2],  # window 4:0
        STEP,  # window 4:1, in the phase already
        STEP + ENTERING["decode"],  # decode 4
    ]
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    said = lines(caplog)
    assert len(said) == len(costs)
    for count, line in enumerate(said[:-1], 1):
        # Every unit takes its estimate on the clock, so the job ends when all the costs are
        # paid: at 647.76 s, 22:11 to the minute.
        assert line.endswith(
            f"about {spoken(sum(costs[count:]))} left, the job ending about 22:11"
        ), line
    assert said[-1].endswith("the job's last unit made")
    assert termed.elapsed == pytest.approx(sum(costs))
    # The terms left out of the kinds', which the clock was given as its own: sizes as before.
    clock: Termed = termed.clocks[-1]
    assert [each for each in clock.observed if each[0] in timing.KINDS] == [
        ("encode", 3),
        ("window", 128),
        ("encode", 1),
        ("window", 64),
        ("decode", 3),
        ("decode", 1),
        ("encode", 21),
        ("window", 256),
        ("window", 256),
        ("decode", 21),
    ]
    assert [each for each in clock.observed if each[0] == "draw"] == []
    # The times the run gave the clock: an encode's with its change of phase, which its model
    # holds (a review's finding: the run starting an encode's time after its change passed every
    # other test); a window's and a decode's without theirs, given apart.
    given = {
        kind: [seconds for each, seconds in clock.seconds if each == kind]
        for kind in (*timing.KINDS, "enter-dit")
    }
    assert given["encode"] == pytest.approx([STEP + ENTERING["encode"]] * 3)
    assert given["window"] == pytest.approx([STEP] * 4)
    assert given["decode"] == pytest.approx([STEP] * 3)
    assert given["enter-dit"] == pytest.approx([ENTERING["dit"]] * 3)


def shot_layout_of(frames: int) -> list[tuple[int, int]]:
    return shot_layout(frames, WINDOW)


def test_the_terms_count_in_until_s_rule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    termed: SimpleNamespace,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The clock starts at 22:00:40 and --until is 22:03. The first window begins at 22:01:45, after
    # the encode's 65 s: its 60 s with 15% more end at 22:02:54, in time; with its change of phase
    # (7 s) and its noise's draw (1.28 s), 68.28 s and 15% more, at 22:03:03.5, not. The run stops
    # before it: the terms are part of the unit's estimate.
    caplog.set_level(logging.INFO)
    made: Steps = termed.steps
    termed.elapsed = 40.0
    source_path = job(tmp_path, monkeypatch)
    deadline = START + timedelta(minutes=3)
    assert (
        upscale(tmp_path, source_path, "out", *JOB, "--until", f"{deadline:%H:%M}")
        == cli.EX_TEMPFAIL
    )
    assert made.calls == ["encode 0"]
    begun = termed.now()
    assert begun + timedelta(seconds=MARGIN * STEP) <= deadline  # the kind alone would fit
    assert begun + timedelta(seconds=MARGIN * (STEP + ENTERING["dit"] + 1.28)) > deadline
    assert (
        f"stopped before shot 1/3's window 1/1, which wouldn't finish by {date(deadline)}"
        " (--until): its estimate 1 min 8 s"
    ) in caplog.text

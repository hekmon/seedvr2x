"""The command line's several runs of a shot (--quality; DESIGN.md, Several runs of a shot,
an option for quality) on the CPU, the model replaced by test_cli_run.py's stand-in: which run
each frame of the output is taken from, the composite equal to single runs of the source trimmed
to each run's first frame, the manifest's record of the runs, a stop in a later run and its
resume, the plan's line, the option refused with a count seedvr2x doesn't offer, and the frames
the runs keep for the decode removed once it is done.

The stand-in's decode writes, for each frame of a run, its shot's first frame plus the frame's
index in the run, over 1000 (test_cli_run.stand_in_decode_stream): so the output frame f of a shot
starting at `start`, taken from the run that starts at frame s of the shot, reads start + f - s.
The rule itself, job.run_of, is held to measurement's composer by test_runs.py; the closed forms
here are the rule written out again, the expected values independent of the code under test."""

import hashlib
import json
import logging
import shutil
import subprocess
from itertools import accumulate, chain
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from test_cli_run import (
    JOB,
    Steps,
    contents,
    decoded,
    first_passes,
    indexes,
    job,
    plan_lines,
    refused,
    source,
    stand_in,  # noqa: F401
    steps,  # noqa: F401
    stopped,
    upscale,
)

from seedvr2x.runtime import job as runtime_job
from seedvr2x.runtime import model, run
from seedvr2x.runtime.shot import padded_length
from seedvr2x.runtime.units import KEPT

# The output's frames as test_cli_run's stand-in gives them (gbrp16le, 128x96), in bytes.
FRAME_BYTES = 3 * 96 * 128 * 2
# The same as the runs keep them for the decode (run.Kept): (H, W, 3) at 16 bits.
KEPT_BYTES = 96 * 128 * 3 * 2
# The source's frames, yuv420p at 64x48.
SOURCE_BYTES = 64 * 48 * 3 // 2

# The plan's line on the --quality level of several runs a shot (planner._quality_said): the
# runs a shot, and the count in all, and the other levels.
RUNS_SAID = (
    "plan: quality: {level}, {runs} runs a shot, {total} in all, each frame from its best run:"
    " about {runs} times standard's GPU time, one run's memory; {kept} GiB of frames on disk for"
    " the longest shot; --quality {others}"
)
# What it says at the default level (the line is there at every level).
STANDARD_SAID = (
    "plan: quality: standard, the default, one run a shot; --quality enhanced or maximum runs 2"
    " or 4 and takes each frame from the one that rebuilds it best, for 2 or 4 times the GPU time"
)


def quality_line(lines: list[str]) -> str:
    """The one line of the plan's lines (plan_lines) that says the --quality level."""
    found = [line for line in lines if line.startswith("plan: quality: ")]
    assert len(found) == 1, lines
    return found[0]


# The --quality levels by their runs a shot as the tests name them (job.QUALITY).
LEVEL = {str(runs): level for level, runs in runtime_job.QUALITY.items()}


def taken(runs: int, frame: int) -> int:
    """The first frame, in the shot, of the run a shot's frame `frame` is taken from with `runs`
    runs a shot (DESIGN.md, Several runs of a shot): written out here, not job.run_of. With 4, the
    run where the frame ends its group of 4 or is the run's first, frame mod 4. With 2, runs 0 and
    2: frame 0 and 1 from run 0 (run 2 doesn't hold them), then the run where the frame's place is
    3 or 2 rather than 0 or 1 (a first frame, place -1, above all): frame 2 from run 2, frame 3
    from run 0 (place 2 above run 2's 0), 4 from run 0 (3), 5 from run 2 (2 above 0), 6 from
    run 2 (3), 7 from run 0, and so on every 4 frames."""
    if runs == 1:
        return 0
    if runs == 4:
        return frame % 4
    return 0 if frame == 1 or frame % 4 in (0, 3) else 2


def frames_of(path: Path) -> list[str]:
    """The MD5 of each frame of a master as stored (gbrp16le)."""
    data = decoded(path)
    assert len(data) % FRAME_BYTES == 0
    return [
        hashlib.md5(data[k : k + FRAME_BYTES]).hexdigest() for k in range(0, len(data), FRAME_BYTES)
    ]


def listed(tmp_path: Path, name: str, text: str = "") -> str:
    """A cut list written beside the job, empty for one shot, its path."""
    path = tmp_path / name
    path.write_text(text)
    return str(path)


@pytest.mark.usefixtures("stand_in")
@pytest.mark.parametrize("segments", ["0", None], ids=["segments-of-a-shot", "one-segment"])
@pytest.mark.parametrize("runs", [None, "1", "2", "4"], ids=["default", "1", "2", "4"])
def test_each_frame_from_its_run(tmp_path: Path, runs: str | None, segments: str | None) -> None:
    # Shots of 1 to 13 frames, so that every length is checked at each of its frames, in a source
    # of 91 = 13 * 14 / 2 frames. With --min-segment 0 every shot is a segment of its own (13
    # masters): each decoded on its own, the shot's kept frames flushed at the segment's end.
    # With the default (5 s, longer than the source) all 13 are one segment, written by one
    # writer: the frames of a shot with several runs then follow those of the shot before them,
    # none lost or repeated at a join. A segment holds whole shots in both (run_job).
    lengths = range(1, 14)
    starts = [0, *accumulate(lengths)]
    cuts = listed(tmp_path, "cuts.txt", "".join(f"{start}\n" for start in starts[1:-1]))
    options = ["--cuts", cuts]
    if runs is not None:
        options += ["--quality", LEVEL[runs]]
    if segments is not None:
        options += ["--min-segment", segments]
    out = tmp_path / "out"
    assert upscale(tmp_path, source(tmp_path / "in.mkv", starts[-1]), "out", *options) == 0
    masters = sorted(out.glob("seg_*.mkv"))
    assert len(masters) == (13 if segments else 1)
    count = int(runs or 1)
    expected = [
        start + frame - taken(count, frame)
        for start, length in zip(starts, lengths, strict=False)
        for frame in range(length)
    ]
    assert list(chain.from_iterable(indexes(master) for master in masters)) == expected
    if count == 1:  # the unchanged output: every frame of the job at its place
        assert expected == list(range(91))


@pytest.mark.parametrize("correction", ["none", "split"])
@pytest.mark.parametrize("runs", [4, 2])
def test_composite_equals_single_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    runs: int,
    correction: str,
) -> None:
    # One shot of 17 frames (its seed the job's, 42, for every run). Run s of the job is a whole
    # run of the shot fed from its frame s: so it equals a job of one run a shot on the source
    # trimmed to start at frame s, the same seed. The job's frame f is then the trimmed job's
    # frame f - s, s = taken(runs, f), bit for bit as stored, and its dumped float32 frame the
    # trimmed job's dumped one. With `split` the output depends on the source's own frames (each
    # run's correction reads the shot's input copy from the run's first frame), with `none` only
    # on the stand-in's: both must hold. Each encode must also read the frames of its trimmed
    # source, not another run's.
    import subprocess

    original = source(tmp_path / "in.mkv", 17)
    cuts = listed(tmp_path, "cuts.txt")
    reads: list[str] = []

    def reading(models: object, read: Any, count: int, *arguments: Any, **keywords: Any) -> object:
        frames: npt.NDArray[np.float32] = read(count)
        reads.append(hashlib.md5(frames.tobytes()).hexdigest())
        return steps.encode(models, lambda n: frames[:n], count, *arguments, **keywords)

    monkeypatch.setattr(run, "encode_shot", reading)
    options = ("--cuts", cuts, "--color-correction", correction)
    offsets = sorted({taken(runs, frame) for frame in range(17)})
    assert offsets == ([0, 1, 2, 3] if runs == 4 else [0, 2])
    singles: dict[int, list[str]] = {}
    dumps: dict[int, Path] = {}
    single_reads: list[str] = []
    for offset in offsets:
        trimmed = tmp_path / f"from_{offset}.mkv"
        subprocess.run(
            [
                *("ffmpeg", "-v", "error", "-i", str(original)),
                *("-vf", f"trim=start_frame={offset},setpts=PTS-STARTPTS"),
                *("-c:v", "ffv1", "-pix_fmt", "yuv420p", str(trimmed)),
            ],
            check=True,
        )
        # The trimmed source holds the source's frames from `offset`, bit for bit.
        assert decoded(trimmed) == decoded(original)[offset * SOURCE_BYTES :]
        dumps[offset] = tmp_path / f"dump_from_{offset}"
        reads.clear()
        name = f"single_{offset}.mkv"
        assert upscale(tmp_path, trimmed, name, *options, "--dump-frames", str(dumps[offset])) == 0
        singles[offset] = frames_of(tmp_path / name)
        assert len(singles[offset]) == 17 - offset and len(reads) == 1
        single_reads += reads
    reads.clear()
    dump = tmp_path / "dump"
    assert upscale(tmp_path, original, "job.mkv", *options, "--quality", LEVEL[str(runs)],
                   "--dump-frames", str(dump)) == 0  # fmt: skip
    assert reads == single_reads
    chosen = [taken(runs, frame) for frame in range(17)]
    assert frames_of(tmp_path / "job.mkv") == [
        singles[offset][frame - offset] for frame, offset in enumerate(chosen)
    ]
    assert sorted(path.name for path in dump.iterdir()) == [f"frame_{i:06d}.npy" for i in range(17)]
    for frame, offset in enumerate(chosen):
        single = np.load(dumps[offset] / f"frame_{frame - offset:06d}.npy")
        assert single.dtype == np.float32
        assert np.array_equal(np.load(dump / f"frame_{frame:06d}.npy"), single)


def stopped_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    *options: str,
) -> Path:
    """test_cli_run's job (shots of 3, 1 and 21 frames, two segments) at 4 runs a shot, stopped
    in its last segment's decode, once every run is sampled: its source."""
    source_path = job(tmp_path, monkeypatch)
    steps.stop = "decode 4"
    stopped(tmp_path, source_path, "out", *JOB, "--quality", "maximum", *options)
    return source_path


def test_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A shot of 9 frames has runs from its frames 1, 2 and 3 after its own; a shot of 3 frames
    # those from 1 and 2 only; a shot of 1 frame none, its record's list empty. The first run's
    # fields are the shot's own. A job of one run a shot records the count, 1, and no list.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    monkeypatch.chdir(tmp_path)
    source_path = source(tmp_path / "in.mkv", 13)
    cuts = listed(tmp_path, "cuts.txt", "3\n12\n")
    options = ("--cuts", cuts, "--window", "5", "--min-segment", "0")
    steps.stop = "decode 12"
    stopped(tmp_path, source_path, "four", *options, "--quality", "maximum")
    content = json.loads((tmp_path / "four" / "manifest.json").read_text())
    assert content["settings"]["quality"] == "maximum"
    short, long, single = content["shots"]
    assert [(shot["start"], shot["end"]) for shot in content["shots"]] == [
        (0, 3),
        (3, 12),
        (12, 13),
    ]
    assert [entry["from"] for entry in short["runs"]] == [1, 2]
    assert [entry["from"] for entry in long["runs"]] == [1, 2, 3]
    assert single["runs"] == []
    for shot in content["shots"]:
        for entry in [shot, *shot["runs"]]:
            assert set(entry) >= {"latents", "windows", "encoded", "windows_done"}
            # Every run sampled before the first decode.
            assert entry["encoded"] and entry["windows_done"] == len(entry["windows"])
    # The windows are each run's own, of its own frames: 9, 8, 7 and 6 frames of the shot.
    assert [entry["latents"] for entry in [long, *long["runs"]]] == [
        (padded_length(frames) - 1) // 4 + 1 for frames in (9, 8, 7, 6)
    ]
    assert quality_line(plan_lines(caplog)) == RUNS_SAID.format(
        level="maximum", runs=4, total=8, kept="0.00", others="standard or enhanced: 1 or 2 runs"
    )
    steps.stop = "decode 12"
    stopped(tmp_path, source_path, "single", *options, "--quality", "standard")
    content = json.loads((tmp_path / "single" / "manifest.json").read_text())
    assert content["settings"]["quality"] == "standard"
    assert all("runs" not in shot for shot in content["shots"])
    # Without the option, the same record.
    stopped(tmp_path, source_path, "default", *options)
    again = json.loads((tmp_path / "default" / "manifest.json").read_text())
    assert again["settings"]["quality"] == "standard"
    assert again["shots"] == content["shots"]


@pytest.mark.parametrize("correction", ["none", "split"])
@pytest.mark.parametrize(
    ("name", "nth", "shot", "present"),
    [
        # In the encode of shot 0's second run: only the first run's window kept.
        ("encode 0", 2, 0, set[str]()),
        # In the first window of shot 4's second run: its latent and the first run's windows kept.
        ("window 4:0", 2, 4, {"run1_"}),
        # In the encode of its third: the second run's windows all kept.
        ("encode 4", 3, 4, {"run1_"}),
        # In the second window of its last: the second and third runs sampled, its latent kept.
        ("window 4:1", 4, 4, {"run1_", "run2_", "run3_"}),
    ],
)
def test_resumed_in_a_later_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    name: str,
    nth: int,
    shot: int,
    present: set[str],
    correction: str,
) -> None:
    # A job of 4 runs a shot stopped inside a later run's encode or window, then run again: it
    # resumes at that unit (every call after it, none before) and the output is the
    # uninterrupted job's, bit for bit. Its units are named for the run (units.run_file) while it
    # waits. Calls are named by the shot, so the stop is the nth of its name (a run's own).
    source_path = job(tmp_path, monkeypatch)
    options = (*JOB, "--quality", "maximum", "--color-correction", correction)
    assert upscale(tmp_path, source_path, "whole", *options) == 0
    sampled = list(steps.calls)
    assert sampled.count(name) >= nth
    seen = 0

    def call(called: str) -> None:
        nonlocal seen
        steps.calls.append(called)
        if called == name:
            seen += 1
            if seen == nth:
                raise KeyboardInterrupt

    steps.calls.clear()
    monkeypatch.setattr(steps, "_call", call)
    stopped(tmp_path, source_path, "out", *options)
    at = len(steps.calls)
    assert steps.calls == sampled[:at] and steps.calls[-1] == name
    kept = tmp_path / "out" / "resume" / f"shot_{shot:06d}"
    files = sorted(path.name for path in kept.iterdir())
    assert {file[:5] for file in files if file.startswith("run")} == present
    steps.calls.clear()
    monkeypatch.setattr(steps, "_call", lambda called: steps.calls.append(called))
    assert upscale(tmp_path, source_path, "out", *options) == 0
    assert steps.calls == sampled[at - 1 :]
    out = tmp_path / "out"
    for master in ("seg_000000.mkv", "seg_000001.mkv"):
        assert decoded(out / master) == decoded(tmp_path / "whole" / master)
    assert contents(out / "checksums") == contents(tmp_path / "whole" / "checksums")
    assert not (out / "resume").exists()


def test_kept_frames_leftover_discarded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A kill in a shot's decode leaves its file of kept frames, derived data the manifest doesn't
    # name (resume.UNIT_FILE): the resume says so and discards it, and the decode, made again
    # whole from the windows kept, gives the uninterrupted job's output.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    options = (*JOB, "--quality", "maximum")
    assert upscale(tmp_path, source_path, "whole", *options) == 0
    stopped_source = stopped_job(tmp_path, monkeypatch, steps)
    assert stopped_source == source_path
    shot = tmp_path / "out" / "resume" / "shot_000004"
    assert not (shot / "runs.kept").exists()  # removed by the decode that was stopped
    (shot / "runs.kept").write_bytes(b"left by a kill")
    steps.stop = None
    steps.calls.clear()
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *options) == 0
    assert "1 leftover discarded" in caplog.text
    assert steps.calls == ["decode 4"] * 4
    out = tmp_path / "out"
    for master in ("seg_000000.mkv", "seg_000001.mkv"):
        assert decoded(out / master) == decoded(tmp_path / "whole" / master)
    assert not list(out.rglob("runs.kept"))


def test_resumed_with_another_count_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The number of runs shapes the output: a resume with another count is refused before the
    # first pass, naming the setting, whichever way, and nothing is made or deleted.
    scans = first_passes(monkeypatch)
    source_path = stopped_job(tmp_path, monkeypatch, steps)
    assert len(scans) == 1
    steps.calls.clear()
    steps.stop = None
    out = tmp_path / "out"
    kept = contents(out)
    for another, said in (
        ((), 'settings.quality: "maximum" -> "standard"'),
        (("--quality", "standard"), 'settings.quality: "maximum" -> "standard"'),
        (("--quality", "enhanced"), 'settings.quality: "maximum" -> "enhanced"'),
    ):
        text = refused(tmp_path, source_path, caplog, *JOB, *another)
        assert "another job than the one asked" in text and said in text
        assert len(scans) == 1 and steps.calls == [] and contents(out) == kept
    # The same command resumes it.
    assert upscale(tmp_path, source_path, "out", *JOB, "--quality", "maximum") == 0
    assert steps.calls == ["decode 4"] * 4
    # And a job of one run, resumed with 4.
    steps.calls.clear()
    steps.stop = "decode 4"
    stopped(tmp_path, source_path, "single", *JOB)
    assert 'settings.quality: "standard" -> "maximum"' in refused(
        tmp_path, source_path, caplog, *JOB, "--quality", "maximum", output="single"
    )


@pytest.mark.usefixtures("stand_in")
@pytest.mark.parametrize("runs", ["4", "2", None], ids=["4", "2", "default"])
def test_plan_says_the_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    runs: str | None,
) -> None:
    # --plan says the runs before anything runs, in its line on the quality level, which every
    # level has, the default's too: the runs counted by what each shot has (JOB's shots of 3, 1
    # and 21 frames have 3, 1 and 4 runs at 4 a shot, 2, 1 and 2 at 2). The other lines are the
    # one run's, the plan holding for every run. The models load, for the time estimates'
    # calibration (DESIGN.md, Memory planner), after the plan's lines.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    with monkeypatch.context():
        assert upscale(tmp_path, source_path, "one", *JOB, "--plan", str(tmp_path / "one.txt")) == 0
        base = plan_lines(caplog)
        caplog.clear()
        options = [] if runs is None else ["--quality", LEVEL[runs]]
        assert upscale(tmp_path, source_path, "out", *JOB, *options, "--plan", "more.txt") == 0
        lines = plan_lines(caplog)
    assert len(base) == 16 and quality_line(base) == STANDARD_SAID
    if runs is None:
        assert lines == base
        return
    total = {"4": 8, "2": 5}[runs]
    others = {
        "4": "standard or enhanced: 1 or 2 runs",
        "2": "standard or maximum: 1 or 4 runs",
    }[runs]
    # The stand-in's 128x96 frames keep 2.4 MB at the most (16 of the 21-frame shot's at 4 a shot),
    # 0.00 GiB as said; tests/test_runs.py holds the figure at 1080p.
    said = RUNS_SAID.format(level=LEVEL[runs], runs=runs, total=total, kept="0.00", others=others)
    assert lines == [said if line == STANDARD_SAID else line for line in base]
    windows = next(line for line in base if line.startswith("plan: windows: "))
    assert windows.startswith("plan: windows: 5 latents at most")
    assert not (tmp_path / "out").joinpath("resume").exists()


@pytest.mark.usefixtures("stand_in")
def test_plan_counts_a_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # One run in all: a one-frame source has run 0 only at 4 a shot.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    monkeypatch.chdir(tmp_path)
    cuts = listed(tmp_path, "cuts.txt")
    options = ("--cuts", cuts, "--quality", "maximum", "--plan", str(tmp_path / "plan.txt"))
    assert upscale(tmp_path, source(tmp_path / "in.mkv", 1), "out", *options) == 0
    assert quality_line(plan_lines(caplog)) == RUNS_SAID.format(
        level="maximum", runs=4, total=1, kept="0.00", others="standard or enhanced: 1 or 2 runs"
    )


@pytest.mark.usefixtures("stand_in")
@pytest.mark.parametrize("given", ["best", "4", "Maximum", ""])
def test_another_level_refused(
    tmp_path: Path, given: str, capsys: pytest.CaptureFixture[str]
) -> None:
    # The levels' names only (job.QUALITY): a count of runs, the code's, is no level.
    with pytest.raises(SystemExit) as exited:
        upscale(tmp_path, tmp_path / "in.mkv", "out", "--quality", given)
    assert exited.value.code == 2
    error = capsys.readouterr().err
    assert "argument --quality: invalid choice" in error
    assert all(level in error for level in ("standard", "enhanced", "maximum"))


def test_kept_frames_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
) -> None:
    # The frames the runs keep for a shot's decode (run.Kept), in a file of the shot's directory
    # allocated whole before the first is put, are there during every decode of a shot with
    # several runs, 1 frame of the shot in 4 less the last run's (the last run's are written as
    # they come), the shot's frames the earlier runs give and no more; and none is left once the
    # job is finished, in the output or beside it.
    source_path = job(tmp_path, monkeypatch)
    sizes: list[int | None] = []
    decode = steps.decode

    def watching(models: Any, merged: Any, *arguments: Any, **keywords: Any) -> None:
        first = int(merged[0, 0, 0, 0])
        file = tmp_path / "out" / "resume" / f"shot_{first:06d}" / "runs.kept"
        sizes.append(file.stat().st_size if file.exists() else None)
        decode(models, merged, *arguments, **keywords)

    monkeypatch.setattr(run, "decode_shot", watching)
    assert upscale(tmp_path, source_path, "out", *JOB, "--quality", "maximum") == 0
    # Shot 0 (3 frames, runs from 0, 1 and 2, the last giving frame 2): frames 0 and 1 kept
    # during each of its 3 decodes; shot 3 (1 frame) one run, no store; shot 4 (21 frames, the
    # last run giving every frame of index 3 mod 4, 5 of them): 16 kept during each of its 4.
    assert sizes == [2 * KEPT_BYTES] * 3 + [None] + [16 * KEPT_BYTES] * 4
    out = tmp_path / "out"
    assert not list(out.rglob("runs.kept")) and not (out / "resume").exists()
    assert sorted(path.name for path in out.iterdir()) == [
        "checksums",
        "frame_index.bin",
        "manifest.json",
        "seg_000000.mkv",
        "seg_000001.mkv",
    ]
    assert indexes(out / "seg_000000.mkv") + indexes(out / "seg_000001.mkv") == [
        *(0 + f - taken(4, f) for f in range(3)),
        3,
        *(4 + f - taken(4, f) for f in range(21)),
    ]


class FloatKept:
    """A stand-in for run.Kept that keeps the frames as float32, as the decode gives them, and
    writes them so, the writers quantising them; in memory, which changes nothing of their
    values. What the 16-bit file must equal byte for byte (design's ruling of 2026-10-20)."""

    def __init__(self, path: Path, frames: list[int], shape: tuple[int, int, int]) -> None:
        self.frames, self._kept, self._next = list(frames), dict[int, Any](), 0

    def __enter__(self) -> "FloatKept":
        return self

    def __exit__(self, *exc: object) -> None:
        assert exc[0] is not None or self._next == len(self.frames)

    def put(self, frame: int, values: npt.NDArray[np.float32]) -> None:
        self._kept[frame] = np.array(values, dtype=np.float32)

    def pending(self, frame: int) -> bool:
        return self._next < len(self.frames) and self.frames[self._next] < frame

    def flush(self, frame: int, write: Any) -> None:
        end = self._next
        while end < len(self.frames) and self.frames[end] < frame:
            end += 1
        if end > self._next:
            write(np.stack([self._kept[f] for f in self.frames[self._next : end]]))
            self._next = end


def written_files(out: Path) -> dict[str, bytes]:
    """Every file of a job's output directory but its manifest, by its path there: a Matroska
    master by its video packets, byte for byte (their MD5s, in order), since the muxer gives each
    file a random segment UID; any other file by its bytes."""
    found: dict[str, bytes] = {}
    for path in sorted(out.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        if path.suffix == ".mkv":
            packets = subprocess.run(
                [
                    *("ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v", "-c", "copy"),
                    *("-f", "framemd5", "-"),
                ],
                capture_output=True,
                check=True,
            ).stdout
            found[str(path.relative_to(out))] = b"\n".join(
                line for line in packets.splitlines() if not line.startswith(b"#")
            )
        else:
            found[str(path.relative_to(out))] = path.read_bytes()
    return found


@pytest.mark.usefixtures("stand_in")
@pytest.mark.parametrize("correction", ["none", "split"])
@pytest.mark.parametrize("output_format", ["gbrp16le", "yuv420p10le", "png", "segment-cmd"])
def test_sixteen_bit_kept_frames_change_no_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output_format: str, correction: str
) -> None:
    # The frames kept between the runs at the 16 bits the writers quantise them to: every master
    # and checksum file byte for byte those of the frames kept as float32 (FloatKept), in every
    # format, the frame dumps too, written by the runs that make them (DESIGN.md, Several runs of
    # a shot), and through a segment command's writer, whose stream is the master's (DESIGN.md,
    # Output, Writer): an FFV1 encode of yuv420p10le, which its packets show. Shots of 5, 1 and 11
    # frames at maximum: every place of the rule, a shot of one run.
    original = source(tmp_path / "in.mkv", 17)
    cuts = listed(tmp_path, "cuts.txt", "5\n6\n")
    command = ("--segment-cmd", "ffmpeg -i - -c:v ffv1 {out}")
    writer = ("yuv420p10le", *command) if output_format == "segment-cmd" else (output_format,)
    options = ("--cuts", cuts, "--quality", "maximum", "--format", *writer)
    options += ("--color-correction", correction)
    outputs: dict[str, dict[str, bytes]] = {}
    for kept in ("sixteen", "float"):
        with monkeypatch.context() as patch:
            if kept == "float":
                patch.setattr(run, "Kept", FloatKept)
            dump = tmp_path / f"dump-{kept}"
            assert upscale(tmp_path, original, kept, *options, "--dump-frames", str(dump)) == 0
        outputs[kept] = written_files(tmp_path / kept)
        outputs[f"{kept}-dump"] = written_files(dump)
    assert outputs["sixteen"] == outputs["float"]
    assert len(outputs["sixteen-dump"]) == 17
    assert outputs["sixteen-dump"] == outputs["float-dump"]


def short_disk(patch: pytest.MonkeyPatch, free: int) -> None:
    """The disk of the job's directory said to have `free` bytes free, through patch."""
    real = shutil.disk_usage

    def usage(path: str | Path) -> Any:
        found = real(path)
        return type(found)(found.total, found.used, free)

    patch.setattr(shutil, "disk_usage", usage)


@pytest.mark.usefixtures("stand_in")
def test_a_disk_short_of_the_kept_frames_refuses_the_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Shots of 5, 1 and 11 frames at 128x96: at maximum the 11-frame shot keeps the most, 9
    # frames of 73,728 bytes at 16 bits (of its 11, the last run gives frames 3 and 7), 663,552
    # bytes; enhanced keeps 6 of them (its last run gives frames 2, 5, 6, 9 and 10). One byte
    # short: refused before the models load, no unit made, the first pass kept as a plan
    # (DESIGN.md, Pause and resume, A plan, decided on 2026-10-21; nothing was written before);
    # with the room, the job runs (DESIGN.md: "a job whose disk can't hold it is refused at its
    # start"). The check counts a tiled pass's scratch file too, which the stand-in's untiled
    # plan has none of (test_cli_plan_kept.py).
    caplog.set_level(logging.INFO, logger="seedvr2x")
    original = source(tmp_path / "in.mkv", 17)
    cuts = listed(tmp_path, "cuts.txt", "5\n6\n")
    options = ("--cuts", cuts, "--quality", "maximum")
    with monkeypatch.context() as patch:
        short_disk(patch, 9 * KEPT_BYTES - 1)
        loads = model.load_models
        patch.setattr(model, "load_models", lambda *a: pytest.fail("the models loaded"))
        assert upscale(tmp_path, original, "out", *options) == 1
        said = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR][-1]
        out = tmp_path / "out"
        assert said == (
            f"{out}: 0.00 GiB free on its disk, where the job writes up to 0.00 GiB there at a"
            " shot's peak, its shot of 11 frames: 0.00 GiB of frames kept between its runs at"
            " --quality maximum; make room there, write the output to a disk with room (a new job,"
            " its first pass again) or take a"
            " lower level (standard none, enhanced keeps 0.00 GiB); the first pass is kept in"
            f" {out}: the same command takes it up, or one changing what is named here, without"
            " decoding the source again"
        )
        assert not (out / "resume").exists()
        assert sorted(path.name for path in out.iterdir()) == ["frame_index.bin", "manifest.json"]
        # --plan says it, and plans, the models loaded for the time estimates' calibration.
        patch.setattr(model, "load_models", loads)
        caplog.clear()
        plan = str(tmp_path / "plan.txt")
        assert upscale(tmp_path, original, "out", *options, "--plan", plan) == 0
        warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any(m.endswith("; a run of this plan would be refused") for m in warned)
    # Room for maximum's 9 frames, to the byte: the job runs.
    with monkeypatch.context() as patch:
        short_disk(patch, 9 * KEPT_BYTES)
        assert upscale(tmp_path, original, "four", *options) == 0
    # Enhanced keeps 6 frames, which fit where maximum's 9 don't.
    with monkeypatch.context() as patch:
        short_disk(patch, 6 * KEPT_BYTES)
        assert upscale(tmp_path, original, "two", "--cuts", cuts, "--quality", "enhanced") == 0
    # Standard keeps nothing: no room asked.
    with monkeypatch.context() as patch:
        short_disk(patch, 0)
        assert upscale(tmp_path, original, "one", "--cuts", cuts) == 0


@pytest.mark.usefixtures("stand_in")
def test_a_killed_decode_s_kept_frames_count_as_room(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A kill in a shot's decode leaves its runs.kept, which the resume discards after the disk
    # check: its bytes count as free, so a disk that held the job holds its resume.
    original = source(tmp_path / "in.mkv", 17)
    cuts = listed(tmp_path, "cuts.txt", "5\n6\n")
    options = ("--cuts", cuts, "--quality", "maximum")
    # The job planned (its manifest, no unit), then the file a kill would leave put in place.
    assert upscale(tmp_path, original, "out", *options, "--plan", str(tmp_path / "plan.txt")) == 0
    left = tmp_path / "out" / "resume" / "shot_000006" / KEPT  # the 11-frame shot's
    left.parent.mkdir(parents=True, exist_ok=True)
    left.write_bytes(bytes(9 * KEPT_BYTES))
    with monkeypatch.context() as patch:
        short_disk(patch, 0)  # the leftover alone is the room
        assert upscale(tmp_path, original, "out", *options) == 0
    assert not left.exists()

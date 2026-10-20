"""The command line's several runs of a shot (--runs-per-shot; DESIGN.md, Several runs of a shot,
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

# The output's frames as test_cli_run's stand-in gives them (gbrp16le, 128x96), in bytes.
FRAME_BYTES = 3 * 96 * 128 * 2
# The same, float32 and counted in the frames the runs keep for the decode (run.Kept): (H, W, 3).
KEPT_BYTES = 96 * 128 * 3 * 4
# The source's frames, yuv420p at 64x48.
SOURCE_BYTES = 64 * 48 * 3 // 2

# The plan's line on several runs (planner._runs_said): the runs a shot, and the count in all.
RUNS_SAID = (
    "plan: {runs} runs a shot, {total} in all (--runs-per-shot), each frame taken from the run"
    " that rebuilds it best: about {runs} times the GPU time of one run, which no estimate gives"
    " yet, in the memory of one run; the longest shot keeps {kept} GiB of frames on disk, beside"
    " the job's manifest, while its runs are decoded"
)

# The counts offered as built (job.RUNS_OFFERED): two runs wait for the user's eyes.
OFFERED = runtime_job.RUNS_OFFERED


@pytest.fixture(autouse=True)
def two_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two runs offered, as they are built, for the tests of what they do: refused until the
    user's eyes have seen them (DESIGN.md, Several runs of a shot), which
    test_two_runs_not_offered_yet holds."""
    monkeypatch.setattr(runtime_job, "RUNS_OFFERED", (1, 2, 4))


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
        options += ["--runs-per-shot", runs]
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
    assert upscale(tmp_path, original, "job.mkv", *options, "--runs-per-shot", str(runs),
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
    stopped(tmp_path, source_path, "out", *JOB, "--runs-per-shot", "4", *options)
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
    stopped(tmp_path, source_path, "four", *options, "--runs-per-shot", "4")
    content = json.loads((tmp_path / "four" / "manifest.json").read_text())
    assert content["settings"]["runs_per_shot"] == 4
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
    assert plan_lines(caplog)[4] == RUNS_SAID.format(runs=4, total="8 runs", kept="0.00")
    steps.stop = "decode 12"
    stopped(tmp_path, source_path, "single", *options, "--runs-per-shot", "1")
    content = json.loads((tmp_path / "single" / "manifest.json").read_text())
    assert content["settings"]["runs_per_shot"] == 1
    assert all("runs" not in shot for shot in content["shots"])
    # Without the option, the same record.
    stopped(tmp_path, source_path, "default", *options)
    again = json.loads((tmp_path / "default" / "manifest.json").read_text())
    assert again["settings"]["runs_per_shot"] == 1
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
    options = (*JOB, "--runs-per-shot", "4", "--color-correction", correction)
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
    options = (*JOB, "--runs-per-shot", "4")
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
        ((), "settings.runs_per_shot: 4 -> 1"),
        (("--runs-per-shot", "1"), "settings.runs_per_shot: 4 -> 1"),
        (("--runs-per-shot", "2"), "settings.runs_per_shot: 4 -> 2"),
    ):
        text = refused(tmp_path, source_path, caplog, *JOB, *another)
        assert "another job than the one asked" in text and said in text
        assert len(scans) == 1 and steps.calls == [] and contents(out) == kept
    # The same command resumes it.
    assert upscale(tmp_path, source_path, "out", *JOB, "--runs-per-shot", "4") == 0
    assert steps.calls == ["decode 4"] * 4
    # And a job of one run, resumed with 4.
    steps.calls.clear()
    steps.stop = "decode 4"
    stopped(tmp_path, source_path, "single", *JOB)
    assert "settings.runs_per_shot: 1 -> 4" in refused(
        tmp_path, source_path, caplog, *JOB, "--runs-per-shot", "4", output="single"
    )


@pytest.mark.usefixtures("stand_in")
@pytest.mark.parametrize("runs", ["4", "2", None], ids=["4", "2", "default"])
def test_plan_says_the_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    runs: str | None,
) -> None:
    # --plan says the runs before anything runs, one line more than a job of one run a shot, which
    # has none: after the line on the windows, the runs counted by what each shot has (JOB's shots
    # of 3, 1 and 21 frames have 3, 1 and 4 runs at 4 a shot, 2, 1 and 2 at 2). The other lines
    # are the one run's, the plan holding for every run. No model loads.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    with monkeypatch.context() as patch:
        patch.setattr(model, "load_models", lambda *a: pytest.fail("the models loaded"))
        assert upscale(tmp_path, source_path, "one", *JOB, "--plan", str(tmp_path / "one.txt")) == 0
        base = plan_lines(caplog)
        caplog.clear()
        options = [] if runs is None else ["--runs-per-shot", runs]
        assert upscale(tmp_path, source_path, "out", *JOB, *options, "--plan", "more.txt") == 0
        lines = plan_lines(caplog)
    assert len(base) == 9 and "runs a shot" not in " ".join(base)
    if runs is None:
        assert lines == base
        return
    total = {"4": "8 runs", "2": "5 runs"}[runs]
    # The stand-in's 128x96 frames keep 2.4 MB at the most (16 of the 21-frame shot's at 4 a shot),
    # 0.00 GiB as said; tests/test_runs.py holds the figure at 1080p.
    said = RUNS_SAID.format(runs=runs, total=total, kept="0.00")
    assert lines == [*base[:4], said, *base[4:]]
    assert base[3].startswith("plan: windows of 5 latents at most")
    assert not (tmp_path / "out").joinpath("resume").exists()


@pytest.mark.usefixtures("stand_in")
def test_plan_counts_a_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # One run in all, said in the singular: a one-frame source has run 0 only at 4 a shot.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(model, "load_models", lambda *a: pytest.fail("the models loaded"))
    cuts = listed(tmp_path, "cuts.txt")
    options = ("--cuts", cuts, "--runs-per-shot", "4", "--plan", str(tmp_path / "plan.txt"))
    assert upscale(tmp_path, source(tmp_path / "in.mkv", 1), "out", *options) == 0
    assert plan_lines(caplog)[4] == RUNS_SAID.format(runs=4, total="1 run", kept="0.00")


@pytest.mark.usefixtures("stand_in")
@pytest.mark.parametrize("given", ["3", "0", "8", "x", ""])
def test_another_count_refused(
    tmp_path: Path, given: str, capsys: pytest.CaptureFixture[str]
) -> None:
    # 1, 2 and 4 are the counts built (job.RUN_OFFSETS): argparse refuses the rest, in one line.
    with pytest.raises(SystemExit) as exited:
        upscale(tmp_path, tmp_path / "in.mkv", "out", "--runs-per-shot", given)
    assert exited.value.code == 2
    error = capsys.readouterr().err
    assert f"argument --runs-per-shot: {given!r}: not a count of runs a shot, 1, 2 or 4" in error


@pytest.mark.usefixtures("stand_in")
def test_two_runs_not_offered_yet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Built, refused in one line until the user's eyes have seen them (DESIGN.md: "Offered or not
    # once the user's eyes have seen it against four runs and one"); 1 and 4 taken.
    monkeypatch.setattr(runtime_job, "RUNS_OFFERED", OFFERED)
    assert OFFERED == (1, 4)
    with pytest.raises(SystemExit) as exited:
        upscale(tmp_path, tmp_path / "in.mkv", "out", "--runs-per-shot", "2")
    assert exited.value.code == 2
    lines = [line for line in capsys.readouterr().err.splitlines() if "--runs-per-shot" in line]
    assert lines[-1].endswith(
        "argument --runs-per-shot: 2: not offered yet, its gain waiting for the user's eyes; 1 or 4"
    )


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
    assert upscale(tmp_path, source_path, "out", *JOB, "--runs-per-shot", "4") == 0
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

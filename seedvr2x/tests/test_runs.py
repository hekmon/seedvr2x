"""Several runs of a shot (DESIGN.md, Several runs of a shot, an option for quality), on the CPU,
with no command line: the rule that says which run each frame is taken from, held to measurement's
composer; the runs of a shot, short ones included; the manifest's records of them (shape, progress,
refusals); what a resume reads back (resume.recorded_job, adopt, differences, UNIT_FILE, _kept);
the units named for their run (units.run_file); and the store of the frames the runs keep for the
shot's decode (run.Kept) with the two writes that fill it and empty it (run._given,
run._interleaved)."""

import errno
import json
import os
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
import torch

from seedvr2x.media.source import Source
from seedvr2x.media.writer import quantise
from seedvr2x.runtime import job, manifest, planner, resume, run
from seedvr2x.runtime.job import (
    PLACES_BEST_FIRST,
    RUN_OFFSETS,
    JobError,
    OutputSegment,
    Run,
    Shot,
    shot_runs,
)
from seedvr2x.runtime.manifest import Manifest
from seedvr2x.runtime.plan import Plan
from seedvr2x.runtime.shot import padded_length, shot_layout
from seedvr2x.runtime.tiling import ScratchError
from seedvr2x.runtime.units import KEPT, Units, run_file

# --- The rule -----------------------------------------------------------------------------------


# The grid study's composer, compose.py (measurement, 2026-10-19; research/docs/numerics.md,
# Every fourth frame: the latent grid): `place` and `pick` copied verbatim, but for the type hints.
# job.run_of is held to `pick`, for every shot length and frame.
def place(j: int) -> int:
    return -1 if j == 0 else (j - 1) % 4


def pick(kind: str, f: int) -> int:
    if kind == "grid4":
        return f % 4
    if f < 2:
        return 0
    p0, p2 = place(f), place(f - 2)
    rank = {-1: 4, 3: 3, 2: 2, 0: 1, 1: 0}  # j = 0 (own latent) first, then places 3, 2, 0, 1
    return 0 if rank[p0] >= rank[p2] else 2


KINDS = {2: "grid2", 4: "grid4"}
# The --quality levels by their runs a shot (job.QUALITY), as a manifest records them.
LEVELS = {runs: level for level, runs in job.QUALITY.items()}
LENGTHS = range(1, 61)


def test_place_is_the_composers() -> None:
    assert [job.place(i) for i in range(61)] == [place(i) for i in range(61)]
    assert [job.place(i) for i in range(10)] == [-1, 0, 1, 2, 3, 0, 1, 2, 3, 0]
    with pytest.raises(ValueError):
        job.place(-1)


@pytest.mark.parametrize("runs", [2, 4])
def test_run_of_is_the_composers_for_every_shot(runs: int) -> None:
    for length in LENGTHS:
        offsets = [r.offset for r in shot_runs(Shot(0, length), runs)]
        for frame in range(length):
            taken = job.run_of(runs, frame)
            assert taken == pick(KINDS[runs], frame), (length, frame)
            assert taken in offsets, (length, frame)


def test_one_run_takes_every_frame() -> None:
    assert [job.run_of(1, frame) for frame in range(61)] == [0] * 61
    with pytest.raises(ValueError):
        job.run_of(4, -1)


@pytest.mark.parametrize("runs", [2, 4])
def test_the_best_place_is_held_by_one_run(runs: int) -> None:
    # No tie for run_of to break (job.run_of's comment): the places of a frame in the runs that
    # hold it are all different, so the best is one run's.
    for frame in range(61):
        held = [frame - offset for offset in RUN_OFFSETS[runs] if offset <= frame]
        ranks = [PLACES_BEST_FIRST.index(place(j)) for j in held]
        assert ranks.count(min(ranks)) == 1, frame


@pytest.mark.parametrize("runs", [1, 2, 4])
def test_every_run_gives_its_first_frame(runs: int) -> None:
    # A run is never wasted: its first frame is a latent of its own, the best place, so it gives at
    # least that frame, whatever the shot's length.
    for length in LENGTHS:
        for r in shot_runs(Shot(0, length), runs):
            assert job.run_of(runs, r.offset) == r.offset, (length, r.offset)


@pytest.mark.parametrize(
    ("frames", "runs", "offsets"),
    [
        (1, 1, [0]),
        (1, 2, [0]),
        (1, 4, [0]),
        (2, 1, [0]),
        (2, 2, [0]),
        (2, 4, [0, 1]),
        (3, 1, [0]),
        (3, 2, [0, 2]),
        (3, 4, [0, 1, 2]),
        (4, 1, [0]),
        (4, 2, [0, 2]),
        (4, 4, [0, 1, 2, 3]),
        (5, 1, [0]),
        (5, 2, [0, 2]),
        (5, 4, [0, 1, 2, 3]),
    ],
)
def test_short_shots_have_the_runs_they_can_hold(
    frames: int, runs: int, offsets: list[int]
) -> None:
    assert [r.offset for r in shot_runs(Shot(7, 7 + frames), runs)] == offsets


def test_a_run_is_the_shot_from_its_frame() -> None:
    shot = Shot(10, 20)
    later = Run(shot, 3)
    assert (later.start, later.end, later.frames) == (13, 20, 7)
    first = Run(shot, 0)
    assert (first.start, first.end, first.frames) == (10, 20, 10)
    # Every run has its shot's seed, deliberately (job.Run): the composite is measurement's.
    assert shot.seed(42) == first.seed(42) == later.seed(42) == 52
    assert [(r.start, r.end) for r in shot_runs(shot, 4)] == [
        (10, 20),
        (11, 20),
        (12, 20),
        (13, 20),
    ]


# What a reviewer reads: the run each frame of a shot of 13 frames is taken from, and its place
# there (-1 its first frame, else 0 to 3 in its group of 4, 3 the last), for 2 and for 4 runs.
TABLE_13 = """
frame        0   1   2   3   4   5   6   7   8   9  10  11  12
N=2 run      0   0   2   0   0   2   2   0   0   2   2   0   0
N=2 place   -1   0  -1   2   3   2   3   2   3   2   3   2   3
N=4 run      0   1   2   3   0   1   2   3   0   1   2   3   0
N=4 place   -1  -1  -1  -1   3   3   3   3   3   3   3   3   3
"""


def test_the_runs_of_a_shot_of_13_frames() -> None:
    rows = {
        line[:10].strip(): [int(cell) for cell in line[10:].split()]
        for line in TABLE_13.strip().splitlines()[1:]
    }
    for runs in (2, 4):
        taken = [job.run_of(runs, frame) for frame in range(13)]
        assert taken == rows[f"N={runs} run"]
        assert [place(frame - r) for frame, r in enumerate(taken)] == rows[f"N={runs} place"]
        assert [job.place(frame - r) for frame, r in enumerate(taken)] == rows[f"N={runs} place"]


# --- The manifest -------------------------------------------------------------------------------


def source(tmp_path: Path, frames: int) -> Source:
    stand_in = SimpleNamespace(
        path=tmp_path / "in.mkv",
        frames=frames,
        stream=SimpleNamespace(
            frame_rate=Fraction(25),
            width=64,
            height=48,
            pix_fmt="yuv420p",
            color_primaries="",
            color_transfer="",
        ),
        sample_aspect=Fraction(1),
        conversion=SimpleNamespace(describe=lambda: "RGB"),
        sha256=None,
        index=None,
        idet=None,
    )
    stand_in.path.write_bytes(b"")
    return cast(Source, stand_in)


def job_of(
    tmp_path: Path,
    lengths: list[int],
    runs: int | None,
    *,
    window: int = 5,
    split: bool = False,
    segments: list[int] | None = None,
) -> Manifest:
    """A manifest of consecutive shots of `lengths` frames, `runs` runs a shot (None: the setting
    absent), the shots in the one segment or in segments of `segments` shots each."""
    starts = [sum(lengths[:i]) for i in range(len(lengths))]
    shots = [Shot(start, start + length) for start, length in zip(starts, lengths, strict=True)]
    settings: dict[str, Any] = {"seed": 42, "window": window}
    if runs is not None:
        settings["quality"] = LEVELS[runs]
    if split:
        settings["color_correction"] = "split"
    count = 1 if runs is None else runs
    layouts = [shot_layout(r.frames, window) for shot in shots for r in shot_runs(shot, count)]
    cuts = [0, len(shots)] if segments is None else segments
    parts = [
        OutputSegment(f"s{i}", shots[a].start, shots[b - 1].end)
        for i, (a, b) in enumerate(pairwise(cuts))
    ]
    for part in parts:
        (tmp_path / f"{part.name}.mkv").write_bytes(b"x" * part.frames)
    return Manifest(
        tmp_path / "manifest.json",
        settings,
        {"gpu": "a GPU"},
        source(tmp_path, shots[-1].end),
        shots,
        layouts,
        parts,
        [f"{part.name}.mkv" for part in parts],
        {},
        Plan(window).record(),
    )


LENGTHS_OF_SHOTS = [1, 2, 3, 9]
SHOT_KEYS = ["start", "end", "seed", "latents", "windows", "encoded", "windows_done"]
RUN_KEYS = ["from", "latents", "windows", "encoded", "windows_done"]


def latents_of(frames: int) -> int:
    return 1 + (padded_length(frames) - 1) // 4


@pytest.mark.parametrize("runs", [1, None])
def test_one_run_a_shot_is_the_record_as_it_was(tmp_path: Path, runs: int | None) -> None:
    record = job_of(tmp_path, LENGTHS_OF_SHOTS, runs)
    assert record.runs_per_shot == 1
    assert len(record.runs) == 4 and record.run_shots == [0, 1, 2, 3]
    for entry in record.content()["shots"]:
        assert list(entry) == SHOT_KEYS
    assert [entry["latents"] for entry in record.content()["shots"]] == [1, 2, 2, 3]
    assert [record.shot_runs(i) for i in range(4)] == [[0], [1], [2], [3]]


@pytest.mark.parametrize(
    ("runs", "offsets"),
    [(4, [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3]]), (2, [[0], [0], [0, 2], [0, 2]])],
)
def test_the_later_runs_are_recorded_in_their_shot(
    tmp_path: Path, runs: int, offsets: list[list[int]]
) -> None:
    record = job_of(tmp_path, LENGTHS_OF_SHOTS, runs)
    shots = record.content()["shots"]
    for index, (entry, shot) in enumerate(zip(shots, record.shots, strict=True)):
        assert list(entry) == [*SHOT_KEYS, "runs"]
        assert [offset for offset in offsets[index]][1:] == [r["from"] for r in entry["runs"]]
        # The shot's own fields are its first run's, the whole shot's.
        assert entry["latents"] == latents_of(shot.frames)
        for later in entry["runs"]:
            assert list(later) == RUN_KEYS
            frames = shot.frames - later["from"]
            assert later["latents"] == latents_of(frames)
            assert later["windows"] == [list(w) for w in shot_layout(frames, 5)]
            assert later["encoded"] is False and later["windows_done"] == 0
    assert [len(entry["runs"]) for entry in shots] == [len(o) - 1 for o in offsets]
    # Runs, their shots, and a shot's runs agree.
    assert [r.offset for r in record.runs] == [o for shot in offsets for o in shot]
    assert record.run_shots == [i for i, o in enumerate(offsets) for _ in o]
    for index in range(4):
        assert [record.shot_of(r) for r in record.shot_runs(index)] == [index] * len(offsets[index])
    assert sum(len(o) for o in offsets) == len(record.layouts) == len(record.encoded)
    assert len(record.windows_done) == len(record.runs)


def test_progress_is_recorded_run_by_run(tmp_path: Path) -> None:
    record = job_of(tmp_path, LENGTHS_OF_SHOTS, 4)
    # Runs 6 to 9 are the 9-frame shot's: the last one is its run from frame 3.
    last = record.shot_runs(3)[3]
    assert (last, record.runs[last].offset) == (9, 3)
    record.shot_encoded(last)
    record.window_done(last, 0)
    shots = manifest.read(record.path)["shots"]
    assert shots[3]["runs"][2]["encoded"] is True and shots[3]["runs"][2]["windows_done"] == 1
    assert shots[3]["runs"][0]["encoded"] is False
    assert shots[3]["encoded"] is False and shots[3]["windows_done"] == 0
    assert all(not entry["encoded"] for entry in shots[:3])
    assert record.content()["shots"] == shots
    with pytest.raises(ValueError, match="run 9: window 3 after 1"):
        record.window_done(last, 3)


def test_planned_looks_at_every_run(tmp_path: Path) -> None:
    record = job_of(tmp_path, LENGTHS_OF_SHOTS, 4)
    assert manifest.planned(record.content())
    assert not manifest.complete(record.content())
    # A later run's encode, or window, is a unit made: no longer a plan.
    encoded = job_of(tmp_path, LENGTHS_OF_SHOTS, 4)
    encoded.shot_encoded(8)
    assert not manifest.planned(encoded.content())
    windowed = job_of(tmp_path, LENGTHS_OF_SHOTS, 4)
    windowed.window_done(8, 0)
    assert not manifest.planned(windowed.content())
    # complete() looks at the segments alone.
    record.shot_encoded(8)
    assert not manifest.complete(record.content())
    record.segment_finished(0, 10)
    assert manifest.complete(record.content())


def test_the_level_is_checked(tmp_path: Path) -> None:
    with pytest.raises(JobError, match='quality "best", where seedvr2x has standard, enhanced and'):
        Manifest(
            tmp_path / "manifest.json",
            {"seed": 42, "quality": "best"},
            {},
            source(tmp_path, 9),
            [Shot(0, 9)],
            [shot_layout(9, 5)],
            [OutputSegment("a", 0, 9)],
            ["a.mkv"],
            {},
            Plan(5).record(),
        )
    # One layout where the runs are two.
    with pytest.raises(ValueError, match="1 layouts for 2 runs"):
        Manifest(
            tmp_path / "manifest.json",
            {"seed": 42, "quality": "enhanced"},
            {},
            source(tmp_path, 9),
            [Shot(0, 9)],
            [shot_layout(9, 5)],
            [OutputSegment("a", 0, 9)],
            ["a.mkv"],
            {},
            Plan(5).record(),
        )


# --- What a resume reads -------------------------------------------------------------------------


def test_the_record_round_trips(tmp_path: Path) -> None:
    record = job_of(tmp_path, LENGTHS_OF_SHOTS, 4)
    for number in (0, 2, 6, 8, 9):
        record.shot_encoded(number)
    record.window_done(9, 0)
    content = json.loads(json.dumps(record.content()))
    back = resume.recorded_job(tmp_path / "manifest.json", content, record.source)
    assert back.runs == record.runs and back.run_shots == record.run_shots
    assert [list(layout) for layout in back.layouts] == [list(layout) for layout in record.layouts]
    assert back.encoded == record.encoded and back.windows_done == record.windows_done
    assert back.content() == content


def test_adopt_refuses_another_count_of_runs(tmp_path: Path) -> None:
    four = job_of(tmp_path, LENGTHS_OF_SHOTS, 4)
    for asked in (2, 1):
        other = job_of(tmp_path, LENGTHS_OF_SHOTS, asked)
        with pytest.raises(JobError, match="runs recorded"):
            resume.adopt(other, four.content())
        with pytest.raises(JobError, match="runs recorded"):
            resume.adopt(four, other.content())
    resume.adopt(four, four.content())


def test_only_how_far_a_later_run_went_is_not_compared(tmp_path: Path) -> None:
    recorded = job_of(tmp_path, LENGTHS_OF_SHOTS, 4)
    asked = job_of(tmp_path, LENGTHS_OF_SHOTS, 4)
    for number in (0, 7, 9):
        asked.shot_encoded(number)
    asked.window_done(9, 0)
    assert resume.differences(recorded.content(), asked.content()) == []
    assert resume.differences(asked.content(), recorded.content()) == []
    # A later run's windows are compared, as a shot's are.
    other = asked.content()
    other["shots"][3]["runs"][1]["windows"] = [[0, 2]]
    assert resume.differences(recorded.content(), other) == [
        "shots[3].runs[1].windows[0][1]: 3 -> 2"
    ]


def test_a_resume_with_another_count_lists_the_setting(tmp_path: Path) -> None:
    four = job_of(tmp_path, LENGTHS_OF_SHOTS, 4)
    one = job_of(tmp_path, LENGTHS_OF_SHOTS, 1)
    found = resume.differences(four.content(), one.content())
    assert 'settings.quality: "maximum" -> "standard"' in found
    assert resume.setting('settings.quality: "maximum" -> "standard"') == "quality"


def test_an_environment_change_counts_every_runs_units(tmp_path: Path) -> None:
    record = job_of(tmp_path, LENGTHS_OF_SHOTS, 4)
    for number in (0, 1, 8, 9):
        record.shot_encoded(number)
    record.window_done(9, 0)
    record.window_done(9, 1)
    record.window_done(1, 0)
    change = resume.environment_change(record.content(), {"gpu": "another GPU"})
    assert change["shots_encoded"] == 4
    assert change["windows_done"] == 3
    assert change["segments_finished"] == 0
    assert change["before"] == {"gpu": "a GPU"} and change["after"] == {"gpu": "another GPU"}


@pytest.mark.parametrize(
    "name",
    [
        "latent.pt",
        "window_0000.pt",
        "run1_latent.pt",
        "run2_latent.pt",
        "run3_window_0012.pt",
        "runs.kept",
        "input.mkv",
        "input.crc32",
    ],
)
def test_unit_files_a_resume_discards(name: str) -> None:
    assert resume.UNIT_FILE.fullmatch(name)


@pytest.mark.parametrize(
    "name",
    ["run4_latent.pt", "run0_latent.pt", "run1_input.mkv", "run1_runs.kept", "run1_window_1.pt"],
)
def test_files_that_are_not_units(name: str) -> None:
    assert not resume.UNIT_FILE.fullmatch(name)


def test_kept_names_the_files_of_every_run(tmp_path: Path) -> None:
    # One shot of 29 frames: every run pads to 29, 8 latents, in 2 windows of 5.
    record = job_of(tmp_path, [29], 4, split=True)
    assert all(len(layout) == 2 for layout in record.layouts)
    assert resume._kept(record, 0) == set()  # pyright: ignore[reportPrivateUsage]
    record.encoded[0] = True  # run 0: encoded, no window yet
    record.encoded[1], record.windows_done[1] = True, 1  # run 1: half done
    record.encoded[2], record.windows_done[2] = True, 2  # run 2: done, its latent given up
    assert resume._kept(record, 0) == {  # pyright: ignore[reportPrivateUsage]
        "latent.pt",
        "input.mkv",
        "input.crc32",
        "run1_latent.pt",
        "run1_window_0000.pt",
        "run2_window_0000.pt",
        "run2_window_0001.pt",
    }


def test_kept_has_no_copy_without_split(tmp_path: Path) -> None:
    record = job_of(tmp_path, [9], 2, window=5)
    record.encoded[1] = True  # the run from frame 2 alone: the copy is its first run's
    assert resume._kept(record, 0) == {"run2_latent.pt"}  # pyright: ignore[reportPrivateUsage]


# --- The units ----------------------------------------------------------------------------------


def tensor(latents: int = 3, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(latents, 2, 2, 16, generator=generator).to(torch.bfloat16)


def test_the_name_of_a_run_s_file() -> None:
    assert run_file(0, "latent.pt") == "latent.pt"
    assert run_file(2, "window_0001.pt") == "run2_window_0001.pt"
    assert run_file(3, "latent.pt") == "run3_latent.pt"


def files(directory: Path) -> list[str]:
    return sorted(entry.name for entry in directory.iterdir())


def test_a_later_run_keeps_its_units_beside_the_first_run_s(tmp_path: Path) -> None:
    # Two shots of 9 frames, 4 runs each: runs 0 to 3 of the first, 4 to 7 of the second.
    record = job_of(tmp_path, [9, 9], 4, segments=[0, 1, 2])
    units = Units(tmp_path, record)
    first, later, other = tensor(seed=1), tensor(seed=2), tensor(seed=3)
    units.save_latent(0, first)
    units.save_latent(2, later)
    units.save_latent(5, other)  # the second shot's run from frame 1
    directory = tmp_path / "resume" / "shot_000000"
    assert units.shot_directory(2) == directory == units.shot_directory(0)
    assert units.shot_directory(5) == tmp_path / "resume" / "shot_000009"
    assert files(directory) == ["latent.pt", "run2_latent.pt"]
    assert files(units.shot_directory(5)) == ["run1_latent.pt"]
    back = units.latent(2)
    assert back is not None and torch.equal(back, later)
    back = units.latent(0)
    assert back is not None and torch.equal(back, first)
    assert units.latent(1) is None and units.latent(3) is None  # not encoded
    assert record.encoded == [True, False, True, False, False, True, False, False]

    windows = [tensor(2, seed=10), tensor(2, seed=11)]
    for number, window in enumerate(windows):
        units.save_window(2, number, window)
    assert files(directory) == [
        "latent.pt",
        "run2_latent.pt",
        "run2_window_0000.pt",
        "run2_window_0001.pt",
    ]
    assert record.windows_done[2] == 2 and record.windows_done[0] == 0
    taken = units.take_windows(2)
    assert len(taken) == 2 and all(torch.equal(a, b) for a, b in zip(taken, windows, strict=True))
    assert units.take_windows(0) == []
    units.drop_latent(2)
    assert files(directory) == ["latent.pt", "run2_window_0000.pt", "run2_window_0001.pt"]
    units.drop_latent(2)  # already gone: fine


def test_the_shot_s_files_are_the_shot_s_for_every_run(tmp_path: Path) -> None:
    record = job_of(tmp_path, [9, 9], 4, segments=[0, 1, 2])
    units = Units(tmp_path, record)
    directory = tmp_path / "resume" / "shot_000009"
    for number in range(4, 8):
        assert units.copy_path(number) == directory / "input.mkv"
        assert units.checksums_path(number) == directory / "input.crc32"
        assert units.kept_path(number) == directory / KEPT == directory / "runs.kept"


def test_a_finished_segment_removes_every_run_s_files(tmp_path: Path) -> None:
    record = job_of(tmp_path, [9, 9], 4, segments=[0, 1, 2])
    units = Units(tmp_path, record)
    for number in (0, 1, 3, 4, 7):
        units.save_latent(number, tensor(seed=number))
        units.save_window(number, 0, tensor(2, seed=number))
    for number in (0, 4):
        units.kept_path(number).write_bytes(b"frames")
    first, second = tmp_path / "resume" / "shot_000000", tmp_path / "resume" / "shot_000009"
    assert files(first) == [
        "latent.pt",
        "run1_latent.pt",
        "run1_window_0000.pt",
        "run3_latent.pt",
        "run3_window_0000.pt",
        "runs.kept",
        "window_0000.pt",
    ]
    units.segment_finished(0)
    assert not first.exists() and second.is_dir()
    units.segment_finished(1)
    assert not (tmp_path / "resume").exists()


# --- The frames the runs keep -------------------------------------------------------------------


def frames_of(*values: int) -> npt.NDArray[np.float32]:
    """Frames (n, 2, 3, 3) float32, each filled with its value / 10, in [0, 1]."""
    return np.stack([np.full((2, 3, 3), value / 10, dtype=np.float32) for value in values])


def sixteen(part: npt.NDArray[np.float32] | npt.NDArray[np.uint16]) -> npt.NDArray[np.uint16]:
    """What a writer writes of frames written: a float frame's quantised values, a kept frame's
    own (writer.quantise)."""
    if part.dtype == np.uint16:
        return part.astype(np.uint16, copy=False)
    return np.stack([quantise(frame.astype(np.float32, copy=False)) for frame in part])


def test_kept_frames_are_allocated_whole_and_written_in_order(tmp_path: Path) -> None:
    path = tmp_path / "shot" / KEPT
    written: list[npt.NDArray[np.float32] | npt.NDArray[np.uint16]] = []
    with run.Kept(path, [1, 2, 4], (2, 3, 3)) as store:
        # Three frames of 2 x 3 x 3 samples, at the 16 bits the writers quantise them to.
        assert path.stat().st_size == 3 * 2 * 3 * 3 * 2
        assert not store.pending(1)
        for frame in (4, 1, 2):  # in any order they are put, only the writes are in the shot's
            store.put(frame, frames_of(frame)[0])
        assert store.pending(2) and store.pending(5)
        store.flush(1, written.append)
        assert written == []
        store.flush(2, written.append)
        assert not store.pending(2) and store.pending(5)
        store.flush(5, written.append)
        assert not store.pending(100)
        store.flush(100, written.append)  # nothing left: no empty write
    assert [part.shape for part in written] == [(1, 2, 3, 3), (2, 2, 3, 3)]
    # Kept and written as the writers' values: a float frame's, quantised.
    assert all(part.dtype == np.uint16 for part in written)
    assert np.array_equal(np.concatenate(written), sixteen(frames_of(1, 2, 4)))
    assert not path.exists()


def test_kept_frames_are_removed_when_the_block_raises(tmp_path: Path) -> None:
    path = tmp_path / KEPT
    with pytest.raises(KeyError, match="decode"):
        with run.Kept(path, [1, 2], (2, 3, 3)) as store:
            store.put(1, frames_of(1)[0])
            assert path.exists()
            raise KeyError("decode")
    assert not path.exists()


def test_kept_frames_never_written_are_an_error(tmp_path: Path) -> None:
    path = tmp_path / KEPT
    with pytest.raises(RuntimeError, match="1 kept frames never written"):
        with run.Kept(path, [1, 2], (2, 3, 3)) as store:
            store.put(1, frames_of(1)[0])
            store.flush(2, lambda _: None)
    # Removed all the same: derived data, never left behind by the block.
    assert not path.exists()


def test_no_kept_frames_make_no_file(tmp_path: Path) -> None:
    path = tmp_path / KEPT
    with run.Kept(path, [], (2, 3, 3)) as store:
        assert not path.exists()
        assert not store.pending(10)
    assert not path.exists()


def test_a_full_disk_is_a_scratch_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / KEPT

    def full(descriptor: int, offset: int, length: int) -> None:
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    monkeypatch.setattr(os, "posix_fallocate", full)
    with pytest.raises(ScratchError, match=str(path)):
        with run.Kept(path, [1, 2], (2, 3, 3)):
            pytest.fail("entered")
    assert not path.exists()


def decoded(offset: int, length: int, chunk: int) -> list[npt.NDArray[np.float32]]:
    """What the decode of the run from frame `offset` of a shot of `length` frames gives, in
    chunks of `chunk` frames: frames (n, 1, 1, 3) whose first value is the frame's index in the
    shot over 100 and the second the run's offset over 10, so that where a frame comes from is
    read in it, through the 16 bits of a kept frame too."""
    frames = np.array(
        [[[[index / 100, offset / 10, 0]]] for index in range(offset, length)], dtype=np.float32
    ).reshape(-1, 1, 1, 3)
    return [frames[start : start + chunk] for start in range(0, len(frames), chunk)]


@pytest.mark.parametrize("chunk", [1, 2, 5])
@pytest.mark.parametrize("length", [1, 2, 3, 5, 9, 13])
@pytest.mark.parametrize("runs", [2, 4])
def test_the_frames_come_out_in_order_each_from_its_run(
    tmp_path: Path, runs: int, length: int, chunk: int
) -> None:
    # run._decode_runs without the decode: each run but the last puts its frames in the store,
    # the last writes its own with the kept ones between.
    all_runs = shot_runs(Shot(0, length), runs)
    taken = [job.run_of(runs, frame) for frame in range(length)]
    last = all_runs[-1].offset
    kept = [frame for frame in range(length) if taken[frame] != last]
    written: list[npt.NDArray[np.float32] | npt.NDArray[np.uint16]] = []
    with run.Kept(tmp_path / KEPT, kept, (1, 1, 3)) as store:
        for each in all_runs[:-1]:
            give = run._given(each, taken, store.put)  # pyright: ignore[reportPrivateUsage]
            for part in decoded(each.offset, length, chunk):
                give(part)
        emit = run._interleaved(  # pyright: ignore[reportPrivateUsage]
            all_runs[-1], taken, store, written.append
        )
        for part in decoded(last, length, chunk):
            emit(part)
        store.flush(length, written.append)
    assert not (tmp_path / KEPT).exists()
    # The last run's frames come as the decode gives them, float32; the kept ones as uint16.
    assert {part.dtype for part in written} <= {np.dtype(np.float32), np.dtype(np.uint16)}
    parts = [sixteen(part) for part in written]
    out = np.concatenate(parts) if parts else np.empty((0, 1, 1, 3), np.uint16)
    assert out.shape == (length, 1, 1, 3)
    assert [round(int(v) / 65535 * 100) for v in out[:, 0, 0, 0]] == list(range(length))
    assert [round(int(v) / 65535 * 10) for v in out[:, 0, 0, 1]] == taken
    assert all(part.shape[0] > 0 for part in written)


# --- The frames kept on disk -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("frames", "runs", "kept"),
    [
        # 4 a shot: the last run, from frame 3, gives frames 3, 7, 11, 15 and 19 of 21.
        (21, 4, 16),
        # 2 a shot: the last, from frame 2, gives 2, 5, 6, 9, 10, 13, 14, 17 and 18.
        (21, 2, 12),
        (1, 4, 0),
        (21, 1, 0),
        (1000, 4, 750),
    ],
)
def test_the_frames_the_earlier_runs_keep(frames: int, runs: int, kept: int) -> None:
    assert job.kept_frames(Shot(0, frames), runs) == kept


def test_the_plan_says_the_disk_the_kept_frames_take() -> None:
    # A shot of 1,000 frames at 1080p, 4 runs: 750 frames kept, 12,441,600 bytes each at 16 bits
    # (1920 x 1080 x 3 x 2, DESIGN.md's 12.4 MB), 8.69 GiB; the shorter shot keeps less.
    said = planner._runs_said([Shot(0, 1000), Shot(1000, 1100)], (1080, 1920), "maximum")  # pyright: ignore[reportPrivateUsage]
    assert "8 runs in all" in said
    assert "the longest shot keeps 8.69 GiB of frames on disk" in said


def test_the_kept_bytes_are_design_s() -> None:
    # DESIGN.md, Several runs of a shot: 12.4 MB a 1080p frame, 49.8 MB at 4K, 37 GB for 750
    # frames of a 1,000-frame 4K shot at four runs.
    assert job.kept_bytes(Shot(0, 2), 4, (1080, 1920)) == 12_441_600  # frame 0 kept
    assert job.kept_bytes(Shot(0, 2), 4, (2160, 3840)) == 49_766_400
    assert job.kept_bytes(Shot(0, 1000), 4, (2160, 3840)) == 750 * 49_766_400  # 37.3 GB

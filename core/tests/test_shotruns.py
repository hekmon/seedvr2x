"""The runs of a shot and the frames each keeps (seedvr2x/shotruns.py; DESIGN.md, Several runs of a
shot), on the CPU, plain Python: the levels and their runs, each run's first frame, the run each
frame is taken from written out apart from the module (the closed forms of DESIGN.md: with 4 runs
the run frame mod 4, with 2 the run where the frame is third or fourth of its group), a short shot
having fewer runs, the frames kept between runs and their bytes, and the module importing no torch.
The rule held to measurement's own composer is the CLI's, apps/seedvr2x/tests/test_runs.py."""

import subprocess
import sys

import pytest

from seedvr2x import shotruns
from seedvr2x.shotruns import QUALITY, RUN_OFFSETS, Run, kept_bytes, kept_frames, run_of, shot_runs
from seedvr2x.shots import Shot


def test_the_levels_and_their_runs() -> None:
    assert QUALITY == {"standard": 1, "enhanced": 2, "maximum": 4}
    assert RUN_OFFSETS == {1: (0,), 2: (0, 2), 4: (0, 1, 2, 3)}
    assert set(RUN_OFFSETS) == set(QUALITY.values())


def test_a_runs_place_counts_from_its_first_frame() -> None:
    assert [shotruns.place(i) for i in range(10)] == [-1, 0, 1, 2, 3, 0, 1, 2, 3, 0]
    with pytest.raises(ValueError):
        shotruns.place(-1)


@pytest.mark.parametrize("frame", range(40))
def test_a_frame_is_taken_from_the_run_the_closed_forms_name(frame: int) -> None:
    assert run_of(1, frame) == 0
    # 4 runs: the run frame mod 4, where the frame ends a group of 4 or is the run's first.
    assert run_of(4, frame) == frame % 4
    # 2 runs: frames 0 and 1 from run 0 (run 2 not begun), after that the run where the frame's
    # place is the better, the places ranked 3 (a group's last), 2, 0, 1, a run's own first
    # frame first of all.
    if frame < 2:
        assert run_of(2, frame) == 0
    else:
        places = {offset: shotruns.place(frame - offset) for offset in (0, 2)}
        best = {-1: 4, 3: 3, 2: 2, 0: 1, 1: 0}
        assert run_of(2, frame) == max(places, key=lambda offset: (best[places[offset]], -offset))
    with pytest.raises(ValueError):
        run_of(4, -1)


def test_every_frame_is_taken_from_a_run_that_holds_it() -> None:
    for runs in (1, 2, 4):
        for frame in range(40):
            assert run_of(runs, frame) <= frame


def test_taken_from_is_run_of_a_frame_at_a_time() -> None:
    for runs in (1, 2, 4):
        assert shotruns.taken_from(13, runs) == [run_of(runs, frame) for frame in range(13)]
    assert shotruns.taken_from(5, 4) == [0, 1, 2, 3, 0]
    assert shotruns.taken_from(9, 2) == [0, 0, 2, 0, 0, 2, 2, 0, 0]


def test_a_shot_has_the_runs_whose_first_frame_it_has() -> None:
    shot = Shot(100, 160)
    assert shot_runs(shot, 1) == [Run(shot, 0)]
    assert [run.offset for run in shot_runs(shot, 4)] == [0, 1, 2, 3]
    run = shot_runs(shot, 4)[2]
    assert (run.start, run.end, run.frames) == (102, 160, 58)
    # Every run has the shot's seed (Run.seed), measurement's composite.
    assert {each.seed(42) for each in shot_runs(shot, 4)} == {142}
    # A shot too short to have a run's first frame has no such run.
    assert [run.offset for run in shot_runs(Shot(0, 1), 4)] == [0]
    assert [run.offset for run in shot_runs(Shot(0, 2), 4)] == [0, 1]
    assert [run.offset for run in shot_runs(Shot(0, 3), 2)] == [0, 2]
    assert [run.offset for run in shot_runs(Shot(0, 2), 2)] == [0]


def test_the_frames_kept_between_runs_and_their_bytes() -> None:
    # None with one run; else every frame the last run does not give.
    assert kept_frames(Shot(0, 50), 1) == 0
    for runs in (2, 4):
        shot = Shot(7, 7 + 61)
        last = shot_runs(shot, runs)[-1].offset
        assert kept_frames(shot, runs) == sum(
            run_of(runs, frame) != last for frame in range(shot.frames)
        )
    assert kept_frames(Shot(0, 1), 4) == 0
    # 16 bits a sample: 12.4 MB a 1080p frame, 49.8 MB at 4K.
    assert kept_bytes(Shot(0, 2), 4, (1080, 1920)) == 12_441_600  # frame 0 kept
    assert kept_bytes(Shot(0, 2), 4, (2160, 3840)) == 49_766_400
    assert shotruns.KEPT_SAMPLE_BYTES == 2


def test_the_module_imports_no_torch() -> None:
    code = "import sys; import seedvr2x.shotruns; sys.exit('torch' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0

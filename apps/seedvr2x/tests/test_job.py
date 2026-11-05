"""A job's layout: cut lists, shots and their seeds."""

from pathlib import Path

import pytest

from job import read_cuts, shots_from_cuts
from seedvr2x.errors import JobError
from seedvr2x.shots import Shot


def test_cut_list(tmp_path: Path) -> None:
    path = tmp_path / "cuts.txt"
    path.write_text("# shots of the trailer\n120\n\n 250  # a flash before\n300\n")
    assert read_cuts(path) == [120, 250, 300]
    path.write_text("120\n12.5\n")
    with pytest.raises(JobError, match=r"cuts.txt:2: '12.5' is not a frame number"):
        read_cuts(path)
    # Fields after the frame number are ignored (DESIGN.md, Input): scores, say.
    path.write_text("120 13.7\n250\t4.2 flash # doubtful\n300 \n")
    assert read_cuts(path) == [120, 250, 300]
    path.write_text("00:00:05.005 120\n")
    with pytest.raises(JobError, match=r"cuts.txt:1: '00:00:05.005' is not a frame number"):
        read_cuts(path)
    with pytest.raises(JobError, match=r"missing\.txt"):
        read_cuts(tmp_path / "missing.txt")


def test_shots_from_cuts() -> None:
    assert shots_from_cuts([], 45) == [Shot(0, 45)]
    shots = shots_from_cuts([10, 11, 30], 45)
    assert shots == [Shot(0, 10), Shot(10, 11), Shot(11, 30), Shot(30, 45)]
    assert [shot.frames for shot in shots] == [10, 1, 19, 15]
    for cuts, message in (
        ([0, 10], "frame 0"),
        ([10, 10], "must increase"),
        ([20, 10], "must increase"),
        ([10, 45], "has 45 frames"),
    ):
        with pytest.raises(JobError, match=message):
            shots_from_cuts(cuts, 45)


def test_seed_per_shot() -> None:
    # The plain seed at frame 0, as numz's single batch; then the seed plus the first frame.
    assert [shot.seed(42) for shot in shots_from_cuts([10, 11], 20)] == [42, 52, 53]

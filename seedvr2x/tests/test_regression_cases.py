"""What a card other than the 96 GB one takes of each regression case from its cases.json
(tests/test_regression.py, case), on the CPU: a tiled case at its own size, with what its debug log
says of each tiled pass there; at 1080 and the test's own grids where the card says nothing."""

import json
from pathlib import Path

import pytest
import test_regression


def test_a_tiled_case_takes_the_card_s_size_and_grids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(test_regression, "REFERENCE", str(tmp_path))
    monkeypatch.setattr(test_regression, "card", lambda: "NVIDIA GeForce RTX 5090")
    directory = tmp_path / "cards" / "NVIDIA-GeForce-RTX-5090"
    directory.mkdir(parents=True)
    grids = ["tiled decode: 2 tiles of 1024 px sharing 64, 2 across and 1 down,"]
    over = "over 45 frames padded to 1536x864; its scratch, 358318080 bytes, in system memory"
    said = {
        "tiled": {
            "d1024": {
                "reference": "seedvr2x_d1024-864",
                "resolution": 864,
                "grids": grids,
                "over": over,
            },
            "e512-d512": {"reference": "seedvr2x_e512-d512"},
        }
    }
    (directory / "cases.json").write_text(json.dumps(said))
    job = test_regression.case("tiled", "d1024", "input_rgb.mkv", 1080, "numz_d1024", "m3")
    assert job == test_regression.Case(
        tmp_path / "m1" / "input_rgb.mkv",
        864,
        directory / "m3" / "seedvr2x_d1024-864",
        tuple(grids),
        over,
    )
    # Nothing said of the size: at 1080, the test's own grids.
    job = test_regression.case("tiled", "e512-d512", "input_rgb.mkv", 1080, "numz_e512-d512", "m3")
    assert (job.resolution, job.grids, job.over) == (1080, None, None)
    assert job.reference == directory / "m3" / "seedvr2x_e512-d512"
    # On the 96 GB card, the test's parameters alone.
    monkeypatch.setattr(test_regression, "card", lambda: test_regression.ROOT_CARD)
    job = test_regression.case("tiled", "d1024", "input_rgb.mkv", 1080, "numz_d1024", "m3")
    assert job == test_regression.Case(
        tmp_path / "m1" / "input_rgb.mkv", 1080, tmp_path / "m3" / "numz_d1024"
    )

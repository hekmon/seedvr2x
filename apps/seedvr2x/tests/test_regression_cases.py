"""What a card takes of each regression case from its directory of references
(tests/test_regression.py, case), on the CPU: the directory named by the card's name, a card with
none skipped; a tiled case at its own size, with what its debug log says of each tiled pass there;
milestone 1's and a padding case's resolution; at 1080 and the test's own grids where the card says
nothing."""

import json
from pathlib import Path

import pytest
import test_regression

CARD = "NVIDIA GeForce RTX 5090"


def card_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """tmp_path as SEEDVR2X_REFERENCE_DIR for the card CARD, whose directory is made."""
    monkeypatch.setattr(test_regression, "REFERENCE", str(tmp_path))
    monkeypatch.setattr(test_regression, "card", lambda: CARD)
    directory = tmp_path / "seedvr2x" / "NVIDIA-GeForce-RTX-5090"
    directory.mkdir(parents=True)
    return directory


def test_a_tiled_case_takes_the_card_s_size_and_grids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = card_in(tmp_path, monkeypatch)
    assert test_regression.card_directory(CARD) == directory
    grids = ["tiled decode: 2 tiles of 1024 px sharing 64, 2 across and 1 down,"]
    over = "over 45 frames padded to 1536x864; its scratch, 358318080 bytes, in system memory"
    said = {"tiled": {"d1024": {"resolution": 864, "grids": grids, "over": over}}}
    (directory / "cases.json").write_text(json.dumps(said))
    job = test_regression.case("tiled", "input_rgb.mkv", 1080, "d1024")
    assert job == test_regression.Case(
        tmp_path / "m1" / "input_rgb.mkv", 864, directory / "d1024", tuple(grids), over
    )
    # Nothing said of the size: at 1080, the test's own grids.
    job = test_regression.case("tiled", "input_rgb.mkv", 1080, "e512-d512")
    assert (job.resolution, job.grids, job.over) == (1080, None, None)
    assert job.reference == directory / "e512-d512"


def test_milestone_1_and_padding_cases_take_the_card_s_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = card_in(tmp_path, monkeypatch)
    said = {"milestone1": {"resolution": 720}, "padding": {"pad1080": {"resolution": 648}}}
    (directory / "cases.json").write_text(json.dumps(said))
    for reference in ("7b", "sharp"):
        job = test_regression.case("milestone1", "input_rgb.mkv", 1080, reference)
        assert job.resolution == 720 and job.reference == directory / reference
    assert test_regression.case("padding", "input_rgb.mkv", 1080, "pad1080").resolution == 648
    # A case the file doesn't name, the source the test's own whatever the file says.
    job = test_regression.case("padding", "input_rgb.mkv", 720, "pad720")
    assert job.resolution == 720 and job.source == tmp_path / "m1" / "input_rgb.mkv"


def test_a_card_with_a_directory_and_no_cases_file_takes_the_test_s_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = card_in(tmp_path, monkeypatch)
    job = test_regression.case("milestone1", "input_rgb.mkv", 1080, "7b")
    assert job == test_regression.Case(tmp_path / "m1" / "input_rgb.mkv", 1080, directory / "7b")


def test_a_card_without_a_directory_skips_every_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(test_regression, "REFERENCE", str(tmp_path))
    monkeypatch.setattr(test_regression, "card", lambda: CARD)
    # Numz's references of the card stay on disk and count for nothing.
    (tmp_path / "cards" / "NVIDIA-GeForce-RTX-5090").mkdir(parents=True)
    with pytest.raises(pytest.skip.Exception) as skipped:
        test_regression.case("milestone1", "input_rgb.mkv", 1080, "7b")
    message = str(skipped.value)
    assert CARD in message and "docstring says how to make them" in message

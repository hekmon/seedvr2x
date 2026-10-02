"""A job's layout: cut lists, shots and their seeds, the output size."""

from fractions import Fraction
from pathlib import Path

import pytest
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF

from seedvr2x.runtime.job import (
    JobError,
    Shot,
    output_size,
    read_cuts,
    shots_from_cuts,
    target_size,
)
from seedvr2x.vendor.data.image.transforms.na_resize import NaResize


def test_cut_list(tmp_path: Path) -> None:
    path = tmp_path / "cuts.txt"
    path.write_text("# shots of the trailer\n120\n\n 250  # a flash before\n300\n")
    assert read_cuts(path) == [120, 250, 300]
    path.write_text("120\n12.5\n")
    with pytest.raises(JobError, match=r"cuts.txt:2: '12.5' is not a frame number"):
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


@pytest.mark.parametrize(
    ("width", "height", "resolution"),
    [(960, 540, 1080), (1920, 1080, 1080), (640, 480, 1080), (720, 1280, 1080), (100, 100, 64)],
)
def test_square_pixels_resize_as_naresize(width: int, height: int, resolution: int) -> None:
    # The explicit size is NaResize's own, and the resize to it is the same call, bit for bit.
    frames = torch.rand(2, 3, height, width)
    resize = NaResize(resolution=resolution, mode="side", downsample_only=False, max_resolution=0)
    theirs = resize(frames)
    target = target_size(width, height, Fraction(1), resolution)
    ours = TVF.resize(frames, list(target), InterpolationMode.BICUBIC, antialias=True)
    assert tuple(theirs.shape[-2:]) == target
    assert torch.equal(ours, theirs)


def test_square_pixels_sizes_as_torchvision() -> None:
    # torchvision computes the long side in floating point, int(size * long / short): the exact
    # fraction, rounded down, gives the same over every size up to 4K.
    for short in range(16, 2161, 7):
        for long in range(short, 3841, 11):
            assert target_size(long, short, Fraction(1), 1080)[1] == int(1080 * long / short)


@pytest.mark.parametrize(
    ("size", "sample_aspect", "expected"),
    [
        ((720, 480), Fraction(32, 27), (1080, 1920)),  # NTSC DVD, 16:9
        ((720, 480), Fraction(8, 9), (1080, 1440)),  # NTSC DVD, 4:3
        ((720, 576), Fraction(64, 45), (1080, 1920)),  # PAL DVD, 16:9
        ((720, 576), Fraction(16, 15), (1080, 1440)),  # PAL DVD, 4:3
        ((1440, 1080), Fraction(4, 3), (1080, 1920)),  # HDV
        ((480, 720), Fraction(32, 27), (1366, 1080)),  # portrait: the width is the short side
    ],
)
def test_display_aspect(
    size: tuple[int, int], sample_aspect: Fraction, expected: tuple[int, int]
) -> None:
    assert target_size(*size, sample_aspect, 1080) == expected


def test_even_output() -> None:
    assert output_size((1080, 1920)) == (1080, 1920)
    assert output_size((1215, 1080)) == (1214, 1080)
    assert output_size((541, 961)) == (540, 960)

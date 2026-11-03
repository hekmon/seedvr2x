"""The frame's sizes (frames.py): the picture a source is resized to, as NaResize and torchvision
do for square pixels, and the output's even size."""

from fractions import Fraction

import pytest
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF

from seedvr2x.frames import output_size, short_side, target_size
from seedvr2x.vendor.data.image.transforms.na_resize import NaResize


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


@pytest.mark.parametrize(
    ("size", "sample_aspect", "expected"),
    [
        ((4032, 3024), Fraction(1), 3024),  # a 12 MP photo
        ((3024, 4032), Fraction(1), 3024),  # the same, portrait
        ((1920, 1080), Fraction(1), 1080),
        ((2000, 2000), Fraction(1), 2000),  # square: the width is the short side
        ((720, 480), Fraction(32, 27), 480),  # NTSC DVD, 16:9: 853.3 x 480 displayed
        ((720, 576), Fraction(16, 15), 576),  # PAL DVD, 4:3
        ((480, 720), Fraction(32, 27), 569),  # portrait: 568.9 x 720 displayed, the width is short
        ((1000, 700), Fraction(1, 2), 500),  # anamorphic: 500 x 700 displayed
        ((3, 1), Fraction(1, 10), 1),  # never 0
    ],
)
def test_short_side_is_the_resolution_that_keeps_the_displayed_size(
    size: tuple[int, int], sample_aspect: Fraction, expected: int
) -> None:
    assert short_side(*size, sample_aspect) == expected


def test_target_size_at_the_short_side_is_the_displayed_size_within_a_pixel() -> None:
    for width, height, sample_aspect in [
        (4032, 3024, Fraction(1)),
        (3024, 4032, Fraction(1)),
        (4033, 3025, Fraction(1)),
        (720, 480, Fraction(32, 27)),
        (480, 720, Fraction(32, 27)),
        (1440, 1080, Fraction(4, 3)),
        (101, 200, Fraction(1)),
    ]:
        shown_width = width * sample_aspect
        side = short_side(width, height, sample_aspect)
        rows, columns = target_size(width, height, sample_aspect, side)
        assert min(rows, columns) == side
        # Exact where the displayed size is whole, else within the pixel the short side is rounded
        # to; floor, as target_size's long side is.
        assert abs(columns - shown_width) < 1.5 and abs(rows - height) <= 1
        if shown_width.denominator == 1:
            assert (rows, columns) == (height, int(shown_width))


def test_even_output() -> None:
    assert output_size((1080, 1920)) == (1080, 1920)
    assert output_size((1215, 1080)) == (1214, 1080)
    assert output_size((541, 961)) == (540, 960)

"""The frame's sizes (frames.py): the picture a source is resized to, as NaResize and torchvision
do for square pixels, and the output's even size."""

from fractions import Fraction

import pytest
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF

from seedvr2x.frames import factor_said, output_size, short_side, target_size
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


def test_the_refusal_of_a_target_too_small_carries_its_parts() -> None:
    from library_fixtures import cli_say, cli_words

    from seedvr2x.errors import Refusal, Setting, TargetTooSmall
    from seedvr2x.frames import check_target

    with pytest.raises(Refusal) as caught:
        check_target((16, 16), 16)
    error = caught.value
    # The command line words the resolution as its option, word for word; the library's own
    # words and the reason, for the image server, name no option, and the way is the setting
    # the message names.
    assert cli_words(error) == (
        "--resolution 16: the frames resized to 16x16 are too small for the padding, at least"
        " 8 rows reflected from the picture under it, up to a multiple of 16, and columns alike:"
        " it takes 17 rows and 16 columns at least"
    )
    assert isinstance(error, TargetTooSmall)
    assert error.kind == "too-small"
    assert (error.resolution, error.size, error.least) == (16, (16, 16), (17, 16))
    assert (
        error.way == "resolution 16" and cli_say(Setting("resolution", "16")) == "--resolution 16"
    )
    assert error.reason.startswith("the picture resized to 16x16 is too small for the padding")
    assert "--" not in error.reason and "resolution" not in error.reason
    assert "--" not in str(error) and str(error).startswith("resolution 16: the frames")
    assert (
        cli_words(error)
        == f"{cli_say(Setting('resolution', '16'))}: the frames resized to 16x16 are too small "
        + (error.reason.split("too small ", 1)[1])
    )


def test_a_picture_whose_even_crop_cannot_be_padded_is_refused_as_too_small() -> None:
    # Resolution 17 on a 40x24 picture: target_size gives 28x17, the output 28x16, under the 17
    # rows the padding takes. A picture's plan is made for the output (still_plan's bound refuses
    # it with a ValueError), so check_picture_target refuses it first, in the picture's words; a
    # video's frames are padded at the target's size and cropped after the decode (shot_pass), so
    # check_target passes it, as it always did.
    from library_fixtures import cli_words

    from seedvr2x.errors import Refusal, TargetTooSmall
    from seedvr2x.frames import check_picture_target, check_target

    target = target_size(40, 24, Fraction(1), 17)
    assert (target, output_size(target)) == ((17, 28), (16, 28))
    check_target(target, 17)
    with pytest.raises(Refusal) as caught:
        check_picture_target(target, 17)
    error = caught.value
    assert isinstance(error, TargetTooSmall)
    assert (error.kind, error.resolution, error.size, error.least) == (
        "too-small",
        17,
        (16, 28),
        (18, 16),
    )
    assert error.reason == (
        "the output 28x16, the 28x17 it is resized to cropped to even sides, is too small for the"
        " padding, at least 8 rows reflected from the picture under it, up to a multiple of 16,"
        " and columns alike: it takes 18 rows and 16 columns at least"
    )
    assert "--" not in error.reason and "resolution" not in error.reason
    assert cli_words(error).startswith("--resolution 17: the output 28x16, the 28x17 it is")
    # What check_target refuses stays refused in its own words; 18 rows and 17 columns (16 once
    # cropped) are padded.
    with pytest.raises(TargetTooSmall, match="the frames resized to 16x16"):
        check_picture_target((16, 16), 16)
    check_picture_target((18, 28), 18)
    check_picture_target((40, 17), 17)
    # Exactly the targets whose output the planner's bound takes (still_plan._paddable).
    from seedvr2x import still_plan

    for height in range(1, 41):
        for width in range(1, 41):
            try:
                check_picture_target((height, width), 7)
                passed = True
            except TargetTooSmall:
                passed = False
            assert passed == still_plan._paddable(output_size((height, width))), (height, width)


@pytest.mark.parametrize(
    ("source", "output", "said"),
    [
        ((320, 240), (1440, 1080), "x4.5"),
        ((1920, 1080), (3840, 2160), "x2"),
        ((542, 305), (1918, 1080), "x3.54"),  # 3.5409...
        ((1000, 1000), (1005, 1005), "x1.01"),  # 1.005 rounds half up
        ((1000, 1000), (1104, 1104), "x1.1"),  # trailing zero dropped
        ((240, 320), (1080, 1440), "x4.5"),  # portrait: the short sides
        ((1920, 1080), (1440, 810), "x0.75"),  # smaller
    ],
)
def test_the_factor_is_the_short_sides_ratio_at_two_decimals_at_most(
    source: tuple[int, int], output: tuple[int, int], said: str
) -> None:
    # The apps' plans say it beside the sizes (the user's third ComfyUI test, 2026-11-08).
    assert factor_said(source, output) == said

"""The frame's sizes (DESIGN.md, Colour and shape, Pipeline step 0): the picture a source is
resized to, the output's size, the padding before the model, and the VAE's input length, 4n + 1.
The library's (DESIGN.md, Repository and apps), plain Python, no torch: the planner needs them
before torch is imported."""

import math
from fractions import Fraction

from seedvr2x.errors import Refusal


def target_size(
    width: int, height: int, sample_aspect: Fraction, resolution: int
) -> tuple[int, int]:
    """(height, width) of the picture the input is resized to: the display picture's short side at
    resolution, the long side in proportion, rounded down. The output has square pixels at the
    source's display aspect (DESIGN.md, Colour and shape).

    For square pixels, this is the size NaResize (side mode) gives numz: torchvision's resize to an
    int size, the short side, the long one int(size * long / short), the width short when the
    picture is square (torchvision's _compute_resized_output_size)."""
    display_width = width * sample_aspect
    if display_width <= height:
        return math.floor(resolution * height / display_width), resolution
    return resolution, math.floor(resolution * display_width / height)


def output_size(target: tuple[int, int]) -> tuple[int, int]:
    """(height, width) of the output frames: the target, each side rounded down to an even
    number, as numz crops them (src/core/generation_utils.py:127-136)."""
    height, width = target
    return height - height % 2, width - width % 2


# The padding of the resized picture before the model (DESIGN.md, Pipeline step 0; measured in
# research/docs/numerics.md, reflect+black+16): at the bottom, the fewest rows reflected from the
# picture, at least REFLECTED, that bring it to a multiple of MULTIPLE (8 to 23), then BLACK rows
# of zeros; at the right, columns the same way, only when the width isn't a multiple of MULTIPLE.
# All are trimmed after the decode (shot._decoded). numz pads zeros alone, up to the multiple (8
# rows at 1080p, none at 720p), which costs the bottom 16 rows 3-10 dB, while the black rows
# anchor the model's tone: without them the whole frame is worse.
MULTIPLE = 16
REFLECTED = 8
BLACK = 16


def padding(size: tuple[int, int]) -> tuple[int, int]:
    """(rows, columns) reflected from a picture of size (height, width), before the black ones:
    the fewest rows, at least REFLECTED, that bring its height to a multiple of MULTIPLE, 8 to
    23; columns the same way, none when its width is a multiple of MULTIPLE already. 1080 rows
    get 8, 720 and 2160 get 16; 1920 columns none, 1912 get 8."""
    height, width = size
    return _reflected(height), _reflected(width) if width % MULTIPLE else 0


def _reflected(side: int) -> int:
    return (-side - REFLECTED) % MULTIPLE + REFLECTED


def padded_size(size: tuple[int, int]) -> tuple[int, int]:
    """(height, width) of a picture of size (height, width) once padded, as the model takes it:
    its rows reflected (padding) and BLACK more, its columns alike when any are reflected; each a
    multiple of MULTIPLE. 1080x1920 is padded to 1104x1920, 720x1280 to 752x1280, 1060x1912 to
    1088x1936."""
    height, width = size
    rows, columns = padding(size)
    return height + rows + BLACK, width + columns + (BLACK if columns else 0)


def check_target(target: tuple[int, int], resolution: int) -> None:
    """Refuse a target, (height, width), the padding can't be made on: torch reflects fewer rows
    than the picture has (torch.nn.functional.pad, mode reflect), and columns alike, so it takes
    17 rows at least (16 would take 16 reflected) and 16 columns (16 take none; fewer would take
    at least as many as they are). --resolution, the target's short side, takes any whole number
    above 0 (main.py)."""
    height, width = target
    rows, columns = padding(target)
    if rows >= height or columns >= width:
        raise Refusal(
            f"--resolution {resolution}: the frames resized to {width}x{height} are too small"
            f" for the padding, at least {REFLECTED} rows reflected from the picture under it,"
            f" up to a multiple of {MULTIPLE}, and columns alike: it takes 17 rows and 16"
            " columns at least"
        )


def padded_length(count: int) -> int:
    """The VAE's input length, 4n + 1: the frames of `count` frames once padded to it, the VAE
    packing 4 frames per latent after the first."""
    return count + (-count + 1) % 4

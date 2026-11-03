"""One picture's figures against what was measured on the RTX 5090 (seedvr2x/peaks.py's
picture_encode_peak, picture_decode_peak and colour_one_frame, seedvr2x/still_plan.py's
copies_peak and Peak; DESIGN.md, Still images, "The plan for one picture"), on the CPU, no model:
every one-frame measure is under the prediction the plan counts with, and the prediction stays
near it, so that a change of the constants that opens a gap, or closes one, fails here.

The figures are copied as literals from the card's batch: measured on the RTX 5090 on 2026-11-03
by design's agent (SDPA, torch 2.14.1, backend:native,expandable_segments:True, a fresh process a
case, the desktop live); the raw lines are kept outside the repository. Each is the peak of
torch's allocated memory over what was allocated before the pass: the weights, and for the
encode its input frame. The padded frames are seedvr2x's own (frames.padded_size): 1080p
1104x1920, 4K 2192x3840, 6000x4000 4032x6000, 8K 4352x7680. A tile of 0 is the untiled pass."""

import math

import pytest

from seedvr2x import peaks, still_plan
from seedvr2x.frames import padded_size
from seedvr2x.passes import Tiles
from seedvr2x.peaks import GIB

PADDED = {
    "1080p": (1104, 1920),
    "4K": (2192, 3840),
    "6000x4000": (4032, 6000),
    "8K": (4352, 7680),
}

# GiB above what was allocated before the pass, by padded frame and tile side (0: untiled).
ENCODE: dict[str, dict[int, float]] = {
    "1080p": {0: 5.0668, 1536: 4.6025, 1024: 3.5143, 768: 2.2903, 512: 1.0367},
    "4K": {0: 14.6214, 2048: 8.344, 1536: 5.6827, 1024: 3.5465, 768: 2.2903, 512: 1.0367},
    "6000x4000": {2048: 8.4987, 1536: 5.6827, 1024: 3.5465, 768: 2.2903, 512: 1.0367},
    "8K": {2048: 8.3791, 1536: 5.6827, 1024: 3.5465, 768: 2.2903, 512: 1.0367},
}
DECODE: dict[str, dict[int, float]] = {
    "1080p": {0: 7.2197, 1536: 6.512, 1024: 4.7844, 768: 3.6354, 512: 1.8858},
    "4K": {2048: 12.1039, 1536: 8.0072, 1024: 4.7844, 768: 3.6354, 512: 1.8858},
    "6000x4000": {2048: 12.1273, 1536: 8.0072, 1024: 4.4296, 768: 3.6354, 512: 1.8858},
    "8K": {2048: 12.1273, 1536: 8.0072, 1024: 4.7844, 768: 3.6354, 512: 1.8858},
}

# The colour correction of one frame, above the weights: the decoded frame, the reference, split
# and the workspace. (Output size, padded frame, GiB).
COLOUR = [
    ((1080, 1920), (1104, 1920), 0.1832),
    ((2160, 3840), (2192, 3840), 0.6369),
    ((4000, 6000), (4032, 6000), 1.7796),
    ((4320, 7680), (4352, 7680), 2.4476),
]

# The picture's copies on the GPU, above the models: (output target, input as stored, the
# encode's GiB, the decode's GiB or None where it was not measured).
COPIES = [
    ((1080, 1920), (540, 960), 0.0608, 0.119),
    ((2160, 3840), (1080, 1920), 0.2433, 0.382),
    ((4000, 6000), (2000, 3000), 0.7041, 1.041),
    ((4320, 7680), (2160, 3840), 0.9733, 1.427),
    ((2160, 3840), (540, 960), 0.1999, None),
    ((2160, 3840), (4320, 7680), 1.1124, 0.974),
    ((2160, 3840), (6750, 12000), 2.4485, 2.043),
    ((1080, 1920), (4320, 7680), 0.9733, 0.800),
]

# What is on the GPU before a pass besides the picture's frame: the VAE's weights and the DiT's
# two slots (no block resident, none swapped: peaks.dit_on_gpu(0, 0)), as the cases ran.
MODELS = peaks.VAE_WEIGHTS + peaks.dit_on_gpu(0, 0)

# How far over a measure the prediction may be, tiled or not: the most the table below has is
# 1.34 (the 4K untiled encode, the slice's fit at 14.6 GiB measured at 1080p's 5.07), and 1.14 for
# a tile (the 6000x4000 encode at 512 px, where 1080p's cut-off tiles took 0.283 over the rest).
SLACK = 1.45

ENCODES = [
    pytest.param(padded, side, gib, id=f"encode-{name}-{side}")
    for name, padded in PADDED.items()
    for side, gib in ENCODE[name].items()
]
DECODES = [
    pytest.param(padded, side, gib, id=f"decode-{name}-{side}")
    for name, padded in PADDED.items()
    for side, gib in DECODE[name].items()
]


def tiles_of(side: int) -> Tiles | None:
    return Tiles(side) if side else None


def encoded(padded: tuple[int, int], side: int) -> float:
    """The prediction of a picture's encode above the models and the picture's frame, in GiB."""
    peak = peaks.picture_encode_peak(padded, tiles_of(side), resident=0, swapped=0)
    return (peak - MODELS - peaks.ONE_FRAME_INPUT * padded[0] * padded[1]) / GIB


def decoded(padded: tuple[int, int], side: int) -> float:
    """The prediction of a picture's decode (no colour correction) above the models, in GiB."""
    output = (padded[0], padded[1])
    peak = peaks.picture_decode_peak(padded, output, tiles_of(side), 0, 0, split=False)
    return (peak - MODELS) / GIB


@pytest.mark.parametrize(("padded", "side", "measured"), ENCODES)
def test_the_encode_prediction_covers_the_measure_and_stays_near_it(
    padded: tuple[int, int], side: int, measured: float
) -> None:
    predicted = encoded(padded, side)
    assert predicted >= measured
    assert predicted <= measured * SLACK


@pytest.mark.parametrize(("padded", "side", "measured"), DECODES)
def test_the_decode_prediction_covers_the_measure_and_stays_near_it(
    padded: tuple[int, int], side: int, measured: float
) -> None:
    predicted = decoded(padded, side)
    assert predicted >= measured
    assert predicted <= measured * SLACK


def test_the_most_a_vae_pass_is_over_counted_is_computed_here_and_under_the_slack() -> None:
    # The comment on SLACK says where the most is; this is the figure it is read off.
    over = {
        ("encode", name, side): encoded(PADDED[name], side) / gib
        for name, row in ENCODE.items()
        for side, gib in row.items()
    } | {
        ("decode", name, side): decoded(PADDED[name], side) / gib
        for name, row in DECODE.items()
        for side, gib in row.items()
    }
    most = max(over, key=lambda each: over[each])
    assert over[most] < SLACK
    assert most == ("encode", "4K", 0)
    tiled = max((each for each in over if each[2]), key=lambda each: over[each])
    assert over[tiled] < 1.2


def test_a_tile_of_the_grids_shape_takes_the_same_at_every_picture_size() -> None:
    # A tile's peak is the same at every picture size to the MiB where the tiles' shapes are the
    # grid's (the 768-px encode and decode at the four sizes, the 1024-px encode at the three
    # larger), and the prediction too: it depends on the tile, not on the frame.
    assert len({encoded(PADDED[name], 768) for name in PADDED}) == 1
    assert len({decoded(PADDED[name], 768) for name in PADDED}) == 1
    assert len({encoded(PADDED[name], 1024) for name in ("4K", "6000x4000", "8K")}) == 1


@pytest.mark.parametrize(("output", "padded", "measured"), COLOUR)
def test_the_colour_correction_prediction_covers_the_measure(
    output: tuple[int, int], padded: tuple[int, int], measured: float
) -> None:
    assert padded_size(output) == padded
    predicted = peaks.colour_one_frame(padded, output) / GIB
    assert predicted >= measured
    # Split's own term is the slice's, over by a fifth at the most (2.83 against 2.45 GiB at 8K:
    # split 2.24 counted against 1.86 measured, and the padded frame's 4352 rows).
    assert predicted <= measured * 1.2


COPY_CASES = [
    pytest.param(target, stored, encode, decode, id=f"{target[1]}x{target[0]}-from-{stored[1]}")
    for target, stored, encode, decode in COPIES
]


@pytest.mark.parametrize(("target", "stored", "encode", "decode"), COPY_CASES)
def test_the_copies_prediction_covers_the_measure_and_is_within_a_quarter(
    target: tuple[int, int], stored: tuple[int, int], encode: float, decode: float | None
) -> None:
    padded = padded_size(target)
    pixels, input_pixels = padded[0] * padded[1], stored[0] * stored[1]
    # The workspace of cuBLAS (peaks.CUBLAS_WORKSPACE, 32 MiB) is counted in every moment: a
    # first picture's encode runs before the first matrix product and doesn't hold it, which the
    # smallest case (0.0608 GiB measured) shows as 1.53 times; the quarter is over the rest.
    workspace = peaks.CUBLAS_WORKSPACE / GIB
    predicted = still_plan.copies_peak("encode", input_pixels, pixels) / GIB
    assert predicted >= encode
    assert predicted <= encode * 1.25 + workspace
    if decode is not None:
        predicted = still_plan.copies_peak("decode", input_pixels, pixels) / GIB
        assert predicted >= decode
        assert predicted <= decode * 1.25


# ---- What a phase takes for a reservation --------------------------------------------------------


def test_the_reserved_figure_is_the_peak_its_room_and_a_fifth_of_the_part_above_the_models() -> (
    None
):
    assert peaks.RESERVED_OVER == 0.2
    peak = still_plan.Peak(allocated=10 * GIB, models=3 * GIB, room=peaks.VAE_ROOM)
    assert peak.above == 7 * GIB
    assert peak.reserved == 10 * GIB + peaks.VAE_ROOM + math.ceil(0.2 * 7 * GIB)
    # The share rounds up to the byte: 1,000,001 bytes above the models, 200,000.2 over them.
    odd = still_plan.Peak(allocated=1_000_004, models=3, room=5)
    assert odd.above == 1_000_001
    assert odd.reserved == 1_000_004 + 5 + 200_001
    # The DiT's phase has no room, and a phase with nothing above its models adds nothing.
    assert still_plan.Peak(allocated=GIB, models=GIB, room=0).reserved == GIB

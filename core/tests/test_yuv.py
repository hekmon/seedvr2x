"""seedvr2x/yuv.py, the library's alone, from closed forms written here and no ffmpeg: the table of
pixel formats (depth, subsampling, the packed formats' planar twins), `reading` (tags kept, the
guesses of an untagged stream, the refusals by kind and in their words), `to_rgb` (limited and full
range, RGB, the chroma sitings worked out by hand, dtypes, frames, clamping, the sizes refused).
Its comparison with zscale is the CLI's (tests/test_yuv_cli.py)."""

import subprocess
import sys

import numpy as np
import pytest
import torch
from library_fixtures import cli_words

from seedvr2x import yuv
from seedvr2x.errors import SourceRefused
from seedvr2x.yuv import PIXEL_FORMATS, Reading, reading, to_rgb

WEIGHTS = {"bt709": (0.2126, 0.0722), "bt470bg": (0.299, 0.114), "smpte170m": (0.299, 0.114)}
MATRIX_NAMES = {"bt709": "709", "bt470bg": "470bg", "smpte170m": "170m", "bt2020nc": "2020_ncl"}


def codes(out: torch.Tensor) -> torch.Tensor:
    """The 16-bit codes of an output, which must be whole numbers."""
    scaled = out.double() * 65535
    assert torch.allclose(scaled, scaled.round(), atol=1e-3)
    return scaled.round().to(torch.int64)


# The table

DEPTH_AND_SUBSAMPLING = {
    "yuv420p": (8, (2, 2)),
    "yuv422p10le": (10, (1, 2)),
    "yuv444p16le": (16, (1, 1)),
    "yuv420p9le": (9, (2, 2)),
    "yuv444p14le": (14, (1, 1)),
    "yuv411p": (8, (1, 4)),
    "yuv410p": (8, (4, 4)),
    "yuv440p": (8, (2, 1)),
    "yuvj420p": (8, (2, 2)),
    "yuvj422p": (8, (1, 2)),
    "yuvj444p": (8, (1, 1)),
    "gbrp": (8, (1, 1)),
    "gbrp12le": (12, (1, 1)),
}


@pytest.mark.parametrize(("name", "expected"), DEPTH_AND_SUBSAMPLING.items())
def test_depth_and_subsampling_of_each_planar_format(
    name: str, expected: tuple[int, tuple[int, int]]
) -> None:
    assert (PIXEL_FORMATS[name].depth, PIXEL_FORMATS[name].subsampling) == expected


def test_every_format_maps_to_a_planar_format_that_maps_to_itself() -> None:
    for name, pixel_format in PIXEL_FORMATS.items():
        assert PIXEL_FORMATS[pixel_format.planar] is PIXEL_FORMATS[pixel_format.planar]
        assert PIXEL_FORMATS[pixel_format.planar].planar == pixel_format.planar, name


PACKED = {
    "nv12": "yuv420p",
    "nv21": "yuv420p",
    "nv16": "yuv422p",
    "nv24": "yuv444p",
    "p010le": "yuv420p10le",
    "p016le": "yuv420p16le",
    "p210le": "yuv422p10le",
    "p412le": "yuv444p12le",
    "yuyv422": "yuv422p",
    "uyvy422": "yuv422p",
    "rgb24": "gbrp",
    "bgr0": "gbrp",
    "rgb48be": "gbrp16le",
    "bgr48le": "gbrp16le",
}


@pytest.mark.parametrize(("packed", "planar"), PACKED.items())
def test_a_packed_format_is_its_planar_twin_of_the_same_depth_and_subsampling(
    packed: str, planar: str
) -> None:
    twin = PIXEL_FORMATS[planar]
    assert PIXEL_FORMATS[packed].planar == planar
    assert (PIXEL_FORMATS[packed].depth, PIXEL_FORMATS[packed].subsampling) == (
        twin.depth,
        twin.subsampling,
    )


# reading()


def test_tagged_values_are_kept_and_nothing_is_guessed() -> None:
    read = reading("yuv420p10le", 1920, 1080, "smpte170m", "pc", "center")
    assert read == Reading("yuv420p10le", "170m", "full", "center", (), False)
    read = reading("yuv422p", 64, 64, "bt2020nc", "tv", "topleft")
    assert (read.matrix, read.color_range, read.chroma_location) == (
        "2020_ncl",
        "limited",
        "topleft",
    )
    assert read.guessed == () and not read.matrix_guessed


def test_untagged_values_are_guessed_and_listed_in_order() -> None:
    read = reading("yuv420p", 1920, 1080)
    assert read.guessed == ("matrix bt709 (from 1920x1080)", "limited range", "chroma sited left")
    assert (read.matrix, read.color_range, read.chroma_location) == ("709", "limited", "left")
    assert read.matrix_guessed
    partial = reading("yuv420p", 720, 480, color_range="pc")
    assert partial.guessed == ("matrix smpte170m (from 720x480)", "chroma sited left")
    assert partial.color_range == "full" and partial.matrix == "170m"
    assert reading("yuv420p", 720, 480, "bt709", "tv", "left").guessed == ()


@pytest.mark.parametrize(
    ("width", "height", "name"),
    [(1280, 720, "bt709"), (1279, 576, "smpte170m"), (1279, 577, "bt709"), (720, 480, "smpte170m")],
)
def test_the_matrix_guess_at_its_boundaries(width: int, height: int, name: str) -> None:
    assert yuv.guess_matrix(width, height) == name
    assert reading("yuv420p", width, height).matrix == MATRIX_NAMES[name]


@pytest.mark.parametrize("word", sorted(yuv.UNTAGGED))
def test_each_word_for_an_absent_tag_is_read_as_absent(word: str) -> None:
    read = reading("yuv420p", 1920, 1080, word, word, word)
    assert read == reading("yuv420p", 1920, 1080)
    assert len(read.guessed) == 3


def test_a_siting_is_guessed_only_where_the_format_is_subsampled() -> None:
    assert reading("yuv444p", 1920, 1080).guessed == (
        "matrix bt709 (from 1920x1080)",
        "limited range",
    )
    assert reading("yuv444p", 1920, 1080).chroma_location == "left"
    assert "chroma sited left" in reading("yuv411p", 64, 64).guessed


def test_the_jpeg_family_is_always_full_range_and_its_range_not_guessed() -> None:
    for tag in ("", "pc", "unknown"):
        read = reading("yuvj420p", 1920, 1080, "bt709", tag, "left")
        assert read.color_range == "full" and read.guessed == ()
    assert reading("yuvj444p", 1920, 1080).guessed == ("matrix bt709 (from 1920x1080)",)


def test_rgb_is_always_gbr_full_and_left_with_nothing_guessed() -> None:
    for name in ("gbrp", "gbrp10le", "rgb24", "bgr0", "rgb48be"):
        read = reading(name, 720, 480, "bt709", "pc", "center")
        assert (read.matrix, read.color_range, read.chroma_location) == ("gbr", "full", "left")
        assert read.guessed == () and not read.matrix_guessed
        assert read == reading(name, 720, 480)
    assert reading("rgb24", 8, 8).planar == "gbrp"


def test_a_matrix_given_replaces_the_tagged_and_the_untagged_one() -> None:
    assert reading("yuv420p", 720, 480, "smpte170m", "tv", "left", matrix="bt709").matrix == "709"
    given = reading("yuv420p", 720, 480, matrix="bt2020nc")
    assert given.matrix == "2020_ncl" and not given.matrix_guessed
    assert given.guessed == ("limited range", "chroma sited left")


REFUSALS = [
    (
        ("foo", 64, 64),
        {},
        "pixel-format",
        "pixel format foo is not supported",
    ),
    (
        ("pal8", 64, 64),
        {},
        "pixel-format",
        "palette pixel format pal8: not the YUV or RGB the conversions are tested for",
    ),
    (
        ("gray", 64, 64),
        {},
        "pixel-format",
        "grey pixel format gray: not the YUV or RGB the conversions are tested for",
    ),
    (
        ("yuva420p", 64, 64),
        {},
        "pixel-format",
        "pixel format yuva420p has an alpha channel: dropping it would change the picture"
        " wherever it isn't opaque",
    ),
    (
        ("yuv420p", 64, 64, "bt2020c"),
        {},
        "matrix",
        "matrix bt2020c is not supported (only bt709, bt470bg, smpte170m, bt2020nc)",
    ),
    (
        ("yuv420p", 64, 64, "smpte240m"),
        {},
        "matrix",
        "matrix smpte240m is not supported (only bt709, bt470bg, smpte170m, bt2020nc)",
    ),
    (
        ("yuv420p", 64, 64),
        {"matrix": "fcc"},
        "matrix",
        "matrix fcc is not supported (only bt709, bt470bg, smpte170m, bt2020nc)",
    ),
    (
        ("gbrp", 64, 64, "", "tv"),
        {},
        "rgb-limited",
        "RGB tagged limited range: rare, most likely a mis-tag, and either reading could be wrong",
    ),
    (
        ("yuvj420p", 64, 64, "", "tv"),
        {},
        "jpeg-limited",
        "yuvj420p, full range by definition, tagged limited range",
    ),
    (
        ("gbrp", 64, 64),
        {"matrix": "bt709"},
        "rgb-matrix",
        "matrix bt709 given: the source is RGB, it has no matrix",
    ),
]


@pytest.mark.parametrize(("args", "keywords", "kind", "message"), REFUSALS)
def test_a_refusal_is_a_source_refused_of_its_kind_in_its_words(
    args: tuple[str | int, ...], keywords: dict[str, str], kind: str, message: str
) -> None:
    with pytest.raises(SourceRefused) as caught:
        reading(*args, **keywords)  # type: ignore[arg-type]
    error = caught.value
    assert (error.kind, str(error), error.reason) == (kind, message, message)
    assert "--" not in str(error) and cli_words(error) == message


def test_every_kind_is_raised_by_some_case() -> None:
    raised = set()
    for args, keywords, _, _ in REFUSALS:
        with pytest.raises(SourceRefused) as caught:
            reading(*args, **keywords)  # type: ignore[arg-type]
        raised.add(caught.value.kind)
    raised.add(_size_refusal().kind)
    assert raised == set(yuv.KINDS)


def _size_refusal() -> SourceRefused:
    with pytest.raises(SourceRefused) as caught:
        to_rgb(
            [torch.zeros((1, 3, 4)), torch.zeros((1, 1, 2)), torch.zeros((1, 1, 2))],
            reading("yuv420p", 4, 3),
        )
    return caught.value


@pytest.mark.parametrize(
    ("planar", "size"),
    [
        ("yuv420p", (65, 16)),
        ("yuv420p", (64, 17)),
        ("yuv422p10le", (63, 16)),
        ("yuv411p", (66, 16)),
        ("yuv410p", (64, 18)),
        ("yuv440p", (64, 15)),
    ],
)
def test_frames_whose_sides_the_subsampling_doesnt_divide_are_refused(
    planar: str, size: tuple[int, int]
) -> None:
    # zscale refuses them ("image dimensions must be divisible by subsampling factor"), so the
    # command line reads no such source; the shapes given are those of the planes cut down.
    width, height = size
    rows, columns = PIXEL_FORMATS[planar].subsampling
    small = torch.zeros((1, height // rows, width // columns), dtype=torch.int32)
    with pytest.raises(SourceRefused) as caught:
        to_rgb(
            [torch.zeros((1, height, width), dtype=torch.int32), small, small],
            reading(planar, width, height),
        )
    assert caught.value.kind == "size"
    assert str(caught.value) == (
        f"frames of {width}x{height} in {planar}: their sides are not multiples of its chroma"
        f" subsampling, {columns} columns by {rows} rows, which the conversion refuses"
    )


def test_chroma_planes_of_another_shape_are_a_caller_s_error() -> None:
    # A decoder's padded rows uncut, say: refused, never stretched.
    luma = torch.zeros((1, 16, 64), dtype=torch.int32)
    padded = torch.zeros((1, 8, 40), dtype=torch.int32)
    with pytest.raises(ValueError, match="chroma planes are"):
        to_rgb([luma, padded, padded], reading("yuv420p", 64, 16))


# to_rgb()


def frame(read: Reading, width: int, height: int, dtype: torch.dtype, *values: int) -> list:
    """One frame of constant planes at the format's subsampling, one value a plane."""
    rows, columns = read.format.subsampling
    shapes = [(height, width)] + [(height // rows, width // columns)] * 2
    return [
        torch.full((1, *shape), value, dtype=dtype)
        for shape, value in zip(shapes, values, strict=True)
    ]


@pytest.mark.parametrize("matrix", MATRIX_NAMES)
@pytest.mark.parametrize(
    ("name", "depth"),
    [("yuv420p", 8), ("yuv420p10le", 10), ("yuv420p12le", 12), ("yuv420p16le", 16)],
)
def test_limited_white_and_black_with_neutral_chroma_are_exact(
    name: str, depth: int, matrix: str
) -> None:
    read = reading(name, 64, 64, matrix, "tv", "left")
    shift = depth - 8
    dtype = torch.uint8 if depth == 8 else torch.int32
    for luma, expected in ((235 << shift, 65535), (16 << shift, 0)):
        out = to_rgb(frame(read, 4, 4, dtype, luma, 128 << shift, 128 << shift), read)
        assert out.dtype == torch.float32 and out.shape == (1, 4, 4, 3)
        assert (codes(out) == expected).all()


@pytest.mark.parametrize(
    ("name", "tag"), [("yuvj420p", ""), ("yuvj420p", "pc"), ("yuv444p", "pc"), ("yuv420p", "pc")]
)
def test_full_range_white_and_black_with_neutral_chroma_are_exact(name: str, tag: str) -> None:
    read = reading(name, 64, 64, "bt709", tag, "left")
    for luma, expected in ((255, 65535), (0, 0)):
        out = to_rgb(frame(read, 4, 4, torch.uint8, luma, 128, 128), read)
        assert (codes(out) == expected).all()


def test_8_bit_rgb_gives_every_code_times_257_in_gbrps_order() -> None:
    read = reading("gbrp", 16, 16)
    v = torch.arange(256, dtype=torch.int32).view(1, 16, 16)
    g, b, r = v, (v + 85) % 256, (v + 170) % 256
    out = codes(to_rgb([g.to(torch.uint8), b.to(torch.uint8), r.to(torch.uint8)], read))
    for channel, plane in enumerate((r, g, b)):
        assert (out[..., channel] == plane.long() * 257).all()


def test_16_bit_rgb_gives_its_codes_back_and_10_bit_rounds_to_nearest() -> None:
    generator = torch.Generator().manual_seed(1)
    planes = [torch.randint(0, 65536, (2, 5, 7), generator=generator) for _ in range(3)]
    planes[0][0, 0, :3] = torch.tensor([0, 65535, 32768])
    out = codes(to_rgb([p.to(torch.int32) for p in planes], reading("gbrp16le", 7, 5)))
    assert (out[..., 0] == planes[2]).all() and (out[..., 1] == planes[0]).all()
    assert (out[..., 2] == planes[1]).all()
    v = torch.arange(1024, dtype=torch.int64).view(1, 32, 32)
    out = codes(to_rgb([v.to(torch.int32)] * 3, reading("gbrp10le", 32, 32)))
    assert (out[..., 1] == (2 * v * 65535 + 1023) // 2046).all()


def random_planes(name: str, frames: int, seed: int, dtype: torch.dtype) -> list[torch.Tensor]:
    read = reading(name, 64, 64, "bt709", "tv", "left")
    rows, columns = read.format.subsampling
    top = (1 << read.format.depth) - 1
    generator = torch.Generator().manual_seed(seed)
    shapes = [(8, 12)] + [(8 // rows, 12 // columns)] * 2
    return [torch.randint(0, top + 1, (frames, *s), generator=generator).to(dtype) for s in shapes]


@pytest.mark.parametrize("name", ["yuv420p", "yuv422p10le", "yuv444p16le", "yuv420p16le"])
def test_the_output_is_whole_codes_over_65535_bit_for_bit(name: str) -> None:
    read = reading(name, 64, 64, "bt709", "tv", "left")
    out = to_rgb(random_planes(name, 2, 3, torch.int32), read)
    assert (codes(out).float() / 65535).view(torch.int32).equal(out.view(torch.int32))
    assert out.min() >= 0 and out.max() <= 1


@pytest.mark.parametrize("matrix", MATRIX_NAMES)
@pytest.mark.parametrize(
    ("name", "exact"), [("yuv444p", True), ("yuv444p16le", False)], ids=["8-bit", "16-bit"]
)
def test_a_grey_gives_equal_channels(name: str, exact: bool, matrix: str) -> None:
    read = reading(name, 64, 64, matrix, "tv", "left")
    depth = read.format.depth
    luma = torch.arange(1 << depth, dtype=torch.int32).view(1, 1, -1)
    neutral = torch.full_like(luma, 128 << (depth - 8))
    out = codes(to_rgb([luma, neutral, neutral], read))
    spread = (out.max(dim=-1).values - out.min(dim=-1).values).max()
    assert spread <= 1
    if not exact:
        # Measured 2026-11-13: 52 to 112 of the 65,536 codes (0.1 to 0.2%) differ by one code.
        print(f"{matrix} 16-bit grey: {int((out[..., 0] != out[..., 1]).sum())} of 65536 differ")
        return
    assert spread == 0


def red_of(read: Reading, cr: np.ndarray, width: int, height: int) -> np.ndarray:
    """The red codes the chroma siting gives, in float64 from the definition: output sample i at
    luma position i + 0.5, chroma sample j at factor * j + (0.5 cosited, factor / 2 centred, the
    last sample's centre factor - 0.5 for the bottom), the two chroma samples around the position
    weighted by distance, an edge's outside neighbour the edge itself. Luma 0.5 of a 16-bit
    limited range, Cb neutral, BT.709."""
    rows, columns = read.format.subsampling
    where = read.chroma_location
    across = 0.5 if where in ("left", "topleft", "bottomleft") else columns / 2
    down = {"top": 0.5, "topleft": 0.5, "bottom": rows - 0.5, "bottomleft": rows - 0.5}.get(
        where, rows / 2
    )

    def spread(values: np.ndarray, factor: int, centre: float, size: int, axis: int) -> np.ndarray:
        out = []
        for i in range(size):
            t = (i + 0.5 - centre) / factor
            low = int(np.floor(t))
            lo, hi = (
                min(max(low, 0), values.shape[axis] - 1),
                min(max(low + 1, 0), values.shape[axis] - 1),
            )
            w = t - low
            out.append((1 - w) * np.take(values, lo, axis) + w * np.take(values, hi, axis))
        return np.stack(out, axis)

    red = (cr.astype(np.float64) - 32768) / 57344
    red = spread(spread(red, columns, across, width, 1), rows, down, height, 0)
    return np.round(np.clip(0.5 + 2 * (1 - 0.2126) * red, 0, 1) * 65535)


def sited(name: str, location: str, cr: np.ndarray, width: int, height: int) -> tuple:
    read = reading(name, 64, 64, "bt709", "tv", location)
    luma = torch.full((1, height, width), 4096 + 28032, dtype=torch.int32)
    chroma = torch.from_numpy(cr).to(torch.int32)[None]
    neutral = torch.full_like(chroma, 32768)
    out = codes(to_rgb([luma, neutral, chroma], read))[0, ..., 0]
    return out.numpy(), red_of(read, cr, width, height)


def test_sited_left_an_even_column_takes_its_sample_and_an_odd_one_the_mean() -> None:
    cr = np.array([[30000, 36000]])
    got, _ = sited("yuv420p16le", "left", cr, 4, 2)
    red = lambda c: round((0.5 + 2 * (1 - 0.2126) * (c - 32768) / 57344) * 65535)  # noqa: E731
    mean = (30000 + 36000) / 2
    # The 4:2:0 frame is 4 x 2: one chroma row, vertically a step of nothing.
    assert abs(got[0] - [red(30000), red(mean), red(36000), red(36000)]).max() <= 1


def test_centred_the_weights_are_a_quarter_and_three_quarters() -> None:
    cr = np.array([[30000, 36000]])
    got, _ = sited("yuv420p16le", "center", cr, 4, 2)
    red = lambda c: round((0.5 + 2 * (1 - 0.2126) * (c - 32768) / 57344) * 65535)  # noqa: E731
    expected = [
        red(30000),
        red(0.75 * 30000 + 0.25 * 36000),
        red(0.25 * 30000 + 0.75 * 36000),
        red(36000),
    ]
    assert abs(got[0] - expected).max() <= 1


@pytest.mark.parametrize("location", sorted(yuv.CHROMA_LOCATIONS))
@pytest.mark.parametrize("name", ["yuv420p16le", "yuv422p16le"])
def test_a_ramp_of_chroma_follows_the_siting_worked_out_by_hand(name: str, location: str) -> None:
    read = reading(name, 64, 64, "bt709", "tv", location)
    rows, columns = read.format.subsampling
    cr = (np.arange(3 * 4).reshape(3, 4) * 1500 + 24000).astype(np.int64)
    got, want = sited(name, location, cr, 4 * columns, 3 * rows)
    assert abs(got - want).max() <= 1


@pytest.mark.parametrize(
    ("name", "location"),
    [
        ("yuv411p", "left"),
        ("yuv411p", "center"),
        ("yuv410p", "topleft"),
        ("yuv410p", "center"),
        ("yuv440p", "top"),
        ("yuv440p", "bottom"),
    ],
)
def test_the_8_bit_formats_with_four_columns_or_rows_follow_their_siting(
    name: str, location: str
) -> None:
    read = reading(name, 64, 64, "bt709", "tv", location)
    rows, columns = read.format.subsampling
    cr = np.array([[100, 140, 90], [150, 120, 130]])[:2, :3]
    luma = torch.full((1, 2 * rows, 3 * columns), 126, dtype=torch.uint8)  # 16 + 219 / 2
    chroma = torch.from_numpy(cr).to(torch.uint8)[None]
    neutral = torch.full_like(chroma, 128)
    got = codes(to_rgb([luma, neutral, chroma], read))[0, ..., 0].numpy()
    red = (cr - 128) / 224
    expected = red_of(read, (red * 57344 + 32768), 3 * columns, 2 * rows)
    # red_of takes a luma of 0.5 and 16-bit limited codes; 126 is (126 - 16) / 219 = 0.5023.
    shift = (126 - 16) / 219 - 0.5
    assert abs(got - (expected + shift * 65535)).max() <= 2


def test_the_dtype_of_the_planes_changes_no_bit() -> None:
    for name, narrow in (("yuv420p", torch.uint8), ("yuv422p16le", torch.uint16)):
        read = reading(name, 64, 64, "bt709", "tv", "left")
        base = random_planes(name, 2, 5, torch.int32)
        a = to_rgb(base, read)
        assert a.view(torch.int32).equal(
            to_rgb([p.to(narrow) for p in base], read).view(torch.int32)
        )


def test_frames_converted_together_equal_each_converted_alone() -> None:
    read = reading("yuv420p10le", 64, 64, "smpte170m", "tv", "topleft")
    planes = random_planes("yuv420p10le", 3, 7, torch.int32)
    together = to_rgb(planes, read)
    for index in range(3):
        alone = to_rgb([p[index : index + 1] for p in planes], read)
        assert alone[0].view(torch.int32).equal(together[index].view(torch.int32))


def test_values_out_of_gamut_clamp_to_the_range() -> None:
    read = reading("yuv444p", 64, 64, "bt709", "pc", "left")
    luma = torch.tensor([[[255, 255, 0, 0]]], dtype=torch.uint8)
    cb = torch.tensor([[[0, 255, 0, 255]]], dtype=torch.uint8)
    cr = torch.tensor([[[255, 0, 255, 0]]], dtype=torch.uint8)
    out = codes(to_rgb([luma, cb, cr], read))
    assert out.min() == 0 and out.max() == 65535
    assert out[0, 0, 0, 0] == 65535 and out[0, 0, 3, 0] == 0


# The import

IMPORTED = """
import sys
from seedvr2x import yuv
yuv.reading("yuv420p", 1920, 1080)
print("torch" in sys.modules)
"""


def test_importing_yuv_imports_no_torch() -> None:
    result = subprocess.run(
        [sys.executable, "-c", IMPORTED], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"

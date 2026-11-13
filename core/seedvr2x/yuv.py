"""A video's frames between the YCbCr or RGB planes a file stores and the pipeline's 16-bit RGB
(DESIGN.md, Input; Repository and apps, the nodes' VIDEO path): the pixel formats, matrices and
chroma sitings the command line reads, its guesses for an untagged stream and its refusals; a
source's planes to RGB in torch by those rules, where zscale is not to be had; and the RGB of an
output to the planes of the nodes' master, 16-bit 4:4:4 YCbCr, BT.709, limited range.

The command line converts with zscale, every parameter explicit (its media/conversion.py, which
takes the table and the rules from here). `to_rgb` makes the same conversion in float32 and was
measured against it, file by file for every format, depth, range, matrix and siting the command
line reads (the CLI's tests/test_yuv_cli.py): within one 16-bit code on every sample. zimg's
order, which the measures agree with: the samples to float by the range, exactly (a limited
range's black at 0 and white at 1, its chroma's neutral at 0 and its extremes at +-0.5 by
224 / 255 of the depth's span; a full range's span the depth's whole, its chroma's neutral at
half), the chroma upsampled in float with the bilinear kernel (conversion.py's KERNEL) at the
positions its siting gives, the matrix, then each sample rounded to 16 bits, clamped to [0, 1].

Plain Python at import, no torch: the command line reads the table before torch is imported.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from seedvr2x.errors import SourceRefused

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class PixelFormat:
    """A decoded pixel format seedvr2x reads."""

    planar: str  # the planar format its planes are: the source's own, or its planar twin
    family: str  # "yuv"; "jpeg", YUV with full range by definition; "rgb"
    subsampled: bool  # chroma subsampled, so its location matters

    @property
    def depth(self) -> int:
        """The bits of each sample: 8 to 16."""
        match = re.search(r"p(\d+)le$", self.planar)
        return int(match[1]) if match else 8

    @property
    def subsampling(self) -> tuple[int, int]:
        """Luma rows and columns per chroma sample: (2, 2) for 4:2:0, (1, 1) for 4:4:4 and RGB."""
        for chroma, factors in _SUBSAMPLING.items():
            if self.planar.startswith(("yuv" + chroma, "yuvj" + chroma)):
                return factors
        return 1, 1


_SUBSAMPLING = {
    "420": (2, 2),
    "422": (1, 2),
    "444": (1, 1),
    "411": (1, 4),
    "410": (4, 4),
    "440": (2, 1),
}


def _pixel_formats() -> dict[str, PixelFormat]:
    formats: dict[str, PixelFormat] = {}
    # Planar formats zscale takes as they are: ffmpeg inserts no conversion before it (checked
    # for each, the CLI's tests/test_decode.py).
    for depth in ("", "9le", "10le", "12le", "14le", "16le"):
        for chroma in ("420", "422", "444"):
            formats[f"yuv{chroma}p{depth}"] = PixelFormat(
                f"yuv{chroma}p{depth}", "yuv", chroma != "444"
            )
        formats[f"gbrp{depth}"] = PixelFormat(f"gbrp{depth}", "rgb", False)
    for name in ("yuv411p", "yuv410p", "yuv440p"):
        formats[name] = PixelFormat(name, "yuv", True)
    for chroma in ("420", "422", "444"):
        formats[f"yuvj{chroma}p"] = PixelFormat(f"yuvj{chroma}p", "jpeg", chroma != "444")
    # Packed and semi-planar formats, read as their planar twin, of the same depth and
    # subsampling: zscale takes no packed format, and the swscale step ffmpeg inserts before it
    # then only repacks, which is exact (checked for each, the CLI's tests/test_decode.py). A
    # packed format left to ffmpeg's negotiation gets a conversion of swscale's choosing instead:
    # 8-bit RGB expanded to 16 bits by swscale, 255 → 65283 (DESIGN.md, Input).
    twins = {
        **dict.fromkeys(("nv12", "nv21"), "yuv420p"),
        "nv16": "yuv422p",
        **dict.fromkeys(("nv24", "nv42"), "yuv444p"),
        **{f"p0{d}le": f"yuv420p{d}le" for d in ("10", "12", "16")},
        **{f"p2{d}le": f"yuv422p{d}le" for d in ("10", "12", "16")},
        **{f"p4{d}le": f"yuv444p{d}le" for d in ("10", "12", "16")},
        **dict.fromkeys(("yuyv422", "uyvy422", "yvyu422"), "yuv422p"),
        **dict.fromkeys(("rgb24", "bgr24", "rgb0", "bgr0", "0rgb", "0bgr"), "gbrp"),
        **dict.fromkeys(("rgb48le", "bgr48le", "rgb48be", "bgr48be"), "gbrp16le"),
    }
    for name, planar in twins.items():
        formats[name] = formats[planar]
    return formats


# The pixel formats read, ffmpeg's names (ffprobe's and PyAV's).
PIXEL_FORMATS = _pixel_formats()

# The matrices read, ffprobe's names to zscale's. The constant-luminance ones and ICtCp need the
# transfer function, which is never converted; FCC, SMPTE 240M and YCgCo are rare and would go
# untested (DESIGN.md, Not in the first version).
MATRICES = {"bt709": "709", "bt470bg": "470bg", "smpte170m": "170m", "bt2020nc": "2020_ncl"}

# The chroma locations, ffprobe's names, which are zscale's too.
CHROMA_LOCATIONS = frozenset({"left", "center", "topleft", "top", "bottomleft", "bottom"})

# The words ffprobe has for a tag that is absent, read as none (the CLI's media/probe.py).
UNTAGGED = frozenset({"", "unknown", "unspecified"})

# Each matrix's luma weights of red and blue, by zscale's name (ITU-R BT.709, BT.601 and BT.2020
# non-constant luminance; BT.470 B/G and SMPTE 170M are BT.601's two names).
WEIGHTS = {
    "709": (0.2126, 0.0722),
    "470bg": (0.299, 0.114),
    "170m": (0.299, 0.114),
    "2020_ncl": (0.2627, 0.0593),
}

# What a source's frames are refused for (SourceRefused.kind).
KINDS = (
    # A pixel format not in PIXEL_FORMATS: alpha, a palette, grey, or one never tested.
    "pixel-format",
    # A matrix not in MATRICES.
    "matrix",
    # RGB tagged limited range.
    "rgb-limited",
    # The JPEG family's YUV, full range by definition, tagged limited range.
    "jpeg-limited",
    # A matrix given for an RGB source, which has none.
    "rgb-matrix",
    # Frames whose sides are not multiples of their chroma subsampling (to_rgb).
    "size",
)


def guess_matrix(width: int, height: int) -> str:
    """The matrix of an untagged YUV stream, ffprobe's name: BT.709 from a width of 1280 or a
    height above 576, else BT.601, mpv's rule for untagged video (DESIGN.md, Input)."""
    return "bt709" if width >= 1280 or height > 576 else "smpte170m"


def unsupported(pix_fmt: str) -> str:
    """Why a pixel format not in PIXEL_FORMATS is refused."""
    if pix_fmt == "pal8":
        return "palette pixel format pal8: not the YUV or RGB the conversions are tested for"
    if pix_fmt.startswith(("gray", "ya", "mono")):
        return f"grey pixel format {pix_fmt}: not the YUV or RGB the conversions are tested for"
    if any(part in pix_fmt for part in ("yuva", "gbrap", "rgba", "bgra", "argb", "abgr", "ayuv")):
        return (
            f"pixel format {pix_fmt} has an alpha channel: dropping it would change the picture"
            " wherever it isn't opaque"
        )
    return f"pixel format {pix_fmt} is not supported"


@dataclass(frozen=True)
class Reading:
    """How a stream's planes are read as RGB: the command line's conversion of it, zscale's names.

    - planar: the planar format the planes are (a packed format's twin);
    - matrix: zscale's name, "gbr" for RGB;
    - color_range: "limited" or "full";
    - chroma_location: zscale's name; no effect without chroma subsampling;
    - guessed: the untagged values guessed, which an application says in a warning;
    - matrix_guessed: the matrix among them."""

    planar: str
    matrix: str
    color_range: str
    chroma_location: str
    guessed: tuple[str, ...] = ()
    matrix_guessed: bool = False

    @property
    def format(self) -> PixelFormat:
        return PIXEL_FORMATS[self.planar]


def _refused(kind: str, reason: str) -> SourceRefused:
    return SourceRefused(reason, kind=kind, reason=reason)


def reading(
    pix_fmt: str,
    width: int,
    height: int,
    color_space: str = "",
    color_range: str = "",
    chroma_location: str = "",
    matrix: str | None = None,
) -> Reading:
    """The reading of a stream by its pixel format, size and tags (ffprobe's names: `color_range`
    "tv" or "pc"; "" or a word of UNTAGGED where a tag is absent), the command line's rules: the
    tags where they are, else its guesses (the matrix by the size, guess_matrix; a limited range;
    the chroma sited left, the H.264, HEVC and MPEG-2 default). `matrix`, ffprobe's name, replaces
    the stream's matrix, tagged or not.

    Raises SourceRefused (its kind in KINDS) for what the command line refuses to read: a pixel
    format with alpha, a palette, grey or otherwise not listed; a matrix not listed; RGB tagged
    limited range; JPEG-family YUV tagged limited range; a matrix given for RGB."""
    color_space, color_range, chroma_location = (
        "" if tag in UNTAGGED else tag for tag in (color_space, color_range, chroma_location)
    )
    pixel_format = PIXEL_FORMATS.get(pix_fmt)
    if pixel_format is None:
        raise _refused("pixel-format", unsupported(pix_fmt))
    if pixel_format.family == "rgb":
        if matrix is not None:
            raise _refused(
                "rgb-matrix", f"matrix {matrix} given: the source is RGB, it has no matrix"
            )
        if color_range == "tv":
            raise _refused(
                "rgb-limited",
                "RGB tagged limited range: rare, most likely a mis-tag, and either reading could"
                " be wrong",
            )
        return Reading(pixel_format.planar, "gbr", "full", "left")
    guessed: list[str] = []
    matrix_guessed = False
    if matrix is None:
        matrix = color_space
        if not matrix:
            matrix = guess_matrix(width, height)
            guessed.append(f"matrix {matrix} (from {width}x{height})")
            matrix_guessed = True
    if matrix not in MATRICES:
        raise _refused("matrix", f"matrix {matrix} is not supported (only {', '.join(MATRICES)})")
    if pixel_format.family == "jpeg":
        if color_range == "tv":
            raise _refused(
                "jpeg-limited", f"{pix_fmt}, full range by definition, tagged limited range"
            )
        read_range = "full"
    elif color_range:
        read_range = "full" if color_range == "pc" else "limited"
    else:
        read_range = "limited"
        guessed.append("limited range")
    location = chroma_location
    if location not in CHROMA_LOCATIONS:
        # Untagged: the H.264, HEVC and MPEG-2 default. ffmpeg declares the JPEG family's centre
        # siting itself (MJPEG, the CLI's tests/test_decode.py).
        location = "left"
        if pixel_format.subsampled:
            guessed.append("chroma sited left")
    return Reading(
        pixel_format.planar,
        MATRICES[matrix],
        read_range,
        location,
        tuple(guessed),
        matrix_guessed,
    )


# The master the nodes write (DESIGN.md, Repository and apps, the nodes' VIDEO path): 16-bit
# 4:4:4 YCbCr, BT.709, limited range, as a Reading of its own planes. ComfyUI's Save Video
# re-encodes it right, where a 16-bit RGB master came out coded as BT.601 and tagged wrong
# (impl-videostudy, 2026-11-12).
MASTER = Reading("yuv444p16le", "709", "limited", "left")


def _span(depth: int, limited: bool) -> tuple[float, float, float, float]:
    """(luma's black, luma's span, chroma's neutral, chroma's span) in codes of `depth` bits, as
    zimg takes a range: a limited range's 16 to 235 and 16 to 240 shifted to the depth, exactly;
    a full range's span the depth's whole, its chroma's neutral at half."""
    if limited:
        shift = 1 << (depth - 8)
        return 16.0 * shift, 219.0 * shift, 128.0 * shift, 224.0 * shift
    top = float((1 << depth) - 1)
    return 0.0, top, float(1 << (depth - 1)), top


def _centre(location: str, factor: int, vertical: bool) -> float:
    """Where chroma sample j sits, in luma samples from the start of the `factor` luma samples it
    covers to its own centre: the middle of them (factor / 2) where centred; with the first
    (0.5) where cosited left or top; with the last (factor - 0.5) where cosited bottom."""
    if vertical:
        if location in ("topleft", "top"):
            return 0.5
        if location in ("bottomleft", "bottom"):
            return factor - 0.5
        return factor / 2
    return 0.5 if location in ("left", "topleft", "bottomleft") else factor / 2


Taps = tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]


def _taps(factor: int, centre: float, count: int, size: int, device: "torch.device") -> Taps:
    """The bilinear upsampling of `count` chroma samples to `size` luma samples along one axis, as
    (below, above, weight): each output sample at luma position i + 0.5 interpolated between the
    two chroma samples around it, their positions sited by `centre`, an edge's outside sample its
    edge's own (zimg mirrors its edges, which a two-tap kernel only ever takes as the edge itself).
    Made on `device` once for a call: built on the host, each would be a copy to the card, a wait
    in C a frame (AGENTS.md, the DiT's waits on the host)."""
    import torch

    position = (torch.arange(size, dtype=torch.float64, device=device) + 0.5 - centre) / factor
    first = torch.floor(position)
    below = first.to(torch.long)
    weight = (position - first).to(torch.float32)  # multiples of 1/8 at most: exact
    return below.clamp(0, count - 1), (below + 1).clamp(0, count - 1), weight


def _upsampled(plane: "torch.Tensor", taps: Taps, dim: int) -> "torch.Tensor":
    """`plane` (..., h, w) along `dim` (-1 or -2) through `taps` (_taps)."""
    below, above, weight = taps
    shape = [1] * plane.dim()
    shape[dim] = weight.shape[0]
    low = plane.index_select(dim, below)
    return low + (plane.index_select(dim, above) - low) * weight.view(shape)


def block_words(planar: str) -> str:
    """The chroma block of `planar` in words, a count of 1 in the singular: "2 columns by 2 rows"
    (4:2:0), "2 columns by 1 row" (4:2:2), "4 columns by 1 row" (4:1:1)."""
    rows, columns = PIXEL_FORMATS[planar].subsampling

    def count(number: int, noun: str) -> str:
        return f"{number} {noun}" if number == 1 else f"{number} {noun}s"

    return f"{count(columns, 'column')} by {count(rows, 'row')}"


def check_size(planar: str, width: int, height: int) -> None:
    """Refuse frames of `width` x `height` in the planar format `planar` whose sides are not
    multiples of its chroma subsampling, which zscale refuses ("image dimensions must be divisible
    by subsampling factor"): an odd side at 4:2:0, a width not a multiple of 4 at 4:1:1.

    Raises SourceRefused ("size")."""
    rows, columns = PIXEL_FORMATS[planar].subsampling
    if height % rows or width % columns:
        raise _refused(
            "size",
            f"frames of {width}x{height} in {planar}: their sides are not multiples of its"
            f" chroma subsampling, {block_words(planar)}, which the conversion refuses",
        )


def to_rgb(planes: Sequence["torch.Tensor"], read: Reading) -> "torch.Tensor":
    """A stream's frames as the command line's reader gives them: (T, H, W, 3) float32 RGB in
    [0, 1], each sample a 16-bit code over 65535, on the planes' device.

    `planes`: the three planes of `read.planar`, in its own order (Y, Cb, Cr for YUV; G, B, R
    for gbrp), each (T, h, w) of the codes as the format stores them (8 to 16 bits, in any
    integer dtype: uint8, uint16, int32), the chroma planes subsampled by the format, cropped to
    the picture (a decoder's padded rows are the caller's to cut). The frames are converted one
    at a time, so that what the conversion holds is a frame's float planes beside the output.

    Raises SourceRefused ("size") for frames whose sides are not multiples of the format's chroma
    subsampling, which zscale refuses ("image dimensions must be divisible by subsampling
    factor"), so that the command line reads no such source; ValueError for planes of other
    shapes than the format's."""
    import torch

    first, second, third = planes
    frames, height, width = first.shape
    pixel_format = read.format
    rows, columns = pixel_format.subsampling
    check_size(read.planar, width, height)
    small = (frames, height // rows, width // columns)
    if tuple(second.shape) != small or tuple(third.shape) != small:
        raise ValueError(
            f"planes of {tuple(first.shape)}, {tuple(second.shape)} and {tuple(third.shape)} for"
            f" {read.planar}, whose chroma planes are {small}"
        )
    depth = pixel_format.depth
    device = first.device
    out = torch.empty((frames, height, width, 3), dtype=torch.float32, device=device)
    location = read.chroma_location
    across = _taps(columns, _centre(location, columns, False), small[2], width, device)
    down = _taps(rows, _centre(location, rows, True), small[1], height, device)
    for index in range(frames):
        a = first[index].to(torch.float32)
        if read.matrix == "gbr":
            top = float((1 << depth) - 1)
            g, b, r = (plane[index].to(torch.float32) / top for plane in planes)
            rgb = (r, g, b)
        else:
            black, span, neutral, chroma_span = _span(depth, read.color_range == "limited")
            luma = (a - black) / span
            # Cb and Cr together, (2, h, w).
            chroma = torch.stack((second[index].to(torch.float32), third[index].to(torch.float32)))
            chroma = (chroma - neutral) / chroma_span
            if columns > 1:
                chroma = _upsampled(chroma, across, -1)
            if rows > 1:
                chroma = _upsampled(chroma, down, -2)
            blue, red = chroma[0], chroma[1]
            kr, kb = WEIGHTS[read.matrix]
            kg = 1 - kr - kb
            r = luma + (2 * (1 - kr)) * red
            b = luma + (2 * (1 - kb)) * blue
            g = (luma - kr * r - kb * b) / kg
            rgb = (r, g, b)
        for channel, values in enumerate(rgb):
            out[index, ..., channel] = torch.round(values.clamp(0, 1) * 65535)
    out /= 65535
    return out


def master_planes(rgb: "torch.Tensor") -> "torch.Tensor":
    """RGB frames (T, H, W, 3) float in [0, 1] as the master's planes: (T, 3, H, W) uint16, Y, Cb,
    Cr, 16-bit 4:4:4 BT.709 limited range (MASTER), each sample rounded to nearest, on the
    frames' device. RGB outside [0, 1] is clamped first.

    The planes hold 16-bit RGB to two codes: their limited range spans 56,064 codes of luma and
    57,344 of chroma for RGB's 65,535, so that a sample read back through MASTER (to_rgb, or the
    command line's zscale) is within one code of the RGB written on most samples and two on a few
    of blue's (the CLI's tests/test_yuv_cli.py and the nodes' tests/test_master.py say how many)."""
    import torch

    frames, height, width, _ = rgb.shape
    black, span, neutral, chroma_span = _span(16, True)
    kr, kb = WEIGHTS["709"]
    kg = 1 - kr - kb
    out = torch.empty((frames, 3, height, width), dtype=torch.uint16, device=rgb.device)
    for index in range(frames):
        frame = rgb[index].to(torch.float32).clamp(0, 1)
        r, g, b = frame[..., 0], frame[..., 1], frame[..., 2]
        luma = kr * r + kg * g + kb * b
        planes = (
            black + span * luma,
            neutral + chroma_span * (b - luma) / (2 * (1 - kb)),
            neutral + chroma_span * (r - luma) / (2 * (1 - kr)),
        )
        for plane, values in enumerate(planes):
            out[index, plane] = torch.round(values).clamp(0, 65535).to(torch.uint16)
    return out

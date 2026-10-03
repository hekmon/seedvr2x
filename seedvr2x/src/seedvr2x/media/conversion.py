"""How a source's frames become 16-bit RGB: zscale's parameters, every one explicit, each the
tags' value or a guess (DESIGN.md, Input)."""

from dataclasses import dataclass

from seedvr2x.media.ffmpeg import MediaError
from seedvr2x.media.probe import VideoStream


@dataclass(frozen=True)
class PixelFormat:
    """A decoded pixel format seedvr2x reads."""

    planar: str  # the format zscale is fed: the source's own, or its planar twin
    family: str  # "yuv"; "jpeg", YUV with full range by definition; "rgb"
    subsampled: bool  # chroma subsampled, so its location matters


def _pixel_formats() -> dict[str, PixelFormat]:
    formats: dict[str, PixelFormat] = {}
    # Planar formats zscale takes as they are: ffmpeg inserts no conversion before it (checked
    # for each, tests/test_decode.py).
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
    # Packed and semi-planar formats, fed as their planar twin, of the same depth and subsampling:
    # zscale takes no packed format, and the swscale step ffmpeg inserts before it then only
    # repacks, which is exact (checked for each, tests/test_decode.py). A packed format left to
    # ffmpeg's negotiation gets a conversion of swscale's choosing instead: 8-bit RGB expanded
    # to 16 bits by swscale, 255 → 65283 (DESIGN.md, Input).
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


PIXEL_FORMATS = _pixel_formats()

# The matrices read, ffprobe's names to zscale's. The constant-luminance ones and ICtCp need the
# transfer function, which is never converted; FCC, SMPTE 240M and YCgCo are rare and would go
# untested (DESIGN.md, Not in the first version).
MATRICES = {"bt709": "709", "bt470bg": "470bg", "smpte170m": "170m", "bt2020nc": "2020_ncl"}

# The chroma locations, ffprobe's names, which are zscale's too.
CHROMA_LOCATIONS = frozenset({"left", "center", "topleft", "top", "bottomleft", "bottom"})

# Chroma upsampled with Catmull-Rom (bicubic b = 0, c = 0.5), Keys' cubic with a = -0.5: step 0's
# kernel (NaResize, torchvision's bicubic with antialias), so every interpolation feeding the
# model is the reference pipeline's. zscale's default, bilinear, blurs chroma edges (DESIGN.md,
# Input).
KERNEL = "f=bicubic:param_a=0:param_b=0.5"


@dataclass(frozen=True)
class Conversion:
    """zscale's parameters for a source: from the planar format it is fed to 16-bit RGB."""

    planar: str  # the pixel format zscale is fed
    matrix: str  # zscale's name, "gbr" for RGB
    color_range: str  # "limited" or "full"
    chroma_location: str  # zscale's name; no effect without chroma subsampling
    guessed: tuple[str, ...] = ()  # the untagged values guessed, said in a warning

    def filters(self) -> str:
        """The filter chain, from the decoded frames to gbrp16le."""
        zscale = [
            f"min={self.matrix}",
            f"rin={self.color_range}",
            f"cin={self.chroma_location}",
            # The same on both sides, so never converted: the source's own tags go to the output
            # as they are (DESIGN.md, Colour and shape).
            "pin=unspecified:p=unspecified:tin=unspecified:t=unspecified",
            "m=gbr:r=full",
            "d=none",  # rounded to nearest, deterministic
            KERNEL,
        ]
        return f"format={self.planar},zscale={':'.join(zscale)},format=gbrp16le"

    @property
    def matrix_tag(self) -> str:
        """The matrix ffprobe's name, as tagged, set by --input-matrix or guessed; "" for RGB."""
        if self.matrix == "gbr":
            return ""
        return next(name for name, value in MATRICES.items() if value == self.matrix)

    def describe(self) -> str:
        if self.matrix == "gbr":
            return "RGB"
        return f"YUV {self.matrix_tag}, {self.color_range} range, chroma {self.chroma_location}"


def guess_matrix(width: int, height: int) -> str:
    """The matrix of an untagged YUV stream: BT.709 from a width of 1280 or a height above 576,
    else BT.601, mpv's rule for untagged video (DESIGN.md, Input)."""
    return "bt709" if width >= 1280 or height > 576 else "smpte170m"


def conversion_for(stream: VideoStream, matrix: str | None = None) -> Conversion:
    """The conversion of a stream, with `matrix` (ffprobe's name) for its matrix when given.

    Raises MediaError for what seedvr2x refuses to read: a pixel format with alpha, a palette,
    grey or otherwise not listed; a matrix not listed; RGB tagged limited range; JPEG-family YUV
    tagged limited range."""
    pixel_format = PIXEL_FORMATS.get(stream.pix_fmt)
    if pixel_format is None:
        raise MediaError(_unsupported(stream.pix_fmt))
    if pixel_format.family == "rgb":
        if matrix is not None:
            raise MediaError(f"--input-matrix {matrix}: the source is RGB, it has no matrix")
        if stream.color_range == "tv":
            raise MediaError(
                "RGB tagged limited range: rare, most likely a mis-tag, and either reading could"
                " be wrong"
            )
        return Conversion(pixel_format.planar, "gbr", "full", "left")
    guessed: list[str] = []
    if matrix is None:
        matrix = stream.color_space
        if not matrix:
            matrix = guess_matrix(stream.width, stream.height)
            guessed.append(f"matrix {matrix} (from {stream.width}x{stream.height})")
    if matrix not in MATRICES:
        raise MediaError(f"matrix {matrix} is not supported (only {', '.join(MATRICES)})")
    if pixel_format.family == "jpeg":
        if stream.color_range == "tv":
            raise MediaError(f"{stream.pix_fmt}, full range by definition, tagged limited range")
        color_range = "full"
    elif stream.color_range:
        color_range = "full" if stream.color_range == "pc" else "limited"
    else:
        color_range = "limited"
        guessed.append("limited range")
    chroma_location = stream.chroma_location
    if chroma_location not in CHROMA_LOCATIONS:
        # Untagged: the H.264, HEVC and MPEG-2 default. ffmpeg declares the JPEG family's centre
        # siting itself (MJPEG, tests/test_decode.py).
        chroma_location = "left"
        if pixel_format.subsampled:
            guessed.append("chroma sited left")
    return Conversion(
        pixel_format.planar, MATRICES[matrix], color_range, chroma_location, tuple(guessed)
    )


def _unsupported(pix_fmt: str) -> str:
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

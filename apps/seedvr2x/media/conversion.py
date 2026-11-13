"""How a source's frames become 16-bit RGB: zscale's parameters, every one explicit, each the
tags' value or a guess (DESIGN.md, Input)."""

from dataclasses import dataclass, replace

from media.ffmpeg import MediaError, zscale
from media.probe import VideoStream


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

# Chroma upsampled with bilinear, zscale's own kernel, given explicitly (DESIGN.md, Input, decided
# on 2026-10-20): against it, Catmull-Rom (bicubic b = 0, c = 0.5) gave outputs whose chroma is
# further from the ground truth, PSNR-Cb 0.17 dB and PSNR-Cr 0.25 lower on average over the 8
# full-reference clips, colour edges 0.16 ΔE00 further, ringing beyond 2 levels 0.37 points
# more, the luma within the seeds' spread (numerics.md, The decode's chroma); every
# full-reference run was fed sources read with bilinear.
KERNEL = "f=bilinear"

# --deband's filter, ffmpeg's own, its thresholds 0.012 on all four planes and its other
# parameters at their defaults (DESIGN.md, Input, "Banded sources"): on the user's dark
# animation's sky, at 16 bits, it brought the terraces under the plain bicubic's, where gradfun
# left lines by the edges and a stronger deband turned a dark mountain to speckle (impl-win-4,
# 2026-11-09); without its blur each of the four random
# references must be within the threshold, which by the edges changed 50% of the pixels against
# the default's 69% and left 44% less speckle, the open sky within 4% of the default's, and the
# user's eyes saw less noise by the clouds' edges (2026-11-10). It keeps nothing from one frame
# to the next: a read by the reader's seek gave the frames of a longer read on five H.264
# sources and an all-intra one, its threads its own or one (impl-box-1, impl-win-5). The
# thresholds are 0.012, the default's 0.02 lowered (decided on 2026-11-11, the eyes' verdict on
# animation): the eye could not choose between the two on it, 0.012 moving 1.4 to 3.5% of the
# luma samples by more than an 8-bit code where 0.02 moved 3.8 to 8.2%, and the painted short
# keeping its fine shadows.
DEBAND = "deband=1thr=0.012:2thr=0.012:3thr=0.012:4thr=0.012:blur=0"

# The 16-bit format of each planar format --deband reads, of the same chroma subsampling, which
# deband takes as it is: its in-between values kept, where the source's own format after it
# would cut them back (impl-win-5). The JPEG family's YUV is full range by definition, which the
# conversion's range says, not the format.
DEEP = {
    **{
        f"{family}{chroma}p{depth}": f"yuv{chroma}p16le"
        for family in ("yuv", "yuvj")
        for chroma in ("420", "422", "444")
        for depth in ("", "9le", "10le", "12le", "14le", "16le")
        if family == "yuv" or not depth
    },
    **{f"gbrp{depth}": "gbrp16le" for depth in ("", "9le", "10le", "12le", "14le", "16le")},
}


@dataclass(frozen=True)
class Conversion:
    """zscale's parameters for a source: from the planar format it is fed to 16-bit RGB."""

    planar: str  # the pixel format zscale is fed
    matrix: str  # zscale's name, "gbr" for RGB
    color_range: str  # "limited" or "full"
    chroma_location: str  # zscale's name; no effect without chroma subsampling
    guessed: tuple[str, ...] = ()  # the untagged values guessed, said in a warning
    matrix_guessed: bool = False  # the matrix among them, the one --input-matrix sets
    deband: bool = False  # --deband: DEBAND at 16 bits before the conversion to RGB

    def filters(self) -> str:
        """The filter chain, from the decoded frames to gbrp16le; with deband, DEBAND first, on
        the frames brought to their 16-bit format (DEEP) by zscale, the conversion to RGB then
        built from that format. zscale brings them there exactly, each sample shifted for a
        limited range and scaled for a full one, where swscale's expansion of a full range is
        inexact (ffmpeg.FILTERS); a limited range's is swscale's own shift, the format filter's
        (tests/test_deband.py). Without deband, the chain of before deband was built."""
        if self.deband:
            deep = DEEP[self.planar]
            to_deep = zscale(
                f"min={self.matrix}",
                f"m={self.matrix}",
                f"rin={self.color_range}",
                f"r={self.color_range}",
                f"cin={self.chroma_location}",
                f"c={self.chroma_location}",
                "pin=unspecified:p=unspecified:tin=unspecified:t=unspecified",
                "d=none",
                KERNEL,
            )
            rgb = replace(self, planar=deep, deband=False).filters()
            return f"format={self.planar},{to_deep},format={deep},{DEBAND},{rgb}"
        to_rgb = zscale(
            f"min={self.matrix}",
            f"rin={self.color_range}",
            f"cin={self.chroma_location}",
            # The same on both sides, so never converted: the source's own tags go to the output
            # as they are (DESIGN.md, Colour and shape).
            "pin=unspecified:p=unspecified:tin=unspecified:t=unspecified",
            "m=gbr:r=full",
            "d=none",  # rounded to nearest, deterministic
            KERNEL,
        )
        return f"format={self.planar},{to_rgb},format=gbrp16le"

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


def conversion_for(
    stream: VideoStream, matrix: str | None = None, deband: bool = False
) -> Conversion:
    """The conversion of a stream, with `matrix` (ffprobe's name) for its matrix when given, and
    --deband's filter in it when `deband`.

    Raises MediaError for what seedvr2x refuses to read: a pixel format with alpha, a palette,
    grey or otherwise not listed; a matrix not listed; RGB tagged limited range; JPEG-family YUV
    tagged limited range; with `deband`, a format of a chroma subsampling no 16-bit format has."""
    pixel_format = PIXEL_FORMATS.get(stream.pix_fmt)
    if pixel_format is None:
        raise MediaError(_unsupported(stream.pix_fmt))
    if deband and pixel_format.planar not in DEEP:
        # yuv411p, yuv410p and yuv440p: ffmpeg has no 16-bit format of their subsampling, and
        # deband takes them at 8 bits alone, where its in-between values would be cut back.
        raise MediaError(
            f"--deband: pixel format {stream.pix_fmt}: ffmpeg has no 16-bit format of its chroma"
            " subsampling to deband it in, and at 8 bits the smoothing would be cut back to the"
            " steps it removes; run without --deband"
        )
    conversion = _conversion_for(stream, matrix, pixel_format)
    return replace(conversion, deband=True) if deband else conversion


def _conversion_for(
    stream: VideoStream, matrix: str | None, pixel_format: PixelFormat
) -> Conversion:
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
    matrix_guessed = False
    if matrix is None:
        matrix = stream.color_space
        if not matrix:
            matrix = guess_matrix(stream.width, stream.height)
            guessed.append(f"matrix {matrix} (from {stream.width}x{stream.height})")
            matrix_guessed = True
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
        pixel_format.planar,
        MATRICES[matrix],
        color_range,
        chroma_location,
        tuple(guessed),
        matrix_guessed,
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

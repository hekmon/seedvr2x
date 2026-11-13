"""How a source's frames become 16-bit RGB: zscale's parameters, every one explicit, each the
tags' value or a guess (DESIGN.md, Input). The formats, matrices and chroma locations read, the
guesses for an untagged stream and the refusals are the library's (seedvr2x/yuv.py), which the
nodes convert by in torch where zscale is not to be had; the words of the refusals that name an
option are the command line's."""

from dataclasses import dataclass, replace

from media.ffmpeg import MediaError, zscale
from media.probe import VideoStream
from seedvr2x.errors import SourceRefused
from seedvr2x.yuv import CHROMA_LOCATIONS as CHROMA_LOCATIONS
from seedvr2x.yuv import MATRICES as MATRICES
from seedvr2x.yuv import PIXEL_FORMATS as PIXEL_FORMATS
from seedvr2x.yuv import PixelFormat as PixelFormat
from seedvr2x.yuv import guess_matrix as guess_matrix
from seedvr2x.yuv import reading, unsupported

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
        raise MediaError(unsupported(stream.pix_fmt))
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
    if pixel_format.family == "rgb" and matrix is not None:
        raise MediaError(f"--input-matrix {matrix}: the source is RGB, it has no matrix")
    try:
        read = reading(
            stream.pix_fmt,
            stream.width,
            stream.height,
            stream.color_space,
            stream.color_range,
            stream.chroma_location,
            matrix,
        )
    except SourceRefused as error:
        raise MediaError(str(error)) from error
    return Conversion(
        read.planar,
        read.matrix,
        read.color_range,
        read.chroma_location,
        read.guessed,
        read.matrix_guessed,
    )

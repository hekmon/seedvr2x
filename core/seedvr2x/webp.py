"""The lossless WebP writer of a still picture (DESIGN.md, Still images, Output; The image server,
A request): 8 bits, RGB, with the source's ICC profile and no orientation: the caller writes the
picture upright (picture.upright). Pillow (libwebp) encodes; the module quantises as png.py's
8-bit path does, which is the video writers' (apps/seedvr2x/media/writer.py's quantise).

A WebP holds what a PNG does not carry the same way, and not all of it: 8 bits per sample, no grey
mode, and, of PNG's colour chunks, none but an ICC profile (its sRGB, which is WebP's default,
needs no chunk). check() refuses a picture it would change the colour of, before the models run,
naming the way out, a PNG. The bytes are a function of the pixels and the options: libwebp's
encoder is deterministic, and nothing of the time or the machine is added."""

import io
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from PIL import Image

from seedvr2x import png
from seedvr2x.errors import Refusal

# libwebp's effort in lossless mode: Pillow's `method` (0 fastest, 6 slowest) and `quality` (0 to
# 100, which in lossless mode is the effort too, not a loss). Measured on a 32-core machine busy
# with another job (libwebp 1.6, Pillow 12.3, one thread), on png.py's two pictures: a 4K
# (3,840 x 2,880) picture drawn from a 320 x 240 photograph, upscaled in float with a smooth
# resampler, "smooth", and the same with a grain of 0.7 level of 8 bits, "grain". Size in MB, time
# in s (png.write at 8 bits, for reference: smooth 1.26 MB 0.63 s, grain 10.28 MB 4.51 s):
#   method, quality   smooth          grain
#   0, 0           2.05  0.31      9.93  0.34
#   0, 50          2.04  0.19      9.87  0.91
#   0, 100         1.23  0.42      9.84  7.63
#   4, 0           0.82  0.74      7.36  1.22
#   4, 25          0.82  0.92      7.36  1.36
#   4, 50          0.73  1.00      7.36  2.11
#   4, 75          0.73  1.19      7.35  3.53
#   4, 100         0.73  1.37      7.36  5.35
#   5, 50          0.73  1.20      7.36  2.20
#   6, 0           0.82  0.83      7.36  1.17
#   6, 50          0.73  1.18      7.36  2.30
#   6, 100         0.73  4.39      7.35  21.89
# Method 4 and quality 50: method 0 is 35% bigger on the grain and 2.8 times on the smooth picture,
# and methods 5 and 6 save nothing on 4; at method 4, quality 50 is 11% smaller than 0 on the
# smooth picture and as small as 100 (the grain's size does not move with the quality) in 1.4 and
# 2.5 times less time. Against the PNG of the same 8 bits, 42% smaller (smooth) and 28% smaller
# (grain), in 1.6 and 0.5 times its time.
# A 4K picture takes 1 to 2 s, an 8K one 4 times that, which is little against the model's minutes.
METHOD = 4
QUALITY = 50

# The chunk types of PNG colour that WebP has no chunk for (a PNG's sRGB alone is WebP's
# default, taken as nothing). Without a profile, the picture's colour would be taken as sRGB's.
_UNHELD = (b"cICP", b"cHRM", b"gAMA")

# The largest side of a WebP (the bitstream's 14 bits: 16,383).
MAX_SIDE = 16383


def why_not(bits: int, icc: bytes | None, chunks: Sequence[bytes]) -> str | None:
    """Why a lossless WebP cannot hold a picture, as a sentence naming the way out (a PNG), or None
    when it can. bits is the depth the picture is stored at (picture.Picture.bits: 8 or 16), icc
    and chunks are the picture's (Picture.icc, Picture.chunks, each chunk whole as a file holds
    it: length, type, data, CRC). Used before the models run and, by the image server, to answer
    PNG where the client takes it."""
    if bits != 8:
        return (
            f"a lossless WebP holds 8 bits per sample and this picture is stored at {bits}, which"
            " it would round away: write a PNG (a .png file), which holds them"
        )
    if icc is None:
        found = [kind.decode() for kind in _UNHELD if any(c[4:8] == kind for c in chunks)]
        if found:
            return (
                f"a lossless WebP holds no {', '.join(found)} colour chunk and this picture has no"
                " ICC profile to carry its colour instead, so its colour would change: write a"
                " PNG (a .png file), which holds them"
            )
    return None


def check(bits: int, icc: bytes | None, chunks: Sequence[bytes]) -> None:
    """Refuse (errors.Refusal) a picture a lossless WebP cannot hold, in why_not's words."""
    if (why := why_not(bits, icc, chunks)) is not None:
        raise Refusal(why)


def write(
    rgb: npt.NDArray[np.float32],
    *,
    grey: bool,
    icc: bytes | None,
    chunks: Sequence[bytes],
) -> bytes:
    """The file of a picture: rgb (H, W, 3) float32, any finite values (clipped to [0, 1]), any
    memory layout, written at 8 bits, rounded to the nearest code as png.write rounds. grey takes
    the mean of the three planes, in float32 and before the quantisation, as png.write does, and
    writes it as RGB with three equal planes: WebP has no grey mode.

    The ICC profile icc if it is not None; no EXIF block, the picture being written upright
    (picture.upright, DESIGN.md, Still images, Orientation). chunks, the PNG colour chunks of the
    source (Picture.chunks), are not written: a WebP has none, and cICP, cHRM or gAMA without a
    profile is refused, as check() does. The source's depth is the caller's to check(): the array
    has lost it.

    Refusal for a side over MAX_SIDE; ValueError for an array that is not a picture."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or 0 in rgb.shape:
        raise ValueError(f"rgb must be (H, W, 3) and not empty, not {rgb.shape}")
    check(8, icc, chunks)
    height, width = rgb.shape[:2]
    if max(height, width) > MAX_SIDE:
        raise Refusal(
            f"a WebP is at most {MAX_SIDE:,} pixels on a side and this picture is {width}x{height}:"
            " write a PNG (a .png file)"
        )
    codes = png.quantised(rgb, 8, grey)
    if grey:
        codes = np.repeat(codes, 3, axis=2)
    options: dict[str, object] = {"lossless": True, "method": METHOD, "quality": QUALITY}
    if icc is not None:
        options["icc_profile"] = icc
    out = io.BytesIO()
    Image.fromarray(codes, "RGB").save(out, "WEBP", **options)
    return out.getvalue()

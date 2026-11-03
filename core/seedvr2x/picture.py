"""A still picture read from its bytes (DESIGN.md, Still images): what a file is by its content,
and its pixels as the model takes them, with what the output carries of the file (ICC profile,
PNG colour chunks, EXIF orientation).

Pillow identifies and decodes. torchvision's decoder reads the 16-bit PNGs whose colour a
Pillow decode would cut to 8 bits (imported where it is needed: the module imports no torch).
The library takes bytes, never a path, so that the command line and an image server read a
picture the same way. Every refusal says why and what to do."""

import io
import logging
import re
import shlex
import struct
import zlib
from collections import deque
from dataclasses import dataclass
from typing import BinaryIO, cast

import numpy as np
import numpy.typing as npt
from PIL import Image

from seedvr2x.errors import Refusal

logger = logging.getLogger(__name__)

# Pillow's names of the formats read as pictures, the only plugins asked (DESIGN.md: PNG, JPEG,
# WebP, TIFF, BMP and a GIF of one frame). The file's content tells which, never its name.
FORMATS = ("PNG", "JPEG", "WEBP", "TIFF", "BMP", "GIF")

# A JPEG with a second image after the first (Pillow's "MPO": a camera's stereo pair, a phone
# photo carrying its gain map as the second image) is read as its first image, said at WARNING
# (information is left out), when True; refused like a file of several frames, when False.
# Pillow opens it as format "MPO" whatever formats are asked (its JPEG plugin hands over to the MPO
# one by the file's MP marker), and counts the images in n_frames.
MPO_FIRST_IMAGE = True

# The rate a refusal's command for an animated picture or a stream of JPEGs gives the video it
# makes, in frames a second, the user's to choose (_several_frames): a GIF's delays are uneven,
# which the video path refuses (media/rate.py's timing refusal), so the command fixes one.
VIDEO_RATE = 10
MJPEG_RATE = 25

# The EXIF tag of the orientation, 1 to 8.
_ORIENTATION = 0x0112

# Pillow's modes with an alpha plane, and the greys: "1" is bilevel, expanded to 0 and 255.
_ALPHA_MODES = ("RGBA", "LA", "PA")
_GREY_MODES = ("L", "LA", "1")
_8_BIT_MODES = ("RGB", "P", "L", "1", *_ALPHA_MODES)

# The PNG colour chunks the output carries whole (DESIGN.md, Still images, Colour). Pillow reads
# no cICP, and the others' values are not kept by it either.
_COLOUR_CHUNKS = (b"cICP", b"cHRM", b"gAMA", b"sRGB")

# cICP's transfer characteristics of HDR (ITU-T H.273): 16 is PQ, 18 is HLG.
_HDR_TRANSFERS = {16: "PQ", 18: "HLG"}

# A PNG's colour types with a grey plane, and with an alpha plane, and the channels each has.
_PNG_GREY = (0, 4)
_PNG_ALPHA = (4, 6)
_PNG_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}

# The TIFF tags and values that tell a camera's raw file built on TIFF (DNG, NEF, CR2, ARW, PEF,
# SRW, ERF, 3FR, DCR, KDC, MOS and the like, which all start as TIFF files and which Pillow's TIFF
# reader opens, or refuses as an unknown pixel mode), by their sources:
# NewSubfileType (TIFF 6.0, section 8): bit 0 set marks an image as a reduced-resolution version
# of another image in the file, the thumbnail or preview a raw file puts in its first IFD.
_NEW_SUBFILE_TYPE = 254
_REDUCED = 1
# PhotometricInterpretation (TIFF 6.0, section 8, extended): 32803 is a colour filter array
# (TIFF/EP, ISO 12234-2, and DNG 1.7), the sensor's mosaic before any demosaicing; 34892 is
# LinearRaw (DNG 1.7), a sensor's samples already demosaiced but not developed. Neither is a
# colour picture, and Pillow has no pixel mode for either.
_PHOTOMETRIC = 262
_RAW_PHOTOMETRIC = {32803: "a colour filter array", 34892: "linear raw samples"}
# SubIFDs (TIFF Technical Note 1, Adobe PageMaker 6.0 TIFF Technical Notes): where TIFF/EP and
# DNG keep the full image under a first IFD that is the preview.
_SUB_IFDS = 330
# DNGVersion (DNG 1.7, section "DNG tags"): four bytes in the first IFD of every DNG, whose
# presence says the file is one, whatever its pages hold.
_DNG_VERSION = 50706
# Canon's CR2 puts "CR", its major and minor versions after the 8 bytes of TIFF header (the
# format as ExifTool documents it).
_CR2_MAGIC = b"CR"
# Classic TIFF's headers (TIFF 6.0, section 2): the byte order, then 42. BigTIFF has no raw files.
_TIFF_ORDER = {b"II*\x00": "<", b"MM\x00*": ">"}
# IFDs walked at most (the chain and the first IFD's SubIFDs): a raw file holds a handful, and a
# damaged or hostile file's chain must end.
_MAX_IFDS = 64
# TIFF's field types a tag read here has (TIFF 6.0, section 2; IFD is TIFF Technical Note 1's).
_SHORT, _LONG, _IFD = 3, 4, 13


@dataclass(frozen=True)
class Picture:
    """A still picture as it is stored and as the model takes it."""

    # (H, W, 3) float32 in [0, 1], C-contiguous: the stored code over 255 (65535 at 16 bits), grey
    # and palette expanded exactly. rint(x * 65535) (255) gives the code back for every code.
    pixels: npt.NDArray[np.float32]
    format: str  # one of FORMATS
    bits: int  # 8 or 16: the depth the output is written at (8 for anything stored under 16)
    grey: bool  # stored grey, in any grey mode: the output is written grey
    icc: bytes | None  # the ICC profile as stored (decompressed), for the output
    chunks: tuple[bytes, ...]  # a PNG's cICP, cHRM, gAMA and sRGB, whole, in file order
    orientation: int  # EXIF orientation 1 to 8, 1 when there is none; pixels are not rotated
    opaque_alpha_dropped: bool  # an alpha plane opaque everywhere was left out


def identify(file: BinaryIO) -> str | None:
    """The format of an open file read from its start, if its content is one of FORMATS, else
    None (a video, an empty file, text, anything else). Reads the header alone and decodes
    nothing, so it is quick on a file of any size. A file that starts as one of FORMATS but is
    damaged is told so too, read() refusing it. The file's position is left where it was."""
    position = file.tell()
    try:
        try:
            with Image.open(file, formats=FORMATS) as img:
                return _format(img)
        except Exception:
            # A header too damaged to open (or a size Pillow's guard refuses) still starts as its
            # format, whose reading then says what is wrong; any other content is not a picture.
            file.seek(0)
            return _sniffed(file.read(32))
    finally:
        file.seek(position)


def read(data: bytes, name: str = "FILE") -> Picture:
    """The picture in data, a file's whole content; name is the file's, said in the refusals'
    commands. Raises Refusal for what seedvr2x does not read: several frames, transparency, a
    mode other than RGB, grey and palette, a 16-bit TIFF, HDR, a damaged file, a size beyond
    Pillow's guard, a camera's raw file built on TIFF."""
    # Before Pillow opens it: it opens a raw file's preview as a picture, and refuses a colour
    # filter array or linear raw image as an unknown pixel mode, which the damaged-file refusal
    # below would then call a header it cannot read.
    why = _raw_why(data)
    if why is not None:
        raise _raw(name, why)
    try:
        img = Image.open(io.BytesIO(data), formats=FORMATS)
    except Image.DecompressionBombError as error:
        raise _too_large(name, error) from error
    except Image.UnidentifiedImageError as error:
        # Pillow says so too of a header its plugin cannot read: told apart by the first bytes.
        fmt = _sniffed(data[:32])
        if fmt == "PNG":
            _walk_png(data, name)  # says which chunk is wrong, if it is one
        if fmt is not None:
            raise _damaged(name, ValueError(f"its {fmt} header cannot be read")) from error
        raise Refusal(
            f"{name} is not a picture seedvr2x reads: it reads {', '.join(FORMATS)} (told by"
            " the file's content, never its name), and a video as a video"
        ) from error
    except Exception as error:
        raise _damaged(name, error) from error
    with img:
        return _read(img, data, name)


def _read(img: Image.Image, data: bytes, name: str) -> Picture:
    _say_large(img.size, name)
    fmt = _format(img)
    frames = getattr(img, "n_frames", 1)
    if frames > 1 and not (MPO_FIRST_IMAGE and img.format == "MPO"):
        raise _several_frames(name, fmt, frames)
    if img.format == "JPEG":
        # Pillow reads the first image of JPEGs one after another (a raw MJPEG stream) and says
        # nothing of the rest.
        images = _jpeg_images(data)
        if images > 1:
            raise _several_frames(name, fmt, images, stream=True)
    if fmt == "TIFF" and _sixteen_bit_tiff(img):
        raise _tiff16(name, img.info.get("icc_profile") is not None)
    png = _walk_png(data, name) if fmt == "PNG" else None
    try:
        if png is not None and png.depth == 16:
            pixels, grey, dropped = _decoded16(data, png, name)
        else:
            pixels, grey, dropped = _decoded8(img, name)
        icc = img.info.get("icc_profile")
        orientation = _orientation(img, name)
    except Refusal:
        raise
    except Image.DecompressionBombError as error:
        raise _too_large(name, error) from error
    except Exception as error:
        raise _damaged(name, error) from error
    if dropped:
        logger.info(
            "%s: its alpha plane is opaque everywhere, left out: the picture is unchanged", name
        )
    if frames > 1:
        # At WARNING: information is lost (a stereo pair's second view, an Ultra HDR photo's gain
        # map), unlike an opaque alpha plane dropped.
        logger.warning(
            "%s: %d images in the file (a JPEG with a second image after the first): the first is"
            " read, %d left out (a stereo pair's second view, a photo's gain map), and the picture"
            " is the first image alone",
            name,
            frames,
            frames - 1,
        )
    return Picture(
        pixels=pixels,
        format=fmt,
        bits=16 if png is not None and png.depth == 16 else 8,
        grey=grey,
        icc=icc if isinstance(icc, bytes) else None,
        chunks=png.chunks if png is not None else (),
        orientation=orientation,
        opaque_alpha_dropped=dropped,
    )


def _say_large(size: tuple[int, int], name: str) -> None:
    """One line, at INFO, for a picture that Pillow's check against decompression bombs warns
    of (more pixels than Image.MAX_IMAGE_PIXELS, which seedvr2x leaves as it is, and up to twice
    that: over it, Pillow's open refuses and _too_large says so). The line is seedvr2x's own, said
    from the size the file declares; Pillow's warning at open is left to the application to
    silence, once and for the whole process (the CLI's main.py does, and the image server will):
    warnings.catch_warnings, which would silence it here, swaps the module's global filters and is
    not safe on several threads (Python 3.13). INFO because nothing is wrong or lost: a camera or
    a scanner makes such pictures, and a file made to fill the memory is the case _too_large
    refuses."""
    limit = Image.MAX_IMAGE_PIXELS
    width, height = size
    if limit is None or max(1, width) * max(1, height) <= limit:
        return
    logger.info(
        "%s: %dx%d, %.1f megapixels, over the %.1f at which Pillow, the library that reads"
        " pictures here, starts to guard against files made to fill the memory (it refuses over"
        " %.1f): read as a picture, a camera's or a scanner's being this large",
        name,
        width,
        height,
        width * height / 1e6,
        limit / 1e6,
        2 * limit / 1e6,
    )


def _format(img: Image.Image) -> str:
    """The format, Pillow's MPO being a JPEG with more than one image."""
    return "JPEG" if img.format == "MPO" else str(img.format)


def _sniffed(head: bytes) -> str | None:
    """The format of FORMATS whose plugin accepts the first bytes of a file, if any."""
    for fmt in FORMATS:
        accept = Image.OPEN[fmt][1]
        if accept is not None and accept(head):
            return fmt
    return None


# The pixels ------------------------------------------------------------------------------------


def _decoded8(img: Image.Image, name: str) -> tuple[npt.NDArray[np.float32], bool, bool]:
    """(pixels, grey, alpha dropped) of a picture Pillow decodes at 8 bits, exactly: the same
    values as ffmpeg's decode for PNG, BMP, TIFF, GIF and lossless WebP."""
    mode = img.mode
    if mode not in _8_BIT_MODES:
        raise Refusal(
            f"{name} is in the {mode!r} colour mode, which seedvr2x does not read: it reads RGB,"
            " grey and palette pictures. Convert it to RGB in an image editor, keeping its colour"
            " profile, and upscale the result"
        )
    dropped = False
    source = img
    if mode in _ALPHA_MODES or "transparency" in img.info:
        # convert gives the alpha plane of a transparent palette or colour too (a tRNS chunk, a
        # GIF's transparent index), the pictures that carry no alpha band.
        rgba = img.convert("RGBA")
        low, _ = rgba.getchannel("A").getextrema()
        if low != 255:
            raise _transparent(name)
        dropped = True
        if mode == "P":
            # Pillow warns of a palette with transparency in bytes converted to RGB; to RGBA first
            # it does not, and the colours are the same.
            source = rgba
    grey = mode in _GREY_MODES
    plain = source.convert("L" if grey else "RGB") if source.mode not in ("RGB", "L") else source
    codes = np.asarray(plain)
    return _expanded(codes, 255), grey, dropped


def _expanded(codes: npt.NDArray[np.uint8], top: float) -> npt.NDArray[np.float32]:
    """Codes (H, W) or (H, W, 3) over top, as (H, W, 3) float32 in C order: a grey plane repeated.
    The division is float32's, the one the 16-bit path makes."""
    out = np.empty((*codes.shape[:2], 3), np.float32)
    np.divide(codes if codes.ndim == 3 else codes[..., np.newaxis], np.float32(top), out=out)
    return out


def _decoded16(data: bytes, png: "_Png", name: str) -> tuple[npt.NDArray[np.float32], bool, bool]:
    """(pixels, grey, alpha dropped) of a 16-bit PNG, by torchvision's libpng decoder: Pillow cuts
    RGB, RGBA and grey-alpha at 16 bits to 8. Bit for bit ffmpeg's rgb48 and gray16 decodes on
    the pictures tested (tests/test_picture.py)."""
    import torch

    # torchvision.io ships no type stubs.
    from torchvision.io import (  # pyright: ignore[reportMissingTypeStubs]
        ImageReadMode,
        decode_image,
    )

    tensor = decode_image(
        torch.frombuffer(bytearray(data), dtype=torch.uint8), mode=ImageReadMode.UNCHANGED
    )
    planes = tensor.numpy()
    expected = _PNG_CHANNELS[png.colour_type]
    if planes.shape[0] != expected:
        raise _damaged(name, ValueError(f"{planes.shape[0]} channels decoded, {expected} stored"))
    grey = png.colour_type in _PNG_GREY
    colour = planes[:1] if grey else planes[:3]
    dropped = False
    if png.colour_type in _PNG_ALPHA:
        if planes[expected - 1].min() != 65535:
            raise _transparent(name)
        dropped = True
    elif png.transparency is not None:
        # A colour key (tRNS): the pixels equal to it are transparent.
        key = np.array(struct.unpack(f">{expected}H", png.transparency), np.uint16)
        if (colour == key[:, np.newaxis, np.newaxis]).all(axis=0).any():
            raise _transparent(name)
        dropped = True
    top = np.float32(65535)
    pixels = np.empty((*colour.shape[1:], 3), np.float32)
    for channel in range(3):
        pixels[..., channel] = colour[0 if grey else channel] / top
    return pixels, grey, dropped


# A JPEG's images ------------------------------------------------------------------------------

# A JPEG's marker after a scan's entropy-coded data: FF, any number of FF fill bytes, then a code
# that is not 00 (a stuffed FF), RST0-7 (D0-D7, inside the scan) or FF (ITU-T T.81: a data byte FF
# is always followed by 00 in the scan, the markers' codes are 01 to FE).
_AFTER_SCAN = re.compile(rb"\xff+[\x01-\xcf\xd8-\xfe]")
_SOI = b"\xff\xd8\xff"
# The markers without a length (T.81, Table B.1): TEM, RST0-7, SOI, EOI.
_STANDALONE = frozenset({0x01, *range(0xD0, 0xDA)})


def _jpeg_end(data: bytes, start: int) -> int | None:
    """The offset just past the JPEG image that starts at `start` (its SOI), walked by its marker
    segments' lengths and its scans' entropy-coded data, so that the SOI and EOI of an EXIF
    thumbnail inside an APP1 segment, which a search for FF D9 would take for the image's end, are
    skipped as the segment they sit in; None when the walk meets what is no JPEG (a cut file, a
    length beyond the end, a marker missing) and so ends nowhere."""
    size = len(data)
    position = start + 2
    while True:
        while (
            position < size
            and data[position] == 0xFF
            and data[position + 1 : position + 2] == b"\xff"
        ):
            position += 1  # fill bytes before a marker
        if position + 2 > size or data[position] != 0xFF:
            return None
        code = data[position + 1]
        position += 2
        if code == 0xD9:
            return position
        if code == 0xD8 or code == 0x00:
            return None  # a second SOI inside an image, or a stuffed byte outside a scan
        if code in _STANDALONE:
            continue
        if position + 2 > size:
            return None
        position += int.from_bytes(data[position : position + 2], "big")
        if position > size:
            return None
        if code == 0xDA:  # a scan: its entropy-coded data, up to the next marker
            found = _AFTER_SCAN.search(data, position)
            if found is None:
                return None
            position = found.start()


def _jpeg_images(data: bytes) -> int:
    """How many JPEG images the file holds one after another, 1 for a single image: after each
    image's end (_jpeg_end), padding of zero bytes is skipped, and another SOI followed by a
    marker (FF D8 FF) is another image. A walk that ends nowhere is Pillow's to refuse or read:
    1 for the first, and an image started after the last is counted whole."""
    images = 0
    start = 0
    while True:
        images += 1
        end = _jpeg_end(data, start)
        if end is None:
            return images
        while end < len(data) and data[end] == 0:
            end += 1
        if not data.startswith(_SOI, end):
            return images
        start = end


# Metadata --------------------------------------------------------------------------------------


def _orientation(img: Image.Image, name: str) -> int:
    """The EXIF orientation (JPEG, PNG eXIf, WebP, TIFF), 1 when there is none or it is not 1 to
    8, said at WARNING: the file's pixels are processed as stored, the value carried."""
    try:
        value = img.getexif().get(_ORIENTATION)
    except Exception as error:
        logger.warning("%s: its EXIF data cannot be read (%s): orientation 1 taken", name, error)
        return 1
    if value is None:
        return 1
    if isinstance(value, int) and 1 <= value <= 8:
        return value
    logger.warning("%s: its EXIF orientation is %r, which is not 1 to 8: taken as 1", name, value)
    return 1


def _sixteen_bit_tiff(img: Image.Image) -> bool:
    """Whether a TIFF's samples are 16 bits: Pillow reads RGB at 16 bits as 8 (BitsPerSample,
    tag 258, holds the truth)."""
    samples: object = getattr(img, "tag_v2", {}).get(258, ())
    if isinstance(samples, tuple):
        return 16 in cast(tuple[object, ...], samples)
    return samples == 16


# A camera's raw file ----------------------------------------------------------------------------

_Entry = tuple[int, int, int]  # an IFD entry: its field type, its count, the offset of its value


def _raw_why(data: bytes) -> str | None:
    """Why `data` is a camera's raw file built on TIFF, in words that complete "it is a raw file
    because ...", or None when no marker of one is found (a file that is no TIFF, or a damaged
    one, included: Pillow's reading says what is wrong with it). Reads the TIFF header and the
    IFDs' entries from the bytes, never the image: the first image Pillow reads is a raw file's
    preview, and its other images are the raw data, which Pillow has no pixel mode for.

    The markers, by where they are written (the constants above give their sources): CR2's
    header; the DNGVersion tag; the first IFD marked a reduced-resolution version of another
    image; a colour filter array or linear raw PhotometricInterpretation in any IFD of the chain
    (the pages Pillow counts), or in a SubIFD of the first, where TIFF/EP and DNG put the full
    image under a preview."""
    order = _TIFF_ORDER.get(data[:4])
    if order is None or len(data) < 8:
        return None
    if data[8:10] == _CR2_MAGIC:
        return "its header is Canon CR2's: CR follows the TIFF header"
    pending = deque(struct.unpack_from(f"{order}I", data, 4))
    seen: set[int] = set()
    first = True
    while pending and len(seen) < _MAX_IFDS:
        offset = pending.popleft()
        if offset in seen or (entries := _entries(data, order, offset)) is None:
            continue
        seen.add(offset)
        tags, following = entries
        if first:
            if _DNG_VERSION in tags:
                return f"it carries DNG's version tag (DNGVersion, tag {_DNG_VERSION})"
            if any(
                value & _REDUCED for value in _numbers(data, order, tags.get(_NEW_SUBFILE_TYPE))
            ):
                return (
                    "its first image is marked a reduced-resolution version of another image in"
                    f" the file (NewSubfileType, tag {_NEW_SUBFILE_TYPE}), as the preview a raw"
                    " file holds first is"
                )
            pending.extend(_numbers(data, order, tags.get(_SUB_IFDS)))
            first = False
        for value in _numbers(data, order, tags.get(_PHOTOMETRIC)):
            if value in _RAW_PHOTOMETRIC:
                return (
                    f"one of its images holds {_RAW_PHOTOMETRIC[value]} (PhotometricInterpretation"
                    f" {value}), a camera's sensor samples and no colour picture"
                )
        if following:
            pending.append(following)
    return None


def _entries(data: bytes, order: str, offset: int) -> tuple[dict[int, _Entry], int] | None:
    """The entries of the IFD at `offset` by tag, and the offset of the next IFD (0 for none); None
    when the IFD does not fit in the file."""
    if offset < 8 or offset + 2 > len(data):
        return None
    (count,) = struct.unpack_from(f"{order}H", data, offset)
    end = offset + 2 + 12 * count
    if end + 4 > len(data):
        return None
    tags: dict[int, _Entry] = {}
    for at in range(offset + 2, end, 12):
        tag, kind, number = struct.unpack_from(f"{order}HHI", data, at)
        tags[tag] = (kind, number, at + 8)
    (following,) = struct.unpack_from(f"{order}I", data, end)
    return tags, following


def _numbers(data: bytes, order: str, entry: _Entry | None) -> list[int]:
    """The values of an entry of SHORT, LONG or IFD type, the first _MAX_IFDS at most; empty for
    no entry, another type or a value outside the file. A value of 4 bytes or fewer is in the
    entry itself, else at the offset the entry holds (TIFF 6.0, section 2)."""
    if entry is None:
        return []
    kind, number, field = entry
    size = {_SHORT: 2, _LONG: 4, _IFD: 4}.get(kind)
    if size is None:
        return []
    at = field if number * size <= 4 else struct.unpack_from(f"{order}I", data, field)[0]
    number = min(number, _MAX_IFDS)
    if at + number * size > len(data):
        return []
    return list(struct.unpack_from(f"{order}{('H' if size == 2 else 'I') * number}", data, at))


# PNG's chunks ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Png:
    depth: int
    colour_type: int
    chunks: tuple[bytes, ...]
    transparency: bytes | None


def _walk_png(data: bytes, name: str) -> _Png:
    """The chunks of a PNG up to the first IDAT, from its bytes: the IHDR's depth and colour type
    (Pillow's mode tells neither at 16 bits), the colour chunks whole, the tRNS data. Every chunk
    walked is checked: its length, and its CRC."""
    position = 8
    end = len(data)
    first = True
    kept: list[bytes] = []
    transparency = None
    depth = colour_type = 0
    while True:
        if position + 12 > end:
            raise _damaged(name, ValueError("the PNG ends before its image data"))
        (length,) = struct.unpack_from(">I", data, position)
        kind = data[position + 4 : position + 8]
        stop = position + 12 + length
        if stop > end:
            raise _damaged(
                name, ValueError(f"the PNG's {kind.decode('latin-1')} chunk is cut short")
            )
        body = data[position + 8 : position + 8 + length]
        (crc,) = struct.unpack_from(">I", data, stop - 4)
        if zlib.crc32(body, zlib.crc32(kind)) != crc:
            raise _damaged(
                name, ValueError(f"the PNG's {kind.decode('latin-1')} chunk fails its CRC")
            )
        if first:
            if kind != b"IHDR" or length != 13:
                raise _damaged(name, ValueError("the PNG does not start with its IHDR"))
            depth, colour_type = body[8], body[9]
            first = False
        elif kind == b"IDAT":
            break
        elif kind in _COLOUR_CHUNKS:
            if kind == b"cICP":
                _check_cicp(body, name)
            kept.append(data[position:stop])
        elif kind == b"tRNS":
            transparency = body
        position = stop
    if colour_type not in _PNG_CHANNELS:
        raise _damaged(name, ValueError(f"colour type {colour_type}"))
    if depth == 16 and transparency is not None and colour_type not in _PNG_ALPHA:
        if len(transparency) != 2 * _PNG_CHANNELS[colour_type]:
            raise _damaged(name, ValueError("the PNG's tRNS chunk has the wrong length"))
    return _Png(depth, colour_type, tuple(kept), transparency)


def _check_cicp(body: bytes, name: str) -> None:
    """A cICP chunk (four bytes: primaries, transfer, matrix, range): HDR is refused, as for video
    (media/source.py's HDR)."""
    if len(body) != 4:
        raise _damaged(name, ValueError("the PNG's cICP chunk is not 4 bytes"))
    hdr = _HDR_TRANSFERS.get(body[1])
    if hdr is not None:
        raise Refusal(
            f"{name} is an HDR picture (its cICP chunk says the {hdr} transfer): the model was"
            " trained on SDR video, and what it makes of HDR-coded pixels is unmeasured. For an"
            " SDR upscale, tone-map the picture to SDR first, a grading choice for your own"
            " tools, as a gamut conversion is"
        )


# The refusals ----------------------------------------------------------------------------------


def _damaged(name: str, error: Exception) -> Refusal:
    return Refusal(
        f"{name} is damaged or truncated, and cannot be decoded ({error}). Make a good copy of it,"
        " or save it again from the program that made it"
    )


def _too_large(name: str, error: Exception) -> Refusal:
    # Pillow raises over twice Image.MAX_IMAGE_PIXELS (178,956,970 pixels at Pillow 12.3), its guard
    # against decompression bombs; its message holds the picture's pixel count, which a header
    # alone gives, so the decode need not be tried.
    limit = 2 * (Image.MAX_IMAGE_PIXELS or 0)
    counted = re.search(r"\(([\d,]+) pixels\)", str(error))
    count = f"{int(counted[1].replace(',', '')):,} pixels" if counted else "more pixels"
    # Into an 8192-px square, the ratio kept: 67 Mpixels at most, and always smaller, since a
    # picture over the limit has a side over 8192 (8192 x 8192 is 67,108,864 pixels).
    shrink = "scale=8192:8192:force_original_aspect_ratio=decrease"
    return Refusal(
        f"{name} has {count}, over the {limit:,} pixels that Pillow, the library that reads"
        " pictures here, takes in: its limit against decompression bombs (files made to fill the"
        " memory), which seedvr2x leaves as it is. Make a smaller copy of it first, in an image"
        " editor or with ffmpeg (at most 8192 pixels on a side), and upscale the copy:"
        f" ffmpeg -i {shlex.quote(name)} -vf {shlex.quote(shrink)}"
        f" {shlex.quote(f'{name}.small.png')}"
    )


def _transparent(name: str) -> Refusal:
    # The command, checked on a PNG with transparent and half-transparent pixels: the picture on
    # white to within half a level of 8 bits, at any size (drawbox's replace=1 writes white over
    # a copy, alpha included; without it, the white is blended and the result is wrong).
    graph = (
        "format=rgba,split[a][b];[a]drawbox=c=white:t=fill:replace=1[bg];"
        "[bg][b]overlay=format=rgb,format=rgb24"
    )
    return Refusal(
        f"{name} has transparent pixels, and seedvr2x refuses it: the model would redraw the"
        " colours hidden under them and fringe the edges. Flatten it onto a background first,"
        " in an image editor (which keeps its colour profile), or at 8 bits onto white with"
        f" ffmpeg (which drops it): ffmpeg -i {shlex.quote(name)} -vf {shlex.quote(graph)}"
        f" -frames:v 1 {shlex.quote(f'{name}.flat.png')}"
    )


def _several_frames(name: str, fmt: str, frames: int, stream: bool = False) -> Refusal:
    """The refusal of a file of several frames, with the command that makes a video of it: a
    fixed rate (VIDEO_RATE), since the delays of a GIF are uneven and the video path refuses a
    video whose timestamps contradict its rate; the rate is the user's to choose. The format
    conversion drops an alpha plane without blending it (checked with a half-transparent frame on
    ffmpeg n9.0.2: the colour comes out as stored). `stream`: JPEGs
    one after another (a raw MJPEG stream, which ffmpeg reads at MJPEG_RATE unless told)."""
    head = f"{name} holds {frames} frames, and seedvr2x reads one picture from a still."
    if fmt == "TIFF":
        return Refusal(
            f"{head} Save each page of it as a file of its own (an image editor, or libtiff's"
            " tiffsplit), and upscale each"
        )
    video = shlex.quote(f"{name}.mkv")
    if stream:
        return Refusal(
            f"{head} To upscale it as a video, make one of it first, choosing the rate"
            f" ({MJPEG_RATE} frames a second here): ffmpeg -f mjpeg -framerate {MJPEG_RATE}"
            f" -i {shlex.quote(name)} -pix_fmt rgb24 -c:v ffv1 {video}"
        )
    return Refusal(
        f"{head} To upscale it as a video, make one of it first, choosing the rate"
        f" ({VIDEO_RATE} frames a second here; its transparency, if any, is dropped, the colours"
        " under it showing):"
        f" ffmpeg -i {shlex.quote(name)} -vf fps={VIDEO_RATE},format=rgb24 -c:v ffv1 {video}"
    )


def _raw(name: str, why: str) -> Refusal:
    return Refusal(
        f"{name} is a camera's raw file, or the preview of one ({why}): seedvr2x upscales pictures,"
        " and a raw file's preview is not the picture the camera recorded, nor can its sensor"
        " samples be read as one. Develop it first in a raw developer (a program that turns a"
        " camera's raw file into a picture, choosing its white balance, tone and colour), saving"
        " a 16-bit PNG, and upscale the PNG"
    )


def _tiff16(name: str, icc: bool) -> Refusal:
    # ffmpeg 9.0 makes the PNG at 16 bits (rgb48 for colour, gray16 for grey, by the encoder's
    # own choice from the TIFF's pixel format) but carries no ICC profile into it.
    command = f"ffmpeg -i {shlex.quote(name)} {shlex.quote(f'{name}.png')}"
    way = (
        f"Make a 16-bit PNG of it in an image editor, which keeps its colour profile (ffmpeg's"
        f" command, {command}, makes a PNG at 16 bits but drops the profile)"
        if icc
        else f"Make a 16-bit PNG of it, which seedvr2x reads whole: {command}"
    )
    return Refusal(
        f"{name} is a TIFF of 16 bits per sample, which seedvr2x does not read yet: its decoder"
        f" cuts it to 8 bits. {way}"
    )

"""The PNG writer of a still picture (DESIGN.md, Still images, Output): 16-bit or 8-bit, RGB or
grey, with the source's colour chunks and ICC profile, and no orientation: the caller writes the
picture upright (picture.upright). numpy and the standard library's zlib only, no torch: no
library locked writes a 48-bit RGB PNG (Pillow cannot, and torchvision's encoder takes uint8
alone), and ffmpeg is not run for a still.

Quantised as the video writers' are (apps/seedvr2x/media/writer.py's quantise): clipped to
[0, 1], scaled by 65535 (255 at 8 bits) and rounded to nearest. The bytes are a function of the
pixels and the options: nothing of the time or the machine is in them, beyond zlib's own output
for a given level; the writer's version only in the record the caller gives (xmp, software:
record.py)."""

import struct
import zlib
from collections.abc import Iterator, Sequence

import numpy as np
import numpy.typing as npt

SIGNATURE = b"\x89PNG\r\n\x1a\n"

# The five filters of the PNG specification (section 9), and the choice of one per row.
NONE, SUB, UP, AVERAGE, PAETH, ADAPTIVE = range(6)

# The filter and the zlib level of write(): a filter per row, the one with the smallest sum of
# absolute values (ADAPTIVE), at level 6. Measured on a 32-core machine on a 4K (3,840 x 2,880)
# picture drawn from a 320 x 240 photograph, upscaled in float with a smooth resampler, once as it
# is and once with a grain of 0.7 level of 8 bits (what a model's output has in its low bits).
# Size in MB, time in s, 16 bits at level 6 (none, sub, up, average, paeth, adaptive):
#   smooth  9.39 0.35 | 8.24 0.43 | 7.82 0.40 | 8.33 0.48 | 7.49 0.69 | 7.49 0.91
#   grain  46.36 3.14 | 47.44 2.99 | 47.27 3.09 | 48.45 2.80 | 47.90 3.32 | 46.29 4.06
# and 8 bits at level 6, grain: 11.45 2.27 | 10.01 3.12 | 10.35 3.64 | 9.71 3.70 | 10.12 4.49 |
# 10.15 4.00. The grain leaves 16-bit files within 5% of each other whatever the filter, the
# adaptive choice the smallest or within 4.5% of it everywhere, at 1.3 times the time of up, which
# is little against the model's minutes per picture. Level 9 (adaptive) takes 2.3 to 7.6 times
# as long as 6 for 0.5 to 4.5% less; level 1 is 2 to 7% bigger at 16 bits (33% at 8 bits, smooth)
# and 2 to 4 times faster. An 8K picture is 4 times these figures.
FILTER = ADAPTIVE
LEVEL = 6

# Rows are filtered and compressed in bands of about this many bytes, so that the filters' int16
# temporaries stay a few times the band, never a few times the picture.
BAND_BYTES = 4 << 20

# The compressed data is cut in IDAT chunks of at most this many bytes (the specification allows
# up to 2^31 - 1).
IDAT_BYTES = 1 << 20

# The ICC profile's name in iCCP: it is only a label, which readers ignore.
ICC_NAME = b"ICC profile"

# The keyword of the iTXt chunk that holds an XMP packet (XMP Specification Part 3, 1.1.5), and of
# the tEXt chunk that names the program that wrote the file (PNG's predefined keywords, 11.3.4.2).
XMP_KEYWORD = b"XML:com.adobe.xmp"
SOFTWARE_KEYWORD = b"Software"


def write(
    rgb: npt.NDArray[np.float32],
    *,
    bits: int,
    grey: bool,
    icc: bytes | None,
    chunks: Sequence[bytes],
    xmp: bytes | None = None,
    software: str | None = None,
    filter_type: int | None = None,
    level: int | None = None,
) -> bytes:
    """The file of a picture: rgb (H, W, 3) float32, any finite values (clipped to [0, 1]), any
    memory layout, written at bits (8 or 16) per sample; grey writes colour type 0 from the mean
    of the three planes (taken in float32, before the quantisation), else colour type 2.

    Chunks, in this order: IHDR; chunks, the source's colour chunks (cICP, cHRM, gAMA, sRGB),
    each whole as the file held it (length, type, data, CRC), in the order given; iCCP holding
    icc (zlib-compressed under ICC_NAME) if icc is not None; tEXt Software holding software
    (Latin-1) if it is not None; iTXt XML:com.adobe.xmp holding the XMP packet xmp (UTF-8,
    uncompressed, no language) if it is not None; IDAT; IEND. The last two are seedvr2x's record
    (record.Record.creator and .xmp; DESIGN.md, The record in each output). No eXIf: the picture
    is written upright (picture.upright, DESIGN.md, Still images, Orientation), so no tool after
    this one has a tag to honour. All before the image data, where the browsers read them.

    filter_type is one of the module's NONE to ADAPTIVE (ADAPTIVE: per row, the filter of
    the smallest sum of absolute values, as libpng's heuristic), FILTER if None; level is zlib's,
    LEVEL if None. The choice changes the size and the time, never the decoded pixels."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or 0 in rgb.shape:
        raise ValueError(f"rgb must be (H, W, 3) and not empty, not {rgb.shape}")
    if bits not in (8, 16):
        raise ValueError(f"bits must be 8 or 16, not {bits}")
    filter_type = FILTER if filter_type is None else filter_type
    if filter_type not in range(ADAPTIVE + 1):
        raise ValueError(f"filter_type must be 0 to {ADAPTIVE}, not {filter_type}")
    height, width = rgb.shape[:2]
    channels = 1 if grey else 3
    header = struct.pack(">IIBBBBB", width, height, bits, 0 if grey else 2, 0, 0, 0)
    parts = [SIGNATURE, chunk(b"IHDR", header), *chunks]
    if icc is not None:
        parts.append(chunk(b"iCCP", ICC_NAME + b"\0\0" + zlib.compress(icc, 9)))
    if software is not None:
        # tEXt is Latin-1, with no NUL: refused here, never cut (ValueError).
        if "\0" in software:
            raise ValueError("a PNG text holds no NUL")
        parts.append(chunk(b"tEXt", SOFTWARE_KEYWORD + b"\0" + software.encode("latin-1")))
    if xmp is not None:
        # Keyword, compression flag and method (0, 0: uncompressed, as XMP's specification asks
        # for a packet that tools may find by scanning), language and translated keyword empty.
        parts.append(chunk(b"iTXt", XMP_KEYWORD + b"\0\0\0\0\0" + xmp))
    for piece in _idat(rgb, bits, channels, filter_type, LEVEL if level is None else level):
        parts.append(chunk(b"IDAT", piece))
    parts.append(chunk(b"IEND", b""))
    return b"".join(parts)


def chunk(kind: bytes, data: bytes) -> bytes:
    """A whole chunk: length, type, data, and the CRC of type and data."""
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", _crc(kind, data))


def _crc(kind: bytes, data: bytes) -> int:
    return zlib.crc32(data, zlib.crc32(kind))


def _idat(
    rgb: npt.NDArray[np.float32], bits: int, channels: int, filter_type: int, level: int
) -> Iterator[bytes]:
    """The compressed image data, in pieces of at most IDAT_BYTES."""
    height, width = rgb.shape[:2]
    bpp = channels * bits // 8
    row_bytes = width * bpp
    band = max(1, BAND_BYTES // row_bytes)
    compressor = zlib.compressobj(level)
    previous = np.zeros(row_bytes, np.uint8)
    pending = bytearray()
    for top in range(0, height, band):
        samples = quantised(rgb[top : top + band], bits, channels == 1)
        raw = samples.view(np.uint8).reshape(samples.shape[0], row_bytes)
        pending += compressor.compress(_filtered(raw, previous, bpp, filter_type).tobytes())
        previous = raw[-1]
        while len(pending) >= IDAT_BYTES:
            yield bytes(pending[:IDAT_BYTES])
            del pending[:IDAT_BYTES]
    pending += compressor.flush()
    for start in range(0, len(pending), IDAT_BYTES):
        yield bytes(pending[start : start + IDAT_BYTES])


def quantised(
    rows: npt.NDArray[np.float32], bits: int, grey: bool
) -> npt.NDArray[np.uint8] | npt.NDArray[np.uint16]:
    """(R, W, 3) float32 to the samples of the rows, in network order: (R, W, C) bytes at 8 bits,
    (R, W, C) big-endian uint16 at 16, C being 1 for grey. media/writer.py's quantise: clipped,
    scaled, rint, the mean of the planes taken before."""
    if grey:
        # Float32 throughout (the planes are float32): the three equal planes of a model's grey
        # picture give back their value to within the quantisation, never a code away.
        x = (rows[..., 0] + rows[..., 1] + rows[..., 2]) / np.float32(3)
        x = x[..., np.newaxis]
    else:
        x = rows.astype(np.float32, copy=True)
    np.clip(x, 0.0, 1.0, out=x)
    x *= 65535 if bits == 16 else 255
    np.rint(x, out=x)
    # C order whatever the input's: the rows are viewed as bytes.
    return x.astype(">u2" if bits == 16 else np.uint8, order="C")


def _filtered(
    raw: npt.NDArray[np.uint8], previous: npt.NDArray[np.uint8], bpp: int, filter_type: int
) -> npt.NDArray[np.uint8]:
    """The rows of raw (R, N) bytes with the filter type byte in front of each: (R, N + 1).
    previous is the row above the first (zeros for the picture's first). The filters predict from
    the raw bytes, as the decoder rebuilds them, so a whole band is filtered at once."""
    rows, size = raw.shape
    out = np.empty((rows, size + 1), np.uint8)
    if filter_type != ADAPTIVE:
        out[:, 0] = filter_type
        out[:, 1:] = _apply(raw, previous, bpp, filter_type)
        return out
    candidates = np.stack([_apply(raw, previous, bpp, t) for t in range(PAETH + 1)])
    # libpng's heuristic: the filter whose bytes, read as signed, have the smallest sum of
    # absolute values (they cluster round zero when the prediction is good).
    cost = np.abs(candidates.view(np.int8).astype(np.int16)).sum(axis=2, dtype=np.int64)
    best = cost.argmin(axis=0)
    out[:, 0] = best
    out[:, 1:] = candidates[best, np.arange(rows)]
    return out


def _apply(
    raw: npt.NDArray[np.uint8], previous: npt.NDArray[np.uint8], bpp: int, filter_type: int
) -> npt.NDArray[np.uint8]:
    """One filter over the rows (R, N): bytes modulo 256, the pixel to the left (a), above (b)
    and above-left (c) of PNG's specification being zero outside the picture."""
    if filter_type == NONE:
        return raw
    left = np.zeros_like(raw)
    left[:, bpp:] = raw[:, :-bpp]
    if filter_type == SUB:
        return raw - left
    above = np.concatenate([previous[np.newaxis], raw[:-1]])
    if filter_type == UP:
        return raw - above
    if filter_type == AVERAGE:
        return raw - ((left.astype(np.uint16) + above) >> 1).astype(np.uint8)
    upper_left = np.zeros_like(raw)
    upper_left[:, bpp:] = above[:, :-bpp]
    a, b, c = (x.astype(np.int16) for x in (left, above, upper_left))
    # Paeth's predictor: the neighbour nearest a + b - c, ties to a, then b.
    from_b = np.abs(a - c)  # |p - b| = |a - c|
    from_a = np.abs(b - c)  # |p - a| = |b - c|
    from_c = np.abs(a + b - c - c)  # |p - c|
    prediction = np.where(
        (from_a <= from_b) & (from_a <= from_c), a, np.where(from_b <= from_c, b, c)
    )
    return raw - prediction.astype(np.uint8)

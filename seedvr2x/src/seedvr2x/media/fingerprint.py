"""The conversion chain's fingerprint (DESIGN.md, Pause and resume): the SHA-256 of what the
decode's and the writers' conversions make of the same frames, run at startup. A resume compares
it with the job's record, as it does the rest of the environment: zimg, which does the
conversions (zscale), is a library of its own, and an ffmpeg linked to the system's libzimg takes
its upgrades without a new version string."""

import hashlib
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from seedvr2x.media.conversion import Conversion
from seedvr2x.media.decode import decode_output
from seedvr2x.media.ffmpeg import MediaError
from seedvr2x.media.writer import Tags, master_filters, png_filters

# The decode's conversions run: the most common source first, then each depth, chroma
# subsampling, matrix, range and chroma location zscale is given, at least once; RGB, expanded to
# 16 bits.
DECODES = (
    Conversion("yuv420p", "709", "limited", "left"),
    Conversion("yuv420p10le", "2020_ncl", "limited", "topleft"),
    Conversion("yuv422p10le", "170m", "limited", "left"),
    Conversion("yuv444p12le", "470bg", "full", "left"),
    Conversion("yuv420p9le", "709", "full", "center"),
    Conversion("yuv420p14le", "170m", "limited", "bottomleft"),
    Conversion("yuv422p16le", "2020_ncl", "full", "left"),
    Conversion("yuv411p", "170m", "limited", "left"),
    Conversion("yuv410p", "470bg", "limited", "top"),
    Conversion("yuv440p", "709", "limited", "bottom"),
    Conversion("yuvj420p", "470bg", "full", "center"),
    Conversion("gbrp", "gbr", "full", "left"),
    Conversion("gbrp10le", "gbr", "full", "left"),
    Conversion("gbrp12le", "gbr", "full", "left"),
    Conversion("gbrp16le", "gbr", "full", "left"),
)

# The writers' tags, which they copy, never convert.
TAGS = Tags("bt709", "bt709")

# The test frames' size: a few KiB per frame, each plane many samples across.
WIDTH, HEIGHT = 256, 64

# The chroma subsampling of a YUV format, horizontal and vertical, by the digits of its name.
SUBSAMPLING = {
    "420": (2, 2),
    "422": (2, 1),
    "444": (1, 1),
    "411": (4, 1),
    "410": (4, 4),
    "440": (1, 2),
}
PLANAR = re.compile(r"(?:yuvj?(\d{3})|gbr)p(?:(\d+)le)?")


def fingerprint() -> str:
    """The SHA-256 of the test conversions' output, in hex.

    Raises MediaError when ffmpeg fails them."""
    conversions = _conversions()
    with tempfile.TemporaryDirectory(prefix="seedvr2x-") as name:
        directory = Path(name)
        command = ["ffmpeg", "-hide_banner", "-nostdin", "-nostats", "-loglevel", "error", "-y"]
        for index, (pix_fmt, _) in enumerate(conversions):
            frame = directory / f"{index}.in"
            frame.write_bytes(_frame(pix_fmt))
            command += ["-f", "rawvideo", "-pix_fmt", pix_fmt, "-s", f"{WIDTH}x{HEIGHT}"]
            command += ["-i", str(frame)]
        for index, (_, output) in enumerate(conversions):
            command += ["-map", f"{index}:v:0", *output, str(directory / f"{index}.out")]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise MediaError(
                f"ffmpeg failed the startup check's test conversions: {result.stderr.strip()}"
            )
        digest = hashlib.sha256()
        for index in range(len(conversions)):
            digest.update(hashlib.sha256((directory / f"{index}.out").read_bytes()).digest())
    return digest.hexdigest()


def _conversions() -> list[tuple[str, list[str]]]:
    """Each test conversion: the pixel format it is fed, and its output's options, the decode's
    or a writer's own up to the encoder. The encoders are lossless and ffmpeg's own, so its
    version names them."""
    conversions = [(conversion.planar, decode_output(conversion)) for conversion in DECODES]
    for pix_fmt, matrix in (
        ("gbrp16le", ""),
        ("yuv420p10le", "bt709"),
        ("yuv420p10le", "smpte170m"),
    ):
        chain, options = master_filters(pix_fmt, matrix, TAGS)
        conversions.append(("gbrp16le", _raw(chain, options, pix_fmt)))
    chain, options = png_filters(TAGS)
    conversions.append(("rgb48be", _raw(chain, options, "rgb48be")))
    return conversions


def _raw(chain: str, options: list[str], pix_fmt: str) -> list[str]:
    return [
        "-fps_mode",
        "passthrough",
        "-vf",
        chain,
        *options,
        "-f",
        "rawvideo",
        "-pix_fmt",
        pix_fmt,
    ]


def _frame(pix_fmt: str) -> bytes:
    """A frame of pix_fmt, WIDTH x HEIGHT, drawn from SHAKE-256 seeded by the format's name: the
    same bytes on every run, each sample any value its depth allows."""
    if pix_fmt == "rgb48be":
        samples, depth = 3 * WIDTH * HEIGHT, 16
    else:
        match = PLANAR.fullmatch(pix_fmt)
        if match is None:
            raise ValueError(f"no test frame in {pix_fmt}")
        depth = int(match[2] or 8)
        if match[1] is None:
            samples = 3 * WIDTH * HEIGHT
        else:
            across, down = SUBSAMPLING[match[1]]
            samples = WIDTH * HEIGHT + 2 * (WIDTH // across) * (HEIGHT // down)
    data = hashlib.shake_256(pix_fmt.encode()).digest(samples * (1 if depth == 8 else 2))
    if depth in (8, 16):
        return data
    return (np.frombuffer(data, dtype="<u2") & ((1 << depth) - 1)).astype("<u2").tobytes()

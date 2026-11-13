"""The conversion chain's fingerprint (DESIGN.md, Pause and resume): the SHA-256 of what the
decode's and the writers' conversions make of the same frames, run at startup. A resume compares
it with the job's record, as it does the rest of the environment: zimg, which does the
conversions (zscale), is a library of its own, and an ffmpeg linked to the system's libzimg takes
its upgrades without a new version string.

Two digests, not one: the conversions every job runs (`fingerprint`), and --deband's chains
(`deband_fingerprint`), which only a job that took --deband runs and records, under a key of its
own (environment.current), and which a resume compares only between two such jobs
(resume.differences), as inductor's packages count only between two compiled jobs. One digest of
both would refuse the resume of a job that never debanded when ffmpeg's deband alone changed."""

import hashlib
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from media.conversion import Conversion
from media.decode import decode_output
from media.ffmpeg import MediaError, ends_with_us
from media.writer import Tags, master_filters, png_filters

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

# --deband's chains (Conversion.deband), their zscale bringing a limited and a full range to 16
# bits before the filter, run on a gradient frame (_gradient), not the noise of _frame: deband
# changed 0.13% of the output samples of the noise frame in a limited range and 2.2% in a full
# one, and 67% and 80% of the gradient's (ffmpeg 9.0.2, 256x64 yuv420p, the chain with deband
# against the same without it, 2026-11-11), so a change in the filter moves this digest.
DEBANDS = (
    Conversion("yuv420p", "709", "limited", "left", deband=True),
    Conversion("yuvj420p", "470bg", "full", "center", deband=True),
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
    return _digest(_conversions())


def deband_fingerprint() -> str:
    """The SHA-256 of --deband's test chains' output, in hex, on a gradient frame: the digest a
    --deband job records beside the conversions' (environment.current).

    Raises MediaError when ffmpeg fails them."""
    return _digest(
        [(conversion.planar, decode_output(conversion)) for conversion in DEBANDS], gradient=True
    )


def _digest(conversions: list[tuple[str, list[str]]], gradient: bool = False) -> str:
    """The SHA-256 of what ffmpeg makes of the test frames of `conversions`: those of _frame, or
    of _gradient."""
    with tempfile.TemporaryDirectory(prefix="seedvr2x-") as name:
        directory = Path(name)
        command = ["ffmpeg", "-hide_banner", "-nostdin", "-nostats", "-loglevel", "error", "-y"]
        for index, (pix_fmt, _) in enumerate(conversions):
            frame = directory / f"{index}.in"
            frame.write_bytes(_gradient(pix_fmt) if gradient else _frame(pix_fmt))
            command += ["-f", "rawvideo", "-pix_fmt", pix_fmt, "-s", f"{WIDTH}x{HEIGHT}"]
            command += ["-i", str(frame)]
        for index, (_, output) in enumerate(conversions):
            command += ["-map", f"{index}:v:0", *output, str(directory / f"{index}.out")]
        result = subprocess.run(
            command, capture_output=True, text=True, check=False, preexec_fn=ends_with_us()
        )
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


def _gradient(pix_fmt: str) -> bytes:
    """A yuv420p or yuvj420p frame, WIDTH x HEIGHT, of smooth ramps across, the kind of frame
    banding lives on: luma from 40 to 200 over the width, the chroma planes 100 to 140 and 140 to
    100. Deterministic, 24 KiB. Deband changed 67% (limited) and 80% (full) of the samples
    ffmpeg made of it, against 0.13% and 2.2% of _frame's noise (DEBANDS)."""
    if pix_fmt not in ("yuv420p", "yuvj420p"):
        raise ValueError(f"no gradient frame in {pix_fmt}")

    def ramp(width: int, height: int, start: int, end: int) -> bytes:
        across = start + (end - start) * np.arange(width) // (width - 1)
        return np.tile(across.astype(np.uint8), (height, 1)).tobytes()

    return (
        ramp(WIDTH, HEIGHT, 40, 200)
        + ramp(WIDTH // 2, HEIGHT // 2, 100, 140)
        + ramp(WIDTH // 2, HEIGHT // 2, 140, 100)
    )

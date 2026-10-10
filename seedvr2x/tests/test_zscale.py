"""Every zscale on one slice (DESIGN.md, Input): the decode's and the yuv420p10le writer's
conversions give the same bytes whatever the filter threads ffmpeg runs them with, and the chains
seedvr2x builds hold zscale on one slice only. The conversions skip without a build seedvr2x
accepts."""

import hashlib
import inspect
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from seedvr2x.media import ffmpeg, fingerprint
from seedvr2x.media.conversion import CHROMA_LOCATIONS, MATRICES, PIXEL_FORMATS, Conversion
from seedvr2x.media.decode import decode_output
from seedvr2x.media.ffmpeg import MediaError
from seedvr2x.media.reader import read_command
from seedvr2x.media.writer import Tags, master_filters, png_filters, yuv_matrix


def _usable() -> bool:
    try:
        ffmpeg.check()
    except MediaError:
        return False
    return True


needs_ffmpeg = pytest.mark.skipif(
    not _usable(), reason="needs ffmpeg 7.1 or later with zscale, scdet and ffv1"
)

SRC = Path(__file__).resolve().parents[1] / "src" / "seedvr2x"

# zscale makes at most one slice per 64 rows (MIN_TILESIZE, libavfilter/vf_zscale.c): 1080 rows
# allow 16, where a 64-row frame is one slice whatever the threads, and shows nothing.
WIDTH, HEIGHT = 1920, 1080
SLICES = 16
ONE_SLICE = "zscale=threads=1:"
# A zscale written otherwise: bare, named (zscale@name=), or with another option first.
NOT_ONE_SLICE = re.compile(r"zscale(?!=threads=1:)")


def _samples(name: str, count: int, depth: int) -> bytes:
    """count little-endian 16-bit samples of depth bits, drawn from SHAKE-256 seeded by name (as
    media/fingerprint.py's frames): the same on every run, the chroma varying everywhere."""
    data = hashlib.shake_256(name.encode()).digest(2 * count)
    return (np.frombuffer(data, dtype="<u2") & ((1 << depth) - 1)).astype("<u2").tobytes()


def _converted(frame: bytes, pix_fmt: str, output: list[str], threads: str, count: int) -> str:
    """The SHA-256 of ffmpeg's output for a raw WIDTH x HEIGHT frame of pix_fmt, its filter graph
    on count threads: `threads` is the graph's option, -filter_threads for -vf's and
    -filter_complex_threads for -filter_complex's (ffmpeg n9.0.2: global options, each reaching
    its own kind of graph only)."""
    result = subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-nostdin"),
            *("-f", "rawvideo", "-pix_fmt", pix_fmt, "-s", f"{WIDTH}x{HEIGHT}", "-i", "-"),
            *(threads, str(count), *output, "-"),
        ],
        input=frame,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode()
    return hashlib.sha256(result.stdout).hexdigest()


def _on_one_slice(frame: bytes, pix_fmt: str, output: list[str], threads: str) -> None:
    """The output's bytes are the same on 1 and SLICES filter threads, and are those of the chain
    without threads=1 on one thread, which on SLICES threads gives others: the slicing that
    threads=1 removes, a control without which the first comparison would prove nothing."""
    ours = [_converted(frame, pix_fmt, output, threads, count) for count in (1, SLICES)]
    assert ours[0] == ours[1], f"other bytes on {SLICES} filter threads than on 1"
    sliced = [arg.replace(ONE_SLICE, "zscale=") for arg in output]
    assert sliced != output, "no zscale on one slice in the chain"
    default = [_converted(frame, pix_fmt, sliced, threads, count) for count in (1, SLICES)]
    assert default[0] == ours[0], "threads=1 changed more than the slicing"
    assert default[1] != default[0], (
        f"zscale without threads=1 gave the one-slice bytes on {SLICES} filter threads: this"
        " ffmpeg no longer slices it differently (it did, n9.0.2), and this test no longer shows"
        " what threads=1 prevents"
    )


@needs_ffmpeg
def test_decode_on_one_slice() -> None:
    # A 10-bit 4:2:0 source reads off the exact conversion from 4 slices on, nearly every sample;
    # an 8-bit one reads exactly at any count, and would show nothing (research/docs/numerics.md,
    # The master's chroma: 4:2:0 kernels and zscale's slices).
    frame = _samples("yuv420p10le", WIDTH * HEIGHT * 3 // 2, 10)
    output = decode_output(Conversion("yuv420p10le", "709", "limited", "left"))
    _on_one_slice(frame, "yuv420p10le", output, "-filter_threads")


@needs_ffmpeg
def test_reads_on_one_slice() -> None:
    # A source's reads run the decode's conversion after a select, in the -vf graph of an output
    # of its own (media/reader.py), its slices following -filter_threads: the frames as a read
    # gives them.
    frame = _samples("yuv420p10le", WIDTH * HEIGHT * 3 // 2, 10)
    conversion = Conversion("yuv420p10le", "709", "limited", "left")
    command = read_command(Path("in.mkv"), conversion, None, 0, 3)
    output = command[command.index("-map") : command.index("pipe:1")]
    assert output[output.index("-vf") + 1].startswith("select=gte(pts\\,0),")
    _on_one_slice(frame, "yuv420p10le", output, "-filter_threads")


@needs_ffmpeg
def test_writer_on_one_slice() -> None:
    # The yuv420p10le master changes with the slice count: up to 1.7% of its chroma samples
    # (numerics.md, as above). In a -filter_complex graph, as FFV1Writer runs it.
    frame = _samples("gbrp16le", WIDTH * HEIGHT * 3, 16)
    chain, options = master_filters("yuv420p10le", "bt709", Tags())
    output = [
        *("-filter_complex", f"[0:v]{chain}[master]", "-map", "[master]"),
        *("-fps_mode", "passthrough", *options, "-f", "rawvideo", "-pix_fmt", "yuv420p10le"),
    ]
    _on_one_slice(frame, "gbrp16le", output, "-filter_complex_threads")


# The two checks below, together: the only "zscale=" written in seedvr2x's own code (the vendored
# model code aside, which has none) is ffmpeg.zscale's; and in every chain the sweep builds as
# seedvr2x does, every zscale, however spelt, is "zscale=threads=1:", and each decode chain and
# yuv420p10le master chain has one. A chain the sweep doesn't build is held by the first check
# alone, which a bare zscale or a zscale@name= gets past.
def test_every_zscale_from_the_helper() -> None:
    found: dict[str, int] = {}
    for path in sorted(SRC.rglob("*.py")):
        relative = path.relative_to(SRC)
        if relative.parts[0] != "vendor" and (count := path.read_text().count("zscale=")):
            found[relative.as_posix()] = count
    assert found == {"media/ffmpeg.py": 1}
    assert "zscale=" in inspect.getsource(ffmpeg.zscale)
    assert ffmpeg.zscale("m=709", "d=none") == "zscale=threads=1:m=709:d=none"


def _text(chain: str, options: list[str]) -> str:
    return " ".join([chain, *options])


def test_every_chain_on_one_slice() -> None:
    # The decode's chains: every planar format zscale is fed, with each matrix ("gbr" for RGB),
    # both ranges and every chroma location (RGB's always left, as conversion_for gives it).
    # runtime/shot.py's COPY_READ (gbrp16le, gbr, full, left) is one of them, not imported:
    # shot.py imports torch.
    decodes = [
        Conversion(planar, matrix, color_range, location).filters()
        for planar in sorted({pixel_format.planar for pixel_format in PIXEL_FORMATS.values()})
        for matrix in (["gbr"] if PIXEL_FORMATS[planar].family == "rgb" else MATRICES.values())
        for color_range in ("limited", "full")
        for location in (["left"] if matrix == "gbr" else sorted(CHROMA_LOCATIONS))
    ]
    # The FFV1 masters' chains and options with each matrix yuv_matrix gives (at HD sizes and
    # below, for every matrix the decode reads), PNG's, and the startup fingerprint's.
    matrices = {
        yuv_matrix(width, height, source)
        for width, height in ((1920, 1080), (720, 576))
        for source in ("", *MATRICES)
    }
    tags = Tags("bt709", "bt709")
    yuv = [_text(*master_filters("yuv420p10le", matrix, tags)) for matrix in sorted(matrices)]
    rgb = [_text(*master_filters("gbrp16le", matrix, tags)) for matrix in sorted(matrices)]
    tests = [" ".join(output) for _, output in fingerprint._conversions()]
    for text in (*decodes, *yuv, *rgb, _text(*png_filters(tags)), *tests):
        assert NOT_ONE_SLICE.search(text) is None, text
    for text in (*decodes, *yuv):
        assert ONE_SLICE in text, text
    assert any(ONE_SLICE in text for text in tests)

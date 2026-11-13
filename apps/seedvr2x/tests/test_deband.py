"""--deband's filter in the reads' conversion (media/conversion.py; DESIGN.md, Input, "Banded
sources"): ffmpeg's deband at 16 bits before the conversion to RGB, every frame a job reads going
through it, the first pass's not; a conversion without it the chain of before, byte for byte. The
reads skip without a build seedvr2x accepts, the inter-coded one without libx264."""

import hashlib
import re
import subprocess
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from test_conversion import stream
from test_probe import has_encoder

from media import ffmpeg
from media.conversion import DEBAND, DEEP, PIXEL_FORMATS, Conversion, conversion_for
from media.ffmpeg import MediaError
from media.reader import read_command
from media.source import Source, declare, examine, first_pass


def _usable() -> bool:
    try:
        ffmpeg.check()
    except MediaError:
        return False
    return True


needs_ffmpeg = pytest.mark.skipif(
    not _usable(), reason="needs ffmpeg 7.1 or later with zscale, idet and ffv1"
)

# The chains of a conversion without --deband as main wrote them before the option (1491099):
# the decode's, and a read's from a frame on, its select before it.
PLAIN = (
    "format=yuv420p,zscale=threads=1:min=709:rin=limited:cin=left:pin=unspecified:p=unspecified"
    ":tin=unspecified:t=unspecified:m=gbr:r=full:d=none:f=bilinear,format=gbrp16le"
)
PLAIN_RGB = (
    "format=gbrp,zscale=threads=1:min=gbr:rin=full:cin=left:pin=unspecified:p=unspecified"
    ":tin=unspecified:t=unspecified:m=gbr:r=full:d=none:f=bilinear,format=gbrp16le"
)
# With it, on the same source: brought to 16 bits by zscale, matrix, range and siting kept, then
# deband, then the conversion of before built from the 16-bit format.
DEBANDED = (
    "format=yuv420p,zscale=threads=1:min=709:m=709:rin=limited:r=limited:cin=left:c=left"
    ":pin=unspecified:p=unspecified:tin=unspecified:t=unspecified:d=none:f=bilinear"
    ",format=yuv420p16le,deband=1thr=0.012:2thr=0.012:3thr=0.012:4thr=0.012:blur=0,"
    "format=yuv420p16le,zscale=threads=1:min=709:rin=limited:cin=left:pin=unspecified"
    ":p=unspecified:tin=unspecified:t=unspecified:m=gbr:r=full:d=none:f=bilinear,format=gbrp16le"
)

# A banded clip: a slow gradient of 32 codes across 256 pixels, an 8-bit step every 8 or so, the
# kind of flat steps deband takes out; tagged so that every read takes its tags, none guessed.
WIDTH, HEIGHT, FRAMES = 256, 144, 48
TAGS = "range=tv:colorspace=bt709:color_primaries=bt709:color_trc=bt709:chroma_location=left"
BANDED = (
    f"gradients=s={WIDTH}x{HEIGHT}:c0=0x302818:c1=0x584838:speed=0.02:r=25:d={FRAMES / 25},"
    f"format=yuv420p,setparams={TAGS}"
)


def run(*arguments: str, cwd: Path | None = None) -> bytes:
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-y", *arguments], capture_output=True, cwd=cwd
    )
    assert result.returncode == 0, result.stderr.decode()
    return result.stdout


def md5s(source: Source, first: int = 0) -> list[str]:
    """The MD5 of each frame a read of source gives from frame `first`, its 16-bit RGB: the
    reader's output's framemd5."""
    with source.reader(first) as reader:
        frames = reader.read(source.frames - first)
    return [hashlib.md5(frame.tobytes()).hexdigest() for frame in frames]


def test_without_deband_the_chain_is_the_one_of_before() -> None:
    plain = Conversion("yuv420p", "709", "limited", "left")
    assert not plain.deband
    assert plain.filters() == PLAIN
    assert Conversion("gbrp", "gbr", "full", "left").filters() == PLAIN_RGB
    command = read_command(Path("in.mkv"), plain, "1.000000", 25, "pipe:3")
    assert command[command.index("-vf") + 1] == f"select=gte(pts\\,25),{PLAIN}"
    assert not conversion_for(stream(pix_fmt="yuv420p")).deband


def test_with_deband_the_frames_are_debanded_at_16_bits_before_rgb() -> None:
    assert DEBAND == "deband=1thr=0.012:2thr=0.012:3thr=0.012:4thr=0.012:blur=0"
    debanded = Conversion("yuv420p", "709", "limited", "left", deband=True)
    assert debanded.filters() == DEBANDED
    command = read_command(Path("in.mkv"), debanded, None, None, "pipe:3")
    assert command[command.index("-vf") + 1] == DEBANDED
    taken = conversion_for(stream(pix_fmt="yuv420p", color_space="bt709"), deband=True)
    assert taken.deband and taken.filters() == DEBANDED


def test_each_format_debands_in_its_16_bit_twin() -> None:
    # Every planar format a source is read in but three has a 16-bit format of its subsampling,
    # deband's input; the JPEG family's YUV, read at full range, plain YUV's 16 bits.
    planars = {pixel_format.planar for pixel_format in PIXEL_FORMATS.values()}
    assert planars - set(DEEP) == {"yuv411p", "yuv410p", "yuv440p"}
    for planar, deep in DEEP.items():
        chroma = re.fullmatch(r"(?:yuvj?(\d{3})|gbr)p(?:\d+le)?", planar)
        assert chroma is not None, planar
        assert deep == ("gbrp16le" if chroma[1] is None else f"yuv{chroma[1]}p16le"), planar
        chain = Conversion(planar, "709", "limited", "left", deband=True).filters()
        assert chain.startswith(f"format={planar},zscale=threads=1:")
        assert f",format={deep},{DEBAND},format={deep},zscale=threads=1:" in chain


@pytest.mark.parametrize("pix_fmt", ["yuv411p", "yuv410p", "yuv440p"])
def test_a_format_with_no_16_bit_twin_is_refused_with_deband(pix_fmt: str) -> None:
    assert not conversion_for(stream(pix_fmt=pix_fmt)).deband
    with pytest.raises(
        MediaError, match=f"^--deband: pixel format {pix_fmt}: ffmpeg has no 16-bit"
    ):
        conversion_for(stream(pix_fmt=pix_fmt), deband=True)


def _expanded(pix_fmt: str, conversion: Conversion, data: bytes) -> npt.NDArray[np.uint16]:
    """The 16-bit samples of a raw frame of pix_fmt, WIDTH x HEIGHT, brought to 16 bits as the
    deband chain brings them: its head, up to the format deband takes."""
    chain = conversion.filters()
    head = chain[: chain.index(f",{DEBAND}")]
    out = subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-nostdin", "-f", "rawvideo", "-pix_fmt", pix_fmt),
            *("-s", f"{WIDTH}x{HEIGHT}", "-i", "-", "-vf", head, "-f", "rawvideo", "-"),
        ],
        input=data,
        capture_output=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr.decode()
    return np.frombuffer(out.stdout, dtype="<u2")


@needs_ffmpeg
def test_the_16_bits_are_exact() -> None:
    # zscale brings a limited range to 16 bits by a shift (swscale's own expansion too: below),
    # a full range by v * 65535 / 255 exactly: an 8-bit RGB source's every sample, an 8-bit
    # JPEG-family source's luma, its chroma centred.
    samples = WIDTH * HEIGHT * 3 // 2
    rng = np.random.default_rng(7)
    eight = rng.integers(0, 256, samples, dtype=np.uint8)
    limited = Conversion("yuv420p", "709", "limited", "left", deband=True)
    assert np.array_equal(_expanded("yuv420p", limited, eight.tobytes()), eight.astype("<u2") << 8)
    ten = rng.integers(0, 1024, samples).astype("<u2")
    limited10 = Conversion("yuv420p10le", "2020_ncl", "limited", "topleft", deband=True)
    assert np.array_equal(_expanded("yuv420p10le", limited10, ten.tobytes()), ten << 6)
    rgb = rng.integers(0, 256, WIDTH * HEIGHT * 3, dtype=np.uint8)
    full = Conversion("gbrp", "gbr", "full", "left", deband=True)
    assert np.array_equal(_expanded("gbrp", full, rgb.tobytes()), rgb.astype("<u2") * 257)
    jpeg = Conversion("yuvj420p", "470bg", "full", "center", deband=True)
    expanded = _expanded("yuvj420p", jpeg, eight.tobytes()).astype(np.int64)
    luma, chroma = WIDTH * HEIGHT, eight.astype(np.int64)
    assert np.array_equal(expanded[:luma], chroma[:luma] * 257)
    # A full range's chroma centred, 128 at 32768: v * 257 - 128, 0 kept at 0 (the reviewer's
    # reading of zimg, 2026-11-11, held here).
    assert np.array_equal(expanded[luma:], np.maximum(chroma[luma:] * 257 - 128, 0))


@pytest.fixture(scope="module")
def banded(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The banded clip as an FFV1 file, a keyframe every 8 frames."""
    path = tmp_path_factory.mktemp("banded") / "banded.mkv"
    run("-f", "lavfi", "-i", BANDED, "-c:v", "ffv1", "-g", "8", str(path))
    return path


def _fields(conversion: Conversion) -> tuple[str, str, str, str]:
    return conversion.planar, conversion.matrix, conversion.color_range, conversion.chroma_location


@needs_ffmpeg
def test_a_debanded_read_is_the_source_debanded_by_ffmpeg_and_read_plain(
    banded: Path, tmp_path: Path
) -> None:
    # The reference is made by ffmpeg alone, as the study made its inputs (impl-box-1): the
    # source brought to 16 bits by the format filter (swscale's shift, a limited range's exact
    # expansion: test_the_16_bits_are_exact), deband on one thread of its own, into a 16-bit FFV1
    # file, read without --deband. The read with it gives the same frames, frame for frame, its
    # deband on ffmpeg's threads.
    prefiltered = tmp_path / "prefiltered.mkv"
    run(
        *("-i", str(banded), "-vf", f"format=yuv420p16le,{DEBAND}:threads=1,setparams={TAGS}"),
        *("-c:v", "ffv1", str(prefiltered)),
    )
    reference = examine(prefiltered)
    assert reference.stream.pix_fmt == "yuv420p16le" and not reference.conversion.deband
    debanded = examine(banded, deband=True)
    # The tags carried through, so the two reads convert alike: the comparison is the deband's.
    assert _fields(reference.conversion)[1:] == _fields(debanded.conversion)[1:]
    assert reference.conversion.guessed == debanded.conversion.guessed == ()
    assert md5s(debanded) == md5s(reference)
    # A control: deband changes this clip's frames, every one of them.
    plain = md5s(examine(banded))
    assert all(one != other for one, other in zip(md5s(debanded), plain, strict=True))


class Recording:
    """A shot detector that keeps the frames the first pass gives it."""

    def __init__(self) -> None:
        self.frames: list[npt.NDArray[np.uint8]] = []

    def push(self, frames: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        self.frames.append(frames.copy())
        return np.zeros(len(frames), dtype=np.float32)

    def finish(self) -> npt.NDArray[np.float32]:
        return np.zeros(0, dtype=np.float32)


@needs_ffmpeg
def test_the_first_pass_reads_the_source_as_it_is(banded: Path) -> None:
    # The shot detector's frames and the frame index are the source's as decoded, with --deband
    # or without: the cuts and every read's checks stay the source's. The first pass takes no
    # conversion today (media/scan.py), so this holds the contract against a change that would
    # give it one, not the reads.
    found = []
    for deband in (False, True):
        detector = Recording()
        scanned = first_pass(declare(banded, deband=deband), detector=detector)
        assert scanned.index is not None
        index = scanned.index
        found.append((index.pts, index.crc32, index.keyframes, np.concatenate(detector.frames)))
    assert len(found[0][3]) == FRAMES
    for plain, debanded in zip(*found, strict=True):
        assert np.array_equal(plain, debanded)


@needs_ffmpeg
@pytest.mark.skipif(not has_encoder("libx264"), reason="needs ffmpeg with libx264")
def test_a_shot_read_by_seek_is_the_frames_of_a_whole_read(tmp_path: Path) -> None:
    # A shot's first frame inside a GOP of an inter-coded source, read by the reader's seek from
    # the keyframe before its own, gives with --deband the frames a read from the start gives:
    # deband keeps nothing from one frame to the next (impl-box-1 on five H.264 sources).
    path = tmp_path / "banded.mkv"
    run(
        *("-f", "lavfi", "-i", BANDED, "-c:v", "libx264", "-preset", "veryfast", "-crf", "18"),
        *("-x264-params", "keyint=12:min-keyint=12:bframes=3:b-pyramid=normal:log-level=error"),
        str(path),
    )
    source = examine(path, deband=True)
    assert source.frames == FRAMES and source.conversion.deband
    first = 30  # inside the GOP of frame 24's keyframe
    assert source.index is not None and source.index.pts[first] not in source.index.keyframes
    whole = md5s(source)
    with source.reader(first) as reader:
        frames = reader.read(FRAMES - first)
        attempts = reader.attempts
    assert attempts[0].keyframe is not None, "a read from the start, no seek"
    assert [hashlib.md5(frame.tobytes()).hexdigest() for frame in frames] == whole[first:]
    plain = md5s(examine(path))
    assert sum(one != other for one, other in zip(whole, plain, strict=True)) == FRAMES

"""The library's conversion of a source's planes to RGB in torch (seedvr2x/yuv.py, `reading` and
`to_rgb`) held to the command line's own frames: its reader through zscale (media/reader.py,
media/conversion.py), on files made here for every planar format, depth, range, matrix and chroma
siting the command line reads, tagged and untagged. The planes the library is given are those
ffmpeg decodes from the file, in its planar format, and its reading is made from the tags the
command line's probe reads there: the same as the command line's conversion, field for field, and
the frames within one 16-bit code on every sample, the share one code off said in a failure's
message and over the whole grid by `test_the_grid_s_figures`, which runs last.

The files: FFV1 in Matroska holding the planes as they are; NUT where a stream must be untagged
(Matroska writes a limited range for an untagged one); x264 and x265 in MP4 and Matroska for a
4:2:0 siting at the bottom, which Matroska's fields can't say; MJPEG for the JPEG family, which
FFV1 stores as YUV of full range. A siting at the bottom of 4:2:2, 4:1:1, 4:1:0 and 4:4:0, which
no codec here carries (H.264's and HEVC's siting is 4:2:0's), is held through the command line's
decode of the raw planes, its conversion made by conversion_for from a stream so tagged. Then the
planes of real codecs as PyAV decodes them (the nodes' way), and the master's planes
(`master_planes`) read back by the command line. Skipped without an ffmpeg seedvr2x accepts; PyAV's
cases without PyAV, saying so.

Measured on 2026-11-13 (ffmpeg n9.0.2's zscale; torch 2.14.1 on the CPU): over the grid's 115
cases, no sample more than one code off, 0.14% of the samples one off, 0.28% at most in a case."""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import torch
from test_conversion import stream

from media import ffmpeg
from media.conversion import Conversion, conversion_for
from media.decode import decode_output
from media.ffmpeg import MediaError
from media.source import examine
from seedvr2x import yuv

# PyAV is never imported in this process: with it imported, the command line's first pass hung in
# every run of this file under pytest (6 of 6, 2026-11-13; 6 of 6 passed without it), a thread it
# starts never running and a child it forks never reaching exec (media/scan.py, the packets'
# thread and its ffprobe); the cause not isolated (a script reading one file a hundred times
# with PyAV imported did not hang). The nodes' process, which imports it, runs no such reader. Its
# decode runs in a process of its own (PYAV_PLANES). A dev dependency of the CLI's environment
# alone (AGENTS.md, Environment).
HAS_AV = importlib.util.find_spec("av") is not None


def _usable() -> bool:
    try:
        ffmpeg.check()
    except MediaError:
        return False
    return True


def _encoder(name: str) -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    listing = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True
    ).stdout
    return any(line.split()[1:2] == [name] for line in listing.splitlines() if line.strip())


pytestmark = pytest.mark.skipif(
    not _usable(), reason="needs ffmpeg 7.1 or later with zscale, idet and ffv1"
)
needs_av = pytest.mark.skipif(
    not HAS_AV, reason="needs PyAV (av), a dev dependency of the CLI's environment"
)

FRAMES = 2
WIDTH, HEIGHT = 64, 16  # multiples of 4: 4:1:1 and 4:1:0 included
Planes = tuple[npt.NDArray[np.uint16], npt.NDArray[np.uint16], npt.NDArray[np.uint16]]

# Every case compared, for test_the_grid_s_figures: samples, samples one code off.
FIGURES: dict[str, tuple[int, int]] = {}

PLANAR = sorted({pixel_format.planar for pixel_format in yuv.PIXEL_FORMATS.values()})
SITINGS = sorted(yuv.CHROMA_LOCATIONS)
# The sitings Matroska's fields say (ChromaSitingHorz and ChromaSitingVert: left or half, top or
# half); "bottom" and "bottomleft" are written as unspecified.
IN_MATROSKA = ["center", "left", "top", "topleft"]
SETPARAMS = {"color_space": "colorspace", "color_range": "range"}
ENCODER = {
    "color_space": "-colorspace",
    "color_range": "-color_range",
    "chroma_location": "-chroma_sample_location",
}


def planes_of(planar: str, width: int, height: int, seed: int) -> Planes:
    """The planes of FRAMES frames of `planar` (its own order: Y, Cb, Cr, or G, B, R), (T, h, w)
    codes: the left half over the whole depth (out of gamut, clamped), the right half a picture's
    values (luma over its whole span, chroma within an eighth of its own around neutral)."""
    pixel_format = yuv.PIXEL_FORMATS[planar]
    rng = np.random.default_rng(seed)
    depth = pixel_format.depth
    rows, columns = pixel_format.subsampling
    small = (FRAMES, height // rows, width // columns)
    first = rng.integers(0, 1 << depth, (FRAMES, height, width)).astype(np.uint16)
    second = rng.integers(0, 1 << depth, small).astype(np.uint16)
    third = rng.integers(0, 1 << depth, small).astype(np.uint16)
    if pixel_format.family != "rgb":
        shift = 1 << (depth - 8)
        half, chroma_half = width // 2, small[2] // 2
        first[..., half:] = rng.integers(16 * shift, 236 * shift, (FRAMES, height, width - half))
        for plane in (second, third):
            plane[..., chroma_half:] = rng.integers(
                100 * shift, 156 * shift, (*small[:2], small[2] - chroma_half)
            )
    return first, second, third


def raw(planes: Planes, planar: str) -> bytes:
    word = np.dtype(np.uint8) if yuv.PIXEL_FORMATS[planar].depth == 8 else np.dtype("<u2")
    return b"".join(
        plane[index].astype(word).tobytes() for index in range(FRAMES) for plane in planes
    )


def written(
    path: Path, planes: Planes, planar: str, tags: dict[str, str], codec: list[str]
) -> Path:
    """A file of `planes` encoded by `codec`, tagged `tags` (ffprobe's names); the frames carry
    the tags too, else ffmpeg converts them to the encoder's."""
    _, height, width = planes[0].shape
    setparams = ":".join(f"{SETPARAMS.get(name, name)}={value}" for name, value in tags.items())
    command = [
        *("ffmpeg", "-v", "error", "-nostdin", "-y"),
        *("-f", "rawvideo", "-pix_fmt", planar, "-s", f"{width}x{height}", "-framerate", "25"),
        *("-i", "-"),
        *(["-vf", f"setparams={setparams}"] if setparams else []),
        *codec,
        *(argument for name, value in tags.items() for argument in (ENCODER[name], value)),
        str(path),
    ]
    subprocess.run(command, input=raw(planes, planar), check=True)
    return path


def decoded(path: Path, planar: str, width: int, height: int) -> Planes:
    """The planes ffmpeg decodes from the file, in `planar` (the stream's own format, or its
    planar twin, which swscale only repacks), (T, h, w) codes."""
    pixel_format = yuv.PIXEL_FORMATS[planar]
    word = np.dtype(np.uint8) if pixel_format.depth == 8 else np.dtype("<u2")
    data = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", planar, "-"],
        capture_output=True,
        check=True,
    ).stdout
    rows, columns = pixel_format.subsampling
    shapes = [(height, width), *[(height // rows, width // columns)] * 2]
    frame = sum(h * w for h, w in shapes)
    values = np.frombuffer(data, dtype=word).reshape(-1, frame)
    split: list[npt.NDArray[np.uint16]] = []
    start = 0
    for h, w in shapes:
        split.append(values[:, start : start + h * w].reshape(-1, h, w).astype(np.uint16))
        start += h * w
    return split[0], split[1], split[2]


def the_library_s(planes: Planes, read: yuv.Reading) -> npt.NDArray[np.int64]:
    tensors = [torch.from_numpy(plane.astype(np.int32)) for plane in planes]
    return torch.round(yuv.to_rgb(tensors, read) * 65535).to(torch.int64).numpy()


def compared(name: str, library: npt.NDArray[np.int64], command_line: npt.NDArray[np.uint16]):
    difference = np.abs(library - command_line.astype(np.int64))
    one = int((difference == 1).sum())
    FIGURES[name] = (difference.size, one)
    assert difference.max() <= 1, (
        f"{name}: {int((difference > 1).sum())} samples more than one code off, at most"
        f" {difference.max()}; {one / difference.size:.4%} one off"
    )


def fields(conversion: Conversion | yuv.Reading) -> tuple[object, ...]:
    """What a conversion does: its planes' depth and subsampling, its matrix, range and siting,
    and what it guessed (the JPEG family's format names the full range FFV1 stores as YUV's)."""
    pixel_format = yuv.PIXEL_FORMATS[conversion.planar]
    return (
        pixel_format.depth,
        pixel_format.subsampling,
        conversion.matrix,
        conversion.color_range,
        conversion.chroma_location,
        conversion.guessed,
        conversion.matrix_guessed,
    )


def held(path: Path, meant: yuv.Reading) -> None:
    """The file read by the command line and by the library, the library's reading made from the
    tags the command line's probe reads, and `meant`, the reading of the tags written: the tags
    arrived, the two conversions are one, and the frames are within one code."""
    source = examine(path)
    found = source.stream
    with source.decoder() as reader:
        frames = reader.read(source.frames)
    read = yuv.reading(
        found.pix_fmt,
        found.width,
        found.height,
        found.color_space,
        found.color_range,
        found.chroma_location,
    )
    assert (read.planar, *fields(read)[2:]) == (
        source.conversion.planar,
        *fields(source.conversion)[2:],
    )
    assert fields(read) == fields(meant)
    planes = decoded(path, read.planar, found.width, found.height)
    compared(f"{path.name}: {meant}", the_library_s(planes, read), frames)


def ffv1(tmp_path: Path, planar: str, tags: dict[str, str], seed: int, size=(WIDTH, HEIGHT)):
    """An FFV1 file holding the planes as they are, in Matroska where it is tagged and in NUT,
    which says no tag, where nothing is."""
    width, height = size
    planes = planes_of(planar, width, height, seed)
    path = tmp_path / ("source.mkv" if tags else "source.nut")
    written(path, planes, planar, tags, ["-c:v", "ffv1"])
    assert decoded(path, planar, width, height)[0].tobytes() == planes[0].tobytes()
    return path


def _format_cases() -> list[tuple[str, dict[str, str]]]:
    # Every planar format of the YUV and RGB families in each range it takes, the matrix and the
    # siting turning over the formats so that each value meets several depths.
    cases: list[tuple[str, dict[str, str]]] = []
    matrices = sorted(yuv.MATRICES)
    for index, planar in enumerate(p for p in PLANAR if yuv.PIXEL_FORMATS[p].family != "jpeg"):
        if yuv.PIXEL_FORMATS[planar].family == "rgb":
            cases += [(planar, {"color_range": "pc"}), (planar, {})]
            continue
        cases += [
            (
                planar,
                {
                    "color_space": matrices[index % len(matrices)],
                    "color_range": color_range,
                    "chroma_location": IN_MATROSKA[index % len(IN_MATROSKA)],
                },
            )
            for color_range in ("tv", "pc")
        ]
    return cases


@pytest.mark.parametrize(("planar", "tags"), _format_cases())
def test_every_format_in_each_range(tmp_path: Path, planar: str, tags: dict[str, str]) -> None:
    path = ffv1(tmp_path, planar, tags, seed=len(planar) + len(tags))
    held(path, yuv.reading(planar, WIDTH, HEIGHT, **tags))


@pytest.mark.parametrize("siting", IN_MATROSKA)
@pytest.mark.parametrize(
    "planar", ["yuv420p", "yuv422p", "yuv411p", "yuv410p", "yuv440p", "yuv420p10le"]
)
def test_every_siting_matroska_says(tmp_path: Path, planar: str, siting: str) -> None:
    tags = {"color_space": "bt709", "color_range": "tv", "chroma_location": siting}
    path = ffv1(tmp_path, planar, tags, seed=SITINGS.index(siting))
    held(path, yuv.reading(planar, WIDTH, HEIGHT, **tags))


@pytest.mark.parametrize("siting", ["bottom", "bottomleft"])
@pytest.mark.parametrize(
    ("planar", "codec", "suffix"),
    [
        ("yuv420p", "libx264", "mp4"),
        ("yuv420p10le", "libx264", "mp4"),
        ("yuv420p12le", "libx265", "mkv"),
    ],
)
def test_a_siting_at_the_bottom(
    tmp_path: Path, planar: str, codec: str, suffix: str, siting: str
) -> None:
    if not _encoder(codec):
        pytest.skip(f"needs ffmpeg's {codec}")
    tags = {"color_space": "bt709", "color_range": "tv", "chroma_location": siting}
    planes = planes_of(planar, WIDTH, HEIGHT, seed=7)
    options = ["-x265-params", "log-level=error"] if codec == "libx265" else []
    path = written(tmp_path / f"source.{suffix}", planes, planar, tags, ["-c:v", codec, *options])
    held(path, yuv.reading(planar, WIDTH, HEIGHT, **tags))


@pytest.mark.parametrize("siting", ["bottom", "bottomleft"])
@pytest.mark.parametrize("planar", ["yuv422p", "yuv422p10le", "yuv411p", "yuv410p", "yuv440p"])
def test_a_siting_at_the_bottom_no_codec_carries(planar: str, siting: str) -> None:
    # The command line's decode (media/decode.py's output options, its Decoder's) of the raw
    # planes, its conversion that of a stream so tagged.
    tags = {"color_space": "bt709", "color_range": "tv", "chroma_location": siting}
    planes = planes_of(planar, WIDTH, HEIGHT, seed=11)
    conversion = conversion_for(stream(pix_fmt=planar, width=WIDTH, height=HEIGHT, **tags))
    read = yuv.reading(planar, WIDTH, HEIGHT, **tags)
    assert fields(conversion) == fields(read)
    command = [
        *("-f", "rawvideo", "-pix_fmt", planar, "-s", f"{WIDTH}x{HEIGHT}", "-framerate", "25"),
        *("-i", "pipe:0"),
    ]
    data = raw(planes, planar)
    process = subprocess.run(
        ["ffmpeg", "-v", "error", *command, *decode_output(conversion), "-"],
        input=data,
        capture_output=True,
        check=True,
    )
    frames = np.frombuffer(process.stdout, dtype="<u2").reshape(FRAMES, 3, HEIGHT, WIDTH)
    rgb = np.stack((frames[:, 2], frames[:, 0], frames[:, 1]), axis=-1)  # G, B, R planes
    compared(f"{planar} {siting} (raw)", the_library_s(planes, read), rgb)


@pytest.mark.parametrize("matrix", sorted(yuv.MATRICES))
@pytest.mark.parametrize("color_range", ["tv", "pc"])
def test_every_matrix(tmp_path: Path, matrix: str, color_range: str) -> None:
    tags = {"color_space": matrix, "color_range": color_range, "chroma_location": "left"}
    path = ffv1(tmp_path, "yuv420p10le", tags, seed=3)
    held(path, yuv.reading("yuv420p10le", WIDTH, HEIGHT, **tags))


@pytest.mark.parametrize(
    ("size", "guess"),
    [
        ((1280, 16), "bt709"),
        ((720, 16), "smpte170m"),
        ((16, 578), "bt709"),
        ((16, 576), "smpte170m"),
    ],
)
def test_an_untagged_matrix_guessed_by_the_size(
    tmp_path: Path, size: tuple[int, int], guess: str
) -> None:
    path = ffv1(tmp_path, "yuv420p", {}, seed=5, size=size)
    meant = yuv.reading("yuv420p", *size)
    assert (meant.matrix, meant.matrix_guessed) == (yuv.MATRICES[guess], True)
    held(path, meant)


@pytest.mark.parametrize("planar", ["yuv420p", "yuv422p10le", "yuv444p16le", "yuv410p", "gbrp"])
def test_wholly_untagged(tmp_path: Path, planar: str) -> None:
    path = ffv1(tmp_path, planar, {}, seed=9)
    meant = yuv.reading(planar, WIDTH, HEIGHT)
    assert meant.guessed or planar == "gbrp"
    held(path, meant)


@pytest.mark.parametrize("planar", ["yuvj420p", "yuvj422p", "yuvj444p"])
def test_the_jpeg_family(tmp_path: Path, planar: str) -> None:
    # MJPEG says its full range, centre siting and BT.601 itself (the CLI's test_decode.py).
    planes = planes_of(planar, WIDTH, HEIGHT, seed=13)
    path = written(tmp_path / "source.mkv", planes, planar, {}, ["-c:v", "mjpeg", "-q:v", "2"])
    tags = {"color_space": "bt470bg", "color_range": "pc", "chroma_location": "center"}
    held(path, yuv.reading(planar, WIDTH, HEIGHT, **tags))


@pytest.mark.parametrize(
    ("pix_fmt", "tags", "matrix"),
    [
        ("yuva420p", {}, None),
        ("pal8", {}, None),
        ("gray16le", {}, None),
        ("yuv420p", {"color_space": "bt2020c"}, None),
        ("yuv420p", {"color_space": "smpte240m"}, None),
        ("gbrp", {"color_range": "tv"}, None),
        ("yuvj420p", {"color_range": "tv"}, None),
        ("gbrp", {}, "bt709"),
        ("yuv420p", {"color_space": "bt709"}, "bt2020c"),
    ],
)
def test_the_refusals_are_the_command_line_s(
    pix_fmt: str, tags: dict[str, str], matrix: str | None
) -> None:
    with pytest.raises(MediaError) as command_line:
        conversion_for(stream(pix_fmt=pix_fmt, width=WIDTH, height=HEIGHT, **tags), matrix)
    with pytest.raises(yuv.SourceRefused) as library:
        yuv.reading(pix_fmt, WIDTH, HEIGHT, matrix=matrix, **tags)
    if library.value.kind == "rgb-matrix":  # the command line names its option
        assert str(command_line.value) == (
            f"--input-matrix {matrix}: the source is RGB, it has no matrix"
        )
    else:
        assert str(command_line.value) == str(library.value)
    assert "--" not in str(library.value)


# Real codecs, their planes as PyAV decodes them, the nodes' way: the stream's own planar format,
# or its planar twin by PyAV's reformat (swscale's repack).
REAL = [
    ("yuv420p", ["-c:v", "libx264", "-crf", "20"], "mp4", "libx264"),
    ("yuv420p10le", ["-c:v", "libx264", "-crf", "20"], "mkv", "libx264"),
    ("yuv422p10le", ["-c:v", "prores_ks", "-profile:v", "3"], "mov", "prores_ks"),
    ("yuv420p", ["-c:v", "libx265", "-x265-params", "log-level=error"], "mkv", "libx265"),
    ("yuvj422p", ["-c:v", "mjpeg", "-q:v", "3"], "avi", "mjpeg"),
    ("gbrp", ["-c:v", "ffv1"], "mkv", "ffv1"),
]


# PyAV's decode of a file's planes, in its own process (HAS_AV): argv the file, the planar format
# and the .npz the planes go to, each frame's planes cropped from their line size; a frame of
# another format reformatted to it first (swscale's repack, the nodes' way for a packed format).
PYAV_PLANES = """
import sys
import av
import numpy as np

path, planar, out = sys.argv[1:]
word = np.dtype(np.uint8) if av.VideoFormat(planar).components[0].bits == 8 else np.dtype("<u2")
frames = []
with av.open(path) as container:
    for frame in container.decode(video=0):
        if frame.format.name != planar:
            frame = frame.reformat(format=planar)
        frames.append(
            [
                np.frombuffer(memoryview(plane), dtype=word)
                .reshape(plane.height, plane.line_size // word.itemsize)[:, : plane.width]
                .astype(np.uint16)
                for plane in frame.planes
            ]
        )
np.savez(out, *(np.stack(plane) for plane in zip(*frames)))
"""


def pyav_planes(path: Path, planar: str) -> Planes:
    out = path.with_suffix(".npz")
    subprocess.run([sys.executable, "-c", PYAV_PLANES, str(path), planar, str(out)], check=True)
    with np.load(out) as planes:
        return planes["arr_0"], planes["arr_1"], planes["arr_2"]


@needs_av
@pytest.mark.parametrize(("planar", "codec", "suffix", "encoder"), REAL)
def test_real_codecs_decoded_by_pyav(
    tmp_path: Path, planar: str, codec: list[str], suffix: str, encoder: str
) -> None:
    if not _encoder(encoder):
        pytest.skip(f"needs ffmpeg's {encoder}")
    tags = {} if encoder in ("mjpeg", "ffv1") else {"color_space": "bt709", "color_range": "tv"}
    planes = planes_of(planar, WIDTH, HEIGHT, seed=17)
    path = written(tmp_path / f"source.{suffix}", planes, planar, tags, codec)
    source = examine(path)
    found = source.stream
    with source.decoder() as reader:
        frames = reader.read(source.frames)
    read = yuv.reading(
        found.pix_fmt,
        found.width,
        found.height,
        found.color_space,
        found.color_range,
        found.chroma_location,
    )
    by_pyav = pyav_planes(path, read.planar)
    by_ffmpeg = decoded(path, read.planar, WIDTH, HEIGHT)
    for mine, theirs in zip(by_pyav, by_ffmpeg, strict=True):
        assert np.array_equal(mine, theirs)  # PyAV's decode is the command line's ffmpeg's
    compared(f"{encoder} {planar} by PyAV", the_library_s(by_pyav, read), frames)


def test_the_master_read_back_by_the_command_line(tmp_path: Path) -> None:
    # The master's planes of 16-bit RGB, stored as FFV1 tagged BT.709 limited: the command line
    # reads them as it reads any source, within one code of the library's own reading of them
    # (MASTER), and within two of the RGB written, the master's limited range spanning fewer
    # codes than RGB's (yuv.master_planes).
    rng = np.random.default_rng(19)
    codes = rng.integers(0, 65536, (FRAMES, HEIGHT, WIDTH, 3))
    rgb = torch.from_numpy(codes.astype(np.float32)) / 65535
    master = yuv.master_planes(rgb).numpy()
    planes = (master[:, 0], master[:, 1], master[:, 2])
    tags = {"color_space": "bt709", "color_range": "tv"}
    path = written(tmp_path / "master.mkv", planes, "yuv444p16le", tags, ["-c:v", "ffv1"])
    held(path, yuv.MASTER)
    source = examine(path)
    with source.decoder() as reader:
        frames = reader.read(source.frames).astype(np.int64)
    difference = np.abs(frames - codes)
    print(
        f"the master read back: {np.mean(difference == 0):.2%} equal,"
        f" {np.mean(difference == 1):.2%} one code off, {np.mean(difference == 2):.2%} two"
    )
    assert difference.max() <= 2


def test_the_grid_s_figures() -> None:
    # Last in the file, after the cases above: every sample within one code, the share one off.
    if not FIGURES:
        pytest.skip("no case ran before this one")
    samples = sum(total for total, _ in FIGURES.values())
    one = sum(off for _, off in FIGURES.values())
    worst = max(off / total for total, off in FIGURES.values())
    print(
        f"{len(FIGURES)} cases, {samples} samples: {one / samples:.4%} one code off,"
        f" {worst:.4%} at most in a case"
    )
    assert one / samples < 0.01

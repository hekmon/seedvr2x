"""A source, examined before any GPU work: what it declares, what seedvr2x refuses, how its frames
become RGB, and the first pass (DESIGN.md, Input). A source is a video file, or a directory of
segments, each a file examined alike. A job resumed checks its record between what the source
declares and its first pass, whose record it then trusts when nothing changed (DESIGN.md, Pause
and resume)."""

import logging
import threading
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from seedvr2x.media.conversion import Conversion, conversion_for
from seedvr2x.media.decode import Decoder
from seedvr2x.media.ffmpeg import MediaError, input_args
from seedvr2x.media.files import sha256
from seedvr2x.media.probe import VideoStream, probe
from seedvr2x.media.scan import scan, timing_error

logger = logging.getLogger(__name__)

# The first two rows of a display matrix that neither rotates, flips nor scales (a b u, c d v;
# a to d in 16.16 fixed point). The third row, a translation, changes no pixel.
IDENTITY = (65536, 0, 0, 0, 65536, 0)


@dataclass(frozen=True)
class Declared:
    """A video file as it declares itself, accepted: its first video stream, how its frames
    become RGB, and its sample aspect; its frames not counted yet (first_pass)."""

    path: Path
    stream: VideoStream
    conversion: Conversion
    sample_aspect: Fraction  # the override, else the declared one, else 1 (square pixels)


@dataclass(frozen=True)
class FirstPass:
    """What the first pass found in a file: its frames, decoded and counted, at the constant rate
    declared; and its content's SHA-256, when hashed."""

    frames: int
    sha256: str | None = None


@dataclass(frozen=True)
class Source:
    """A video file seedvr2x reads: its first video stream."""

    path: Path
    stream: VideoStream
    conversion: Conversion
    frames: int  # counted by the first pass
    sample_aspect: Fraction  # the override, else the declared one, else 1 (square pixels)
    sha256: str | None = None  # the file's content, hashed for a manifest

    @property
    def display_aspect(self) -> Fraction:
        return self.sample_aspect * self.stream.width / self.stream.height

    def decoder(self) -> Decoder:
        """A decoder of every frame, from the first, checked against the first pass's count."""
        return Decoder(
            input_args(self.path),
            self.conversion,
            self.stream.width,
            self.stream.height,
            self.frames,
        )


def examine(path: Path, matrix: str | None = None, sample_aspect: Fraction | None = None) -> Source:
    """Probe path, refuse what seedvr2x doesn't read (declare), then decode it once to count and
    time its frames (first_pass). `matrix` (ffprobe's name) and `sample_aspect` override the
    source's tags.

    Raises MediaError with the reason for a refusal."""
    declared = declare(path, matrix, sample_aspect)
    return counted(declared, first_pass(declared))


def declare(
    path: Path, matrix: str | None = None, sample_aspect: Fraction | None = None
) -> Declared:
    """Probe path and refuse what seedvr2x doesn't read, by what it declares. `matrix` (ffprobe's
    name) and `sample_aspect` override the source's tags.

    Raises MediaError with the reason for a refusal."""
    stream = probe(path)
    try:
        refusal = declared_refusal(stream)
        if refusal:
            raise MediaError(refusal)
        conversion = conversion_for(stream, matrix)
    except MediaError as error:
        raise MediaError(f"{path}: {error}") from None
    if sample_aspect is None:
        sample_aspect = stream.sample_aspect
        if sample_aspect is None:
            logger.info("%s: no sample aspect declared, read as square pixels", path)
            sample_aspect = Fraction(1)
    if conversion.guessed:
        logger.warning(
            "%s: untagged, guessed: %s (--input-matrix sets the matrix)",
            path,
            ", ".join(conversion.guessed),
        )
    return Declared(path, stream, conversion, sample_aspect)


def first_pass(declared: Declared, hashed: bool = False) -> FirstPass:
    """Decode every frame of the file once, to count and time them, refused unless at the constant
    rate declared; and, when `hashed`, hash its content meanwhile, from another thread: the file
    read a second time, at the speed of the disk, while ffmpeg decodes it.

    Raises MediaError with the reason for a refusal."""
    path, stream = declared.path, declared.stream
    hashed_as: list[str | Exception] = []

    def hash_content() -> None:
        try:
            hashed_as.append(sha256(path))
        except Exception as error:  # raised again by the caller, below
            hashed_as.append(error)

    # A daemon: should the scan fail, the hash isn't waited for, and ends with the process.
    reader = threading.Thread(target=hash_content, daemon=True)
    if hashed:
        reader.start()
    scanned = scan(path)
    refusal = timing_error(scanned, stream.frame_rate, stream.avg_frame_rate)
    if refusal:
        raise MediaError(f"{path}: {refusal}")
    if not hashed:
        return FirstPass(scanned.frames)
    reader.join()
    [digest] = hashed_as
    if isinstance(digest, OSError):
        raise MediaError(f"{path}: not readable: {digest}") from digest
    if isinstance(digest, Exception):
        raise digest
    return FirstPass(scanned.frames, digest)


def counted(declared: Declared, found: FirstPass) -> Source:
    """The source declared, its frames counted by its first pass, or by the record of one."""
    stream = declared.stream
    source = Source(
        declared.path,
        stream,
        declared.conversion,
        found.frames,
        declared.sample_aspect,
        found.sha256,
    )
    logger.info(
        "%s: %d frames, %dx%d %s at %s fps, sample aspect %s (display %s), read as %s",
        source.path,
        source.frames,
        stream.width,
        stream.height,
        stream.pix_fmt,
        stream.frame_rate,
        _ratio(source.sample_aspect),
        _ratio(source.display_aspect),
        source.conversion.describe(),
    )
    return source


# The segments of a directory: the files `sptenc encode <dir>` reads, in its order
# (cmd/sptenc/helpers.go, getSegmentsFromDir): .mkv and .mp4, the extension in any case,
# subdirectories skipped, sorted by name byte for byte. sptenc's split writes seg_%06d.mkv.
# seedvr2x reads a directory the same way (DESIGN.md, Input).
SEGMENT_EXTENSIONS = (".mkv", ".mp4")


def segment_files(directory: Path) -> list[Path]:
    """The segments of a directory, in their order."""
    files = [
        entry
        for entry in directory.iterdir()
        if entry.suffix.lower() in SEGMENT_EXTENSIONS and not entry.is_dir()
    ]
    return sorted(files, key=lambda path: path.name.encode())


def examine_directory(
    directory: Path, matrix: str | None = None, sample_aspect: Fraction | None = None
) -> list[Source]:
    """Examine each segment of a directory: declared alike (declare_directory), then their first
    passes.

    Raises MediaError for an empty directory, a segment refused, or segments that differ."""
    sources = [
        counted(each, first_pass(each))
        for each in declare_directory(directory, matrix, sample_aspect)
    ]
    logger.info(
        "%s: %d segments, %d frames", directory, len(sources), sum(s.frames for s in sources)
    )
    return sources


def declare_directory(
    directory: Path, matrix: str | None = None, sample_aspect: Fraction | None = None
) -> list[Declared]:
    """Declare each segment of a directory (declare), which must make one source: the same size,
    sample aspect, frame rate, conversion and colour tags throughout, since the output mirrors
    the segments as one job of one geometry, and sptenc encodes them alike.

    Raises MediaError for an empty directory, a segment refused, or segments that differ."""
    files = segment_files(directory)
    if not files:
        raise MediaError(f"{directory}: no segment (.mkv or .mp4 files)")
    segments = [declare(path, matrix, sample_aspect) for path in files]
    first = segments[0]
    for segment in segments[1:]:
        for what, ours, theirs in (
            ("size", _size(segment.stream), _size(first.stream)),
            ("sample aspect", segment.sample_aspect, first.sample_aspect),
            ("frame rate", segment.stream.frame_rate, first.stream.frame_rate),
            ("conversion", segment.conversion.describe(), first.conversion.describe()),
            ("pixel format", segment.conversion.planar, first.conversion.planar),
            ("primaries", segment.stream.color_primaries, first.stream.color_primaries),
            ("transfer", segment.stream.color_transfer, first.stream.color_transfer),
        ):
            if ours != theirs:
                raise MediaError(
                    f"{segment.path}: {what} {ours or 'untagged'}, where {first.path.name} has"
                    f" {theirs or 'untagged'}: the segments of a directory must make one source"
                )
    return segments


def _size(stream: VideoStream) -> str:
    return f"{stream.width}x{stream.height}"


def declared_refusal(stream: VideoStream) -> str:
    """Why a stream is refused for what it declares, or "": interlacing, a rotation or flip, a
    declared crop (DESIGN.md, Not in the first version)."""
    # sptenc's rule (ffmpeg/probe.go, IsInterlaced): "unknown" is not interlaced, or every file
    # ffprobe can't tell anything about would be refused.
    if stream.field_order not in ("", "progressive"):
        return f"interlaced (field order {stream.field_order}): not supported"
    if stream.display_matrix is not None and stream.display_matrix[:6] != IDENTITY:
        return f"rotated or flipped by its display matrix {stream.display_matrix}: not supported"
    if stream.cropped:
        # ffmpeg applies the crop on decode, while ffprobe declares the uncropped size.
        return "cropped by its container: not supported"
    return ""


def _ratio(value: Fraction) -> str:
    return f"{value.numerator}:{value.denominator}"

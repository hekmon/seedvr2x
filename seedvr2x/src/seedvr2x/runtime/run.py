"""A job's run: its shots upscaled in order, their frames read from the source and written as they
come (DESIGN.md, Pipeline, per shot), unit by unit (DESIGN.md, Pause and resume)."""

import logging
import resource
import time
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from types import TracebackType
from typing import Self

import numpy as np
import numpy.typing as npt
import torch

from seedvr2x.media.checksums import write_checksums
from seedvr2x.media.decode import to_float32
from seedvr2x.media.ffmpeg import MediaError
from seedvr2x.media.files import make_directories
from seedvr2x.media.reader import Reader, listed
from seedvr2x.media.source import Source
from seedvr2x.media.writer import FFV1Writer, Tags, Writer
from seedvr2x.runtime.job import OutputSegment, Shot
from seedvr2x.runtime.model import Models
from seedvr2x.runtime.shot import (
    CopyError,
    Correction,
    decode_shot,
    encode_shot,
    finite,
    merge_windows,
    sample_windows,
    shot_layout,
)
from seedvr2x.runtime.stop import Stop
from seedvr2x.runtime.units import Units

logger = logging.getLogger(__name__)

GIB = 1024**3

# The frames read at a time when a shot's input copy is made again: 8 at 1080p are 200 MB of
# float32.
COPY_READS = 8


@dataclass(frozen=True)
class Copies:
    """split's input copies (DESIGN.md, Colour correction): the input's frame size, as stored,
    and rate."""

    width: int
    height: int
    frame_rate: Fraction


def run_job(
    models: Models,
    source: Source,
    shots: Sequence[Shot],
    segments: Sequence[OutputSegment],
    target: tuple[int, int],
    seed: int,
    window: int | None,
    units: Units,
    write: Callable[[npt.NDArray[np.float32]], None],
    stop: Stop | None = None,
    split: bool = False,
    *,
    numz_padding: bool = False,
) -> None:
    """Upscale the shots of a job, which cover its source in order (job.shots_from_cuts), each
    with its own seed (Shot.seed), its frames resized to target (job.target_size), its DiT windows
    capped at `window` latents; write(frames) takes the output frames as they come, (n, H', W', 3)
    float32 in [0, 1], the segments' in turn.

    A shot's units run in order: its encode, then its windows, each kept by `units` as it is
    done, then its decode. With units kept on disk, a segment's shots are all encoded and sampled
    before the first of them is decoded: the segment's decode and write is then a unit of its own,
    and units never interleave. With nothing kept, each shot is decoded as soon as it is sampled,
    its steps in a row, and nothing builds up.

    What units kept already is skipped (DESIGN.md, Pause and resume): a finished segment; a
    shot's encode and the windows done; the input frames of what is skipped, never decoded, the
    next ones reached through the frame index (Inputs). An unfinished segment's decode and write
    restart whole, from its shots' windows kept.

    stop, when given, is checked before each unit (Stop.check). With split, the frames each
    encode reads are copied as they come (units.copy_path), and the shot's decode corrects each
    slice against them as it comes out (DESIGN.md, Colour correction). A copy is derived data: one
    missing, or whose checksums are, is made again from the input before the shot's windows, its
    latent and windows kept; one its decode can't read whole and intact (CopyError) is removed,
    its checksums with it, and the run stops there, for the next run to make it again.

    numz_padding is for tests only (cli.NUMZ_PADDING): each encode's frames padded as numz pads
    them (shot.encode_shot)."""
    stream = source.stream
    copies = Copies(stream.width, stream.height, source.frame_rate) if split else None
    groups = [
        [i for i, shot in enumerate(shots) if s.start <= shot.start < s.end] for s in segments
    ]
    if sorted(i for group in groups for i in group) != list(range(len(shots))):
        raise ValueError("the segments don't hold whole shots")
    with Inputs(source) as inputs:
        for segment, group in enumerate(groups):
            if units.finished(segment):
                continue
            sample = (models, inputs, shots, target, seed, window, units, stop, copies)
            if units.persistent:
                for index in group:
                    _sample(index, *sample, numz_padding)
                _begin(stop, f"segment {segment + 1}/{len(segments)}'s decode and write")
            for index in group:
                if not units.persistent:
                    _sample(index, *sample, numz_padding)
                    _begin(stop, f"shot {index + 1}/{len(shots)}'s decode")
                _decode(models, shots, index, target, window, units, write, copies)


def _begin(stop: Stop | None, unit: str) -> None:
    """Before a unit: stop there if asked (Stop.check), else name it for a Ctrl-C's notice."""
    if stop is not None:
        stop.check()
        stop.unit = unit


def _sample(
    index: int,
    models: Models,
    inputs: "Inputs",
    shots: Sequence[Shot],
    target: tuple[int, int],
    seed: int,
    window: int | None,
    units: Units,
    stop: Stop | None,
    copies: Copies | None,
    numz_padding: bool,
) -> None:
    """Shot `index`'s encode and windows, those not kept yet; with copies, the frames the encode
    reads copied, the copy and its checksums whole before the latent is recorded, so recorded
    with it, and the copy of a shot encoded already made again if it or its checksums are
    missing. numz_padding as run_job takes it."""
    shot = shots[index]
    name = f"shot {index + 1}/{len(shots)}"
    layout = shot_layout(shot.frames, window)
    done = units.windows_done(index)
    latent = None if done == len(layout) else units.latent(index)
    encoded = done == len(layout) or latent is not None
    if copies is not None and encoded:
        if not (units.copy_path(index).exists() and units.checksums_path(index).exists()):
            _copy_again(index, name, inputs, shots, units, stop, copies)
    if done == len(layout):
        return
    if latent is None:
        _begin(stop, f"{name}'s encode")
        logger.debug("%s: encoding", name)
        started = _started(models)
        read = inputs.reader(shot)
        if copies is None:
            latent = encode_shot(
                *(models, read, shot.frames, target, shot.seed(seed)), numz_padding=numz_padding
            )
        else:
            with _copying(units, index, copies) as copy:
                latent = encode_shot(
                    *(models, _copied(read, copy), shot.frames, target, shot.seed(seed)),
                    numz_padding=numz_padding,
                )
        finite(latent, f"{name}'s encode")
        units.save_latent(index, latent)
        logger.info(
            "%s, frames %d to %d (%d), seed %d: encoded in %s",
            name,
            shot.start,
            shot.end - 1,
            shot.frames,
            shot.seed(seed),
            _spent(models, started),
        )
    else:
        latent = latent.to(models.device)
    sampled = sample_windows(models, latent, layout, shot.seed(seed), done)
    for number in range(done, len(layout)):
        _begin(stop, f"{name}'s window {number + 1}/{len(layout)}")
        logger.debug("%s: window %d/%d", name, number + 1, len(layout))
        started = _started(models)
        output = next(sampled)
        finite(output, f"{name}'s window {number + 1}/{len(layout)}")
        units.save_window(index, number, output)
        start, end = layout[number]
        logger.info(
            "%s: window %d/%d, latents %d to %d, in %s",
            name,
            number + 1,
            len(layout),
            start,
            end - 1,
            _spent(models, started),
        )
    del sampled, latent
    units.drop_latent(index)


def _copy_again(
    index: int,
    name: str,
    inputs: "Inputs",
    shots: Sequence[Shot],
    units: Units,
    stop: Stop | None,
    copies: Copies,
) -> None:
    """Make shot `index`'s input copy again, as its encode did, from the input frames it read,
    reached through the frame index (Inputs): bit for bit the frames the first copy held, the
    input's content being checked (DESIGN.md, Colour correction), and each frame against the
    index, unless an environment change accepted since then changed ffmpeg or its conversions,
    the copy then being the new decode's frames (--accept-env-change)."""
    shot = shots[index]
    _begin(stop, f"{name}'s input copy")
    started = time.monotonic()
    paths = (units.copy_path(index), units.checksums_path(index))
    missing = " and ".join(path.name for path in paths if not path.exists())
    read = inputs.reader(shot)
    with _copying(units, index, copies) as copy:
        for first in range(0, shot.frames, COPY_READS):
            copy.write(read(min(COPY_READS, shot.frames - first)))
    logger.info(
        "%s: %s missing: its input copy made again from the input in %.1f s",
        name,
        missing,
        time.monotonic() - started,
    )


def _decode(
    models: Models,
    shots: Sequence[Shot],
    index: int,
    target: tuple[int, int],
    window: int | None,
    units: Units,
    write: Callable[[npt.NDArray[np.float32]], None],
    copies: Copies | None,
) -> None:
    """Shot `index`'s decode, from its windows kept, its frames written as they come: with copies,
    each slice corrected against its input copy first."""
    shot = shots[index]
    name = f"shot {index + 1}/{len(shots)}"
    logger.debug("%s: decoding", name)
    started = _started(models)
    merged = merge_windows(units.take_windows(index), shot_layout(shot.frames, window))
    correction = None
    if copies is not None:
        paths = (units.copy_path(index), units.checksums_path(index))
        correction = Correction(*paths, copies.width, copies.height)
    try:
        decode_shot(models, merged, shot.frames, target, write, name, shot.start, correction)
    except CopyError:
        # Derived data: removed, the next run makes it again from the input (_sample).
        units.copy_path(index).unlink(missing_ok=True)
        units.checksums_path(index).unlink(missing_ok=True)
        raise
    units.shot_decoded(index)
    logger.info(
        "%s: decoded in %s; RAM %.2f GiB, peak %.2f GiB",
        name,
        _spent(models, started),
        _resident() / GIB,
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / GIB,
    )


@contextmanager
def _copying(units: Units, index: int, copies: Copies) -> Generator[FFV1Writer]:
    """The writer of shot `index`'s input copy, its directory made first: synced in its parent, as
    a unit's directory is, the copy being recorded with the shot's latent. Once the copy is whole,
    its frames' checksums are written whole beside it (DESIGN.md, Colour correction, input copy).
    Checksums of a copy written before go first, so that checksums beside a copy are always its
    own, even after a stop between the two files: the copy's rename syncs their directory."""
    path = units.copy_path(index)
    make_directories(path.parent)
    units.checksums_path(index).unlink(missing_ok=True)
    size = (copies.width, copies.height, copies.frame_rate)
    with FFV1Writer(path, "gbrp16le", *size, Tags()) as copy:
        yield copy
    write_checksums(units.checksums_path(index), copy.checksums)


def _copied(
    read: Callable[[int], npt.NDArray[np.float32]], copy: Writer
) -> Callable[[int], npt.NDArray[np.float32]]:
    """read, the frames it gives written to copy as well."""

    def read_and_copy(count: int) -> npt.NDArray[np.float32]:
        frames = read(count)
        copy.write(frames)
        return frames

    return read_and_copy


class Inputs:
    """The source's frames, read in order, each checked against the frame index (Reader). A reader
    is opened at the first frame asked, and finished once every frame is read, its count checked.
    Frames not asked for, those of shots a resume skips, are never decoded: the next frame asked
    is reached through the frame index, a reader opened there (DESIGN.md, Pause and resume). A run
    reading every shot from the source's first frame keeps one reader from the start, which seeks
    nowhere, its frames checked all the same: PROVISIONAL (DESIGN.md checks a read from frame n),
    one code path for the run and its resume, at the cost of a CRC-32 a frame in ffmpeg, and a
    source changed under a run said: its frames' count and pts checked, a picture other than the
    first pass's taken from the start with a warning (Reader.otherwise), every such frame of the
    run summed up at its end. Used as a context manager, which stops the reader left open on an
    exception."""

    def __init__(self, source: Source) -> None:
        self._source = source
        self._reader: Reader | None = None
        self._position = 0  # the index of the next frame the reader gives
        self._otherwise: list[int] = []  # each reader's Reader.otherwise, once it is done

    def reader(self, shot: Shot) -> Callable[[int], npt.NDArray[np.float32]]:
        """read(n), giving the next n of shot's frames, (n, H, W, 3) float32 in [0, 1], from its
        first: shots are asked for in order, a reader opened at a shot whose first frame isn't the
        next one."""
        path = self._source.path
        if self._position > shot.start:
            raise ValueError(f"frame {shot.start} asked, frame {self._position} is next")
        if self._reader is None or self._position < shot.start:
            if self._reader is not None:
                self._reader.stop()
                self._otherwise += self._reader.otherwise
            self._reader = None  # a failure opening the next leaves none open
            self._reader = self._source.reader(shot.start)
            self._position = shot.start
        reader = self._reader

        def read(count: int) -> npt.NDArray[np.float32]:
            frames = reader.read(count)
            if frames.shape[0] != count:
                raise MediaError(f"{path}: the stream ended after {reader.position} frames")
            self._position += count
            return to_float32(frames)

        return read

    def _close(self, finish: bool) -> None:
        """Finish the open reader when `finish` and its every frame is read; else stop it."""
        reader, self._reader = self._reader, None
        if reader is None:
            return
        try:
            if finish and self._position == self._source.frames:
                reader.finish()
            else:
                reader.stop()
        finally:
            self._otherwise += reader.otherwise

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self._close(finish=error is None)
        finally:
            if self._otherwise:
                # The run's last word on it (Reader._take): what it means for a resume.
                count = len(self._otherwise)
                logger.warning(
                    "%s: %d frame%s in all decoded otherwise than in the first pass, read from the"
                    " start and taken as decoded (%s): an output is bit-identical across a resume"
                    " only for a source that decodes the same each time",
                    self._source.path,
                    count,
                    "" if count == 1 else "s",
                    listed(self._otherwise),
                )


def _started(models: Models) -> float:
    torch.cuda.reset_peak_memory_stats(models.device)
    return time.monotonic()


def _spent(models: Models, started: float) -> str:
    """The time since started and the VRAM peak since, as the logs give them."""
    peak = torch.cuda.max_memory_allocated(models.device) / GIB
    return f"{time.monotonic() - started:.1f} s, VRAM peak {peak:.2f} GiB"


def _resident() -> int:
    """The process's resident memory now, in bytes (Linux)."""
    with open("/proc/self/statm") as statm:
        return int(statm.read().split()[1]) * resource.getpagesize()

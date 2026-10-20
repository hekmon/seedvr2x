"""A job's run: its shots upscaled in order, their frames read from the source and written as they
come (DESIGN.md, Pipeline, per shot), unit by unit (DESIGN.md, Pause and resume)."""

import logging
import math
import mmap
import os
import resource
import time
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
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
from seedvr2x.media.writer import FFV1Writer, Tags, Writer, quantise
from seedvr2x.runtime.job import KEPT_SAMPLE_BYTES, OutputSegment, Run, Shot, output_size, run_of
from seedvr2x.runtime.model import Models
from seedvr2x.runtime.plan import Plan, Tiles
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
from seedvr2x.runtime.tiling import ScratchError
from seedvr2x.runtime.units import Units

logger = logging.getLogger(__name__)

GIB = 1024**3

# The frames read at a time when a shot's input copy is made again: 8 at 1080p are 200 MB of
# float32.
COPY_READS = 8


# What the output's frames are written by (run_job's write): (n, H', W', 3) float32 in [0, 1] as the
# decode gives them, or uint16, the values the writers quantise them to (writer.quantise), a shot's
# frames kept between its runs (Kept).
Write = Callable[[npt.NDArray[np.float32] | npt.NDArray[np.uint16]], None]


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
    plan: Plan,
    units: Units,
    write: Write,
    stop: Stop | None = None,
    split: bool = False,
    *,
    numz_padding: bool = False,
    numz_draw: bool = False,
    numz_tail: bool = False,
    clock: object | None = None,
    dump: Callable[[int, npt.NDArray[np.float32]], None] | None = None,
) -> None:
    """Upscale the shots of a job, which cover its source in order (job.shots_from_cuts), each
    with its own seed (Shot.seed), its frames resized to target (job.target_size), as the job's
    plan says (runtime/plan.py): its DiT windows capped at plan.window latents, its encodes and
    decodes tiled or not; write(frames) takes the output frames as they come, (n, H', W', 3)
    float32 in [0, 1], the segments' in turn; with several runs a shot, a shot's frames kept
    between its runs come as uint16, the values the writers quantise a frame to
    (writer.quantise), which they take as they are. dump(frame, values), when given, is told each
    of those kept frames as its run decodes it, its index in the job and its float32 values, for
    --dump-frames (DESIGN.md, Several runs of a shot: "Frames asked for in float are written by
    the run that makes them"); the frames write is given as float32 it isn't told.

    Before each unit, the models are told its phase (Models.enter), which puts on the GPU what
    the plan has there for it.

    A shot's units run in order: its encode, then its windows, each kept by `units` as it is
    done, then its decode. A segment's shots are all encoded and sampled before the first of them
    is decoded: the segment's decode and write is then a unit of its own, and units never
    interleave. With several runs a shot (units.manifest.runs_per_shot; DESIGN.md, Several runs
    of a shot), each run is a whole run of the shot fed from its first frame, its encode and its
    windows units of their own, in turn; the shot's decode decodes its runs in turn, each giving
    the frames job.run_of takes from it, and writes the shot's frames in their order
    (_decode_runs).

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
    them (shot.encode_shot). So is numz_draw (cli.NUMZ_DRAW): each shot's noise drawn on the
    device, as numz draws it (shot.sample_windows). So is numz_tail (cli.NUMZ_TAIL): each shot's
    end padded as numz pads it, mirrored, not repeated (shot.pad_4n1).

    clock is --until's, with the time estimates each unit's cost comes from (_begin; DESIGN.md,
    Pause and resume, Stopping): not built yet, None."""
    if clock is not None:
        raise NotImplementedError("--until isn't built yet")
    stream = source.stream
    copies = Copies(stream.width, stream.height, source.frame_rate) if split else None
    groups = [
        [i for i, shot in enumerate(shots) if s.start <= shot.start < s.end] for s in segments
    ]
    if sorted(i for group in groups for i in group) != list(range(len(shots))):
        raise ValueError("the segments don't hold whole shots")
    if list(units.manifest.shots) != list(shots):
        raise ValueError("the units' manifest is of other shots")
    with Inputs(source) as inputs:
        for segment, group in enumerate(groups):
            if units.finished(segment):
                continue
            sample = (models, inputs, shots, target, seed, plan, units, stop, copies)
            for index in group:
                for run in units.manifest.shot_runs(index):
                    _sample(run, *sample, numz_padding, numz_draw, numz_tail)
            _begin(stop, f"segment {segment + 1}/{len(segments)}'s decode and write", None)
            for index in group:
                runs = units.manifest.shot_runs(index)
                if len(runs) == 1:
                    _decode(models, shots, runs[0], target, plan, units, write, copies)
                else:
                    _decode_runs(models, shots, index, target, plan, units, write, copies, dump)


def _begin(stop: Stop | None, unit: str, cost: float | None) -> None:
    """Before a unit: stop there if asked (Stop.check), else name it for a Ctrl-C's notice. cost
    is the time the unit is expected to take, in seconds, None when nothing estimates it: for
    --until, which stops before a unit that wouldn't finish in time (DESIGN.md, Pause and resume,
    Stopping); not built yet, so not looked at."""
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
    plan: Plan,
    units: Units,
    stop: Stop | None,
    copies: Copies | None,
    numz_padding: bool,
    numz_draw: bool,
    numz_tail: bool,
) -> None:
    """Run `index`'s encode and windows (units.manifest.runs), those not kept yet; with copies, the
    frames a shot's first run's encode reads copied, the copy and its checksums whole before the
    latent is recorded, so recorded with it, and the copy of a shot whose first run is encoded
    already made again if it or its checksums are missing. A later run makes no copy: its
    correction reads its shot's from its own first frame (shot.Correction's first).
    numz_padding, numz_draw and numz_tail as run_job takes them.

    Between two of the shot's units the GPU holds nothing of the shot: its latent waits in
    system memory and each window's output in its file (units.Units), and its noise, drawn
    there, in system memory with the latent (shot.sample_windows). A unit's logged VRAM peak is
    then its own; the first window a run makes of a shot takes the noise's draw's time too, on
    the CPU."""
    run = units.manifest.runs[index]
    shot = Shot(run.start, run.end)  # the frames the run is fed
    name = _name(units, shots, index)
    layout = shot_layout(run.frames, plan.window)
    done = units.windows_done(index)
    latent = None if done == len(layout) else units.latent(index)
    encoded = done == len(layout) or latent is not None
    copying = copies if run.offset == 0 else None
    if copying is not None and encoded:
        if not (units.copy_path(index).exists() and units.checksums_path(index).exists()):
            _copy_again(index, name, inputs, shot, units, stop, copying)
    if done == len(layout):
        return
    if latent is None:
        _begin(stop, f"{name}'s encode", None)
        logger.debug("%s: encoding", name)
        started = _started(models)
        models.enter("encode")
        read = inputs.reader(shot)
        tiles = plan.encode_tiles
        scratch = _scratch(units, index, tiles)
        if copying is None:
            latent = encode_shot(
                *(models, read, shot.frames, target, run.seed(seed)),
                numz_padding=numz_padding,
                numz_tail=numz_tail,
                tiles=tiles,
                scratch=scratch,
                allowance=plan.scratch_allowance,
            )
        else:
            with _copying(units, index, copying) as copy:
                latent = encode_shot(
                    *(models, _copied(read, copy), shot.frames, target, run.seed(seed)),
                    numz_padding=numz_padding,
                    numz_tail=numz_tail,
                    tiles=tiles,
                    scratch=scratch,
                    allowance=plan.scratch_allowance,
                )
        finite(latent, f"{name}'s encode")
        # In system memory, as its encode gives it: the one the units keep or wrote, which the
        # windows are sampled from, as a resumed run samples them from the one read back.
        latent = units.save_latent(index, latent)
        logger.info(
            "%s, frames %d to %d (%d), seed %d: encoded in %s",
            name,
            shot.start,
            shot.end - 1,
            shot.frames,
            run.seed(seed),
            _spent(models, started),
        )
    # The latent is in system memory, the one the units keep or read back, where the noise is
    # drawn from it, with the first window made here (shot.sample_windows).
    sampled = sample_windows(models, latent, layout, run.seed(seed), done, numz_draw=numz_draw)
    for number in range(done, len(layout)):
        _begin(stop, f"{name}'s window {number + 1}/{len(layout)}", None)
        logger.debug("%s: window %d/%d", name, number + 1, len(layout))
        started = _started(models)
        models.enter("dit")
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
        # Kept in its file (units.Units.save_window): the output leaves the GPU before the next
        # window runs, so that a window's peak is its own.
        del output
    del sampled, latent
    units.drop_latent(index)


def _name(units: Units, shots: Sequence[Shot], index: int) -> str:
    """Run `index`'s name in the logs: its shot's, "shot 3/12", with several runs a shot the run's
    too, "shot 3/12, run 2/4 (from its frame 1)"."""
    manifest = units.manifest
    shot = manifest.shot_of(index)
    name = f"shot {shot + 1}/{len(shots)}"
    if manifest.runs_per_shot == 1:
        return name
    run, runs = manifest.runs[index], manifest.shot_runs(shot)
    # Of the shot's own runs: a shot too short for some has fewer than runs_per_shot.
    return f"{name}, run {runs.index(index) + 1}/{len(runs)} (from its frame {run.offset})"


def _copy_again(
    index: int,
    name: str,
    inputs: "Inputs",
    shot: Shot,
    units: Units,
    stop: Stop | None,
    copies: Copies,
) -> None:
    """Make the input copy of run `index`, a shot's first run whose frames are `shot`, again, as
    its encode did, from the input frames it read,
    reached through the frame index (Inputs): bit for bit the frames the first copy held, the
    input's content being checked (DESIGN.md, Colour correction), and each frame against the
    index, unless an environment change accepted since then changed ffmpeg or its conversions,
    the copy then being the new decode's frames (--accept-env-change)."""
    _begin(stop, f"{name}'s input copy", None)
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
    plan: Plan,
    units: Units,
    write: Callable[[npt.NDArray[np.float32]], None],
    copies: Copies | None,
) -> None:
    """Run `index`'s decode, from its windows kept, its frames written as they come: with copies,
    each slice corrected against its shot's input copy first, from the run's first frame."""
    run = units.manifest.runs[index]
    name = _name(units, shots, index)
    logger.debug("%s: decoding", name)
    started = _started(models)
    models.enter("decode")
    merged = merge_windows(units.take_windows(index), shot_layout(run.frames, plan.window))
    correction = None
    if copies is not None:
        paths = (units.copy_path(index), units.checksums_path(index))
        correction = Correction(*paths, copies.width, copies.height, run.offset)
    try:
        decode_shot(
            *(models, merged, run.frames, target, write, name, run.start, correction),
            tiles=plan.decode_tiles,
            scratch=_scratch(units, index, plan.decode_tiles),
            allowance=plan.scratch_allowance,
        )
    except CopyError:
        # Derived data: removed, the next run makes it again from the input (_sample).
        units.copy_path(index).unlink(missing_ok=True)
        units.checksums_path(index).unlink(missing_ok=True)
        raise
    logger.info(
        "%s: decoded in %s; RAM %.2f GiB, peak %.2f GiB",
        name,
        _spent(models, started),
        _resident() / GIB,
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / GIB,
    )


def _decode_runs(
    models: Models,
    shots: Sequence[Shot],
    index: int,
    target: tuple[int, int],
    plan: Plan,
    units: Units,
    write: Write,
    copies: Copies | None,
    dump: Callable[[int, npt.NDArray[np.float32]], None] | None = None,
) -> None:
    """Shot `index`'s decode with several runs (DESIGN.md, Several runs of a shot): its runs
    decoded in turn (_decode), each giving the frames job.run_of takes from it. Those of every run
    but the last are kept in the shot's file of frames (Kept, units.kept_path), as the runs give
    them; the last run's are written as they come, each after the kept frames before it, and
    the kept frames after its last once its decode is done: the shot's frames in their order,
    each written once. So the file holds only the frames the earlier runs give, a frame each,
    the shot's at the most. dump, as run_job takes it, is told each kept frame as it is decoded."""
    manifest = units.manifest
    runs = manifest.shot_runs(index)
    shot = shots[index]
    count = manifest.runs_per_shot
    taken = [run_of(count, frame) for frame in range(shot.frames)]
    last = manifest.runs[runs[-1]].offset
    kept = [frame for frame in range(shot.frames) if taken[frame] != last]
    height, width = output_size(target)
    with Kept(units.kept_path(runs[0]), kept, (height, width, 3)) as store:
        for number in runs[:-1]:
            run = manifest.runs[number]
            given = _given(run, taken, store.put, shot.start, dump)
            _decode(models, shots, number, target, plan, units, given, copies)
        emit = _interleaved(manifest.runs[runs[-1]], taken, store, write)
        _decode(models, shots, runs[-1], target, plan, units, emit, copies)
        store.flush(shot.frames, write)


def _given(
    run: Run,
    taken: Sequence[int],
    put: Callable[[int, npt.NDArray[np.float32]], None],
    first: int = 0,
    dump: Callable[[int, npt.NDArray[np.float32]], None] | None = None,
) -> Callable[[npt.NDArray[np.float32]], None]:
    """A run's decode's write: of the frames it gives in order, (n, H', W', 3) float32, those
    taken from it (taken[frame], the run each of the shot's frames is taken from) put in the
    shot's kept frames, by their frame in the shot, and told dump with their index in the job,
    the shot's first frame being `first`."""
    position = run.offset

    def keep(frames: npt.NDArray[np.float32]) -> None:
        nonlocal position
        for frame in frames:
            if taken[position] == run.offset:
                if dump is not None:
                    dump(first + position, frame)
                put(position, frame)
            position += 1

    return keep


def _interleaved(
    run: Run,
    taken: Sequence[int],
    store: "Kept",
    write: Write,
) -> Callable[[npt.NDArray[np.float32]], None]:
    """The last run's decode's write: of the frames it gives in order, those taken from it written,
    each after the kept frames before it (Kept.flush)."""
    position = run.offset

    def emit(frames: npt.NDArray[np.float32]) -> None:
        nonlocal position
        start = None  # the first of the frames given here written next, with those after it
        for offset in range(frames.shape[0]):
            frame = position + offset
            if taken[frame] != run.offset:
                if start is not None:
                    write(frames[start:offset])
                    start = None
                continue
            if store.pending(frame):
                if start is not None:
                    write(frames[start:offset])
                store.flush(frame, write)
                start = offset
            elif start is None:
                start = offset
        if start is not None:
            write(frames[start:])
        position += frames.shape[0]

    return emit


class Kept:
    """The frames a shot's runs but its last give (_decode_runs), `frames`, the shot's frames
    they are, in order, each (H', W', 3) as `shape`: in a file at path, allocated whole before the
    first is put, so that a full disk refuses there, mapped through the descriptor it was
    allocated by, removed when the block ends, on success and failure alike; a kill leaves it, for
    the resume to discard (resume.UNIT_FILE). Each frame kept as the uint16 values the writers
    quantise it to (writer.quantise), which they take as they are, so that no byte of a master
    changes against a single run's (DESIGN.md, Several runs of a shot, the ruling of 2026-10-20:
    "at the 16 bits the master is written from"): 12.4 MB a 1080p frame, 49.8 MB at 4K
    (job.kept_bytes; --plan says the most a shot's file takes, planner._runs_said; the job is
    refused at its start when its disk can't hold it). The float32 frames --dump-frames writes
    are dumped by the run that decodes them (_given), never from here. A file, never system
    memory: a tiled decode's scratch may hold the plan's allowance meanwhile (DESIGN.md, Memory
    planner). Its
    pages, mapped shared, count in the process's resident memory the decode's log line gives
    (_resident), as a tiled scratch's file does: page cache the system writes out and drops.
    PROVISIONAL (implementation, 2026-10-20): DESIGN.md says the runs keep only the frames they
    give, not where."""

    def __init__(self, path: Path, frames: Sequence[int], shape: tuple[int, int, int]) -> None:
        self.path, self.frames, self.shape = path, list(frames), shape
        self._slot = {frame: slot for slot, frame in enumerate(self.frames)}
        self._next = 0  # the slot of the next frame to write
        self._array: npt.NDArray[np.uint16] | None = None

    def __enter__(self) -> Self:
        if not self.frames:
            return self
        size = len(self.frames) * math.prod(self.shape) * KEPT_SAMPLE_BYTES
        make_directories(self.path.parent)
        # Derived data: a killed run's file, if the resume left it there, is replaced, unlinked
        # rather than truncated, as tiling.Scratch replaces its own.
        self.path.unlink(missing_ok=True)
        try:
            with self.path.open("x+b") as file:
                os.posix_fallocate(file.fileno(), 0, size)
                mapping = mmap.mmap(file.fileno(), size)
        except OSError as error:
            self.path.unlink(missing_ok=True)
            raise ScratchError(
                f"{self.path}: the frames of the shot's runs, {size} bytes: {error}"
            ) from error
        self._array = np.frombuffer(mapping, dtype=np.uint16).reshape(len(self.frames), *self.shape)
        return self

    def put(self, frame: int, values: npt.NDArray[np.float32]) -> None:
        """Keep the shot's frame `frame`, (H', W', 3) float32 in [0, 1], as the values the writers
        quantise it to."""
        assert self._array is not None
        self._array[self._slot[frame]] = quantise(values)

    def pending(self, frame: int) -> bool:
        """Whether a kept frame before `frame` is still to be written."""
        return self._next < len(self.frames) and self.frames[self._next] < frame

    def flush(self, frame: int, write: Write) -> None:
        """Write the kept frames before `frame` not written yet, in order, copies of the file's,
        uint16."""
        end = self._next
        while end < len(self.frames) and self.frames[end] < frame:
            end += 1
        if end > self._next:
            assert self._array is not None
            write(np.array(self._array[self._next : end]))
            self._next = end

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._array = None  # the mapping goes with its last array; copies alone were written
        if not self.frames:
            return
        try:
            self.path.unlink(missing_ok=True)
        except OSError as failure:
            logger.warning("%s: the frames of the shot's runs, not removed: %s", self.path, failure)
        if error is None and self._next != len(self.frames):
            raise RuntimeError(f"{len(self.frames) - self._next} kept frames never written")


def _scratch(units: Units, index: int, tiles: Tiles | None) -> Path | None:
    """Where a tiled pass of shot `index` keeps its scratch once it outgrows its allowance of
    system memory (DESIGN.md, Pipeline, per shot, step 3): the shot's own directory, among its
    units. None for an untiled pass, which keeps none."""
    return None if tiles is None else units.shot_directory(index)


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
    is opened at the first frame asked, and finished as the source's last frame is read, the
    source's end checked there (Reader.finish: ffmpeg done, with no error and no frame more than
    the first pass counted): by the read that gives the last frame, so before its caller records
    anything made of it, the last shot's latent or its input copy made again, and so before that
    shot's windows and the job's last segment. A source that fails the check leaves the last shot
    unread: the same command reads it again, and checks again. Until 2026-10-12 the end was
    checked once the job had run, its last segment recorded: the next run then said a
    directory's job finished, or assembled a .mkv path's file, short of the frame the source
    gave beyond its count (a finding of milestone 6's wiring).
    PROVISIONAL (implementation, 2026-10-12, a question for design, which decided "the check
    comes before the last segment is recorded"): before the last shot's latent, earlier than
    asked. Checked between that shot's windows and its segment's record, a failed run would keep
    the windows, and the next would decode them, never reading the source, so never checking it.
    Frames not asked for, those of shots a resume skips, are never decoded: the next frame asked
    is reached through the frame index, a reader opened there (DESIGN.md, Pause and resume). A run
    reading every shot from the source's first frame keeps one reader from the start, which seeks
    nowhere, its frames checked all the same (DESIGN.md, Input, has the check for a read of
    "frames n to m"; from the first frame too, accepted by design as built, 2026-10-10):
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
        next one, a later run of the shot read before it among them, through the frame index
        (DESIGN.md, Several runs of a shot: each run fed the shot from its frame)."""
        path = self._source.path
        if self._reader is None or self._position != shot.start:
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
            if self._position == self._source.frames:
                self._close(finish=True)  # the source's end, before what is made of these frames
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
                # The run's last word on it (Reader._take): what it means for a resume. Each frame
                # once, in order: with several runs a shot, each run reads its frames again.
                self._otherwise = sorted(set(self._otherwise))
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

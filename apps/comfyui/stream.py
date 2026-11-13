"""Upscale Video's pass over a clip, streamed (DESIGN.md, Repository and apps, the nodes' VIDEO
path): the command line's pass in ComfyUI's process, shot by shot, never the clip whole. Each shot
runs on the library's bricks in the command line's order (its run.py: encode_shot, sample_windows
with each window in a file, the latent let go, merge_windows, decode_shot), each of its runs at
enhanced and maximum the same; what the command line keeps in files the nodes keep in files too,
in a directory of the run's own in ComfyUI's temp folder (Workspace), removed at the run's end
however it ends: a tiled pass's scratch beyond the plan's allowance (FileScratch), the shot's
input as the encode read it, for the colour correction and the later runs (Copy), the windows,
and the frames of the earlier runs that wait for the frames before them (Order). Each frame
decoded is written to the node's file as it comes, its planes made on the card
(yuv.master_planes) and its FFV1 encode in a thread of its own (Writer), beside the card's work.

So the plan is the command line's (shot_plan.plan, its windows in files, in_memory False), and
the host holds at a time a slice of the source's frames, a window's output on its way to its file,
the writer's queue and what the command line's run holds of a shot (shot_plan.host_moments):
`held_bytes` counts the frames, which the plan takes off the memory the host has available. The
output's bits are the library's pass on the same RGB frames (clip.upscale), held by the nodes'
tests/test_stream.py."""

import gc
import logging
import mmap
import os
import queue
import shutil
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Generator, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import TracebackType
from typing import Literal, Protocol, Self, cast

import numpy as np
import numpy.typing as npt
import torch

from seedvr2x import model, sampling, yuv
from seedvr2x.model import COMPUTE_DTYPE, Models
from seedvr2x.passes import Tiles
from seedvr2x.shot_pass import decode_shot, encode_shot, finite, merge_windows, sample_windows
from seedvr2x.shotruns import QUALITY, kept_bytes, shot_runs, taken_from
from seedvr2x.shots import Shot, shot_layout
from seedvr2x.tilegrid import CHANNELS, scratch_bytes

from . import words
from .master import Master

logger = logging.getLogger("seedvr2x_comfyui.stream")

# The slices the writer's queue holds beside the one it encodes: the card waits for the encoder
# beyond them, so that the frames never pile up in memory however slow the encode is.
QUEUED = 2
# The frames of a decode's slice at the most: its first latent with the next, 5 frames, then 4
# (model.decode_slices, the VAE's slicing of one latent at a time).
SLICE_FRAMES = 5
# The frames of the source a read holds, at the most: the encode's slices (model.encode_slices),
# 5 frames, then 4.
READ_FRAMES = 5


def held_bytes(source: tuple[int, int], output: tuple[int, int]) -> int:
    """The bytes of frames the streamed pass holds in system memory at the most, beside what the
    plan counts of a shot (shot_plan.host_moments), for a source of (width, height) and an output
    of (width, height): a read's frames, float32 (READ_FRAMES), and the colour correction's
    slice of the shot's input read again beside it; a decode's slice, float32, the writer's queue
    and the slice it writes, float32, with that slice's planes (2 bytes a sample), and the
    preview's frame."""
    source_frame = source[0] * source[1] * 3
    output_frame = output[0] * output[1] * 3
    reading = 2 * READ_FRAMES * source_frame * 4
    writing = (QUEUED + 2) * SLICE_FRAMES * output_frame * 4 + SLICE_FRAMES * output_frame * 2
    return reading + writing + output_frame * 4


# The bytes a pixel of the node's file takes at the most, measured: FFV1 at level 3 of 16-bit 4:4:4
# YCbCr frames of gradients and noise came to 8.19 MB a 1080p frame, 3.95 bytes a pixel, smooth
# frames to a sixth of it, heavy texture to 1.2 times (impl-videostudy, 2026-11-12, synthetic
# frames; real content's rate unmeasured): the larger, for a disk that must hold the file.
FILE_BYTES_PER_PIXEL = 4.8


def disk_bytes(
    shots: Sequence[Shot],
    source: tuple[int, int],
    output: tuple[int, int],
    padded: tuple[int, int],
    quality: str,
    again: bool,
    passes: "Passes",
) -> tuple[int, int]:
    """What the run writes in ComfyUI's temp folder, in bytes: (the node's file, at
    FILE_BYTES_PER_PIXEL; the run's working files at their most, a shot's: its input copy where
    it is read again (`again`), the frames its earlier runs keep (shotruns.kept_bytes, the
    command line's figure), and a tiled pass's scratch beyond the plan's allowance, one pass's at
    a time), for shots of a source of (width, height) to an output of (width, height), the
    output's frames padded to (height, width). The windows' files, half a byte a padded pixel and
    a latent, are left out: under a hundredth of the rest."""
    frames = sum(each.frames for each in shots)
    file = round(frames * output[0] * output[1] * FILE_BYTES_PER_PIXEL)
    level = QUALITY[quality]
    working = 0
    for each in shots:
        copy = each.frames * source[0] * source[1] * 3 * 2 if again else 0
        kept = kept_bytes(each, level, (output[1], output[0]))
        tiled = passes.encode_tiles is not None or passes.decode_tiles is not None
        scratch = max(0, scratch_bytes(each.frames, padded) - passes.allowance) if tiled else 0
        working = max(working, copy + kept + scratch)
    return file, working


# ---- The run's files -----------------------------------------------------------------------------


class Workspace:
    """A run's directory `root`, run-<id> in `folder` (seedvr2x's in ComfyUI's temp folder), holding
    what the run keeps in files while it runs; removed whole when the block ends, on success and
    failure alike. A file Windows won't remove while it is still mapped is left for the next run's
    start, which removes every leftover run directory: ComfyUI runs one prompt at a time, so no
    other run's is there (ComfyUI empties its temp folder as it starts and ends too)."""

    def __init__(self, folder: Path) -> None:
        self.parent = folder
        self.root = folder / f"run-{uuid.uuid4().hex}"

    def __enter__(self) -> Self:
        if self.parent.is_dir():
            for left in self.parent.glob("run-*"):
                shutil.rmtree(left, ignore_errors=True)
        self.root.mkdir(parents=True)
        return self

    def path(self, name: str) -> Path:
        return self.root / name

    def __exit__(self, *_: object) -> None:
        remove_pending()
        try:
            shutil.rmtree(self.root)
        except OSError as error:
            logger.debug("%s: not removed yet, at the next run: %s", self.root, error)


# Scratch files Windows refused to remove while they were still mapped: a tiled pass keeps its
# store's tensor bound past the store's block (tiling.py's `with ... as inputs`), so the mapping
# outlives the removal, which Linux makes under it and Windows refuses (WinError 32 or 5). Removed
# at each later store's start, at each shot's end and with the run's directory, once unmapped (the
# command line's scratch.py, _PENDING, the same: impl-win-3's review, 2026-11-09; impl-video-2's
# review found the nodes' first draft left every one until the run's end).
_PENDING: list[Path] = []


def remove_pending() -> None:
    """Remove the scratch files whose removal was refused while they were mapped, those whose
    mapping has gone since: a collection first, a reference cycle being able to hold the last
    tensor on one. Nothing pending (Linux always): no work."""
    if not _PENDING:
        return
    gc.collect()
    for path in list(_PENDING):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            continue  # still mapped: tried again later
        _PENDING.remove(path)


class FileScratch:
    """A tiled pass's store (tiling.Store), the command line's (its scratch.py): system memory up to
    the plan's allowance (Plan.scratch_allowance), else a file of the run's directory mapped
    shared, its room reserved before it is written (os.posix_fallocate on Linux; on Windows, the
    mapping sets the file's end, NTFS allocating it), so that a full disk refuses there and never
    as a fault on a page of the mapping; removed as the block ends."""

    def __init__(self, workspace: Workspace, allowance: int) -> None:
        self.workspace, self.allowance = workspace, allowance
        self._open = False

    def where(self, frames: int, padded: tuple[int, int]) -> str:
        if scratch_bytes(frames, padded) <= self.allowance:
            return "system memory"
        return f"a file of {self.workspace.root}"

    @contextmanager
    def store(self, frames: int, padded: tuple[int, int]) -> Generator[torch.Tensor]:
        if self._open:
            raise RuntimeError("a scratch holds one store at a time")
        remove_pending()
        size = scratch_bytes(frames, padded)
        shape = (CHANNELS, frames, *padded)
        self._open = True
        try:
            if size <= self.allowance:
                yield torch.zeros(shape, dtype=COMPUTE_DTYPE)
                return
            path = self.workspace.path(f"scratch-{uuid.uuid4().hex}")
            with path.open("x+b") as file:
                if sys.platform != "win32":
                    os.posix_fallocate(file.fileno(), 0, size)
                mapping = mmap.mmap(file.fileno(), size)
            try:
                yield torch.frombuffer(mapping, dtype=COMPUTE_DTYPE).view(shape)
            finally:
                del mapping  # the store's tensors hold it: it goes with the last of them
                try:
                    path.unlink()
                except OSError as error:
                    _PENDING.append(path)
                    # Its text, not the error: the error's traceback holds this frame.
                    logger.debug("%s: still mapped, removed once it is not: %s", path, str(error))
        finally:
            self._open = False


class Copy:
    """A shot's input as the encode read it, in a file of the run's directory: each frame's 16-bit
    codes (H, W, 3) uint16, written as the frames are read and read back by the colour correction
    and by the shot's later runs (DESIGN.md, Colour correction; Several runs of a shot), as the
    command line reads its shot's input copy. Exact: the frames read are 16-bit codes over 65535
    (yuv.to_rgb), which the codes give back bit for bit. Written through the file, never a
    mapping, so that a full disk is an error and not a fault."""

    def __init__(self, path: Path, size: tuple[int, int]) -> None:
        self.path, self.size = path, size
        self.frames = 0
        self._file = path.open("x+b")

    @property
    def frame_bytes(self) -> int:
        return self.size[0] * self.size[1] * 3 * 2

    def append(self, frames: npt.NDArray[np.float32]) -> None:
        codes = np.rint(frames * np.float32(65535)).astype("<u2")
        self._file.seek(self.frames * self.frame_bytes)
        self._file.write(codes.tobytes())
        self.frames += frames.shape[0]

    def reader(self, start: int) -> Callable[[int], npt.NDArray[np.float32]]:
        """read(n), the copy's next n frames from frame `start`, (n, H, W, 3) float32 in [0, 1]."""
        position = start
        width, height = self.size

        def read(n: int) -> npt.NDArray[np.float32]:
            nonlocal position
            if position + n > self.frames:
                raise ValueError(f"frames {position} to {position + n} of a copy of {self.frames}")
            self._file.seek(position * self.frame_bytes)
            data = self._file.read(n * self.frame_bytes)
            codes = np.frombuffer(data, dtype="<u2").reshape(n, height, width, 3)
            position += n
            return codes.astype(np.float32) / np.float32(65535)

        return read

    def close(self) -> None:
        self._file.close()


# ---- A shot's frames -----------------------------------------------------------------------------


class ShotFrames(Protocol):
    """A shot's frames for its pass: read for its first run's encode, from the source; read again
    from `start` for a later run's encode and for the colour correction's reference."""

    @property
    def size(self) -> tuple[int, int]: ...

    def first(self) -> Callable[[int], npt.NDArray[np.float32]]: ...

    def again(self, start: int) -> Callable[[int], npt.NDArray[np.float32]]: ...


class InMemory:
    """A shot of frames ComfyUI holds already (a VIDEO with no file of its own): (T, H, W, 3)
    float32 in [0, 1], read as views and read again the same way."""

    def __init__(self, frames: npt.NDArray[np.float32]) -> None:
        self.frames = frames

    @property
    def size(self) -> tuple[int, int]:
        return self.frames.shape[2], self.frames.shape[1]

    def first(self) -> Callable[[int], npt.NDArray[np.float32]]:
        return self.again(0)

    def again(self, start: int) -> Callable[[int], npt.NDArray[np.float32]]:
        position = start

        def read(n: int) -> npt.NDArray[np.float32]:
            nonlocal position
            part = self.frames[position : position + n]
            position += n
            return part

        return read


class FromFile:
    """A shot of a file's frames: its first run reads them from the file's forward decode
    (`read`, video.Decoded.next_frames), each copied as it is read where the shot reads them
    again (Copy), whose frames the later runs and the colour correction read."""

    def __init__(
        self,
        read: Callable[[int], npt.NDArray[np.float32]],
        size: tuple[int, int],
        copy: Copy | None,
    ) -> None:
        self._read, self._size, self.copy = read, size, copy

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    def first(self) -> Callable[[int], npt.NDArray[np.float32]]:
        def read(n: int) -> npt.NDArray[np.float32]:
            frames = self._read(n)
            if self.copy is not None:
                self.copy.append(frames)
            return frames

        return read

    def again(self, start: int) -> Callable[[int], npt.NDArray[np.float32]]:
        if self.copy is None:
            raise RuntimeError("a shot read again with no copy of its input")
        return self.copy.reader(start)


@dataclass(frozen=True)
class Reference:
    """split's reference (shot_pass.Reference): the shot's frames read again from their start,
    the run's first frame being `first`."""

    frames: ShotFrames
    first: int

    @property
    def width(self) -> int:
        return self.frames.size[0]

    @property
    def height(self) -> int:
        return self.frames.size[1]

    @contextmanager
    def reader(self, count: int) -> Generator[Callable[[int], npt.NDArray[np.float32]]]:
        yield self.frames.again(0)


# ---- The writer ----------------------------------------------------------------------------------


class Writer:
    """The master written in a thread of its own: frames put in a queue of QUEUED slices, which a
    put waits on when full, so that the card waits for the encoder rather than the frames piling
    up; each slice's planes made there, on the CPU (planes), and encoded (master.Master.
    write_planes), beside the card's work, which holds nothing of them (the command line makes its
    masters' planes in ffmpeg, on the CPU, too); an error of the thread raised by the next put or
    by finish, worded (words.not_written). A slice is given as its RGB frames (put) or as the
    planes made of it already (put_planes, the frames an earlier run kept: Order). `seconds` is the
    thread's time making the planes, encoding and writing, `frames` the frames it wrote."""

    def __init__(self, master: Master) -> None:
        self.master = master
        self.seconds = 0.0
        self.frames = 0
        self._queue: queue.Queue[npt.NDArray[np.float32] | npt.NDArray[np.uint16] | None] = (
            queue.Queue(QUEUED)
        )
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._work, name="seedvr2x writer", daemon=True)
        self._thread.start()

    def _work(self) -> None:
        try:
            while True:
                given = self._queue.get()
                if given is None or self._stop.is_set():
                    return
                began = time.monotonic()
                made = (
                    planes(cast(npt.NDArray[np.float32], given))
                    if given.dtype == np.float32
                    else given
                )
                self.master.write_planes(cast(npt.NDArray[np.uint16], made))
                self.seconds += time.monotonic() - began
                self.frames += made.shape[0]
        except BaseException as error:
            self._error = error

    def _raise(self) -> None:
        if self._error is not None:
            raise words.not_written(self.master.path, self._error) from self._error

    def put(self, frames: npt.NDArray[np.float32]) -> None:
        """RGB frames, (n, H, W, 3) float32 in [0, 1], appended to the file."""
        self._put(frames)

    def put_planes(self, made: npt.NDArray[np.uint16]) -> None:
        """Frames given as their planes, (n, 3, H, W) uint16 (planes), appended to the file."""
        self._put(made)

    def _put(self, given: npt.NDArray[np.float32] | npt.NDArray[np.uint16]) -> None:
        while True:
            self._raise()
            if not self._thread.is_alive():
                raise RuntimeError("the file's writer has stopped")
            try:
                self._queue.put(given, timeout=0.1)
                return
            except queue.Full:
                continue

    def finish(self) -> None:
        """Every frame put written, then the thread ended; its error raised."""
        self._put_end()
        self._thread.join()
        self._raise()

    def _put_end(self) -> None:
        while self._thread.is_alive():
            try:
                self._queue.put(None, timeout=0.1)
                return
            except queue.Full:
                continue

    def abort(self) -> None:
        """The thread stopped after the slice it writes, what is queued dropped."""
        self._stop.set()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._put_end()
        self._thread.join()


def planes(frames: npt.NDArray[np.float32]) -> npt.NDArray[np.uint16]:
    """The master's planes of frames (n, H, W, 3) float32 in [0, 1]: (n, 3, H, W) uint16, made on
    the CPU (yuv.master_planes)."""
    return yuv.master_planes(torch.as_tensor(np.ascontiguousarray(frames))).numpy()


class Order:
    """A shot's frames written in their order (the command line's run.py, Kept): a frame the run
    being decoded gives is put to the writer at once where every frame before it is; else it
    waits, as its planes (planes: 2 bytes a sample, where the frame is 4), in a file of the run's
    directory until those are, written there through the file. One frame a slot, the shot's at the
    most; the file removed by close."""

    def __init__(self, path: Path, count: int, size: tuple[int, int], writer: "Writer") -> None:
        self.path, self.count, self.writer = path, count, writer
        self.frame_bytes = size[0] * size[1] * 3 * 2
        self.shape = (3, size[1], size[0])
        self.next: int = 0
        self._waiting: set[int] = set()
        self._file = None if count < 2 else path.open("x+b")

    def give(self, first: int, frames: npt.NDArray[np.float32]) -> None:
        """Frames (n, H', W', 3) float32, the shot's `first` and those after it, one after the
        other, each a frame the run decoding them keeps."""
        if first == self.next:
            self.writer.put(frames)
            self.next += frames.shape[0]
            self._flush()
            return
        made = planes(frames)
        assert self._file is not None
        for index, values in enumerate(made):
            self._file.seek((first + index) * self.frame_bytes)
            self._file.write(values.tobytes())
            self._waiting.add(first + index)

    def _flush(self) -> None:
        while self.next in self._waiting:
            assert self._file is not None
            self._file.seek(self.next * self.frame_bytes)
            waited = np.frombuffer(self._file.read(self.frame_bytes), dtype=np.uint16)
            self.writer.put_planes(waited.reshape(1, *self.shape).copy())
            self._waiting.discard(self.next)
            self.next += 1

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self.path.unlink(missing_ok=True)


# ---- A shot's pass -------------------------------------------------------------------------------

Stage = Literal["run", "encode", "window", "decode"]
Step = Callable[[Stage, int, int], None]
Tile = Callable[[Literal["encode", "decode"], int, int], None]
Preview = Callable[[int, int, npt.NDArray[np.float32] | None], None]


@dataclass(frozen=True)
class Passes:
    """The plan's choices for the passes (passes.Plan's): the window cap in latents, the tiles of
    each VAE pass, the decoder's tail a frame at a time, and the scratch's allowance in bytes."""

    window: int | None
    encode_tiles: Tiles | None
    decode_tiles: Tiles | None
    tail_frames: bool
    allowance: int


@torch.no_grad()
def upscale_shot(
    models: Models,
    frames: ShotFrames,
    shot: Shot,
    target: tuple[int, int],
    seed: int,
    passes: Passes,
    *,
    correct: bool,
    quality: str,
    workspace: Workspace,
    writer: Writer,
    step: Step | None = None,
    tile: Tile | None = None,
    preview: Preview | None = None,
) -> None:
    """Shot `shot` of the clip (its frames in the clip, which seed it as the command line seeds a
    shot: shots.Shot.seed) through the library's pass at `quality`, as clip.upscale runs it, with
    the plan's `passes`: the frames given to `writer` in their order (Order), a run's as its decode
    gives them, a later run's taken from it (shotruns.taken_from). `frames` gives the shot's frames
    (ShotFrames). `step` is told as clip.upscale tells it, `tile` each tile of a tiled pass, and
    `preview` (done, count, the newest frame the run keeps so far, (H', W', 3) float32: its first is
    its own) after each slice decoded; each may raise, BaseException included, to stop the shot,
    which leaves nothing on the device but what the models hold."""
    count = shot.frames
    level = QUALITY[quality]
    runs = shot_runs(Shot(0, count), level)
    taken = taken_from(count, level)
    shot_seed = shot.seed(seed)
    order = Order(workspace.path(f"order-{shot.start}"), count, _output(target), writer)
    try:
        with model.marking(models.device):
            for number, each in enumerate(runs, 1):
                if step is not None and len(runs) > 1:
                    step("run", number, len(runs))
                read = frames.first() if number == 1 else frames.again(each.offset)
                _run(
                    *(models, read, frames, each.offset, count, target, shot_seed, passes),
                    *(correct, workspace, step, tile, preview, shot.start),
                    taken,
                    order,
                )
    except BaseException as error:
        traceback_ = error.__traceback__
        model.unless_failed(models.device, lambda: traceback.clear_frames(traceback_))
        raise
    finally:
        order.close()
        remove_pending()
    if order.next != count:
        raise RuntimeError(f"{count - order.next} frames of the shot never written")


def _output(target: tuple[int, int]) -> tuple[int, int]:
    from seedvr2x.frames import output_size

    height, width = output_size(target)
    return width, height


def _run(
    models: Models,
    read: Callable[[int], npt.NDArray[np.float32]],
    frames: ShotFrames,
    offset: int,
    shot_frames: int,
    target: tuple[int, int],
    seed: int,
    passes: Passes,
    correct: bool,
    workspace: Workspace,
    step: Step | None,
    tile: Tile | None,
    preview: Preview | None,
    start: int,
    taken: list[int],
    order: Order,
) -> None:
    """The shot's run from its frame `offset` (clip._run's steps, the windows in files as the
    command line keeps them)."""
    count = shot_frames - offset
    layout = shot_layout(count, passes.window)
    reference = Reference(frames, offset) if correct else None
    models.enter("encode")
    latent = encode_shot(
        models,
        read,
        count,
        target,
        seed,
        tiles=passes.encode_tiles,
        store=FileScratch(workspace, passes.allowance),
        tile_step=None if tile is None else partial(tile, "encode"),
    )
    finite(latent, "the clip's encode")
    if step is not None:
        step("encode", 1, 1)
    first, last = start + offset, start + shot_frames - 1
    paths = _windows(models, latent, layout, seed, step, (first, last), workspace)
    # The latent and the noise let go before the merge, as the command line's run does
    # (shot_plan.host_moments: its windows read back from their files).
    del latent
    windows = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    for path in paths:
        path.unlink()
    merged = merge_windows(windows, layout)
    del windows
    models.enter("decode")
    written = 0
    newest: npt.NDArray[np.float32] | None = None

    def write(part: npt.NDArray[np.float32]) -> None:
        nonlocal written, newest
        place = offset + written
        mine = [taken[place + i] == offset for i in range(part.shape[0])]
        # The run's frames in runs of consecutive ones it keeps, each given in its order.
        index = 0
        while index < part.shape[0]:
            if not mine[index]:
                index += 1
                continue
            end = index
            while end < part.shape[0] and mine[end]:
                end += 1
            order.give(place + index, part[index:end])
            newest = part[end - 1].copy()  # a copy: the slice itself goes once written
            index = end
        written += part.shape[0]
        if step is not None:
            step("decode", written, count)
        if preview is not None:
            preview(written, count, newest)

    decode_shot(
        models,
        merged,
        count,
        target,
        write,
        "the clip",
        first,
        reference,
        tiles=passes.decode_tiles,
        store=FileScratch(workspace, passes.allowance),
        tile_step=None if tile is None else partial(tile, "decode"),
        tail_frames=passes.tail_frames,
    )


def _windows(
    models: Models,
    latent: torch.Tensor,
    layout: list[tuple[int, int]],
    seed: int,
    step: Step | None,
    frames: tuple[int, int],
    workspace: Workspace,
) -> list[Path]:
    """Each window sampled (shot_pass.sample_windows), the DiT's phase entered before it, checked,
    brought to system memory with its memory layout (sampling.moved) and saved to a file of the
    run's directory by torch.save, which keeps it (the command line's units.save_window): the
    windows' files, in order."""
    sampled: Iterator[torch.Tensor] = sample_windows(models, latent, layout, seed)
    paths: list[Path] = []
    try:
        for number in range(len(layout)):
            models.enter("dit")
            output = next(sampled)
            what = (
                f"the clip's window {number + 1}/{len(layout)} (frames {frames[0]} to {frames[1]})"
            )
            finite(output, what)
            moved = sampling.moved(output, sampling.CPU, what)
            del output
            path = workspace.path(f"window-{uuid.uuid4().hex}.pt")
            torch.save(moved, path)
            paths.append(path)
            del moved
            if step is not None:
                step("window", number + 1, len(layout))
    finally:
        if isinstance(sampled, Generator):
            model.unless_failed(models.device, sampled.close)
    return paths


class Closing:
    """Master's context, the file removed when the run fails or its close does (a full disk at
    its end), after it is closed."""

    def __init__(self, master: Master) -> None:
        self.master = master

    def __enter__(self) -> Master:
        return self.master.__enter__()

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback_: TracebackType | None,
    ) -> None:
        try:
            self.master.__exit__(kind, error, traceback_)
        except BaseException:
            self._remove()
            raise
        if kind is not None:
            self._remove()

    def _remove(self) -> None:
        """The partial file removed; a removal that fails said, never raised over the error that
        ended the run."""
        try:
            self.master.path.unlink(missing_ok=True)
        except OSError as failure:
            logger.warning("the partial file %s, not removed: %s", self.master.path, failure)

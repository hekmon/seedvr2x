"""VAE tiling (DESIGN.md, Pipeline, per shot, steps 1 and 3): a shot encoded or decoded tile by
tile, when the untiled pass doesn't fit the card.

A tiled pass can't stream. The VAE is causal, and each tile keeps its own memory from the shot's
first slice to its last: the tiles run one after the other, each over every frame, as numz's do
over a batch, and a frame is whole only after the last tile. Every tile advancing together, slice
by slice, would hold every tile's causal memory at once, 8.05 GiB per megapixel decoded
(research/docs/decode-resume.md). So the frames wait in the shot's scratch (Scratch), in bfloat16:
the decode sums its weighted tiles there, and hands the frames on once the last tile is in, in the
slices a streamed decode gives; the encode keeps its transformed input there, exact, and crops
each tile from it, with no seeking back into the source.

The grid and the blend are numz's, bit for bit (v1): tiles laid from the top left corner, a stride
of the tile's side less the overlap, a cosine ramp across the overlap on the edges inside the
picture, in bfloat16; each tile weighted on the device, summed on the CPU, and the sums divided on
the device by the summed weights, as numz's CLI runs them with its default tensor_offload_device,
the CPU (inference_cli.py:1417). Adapted from numz's tiled_encode and tiled_decode
(src/models/video_vae_v3/modules/attn_video_vae.py:1302, 1470 at 4490bd1), Apache-2.0 (NOTICE).
Line references are to numz at 4490bd1.

It departs from numz in what a shot of any length takes, each operation staying numz's, element by
element, on numz's device, so that no value changes:
- numz sums a batch's tiles in one tensor on the CPU and moves it whole to the GPU to divide it;
  the sums wait in the scratch here, and are divided slice by slice as they are handed on.
- numz crops each tile from the batch's input, whole on the GPU; the input waits in the scratch
  here, and a tile's crop goes to the device slice by slice.
- numz's tiles run over a batch; each runs over the whole shot here.
- numz scales a batch's latents on the GPU and crops each tile from them there; the shot's
  latents stay in system memory here, and a tile's crop of each slice's goes to the device as its
  turn comes (model.decode_sliced).
- numz sums a tile's moments whole, the mean and the log-variance, and takes the mean of the
  quotient; only the mean is weighted and summed here, slice by slice, in system memory, which is
  the same sum for the half that is kept, and the sums are divided, and the latent made, slice by
  slice on the device.
So nothing on the device grows with the shot (DESIGN.md, Memory planner).
"""

import contextlib
import logging
import mmap
import os
from collections.abc import Generator, Iterable, Iterator
from pathlib import Path

import torch
from torch import Tensor

from seedvr2x.media.ffmpeg import MediaError
from seedvr2x.media.files import make_directories
from seedvr2x.runtime import model
from seedvr2x.runtime.model import COMPUTE_DTYPE, Models
from seedvr2x.runtime.plan import Tiles

# The grid's arithmetic, kept without torch for the planner to import (runtime/tilegrid.py): its
# names are this module's too.
from seedvr2x.runtime.tilegrid import CHANNELS as CHANNELS
from seedvr2x.runtime.tilegrid import SCALE as SCALE
from seedvr2x.runtime.tilegrid import SCRATCH as SCRATCH
from seedvr2x.runtime.tilegrid import Tile as Tile
from seedvr2x.runtime.tilegrid import grid as grid
from seedvr2x.runtime.tilegrid import largest as largest
from seedvr2x.runtime.tilegrid import latent_size, latent_tiles, scratch_bytes

logger = logging.getLogger(__name__)

# The bytes of a scratch kept in system memory where its pass is given no allowance (Scratch), a
# larger one going to its file: for a test or a script that calls shot.encode_shot or
# shot.decode_shot without one. 8 GiB, which holds 675 frames at 1080p (padded to 1104x1920,
# 12,718,080 bytes each) and 170 at 4K (2192x3840, 50,503,680). A run gives each pass its plan's
# allowance, which the planner sets from the host's memory (planner.plan; DESIGN.md, Memory
# planner, Host RAM).
DEFAULT_ALLOWANCE = 8 * 2**30


class ScratchError(MediaError):
    """A scratch that can't be made: no directory to put its file in, no room for it there, or no
    mapping it; the message says which, with the bytes it takes.

    PROVISIONAL: a MediaError, so that the command line reports it as a file the run can't write,
    its message and exit status 1 (cli._run), the units made before it kept: a full disk is the
    user's to fix, and the same command then resumes. DESIGN.md doesn't say how it is reported."""


class Scratch:
    """Where a tiled shot's frames wait (DESIGN.md, Pipeline step 3): a store of (3, frames, H, W)
    bfloat16 on the CPU, zeros at first. In system memory when its bytes (needed) are within
    `allowance` (None: DEFAULT_ALLOWANCE), else a memory-mapped file in `directory`, the shot's
    unit directory (SCRATCH).

    The file is derived data, never recorded: it goes when its store is closed, on success and on
    failure alike, a mapping the system refuses included, and nothing stays open on it. A run
    killed leaves it, for the resume to discard."""

    def __init__(self, directory: Path | None, allowance: int | None = None) -> None:
        self.directory = directory
        self.allowance = DEFAULT_ALLOWANCE if allowance is None else allowance
        self._open = False

    @staticmethod
    def needed(frames: int, padded: tuple[int, int]) -> int:
        """The bytes of a store of `frames` frames padded to (height, width) pixels: 12.7 MB per
        1080p frame, padded to 1104x1920, 50.5 MB per 4K frame, padded to 2192x3840
        (tilegrid.scratch_bytes, there without torch for the planner to count with)."""
        return scratch_bytes(frames, padded)

    def where(self, frames: int, padded: tuple[int, int]) -> str:
        """Where a store of that size is while it is open, for the logs: system memory, or its
        file's path."""
        if self.directory is None or self.needed(frames, padded) <= self.allowance:
            return "system memory"
        return str(self.directory / SCRATCH)

    @contextlib.contextmanager
    def store(self, frames: int, padded: tuple[int, int]) -> Generator[Tensor]:
        """A store of `frames` frames padded to (height, width) pixels, open while the with block
        runs: (3, frames, height, width) bfloat16 on the CPU, zeros. One at a time.

        ScratchError when it takes a file and there is no directory, no room for it there, or no
        mapping it. The file is removed on leaving the block; its mapping goes with the last
        tensor on it, so nothing taken from the store is kept past the block but copies."""
        if self._open:
            raise RuntimeError("a scratch holds one store at a time")
        size = self.needed(frames, padded)
        shape = (CHANNELS, frames, *padded)
        self._open = True
        try:
            if size <= self.allowance:
                yield torch.zeros(shape, dtype=COMPUTE_DTYPE)
                return
            with self._file(size) as flat:
                yield flat.view(shape)
        finally:
            self._open = False

    @contextlib.contextmanager
    def _file(self, size: int) -> Generator[Tensor]:
        """The scratch's file, new, `size` bytes of zeros, mapped (_mapped): a flat tensor, the
        file there while the with block runs."""
        if self.directory is None:
            # PROVISIONAL: the planner refuses such a job before the run (DESIGN.md, Memory
            # planner); DESIGN.md doesn't say what the run itself does.
            raise ScratchError(
                f"a scratch of {size} bytes, over the {self.allowance} allowed in system memory,"
                " and no directory for its file"
            )
        path = self.directory / SCRATCH
        made = False
        try:
            try:
                make_directories(self.directory)
                # PROVISIONAL: a killed run's file, if the resume left it there, is replaced, not
                # refused: it is derived data. Unlinked, never truncated, so that a process still
                # mapping it keeps its own.
                path.unlink(missing_ok=True)
                # Read and written: the mapping is of this descriptor, and writes.
                with path.open("x+b") as file:
                    made = True
                    # Allocated whole: a full disk fails here, not later as a SIGBUS on a page of
                    # the mapping.
                    os.posix_fallocate(file.fileno(), 0, size)
                    # Mapped while the file allocated is still open, through its descriptor.
                    flat = _mapped(file.fileno(), path, size)
            except OSError as error:
                raise ScratchError(f"{path}: the shot's scratch, {size} bytes: {error}") from error
            yield flat
        finally:
            # Whatever ended the block, an allocation interrupted included. A file that stays
            # changes no result: said, and the run goes on.
            try:
                if made:
                    path.unlink(missing_ok=True)
            except OSError as error:
                logger.warning("%s: the shot's scratch, not removed: %s", path, error)


def _mapped(descriptor: int, path: Path, size: int) -> Tensor:
    """The file open as `descriptor`, read and written, `size` bytes, as a flat tensor in
    COMPUTE_DTYPE: a shared mapping, so that what is written through the tensor is the file's,
    which the system writes out and drops from memory as it needs to.

    Mapped through the descriptor itself (mmap), which is the file that was allocated, whatever
    its name has become: by its name, path, a link put there since the allocation would be mapped
    in its place, and another file written (tests/test_tiling.py). path is for the message.

    The mapping is never closed here: the tensor's storage keeps the mapping object, which goes
    with the last tensor on it, as does the duplicate of the descriptor it holds, the descriptor
    given closed or not. Closed under a tensor, it would leave the tensor on memory that is no
    longer mapped, without an error. A mapping the system refuses is a ScratchError, and leaves
    nothing open on the file: torch.from_file, which maps a name, kept its own descriptor on it
    then."""
    try:
        # Shared, read and written: mmap's defaults.
        mapping = mmap.mmap(descriptor, size)
    except OSError as error:
        raise ScratchError(
            f"{path}: the shot's scratch, {size} bytes, not mapped: {error}"
        ) from error
    return torch.frombuffer(mapping, dtype=COMPUTE_DTYPE)


@torch.no_grad()
def encode_tiled(
    models: Models, slices: Iterable[Tensor], frames: int, tiles: Tiles, scratch: Scratch
) -> Tensor:
    """VAE latent of a shot of `frames` frames (4n + 1) fed slice by slice, as model.encode_stream
    takes them (each (C, t, H, W) in [-1, 1], COMPUTE_DTYPE on the device, t following
    model.encode_slices), encoded tile by tile (grid): (T', h, w, 16) in channel-major memory, in
    system memory, as model.encode_stream gives it; numz's tiled encode of the shot, bit for bit.
    A grid of one tile gives model.encode_stream's latent, the scratch untouched.

    The slices are first kept in the scratch as they come, on the CPU. Then each tile's crop of
    them goes through the encoder in the same slices, one pass with its own causal memory
    (model.encode_sliced); the mean of each slice's moments (model.mode), the half the latent is
    made from, is weighted on the device, moved to the CPU and added to its latents' place in the
    sums, and the tile's weights to their sum, on the CPU (attn_video_vae.py:1426-1457). After
    the last tile each slice of the sums (model.latent_slices) goes to the device, is divided
    there by the summed weights (:1459-1463) and comes back, and the latent is made from them
    (model.to_latent), slice by slice too: nothing on the device grows with the shot. The sums
    are latents, small, the size of the shot's latent: only the input takes the scratch, which is
    closed before this returns.

    ValueError when the slices don't follow model.encode_slices, differ in size, or aren't in
    COMPUTE_DTYPE, the scratch's and the VAE's."""
    sizes = model.encode_slices(models, frames)
    slices = iter(slices)
    first = next(slices, None)
    if first is None:
        raise ValueError(f"0 slices, the VAE takes {sizes}")
    padded = (first.shape[2], first.shape[3])
    # Read for the picture's size, and handed back with the rest, no reference kept: the first
    # slice leaves the device when the pass that takes it lets it go, not when the slices end.
    slices = _handed_back(first, slices)
    del first
    if len(grid(latent_size(padded), tiles)) == 1:
        _untiled("encode", tiles, padded)
        return model.encode_stream(models, slices, frames)
    sums, count = _means(models, slices, sizes, padded, tiles, scratch)
    divisor = count.to(models.device).clamp(min=1e-6)
    start = 0
    for size in model.latent_slices(models, sums.shape[2]):
        # A copy on the CPU too, where to() alone gives the sums' own memory.
        mean = sums[:, :, start : start + size].to(models.device, copy=True).div_(divisor)
        model.synchronize(models.device)
        sums[:, :, start : start + size] = mean.to("cpu")
        start += size
    return model.to_latent(models, sums)


def _handed_back(first: Tensor, rest: Iterator[Tensor]) -> Iterator[Tensor]:
    """first, then rest: a shot's slices again, once the first was read ahead of them. The first
    is dropped here when the next is asked for, where itertools.chain would keep it until rest
    ends."""
    yield first
    del first
    yield from rest


def _means(
    models: Models,
    slices: Iterable[Tensor],
    sizes: list[int],
    padded: tuple[int, int],
    tiles: Tiles,
    scratch: Scratch,
) -> tuple[Tensor, Tensor]:
    """The tiles' weighted means (model.mode) of a shot fed in slices of `sizes` frames, summed,
    (1, 16, T', h, w), and the summed weights, (h, w), on the CPU, in the VAE's dtype
    (encode_tiled). The scratch's store is open only here: nothing returned is on it."""
    frames = sum(sizes)
    height, width = latent_size(padded)
    overlap = latent_tiles(tiles)[1]
    latents = 1 + (frames - 1) // models.runner.vae.temporal_downsample_factor
    sums: Tensor | None = None
    count = torch.zeros((height, width), dtype=COMPUTE_DTYPE)
    layout = grid((height, width), tiles)
    with scratch.store(frames, padded) as inputs:
        _tiled("encode", layout, tiles, frames, padded, scratch)
        _keep(models, inputs, slices, sizes)
        # The ramp is over the overlap in latents, on the input's device in its dtype
        # (attn_video_vae.py:1353-1360), which the moments share (_encoded).
        like = torch.empty(0, device=models.device, dtype=inputs.dtype)
        ramp = _ramp(overlap, like)
        for tile in layout:
            row_weights = _weights(
                like, tile.rows, overlap, ramp, tile.top > 0, tile.bottom < height
            )
            column_weights = _weights(
                like, tile.columns, overlap, ramp, tile.left > 0, tile.right < width
            )
            place = (slice(tile.top, tile.bottom), slice(tile.left, tile.right))
            start = 0
            # Closed whatever ends the tile, which clears its causal memory at once. A slice's
            # mean leaves the device as it comes: a tile's moments, whole, would grow there with
            # the shot, 1 MiB a latent under 1024-px tiles, where numz's stay for a batch.
            crops = _crops(models, inputs, tile, sizes)
            with contextlib.closing(_encoded(models, crops, tile, frames, latents)) as passes:
                for moments in passes:
                    # numz weights and sums the moments whole, the log-variance with the mean
                    # (:1426-1457), and keeps the mean of the quotient (src/core/infer.py:183-184):
                    # each channel is weighted, summed and divided alone, so the mean's are the
                    # same.
                    mean = model.mode(moments)
                    if sums is None:
                        # Sized by the first moments, as numz's (:1406-1419).
                        shape = (*mean.shape[:2], latents, height, width)
                        sums = torch.zeros(shape, dtype=mean.dtype)
                    mean.mul_(row_weights.view(-1, 1)).mul_(column_weights)
                    model.synchronize(models.device)
                    end = start + mean.shape[2]
                    sums[:, :, start:end, *place] += mean.to("cpu")
                    start = end
            model.synchronize(models.device)
            count[place].addcmul_(
                row_weights.to("cpu").view(-1, 1), column_weights.to("cpu").view(1, -1)
            )
    assert sums is not None, "a grid has a tile"
    return sums, count


def _encoded(
    models: Models, crops: Iterable[Tensor], tile: Tile, frames: int, latents: int
) -> Generator[Tensor]:
    """A tile's crop of the input encoded slice by slice (model.encode_sliced), each slice's
    moments checked for their place in the sums: (1, c, t', the tile's rows, its columns) in
    COMPUTE_DTYPE, `latents` latents in all. Moments of another shape could be broadcast into
    that place, and in another dtype rounded; numz's own guard, a crop of the tile's moments to
    their place (attn_video_vae.py:1421-1424), changes nothing on padded frames, multiples of
    SCALE, and is this check here."""
    given = 0
    for moments in model.encode_sliced(models, crops, frames):
        expected = (tile.rows, tile.columns)
        if (
            moments.dim() != 5
            or moments.shape[0] != 1
            or tuple(moments.shape[3:]) != expected
            or moments.dtype != COMPUTE_DTYPE
            or given + moments.shape[2] > latents
        ):
            raise RuntimeError(
                f"latents {given} on encoded as {tuple(moments.shape)} {moments.dtype}, not"
                f" (1, c, t, {expected[0]}, {expected[1]}) {COMPUTE_DTYPE} within {latents}"
            )
        yield moments
        given += moments.shape[2]
    if given != latents:
        raise RuntimeError(f"{given} latents encoded, not {latents}")


def _keep(models: Models, inputs: Tensor, slices: Iterable[Tensor], sizes: list[int]) -> None:
    """Keep the encoder's slices in the store `inputs` as they come, on the CPU, each checked as
    model.encode_stream checks it, and for the store's shape and dtype: a slice in another dtype
    would be rounded to the store's."""
    start = kept = 0
    for index, x in enumerate(slices):
        if index >= len(sizes) or x.shape[1] != sizes[index]:
            raise ValueError(f"slice {index} of {x.shape[1]} frames, the VAE takes {sizes}")
        expected = (inputs.shape[0], sizes[index], *inputs.shape[2:])
        if x.shape != expected or x.dtype != inputs.dtype:
            raise ValueError(
                f"slice {index} is {tuple(x.shape)} {x.dtype}, not {expected} {inputs.dtype}"
            )
        model.synchronize(models.device)
        inputs[:, start : start + sizes[index]] = x.to("cpu")
        start += sizes[index]
        kept += 1
    if kept != len(sizes):
        raise ValueError(f"{kept} slices, the VAE takes {sizes}")


def _crops(models: Models, inputs: Tensor, tile: Tile, sizes: list[int]) -> Iterator[Tensor]:
    """The tile's crop of the store `inputs`, on the device, in the slices the input came in
    (attn_video_vae.py:1373-1377, 1391)."""
    rows = slice(tile.top * SCALE, tile.bottom * SCALE)
    columns = slice(tile.left * SCALE, tile.right * SCALE)
    start = 0
    for size in sizes:
        yield inputs[:, start : start + size, rows, columns].to(models.device)
        start += size


@torch.no_grad()
def decode_tiled(
    models: Models, latent: Tensor, tiles: Tiles, scratch: Scratch
) -> Generator[Tensor]:
    """VAE decode of latent (T', h, w, 16), as model.decode_stream takes it, a shot's in system
    memory, tile by tile (grid): frames (C, t, H, W) in [-1, 1], unclamped, on the device, in the
    slices model.decode_stream gives (t following model.decode_slices), handed on once the last
    tile is decoded; numz's tiled decode of the shot, bit for bit. A grid of one tile gives
    model.decode_stream's slices, the scratch untouched.

    Each tile's crop of the latents is decoded slice by slice, one pass with its own causal
    memory (model.decode_sliced), the latent staying where it is, a slice's crop alone going to
    the device; each slice is weighted on the device, moved to the CPU and
    added to its frames' place in the scratch, and the tile's weights to their sum, on the CPU
    (attn_video_vae.py:1588-1619). After the last tile, each slice of the sums goes to the device
    and is divided there by the summed weights (:1621-1625).

    A generator: the scratch's store is open from the first tile until it ends, is closed or
    fails, and what it gives are copies, none on the store. ValueError unless the latent is in
    COMPUTE_DTYPE, the scratch's and the VAE's."""
    layout = grid((latent.shape[1], latent.shape[2]), tiles)
    if len(layout) == 1:
        _untiled("decode", tiles, (latent.shape[1] * SCALE, latent.shape[2] * SCALE))
        yield from model.decode_stream(models, latent)
        return
    model.decodable(models, latent)
    if latent.dtype != COMPUTE_DTYPE:
        raise ValueError(f"latent {latent.dtype}, a tiled decode takes {COMPUTE_DTYPE}")
    sizes = model.decode_slices(models, latent.shape[0])
    height, width = latent.shape[1], latent.shape[2]
    padded = (height * SCALE, width * SCALE)
    # The decode's ramp is over the overlap in output pixels, on the device in the dtype of the
    # decoder's input (:1516-1523), the latent's, which the decoded frames share.
    overlap = tiles.overlap
    like = torch.empty(0, device=models.device, dtype=latent.dtype)
    ramp = _ramp(overlap, like)
    count = torch.zeros(padded, dtype=COMPUTE_DTYPE)
    with scratch.store(sum(sizes), padded) as sums:
        _tiled("decode", layout, tiles, sum(sizes), padded, scratch)
        for tile in layout:
            rows = slice(tile.top * SCALE, tile.bottom * SCALE)
            columns = slice(tile.left * SCALE, tile.right * SCALE)
            row_weights = _weights(
                like, tile.rows * SCALE, overlap, ramp, tile.top > 0, tile.bottom < height
            )
            column_weights = _weights(
                like, tile.columns * SCALE, overlap, ramp, tile.left > 0, tile.right < width
            )
            start = 0
            # Closed whatever ends the tile, which clears its causal memory at once.
            with contextlib.closing(_decoded(models, latent, tile, sizes)) as slices:
                for decoded in slices:
                    decoded.mul_(row_weights.view(-1, 1)).mul_(column_weights)
                    model.synchronize(models.device)
                    end = start + decoded.shape[1]
                    sums[:, start:end, rows, columns] += decoded.to("cpu")
                    start = end
            model.synchronize(models.device)
            count[rows, columns].addcmul_(
                row_weights.to("cpu").view(-1, 1), column_weights.to("cpu").view(1, -1)
            )
        divisor = count.to(models.device).clamp(min=1e-6)
        start = 0
        for size in sizes:
            # A copy on the CPU too, where to() alone gives the store's own memory, which goes
            # with the store.
            yield sums[:, start : start + size].to(models.device, copy=True).div_(divisor)
            start += size


def _decoded(models: Models, latent: Tensor, tile: Tile, sizes: list[int]) -> Generator[Tensor]:
    """The tile's crop of latent (T', h, w, 16) decoded slice by slice (model.decode_sliced, the
    crop numz takes of the decoder's input: attn_video_vae.py:1553), each slice checked for
    its place in the scratch: (3, t, 8 rows, 8 columns) in COMPUTE_DTYPE, t following sizes. A
    slice of another shape could be broadcast into that place, and one in another dtype
    rounded."""
    height, width = tile.rows * SCALE, tile.columns * SCALE
    rows, columns = slice(tile.top, tile.bottom), slice(tile.left, tile.right)
    given = 0
    for decoded in model.decode_sliced(models, latent, rows, columns):
        if given == len(sizes):
            raise RuntimeError(f"more slices decoded than the {len(sizes)} of {sizes} frames")
        expected = (CHANNELS, sizes[given], height, width)
        if decoded.shape != expected or decoded.dtype != COMPUTE_DTYPE:
            raise RuntimeError(
                f"slice {given} decoded as {tuple(decoded.shape)} {decoded.dtype},"
                f" not {expected} {COMPUTE_DTYPE}"
            )
        yield decoded
        given += 1
    if given != len(sizes):
        raise RuntimeError(f"{given} slices decoded, not the {len(sizes)} of {sizes} frames")


def _tiled(
    kind: str,
    layout: list[Tile],
    tiles: Tiles,
    frames: int,
    padded: tuple[int, int],
    scratch: Scratch,
) -> None:
    """Say a tiled pass, the encode or the decode (kind), once its scratch's store is open: its
    grid over the shot's `frames` frames padded to (height, width) pixels, and where they wait."""
    down = len({tile.top for tile in layout})
    logger.debug(
        "tiled %s: %d tiles of %d px sharing %d, %d across and %d down, over %d frames padded to"
        " %dx%d; its scratch, %d bytes, in %s",
        *(kind, len(layout), tiles.size, tiles.overlap, len(layout) // down, down),
        *(frames, padded[1], padded[0]),
        *(Scratch.needed(frames, padded), scratch.where(frames, padded)),
    )


def _untiled(kind: str, tiles: Tiles, padded: tuple[int, int]) -> None:
    """Say a pass the plan tiles that runs untiled, the picture being within one tile."""
    logger.debug(
        "tiled %s: the frames, padded to %dx%d, are within one tile of %d px: the untiled pass",
        *(kind, padded[1], padded[0], tiles.size),
    )


def _ramp(steps: int, like: Tensor) -> Tensor | None:
    """numz's ramp across an overlap of `steps` latents (the encode) or output pixels (the
    decode): a raised cosine from 0 to 1, both ends included, computed on like's device in its
    dtype, as numz computes it on the tensor it tiles (attn_video_vae.py:1353-1360, 1516-1523):
    its bits are that device's. None without an overlap."""
    if steps == 0:
        return None
    t = torch.linspace(0, 1, steps=steps, device=like.device, dtype=like.dtype)
    return 0.5 - 0.5 * torch.cos(t * torch.pi)


def _weights(
    like: Tensor, side: int, overlap: int, ramp: Tensor | None, before: bool, after: bool
) -> Tensor:
    """A tile's weights along its `side` rows or columns, numz's (attn_video_vae.py:1426-1443,
    1588-1605), on like's device in its dtype: ones, then the ramp rising across the overlap where
    a tile lies before this one (`before`), then falling across it where one lies after
    (`after`). An edge on the picture's border keeps its ones."""
    ov = max(0, min(overlap, side - 1))
    weights = torch.ones((side,), device=like.device, dtype=like.dtype)
    if ramp is not None and ov > 0:
        if before:
            weights[:ov] = ramp[:ov]
        if after:
            weights[-ov:] = 1 - ramp[:ov]
    return weights

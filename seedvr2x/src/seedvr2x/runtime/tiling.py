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
"""

import contextlib
import logging
import os
from collections.abc import Generator, Iterable, Iterator
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import torch
from torch import Tensor

from seedvr2x.media.ffmpeg import MediaError
from seedvr2x.media.files import make_directories
from seedvr2x.runtime import model
from seedvr2x.runtime.model import COMPUTE_DTYPE, Models
from seedvr2x.runtime.plan import Tiles

logger = logging.getLogger(__name__)

# Output pixels per latent, each way: the VAE's spatial factor (spatial_downsample_factor, which
# numz's tiling reads: attn_video_vae.py:1323, 1480).
SCALE = 8

# A frame's channels, as the VAE takes and gives them.
CHANNELS = 3

# A scratch's file, in the shot's unit directory: raw bfloat16, (CHANNELS, frames, H, W).
SCRATCH = "scratch.bf16"

# The bytes of a scratch kept in system memory where its pass is given no allowance (Scratch), a
# larger one going to its file: for a test or a script that calls shot.encode_shot or
# shot.decode_shot without one. 8 GiB, which holds 675 frames at 1080p (padded to 1104x1920,
# 12,718,080 bytes each) and 170 at 4K (2192x3840, 50,503,680). A run gives each pass its plan's
# allowance, which the planner sets from the host's memory (planner.plan; DESIGN.md, Memory
# planner, Host RAM).
DEFAULT_ALLOWANCE = 8 * 2**30


@dataclass(frozen=True)
class Tile:
    """One tile of the grid, in latents: rows [top, bottom) and columns [left, right) of the
    picture's. Its frames are SCALE times that, in output pixels."""

    top: int
    bottom: int
    left: int
    right: int

    @property
    def rows(self) -> int:
        return self.bottom - self.top

    @property
    def columns(self) -> int:
        return self.right - self.left


def grid(latent_size: tuple[int, int], tiles: Tiles) -> list[Tile]:
    """numz's tiles over a picture of latent_size (rows, columns) latents, in the order numz runs
    them, row by row from the top left corner (attn_video_vae.py:1363-1371, 1526-1534): tiles of
    tiles.size / 8 latents a side, one every side less tiles.overlap / 8, cut off at the picture's
    edges; a tile lying whole within the overlap of the one before it is left out. One tile is the
    whole picture: the untiled pass, as numz runs it then (:1315-1317, 1489-1491).

    The encode and the decode share it: numz's tiled_encode lays it over ceil(H / 8) latent rows
    (:1337-1338), and padded frames are multiples of 16 (job.padded_size).

    ValueError unless tiles.size and tiles.overlap are multiples of 8, 0 <= overlap < size."""
    side, overlap = _latent_tiles(tiles)
    rows, columns = (_spans(length, side, overlap) for length in latent_size)
    return [Tile(top, bottom, left, right) for top, bottom in rows for left, right in columns]


def largest(padded: tuple[int, int], tiles: Tiles) -> int:
    """The area, in output pixels, of the largest tile of the grid over frames padded to
    (height, width) pixels (job.padded_size), cut off at the picture's edges: a 1280-px tile on a
    1104x1920 frame is 1104 x 1280 = 1,413,120. The tile a pass's peak memory follows (DESIGN.md,
    Memory planner)."""
    area = max(tile.rows * tile.columns for tile in grid(_latent_size(padded), tiles))
    return area * SCALE**2


def _latent_tiles(tiles: Tiles) -> tuple[int, int]:
    """A tile's side and its overlap, in latents, as numz derives them from output pixels
    (attn_video_vae.py:1329-1332, 1486-1497)."""
    size, overlap = tiles.size, tiles.overlap
    if size % SCALE or overlap % SCALE or not 0 <= overlap < size:
        raise ValueError(
            f"tiles of {size} px sharing {overlap}: a tile's side and its overlap are multiples"
            f" of {SCALE}, the overlap from 0 up and under the side"
        )
    side = size // SCALE
    return side, min(overlap // SCALE, side - 1)


def _latent_size(padded: tuple[int, int]) -> tuple[int, int]:
    """The latents' (rows, columns) of frames padded to (height, width) pixels."""
    height, width = padded
    if height % SCALE or width % SCALE or min(height, width) < SCALE:
        raise ValueError(f"frames of {width}x{height}: padded frames are multiples of {SCALE}")
    return height // SCALE, width // SCALE


def _spans(length: int, side: int, overlap: int) -> list[tuple[int, int]]:
    """[start, end) of the tiles along `length` latents, `side` long, sharing `overlap`."""
    if length < 1:
        raise ValueError(f"a picture of {length} latents")
    spans: list[tuple[int, int]] = []
    for start in range(0, length, side - overlap):
        end = min(start + side, length)
        # numz: "Skip if fully within overlap of previous tiles".
        if start > 0 and end - start <= overlap:
            continue
        spans.append((start, end))
    return spans


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
    failure alike. A run killed leaves it, for the resume to discard."""

    def __init__(self, directory: Path | None, allowance: int | None = None) -> None:
        self.directory = directory
        self.allowance = DEFAULT_ALLOWANCE if allowance is None else allowance
        self._open = False

    @staticmethod
    def needed(frames: int, padded: tuple[int, int]) -> int:
        """The bytes of a store of `frames` frames padded to (height, width) pixels: 12.7 MB per
        1080p frame, padded to 1104x1920, 50.5 MB per 4K frame, padded to 2192x3840."""
        return CHANNELS * frames * padded[0] * padded[1] * COMPUTE_DTYPE.itemsize

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
            with self._file(size) as path:
                yield _mapped(path, size).view(shape)
        finally:
            self._open = False

    @contextlib.contextmanager
    def _file(self, size: int) -> Generator[Path]:
        """The scratch's file, new, `size` bytes of zeros, there while the with block runs."""
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
                with path.open("xb") as file:
                    made = True
                    # Allocated whole: a full disk fails here, not later as a SIGBUS on a page of
                    # the mapping.
                    os.posix_fallocate(file.fileno(), 0, size)
            except OSError as error:
                raise ScratchError(f"{path}: the shot's scratch, {size} bytes: {error}") from error
            yield path
        finally:
            # Whatever ended the block, an allocation interrupted included. A file that stays
            # changes no result: said, and the run goes on.
            try:
                if made:
                    path.unlink(missing_ok=True)
            except OSError as error:
                logger.warning("%s: the shot's scratch, not removed: %s", path, error)


def _mapped(path: Path, size: int) -> Tensor:
    """The file at path, `size` bytes, as a flat tensor in COMPUTE_DTYPE: a shared mapping, so
    that what is written through the tensor is the file's, which the system writes out and drops
    from memory as it needs to."""
    elements = size // COMPUTE_DTYPE.itemsize
    try:
        return torch.from_file(str(path), shared=True, size=elements, dtype=COMPUTE_DTYPE)
    except RuntimeError as error:
        raise ScratchError(
            f"{path}: the shot's scratch, {size} bytes, not mapped: {error}"
        ) from error


@torch.no_grad()
def encode_tiled(
    models: Models, slices: Iterable[Tensor], frames: int, tiles: Tiles, scratch: Scratch
) -> Tensor:
    """VAE latent of a shot of `frames` frames (4n + 1) fed slice by slice, as model.encode_stream
    takes them (each (C, t, H, W) in [-1, 1], COMPUTE_DTYPE on the device, t following
    model.encode_slices), encoded tile by tile (grid): (T', h, w, 16) in channel-major memory, on
    the device; numz's tiled encode of the shot, bit for bit. A grid of one tile gives
    model.encode_stream's latent, the scratch untouched.

    The slices are first kept in the scratch as they come, on the CPU. Then each tile's crop of
    them goes through the encoder in the same slices, one pass with its own causal memory
    (model.encode_sliced); each slice's moments are weighted on the device, moved to the CPU and
    added to their latents' place in the sums, and the tile's weights to their sum, on the CPU
    (attn_video_vae.py:1426-1457). After the last tile the sums are divided on the device by the
    summed weights (:1459-1463), and the latent is made from them (model.to_latent). The moments'
    sums are latents, small, twice the shot's latent: only the input takes the scratch, which is
    closed before this returns.

    ValueError when the slices don't follow model.encode_slices, differ in size, or aren't in
    COMPUTE_DTYPE, the scratch's and the VAE's."""
    sizes = model.encode_slices(models, frames)
    slices = iter(slices)
    first = next(slices, None)
    if first is None:
        raise ValueError(f"0 slices, the VAE takes {sizes}")
    slices = chain((first,), slices)
    padded = (first.shape[2], first.shape[3])
    del first  # the chain's alone now: on the device until it is kept, not until the last tile
    if len(grid(_latent_size(padded), tiles)) == 1:
        _untiled("encode", tiles, padded)
        return model.encode_stream(models, slices, frames)
    sums, count = _moments(models, slices, sizes, padded, tiles, scratch)
    sums, count = sums.to(models.device), count.to(models.device)
    sums.div_(count.clamp(min=1e-6))
    return model.to_latent(models, sums)


def _moments(
    models: Models,
    slices: Iterable[Tensor],
    sizes: list[int],
    padded: tuple[int, int],
    tiles: Tiles,
    scratch: Scratch,
) -> tuple[Tensor, Tensor]:
    """The tiles' weighted moments of a shot fed in slices of `sizes` frames, summed,
    (1, 32, T', h, w), and the summed weights, (h, w), on the CPU, in the VAE's dtype
    (encode_tiled). The scratch's store is open only here: nothing returned is on it."""
    frames = sum(sizes)
    height, width = _latent_size(padded)
    overlap = _latent_tiles(tiles)[1]
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
            # moments leave the device as they come: a tile's, whole, would grow there with the
            # shot, 1 MiB a latent under 1024-px tiles, where numz's stay for a batch.
            crops = _crops(models, inputs, tile, sizes)
            with contextlib.closing(_encoded(models, crops, tile, frames, latents)) as passes:
                for moments in passes:
                    if sums is None:
                        # Sized by the first moments, as numz's (:1406-1419).
                        shape = (*moments.shape[:2], latents, height, width)
                        sums = torch.zeros(shape, dtype=moments.dtype)
                    moments.mul_(row_weights.view(-1, 1)).mul_(column_weights)
                    model.synchronize(models.device)
                    end = start + moments.shape[2]
                    sums[:, :, start:end, *place] += moments.to("cpu")
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
    """VAE decode of latent (T', h, w, 16), as model.decode_stream takes it, tile by tile (grid):
    frames (C, t, H, W) in [-1, 1], unclamped, on the device, in the slices model.decode_stream
    gives (t following model.decode_slices), handed on once the last tile is decoded; numz's
    tiled decode of the shot, bit for bit. A grid of one tile gives model.decode_stream's slices,
    the scratch untouched.

    Each tile's crop of the latents is decoded slice by slice, one pass with its own causal
    memory (model.decode_sliced); each slice is weighted on the device, moved to the CPU and
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
    z = model.decoder_input(models, latent)
    if z.dtype != COMPUTE_DTYPE:
        raise ValueError(f"latent {z.dtype}, a tiled decode takes {COMPUTE_DTYPE}")
    sizes = model.decode_slices(models, z.shape[2])
    height, width = z.shape[3], z.shape[4]
    padded = (height * SCALE, width * SCALE)
    # The decode's ramp is over the overlap in output pixels, on z's device in its dtype
    # (:1516-1523), which the decoded frames share.
    overlap = tiles.overlap
    ramp = _ramp(overlap, z)
    count = torch.zeros(padded, dtype=COMPUTE_DTYPE)
    with scratch.store(sum(sizes), padded) as sums:
        _tiled("decode", layout, tiles, sum(sizes), padded, scratch)
        for tile in layout:
            rows = slice(tile.top * SCALE, tile.bottom * SCALE)
            columns = slice(tile.left * SCALE, tile.right * SCALE)
            row_weights = _weights(
                z, tile.rows * SCALE, overlap, ramp, tile.top > 0, tile.bottom < height
            )
            column_weights = _weights(
                z, tile.columns * SCALE, overlap, ramp, tile.left > 0, tile.right < width
            )
            crop = z[:, :, :, tile.top : tile.bottom, tile.left : tile.right]
            start = 0
            # Closed whatever ends the tile, which clears its causal memory at once.
            with contextlib.closing(_decoded(models, crop, sizes)) as slices:
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


def _decoded(models: Models, z: Tensor, sizes: list[int]) -> Generator[Tensor]:
    """A tile's crop of z decoded slice by slice (model.decode_sliced), each slice checked for
    its place in the scratch: (3, t, 8h, 8w) in COMPUTE_DTYPE, t following sizes. A slice of
    another shape could be broadcast into that place, and one in another dtype rounded."""
    height, width = z.shape[3] * SCALE, z.shape[4] * SCALE
    given = 0
    for decoded in model.decode_sliced(models, z):
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

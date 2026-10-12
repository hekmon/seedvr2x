"""The arithmetic of VAE tiling (DESIGN.md, Pipeline, per shot, steps 1 and 3; Memory planner):
the tiles numz lays over a picture, the largest one's area, and the bytes of a tiled shot's
scratch. Nothing here imports torch, nor anything that does: the planner imports it
(runtime/planner.py), and counts with it before the command line imports torch, as it does with
runtime/plan.py. runtime/tiling.py runs the passes over this grid, and gives every name here as
its own.

The grid is numz's, bit for bit (v1): tiles laid from the top left corner, a stride of the tile's
side less the overlap. Line references are to numz at 4490bd1
(src/models/video_vae_v3/modules/attn_video_vae.py)."""

from dataclasses import dataclass

from seedvr2x.runtime.plan import Tiles

# Output pixels per latent, each way: the VAE's spatial factor (spatial_downsample_factor, which
# numz's tiling reads: attn_video_vae.py:1323, 1480).
SCALE = 8

# A frame's channels, as the VAE takes and gives them.
CHANNELS = 3

# The bytes of a value in a scratch: bfloat16's, the pipeline's dtype (model.COMPUTE_DTYPE, which
# tests/test_tiling.py holds this to).
VALUE_BYTES = 2

# A scratch's file, in the shot's unit directory: raw bfloat16, (CHANNELS, frames, H, W).
SCRATCH = "scratch.bf16"


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
    side, overlap = latent_tiles(tiles)
    rows, columns = (_spans(length, side, overlap) for length in latent_size)
    return [Tile(top, bottom, left, right) for top, bottom in rows for left, right in columns]


def largest(padded: tuple[int, int], tiles: Tiles) -> int:
    """The area, in output pixels, of the largest tile of the grid over frames padded to
    (height, width) pixels (job.padded_size), cut off at the picture's edges: a 1280-px tile on a
    1104x1920 frame is 1104 x 1280 = 1,413,120. The tile DESIGN.md counts a pass's peak memory by
    (Memory planner); a shorter tile of the grid can take more, and the planner counts each of
    the grid's shapes (planner._fit)."""
    area = max(tile.rows * tile.columns for tile in grid(latent_size(padded), tiles))
    return area * SCALE**2


def scratch_bytes(frames: int, padded: tuple[int, int]) -> int:
    """The bytes of a tiled shot's scratch (tiling.Scratch), `frames` frames padded to
    (height, width) pixels in bfloat16: 12.7 MB per 1080p frame, padded to 1104x1920, 50.5 MB
    per 4K frame, padded to 2192x3840 (DESIGN.md, Pipeline, per shot, step 3)."""
    return CHANNELS * frames * padded[0] * padded[1] * VALUE_BYTES


def latent_tiles(tiles: Tiles) -> tuple[int, int]:
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


def latent_size(padded: tuple[int, int]) -> tuple[int, int]:
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

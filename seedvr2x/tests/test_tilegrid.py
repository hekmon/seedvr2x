"""The arithmetic of VAE tiling without torch (runtime/tilegrid.py), as the planner is to take it:
the grid, the largest tile and the scratch's bytes. tests/test_tiling.py holds the same grid to
numz's own tiles, through runtime/tiling.py, whose names these are."""

import random
import subprocess
import sys

import pytest

from seedvr2x.runtime import tilegrid, tiling
from seedvr2x.runtime.plan import OVERLAP, Tiles
from seedvr2x.runtime.tilegrid import Tile

# What a process has imported once it has the module and has used it.
IMPORTED = """
import sys
from seedvr2x.runtime import tilegrid
from seedvr2x.runtime.plan import Tiles

assert tilegrid.largest((1104, 1920), Tiles(1024)) == 1024 * 1024
assert tilegrid.scratch_bytes(1, (1104, 1920)) == 12_718_080
print(*sorted(name for name in ("torch", "numpy", "torchvision") if name in sys.modules))
"""


def test_imports_no_torch() -> None:
    # For the planner to count with before the command line imports torch (tests/test_cli.py).
    result = subprocess.run(
        [sys.executable, "-c", IMPORTED], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def test_tilings_names_are_these() -> None:
    # One home: what tiling runs its passes over, and what its tests hold to numz's tiles.
    assert tiling.grid is tilegrid.grid and tiling.largest is tilegrid.largest
    assert tiling.Tile is tilegrid.Tile
    assert (tiling.SCALE, tiling.CHANNELS, tiling.SCRATCH) == (8, 3, "scratch.bf16")
    assert (tilegrid.SCALE, tilegrid.CHANNELS, tilegrid.SCRATCH) == (8, 3, "scratch.bf16")
    for frames, padded in ((1, (1104, 1920)), (17, (64, 96)), (301, (2192, 3840))):
        assert tiling.Scratch.needed(frames, padded) == tilegrid.scratch_bytes(frames, padded)


def test_grid_of_real_frames() -> None:
    # 1080p, padded to 1104x1920, under 1024-px tiles sharing 64: the sliver row DESIGN.md names,
    # 144 px.
    full_hd = tilegrid.grid((138, 240), Tiles(1024))
    assert full_hd == [
        Tile(0, 128, 0, 128),
        Tile(0, 128, 120, 240),
        Tile(120, 138, 0, 128),
        Tile(120, 138, 120, 240),
    ]
    assert (full_hd[2].rows * 8, full_hd[2].columns * 8, OVERLAP) == (144, 1024, 64)
    # A 1280-px tile holds its rows: two tiles, side by side.
    assert tilegrid.grid((138, 240), Tiles(1280)) == [Tile(0, 138, 0, 160), Tile(0, 138, 152, 240)]
    # 4K, padded to 2192x3840, under 2048-px tiles; and whole under tiles of its own width.
    assert len(tilegrid.grid((274, 480), Tiles(2048))) == 4
    assert tilegrid.grid((274, 480), Tiles(3840)) == [Tile(0, 274, 0, 480)]
    # Rows outer, columns inner.
    order = [(t.top, t.left) for t in tilegrid.grid((5, 7), Tiles(24, 8))]
    assert order == sorted(order) and len({top for top, _ in order}) == 2


def test_grid_covers_the_picture() -> None:
    # Every latent in a tile, the tiles one stride apart, each but the first adding latents of
    # its own beyond the overlap.
    rng = random.Random(0)
    for _ in range(300):
        side = rng.randint(1, 12)
        overlap = rng.randint(0, side - 1)
        rows, columns = rng.randint(1, 40), rng.randint(1, 40)
        tiles = tilegrid.grid((rows, columns), Tiles(side * 8, overlap * 8))
        assert tilegrid.latent_tiles(Tiles(side * 8, overlap * 8)) == (side, overlap)
        covered = {
            (r, c) for t in tiles for r in range(t.top, t.bottom) for c in range(t.left, t.right)
        }
        assert len(covered) == rows * columns
        assert all(t.top % (side - overlap) == 0 and t.left % (side - overlap) == 0 for t in tiles)
        assert all(0 < t.rows <= side and 0 < t.columns <= side for t in tiles)
        assert all(t.rows > overlap for t in tiles if t.top) and tiles[0] == Tile(
            0, min(side, rows), 0, min(side, columns)
        )


def test_largest() -> None:
    assert tilegrid.largest((1104, 1920), Tiles(1280)) == 1104 * 1280 == 1_413_120
    assert tilegrid.largest((1104, 1920), Tiles(1024)) == 1024 * 1024
    assert tilegrid.largest((1104, 1920), Tiles(2048)) == 1104 * 1920
    assert tilegrid.largest((2192, 3840), Tiles(2048)) == 2048 * 2048
    assert tilegrid.largest((2192, 3840), Tiles(512)) == 512 * 512
    rng = random.Random(1)
    for _ in range(200):
        side = rng.randint(1, 12)
        tiles = Tiles(side * 8, rng.randint(0, side - 1) * 8)
        rows, columns = rng.randint(1, 40), rng.randint(1, 40)
        expected = min(side, rows) * min(side, columns) * 64  # the first tile, cut off
        assert tilegrid.largest((rows * 8, columns * 8), tiles) == expected
        assert tilegrid.latent_size((rows * 8, columns * 8)) == (rows, columns)


def test_scratch_bytes() -> None:
    # DESIGN.md, Pipeline step 3: 12.7 MB per 1080p frame, 50.5 MB per 4K frame, padded.
    assert tilegrid.scratch_bytes(1, (1104, 1920)) == 3 * 1104 * 1920 * 2 == 12_718_080
    assert tilegrid.scratch_bytes(1, (2192, 3840)) == 50_503_680
    assert tilegrid.scratch_bytes(40, (1104, 1920)) == 508_723_200
    assert tilegrid.scratch_bytes(17, (64, 96)) == 17 * 3 * 64 * 96 * 2


@pytest.mark.parametrize(
    "tiles",
    [Tiles(0, 0), Tiles(60, 8), Tiles(64, 60), Tiles(64, 64), Tiles(64, 72), Tiles(64, -8)],
)
def test_bad_tiles(tiles: Tiles) -> None:
    with pytest.raises(ValueError, match="multiples of 8"):
        tilegrid.grid((10, 10), tiles)
    with pytest.raises(ValueError, match="multiples of 8"):
        tilegrid.largest((80, 80), tiles)


def test_bad_sizes() -> None:
    with pytest.raises(ValueError):
        tilegrid.grid((0, 10), Tiles(64))
    for padded in ((100, 96), (96, 100), (0, 96)):
        with pytest.raises(ValueError, match="multiples of 8"):
            tilegrid.largest(padded, Tiles(64))

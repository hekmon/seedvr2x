"""The memory planner (runtime/planner.py; DESIGN.md, Memory planner). Its model: each phase's
predicted peak against the measurements it was fitted on, every expected figure copied from
research/docs/planner-limits.md, research/docs/vram.md or DESIGN.md, the table named, or
measured on seedvr2x with milestone 3 (2026-10-11), said so; where the weights are and what
they take against runtime/blockswap.py's own rules, the tiles and their scratch against
runtime/tiling.py's. Its plans: DESIGN.md's window figures, the order of its choices, vram.md's
recipes per card size, the pins taken or refused, the floor, the system memory, and a resume's
plan around a recorded shape. On the CPU, no weights: the models are built on the meta device
where their shapes are counted."""

import re
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from seedvr2x.runtime import planner
from seedvr2x.runtime.job import Shot, output_size, padded_size
from seedvr2x.runtime.plan import Plan, Tiles
from seedvr2x.runtime.planner import BLOCK, GIB, MARGIN, Pins, PlanError
from seedvr2x.runtime.shot import padded_length, shot_layout

# numz's padded sizes, multiples of 16 (planner-limits.md's tables), and seedvr2x's own
# (job.padded_size; DESIGN.md, Memory planner: 1,104 rows at 1080p, 2,192 at 4K).
NUMZ_1080, NUMZ_4K = (1088, 1920), (2160, 3840)
OURS_1080, OURS_4K = (1104, 1920), (2192, 3840)

# planner-limits.md, "The DiT's memory: tokens plus windows", its 26 passing points: the padded
# size, L, and the columns Tokens, Windows and Peak (GiB).
DIT_POINTS = [
    (NUMZ_1080, 1, 8_160, 30, 16.94),
    (NUMZ_1080, 2, 16_320, 60, 18.01),
    (NUMZ_1080, 3, 24_480, 90, 19.08),
    (NUMZ_1080, 6, 48_960, 120, 22.13),
    (NUMZ_1080, 12, 97_920, 150, 28.15),
    (NUMZ_1080, 21, 171_360, 120, 36.97),
    (NUMZ_1080, 30, 244_800, 150, 45.96),
    (NUMZ_1080, 40, 326_400, 180, 55.93),
    (NUMZ_1080, 50, 408_000, 210, 65.91),
    (NUMZ_1080, 60, 489_600, 240, 75.88),
    (NUMZ_1080, 70, 571_200, 300, 85.94),
    (NUMZ_1080, 75, 612_000, 300, 90.89),
    (NUMZ_1080, 77, 628_320, 330, 92.95),
    (NUMZ_1080, 78, 636_480, 330, 93.94),
    (NUMZ_4K, 1, 32_400, 100, 20.07),
    (NUMZ_4K, 2, 64_800, 200, 24.26),
    (NUMZ_4K, 3, 97_200, 300, 28.46),
    (NUMZ_4K, 4, 129_600, 400, 32.65),
    (NUMZ_4K, 6, 194_400, 400, 40.51),
    (NUMZ_4K, 8, 259_200, 500, 48.64),
    (NUMZ_4K, 10, 324_000, 400, 56.23),
    (NUMZ_4K, 12, 388_800, 500, 64.35),
    (NUMZ_4K, 14, 453_600, 400, 71.94),
    (NUMZ_4K, 16, 518_400, 500, 80.07),
    (NUMZ_4K, 18, 583_200, 500, 87.93),
    (NUMZ_4K, 19, 615_600, 500, 91.86),
]


def gib(count: float) -> float:
    return count / GIB


def test_constants() -> None:
    # The figures the documents give, in bytes: DESIGN.md, Memory planner (Budget, Host RAM,
    # Per-phase peaks) and BlockSwap, rewritten; the weights' parts as milestone 3 measured them
    # on seedvr2x (2026-10-11): a block 0.4220 GiB, the 36 15.19, the DiT's other parameters
    # 0.155, the VAE 0.4669, DESIGN.md's 0.47, and 0.058 of buffers and tables that never move.
    assert gib(planner.MARGIN) == pytest.approx(0.6)
    assert gib(planner.DIT_LOADED) == pytest.approx(15.871)
    assert planner.PER_TOKEN == 127.16 * 1024 and planner.PER_WINDOW == 2.726 * 1024**2
    assert (planner.BLOCKS, planner.SLOTS) == (36, 2)
    assert f"{gib(planner.BLOCK):.4f}" == "0.4220" and f"{gib(36 * planner.BLOCK):.2f}" == "15.19"
    assert f"{gib(planner.DIT_OTHER):.3f}" == "0.155"
    assert f"{gib(planner.VAE_WEIGHTS):.4f}" == "0.4669"
    assert gib(planner.DIT_BUFFERS) == pytest.approx(0.058)
    # The DiT's 15.35 GiB, and with the VAE the 15.81 a plan pins that keeps the two apart.
    dit = 36 * planner.BLOCK + planner.DIT_OTHER
    assert f"{gib(dit):.2f}" == "15.35" and f"{gib(dit + planner.VAE_WEIGHTS):.2f}" == "15.81"
    # The loaded models by their parts: 15.8723 GiB, 0.0013 over the fit's constant, within the
    # 0.005 GiB it holds its points to.
    parts = dit + planner.DIT_BUFFERS + planner.VAE_WEIGHTS
    assert gib(parts - planner.DIT_LOADED) == pytest.approx(0.0013, abs=5e-5)
    assert gib(planner.HOST_RESIDENT) == pytest.approx(2.4)
    assert (planner.TILE_MIN, planner.TILE_STEP, planner.SLICE) == (512, 64, 5)
    # The corrections of the planner's first two matrices (test_first_matrix): 0.4 GiB on every
    # encode (PROVISIONAL, until the encode is refitted on seedvr2x); no tile chosen above 2048
    # px, and 1.5 GiB left under the budget by every VAE pass besides its peak (DESIGN.md, Memory
    # planner: What follows from them; Budget, provisional there).
    assert gib(planner.ENCODE_EXTRA) == pytest.approx(0.4)
    assert planner.TILE_MAX == 2048 and gib(planner.VAE_ROOM) == pytest.approx(1.5)


def test_literals_equal_blockswaps_and_tilings() -> None:
    # The planner imports no torch, so holds literals of what runtime/blockswap.py and
    # runtime/tiling.py define, which do: the same figures.
    from seedvr2x.runtime import blockswap, tiling

    assert (planner.BLOCK, planner.SLOTS) == (blockswap.BLOCK_BYTES, blockswap.SLOTS)
    assert planner.GIB == blockswap.GIB
    assert planner.SCRATCH_PIXEL == tiling.Scratch.needed(1, (1, 1))


def test_weights_by_the_models_shapes() -> None:
    # The parts of the weights as the 7B and the VAE are built, on the meta device, in their 2
    # bytes a value: a block and the DiT's other parameters by their values; the VAE as its part
    # lays its parameters out, each on a multiple of blockswap.ALIGN (blockswap.Part). And a
    # part's pinned copy, of its laid-out bytes rounded up to a page (blockswap.HostCopy): what
    # the planner counts by its own figures.
    import torch
    from omegaconf import OmegaConf

    import seedvr2x.vendor
    from seedvr2x.runtime import blockswap
    from seedvr2x.vendor.common.config import create_object, load_config

    vendor = Path(seedvr2x.vendor.__file__).parent
    config: Any = load_config(str(vendor / "configs_7b" / "main.yaml"))
    vae_config = load_config(str(vendor / "models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml"))
    vae_model: Any = OmegaConf.merge(config.vae.model, vae_config)
    with torch.device("meta"):
        dit = create_object(config.dit.model)
        vae = create_object(vae_model)

    def laid_out(sizes: list[int]) -> int:
        return sum(-(-size // blockswap.ALIGN) * blockswap.ALIGN for size in sizes)

    block = [2 * p.numel() for key, p in dit.named_parameters() if key.startswith("blocks.0.")]
    other = [2 * p.numel() for key, p in dit.named_parameters() if not key.startswith("blocks.")]
    weights = [2 * parameter.numel() for parameter in vae.parameters()]
    assert len(dit.blocks) == planner.BLOCKS
    assert (sum(block), sum(other)) == (planner.BLOCK, planner.DIT_OTHER)
    assert laid_out(weights) == planner.VAE_WEIGHTS
    plan = Plan(5, swapped=36, resident_encode=0, vae_in_dit=False)
    pinned = planner._pinned(plan)  # pyright: ignore[reportPrivateUsage]
    assert pinned == 36 * paged(laid_out(block)) + paged(laid_out(other)) + paged(laid_out(weights))
    assert f"{gib(pinned):.2f}" == "15.81"


@pytest.mark.parametrize(("padded", "latents", "tokens", "windows", "peak"), DIT_POINTS)
def test_dit_fit(
    padded: tuple[int, int], latents: int, tokens: int, windows: int, peak: float
) -> None:
    # The table's tokens and windows exactly. Its peaks are rounded to 0.01 GiB and the fit holds
    # them to 0.005 (planner-limits.md: "fit all 26 passing points to 0.005 GiB"), so the model
    # is compared with the column as printed within the two together, 0.01 GiB; the furthest is
    # 0.0051 (1080p, 78 latents: 93.9349 against 93.94).
    assert planner.tokens(padded, latents) == tokens
    assert planner.attention_windows(padded, latents) == windows
    assert abs(gib(planner.dit_peak(padded, latents)) - peak) <= 0.01


def test_dit_fit_furthest() -> None:
    furthest = max(
        abs(gib(planner.dit_peak(padded, latents)) - peak)
        for padded, latents, _, _, peak in DIT_POINTS
    )
    assert furthest == pytest.approx(0.0051, abs=5e-5)


@pytest.mark.parametrize(
    ("padded", "latents", "layouts"),
    [
        # planner-limits.md, "The DiT's attention windows", Windows (regular / shifted).
        (NUMZ_1080, 1, (25, 30)),
        (NUMZ_1080, 3, (75, 90)),
        (NUMZ_1080, 6, (75, 120)),
        (NUMZ_1080, 12, (100, 150)),
        (NUMZ_1080, 21, (100, 120)),
        (NUMZ_1080, 30, (100, 150)),
        (NUMZ_1080, 50, (175, 210)),
        (NUMZ_1080, 70, (225, 300)),
        (NUMZ_1080, 78, (250, 330)),
        (NUMZ_4K, 1, (81, 100)),
        (NUMZ_4K, 2, (162, 200)),
        (NUMZ_4K, 4, (324, 400)),
        (NUMZ_4K, 6, (243, 400)),
        (NUMZ_4K, 8, (324, 500)),
        # Its last row, 10 to 20 latents: 324 regular, 400 to 500 shifted.
        (NUMZ_4K, 10, (324, 400)),
        (NUMZ_4K, 12, (324, 500)),
        (NUMZ_4K, 14, (324, 400)),
        (NUMZ_4K, 16, (324, 500)),
        (NUMZ_4K, 18, (324, 500)),
        (NUMZ_4K, 19, (324, 500)),
        (NUMZ_4K, 20, (324, 500)),
    ],
)
def test_attention_layouts(padded: tuple[int, int], latents: int, layouts: tuple[int, int]) -> None:
    assert planner.attention_layouts(padded, latents) == layouts
    assert planner.attention_windows(padded, latents) == max(layouts)


def test_out_of_memory_rows() -> None:
    # The table's rows that ran out of memory on the 96 GB card, 94.50 GiB free: 1080p at 79 and
    # 80 latents, 4K at 20, the Fit column 94.93, 95.92 and 95.79: over the budget.
    budget = 94.50 * GIB - planner.MARGIN
    rows = [
        (NUMZ_1080, 79, 644_640, 94.93),
        (NUMZ_1080, 80, 652_800, 95.92),
        (NUMZ_4K, 20, 648_000, 95.79),
    ]
    for padded, latents, tokens, fit in rows:
        assert planner.tokens(padded, latents) == tokens
        assert planner.dit_peak(padded, latents) > budget
        assert abs(gib(planner.dit_peak(padded, latents)) - fit) <= 0.01


def test_seedvr2x_sizes() -> None:
    # DESIGN.md, Memory planner, "By the fit, in step 0's padding": 1,104 rows at 1080p, 2,192
    # at 4K; 6 latents at 1080p take 22.22 GiB; on a budget of 93.74 GiB, 76 latents fit (92.98)
    # and 77 don't (94.07). And on the bare 96 GB card as seedvr2x reads it once a first tensor
    # is allocated, 94.2746 GiB free (measured 2026-10-10), a budget of 93.67: the same.
    assert padded_size((1080, 1920)) == OURS_1080 and padded_size((2160, 3840)) == OURS_4K
    assert planner.tokens(OURS_1080, 1) == 8_280 and planner.tokens(OURS_4K, 1) == 32_880
    assert round(gib(planner.dit_peak(OURS_1080, 6)), 2) == 22.22
    assert round(gib(planner.dit_peak(OURS_1080, 76)), 2) == 92.98
    assert round(gib(planner.dit_peak(OURS_1080, 77)), 2) == 94.07
    for free in (94.34, 94.2746):
        budget = free * GIB - planner.MARGIN
        assert planner.dit_peak(OURS_1080, 76) <= budget < planner.dit_peak(OURS_1080, 77)


def test_dit_swapped_and_vae_away() -> None:
    # A swapped block takes its 0.42 GiB off the peak, but for the two in the slots, and the VAE
    # sent away its 0.47 (DESIGN.md, Memory planner, Per-phase peaks; numz: 28.15 to 27.68 GiB).
    whole = planner.dit_peak(NUMZ_1080, 12)
    assert planner.dit_peak(NUMZ_1080, 12, 1) == planner.dit_peak(NUMZ_1080, 12, 2) == whole
    assert whole - planner.dit_peak(NUMZ_1080, 12, 3) == planner.BLOCK
    assert whole - planner.dit_peak(NUMZ_1080, 12, 36) == 34 * planner.BLOCK
    away = planner.dit_peak(NUMZ_1080, 12, vae_in_dit=False)
    assert whole - away == planner.VAE_WEIGHTS
    assert abs(gib(whole) - 28.15) <= 0.01 and abs(gib(away) - 27.68) <= 0.01
    for swapped in (-1, 37):
        with pytest.raises(ValueError, match="blocks swapped of the DiT's 36"):
            planner.dit_peak(NUMZ_1080, 12, swapped)
    # Measured on seedvr2x (milestone 3, 2026-10-11), one window of 1 latent at 1080p: 17.016
    # GiB with no block swapped, 16.595 with 3, 12.796 with 12 and 2.668 with 36. Each swapped
    # block beyond the two slots takes its bytes off, to 0.001 GiB, as the model has it; the
    # peak itself is 0.061 GiB over numz's fit, whose constants the model keeps, at every count.
    measured = {0: 17.016, 3: 16.595, 12: 12.796, 36: 2.668}
    for swapped, peak in measured.items():
        saved = max(swapped - planner.SLOTS, 0) * planner.BLOCK
        assert abs(measured[0] - peak - gib(saved)) <= 0.0015  # two figures rounded to 0.001
        assert planner.dit_peak(OURS_1080, 1) - planner.dit_peak(OURS_1080, 1, swapped) == saved
        assert 0.0605 <= peak - gib(planner.dit_peak(OURS_1080, 1, swapped)) <= 0.0625


def test_dit_on_gpu() -> None:
    # What of the DiT a VAE phase finds on the GPU, as blockswap.Placement.enter leaves it there
    # (milestone 3): its buffers and tables always; with None, nothing moving, its other
    # parameters, the blocks that aren't swapped and the slots; with 0, nothing more; with a
    # number, that many blocks and the other parameters, the slots freed.
    block, other, never = planner.BLOCK, planner.DIT_OTHER, planner.DIT_BUFFERS
    assert planner.dit_on_gpu(None, 0) == never + other + 36 * block
    # All of the loaded models but the VAE's weights, as the DiT's fit has them: 0.0013 GiB over.
    loaded = planner.DIT_LOADED - planner.VAE_WEIGHTS
    assert gib(planner.dit_on_gpu(None, 0) - loaded) == pytest.approx(0.0013, abs=5e-5)
    assert planner.dit_on_gpu(0, 0) == never
    assert planner.dit_on_gpu(12, 0) == never + other + 12 * block
    assert planner.dit_on_gpu(36, 0) == planner.dit_on_gpu(None, 0)
    # One and two swapped blocks sit in their slots while nothing moves: nothing saved. A phase
    # entered with a number frees the slots, and with 0 the other parameters too, whatever is
    # swapped.
    assert planner.dit_on_gpu(None, 1) == planner.dit_on_gpu(None, 2) == planner.dit_on_gpu(None, 0)
    assert (
        planner.dit_on_gpu(0, 1) == planner.dit_on_gpu(0, 2) == planner.dit_on_gpu(0, 36) == never
    )
    assert planner.dit_on_gpu(10, 2) == never + other + 10 * block
    assert planner.dit_on_gpu(34, 2) == planner.dit_on_gpu(None, 2) - 2 * block
    assert planner.dit_on_gpu(None, 36) == never + other + 2 * block
    assert planner.dit_on_gpu(None, 20) == never + other + 18 * block
    assert planner.dit_on_gpu(16, 20) == never + other + 16 * block
    assert planner.dit_on_gpu(4, 20) == never + other + 4 * block
    for resident, swapped in ((1, 36), (17, 20), (-1, 0), (0, 37), (None, 37)):
        with pytest.raises(ValueError, match="of the DiT's 36"):
            planner.dit_on_gpu(resident, swapped)


def test_dit_on_gpu_is_the_placements() -> None:
    # blockswap.Placement's own count of the DiT's parameters on the device in a phase
    # (device_bytes), on a model of 36 blocks of 512 bytes and 1,024 of other parameters, its
    # host copies plain memory: dit_on_gpu's three cases, block for block, for every count of
    # swapped blocks and every phase's residency the loader accepts.
    import torch

    from seedvr2x.runtime import blockswap

    class Host(blockswap.Cuda):
        def host(self, size: int) -> blockswap.HostCopy:
            return blockswap.HostCopy(size, pin=False)

    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(256, dtype=torch.float16))

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vid_in = torch.nn.Parameter(torch.zeros(512, dtype=torch.float16))
            self.blocks = torch.nn.ModuleList(Block() for _ in range(planner.BLOCKS))

    block, other = 512, 1024
    device = torch.device("cpu")
    for swapped in (0, 1, 2, 3, 20, 35, 36):
        fixed = planner.BLOCKS - swapped
        model = Model().requires_grad_(False)
        parts = blockswap.host(model, blockswap.dit_parts(planner.BLOCKS, 0, True), Host(device))
        placement = blockswap.Placement(model, device, swapped, parts=parts, backend=Host(device))
        for resident in (None, *sorted({0, 1, fixed // 2, fixed} & set(range(fixed + 1)))):
            ours = planner.dit_on_gpu(resident, swapped) - planner.DIT_BUFFERS
            blocks, others = divmod(ours, planner.BLOCK)
            assert others in (0, planner.DIT_OTHER)
            theirs = placement.device_bytes(resident)
            assert theirs == blocks * block + (other if others else 0), (swapped, resident)
        with pytest.raises(ValueError, match="blocks to keep"):
            placement.device_bytes(fixed + 1)


def test_grid_and_tile_area() -> None:
    # planner-limits.md, "Tiled VAE at 4K", Tiles (rows x columns), overlap 128: 1024 px 3 x 5,
    # 1536 px 2 x 3, 2048 px 2 x 2; vram.md, Tiling: 60 tiles of 512 px and 24 of 768 at overlap
    # 128, 12 of 1024 at overlap 64, 15 at 128 and at 256.
    assert planner.grid(NUMZ_4K, Tiles(1024, 128)) == (3, 5)
    assert planner.grid(NUMZ_4K, Tiles(1536, 128)) == (2, 3)
    assert planner.grid(NUMZ_4K, Tiles(2048, 128)) == (2, 2)
    assert planner.grid(NUMZ_4K, Tiles(512, 128)) == (6, 10)
    assert planner.grid(NUMZ_4K, Tiles(768, 128)) == (4, 6)
    assert planner.grid(NUMZ_4K, Tiles(1024, 64)) == (3, 4)
    assert planner.grid(NUMZ_4K, Tiles(1024, 256)) == (3, 5)
    # A side the tile covers takes one tile; seedvr2x's 1080p under 1024-px tiles keeps numz's
    # sliver row (DESIGN.md, Pipeline, per shot, step 3: 144 px at 1080p).
    assert planner.grid(OURS_1080, Tiles(1280)) == (1, 2)
    assert planner.grid(OURS_1080, Tiles(1024)) == (2, 2)
    assert planner.grid(OURS_1080, Tiles(1920)) == planner.grid(OURS_1080, Tiles(4096)) == (1, 1)
    assert planner.grid(OURS_4K, Tiles(1024)) == (3, 4)
    # The largest tile, cut off at the picture's edges (DESIGN.md, Memory planner: "a 1280-px
    # tile at 1080p is 1104x1280"), in megapixels.
    assert planner.tile_area(OURS_1080, Tiles(1280)) == 1104 * 1280 / 1e6
    assert planner.tile_area(OURS_1080, Tiles(1024)) == 1024 * 1024 / 1e6
    assert planner.tile_area(NUMZ_4K, Tiles(2048)) == 2048 * 2048 / 1e6
    assert planner.tile_area(OURS_1080, Tiles(1920)) == 1104 * 1920 / 1e6
    assert not planner.covers(OURS_1080, Tiles(1856)) and planner.covers(OURS_1080, Tiles(1920))


def test_grid_is_the_tiled_passes() -> None:
    # The planner's grid, counted in pixels, is the tiled passes' own over the latents
    # (tiling.grid), its largest tile theirs (tiling.largest), and a tile covers the frame
    # exactly where their grid is one tile, the pass then running untiled: over padded sizes
    # from 16 px to beyond 4K, every tile the planner makes or takes up to 4096 px, at the
    # plan's overlap, and at the others the studies ran.
    from seedvr2x.runtime import tiling

    sides = (16, 128, 432, 512, 528, 544, 576, 720, 1024, 1088, 1104, 1280, 1920, 2192, 3840, 4320)
    sizes = [(height, width) for height in sides for width in sides]
    checked = 0
    for overlap, tile_sizes in ((64, range(512, 4097, 64)), (128, (512, 1024, 2048)), (0, (512,))):
        for size in tile_sizes:
            tiles = Tiles(size, overlap)
            for padded in sizes:
                theirs = tiling.grid((padded[0] // 8, padded[1] // 8), tiles)
                rows, columns = planner.grid(padded, tiles)
                assert len({tile.top for tile in theirs}) == rows
                assert len({tile.left for tile in theirs}) == columns
                assert len(theirs) == rows * columns
                assert tiling.largest(padded, tiles) == round(
                    planner.tile_area(padded, tiles) * 1e6
                )
                assert planner.covers(padded, tiles) == (len(theirs) == 1)
                checked += 1
    assert checked == 256 * (57 + 3 + 1)


def vae_fits(padded: tuple[int, int], tiles: Tiles | None) -> tuple[float, float]:
    """The encode's and the decode's fits, GiB above what is on the GPU before the call: the
    peaks less the VAE's weights and what of the DiT never moves, the DiT away, as the tables
    give them, measured on numz: the encode's without what seedvr2x's holds over its fit
    (planner.ENCODE_EXTRA)."""
    before = planner.VAE_WEIGHTS + planner.dit_on_gpu(0, 0)
    encode = planner.encode_peak(padded, tiles, resident=0) - before - planner.ENCODE_EXTRA
    decode = planner.decode_peak(padded, output_size(padded), tiles, resident=0) - before
    return gib(encode), gib(decode)


def tile_fits(size: int, area: float) -> tuple[float, float]:
    """The tiled fits of the encode and of the decode for a tile of `size` px a side and `area`
    megapixels, in GiB: the refit from 1024 px, vram.md's below (DESIGN.md, Memory planner,
    Per-phase peaks)."""
    if size >= 1024:
        return 1.36 + 8.66 * area, 0.44 + 16.29 * area
    return 1.7 + 8.4 * area, 1.6 + 15.6 * area


def test_vae_tiled_squares_measured() -> None:
    # DESIGN.md, Memory planner, Per-phase peaks: "the planner takes the measured peak of every
    # size it may choose": a square tile, one the frame doesn't cut off, takes the peak measured
    # on seedvr2x's own tiled passes (2026-10-11, 21 frames), encode and decode, exact, whatever
    # the frame and the overlap.
    measured = {
        512: (3.52, 5.89),
        768: (7.08, 11.11),
        1024: (10.47, 17.46),
        1536: (21.50, 38.73),
        2048: (37.76, 68.51),
    }
    assert measured == planner.MEASURED
    for size, peaks in measured.items():
        for padded in (NUMZ_4K, OURS_4K, (2048, 2064)):
            assert vae_fits(padded, Tiles(size)) == pytest.approx(peaks, abs=1e-8)
            assert vae_fits(padded, Tiles(size, 128)) == vae_fits(padded, Tiles(size))
    for size in (512, 768, 1024):
        assert vae_fits(OURS_1080, Tiles(size)) == pytest.approx(measured[size], abs=1e-8)
    # "From 1024 px up the fits hold within 0.3 GiB. Below, they fall up to 0.43 GiB short (768
    # px)": the encode's there, the decode's 0.31; at 512 px the encode's fit is 0.38 over.
    over = {size: tile_fits(size, size**2 / 1e6) for size in measured}
    for size, peaks in measured.items():
        spread = [fit - peak for fit, peak in zip(over[size], peaks, strict=True)]
        assert max(map(abs, spread)) < (0.3 if size >= 1024 else 0.44)
        over[size] = (spread[0], spread[1])
    assert over[768] == pytest.approx((-0.426, -0.309), abs=1e-3)
    assert over[512] == pytest.approx((0.382, -0.201), abs=1e-3)
    # numz's own, which the refit was drawn from (planner-limits.md, "Tiled VAE at 4K", Encode
    # peak and Decode peak), are within 0.26 GiB of them, and vram.md's (Tiling, 4K, batch 9, one
    # decimal) within 0.11.
    numz = {1024: (10.44, 17.52), 1536: (21.45, 38.86), 2048: (37.67, 68.76)}
    for size, peaks in numz.items():
        assert max(abs(a - b) for a, b in zip(peaks, measured[size], strict=True)) < 0.26
    rounded = {512: (3.5, 5.9), 768: (7.1, 11.1), 1024: (10.4, 17.4), 1536: (21.4, 38.7)}
    for size, peaks in rounded.items():
        assert max(abs(a - b) for a, b in zip(peaks, measured[size], strict=True)) < 0.11


def test_vae_tiled_between_and_cut_off() -> None:
    # A square of a side between two measured ones takes the larger of its fit and of the
    # straight line between those two, by area (PROVISIONAL, until every size is measured). At
    # 640 px, 45% of the way from 512 to 768 px by area: the encode's fit, 5.14 GiB, over the
    # line's 5.12; the decode's line, 8.24, over the fit's 7.99, which was 0.13 GiB under the
    # first matrix's measured peak (test_first_matrix).
    share = (640**2 - 512**2) / (768**2 - 512**2)
    assert share == pytest.approx(0.45)
    lines = (3.52 + share * (7.08 - 3.52), 5.89 + share * (11.11 - 5.89))
    fits = tile_fits(640, 0.4096)
    assert [round(value, 2) for value in (*fits, *lines)] == [5.14, 7.99, 5.12, 8.24]
    assert vae_fits(OURS_1080, Tiles(640)) == pytest.approx((fits[0], lines[1]), abs=1e-8)
    # At 896 px the encode's is the line, 0.21 GiB over its fit; at 1280 px, a square at 4K,
    # both are the fits, 0.1 GiB over the lines.
    share = (896**2 - 768**2) / (1024**2 - 768**2)
    line = 7.08 + share * (10.47 - 7.08)
    assert vae_fits(OURS_1080, Tiles(896))[0] == pytest.approx(line, abs=1e-8)
    assert line - tile_fits(896, 0.802816)[0] == pytest.approx(0.211, abs=1e-3)
    assert vae_fits(OURS_4K, Tiles(1280)) == pytest.approx(tile_fits(1280, 1.6384), abs=1e-8)
    share = (1280**2 - 1024**2) / (1536**2 - 1024**2)
    assert tile_fits(1280, 1.6384)[0] - (10.47 + share * (21.50 - 10.47)) == pytest.approx(
        0.116, abs=1e-3
    )
    # Every square from 512 to 2048 px is at or above its fit unless it is measured, and a
    # larger tile never takes less, square or cut off (2160 rows cut the tiles off from 2176
    # px), below the squares measured and above (2112 px, by its fit).
    encodes: list[float] = []
    decodes: list[float] = []
    for size in range(512, 3840, 64):
        encode, decode = vae_fits(NUMZ_4K, Tiles(size))
        fits = tile_fits(size, min(size, 2160) * size / 1e6)
        if size in planner.MEASURED:
            assert (encode, decode) == pytest.approx(planner.MEASURED[size], abs=1e-8)
        elif size < 2048:
            assert encode >= fits[0] - 1e-8 and decode >= fits[1] - 1e-8
        else:
            assert (encode, decode) == pytest.approx(fits, abs=1e-8)
        encodes.append(encode)
        decodes.append(decode)
    assert encodes == sorted(encodes) and decodes == sorted(decodes)

    # A tile cut off at the picture's edges, measured or not as a square, takes the larger of
    # its fit at its area and of the line between the two measured squares that area lies
    # between. DESIGN.md: a 1280-px tile at 1080p, 1104x1280, measured at 13.55 and 23.34 GiB,
    # "the fits serving the tiles cut off": 13.60 and 23.46, over the lines' 13.54 and 23.38.
    def between(area: float) -> tuple[float, float]:
        """The line between the measured squares around an area, in megapixels: encode, decode."""
        below = max(side for side in planner.MEASURED if side**2 <= area * 1e6)
        above = min(side for side in planner.MEASURED if side**2 > area * 1e6)
        share = (area * 1e6 - below**2) / (above**2 - below**2)
        low, high = planner.MEASURED[below], planner.MEASURED[above]
        return low[0] + share * (high[0] - low[0]), low[1] + share * (high[1] - low[1])

    encode, decode = vae_fits(OURS_1080, Tiles(1280))
    assert (encode, decode) == pytest.approx(tile_fits(1280, 1.41312), abs=1e-8)
    assert (round(encode, 2), round(decode, 2)) == (13.60, 23.46)
    assert [round(line, 2) for line in between(1.41312)] == [13.54, 23.38]
    assert 0 <= encode - 13.55 < 0.05 and 0 <= decode - 23.34 < 0.12
    # Under 1024 px the small tiles' fit is short of a tile cut off by a few rows: at 720p,
    # padded to 752x1280, a tile of 768 px, 752x768, takes the line's 6.95 and 10.91 GiB, where
    # its fit gives 6.55 and 10.61 and the square of 768 px was measured at 7.08 and 11.11.
    area = 752 * 768 / 1e6
    assert [round(fit, 2) for fit in tile_fits(768, area)] == [6.55, 10.61]
    assert vae_fits((752, 1280), Tiles(768)) == pytest.approx(between(area), abs=1e-8)
    assert [round(peak, 2) for peak in vae_fits((752, 1280), Tiles(768))] == [6.95, 10.91]
    # The fit changes at 1024 px: vram.md's below, the refit from there; a tile of 960 px cut
    # off at 944 rows takes the encode's line, 0.11 GiB over its fit.
    for size in (960, 1024, 1536, 2048):
        area = 944 * size / 1e6
        larger = [max(pair) for pair in zip(tile_fits(size, area), between(area), strict=True)]
        assert vae_fits((944, 3840), Tiles(size)) == pytest.approx(larger, abs=1e-8)
    assert vae_fits((944, 3840), Tiles(960))[0] - tile_fits(960, 944 * 960 / 1e6)[0] == (
        pytest.approx(0.106, abs=1e-3)
    )


def test_vae_untiled() -> None:
    # DESIGN.md, Memory planner, Per-phase peaks: encode 1.2 + 8.8 P, decode 0.8 + 16.1 P GiB,
    # P the padded megapixels. With the VAE's weights, vram.md's measured peaks (Recipe per card
    # size, 48 GB at 1080p, untiled: 21.44 / 35.72 in numz's padding, its batch's frames on the
    # GPU for the encode) and DESIGN.md's untiled 4K decode, "about 134 GiB".
    assert vae_fits(NUMZ_1080, None) == pytest.approx(
        (1.2 + 8.8 * 2.08896, 0.8 + 16.1 * 2.08896), abs=1e-6
    )
    encode, decode = (fit + gib(planner.VAE_WEIGHTS) for fit in vae_fits(NUMZ_1080, None))
    assert abs(decode - 35.72) < 0.9 and abs(encode - 21.44) < 1.5
    assert round(vae_fits(NUMZ_4K, None)[1]) == 134
    # The largest tile the planner makes at 1080p, 1856 px, takes less; one that covers the
    # whole padded frame is no tiling (DESIGN.md, Pipeline, per shot, step 3), and has no peak.
    assert vae_fits(OURS_1080, Tiles(1856)) < vae_fits(OURS_1080, None)
    for size in (1920, 4096):
        with pytest.raises(ValueError, match=f"tiles of {size} px over frames padded to 1920x1104"):
            planner.encode_peak(OURS_1080, Tiles(size))
        with pytest.raises(ValueError, match="one covers the whole frame, which is no tiling"):
            planner.decode_peak(OURS_1080, HD, Tiles(size))


def test_vae_untiled_small_frames() -> None:
    # The untiled fits come from frames of 720p and up; under 1.25 megapixels for the encode and
    # 1.6 for the decode, an untiled pass takes the small tiles' fit at its frame's area, the
    # larger of the two there (planner._fit; PROVISIONAL). 1080p and up keep the untiled fits.
    area = 576 * 960 / 1e6
    assert vae_fits((576, 960), None) == pytest.approx(
        (1.7 + 8.4 * area, 1.6 + 15.6 * area), abs=1e-6
    )
    area = 1104 * 1920 / 1e6
    assert vae_fits(OURS_1080, None) == pytest.approx(
        (1.2 + 8.8 * area, 0.8 + 16.1 * area), abs=1e-6
    )
    # Which of the encode's and the decode's is its own untiled fit, either side of where the
    # two fits cross.
    for megapixels, own in ((1.24, (False, False)), (1.26, (True, False)), (1.61, (True, True))):
        fitted = vae_fits((1000, round(megapixels * 1000)), None)
        untiled = (1.2 + 8.8 * megapixels, 0.8 + 16.1 * megapixels)
        assert (abs(fitted[0] - untiled[0]) < 1e-6, abs(fitted[1] - untiled[1]) < 1e-6) == own
    # Measured on seedvr2x (the 96 GB card, 2026-10-11), every weight on the GPU, the DiT's
    # blocks and the VAE, split on: the peaks of the encode and of the decode at 540p, padded to
    # 576x960, and at 1080p. The decode's prediction is within 0.05 GiB under and 0.6 over; the
    # encode's holds ENCODE_EXTRA over its fit.
    there = planner.dit_on_gpu(None, 0) + planner.VAE_WEIGHTS
    assert padded_size((540, 960)) == (576, 960)
    measured = {((540, 960), (576, 960)): (22.53, 26.29), (HD, OURS_1080): (35.36, 50.98)}
    over: dict[tuple[int, int], tuple[float, float]] = {}
    for (target, padded), (encode, decode) in measured.items():
        fitted = round(vae_fits(padded, None)[0] * GIB)
        assert planner.encode_peak(padded) - there == fitted + planner.ENCODE_EXTRA
        predicted = planner.decode_peak(padded, target, split=True)
        assert -0.05 <= gib(predicted) - decode <= 0.6
        over[target] = (gib(planner.encode_peak(padded)) - encode, gib(predicted) - decode)
    assert over[HD] == pytest.approx((0.766, 0.519), abs=1e-3)
    # The 540p encode's fit is 0.31 GiB under its measured peak, with the small tiles' fit too
    # (0.59 under by the untiled fit): one of the three excesses ENCODE_EXTRA, 0.4 GiB, is the
    # largest of (test_first_matrix), which leaves its prediction 0.09 over. Probably the
    # slice's source frames on the GPU, 5 frames of 1920x1080 in float32, and the resize's
    # temporaries, which the fits, measured on numz, never held.
    assert over[540, 960] == pytest.approx((0.087, -0.016), abs=1e-3)


def test_vae_phase_peaks() -> None:
    # A VAE phase's peak: its fit, the VAE's weights (0.47 GiB) and what of the DiT stays there;
    # DESIGN.md, BlockSwap, rewritten: an untiled 1080p decode "takes about 35 GiB with the VAE
    # alone", and the DiT's 15.35 GiB beside it pass a 48 GB card's budget.
    alone = planner.decode_peak(OURS_1080, (1080, 1920), resident=0)
    beside = planner.decode_peak(OURS_1080, (1080, 1920))
    assert round(gib(alone), 1) == 35.5 and round(gib(beside), 1) == 50.8
    assert beside - alone == 36 * planner.BLOCK + planner.DIT_OTHER
    assert f"{gib(beside - alone):.2f}" == "15.35"
    assert beside > (48 - 1.4285) * GIB - planner.MARGIN > alone
    # The VAE alone whatever is swapped: a phase that keeps nothing of the DiT frees its slots.
    # Nothing moving, every block swapped, the other parameters and the two slots stay: 1.0 GiB.
    assert planner.decode_peak(OURS_1080, (1080, 1920), resident=0, swapped=36) == alone
    slots = planner.decode_peak(OURS_1080, (1080, 1920), swapped=36)
    assert slots - alone == 2 * planner.BLOCK + planner.DIT_OTHER
    assert f"{gib(slots - alone):.2f}" == "1.00"
    encode = planner.encode_peak(OURS_1080, Tiles(1024), 4, 20)
    # The encode's holds what seedvr2x's takes over its peak too, 0.4 GiB: here over a square
    # tile's measured 10.47 GiB.
    assert encode == round(10.47 * GIB) + planner.ENCODE_EXTRA + (
        planner.VAE_WEIGHTS + planner.dit_on_gpu(4, 20)
    )


def test_split_term() -> None:
    # research/docs/colour.md, Cost and what streams: split takes 0.56 GiB per 4K frame on the
    # GPU; it corrects a slice of the decode at once, 5 frames at most: 2.8 GiB at 4K, a quarter
    # of it at 1080p. The untiled decode, streamed, holds it over its own peak.
    plain = planner.decode_peak(OURS_4K, (2160, 3840), None, 0)
    split = planner.decode_peak(OURS_4K, (2160, 3840), None, 0, split=True)
    assert gib(split - plain) == pytest.approx(5 * 0.56, abs=1e-6)
    plain = planner.decode_peak(OURS_1080, (1080, 1920), resident=0)
    split = planner.decode_peak(OURS_1080, (1080, 1920), resident=0, split=True)
    assert gib(split - plain) == pytest.approx(5 * 0.56 / 4, abs=1e-6)
    # A tiled decode corrects its frames after its last tile, the tile's working memory gone:
    # its peak is the tile's, which the correction's is under at every size seedvr2x makes
    # (test_first_matrix: the tiled decodes measured under the two summed by that term).
    for padded, output in ((OURS_4K, (2160, 3840)), (OURS_1080, (1080, 1920))):
        for size in (512, 1024, 1856):
            plain = planner.decode_peak(padded, output, Tiles(size), 0)
            assert planner.decode_peak(padded, output, Tiles(size), 0, split=True) == plain
    # The larger of the two all the same: 8K's slice, 11.2 GiB, over a 512-px tile's 5.89.
    before = planner.VAE_WEIGHTS + planner.dit_on_gpu(0, 0)
    plain = planner.decode_peak((4336, 7680), (4320, 7680), Tiles(512), 0)
    split = planner.decode_peak((4336, 7680), (4320, 7680), Tiles(512), 0, split=True)
    assert round(gib(plain - before), 2) == 5.89
    assert gib(split - before) == pytest.approx(4 * 5 * 0.56, abs=1e-6)


def test_scratch() -> None:
    # DESIGN.md, Pipeline, per shot, step 3: 12.7 MB per 1080p frame and 50.5 MB per 4K frame,
    # the padding included; a shot's frames padded to 4n + 1.
    assert round(planner.scratch(OURS_1080, 1) / 1e6, 1) == 12.7
    assert round(planner.scratch(OURS_4K, 1) / 1e6, 1) == 50.5
    assert planner.scratch(OURS_1080, 301) == 301 * 6 * 1104 * 1920
    assert (
        planner.scratch(OURS_1080, 300)
        == planner.scratch(OURS_1080, 298)
        == (planner.scratch(OURS_1080, 301))
    )
    assert planner.scratch(OURS_4K, 2) == 5 * planner.scratch(OURS_4K, 1)
    # The store either tiled pass opens (tiling.encode_tiled, tiling.decode_tiled): the shot's
    # frames padded to 4n + 1, of padded frames, the same for the encode's kept input and for
    # the decode's sums.
    from seedvr2x.runtime import tiling

    for frames in (1, 2, 5, 298, 300, 301):
        for padded in (OURS_1080, OURS_4K, (544, 512)):
            needed = tiling.Scratch.needed(padded_length(frames), padded)
            assert planner.scratch(padded, frames) == needed


def test_grows_with_the_shot() -> None:
    # What a VAE pass keeps on the GPU of the shot itself, per latent of the shot, as measured on
    # seedvr2x's own passes (DESIGN.md, Memory planner, Per-phase peaks: "Each is being taken off
    # the GPU; until then the planner counts them"; PROVISIONAL, until the passes no longer keep
    # them there). A decode, tiled or not, the shot's latent and a copy: 2.02 MiB a latent at
    # 1080p, 8.03 at 4K, with split or without. A tiled encode, its sums at its end: 4 MiB at
    # 1080p, 16 at 4K. An untiled encode, its moments: 6 MiB at 1080p, and by the padded area
    # elsewhere, 23.8 at 4K.
    def grown(peak: Callable[..., int], *job: Any, **options: Any) -> float:
        """What a pass's peak grows by per latent of its shot, in MiB: by as much each."""
        of = [peak(*job, **options, latents=latents) for latents in (0, 1000, 2000)]
        assert of[2] - of[1] == pytest.approx(of[1] - of[0], abs=2)
        return (of[1] - of[0]) / 1000 / 1024**2

    for padded, output, decode, sums, moments in (
        (OURS_1080, (1080, 1920), 2.02, 4, 6.0),
        (OURS_4K, (2160, 3840), 8.03, 16, 23.8),
    ):
        for tiles in (None, Tiles(512), Tiles(1024), Tiles(1856)):
            for split in (False, True):
                kept = grown(planner.decode_peak, padded, output, tiles, 0, split=split)
                assert round(kept, 2) == decode
            kept = grown(planner.encode_peak, padded, tiles)
            assert (round(kept, 1) == moments) if tiles is None else (round(kept) == sums)
    megapixels = 1104 * 1920 / 1e6
    assert (planner.DECODE_LATENT, planner.ENCODE_TILED_SUMS) == (0.954, 1.9)
    assert planner.ENCODE_UNTILED_MOMENTS == 2.83 == round(6 / megapixels, 2)
    assert (round(2.02 / megapixels, 3), round(4 / megapixels, 1)) == (0.953, 1.9)
    # By the megapixel of the padded frame: 540p, padded to 576x960, a quarter of 1080p's and
    # a little more, 0.53 MiB a latent in a decode; a peak without a shot's length holds none.
    small = grown(planner.decode_peak, (576, 960), (540, 960))
    assert small == pytest.approx(0.954 * 576 * 960 / 1e6, abs=1e-6) and round(small, 2) == 0.53
    whole = planner.decode_peak(OURS_1080, (1080, 1920))
    assert whole == planner.decode_peak(OURS_1080, (1080, 1920), latents=0)
    assert planner.encode_peak(OURS_1080) == planner.encode_peak(OURS_1080, latents=0)


# The planner itself. Targets (job.target_size) and the memory a card has free. SMALL is a
# frame little larger than a tile, padded to 544x512, where the untiled decode takes little more
# than the tiled one (test_floor).
HD, UHD = (1080, 1920), (2160, 3840)
SMALL = (512, 512)


def shots(*latents: int) -> list[Shot]:
    """Shots of these lengths in latents, one after the other: 4n - 3 frames each."""
    found: list[Shot] = []
    for count in latents:
        start = found[-1].end if found else 0
        found.append(Shot(start, start + 4 * count - 3))
    return found


def bare(plan: Plan) -> Plan:
    """A plan without what its scratch may take of system memory, which follows the host's
    memory and the longest shot (test_scratch_allowance): its shape and where its weights are."""
    return replace(plan, scratch_allowance=0)


def numz_free(card: int) -> int:
    """The memory numz's process found free on a card of `card` GB, in bytes (DESIGN.md, Memory
    planner): 94.50 GiB on the 96 GB card, N - 1.44 on a card emulated at N GB."""
    return round((94.50 if card == 96 else card - 1.44) * GIB)


def our_free(card: int) -> int:
    """The memory seedvr2x itself found free after its shot detector (cli._free_memory; measured
    2026-10-10): 94.2746 GiB on the bare 96 GB card, N - 1.4285 on one emulated. The inputs of
    these tests, whose figures follow from them; read with a first allocation in place, it is
    94.24 GiB and N - 1.46 (DESIGN.md, Memory planner, Budget)."""
    return round((94.2746 if card == 96 else card - 1.4285) * GIB)


def by_the_fit(free: int, target: tuple[int, int], swapped: int) -> int:
    """The longest window, in latents, whose peak fits `free` less the margin by the fit alone,
    the VAE's weights on the GPU in the DiT's phase, as numz has them: how DESIGN.md reckons its
    window figures."""
    latents = 0
    while planner.dit_peak(padded_size(target), latents + 1, swapped) <= free - MARGIN:
        latents += 1
    return latents


def in_one_window(free: int, target: tuple[int, int], pins: Pins) -> int:
    """The longest shot, in latents, that the planner runs in one window, alone in its job."""
    latents = 0
    while True:
        try:
            _, peaks = planner.plan(free, target, shots(latents + 1), pins, None)
        except PlanError:
            return latents
        if peaks.window != latents + 1:
            return latents
        latents += 1


@pytest.mark.parametrize(
    ("target", "card", "swapped", "design", "ours", "planned"),
    [
        # DESIGN.md, Memory planner, "By the fit, in step 0's padding": every block on the GPU,
        # at 1080p 5 latents on 24 GB, 13 on 32, 29 on 48 and 76 on 96; at 4K 3 on 32 GB, 7 on
        # 48, 19 on 96.
        (HD, 24, 0, 5, 5, 6),
        (HD, 32, 0, 13, 13, 14),
        (HD, 48, 0, 29, 29, 30),
        (HD, 96, 0, 76, 76, 77),
        (UHD, 32, 0, 3, 3, 3),
        (UHD, 48, 0, 7, 7, 7),
        (UHD, 96, 0, 19, 19, 19),
        # All 36 blocks swapped: at 1080p "about" 11 on 16 GB, 19 on 24 and 27 on 32; at 4K 2 on
        # 16 GB, 4 on 24, 6 on 32, 10 on 48 and 22 on 96. With a block's measured bytes, 12 fit
        # on 16 GB by seedvr2x's own reading of the free memory, by 0.3 MiB (13.9712 GiB in a
        # budget of 13.9715), and 11 by numz's, 12 MiB under it.
        (HD, 16, 36, 11, 12, 12),
        (HD, 24, 36, 19, 19, 20),
        (HD, 32, 36, 27, 27, 28),
        (UHD, 16, 36, 2, 2, 3),
        (UHD, 24, 36, 4, 4, 5),
        (UHD, 32, 36, 6, 6, 6),
        (UHD, 48, 36, 10, 10, 10),
        (UHD, 96, 36, 22, 22, 22),
    ],
)
def test_design_windows(
    target: tuple[int, int], card: int, swapped: int, design: int, ours: int, planned: int
) -> None:
    # DESIGN.md's figures are the fit's with the VAE's weights in the DiT's phase, each of the 15
    # found again with the memory numz's process found free (design), and with seedvr2x's own
    # reading (ours) but for one. The planner, its blocks pinned alike, sends the VAE's 0.47 GiB
    # away when the window wants them (its first choice): at 1080p, where a latent takes 1.0
    # GiB, one latent more on every card; at 4K one more on 16 and 24 GB, 3 and 5.
    pins = Pins(swap_blocks=swapped)
    for free, fit in ((numz_free(card), design), (our_free(card), ours)):
        assert by_the_fit(free, target, swapped) == fit
        assert in_one_window(free, target, pins) == planned
        made, peaks = planner.plan(free, target, shots(planned), pins, None)
        assert (made.window, made.swapped, peaks.window) == (planned, swapped, planned)
        assert made.vae_in_dit == (planned == fit)
        assert peaks.dit <= peaks.budget == free - MARGIN


def test_design_first_reading() -> None:
    # DESIGN.md: 94.34 GiB free on the 96 GB card, a budget of 93.74 GiB, "which 76 latents fit
    # (92.98) and 77 don't (94.07)": so by the fit; the planner runs 77, the VAE away (93.60).
    free = round(94.34 * GIB)
    assert by_the_fit(free, HD, 0) == 76
    made, peaks = planner.plan(free, HD, shots(77), Pins(swap_blocks=0), None)
    assert (made.window, made.vae_in_dit, round(peaks.dit / GIB, 2)) == (77, False, 93.60)
    # Nothing pinned, it swaps blocks rather than send the VAE away alone: 78 is beyond both.
    assert in_one_window(free, HD, Pins(swap_blocks=0)) == 77


def test_window_is_the_first_choice() -> None:
    # On a 24 GB card at 1080p, a shot of 12 latents runs in one window: the fewest blocks that
    # hold it are swapped, 16, the VAE's weights leaving before a 17th is.
    free, padded = our_free(24), OURS_1080
    made, peaks = planner.plan(free, HD, shots(12), Pins(), None)
    assert (made.window, made.swapped, made.vae_in_dit) == (12, 16, False)
    budget = free - MARGIN
    assert planner.dit_peak(padded, 12, 15, vae_in_dit=False) > budget
    assert planner.dit_peak(padded, 12, 16) > budget >= planner.dit_peak(padded, 12, 16, False)
    assert peaks.dit == planner.dit_peak(padded, 12, 16, False) and peaks.bound == "dit"
    # One and two swapped blocks save nothing, so are never chosen: no block, then three.
    for latents in range(1, 30):
        assert planner.plan(free, HD, shots(latents), Pins(), None)[0].swapped not in (1, 2)
    # The VAE stays where the window fits with it: 13 latents on a 32 GB card, no block swapped;
    # it leaves for 14, before a block is swapped.
    made, _ = planner.plan(our_free(32), HD, shots(13), Pins(), None)
    assert (made.window, made.swapped, made.vae_in_dit) == (13, 0, True)
    made, _ = planner.plan(our_free(32), HD, shots(14), Pins(), None)
    assert (made.window, made.swapped, made.vae_in_dit) == (14, 0, False)
    made, _ = planner.plan(our_free(32), HD, shots(15), Pins(), None)
    assert (made.window, made.swapped, made.vae_in_dit) == (15, 5, False)
    assert planner.dit_peak(padded, 15, 4, vae_in_dit=False) > our_free(32) - MARGIN


def test_large_card_swaps_before_it_shortens() -> None:
    # The 96 GB card swaps blocks on a shot longer than its 76 latents before it shortens the
    # window (DESIGN.md, Memory planner, Choices, in order): 80 latents in one window, 9 blocks
    # swapped; and nothing caps it, a shot it holds whole staying untiled, every weight there.
    free = our_free(96)
    made, peaks = planner.plan(free, HD, shots(80), Pins(), None)
    assert (made.window, peaks.window, made.swapped, made.vae_in_dit) == (80, 80, 9, False)
    assert planner.dit_peak(OURS_1080, 80, 8, vae_in_dit=False) > free - MARGIN
    made, peaks = planner.plan(free, HD, shots(60), Pins(), None)
    assert made == Plan(60) and peaks.bound == "dit"
    # A longer shot than every block swapped holds, 91 latents: two windows of 47, the plan's
    # cap, with no block swapped.
    made, peaks = planner.plan(free, HD, shots(92), Pins(), None)
    assert (made.window, peaks.window, made.swapped) == (47, 47, 0)
    assert in_one_window(free, HD, Pins()) == 91


def test_the_longest_window_run_sets_the_peak() -> None:
    # A shot of 80 latents under a cap of 76 runs two windows of 41: the DiT holds 41 latents,
    # not 76, and no block is swapped for it; with the latent and the noise of the shot's 39
    # others, which stay on the GPU beside the window (planner.dit_kept), 0.08 GiB.
    free = our_free(96)
    made, peaks = planner.plan(free, HD, shots(80), Pins(window=76), None)
    assert made == Plan(76)
    kept = planner.dit_kept(OURS_1080, 39)
    assert peaks.window == 41 and peaks.dit == planner.dit_peak(OURS_1080, 41) + kept
    assert round(gib(kept), 2) == 0.08
    assert shot_layout(shots(80)[0].frames, 76) == [(0, 41), (39, 80)]
    # So with its job: the cap sought is the largest whose longest window fits, which a shot's
    # length may leave shorter than it. A 76-latent shot on a 24 GB card, every block swapped: a
    # cap of 20, the longest that fits there, lays it out in 5 windows of 17 at most, and the
    # fewest blocks that hold 17 latents are swapped, 28. The plan's cap is those 17, which lay
    # the shot out the same, not a 20 the job never runs.
    made, peaks = planner.plan(our_free(24), HD, shots(76), Pins(), None)
    assert (made.window, peaks.window, made.swapped) == (17, 17, 28)
    assert in_one_window(our_free(24), HD, Pins()) == 20
    assert shot_layout(301, 17) == shot_layout(301, 20) and len(shot_layout(301, 20)) == 5
    # A shot of 38 latents there runs two windows of 20 under every cap from 20 to 37: 20, not
    # the 37 the search stops at. Every block is swapped for it: 35 hold a window of 20 alone,
    # not with the shot's 18 other latents beside it, 0.04 GiB.
    made, peaks = planner.plan(our_free(24), HD, shots(38), Pins(), None)
    assert (made.window, peaks.window, made.swapped) == (20, 20, 36)
    alone = planner.dit_peak(OURS_1080, 20, 35, vae_in_dit=False)
    assert alone <= our_free(24) - MARGIN < alone + planner.dit_kept(OURS_1080, 18)
    assert shot_layout(149, 37) == shot_layout(149, 20) == [(0, 20), (18, 38)]
    # A shot of 400 latents on the 96 GB card, no block swapped: 6 windows of 69 at most, under
    # every cap up to 81, the largest that fits, which 5 windows of 82 don't.
    made, peaks = planner.plan(our_free(96), HD, shots(400), Pins(swap_blocks=0), None)
    assert (made.window, peaks.window) == (69, 69)
    assert shot_layout(1597, 69) == shot_layout(1597, 81) and len(shot_layout(1597, 81)) == 6
    assert len(shot_layout(1597, 82)) == 5
    assert in_one_window(our_free(96), HD, Pins(swap_blocks=0)) == 77
    # A split shot's windows can be 4 latents long, under the 5 a cap is at least: a shot of 6
    # latents where 5 fit and not 6 runs two windows of 4, and its plan's cap is 5. At 4K, where
    # the DiT's smallest window is the size's floor (at 1080p the VAE's is, with its room).
    free = planner.dit_peak(OURS_4K, 5, 36, vae_in_dit=False) + MARGIN
    made, peaks = planner.plan(free, UHD, shots(6, 2), Pins(), None)
    assert (made.window, peaks.window) == (5, 4)
    assert shot_layout(21, 5) == [(0, 4), (2, 6)]
    planner.check_recorded(made, shots(6, 2), UHD)
    free = our_free(96)
    # The pin is the cap exactly, even above the longest shot, each shot then one window.
    made, peaks = planner.plan(free, HD, shots(3, 40, 2), Pins(window=60), None)
    assert (made.window, peaks.window) == (60, 40)
    # A job whose shots are all shorter than the smallest cap has its longest shot's length.
    made, peaks = planner.plan(free, HD, shots(3, 2), Pins(), None)
    assert (made.window, peaks.window) == (3, 3)
    made, peaks = planner.plan(free, HD, shots(3, 2), Pins(window=5), None)
    assert (made.window, peaks.window) == (5, 3)


def test_own_cap_lays_the_shots_out_alike() -> None:
    # The planner's own cap for a job that fits under a cap, the longest window the job really
    # runs under it (planner._own_cap), lays every shot out as that cap does (shot.shot_layout),
    # and so runs the same longest window: for shots of many lengths, alone and together, under
    # every cap from the smallest a job can run to beyond its longest shot. 5 latents at least
    # when a shot is split, whose windows can be 4 long: no layout exists under that.
    own_cap = planner._own_cap  # pyright: ignore[reportPrivateUsage]
    longest = planner._longest  # pyright: ignore[reportPrivateUsage]
    job = planner._job  # pyright: ignore[reportPrivateUsage]
    films = [
        *((length,) for length in range(1, 60)),
        (6, 6, 6),
        (1, 4, 5, 6),
        (38,),
        (76,),
        (80, 3),
        (400,),
        (3, 40, 2),
        (1, 4, 5, 6, 9, 17, 40, 80, 133),
        tuple(1 + (7919 * index) % 700 for index in range(60)),
    ]
    trimmed = 0
    for lengths in films:
        film = shots(*lengths)
        counted = job(GIB, HD, film, False, False)
        for cap in range(counted.smallest, max(lengths) + 3):
            own = own_cap(counted, cap)
            run = longest(counted.lengths, cap)
            split = run < max(lengths)
            assert own == (max(run, 5) if split else max(lengths)) <= cap
            for shot in film:
                assert shot_layout(shot.frames, own) == shot_layout(shot.frames, cap)
            assert longest(counted.lengths, own) == run
            trimmed += own < cap
    assert trimmed > 1000
    # Shots of 6 latents on a card that holds 5: two windows of 4 each, under a cap of 5.
    counted = job(GIB, HD, shots(6, 6, 6), False, False)
    assert (longest(counted.lengths, 5), own_cap(counted, 5)) == (4, 5)
    assert shot_layout(21, 5) == [(0, 4), (2, 6)]
    with pytest.raises(ValueError, match="windows of 4 latents leave none of their own"):
        shot_layout(21, 4)


def test_longest_window_by_the_layouts() -> None:
    # The longest window shots really run under a cap is their layouts' (shot.shot_layout), and
    # never falls as the cap rises, which the planner's search of the cap stands on.
    longest = planner._longest  # pyright: ignore[reportPrivateUsage]
    lengths = (1, 4, 5, 6, 9, 17, 40, 80, 133)
    before = 0
    for cap in range(5, 140):
        layouts = [shot_layout(4 * length - 3, cap) for length in lengths]
        expected = max(end - start for layout in layouts for start, end in layout)
        assert longest(lengths, cap) == expected >= before
        assert expected <= max(cap, 4)
        before = expected
    assert longest(lengths, None) == longest(lengths, 133) == longest(lengths, 500) == 133
    assert longest((80,), 76) == 41


def test_a_film_is_planned_at_once() -> None:
    # Thousands of shots, windows of hundreds of latents: well under a second, the caps bisected
    # by their tokens before the model's window functions are asked. The bound is 10 s, for a
    # machine busy with something else: what it must catch is the bisection lost, not a slow day.
    lengths = [1 + (7919 * index) % 700 for index in range(3000)]
    film = shots(*lengths)
    for card in (12, 24, 96):
        started = time.perf_counter()
        made, peaks = planner.plan(our_free(card), HD, film, Pins(), 64 * GIB)
        lines = planner.said(made, peaks, "a card", HD, film, Pins())
        assert time.perf_counter() - started < 10
        assert peaks.window == made.window and len(lines) >= 9  # the planner's own cap
    # One latent under the 91 a card holds of a shot in one window: the film's longest shot,
    # 700 latents, keeps 1.2 GiB of its own beside each of its windows.
    assert planner.plan(our_free(96), HD, film, Pins(), None)[0].window == 90
    assert round(gib(planner.dit_kept(OURS_1080, 700 - 90)), 1) == 1.2


def test_vae_choices() -> None:
    # Untiled where it fits, the DiT's blocks leaving the phase before a tile shrinks; else the
    # largest tile that fits with them away, then as many of them left there as fit.
    # 96 GB at 1080p: both untiled, every block there.
    made, _ = planner.plan(our_free(96), HD, shots(60), Pins(), None)
    assert (made.encode_tiles, made.decode_tiles) == (None, None)
    assert (made.resident_encode, made.resident_decode) == (None, None)
    # 48 GB at 1080p: the untiled decode, about 35 GiB with the VAE alone, fits with 24 of the
    # DiT's blocks beside it and not 25 (DESIGN.md, BlockSwap, rewritten, Per phase) by its peak;
    # with the room every VAE pass is held to, 1.5 GiB under the budget, 20 stay: 21 would, by
    # 2.6 MiB, without the latents of the shot, 21 of them, 42 MiB there through the decode.
    free, room = our_free(48), planner.VAE_ROOM
    budget = free - MARGIN
    made, peaks = planner.plan(free, HD, shots(21), Pins(), None)
    assert made == Plan(21, resident_decode=20)
    untiled = [
        planner.decode_peak(OURS_1080, HD, None, blocks, latents=21) for blocks in (20, 21, 24, 25)
    ]
    assert untiled[3] > budget >= untiled[2]
    assert untiled[1] + room > budget >= untiled[0] + room
    assert budget >= planner.decode_peak(OURS_1080, HD, None, 21) + room
    assert peaks.decode == untiled[0] and peaks.bound == "decode"  # the peak, without the room
    # 32 GB at 1080p: the decode in the largest tiles that fit with the DiT away, 1472 px, cut
    # off at the picture's edges, then two of its blocks left; the encode untiled, 17 left.
    free = our_free(32)
    budget = free - MARGIN
    made, peaks = planner.plan(free, HD, shots(12), Pins(), None)
    assert bare(made) == Plan(12, None, Tiles(1472), 0, 17, 2)
    assert planner.decode_peak(OURS_1080, HD, Tiles(1536), 0, latents=12) + room > budget
    more = planner.decode_peak(OURS_1080, HD, Tiles(1472), 3, latents=12)
    assert more + room > budget >= peaks.decode + room
    more = planner.encode_peak(OURS_1080, None, 18, latents=12)
    assert more + room > budget >= peaks.encode + room
    # 24 GB at 1080p: the untiled encode fits with the DiT away by its peak, and not with its
    # room: tiled, in the largest tiles under the frame's longer side.
    made, peaks = planner.plan(our_free(24), HD, shots(12), Pins(), None)
    assert bare(made) == Plan(12, Tiles(1856), Tiles(1088), 16, 0, 0, vae_in_dit=False)
    untiled_encode = planner.encode_peak(OURS_1080, None, 0, 16, latents=12)
    assert untiled_encode <= our_free(24) - MARGIN < untiled_encode + room
    # split's slice is counted with an untiled decode's peak, which leaves fewer of the DiT's
    # blocks beside it, 19 on 48 GB for 20; a tiled decode's peak doesn't hold it, the same plan
    # with it.
    made, _ = planner.plan(our_free(48), HD, shots(21), Pins(), None, split=True)
    assert made == Plan(21, resident_decode=19)
    made, _ = planner.plan(free, HD, shots(12), Pins(), None, split=True)
    assert bare(made) == Plan(12, None, Tiles(1472), 0, 17, 2)
    # 24 GB at 4K: both tiled, the decode's smaller, its peak about twice the encode's by the
    # megapixel; every block swapped, and the DiT's two slots and its other parameters, 1.0 GiB,
    # leave both passes before a tile shrinks. Beside the encode's tiles they fit a shot of one
    # latent, nothing then moving for its encode, and pass its room with the sums of a longer
    # one's latents, 16 MiB each.
    made, peaks = planner.plan(our_free(24), UHD, shots(5), Pins(), None)
    assert bare(made) == Plan(5, Tiles(1408), Tiles(1088), 36, 0, 0, vae_in_dit=False)
    assert planner.grid(OURS_4K, Tiles(1408)) == (2, 3)
    assert planner.grid(OURS_4K, Tiles(1088)) == (3, 4)
    assert peaks.scratch == planner.scratch(OURS_4K, 17) and peaks.tokens == 32_880
    budget = our_free(24) - MARGIN
    beside = [planner.encode_peak(OURS_4K, Tiles(1408), None, 36, latents=shot) for shot in (1, 2)]
    assert beside[1] + room > budget >= beside[0] + room


def test_a_shot_stays_on_the_gpu_beside_its_window() -> None:
    # A shot longer than its window keeps its latent on the GPU through its windows
    # (run.run_job), and its noise, drawn whole (shot.sample_windows): 1 byte a padded pixel and
    # a latent, 2.02 MiB a latent at 1080p and 8.03 at 4K, what a decode keeps of them
    # (planner.DECODE_LATENT), counted for the latents beyond the window, whose own the fit
    # holds. A one-file output keeps the windows made there too, until the shot's decode
    # (units.Units.save_window): half as much a latent, 1.01 and 4.02 MiB.
    assert planner.DIT_LATENT == planner.DECODE_LATENT == 0.954
    assert planner.DIT_WINDOWS_KEPT == 0.477
    for padded, latent, window in ((OURS_1080, 2.02, 1.01), (OURS_4K, 8.03, 4.02)):
        assert round(planner.dit_kept(padded, 1) / planner.MIB, 2) == latent
        assert round(planner.dit_kept(padded, 0, 1) / planner.MIB, 2) == window
        assert planner.dit_kept(padded, 0) == 0
        assert planner.dit_kept(padded, 7, 5) == pytest.approx(
            7 * planner.dit_kept(padded, 1) + 5 * planner.dit_kept(padded, 0, 1), abs=2
        )
    # 4K on the 96 GB card, one shot of 601 frames, 151 latents. By its window alone the plan
    # was windows of 21 with 18 blocks swapped, 93.45 GiB, 0.22 under the budget; the shot's 130
    # other latents take 1.02 GiB beside it, 94.47 in all, over the free memory itself. That plan
    # no longer fits, and the planner swaps two blocks more.
    free, film = our_free(96), shots(151)
    before = Plan(21, None, Tiles(2048), 18, vae_in_dit=False)
    alone = planner.dit_peak(OURS_4K, 21, 18, vae_in_dit=False)
    kept = planner.dit_kept(OURS_4K, 130)
    assert (round(gib(alone), 2), round(gib(kept), 2)) == (93.45, 1.02)
    assert alone <= free - MARGIN and alone + kept > free
    assert planner.fits(before, free, UHD, film, None) == (
        "the DiT's phase, a window of 21 latents with 18 blocks swapped and the VAE off the GPU,"
        " the shot's own latents counted, takes 95.08 GiB of free VRAM and the driver reports"
        " 94.27 GiB free, 0.81 GiB short"
    )
    made, peaks = planner.plan(free, UHD, film, Pins(), None)
    assert bare(made) == replace(before, swapped=20)
    assert peaks.dit == planner.dit_peak(OURS_4K, 21, 20, vae_in_dit=False) + kept
    assert round(gib(peaks.dit), 2) == 93.63 and peaks.dit <= free - MARGIN
    # A one-file output keeps the windows made too: 8 windows of 21 and 20 latents, the 7 before
    # the last 145 latents, 0.57 GiB. The plan of an output directory doesn't fit it, and two
    # blocks more are swapped.
    sizes = [end - start for start, end in shot_layout(601, 21)]
    assert (len(sizes), max(sizes), sum(sizes[:-1])) == (8, 21, 145)
    assert round(gib(planner.dit_kept(OURS_4K, 0, 145)), 2) == 0.57
    assert planner.fits(made, free, UHD, film, None, on_disk=False) == (
        "the DiT's phase, a window of 21 latents with 20 blocks swapped and the VAE off the GPU,"
        " the shot's own latents counted, takes 94.80 GiB of free VRAM and the driver reports"
        " 94.27 GiB free, 0.53 GiB short"
    )
    one_file, peaks = planner.plan(free, UHD, film, Pins(), None, on_disk=False)
    assert bare(one_file) == replace(before, swapped=22)
    kept = planner.dit_kept(OURS_4K, 130, 145)
    assert peaks.dit == planner.dit_peak(OURS_4K, 21, 22, vae_in_dit=False) + kept
    # 1080p on a 24 GB card, one shot of 1,197 frames, 300 latents: windows of 20 with 35 blocks
    # swapped were 21.96 GiB, under the budget's 21.97, and 22.51 of the 22.57 free with the
    # shot's 280 other latents, 0.55 GiB. The window is now 19 latents, 34 blocks swapped.
    free, film = our_free(24), shots(300)
    before = Plan(20, Tiles(1728), Tiles(1024), 35, 0, None, vae_in_dit=False)
    alone = planner.dit_peak(OURS_1080, 20, 35, vae_in_dit=False)
    kept = planner.dit_kept(OURS_1080, 280)
    assert (round(gib(alone), 2), round(gib(alone + kept), 2)) == (21.96, 22.51)
    assert alone <= free - MARGIN < alone + kept
    assert planner.fits(before, free, HD, film, None) == (
        "the DiT's phase, a window of 20 latents with 35 blocks swapped and the VAE off the GPU,"
        " the shot's own latents counted, takes 23.12 GiB of free VRAM and the driver reports"
        " 22.57 GiB free, 0.55 GiB short"
    )
    made, peaks = planner.plan(free, HD, film, Pins(), None)
    assert bare(made) == replace(before, window=19, swapped=34)
    assert peaks.dit == planner.dit_peak(OURS_1080, 19, 34, vae_in_dit=False) + (
        planner.dit_kept(OURS_1080, 281)
    )
    # The plan's lines say it where the DiT's phase bounds the plan.
    assert planner.said(made, peaks, "a card", HD, film, Pins())[7] == (
        "plan: predicted peaks: encode 19.99 GiB, DiT 21.93 GiB, decode 20.42 GiB; the DiT's phase"
        " bounds the plan, the shot's own latents counted, 0.04 GiB under the budget"
    )
    # A job of many short shots, each in one window, keeps nothing beside a window: its plan is
    # the one it was, its DiT's peak the window's, on disk or in memory.
    for card, target, padded, film, plan in (
        (24, HD, OURS_1080, shots(*range(1, 18)), Plan(17, Tiles(1856), Tiles(1088), 28, 0, 0)),
        (96, UHD, OURS_4K, shots(*range(1, 20)), Plan(19, None, Tiles(2048), vae_in_dit=True)),
    ):
        plan = replace(plan, vae_in_dit=plan.swapped == 0)
        for on_disk in (True, False):
            made, peaks = planner.plan(our_free(card), target, film, Pins(), None, on_disk=on_disk)
            assert bare(made) == plan
            assert peaks.dit == planner.dit_peak(
                padded, peaks.window, made.swapped, made.vae_in_dit
            )
            assert (
                "the shot's own" not in planner.said(made, peaks, "a card", target, film, Pins())[7]
            )


def test_a_long_shot_takes_its_share() -> None:
    # What a pass keeps of the shot grows with it, and is counted for the job's longest shot: a
    # shot of 1,000 latents, 3,997 frames, against one of 20, the cap pinned at 5 latents. At
    # 1080p its decode keeps 1.97 GiB of latents, its untiled encode 5.86 GiB of moments, a
    # tiled one 3.93 GiB of sums; at 4K, a decode 7.84 GiB and a tiled encode 15.62. And the
    # DiT's phase keeps the shot's latent and noise beside each window, 1.96 GiB of the 995
    # latents beyond it at 1080p and 7.80 at 4K (planner.dit_kept).
    pins = Pins(window=5)
    assert round(gib(planner.dit_kept(OURS_1080, 995)), 2) == 1.96
    assert round(gib(planner.dit_kept(OURS_4K, 995)), 2) == 7.80

    def planned(card: int, target: tuple[int, int], latents: int) -> Plan:
        made, peaks = planner.plan(our_free(card), target, shots(latents), pins, None)
        assert planner.fits(made, our_free(card), target, shots(latents), None) is None
        assert (made.window, peaks.window) == (5, 5)
        return bare(made)

    # 24 GB at 1080p: smaller tiles, the encode's and the decode's, and 4 blocks swapped for
    # the windows, the VAE's weights away, where every weight stayed.
    assert planned(24, HD, 20) == Plan(5, Tiles(1856), Tiles(1088), 0, 0, 0)
    assert planned(24, HD, 1000) == Plan(5, Tiles(1472), Tiles(1024), 4, 0, 0, vae_in_dit=False)
    # 32 GB: the encode still untiled, 3 of the DiT's blocks beside it for 17; the decode's
    # tiles one size down, the DiT away.
    assert planned(32, HD, 20) == Plan(5, None, Tiles(1472), 0, 17, 1)
    assert planned(32, HD, 1000) == Plan(5, None, Tiles(1408), 0, 3, 0)
    # 48 GB: both untiled still, 16 blocks beside the decode for 20.
    assert planned(48, HD, 20) == Plan(5, resident_decode=20)
    assert planned(48, HD, 1000) == Plan(5, resident_decode=16)
    # 4K on 24 GB: the smallest window fits with 0.06 GiB to spare, every block swapped, and
    # the long shot's own 7.80 GiB beside it don't: refused, the floor saying what it counts.
    assert planned(24, UHD, 20) == Plan(5, Tiles(1408), Tiles(1088), 36, 0, 0, vae_in_dit=False)
    assert refused(our_free(24), UHD, shots(1000), pins, None).startswith(
        "an output of 3840x2160 takes 30.20 GiB of free VRAM (its smallest window, 5 latents,"
        " with every block of the DiT swapped and the VAE off the GPU, the shot's own latents"
        " counted: a peak of 29.60 GiB, and 0.60 of margin), and the driver reports 22.57 GiB"
        " free: the largest output that fits there is --resolution "
    )
    # On the 96 GB card: the encode tiled where it wasn't, and a block leaving the decode.
    assert planned(96, UHD, 20) == Plan(5, None, Tiles(2048))
    assert planned(96, UHD, 1000) == Plan(5, Tiles(2048), Tiles(2048), resident_decode=35)
    for padded, output, kept in ((OURS_1080, HD, 1.97), (OURS_4K, UHD, 7.84)):
        grown = planner.decode_peak(padded, output, latents=1000)
        assert round(gib(grown - planner.decode_peak(padded, output)), 2) == kept
    for padded, tiles, kept in (
        (OURS_1080, None, 5.86),
        (OURS_1080, Tiles(1024), 3.93),
        (OURS_4K, Tiles(512), 15.62),
    ):
        grown = planner.encode_peak(padded, tiles, latents=1000)
        assert round(gib(grown - planner.encode_peak(padded, tiles)), 2) == kept


# research/docs/vram.md, "Recipe per card size, validated", the 7B fp16's rows: the target, the
# card (GB), the batch (frames), the swapped blocks, the tiles (encode, decode; None untiled),
# and the measured torch peaks, encode / DiT / decode (GiB).
RECIPES = [
    (HD, 24, 45, 24, (1344, 1024), (15.54, 18.43, 18.18)),
    (HD, 32, 45, 0, (None, 1280), (20.58, 28.15, 23.83)),
    (HD, 48, 81, 0, (None, None), (21.44, 36.97, 35.72)),
    (UHD, 24, 13, 36, (1344, 1024), (18.00, 17.86, 18.00)),
    (UHD, 32, 25, 36, (1600, 1280), (25.11, 29.65, 27.68)),
    (UHD, 48, 25, 0, (2048, 1536), (39.63, 44.44, 39.46)),
]


@pytest.mark.parametrize(("target", "card", "frames", "swapped", "tiles", "measured"), RECIPES)
def test_recipes(
    target: tuple[int, int],
    card: int,
    frames: int,
    swapped: int,
    tiles: tuple[int | None, int | None],
    measured: tuple[float, float, float],
) -> None:
    # Each recipe as numz ran it, in numz's padding, on the memory its process found free: one
    # batch as one shot in one window, the DiT off the GPU in the VAE's phases, the VAE's weights
    # on it in the DiT's.
    free, shot = numz_free(card), [Shot(0, frames)]
    latents = (frames - 1) // 4 + 1
    padded = (-(-target[0] // 16) * 16, target[1])
    encode, decode = (None if size is None else Tiles(size) for size in tiles)
    recipe = Plan(latents, encode, decode, swapped, 0, 0)
    # The DiT's peak: the fit's, within 0.01 GiB of the measured one with no block swapped.
    # Swapped, the model keeps two slots on the GPU where numz's swap keeps one block there at a
    # time (DESIGN.md, BlockSwap, rewritten): one block's 0.42 GiB over numz's, within 0.03.
    dit = planner.dit_peak(padded, latents, swapped) / GIB
    assert abs(dit - (BLOCK / GIB if swapped else 0) - measured[1]) <= (0.03 if swapped else 0.01)
    # The VAE's peaks, with what numz has on the GPU then, the VAE alone: the decode within 0.9
    # GiB of the measured one (vram.md: "every measured DiT and decode peak is within 0.9 GiB of
    # the formulas"; the furthest 0.82, the untiled 1080p one); the encode under it, numz
    # holding its batch's frames on the GPU, by 0.13 to 0.99 GiB at 1080p and 0.13 to 1.00 at 4K
    # with the 0.4 GiB seedvr2x's own encode holds over the fit (planner.ENCODE_EXTRA; vram.md
    # says up to 0.9 at 1080p and 1.4 to 2.2 at 4K, over its own fits). The passes' own peaks,
    # a square tile's the one measured on seedvr2x (planner.MEASURED), without what seedvr2x
    # keeps of a shot, which numz's batches never had.
    alone = planner.dit_on_gpu(0, swapped)
    fitted_encode = (planner.encode_peak(padded, encode, 0, swapped) - alone) / GIB
    fitted_decode = (planner.decode_peak(padded, target, decode, 0, swapped) - alone) / GIB
    assert abs(fitted_decode - measured[2]) <= 0.9
    assert 0.1 <= measured[0] - fitted_encode <= 1.1
    # It fits by the model, but for the 4K one on 32 GB: its 7 latents with every block swapped
    # take 30.09 GiB with the model's two slots, 0.14 over the budget, where numz measured
    # 29.65; it fits with the VAE's weights away, as the planner has them there.
    said = planner.fits(recipe, free, target, shot, None, numz_padding=True)
    if (target, card) == (UHD, 32):
        assert said == (
            "the DiT's phase, a window of 7 latents with 36 blocks swapped, takes 30.70 GiB of"
            " free VRAM and the driver reports 30.56 GiB free, 0.14 GiB short"
        )
        recipe = replace(recipe, vae_in_dit=False)
        said = planner.fits(recipe, free, target, shot, None, numz_padding=True)
    assert said is None
    # The planner's own plan for that shot on that card is at least as good: the window whole,
    # no more blocks swapped, no smaller tiles, each VAE pass with its room (planner.VAE_ROOM),
    # which the recipes have too, the DiT away, with what each pass keeps of the shot. On 24 GB
    # at 1080p its encode is in tiles of 1856 px, the untiled one lacking that room, and on 32
    # GB untiled with 18 of the DiT's blocks; at 4K on 48 GB its tiles stop at 2048 px
    # (planner.TILE_MAX).
    made, peaks = planner.plan(free, target, shot, Pins(), None, numz_padding=True)
    assert (made.window, peaks.window) == (latents, latents)
    assert made.swapped <= swapped
    for ours, theirs in ((made.encode_tiles, encode), (made.decode_tiles, decode)):
        assert ours is None or (theirs is not None and ours.size >= theirs.size)
    own = {
        (HD, 24): Plan(12, Tiles(1856), Tiles(1088), 16, 1, 0, vae_in_dit=False),
        (HD, 32): Plan(12, None, Tiles(1536), 0, 18, 0),
        (HD, 48): Plan(21, resident_decode=22),
        (UHD, 24): Plan(4, Tiles(1408), Tiles(1088), 27, 1, 0, vae_in_dit=False),
        (UHD, 32): Plan(7, Tiles(1728), Tiles(1280), 36, 0, 0, vae_in_dit=False),
        (UHD, 48): Plan(7, Tiles(2048), Tiles(1600), 0, 13, 3),
    }
    assert bare(made) == own[target, card]


# The planner's first two matrices on seedvr2x (the 96 GB card emulating smaller ones, 2026-10-11,
# split on, one shot): the target, the free memory seedvr2x read (GiB), the shot's frames, the
# plan as it first ran and the torch peaks measured, encode / DiT / decode (GiB), None where the
# run ended out of memory in its encode, the job's first unit; and for those three the plan they
# ran again with, the planner then holding an untiled encode alone to a room, with the peaks of
# the phases that ran, encode / DiT (GiB):
# - 1080p, 22.54 GiB free: the encode in tiles of 1856 px ran (21.03 predicted, 20.57), the DiT
#   (21.54, 21.56 and 21.58); the decode in tiles of 1152 px, predicted 21.68, ran out of memory,
#   19.21 GiB allocated and 2.43 asked for, 5 MiB free on the device;
# - 1080p, 30.54: the untiled encode with 13 blocks ran (26.42, 25.56), the DiT (29.76, 29.78);
#   the decode in tiles of 1600 px, predicted 29.74, ran out of memory, 26.27 and 3.37, 5 MiB;
# - 4K, 46.54: the encode in tiles of 2048 px with 17 blocks, predicted 45.94, the budget
#   itself, ran out of memory, 41.60 and 4.00, 15 MiB free.
MATRIX = [
    (
        *(HD, 10.54, 53, Plan(8, Tiles(896), Tiles(640), 35, 1, 1, vae_in_dit=False)),
        *((9.66, 9.93, 9.22), None),
    ),
    (
        *(HD, 14.54, 85, Plan(12, Tiles(1216), Tiles(832), 35, 0, 0, vae_in_dit=False)),
        *((13.91, 13.95, 12.87), None),
    ),
    (
        *(HD, 22.54, 149, Plan(20, None, Tiles(1088), 36, None, 0, vae_in_dit=False), None),
        (Plan(20, Tiles(1856), Tiles(1152), 36, None, 0, vae_in_dit=False), 20.57, 21.58),
    ),
    (
        *(HD, 30.54, 113, Plan(16, None, Tiles(1536), 7, 22, 1, vae_in_dit=False), None),
        (Plan(16, None, Tiles(1600), 7, 13, 0, vae_in_dit=False), 25.56, 29.78),
    ),
    (
        *(UHD, 22.54, 17, Plan(5, Tiles(1472), Tiles(1024), 36, vae_in_dit=False)),
        *((21.26, 21.78, 19.01), None),
    ),
    (
        *(UHD, 30.54, 37, Plan(6, Tiles(1792), Tiles(1216), 27, 0, 4, vae_in_dit=False)),
        *((29.53, 29.88, 26.85), None),
    ),
    (
        *(UHD, 46.54, 41, Plan(7, Tiles(2304), Tiles(1600), 0, 0, 0), None),
        (Plan(7, Tiles(2048), Tiles(1600), 0, 17, 7), None, None),
    ),
]


@pytest.mark.parametrize(("target", "free", "frames", "ran", "measured", "again"), MATRIX)
def test_first_matrix(
    target: tuple[int, int],
    free: float,
    frames: int,
    ran: Plan,
    measured: tuple[float, float, float] | None,
    again: tuple[Plan, float | None, float | None] | None,
) -> None:
    # The corrections drawn from the two matrices: 0.4 GiB on every encode
    # (planner.ENCODE_EXTRA) and split's term off a tiled decode's peak (planner.decode_peak),
    # both PROVISIONAL (implementation, 2026-10-11); no tile chosen above 2048 px
    # (planner.TILE_MAX) and every VAE pass held to 1.5 GiB of room under the budget besides its
    # peak (planner.VAE_ROOM), DESIGN.md's (Memory planner: What follows from them; Budget,
    # provisional there): any VAE pass, encode or decode, tiled or not, ran out of memory with
    # its peak close to the free memory. Each case planned again with the memory it read; the
    # DiT's phase is the one that ran everywhere, and every VAE pass leaves 2.1 GiB or more under
    # the free memory. With the squares' measured peaks (planner.MEASURED) and what each pass
    # keeps of its shot, the encode's tiles one size down in the first case, and one block fewer
    # beside a pass in the fourth and the last:
    # - 1080p, 10.54 GiB free: encode in tiles of 768 px, the DiT away; decode in 576 px, one
    #   block of the DiT there;
    # - 1080p, 14.54: encode in tiles of 1024 px, decode in 768 px, the DiT away, its 36th
    #   block swapped;
    # - 1080p, 22.54: encode in tiles of 1856 px, 1 x 2, decode in 1088 px, 2 x 2, the DiT away;
    # - 1080p, 30.54: encode untiled with 17 of the DiT's blocks, decode in 1472 px with one;
    # - 4K, 22.54: encode in tiles of 1408 px, decode in 1088 px, the DiT away;
    # - 4K, 30.54: encode in tiles of 1728 px, the DiT away, decode in 1280 px with one block;
    # - 4K, 46.54: encode in tiles of 2048 px with 12 of the DiT's blocks, decode in 1600 px
    #   with 3.
    own = {
        (HD, 10.54): Plan(8, Tiles(768), Tiles(576), 35, 0, 1, vae_in_dit=False),
        (HD, 14.54): Plan(12, Tiles(1024), Tiles(768), 36, 0, 0, vae_in_dit=False),
        (HD, 22.54): Plan(20, Tiles(1856), Tiles(1088), 36, 0, 0, vae_in_dit=False),
        (HD, 30.54): Plan(16, None, Tiles(1472), 7, 17, 1, vae_in_dit=False),
        (UHD, 22.54): Plan(5, Tiles(1408), Tiles(1088), 36, 0, 0, vae_in_dit=False),
        (UHD, 30.54): Plan(6, Tiles(1728), Tiles(1280), 27, 0, 1, vae_in_dit=False),
        (UHD, 46.54): Plan(7, Tiles(2048), Tiles(1600), 0, 12, 3),
    }
    reported, shot, padded = round(free * GIB), [Shot(0, frames)], padded_size(target)
    latents = (padded_length(frames) - 1) // 4 + 1
    made, peaks = planner.plan(reported, target, shot, Pins(), None, split=True)
    assert bare(made) == own[target, free]
    assert planner.fits(made, reported, target, shot, None, split=True) is None
    for phase in ("encode", "decode"):
        assert reported - peaks.peak(phase) >= MARGIN + planner.VAE_ROOM
    for tiles in (made.encode_tiles, made.decode_tiles):
        assert tiles is None or tiles.size <= planner.TILE_MAX
    # The DiT's phase is the one that ran. But at 1080p with 14.54 GiB free, a budget of 13.94:
    # its 35 swapped blocks peaked at 13.95 GiB, which the model now says, 13.946, with the
    # latent and the noise of the shot's 10 latents beyond its window (planner.dit_kept), and
    # the planner swaps the 36th.
    swapped = 36 if (target, free) == (HD, 14.54) else ran.swapped
    assert (made.window, made.swapped, made.vae_in_dit) == (ran.window, swapped, ran.vae_in_dit)
    # The DiT's phase as it ran, by the model: its longest window's peak, and what the phase
    # keeps of the shot beside it.
    window = max(end - start for start, end in shot_layout(frames, ran.window))
    alone = gib(planner.dit_peak(padded, window, ran.swapped, ran.vae_in_dit))
    ran_dit = alone + gib(planner.dit_kept(padded, latents - window))

    def vae_peaks(plan: Plan) -> tuple[float, float]:
        """A plan's encode and decode peaks for the shot by the corrected model, in GiB."""
        there = (plan.resident_encode, plan.swapped), (plan.resident_decode, plan.swapped)
        encode = planner.encode_peak(padded, plan.encode_tiles, *there[0], latents=latents)
        decode = planner.decode_peak(
            padded, target, plan.decode_tiles, *there[1], split=True, latents=latents
        )
        return gib(encode), gib(decode)

    if measured is None:
        # The plan that ran out of memory in its encode is no longer one, nor the one it ran
        # again with, whose decode did, or whose encode at 4K.
        assert again is not None
        assert planner.fits(ran, reported, target, shot, None, split=True) is not None
        assert planner.fits(again[0], reported, target, shot, None, split=True) is not None
        assert (again[0].window, again[0].swapped) == (made.window, made.swapped)
        # The phases that ran the second time: the encode's predicted peak at or above the
        # measured one and within 1.05 GiB of it, the DiT's within 0.01 of it (two windows of 20
        # latents 0.007 over their 21.573, of 16 latents 0.005 under their 29.785; by the window
        # alone, 0.043 and 0.021 over). The furthest, the untiled 1080p encode with 13 blocks:
        # 26.59 GiB for 25.56 measured, 0.17 of it the moments of its 29 latents, which the
        # measured peak held too.
        if again[1] is not None and again[2] is not None:
            assert 0 <= vae_peaks(again[0])[0] - again[1] <= 1.05
            assert abs(ran_dit - again[2]) <= 0.01 < again[2] - alone
        return
    # The plans that passed, as they ran, by the corrected model: each phase's predicted peak
    # at or above the measured one and within 0.9 GiB of it. But the DiT's, within 0.012 GiB of
    # it either way: the two-window shots measured 0.008, 0.004 and 0.006 GiB over it, where
    # they were 0.020, 0.024 and 0.037 over their window's alone, and the one-window shot, which
    # keeps nothing beside it, 0.011 under (the model's, numz's fit; planner.DIT_LOADED). The
    # 1080p decode in tiles of 640 px, which the small tiles' fit left 0.13 GiB under its
    # measured 9.22, is covered: 9.37, by the line between the squares measured at 512 and 768
    # px, 0.25 over the fit, and the 14 latents of its shot, 0.03. The closest, the 4K decode in
    # tiles of 1024 px: 19.02 for 19.01 measured, its square's measured 17.46 GiB and its 5
    # latents' 0.04.
    encode, decode = vae_peaks(ran)
    assert 0 <= encode - measured[0] <= 0.9
    assert abs(ran_dit - measured[1]) <= 0.012
    assert (measured[1] - alone >= 0.02) == (latents > window)
    assert 0 <= decode - measured[2] <= 0.9
    if (target, free) == (HD, 10.54):
        assert round(decode, 2) == 9.37


def refused(*job: object, **options: object) -> str:
    with pytest.raises(PlanError) as error:
        planner.plan(*job, **options)  # pyright: ignore[reportArgumentType]
    return str(error.value)


def test_window_pin() -> None:
    # --window is the cap exactly, or refused, the refusal naming the longest window that fits,
    # the planner's own cap: with every block swapped, or with the blocks pinned.
    free = our_free(24)
    made, peaks = planner.plan(free, HD, shots(76), Pins(window=20), None)
    assert (made.window, peaks.window) == (20, 17)
    made, peaks = planner.plan(free, HD, shots(76), Pins(window=8), None)
    assert (made.window, peaks.window, made.swapped) == (8, 8, 7)
    assert refused(free, HD, shots(76), Pins(window=30), None) == (
        "--window 30: its longest window, 27 latents, takes 29.27 GiB of free VRAM with every"
        " block of the DiT swapped, the shot's own latents counted, and the driver reports 22.57"
        " GiB free: the longest window that fits is --window 17"
    )
    assert refused(free, HD, shots(76), Pins(window=12, swap_blocks=0), None) == (
        "--window 12: its longest window, 12 latents, takes 28.58 GiB of free VRAM with"
        " --swap-blocks 0, the shot's own latents counted, and the driver reports 22.57 GiB free:"
        " the longest window that fits is --window 6"
    )
    made, _ = planner.plan(free, HD, shots(76), Pins(window=6, swap_blocks=0), None)
    assert (made.window, made.swapped, made.vae_in_dit) == (6, 0, False)


def test_swap_pin() -> None:
    # --swap-blocks is the count exactly, the cap then the longest window that fits with it;
    # refused when no window fits with it, the refusal naming the fewest blocks that hold the
    # window; and above the DiT's 36.
    free = our_free(24)
    made, peaks = planner.plan(free, HD, shots(76), Pins(swap_blocks=36), None)
    assert (made.window, peaks.window, made.swapped, made.vae_in_dit) == (17, 17, 36, True)
    made, peaks = planner.plan(free, HD, shots(76), Pins(swap_blocks=0), None)
    assert (made.window, peaks.window, made.swapped, made.vae_in_dit) == (6, 6, 0, False)
    # One and two swapped blocks save nothing, and are taken as pinned.
    made, _ = planner.plan(free, HD, shots(76), Pins(swap_blocks=2), None)
    assert (made.window, made.swapped) == (6, 2)
    free = our_free(32)
    assert refused(free, UHD, shots(40), Pins(swap_blocks=0), None) == (
        "--swap-blocks 0: a window of 5 latents, the shortest this job can run, takes 37.02 GiB of"
        " free VRAM with 0 blocks swapped, the shot's own latents counted, and the driver reports"
        " 30.57 GiB free: the fewest blocks that hold it are --swap-blocks 18"
    )
    made, _ = planner.plan(free, UHD, shots(40), Pins(swap_blocks=18), None)
    assert (made.window, made.swapped, made.vae_in_dit) == (5, 18, False)
    assert refused(free, UHD, shots(40), Pins(window=6, swap_blocks=0), None) == (
        "--swap-blocks 0: a window of 6 latents, --window 6's longest, takes 41.26 GiB of free"
        " VRAM with 0 blocks swapped, the shot's own latents counted, and the driver reports 30.57"
        " GiB free: the fewest blocks that hold it are --swap-blocks 28"
    )
    assert refused(free, UHD, shots(40), Pins(swap_blocks=37), None) == (
        "--swap-blocks 37: the DiT has 36 blocks, so from 0 to 36"
    )


def test_tile_pins() -> None:
    # A tile pin is the tile exactly, or refused, naming the largest tile that fits; 0 pins an
    # untiled pass, refused likewise. A pass pinned is held to the room the planner's own is.
    free = our_free(24)
    made, _ = planner.plan(free, UHD, shots(5), Pins(), None)
    assert (made.encode_tiles, made.decode_tiles) == (Tiles(1408), Tiles(1088))
    pins = Pins(encode_tile=1024, decode_tile=1024)
    made, peaks = planner.plan(free, UHD, shots(5), pins, None)
    assert (made.encode_tiles, made.decode_tiles) == (Tiles(1024), Tiles(1024))
    assert refused(free, UHD, shots(5), Pins(decode_tile=2048), None) == (
        "--decode-tile 2048: the decode in tiles of 2048 px takes 71.18 GiB of free VRAM, the"
        " DiT off the GPU, 1.50 of them the room a VAE pass is held to, and the driver reports"
        " 22.57 GiB free: the largest tile that fits is --decode-tile 1088"
    )
    assert refused(free, UHD, shots(5), Pins(encode_tile=0), None) == (
        "--encode-tile 0: the encode untiled takes 78.42 GiB of free VRAM, the DiT off the GPU,"
        " 1.50 of them the room a VAE pass is held to, and the driver reports 22.57 GiB free: the"
        " largest tile that fits is --encode-tile 1408"
    )
    # At 1080p an untiled encode fits a 24 GB card by its peak, 21.45 GiB of free VRAM with the
    # margin, and is refused for its room.
    assert refused(free, HD, shots(12), Pins(encode_tile=0), None) == (
        "--encode-tile 0: the encode untiled takes 22.95 GiB of free VRAM, the DiT off the GPU,"
        " 1.50 of them the room a VAE pass is held to, and the driver reports 22.57 GiB free: the"
        " largest tile that fits is --encode-tile 1856"
    )
    # The planner chooses no tile above 2048 px, the largest measured: on the 96 GB card a 4K
    # decode is in tiles of 2048 px. A larger one pinned is taken where it fits by the fit, and a
    # refusal names none. The log is given a warning for each such pass, the encode's then the
    # decode's (planner.beyond_measured), for a plan taken up from a record as for one made, and
    # none for the planner's own tiles.
    made, _ = planner.plan(our_free(96), UHD, shots(5), Pins(), None)
    assert bare(made) == Plan(5, None, Tiles(2048))
    assert planner.beyond_measured(made) == []
    made, _ = planner.plan(our_free(96), UHD, shots(5), Pins(encode_tile=2304), None)
    assert bare(made) == Plan(5, Tiles(2304), Tiles(2048))
    assert planner.beyond_measured(made) == [
        "the encode in tiles of 2304 px: its predicted peak is the fit's, beyond the sizes"
        " measured (up to 2048 px), and the pass may take more"
    ]
    recorded = Plan(5, Tiles(2112), Tiles(2304))
    again, _ = planner.plan(our_free(96), UHD, shots(5), Pins(), None, recorded=recorded)
    assert (again.encode_tiles, again.decode_tiles) == (Tiles(2112), Tiles(2304))
    assert planner.beyond_measured(again) == [
        "the encode in tiles of 2112 px: its predicted peak is the fit's, beyond the sizes"
        " measured (up to 2048 px), and the pass may take more",
        "the decode in tiles of 2304 px: its predicted peak is the fit's, beyond the sizes"
        " measured (up to 2048 px), and the pass may take more",
    ]
    assert planner.beyond_measured(Plan(5, Tiles(2048), Tiles(512))) == []
    assert refused(our_free(48), UHD, shots(5), Pins(decode_tile=2304), None) == (
        "--decode-tile 2304: the decode in tiles of 2304 px takes 85.38 GiB of free VRAM, the"
        " DiT off the GPU, 1.50 of them the room a VAE pass is held to, and the driver reports"
        " 46.57 GiB free: the largest tile that fits is --decode-tile 1600"
    )
    # A tile pinned where the untiled pass would fit is taken as pinned; and 0 where it fits.
    pins = Pins(encode_tile=0, decode_tile=1024)
    made, peaks = planner.plan(our_free(96), HD, shots(40), pins, None)
    assert bare(made) == Plan(40, None, Tiles(1024))
    assert peaks.scratch == planner.scratch(OURS_1080, 157)
    # The largest tile under the padded frame's longer side is one: 1856 px at 1080p.
    made, _ = planner.plan(our_free(96), HD, shots(40), Pins(decode_tile=1856), None)
    assert made.decode_tiles == Tiles(1856) and planner.grid(OURS_1080, Tiles(1856)) == (1, 2)


def test_tile_pin_that_covers_the_frame_refused() -> None:
    # DESIGN.md, Pipeline, per shot, step 3: "A tile that covers the whole padded frame is no
    # tiling: such a pin is refused, naming 0, the untiled pass, and the planner never plans
    # one." Whatever the memory, on the 96 GB card, where the untiled pass it names fits; the
    # encode's first.
    free = our_free(96)
    pins = Pins(encode_tile=1920, decode_tile=4096)
    assert refused(free, HD, shots(40), pins, 64 * GIB) == (
        "--encode-tile 1920: a tile of 1920 px covers the whole padded frame, 1920x1104:"
        " --encode-tile 0 pins an untiled encode"
    )
    assert refused(free, HD, shots(40), Pins(encode_tile=0, decode_tile=4096), None) == (
        "--decode-tile 4096: a tile of 4096 px covers the whole padded frame, 1920x1104:"
        " --decode-tile 0 pins an untiled decode"
    )
    assert planner.covers(OURS_1080, Tiles(1920)) and not planner.covers(OURS_1080, Tiles(1856))
    made, peaks = planner.plan(free, HD, shots(40), Pins(encode_tile=0, decode_tile=0), None)
    assert made == Plan(40) and peaks.scratch == 0
    made, _ = planner.plan(free, HD, shots(40), Pins(encode_tile=1856, decode_tile=1856), None)
    assert bare(made) == Plan(40, Tiles(1856), Tiles(1856))
    # Before the size's own refusal, where it doesn't fit, and after --swap-blocks' range.
    assert refused(numz_free(16), UHD, shots(40), Pins(decode_tile=3840), None) == (
        "--decode-tile 3840: a tile of 3840 px covers the whole padded frame, 3840x2192:"
        " --decode-tile 0 pins an untiled decode"
    )
    pins = Pins(swap_blocks=37, decode_tile=3840)
    assert refused(free, UHD, shots(40), pins, None).startswith("--swap-blocks 37: the DiT has")
    # On a frame too small for a tile, every tile covers it: 0 alone is taken, where the size
    # fits, by its untiled encode and the room it is held to (test_floor).
    tiny = planner.encode_peak((128, 128), None, 0, latents=6) + planner.VAE_ROOM + MARGIN
    for pin in (512, 1024):
        assert refused(tiny, (96, 128), shots(6), Pins(decode_tile=pin), None) == (
            f"--decode-tile {pin}: a tile of {pin} px covers the whole padded frame, 128x128:"
            " --decode-tile 0 pins an untiled decode"
        )
    made, _ = planner.plan(tiny, (96, 128), shots(6), Pins(encode_tile=0, decode_tile=0), None)
    assert (made.encode_tiles, made.decode_tiles) == (None, None)
    # A recorded plan with such tiles is none that seedvr2x records: its job, resumed, is
    # refused, by the planner and by the check the command line names the manifest with.
    for recorded, said in (
        (Plan(40, Tiles(1920), Tiles(4096)), "its encode_tiles, of 1920 px"),
        (Plan(40, None, Tiles(4096)), "its decode_tiles, of 4096 px"),
    ):
        untiled = "encode" if recorded.encode_tiles else "decode"
        whole = (
            f"not a plan as seedvr2x records one: {said}, with frames padded to 1920x1104: a"
            f" tile that covers the whole frame is no tiling, and an untiled {untiled} records none"
        )
        assert refused(free, HD, shots(40), Pins(), None, recorded=recorded) == whole
        with pytest.raises(PlanError) as error:
            planner.check_recorded(recorded, shots(40), HD)
        assert str(error.value) == whole
    # In numz's padding, the tests' own, 1080p is 1088 rows, and 4K 2160x3840: the same tiles.
    planner.check_recorded(Plan(5, Tiles(3776), Tiles(1024)), shots(40), UHD, numz_padding=True)
    with pytest.raises(PlanError, match="with frames padded to 3840x2160"):
        planner.check_recorded(Plan(5, Tiles(3840)), shots(40), UHD, numz_padding=True)
    # Nor has such a plan peaks, or a sentence of fits: no pass is run in such tiles.
    with pytest.raises(ValueError, match="tiles of 1920 px over frames padded to 1920x1104"):
        planner.fits(Plan(12, Tiles(1920)), free, HD, shots(12), None)


def test_floor() -> None:
    # A size that doesn't fit is refused, never degraded, the refusal counting in free VRAM:
    # what the size takes, what the driver reports free, and the largest output that fits there
    # (DESIGN.md, Memory planner). At 4K the smallest window, 5 latents, takes 22.40 GiB of free
    # VRAM by the model, every block swapped and the VAE away: a peak of 21.80 GiB and the margin
    # (DESIGN.md: "22 to 23 GiB of free VRAM by the fit"). A shot longer than it keeps its
    # latent and its noise beside the window, 8.03 MiB a latent: 0.27 GiB for one of 40 latents,
    # which the refusal says it counts.
    assert refused(numz_free(16), UHD, shots(5), Pins(), None).startswith(
        "an output of 3840x2160 takes 22.40 GiB of free VRAM (its smallest window, 5 latents, with"
        " every block of the DiT swapped and the VAE off the GPU: a peak of 21.80 GiB, and 0.60 of"
        " margin), and the driver reports 14.56 GiB free: the largest output that fits there is"
    )
    assert refused(numz_free(16), UHD, shots(40), Pins(), None) == (
        "an output of 3840x2160 takes 22.67 GiB of free VRAM (its smallest window, 5 latents, with"
        " every block of the DiT swapped and the VAE off the GPU, the shot's own latents counted:"
        " a peak of 22.07 GiB, and 0.60 of margin), and the driver reports 14.56 GiB free: the"
        " largest output that fits there is --resolution 1664, about 2958x1664"
    )
    # The pins change nothing to it.
    pins = Pins(window=5, swap_blocks=36, decode_tile=512)
    assert refused(numz_free(16), UHD, shots(40), pins, None).startswith("an output of 3840x2160")
    # An emulated 24 GB card has that, by 0.16 GiB (22.56 GiB free; 22.57 by seedvr2x's own
    # reading): 4K runs there in windows of 5 latents, which bound the plan, while the rest of
    # the shot fits in those 0.16 GiB beside a window: a shot of 22 latents, 85 frames, its
    # other 17 latents 0.13 GiB, in tiles of 1408 and 1088 px with the DiT away; not one of 40.
    for free in (numz_free(24), our_free(24)):
        made, peaks = planner.plan(free, UHD, shots(22), Pins(), None)
        assert bare(made) == Plan(5, Tiles(1408), Tiles(1088), 36, 0, 0, vae_in_dit=False)
        assert round(peaks.dit / GIB, 2) == 21.92 and peaks.bound == "dit"
        assert peaks.dit == planner.dit_peak(OURS_4K, 5, 36, vae_in_dit=False) + planner.dit_kept(
            OURS_4K, 17
        )
        assert refused(free, UHD, shots(40), Pins(), None).startswith(
            "an output of 3840x2160 takes 22.67 GiB of free VRAM (its smallest window, 5 latents,"
            " with every block of the DiT swapped and the VAE off the GPU, the shot's own latents"
            " counted: a peak of 22.07 GiB, and 0.60 of margin), and the driver reports 22.5"
        )
    # With 0.2 GiB of it taken by anything else, the shot of 22 latents is refused, the second
    # size down fitting.
    assert refused(round(22.36 * GIB), UHD, shots(22), Pins(), None) == (
        "an output of 3840x2160 takes 22.53 GiB of free VRAM (its smallest window, 5 latents, with"
        " every block of the DiT swapped and the VAE off the GPU, the shot's own latents counted:"
        " a peak of 21.93 GiB, and 0.60 of margin), and the driver reports 22.36 GiB free: the"
        " largest output that fits there is --resolution 2128, about 3782x2128"
    )
    made, _ = planner.plan(round(22.36 * GIB), (2128, 3783), shots(22), Pins(), None)
    assert made.window == 5
    # A job whose shots are all shorter runs its longest shot's length: 3 latents on 16 GB, in
    # the largest tiles that then fit, the DiT's slots beside the encode's.
    made, _ = planner.plan(numz_free(16), UHD, shots(3, 2), Pins(), None)
    assert bare(made) == Plan(3, Tiles(1024), Tiles(768), 36, None, 0, vae_in_dit=False)
    # A portrait target is scaled by its short side too, the output said width x height.
    assert refused(numz_free(16), (3840, 2160), shots(40), Pins(), None).endswith(
        "the largest output that fits there is --resolution 1696, about 1696x3014"
    )
    # At 1080p the floor is the VAE's, its decode in 512-px tiles with the DiT away and the room
    # a VAE pass is held to: for a job whose longest shot is 40 latents, 8.60 GiB of free VRAM,
    # 5.89 the tile's measured peak and 0.08 the shot's latents, which an 8 GB card hasn't (6.56
    # GiB free on an emulated one); the DiT's smallest window takes 6.92.
    assert refused(numz_free(8), HD, shots(40), Pins(), None) == (
        "an output of 1920x1080 takes 8.60 GiB of free VRAM (its decode in tiles of 512 px, with"
        " the DiT off the GPU: a peak of 6.50 GiB, 1.50 of room a VAE pass is held to, and 0.60 of"
        " margin), and the driver reports 6.56 GiB free: the largest output that fits there is"
        " --resolution 256, about 454x256"
    )
    made, _ = planner.plan(round(8.6 * GIB), HD, shots(40), Pins(), None)
    assert bare(made) == Plan(6, Tiles(640), Tiles(512), 35, 0, 0, vae_in_dit=False)
    assert refused(GIB, HD, shots(40), Pins(), None).endswith(
        "and the driver reports 1.00 GiB free: no output size fits there"
    )
    # The floor rises with the longest shot, and with the frame under it: 8.53 GiB at 1080p for
    # a shot of 6 latents, 10.49 for one of 1,000, 9.04 then at 540p; at 4K, where the DiT's
    # smallest window is the floor, 22.40 for a shot of 5 latents and 30.20 for one of 1,000,
    # the latent and the noise of its 995 other latents 7.80 GiB beside the window.
    for target, latents, said in (
        (HD, 6, "an output of 1920x1080 takes 8.53 GiB of free VRAM (its decode in tiles"),
        (HD, 1000, "an output of 1920x1080 takes 10.49 GiB of free VRAM (its decode in tiles"),
        ((540, 960), 1000, "an output of 960x540 takes 9.04 GiB of free VRAM (its decode in"),
        (UHD, 5, "an output of 3840x2160 takes 22.40 GiB of free VRAM (its smallest window"),
        (UHD, 1000, "an output of 3840x2160 takes 30.20 GiB of free VRAM (its smallest window"),
    ):
        assert refused(4 * GIB, target, shots(latents), Pins(), None).startswith(said)
    # The same pass bounds a smaller frame: 512x512, padded to 544x512, from 8.53 GiB free with
    # a shot of 20 latents. Its untiled decode takes more, 6.48 GiB against the tile's 6.42, and
    # is no floor: pinned, it is refused under 8.58 GiB free, naming the tile.
    assert padded_size(SMALL) == (544, 512)
    room = planner.VAE_ROOM
    untiled, tiled = (
        planner.decode_peak((544, 512), SMALL, tiles, 0, 36, latents=20)
        for tiles in (None, Tiles(512))
    )
    assert (round(gib(untiled), 2), round(gib(tiled), 2)) == (6.48, 6.42)
    made, peaks = planner.plan(tiled + room + MARGIN, SMALL, shots(20), Pins(), None)
    assert (made.window, made.encode_tiles, made.decode_tiles) == (20, None, Tiles(512))
    assert (peaks.decode, peaks.bound, made.resident_decode) == (tiled, "decode", 0)
    assert refused(tiled + room + MARGIN - 1, SMALL, shots(20), Pins(), None) == (
        "an output of 512x512 takes 8.53 GiB of free VRAM (its decode in tiles of 512 px, with the"
        " DiT off the GPU: a peak of 6.43 GiB, 1.50 of room a VAE pass is held to, and 0.60 of"
        " margin), and the driver reports 8.52 GiB free: the largest output that fits there is"
        " --resolution 496, about 496x496"
    )
    made, _ = planner.plan(tiled + room + MARGIN - 1, (496, 496), shots(20), Pins(), None)
    assert (made.encode_tiles, made.decode_tiles) == (None, None)  # untiled, which fits there
    untiled_pins = Pins(encode_tile=0, decode_tile=0)
    assert refused(round(8.55 * GIB), SMALL, shots(20), untiled_pins, None) == (
        "--decode-tile 0: the decode untiled takes 8.58 GiB of free VRAM, the DiT off the GPU,"
        " 1.50 of them the room a VAE pass is held to, and the driver reports 8.55 GiB free: the"
        " largest tile that fits is --decode-tile 512"
    )
    made, peaks = planner.plan(untiled + room + MARGIN, SMALL, shots(20), untiled_pins, None)
    assert (made.decode_tiles, peaks.decode) == (None, untiled)
    # A frame little larger than the smallest tile, 528x488, padded to 512x528: its decode
    # takes less untiled, by the small tiles' fit at its area, than in a tile of 512 px, a
    # square, by its measured peak. The untiled pass is its floor, and said so; the tile, pinned
    # where it doesn't fit, is refused naming 0.
    narrow = (488, 528)
    assert padded_size(narrow) == (512, 528)
    untiled, tiled = (
        planner.decode_peak((512, 528), narrow, tiles, 0, 36, latents=20)
        for tiles in (None, Tiles(512))
    )
    assert (round(gib(untiled), 2), round(gib(tiled), 2)) == (6.35, 6.42)
    made, peaks = planner.plan(untiled + room + MARGIN, narrow, shots(20), Pins(), None)
    assert (made.encode_tiles, made.decode_tiles, peaks.bound) == (None, None, "decode")
    assert refused(untiled + room + MARGIN - 1, narrow, shots(20), Pins(), None) == (
        "an output of 528x488 takes 8.45 GiB of free VRAM (its decode untiled, with the DiT off"
        " the GPU: a peak of 6.35 GiB, 1.50 of room a VAE pass is held to, and 0.60 of margin),"
        " and the driver reports 8.44 GiB free: the largest output that fits there is"
        " --resolution 464, about 502x464"
    )
    assert refused(untiled + room + MARGIN, narrow, shots(20), Pins(decode_tile=512), None) == (
        "--decode-tile 512: the decode in tiles of 512 px takes 8.52 GiB of free VRAM, the DiT"
        " off the GPU, 1.50 of them the room a VAE pass is held to, and the driver reports 8.44"
        " GiB free: no tile fits, the untiled decode does, --decode-tile 0"
    )
    made, _ = planner.plan(tiled + room + MARGIN, narrow, shots(20), Pins(decode_tile=512), None)
    assert made.decode_tiles == Tiles(512)
    # The size named is the target's scaled, 2 px off what --resolution makes of the source: a
    # 1920x1080 source is 1066x600 at --resolution 600, and 796x448 at --resolution 448.
    assert refused(round(8.1 * GIB), (600, 1066), shots(40), Pins(), None) == (
        "an output of 1066x600 takes 8.55 GiB of free VRAM (its decode in tiles of 512 px, with"
        " the DiT off the GPU: a peak of 6.45 GiB, 1.50 of room a VAE pass is held to, and 0.60 of"
        " margin), and the driver reports 8.10 GiB free: the largest output that fits there is"
        " --resolution 448, about 794x448"
    )
    assert planner.plan(round(8.1 * GIB), (448, 796), shots(40), Pins(), None)[0].window == 21
    # A frame too small for a tile has its untiled passes for floor, by the small tiles' fit at
    # its area: the encode's on a frame this small, with the 0.4 GiB seedvr2x's encode holds
    # over its fit.
    encode = planner.encode_peak((128, 128), None, 0, 36, latents=6)
    decode = planner.decode_peak((128, 128), (96, 128), None, 0, 36, latents=6)
    assert (round(gib(encode), 2), round(gib(decode), 2)) == (2.76, 2.38)
    assert refused(encode + room + MARGIN - 1, (96, 128), shots(6), Pins(), None).startswith(
        "an output of 128x96 takes 4.87 GiB of free VRAM (its encode untiled, too small for a"
        " tile, with the DiT off the GPU: a peak of 2.77 GiB, 1.50 of room a VAE pass is held to,"
        " and 0.60 of margin)"
    )
    made, peaks = planner.plan(encode + room + MARGIN, (96, 128), shots(6), Pins(), None)
    assert (made.encode_tiles, made.decode_tiles, made.resident_encode) == (None, None, 0)
    assert (peaks.encode, peaks.bound) == (encode, "encode")


def test_least_pass_is_the_least() -> None:
    # The pass a size's floor counts for its encode and for its decode (planner._least_pass) is
    # the least of every pass the planner may make, untiled or in tiles from 512 px to the
    # largest under the frame's longer side, whatever the shot's length and with split or
    # without: over frames from 48 px a side to 4K, in numz's padding, the target's own size.
    least_pass = planner._least_pass  # pyright: ignore[reportPrivateUsage]
    vae_need = planner._vae_need  # pyright: ignore[reportPrivateUsage]
    tile_sizes = planner._tile_sizes  # pyright: ignore[reportPrivateUsage]
    job = planner._job  # pyright: ignore[reportPrivateUsage]
    heights = (48, 64, 80, 128, 144, 160, 496, 512, 528, 544, 576, 1024, 1104, 2192)
    widths = (48, 512, 528, 544, 576, 1040, 1104, 1920, 3840)
    found: dict[tuple[str, int | None], set[tuple[int, int]]] = {}
    for padded in ((height, width) for height in heights for width in widths):
        for latents in (1, 40, 1000):
            for split in (False, True):
                counted = job(GIB, padded, shots(latents), split, True)
                assert counted.padded == padded
                passes = (None, *(Tiles(size) for size in tile_sizes(padded)))
                for phase in ("encode", "decode"):
                    needs = {tiles: vae_need(counted, phase, tiles, 0, 36) for tiles in passes}
                    least = least_pass(counted, phase)
                    assert needs[least] == min(needs.values())
                    if len(passes) > 1:
                        found.setdefault((phase, least and least.size), set()).add(padded)
    # The smallest tiles in most frames. The untiled decode on a frame little larger than a
    # tile, 512x528, a square tile taking its measured peak (test_floor). And tiles of 1024 px
    # on a strip over 1024 px long, under 74 px high for the encode and 134 for the decode,
    # where the refit's line is under the small tiles'.
    assert set(found) == {
        ("encode", 512),
        ("encode", 1024),
        ("decode", 512),
        ("decode", 1024),
        ("decode", None),
    }
    assert found["decode", None] == {(512, 528), (528, 512)}
    assert {min(padded) for padded in found["encode", 1024]} == {48, 64}
    assert {min(padded) for padded in found["decode", 1024]} == {48, 64, 80, 128}
    assert len(found["encode", 512]) == 100 and len(found["decode", 512]) == 92


def needs(
    made: Plan, target: tuple[int, int], film: list[Shot], split: bool, on_disk: bool = True
) -> dict[str, int]:
    """What each phase of a plan needs under the budget for a job, in bytes, by the planner's
    public functions and the run's own layout (shot.shot_layout) alone: a VAE pass its peak for
    the longest shot and its room; the DiT's phase the largest over the shots of the shot's
    longest window and what it keeps of the shot beside it."""
    padded = padded_size(target)
    lengths = {(padded_length(shot.frames) - 1) // 4 + 1: shot.frames for shot in film}
    longest = max(lengths)
    dit = 0
    for latents, frames in lengths.items():
        sizes = [end - start for start, end in shot_layout(frames, made.window)]
        kept = planner.dit_kept(padded, latents - max(sizes), 0 if on_disk else sum(sizes[:-1]))
        window = planner.dit_peak(padded, max(sizes), made.swapped, made.vae_in_dit)
        dit = max(dit, window + kept)
    encode = planner.encode_peak(
        padded, made.encode_tiles, made.resident_encode, made.swapped, latents=longest
    )
    decode = planner.decode_peak(
        *(padded, output_size(target), made.decode_tiles, made.resident_decode, made.swapped),
        split,
        latents=longest,
    )
    return {"encode": encode + planner.VAE_ROOM, "dit": dit, "decode": decode + planner.VAE_ROOM}


def host_need(made: Plan) -> int:
    """The system memory a plan takes, in bytes, counted here: the process's own, a pinned copy
    of each part of the weights that leaves the GPU in some phase, and the scratch's allowance."""
    stay = 36 - made.swapped
    residents = (made.resident_encode, made.resident_decode)
    there = [stay if resident is None else resident for resident in residents]
    parts = [BLOCK] * (36 - min(stay, *there))
    parts += [planner.DIT_OTHER] if 0 in residents else []
    parts += [] if made.vae_in_dit else [planner.VAE_WEIGHTS]
    return planner.HOST_RESIDENT + sum(paged(part) for part in parts) + made.scratch_allowance


def test_every_plan_fits() -> None:
    # Over a grid of sizes, shots, memory and pins, with split and without, the system memory
    # unchecked, ample or short, an output directory or one file: a plan returned fits, each
    # phase's need counted here from the public functions (needs, host_need), not by the check
    # plan() itself makes (fits); its recorded peaks are those; and it takes every pin exactly.
    # A size refused names a --resolution that plans, the largest that does.
    targets = ((96, 128), SMALL, (528, 560), (600, 1066), HD, UHD)
    films = (shots(1), shots(3, 2), shots(40), shots(5, 76, 133))
    frees = (4.0, 7.3, 7.6, 10.5, 14.56, 22.56, 46.56, 94.27)
    # Each job takes one of these in turn: split, the system memory available, on disk.
    kinds = [
        (split, host, on_disk)
        for split in (False, True)
        for host in (None, 64 * GIB, 12 * GIB)
        for on_disk in (True, False)
    ]
    pin_sets = (
        Pins(),
        Pins(window=5),
        Pins(window=12, swap_blocks=36),
        Pins(swap_blocks=0),
        Pins(swap_blocks=20),
        Pins(encode_tile=0, decode_tile=0),
        Pins(decode_tile=512),
        Pins(encode_tile=1024, decode_tile=768),
        Pins(encode_tile=2304, decode_tile=2112),  # above TILE_MAX; refused on a frame they cover
    )
    plans = floors = tiled = short = jobs = 0
    for target in targets:
        for film in films:
            for free in (round(count * GIB) for count in frees):
                for pins in pin_sets:
                    for turn in range(3):
                        split, host, on_disk = kinds[(jobs + 5 * turn) % len(kinds)]
                        try:
                            made, peaks = planner.plan(
                                *(free, target, film, pins, host), split=split, on_disk=on_disk
                            )
                        except PlanError as error:
                            floors += floor_named(str(error), free, target, film, split, on_disk)
                            short += "of system memory" in str(error)
                            continue
                        plans += 1
                        need = needs(made, target, film, split, on_disk)
                        assert max(need.values()) <= free - MARGIN
                        assert need == {phase: peaks.need(phase) for phase in planner.PHASES}
                        assert host is None or host_need(made) <= host
                        assert peaks.host == host_need(made)
                    jobs += 1
                    try:
                        made, _ = planner.plan(free, target, film, pins, None)
                    except PlanError:
                        continue
                    assert pins.window in (None, made.window)
                    assert pins.swap_blocks in (None, made.swapped)
                    for pin, tiles in (
                        (pins.encode_tile, made.encode_tiles),
                        (pins.decode_tile, made.decode_tiles),
                    ):
                        assert pin is None or tiles == (Tiles(pin) if pin else None)
                        # No plan has a tile that covers the frame, pinned or not.
                        assert tiles is None or not planner.covers(padded_size(target), tiles)
                        tiled += tiles is not None
    assert plans > 1500 and floors > 1000 and tiled > 300 and short > 300


def reference(free: int, target: tuple[int, int], film: list[Shot], split: bool) -> Plan | None:
    """The unpinned plan of a job by DESIGN.md's order (Memory planner, Choices, in order),
    searched by brute force with what each phase needs (needs), no shortcut of the planner's;
    None where no plan fits. Its scratch's allowance left out (bare)."""
    budget, padded = free - MARGIN, padded_size(target)
    lengths = [(padded_length(shot.frames) - 1) // 4 + 1 for shot in film]

    def need(phase: str, plan: Plan) -> int:
        return needs(plan, target, film, split)[phase]

    # 1. The window: the largest cap that fits, every block swapped and the VAE away; the plan
    # takes the longest window the job really runs under it, 5 at least once a shot is split.
    caps = range(max(lengths), min(5, max(lengths)) - 1, -1)
    away = Plan(None, swapped=36, resident_encode=0, resident_decode=0, vae_in_dit=False)
    cap = next((cap for cap in caps if need("dit", replace(away, window=cap)) <= budget), None)
    if cap is None:
        return None
    layouts = [shot_layout(shot.frames, cap) for shot in film]
    run = max(end - start for layout in layouts for start, end in layout)
    cap = run if run == max(lengths) else max(run, 5)
    # 2. The fewest swapped blocks that hold it, the VAE's weights leaving before one more.
    swapped, vae_in_dit = next(
        (count, there)
        for count in range(37)
        for there in (True, False)
        if need("dit", Plan(cap, swapped=count, vae_in_dit=there)) <= budget
    )
    # 3. The largest tiles, none where the untiled pass fits, the DiT away; then as much of the
    # DiT left beside each pass as fits: all of it as it is, a number of its blocks, nothing.
    made = replace(away, window=cap, swapped=swapped, vae_in_dit=vae_in_dit)
    sizes = range(min(planner.TILE_MAX, (max(padded) - 1) // 64 * 64), 511, -64)
    for phase in ("decode", "encode"):
        for tiles in (None, *(Tiles(size) for size in sizes)):
            made = replace(made, **{f"{phase}_tiles": tiles})
            if need(phase, made) <= budget:
                break
        else:
            return None
        for resident in (None, *range(36 - swapped, 0, -1), 0):
            made = replace(made, **{f"resident_{phase}": resident})
            if need(phase, made) <= budget:
                break
    return made


def test_the_plan_is_the_brute_force_one() -> None:
    # The planner's unpinned plan against a brute-force search in DESIGN.md's order (reference),
    # over 96 jobs: three sizes, eight free memories from a 12 GB card's to the 96 GB card's,
    # short shots, long ones and a film's mix, with split and without; refused where the search
    # finds no plan.
    films = (shots(4), shots(12, 40, 3), shots(76), shots(300, 21, 5))
    frees = (10.54, 14.54, 22.54, 30.54, 38.54, 46.54, 62.54, 94.24)
    found = refusals = 0
    for index, target in enumerate(((540, 960), HD, UHD)):
        for film in films:
            for free in (round(count * GIB) for count in frees):
                split = bool((found + refusals + index) % 2)
                expected = reference(free, target, film, split)
                if expected is None:
                    assert refused(free, target, film, Pins(), None, split=split)
                    refusals += 1
                    continue
                made, _ = planner.plan(free, target, film, Pins(), None, split=split)
                assert bare(made) == expected, (free / GIB, target, film, split)
                found += 1
    assert found > 70 and refusals > 10


def floor_named(
    said: str,
    free: int,
    target: tuple[int, int],
    film: list[Shot],
    split: bool = False,
    on_disk: bool = True,
) -> bool:
    """Whether a refusal is the floor's; the --resolution it names then plans, the target scaled
    by its short side as the planner scales it, and the next multiple of 16 doesn't."""
    if not said.startswith("an output of"):
        return False
    named = re.search(r"is --resolution (\d+), about", said)
    if named is None:
        assert said.endswith("no output size fits there")
        return True
    short = int(named[1])
    height, width = target

    def scaled(short: int) -> tuple[int, int]:
        return (short, short * width // height)

    planner.plan(free, scaled(short), film, Pins(), None, split=split, on_disk=on_disk)
    if short + 16 < height:
        larger = (free, scaled(short + 16), film, Pins(), None)
        assert refused(*larger, split=split, on_disk=on_disk).startswith("an output of")
    return True


def test_host_memory() -> None:
    # The system memory the plan takes: the process's own 2.4 GiB, a pinned copy of each part of
    # the weights that leaves the GPU in some phase, and what a tiled shot's scratch may take of
    # it, the plan's allowance (test_scratch_allowance): here the longest shot's scratch whole.
    # The first two over what is available: refused, naming both.
    free = our_free(24)
    made, peaks = planner.plan(free, HD, shots(76), Pins(), 64 * GIB)
    assert bare(made) == Plan(17, Tiles(1856), Tiles(1088), 28, 0, 0, vae_in_dit=False)
    kept = planner.scratch(OURS_1080, 301)
    assert peaks.scratch == made.scratch_allowance == kept == 301 * 12_718_080
    # Every block leaves in the VAE's phases, the DiT's other parameters with them, and the VAE
    # in the DiT's phase: each part's bytes rounded up to a page, 15.81 GiB (measured, milestone
    # 3).
    pinned = 36 * paged(BLOCK) + paged(planner.DIT_OTHER) + paged(planner.VAE_WEIGHTS)
    assert peaks.host == planner.HOST_RESIDENT + pinned + kept
    assert f"{gib(pinned):.2f}" == "15.81" and pinned - 36 * BLOCK < 501_500_000 + 166_500_000
    # The refusal says each part as the plan's line does, to the nearest hundredth, the pinned
    # copies 15.81 GiB in both, and its total is their sum.
    assert refused(free, HD, shots(76), Pins(), 8 * GIB) == (
        "the plan takes 18.21 GiB of system memory (2.40 for the process, 15.81 pinned for 36"
        " blocks, the DiT's other parameters and the VAE's weights that leave the GPU) and 8.00"
        " GiB are available, 10.21 GiB short"
    )
    assert planner.said(made, peaks, "a card", HD, shots(76), Pins())[5] == (
        "plan: 15.81 GiB of system memory pinned for 36 blocks, the DiT's other parameters and the"
        " VAE; 21.78 GiB of system memory in all"
    )
    # A job is refused for those two alone: with no more than they take, its scratch is allowed
    # nothing, and every shot's goes to its file.
    least = planner.HOST_RESIDENT + pinned
    scarce, on_disk = planner.plan(free, HD, shots(76), Pins(), least)
    assert scarce == bare(made) and (on_disk.host, on_disk.scratch) == (least, kept)
    # One byte less is refused, 18.2144 GiB taken where a byte less is there: what is there is
    # said a hundredth under what is taken, which the two rounded alike would hide.
    assert refused(free, HD, shots(76), Pins(), least - 1) == (
        "the plan takes 18.21 GiB of system memory (2.40 for the process, 15.81 pinned for 36"
        " blocks, the DiT's other parameters and the VAE's weights that leave the GPU) and 18.20"
        " GiB are available, 0.01 GiB short"
    )
    assert planner.plan(free, HD, shots(76), Pins(), None) == (made, peaks)  # not checked
    # Nothing leaves the GPU, nothing tiled: the process alone.
    made, peaks = planner.plan(our_free(96), HD, shots(40), Pins(), 4 * GIB)
    assert made == Plan(40) and peaks.host == planner.HOST_RESIDENT and peaks.scratch == 0
    assert refused(our_free(96), HD, shots(40), Pins(), 2 * GIB) == (
        "the plan takes 2.40 GiB of system memory (2.40 for the process) and 2.00 GiB are"
        " available, 0.40 GiB short"
    )
    # A long tiled shot's scratch counts up to its allowance: 4K, 601 frames, 28.27 GiB, over
    # half of what a host with 64 GiB available has left of them, 27.97, so that it goes whole
    # to its memory-mapped file (tiling.Scratch). Nothing moves between the phases: the 20
    # swapped blocks alone are pinned, with the VAE.
    made, peaks = planner.plan(our_free(96), UHD, shots(151), Pins(), 64 * GIB)
    assert bare(made) == Plan(21, None, Tiles(2048), 20, vae_in_dit=False)
    pinned = 20 * paged(BLOCK) + paged(planner.VAE_WEIGHTS)
    assert peaks.scratch == 601 * 50_503_680 > made.scratch_allowance == (64 * GIB - pinned) // 2
    assert peaks.host == planner.HOST_RESIDENT + pinned + made.scratch_allowance
    # Two swapped blocks save nothing in the DiT's phase, their slots there, and wait in pinned
    # copies all the same, with the 14 more the untiled decode sends away, which frees their
    # slots: 20 blocks stay beside it, as with none swapped.
    made, peaks = planner.plan(our_free(48), HD, shots(12), Pins(swap_blocks=2), None)
    assert made == Plan(12, swapped=2, resident_decode=20)
    assert peaks.host == planner.HOST_RESIDENT + 16 * paged(BLOCK)
    unswapped, _ = planner.plan(our_free(48), HD, shots(12), Pins(swap_blocks=0), None)
    assert unswapped == Plan(12, resident_decode=20)


def test_scratch_allowance() -> None:
    # DESIGN.md, Memory planner, Host RAM: "The planner sets the scratch's allowance per job:
    # half of what the host has available once the pinned copies are counted, and no more than
    # the longest tiled shot needs"; none where no pass is tiled. 1080p on a 24 GB card, both
    # passes tiled, every part of the weights pinned, 15.81 GiB: the longest shot's 301 frames
    # take 3.57 GiB.
    free, film = our_free(24), shots(5, 76, 17)
    pinned = 36 * paged(BLOCK) + paged(planner.DIT_OTHER) + paged(planner.VAE_WEIGHTS)
    kept = planner.scratch(OURS_1080, 301)

    def allowed(host: int | None) -> int:
        made, peaks = planner.plan(free, HD, film, Pins(), host)
        assert bare(made) == Plan(17, Tiles(1856), Tiles(1088), 28, 0, 0, vae_in_dit=False)
        assert peaks.scratch == kept
        assert peaks.host == planner.HOST_RESIDENT + pinned + made.scratch_allowance
        assert host is None or peaks.host <= host
        return made.scratch_allowance

    assert allowed(64 * GIB) == allowed(23 * GIB) == kept
    assert allowed(22 * GIB) == (22 * GIB - pinned) // 2 < kept
    assert f"{gib(allowed(22 * GIB)):.2f}" == "3.09"
    # No more than what the process's own 2.4 GiB and the pinned copies leave, which half of
    # what the copies leave passes under 4.8 GiB (PROVISIONAL): down to nothing, the job never
    # refused for its allowance.
    assert allowed(19 * GIB) == 19 * GIB - pinned - planner.HOST_RESIDENT < (19 * GIB - pinned) // 2
    assert f"{gib(allowed(19 * GIB)):.2f}" == "0.79"
    assert allowed(planner.HOST_RESIDENT + pinned) == 0
    # What the host has available unknown: the longest shot's scratch, 8 GiB at most
    # (PROVISIONAL), reached by 601 frames at 4K.
    assert allowed(None) == kept and planner.ALLOWANCE_UNKNOWN_HOST == 8 * GIB
    made, peaks = planner.plan(our_free(96), UHD, shots(151), Pins(), None)
    assert (made.scratch_allowance, peaks.scratch) == (8 * GIB, 601 * 50_503_680)
    # Nothing tiled, nothing allowed, whatever the host has.
    made, _ = planner.plan(our_free(96), HD, film, Pins(), 64 * GIB)
    assert made == Plan(76) and made.scratch_allowance == 0
    # It shapes nothing, and is recorded with the memory, for information; a resume plans it
    # again, from what the host has that day, whatever was recorded.
    made, _ = planner.plan(free, HD, film, Pins(), 22 * GIB)
    assert made.record()["memory"]["scratch_allowance"] == (22 * GIB - pinned) // 2
    assert Plan.recorded(made.record()) == Plan(17, Tiles(1856), Tiles(1088))
    again, _ = planner.plan(free, HD, film, Pins(), 64 * GIB, recorded=made)
    assert again == replace(made, scratch_allowance=kept)
    # The plan's lines say it, and the shots whose scratch goes to its file instead, whole: of
    # 5, 76 and 17 latents, the one over 3.09 GiB.
    made, peaks = planner.plan(free, HD, film, Pins(), 64 * GIB)
    assert planner.said(made, peaks, "a card", HD, film, Pins())[-2] == (
        "plan: scratch of the longest tiled shot: 3.57 GiB; an allowance of 3.57 GiB of system"
        " memory, which holds every shot's"
    )
    made, peaks = planner.plan(free, HD, film, Pins(), 22 * GIB)
    assert planner.said(made, peaks, "a card", HD, film, Pins())[-2] == (
        "plan: scratch of the longest tiled shot: 3.57 GiB; an allowance of 3.09 GiB of system"
        " memory, over which the scratch of 1 of 3 shots goes whole to a memory-mapped file"
    )
    made, peaks = planner.plan(free, HD, film, Pins(), planner.HOST_RESIDENT + pinned)
    assert planner.said(made, peaks, "a card", HD, film, Pins())[-2] == (
        "plan: scratch of the longest tiled shot: 3.57 GiB; an allowance of 0.00 GiB of system"
        " memory, over which the scratch of 3 of 3 shots goes whole to a memory-mapped file"
    )
    # A plan from elsewhere is judged with the allowance it has (fits), said with the rest.
    large = replace(made, scratch_allowance=6 * GIB)
    assert planner.fits(large, free, HD, film, 24 * GIB) == (
        "the plan takes 24.21 GiB of system memory (2.40 for the process, 15.81 pinned for 36"
        " blocks, the DiT's other parameters and the VAE's weights that leave the GPU, 6.00 for a"
        " tiled shot's scratch) and 24.00 GiB are available, 0.21 GiB short"
    )
    assert planner.fits(large, free, HD, film, 25 * GIB) is None
    # The plan's lines say the same figures, and "in all" the refusal's total, their sum.
    assert planner.said(large, peaks, "a card", HD, film, Pins())[5::3] == [
        "plan: 15.81 GiB of system memory pinned for 36 blocks, the DiT's other parameters and the"
        " VAE; 24.21 GiB of system memory in all",
        "plan: scratch of the longest tiled shot: 3.57 GiB; an allowance of 6.00 GiB of system"
        " memory, which holds every shot's",
    ]


def paged(size: int) -> int:
    """`size` bytes rounded up to a page, as a part's pinned copy takes them
    (blockswap.HostCopy)."""
    import mmap

    return -(-size // mmap.PAGESIZE) * mmap.PAGESIZE


def test_pinned_is_the_loaders() -> None:
    # What the plan counts as pinned is what the loader pins for it (model.load_models): the
    # parts model._dit_parts names of the DiT's parameters, each block that is swapped or that a
    # VAE phase doesn't keep, and the other parameters when one keeps nothing; and the VAE's
    # weights when they leave the DiT's phase. Over the plans the planner makes for many jobs,
    # and others it could: every count the loader accepts.
    from seedvr2x.runtime import blockswap, model

    leaving = planner._leaving  # pyright: ignore[reportPrivateUsage]
    pinned = planner._pinned  # pyright: ignore[reportPrivateUsage]
    keys = [
        *(f"blocks.{index}.attn.weight" for index in range(planner.BLOCKS)),
        "vid_in.proj.weight",
    ]
    plans = [
        Plan(5, swapped=swapped, resident_encode=encode, resident_decode=decode, vae_in_dit=vae)
        for swapped in (0, 1, 2, 3, 20, 35, 36)
        for encode in (None, 0, 1, 36 - swapped)
        for decode in (None, 0, 36 - swapped)
        for vae in (True, False)
        if (encode is None or encode <= 36 - swapped)
    ]
    for card in (8, 12, 16, 24, 32, 48, 96):
        for target, film in ((HD, shots(5, 76)), (UHD, shots(40)), (SMALL, shots(20))):
            try:
                plans.append(planner.plan(our_free(card), target, film, Pins(), None)[0])
            except PlanError:
                continue
    kinds: set[tuple[bool, bool, bool]] = set()
    for plan in plans:
        part_of = model._dit_parts(plan, planner.BLOCKS)  # pyright: ignore[reportPrivateUsage]
        parts = set() if part_of is None else {part_of(key) for key in keys} - {None}
        blocks = len(parts - {blockswap.OTHER})
        other = blockswap.OTHER in parts
        assert leaving(plan) == blocks, plan
        expected = blocks * paged(BLOCK) + other * paged(planner.DIT_OTHER)
        expected += 0 if plan.vae_in_dit else paged(planner.VAE_WEIGHTS)
        assert pinned(plan) == expected, plan
        kinds.add((blocks > 0, other, plan.vae_in_dit))
    assert len(plans) > 150 and len(kinds) == 6  # no other parameters pinned without a block


def test_resumed_around_the_recorded_shape() -> None:
    # A resume takes the recorded shape, cap, tiles and compile flags, as it is, and plans where
    # the weights are again from the memory free that day (DESIGN.md, Memory planner, A plan
    # shapes the output a little): with more, fewer blocks swapped; with less, more.
    recorded = Plan(17, Tiles(1856), Tiles(1088), 28, 0, 0, vae_in_dit=False)
    job = (HD, shots(76), Pins(), None)
    shape = Plan(17, Tiles(1856), Tiles(1088))
    assert bare(planner.plan(our_free(24), *job)[0]) == recorded
    assert bare(planner.plan(our_free(24), *job, recorded=shape)[0]) == recorded
    made, peaks = planner.plan(our_free(48), *job, recorded=recorded)
    assert bare(made) == shape
    assert (peaks.window, peaks.free, peaks.bound) == (17, our_free(48), "decode")
    # Planned whole, that card has no tiles and two windows of 39 latents: the shape is the
    # record's.
    whole, peaks = planner.plan(our_free(48), *job)
    assert whole == Plan(39, swapped=24, vae_in_dit=False) and peaks.window == 39
    # A cap recorded above the windows it lays out, as a pinned one is, stays as recorded.
    made, peaks = planner.plan(our_free(24), *job, recorded=replace(shape, window=20))
    assert bare(made) == replace(recorded, window=20) and peaks.window == 17
    # A shape recorded with smaller tiles, which leave room, nothing then moving for the encode:
    # with 0.58 GiB less, two more blocks swapped, and two fewer left in the decode.
    smaller = Plan(17, Tiles(1088), Tiles(1024))
    made, _ = planner.plan(our_free(24), *job, recorded=smaller)
    assert bare(made) == Plan(17, Tiles(1088), Tiles(1024), 28, None, 5, vae_in_dit=False)
    made, _ = planner.plan(round(21.99 * GIB), *job, recorded=smaller)
    assert bare(made) == Plan(17, Tiles(1088), Tiles(1024), 30, None, 3, vae_in_dit=False)
    # The recorded tiles themselves take 22.43 GiB of free VRAM with the DiT away, their room
    # counted and the sums of the shot's 76 latents, 0.30: with less, the shape no longer fits.
    assert refused(round(21.99 * GIB), *job, recorded=recorded) == (
        "the recorded plan no longer fits the memory free now: the encode, in tiles of 1856 px,"
        " with the DiT off the GPU, takes 22.43 GiB of free VRAM, 1.50 of them the room a VAE pass"
        " is held to, and the driver reports 21.99 GiB free, 0.44 GiB short"
    )
    # Its tiles too, their overlap with them, and its compile flags, whatever would be chosen.
    odd = Plan(8, Tiles(768, 128), Tiles(576, 96), compile_dit=True)
    made, _ = planner.plan(our_free(96), *job, recorded=odd)
    assert bare(made) == odd
    # A recorded shape that no longer fits stops the resume, naming the phase and the shortfall,
    # and never another pin: here its encode, then its windows.
    assert refused(our_free(16), *job, recorded=recorded) == (
        "the recorded plan no longer fits the memory free now: the encode, in tiles of 1856 px,"
        " with the DiT off the GPU, takes 22.43 GiB of free VRAM, 1.50 of them the room a VAE pass"
        " is held to, and the driver reports 14.57 GiB free, 7.86 GiB short"
    )
    # A pass recorded is held to its room as one planned is: on 24 GB at 1080p an untiled
    # encode's peak fits with the DiT away, and the shape doesn't.
    assert refused(our_free(24), *job, recorded=Plan(17, None, Tiles(1088))) == (
        "the recorded plan no longer fits the memory free now: the encode, untiled, with the"
        " DiT off the GPU, takes 23.33 GiB of free VRAM, 1.50 of them the room a VAE pass is held"
        " to, and the driver reports 22.57 GiB free, 0.76 GiB short"
    )
    tiled = Plan(20, Tiles(768), Tiles(640))
    assert refused(our_free(16), *job, recorded=tiled) == (
        "the recorded plan no longer fits the memory free now: the DiT's phase, a window of 17"
        " latents with 36 blocks swapped and the VAE off the GPU, the shot's own latents counted,"
        " takes 19.17 GiB of free VRAM and the driver reports 14.57 GiB free, 4.60 GiB short"
    )
    assert refused(our_free(24), HD, shots(76), Pins(), 8 * GIB, recorded=recorded) == (
        "the recorded plan no longer fits the memory free now: the plan takes 18.21 GiB of system"
        " memory (2.40 for the process, 15.81 pinned for 36 blocks, the DiT's other parameters"
        " and the VAE's weights that leave the GPU) and 8.00 GiB are available, 10.21 GiB short"
    )
    # --swap-blocks still applies, taken exactly; where it alone doesn't hold the recorded
    # windows, it is the pin that is refused.
    made, _ = planner.plan(our_free(24), HD, shots(76), Pins(swap_blocks=36), None, recorded=tiled)
    assert bare(made) == Plan(20, Tiles(768), Tiles(640), 36)
    pins = Pins(swap_blocks=3)
    assert refused(our_free(24), HD, shots(76), pins, None, recorded=recorded) == (
        "--swap-blocks 3: a window of 17 latents, the recorded plan's, takes 33.09 GiB of free"
        " VRAM with 3 blocks swapped, the shot's own latents counted, and the driver reports 22.57"
        " GiB free: the fewest blocks that hold it are --swap-blocks 28"
    )
    # The other pins were the record's when it was made: the shape is taken from it alone.
    pins = Pins(window=5, encode_tile=512, decode_tile=512)
    made, _ = planner.plan(our_free(24), HD, shots(76), pins, None, recorded=recorded)
    assert bare(made) == recorded


def test_recorded_cap_checked() -> None:
    # A cap under 5 latents is the planner's own for a job whose longest shot is as short, each
    # shot one window; with a longer shot no layout exists for it (shot.window_layout), and a
    # record that has one was edited: refused, where the layout would raise.
    made, _ = planner.plan(our_free(96), HD, shots(3, 2), Pins(), None)
    assert made.window == 3
    planner.check_recorded(made, shots(3, 2), HD)
    planner.check_recorded(Plan(4), shots(4, 1), HD)
    planner.check_recorded(Plan(5), shots(76), HD)
    planner.check_recorded(Plan(None), shots(76), HD)
    with pytest.raises(PlanError) as error:
        planner.check_recorded(Plan(3), shots(3, 6), HD)
    assert str(error.value) == (
        "not a plan as seedvr2x records one: its window, 3, with a shot of 6 latents: a cap that"
        " a shot is longer than is 5 latents at least"
    )
    with pytest.raises(ValueError, match="windows of 3 latents leave none of their own"):
        shot_layout(21, 3)
    with pytest.raises(PlanError, match="its window, 4, with a shot of 5 latents"):
        planner.check_recorded(Plan(4), shots(5), HD)


def test_fits() -> None:
    # A plan's phases against the budget, in a shot's order, the first that doesn't fit named
    # with what it is short of; then the system memory.
    free, job = our_free(24), (HD, shots(12), None)
    # A VAE pass is judged with its room, which the sentence says.
    assert planner.fits(Plan(12), free, *job) == (
        "the encode, untiled, with the DiT's 36 blocks on the GPU, takes 38.30 GiB of free VRAM,"
        " 1.50 of them the room a VAE pass is held to, and the driver reports 22.57 GiB free, 15.73"
        " GiB short"
    )
    # Nothing moving, the slots stay with the blocks that aren't swapped; a phase entered with a
    # number of blocks, all of them here, frees them: two blocks' 0.84 GiB less.
    assert planner.fits(Plan(12, swapped=17), free, *job) == (
        "the encode, untiled, with the DiT's 19 unswapped blocks and its 2 slots on the GPU, takes"
        " 31.97 GiB of free VRAM, 1.50 of them the room a VAE pass is held to, and the driver"
        " reports 22.57 GiB free, 9.40 GiB short"
    )
    assert planner.fits(Plan(12, swapped=17, resident_encode=19), free, *job) == (
        "the encode, untiled, with 19 of the DiT's blocks on the GPU, takes 31.13 GiB of free"
        " VRAM, 1.50 of them the room a VAE pass is held to, and the driver reports 22.57 GiB free,"
        " 8.56 GiB short"
    )
    away = Plan(12, Tiles(1600), None, 17, 0, 0, vae_in_dit=False)
    assert planner.fits(away, free, *job) == (
        "the decode, untiled, with the DiT off the GPU, takes 37.58 GiB of free VRAM, 1.50 of"
        " them the room a VAE pass is held to, and the driver reports 22.57 GiB free, 15.01 GiB"
        " short"
    )
    assert planner.fits(replace(away, swapped=3), free, *job) == (
        "the DiT's phase, a window of 12 latents with 3 blocks swapped and the VAE off the GPU,"
        " takes 28.04 GiB of free VRAM and the driver reports 22.57 GiB free, 5.47 GiB short"
    )
    tiled = replace(away, decode_tiles=Tiles(1088))
    assert planner.fits(tiled, free, *job) is None
    # The system memory, with what the plan allows its scratch: nothing here, then the shot's.
    assert planner.fits(tiled, free, HD, shots(12), 4 * GIB) == (
        "the plan takes 18.21 GiB of system memory (2.40 for the process, 15.81 pinned for 36"
        " blocks, the DiT's other parameters and the VAE's weights that leave the GPU) and 4.00"
        " GiB are available, 14.21 GiB short"
    )
    allowed = replace(tiled, scratch_allowance=planner.scratch(OURS_1080, 45))
    assert planner.fits(allowed, free, *job) is None
    assert planner.fits(allowed, free, HD, shots(12), 4 * GIB) == (
        "the plan takes 18.74 GiB of system memory (2.40 for the process, 15.81 pinned for 36"
        " blocks, the DiT's other parameters and the VAE's weights that leave the GPU, 0.53 for a"
        " tiled shot's scratch) and 4.00 GiB are available, 14.74 GiB short"
    )
    # One block left unswapped is one block, and with every block swapped the slots alone stay,
    # with the DiT's other parameters.
    assert planner.fits(Plan(12, Tiles(1600), swapped=35), free, *job) == (
        "the decode, untiled, with the DiT's 1 unswapped block and its 2 slots on the GPU, takes"
        " 39.00 GiB of free VRAM, 1.50 of them the room a VAE pass is held to, and the driver"
        " reports 22.57 GiB free, 16.43 GiB short"
    )
    assert planner.fits(Plan(12, Tiles(1600), swapped=36), free, *job) == (
        "the decode, untiled, with the DiT's 2 slots on the GPU, takes 38.58 GiB of free VRAM,"
        " 1.50 of them the room a VAE pass is held to, and the driver reports 22.57 GiB free, 16.01"
        " GiB short"
    )
    assert planner.fits(replace(tiled, resident_decode=1), free, *job) == (
        "the decode, in tiles of 1088 px, with 1 of the DiT's blocks on the GPU, takes 22.95 GiB"
        " of free VRAM, 1.50 of them the room a VAE pass is held to, and the driver reports 22.57"
        " GiB free, 0.38 GiB short"
    )
    # With split, its slice is an untiled decode's too: on 48 GB 20 blocks fit beside it
    # without it, 19 with it. A tiled decode's peak doesn't hold it: the same plan fits.
    assert planner.fits(tiled, free, *job, split=True) is None
    beside = Plan(12, resident_decode=20)
    assert planner.fits(beside, our_free(48), *job) is None
    assert planner.fits(beside, our_free(48), *job, split=True) == (
        "the decode, untiled, with 20 of the DiT's blocks on the GPU, takes 46.88 GiB of free"
        " VRAM, 1.50 of them the room a VAE pass is held to, and the driver reports 46.57 GiB free,"
        " 0.31 GiB short"
    )
    assert planner.fits(replace(beside, resident_decode=19), our_free(48), *job, split=True) is None


def test_peaks_record() -> None:
    # What the manifest keeps of a plan's memory, for information: bytes, the window in latents.
    made, peaks = planner.plan(our_free(24), HD, shots(76), Pins(), 64 * GIB)
    assert peaks.record() == {
        "free": our_free(24),
        "budget": our_free(24) - MARGIN,
        "encode_peak": planner.encode_peak(OURS_1080, Tiles(1856), 0, 28, latents=76),
        # The window's, and the latent and the noise of the shot's 59 other latents beside it.
        "dit_peak": planner.dit_peak(OURS_1080, 17, 28, vae_in_dit=False)
        + planner.dit_kept(OURS_1080, 59),
        "decode_peak": planner.decode_peak(OURS_1080, HD, Tiles(1088), 0, 28, latents=76),
        "bound": "dit",
        "host": peaks.host,
        "scratch": planner.scratch(OURS_1080, 301),
        "longest_window": 17,
    }
    assert [peaks.peak(phase) for phase in planner.PHASES] == [
        peaks.encode,
        peaks.dit,
        peaks.decode,
    ]
    # The phase that bounds counts a VAE phase with the room it is held to, as the planner
    # judged it: a decode at the DiT's peak bounds the plan.
    assert (
        peaks.need("dit") == peaks.dit and peaks.need("decode") == peaks.decode + planner.VAE_ROOM
    )
    under = replace(peaks, encode=0)
    assert replace(under, decode=peaks.dit - planner.VAE_ROOM + 1).bound == "decode"
    assert replace(under, decode=peaks.dit - planner.VAE_ROOM).bound == "dit"
    assert set(peaks.record()).isdisjoint(made.record()["memory"])
    assert set(made.record()["memory"]) == {
        "swapped_blocks",
        "resident_encode",
        "resident_decode",
        "vae_in_dit",
        "scratch_allowance",
    }


def test_said() -> None:
    # The plan's lines, as --plan prints them and every run logs them (DESIGN.md, Memory planner,
    # --plan): 1080p on a 24 GB card, a shot of 301 frames, with split.
    job = (HD, shots(76), Pins())
    made, peaks = planner.plan(numz_free(24), *job, 64 * GIB, split=True)
    assert planner.said(made, peaks, "NVIDIA RTX PRO 6000 Blackwell", *job) == [
        "plan: NVIDIA RTX PRO 6000 Blackwell, 22.56 GiB free, a budget of 21.96 GiB",
        "plan: frames padded to 1920x1104, 8,280 tokens a latent",
        "plan: 1 shot, the longest of 301 frames (76 latents)",
        "plan: windows of 17 latents at most: 0 of 1 shot in one window, 4 joins",
        "plan: 28 of the DiT's 36 blocks swapped, the VAE off the GPU in the DiT's phase; in the"
        " encode, the DiT off the GPU; in the decode, the DiT off the GPU",
        "plan: 15.81 GiB of system memory pinned for 36 blocks, the DiT's other parameters and the"
        " VAE; 21.78 GiB of system memory in all",
        "plan: encode in tiles of 1856 px, 1 x 2, decode in tiles of 1088 px, 2 x 2, each held"
        " 1.50 GiB under the budget",
        "plan: predicted peaks: encode 20.33 GiB, DiT 21.94 GiB, decode 20.40 GiB; the DiT's phase"
        " bounds the plan, the shot's own latents counted, 0.02 GiB under the budget",
        "plan: scratch of the longest tiled shot: 3.57 GiB; an allowance of 3.57 GiB of system"
        " memory, which holds every shot's",
        "plan: nothing pinned",
    ]
    # 4K on the 96 GB card as DESIGN.md first read it, 94.34 GiB free, the same shot, pinned:
    # the untiled encode bounds the plan with the room a VAE pass is held to, which the line
    # says; the peaks are the predicted ones, without the room. Its moments, 1.77 GiB for the
    # shot's 76 latents, send 3 of the DiT's blocks away for it.
    pins = Pins(window=19, swap_blocks=0, encode_tile=0, decode_tile=2048)
    made, peaks = planner.plan(round(94.34 * GIB), UHD, shots(76), pins, 64 * GIB, split=True)
    assert planner.said(made, peaks, "a card", UHD, shots(76), pins)[3:] == [
        "plan: windows of 19 latents at most, the longest run 17 latents: 0 of 1 shot in one"
        " window, 4 joins",
        "plan: no block of the DiT swapped, the VAE on the GPU in the DiT's phase; in the encode,"
        " 33 of the DiT's blocks on the GPU; in the decode, the DiT's 36 blocks on the GPU",
        "plan: 1.27 GiB of system memory pinned for 3 blocks; 17.83 GiB of system memory in all",
        "plan: encode untiled, decode in tiles of 2048 px, 2 x 2, each held 1.50 GiB under the"
        " budget",
        "plan: predicted peaks: encode 92.05 GiB, DiT 85.18 GiB, decode 84.98 GiB; the encode"
        " bounds the plan, its room counted, 0.19 GiB under the budget",
        "plan: scratch of the longest tiled shot: 14.16 GiB; an allowance of 14.16 GiB of system"
        " memory, which holds every shot's",
        "plan: pinned: --window 19, --swap-blocks 0, --encode-tile 0, --decode-tile 2048",
    ]
    # A recorded plan without a cap, each shot in one window; one block left unswapped, and
    # none; what leaves the GPU in some phase, pinned; and a job of one frame, its cap one
    # latent.
    job = (HD, shots(40), Pins())
    made, peaks = planner.plan(our_free(96), *job, None, recorded=Plan(None))
    assert planner.said(made, peaks, "a card", *job)[3] == (
        "plan: no cap on the windows, the longest run 40 latents: 1 of 1 shot in one window,"
        " 0 joins"
    )
    made, peaks = planner.plan(our_free(96), *job, None, recorded=Plan(40))
    said = planner.said(replace(made, swapped=35), peaks, "a card", *job)
    assert said[4].endswith(
        "in the encode, the DiT's 1 unswapped block and its 2 slots on the GPU; in the decode,"
        " the DiT's 1 unswapped block and its 2 slots on the GPU"
    )
    said = planner.said(replace(made, swapped=36), peaks, "a card", *job)
    assert said[4].endswith(
        "in the encode, the DiT's 2 slots on the GPU; in the decode, the DiT's 2 slots on the GPU"
    )
    away = replace(made, swapped=20, resident_encode=16, resident_decode=0)
    assert planner.said(away, peaks, "a card", *job)[4:6] == [
        "plan: 20 of the DiT's 36 blocks swapped, the VAE on the GPU in the DiT's phase; in the"
        " encode, 16 of the DiT's blocks on the GPU; in the decode, the DiT off the GPU",
        "plan: 15.35 GiB of system memory pinned for 36 blocks and the DiT's other parameters;"
        " 17.75 GiB of system memory in all",
    ]
    job = (HD, shots(1), Pins())
    made, peaks = planner.plan(our_free(96), *job, None)
    assert planner.said(made, peaks, "a card", *job)[2:4] == [
        "plan: 1 shot, the longest of 1 frame (1 latent)",
        "plan: windows of 1 latent at most: 1 of 1 shot in one window, 0 joins",
    ]
    # A cap no shot reaches: each in one window, no join.
    job = (HD, shots(3, 40, 2), Pins())
    made, peaks = planner.plan(our_free(96), *job, None)
    assert planner.said(made, peaks, "a card", *job)[2:4] == [
        "plan: 3 shots, the longest of 157 frames (40 latents)",
        "plan: windows of 40 latents at most: 3 of 3 shots in one window, 0 joins",
    ]

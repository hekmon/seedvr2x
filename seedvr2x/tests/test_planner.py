"""The memory planner (runtime/planner.py; DESIGN.md, Memory planner). Its model: each phase's
predicted peak against the measurements it was fitted on, every expected figure copied from
research/docs/planner-limits.md, research/docs/vram.md or DESIGN.md, the table named, or
measured on seedvr2x with milestone 3 (2026-10-11), said so; where the weights are and what
they take against runtime/blockswap.py's own rules; the tiles' measured peaks, and what a tile
nobody measured takes. Its plans: DESIGN.md's window figures, the order of its choices, vram.md's
recipes per card size, the pins taken or refused, the floor, the DiT's phase judged by its
longest window whatever the shots' lengths, the system memory, and a resume's plan around a
recorded shape. On the CPU, no weights: the models are built on the meta device where their
shapes are counted."""

import re
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import test_regression

from seedvr2x.runtime import planner, tilegrid, weights
from seedvr2x.runtime.job import Shot, output_size, padded_size
from seedvr2x.runtime.memory import Reading
from seedvr2x.runtime.plan import Phase, Plan, Tiles
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
    # The blocks, the slots and the DiT's other parameters are the model's own constants
    # (runtime/weights.py), the 7B's, the one DiT seedvr2x runs; a block is one size, which the
    # planner counts the 36 by.
    assert planner.DIT is weights.ACCEPTED["dit"] is weights.DIT_7B
    assert planner.DIT.blocks == (planner.BLOCK,) * planner.BLOCKS
    assert (planner.SLOTS, planner.DIT_OTHER) == (weights.SLOTS, weights.DIT_7B.other)
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
    # px, and 1.25 GiB left under the budget by every VAE pass besides its peak (DESIGN.md, Memory
    # planner: What follows from them; Budget; 1.5 until 2026-10-18, provisional then).
    assert gib(planner.ENCODE_EXTRA) == pytest.approx(0.4)
    assert planner.TILE_MAX == 2048 and gib(planner.VAE_ROOM) == pytest.approx(1.25)


IMPORTED = """
import sys

from seedvr2x.runtime import planner
from seedvr2x.runtime.plan import Tiles

padded = (1104, 1920)
assert planner.encode_peak(padded, Tiles(1024)) > planner.encode_peak(padded, Tiles(512))
print(*sorted(name for name in ("torch", "numpy", "torchvision") if name in sys.modules))
"""


def test_imports_no_torch() -> None:
    # The command line imports the planner before its first refusals, which don't wait for torch
    # (tests/test_cli.py): a process that has imported it, with the models' constants
    # (runtime/weights.py) and the reading's type (runtime/memory.py), and counted a tile's peak
    # over the tiled passes' grid (runtime/tilegrid.py), has loaded no torch. Planning a job
    # does load it, for the shots' layouts (runtime/shot.py).
    result = subprocess.run(
        [sys.executable, "-c", IMPORTED], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


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


def largest(padded: tuple[int, int], tiles: Tiles) -> tuple[int, int]:
    """(rows, columns), in pixels, of the largest tile of a tiled pass over padded frames: the
    first of the shapes the planner counts a pass by."""
    return planner._shapes(padded, tiles)[0]  # pyright: ignore[reportPrivateUsage]


def test_grid_and_largest_tile() -> None:
    # The grid the plan's lines say and the tile a pass's peak follows are the tiled passes' own
    # (runtime/tilegrid.py, which tests/test_tilegrid.py and tests/test_tiling.py hold to
    # numz's). planner-limits.md, "Tiled VAE at 4K", Tiles (rows x columns), overlap 128: 1024 px
    # 3 x 5, 1536 px 2 x 3, 2048 px 2 x 2; vram.md, Tiling: 60 tiles of 512 px and 24 of 768 at
    # overlap 128, 12 of 1024 at overlap 64, 15 at 128 and at 256.
    grid = planner._grid  # pyright: ignore[reportPrivateUsage]
    assert grid(NUMZ_4K, Tiles(1024, 128)) == (3, 5)
    assert grid(NUMZ_4K, Tiles(1536, 128)) == (2, 3)
    assert grid(NUMZ_4K, Tiles(2048, 128)) == (2, 2)
    assert grid(NUMZ_4K, Tiles(512, 128)) == (6, 10)
    assert grid(NUMZ_4K, Tiles(768, 128)) == (4, 6)
    assert grid(NUMZ_4K, Tiles(1024, 64)) == (3, 4)
    assert grid(NUMZ_4K, Tiles(1024, 256)) == (3, 5)
    # A side the tile covers takes one tile; seedvr2x's 1080p under 1024-px tiles keeps numz's
    # sliver row (DESIGN.md, Pipeline, per shot, step 3: 144 px at 1080p).
    assert grid(OURS_1080, Tiles(1280)) == (1, 2)
    assert grid(OURS_1080, Tiles(1024)) == (2, 2)
    assert grid(OURS_1080, Tiles(1920)) == grid(OURS_1080, Tiles(4096)) == (1, 1)
    assert grid(OURS_4K, Tiles(1024)) == (3, 4)
    # The largest tile, cut off at the picture's edges (DESIGN.md, Memory planner: "a 1280-px
    # tile at 1080p is 1104x1280"), rows by columns: an upright 1080p cuts its columns off.
    assert largest(OURS_1080, Tiles(1280)) == (1104, 1280)
    assert largest(OURS_1080, Tiles(1024)) == largest(NUMZ_4K, Tiles(1024, 128)) == (1024, 1024)
    assert largest(NUMZ_4K, Tiles(2048)) == (2048, 2048)
    assert largest(padded_size((1920, 1080)), Tiles(1280)) == (1280, 1104)
    assert not planner.covers(OURS_1080, Tiles(1856)) and planner.covers(OURS_1080, Tiles(1920))


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


# The 26 entries of planner.MEASURED that the one-output convolution (vendor/models/video_vae_v3/
# modules/causal_inflation_lib.py, ONE_OUTPUT_FROM) changed, as the table had them before it: the
# 2026-10-11 measure, which MEASURED kept for the passes the convolution doesn't take. The 15
# squares from 1152 px and the 11 tiles cut off at 1,104 rows from 1216 px (encode, decode, GiB).
BEFORE_ONE_OUTPUT = {
    (1152, 1152): (13.24, 21.95),
    (1216, 1216): (14.15, 24.40),
    (1280, 1280): (15.67, 26.98),
    (1344, 1344): (16.79, 29.71),
    (1408, 1408): (18.46, 32.57),
    (1472, 1472): (19.74, 35.55),
    (1536, 1536): (21.50, 38.73),
    (1600, 1600): (23.21, 41.96),
    (1664, 1664): (25.10, 45.30),
    (1728, 1728): (27.01, 48.79),
    (1792, 1792): (29.00, 52.48),
    (1856, 1856): (31.12, 56.32),
    (1920, 1920): (33.26, 60.16),
    (1984, 1984): (35.46, 64.22),
    (2048, 2048): (37.76, 68.51),
    # Cut off at 1080p's 1,104 rows.
    (1104, 1216): (13.39, 22.20),
    (1104, 1280): (13.55, 23.34),
    (1104, 1344): (14.22, 24.50),
    (1104, 1408): (14.90, 25.67),
    (1104, 1472): (15.58, 26.78),
    (1104, 1536): (16.26, 27.94),
    (1104, 1600): (16.42, 29.06),
    (1104, 1664): (17.08, 30.22),
    (1104, 1728): (17.74, 31.39),
    (1104, 1792): (18.39, 32.49),
    (1104, 1856): (19.05, 33.65),
}


def test_vae_tiles_measured() -> None:
    # DESIGN.md, Memory planner, Per-phase peaks: "the planner takes the measured peak of every
    # tile measured, the 25 squares and the 20 tiles cut off at those two sizes' rows, the larger
    # where one was measured twice": by the tile's rows and columns, on seedvr2x's own tiled
    # passes (2026-10-11, 21 frames), encode and decode.
    measured = planner.MEASURED
    squares = {rows: peaks for (rows, columns), peaks in measured.items() if rows == columns}
    assert list(squares) == list(range(512, 2049, 64)) and len(measured) == 45
    assert [key for key in measured if key[0] == 1104] == [(1104, t) for t in range(1152, 1857, 64)]
    assert [key for key in measured if key[0] == 752] == [(752, t) for t in range(768, 1217, 64)]
    # DESIGN.md's own, the measure of 2026-10-11: "512 px 3.52 and 5.89 GiB, 768 px 7.08 and
    # 11.11, 1024 px 10.47 and 17.46, 1280 px cut off at 1080p's 1,104 rows 13.55 and 23.34,
    # 1536 px 21.50 and 38.73, 2048 px 37.76 and 68.51". The table holds the first three as
    # they are; the other three are among the 26 entries the one-output convolution changed
    # (BEFORE_ONE_OUTPUT), and are held to the measure with it: the decode of 1280 px cut off at
    # 1,104 rows 23.16, of 1536 px 36.75, and 36.33 and 63.35 for 2048 px.
    first = {512: (3.52, 5.89), 768: (7.08, 11.11), 1024: (10.47, 17.46)}
    assert {size: squares[size] for size in first} == first
    assert {size: BEFORE_ONE_OUTPUT[size, size] for size in (1536, 2048)} == {
        1536: (21.50, 38.73),
        2048: (37.76, 68.51),
    }
    assert {size: squares[size] for size in (1536, 2048)} == {
        1536: (21.50, 36.75),
        2048: (36.33, 63.35),
    }
    assert BEFORE_ONE_OUTPUT[1104, 1280] == (13.55, 23.34) and measured[1104, 1280] == (
        13.55,
        23.16,
    )
    # The table is the 2026-10-11 measure but for 26 entries, which the one-output convolution
    # changed: the encodes of the squares from 1664 px (7) and the decodes of the squares from
    # 1152 px (15) and of the tiles cut off at 1,104 rows from 1216 px (11), 33 figures. Each
    # is under the old one (0.18 to 5.17 GiB, 5.16 by the table's hundredths) but the decodes of
    # 1152 px, 21.95 to 21.99, and of 1104x1216, 22.20 to 22.25; and the rest of the table is as
    # it was.
    old = measured | BEFORE_ONE_OUTPUT
    old_squares = {key[0]: peaks for key, peaks in old.items() if key[0] == key[1]}
    assert len(BEFORE_ONE_OUTPUT) == 26 and all(key in measured for key in BEFORE_ONE_OUTPUT)
    assert {key for key in measured if measured[key] != old[key]} == set(BEFORE_ONE_OUTPUT)
    higher = {key for key, peaks in BEFORE_ONE_OUTPUT.items() if measured[key][1] > peaks[1]}
    assert higher == {(1152, 1152), (1104, 1216)}
    lower: list[float] = []
    for key, (encode, decode) in BEFORE_ONE_OUTPUT.items():
        assert measured[key][0] <= encode
        if key not in higher:
            assert measured[key][1] < decode
            lower.append(decode - measured[key][1])
        lower.append(encode - measured[key][0])
    assert sum(1 for gap in lower if gap > 0) == 24 + 7
    assert (round(min(gap for gap in lower if gap > 0), 2), round(max(lower), 2)) == (0.18, 5.16)
    assert {key for key, peaks in BEFORE_ONE_OUTPUT.items() if measured[key][0] != peaks[0]} == {
        (size, size) for size in range(1664, 2049, 64)
    }
    # The whole table by its sums, in GiB: the 45 encodes and the 45 decodes, then the
    # 2026-10-11 measure's.
    encodes, decodes = zip(*measured.values(), strict=True)
    assert (round(sum(encodes), 2), round(sum(decodes), 2)) == (686.44, 1159.17)
    encodes, decodes = zip(*old.values(), strict=True)
    assert (round(sum(encodes), 2), round(sum(decodes), 2)) == (691.72, 1203.28)
    # A square takes its measure on a frame its squares lay whole, whatever the overlap; a
    # larger square more. On another frame the grid's last tiles are cut off, and the pass takes
    # the most any of its tiles does, its square's measure at the least
    # (test_a_pass_takes_the_tile_that_takes_the_most).
    for size, peaks in squares.items():
        for overlap in (64, 128):
            whole = (2 * size - overlap, 3 * size - 2 * overlap)
            assert tile_shapes(whole, Tiles(size, overlap)) == {(size, size)}
            assert vae_fits(whole, Tiles(size, overlap)) == pytest.approx(peaks, abs=1e-8)
        for padded in (NUMZ_4K, OURS_4K, (2048, 2064)):
            encode, decode = vae_fits(padded, Tiles(size))
            assert encode >= peaks[0] - 1e-8 and decode >= peaks[1] - 1e-8
    assert list(squares.values()) == sorted(squares.values())
    assert list(decodes[:25]) == sorted(decodes[:25])
    for size in range(512, 1089, 64):
        assert vae_fits(OURS_1080, Tiles(size)) == pytest.approx(squares[size], abs=1e-8)
    # The tiles cut off at 1080p's rows and at 720p's take theirs, in seedvr2x's padding; and a
    # larger one doesn't always take more: at 720p the encode in tiles of 960 px, 7.70 GiB, is
    # under the one in tiles of 896, 8.09.
    for size in range(1152, 1857, 64):
        assert vae_fits(OURS_1080, Tiles(size)) == pytest.approx(measured[1104, size], abs=1e-8)
    assert padded_size((720, 1280)) == (752, 1280)
    for size in range(768, 1217, 64):
        assert vae_fits((752, 1280), Tiles(size)) == pytest.approx(measured[752, size], abs=1e-8)
    assert measured[752, 960][0] == 7.70 < measured[752, 896][0] == 8.09
    # "The fits don't hold to the margin. From 1024 px up a decode sits 0.07 to 0.34 GiB under
    # its fit" (0.06 by the larger of 1024 px's two measures), "and below up to 0.67 over it
    # (576 px). An encode moves in steps, not with the tile's area: from 0.39 GiB under its fit
    # (1472 px) to 0.80 over (832 px)" (0.38 by the table's hundredths). That is the 2026-10-11
    # measure, before the one-output convolution (old); the fits were drawn from it, and the
    # table with it sits under them by more (the tiles nobody measured take their fits).
    over = {size: tile_fits(size, size**2 / 1e6) for size in squares}
    for size, peaks in old_squares.items():
        over[size] = (peaks[0] - over[size][0], peaks[1] - over[size][1])
    large = [over[size][1] for size in squares if size >= 1024]
    assert (round(max(large), 2), round(min(large), 2)) == (-0.06, -0.34)
    assert over[1024][1] == max(large) and round(over[576][1], 2) == 0.67
    assert max(over[size][1] for size in squares) == over[576][1]
    encode = {size: over[size][0] for size in squares}
    assert min(encode, key=lambda size: encode[size]) == 1472 and round(encode[1472], 2) == -0.38
    assert max(encode, key=lambda size: encode[size]) == 832 and round(encode[832], 2) == 0.80
    # numz's own, which the refit was drawn from (planner-limits.md, "Tiled VAE at 4K", Encode
    # peak and Decode peak), are within 0.26 GiB of them, and vram.md's (Tiling, 4K, batch 9, one
    # decimal) within 0.11: of the 2026-10-11 measure (old_squares), numz's passes not having the
    # one-output convolution.
    numz = {1024: (10.44, 17.52), 1536: (21.45, 38.86), 2048: (37.67, 68.76)}
    for size, peaks in numz.items():
        assert max(abs(a - b) for a, b in zip(peaks, old_squares[size], strict=True)) < 0.26
    rounded = {512: (3.5, 5.9), 768: (7.1, 11.1), 1024: (10.4, 17.4), 1536: (21.4, 38.7)}
    for size, peaks in rounded.items():
        assert max(abs(a - b) for a, b in zip(peaks, old_squares[size], strict=True)) < 0.11


def test_vae_tiles_nobody_measured() -> None:
    # DESIGN.md, Memory planner, Per-phase peaks: "A cut-off tile of another size takes the
    # larger of its fit and of the squares' line at its area, and 0.3 GiB more, the most a
    # measured one stood above that being 0.27."
    rule = planner._rule  # pyright: ignore[reportPrivateUsage]
    assert planner.CUT_OFF_EXTRA == round(0.3 * GIB)

    def between(pixels: int) -> tuple[float, float]:
        """The line between the measured squares around an area: encode, decode."""
        sides = [rows for rows, columns in planner.MEASURED if rows == columns]
        below = max(side for side in sides if side**2 <= pixels)
        above = min(side for side in sides if side**2 > pixels)
        share = (pixels - below**2) / (above**2 - below**2)
        low, high = planner.MEASURED[below, below], planner.MEASURED[above, above]
        return low[0] + share * (high[0] - low[0]), low[1] + share * (high[1] - low[1])

    # The 0.27, by every measured cut-off tile against the rule it would take unmeasured: the
    # 20 tiles' encodes and decodes. The decode of 752x896, 12.38 GiB measured for a fit of 12.11
    # and a line of 12.09, stands the furthest above it; the encodes from 0.68 under (752x960)
    # to 0.21 over (1104x1536).
    above: dict[tuple[int, int, int], float] = {}
    for (rows, columns), peaks in planner.MEASURED.items():
        if rows == columns:
            continue
        fits, lines = tile_fits(columns, rows * columns / 1e6), between(rows * columns)
        for column in (0, 1):
            ruled = rule(columns, rows * columns, column)
            assert ruled == pytest.approx(max(fits[column], lines[column]), abs=1e-9)
            above[rows, columns, column] = peaks[column] - ruled
    assert len(above) == 40
    worst = max(above, key=lambda key: above[key])
    assert worst == (752, 896, 1) and round(above[worst], 2) == 0.27
    assert above[worst] * GIB < planner.CUT_OFF_EXTRA
    assert [round(rule(896, 752 * 896, 1), 2), round(between(752 * 896)[1], 2)] == [12.11, 12.09]
    encoded = [value for key, value in above.items() if key[2] == 0]
    assert (round(min(encoded), 2), round(max(encoded), 2)) == (-0.68, 0.21)
    assert (above[752, 960, 0], above[1104, 1536, 0]) == (min(encoded), max(encoded))

    # A cut-off tile of another size: 1440p, padded to 1472x2560, in tiles of 1536 px, 1472x1536,
    # takes its fits' 20.94 and 37.27 GiB, over the lines' 20.60 and 35.34 (37.11 before the
    # one-output convolution lowered the squares' decodes), and 0.3.
    assert padded_size((1440, 2560)) == (1472, 2560)
    pixels = 1472 * 1536
    fits, lines = tile_fits(1536, pixels / 1e6), between(pixels)
    assert [round(value, 2) for value in (*fits, *lines)] == [20.94, 37.27, 20.60, 35.34]
    taken = vae_fits((1472, 2560), Tiles(1536))
    assert taken == pytest.approx((fits[0] + 0.3, fits[1] + 0.3), abs=1e-8)
    assert [round(peak, 2) for peak in taken] == [21.24, 37.57]
    # The fit changes at 1024 px: vram.md's below, the refit from there; on a strip of 944
    # rows, the encode's line is over its fit up to 1536 px, the decode's at 1024 px alone.
    for size in (960, 1024, 1536, 2048):
        pixels = 944 * size
        pairs = zip(tile_fits(size, pixels / 1e6), between(pixels), strict=True)
        larger = [max(pair) + 0.3 for pair in pairs]
        assert vae_fits((944, 3840), Tiles(size)) == pytest.approx(larger, abs=1e-8)
    assert [round(peak, 2) for peak in vae_fits((944, 3840), Tiles(960))] == [9.96, 16.04]
    # Rows and columns aren't taken for one another: an upright 1080p, padded to 1952x1104, in
    # tiles of 1280 px, 1280 rows by 1,104 columns, isn't the 1104x1280 tile measured (13.55 and
    # 23.16 GiB; 23.34 before the one-output convolution), and takes the encode's line and the
    # decode's fit, and 0.3: 14.06 and 23.76.
    assert padded_size((1920, 1080)) == (1952, 1104)
    upright = vae_fits((1952, 1104), Tiles(1280))
    lines, fits = between(1280 * 1104), tile_fits(1280, 1.41312)
    assert upright == pytest.approx((lines[0] + 0.3, fits[1] + 0.3), abs=1e-8)
    assert [round(peak, 2) for peak in upright] == [14.06, 23.76]
    # One more cut-off tile was measured, in numz's padding, which the tests alone run (1,088
    # rows at 1080p): 1088x1280, at 13.47 and 23.04 GiB (the first whole frames in tiles,
    # 2026-10-11). The rule and its 0.3 give 13.93 and 23.43.
    assert [round(peak, 2) for peak in vae_fits(NUMZ_1080, Tiles(1280))] == [13.93, 23.43]

    # Above TILE_MAX, which only a pin gives, a tile takes its rule alone, square or cut off
    # (DESIGN.md, Memory planner, Pins: "A tile pin above 2048 px is taken by its fit, with a
    # warning: no peak is measured there"): 2160 rows cut the tiles off from 2176 px. And a
    # larger tile never takes less there, from the squares measured up.
    encodes: list[float] = []
    decodes: list[float] = []
    for size in range(512, 3840, 64):
        encode, decode = vae_fits(NUMZ_4K, Tiles(size))
        if size <= planner.TILE_MAX:
            assert (encode, decode) == pytest.approx(most(NUMZ_4K, Tiles(size)), abs=1e-8)
        else:
            fits = tile_fits(size, min(size, 2160) * size / 1e6)
            assert (encode, decode) == pytest.approx(fits, abs=1e-8)
        encodes.append(encode)
        decodes.append(decode)
    assert encodes == sorted(encodes) and decodes == sorted(decodes)
    # The line serves it where its area lies between two squares: none does at 1440p in tiles
    # of 2112 px, 1472x2112, whose fits are over the lines.
    pixels = 1472 * 2112
    fits = tile_fits(2112, pixels / 1e6)
    assert all(fit > line for fit, line in zip(fits, between(pixels), strict=True))
    assert vae_fits((1472, 2560), Tiles(2112)) == pytest.approx(fits, abs=1e-8)


def tile_shapes(padded: tuple[int, int], tiles: Tiles) -> set[tuple[int, int]]:
    """(rows, columns), in pixels, of the tiles of the tiled passes' grid over padded frames."""
    laid = tilegrid.grid(tilegrid.latent_size(padded), tiles)
    return {(tile.rows * 8, tile.columns * 8) for tile in laid}


def most(padded: tuple[int, int], tiles: Tiles) -> tuple[float, float]:
    """The most a tile of the grid takes, the encode's and the decode's, in GiB, counted here
    tile by tile: its measure, or its rule, with 0.3 GiB for one that isn't the tiles' square,
    up to 2048 px."""
    rule = planner._rule  # pyright: ignore[reportPrivateUsage]
    taken: list[tuple[float, float]] = []
    for rows, columns in tile_shapes(padded, tiles):
        if (rows, columns) in planner.MEASURED:
            taken.append(planner.MEASURED[rows, columns])
            continue
        cut = (rows, columns) != (tiles.size, tiles.size) and tiles.size <= planner.TILE_MAX
        extra = 0.3 if cut else 0.0
        ruled = [rule(tiles.size, rows * columns, column) + extra for column in (0, 1)]
        taken.append((ruled[0], ruled[1]))
    return max(encode for encode, _ in taken), max(decode for _, decode in taken)


def test_a_pass_takes_the_tile_that_takes_the_most() -> None:
    # A tiled pass runs its tiles one after the other, and peaks where the one that takes the
    # most does: not always its largest. The grid's last tile along a side is shorter
    # (tilegrid.grid), and by the table itself a shorter tile can take more: at 720p's rows the
    # encode of 752x896 measured 8.09 GiB and that of 752x960 7.70. Frames padded to 752x1792,
    # 2688 or 3584 in tiles of 960 px are laid 752x960 then 752x896: the encode takes the
    # second tile's 8.09, where the largest tile alone gave 7.70; the decode the first's, 12.69
    # over 12.38.
    measured = planner.MEASURED
    assert (measured[752, 960], measured[752, 896]) == ((7.70, 12.69), (8.09, 12.38))
    for columns in (1792, 2688, 3584):
        assert tile_shapes((752, columns), Tiles(960)) == {(752, 960), (752, 896)}
        assert vae_fits((752, columns), Tiles(960)) == pytest.approx((8.09, 12.69), abs=1e-8)
    assert padded_size((720, 1792)) == (752, 1792)
    # The planner's choice follows: with what the 960-px encode takes by its largest tile alone
    # and no more, it isn't chosen, nor pinned.
    job = ((720, 1792), shots(12))
    least = planner.encode_peak((752, 1792), Tiles(960), 0, 36) + planner.VAE_ROOM + MARGIN
    short = least - round(0.39 * GIB)
    assert planner.plan(least, *job, Pins(encode_tile=960), None)[0].encode_tiles == Tiles(960)
    assert refused(short, *job, Pins(encode_tile=960), None).startswith(
        "--encode-tile 960: the encode in tiles of 960 px takes "
    )
    made, peaks = planner.plan(short, *job, Pins(), None)
    assert made.encode_tiles not in (None, Tiles(960))
    assert peaks.need("encode") <= short - MARGIN
    # Over grids of every tile size: a pass takes what its tile that takes the most does, each
    # shape of its grid counted here (most), and never less than a measure of one of its tiles;
    # in some the largest tile isn't that tile, for the encode and for the decode.
    other = [0, 0]
    grids = 0
    for rows in (544, 752, 944, 1104, 1472, 2192):
        for columns in range(768, 3841, 112):
            for size in range(512, 2049, 64):
                tiles = Tiles(size)
                if planner.covers((rows, columns), tiles):
                    continue
                grids += 1
                taken = vae_fits((rows, columns), tiles)
                assert taken == pytest.approx(most((rows, columns), tiles), abs=1e-8)
                shapes = tile_shapes((rows, columns), tiles)
                for shape in shapes & set(measured):
                    assert taken[0] >= measured[shape][0] - 1e-8
                    assert taken[1] >= measured[shape][1] - 1e-8
                first = largest((rows, columns), tiles)
                for column in (0, 1):
                    one = planner._tile(size, *first, column)  # pyright: ignore[reportPrivateUsage]
                    other[column] += gib(one) < taken[column] - 1e-8
    assert grids > 2000 and other[0] > 100 and other[1] > 10


def test_recorded_tiles_without_a_grid_refused() -> None:
    # A record's tiles are whole numbers, the overlap under the side (Plan.recorded); the tiled
    # passes lay a grid of multiples of 8 px alone (tilegrid.latent_tiles), and the planner
    # counts a tile's peak over that grid: a record with other tiles was edited, and is refused
    # as one that seedvr2x never recorded, on a resume's check and where its plan is made again.
    for tiles in (Tiles(1001), Tiles(1024, 60)):
        said = f"tiles of {tiles.size} px sharing {tiles.overlap}"
        for recorded in (Plan(12, encode_tiles=tiles), Plan(12, None, tiles)):
            name = "encode_tiles" if recorded.encode_tiles else "decode_tiles"
            match = f"not a plan as seedvr2x records one: its {name}: {said}: a tile's side"
            with pytest.raises(PlanError, match=match):
                planner.check_recorded(recorded, shots(12), HD)
            with pytest.raises(PlanError, match=match):
                planner.plan(our_free(24), HD, shots(12), Pins(), None, recorded=recorded)


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
    # the padding included; a shot's frames padded to 4n + 1. The store either tiled pass opens
    # (tilegrid.scratch_bytes, which tests/test_tilegrid.py holds to tiling.Scratch.needed).
    assert round(planner.scratch(OURS_1080, 1) / 1e6, 1) == 12.7
    assert round(planner.scratch(OURS_4K, 1) / 1e6, 1) == 50.5
    assert planner.scratch(OURS_1080, 301) == 301 * 6 * 1104 * 1920
    assert (
        planner.scratch(OURS_1080, 300)
        == planner.scratch(OURS_1080, 298)
        == (planner.scratch(OURS_1080, 301))
    )
    assert planner.scratch(OURS_4K, 2) == 5 * planner.scratch(OURS_4K, 1)


def test_no_pass_grows_with_the_shot() -> None:
    # DESIGN.md, Memory planner, Per-phase peaks: "A pass's peak doesn't grow with the shot: no
    # VAE pass keeps anything of the shot on the GPU ... At 1080p a shot of 301 frames peaked
    # where one of 21 does, to the byte, untiled and in 1024-px tiles, as did one of 101 frames
    # at 4K in those tiles." The planner counted three terms by the shot's latents while the
    # passes kept them there (2 to 6 MiB a latent at 1080p); none is left: a job's encode and
    # decode are predicted the same whatever its longest shot, and so is the DiT's phase under
    # the same window, a shot's noise drawn in system memory (DESIGN.md, Pipeline step 2).
    pins = Pins(window=5, swap_blocks=36, encode_tile=1024, decode_tile=1024)
    for target, free in ((HD, our_free(24)), (UHD, our_free(24)), (HD, our_free(96))):
        for tiled in (pins, replace(pins, encode_tile=None, decode_tile=None)):
            short = planner.plan(free, target, shots(6), tiled, None, split=True)[1]
            long = planner.plan(free, target, shots(6, 76), tiled, None, split=True)[1]
            assert (short.encode, short.decode) == (long.encode, long.decode)
            # Windows of 5 at the most for a shot of 76 latents as for one of 20,000.
            longer = planner.plan(free, target, shots(6, 20_000), tiled, None, split=True)[1]
            assert (longer.encode, longer.dit, longer.decode) == (
                long.encode,
                long.dit,
                long.decode,
            )
            assert longer.window == long.window == 5


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
    these tests, whose figures follow from them; as a run reads it now, a first allocation
    and a small convolution's and product's libraries in place, it is 94.21 GiB and N - 1.49
    (cli._free_memory, measured 2026-10-11)."""
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
    # not 76, and no block is swapped for it. Its peak is that window's: nothing of the shot's
    # 39 other latents is on the GPU beside it (shot.sample_windows).
    free = our_free(96)
    made, peaks = planner.plan(free, HD, shots(80), Pins(window=76), None)
    assert made == Plan(76)
    assert peaks.window == 41 and peaks.dit == planner.dit_peak(OURS_1080, 41)
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
    # the 37 the search stops at, with the fewest blocks that hold a window of 20, 35 of them.
    made, peaks = planner.plan(our_free(24), HD, shots(38), Pins(), None)
    assert (made.window, peaks.window, made.swapped) == (20, 20, 35)
    held = [planner.dit_peak(OURS_1080, 20, blocks, vae_in_dit=False) for blocks in (34, 35)]
    assert held[1] <= our_free(24) - MARGIN < held[0] and peaks.dit == held[1]
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
    # The 91 latents the card holds of a shot in one window, however long the film's longest
    # shot, 700 latents: nothing of it is on the GPU beside one of its windows.
    made, peaks = planner.plan(our_free(96), HD, film, Pins(), None)
    assert (made.window, made.swapped, in_one_window(our_free(96), HD, Pins())) == (91, 36, 91)
    assert peaks.dit == planner.dit_peak(OURS_1080, 91, 36, vae_in_dit=False)


def test_vae_choices() -> None:
    # Untiled where it fits, the DiT's blocks leaving the phase before a tile shrinks; else the
    # largest tile that fits with them away, then as many of them left there as fit.
    # 96 GB at 1080p: both untiled, every block there.
    made, _ = planner.plan(our_free(96), HD, shots(60), Pins(), None)
    assert (made.encode_tiles, made.decode_tiles) == (None, None)
    assert (made.resident_encode, made.resident_decode) == (None, None)
    # 48 GB at 1080p: the untiled decode, about 35 GiB with the VAE alone, fits with 24 of the
    # DiT's blocks beside it and not 25 (DESIGN.md, BlockSwap, rewritten, Per phase) by its peak;
    # with the room every VAE pass is held to, 1.25 GiB under the budget, 21 stay, by 258.6 MiB,
    # whatever the shot's length.
    free, room = our_free(48), planner.VAE_ROOM
    budget = free - MARGIN
    made, peaks = planner.plan(free, HD, shots(21), Pins(), None)
    assert made == Plan(21, resident_decode=21)
    untiled = [planner.decode_peak(OURS_1080, HD, None, blocks) for blocks in (21, 22, 24, 25)]
    assert untiled[3] > budget >= untiled[2]
    assert untiled[1] + room > budget >= untiled[0] + room
    assert round((budget - untiled[0] - room) / 1024**2, 1) == 258.6
    assert peaks.decode == untiled[0] and peaks.bound == "decode"  # the peak, without the room
    # 32 GB at 1080p: the decode in the largest tiles that fit with the DiT away, 1536 px, cut
    # off at the picture's edges and measured so, 27.30 GiB under the one-output convolution
    # (27.94 before it, with none of its blocks left beside them), one of its blocks left beside
    # them, by 327 MiB (71 with 1.5 GiB of room); the encode untiled, 18 left (17).
    free = our_free(32)
    budget = free - MARGIN
    made, peaks = planner.plan(free, HD, shots(12), Pins(), None)
    assert bare(made) == Plan(12, None, Tiles(1536), 0, 18, 1)
    assert planner.decode_peak(OURS_1080, HD, Tiles(1600), 0) + room > budget
    more = planner.decode_peak(OURS_1080, HD, Tiles(1536), 2)
    assert more + room > budget >= peaks.decode + room
    assert round((budget - peaks.decode - room) / 1024**2) == 327
    more = planner.encode_peak(OURS_1080, None, 19)
    assert more + room > budget >= peaks.encode + room
    # 24 GB at 1080p: the untiled encode fits with the DiT away by its peak, and not with its
    # room: tiled, in the largest tiles under the frame's longer side.
    made, peaks = planner.plan(our_free(24), HD, shots(12), Pins(), None)
    assert bare(made) == Plan(12, Tiles(1856), Tiles(1088), 16, 1, 0, vae_in_dit=False)
    untiled_encode = planner.encode_peak(OURS_1080, None, 0, 16)
    assert untiled_encode <= our_free(24) - MARGIN < untiled_encode + room
    # split's slice is counted with an untiled decode's peak, which leaves fewer of the DiT's
    # blocks beside it, 19 on 48 GB for 21; a tiled decode's peak doesn't hold it, the same plan
    # with it.
    made, _ = planner.plan(our_free(48), HD, shots(21), Pins(), None, split=True)
    assert made == Plan(21, resident_decode=19)
    made, _ = planner.plan(free, HD, shots(12), Pins(), None, split=True)
    assert bare(made) == Plan(12, None, Tiles(1536), 0, 18, 1)
    # 24 GB at 4K: both tiled, the decode's smaller, its peak about twice the encode's by the
    # megapixel; every block swapped. The encode is in tiles of 1472 px, the DiT away, by 58 MiB,
    # where the next tiles up don't fit with it away (with 1.5 GiB of room, tiles of 1408 px, and
    # the DiT's two slots and its other parameters, 1.0 GiB, beside them, by 90 MiB); the DiT
    # leaves the decode, and no tile shrinks for it.
    made, peaks = planner.plan(our_free(24), UHD, shots(5), Pins(), None)
    assert bare(made) == Plan(5, Tiles(1472), Tiles(1088), 36, 0, 0, vae_in_dit=False)
    grid = planner._grid  # pyright: ignore[reportPrivateUsage]
    assert (grid(OURS_4K, Tiles(1472)), grid(OURS_4K, Tiles(1088))) == ((2, 3), (3, 4))
    assert peaks.scratch == planner.scratch(OURS_4K, 17) and peaks.tokens == 32_880
    budget = our_free(24) - MARGIN
    away = planner.encode_peak(OURS_4K, Tiles(1472), 0, 36)
    assert round((budget - away - room) / 1024**2) == 58
    assert planner.encode_peak(OURS_4K, Tiles(1472), None, 36) + room > budget
    assert planner.encode_peak(OURS_4K, Tiles(1536), 0, 36) + room > budget
    assert planner.decode_peak(OURS_4K, UHD, Tiles(1088), None, 36) + room > budget
    assert planner.decode_peak(OURS_4K, UHD, Tiles(1152), 0, 36) + room > budget


def test_a_latent_is_counted_by_its_tensor() -> None:
    # What the planner counts a shot's latent and its noise by in system memory
    # (planner.host_moments; LATENT_CHANNELS, LATENT_SIDE, LATENT_VALUE), held to the VAE's
    # config and to tensors of that shape built here: (latents, H / 8, W / 8, 16) in bfloat16,
    # half a byte a padded pixel and a latent, 1,059,840 bytes a latent at 1080p and 4,208,640 at
    # 4K. The noise is drawn in the latent's shape and dtype (shot._drawn), so it counts the same.
    import torch

    import seedvr2x.vendor
    from seedvr2x.runtime.model import COMPUTE_DTYPE
    from seedvr2x.vendor.common.config import load_config

    vendor = Path(seedvr2x.vendor.__file__).parent
    vae: Any = load_config(str(vendor / "models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml"))
    assert (vae.latent_channels, vae.spatial_downsample_factor) == (16, 8)
    assert (planner.LATENT_CHANNELS, planner.LATENT_SIDE) == (16, 8)
    assert planner.LATENT_VALUE == torch.empty(0, dtype=COMPUTE_DTYPE).element_size() == 2
    for padded, bytes_a_latent in ((OURS_1080, 1_059_840), (OURS_4K, 4_208_640)):
        height, width = padded
        assert bytes_a_latent == height * width // 2
        for latents in (1, 5, 151):
            sizes = (latents, height // 8, width // 8, 16)
            latent = torch.empty(sizes, dtype=COMPUTE_DTYPE, device="meta")
            assert latent.numel() * latent.element_size() == bytes_a_latent * latents
            assert planner.host_moments(padded, latents, None)["decode"] == bytes_a_latent * latents


def test_a_window_is_all_the_phase_holds_of_a_shot() -> None:
    # The DiT's phase holds nothing of a shot beside the window it runs, the shot's noise drawn
    # in system memory (shot.sample_windows): its peak is its longest window's, whatever the
    # shot's length.
    # 4K on the 96 GB card, one shot of 601 frames, 151 latents: windows of 21 with 18 blocks
    # swapped, 93.45 GiB, 0.22 under the budget.
    free, film = our_free(96), shots(151)
    made, peaks = planner.plan(free, UHD, film, Pins(), None)
    assert bare(made) == Plan(21, None, Tiles(2048), 18, vae_in_dit=False)
    assert peaks.dit == planner.dit_peak(OURS_4K, 21, 18, vae_in_dit=False) <= free - MARGIN
    assert (round(gib(peaks.dit), 2), round(gib(free - MARGIN - peaks.dit), 2)) == (93.45, 0.22)
    assert planner.dit_peak(OURS_4K, 21, 17, vae_in_dit=False) > free - MARGIN
    # The same shot on a card emulated at 24 GB, 22.51 GiB free: windows of 5 latents, the
    # smallest, every block swapped, 21.79 GiB, 0.12 under the budget; its VAE passes in the
    # tiles of a shot of 5 latents (test_vae_choices), and the encode, 20.38 GiB with the DiT's
    # slots beside its tiles, 1.25 GiB of room under the budget, which the DiT's phase, with none,
    # bounds (the encode did with 1.5).
    free = round(22.51 * GIB)
    made, peaks = planner.plan(free, UHD, film, Pins(), None)
    assert bare(made) == Plan(5, Tiles(1408), Tiles(1088), 36, None, 0, vae_in_dit=False)
    assert peaks.dit == planner.dit_peak(OURS_4K, 5, 36, vae_in_dit=False)
    assert (round(gib(peaks.dit), 2), round(gib(free - MARGIN - peaks.dit), 2)) == (21.79, 0.12)
    assert (peaks.bound, round(gib(peaks.encode), 2)) == ("dit", 20.38)
    assert fits(made, free, UHD, film, None) is None
    # 1080p on a 24 GB card, one shot of 1,197 frames, 300 latents: windows of 20 with 35
    # blocks swapped, 21.96 GiB, under the budget's 21.97.
    free, film = our_free(24), shots(300)
    made, peaks = planner.plan(free, HD, film, Pins(), None)
    assert bare(made) == Plan(20, Tiles(1856), Tiles(1088), 35, 1, 0, vae_in_dit=False)
    assert peaks.dit == planner.dit_peak(OURS_1080, 20, 35, vae_in_dit=False) <= free - MARGIN
    assert (round(gib(peaks.dit), 2), round(gib(free - MARGIN), 2)) == (21.96, 21.97)
    # Where the DiT's phase bounds a plan by a window, its line says no more.
    assert planner.said(made, peaks, "a card", HD, film, Pins())[7] == (
        "plan: predicted peaks: encode 20.55 GiB, DiT 21.96 GiB, decode 20.17 GiB; the DiT's phase"
        " bounds the plan, 0.01 GiB under the budget"
    )
    # A job of many short shots, each in one window: the longest's.
    for card, target, padded, film, plan in (
        (24, HD, OURS_1080, shots(*range(1, 18)), Plan(17, Tiles(1856), Tiles(1088), 28, 1, 0)),
        (96, UHD, OURS_4K, shots(*range(1, 20)), Plan(19, None, Tiles(2048), vae_in_dit=True)),
    ):
        plan = replace(plan, vae_in_dit=plan.swapped == 0)
        made, peaks = planner.plan(our_free(card), target, film, Pins(), None)
        assert bare(made) == plan
        assert peaks.dit == planner.dit_peak(padded, peaks.window, made.swapped, made.vae_in_dit)


def test_no_shot_is_too_long_for_the_dit_s_phase() -> None:
    # The DiT's phase is judged by its longest window alone, a shot's noise drawn in system
    # memory (shot.sample_windows; DESIGN.md, Pipeline step 2): however long the shot, the
    # window decides. 1080p on a 24 GB card, --window 5, whose peak, 21.13 GiB with every weight
    # there, leaves 0.84 under the budget of 21.97: every weight stays there for a shot of 20
    # latents as for one of 20,000 (79,997 frames). Drawn on the GPU beside the shot's latent,
    # 2 MiB a latent at 1080p, the noise took the VAE's weights away from a shot of 3,090
    # latents, and a third block from one of 3,327 (until 2026-10-16).
    free, pins = our_free(24), Pins(window=5)
    budget = free - MARGIN
    window = planner.dit_peak(OURS_1080, 5)
    assert (round(gib(window), 2), round(gib(budget - window), 2)) == (21.13, 0.84)
    for latents in (20, 3090, 3327, 4000, 20_000):
        made, peaks = planner.plan(free, HD, shots(latents), pins, None)
        assert (made.window, peaks.window, made.swapped, made.vae_in_dit) == (5, 5, 0, True)
        assert peaks.dit == peaks.record()["dit_peak"] == window
        assert fits(made, free, HD, shots(latents), None) is None
    # A plan from elsewhere with no block swapped fits a shot of 4,000 latents, and --swap-blocks
    # 0 is taken there, where the draw refused it.
    short = Plan(5, Tiles(512), Tiles(512), 0, 0, 0)
    assert fits(short, free, HD, shots(4000), None) is None
    made, _ = planner.plan(free, HD, shots(4000), Pins(swap_blocks=0), None)
    assert (made.window, made.swapped, made.vae_in_dit) == (6, 0, False)
    # A --window refused is refused for its window.
    assert refused(free, HD, shots(3000), Pins(window=30), None).startswith(
        "--window 30: its longest window, 30 latents, takes "
    )


def test_a_long_shot_at_4k_plans_on_a_24_gb_card() -> None:
    # 4K with 22.51 GiB free, a card emulated at 24 GB: windows of 5 latents, every block
    # swapped and the VAE away, for shots of any length, the DiT's phase bounding the plan
    # (test_a_window_is_all_the_phase_holds_of_a_shot). The draw on the GPU refused a shot
    # of 2,661 latents there, 10,641 frames, naming a cut (until 2026-10-16).
    free = round(22.51 * GIB)
    expected = Plan(5, Tiles(1408), Tiles(1088), 36, None, 0, vae_in_dit=False)
    for film in (shots(151), shots(2661), shots(3000, 40), shots(40, 10_000, 5)):
        made, peaks = planner.plan(free, UHD, film, Pins(), None)
        assert bare(made) == expected
        assert peaks.dit == planner.dit_peak(OURS_4K, 5, 36, vae_in_dit=False)
        assert peaks.bound == "dit"
    # A size the smallest window doesn't fit is refused for that window, naming no cut: the
    # largest size that fits is 4K, the shot's length whatever it is.
    said = refused(free, (4320, 7680), shots(2661), Pins(), None)
    assert "--cuts" not in said and "noise" not in said
    assert said.endswith("the largest output that fits there is --resolution 2160, about 3840x2160")


def test_a_long_shot_plans_as_a_short_one() -> None:
    # No VAE pass keeps anything of the shot on the GPU (DESIGN.md, Memory planner, Per-phase
    # peaks), and the DiT's phase nothing of it beside a window: a shot of 1,000 latents, 3,997
    # frames, plans as one of 20, the cap pinned at 5 latents, on every card here.
    pins = Pins(window=5)

    def planned(card: int, target: tuple[int, int], latents: int) -> Plan:
        made, peaks = planner.plan(our_free(card), target, shots(latents), pins, None)
        assert fits(made, our_free(card), target, shots(latents), None) is None
        assert (made.window, peaks.window) == (5, 5)
        assert peaks.dit == planner.dit_peak(padded_size(target), 5, made.swapped, made.vae_in_dit)
        return bare(made)

    for card, target, plan in (
        (24, HD, Plan(5, Tiles(1856), Tiles(1088), 0, 1, 0)),
        (32, HD, Plan(5, None, Tiles(1536), 0, 18, 1)),
        (48, HD, Plan(5, resident_decode=21)),
        (24, UHD, Plan(5, Tiles(1472), Tiles(1088), 36, 0, 0, vae_in_dit=False)),
        (96, UHD, Plan(5, None, Tiles(2048))),
    ):
        assert planned(card, target, 20) == planned(card, target, 1000) == plan


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
    monkeypatch: pytest.MonkeyPatch,
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
    # holding its batch's frames on the GPU, by 0.13 to 0.99 GiB at 1080p and 0.34 to 1.03 at 4K
    # with the 0.4 GiB seedvr2x's own encode holds over the fit (planner.ENCODE_EXTRA; vram.md
    # says up to 0.9 at 1080p and 1.4 to 2.2 at 4K, over its own fits). A square tile's peak is
    # the one measured on seedvr2x (planner.MEASURED); one cut off at numz's 1,088 rows, which
    # nobody measured, takes its rule and 0.3 GiB. numz's passes don't have the one-output
    # convolution: they are held to the table of 2026-10-11 (BEFORE_ONE_OUTPUT), the one the
    # planner then had.
    alone = planner.dit_on_gpu(0, swapped)
    with monkeypatch.context() as before:
        before.setattr(planner, "MEASURED", planner.MEASURED | BEFORE_ONE_OUTPUT)
        fitted_encode = (planner.encode_peak(padded, encode, 0, swapped) - alone) / GIB
        fitted_decode = (planner.decode_peak(padded, target, decode, 0, swapped) - alone) / GIB
    assert abs(fitted_decode - measured[2]) <= 0.9
    assert 0.1 <= measured[0] - fitted_encode <= 1.1
    # It fits by the model, but for the 4K one on 32 GB: its 7 latents with every block swapped
    # take 30.09 GiB with the model's two slots, 0.14 over the budget, where numz measured
    # 29.65; it fits with the VAE's weights away, as the planner has them there.
    said = fits(recipe, free, target, shot, None, numz_padding=True)
    if (target, card) == (UHD, 32):
        assert said == (
            "the DiT's phase, a window of 7 latents with 36 blocks swapped, takes 30.70 GiB of"
            " free VRAM and the driver reports 30.56 GiB free, 0.14 GiB short"
        )
        recipe = replace(recipe, vae_in_dit=False)
        said = fits(recipe, free, target, shot, None, numz_padding=True)
    assert said is None
    # The planner's own plan for that shot on that card is at least as good: the window whole,
    # no more blocks swapped, no smaller tiles, each VAE pass with its room (planner.VAE_ROOM),
    # which the recipes have too, the DiT away. On 24 GB at 1080p its encode is untiled (in tiles
    # of 1856 px with 1.5 GiB of room, the untiled one lacking that room), and on 32 GB untiled
    # with 19 of the DiT's blocks (18); at 4K on 48 GB its tiles stop at 2048 px
    # (planner.TILE_MAX).
    made, peaks = planner.plan(free, target, shot, Pins(), None, numz_padding=True)
    assert (made.window, peaks.window) == (latents, latents)
    assert made.swapped <= swapped
    for ours, theirs in ((made.encode_tiles, encode), (made.decode_tiles, decode)):
        assert ours is None or (theirs is not None and ours.size >= theirs.size)
    own = {
        (HD, 24): Plan(12, None, Tiles(1088), 16, 0, 0, vae_in_dit=False),
        (HD, 32): Plan(12, None, Tiles(1536), 0, 19, 0),
        (HD, 48): Plan(21, resident_decode=22),
        (UHD, 24): Plan(4, Tiles(1472), Tiles(1088), 27, 0, 0, vae_in_dit=False),
        (UHD, 32): Plan(7, Tiles(1728), Tiles(1280), 36, None, None, vae_in_dit=False),
        (UHD, 48): Plan(7, Tiles(2048), Tiles(1664), 0, 17, 3),
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


# Shots of several windows at 1080p, as measured on the 96 GB card emulating smaller ones
# (research/scripts/vram_cap.py), the run keeping nothing of a shot on the GPU between its
# windows: the card emulated (GB), the shot's latents, and the DiT's peak measured for each
# length of its windows (GiB, the largest of the units' logged peaks), every block swapped and
# the VAE away. The first, 2026-10-11, a shot of 189 frames in six windows of 8 latents and two
# of 7, as that change was made (9.57 and 8.49 GiB before it, the shot's latent and noise
# then waiting on the GPU); the three others, 2026-10-12, by research/scripts/planner_matrix.py
# on the planner as it took the measured tiles, before this file counted the host's memory:
# shots of 53, 85 and 149 frames in two windows.
WINDOWED = [
    (12, 48, {8: 9.49, 7: 8.40}),
    (12, 14, {8: 9.49}),
    (16, 22, {12: 13.50}),
    (24, 38, {20: 21.53}),
]


@pytest.mark.parametrize(("card", "latents", "measured"), WINDOWED)
def test_a_shot_s_windows_peak_as_one_window_shots(
    card: int, latents: int, measured: dict[int, float]
) -> None:
    # A shot of several windows peaks, window by window, where a one-window shot of each
    # window's length is predicted to (planner.dit_peak): nothing of the shot is counted beside
    # a window, nor was there. The card's free memory as a run read it there, 1,600,978,944
    # bytes under N GiB (cli._free_memory: N - 1.49); the plan is the one that ran, every block
    # swapped and the VAE away, its cap the longest window.
    free = card * GIB - 1_600_978_944
    made, peaks = planner.plan(free, HD, shots(latents), Pins(), None, split=True)
    assert (made.window, made.swapped, made.vae_in_dit) == (max(measured), 36, False)
    windows = {end - start for start, end in shot_layout(4 * latents - 3, made.window)}
    assert windows == set(measured)
    for length, peak in measured.items():
        predicted = planner.dit_peak(OURS_1080, length, 36, vae_in_dit=False)
        assert abs(gib(predicted) - peak) <= 0.011, (length, gib(predicted), peak)
    # The plan's own figure for the phase is its longest window's.
    assert peaks.dit == planner.dit_peak(OURS_1080, max(measured), 36, vae_in_dit=False)


@pytest.mark.parametrize(("target", "free", "frames", "ran", "measured", "again"), MATRIX)
def test_first_matrix(
    target: tuple[int, int],
    free: float,
    frames: int,
    ran: Plan,
    measured: tuple[float, float, float] | None,
    again: tuple[Plan, float | None, float | None] | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The corrections drawn from the two matrices: 0.4 GiB on every encode
    # (planner.ENCODE_EXTRA) and split's term off a tiled decode's peak (planner.decode_peak),
    # both PROVISIONAL (implementation, 2026-10-11); no tile chosen above 2048 px
    # (planner.TILE_MAX) and every VAE pass held to a room under the budget besides its peak
    # (planner.VAE_ROOM; 1.5 GiB when the matrices were drawn on, 1.25 since 2026-10-18),
    # DESIGN.md's (Memory planner: What follows from them; Budget): any VAE pass, encode or
    # decode, tiled or not, ran out of memory with its peak close to the free memory. Each case
    # planned again with the memory it read; the DiT's phase is the one that ran everywhere, and
    # every VAE pass leaves 1.85 GiB or more under the free memory (2.1 with 1.5). With every
    # tile's measured peak (planner.MEASURED), no pass counted by its shot's length, the plans of
    # today:
    # - 1080p, 10.54 GiB free: encode in tiles of 768 px with one block of the DiT, decode in 640
    #   px, the DiT away;
    # - 1080p, 14.54: encode in tiles of 1024 px, decode in 768 px, one block of the DiT there
    #   in each;
    # - 1080p, 22.54: encode in tiles of 1856 px, 1 x 2, decode in 1088 px, 2 x 2, the DiT away;
    # - 1080p, 30.54: encode untiled with 18 of the DiT's blocks, decode in 1536 px with one;
    # - 4K, 22.54: encode in tiles of 1472 px, the DiT away; decode in 1088 px, the DiT away;
    # - 4K, 30.54: encode in tiles of 1728 px with two blocks of the DiT, decode in 1280 px with
    #   three;
    # - 4K, 46.54: encode in tiles of 2048 px with 17 of the DiT's blocks, decode in 1664 px
    #   with 3.
    own = {
        (HD, 10.54): Plan(8, Tiles(768), Tiles(640), 35, 1, 0, vae_in_dit=False),
        (HD, 14.54): Plan(12, Tiles(1024), Tiles(768), 35, 1, 1, vae_in_dit=False),
        (HD, 22.54): Plan(20, Tiles(1856), Tiles(1088), 36, 0, 0, vae_in_dit=False),
        (HD, 30.54): Plan(16, None, Tiles(1536), 7, 18, 1, vae_in_dit=False),
        (UHD, 22.54): Plan(5, Tiles(1472), Tiles(1088), 36, 0, 0, vae_in_dit=False),
        (UHD, 30.54): Plan(6, Tiles(1728), Tiles(1280), 27, 2, 3, vae_in_dit=False),
        (UHD, 46.54): Plan(7, Tiles(2048), Tiles(1664), 0, 17, 3),
    }
    reported, shot, padded = round(free * GIB), [Shot(0, frames)], padded_size(target)
    latents = (padded_length(frames) - 1) // 4 + 1
    made, peaks = planner.plan(reported, target, shot, Pins(), None, split=True)
    assert bare(made) == own[target, free]
    assert fits(made, reported, target, shot, None, split=True) is None
    for phase in ("encode", "decode"):
        assert reported - peaks.peak(phase) >= MARGIN + planner.VAE_ROOM
    for tiles in (made.encode_tiles, made.decode_tiles):
        assert tiles is None or tiles.size <= planner.TILE_MAX
    # The DiT's phase is the one that ran: its window, its swapped blocks, the VAE away or not.
    assert (made.window, made.swapped, made.vae_in_dit) == (ran.window, ran.swapped, ran.vae_in_dit)
    # The DiT's phase by the model: its longest window's peak.
    # Of the DiT's peaks these matrices measured (2026-10-11), one is of a shot in one window, 5
    # latents at 4K, and is held to the model below. The others are of shots in two windows,
    # measured while a run kept its shot's whole latent and noise on the GPU through them,
    # 0.020 to 0.043 GiB over their window's own: figures of a run that held more than a
    # window, which the model doesn't predict.
    window = max(end - start for start, end in shot_layout(frames, ran.window))
    alone = planner.dit_peak(padded, window, ran.swapped, ran.vae_in_dit)
    assert peaks.dit == alone

    def vae_peaks(plan: Plan) -> tuple[float, float]:
        """A plan's encode and decode peaks by the corrected model, in GiB, with the table of
        2026-10-11 (BEFORE_ONE_OUTPUT): the runs it is held to ran without the one-output
        convolution."""
        there = (plan.resident_encode, plan.swapped), (plan.resident_decode, plan.swapped)
        with monkeypatch.context() as before:
            before.setattr(planner, "MEASURED", planner.MEASURED | BEFORE_ONE_OUTPUT)
            encode = planner.encode_peak(padded, plan.encode_tiles, *there[0])
            decode = planner.decode_peak(padded, target, plan.decode_tiles, *there[1], split=True)
        return gib(encode), gib(decode)

    # What a decode of those runs kept of its shot on the GPU, which no pass keeps since, in
    # GiB: the shot's latent and a copy of it, 0.954 MiB a megapixel of the padded frame and a
    # latent as measured then (2.02 MiB a latent at 1080p, 8.03 at 4K). Their measured peaks
    # held it, where the model no longer counts it.
    kept_then = 0.954 * padded[0] * padded[1] / 1e6 * latents / 1024

    if measured is None:
        # The plan that ran out of memory in its encode is no longer one, nor the one it ran
        # again with, whose decode did, or whose encode at 4K: but for that one, the 4K encode
        # in tiles of 2048 px with 17 blocks, which the planner takes now: its predicted peak is
        # 1.43 GiB lower under the one-output convolution (44.58 GiB for 46.01) and the room is
        # 1.25 GiB (1.5 was 0.14 GiB short of it), 0.11 GiB left under the budget. That run
        # (41.60 GiB allocated, 4.00 asked, 15 MiB free) had neither.
        assert again is not None
        assert fits(ran, reported, target, shot, None, split=True) is not None
        fitting = fits(again[0], reported, target, shot, None, split=True)
        assert (fitting is None) == ((target, free) == (UHD, 46.54))
        assert (again[0].window, again[0].swapped) == (made.window, made.swapped)
        # The encode that ran the second time: its predicted peak at or above the measured one
        # and within 0.9 GiB of it. The untiled 1080p encode with 13 blocks: 26.42 GiB for
        # 25.56 measured, which held the moments of its 29 latents then, 0.17; the one in tiles
        # of 1856 px, 20.97 for 20.57.
        if again[1] is not None:
            over = vae_peaks(again[0])[0] - again[1]
            assert round(over, 2) == (0.40 if again[0].encode_tiles else 0.86)
        return
    # The plans that passed, as they ran, by the corrected model, each tile by its own measured
    # peak. A tiled decode's predicted peak, with what the run kept of its shot then, is the
    # measured one, within the half hundredth the log rounds it to under and 0.02 GiB over: the
    # 1080p decode in tiles of 640 px, which the small tiles' fit left 0.13 GiB under its
    # measured 9.22, at 9.19 and 0.03 for the 14 latents of its shot; the 4K decode in tiles of
    # 1024 px at 18.98 and 0.04 for 19.01. A tiled encode's is 0.39 to 0.41 GiB over the
    # measured one, which is the tile's measured peak with the weights there: what seedvr2x's
    # encode was taken to hold over its fit (planner.ENCODE_EXTRA, PROVISIONAL), the measured
    # tiles no longer leave it short of.
    encode, decode = vae_peaks(ran)
    assert 0.39 <= encode - measured[0] <= 0.41
    assert -0.005 <= decode + kept_then - measured[2] <= 0.02
    if (target, free) == (HD, 10.54):
        assert (round(decode, 2), round(kept_then, 2)) == (9.19, 0.03)
    if (target, free) == (UHD, 22.54):
        assert (round(decode, 2), round(kept_then, 2)) == (18.98, 0.04)
    # The DiT's peak of the one shot that ran in one window: 21.78 GiB measured for 21.791,
    # the window's peak by the model (numz's fit; planner.DIT_LOADED), 0.011 under.
    if latents == window:
        assert (target, free, latents) == (UHD, 22.54, 5)
        assert 0 <= gib(alone) - measured[1] <= 0.012


def fits(
    plan: Plan,
    free: int | Reading,
    target: tuple[int, int],
    film: list[Shot],
    host_available: int | None,
    *,
    split: bool = False,
    numz_padding: bool = False,
    attention: str = planner.FLASH_ATTN_2,
) -> str | None:
    """Why a plan given, the planner's own or one made here by hand, doesn't fit a job's memory:
    the sentence the planner says of every plan it makes, by the two functions planner.plan
    judges with (planner._job, planner._unfit); None when it fits. The tests' own: no run
    judges a plan handed to it, and the planner keeps no public function for it."""
    job = planner._job(free, target, film, split, numz_padding, attention)  # pyright: ignore[reportPrivateUsage]
    return planner._unfit(job, plan, host_available)  # pyright: ignore[reportPrivateUsage]


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
        "--window 30: its longest window, 27 latents, takes 29.17 GiB of free VRAM with every"
        " block of the DiT swapped, and the driver reports 22.57 GiB free: the longest window"
        " that fits is --window 17"
    )
    assert refused(free, HD, shots(76), Pins(window=12, swap_blocks=0), None) == (
        "--window 12: its longest window, 12 latents, takes 28.46 GiB of free VRAM with"
        " --swap-blocks 0, and the driver reports 22.57 GiB free: the longest window that fits is"
        " --window 6"
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
        "--swap-blocks 0: a window of 5 latents, the shortest this job can run, takes 36.74 GiB of"
        " free VRAM with 0 blocks swapped, and the driver reports 30.57 GiB free: the fewest"
        " blocks that hold it are --swap-blocks 17"
    )
    made, _ = planner.plan(free, UHD, shots(40), Pins(swap_blocks=17), None)
    assert (made.window, made.swapped, made.vae_in_dit) == (5, 17, False)
    assert refused(free, UHD, shots(40), Pins(window=6, swap_blocks=0), None) == (
        "--swap-blocks 0: a window of 6 latents, --window 6's longest, takes 41.00 GiB of free"
        " VRAM with 0 blocks swapped, and the driver reports 30.57 GiB free: the fewest blocks"
        " that hold it are --swap-blocks 27"
    )
    assert refused(free, UHD, shots(40), Pins(swap_blocks=37), None) == (
        "--swap-blocks 37: the DiT has 36 blocks, so from 0 to 36"
    )


def test_tile_pins() -> None:
    # A tile pin is the tile exactly, or refused, naming the largest tile that fits; 0 pins an
    # untiled pass, refused likewise. A pass pinned is held to the room the planner's own is.
    free = our_free(24)
    made, _ = planner.plan(free, UHD, shots(5), Pins(), None)
    assert (made.encode_tiles, made.decode_tiles) == (Tiles(1472), Tiles(1088))
    pins = Pins(encode_tile=1024, decode_tile=1024)
    made, peaks = planner.plan(free, UHD, shots(5), pins, None)
    assert (made.encode_tiles, made.decode_tiles) == (Tiles(1024), Tiles(1024))
    assert refused(free, UHD, shots(5), Pins(decode_tile=2048), None) == (
        "--decode-tile 2048: the decode in tiles of 2048 px takes 65.73 GiB of free VRAM, the"
        " DiT off the GPU, 1.25 of them the room a VAE pass is held to, and the driver reports"
        " 22.57 GiB free: the largest tile that fits is --decode-tile 1088"
    )
    assert refused(free, UHD, shots(5), Pins(encode_tile=0), None) == (
        "--encode-tile 0: the encode untiled takes 78.05 GiB of free VRAM, the DiT off the GPU,"
        " 1.25 of them the room a VAE pass is held to, and the driver reports 22.57 GiB free: the"
        " largest tile that fits is --encode-tile 1472"
    )
    # At 1080p an untiled encode fits a 24 GB card by its peak, 21.38 GiB of free VRAM with the
    # margin, and is refused for its room.
    assert round(gib(planner.encode_peak(OURS_1080, None, 0) + MARGIN), 2) == 21.38
    assert refused(free, HD, shots(12), Pins(encode_tile=0), None) == (
        "--encode-tile 0: the encode untiled takes 22.63 GiB of free VRAM, the DiT off the GPU,"
        " 1.25 of them the room a VAE pass is held to, and the driver reports 22.57 GiB free: the"
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
        "--decode-tile 2304: the decode in tiles of 2304 px takes 85.09 GiB of free VRAM, the"
        " DiT off the GPU, 1.25 of them the room a VAE pass is held to, and the driver reports"
        " 46.57 GiB free: the largest tile that fits is --decode-tile 1664"
    )
    # A tile pinned where the untiled pass would fit is taken as pinned; and 0 where it fits.
    pins = Pins(encode_tile=0, decode_tile=1024)
    made, peaks = planner.plan(our_free(96), HD, shots(40), pins, None)
    assert bare(made) == Plan(40, None, Tiles(1024))
    assert peaks.scratch == planner.scratch(OURS_1080, 157)
    # The largest tile under the padded frame's longer side is one: 1856 px at 1080p.
    made, _ = planner.plan(our_free(96), HD, shots(40), Pins(decode_tile=1856), None)
    grid = planner._grid  # pyright: ignore[reportPrivateUsage]
    assert made.decode_tiles == Tiles(1856) and grid(OURS_1080, Tiles(1856)) == (1, 2)


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
    tiny = planner.encode_peak((128, 128), None, 0) + planner.VAE_ROOM + MARGIN
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
        fits(Plan(12, Tiles(1920)), free, HD, shots(12), None)


def test_floor() -> None:
    # A size that doesn't fit is refused, never degraded, the refusal counting in free VRAM:
    # what the size takes, what the driver reports free, and the largest output that fits there
    # (DESIGN.md, Memory planner). At 4K the smallest window, 5 latents, takes 22.40 GiB of free
    # VRAM by the model, every block swapped and the VAE away: a peak of 21.80 GiB and the margin
    # (DESIGN.md: "At 4K the smallest window takes 22.40 GiB of free VRAM"). Whatever the shot's
    # length: nothing of a longer shot is on the GPU beside a window, its noise drawn in system
    # memory.
    assert refused(numz_free(16), UHD, shots(5), Pins(), None).startswith(
        "an output of 3840x2160 takes 22.40 GiB of free VRAM (its smallest window, 5 latents, with"
        " every block of the DiT swapped and the VAE off the GPU: a peak of 21.80 GiB, and 0.60 of"
        " margin), and the driver reports 14.56 GiB free: the largest output that fits there is"
    )
    assert refused(numz_free(16), UHD, shots(40), Pins(), None) == (
        "an output of 3840x2160 takes 22.40 GiB of free VRAM (its smallest window, 5 latents, with"
        " every block of the DiT swapped and the VAE off the GPU: a peak of 21.80 GiB, and 0.60 of"
        " margin), and the driver reports 14.56 GiB free: the largest output that fits there is"
        " --resolution 1680, about 2986x1680"
    )
    # The pins change nothing to it.
    pins = Pins(window=5, swap_blocks=36, decode_tile=512)
    assert refused(numz_free(16), UHD, shots(40), pins, None).startswith("an output of 3840x2160")
    # The same floor for a shot of 5 latents, of 40 and of 1,000, 3,997 frames: 22.40 GiB of
    # free VRAM plans, in windows of 5 latents, which bound the plan, and 0.01 GiB less is
    # refused, the next size down fitting.
    for latents in (5, 40, 1000):
        made, peaks = planner.plan(round(22.40 * GIB), UHD, shots(latents), Pins(), None)
        assert (made.window, made.swapped, made.vae_in_dit) == (5, 36, False)
        assert peaks.dit == planner.dit_peak(OURS_4K, 5, 36, vae_in_dit=False)
        assert round(peaks.dit / GIB, 2) == 21.79 and peaks.bound == "dit"
        assert refused(round(22.39 * GIB), UHD, shots(latents), Pins(), None) == (
            "an output of 3840x2160 takes 22.40 GiB of free VRAM (its smallest window, 5 latents,"
            " with every block of the DiT swapped and the VAE off the GPU: a peak of 21.80 GiB,"
            " and 0.60 of margin), and the driver reports 22.39 GiB free: the largest output that"
            " fits there is --resolution 2144, about 3810x2144"
        )
    made, _ = planner.plan(round(22.39 * GIB), (2144, 3811), shots(1000), Pins(), None)
    assert made.window == 5
    # An emulated 24 GB card has that, by 0.16 GiB (22.56 GiB free; 22.57 by seedvr2x's own
    # reading): 4K runs there in windows of 5 latents, a shot of 22 latents, 85 frames, as one
    # of 40 or of 1,000, in tiles of 1472 and 1088 px, the DiT away (1408 px with the DiT's slots
    # beside the encode's, with 1.5 GiB of room).
    for free in (numz_free(24), our_free(24)):
        for latents in (22, 40, 1000):
            made, peaks = planner.plan(free, UHD, shots(latents), Pins(), None)
            assert bare(made) == Plan(5, Tiles(1472), Tiles(1088), 36, 0, 0, vae_in_dit=False)
            assert peaks.dit == planner.dit_peak(OURS_4K, 5, 36, vae_in_dit=False)
    # A job whose shots are all shorter runs its longest shot's length: 3 latents on 16 GB, in
    # the largest tiles that then fit, the DiT's slots beside the encode's and the decode's.
    made, _ = planner.plan(numz_free(16), UHD, shots(3, 2), Pins(), None)
    assert bare(made) == Plan(3, Tiles(1024), Tiles(768), 36, None, None, vae_in_dit=False)
    # A portrait target is scaled by its short side too, the output said width x height.
    assert refused(numz_free(16), (3840, 2160), shots(40), Pins(), None).endswith(
        "the largest output that fits there is --resolution 1696, about 1696x3014"
    )
    # At 1080p the floor is the VAE's, its decode in 512-px tiles with the DiT away and the room
    # a VAE pass is held to: 8.27 GiB of free VRAM (8.52 with 1.5 GiB of room, which DESIGN.md
    # has: "At 1080p that is 8.5 GiB of free VRAM: an 8 GB card stays out"), 5.89 the tile's
    # measured peak, which an 8 GB card hasn't (6.56 GiB free on an emulated one); the DiT's
    # smallest window takes 6.92.
    assert refused(numz_free(8), HD, shots(40), Pins(), None) == (
        "an output of 1920x1080 takes 8.27 GiB of free VRAM (its decode in tiles of 512 px, with"
        " the DiT off the GPU: a peak of 6.42 GiB, 1.25 of room a VAE pass is held to, and 0.60 of"
        " margin), and the driver reports 6.56 GiB free: the largest output that fits there is"
        " --resolution 288, about 512x288"
    )
    made, _ = planner.plan(round(8.6 * GIB), HD, shots(40), Pins(), None)
    assert bare(made) == Plan(6, Tiles(640), Tiles(512), 35, 1, 0, vae_in_dit=False)
    made, _ = planner.plan(round(8.52 * GIB), HD, shots(40), Pins(), None)
    assert bare(made) == Plan(6, Tiles(640), Tiles(512), 35, 1, 0, vae_in_dit=False)
    made, _ = planner.plan(round(8.27 * GIB), HD, shots(40), Pins(), None)
    assert bare(made) == Plan(6, Tiles(640), Tiles(512), 36, 0, 0, vae_in_dit=False)
    assert refused(round(8.26 * GIB), HD, shots(40), Pins(), None).endswith(
        "and the driver reports 8.26 GiB free: the largest output that fits there is"
        " --resolution 464, about 824x464"
    )
    assert refused(GIB, HD, shots(40), Pins(), None).endswith(
        "and the driver reports 1.00 GiB free: no output size fits there"
    )
    # Neither floor rises with the longest shot, no pass keeping anything of it on the GPU, its
    # noise drawn in system memory; nor the VAE's with the frame, while a tile of 512 px is
    # whole in it: 8.27 GiB at 1080p for a shot of 6 latents as for one of 1,000 or 20,000, and
    # at 540p. At 4K, where the DiT's smallest window is the floor, 22.40 for a shot of 5
    # latents as for one of 1,000 or 20,000.
    for target, latents, said in (
        (HD, 6, "an output of 1920x1080 takes 8.27 GiB of free VRAM (its decode in tiles"),
        (HD, 1000, "an output of 1920x1080 takes 8.27 GiB of free VRAM (its decode in tiles"),
        ((540, 960), 1000, "an output of 960x540 takes 8.27 GiB of free VRAM (its decode in"),
        (UHD, 5, "an output of 3840x2160 takes 22.40 GiB of free VRAM (its smallest window"),
        (UHD, 1000, "an output of 3840x2160 takes 22.40 GiB of free VRAM (its smallest window"),
        (HD, 20_000, "an output of 1920x1080 takes 8.27 GiB of free VRAM (its decode in tiles"),
        (UHD, 20_000, "an output of 3840x2160 takes 22.40 GiB of free VRAM (its smallest window"),
    ):
        assert refused(4 * GIB, target, shots(latents), Pins(), None).startswith(said)
    # The 8.27 GiB hold to a shot of 20,000 latents, 79,997 frames, where the noise drawn on the
    # GPU beside the shot's latent made the floor from a shot of 3,477 latents (until
    # 2026-10-16).
    for latents in (3476, 3477, 20_000):
        made, _ = planner.plan(round(8.27 * GIB), HD, shots(latents), Pins(), None)
        assert bare(made) == Plan(6, Tiles(640), Tiles(512), 36, 0, 0, vae_in_dit=False)
    # The same pass bounds a smaller frame: 512x512, padded to 544x512, from 8.52 GiB free. Its
    # untiled decode takes more, 6.47 GiB against the tile's 6.41, and is no floor: pinned, it
    # is refused under 8.32 GiB free, naming the tile.
    assert padded_size(SMALL) == (544, 512)
    room = planner.VAE_ROOM
    untiled, tiled = (
        planner.decode_peak((544, 512), SMALL, tiles, 0, 36) for tiles in (None, Tiles(512))
    )
    assert (round(gib(untiled), 2), round(gib(tiled), 2)) == (6.47, 6.41)
    made, peaks = planner.plan(tiled + room + MARGIN, SMALL, shots(20), Pins(), None)
    assert (made.window, made.encode_tiles, made.decode_tiles) == (20, None, Tiles(512))
    assert (peaks.decode, peaks.bound, made.resident_decode) == (tiled, "decode", 0)
    assert refused(tiled + room + MARGIN - 1, SMALL, shots(20), Pins(), None) == (
        "an output of 512x512 takes 8.27 GiB of free VRAM (its decode in tiles of 512 px, with the"
        " DiT off the GPU: a peak of 6.42 GiB, 1.25 of room a VAE pass is held to, and 0.60 of"
        " margin), and the driver reports 8.26 GiB free: the largest output that fits there is"
        " --resolution 496, about 496x496"
    )
    made, _ = planner.plan(tiled + room + MARGIN - 1, (496, 496), shots(20), Pins(), None)
    assert (made.encode_tiles, made.decode_tiles) == (None, None)  # untiled, which fits there
    untiled_pins = Pins(encode_tile=0, decode_tile=0)
    assert refused(round(8.30 * GIB), SMALL, shots(20), untiled_pins, None) == (
        "--decode-tile 0: the decode untiled takes 8.32 GiB of free VRAM, the DiT off the GPU,"
        " 1.25 of them the room a VAE pass is held to, and the driver reports 8.30 GiB free: the"
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
        planner.decode_peak((512, 528), narrow, tiles, 0, 36) for tiles in (None, Tiles(512))
    )
    assert (round(gib(untiled), 2), round(gib(tiled), 2)) == (6.34, 6.41)
    made, peaks = planner.plan(untiled + room + MARGIN, narrow, shots(20), Pins(), None)
    assert (made.encode_tiles, made.decode_tiles, peaks.bound) == (None, None, "decode")
    assert refused(untiled + room + MARGIN - 1, narrow, shots(20), Pins(), None) == (
        "an output of 528x488 takes 8.20 GiB of free VRAM (its decode untiled, with the DiT off"
        " the GPU: a peak of 6.35 GiB, 1.25 of room a VAE pass is held to, and 0.60 of margin),"
        " and the driver reports 8.19 GiB free: the largest output that fits there is"
        " --resolution 464, about 502x464"
    )
    assert refused(untiled + room + MARGIN, narrow, shots(20), Pins(decode_tile=512), None) == (
        "--decode-tile 512: the decode in tiles of 512 px takes 8.27 GiB of free VRAM, the DiT"
        " off the GPU, 1.25 of them the room a VAE pass is held to, and the driver reports 8.19"
        " GiB free: no tile fits, the untiled decode does, --decode-tile 0"
    )
    made, _ = planner.plan(tiled + room + MARGIN, narrow, shots(20), Pins(decode_tile=512), None)
    assert made.decode_tiles == Tiles(512)
    # The size named is the target's scaled, 2 px off what --resolution makes of the source: a
    # 1920x1080 source is 1066x600 at --resolution 600, and 796x448 at --resolution 448. Its
    # tiles of 512 px are cut off at its 480 padded rows, a size nobody measured: their rule
    # and 0.3 GiB, 8.11 GiB of free VRAM, where the next size up, 824x464, takes 8.24.
    assert refused(round(8.15 * GIB), (600, 1066), shots(40), Pins(), None) == (
        "an output of 1066x600 takes 8.27 GiB of free VRAM (its decode in tiles of 512 px, with"
        " the DiT off the GPU: a peak of 6.42 GiB, 1.25 of room a VAE pass is held to, and 0.60 of"
        " margin), and the driver reports 8.15 GiB free: the largest output that fits there is"
        " --resolution 448, about 794x448"
    )
    assert planner.plan(round(8.15 * GIB), (448, 796), shots(40), Pins(), None)[0].window == 21
    assert refused(round(7.85 * GIB), (448, 796), shots(40), Pins(), None).startswith(
        "an output of 796x448 takes 8.11 GiB of free VRAM (its decode in tiles of 512 px, with"
    )
    assert refused(round(8.15 * GIB), (464, 824), shots(40), Pins(), None).startswith(
        "an output of 824x464 takes 8.24 GiB of free VRAM (its decode in tiles of 512 px, with"
    )
    # A frame too small for a tile has its untiled passes for floor, by the small tiles' fit at
    # its area: the encode's on a frame this small, with the 0.4 GiB seedvr2x's encode holds
    # over its fit.
    encode = planner.encode_peak((128, 128), None, 0, 36)
    decode = planner.decode_peak((128, 128), (96, 128), None, 0, 36)
    assert (round(gib(encode), 2), round(gib(decode), 2)) == (2.76, 2.38)
    assert refused(encode + room + MARGIN - 1, (96, 128), shots(6), Pins(), None).startswith(
        "an output of 128x96 takes 4.62 GiB of free VRAM (its encode untiled, too small for a"
        " tile, with the DiT off the GPU: a peak of 2.77 GiB, 1.25 of room a VAE pass is held to,"
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
    # The smallest tiles in most frames. The untiled pass on a frame little larger than a tile:
    # the decode's on 512x528, a square tile taking its measured peak (test_floor); the encode's
    # and the decode's on frames up to 576 px long whose shorter side cuts the tile off, which
    # then takes 0.3 GiB over its rule (planner.CUT_OFF_EXTRA). And tiles of 1024 px on a strip
    # over 1024 px long, under 74 px high for the encode and 134 for the decode, where the
    # refit's line is under the small tiles'.
    assert set(found) == {
        ("encode", 512),
        ("encode", 1024),
        ("encode", None),
        ("decode", 512),
        ("decode", 1024),
        ("decode", None),
    }
    assert {(512, 528), (528, 512)} < found["decode", None] > {(496, 528), (48, 576)}
    assert {(496, 528), (496, 576), (48, 576)} < found["encode", None]
    assert all(max(padded) <= 576 or min(padded) == 48 for padded in found["encode", None])
    assert all(max(padded) <= 576 for padded in found["decode", None])
    assert {min(padded) for padded in found["encode", 1024]} == {48, 64}
    assert {min(padded) for padded in found["decode", 1024]} == {48, 64, 80, 128}
    assert len(found["encode", 512]) == 75 and len(found["decode", 512]) == 70
    assert len(found["encode", None]) == len(found["decode", None]) == 25


def needs(made: Plan, target: tuple[int, int], film: list[Shot], split: bool) -> dict[str, int]:
    """What each phase of a plan needs under the budget for a job, in bytes, by the planner's
    public functions and the run's own layout (shot.shot_layout) alone: a VAE pass its peak and
    its room; the DiT's phase the longest window a shot runs."""
    padded = padded_size(target)
    lengths = {(padded_length(shot.frames) - 1) // 4 + 1: shot.frames for shot in film}
    window = max(
        end - start
        for frames in lengths.values()
        for start, end in shot_layout(frames, made.window)
    )
    dit = planner.dit_peak(padded, window, made.swapped, made.vae_in_dit)
    encode = planner.encode_peak(padded, made.encode_tiles, made.resident_encode, made.swapped)
    decode = planner.decode_peak(
        *(padded, output_size(target), made.decode_tiles, made.resident_decode, made.swapped),
        split,
    )
    return {"encode": encode + planner.VAE_ROOM, "dit": dit, "decode": decode + planner.VAE_ROOM}


def shot_held(made: Plan, target: tuple[int, int], film: list[Shot]) -> int:
    """The most the run holds of one of a job's shots in system memory, in bytes, counted here
    for every shot, by the run's own layout (shot.shot_layout) and in half latents, a quarter of
    a byte a padded pixel: at the encode's end the latent and its buffer; the latent with what
    the check of one latent allocates, 5 halves of one; at a window the latent and the noise
    with the window's copy, the longest's; at the decode's start the windows with what their
    merge makes, from two windows on the merged latent and at a join 3 times the 2 latents
    mixed."""
    height, width = padded_size(target)
    most = 0
    for shot in film:
        windows = [end - start for start, end in shot_layout(shot.frames, made.window)]
        latents = (padded_length(shot.frames) - 1) // 4 + 1
        merging = sum(windows)
        merging += latents + 6 if len(windows) > 1 else 0
        held = (4 * latents, 2 * latents + 5, 2 * (2 * latents + max(windows)), 2 * merging)
        most = max(most, *held)
    return most * (height * width // 4)


def host_need(made: Plan, target: tuple[int, int], film: list[Shot]) -> int:
    """The system memory a plan takes for a job, in bytes, counted here: the process's own, a
    pinned copy of each part of the weights that leaves the GPU in some phase, the most the run
    holds of one of its shots (shot_held), and the scratch's allowance."""
    stay = 36 - made.swapped
    residents = (made.resident_encode, made.resident_decode)
    there = [stay if resident is None else resident for resident in residents]
    parts = [BLOCK] * (36 - min(stay, *there))
    parts += [planner.DIT_OTHER] if 0 in residents else []
    parts += [] if made.vae_in_dit else [planner.VAE_WEIGHTS]
    pinned = sum(paged(part) for part in parts)
    held = shot_held(made, target, film)
    return planner.HOST_RESIDENT + pinned + held + made.scratch_allowance


def test_every_plan_fits() -> None:
    # Over a grid of sizes, shots, memory and pins, with split and without, the system memory
    # unchecked, ample or short: a plan returned
    # fits, each phase's need counted here from the public functions (needs), and the system
    # memory from the run's own layout, every shot's (host_need), not by the check plan() itself
    # makes (fits); its recorded peaks are those; and it takes every pin exactly. A size refused
    # names a --resolution that plans, the largest that does.
    targets = ((96, 128), SMALL, (528, 560), (600, 1066), HD, UHD)
    films = (shots(1), shots(3, 2), shots(40), shots(5, 76, 133))
    frees = (4.0, 7.3, 7.6, 10.5, 14.56, 22.56, 46.56, 94.27)
    # Each job takes three of these in turn: split, the system memory available.
    hosts = (None, 64 * GIB, 12 * GIB)
    kinds = [(split, host) for split in (False, True) for host in hosts]
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
                        split, host = kinds[(jobs + 5 * turn) % len(kinds)]
                        job = (free, target, film, pins, host)
                        try:
                            made, peaks = planner.plan(*job, split=split)
                        except PlanError as error:
                            floors += floor_named(str(error), free, target, film, split)
                            short += "of system memory" in str(error)
                            continue
                        plans += 1
                        need = needs(made, target, film, split)
                        assert max(need.values()) <= free - MARGIN
                        assert need == {phase: peaks.need(phase) for phase in planner.PHASES}
                        taken = host_need(made, target, film)
                        assert host is None or taken <= host
                        assert peaks.host == taken
                        assert peaks.shot == shot_held(made, target, film)
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
    assert plans > 1500 and floors > 1000 and tiled > 300 and short > 250


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
    # over 108 jobs: three sizes, nine free memories from an 8 GB card's to the 96 GB card's,
    # short shots, long ones and a film's mix, with split and without; refused where the search
    # finds no plan.
    films = (shots(4), shots(12, 40, 3), shots(76), shots(300, 21, 5))
    frees = (6.54, 10.54, 14.54, 22.54, 30.54, 38.54, 46.54, 62.54, 94.24)
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
) -> bool:
    """Whether a refusal is the floor's; the --resolution it names then plans, the target scaled
    by its short side as the planner scales it, and the next multiple of 16 doesn't."""
    if not said.startswith("an output of"):
        return False
    named = re.search(r"is --resolution (\d+), about", said)
    if named is None:
        assert ": no output size fits there" in said
        return True
    short = int(named[1])
    height, width = target

    def scaled(short: int) -> tuple[int, int]:
        return (short, short * width // height)

    planner.plan(free, scaled(short), film, Pins(), None, split=split)
    if short + 16 < height:
        larger = (free, scaled(short + 16), film, Pins(), None)
        assert refused(*larger, split=split).startswith("an output of")
    return True


def test_host_memory() -> None:
    # The system memory the plan takes: the process's own 2.4 GiB, a pinned copy of each part of
    # the weights that leaves the GPU in some phase, what the run holds of a shot at the moment
    # it holds the most (test_host_moments), and what a tiled shot's scratch may take
    # of it, the plan's allowance (test_scratch_allowance): here the longest shot's scratch
    # whole. The first three over what is available: refused, naming each.
    free = our_free(24)
    made, peaks = planner.plan(free, HD, shots(76), Pins(), 64 * GIB)
    assert bare(made) == Plan(17, Tiles(1856), Tiles(1088), 28, 1, 0, vae_in_dit=False)
    kept = planner.scratch(OURS_1080, 301)
    assert peaks.scratch == made.scratch_allowance == kept == 301 * 12_718_080
    # Every block leaves in the VAE's phases, the DiT's other parameters with them, and the VAE
    # in the DiT's phase: each part's bytes rounded up to a page, 15.81 GiB (measured, milestone
    # 3).
    pinned = 36 * paged(BLOCK) + paged(planner.DIT_OTHER) + paged(planner.VAE_WEIGHTS)
    # The shot's 76 latents in windows of 17: its latent and its noise beside a window's output,
    # 169 latents' worth of 1,059,840 bytes at 1080p: 0.17 GiB.
    held = 169 * 1_059_840
    assert (peaks.shot, peaks.moment, peaks.holder) == (held, "windows", 76)
    assert peaks.host == planner.HOST_RESIDENT + pinned + held + kept
    assert f"{gib(pinned):.2f}" == "15.81" and pinned - 36 * BLOCK < 501_500_000 + 166_500_000
    # The refusal says each part as the plan's line does, to the nearest hundredth, and its
    # total is their sum; the shot's part with its length and the moment that takes it. Of the
    # plan that takes the least of it at the card's window: the VAE's weights kept on the GPU
    # in the DiT's phase, one block more swapped for them (test_the_vae_stays_where_the_host_is_
    # short), then the DiT's 7 other blocks kept on the GPU in the VAE's phases, their tiles
    # shrunk for it, the 29 swapped alone pinned, 12.24 GiB, which the refusal says
    # (test_on_a_short_host_the_tiles_shrink_before_the_window). No shorter window's plan fits
    # 3 GiB either; 4 GiB run windows of 6 with 3 blocks swapped (refused with 1.5 GiB of room,
    # no window's plan fitting), the VAE kept, in tiles of 640 and 512 px with 33 of the DiT's
    # blocks beside each; 8 GiB, refused until then, run windows of 9 (that test).
    four, _ = planner.plan(free, HD, shots(76), Pins(), 4 * GIB)
    assert bare(four) == Plan(6, Tiles(640), Tiles(512), 3, 33, 33)
    assert refused(free, HD, shots(76), Pins(), 3 * GIB) == (
        "the plan takes 14.81 GiB of system memory (2.40 for the process, 12.24 pinned for 29"
        " blocks that leave the GPU, 0.17 for the longest shot, 301 frames (76 latents), at its"
        " longest window, its latent and its noise beside the window's output) and 3.00 GiB are"
        " available, 11.81 GiB short, its VAE's tiles shrunk to keep the DiT on the GPU: the"
        " encode in tiles of 1600 px, with 7 of the DiT's blocks on the GPU; the decode in tiles"
        " of 960 px, with the DiT's 7 unswapped blocks and its 2 slots on the GPU"
    )
    assert planner.said(made, peaks, "a card", HD, shots(76), Pins())[5] == (
        "plan: 15.81 GiB of system memory pinned for 36 blocks, the DiT's other parameters and the"
        " VAE; 0.17 GiB for the longest shot at the most, at its longest window, its latent and"
        " its noise beside the window's output; 21.95 GiB of system memory in all"
    )
    # A job is refused for those three alone: with no more than they take, its scratch is
    # allowed nothing, and every shot's goes to its file.
    least = planner.HOST_RESIDENT + pinned + held
    scarce, taken = planner.plan(free, HD, shots(76), Pins(), least)
    assert scarce == bare(made) and (taken.host, taken.scratch) == (least, kept)
    # One byte less than a plan takes, with --swap-blocks and --window pinned, which are taken
    # as they are, and its tiles too, which no host shrinks then, is refused: 18.3813 GiB
    # taken, said 18.38 by its parts, where a byte less is there: what is there is said a
    # hundredth under what is taken, which the two rounded alike would hide. It names the
    # longest window whose plan fits, the --window that plans there.
    assert f"{gib(least):.4f}" == "18.3813"
    tiled = Pins(encode_tile=1856, decode_tile=1088)
    assert refused(free, HD, shots(76), replace(tiled, swap_blocks=28, window=17), least - 1) == (
        "the plan takes 18.38 GiB of system memory (2.40 for the process, 15.81 pinned for 36"
        " blocks, the DiT's other parameters and the VAE's weights that leave the GPU, 0.17 for"
        " the longest shot, 301 frames (76 latents), at its longest window, its latent and its"
        " noise beside the window's output) and 18.37 GiB are available, 0.01 GiB short: the"
        " longest window whose plan fits it is --window 15"
    )
    # Without --window, that window is the planner's own: under it the VAE's weights stay
    # beside the 28 blocks' windows, unpinned (test_the_planner_takes_the_longest_window_the_
    # host_holds).
    kept, taken = planner.plan(free, HD, shots(76), replace(tiled, swap_blocks=28), least - 1)
    assert (kept.window, kept.swapped, kept.vae_in_dit) == (15, 28, True)
    assert taken.host <= least - 1 and taken.shortened == planner.Shortened(17, 1838, 1837)
    pinned_so = planner.plan(
        free, HD, shots(76), replace(tiled, swap_blocks=28, window=15), least - 1
    )
    assert pinned_so == (kept, replace(taken, shortened=None))
    assert planner.plan(free, HD, shots(76), Pins(), None) == (made, peaks)  # not checked
    # Nothing leaves the GPU, nothing tiled: the process and the shot, 40 latents in one window,
    # 120 latents' worth through it, 0.12 GiB.
    made, peaks = planner.plan(our_free(96), HD, shots(40), Pins(), 4 * GIB)
    assert made == Plan(40) and peaks.scratch == 0
    assert peaks.host == planner.HOST_RESIDENT + 120 * 1_059_840
    assert refused(our_free(96), HD, shots(40), Pins(), 2 * GIB) == (
        "the plan takes 2.52 GiB of system memory (2.40 for the process, 0.12 for the longest"
        " shot, 157 frames (40 latents), at its longest window, its latent and its noise beside"
        " the window's output) and 2.00 GiB are available, 0.52 GiB short"
    )
    # A long tiled shot's scratch counts up to its allowance: 4K, 601 frames, 28.27 GiB, over
    # half of what a host with 64 GiB available has left of the pinned copies and of the shot,
    # 151 latents in windows of 21, 323 latents' worth of 4,208,640 bytes, 1.27 GiB: 27.34, so
    # that it goes whole to its memory-mapped file (tiling.Scratch). Nothing moves between the
    # phases: the 18 swapped blocks alone are pinned, with the VAE.
    made, peaks = planner.plan(our_free(96), UHD, shots(151), Pins(), 64 * GIB)
    assert bare(made) == Plan(21, None, Tiles(2048), 18, vae_in_dit=False)
    pinned = 18 * paged(BLOCK) + paged(planner.VAE_WEIGHTS)
    held = (2 * 151 + 21) * 4_208_640
    left = 64 * GIB - pinned - held
    assert peaks.scratch == 601 * 50_503_680 > made.scratch_allowance == left // 2
    assert (f"{gib(held):.2f}", f"{gib(left // 2):.2f}") == ("1.27", "27.34")
    assert peaks.host == planner.HOST_RESIDENT + pinned + held + made.scratch_allowance
    # Two swapped blocks save nothing in the DiT's phase, their slots there, and wait in pinned
    # copies all the same, with the 13 more the untiled decode sends away, which frees their
    # slots: 21 blocks stay beside it, as with none swapped.
    made, peaks = planner.plan(our_free(48), HD, shots(12), Pins(swap_blocks=2), None)
    assert made == Plan(12, swapped=2, resident_decode=21)
    assert peaks.host == planner.HOST_RESIDENT + 15 * paged(BLOCK) + 36 * 1_059_840
    unswapped, _ = planner.plan(our_free(48), HD, shots(12), Pins(swap_blocks=0), None)
    assert unswapped == Plan(12, resident_decode=21)


def test_the_vae_stays_where_the_host_is_short() -> None:
    # DESIGN.md has the VAE's weights leave the DiT's phase before a block is swapped (Memory
    # planner, Choices, 1), and this in the paragraph after 3. 1080p on a 24 GB card, a shot of
    # 301 frames: 28 blocks swapped and the VAE away. Away, its weights are pinned, 0.47 GiB of
    # the 15.81 pinned there, every block leaving in the VAE's phases anyway. Where that plan
    # takes more system memory than is available, the same shape with one block more swapped
    # and the VAE staying takes 0.46 GiB less, and is the plan (planner._vae_kept): no bit
    # moves, nor any plan on a host that holds the first.
    free, job = our_free(24), (HD, shots(76), Pins())
    pinned = 36 * paged(BLOCK) + paged(planner.DIT_OTHER) + paged(planner.VAE_WEIGHTS)
    held = 169 * 1_059_840
    first = planner.HOST_RESIDENT + pinned + held
    made, peaks = planner.plan(free, *job, first)
    assert (made.swapped, made.vae_in_dit, peaks.host) == (28, False, first)
    kept, peaks = planner.plan(free, *job, first - 1)
    assert kept == replace(
        made, swapped=29, vae_in_dit=True, scratch_allowance=first - 1 - least_of(kept)
    )
    assert peaks.host == least_of(kept) + kept.scratch_allowance == first - 1
    # 29 is the fewest that hold the window with the VAE there, and the DiT's peak fits.
    assert peaks.dit == planner.dit_peak(OURS_1080, 17, 29) <= free - MARGIN
    assert planner.dit_peak(OURS_1080, 17, 28) > free - MARGIN
    assert planner.said(kept, peaks, "a card", *job)[4:6] == [
        "plan: 29 of the DiT's 36 blocks swapped, the VAE on the GPU in the DiT's phase; in the"
        " encode, 1 of the DiT's blocks on the GPU; in the decode, the DiT off the GPU",
        "plan: 15.35 GiB of system memory pinned for 36 blocks and the DiT's other parameters;"
        " 0.17 GiB for the longest shot at the most, at its longest window, its latent and its"
        " noise beside the window's output; 18.39 GiB of system memory in all",
    ]
    # (The line's total is its parts' as said, the scratch's allowance, 0.47 GiB, among them.)
    # Down to what that plan takes, 17.91 GiB; a byte under, its window pinned or not, the DiT
    # stays on the GPU in the VAE's phases and their tiles shrink, the next step
    # (test_on_a_short_host_the_tiles_shrink_before_the_window).
    least = least_of(kept)
    assert f"{gib(least):.2f}" == "17.91" and first - least == paged(planner.VAE_WEIGHTS)
    scarce, peaks = planner.plan(free, *job, least)
    assert scarce == replace(kept, scratch_allowance=0) and peaks.host == least
    for pins in (Pins(), Pins(window=17)):
        shrunk, peaks = planner.plan(free, HD, shots(76), pins, least - 1)
        assert (shrunk.encode_tiles, shrunk.decode_tiles) == (Tiles(1856), Tiles(1024))
        assert (shrunk.swapped, shrunk.vae_in_dit) == (29, True)
        assert peaks.shrunk == planner.Shrunk(Tiles(1856), Tiles(1088), 1792, 1791)
    # --swap-blocks is taken exactly, the VAE's place around it as on any host; the tiles still
    # shrink around it.
    pins = Pins(swap_blocks=28, window=17)
    shrunk, peaks = planner.plan(free, HD, shots(76), pins, first - 1)
    assert (shrunk.swapped, shrunk.vae_in_dit, shrunk.decode_tiles) == (28, False, Tiles(1024))
    assert peaks.shrunk == planner.Shrunk(Tiles(1856), Tiles(1088), 1838, 1837)
    # A resume plans the weights' places the same way around its recorded shape.
    shape = Plan.recorded(made.record())
    again, _ = planner.plan(free, *job, first - 1, recorded=shape)
    assert bare(again) == bare(kept)
    assert bare(planner.plan(free, *job, first, recorded=shape)[0]) == bare(made)
    # A plan the GPU's memory refuses is refused for it, and one whose window no count of
    # swapped blocks holds with the VAE there keeps the VAE away, refused for the system
    # memory it takes: 4K's smallest window on that card, every block swapped.
    assert "of free VRAM" in refused(round(6.5 * GIB), *job, 8 * GIB)
    made, peaks = planner.plan(free, UHD, shots(5), Pins(), None)
    assert (made.swapped, made.vae_in_dit) == (36, False)
    assert refused(free, UHD, shots(5), Pins(), peaks.host - made.scratch_allowance - 1).startswith(
        "the plan takes 18.27 GiB of system memory (2.40 for the process, 15.81 pinned for 36"
        " blocks, the DiT's other parameters and the VAE's weights that leave the GPU, 0.06 for"
    )


def least_of(made: Plan) -> int:
    """What a plan takes of system memory without its scratch's allowance, for the 301-frame
    shot of test_the_vae_stays_where_the_host_is_short, in bytes."""
    return host_need(replace(made, scratch_allowance=0), HD, shots(76))


def test_the_planner_takes_the_longest_window_the_host_holds() -> None:
    # The choices are made for the GPU's memory, the longest window first, and the system memory
    # is checked after them: more free VRAM turned a job that was planned into one refused for
    # its pinned blocks. 1080p, a shot of 301 frames, 16 GiB of system memory available:
    # planned with 46.5, 60 and 70 GiB of free VRAM, in two windows of 39 latents, and with
    # 94.2, in one window with nothing swapped; with 80, one window of 76 with 35 blocks
    # pinned, 17.40 GiB, was refused. The planner takes by itself the longest window whose plan
    # the host holds too (DESIGN.md, Memory planner, Choices; design, 2026-10-12): the plan
    # --window 39 makes, and its lines say which window the card alone holds, and why not.
    job, host = (HD, shots(76)), 16 * GIB
    for free in (46.5, 60, 70):
        made, peaks = planner.plan(round(free * GIB), *job, Pins(), host, split=True)
        assert made.window == 39 and peaks.shortened is None
    alone, _ = planner.plan(round(94.2 * GIB), *job, Pins(), host, split=True)
    assert (alone.window, alone.swapped) == (76, 0)
    made, peaks = planner.plan(80 * GIB, *job, Pins(), host, split=True)
    assert (made.window, peaks.window, made.swapped) == (39, 39, 0) and peaks.host <= host
    assert peaks.shortened == planner.Shortened(76, 1740, 1600)
    pinned = planner.plan(80 * GIB, *job, Pins(window=39), host, split=True)
    assert pinned == (made, replace(peaks, shortened=None))
    assert planner.said(made, peaks, "a card", *job, Pins())[3] == (
        "plan: windows of 39 latents at most: 0 of 1 shot in one window, 1 join; the card alone"
        " holds windows of 76 latents, whose plan takes 17.40 GiB of system memory where 16.00"
        " GiB are available: the longest window whose plan fits both is taken"
    )
    assert planner.said(pinned[0], pinned[1], "a card", *job, Pins(window=39))[3] == (
        "plan: windows of 39 latents at most: 0 of 1 shot in one window, 1 join"
    )
    # The host's memory unread, nothing is shortened: the card's own window.
    unread, peaks = planner.plan(80 * GIB, *job, Pins(), None, split=True)
    assert (unread.window, unread.swapped, peaks.shortened) == (76, 34, None)
    # Never under --window: the pin is refused, naming that window, the --window that plans.
    assert refused(80 * GIB, *job, Pins(window=76), host, split=True) == (
        "the plan takes 17.40 GiB of system memory (2.40 for the process, 14.77 pinned for 35"
        " blocks that leave the GPU, 0.23 for the longest shot, 301 frames (76 latents), at its"
        " longest window, its latent and its noise beside the window's output) and 16.00 GiB are"
        " available, 1.40 GiB short: the longest window whose plan fits it is --window 39"
    )
    # With 30 GiB free every window's plan sent the DiT away in the VAE's phases, 36 blocks
    # pinned whatever the cap, and the job was refused: the DiT kept on the GPU there, their
    # tiles shrunk, before the window shortens (test_on_a_short_host_the_tiles_shrink_before_the_
    # window), it runs windows of 21, the decode in tiles of 1408 px for the card's 1536 (1344 for
    # 1472 before the one-output convolution and its room).
    made, peaks = planner.plan(30 * GIB, *job, Pins(), host, split=True)
    assert (made.window, made.encode_tiles, made.decode_tiles) == (21, None, Tiles(1408))
    assert peaks.shortened == planner.Shortened(27, 1782, 1600)
    assert peaks.shrunk == planner.Shrunk(None, Tiles(1536), 1792, 1600)
    # Where no window's plan fits both, the job is refused, and nothing more is said of the
    # window: under the process's own 2.4 GiB.
    assert "and 2.00 GiB are available, " in refused(30 * GIB, *job, Pins(), 2 * GIB, split=True)
    # Nor on a resume, whose window is the recorded one; nor for the GPU's memory.
    shape = Plan.recorded(planner.plan(80 * GIB, *job, Pins(), None, split=True)[0].record())
    again = refused(80 * GIB, *job, Pins(), host, split=True, recorded=shape)
    assert again.endswith("and 16.00 GiB are available, 1.40 GiB short")
    assert "--window" not in refused(round(6.5 * GIB), *job, Pins(), host)
    # Over a grid of jobs, memories and hosts: a plan with a window shortened for the host is
    # the plan that window pinned makes, every longer window refused there for the system
    # memory, up to the one the card alone gives; and a job refused for the system memory has
    # no window that plans.
    shortened = unheld = 0
    films = (shots(76), shots(40), shots(12, 133), shots(5))
    frees = (22.5, 30, 37, 48, 62, 80, 83)
    longest = planner._longest  # pyright: ignore[reportPrivateUsage]
    for target in ((540, 960), (720, 1280), HD, UHD):
        for film in films:
            lengths = tuple(sorted({(padded_length(shot.frames) - 1) // 4 + 1 for shot in film}))
            for free in (round(count * GIB) for count in frees):
                for available in (round(5.3 * GIB), round(13.11 * GIB), 16 * GIB, 18 * GIB):
                    case = (free, target, film)

                    def plans(
                        cap: int,
                        free: int = free,
                        target: tuple[int, int] = target,
                        film: list[Shot] = film,
                        available: int = available,
                    ) -> str | None:
                        try:
                            planner.plan(
                                free, target, film, Pins(window=cap), available, split=True
                            )
                        except PlanError as error:
                            return str(error)
                        return None

                    window = planner.plan(*case, Pins(), None, split=True)[0].window
                    assert window is not None
                    try:
                        made, peaks = planner.plan(*case, Pins(), available, split=True)
                    except PlanError as error:
                        if "of system memory" in str(error):
                            said = str(error)
                            shrunk = ", its VAE's tiles shrunk to keep the DiT on the GPU: "
                            assert said.endswith(" GiB short") or shrunk in said, said
                            assert all(plans(cap) is not None for cap in range(5, window + 1))
                            unheld += 1
                        continue
                    if peaks.shortened is None:
                        assert made.window == window
                        continue
                    assert made.window is not None and made.window < window
                    assert peaks.shortened.window == window and peaks.host <= available
                    assert peaks.shortened.taken > peaks.shortened.available
                    again = planner.plan(*case, Pins(window=made.window), available, split=True)
                    assert again == (made, replace(peaks, shortened=None))
                    for cap in range(made.window + 1, window + 1):
                        said = plans(cap)
                        if longest(lengths, cap) == peaks.window:
                            assert said is None  # the same windows, under a larger cap
                        else:
                            assert said is not None and "of system memory" in said, (cap, said)
                    shortened += 1
    # 87 and 20 (2026-10-16): fewer jobs are refused for the system memory since the DiT stays
    # on the GPU in the VAE's phases before a window shortens.
    assert shortened > 20 and unheld >= 20


def test_on_a_short_host_the_tiles_shrink_before_the_window() -> None:
    # DESIGN.md, Memory planner, Choices (decided on 2026-10-13): where the plan takes more
    # system memory than the host has, "before the window shortens, the DiT stays on the GPU for
    # the VAE's phases and the tiles shrink, the window being the larger lever", after the VAE's
    # weights kept for the DiT's phase (test_the_vae_stays_where_the_host_is_short). The largest
    # tiles first send the most of the DiT away, each block in a pinned copy: more free VRAM
    # pinned more and shortened the window, or refused the job: refusals that smaller tiles run,
    # and windows that fell as the free VRAM grew (DESIGN.md, Memory planner, Choices; To
    # measure, The planner on a short host). One that fell, as a grid of plans made it: 1440p,
    # shots of 5 and 39 latents (17 and 152 frames there), split, 17.338 GiB of system memory.
    film, host = shots(5, 39), 18_616_624_832
    qhd = (1440, 2560)
    below, made = 46_170_899_209, 48_855_253_769  # 43.0 and 45.5 GiB, the grid's
    planned, peaks = planner.plan(below, qhd, film, Pins(), host, split=True)
    assert (planned.window, planned.decode_tiles, planned.resident_decode) == (21, Tiles(1600), 3)
    assert peaks.shrunk is None and peaks.shortened is None
    # With 2.5 GiB more, the card alone takes the decode in tiles of 1728 px, which keep one block
    # of the DiT on the GPU, 35 pinned with the VAE's weights kept: 17.34 GiB, 0.006 over. The
    # window fell to 15 for 0.011 GiB of the shot's latents; the tiles shrink instead, 1664 px
    # keeping 4 blocks there, and the window stays.
    planned, peaks = planner.plan(made, qhd, film, Pins(), host, split=True)
    assert (planned.window, planned.swapped, planned.vae_in_dit) == (21, 24, True)
    assert (planned.encode_tiles, planned.decode_tiles) == (None, Tiles(1664))
    assert (planned.resident_encode, planned.resident_decode) == (None, 5)
    assert peaks.shortened is None and peaks.host <= host
    assert peaks.shrunk == planner.Shrunk(None, Tiles(1728), 1734, 1733)
    assert planner.said(planned, peaks, "a card", qhd, film, Pins())[6] == (
        "plan: encode untiled, decode in tiles of 1664 px, 1 x 2, each held 1.25 GiB under the"
        " budget; the card alone takes the encode untiled and the decode in tiles of 1728 px,"
        " whose plan takes 17.34 GiB of system memory where 17.33 GiB are available: the DiT kept"
        " on the GPU in the VAE's phases, the largest tiles whose plan fits both are taken"
    )
    # The plan --window 21 makes, which a pinned window's never shortens; nor a pass pinned in
    # its tiles: under --decode-tile 1728 the window falls to 15, as it did, the decode's tiles
    # taken as asked.
    pinned = planner.plan(made, qhd, film, Pins(window=21), host, split=True)
    assert pinned == (planned, peaks)
    planned, peaks = planner.plan(made, qhd, film, Pins(decode_tile=1728), host, split=True)
    assert (planned.window, planned.decode_tiles, peaks.shrunk) == (15, Tiles(1728), None)
    assert peaks.shortened == planner.Shortened(21, 1734, 1733)
    # One of the 828: 4K, a shot of 5 frames, 13.11 GiB of system memory. With 78 GiB free the
    # decode in tiles of 2048 px keeps 28 blocks of the DiT on the GPU (15 before the one-output
    # convolution lowered its peak by 5.16 GiB), and the plan fits; with 80.5 the card alone
    # takes the encode untiled, which keeps none of the DiT, 15.92 GiB, and the job was refused,
    # no window being shorter; the encode in tiles of 2048 px keeps it all, the decode 34
    # blocks (21 before).
    film, host = shots(2), round(13.11 * GIB)
    planned, peaks = planner.plan(round(78 * GIB), UHD, film, Pins(), host)
    assert (planned.encode_tiles, planned.decode_tiles, peaks.shrunk) == (
        Tiles(2048),
        Tiles(2048),
        None,
    )
    planned, peaks = planner.plan(round(80.5 * GIB), UHD, film, Pins(), host)
    assert (planned.encode_tiles, planned.decode_tiles) == (Tiles(2048), Tiles(2048))
    assert (planned.resident_encode, planned.resident_decode, planned.swapped) == (None, 34, 0)
    assert peaks.shrunk == planner.Shrunk(None, Tiles(2048), 1550, 1311)
    # 1080p on a 24 GB card, a shot of 301 frames, 8 GiB available: refused until then, every
    # window's VAE passes in tiles of 1856 and 1088 px sending the DiT away, its 36 blocks
    # pinned (test_host_memory); windows of 10 run, the VAE away, 25 of the DiT's blocks kept on
    # the GPU in the encode and all its unswapped ones in the decode, the encode in tiles of
    # 896 px, the decode of 640 (test_the_vae_kept_first_costs_no_window).
    planned, peaks = planner.plan(our_free(24), HD, shots(76), Pins(), 8 * GIB)
    assert (planned.window, planned.swapped, planned.vae_in_dit) == (10, 11, False)
    assert (planned.encode_tiles, planned.decode_tiles) == (Tiles(896), Tiles(640))
    assert (planned.resident_encode, planned.resident_decode) == (25, None)
    assert peaks.shrunk == planner.Shrunk(Tiles(1856), Tiles(1088), 1838, 800)
    assert peaks.shortened == planner.Shortened(17, 1481, 800)


def test_the_vae_kept_first_costs_no_window() -> None:
    # The VAE's weights kept on the GPU for the DiT's phase (test_the_vae_stays_where_the_host_
    # is_short) swap one or two blocks more, each pinned whatever the VAE's phases keep: the
    # tiles shrunk under that plan could reach no plan the host holds where they reach one with
    # the VAE away, and the window shortened, or the job was refused, for it (found by a grid
    # of plans, 2026-10-16). The tiles are shrunk from the plan as the choices made it too,
    # taken where the VAE kept doesn't fit: two of that grid's jobs, as it made them.
    def film(*frames: int) -> list[Shot]:
        ends = [sum(frames[: index + 1]) for index in range(len(frames))]
        return [Shot(end - count, end) for end, count in zip(ends, frames, strict=True)]

    # 4K, shots of 23, 24, 150 and 600 frames, 63.0 GiB free, 11.22 GiB of system memory:
    # windows of 12 latents were taken, the VAE kept; 13 run with it away, nothing of the DiT
    # leaving the GPU in the VAE's phases, its 16 swapped blocks and the VAE pinned.
    made, peaks = planner.plan(67_645_735_689, UHD, film(23, 24, 150, 600), Pins(), 12_050_198_642)
    assert (made.window, made.swapped, made.vae_in_dit) == (13, 16, False)
    assert (made.encode_tiles, made.decode_tiles) == (Tiles(2048), Tiles(1792))
    assert (made.resident_encode, made.resident_decode) == (None, None)
    assert peaks.shrunk == planner.Shrunk(Tiles(2048), Tiles(1920), 1473, 1122)
    # 5K, a shot of 149 frames, split, 14.32 GiB: refused with 43.0 GiB free; windows of 5 run.
    made, peaks = planner.plan(
        46_170_899_209, (2880, 5120), film(149), Pins(), 15_379_895_297, split=True
    )
    assert (made.window, made.swapped, made.vae_in_dit) == (5, 25, False)
    assert (made.encode_tiles, made.decode_tiles) == (Tiles(1984), Tiles(1472))
    assert peaks.shrunk == planner.Shrunk(Tiles(2048), Tiles(1600), 1836, 1432)
    # A byte under what that plan takes but its scratch, no window's plan fits, 5 latents being
    # the smallest: the refusal says the least any tiles take there, the VAE away, its 25
    # swapped blocks and the VAE's weights pinned, where the VAE kept pins 27 blocks.
    least = peaks.host - made.scratch_allowance
    assert refused(
        46_170_899_209, (2880, 5120), film(149), Pins(), least - 1, split=True
    ).startswith(
        "the plan takes 14.14 GiB of system memory (2.40 for the process, 11.02 pinned for 25"
        " blocks and the VAE's weights that leave the GPU, 0.72 for the longest shot"
    )
    # Where neither fits, the figures said are of the plan that takes the least, the VAE kept
    # or not: 720p, shots of 12 and 133 latents, 19 GiB free, 16.5 GiB of system memory. At the
    # card's window, 35 latents, the VAE kept (35 blocks swapped, the one unswapped block kept
    # in the decode, nothing more to keep) takes 17.30 GiB, and the VAE away with the decode's
    # tiles shrunk from 1216 to 1152 px 17.35: the plan's line says the 17.30, windows of 29
    # taken.
    made, peaks = planner.plan(19 * GIB, (720, 1280), shots(12, 133), Pins(), round(16.5 * GIB))
    assert made.window == 29 and peaks.shortened == planner.Shortened(35, 1730, 1650)
    # Where the VAE kept and the tiles shrunk fit, that is the plan, as before: the grid's job
    # whose window fell (test_on_a_short_host_the_tiles_shrink_before_the_window).
    made, _ = planner.plan(
        48_855_253_769, (1440, 2560), shots(5, 39), Pins(), 18_616_624_832, split=True
    )
    assert (made.window, made.swapped, made.vae_in_dit) == (21, 24, True)


def test_the_tiles_shrunk_for_the_host_are_the_largest_that_fit_it() -> None:
    # Over a grid of jobs, memories and hosts: a plan whose tiles the planner shrank for the
    # host's memory (Peaks.shrunk) fits the host, takes no pass's tiles over the card's own, and
    # keeps more of the DiT on the GPU than that plan; and each pass it shrank, made one tile
    # larger, its residency counted again, takes more system memory than the host has: the
    # tiles shrink no more than the host needs. Its window is the planner's, its own or the
    # longest whose plan fits both (test_the_planner_takes_the_longest_window_the_host_holds).
    fitting, resident = planner._fitting, planner._resident  # pyright: ignore[reportPrivateUsage]
    shrunk = 0
    films = (shots(2), shots(12), shots(40), shots(76), shots(5, 39))
    for target in ((720, 1280), HD, (1440, 2560), UHD):
        for film in films:
            for free in (round(count * GIB) for count in (20, 24, 30, 40, 48, 60, 80)):
                for host in (round(count * GIB) for count in (6, 10, 14, 18)):
                    try:
                        made, peaks = planner.plan(free, target, film, Pins(), host)
                    except PlanError:
                        continue
                    if peaks.shrunk is None:
                        continue
                    shrunk += 1
                    job = planner._job(free, target, film, False, False)  # pyright: ignore[reportPrivateUsage]
                    assert peaks.host <= host and fits(made, free, target, film, host) is None
                    alone = replace(
                        made,
                        encode_tiles=peaks.shrunk.encode,
                        decode_tiles=peaks.shrunk.decode,
                        resident_encode=resident(job, "encode", peaks.shrunk.encode, made.swapped),
                        resident_decode=resident(job, "decode", peaks.shrunk.decode, made.swapped),
                    )
                    assert planner._leaving(made) < planner._leaving(alone)  # pyright: ignore[reportPrivateUsage]
                    passes_shrunk: tuple[tuple[Phase, Tiles | None, Tiles | None], ...] = (
                        ("encode", made.encode_tiles, peaks.shrunk.encode),
                        ("decode", made.decode_tiles, peaks.shrunk.decode),
                    )
                    for phase, own, theirs in passes_shrunk:
                        passes = fitting(job, phase, made.swapped)
                        assert passes.index(theirs) <= passes.index(own)
                        if own == theirs:
                            continue
                        larger = passes[passes.index(own) - 1]
                        more = replace(
                            made,
                            **{
                                f"{phase}_tiles": larger,
                                f"resident_{phase}": resident(job, phase, larger, made.swapped),
                                "scratch_allowance": 0,
                            },
                        )
                        said = fits(more, free, target, film, host)
                        assert said is not None and "of system memory" in said, (phase, said)
    assert shrunk > 20  # 246 (2026-10-16)


def test_the_host_s_window_is_searched_from_the_next_one_down_to_the_smallest() -> None:
    # The two ends of the planner's search for the window the host holds
    # (planner._window_that_runs), each by a job a brute force over the planner found (a
    # review, 2026-10-12): the grid above has none at either end.
    # The first window under the card's own. 4K, shots of 63, 60, 89 and 53 latents, 80 GiB
    # free, --decode-tile 1280 and 17.33 GiB of system memory: the card alone holds windows of
    # 19, whose plan pins 35 blocks, 17.93 GiB; windows of 18 plan there and are taken, not
    # those of 17, which plan too.
    film, pins, host = shots(63, 60, 89, 53), Pins(decode_tile=1280), round(17.33 * GIB)
    alone, _ = planner.plan(80 * GIB, UHD, film, pins, None)
    made, peaks = planner.plan(80 * GIB, UHD, film, pins, host)
    assert (alone.window, made.window) == (19, 18)
    assert peaks.shortened == planner.Shortened(19, 1793, 1733)
    assert planner.plan(80 * GIB, UHD, film, replace(pins, window=17), host)[0].window == 17
    assert "of system memory" in refused(80 * GIB, UHD, film, replace(pins, window=19), host)
    # The smallest window the job can run. 1080p, a shot of 45 latents, a run held to its card
    # that counts on 8 GiB, under SDPA, 18 GiB of system memory: the card alone holds windows
    # of 6, every part of the weights pinned, 18.32 GiB; windows of 5, the smallest, plan, with
    # the VAE's weights kept on the GPU, and are taken, where the job was refused had the
    # search stopped a window short.
    held = Reading(13 * GIB, None, 8 * GIB, True, round(0.5 * GIB))
    job = (held, HD, shots(45))
    alone, _ = planner.plan(*job, Pins(), None, attention=planner.SDPA)
    made, peaks = planner.plan(*job, Pins(), 18 * GIB, attention=planner.SDPA)
    assert (alone.window, made.window, made.vae_in_dit) == (6, 5, True)
    assert peaks.shortened == planner.Shortened(6, 1832, 1800)
    assert "of system memory" in refused(*job, Pins(window=6), 18 * GIB, attention=planner.SDPA)


def test_scratch_allowance() -> None:
    # DESIGN.md, Memory planner, Host RAM: "The planner sets the scratch's allowance per job:
    # half of what the host has available once the pinned copies are counted, and no more than
    # the longest tiled shot needs"; none where no pass is tiled. Once what the run holds of the
    # longest shot is counted with the copies, ahead of the allowance. 1080p on a 24 GB card,
    # both passes tiled, every part of the weights pinned, 15.81 GiB: the longest shot's 301
    # frames take 3.57 GiB of scratch, and its 76 latents 0.17 GiB through its windows of 17.
    free, film = our_free(24), shots(5, 76, 17)
    pinned = 36 * paged(BLOCK) + paged(planner.DIT_OTHER) + paged(planner.VAE_WEIGHTS)
    kept = planner.scratch(OURS_1080, 301)
    held = 169 * 1_059_840

    def allowed(host: int | None) -> int:
        made, peaks = planner.plan(free, HD, film, Pins(), host)
        assert bare(made) == Plan(17, Tiles(1856), Tiles(1088), 28, 1, 0, vae_in_dit=False)
        assert peaks.scratch == kept and peaks.shot == held
        assert peaks.host == planner.HOST_RESIDENT + pinned + held + made.scratch_allowance
        assert host is None or peaks.host <= host
        return made.scratch_allowance

    assert allowed(64 * GIB) == allowed(24 * GIB) == kept
    assert allowed(23 * GIB) == (23 * GIB - pinned - held) // 2 < kept
    assert allowed(22 * GIB) == (22 * GIB - pinned - held) // 2
    assert (f"{gib(allowed(23 * GIB)):.2f}", f"{gib(allowed(22 * GIB)):.2f}") == ("3.51", "3.01")
    # No more than what the process's own 2.4 GiB, the pinned copies and the shot leave, which
    # half of what the last two leave passes under 4.8 GiB (PROVISIONAL): down to nothing, the
    # job never refused for its allowance.
    left = 19 * GIB - pinned - held
    assert allowed(19 * GIB) == left - planner.HOST_RESIDENT < left // 2
    assert f"{gib(allowed(19 * GIB)):.2f}" == "0.62"
    assert allowed(planner.HOST_RESIDENT + pinned + held) == 0
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
    assert made.record()["memory"]["scratch_allowance"] == (22 * GIB - pinned - held) // 2
    assert Plan.recorded(made.record()) == Plan(17, Tiles(1856), Tiles(1088))
    again, _ = planner.plan(free, HD, film, Pins(), 64 * GIB, recorded=made)
    assert again == replace(made, scratch_allowance=kept)
    # The plan's lines say it, and the shots whose scratch goes to its file instead, whole: of
    # 5, 76 and 17 latents, the one over 3.01 GiB.
    made, peaks = planner.plan(free, HD, film, Pins(), 64 * GIB)
    assert planner.said(made, peaks, "a card", HD, film, Pins())[-2] == (
        "plan: scratch of the longest tiled shot: 3.57 GiB; an allowance of 3.57 GiB of system"
        " memory, which holds every shot's"
    )
    made, peaks = planner.plan(free, HD, film, Pins(), 22 * GIB)
    assert planner.said(made, peaks, "a card", HD, film, Pins())[-2] == (
        "plan: scratch of the longest tiled shot: 3.57 GiB; an allowance of 3.01 GiB of system"
        " memory, over which the scratch of 1 of 3 shots goes whole to a memory-mapped file"
    )
    made, peaks = planner.plan(free, HD, film, Pins(), planner.HOST_RESIDENT + pinned + held)
    assert planner.said(made, peaks, "a card", HD, film, Pins())[-2] == (
        "plan: scratch of the longest tiled shot: 3.57 GiB; an allowance of 0.00 GiB of system"
        " memory, over which the scratch of 3 of 3 shots goes whole to a memory-mapped file"
    )
    # A plan from elsewhere is judged with the allowance it has (fits), said with the rest.
    large = replace(made, scratch_allowance=6 * GIB)
    assert fits(large, free, HD, film, 24 * GIB) == (
        "the plan takes 24.38 GiB of system memory (2.40 for the process, 15.81 pinned for 36"
        " blocks, the DiT's other parameters and the VAE's weights that leave the GPU, 0.17 for"
        " the longest shot, 301 frames (76 latents), at its longest window, its latent and its"
        " noise beside the window's output, 6.00 for a tiled shot's scratch) and 24.00 GiB are"
        " available, 0.38 GiB short"
    )
    assert fits(large, free, HD, film, 25 * GIB) is None
    # The plan's lines say the same figures, and "in all" the refusal's total, their sum.
    assert planner.said(large, peaks, "a card", HD, film, Pins())[5::3] == [
        "plan: 15.81 GiB of system memory pinned for 36 blocks, the DiT's other parameters and the"
        " VAE; 0.17 GiB for the longest shot at the most, at its longest window, its latent and"
        " its noise beside the window's output; 24.38 GiB of system memory in all",
        "plan: scratch of the longest tiled shot: 3.57 GiB; an allowance of 6.00 GiB of system"
        " memory, which holds every shot's",
    ]


def test_host_moments() -> None:
    # What the run holds of a shot in system memory at each of its moments
    # (planner.host_moments), by the tensors' sizes: a latent is 16 channels of bfloat16 over an
    # eighth of the padded frame a side, half a byte a padded pixel: 1,059,840 bytes at 1080p,
    # padded to 1104x1920, and 4,208,640 at 4K, padded to 2192x3840 (DESIGN.md, Memory planner,
    # Host RAM: 1.01 and 4.01 MiB). tests/test_shot.py measures the moments that run on the CPU.
    latent = planner.host_moments(OURS_1080, 1, None)["decode"]
    assert latent == 16 * 2 * (1104 // 8) * (1920 // 8) == 1104 * 1920 // 2 == 1_059_840
    assert planner.host_moments(OURS_4K, 1, None)["decode"] == 2192 * 3840 // 2 == 4_208_640
    assert (f"{latent / 2**20:.2f}", f"{4_208_640 / 2**20:.2f}") == ("1.01", "4.01")

    def moments(latents: int, cap: int | None) -> dict[str, float]:
        counted = planner.host_moments(OURS_1080, latents, cap)
        assert tuple(counted) == planner.MOMENTS
        return {moment: count / latent for moment, count in counted.items()}

    # In latents. A shot in one window, 40 latents: the latent and the buffer it is made from;
    # the latent, and of the one latent its check reads at a time its absolute values and three
    # masks of half its bytes each; the latent, the noise and the window's copy; the window,
    # which is the merged latent, at the decode's start and through it.
    one = {"encode": 80, "check": 42.5, "windows": 120, "merge": 40, "decode": 40}
    assert moments(40, None) == moments(40, 40) == moments(40, 44) == one
    # 48 latents under a cap of 30, two windows of 25 sharing 2: the longer window's copy
    # beside the latent and the noise; at the merge the two windows, the merged latent and, as
    # the 2 shared latents are mixed, their two products and the sum.
    assert shot_layout(189, 30) == [(0, 25), (23, 48)]
    two = {"encode": 96, "check": 50.5, "windows": 96 + 25, "merge": 50 + 48 + 6, "decode": 48}
    assert moments(48, 30) == two
    # Under a cap of 8, eight windows, six of 8 latents and two of 7, 62 latents in all: at the
    # merge, the windows, the merged latent and one join's mix, no more for eight joins than
    # for one.
    assert [end - start for start, end in shot_layout(189, 8)] == [8] * 6 + [7] * 2
    eight = {"encode": 96, "check": 50.5, "windows": 96 + 8, "merge": 62 + 48 + 6, "decode": 48}
    assert moments(48, 8) == eight
    # A shot of 1,000 latents under a cap of 5, the shortest: 332 windows of 5 and one of 4,
    # 1,664 latents. The merge holds the most, 2,670 latents for the windows' 2,005; until
    # 2026-10-12 it held 3,664, the shot made anew at each join, and the check 3,500, the
    # latent read whole.
    long = moments(1000, 5)
    assert (long["check"], long["windows"], long["merge"]) == (1002.5, 2005, 1664 + 1000 + 6)
    # Bytes a latent through a shot's one window, the moment that holds the most of it: 1.5 a
    # padded pixel, 3.03 MiB at 1080p and 12.04 at 4K. A shot of 1,000 latents takes 2.09 GiB
    # at 1080p in windows of 20 and 2.64 in windows of 5, as its decode merges them; at 4K
    # 10.47 in windows of 5. A minute's shot at 24 fps, 361 latents: 0.76 and 0.95 GiB at
    # 1080p, 3.79 at 4K in windows of 5.
    window = [planner.host_moments(padded, 1, None)["windows"] for padded in (OURS_1080, OURS_4K)]
    assert window == [3_179_520, 12_625_920] == [1104 * 1920 * 3 // 2, 2192 * 3840 * 3 // 2]
    assert [f"{count / 2**20:.2f}" for count in window] == ["3.03", "12.04"]

    def most(padded: tuple[int, int], latents: int, cap: int) -> str:
        return f"{gib(max(planner.host_moments(padded, latents, cap).values())):.2f}"

    assert [most(OURS_1080, 1000, cap) for cap in (20, 5)] == ["2.09", "2.64"]
    assert [most(OURS_1080, 361, cap) for cap in (20, 5)] == ["0.76", "0.95"]
    assert (most(OURS_4K, 1000, 5), most(OURS_4K, 361, 5)) == ("10.47", "3.79")
    assert planner.host_moments(OURS_4K, 1000, 5)["merge"] == 2670 * 4_208_640


def test_the_shot_that_holds_the_most() -> None:
    # The plan counts the system memory of one shot, the one that holds the most
    # (planner._shot_held): the longest, but where a shorter one in fewer windows holds more.
    # Under a cap, the most a shot holds can fall as the shot grows into one window more: 12
    # latents in one window keep 36 at the window's, 13 in two windows of 8 and 7 keep 34, at
    # the window's (26 and 8) as at the merge (15, 13 and 6). The moment that holds the most:
    # the check for a shot of 1 latent; then the windows'; and the merge once a shot's n windows
    # are shorter than 2 n + 4 latents.
    def most(latents: int, cap: int | None) -> tuple[int, str]:
        counted = planner.host_moments(OURS_1080, latents, cap)
        moment = max(planner.MOMENTS, key=lambda moment: counted[moment])
        return counted[moment], moment

    one, two = (planner.host_moments(OURS_1080, count, 12) for count in (12, 13))
    assert (one["windows"] // 1_059_840, two["windows"] // 1_059_840) == (36, 34)
    assert two["merge"] // 1_059_840 == 34 and most(13, 12)[0] < most(12, 12)[0]
    for cap in (None, 5, 6, 7, 8, 9, 12, 20, 44, 91):
        held = [most(latents, cap) for latents in range(1, 600)]
        moments = [moment for _, moment in held]
        assert moments[0] == "check" and moments[1] == "windows"
        for latents, moment in enumerate(moments[1:], 2):
            windows = shot_layout(4 * latents - 3, cap)
            longest = max(end - start for start, end in windows)
            merges = len(windows) > 1 and 2 * len(windows) + 4 > longest
            assert moment == ("merge" if merges else "windows"), (cap, latents)
        # It falls only where a shot takes one window more than the shot a latent shorter: 38
        # latents under a cap of 20 hold 96 at a window's, two of 20, and 39, in three of 15,
        # 14 and 14, hold 93.
        for (low, _), (high, _), latents in zip(held, held[1:], range(2, 600), strict=False):
            more = len(shot_layout(4 * latents - 3, cap)) > len(shot_layout(4 * latents - 7, cap))
            assert high >= low or more, (cap, latents)
        assert cap != 20 or [count // 1_059_840 for count, _ in held[37:39]] == [96, 93]
    # Through the planner: a film's plan counts the shot that holds the most under its cap.
    # 4K on a 24 GB card, windows of 5: the longest shot's merge, 2,670 latents of the shot of
    # 1,000; at 1080p, windows of 20, its merge too, 2,116.
    film = shots(40, 1000, 7)
    made, peaks = planner.plan(our_free(24), UHD, film, Pins(), None)
    assert made.window == 5 and (peaks.shot, peaks.moment) == (2670 * 4_208_640, "merge")
    made, peaks = planner.plan(our_free(24), HD, film, Pins(), None)
    assert made.window == 20 and (peaks.shot, peaks.moment) == (2116 * 1_059_840, "merge")
    assert peaks.record()["host_shot"] == peaks.shot and peaks.holder == 1000
    # A shot of the cap's length beside a longer one holds more than it: 1080p on a 96 GB
    # card, shots of 76 and 77 latents under --window 76: the first holds 228 latents' worth
    # through its one window, the second, in two windows of 40 and 39, 194. The plan's line and
    # a refusal name that shot.
    film = shots(76, 77)
    made, peaks = planner.plan(our_free(96), HD, film, Pins(window=76), None)
    assert (made.window, peaks.holder, peaks.moment) == (76, 76, "windows")
    assert peaks.shot == 228 * 1_059_840 > max(planner.host_moments(OURS_1080, 77, 76).values())
    assert max(planner.host_moments(OURS_1080, 77, 76).values()) == 194 * 1_059_840
    assert planner.said(made, peaks, "a card", HD, film, Pins(window=76))[5] == (
        "plan: no system memory pinned, nothing leaving the GPU; 0.23 GiB for a shot of 76"
        " latents at the most, more than the longest, at its longest window, its latent and its"
        " noise beside the window's output; 2.63 GiB of system memory in all"
    )
    assert refused(our_free(96), HD, film, Pins(window=76), 2 * GIB) == (
        "the plan takes 2.63 GiB of system memory (2.40 for the process, 0.23 for a shot of 76"
        " latents, which holds more than the longest, at its longest window, its latent and its"
        " noise beside the window's output) and 2.00 GiB are available, 0.63 GiB short"
    )
    # Of two shots that hold as much, the longer is named: under a cap of 12, a shot of 12
    # latents in one window and one of 14 in two windows of 8 hold 36 latents' worth each, and
    # the plan's line says the longest shot's. One shot's is the longest's.
    most_held = planner._most_held  # pyright: ignore[reportPrivateUsage]
    assert most_held(OURS_1080, (12, 13), 12) == (36 * 1_059_840, "windows", 12)
    assert most_held(OURS_1080, (13,), 12) == (34 * 1_059_840, "windows", 13)
    twelve, fourteen = (planner.host_moments(OURS_1080, count, 12) for count in (12, 14))
    assert max(twelve.values()) == max(fourteen.values()) == 36 * 1_059_840
    assert most_held(OURS_1080, (12, 14), 12) == (36 * 1_059_840, "windows", 14)
    film = shots(12, 14)
    made, peaks = planner.plan(our_free(96), HD, film, Pins(window=12), None)
    assert (peaks.shot, peaks.holder) == (36 * 1_059_840, 14)
    assert (
        " for the longest shot at the most, at its longest window, "
        in (planner.said(made, peaks, "a card", HD, film, Pins(window=12))[5])
    )
    # The moment that holds the most is one of three: the encode's end and the decode never
    # are, and have no sentence (the loop above reads shots of 1 to 599 latents under ten caps).
    assert set(planner._MOMENT_SAID) == {"check", "windows", "merge"}  # pyright: ignore[reportPrivateUsage]
    assert set(planner.MOMENTS) - set(planner._MOMENT_SAID) == {"encode", "decode"}  # pyright: ignore[reportPrivateUsage]
    # Every length is read that could hold the most: over many films, the most of every shot's
    # moments, counted here.
    for cap in (None, 5, 9, 20, 44):
        for lengths in ((1,), (3, 4), (5, 6, 7), (9, 10, 11, 40), (19, 20, 21, 22, 300), (44, 45)):
            counted = [planner.host_moments(OURS_1080, count, cap) for count in lengths]
            assert most_held(OURS_1080, lengths, cap)[0] == max(
                max(moments.values()) for moments in counted
            )


def test_a_long_shot_the_host_cannot_hold() -> None:
    # A job whose longest shot fits the GPU and not the host: 4K on a 24 GB card, a shot of
    # 3,997 frames, 1,000 latents, in windows of 5, the shortest: no shorter window to take.
    # Every part of the weights is pinned, 15.81 GiB, and the shot holds 10.47 GiB as its decode
    # merges its 333 windows: 28.68 GiB with the process, refused on a host with 28 GiB
    # available, naming each part. (14.36 GiB and 32.57 until 2026-10-12, the shot made anew at
    # each join.)
    free, film = our_free(24), shots(1000)
    pinned = 36 * paged(BLOCK) + paged(planner.DIT_OTHER) + paged(planner.VAE_WEIGHTS)
    held = 2670 * 4_208_640
    assert refused(free, UHD, film, Pins(), 28 * GIB) == (
        "the plan takes 28.68 GiB of system memory (2.40 for the process, 15.81 pinned for 36"
        " blocks, the DiT's other parameters and the VAE's weights that leave the GPU, 10.47 for"
        " the longest shot, 3997 frames (1000 latents), as its decode merges its windows) and"
        " 28.00 GiB are available, 0.68 GiB short"
    )
    # It plans from what the three take, its tiled passes' scratch then allowed nothing, and
    # on a host whose memory isn't read; and the same shot at 1080p, 2.09 GiB as its decode
    # merges its windows of 20, plans in 32 GiB with its scratch allowed half of what is left.
    least = planner.HOST_RESIDENT + pinned + held
    made, peaks = planner.plan(free, UHD, film, Pins(), least)
    assert (made.window, made.scratch_allowance, peaks.host) == (5, 0, least)
    assert (made.encode_tiles, made.decode_tiles) == (Tiles(1472), Tiles(1088))
    assert refused(free, UHD, film, Pins(), least - 1).startswith("the plan takes 28.68 GiB of")
    # The fit of a plan from elsewhere counts the shot too (fits).
    short = fits(made, free, UHD, film, least - 1)
    assert short is not None and "as its decode merges its windows) and" in short
    assert fits(made, free, UHD, film, least) is None
    unread, peaks = planner.plan(free, UHD, film, Pins(), None)
    assert bare(unread) == bare(made) and unread.scratch_allowance == 8 * GIB
    assert peaks.host == least + 8 * GIB
    made, peaks = planner.plan(free, HD, film, Pins(), 32 * GIB)
    left = 32 * GIB - pinned - 2116 * 1_059_840
    assert (made.window, peaks.moment, made.scratch_allowance) == (20, "merge", left // 2)
    assert f"{gib(peaks.shot):.2f}" == "2.09" and peaks.host <= 32 * GIB
    assert f"{gib(left // 2):.2f}" == "7.05"
    # And with the allowance the plan has.
    assert fits(made, free, HD, film, 32 * GIB) is None
    assert fits(made, free, HD, film, peaks.host - 1) == (
        "the plan takes 27.35 GiB of system memory (2.40 for the process, 15.81 pinned for 36"
        " blocks, the DiT's other parameters and the VAE's weights that leave the GPU, 2.09 for"
        " the longest shot, 3997 frames (1000 latents), as its decode merges its windows, 7.05"
        " for a tiled shot's scratch) and 27.34 GiB are available, 0.01 GiB short"
    )


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
    recorded = Plan(17, Tiles(1856), Tiles(1088), 28, 1, 0, vae_in_dit=False)
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
    # with 0.58 GiB less, two more blocks swapped, and one fewer left in the decode.
    smaller = Plan(17, Tiles(1088), Tiles(1024))
    made, _ = planner.plan(our_free(24), *job, recorded=smaller)
    assert bare(made) == Plan(17, Tiles(1088), Tiles(1024), 28, None, 6, vae_in_dit=False)
    made, _ = planner.plan(round(21.99 * GIB), *job, recorded=smaller)
    assert bare(made) == Plan(17, Tiles(1088), Tiles(1024), 30, None, 4, vae_in_dit=False)
    # The recorded tiles themselves take 22.03 GiB of free VRAM with the DiT away, their room
    # counted (the decode's; the encode's take 21.83, where 1.5 GiB of room made them the
    # larger, 22.08), whatever the shot's length: with less, the shape no longer fits.
    assert refused(round(21.99 * GIB), *job, recorded=recorded) == (
        "the recorded plan no longer fits the memory free now: the decode, in tiles of 1088 px,"
        " with the DiT off the GPU, takes 22.03 GiB of free VRAM, 1.25 of them the room a VAE pass"
        " is held to, and the driver reports 21.99 GiB free, 0.04 GiB short"
    )
    # Its tiles too, their overlap with them, and its compile flags, whatever would be chosen.
    odd = Plan(8, Tiles(768, 128), Tiles(576, 96), compile_dit=True)
    made, _ = planner.plan(our_free(96), *job, recorded=odd)
    assert bare(made) == odd
    # A recorded shape that no longer fits stops the resume, naming the phase and the shortfall,
    # and never another pin: here its encode, then its windows.
    assert refused(our_free(16), *job, recorded=recorded) == (
        "the recorded plan no longer fits the memory free now: the encode, in tiles of 1856 px,"
        " with the DiT off the GPU, takes 21.83 GiB of free VRAM, 1.25 of them the room a VAE pass"
        " is held to, and the driver reports 14.57 GiB free, 7.26 GiB short"
    )
    # A pass recorded is held to its room as one planned is: on 24 GB at 1080p an untiled
    # encode's peak fits with the DiT away, and the shape doesn't.
    assert refused(our_free(24), *job, recorded=Plan(17, None, Tiles(1088))) == (
        "the recorded plan no longer fits the memory free now: the encode, untiled, with the"
        " DiT off the GPU, takes 22.63 GiB of free VRAM, 1.25 of them the room a VAE pass is held"
        " to, and the driver reports 22.57 GiB free, 0.06 GiB short"
    )
    tiled = Plan(20, Tiles(768), Tiles(640))
    assert refused(our_free(16), *job, recorded=tiled) == (
        "the recorded plan no longer fits the memory free now: the DiT's phase, a window of 17"
        " latents with 36 blocks swapped and the VAE off the GPU, takes 19.05 GiB of free VRAM and"
        " the driver reports 14.57 GiB free, 4.48 GiB short"
    )
    assert refused(our_free(24), HD, shots(76), Pins(), 8 * GIB, recorded=recorded) == (
        "the recorded plan no longer fits the memory free now: the plan takes 17.92 GiB of system"
        " memory (2.40 for the process, 15.35 pinned for 36 blocks and the DiT's other parameters"
        " that leave the GPU, 0.17 for the longest shot, 301 frames (76 latents), at its longest"
        " window, its latent and its noise beside the window's output) and 8.00 GiB are"
        " available, 9.92 GiB short"
    )
    # --swap-blocks still applies, taken exactly; where it alone doesn't hold the recorded
    # windows, it is the pin that is refused.
    made, _ = planner.plan(our_free(24), HD, shots(76), Pins(swap_blocks=36), None, recorded=tiled)
    assert bare(made) == Plan(20, Tiles(768), Tiles(640), 36)
    pins = Pins(swap_blocks=3)
    assert refused(our_free(24), HD, shots(76), pins, None, recorded=recorded) == (
        "--swap-blocks 3: a window of 17 latents, the recorded plan's, takes 32.98 GiB of free"
        " VRAM with 3 blocks swapped, and the driver reports 22.57 GiB free: the fewest blocks"
        " that hold it are --swap-blocks 28"
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
    assert fits(Plan(12), free, *job) == (
        "the encode, untiled, with the DiT's 36 blocks on the GPU, takes 37.98 GiB of free VRAM,"
        " 1.25 of them the room a VAE pass is held to, and the driver reports 22.57 GiB free, 15.41"
        " GiB short"
    )
    # Nothing moving, the slots stay with the blocks that aren't swapped; a phase entered with a
    # number of blocks, all of them here, frees them: two blocks' 0.84 GiB less.
    assert fits(Plan(12, swapped=17), free, *job) == (
        "the encode, untiled, with the DiT's 19 unswapped blocks and its 2 slots on the GPU, takes"
        " 31.65 GiB of free VRAM, 1.25 of them the room a VAE pass is held to, and the driver"
        " reports 22.57 GiB free, 9.08 GiB short"
    )
    assert fits(Plan(12, swapped=17, resident_encode=19), free, *job) == (
        "the encode, untiled, with 19 of the DiT's blocks on the GPU, takes 30.81 GiB of free"
        " VRAM, 1.25 of them the room a VAE pass is held to, and the driver reports 22.57 GiB free,"
        " 8.24 GiB short"
    )
    away = Plan(12, Tiles(1600), None, 17, 0, 0, vae_in_dit=False)
    assert fits(away, free, *job) == (
        "the decode, untiled, with the DiT off the GPU, takes 37.31 GiB of free VRAM, 1.25 of"
        " them the room a VAE pass is held to, and the driver reports 22.57 GiB free, 14.74 GiB"
        " short"
    )
    assert fits(replace(away, swapped=3), free, *job) == (
        "the DiT's phase, a window of 12 latents with 3 blocks swapped and the VAE off the GPU,"
        " takes 28.04 GiB of free VRAM and the driver reports 22.57 GiB free, 5.47 GiB short"
    )
    tiled = replace(away, decode_tiles=Tiles(1088))
    assert fits(tiled, free, *job) is None
    # The system memory, with what the run holds of the shot, 12 latents through its one
    # window, and what the plan allows its scratch: nothing here, then the shot's.
    assert fits(tiled, free, HD, shots(12), 4 * GIB) == (
        "the plan takes 18.25 GiB of system memory (2.40 for the process, 15.81 pinned for 36"
        " blocks, the DiT's other parameters and the VAE's weights that leave the GPU, 0.04 for"
        " the longest shot, 45 frames (12 latents), at its longest window, its latent and its"
        " noise beside the window's output) and 4.00 GiB are available, 14.25 GiB short"
    )
    allowed = replace(tiled, scratch_allowance=planner.scratch(OURS_1080, 45))
    assert fits(allowed, free, *job) is None
    assert fits(allowed, free, HD, shots(12), 4 * GIB) == (
        "the plan takes 18.78 GiB of system memory (2.40 for the process, 15.81 pinned for 36"
        " blocks, the DiT's other parameters and the VAE's weights that leave the GPU, 0.04 for"
        " the longest shot, 45 frames (12 latents), at its longest window, its latent and its"
        " noise beside the window's output, 0.53 for a tiled shot's scratch) and 4.00 GiB are"
        " available, 14.78 GiB short"
    )
    # One block left unswapped is one block, and with every block swapped the slots alone stay,
    # with the DiT's other parameters.
    assert fits(Plan(12, Tiles(1600), swapped=35), free, *job) == (
        "the decode, untiled, with the DiT's 1 unswapped block and its 2 slots on the GPU, takes"
        " 38.73 GiB of free VRAM, 1.25 of them the room a VAE pass is held to, and the driver"
        " reports 22.57 GiB free, 16.16 GiB short"
    )
    assert fits(Plan(12, Tiles(1600), swapped=36), free, *job) == (
        "the decode, untiled, with the DiT's 2 slots on the GPU, takes 38.31 GiB of free VRAM,"
        " 1.25 of them the room a VAE pass is held to, and the driver reports 22.57 GiB free, 15.74"
        " GiB short"
    )
    assert fits(replace(tiled, resident_decode=1), free, *job) == (
        "the decode, in tiles of 1088 px, with 1 of the DiT's blocks on the GPU, takes 22.61 GiB"
        " of free VRAM, 1.25 of them the room a VAE pass is held to, and the driver reports 22.57"
        " GiB free, 0.04 GiB short"
    )
    # With split, its slice is an untiled decode's too: on 48 GB 21 blocks fit beside it
    # without it, 19 with it. A tiled decode's peak doesn't hold it: the same plan fits.
    assert fits(tiled, free, *job, split=True) is None
    beside = Plan(12, resident_decode=21)
    assert fits(beside, our_free(48), *job) is None
    assert fits(replace(beside, resident_decode=22), our_free(48), *job) is not None
    assert fits(beside, our_free(48), *job, split=True) == (
        "the decode, untiled, with 21 of the DiT's blocks on the GPU, takes 47.02 GiB of free"
        " VRAM, 1.25 of them the room a VAE pass is held to, and the driver reports 46.57 GiB free,"
        " 0.45 GiB short"
    )
    assert fits(replace(beside, resident_decode=20), our_free(48), *job, split=True) == (
        "the decode, untiled, with 20 of the DiT's blocks on the GPU, takes 46.60 GiB of free"
        " VRAM, 1.25 of them the room a VAE pass is held to, and the driver reports 46.57 GiB free,"
        " 0.03 GiB short"
    )
    assert fits(replace(beside, resident_decode=19), our_free(48), *job, split=True) is None


def test_a_resume_names_the_pin_s_part_of_its_shortfall() -> None:
    # A job resumed under --swap-blocks whose recorded shape fits under no count of swapped
    # blocks: the pin can't be refused for another count, as it is where one holds the shape
    # (test_resumed_around_the_recorded_shape), and the DiT's phase is counted with the blocks
    # it pins. The refusal says which part of the shortfall is the pin's, and what the shape is
    # short of with every block swapped and the VAE away: a window of 21 latents at 1080p with
    # 22 GiB free, 1.07 GiB short, and 15.41 with --swap-blocks 0.
    recorded = Plan(21, Tiles(512), Tiles(512))
    free, job = 22 * GIB, (HD, shots(40))
    start = "the recorded plan no longer fits the memory free now: the DiT's phase, a window of 21"
    shape = (
        f"{start} latents with 36 blocks swapped and the VAE off the GPU, takes 23.07 GiB of free"
        " VRAM and the driver reports 22.00 GiB free, 1.07 GiB short"
    )
    assert refused(free, *job, Pins(), None, recorded=recorded) == shape
    assert refused(free, *job, Pins(swap_blocks=36), None, recorded=recorded) == shape
    assert refused(free, *job, Pins(swap_blocks=0), None, recorded=recorded) == (
        f"{start} latents with 0 blocks swapped and the VAE off the GPU, takes 37.41 GiB of free"
        " VRAM and the driver reports 22.00 GiB free, 15.41 GiB short, 14.34 GiB of them"
        " --swap-blocks 0's: with every block of the DiT swapped and the VAE off the GPU, the"
        " recorded plan is 1.07 GiB short"
    )
    assert refused(free, *job, Pins(swap_blocks=20), None, recorded=recorded) == (
        f"{start} latents with 20 blocks swapped and the VAE off the GPU, takes 29.82 GiB of free"
        " VRAM and the driver reports 22.00 GiB free, 7.82 GiB short, 6.75 GiB of them"
        " --swap-blocks 20's: with every block of the DiT swapped and the VAE off the GPU, the"
        " recorded plan is 1.07 GiB short"
    )
    # On a run held to the card, after what would have to be freed, where memory freed on the
    # card can give it: the whole card's figure the one the plan is held under, 17 GiB under
    # the driver's here. The plan counts on its budget, no margin on top.
    kept = Reading(round(45.05 * GIB), round(28.05 * GIB), free - MARGIN, True, 3 * GIB)
    assert refused(kept, *job, Pins(swap_blocks=0), None, recorded=recorded).endswith(
        ", the run held to the card's own memory, 15.41 GiB short, which would have to be freed"
        " on the card, 14.34 GiB of them --swap-blocks 0's: with every block of the DiT swapped"
        " and the VAE off the GPU, the recorded plan is 1.07 GiB short"
    )
    # Where it can't, the card's figure not read: what it is short of, and no more.
    kept = Reading(round(28.05 * GIB), None, free - MARGIN, True, 3 * GIB)
    assert refused(kept, *job, Pins(swap_blocks=0), None, recorded=recorded).endswith(
        ", the run held to the card's own memory, 15.41 GiB short, 14.34 GiB of them"
        " --swap-blocks 0's: with every block of the DiT swapped and the VAE off the GPU, the"
        " recorded plan is 1.07 GiB short"
    )
    # Another phase refused first is the shape's own, whatever is pinned: no part of the pin's.
    small = refused(round(6.25 * GIB), *job, Pins(swap_blocks=0), None, recorded=recorded)
    assert small.startswith("the recorded plan no longer fits the memory free now: the encode, in")
    assert "--swap-blocks" not in small
    # Nor has a new job's refusal any: there the pin is refused, naming the fewest blocks.
    assert "of them" not in refused(free, *job, Pins(swap_blocks=0, window=21), None)


def test_peaks_record() -> None:
    # What the manifest keeps of a plan's memory, for information: bytes, the window in latents.
    made, peaks = planner.plan(our_free(24), HD, shots(76), Pins(), 64 * GIB)
    assert peaks.record() == {
        "free": our_free(24),
        # A driver that refuses: what it reports free is what the plan was made in, and the
        # whole card's free memory isn't read.
        "driver_free": our_free(24),
        "card_free": None,
        "attention": "flash_attn_2",
        "budget": our_free(24) - MARGIN,
        "encode_peak": planner.encode_peak(OURS_1080, Tiles(1856), 1, 28),
        # The longest window's.
        "dit_peak": planner.dit_peak(OURS_1080, 17, 28, vae_in_dit=False),
        "decode_peak": planner.decode_peak(OURS_1080, HD, Tiles(1088), 0, 28),
        "bound": "dit",
        "host": peaks.host,
        # Of it, the longest shot's part: its 76 latents and their noise beside a window of 17.
        "host_shot": (2 * 76 + 17) * 1_059_840,
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
        "plan: frames padded to 1920x1104, 8,280 tokens a latent, attention on FlashAttention 2",
        "plan: 1 shot, the longest of 301 frames (76 latents)",
        "plan: windows of 17 latents at most: 0 of 1 shot in one window, 4 joins",
        "plan: 28 of the DiT's 36 blocks swapped, the VAE off the GPU in the DiT's phase; in the"
        " encode, 1 of the DiT's blocks on the GPU; in the decode, the DiT off the GPU",
        "plan: 15.81 GiB of system memory pinned for 36 blocks, the DiT's other parameters and the"
        " VAE; 0.17 GiB for the longest shot at the most, at its longest window, its latent and"
        " its noise beside the window's output; 21.95 GiB of system memory in all",
        "plan: encode in tiles of 1856 px, 1 x 2, decode in tiles of 1088 px, 2 x 2, each held"
        " 1.25 GiB under the budget",
        "plan: predicted peaks: encode 20.55 GiB, DiT 21.82 GiB, decode 20.17 GiB; the DiT's phase"
        " bounds the plan, 0.14 GiB under the budget",
        "plan: scratch of the longest tiled shot: 3.57 GiB; an allowance of 3.57 GiB of system"
        " memory, which holds every shot's",
        "plan: nothing pinned",
    ]
    # 4K on the 96 GB card as DESIGN.md first read it, 94.34 GiB free, the same shot, pinned:
    # the untiled encode bounds the plan with the room a VAE pass is held to, which the line
    # says; the peaks are the predicted ones, without the room. Nothing leaves the GPU for it,
    # whatever the shot's length.
    pins = Pins(window=19, swap_blocks=0, encode_tile=0, decode_tile=2048)
    made, peaks = planner.plan(round(94.34 * GIB), UHD, shots(76), pins, 64 * GIB, split=True)
    assert planner.said(made, peaks, "a card", UHD, shots(76), pins)[3:] == [
        "plan: windows of 19 latents at most, the longest run 17 latents: 0 of 1 shot in one"
        " window, 4 joins",
        "plan: no block of the DiT swapped, the VAE on the GPU in the DiT's phase; in the encode,"
        " the DiT's 36 blocks on the GPU; in the decode, the DiT's 36 blocks on the GPU",
        "plan: no system memory pinned, nothing leaving the GPU; 0.66 GiB for the longest shot at"
        " the most, at its longest window, its latent and its noise beside the window's output;"
        " 17.22 GiB of system memory in all",
        "plan: encode untiled, decode in tiles of 2048 px, 2 x 2, each held 1.25 GiB under the"
        " budget",
        "plan: predicted peaks: encode 91.54 GiB, DiT 84.72 GiB, decode 79.22 GiB; the encode"
        " bounds the plan, its room counted, 0.95 GiB under the budget",
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
        " 0.12 GiB for the longest shot at the most, at its longest window, its latent and its"
        " noise beside the window's output; 17.87 GiB of system memory in all",
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


# What a run held to its card counts on in these tests, on a driver that gives memory the card
# doesn't have: the free memory less 5.9 GiB, figures of the tests' own: 0.8 GiB left outside
# torch's limit, and 5.1 that the plan leaves torch's allocator beyond a peak, the planner's
# margin in it. The module's two constants (memory.HELD_BACK and memory.RESERVE) are measured or
# decided, and move with their measures.
LEFT = round(0.8 * GIB)
KEPT = round(5.3 * GIB) + MARGIN


def held(reported: float, card: float | None) -> Reading:
    """The reading of a run held to the card, from figures in GiB: what the driver reports free
    to the process, and what the whole card has free (None: not read); the plan counts on the
    smaller, less KEPT, LEFT of them what torch's limit leaves."""
    driver = round(reported * GIB)
    whole = None if card is None else round(card * GIB)
    return Reading(driver, whole, min(driver, whole or driver) - KEPT, True, LEFT)


# The memory a plan is made in, each time a budget of 22.15 GiB, and how its words name it: the
# plan's first line after the card's name, and what a refusal says is there. A driver that
# refuses: one figure, what it reports free, 22.75 GiB, the budget the planner's margin under
# it. A run held to the card: what the plan counts on, 22.15 GiB, which is its budget, no margin
# on top (DESIGN.md, Allocator: the margin is counted inside what torch's allocator is left);
# of which figure, the driver's or the whole card's, with the other, in runtime/memory.py's own
# words; the whole card's wasn't read, is the smaller of the two, or the larger; and what
# torch's limit leaves under it apart from what the plan leaves again, two figures that were
# given as one. Then what the driver reported, in GiB, as both say it; and what memory freed on
# the card could give the plan, in GiB: nothing on a driver that refuses, nor where the card's
# figure isn't the one the plan is held under.
HELD_LINE = (
    "; the run is held to the card's own memory, its driver giving more than the card has:"
    " torch is limited 0.80 GiB under that, and the plan counts on 5.10 GiB less again, what it"
    " leaves torch's allocator beyond a step's peak: a budget of 22.15 GiB"
)
WORDS = [
    (
        round(22.75 * GIB),
        "22.75 GiB free, a budget of 22.15 GiB",
        "the driver reports 22.75 GiB free",
        22.75,
        0.0,
    ),
    (
        held(28.05, None),
        "28.05 GiB free as its driver reports it to this process (the whole card's, NVML's,"
        f" wasn't read){HELD_LINE}",
        "the plan counts on 22.15 GiB of the 28.05 GiB free as its driver reports it to this"
        " process (the whole card's, NVML's, wasn't read), the run held to the card's own memory",
        28.05,
        0.0,
    ),
    (
        held(30.03, 28.05),
        "28.05 GiB free on the whole card by NVML (its driver reports 30.03 GiB free to this"
        f" process){HELD_LINE}",
        "the plan counts on 22.15 GiB of the 28.05 GiB free on the whole card by NVML (its"
        " driver reports 30.03 GiB free to this process), the run held to the card's own memory",
        30.03,
        1.98,
    ),
    (
        held(28.05, 29.20),
        "28.05 GiB free as its driver reports it to this process (the whole card has 29.20 GiB"
        f" free by NVML){HELD_LINE}",
        "the plan counts on 22.15 GiB of the 28.05 GiB free as its driver reports it to this"
        " process (the whole card has 29.20 GiB free by NVML), the run held to the card's own"
        " memory",
        28.05,
        0.0,
    ),
]


@pytest.mark.parametrize(("free", "line", "there", "driver", "freeable"), WORDS)
def test_each_figure_of_the_free_memory_is_named(
    free: int | Reading, line: str, there: str, driver: float, freeable: float
) -> None:
    # DESIGN.md, Memory planner, Allocator: on a driver that gives memory the card doesn't have
    # the plan counts on less than the driver reports, and "the driver reports X GiB free" of
    # that figure would be false: the plan's first line and its five refusals name each figure
    # for what it is, and a refusal says what would have to be freed on the card for what was
    # asked, where memory freed there can give it, and what it is short of where it can't (the
    # card's figure not read, the driver's the smaller, or more than the two are apart). On a
    # driver that refuses they say what they said. What something takes is said without the
    # planner's margin on a held run, 0.60 GiB less: the plan has none there.
    is_held = isinstance(free, Reading)
    film = shots(31)

    def takes(refusing: str) -> str:
        return refusing if not is_held else f"{(round(float(refusing) * 100) - 60) / 100:.2f}"

    def to_free(short: str) -> str:
        if not is_held:
            return ""
        if float(short) > freeable:
            return f"{short} GiB short; "
        return f"{short} GiB more would have to be freed on the card for it; "

    def freed(short: str) -> str:
        return ", which would have to be freed on the card" if float(short) <= freeable else ""

    margin = "" if is_held else ", and 0.60 of margin"
    # The plan is the one of a budget of 22.15 GiB, whatever names the figure; its record keeps
    # what the plan was made in as `free`, what the driver reported and what the card had
    # beside it.
    made, peaks = planner.plan(free, HD, film, Pins(), None)
    assert (made, peaks.budget) == (
        planner.plan(round(22.75 * GIB), HD, film, Pins(), None)[0],
        24427626496 - MARGIN,
    )
    assert peaks.free == peaks.budget + (0 if is_held else MARGIN)
    said = planner.said(made, peaks, "NVIDIA GeForce RTX 5090", HD, film, Pins())
    assert said[0] == f"plan: NVIDIA GeForce RTX 5090, {line}"
    assert (
        said[1:]
        == planner.said(
            *planner.plan(round(22.75 * GIB), HD, film, Pins(), None), "a card", HD, film, Pins()
        )[1:]
    )
    record = peaks.record()
    if isinstance(free, Reading):
        assert (record["free"], record["driver_free"], record["card_free"]) == (
            free.counted,
            free.reported,
            free.card,
        )
        assert record["budget"] == free.counted
    else:
        assert (record["free"], record["driver_free"], record["card_free"]) == (free, free, None)
        # A number is a driver that refuses, which reports the card's free memory.
        assert planner.plan(Reading(free, None, free, False), HD, film, Pins(), None) == (
            made,
            peaks,
        )
    # --window refused, here with --swap-blocks pinned beside it.
    window_refused = refused(free, HD, film, Pins(swap_blocks=0, window=30), None)
    assert window_refused == (
        f"--window 30: its longest window, 17 latents, takes {takes('33.40')} GiB of free VRAM"
        f" with --swap-blocks 0, and {there}: {to_free('10.65')}the longest window that fits is"
        " --window 6"
    )
    # What the driver reported is read from either, the plan's lines as a run logs them and a
    # refusal, by the regression's unpinned case, which a small card alone skips
    # (tests/test_regression.py, reported).
    logged = "".join(f"12:00:00 INFO seedvr2x.cli: {said_line}\n" for said_line in said)
    assert test_regression.reported(logged) == driver
    assert test_regression.reported(f"12:00:00 ERROR seedvr2x.cli: {window_refused}\n") == driver
    # --swap-blocks refused, on a resume whose recorded window the blocks pinned don't hold:
    # 1.69 GiB short, which memory freed on the card gives where the card's figure is 1.98 GiB
    # under the driver's.
    recorded = Plan(8, Tiles(1856), Tiles(1088))
    assert refused(free, HD, film, Pins(swap_blocks=0), None, recorded=recorded) == (
        f"--swap-blocks 0: a window of 8 latents, the recorded plan's, takes {takes('24.44')} GiB"
        f" of free VRAM with 0 blocks swapped, and {there}: {to_free('1.69')}the fewest blocks"
        " that hold it are --swap-blocks 6"
    )
    # A tile pin refused.
    assert refused(free, HD, film, Pins(decode_tile=1856), None) == (
        f"--decode-tile 1856: the decode in tiles of 1856 px takes {takes('34.76')} GiB of free"
        f" VRAM, the DiT off the GPU, 1.25 of them the room a VAE pass is held to, and {there}:"
        f" {to_free('12.01')}the largest tile that fits is --decode-tile 1088"
    )
    # A size that doesn't fit.
    assert refused(free, (4320, 7680), shots(5), Pins(), None) == (
        f"an output of 7680x4320 takes {takes('83.71')} GiB of free VRAM (its smallest window, 5"
        " latents, with every block of the DiT swapped and the VAE off the GPU: a peak of 83.11"
        f" GiB{margin}), and {there}: {to_free('60.96')}the largest output that fits there is"
        " --resolution 2160, about 3840x2160"
    )
    # A plan that doesn't fit, and a recorded one that no longer does: the shortfall is what
    # would have to be freed.
    assert fits(Plan(12), free, HD, shots(12), None) == (
        f"the encode, untiled, with the DiT's 36 blocks on the GPU, takes {takes('37.98')} GiB of"
        f" free VRAM, 1.25 of them the room a VAE pass is held to, and {there}, 15.23 GiB short"
        f"{freed('15.23')}"
    )
    recorded = Plan(31, Tiles(1856), Tiles(1088))
    assert refused(free, HD, film, Pins(), None, recorded=recorded) == (
        "the recorded plan no longer fits the memory free now: the DiT's phase, a window of 31"
        f" latents with 36 blocks swapped and the VAE off the GPU, takes {takes('33.19')} GiB of"
        f" free VRAM and {there}, 10.44 GiB short{freed('10.44')}"
    )


def test_a_held_run_takes_no_margin_on_top() -> None:
    # DESIGN.md, Allocator (2026-10-12): "On such a driver the planner's 0.6 GiB margin is
    # counted inside the 4.5, which measure the same need: with both, 4K's smallest window is
    # refused on that card beside its usual desktop." What a held run counts on is its budget:
    # a phase that takes it to the byte is planned, and one byte more refused; on a driver that
    # refuses the margin stands over the same peak.
    film = shots(5)
    peak = planner.dit_peak(padded_size(HD), 5, 0, False)
    pins = Pins(window=5, swap_blocks=0)

    def on_card(counted: int) -> Reading:
        return Reading(counted + round(5.0 * GIB), None, counted, True, round(0.5 * GIB))

    made, peaks = planner.plan(on_card(peak), HD, film, pins, None)
    assert (peaks.dit, peaks.free, peaks.budget) == (peak, peak, peak)
    lines = planner.said(made, peaks, "a card", HD, film, pins)
    assert sum(
        "the DiT's phase bounds the plan, 0.00 GiB under the budget" in line for line in lines
    )
    assert refused(on_card(peak - 1), HD, film, pins, None).startswith("--swap-blocks 0: a window")
    # The same peak on a driver that refuses: its margin over it, to the byte.
    made, peaks = planner.plan(peak + MARGIN, HD, film, pins, None)
    assert (peaks.dit, peaks.free, peaks.budget) == (peak, peak + MARGIN, peak)
    assert refused(peak + MARGIN - 1, HD, film, pins, None).startswith("--swap-blocks 0: a window")
    # A VAE pass keeps its room on both (planner.VAE_ROOM), which is another need: an untiled
    # decode pinned, the DiT off the GPU.
    tight = Pins(window=5, swap_blocks=36, decode_tile=0)
    decode = planner.decode_peak(padded_size(HD), output_size(HD), None, 0, 36)
    need = max(decode + planner.VAE_ROOM, planner.dit_peak(padded_size(HD), 5, 36, False))
    assert need == decode + planner.VAE_ROOM
    planner.plan(on_card(need), HD, film, tight, None)
    assert refused(on_card(need - 1), HD, film, tight, None).startswith("--decode-tile 0: ")
    planner.plan(need + MARGIN, HD, film, tight, None)
    assert refused(need + MARGIN - 1, HD, film, tight, None).startswith("--decode-tile 0: ")


def test_a_held_run_s_choices_for_the_host_take_no_margin_either() -> None:
    # The two choices made for the host's memory, the VAE's weights kept in the DiT's phase
    # (planner._vae_kept) and the planner's own shorter window (planner._window_that_runs), are
    # made only for a plan the card holds: judged, on a run held to the card, against what the
    # plan counts on whole, no margin on top (planner._Job.margin), as the plan itself is. A
    # held run counting on N bytes plans as a driver that refuses with N and the margin free:
    # the same windows, blocks and tiles, on the same host.
    def on_card(counted: int) -> Reading:
        return Reading(counted + round(5.0 * GIB), None, counted, True, round(0.5 * GIB))

    job = (HD, shots(76), Pins())
    # 1080p on a 24 GB card, a shot of 301 frames, a host one byte short of the plan: 29 blocks
    # swapped and the VAE staying for 28 and the VAE away, the windows of 17 kept
    # (test_the_vae_stays_where_the_host_is_short). The DiT's phase stands under the margin
    # from the budget there: with a margin on top, the held plan would pass for one the card
    # doesn't hold, and its window be shortened instead.
    free = our_free(24)
    pinned = 36 * paged(BLOCK) + paged(planner.DIT_OTHER) + paged(planner.VAE_WEIGHTS)
    first = planner.HOST_RESIDENT + pinned + 169 * 1_059_840
    refusing, peaks = planner.plan(free, *job, first - 1)
    assert (refusing.window, refusing.swapped, refusing.vae_in_dit) == (17, 29, True)
    assert 0 < peaks.budget - peaks.dit < MARGIN
    held, on_it = planner.plan(on_card(free - MARGIN), *job, first - 1)
    assert held == refusing and on_it.budget == on_it.free == peaks.budget
    assert on_it.shortened is None and peaks.shortened is None
    # 80 GiB to count on and 16 GiB of system memory: two windows of 39, the card's own window
    # of 76 pinning 35 blocks (test_the_planner_takes_the_longest_window_the_host_holds).
    refusing, peaks = planner.plan(80 * GIB, *job, 16 * GIB, split=True)
    alone, by_itself = planner.plan(80 * GIB, *job, None, split=True)
    assert (refusing.window, alone.window, peaks.shortened is not None) == (39, 76, True)
    assert 0 < by_itself.budget - by_itself.dit < MARGIN
    held, on_it = planner.plan(on_card(80 * GIB - MARGIN), *job, 16 * GIB, split=True)
    assert held == refusing and on_it.shortened == peaks.shortened
    assert (on_it.free, on_it.budget) == (80 * GIB - MARGIN,) * 2


def test_the_unpinned_case_skips_a_small_card_alone() -> None:
    # tests/test_regression.py's unpinned case, on a card whose plan can't make numz's untiled
    # master (test_regression.unpinned): skipped where the driver reports under 40 GiB free,
    # failed where it reports 40 or more, a plan that tiles or is refused there being wrong;
    # and failed where the log doesn't say, a refusal for the system memory alone. By the
    # planner's own lines and refusals, as a --plan run logs them.
    def logged(lines: list[str], level: str = "INFO") -> str:
        return "".join(f"12:00:00 {level} seedvr2x.cli: {line}\n" for line in lines)

    job = (HD, shots(12), Pins())
    made, peaks = planner.plan(our_free(24), *job, None, split=True, numz_padding=True)
    lines = planner.said(made, peaks, "a card", *job, numz_padding=True)
    assert "in tiles of" in lines[6] and test_regression.reported(logged(lines)) == 22.57
    with pytest.raises(pytest.skip.Exception) as skipped:
        test_regression.unpinned(logged(lines), f"the plan tiles a pass: {lines[6]}")
    assert str(skipped.value).startswith(
        "the driver reports 22.57 GiB free, under the 40 a card makes numz's untiled master"
        " from: the plan tiles a pass: plan: encode untiled, decode in tiles of "
    )
    # 48 GB: 46.57 GiB reported, and both passes untiled, as numz ran them: no verdict to give.
    made, peaks = planner.plan(our_free(48), *job, None, split=True, numz_padding=True)
    assert (made.encode_tiles, made.decode_tiles) == (None, None)
    lines = planner.said(made, peaks, "a card", *job, numz_padding=True)
    assert test_regression.reported(logged(lines)) == 46.57
    # A plan that tiles where 46.57 GiB are reported, a larger job's here: failed.
    larger = (UHD, shots(12), Pins())
    made, peaks = planner.plan(our_free(48), *larger, None)
    lines = planner.said(made, peaks, "a card", *larger)
    with pytest.raises(pytest.fail.Exception) as failed:
        test_regression.unpinned(logged(lines), f"the plan tiles a pass: {lines[6]}")
    assert str(failed.value).startswith(
        "the driver reports 46.57 GiB free, 40 or more, where the plan must run both passes"
        " untiled: the plan tiles a pass: plan: encode "
    )
    # Refused: skipped on a small card, failed on a large one, by the refusal's own figure.
    small = refused(round(6.5 * GIB), *job, None)
    with pytest.raises(pytest.skip.Exception, match=r"the driver reports 6\.50 GiB free, under"):
        test_regression.unpinned(logged([small], "ERROR"), f"the plan is refused: {small}")
    large = refused(our_free(48), (4320, 7680), shots(5), Pins(), None)
    with pytest.raises(pytest.fail.Exception, match=r"the driver reports 46\.57 GiB free, 40 or"):
        test_regression.unpinned(logged([large], "ERROR"), f"the plan is refused: {large}")
    # A refusal for the system memory names no figure of the driver's: failed, saying so.
    host = refused(our_free(48), *job, 2 * GIB)
    assert test_regression.reported(logged([host], "ERROR")) is None
    with pytest.raises(pytest.fail.Exception, match="what the driver reports free isn't in the"):
        test_regression.unpinned(logged([host], "ERROR"), f"the plan is refused: {host}")


# The DiT's windows under SDPA, each the allocated peak of one forward in bytes, the largest of
# the window's runs, in seedvr2x's padding: the padded frame's rows (1104 for 1080p, 2192 for
# 4K), the window's latents, the DiT's blocks swapped, whether the VAE's weights were on the
# GPU. SDPA_BOX: the sweep on the 96 GB card (2026-10-12, 29 windows, seedvr2x's own forward).
# SDPA_DESKTOP: an RTX 5090 behind WSL2's GPU device (2026-10-11 and 12, 16 windows,
# research/scripts/hold_probe.py), the VAE's weights on the GPU in every one.
SDPA_BOX = [
    (1104, 1, 0, True, 18_271_269_986),
    (1104, 1, 36, True, 2_864_739_426),
    (1104, 2, 0, True, 19_494_654_386),
    (1104, 3, 0, True, 20_718_038_786),
    (1104, 6, 0, True, 24_197_301_266),
    (1104, 6, 36, True, 8_790_770_706),
    (1104, 12, 0, True, 31_057_881_026),
    (1104, 12, 12, True, 26_526_549_954),
    (1104, 12, 36, True, 15_651_350_466),
    (1104, 21, 0, True, 41_113_024_826),
    (1104, 21, 36, False, 25_205_197_812),
    (1104, 30, 0, True, 51_355_472_786),
    (1104, 40, 0, True, 62_726_178_866),
    (1104, 50, 0, True, 74_092_408_466),
    (1104, 60, 0, True, 85_460_847_026),
    (1104, 70, 0, True, 96_922_289_186),
    (2192, 1, 0, True, 21_844_586_906),
    (2192, 2, 0, True, 26_641_290_746),
    (2192, 3, 0, True, 31_437_994_586),
    (2192, 4, 0, True, 36_234_698_426),
    (2192, 5, 36, False, 24_483_353_532),
    (2192, 6, 0, True, 45_188_097_506),
    (2192, 8, 0, True, 54_460_689_986),
    (2192, 8, 36, True, 39_054_159_426),
    (2192, 10, 0, True, 63_092_889_746),
    (2192, 12, 0, True, 72_365_746_466),
    (2192, 14, 0, True, 80_999_345_906),
    (2192, 16, 0, True, 90_272_364_626),
    (2192, 17, 0, True, 94_428_462_626),
]
SDPA_DESKTOP = [
    (1104, 1, 0, True, 18_297_990_656),
    (1104, 1, 36, True, 2_888_093_184),
    (1104, 6, 0, True, 24_228_963_840),
    (1104, 6, 36, True, 8_814_077_440),
    (1104, 8, 0, True, 26_575_523_840),
    (1104, 8, 36, True, 11_167_731_712),
    (1104, 9, 0, True, 27_605_780_992),
    (1104, 9, 36, True, 12_195_855_360),
    (1104, 12, 36, True, 15_678_309_888),
    (1104, 17, 36, True, 21_225_550_336),
    (1104, 21, 36, True, 25_733_936_128),
    (2192, 1, 36, True, 6_460_755_456),
    (2192, 2, 36, True, 11_256_947_712),
    (2192, 3, 36, True, 16_054_198_272),
    (2192, 4, 36, True, 20_857_285_632),
    (2192, 5, 36, True, 25_011_469_312),
]


def test_the_dit_s_fit_is_its_attention_backend_s() -> None:
    # DESIGN.md, Memory planner, Per-phase peaks: "The fit is FlashAttention 2's ... Under SDPA,
    # which a machine without a FlashAttention build runs, the DiT peaks higher ... The planner
    # holds a fit per attention backend and takes the run's." SDPA's: 4.69% more a token and
    # 0.141 GiB, an upper bound of every window measured under it on two cards, 29 on the 96 GB
    # card and 16 on a desktop card, each counted with its own swapped blocks and VAE: by 0.017
    # to 0.131 GiB on the first, by 0.0009 to 0.105 on the second, whose 4 latents of 4K set the
    # constant: the first card's 29 alone are bounded by 0.124 GiB, which that window passes.
    flash, sdpa = planner.FLASH_ATTN_2, planner.SDPA
    assert planner.DIT_ATTENTION == {flash: (0, 1.0), sdpa: (round(0.141 * GIB), 1.0469)}
    padded = {1104: OURS_1080, 2192: OURS_4K}
    assert (len(SDPA_BOX), len(SDPA_DESKTOP)) == (29, 16)
    assert {(rows, max(row[1] for row in SDPA_BOX if row[0] == rows)) for rows in padded} == {
        (1104, 70),
        (2192, 17),
    }
    left: dict[str, list[tuple[int, int, int]]] = {"box": [], "desktop": []}
    over: list[float] = []
    for card, measured in (("box", SDPA_BOX), ("desktop", SDPA_DESKTOP)):
        for rows, latents, swapped, vae, peak in measured:
            window = (padded[rows], latents, swapped, vae)
            fitted = planner.dit_peak(*window)
            assert fitted == planner.dit_peak(*window, attention=flash)
            bound = planner.dit_peak(*window, attention=sdpa)
            tokens = planner.tokens(padded[rows], latents)
            more = 0.141 * GIB + 0.0469 * planner.PER_TOKEN * tokens
            assert bound - fitted == pytest.approx(more, abs=2)  # two peaks, each rounded
            assert bound >= peak, (card, rows, latents, swapped)
            left[card].append((bound - peak, rows, latents))
            over.append(gib(peak - fitted))
    assert (round(min(over), 2), round(max(over), 2)) == (0.06, 3.31)
    least, most = min(left["box"]), max(left["box"])
    assert (round(gib(least[0]), 4), *least[1:]) == (0.0173, 2192, 8)
    assert (round(gib(most[0]), 4), *most[1:]) == (0.1314, 1104, 60)
    least, most = min(left["desktop"]), max(left["desktop"])
    assert (round(gib(least[0]), 4), *least[1:]) == (0.0009, 2192, 4)
    assert (round(gib(most[0]), 4), *most[1:]) == (0.1049, 1104, 1)
    # The 96 GB card's own bound, 0.124 GiB, is under that desktop window by 0.016.
    short = least[0] - round(0.141 * GIB) + round(0.124 * GIB)
    assert round(gib(short), 3) == -0.016
    # FlashAttention 2's fit is as it was: 12 latents at 1080p, 28.320 GiB.
    assert planner.dit_peak(OURS_1080, 12) == 30_407_967_638
    # The VAE's weights away, as under FlashAttention 2.
    apart = planner.dit_peak(OURS_1080, 6, 0, False, sdpa)
    assert planner.dit_peak(OURS_1080, 6, 0, True, sdpa) - apart == planner.VAE_WEIGHTS
    with pytest.raises(ValueError, match="attention 'xformers': the DiT's fit is flash_attn_2's"):
        planner.dit_peak(OURS_1080, 6, attention="xformers")
    with pytest.raises(ValueError, match="attention 'xformers': the DiT's fit is flash_attn_2's"):
        planner.plan(our_free(24), HD, shots(6), Pins(), None, attention="xformers")

    # The plan is made for the backend the run takes. 1080p on a 24 GB card, a shot of 301
    # frames: the same windows of 17 latents, 30 blocks swapped under SDPA for 28; the plan's
    # second line and its record say which.
    free, job = our_free(24), (HD, shots(76), Pins(), None)
    made, peaks = planner.plan(free, *job)
    assert bare(made) == Plan(17, Tiles(1856), Tiles(1088), 28, 1, 0, vae_in_dit=False)
    other, slower = planner.plan(free, *job, attention=sdpa)
    assert bare(other) == Plan(17, Tiles(1856), Tiles(1088), 30, 1, 0, vae_in_dit=False)
    assert slower.dit == planner.dit_peak(OURS_1080, 17, 30, False, sdpa)
    assert (peaks.record()["attention"], slower.record()["attention"]) == (flash, sdpa)
    assert planner.said(other, slower, "a card", HD, shots(76), Pins())[1] == (
        "plan: frames padded to 1920x1104, 8,280 tokens a latent, attention on PyTorch's SDPA"
    )
    # A window shorter where the card was full: a shot of 1,197 frames in windows of 19 latents
    # for 20, every block swapped.
    made, _ = planner.plan(free, HD, shots(300), Pins(), None)
    other, _ = planner.plan(free, HD, shots(300), Pins(), None, attention=sdpa)
    assert (made.window, made.swapped, other.window, other.swapped) == (20, 35, 19, 36)
    # 4K's smallest window, 5 latents, takes 23.47 GiB of free VRAM by SDPA's fit (22.80 GiB
    # measured on the 96 GB card, SDPA_BOX, and the margin: 23.40) where FlashAttention 2's
    # takes 22.40: a card emulated at 24 GB, 22.51 GiB free, plans 4K with FlashAttention 2 and
    # is refused it without. No window goes under 5 latents to make the room (DESIGN.md).
    emulated = 24_168_824_832
    assert planner.plan(emulated, UHD, shots(5), Pins(), None)[0].window == 5
    assert refused(emulated, UHD, shots(5), Pins(), None, attention=sdpa).startswith(
        "an output of 3840x2160 takes 23.47 GiB of free VRAM (its smallest window, 5 latents, with"
        " every block of the DiT swapped and the VAE off the GPU: a peak of 22.87 GiB, and 0.60 of"
        " margin), and the driver reports 22.50 GiB free: the largest output that fits there is"
        " --resolution 2096, "
    )


def test_resumed_on_another_attention_backend() -> None:
    # Where the weights are is planned again on every resume, from the memory free that day
    # (DESIGN.md, Memory planner, A plan shapes the output a little), and for the attention
    # backend of that day's run: FlashAttention installed since, or gone, the recorded shape is
    # held to today's fit. A shape recorded with FlashAttention 2, taken up on SDPA: the same
    # windows and tiles, two more blocks swapped; and the other way round, two fewer.
    free, job = our_free(24), (HD, shots(76), Pins(), None)
    shape = Plan(17, Tiles(1856), Tiles(1088))
    with_flash = Plan(17, Tiles(1856), Tiles(1088), 28, 1, 0, vae_in_dit=False)
    assert bare(planner.plan(free, *job, recorded=shape)[0]) == with_flash
    resumed, peaks = planner.plan(free, *job, recorded=with_flash, attention=planner.SDPA)
    assert bare(resumed) == replace(with_flash, swapped=30)
    assert peaks.attention == planner.SDPA
    back, peaks = planner.plan(free, *job, recorded=resumed, attention=planner.FLASH_ATTN_2)
    assert (bare(back), peaks.attention) == (with_flash, planner.FLASH_ATTN_2)
    # A shape whose windows only FlashAttention 2's fit holds on that card no longer fits there
    # without it, and is refused as any recorded shape that no longer fits: its windows of 20
    # latents take 23.23 GiB of free VRAM by SDPA's fit, every block swapped.
    long = (HD, shots(300), Pins(), None)
    recorded, _ = planner.plan(free, *long)
    assert bare(recorded) == Plan(20, Tiles(1856), Tiles(1088), 35, 1, 0, vae_in_dit=False)
    assert fits(recorded, free, HD, shots(300), None) is None
    assert fits(recorded, free, HD, shots(300), None, attention=planner.SDPA) == (
        "the DiT's phase, a window of 20 latents with 35 blocks swapped and the VAE off the GPU,"
        " takes 23.65 GiB of free VRAM and the driver reports 22.57 GiB free, 1.08 GiB short"
    )
    assert refused(free, *long, recorded=recorded, attention=planner.SDPA) == (
        "the recorded plan no longer fits the memory free now: the DiT's phase, a window of 20"
        " latents with 36 blocks swapped and the VAE off the GPU, takes 23.23 GiB of free VRAM"
        " and the driver reports 22.57 GiB free, 0.66 GiB short"
    )

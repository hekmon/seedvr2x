"""The bricks' peaks (peaks.py; DESIGN.md, Memory planner, Per-phase peaks): the DiT's peak
against the measurements it was fitted on, every expected figure copied from
research/docs/planner-limits.md, research/docs/vram.md or DESIGN.md, the table named, or measured
on seedvr2x with milestone 3 (2026-10-11), said so; where the weights are and what they take
against seedvr2x/blockswap.py's own rules; the tiles' measured peaks, and what a tile nobody
measured takes. On the CPU, no weights: the models are built on the meta device where their
shapes are counted."""

import pytest

from seedvr2x import peaks, tilegrid
from seedvr2x.frames import output_size, padded_size
from seedvr2x.passes import Tiles
from seedvr2x.peaks import GIB

# A 1080p target (tests/test_planner.py has it with the 4K one).
HD = (1080, 1920)

# numz's padded sizes, multiples of 16 (planner-limits.md's tables), and seedvr2x's own
# (frames.padded_size; DESIGN.md, Memory planner: 1,104 rows at 1080p, 2,192 at 4K).
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


@pytest.mark.parametrize(("padded", "latents", "tokens", "windows", "peak"), DIT_POINTS)
def test_dit_fit(
    padded: tuple[int, int], latents: int, tokens: int, windows: int, peak: float
) -> None:
    # The table's tokens and windows exactly. Its peaks are rounded to 0.01 GiB and the fit holds
    # them to 0.005 (planner-limits.md: "fit all 26 passing points to 0.005 GiB"), so the model
    # is compared with the column as printed within the two together, 0.01 GiB; the furthest is
    # 0.0051 (1080p, 78 latents: 93.9349 against 93.94).
    assert peaks.tokens(padded, latents) == tokens
    assert peaks.attention_windows(padded, latents) == windows
    assert abs(gib(peaks.dit_peak(padded, latents)) - peak) <= 0.01


def test_dit_fit_furthest() -> None:
    furthest = max(
        abs(gib(peaks.dit_peak(padded, latents)) - peak)
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
    assert peaks.attention_layouts(padded, latents) == layouts
    assert peaks.attention_windows(padded, latents) == max(layouts)


def test_dit_swapped_and_vae_away() -> None:
    # A swapped block takes its 0.42 GiB off the peak, but for the two in the slots, and the VAE
    # sent away its 0.47 (DESIGN.md, Memory planner, Per-phase peaks; numz: 28.15 to 27.68 GiB).
    whole = peaks.dit_peak(NUMZ_1080, 12)
    assert peaks.dit_peak(NUMZ_1080, 12, 1) == peaks.dit_peak(NUMZ_1080, 12, 2) == whole
    assert whole - peaks.dit_peak(NUMZ_1080, 12, 3) == peaks.BLOCK
    assert whole - peaks.dit_peak(NUMZ_1080, 12, 36) == 34 * peaks.BLOCK
    away = peaks.dit_peak(NUMZ_1080, 12, vae_in_dit=False)
    assert whole - away == peaks.VAE_WEIGHTS
    assert abs(gib(whole) - 28.15) <= 0.01 and abs(gib(away) - 27.68) <= 0.01
    for swapped in (-1, 37):
        with pytest.raises(ValueError, match="blocks swapped of the DiT's 36"):
            peaks.dit_peak(NUMZ_1080, 12, swapped)
    # Measured on seedvr2x (milestone 3, 2026-10-11), one window of 1 latent at 1080p: 17.016
    # GiB with no block swapped, 16.595 with 3, 12.796 with 12 and 2.668 with 36. Each swapped
    # block beyond the two slots takes its bytes off, to 0.001 GiB, as the model has it; the
    # peak itself is 0.061 GiB over numz's fit, whose constants the model keeps, at every count.
    measured = {0: 17.016, 3: 16.595, 12: 12.796, 36: 2.668}
    for swapped, peak in measured.items():
        saved = max(swapped - peaks.SLOTS, 0) * peaks.BLOCK
        assert abs(measured[0] - peak - gib(saved)) <= 0.0015  # two figures rounded to 0.001
        assert peaks.dit_peak(OURS_1080, 1) - peaks.dit_peak(OURS_1080, 1, swapped) == saved
        assert 0.0605 <= peak - gib(peaks.dit_peak(OURS_1080, 1, swapped)) <= 0.0625


def test_dit_on_gpu() -> None:
    # What of the DiT a VAE phase finds on the GPU, as blockswap.Placement.enter leaves it there
    # (milestone 3): its buffers and tables always; with None, nothing moving, its other
    # parameters, the blocks that aren't swapped and the slots; with 0, nothing more; with a
    # number, that many blocks and the other parameters, the slots freed.
    block, other, never = peaks.BLOCK, peaks.DIT_OTHER, peaks.DIT_BUFFERS
    assert peaks.dit_on_gpu(None, 0) == never + other + 36 * block
    # All of the loaded models but the VAE's weights, as the DiT's fit has them: 0.0013 GiB over.
    loaded = peaks.DIT_LOADED - peaks.VAE_WEIGHTS
    assert gib(peaks.dit_on_gpu(None, 0) - loaded) == pytest.approx(0.0013, abs=5e-5)
    assert peaks.dit_on_gpu(0, 0) == never
    assert peaks.dit_on_gpu(12, 0) == never + other + 12 * block
    assert peaks.dit_on_gpu(36, 0) == peaks.dit_on_gpu(None, 0)
    # One and two swapped blocks sit in their slots while nothing moves: nothing saved. A phase
    # entered with a number frees the slots, and with 0 the other parameters too, whatever is
    # swapped.
    assert peaks.dit_on_gpu(None, 1) == peaks.dit_on_gpu(None, 2) == peaks.dit_on_gpu(None, 0)
    assert peaks.dit_on_gpu(0, 1) == peaks.dit_on_gpu(0, 2) == peaks.dit_on_gpu(0, 36) == never
    assert peaks.dit_on_gpu(10, 2) == never + other + 10 * block
    assert peaks.dit_on_gpu(34, 2) == peaks.dit_on_gpu(None, 2) - 2 * block
    assert peaks.dit_on_gpu(None, 36) == never + other + 2 * block
    assert peaks.dit_on_gpu(None, 20) == never + other + 18 * block
    assert peaks.dit_on_gpu(16, 20) == never + other + 16 * block
    assert peaks.dit_on_gpu(4, 20) == never + other + 4 * block
    for resident, swapped in ((1, 36), (17, 20), (-1, 0), (0, 37), (None, 37)):
        with pytest.raises(ValueError, match="of the DiT's 36"):
            peaks.dit_on_gpu(resident, swapped)


def test_dit_on_gpu_is_the_placements() -> None:
    # blockswap.Placement's own count of the DiT's parameters on the device in a phase
    # (device_bytes), on a model of 36 blocks of 512 bytes and 1,024 of other parameters, its
    # host copies plain memory: dit_on_gpu's three cases, block for block, for every count of
    # swapped blocks and every phase's residency the loader accepts.
    import torch

    from seedvr2x import blockswap

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
            self.blocks = torch.nn.ModuleList(Block() for _ in range(peaks.BLOCKS))

    block, other = 512, 1024
    device = torch.device("cpu")
    for swapped in (0, 1, 2, 3, 20, 35, 36):
        fixed = peaks.BLOCKS - swapped
        model = Model().requires_grad_(False)
        parts = blockswap.host(model, blockswap.dit_parts(peaks.BLOCKS, 0, True), Host(device))
        placement = blockswap.Placement(model, device, swapped, parts=parts, backend=Host(device))
        for resident in (None, *sorted({0, 1, fixed // 2, fixed} & set(range(fixed + 1)))):
            ours = peaks.dit_on_gpu(resident, swapped) - peaks.DIT_BUFFERS
            blocks, others = divmod(ours, peaks.BLOCK)
            assert others in (0, peaks.DIT_OTHER)
            theirs = placement.device_bytes(resident)
            assert theirs == blocks * block + (other if others else 0), (swapped, resident)
        with pytest.raises(ValueError, match="blocks to keep"):
            placement.device_bytes(fixed + 1)


def test_the_int8_dit_s_figures() -> None:
    # The DiT in seedvr2x's int8 files: its weights by its headers (weights.DIT_7B_INT8), 0.2113
    # GiB a block where fp16's take 0.4220, its other parameters fp16's; loaded, the fp16 fit's
    # constant with its weights' bytes exchanged for these, 8.285 GiB for 15.871.
    int8, fp16 = peaks.INT8, peaks.FP16
    assert peaks.block_bytes(fp16) == peaks.BLOCK == 453_133_312
    assert peaks.block_bytes(int8) == 226_862_080
    assert f"{gib(peaks.block_bytes(int8)):.4f}" == "0.2113"
    assert peaks.other_bytes(int8) == peaks.other_bytes(fp16) == peaks.DIT_OTHER
    assert peaks.loaded(fp16) == peaks.DIT_LOADED
    saved = 36 * (453_133_312 - 226_862_080)
    assert peaks.loaded(int8) == peaks.DIT_LOADED - saved
    assert f"{gib(peaks.loaded(int8)):.3f}" == "8.285" and f"{gib(saved):.3f}" == "7.586"
    # Its window's peak: fp16's with those weights, and the W8A8 path's buffers (provisional,
    # peaks.INT8_*): the same tokens and windows, on either attention backend.
    for padded, latents in ((OURS_1080, 1), (OURS_1080, 20), (OURS_4K, 7)):
        for attention in (peaks.FLASH_ATTN_2, peaks.SDPA):
            plain = peaks.dit_peak(padded, latents, attention=attention)
            quantised = peaks.dit_peak(padded, latents, attention=attention, dit_model=int8)
            assert quantised == plain - saved + peaks.INT8_BUFFERS
    # A swapped block takes its own bytes off, but for the two in the slots; the VAE away its.
    whole = peaks.dit_peak(OURS_1080, 12, dit_model=int8)
    assert peaks.dit_peak(OURS_1080, 12, 2, dit_model=int8) == whole
    assert whole - peaks.dit_peak(OURS_1080, 12, 36, dit_model=int8) == 34 * 226_862_080
    assert whole - peaks.dit_peak(OURS_1080, 12, vae_in_dit=False, dit_model=int8) == (
        peaks.VAE_WEIGHTS
    )
    least = peaks.dit_least(OURS_1080, 12, 5, True, peaks.FLASH_ATTN_2, int8)
    assert least <= peaks.dit_peak(OURS_1080, 12, 5, dit_model=int8)
    # What of it a VAE phase finds on the GPU, by its blocks' bytes; and the passes over it.
    never, other, block = peaks.DIT_BUFFERS, peaks.DIT_OTHER, 226_862_080
    assert peaks.dit_on_gpu(None, 0, int8) == never + other + 36 * block
    assert peaks.dit_on_gpu(None, 20, int8) == never + other + 18 * block
    assert peaks.dit_on_gpu(10, 2, int8) == never + other + 10 * block
    assert peaks.dit_on_gpu(0, 0, int8) == never
    for resident in (None, 0, 12):
        apart = peaks.dit_on_gpu(resident, 0) - peaks.dit_on_gpu(resident, 0, int8)
        tiles = Tiles(1024)
        assert peaks.encode_peak(OURS_1080, tiles, resident) - apart == peaks.encode_peak(
            OURS_1080, tiles, resident, dit_model=int8
        )
        assert peaks.decode_peak(OURS_1080, HD, tiles, resident) - apart == peaks.decode_peak(
            OURS_1080, HD, tiles, resident, dit_model=int8
        )
    # Another architecture has no figures here.
    from seedvr2x import weights

    for call in (
        lambda: peaks.dit_peak(OURS_1080, 1, dit_model=weights.DIT_3B),
        lambda: peaks.dit_on_gpu(None, 0, weights.VAE),
    ):
        with pytest.raises(ValueError, match="the DiT's figures are for"):
            call()


def largest(padded: tuple[int, int], tiles: Tiles) -> tuple[int, int]:
    """(rows, columns), in pixels, of the largest tile of a tiled pass over padded frames: the
    first of the shapes the planner counts a pass by."""
    return peaks.tile_shapes(padded, tiles)[0]  # pyright: ignore[reportPrivateUsage]


def test_grid_and_largest_tile() -> None:
    # The grid the plan's lines say and the tile a pass's peak follows are the tiled passes' own
    # (seedvr2x/tilegrid.py, which tests/test_tilegrid.py and tests/test_tiling.py hold to
    # numz's). planner-limits.md, "Tiled VAE at 4K", Tiles (rows x columns), overlap 128: 1024 px
    # 3 x 5, 1536 px 2 x 3, 2048 px 2 x 2; vram.md, Tiling: 60 tiles of 512 px and 24 of 768 at
    # overlap 128, 12 of 1024 at overlap 64, 15 at 128 and at 256.
    grid = peaks.grid_shapes  # pyright: ignore[reportPrivateUsage]
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
    assert not peaks.covers(OURS_1080, Tiles(1856)) and peaks.covers(OURS_1080, Tiles(1920))


def vae_fits(padded: tuple[int, int], tiles: Tiles | None) -> tuple[float, float]:
    """The encode's and the decode's fits, GiB above what is on the GPU before the call: the
    peaks less the VAE's weights and what of the DiT never moves, the DiT away, as the tables
    give them, measured on numz: the encode's without what seedvr2x's holds over its fit
    (peaks.ENCODE_EXTRA)."""
    before = peaks.VAE_WEIGHTS + peaks.dit_on_gpu(0, 0)
    encode = peaks.encode_peak(padded, tiles, resident=0) - before - peaks.ENCODE_EXTRA
    decode = peaks.decode_peak(padded, output_size(padded), tiles, resident=0) - before
    return gib(encode), gib(decode)


def tile_fits(size: int, area: float) -> tuple[float, float]:
    """The tiled fits of the encode and of the decode for a tile of `size` px a side and `area`
    megapixels, in GiB: the refit from 1024 px, vram.md's below (DESIGN.md, Memory planner,
    Per-phase peaks)."""
    if size >= 1024:
        return 1.36 + 8.66 * area, 0.44 + 16.29 * area
    return 1.7 + 8.4 * area, 1.6 + 15.6 * area


# The 26 entries of peaks.MEASURED that the one-output convolution (vendor/models/video_vae_v3/
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
    measured = peaks.MEASURED
    squares = {
        rows: predicted for (rows, columns), predicted in measured.items() if rows == columns
    }
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
    old_squares = {key[0]: predicted for key, predicted in old.items() if key[0] == key[1]}
    assert len(BEFORE_ONE_OUTPUT) == 26 and all(key in measured for key in BEFORE_ONE_OUTPUT)
    assert {key for key in measured if measured[key] != old[key]} == set(BEFORE_ONE_OUTPUT)
    higher = {
        key for key, predicted in BEFORE_ONE_OUTPUT.items() if measured[key][1] > predicted[1]
    }
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
    assert {
        key for key, predicted in BEFORE_ONE_OUTPUT.items() if measured[key][0] != predicted[0]
    } == {(size, size) for size in range(1664, 2049, 64)}
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
    for size, predicted in squares.items():
        for overlap in (64, 128):
            whole = (2 * size - overlap, 3 * size - 2 * overlap)
            assert tile_shapes(whole, Tiles(size, overlap)) == {(size, size)}
            assert vae_fits(whole, Tiles(size, overlap)) == pytest.approx(predicted, abs=1e-8)
        for padded in (NUMZ_4K, OURS_4K, (2048, 2064)):
            encode, decode = vae_fits(padded, Tiles(size))
            assert encode >= predicted[0] - 1e-8 and decode >= predicted[1] - 1e-8
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
    for size, predicted in old_squares.items():
        over[size] = (predicted[0] - over[size][0], predicted[1] - over[size][1])
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
    for size, predicted in numz.items():
        assert max(abs(a - b) for a, b in zip(predicted, old_squares[size], strict=True)) < 0.26
    rounded = {512: (3.5, 5.9), 768: (7.1, 11.1), 1024: (10.4, 17.4), 1536: (21.4, 38.7)}
    for size, predicted in rounded.items():
        assert max(abs(a - b) for a, b in zip(predicted, old_squares[size], strict=True)) < 0.11


def test_vae_tiles_nobody_measured() -> None:
    # DESIGN.md, Memory planner, Per-phase peaks: "A cut-off tile of another size takes the
    # larger of its fit and of the squares' line at its area, and 0.3 GiB more, the most a
    # measured one stood above that being 0.27."
    rule = peaks._rule  # pyright: ignore[reportPrivateUsage]
    assert peaks.CUT_OFF_EXTRA == round(0.3 * GIB)

    def between(pixels: int) -> tuple[float, float]:
        """The line between the measured squares around an area: encode, decode."""
        sides = [rows for rows, columns in peaks.MEASURED if rows == columns]
        below = max(side for side in sides if side**2 <= pixels)
        above = min(side for side in sides if side**2 > pixels)
        share = (pixels - below**2) / (above**2 - below**2)
        low, high = peaks.MEASURED[below, below], peaks.MEASURED[above, above]
        return low[0] + share * (high[0] - low[0]), low[1] + share * (high[1] - low[1])

    # The 0.27, by every measured cut-off tile against the rule it would take unmeasured: the
    # 20 tiles' encodes and decodes. The decode of 752x896, 12.38 GiB measured for a fit of 12.11
    # and a line of 12.09, stands the furthest above it; the encodes from 0.68 under (752x960)
    # to 0.21 over (1104x1536).
    above: dict[tuple[int, int, int], float] = {}
    for (rows, columns), predicted in peaks.MEASURED.items():
        if rows == columns:
            continue
        fits, lines = tile_fits(columns, rows * columns / 1e6), between(rows * columns)
        for column in (0, 1):
            ruled = rule(columns, rows * columns, column)
            assert ruled == pytest.approx(max(fits[column], lines[column]), abs=1e-9)
            above[rows, columns, column] = predicted[column] - ruled
    assert len(above) == 40
    worst = max(above, key=lambda key: above[key])
    assert worst == (752, 896, 1) and round(above[worst], 2) == 0.27
    assert above[worst] * GIB < peaks.CUT_OFF_EXTRA
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
        if size <= peaks.TILE_MAX:
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
    rule = peaks._rule  # pyright: ignore[reportPrivateUsage]
    taken: list[tuple[float, float]] = []
    for rows, columns in tile_shapes(padded, tiles):
        if (rows, columns) in peaks.MEASURED:
            taken.append(peaks.MEASURED[rows, columns])
            continue
        cut = (rows, columns) != (tiles.size, tiles.size) and tiles.size <= peaks.TILE_MAX
        extra = 0.3 if cut else 0.0
        ruled = [rule(tiles.size, rows * columns, column) + extra for column in (0, 1)]
        taken.append((ruled[0], ruled[1]))
    return max(encode for encode, _ in taken), max(decode for _, decode in taken)


def test_vae_untiled() -> None:
    # DESIGN.md, Memory planner, Per-phase peaks: encode 1.2 + 8.8 P, decode 0.8 + 16.1 P GiB,
    # P the padded megapixels. With the VAE's weights, vram.md's measured peaks (Recipe per card
    # size, 48 GB at 1080p, untiled: 21.44 / 35.72 in numz's padding, its batch's frames on the
    # GPU for the encode) and DESIGN.md's untiled 4K decode, "about 134 GiB".
    assert vae_fits(NUMZ_1080, None) == pytest.approx(
        (1.2 + 8.8 * 2.08896, 0.8 + 16.1 * 2.08896), abs=1e-6
    )
    encode, decode = (fit + gib(peaks.VAE_WEIGHTS) for fit in vae_fits(NUMZ_1080, None))
    assert abs(decode - 35.72) < 0.9 and abs(encode - 21.44) < 1.5
    assert round(vae_fits(NUMZ_4K, None)[1]) == 134
    # The largest tile the planner makes at 1080p, 1856 px, takes less; one that covers the
    # whole padded frame is no tiling (DESIGN.md, Pipeline, per shot, step 3), and has no peak.
    assert vae_fits(OURS_1080, Tiles(1856)) < vae_fits(OURS_1080, None)
    for size in (1920, 4096):
        with pytest.raises(ValueError, match=f"tiles of {size} px over frames padded to 1920x1104"):
            peaks.encode_peak(OURS_1080, Tiles(size))
        with pytest.raises(ValueError, match="one covers the whole frame, which is no tiling"):
            peaks.decode_peak(OURS_1080, HD, Tiles(size))


def test_vae_untiled_small_frames() -> None:
    # The untiled fits come from frames of 720p and up; under 1.25 megapixels for the encode and
    # 1.6 for the decode, an untiled pass takes the small tiles' fit at its frame's area, the
    # larger of the two there (peaks._fit; PROVISIONAL). 1080p and up keep the untiled fits.
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
    there = peaks.dit_on_gpu(None, 0) + peaks.VAE_WEIGHTS
    assert padded_size((540, 960)) == (576, 960)
    measured = {((540, 960), (576, 960)): (22.53, 26.29), (HD, OURS_1080): (35.36, 50.98)}
    over: dict[tuple[int, int], tuple[float, float]] = {}
    for (target, padded), (encode, decode) in measured.items():
        fitted = round(vae_fits(padded, None)[0] * GIB)
        assert peaks.encode_peak(padded) - there == fitted + peaks.ENCODE_EXTRA
        predicted = peaks.decode_peak(padded, target, split=True)
        assert -0.05 <= gib(predicted) - decode <= 0.6
        over[target] = (gib(peaks.encode_peak(padded)) - encode, gib(predicted) - decode)
    assert over[HD] == pytest.approx((0.766, 0.519), abs=1e-3)
    # The 540p encode's fit is 0.31 GiB under its measured peak, with the small tiles' fit too
    # (0.59 under by the untiled fit): one of the three excesses ENCODE_EXTRA, 0.4 GiB, is the
    # largest of (test_first_matrix), which leaves its prediction 0.09 over. Probably the
    # slice's source frames on the GPU, 5 frames of 1920x1080 in float32, and the resize's
    # temporaries, which the fits, measured on numz, never held.
    assert over[540, 960] == pytest.approx((0.087, -0.016), abs=1e-3)


def test_split_term() -> None:
    # research/docs/colour.md, Cost and what streams: split takes 0.56 GiB per 4K frame on the
    # GPU; it corrects a slice of the decode at once, 5 frames at most: 2.8 GiB at 4K, a quarter
    # of it at 1080p. The untiled decode, streamed, holds it over its own peak.
    plain = peaks.decode_peak(OURS_4K, (2160, 3840), None, 0)
    split = peaks.decode_peak(OURS_4K, (2160, 3840), None, 0, split=True)
    assert gib(split - plain) == pytest.approx(5 * 0.56, abs=1e-6)
    plain = peaks.decode_peak(OURS_1080, (1080, 1920), resident=0)
    split = peaks.decode_peak(OURS_1080, (1080, 1920), resident=0, split=True)
    assert gib(split - plain) == pytest.approx(5 * 0.56 / 4, abs=1e-6)
    # A tiled decode corrects its frames after its last tile, the tile's working memory gone:
    # its peak is the tile's, which the correction's is under at every size seedvr2x makes
    # (test_first_matrix: the tiled decodes measured under the two summed by that term).
    for padded, output in ((OURS_4K, (2160, 3840)), (OURS_1080, (1080, 1920))):
        for size in (512, 1024, 1856):
            plain = peaks.decode_peak(padded, output, Tiles(size), 0)
            assert peaks.decode_peak(padded, output, Tiles(size), 0, split=True) == plain
    # The larger of the two all the same: 8K's slice, 11.2 GiB, over a 512-px tile's 5.89.
    before = peaks.VAE_WEIGHTS + peaks.dit_on_gpu(0, 0)
    plain = peaks.decode_peak((4336, 7680), (4320, 7680), Tiles(512), 0)
    split = peaks.decode_peak((4336, 7680), (4320, 7680), Tiles(512), 0, split=True)
    assert round(gib(plain - before), 2) == 5.89
    assert gib(split - before) == pytest.approx(4 * 5 * 0.56, abs=1e-6)


def test_scratch() -> None:
    # DESIGN.md, Pipeline, per shot, step 3: 12.7 MB per 1080p frame and 50.5 MB per 4K frame,
    # the padding included; a shot's frames padded to 4n + 1. The store either tiled pass opens
    # (tilegrid.scratch_bytes, which tests/test_tilegrid.py holds to tiling.Scratch.needed).
    assert round(peaks.scratch(OURS_1080, 1) / 1e6, 1) == 12.7
    assert round(peaks.scratch(OURS_4K, 1) / 1e6, 1) == 50.5
    assert peaks.scratch(OURS_1080, 301) == 301 * 6 * 1104 * 1920
    assert (
        peaks.scratch(OURS_1080, 300)
        == peaks.scratch(OURS_1080, 298)
        == (peaks.scratch(OURS_1080, 301))
    )
    assert peaks.scratch(OURS_4K, 2) == 5 * peaks.scratch(OURS_4K, 1)

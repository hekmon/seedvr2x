"""The memory planner's model (runtime/planner.py; DESIGN.md, Memory planner): each phase's
predicted peak against the measurements it was fitted on, every expected figure copied from
research/docs/planner-limits.md, research/docs/vram.md or DESIGN.md, the table named. On the CPU,
no model."""

import pytest

from seedvr2x.runtime import planner
from seedvr2x.runtime.job import output_size, padded_size
from seedvr2x.runtime.plan import Tiles
from seedvr2x.runtime.planner import GIB

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
    # Per-phase peaks) and BlockSwap, rewritten.
    assert gib(planner.MARGIN) == pytest.approx(0.6)
    assert gib(planner.DIT_LOADED) == pytest.approx(15.871)
    assert planner.PER_TOKEN == 127.16 * 1024 and planner.PER_WINDOW == 2.726 * 1024**2
    assert gib(planner.VAE_WEIGHTS) == pytest.approx(0.47)
    assert (planner.BLOCKS, planner.SLOTS) == (36, 2)
    assert gib(planner.BLOCK * 36) == pytest.approx(15.19)
    assert gib(planner.BLOCK) == pytest.approx(0.4219, abs=5e-5)  # a block's "0.42 GiB"
    assert gib(planner.DIT_REST) == pytest.approx(0.211, abs=1e-6)
    assert gib(planner.HOST_RESIDENT) == pytest.approx(2.4)
    assert (planner.TILE_MIN, planner.TILE_STEP, planner.SLICE) == (512, 64, 5)


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
    assert whole - planner.dit_peak(NUMZ_1080, 12, 3) == pytest.approx(planner.BLOCK, abs=1)
    assert whole - planner.dit_peak(NUMZ_1080, 12, 36) == pytest.approx(34 * planner.BLOCK, abs=1)
    away = planner.dit_peak(NUMZ_1080, 12, vae_in_dit=False)
    assert whole - away == planner.VAE_WEIGHTS
    assert abs(gib(whole) - 28.15) <= 0.01 and abs(gib(away) - 27.68) <= 0.01
    for swapped in (-1, 37):
        with pytest.raises(ValueError, match="blocks swapped of the DiT's 36"):
            planner.dit_peak(NUMZ_1080, 12, swapped)


def test_dit_on_gpu() -> None:
    # What of the DiT a VAE phase finds on the GPU: all of it but the VAE's weights when nothing
    # leaves (None), the rest of it alone with every block away (0), and the blocks left there;
    # a swapped block's slot stays, two at most.
    block, rest = planner.BLOCK, planner.DIT_REST
    assert planner.dit_on_gpu(None, 0) == planner.DIT_LOADED - planner.VAE_WEIGHTS
    assert planner.dit_on_gpu(0, 0) == rest
    assert planner.dit_on_gpu(12, 0) == round(rest + 12 * block)
    assert planner.dit_on_gpu(36, 0) == planner.dit_on_gpu(None, 0)
    # One and two swapped blocks sit in their slots: nothing saved while the others stay.
    assert planner.dit_on_gpu(None, 1) == planner.dit_on_gpu(None, 2) == planner.dit_on_gpu(None, 0)
    assert planner.dit_on_gpu(0, 1) == round(rest + block)
    assert planner.dit_on_gpu(0, 2) == round(rest + 2 * block)
    assert planner.dit_on_gpu(10, 2) == round(rest + 12 * block)
    assert planner.dit_on_gpu(None, 36) == planner.dit_on_gpu(0, 36) == round(rest + 2 * block)
    assert planner.dit_on_gpu(None, 20) == round(rest + 18 * block)
    assert planner.dit_on_gpu(4, 20) == round(rest + 6 * block)
    for resident, swapped in ((1, 36), (17, 20), (-1, 0), (0, 37)):
        with pytest.raises(ValueError, match="of the DiT's 36"):
            planner.dit_on_gpu(resident, swapped)


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


def vae_fits(padded: tuple[int, int], tiles: Tiles | None) -> tuple[float, float]:
    """The encode's and the decode's fits, GiB above what is on the GPU before the call: the
    peaks less the VAE's weights and the DiT's rest, every block away, as the tables give them."""
    before = planner.VAE_WEIGHTS + planner.dit_on_gpu(0, 0)
    encode = planner.encode_peak(padded, tiles, resident=0) - before
    decode = planner.decode_peak(padded, output_size(padded), tiles, resident=0) - before
    return gib(encode), gib(decode)


def test_vae_tiled_at_4k() -> None:
    # planner-limits.md, "Tiled VAE at 4K", Encode peak and Decode peak, above what was allocated
    # before the call: the refit holds 1024 and 2048 px and the decode at 1536 px ("1536 within
    # 0.01 GiB") within 0.015 GiB of the columns as printed, its 0.01 and their rounding (the
    # furthest 0.013: the decode at 1536 px and the encode at 2048); the encode at 1536 px is
    # measured under it ("1536 0.33 GiB below": 21.45 against 21.79).
    measured = {1024: (10.44, 17.52), 1536: (21.45, 38.86), 2048: (37.67, 68.76)}
    for size, (encode, decode) in measured.items():
        fitted = vae_fits(NUMZ_4K, Tiles(size, 128))
        assert fitted == vae_fits(NUMZ_4K, Tiles(size))  # the overlap doesn't move them
        assert abs(fitted[1] - decode) <= 0.015
        if size == 1536:
            assert 0.33 <= fitted[0] - encode <= 0.345
        else:
            assert abs(fitted[0] - encode) <= 0.015
    # vram.md, Tiling (4K, batch 9), one decimal: 1024 px 10.4 / 17.4, 1536 px 21.4 / 38.7,
    # which the refit holds within 0.2 GiB but for that encode at 1536 px (0.39); and below 1024
    # px its own fit, residuals under 0.6 GiB: 512 px 3.5 / 5.9, 768 px 7.1 / 11.1.
    for size, (encode, decode) in {1024: (10.4, 17.4), 1536: (21.4, 38.7)}.items():
        fitted = vae_fits(NUMZ_4K, Tiles(size))
        assert 0 <= fitted[0] - encode <= 0.4 and 0 <= fitted[1] - decode <= 0.2
    for size, (encode, decode) in {512: (3.5, 5.9), 768: (7.1, 11.1)}.items():
        fitted = vae_fits(NUMZ_4K, Tiles(size))
        assert abs(fitted[0] - encode) < 0.6 and abs(fitted[1] - decode) < 0.6
    # The fit changes at 1024 px, and a larger tile never takes less.
    assert vae_fits(NUMZ_4K, Tiles(960)) == pytest.approx(
        (1.7 + 8.4 * 0.9216, 1.6 + 15.6 * 0.9216), abs=1e-6
    )
    assert vae_fits(NUMZ_4K, Tiles(1024)) == pytest.approx(
        (1.36 + 8.66 * 1.048576, 0.44 + 16.29 * 1.048576), abs=1e-6
    )
    sizes = [vae_fits(NUMZ_4K, Tiles(size)) for size in range(512, 3840, 64)]
    assert sizes == sorted(sizes)


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
    # A tile that covers the whole padded frame is the untiled pass.
    assert vae_fits(OURS_1080, Tiles(1920)) == vae_fits(OURS_1080, None)
    assert vae_fits(OURS_1080, Tiles(1856)) < vae_fits(OURS_1080, None)
    # A 1280-px tile at 1080p by its area, 1104x1280.
    assert vae_fits(OURS_1080, Tiles(1280))[1] == pytest.approx(0.44 + 16.29 * 1.41312, abs=1e-6)


def test_vae_phase_peaks() -> None:
    # A VAE phase's peak: its fit, the VAE's weights (0.47 GiB) and what of the DiT stays there;
    # DESIGN.md, BlockSwap, rewritten: an untiled 1080p decode "takes about 35 GiB with the VAE
    # alone", and the DiT's 15.35 GiB beside it pass a 48 GB card's budget.
    alone = planner.decode_peak(OURS_1080, (1080, 1920), resident=0)
    beside = planner.decode_peak(OURS_1080, (1080, 1920))
    assert round(gib(alone), 1) == 35.6 and round(gib(beside), 1) == 50.8
    assert beside - alone == round(36 * planner.BLOCK)
    assert beside > (48 - 1.4285) * GIB - planner.MARGIN > alone
    swapped = planner.decode_peak(OURS_1080, (1080, 1920), resident=0, swapped=36)
    assert swapped - alone == round(2 * planner.BLOCK)
    encode = planner.encode_peak(OURS_1080, Tiles(1024), 4, 20)
    assert encode == round((1.36 + 8.66 * 1.048576) * GIB) + planner.VAE_WEIGHTS + (
        planner.dit_on_gpu(4, 20)
    )


def test_split_term() -> None:
    # research/docs/colour.md, Cost and what streams: split takes 0.56 GiB per 4K frame on the
    # GPU; it corrects a slice of the decode at once, 5 frames at most, tiled or not: 2.8 GiB at
    # 4K, a quarter of it at 1080p.
    plain = planner.decode_peak(OURS_4K, (2160, 3840), Tiles(1024), 0)
    split = planner.decode_peak(OURS_4K, (2160, 3840), Tiles(1024), 0, split=True)
    assert gib(split - plain) == pytest.approx(5 * 0.56, abs=1e-6)
    plain = planner.decode_peak(OURS_1080, (1080, 1920), resident=0)
    split = planner.decode_peak(OURS_1080, (1080, 1920), resident=0, split=True)
    assert gib(split - plain) == pytest.approx(5 * 0.56 / 4, abs=1e-6)


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
    assert planner.SCRATCH_ALLOWANCE == 8 * GIB

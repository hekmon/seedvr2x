"""The memory planner (runtime/planner.py; DESIGN.md, Memory planner). Its model: each phase's
predicted peak against the measurements it was fitted on, every expected figure copied from
research/docs/planner-limits.md, research/docs/vram.md or DESIGN.md, the table named. Its plans:
DESIGN.md's window figures, the order of its choices, vram.md's recipes per card size, the pins
taken or refused, the floor, the system memory, and a resume's plan around a recorded shape. On
the CPU, no model."""

import re
import time
from dataclasses import replace

import pytest

from seedvr2x.runtime import planner
from seedvr2x.runtime.job import Shot, output_size, padded_size
from seedvr2x.runtime.plan import Plan, Tiles
from seedvr2x.runtime.planner import BLOCK, GIB, MARGIN, Pins, PlanError
from seedvr2x.runtime.shot import shot_layout

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


# The planner itself. Targets (job.target_size) and the memory a card has free. SMALL is a
# frame little larger than a tile, padded to 544x512, where the untiled passes take less than the
# tiled ones (test_floor).
HD, UHD = (1080, 1920), (2160, 3840)
SMALL = (512, 512)


def shots(*latents: int) -> list[Shot]:
    """Shots of these lengths in latents, one after the other: 4n - 3 frames each."""
    found: list[Shot] = []
    for count in latents:
        start = found[-1].end if found else 0
        found.append(Shot(start, start + 4 * count - 3))
    return found


def numz_free(card: int) -> int:
    """The memory numz's process found free on a card of `card` GB, in bytes (DESIGN.md, Memory
    planner): 94.50 GiB on the 96 GB card, N - 1.44 on a card emulated at N GB."""
    return round((94.50 if card == 96 else card - 1.44) * GIB)


def our_free(card: int) -> int:
    """The memory seedvr2x itself finds free, a first tensor allocated (cli._free_memory;
    measured 2026-10-10): 94.2746 GiB on the bare 96 GB card, N - 1.4285 on one emulated."""
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
    ("target", "card", "swapped", "design", "planned"),
    [
        # DESIGN.md, Memory planner, "By the fit, in step 0's padding": every block on the GPU,
        # at 1080p 5 latents on 24 GB, 13 on 32, 29 on 48 and 76 on 96; at 4K 3 on 32 GB, 7 on
        # 48, 19 on 96.
        (HD, 24, 0, 5, 6),
        (HD, 32, 0, 13, 14),
        (HD, 48, 0, 29, 30),
        (HD, 96, 0, 76, 77),
        (UHD, 32, 0, 3, 3),
        (UHD, 48, 0, 7, 7),
        (UHD, 96, 0, 19, 19),
        # All 36 blocks swapped: at 1080p "about" 11 on 16 GB, 19 on 24 and 27 on 32; at 4K 2 on
        # 16 GB, 4 on 24, 6 on 32, 10 on 48 and 22 on 96.
        (HD, 16, 36, 11, 12),
        (HD, 24, 36, 19, 20),
        (HD, 32, 36, 27, 28),
        (UHD, 16, 36, 2, 3),
        (UHD, 24, 36, 4, 5),
        (UHD, 32, 36, 6, 6),
        (UHD, 48, 36, 10, 10),
        (UHD, 96, 36, 22, 22),
    ],
)
def test_design_windows(
    target: tuple[int, int], card: int, swapped: int, design: int, planned: int
) -> None:
    # DESIGN.md's figures are the fit's with the VAE's weights in the DiT's phase, each of the 15
    # found again, with the memory numz's process found free and with seedvr2x's own reading.
    # The planner, its blocks pinned alike, sends the VAE's 0.47 GiB away when the window wants
    # them (its first choice): at 1080p, where a latent takes 1.0 GiB, one latent more on every
    # card; at 4K one more on 16 and 24 GB, 3 and 5.
    pins = Pins(swap_blocks=swapped)
    for free in (numz_free(card), our_free(card)):
        assert by_the_fit(free, target, swapped) == design
        assert in_one_window(free, target, pins) == planned
        made, peaks = planner.plan(free, target, shots(planned), pins, None)
        assert (made.window, made.swapped, peaks.window) == (planned, swapped, planned)
        assert made.vae_in_dit == (planned == design)
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
    # A longer shot than every block swapped holds: the largest cap that fits, 91 latents.
    made, peaks = planner.plan(free, HD, shots(92), Pins(), None)
    assert (peaks.window, made.swapped) == (47, 0)
    assert in_one_window(free, HD, Pins()) == 91


def test_the_longest_window_run_sets_the_peak() -> None:
    # A shot of 80 latents under a cap of 76 runs two windows of 41: the DiT holds 41 latents,
    # not 76, and no block is swapped for it.
    free = our_free(96)
    made, peaks = planner.plan(free, HD, shots(80), Pins(window=76), None)
    assert made == Plan(76)
    assert peaks.window == 41 and peaks.dit == planner.dit_peak(OURS_1080, 41)
    assert shot_layout(shots(80)[0].frames, 76) == [(0, 41), (39, 80)]
    # So with its job: the cap is the largest whose longest window fits, which a shot's length
    # may leave shorter than the cap. A 76-latent shot on a 24 GB card, every block swapped: a
    # cap of 20, the longest that fits there, lays it out in 5 windows of 17 at most, and the
    # fewest blocks that hold 17 latents are swapped, 28.
    made, peaks = planner.plan(our_free(24), HD, shots(76), Pins(), None)
    assert (made.window, peaks.window, made.swapped) == (20, 17, 28)
    assert in_one_window(our_free(24), HD, Pins()) == 20
    # The pin is the cap exactly, even above the longest shot, each shot then one window.
    made, peaks = planner.plan(free, HD, shots(3, 40, 2), Pins(window=60), None)
    assert (made.window, peaks.window) == (60, 40)
    # A job whose shots are all shorter than the smallest cap has its longest shot's length.
    made, peaks = planner.plan(free, HD, shots(3, 2), Pins(), None)
    assert (made.window, peaks.window) == (3, 3)
    made, peaks = planner.plan(free, HD, shots(3, 2), Pins(window=5), None)
    assert (made.window, peaks.window) == (5, 3)


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
        assert made.window is not None and peaks.window <= made.window and len(lines) >= 9
    assert planner.plan(our_free(96), HD, film, Pins(), None)[0].window == 91


def test_vae_choices() -> None:
    # Untiled where it fits, the DiT's blocks leaving the phase before a tile shrinks; else the
    # largest tile that fits with them away, then as many of them left there as fit.
    # 96 GB at 1080p: both untiled, every block there.
    made, _ = planner.plan(our_free(96), HD, shots(60), Pins(), None)
    assert (made.encode_tiles, made.decode_tiles) == (None, None)
    assert (made.resident_encode, made.resident_decode) == (None, None)
    # 48 GB at 1080p: the untiled decode, about 35 GiB with the VAE alone, fits with 24 of the
    # DiT's blocks beside it and not 25 (DESIGN.md, BlockSwap, rewritten, Per phase).
    free = our_free(48)
    budget = free - MARGIN
    made, peaks = planner.plan(free, HD, shots(21), Pins(), None)
    assert made == Plan(21, resident_decode=24)
    assert planner.decode_peak(OURS_1080, HD, None, 25) > budget >= peaks.decode
    assert peaks.decode == planner.decode_peak(OURS_1080, HD, None, 24) and peaks.bound == "decode"
    # 32 GB at 1080p: the decode in the largest tiles that fit with the DiT away, 1600 px, cut
    # off at the picture's edges, and none of its blocks left; the encode untiled, 22 left.
    free = our_free(32)
    budget = free - MARGIN
    made, peaks = planner.plan(free, HD, shots(12), Pins(), None)
    assert made == Plan(12, None, Tiles(1600), 0, 22, 0)
    assert planner.decode_peak(OURS_1080, HD, Tiles(1664), 0) > budget
    assert planner.decode_peak(OURS_1080, HD, Tiles(1600), 1) > budget >= peaks.decode
    assert planner.encode_peak(OURS_1080, None, 23) > budget >= peaks.encode
    # split's slice is counted in the decode's phase: a smaller tile with it.
    made, _ = planner.plan(free, HD, shots(12), Pins(), None, split=True)
    assert made == Plan(12, None, Tiles(1536), 0, 22, 1)
    # 24 GB at 4K: both tiled, the decode's smaller, its peak about twice the encode's by the
    # megapixel; every block swapped, their two slots there.
    made, peaks = planner.plan(our_free(24), UHD, shots(5), Pins(), None)
    assert made == Plan(5, Tiles(1472), Tiles(1088), 36, vae_in_dit=False)
    assert planner.grid(OURS_4K, Tiles(1472)) == (2, 3) and planner.grid(OURS_4K, Tiles(1088)) == (
        3,
        4,
    )
    assert peaks.scratch == planner.scratch(OURS_4K, 17) and peaks.tokens == 32_880


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
    # holding its batch's frames on the GPU, by 0.53 to 1.39 GiB at 1080p and 0.53 to 1.48 at 4K
    # (vram.md says up to 0.9 at 1080p and 1.4 to 2.2 at 4K, over its own fits).
    alone = planner.dit_on_gpu(0, swapped)
    fitted_encode = (planner.encode_peak(padded, encode, 0, swapped) - alone) / GIB
    fitted_decode = (planner.decode_peak(padded, target, decode, 0, swapped) - alone) / GIB
    assert abs(fitted_decode - measured[2]) <= 0.9
    assert 0.5 <= measured[0] - fitted_encode <= 1.5
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
    # no more blocks swapped, no smaller tiles.
    made, peaks = planner.plan(free, target, shot, Pins(), None, numz_padding=True)
    assert (made.window, peaks.window) == (latents, latents)
    assert made.swapped <= swapped
    for ours, theirs in ((made.encode_tiles, encode), (made.decode_tiles, decode)):
        assert ours is None or (theirs is not None and ours.size >= theirs.size)
    own = {
        (HD, 24): Plan(12, None, Tiles(1088), 16, 2, 1, vae_in_dit=False),
        (HD, 32): Plan(12, None, Tiles(1600), 0, 22, 1),
        (HD, 48): Plan(21, resident_decode=25),
        (UHD, 24): Plan(4, Tiles(1472), Tiles(1088), 27, 0, 1, vae_in_dit=False),
        (UHD, 32): Plan(7, Tiles(1728), Tiles(1280), 36, vae_in_dit=False),
        (UHD, 48): Plan(7, Tiles(2304), Tiles(1600), 0, 1, 7),
    }
    assert made == own[target, card]


def refused(*job: object, **options: object) -> str:
    with pytest.raises(PlanError) as error:
        planner.plan(*job, **options)  # pyright: ignore[reportArgumentType]
    return str(error.value)


def test_window_pin() -> None:
    # --window is the cap exactly, or refused, the refusal naming the longest cap that fits:
    # with every block swapped, or with the blocks pinned.
    free = our_free(24)
    made, peaks = planner.plan(free, HD, shots(76), Pins(window=20), None)
    assert (made.window, peaks.window) == (20, 17)
    made, peaks = planner.plan(free, HD, shots(76), Pins(window=8), None)
    assert (made.window, peaks.window, made.swapped) == (8, 8, 7)
    assert refused(free, HD, shots(76), Pins(window=30), None) == (
        "--window 30: its longest window, 27 latents, takes 29.17 GiB of free VRAM with every"
        " block of the DiT swapped, and the driver reports 22.57 GiB free: the longest window that"
        " fits is --window 20"
    )
    assert refused(free, HD, shots(76), Pins(window=12, swap_blocks=0), None) == (
        "--window 12: its longest window, 12 latents, takes 28.45 GiB of free VRAM with"
        " --swap-blocks 0, and the driver reports 22.57 GiB free: the longest window that fits is"
        " --window 6"
    )
    made, _ = planner.plan(free, HD, shots(76), Pins(window=6, swap_blocks=0), None)
    assert (made.window, made.swapped, made.vae_in_dit) == (6, 0, False)


def test_swap_pin() -> None:
    # --swap-blocks is the count exactly, the cap then the largest that fits with it; refused
    # when no window fits with it, the refusal naming the fewest blocks that hold the window; and
    # above the DiT's 36.
    free = our_free(24)
    made, peaks = planner.plan(free, HD, shots(76), Pins(swap_blocks=36), None)
    assert (made.window, peaks.window, made.swapped, made.vae_in_dit) == (20, 17, 36, True)
    made, peaks = planner.plan(free, HD, shots(76), Pins(swap_blocks=0), None)
    assert (made.window, peaks.window, made.swapped, made.vae_in_dit) == (6, 6, 0, False)
    # One and two swapped blocks save nothing, and are taken as pinned.
    made, _ = planner.plan(free, HD, shots(76), Pins(swap_blocks=2), None)
    assert (made.window, made.swapped) == (6, 2)
    free = our_free(32)
    assert refused(free, UHD, shots(40), Pins(swap_blocks=0), None) == (
        "--swap-blocks 0: a window of 5 latents, the shortest this job can run, takes 36.74 GiB of"
        " free VRAM with 0 blocks swapped, and the driver reports 30.57 GiB free: the fewest blocks"
        " that hold it are --swap-blocks 17"
    )
    made, _ = planner.plan(free, UHD, shots(40), Pins(swap_blocks=17), None)
    assert (made.window, made.swapped, made.vae_in_dit) == (5, 17, False)
    assert refused(free, UHD, shots(40), Pins(window=6, swap_blocks=0), None) == (
        "--swap-blocks 0: a window of 6 latents, --window 6's longest, takes 40.99 GiB of free"
        " VRAM with 0 blocks swapped, and the driver reports 30.57 GiB free: the fewest blocks"
        " that hold it are --swap-blocks 27"
    )
    assert refused(free, UHD, shots(40), Pins(swap_blocks=37), None) == (
        "--swap-blocks 37: the DiT has 36 blocks, so from 0 to 36"
    )


def test_tile_pins() -> None:
    # A tile pin is the tile exactly, or refused, naming the largest tile that fits; 0 pins an
    # untiled pass, refused likewise; a tile that covers the whole padded frame is no tile.
    free = our_free(24)
    made, _ = planner.plan(free, UHD, shots(5), Pins(), None)
    assert (made.encode_tiles, made.decode_tiles) == (Tiles(1472), Tiles(1088))
    pins = Pins(encode_tile=1024, decode_tile=1024)
    made, peaks = planner.plan(free, UHD, shots(5), pins, None)
    assert (made.encode_tiles, made.decode_tiles) == (Tiles(1024), Tiles(1024))
    assert refused(free, UHD, shots(5), Pins(decode_tile=2048), None) == (
        "--decode-tile 2048: the decode in tiles of 2048 px takes 70.90 GiB of free VRAM, the"
        " DiT's blocks off the GPU, and the driver reports 22.57 GiB free: the largest tile that"
        " fits is --decode-tile 1088"
    )
    assert refused(free, UHD, shots(5), Pins(encode_tile=0), None) == (
        "--encode-tile 0: the encode untiled takes 77.40 GiB of free VRAM, the DiT's blocks off"
        " the GPU, and the driver reports 22.57 GiB free: the largest tile that fits is"
        " --encode-tile 1472"
    )
    assert refused(free, UHD, shots(5), Pins(decode_tile=3840), None) == (
        "--decode-tile 3840: a tile of 3840 px covers the whole padded frame, 3840x2192:"
        " --decode-tile 0 pins an untiled decode"
    )
    assert refused(free, HD, shots(5), Pins(encode_tile=1920), None) == (
        "--encode-tile 1920: a tile of 1920 px covers the whole padded frame, 1920x1104:"
        " --encode-tile 0 pins an untiled encode"
    )
    # A tile pinned where the untiled pass would fit is taken as pinned; and 0 where it fits.
    pins = Pins(encode_tile=0, decode_tile=1024)
    made, peaks = planner.plan(our_free(96), HD, shots(40), pins, None)
    assert made == Plan(40, None, Tiles(1024))
    assert peaks.scratch == planner.scratch(OURS_1080, 157)
    # The largest tile under the padded frame's longer side is one: 1856 px at 1080p.
    made, _ = planner.plan(our_free(96), HD, shots(40), Pins(decode_tile=1856), None)
    assert made.decode_tiles == Tiles(1856) and planner.grid(OURS_1080, Tiles(1856)) == (1, 2)
    # A frame little larger than a tile, 544x512 padded, where the untiled decode takes less
    # than its one tile size: the tile refused names the untiled pass, which fits, and no tile.
    assert refused(round(7.6 * GIB), SMALL, shots(40), Pins(decode_tile=512), None) == (
        "--decode-tile 512: the decode in tiles of 512 px takes 7.82 GiB of free VRAM, the DiT's"
        " blocks off the GPU, and the driver reports 7.60 GiB free: no tile fits, the untiled"
        " decode does, --decode-tile 0"
    )


def test_floor() -> None:
    # A size that doesn't fit is refused, never degraded, the refusal counting in free VRAM:
    # what the size takes, what the driver reports free, and the largest output that fits there
    # (DESIGN.md, Memory planner). At 4K the smallest window, 5 latents, takes 22.40 GiB of free
    # VRAM by the model, every block swapped and the VAE away: a peak of 21.80 GiB and the margin
    # (DESIGN.md: "22 to 23 GiB of free VRAM by the fit").
    assert refused(numz_free(16), UHD, shots(40), Pins(), None) == (
        "an output of 3840x2160 takes 22.40 GiB of free VRAM (its smallest window, 5 latents, with"
        " every block of the DiT swapped and the VAE off the GPU: a peak of 21.80 GiB, and 0.60 of"
        " margin), and the driver reports 14.56 GiB free: the largest output that fits there is"
        " --resolution 1680, about 2986x1680"
    )
    # The pins change nothing to it.
    pins = Pins(window=5, swap_blocks=36, decode_tile=512)
    assert refused(numz_free(16), UHD, shots(40), pins, None).startswith("an output of 3840x2160")
    # An emulated 24 GB card has that, by 0.16 GiB (22.56 GiB free; 22.57 by seedvr2x's own
    # reading): 4K runs there in windows of 5 latents.
    for free in (numz_free(24), our_free(24)):
        made, peaks = planner.plan(free, UHD, shots(40), Pins(), None)
        assert made == Plan(5, Tiles(1472), Tiles(1088), 36, vae_in_dit=False)
        assert round(peaks.dit / GIB, 2) == 21.79 and peaks.bound == "dit"
    # With 0.2 GiB of it taken by anything else, it is refused, the next size down fitting.
    assert refused(round(22.36 * GIB), UHD, shots(40), Pins(), None) == (
        "an output of 3840x2160 takes 22.40 GiB of free VRAM (its smallest window, 5 latents, with"
        " every block of the DiT swapped and the VAE off the GPU: a peak of 21.80 GiB, and 0.60 of"
        " margin), and the driver reports 22.36 GiB free: the largest output that fits there is"
        " --resolution 2144, about 3810x2144"
    )
    made, _ = planner.plan(round(22.36 * GIB), (2144, 3811), shots(40), Pins(), None)
    assert made.window == 5
    # A job whose shots are all shorter runs its longest shot's length: 3 latents on 16 GB.
    made, _ = planner.plan(numz_free(16), UHD, shots(3, 2), Pins(), None)
    assert made == Plan(3, Tiles(1088), Tiles(832), 36, vae_in_dit=False)
    # A portrait target is scaled by its short side too, the output said width x height.
    assert refused(numz_free(16), (3840, 2160), shots(40), Pins(), None).endswith(
        "the largest output that fits there is --resolution 1696, about 1696x3014"
    )
    # The VAE's floor, 512-px tiles with the DiT's blocks away: 1080p on an 8 GB card.
    assert refused(numz_free(8), HD, shots(40), Pins(), None) == (
        "an output of 1920x1080 takes 7.82 GiB of free VRAM (its decode in tiles of 512 px, with"
        " the DiT's blocks off the GPU: a peak of 7.22 GiB, and 0.60 of margin), and the driver"
        " reports 6.56 GiB free: the largest output that fits there is --resolution 320, about"
        " 568x320"
    )
    assert refused(GIB, HD, shots(40), Pins(), None).endswith(
        "and the driver reports 1.00 GiB free: no output size fits there"
    )
    # A frame little larger than a tile has its untiled passes for floor too, which take less
    # than its tiles there, the small tiles' fits starting higher: 512x512, padded to 544x512,
    # decodes untiled in 6.81 GiB and in tiles of 512 px in 7.22. It runs from 7.41 GiB free,
    # untiled, pinned so or not, and is refused under that by the untiled pass's figures.
    assert padded_size(SMALL) == (544, 512)
    untiled, tiled = (
        planner.decode_peak((544, 512), SMALL, tiles, 0, 36) for tiles in (None, Tiles(512))
    )
    assert (round(gib(untiled), 2), round(gib(tiled), 2)) == (6.81, 7.21)
    for pins in (Pins(), Pins(encode_tile=0, decode_tile=0)):
        for free in (untiled + MARGIN, round(7.6 * GIB)):
            made, peaks = planner.plan(free, SMALL, shots(40), pins, None)
            assert (made.window, made.encode_tiles, made.decode_tiles) == (40, None, None)
            assert peaks.decode == untiled or made.resident_decode != 0
            assert planner.fits(made, free, SMALL, shots(40), None) is None
        assert refused(round(7.3 * GIB), SMALL, shots(40), pins, None) == (
            "an output of 512x512 takes 7.41 GiB of free VRAM (its decode untiled, which takes"
            " less than tiles at that size, with the DiT's blocks off the GPU: a peak of 6.81 GiB,"
            " and 0.60 of margin), and the driver reports 7.30 GiB free: the largest output that"
            " fits there is --resolution 496, about 496x496"
        )
    made, _ = planner.plan(round(7.3 * GIB), (496, 496), shots(40), Pins(), None)
    assert (made.encode_tiles, made.decode_tiles) == (None, None)
    # The other padded sizes found refused where a plan fits, each planned at the least its
    # untiled decode takes, and refused a byte under.
    sizes = (((506, 506), (544, 544)), ((522, 490), (560, 528)), ((506, 402), (544, 432)))
    for target, padded in sizes:
        assert padded_size(target) == padded
        floor = planner.decode_peak(padded, target, None, 0, 36) + MARGIN
        assert floor < planner.decode_peak(padded, target, Tiles(512), 0, 36) + MARGIN
        made, _ = planner.plan(floor, target, shots(40), Pins(), None)
        assert made.decode_tiles is None
        assert "untiled, which takes less than tiles" in refused(
            floor - 1, target, shots(40), Pins(), None
        )
    # The size named is the target's scaled, 2 px off what --resolution makes of the source: a
    # 1920x1080 source is 1066x600 at --resolution 600, and 796x448 at --resolution 448.
    assert refused(round(7.6 * GIB), (600, 1066), shots(40), Pins(), None).endswith(
        "the largest output that fits there is --resolution 448, about 794x448"
    )
    assert planner.plan(round(7.6 * GIB), (448, 796), shots(40), Pins(), None)[0].window == 39
    # A frame too small for a tile has its untiled passes for floor.
    encode = planner.encode_peak((128, 128), None, 0, 36)
    assert refused(encode + MARGIN - 1, (96, 128), shots(6), Pins(), None).startswith(
        "an output of 128x96 takes 3.47 GiB of free VRAM (its encode untiled, too small for a"
        " tile, with the DiT's blocks off the GPU: a peak of 2.87 GiB, and 0.60 of margin)"
    )
    made, _ = planner.plan(encode + MARGIN, (96, 128), shots(6), Pins(), None)
    assert (made.encode_tiles, made.resident_encode, made.swapped) == (None, 0, 32)


def test_every_plan_fits() -> None:
    # Over a grid of sizes, shots, memory and pins: a plan returned fits (fits), and takes
    # every pin exactly; a size refused names a --resolution that plans, the largest that does.
    targets = ((96, 128), SMALL, (528, 560), (600, 1066), HD, UHD)
    films = (shots(1), shots(3, 2), shots(40), shots(5, 76, 133))
    frees = (4.0, 7.3, 7.6, 10.5, 14.56, 22.56, 46.56, 94.27)
    pin_sets = (
        Pins(),
        Pins(window=5),
        Pins(window=12, swap_blocks=36),
        Pins(swap_blocks=0),
        Pins(swap_blocks=20),
        Pins(encode_tile=0, decode_tile=0),
        Pins(decode_tile=512),
        Pins(encode_tile=1024, decode_tile=768),
    )
    plans = floors = 0
    for target in targets:
        for film in films:
            for free in (round(count * GIB) for count in frees):
                for pins in pin_sets:
                    try:
                        made, _ = planner.plan(free, target, film, pins, None)
                    except PlanError as error:
                        floors += floor_named(str(error), free, target, film)
                        continue
                    plans += 1
                    assert planner.fits(made, free, target, film, None) is None
                    assert pins.window in (None, made.window)
                    assert pins.swap_blocks in (None, made.swapped)
                    for pin, tiles in (
                        (pins.encode_tile, made.encode_tiles),
                        (pins.decode_tile, made.decode_tiles),
                    ):
                        assert pin is None or tiles == (Tiles(pin) if pin else None)
    assert plans > 500 and floors > 100


def floor_named(said: str, free: int, target: tuple[int, int], film: list[Shot]) -> bool:
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

    planner.plan(free, scaled(short), film, Pins(), None)
    if short + 16 < height:
        assert refused(free, scaled(short + 16), film, Pins(), None).startswith("an output of")
    return True


def test_host_memory() -> None:
    # The system memory the plan takes: the process's own 2.4 GiB, a pinned copy of each block
    # that leaves the GPU in some phase and of the VAE's weights when they leave, and a tiled
    # shot's scratch up to its allowance; over what is available, refused, naming both.
    free = our_free(24)
    made, peaks = planner.plan(free, HD, shots(76), Pins(), 64 * GIB)
    assert made == Plan(20, None, Tiles(1088), 28, 1, 1, vae_in_dit=False)
    kept = planner.scratch(OURS_1080, 301)
    assert peaks.scratch == kept == 301 * 12_718_080
    pinned = round(35 * BLOCK) + planner.VAE_WEIGHTS
    assert peaks.host == planner.HOST_RESIDENT + pinned + kept
    assert refused(free, HD, shots(76), Pins(), 8 * GIB) == (
        "the plan takes 21.21 GiB of system memory (2.40 for the process, 15.24 pinned for the 35"
        " blocks of the DiT and the VAE's weights that leave the GPU, 3.57 for a tiled shot's"
        " scratch) and 8.00 GiB are available, 13.21 GiB short"
    )
    assert planner.plan(free, HD, shots(76), Pins(), peaks.host) == (made, peaks)
    assert "of system memory" in refused(free, HD, shots(76), Pins(), peaks.host - 1)
    assert planner.plan(free, HD, shots(76), Pins(), None) == (made, peaks)  # not checked
    # Nothing leaves the GPU, nothing tiled: the process alone.
    made, peaks = planner.plan(our_free(96), HD, shots(40), Pins(), 4 * GIB)
    assert made == Plan(40) and peaks.host == planner.HOST_RESIDENT and peaks.scratch == 0
    assert refused(our_free(96), HD, shots(40), Pins(), 2 * GIB) == (
        "the plan takes 2.40 GiB of system memory (2.40 for the process) and 2.00 GiB are"
        " available, 0.40 GiB short"
    )
    # A long tiled shot's scratch counts up to its allowance, the rest a file: 4K, 601 frames.
    made, peaks = planner.plan(our_free(96), UHD, shots(151), Pins(), 64 * GIB)
    assert made.decode_tiles is not None and peaks.scratch == 601 * 50_503_680
    assert (
        peaks.host
        == planner.HOST_RESIDENT
        + round(
            planner._leaving(made) * BLOCK  # pyright: ignore[reportPrivateUsage]
        )
        + (0 if made.vae_in_dit else planner.VAE_WEIGHTS)
        + planner.SCRATCH_ALLOWANCE
    )
    # Two swapped blocks save nothing on the GPU, their slots there in every phase, and wait in
    # pinned copies all the same, with the 12 the untiled decode sends away.
    made, peaks = planner.plan(our_free(48), HD, shots(12), Pins(swap_blocks=2), None)
    assert made == Plan(12, swapped=2, resident_decode=22)
    assert peaks.host == planner.HOST_RESIDENT + round(14 * BLOCK)


def test_resumed_around_the_recorded_shape() -> None:
    # A resume takes the recorded shape, cap, tiles and compile flags, as it is, and plans where
    # the weights are again from the memory free that day (DESIGN.md, Memory planner, A plan
    # shapes the output a little): with more, fewer blocks swapped; with less, more.
    recorded = Plan(20, None, Tiles(1088), 28, 1, 1, vae_in_dit=False)
    job = (HD, shots(76), Pins(), None)
    assert planner.plan(our_free(24), *job)[0] == recorded
    assert planner.plan(our_free(24), *job, recorded=Plan(20, None, Tiles(1088)))[0] == recorded
    made, peaks = planner.plan(our_free(48), *job, recorded=recorded)
    assert made == Plan(20, None, Tiles(1088))
    assert (peaks.window, peaks.free, peaks.bound) == (17, our_free(48), "encode")
    # Planned whole, that card has no tiles and two windows of 39 latents: the shape is the
    # record's.
    whole, peaks = planner.plan(our_free(48), *job)
    assert whole == Plan(75, swapped=24, vae_in_dit=False) and peaks.window == 39
    # With 0.58 GiB less, two more blocks swapped, and none left in the VAE's phases.
    made, _ = planner.plan(round(21.99 * GIB), *job, recorded=recorded)
    assert made == Plan(20, None, Tiles(1088), 30, 0, 0, vae_in_dit=False)
    # Its tiles too, their overlap with them, and its compile flags, whatever would be chosen.
    odd = Plan(8, Tiles(768, 128), Tiles(576, 96), compile_dit=True)
    made, _ = planner.plan(our_free(96), *job, recorded=odd)
    assert made == odd
    # A recorded shape that no longer fits stops the resume, naming the phase and the shortfall,
    # and never another pin: here its untiled encode, then its windows.
    assert refused(our_free(16), *job, recorded=recorded) == (
        "the recorded plan no longer fits the memory free now: the encode, untiled, with the"
        " DiT's blocks off the GPU, takes 21.98 GiB of free VRAM and the driver reports 14.57 GiB"
        " free, 7.41 GiB short"
    )
    tiled = Plan(20, Tiles(1152), Tiles(1088))
    assert refused(our_free(16), *job, recorded=tiled) == (
        "the recorded plan no longer fits the memory free now: the DiT's phase, a window of 17"
        " latents with 36 blocks swapped and the VAE off the GPU, takes 19.05 GiB of free VRAM"
        " and the driver reports 14.57 GiB free, 4.48 GiB short"
    )
    assert refused(our_free(24), HD, shots(76), Pins(), 8 * GIB, recorded=recorded) == (
        "the recorded plan no longer fits the memory free now: the plan takes 21.21 GiB of system"
        " memory (2.40 for the process, 15.24 pinned for the 35 blocks of the DiT and the VAE's"
        " weights that leave the GPU, 3.57 for a tiled shot's scratch) and 8.00 GiB are"
        " available, 13.21 GiB short"
    )
    # --swap-blocks still applies, taken exactly; where it alone doesn't hold the recorded
    # windows, it is the pin that is refused.
    made, _ = planner.plan(our_free(24), HD, shots(76), Pins(swap_blocks=36), None, recorded=tiled)
    assert made == Plan(20, Tiles(1152), Tiles(1088), 36)
    pins = Pins(swap_blocks=3)
    assert refused(our_free(24), HD, shots(76), pins, None, recorded=recorded) == (
        "--swap-blocks 3: a window of 17 latents, the recorded plan's, takes 32.97 GiB of free"
        " VRAM with 3 blocks swapped, and the driver reports 22.57 GiB free: the fewest blocks"
        " that hold it are --swap-blocks 28"
    )
    # The other pins were the record's when it was made: the shape is taken from it alone.
    pins = Pins(window=5, encode_tile=512, decode_tile=512)
    assert planner.plan(our_free(24), HD, shots(76), pins, None, recorded=recorded)[0] == recorded


def test_recorded_cap_checked() -> None:
    # A cap under 5 latents is the planner's own for a job whose longest shot is as short, each
    # shot one window; with a longer shot no layout exists for it (shot.window_layout), and a
    # record that has one was edited: refused, where the layout would raise.
    made, _ = planner.plan(our_free(96), HD, shots(3, 2), Pins(), None)
    assert made.window == 3
    planner.check_recorded(made, shots(3, 2))
    planner.check_recorded(Plan(4), shots(4, 1))
    planner.check_recorded(Plan(5), shots(76))
    planner.check_recorded(Plan(None), shots(76))
    with pytest.raises(PlanError) as error:
        planner.check_recorded(Plan(3), shots(3, 6))
    assert str(error.value) == (
        "not a plan as seedvr2x records one: its window, 3, with a shot of 6 latents: a cap that"
        " a shot is longer than is 5 latents at least"
    )
    with pytest.raises(ValueError, match="windows of 3 latents leave none of their own"):
        shot_layout(21, 3)
    with pytest.raises(PlanError, match="its window, 4, with a shot of 5 latents"):
        planner.check_recorded(Plan(4), shots(5))


def test_fits() -> None:
    # A plan's phases against the budget, in a shot's order, the first that doesn't fit named
    # with what it is short of; then the system memory.
    free, job = our_free(24), (HD, shots(12), None)
    assert planner.fits(Plan(12), free, *job) == (
        "the encode, untiled, with the DiT's 36 blocks on the GPU, takes 36.33 GiB of free VRAM"
        " and the driver reports 22.57 GiB free, 13.76 GiB short"
    )
    assert planner.fits(Plan(12, swapped=17), free, *job) == (
        "the encode, untiled, with the DiT's 19 unswapped blocks on the GPU, takes 30.00 GiB of"
        " free VRAM and the driver reports 22.57 GiB free, 7.43 GiB short"
    )
    away = Plan(12, None, None, 17, 0, 0, vae_in_dit=False)
    assert planner.fits(away, free, *job) == (
        "the decode, untiled, with the DiT's blocks off the GPU, takes 37.06 GiB of free VRAM and"
        " the driver reports 22.57 GiB free, 14.49 GiB short"
    )
    assert planner.fits(replace(away, swapped=3), free, *job) == (
        "the DiT's phase, a window of 12 latents with 3 blocks swapped and the VAE off the GPU,"
        " takes 28.03 GiB of free VRAM and the driver reports 22.57 GiB free, 5.46 GiB short"
    )
    tiled = replace(away, decode_tiles=Tiles(1088))
    assert planner.fits(tiled, free, *job) is None
    assert planner.fits(tiled, free, HD, shots(12), 4 * GIB) == (
        "the plan takes 18.60 GiB of system memory (2.40 for the process, 15.66 pinned for the 36"
        " blocks of the DiT and the VAE's weights that leave the GPU, 0.54 for a tiled shot's"
        " scratch) and 4.00 GiB are available, 14.60 GiB short"
    )
    # One block left unswapped is one block.
    assert planner.fits(Plan(12, swapped=35), free, *job) == (
        "the decode, untiled, with the DiT's 1 unswapped block on the GPU, takes 37.48 GiB of"
        " free VRAM and the driver reports 22.57 GiB free, 14.91 GiB short"
    )
    assert planner.fits(replace(tiled, resident_decode=4), free, *job) == (
        "the decode, in tiles of 1088 px, with 4 of the DiT's blocks on the GPU, takes 23.54 GiB"
        " of free VRAM and the driver reports 22.57 GiB free, 0.97 GiB short"
    )
    # With split, its slice is the decode's too: one block there fits without it, not with it.
    there = replace(tiled, resident_decode=1)
    assert planner.fits(there, free, *job) is None
    assert planner.fits(there, free, *job, split=True) == (
        "the decode, in tiles of 1088 px, with 1 of the DiT's blocks on the GPU, takes 22.98 GiB"
        " of free VRAM and the driver reports 22.57 GiB free, 0.41 GiB short"
    )


def test_peaks_record() -> None:
    # What the manifest keeps of a plan's memory, for information: bytes, the window in latents.
    made, peaks = planner.plan(our_free(24), HD, shots(76), Pins(), 64 * GIB)
    assert peaks.record() == {
        "free": our_free(24),
        "budget": our_free(24) - MARGIN,
        "encode_peak": planner.encode_peak(OURS_1080, None, 1, 28),
        "dit_peak": planner.dit_peak(OURS_1080, 17, 28, vae_in_dit=False),
        "decode_peak": planner.decode_peak(OURS_1080, HD, Tiles(1088), 1, 28),
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
    assert replace(peaks, decode=peaks.dit + 1).bound == "decode"
    assert set(peaks.record()).isdisjoint(made.record()["memory"])


def test_said() -> None:
    # The plan's lines, as --plan prints them and every run logs them (DESIGN.md, Memory planner,
    # --plan): 1080p on a 24 GB card, a shot of 301 frames, with split.
    job = (HD, shots(76), Pins())
    made, peaks = planner.plan(numz_free(24), *job, 64 * GIB, split=True)
    assert planner.said(made, peaks, "NVIDIA RTX PRO 6000 Blackwell", *job) == [
        "plan: NVIDIA RTX PRO 6000 Blackwell, 22.56 GiB free, a budget of 21.96 GiB",
        "plan: frames padded to 1920x1104, 8,280 tokens a latent",
        "plan: 1 shot, the longest of 301 frames (76 latents)",
        "plan: windows of 20 latents at most, the longest run 17 latents: 0 of 1 shot in one"
        " window, 4 joins",
        "plan: 28 of the DiT's 36 blocks swapped, the VAE off the GPU in the DiT's phase; in the"
        " encode, 1 of the DiT's blocks on the GPU; in the decode, the DiT's blocks off the GPU",
        "plan: 15.66 GiB of system memory pinned for 36 blocks and the VAE; 21.63 GiB of system"
        " memory in all",
        "plan: encode untiled, decode in tiles of 1088 px, 2 x 2",
        "plan: predicted peaks: encode 21.80 GiB, DiT 21.82 GiB, decode 21.95 GiB; the decode"
        " bounds the plan, 0.01 GiB under the budget",
        "plan: scratch of the longest tiled shot: 3.57 GiB, in system memory",
        "plan: nothing pinned",
    ]
    # 4K on the 96 GB card as DESIGN.md first read it, 94.34 GiB free, the same shot, pinned.
    pins = Pins(window=19, swap_blocks=0, encode_tile=0, decode_tile=2048)
    made, peaks = planner.plan(round(94.34 * GIB), UHD, shots(76), pins, 64 * GIB, split=True)
    assert planner.said(made, peaks, "a card", UHD, shots(76), pins)[3:] == [
        "plan: windows of 19 latents at most, the longest run 17 latents: 0 of 1 shot in one"
        " window, 4 joins",
        "plan: no block of the DiT swapped, the VAE on the GPU in the DiT's phase; in the encode,"
        " the DiT's 36 blocks on the GPU; in the decode, the DiT's 36 blocks on the GPU",
        "plan: no system memory pinned, nothing leaving the GPU; 10.40 GiB of system memory in all",
        "plan: encode untiled, decode in tiles of 2048 px, 2 x 2",
        "plan: predicted peaks: encode 91.14 GiB, DiT 84.72 GiB, decode 87.44 GiB; the encode"
        " bounds the plan, 2.60 GiB under the budget",
        "plan: scratch of the longest tiled shot: 14.16 GiB, 8.00 GiB of it in system memory, the"
        " rest in a file",
        "plan: pinned: --window 19, --swap-blocks 0, --encode-tile 0, --decode-tile 2048",
    ]
    # A recorded plan without a cap, each shot in one window; one block left unswapped; and a
    # job of one frame, its cap one latent.
    job = (HD, shots(40), Pins())
    made, peaks = planner.plan(our_free(96), *job, None, recorded=Plan(None))
    assert planner.said(made, peaks, "a card", *job)[3] == (
        "plan: no cap on the windows, the longest run 40 latents: 1 of 1 shot in one window,"
        " 0 joins"
    )
    made, peaks = planner.plan(our_free(96), *job, None, recorded=Plan(40))
    said = planner.said(replace(made, swapped=35), peaks, "a card", *job)
    assert said[4].endswith(
        "in the encode, the DiT's 1 unswapped block on the GPU; in the decode, the DiT's 1"
        " unswapped block on the GPU"
    )
    job = (HD, shots(1), Pins())
    made, peaks = planner.plan(our_free(96), *job, None)
    assert planner.said(made, peaks, "a card", *job)[2:4] == [
        "plan: 1 shot, the longest of 1 frame (1 latent)",
        "plan: windows of 1 latent at most, the longest run 1 latent: 1 of 1 shot in one window,"
        " 0 joins",
    ]
    # A cap no shot reaches: each in one window, no join.
    job = (HD, shots(3, 40, 2), Pins())
    made, peaks = planner.plan(our_free(96), *job, None)
    assert planner.said(made, peaks, "a card", *job)[2:4] == [
        "plan: 3 shots, the longest of 157 frames (40 latents)",
        "plan: windows of 40 latents at most, the longest run 40 latents: 3 of 3 shots in one"
        " window, 0 joins",
    ]

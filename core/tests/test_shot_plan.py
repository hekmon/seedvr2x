"""The memory planner's plan (seedvr2x/shot_plan.py; DESIGN.md, Memory planner). Its model,
each phase's predicted peak, is the bricks' (peaks.py, tests/test_peaks.py); here the model's
constants and what the plan counts with them: where the weights are and what they take against
seedvr2x/blockswap.py's own rules, the tiles a pass takes. Its plans: DESIGN.md's window figures,
the order of its choices, vram.md's recipes per card size, the pins taken or refused, the floor,
the DiT's phase judged by its longest window whatever the shots' lengths, the system memory, and
a resume's plan around a recorded shape. On the CPU, no weights: the models are built on the meta
device where their shapes are counted. What the command line does with a plan, its lines and a
resume's checks, is the CLI's tests/test_planner.py's."""

import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from library_fixtures import cli_say, cli_words, plain_fit  # noqa: F401
from test_peaks import (
    BEFORE_ONE_OUTPUT,
    OURS_4K,
    OURS_1080,
    gib,
    largest,
    most,
    tile_shapes,
    vae_fits,
)

from seedvr2x import memory, peaks, shot_plan, weights
from seedvr2x.errors import (
    CompileRefused,
    HostShort,
    MemoryShort,
    PinRefused,
    PlanRefused,
    Setting,
)
from seedvr2x.frames import output_size, padded_length, padded_size
from seedvr2x.memory import Reading
from seedvr2x.passes import Phase, Plan, Tiles
from seedvr2x.peaks import BLOCK, GIB, MARGIN
from seedvr2x.shot_plan import Pins, PlanError
from seedvr2x.shots import Shot, shot_layout


def test_constants() -> None:
    # The figures the documents give, in bytes: DESIGN.md, Memory planner (Budget, Host RAM,
    # Per-phase peaks) and BlockSwap, rewritten; the weights' parts as milestone 3 measured them
    # on seedvr2x (2026-10-11): a block 0.4220 GiB, the 36 15.19, the DiT's other parameters
    # 0.155, the VAE 0.4669, DESIGN.md's 0.47; and 59,643,736 bytes of buffers that never move,
    # cuBLAS's workspace among them (the census of 2026-11-07, the RoPE tables cleared before a VAE
    # phase; 0.058 GiB before, measured after windows of 1 latent, their tables in it).
    assert gib(peaks.MARGIN) == pytest.approx(0.6)
    assert gib(peaks.DIT_LOADED) == pytest.approx(15.871)
    assert peaks.PER_TOKEN == 61.243 * 1024 and peaks.PER_WINDOW == 0.996 * 1024**2
    assert (peaks.BLOCKS, peaks.SLOTS) == (36, 2)
    # The blocks, the slots and the DiT's other parameters are the model's own constants
    # (seedvr2x/weights.py), the 7B's, the one DiT seedvr2x runs, in fp16 (peaks.DIT, the
    # module's BLOCK), in int8 or in the dynamic GGUF files; a block is one size in fp16 and int8,
    # which the planner counts the 36 by, and its own in a GGUF file (peaks.block_sizes).
    assert peaks.DIT is peaks.FP16 is weights.DIT_7B
    assert (
        peaks.DITS
        == weights.ACCEPTED["dit"]
        == (
            weights.DIT_7B,
            weights.DIT_7B_INT8,
            weights.DIT_7B_DYN,
            weights.DIT_7B_SHARP_DYN,
        )
    )
    assert peaks.DIT.blocks == (peaks.BLOCK,) * peaks.BLOCKS
    assert peaks.INT8.blocks == (peaks.block_bytes(peaks.INT8),) * peaks.BLOCKS
    assert (peaks.SLOTS, peaks.DIT_OTHER) == (weights.SLOTS, weights.DIT_7B.other)
    assert f"{gib(peaks.BLOCK):.4f}" == "0.4220" and f"{gib(36 * peaks.BLOCK):.2f}" == "15.19"
    assert f"{gib(peaks.DIT_OTHER):.3f}" == "0.155"
    assert f"{gib(peaks.VAE_WEIGHTS):.4f}" == "0.4669"
    assert peaks.DIT_BUFFERS == 59_643_736
    # The DiT's 15.35 GiB, and with the VAE the 15.81 a plan pins that keeps the two apart.
    dit = 36 * peaks.BLOCK + peaks.DIT_OTHER
    assert f"{gib(dit):.2f}" == "15.35" and f"{gib(dit + peaks.VAE_WEIGHTS):.2f}" == "15.81"
    # The loaded models by their parts: 15.8699 GiB, 0.0011 under the fit's constant, within the
    # 0.005 GiB it holds its points to (0.0013 over while the buffers held a 1-latent window's
    # RoPE tables, before 2026-11-07).
    parts = dit + peaks.DIT_BUFFERS + peaks.VAE_WEIGHTS
    assert gib(parts - peaks.DIT_LOADED) == pytest.approx(-0.0011, abs=5e-5)
    assert gib(shot_plan.HOST_RESIDENT) == pytest.approx(2.4)
    assert (peaks.TILE_MIN, peaks.TILE_STEP, peaks.SLICE) == (512, 64, 5)
    # The corrections of the planner's first two matrices (test_first_matrix): 0.4 GiB on every
    # encode (PROVISIONAL, until the encode is refitted on seedvr2x); no tile chosen above 2048
    # px, and 1.25 GiB left under the budget by every VAE pass besides its peak (DESIGN.md, Memory
    # planner: What follows from them; Budget; 1.5 until 2026-10-18, provisional then).
    assert gib(peaks.ENCODE_EXTRA) == pytest.approx(0.4)
    assert peaks.TILE_MAX == 2048 and gib(peaks.VAE_ROOM) == pytest.approx(1.25)


IMPORTED = """
import sys

from seedvr2x import peaks
from seedvr2x import memory
from seedvr2x import shot_plan
from seedvr2x.shots import Shot
from seedvr2x.passes import Plan, Tiles

padded = (1104, 1920)
assert peaks.encode_peak(padded, Tiles(1024)) > peaks.encode_peak(padded, Tiles(512))
# A resume's early look at the card plans its recorded shape before torch is imported
# (cli._early_memory): the shots' layouts (seedvr2x/shots.py) need none.
made, _ = shot_plan.plan(
    memory.early(96 << 30), (1080, 1920), [Shot(0, 300)], shot_plan.Pins(), None, recorded=Plan(8)
)
assert made.window == 8
print(*sorted(name for name in ("torch", "numpy", "torchvision") if name in sys.modules))
"""


def test_imports_no_torch() -> None:
    # The command line imports the planner before its first refusals, which don't wait for torch
    # (tests/test_cli.py): a process that has imported it, with the models' constants
    # (seedvr2x/weights.py) and the reading's type (seedvr2x/memory.py), and counted a tile's peak
    # over the tiled passes' grid (seedvr2x/tilegrid.py), has loaded no torch, nor has planning a
    # recorded shape in a reading taken before it (memory.early): the shots' layouts are
    # seedvr2x/shots.py's.
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
    from seedvr2x import blockswap
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
    assert len(dit.blocks) == peaks.BLOCKS
    assert (sum(block), sum(other)) == (peaks.BLOCK, peaks.DIT_OTHER)
    assert laid_out(weights) == peaks.VAE_WEIGHTS
    plan = Plan(5, swapped=36, resident_encode=0, vae_in_dit=False)
    pinned = shot_plan._pinned(plan, peaks.FP16)  # pyright: ignore[reportPrivateUsage]
    assert pinned == 36 * paged(laid_out(block)) + paged(laid_out(other)) + paged(laid_out(weights))
    assert f"{gib(pinned):.2f}" == "15.81"
    # In int8, by the int8 file's header (weights.DIT_7B_INT8): 8.23 GiB.
    quantised = shot_plan._pinned(plan, peaks.INT8)  # pyright: ignore[reportPrivateUsage]
    int8_block = peaks.block_bytes(peaks.INT8)
    assert quantised == 36 * paged(int8_block) + paged(peaks.DIT_OTHER) + paged(peaks.VAE_WEIGHTS)
    assert f"{gib(quantised):.2f}" == "8.23"


def test_out_of_memory_rows() -> None:
    # The windows that run out of memory on the 96 GB card, 94.50 GiB free, on the lean attention's
    # fit: 1080p at 160 and 161 latents, 4K at 41, the fit's 93.94, 94.42 and 95.27 GiB: over the
    # budget. (numz's table, the plain attention's, ran out at 79 and 80 latents and 20 at 4K:
    # 94.93, 95.92 and 95.79.)
    budget = 94.50 * GIB - peaks.MARGIN
    rows = [
        (OURS_1080, 160, 1_324_800, 93.94),
        (OURS_1080, 161, 1_333_080, 94.42),
        (OURS_4K, 41, 1_348_080, 95.27),
    ]
    for padded, latents, tokens, fit in rows:
        assert peaks.tokens(padded, latents) == tokens
        assert peaks.dit_peak(padded, latents) > budget
        assert abs(gib(peaks.dit_peak(padded, latents)) - fit) <= 0.01
    assert peaks.dit_peak(OURS_1080, 159) <= budget and peaks.dit_peak(OURS_4K, 40) <= budget


def test_seedvr2x_sizes() -> None:
    # DESIGN.md, Memory planner, "By the fit, in step 0's padding": 1,104 rows at 1080p, 2,192
    # at 4K; 6 latents at 1080p take 18.97 GiB (22.22 on the plain attention); on a budget of 93.74
    # GiB, 159 latents fit (93.46) and 160 don't (93.94) (76 and 77 on the plain one's). And on the
    # bare 96 GB card as seedvr2x reads it once a first tensor is allocated, 94.2746 GiB free
    # (measured 2026-10-10), a budget of 93.67: the same.
    assert padded_size((1080, 1920)) == OURS_1080 and padded_size((2160, 3840)) == OURS_4K
    assert peaks.tokens(OURS_1080, 1) == 8_280 and peaks.tokens(OURS_4K, 1) == 32_880
    assert round(gib(peaks.dit_peak(OURS_1080, 6)), 2) == 18.97
    assert round(gib(peaks.dit_peak(OURS_1080, 159)), 2) == 93.46
    assert round(gib(peaks.dit_peak(OURS_1080, 160)), 2) == 93.94
    for free in (94.34, 94.2746):
        budget = free * GIB - peaks.MARGIN
        assert peaks.dit_peak(OURS_1080, 159) <= budget < peaks.dit_peak(OURS_1080, 160)


def test_a_pass_takes_the_tile_that_takes_the_most() -> None:
    # A tiled pass runs its tiles one after the other, and peaks where the one that takes the
    # most does: not always its largest. The grid's last tile along a side is shorter
    # (tilegrid.grid), and by the table itself a shorter tile can take more: at 720p's rows the
    # encode of 752x896 measured 8.09 GiB and that of 752x960 7.70. Frames padded to 752x1792,
    # 2688 or 3584 in tiles of 960 px are laid 752x960 then 752x896: the encode takes the
    # second tile's 8.09, where the largest tile alone gave 7.70; the decode the first's, 12.69
    # over 12.38.
    measured = peaks.MEASURED
    assert (measured[752, 960], measured[752, 896]) == ((7.70, 12.69), (8.09, 12.38))
    for columns in (1792, 2688, 3584):
        assert tile_shapes((752, columns), Tiles(960)) == {(752, 960), (752, 896)}
        assert vae_fits((752, columns), Tiles(960)) == pytest.approx((8.09, 12.69), abs=1e-8)
    assert padded_size((720, 1792)) == (752, 1792)
    # The planner's choice follows: with what the 960-px encode takes by its largest tile alone
    # and no more, it isn't chosen, nor pinned.
    job = ((720, 1792), shots(12))
    least = peaks.encode_peak((752, 1792), Tiles(960), 0, 36) + peaks.VAE_ROOM + MARGIN
    short = least - round(0.39 * GIB)
    assert shot_plan.plan(least, *job, Pins(encode_tile=960), None)[0].encode_tiles == Tiles(960)
    assert refused(short, *job, Pins(encode_tile=960), None).startswith(
        "--encode-tile 960: the encode in tiles of 960 px takes "
    )
    made, predicted = shot_plan.plan(short, *job, Pins(), None)
    assert made.encode_tiles not in (None, Tiles(960))
    assert predicted.need("encode") <= short - MARGIN
    # Over grids of every tile size: a pass takes what its tile that takes the most does, each
    # shape of its grid counted here (most), and never less than a measure of one of its tiles;
    # in some the largest tile isn't that tile, for the encode and for the decode.
    other = [0, 0]
    grids = 0
    for rows in (544, 752, 944, 1104, 1472, 2192):
        for columns in range(768, 3841, 112):
            for size in range(512, 2049, 64):
                tiles = Tiles(size)
                if peaks.covers((rows, columns), tiles):
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
                    one = peaks._tile(size, *first, column)  # pyright: ignore[reportPrivateUsage]
                    other[column] += gib(one) < taken[column] - 1e-8
    assert grids > 2000 and other[0] > 100 and other[1] > 10


def test_vae_phase_peaks() -> None:
    # A VAE phase's peak: its fit, the VAE's weights (0.47 GiB) and what of the DiT stays there;
    # DESIGN.md, BlockSwap, rewritten: an untiled 1080p decode "takes about 35 GiB with the VAE
    # alone", and the DiT's 15.35 GiB beside it pass a 48 GB card's budget (35.45 alone, 35.5
    # to a tenth before the buffers lost the RoPE tables, 2026-11-07).
    alone = peaks.decode_peak(OURS_1080, (1080, 1920), resident=0)
    beside = peaks.decode_peak(OURS_1080, (1080, 1920))
    assert round(gib(alone), 1) == 35.4 and round(gib(beside), 1) == 50.8
    assert beside - alone == 36 * peaks.BLOCK + peaks.DIT_OTHER
    assert f"{gib(beside - alone):.2f}" == "15.35"
    assert beside > (48 - 1.4285) * GIB - peaks.MARGIN > alone
    # The VAE alone whatever is swapped: a phase that keeps nothing of the DiT frees its slots.
    # Nothing moving, every block swapped, the other parameters and the two slots stay: 1.0 GiB.
    assert peaks.decode_peak(OURS_1080, (1080, 1920), resident=0, swapped=36) == alone
    slots = peaks.decode_peak(OURS_1080, (1080, 1920), swapped=36)
    assert slots - alone == 2 * peaks.BLOCK + peaks.DIT_OTHER
    assert f"{gib(slots - alone):.2f}" == "1.00"
    encode = peaks.encode_peak(OURS_1080, Tiles(1024), 4, 20)
    # The encode's holds what seedvr2x's takes over its peak too, 0.4 GiB: here over a square
    # tile's measured 10.47 GiB.
    assert encode == round(10.47 * GIB) + peaks.ENCODE_EXTRA + (
        peaks.VAE_WEIGHTS + peaks.dit_on_gpu(4, 20)
    )


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
            short = shot_plan.plan(free, target, shots(6), tiled, None, split=True)[1]
            long = shot_plan.plan(free, target, shots(6, 76), tiled, None, split=True)[1]
            assert (short.encode, short.decode) == (long.encode, long.decode)
            # Windows of 5 at the most for a shot of 76 latents as for one of 20,000.
            longer = shot_plan.plan(free, target, shots(6, 20_000), tiled, None, split=True)[1]
            assert (longer.encode, longer.dit, longer.decode) == (
                long.encode,
                long.dit,
                long.decode,
            )
            assert longer.window == long.window == 5


# The planner itself. Targets (frames.target_size) and the memory a card has free. SMALL is a
# frame little larger than a tile, padded to 544x512, where the untiled decode takes little more
# than the tiled one (test_floor).
HD, UHD = (1080, 1920), (2160, 3840)


SMALL = (512, 512)


NO_PINS = Pins()


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
    while peaks.dit_peak(padded_size(target), latents + 1, swapped) <= free - MARGIN:
        latents += 1
    return latents


def in_one_window(free: int, target: tuple[int, int], pins: Pins) -> int:
    """The longest shot, in latents, that the planner runs in one window, alone in its job."""
    latents = 0
    while True:
        try:
            _, peaks = shot_plan.plan(free, target, shots(latents + 1), pins, None)
        except PlanError:
            return latents
        if peaks.window != latents + 1:
            return latents
        latents += 1


@pytest.mark.parametrize(
    ("target", "card", "swapped", "design", "ours", "planned"),
    [
        # DESIGN.md, Memory planner, "By the fit, in step 0's padding", refitted on the lean
        # attention (61.243 KiB a token, 0.996 MiB a window; research/docs/small-cards.md): every
        # block on the GPU, at 1080p 12 latents on 24 GB, 28 on 32, 61 on 48 and 159 on 96; at 4K
        # 7 on 32 GB, 15 on 48, 40 on 96. (The plain attention's: 5, 13, 29 and 76 at 1080p; 3,
        # 7 and 19 at 4K.) All 36 blocks swapped: at 1080p 25 on 16 GB, 41 on 24 and 58 on 32; at
        # 4K 6 on 16 GB, 10 on 24, 14 on 32, 22 on 48 and 47 on 96 (plain: 11-12, 19, 27; 2, 4,
        # 6, 10, 22). numz's reading of the free memory and seedvr2x's now give the same lengths
        # at every one of the 15 (the plain attention's 16 GB at 1080p differed by one, 11 and 12).
        (HD, 24, 0, 12, 12, 13),
        (HD, 32, 0, 28, 28, 29),
        (HD, 48, 0, 61, 61, 62),
        (HD, 96, 0, 159, 159, 160),
        (UHD, 32, 0, 7, 7, 7),
        (UHD, 48, 0, 15, 15, 15),
        (UHD, 96, 0, 40, 40, 40),
        (HD, 16, 36, 25, 25, 26),
        (HD, 24, 36, 41, 41, 42),
        (HD, 32, 36, 58, 58, 59),
        (UHD, 16, 36, 6, 6, 6),
        (UHD, 24, 36, 10, 10, 10),
        (UHD, 32, 36, 14, 14, 14),
        (UHD, 48, 36, 22, 22, 23),
        (UHD, 96, 36, 47, 47, 47),
    ],
)
def test_design_windows(
    target: tuple[int, int], card: int, swapped: int, design: int, ours: int, planned: int
) -> None:
    # DESIGN.md's figures are the fit's with the VAE's weights in the DiT's phase, each of the 15
    # found again with the memory numz's process found free (design), and with seedvr2x's own
    # reading (the figure after it). The planner, its blocks pinned alike, sends the VAE's 0.47
    # GiB away when the window wants them (its first choice): at 1080p, where a latent takes about
    # 0.5 GiB, one latent more on every card; at 4K one more on 48 GB, 23.
    pins = Pins(swap_blocks=swapped)
    for free, fit in ((numz_free(card), design), (our_free(card), ours)):
        assert by_the_fit(free, target, swapped) == fit
        assert in_one_window(free, target, pins) == planned
        made, peaks = shot_plan.plan(free, target, shots(planned), pins, None)
        assert (made.window, made.swapped, peaks.window) == (planned, swapped, planned)
        assert made.vae_in_dit == (planned == fit)
        assert peaks.dit <= peaks.budget == free - MARGIN


def test_design_first_reading() -> None:
    # DESIGN.md: 94.34 GiB free on the 96 GB card, a budget of 93.74 GiB, which 159 latents fit
    # (93.46) and 160 don't (93.94) on the lean attention (76 and 77 on the plain one's, 92.98 and
    # 94.07): so by the fit; the planner runs 160, the VAE away (93.47).
    free = round(94.34 * GIB)
    assert by_the_fit(free, HD, 0) == 159
    made, peaks = shot_plan.plan(free, HD, shots(160), Pins(swap_blocks=0), None)
    assert (made.window, made.vae_in_dit, round(peaks.dit / GIB, 2)) == (160, False, 93.47)
    # Nothing pinned, it swaps blocks rather than send the VAE away alone: 161 is beyond both.
    assert in_one_window(free, HD, Pins(swap_blocks=0)) == 160


def test_window_is_the_first_choice() -> None:
    # On a 24 GB card at 1080p, a shot of 18 latents runs in one window: the fewest blocks that
    # hold it are swapped, 8, the VAE's weights leaving before a 9th is. (12 latents and 16 blocks
    # on the plain attention's figures: the lean attention holds 12 whole, the VAE on the GPU; 18
    # is the first length whose DiT phase, not the encode, bounds the plan.)
    free, padded = our_free(24), OURS_1080
    made, predicted = shot_plan.plan(free, HD, shots(18), Pins(), None)
    assert (made.window, made.swapped, made.vae_in_dit) == (18, 8, False)
    budget = free - MARGIN
    assert peaks.dit_peak(padded, 18, 7, vae_in_dit=False) > budget
    assert peaks.dit_peak(padded, 18, 8) > budget >= peaks.dit_peak(padded, 18, 8, False)
    assert predicted.dit == peaks.dit_peak(padded, 18, 8, False) and predicted.bound == "dit"
    # One and two swapped blocks save nothing, so are never chosen: no block, then three.
    for latents in range(1, 30):
        assert shot_plan.plan(free, HD, shots(latents), Pins(), None)[0].swapped not in (1, 2)
    # The VAE stays where the window fits with it: 28 latents on a 32 GB card, no block swapped;
    # it leaves for 29, before a block is swapped (13 and 14 on the plain attention's figures).
    made, _ = shot_plan.plan(our_free(32), HD, shots(28), Pins(), None)
    assert (made.window, made.swapped, made.vae_in_dit) == (28, 0, True)
    made, _ = shot_plan.plan(our_free(32), HD, shots(29), Pins(), None)
    assert (made.window, made.swapped, made.vae_in_dit) == (29, 0, False)
    made, _ = shot_plan.plan(our_free(32), HD, shots(31), Pins(), None)
    assert (made.window, made.swapped, made.vae_in_dit) == (31, 4, False)
    assert peaks.dit_peak(padded, 31, 3, vae_in_dit=False) > our_free(32) - MARGIN


def test_large_card_swaps_before_it_shortens() -> None:
    # The 96 GB card swaps blocks on a shot longer than its 160 latents before it shortens the
    # window (DESIGN.md, Memory planner, Choices, in order): 164 latents in one window, 7 blocks
    # swapped (80 and 9 on the plain attention's figures); and nothing caps it, a shot it holds
    # whole staying untiled, every weight there.
    free = our_free(96)
    made, predicted = shot_plan.plan(free, HD, shots(164), Pins(), None)
    assert (made.window, predicted.window, made.swapped, made.vae_in_dit) == (164, 164, 7, False)
    assert peaks.dit_peak(OURS_1080, 164, 6, vae_in_dit=False) > free - MARGIN
    made, predicted = shot_plan.plan(free, HD, shots(159), Pins(), None)
    assert made == Plan(159) and predicted.bound == "dit"
    # A longer shot than every block swapped holds, 189 latents: two windows of 96, the plan's
    # cap, with no block swapped (91 and 47 on the plain attention's figures).
    made, predicted = shot_plan.plan(free, HD, shots(190), Pins(), None)
    assert (made.window, predicted.window, made.swapped) == (96, 96, 0)
    assert in_one_window(free, HD, Pins()) == 189


def test_own_cap_lays_the_shots_out_alike() -> None:
    # The planner's own cap for a job that fits under a cap, the longest window the job really
    # runs under it (shot_plan._own_cap), lays every shot out as that cap does (shots.shot_layout),
    # and so runs the same longest window: for shots of many lengths, alone and together, under
    # every cap from the smallest a job can run to beyond its longest shot. 5 latents at least
    # when a shot is split, whose windows can be 4 long: no layout exists under that.
    own_cap = shot_plan._own_cap  # pyright: ignore[reportPrivateUsage]
    longest = shot_plan._longest  # pyright: ignore[reportPrivateUsage]
    job = shot_plan._job  # pyright: ignore[reportPrivateUsage]
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
        counted = job(GIB, HD, film, False)
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
    counted = job(GIB, HD, shots(6, 6, 6), False)
    assert (longest(counted.lengths, 5), own_cap(counted, 5)) == (4, 5)
    assert shot_layout(21, 5) == [(0, 4), (2, 6)]
    with pytest.raises(ValueError, match="windows of 4 latents leave none of their own"):
        shot_layout(21, 4)


def test_longest_window_by_the_layouts() -> None:
    # The longest window shots really run under a cap is their layouts' (shots.shot_layout), and
    # never falls as the cap rises, which the planner's search of the cap stands on.
    longest = shot_plan._longest  # pyright: ignore[reportPrivateUsage]
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


def test_vae_choices() -> None:
    # Untiled where it fits, the DiT's blocks leaving the phase before a tile shrinks; else the
    # largest tile that fits with them away, then as many of them left there as fit.
    # 96 GB at 1080p: both untiled, every block there.
    made, _ = shot_plan.plan(our_free(96), HD, shots(60), Pins(), None)
    assert (made.encode_tiles, made.decode_tiles) == (None, None)
    assert (made.resident_encode, made.resident_decode) == (None, None)
    # 48 GB at 1080p: the untiled decode, about 35 GiB with the VAE alone, fits with 24 of the
    # DiT's blocks beside it and not 25 (DESIGN.md, BlockSwap, rewritten, Per phase) by its peak;
    # with the room every VAE pass is held to, 1.25 GiB under the budget, 21 stay, by 261.1 MiB
    # (258.6 before the buffers lost the RoPE tables, 2026-11-07), whatever the shot's length.
    free, room = our_free(48), peaks.VAE_ROOM
    budget = free - MARGIN
    made, predicted = shot_plan.plan(free, HD, shots(21), Pins(), None)
    assert made == Plan(21, resident_decode=21)
    untiled = [peaks.decode_peak(OURS_1080, HD, None, blocks) for blocks in (21, 22, 24, 25)]
    assert untiled[3] > budget >= untiled[2]
    assert untiled[1] + room > budget >= untiled[0] + room
    assert round((budget - untiled[0] - room) / 1024**2, 1) == 261.1
    assert (
        predicted.decode == untiled[0] and predicted.bound == "decode"
    )  # the peak, without the room
    # 32 GB at 1080p: the decode in the largest tiles that fit with the DiT away, 1536 px, cut
    # off at the picture's edges and measured so, 27.30 GiB under the one-output convolution
    # (27.94 before it, with none of its blocks left beside them), one of its blocks left beside
    # them, by 327 MiB (71 with 1.5 GiB of room); the encode untiled, 18 left (17).
    free = our_free(32)
    budget = free - MARGIN
    made, predicted = shot_plan.plan(free, HD, shots(12), Pins(), None)
    assert bare(made) == Plan(12, None, Tiles(1536), 0, 18, 1)
    assert peaks.decode_peak(OURS_1080, HD, Tiles(1600), 0) + room > budget
    more = peaks.decode_peak(OURS_1080, HD, Tiles(1536), 2)
    assert more + room > budget >= predicted.decode + room
    assert round((budget - predicted.decode - room) / 1024**2) == 330  # 327 with the tables
    more = peaks.encode_peak(OURS_1080, None, 19)
    assert more + room > budget >= predicted.encode + room
    # 24 GB at 1080p: the untiled encode fits with the DiT away by its peak, and not with its
    # room: tiled, in the largest tiles under the frame's longer side.
    # (Shots of 25 latents, the 16 blocks swapped as the plain attention's 12 latents had them.)
    made, predicted = shot_plan.plan(our_free(24), HD, shots(25), Pins(), None)
    assert bare(made) == Plan(25, Tiles(1856), Tiles(1088), 16, 1, 0, vae_in_dit=False)
    untiled_encode = peaks.encode_peak(OURS_1080, None, 0, 16)
    assert untiled_encode <= our_free(24) - MARGIN < untiled_encode + room
    # split's slice is counted with an untiled decode's peak, which leaves fewer of the DiT's
    # blocks beside it, 19 on 48 GB for 21; a tiled decode's peak doesn't hold it, the same plan
    # with it.
    made, _ = shot_plan.plan(our_free(48), HD, shots(21), Pins(), None, split=True)
    assert made == Plan(21, resident_decode=19)
    made, _ = shot_plan.plan(free, HD, shots(12), Pins(), None, split=True)
    assert bare(made) == Plan(12, None, Tiles(1536), 0, 18, 1)
    # 24 GB at 4K: both tiled, the decode's smaller, its peak about twice the encode's by the
    # megapixel; every block swapped. The encode is in tiles of 1472 px, the DiT away, by 58 MiB,
    # where the next tiles up don't fit with it away (with 1.5 GiB of room, tiles of 1408 px, and
    # the DiT's two slots and its other parameters, 1.0 GiB, beside them, by 90 MiB); the DiT
    # leaves the decode, and no tile shrinks for it.
    # (Every block swapped is pinned: the plain attention's plan swapped them all for 5 latents,
    # the lean one's unpinned plan at 24 GB swaps 29 for 9 and 34 for 10, the encode's and the
    # decode's tiles being the same.)
    made, predicted = shot_plan.plan(our_free(24), UHD, shots(5), Pins(swap_blocks=36), None)
    assert bare(made) == Plan(5, Tiles(1472), Tiles(1088), 36, 0, 0)  # the VAE stays now
    grid = peaks.grid_shapes  # pyright: ignore[reportPrivateUsage]
    assert (grid(OURS_4K, Tiles(1472)), grid(OURS_4K, Tiles(1088))) == ((2, 3), (3, 4))
    assert predicted.scratch == peaks.scratch(OURS_4K, 17) and predicted.tokens == 32_880
    budget = our_free(24) - MARGIN
    away = peaks.encode_peak(OURS_4K, Tiles(1472), 0, 36)
    assert round((budget - away - room) / 1024**2) == 60  # 58 with the tables
    assert peaks.encode_peak(OURS_4K, Tiles(1472), None, 36) + room > budget
    assert peaks.encode_peak(OURS_4K, Tiles(1536), 0, 36) + room > budget
    assert peaks.decode_peak(OURS_4K, UHD, Tiles(1088), None, 36) + room > budget
    assert peaks.decode_peak(OURS_4K, UHD, Tiles(1152), 0, 36) + room > budget


def test_a_latent_is_counted_by_its_tensor() -> None:
    # What the planner counts a shot's latent and its noise by in system memory
    # (shot_plan.host_moments; LATENT_CHANNELS, LATENT_SIDE, LATENT_VALUE), held to the VAE's
    # config and to tensors of that shape built here: (latents, H / 8, W / 8, 16) in bfloat16,
    # half a byte a padded pixel and a latent, 1,059,840 bytes a latent at 1080p and 4,208,640 at
    # 4K. The noise is drawn in the latent's shape and dtype (sampling.drawn), so it counts the
    # same.
    import torch

    import seedvr2x.vendor
    from seedvr2x.model import COMPUTE_DTYPE
    from seedvr2x.vendor.common.config import load_config

    vendor = Path(seedvr2x.vendor.__file__).parent
    vae: Any = load_config(str(vendor / "models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml"))
    assert (vae.latent_channels, vae.spatial_downsample_factor) == (16, 8)
    assert (shot_plan.LATENT_CHANNELS, shot_plan.LATENT_SIDE) == (16, 8)
    assert shot_plan.LATENT_VALUE == torch.empty(0, dtype=COMPUTE_DTYPE).element_size() == 2
    for padded, bytes_a_latent in ((OURS_1080, 1_059_840), (OURS_4K, 4_208_640)):
        height, width = padded
        assert bytes_a_latent == height * width // 2
        for latents in (1, 5, 151):
            sizes = (latents, height // 8, width // 8, 16)
            latent = torch.empty(sizes, dtype=COMPUTE_DTYPE, device="meta")
            assert latent.numel() * latent.element_size() == bytes_a_latent * latents
            assert (
                shot_plan.host_moments(padded, latents, None)["decode"] == bytes_a_latent * latents
            )


def test_no_shot_is_too_long_for_the_dit_s_phase() -> None:
    # The DiT's phase is judged by its longest window alone, a shot's noise drawn in system
    # memory (shot_pass.sample_windows; DESIGN.md, Pipeline step 2): however long the shot, the
    # window decides. 1080p on a 24 GB card, --window 11, whose peak, 21.42 GiB with every weight
    # there, leaves 0.55 under the budget of 21.97 (--window 5, 21.13 GiB and 0.84 under, on the
    # plain attention; the lean one holds 11 latents where it held 5): every weight stays there
    # for a shot of 20 latents as for one of 20,000 (79,997 frames). Drawn on the GPU beside the
    # shot's latent, 2 MiB a latent at 1080p, the noise took the VAE's weights away from a shot of
    # 3,090 latents, and a third block from one of 3,327 (until 2026-10-16).
    free, pins = our_free(24), Pins(window=11)
    budget = free - MARGIN
    window = peaks.dit_peak(OURS_1080, 11)
    assert (round(gib(window), 2), round(gib(budget - window), 2)) == (21.42, 0.55)
    for latents in (20, 3090, 3327, 4000, 20_000):
        made, predicted = shot_plan.plan(free, HD, shots(latents), pins, None)
        assert (made.window, predicted.window, made.swapped, made.vae_in_dit) == (11, 11, 0, True)
        assert predicted.dit == predicted.record()["dit_peak"] == window
        assert fits(made, free, HD, shots(latents), None) is None
    # A plan from elsewhere with no block swapped fits a shot of 4,000 latents, and --swap-blocks
    # 0 is taken there, where the draw refused it.
    short = Plan(5, Tiles(512), Tiles(512), 0, 0, 0)
    assert fits(short, free, HD, shots(4000), None) is None
    made, _ = shot_plan.plan(free, HD, shots(4000), Pins(swap_blocks=0), None)
    assert (made.window, made.swapped, made.vae_in_dit) == (13, 0, False)
    # A --window refused is refused for its window.
    assert refused(free, HD, shots(3000), Pins(window=60), None).startswith(
        "--window 60: its longest window, 60 latents, takes "
    )


def test_a_long_shot_at_4k_plans_on_a_24_gb_card() -> None:
    # 4K with 11.7 GiB free (22.51 GiB, a card emulated at 24 GB, on the plain attention: the
    # lean one's 4K floor is 11.64 GiB, so the same boundary sits at 11.7): windows of 5 latents,
    # every block swapped and the VAE away, for shots of any length, the DiT's phase bounding
    # the plan (test_a_window_is_all_the_phase_holds_of_a_shot). The draw on the GPU refused a shot
    # of 2,661 latents there, 10,641 frames, naming a cut (until 2026-10-16).
    free = round(11.7 * GIB)
    expected = Plan(5, Tiles(896), Tiles(640), 36, 0, None, vae_in_dit=False)
    for film in (shots(151), shots(2661), shots(3000, 40), shots(40, 10_000, 5)):
        made, predicted = shot_plan.plan(free, UHD, film, Pins(), None)
        assert bare(made) == expected
        assert predicted.dit == peaks.dit_peak(OURS_4K, 5, 36, vae_in_dit=False)
        assert predicted.bound == "dit"
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
        made, predicted = shot_plan.plan(our_free(card), target, shots(latents), pins, None)
        assert fits(made, our_free(card), target, shots(latents), None) is None
        assert (made.window, predicted.window) == (5, 5)
        assert predicted.dit == peaks.dit_peak(
            padded_size(target), 5, made.swapped, made.vae_in_dit
        )
        return bare(made)

    for card, target, plan in (
        (24, HD, Plan(5, Tiles(1856), Tiles(1088), 0, 1, 0)),
        (32, HD, Plan(5, None, Tiles(1536), 0, 18, 1)),
        (48, HD, Plan(5, resident_decode=21)),
        # (11 blocks swapped on the lean attention, 36 on the plain one's.)
        (24, UHD, Plan(5, Tiles(1472), Tiles(1088), 11, 0, 0, vae_in_dit=False)),
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


# Takes the plain attention's fit: these runs measured the plain attention; the lean fit is held to
# its own measures (LEAN_WINDOWS, tests/test_peaks.py).
@pytest.mark.usefixtures("plain_fit")
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
    # numz's padding, zeros up to multiples of 16, which these recipes were measured in: the
    # planner's own padding (frames.padded_size) put back for this test.
    monkeypatch.setattr(shot_plan, "padded_size", lambda size: (-(-size[0] // 16) * 16, size[1]))
    encode, decode = (None if size is None else Tiles(size) for size in tiles)
    recipe = Plan(latents, encode, decode, swapped, 0, 0)
    # The DiT's peak: the fit's, within 0.01 GiB of the measured one with no block swapped.
    # Swapped, the model keeps two slots on the GPU where numz's swap keeps one block there at a
    # time (DESIGN.md, BlockSwap, rewritten): one block's 0.42 GiB over numz's, within 0.03.
    dit = peaks.dit_peak(padded, latents, swapped) / GIB
    assert abs(dit - (BLOCK / GIB if swapped else 0) - measured[1]) <= (0.03 if swapped else 0.01)
    # The VAE's peaks, with what numz has on the GPU then, the VAE alone: the decode within 0.9
    # GiB of the measured one (vram.md: "every measured DiT and decode peak is within 0.9 GiB of
    # the formulas"; the furthest 0.82, the untiled 1080p one); the encode under it, numz
    # holding its batch's frames on the GPU, by 0.13 to 0.99 GiB at 1080p and 0.34 to 1.03 at 4K
    # with the 0.4 GiB seedvr2x's own encode holds over the fit (peaks.ENCODE_EXTRA; vram.md
    # says up to 0.9 at 1080p and 1.4 to 2.2 at 4K, over its own fits). A square tile's peak is
    # the one measured on seedvr2x (peaks.MEASURED); one cut off at numz's 1,088 rows, which
    # nobody measured, takes its rule and 0.3 GiB. numz's passes don't have the one-output
    # convolution: they are held to the table of 2026-10-11 (BEFORE_ONE_OUTPUT), the one the
    # planner then had.
    alone = peaks.dit_on_gpu(0, swapped)
    with monkeypatch.context() as before:
        before.setattr(peaks, "MEASURED", peaks.MEASURED | BEFORE_ONE_OUTPUT)
        fitted_encode = (peaks.encode_peak(padded, encode, 0, swapped) - alone) / GIB
        fitted_decode = (peaks.decode_peak(padded, target, decode, 0, swapped) - alone) / GIB
    assert abs(fitted_decode - measured[2]) <= 0.9
    assert 0.1 <= measured[0] - fitted_encode <= 1.1
    # It fits by the model, but for the 4K one on 32 GB: its 7 latents with every block swapped
    # take 30.09 GiB with the model's two slots, 0.14 over the budget, where numz measured
    # 29.65; it fits with the VAE's weights away, as the planner has them there.
    said = fits(recipe, free, target, shot, None)
    if (target, card) == (UHD, 32):
        assert said == (
            "the DiT's phase, a window of 7 latents with 36 blocks swapped, takes 30.70 GiB of"
            " free VRAM and the driver reports 30.56 GiB free, 0.14 GiB short"
        )
        recipe = replace(recipe, vae_in_dit=False)
        said = fits(recipe, free, target, shot, None)
    assert said is None
    # The planner's own plan for that shot on that card is at least as good: the window whole,
    # no more blocks swapped, no smaller tiles, each VAE pass with its room (peaks.VAE_ROOM),
    # which the recipes have too, the DiT away. On 24 GB at 1080p its encode is untiled (in tiles
    # of 1856 px with 1.5 GiB of room, the untiled one lacking that room), and on 32 GB untiled
    # with 19 of the DiT's blocks (18); at 4K on 48 GB its tiles stop at 2048 px
    # (peaks.TILE_MAX).
    made, predicted = shot_plan.plan(free, target, shot, Pins(), None)
    assert (made.window, predicted.window) == (latents, latents)
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


# Takes the plain attention's fit: these runs measured the plain attention; the lean fit is held to
# its own measures (LEAN_WINDOWS, tests/test_peaks.py).
@pytest.mark.usefixtures("plain_fit")
@pytest.mark.parametrize(("card", "latents", "measured"), WINDOWED)
def test_a_shot_s_windows_peak_as_one_window_shots(
    card: int, latents: int, measured: dict[int, float]
) -> None:
    # A shot of several windows peaks, window by window, where a one-window shot of each
    # window's length is predicted to (peaks.dit_peak): nothing of the shot is counted beside
    # a window, nor was there. The card's free memory as a run read it there, 1,600,978,944
    # bytes under N GiB (cli._free_memory: N - 1.49); the plan is the one that ran, every block
    # swapped and the VAE away, its cap the longest window.
    free = card * GIB - 1_600_978_944
    made, forecast = shot_plan.plan(free, HD, shots(latents), Pins(), None, split=True)
    assert (made.window, made.swapped, made.vae_in_dit) == (max(measured), 36, False)
    windows = {end - start for start, end in shot_layout(4 * latents - 3, made.window)}
    assert windows == set(measured)
    for length, peak in measured.items():
        predicted = peaks.dit_peak(OURS_1080, length, 36, vae_in_dit=False)
        assert abs(gib(predicted) - peak) <= 0.011, (length, gib(predicted), peak)
    # The plan's own figure for the phase is its longest window's.
    assert forecast.dit == peaks.dit_peak(OURS_1080, max(measured), 36, vae_in_dit=False)


# Takes the plain attention's fit: these runs measured the plain attention; the lean fit is held to
# its own measures (LEAN_WINDOWS, tests/test_peaks.py).
@pytest.mark.usefixtures("plain_fit")
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
    # (peaks.ENCODE_EXTRA) and split's term off a tiled decode's peak (peaks.decode_peak),
    # both PROVISIONAL (implementation, 2026-10-11); no tile chosen above 2048 px
    # (peaks.TILE_MAX) and every VAE pass held to a room under the budget besides its peak
    # (peaks.VAE_ROOM; 1.5 GiB when the matrices were drawn on, 1.25 since 2026-10-18),
    # DESIGN.md's (Memory planner: What follows from them; Budget): any VAE pass, encode or
    # decode, tiled or not, ran out of memory with its peak close to the free memory. Each case
    # planned again with the memory it read; the DiT's phase is the one that ran everywhere, and
    # every VAE pass leaves 1.85 GiB or more under the free memory (2.1 with 1.5). With every
    # tile's measured peak (peaks.MEASURED), no pass counted by its shot's length, the plans of
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
    made, predicted = shot_plan.plan(reported, target, shot, Pins(), None, split=True)
    assert bare(made) == own[target, free]
    assert fits(made, reported, target, shot, None, split=True) is None
    for phase in ("encode", "decode"):
        assert reported - predicted.peak(phase) >= MARGIN + peaks.VAE_ROOM
    for tiles in (made.encode_tiles, made.decode_tiles):
        assert tiles is None or tiles.size <= peaks.TILE_MAX
    # The DiT's phase is the one that ran: its window, its swapped blocks, the VAE away or not.
    assert (made.window, made.swapped, made.vae_in_dit) == (ran.window, ran.swapped, ran.vae_in_dit)
    # The DiT's phase by the model: its longest window's peak.
    # Of the DiT's peaks these matrices measured (2026-10-11), one is of a shot in one window, 5
    # latents at 4K, and is held to the model below. The others are of shots in two windows,
    # measured while a run kept its shot's whole latent and noise on the GPU through them,
    # 0.020 to 0.043 GiB over their window's own: figures of a run that held more than a
    # window, which the model doesn't predict.
    window = max(end - start for start, end in shot_layout(frames, ran.window))
    alone = peaks.dit_peak(padded, window, ran.swapped, ran.vae_in_dit)
    assert predicted.dit == alone

    def vae_peaks(plan: Plan) -> tuple[float, float]:
        """A plan's encode and decode peaks by the corrected model, in GiB, with the table of
        2026-10-11 (BEFORE_ONE_OUTPUT): the runs it is held to ran without the one-output
        convolution."""
        there = (plan.resident_encode, plan.swapped), (plan.resident_decode, plan.swapped)
        with monkeypatch.context() as before:
            before.setattr(peaks, "MEASURED", peaks.MEASURED | BEFORE_ONE_OUTPUT)
            encode = peaks.encode_peak(padded, plan.encode_tiles, *there[0])
            decode = peaks.decode_peak(padded, target, plan.decode_tiles, *there[1], split=True)
        return gib(encode), gib(decode)

    # What a decode of those runs kept of its shot on the GPU, which no pass keeps since, in
    # GiB: the shot's latent and a copy of it, 0.954 MiB a megapixel of the padded frame and a
    # latent as measured then (2.02 MiB a latent at 1080p, 8.03 at 4K). Their measured peaks
    # held it, where the model no longer counts it.
    kept_then = 0.954 * padded[0] * padded[1] / 1e6 * latents / 1024
    # And the RoPE tables of its windows, which no VAE phase holds since they are cleared where
    # the DiT's work ends (2026-11-07): the DiT's buffers were counted 0.058 GiB then, a 1-latent
    # window's tables in it, 2.6 MB over the census's DIT_BUFFERS; the run's own windows' tables
    # were at least that.
    kept_then += 0.058 - gib(peaks.DIT_BUFFERS)

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
    # encode was taken to hold over its fit (peaks.ENCODE_EXTRA, PROVISIONAL), the measured
    # tiles no longer leave it short of.
    encode, decode = vae_peaks(ran)
    assert 0.39 <= encode - measured[0] <= 0.41
    assert -0.005 <= decode + kept_then - measured[2] <= 0.02
    if (target, free) == (HD, 10.54):
        assert (round(decode, 2), round(kept_then, 2)) == (9.19, 0.03)
    if (target, free) == (UHD, 22.54):
        assert (round(decode, 2), round(kept_then, 2)) == (18.98, 0.04)
    # The DiT's peak of the one shot that ran in one window: 21.78 GiB measured for 21.791,
    # the window's peak by the model (numz's fit; peaks.DIT_LOADED), 0.011 under.
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
) -> str | None:
    """Why a plan given, the planner's own or one made here by hand, doesn't fit a job's memory:
    the sentence the planner says of every plan it makes, by the two functions shot_plan.plan
    judges with (shot_plan._job, shot_plan._unfit); None when it fits. The tests' own: no run
    judges a plan handed to it, and the planner keeps no public function for it."""
    job = shot_plan._job(free, target, film, split)  # pyright: ignore[reportPrivateUsage]
    return shot_plan._unfit(job, plan, host_available)  # pyright: ignore[reportPrivateUsage]


def refused(*job: object, **options: object) -> str:
    # The plain plan's refusal, without the last rungs (plan's `rungs`): what these tests hold is
    # the planner's own choices and refusals, which a card the rungs now run still gives, no rung
    # taken; the rungs have their own tests (test_a_clip_that_fits_no_other_way_*).
    options.setdefault("rungs", False)
    with pytest.raises(PlanError) as error:
        shot_plan.plan(*job, **options)  # pyright: ignore[reportArgumentType]
    return cli_words(error.value)


def test_window_pin() -> None:
    # --window is the cap exactly, or refused, the refusal naming the longest window that fits,
    # the planner's own cap: with every block swapped, or with the blocks pinned.
    free = our_free(24)
    made, peaks = shot_plan.plan(free, HD, shots(76), Pins(window=20), None)
    assert (made.window, peaks.window) == (20, 17)
    made, peaks = shot_plan.plan(free, HD, shots(76), Pins(window=8), None)
    assert (made.window, peaks.window, made.swapped) == (8, 8, 0)  # 7 swapped before the refit
    # (Lean attention: 30 latents would plan now, the 24 GB card holding 41 with every block
    # swapped; the refusal takes a pin of the whole shot, 76, over the 39 that fit.)
    assert refused(free, HD, shots(76), Pins(window=76), None) == (
        "--window 76: its longest window, 76 latents, takes 38.79 GiB of free VRAM with every"
        " block of the DiT swapped, and the driver reports 22.57 GiB free: the longest window"
        " that fits is --window 39"
    )
    assert refused(free, HD, shots(76), Pins(window=20, swap_blocks=0), None) == (
        "--window 20: its longest window, 17 latents, takes 24.43 GiB of free VRAM with"
        " --swap-blocks 0, and the driver reports 22.57 GiB free: the longest window that fits is"
        " --window 13"
    )
    made, _ = shot_plan.plan(free, HD, shots(76), Pins(window=13, swap_blocks=0), None)
    assert (made.window, made.swapped, made.vae_in_dit) == (13, 0, False)


def test_swap_pin() -> None:
    # --swap-blocks is the count exactly, the cap then the longest window that fits with it;
    # refused when no window fits with it, the refusal naming the fewest blocks that hold the
    # window; and above the DiT's 36.
    free = our_free(24)
    made, peaks = shot_plan.plan(free, HD, shots(76), Pins(swap_blocks=36), None)
    assert (made.window, peaks.window, made.swapped, made.vae_in_dit) == (
        39,
        39,
        36,
        True,
    )  # 17 on the plain attention
    made, peaks = shot_plan.plan(free, HD, shots(76), Pins(swap_blocks=0), None)
    assert (made.window, peaks.window, made.swapped, made.vae_in_dit) == (13, 13, 0, False)  # 6
    # One and two swapped blocks save nothing, and are taken as pinned.
    made, _ = shot_plan.plan(free, HD, shots(76), Pins(swap_blocks=2), None)
    assert (made.window, made.swapped) == (13, 2)  # 6 on the plain attention
    # (The 24 GB card at 4K, where the 32 GB one was: on the lean attention 5 latents with no
    # block swapped take 25.98 GiB, which 32 GB holds; 36.74 GiB and 17 blocks before the refit.)
    free = our_free(24)
    assert refused(free, UHD, shots(40), Pins(swap_blocks=0), None) == (
        "--swap-blocks 0: a window of 5 latents, the shortest this job can run, takes 25.98 GiB of"
        " free VRAM with 0 blocks swapped, and the driver reports 22.57 GiB free: the fewest"
        " blocks that hold it are --swap-blocks 11"
    )
    made, _ = shot_plan.plan(free, UHD, shots(40), Pins(swap_blocks=11), None)
    assert (made.window, made.swapped, made.vae_in_dit) == (5, 11, False)
    assert refused(free, UHD, shots(40), Pins(window=6, swap_blocks=0), None) == (
        "--swap-blocks 0: a window of 6 latents, --window 6's longest, takes 28.00 GiB of free"
        " VRAM with 0 blocks swapped, and the driver reports 22.57 GiB free: the fewest blocks"
        " that hold it are --swap-blocks 15"
    )
    assert refused(free, UHD, shots(40), Pins(swap_blocks=37), None) == (
        "--swap-blocks 37: the DiT has 36 blocks, so from 0 to 36"
    )


def test_floor() -> None:
    # A size that doesn't fit is refused, never degraded, the refusal counting in free VRAM:
    # what the size takes, what the driver reports free, and the largest output that fits there
    # (DESIGN.md, Memory planner). At 4K the smallest window, 5 latents, takes 11.64 GiB of free
    # VRAM by the model on the lean attention, every block swapped and the VAE away: a peak of
    # 11.04 GiB and the margin (11.64 GiB, a peak of 21.80, on the plain attention's fit that
    # DESIGN.md's "At 4K the smallest window takes 11.64 GiB of free VRAM" gave). Whatever the
    # shot's length: nothing of a longer shot is on the GPU beside a window, its noise drawn in
    # system memory. The cards below are the 12 GB that is short of it, and the 14 GB that holds
    # it (the 16 GB and the 24 GB of the plain fit).
    assert refused(numz_free(12), UHD, shots(5), Pins(), None).startswith(
        "an output of 3840x2160 takes 11.64 GiB of free VRAM (its smallest window, 5 latents, with"
        " every block of the DiT swapped and the VAE off the GPU: a peak of 11.04 GiB, and 0.60 of"
        " margin), and the driver reports 10.56 GiB free: the largest output that fits there is"
    )
    assert refused(numz_free(12), UHD, shots(40), Pins(), None) == (
        "an output of 3840x2160 takes 11.64 GiB of free VRAM (its smallest window, 5 latents, with"
        " every block of the DiT swapped and the VAE off the GPU: a peak of 11.04 GiB, and 0.60 of"
        " margin), and the driver reports 10.56 GiB free: the largest output that fits there is"
        " --resolution 2016, about 3584x2016"
    )
    # The pins change nothing to it.
    pins = Pins(window=5, swap_blocks=36, decode_tile=512)
    assert refused(numz_free(12), UHD, shots(40), pins, None).startswith("an output of 3840x2160")
    # The same floor for a shot of 5 latents, of 40 and of 1,000, 3,997 frames: 11.64 GiB of
    # free VRAM plans, in windows of 5 latents, which bound the plan, and 0.01 GiB less is
    # refused, the next size down fitting.
    for latents in (5, 40, 1000):
        made, predicted = shot_plan.plan(round(11.64 * GIB), UHD, shots(latents), Pins(), None)
        assert (made.window, made.swapped, made.vae_in_dit) == (5, 36, False)
        assert predicted.dit == peaks.dit_peak(OURS_4K, 5, 36, vae_in_dit=False)
        assert round(predicted.dit / GIB, 2) == 11.03 and predicted.bound == "dit"
        assert refused(round(11.63 * GIB), UHD, shots(latents), Pins(), None) == (
            "an output of 3840x2160 takes 11.64 GiB of free VRAM (its smallest window, 5 latents,"
            " with every block of the DiT swapped and the VAE off the GPU: a peak of 11.04 GiB,"
            " and 0.60 of margin), and the driver reports 11.63 GiB free: the largest output that"
            " fits there is --resolution 2144, about 3810x2144"
        )
    made, _ = shot_plan.plan(round(11.63 * GIB), (2144, 3811), shots(1000), Pins(), None)
    assert made.window == 5
    # An emulated 14 GB card has that, by 0.92 GiB (12.56 GiB free; 12.57 by seedvr2x's own
    # reading): 4K runs there in windows of 5 latents, a shot of 22 latents, 85 frames, as one
    # of 40 or of 1,000, in tiles of 896 and 704 px, 34 blocks swapped, the DiT away.
    for free in (numz_free(14), our_free(14)):
        for latents in (22, 40, 1000):
            made, predicted = shot_plan.plan(free, UHD, shots(latents), Pins(), None)
            assert bare(made) == Plan(5, Tiles(896), Tiles(704), 34, 2, 0, vae_in_dit=False)
            assert predicted.dit == peaks.dit_peak(OURS_4K, 5, 34, vae_in_dit=False)
    # A job whose shots are all shorter runs its longest shot's length: 3 latents on 12 GB, in
    # the largest tiles that then fit, the DiT's slots beside the encode's and the decode's.
    made, _ = shot_plan.plan(numz_free(12), UHD, shots(3, 2), Pins(), None)
    assert bare(made) == Plan(3, Tiles(768), Tiles(640), 30, 1, 0, vae_in_dit=False)
    # A portrait target is scaled by its short side too, the output said width x height.
    assert refused(numz_free(12), (3840, 2160), shots(40), Pins(), None).endswith(
        "the largest output that fits there is --resolution 2048, about 2048x3640"
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
    made, _ = shot_plan.plan(round(8.6 * GIB), HD, shots(40), Pins(), None)
    # (Windows of 12 latents now, 6 on the plain attention; blocks swapped 34 for 35 there.)
    assert bare(made) == Plan(12, Tiles(640), Tiles(512), 34, 1, 0, vae_in_dit=False)
    made, _ = shot_plan.plan(round(8.52 * GIB), HD, shots(40), Pins(), None)
    assert bare(made) == Plan(12, Tiles(640), Tiles(512), 35, 1, 0, vae_in_dit=False)
    # (8.265: 8.27 GiB keeps a block in the encode since the buffers lost the RoPE tables, 2.6 MB.)
    made, _ = shot_plan.plan(round(8.265 * GIB), HD, shots(40), Pins(), None)
    assert bare(made) == Plan(12, Tiles(640), Tiles(512), 35, 0, 0, vae_in_dit=False)
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
    # at 540p. At 4K, where the DiT's smallest window is the floor, 11.64 (22.40 plain) for a
    # shot of 5 latents as for one of 1,000 or 20,000.
    for target, latents, said in (
        (HD, 6, "an output of 1920x1080 takes 8.27 GiB of free VRAM (its decode in tiles"),
        (HD, 1000, "an output of 1920x1080 takes 8.27 GiB of free VRAM (its decode in tiles"),
        ((540, 960), 1000, "an output of 960x540 takes 8.27 GiB of free VRAM (its decode in"),
        (UHD, 5, "an output of 3840x2160 takes 11.64 GiB of free VRAM (its smallest window"),
        (UHD, 1000, "an output of 3840x2160 takes 11.64 GiB of free VRAM (its smallest window"),
        (HD, 20_000, "an output of 1920x1080 takes 8.27 GiB of free VRAM (its decode in tiles"),
        (UHD, 20_000, "an output of 3840x2160 takes 11.64 GiB of free VRAM (its smallest window"),
    ):
        assert refused(4 * GIB, target, shots(latents), Pins(), None).startswith(said)
    # The 8.27 GiB hold to a shot of 20,000 latents, 79,997 frames, where the noise drawn on the
    # GPU beside the shot's latent made the floor from a shot of 3,477 latents (until
    # 2026-10-16).
    for latents in (3476, 3477, 20_000):
        made, _ = shot_plan.plan(round(8.27 * GIB), HD, shots(latents), Pins(), None)
        # (13 latents, every block swapped: the shots of this length split into windows of 13.)
        assert bare(made) == Plan(13, Tiles(640), Tiles(512), 36, 0, 0, vae_in_dit=False)
    # The same pass bounds a smaller frame: 512x512, padded to 544x512, from 8.52 GiB free. Its
    # untiled decode takes more, 6.47 GiB against the tile's 6.41, and is no floor: pinned, it
    # is refused under 8.32 GiB free, naming the tile.
    assert padded_size(SMALL) == (544, 512)
    room = peaks.VAE_ROOM
    untiled, tiled = (
        peaks.decode_peak((544, 512), SMALL, tiles, 0, 36) for tiles in (None, Tiles(512))
    )
    assert (round(gib(untiled), 2), round(gib(tiled), 2)) == (6.47, 6.41)
    made, predicted = shot_plan.plan(tiled + room + MARGIN, SMALL, shots(20), Pins(), None)
    assert (made.window, made.encode_tiles, made.decode_tiles) == (20, None, Tiles(512))
    assert (predicted.decode, predicted.bound, made.resident_decode) == (tiled, "decode", 0)
    assert refused(tiled + room + MARGIN - 1, SMALL, shots(20), Pins(), None) == (
        "an output of 512x512 takes 8.27 GiB of free VRAM (its decode in tiles of 512 px, with the"
        " DiT off the GPU: a peak of 6.42 GiB, 1.25 of room a VAE pass is held to, and 0.60 of"
        " margin), and the driver reports 8.26 GiB free: the largest output that fits there is"
        " --resolution 496, about 496x496"
    )
    made, _ = shot_plan.plan(tiled + room + MARGIN - 1, (496, 496), shots(20), Pins(), None)
    assert (made.encode_tiles, made.decode_tiles) == (None, None)  # untiled, which fits there
    untiled_pins = Pins(encode_tile=0, decode_tile=0)
    assert refused(round(8.30 * GIB), SMALL, shots(20), untiled_pins, None) == (
        "--decode-tile 0: the decode untiled takes 8.32 GiB of free VRAM, the DiT off the GPU,"
        " 1.25 of them the room a VAE pass is held to, and the driver reports 8.30 GiB free: the"
        " largest tile that fits is --decode-tile 512"
    )
    made, predicted = shot_plan.plan(untiled + room + MARGIN, SMALL, shots(20), untiled_pins, None)
    assert (made.decode_tiles, predicted.decode) == (None, untiled)
    # A frame little larger than the smallest tile, 528x488, padded to 512x528: its decode
    # takes less untiled, by the small tiles' fit at its area, than in a tile of 512 px, a
    # square, by its measured peak. The untiled pass is its floor, and said so; the tile, pinned
    # where it doesn't fit, is refused naming 0.
    narrow = (488, 528)
    assert padded_size(narrow) == (512, 528)
    untiled, tiled = (
        peaks.decode_peak((512, 528), narrow, tiles, 0, 36) for tiles in (None, Tiles(512))
    )
    assert (round(gib(untiled), 2), round(gib(tiled), 2)) == (6.34, 6.41)
    made, predicted = shot_plan.plan(untiled + room + MARGIN, narrow, shots(20), Pins(), None)
    assert (made.encode_tiles, made.decode_tiles, predicted.bound) == (None, None, "decode")
    assert refused(untiled + room + MARGIN - 1, narrow, shots(20), Pins(), None) == (
        "an output of 528x488 takes 8.19 GiB of free VRAM (its decode untiled, with the DiT off"
        " the GPU: a peak of 6.34 GiB, 1.25 of room a VAE pass is held to, and 0.60 of margin),"
        " and the driver reports 8.18 GiB free: the largest output that fits there is"
        " --resolution 464, about 502x464"
    )
    assert refused(untiled + room + MARGIN, narrow, shots(20), Pins(decode_tile=512), None) == (
        "--decode-tile 512: the decode in tiles of 512 px takes 8.27 GiB of free VRAM, the DiT"
        " off the GPU, 1.25 of them the room a VAE pass is held to, and the driver reports 8.18"
        " GiB free: no tile fits, the untiled decode does, --decode-tile 0"
    )
    made, _ = shot_plan.plan(tiled + room + MARGIN, narrow, shots(20), Pins(decode_tile=512), None)
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
    # (The whole shot of 40 latents in one window on the lean attention; 21 on the plain one's.)
    assert shot_plan.plan(round(8.15 * GIB), (448, 796), shots(40), Pins(), None)[0].window == 40
    assert refused(round(7.85 * GIB), (448, 796), shots(40), Pins(), None).startswith(
        "an output of 796x448 takes 8.11 GiB of free VRAM (its decode in tiles of 512 px, with"
    )
    assert refused(round(8.15 * GIB), (464, 824), shots(40), Pins(), None).startswith(
        "an output of 824x464 takes 8.24 GiB of free VRAM (its decode in tiles of 512 px, with"
    )
    # A frame too small for a tile has its untiled passes for floor, by the small tiles' fit at
    # its area: the encode's on a frame this small, with the 0.4 GiB seedvr2x's encode holds
    # over its fit.
    encode = peaks.encode_peak((128, 128), None, 0, 36)
    decode = peaks.decode_peak((128, 128), (96, 128), None, 0, 36)
    assert (round(gib(encode), 2), round(gib(decode), 2)) == (2.76, 2.38)
    assert refused(encode + room + MARGIN - 1, (96, 128), shots(6), Pins(), None).startswith(
        "an output of 128x96 takes 4.62 GiB of free VRAM (its encode untiled, too small for a"
        " tile, with the DiT off the GPU: a peak of 2.77 GiB, 1.25 of room a VAE pass is held to,"
        " and 0.60 of margin)"
    )
    made, predicted = shot_plan.plan(encode + room + MARGIN, (96, 128), shots(6), Pins(), None)
    assert (made.encode_tiles, made.decode_tiles, made.resident_encode) == (None, None, 0)
    assert (predicted.encode, predicted.bound) == (encode, "encode")


def test_least_pass_is_the_least(monkeypatch: pytest.MonkeyPatch) -> None:
    # The pass a size's floor counts for its encode and for its decode (shot_plan._least_pass) is
    # the least of every pass the planner may make, untiled or in tiles from 512 px to the
    # largest under the frame's longer side, whatever the shot's length and with split or
    # without: over frames from 48 px a side to 4K, the target's own size (the padding replaced by
    # none, every side being a multiple of 16).
    least_pass = shot_plan._least_pass  # pyright: ignore[reportPrivateUsage]
    vae_need = shot_plan._vae_need  # pyright: ignore[reportPrivateUsage]
    tile_sizes = shot_plan._tile_sizes  # pyright: ignore[reportPrivateUsage]
    job = shot_plan._job  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(shot_plan, "padded_size", lambda size: size)
    heights = (48, 64, 80, 128, 144, 160, 496, 512, 528, 544, 576, 1024, 1104, 2192)
    widths = (48, 512, 528, 544, 576, 1040, 1104, 1920, 3840)
    found: dict[tuple[str, int | None], set[tuple[int, int]]] = {}
    for padded in ((height, width) for height in heights for width in widths):
        for latents in (1, 40, 1000):
            for split in (False, True):
                counted = job(GIB, padded, shots(latents), split)
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
    # then takes 0.3 GiB over its rule (peaks.CUT_OFF_EXTRA). And tiles of 1024 px on a strip
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
    public functions and the run's own layout (shots.shot_layout) alone: a VAE pass its peak and
    its room; the DiT's phase the longest window a shot runs. A plan with the last rungs by their
    own figures: the decode's tail (peaks.decode_peak's `tail`), the room down to its floor
    (Plan.vae_floor, peaks.VAE_ROOM_LEAST) in place of VAE_ROOM."""
    padded = padded_size(target)
    lengths = {(padded_length(shot.frames) - 1) // 4 + 1: shot.frames for shot in film}
    window = max(
        end - start
        for frames in lengths.values()
        for start, end in shot_layout(frames, made.window)
    )
    dit = peaks.dit_peak(padded, window, made.swapped, made.vae_in_dit)
    encode = peaks.encode_peak(padded, made.encode_tiles, made.resident_encode, made.swapped)
    decode = peaks.decode_peak(
        *(padded, output_size(target), made.decode_tiles, made.resident_decode, made.swapped),
        split,
        tail=made.tail_frames,
    )
    room = peaks.VAE_ROOM if made.vae_floor is None else made.vae_floor
    return {"encode": encode + room, "dit": dit, "decode": decode + room}


def shot_held(made: Plan, target: tuple[int, int], film: list[Shot]) -> int:
    """The most the run holds of one of a job's shots in system memory, in bytes, counted here
    for every shot, by the run's own layout (shots.shot_layout) and in half latents, a quarter of
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
    parts += [peaks.DIT_OTHER] if 0 in residents else []
    parts += [] if made.vae_in_dit else [peaks.VAE_WEIGHTS]
    pinned = sum(paged(part) for part in parts)
    held = shot_held(made, target, film)
    return shot_plan.HOST_RESIDENT + pinned + held + made.scratch_allowance


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
    plans = floors = tiled = short = jobs = rung_plans = 0
    for target in targets:
        for film in films:
            for free in (round(count * GIB) for count in frees):
                for pins in pin_sets:
                    for turn in range(3):
                        split, host = kinds[(jobs + 5 * turn) % len(kinds)]
                        job = (free, target, film, pins, host)
                        try:
                            made, predicted = shot_plan.plan(*job, split=split)
                        except PlanError as error:
                            # Every refusal of the sweep, in the library's own words, names no
                            # option (errors.Refusal.words has the application's).
                            assert "--" not in str(error), str(error)
                            floors += floor_named(cli_words(error), free, target, film, split)
                            short += "of system memory" in cli_words(error)
                            continue
                        plans += 1
                        rung_plans += made.tail_frames
                        need = needs(made, target, film, split)
                        assert max(need.values()) <= free - MARGIN
                        assert need == {phase: predicted.need(phase) for phase in shot_plan.PHASES}
                        taken = host_need(made, target, film)
                        assert host is None or taken <= host
                        assert predicted.host == taken
                        assert predicted.shot == shot_held(made, target, film)
                    jobs += 1
                    try:
                        made, _ = shot_plan.plan(free, target, film, pins, None)
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
                        assert tiles is None or not peaks.covers(padded_size(target), tiles)
                        tiled += tiles is not None
    # A size the floor refused before the rungs is still a floor or a plan with a rung (the
    # rungs only plan what was refused), so the two together are the floors of before.
    assert plans > 1500 and floors + rung_plans > 1000 and tiled > 300 and short > 250
    assert rung_plans > 0


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
    sizes = range(min(peaks.TILE_MAX, (max(padded) - 1) // 64 * 64), 511, -64)
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
    found = refusals = rungs_planned = 0
    for index, target in enumerate(((540, 960), HD, UHD)):
        for film in films:
            for free in (round(count * GIB) for count in frees):
                split = bool((found + refusals + index) % 2)
                expected = reference(free, target, film, split)
                if expected is None:
                    assert refused(free, target, film, Pins(), None, split=split)
                    refusals += 1
                    # What the rungs plan there fits by their own figures, tail first.
                    try:
                        rung, _ = shot_plan.plan(free, target, film, Pins(), None, split=split)
                    except PlanError:
                        continue
                    assert rung.tail_frames
                    assert max(needs(rung, target, film, split).values()) <= free - MARGIN
                    rungs_planned += 1
                    continue
                made, _ = shot_plan.plan(free, target, film, Pins(), None, split=split)
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

    shot_plan.plan(free, scaled(short), film, Pins(), None, split=split)
    if short + 16 < height:
        larger = (free, scaled(short + 16), film, Pins(), None)
        assert refused(*larger, split=split).startswith("an output of")
    return True


def test_fewer_swapped_compares_each_pass_on_its_own() -> None:
    # The second plan of a host-short search is taken over the first (fitting as it does) only
    # with tiles no smaller on the decode and on the encode each, and fewer blocks swapped: not
    # by the two sizes as a tuple, whose comparison is lexicographic and would pass a larger
    # decode with a smaller encode. Plans made from a real one by `replace`.
    fewer_swapped = shot_plan._fewer_swapped  # pyright: ignore[reportPrivateUsage]
    base, _ = shot_plan.plan(60 * GIB, UHD, [Shot(0, 24)], Pins(), 40 * GIB)

    def made(decode: Tiles | None, encode: Tiles | None, swapped: int) -> Plan:
        return replace(base, decode_tiles=decode, encode_tiles=encode, swapped=swapped)

    first = made(Tiles(1024), Tiles(1024), 6)
    fewer = first.swapped - 1
    assert fewer_swapped(first, made(Tiles(1024), Tiles(1024), fewer))
    # A larger decode with a smaller encode, and the reverse.
    assert not fewer_swapped(first, made(Tiles(1280), Tiles(768), fewer))
    assert not fewer_swapped(first, made(Tiles(768), Tiles(1280), fewer))
    # Both larger.
    assert fewer_swapped(first, made(Tiles(1280), Tiles(1280), fewer))
    # Both smaller.
    assert not fewer_swapped(first, made(Tiles(768), Tiles(768), fewer))
    # A pass untiled is the largest, on either pass.
    assert fewer_swapped(first, made(None, Tiles(1024), fewer))
    assert fewer_swapped(first, made(Tiles(1024), None, fewer))
    assert fewer_swapped(first, made(None, None, fewer))
    assert not fewer_swapped(made(None, Tiles(1024), 6), made(Tiles(1024), None, fewer))
    assert not fewer_swapped(made(None, None, 6), made(None, Tiles(2048), fewer))
    # The same count swapped, or more, is no gain.
    assert not fewer_swapped(first, made(Tiles(1024), Tiles(1024), first.swapped))
    assert not fewer_swapped(first, made(None, None, first.swapped + 1))


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

    # 4K, shots of 23, 24, 150 and 600 frames, 60.0 GiB free, 13.5 GiB of system memory (63.0 GiB
    # and 11.22 before the lean attention's refit, which gave windows of 12 with the VAE kept and
    # 13 with it away; the inputs are found again by a sweep of the planner's, where the plan is
    # of the same kind): windows of 27 latents run with the VAE away, nothing of the DiT
    # leaving the GPU in the VAE's phases, its 22 swapped blocks and the VAE pinned.
    made, peaks = shot_plan.plan(60 * GIB, UHD, film(23, 24, 150, 600), Pins(), round(13.5 * GIB))
    assert (made.window, made.swapped, made.vae_in_dit) == (27, 22, False)
    assert (made.encode_tiles, made.decode_tiles) == (Tiles(2048), Tiles(1792))  # as before
    assert (made.resident_encode, made.resident_decode) == (None, None)
    assert peaks.shrunk == shot_plan.Shrunk(Tiles(2048), Tiles(1920), 1766, 1350)
    # 5K, a shot of 149 frames, split, 18.375 GiB (14.32 GiB and 43.0 GiB free before the refit,
    # the lean attention running longer windows there; found again by a sweep): windows of 5 run,
    # the smallest.
    made, peaks = shot_plan.plan(
        20 * GIB, (2880, 5120), film(149), Pins(), round(18.375 * GIB), split=True
    )
    assert (made.window, made.swapped, made.vae_in_dit) == (5, 35, False)
    assert (made.encode_tiles, made.decode_tiles) == (Tiles(1280), Tiles(960))
    assert peaks.shrunk == shot_plan.Shrunk(Tiles(1344), Tiles(1024), 1893, 1837)
    # A byte under what that plan takes but its scratch, no window's plan fits, 5 latents being
    # the smallest: the refusal says the least any tiles take there, the VAE away, its 25
    # swapped blocks and the VAE's weights pinned, where the VAE kept pins more blocks.
    least = peaks.host - made.scratch_allowance
    assert refused(20 * GIB, (2880, 5120), film(149), Pins(), least - 1, split=True).startswith(
        "the plan takes 18.36 GiB of system memory (2.40 for the process, 15.24 pinned for 35"
        " blocks and the VAE's weights that leave the GPU, 0.72 for the longest shot"
    )
    # Where neither fits, the figures said are of the plan that takes the least, the VAE kept
    # or not: 720p, shots of 12 and 133 latents, 19 GiB free, 16.5 GiB of system memory. At the
    # card's window, 35 latents, the VAE kept (35 blocks swapped, the one unswapped block kept
    # in the decode, nothing more to keep) takes 17.30 GiB, and the VAE away with the decode's
    # tiles shrunk from 1216 to 1152 px 17.35: the plan's line says the 17.30, windows of 29
    # taken. (On the lean attention, which runs longer windows: the card's window 68 latents, the
    # VAE kept taking 16.05 GiB where 12.0 are available, windows of 46 taken; the host is 12.0
    # GiB, not 16.5.)
    made, peaks = shot_plan.plan(19 * GIB, (720, 1280), shots(12, 133), Pins(), round(12.0 * GIB))
    assert made.window == 46 and peaks.shortened == shot_plan.Shortened(68, 1605, 1200)
    # Where the VAE kept and the tiles shrunk fit, that is the plan, but where the VAE away fits
    # too with tiles no smaller and fewer blocks swapped (shot_plan._fewer_swapped, 2026-11-11):
    # the grid's job whose window fell (test_on_a_short_host_the_tiles_shrink_before_the_window),
    # the same window and tiles with one block fewer swapped, the VAE away (39, 14 and the VAE
    # kept before; windows of 21 and 24 before 2026-10-16).
    made, _ = shot_plan.plan(
        48_855_253_769, (1440, 2560), shots(5, 39), Pins(), 18_616_624_832, split=True
    )
    assert (made.window, made.swapped, made.vae_in_dit) == (39, 13, False)
    assert (made.encode_tiles, made.decode_tiles) == (None, Tiles(1664))


def test_the_tiles_shrunk_for_the_host_are_the_largest_that_fit_it() -> None:
    # Over a grid of jobs, memories and hosts: a plan whose tiles the planner shrank for the
    # host's memory (Peaks.shrunk) fits the host, takes no pass's tiles over the card's own, and
    # keeps more of the DiT on the GPU than that plan; and each pass it shrank, made one tile
    # larger, its residency counted again, takes more system memory than the host has: the
    # tiles shrink no more than the host needs. Its window is the planner's, its own or the
    # longest whose plan fits both (test_the_planner_takes_the_longest_window_the_host_holds).
    fitting, resident = shot_plan._fitting, shot_plan._resident  # pyright: ignore[reportPrivateUsage]
    shrunk = 0
    films = (shots(2), shots(12), shots(40), shots(76), shots(5, 39))
    for target in ((720, 1280), HD, (1440, 2560), UHD):
        for film in films:
            for free in (round(count * GIB) for count in (20, 24, 30, 40, 48, 60, 80)):
                for host in (round(count * GIB) for count in (6, 10, 14, 18)):
                    try:
                        made, peaks = shot_plan.plan(free, target, film, Pins(), host)
                    except PlanError:
                        continue
                    if peaks.shrunk is None:
                        continue
                    shrunk += 1
                    job = shot_plan._job(free, target, film, False)  # pyright: ignore[reportPrivateUsage]
                    assert peaks.host <= host and fits(made, free, target, film, host) is None
                    alone = replace(
                        made,
                        encode_tiles=peaks.shrunk.encode,
                        decode_tiles=peaks.shrunk.decode,
                        resident_encode=resident(job, "encode", peaks.shrunk.encode, made.swapped),
                        resident_decode=resident(job, "decode", peaks.shrunk.decode, made.swapped),
                    )
                    assert shot_plan._leaving(made) < shot_plan._leaving(alone)  # pyright: ignore[reportPrivateUsage]
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
    # (shot_plan._window_that_runs), each by a job a brute force over the planner found (a
    # review, 2026-10-12): the grid above has none at either end.
    # The first window under the card's own. 4K, shots of 63, 60, 89 and 53 latents, 80 GiB
    # free (80 GiB and 17.33 GiB of system memory before the lean attention's refit, which gave
    # the same windows, 19 and 18, there; the inputs are found again by a sweep), --decode-tile
    # 1280 and, with 40 GiB free, 17.0 GiB of system memory: the card alone holds windows of 19,
    # whose plan pins 30 blocks, 17.55 GiB; windows of 18 plan there and are taken, not those of
    # 17, which plan too.
    film, pins, host = shots(63, 60, 89, 53), Pins(decode_tile=1280), round(17.0 * GIB)
    alone, _ = shot_plan.plan(40 * GIB, UHD, film, pins, None)
    made, predicted = shot_plan.plan(40 * GIB, UHD, film, pins, host)
    assert (alone.window, made.window) == (19, 18)
    assert predicted.shortened == shot_plan.Shortened(19, 1755, 1700)
    assert shot_plan.plan(40 * GIB, UHD, film, replace(pins, window=17), host)[0].window == 17
    assert "of system memory" in refused(40 * GIB, UHD, film, replace(pins, window=19), host)
    # The smallest window the job can run. 4K (1080p before the refit: on the lean attention its
    # shot of 45 latents never comes down to 5), a shot of 45 latents, a run held to its card
    # that counts on 17.05 GiB (16.95 before the DiT's share went from 0.3 to 0.33 of P, which
    # left the card alone windows of 5 there), 18 GiB of system memory: the card
    # alone holds windows of 6, every part of the weights pinned, 18.66 GiB; windows of 5, the
    # smallest, plan, and are taken, where the job was refused had the search stopped a window
    # short: the VAE away, 30 blocks swapped, the tiles those the VAE kept gives with 31
    # (shot_plan._fewer_swapped, 2026-11-11). The figure counted is the card's less
    # torch's limit, each step left its own reserve beyond its peak (peaks.reserve; it was 13.125
    # GiB when the flat 4.5 GiB was taken off it, 17.625 less 4.5).
    held = Reading(13 * GIB, None, round(17.05 * GIB), True, round(0.5 * GIB))
    job = (held, UHD, shots(45))
    alone, _ = shot_plan.plan(*job, Pins(), None)
    made, predicted = shot_plan.plan(*job, Pins(), 18 * GIB)
    assert (alone.window, made.window, made.swapped, made.vae_in_dit) == (6, 5, 30, False)
    assert (made.encode_tiles, made.decode_tiles) == (Tiles(1088), Tiles(832))
    assert predicted.shortened == shot_plan.Shortened(6, 1866, 1800)
    assert "of system memory" in refused(*job, Pins(window=6), 18 * GIB)


def test_host_moments() -> None:
    # What the run holds of a shot in system memory at each of its moments
    # (shot_plan.host_moments), by the tensors' sizes: a latent is 16 channels of bfloat16 over an
    # eighth of the padded frame a side, half a byte a padded pixel: 1,059,840 bytes at 1080p,
    # padded to 1104x1920, and 4,208,640 at 4K, padded to 2192x3840 (DESIGN.md, Memory planner,
    # Host RAM: 1.01 and 4.01 MiB). tests/test_shot_pass.py measures the moments that run on the
    # CPU.
    latent = shot_plan.host_moments(OURS_1080, 1, None)["decode"]
    assert latent == 16 * 2 * (1104 // 8) * (1920 // 8) == 1104 * 1920 // 2 == 1_059_840
    assert shot_plan.host_moments(OURS_4K, 1, None)["decode"] == 2192 * 3840 // 2 == 4_208_640
    assert (f"{latent / 2**20:.2f}", f"{4_208_640 / 2**20:.2f}") == ("1.01", "4.01")

    def moments(latents: int, cap: int | None) -> dict[str, float]:
        counted = shot_plan.host_moments(OURS_1080, latents, cap)
        assert tuple(counted) == shot_plan.MOMENTS
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
    window = [shot_plan.host_moments(padded, 1, None)["windows"] for padded in (OURS_1080, OURS_4K)]
    assert window == [3_179_520, 12_625_920] == [1104 * 1920 * 3 // 2, 2192 * 3840 * 3 // 2]
    assert [f"{count / 2**20:.2f}" for count in window] == ["3.03", "12.04"]

    def most(padded: tuple[int, int], latents: int, cap: int) -> str:
        return f"{gib(max(shot_plan.host_moments(padded, latents, cap).values())):.2f}"

    assert [most(OURS_1080, 1000, cap) for cap in (20, 5)] == ["2.09", "2.64"]
    assert [most(OURS_1080, 361, cap) for cap in (20, 5)] == ["0.76", "0.95"]
    assert (most(OURS_4K, 1000, 5), most(OURS_4K, 361, 5)) == ("10.47", "3.79")
    assert shot_plan.host_moments(OURS_4K, 1000, 5)["merge"] == 2670 * 4_208_640


def test_a_long_shot_the_host_cannot_hold() -> None:
    # A job whose longest shot fits the GPU and not the host: 4K on a card with 11.7 GiB free (the
    # lean attention's 4K floor is 11.64; the plain one's 24 GB card), a shot of
    # 3,997 frames, 1,000 latents, in windows of 5, the shortest: no shorter window to take.
    # Every part of the weights is pinned, 15.81 GiB, and the shot holds 10.47 GiB as its decode
    # merges its 333 windows: 28.68 GiB with the process, refused on a host with 28 GiB
    # available, naming each part. (14.36 GiB and 32.57 until 2026-10-12, the shot made anew at
    # each join.)
    free, film = round(11.7 * GIB), shots(1000)
    pinned = 36 * paged(BLOCK) + paged(peaks.DIT_OTHER) + paged(peaks.VAE_WEIGHTS)
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
    least = shot_plan.HOST_RESIDENT + pinned + held
    made, predicted = shot_plan.plan(free, UHD, film, Pins(), least)
    assert (made.window, made.scratch_allowance, predicted.host) == (5, 0, least)
    assert (made.encode_tiles, made.decode_tiles) == (Tiles(896), Tiles(640))
    assert refused(free, UHD, film, Pins(), least - 1).startswith("the plan takes 28.68 GiB of")
    # The fit of a plan from elsewhere counts the shot too (fits).
    short = fits(made, free, UHD, film, least - 1)
    assert short is not None and "as its decode merges its windows) and" in short
    assert fits(made, free, UHD, film, least) is None
    unread, predicted = shot_plan.plan(free, UHD, film, Pins(), None)
    assert bare(unread) == bare(made) and unread.scratch_allowance == 8 * GIB
    assert predicted.host == least + 8 * GIB
    made, predicted = shot_plan.plan(free, HD, film, Pins(), 32 * GIB)
    left = 32 * GIB - pinned - 2116 * 1_059_840
    assert (made.window, predicted.moment, made.scratch_allowance) == (20, "merge", left // 2)
    assert f"{gib(predicted.shot):.2f}" == "2.09" and predicted.host <= 32 * GIB
    assert f"{gib(left // 2):.2f}" == "7.05"
    # And with the allowance the plan has.
    assert fits(made, free, HD, film, 32 * GIB) is None
    assert fits(made, free, HD, film, predicted.host - 1) == (
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
    from seedvr2x import blockswap, model

    leaving = shot_plan._leaving  # pyright: ignore[reportPrivateUsage]
    pinned = shot_plan._pinned  # pyright: ignore[reportPrivateUsage]
    keys = [
        *(f"blocks.{index}.attn.weight" for index in range(peaks.BLOCKS)),
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
                plans.append(shot_plan.plan(our_free(card), target, film, Pins(), None)[0])
            except PlanError:
                continue
    kinds: set[tuple[bool, bool, bool]] = set()
    for plan in plans:
        part_of = model._dit_parts(plan.model_plan, peaks.BLOCKS)  # pyright: ignore[reportPrivateUsage]
        parts = set() if part_of is None else {part_of(key) for key in keys} - {None}
        blocks = len(parts - {blockswap.OTHER})
        other = blockswap.OTHER in parts
        assert leaving(plan) == blocks, plan
        for dit_model in peaks.DITS:
            expected = sum(paged(size) for size in peaks.leaving_blocks(dit_model, blocks))
            expected += other * paged(peaks.DIT_OTHER)
            expected += 0 if plan.vae_in_dit else paged(peaks.VAE_WEIGHTS)
            assert pinned(plan, dit_model) == expected, plan
        kinds.add((blocks > 0, other, plan.vae_in_dit))
    assert len(plans) > 150 and len(kinds) == 6  # no other parameters pinned without a block


def test_resumed_around_the_recorded_shape() -> None:
    # A resume takes the recorded shape, cap, tiles and compile flags, as it is, and plans where
    # the weights are again from the memory free that day (DESIGN.md, Memory planner, A plan
    # shapes the output a little): with more, fewer blocks swapped; with less, more.
    # (Windows of 39 latents, 32 blocks swapped on the lean attention; 17 and 28 on the plain
    # one's.)
    recorded = Plan(39, Tiles(1856), Tiles(1088), 32, 1, 0, vae_in_dit=False)
    job = (HD, shots(76), Pins(), None)
    shape = Plan(39, Tiles(1856), Tiles(1088))
    assert bare(shot_plan.plan(our_free(24), *job)[0]) == recorded
    assert bare(shot_plan.plan(our_free(24), *job, recorded=shape)[0]) == recorded
    made, peaks = shot_plan.plan(our_free(48), *job, recorded=recorded)
    assert bare(made) == shape
    assert (peaks.window, peaks.free, peaks.bound) == (39, our_free(48), "decode")
    # Planned whole, that card has no tiles and one window of the shot's 76 latents (two windows
    # of 39 and no tiles on the plain attention's figures): the shape is the record's.
    whole, peaks = shot_plan.plan(our_free(48), *job)
    assert whole == Plan(76, swapped=18, vae_in_dit=False) and peaks.window == 76
    # A cap recorded above the windows it lays out, as a pinned one is, stays as recorded.
    made, peaks = shot_plan.plan(our_free(24), *job, recorded=replace(shape, window=42))
    assert bare(made) == replace(recorded, window=42) and peaks.window == 39
    # A shape recorded with smaller tiles, which leave room, nothing then moving for the encode:
    # with 0.58 GiB less, two more blocks swapped (nothing left in either pass: 28 and 30 swapped
    # with 6 and 4 blocks left in the decode on the plain attention's figures).
    smaller = Plan(39, Tiles(1088), Tiles(1024))
    made, _ = shot_plan.plan(our_free(24), *job, recorded=smaller)
    assert bare(made) == Plan(39, Tiles(1088), Tiles(1024), 32, None, None, vae_in_dit=False)
    made, _ = shot_plan.plan(round(21.99 * GIB), *job, recorded=smaller)
    assert bare(made) == Plan(39, Tiles(1088), Tiles(1024), 34, None, None, vae_in_dit=False)
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
    made, _ = shot_plan.plan(our_free(96), *job, recorded=odd)
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
    assert refused(our_free(24), *job, recorded=Plan(39, None, Tiles(1088))) == (
        "the recorded plan no longer fits the memory free now: the encode, untiled, with the"
        " DiT off the GPU, takes 22.63 GiB of free VRAM, 1.25 of them the room a VAE pass is held"
        " to, and the driver reports 22.57 GiB free, 0.06 GiB short"
    )
    # (A recorded window of 40, laid out as 39: 20 latents fit 16 GB now, where 17 did not.)
    tiled = Plan(40, Tiles(768), Tiles(640))
    assert refused(our_free(16), *job, recorded=tiled) == (
        "the recorded plan no longer fits the memory free now: the DiT's phase, a window of 39"
        " latents with 36 blocks swapped and the VAE off the GPU, takes 20.78 GiB of free VRAM and"
        " the driver reports 14.57 GiB free, 6.21 GiB short"
    )
    assert refused(our_free(24), HD, shots(76), Pins(), 8 * GIB, recorded=recorded) == (
        "the recorded plan no longer fits the memory free now: the plan takes 17.94 GiB of system"
        " memory (2.40 for the process, 15.35 pinned for 36 blocks and the DiT's other parameters"
        " that leave the GPU, 0.19 for the longest shot, 301 frames (76 latents), at its longest"
        " window, its latent and its noise beside the window's output) and 8.00 GiB are"
        " available, 9.94 GiB short"
    )
    # --swap-blocks still applies, taken exactly; where it alone doesn't hold the recorded
    # windows, it is the pin that is refused.
    made, _ = shot_plan.plan(
        our_free(24), HD, shots(76), Pins(swap_blocks=36), None, recorded=tiled
    )
    assert bare(made) == Plan(40, Tiles(768), Tiles(640), 36)
    pins = Pins(swap_blocks=3)
    assert refused(our_free(24), HD, shots(76), pins, None, recorded=recorded) == (
        "--swap-blocks 3: a window of 39 latents, the recorded plan's, takes 34.70 GiB of free"
        " VRAM with 3 blocks swapped, and the driver reports 22.57 GiB free: the fewest blocks"
        " that hold it are --swap-blocks 32"
    )
    # The other pins were the record's when it was made: the shape is taken from it alone.
    pins = Pins(window=5, encode_tile=512, decode_tile=512)
    made, _ = shot_plan.plan(our_free(24), HD, shots(76), pins, None, recorded=recorded)
    assert bare(made) == recorded


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
        "the encode, untiled, with 19 of the DiT's blocks on the GPU, takes 30.80 GiB of free"
        " VRAM, 1.25 of them the room a VAE pass is held to, and the driver reports 22.57 GiB free,"
        " 8.23 GiB short"
    )
    away = Plan(12, Tiles(1600), None, 17, 0, 0, vae_in_dit=False)
    assert fits(away, free, *job) == (
        "the decode, untiled, with the DiT off the GPU, takes 37.30 GiB of free VRAM, 1.25 of"
        " them the room a VAE pass is held to, and the driver reports 22.57 GiB free, 14.73 GiB"
        " short"
    )
    # (The DiT's phase needs a longer window to fall short on the lean attention: 12 latents with
    # 3 blocks swapped fit, the sentence's 28.04 GiB and 5.47 short being the plain one's; 24
    # latents take 27.42.)
    assert fits(replace(away, window=24, swapped=3), free, HD, shots(24), None) == (
        "the DiT's phase, a window of 24 latents with 3 blocks swapped and the VAE off the GPU,"
        " takes 27.42 GiB of free VRAM and the driver reports 22.57 GiB free, 4.85 GiB short"
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
    allowed = replace(tiled, scratch_allowance=peaks.scratch(OURS_1080, 45))
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
        "the decode, untiled, with the DiT's 2 slots on the GPU, takes 38.30 GiB of free VRAM,"
        " 1.25 of them the room a VAE pass is held to, and the driver reports 22.57 GiB free, 15.73"
        " GiB short"
    )
    assert fits(replace(tiled, resident_decode=1), free, *job) == (
        "the decode, in tiles of 1088 px, with 1 of the DiT's blocks on the GPU, takes 22.60 GiB"
        " of free VRAM, 1.25 of them the room a VAE pass is held to, and the driver reports 22.57"
        " GiB free, 0.03 GiB short"
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
    # short of with every block swapped and the VAE away: a window of 44 latents at 1080p with
    # 22 GiB free, 1.20 GiB short, and 15.54 with --swap-blocks 0 (21 latents, 1.07 and 15.41 on
    # the plain attention's figures; the shot is of 44 latents, one window).
    recorded = Plan(44, Tiles(512), Tiles(512))
    free, job = 22 * GIB, (HD, shots(44))
    start = "the recorded plan no longer fits the memory free now: the DiT's phase, a window of 44"
    shape = (
        f"{start} latents with 36 blocks swapped and the VAE off the GPU, takes 23.20 GiB of free"
        " VRAM and the driver reports 22.00 GiB free, 1.20 GiB short"
    )
    assert refused(free, *job, Pins(), None, recorded=recorded) == shape
    assert refused(free, *job, Pins(swap_blocks=36), None, recorded=recorded) == shape
    assert refused(free, *job, Pins(swap_blocks=0), None, recorded=recorded) == (
        f"{start} latents with 0 blocks swapped and the VAE off the GPU, takes 37.54 GiB of free"
        " VRAM and the driver reports 22.00 GiB free, 15.54 GiB short, 14.34 GiB of them"
        " --swap-blocks 0's: with every block of the DiT swapped and the VAE off the GPU, the"
        " recorded plan is 1.20 GiB short"
    )
    assert refused(free, *job, Pins(swap_blocks=20), None, recorded=recorded) == (
        f"{start} latents with 20 blocks swapped and the VAE off the GPU, takes 29.95 GiB of free"
        " VRAM and the driver reports 22.00 GiB free, 7.95 GiB short, 6.75 GiB of them"
        " --swap-blocks 20's: with every block of the DiT swapped and the VAE off the GPU, the"
        " recorded plan is 1.20 GiB short"
    )
    # On a run held to the card, after what would have to be freed, where memory freed on the
    # card can give it: the whole card's figure the one the plan is held under, 17 GiB under
    # the driver's here. The plan counts on its budget, no margin on top. Each reading is torch's
    # own allocator without the segments, whose window has no reserve beyond its peak: the figures
    # are those of the plan's words alone.
    kept = Reading(
        round(45.05 * GIB), round(28.05 * GIB), free - MARGIN, True, 3 * GIB, memory.PLAIN
    )
    assert refused(kept, *job, Pins(swap_blocks=0), None, recorded=recorded).endswith(
        ", the run held to the card's own memory, 15.54 GiB short, which would have to be freed"
        " on the card, 14.34 GiB of them --swap-blocks 0's: with every block of the DiT swapped"
        " and the VAE off the GPU, the recorded plan is 1.20 GiB short"
    )
    # Where it can't, the card's figure not read: what it is short of, and no more.
    kept = Reading(round(28.05 * GIB), None, free - MARGIN, True, 3 * GIB, memory.PLAIN)
    assert refused(kept, *job, Pins(swap_blocks=0), None, recorded=recorded).endswith(
        ", the run held to the card's own memory, 15.54 GiB short, 14.34 GiB of them"
        " --swap-blocks 0's: with every block of the DiT swapped and the VAE off the GPU, the"
        " recorded plan is 1.20 GiB short"
    )
    # Another phase refused first is the shape's own, whatever is pinned: no part of the pin's.
    small = refused(round(6.25 * GIB), *job, Pins(swap_blocks=0), None, recorded=recorded)
    assert small.startswith("the recorded plan no longer fits the memory free now: the encode, in")
    assert "--swap-blocks" not in small
    # Nor has a new job's refusal any: there the pin is refused, naming the fewest blocks.
    assert "of them" not in refused(free, *job, Pins(swap_blocks=0, window=44), None)


def test_peaks_record() -> None:
    # What the manifest keeps of a plan's memory, for information: bytes, the window in latents.
    made, predicted = shot_plan.plan(our_free(24), HD, shots(76), Pins(), 64 * GIB)
    assert predicted.record() == {
        "free": our_free(24),
        # A driver that refuses: what it reports free is what the plan was made in, and the
        # whole card's free memory isn't read.
        "driver_free": our_free(24),
        "card_free": None,
        # No reserve: the driver refuses memory the card doesn't have.
        "reserve": None,
        # The DiT's format the peaks were predicted for: the default's, fp16.
        "dit_format": "fp16",
        "budget": our_free(24) - MARGIN,
        "encode_peak": peaks.encode_peak(OURS_1080, Tiles(1856), 1, 32),
        # The longest window's.
        "dit_peak": peaks.dit_peak(OURS_1080, 39, 32, vae_in_dit=False),
        "decode_peak": peaks.decode_peak(OURS_1080, HD, Tiles(1088), 0, 32),
        "bound": "dit",
        "host": predicted.host,
        # Of it, the longest shot's part: its 76 latents and their noise beside a window of 39
        # (17 on the plain attention).
        "host_shot": (2 * 76 + 39) * 1_059_840,
        "scratch": peaks.scratch(OURS_1080, 301),
        "longest_window": 39,
    }
    assert [predicted.peak(phase) for phase in shot_plan.PHASES] == [
        predicted.encode,
        predicted.dit,
        predicted.decode,
    ]
    # The phase that bounds counts a VAE phase with the room it is held to, as the planner
    # judged it: a decode at the DiT's peak bounds the plan.
    assert (
        predicted.need("dit") == predicted.dit
        and predicted.need("decode") == predicted.decode + peaks.VAE_ROOM
    )
    under = replace(predicted, encode=0)
    assert replace(under, decode=predicted.dit - peaks.VAE_ROOM + 1).bound == "decode"
    assert replace(under, decode=predicted.dit - peaks.VAE_ROOM).bound == "dit"
    assert set(predicted.record()).isdisjoint(made.record()["memory"])
    assert set(made.record()["memory"]) == {
        "swapped_blocks",
        "resident_encode",
        "resident_decode",
        "vae_in_dit",
        "scratch_allowance",
    }


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
    smaller, less KEPT, LEFT of them what torch's limit leaves. Torch's own allocator without the
    segments (memory.PLAIN): a window has no reserve beyond its peak and a VAE pass VAE_ROOM,
    whatever the figure counted, so that the plans these tests compare are the ones the
    flat reserve gave (the per-step reserve is held by test_the_reserve_is_the_step_s_*)."""
    driver = round(reported * GIB)
    whole = None if card is None else round(card * GIB)
    reading = Reading(driver, whole, min(driver, whole or driver) - KEPT, True, LEFT, memory.PLAIN)
    return reading


def test_a_recorded_shape_refused_says_what_it_falls_short_of_apart() -> None:
    # A resume's refusal for want of memory carries what it is short of in parts, for the command
    # line to word first with the figures apart (PlanError.shortfall; DESIGN.md, Memory planner, A
    # plan shapes the output a little, decided on 2026-10-28): the phase, the GiB, whether
    # programs closed on the card give it back. The message stays the figures' sentence.
    job = (HD, shots(76), Pins(), None)
    recorded = Plan(39, Tiles(1856), Tiles(1088))  # 17 on the plain attention

    def refusal(free: int | Reading, **options: Any) -> PlanError:
        with pytest.raises(PlanError) as error:
            shot_plan.plan(free, *job, rungs=False, **options)  # the plain plan's refusal
        return error.value

    # A driver that refuses: the encode, 7.26 GiB short, which closing programs gives back.
    error = refusal(our_free(16), recorded=recorded)
    assert error.shortfall == shot_plan.Shortfall("the encode", "7.26", freeable=True)
    assert "7.26 GiB short" in str(error)
    assert refusal(our_free(16), recorded=Plan(40, Tiles(768), Tiles(640))).shortfall == (
        shot_plan.Shortfall("the DiT's phase", "6.21")
    )
    assert refusal(round(21.99 * GIB), recorded=recorded).shortfall == (
        shot_plan.Shortfall("the decode", "0.04")
    )
    # The system's memory, not the card's.
    with pytest.raises(PlanError) as hosted:
        shot_plan.plan(our_free(24), HD, shots(76), Pins(), 8 * GIB, recorded=recorded)
    assert hosted.value.shortfall == shot_plan.Shortfall("", "", host=True)
    # A new job's refusal and a pin's carry none: the plan has no recorded shape to keep.
    with pytest.raises(PlanError) as untiled:
        shot_plan.plan(our_free(16), HD, shots(76), Pins(decode_tile=0), None)
    assert untiled.value.shortfall is None
    with pytest.raises(PlanError) as pinned:
        shot_plan.plan(our_free(24), HD, shots(76), Pins(swap_blocks=3), None, recorded=recorded)
    assert pinned.value.shortfall is None and pinned.value.instead == Setting("swap_blocks", "32")
    # A run held to the card (torch's own allocator without the segments): the plan's figures; and
    # whether memory freed on the card gives it back (shot_plan._freeable): the card's figure the
    # smaller, up to the driver's.
    film = shots(5)
    decode = peaks.decode_peak(padded_size(HD), output_size(HD), None, 0, 36)
    need = decode + peaks.VAE_ROOM
    shape = Plan(5, None, None)

    def on_card(card: int, driver: int) -> Reading:
        counted = card - memory.HELD_BACK - memory.RESERVE
        return Reading(driver, card, counted, True, memory.HELD_BACK, memory.PLAIN)

    # Short by 1 GiB: the driver reports 2 GiB more than the card has, so that memory freed on
    # the card gives it back.
    short = on_card(need + memory.HELD_BACK + memory.RESERVE - GIB, need + 5 * GIB)
    with pytest.raises(PlanError) as held:
        shot_plan.plan(short, HD, film, Pins(swap_blocks=36), None, recorded=shape)
    assert held.value.shortfall is not None and not held.value.shortfall.host
    # torch's own allocator without the segments (PLAIN) takes no room rung (only the segments,
    # where it was measured), and a recorded shape without the tail takes no tail: the plain plan's
    # refusal, 1.01 short, with nothing counted.
    assert held.value.shortfall == shot_plan.Shortfall("the decode", "1.01", freeable=True)
    assert "least the plan takes" not in str(held.value)
    # The driver reporting no more than the card has: nothing freed gives it.
    flat = on_card(short.card or 0, short.card or 0)
    with pytest.raises(PlanError) as stuck:
        shot_plan.plan(flat, HD, film, Pins(swap_blocks=36), None, recorded=shape)
    assert stuck.value.shortfall is not None and not stuck.value.shortfall.freeable
    assert stuck.value.shortfall.short == held.value.shortfall.short


def test_a_refusal_names_the_phase_that_does_not_fit() -> None:
    # PlanError.phase, for plan_again to know which part of a shape to release: the phase over
    # the budget in the order the plan judges them, "host" for the system's memory, and for a
    # pin the pass or the window it refuses; empty for a pin that is no pin.
    def phase_of(free: int, pins: Pins = NO_PINS, host: int | None = None, **options: Any) -> str:
        with pytest.raises(PlanError) as error:
            shot_plan.plan(free, HD, shots(76), pins, host, rungs=False, **options)
        return error.value.phase

    shape = Plan(39, Tiles(1856), Tiles(1088))  # 17 on the plain attention
    assert phase_of(our_free(16), recorded=shape) == "encode"
    assert phase_of(our_free(16), recorded=Plan(40, Tiles(768), Tiles(640))) == "dit"
    assert phase_of(round(21.99 * GIB), recorded=shape) == "decode"
    assert phase_of(our_free(24), host=8 * GIB, recorded=shape) == "host"
    assert phase_of(our_free(16), Pins(decode_tile=0)) == "decode"
    assert phase_of(our_free(16), Pins(window=60)) == "dit"  # 30 on the plain one's
    assert phase_of(our_free(96), Pins(swap_blocks=37)) == ""


def test_a_held_run_s_choices_for_the_host_take_no_margin_either() -> None:
    # The two choices made for the host's memory, the VAE's weights kept in the DiT's phase
    # (shot_plan._vae_kept) and the planner's own shorter window (shot_plan._window_that_runs), are
    # made only for a plan the card holds: judged, on a run held to the card, against what the
    # plan counts on whole, no margin on top (shot_plan._Job.margin), as the plan itself is. A
    # held run counting on N bytes plans as a driver that refuses with N and the margin free:
    # the same windows, blocks and tiles, on the same host. Torch's own allocator without the
    # segments (memory.PLAIN), where a window has no reserve beyond its peak: under the segments
    # the window's reserve (peaks.reserve) is at least that margin, and the plans differ
    # (test_the_reserve_is_the_step_s_beyond_its_peak).
    def on_card(counted: int) -> Reading:
        return Reading(
            counted + round(5.0 * GIB), None, counted, True, round(0.5 * GIB), memory.PLAIN
        )

    job = (HD, shots(76), Pins())
    # 1080p on a 24 GB card, a shot of 301 frames, a host one byte short of the plan: 33 blocks
    # swapped and the VAE staying for 32 and the VAE away, the windows of 39 kept (17, 29 and 28
    # on the plain attention's figures; test_the_vae_stays_where_the_host_is_short). The DiT's
    # phase stands under the margin from the budget there: with a margin on top, the held plan
    # would pass for one the card doesn't hold, and its window be shortened instead.
    free = our_free(24)
    pinned = 36 * paged(BLOCK) + paged(peaks.DIT_OTHER) + paged(peaks.VAE_WEIGHTS)
    first = shot_plan.HOST_RESIDENT + pinned + 191 * 1_059_840
    refusing, predicted = shot_plan.plan(free, *job, first - 1)
    assert (refusing.window, refusing.swapped, refusing.vae_in_dit) == (39, 33, True)
    assert 0 < predicted.budget - predicted.dit < MARGIN
    held, on_it = shot_plan.plan(on_card(free - MARGIN), *job, first - 1)
    assert held == refusing and on_it.budget == on_it.free == predicted.budget
    assert on_it.shortened is None and predicted.shortened is None
    # 40 GiB to count on and 16 GiB of system memory: two windows of 39, the card's own window
    # of 76 pinning 34 blocks (80 GiB and 35 on the plain attention's figures, which the lean one
    # holds whole) (test_the_planner_takes_the_longest_window_the_host_holds).
    refusing, predicted = shot_plan.plan(40 * GIB, *job, 16 * GIB, split=True)
    alone, by_itself = shot_plan.plan(40 * GIB, *job, None, split=True)
    assert (refusing.window, alone.window, predicted.shortened is not None) == (39, 76, True)
    assert 0 < by_itself.budget - by_itself.dit < MARGIN
    held, on_it = shot_plan.plan(on_card(40 * GIB - MARGIN), *job, 16 * GIB, split=True)
    assert held == refusing and on_it.shortened == predicted.shortened
    assert (on_it.free, on_it.budget) == (40 * GIB - MARGIN,) * 2


def test_resumed_on_the_one_fit() -> None:
    # Where the weights are is planned again on every resume, from the memory free that day
    # (DESIGN.md, Memory planner, A plan shapes the output a little): the recorded shape is held
    # to today's fit. On the plain attention a shape recorded with FlashAttention 2 and taken up
    # on SDPA had the same windows and tiles and two more blocks swapped; the lean attention gave
    # both one fit (peaks.DIT_ATTENTION) and SDPA is the one kernel left, so a shape recorded by
    # a run of either, before or after FlashAttention 2 went, resumes as the same plan.
    free, job = our_free(24), (HD, shots(76), Pins(), None)
    shape = Plan(39, Tiles(1856), Tiles(1088))
    recorded = Plan(39, Tiles(1856), Tiles(1088), 32, 1, 0, vae_in_dit=False)
    assert bare(shot_plan.plan(free, *job, recorded=shape)[0]) == recorded
    resumed, predicted = shot_plan.plan(free, *job, recorded=recorded)
    assert bare(resumed) == recorded
    assert "attention" not in predicted.record()
    # A shape whose windows hold on that card is one plan, fitting or refused alike.
    long = (HD, shots(300), Pins(), None)
    recorded, first = shot_plan.plan(free, *long)
    assert bare(recorded) == Plan(40, Tiles(1856), Tiles(1088), 33, 1, 0, vae_in_dit=False)
    assert fits(recorded, free, HD, shots(300), None) is None
    same, again = shot_plan.plan(free, *long, recorded=recorded)
    assert same == recorded and again.dit == first.dit


# ---- Weights held: the pinned copies exist already (shot_plan.plan's held) -----------------------


def one_film(*frames: int) -> list[Shot]:
    ends = [sum(frames[: index + 1]) for index in range(len(frames))]
    return [Shot(end - count, end) for end, count in zip(ends, frames, strict=True)]


# Three jobs a short host makes the planner shrink tiles, shorten the window or keep the VAE
# for (test_the_vae_kept_first_costs_no_window's): the free memory, target, shots, host, split.
SHORT_HOSTS = {
    "4k": (60 * GIB, UHD, one_film(23, 24, 150, 600), round(13.5 * GIB), False),
    "5k": (20 * GIB, (2880, 5120), one_film(149), round(18.375 * GIB), True),
    "720p": (19 * GIB, (720, 1280), shots(12, 133), round(12.0 * GIB), False),
}


@pytest.mark.parametrize("name", list(SHORT_HOSTS))
def test_held_weights_make_the_plan_the_card_alone_gives(name: str) -> None:
    # The weights' pinned copies exist already: none is counted against the host, so that the
    # choices made only to pin less of it (the VAE kept for the DiT's phase, the DiT kept in the
    # VAE's phases with tiles shrunk, a window shortened for the blocks it pins) are never taken:
    # the plan is the one the card alone gives, a host unchecked, but for the scratch's allowance,
    # which follows what the host has left.
    free, target, film, host, split = SHORT_HOSTS[name]
    plain, plain_peaks = shot_plan.plan(free, target, film, Pins(), host, split=split)
    made, held_peaks = shot_plan.plan(free, target, film, Pins(), host, split=split, held=True)
    card, _ = shot_plan.plan(free, target, film, Pins(), None, split=split)
    # Where the plan without it did take such a choice, or the tests below mean nothing.
    assert (
        plain_peaks.shrunk is not None
        or plain_peaks.shortened is not None
        or (plain.vae_in_dit != card.vae_in_dit)
    )
    assert replace(made, scratch_allowance=0) == replace(card, scratch_allowance=0)
    assert held_peaks.shrunk is None and held_peaks.shortened is None
    # Nothing pinned in the host count: the process's own, the shot and the scratch.
    assert shot_plan._pinned(made, peaks.FP16, held=True) == 0  # pyright: ignore[reportPrivateUsage]
    assert held_peaks.host == (shot_plan.HOST_RESIDENT + held_peaks.shot + made.scratch_allowance)
    assert held_peaks.host < plain_peaks.host


def test_a_job_refused_for_the_host_is_planned_with_the_weights_held() -> None:
    free, target, film, _, split = SHORT_HOSTS["5k"]
    host = 5 * GIB
    words = refused(free, target, film, Pins(), host, split=split)
    assert "pinned for" in words and "5.00 GiB are available" in words
    made, held_peaks = shot_plan.plan(free, target, film, Pins(), host, split=split, held=True)
    assert held_peaks.host <= host
    # The same weights still leave the GPU in the plan's phases, held or not: only the host's
    # count of them is 0.
    assert made.swapped == 35 and not made.vae_in_dit  # 25 on the plain attention
    # A host that doesn't hold the process, a shot and no scratch is refused all the same, and
    # its words count no pinned copy.
    words = refused(free, target, film, Pins(), 2 * GIB, split=split, held=True)
    assert "for the process" in words and "pinned for" not in words
    assert "2.00 GiB are available" in words


def test_held_false_plans_every_job_as_before() -> None:
    for free, target, film, host, split in SHORT_HOSTS.values():
        for available in (None, host, 5 * GIB):
            job = (free, target, film, Pins(), available)
            try:
                before = shot_plan.plan(*job, split=split)
            except PlanError as error:
                assert refused(*job, split=split, held=False) == str(error)
                continue
            assert shot_plan.plan(*job, split=split, held=False) == before
    # Pinned parts: the planner's function counts them unless it is told they exist.
    plan_, _ = shot_plan.plan(*SHORT_HOSTS["5k"][:3], Pins(), None, split=True)
    assert shot_plan._pinned(plan_, peaks.FP16) > 0  # pyright: ignore[reportPrivateUsage]
    assert shot_plan._pinned(plan_, peaks.FP16, held=True) == 0  # pyright: ignore[reportPrivateUsage]


def test_the_choices_for_the_host_are_not_taken_on_held_weights() -> None:
    # The two functions that make them, called on a job that applies to them: they return a plan
    # for the job as it was, and None for one whose weights are held.
    free, target, film, host, split = SHORT_HOSTS["720p"]
    plain_job = shot_plan._job(free, target, film, split)  # pyright: ignore[reportPrivateUsage]
    held_job = shot_plan._job(free, target, film, split, held=True)  # pyright: ignore[reportPrivateUsage]
    pins = Pins()
    start, _ = shot_plan.plan(free, target, film, pins, None, split=split)
    assert shot_plan._vae_kept(plain_job, replace(start, vae_in_dit=False), host) is not None  # pyright: ignore[reportPrivateUsage]
    assert shot_plan._vae_kept(held_job, replace(start, vae_in_dit=False), host) is None  # pyright: ignore[reportPrivateUsage]
    free, target, film, host, split = SHORT_HOSTS["5k"]
    plain_job = shot_plan._job(free, target, film, split)  # pyright: ignore[reportPrivateUsage]
    held_job = shot_plan._job(free, target, film, split, held=True)  # pyright: ignore[reportPrivateUsage]
    start, _ = shot_plan.plan(free, target, film, pins, None, split=split)
    assert shot_plan._dit_kept(plain_job, start, host, pins) is not None  # pyright: ignore[reportPrivateUsage]
    assert shot_plan._dit_kept(held_job, start, host, pins) is None  # pyright: ignore[reportPrivateUsage]


# ---- The refusals by kind (errors.PlanRefused) --------------------------------------------------


def raised(*job: object, **options: object) -> PlanRefused:
    """The refusal shot_plan.plan makes of a job, whatever its kind."""
    with pytest.raises(PlanRefused) as error:
        shot_plan.plan(*job, **options)  # pyright: ignore[reportArgumentType]
    return error.value


def test_each_refusal_of_the_planner_is_of_its_kind_with_its_facts() -> None:
    recorded = Plan(39, Tiles(1856), Tiles(1088))  # 17 on the plain attention
    # A swap count above the DiT's blocks, and a tile covering the frame: pins that are no pins,
    # before any memory is counted.
    error = raised(our_free(24), HD, shots(76), Pins(swap_blocks=40), None)
    assert type(error) is PinRefused and error.kind == "pin" and error.why == "range"
    assert error.pin == Setting("swap_blocks", "40") and error.instead is None
    error = raised(our_free(24), HD, shots(76), Pins(decode_tile=2048), None)
    assert type(error) is PinRefused and error.why == "covers" and error.phase == ""
    assert error.pin == Setting("decode_tile", "2048") and error.instead is None
    assert str(error).endswith("decode tile 0 pins an untiled decode")
    # A pin the memory doesn't hold, with the pin that does: the window, the swapped blocks, a tile.
    error = raised(our_free(24), HD, shots(3000), Pins(window=60), None)
    assert type(error) is PinRefused and error.why == "memory" and error.phase == "dit"
    assert error.pin == Setting("window", "60")
    assert error.instead is not None and error.instead.name == "window"
    assert error.need is not None and error.there is not None and error.need > error.there
    error = raised(our_free(24), HD, shots(76), Pins(swap_blocks=3), None, recorded=recorded)
    assert type(error) is PinRefused and error.why == "memory" and error.phase == ""
    assert error.pin == Setting("swap_blocks", "3")
    assert error.instead == Setting("swap_blocks", "32")
    error = raised(our_free(16), HD, shots(76), Pins(decode_tile=0), None)
    assert type(error) is PinRefused and error.why == "memory" and error.phase == "decode"
    assert error.pin == Setting("decode_tile", "0")
    assert error.instead is not None and error.instead.name == "decode_tile"
    assert error.need is not None and error.there is not None and error.need > error.there
    assert error.there == our_free(16)
    # A recorded plan the planner makes for no job.
    error = raised(our_free(24), HD, shots(76), Pins(), None, recorded=Plan(17, Tiles(2048)))
    assert type(error) is PinRefused and error.why == "recorded" and error.pin is None
    # A size that doesn't fit, with no pin to blame: the card's free VRAM, the phase and what
    # fits, the largest resolution; and a resume's shape, short in the same facts.
    error = raised(our_free(6), (540, 960), shots(5), Pins(), None)
    assert type(error) is MemoryShort and error.kind == "memory" and error.phase == "decode"
    assert error.need is not None and error.there == our_free(6) and error.short is not None
    assert error.short == error.need - our_free(6) and error.freeable is True
    assert error.reserve is None and error.instead is not None
    assert error.instead.name == "resolution" and error.instead.value.isdecimal()
    error = raised(our_free(16), HD, shots(76), Pins(), None, recorded=recorded)
    assert type(error) is MemoryShort and error.phase == "encode" and error.instead is None
    # The recorded shape's need, the room at 1.25 GiB: a driver that refuses takes no room rung.
    assert error.shortfall == shot_plan.Shortfall("the encode", "7.26", freeable=True)
    assert error.need is not None and error.short == error.need - our_free(16)
    # The host's memory.
    error = raised(our_free(24), HD, shots(76), Pins(), 8 * GIB, recorded=recorded)
    assert type(error) is HostShort and error.kind == "host" and error.phase == "host"
    assert error.available == 8 * GIB and error.need is not None and error.need > 8 * GIB
    assert error.short == error.need - error.available
    # (--window 20 and the window 13 that holds, for the plain attention's 10 and 6.)
    error = raised(our_free(16), HD, shots(76), Pins(window=20), 12 * GIB)
    assert type(error) is HostShort and error.instead == Setting("window", "13")
    # A held run's refusal says the reserve it was made with, in bytes: under torch's own allocator
    # without the segments the flat one taken off the figure, under the segments the refused
    # step's own (peaks.reserve), and the kind and facts are kept through the rewording.
    decode = peaks.decode_peak(padded_size(HD), output_size(HD), None, 0, 36)
    card = decode + peaks.VAE_ROOM + memory.HELD_BACK + memory.RESERVE - GIB
    reading = Reading(card + 3 * GIB, card, card - memory.HELD_BACK - memory.RESERVE, True, 0)
    reading = replace(reading, left=memory.HELD_BACK, allocator=memory.PLAIN)
    error = raised(reading, HD, shots(5), Pins(swap_blocks=36), None, recorded=Plan(5, None, None))
    assert type(error) is MemoryShort and error.phase == "decode"
    assert error.reserve == memory.RESERVE and error.freeable is True
    assert "--" not in str(error) and cli_words(error) == str(error)
    working = peaks.vae_working(decode, 0, 36)
    own = peaks.reserve(peaks.DECODE_PHASE, working, memory.SEGMENTS)
    segments = replace(reading, allocator=memory.SEGMENTS, counted=card - memory.HELD_BACK - GIB)
    error = raised(segments, HD, shots(5), Pins(swap_blocks=36), None, recorded=Plan(5, None, None))
    assert type(error) is MemoryShort and error.phase == "decode" and error.reserve == own
    assert "the reserve left beyond its peak" in str(error) and "--" not in str(error)
    # The VAE compiled: a pass pinned beyond the shapes measured (a decode tile of 2112 px, over the
    # 2048 px measured on 2026-11-11, at 4K: no such tile is valid at 1080p), and the job the plain
    # plan fits.
    error = raised(our_free(96), UHD, shots(76), Pins(decode_tile=2112), None, compile_vae=True)
    assert type(error) is CompileRefused and error.kind == "compile" and error.phase == "decode"
    assert error.within == (Setting("decode_tile", "2048"),)
    assert str(error).splitlines()[-2:] == [
        "The same command without compiling fits",
        "The same command with decode tile 2048 fits, within the shapes measured",
    ]
    error = raised(
        *(our_free(96), UHD, shots(76), Pins(decode_tile=2112), None),
        *(),
        compile_vae=True,
        compile_fixed=True,
    )
    assert type(error) is PinRefused and error.why == "compile" and error.phase == "decode"


def test_a_plan_s_refusal_in_the_library_s_words_names_no_option() -> None:
    # str() of every kind of refusal above is the library's own words, which name no option of
    # any application; the command line's say gives back the option each names.
    recorded = Plan(39, Tiles(1856), Tiles(1088))  # 17 on the plain attention
    named = {
        "--swap-blocks 40": raised(our_free(24), HD, shots(76), Pins(swap_blocks=40), None),
        "--decode-tile 2048": raised(our_free(24), HD, shots(76), Pins(decode_tile=2048), None),
        "--window 60": raised(our_free(24), HD, shots(3000), Pins(window=60), None),
        "--swap-blocks 32": raised(
            *(our_free(24), HD, shots(76), Pins(swap_blocks=3), None), recorded=recorded
        ),
        "--decode-tile 1856": raised(our_free(16), HD, shots(76), Pins(decode_tile=0), None),
        "--resolution 48": raised(our_free(6), (540, 960), shots(5), Pins(), None),
        "--compile": raised(
            *(our_free(96), UHD, shots(76), Pins(decode_tile=2112), None), compile_vae=True
        ),
    }
    for option, error in named.items():
        assert "--" not in str(error), str(error)
        assert error.words(cli_say) != str(error)
        assert option.split()[0] in cli_words(error), option
    # The ones that name no setting are the same in every words.
    for error in (
        raised(our_free(24), HD, shots(76), Pins(), 8 * GIB, recorded=recorded),
        raised(our_free(16), HD, shots(76), Pins(), None, recorded=recorded),
        raised(our_free(24), HD, shots(76), Pins(), None, recorded=Plan(17, Tiles(2048))),
    ):
        assert "--" not in str(error) and cli_words(error) == str(error)


def test_the_reserve_is_the_step_s_beyond_its_peak(monkeypatch: pytest.MonkeyPatch) -> None:
    # A held run is judged phase by phase: a phase needs its peak and its reserve (peaks.reserve)
    # under the budget, which is the figure counted whole (no margin on top, the reserve's floor
    # standing for it). DESIGN.md, Allocator, Small cards, decided on 2026-11-06.
    def reading(allocator: str, counted: float = 24.0) -> Reading:
        return Reading(
            round(counted * GIB) + LEFT, None, round(counted * GIB), True, LEFT, allocator
        )

    for allocator in (memory.SEGMENTS, memory.POOL):
        for target, latents in ((HD, 76), (UHD, 40)):
            made, predicted = shot_plan.plan(
                reading(allocator), target, shots(latents), Pins(), None
            )
            padded = padded_size(target)
            working = peaks.dit_working(padded, predicted.window)
            assert predicted.reserves["dit"] == peaks.reserve(peaks.DIT_PHASE, working, allocator)
            vae_phases: tuple[tuple[Phase, int | None], ...] = (
                ("encode", made.resident_encode),
                ("decode", made.resident_decode),
            )
            for phase, resident in vae_phases:
                own = peaks.reserve(
                    phase,
                    peaks.vae_working(predicted.peak(phase), resident, made.swapped),
                    allocator,
                )
                assert predicted.reserves[phase] == own, (allocator, phase)
            assert predicted.budget == predicted.free
            for phase in shot_plan.PHASES:
                assert predicted.need(phase) == predicted.peak(phase) + predicted.reserves[phase]
                assert predicted.need(phase) <= predicted.budget, (allocator, target, phase)
            assert predicted.reserve_taken() == predicted.reserves
            assert predicted.record()["reserve"] == predicted.reserves
    # Both sides of the floors: a small window at 1080p is under the DiT's 0.33 x P = 0.6 GiB, and
    # its reserve is the floor (MARGIN); a long one at 4K is over it, capped where 0.33 x P
    # is over 4.5 GiB (a window of 40 latents at 4K is left 4.5 GiB, hold_probe.py's scan).
    small = shot_plan.plan(reading(memory.SEGMENTS), HD, shots(1), Pins(), None)[1]
    assert small.reserves["dit"] == MARGIN
    large = shot_plan.plan(reading(memory.SEGMENTS, 96.0), UHD, shots(40), Pins(), None)[1]
    working = peaks.dit_working(padded_size(UHD), large.window)
    assert working * 0.33 > peaks.DIT_RESERVE_CAP
    assert MARGIN < large.reserves["dit"] == peaks.DIT_RESERVE_CAP == round(4.5 * GIB)
    assert large.reserves["dit"] == memory.RESERVE
    # The plan of such a case is the one made with a reserve of 4.5 GiB for the window, whatever
    # its peak: the cap holds in the search as well as in the figure.
    real = peaks.reserve

    def flat(phase: str, working: int, *args: object, **kwargs: object) -> int:
        if phase == peaks.DIT_PHASE:
            return round(4.5 * GIB)
        return real(phase, working, *args, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(peaks, "reserve", flat)
    flatly = shot_plan.plan(reading(memory.SEGMENTS, 96.0), UHD, shots(40), Pins(), None)[1]
    monkeypatch.undo()
    assert flatly == large
    # Torch's own allocator without the segments, and a driver that refuses: as before the reserve
    # was measured by the step. A window has none beyond its peak, a VAE pass VAE_ROOM, and the
    # record keeps the flat figure for every phase, or nothing.
    plain = shot_plan.plan(reading(memory.PLAIN), HD, shots(76), Pins(), None)[1]
    assert plain.reserves == {"encode": peaks.VAE_ROOM, "dit": 0, "decode": peaks.VAE_ROOM}
    assert plain.record()["reserve"] == dict.fromkeys(shot_plan.PHASES, memory.RESERVE)
    refusing = shot_plan.plan(our_free(24), HD, shots(76), Pins(), None)[1]
    assert refusing.reserves == plain.reserves and refusing.record()["reserve"] is None
    assert refusing.budget == refusing.free - MARGIN
    # cudaMallocAsync's pool: a window's max(0.85, 0.24 x P), a VAE pass's max(1.25, 0.35 x P).
    pool = shot_plan.plan(reading(memory.POOL), UHD, shots(40), Pins(), None)[1]
    assert pool.reserves["dit"] == round(0.24 * peaks.dit_working(padded_size(UHD), pool.window))


def test_the_segments_need_less_than_the_flat_reserve_where_the_step_is_small() -> None:
    # The same card under both: the flat 4.5 GiB off the figure (PLAIN) and the segments' reserve,
    # the figure counted being the card's less torch's limit. On a card where each step's reserve
    # is under 4.5 GiB (0.33 x P under it for the DiT, 0.2 x P for a VAE pass) the plan holds a
    # window, tiles and a placement at least as large; the cases that need more are over it.
    def taken(allocator: str, free: float) -> Reading:
        driver = round(free * GIB)
        counted = driver - LEFT - (memory.RESERVE if allocator == memory.PLAIN else 0)
        return Reading(driver, None, counted, True, LEFT, allocator)

    moved = 0
    for target, latents in ((HD, 76), (HD, 20), (UHD, 17)):
        for free in (14.0, 16.0, 24.0, 32.0):
            try:
                flat, flat_peaks = shot_plan.plan(
                    taken(memory.PLAIN, free), target, shots(latents), Pins(), None
                )
            except PlanError:
                flat, flat_peaks = None, None
            try:
                step, step_peaks = shot_plan.plan(
                    taken(memory.SEGMENTS, free), target, shots(latents), Pins(), None
                )
            except PlanError:
                step, step_peaks = None, None
            if flat is None:
                moved += step is not None  # a card the flat reserve refuses, the segments run
                continue
            assert flat_peaks is not None
            assert step is not None and step_peaks is not None, (target, latents, free)
            over = max(step_peaks.reserves.values()) > memory.RESERVE
            if not over:
                # A longer window may swap more blocks (the planner's choice, window first).
                assert (step.window or 0) >= (flat.window or 0), (target, latents, free)
                if step.window == flat.window:
                    assert step.swapped <= flat.swapped, (target, latents, free)
            moved += step != flat
    assert moved


def test_a_measured_margin_over_the_rule_is_the_reserve_of_every_step_of_the_plan() -> None:
    # Reading.margin (the probe's allocator margin, read from its cache): the larger of it and
    # the rule's share is each step's share of P (peaks.reserve), so a plan with a margin over
    # the rule reserves more by step and never plans more (longer windows, larger tiles or fewer
    # blocks swapped), and one under the rule is the rule's plan to the byte. PLAIN takes none.
    def reading(allocator: str, margin: float | None) -> Reading:
        counted = round(24 * GIB)
        return Reading(counted + LEFT, None, counted, True, LEFT, allocator, margin=margin)

    for allocator in (memory.SEGMENTS, memory.POOL):
        base = shot_plan.plan(reading(allocator, None), UHD, shots(40), Pins(), None)
        under = shot_plan.plan(reading(allocator, 0.05), UHD, shots(40), Pins(), None)
        over = shot_plan.plan(reading(allocator, 0.6), UHD, shots(40), Pins(), None)
        assert under[0] == base[0] and under[1].reserves == base[1].reserves, allocator
        assert replace(under[1], reading=base[1].reading) == base[1], allocator
        assert over[1].reserves["dit"] > base[1].reserves["dit"], allocator
        assert over[1].reserves["dit"] == round(
            0.6 * peaks.dit_working(padded_size(UHD), over[1].window)
        )
        for phase in shot_plan.PHASES:
            assert over[1].need(phase) <= over[1].budget
            assert (
                over[1].reserves[phase] >= base[1].reserves[phase]
                or over[1].window < base[1].window
            )
        # Never planned more: the window no longer, the blocks swapped no fewer.
        assert over[1].window <= base[1].window and over[0].swapped >= base[0].swapped
    plain = shot_plan.plan(reading(memory.PLAIN, 0.6), HD, shots(76), Pins(), None)
    same = shot_plan.plan(reading(memory.PLAIN, None), HD, shots(76), Pins(), None)
    assert plain[0] == same[0] and plain[1].reserves == same[1].reserves


# ---- The floor, last rungs (DESIGN.md, Allocator, Small cards): a clip that fits no other way ----


def segments_at(free: float, allocator: str = memory.SEGMENTS) -> Reading:
    """The reading of a held run with `free` GiB reported, LEFT outside torch's limit."""
    driver = round(free * GIB)
    counted = driver - LEFT - (memory.RESERVE if allocator == memory.PLAIN else 0)
    return Reading(driver, None, counted, True, LEFT, allocator)


def rung_of(plan: Plan) -> int:
    """0 no rung, 1 the tail, 2 the tail and the room."""
    return 0 if not plan.tail_frames else 1 if plan.vae_floor is None else 2


def clip_plan(free: float, **options: Any) -> tuple[Plan, Any]:
    return shot_plan.plan(segments_at(free), HD, shots(19), NO_PINS, None, **options)


def test_a_clip_that_fits_without_the_rungs_never_takes_one() -> None:
    for free in (9.0, 9.5, 12.0, 16.0, 24.0, 32.0, 96.0):
        made, taken = clip_plan(free)
        assert rung_of(made) == 0 and made.vae_floor is None, free
        assert "tail_frames" not in made.record()
        assert "vae_floor" not in made.record()["memory"]
        assert taken.reserves["decode"] >= peaks.VAE_RESERVE_FLOOR


def test_a_clip_that_fits_no_other_way_takes_the_tail_then_the_room() -> None:
    # HD, 19 latents, expandable segments: the tail from 7.0 GiB reported free to 8.25, the room
    # besides below 7.0, nothing from 8.5; refused under 6.5. (The levels are the planner's own,
    # by the figures of peaks.TAIL_MEASURED, PROVISIONAL.)
    assert rung_of(clip_plan(8.0)[0]) == 1
    assert rung_of(clip_plan(7.0)[0]) == 1
    assert rung_of(clip_plan(6.75)[0]) == 2
    assert rung_of(clip_plan(6.5)[0]) == 2
    assert rung_of(clip_plan(9.0)[0]) == 0
    with pytest.raises(PlanError):
        clip_plan(6.0)


def test_the_rungs_come_in_order_over_the_cards() -> None:
    levels = [rung_of(clip_plan(free / 4)[0]) for free in range(26, 40)]
    assert levels == sorted(levels, reverse=True)
    assert set(levels) == {0, 1, 2}
    for level, free in zip(levels, range(26, 40), strict=True):
        # The room is never taken without the tail; and a rung only where the plain plan refuses.
        made = clip_plan(free / 4)[0]
        assert (made.vae_floor is None) or made.tail_frames
        job = shot_plan._job(  # pyright: ignore[reportPrivateUsage]
            segments_at(free / 4), HD, shots(19), False
        )
        if level:
            with pytest.raises(PlanError):
                shot_plan._attempt(job, NO_PINS, None, None)  # pyright: ignore[reportPrivateUsage]
        else:
            shot_plan._attempt(job, NO_PINS, None, None)  # pyright: ignore[reportPrivateUsage]


def test_the_plans_with_rungs_hold_their_figures() -> None:
    made, taken = clip_plan(6.5)
    assert made.tail_frames and made.vae_floor == peaks.VAE_ROOM_LEAST
    padded, output = padded_size(HD), output_size(HD)
    # The decode's peak is the tail's, the pass's own reserve the share over the least floor.
    expected = peaks.decode_peak(
        padded,
        output,
        made.decode_tiles,
        made.resident_decode,
        made.swapped,
        split=False,
        tail=True,
    )
    assert taken.decode == expected
    working = peaks.vae_working(taken.decode, made.resident_decode, made.swapped)
    assert taken.reserves["decode"] == max(
        peaks.VAE_ROOM_LEAST, round(peaks.VAE_RESERVE_SHARE * working)
    )
    assert taken.reserves["encode"] >= peaks.VAE_ROOM_LEAST
    assert taken.reserves["encode"] < peaks.VAE_ROOM
    # The tail alone keeps the rule's floor.
    tail, taken_tail = clip_plan(8.0)
    assert tail.vae_floor is None and tail.tail_frames
    assert taken_tail.reserves["decode"] >= peaks.VAE_RESERVE_FLOOR


def test_a_plan_with_rungs_is_recorded_and_read_back() -> None:
    made, _ = clip_plan(6.5)
    record = made.record()
    assert record["tail_frames"] is True  # shapes the output: compared on a resume
    assert record["memory"]["vae_floor"] == peaks.VAE_ROOM_LEAST  # for information
    read = Plan.recorded(record)
    assert read.tail_frames and read.vae_floor is None  # the room is planned again
    # A record without the key (a plan before the rung, or without it) is without the tail.
    plain, _ = clip_plan(16.0)
    assert Plan.recorded(plain.record()).tail_frames is False
    bad = dict(record, tail_frames="yes")
    with pytest.raises(Exception, match="tail_frames"):
        Plan.recorded(bad)


def test_a_resume_holding_the_tail_keeps_it_and_one_without_never_takes_it() -> None:
    tail, _ = clip_plan(8.0)
    recorded = Plan.recorded(tail.record())
    # The same card: the tail kept, the shape as recorded.
    again, _ = clip_plan(8.0, recorded=recorded)
    assert again.tail_frames and again.vae_floor is None
    assert (again.window, again.encode_tiles, again.decode_tiles) == (
        tail.window,
        tail.encode_tiles,
        tail.decode_tiles,
    )
    # A card with room for the plain decode: the recorded tail still holds (the units made have
    # its frames), never dropped for the memory.
    roomy, _ = clip_plan(24.0, recorded=recorded)
    assert roomy.tail_frames
    # A smaller card: the room taken too, on the recorded shape (tiles of 512 px, the least).
    least = Plan(6, Tiles(512), Tiles(512), tail_frames=True)
    with_room, _ = clip_plan(6.5, recorded=least)
    assert with_room.tail_frames and with_room.vae_floor == peaks.VAE_ROOM_LEAST
    assert (with_room.window, with_room.encode_tiles, with_room.decode_tiles) == (
        6,
        Tiles(512),
        Tiles(512),
    )
    with pytest.raises(PlanError):  # the shape of the card at 8.0 does not fit a card of 7.0
        clip_plan(7.0, recorded=recorded)
    # A plain plan's record on a card that needs a rung: the tail is not taken (it would change
    # the frames at 512 and 576 px); the room alone may be, which changes no bit; else refused.
    plain = Plan.recorded(clip_plan(16.0)[0].record())
    for free in (6.5, 7.0, 8.0, 8.5):
        try:
            made, _ = clip_plan(free, recorded=plain)
        except PlanError:
            continue
        assert not made.tail_frames, free


def test_no_rung_is_taken_under_compile() -> None:
    # A compiled decode with the tail is not measured: refused, never run with a rung.
    with pytest.raises(PlanError):
        clip_plan(8.0, compile_vae=True)


def test_a_pinned_decode_tile_is_helped_by_the_rungs_too() -> None:
    made, _ = shot_plan.plan(segments_at(8.0), HD, shots(19), Pins(decode_tile=512), None)
    assert made.tail_frames and made.decode_tiles == Tiles(512)
    with pytest.raises(PlanError):
        shot_plan.plan(segments_at(6.0), HD, shots(19), Pins(decode_tile=512), None)


def test_the_rungs_are_the_dit_s_and_the_hosts_refusals_no_help() -> None:
    # A refusal for the system memory is final: no rung lowers the host's figures.
    with pytest.raises(PlanError) as refused:
        shot_plan.plan(segments_at(8.0), HD, shots(19), NO_PINS, 1 * GIB)
    assert refused.value.phase == "host"


def test_torch_s_own_allocator_without_the_segments_takes_the_tail_alone() -> None:
    # The room rung (the VAE passes' floor down to 0.5 GiB) only under the segments, where it was
    # measured (DESIGN.md: "where its frames still hold"): under PLAIN a pass packed closer than
    # about 1 GiB can give other frames and say nothing (peaks.VAE_ROOM). The tail stays.
    seen = {
        reading_free: rung_of(
            shot_plan.plan(segments_at(reading_free, memory.PLAIN), HD, shots(19), NO_PINS, None)[0]
        )
        for reading_free in (free / 2 for free in range(24, 80))
        if _planned(segments_at(reading_free, memory.PLAIN))
    }
    assert set(seen.values()) >= {0, 1}, "a PLAIN card the tail lifts"
    assert 2 not in seen.values()
    for free in seen:
        assert (
            shot_plan.plan(segments_at(free, memory.PLAIN), HD, shots(19), NO_PINS, None)[
                0
            ].vae_floor
            is None
        )
    # Where the tail alone refuses, the refusal is the tail's, with nothing said of a room.
    refused = [
        free / 2 for free in range(10, 24) if not _planned(segments_at(free / 2, memory.PLAIN))
    ]
    assert refused
    with pytest.raises(PlanError) as plain:
        shot_plan.plan(segments_at(max(refused), memory.PLAIN), HD, shots(19), NO_PINS, None)
    assert "the VAE passes left" not in str(plain.value)
    assert str(plain.value).endswith(
        ", with the decoder's tail a frame at a time, the least the plan takes"
    )


def _planned(reading: Reading) -> bool:
    try:
        shot_plan.plan(reading, HD, shots(19), NO_PINS, None)
    except PlanError:
        return False
    return True


def test_the_pool_and_a_driver_that_refuses_take_the_tail_alone_never_the_room() -> None:
    # The pool (cudaMallocAsync) and a driver that refuses (no lending): the tail where the plain
    # decode is refused, never the room rung, which was measured under the segments alone.
    found = {
        rung_of(shot_plan.plan(round(free / 4 * GIB), HD, shots(19), NO_PINS, None)[0])
        for free in range(20, 80)
        if _planned_number(round(free / 4 * GIB))
    }
    assert 0 in found and 1 in found and 2 not in found
    for allocator in (memory.POOL,):
        rungs = {
            rung_of(
                shot_plan.plan(segments_at(free / 4, allocator), HD, shots(19), NO_PINS, None)[0]
            )
            for free in range(20, 80)
            if _planned(segments_at(free / 4, allocator))
        }
        assert 2 not in rungs, allocator
    # Refused with the tail alone tried: the words count the tail and no room.
    with pytest.raises(PlanError) as refused:
        shot_plan.plan(round(5.0 * GIB), HD, shots(19), NO_PINS, None)
    assert "the VAE passes left" not in str(refused.value)
    assert str(refused.value).endswith(
        ", with the decoder's tail a frame at a time, the least the plan takes"
    )


def _planned_number(free: int) -> bool:
    try:
        shot_plan.plan(free, HD, shots(19), NO_PINS, None)
    except PlanError:
        return False
    return True


TINY = (96, 128)


def test_a_plan_refused_for_the_encode_takes_the_room_alone_never_the_tail() -> None:
    # Small cards, 2026-11-06: the tail lowers the decode alone, so a plan refused for the ENCODE
    # takes the room (segments only) and never the tail, whose frames differ at some tile sides.
    # A 128x96 frame in 6 latents is encode-bound: refused under 4.125 GiB reported free, the
    # room alone from there to 4.75, nothing from 4.875.
    def at(free: float) -> Plan:
        return shot_plan.plan(segments_at(free), TINY, shots(6), NO_PINS, None)[0]

    assert rung_of(at(4.875)) == 0 and at(4.875).vae_floor is None
    for free in (4.125, 4.25, 4.5, 4.75):
        made = at(free)
        assert made.vae_floor == peaks.VAE_ROOM_LEAST and not made.tail_frames, free
    with pytest.raises(PlanError) as refused:
        at(4.0)
    assert refused.value.phase == "encode"
    assert str(refused.value).endswith(
        ", with the VAE passes left 0.50 GiB, the least the plan takes"
    )
    assert "tail" not in str(refused.value)
    # Under the pool, the plain allocator and a driver that refuses: no room, so no rung at all.
    for allocator in (memory.POOL, memory.PLAIN):
        with pytest.raises(PlanError) as plain:
            shot_plan.plan(segments_at(4.5, allocator), TINY, shots(6), NO_PINS, None)
        assert "least the plan takes" not in str(plain.value), allocator
    with pytest.raises(PlanError) as driver:
        shot_plan.plan(round(4.5 * GIB), TINY, shots(6), NO_PINS, None)
    assert "least the plan takes" not in str(driver.value)


def test_a_plan_refused_for_the_decode_climbs_the_tail_then_the_tail_with_the_room() -> None:
    # A decode-bound clip (the 1080p one) as before: the tail, then the tail with the room, the
    # room only under the segments.
    assert [rung_of(clip_plan(free)[0]) for free in (9.0, 8.0, 6.75)] == [0, 1, 2]
    assert rung_of(clip_plan(8.0)[0]) == 1 and clip_plan(8.0)[0].vae_floor is None


def test_the_ladder_by_the_phase_refused_and_the_allocator() -> None:
    ladder = shot_plan._ladder  # pyright: ignore[reportPrivateUsage]
    least = peaks.VAE_ROOM_LEAST
    assert [rung[:2] for rung in ladder("decode", None, True)] == [(True, None), (True, least)]
    assert [rung[:2] for rung in ladder("decode", None, False)] == [(True, None)]
    assert [rung[:2] for rung in ladder("encode", None, True)] == [(False, least), (True, least)]
    assert ladder("encode", None, False) == ()
    # A tail fixed by a resume is no rung; the room alone, under the segments alone.
    assert [rung[:2] for rung in ladder("decode", True, True)] == [(True, least)]
    assert [rung[:2] for rung in ladder("encode", False, True)] == [(False, least)]
    assert ladder("decode", True, False) == () and ladder("decode", False, False) == ()


TAIL_NAMED = "the decoder's tail (its last blocks a frame at a time)"
ROOM_NAMED = "the VAE's least room (the VAE passes' reserve floor lowered to 0.50 GiB)"


def no_rung_compiled(compile_: str) -> str:
    """The refusal's line that --compile takes no rung, with `compile_` as the words say it."""
    return f"{compile_} takes none of the floor's last rungs, not measured under {compile_}"


def test_a_compile_job_only_a_rung_fits_is_refused_saying_it_runs_without_compile() -> None:
    # Under --compile no rung is tried (a compiled decode with the tail, or compiled in the
    # lowered room, is not measured). A job the plain plan fits only with a rung (DESIGN.md,
    # Allocator, Small cards, decided on 2026-11-06) is refused, CompileRefused: the compiled
    # plan's own sentence, a line saying that --compile takes no rung, then that the same command
    # without --compile runs, with the rungs that plan takes, named as the plan's lines name them
    # (the CLI's planner._rungs_said), its facts carrying them.
    least = peaks.VAE_ROOM_LEAST
    for free, tail, floor, rungs in (
        (8.0, True, None, TAIL_NAMED),
        (7.0, True, None, TAIL_NAMED),
        (6.5, True, least, f"{TAIL_NAMED} and {ROOM_NAMED}"),
    ):
        made = clip_plan(free)[0]
        assert (made.tail_frames, made.vae_floor) == (tail, floor), free
        with pytest.raises(PlanError) as refused:
            clip_plan(free, compile_vae=True)
        error = refused.value
        assert type(error) is CompileRefused and error.kind == "compile", free
        assert (error.tail_frames, error.vae_floor, error.within) == (tail, floor, ())
        # The compiled plan's own refusal is the first line, its phase and facts the error's.
        with pytest.raises(PlanError) as compiled:
            clip_plan(free, compile_vae=True, compile_fixed=True)
        first, *rest = str(error).split("\n")
        assert first == str(compiled.value) and error.phase == compiled.value.phase == "decode"
        assert error.instead == compiled.value.instead
        assert rest == [
            no_rung_compiled("compiling"),
            f"The same command without compiling runs, with {rungs}",
        ]
        assert "--" not in str(error)
        assert cli_words(error).split("\n")[1:] == [
            no_rung_compiled("--compile"),
            f"The same command without --compile runs, with {rungs}",
        ]


def test_a_compile_job_only_the_tail_fits_under_the_plain_allocator_names_the_tail_alone() -> None:
    # Under torch's own allocator without the segments the VAE's least room is never a rung
    # (shot_plan.plan's rungs): the refusal names the tail alone, no rung the job could not take.
    with pytest.raises(PlanError) as refused:
        shot_plan.plan(
            segments_at(12.0, memory.PLAIN), HD, shots(19), NO_PINS, None, compile_vae=True
        )
    error = refused.value
    assert type(error) is CompileRefused
    assert (error.tail_frames, error.vae_floor) == (True, None)
    assert cli_words(error).split("\n")[1:] == [
        no_rung_compiled("--compile"),
        f"The same command without --compile runs, with {TAIL_NAMED}",
    ]
    assert "least room" not in cli_words(error)


def test_a_compile_job_no_plain_plan_fits_is_refused_as_it_was_before_the_rungs() -> None:
    # A job the plain plan refuses with every rung too keeps the plain plan's refusal without
    # them (plan's `rungs`), never the last rung's figures or the compiled pass's CompileRefused.
    for free in (5.0, 6.0):
        with pytest.raises(PlanError):
            clip_plan(free)
        with pytest.raises(PlanError) as refused:
            clip_plan(free, compile_vae=True)
        assert type(refused.value) is MemoryShort, free
        with pytest.raises(PlanError) as plain:
            shot_plan.plan(
                segments_at(free), HD, shots(19), NO_PINS, None, rungs=False, compile_fixed=True
            )
        assert str(refused.value) == str(plain.value) and refused.value.phase == plain.value.phase


def test_a_compile_job_whose_pin_is_beyond_the_shapes_keeps_the_plain_refusal() -> None:
    # A rung helps a VAE pass short of memory, not a pin beyond the shapes measured compiled: a
    # 1200x2133 job (past the 1080p frame, whose untiled compiled passes are measured) pinned
    # untiled on a 40 GiB card, which the plain plan fits only with the tail, is
    # refused as before the rungs (the plain plan's PinRefused), not told to drop --compile.
    job = (40 * GIB, (1200, 2133), shots(300), Pins(encode_tile=0, decode_tile=0), None)
    options: dict[str, Any] = {"split": True}
    made, _ = shot_plan.plan(*job, **options)
    assert made.tail_frames
    with pytest.raises(PlanError) as compiled:
        shot_plan.plan(*job, compile_vae=True, **options)
    with pytest.raises(PlanError) as plain:
        shot_plan.plan(*job, rungs=False, compile_fixed=True, **options)
    assert type(compiled.value) is PinRefused and str(compiled.value) == str(plain.value)
    # A window pin the planner rejects once memory no longer refuses it keeps the plain refusal
    # too, the memory's, as before the rungs.
    with pytest.raises(PlanError) as short:
        shot_plan.plan(segments_at(8.0), HD, shots(19), Pins(window=4), None, compile_vae=True)
    assert type(short.value) is MemoryShort


def test_the_rungs_are_named_as_the_plans_lines_name_them() -> None:
    named = shot_plan._rungs_named  # pyright: ignore[reportPrivateUsage]
    least = peaks.VAE_ROOM_LEAST
    assert named(True, None) == TAIL_NAMED
    assert named(True, least) == f"{TAIL_NAMED} and {ROOM_NAMED}"
    assert named(False, least) == ROOM_NAMED


def test_a_refusal_after_every_rung_is_the_last_rungs_and_says_what_it_counted() -> None:
    # DESIGN.md, Allocator, Small cards: where every rung was tried and the plan is still refused,
    # the refusal is the last attempt's, its figures the least the plan takes (less than the
    # plain plan's), its words saying the rungs it counted.
    with pytest.raises(PlanError) as plain:
        clip_plan(6.0, rungs=False)
    with pytest.raises(PlanError) as last:
        clip_plan(6.0)
    assert type(last.value) is type(plain.value) and last.value.phase == plain.value.phase
    shorter, longer = last.value, plain.value
    assert isinstance(shorter, MemoryShort) and isinstance(longer, MemoryShort)
    assert shorter.need is not None and longer.need is not None
    assert shorter.short is not None and longer.short is not None
    assert shorter.need < longer.need and shorter.short < longer.short
    assert str(last.value).endswith(
        ", with the decoder's tail a frame at a time and the VAE passes left 0.50 GiB, the least"
        " the plan takes"
    )
    assert "least the plan takes" not in str(plain.value)
    # A resume's recorded shape without the tail: the room alone is counted, and said.
    with pytest.raises(PlanError) as recorded:
        clip_plan(6.0, recorded=Plan(6, Tiles(512), Tiles(512)))
    assert str(recorded.value).endswith(
        ", with the VAE passes left 0.50 GiB, the least the plan takes"
    )
    # With the tail recorded: both, as a new plan's.
    with pytest.raises(PlanError) as with_tail:
        clip_plan(6.0, recorded=Plan(6, Tiles(512), Tiles(512), tail_frames=True))
    assert str(with_tail.value).endswith(
        ", with the decoder's tail a frame at a time and the VAE passes left 0.50 GiB, the least"
        " the plan takes"
    )
    # A compile job keeps its words, and so does a limit a rung gets the card past (the host's).
    with pytest.raises(PlanError) as compiled:
        clip_plan(6.0, compile_vae=True)
    assert "least the plan takes" not in str(compiled.value)
    with pytest.raises(PlanError) as host:
        shot_plan.plan(segments_at(8.0), HD, shots(19), NO_PINS, 1 * GIB)
    assert host.value.phase == "host" and "least the plan takes" not in str(host.value)

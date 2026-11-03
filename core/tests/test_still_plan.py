"""The plan for one picture (seedvr2x/still_plan.py; DESIGN.md, Still images, "The plan for one
picture"), on the CPU, no model and no torch: its windows against the DiT's own window functions,
the proof that every picture within a bound has a pass that fits, checked by planning pictures
one by one with peaks.py's public figures, the choices in the planner's order, the pins, the
refusals and the system memory. Cards are emulated as the planner's tests have them: a driver that
refuses reports N - 1.49 GiB free on a card of N GiB (DESIGN.md, Memory planner, Budget)."""

import mmap
import random
import re
import subprocess
import sys
import time
from dataclasses import replace

import pytest

from seedvr2x import memory, peaks, still_plan
from seedvr2x.frames import check_target, output_size, padded_size
from seedvr2x.memory import Reading
from seedvr2x.passes import ModelPlan, Tiles
from seedvr2x.peaks import GIB, VAE_ROOM
from seedvr2x.still_plan import PictureTiles, Pins, PlanRefusal, StillPlan
from seedvr2x.vendor.models.dit_7b.window import (
    make_720Pwindows_bysize,
    make_shifted_720Pwindows_bysize,
)
from seedvr2x.weights import Architecture

# pyright: reportPrivateUsage=false

EIGHT_K, SIX_BY_FOUR, FOUR_K = (7680, 4320), (6000, 4000), (3840, 2160)
PROCESS = round(2.4 * GIB)  # the planner's HOST_RESIDENT, the process's own system memory


def card(gib: float) -> Reading:
    """What a driver that refuses reports free on an emulated card of `gib` GiB."""
    free = round((gib - 1.49) * GIB)
    return Reading(free, None, free, False)


def held(counted: float, left: float = 0.0) -> Reading:
    """A run held to the card whose plan counts on `counted` GiB (memory.Reading)."""
    count = round(counted * GIB)
    return Reading(count + round(left * GIB), None, count, True, round(left * GIB))


def made(
    bound: still_plan.Bound,
    free: int | Reading,
    host: int | None = None,
    *,
    attention: str = peaks.FLASH_ATTN_2,
    dit_model: Architecture = peaks.FP16,
    split: bool = True,
    pins: Pins = still_plan.NO_PINS,
) -> StillPlan:
    return still_plan.plan(
        bound,
        free,
        host,
        process=PROCESS,
        attention=attention,
        dit_model=dit_model,
        split=split,
        pins=pins,
    )


def found(pattern: str, text: str) -> re.Match[str]:
    match = re.search(pattern, text)
    assert match is not None, (pattern, text)
    return match


# ---- No torch ------------------------------------------------------------------------------------

IMPORTED = """
import sys

from seedvr2x import still_plan
from seedvr2x.peaks import GIB

plan = still_plan.plan((3840, 2160), 31 * GIB, 64 * GIB, process=2 * GIB)
assert still_plan.windows_most((3840, 2160)) == 100
assert still_plan.tiles(plan, (2160, 3840)).decode is not None
assert still_plan.said(plan, (2160, 3840))
print(*sorted(name for name in ("torch", "numpy", "torchvision") if name in sys.modules))
"""


def test_imports_no_torch() -> None:
    # The command line plans before it imports torch (tests/test_cli.py): a process that has made a
    # plan and the tiles of a picture has loaded none, nor numpy.
    result = subprocess.run(
        [sys.executable, "-c", IMPORTED], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


# ---- What a picture within a bound is ------------------------------------------------------------


def test_padding_is_separable() -> None:
    # The rows a padded frame has depend on the height alone, the columns on the width alone
    # (frames.padding), so that a bound's pictures are the pairs of its padded rows and columns
    # (still_plan._rows, _columns), and check_target's two tests are as separable.
    rng = random.Random(7)
    for _ in range(3000):
        height, width = rng.randint(1, 5000), rng.randint(1, 5000)
        rows, columns = padded_size((height, width))
        assert rows == padded_size((height, 16))[0]
        assert columns == padded_size((32, width))[1]
        try:
            check_target((height, width), 0)
            fits = True
        except ValueError:
            fits = False
        assert fits == (still_plan._paddable((height, width)))
        assert fits == (height >= 17 and width >= 16)


def test_a_bounds_padded_sides_are_those_of_every_target_within_it() -> None:
    # Brute force, target by target: output_size within the bound, in the padding's reach.
    for long, short in [(66, 34), (130, 50), (200, 200), (322, 18)]:
        rows: set[int] = set()
        columns: set[int] = set()
        for height in range(1, short + 3):
            for width in range(1, long + 3):
                if min(output_size((height, width))) > short:
                    continue
                if max(output_size((height, width))) > long:
                    continue
                try:
                    check_target((height, width), 0)
                except ValueError:
                    continue
                rows.add(padded_size((height, width))[0])
                columns.add(padded_size((height, width))[1])
        # The landscape orientation, the short side along the rows; the portrait's are the rows of
        # the long side and the columns of the short one.
        assert rows == set(still_plan._rows(short))
        assert columns == set(still_plan._columns(long))


def test_a_target_side_over_an_even_bound_by_one_is_within_it() -> None:
    # frames.output_size crops each side to even: 2161 rows give 2160.
    assert max(still_plan._rows(2160)) == padded_size((2161, 16))[0]
    assert max(still_plan._columns(3840)) == padded_size((32, 3841))[1]
    # A width of 3841 reflects 15 columns where 3840 reflects none: the bound's frame is wider.
    assert padded_size((32, 3841))[1] == 3872 > padded_size((32, 3840))[1] == 3840


# ---- The DiT's windows ---------------------------------------------------------------------------


def vendored(rows: int, columns: int) -> int:
    """The windows of one latent of rows x columns tokens by the vendored window functions."""
    size = (1, rows, columns)
    return max(
        len(make_720Pwindows_bysize(size, peaks.NUM_WINDOWS)),
        len(make_shifted_720Pwindows_bysize(size, peaks.NUM_WINDOWS)),
    )


def test_the_windows_counted_without_lists_are_the_vendored_lengths_on_a_sweep() -> None:
    for rows in range(1, 90):
        for columns in range(1, 90):
            assert still_plan._windows(rows, columns) == vendored(rows, columns), (rows, columns)


def test_the_windows_counted_without_lists_are_the_vendored_lengths_on_random_grids() -> None:
    rng = random.Random(11)
    for _ in range(1500):
        rows, columns = rng.randint(1, 700), rng.randint(1, 700)
        assert still_plan._windows(rows, columns) == vendored(rows, columns), (rows, columns)


def test_the_windows_are_not_monotone_in_the_token_grid() -> None:
    # The figures windows_most's docstring gives: 441 at the most up to 300 x 560 tokens, and 240
    # then 289 windows from one column to the next.
    assert max(still_plan._windows(h, w) for h in range(1, 301) for w in range(1, 561)) == 441
    assert (still_plan._windows(174, 558), still_plan._windows(174, 559)) == (240, 289)
    assert (vendored(174, 558), vendored(174, 559)) == (240, 289)


def test_the_windows_of_strips_are_the_vendored_lengths() -> None:
    # Strips, where a window is 1 token and the shifted layout's first window is empty
    # (int(1/2) = 0), or the last one starts past the axis.
    for rows, columns in [
        (1, 300),
        (1, 700),
        (2, 700),
        (3, 900),
        (700, 1),
        (900, 3),
        (1, 1),
        (5, 5),
    ]:
        assert still_plan._windows(rows, columns) == vendored(rows, columns), (rows, columns)
    assert still_plan._axis_shifted(10, 1) == 10 and still_plan._axis_regular(10, 1) == 10
    assert still_plan._axis_shifted(7, 7) == still_plan._axis_regular(7, 9) == 1


def brute_force(bound: still_plan.Bound) -> int:
    """The most windows of any picture within the bound, target by target through the padding and
    the vendored window functions (peaks.attention_layouts)."""
    long, short = bound
    padded: set[tuple[int, int]] = set()
    for height in range(1, long + 2):
        for width in range(1, long + 2):
            out = output_size((height, width))
            if min(out) > short or max(out) > long:
                continue
            try:
                check_target((height, width), 0)
            except ValueError:
                continue
            padded.add(padded_size((height, width)))
    return max(peaks.attention_windows(each, 1) for each in padded)


@pytest.mark.parametrize("bound", [(66, 34), (130, 50), (200, 120), (250, 250), (322, 18)])
def test_windows_most_is_the_brute_force_over_every_picture(bound: still_plan.Bound) -> None:
    assert still_plan.windows_most(bound) == brute_force(bound)


def test_windows_most_at_8k_is_quick() -> None:
    # 187,000 token grids (both orientations), counted in plain Python: 0.10 s where measured.
    still_plan.windows_most.cache_clear()
    start = time.perf_counter()
    most = still_plan.windows_most(EIGHT_K)
    assert time.perf_counter() - start < 1.0
    assert most == 361


def test_a_smaller_picture_can_take_more_windows_than_the_bound() -> None:
    # The windows are not monotone in the picture's size: a 3000x2000 picture takes 72 attention
    # windows, a 2970x1898 one 81 (a window is 2.726 MiB, peaks.PER_WINDOW).
    bound = (3000, 2000)
    own = max(
        peaks.attention_windows(padded_size((2000, 3000)), 1),
        peaks.attention_windows(padded_size((3000, 2000)), 1),
    )
    smaller = padded_size((1898, 2970))
    assert own == 72 and peaks.attention_windows(smaller, 1) == 81
    assert still_plan.windows_most(bound) == 81
    # The plan holds the DiT's phase for it: its peak is no smaller than that picture's own, and
    # the bound's largest frame's tokens carry the windows of the picture that takes the most.
    plan = made(bound, card(24))
    swapped, vae = plan.models.swapped, plan.models.vae_in_dit
    assert plan.windows == 81
    assert plan.dit_peak >= peaks.dit_peak(smaller, 1, swapped, vae)
    frame = still_plan._most(bound)
    extra = 81 - peaks.attention_windows(frame, 1)
    assert extra > 0
    assert plan.dit_peak - peaks.dit_peak(frame, 1, swapped, vae) == pytest.approx(
        extra * peaks.PER_WINDOW, abs=2
    )


# ---- A plan holds for every picture within its bound ---------------------------------------------


def sides(limit: int, low: int, seed: int) -> list[int]:
    """Target sides from `low` to `limit` + 1: the edges, the padding's steps, the tiles' (512
    px) and a seeded spread."""
    rng = random.Random(seed)
    edges = {low, low + 1, 24, 25, 48, 49, 511, 512, 513, 1023, 1024, 1025, limit - 1, limit}
    spread = {rng.randrange(low, limit + 2) for _ in range(14)}
    return sorted(side for side in edges | spread | {limit + 1} if low <= side <= limit + 1)


def first_fit(plan: StillPlan, target: tuple[int, int]) -> PictureTiles:
    """The tiles of a picture checked against peaks.py alone: the first pass of the planner's order
    (untiled, then tiles from the largest down, none covering the frame) whose peak and room fit
    the budget, for each VAE phase under the plan's placement."""
    found = still_plan.tiles(plan, target)
    padded, output = padded_size(target), output_size(target)
    models = plan.models
    largest = min(peaks.TILE_MAX, (max(padded) - 1) // peaks.TILE_STEP * peaks.TILE_STEP)
    options = [None, *(Tiles(size) for size in range(largest, peaks.TILE_MIN - 1, -64))]
    for phase, chosen, resident in (
        ("encode", found.encode, models.resident_encode),
        ("decode", found.decode, models.resident_decode),
    ):
        fits = [
            (
                peaks.encode_peak(padded, each, resident, models.swapped)
                if phase == "encode"
                else peaks.decode_peak(
                    padded, output, each, resident, models.swapped, plan.split, frames=1
                )
            )
            + VAE_ROOM
            <= plan.budget
            for each in options
        ]
        assert any(fits), (phase, target)
        assert options.index(chosen) == fits.index(True), (phase, target, chosen)
    return found


def sweep(plan: StillPlan, seed: int) -> int:
    """First-fit checks over the pictures within the plan's bound, both orientations."""
    long, short = plan.bound
    count = 0
    for height in sides(short, 17, seed):
        for width in sides(long, 17, seed + 1):
            for target in ((height, width), (width, height)):
                first_fit(plan, target)
                count += 1
    return count


@pytest.mark.parametrize(
    ("bound", "gib"),
    [
        (EIGHT_K, 24),
        (EIGHT_K, 32),
        (EIGHT_K, 48),
        (SIX_BY_FOUR, 16),
        (SIX_BY_FOUR, 24),
        (SIX_BY_FOUR, 32),
        (SIX_BY_FOUR, 48),
    ],
)
def test_every_picture_within_the_bound_gets_tiles_that_fit(
    bound: still_plan.Bound, gib: int
) -> None:
    plan = made(bound, card(gib), 64 * GIB)
    assert sweep(plan, seed=gib) > 500
    assert plan.models.compile_vae is False


def test_pictures_much_smaller_than_the_bound_get_their_own_tiles() -> None:
    # Map's cases: a 6000x4000 picture under an 8K bound, and a 2000x2000 one. On a 32 GB card the
    # 8K bound's encode is in tiles of 1728 px and its decode in 1280; the 6000x4000 picture takes
    # the same, the 1080p one an untiled encode and a decode in 1472.
    plan = made(EIGHT_K, card(32))
    assert plan.models.resident_encode == 2 and plan.models.resident_decode == 3
    for target in [(4320, 7680), (4000, 6000), (6000, 4000), (2000, 2000)]:
        found = first_fit(plan, target)
        assert (found.encode, found.decode) == (Tiles(1728), Tiles(1280))
    smaller = first_fit(plan, (1080, 1920))
    assert (smaller.encode, smaller.decode) == (None, Tiles(1472))
    tiny = first_fit(plan, (270, 480))
    assert (tiny.encode, tiny.decode) == (None, None)


def test_the_largest_pictures_tiles_are_not_every_pictures() -> None:
    # DESIGN.md: a bound's largest picture does not stand for the others. On a 24 GB card a 4K
    # bound's encode is in tiles of 1408 px and a 1080p picture within it takes tiles of 1792 px,
    # the DiT's two blocks kept for the encode being the placement of the bound.
    plan = made(FOUR_K, card(24))
    assert plan.models.resident_encode == 2
    assert still_plan.tiles(plan, (2160, 3840)).encode == Tiles(1408)
    assert still_plan.tiles(plan, (1080, 1920)).encode == Tiles(1792)
    # And a bound's own two orientations take other tiles: measured peaks are by rows and columns.
    low = made(FOUR_K, card(16))
    assert still_plan.tiles(low, (2160, 3840)).decode == Tiles(768)
    assert still_plan.tiles(low, (3840, 2160)).decode == Tiles(704)


def test_the_whole_card_gets_the_untiled_pass_where_it_fits() -> None:
    plan = made((1920, 1080), card(96), 128 * GIB)
    found = still_plan.tiles(plan, (1080, 1920))
    assert found.encode is None and found.decode is None
    assert plan.models == ModelPlan(0, None, None, True)
    assert plan.pinned == 0 and plan.host == PROCESS
    # 4K: the untiled encode fits (91.5 GiB) and the decode is in tiles of 2048 px.
    four_k = still_plan.tiles(made(FOUR_K, card(96)), (2160, 3840))
    assert four_k.encode is None and four_k.decode == Tiles(2048)


# ---- The placement lowered by the proof ----------------------------------------------------------


def every_target(bound: still_plan.Bound) -> list[tuple[int, int]]:
    """Every picture within the bound, both orientations."""
    long, short = bound
    landscape = [(h, w) for h in range(17, short + 2) for w in range(16, long + 2)]
    return landscape + [(h, w) for h in range(17, long + 2) for w in range(16, short + 2)]


def extreme_targets(bound: still_plan.Bound) -> list[tuple[int, int]]:
    """For every padded frame a picture within the bound is cut to, the picture that gives it the
    largest output, both orientations: a VAE pass depends on the picture by its padded frame and,
    with the colour correction, its output's area, which grows with the target, so that these
    have every picture's worst case. Found target by target, through frames.padded_size."""
    long, short = bound

    def largest(side: int, low: int, padded: tuple[int, int], axis: int) -> list[int]:
        found: dict[int, int] = {}
        for target in range(low, side + 2):
            if target - target % 2 <= side:
                found[
                    padded_size((target, padded[1]) if axis == 0 else (padded[0], target))[axis]
                ] = target
        return sorted(found.values())

    rows_short = largest(short, 17, (0, 16), 0)
    rows_long = largest(long, 17, (0, 16), 0)
    columns_short = largest(short, 16, (32, 0), 1)
    columns_long = largest(long, 16, (32, 0), 1)
    return [(h, w) for h in rows_short for w in columns_long] + [
        (h, w) for h in rows_long for w in columns_short
    ]


def unfit(
    plan: StillPlan, targets: list[tuple[int, int]], first: bool = False
) -> list[tuple[int, int]]:
    """The pictures that have no pass under the plan: tiles asserts on them (the proof is wrong);
    the first alone with `first`."""
    found: list[tuple[int, int]] = []
    for target in targets:
        try:
            still_plan.tiles(plan, target)
        except AssertionError:
            found.append(target)
            if first:
                break
    return found


def test_every_picture_of_a_small_bound_has_tiles_that_fit() -> None:
    # Target by target, 12,000 pictures; the extreme targets are the same bound's worst cases.
    plan = made((100, 60), held(4.0))
    assert unfit(plan, every_target(plan.bound)) == []
    assert set(extreme_targets(plan.bound)) <= set(every_target(plan.bound))
    assert unfit(plan, extreme_targets(plan.bound)) == []


def test_every_padded_frame_of_a_720p_bound_has_a_pass_whatever_the_pins() -> None:
    # The review's check (2026-11-03): every padded frame a picture within the bound is cut to,
    # with its largest output, both orientations (extreme_targets: the worst case of every
    # picture), under budgets from tight to roomy and the pins that change the proof; a plan
    # refused at a budget proves nothing there, and at least one budget a configuration is
    # planned at is asked of each.
    bound = (1280, 720)
    targets = extreme_targets(bound)
    pinned = [
        Pins(),
        Pins(encode_tile=512),
        Pins(decode_tile=768),
        Pins(encode_tile=1024, decode_tile=640),
        Pins(swap_blocks=36),
    ]
    for pins in pinned:
        planned = 0
        for gib in (5.0, 6.0, 8.0, 12.0, 24.0):
            try:
                plan = made(bound, held(gib), pins=pins)
            except still_plan.PlanRefusal:
                continue
            planned += 1
            assert unfit(plan, targets) == [], (pins, gib)
        assert planned, pins


def test_the_residents_are_lowered_until_every_picture_has_a_pass() -> None:
    # --encode-tile 512 on a 1080p bound with 8 GiB to count on: the bound's own picture leaves the
    # encode 5 blocks of the DiT on the GPU in its tiles; the tiles of a picture within the bound
    # that cuts off in a shape the bound's grid doesn't have (a cut-off tile is counted at its
    # rule plus 0.3 GiB, DESIGN.md) take more, and are pinned to 512 px: with 4 blocks the
    # picture has no pass. The plan keeps 3.
    bound = (1920, 1080)
    plan = made(bound, held(8.0), pins=Pins(encode_tile=512))
    assert plan.models.swapped == 23 and plan.models.resident_encode == 3
    job = still_plan._job_of(plan)
    own = [
        still_plan._resident(job, "encode", frame, (1080, 1920), plan.models.swapped)
        for frame in still_plan._frames(bound)
    ]
    assert own == [5, 5]
    targets = extreme_targets(bound)
    assert unfit(plan, targets) == []
    for kept in (4, 5):
        unlowered = replace(plan, models=replace(plan.models, resident_encode=kept))
        assert unfit(unlowered, targets, first=True), kept
    # The decode, unpinned: its own picture's 0 stands.
    assert plan.models.resident_decode == 0


def test_an_unpinned_phase_is_rarely_lowered() -> None:
    # An unpinned picture takes the less of the untiled pass and of 512-px tiles, which the
    # bound's own pass is, so that the bound's choice stands on the cards the planner's tests use.
    for bound, gib in [(FOUR_K, 16), (FOUR_K, 24), (SIX_BY_FOUR, 32), ((1920, 1080), 12)]:
        plan = made(bound, card(gib))
        job = still_plan._job_of(plan)
        for phase in ("encode", "decode"):
            own = min(
                (
                    still_plan._resident(
                        job, phase, frame, (bound[1], bound[0]), plan.models.swapped
                    )
                    for frame in still_plan._frames(bound)
                ),
                key=lambda each: (still_plan._level(each, plan.models.swapped), each is None),
            )
            assert own == getattr(plan.models, f"resident_{phase}"), (bound, gib, phase)


# ---- The choices, in the planner's order ---------------------------------------------------------


def test_the_fewest_swapped_blocks_with_the_vae_leaving_first() -> None:
    # peaks.dit_peak's order: no block swapped, the VAE on the GPU then off, then 3 to 36 blocks
    # (one or two save nothing). At 4K, one latent: 20 GiB with the VAE there.
    assert made(FOUR_K, card(96)).models.swapped == 0
    plan = made(FOUR_K, card(24))
    assert (plan.models.swapped, plan.models.vae_in_dit) == (0, True)
    plan = made(FOUR_K, card(22))
    assert (plan.models.swapped, plan.models.vae_in_dit) == (0, False)
    plan = made(FOUR_K, card(16))
    assert plan.models.swapped > 2 and not plan.models.vae_in_dit
    for swapped in range(plan.models.swapped):
        if swapped in (1, 2):
            continue
        for vae in (True, False):
            if swapped == plan.models.swapped and vae == plan.models.vae_in_dit:
                continue
            least = peaks.dit_least(still_plan._most(FOUR_K), 1, swapped, vae, plan.attention)
            assert round(least + peaks.PER_WINDOW * plan.windows) > plan.budget
    assert plan.dit_peak <= plan.budget


def test_the_dit_phase_counts_the_most_tokens_and_the_most_windows() -> None:
    plan = made(EIGHT_K, card(48))
    frame = still_plan._most(EIGHT_K)
    expected = round(
        peaks.dit_least(frame, 1, plan.models.swapped, plan.models.vae_in_dit, plan.attention)
        + peaks.PER_WINDOW * still_plan.windows_most(EIGHT_K)
    )
    assert plan.dit_peak == expected == peaks.dit_peak(frame, 1, 0, True)
    assert plan.windows == 361


def test_the_vae_passes_hold_their_room_under_the_budget_at_the_bounds_own_picture() -> None:
    for gib in (24, 32, 48):
        plan = made(EIGHT_K, card(gib))
        found = first_fit(plan, (4320, 7680))
        assert found.encode_peak + VAE_ROOM <= plan.budget
        assert found.decode_peak + VAE_ROOM <= plan.budget


def test_the_budget_has_the_margin_on_a_driver_that_refuses_and_none_on_a_held_run() -> None:
    refusing = made(FOUR_K, card(24))
    assert (
        refusing.margin == peaks.MARGIN
        and refusing.budget == refusing.reading.counted - peaks.MARGIN
    )
    on_card = made(FOUR_K, held(22))
    assert on_card.margin == 0 and on_card.budget == on_card.reading.counted


def test_int8_needs_less_and_sdpa_more() -> None:
    fp16 = made(EIGHT_K, card(24))
    int8 = made(EIGHT_K, card(24), dit_model=peaks.INT8)
    sdpa = made(EIGHT_K, card(24), attention=peaks.SDPA)
    assert int8.models.swapped < fp16.models.swapped
    assert int8.pinned < fp16.pinned
    assert sdpa.models.swapped >= fp16.models.swapped
    same = made(EIGHT_K, card(96))
    assert made(EIGHT_K, card(96), attention=peaks.SDPA).dit_peak > same.dit_peak
    assert made(EIGHT_K, card(96), dit_model=peaks.INT8).dit_peak < same.dit_peak
    # The DiT's floor, every block swapped (the two slots stay): 18.52 GiB of free VRAM at 8K, 18.48
    # in int8, and 19.40 under SDPA.
    free = Reading(round(19 * GIB), None, round(19 * GIB), False)
    made(EIGHT_K, free)
    made(EIGHT_K, free, dit_model=peaks.INT8)
    with pytest.raises(PlanRefusal) as refused:
        made(EIGHT_K, free, attention=peaks.SDPA)
    assert refused.value.phase == "dit"


def test_the_colour_correction_term_is_counted_when_the_decode_splits() -> None:
    # An untiled decode corrects each slice while it holds its working memory: its peak is the
    # two together, one frame's term at the picture's output (0.56 GiB at 4K, peaks.SPLIT_FRAME_4K),
    # which 1080p's quarter of the area scales.
    free = card(96)
    with_split = still_plan.tiles(made((1920, 1080), free, split=True), (1080, 1920))
    without = still_plan.tiles(made((1920, 1080), free, split=False), (1080, 1920))
    assert with_split.decode is None and without.decode is None
    term = peaks.SPLIT_FRAME_4K * 1080 * 1920 / peaks.PIXELS_4K
    assert with_split.decode_peak - without.decode_peak == pytest.approx(term, abs=2)
    # A tiled decode takes the larger of its tile and the term, so the term is the tile's here.
    tiled = still_plan.tiles(made(EIGHT_K, card(24), split=True), (4320, 7680))
    plain = still_plan.tiles(made(EIGHT_K, card(24), split=False), (4320, 7680))
    assert tiled.decode == plain.decode and tiled.decode is not None
    assert tiled.decode_peak == plain.decode_peak


# ---- The floor -----------------------------------------------------------------------------------


def test_the_floor_refusal_names_the_phase_what_it_takes_and_the_largest_bound_that_fits() -> None:
    with pytest.raises(PlanRefusal) as refused:
        made(EIGHT_K, card(16))
    words = str(refused.value)
    assert refused.value.phase == "dit"
    assert "pictures within 7680x4320 take" in words and "GiB of free VRAM" in words
    assert "the driver reports 14.51 GiB free" in words
    fits = refused.value.fits
    assert fits is not None and f"the largest bound that fits there is {fits[0]}x{fits[1]}" in words
    # The planner's walk down, linearly: the short side in steps of 16 from the multiple under it,
    # the long side in proportion; the first that plans is the refusal's, and the one before not.
    steps = []
    for side in range((EIGHT_K[1] - 1) // 16 * 16, 0, -16):
        steps.append((side * EIGHT_K[0] // EIGHT_K[1] // 2 * 2, side))
    index = steps.index(fits)
    made(fits, card(16))
    with pytest.raises(PlanRefusal):
        made(steps[index - 1], card(16))


def test_the_floor_of_a_vae_phase_is_the_proof_at_the_smallest_tiles() -> None:
    # 1080p with 7.5 GiB free: the DiT's floor is lower, the decode's 512-px tiles take 6.41 GiB
    # at the most over the pictures within the bound (the colour correction's term at the most
    # an output gives), and the room every VAE pass is held to 1.25 more.
    free = Reading(round(7.5 * GIB), None, round(7.5 * GIB), False)
    with pytest.raises(PlanRefusal) as refused:
        made((1920, 1080), free)
    assert refused.value.phase == "decode"
    assert "its decode in tiles of 512 px or untiled, with the DiT off the GPU" in str(
        refused.value
    )
    assert "1.25 of room a VAE pass is held to, and 0.60 of margin" in str(refused.value)
    totals = re.search(
        r"take (\d+\.\d+) GiB of free VRAM \(.*a peak of (\d+\.\d+) GiB, (\d+\.\d+)",
        str(refused.value),
    )
    assert totals is not None
    total, peak, room = (float(each) for each in totals.groups())
    assert total == pytest.approx(peak + room + 0.6, abs=0.011)


def test_no_output_size_fits_in_a_tiny_card() -> None:
    with pytest.raises(PlanRefusal) as refused:
        made(FOUR_K, Reading(GIB, None, GIB, False))
    assert refused.value.fits is None and "no output size fits there" in str(refused.value)


def test_a_bound_that_is_no_output_size_is_a_value_error() -> None:
    for bound in [(3841, 2160), (3840, 2161), (2160, 3840), (3840, 16)]:
        with pytest.raises(ValueError, match="a bound of"):
            made(bound, card(96))


# ---- Pins ----------------------------------------------------------------------------------------


def test_swap_blocks_pinned_is_taken_exactly() -> None:
    free = card(48)
    plan = made(FOUR_K, free, pins=Pins(swap_blocks=12))
    assert plan.models.swapped == 12 and plan.pins.swap_blocks == 12
    # The VAE stays if it fits with them, leaves if only that fits.
    assert made(FOUR_K, free, pins=Pins(swap_blocks=0)).models.vae_in_dit
    assert not made(FOUR_K, card(22), pins=Pins(swap_blocks=0)).models.vae_in_dit
    assert made(FOUR_K, free, pins=Pins(swap_blocks=36)).models.swapped == 36


def test_swap_blocks_pinned_that_does_not_hold_the_dit_is_refused_naming_the_fewest() -> None:
    with pytest.raises(PlanRefusal) as refused:
        made(FOUR_K, card(20), pins=Pins(swap_blocks=0))
    words = str(refused.value)
    assert refused.value.phase == "dit"
    assert (
        "--swap-blocks 0: the DiT's phase, one latent of the largest picture within 3840x2160"
        in words
    )
    assert (
        "takes 20.30 GiB of free VRAM with 0 blocks swapped, and the driver reports 18.51" in words
    )
    fewest = made(FOUR_K, card(20)).models.swapped
    assert fewest == 7 and f"the fewest blocks that hold it are --swap-blocks {fewest}" in words
    assert made(FOUR_K, card(20), pins=Pins(swap_blocks=fewest)).models.swapped == fewest


def test_swap_blocks_above_the_dits_blocks_is_refused_before_memory() -> None:
    with pytest.raises(PlanRefusal, match="the DiT has 36 blocks, so from 0 to 36") as refused:
        made(FOUR_K, Reading(0, None, 0, False), pins=Pins(swap_blocks=37))
    assert refused.value.phase == "pin"


def test_a_tile_pin_that_covers_the_whole_padded_frame_is_refused() -> None:
    biggest = max(still_plan._frames(FOUR_K), key=max)
    pin = (max(biggest) + 63) // 64 * 64
    with pytest.raises(PlanRefusal) as refused:
        made(FOUR_K, card(96), pins=Pins(decode_tile=pin))
    height, width = biggest
    assert str(refused.value) == (
        f"--decode-tile {pin}: a tile of {pin} px covers the whole padded frame, {width}x{height}:"
        " --decode-tile 0 pins an untiled decode"
    )
    assert refused.value.phase == "pin"
    with pytest.raises(PlanRefusal, match="--encode-tile 4096"):
        made(FOUR_K, Reading(0, None, 0, False), pins=Pins(encode_tile=4096))


def test_a_tile_pin_is_proved_for_its_own_tiles_and_smaller_pictures_are_untiled() -> None:
    plan = made(SIX_BY_FOUR, card(48), pins=Pins(decode_tile=1024, encode_tile=1536))
    big = still_plan.tiles(plan, (4000, 6000))
    assert big.decode == Tiles(1024) and big.encode == Tiles(1536)
    # A picture whose padded frame the pin covers is no tiling: the untiled pass; the encode's
    # 1536 px covers a 1200x1400 picture's frame, the decode's 1024 px doesn't.
    small = still_plan.tiles(plan, (600, 1000))
    assert small.decode is None and small.encode is None
    middle = still_plan.tiles(plan, (1200, 1400))
    assert middle.decode == Tiles(1024) and middle.encode is None
    for target in [(4000, 6000), (600, 1000), (1200, 1400), (3000, 4000), (4000, 3000)]:
        found = still_plan.tiles(plan, target)
        padded, output = padded_size(target), output_size(target)
        models = plan.models
        assert (
            peaks.decode_peak(
                padded, output, found.decode, models.resident_decode, models.swapped, True, frames=1
            )
            + VAE_ROOM
            <= plan.budget
        )


def test_a_tile_pin_of_0_is_the_untiled_pass_for_every_picture() -> None:
    plan = made((1920, 1080), card(48), pins=Pins(encode_tile=0, decode_tile=0))
    for target in [(1080, 1920), (1920, 1080), (270, 480), (1081, 1921)]:
        found = still_plan.tiles(plan, target)
        assert found.encode is None and found.decode is None
    # 1080p untiled in the decode takes 38.5 GiB of free VRAM, more than 32 GB cards have.
    with pytest.raises(PlanRefusal, match="--decode-tile 0: the decode untiled") as refused:
        made((1920, 1080), card(32), pins=Pins(decode_tile=0))
    assert refused.value.phase == "pin"
    assert "the largest tile that fits is --decode-tile " in str(refused.value)


def test_a_tile_pin_that_does_not_fit_names_the_largest_tile_that_does() -> None:
    free = card(32)
    with pytest.raises(PlanRefusal) as refused:
        made(FOUR_K, free, pins=Pins(encode_tile=2048))
    words = str(refused.value)
    assert words.startswith(
        "--encode-tile 2048: the encode in tiles of 2048 px, for the pictures within 3840x2160"
    )
    assert refused.value.phase == "pin"
    largest = int(found(r"the largest tile that fits is --encode-tile (\d+)", words).group(1))
    # That tile is taken, the next size up refused.
    plan = made(FOUR_K, free, pins=Pins(encode_tile=largest))
    assert still_plan.tiles(plan, (2160, 3840)).encode == Tiles(largest)
    with pytest.raises(PlanRefusal):
        made(FOUR_K, free, pins=Pins(encode_tile=largest + 64))


def test_a_pin_that_no_tile_serves_names_the_untiled_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    # Where the bound's pictures are all too small for a tile (their padded frames within 512 px),
    # the planner's sizes are none: with the cut-off tiles made dear, a pin of 496 px fails where
    # the untiled pass, which the floor proves, fits.
    monkeypatch.setattr(peaks, "CUT_OFF_EXTRA", 40 * GIB)
    still_plan._worst.cache_clear()
    try:
        assert max(max(frame) for frame in still_plan._frames((480, 320))) <= 512
        with pytest.raises(PlanRefusal) as refused:
            made((480, 320), card(48), pins=Pins(decode_tile=496))
        assert "no tile fits, the untiled decode does, --decode-tile 0" in str(refused.value)
    finally:
        monkeypatch.undo()
        still_plan._worst.cache_clear()


# ---- The system memory ---------------------------------------------------------------------------


def test_the_host_figure_is_the_process_the_pinned_copies_and_the_scratch() -> None:
    plan = made(EIGHT_K, card(32), 64 * GIB)
    scratch = max(peaks.scratch(frame, 1) for frame in still_plan._frames(EIGHT_K))
    assert plan.host == PROCESS + plan.pinned + scratch and scratch > 0
    # Pinned: each part page-rounded; 8 swapped blocks and what the VAE phases send away.
    blocks = plan.models.swapped
    assert still_plan._leaving(plan.models) >= blocks
    parts = [peaks.BLOCK] * still_plan._leaving(plan.models) + [peaks.VAE_WEIGHTS]
    assert plan.pinned == sum(-(-part // mmap.PAGESIZE) * mmap.PAGESIZE for part in parts)
    # Untiled passes everywhere: no scratch.
    assert made((1920, 1080), card(96), 64 * GIB).host == PROCESS


def test_the_vae_is_kept_with_more_swapped_blocks_when_that_pins_less_and_fits_the_host() -> None:
    free = card(24)
    base = made(EIGHT_K, free, 64 * GIB)
    assert not base.models.vae_in_dit and base.models.swapped == 27
    tight = made(EIGHT_K, free, base.host - 1)
    assert tight.models.vae_in_dit and tight.models.swapped > base.models.swapped
    assert tight.host < base.host and tight.host <= base.host - 1
    # Never under --swap-blocks, taken exactly.
    pinned = made(EIGHT_K, free, base.host - 1 + 64 * GIB, pins=Pins(swap_blocks=27))
    assert pinned.models.swapped == 27


def test_more_of_the_dit_stays_on_the_gpu_in_the_vae_phases_when_the_host_is_short() -> None:
    free = card(32)
    base = made(FOUR_K, free, 64 * GIB)
    assert base.models.vae_in_dit
    short = made(FOUR_K, free, round(16.0 * GIB))
    assert short.host <= round(16.0 * GIB) < base.host
    assert short.pinned < base.pinned
    for phase in ("encode", "decode"):
        kept, before = (
            still_plan._level(getattr(each.models, f"resident_{phase}"), each.models.swapped)
            for each in (short, base)
        )
        assert kept > before
    assert short.models.swapped == base.models.swapped
    # What it keeps is proved for every picture: the tiles still exist for them.
    assert sweep(short, seed=5) > 100
    # The smallest that fits: one block fewer pinned would not.
    fewer = made(FOUR_K, free, short.host - 1)
    assert fewer.pinned < short.pinned


def test_a_host_that_cannot_hold_the_plan_is_refused_with_its_parts_adding_up() -> None:
    with pytest.raises(PlanRefusal) as refused:
        made(EIGHT_K, card(24), round(12 * GIB))
    words = str(refused.value)
    assert refused.value.phase == "host"
    numbers = [
        float(each)
        for each in re.findall(r"(\d+\.\d+) (?:for the process|pinned for|for a tiled pass)", words)
    ]
    total = float(found(r"takes (\d+\.\d+) GiB of system memory", words).group(1))
    assert len(numbers) == 3 and total == pytest.approx(sum(numbers), abs=0.001)
    assert "12.00 GiB are available" in words and "2.41 GiB short" in words


def test_the_host_is_unchecked_when_it_is_unread() -> None:
    assert made(EIGHT_K, card(24), None).host_available is None


# ---- A held run and the tight reserve ------------------------------------------------------------


def test_the_tight_reserve_is_taken_where_the_usual_one_is_refused_and_said() -> None:
    # 11 GiB reported, 0.5 held back: the plan counts on 6 with the reserve of 4.5, 8 with 2.5.
    reading = Reading(round(11 * GIB), None, round(6 * GIB), True, round(0.5 * GIB))
    with pytest.raises(PlanRefusal):
        made((1920, 1080), replace(reading, expandable=False))
    plan = made((1920, 1080), reading)
    assert plan.reading.tight and plan.reading.counted == round(8 * GIB)
    assert plan.margin == 0 and plan.budget == plan.reading.counted
    assert plan.tightened.startswith("pictures within 1920x1080 take 7.67 GiB of free VRAM")
    assert sweep(plan, seed=2) > 100
    lines = still_plan.said(plan)
    assert any("reserve: 2.50 GiB and not 4.50" in line for line in lines)


def test_expandable_segments_off_refuses_saying_the_tight_reserve_would_run_it() -> None:
    reading = Reading(
        round(11 * GIB), None, round(6 * GIB), True, round(0.5 * GIB), expandable=False
    )
    with pytest.raises(PlanRefusal) as refused:
        made((1920, 1080), reading)
    words = str(refused.value)
    assert "expandable segments" in words and "reserve of 2.50 GiB" in words
    assert refused.value.phase == "decode"


def test_refused_with_both_reserves_says_both() -> None:
    reading = Reading(round(8 * GIB), None, round(3 * GIB), True, round(0.5 * GIB))
    with pytest.raises(PlanRefusal) as refused:
        made((1920, 1080), reading)
    words = str(refused.value)
    assert "refused with the reserve of 4.50 GiB" in words and "and with 2.50" in words


def test_a_tight_reading_is_not_planned_again() -> None:
    tight = memory.tightened(Reading(round(8 * GIB), None, round(3 * GIB), True, round(0.5 * GIB)))
    with pytest.raises(PlanRefusal) as refused:
        made((1920, 1080), tight)
    assert "refused with the reserve" not in str(refused.value)


def test_pins_that_need_no_memory_are_refused_even_when_the_run_is_held() -> None:
    reading = Reading(round(11 * GIB), None, round(6 * GIB), True, round(0.5 * GIB))
    with pytest.raises(PlanRefusal, match="the DiT has 36 blocks"):
        made((1920, 1080), reading, pins=Pins(swap_blocks=40))


# ---- tiles() and the refusal of a target outside the bound ---------------------------------------


def test_a_target_outside_the_bound_is_a_value_error() -> None:
    plan = made(FOUR_K, card(32))
    with pytest.raises(ValueError, match="not within the plan's bound"):
        still_plan.tiles(plan, (2162, 3840))
    with pytest.raises(ValueError, match="not within the plan's bound"):
        still_plan.tiles(plan, (3000, 3000))
    with pytest.raises(ValueError, match="too small for the padding"):
        still_plan.tiles(plan, (16, 100))
    with pytest.raises(ValueError, match="too small for the padding"):
        still_plan.tiles(plan, (100, 15))


def test_a_target_one_pixel_over_an_even_side_is_within_the_bound() -> None:
    # frames.output_size crops 2161x3841 to 2160x3840, 3841x2161 likewise.
    plan = made(FOUR_K, card(32))
    for target in [(2161, 3841), (3841, 2161), (2161, 3840), (2160, 3841), (3840, 2160)]:
        first_fit(plan, target)
    for target in [(2162, 3841), (3841, 2162), (2161, 3842)]:
        with pytest.raises(ValueError):
            still_plan.tiles(plan, target)


# ---- The plan said -------------------------------------------------------------------------------


def test_said_without_a_target_gives_the_plan_in_the_planners_words() -> None:
    plan = made(EIGHT_K, card(24), 64 * GIB)
    lines = still_plan.said(plan)
    text = "\n".join(lines)
    assert all(line.startswith("plan: ") for line in lines)
    assert "plan: memory: 22.51 GiB free, a budget of 21.91 GiB: the margin of 0.60 GiB" in text
    assert (
        "bound: pictures within 7680x4320 in either orientation, padded to at most 7712x4352"
        in text
    )
    assert "attention: FlashAttention 2" in text
    assert f"blocks swapped: {plan.models.swapped} of the DiT's 36" in text
    assert "the VAE: off the GPU in the DiT's phase, its 0.47 GiB given to the DiT" in text
    assert f"{plan.dit_peak / GIB:.2f} GiB at the most" in text and f"{plan.windows}" in text
    assert (
        "in the encode, 2 of the DiT's blocks on the GPU; in the decode, the DiT off the GPU"
        in text
    )
    assert (
        "system memory: " in text
        and "pinned for 36 blocks, the DiT's other parameters and the VAE" in text
    )
    assert "picture:" not in text and "decode: " not in text.replace("in the decode: ", "")


def test_said_with_a_target_gives_the_pictures_passes_and_their_peaks() -> None:
    plan = made(EIGHT_K, card(32), 64 * GIB)
    big = "\n".join(still_plan.said(plan, (4320, 7680)))
    found = still_plan.tiles(plan, (4320, 7680))
    assert found.decode is not None and found.encode is not None
    assert f"decode: tiles of {found.decode.size} px overlapping by 64" in big
    assert f"encode: tiles of {found.encode.size} px overlapping by 64" in big
    assert f"peak {found.decode_peak / GIB:.2f} GiB" in big
    assert (
        "the VAE's room: what each VAE pass leaves free" in big and "1.25 GiB of the budget" in big
    )
    small = "\n".join(still_plan.said(plan, (270, 480)))
    assert "encode: untiled, the whole frame fitting the budget" in small
    grid = re.search(r"decode: tiles of \d+ px overlapping by 64, (\d+) across, (\d+) down", big)
    assert grid is not None
    columns, rows = (int(each) for each in grid.groups())
    assert (rows, columns) == peaks.grid_shapes(padded_size((4320, 7680)), found.decode)


def test_said_names_the_pins_and_the_formats() -> None:
    plan = made(FOUR_K, card(48), pins=Pins(decode_tile=1024, swap_blocks=12), dit_model=peaks.INT8)
    text = "\n".join(still_plan.said(plan, (2160, 3840)))
    assert "pinned by --swap-blocks" in text and "pinned by --decode-tile" in text
    assert "DiT: its weights in int8" in text
    sdpa = "\n".join(still_plan.said(made(FOUR_K, card(48), attention=peaks.SDPA)))
    assert "attention: PyTorch's SDPA, FlashAttention 2 not being installed" in sdpa


def test_said_on_a_held_run_names_the_figures() -> None:
    reading = Reading(round(26 * GIB), round(25 * GIB), round(20 * GIB), True, round(0.5 * GIB))
    text = "\n".join(still_plan.said(made(FOUR_K, reading)))
    assert (
        "25.00 GiB free on the whole card by NVML: the run is held to the card's own memory" in text
    )
    assert "budget: 20.00 GiB, the 25.00 GiB free less torch's limit, 0.50 GiB under it" in text
    assert "the reserve of 4.50 GiB left to torch's allocator" in text


def test_plan_returns_the_models_the_loaders_follow() -> None:
    plan = made(EIGHT_K, card(32))
    assert isinstance(plan.models, ModelPlan) and not plan.models.compile_vae
    assert plan.bound == EIGHT_K and plan.split and plan.attention == peaks.FLASH_ATTN_2
    assert plan.process == PROCESS and plan.tightened == ""

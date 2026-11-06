"""The plan for one picture (seedvr2x/still_plan.py; DESIGN.md, Still images, "The plan for one
picture"), on the CPU, no model and no torch: its windows against the DiT's own window functions,
the proof that every picture within a bound has a pass that fits, checked by planning pictures
one by one with peaks.py's public figures, the choices in the planner's order, the pins, the
refusals and the system memory. Cards are emulated as the planner's tests have them: a driver that
refuses reports N - 1.49 GiB free on a card of N GiB (DESIGN.md, Memory planner, Budget)."""

import math
import mmap
import random
import re
import subprocess
import sys
import time
from dataclasses import replace
from typing import Any

import pytest
from library_fixtures import cli_words

from seedvr2x import memory, peaks, still_plan
from seedvr2x.errors import (
    HostShort,
    InputTooLarge,
    MemoryShort,
    PinRefused,
    PlanRefused,
    Setting,
)
from seedvr2x.frames import check_target, output_size, padded_size
from seedvr2x.memory import Reading
from seedvr2x.passes import ModelPlan, Phase, Tiles
from seedvr2x.peaks import GIB, VAE_ROOM
from seedvr2x.still_plan import PictureTiles, Pins, PlanRefusal, StillPlan
from seedvr2x.vendor.models.dit_7b.window import (
    make_720Pwindows_bysize,
    make_shifted_720Pwindows_bysize,
)
from seedvr2x.weights import DIT_3B, Architecture

# pyright: reportPrivateUsage=false

EIGHT_K, SIX_BY_FOUR, FOUR_K = (7680, 4320), (6000, 4000), (3840, 2160)
PROCESS = round(2.4 * GIB)  # the planner's HOST_RESIDENT, the process's own system memory
INPUT = 12_192_768  # a 4032x3024 phone photo: the input the helper plans for unless a test says


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
    input_pixels: int = INPUT,
    picture: tuple[int, int] | None = None,
) -> StillPlan:
    return still_plan.plan(
        bound,
        free,
        host,
        process=PROCESS,
        input_pixels=input_pixels,
        attention=attention,
        dit_model=dit_model,
        split=split,
        pins=pins,
        picture=picture,
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

plan = still_plan.plan(
    (3840, 2160), 31 * GIB, 64 * GIB, process=2 * GIB, input_pixels=12_000_000
)
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
    # windows, a 2970x1898 one 81 (a window is 0.996 MiB, peaks.PER_WINDOW).
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


def measured_untiled(phase: str, padded: tuple[int, int]) -> bool:
    """Whether one frame's untiled pass was measured on a frame as large as `padded`, either
    orientation, read from peaks.ONE_FRAME_UNTILED_MEASURED apart from the planner's own test."""
    rows, columns = peaks.ONE_FRAME_UNTILED_MEASURED[0 if phase == "encode" else 1]
    return max(padded) <= max(rows, columns) and min(padded) <= min(rows, columns)


def first_fit(plan: StillPlan, target: tuple[int, int], budget: int | None = None) -> PictureTiles:
    """The tiles of a picture checked against peaks.py alone: the first pass of the planner's order
    (untiled where one frame's untiled pass was measured on such a frame, then tiles from the
    largest down, none covering the frame) whose peak and room fit the budget, for each VAE phase
    under the plan's placement. `budget`: the bytes the passes are held to in place of the plan's
    (still_plan.tiles)."""
    found = still_plan.tiles(plan, target, budget)
    held = plan.budget if budget is None else budget
    padded, output = padded_size(target), output_size(target)
    models = plan.models
    largest = min(peaks.TILE_MAX, (max(padded) - 1) // peaks.TILE_STEP * peaks.TILE_STEP)
    tiled = [Tiles(size) for size in range(largest, peaks.TILE_MIN - 1, -64)]
    for phase, chosen, resident in (
        ("encode", found.encode, models.resident_encode),
        ("decode", found.decode, models.resident_decode),
    ):
        options = [*([None] if measured_untiled(phase, padded) else []), *tiled]
        fits = [
            (
                peaks.picture_encode_peak(padded, each, resident, models.swapped)
                if phase == "encode"
                else peaks.picture_decode_peak(
                    padded, output, each, resident, models.swapped, plan.split
                )
            )
            + VAE_ROOM
            <= held
            for each in options
        ]
        assert any(fits), (phase, target)
        assert options.index(chosen) == fits.index(True), (phase, target, chosen)
    return found


def sweep(plan: StillPlan, seed: int, budget: int | None = None) -> int:
    """First-fit checks over the pictures within the plan's bound, both orientations, their passes
    held to `budget` (None: the plan's)."""
    long, short = plan.bound
    count = 0
    for height in sides(short, 17, seed):
        for width in sides(long, 17, seed + 1):
            for target in ((height, width), (width, height)):
                first_fit(plan, target, budget)
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
    # Map's cases: a 3840x2160 picture under a 4K bound, a 2000x3000 one and a 2000x2000 one. On a
    # 12 GB card the 4K bound's encode is in tiles of 1728 px and its decode in 1280; the smaller
    # pictures whose frames the tiles still cut take the same, the 1080p one an untiled encode and
    # an untiled decode (one frame's passes, measured on the RTX 5090, leave the 6000x4000 and 8K
    # bounds on the largest tile, 2048 px, on every card the plan accepts).
    plan = made(FOUR_K, card(12))
    assert plan.models.resident_encode == 0 and plan.models.resident_decode == 1
    for target in [(2160, 3840), (3840, 2160), (2000, 3000), (3000, 2000), (2000, 2000)]:
        found = first_fit(plan, target)
        assert (found.encode, found.decode) == (Tiles(1728), Tiles(1280))
    smaller = first_fit(plan, (1080, 1920))
    assert (smaller.encode, smaller.decode) == (None, None)
    tiny = first_fit(plan, (270, 480))
    assert (tiny.encode, tiny.decode) == (None, None)
    # On a 10 GB card the 1080p picture is in between: its encode untiled, its decode in tiles.
    low = first_fit(made(FOUR_K, card(10)), (1080, 1920))
    assert (low.encode, low.decode) == (None, Tiles(1088))


def test_the_largest_pictures_tiles_are_not_every_pictures() -> None:
    # DESIGN.md: a bound's largest picture does not stand for the others. On a 12 GB card a 4K
    # bound's encode is in tiles of 1728 px and a 1080p picture within it takes none (its frame is
    # within the tile's reach), the DiT being off the GPU in the encode for the bound.
    plan = made(FOUR_K, card(12))
    assert plan.models.resident_encode == 0
    assert still_plan.tiles(plan, (2160, 3840)).encode == Tiles(1728)
    assert still_plan.tiles(plan, (1080, 1920)).encode is None
    # And a bound's own two orientations take other tiles: measured peaks are by rows and columns
    # (a 1080p bound on a 10.5 GB card, its decode in tiles of 1536 px landscape, 1216 portrait).
    low = made((1920, 1080), card(10.5))
    assert still_plan.tiles(low, (1080, 1920)).decode == Tiles(1536)
    assert still_plan.tiles(low, (1920, 1080)).decode == Tiles(1216)


def test_the_whole_card_gets_the_untiled_pass_where_it_fits() -> None:
    plan = made((1920, 1080), card(96), 128 * GIB)
    found = still_plan.tiles(plan, (1080, 1920))
    assert found.encode is None and found.decode is None
    assert plan.models == ModelPlan(0, None, None, True)
    # Nothing pinned. The host holds the scratch of a tiled pass all the same: a bound's pictures
    # one pixel over an even side pad to 1120x1952, past the 1104x1920 one frame's untiled decode
    # was measured on (peaks.ONE_FRAME_UNTILED_MEASURED), so the proof counts them in tiles.
    scratch = max(peaks.scratch(frame, 1) for frame in still_plan._frames((1920, 1080)))
    assert plan.pinned == 0 and plan.host == PROCESS + scratch and scratch > 0
    assert still_plan.tiles(plan, (1081, 1921)).decode is not None
    # 4K: the untiled encode, measured up to 4K's padded frame (38.1 GiB); the decode is in tiles
    # of 2048 px, 4K being past the 1080p frame its untiled pass was measured on. 8K: both in
    # tiles of 2048 px.
    four_k = still_plan.tiles(made(FOUR_K, card(96)), (2160, 3840))
    assert four_k.encode is None and four_k.decode == Tiles(2048)
    eight_k = still_plan.tiles(made(EIGHT_K, card(96)), (4320, 7680))
    assert eight_k.encode == Tiles(2048) and eight_k.decode == Tiles(2048)


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
    # --encode-tile 512 on a 1080p bound with 6 GiB to count on: the bound's own picture leaves the
    # encode 7 blocks of the DiT on the GPU in its tiles; the tiles of a picture within the bound
    # that cuts off in a shape the bound's grid doesn't have (a cut-off tile is counted at its
    # rule plus 0.3 GiB, DESIGN.md) take more, and are pinned to 512 px: with 7 blocks the
    # picture has no pass. The plan keeps 6. (With 8 GiB, as before one frame's figures, the
    # bound's own picture leaves 11 and the plan keeps 11: the lowering is at 6 to 6.5 GiB.)
    bound = (1920, 1080)
    plan = made(bound, held(6.0), pins=Pins(encode_tile=512))
    assert plan.models.swapped == 26 and plan.models.resident_encode == 6
    job = still_plan._job_of(plan)
    own = [
        still_plan._resident(job, "encode", frame, (1080, 1920), plan.models.swapped)
        for frame in still_plan._frames(bound)
    ]
    assert own == [7, 7]
    targets = extreme_targets(bound)
    assert unfit(plan, targets) == []
    for kept in (7, 8):
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
    # (one or two save nothing). At 4K, one latent: 17.99 GiB with the VAE there (20 on the plain
    # attention).
    assert made(FOUR_K, card(96)).models.swapped == 0
    plan = made(FOUR_K, card(24))
    assert (plan.models.swapped, plan.models.vae_in_dit) == (0, True)
    # Card 22 on the plain attention's figures, 20 on the lean one's (the VAE leaves between
    # 20.5 and 20 GiB).
    plan = made(FOUR_K, card(20))
    assert (plan.models.swapped, plan.models.vae_in_dit) == (0, False)
    assert made(FOUR_K, card(20.5)).models.vae_in_dit
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


def test_int8_needs_less_and_sdpa_the_same() -> None:
    # SDPA planned more than FlashAttention 2 on the plain attention (swapped >=, dit_peak >);
    # on the lean attention the two kernels share one fit, and one plan.
    fp16 = made(EIGHT_K, card(24))
    int8 = made(EIGHT_K, card(24), dit_model=peaks.INT8)
    sdpa = made(EIGHT_K, card(24), attention=peaks.SDPA)
    assert int8.models.swapped < fp16.models.swapped
    assert int8.pinned < fp16.pinned
    assert sdpa.models == fp16.models and sdpa.pinned == fp16.pinned
    same = made(EIGHT_K, card(96))
    assert made(EIGHT_K, card(96), attention=peaks.SDPA).dit_peak == same.dit_peak
    assert made(EIGHT_K, card(96), dit_model=peaks.INT8).dit_peak < same.dit_peak
    # The DiT's floor, every block swapped (the two slots stay): 9.745 GiB of free VRAM at 8K in
    # fp16 and under SDPA, 9.700 in int8 (18.52, 18.48 and 19.40 on the plain attention, whose
    # card of 19 GiB is now far above all three).
    free = Reading(round(9.72 * GIB), None, round(9.72 * GIB), False)
    made(EIGHT_K, free, dit_model=peaks.INT8)
    for attention in (peaks.FLASH_ATTN_2, peaks.SDPA):
        with pytest.raises(PlanRefusal) as refused:
            made(EIGHT_K, free, attention=attention)
        assert refused.value.phase == "dit"
    roomy = Reading(round(9.8 * GIB), None, round(9.8 * GIB), False)
    made(EIGHT_K, roomy)
    made(EIGHT_K, roomy, attention=peaks.SDPA)


@pytest.mark.parametrize("gguf", [peaks.GGUF_7B, peaks.GGUF_7B_SHARP], ids=["7b", "sharp"])
def test_the_gguf_dit_is_planned_by_its_own_blocks(gguf: Architecture) -> None:
    # The dynamic GGUF file's DiT, accepted (peaks.DITS): its weights 4.43 GiB where fp16's take
    # 15.35 and int8's 7.76, so fewer blocks swapped and less pinned than either on the same card.
    # (A card of 24 GiB on the plain attention's figures, where the three still swapped blocks at
    # 8K; the lean attention swaps none of int8's or the GGUF's there, and the ordering shows at
    # 14: 30, 23 and 9 blocks.)
    fp16 = made(EIGHT_K, card(14))
    int8 = made(EIGHT_K, card(14), dit_model=peaks.INT8)
    held = made(EIGHT_K, card(14), dit_model=gguf)
    assert held.models.swapped < int8.models.swapped < fp16.models.swapped
    assert held.pinned < int8.pinned < fp16.pinned
    assert made(EIGHT_K, card(96), dit_model=gguf).dit_peak < made(EIGHT_K, card(96)).dit_peak
    # Its blocks differ (a type per matrix): what is pinned is the sum of the blocks that leave,
    # each page-rounded, the last ones by their own bytes, not as many of the largest.
    sizes = peaks.block_sizes(gguf)
    assert len(set(sizes)) > 1 and sizes[0] != sizes[-1]
    everything = ModelPlan(36, 0, 0, vae_in_dit=False)
    some = ModelPlan(3, None, None, vae_in_dit=True)
    pinned = still_plan._pinned  # pyright: ignore[reportPrivateUsage]

    def pages(*parts: int) -> int:
        return sum(-(-part // mmap.PAGESIZE) * mmap.PAGESIZE for part in parts)

    assert pinned(some, gguf) == pages(*sizes[-3:]) < 3 * pages(max(sizes))
    assert pinned(everything, gguf) == pages(*sizes, peaks.DIT_OTHER, peaks.VAE_WEIGHTS)
    assert f"{pinned(everything, gguf) / GIB:.2f}" == "4.90"
    # A card the DiT could not otherwise fit, 8 GB at 4K: 9 swapped (27 on the plain attention).
    small = made(FOUR_K, card(8), dit_model=gguf)
    assert small.models.swapped == 9 and small.models.vae_in_dit is False
    # The plan says what the file is and what its buffers are counted as.
    text = "\n".join(still_plan.said(made(FOUR_K, card(24), dit_model=gguf)))
    assert (
        "DiT: its weights in seedvr2x's dynamic GGUF, 4.43 GiB, under a third of fp16's; a"
        f" window's memory counted as fp16's plus {peaks.GGUF_BUFFERS / GIB:.2f} GiB for the"
        " dequantisation's buffers; --dit-model names the file"
    ) in text
    # Another architecture is still refused, in words that name the GGUF files.
    with pytest.raises(ValueError, match="for its fp16, int8 and dynamic GGUF files"):
        made(EIGHT_K, card(24), dit_model=DIT_3B)


def test_the_colour_correction_term_is_counted_when_the_decode_splits() -> None:
    # A picture's decode is closed before its colour correction starts (still.upscale), so their
    # working memory is never alive at once: the decode's peak is the larger of the pass and the
    # correction's one frame (peaks.colour_one_frame: split's term at the output's area, 0.56 GiB
    # at 4K, peaks.SPLIT_FRAME_4K, which 1080p's quarter of the area scales, with the decoded frame,
    # the reference and cuBLAS's workspace). At 1080p an untiled decode, 0.2 of its slice figure,
    # is far over the correction's 0.18 GiB: the term changes nothing.
    free = card(96)
    with_split = still_plan.tiles(made((1920, 1080), free, split=True), (1080, 1920))
    without = still_plan.tiles(made((1920, 1080), free, split=False), (1080, 1920))
    assert with_split.decode is None and without.decode is None
    term = peaks.colour_one_frame(padded_size((1080, 1920)), (1080, 1920))
    assert term < without.decode_peak - peaks.VAE_WEIGHTS
    assert with_split.decode_peak == without.decode_peak
    # A tiled decode in 512-px tiles (pinned) on an 8K picture is under the correction: the
    # term, which grows with the frame where the tile's does not, is the peak, and the plan
    # without the correction takes the tile's.
    pin = Pins(decode_tile=512)
    tiled = still_plan.tiles(made(EIGHT_K, card(24), split=True, pins=pin), (4320, 7680))
    plain = still_plan.tiles(made(EIGHT_K, card(24), split=False, pins=pin), (4320, 7680))
    assert tiled.decode == plain.decode == Tiles(512)
    models = made(EIGHT_K, card(24), split=True, pins=pin).models
    term = peaks.colour_one_frame(padded_size((4320, 7680)), output_size((4320, 7680)))
    assert tiled.decode_peak - plain.decode_peak > 0
    assert tiled.decode_peak == term + peaks.VAE_WEIGHTS + peaks.dit_on_gpu(
        models.resident_decode, models.swapped
    )
    # A tile that is the larger leaves the peak as it is: the unpinned plan's.
    big_tiled = still_plan.tiles(made(EIGHT_K, card(24), split=True), (4320, 7680))
    big_plain = still_plan.tiles(made(EIGHT_K, card(24), split=False), (4320, 7680))
    assert big_tiled.decode == big_plain.decode and big_tiled.decode is not None
    assert big_tiled.decode_peak == big_plain.decode_peak


# ---- The floor -----------------------------------------------------------------------------------


def test_the_floor_refusal_names_the_phase_what_it_takes_and_the_largest_bound_that_fits() -> None:
    # A card of 10 GiB: the 16 GiB card of the plain attention's figures plans 8K now (the DiT's
    # floor at 8K is 9.75 GiB free where it was 18.52).
    with pytest.raises(PlanRefusal) as refused:
        made(EIGHT_K, card(10))
    words = cli_words(refused.value)
    assert refused.value.phase == "dit"
    assert "pictures within 7680x4320 take" in words and "GiB of free VRAM" in words
    assert "the driver reports 8.51 GiB free" in words
    fits = refused.value.fits
    assert fits is not None and f"the largest bound that fits there is {fits[0]}x{fits[1]}" in words
    # The planner's walk down, linearly: the short side in steps of 16 from the multiple under it,
    # the long side in proportion; the first that plans is the refusal's, and the one before not.
    steps = []
    for side in range((EIGHT_K[1] - 1) // 16 * 16, 0, -16):
        steps.append((side * EIGHT_K[0] // EIGHT_K[1] // 2 * 2, side))
    index = steps.index(fits)
    made(fits, card(10))
    with pytest.raises(PlanRefusal):
        made(steps[index - 1], card(10))


def test_the_floor_of_a_vae_phase_is_the_proof_at_the_smallest_tiles() -> None:
    # 1080p with 4 GiB free: the DiT's floor is lower, the decode's 512-px tiles take 2.47 GiB
    # at the most over the pictures within the bound (the VAE's 0.47 GiB and 2.00 for a tile, one
    # frame's pass at 0.33 of the slice's figure, over the colour correction's 0.21 at the most an
    # output gives), and the room every VAE pass is held to 1.25 more.
    free = Reading(round(4.0 * GIB), None, round(4.0 * GIB), False)
    with pytest.raises(PlanRefusal) as refused:
        made((1920, 1080), free)
    assert refused.value.phase == "decode"
    assert "its decode in tiles of 512 px or untiled, with the DiT off the GPU" in str(
        refused.value
    )
    assert "1.25 of room a VAE pass is held to, and 0.60 of margin" in cli_words(refused.value)
    totals = re.search(
        r"take (\d+\.\d+) GiB of free VRAM \(.*a peak of (\d+\.\d+) GiB, (\d+\.\d+)",
        cli_words(refused.value),
    )
    assert totals is not None
    total, peak, room = (float(each) for each in totals.groups())
    assert total == pytest.approx(peak + room + 0.6, abs=0.011)


def test_no_output_size_fits_in_a_tiny_card() -> None:
    with pytest.raises(PlanRefusal) as refused:
        made(FOUR_K, Reading(GIB, None, GIB, False))
    assert refused.value.fits is None and "no output size fits there" in cli_words(refused.value)


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
    # (card 22 on the plain attention's figures, 20 on the lean one's)
    assert not made(FOUR_K, card(20), pins=Pins(swap_blocks=0)).models.vae_in_dit
    assert made(FOUR_K, free, pins=Pins(swap_blocks=36)).models.swapped == 36


def test_swap_blocks_pinned_that_does_not_hold_the_dit_is_refused_naming_the_fewest() -> None:
    with pytest.raises(PlanRefusal) as refused:
        made(FOUR_K, card(19), pins=Pins(swap_blocks=0))
    words = cli_words(refused.value)
    assert refused.value.phase == "dit"
    assert (
        "--swap-blocks 0: the DiT's phase, one latent of the largest picture within 3840x2160"
        in words
    )
    assert (
        "takes 18.12 GiB of free VRAM with 0 blocks swapped, and the driver reports 17.51" in words
    )
    fewest = made(FOUR_K, card(19)).models.swapped
    assert fewest == 4 and f"the fewest blocks that hold it are --swap-blocks {fewest}" in words
    assert made(FOUR_K, card(19), pins=Pins(swap_blocks=fewest)).models.swapped == fewest


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
    assert cli_words(refused.value) == (
        f"--decode-tile {pin}: a tile of {pin} px covers the whole padded frame, {width}x{height}:"
        " --decode-tile 0 pins an untiled decode"
    )
    assert refused.value.phase == "pin"
    with pytest.raises(PlanRefusal) as covered:
        made(FOUR_K, Reading(0, None, 0, False), pins=Pins(encode_tile=4096))
    assert cli_words(covered.value).startswith("--encode-tile 4096")


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
    # The pin is taken for every picture, the bound's pictures one pixel over an even side too,
    # whose padded frame (1120x1952) is past the 1104x1920 one frame's untiled decode was measured
    # on: the pin takes the video slice's figure there (peaks.one_frame_ratio 1.0), 38.38 GiB of
    # free VRAM with the DiT off the GPU, far more than an 11 GB card reports free (9.51).
    with pytest.raises(PlanRefusal) as refused:
        made((1920, 1080), card(11), pins=Pins(decode_tile=0))
    assert refused.value.phase == "pin"
    assert cli_words(refused.value).startswith("--decode-tile 0: the decode untiled")
    assert "takes 38.38 GiB of free VRAM, the DiT off the GPU" in cli_words(refused.value)
    assert "the largest tile that fits is --decode-tile " in cli_words(refused.value)


def test_a_tile_pin_that_does_not_fit_names_the_largest_tile_that_does() -> None:
    # A 12 GB card: the 2048-px encode takes 12.31 GiB of free VRAM there (one frame's pass, 0.24
    # of the slice's figure; a 32 GB card holds it).
    free = card(12)
    with pytest.raises(PlanRefusal) as refused:
        made(FOUR_K, free, pins=Pins(encode_tile=2048))
    words = cli_words(refused.value)
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
    # the untiled pass, which the floor proves, fits. One frame's decode takes 0.33 of the slice's
    # figure at 512 px (peaks.ONE_FRAME_TILED), so that the extra is 150 GiB for 50 GiB, over the
    # 46 GiB the 48 GB card's budget is.
    monkeypatch.setattr(peaks, "CUT_OFF_EXTRA", 150 * GIB)
    still_plan._worst.cache_clear()
    try:
        assert max(max(frame) for frame in still_plan._frames((480, 320))) <= 512
        with pytest.raises(PlanRefusal) as refused:
            made((480, 320), card(48), pins=Pins(decode_tile=496))
        assert "no tile fits, the untiled decode does, --decode-tile 0" in cli_words(refused.value)
    finally:
        monkeypatch.undo()
        still_plan._worst.cache_clear()


# ---- The system memory ---------------------------------------------------------------------------


def test_the_host_figure_is_the_process_the_pinned_copies_and_the_scratch() -> None:
    plan = made(EIGHT_K, card(24), 64 * GIB)
    scratch = max(peaks.scratch(frame, 1) for frame in still_plan._frames(EIGHT_K))
    assert plan.host == PROCESS + plan.pinned + scratch and scratch > 0
    # Pinned: each part page-rounded; the swapped blocks (6 at 24 GB, 8K) and what the VAE
    # phases send away.
    blocks = plan.models.swapped
    assert still_plan._leaving(plan.models) >= blocks
    parts = [peaks.BLOCK] * still_plan._leaving(plan.models) + [peaks.VAE_WEIGHTS]
    assert plan.pinned == sum(-(-part // mmap.PAGESIZE) * mmap.PAGESIZE for part in parts)
    # Untiled passes everywhere: no scratch. A 1600x900 bound, whose padded frames (928x1632) are
    # within the frame one frame's untiled passes were measured on (1080p's 1920x1104 is not: its
    # pictures one pixel over an even side pad past it and are tiled, with scratch).
    assert made((1600, 900), card(96), 64 * GIB).host == PROCESS


def test_the_vae_is_kept_with_more_swapped_blocks_when_that_pins_less_and_fits_the_host() -> None:
    free = card(24)
    base = made(EIGHT_K, free, 64 * GIB)
    assert (
        not base.models.vae_in_dit and base.models.swapped == 6
    )  # 27 on the plain attention's figures
    tight = made(EIGHT_K, free, base.host - 1)
    assert tight.models.vae_in_dit and tight.models.swapped > base.models.swapped
    assert tight.host < base.host and tight.host <= base.host - 1
    # Never under --swap-blocks, taken exactly.
    pinned = made(EIGHT_K, free, base.host - 1 + 64 * GIB, pins=Pins(swap_blocks=6))
    assert pinned.models.swapped == 6


def test_more_of_the_dit_stays_on_the_gpu_in_the_vae_phases_when_the_host_is_short() -> None:
    # A 24 GB card: with one frame's passes (peaks.picture_encode_peak) and the untiled passes
    # offered only within the frames they were measured on (peaks.ONE_FRAME_UNTILED_MEASURED), the
    # base plan keeps 26 blocks of the DiT in the encode and 17 in the decode; a host of 10 GiB
    # (the base plan takes 10.47) moves the decode's alone, the encode keeping the 26 it already
    # has at the most it can there, so that a host of 6 GiB is chosen: it moves both (28 and 28).
    free = card(24)
    base = made(FOUR_K, free, 64 * GIB)
    assert base.models.vae_in_dit
    assert (base.models.resident_encode, base.models.resident_decode) == (26, 17)
    short = made(FOUR_K, free, round(6.0 * GIB))
    assert short.host <= round(6.0 * GIB) < base.host
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
        made(EIGHT_K, card(24), round(4 * GIB))
    words = cli_words(refused.value)
    assert refused.value.phase == "host"
    numbers = [
        float(each)
        for each in re.findall(r"(\d+\.\d+) (?:for the process|pinned for|for a tiled pass)", words)
    ]
    total = float(found(r"takes (\d+\.\d+) GiB of system memory", words).group(1))
    assert len(numbers) == 3 and total == pytest.approx(sum(numbers), abs=0.001)
    assert "4.00 GiB are available" in words and "1.54 GiB short" in words


def test_the_host_is_unchecked_when_it_is_unread() -> None:
    assert made(EIGHT_K, card(24), None).host_available is None


# ---- A held run and the tight reserve ------------------------------------------------------------


def test_the_tight_reserve_is_taken_where_the_usual_one_is_refused_and_said() -> None:
    # 7 GiB reported, 0.5 held back: the plan counts on 2 with the reserve of 4.5, 4 with 2.5. The
    # 1080p bound's decode takes 3.72 GiB (one frame's 512-px tiles, 2.47, and the room), between
    # them; 11 GiB reported before one frame's passes were measured (7.67 GiB taken, 6 and 8).
    reading = Reading(round(7 * GIB), None, round(2 * GIB), True, round(0.5 * GIB))
    with pytest.raises(PlanRefusal):
        made((1920, 1080), replace(reading, expandable=False))
    plan = made((1920, 1080), reading)
    assert plan.reading.tight and plan.reading.counted == round(4 * GIB)
    assert plan.margin == 0 and plan.budget == plan.reading.counted
    assert plan.tightened.startswith("pictures within 1920x1080 take 3.72 GiB of free VRAM")
    assert sweep(plan, seed=2) > 100
    lines = still_plan.said(plan)
    assert any("reserve: 2.50 GiB and not 4.50" in line for line in lines)


def test_expandable_segments_off_refuses_saying_the_tight_reserve_would_run_it() -> None:
    reading = Reading(
        round(7 * GIB), None, round(2 * GIB), True, round(0.5 * GIB), expandable=False
    )
    with pytest.raises(PlanRefusal) as refused:
        made((1920, 1080), reading)
    words = cli_words(refused.value)
    assert "expandable segments" in words and "reserve of 2.50 GiB" in words
    assert refused.value.phase == "decode"


def test_refused_with_both_reserves_says_both() -> None:
    # 6 GiB reported: 1 counted with the reserve of 4.5, 3 with 2.5, under the decode's 3.72.
    reading = Reading(round(6 * GIB), None, round(1 * GIB), True, round(0.5 * GIB))
    with pytest.raises(PlanRefusal) as refused:
        made((1920, 1080), reading)
    words = cli_words(refused.value)
    assert "refused with the reserve of 4.50 GiB" in words and "and with 2.50" in words


def test_a_tight_reading_is_not_planned_again() -> None:
    tight = memory.tightened(Reading(round(6 * GIB), None, round(1 * GIB), True, round(0.5 * GIB)))
    with pytest.raises(PlanRefusal) as refused:
        made((1920, 1080), tight)
    assert "refused with the reserve" not in cli_words(refused.value)


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
    # The VAE passes keep what fits of the DiT beside them: 26 blocks in the encode and 17 in the
    # decode at 24 GB, 8K (on the plain attention's figures, 27 blocks swapped there and the 9
    # unswapped stayed in both).
    assert (
        "in the encode, 26 of the DiT's blocks on the GPU; in the decode, 17 of the DiT's blocks"
        " on the GPU" in text
    )
    # Both VAE passes small beside one frame's measured figures, the unswapped blocks and the two
    # slots stay in each: the plain attention's plan at this card, 27 swapped, here pinned to it.
    pinned = "\n".join(
        still_plan.said(made(EIGHT_K, card(24), 64 * GIB, pins=Pins(swap_blocks=27)))
    )
    assert (
        "in the encode, the DiT's 9 unswapped blocks and its 2 slots on the GPU; in the decode, "
        "the DiT's 9 unswapped blocks and its 2 slots on the GPU" in pinned
    )
    # And the residents counted in blocks, one phase with the DiT off the GPU (4K, 16 GB: the
    # untiled passes being offered only within the frames they were measured on, a 24 GB card keeps
    # 26 and 17 blocks).
    four_k = "\n".join(still_plan.said(made(FOUR_K, card(16), 64 * GIB)))
    assert (
        "in the encode, 7 of the DiT's blocks on the GPU; in the decode, the DiT off the GPU"
        in four_k
    )
    assert (
        "system memory: " in text
        and f"pinned for {still_plan._leaving(plan.models)} blocks and the VAE" in text
    )
    assert "picture:" not in text and "decode: " not in text.replace("in the decode: ", "")


def test_said_puts_the_smaller_files_line_after_the_blocks_swapped() -> None:
    # The line of the smaller DiT files that plan the job better (the command line's stills.py makes
    # it, as the planner's smaller_swaps does for a video) comes after the blocks swapped line, as
    # the planner's; with none given the lines are the plan's alone.
    plan = made(EIGHT_K, card(24), 64 * GIB)
    plain = still_plan.said(plan)
    line = "smaller DiT files plan this job better: this plan swaps 27 blocks; int8 0 blocks"
    lines = still_plan.said(plan, smaller=line)
    at = next(
        number for number, each in enumerate(lines) if each.startswith("plan: blocks swapped")
    )
    assert lines[at + 1] == f"plan: {line}"
    assert [each for each in lines if each != f"plan: {line}"] == plain


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


# ---- The picture's own copies on the GPU ---------------------------------------------------------

MEGAPIXELS_100 = 100_000_000
LARGE_PHOTO = 12000 * 9000  # 108 MP, stored: resized down to a 1080p output on the GPU
HD, HD_PADDED = (1920, 1080), 1120 * 1952  # the bound's largest padded frame (odd sides rounded up)


def test_the_copies_follow_the_rule_measured_on_the_rtx_5090() -> None:
    # The rule of still_plan's comment (2026-11-03, 8 shapes from 960x540 to 12000x6750): the
    # resize holds 30 bytes a stored pixel in the encode and 24 in the decode, and 24 a pixel of
    # its output in both; cuBLAS's workspace (32 MiB) is held beside every moment. With no output
    # (padded pixels 0), a 100 MP input's first moment is the resize's alone: 3.0 GB in the encode
    # (the stored picture 6, its float32 copy 12, the contiguous copy 12 bytes a pixel), 2.4 GB in
    # the decode (the float32 stored picture and its contiguous copy), and the later moments hold
    # the stored picture alone: bfloat16 in the encode (0.6 GB), float32 in the decode (1.2 GB).
    workspace = peaks.CUBLAS_WORKSPACE
    assert workspace == 32 * 1024 * 1024
    encode = [moment.at(MEGAPIXELS_100, 0) for moment in still_plan.COPIES["encode"]]
    assert encode == [3_000_000_000 + workspace] + [600_000_000 + workspace] * 3
    decode = [moment.at(MEGAPIXELS_100, 0) for moment in still_plan.COPIES["decode"]]
    assert decode == [2_400_000_000 + workspace] + [1_200_000_000 + workspace] * 3
    assert still_plan.RESIZE == {"encode": (30, 24), "decode": (24, 24)}
    # Per pixel of the padded frame, bytes, the encode's moments (bfloat16 sample 6, float32 12):
    # the resize's output twice in float32, 24; the clamp 12 (two bfloat16); the padding 18
    # (three); the normalisation 12 (two)...
    assert [moment.per_padded for moment in still_plan.COPIES["encode"]] == [24, 12, 18, 12]
    # ... the decode's in float32 (the reference), and the decoded frame's 6 (bfloat16, 3
    # channels) beside each: 24 + 6, 24 + 6, 36 + 6, 24 + 6.
    assert [moment.per_padded for moment in still_plan.COPIES["decode"]] == [30, 30, 42, 30]
    assert [moment.per_input for moment in still_plan.COPIES["encode"]] == [30, 6, 6, 6]
    assert [moment.per_input for moment in still_plan.COPIES["decode"]] == [24, 12, 12, 12]
    assert {moment.held for each in still_plan.COPIES.values() for moment in each} == {workspace}
    # At a 100 MP input and 1080p's frame, the resize is the peak of both phases.
    assert still_plan.copies_peak("encode", MEGAPIXELS_100, HD_PADDED) == (
        30 * MEGAPIXELS_100 + 24 * HD_PADDED + workspace
    )
    assert still_plan.copies_peak("decode", MEGAPIXELS_100, HD_PADDED) == (
        24 * MEGAPIXELS_100 + 30 * HD_PADDED + workspace
    )


def test_a_24_megapixel_input_to_4k_and_a_100_megapixel_one_to_1080p_have_their_figures() -> None:
    four_k = made(FOUR_K, card(32), input_pixels=24_000_000)
    padded = 2192 * 3872
    assert still_plan._padded_pixels(FOUR_K) == padded
    assert still_plan.input_copies(four_k) == {
        "encode": 30 * 24_000_000 + 24 * padded + peaks.CUBLAS_WORKSPACE,
        # The resize's moment (24 + 30) is over the padding's (12 + 42) at this ratio of the sizes.
        "decode": 24 * 24_000_000 + 30 * padded + peaks.CUBLAS_WORKSPACE,
    }
    hd = made(HD, card(32), input_pixels=MEGAPIXELS_100)
    assert still_plan.input_copies(hd) == {
        "encode": 30 * MEGAPIXELS_100 + 24 * HD_PADDED + peaks.CUBLAS_WORKSPACE,
        "decode": 24 * MEGAPIXELS_100 + 30 * HD_PADDED + peaks.CUBLAS_WORKSPACE,
    }
    assert hd.input_pixels == MEGAPIXELS_100
    # Without the colour correction there is no reference: the encode's moment alone.
    plain = made(HD, card(32), input_pixels=MEGAPIXELS_100, split=False)
    assert list(still_plan.input_copies(plain)) == ["encode"]


def test_a_large_input_downscaled_is_planned_with_its_copies_counted() -> None:
    # 12000x9000 to a 1080p output on a 16 GB card: 3.10 GiB of copies in the encode, beside the
    # models and under the budget, so the placement is the small picture's.
    plan = made(HD, card(16), input_pixels=LARGE_PHOTO)
    copies = still_plan.input_copies(plan)
    assert copies["encode"] == 30 * LARGE_PHOTO + 24 * HD_PADDED + peaks.CUBLAS_WORKSPACE
    job = still_plan._job_of(plan)
    residents: list[tuple[Phase, int | None]] = [
        ("encode", plan.models.resident_encode),
        ("decode", plan.models.resident_decode),
    ]
    for phase, resident in residents:
        assert still_plan._copies_need(job, phase, resident, plan.models.swapped) <= plan.budget
    assert plan.models == made(HD, card(16), input_pixels=INPUT).models
    text = "\n".join(still_plan.said(plan))
    assert "the input: 108.0 MP at its stored size" in text
    assert f"{copies['encode'] / GIB:.2f} GiB in the encode" in text
    assert f"{copies['decode'] / GIB:.2f} GiB in the decode" in text
    assert "by the rule measured on the RTX 5090" in text and "not yet measured" not in text
    assert "the input:" in "\n".join(still_plan.said(plan, (1080, 1920)))


def test_the_residents_are_lowered_where_the_copies_need_it() -> None:
    # 460 MP on a 1080p bound with a 16 GB card (770 MP before the copies were measured: the
    # resize holds 30 bytes a stored pixel in the encode, not 18, so the same moment, 12.9 GiB
    # at 460 MP, is reached with 0.6 of the input): the encode's moment doesn't fit beside the 15
    # blocks of the DiT that the passes keep for a small picture, nor beside 1, and fits beside
    # none: lowered to the most that holds, as the floor proof lowers them. The decode's moment,
    # in float32 at 24 bytes a pixel, is lowered too, to 6 of its 8 (10 before the untiled passes
    # were offered only within the frames they were measured on).
    big = 460_000_000
    plan = made(HD, card(16), input_pixels=big)
    small = made(HD, card(16), input_pixels=INPUT)
    assert small.models.resident_encode == 15 and small.models.resident_decode == 8
    assert plan.models.resident_encode == 0 and plan.models.resident_decode == 6
    job, swapped = still_plan._job_of(plan), plan.models.swapped
    assert still_plan._copies_need(job, "encode", 0, swapped) <= plan.budget
    # One block more would not fit the copies: lowered by the moment, not by the proof (which
    # holds at 15).
    assert still_plan._copies_need(job, "encode", 1, swapped) > plan.budget
    assert still_plan._proof_need(job, "encode", 1, swapped) <= plan.budget
    assert still_plan._copies_need(job, "decode", 6, swapped) <= plan.budget
    assert still_plan._copies_need(job, "decode", 7, swapped) > plan.budget
    # A little less input: 1 block, the least lowering at 0 (450 MP); and 390 MP lowers the encode
    # to 5 alone, the decode's moment still holding at the small picture's 8.
    middle = made(HD, card(16), input_pixels=450_000_000)
    assert middle.models.resident_encode == 1
    some = made(HD, card(16), input_pixels=390_000_000)
    assert some.models.resident_encode == 5 and some.models.resident_decode == 8
    # The planned tiles are the small picture's: the copies are gone before the encode.
    tiles, tiles_small = (still_plan.tiles(each, (1080, 1920)) for each in (plan, small))
    assert (tiles.encode, tiles.decode) == (tiles_small.encode, tiles_small.decode)


def test_a_picture_whose_copies_fit_nowhere_is_refused_naming_the_largest_input_that_does() -> None:
    with pytest.raises(PlanRefusal) as refused:
        made(HD, card(16), input_pixels=1_000_000_000)
    words = cli_words(refused.value)
    assert refused.value.phase == "encode"
    fits = refused.value.fits_input
    assert fits is not None and fits < 1_000_000_000
    assert "the picture's input, 1000.0 MP at its stored size" in words
    assert "copied to the GPU and resized there in the encode's phase" in words
    assert "the driver reports 14.51 GiB free" in words and "GiB of free VRAM" in words
    assert f"the largest input that fits is {fits // 100_000 / 10:.1f} MP" in words
    width, height = (int(each) for each in found(r"about (\d+)x(\d+) at the bound", words).groups())
    assert width * height <= fits and width * height > fits * 0.999 and width > height
    assert abs(width / height - 1920 / 1080) < 0.01
    assert "a smaller --resolution" in words
    # The input that fits is planned, at resident 0 in the encode; one pixel more is not.
    plan = made(HD, card(16), input_pixels=fits)
    assert plan.models.resident_encode == 0
    with pytest.raises(PlanRefusal) as over:
        made(HD, card(16), input_pixels=fits + 1)
    assert over.value.fits_input == fits
    # The copies' figures add up: the peak, and the models with the margin.
    copies = still_plan.copies_peak("encode", 1_000_000_000, HD_PADDED)
    assert f"its copies take {math.ceil(copies / GIB * 100) / 100:.2f} GiB at the peak" in words


def test_the_largest_input_that_fits_holds_every_phase_the_copies_are_made_in() -> None:
    # 8K on a 48 GB card, searched by the planner's own refusal: the input it names is planned,
    # and its copies fit at resident 0 in the encode and in the decode, with the colour correction.
    bound, free = EIGHT_K, card(48)
    with pytest.raises(PlanRefusal) as refused:
        made(bound, free, input_pixels=5_000_000_000)
    fits = refused.value.fits_input
    assert fits is not None and refused.value.phase in ("encode", "decode")
    plan = made(bound, free, input_pixels=fits)
    job = still_plan._job_of(plan)
    needs = [still_plan._copies_need(job, phase, 0, 0) for phase in ("encode", "decode")]
    # The binding moment's 30 bytes a stored pixel (the encode's resize): one pixel more is over.
    assert max(needs) <= plan.budget < max(needs) + 30
    with pytest.raises(PlanRefusal):
        made(bound, free, input_pixels=fits + 1)


def test_without_the_colour_correction_the_decode_makes_no_copy() -> None:
    # Only the encode's moment is judged: the decode is not the phase refused.
    with pytest.raises(PlanRefusal) as refused:
        made(HD, card(16), input_pixels=1_000_000_000, split=False)
    assert refused.value.phase == "encode"
    plan = made(HD, card(16), input_pixels=470_000_000, split=False)
    assert list(still_plan.input_copies(plan)) == ["encode"]
    assert "GiB in the decode" not in "\n".join(still_plan.said(plan))


def test_no_input_fits_where_the_outputs_own_copies_are_over_the_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A moment whose padded-frame share alone is over the room (the table replaced): no input
    # helps, the refusal says so and names the output.
    huge = still_plan.Moment("a tensor of the output's size", 1, 4_000_000_000)
    monkeypatch.setitem(still_plan.COPIES, "encode", (huge,))
    with pytest.raises(PlanRefusal) as refused:
        made(HD, card(16))
    assert refused.value.fits_input is None and refused.value.phase == "encode"
    assert "no input fits there, the output's own copies being over the memory" in str(
        refused.value
    )
    assert "a tensor of the output's size" in cli_words(refused.value)


def test_a_small_inputs_size_changes_nothing_of_the_plan() -> None:
    # The copies of an ordinary photo are far under every phase's room, so that the plan, its
    # tiles and its lines are those of any other small input (but the input's own line).
    for bound, gib in [(HD, 12), (HD, 16), (FOUR_K, 16), (FOUR_K, 24), (SIX_BY_FOUR, 32)]:
        plans = [
            made(bound, card(gib), 64 * GIB, input_pixels=pixels)
            for pixels in (1_000, 2_000_000, 12_192_768, 24_000_000, 50_000_000)
        ]
        for each in plans[1:]:
            assert replace(each, input_pixels=1_000) == plans[0], (bound, gib)
            assert still_plan.tiles(each, (bound[1], bound[0])) == still_plan.tiles(
                plans[0], (bound[1], bound[0])
            )


def test_a_refusal_for_the_copies_survives_the_tight_retry() -> None:
    with pytest.raises(PlanRefusal) as refused:
        made(HD, held(16.0), input_pixels=2_000_000_000)
    assert refused.value.phase == "encode" and refused.value.fits_input is not None
    assert "GiB more would have to be freed" in cli_words(refused.value) or "short" in cli_words(
        refused.value
    )


def test_an_input_of_no_pixels_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="an input of 0 pixels"):
        made(HD, card(16), input_pixels=0)


# ---- What a phase takes, for a reservation -------------------------------------------------------

PEAKED = [
    pytest.param(FOUR_K, card(24), peaks.FLASH_ATTN_2, INPUT, id="4K-24GB-FA2"),
    pytest.param(FOUR_K, card(24), peaks.SDPA, INPUT, id="4K-24GB-SDPA"),
    pytest.param(FOUR_K, card(24), peaks.FLASH_ATTN_2, 24_000_000, id="4K-24GB-24MP"),
    pytest.param(HD, card(24), peaks.FLASH_ATTN_2, INPUT, id="1080p-24GB-FA2"),
    pytest.param(HD, card(24), peaks.SDPA, INPUT, id="1080p-24GB-SDPA"),
    pytest.param(HD, card(16), peaks.FLASH_ATTN_2, LARGE_PHOTO, id="1080p-16GB-108MP"),
]


VAE_PHASES: tuple[Phase, Phase] = ("encode", "decode")


def residents(plan: StillPlan) -> dict[Phase, int | None]:
    """The blocks of the DiT the plan keeps on the GPU in each VAE phase."""
    return {"encode": plan.models.resident_encode, "decode": plan.models.resident_decode}


def peak_of(bound: still_plan.Bound) -> tuple[int, int]:
    """The bound's own picture, landscape, as a target (rows, columns)."""
    return bound[1], bound[0]


@pytest.mark.parametrize(("bound", "free", "attention", "pixels"), PEAKED)
def test_the_bounds_own_picture_takes_no_more_of_the_dit_phase_and_its_vae_phases_fit_the_budget(
    bound: still_plan.Bound, free: Reading, attention: str, pixels: int
) -> None:
    # bound_peaks is the plan's guarantee, not a ceiling over every picture: the DiT's phase is
    # the most any picture within the bound takes, so that a picture's own is equal or under it;
    # a VAE phase is the proof's worst pass (_proof_need: 512-px tiles or untiled, the floor
    # every picture has) or the input's copies, the larger, while a picture takes the largest
    # tile that fits the budget (tiles), up to it: its pass is over the proof's, and the plan
    # holds it under the budget with its room.
    plan = made(bound, free, attention=attention, input_pixels=pixels)
    most = still_plan.bound_peaks(plan)
    assert list(most) == list(still_plan.PHASES)
    job, models = still_plan._job_of(plan), plan.models
    for phase, resident in residents(plan).items():
        proof = still_plan._proof_need(job, phase, resident, models.swapped) - VAE_ROOM
        copies = still_plan._copies_need(job, phase, resident, models.swapped)
        floor = max(proof, copies) if phase in still_plan._copy_phases(job) else proof
        assert most[phase].allocated == floor
        assert most[phase].allocated + most[phase].room <= plan.budget
    assert most["dit"].allocated == plan.dit_peak
    for target in (peak_of(bound), (bound[0], bound[1])):
        own = still_plan.picture_peaks(plan, target, pixels)
        assert list(own) == list(most)
        assert own["dit"].allocated <= most["dit"].allocated, target
        assert own["dit"].reserved <= most["dit"].reserved, target
        for phase in VAE_PHASES:
            assert own[phase].models == most[phase].models, (phase, target)
            assert own[phase].room == most[phase].room, (phase, target)
            assert own[phase].allocated + own[phase].room <= plan.budget, (phase, target)
            assert own[phase].allocated >= most[phase].allocated, (phase, target)


@pytest.mark.parametrize(("bound", "free", "attention", "pixels"), PEAKED)
def test_every_smaller_picture_and_input_takes_less_of_the_dit_phase_and_fits_the_budget(
    bound: still_plan.Bound, free: Reading, attention: str, pixels: int
) -> None:
    plan = made(bound, free, attention=attention, input_pixels=pixels)
    most = still_plan.bound_peaks(plan)
    long, short = bound
    targets = [
        (short // 2 // 2 * 2, long // 2 // 2 * 2),
        (short // 3 // 2 * 2, long // 3 // 2 * 2),
        (long // 3 // 2 * 2, short // 3 // 2 * 2),
        (1000, 1500),
        (270, 480),
        (short - 2, long - 2),
    ]
    for target in targets:
        for input_pixels in (1, 1_000, pixels // 2, pixels):
            found = still_plan.picture_peaks(plan, target, input_pixels)
            assert found["dit"].allocated <= most["dit"].allocated, target
            assert found["dit"].reserved <= most["dit"].reserved, target
            for phase in VAE_PHASES:
                assert found[phase].allocated + found[phase].room <= plan.budget, (phase, target)
            # The input's copies are the picture's own: a smaller input never takes more.
            smaller = still_plan.picture_peaks(plan, target, 1)
            for phase in VAE_PHASES:
                assert smaller[phase].allocated <= found[phase].allocated, (phase, target)


@pytest.mark.parametrize(("bound", "free", "attention", "pixels"), PEAKED)
def test_a_phases_models_and_room_follow_the_placement(
    bound: still_plan.Bound, free: Reading, attention: str, pixels: int
) -> None:
    plan = made(bound, free, attention=attention, input_pixels=pixels)
    models = plan.models
    vae = peaks.VAE_WEIGHTS if models.vae_in_dit else 0
    expected = peaks.dit_on_gpu(None, models.swapped, plan.dit_model) + vae
    for found in (
        still_plan.bound_peaks(plan),
        still_plan.picture_peaks(plan, peak_of(bound), pixels),
    ):
        assert found["dit"].models == expected
        assert found["dit"].room == 0
        assert found["encode"].room == found["decode"].room == VAE_ROOM
        # What is on the GPU in a VAE phase: the VAE's weights and what of the DiT stays.
        for phase, resident in residents(plan).items():
            assert found[phase].models == (
                peaks.VAE_WEIGHTS + peaks.dit_on_gpu(resident, models.swapped, plan.dit_model)
            )
            assert found[phase].allocated > found[phase].models
        assert found["dit"].allocated >= found["dit"].models
    # The DiT's phase at the bound is the plan's own.
    assert still_plan.bound_peaks(plan)["dit"].allocated == plan.dit_peak


def test_the_bounds_dit_phase_is_the_dit_peak_and_a_pictures_is_its_own_windows() -> None:
    plan = made(FOUR_K, card(24))
    own = still_plan.picture_peaks(plan, (2160, 3840), INPUT)["dit"]
    padded = padded_size((2160, 3840))
    assert own.allocated == peaks.dit_peak(
        padded, 1, plan.models.swapped, plan.models.vae_in_dit, plan.attention, plan.dit_model
    )
    # A picture whose padded frame takes fewer tokens takes less of the DiT's phase.
    small = still_plan.picture_peaks(plan, (1080, 1920), INPUT)["dit"]
    assert small.allocated < own.allocated <= plan.dit_peak


def test_the_vae_phases_of_a_picture_are_the_larger_of_its_pass_and_its_copies() -> None:
    # The planner's tiles for the picture and the copies at its sizes (tiles, copies_peak): a
    # 108 MP input to 1080p, whose copies take 3.10 GiB in the encode.
    plan = made(HD, card(16), input_pixels=LARGE_PHOTO)
    target = (1080, 1920)
    found = still_plan.picture_peaks(plan, target, LARGE_PHOTO)
    chosen = still_plan.tiles(plan, target)
    padded = padded_size(target)
    passes: dict[Phase, int] = {"encode": chosen.encode_peak, "decode": chosen.decode_peak}
    for phase, peak in passes.items():
        copies = still_plan.copies_peak(phase, LARGE_PHOTO, padded[0] * padded[1])
        assert found[phase].allocated == max(peak, found[phase].models + copies)
    # A small input leaves the pass: the copies don't reach it.
    small = still_plan.picture_peaks(plan, target, 1_000)
    assert small["encode"].allocated == chosen.encode_peak
    assert small["decode"].allocated == chosen.decode_peak
    # Without the correction there are no copies in the decode, and the pass is its peak.
    plain = made(HD, card(16), input_pixels=LARGE_PHOTO, split=False)
    again = still_plan.tiles(plain, target)
    assert still_plan.picture_peaks(plain, target, LARGE_PHOTO)["decode"].allocated == (
        again.decode_peak
    )


def test_an_input_over_the_plans_or_under_a_pixel_is_a_value_error() -> None:
    plan = made(FOUR_K, card(24))
    still_plan.picture_peaks(plan, (2160, 3840), plan.input_pixels)
    still_plan.picture_peaks(plan, (2160, 3840), 1)
    with pytest.raises(ValueError, match="an input of"):
        still_plan.picture_peaks(plan, (2160, 3840), plan.input_pixels + 1)
    with pytest.raises(ValueError, match="an input of"):
        still_plan.picture_peaks(plan, (2160, 3840), 0)
    # A target outside the bound is tiles' own refusal.
    with pytest.raises(ValueError, match="not within the plan's bound"):
        still_plan.picture_peaks(plan, (3000, 3000), 1)


# ---- One frame's untiled pass is offered within the frame it was measured on --------------------


def test_the_4k_portrait_bounds_encode_is_not_untiled_on_a_96_gib_card() -> None:
    # A 4K bound's portrait picture pads to 3872x2160, past 4K landscape's 2192x3840 which one
    # frame's untiled encode was measured on (peaks.ONE_FRAME_UNTILED_MEASURED, either orientation:
    # it is the larger side that passes, 3872 over 3840). Its untiled encode at the video slice's
    # figure, 90.7 GiB, fits the 96 GiB card's budget with its room, and was chosen before the
    # measured frame bounded the offer; now it is tiled, at the 26.5 GiB of 2048-px tiles.
    plan = made(FOUR_K, card(96), 64 * GIB)
    portrait, landscape = (3840, 2160), (2160, 3840)
    assert padded_size(portrait) == (3872, 2160) and padded_size(landscape) == (2192, 3840)
    assert not measured_untiled("encode", padded_size(portrait))
    assert measured_untiled("encode", padded_size(landscape))
    untiled = peaks.picture_encode_peak(padded_size(portrait), None, None, 0)
    assert untiled + VAE_ROOM <= plan.budget and untiled > 90 * GIB
    chosen = still_plan.tiles(plan, portrait)
    assert chosen.encode == Tiles(2048) and chosen.encode_peak < 30 * GIB
    assert still_plan.tiles(plan, landscape).encode is None
    # And the phase the bound's pictures are proved at: the encode counted at the tiles, not at
    # the slice's untiled figure.
    assert still_plan.bound_peaks(plan)["encode"].allocated < 30 * GIB


def test_no_untiled_pass_is_chosen_past_the_frame_it_was_measured_on() -> None:
    # On a card that fits them at the slice's figure, over a bound's pictures in both
    # orientations: an untiled pass is chosen only where measured_untiled says so.
    plan = made(FOUR_K, card(96), 64 * GIB)
    for height in (17, 540, 1080, 1088, 1104, 1920, 2160, 3840):
        for width in (24, 1080, 1920, 2160, 3840):
            target = (height, width)
            output = output_size(target)
            if max(output) > 3840 or min(output) > 2160:
                continue
            found = still_plan.tiles(plan, target)
            padded = padded_size(target)
            for phase, chosen in (("encode", found.encode), ("decode", found.decode)):
                if chosen is None:
                    assert measured_untiled(phase, padded), (phase, target)


# ---- A picture's passes held to a budget -----------------------------------------------------

BUDGETED = [
    pytest.param(FOUR_K, 96, id="4K-96GB"),
    pytest.param(FOUR_K, 24, id="4K-24GB"),
    pytest.param(HD, 48, id="1080p-48GB"),
    pytest.param(SIX_BY_FOUR, 32, id="6000x4000-32GB"),
]


def lowest_budget(plan: StillPlan) -> int:
    """The least a picture's passes may be held to (still_plan.tiles): the worst VAE phase of the
    proof, its room with it."""
    return max(
        peak.allocated + peak.room
        for phase, peak in still_plan.bound_peaks(plan).items()
        if phase != "dit"
    )


@pytest.mark.parametrize(("bound", "gib"), BUDGETED)
def test_every_picture_within_the_bound_gets_passes_within_the_lowest_budget(
    bound: still_plan.Bound, gib: int
) -> None:
    plan = made(bound, card(gib), 64 * GIB)
    budget = lowest_budget(plan)
    assert budget <= plan.budget
    assert sweep(plan, seed=gib, budget=budget) > 100
    # The picture's phases, with that budget, are under it: the passes, then the copies.
    long, short = bound
    for target in ((short, long), (long, short)):
        found = still_plan.picture_peaks(plan, target, plan.input_pixels, budget)
        for phase in VAE_PHASES:
            assert found[phase].allocated + found[phase].room <= budget, (phase, target)


def test_a_budget_chooses_smaller_tiles_than_the_plans_and_the_plans_budget_is_none() -> None:
    plan = made(FOUR_K, card(96), 64 * GIB)
    target = (3840, 2160)
    assert still_plan.tiles(plan, target, plan.budget) == still_plan.tiles(plan, target)
    assert still_plan.picture_peaks(plan, target, 1_000_000, plan.budget) == (
        still_plan.picture_peaks(plan, target, 1_000_000)
    )
    # A 4K landscape picture's encode is untiled at the plan's budget (38.1 GiB, measured), and
    # in tiles within a budget under that.
    landscape = (2160, 3840)
    assert still_plan.tiles(plan, landscape).encode is None
    low = lowest_budget(plan)
    assert low < 20 * GIB
    held = still_plan.tiles(plan, landscape, low)
    assert held.encode is not None and held.encode_peak + VAE_ROOM <= low
    assert held.decode is not None and held.decode_peak + VAE_ROOM <= low
    # In between: the untiled encode's peak with its room is the least budget that keeps it.
    untiled = still_plan.tiles(plan, landscape).encode_peak + VAE_ROOM
    assert still_plan.tiles(plan, landscape, untiled).encode is None
    assert still_plan.tiles(plan, landscape, untiled - 1).encode is not None


def test_a_budget_over_the_plans_is_a_value_error() -> None:
    plan = made(FOUR_K, card(24), 64 * GIB)
    still_plan.tiles(plan, (2160, 3840), plan.budget)
    with pytest.raises(ValueError, match="over the plan's"):
        still_plan.tiles(plan, (2160, 3840), plan.budget + 1)
    with pytest.raises(ValueError, match="over the plan's"):
        still_plan.picture_peaks(plan, (2160, 3840), 1_000_000, plan.budget + 1)


def test_a_pinned_untiled_pass_is_taken_past_the_measured_frame_at_the_slices_figure() -> None:
    # --encode-tile 0 on a 96 GiB card: the 4K portrait picture, past the frame measured, is
    # untiled as pinned, and its peak is the slice's figure (peaks.one_frame_ratio 1.0), not the
    # measured ratio of it.
    plan = made(FOUR_K, card(96), 64 * GIB, pins=Pins(encode_tile=0))
    portrait = (3840, 2160)
    found = still_plan.tiles(plan, portrait)
    assert found.encode is None
    padded = padded_size(portrait)
    assert not measured_untiled("encode", padded)
    models = plan.models
    slice_figure = round(peaks._fit(padded, None, 0)) + 6 * padded[0] * padded[1]
    assert found.encode_peak == slice_figure + peaks.VAE_WEIGHTS + peaks.dit_on_gpu(
        models.resident_encode, models.swapped
    )
    measured = still_plan.tiles(plan, (2160, 3840))
    assert measured.encode is None and measured.encode_peak < 0.5 * found.encode_peak


# ---- The command line's one picture, as stored ---------------------------------------------------

# The card the CLI's 1080p picture was refused on (2026-11-04, the RTX 5090 with its desktop): 24
# GiB free, the 7B in fp16, and a 4032x3024 input's copies counted (INPUT).
TONIGHT = 24 * GIB
LANDSCAPE_HD, PORTRAIT_HD = (1080, 1920), (1920, 1080)


def test_a_landscape_1080p_picture_pinned_untiled_is_taken_where_its_own_plan_is_untiled() -> None:
    # Its own plan takes both passes untiled; pinned 0 and 0, the same passes and peaks.
    own = made(HD, TONIGHT, picture=LANDSCAPE_HD)
    chosen = still_plan.tiles(own, LANDSCAPE_HD)
    assert chosen.encode is None and chosen.decode is None
    pinned = made(HD, TONIGHT, pins=Pins(0, 0), picture=LANDSCAPE_HD)
    assert still_plan.tiles(pinned, LANDSCAPE_HD) == chosen
    assert pinned.models == own.models
    text = "\n".join(still_plan.said(pinned, LANDSCAPE_HD))
    assert "encode: untiled, pinned by --encode-tile" in text
    assert "decode: untiled, pinned by --decode-tile" in text
    # The image server's bound, either orientation, refuses the same pin as before: its portrait
    # frame is past the one frame's untiled decode was measured on, at the slice's figure.
    with pytest.raises(PlanRefusal) as refused:
        made(HD, TONIGHT, pins=Pins(0, 0))
    assert refused.value.phase == "pin"
    assert cli_words(refused.value).startswith(
        "--decode-tile 0: the decode untiled, for the pictures within 1920x1080, takes 38."
    )


def test_one_orientation_of_the_bound_would_not_have_taken_the_pin() -> None:
    # Why the CLI's plan is its picture alone and not its orientation of the bound: a smaller
    # landscape picture within 1920x1080 pads past the picture's frame, and past the frame one
    # frame's untiled decode was measured on, so its pin of 0 takes the slice's figure.
    assert padded_size(LANDSCAPE_HD) == (1104, 1920)
    assert measured_untiled("decode", padded_size(LANDSCAPE_HD))
    smaller = (1079, 1919)
    assert output_size(smaller) == (1078, 1918) and padded_size(smaller) == (1104, 1952)
    assert not measured_untiled("decode", padded_size(smaller))
    untiled = peaks.picture_decode_peak(padded_size(smaller), output_size(smaller), None, 0, 0)
    assert untiled + VAE_ROOM > TONIGHT


def test_a_portrait_picture_is_judged_by_its_own_frame() -> None:
    # 1080x1920 as stored pads to 1104x1952, past the decode measured untiled: its own plan tiles
    # the decode, and a pin of 0 is refused for it, in the picture's words.
    assert padded_size(PORTRAIT_HD) == (1952, 1104)
    own = made(HD, TONIGHT, picture=PORTRAIT_HD)
    assert still_plan.tiles(own, PORTRAIT_HD).decode is not None
    with pytest.raises(PlanRefusal) as refused:
        made(HD, TONIGHT, pins=Pins(decode_tile=0), picture=PORTRAIT_HD)
    assert cli_words(refused.value).startswith(
        "--decode-tile 0: the decode untiled, for the picture, 1080x1920, takes"
    )
    # A portrait picture whose frame is within the one measured, 1072x1920 padded, takes it.
    measured = (1888, 1072)
    assert padded_size(measured) == (1920, 1072)
    assert measured_untiled("decode", padded_size(measured))
    pinned = made((1888, 1072), TONIGHT, pins=Pins(0, 0), picture=measured)
    found = still_plan.tiles(pinned, measured)
    assert found.encode is None and found.decode is None


def test_a_pictures_plan_counts_its_own_frame_and_windows() -> None:
    # The DiT's phase is the picture's own (peaks.dit_peak at its frame and windows), never over
    # the bound's, which counts the most any picture within it takes.
    for picture, bound in ((LANDSCAPE_HD, HD), (PORTRAIT_HD, HD), ((3000, 2000), (3000, 2000))):
        own = made(bound, card(32), picture=picture)
        either = made(bound, card(32))
        assert own.windows == still_plan._windows(*(side // 16 for side in padded_size(picture)))
        assert own.dit_peak == still_plan.picture_peaks(own, picture, INPUT)["dit"].allocated
        assert own.dit_peak <= either.dit_peak and own.windows <= either.windows
    # 1920x1080 landscape: 30 windows, where its bound's portrait pictures take 36.
    assert made(HD, card(32), picture=LANDSCAPE_HD).windows == 30
    assert made(HD, card(32)).windows == 36


@pytest.mark.parametrize("bound", [EIGHT_K, FOUR_K, HD, (480, 320)])
def test_the_servers_bound_is_unchanged_without_a_picture(bound: still_plan.Bound) -> None:
    # The image server gives no picture: its plan is every picture within the bound in either
    # orientation, its figures and words as they were (the dump of 4,179 lines of plans, refusals,
    # lines and peaks compared before and after, impl-imgserver patch 10, 2026-11-04).
    reading = card(24)
    plan = still_plan.plan(bound, reading, None, process=PROCESS, input_pixels=INPUT)
    assert plan.picture is None
    assert repr(plan) == repr(made(bound, reading, picture=None))
    assert "in either orientation" in still_plan.said(plan)[1]
    long, short = bound
    for target in ((short, long), (long, short)):
        first_fit(plan, target)


def test_a_pictures_plan_says_the_picture_and_takes_its_target_alone() -> None:
    plan = made(HD, TONIGHT, picture=LANDSCAPE_HD)
    lines = still_plan.said(plan)
    assert lines[1] == (
        "plan: bound: the picture alone, 1920x1080 as stored, padded to 1920x1104 for the model,"
        " 8,280 tokens, one latent"
    )
    assert not any("either orientation" in line for line in lines)
    assert any("GiB for the picture, one latent, 30 attention windows" in line for line in lines)
    portrait = made(HD, TONIGHT, picture=PORTRAIT_HD)
    assert (
        "the picture alone, 1080x1920 as stored, padded to 1104x1952"
        in still_plan.said(portrait)[1]
    )
    # The other orientation, a smaller picture or one a pixel over: not the plan's picture.
    for target in (PORTRAIT_HD, (540, 960), (1081, 1920)):
        with pytest.raises(ValueError, match="the plan is for one picture, 1920x1080 as stored"):
            still_plan.tiles(plan, target)
    with pytest.raises(ValueError, match="the plan is for one picture, 1080x1920 as stored"):
        still_plan.tiles(portrait, LANDSCAPE_HD)
    # A bound that is not the picture's output is no plan.
    for bound in ((1920, 1088), (3840, 2160)):
        with pytest.raises(ValueError, match="the bound is the picture's output"):
            made(bound, TONIGHT, picture=LANDSCAPE_HD)


def test_a_picture_refused_for_the_memory_names_the_largest_picture_that_fits() -> None:
    # In its own orientation and words: the floor's refusal of one picture, walked down at its
    # ratio as the planner walks.
    for picture, said in ((LANDSCAPE_HD, "1920x1080"), (PORTRAIT_HD, "1080x1920")):
        with pytest.raises(PlanRefusal) as refused:
            made(HD, 4 * GIB, input_pixels=1000, picture=picture)
        error = refused.value
        assert error.fits is not None
        assert cli_words(error).startswith(f"the picture, {said}, takes")
        long, short = error.fits
        fits = (long, short) if picture[0] > picture[1] else (short, long)
        assert cli_words(error).endswith(
            f"the largest picture that fits there is {fits[1]}x{fits[0]}"
        )
        made(error.fits, 4 * GIB, input_pixels=1000, picture=fits)


# ---- Weights held: the pinned copies exist already (still_plan.plan's held) ----------------------


def held_made(bound: still_plan.Bound, free: int | Reading, host: int | None) -> StillPlan:
    return still_plan.plan(
        bound, free, host, process=PROCESS, input_pixels=INPUT, split=True, held=True
    )


def test_held_weights_count_no_pinned_memory_and_are_planned_as_the_card_alone_gives() -> None:
    # The same two jobs as the choices for a short host above: with the copies existing already,
    # neither the VAE kept for the DiT's phase nor the DiT kept in the VAE's phases is taken.
    free = card(24)
    # The first takes the VAE kept (a byte under what its plan as the card gives it takes), the
    # second the DiT kept in the VAE's phases.
    eight = made(EIGHT_K, free, 64 * GIB).host - 1
    for bound, host in ((EIGHT_K, eight), (FOUR_K, round(6.0 * GIB))):
        plain, unchecked = made(bound, free, host), made(bound, free, None)
        kept = held_made(bound, free, host)
        assert (plain.models, plain.pinned) != (unchecked.models, unchecked.pinned)
        assert kept.models == unchecked.models and kept.held and not plain.held
        assert kept.pinned == 0
        assert kept.host == unchecked.host - unchecked.pinned
        assert kept.host_available == host


def test_a_job_the_host_refused_is_planned_with_the_weights_held() -> None:
    free, host = card(24), round(4 * GIB)
    with pytest.raises(PlanRefusal) as refused:
        made(EIGHT_K, free, host)
    assert refused.value.phase == "host" and "pinned for" in str(refused.value)
    kept = held_made(EIGHT_K, free, host)
    assert kept.host <= host
    # A host that holds neither the process nor a scratch refuses it too, counting no copy.
    with pytest.raises(PlanRefusal) as short:
        held_made(EIGHT_K, free, round(2.5 * GIB))
    assert short.value.phase == "host"
    assert "for the process" in str(short.value) and "pinned for" not in str(short.value)


def test_held_false_changes_no_plan() -> None:
    for bound, free, host in (
        (EIGHT_K, card(24), 64 * GIB),
        (EIGHT_K, card(24), None),
        (FOUR_K, card(24), round(6.0 * GIB)),
        (FOUR_K, card(16), round(8.0 * GIB)),
    ):
        kwargs: dict[str, Any] = dict(process=PROCESS, input_pixels=INPUT, split=True)
        try:
            before = still_plan.plan(bound, free, host, **kwargs)
        except PlanRefusal as error:
            # The same refusal, in the same words.
            with pytest.raises(PlanRefusal) as again:
                still_plan.plan(bound, free, host, **kwargs, held=False)
            assert str(again.value) == str(error) and again.value.phase == error.phase
            continue
        assert still_plan.plan(bound, free, host, **kwargs, held=False) == before
    assert not made(EIGHT_K, card(24), 64 * GIB).held


def test_the_choices_for_the_host_are_not_made_on_held_weights() -> None:
    free = card(24)
    base = made(EIGHT_K, free, 64 * GIB)
    plain_job, held_job = (replace(still_plan._job_of(base), held=flag) for flag in (False, True))
    assert still_plan._vae_kept(plain_job, base) is not None
    assert still_plan._vae_kept(held_job, base) is None
    four = made(FOUR_K, free, 64 * GIB)
    plain_job, held_job = (replace(still_plan._job_of(four), held=flag) for flag in (False, True))
    assert still_plan._dit_kept(plain_job, four) is not None
    assert still_plan._dit_kept(held_job, four) is None
    # A plan made held keeps its flag for the figures that follow from it.
    assert still_plan._job_of(held_made(FOUR_K, free, None)).held


def test_a_held_plan_says_none_of_the_copies_are_counted() -> None:
    plan = held_made(FOUR_K, card(24), None)
    words = "\n".join(still_plan.said(plan))
    assert "none counted for the weights, pinned already" in words
    assert "GiB pinned for" not in words


# ---- The refusals by kind (errors.PlanRefused) --------------------------------------------------


def refused_by(*job: object, **options: object) -> PlanRefused:
    """The refusal still_plan.plan makes of a picture, whatever its kind (made's own arguments)."""
    with pytest.raises(PlanRefused) as error:
        made(*job, **options)  # pyright: ignore[reportArgumentType]
    return error.value


def test_each_refusal_of_a_picture_is_of_its_kind_with_its_facts() -> None:
    # Pins that are no pins, before any memory is counted: phase "pin", the pin asked.
    error = refused_by(FOUR_K, Reading(0, None, 0, False), pins=Pins(swap_blocks=37))
    assert type(error) is PinRefused and error.kind == "pin" and error.why == "range"
    assert error.phase == "pin" and error.pin == Setting("swap_blocks", "37")
    biggest = max(still_plan._frames(FOUR_K), key=max)
    pin = (max(biggest) + 63) // 64 * 64
    error = refused_by(FOUR_K, card(96), pins=Pins(decode_tile=pin))
    assert type(error) is PinRefused and error.why == "covers" and error.phase == "pin"
    assert error.pin == Setting("decode_tile", str(pin)) and error.instead is None
    # A pin the memory doesn't hold, with the one that does.
    error = refused_by((1920, 1080), card(11), pins=Pins(decode_tile=0))
    assert type(error) is PinRefused and error.why == "memory" and error.phase == "pin"
    assert error.pin == Setting("decode_tile", "0")
    assert error.instead is not None and error.instead.name == "decode_tile"
    assert error.need is not None and error.there is not None and error.need > error.there
    error = refused_by(FOUR_K, card(19), pins=Pins(swap_blocks=0))
    assert type(error) is PinRefused and error.why == "memory" and error.phase == "dit"
    assert error.pin == Setting("swap_blocks", "0") and error.instead == Setting("swap_blocks", "4")
    # The floor: a bound that doesn't fit, the phase, the figures and the largest that does.
    error = refused_by(EIGHT_K, card(10))
    assert type(error) is MemoryShort and error.kind == "memory" and error.phase == "dit"
    assert error.fits is not None and error.instead is None
    assert error.need is not None and error.there == int(card(10).counted)
    assert error.short == error.need - int(card(10).counted) and error.freeable is True
    assert error.reserve is None
    # The picture's input, copied to the GPU, which the output's size doesn't bound.
    error = refused_by(HD, held(16.0), input_pixels=2_000_000_000)
    assert type(error) is InputTooLarge and error.kind == "input" and error.phase == "encode"
    assert error.pixels == 2_000_000_000 and error.fits_input is not None
    assert error.need is not None and error.there is not None and error.need > error.there
    assert error.instead == Setting("resolution")
    # The host's memory.
    error = refused_by(EIGHT_K, card(24), round(4 * GIB))
    assert type(error) is HostShort and error.kind == "host" and error.phase == "host"
    assert error.available == round(4 * GIB) and error.need is not None
    assert error.short == error.need - round(4 * GIB) > 0


def test_a_picture_s_refusal_in_the_library_s_words_names_no_option() -> None:
    # The plain words name no option; the command line's say gives back the option each names
    # (the input's names a smaller --resolution, which the image server has no option for).
    error = refused_by(FOUR_K, Reading(0, None, 0, False), pins=Pins(swap_blocks=37))
    assert str(error).startswith("swapped blocks 37: the DiT has 36 blocks")
    error = refused_by((1920, 1080), card(11), pins=Pins(decode_tile=0))
    assert str(error).startswith("decode tile 0: the decode untiled")
    assert str(error).endswith("the largest tile that fits is decode tile 1088")
    assert cli_words(error).endswith("the largest tile that fits is --decode-tile 1088")
    error = refused_by(HD, held(16.0), input_pixels=2_000_000_000)
    assert "a smaller resolution" in str(error) and "a smaller --resolution" in cli_words(error)
    for each in (
        refused_by(EIGHT_K, card(10)),
        refused_by(EIGHT_K, card(24), round(4 * GIB)),
        refused_by(FOUR_K, card(19), pins=Pins(swap_blocks=0)),
        error,
    ):
        assert "--" not in str(each)
    # The tight retry's rewording keeps the kind and the facts (the words are the first's).
    reading = held(8.5)
    first = refused_by(HD, replace(reading, tight=True), input_pixels=2_000_000_000)
    again = refused_by(HD, reading, input_pixels=2_000_000_000)
    assert type(again) is type(first) and again.phase == first.phase
    assert str(again).endswith("whose figures these are") and "--" not in str(again)

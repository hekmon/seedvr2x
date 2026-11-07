"""The reservation (reservation.py; DESIGN.md, The image server, Its start) on the CPU, no torch:
its arithmetic exactly (reserved, share, whole, ceilings, body_limit, host_process), the plans
for a 4K bound from inputs of 50 MP on cards emulated by their free memory (Reading(free, None,
free, False), a driver that refuses; core/tests/test_still_plan.py's way), the refusal naming
the most pictures and the largest size that would fit, `largest` against a linear walk of the
same candidates, and the number of plans each try takes.

The tries (`reservation._made` makes a plan, finds the reservation over the budget, and plans
again on less, up to TRIES = 8; the next reading is lowered under what the binding phase takes:
the picture budget, or the DiT's peak at one picture when it binds), counted at 3840x2160 from
50 MP with 64 GiB of system memory, queue 8, by the plans `still_plan.plan` is asked for, over
2.0 to 100 GiB free in steps of 0.3 GiB, both attention kernels, at 1, 2 and 4 pictures:
- one picture: 1 plan where the first fits (48 GiB and up) or the plan itself refuses at once,
  2 to 5 where the first comes out short and is lowered (2 at 8 and 12 GiB with FlashAttention 2,
  3 at 16, 4 at 24 and 32); of the 654 cases, 414 take 1, 88 take 2, 64 take 3, 82 take 4 and 6
  take 5;
- two or four pictures: the same figures at both, the reservation not growing with the count
  (the server runs one picture in each model at a time), each pin tried (reservation._several:
  the plan's own tiles, then both VAE passes pinned to each of the 25 sides from 2048 to 512 px)
  taking 3 plans at the most: 1 plan where the first fits (366 cases), 2 where it is lowered and
  fits (64); 14 to 33 where the plan's own tiles move the models or reserve too much and smaller
  tiles are tried, 86 of them fitting so and 138 refused once every side was tried.
No case takes more than 5 at one picture, nor more than 3 for each pin at two or more: the tests
hold them.

The night's case on the RTX 5090 under WSL2 (2026-11-04, server_int8_n2.log and
server_fp16_n2.log): two pictures within 1920x1080 from inputs of 0.6 MP, the run held to the
card, 18.45 GiB counted of the 23.45 free on the whole card, SDPA: the int8 DiT whole with both
passes in tiles of 1216 px, the fp16 refused, naming what fits."""

import math
from dataclasses import replace

import pytest

import reservation
from reservation import Asked, ReservationRefused
from seedvr2x import memory, peaks, still_plan
from seedvr2x.errors import Refusal
from seedvr2x.memory import Reading
from seedvr2x.passes import Phase
from seedvr2x.still_plan import Bound, Peak, Pins, PlanRefusal
from seedvr2x.weights import Architecture

# pyright: reportPrivateUsage=false

GIB = peaks.GIB
FOUR_K: Bound = (3840, 2160)
INPUT = 50_000_000
SYSTEM = 64 * GIB


def reading(free_gib: float) -> Reading:
    free = round(free_gib * GIB)
    return Reading(free, None, free, False)


def asked(
    free_gib: float,
    count: int = 1,
    *,
    attention: str = peaks.FLASH_ATTN_2,
    bound: Bound = FOUR_K,
    queue: int = 8,
) -> Asked:
    return Asked(bound, INPUT, count, queue, reading(free_gib), SYSTEM, attention, peaks.FP16)


ATTENTIONS = [peaks.FLASH_ATTN_2, peaks.SDPA]
VAE: tuple[Phase, Phase] = ("encode", "decode")


class Counted:
    """still_plan.plan replaced by a wrapper that counts the plans asked for."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.count = 0
        original = still_plan.plan

        def counting(*args: object, **kwargs: object) -> still_plan.StillPlan:
            self.count += 1
            return original(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(still_plan, "plan", counting)

    def tries(self, wanted: Asked, bound: Bound, count: int) -> tuple[int, str]:
        self.count = 0
        try:
            reservation._made(wanted, bound, count)
            outcome = "fits"
        except PlanRefusal:
            outcome = "plan refused"
        except ReservationRefused:
            outcome = "reservation refused"
        return self.count, outcome


# ---- The arithmetic ------------------------------------------------------------------------------

SYNTHETIC: dict[Phase, Peak] = {
    "encode": Peak(10 * GIB, 4 * GIB, 1 * GIB),
    "dit": Peak(8 * GIB, 4 * GIB, 0),
    "decode": Peak(12 * GIB, 3 * GIB, 1 * GIB),
}


def test_a_peaks_reserved_figure_is_its_peak_its_room_and_the_share_over_the_models() -> None:
    peak = SYNTHETIC["encode"]
    assert peak.reserved == peak.allocated + peak.room + math.ceil(peaks.RESERVED_OVER * 6 * GIB)


def test_one_picture_reserves_its_largest_phase() -> None:
    assert reservation.reserved(SYNTHETIC, 1) == max(p.reserved for p in SYNTHETIC.values())


def test_a_share_is_the_largest_phase_above_the_models() -> None:
    shares = {phase: peak.reserved - peak.models for phase, peak in SYNTHETIC.items()}
    assert reservation.share(SYNTHETIC) == max(shares.values())
    assert max(shares, key=lambda phase: shares[phase]) == "decode"


@pytest.mark.parametrize("count", [2, 3, 4])
def test_several_pictures_reserve_the_models_one_vae_share_and_the_dits_share(count: int) -> None:
    # The server runs one picture in each model at a time: the DiT phase's models (4 GiB), the
    # larger of the encode's and the decode's reserved less their models (8.2 and 11.8 GiB, the
    # decode's: 12 + 1 + 0.2 x 9 less 3), and the DiT's (8 + 0.2 x 4 less 4: 4.8 GiB), whatever
    # the count.
    decode = 12 * GIB + 1 * GIB + math.ceil(peaks.RESERVED_OVER * 9 * GIB) - 3 * GIB
    dit = 8 * GIB + math.ceil(peaks.RESERVED_OVER * 4 * GIB) - 4 * GIB
    assert reservation.reserved(SYNTHETIC, count) == 4 * GIB + decode + dit


def test_the_larger_vae_phase_gives_the_vae_share() -> None:
    swapped: dict[Phase, Peak] = {**SYNTHETIC, "encode": Peak(30 * GIB, 4 * GIB, 1 * GIB)}
    encode = 30 * GIB + 1 * GIB + math.ceil(peaks.RESERVED_OVER * 26 * GIB) - 4 * GIB
    dit = 8 * GIB + math.ceil(peaks.RESERVED_OVER * 4 * GIB) - 4 * GIB
    assert reservation.reserved(swapped, 2) == 4 * GIB + encode + dit


def test_the_reservation_does_not_grow_with_the_count_past_two() -> None:
    found = {count: reservation.reserved(SYNTHETIC, count) for count in (2, 3, 4, 8)}
    assert len(set(found.values())) == 1
    assert found[2] > reservation.reserved(SYNTHETIC, 1)


def test_no_pictures_is_refused() -> None:
    with pytest.raises(ValueError):
        reservation.reserved(SYNTHETIC, 0)


def test_the_reservation_of_a_real_bound_follows_its_peaks() -> None:
    plan = still_plan.plan(
        FOUR_K,
        reading(96),
        SYSTEM,
        process=reservation.host_process(1, 8, INPUT, FOUR_K),
        input_pixels=INPUT,
        attention=peaks.SDPA,
        dit_model=peaks.FP16,
        split=True,
    )
    found = still_plan.bound_peaks(plan)
    assert reservation.reserved(found, 1) == max(p.reserved for p in found.values())
    vae = max(found[phase].reserved - found[phase].models for phase in ("encode", "decode"))
    dit = found["dit"].reserved - found["dit"].models
    assert reservation.reserved(found, 3) == found["dit"].models + vae + dit
    assert reservation.reserved(found, 3) == reservation.reserved(found, 2)


def test_the_body_limit_is_eight_bytes_a_pixel_and_16_mib() -> None:
    assert reservation.body_limit(0) == 16 * 1024**2
    assert reservation.body_limit(50_000_000) == 8 * 50_000_000 + 16 * 1024**2


def test_the_host_process_is_the_servers_own_plus_the_running_and_the_waiting() -> None:
    output = 3840 * 2160
    running = 12 * INPUT + 42 * output
    waiting = 12 * INPUT + reservation.body_limit(INPUT)
    assert reservation.PROCESS == round(2.4 * GIB)
    assert reservation.host_process(1, 0, INPUT, FOUR_K) == reservation.PROCESS + running
    assert reservation.host_process(2, 0, INPUT, FOUR_K) == reservation.PROCESS + 2 * running
    assert reservation.host_process(1, 8, INPUT, FOUR_K) == (
        reservation.PROCESS + running + 8 * waiting
    )
    # The bound's orientation does not matter, only its pixels.
    assert reservation.host_process(2, 3, INPUT, (2160, 3840)) == reservation.host_process(
        2, 3, INPUT, FOUR_K
    )


def test_the_figures_are_said_in_gib_and_megapixels() -> None:
    assert reservation.gib(3 * GIB // 2) == "1.50"
    assert reservation.megapixels(50_000_000) == "50.0"
    assert reservation.megapixels(8_294_400) == "8.3"  # rounded up


# ---- Ceilings and whole --------------------------------------------------------------------------


def four_k_plan(free_gib: float, *, count: int = 1) -> still_plan.StillPlan:
    return still_plan.plan(
        FOUR_K,
        reading(free_gib),
        SYSTEM,
        process=reservation.host_process(count, 8, INPUT, FOUR_K),
        input_pixels=INPUT,
        attention=peaks.FLASH_ATTN_2,
        dit_model=peaks.FP16,
        split=True,
    )


def test_the_ceilings_are_the_dits_peak_and_the_budget_for_the_vae_phases() -> None:
    plan = four_k_plan(96)
    floor = still_plan.bound_peaks(plan)
    # Held to the plan's own budget, a picture's passes fill it with their room.
    found = reservation.ceilings(plan, plan.budget)
    assert found["dit"] == floor["dit"]
    for phase in ("encode", "decode"):
        assert found[phase].allocated == plan.budget - floor[phase].room
        assert found[phase].allocated > floor[phase].allocated
        assert found[phase].models == floor[phase].models
        assert found[phase].room == floor[phase].room
    assert list(found) == list(floor)
    # Held to a picture budget under it, the VAE phases' ceilings are that budget less their room.
    budget = reservation.picture_budget(plan, INPUT)
    assert budget < plan.budget
    held = reservation.ceilings(plan, budget)
    assert held["dit"] == floor["dit"]
    for phase in ("encode", "decode"):
        assert held[phase].allocated == budget - floor[phase].room
        assert held[phase].allocated < found[phase].allocated


def test_the_ceilings_keep_a_floor_over_the_budget() -> None:
    # Where the floor passes the budget less the room (a budget lowered by hand), the floor
    # stands: a ceiling is never under what the proof takes.
    plan = four_k_plan(96)
    floor = still_plan.bound_peaks(plan)
    found = reservation.ceilings(plan, 1 * GIB)
    for phase in ("encode", "decode"):
        assert found[phase].allocated == floor[phase].allocated
        assert floor[phase].allocated > 1 * GIB - floor[phase].room


# ---- The picture budget --------------------------------------------------------------------------


def sides(limit: int, low: int = 17) -> list[int]:
    """Target sides from `low` to `limit` + 1, as core/tests/test_still_plan.py's `sides`: the
    edges, the padding's steps, the tiles' and a spread."""
    edges = {low, low + 1, 24, 25, 48, 49, 511, 512, 513, 1023, 1024, 1025, limit - 1, limit}
    spread = {low + (limit - low) * k // 9 for k in range(10)}
    return sorted(side for side in edges | spread | {limit + 1} if low <= side <= limit + 1)


def pictures(bound: Bound) -> list[tuple[int, int]]:
    long, short = bound
    return [
        target
        for height in sides(short)
        for width in sides(long)
        for target in ((height, width), (width, height))
    ]


@pytest.mark.parametrize("free", [96, 48, 24, 12])
def test_the_picture_budget_is_what_the_bounds_own_pictures_take_with_their_room(
    free: float,
) -> None:
    plan = made_plan(free, peaks.FLASH_ATTN_2)
    budget = reservation.picture_budget(plan, INPUT)
    floor = still_plan.bound_peaks(plan)
    long, short = plan.bound
    own = [
        peak.allocated + peak.room
        for target in ((short, long), (long, short))
        for phase, peak in still_plan.picture_peaks(plan, target, INPUT).items()
        if phase != "dit"
    ]
    # At least the proof's worst pass and its copies with their room, at most the plan's budget,
    # and no less than the bound's own pictures take.
    assert budget >= max(floor[phase].allocated + floor[phase].room for phase in VAE)
    assert budget >= max(own)
    assert budget <= plan.budget
    assert budget == min(
        max(max(own), *(floor[phase].allocated + floor[phase].room for phase in VAE)),
        plan.budget,
    )


@pytest.mark.parametrize("free", [96, 24])
def test_every_picture_within_the_bound_gets_passes_within_the_picture_budget(free: float) -> None:
    made = reservation.make(asked(free))
    plan, budget = made.plan, made.budget
    assert budget == reservation.picture_budget(plan, INPUT)
    ceiling = reservation.ceilings(plan, budget)
    checked = 0
    for target in pictures(plan.bound):
        for input_pixels in (INPUT, 1_000_000):
            found = still_plan.picture_peaks(plan, target, input_pixels, budget)
            for phase in VAE:
                assert found[phase].allocated + found[phase].room <= budget, (phase, target)
                assert found[phase].allocated <= ceiling[phase].allocated, (phase, target)
                assert found[phase].reserved <= ceiling[phase].reserved, (phase, target)
            assert found["dit"].allocated <= ceiling["dit"].allocated, target
            assert reservation.share(found) <= made.share, target
            checked += 1
    assert checked > 500


def test_a_4k_bound_from_50_mp_on_96_gib_reserves_a_part_of_the_card() -> None:
    made = reservation.make(asked(96))
    # Not the whole card: the models (15.87 GiB) and the room and share of the pictures the
    # bound's own take, about 45.6 GiB (40.7 before the decode's untiled frame was 4K's), where
    # the plan's budget is the card's 95.4.
    assert made.reservation < 0.5 * 96 * GIB
    assert reservation.gib(made.reservation) == "45.59"
    assert made.budget < made.plan.budget
    assert made.budget == reservation.picture_budget(made.plan, INPUT)
    assert made.reservation == reservation.reserved(reservation.ceilings(made.plan, made.budget), 1)
    # And a card half the size reserves the same: it is the bound that decides.
    assert reservation.make(asked(64)).reservation == made.reservation


@pytest.mark.parametrize("attention", ATTENTIONS)
def test_whole_where_every_model_stays_on_the_card(attention: str) -> None:
    assert reservation.whole(made_plan(96, attention)) is True
    # 24 GiB free: the DiT's blocks swap or the DiT leaves for the VAE's passes.
    assert reservation.whole(made_plan(24, attention)) is False


def made_plan(free_gib: float, attention: str) -> still_plan.StillPlan:
    return reservation.make(asked(free_gib, attention=attention)).plan


# ---- A reservation on a card ---------------------------------------------------------------------


@pytest.mark.parametrize("attention", ATTENTIONS)
def test_one_4k_picture_from_50_mp_fits_in_24_gib(attention: str) -> None:
    wanted = asked(24, attention=attention)
    made = reservation.make(wanted)
    assert made.count == 1
    assert made.plan.bound == FOUR_K
    assert made.plan.input_pixels == INPUT
    assert made.plan.attention == attention
    # The reservation is held within the budget the original reading gives the plan.
    budget = wanted.reading.counted - made.plan.margin
    assert 0 < made.reservation <= budget
    ceiling = reservation.ceilings(made.plan, made.budget)
    assert made.budget == reservation.picture_budget(made.plan, INPUT) <= made.plan.budget
    assert made.reservation == reservation.reserved(ceiling, 1)
    assert made.pool == made.share == reservation.share(ceiling)
    assert made.share <= made.reservation
    assert made.models == still_plan.bound_peaks(made.plan)["dit"].models


@pytest.mark.parametrize("attention", ATTENTIONS)
def test_two_4k_pictures_do_not_fit_in_22_gib(attention: str) -> None:
    # The 7B whole at 4K and two pictures: more than 22 GiB (a first fit between 22.5 and 22.75
    # on the lean attention; 24 GiB held two before it).
    with pytest.raises(ReservationRefused) as error:
        reservation.make(asked(22, 2, attention=attention))
    assert error.value.most == 1
    assert isinstance(error.value, Refusal)
    assert "--max-concurrent" in str(error.value)
    assert "one picture at once within 3840x2160" in str(error.value)
    largest = error.value.largest
    assert largest is not None and largest[0] < 3840 and largest[1] < 2160
    assert "--max-size" in str(error.value)


@pytest.mark.parametrize("attention", ATTENTIONS)
def test_two_4k_pictures_fit_in_96_gib(attention: str) -> None:
    made = reservation.make(asked(96, 2, attention=attention))
    assert made.count == 2
    assert reservation.whole(made.plan)
    assert made.reservation <= asked(96, 2).reading.counted - made.plan.margin
    # The models, one picture's share in the VAE and one in the DiT: not two shares of the
    # larger; the pool is the admission's, a share for each picture in flight.
    ceiling = reservation.ceilings(made.plan, made.budget)
    vae = max(ceiling[phase].reserved - ceiling[phase].models for phase in VAE)
    dit = ceiling["dit"].reserved - ceiling["dit"].models
    assert made.reservation == made.models + vae + dit
    assert made.pool == 2 * made.share
    assert made.pool >= made.reservation - made.models


def test_the_reservation_of_a_real_bound_does_not_grow_with_the_count_past_two() -> None:
    found = {count: reservation.make(asked(96, count)) for count in (1, 2, 3, 4)}
    assert found[2].reservation > found[1].reservation
    assert found[2].reservation == found[3].reservation == found[4].reservation
    # What grows is the admission's pool, a share for each picture in flight, and the picture
    # budget and the share are the bound's.
    assert [found[count].pool for count in (1, 2, 3, 4)] == [
        count * found[1].share for count in (1, 2, 3, 4)
    ]
    assert len({made.budget for made in found.values()}) == 1
    assert len({made.share for made in found.values()}) == 1


def test_several_pictures_on_a_card_that_moves_blocks_are_refused_for_the_whole_dit() -> None:
    with pytest.raises(ReservationRefused, match="whole DiT on the card") as error:
        reservation._made(asked(22, 2), FOUR_K, 2)
    assert error.value.most == 0 and error.value.largest is None


def test_one_picture_swapping_blocks_is_allowed() -> None:
    made = reservation.make(asked(16))
    assert not reservation.whole(made.plan)
    assert made.plan.models.swapped > 0 or not made.plan.models.vae_in_dit


def test_the_pins_stand_for_one_picture_and_not_for_several() -> None:
    pinned = replace(asked(96), pins=still_plan.Pins(swap_blocks=2))
    assert reservation.make(pinned).plan.models.swapped == 2
    # Several pictures plan with no block swapped, whatever the pins say.
    two = replace(asked(96, 2), pins=still_plan.Pins(swap_blocks=2))
    made = reservation._made(two, FOUR_K, 2)
    assert made.plan.models.swapped == 0 and made.plan.pins == still_plan.Pins(swap_blocks=0)


def test_a_smaller_queue_leaves_the_reservation_room() -> None:
    # The pictures waiting hold their bodies in system memory: the plan's `process`.
    full = reservation.make(asked(24, queue=8)).plan
    none = reservation.make(asked(24, queue=0)).plan
    assert full.process > none.process
    assert full.process - none.process == 8 * (12 * INPUT + reservation.body_limit(INPUT))


# ---- What would fit ------------------------------------------------------------------------------


def candidates(bound: Bound) -> list[Bound]:
    """largest()'s candidates written apart: from the multiple of 16 under the short side down
    in steps of 16, the long side in proportion to an even number, none under 32."""
    long, short = bound
    return [
        (side * long // short // 2 * 2, side)
        for side in range((short - 1) // 16 * 16, 0, -16)
        if side >= 32
    ]


@pytest.mark.parametrize(
    ("bound", "free"),
    [(FOUR_K, 4.25), (FOUR_K, 4.5), (FOUR_K, 4.4), ((1920, 1080), 4.25), ((1920, 1080), 4.5)],
)
def test_the_largest_bound_is_the_first_that_fits_walking_down(bound: Bound, free: float) -> None:
    wanted = asked(free, bound=bound)
    assert not reservation.fits(wanted, bound, 1)
    walk = [reservation.fits(wanted, each, 1) for each in candidates(bound)]
    # A smaller bound never takes more: false for the larger, then true for the smaller.
    assert walk == sorted(walk)
    assert any(walk)
    found = reservation.largest(wanted)
    first = walk.index(True)
    assert found == candidates(bound)[first]
    assert found is not None
    # At the asked ratio, and the next candidate up does not fit.
    assert found[0] == found[1] * bound[0] // bound[1] // 2 * 2
    assert first > 0
    assert not reservation.fits(wanted, candidates(bound)[first - 1], 1)
    assert reservation.fits(wanted, found, 1)


def test_the_largest_bound_is_found_in_few_plans(monkeypatch: pytest.MonkeyPatch) -> None:
    counter = Counted(monkeypatch)
    wanted = asked(4.5)
    total = len(candidates(FOUR_K))
    reservation.largest(wanted)
    # Bisected: about log2(candidates) probes of 5 plans at the most, not a walk of them all.
    assert counter.count <= 5 * (math.ceil(math.log2(total)) + 1)
    assert counter.count < total


def test_the_refusal_names_what_would_fit() -> None:
    with pytest.raises(ReservationRefused) as error:
        reservation.make(asked(4.5))
    assert error.value.most == 0
    found = error.value.largest
    assert found is not None
    assert f"{found[0]}x{found[1]}" in str(error.value)
    assert "(--max-size)" in str(error.value)
    assert reservation.fits(asked(4.5), found, 1)


def test_a_card_too_small_for_any_picture_is_refused_saying_so() -> None:
    with pytest.raises(ReservationRefused) as error:
        reservation.make(asked(1.0))
    assert error.value.most == 0
    assert error.value.largest is None
    assert "no picture fits" in str(error.value)
    assert reservation.most(asked(1.0, 2)) == 0
    assert reservation.largest(asked(1.0)) is None


def test_most_is_the_count_that_fits_up_to_the_asked_one() -> None:
    assert reservation.most(asked(22, 4)) == 1
    assert reservation.most(asked(24, 1)) == 1
    assert reservation.most(asked(96, 1)) == 1


def test_fits_says_a_refusal_as_false() -> None:
    assert reservation.fits(asked(24), FOUR_K, 1)
    assert not reservation.fits(asked(22, 2), FOUR_K, 2)
    assert not reservation.fits(asked(2), FOUR_K, 1)


# ---- The tries -----------------------------------------------------------------------------------


def most_tries(count: int) -> int:
    """The most plans a start takes at `count` pictures for each pin it tries (the file's
    docstring)."""
    return 5 if count == 1 else 3


class Tried:
    """reservation._tried replaced by a wrapper that counts the plans each pin takes."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, counter: Counted) -> None:
        self.each: list[int] = []
        original = reservation._tried

        def counting(*args: object) -> reservation.Made:
            before = counter.count
            try:
                return original(*args)  # type: ignore[arg-type]
            finally:
                self.each.append(counter.count - before)

        monkeypatch.setattr(reservation, "_tried", counting)


@pytest.mark.parametrize("count", [1, 2, 4])
@pytest.mark.parametrize("free", [2, 6, 8, 12, 16, 24, 32, 48, 96])
def test_the_tries_stay_under_the_limit(
    free: int, count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = Counted(monkeypatch)
    tried = Tried(monkeypatch, counter)
    plans, _ = counter.tries(asked(free, count), FOUR_K, count)
    assert 1 <= len(tried.each) <= (1 if count == 1 else len(reservation._several(FOUR_K)))
    assert sum(tried.each) == plans
    assert all(1 <= each <= reservation.TRIES for each in tried.each)
    assert max(tried.each) <= most_tries(count)  # the figures in this file's docstring


@pytest.mark.parametrize("count", [1, 2, 4])
def test_the_tries_stay_under_the_limit_on_cards_between(
    count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 2.0 to 100 GiB free in steps of 0.3, both kernels: the file's docstring counts them.
    counter = Counted(monkeypatch)
    tried = Tried(monkeypatch, counter)
    seen: dict[int, int] = {}
    outcomes: dict[str, int] = {}
    for tenths in range(20, 1001, 3):
        for attention in ATTENTIONS:
            tried.each = []
            wanted = asked(tenths / 10, count, attention=attention)
            plans, outcome = counter.tries(wanted, FOUR_K, count)
            assert max(tried.each) <= most_tries(count), (tenths, attention)
            seen[plans] = seen.get(plans, 0) + 1
            if outcome == "fits" and count > 1:
                pinned = reservation._made(wanted, FOUR_K, count).plan.pins.encode_tile
                outcome = "fits" if pinned is None else "fits pinned"
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
    assert sum(seen.values()) == 654
    if count == 1:
        # {2: 88, 4: 82} before the 4K decode's ratio at 832 px was its measure's (2026-11-07).
        assert seen == {1: 414, 2: 86, 3: 64, 4: 84, 5: 6}
    else:
        assert seen[1] == 366 and seen[2] == 64
        assert min(plans for plans in seen if plans > 2) == 14 and max(seen) == 33
        assert outcomes == {"fits": 430, "fits pinned": 86, "reservation refused": 138}


def test_the_tries_of_a_fit_follow_how_far_the_first_plan_is_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The first plan counts the whole budget; the next is lowered under what the binding phase
    # takes (the picture budget, or the DiT's peak at one picture when it binds): a card the
    # bound's pictures fill once needs 1 plan (48 GiB and up), a smaller one lowers the plan
    # again, 2 to 4 times by 8 to 32 GiB.
    counter = Counted(monkeypatch)
    expected = {8: 2, 12: 2, 16: 3, 24: 4, 32: 4, 48: 1, 96: 1}
    for free, plans in expected.items():
        assert counter.tries(asked(free), FOUR_K, 1) == (plans, "fits"), free


@pytest.mark.parametrize("count", [2, 4])
def test_several_pictures_take_as_many_tries_as_two_and_never_more_than_three(
    count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = Counted(monkeypatch)
    tried = Tried(monkeypatch, counter)
    # 52 GiB and up take the first plan: the 4K bound's decode untiled (patch 04) is a larger
    # share of the budget, so that 44 and 48 GiB, which took it before, lower the plan once.
    for free in (52, 96):
        assert counter.tries(asked(free, count), FOUR_K, count) == (1, "fits")
    # Below, no tile keeps the DiT whole with a reservation that fits: every pin tried, 3 plans
    # at the most each, refused for the whole DiT.
    tried.each = []
    plans, outcome = counter.tries(asked(22, count), FOUR_K, count)
    assert outcome == "reservation refused" and plans == sum(tried.each)
    assert len(tried.each) == len(reservation._several(FOUR_K)) and max(tried.each) <= 3
    tried.each = []
    counter.tries(asked(32, count), FOUR_K, count)
    assert max(tried.each) <= 3


def test_a_reading_that_never_fits_ends_in_a_refusal_not_a_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Whatever the plan asks, the tries end: a plan that always comes out over its budget ends
    # in the reservation's refusal after TRIES plans.
    counter = Counted(monkeypatch)
    original = reservation.ceilings

    def over(plan: still_plan.StillPlan, budget: int) -> dict[Phase, Peak]:
        found = original(plan, budget)
        return {
            phase: Peak(peak.allocated + 1000 * GIB, peak.models, peak.room)
            for phase, peak in found.items()
        }

    monkeypatch.setattr(reservation, "ceilings", over)
    with pytest.raises(ReservationRefused, match="doesn't fit the memory"):
        reservation._made(asked(96), FOUR_K, 1)
    assert counter.count <= reservation.TRIES


# ---- Several pictures at once: the VAE's passes tiled smaller -----------------------------------

FULL_HD: Bound = (1920, 1080)
# The night's inputs, 1024x576, 0.6 MP (the warm-up's).
NIGHT_INPUT = 1024 * 576


def night(
    dit: Architecture, count: int, allocator: str = memory.PLAIN, counted: float = 18.45
) -> Asked:
    """The RTX 5090 under WSL2 (2026-11-04, the n2 runs' logs): the run held to the card, "the
    plan counts on 18.45 GiB of the GPU's memory, of the 23.45 GiB free on the whole card by NVML
    (its driver reports 30.03 GiB free to this process)", 0.5 left outside torch's limit; SDPA,
    FlashAttention 2 not installed; two pictures within 1920x1080 from 0.6 MP. The 18.45 GiB is
    the card's 23.45 less the 0.5 and the flat 4.5 GiB reserve: the figure of torch's own
    allocator without the segments (memory.PLAIN), which is what these tests hold (the plans of
    that night, unchanged). Under the segments the same card counts 22.95 GiB (memory.counted),
    and `counted` names another figure for the tests of the per-step rule."""
    held = Reading(
        round(30.03 * GIB),
        round(23.45 * GIB),
        round(counted * GIB),
        True,
        memory.HELD_BACK,
        allocator,
    )
    return Asked(FULL_HD, NIGHT_INPUT, count, 8, held, SYSTEM, peaks.SDPA, dit)


@pytest.mark.parametrize("bound", [FULL_HD, FOUR_K, (1280, 720), (1918, 1078), (2048, 1152)])
def test_the_tiles_tried_are_the_plans_own_sizes(bound: Bound) -> None:
    sides = still_plan._tile_sizes(max(still_plan._frames(bound), key=max))
    assert reservation.tile_sides(bound) == sides
    assert reservation._several(bound) == [
        Pins(swap_blocks=0),
        *(Pins(side, side, swap_blocks=0) for side in sides),
    ]
    # A bound too small for a tile: the plan's own tiles alone.
    assert reservation._several((128, 96)) == [Pins(swap_blocks=0)]


def test_two_pictures_on_the_int8_dit_keep_it_whole_in_tiles_of_1216() -> None:
    # Refused that night: the plan's own tiles kept the DiT whole at the memory counted, but
    # reserved more than it, and lowered the DiT beside the decode on the memory lowered.
    wanted = night(peaks.INT8, 2)
    first = reservation._plan(wanted, FULL_HD, 2, wanted.reading, Pins(swap_blocks=0))
    assert reservation.whole(first)
    ceiling = reservation.ceilings(first, reservation.picture_budget(first, NIGHT_INPUT))
    assert reservation.reserved(ceiling, 2) > first.budget
    with pytest.raises(reservation._Moves, match="the plan moves the models"):
        reservation._tried(wanted, FULL_HD, 2, Pins(swap_blocks=0))
    made = reservation.make(wanted)
    assert reservation.whole(made.plan)
    assert made.plan.pins == Pins(1216, 1216, swap_blocks=0)
    assert made.reservation <= wanted.reading.counted - made.plan.margin
    assert reservation.gib(made.reservation) == "18.43"
    assert (
        reservation.gib(made.models) == "8.29"
    )  # the int8 DiT and the VAE, as the plan counts them
    assert made.reservation == reservation.reserved(reservation.ceilings(made.plan, made.budget), 2)
    # The first side that does: every larger one moves the models or reserves too much.
    for side in reservation.tile_sides(FULL_HD):
        if side == 1216:
            break
        with pytest.raises(ReservationRefused):
            reservation._tried(wanted, FULL_HD, 2, Pins(side, side, swap_blocks=0))
    # Every picture within the bound has its passes, in those tiles or untiled where its frame is
    # no larger, within the picture budget the reservation counts.
    for target in ((1080, 1920), (1920, 1080), (720, 1280), (360, 640)):
        tiles = still_plan.tiles(made.plan, target, made.budget)
        for phase, chosen in (("encode", tiles.encode), ("decode", tiles.decode)):
            assert chosen is None or chosen.size == 1216, (target, phase)
    small = still_plan.tiles(made.plan, (360, 640), made.budget)
    assert small.encode is None and small.decode is None
    # Four at once reserve the same: one picture in each model at a time.
    assert reservation.make(night(peaks.INT8, 4)).reservation == made.reservation


def test_two_pictures_on_the_fp16_dit_are_refused_naming_what_fits() -> None:
    # Its 15.87 GiB of DiT and two pictures' shares pass the 18.45 GiB counted at every tile:
    # refused, naming one picture at that size and the largest size at two.
    wanted = night(peaks.FP16, 2)
    for pins in reservation._several(FULL_HD):
        with pytest.raises(ReservationRefused):
            reservation._tried(wanted, FULL_HD, 2, pins)
    with pytest.raises(ReservationRefused) as error:
        reservation.make(wanted)
    assert str(error.value) == (
        "2 pictures at once need the whole DiT on the card, every block of it in every phase and"
        " the VAE beside it, and no plan for pictures within 1920x1080 keeps it there with their"
        " reservation in the memory the plan counts on, the VAE's passes tiled down to 512 px;"
        " what would fit: one picture at once within 1920x1080 (--max-concurrent), or 2 pictures"
        " at once within 540x304 (--max-size)"
    )
    assert error.value.most == 1 and error.value.largest == (540, 304)


@pytest.mark.parametrize("dit", [peaks.INT8, peaks.FP16])
def test_one_picture_takes_the_plans_own_tiles(dit: Architecture) -> None:
    # One picture at a time: the asked pins alone, none here, the plan's own tiles and placement
    # (the night's: the int8 DiT lowered to 35 blocks beside the decode, the fp16 to 23 and 17
    # beside the encode and the decode, the VAE now in the DiT's phase for both; 30, and 22 and 15,
    # before the decode's untiled frame was 4K's, 2026-11-06), as before.
    wanted = night(dit, 1)
    made = reservation.make(wanted)
    assert made.plan.pins == still_plan.NO_PINS
    assert made == reservation._tried(wanted, FULL_HD, 1, still_plan.NO_PINS)
    models = made.plan.models
    if dit == peaks.INT8:
        assert (models.resident_encode, models.resident_decode, models.vae_in_dit) == (
            None,
            35,
            True,
        )
        assert reservation.gib(made.reservation) == "18.27"
    else:
        assert (models.resident_encode, models.resident_decode, models.vae_in_dit) == (
            23,
            17,
            True,
        )
        assert reservation.gib(made.reservation) == "18.45"


def test_the_wsl2_card_under_the_segments_keeps_more_of_the_dit_than_under_the_flat_reserve() -> (
    None
):
    # The same card, torch's own allocator with its segments (the default on a lending driver):
    # counted 22.95 GiB, no flat 4.5 off, each phase judged by its peak and its own reserve
    # (peaks.reserve), no share for torch's slack added to the reservation (Peak.over 0). Every
    # figure is what the planner gives; the direction: never less of the DiT on the card, never
    # a smaller tile, a pair refused under the flat reserve served.
    plain_int8 = reservation.make(night(peaks.INT8, 1))
    for allocator, count, dit, expected in (
        (memory.SEGMENTS, 1, peaks.INT8, ((None, None), "17.23")),
        (memory.SEGMENTS, 1, peaks.FP16, ((None, 31), "22.71")),
        (memory.POOL, 1, peaks.INT8, ((None, None), "18.35")),
        (memory.POOL, 1, peaks.FP16, ((35, 28), "22.56")),
    ):
        found = reservation.make(night(dit, count, allocator, 22.95))
        models = found.plan.models
        assert (
            (models.resident_encode, models.resident_decode),
            reservation.gib(found.reservation),
        ) == expected, (
            allocator,
            dit.layout,
        )
        assert found.plan.pins == still_plan.NO_PINS and models.swapped == 0
        assert found.reservation <= found.plan.reading.counted - found.plan.margin
        assert all(each.over == 0.0 for each in still_plan.bound_peaks(found.plan).values())
    assert (
        plain_int8.plan.models.resident_decode == 35
    )  # the flat reserve's, lowered beside the decode
    # Two pictures keep the DiT whole in the plan's own tiles on the int8 file (18.82 GiB) where
    # the flat reserve needed 1216-px tiles (18.43), and the fp16 pair, refused under the flat
    # reserve, is served in 832-px tiles (22.67 GiB: 22.46 before the 4K decode's ratio at 832 px
    # was its measure's, 0.347 for 768's 0.33, 2026-11-07).
    two = reservation.make(night(peaks.INT8, 2, memory.SEGMENTS, 22.95))
    assert two.plan.pins == Pins(swap_blocks=0) and reservation.gib(two.reservation) == "18.82"
    pooled = reservation.make(night(peaks.INT8, 2, memory.POOL, 22.95))
    assert (
        pooled.plan.pins == Pins(swap_blocks=0) and reservation.gib(pooled.reservation) == "20.19"
    )
    pair = reservation.make(night(peaks.FP16, 2, memory.SEGMENTS, 22.95))
    assert pair.plan.pins == Pins(832, 832, swap_blocks=0)
    assert reservation.gib(pair.reservation) == "22.67"
    with pytest.raises(ReservationRefused) as flat:
        reservation.make(night(peaks.FP16, 2))
    assert flat.value.most == 1 and flat.value.largest == (540, 304)


def test_the_mechanism_of_the_tiles_tried_holds_under_the_segments_on_a_card_that_counts_less() -> (
    None
):
    # The int8 pair of the night, on the segments with 19.5 GiB counted (another program holds
    # about 5.5 GiB of the card; 19.5, the figure before the decode's untiled frame was 4K's,
    # now keeps the plan's own tiles): the plan's own tiles keep the DiT whole but reserve more
    # than the budget, so 1152-px tiles are the first that fit, as under the flat reserve at 18.45.
    wanted = night(peaks.INT8, 2, memory.SEGMENTS, 17.5)
    first = reservation._plan(wanted, FULL_HD, 2, wanted.reading, Pins(swap_blocks=0))
    assert reservation.whole(first)
    ceiling = reservation.ceilings(first, reservation.picture_budget(first, NIGHT_INPUT))
    assert reservation.reserved(ceiling, 2) > first.budget
    with pytest.raises(reservation._Moves, match="the plan moves the models"):
        reservation._tried(wanted, FULL_HD, 2, Pins(swap_blocks=0))
    made = reservation.make(wanted)
    assert made.plan.pins == Pins(1152, 1152, swap_blocks=0)
    assert reservation.gib(made.reservation) == "17.22"
    assert made.reservation <= wanted.reading.counted - made.plan.margin
    # The fp16 pair does not fit there at all (22.46 GiB on 22.95): one picture, or 768x432.
    with pytest.raises(ReservationRefused) as error:
        reservation.make(night(peaks.FP16, 2, memory.SEGMENTS, 19.5))
    assert error.value.most == 1 and error.value.largest == (768, 432)


def test_a_reservation_adds_the_slack_over_the_peak_under_torchs_own_allocator_alone() -> None:
    # Peak.over: RESERVED_OVER where the reserve is the flat 4.5 off the figure (a phase's room
    # is not the whole of what torch leaves), nothing under the segments and the pool, whose
    # reserve beyond the peak is the whole of it. The same bound, the same card (24 GiB): the
    # reservation under the segments is the phases' peaks and reserves, to the byte.
    for allocator, over in (
        (memory.PLAIN, peaks.RESERVED_OVER),
        (memory.SEGMENTS, 0.0),
        (memory.POOL, 0.0),
    ):
        assert still_plan.reserved_share(allocator) == over
        found = reservation.make(night(peaks.INT8, 1, allocator, 22.0))
        for phase, each in still_plan.bound_peaks(found.plan).items():
            assert each.over == over, (allocator, phase)
            assert each.reserved == each.allocated + each.room + math.ceil(over * each.above)
    # The growth rate the tries lower the budget by: the allocator's largest share of P.
    assert reservation._growth(memory.PLAIN) == peaks.RESERVED_OVER
    assert reservation._growth(memory.SEGMENTS) == peaks.DIT_RESERVE_SHARE
    assert reservation._growth(memory.POOL) == peaks.POOL_VAE
    assert reservation._growth(memory.SEGMENTS, 0.5) == 0.5  # a margin the probe measured


# ---- The smaller DiT files a refused start names ------------------------------------------------

SHARP_INT8 = "int8 (--dit-model seedvr2x_ema_7b_sharp_int8_convrot.safetensors, 8.3 GB)"
SHARP_GGUF = "the dynamic GGUF (--dit-model seedvr2x_ema_7b_sharp_dyn.gguf, 4.8 GB)"


def two_in_full_hd(free_gib: float, dit: Architecture, name: str) -> Asked:
    """Two pictures within 1920x1080 from 0.6 MP on `free_gib` GiB, the DiT file `name` in the
    format dit (SDPA): they need the whole DiT on the card, which fp16 never gets in 18 GiB or
    less, int8 from 14 and the GGUF from 10 (measured by the plan, the tests below)."""
    held = asked(free_gib, 2, bound=FULL_HD, attention=peaks.SDPA)
    return replace(held, dit_model=dit, input_pixels=NIGHT_INPUT, dit_file=name)


def test_a_refused_start_names_the_smaller_files_that_fit() -> None:
    # DESIGN.md, Weights, "The dynamic GGUF before v0.1.0 too": where the card is short for the
    # fp16 DiT, the start's refusal ends with the smaller files of the same 7B that would fit the
    # same bound and count, found by planning it again in their figures. The refusal's own words
    # stay before them.
    sharp = "seedvr2x_ema_7b_sharp_fp16.safetensors"
    for free, named in (
        (16, f"{SHARP_INT8} or {SHARP_GGUF}"),
        (12, SHARP_GGUF),
    ):
        wanted = two_in_full_hd(free, peaks.FP16, sharp)
        with pytest.raises(ReservationRefused) as error:
            reservation.make(wanted)
        with pytest.raises(ReservationRefused) as plain:
            reservation.make(replace(wanted, dit_file=""))
        assert (
            str(error.value)
            == f"{plain.value}; or a smaller DiT file that plans this bound: {named}"
        )
        assert (error.value.most, error.value.largest) == (plain.value.most, plain.value.largest)
    # Where the server was given --model-dir, the way out says once that the file goes there; the
    # pinned names alone would send the user to a directory that may not hold them.
    wanted = replace(two_in_full_hd(16, peaks.FP16, sharp), model_dir=True)
    with pytest.raises(ReservationRefused) as error:
        reservation.make(wanted)
    assert str(error.value).endswith(
        f"; or a smaller DiT file that plans this bound, put in --model-dir: {SHARP_INT8} or"
        f" {SHARP_GGUF}"
    )
    assert str(error.value).count("--model-dir") == 1
    # The regular 7B's file gets the regular 7B's files.
    regular = two_in_full_hd(16, peaks.FP16, "seedvr2x_ema_7b_fp16.safetensors")
    with pytest.raises(ReservationRefused) as error:
        reservation.make(regular)
    assert str(error.value).endswith(
        "int8 (--dit-model seedvr2x_ema_7b_int8_convrot.safetensors, 8.3 GB) or the dynamic GGUF"
        " (--dit-model seedvr2x_ema_7b_dyn.gguf, 4.8 GB)"
    )
    # An int8 DiT is offered the GGUF alone; the GGUF DiT nothing; nothing that fits, nothing.
    int8 = two_in_full_hd(12, peaks.INT8, "seedvr2x_ema_7b_sharp_int8_convrot.safetensors")
    with pytest.raises(ReservationRefused) as error:
        reservation.make(int8)
    assert str(error.value).endswith(f"a smaller DiT file that plans this bound: {SHARP_GGUF}")
    for held in (
        two_in_full_hd(9, peaks.GGUF_7B_SHARP, "seedvr2x_ema_7b_sharp_dyn.gguf"),
        two_in_full_hd(9, peaks.FP16, sharp),
    ):
        with pytest.raises(ReservationRefused) as error:
            reservation.make(held)
        assert "smaller DiT file" not in str(error.value)
    # A start that fits is untouched: the same plan and reservation with or without a name.
    fits = two_in_full_hd(16, peaks.INT8, "seedvr2x_ema_7b_sharp_int8_convrot.safetensors")
    assert reservation.make(fits) == reservation.make(replace(fits, dit_file=""))


# ---- The smaller DiT files the start's plan line names -------------------------------------------


def one_in_full_hd(free_gib: float, dit: Architecture, name: str) -> Asked:
    """One picture within 1920x1080 from 0.6 MP on `free_gib` GiB, the DiT file `name` in the
    format dit (SDPA). The plan swaps blocks of fp16's 7B below about 20 GiB (21 at 10, 17 at 12,
    3 at 18), of int8's 5 at 10 and none from 11, and none of the GGUF's from 7 (the plan, the
    probe behind these tests)."""
    return replace(two_in_full_hd(free_gib, dit, name), count=1)


def line(asked: Asked) -> str | None:
    return reservation.smaller_line(asked, reservation.make(asked))


def test_the_start_says_the_smaller_files_that_swap_fewer_blocks() -> None:
    # DESIGN.md, Weights, "The dynamic GGUF before v0.1.0 too": the image server's start says what
    # the command line's plan says, a picture being one latent: blocks alone, no window.
    sharp = "seedvr2x_ema_7b_sharp_fp16.safetensors"
    assert line(one_in_full_hd(10, peaks.FP16, sharp)) == (
        "smaller DiT files plan this bound better: this plan swaps 21 blocks; int8 5 blocks;"
        " the dynamic GGUF 0 blocks; --dit-model names them"
    )
    assert line(one_in_full_hd(12, peaks.FP16, sharp)) == (
        "smaller DiT files plan this bound better: this plan swaps 16 blocks; int8 0 blocks;"
        " the dynamic GGUF 0 blocks; --dit-model names them"
    )
    # An int8 DiT is offered the GGUF alone.
    assert line(
        one_in_full_hd(10, peaks.INT8, "seedvr2x_ema_7b_sharp_int8_convrot.safetensors")
    ) == (
        "smaller DiT files plan this bound better: this plan swaps 5 blocks; the dynamic GGUF 0"
        " blocks; --dit-model names them"
    )
    # The regular 7B's file gets the regular 7B's files.
    regular = line(one_in_full_hd(12, peaks.FP16, "seedvr2x_ema_7b_fp16.safetensors"))
    assert regular is not None and "seedvr2x_ema_7b_sharp" not in regular
    assert regular.startswith(
        "smaller DiT files plan this bound better: this plan swaps 16 blocks;"
    )


def test_the_start_says_nothing_where_no_smaller_file_swaps_fewer() -> None:
    sharp = "seedvr2x_ema_7b_sharp_fp16.safetensors"
    # The GGUF is the smallest format; the plan swaps none; the name is unknown; several pictures
    # at once keep the DiT whole, so their plan swaps none.
    assert line(one_in_full_hd(10, peaks.GGUF_7B_SHARP, "seedvr2x_ema_7b_sharp_dyn.gguf")) is None
    assert line(one_in_full_hd(24, peaks.FP16, sharp)) is None
    assert line(one_in_full_hd(12, peaks.FP16, "")) is None
    assert (
        line(two_in_full_hd(16, peaks.INT8, "seedvr2x_ema_7b_sharp_int8_convrot.safetensors"))
        is None
    )


def test_an_unpinned_dit_name_gives_both_7bs_files_in_the_start_line() -> None:
    # A file that is not a pinned one does not say which 7B it holds (pull.smaller_files): each
    # 7B's files, in brackets, the sharp 7B's first.
    assert line(one_in_full_hd(12, peaks.FP16, "w.safetensors")) == (
        "smaller DiT files plan this bound better: this plan swaps 16 blocks; for the sharp 7B"
        " (int8 0 blocks; the dynamic GGUF 0 blocks); for the 7B (int8 0 blocks; the dynamic GGUF"
        " 0 blocks); --dit-model names them"
    )

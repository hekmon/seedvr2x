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
  3 at 16, 4 at 24 and 32); of the 654 cases, 449 take 1, 64 take 2, 55 take 3, 80 take 4 and 6
  take 5;
- two or four pictures: the same figures at both, the reservation not growing with the count
  (the server runs one picture in each model at a time): 1 plan where the first fits or the plan
  moves blocks (refused for the whole DiT), 2 where it is lowered and fits, 3 at the most (a
  refusal after two lowerings).
No case takes more than 5, and none more than 3 at two or more pictures: the tests hold them."""

import math
from dataclasses import replace

import pytest

import reservation
from reservation import Asked, ReservationRefused
from seedvr2x import peaks, still_plan
from seedvr2x.errors import Refusal
from seedvr2x.memory import Reading
from seedvr2x.passes import Phase
from seedvr2x.still_plan import Bound, Peak, PlanRefusal

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
    # bound's own take, about 40.7 GiB, where the plan's budget is the card's 95.4.
    assert made.reservation < 0.5 * 96 * GIB
    assert reservation.gib(made.reservation) == "40.66"
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
def test_two_4k_pictures_do_not_fit_in_24_gib(attention: str) -> None:
    # The 7B whole at 4K and two pictures: more than 24 GiB.
    with pytest.raises(ReservationRefused) as error:
        reservation.make(asked(24, 2, attention=attention))
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
        reservation._made(asked(24, 2), FOUR_K, 2)
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
    assert reservation._plan(two, FOUR_K, 2, two.reading).models.swapped == 0


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
    [(FOUR_K, 4.5), (FOUR_K, 5.0), (FOUR_K, 6.0), (FOUR_K, 6.5), ((1920, 1080), 4.5)],
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
    wanted = asked(6.0)
    total = len(candidates(FOUR_K))
    reservation.largest(wanted)
    # Bisected: about log2(candidates) probes of 5 plans at the most, not a walk of them all.
    assert counter.count <= 5 * (math.ceil(math.log2(total)) + 1)
    assert counter.count < total


def test_the_refusal_names_what_would_fit() -> None:
    with pytest.raises(ReservationRefused) as error:
        reservation.make(asked(6.0))
    assert error.value.most == 0
    found = error.value.largest
    assert found is not None
    assert f"{found[0]}x{found[1]}" in str(error.value)
    assert "(--max-size)" in str(error.value)
    assert reservation.fits(asked(6.0), found, 1)


def test_a_card_too_small_for_any_picture_is_refused_saying_so() -> None:
    with pytest.raises(ReservationRefused) as error:
        reservation.make(asked(1.0))
    assert error.value.most == 0
    assert error.value.largest is None
    assert "no picture fits" in str(error.value)
    assert reservation.most(asked(1.0, 2)) == 0
    assert reservation.largest(asked(1.0)) is None


def test_most_is_the_count_that_fits_up_to_the_asked_one() -> None:
    assert reservation.most(asked(24, 4)) == 1
    assert reservation.most(asked(24, 1)) == 1
    assert reservation.most(asked(96, 1)) == 1


def test_fits_says_a_refusal_as_false() -> None:
    assert reservation.fits(asked(24), FOUR_K, 1)
    assert not reservation.fits(asked(24, 2), FOUR_K, 2)
    assert not reservation.fits(asked(2), FOUR_K, 1)


# ---- The tries -----------------------------------------------------------------------------------


def most_tries(count: int) -> int:
    """The most plans a start takes at `count` pictures (the file's docstring)."""
    return 5 if count == 1 else 3


@pytest.mark.parametrize("count", [1, 2, 4])
@pytest.mark.parametrize("free", [2, 6, 8, 12, 16, 24, 32, 48, 96])
def test_the_tries_stay_under_the_limit(
    free: int, count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = Counted(monkeypatch)
    plans, _ = counter.tries(asked(free, count), FOUR_K, count)
    assert 1 <= plans <= reservation.TRIES
    assert plans <= most_tries(count)  # the figures in this file's docstring


@pytest.mark.parametrize("count", [1, 2, 4])
def test_the_tries_stay_under_the_limit_on_cards_between(
    count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 2.0 to 100 GiB free in steps of 0.3, both kernels: the file's docstring counts them.
    counter = Counted(monkeypatch)
    seen: dict[int, int] = {}
    for tenths in range(20, 1001, 3):
        for attention in ATTENTIONS:
            plans, _ = counter.tries(asked(tenths / 10, count, attention=attention), FOUR_K, count)
            assert plans <= most_tries(count), (tenths, attention)
            seen[plans] = seen.get(plans, 0) + 1
    assert sum(seen.values()) == 654
    if count == 1:
        assert seen == {1: 449, 2: 64, 3: 55, 4: 80, 5: 6}
    else:
        assert max(seen) == 3


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
    for free in (48, 96):
        assert counter.tries(asked(free, count), FOUR_K, count) == (1, "fits")
    # Below, the plan moves blocks: refused at once, for the whole DiT.
    assert counter.tries(asked(24, count), FOUR_K, count) == (1, "reservation refused")
    assert counter.tries(asked(32, count), FOUR_K, count)[0] <= 3


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

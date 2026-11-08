"""The room a run asks of ComfyUI (models.session): the plan made for the card as it would be with
ComfyUI's own models gone, ComfyUI asked to free the free memory that plan's largest phase needs
and only if the card has less, never more, its models and seedvr2x's registration kept."""

import comfy.model_management as management
import numpy as np
import pytest
import torch
from comfyui import models, nodes, plans, words
from standins import (
    GIB,
    PICTURE,
    RESOLUTION,
    StandIn,
    comfy_model,
    handle,
    upscale_image,
)

from seedvr2x import memory, still_plan
from seedvr2x.memory import Reading

SETTINGS = plans.Settings()


def batch(count: int = 1) -> torch.Tensor:
    return torch.from_numpy(np.random.default_rng(3).random((count, *PICTURE, 3), dtype=np.float32))


def upscale(h: models.Handle) -> str:
    return upscale_image(h, SETTINGS, batch(), RESOLUTION, "9")[1]


def needed(card: int, machine: StandIn, tuning: plans.Settings = SETTINGS) -> tuple[int, Reading]:
    """The free memory the stand-in picture's plan needs in a card of `card` bytes, derived
    from the plan's own phases: the largest phase with the room a VAE pass is held to, and the
    margin (a driver that refuses holds back nothing else). And the reading it was planned in."""
    target = (48, 80)
    reading = Reading(card, None, card, False)
    planned = nodes.picture_planner(tuning, target, PICTURE)(
        reading, machine.host_bytes, machine.dit_format, True
    )
    made = planned.plan[0]
    phases = still_plan.picture_peaks(made, target, PICTURE[0] * PICTURE[1])
    return max(each.allocated + each.room for each in phases.values()) + made.margin, reading


def asks(machine: StandIn) -> list[tuple[float, object, list[object]]]:
    """The calls to free_memory the run made: those that keep seedvr2x's registration loaded (the
    manager's own call when a model is registered keeps nothing)."""
    return [call for call in management.free_calls if call[2]]  # type: ignore[misc]


def test_nothing_is_asked_when_the_card_has_the_room(machine: StandIn, tmp_path) -> None:
    resident = comfy_model(4 * GIB)
    upscale(handle(tmp_path))
    assert asks(machine) == []
    assert resident.loaded_size() == 4 * GIB, "ComfyUI's model stays where it is"


def test_exactly_the_room_is_not_asked_and_a_byte_less_is(machine: StandIn, tmp_path) -> None:
    resident = comfy_model(12 * GIB)
    need, _ = needed(32 * GIB, machine)
    # The card as the plan sees it with the resident gone: 32 GiB. Free as it is: 20 GiB.
    management.card.outside = 32 * GIB - 12 * GIB - need
    assert machine.free() == need
    upscale(handle(tmp_path))
    assert asks(machine) == [] and resident.loaded_size() == 12 * GIB
    management.card.outside += 1
    upscale(handle(tmp_path))
    ((wanted, _device, kept),) = asks(machine)
    assert wanted == need + models.ASK_MARGIN and len(kept) == 1
    assert resident.loaded_size() == 12 * GIB - 1 - models.ASK_MARGIN


def test_comfy_is_asked_for_the_plans_room_alone_and_its_models_are_freed_no_further(
    machine: StandIn, tmp_path
) -> None:
    resident = comfy_model(20 * GIB)
    need, _ = needed(32 * GIB, machine)
    free_before = machine.free()
    assert free_before == 12 * GIB < need
    upscale(handle(tmp_path))
    ((wanted, device, kept),) = asks(machine)
    assert wanted == need + models.ASK_MARGIN and device == machine.device
    # The registration protected, and ComfyUI's model freed by what was short, partly, no more.
    entry = models.kept(handle(tmp_path))
    assert entry is not None and [each.model for each in kept] == [entry.patcher]  # type: ignore[attr-defined]
    assert resident.loaded_size() == 20 * GIB - (need + models.ASK_MARGIN - free_before)
    assert machine.free() == need + models.ASK_MARGIN
    assert [each.model for each in management.current_loaded_models].count(resident) == 1
    assert entry.patcher in [each.model for each in management.current_loaded_models]


def test_the_plan_is_made_for_the_card_without_comfys_models(machine: StandIn, tmp_path) -> None:
    resident = comfy_model(20 * GIB)
    seen: list[Reading] = []
    real = nodes.picture_planner(SETTINGS, (48, 80), PICTURE)

    def planner(reading: Reading, *args: object) -> object:
        seen.append(reading)
        return real(reading, *args)  # type: ignore[arg-type]

    with models.session(handle(tmp_path), planner):  # type: ignore[arg-type]
        pass
    assert resident.loaded_size() < 20 * GIB  # freed by what the plan needed
    # Early (the files not read yet), then for the run: both counted in the whole card.
    assert [each.reported for each in seen] == [32 * GIB, 32 * GIB]
    assert all(each.counted == 32 * GIB for each in seen)


def test_a_card_still_short_after_comfy_freed_what_it_could_is_refused_in_the_nodes_words(
    machine: StandIn, tmp_path
) -> None:
    # A model that says it frees nothing, 12 GiB of the 32: the plan counts on 32 less the 20
    # other programs hold, and what ComfyUI is asked to free doesn't come back.
    stuck = comfy_model(12 * GIB, stuck=True)
    management.card.outside = 12 * GIB
    with pytest.raises(words.Refused) as refused:
        upscale(handle(tmp_path))
    assert "--" not in str(refused.value)
    assert "after ComfyUI was asked to free its models" in str(refused.value)
    assert "other programs' memory is what is left" in str(refused.value)
    assert stuck.loaded_size() == 0
    entry = models.kept(handle(tmp_path))
    assert entry is not None and not entry.busy
    held = machine.made[0]
    assert held.placed == []  # type: ignore[attr-defined]


def test_the_room_is_in_the_plans_text(machine: StandIn, tmp_path) -> None:
    resident = comfy_model(20 * GIB)
    text = upscale(handle(tmp_path))
    assert resident.loaded_size() < 20 * GIB
    assert "ComfyUI's own models use 20.00 GiB" in text
    assert "ComfyUI was asked to unload models until" in text and "no need to" not in text
    management.reset()
    models.reset()
    models.use(machine)
    assert "no need to unload ComfyUI's models" in upscale(handle(tmp_path))


def test_a_run_refused_for_memory_is_refused_before_the_files_are_read(
    machine: StandIn, tmp_path
) -> None:
    management.card.total = round(6.0 * GIB)
    big = torch.zeros((1, 1080, 1920, 3), dtype=torch.float32)
    with pytest.raises(words.Refused) as refused:
        upscale_image(handle(tmp_path), SETTINGS, big, 3240, "9")
    assert machine.loads == [] and machine.made == []
    assert "--" not in str(refused.value)


def test_the_smaller_files_that_plan_the_run_are_named_in_the_nodes_terms(
    machine: StandIn, tmp_path
) -> None:
    from seedvr2x import pull

    management.card.total = round(6.0 * GIB)
    big = torch.zeros((1, 1080, 1920, 3), dtype=torch.float32)
    with pytest.raises(words.Refused) as refused:
        upscale_image(
            handle(tmp_path, dit="seedvr2x_ema_7b_sharp_fp16.safetensors"), SETTINGS, big, 3240, "9"
        )
    text = str(refused.value)
    assert "choose the dynamic GGUF file (4.8 GB) in seedvr2x Models" in text, text
    assert "--" not in text and "command" not in text
    assert pull.PINNED["seedvr2x_ema_7b_sharp_dyn.gguf"].size < 5 * GIB


def test_a_driver_that_lends_system_memory_sets_no_limit_on_torch(
    machine: StandIn, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI holds itself to the card there by torch's memory fraction; ComfyUI's process can't
    take it back (test_process_state.py), so the nodes ask the driver for nothing of the kind."""
    machine.lends = True
    machine.held_card = 32 * GIB
    resident = comfy_model(20 * GIB)
    upscale(handle(tmp_path))
    assert resident.loaded_size() > 0
    assert not hasattr(machine, "hold")


def test_the_card_is_read_again_after_torchs_cache_is_emptied(machine: StandIn, tmp_path) -> None:
    # ComfyUI's partial unload frees into torch's cache, which the driver's free memory leaves
    # out until it is emptied: without that, a run the freed memory would hold is refused.
    resident = comfy_model(20 * GIB, caching=True)
    need, _ = needed(32 * GIB, machine)
    upscale(handle(tmp_path))
    ((wanted, _device, _kept),) = asks(machine)
    assert wanted == need + models.ASK_MARGIN
    assert resident.loaded_size() == 20 * GIB - (need + models.ASK_MARGIN - 12 * GIB)
    assert management.card.cached == 0 and machine.free() == need + models.ASK_MARGIN


def test_comfys_partial_unload_keeping_a_buffer_still_leaves_the_plans_room(
    machine: StandIn, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ComfyUI v0.44.0's partial unload frees to the figure asked less a buffer of 94 MB: asked
    exactly the plan's room the card read short and the run was refused (2026-11-05, RTX 5090);
    asked the margin more it runs, the plan unchanged."""
    buffer = 94 * 1024**2
    need, _ = needed(32 * GIB, machine)
    assert buffer < models.ASK_MARGIN

    management.PARTIAL_BUFFER = buffer
    exact = comfy_model(20 * GIB)
    with monkeypatch.context() as exactly:
        exactly.setattr(models, "ASK_MARGIN", 0)
        with pytest.raises(words.Refused) as refused:
            upscale(handle(tmp_path))
    assert "after ComfyUI was asked to free its models" in str(refused.value)
    assert machine.free() == need - buffer
    del exact

    management.reset()
    models.reset()
    models.use(machine)
    management.PARTIAL_BUFFER = buffer
    resident = comfy_model(20 * GIB)
    text = upscale(handle(tmp_path))
    ((wanted, _device, _kept),) = asks(machine)
    assert wanted == need + models.ASK_MARGIN
    assert machine.free() == need + models.ASK_MARGIN - buffer >= need
    assert resident.loaded_size() > 0, "ComfyUI's model is freed partly, not whole"
    # The plan does not count the margin: its text gives the plan's own need.
    margin = models.ASK_MARGIN
    assert f"asked to unload models until {(need + margin) / GIB:.2f} GiB was free" in text
    assert f"the {need / GIB:.2f} GiB this run needs, with a safety margin" in text
    assert f"and {margin / GIB:.2f} GiB more, as ComfyUI leaves some memory unfreed" in text, text


def test_the_margin_is_asked_only_where_comfy_is_asked(machine: StandIn, tmp_path) -> None:
    resident = comfy_model(12 * GIB)
    need, _ = needed(32 * GIB, machine)
    # Free: the plan's room and not a byte over the margin more: nothing is asked.
    management.card.outside = 32 * GIB - 12 * GIB - need
    assert need <= machine.free() < need + models.ASK_MARGIN
    upscale(handle(tmp_path))
    assert asks(machine) == [] and resident.loaded_size() == 12 * GIB


def test_the_plan_counts_the_card_less_the_room_comfy_keeps_for_the_system(
    machine: StandIn, tmp_path
) -> None:
    # ComfyUI's own loads leave `extra_reserved_memory` free for the desktop; so does the plan.
    management.EXTRA_RESERVED = 2 * GIB
    seen: list[Reading] = []
    real = nodes.picture_planner(SETTINGS, (48, 80), PICTURE)

    def planner(reading: Reading, *args: object) -> object:
        seen.append(reading)
        return real(reading, *args)  # type: ignore[arg-type]

    with models.session(handle(tmp_path), planner):  # type: ignore[arg-type]
        pass
    assert [each.reported for each in seen] == [30 * GIB, 30 * GIB]
    assert all(each.counted == 30 * GIB for each in seen)


def test_the_room_kept_for_the_system_is_asked_of_comfy_and_nothing_twice(
    machine: StandIn, tmp_path
) -> None:
    resident = comfy_model(12 * GIB)
    need, _ = needed(30 * GIB, machine)
    reserve = 2 * GIB
    management.EXTRA_RESERVED = reserve
    # Free beyond the reserve: exactly the plan's need, so nothing is asked; a byte less, and it is.
    management.card.outside = 32 * GIB - 12 * GIB - need - reserve
    assert machine.free() == need + reserve
    upscale(handle(tmp_path))
    assert asks(machine) == [] and resident.loaded_size() == 12 * GIB
    management.card.outside += 1
    upscale(handle(tmp_path))
    ((wanted, _device, _kept),) = asks(machine)
    assert wanted == need + reserve + models.ASK_MARGIN
    assert machine.free() == wanted


def test_a_card_short_of_the_plan_once_the_reserve_is_kept_is_refused_in_the_nodes_words(
    machine: StandIn, tmp_path
) -> None:
    stuck = comfy_model(12 * GIB, stuck=True)
    need, _ = needed(32 * GIB, machine)
    # Room for the plan without the reserve, not with it, and nothing ComfyUI can free.
    management.card.outside = 32 * GIB - 12 * GIB - need
    management.EXTRA_RESERVED = 1 * GIB
    with pytest.raises(words.Refused) as refused:
        upscale(handle(tmp_path))
    assert "--" not in str(refused.value)
    assert stuck.loaded_size() == 0


def test_a_reserve_over_the_free_memory_leaves_nothing_and_never_less(
    machine: StandIn, tmp_path
) -> None:
    management.card.outside = 30 * GIB
    management.EXTRA_RESERVED = 5 * GIB
    seen: list[Reading] = []

    def planner(reading: Reading, *args: object) -> object:
        seen.append(reading)
        raise words.Refused("stop here")

    with pytest.raises(words.Refused):
        with models.session(handle(tmp_path), planner):  # type: ignore[arg-type]
            pass
    assert seen and all(each.reported == 0 and each.counted == 0 for each in seen)


def test_the_card_line_says_the_room_left_to_the_rest_of_the_card_only_where_there_is_one(
    machine: StandIn, tmp_path
) -> None:
    management.EXTRA_RESERVED = 700 * 1024**2
    text = upscale(handle(tmp_path))
    assert (
        "32.00 GiB free; ComfyUI's own models use 0.00 GiB of it; 0.68 GiB is left to the rest of"
        " the card (ComfyUI's reserve), and seedvr2x counts on 31.32 GiB" in text
    )
    management.reset()
    models.reset()
    models.use(machine)
    text = upscale(handle(tmp_path))
    assert "left to the rest of the card" not in text and "reserve" not in text
    assert "ComfyUI's own models use 0.00 GiB of it, and seedvr2x counts on 32.00 GiB" in text


def test_the_card_line_names_the_larger_room_once_on_a_held_run(machine: StandIn, tmp_path) -> None:
    machine.held_card = 32 * GIB
    room = (memory.HELD_BACK / GIB, "seedvr2x's least on this driver")
    for reserve, said in (
        (0, room),
        (memory.HELD_BACK // 2, room),
        (memory.HELD_BACK, (memory.HELD_BACK / GIB, "ComfyUI's reserve")),  # equal: ComfyUI's
        (2 * GIB, (2.0, "ComfyUI's reserve")),
    ):
        management.reset()
        models.reset()
        models.use(machine)
        management.EXTRA_RESERVED = reserve
        text = upscale(handle(tmp_path))
        figure, source = said
        assert f"{figure:.2f} GiB is left to the rest of the card ({source})" in text, reserve
        assert text.count("left to the rest of the card") == 1
        assert "for the system" not in text


def held_need(machine: StandIn, tmp_path) -> int:
    """The free memory the stand-in picture's plan needs on a held card (needed_free), read from
    the plan the nodes made on a card with room to spare."""
    seen: list[models.Needs] = []
    real = nodes.picture_planner(SETTINGS, (48, 80), PICTURE)

    def planner(reading: Reading, *args: object) -> object:
        planned = real(reading, *args)  # type: ignore[arg-type]
        seen.append(planned.needs)
        return planned

    with models.session(handle(tmp_path), planner):  # type: ignore[arg-type]
        pass
    return models.needed_free(seen[-1])


def test_the_room_a_held_run_asks_is_the_larger_of_comfys_reserve_and_seedvr2xs_least(
    machine: StandIn, tmp_path
) -> None:
    """One need, not their sum (DESIGN.md, Allocator, Windows itself): a card with exactly the
    plan's need and the room free is not asked, a byte less is, and ComfyUI is asked for the need
    and the room once."""
    machine.held_card = 32 * GIB
    need = held_need(machine, tmp_path)
    assert need >= memory.HELD_BACK  # the need holds the least back already (needed_free)
    for reserve in (0, memory.HELD_BACK // 2, memory.HELD_BACK, 2 * GIB):
        management.reset()
        models.reset()
        models.use(machine)
        management.EXTRA_RESERVED = reserve
        resident = comfy_model(12 * GIB)
        beyond = max(reserve - memory.HELD_BACK, 0)
        # Free: the need and the room beyond what the need holds back, exactly: nothing is asked.
        management.card.outside = 32 * GIB - 12 * GIB - need - beyond
        assert machine.free() == need + beyond
        upscale(handle(tmp_path))
        assert asks(machine) == [] and resident.loaded_size() == 12 * GIB, reserve
        management.card.outside += 1
        upscale(handle(tmp_path))
        ((wanted, _device, _kept),) = asks(machine)
        assert wanted == need + beyond + models.ASK_MARGIN, reserve


def test_the_plan_of_a_held_run_counts_the_card_less_the_larger_room(
    machine: StandIn, tmp_path
) -> None:
    machine.held_card = 32 * GIB
    for reserve, room in ((GIB // 4, memory.HELD_BACK), (3 * GIB, 3 * GIB)):
        management.reset()
        models.reset()
        models.use(machine)
        management.EXTRA_RESERVED = reserve
        resident = comfy_model(5 * GIB)
        seen: list[Reading] = []

        def planner(reading: Reading, *args: object, into: list[Reading] = seen) -> object:
            into.append(reading)
            raise words.Refused("stop here")

        with pytest.raises(words.Refused):
            with models.session(handle(tmp_path), planner):  # type: ignore[arg-type]
                pass
        # The card is 32 GiB with ComfyUI's 5 GiB on it: free 27, 32 counted without them, and the
        # room off once.
        assert {each.counted for each in seen} == {32 * GIB - room}, reserve
        del resident

"""The run's reading of the card and the machine behind it (models.py): ComfyUI's models counted
free for the plan, the room a plan needs, the machine on a CPU or a card, torch's allocator
setting as startup is given it, the allocator said once."""

import logging
import os
from dataclasses import replace

import comfy.model_management as management
import pytest
import torch
from comfyui import models, plans, words
from standins import (
    GIB,
    PICTURE,
    RESOLUTION,
    StandIn,
    comfy_model,
    handle,
    upscale_image,
    upscale_video,
)

from seedvr2x import memory
from seedvr2x.memory import Reading


def test_comfys_models_are_counted_free_for_a_driver_that_refuses() -> None:
    reading = Reading(10 * GIB, None, 10 * GIB, False)
    shifted = models.with_comfy_gone(reading, 5 * GIB)
    assert shifted == Reading(15 * GIB, None, 15 * GIB, False)
    assert models.with_comfy_gone(reading, 0) is reading


def test_the_room_kept_for_the_system_is_counted_gone_and_never_below_nothing() -> None:
    reading = Reading(10 * GIB, None, 10 * GIB, False)
    assert models.with_comfy_gone(reading, 5 * GIB, 2 * GIB) == Reading(
        13 * GIB, None, 13 * GIB, False
    )
    assert models.with_comfy_gone(reading, 0, 2 * GIB) == Reading(8 * GIB, None, 8 * GIB, False)
    assert models.with_comfy_gone(reading, 0, 12 * GIB) == Reading(0, None, 0, False)


def test_the_room_is_the_larger_of_comfys_reserve_and_the_least_a_held_reading_leaves() -> None:
    # One need, not the sum: memory.counted takes the reading's `left` off, so the shift only adds
    # what the reserve asks beyond it (DESIGN.md, Allocator, Windows itself).
    taken = 12 * GIB
    for left, reserve, room in (
        (memory.HELD_BACK, 0, memory.HELD_BACK),
        (memory.HELD_BACK, memory.HELD_BACK // 2, memory.HELD_BACK),
        (memory.HELD_BACK, memory.HELD_BACK, memory.HELD_BACK),
        (memory.HELD_BACK, 2 * GIB, 2 * GIB),
        # The card's figure unread: the larger least.
        (memory.HELD_BACK_UNREAD, 2 * GIB, memory.HELD_BACK_UNREAD),
        (memory.HELD_BACK_UNREAD, 4 * GIB, 4 * GIB),
    ):
        shell = Reading(taken, None, 0, True, left, memory.SEGMENTS)
        reading = replace(shell, counted=memory.counted(taken, shell))
        shifted = models.with_comfy_gone(reading, 5 * GIB, reserve)
        assert shifted.counted == 17 * GIB - room, (left, reserve)
        assert models.room_beyond(reading, reserve) == room - left
    # Torch's own allocator without the segments keeps its flat reserve on top of the room.
    shell = Reading(taken, taken, 0, True, memory.HELD_BACK, memory.PLAIN)
    reading = replace(shell, counted=memory.counted(taken, shell))
    shifted = models.with_comfy_gone(reading, 0, 2 * GIB)
    assert shifted.counted == taken - 2 * GIB - memory.RESERVE
    # A driver that refuses leaves what ComfyUI keeps alone.
    plain = Reading(taken, None, taken, False)
    assert models.with_comfy_gone(plain, 0, 2 * GIB).counted == taken - 2 * GIB


def test_comfys_models_are_counted_free_for_a_held_run_as_the_library_counts_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A driver that lends system memory: the library's reading of `free` bytes with the whole
    # card at `card`, and the same reading made on a card that ComfyUI's models had left.
    monkeypatch.setattr(memory, "_spills", True)
    monkeypatch.setattr(memory, "_native", True)
    monkeypatch.setattr(memory, "_segments", True)
    for free, card, held in (
        (20 * GIB, 18 * GIB, 6 * GIB),
        (18 * GIB, 20 * GIB, 6 * GIB),
        (30 * GIB, None, 3 * GIB),
    ):
        # NVML's reading of the whole card: its figure, or a failure (a string) where it is None.
        monkeypatch.setattr(
            memory, "_twice", lambda index, card=card: "NVML failed" if card is None else card
        )
        now = memory.reading(free)
        monkeypatch.setattr(
            memory,
            "_twice",
            lambda index, card=card, held=held: "NVML failed" if card is None else card + held,
        )
        gone = memory.reading(free + held)
        assert models.with_comfy_gone(now, held) == gone, (free, card)
        assert now.held and gone.counted > now.counted


def test_the_free_memory_a_plan_needs_is_its_phase_its_margin_and_what_the_reading_holds_back() -> (
    None
):
    plain = Reading(10 * GIB, None, 10 * GIB, False)
    needs = models.Needs(peak=6 * GIB, margin=GIB // 2, reading=plain)
    assert models.needed_free(needs) == 6 * GIB + GIB // 2
    # What a held run holds back depends on the allocator: under torch's own allocator without
    # the segments the plan counts RESERVE off the card besides torch's limit (memory.counted),
    # which the room asks for; under the segments and cudaMallocAsync only the limit's share,
    # the step's own reserve being in the peak the plan's needs carry (Needs.peak).
    for allocator, reserve in (
        (memory.PLAIN, memory.RESERVE),
        (memory.SEGMENTS, 0),
        (memory.POOL, 0),
    ):
        shell = Reading(10 * GIB, 9 * GIB, 0, True, memory.HELD_BACK, allocator)
        held = Reading(
            10 * GIB,
            9 * GIB,
            memory.counted(9 * GIB, shell),
            True,
            memory.HELD_BACK,
            allocator,
        )
        assert held.counted == 9 * GIB - memory.HELD_BACK - reserve
        assert models.needed_free(models.Needs(6 * GIB, 0, held)) == (
            6 * GIB + memory.HELD_BACK + reserve
        ), allocator


def test_comfys_models_on_the_card_are_the_loaded_sizes_of_its_own_entries_on_that_device(
    machine: StandIn,
) -> None:
    a, b = comfy_model(3 * GIB), comfy_model(5 * GIB, loaded=2 * GIB)
    ours = comfy_model(0)
    other = management.get_torch_device()
    assert models.comfy_holds(other, None) == 5 * GIB
    assert models.comfy_holds(other, ours) == 5 * GIB  # type: ignore[arg-type]
    assert models.comfy_holds(torch.device("cuda", 1), None) == 0
    del a, b


def run_image(h: models.Handle) -> None:
    upscale_image(h, plans.Settings(), torch.zeros(1, *PICTURE, 3), RESOLUTION, "1")


def test_the_allocator_is_said_once_where_the_driver_lends_system_memory(
    machine: StandIn, tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    machine.lends = True
    machine.held_card = 32 * GIB
    with caplog.at_level(logging.WARNING, logger="seedvr2x_comfyui.models"):
        run_image(handle(tmp_path))
        run_image(handle(tmp_path))
    said = [line for line in caplog.messages if "--disable-cuda-malloc" in line]
    assert len(said) == 1 and "cudaMallocAsync" in said[0]


def test_torchs_own_allocator_without_expandable_segments_is_told_the_nodes_way_once(
    machine: StandIn, tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    """Under --disable-cuda-malloc with nothing in PYTORCH_CUDA_ALLOC_CONF (ComfyUI v0.44.0, the
    RTX 5090 under WSL2): the nodes' words, not the command line's "add ... or unset it"."""
    machine.lends = True
    machine.held_card = 32 * GIB
    machine.allocator_name = "native"
    machine.expandable = False
    with caplog.at_level(logging.WARNING, logger="seedvr2x_comfyui.models"):
        run_image(handle(tmp_path))
        run_image(handle(tmp_path))
    said = [line for line in caplog.messages if "PYTORCH_CUDA_ALLOC_CONF" in line]
    assert len(said) == 1
    assert "set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True before starting ComfyUI" in said[0]
    assert "slows down instead" in said[0] and "unset" not in said[0]
    assert not [line for line in caplog.messages if "disable-cuda-malloc" in line]


@pytest.mark.parametrize("allocator", ["cudaMallocAsync", "native"])
def test_on_windows_the_allocator_is_said_with_no_way_to_expandable_segments(
    machine: StandIn,
    tmp_path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    allocator: str,
) -> None:
    # Torch's Windows build has no expandable segments (memory._allocator): cudaMallocAsync,
    # ComfyUI's default, is said at INFO, the plan leaving its pool room; torch's own allocator
    # (--disable-cuda-malloc) is warned of, with ComfyUI's default as the way, once.
    monkeypatch.setattr(memory.sys, "platform", "win32")
    machine.lends = True
    machine.held_card = 32 * GIB
    machine.allocator_name = allocator
    machine.expandable = False
    with caplog.at_level(logging.INFO, logger="seedvr2x_comfyui.models"):
        run_image(handle(tmp_path))
        run_image(handle(tmp_path))
    said = [r for r in caplog.records if "ComfyUI runs torch" in r.getMessage()]
    assert len(said) == 1
    assert not [line for line in caplog.messages if "expandable_segments:True" in line]
    if allocator == "cudaMallocAsync":
        assert said[0].levelno == logging.INFO and "leaves room for it" in said[0].getMessage()
    else:
        assert said[0].levelno == logging.WARNING
        assert "no expandable segments on Windows" in said[0].getMessage()
        assert "without --disable-cuda-malloc" in said[0].getMessage()


def test_the_lending_driver_is_told_in_the_nodes_words_once(
    machine: StandIn, tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    machine.lends = True
    machine.held_card = 32 * GIB
    with caplog.at_level(logging.INFO, logger="seedvr2x_comfyui.models"):
        run_image(handle(tmp_path))
        run_image(handle(tmp_path))
    said = [line for line in caplog.messages if "borrows your computer's RAM" in line]
    assert len(said) == 1 and "(Windows' shared GPU memory), which is slow" in said[0]
    assert "seedvr2x's plan leaves room on the card" in said[0]
    assert "reserve" not in said[0] and "torch" not in said[0]


def test_the_librarys_command_line_warnings_are_not_shown_in_comfyuis_process(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def startup() -> str:
        memory.logger.warning(
            "the allocator is the environment's, %s: torch's own without expandable segments;"
            " add expandable_segments:True to it, or unset it",
            "backend:native",
        )
        memory.logger.warning(
            "the allocator is the environment's, %s: unset it for torch's own allocator",
            "backend:cudaMallocAsync",
        )
        memory.logger.warning(
            "seedvr2x holds itself to the card's own memory there: torch limited to the memory"
            " free less %.1f GiB",
            1.0,
        )
        # The driver that lends, as memory.startup logs it (through "%s"): the nodes' own INFO
        # line says it (test_the_lending_driver_is_told_in_the_nodes_words_once).
        memory.logger.warning("%s", memory.LENDS)
        # WSL2's driver that didn't answer, taken for one that lends: the same finding, said by
        # the nodes' own line too.
        memory.logger.warning("%s", memory.UNANSWERED)
        memory.logger.warning("another warning of the library's")
        return ""

    monkeypatch.setattr(memory, "startup", startup)
    monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: "native")
    with caplog.at_level(logging.WARNING, logger=memory.logger.name):
        models._startup()  # pyright: ignore[reportPrivateUsage]
        assert caplog.messages == ["[seedvr2x] another warning of the library's"]
        # The filter is taken off with the call: the command line's own runs say it.
        memory.logger.warning("the allocator is the environment's, %s", "x")
    assert caplog.messages[-1] == "[seedvr2x] the allocator is the environment's, x"


@pytest.mark.parametrize(
    ("lends", "allocator"), [(False, "cudaMallocAsync"), (True, "native"), (False, "native")]
)
def test_the_allocator_is_not_said_otherwise(
    machine: StandIn, tmp_path, caplog: pytest.LogCaptureFixture, lends: bool, allocator: str
) -> None:
    machine.lends = lends
    machine.allocator_name = allocator
    if lends:
        machine.held_card = 32 * GIB
    with caplog.at_level(logging.WARNING, logger="seedvr2x_comfyui.models"):
        run_image(handle(tmp_path))
    assert not [line for line in caplog.messages if "disable-cuda-malloc" in line]


def test_startup_is_given_the_allocator_torch_runs_and_the_environment_is_left_as_it_was(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, str | None]] = []

    def startup() -> str:
        seen.append({name: os.environ.get(name) for name in memory.VARIABLES})
        return ""

    monkeypatch.setattr(memory, "startup", startup)
    first, second = memory.VARIABLES
    # No setting: torch runs what ComfyUI chose, which startup reads from the environment.
    for backend in ("native", "cudaMallocAsync"):
        monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda backend=backend: backend)
        monkeypatch.delenv(first, raising=False)
        monkeypatch.delenv(second, raising=False)
        models._startup()  # pyright: ignore[reportPrivateUsage]
        assert seen[-1] == {first: f"backend:{backend}", second: None}
        assert first not in os.environ and second not in os.environ
    # A setting (ComfyUI's own appends to it, or the user's): read as it stands.
    monkeypatch.setenv(first, "backend:cudaMallocAsync")
    models._startup()  # pyright: ignore[reportPrivateUsage]
    assert seen[-1][first] == "backend:cudaMallocAsync"
    assert os.environ[first] == "backend:cudaMallocAsync"
    # An empty one is no setting, as startup takes it, and is put back as it was.
    monkeypatch.setenv(first, "")
    monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: "native")
    models._startup()  # pyright: ignore[reportPrivateUsage]
    assert seen[-1][first] == "backend:native" and os.environ[first] == ""


def test_the_cuda_machine_refuses_a_comfyui_that_runs_on_the_cpu_or_without_a_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(management, "device", torch.device("cpu"))
    with pytest.raises(words.Refused, match="ComfyUI is running on the CPU") as refused:
        models.CudaMachine()
    assert "--" not in str(refused.value)
    monkeypatch.setattr(management, "device", torch.device("cuda", 0))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(words.Refused, match="torch finds no CUDA device"):
        models.CudaMachine()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    with pytest.raises(words.Refused, match="bfloat16"):
        models.CudaMachine()


# ---- The reserve by step, said in the nodes' words; the probe's margin carried to the plan ------


class Seen(Exception):
    """Raised by a planner that has seen what it was given."""


def spare_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("spare memory:")]


@pytest.mark.parametrize(
    ("allocator_name", "expandable", "way", "figures"),
    [
        # torch's own allocator with its segments, ComfyUI's --disable-cuda-malloc: the floors
        # of a small picture's steps (peaks.DIT_RESERVE_FLOOR, VAE_RESERVE_FLOOR).
        (
            "native",
            True,
            "PyTorch's default memory handling",
            "0.60 GiB while the model works, 1.25 GiB while an image is encoded, 1.25 GiB",
        ),
        # cudaMallocAsync, ComfyUI's default: the pool's floors (peaks.POOL_DIT_FLOOR).
        (
            "cudaMallocAsync",
            True,
            "ComfyUI's cudaMallocAsync setting",
            "0.85 GiB while the model works, 1.25 GiB while an image is encoded, 1.25 GiB",
        ),
    ],
)
def test_a_held_plan_says_the_spare_memory_of_each_step_in_the_nodes_words(
    machine: StandIn, tmp_path, allocator_name: str, expandable: bool, way: str, figures: str
) -> None:
    machine.lends = True
    machine.held_card = 32 * GIB
    machine.allocator_name = allocator_name
    machine.expandable = expandable
    for clip in (False, True):
        if clip:
            text = upscale_video(
                handle(tmp_path),
                plans.Settings(),
                torch.zeros(9, *PICTURE, 3),
                RESOLUTION,
                "1",
                plan_only=True,
            )[1]
        else:
            text = upscale_image(
                handle(tmp_path),
                plans.Settings(),
                torch.zeros(1, *PICTURE, 3),
                RESOLUTION,
                "1",
                plan_only=True,
            )[1]
        found = spare_lines(text)
        assert len(found) == 1, text
        assert way in found[0] and figures in found[0] and "while it is decoded" in found[0]
        # ComfyUI users' words: no library term, no option.
        assert "--" not in found[0] and "reserve" not in found[0].replace("spare", "")
        assert "peak" in found[0]


def test_a_plan_without_segments_says_the_flat_spare_memory_and_a_refusing_driver_none(
    machine: StandIn, tmp_path
) -> None:
    machine.lends = True
    machine.held_card = 32 * GIB
    machine.allocator_name = "native"
    machine.expandable = False
    text = upscale_image(
        handle(tmp_path),
        plans.Settings(),
        torch.zeros(1, *PICTURE, 3),
        RESOLUTION,
        "1",
        plan_only=True,
    )[1]
    assert spare_lines(text) == [
        "spare memory: 4.50 GiB of the card is kept free beyond what any step needs, for PyTorch's"
        " own use"
    ]
    machine.held_card = None  # a driver that refuses: no reserve, no line
    text = upscale_image(
        handle(tmp_path),
        plans.Settings(),
        torch.zeros(1, *PICTURE, 3),
        RESOLUTION,
        "1",
        plan_only=True,
    )[1]
    assert spare_lines(text) == []


def test_comfys_models_gone_count_the_card_by_the_allocators_rule() -> None:
    # with_comfy_gone counts the figure again through memory.counted: the flat reserve off under
    # torch's own allocator without the segments alone, nothing more elsewhere.
    for allocator, off in (
        (memory.PLAIN, memory.RESERVE),
        (memory.SEGMENTS, 0),
        (memory.POOL, 0),
    ):
        taken = 12 * GIB
        shell = Reading(taken, taken, 0, True, memory.HELD_BACK, allocator)
        reading = replace(shell, counted=memory.counted(taken, shell), margin=0.4)
        shifted = models.with_comfy_gone(reading, 5 * GIB)
        assert shifted.counted == 17 * GIB - memory.HELD_BACK - off, allocator
        assert shifted.allocator == allocator and shifted.margin == 0.4  # carried through


def test_the_probes_allocator_margin_reaches_the_plan_and_none_is_none(
    machine: StandIn, tmp_path
) -> None:
    # The cache's margin (probe.Probed.allocator_margin, a share of a step's P) rides on the
    # plan's reading (Reading.margin), where peaks.reserve counts the larger of it and the rule.
    # Nothing builds the measurement, so a margin is only what a cache holds: here the stand-in's.
    machine.lends = True
    machine.held_card = 32 * GIB
    seen: list[float | None] = []

    def planner(reading, host, architecture, held):  # type: ignore[no-untyped-def]
        seen.append(reading.margin)
        raise Seen

    for margin in (None, 0.5, None):
        models.reset()
        machine.allocator_margin = margin
        with pytest.raises(Seen), models.session(handle(tmp_path), planner):
            pass
        assert seen[-1] == margin
    # The probe runs once a layout and kernel (reset forgets it); a plan only reads no probe, and
    # counts the margin an earlier run of the process found.
    models.reset()
    machine.allocator_margin = 0.5
    with pytest.raises(Seen), models.session(handle(tmp_path), planner, plan_only=True):
        pass
    assert seen[-1] is None
    with pytest.raises(Seen), models.session(handle(tmp_path), planner):
        pass
    with pytest.raises(Seen), models.session(handle(tmp_path), planner, plan_only=True):
        pass
    assert seen[-2:] == [0.5, 0.5]

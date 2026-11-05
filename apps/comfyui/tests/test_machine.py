"""The run's reading of the card and the machine behind it (models.py): ComfyUI's models counted
free for the plan, the room a plan needs, the machine on a CPU or a card, torch's allocator
setting as startup is given it, the allocator said once."""

import logging
import os

import comfy.model_management as management
import pytest
import torch
from comfyui import models, plans, words
from standins import GIB, PICTURE, RESOLUTION, StandIn, comfy_model, handle, upscale_image

from seedvr2x import memory
from seedvr2x.memory import Reading


def test_comfys_models_are_counted_free_for_a_driver_that_refuses() -> None:
    reading = Reading(10 * GIB, None, 10 * GIB, False)
    shifted = models.with_comfy_gone(reading, 5 * GIB)
    assert shifted == Reading(15 * GIB, None, 15 * GIB, False)
    assert models.with_comfy_gone(reading, 0) is reading


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
    held = Reading(
        10 * GIB,
        9 * GIB,
        9 * GIB - memory.HELD_BACK - memory.RESERVE,
        True,
        memory.HELD_BACK,
    )
    assert models.needed_free(models.Needs(6 * GIB, 0, held)) == (
        6 * GIB + memory.HELD_BACK + memory.RESERVE
    )
    tight = memory.tightened(held)
    assert models.needed_free(models.Needs(6 * GIB, 0, tight)) == (
        6 * GIB + memory.HELD_BACK + memory.RESERVE_TIGHT
    )


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


def test_the_lending_driver_is_told_in_the_nodes_words_once(
    machine: StandIn, tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    machine.lends = True
    machine.held_card = 32 * GIB
    with caplog.at_level(logging.INFO, logger="seedvr2x_comfyui.models"):
        run_image(handle(tmp_path))
        run_image(handle(tmp_path))
    said = [line for line in caplog.messages if "the plan keeps the reserve" in line]
    assert len(said) == 1 and "no limit is set on torch's memory" in said[0]
    assert "the plan keeps the reserve" in said[0]


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
        memory.logger.warning("another warning of the library's")
        return ""

    monkeypatch.setattr(memory, "startup", startup)
    monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: "native")
    with caplog.at_level(logging.WARNING, logger=memory.logger.name):
        models._startup()  # pyright: ignore[reportPrivateUsage]
        assert caplog.messages == ["another warning of the library's"]
        # The filter is taken off with the call: the command line's own runs say it.
        memory.logger.warning("the allocator is the environment's, %s", "x")
    assert caplog.messages[-1] == "the allocator is the environment's, x"


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

"""What a run leaves of ComfyUI's process as it found it: torch's memory limit never set (it
can't be undone there), and the global random generators the library seeds put back."""

import random
from typing import Any

import comfy.model_management as management
import numpy as np
import pytest
import torch
from comfy.utils import ProgressBar
from comfyui import models, plans
from standins import GIB, PICTURE, RESOLUTION, StandIn, handle, upscale_image, upscale_video

from seedvr2x import memory

SETTINGS = plans.Settings(seed=3)


def pictures(count: int = 1) -> torch.Tensor:
    return torch.from_numpy(np.random.default_rng(5).random((count, *PICTURE, 3), dtype=np.float32))


def test_torchs_memory_limit_is_never_set_not_even_where_the_driver_lends_system_memory(
    machine: StandIn, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """1.0 is a cap at the card's total, not no limit, so a limit set for a run could not be put
    back in ComfyUI's process; and under cudaMallocAsync it would not hold (models.CudaMachine)."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("torch's memory limit was touched")

    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", refuse)
    monkeypatch.setattr(torch.cuda.memory, "set_per_process_memory_fraction", refuse)
    monkeypatch.setattr(memory, "hold", refuse)
    machine.lends = True
    machine.held_card = 32 * GIB
    assert not hasattr(models.CudaMachine, "hold") and not hasattr(models.Machine, "hold")
    upscale_image(handle(tmp_path), SETTINGS, pictures(2), RESOLUTION)
    upscale_video(handle(tmp_path, keep=False), SETTINGS, pictures(5), RESOLUTION)


type States = tuple[tuple[Any, ...], tuple[Any, ...], torch.Tensor]


def states() -> States:
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())


def same(before: States, after: States) -> bool:
    (py, (kind, keys, *rest), cpu), (py_b, (kind_b, keys_b, *rest_b), cpu_b) = after, before
    return (
        py == py_b
        and (kind, rest) == (kind_b, rest_b)
        and bool((keys == keys_b).all())
        and bool((cpu == cpu_b).all())
    )


def seeded() -> States:
    """The generators in a state ComfyUI's other nodes could be relying on: seeded, and drawn from
    a little."""
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    _ = (random.random(), np.random.random(), torch.rand(2))
    return states()


@pytest.mark.parametrize("how", ["image", "clip", "kept", "cancel"])
def test_the_global_random_generators_are_as_they_were_after_a_run(
    machine: StandIn, tmp_path, how: str
) -> None:
    before = seeded()
    h = handle(tmp_path, keep=how != "clip")
    if how == "cancel":
        # Cancel after the first step: the encode has seeded the generators by then.
        ProgressBar.on_update = lambda value: management.interrupt_current_processing()
        with pytest.raises(management.InterruptProcessingException):
            upscale_image(h, SETTINGS, pictures(), RESOLUTION)
    elif how == "clip":
        upscale_video(h, SETTINGS, pictures(5), RESOLUTION)
    else:
        upscale_image(h, SETTINGS, pictures(2), RESOLUTION)
    assert same(before, states())


def test_the_run_does_seed_the_generators_so_that_putting_them_back_is_what_is_tested(
    machine: StandIn, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = seeded()
    # With the put-back off, the same run moves them: the previous test would catch the leak.
    monkeypatch.setattr(models, "rng_states", lambda: lambda: None)
    upscale_image(handle(tmp_path), SETTINGS, pictures(), RESOLUTION)
    assert not same(before, states())


def test_the_cuda_generators_are_put_back_where_cuda_is_initialised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    held = [torch.tensor([1, 2, 3], dtype=torch.uint8)]
    put: list[list[torch.Tensor]] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: held)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: put.append(states))
    restore = models.rng_states()
    random.seed(9)
    restore()
    assert put == [held]
    # Not initialised: torch's CUDA generators are not touched (reading them would initialise it).
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    put.clear()
    models.rng_states()()
    assert put == []

"""The nodes' one registration with ComfyUI's model manager (models.Registration): what it tells
the manager, what ComfyUI's own ways of freeing do to it, and how a run finds the models after."""

import gc
import logging
import weakref

import comfy.model_management as management
import numpy as np
import pytest
import torch
from comfyui import models, plans
from standins import PICTURE, RESOLUTION, StandIn, handle, upscale_image

from seedvr2x.passes import HELD


def run(h: models.Handle, seed: int = 0, count: int = 1) -> tuple[torch.Tensor, str]:
    batch = torch.from_numpy(
        np.random.default_rng(seed).random((count, *PICTURE, 3), dtype=np.float32)
    )
    return upscale_image(h, plans.Settings(), batch, RESOLUTION, "5")


def test_comfyuis_console_names_the_registration_seedvr2x(machine: StandIn, tmp_path) -> None:
    # ComfyUI names a model by its class's name (v0.44.0, model_management.py's load_models_gpu:
    # "Requested to load %s" of the patcher's model): seedvr2x, not the stub's own name.
    h = handle(tmp_path)
    run(h)
    entry = models.kept(h)
    assert entry is not None and entry.patcher is not None
    assert type(entry.patcher.model).__name__ == "seedvr2x"
    assert type(entry.patcher.model).__qualname__ == "seedvr2x"


def test_a_kept_run_registers_one_patcher_that_holds_nothing_of_the_card(
    machine: StandIn, tmp_path
) -> None:
    h = handle(tmp_path)
    run(h)
    run(h, 1)
    # Read once, registered once, placed and released at every run.
    assert machine.loads == [HELD]
    assert len(management.loads) == 1 and len(management.current_loaded_models) == 1
    entry = models.kept(h)
    assert entry is not None and entry.patcher is not None
    patcher = entry.patcher
    assert management.current_loaded_models[0].model is patcher
    assert (patcher.model_size(), patcher.loaded_size(), patcher.is_dynamic()) == (0, 0, False)
    assert patcher.load_device == management.get_torch_device() == machine.device
    assert patcher.offload_device == torch.device("cpu") and patcher.model.device == torch.device(
        "cpu"
    )
    # It never tells ComfyUI it loads or frees VRAM; the host memory it holds, it says.
    assert patcher.partially_load(machine.device, 1e12) == 0
    assert patcher.partially_unload(torch.device("cpu"), 1e12) == 0
    assert patcher.loaded_ram_size() == entry.host == 1024
    held = machine.made[0]
    assert len(held.placed) == 2 and held.released == 3 and not held.on_card  # type: ignore[attr-defined]
    # The registration costs nothing on the card.
    assert management.card.running == 0


def test_the_registry_not_the_handle_keeps_the_weights(machine: StandIn, tmp_path) -> None:
    h = handle(tmp_path)
    run(h)
    alive = weakref.ref(machine.made[0])
    # ComfyUI's caches drop a node's output, the handle with it: the weights stay, and so does the
    # registration, whatever ComfyUI's garbage collection (cleanup of models whose stub died).
    del h
    gc.collect()
    management.cleanup_models()
    assert alive() is not None and len(management.current_loaded_models) == 1
    assert management.current_loaded_models[0].model is not None
    run(handle(tmp_path))
    assert len(machine.loads) == 1


def test_unload_models_drops_the_weights_and_the_next_run_loads_again(
    machine: StandIn, tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    h = handle(tmp_path)
    run(h)
    first = machine.made[0]
    alive = weakref.ref(first)
    del first
    # Unload models and POST /free: every model of the card asked for, the registration detached.
    management.unload_all_models()
    machine.made.clear()  # the test's own record of them
    gc.collect()
    assert management.current_loaded_models == [] and models.kept(h) is None
    assert alive() is None, "the pinned weights are freed"
    with caplog.at_level(logging.INFO, logger="seedvr2x_comfyui.models"):
        run(h, 1)
    assert machine.loads == [HELD, HELD]
    assert any("ComfyUI having dropped them" in line for line in caplog.messages)
    assert len(management.current_loaded_models) == 1
    models_now = models.kept(h)
    assert models_now is not None and models_now.patcher is not None
    assert management.current_loaded_models[0].model is models_now.patcher


def test_a_registration_gone_from_the_manager_alone_is_registered_again_without_a_read(
    machine: StandIn, tmp_path
) -> None:
    h = handle(tmp_path)
    run(h)
    management.current_loaded_models.clear()
    run(h)
    assert len(machine.loads) == 1 and len(management.loads) == 2
    assert len(management.current_loaded_models) == 1


def test_choosing_another_model_drops_the_first_so_two_are_never_pinned(
    machine: StandIn, tmp_path
) -> None:
    first = handle(tmp_path, dit="a.safetensors")
    other = handle(tmp_path, dit="b.safetensors")
    run(first)
    run(other)
    assert models.kept(first) is None
    kept_other = models.kept(other)
    assert kept_other is not None
    # The dropped entry's patcher and stub die with it, and ComfyUI's manager forgets the listing.
    gc.collect()
    management.cleanup_models()
    assert [each.model for each in management.current_loaded_models] == [kept_other.patcher]


def test_models_not_kept_are_read_for_each_run_and_dropped_after_it(
    machine: StandIn, tmp_path
) -> None:
    h = handle(tmp_path, keep=False)
    run(h)
    alive = weakref.ref(machine.made[0])
    run(h, 1)
    assert len(machine.loads) == 2 and HELD not in machine.loads
    assert management.current_loaded_models == [] and management.loads == []
    assert models.kept(h) is None
    machine.made.clear()
    gc.collect()
    assert alive() is None


def test_smart_memory_off_reads_the_models_for_each_run_and_says_so_once(
    machine: StandIn, tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    management.DISABLE_SMART_MEMORY = True
    h = handle(tmp_path)
    with caplog.at_level(logging.WARNING, logger="seedvr2x_comfyui.models"):
        run(h)
        run(h, 1)
    assert len(machine.loads) == 2 and management.loads == []
    assert sum("disable-smart-memory" in line for line in caplog.messages) == 1

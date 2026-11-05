"""Keep in RAM between runs: auto, on, off (models.decide_keeping) against ComfyUI's own cap on
pinned memory, what is said of it on the node and in the console, the files switched (the first
file's pinned RAM freed before the second is pinned), ComfyUI's Unload models under a cached
Models output (the weights read again, said once), and the Models node that reads nothing."""

import gc
import logging
import weakref

import comfy.model_management as management
import numpy as np
import pytest
import server
import torch
from comfyui import models, plans
from standins import (
    GIB,
    PICTURE,
    RESOLUTION,
    StandIn,
    execute,
    handle,
    upscale_image,
)

from seedvr2x import pull
from seedvr2x.passes import HELD

INT8 = "seedvr2x_ema_7b_sharp_int8_convrot.safetensors"
FP16 = "seedvr2x_ema_7b_sharp_fp16.safetensors"
SETTINGS = plans.Settings()


def pictures(count: int = 1) -> torch.Tensor:
    return torch.from_numpy(np.random.default_rng(8).random((count, *PICTURE, 3), dtype=np.float32))


def size_of(dit: str) -> int:
    return pull.PINNED[dit].size + pull.PINNED[plans.VAE_FILE].size


def test_the_kept_set_is_the_dits_file_and_the_vaes_by_their_pins_sizes(tmp_path) -> None:
    assert models.pinned_size(handle(tmp_path, dit=INT8)) == size_of(INT8)
    assert models.pinned_size(handle(tmp_path, dit=FP16)) == size_of(FP16)
    # A file seedvr2x does not pin: its size on disk in the model directory, 0 where it is absent.
    assert (
        models.pinned_size(handle(tmp_path, dit="mine.safetensors"))
        == pull.PINNED[plans.VAE_FILE].size
    )
    (tmp_path / "mine.safetensors").write_bytes(b"x" * 1000)
    assert models.pinned_size(handle(tmp_path, dit="mine.safetensors")) == (
        pull.PINNED[plans.VAE_FILE].size + 1000
    )


@pytest.mark.parametrize(
    ("mode", "dit", "cap", "kept", "starts"),
    [
        ("auto", INT8, 12.8 * GIB, True, "kept, 8.2"),
        ("auto", FP16, 12.8 * GIB, False, "not kept: 15."),
        ("on", FP16, 12.8 * GIB, True, "kept, 15."),
        ("off", INT8, 12.8 * GIB, False, "not kept: the files are read again"),
        ("auto", INT8, -1, False, "not kept: ComfyUI allows no RAM for this on this computer"),
        ("on", INT8, -1, True, "kept, 8."),
        ("auto", INT8, size_of(INT8), True, "kept, 8.2"),
        ("auto", INT8, size_of(INT8) - 1, False, "not kept: 8."),
    ],
)
def test_auto_keeps_the_weights_where_the_kept_set_fits_under_comfys_cap(
    tmp_path, mode: str, dit: str, cap: float, kept: bool, starts: str
) -> None:
    management.MAX_PINNED_MEMORY = cap
    decided = models.decide_keeping(handle(tmp_path, keep=mode, dit=dit))
    assert decided.kept is kept and decided.mode == mode
    assert decided.text.startswith(starts), decided.text
    assert decided.size == size_of(dit) and decided.cap == int(cap)


def test_what_is_said_of_the_decision_has_the_sizes_and_the_cap() -> None:
    kept = plans.Keeping("auto", True, round(7.9 * GIB), round(12.8 * GIB), "x")
    assert kept.text == "kept, 7.90 GiB (ComfyUI allows 12.80 GiB)"
    assert (
        kept.line == "keep in RAM between runs: auto -> kept, 7.90 GiB (ComfyUI allows 12.80 GiB)"
    )
    over = plans.Keeping("auto", False, round(15.8 * GIB), round(12.8 * GIB), "x")
    assert over.text == "not kept: 15.80 GiB is over the 12.80 GiB ComfyUI allows"
    forced = plans.Keeping("on", True, round(15.8 * GIB), round(12.8 * GIB), "x")
    assert forced.text == "kept, 15.80 GiB, over the 12.80 GiB ComfyUI allows"
    assert forced.line.startswith("keep in RAM between runs: on -> kept, 15.80 GiB")
    off = plans.Keeping("off", False, 1, 1, "x")
    assert (
        off.line
        == "keep in RAM between runs: off -> not kept: the files are read again for each run"
    )
    nothing = plans.Keeping("on", True, round(8.2 * GIB), -1, "x")
    assert nothing.text == "kept, 8.20 GiB, though ComfyUI allows no RAM for this on this computer"


def test_the_cap_is_comfys_own_value_where_it_exposes_one(
    caplog: pytest.LogCaptureFixture,
) -> None:
    management.MAX_PINNED_MEMORY = 12.8 * GIB
    assert models.pinned_cap() == (int(12.8 * GIB), "ComfyUI's MAX_PINNED_MEMORY")
    management.MAX_PINNED_MEMORY = -1
    assert models.pinned_cap()[0] == -1


def test_the_cap_is_its_rule_where_the_manager_exposes_none(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delattr(management, "MAX_PINNED_MEMORY")
    monkeypatch.setattr(management, "SYSTEM_RAM", 32 * GIB)
    with caplog.at_level(logging.WARNING, logger="seedvr2x_comfyui.models"):
        cap, source = models.pinned_cap()
    # Linux: the larger of 40% and the smaller of 90% and the RAM less 4 GiB: 28 of 32 GiB.
    assert cap == 28 * GIB and "rule" in source
    assert any("exposes no MAX_PINNED_MEMORY" in line for line in caplog.messages)
    monkeypatch.setattr(management, "WINDOWS", True, raising=False)
    assert models.pinned_cap()[0] == int(32 * GIB * 0.40)


def test_the_run_says_the_decision_on_the_node_and_in_the_console(
    machine: StandIn, tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    management.MAX_PINNED_MEMORY = 12.8 * GIB
    with caplog.at_level(logging.INFO):
        _, text = upscale_image(
            handle(tmp_path, keep="auto", dit=FP16), SETTINGS, pictures(), RESOLUTION, "6"
        )
    assert "keep in RAM between runs: auto -> not kept: 15." in text
    assert "GiB is over the 12.80 GiB ComfyUI allows" in text
    instance = server.PromptServer.instance
    assert instance is not None and instance.texts == [(text, "6")]
    said = "seedvr2x: keep in RAM between runs: auto -> not kept: 15."
    assert any(line.startswith(said) for line in caplog.messages), "the console says it too"
    # Not kept: read for the run alone, nothing pinned in the registry, nothing registered.
    assert HELD not in machine.loads and models.kept(handle(tmp_path, dit=FP16)) is None
    assert management.current_loaded_models == []


def test_auto_that_fits_keeps_them_and_a_second_run_reads_nothing(
    machine: StandIn, tmp_path
) -> None:
    management.MAX_PINNED_MEMORY = 12.8 * GIB
    h = handle(tmp_path, keep="auto", dit=INT8)
    _, text = upscale_image(h, SETTINGS, pictures(), RESOLUTION, "1")
    upscale_image(h, SETTINGS, pictures(), RESOLUTION, "1")
    assert machine.loads == [HELD] and "keep in RAM between runs: auto -> kept, " in text
    assert models.kept(h) is not None


def test_the_cap_that_is_exceeded_is_not_a_refusal_with_on(machine: StandIn, tmp_path) -> None:
    management.MAX_PINNED_MEMORY = 1 * GIB
    _, text = upscale_image(
        handle(tmp_path, keep="on", dit=INT8), SETTINGS, pictures(), RESOLUTION, "1"
    )
    assert "on -> kept, " in text and "ComfyUI allows" in text and machine.loads == [HELD]


def test_the_first_files_pinned_ram_is_freed_before_the_second_is_pinned(
    machine: StandIn, tmp_path
) -> None:
    first, second = handle(tmp_path, dit=INT8), handle(tmp_path, dit=FP16)
    upscale_image(first, SETTINGS, pictures(), RESOLUTION, "1")
    alive = weakref.ref(machine.made[0])
    seen: list[bool] = []
    real_load = machine.load

    def load(files, plan):  # type: ignore[no-untyped-def]
        # At the moment the second set is read into pinned memory: the first is gone.
        machine.made.clear()
        gc.collect()
        seen.append(alive() is None and models.kept(first) is None)
        return real_load(files, plan)

    machine.load = load  # type: ignore[method-assign]
    upscale_image(second, SETTINGS, pictures(), RESOLUTION, "1")
    assert seen == [True]
    assert models.kept(second) is not None


@pytest.mark.parametrize("how", ["off", "does not fit"])
def test_turning_keeping_off_or_over_the_cap_frees_the_kept_set_before_the_run_reads_its_own(
    machine: StandIn, tmp_path, how: str
) -> None:
    kept = handle(tmp_path, dit=INT8)
    upscale_image(kept, SETTINGS, pictures(), RESOLUTION, "1")
    assert models.kept(kept) is not None
    alive = weakref.ref(machine.made[0])
    machine.made.clear()
    if how == "off":
        again = handle(tmp_path, keep="off", dit=INT8)
    else:
        # ComfyUI now pins nothing (its cap is off): the kept set, counted by what it pins, no
        # longer fits.
        management.MAX_PINNED_MEMORY = -1
        again = handle(tmp_path, keep="auto", dit=INT8)
    upscale_image(again, SETTINGS, pictures(), RESOLUTION, "1")
    gc.collect()
    assert models.kept(kept) is None and alive() is None
    assert machine.loads[-1] != HELD


def test_unload_models_under_a_cached_models_output_reads_the_files_again_and_says_so_once(
    machine: StandIn, tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    h = handle(tmp_path)
    upscale_image(h, SETTINGS, pictures(), RESOLUTION, "1")
    management.unload_all_models()  # the button; the Models output, h, is still cached
    assert models.kept(h) is None
    with caplog.at_level(logging.INFO, logger="seedvr2x_comfyui"):
        _, again = upscale_image(h, SETTINGS, pictures(), RESOLUTION, "2")
    note = "ComfyUI's Unload models had dropped seedvr2x's weights from RAM"
    assert note in again and "read from the files again" in again
    instance = server.PromptServer.instance
    assert instance is not None and (again, "2") in instance.texts, "said on the node"
    assert any("ComfyUI having dropped them" in line for line in caplog.messages), (
        "and in the console"
    )
    assert machine.loads == [HELD, HELD]
    _, third = upscale_image(h, SETTINGS, pictures(), RESOLUTION, "3")
    assert note not in third and len(machine.loads) == 2, "once"


def test_the_models_node_reads_nothing(machine: StandIn, tmp_path) -> None:
    from comfyui import nodes

    out = execute(
        nodes.Models,
        model=plans.DIT_LABELS[INT8],
        model_dir=str(tmp_path),
        keep_in_ram_between_runs="on",
    )
    assert out[0] == models.Handle(INT8, str(tmp_path), "on")
    assert machine.loads == [] and machine.probes == [] and machine.headers == []
    assert management.loads == [] and models.kept(out[0]) is None

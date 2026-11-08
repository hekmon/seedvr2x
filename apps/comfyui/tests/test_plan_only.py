"""Plan only (a BOOLEAN on each upscale node): the plan made, said on the node and in the console
and given as the plan output; the files' headers read and nothing else; ComfyUI asked to free
nothing; the picture output an execution blocker, so that the nodes after it do not run."""

import logging

import comfy.model_management as management
import numpy as np
import pytest
import server
import torch
from comfy.utils import ProgressBar
from comfy_execution.graph_utils import ExecutionBlocker
from comfyui import console, models, nodes, plans, words
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

from seedvr2x.passes import HELD

SETTINGS = plans.Settings()


class Seen(Exception):
    """Raised by a planner that has seen what it was given."""


def pictures(count: int = 1) -> torch.Tensor:
    return torch.from_numpy(np.random.default_rng(4).random((count, *PICTURE, 3), dtype=np.float32))


def nothing_was_done(machine: StandIn) -> None:
    assert machine.loads == [] and machine.made == [] and machine.probes == []
    assert management.loads == [] and management.current_loaded_models == []
    assert ProgressBar.created == []
    assert not [call for call in management.free_calls if call[2]], "ComfyUI asked to free nothing"


@pytest.mark.parametrize("keep", [True, False])
def test_a_picture_plan_only_says_the_plan_and_blocks_what_follows_the_picture(
    machine: StandIn, tmp_path, caplog: pytest.LogCaptureFixture, keep: bool
) -> None:
    with caplog.at_level(logging.INFO, logger="seedvr2x_comfyui.nodes"):
        blocked, text = upscale_image(
            handle(tmp_path, keep=keep), SETTINGS, pictures(2), RESOLUTION, "12", plan_only=True
        )
    # An execution blocker with no message: the executor passes it to every node that takes the
    # picture, which then does not run, without reporting an error (a message would be one).
    assert isinstance(blocked, ExecutionBlocker) and blocked.message is None
    assert text.startswith("plan (plan only): 2 images")
    assert "plan only: nothing is loaded and ComfyUI's models stay as they are" in text
    instance = server.PromptServer.instance
    # The node says "planning…" and then the plan's main numbers in one short line; the plan is
    # in the console.
    assert instance is not None and instance.texts[0] == ("planning\u2026", "12")
    assert [each for each, _ in instance.texts[1:]] == ["plan: encode whole \u00b7 decode whole"]
    assert console.prefixed(text) in caplog.messages
    nothing_was_done(machine)
    assert models.kept(handle(tmp_path)) is None
    assert len(machine.headers) == 1, "the DiT's header was read"


def test_a_clip_plan_only_says_the_plan_and_blocks_the_frames(machine: StandIn, tmp_path) -> None:
    blocked, text = upscale_video(
        handle(tmp_path), plans.Settings(window=5), pictures(33), RESOLUTION, "3", plan_only=True
    )
    assert isinstance(blocked, ExecutionBlocker) and blocked.message is None
    assert "(plan only): a clip of 33 frames" in text and "window length: " in text
    instance = server.PromptServer.instance
    assert instance is not None
    assert [each for each, _ in instance.texts] == [
        "planning\u2026",
        "plan: window 17 frames \u00b7 decode whole",
    ]
    nothing_was_done(machine)


def test_the_plan_only_text_is_the_text_of_the_run(machine: StandIn, tmp_path) -> None:
    """The plan a run says is the plan only's, but for the header and the room's line: the
    settings and what they came to are the same."""
    _, only = upscale_image(
        handle(tmp_path), plans.Settings(seed=3), pictures(), RESOLUTION, "1", plan_only=True
    )
    _, ran = upscale_image(handle(tmp_path), plans.Settings(seed=3), pictures(), RESOLUTION, "1")

    def settings_lines(text: str) -> list[str]:
        return [
            line
            for line in text.splitlines()
            if line.split(":")[0] in {"attention", "blocks swapped", "encode tile", "decode tile"}
        ]

    assert settings_lines(only) == settings_lines(ran) and settings_lines(ran)


def test_a_plan_only_asks_comfyui_to_free_nothing_and_says_what_a_run_would_need(
    machine: StandIn, tmp_path
) -> None:
    resident = comfy_model(20 * GIB)
    _, text = upscale_image(handle(tmp_path), SETTINGS, pictures(), RESOLUTION, "1", plan_only=True)
    assert resident.loaded_size() == 20 * GIB, "ComfyUI's model stays where it is"
    assert "a run would need up to" in text and "free on the card, with a safety margin" in text
    assert "ComfyUI's own models use 20.00 GiB" in text
    assert not [call for call in management.free_calls if call[2]]


def test_a_plan_only_is_refused_as_a_run_is_before_anything_loads(
    machine: StandIn, tmp_path
) -> None:
    # 4.0 GiB: a picture of 1080p fits 5.6 since the lean attention (2026-11-06), which this
    # card's total was picked to refuse (between 4.0 and 4.5 GiB now).
    management.card.total = round(4.0 * GIB)
    big = torch.zeros((1, 1080, 1920, 3), dtype=torch.float32)
    with pytest.raises(words.Refused) as refused:
        upscale_image(
            handle(tmp_path, dit="seedvr2x_ema_7b_sharp_fp16.safetensors"),
            SETTINGS,
            big,
            2160,
            "9",
            plan_only=True,
        )
    assert "--" not in str(refused.value)
    nothing_was_done(machine)


def test_weights_a_run_would_pin_come_out_of_the_memory_the_host_has_for_the_plan(
    machine: StandIn, tmp_path
) -> None:
    seen: list[int | None] = []

    def planner(reading, host, attention, architecture, held):  # type: ignore[no-untyped-def]
        seen.append(host)
        raise Seen

    kept = handle(tmp_path, keep="on", dit="seedvr2x_ema_7b_sharp_int8_convrot.safetensors")
    for h, expected in (
        (kept, 64 * GIB - models.pinned_size(kept)),
        (handle(tmp_path, keep="off"), 64 * GIB),
    ):
        with pytest.raises(Seen), models.session(h, None, planner, plan_only=True):
            pass
        assert seen[-1] == expected
    # Weights already pinned are not in the available figure: nothing taken off it.
    upscale_image(kept, SETTINGS, pictures(), RESOLUTION, "1")
    with pytest.raises(Seen), models.session(kept, None, planner, plan_only=True):
        pass
    assert seen[-1] == 64 * GIB


def test_a_kept_models_plan_only_does_not_load_register_or_place_the_weights(
    machine: StandIn, tmp_path
) -> None:
    h = handle(tmp_path)
    upscale_image(h, SETTINGS, pictures(), RESOLUTION, "1")
    placed = len(machine.made[0].placed)  # type: ignore[attr-defined]
    upscale_image(h, SETTINGS, pictures(), RESOLUTION, "1", plan_only=True)
    assert machine.loads == [HELD] and len(machine.made[0].placed) == placed  # type: ignore[attr-defined]
    assert len(management.current_loaded_models) == 1


def test_the_models_of_a_plan_only_run_are_refused() -> None:
    run = models.Run(None, None, None, "sdpa", models.Handle("a"), None, ())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="plan only"):
        run.models  # noqa: B018


def test_the_node_is_the_one_that_blocks(machine: StandIn, tmp_path) -> None:
    # The same node run without plan only gives the pictures, no blocker.
    out, _ = upscale_image(handle(tmp_path), SETTINGS, pictures(), RESOLUTION, "1")
    assert not isinstance(out, ExecutionBlocker) and out.shape == (1, 48, 80, 3)
    assert nodes.UpscaleImage.GET_SCHEMA().outputs[0].io_type == "IMAGE"

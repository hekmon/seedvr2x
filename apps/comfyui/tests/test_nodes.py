"""The upscale nodes over the stand-in models (CPU, no weights): each picture of a batch equal bit
for bit to the library's still.upscale of it alone, a clip to clip.upscale, the alpha channel's
rule, the plan said three ways, the settings' choices reaching the plan and the pass."""

import logging

import numpy as np
import pytest
import server
import torch
from comfy.utils import ProgressBar
from comfyui import models, nodes, plans, words
from standins import (
    PICTURE,
    RESOLUTION,
    StandIn,
    handle,
    standin_models,
    upscale_image,
    upscale_video,
)

from seedvr2x import clip, peaks, still
from seedvr2x.memory import Reading

TARGET = (48, 80)
GIB = peaks.GIB


def pictures(count: int, channels: int = 3) -> torch.Tensor:
    rng = np.random.default_rng(11)
    return torch.from_numpy(rng.random((count, *PICTURE, channels), dtype=np.float32))


def fresh(machine: StandIn) -> models.Models:  # type: ignore[name-defined]
    """Models as the machine loads them, for the library's own call to compare with."""
    return standin_models(machine.vae_models, [], held=False)  # type: ignore[return-value]


def planned_clip(machine: StandIn, settings: plans.Settings, count: int) -> clip.Plan:  # type: ignore[name-defined]
    reading = Reading(32 * GIB, None, 32 * GIB, False)
    made = nodes.clip_planner(settings, TARGET, count)(
        reading, machine.host_bytes, peaks.FLASH_ATTN_2, machine.dit_format, True
    )
    return made.plan[0]


def test_each_picture_of_a_batch_is_the_libraries_picture_alone(machine: StandIn, tmp_path) -> None:
    batch = pictures(3)
    settings = plans.Settings(seed=7)
    out, plan = upscale_image(handle(tmp_path), settings, batch, RESOLUTION, "4")
    assert out.shape == (3, 48, 80, 3) and out.dtype == torch.float32
    alone = fresh(machine)
    for index in range(3):
        expected = still.upscale(
            alone,
            batch[index].numpy(),
            TARGET,
            7,
            encode_tiles=None,
            decode_tiles=None,
            correct=True,
        )
        assert out[index].numpy().tobytes() == expected.tobytes(), index
    assert out[0].numpy().tobytes() != out[1].numpy().tobytes()
    assert plan.startswith("seedvr2x plan: 3 pictures")


def test_a_clip_is_the_libraries_clip(machine: StandIn, tmp_path) -> None:
    frames = pictures(33)
    settings = plans.Settings(window=5, color_correction="none", seed=9)
    out, plan = upscale_video(handle(tmp_path), settings, frames, RESOLUTION, "4")
    made = planned_clip(machine, settings, 33)
    assert made.window == 5
    expected = clip.upscale(
        fresh(machine),
        frames.numpy(),
        TARGET,
        9,
        window=5,
        encode_tiles=made.encode_tiles,
        decode_tiles=made.decode_tiles,
        correct=False,
    )
    assert out.shape == (33, 48, 80, 3)
    assert out.numpy().tobytes() == expected.tobytes()
    assert "3 DiT windows, windows of at most 5 latents" in plan


def test_a_clips_bar_moves_with_its_steps_and_ends_full(machine: StandIn, tmp_path) -> None:
    upscale_video(handle(tmp_path), plans.Settings(window=5), pictures(33), RESOLUTION, "4")
    (bar,) = ProgressBar.created
    assert bar.updates == sorted(bar.updates) and bar.updates[-1] == bar.total == nodes.BAR
    # The encode, the three windows, and the decode's slices.
    assert bar.updates[0] == nodes.ENCODE_SHARE
    assert len(bar.updates) >= 1 + 3 + 1


def test_the_color_correction_and_the_seed_reach_the_pass(machine: StandIn, tmp_path) -> None:
    batch = pictures(1)
    corrected = upscale_image(handle(tmp_path), plans.Settings(seed=1), batch, RESOLUTION, "1")[0]
    plain = upscale_image(
        handle(tmp_path), plans.Settings(color_correction="none", seed=1), batch, RESOLUTION, "1"
    )[0]
    other = upscale_image(handle(tmp_path), plans.Settings(seed=2), batch, RESOLUTION, "1")[0]
    alone = fresh(machine)
    for got, correct, seed in ((corrected, True, 1), (plain, False, 1), (other, True, 2)):
        expected = still.upscale(
            alone,
            batch[0].numpy(),
            TARGET,
            seed,
            encode_tiles=None,
            decode_tiles=None,
            correct=correct,
        )
        assert got[0].numpy().tobytes() == expected.tobytes()
    assert corrected.numpy().tobytes() != plain.numpy().tobytes()


def test_an_alpha_channel_opaque_everywhere_is_dropped_and_any_other_refused(
    machine: StandIn, tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    rgb = pictures(1)
    opaque = torch.cat([rgb, torch.ones(1, *PICTURE, 1)], dim=3)
    with caplog.at_level(logging.INFO, logger="seedvr2x_comfyui.nodes"):
        out = upscale_image(handle(tmp_path), plans.Settings(), opaque, RESOLUTION, "1")
    expected = upscale_image(handle(tmp_path), plans.Settings(), rgb, RESOLUTION, "1")
    assert out[0].numpy().tobytes() == expected[0].numpy().tobytes()
    assert any("opaque everywhere" in line for line in caplog.messages)
    clear = opaque.clone()
    clear[0, 3, 3, 3] = 0.5
    with pytest.raises(words.Refused, match="transparent pixels") as refused:
        upscale_image(handle(tmp_path), plans.Settings(), clear, RESOLUTION, "1")
    assert "--" not in str(refused.value)
    with pytest.raises(words.Refused, match="2 channels"):
        upscale_image(handle(tmp_path), plans.Settings(), rgb[..., :2], RESOLUTION, "1")
    with pytest.raises(words.Refused, match="too small") as small:
        upscale_image(handle(tmp_path), plans.Settings(), rgb, 10, "1")
    assert "--" not in str(small.value) and "resolution" in str(small.value)


def test_the_plan_is_said_on_the_node_in_the_console_and_given_as_text(
    machine: StandIn, tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = plans.Settings(swap_blocks=0, decode_tile=None, attention="sdpa", seed=5)
    with caplog.at_level(logging.INFO, logger="seedvr2x_comfyui.nodes"):
        _, text = upscale_image(handle(tmp_path), settings, pictures(1), RESOLUTION, "31")
    instance = server.PromptServer.instance
    assert instance is not None and instance.texts == [(text, "31")]
    assert text in caplog.messages
    # The plan's parts in the nodes' words: the pictures, the card and ComfyUI's models, what
    # ComfyUI was asked, the DiT's placement and kernel, the passes, the choices, the models' RAM.
    for part in (
        "1 picture 40x24 to 80x48 (resolution 48)",
        "card: a stand-in card, 32.00 GiB free; ComfyUI's own models hold 0.00 GiB",
        "ComfyUI was asked to free nothing",
        # Each setting and what it came to: a pin as set, the rest as the plan made it.
        "attention: PyTorch's SDPA, as set",
        "swap blocks: none, the DiT stays on the card in its phase, as set",
        "encode tile: auto -> one pass, untiled, predicted peak",
        "decode tile: auto -> one pass, untiled, predicted peak",
        "colour correction split, seed 5",
        "models: kept, ",
    ):
        assert part in text, part
    assert "--" not in text
    _, video = upscale_video(handle(tmp_path), settings, pictures(5), RESOLUTION, "32")
    assert "a clip of 5 frames 40x24 to 80x48" in video and "1 DiT window" in video
    assert "window cap: auto -> at most 2 latents" in video and "--" not in video
    assert "system memory: the plan counts" in video
    assert [node for _, node in instance.texts] == ["31", "32"]


def test_the_text_names_the_files_not_kept(machine: StandIn, tmp_path) -> None:
    _, text = upscale_image(
        handle(tmp_path, keep=False), plans.Settings(), pictures(1), RESOLUTION, "1"
    )
    assert "models: not kept, as asked: the files are read for each run and dropped" in text


def test_a_run_without_a_node_id_says_nothing_on_a_node(machine: StandIn, tmp_path) -> None:
    upscale_image(handle(tmp_path), plans.Settings(), pictures(1), RESOLUTION, None)
    instance = server.PromptServer.instance
    assert instance is not None and instance.texts == []


def test_the_attention_chosen_in_tuning_reaches_the_models(machine: StandIn, tmp_path) -> None:
    h = handle(tmp_path)
    upscale_image(h, plans.Settings(attention="sdpa"), pictures(1), RESOLUTION, "1")
    assert machine.made[0].attention == "sdpa"
    assert machine.probes == [(peaks.FP16.name, "sdpa")]
    upscale_image(h, plans.Settings(), pictures(1), RESOLUTION, "1")
    assert machine.made[0].attention == peaks.FLASH_ATTN_2
    assert len(machine.probes) == 2
    machine.attention_name = peaks.SDPA
    with pytest.raises(words.Refused, match="FlashAttention 2 isn't installed") as refused:
        upscale_image(h, plans.Settings(attention="flash_attn_2"), pictures(1), RESOLUTION, "1")
    assert "--" not in str(refused.value)


def test_the_probe_runs_once_a_layout_and_kernel_and_its_refusal_is_worded(
    machine: StandIn, tmp_path
) -> None:
    from pathlib import Path

    from seedvr2x.probe import ProbeRefusal, Result

    h = handle(tmp_path)
    upscale_image(h, plans.Settings(), pictures(1), RESOLUTION, "1")
    upscale_image(h, plans.Settings(), pictures(1), RESOLUTION, "1")
    assert machine.probes == [(peaks.FP16.name, peaks.FLASH_ATTN_2)]
    models.reset()
    result = Result("matmul bfloat16", 3.0, 0.1, "spacings", False, "2026-11-05T10:00:00")
    failure = ProbeRefusal(
        result, {"gpu": "a GPU", "torch": "2.9.1"}, Path("probe.json"), cached=False
    )

    def refuse(*args: object) -> None:
        raise failure

    machine.probe = refuse  # type: ignore[method-assign]
    with pytest.raises(words.Refused, match="matmul bfloat16 gives the wrong numbers") as refused:
        upscale_image(h, plans.Settings(), pictures(1), RESOLUTION, "1")
    assert "ComfyUI's environment" in str(refused.value) and "--" not in str(refused.value)
    assert machine.loads == [machine.loads[0]]  # refused before the second load


def test_a_batchs_output_is_allocated_once_and_never_stacked(
    machine: StandIn, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = pictures(3)
    expected, _ = upscale_image(handle(tmp_path), plans.Settings(seed=7), batch, RESOLUTION, "4")

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the pictures were collected and stacked: held twice")

    # A list of the pictures and np.stack would hold every output twice at the stack.
    monkeypatch.setattr(np, "stack", refuse)
    monkeypatch.setattr(np, "concatenate", refuse)
    empties: list[tuple[int, ...]] = []
    real = np.empty

    def empty(shape: tuple[int, ...], *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        empties.append(tuple(shape))
        return real(shape, *args, **kwargs)  # type: ignore[call-overload]

    monkeypatch.setattr(np, "empty", empty)
    out, _ = upscale_image(handle(tmp_path), plans.Settings(seed=7), batch, RESOLUTION, "4")
    assert out.numpy().tobytes() == expected.numpy().tobytes()
    assert (3, 48, 80, 3) in empties and empties.count((3, 48, 80, 3)) == 1


def test_the_batchs_output_is_counted_in_the_process_memory_the_plan_is_given(
    machine: StandIn, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from seedvr2x import still_plan

    out_pixels = 48 * 80 * 3 * 4
    one = nodes.picture_process(PICTURE, TARGET)
    # One picture: as before the batch was counted (the picture as stored, the output and the
    # decode's copy of it); each picture more, one output more.
    assert one == nodes.HOST_RESIDENT + PICTURE[0] * PICTURE[1] * 3 * 4 + 2 * out_pixels
    assert nodes.picture_process(PICTURE, TARGET, 3) == one + 2 * out_pixels
    given: list[int] = []
    real = still_plan.plan

    def plan(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        given.append(int(kwargs["process"]))  # type: ignore[call-overload]
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(still_plan, "plan", plan)
    upscale_image(handle(tmp_path), plans.Settings(), pictures(3), RESOLUTION, "4")
    assert given and set(given) == {nodes.picture_process(PICTURE, TARGET, 3)}


def test_a_clip_is_one_shot_run_by_one_function_called_once(
    machine: StandIn, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A list of shots (frame ranges from a later Cuts node) is a loop around this call."""
    calls: list[tuple[int, plans.Settings]] = []
    real = nodes.UpscaleVideo._shot  # pyright: ignore[reportPrivateUsage]

    def shot(loaded, shot_frames, target, settings, made, bar):  # type: ignore[no-untyped-def]
        calls.append((len(shot_frames), settings))
        return real(loaded, shot_frames, target, settings, made, bar)

    monkeypatch.setattr(nodes.UpscaleVideo, "_shot", staticmethod(shot))
    out, _ = upscale_video(handle(tmp_path), plans.Settings(seed=2), pictures(9), RESOLUTION, "1")
    assert calls == [(9, plans.Settings(seed=2))] and out.shape == (9, 48, 80, 3)

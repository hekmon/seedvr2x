"""The upscale nodes over the stand-in models (CPU, no weights): each picture of a batch equal bit
for bit to the library's still.upscale of it alone, a clip to clip.upscale, the alpha channel's
rule, the plan said three ways, the settings' choices reaching the plan and the pass."""

import logging
import re
from typing import Any

import numpy as np
import pytest
import server
import torch
from comfy.utils import ProgressBar
from comfyui import console, models, nodes, plans, progress, words
from standins import (
    PICTURE,
    RESOLUTION,
    StandIn,
    handle,
    in_console,
    standin_models,
    texts,
    upscale_image,
    upscale_video,
    where,
)

from seedvr2x import clip, peaks, still
from seedvr2x.memory import Reading
from seedvr2x.shots import Shot

TARGET = (48, 80)
GIB = peaks.GIB


def pictures(count: int, channels: int = 3) -> torch.Tensor:
    rng = np.random.default_rng(11)
    return torch.from_numpy(rng.random((count, *PICTURE, channels), dtype=np.float32))


def fresh(machine: StandIn) -> models.Models:  # type: ignore[name-defined]
    """Models as the machine loads them, for the library's own call to compare with."""
    return standin_models(machine.vae_models, [], held=False)  # type: ignore[return-value]


def planned_clip(machine: StandIn, settings: plans.Settings, cuts: tuple[Shot, ...] | int):  # type: ignore[no-untyped-def]
    reading = Reading(32 * GIB, None, 32 * GIB, False)
    shot_list = [Shot(0, cuts)] if isinstance(cuts, int) else cuts
    made = nodes.clip_planner(settings, TARGET, shot_list)(
        reading, machine.host_bytes, machine.dit_format, True
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
    assert plan.startswith("plan: 3 images")


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
    assert "3 windows of the model, the longest" in plan
    assert "window length: at most 17 frames, as you set it" in plan


def test_a_clips_bar_moves_with_its_steps_and_ends_full(machine: StandIn, tmp_path) -> None:
    upscale_video(handle(tmp_path), plans.Settings(window=5), pictures(33), RESOLUTION, "4")
    (bar,) = ProgressBar.created
    assert bar.updates == sorted(bar.updates) and bar.updates[-1] == bar.total == progress.BAR
    # The pass beginning, the encode, the three windows, and the decode's slices (tests/
    # test_progress.py holds the bar's arithmetic).
    assert bar.updates[0] == 0
    assert len(bar.updates) >= 1 + 1 + 3 + 1


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


def test_a_refusal_shows_one_line_on_the_node_and_its_text_once_in_the_console(
    machine: StandIn, tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    """Every refusal path shows the node's line (a context manager around each execute's body),
    the refusal's text staying in ComfyUI's error and, once, in the console at WARNING."""
    rgb = pictures(1)
    clear = torch.cat([rgb, torch.full((1, *PICTURE, 1), 0.5)], dim=3)
    with caplog.at_level(logging.INFO, logger="seedvr2x_comfyui"):
        with pytest.raises(words.Refused, match="transparent pixels") as refused:
            upscale_image(handle(tmp_path), plans.Settings(), clear, RESOLUTION, "5")
        with pytest.raises(words.Refused, match="2 channels"):
            upscale_video(handle(tmp_path), plans.Settings(), rgb[..., :2], RESOLUTION, "6")
    instance = server.PromptServer.instance
    assert instance is not None
    assert [text for text, node in instance.texts if node == "5"] == [
        "planning\u2026",
        "refused: see the error",
    ]
    assert [text for text, node in instance.texts if node == "6"][-1] == progress.REFUSED
    said = [
        (r.levelno, r.getMessage().removeprefix(console.PREFIX))
        for r in caplog.records
        if "refused: " in r.getMessage()
    ]
    assert len(said) == 2, "once each"
    assert said[0] == (logging.WARNING, f"refused: {refused.value}")
    assert said[1][0] == logging.WARNING
    assert said[1][1].startswith("refused: the image has 2 channels")


def test_the_run_is_said_in_the_console_with_its_frames_and_time(
    machine: StandIn, tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    """What the node's line ends with ("33 frames in 0 s") is the console's too, in full, and the
    shot lines of a clip of several shots are the console's alone."""
    with caplog.at_level(logging.INFO, logger="seedvr2x_comfyui.nodes"):
        upscale_image(handle(tmp_path), plans.Settings(), pictures(3), RESOLUTION, "1")
        run_shots(machine, tmp_path, pictures(21), (Shot(0, 9), Shot(9, 21)), window=5)
        upscale_video(handle(tmp_path), plans.Settings(), pictures(1), RESOLUTION, "3")
    lines = [each.removeprefix(console.PREFIX) for each in caplog.messages]
    assert any(re.fullmatch(r"3 pictures upscaled in \d+ s", each) for each in lines)
    assert any(re.fullmatch(r"21 frames upscaled in \d+ s", each) for each in lines)
    assert any(re.fullmatch(r"1 frame upscaled in \d+ s", each) for each in lines)
    assert "shot 1 of 2, frames 0 to 8" in lines and "shot 2 of 2, frames 9 to 20" in lines


def test_a_resolution_whose_even_crop_is_too_small_is_refused_for_a_picture_alone(
    machine: StandIn, tmp_path
) -> None:
    """Resolution 17 on a 40x24 picture is 28x17, 28x16 once cropped to even sides: a picture's
    plan is made for the crop, so the picture is refused in the nodes' words, saying the output's
    size and the 18 rows it needs. A clip at 17 is planned as the command line plans a video: its
    frames are padded at the target's 17 rows and cropped after the decode."""
    rgb = pictures(1)
    with pytest.raises(words.Refused, match="too small") as small:
        upscale_image(handle(tmp_path), plans.Settings(), rgb, 17, "1")
    assert str(small.value) == (
        "resolution 17 on this node makes a 28x16 image, which is too small: seedvr2x needs 18 px"
        " on the short side at least; raise resolution on this node"
    )
    assert machine.loads == []
    out, plan = upscale_video(handle(tmp_path), plans.Settings(), pictures(5), 17, "1")
    assert out.shape == (5, 16, 28, 3) and plan.startswith("plan: a clip of 5 frames")


def test_a_seed_some_shot_would_seed_past_the_range_is_refused_before_anything_loads(
    machine: StandIn, tmp_path
) -> None:
    """A shot is seeded with the seed plus its first frame, its encode with 1,000,000 more
    (job.check_seed on the command line): the seed widget's largest value is right for a shot at
    frame 0 only."""
    frames = pictures(21)
    cuts = (Shot(0, 9), Shot(9, 21))
    with pytest.raises(
        words.Refused, match="seed 4,293,967,295 is too high for this cut list"
    ) as r:
        run_shots(machine, tmp_path, frames, cuts, seed=plans.SEED_MAX)
    assert "the last starts at frame 9" in str(r.value)
    assert "can be 4,293,967,286 at most" in str(r.value) and "--" not in str(r.value)
    assert machine.loads == []
    # A seed past the range is folded first, and the message names both.
    given = plans.SEED_MAX + 1 + plans.SEED_MAX
    with pytest.raises(words.Refused) as folded:
        run_shots(machine, tmp_path, frames, cuts, seed=given)
    assert (
        f"seed {plans.SEED_MAX:,} ({given:,} given, taken modulo 4,293,967,296) is too high"
        in str(folded.value)
    )
    assert machine.loads == []
    # The highest seed that works is taken, and so is any seed with one shot.
    run_shots(machine, tmp_path, frames, cuts, seed=plans.SEED_MAX - 9, window=5)
    run_shots(machine, tmp_path, frames, (Shot(0, 21),), seed=plans.SEED_MAX, window=5)


def test_four_channels_with_an_opaque_alpha_are_a_view_not_a_copy() -> None:
    rgb = pictures(2)
    opaque = torch.cat([rgb, torch.ones(2, *PICTURE, 1)], dim=3)
    array = nodes._array(opaque)
    assert array.shape == (2, *PICTURE, 3) and np.shares_memory(array, opaque.numpy())
    assert array.tobytes() == rgb.numpy().tobytes()
    assert np.shares_memory(nodes._array(rgb), rgb.numpy())


def test_a_clip_with_an_opaque_alpha_is_the_clip_without_it(machine: StandIn, tmp_path) -> None:
    rgb = pictures(9)
    opaque = torch.cat([rgb, torch.ones(9, *PICTURE, 1)], dim=3)
    cuts = (Shot(0, 4), Shot(4, 9))
    with_alpha, _ = run_shots(machine, tmp_path, opaque, cuts, window=5)
    without, _ = run_shots(machine, tmp_path, rgb, cuts, window=5)
    assert with_alpha.numpy().tobytes() == without.numpy().tobytes()


def test_the_plan_is_said_whole_in_the_console_and_the_node_shows_one_short_line(
    machine: StandIn, tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = plans.Settings(swap_blocks=0, decode_tile=None, seed=5)
    with caplog.at_level(logging.INFO, logger="seedvr2x_comfyui.nodes"):
        _, text = upscale_image(handle(tmp_path), settings, pictures(1), RESOLUTION, "31")
    instance = server.PromptServer.instance
    # The node shows "planning…" and then the run's progress, never the plan; the plan whole is
    # in the console, each of its lines prefixed.
    assert instance is not None and instance.texts[0] == ("planning\u2026", "31")
    assert all(text not in each for each, _ in instance.texts)
    assert in_console(text, caplog.messages)
    # The plan's parts in the nodes' words: the pictures, the card and ComfyUI's models, what
    # ComfyUI was asked, the DiT's placement, the passes, the choices, the models' RAM.
    for part in (
        "1 image 40x24 to 80x48 (x2, resolution 48)",
        "graphics card: a stand-in card, 32.00 GiB free; ComfyUI's own models use 0.00 GiB",
        "no need to unload ComfyUI's models",
        # Each setting and what it came to: a value given as set, the rest as the plan chose.
        "blocks swapped: none, the whole model on the card while it works, as you set it",
        "encode tile: auto -> the whole image at once, no tiles (needs up to ",
        "decode tile: auto -> the whole image at once, no tiles (needs up to ",
        "colour correction: on (split)",
        "seed: 5",
        "keep in RAM between runs: on -> kept, ",
    ):
        assert part in text, part
    assert "--" not in text
    _, video = upscale_video(handle(tmp_path), settings, pictures(5), RESOLUTION, "32")
    assert "a clip of 5 frames 40x24 to 80x48" in video
    assert "the whole clip in one window of the model" in video
    assert "window length: auto -> at most 5 frames" in video
    assert "RAM: this run needs about" in video and "--" not in video
    assert {node for _, node in instance.texts} == {"31", "32"}


def test_the_text_names_the_files_not_kept(machine: StandIn, tmp_path) -> None:
    _, text = upscale_image(
        handle(tmp_path, keep=False), plans.Settings(), pictures(1), RESOLUTION, "1"
    )
    assert "keep in RAM between runs: off -> not kept: the files are read again" in text


def test_a_run_without_a_node_id_says_nothing_on_a_node(machine: StandIn, tmp_path) -> None:
    upscale_image(handle(tmp_path), plans.Settings(), pictures(1), RESOLUTION, None)
    instance = server.PromptServer.instance
    assert instance is not None and instance.texts == []


def test_the_probe_runs_once_a_layout_and_its_refusal_is_worded(machine: StandIn, tmp_path) -> None:
    from pathlib import Path

    from seedvr2x.probe import ProbeRefusal, Result

    h = handle(tmp_path)
    upscale_image(h, plans.Settings(), pictures(1), RESOLUTION, "1")
    upscale_image(h, plans.Settings(), pictures(1), RESOLUTION, "1")
    assert machine.probes == [peaks.FP16.name]
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


def run_shots(
    machine: StandIn, tmp_path, frames: torch.Tensor, cuts: tuple[Shot, ...], **settings: Any
) -> tuple[torch.Tensor, str]:
    return upscale_video(
        handle(tmp_path),
        plans.Settings(**settings),
        frames,
        RESOLUTION,
        "1",
        cuts=cuts,  # type: ignore[arg-type]
    )


def libraries_clip(
    machine: StandIn,
    frames: torch.Tensor,
    cuts: tuple[Shot, ...],
    seed: int,
    quality: str = "standard",
    **plan,
):  # type: ignore[no-untyped-def]
    """What the library gives each shot alone (clip.upscale, its seed the shot's by shots.Shot),
    the shots' frames one after the other."""
    settings = plans.Settings(seed=seed, **plan)
    made = planned_clip(machine, settings, cuts)
    alone = fresh(machine)
    parts = [
        clip.upscale(
            alone,
            frames.numpy()[each.start : each.end],
            TARGET,
            seed + each.start,
            window=made.window,
            encode_tiles=made.encode_tiles,
            decode_tiles=made.decode_tiles,
            correct=True,
            quality=quality,
        )
        for each in cuts
    ]
    return np.concatenate(parts)


@pytest.mark.parametrize(
    "cuts",
    [
        (Shot(0, 21),),
        (Shot(0, 9), Shot(9, 21)),
        (Shot(0, 5), Shot(5, 18), Shot(18, 21)),
    ],
    ids=["1 shot", "2 shots", "3 shots"],
)
def test_a_clip_of_shots_is_the_libraries_shots_one_after_the_other(
    machine: StandIn, tmp_path, cuts: tuple[Shot, ...]
) -> None:
    frames = pictures(21)
    out, _ = run_shots(machine, tmp_path, frames, cuts, seed=4, window=5)
    expected = libraries_clip(machine, frames, cuts, 4, window=5)
    assert out.shape == (21, 48, 80, 3) and out.dtype == torch.float32
    assert out.numpy().tobytes() == expected.tobytes()


@pytest.mark.parametrize("quality", ["enhanced", "maximum"])
@pytest.mark.parametrize(
    "cuts",
    [(Shot(0, 21),), (Shot(0, 9), Shot(9, 21)), (Shot(0, 2), Shot(2, 21))],
    ids=["1 shot", "2 shots", "a shot of 2 frames"],
)
def test_a_clip_at_a_quality_is_the_librarys_clip_at_it_shot_by_shot(
    machine: StandIn, tmp_path, cuts: tuple[Shot, ...], quality: str
) -> None:
    frames = pictures(21)
    out, text = upscale_video(
        handle(tmp_path),
        plans.Settings(seed=4, window=5),
        frames,
        RESOLUTION,
        "1",
        cuts=cuts,
        quality=quality,
    )
    expected = libraries_clip(machine, frames, cuts, 4, quality, window=5)
    assert out.numpy().tobytes() == expected.tobytes()
    # Not the standard run's: the level reached the pass.
    standard, _ = run_shots(machine, tmp_path, frames, cuts, seed=4, window=5)
    assert out.numpy().tobytes() != standard.numpy().tobytes()
    assert f"quality: {quality}, the model runs " in text


def test_a_level_the_node_does_not_have_is_refused_before_anything_loads(
    machine: StandIn, tmp_path
) -> None:
    frames = pictures(9)
    _, text = upscale_video(handle(tmp_path), plans.Settings(window=5), frames, RESOLUTION)
    assert "quality: standard, the model runs once over each shot" in text
    loads = len(machine.loads)
    # A combo is held to its options when the prompt is queued; a link may carry anything.
    with pytest.raises(words.Refused, match="quality 'ultra': must be one of standard"):
        upscale_video(
            handle(tmp_path), plans.Settings(window=5), frames, RESOLUTION, quality="ultra"
        )
    assert len(machine.loads) == loads


def test_the_runs_of_a_shot_write_into_the_one_output_array_the_node_allocates(
    machine: StandIn, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    given: list[tuple[int, ...]] = []
    real = clip.upscale

    def spy(models, shot_frames, target, seed, **kwargs):  # type: ignore[no-untyped-def]
        given.append(kwargs["output"].shape)
        assert kwargs["quality"] == "maximum"
        return real(models, shot_frames, target, seed, **kwargs)

    monkeypatch.setattr(clip, "upscale", spy)
    out, _ = upscale_video(
        handle(tmp_path),
        plans.Settings(window=5),
        pictures(21),
        RESOLUTION,
        cuts=(Shot(0, 9), Shot(9, 21)),
        quality="maximum",
    )
    assert given == [(9, 48, 80, 3), (12, 48, 80, 3)] and out.shape == (21, 48, 80, 3)


def test_a_bar_with_several_runs_moves_forward_over_them_and_ends_full(
    machine: StandIn, tmp_path
) -> None:
    cuts = (Shot(0, 9), Shot(9, 21))
    upscale_video(
        handle(tmp_path), plans.Settings(window=5), pictures(21), RESOLUTION, cuts=cuts,
        quality="maximum",
    )  # fmt: skip
    (bar,) = ProgressBar.created
    assert bar.updates == sorted(bar.updates) and bar.updates[-1] == bar.total == progress.BAR
    # The bar's units are every run's own (tests/test_progress.py holds the arithmetic): each run
    # of each shot is said on the node, four runs a shot.
    instance = server.PromptServer.instance
    assert instance is not None
    # Each run of each shot on the node, the word "shot" gone where the run and the time need its
    # room (tests/test_lines.py holds the order).
    said = {where(text) for text, _ in instance.texts[1:-1]}
    runs = {re.match(r"(?:shot )?(\d)/2 \u00b7 run (\d)/4", each) for each in said}
    assert {each.groups() for each in runs if each is not None} == {
        (str(shot), str(n)) for shot in (1, 2) for n in (1, 2, 3, 4)
    }


def test_the_shots_are_seeded_as_the_command_line_seeds_them(machine: StandIn, tmp_path) -> None:
    """A shot's seed is the job's plus its first frame (shots.Shot.seed): the second shot of a
    clip is the library's clip of its frames run at the seed plus 9, and not at the seed."""
    frames = pictures(21)
    cuts = (Shot(0, 9), Shot(9, 21))
    out, _ = run_shots(machine, tmp_path, frames, cuts, seed=4, window=5)
    made = planned_clip(machine, plans.Settings(seed=4, window=5), cuts)
    kept: dict[str, Any] = dict(
        window=made.window,
        encode_tiles=made.encode_tiles,
        decode_tiles=made.decode_tiles,
        correct=True,
    )
    second = frames.numpy()[9:]
    at_13 = clip.upscale(fresh(machine), second, TARGET, 13, **kept)
    at_4 = clip.upscale(fresh(machine), second, TARGET, 4, **kept)
    assert out[9:].numpy().tobytes() == at_13.tobytes() != at_4.tobytes()


def test_no_window_crosses_a_cut(
    machine: StandIn, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each shot is a pass of its own over its own frames, in order, under the plan's one cap; and
    the frames of one shot change nothing of another's output."""
    calls: list[tuple[int, int, int | None]] = []
    real = clip.upscale

    def spy(models, shot_frames, target, seed, **kwargs):  # type: ignore[no-untyped-def]
        calls.append((len(shot_frames), kwargs["start"], kwargs["window"]))
        return real(models, shot_frames, target, seed, **kwargs)

    monkeypatch.setattr(clip, "upscale", spy)
    frames = pictures(21)
    cuts = (Shot(0, 9), Shot(9, 21))
    out, _ = run_shots(machine, tmp_path, frames, cuts, window=5)
    assert calls == [(9, 0, 5), (12, 9, 5)]
    other = frames.clone()
    other[9:] = torch.flip(other[9:], dims=[0])
    changed, _ = run_shots(machine, tmp_path, other, cuts, window=5)
    assert changed[:9].numpy().tobytes() == out[:9].numpy().tobytes()
    assert changed[9:].numpy().tobytes() != out[9:].numpy().tobytes()


def test_one_output_array_is_allocated_for_the_clip_and_the_shots_written_into_it(
    machine: StandIn, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    given: list[tuple[int, ...]] = []
    real = clip.upscale

    def spy(models, shot_frames, target, seed, **kwargs):  # type: ignore[no-untyped-def]
        given.append(kwargs["output"].shape)
        return real(models, shot_frames, target, seed, **kwargs)

    monkeypatch.setattr(clip, "upscale", spy)
    out, _ = run_shots(machine, tmp_path, pictures(21), (Shot(0, 9), Shot(9, 21)), window=5)
    assert given == [(9, 48, 80, 3), (12, 48, 80, 3)] and out.shape == (21, 48, 80, 3)


def test_a_clips_bar_runs_over_all_its_shots_and_ends_full(machine: StandIn, tmp_path) -> None:
    run_shots(machine, tmp_path, pictures(21), (Shot(0, 9), Shot(9, 21)), window=5)
    (bar,) = ProgressBar.created
    assert bar.updates == sorted(bar.updates) and bar.updates[-1] == bar.total == progress.BAR
    # One bar over both shots, each said on the node.
    instance = server.PromptServer.instance
    assert instance is not None
    said = {where(text).split(progress.SEPARATOR)[0] for text, _ in instance.texts[1:-1]}
    assert said == {"shot 1/2", "shot 2/2"}


def test_the_plan_says_each_shot(
    machine: StandIn, tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    cuts = (Shot(0, 5), Shot(5, 29), Shot(29, 33))
    with caplog.at_level(logging.INFO, logger="seedvr2x_comfyui.nodes"):
        _, text = run_shots(machine, tmp_path, pictures(33), cuts, window=5)
    lines = text.splitlines()
    assert lines[0] == (
        "plan: a clip of 33 frames 40x24 to 80x48 (x2, resolution 48), in 3 shots, each"
        " upscaled on its own"
    )
    assert lines[1].startswith("shot 1: frames 0 to 4 (5 frames): ")
    assert lines[2].startswith("shot 2: frames 5 to 28 (24 frames): ")
    assert "windows of the model, the longest" in lines[2]
    assert lines[3].startswith("shot 3: frames 29 to 32 (4 frames): the whole shot in one window")
    assert "window length: at most 17 frames, as you set it" in text
    # Each shot said as it starts, a line of its own (the plan's lines are records of their own
    # too, one text: texts rebuilds it).
    assert [m for m in texts(caplog.records) if m.startswith("shot ")] == [
        "shot 1 of 3, frames 0 to 4",
        "shot 2 of 3, frames 5 to 28",
        "shot 3 of 3, frames 29 to 32",
    ]
    # One shot: the clip's first line as before, no line for the shot.
    _, one = run_shots(machine, tmp_path, pictures(5), (Shot(0, 5),))
    assert "shot 1" not in one and "the whole clip in one window of the model" in one


def test_the_plan_is_made_once_for_the_longest_shot(
    machine: StandIn, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made: list[tuple[object, ...]] = []
    real = clip.plan

    def spy(free, target, cuts, *args, **kwargs):  # type: ignore[no-untyped-def]
        made.append(tuple(cuts))
        return real(free, target, cuts, *args, **kwargs)

    monkeypatch.setattr(clip, "plan", spy)
    cuts = (Shot(0, 5), Shot(5, 18), Shot(18, 21))
    run_shots(machine, tmp_path, pictures(21), cuts, window=5)
    # Once for the clip's shots whole (the early check before the weights are read, and the plan).
    assert made == [cuts, cuts]


def test_a_cut_list_that_is_not_for_these_frames_is_refused(machine: StandIn, tmp_path) -> None:
    frames = pictures(9)
    for cuts in ((Shot(0, 5),), (Shot(0, 5), Shot(6, 9)), (Shot(0, 12),), (Shot(0, 5), Shot(4, 9))):
        with pytest.raises(words.Refused, match="cut list covers") as refused:
            run_shots(machine, tmp_path, frames, cuts)
        assert "--" not in str(refused.value)
    with pytest.raises(words.Refused, match="takes the cuts output of seedvr2x Cuts"):
        run_shots(machine, tmp_path, frames, [Shot(0, 9)])  # type: ignore[arg-type]
    assert machine.loads == []


def test_an_unwired_cut_list_is_one_shot_said_once_in_the_console(
    machine: StandIn, tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    """The cuts input is optional (the user, 2026-11-08): unwired, the clip is the library's clip
    of all its frames, bit for bit what a cut list of one shot gives, and the console says why
    once; wired, nothing of it."""
    frames = pictures(21)
    with caplog.at_level(logging.INFO, logger="seedvr2x_comfyui.nodes"):
        out, _ = upscale_video(handle(tmp_path), plans.Settings(window=5), frames, RESOLUTION, "1")
        said = [each.removeprefix(console.PREFIX) for each in caplog.messages]
        wired, _ = run_shots(machine, tmp_path, frames, (Shot(0, 21),), window=5)
    expected = (
        "no cut list: the clip is upscaled as one shot (wire seedvr2x Cuts for a clip with"
        " several shots)"
    )
    assert said.count(expected) == 1
    assert [each.removeprefix(console.PREFIX) for each in caplog.messages].count(expected) == 1
    made = planned_clip(machine, plans.Settings(window=5), (Shot(0, 21),))
    alone = clip.upscale(
        fresh(machine),
        frames.numpy(),
        TARGET,
        42,
        window=made.window,
        encode_tiles=made.encode_tiles,
        decode_tiles=made.decode_tiles,
        correct=True,
    )
    assert out.numpy().tobytes() == alone.tobytes() == wired.numpy().tobytes()


def test_a_clip_the_host_cant_hold_is_refused_before_a_model_loads_in_the_nodes_words(
    machine: StandIn, tmp_path
) -> None:
    machine.host_bytes = 1 * GIB
    with pytest.raises(words.Refused, match="This clip needs about") as refused:
        upscale_video(
            handle(tmp_path, keep=False),
            plans.Settings(),
            pictures(21),
            RESOLUTION,
            "1",
            cuts=(Shot(0, 21),),
        )
    text = str(refused.value)
    assert "kept whole" in text and "seedvr2x's command line, which streams it" in text
    assert "apps/seedvr2x" in text and "a clip of any length fits" in text
    assert text.index("command line") < text.index("close programs that use the computer")
    assert "give it fewer frames" in text and "--" not in text
    assert machine.loads == []
    # A picture says none of it.
    with pytest.raises(words.Refused, match="This run needs about") as picture_refused:
        upscale_image(handle(tmp_path), plans.Settings(), pictures(1), RESOLUTION, "1")
    assert "command line" not in str(picture_refused.value)


def test_the_plan_counts_the_input_in_what_it_says_and_not_in_the_comparison(
    machine: StandIn, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The input is ComfyUI's, held before the node reads the RAM available: the text says that
    it is out of that figure, and the plan compares its own with the figure as read."""
    given: list[int | None] = []
    real = clip.plan

    def spy(free, target, cuts, pins, host, **kwargs):  # type: ignore[no-untyped-def]
        given.append(host)
        return real(free, target, cuts, pins, host, **kwargs)

    monkeypatch.setattr(clip, "plan", spy)
    _, text = run_shots(machine, tmp_path, pictures(21), (Shot(0, 21),))
    # The plan before the weights are read has no host figure; the one after has it as read.
    assert given == [None, machine.host_bytes]
    line = next(each for each in text.splitlines() if each.startswith("RAM: "))
    assert "for the clip's output, kept whole," in line and "for seedvr2x's own work" in line
    assert "the input's frames, 0.00 GiB, are not in that" in line
    assert "ComfyUI has them in memory already, so they are out of the RAM available" in line


def test_the_nodes_load_with_one_line_naming_the_command_line_and_no_run_says_it(
    machine: StandIn, tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    import importlib

    import comfyui

    with caplog.at_level(logging.INFO, logger="seedvr2x_comfyui"):
        importlib.reload(comfyui)
        said = [m for m in caplog.messages if "command line" in m]
        assert len(said) == 1
        assert "apps/seedvr2x" in said[0] and "README" in said[0] and "http" not in said[0]
        assert "shot by shot" in said[0] and "resumes after a stop" in said[0]
        assert "streams the clip" in said[0] and "film" not in said[0]
        caplog.clear()
        run_shots(machine, tmp_path, pictures(5), (Shot(0, 5),))
        upscale_image(handle(tmp_path), plans.Settings(), pictures(1), RESOLUTION, "1")
    assert not [m for m in caplog.messages if "command line" in m]


FOLDED = (-214223524269425, 2**64 - 1)


@pytest.mark.parametrize("given", FOLDED)
def test_a_folded_seed_gives_the_picture_of_the_seed_used(
    machine: StandIn, tmp_path, given: int
) -> None:
    batch = pictures(2)
    out, plan = upscale_image(handle(tmp_path), plans.Settings(seed=given), batch, RESOLUTION, "1")
    used = given % plans.SEEDS
    same, _ = upscale_image(handle(tmp_path), plans.Settings(seed=used), batch, RESOLUTION, "1")
    assert out.numpy().tobytes() == same.numpy().tobytes()
    assert f"seed: {used} ({given} given, taken modulo 4,293,967,296)" in plan


@pytest.mark.parametrize("given", FOLDED)
def test_a_folded_seed_gives_the_clip_of_the_seed_used(
    machine: StandIn, tmp_path, given: int
) -> None:
    frames = pictures(9)
    cuts = (Shot(0, 4), Shot(4, 9))
    used = given % plans.SEEDS
    out, plan = run_shots(machine, tmp_path, frames, cuts, seed=given, window=5)
    same, _ = run_shots(machine, tmp_path, frames, cuts, seed=used, window=5)
    assert out.numpy().tobytes() == same.numpy().tobytes()
    assert f"seed: {used} ({given} given, taken modulo 4,293,967,296)" in plan

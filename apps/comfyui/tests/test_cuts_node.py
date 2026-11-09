"""seedvr2x Cuts over the untrained detector's weights (the file written as core's test_detector
does) and over scripted probabilities: the cut list it gives Upscale Video, the text of the cuts
and the possible cuts, the detector's file the one seedvr2x Models provided (the handle's folder's
alone, refused where the folder lacks it, or Hugging Face's cache's) found, checked and given
back; the cut list as Preview as Text shows it, a line a cut, timed at the frame rate wired.
"""

import json
import logging
import math
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

import comfy.model_management as management
import numpy as np
import numpy.typing as npt
import pytest
import server
import torch
from comfyui import console, models, nodes, plans, words
from standins import (
    PICTURE,
    RESOLUTION,
    StandIn,
    execute,
    handle,
    nodes_console,
    upscale_video,
)
from test_detector import untrained_weights

from seedvr2x import detector as shot_detector
from seedvr2x import pull, shotcuts, waiting
from seedvr2x.errors import ModelDamaged, ModelUnreachable
from seedvr2x.shots import Shot
from seedvr2x.weights import AGAIN


def unprefixed(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The console's lines without the "[seedvr2x] " every one of seedvr2x's begins with
    (console.py, held by tests/test_console.py): the words are what these tests hold."""
    return [line.removeprefix(console.PREFIX) for line in caplog.messages]


def clip_images(count: int, channels: int = 3) -> torch.Tensor:
    rng = np.random.default_rng(5)
    return torch.from_numpy(rng.random((count, *PICTURE, channels), dtype=np.float32))


def run_cuts(
    images: torch.Tensor,
    threshold: float = shotcuts.THRESHOLD,
    model_dir: str = "",
    unique_id: str | None = "7",
) -> tuple[tuple[Shot, ...], str]:
    """The node run with a Models handle on `model_dir` ("": Hugging Face's cache): its cut list's
    shots and the console's text."""
    found, said = run_cuts_value(images, threshold, model_dir, unique_id)
    return found.shots, said


def run_cuts_value(
    images: torch.Tensor,
    threshold: float = shotcuts.THRESHOLD,
    model_dir: str = "",
    unique_id: str | None = "7",
    frame_rate: object = None,
) -> tuple[plans.CutList, str]:
    """run_cuts' run, its output as the node gives it, plans.CutList; `frame_rate` given where it
    is not None, as a link gives it (unwired, ComfyUI gives no value at all)."""
    rate = {} if frame_rate is None else {"frame_rate": frame_rate}
    with nodes_console() as said:
        out = execute(
            nodes.Cuts,
            unique_id,
            images=images,
            models=models.Handle(plans.DEFAULT_DIT, model_dir),
            threshold=threshold,
            **rate,
        )
    # The one text the node says whole: the cuts.
    assert isinstance(out[0], plans.CutList), out[0]
    return out[0], said.messages[0] if said.messages else ""


@pytest.fixture
def weights_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A folder holding the untrained detector's weights as transnetv2.safetensors, which are no
    pin's (the pin's check passed, as the other tests' stand-in files are)."""
    folder = tmp_path / "models"
    folder.mkdir()
    untrained_weights(folder)
    monkeypatch.setattr(pull, "check_pin", lambda *args, **kwargs: None)
    return folder


def scripted(
    monkeypatch: pytest.MonkeyPatch, values: dict[int, float], count: int
) -> list[shot_detector.Detector]:
    """The detector's probabilities replaced by these (0.0 elsewhere), the detectors made
    recorded."""
    made: list[shot_detector.Detector] = []
    real = shot_detector.Detector

    class Recording(real):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            made.append(self)

    probabilities = np.zeros(count, np.float32)
    for frame, value in values.items():
        probabilities[frame] = value
    monkeypatch.setattr(shot_detector, "Detector", Recording)
    monkeypatch.setattr(
        shot_detector, "probabilities", lambda found, images, progress=None: probabilities
    )
    return made


def test_the_cuts_are_the_shots_between_the_detections_and_the_text_says_them(
    monkeypatch: pytest.MonkeyPatch, weights_dir: Path
) -> None:
    # float32's 0.3 reaches 0.3 (shotcuts compares exactly), 0.299 does not: a possible cut.
    made = scripted(monkeypatch, {29: 0.9, 59: 0.3, 79: 0.2, 80: 0.15, 99: 0.99}, 100)
    cuts, text = run_cuts(clip_images(100), model_dir=str(weights_dir))
    # The cut is the frame after the peak; a peak on the last frame gives none.
    assert cuts == (Shot(0, 30), Shot(30, 60), Shot(60, 100))
    assert text.splitlines() == [
        "100 frames, 2 cuts found at threshold 0.3: 3 shots",
        "cut at frame 30 (probability 0.899)",
        "cut at frame 60 (probability 0.300)",
        "1 possible cut, not in the list: scenes the detector found less sure than 0.3 (from 0.1),"
        " to check by eye; a lower threshold makes them cuts",
        "possible cut at frame 80 (probability 0.200)",
    ]
    # Given back: the detector is closed once the node is over.
    assert len(made) == 1 and made[0]._model is None  # pyright: ignore[reportPrivateUsage]


def test_a_clip_with_no_cut_is_one_shot_and_says_no_possible_cuts(
    monkeypatch: pytest.MonkeyPatch, weights_dir: Path
) -> None:
    scripted(monkeypatch, {}, 40)
    cuts, text = run_cuts(clip_images(40), model_dir=str(weights_dir))
    assert cuts == (Shot(0, 40),)
    assert text.splitlines() == [
        "40 frames, 0 cuts found at threshold 0.3: 1 shot",
        "no possible cuts: nothing between 0.1 and 0.3",
    ]


@pytest.mark.parametrize("threshold", [0.1, 0.05])
def test_at_a_threshold_of_0_1_or_under_every_scene_change_is_a_cut(
    monkeypatch: pytest.MonkeyPatch, weights_dir: Path, threshold: float
) -> None:
    scripted(monkeypatch, {}, 40)
    _, text = run_cuts(clip_images(40), threshold=threshold, model_dir=str(weights_dir))
    assert text.splitlines()[1] == (
        "no possible cuts: at a threshold of 0.1 or under, every scene change the detector sees is"
        " a cut"
    )


def test_the_threshold_decides_which_detections_are_cuts(
    monkeypatch: pytest.MonkeyPatch, weights_dir: Path
) -> None:
    scripted(monkeypatch, {9: 0.4, 19: 0.8}, 30)
    low, _ = run_cuts(clip_images(30), threshold=0.3, model_dir=str(weights_dir))
    high, text = run_cuts(clip_images(30), threshold=0.5, model_dir=str(weights_dir))
    assert low == (Shot(0, 10), Shot(10, 20), Shot(20, 30)) and high == (Shot(0, 20), Shot(20, 30))
    assert "possible cut at frame 10 (probability 0.400)" in text


def test_the_untrained_detector_runs_on_the_frames_and_its_cuts_are_the_librarys(
    weights_dir: Path,
) -> None:
    """Nothing scripted: the node's path is shrink, the windows and shotcuts, as the library
    draws the cuts from the probabilities of the same frames."""
    images = clip_images(137)
    found = shot_detector.Detector(weights_dir / "transnetv2.safetensors", torch.device("cpu"))
    with found:
        probabilities: npt.NDArray[np.float32] = shot_detector.probabilities(found, images)
    for threshold in (0.3, 0.5, 0.9):
        cuts, text = run_cuts(images, threshold=threshold, model_dir=str(weights_dir))
        wanted = shotcuts.detected(probabilities, threshold)
        starts = [0, *(each.frame for each in wanted)]
        assert cuts == tuple(Shot(a, b) for a, b in zip(starts, [*starts[1:], 137], strict=True)), (
            threshold
        )
        assert text == plans.cuts_text(
            137, threshold, wanted, shotcuts.possible(probabilities, threshold)
        )
        assert sum(each.frames for each in cuts) == 137


def test_the_node_shows_one_line_and_the_alpha_is_left_out(
    monkeypatch: pytest.MonkeyPatch, weights_dir: Path
) -> None:
    seen: list[tuple[int, ...]] = []

    def probabilities(
        found: object, images: torch.Tensor, progress: object = None
    ) -> npt.NDArray[np.float32]:
        seen.append(tuple(images.shape))
        return np.zeros(len(images), np.float32)

    monkeypatch.setattr(shot_detector, "probabilities", probabilities)
    _, text = run_cuts(clip_images(12, channels=4), model_dir=str(weights_dir), unique_id="9")
    instance = server.PromptServer.instance
    # The one short line (the cuts themselves are in the console): shots and possible cuts.
    assert instance is not None and instance.texts == [("1 shot \u00b7 no possible cut", "9")]
    assert text == (
        "12 frames, 0 cuts found at threshold 0.3: 1 shot\n"
        "no possible cuts: nothing between 0.1 and 0.3"
    )
    assert seen == [(12, *PICTURE, 3)]


def test_the_detectors_file_is_fetched_from_hugging_faces_cache_and_checked_against_its_pin(
    monkeypatch: pytest.MonkeyPatch, weights_dir: Path
) -> None:
    asked: list[tuple[str, Path, bool]] = []
    scripted(monkeypatch, {}, 6)
    monkeypatch.setattr(
        pull,
        "fetch",
        lambda pins, download=None, progress=None: {name: weights_dir / name for name in pins},
    )

    def check(name: str, path: Path, pin: pull.Pin, hashes, cache, role=None, **kwargs):  # type: ignore[no-untyped-def]
        asked.append((name, path, pin is pull.PINNED[pull.DETECTOR] and role == "detector"))

    monkeypatch.setattr(pull, "check_pin", check)
    run_cuts(clip_images(6))
    assert asked == [(pull.DETECTOR, weights_dir / pull.DETECTOR, True)]


def test_a_file_that_is_not_the_pin_is_refused_in_the_nodes_words_for_models_folder(
    tmp_path: Path,
) -> None:
    untrained_weights(tmp_path)
    with pytest.raises(words.Refused, match="is not seedvr2x's own file") as refused:
        run_cuts(clip_images(6), model_dir=str(tmp_path))
    text = str(refused.value)
    assert "hf download" not in text and "--" not in text
    # The folder is seedvr2x Models' input, not this node's (which has none).
    assert 'unwire the "model folder" input of seedvr2x Models to use' in text
    assert "seedvr2x Cuts" not in text
    # Sizes with their separators, as the nodes write them elsewhere.
    assert re.search(
        r"size \d{1,3}(,\d{3})* bytes, where seedvr2x's has \d{1,3}(,\d{3})+ bytes", text
    )


def test_a_file_that_cannot_be_had_or_is_damaged_names_seedvr2x_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = pull.DETECTOR

    def offline(pins: Any, download: Any = None, progress: Any = None) -> Any:
        raise ModelUnreachable("offline", file=name, why="offline")

    monkeypatch.setattr(pull, "fetch", offline)
    with pytest.raises(words.Refused, match="Hugging Face's offline mode is on") as refused:
        run_cuts(clip_images(6))
    text = str(refused.value)
    assert 'wire a folder that has them to the "model folder" input of seedvr2x Models' in text
    assert "seedvr2x Cuts" not in text

    # A damaged file in the folder wired to Models: the way out unwires that input.
    folder = tmp_path / "models"
    folder.mkdir()
    untrained_weights(folder)

    def damaged(*args: Any, **kwargs: Any) -> None:
        raise ModelDamaged(f"{folder / name}: cut short; {AGAIN}", file=name, role="detector")

    monkeypatch.setattr(pull, "check_pin", damaged)
    with pytest.raises(words.Refused) as refused:
        run_cuts(clip_images(6), model_dir=str(folder))
    text = str(refused.value)
    assert 'unwire the "model folder" input of seedvr2x Models' in text
    assert "seedvr2x Cuts" not in text and "--" not in text

    # The cache's file damaged: the way out deletes it from the cache, no folder named.
    monkeypatch.setattr(
        pull, "fetch", lambda pins, download=None, progress=None: {n: folder / n for n in pins}
    )
    with pytest.raises(words.Refused) as refused:
        run_cuts(clip_images(6))
    assert "Hugging Face's cache" in str(refused.value) and "wire" not in str(refused.value)


def shown_on(node: str) -> list[str]:
    instance = server.PromptServer.instance
    assert instance is not None
    return [text for text, each in instance.texts if each == node]


def test_the_node_shows_the_scan_then_the_shots_and_the_possible_cuts_in_one_line(
    monkeypatch: pytest.MonkeyPatch, weights_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    probabilities = np.zeros(100, np.float32)
    probabilities[[29, 59, 79, 80, 99]] = [0.9, 0.3, 0.2, 0.15, 0.99]

    def scan(found: object, images: torch.Tensor, progress: Any = None) -> npt.NDArray[np.float32]:
        for done in (50, 100):
            progress(done, 100)
        return probabilities

    monkeypatch.setattr(shot_detector, "probabilities", scan)
    with caplog.at_level(logging.INFO, logger="seedvr2x_comfyui"):
        run_cuts(clip_images(100), model_dir=str(weights_dir), unique_id="3")
    assert shown_on("3") == [
        "scanning 50/100 frames",
        "scanning 100/100 frames",
        "3 shots \u00b7 1 possible cut below 0.30",
    ]
    # The console has the cuts with their probabilities and the possible cut, without a lead.
    lines = [each.removeprefix(console.PREFIX) for m in caplog.messages for each in m.splitlines()]
    assert "cut at frame 30 (probability 0.899)" in lines
    assert "possible cut at frame 80 (probability 0.200)" in lines
    assert "100 frames, 2 cuts found at threshold 0.3: 3 shots" in lines


def test_a_refusal_shows_its_line_and_is_said_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        pull,
        "check_pin",
        lambda *a, **k: (_ for _ in ()).throw(
            ModelDamaged(f"x: cut short; {AGAIN}", file=pull.DETECTOR, role="detector")
        ),
    )
    monkeypatch.setattr(
        pull, "fetch", lambda pins, download=None, progress=None: {n: tmp_path / n for n in pins}
    )
    with caplog.at_level(logging.INFO, logger="seedvr2x_comfyui"):
        with pytest.raises(words.Refused):
            run_cuts(clip_images(6), unique_id="5")
    assert shown_on("5") == ["refused: see the error"]
    refused = [line for line in unprefixed(caplog) if line.startswith("refused: ")]
    assert len(refused) == 1 and "delete it from Hugging Face's cache" in refused[0]


def test_a_value_that_is_no_handle_is_refused_naming_the_node(weights_dir: Path) -> None:
    with pytest.raises(TypeError, match="seedvr2x Cuts takes the output of seedvr2x Models"):
        execute(nodes.Cuts, "7", images=clip_images(3), models="a folder", threshold=0.3)


def test_images_that_are_no_clip_are_refused(weights_dir: Path) -> None:
    with pytest.raises(words.Refused, match="empty"):
        run_cuts(torch.zeros(0, *PICTURE, 3), model_dir=str(weights_dir))
    with pytest.raises(words.Refused, match="2 channels"):
        run_cuts(torch.zeros(3, *PICTURE, 2), model_dir=str(weights_dir))
    for bad in (0.0, 1.5, -1.0):
        with pytest.raises(words.Refused, match="above 0 and at most 1"):
            run_cuts(clip_images(3), threshold=bad, model_dir=str(weights_dir))


def test_the_cut_list_it_gives_goes_into_upscale_video(
    machine: StandIn, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, weights_dir: Path
) -> None:
    """Cuts into Upscale Video: the shots found are the shots upscaled, covering the clip."""
    scripted(monkeypatch, {8: 0.9, 20: 0.7}, 33)
    frames = clip_images(33)
    cuts, _ = run_cuts_value(frames, model_dir=str(weights_dir))
    assert cuts.shots == (Shot(0, 9), Shot(9, 21), Shot(21, 33))
    # The node's own value, as ComfyUI hands it on: taken as the tuple of its shots was.
    _, plan = upscale_video(
        handle(tmp_path), plans.Settings(window=5), frames, RESOLUTION, cuts=cuts
    )
    assert "in 3 shots, each upscaled on its own" in plan
    assert "shot 2: frames 9 to 20 (12 frames): " in plan


def test_the_cut_lists_text_is_the_cuts_a_line_each_and_no_json_so_preview_any_shows_it(
    monkeypatch: pytest.MonkeyPatch, weights_dir: Path
) -> None:
    # ComfyUI's Preview Any (v0.44.0) shows json.dumps of a value and, where that raises, str():
    # the cut list is no JSON value, and its str() is the cuts, a line each, numbered from 1, by
    # the frame each starts on, counted from 0 (the user's fourth test, 2026-11-08: not the
    # console's text, which keeps the probabilities and the possible cuts).
    values = {8: 0.9, 20: 0.7, 25: 0.2}
    scripted(monkeypatch, values, 33)
    found, said = run_cuts_value(clip_images(33), model_dir=str(weights_dir))
    assert str(found) == found.text == "Cut #1 - frame 9\nCut #2 - frame 21"
    probabilities = np.zeros(33, np.float32)
    for frame, value in values.items():
        probabilities[frame] = value
    threshold = shotcuts.THRESHOLD
    assert said == plans.cuts_text(
        33,
        threshold,
        shotcuts.detected(probabilities, threshold),
        shotcuts.possible(probabilities, threshold),
    )
    assert "cut at frame 9 (probability" in said and "possible cut at frame 26" in said
    assert "probability" not in found.text and "26" not in found.text
    with pytest.raises(TypeError):
        json.dumps(found)
    tooltip = nodes.Cuts.GET_SCHEMA().outputs[0].tooltip or ""
    assert "Preview as Text node (PreviewAny)" in tooltip
    assert "Cut #1 - 0:00:01.440 (frame 36)" in tooltip


def test_with_the_frame_rate_each_cut_has_its_time_as_the_command_line_writes_it(
    monkeypatch: pytest.MonkeyPatch, weights_dir: Path
) -> None:
    # 25 fps: frame 36 at 1.44 s, the user's form; the time the command line's (shotcuts.timecode,
    # H:MM:SS.mmm rounded up to the millisecond), the frame's on a constant rate from frame 0.
    scripted(monkeypatch, {35: 0.9, 2249: 0.8}, 2300)
    found, _ = run_cuts_value(clip_images(2300), model_dir=str(weights_dir), frame_rate=25.0)
    assert found.text.splitlines() == [
        "Cut #1 - 0:00:01.440 (frame 36)",
        "Cut #2 - 0:01:30.000 (frame 2250)",
    ]
    # The float Get Video Components gives for 24000/1001 is read as 24000/1001 again: frame 36 at
    # 1.5015 s exactly, rounded up to 0:00:01.502, frame 1 at 41.708 ms to 0:00:00.042.
    scripted(monkeypatch, {0: 0.9, 35: 0.9}, 40)
    found, _ = run_cuts_value(
        clip_images(40), model_dir=str(weights_dir), frame_rate=float(Fraction(24000, 1001))
    )
    assert found.text.splitlines() == [
        "Cut #1 - 0:00:00.042 (frame 1)",
        "Cut #2 - 0:00:01.502 (frame 36)",
    ]
    # A whole number from a link is a rate too.
    scripted(monkeypatch, {35: 0.9}, 40)
    found, _ = run_cuts_value(clip_images(40), model_dir=str(weights_dir), frame_rate=25)
    assert found.text == "Cut #1 - 0:00:01.440 (frame 36)"


def test_a_clip_with_no_cut_says_so_in_one_line(
    monkeypatch: pytest.MonkeyPatch, weights_dir: Path
) -> None:
    for rate in (None, 25.0):
        scripted(monkeypatch, {39: 0.9}, 40)  # a peak on the last frame cuts nothing
        found, _ = run_cuts_value(clip_images(40), model_dir=str(weights_dir), frame_rate=rate)
        assert found.shots == (Shot(0, 40),)
        assert str(found) == "no cut: the clip is one shot"


@pytest.mark.parametrize("rate", [0.0, -25.0, math.nan, math.inf, -math.inf, True, "25"])
def test_a_frame_rate_that_is_no_positive_finite_number_is_refused_before_the_detector_runs(
    monkeypatch: pytest.MonkeyPatch, weights_dir: Path, rate: object
) -> None:
    made = scripted(monkeypatch, {8: 0.9}, 20)
    with pytest.raises(words.Refused) as refused:
        run_cuts_value(clip_images(20), model_dir=str(weights_dir), frame_rate=rate)
    message = str(refused.value)
    assert message.startswith(
        f"frame rate {rate!r}: the clip's frames per second, a finite number above 0"
    )
    assert "Get Video Components" in message and "fps" in message and "--" not in message
    assert not made


def test_a_rate_given_as_a_float_is_the_fraction_it_stands_for() -> None:
    for rate in (
        Fraction(25),
        Fraction(24000, 1001),
        Fraction(30000, 1001),
        Fraction(60000, 1001),
        Fraction(120000, 1001),
        Fraction(2997, 125),
        Fraction(15360, 641),
    ):
        assert plans.rate_of(float(rate)) == rate, rate
    # Under 5e-7 fps the nearest fraction of a small denominator is 0: the float itself, so that a
    # positive rate never divides by zero (the review, 2026-11-09).
    assert plans.rate_of(1e-9) == Fraction(1e-9) > 0
    assert plans.cut_lines([shotcuts.Cut(1)], 1e-9).startswith("Cut #1 - 277777")


def test_the_node_has_the_cuts_output_and_a_hidden_id() -> None:
    schema = nodes.Cuts.GET_SCHEMA()
    # The cuts and the possible cuts are said in the console: no text output.
    assert [(each.id, each.io_type) for each in schema.outputs] == [("cuts", "SEEDVR2X_CUTS")]
    inputs: dict[str, Any] = {each.id: each for each in schema.inputs}
    # No mode (wired, it detects) and no folder (seedvr2x Models' alone): the detector's file
    # comes with the models handle, the upscale nodes' own type.
    assert list(inputs) == ["images", "models", "threshold", "frame_rate"]
    assert inputs["models"].io_type == "SEEDVR2X_MODELS" and not inputs["models"].optional
    assert "shot detector" in inputs["models"].tooltip
    assert inputs["threshold"].default == shotcuts.THRESHOLD == 0.3
    assert inputs["threshold"].io_type == "FLOAT" and inputs["threshold"].round is False
    assert "one shot" not in inputs["threshold"].tooltip
    # The frame rate an optional socket and no widget, as Models' folder: a FLOAT from a link,
    # Get Video Components' fps.
    rate = inputs["frame_rate"]
    assert rate.io_type == "FLOAT" and rate.optional and rate.force_input and rate.default is None
    assert rate.display_name == "frame rate" and "Get Video Components" in rate.tooltip
    found = nodes.Cuts.INPUT_TYPES()
    assert list(found["required"]) == ["images", "models", "threshold"]
    assert list(found["optional"]) == ["frame_rate"]
    assert found["optional"]["frame_rate"][1]["forceInput"] is True
    assert "seedvr2x Models" in schema.description and "one shot" not in schema.description


def test_a_failed_card_refuses_the_cuts_node_and_a_cuda_error_marks_it(
    weights_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A CUDA error the detector meets marks the card, the error kept and worded, torch's cache
    # left alone; the node is then refused at its start, as the upscale nodes are
    # (words.card_failed; seedvr2x/waiting.py's fail).
    def failing(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(shot_detector, "probabilities", failing)
    emptied = len(management.empties)
    with pytest.raises(words.Refused, match="the graphics card failed: CUDA error: an illegal"):
        run_cuts(clip_images(20), model_dir=str(weights_dir))
    assert waiting.failed(0) is not None and len(management.empties) == emptied
    assert len(models._failures) == 1  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(words.Refused, match="failed during an earlier seedvr2x run"):
        run_cuts(clip_images(20), model_dir=str(weights_dir))
    models.reset()


def no_cache(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, ...]]:
    """Hugging Face's cache made to fail any test that reads it or downloads into it, the calls
    recorded: a model folder is the user's choice of neither (the user, 2026-11-08)."""
    asked: list[tuple[str, ...]] = []

    def never(*args: Any, **kwargs: Any) -> Any:
        asked.append(tuple(str(each) for each in args))
        raise AssertionError("Hugging Face's cache asked with a model folder given")

    for name in ("fetch", "cached", "cache", "resolve"):
        monkeypatch.setattr(pull, name, never)
    return asked


def test_the_detector_is_the_folders_alone_where_one_is_given_else_the_caches(
    weights_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripted(monkeypatch, {}, 12)
    pinned: list[Path] = []
    monkeypatch.setattr(pull, "check_pin", lambda name, path, *args, **kwargs: pinned.append(path))
    # A folder that has it: its file, the cache neither read nor written.
    with monkeypatch.context() as inside:
        asked = no_cache(inside)
        run_cuts(clip_images(12), model_dir=str(weights_dir))
        assert asked == [] and pinned == [weights_dir / pull.DETECTOR]
    # A folder without it (Models' folder holds the DiT and the VAE): refused, naming the file,
    # the folder and where to get it; nothing fetched and the cache not read.
    empty = tmp_path / "empty"
    empty.mkdir()
    pinned.clear()
    with monkeypatch.context() as inside:
        asked = no_cache(inside)
        with pytest.raises(words.Refused) as refused:
            run_cuts(clip_images(12), model_dir=str(empty))
        assert asked == [] and pinned == []
    text = str(refused.value)
    assert text.startswith(f"{pull.DETECTOR}, the shot detector's file that seedvr2x Cuts reads,")
    assert f"is not in the model folder wired to seedvr2x Models ({empty})" in text
    assert pull.url(pull.DETECTOR) in text
    assert 'or unwire the "model folder" input of seedvr2x Models to use' in text
    assert "--" not in text
    assert list(empty.iterdir()) == [], "nothing is written into a user's folder"
    # No folder: the cache's.
    fetched: list[tuple[str, ...]] = []

    def fetch(pins: Any, download: Any = None, progress: Any = None) -> Any:
        fetched.append(tuple(pins))
        return {name: weights_dir / name for name in pins}

    monkeypatch.setattr(pull, "fetch", fetch)
    run_cuts(clip_images(12))
    assert fetched == [(pull.DETECTOR,)]


def test_a_file_the_cache_lost_since_models_ran_is_fetched_saying_so_first_on_a_bar(
    weights_dir: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    scripted(monkeypatch, {}, 12)
    kept = {"in": True}
    monkeypatch.setattr(pull, "cached", lambda name: weights_dir / name if kept["in"] else None)

    def fetch(pins: Any, download: Any = None, progress: Any = None) -> Any:
        # As pull.fetch: a file in the cache is told nothing; one missing, its bytes.
        for name in pins:
            if not kept["in"]:
                progress(name, 0, 100)
                progress(name, 100, 100)
        return {name: weights_dir / name for name in pins}

    monkeypatch.setattr(pull, "fetch", fetch)
    caplog.set_level(logging.INFO, logger="seedvr2x_comfyui")
    run_cuts(clip_images(12), unique_id="8")
    said = "shot detector is not in Hugging Face's cache: downloading it"
    assert said not in unprefixed(caplog), "found in the cache: nothing said"
    assert not any(text.startswith("downloading") for text in shown_on("8"))
    kept["in"] = False
    caplog.clear()
    run_cuts(clip_images(12), unique_id="9")
    lines = unprefixed(caplog)
    assert lines.index(said) < lines.index("downloading the shot detector")
    # The second run's own bar and lines: the download, then the scan.
    assert any(text.startswith("downloading 0.0/0.0 MB") for text in shown_on("9"))


def test_a_detectors_file_gone_from_the_folder_since_models_ran_is_refused_never_fetched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # seedvr2x Models ran on a folder that had the file, and ComfyUI keeps its output between
    # prompts: a file taken from the folder since is refused at Cuts' run, never taken from the
    # cache (the user, 2026-11-08), whatever the cache holds.
    scripted(monkeypatch, {}, 12)
    folder = tmp_path / "a folder of models"
    folder.mkdir()
    untrained_weights(folder)
    monkeypatch.setattr(pull, "check_pin", lambda *args, **kwargs: None)
    handle = models.Handle(plans.DEFAULT_DIT, str(folder))
    asked = no_cache(monkeypatch)
    execute(nodes.Cuts, "7", images=clip_images(12), models=handle, threshold=shotcuts.THRESHOLD)
    (folder / pull.DETECTOR).unlink()
    caplog.set_level(logging.DEBUG, logger="seedvr2x_comfyui")
    with pytest.raises(words.Refused, match="is not in the model folder wired to seedvr2x Models"):
        execute(
            nodes.Cuts, "7", images=clip_images(12), models=handle, threshold=shotcuts.THRESHOLD
        )
    assert asked == []
    said = unprefixed(caplog)
    assert not any("cache's is read" in line or "downloading" in line for line in said)


def test_comfyuis_cancel_is_the_check_of_the_detectors_waits_its_close_among_them(
    monkeypatch: pytest.MonkeyPatch, weights_dir: Path
) -> None:
    # The detector's close waits on the card through the poll (detector.Detector.close), which a
    # check ends: the Cuts node gives it ComfyUI's Cancel, as the upscale nodes give theirs, and
    # the check before is back after the node.
    scripted(monkeypatch, {}, 12)
    checks: list[object] = []
    close = shot_detector.Detector.close

    def recorded(self: shot_detector.Detector) -> None:
        checks.append(getattr(waiting._named, "check", None))  # pyright: ignore[reportPrivateUsage]
        close(self)

    monkeypatch.setattr(shot_detector.Detector, "close", recorded)
    run_cuts(clip_images(12), model_dir=str(weights_dir))
    assert checks == [management.throw_exception_if_processing_interrupted]
    assert getattr(waiting._named, "check", None) is None  # pyright: ignore[reportPrivateUsage]

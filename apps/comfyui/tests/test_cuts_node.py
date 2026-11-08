"""seedvr2x Cuts over the untrained detector's weights (the file written as core's test_detector
does) and over scripted probabilities: the cut list it gives Upscale Video, the text of the cuts
and the possible cuts, the one shot mode, the detector's file found, checked and given back."""

import logging
import re
from pathlib import Path
from typing import Any

import comfy.model_management as management
import numpy as np
import numpy.typing as npt
import pytest
import server
import torch
from comfyui import console, models, nodes, plans, words
from standins import PICTURE, RESOLUTION, StandIn, execute, handle, upscale_video
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
    mode: str = "detect",
    threshold: float = shotcuts.THRESHOLD,
    model_dir: str = "",
    unique_id: str | None = "7",
) -> tuple[tuple[Shot, ...], str]:
    out = execute(
        nodes.Cuts,
        unique_id,
        images=images,
        mode=mode,
        threshold=threshold,
        model_dir=model_dir,
    )
    return out[0], out[1]


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


def test_one_shot_is_the_whole_clip_and_runs_no_detector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the detector ran in one shot mode")

    monkeypatch.setattr(shot_detector, "Detector", refuse)
    monkeypatch.setattr(pull, "fetch", refuse)
    cuts, text = run_cuts(clip_images(21), mode="one shot")
    assert cuts == (Shot(0, 21),)
    assert text == "seedvr2x Cuts: one shot, the whole clip of 21 frames; no detection was run"


def test_the_cuts_are_the_shots_between_the_detections_and_the_text_says_them(
    monkeypatch: pytest.MonkeyPatch, weights_dir: Path
) -> None:
    # float32's 0.3 reaches 0.3 (shotcuts compares exactly), 0.299 does not: a possible cut.
    made = scripted(monkeypatch, {29: 0.9, 59: 0.3, 79: 0.2, 80: 0.15, 99: 0.99}, 100)
    cuts, text = run_cuts(clip_images(100), model_dir=str(weights_dir))
    # The cut is the frame after the peak; a peak on the last frame gives none.
    assert cuts == (Shot(0, 30), Shot(30, 60), Shot(60, 100))
    assert text.splitlines() == [
        "seedvr2x Cuts: 100 frames, 2 cuts found at threshold 0.3: 3 shots",
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
        "seedvr2x Cuts: 40 frames, 0 cuts found at threshold 0.3: 1 shot",
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


def test_the_text_is_said_on_the_node_and_the_alpha_is_left_out(
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
    assert instance is not None and instance.texts == [(text, "9")]
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


def test_a_file_that_is_not_in_the_folder_is_refused_in_the_nodes_words(tmp_path: Path) -> None:
    with pytest.raises(words.Refused, match="is not in the folder wired to the") as refused:
        run_cuts(clip_images(6), model_dir=str(tmp_path))
    assert "transnetv2.safetensors" in str(refused.value) and "--" not in str(refused.value)
    # The folder is the Cuts node's own input, not the Models node's.
    assert 'the "model folder" input of seedvr2x Cuts' in str(refused.value)
    assert "seedvr2x Models" not in str(refused.value)
    assert "unwire the input to use Hugging Face's cache" in str(refused.value)


def test_a_file_that_is_not_the_pin_is_refused_in_the_nodes_words(
    tmp_path: Path,
) -> None:
    untrained_weights(tmp_path)
    with pytest.raises(words.Refused, match="is not seedvr2x's own file") as refused:
        run_cuts(clip_images(6), model_dir=str(tmp_path))
    text = str(refused.value)
    assert "hf download" not in text and "--" not in text
    assert 'unwire the "model folder" input of seedvr2x Cuts' in text
    assert "seedvr2x Models" not in text
    # Sizes with their separators, as the nodes write them elsewhere.
    assert re.search(
        r"size \d{1,3}(,\d{3})* bytes, where seedvr2x's has \d{1,3}(,\d{3})+ bytes", text
    )


def test_a_file_that_cannot_be_had_or_is_damaged_names_the_cuts_node_and_its_own_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = pull.DETECTOR

    def offline(pins: Any, download: Any = None, progress: Any = None) -> Any:
        raise ModelUnreachable("offline", file=name, why="offline")

    monkeypatch.setattr(pull, "fetch", offline)
    with pytest.raises(words.Refused, match="Hugging Face's offline mode is on") as refused:
        run_cuts(clip_images(6))
    text = str(refused.value)
    assert 'wire a folder that has them to the "model folder" input of seedvr2x Cuts' in text
    assert "seedvr2x Models" not in text

    # A damaged file in a folder wired to the node: the way out unwires the input, on this node.
    folder = tmp_path / "models"
    folder.mkdir()
    untrained_weights(folder)

    def damaged(*args: Any, **kwargs: Any) -> None:
        raise ModelDamaged(f"{folder / name}: cut short; {AGAIN}", file=name, role="detector")

    monkeypatch.setattr(pull, "check_pin", damaged)
    with pytest.raises(words.Refused) as refused:
        run_cuts(clip_images(6), model_dir=str(folder))
    text = str(refused.value)
    assert 'unwire the "model folder" input of seedvr2x Cuts' in text
    assert "seedvr2x Models" not in text and "--" not in text


def test_images_that_are_no_clip_are_refused(weights_dir: Path) -> None:
    with pytest.raises(words.Refused, match="empty"):
        run_cuts(torch.zeros(0, *PICTURE, 3), model_dir=str(weights_dir))
    with pytest.raises(words.Refused, match="2 channels"):
        run_cuts(torch.zeros(3, *PICTURE, 2), model_dir=str(weights_dir))
    with pytest.raises(words.Refused, match="mode 'sometimes'"):
        run_cuts(clip_images(3), mode="sometimes", model_dir=str(weights_dir))
    for bad in (0.0, 1.5, -1.0):
        with pytest.raises(words.Refused, match="above 0 and at most 1"):
            run_cuts(clip_images(3), threshold=bad, model_dir=str(weights_dir))


def test_the_cut_list_it_gives_goes_into_upscale_video(
    machine: StandIn, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, weights_dir: Path
) -> None:
    """Cuts into Upscale Video: the shots found are the shots upscaled, covering the clip."""
    scripted(monkeypatch, {8: 0.9, 20: 0.7}, 33)
    frames = clip_images(33)
    cuts, _ = run_cuts(frames, model_dir=str(weights_dir))
    assert cuts == (Shot(0, 9), Shot(9, 21), Shot(21, 33))
    _, plan = upscale_video(
        handle(tmp_path), plans.Settings(window=5), frames, RESOLUTION, cuts=cuts
    )
    assert "in 3 shots, each upscaled on its own" in plan
    assert "shot 2: frames 9 to 20 (12 frames): " in plan


def test_the_node_has_the_cuts_and_text_outputs_and_a_hidden_id() -> None:
    schema = nodes.Cuts.GET_SCHEMA()
    assert [(each.id, each.io_type) for each in schema.outputs] == [
        ("cuts", "SEEDVR2X_CUTS"),
        ("text", "STRING"),
    ]
    inputs: dict[str, Any] = {each.id: each for each in schema.inputs}
    assert list(inputs) == ["images", "mode", "threshold", "model_dir"]
    assert inputs["mode"].options == ["detect", "one shot"] and inputs["mode"].default == "detect"
    assert inputs["threshold"].default == shotcuts.THRESHOLD == 0.3
    assert inputs["threshold"].io_type == "FLOAT" and inputs["threshold"].round is False
    folder = inputs["model_dir"]
    assert folder.display_name == "model folder" and folder.io_type == "STRING"
    # An optional input and no widget: it asks for nothing to be filled in (the user, 2026-11-08).
    assert folder.optional and folder.force_input and folder.as_dict()["forceInput"] is True
    assert folder.tooltip.startswith("Optional.") and pull.DETECTOR in folder.tooltip
    assert "Hugging Face's cache" in folder.tooltip and "Not connected" in folder.tooltip
    found = nodes.Cuts.INPUT_TYPES()
    assert "model_dir" in found["optional"] and "model_dir" not in found["required"]
    assert found["optional"]["model_dir"][1]["forceInput"] is True
    assert "default" not in found["optional"]["model_dir"][1], "no value stands in for a folder"


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


def test_the_folder_input_is_optional_and_an_unwired_one_is_the_cache(
    weights_dir: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    scripted(monkeypatch, {}, 12)
    fetched: list[object] = []

    def fetch(pins: Any, download: Any = None, progress: Any = None) -> Any:
        fetched.append(tuple(pins))
        return {name: weights_dir / name for name in pins}

    monkeypatch.setattr(pull, "fetch", fetch)
    # Not given at all, as the executor leaves an unwired optional input.
    node_out = execute(nodes.Cuts, "7", images=clip_images(12), mode="detect", threshold=0.3)
    assert node_out[0] == (Shot(0, 12),)
    assert fetched == [(pull.DETECTOR,)], "Hugging Face's cache, fetched by its pin"
    # Quotes (Windows' "Copy as path") and spaces around a folder are taken off.
    fetched.clear()
    run_cuts(clip_images(12), model_dir=f' "{weights_dir}" ')
    assert fetched == [], "a folder: nothing fetched"
    # A value empty once stripped is an unwired input, said.
    caplog.set_level(logging.INFO, logger="seedvr2x_comfyui")
    run_cuts(clip_images(12), model_dir=' "" ')
    assert fetched == [(pull.DETECTOR,)]
    assert "the model folder given is empty: Hugging Face's cache is used" in unprefixed(caplog)


def test_a_value_that_is_no_folder_is_refused_before_anything_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def never(*args: object, **kwargs: object) -> None:
        raise AssertionError("went on past a folder that does not exist")

    monkeypatch.setattr(shot_detector, "Detector", never)
    monkeypatch.setattr(pull, "fetch", never)
    for value in (str(tmp_path / "nowhere"), str(tmp_path / "file.txt")):
        (tmp_path / "file.txt").write_text("x")
        with pytest.raises(words.Refused, match="is not a folder") as refused:
            run_cuts(clip_images(6), model_dir=value)
        assert 'the "model folder" input of seedvr2x Cuts' in str(refused.value)
        assert "or unwire the input to use Hugging Face's cache" in str(refused.value)


def test_the_console_says_where_the_detectors_file_is_and_how_large_it_is(
    weights_dir: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    scripted(monkeypatch, {}, 12)
    caplog.set_level(logging.INFO, logger="seedvr2x_comfyui")
    size = plans.size_text(pull.PINNED[pull.DETECTOR].size)
    assert size == "30.5 MB"
    run_cuts(clip_images(12), model_dir=str(weights_dir))
    # A folder's file by its own size (here the untrained weights' file the test wrote).
    own = plans.size_text((weights_dir / pull.DETECTOR).stat().st_size)
    assert f"shot detector, {own}, in the model folder: {weights_dir}" in unprefixed(caplog)
    # From the cache: found there, then downloaded into it where the cache lacked it.
    cache = Path("/the/cache")
    monkeypatch.setattr(pull, "cache", lambda: cache)
    here = {"in": True}
    monkeypatch.setattr(pull, "cached", lambda name: weights_dir / name if here["in"] else None)
    monkeypatch.setattr(
        pull,
        "fetch",
        lambda pins, download=None, progress=None: {name: weights_dir / name for name in pins},
    )
    caplog.clear()
    run_cuts(clip_images(12))
    assert f"shot detector, {size}, in Hugging Face's cache: {cache}; HF_HOME moves it" in (
        unprefixed(caplog)
    )
    here["in"] = False
    caplog.clear()
    run_cuts(clip_images(12))
    assert (
        f"shot detector, {size}, downloaded into Hugging Face's cache: {cache}; HF_HOME moves it"
        in (unprefixed(caplog))
    )

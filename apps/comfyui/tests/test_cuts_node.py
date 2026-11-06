"""seedvr2x Cuts over the untrained detector's weights (the file written as core's test_detector
does) and over scripted probabilities: the cut list it gives Upscale Video, the text of the cuts
and the possible cuts, the one shot mode, the detector's file found, checked and given back."""

import re
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import server
import torch
from comfyui import nodes, plans, words
from standins import PICTURE, RESOLUTION, StandIn, execute, handle, upscale_video
from test_detector import untrained_weights

from seedvr2x import detector as shot_detector
from seedvr2x import pull, shotcuts
from seedvr2x.errors import ModelDamaged, ModelUnreachable
from seedvr2x.shots import Shot
from seedvr2x.weights import AGAIN


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
    with pytest.raises(words.Refused, match="is not in the folder set as model folder") as refused:
        run_cuts(clip_images(6), model_dir=str(tmp_path))
    assert "transnetv2.safetensors" in str(refused.value) and "--" not in str(refused.value)
    # The folder is the Cuts node's own, not the Models node's.
    assert "in seedvr2x Cuts" in str(refused.value) and "seedvr2x Models" not in str(refused.value)


def test_a_file_that_is_not_the_pin_is_refused_in_the_nodes_words(
    tmp_path: Path,
) -> None:
    untrained_weights(tmp_path)
    with pytest.raises(words.Refused, match="is not seedvr2x's own file") as refused:
        run_cuts(clip_images(6), model_dir=str(tmp_path))
    text = str(refused.value)
    assert "hf download" not in text and "--" not in text
    assert "empty in seedvr2x Cuts" in text and "seedvr2x Models" not in text
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
    assert "folder in seedvr2x Cuts" in text and "seedvr2x Models" not in text

    # A damaged file in a folder set on the node: the way out leaves the folder empty, on this node.
    folder = tmp_path / "models"
    folder.mkdir()
    untrained_weights(folder)

    def damaged(*args: Any, **kwargs: Any) -> None:
        raise ModelDamaged(f"{folder / name}: cut short; {AGAIN}", file=name, role="detector")

    monkeypatch.setattr(pull, "check_pin", damaged)
    with pytest.raises(words.Refused) as refused:
        run_cuts(clip_images(6), model_dir=str(folder))
    text = str(refused.value)
    assert "empty in seedvr2x Cuts" in text and "seedvr2x Models" not in text
    assert "--" not in text


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
    assert inputs["model_dir"].display_name == "model folder"

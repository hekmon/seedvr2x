"""The shot detector on decoded frames held in memory (seedvr2x/detector.py's shrink and
probabilities; DESIGN.md, Repository and apps, seedvr2x Cuts), on the CPU: ComfyUI's IMAGE batch,
(T, H, W, 3) float32 in [0, 1], shrunk to TransNetV2's 48x27 in torch with no ffmpeg.

shrink: shapes and dtype; the codes of a constant frame exact, values outside [0, 1] clipped; a
frame against a hand computation (the filter written here in float64 from its definition, edges
renormalised as torch's antialiased resize does); the same bits whatever the frames around it (a
frame alone, and in a batch longer than the chunk it is shrunk in); every refusal. probabilities:
a clip equal, bit for bit, to the frames shrunk and pushed to a Detector at once, on the untrained
model's weights (test_detector.py's) and again on TransNetV2's where SEEDVR2X_MODEL_DIR has them,
where a clip of four shots is cut at its shot changes. How the shrink compares with the official
extraction's (ffmpeg's scaler) on real clips is measured in shrink's docstring, not held here."""

import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import torch
from test_detector import untrained_weights

from seedvr2x import pull
from seedvr2x.detector import HEIGHT, SHRINK_CHUNK, WIDTH, Detector, probabilities, shrink

CPU = torch.device("cpu")


def random_images(count: int, height: int, width: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand((count, height, width, 3), generator=generator)


def test_shape_dtype_and_nothing() -> None:
    codes = shrink(random_images(3, 108, 192, 0))
    assert codes.shape == (3, HEIGHT, WIDTH, 3) and codes.dtype == np.uint8
    empty = shrink(torch.zeros((0, 108, 192, 3)))
    assert empty.shape == (0, HEIGHT, WIDTH, 3) and empty.dtype == np.uint8


@pytest.mark.parametrize("size", [(27, 48), (54, 96), (1080, 1920), (480, 854), (100, 40)])
def test_a_constant_frame_keeps_its_codes(size: tuple[int, int]) -> None:
    # Every weight sums to 1, so a frame of one value stays that value, at every size, an odd
    # one among them: 255 * value rounded to the nearest code (values off a tie, which float32's
    # arithmetic could round either way).
    values = torch.tensor([0.0, 1.0, 100.3 / 255, 0.2, 1 / 255, 200.7 / 255])
    images = values[:, None, None, None].expand(6, size[0], size[1], 3).contiguous()
    expected = [0, 255, 100, 51, 1, 201]
    codes = shrink(images)
    for frame, code in enumerate(expected):
        assert np.all(codes[frame] == code), (frame, code)


def test_values_outside_the_range_are_clipped() -> None:
    images = torch.stack([torch.full((54, 96, 3), 3.0), torch.full((54, 96, 3), -2.0)])
    codes = shrink(images)
    assert np.all(codes[0] == 255) and np.all(codes[1] == 0)


def keys_matrix(n_in: int, n_out: int) -> npt.NDArray[np.float64]:
    """The (n_out, n_in) matrix of a downscale by an antialiased bicubic, from its definition in
    float64: Keys' cubic with a = -0.5 (Catmull-Rom) stretched by the ratio, centres aligned,
    taps off the edge dropped and the rest renormalised."""
    ratio = n_in / n_out

    def cubic(x: float) -> float:
        x = abs(x)
        if x < 1:
            return 1.5 * x**3 - 2.5 * x**2 + 1
        if x < 2:
            return -0.5 * x**3 + 2.5 * x**2 - 4 * x + 2
        return 0.0

    matrix = np.zeros((n_out, n_in))
    for out in range(n_out):
        centre = (out + 0.5) * ratio
        for tap in range(n_in):
            matrix[out, tap] = cubic((tap + 0.5 - centre) / ratio)
        matrix[out] /= matrix[out].sum()
    return matrix


def test_a_frame_against_a_hand_computation() -> None:
    # A 108x192 frame (a ratio of 4), smooth with a hard edge, shrunk by the matrices above in
    # float64, rounded to the nearest code: the same codes but where float32's arithmetic (1e-7
    # off) falls on the other side of a tie (one code off, a few in a thousand at most).
    rows, columns = np.mgrid[0:108, 0:192]
    image = np.stack(
        [
            0.5 + 0.5 * np.sin(rows / 9.0) * np.cos(columns / 13.0),
            (columns > 100).astype(np.float64) * 0.75 + 0.12,
            (rows + columns) / 300.0,
        ],
        axis=-1,
    )
    expected = np.einsum("oh,hwc,pw->opc", keys_matrix(108, HEIGHT), image, keys_matrix(192, WIDTH))
    expected = np.clip(np.round(expected * 255), 0, 255)
    codes = shrink(torch.from_numpy(image.astype(np.float32))[None])[0].astype(np.float64)
    difference = np.abs(codes - expected)
    assert difference.max() <= 1
    assert (difference > 0).mean() < 0.01


def test_the_same_bits_whatever_the_frames_around_it() -> None:
    # Shrunk a chunk at a time: a frame's codes don't depend on the frames beside it nor on its
    # place in the chunk. Odd count past two chunks.
    count = 2 * SHRINK_CHUNK + 3
    images = random_images(count, 54, 96, 1)
    together = shrink(images)
    alone = np.concatenate([shrink(images[i : i + 1]) for i in range(count)])
    assert together.tobytes() == alone.tobytes()


@pytest.mark.parametrize(
    ("images", "match"),
    [
        (torch.zeros((54, 96, 3)), "where the shot detector takes"),
        (torch.zeros((2, 54, 96, 4)), "where the shot detector takes"),
        (torch.zeros((2, 3, 54, 96)), "where the shot detector takes"),
        (torch.zeros((2, 54, 96, 3), dtype=torch.float64), "float64, where"),
        (torch.zeros((2, 54, 96, 3), dtype=torch.uint8), "uint8, where"),
        (torch.zeros((2, 0, 96, 3)), "no pixel"),
        (torch.zeros((2, 54, 0, 3)), "no pixel"),
    ],
)
def test_images_refused(images: torch.Tensor, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        shrink(images)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_value_that_is_no_number_is_refused_with_its_frames(bad: float) -> None:
    images = random_images(SHRINK_CHUNK + 2, 54, 96, 2)
    images[SHRINK_CHUNK + 1, 3, 4, 1] = bad
    with pytest.raises(ValueError, match=f"frames {SHRINK_CHUNK} to {SHRINK_CHUNK + 1}"):
        shrink(images)


@pytest.fixture(scope="module")
def untrained(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return untrained_weights(tmp_path_factory.mktemp("untrained_images"))


MODELS = os.environ.get("SEEDVR2X_MODEL_DIR")


@pytest.fixture(scope="module")
def trained() -> Path:
    """TransNetV2's weights in SEEDVR2X_MODEL_DIR, checked by their pin; skipped without."""
    if not MODELS or not (Path(MODELS) / pull.DETECTOR).is_file():
        pytest.skip(f"needs SEEDVR2X_MODEL_DIR holding {pull.DETECTOR}")
    path = Path(MODELS) / pull.DETECTOR
    pull.check_pin(pull.DETECTOR, path, pull.PINNED[pull.DETECTOR], pull.Hashes(), None)
    return path


def test_probabilities_are_the_shrunk_frames_pushed(untrained: Path) -> None:
    # 437 frames, more than eight windows' steps and a partial last window: one probability per
    # frame, float32 in [0, 1], the same bits as the frames shrunk whole and pushed at once.
    images = random_images(437, 54, 96, 3)
    with Detector(untrained, CPU) as found:
        got = probabilities(found, images)
    assert got.shape == (437,) and got.dtype == np.float32
    assert got.min() >= 0 and got.max() <= 1
    with Detector(untrained, CPU) as again:
        expected = np.concatenate([again.push(shrink(images)), again.finish()])
    assert got.tobytes() == expected.tobytes()


def test_probabilities_of_no_frame_and_of_a_used_detector(untrained: Path) -> None:
    with Detector(untrained, CPU) as found:
        assert probabilities(found, torch.zeros((0, 54, 96, 3))).shape == (0,)
    with Detector(untrained, CPU) as used:
        used.push(shrink(random_images(2, 54, 96, 4)))
        with pytest.raises(ValueError, match="one clip to a detector"):
            probabilities(used, random_images(2, 54, 96, 4))


def test_a_clip_of_four_shots_is_cut_at_its_shot_changes(trained: Path) -> None:
    # Four shots of different moving patterns, 60 frames each, at 108x192: TransNetV2's own
    # weights peak on the last frame of each shot before a change, above 0.5, and stay under 0.1
    # inside the shots.
    rows, columns = torch.meshgrid(torch.arange(108.0), torch.arange(192.0), indexing="ij")
    frames: list[torch.Tensor] = []
    for shot in range(4):
        generator = torch.Generator().manual_seed(shot)
        phase = torch.rand(3, generator=generator) * 6.28
        scale = 6.0 + 5.0 * shot
        for step in range(60):
            channels = [
                0.5
                + 0.5 * torch.sin((rows + 0.4 * step) / scale * (c + 1) + phase[c] + columns / 25.0)
                for c in range(3)
            ]
            frames.append(torch.stack(channels, dim=-1))
    with Detector(trained, CPU) as found:
        got = probabilities(found, torch.stack(frames))
    peaks = {int(i) for i in np.flatnonzero(got >= 0.5)}
    assert peaks == {59, 119, 179}, (sorted(peaks), got[[58, 59, 60, 118, 119, 120, 178, 179, 180]])
    inside = np.delete(got, [58, 59, 60, 118, 119, 120, 178, 179, 180])
    assert inside.max() < 0.1

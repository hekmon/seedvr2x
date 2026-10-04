"""The shot pipeline's parts on the CPU: windows, mixing weights, padding, the encoder's input
and lab's reference, streamed encode and decode."""

from itertools import pairwise
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
import torch
from omegaconf import OmegaConf

from seedvr2x.runtime import model
from seedvr2x.runtime.shot import (
    encoder_inputs,
    input_slices,
    mix_weights,
    pad_4n1,
    padded_length,
    reference_inputs,
    window_layout,
)

# The models as far as the encoder's input and lab's reference use them: the CPU and the VAE's
# slicing (model.encode_slices).
SLICING = cast(
    model.Models,
    SimpleNamespace(
        device=torch.device("cpu"),
        runner=SimpleNamespace(vae=SimpleNamespace(use_slicing=True, slicing_sample_min_size=4)),
    ),
)


def test_one_window_when_the_shot_fits() -> None:
    assert window_layout(21, 21) == [(0, 21)]
    assert window_layout(5, 6) == [(0, 5)]


def test_study_layout() -> None:
    # Clip B of the stitching study: 81 frames, 21 latents, windows of 6 sharing 2.
    assert window_layout(21, 6) == [(0, 6), (4, 10), (8, 14), (12, 18), (16, 21)]


@pytest.mark.parametrize("window", [5, 6, 7, 11, 30])
def test_balanced_layouts(window: int) -> None:
    for latents in range(window + 1, 200):
        layout = window_layout(latents, window)
        lengths = [e - s for s, e in layout]
        assert layout[0][0] == 0 and layout[-1][1] == latents
        assert max(lengths) <= window and max(lengths) - min(lengths) <= 1
        assert lengths == sorted(lengths, reverse=True)
        assert all(e0 - s == 2 for (_, e0), (s, _) in pairwise(layout))
        # no latent in three windows
        assert all(s2 >= e0 for (_, e0), (s2, _) in zip(layout, layout[2:], strict=False))
        # as few windows as the cap allows: one fewer would need a longer one
        fewer = len(layout) - 1
        assert fewer * window - (fewer - 1) * 2 < latents


def test_window_too_short() -> None:
    with pytest.raises(ValueError):
        window_layout(21, 4)


def test_cosine_weights_exact_in_bfloat16() -> None:
    assert mix_weights(2) == pytest.approx([0.75, 0.25])
    as_bf16 = torch.tensor(mix_weights(2)).to(torch.bfloat16).float().tolist()
    assert as_bf16 == [0.75, 0.25]


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (1, [0]),
        (2, [0, 1, 1, 1, 1]),
        (3, [0, 1, 2, 1, 0]),
        (5, [0, 1, 2, 3, 4]),
        (10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6]),
    ],
)
def test_padding_as_numz(count: int, expected: list[int]) -> None:
    frames = torch.arange(count).view(count, 1, 1, 1).expand(count, 2, 2, 3)
    assert pad_4n1(frames)[:, 0, 0, 0].tolist() == expected


class Reads:
    """A shot's frames, read as upscale_shot reads them: n at a time, each read recorded."""

    def __init__(self, frames: npt.NDArray[np.float32]) -> None:
        self.frames, self.read, self.counts = frames, 0, list[int]()

    def __call__(self, count: int) -> npt.NDArray[np.float32]:
        self.counts.append(count)
        self.read += count
        return self.frames[self.read - count : self.read]


def vae_slices(frames: int) -> list[int]:
    return model.encode_slices(SLICING, frames)


@pytest.mark.parametrize("count", range(1, 22))
def test_streamed_input_is_the_padded_shot(count: int) -> None:
    # Read in the VAE's slices (5 frames, then 4), padded at the end as pad_4n1 pads the whole.
    frames = np.random.default_rng(count).random((count, 2, 3, 3), dtype=np.float32)
    reads = Reads(frames)
    sizes = vae_slices(padded_length(count))
    slices = list(input_slices(reads, count, sizes))
    assert [s.shape[0] for s in slices] == sizes
    assert sum(reads.counts) == count and all(n <= 5 for n in reads.counts)
    assert torch.equal(torch.cat(slices), pad_4n1(model.to_input(frames)))


def test_vae_slices() -> None:
    assert vae_slices(1) == [1]
    assert vae_slices(5) == [5]
    assert vae_slices(9) == [5, 4]
    assert vae_slices(45) == [5, *[4] * 10]


# Frames resized 2x, as milestone 1's are, to a height the transform pads (24 to 32).
SOURCE, TARGET = (12, 16), (24, 32)


@pytest.mark.parametrize("count", [1, 6, 13])
def test_reference_is_the_transform_in_float32(count: int) -> None:
    # lab's reference (DESIGN.md, Colour correction, Numerics): the frames read, through the
    # encoder's transform in float32, with no float16 or bfloat16 step, which random values would
    # show; slice by slice as the encoder's input.
    frames = np.random.default_rng(count).random((count, *SOURCE, 3), dtype=np.float32)
    # From a copy, before the reference is built: on the CPU, the reference's slices hold the
    # frames read without a copy, so a step altering them in place would alter both sides alike.
    video = pad_4n1(torch.from_numpy(frames.copy())).permute(0, 3, 1, 2)
    expected = model.input_transform(TARGET)(video)
    slices = list(reference_inputs(SLICING, Reads(frames), count, TARGET))
    assert [s.shape[1] for s in slices] == vae_slices(padded_length(count))
    assert all(s.dtype == torch.float32 for s in slices)
    assert torch.equal(torch.cat(slices, dim=1), expected)


@pytest.mark.parametrize("count", [1, 6, 13])
def test_encoder_input_is_numz(count: int) -> None:
    # The encoder's input stays numz's (milestone 1): float16 frames, cast to bfloat16 before the
    # transform (generation_phases.py:380-413).
    frames = np.random.default_rng(count).random((count, *SOURCE, 3), dtype=np.float32)
    slices = list(encoder_inputs(SLICING, Reads(frames), count, TARGET))
    assert [s.shape[1] for s in slices] == vae_slices(padded_length(count))
    assert all(s.dtype == model.COMPUTE_DTYPE == torch.bfloat16 for s in slices)
    video = pad_4n1(model.to_input(frames)).permute(0, 3, 1, 2).to(torch.bfloat16)
    assert torch.equal(torch.cat(slices, dim=1), model.input_transform(TARGET)(video))


@pytest.fixture(scope="module")
def cpu_models() -> model.Models:
    """A VAE built from the configs with random weights, on the CPU, in a runner."""
    from seedvr2x.vendor.common.config import create_object, load_config
    from seedvr2x.vendor.core.infer import VideoDiffusionInfer

    config: Any = load_config(str(model.VENDOR / "configs_7b" / "main.yaml"))
    vae_yaml = model.VENDOR / "models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml"
    config.vae.model = OmegaConf.merge(config.vae.model, load_config(str(vae_yaml)))
    config.vae.dtype = "float32"
    torch.manual_seed(0)
    vae_config: Any = config.vae.model
    vae = create_object(vae_config).eval()
    vae.set_causal_slicing(**config.vae.slicing)
    runner: Any = VideoDiffusionInfer(config, cast(Any, SimpleNamespace(log=lambda *a, **k: None)))
    runner.vae = vae
    return cast(model.Models, SimpleNamespace(runner=runner))


@pytest.mark.parametrize("count", [1, 2, 6, 13])
def test_streamed_encode_is_the_one_pass_encode(cpu_models: model.Models, count: int) -> None:
    # The shot read in slices, each prepared and encoded as it comes, must give the latent of the
    # whole shot prepared and encoded at once, bit for bit, in the same memory layout (the noise
    # drawn from it depends on it).
    frames = np.random.default_rng(count).random((count, 32, 48, 3), dtype=np.float32)
    transform = model.input_transform((32, 48))
    padded = padded_length(count)
    with torch.no_grad():
        video = pad_4n1(model.to_input(frames)).permute(0, 3, 1, 2).to(torch.float32)
        one_pass = model.encode(cpu_models, transform(video))
        slices = (
            transform(part.permute(0, 3, 1, 2).to(torch.float32))
            for part in input_slices(Reads(frames), count, model.encode_slices(cpu_models, padded))
        )
        streamed = model.encode_stream(cpu_models, slices, padded)
    assert streamed.shape == one_pass.shape == ((padded - 1) // 4 + 1, 4, 6, 16)
    assert torch.equal(streamed, one_pass)
    assert streamed.stride() == one_pass.stride()


def test_streamed_decode_is_the_one_pass_decode(cpu_models: model.Models) -> None:
    # The streamed decode must give the one-pass decode's frames bit for bit, through the same
    # slices and causal memory.
    models, runner = cpu_models, cast(Any, cpu_models).runner
    latent = torch.randn(4, 4, 6, 16)  # (T', h, w, C): 13 frames of 32x48
    with torch.no_grad():
        one_pass = runner.vae_decode([latent])[0]
        streamed = torch.cat(list(model.decode_stream(models, latent)), dim=1)
    assert streamed.shape == one_pass.shape == (3, 13, 32, 48)
    assert torch.equal(streamed, one_pass)

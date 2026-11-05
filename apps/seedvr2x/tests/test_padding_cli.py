"""The frames' padding before the model, as the shot's pass applies it (seedvr2x/shot_pass.py;
DESIGN.md, Pipeline step 0), on the CPU: the encoder's slices padded as asked, the decode trimmed,
split's reference the picture. The padding itself, held bit for bit to
research/scripts/numerics_patch.py's Pad, is test_padding.py's (core/tests/)."""

import math
from collections.abc import Iterable
from types import SimpleNamespace
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest
import torch
from torchvision.transforms import InterpolationMode, Normalize
from torchvision.transforms import functional as TVF

from seedvr2x import model
from seedvr2x.frames import (
    output_size,
    padded_length,
    padded_size,
)
from seedvr2x.shot_pass import _frames, decode_shot, encode_shot, encoder_inputs, reference_inputs


@pytest.mark.parametrize("target", [(24, 32), (30, 37)])
def test_encoder_slices_padded_as_asked(
    monkeypatch: pytest.MonkeyPatch, target: tuple[int, int]
) -> None:
    # Each slice the encoder takes, from encoder_inputs and through encode_shot:
    # frames.padded_size. (24, 32) is padded to (48, 32); (30, 37) to (64, 64).
    from test_shot_pass import SLICING, Reads

    count = 6
    frames = np.random.default_rng(target[1]).random((count, 15, 20, 3), dtype=np.float32)
    sizes = model.encode_slices(SLICING, padded_length(count))  # 5 frames, then 4
    encoded: list[tuple[int, int]] = []

    def encode_stream(
        models: model.Models, inputs: Iterable[torch.Tensor], length: int
    ) -> torch.Tensor:
        encoded.extend((x.shape[-2], x.shape[-1]) for x in inputs)
        return torch.zeros(1)

    monkeypatch.setattr(model, "encode_stream", encode_stream)
    size = padded_size(target)
    given = encoder_inputs(SLICING, Reads(frames), count, target)
    assert [(x.shape[-2], x.shape[-1]) for x in given] == [size] * len(sizes)
    encoded.clear()
    encode_shot(SLICING, Reads(frames), count, target, 0)
    assert encoded == [size] * len(sizes)


@pytest.mark.parametrize(
    "target", [(1080, 1920), (720, 1280), (2160, 3840), (1060, 1912), (1081, 1913), (17, 16)]
)
def test_decode_trimmed(monkeypatch: pytest.MonkeyPatch, target: tuple[int, int]) -> None:
    # The decode's frames at the padded size, NaN in the padding, which the crop must drop whole
    # (a NaN written stops the run: shot_pass.finite) and no more: the output's size, the target's
    # even sides as before, and the picture as decoded.
    padded = padded_size(target)
    decoded = torch.full((3, 2, *padded), math.nan, dtype=torch.bfloat16)
    generator = torch.Generator().manual_seed(target[0])
    picture = torch.rand(3, 2, *target, generator=generator) * 2.4 - 1.2  # clamped beyond 1
    decoded[..., : target[0], : target[1]] = picture.bfloat16()
    height, width = output_size(target)
    kept = decoded[..., :height, :width].permute(1, 2, 3, 0)
    expected = kept.clamp(-1, 1).mul(0.5).add(0.5).float().numpy()
    # Two slices of one frame, as the VAE's slices come.
    slices = [decoded[:, :1], decoded[:, 1:]]
    monkeypatch.setattr(model, "decode_stream", lambda models, latent: iter(slices))
    written: list[npt.NDArray[np.float32]] = []
    models = cast(model.Models, SimpleNamespace(device=torch.device("cpu")))
    decode_shot(models, torch.zeros(1), 2, target, written.append)
    frames = np.concatenate(written)
    assert frames.shape == (2, height, width, 3)
    assert np.array_equal(frames, expected)


@pytest.mark.parametrize("target", [(24, 32), (17, 16), (30, 37), (541, 961)])
def test_reference_is_the_picture(target: tuple[int, int]) -> None:
    # split's reference, cropped as the decode's frames are (_frames): the picture alone, whatever
    # padding the transform made: the frames resized, clamped and normalised.
    from test_shot_pass import SLICING, Reads

    count = 6
    rng = np.random.default_rng(target[1])
    frames = rng.random((count, 15, 20, 3), dtype=np.float32)
    height, width = output_size(target)
    take = _frames(reference_inputs(SLICING, Reads(frames), count, target), height, width)
    reference = take(count)
    video = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2)
    size = list(target)
    resized = TVF.resize(video, size, InterpolationMode.BICUBIC, antialias=True).clamp(0.0, 1.0)
    picture = Normalize(0.5, 0.5)(resized)[..., :height, :width]
    assert reference.shape == (count, 3, height, width)
    assert torch.equal(reference, picture)
    out = model.input_transform(target)(video)
    assert torch.equal(out.permute(1, 0, 2, 3)[..., :height, :width], picture)

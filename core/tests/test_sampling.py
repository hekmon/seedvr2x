"""The DiT's sampling of latents (sampling.py; DESIGN.md, Pipeline step 2) on the CPU: the noise
drawn in system memory from the latent's layout, held to its digest, not a card's, and the
log line saying where it was drawn; `draw`, the same noise alone; and `moved`, a tensor on a
device with its strides kept. The windows over a shot, which call these, are test_shot_pass.py's."""

import logging
import re
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from stand_ins import DEVICE, channel_major, cpu_models  # noqa: F401

from seedvr2x import model, sampling

# The noise of a shot of 21 frames at 1080p, 6 latents of 138 x 240 positions (1104 x 1920
# padded) in the layout an encode gives them, from seed 42: the SHA-256 of the bfloat16 values'
# bits, in the latent's index order (test_the_draw_is_held_to_its_digest).
DRAW_DIGEST = "90d8e6bcc5b463868eed13bf2f09493f998a771f4fc347d3e0f95492065f5f1e"


def test_the_draw_is_held_to_its_digest(cpu_models: model.Models) -> None:  # noqa: F811
    # The noise drawn on the CPU is the same on every card and every machine (DESIGN.md,
    # Pipeline step 2): held here to its digest for a fixed seed and shape, so that another
    # generator, another seeding or memory layout fails, and to its generator's being its own.
    # The layout is the encode's:
    # channel-major, a permuted view, which a 1-frame encode through the VAE gives here, its
    # strides those of a latent so built. The noise keeps the latent's shape, dtype and strides.
    import hashlib

    # One frame: a shot of 1 is not padded to 4n + 1 (shot.py, pad_4n1).
    frames = np.zeros((1, 32, 48, 3), dtype=np.float32)
    with torch.no_grad():
        video = model.to_input(frames).permute(0, 3, 1, 2).to(torch.float32)
        encoded = model.encode(cpu_models, model.input_transform((32, 48))(video))

    def channel_major_of(shape: tuple[int, ...]) -> torch.Tensor:
        latents, height, width, channels = shape
        empty = torch.zeros(channels, latents, height, width, dtype=model.COMPUTE_DTYPE)
        return empty.permute(1, 2, 3, 0)

    assert encoded.stride() == channel_major_of(tuple(encoded.shape)).stride()
    latent = channel_major_of((6, 138, 240, 16))
    noise, kept = sampling.drawn(DEVICE, latent, 42)
    assert kept is latent and noise.device.type == "cpu"
    assert noise.shape == latent.shape and noise.stride() == latent.stride()
    assert noise.dtype == model.COMPUTE_DTYPE == torch.bfloat16

    def digest(noise: torch.Tensor) -> str:
        return hashlib.sha256(noise.contiguous().view(torch.int16).numpy().tobytes()).hexdigest()

    assert digest(noise) == DRAW_DIGEST
    # The draw's generator is its own: torch's global one drawn from between the seeding and
    # the draw, as anything else in the process could, moves no value.
    set_seed = model.set_seed

    def seeded_then_drawn_from(seed: int) -> None:
        set_seed(seed)
        torch.rand(3)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(model, "set_seed", seeded_then_drawn_from)
        again, _ = sampling.drawn(DEVICE, latent, 42)
    assert digest(again) == DRAW_DIGEST


def test_nothing_of_the_draw_goes_to_a_gpu(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # On a GPU the noise is drawn in system memory, nothing of the shot sent there for it, and
    # its values are those the CPU draws, said in the -v line. No GPU here: a device by its name
    # alone, the copies to it and back recorded and left out.
    gpu = cast(model.Models, SimpleNamespace(device=torch.device("cuda")))
    sent: list[str] = []

    def moving(tensor: torch.Tensor, device: torch.device, what: str) -> torch.Tensor:
        sent.append(f"{what} to {device.type}")
        return tensor

    monkeypatch.setattr(sampling, "moved", moving)
    caplog.set_level(logging.DEBUG, logger="seedvr2x.sampling")
    latent = channel_major(9)
    model.set_seed(1234)
    expected = torch.randn_like(latent, dtype=model.COMPUTE_DTYPE)
    noise, kept = sampling.drawn(gpu, latent, 1234)
    assert torch.equal(noise, expected) and kept is latent
    assert sent == ["the shot's latent to cpu"]
    assert re.fullmatch(
        r"noise drawn on the CPU for the shot's 9 latents in \d+\.\d\d s: 0\.000 GiB",
        caplog.messages[-1],
    )


def test_draw_is_the_noise_alone() -> None:
    # The calibration times the draw apart from the window (calibration.py): the noise
    # sampling.draw gives is the one drawn gives, from the same seed, and the latent is not
    # touched.
    latent = torch.zeros(16, 5, 3, 4, dtype=model.COMPUTE_DTYPE).permute(1, 2, 3, 0)
    noise, kept = sampling.drawn(DEVICE, latent, 7)
    assert kept is latent
    alone = sampling.draw(DEVICE, latent, 7)
    assert torch.equal(alone, noise) and alone.stride() == latent.stride()
    assert not torch.equal(sampling.draw(DEVICE, latent, 8), noise)


def test_moved_refuses_a_copy_laid_out_anew() -> None:
    # A copy that comes back with other strides than the tensor's (a slice along time, laid out
    # dense on its own) is refused, naming what was moved; the tensor itself is returned where
    # it is there already.
    latent = torch.randn(16, 5, 3, 4).permute(1, 2, 3, 0)
    assert sampling.moved(latent, torch.device("cpu"), "the latent") is latent

    class Relaid(torch.Tensor):
        def to(self, *args: Any, **kwargs: Any) -> torch.Tensor:
            return self.contiguous()

    with pytest.raises(RuntimeError, match=r"^the shot's latent: its copy on cpu has strides"):
        sampling.moved(latent.as_subclass(Relaid), torch.device("cpu"), "the shot's latent")

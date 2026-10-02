"""The shot pipeline's parts on the CPU: windows, mixing weights, padding, streamed decode."""

from itertools import pairwise
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from omegaconf import OmegaConf

from seedvr2x.runtime import model
from seedvr2x.runtime.shot import mix_weights, pad_4n1, window_layout


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


def test_streamed_decode_is_the_one_pass_decode() -> None:
    # A VAE built from the configs with random weights, on the CPU: the streamed decode must give
    # the one-pass decode's frames bit for bit, through the same slices and causal memory.
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
    models = cast(model.Models, SimpleNamespace(runner=runner))
    latent = torch.randn(4, 4, 6, 16)  # (T', h, w, C): 13 frames of 32x48
    with torch.no_grad():
        one_pass = runner.vae_decode([latent])[0]
        streamed = torch.cat(list(model.decode_stream(models, latent)), dim=1)
    assert streamed.shape == one_pass.shape == (3, 13, 32, 48)
    assert torch.equal(streamed, one_pass)

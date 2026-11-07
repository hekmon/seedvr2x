"""What the library's tests and the CLI's both stand in with, and neither owns: a VAE of random
weights on the CPU, a stand-in for the models' holder, a latent in the memory the VAE encode gives
it. In core/tests, on the CLI's tests' pythonpath (apps/seedvr2x/pyproject.toml); no CLI module
is imported here."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from omegaconf import OmegaConf
from torch import Tensor

from seedvr2x import model

# The stand-in VAE's channels per block: the shipped 128, 256, 512 and 512 narrowed to 32, 32,
# 64 and 64. The library reads no channel count of the VAE (its scaling factor, the latent's 16
# channels, the factors 8 and 4 and the slices of frames are the shipped ones), and neither
# width reaches a convolution's split at the tests' sizes. Two widths, as shipped, so that a
# resnet changes its channels and has its shortcut, in the encoder and in the decoder's tail
# (up_blocks[2:], test_tail_frames.py), and a norm's groups hold two channels at 64. A block's
# convolutions cost its channels squared at its resolution, 53,248 against 1,600 summed over the
# four (33 times fewer); measured on 2026-11-07, 16 threads, a 33-frame clip in 3 windows,
# corrected, took 2.4 s at the shipped widths and 0.27 s narrowed, 4.4 and 0.61 s with both
# passes tiled, and 250.6 M parameters became 4.2 M. The shipped widths stay on the CPU in
# shipped_cpu_models' two cases (test_clip_cli.py, test_still_cli.py) and test_tiling.py's.
NARROW = [32, 32, 64, 64]


@pytest.fixture(scope="module")
def cpu_models() -> model.Models:
    """A VAE built from the configs with random weights, its blocks narrowed to NARROW's
    channels, on the CPU, in a runner."""
    return vae_models(NARROW)


@pytest.fixture(scope="module")
def shipped_cpu_models() -> model.Models:
    """cpu_models at the shipped widths, the configs' own: 128, 256, 512 and 512 channels."""
    return vae_models(None)


def vae_models(widths: list[int] | None) -> model.Models:
    """A VAE built from the configs with random weights, its blocks' channels `widths` (None:
    the configs' own), on the CPU in float32, in a runner."""
    from seedvr2x.vendor.common.config import create_object, load_config
    from seedvr2x.vendor.core.infer import VideoDiffusionInfer

    config: Any = load_config(str(model.VENDOR / "configs_7b" / "main.yaml"))
    vae_yaml = model.VENDOR / "models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml"
    config.vae.model = OmegaConf.merge(config.vae.model, load_config(str(vae_yaml)))
    if widths is not None:
        config.vae.model.block_out_channels = widths
    config.vae.dtype = "float32"
    torch.manual_seed(0)
    vae_config: Any = config.vae.model
    vae = create_object(vae_config).eval()
    vae.set_causal_slicing(**config.vae.slicing)
    runner: Any = VideoDiffusionInfer(config, cast(Any, SimpleNamespace(log=lambda *a, **k: None)))
    runner.vae = vae
    return cast(model.Models, SimpleNamespace(runner=runner, device=torch.device("cpu")))


# The models' holder where only the device matters, the CPU: a copy to the device or back is the
# tensor itself (test_sampling.py, test_shot_pass.py).
DEVICE = cast(model.Models, SimpleNamespace(device=torch.device("cpu")))


def channel_major(latents: int, seed: int = 0) -> Tensor:
    """A latent (T', h, w, 16) in the memory the VAE encode gives it: channel-major, a permuted
    view (model.encode)."""
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(16, latents, 3, 5, generator=generator).to(torch.bfloat16).permute(1, 2, 3, 0)
    )

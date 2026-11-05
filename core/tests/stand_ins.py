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

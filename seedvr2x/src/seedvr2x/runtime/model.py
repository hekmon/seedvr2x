"""The seam between seedvr2x and the vendored model code.

Builds the DiT, the VAE and the diffusion runner (VideoDiffusionInfer) from the vendored configs,
loads their weights and sets them up the way numz does, so that milestone 1 reproduces numz's
output (DESIGN.md, Validation milestones). Line references are to numz at 4490bd1.

The vendored code is untyped: this module is the one place where the rest of seedvr2x calls it,
through typed functions.
"""

# pyright: basic

import ctypes
import logging
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from omegaconf import ListConfig, OmegaConf
from safetensors.torch import load_file
from torch import Tensor
from torchvision.transforms import Compose, InterpolationMode, Lambda, Normalize
from torchvision.transforms import functional as TVF

from seedvr2x.runtime import weights
from seedvr2x.runtime.job import BLACK, padding
from seedvr2x.vendor.common.config import create_object, load_config
from seedvr2x.vendor.common.seed import set_seed as vendor_set_seed
from seedvr2x.vendor.core.infer import VideoDiffusionInfer
from seedvr2x.vendor.data.image.transforms.divisible_crop import DivisiblePad
from seedvr2x.vendor.models.dit_3b import attention as attention_3b
from seedvr2x.vendor.models.dit_7b import attention as attention_7b
from seedvr2x.vendor.models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d
from seedvr2x.vendor.models.video_vae_v3.modules.types import MemoryState

logger = logging.getLogger(__name__)

VENDOR = Path(__file__).resolve().parents[1] / "vendor"

# numz's pipeline dtype: bfloat16 on any GPU computing it (src/optimization/compatibility.py:
# 684-698). The VAE runs in it, its weights cast at load, and the DiT under autocast to it.
COMPUTE_DTYPE = torch.bfloat16


class _DebugLog:
    """numz's Debug object as far as the vendored code uses it (log), sent to logging."""

    def log(self, message: str, *args: Any, **kwargs: Any) -> None:
        logger.debug("%s", message)


@dataclass
class Models:
    """The loaded models, on one device, ready to run."""

    runner: Any  # VideoDiffusionInfer, untyped, its dit and vae set by load_models
    text_pos: Tensor  # (58, 5120) bfloat16: the positive text embedding, the one used at cfg 1
    text_neg: Tensor  # (64, 5120) bfloat16
    attention: str  # flash_attn_2 or sdpa
    device: torch.device


def attention_backend() -> str:
    """flash_attn_2 when FlashAttention 2 imports, else sdpa (DESIGN.md, Vendored model code)."""
    available = attention_7b.FLASH_ATTN_2_AVAILABLE and attention_3b.FLASH_ATTN_2_AVAILABLE
    return "flash_attn_2" if available else "sdpa"


def nvidia_driver() -> str | None:
    """The NVIDIA driver's version, as nvidia-smi says it, read from NVML, the driver's own
    library, as torch reads the device count from it (torch/cuda/__init__.py,
    _raw_device_count_nvml); None, logged, when NVML can't say. torch has no call for it."""
    try:
        nvml = ctypes.CDLL("libnvidia-ml.so.1")
    except OSError as error:
        logger.warning("the NVIDIA driver's version is unknown: %s", error)
        return None
    status = nvml.nvmlInit_v2()
    if status != 0:
        logger.warning("the NVIDIA driver's version is unknown: NVML's initialisation: %d", status)
        return None
    try:
        version = ctypes.create_string_buffer(80)  # NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE
        status = nvml.nvmlSystemGetDriverVersion(version, ctypes.c_uint(len(version)))
        if status != 0:
            logger.warning("the NVIDIA driver's version is unknown: NVML's answer: %d", status)
            return None
        return version.value.decode()
    finally:
        nvml.nvmlShutdown()


def load_models(model_dir: Path, dit_file: str, vae_file: str, device: torch.device) -> Models:
    """Build and load the DiT in dit_file and the VAE in vae_file, from model_dir, each checked
    first (weights.check, ModelError): the DiT is built from the config of what its file is."""
    # numz goes by "7b" in the file name (src/core/model_configuration.py:718-720), so a renamed
    # file loads as what its name says; seedvr2x goes by the file's tensors (DESIGN.md, Weights).
    config_dir = weights.check(model_dir / dit_file, "dit").config
    weights.check(model_dir / vae_file, "vae")
    assert config_dir is not None, "the DiT accepted has its config"
    config: Any = load_config(str(VENDOR / config_dir / "main.yaml"))
    runner: Any = VideoDiffusionInfer(config, _DebugLog())
    OmegaConf.set_readonly(runner.config, False)

    # The VAE's architecture is a config apart, merged into the runner's (model_configuration.py:
    # 1117-1131); its dtype is the pipeline's.
    vae_config: Any = load_config(str(VENDOR / "models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml"))
    vae_config.spatial_downsample_factor = vae_config.get("spatial_downsample_factor", 8)
    vae_config.temporal_downsample_factor = vae_config.get("temporal_downsample_factor", 4)
    runner.config.vae.model = OmegaConf.merge(runner.config.vae.model, vae_config)
    runner.config.vae.dtype = str(COMPUTE_DTYPE).split(".")[-1]

    # Read back from the tree, as numz does: the node assigned is a copy, attached to the config.
    vae_model_config: Any = runner.config.vae.model
    with torch.device("meta"):
        vae = create_object(vae_model_config)
    _load_weights(vae, model_dir / vae_file, device, COMPUTE_DTYPE)
    # Out of training mode, the causal convolutions keep their memory across slices
    # (causal_inflation_lib.py:242, 268). Slicing and memory limits from the config
    # (model_configuration.py:1241-1259): fixed thresholds on tensor sizes, not on free memory.
    vae.requires_grad_(False).eval()
    vae.set_causal_slicing(**runner.config.vae.slicing)
    vae.set_memory_limit(**runner.config.vae.memory_limit)
    runner.vae = vae

    # The DiT keeps its weights' dtype (the override is commented out at model_configuration.py:
    # 1048-1052). numz leaves it in training mode, which changes nothing: its only use is a
    # gradient checkpointing stub (nadit.py:30-31).
    with torch.device("meta"):
        dit = create_object(runner.config.dit.model)
    _load_weights(dit, model_dir / dit_file, device, None)
    dit.requires_grad_(False).eval()
    attention = attention_backend()
    if attention == "sdpa":
        logger.warning("FlashAttention 2 is not installed: attention runs on PyTorch's SDPA")
    # Set on every attention module, as numz does (model_configuration.py:1204-1210).
    for module in dit.modules():
        if isinstance(
            module, attention_7b.FlashAttentionVarlen | attention_3b.FlashAttentionVarlen
        ):
            module.attention_mode = attention
            module.compute_dtype = COMPUTE_DTYPE
    runner.dit = dit

    # numz's one-step sampling: cfg 1, no rescale, one step (generation_phases.py:599-602).
    runner.config.diffusion.cfg.scale = 1.0
    runner.config.diffusion.cfg.rescale = 0.0
    runner.config.diffusion.timesteps.sampling.steps = 1
    runner.configure_diffusion(device=device, dtype=COMPUTE_DTYPE)

    text_pos = torch.load(VENDOR / "pos_emb.pt", weights_only=True).to(device, COMPUTE_DTYPE)
    text_neg = torch.load(VENDOR / "neg_emb.pt", weights_only=True).to(device, COMPUTE_DTYPE)
    return Models(runner, text_pos, text_neg, attention, device)


def _load_weights(
    model: torch.nn.Module, path: Path, device: torch.device, dtype: torch.dtype | None
) -> None:
    """Load a safetensors checkpoint into a model built on the meta device, as numz does
    (src/core/model_loader.py:547-616, 777-835): the file's dtypes, or every floating tensor cast
    to dtype; then the buffers the checkpoint doesn't hold, still on meta, filled with zeros."""
    state = load_file(path, device=str(device))
    if dtype is not None:
        state = {k: v.to(dtype) if v.is_floating_point() else v for k, v in state.items()}
    # strict=False as numz, but the mismatches are reported, where numz ignores them silently.
    result = model.load_state_dict(state, strict=False, assign=True)
    if result.missing_keys or result.unexpected_keys:
        logger.warning(
            "%s: %d keys missing from the checkpoint, %d unexpected: %s",
            path.name,
            len(result.missing_keys),
            len(result.unexpected_keys),
            (result.missing_keys + result.unexpected_keys)[:8],
        )
    del state
    for name, buffer in list(model.named_buffers()):
        if buffer.device.type == "meta":
            module_path, _, buffer_name = name.rpartition(".")
            module = model.get_submodule(module_path)
            zeros = torch.zeros_like(buffer, device=device)
            module.register_buffer(buffer_name, zeros, persistent=False)
    for name, param in model.named_parameters():
        if param.device.type == "meta":
            raise RuntimeError(f"{path.name}: no weight for {name}")


def to_input(frames: npt.NDArray[np.float32]) -> Tensor:
    """numz's input tensor from frames (T, H, W, 3) float32 in [0, 1]: float16 on the CPU
    (inference_cli.py:613-618, 697), which rounds some 8-bit values differently from a direct
    bfloat16 cast."""
    return torch.from_numpy(frames).to(torch.float16)


def input_transform(
    target: tuple[int, int], *, numz_padding: bool = False
) -> Callable[[Tensor], Tensor]:
    """numz's input preparation (src/core/generation_utils.py:72-84) of frames (T, C, H, W), but
    for its padding: resized to target, (height, width) (torchvision's bicubic, antialias on),
    values clamped to [0, 1], padded at the bottom and right (pad), normalised to [-1, 1], and
    moved to (C, T, H', W'), (H', W') = job.padded_size(target).

    The resize is NaResize's (side_resize.py: TVF.resize, bicubic, antialias on) to an explicit
    size, the display aspect's (job.target_size). For square pixels it is the size torchvision
    computes from NaResize's int, so the same call, bit for bit (tests/test_job.py).

    numz_padding is for tests only (cli.NUMZ_PADDING): numz's own padding, DivisiblePad((16, 16)),
    zeros up to multiples of 16, in which milestone 1's regression holds the output to numz's."""
    size = list(target)
    return Compose(
        [
            Lambda(lambda x: TVF.resize(x, size, InterpolationMode.BICUBIC, antialias=True)),
            Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
            DivisiblePad((16, 16)) if numz_padding else Lambda(pad),
            Normalize(0.5, 0.5),
            Lambda(lambda x: x.permute(1, 0, 2, 3)),
        ]
    )


def pad(frames: Tensor) -> Tensor:
    """frames (T, C, H, W) padded for the model (DESIGN.md, Pipeline step 0): at the bottom, the
    rows job.padding gives, reflected from the picture (torch's reflect, the edge row not repeated:
    row H + k is row H - 2 - k), then job.BLACK rows of zeros, black once normalised; at the right,
    columns the same way when any are reflected. As research/scripts/numerics_patch.py's Pad pads
    for NUM_PAD=reflect>=8+black+16, the variant measured, bit for bit: the reflection in one call,
    then the zeros in another (research/docs/numerics.md, reflect+black+16)."""
    rows, columns = padding((frames.shape[-2], frames.shape[-1]))
    reflected = torch.nn.functional.pad(frames, (0, columns, 0, rows), mode="reflect")
    black = (0, BLACK if columns else 0, 0, BLACK)
    return torch.nn.functional.pad(reflected, black, mode="constant", value=0.0)


def synchronize(device: torch.device) -> None:
    """Wait for the work queued on device, polling every 10 ms rather than blocking: Python runs
    a signal's handler between two polls, where a blocking wait holds it until the GPU is done,
    so that Ctrl-C is answered at once, not after a decode slice or a window (runtime/stop.py)."""
    if device.type != "cuda":
        return
    done = torch.cuda.Event()
    done.record()
    while not done.query():
        time.sleep(0.01)


def set_seed(seed: int) -> None:
    """Seed Python's, NumPy's and torch's global generators, as numz does (common/seed.py)."""
    vendor_set_seed(seed)


def encode(models: Models, video: Tensor) -> Tensor:
    """VAE latent of video (C, T, H, W), as the runner returns it: (T', h, w, 16) in channel-major
    memory (a permuted view), which the noise drawn from it depends on."""
    return models.runner.vae_encode([video])[0]


def encode_slices(models: Models, frames: int) -> list[int]:
    """The frames of each slice the VAE encodes a shot of `frames` frames (4n + 1) in, as its
    slicing_encode cuts it: the first frame with the next split_size (4), then split_size at a
    time; the shot in one piece when it is no longer than 1 + split_size."""
    vae = models.runner.vae
    size: int = vae.slicing_sample_min_size
    if not (vae.use_slicing and frames - 1 > size):
        return [frames]
    whole, rest = divmod(frames - 1, size)
    parts = [size] * whole + ([rest] if rest else [])
    return [1 + parts[0], *parts[1:]]


@torch.no_grad()
def encode_stream(models: Models, slices: Iterable[Tensor], frames: int) -> Tensor:
    """VAE latent of a shot of `frames` frames (4n + 1) fed slice by slice: each (C, t, H, W) in
    [-1, 1] as input_transform gives it, t following encode_slices. Returns encode's latent bit
    for bit, (T', h, w, 16) in channel-major memory.

    Adapted from numz's vae_encode (src/core/infer.py:117) and slicing_encode
    (src/models/video_vae_v3/modules/attn_video_vae.py:1254), ByteDance's (e4de8c2) as numz
    modified them, Apache-2.0 (NOTICE). The steps are theirs: the same slices, with the same
    causal memory states, then the posterior's mode (diffusers' DiagonalGaussianDistribution: the
    first half of the channels), shifted and scaled. Only the input is never whole in memory."""
    runner, vae = models.runner, models.runner.vae
    scale = runner.config.vae.scaling_factor
    shift = runner.config.vae.get("shifting_factor", 0.0)
    sizes = encode_slices(models, frames)
    encoded: list[Tensor] = []
    try:
        for index, x in enumerate(slices):
            if index >= len(sizes) or x.shape[1] != sizes[index]:
                raise ValueError(f"slice {index} of {x.shape[1]} frames, the VAE takes {sizes}")
            if next(vae.parameters()).dtype != x.dtype:
                raise ValueError(f"input {x.dtype}, VAE {next(vae.parameters()).dtype}")
            if len(sizes) == 1:
                state = MemoryState.DISABLED
            else:
                state = MemoryState.INITIALIZING if index == 0 else MemoryState.ACTIVE
            encoded.append(vae._encode(x.unsqueeze(0), memory_state=state))
        if len(encoded) != len(sizes):
            raise ValueError(f"{len(encoded)} slices, the VAE takes {sizes}")
    finally:
        for module in vae.modules():
            if isinstance(module, InflatedCausalConv3d) and module.memory is not None:
                module.memory = None
    h = torch.cat(encoded, dim=2) if len(encoded) > 1 else encoded[0]
    mean = torch.chunk(h, 2, dim=1)[0]  # (1, 16, T', h, w)
    if isinstance(scale, ListConfig):
        scale = torch.tensor(scale, device=mean.device, dtype=mean.dtype)
    if isinstance(shift, ListConfig):
        shift = torch.tensor(shift, device=mean.device, dtype=mean.dtype)
    return ((mean.movedim(1, -1) - shift) * scale).squeeze(0)


def condition(models: Models, noise: Tensor, latent: Tensor) -> Tensor:
    """The DiT's conditioning for the super-resolution task: (T', h, w, 17)."""
    return models.runner.get_condition(noise, task="sr", latent_blur=latent)


def sample(models: Models, noise: Tensor, cond: Tensor) -> Tensor:
    """One Euler step of the DiT from noise, under bfloat16 autocast when its weights are another
    dtype, as numz runs it (generation_phases.py:704-724)."""
    autocast = next(models.runner.dit.parameters()).dtype != COMPUTE_DTYPE
    with torch.no_grad(), torch.autocast(models.device.type, COMPUTE_DTYPE, enabled=autocast):
        return models.runner.inference(
            noises=[noise],
            conditions=[cond],
            texts_pos=[models.text_pos],
            texts_neg=[models.text_neg],
        )[0]


def decode(models: Models, latent: Tensor) -> Tensor:
    """VAE decode of latent (T', h, w, 16): frames (C, T, H, W) in [-1, 1], unclamped."""
    return models.runner.vae_decode([latent])[0]


@torch.no_grad()
def decode_stream(models: Models, latent: Tensor) -> Iterator[Tensor]:
    """VAE decode of latent (T', h, w, 16), slice by slice: frames (C, t, H, W) in [-1, 1],
    unclamped, as they come.

    Adapted from numz's vae_decode (src/core/infer.py:203) and slicing_decode
    (src/models/video_vae_v3/modules/attn_video_vae.py:1278), ByteDance's (e4de8c2) as numz
    modified them, Apache-2.0 (NOTICE). The steps are theirs: the same scaling, the same slices
    (the first two latents, then one at a time) with the same causal memory states. So the frames
    are the one-pass decode's, bit for bit; only their concatenation is left out.
    """
    runner, vae = models.runner, models.runner.vae
    scale = runner.config.vae.scaling_factor
    shift = runner.config.vae.get("shifting_factor", 0.0)
    if isinstance(scale, ListConfig):
        scale = torch.tensor(scale, device=latent.device, dtype=latent.dtype)
    if isinstance(shift, ListConfig):
        shift = torch.tensor(shift, device=latent.device, dtype=latent.dtype)
    if next(vae.parameters()).dtype != latent.dtype:
        raise ValueError(f"latent {latent.dtype}, VAE {next(vae.parameters()).dtype}")
    z = (latent.unsqueeze(0) / scale + shift).movedim(-1, 1)  # (1, 16, T', h, w)
    if vae.use_slicing and (z.shape[2] - 1) > vae.slicing_latent_min_size:
        slices = z[:, :, 1:].split(split_size=vae.slicing_latent_min_size, dim=2)
        try:
            first = torch.cat((z[:, :, :1], slices[0]), dim=2)
            yield vae._decode(first, memory_state=MemoryState.INITIALIZING)[0]
            for z_slice in slices[1:]:
                yield vae._decode(z_slice, memory_state=MemoryState.ACTIVE)[0]
        finally:
            for module in vae.modules():
                if isinstance(module, InflatedCausalConv3d) and module.memory is not None:
                    module.memory = None
    else:
        yield vae._decode(z)[0]

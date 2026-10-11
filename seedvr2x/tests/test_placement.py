"""The DiT's placement on a real card (runtime/blockswap.py): milestone 3's first gate, swapping
changes no bit of the DiT's output (DESIGN.md, BlockSwap, rewritten), for 0, 3, 12 and all 36
blocks swapped, without the prefetch and with it; the DiT's peak falls by each swapped block but
the two in the slots; a model sent away leaves the device to the other; and a plan's load never
has a swapped DiT whole on the device.

Needs a GPU (pytest -m gpu) with the 7B whole on it and a window beside (18 GiB free), the
weights (SEEDVR2X_MODEL_DIR: a 7B DiT in fp16 and the VAE, numz's files or seedvr2x's own), and
16 GiB of host memory to pin. The windows are random latents of 2 frames at 960x540, as
research/scripts/placement_bench.py draws them, which measures the same at a job's sizes, with
the times: the simulation of tests/test_blockswap.py holds the events' order, this the bits on
the hardware. It runs under the allocator the process has, torch's own unless
PYTORCH_CUDA_ALLOC_CONF says otherwise (torch reads it when imported, before this module can
set it); the probe runs under the command line's, cudaMallocAsync.
"""

import gc
import hashlib
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch import Tensor

from seedvr2x.runtime import blockswap, job, model
from seedvr2x.runtime.blockswap import BLOCK_BYTES, SLOTS, Placement, dit_parts
from seedvr2x.runtime.plan import Plan

MODELS = os.environ.get("SEEDVR2X_MODEL_DIR")
DITS = (
    "seedvr2_ema_7b_fp16.safetensors",
    "seedvr2x_ema_7b_fp16.safetensors",
    "seedvr2x_ema_7b_sharp_fp16.safetensors",
)
VAES = ("ema_vae_fp16.safetensors", "seedvr2x_ema_vae_fp16.safetensors")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not MODELS, reason="needs SEEDVR2X_MODEL_DIR"),
]

GIB = 1 << 30
TARGET = (540, 960)
LATENTS = 2
FORWARDS = 2
# What the allocator's rounding and the blocks' cached RoPE tables may add to a figure.
SLACK = 64 << 20


def found(names: tuple[str, ...]) -> Path:
    assert MODELS is not None
    for name in names:
        if (Path(MODELS) / name).is_file():
            return Path(MODELS) / name
    pytest.skip(f"none of {', '.join(names)} in SEEDVR2X_MODEL_DIR")


def digest(tensor: Tensor) -> str:
    return hashlib.sha256(tensor.float().cpu().contiguous().numpy().tobytes()).hexdigest()


def windows(models: model.Models) -> tuple[list[str], int]:
    """FORWARDS windows through the DiT, as shot.sample_windows runs one: their digests, and the
    device's peak over them, in bytes."""
    height, width = job.padded_size(TARGET)
    generator = torch.Generator().manual_seed(7)
    values = torch.randn((16, LATENTS, height // 8, width // 8), generator=generator)
    latent = values.to(model.COMPUTE_DTYPE).permute(1, 2, 3, 0).to(models.device)
    torch.cuda.synchronize(models.device)
    torch.cuda.reset_peak_memory_stats(models.device)
    digests: list[str] = []
    for index in range(FORWARDS):
        model.set_seed(42 + index)
        noise = torch.randn_like(latent, dtype=model.COMPUTE_DTYPE)
        digests.append(digest(model.sample(models, noise, model.condition(models, noise, latent))))
    return digests, torch.cuda.max_memory_allocated(models.device)


def round_trip(models: model.Models) -> str:
    """A small VAE encode and decode's digest: 5 frames of 256x256."""
    video = torch.rand((3, 5, 256, 256), generator=torch.Generator().manual_seed(7)) * 2 - 1
    with torch.no_grad():
        latent = model.encode(models, video.to(models.device, model.COMPUTE_DTYPE))
        return digest(latent) + digest(model.decode(models, latent))


def allocated(device: torch.device) -> int:
    torch.cuda.synchronize(device)
    return torch.cuda.memory_allocated(device)


@dataclass
class Loaded:
    """The models loaded whole, as a run without a plan loads them, with what the placements
    are held to: the DiT's windows' digests, their peak, in bytes, and the VAE's round trip."""

    models: model.Models
    digests: list[str]
    peak: int
    vae: str


@pytest.fixture(scope="module")
def loaded() -> Iterator[Loaded]:
    models = model.load_models(found(DITS), found(VAES), torch.device("cuda:0"))
    digests, peak = windows(models)
    yield Loaded(models, digests, peak, round_trip(models))
    models.runner.dit = models.runner.vae = None
    gc.collect()
    torch.cuda.empty_cache()


def test_swapping_changes_no_bit_and_takes_its_blocks_off_the_peak(loaded: Loaded) -> None:
    models = loaded.models
    device, dit = models.device, models.runner.dit
    whole = allocated(device)
    backend = blockswap.Cuda(device)
    # Every parameter of the DiT to a pinned host copy: 15.35 GiB, and the DiT away.
    parts = blockswap.host(dit, dit_parts(len(dit.blocks), 0, other=True), backend)
    assert all(part.host.pinned for part in parts.values())
    pinned = sum(part.host.bytes.numel() for part in parts.values())
    assert f"{pinned / GIB:.2f}" == "15.35"
    # Under torch's own allocator the parameters' blocks, freed, stay in its cache, and the
    # buffers that take their place below are new memory: 31 GiB asked of a 32 GB card, which a
    # driver that spills into system memory gives, to fail later (a sticky CUDA error, met on a
    # desktop card under WSL). A test's sequence: the loader never has the DiT on the device.
    torch.cuda.empty_cache()
    away = allocated(device)
    assert abs(whole - away - pinned) < SLACK
    # A block's bytes, each of its 30 parameters on a multiple of 512.
    sizes = {part.size for name, part in parts.items() if name != blockswap.OTHER}
    assert len(sizes) == 1 and BLOCK_BYTES <= min(sizes) < BLOCK_BYTES + 30 * 512
    for swapped, prefetch, verify in [
        (0, False, False),
        (3, False, False),
        (3, True, False),
        (12, False, False),
        (12, True, False),
        (36, False, False),
        (36, True, False),
        (36, True, True),
    ]:
        said = f"{swapped} swapped, prefetch {prefetch}, verify {verify}"
        placement = Placement(dit, device, swapped, prefetch, parts=parts, verify=verify)
        placement.enter("dit")
        here = allocated(device)
        assert abs(here - away - placement.device_bytes()) < SLACK, said
        digests, peak = windows(models)
        # The gate: the same bits as the DiT loaded whole.
        assert digests == loaded.digests, said
        # The peak: each swapped block's bytes less, but the two in the slots.
        expected = loaded.peak - max(swapped - SLOTS, 0) * BLOCK_BYTES
        assert abs(peak - expected) < SLACK, (said, peak / GIB, expected / GIB)
        if verify:
            assert placement.verified == swapped * FORWARDS
        # Away, the device holds what it held without the DiT: the VAE alone finds it so.
        placement.enter("encode", 0)
        assert abs(allocated(device) - away) < SLACK, said
        assert round_trip(models) == loaded.vae, said
        placement.close()
    assert abs(allocated(device) - away) < SLACK


def test_a_plan_s_load_never_has_the_dit_whole_on_the_device(loaded: Loaded) -> None:
    # All 36 blocks swapped, the DiT away in the VAE's phases and the VAE in the DiT's: what a
    # 16 GB card runs. The first models' weights are let go first: two DiTs would pass the card.
    device = loaded.models.device
    loaded.models.runner.dit = loaded.models.runner.vae = None
    gc.collect()
    torch.cuda.empty_cache()
    before = allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    plan = Plan(None, swapped=36, resident_encode=0, resident_decode=0, vae_in_dit=False)
    models = model.load_models(found(DITS), found(VAES), device, plan=plan)
    placement, vae = models.placement, models.vae
    assert placement is not None and vae is not None
    # The load's peak on the device: the buffers, and the VAE's weights one tensor at a time,
    # cast there on their way to their host copy: never a block of the DiT (0.026 GiB measured).
    assert torch.cuda.max_memory_allocated(device) - before < SLACK
    resting = allocated(device)
    assert resting - before < SLACK
    assert f"{placement.host_bytes / GIB:.2f}" == "15.35" and vae.host.pinned
    for _ in range(2):
        models.enter("encode")
        assert abs(allocated(device) - resting - vae.size) < SLACK
        assert round_trip(models) == loaded.vae
        models.enter("dit")
        # Two slots and the parameters outside the blocks: 1.0 GiB of the DiT's 15.35.
        assert abs(allocated(device) - resting - placement.device_bytes()) < SLACK
        assert f"{placement.device_bytes() / GIB:.2f}" == "1.00"
        digests, peak = windows(models)
        assert digests == loaded.digests
        expected = loaded.peak - 34 * BLOCK_BYTES - vae.size
        assert abs(peak - expected) < 2 * SLACK, (peak / GIB, expected / GIB)
        models.enter("decode")
        assert abs(allocated(device) - resting - vae.size) < SLACK
        assert round_trip(models) == loaded.vae

"""The DiT's placement on a real card (runtime/blockswap.py): milestone 3's first gate, swapping
changes no bit of the DiT's output (DESIGN.md, BlockSwap, rewritten), for 0, 3, 12 and all 36
blocks swapped, without the prefetch and with it; the DiT's peak falls by each swapped block but
the two in the slots; a model sent away leaves the device to the other; a plan's load never
has a swapped DiT whole on the device; and no load holds more system memory at its peak than it
leaves, but the buffer its tensors are read through.

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
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch import Tensor

from seedvr2x.runtime import blockswap, job, model
from seedvr2x.runtime.blockswap import Placement, dit_parts
from seedvr2x.runtime.plan import Plan
from seedvr2x.runtime.weights import DIT_7B, SLOTS

BLOCK_BYTES = DIT_7B.blocks[0]  # a block of the 7B in fp16: 0.4220 GiB

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
# What a load's peak in system memory may pass what the load leaves resident by: the buffer its
# tensors are read through, 108 MiB for the 7B's largest, and what torch takes and gives back
# on the way. Read from a mapping of the file, the peak passed it by the file's bytes bound for
# the device, 15.3 GiB of a 7B loaded whole (research/scripts/load_peak.py).
LOAD_OVER = GIB // 2


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


def resident(peak: bool = False) -> int:
    """The process's resident system memory now, in bytes, or with `peak` its most since
    measure_resident: VmRSS and VmHWM of /proc/self/status."""
    name = "VmHWM:" if peak else "VmRSS:"
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(name):
            return int(line.split()[1]) * 1024
    raise RuntimeError(f"no {name} in /proc/self/status")


def measure_resident() -> None:
    """Start the kernel's high-water mark of this process's resident memory again from what
    is resident now (proc(5): 5 written to /proc/PID/clear_refs)."""
    Path("/proc/self/clear_refs").write_text("5")


@dataclass
class Loaded:
    """The models loaded whole, as a run without a plan loads them, with what the placements
    are held to: the DiT's windows' digests, their peak, in bytes, and the VAE's round trip."""

    models: model.Models
    digests: list[str]
    peak: int
    vae: str
    # The process's resident system memory at the load's peak, and once loaded, in bytes.
    host_peak: int
    host_after: int


@pytest.fixture(scope="module")
def loaded() -> Iterator[Loaded]:
    measure_resident()
    models = model.load_models(found(DITS), found(VAES), torch.device("cuda:0"))
    host_peak, host_after = resident(peak=True), resident()
    digests, peak = windows(models)
    yield Loaded(models, digests, peak, round_trip(models), host_peak, host_after)
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


def test_a_load_s_peak_in_system_memory_is_what_it_leaves(loaded: Loaded) -> None:
    # The models loaded whole, every weight bound for the device: the process never held the
    # file's pages, 16.5 GiB resident at such a load's peak when it read the file through a
    # mapping, for 1.25 once loaded (this test on that read: 16.48 for 1.22).
    over = loaded.host_peak - loaded.host_after
    assert over < LOAD_OVER, (loaded.host_peak / GIB, loaded.host_after / GIB)


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
    measure_resident()
    models = model.load_models(found(DITS), found(VAES), device, plan=plan)
    host_peak, host_after = resident(peak=True), resident()
    placement, vae = models.placement, models.vae
    assert placement is not None and vae is not None
    # In system memory, the pinned copies and no more: the VAE's weights, cast on the device,
    # come from the buffer the file is read through, not from a mapping of the file.
    assert host_peak - host_after < LOAD_OVER, (host_peak / GIB, host_after / GIB)
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


def test_the_driver_is_asked() -> None:
    # The startup probe (runtime/memory.py) on a real driver, its calls through ctypes and all:
    # it answers, in under two seconds, and leaves the card as it found it. A driver reached
    # without WSL's device refuses what the card can't hold; under WSL, Windows' driver gave it
    # (driver 616.92, 2026-10-11), and whichever it does now is said in the test's output.
    # Asked, by one request of OVER more than it reports free, where that figure and the whole
    # card's by NVML agree within OVER, or the card's isn't read. Not asked where it reports more
    # than the card has by more than OVER, other programs' memory left out of its figure: taken
    # for one that spills, the card's figure given with the answer (memory.probe). Which of the
    # two the card is in is read here too, before the probe, and the answer held to it: a probe
    # that asks beside another program's memory takes the card from that program.
    from seedvr2x.runtime import memory

    device = torch.device("cuda:0")
    torch.cuda.synchronize(device)
    before = torch.cuda.mem_get_info(device)[0]
    card = memory._read(0)  # pyright: ignore[reportPrivateUsage]
    started = time.perf_counter()
    found = memory.probe()
    took = time.perf_counter() - started
    assert found is not None and took < 2.0, (found, took)
    print(f"{found.said()}; {took:.2f} s; WSL's GPU device: {memory.paravirtual()}")
    assert found.total >= found.free
    if found.card is None:
        assert found.asked == found.free + memory.OVER
        # SLACK: the card's figure is read a moment before the probe's own.
        assert not isinstance(card, int) or before <= card + memory.OVER + SLACK, (before, card)
    else:
        assert found.spills and found.asked == 0, found
        assert found.free > found.card + memory.OVER, found
    if not memory.paravirtual():
        assert not found.spills
    assert abs(torch.cuda.mem_get_info(device)[0] - before) < SLACK

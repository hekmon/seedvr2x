"""The seam between seedvr2x and the vendored model code.

Builds the DiT, the VAE and the diffusion runner (VideoDiffusionInfer) from the vendored configs,
loads their weights and sets them up the way numz does, so that milestone 1 reproduces numz's
output (DESIGN.md, Validation milestones). Line references are to numz at 4490bd1.

The vendored code is untyped: this module is the one place where the rest of seedvr2x calls it,
through typed functions.
"""

# pyright: basic

import ctypes
import functools
import logging
import os
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from omegaconf import ListConfig, OmegaConf
from torch import Tensor
from torchvision.transforms import Compose, InterpolationMode, Lambda, Normalize
from torchvision.transforms import functional as TVF
from tqdm import tqdm

from seedvr2x import blockswap, int8, kquant, memory, weights
from seedvr2x.blockswap import Part, Placement
from seedvr2x.frames import BLACK, padding
from seedvr2x.passes import HELD, Compilation, ModelPlan, Phase
from seedvr2x.pull import Hashes
from seedvr2x.vendor.common.config import create_object, load_config
from seedvr2x.vendor.common.seed import set_seed as vendor_set_seed
from seedvr2x.vendor.core.infer import VideoDiffusionInfer
from seedvr2x.vendor.models.dit_3b import attention as attention_3b
from seedvr2x.vendor.models.dit_7b import attention as attention_7b
from seedvr2x.vendor.models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d
from seedvr2x.vendor.models.video_vae_v3.modules.types import MemoryState
from seedvr2x.weights import SLOTS

logger = logging.getLogger(__name__)

VENDOR = Path(__file__).resolve().parent / "vendor"

# numz's pipeline dtype: bfloat16 on any GPU computing it (src/optimization/compatibility.py:
# 684-698). The VAE runs in it, its weights cast at load, and the DiT under autocast to it.
COMPUTE_DTYPE = torch.bfloat16


class _DebugLog:
    """numz's Debug object as far as the vendored code uses it (log), sent to logging."""

    def log(self, message: str, *args: Any, **kwargs: Any) -> None:
        logger.debug("%s", message)


def sampler_bars(shown: bool) -> None:
    """Show the vendored sampler's progress bar, one per window ("EulerSampler: 100%|...",
    printed by tqdm in samplers/base.py), under --verbose only, and silence it otherwise: 848
    lines in a night on milestone 7 (DESIGN.md, Architecture, What it prints, decided on
    2026-10-28; built on 2026-10-29). Done from outside, on the module's own `tqdm` name, since
    vendor/ is numz's tree path for path and never edited here (AGENTS.md); TQDM_DISABLE would
    reach every tqdm in the process, a download's bar among them (huggingface_hub's, which keeps
    its own), where this one name reaches the sampler's alone. Called with each load of the
    models, so that the last call's setting stands."""
    from seedvr2x.vendor.common.diffusion.samplers import base

    base.tqdm = functools.partial(tqdm, disable=not shown)


# The VAE's weights as a part that leaves the device (blockswap.Part), named for the messages.
VAE_PART = "the VAE"


@dataclass
class Models:
    """The loaded models, on one device, ready to run: every weight there, or with a plan
    that sends some away, each phase's (enter)."""

    runner: Any  # VideoDiffusionInfer, untyped, its dit and vae set by load_models
    text_pos: Tensor  # (58, 5120) bfloat16: the positive text embedding, the one used at cfg 1
    text_neg: Tensor  # (64, 5120) bfloat16
    attention: str  # flash_attn_2 or sdpa
    device: torch.device
    model_plan: ModelPlan | None = None  # what the weights' places follow; None: all on the device
    placement: Placement | None = None  # the DiT's, when the plan swaps blocks or sends it away
    vae: Part | None = None  # the VAE's weights, when the plan sends them away
    phase: Phase | None = None  # the phase entered last; None after the load, the parts away
    # The DiT's parts, every one, when the models were loaded with passes.HELD: kept between runs
    # in their pinned copies, placed per run by place, sent back by release. None: the models
    # were loaded for one plan, which no run can change.
    parts: dict[str, Part] | None = None

    def place(self, model_plan: ModelPlan) -> None:
        """Place models loaded with passes.HELD for a run's plan: whatever the last run left is
        sent back to the pinned copies first, then the buffers and the text embeddings come back
        to the device (release took them off it; a copy, the same values), and a new Placement
        takes the same parts, no copy and no read, with model_plan's swapped blocks. No weight is
        on the device after it: Models.enter brings each phase's, as for a job's models. A method
        rather than a function of load_models' module: it changes the models it is called on.

        ValueError for models not loaded with HELD (their parts for one plan are all there is to
        place), for a plan for another DiT than this one (_dit_parts), and for a plan with
        compile_vae, a VAE compiled being a load's matter (compiling.compile_vae): never run
        plain in its place."""
        parts = self._held("place")
        blocks = len(self.runner.dit.blocks)
        _dit_parts(model_plan, blocks)
        if model_plan.compile_vae:
            raise ValueError("a plan with compile_vae for models placed, the VAE compiled at load")
        self._send_back()
        to_device = functools.partial(_to_device, device=self.device)
        _move_buffers(self.runner.dit, to_device)
        _move_buffers(self.runner.vae, to_device)
        self.text_pos = _to_device(self.text_pos, self.device)
        self.text_neg = _to_device(self.text_neg, self.device)
        self.placement = Placement(self.runner.dit, self.device, model_plan.swapped, parts=parts)
        self.model_plan = model_plan

    def release(self) -> None:
        """Send every weight back to its pinned copy, after a run, an error or a cancel: the
        DiT's placement closed (its device work waited for, its hooks off, its slots freed), the
        VAE's part left, what the VAE's causal convolutions and the DiT's RoPE keep from call to
        call dropped, torch's cache trimmed (blockswap.Cuda.trim), the phase None. The models are
        then in none: place them for the next run. Idempotent, and safe after a run that raised
        in a window or a decode, whatever phase it was in: a part never brought is as it was.

        ValueError for models not loaded with HELD. A release that raises (a CUDA error in the
        wait) leaves the models as they were, to be released again.

        Nothing of the models is left on the device: the buffers of the DiT and of the VAE (the
        7B's RoPE tables, 108 of them, 24,774,336 B) and the text embeddings (1,249,280 B) are
        copied to system memory, pageable, and place copies them back, exactly: a copy changes no
        bit."""
        self._send_back()
        _move_buffers(self.runner.dit, _to_host)
        _move_buffers(self.runner.vae, _to_host)
        self.text_pos = _to_host(self.text_pos)
        self.text_neg = _to_host(self.text_neg)
        blockswap.Cuda(self.device).trim()

    def _send_back(self) -> None:
        """The weights of release and place: the placement closed, the VAE's part left, the
        caches the modules keep dropped, no phase and no plan."""
        self._held("release")
        self.phase = None
        self.model_plan = None
        forget(self)
        backend = blockswap.Cuda(self.device)
        if self.vae is not None and self.vae.device is not None:
            backend.wait(backend.compute())
            self.vae.leave()
        if self.placement is not None:
            self.placement.close()
            self.placement = None

    def _held(self, what: str) -> dict[str, Part]:
        if self.parts is None:
            raise ValueError(f"{what}: the models weren't loaded with passes.HELD")
        return self.parts

    def enter(self, phase: Phase) -> None:
        """Before a unit of `phase`, a shot's encode, a window or a decode: put on the device
        what the plan has there in that phase, the rest waiting in its pinned copy (DESIGN.md,
        BlockSwap, rewritten, Per phase). Called before every unit, so most calls find the phase
        entered already.

        What leaves goes first, so that the two models' weights are never there together when
        the plan has them apart: the VAE's before the DiT comes back, the DiT's before the
        VAE's do. What comes back is queued, one copy per part, ahead of the unit's work: the
        7B whole, 15.35 GiB, in 0.30 s from its pinned copies on a desktop's PCIe 5.0 link
        (research/scripts/placement_bench.py)."""
        # Where seedvr2x holds itself to the card (a driver that gives system memory in its
        # place), other programs filling the card stop the run here, before the unit passes it
        # (seedvr2x/memory.py).
        memory.check(self.device)
        if self.parts is not None and self.model_plan is None:
            raise RuntimeError("the models are released: Models.place puts them for a run")
        if self.model_plan is None or phase == self.phase:
            return
        # Unknown until every move is done: an enter that raises half-way, the VAE gone and the
        # DiT not back, must not pass for the phase it left.
        self.phase = None
        started = time.monotonic()
        backend = blockswap.Cuda(self.device)
        # The VAE's place in this phase, when it has a part: away in the DiT's phase where the
        # plan says so. A job's models have a VAE part only for a plan that does (load_models),
        # so for them this is "the DiT's phase" as before; models loaded with HELD always have
        # one, and a plan placed on them (place) may keep the VAE there.
        away = phase == "dit" and not self.model_plan.vae_in_dit
        if self.vae is not None and away and self.vae.device is not None:
            backend.wait(backend.compute())
            self.vae.leave()
            backend.trim()
        if self.placement is not None:
            resident = {
                "encode": self.model_plan.resident_encode,
                "dit": None,
                "decode": self.model_plan.resident_decode,
            }[phase]
            self.placement.enter(phase, resident)
        if self.vae is not None and not away and self.vae.device is None:
            self.vae.enter(backend)
        self.phase = phase
        logger.debug("phase %s entered in %.3f s", phase, time.monotonic() - started)


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


def load_models(
    dit_file: Path,
    vae_file: Path,
    device: torch.device,
    hashes: Hashes | None = None,
    model_plan: ModelPlan | None = None,
    compilation: Compilation | None = None,
) -> Models:
    """Build and load the DiT in the file at dit_file and the VAE in the file at vae_file, as
    pull.resolve found them, in --model-dir or in Hugging Face's cache, each checked first
    (weights.check, ModelError): the DiT is built from the config of what its file is, and from
    an int8 file its block matrices as int8.Int8Linear, multiplied in int8 (seedvr2x/int8.py), from
    a dynamic GGUF file as kquant.GGUFLinear, dequantised before each multiply. A
    file hashed this run (hashes) must still be the one hashed (Hashes.unchanged, ModelError),
    before anything is built and right after its own load. With hashes, each file is hashed as it
    is read, once, and its SHA-256 proved or given at the end of its own load (Hashes.proved): a
    file whose bytes aren't those the run took them for is refused there, before the model is
    used (DESIGN.md, Weights, decided on 2026-11-03).

    With passes.HELD itself, that object and not a plan equal to it, the models are kept between
    runs: Models.parts holds every part of the DiT, and Models.place gives each run its plan. A
    job's plan that equals HELD by value is a job's: its models serve that plan alone.

    Without a plan, every weight is on the device, for good. With one, the weights it has away
    from the device in some phase, the DiT's swapped blocks among them, are read straight into
    pinned host copies and wait there for Models.enter: a swapped DiT is never whole on the
    device, which a 16 GB card couldn't hold (DESIGN.md, BlockSwap, rewritten). The host
    refusing those copies, or short of the memory they take, is a blockswap.HostMemoryError,
    which names it.

    With a plan that has compile_vae and its compilation (planner.compile_shapes), the VAE is
    compiled and warmed right after it is loaded, before the DiT is read: alone on the card, its
    shapes compiled with torch's own workers, which are shut before the DiT's blocks are read
    into their pinned copies (seedvr2x/compiling.py; DESIGN.md, Memory planner, "torch.compile:
    the VAE, opt-in, in v0.1.0"). compilation.seconds says how long that took."""
    # numz goes by "7b" in the file name (src/core/model_configuration.py:718-720), so a renamed
    # file loads as what its name says; seedvr2x goes by the file's tensors (DESIGN.md, Weights).
    sampler_bars(logger.isEnabledFor(logging.DEBUG))
    dit_architecture = weights.check(dit_file, "dit")
    config_dir = dit_architecture.config
    weights.check(vae_file, "vae")

    # Each file's stat again, before anything is built and right after its own load, which ends
    # seconds after the other's: a file replaced before its load or during it is refused, never
    # run unchecked (pull.Hashes, which says what a stat misses).
    def unchanged(path: Path) -> None:
        if hashes is not None:
            hashes.unchanged(path)

    unchanged(dit_file)
    unchanged(vae_file)

    # Each file hashed as its load reads it, every byte once, and its SHA-256 proved at the load's
    # end, before anything is made of it (DESIGN.md, Weights, decided on 2026-11-03).
    def hashed(path: Path) -> Callable[[str, os.stat_result], None] | None:
        if hashes is None:
            return None
        return functools.partial(hashes.proved, path)

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
    # The VAE's 0.47 GiB leave for the DiT's phase when the plan says so: one part.
    vae_leaves = model_plan is not None and not model_plan.vae_in_dit
    vae_parts = _load_weights(
        *(vae, vae_file, device, COMPUTE_DTYPE, (lambda key: VAE_PART) if vae_leaves else None),
        hashed=hashed(vae_file),
    )
    unchanged(vae_file)
    # Out of training mode, the causal convolutions keep their memory across slices
    # (causal_inflation_lib.py:242, 268). Slicing and memory limits from the config
    # (model_configuration.py:1241-1259): fixed thresholds on tensor sizes, not on free memory.
    vae.requires_grad_(False).eval()
    vae.set_causal_slicing(**runner.config.vae.slicing)
    vae.set_memory_limit(**runner.config.vae.memory_limit)
    runner.vae = vae

    # The models, as far as the VAE alone: what compiles it (compiling.compile_vae) runs its
    # passes through the run's own functions, which take them. The text embeddings are read here
    # for it, the DiT's place in them filled below.
    text_pos = torch.load(VENDOR / "pos_emb.pt", weights_only=True).to(device, COMPUTE_DTYPE)
    text_neg = torch.load(VENDOR / "neg_emb.pt", weights_only=True).to(device, COMPUTE_DTYPE)
    models = Models(runner, text_pos, text_neg, attention_backend(), device, model_plan)
    models.vae = vae_parts.get(VAE_PART)
    if model_plan is not None and model_plan.compile_vae:
        if compilation is None:
            raise ValueError("a plan with compile_vae, and nothing to compile the VAE with")
        # The VAE compiled before the DiT is read: it alone is on the card, and torch's compile
        # workers, shut at its end, hold none of the host memory the DiT's pinned copies take.
        from seedvr2x import compiling

        compiling.compile_vae(models, compilation)

    # The DiT keeps its weights' dtype (the override is commented out at model_configuration.py:
    # 1048-1052). numz leaves it in training mode, which changes nothing: its only use is a
    # gradient checkpointing stub (nadit.py:30-31).
    with torch.device("meta"):
        dit = create_object(runner.config.dit.model)
    skip = None
    if dit_architecture.layout == weights.INT8:
        # An int8 file's block matrices in the INT8 layout: an Int8Linear in each nn.Linear's
        # place, swapped on the meta device before the load, so that vendor/ stays numz's
        # (AGENTS.md). Their markers, checked with the file (weights.check), are no tensor of
        # the model's: never read.
        _swap_int8(dit)
        skip = _marker
    elif dit_architecture.layout == weights.GGUF:
        # A dynamic GGUF file's block matrices as quantised bytes: a GGUFLinear in each
        # nn.Linear's place, of the ggml type the architecture pins for it, on the meta device
        # before the load, as int8's. Nothing in the file is skipped.
        _swap_gguf(dit, dit_architecture)
    dit_parts = _load_weights(
        *(dit, dit_file, device, None, _dit_parts(model_plan, len(dit.blocks)), skip),
        hashed=hashed(dit_file),
    )
    unchanged(dit_file)
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

    if model_plan is not None and (dit_parts or model_plan.swapped):
        models.placement = Placement(dit, device, model_plan.swapped, parts=dit_parts)
    # By identity: a job's plan can equal HELD by value (swapped 0, nothing resident, the VAE away)
    # and is still a job's, for one plan, whose models no run can place again.
    if model_plan is HELD:
        models.parts = dict(dit_parts)
    _say_pinned(models, [*dit_parts, *vae_parts])
    return models


def _say_pinned(models: Models, parts: Iterable[str]) -> None:
    """Log the host memory a plan's load has pinned, for the parts named, and the blocks it
    swaps; and when it swaps one or two, that they free nothing."""
    model_plan, placement = models.model_plan, models.placement
    pinned = (placement.host_bytes if placement is not None else 0) + (
        models.vae.host.bytes.numel() if models.vae is not None else 0
    )
    if model_plan is None or not pinned:
        return
    swapped = {0: "none of the DiT's blocks swapped", 1: "the DiT's last block swapped"}.get(
        model_plan.swapped, f"the DiT's last {model_plan.swapped} blocks swapped"
    )
    if model_plan is HELD:
        # HELD's swapped is 0, but nothing stays on the device: a job's plan places the blocks
        # at each run (Models.place), which "none swapped" would contradict.
        swapped = "every weight in its pinned copy, placed by each run's plan"
    logger.info(
        "%.2f GiB of host memory pinned for the weights that leave the GPU: %s; %s",
        pinned / blockswap.GIB,
        blockswap.listing(parts),
        swapped,
    )
    if placement is not None and 0 < model_plan.swapped <= SLOTS:
        # Taken as given, a pin being taken exactly (DESIGN.md, Memory planner): said, since it
        # buys nothing. The planner never chooses it.
        logger.warning(
            "%s: that frees nothing in the DiT's phase, where swapped blocks take turns in %d"
            " buffers on the GPU; from %d swapped blocks up, each frees %.2f GiB there",
            swapped,
            SLOTS,
            SLOTS + 1,
            placement.block_bytes / blockswap.GIB,
        )


def _dit_parts(model_plan: ModelPlan | None, blocks: int) -> Callable[[str], str | None] | None:
    """Which of the DiT's parameters a plan has away from the device in some phase, each by
    its part (blockswap.dit_parts); None when all stay: the blocks it swaps, the last ones; the
    blocks a VAE phase doesn't keep; and the parameters outside the blocks when one keeps
    nothing."""
    if model_plan is None:
        return None
    fixed = blocks - model_plan.swapped
    residents = (model_plan.resident_encode, model_plan.resident_decode)
    if not 0 <= model_plan.swapped <= blocks or any(
        r is not None and not 0 <= r <= fixed for r in residents
    ):
        raise ValueError(f"a plan for another DiT than this one of {blocks} blocks: {model_plan}")
    stay = min(fixed if resident is None else resident for resident in residents)
    other = 0 in residents
    return blockswap.dit_parts(blocks, stay, other) if stay < blocks or other else None


# The DiT's block matrices in an int8 or a GGUF file (weights.MATRIX), 8 a block.
INT8_MATRICES = 288
GGUF_MATRICES = 288


def _swap_int8(dit: torch.nn.Module) -> None:
    """An int8.Int8Linear in the place of each of the 7B's 288 block matrices, built on the meta
    device (int8.swap); RuntimeError for another count, a model the int8 file's tensors aren't
    for, which weights.check has refused already."""
    swapped = int8.swap(dit)
    if len(swapped) != INT8_MATRICES:
        raise RuntimeError(f"{len(swapped)} block matrices swapped to int8, not {INT8_MATRICES}")


def _swap_gguf(dit: torch.nn.Module, architecture: weights.Architecture) -> None:
    """A kquant.GGUFLinear in the place of each of the 7B's 288 block matrices, of the ggml type
    the architecture pins for it (weights.Architecture.dtype_of), built on the meta device
    (kquant.swap); RuntimeError for another count, a model the GGUF file's tensors aren't for,
    which weights.check has refused already."""
    types = {
        f"blocks.{index}.{module}": weights.GGUF_TYPES[kind]
        for index, each in enumerate(architecture.types)
        for module, kind in zip(weights.GGUF_ORDER, each, strict=True)
    }
    swapped = kquant.swap(dit, types)
    if len(swapped) != GGUF_MATRICES:
        raise RuntimeError(f"{len(swapped)} block matrices swapped to GGUF, not {GGUF_MATRICES}")


def _marker(key: str) -> bool:
    """Whether a tensor of an int8 file is one of its markers (weights.MARK), which say its
    layout and are no tensor of the model's."""
    return key.endswith(weights.MARK)


def _load_weights(
    model: torch.nn.Module,
    path: Path,
    device: torch.device,
    dtype: torch.dtype | None,
    part_of: Callable[[str], str | None] | None = None,
    skip: Callable[[str], bool] | None = None,
    *,
    hashed: Callable[[str, os.stat_result], None] | None = None,
) -> dict[str, Part]:
    """Load a safetensors or GGUF checkpoint into a model built on the meta device, as numz does
    (src/core/model_loader.py:547-616, 777-835): the file's dtypes, or every floating tensor cast
    to dtype; then the buffers the checkpoint doesn't hold, still on meta, filled with zeros.
    With part_of, the parameters it names a part for go to that part's pinned host copy, not to
    the device: the parts are returned, by name, away from the device. With or without parts,
    the file is read by blockswap.read, which says why it is never mapped, as safetensors'
    own load_file maps it. The tensors skip says True of aren't kept (blockswap.read). With
    hashed, every byte of the file is hashed as it is read, and hashed is called with its SHA-256
    and the file's status as opened, before the model takes the weights (blockswap.read)."""
    state, parts = blockswap.read(model, path, device, dtype, part_of, skip=skip, hashed=hashed)
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
    for part in parts.values():
        part.adopt(model)
    return parts


def to_input(frames: npt.NDArray[np.float32]) -> Tensor:
    """numz's input tensor from frames (T, H, W, 3) float32 in [0, 1]: float16 on the CPU
    (inference_cli.py:613-618, 697), which rounds some 8-bit values differently from a direct
    bfloat16 cast."""
    return torch.from_numpy(frames).to(torch.float16)


def input_transform(target: tuple[int, int]) -> Callable[[Tensor], Tensor]:
    """numz's input preparation (src/core/generation_utils.py:72-84) of frames (T, C, H, W), but
    for its padding: resized to target, (height, width) (torchvision's bicubic, antialias on),
    values clamped to [0, 1], padded at the bottom and right (pad), normalised to [-1, 1], and
    moved to (C, T, H', W'), (H', W') = frames.padded_size(target).

    The resize is NaResize's (side_resize.py: TVF.resize, bicubic, antialias on) to an explicit
    size, the display aspect's (frames.target_size). For square pixels it is the size torchvision
    computes from NaResize's int, so the same call, bit for bit (tests/test_frames.py)."""
    size = list(target)
    return Compose(
        [
            Lambda(lambda x: TVF.resize(x, size, InterpolationMode.BICUBIC, antialias=True)),
            Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
            Lambda(pad),
            Normalize(0.5, 0.5),
            Lambda(lambda x: x.permute(1, 0, 2, 3)),
        ]
    )


def pad(frames: Tensor) -> Tensor:
    """frames (T, C, H, W) padded for the model (DESIGN.md, Pipeline step 0): at the bottom, the
    rows frames.padding gives, reflected from the picture (torch's reflect, the edge row not
    repeated: row H + k is row H - 2 - k), then frames.BLACK rows of zeros, black once normalised;
    at the right, columns the same way when any are reflected. As
    research/scripts/numerics_patch.py's Pad pads for NUM_PAD=reflect>=8+black+16, the variant
    measured, bit for bit: the reflection in one call, then the zeros in another
    (research/docs/numerics.md, reflect+black+16)."""
    rows, columns = padding((frames.shape[-2], frames.shape[-1]))
    reflected = torch.nn.functional.pad(frames, (0, columns, 0, rows), mode="reflect")
    black = (0, BLACK if columns else 0, 0, BLACK)
    return torch.nn.functional.pad(reflected, black, mode="constant", value=0.0)


def synchronize(device: torch.device) -> None:
    """Wait for the work queued on device, polling every 10 ms rather than blocking: Python runs
    a signal's handler between two polls, where a blocking wait holds it until the GPU is done,
    so that Ctrl-C is answered at once, not after a decode slice or a window (stop.py)."""
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
    for bit, (T', h, w, 16) in channel-major memory, in system memory.

    Adapted from numz's vae_encode (src/core/infer.py:117) and slicing_encode
    (src/models/video_vae_v3/modules/attn_video_vae.py:1254), ByteDance's (e4de8c2) as numz
    modified them, Apache-2.0 (NOTICE). The steps are theirs: the same slices, with the same
    causal memory states (encode_sliced), then the posterior's mode (mode), shifted and scaled
    (to_latent). The input is never whole in memory, and nothing on the device grows with the
    shot: numz keeps every slice's moments there and concatenates them; each slice's mode, the
    half the latent is made from, leaves here as it comes for its place in a buffer in system
    memory, (1, 16, T', h, w), and the latent is made from the buffer slice by slice.

    RuntimeError when the moments aren't the T' = 1 + (frames - 1) / 4 latents of one size and
    dtype the buffer is made for."""
    latents = 1 + (frames - 1) // models.runner.vae.temporal_downsample_factor
    means: Tensor | None = None
    start = 0
    for moments in encode_sliced(models, slices, frames):
        mean = mode(moments)
        if means is None:
            means = torch.empty((*mean.shape[:2], latents, *mean.shape[3:]), dtype=mean.dtype)
        end = start + mean.shape[2]
        expected = (*means.shape[:2], mean.shape[2], *means.shape[3:])
        # Moments of another shape could be broadcast into their place, in another dtype
        # rounded, and too few would leave latents unwritten.
        if mean.shape != expected or mean.dtype != means.dtype or end > latents:
            raise RuntimeError(
                f"latents {start} on encoded as {tuple(mean.shape)} {mean.dtype}, not"
                f" {expected} {means.dtype} within {latents}"
            )
        synchronize(models.device)
        means[:, :, start:end] = mean.to("cpu")
        start = end
    if means is None or start != latents:
        raise RuntimeError(f"{start} latents encoded, not {latents}")
    return to_latent(models, means)


@torch.no_grad()
def encode_sliced(models: Models, slices: Iterable[Tensor], frames: int) -> Iterator[Tensor]:
    """The VAE's moments of `frames` frames (4n + 1) fed slice by slice, as encode_stream takes
    them, each slice's as it is encoded: (1, 32, t', h, w), the posterior's mean then its
    log-variance, the pieces numz's slicing_encode concatenates (attn_video_vae.py:1254).
    ValueError when the slices don't follow encode_slices or aren't in the VAE's dtype.

    One pass of the causal encoder, with a memory of its own from its first slice to its last,
    cleared whatever ends it: a tiled encode makes one such pass per tile, over the tile's crop
    of every slice (seedvr2x/tiling.py)."""
    vae = models.runner.vae
    sizes = encode_slices(models, frames)
    given = 0
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
            yield vae._encode(x.unsqueeze(0), memory_state=state)
            given += 1
        if given != len(sizes):
            raise ValueError(f"{given} slices, the VAE takes {sizes}")
    finally:
        _forget(vae)


def mode(moments: Tensor) -> Tensor:
    """The posterior's mode of the VAE's moments (1, 32, t', h, w), encode_sliced's: their mean,
    the first half of the channels (diffusers' DiagonalGaussianDistribution), (1, 16, t', h, w),
    a view. The half a latent is made from (to_latent); the log-variance is used by nothing."""
    return torch.chunk(moments, 2, dim=1)[0]


def to_latent(models: Models, mean: Tensor) -> Tensor:
    """The latent of a shot's posterior mode (1, 16, T', h, w) (mode), whole in system memory:
    shifted and scaled as numz's vae_encode does (src/core/infer.py:186-188), on the device,
    slice by slice (latent_slices), so that nothing there grows with the shot: (T', h, w, 16) in
    system memory.

    Its memory is numz's latent's: channel-major storage, (16, T', h, w), seen through a permuted
    view, the layout torch gives numz's expression on the whole batch, whatever T', h and w
    (tests/test_tiling.py). The noise drawn from the latent depends on it
    (shot_pass.sample_windows)."""
    runner = models.runner
    scale = runner.config.vae.scaling_factor
    shift = runner.config.vae.get("shifting_factor", 0.0)
    if isinstance(scale, ListConfig):
        scale = torch.tensor(scale, device=models.device, dtype=mean.dtype)
    if isinstance(shift, ListConfig):
        shift = torch.tensor(shift, device=models.device, dtype=mean.dtype)
    channels, latents, height, width = mean.shape[1:]
    storage = torch.empty((channels, latents, height, width), dtype=mean.dtype)
    latent = storage.permute(1, 2, 3, 0)
    start = 0
    # PROVISIONAL: in the decode's slices, 2 latents then 1 at a time (latent_slices), as the
    # sums of a tiled encode are divided (tiling.encode_tiled): DESIGN.md gives these trips no
    # size. Each waits for the device once (synchronize), up to 10 ms a slice; more latents a
    # trip would wait less often, for a constant to choose and megabytes more on the device.
    for size in latent_slices(models, latents):
        # A copy on the CPU too, as on any device, where to() alone gives the buffer's memory.
        piece = mean[:, :, start : start + size].to(models.device, copy=True)
        made = ((piece.movedim(1, -1) - shift) * scale).squeeze(0)
        synchronize(models.device)
        latent[start : start + size] = made.to("cpu")
        start += size
    return latent


def forget(models: Models) -> None:
    """Drop what the models keep on the device between calls besides their parameters, for any
    Models, loaded with passes.HELD or for a job's plan: the VAE's causal convolutions' memory
    from slice to slice (_forget) and the RoPE frequencies the DiT's rotary modules cache
    (_forget_rope), the latter an lru_cache of the vendored class that keeps the module alive
    and, per window shape, its frequencies on the device (1.99 MB a latent at 1080p), up to 128
    shapes. Whoever drops a Models without release (a run's models not kept, an error) calls
    it first, or the cache keeps the dropped DiT's first module and its tensors until 128 other
    shapes push them out. Safe at any time: the values are recomputed from the module's buffer
    to the same bits."""
    _forget(models.runner.vae)
    _forget_rope(models.runner.dit)


def _forget(vae: Any) -> None:
    """Clear the memory the VAE's causal convolutions keep from slice to slice, as numz's
    slicing_encode and slicing_decode do after their last slice (attn_video_vae.py:1269-1273,
    1293-1297)."""
    for module in vae.modules():
        if isinstance(module, InflatedCausalConv3d) and module.memory is not None:
            module.memory = None


def _to_host(tensor: Tensor) -> Tensor:
    """tensor in system memory, pageable, a copy where it was on a device (torch's own `to`,
    which waits for the device's work on it); named so that the tests can stand in for the
    device."""
    return tensor.to("cpu")


def _to_device(tensor: Tensor, device: torch.device) -> Tensor:
    """tensor on device, the same dtype and values."""
    return tensor.to(device)


def _move_buffers(module: torch.nn.Module, move: Callable[[Tensor], Tensor]) -> None:
    """Every registered buffer of module and its submodules moved by `move` (_to_host,
    _to_device), in place in the module's own table:
    rotary_embedding_torch's `cached_freqs` and `dummy` (the module's device is its dummy's), the
    zero-filled buffers _load_weights registers. Nothing else in that package refers to a
    buffer: its `cached_freqs_seq_len` is a Python int, and the module's buffers are not its
    parameters, which the parts move."""
    for each in module.modules():
        for name, buffer in each._buffers.items():  # pyright: ignore[reportPrivateUsage]
            if buffer is not None:
                each._buffers[name] = move(buffer)  # pyright: ignore[reportPrivateUsage]


def _forget_rope(dit: Any) -> None:
    """Drop the RoPE frequencies the DiT's rotary modules cache, per (frames, rows, columns) of
    a window, on the device: the vendored rope.py memoises get_axial_freqs with an lru_cache on
    the class, of up to 128 entries keyed by the module and the shape, which keeps the first
    block's module alive and (frames x rows x columns) x 60 float32 values of the 7B a shape
    (1.99 MB a latent at 1080p), 126 of the 3B's. The values are recomputed from the module's
    buffer when a shape comes again, to the same bits."""
    cleared: set[Any] = set()
    for module in dit.modules():
        method = getattr(type(module), "get_axial_freqs", None)
        clear = getattr(method, "cache_clear", None)
        if clear is not None and method not in cleared:
            clear()
            cleared.add(method)


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


# Every row of a latent, or every column: the whole picture, no tile's crop (decode_sliced).
_ALL = slice(None)


def latent_slices(models: Models, latents: int) -> list[int]:
    """The latents of each slice the VAE decodes `latents` latents in, as its slicing_decode cuts
    them (decode_sliced): the first latent with the next slicing_latent_min_size (1), then that
    many at a time; the latents in one piece when they are no more than
    1 + slicing_latent_min_size. The slices a shot's latents go to the device in from system
    memory: for its decode, and at its encode's end (to_latent, tiling.encode_tiled)."""
    vae = models.runner.vae
    size: int = vae.slicing_latent_min_size
    if not (vae.use_slicing and latents - 1 > size):
        return [latents]
    whole, rest = divmod(latents - 1, size)
    parts = [size] * whole + ([rest] if rest else [])
    return [1 + parts[0], *parts[1:]]


def decode_slices(models: Models, latents: int) -> list[int]:
    """The frames of each slice the VAE decodes `latents` latents in (latent_slices), the first
    latent giving 1 frame and every next one temporal_downsample_factor (4)."""
    factor: int = models.runner.vae.temporal_downsample_factor
    first, *rest = latent_slices(models, latents)
    return [1 + (first - 1) * factor, *(part * factor for part in rest)]


@torch.no_grad()
def decode_stream(models: Models, latent: Tensor) -> Iterator[Tensor]:
    """VAE decode of latent (T', h, w, 16), wherever it is, a shot's in system memory, slice by
    slice: frames (C, t, H, W) in [-1, 1], unclamped, on the device, as they come, t following
    decode_slices.

    Adapted from numz's vae_decode (src/core/infer.py:203) and slicing_decode
    (src/models/video_vae_v3/modules/attn_video_vae.py:1278), ByteDance's (e4de8c2) as numz
    modified them, Apache-2.0 (NOTICE). The steps are theirs: the same scaling (decoder_input),
    the same slices (the first two latents, then one at a time) with the same causal memory
    states (decode_sliced). So the frames are the one-pass decode's, bit for bit; only their
    concatenation is left out, and the latent is never whole on the device.
    """
    yield from decode_sliced(models, latent)


def decodable(models: Models, latent: Tensor) -> None:
    """ValueError unless latent is in the VAE's dtype, the only one it decodes."""
    dtype = next(models.runner.vae.parameters()).dtype
    if dtype != latent.dtype:
        raise ValueError(f"latent {latent.dtype}, VAE {dtype}")


def decoder_input(models: Models, latent: Tensor) -> Tensor:
    """The VAE decoder's input from latent (T', h, w, 16): z (1, 16, T', h, w), the latent scaled
    and shifted back as numz's vae_decode does (src/core/infer.py:233-234), on the latent's
    device. ValueError unless the latent is in the VAE's dtype (decodable)."""
    runner = models.runner
    scale = runner.config.vae.scaling_factor
    shift = runner.config.vae.get("shifting_factor", 0.0)
    if isinstance(scale, ListConfig):
        scale = torch.tensor(scale, device=latent.device, dtype=latent.dtype)
    if isinstance(shift, ListConfig):
        shift = torch.tensor(shift, device=latent.device, dtype=latent.dtype)
    decodable(models, latent)
    return (latent.unsqueeze(0) / scale + shift).movedim(-1, 1)


@torch.no_grad()
def decode_sliced(
    models: Models, latent: Tensor, rows: slice = _ALL, columns: slice = _ALL
) -> Iterator[Tensor]:
    """latent (T', h, w, 16), or its crop to the latents' `rows` and `columns`, a tile's, decoded
    slice by slice, as numz's slicing_decode cuts the decoder's input (attn_video_vae.py:1278):
    frames (C, t, 8h, 8w) in [-1, 1], unclamped, on the device, as they come, t following
    decode_slices. ValueError unless the latent is in the VAE's dtype.

    The latent stays where it is, a shot's in system memory. Each slice's latents
    (latent_slices), cropped, go to the device as their turn comes and are scaled there
    (decoder_input): numz scales the whole batch on the device and slices that, 2 MiB a latent
    at 1080p and 8 at 4K for the two. Scaling works element by element, so the values are the
    same whatever the latent's memory. For a latent laid out as the DiT gives it, contiguous, or
    as the encode does, channel-major, the decoder is also handed each slice in the memory format
    torch read numz's in (tests/test_tiling.py), though no longer as a view of the whole
    batch's; in another memory order, a tile's crop may be read in another format, which changes
    nothing in the VAE as shipped, whose first convolution concatenates its input behind a
    contiguous tensor.

    One pass of the causal decoder, with a memory of its own from its first slice to its last,
    cleared when the generator ends, is closed or fails: a tiled decode makes one such pass per
    tile, over the tile's crop of the latent (seedvr2x/tiling.py)."""
    vae = models.runner.vae
    decodable(models, latent)
    sizes = latent_slices(models, latent.shape[0])
    if len(sizes) == 1:
        yield vae._decode(_slice_input(models, latent[:, rows, columns]))[0]
        return
    try:
        start = 0
        for size in sizes:
            z = _slice_input(models, latent[start : start + size, rows, columns])
            if start == 0:
                # numz's first slice is a concatenation, the first latent and the slice after it
                # (attn_video_vae.py:1284): a tensor of its own, in the memory torch gives one.
                z = torch.cat((z[:, :, :1], z[:, :, 1:]), dim=2)
                yield vae._decode(z, memory_state=MemoryState.INITIALIZING)[0]
            else:
                yield vae._decode(z, memory_state=MemoryState.ACTIVE)[0]
            start += size
    finally:
        _forget(vae)


def _slice_input(models: Models, latents: Tensor) -> Tensor:
    """The decoder's input (decoder_input) for one slice's latents, (t', h, w, 16), a view of a
    shot's latent wherever it is: copied to the device first, in its own layout, on the CPU
    too, where to() alone gives the latent's own memory."""
    return decoder_input(models, latents.to(models.device, copy=True))

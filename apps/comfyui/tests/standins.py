"""What the nodes' tests stand in with: the machine on the CPU (models.Machine), the models it
loads (the library's stand-in VAE, core/tests/test_still.py, and a DiT of arithmetic), ComfyUI's
other models on the stand-in card, and the settings every test starts from. ComfyUI's own modules
are tests/comfy_stand_in."""

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import comfy.model_management as management
import pytest
import torch
from comfy.model_patcher import ModelPatcher
from comfy_api.latest import io
from comfyui import models, plans
from test_still import pipeline_models, stand_in_dit

from seedvr2x import memory, model, peaks, pull, weights
from seedvr2x.memory import Reading
from seedvr2x.passes import ModelPlan
from seedvr2x.pull import ModelFiles
from seedvr2x.shots import Shot
from seedvr2x.weights import Architecture

GIB = peaks.GIB
# A picture of 24x40 pixels at 48 as the resolution: the output of 80x48 the stand-in VAE runs.
PICTURE = (24, 40)
RESOLUTION = 48
NAME = "w.safetensors"


@dataclass
class StandIn:
    """models.Machine on the CPU: ComfyUI's stand-in card is the card (its free memory the driver's
    figure), the stand-in models are what a load gives, and every call is recorded."""

    vae_models: model.Models
    dit_format: Architecture = peaks.FP16
    attention_name: str = peaks.FLASH_ATTN_2
    host_bytes: int | None = 64 * GIB
    lends: bool = False
    allocator_name: str = "cudaMallocAsync"
    # Whether torch's own allocator, where that is the stand-in's, has its segments expandable.
    expandable: bool = True
    # What the driver's reading is for the plan: the stand-in's default is a driver that refuses,
    # where the reported figure is the plan's; a test sets a held one.
    held_card: int | None = None
    loads: list[ModelPlan] = field(default_factory=list[ModelPlan])
    made: list[model.Models] = field(default_factory=list[model.Models])
    probes: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    headers: list[Path] = field(default_factory=list[Path])
    events: list[Any] = field(default_factory=list[Any])
    refuse_load: BaseException | None = None
    device: Any = field(default_factory=lambda: management.get_torch_device())

    def name(self) -> str:
        return "a stand-in card"

    def spills(self) -> bool:
        return self.lends

    def allocator(self) -> str:
        return self.allocator_name

    def free(self) -> int:
        return int(management.get_free_memory(self.device))

    def reading(self, free: int) -> Reading:
        if self.held_card is None:
            return memory.reading(free)
        counted = max(min(free, self.held_card) - memory.HELD_BACK - memory.RESERVE, 0)
        return Reading(
            free, self.held_card, counted, True, memory.HELD_BACK, expandable=self.expandable
        )

    def host(self) -> int | None:
        return self.host_bytes

    def attention(self) -> str:
        return self.attention_name

    def dit_model(self, path: Path) -> Architecture:
        self.headers.append(path)
        return self.dit_format

    def probe(self, architecture: Architecture, attention: str) -> None:
        self.probes.append((architecture.name, attention))

    def load(self, files: ModelFiles, plan: ModelPlan) -> model.Models:
        if self.refuse_load is not None:
            raise self.refuse_load
        self.loads.append(plan)
        loaded = standin_models(self.vae_models, self.events, held=plan == passes_held())
        loaded.attention = self.attention_name  # pyright: ignore[reportAttributeAccessIssue]
        self.made.append(loaded)
        return loaded


def passes_held() -> ModelPlan:
    from seedvr2x.passes import HELD

    return HELD


class Holder:
    """The models' holder as a class that can be weak-referenced (a SimpleNamespace cannot), so
    that a test can tell whether the models are still alive."""


def standin_models(vae_models: model.Models, events: list[Any], *, held: bool) -> model.Models:
    """cpu_models' VAE in bfloat16 (test_still.pipeline_models) with what a node run asks of
    models: place and release recorded, `on_card` telling whether they are placed, the pinned
    parts of models loaded for HELD (1 KiB, a part's size), the DiT's place and the embeddings
    test_held.strays looks at (empty and zero: the DiT is stood in)."""
    made: Any = Holder()
    made.__dict__.update(vars(pipeline_models(vae_models, events)))
    made.runner.dit = torch.nn.ModuleList()
    made.text_pos = made.text_neg = torch.zeros(2)
    made.attention = peaks.FLASH_ATTN_2
    made.parts = {"block": SimpleNamespace(size=1024)} if held else None
    made.vae = None
    made.on_card = False
    made.placed = []
    made.released = 0

    def place(plan: ModelPlan) -> None:
        made.placed.append(plan)
        made.on_card = True

    def release() -> None:
        made.released += 1
        made.on_card = False

    made.place, made.release = place, release
    return cast(model.Models, made)


class Stuck(ModelPatcher):
    """A model of ComfyUI's that says it unloads, and whose memory stays on the card: it counts
    as another program's after, which ComfyUI can't free."""

    def partially_unload(
        self, device_to: Any, memory_to_free: float = 0, force_patch_weights: bool = False
    ) -> int:
        return 0

    def detach(self, unpatch_all: bool = True) -> Any:
        management.card.outside += self.model.model_loaded_weight_memory
        self.model.model_loaded_weight_memory = 0
        return self.model


class Caching(ModelPatcher):
    """A model of ComfyUI's whose partial unload frees its bytes into torch's cache: they are
    free for this process but the driver counts them used, until soft_empty_cache gives them
    back (the stand-in card's `cached`)."""

    def partially_unload(
        self, device_to: Any, memory_to_free: float = 0, force_patch_weights: bool = False
    ) -> int:
        freed = super().partially_unload(device_to, memory_to_free, force_patch_weights)
        management.card.cached += freed
        return freed


def comfy_model(
    size: int, loaded: int | None = None, *, stuck: bool = False, caching: bool = False
) -> ModelPatcher:
    """One of ComfyUI's own models on the stand-in card: `size` bytes, `loaded` of them (default
    all) on it, registered with the manager. The caller keeps the patcher alive. `stuck`: it
    says it frees its memory when asked and doesn't (Stuck); `caching`: it frees it into torch's
    cache (Caching)."""
    stub = Holder()
    stub.device = None  # pyright: ignore[reportAttributeAccessIssue]
    patcher = (Stuck if stuck else Caching if caching else ModelPatcher)(
        stub, management.get_torch_device(), torch.device("cpu"), size=size
    )
    management.load_models_gpu([patcher])
    if loaded is not None:
        stub.model_loaded_weight_memory = loaded  # pyright: ignore[reportAttributeAccessIssue]
    return patcher


@pytest.fixture
def machine(cpu_models: model.Models, monkeypatch: pytest.MonkeyPatch) -> StandIn:
    """The stand-in machine in the nodes' place, the DiT of arithmetic in the library's, the model
    files' checks passed (the stand-in files are no model's), the clip and pictures' VAE on the
    CPU."""
    monkeypatch.setattr(weights, "check_models", lambda *args: None)
    monkeypatch.setattr(pull, "check_pinned", lambda *args, **kwargs: None)
    stand_in_dit(monkeypatch)
    found = StandIn(cpu_models)
    models.use(found)
    return found


def handle(directory: Path, *, keep: bool | str = True, dit: str = NAME) -> models.Handle:
    """A handle on a model directory (the stand-in files need not exist: no check reads them).
    `keep`: a mode (plans.KEEP_MODES), or a bool for on and off."""
    mode = keep if isinstance(keep, str) else "on" if keep else "off"
    return models.Handle(dit, str(directory), mode)


def default_dit() -> str:
    return plans.DEFAULT_DIT


def execute(node: type[io.ComfyNode], unique_id: str | None = "1", **inputs: Any) -> io.NodeOutput:
    """One run of a V3 node as ComfyUI's executor makes it: the inputs by keyword, `execute` called
    on the class clone whose hidden values are the node's id."""
    return node.EXECUTE_NORMALIZED({io.Hidden.unique_id: unique_id}, **inputs)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]


def inputs(settings: plans.Settings, *, video: bool = False) -> dict[str, Any]:
    """The inputs an upscale node is given for settings (the library's pins, None: left to the
    plan): the widgets' values, and a pin only where it is set, as a value node connected to its
    socket gives it, in the library's units (an unconnected optional input is not given); the
    colour correction a BOOLEAN."""
    found: dict[str, Any] = {
        "attention": plans.AUTO if settings.attention is None else settings.attention,
        "color_correction": settings.split,
        "seed": settings.seed,
    }
    pins = {
        "encode_tile": settings.encode_tile,
        "decode_tile": settings.decode_tile,
        "swap_blocks": settings.swap_blocks,
        "window": settings.window if video else None,
    }
    found.update({name: pin for name, pin in pins.items() if pin is not None})
    return found


def upscale_image(
    handle: models.Handle,
    settings: plans.Settings,
    image: torch.Tensor,
    resolution: int,
    unique_id: str | None = "1",
    *,
    plan_only: bool = False,
) -> tuple[Any, str]:
    """seedvr2x Upscale Image run as ComfyUI runs it: (the pictures, the plan's text)."""
    from comfyui import nodes

    out = execute(
        nodes.UpscaleImage,
        unique_id,
        images=image,
        models=handle,
        resolution=resolution,
        plan_only=plan_only,
        **inputs(settings),
    )
    return out[0], out[1]


def upscale_video(
    handle: models.Handle,
    settings: plans.Settings,
    images: torch.Tensor,
    resolution: int,
    unique_id: str | None = "1",
    *,
    plan_only: bool = False,
    cuts: tuple[Shot, ...] | None = None,
) -> tuple[Any, str]:
    """seedvr2x Upscale Video run as ComfyUI runs it: (the frames, the plan's text). `cuts`: the
    cut list seedvr2x Cuts gives, one shot of the whole clip by default."""
    from comfyui import nodes

    out = execute(
        nodes.UpscaleVideo,
        unique_id,
        images=images,
        cuts=(Shot(0, len(images)),) if cuts is None else cuts,
        models=handle,
        resolution=resolution,
        plan_only=plan_only,
        **inputs(settings, video=True),
    )
    return out[0], out[1]


def validate(node: type[io.ComfyNode], **constants: Any) -> bool | str:
    """The node's validate_inputs as the executor calls it at queue time: only the arguments it
    names, and only the constants (a linked input is not among them)."""
    names = inspect.signature(node.validate_inputs).parameters
    return node.validate_inputs(**{key: value for key, value in constants.items() if key in names})

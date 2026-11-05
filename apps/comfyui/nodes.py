"""seedvr2x's four ComfyUI nodes (DESIGN.md, Repository and apps, the ComfyUI nodes), in ComfyUI's
V3 schema (`comfy_api.latest`): a class a node, `define_schema` giving its `io.Schema` (id, display
name, category, inputs and outputs in order, hidden inputs) and a classmethod `execute` giving an
`io.NodeOutput`. `comfy_entrypoint` gives ComfyUI the extension that lists the four.

What the V3 schema changes for a node, read in ComfyUI v0.44.0:
- The executor calls `execute` on a clone of the class whose `hidden` holds what the schema asked
  for: the node's id (`io.Hidden.unique_id`) is `cls.hidden.unique_id`, None on the class itself.
- The inputs arrive by keyword, named as the schema names them; the outputs are
  `io.NodeOutput(...)` in the schema's order.
- The progress bar and Cancel are not part of the schema: `comfy.utils.ProgressBar` reports for
  the node being executed (the executor sets it as the current one), and
  `comfy.model_management.throw_exception_if_processing_interrupted` raises when Cancel was
  pressed, as for any node.

- seedvr2x Models: the DiT file by a name a user reads, a model directory, and whether the weights
  stay in RAM between runs (auto, on, off). Gives a light handle (models.Handle); nothing is read
  here.
- seedvr2x Upscale Image and seedvr2x Upscale Video: each carries its own settings as widgets,
  every number an INT so that a node that outputs an INT can drive it (0 is auto for a tile and
  the window cap, -1 for the blocks swapped). The plan is made at run time, where the input's size
  and the card are known (models.session), said on the node and in ComfyUI's console and given as
  text, then the library's pass over the pictures or the frames in memory. With "plan only" the
  node stops there: the plan is its text output and what follows its pictures is not run.

V3 classes hold no state (the executor runs `execute` on a clone of the class, and ignores
`__init__`): what the nodes keep between runs, the weights in RAM, is in models.py's registry.
Every use of `comfy_api` is in this module, which is the one place to change for another version
of it (`comfy_api.latest` is unstable).

Nothing here saves a file: the nodes give ComfyUI's image, float32, which its savers write, and so
carry no seedvr2x record (the CLI's 16-bit PNG has one)."""

import logging
import time
from fractions import Fraction
from typing import Any, override

import comfy.model_management as management
import numpy as np
import numpy.typing as npt
import torch
from comfy.utils import ProgressBar
from comfy_api.latest import ComfyExtension, io
from comfy_execution.graph_utils import ExecutionBlocker

from seedvr2x import clip, frames, peaks, still, still_plan
from seedvr2x.errors import Refusal
from seedvr2x.memory import Reading
from seedvr2x.passes import Plan
from seedvr2x.shot_plan import HOST_RESIDENT, PHASES, Peaks
from seedvr2x.weights import Architecture

from . import models as models_module
from . import plans, words

logger = logging.getLogger("seedvr2x_comfyui.nodes")

CATEGORY = "seedvr2x"

# A node's progress bar is a monotone position, not a measure: the clip's steps are the encode, the
# DiT's windows and the decode's frames, of unequal lengths. PROVISIONAL (impl-nodes lane 05,
# 2026-11-05): the bar's thousandths are shared 10 to the encode, 700 to the windows and 290 to
# the decode, the DiT being the bulk of a clip's time; unmeasured.
BAR = 1000
ENCODE_SHARE, WINDOW_SHARE = 100, 700

DIT_BY_LABEL = {label: name for name, label in plans.DIT_LABELS.items()}


# ---- seedvr2x Models -----------------------------------------------------------------------------


ModelsType = io.Custom("SEEDVR2X_MODELS")


class Models(io.ComfyNode):
    """The DiT file and where the files are read from."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="seedvr2xModels",
            display_name="seedvr2x Models",
            category=CATEGORY,
            description=(
                "Chooses seedvr2x's DiT file and where the model files are read from. Nothing is"
                " loaded here: the upscale nodes load the files, check them and place them on the"
                " card by a plan."
            ),
            inputs=[
                io.Combo.Input(
                    "model",
                    options=list(DIT_BY_LABEL),
                    default=plans.DIT_LABELS[plans.DEFAULT_DIT],
                    tooltip="The DiT file, with its size and the model card's typical error per"
                    " weight against the original. Take float16 when the card has room for it,"
                    " else the 8.3 GB file, else the 4.8 GB one. Pulled from Hugging Face into"
                    " its cache, at a pinned revision, where missing.",
                ),
                io.String.Input(
                    "model_dir",
                    default="",
                    multiline=False,
                    tooltip="A directory holding the model files under their names, read there"
                    " and nothing downloaded. Empty: Hugging Face's cache, where the command line"
                    " finds the same files.",
                ),
                io.Combo.Input(
                    "keep_in_ram_between_runs",
                    options=list(plans.KEEP_MODES),
                    default="auto",
                    display_name="keep in RAM between runs",
                    tooltip="Keep the weights in pinned system memory between runs, on the card"
                    " only while a node runs. auto: kept where they fit under ComfyUI's own cap"
                    " on pinned memory (the node says the sizes); on: kept whatever the cap; off:"
                    " the next run reads the files again and holds only what leaves the card.",
                ),
            ],
            outputs=[ModelsType.Output("models")],
        )

    @classmethod
    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls, model: str, model_dir: str, keep_in_ram_between_runs: str
    ) -> io.NodeOutput:
        return io.NodeOutput(
            models_module.Handle(DIT_BY_LABEL[model], model_dir.strip(), keep_in_ram_between_runs)
        )


# ---- What the upscale nodes share ----------------------------------------------------------------


def _array(images: torch.Tensor) -> npt.NDArray[np.float32]:
    """ComfyUI's IMAGE, (B, H, W, C) float32 in [0, 1], as the library takes frames: (B, H, W, 3)
    float32 contiguous. Four channels with an alpha opaque everywhere are three (said); any alpha
    that is not is refused (picture.py's rule: the model would redraw the colours under it)."""
    if images.ndim != 4 or images.shape[0] < 1:
        raise words.frames_missing()
    array = np.ascontiguousarray(images.detach().to("cpu", torch.float32).numpy())
    count = array.shape[3]
    if count == 4:
        if bool((array[..., 3] < 1.0).any()):
            raise words.transparent()
        logger.info("seedvr2x: the alpha channel is opaque everywhere, and left out")
        array = np.ascontiguousarray(array[..., :3])
    elif count != 3:
        raise words.channels(count)
    return array


def _target(width: int, height: int, resolution: int) -> tuple[int, int]:
    """(height, width) the input is resized to: the output's short side at `resolution`, square
    pixels (frames.target_size), refused where the padding can't be made on it."""
    target = frames.target_size(width, height, Fraction(1), resolution)
    try:
        frames.check_target(target, resolution)
        # The plan's bound is the output's, its sides even: a short side of 17 passes the check
        # above and is 16 rows once cropped, under what the padding can be made on.
        frames.check_target(frames.output_size(target), resolution)
    except Refusal as error:
        raise words.refusal(error) from error
    return target


def _say(text: str, node: str | None) -> None:
    """The plan on the node (ComfyUI's progress text, which its frontend shows on any node) and in
    ComfyUI's console, at INFO."""
    logger.info("%s", text)
    import server

    instance = server.PromptServer.instance
    if instance is not None and node is not None:
        instance.send_progress_text(text, node)


def _node_id(node: type[io.ComfyNode]) -> str | None:
    """The id of the node being executed (the hidden UNIQUE_ID the schema asks for), None where
    the class is not the executor's clone of it."""
    return None if node.hidden is None else node.hidden.unique_id


def _output(array: npt.NDArray[np.float32]) -> torch.Tensor:
    return torch.from_numpy(array)  # pyright: ignore[reportUnknownMemberType]


def _checked(handle: models_module.Handle) -> None:
    if not isinstance(handle, models_module.Handle):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError("seedvr2x Upscale takes the output of seedvr2x Models")


# ---- The settings an upscale node carries --------------------------------------------------------

# Every number an INT, so that any node that outputs an INT can drive it (ComfyUI's widget-to-input
# flow). Auto is a value no pin can take (plans.py has the grid and why untiled is the value past
# the largest tile): the labels and the tooltips say it, ComfyUI_frontend 1.57's INT widget having
# a min, a max, a step and a display type and no word for a special value.


def _tile_input(name: str, pass_: str) -> Any:
    return io.Int.Input(
        name,
        display_name=f"{pass_} tile, 0 = auto",
        default=plans.AUTO_TILE,
        min=plans.AUTO_TILE,
        max=plans.UNTILED_TILE,
        step=peaks.TILE_STEP,
        tooltip=f"The VAE {pass_}'s tiles, squares of this many output pixels a side, a multiple of"
        f" {peaks.TILE_STEP} from {peaks.TILE_MIN} to {peaks.TILE_MAX}. 0 = auto: the largest"
        f" tile that fits the card, or one pass if that fits. {plans.UNTILED_TILE} = one pass,"
        " untiled. A tile set is taken exactly, and the plan is built around it: a large one"
        " is faster and takes more of the card, a small one leaves the card to the DiT's"
        " windows and costs seams' blending time.",
    )


def _swap_input() -> Any:
    return io.Int.Input(
        "swap_blocks",
        display_name="swap blocks, -1 = auto",
        default=plans.AUTO_SWAP,
        min=plans.AUTO_SWAP,
        max=peaks.BLOCKS,
        step=1,
        tooltip="The DiT's last blocks that wait in system memory and stream through the card,"
        f" their memory left to the windows, from 0 to {peaks.BLOCKS}. -1 = auto: the fewest that"
        " fit. 0 = none, the whole DiT on the card in its phase. More blocks swapped fit a"
        " smaller card and make each window slower.",
    )


def _attention_input() -> Any:
    return io.Combo.Input(
        "attention",
        options=list(plans.ATTENTIONS),
        default=plans.AUTO,
        tooltip="auto: FlashAttention 2 where ComfyUI's environment has it, else PyTorch's SDPA."
        " The kernels give slightly different pictures; flash_attn_2 is refused where it is not"
        " installed.",
    )


def _correction_inputs() -> list[Any]:
    return [
        io.Combo.Input(
            "color_correction",
            options=list(plans.CORRECTIONS),
            default="split",
            display_name="colour correction",
            tooltip="split: the input's coarse lightness and colour under the model's details;"
            " none: the model's colours, which drift.",
        ),
        io.Int.Input(
            "seed",
            default=plans.SEED,
            min=0,
            max=plans.SEED_MAX,
            control_after_generate=io.ControlAfterGenerate.fixed,
            tooltip="The seed of the model's noise: the same input, settings and seed give the"
            " same picture. Fixed after each run by default, so that a run again is the same.",
        ),
    ]


def _plan_only_input() -> Any:
    return io.Boolean.Input(
        "plan_only",
        default=False,
        display_name="plan only",
        tooltip="Make the plan, say it on the node and give it as the plan output, and stop: no"
        " model is loaded, ComfyUI is asked to free nothing, and the nodes after the picture"
        " output do not run. The model files are found and their headers read (a file missing"
        " from Hugging Face's cache is downloaded).",
    )


def _resolution_input() -> Any:
    return io.Int.Input(
        "resolution",
        default=plans.RESOLUTION,
        min=plans.RESOLUTION_MIN,
        max=plans.RESOLUTION_MAX,
        tooltip="The output's short side in pixels, square pixels.",
    )


def _settings(**values: Any) -> plans.Settings:
    """The widgets' values as the settings of the run, or the node's refusal for a value no
    widget takes (a linked INT is not held to the grid by the frontend)."""
    try:
        return plans.Settings.of(**values)
    except plans.BadSetting as error:
        raise words.Refused(
            f"{error}. Set it on this node, or on the node that feeds it."
        ) from error


def _validate(**values: int | None) -> bool | str:
    """validate_inputs of the upscale nodes: the numbers typed in the widgets are checked when the
    prompt is queued, before anything runs. A linked number is not there to check (the executor
    hands over the constants only) and is checked at the node's start. The widgets' min and max
    are checked by ComfyUI itself for the inputs this function does not take."""
    found = plans.problems(**values)
    return "; ".join(found) if found else True


def _plan_only(text: str) -> io.NodeOutput:
    """What a plan only gives: the plan as text, and no picture. The picture output is an
    execution blocker with no message, which the executor passes to every node that takes it
    (silently, as a node that has nothing to give); a message would be reported as an error."""
    return io.NodeOutput(ExecutionBlocker(None), text)


# ---- seedvr2x Upscale Image ----------------------------------------------------------------------


def picture_process(source: tuple[int, int], output: tuple[int, int], count: int = 1) -> int:
    """The process's own system memory the plan counts for a batch of `count` pictures, in bytes:
    a video job's HOST_RESIDENT, plus the arrays counted from their sizes: one picture as stored
    ((H, W, 3) float32), the batch's output (`count` of (H', W', 3) float32, allocated once and
    filled picture by picture) and the decode's copy of the picture being made (one more). It is
    the `process` the plan is given, not the host figure: the plan compares the process's own
    memory with what the system has available, and the batch's output is this process's, taken
    from that memory as the pictures are written. PROVISIONAL, unmeasured (as the CLI's
    stills._process, whose file-writing terms the nodes have none of: they write no file)."""
    height, width = source
    out_height, out_width = output
    out = out_height * out_width * 3 * 4
    return HOST_RESIDENT + height * width * 3 * 4 + (count + 1) * out


type PicturePlan = tuple[still_plan.StillPlan, still_plan.PictureTiles]


def picture_planner(
    settings: plans.Settings, target: tuple[int, int], size: tuple[int, int], count: int = 1
) -> models_module.Planner[PicturePlan]:
    """The planner of a batch of `count` pictures of `size` (height, width) as stored, resized to
    `target` (frames.target_size): still_plan.plan for the batch's one picture, with the batch's
    output counted in the process's memory (picture_process), the settings' pins and colour
    correction, its tiles (still_plan.tiles), and what it asks of the card, the largest phase of
    picture_peaks with the room a VAE pass is held to."""
    height, width = size
    out_height, out_width = frames.output_size(target)
    bound = (max(out_height, out_width), min(out_height, out_width))
    process = picture_process(size, (out_height, out_width), count)

    def planner(
        reading: Reading, host: int | None, attention: str, architecture: Architecture, held: bool
    ) -> models_module.Planned[PicturePlan]:
        made = still_plan.plan(
            bound,
            reading,
            host,
            process=process,
            input_pixels=height * width,
            attention=attention,
            dit_model=architecture,
            split=settings.split,
            pins=settings.picture_pins(),
            picture=target,
            held=held,
        )
        tiles = still_plan.tiles(made, target)
        wanted = still_plan.picture_peaks(made, target, height * width)
        peak = max(each.allocated + each.room for each in wanted.values())
        needs = models_module.Needs(peak, made.margin, made.reading)
        return models_module.Planned((made, tiles), made.models, needs)

    return planner


class UpscaleImage(io.ComfyNode):
    """A batch of pictures, each upscaled on its own."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="seedvr2xUpscaleImage",
            display_name="seedvr2x Upscale Image",
            category=CATEGORY,
            description=(
                "Upscales each picture of the batch on its own with seedvr2x. The plan, made for"
                " the card as it would be without ComfyUI's own models, is said on the node and"
                " given as text; each setting is auto or taken exactly, and the plan is built"
                " around it."
            ),
            inputs=[
                ModelsType.Input("models"),
                io.Image.Input("image"),
                _resolution_input(),
                _tile_input("encode_tile", "encode"),
                _tile_input("decode_tile", "decode"),
                _swap_input(),
                _attention_input(),
                *_correction_inputs(),
                _plan_only_input(),
            ],
            outputs=[io.Image.Output("image"), io.String.Output("plan")],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def validate_inputs(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        encode_tile: int | None = None,
        decode_tile: int | None = None,
        swap_blocks: int | None = None,
    ) -> bool | str:
        return _validate(encode_tile=encode_tile, decode_tile=decode_tile, swap_blocks=swap_blocks)

    @classmethod
    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        models: models_module.Handle,
        image: torch.Tensor,
        resolution: int,
        encode_tile: int,
        decode_tile: int,
        swap_blocks: int,
        attention: str,
        color_correction: str,
        seed: int,
        plan_only: bool,
    ) -> io.NodeOutput:
        _checked(models)
        settings = _settings(
            encode_tile=encode_tile,
            decode_tile=decode_tile,
            swap_blocks=swap_blocks,
            attention=attention,
            color_correction=color_correction,
            seed=seed,
        )
        batch = _array(image)
        count, height, width = batch.shape[:3]
        target = _target(width, height, resolution)
        out_height, out_width = frames.output_size(target)
        planner = picture_planner(settings, target, (height, width), count)

        started = time.monotonic()
        with models_module.session(models, settings.attention, planner, plan_only=plan_only) as run:
            made, tiles = run.planned.plan
            text = plans.picture_text(
                count=count,
                source=(width, height),
                output=(out_width, out_height),
                resolution=resolution,
                dit=models.dit,
                attention=run.attention,
                plan=made,
                tiles=tiles,
                room=run.room,
                settings=settings,
                keeping=run.keeping,
                notes=run.notes,
                held=run.held,
            )
            _say(text, _node_id(cls))
            if plan_only:
                return _plan_only(text)
            bar = ProgressBar(3 * count)
            # The batch's output, allocated once and written into picture by picture: a list of
            # the pictures stacked after would hold them twice (the plan counts it once,
            # picture_process).
            outputs = np.empty((count, out_height, out_width, 3), dtype=np.float32)
            try:
                for index in range(count):
                    outputs[index] = cls._one(
                        run.models, batch[index], target, settings, tiles, bar
                    )
            except still.NonFinite as error:
                raise words.non_finite(str(error)) from error
        logger.info("seedvr2x: %d picture(s) upscaled in %.1f s", count, time.monotonic() - started)
        return io.NodeOutput(_output(outputs), text)

    @staticmethod
    def _one(
        loaded: Any,
        picture: npt.NDArray[np.float32],
        target: tuple[int, int],
        settings: plans.Settings,
        tiles: still_plan.PictureTiles,
        bar: ProgressBar,
    ) -> npt.NDArray[np.float32]:
        """One picture through the three steps of still.upscale, ComfyUI's Cancel honoured and its
        bar moved between them: the steps are the library's own (still.upscale composes them)."""
        latent = still.encoded(loaded, picture, target, settings.seed, tiles=tiles.encode)
        bar.update(1)
        management.throw_exception_if_processing_interrupted()
        latent = still.sampled(loaded, latent, settings.seed)
        bar.update(1)
        management.throw_exception_if_processing_interrupted()
        out = still.decoded(
            loaded, latent, picture, target, tiles=tiles.decode, correct=settings.split
        )
        bar.update(1)
        management.throw_exception_if_processing_interrupted()
        return out


# ---- seedvr2x Upscale Video ----------------------------------------------------------------------


type ClipPlan = tuple[Plan, Peaks]


def clip_planner(
    settings: plans.Settings, target: tuple[int, int], count: int
) -> models_module.Planner[ClipPlan]:
    """The planner of a clip of `count` frames resized to `target`: clip.plan with the settings'
    pins and colour correction, and what it asks of the card, its largest phase (Peaks.need: a
    VAE pass with the room it is held to)."""

    def planner(
        reading: Reading, host: int | None, attention: str, architecture: Architecture, held: bool
    ) -> models_module.Planned[ClipPlan]:
        made, expected = clip.plan(
            reading,
            target,
            count,
            settings.clip_pins(),
            host,
            correct=settings.split,
            attention=attention,
            dit_model=architecture,
            held=held,
        )
        peak = max(expected.need(phase) for phase in PHASES)
        needs = models_module.Needs(peak, expected.free - expected.budget, expected.reading)
        return models_module.Planned((made, expected), made.model_plan, needs)

    return planner


class UpscaleVideo(io.ComfyNode):
    """The frames of one shot, in sliding windows."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="seedvr2xUpscaleVideo",
            display_name="seedvr2x Upscale Video",
            category=CATEGORY,
            description=(
                "Upscales the frames of one shot with seedvr2x, in sliding windows of the DiT."
                " There is no cut detection: a cut inside a window blends two scenes, and nothing"
                " resumes. The command line has both. Each setting is auto or taken exactly, and"
                " the plan is built around it."
            ),
            inputs=[
                ModelsType.Input("models"),
                io.Image.Input("images"),
                _resolution_input(),
                io.Int.Input(
                    "window",
                    display_name="window cap, 0 = auto",
                    default=plans.AUTO_WINDOW,
                    min=plans.AUTO_WINDOW,
                    max=plans.WINDOW_MAX,
                    step=1,
                    tooltip="The cap on the DiT's windows, in latents of 4 frames, from"
                    f" {plans.WINDOW_MIN} to {plans.WINDOW_MAX}. 0 = auto: the longest that fits"
                    " the card. A longer window blends more frames together and takes more of"
                    " the card; a shorter one is cheaper and re-blends more often.",
                ),
                _tile_input("encode_tile", "encode"),
                _tile_input("decode_tile", "decode"),
                _swap_input(),
                _attention_input(),
                *_correction_inputs(),
                _plan_only_input(),
            ],
            outputs=[io.Image.Output("images"), io.String.Output("plan")],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def validate_inputs(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        window: int | None = None,
        encode_tile: int | None = None,
        decode_tile: int | None = None,
        swap_blocks: int | None = None,
    ) -> bool | str:
        return _validate(
            window=window, encode_tile=encode_tile, decode_tile=decode_tile, swap_blocks=swap_blocks
        )

    @classmethod
    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        models: models_module.Handle,
        images: torch.Tensor,
        resolution: int,
        window: int,
        encode_tile: int,
        decode_tile: int,
        swap_blocks: int,
        attention: str,
        color_correction: str,
        seed: int,
        plan_only: bool,
    ) -> io.NodeOutput:
        _checked(models)
        settings = _settings(
            window=window,
            encode_tile=encode_tile,
            decode_tile=decode_tile,
            swap_blocks=swap_blocks,
            attention=attention,
            color_correction=color_correction,
            seed=seed,
        )
        clip_frames = _array(images)
        count, height, width = clip_frames.shape[:3]
        target = _target(width, height, resolution)
        out_height, out_width = frames.output_size(target)

        planner = clip_planner(settings, target, count)

        started = time.monotonic()
        with models_module.session(models, settings.attention, planner, plan_only=plan_only) as run:
            made, expected = run.planned.plan
            text = plans.clip_text(
                frames=count,
                source=(width, height),
                output=(out_width, out_height),
                resolution=resolution,
                dit=models.dit,
                attention=run.attention,
                plan=made,
                expected=expected,
                room=run.room,
                settings=settings,
                keeping=run.keeping,
                notes=run.notes,
                held=run.held,
            )
            _say(text, _node_id(cls))
            if plan_only:
                return _plan_only(text)
            bar = ProgressBar(BAR)
            # One shot, the whole clip, today. A later Cuts node gives a list of frame ranges, and
            # this is then called once for each of them: everything a shot needs is its frames.
            out = cls._shot(run.models, clip_frames, target, settings, made, bar)
        logger.info("seedvr2x: %d frame(s) upscaled in %.1f s", count, time.monotonic() - started)
        return io.NodeOutput(_output(out), text)

    @staticmethod
    def _shot(
        loaded: Any,
        shot_frames: npt.NDArray[np.float32],
        target: tuple[int, int],
        settings: plans.Settings,
        made: Plan,
        bar: ProgressBar,
    ) -> npt.NDArray[np.float32]:
        """One shot, (T, H, W, 3) float32 in [0, 1], through the library's clip pass in the plan's
        windows and tiles, the bar moved and ComfyUI's Cancel honoured between its steps."""

        def step(stage: clip.Stage, done: int, total: int) -> None:
            """The clip's steps on the bar, and ComfyUI's Cancel between them."""
            if stage == "encode":
                position = ENCODE_SHARE
            elif stage == "window":
                position = ENCODE_SHARE + WINDOW_SHARE * done // total
            else:
                position = (
                    ENCODE_SHARE
                    + WINDOW_SHARE
                    + (BAR - ENCODE_SHARE - WINDOW_SHARE) * done // total
                )
            bar.update_absolute(position, BAR)
            management.throw_exception_if_processing_interrupted()

        try:
            return clip.upscale(
                loaded,
                shot_frames,
                target,
                settings.seed,
                window=made.window,
                encode_tiles=made.encode_tiles,
                decode_tiles=made.decode_tiles,
                correct=settings.split,
                step=step,
            )
        except clip.NonFinite as error:
            raise words.non_finite(str(error)) from error


# ---- What ComfyUI loads --------------------------------------------------------------------------

NODES: tuple[type[io.ComfyNode], ...] = (Models, UpscaleImage, UpscaleVideo)


class Extension(ComfyExtension):
    """The three nodes, as ComfyUI's loader asks for them (it awaits get_node_list and registers
    each class under its schema's node_id and display name)."""

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return list(NODES)


async def comfy_entrypoint() -> Extension:
    """What ComfyUI's loader looks for in a custom node's folder when it finds no
    NODE_CLASS_MAPPINGS (v0.44.0: V1 wins where both exist, so this module defines neither of
    V1's dictionaries)."""
    return Extension()

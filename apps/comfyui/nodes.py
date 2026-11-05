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
- seedvr2x Upscale Image and seedvr2x Upscale Video: the pictures and the models' handle, then
  the settings as widgets (resolution, attention, colour correction, seed, plan only), then the
  numbers the plan makes as optional INT inputs with no widget (each pass's tile, the blocks
  swapped, a clip's window length; the user's Desktop test, 2026-11-05): unwired, the plan
  chooses; wired, the value is taken exactly. No value stands for auto. The plan is made at run
  time, where the input's size and the card are known (models.session), said on the node and in
  ComfyUI's console and given as text, then the library's pass over the pictures or the frames in
  memory. With "plan only" the node stops there: the plan is its text output and what follows its
  pictures is not run.

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

from seedvr2x import clip, frames, peaks, pull, still, still_plan
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


def _size(name: str) -> str:
    """A pinned file's size as the Models node's labels say it ("8.3 GB")."""
    return pull.gigabytes(pull.PINNED[name].size)


_FLOAT16, _INT8, _GGUF = (
    _size(f"seedvr2x_ema_7b_sharp_{kind}")
    for kind in ("fp16.safetensors", "int8_convrot.safetensors", "dyn.gguf")
)


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
                "Chooses seedvr2x's model file and where the model files are read from. Nothing is"
                " loaded here: the upscale nodes load the files, check them and fit them to your"
                " graphics card."
            ),
            inputs=[
                io.Combo.Input(
                    "model",
                    options=list(DIT_BY_LABEL),
                    default=plans.DIT_LABELS[plans.DEFAULT_DIT],
                    display_name=plans.LABELS["model"],
                    # The error per weight as models/hf/README.md defines it ("How to read it")
                    # and its table gives it, the video column the same table's ("On video").
                    tooltip="The DiT file, with its size and its error per weight: how far its"
                    " stored weights are from ByteDance's original, relative to their own size"
                    " (the median over the 7B's 288 large matrices, as the model card gives it)."
                    " It measures the file, not the picture: on video, the int8 file upscales as"
                    " close to float16 as a second seed does; the dynamic GGUF drifts further at"
                    " 1080p on some content, cartoons above all. Take float16 when the card has"
                    f" room for it, else the {_INT8} file, else the {_GGUF} one. Downloaded from"
                    " Hugging Face into its cache where missing, always the same version of the"
                    " file.",
                ),
                io.String.Input(
                    "model_dir",
                    default="",
                    multiline=False,
                    display_name=plans.LABELS["model_dir"],
                    tooltip="Leave empty to download the files from Hugging Face the first time"
                    " they are needed, into its cache, where seedvr2x's command line finds them"
                    " too. To use files you already have, give their folder, and nothing is"
                    f" downloaded: it must have the VAE, {plans.VAE_FILE} (which turns images into"
                    " what the model works on, and back), and the model file under seedvr2x's"
                    f" name for it: {plans.DEFAULT_DIT} for {plans.DIT_LABELS[plans.DEFAULT_DIT]};"
                    " _int8_convrot.safetensors or _dyn.gguf in place of _fp16.safetensors for"
                    " the int8 and dynamic GGUF files; no _sharp for the 7B. These are the names"
                    " seedvr2x's command line reads from a folder too; they are not in ComfyUI's"
                    " own models folders.",
                ),
                io.Combo.Input(
                    "keep_in_ram_between_runs",
                    options=list(plans.KEEP_MODES),
                    default="auto",
                    display_name=plans.LABELS["keep_in_ram_between_runs"],
                    # ComfyUI's cap on pinned memory, v0.44.0's model_management.py: 40% of the
                    # RAM on Windows (models.pinned_cap reads it); "on" keeps past it.
                    tooltip="After a run, keep the model in your computer's RAM, so that the next"
                    " run starts in seconds instead of reading the files again (the float16 file"
                    f" is {_FLOAT16}). The graphics card is used only while a seedvr2x node runs,"
                    " whatever you choose. auto: keep it if it fits in the RAM ComfyUI allows for"
                    " this (on Windows, 40% of your RAM); the upscale node says how much it kept."
                    " on: always keep it, even past that limit: Windows may refuse, or your"
                    " computer may slow down. off: free that RAM after each run; the next run"
                    " reads the files again. ComfyUI's Unload models frees it too, and under"
                    " ComfyUI's --disable-smart-memory nothing is kept.",
                ),
            ],
            outputs=[
                ModelsType.Output(
                    "models",
                    display_name="models",
                    tooltip="The model file chosen and where it is read from, for seedvr2x's"
                    " upscale nodes. Nothing is loaded yet.",
                )
            ],
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

# The widgets are the choices with a default the user reads (resolution, attention, colour
# correction, seed, plan only). The numbers the plan makes are optional INT inputs with no widget
# (`force_input`): unwired, the plan chooses, and its text says what; wired, the value is taken
# exactly. Their min, max and step are the planner's own grid (plans.py), which no value node
# takes from them in ComfyUI_frontend 1.57.6: its Int node has its own range, and its legacy
# Primitive, given no widget's config by a socket with none, sends no value at all (read in its
# bundle, 2026-11-05). So the tooltips name the Int node, and a value a link gives is held to the
# grid at the node's start, before anything loads; a value the prompt holds, when it is queued
# (validate_inputs).


# How each optional number's tooltip begins: in ComfyUI_frontend 1.57.6 an Int node gives such
# an input its value, and a Primitive node none (above).
_OPTIONAL = "Optional: connect a number (from an Int node, for one)"


def _tile_input(name: str, pass_: str) -> Any:
    return io.Int.Input(
        name,
        display_name=plans.LABELS[name],
        optional=True,
        force_input=True,
        min=peaks.TILE_MIN,
        max=plans.UNTILED_TILE,
        step=peaks.TILE_STEP,
        tooltip=f"{_OPTIONAL} to set the tile size of the VAE {pass_}, in pixels of the output:"
        f" {peaks.TILE_MIN} to {peaks.TILE_MAX} in steps of {peaks.TILE_STEP}, or"
        f" {plans.UNTILED_TILE} for no tiles, the whole image at once. Unconnected, the plan"
        " chooses: the whole image at once if it fits the card, else the largest tile that does."
        " A larger tile is faster and takes more of the card; a smaller one leaves more room for"
        " the model and blends more seams.",
    )


def _swap_input() -> Any:
    return io.Int.Input(
        "swap_blocks",
        display_name=plans.LABELS["swap_blocks"],
        optional=True,
        force_input=True,
        min=0,
        max=peaks.BLOCKS,
        step=1,
        tooltip=f"{_OPTIONAL} to set how many of the model's {peaks.BLOCKS} blocks wait in RAM"
        " and pass through the card in turn, from 0 (none: the whole model on the card) to"
        f" {peaks.BLOCKS}. Unconnected, the plan chooses the fewest that fit. Swapping more"
        " blocks lets the model fit a smaller graphics card, and slows the run.",
    )


def _window_input() -> Any:
    return io.Int.Input(
        "window",
        display_name=plans.LABELS["window"],
        optional=True,
        force_input=True,
        min=plans.WINDOW_MIN,
        max=plans.WINDOW_MAX,
        step=1,
        tooltip=f"{_OPTIONAL} to set the longest stretch of the clip the model works on at once,"
        f" counted in groups of 4 frames: from {plans.WINDOW_MIN} (about"
        f" {4 * plans.WINDOW_MIN} frames) to {plans.WINDOW_MAX} (about"
        f" {4 * plans.WINDOW_MAX} frames). Unconnected, the plan chooses the longest that fits"
        " the card. A longer window keeps more frames consistent with each other and takes more"
        " of the card; a shorter one is lighter, and the clip is blended across more windows.",
    )


def _attention_input() -> Any:
    return io.Combo.Input(
        "attention",
        options=list(plans.ATTENTIONS),
        default=plans.AUTO,
        display_name=plans.LABELS["attention"],
        # The step's figure: research/docs/small-cards.md, the 4K picture's time, phase by phase
        # (RTX 5090, the same pins): the DiT step 2.60 s with FlashAttention 2 against 3.26 s
        # with SDPA, 20% less. No memory claim: the kernels' peaks are to become one figure.
        tooltip="auto: FlashAttention 2 if it is installed in ComfyUI's environment, else"
        " PyTorch's SDPA. FlashAttention 2 is faster: about a fifth off the model's step for a"
        " 4K picture (measured on an RTX 5090). The two round differently, so their pictures"
        " differ very slightly. flash_attn_2 stops with an error where it is not installed.",
    )


def _correction_inputs() -> list[Any]:
    return [
        io.Boolean.Input(
            "color_correction",
            default=True,
            label_on=plans.CORRECTIONS[0],
            label_off=plans.CORRECTIONS[1],
            display_name=plans.LABELS["color_correction"],
            tooltip="On (split): the input's overall colours and brightness, with the model's"
            " fine detail. Off (none): the model's own colours, which can drift from the"
            " input's.",
        ),
        io.Int.Input(
            "seed",
            default=plans.SEED,
            min=0,
            max=plans.SEED_MAX,
            control_after_generate=io.ControlAfterGenerate.fixed,
            display_name=plans.LABELS["seed"],
            tooltip="The seed of the model's noise: the same input, settings and seed give the"
            " same image on the same computer. Fixed after each run by default, so that running"
            " again gives the same result.",
        ),
    ]


def _plan_only_input() -> Any:
    return io.Boolean.Input(
        "plan_only",
        default=False,
        display_name=plans.LABELS["plan_only"],
        tooltip="Make the plan, show it on the node and give it as the plan output, and stop: no"
        " model is loaded, ComfyUI's loaded models stay as they are, and the nodes connected to"
        " the images output are skipped. The model files are found, and one missing from"
        " Hugging Face's cache is downloaded.",
    )


def _resolution_input() -> Any:
    return io.Int.Input(
        "resolution",
        default=plans.RESOLUTION,
        min=plans.RESOLUTION_MIN,
        max=plans.RESOLUTION_MAX,
        display_name=plans.LABELS["resolution"],
        tooltip="The output's short side, in pixels; the long side keeps the input's shape.",
    )


def _models_input() -> Any:
    return ModelsType.Input(
        "models",
        display_name="models",
        tooltip="The output of seedvr2x Models: the model file and where it is read from.",
    )


def _plan_output() -> Any:
    return io.String.Output(
        "plan",
        display_name="plan",
        tooltip="The plan as text: the settings, what seedvr2x chose for each one left to it, and"
        " the memory the run needs. Preview as Text shows it.",
    )


def _settings(**values: Any) -> plans.Settings:
    """The inputs' values as the settings of the run, or the node's refusal for a value no input
    takes (a value a link gives is not held to an input's grid by ComfyUI)."""
    try:
        return plans.Settings.of(**values)
    except plans.BadSetting as error:
        raise words.Refused(f"{error}. Change the number connected to this node.") from error


def _validate(**values: int | None) -> bool | str:
    """validate_inputs of the upscale nodes: the numbers the prompt holds are checked when it is
    queued, before anything runs. A linked number is not there to check (the executor hands over
    the constants only) and is checked at the node's start; an unwired one is None. ComfyUI
    checks the min and max itself for the inputs this function does not take."""
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
                "Upscales each image of the batch on its own with seedvr2x. Each time it runs,"
                " seedvr2x works out how to fit the job on your graphics card, as if ComfyUI's"
                " other models were unloaded: the plan, shown on the node and given as the plan"
                " output. The optional numbers are the plan's: unconnected, it chooses them; a"
                " number connected is used exactly, and the plan is built around it."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    display_name="images",
                    tooltip="The images to upscale: each image of the batch on its own.",
                ),
                _models_input(),
                _resolution_input(),
                _attention_input(),
                *_correction_inputs(),
                _plan_only_input(),
                _tile_input("encode_tile", "encode"),
                _tile_input("decode_tile", "decode"),
                _swap_input(),
            ],
            outputs=[
                io.Image.Output(
                    "images",
                    display_name="images",
                    tooltip="The upscaled images, in the batch's order.",
                ),
                _plan_output(),
            ],
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
        images: torch.Tensor,
        models: models_module.Handle,
        resolution: int,
        attention: str,
        color_correction: bool,
        seed: int,
        plan_only: bool,
        encode_tile: int | None = None,
        decode_tile: int | None = None,
        swap_blocks: int | None = None,
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
        batch = _array(images)
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
        logger.info("seedvr2x: %d image(s) upscaled in %.1f s", count, time.monotonic() - started)
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
                "Upscales a clip's frames with seedvr2x as one continuous scene, in overlapping"
                " windows that fit your graphics card. Scene cuts are not detected, so a cut"
                " inside a window blends the two scenes, and a stopped run starts again;"
                " seedvr2x's command line handles both. Each time it runs, seedvr2x works out how"
                " to fit the job on your card: the plan, shown on the node and given as the plan"
                " output. The optional numbers are the plan's: unconnected, it chooses them; a"
                " number connected is used exactly, and the plan is built around it."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    display_name="images",
                    tooltip="The clip's frames, in order, upscaled as one continuous scene.",
                ),
                _models_input(),
                _resolution_input(),
                _attention_input(),
                *_correction_inputs(),
                _plan_only_input(),
                _window_input(),
                _tile_input("encode_tile", "encode"),
                _tile_input("decode_tile", "decode"),
                _swap_input(),
            ],
            outputs=[
                io.Image.Output(
                    "images",
                    display_name="images",
                    tooltip="The upscaled frames, in order, for ComfyUI's video nodes.",
                ),
                _plan_output(),
            ],
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
        images: torch.Tensor,
        models: models_module.Handle,
        resolution: int,
        attention: str,
        color_correction: bool,
        seed: int,
        plan_only: bool,
        window: int | None = None,
        encode_tile: int | None = None,
        decode_tile: int | None = None,
        swap_blocks: int | None = None,
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

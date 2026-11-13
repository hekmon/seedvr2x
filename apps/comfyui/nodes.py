"""seedvr2x's four ComfyUI nodes (DESIGN.md, Repository and apps, the ComfyUI nodes), in ComfyUI's
V3 schema (`comfy_api.latest`): a class a node, `define_schema` giving its `io.Schema` (id, display
name, category, inputs and outputs in order, hidden inputs) and a classmethod `execute` giving an
`io.NodeOutput`; and the three that give the plan's settings. `comfy_entrypoint` gives ComfyUI the
extension that lists them.

What the V3 schema changes for a node, read in ComfyUI v0.44.0:
- The executor calls `execute` on a clone of the class whose `hidden` holds what the schema asked
  for: the node's id (`io.Hidden.unique_id`) is `cls.hidden.unique_id`, None on the class itself.
- The inputs arrive by keyword, named as the schema names them; the outputs are
  `io.NodeOutput(...)` in the schema's order.
- The progress bar, the text on the node, the previews and Cancel are not part of the schema:
  `comfy.utils.ProgressBar` reports for the node being executed (the executor sets it as the
  current one), `PromptServer.send_progress_text` puts a text on it, and
  `comfy.model_management.throw_exception_if_processing_interrupted` raises when Cancel was
  pressed, as for any node. progress.py says how the nodes use them.

- seedvr2x Models: the DiT file by a name a user reads, whether the weights stay in RAM between
  runs (auto, on, off) and, as an optional input, a model folder (unwired: Hugging Face's cache,
  where what is missing is downloaded here, on a bar of the node's). Gives a light handle
  (models.Handle); no file is read here.
- seedvr2x Cuts: the shot detector on a clip's frames, its file the one seedvr2x Models provides,
  giving the cut list Upscale Video takes (an optional input there, recommended; unwired, the clip
  is one shot); the cuts and the possible cuts are said in ComfyUI's console.
- seedvr2x Upscale Image and seedvr2x Upscale Video: the pictures and the models' handle, then
  the settings as widgets (resolution, colour correction, seed, plan only), then the
  numbers the plan makes as optional inputs of seedvr2x's own types (each pass's tile, the blocks
  swapped, a clip's window length; the user's Desktop test, 2026-11-05): unwired, the plan
  chooses; wired, the value is taken exactly. No value stands for auto. The plan is made at run
  time, where the input's size and the card are known (models.session), said in ComfyUI's
  console, then the library's pass over the pictures or the frames in memory; a clip's, one pass
  for each shot of its cut list (an optional input of seedvr2x's own type, a clip's fact;
  unwired, the clip is one shot), the plan made once for the longest shot's windows. With "plan
  only" the node stops there: the plan is in the console, the node's first line says its main
  numbers and what follows its pictures is not run.

A node's text box is a status of two lines at most (progress.py), never a log: what a node has to
say besides goes to ComfyUI's console through the module's logger, and a refusal shows "refused:
see the error" on the node (_refusing), its text in ComfyUI's error and once in the console.

V3 classes hold no state (the executor runs `execute` on a clone of the class, and ignores
`__init__`): what the nodes keep between runs, the weights in RAM, is in models.py's registry.
Every use of `comfy_api` is in this module, which is the one place to change for another version
of it (`comfy_api.latest` is unstable).

Nothing here saves a file: the nodes give ComfyUI's image, float32, which its savers write, and so
carry no seedvr2x record (the CLI's 16-bit PNG has one)."""

import logging
import math
import time
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from fractions import Fraction
from typing import Any, cast, override

import comfy.model_management as management
import numpy as np
import numpy.typing as npt
import torch
from comfy.utils import ProgressBar
from comfy_api.latest import ComfyExtension, io
from comfy_execution.graph_utils import ExecutionBlocker

from seedvr2x import clip, frames, peaks, pull, shotcuts, still, still_plan, waiting
from seedvr2x import detector as shot_detector
from seedvr2x.clock import spoken, wall
from seedvr2x.errors import Refusal
from seedvr2x.memory import Reading
from seedvr2x.passes import Plan
from seedvr2x.shot_plan import HOST_RESIDENT, PHASES, Peaks
from seedvr2x.shotruns import QUALITY, shot_runs, taken_from
from seedvr2x.shots import Shot
from seedvr2x.weights import Architecture

from . import console, plans, progress, words
from . import models as models_module

logger = logging.getLogger("seedvr2x_comfyui.nodes")

CATEGORY = "seedvr2x"
# The value nodes in a subcategory of their own (the user, 2026-11-05): ComfyUI's node library
# nests a category at each "/".
SETTINGS_CATEGORY = f"{CATEGORY}/plan settings"

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
                "Chooses seedvr2x's model file and where the model files are read from: it finds"
                " the files and downloads what Hugging Face's cache lacks of them, the shot"
                " detector's file (about 30 MB, for seedvr2x Cuts) included (once, while this"
                " node's inputs stay as they are). It reads no weights into memory: the upscale"
                " nodes and seedvr2x Cuts read the files when they run, check them and fit them"
                " to your graphics card."
            ),
            inputs=[
                io.Combo.Input(
                    "model",
                    options=list(DIT_BY_LABEL),
                    default=plans.DIT_LABELS[plans.DEFAULT_DIT],
                    display_name=plans.LABELS["model"],
                    # The error per weight as models/hf/README.md defines it ("How to read it")
                    # and its table gives it, the video column the same table's ("On video").
                    tooltip="The model file, with its size and its error per weight: how far its"
                    " stored weights are from ByteDance's original, relative to their own size"
                    " (the median over the 7B's 288 large matrices, as the model card gives it)."
                    " It measures the file, not the picture: on video, the int8 file upscales as"
                    " close to float16 as a second seed does; the dynamic GGUF drifts further at"
                    " 1080p on some content, cartoons above all. Take float16 when the card has"
                    f" room for it, else the {_INT8} file, else the {_GGUF} one. Downloaded from"
                    " Hugging Face into its cache where missing, always the same version of the"
                    " file.",
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
                # An optional input and no widget (the user, 2026-11-08): an empty widget makes a
                # user think something must be filled in; an input drawn apart reads as optional
                # at once. Any string node gives it a folder, ComfyUI's own String node among them.
                io.String.Input(
                    "model_dir",
                    optional=True,
                    force_input=True,
                    display_name=plans.LABELS["model_dir"],
                    tooltip="Optional. A folder that has seedvr2x's model files under their"
                    f" names: the VAE, {plans.VAE_FILE} (which turns images into what the model"
                    " works on, and back), and the model file under seedvr2x's name for it:"
                    f" {plans.DEFAULT_DIT} for {plans.DIT_LABELS[plans.DEFAULT_DIT]};"
                    " _int8_convrot.safetensors or _dyn.gguf in place of _fp16.safetensors for"
                    " the int8 and dynamic GGUF files; no _sharp for the 7B. These are the names"
                    " seedvr2x's command line reads from a folder too; and the shot detector's"
                    f" file, {pull.DETECTOR}, which seedvr2x Cuts reads. With a folder every file"
                    " comes from it: nothing is downloaded and Hugging Face's cache is not read."
                    " Connect any string node to it, ComfyUI's String node for one. Not"
                    " connected: Hugging Face's cache, shared with seedvr2x's command line, where"
                    " the files are downloaded the first time they are needed.",
                ),
            ],
            outputs=[
                ModelsType.Output(
                    "models",
                    display_name="models",
                    tooltip="The model file chosen and where the files are read from, for"
                    " seedvr2x's upscale nodes and seedvr2x Cuts. Nothing is loaded yet.",
                )
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls, model: str, keep_in_ram_between_runs: str, model_dir: str | None = None
    ) -> io.NodeOutput:
        """The handle, once its files are there. The node's text (no name: the title has it) is
        the download's bytes over its time while one runs, then one line, the file and whether it
        is kept in RAM; the files' places go to the console."""
        show = _shower(_node_id(cls))
        with _refusing(show):
            handle = models_module.Handle(
                DIT_BY_LABEL[model], models_module.model_folder(model_dir), keep_in_ram_between_runs
            )
            models_module.provide(handle, show)
            kept = "kept" if models_module.decide_keeping(handle).kept else "not kept"
            show(f"{plans.short_label(handle.dit)} \u00b7 {kept} in RAM")
            return io.NodeOutput(handle)


# ---- What the upscale nodes share ----------------------------------------------------------------


def _array(images: torch.Tensor) -> npt.NDArray[np.float32]:
    """ComfyUI's IMAGE, (B, H, W, C) float32 in [0, 1], as the library takes frames: (B, H, W, 3)
    float32: a contiguous CPU float32 tensor is shared, not copied, and four channels with an
    alpha opaque everywhere are the first three as a view of the same memory, not a copy of a long
    clip that no figure of the plan counts (the library only reads its frames and takes strided
    ones: model.to_input is torch.from_numpy on them). Any alpha that is not opaque is refused
    (picture.py's rule: the model would redraw the colours under it)."""
    if images.ndim != 4 or images.shape[0] < 1:
        raise words.frames_missing()
    array = np.ascontiguousarray(images.detach().to("cpu", torch.float32).numpy())
    count = array.shape[3]
    if count == 4:
        if bool((array[..., 3] < 1.0).any()):
            raise words.transparent()
        logger.info("the alpha channel is opaque everywhere, and left out")
        array = array[..., :3]
    elif count != 3:
        raise words.channels(count)
    return array


def _target(width: int, height: int, resolution: int, *, picture: bool) -> tuple[int, int]:
    """(height, width) the input is resized to: the output's short side at `resolution`, square
    pixels (frames.target_size), refused where the padding can't be made on it: on the target for
    a clip, on the output's even crop too for a picture (frames.check_picture_target)."""
    target = frames.target_size(width, height, Fraction(1), resolution)
    try:
        if picture:
            frames.check_picture_target(target, resolution)
        else:
            frames.check_target(target, resolution)
    except Refusal as error:
        raise words.refusal(error) from error
    return target


def _show(text: str, node: str | None) -> None:
    """`text` on the node (ComfyUI's progress text, which its frontend shows on any node, replaced
    by the next)."""
    import server

    # ComfyUI's PromptServer has no `instance` until its server is made (server.py sets it in
    # __init__): a node run without one (a script, the smoke test) says nothing rather than fail
    # with an AttributeError that would hide the node's own error.
    instance = getattr(server.PromptServer, "instance", None)
    if instance is not None and node is not None:
        instance.send_progress_text(text, node)


@contextmanager
def _refusing(show: Callable[[str], None]) -> Generator[None]:
    """The body of a node's execute: a refusal (words.Refused) it raises puts the refusal's short
    line on the node (`line`: "refused: see the error", or the card's state) through `show`, and
    its text in the console once, at WARNING (the model session says a failed card at ERROR
    already: `logged`). The exception goes on to ComfyUI, whose error dialog has the text."""
    try:
        yield
    except words.Refused as error:
        if not error.logged:
            console.say(logger, logging.WARNING, f"refused: {error}")
            error.logged = True
        show(error.line)
        raise
    except BaseException:
        # Anything else (an error, a Cancel, out of memory) before the run's first line of its
        # own: no "planning…" left beside ComfyUI's error. A later line stays: where it stopped.
        if getattr(show, "last", None) == progress.PLANNING:
            show("")
        raise


class Shower:
    """What puts a progress text on the node `node` (_show), and keeps the last one (`last`), so
    that a text said for a moment (models._freeing) can be replaced by the one before it."""

    def __init__(self, node: str | None) -> None:
        self.node = node
        self.last: str | None = None

    def __call__(self, text: str) -> None:
        self.last = text
        _show(text, self.node)


def _shower(node: str | None) -> Callable[[str], None]:
    """What puts a progress text on the node `node` (Shower)."""
    return Shower(node)


def _node_id(node: type[io.ComfyNode]) -> str | None:
    """The id of the node being executed (the hidden UNIQUE_ID the schema asks for), None where
    the class is not the executor's clone of it."""
    return None if node.hidden is None else node.hidden.unique_id


def _output(array: npt.NDArray[np.float32]) -> torch.Tensor:
    return torch.from_numpy(array)  # pyright: ignore[reportUnknownMemberType]


def _checked(handle: models_module.Handle, node: str = "seedvr2x Upscale") -> None:
    if not isinstance(handle, models_module.Handle):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"{node} takes the output of seedvr2x Models")


# ---- The settings an upscale node carries --------------------------------------------------------

# The widgets are the choices with a default the user reads (resolution, colour
# correction, seed, plan only). The numbers the plan makes are optional sockets of seedvr2x's own
# types (the user, 2026-11-05), each given by its value node below (seedvr2x Tile, Blocks Swapped,
# Window Length): a link dragged from the socket and released on the canvas makes ComfyUI offer
# that node, which carries its own range, step and description. Unconnected, the plan chooses,
# and its text says what; connected, the value is taken exactly. A socket carries the library's
# pin (plans.py), and a value no value node gives is refused at the node's start, before anything
# loads.

TileType = io.Custom("SEEDVR2X_TILE")
SwapType = io.Custom("SEEDVR2X_BLOCKS_SWAPPED")
WindowType = io.Custom("SEEDVR2X_WINDOW")


def _tile_input(name: str, pass_: str) -> Any:
    return TileType.Input(
        name,
        display_name=plans.LABELS[name],
        optional=True,
        tooltip=f"Optional: connect a seedvr2x Tile node to set the tile size of the VAE {pass_}."
        " Unconnected, the plan chooses: the whole image at once if it fits the card, else the"
        " largest tile that does.",
    )


def _swap_input() -> Any:
    return SwapType.Input(
        "swap_blocks",
        display_name=plans.LABELS["swap_blocks"],
        optional=True,
        tooltip="Optional: connect a seedvr2x Blocks Swapped node to set how many of the model's"
        " blocks wait in RAM and pass through the card in turn. Unconnected, the plan chooses the"
        " fewest that fit.",
    )


def _window_input() -> Any:
    return WindowType.Input(
        "window",
        display_name=plans.LABELS["window"],
        optional=True,
        tooltip="Optional: connect a seedvr2x Window Length node to set the longest stretch of"
        " the clip the model works on at once. Unconnected, the plan chooses the longest that"
        " fits the card.",
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
            # ComfyUI's own seed widgets' range (KSampler, nodes.py), so that a generation's seed
            # typed here or shared through a primitive node is taken; plans.Settings.of folds it.
            max=0xFFFFFFFFFFFFFFFF,
            control_after_generate=io.ControlAfterGenerate.fixed,
            display_name=plans.LABELS["seed"],
            tooltip="The seed of the model's noise: the same input, settings and seed give the"
            " same image on the same computer. Fixed after each run by default, so that running"
            " again gives the same result. Any whole number is taken, a generation's seed"
            f" included: a seed outside seedvr2x's range, 0 to {plans.SEED_MAX:,}, is taken modulo"
            f" {plans.SEEDS:,}, and the plan in ComfyUI's console says the seed used.",
        ),
    ]


def _quality_input() -> Any:
    return io.Combo.Input(
        "quality",
        options=list(QUALITY),
        default=plans.QUALITY_DEFAULT,
        display_name=plans.LABELS["quality"],
        # MANUAL.md, Several runs of a shot: 2 or 4 times the time, no more memory; sharper lines
        # and less flicker where there is motion, little change on a still shot. The pulse at
        # standard (DESIGN.md, Open questions, "Every fourth frame is the model's best": a group
        # of 4's last frame closest to the original on 8 of 8 clips by VMAF and LPIPS, 0.8 to 5.0
        # dB of PSNR-Y above its second on 7 of 8; less sharp by the luma's Laplacian variance on
        # 6 of 8, a measure the user's eyes contradict), so what was measured (closer) and what
        # the eye sees (sharper: the user, stepping a dark animation, 2026-11-08) are said apart,
        # neither from the other; what enhanced does to it is unmeasured, so it is not said.
        tooltip="How many times the model runs over each shot. standard: once. The model packs a"
        " shot's frames in groups of 4 and rebuilds the last of each group closest to the"
        " original, so at standard one frame in four comes out closer to the original than the"
        " three before it, and looks sharper: edges that pulse, most on line art, dark scenes and"
        " motion. enhanced: twice, and maximum: four times, each run starting a frame or two"
        " further into the shot, each frame kept from the run that rebuilds it best; with maximum,"
        " a run where it is rebuilt closest to the original."
        " The result has sharper lines and less flicker where there is motion, and changes little"
        " on a still shot. It takes 2 (enhanced) or 4 (maximum) times as long as standard, and no"
        " more memory.",
    )


def _plan_only_input() -> Any:
    return io.Boolean.Input(
        "plan_only",
        default=False,
        display_name=plans.LABELS["plan_only"],
        # "no model is loaded" read as seedvr2x Models' work undone (the user's third test,
        # 2026-11-08): what is so, the files found by Models and their weights read by this node.
        tooltip="Make the plan, say it in ComfyUI's console, and stop: the model's weights are"
        " not read into memory (seedvr2x Models finds the files and downloads the missing ones;"
        " this node reads them when it runs, after its plan), ComfyUI's loaded models stay as"
        " they are, and the nodes connected to the images output are skipped.",
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


def _models_input(
    tooltip: str = "The output of seedvr2x Models: the model file and where it is read from.",
) -> Any:
    return ModelsType.Input("models", display_name="models", tooltip=tooltip)


def _settings(**values: Any) -> plans.Settings:
    """The inputs' values as the settings of the run, or the node's refusal for a value no input
    takes (a value a link gives is not held to an input's grid by ComfyUI)."""
    try:
        return plans.Settings.of(**values)
    except plans.BadSetting as error:
        raise words.Refused(f"{error}.") from error


def _plan_only() -> io.NodeOutput:
    """What a plan only gives: no picture. The picture output is an execution blocker with no
    message, which the executor passes to every node that takes it (silently, as a node that has
    nothing to give); a message would be reported as an error."""
    return io.NodeOutput(ExecutionBlocker(None))


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}{'' if number == 1 else 's'}"


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
        reading: Reading, host: int | None, architecture: Architecture, held: bool
    ) -> models_module.Planned[PicturePlan]:
        made = still_plan.plan(
            bound,
            reading,
            host,
            process=process,
            input_pixels=height * width,
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
                " other models were unloaded: the plan, said in ComfyUI's console, while the node"
                " shows where the run is. The optional numbers are the plan's: unconnected, it"
                " chooses them; a number connected is used exactly, and the plan is built around"
                " it."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    display_name="images",
                    tooltip="The images to upscale: each image of the batch on its own.",
                ),
                _models_input(),
                _resolution_input(),
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
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        images: torch.Tensor,
        models: models_module.Handle,
        resolution: int,
        color_correction: bool,
        seed: int,
        plan_only: bool,
        encode_tile: int | None = None,
        decode_tile: int | None = None,
        swap_blocks: int | None = None,
    ) -> io.NodeOutput:
        show = _shower(_node_id(cls))
        with _refusing(show):
            show(progress.PLANNING)
            _checked(models)
            settings = _settings(
                encode_tile=encode_tile,
                decode_tile=decode_tile,
                swap_blocks=swap_blocks,
                color_correction=color_correction,
                seed=seed,
            )
            batch = _array(images)
            count, height, width = batch.shape[:3]
            target = _target(width, height, resolution, picture=True)
            out_height, out_width = frames.output_size(target)
            planner = picture_planner(settings, target, (height, width), count)

            started = time.monotonic()
            with models_module.session(
                models,
                planner,
                plan_only=plan_only,
                preparing=progress.Preparing(show, plans.file_label),
            ) as run:
                made, tiles = run.planned.plan
                console.say(
                    logger,
                    logging.INFO,
                    plans.picture_text(
                        count=count,
                        source=(width, height),
                        output=(out_width, out_height),
                        resolution=resolution,
                        dit=models.dit,
                        plan=made,
                        tiles=tiles,
                        room=run.room,
                        settings=settings,
                        keeping=run.keeping,
                        notes=run.notes,
                        held=run.held,
                    ),
                )
                if plan_only:
                    show(
                        progress.status(
                            plans.picture_line(tiles, made.models.swapped), progress.IN_THE_CONSOLE
                        )
                    )
                    return _plan_only()
                # The bar over the batch's expected time (progress.Progress), the text and a preview
                # a picture.
                moving = progress.Progress(
                    progress.picture_passes(count, target, tiles.encode, tiles.decode),
                    progress.PICTURE,
                    ProgressBar(progress.BAR),
                    show,
                )
                # The batch's output, allocated once and written into picture by picture: a list of
                # the pictures stacked after would hold them twice (the plan counts it once,
                # picture_process).
                outputs = np.empty((count, out_height, out_width, 3), dtype=np.float32)
                # The time said each second, Cancel the waits' check and the units named for them
                # (progress.Progress.running), stopped however the run ends.
                with moving.running():
                    try:
                        for index in range(count):
                            moving.begin(index)
                            outputs[index] = cls._one(
                                run.models, batch[index], target, settings, tiles, moving
                            )
                    except still.NonFinite as error:
                        raise words.non_finite(str(error)) from error
                    moving.finish(_count(count, "picture"))
            took = spoken(wall(time.monotonic() - started))
            logger.info("%s upscaled in %s", _count(count, "picture"), took)
            return io.NodeOutput(_output(outputs))

    @staticmethod
    def _one(
        loaded: Any,
        picture: npt.NDArray[np.float32],
        target: tuple[int, int],
        settings: plans.Settings,
        tiles: still_plan.PictureTiles,
        moving: progress.Progress,
    ) -> npt.NDArray[np.float32]:
        """One picture through the three steps of still.upscale, the bar, the text and ComfyUI's
        Cancel moved between them and, for a tiled pass, between its tiles: the steps are the
        library's own (still.upscale composes them), and what is told them only observes."""
        latent = still.encoded(
            loaded,
            picture,
            target,
            settings.seed,
            tiles=tiles.encode,
            tile=lambda done, total: moving.tile("encode", done, total),
        )
        moving.encoded()
        latent = still.sampled(loaded, latent, settings.seed)
        moving.window(1)
        out = still.decoded(
            loaded,
            latent,
            picture,
            target,
            tiles=tiles.decode,
            correct=settings.split,
            tile=lambda done, total: moving.tile("decode", done, total),
        )
        moving.decoded(1, 1, out)
        return out


# ---- seedvr2x Upscale Video ----------------------------------------------------------------------


type ClipPlan = tuple[Plan, Peaks]

# The cut list Upscale Video takes, seedvr2x Cuts gives: plans.CutList, the clip's shots and the
# text of the cuts, which ComfyUI's Preview as Text node (PreviewAny) shows.
CutsType = io.Custom("SEEDVR2X_CUTS")


def clip_planner(
    settings: plans.Settings, target: tuple[int, int], cuts: Sequence[Shot]
) -> models_module.Planner[ClipPlan]:
    """The planner of a clip cut into `cuts` and resized to `target`: clip.plan with the settings'
    pins and colour correction, the plan made once for the longest shot's windows, and what it asks
    of the card, its largest phase (Peaks.need: a VAE pass with the room it is held to)."""

    def planner(
        reading: Reading, host: int | None, architecture: Architecture, held: bool
    ) -> models_module.Planned[ClipPlan]:
        made, expected = clip.plan(
            reading,
            target,
            cuts,
            settings.clip_pins(),
            host,
            correct=settings.split,
            dit_model=architecture,
            held=held,
        )
        peak = max(expected.need(phase) for phase in PHASES)
        needs = models_module.Needs(peak, expected.free - expected.budget, expected.reading)
        return models_module.Planned((made, expected), made.model_plan, needs)

    return planner


def _cut_list(cuts: object, frames_: int) -> tuple[Shot, ...]:
    """The cut list an input gave, held to the clip: shots from frame 0 to the clip's last, one
    after the other with none between and none empty. Another list is for another clip. The input
    is seedvr2x Cuts' plans.CutList, whose shots are taken; a tuple of shots, as the node took
    before it, is taken as it is."""
    if isinstance(cuts, plans.CutList):
        cuts = cuts.shots
    if not isinstance(cuts, tuple) or not all(isinstance(each, Shot) for each in cuts):  # pyright: ignore[reportUnknownVariableType]
        raise words.cuts_not_a_list()
    found = cast(tuple[Shot, ...], cuts)
    place = 0
    for each in found:
        if each.start != place or each.end <= each.start:
            raise words.cuts_wrong(sum(shot.frames for shot in found), frames_)
        place = each.end
    if place != frames_:
        raise words.cuts_wrong(place, frames_)
    return found


def _check_seed(settings: plans.Settings, shots: tuple[Shot, ...]) -> None:
    """Refuse a seed some shot would seed past numz's range, before anything loads: the largest
    seed used (plans.SEED_MAX) is the one of a shot at frame 0, and a later shot adds its first
    frame (job.check_seed's rule on the command line, which this node can't import). The seed is
    the one used, a seed past the range folded into it (plans.Settings.of), the one given named."""
    last = shots[-1].start
    if settings.seed + last > plans.SEED_MAX:
        raise words.seed_past_the_cuts(
            settings.seed, last, plans.SEED_MAX - last, given=settings.given_seed
        )


class UpscaleVideo(io.ComfyNode):
    """The frames of a clip, shot by shot, each in sliding windows."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="seedvr2xUpscaleVideo",
            display_name="seedvr2x Upscale Video",
            category=CATEGORY,
            description=(
                "Upscales a clip's frames with seedvr2x, each shot of the cut list on its own, in"
                " overlapping windows that fit your graphics card: no window crosses a cut, so no"
                " two scenes blend. The cuts input takes the output of seedvr2x Cuts, which finds"
                " the cuts (recommended); unwired, the clip is upscaled as one shot. The clip is"
                " kept whole in your computer's RAM, its frames and its output, and a run stopped"
                " part way starts over from its first frame. For a clip that doesn't fit in RAM,"
                " use seedvr2x's command line: it streams the clip from its source file shot by"
                " shot, never holding it whole, so that a clip of any length fits, and resumes"
                " after a stop."
                " Each time it runs, seedvr2x works out how to fit the job on your card: the"
                " plan, said shot by shot in ComfyUI's console, while the node shows where the"
                " run is. The optional numbers are the plan's: unconnected, it chooses them; a"
                " number connected is used exactly, and the plan is built around it."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    display_name="images",
                    tooltip="The clip's frames, in order.",
                ),
                _models_input(),
                _resolution_input(),
                _quality_input(),
                *_correction_inputs(),
                _plan_only_input(),
                # The first optional socket (the user, 2026-11-08): optional because a clip of
                # one scene needs no detector, recommended because frames mixed across a cut
                # blend two scenes.
                CutsType.Input(
                    "cuts",
                    optional=True,
                    display_name="cuts",
                    tooltip="Recommended: the shots from seedvr2x Cuts. Unwired, the clip is"
                    " upscaled as one shot, and a clip with cuts would have frames mixed across"
                    " them.",
                ),
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
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        images: torch.Tensor,
        models: models_module.Handle,
        resolution: int,
        color_correction: bool,
        seed: int,
        plan_only: bool,
        quality: str = plans.QUALITY_DEFAULT,
        cuts: plans.CutList | tuple[Shot, ...] | None = None,
        window: int | None = None,
        encode_tile: int | None = None,
        decode_tile: int | None = None,
        swap_blocks: int | None = None,
    ) -> io.NodeOutput:
        show = _shower(_node_id(cls))
        with _refusing(show):
            show(progress.PLANNING)
            _checked(models)
            if quality not in QUALITY:
                # A combo is held to its options at queue time; a link may carry anything.
                raise words.Refused(
                    f"{plans.LABELS['quality']} {quality!r}: must be one of {', '.join(QUALITY)}."
                )
            settings = _settings(
                window=window,
                encode_tile=encode_tile,
                decode_tile=decode_tile,
                swap_blocks=swap_blocks,
                color_correction=color_correction,
                seed=seed,
            )
            clip_frames = _array(images)
            count, height, width = clip_frames.shape[:3]
            if cuts is None:
                logger.info(
                    "no cut list: the clip is upscaled as one shot (wire seedvr2x Cuts for a clip"
                    " with several shots)"
                )
                shot_list = (Shot(0, count),)
            else:
                shot_list = _cut_list(cuts, count)
            _check_seed(settings, shot_list)
            target = _target(width, height, resolution, picture=False)
            out_height, out_width = frames.output_size(target)

            planner = clip_planner(settings, target, shot_list)
            size = words.ClipSize(count, clip.output_bytes(count, target))

            started = time.monotonic()
            with models_module.session(
                models,
                planner,
                plan_only=plan_only,
                clip=size,
                preparing=progress.Preparing(show, plans.file_label),
            ) as run:
                made, expected = run.planned.plan
                console.say(
                    logger,
                    logging.INFO,
                    plans.clip_text(
                        cuts=shot_list,
                        source=(width, height),
                        output=(out_width, out_height),
                        resolution=resolution,
                        dit=models.dit,
                        plan=made,
                        expected=expected,
                        room=run.room,
                        settings=settings,
                        keeping=run.keeping,
                        notes=run.notes,
                        held=run.held,
                        quality=quality,
                    ),
                )
                if plan_only:
                    show(progress.status(plans.clip_line(made), progress.IN_THE_CONSOLE))
                    return _plan_only()
                moving = progress.Progress(
                    progress.clip_passes(
                        shot_list,
                        target,
                        made.window,
                        made.encode_tiles,
                        made.decode_tiles,
                        QUALITY[quality],
                    ),
                    progress.CLIP,
                    ProgressBar(progress.BAR),
                    show,
                )
                # The clip's one output array, each shot written into its place in it (the plan
                # counts it once, clip.output_bytes), the shots one after the other: the windows
                # of a shot never reach another's frames.
                out = np.empty((count, out_height, out_width, 3), dtype=np.float32)
                # The time said each second, Cancel the waits' check and the units named for them
                # (progress.Progress.running), stopped however the run ends.
                with moving.running():
                    for number, each in enumerate(shot_list, 1):
                        if len(shot_list) > 1:
                            logger.info(
                                "shot %d of %d, frames %d to %d",
                                number,
                                len(shot_list),
                                each.start,
                                each.end - 1,
                            )
                        cls._shot(
                            *(run.models, clip_frames, each, target, settings, made, moving, out),
                            *(quality, number),
                        )
                    moving.finish(_count(count, "frame"))
            took = spoken(wall(time.monotonic() - started))
            logger.info("%s upscaled in %s", _count(count, "frame"), took)
            return io.NodeOutput(_output(out))

    @staticmethod
    def _shot(
        loaded: Any,
        clip_frames: npt.NDArray[np.float32],
        shot: Shot,
        target: tuple[int, int],
        settings: plans.Settings,
        made: Plan,
        moving: progress.Progress,
        out: npt.NDArray[np.float32],
        quality: str = plans.QUALITY_DEFAULT,
        number: int = 1,
    ) -> None:
        """One shot of the clip, `clip_frames` [start, end) of the (T, H, W, 3) float32 in [0, 1]
        clip, through the library's clip pass at `quality` in the plan's windows and tiles, written
        into the same frames of `out`. The bar, the text, the preview (the newest frame written, a
        view of `out`) and ComfyUI's Cancel follow the pass's steps and its tiles, and each run's
        start (several at enhanced and maximum; the shot is its number `number`, from 1, of the
        clip's): what is told the library only observes. A shot of one run begins here; one of
        several, as each run starts."""
        level = QUALITY[quality]
        offsets = [each.offset for each in shot_runs(Shot(0, shot.frames), level)]
        taken = taken_from(shot.frames, level)
        if len(offsets) == 1:
            moving.run(number - 1, 1)
        run = 0

        def step(stage: clip.Stage, done: int, count: int) -> None:
            nonlocal run
            if stage == "run":
                run = done - 1
                moving.run(number - 1, done)
            elif stage == "encode":
                moving.encoded()
            elif stage == "window":
                moving.window(done)
            else:
                # The preview is the newest frame decoded so far that this run keeps (the clip's
                # frames offset to offset + done - 1): the others' are dropped from `out`, where
                # a run's slice may hold none of its own. Its first frame is always its own.
                offset = offsets[run]
                newest_first = range(offset + done - 1, offset - 1, -1)
                kept = next(i for i in newest_first if taken[i] == offset)
                moving.decoded(done, count, out[shot.start + kept])

        try:
            clip.upscale(
                loaded,
                clip_frames[shot.start : shot.end],
                target,
                settings.seed,
                window=made.window,
                encode_tiles=made.encode_tiles,
                decode_tiles=made.decode_tiles,
                tail_frames=made.tail_frames,
                correct=settings.split,
                step=step,
                start=shot.start,
                output=out[shot.start : shot.end],
                quality=quality,
                tile=moving.tile,
            )
        except clip.NonFinite as error:
            raise words.non_finite(str(error)) from error


# ---- seedvr2x Cuts -------------------------------------------------------------------------------

THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP = 0.05, 1.0, 0.05


class Cuts(io.ComfyNode):
    """A clip's cuts: the shot detector on its frames."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="seedvr2xCuts",
            display_name="seedvr2x Cuts",
            category=CATEGORY,
            description=(
                "Finds where a clip changes scene, for seedvr2x Upscale Video, which upscales each"
                " shot on its own so that no two scenes blend. seedvr2x's shot detector, TransNetV2"
                " (about 30 MB, provided by seedvr2x Models, which takes it from its model folder"
                " or Hugging Face's cache), runs on your graphics card while the node runs, and on"
                " the processor without one; frames are shrunk to its small input in torch, with"
                " no ffmpeg. It gives the cut list, which ComfyUI's Preview as Text shows one line"
                " a cut, timed where the clip's frame rate is connected; ComfyUI's console has"
                " each cut's probability, and the possible cuts (scenes it is less sure of, to"
                " check by eye)."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    display_name="images",
                    tooltip="The clip's frames, in order: the same images that go to seedvr2x"
                    " Upscale Video.",
                ),
                _models_input(
                    "The output of seedvr2x Models: it gives the shot detector's file (and the"
                    " model file, which this node does not use)."
                ),
                io.Float.Input(
                    "threshold",
                    default=shotcuts.THRESHOLD,
                    min=THRESHOLD_MIN,
                    max=THRESHOLD_MAX,
                    step=THRESHOLD_STEP,
                    round=False,
                    display_name="threshold",
                    tooltip="How sure the detector must be that a frame starts a new shot, from"
                    f" {THRESHOLD_MIN:g} to {THRESHOLD_MAX:g}. The default, {shotcuts.THRESHOLD:g},"
                    " finds nearly every cut in live action and about 9 in 10 in animation, with"
                    " a few false ones, which cost little: a shot cut for nothing only gets its"
                    " own windows. A lower value finds more cuts, with more false ones.",
                ),
                # An optional input and no widget, as Models' folder: ComfyUI's images carry no
                # frame rate, and the one the clip has comes from the node that loaded it.
                io.Float.Input(
                    "frame_rate",
                    optional=True,
                    force_input=True,
                    display_name=plans.LABELS["frame_rate"],
                    tooltip="Optional. The clip's frames per second, for the time of each cut in"
                    " the cut list: connect the fps output of ComfyUI's Get Video Components, the"
                    " node that gives these images. Not connected: each cut is given by its frame"
                    " alone.",
                ),
            ],
            outputs=[
                CutsType.Output(
                    "cuts",
                    display_name="cuts",
                    tooltip="The clip's shots, for seedvr2x Upscale Video's cuts input. Wired to"
                    " ComfyUI's Preview as Text node (PreviewAny) too, it shows the cut list, a"
                    " line a cut: its time, with the frame rate connected, and the frame it starts"
                    " on, counted from 0 (Cut #1 - 0:00:01.440 (frame 36)). The console says each"
                    " cut's probability and the possible cuts, to check by eye.",
                ),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        images: torch.Tensor,
        models: models_module.Handle,
        threshold: float,
        frame_rate: float | None = None,
    ) -> io.NodeOutput:
        show = _shower(_node_id(cls))
        with _refusing(show):
            _checked(models, words.CUTS)
            if images.ndim != 4 or images.shape[0] < 1:
                raise words.frames_missing()
            if images.shape[3] not in (3, 4):
                raise words.channels(int(images.shape[3]))
            count = int(images.shape[0])
            if isinstance(threshold, bool) or not isinstance(threshold, int | float):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise words.Refused(f"threshold {threshold!r}: must be a number")
            if not 0 < threshold <= 1:
                raise words.Refused(f"threshold {threshold}: must be above 0 and at most 1")
            if frame_rate is not None and (
                isinstance(frame_rate, bool)
                or not isinstance(frame_rate, int | float)  # pyright: ignore[reportUnnecessaryIsInstance]
                or not math.isfinite(frame_rate)
                or frame_rate <= 0
            ):
                raise words.frame_rate_wrong(frame_rate)
            found = cls._probabilities(images, models, show)
            cuts = shotcuts.detected(found, float(threshold))
            maybe = shotcuts.possible(found, float(threshold))
            text = plans.cuts_text(count, float(threshold), cuts, maybe)
            console.say(logger, logging.INFO, text)
            show(plans.cuts_line(len(cuts) + 1, len(maybe), float(threshold)))
            starts = [0, *(each.frame for each in cuts)]
            cut_list = tuple(
                Shot(start, end) for start, end in zip(starts, [*starts[1:], count], strict=True)
            )
            return io.NodeOutput(
                plans.CutList(
                    cut_list,
                    plans.cut_lines(cuts, None if frame_rate is None else float(frame_rate)),
                )
            )

    @staticmethod
    def _probabilities(
        images: torch.Tensor, handle: models_module.Handle, show: Callable[[str], None]
    ) -> npt.NDArray[np.float32]:
        """The detector's probability for each frame, (T,) float32: its file the one seedvr2x Models
        provided: with a model folder, the folder's alone, refused where the folder lacks it
        (words.detector_missing: nothing fetched, the cache not read, a folder being the user's
        choice of neither); without, Hugging Face's cache's, fetched again, on the node's bar and
        text, where the cache has lost it; checked against its pin, loaded onto the card ComfyUI
        runs on (the processor where that is not a card), run (the frames scanned on a bar of their
        own, a batch at a time, and ComfyUI's Cancel honoured between batches) and given back. Three
        channels of the images: an alpha is of no use to it. On a card a CUDA error failed
        (waiting.fail) it is refused at its start, as the upscale nodes are; one it meets marks the
        card, and nothing is emptied from torch's cache after it (the detector keeps its model:
        detector.Detector.close)."""
        device = management.get_torch_device()
        card = (device.index or 0) if getattr(device, "type", "") == "cuda" else 0
        failed = waiting.failed(card)
        if failed is not None:
            raise words.card_failed(failed, earlier=True)
        directory = handle.directory
        name, pin = pull.DETECTOR, pull.PINNED[pull.DETECTOR]
        hashes = pull.Hashes()
        if directory is not None and not (directory / name).is_file():
            # Not in the folder, from the start or gone since Models ran: never fetched, the
            # cache never read (the user, 2026-11-08: a folder is the choice of no cache and no
            # download, however it got its files).
            raise words.detector_missing(directory)
        try:
            if directory is None:
                if pull.cached(name) is None:
                    # Models fetched it when it ran, or could not (models.provide), and ComfyUI
                    # keeps its output between prompts (the same words as models._prepare's for
                    # the DiT and the VAE).
                    logger.info(
                        "%s is not in Hugging Face's cache: downloading it", plans.short_label(name)
                    )
                preparing = progress.Preparing(show, plans.file_label)
                path = pull.fetch({name: pin}, progress=preparing.download)[name]
                management.throw_exception_if_processing_interrupted()
            else:
                path = directory / name
            cache = None if directory is not None else pull.cache()
            pull.check_pin(name, path, pin, hashes, cache, "detector", load=True)
            if getattr(device, "type", "") != "cuda" or not torch.cuda.is_available():
                device = torch.device("cpu")
            # ComfyUI's Cancel the check of the detector's waits on the card, its close's among
            # them (detector.Detector.close waits through the poll): a card that stops answering
            # there is left by Cancel, the model kept, as the upscale nodes' waits are.
            with (
                waiting.checking(management.throw_exception_if_processing_interrupted),
                shot_detector.Detector(path, device, hashes=hashes) as found,
            ):
                return shot_detector.probabilities(
                    found,
                    images[..., :3].detach().to("cpu", torch.float32),
                    progress.scanned(ProgressBar(int(images.shape[0])), show),
                )
        except Refusal as error:
            raise words.refusal(error, model_dir=directory is not None) from error
        except BaseException as error:
            if not waiting.cuda_failed(error):
                raise
            # The error and its frames kept for the process's life (models._failures), and worded.
            waiting.fail(card, error)
            models_module._failures.append(error)  # pyright: ignore[reportPrivateUsage]
            raise words.card_failed(waiting.failed(card) or str(error)) from error
        finally:
            if waiting.failed(card) is None:
                management.soft_empty_cache()


# ---- The value nodes: a setting of the plan's, its range, step and description -------------------


_LINKED = "Change the number connected to this node."
_FROM_INT = (
    " The number can also come from another node, an Int node for one, through the widget's"
    " input socket."
)


def _value(problem: str | None) -> None:
    """A value node's refusal of a value a link gave it off its grid, in the nodes' words (a
    value the prompt holds is refused when it is queued, validate_inputs)."""
    if problem is not None:
        raise words.Refused(f"{problem}. {_LINKED}")


def _queued(problem: str | None) -> bool | str:
    """validate_inputs of a value node: True, or the reason a value the prompt holds is off the
    node's grid. ComfyUI's own check of min and max is left out for the inputs it names, and
    ComfyUI checks no step: this one checks both."""
    return True if problem is None else problem


class Tile(io.ComfyNode):
    """A tile for the VAE's passes: the value of an upscale node's encode or decode tile."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="seedvr2xTile",
            display_name="seedvr2x Tile",
            category=SETTINGS_CATEGORY,
            description=(
                "A tile size for seedvr2x's upscale nodes: connect it to their encode tile or"
                " decode tile input. The VAE turns the image into what the model works on (the"
                " encode) and back (the decode) in square tiles of this many output pixels,"
                " blended at their seams. A larger tile is faster and takes more of the graphics"
                " card; a smaller one leaves more room for the model, and has more seams to"
                " blend. Whole image at once: no tiles, the fastest, and the most memory. Leave"
                " the upscale node's input unconnected and the plan chooses for you: the whole"
                " image at once if it fits the card, else the largest tile that does." + _FROM_INT
            ),
            inputs=[
                io.Int.Input(
                    "tile_size",
                    default=plans.TILE_DEFAULT,
                    min=peaks.TILE_MIN,
                    max=peaks.TILE_MAX,
                    step=peaks.TILE_STEP,
                    display_name="tile size (px)",
                    tooltip="The side of a square tile, in pixels of the output:"
                    f" {peaks.TILE_MIN} to {peaks.TILE_MAX} in steps of {peaks.TILE_STEP}. Not"
                    " used when whole image at once is on.",
                ),
                io.Boolean.Input(
                    "whole_image",
                    default=False,
                    display_name="whole image at once",
                    tooltip="On: no tiles, the whole image in one pass; the tile size is not used.",
                ),
            ],
            outputs=[
                TileType.Output(
                    "tile",
                    display_name="tile",
                    tooltip="The tile, for an upscale node's encode tile or decode tile input.",
                )
            ],
        )

    @classmethod
    def validate_inputs(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls, tile_size: int | None = None
    ) -> bool | str:
        return _queued(None if tile_size is None else plans.tile_size_problem(tile_size))

    @classmethod
    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls, tile_size: int, whole_image: bool
    ) -> io.NodeOutput:
        _value(plans.tile_size_problem(tile_size))
        if not isinstance(whole_image, bool):  # pyright: ignore[reportUnnecessaryIsInstance]
            _value(f"whole image at once {whole_image!r}: must be on or off")
        return io.NodeOutput(plans.tile_pin(tile_size, whole_image))


class BlocksSwapped(io.ComfyNode):
    """The DiT's blocks swapped: the value of an upscale node's blocks swapped."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="seedvr2xBlocksSwapped",
            display_name="seedvr2x Blocks Swapped",
            category=SETTINGS_CATEGORY,
            description=(
                f"How many of the model's {peaks.BLOCKS} blocks wait in your computer's RAM and"
                " pass through the graphics card in turn, for seedvr2x's upscale nodes: connect it"
                " to their blocks swapped input. 0 keeps the whole model on the card, the fastest;"
                " each block swapped leaves the card more room, so that the model fits a smaller"
                " card, and slows the run. Leave the upscale node's input unconnected and the plan"
                " chooses the fewest that fit." + _FROM_INT
            ),
            inputs=[
                io.Int.Input(
                    "blocks",
                    default=plans.SWAP_DEFAULT,
                    min=0,
                    max=peaks.BLOCKS,
                    step=1,
                    display_name=plans.LABELS["swap_blocks"],
                    tooltip=f"From 0 (none: the whole model on the card) to {peaks.BLOCKS}.",
                )
            ],
            outputs=[
                SwapType.Output(
                    "blocks_swapped",
                    display_name=plans.LABELS["swap_blocks"],
                    tooltip="The blocks swapped, for an upscale node's blocks swapped input.",
                )
            ],
        )

    @classmethod
    def validate_inputs(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls, blocks: int | None = None
    ) -> bool | str:
        return _queued(None if blocks is None else plans.blocks_problem(blocks))

    @classmethod
    def execute(cls, blocks: int) -> io.NodeOutput:  # pyright: ignore[reportIncompatibleMethodOverride]
        _value(plans.blocks_problem(blocks))
        return io.NodeOutput(blocks)


class WindowLength(io.ComfyNode):
    """A clip's window length: the value of Upscale Video's window length, set in frames on the
    model's grid and given in the planner's latents."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="seedvr2xWindowLength",
            display_name="seedvr2x Window Length",
            category=SETTINGS_CATEGORY,
            description=(
                "The longest stretch of a clip the model works on at once, in frames, for"
                " seedvr2x Upscale Video: connect it to its window length input. The model takes"
                " frames 4 at a time after the first, so a window is"
                f" {plans.WINDOW_FRAMES_MIN}, {plans.WINDOW_FRAMES_MIN + 4},"
                f" {plans.WINDOW_FRAMES_MIN + 8}... frames, up to {plans.WINDOW_FRAMES_MAX}. A"
                " longer window keeps more frames consistent with each other and takes more of"
                " the graphics card; a shorter one is lighter, and the clip is blended across"
                " more windows. Leave the input unconnected and the plan chooses the longest that"
                " fits the card." + _FROM_INT
            ),
            inputs=[
                io.Int.Input(
                    "frames",
                    default=plans.WINDOW_FRAMES_DEFAULT,
                    min=plans.WINDOW_FRAMES_MIN,
                    max=plans.WINDOW_FRAMES_MAX,
                    step=plans.WINDOW_FRAMES_STEP,
                    display_name="window length (frames)",
                    tooltip=f"{plans.WINDOW_FRAMES_MIN} to {plans.WINDOW_FRAMES_MAX} frames, in"
                    f" steps of {plans.WINDOW_FRAMES_STEP}.",
                )
            ],
            outputs=[
                WindowType.Output(
                    "window_length",
                    display_name=plans.LABELS["window"],
                    tooltip="The window length, for Upscale Video's window length input.",
                )
            ],
        )

    @classmethod
    def validate_inputs(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls, frames: int | None = None
    ) -> bool | str:
        return _queued(None if frames is None else plans.window_frames_problem(frames))

    @classmethod
    def execute(cls, frames: int) -> io.NodeOutput:  # pyright: ignore[reportIncompatibleMethodOverride]
        _value(plans.window_frames_problem(frames))
        return io.NodeOutput(plans.window_latents(frames))


# ---- What ComfyUI loads --------------------------------------------------------------------------

NODES: tuple[type[io.ComfyNode], ...] = (
    Models,
    Cuts,
    UpscaleImage,
    UpscaleVideo,
    Tile,
    BlocksSwapped,
    WindowLength,
)


class Extension(ComfyExtension):
    """The nodes, as ComfyUI's loader asks for them (it awaits get_node_list and registers
    each class under its schema's node_id and display name)."""

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return list(NODES)


async def comfy_entrypoint() -> Extension:
    """What ComfyUI's loader looks for in a custom node's folder when it finds no
    NODE_CLASS_MAPPINGS (v0.44.0: V1 wins where both exist, so this module defines neither of
    V1's dictionaries)."""
    return Extension()

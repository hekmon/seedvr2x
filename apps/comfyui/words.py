"""The nodes' words (DESIGN.md, Repository and apps, the ComfyUI nodes, "Refusals by kind"): what
each refusal of the library, and each the nodes make themselves, says to a ComfyUI user, with its
way out in ComfyUI's terms ("choose the int8 file (8.3 GB) in seedvr2x Models", "decode tile 512
on this node"), never an option of the command line.

The library raises a plan's refusal as a class of its kind carrying its facts (errors.PlanRefused:
MemoryShort, HostShort, InputTooLarge, PinRefused, CompileRefused) and a message that is a
function of how an application words a setting (errors.Say, Refusal.words). `say` is the nodes'
words for a setting: the widget as the user sees it and the node it is in. A refusal may be
worded long after it was raised, by whoever holds it: nothing here reads a state of the run, only
the refusal and the facts handed in (the model file, whether a model folder was set, whether the
models are kept in RAM, the smaller files of the same 7B that would plan the run, the clip's size).

The four kinds of a plan that a user meets (memory, host, input, pin) and the loader's host
refusal are worded here from their facts, in a ComfyUI user's words (the card, the RAM, the
output, the tile), never by passing on the library's sentence: its terms (the DiT, VRAM, pinned
copies, latents, slots, phases) are the library's own and its figures are of its planner. The
way out names the nodes' inputs and value nodes ("connect a seedvr2x Tile of 1024 px to decode
tile"). Only the refusals the nodes can't reach (a compiled VAE, a recorded plan) keep the
library's words.

The node raises the result, `Refused`, an exception ComfyUI shows with its message."""

from collections.abc import Sequence
from dataclasses import dataclass

from seedvr2x import peaks, pull
from seedvr2x.errors import (
    CompileRefused,
    HostMemoryError,
    HostShort,
    InputTooLarge,
    MemoryShort,
    ModelChanged,
    ModelDamaged,
    ModelError,
    ModelMissing,
    ModelNotPinned,
    ModelsRefused,
    ModelUnreachable,
    PinRefused,
    PlanRefused,
    Refusal,
    Setting,
    TargetTooSmall,
)
from seedvr2x.memory import GIB
from seedvr2x.probe import ProbeRefusal
from seedvr2x.weights import AGAIN

from . import plans
from .progress import FAILED, REFUSED, STOPPED

MODELS = "seedvr2x Models"
CUTS = "seedvr2x Cuts"
KEEP = plans.LABELS["keep_in_ram_between_runs"]
FOLDER = plans.LABELS["model_dir"]


def folder_input(node: str = MODELS) -> str:
    """The model folder as the nodes name it in a sentence: an optional input, never a widget
    (the user, 2026-11-08), so the way out is to wire it or unwire it."""
    return f'the "{FOLDER}" input of {node}'


class Refused(Exception):
    """A run seedvr2x's nodes refuse, or one that can't go on: the message says what and the way
    out, in the nodes' terms, in ComfyUI's error dialog and, once, in its console at WARNING.
    The node's text box says `line`, one short line (the user, 2026-11-08) pointing at the
    error. `logged`: the console has the refusal already (models.session says a failed card at
    ERROR)."""

    line: str = REFUSED
    logged: bool = False

    def __init__(self, message: str = "", line: str | None = None) -> None:
        super().__init__(message)
        if line is not None:
            self.line = line


# ---- A setting, as the nodes call it -------------------------------------------------------------


def say(setting: Setting, node: str = MODELS) -> str:
    """The nodes' words for a setting a library refusal names (errors.Say): the input by its label,
    its value in the unit its value node sets it in, and that it is on this node. The model folder
    is the one of `node`, seedvr2x Models by default (seedvr2x Cuts has a folder of its own). A
    tile the library pins at 0 is the whole image at once (seedvr2x Tile's switch), a window in
    latents is said in frames (seedvr2x Window Length's unit)."""
    name, value = setting.name, setting.value
    if name in ("window", "swap_blocks", "encode_tile", "decode_tile"):
        label = plans.LABELS[name]
        if name.endswith("_tile") and value == "0":
            return f"{label} set to the whole image at once on this node"
        if name.endswith("_tile") and value:
            value = f"{value} px"
        elif name == "window" and value.isdigit():
            value = f"{plans.window_frames(int(value))} frames"
        return f"{label} {value} on this node" if value else f"{label} on this node"
    if name == "resolution":
        return f"resolution {value} on this node" if value else "resolution on this node"
    if name == "dit_model":
        label = plans.DIT_LABELS.get(value)
        return f"the model {label} in {MODELS}" if label else f"the model in {MODELS}"
    if name == "vae_model":
        return "the VAE file"
    if name == "model_dir":
        return folder_input(node)
    if name == "cuts":
        return "a cut list, which the nodes don't take"
    if name == "compile":
        return "compiling the VAE, which the nodes don't offer"
    raise ValueError(f"{name!r} is not one of the settings a refusal names")


def _gib(count: float) -> str:
    return f"{count / GIB:.2f} GiB"


# ---- A plan's refusals ---------------------------------------------------------------------------

_FILES = {"int8": "the int8 file", "the dynamic GGUF": "the dynamic GGUF file"}


def _smaller(smaller: Sequence[pull.Smaller], model_dir: bool) -> str:
    """The way out of the smaller files that plan the run, named as the Models node lists them:
    the file by its kind and size, the node it is chosen in; with a model folder wired, the file
    must be there under its name."""
    if not smaller:
        return ""
    named = [
        f"{_FILES.get(each.label, each.label)} ({pull.gigabytes(each.size)})" for each in smaller
    ]
    way = f"choose {' or '.join(named)} in {MODELS}, which plans this run"
    if model_dir:
        names = ", ".join(each.name for each in smaller)
        way += f" (with a folder wired to {folder_input()}, the file must be in it: {names})"
    return way


@dataclass(frozen=True)
class ClipSize:
    """What a refusal needs to know of the clip being upscaled, which only the node knows: its
    frames and the bytes of the output array it holds whole (clip.output_bytes). None for a
    picture's, whose arrays the plan counts itself."""

    frames: int
    output: int


# Where a refusal's memory is needed, in the nodes' words, by the phase of the run the library
# counts it in.
_WHERE = {
    "encode": "to turn the images into what the model works on",
    "dit": "while the model works",
    "decode": "to turn the model's result back into images",
}

# What the clip's whole in RAM and the command line's way, said in the clip's host refusal. The
# repository's folder, not a URL (the user, 2026-11-06).
_FILMS = (
    "These nodes keep a clip whole in RAM, as ComfyUI does any video. seedvr2x's command line, in"
    " the repository's apps/seedvr2x beside these nodes (see its README), is the tool for a long"
    " film: it reads it shot by shot and never keeps it whole, picks up again after a stop, and"
    " writes lossless files."
)


def _host_way(keep: bool) -> str:
    """The way out of a host short of memory: a smaller file takes less, and a model not kept
    between runs takes only what does not fit on the card, for the run."""
    return (
        f"set {KEEP} to off in {MODELS}, so that the model is read for each run and only the part"
        " that does not fit on the card is in RAM"
        if keep
        else "close programs that use the computer's RAM"
    )


def _shown(setting: Setting | None) -> str:
    """A setting's value in the unit its value node sets it in, after its label: "decode tile 512
    px", "blocks swapped 30", "window length 21 frames"; a tile of 0 is the whole image at once."""
    assert setting is not None
    name, value = setting.name, setting.value
    label = plans.LABELS[name]
    if name.endswith("_tile"):
        return f"{label} set to the whole image at once" if value == "0" else f"{label} {value} px"
    if name == "window":
        return f"{label} {plans.window_frames(int(value))} frames"
    return f"{label} {value}"


def _instead(setting: Setting | None) -> str:
    """The way out a refusal's `instead` names, in the nodes' terms: the value that fits, and the
    value node and socket that take it. "" where the refusal names none."""
    if setting is None:
        return ""
    name, value = setting.name, setting.value
    label = plans.LABELS.get(name, name)
    if name == "resolution":
        return f"set resolution to {value} or lower on this node" if value else ""
    if name.endswith("_tile"):
        if value == "0":
            return f"connect a seedvr2x Tile with whole image at once on to {label}"
        return f"connect a seedvr2x Tile of {value} px or less to {label}"
    if name == "swap_blocks":
        return f"connect a seedvr2x Blocks Swapped of {value} or more to {label}"
    if name == "window":
        frames = plans.window_frames(int(value))
        return f"connect a seedvr2x Window Length of {frames} frames or less to {label}"
    return ""


def _card(error: MemoryShort | PinRefused | InputTooLarge, what: str, phase: str = "") -> str:
    """What `what` needs of the card's free memory, and what the card has for it, from the
    refusal's facts (need and there, in bytes) and the phase it is needed in (the refusal's, else
    `phase`: a picture's tile pin has "pin" for its phase and is a pass's)."""
    where = _WHERE.get(error.phase) or _WHERE.get(phase, "")
    need, there = error.need, error.there
    if need is None or there is None:
        return f"{what} does not fit the free memory of the graphics card"
    return (
        f"{what} needs {_gib(need)} free on the graphics card{' ' + where if where else ''}, and"
        f" the card has {_gib(there)} free for it, counting ComfyUI's own models as unloaded"
        f" ({_gib(need - there)} short)"
    )


type Words = tuple[str, list[str]]


def _memory(error: MemoryShort, clip: ClipSize | None) -> Words:
    """A job that doesn't fit the card: the facts as a sentence, then the ways out (a resolution
    that fits, where one does; programs closed on the card, where that gives the memory back)."""
    reason = _card(error, "this clip" if clip is not None else "this picture")
    ways: list[str] = []
    fits = _instead(error.instead)
    if not fits and error.fits is not None:
        # A picture's refusal names the largest output that fits as (long, short) pixels: its
        # short side is the resolution.
        fits = f"set resolution to {error.fits[1]} or lower on this node"
    if fits:
        ways.append(fits)
    else:
        reason += ": no resolution fits"
    if error.freeable:
        ways.append("close programs that use the graphics card")
    return reason, ways


def _pin(error: PinRefused) -> Words:
    """A value connected that the plan can't take exactly: a number out of the model's range, a
    tile as large as the image, or a value that doesn't fit the card (with the one that does)."""
    pin = error.pin
    assert pin is not None
    label = plans.LABELS[pin.name]
    if error.why == "range":
        return (
            f"{_shown(pin)} is more than the model has: {peaks.BLOCKS} blocks",
            [f"connect a seedvr2x Blocks Swapped of 0 to {peaks.BLOCKS} to {label}"],
        )
    if error.why == "covers":
        return (
            f"{_shown(pin)} is as large as the whole image, so there would be no tiles",
            [
                f"connect a seedvr2x Tile with whole image at once on to {label}, for the whole"
                " image in one pass",
                f"connect a smaller seedvr2x Tile to {label}, for tiles",
            ],
        )
    ways = [_instead(error.instead)] if error.instead is not None else []
    if error.instead is not None and pin.name.endswith("_tile") and error.instead.value == "0":
        ways = [f"connect a seedvr2x Tile with whole image at once on to {label}, no tile fitting"]
    ways = [each for each in ways if each]
    ways.append(f"leave {label} unconnected, and the plan chooses one that fits")
    return _card(error, _shown(pin), pin.name.removesuffix("_tile")), ways


def _input(error: InputTooLarge) -> Words:
    """A picture too large to be copied to the card and resized there: its size, what its copies
    need, the largest picture that fits and the resolution that lowers the output's part."""
    pixels = error.pixels
    size = "this picture" if pixels is None else f"this picture, {_megapixels(pixels)},"
    what = f"{size} is copied to the graphics card at its own size and resized there, which"
    fits = error.fits_input
    ways = (
        [
            f"give the node a picture no larger than about {_megapixels(fits)}",
            "lower resolution on this node, which lowers the output's part of the memory",
        ]
        if fits is not None
        else ["lower resolution on this node: no picture this large fits at this resolution"]
    )
    return _card(error, what), ways


def _megapixels(pixels: int) -> str:
    return f"{pixels / 1_000_000:.1f} megapixels"


def _host(error: HostShort, keep: bool, clip: ClipSize | None) -> Words:
    """A plan that takes more RAM than the computer has available. For a clip, whose output the
    plan takes off the memory available first (clip.plan), the sum is said (the output, held
    whole, and the rest of the run's own) and what the command line does instead."""
    ways = [
        _host_way(keep),
        "lower resolution on this node, or give it fewer frames"
        if clip is not None
        else "lower resolution on this node, or give it fewer images",
    ]
    window = _instead(error.instead)
    if window:
        ways.append(window)
    need, available = error.need, error.available
    if need is None or available is None:
        return "this run needs more RAM than the computer has available", ways
    if clip is None:
        return (
            f"this run needs about {_gib(need)} of your computer's RAM, and {_gib(available)} are"
            f" available ({_gib(need - available)} short)",
            ways,
        )
    if need == clip.output:
        # The output alone is over what is available: the plan stopped there.
        said = (
            f"the output of this clip, {clip.frames} frames kept whole, takes {_gib(need)} of RAM,"
            f" and the computer has {_gib(available)} available"
        )
    else:
        total, there = need + clip.output, available + clip.output
        said = (
            f"this clip needs about {_gib(total)} of your computer's RAM: {_gib(clip.output)} for"
            f" the output, {clip.frames} frames kept whole, and {_gib(need)} for seedvr2x's own"
            " work (the parts of the model that wait in RAM, the longest shot's windows and the"
            f" tiles' working copies); {_gib(there)} are available ({_gib(total - there)} short)"
        )
    return f"{said}. {_FILMS[:-1]}", ways


def _plan(
    error: PlanRefused,
    smaller: Sequence[pull.Smaller],
    model_dir: bool,
    keep: bool,
    clip: ClipSize | None,
) -> str:
    """A plan's refusal in the nodes' words, from its kind's facts (see the module's docstring)."""
    if isinstance(error, CompileRefused) or (
        isinstance(error, PinRefused) and error.why in ("compile", "recorded")
    ):
        # The nodes never compile the VAE and never resume a recorded plan, so these refusals are
        # ones they can't reach; worded all the same, the library's "command" being a run here.
        return error.words(say).replace("The same command", "The same run")
    if isinstance(error, MemoryShort):
        reason, ways = _memory(error, clip)
    elif isinstance(error, PinRefused):
        reason, ways = _pin(error)
    elif isinstance(error, InputTooLarge):
        reason, ways = _input(error)
    elif isinstance(error, HostShort):
        reason, ways = _host(error, keep, clip)
    else:
        return error.words(say)
    if not isinstance(error, PinRefused):
        # A smaller file of the same 7B that plans the run is a way out apart from the settings.
        way = _smaller(smaller, model_dir)
        if way:
            ways.append(way)
    text = f"{reason[0].upper()}{reason[1:]}."
    return f"{text}\nYou can {'; or '.join(ways)}." if ways else text


# ---- The probe's ---------------------------------------------------------------------------------


def _probe(error: ProbeRefusal) -> str:
    """The environment probe's refusal: what is wrong with the kernel, in ComfyUI's environment;
    the file that keeps the measure, to delete after the environment is fixed."""
    return (
        f"{error.reason}. This is a problem of ComfyUI's environment (torch, its CUDA, the driver),"
        " not of the picture or the settings"
        + (f"; it was measured before, at {error.result.at} UTC" if error.cached else "")
        + f". {error.way[0].upper()}{error.way[1:]}."
    )


# ---- The model files' and the loader's refusals, by their kind and facts -------------------------

# The GB of the smaller files a host short of pinned memory is advised, the sharp 7B's (the
# Models node's default), as the Models node lists them.
_INT8, _GGUF = "seedvr2x_ema_7b_sharp_int8_convrot.safetensors", "seedvr2x_ema_7b_sharp_dyn.gguf"


def _host_memory(error: HostMemoryError, keep: bool) -> str:
    """The computer that can't hold the parts of the model that leave the card: what was asked,
    what it has, and what asks for less."""
    asked = "" if error.need is None else f" {_gib(error.need)} of"
    has = "" if error.available is None else f", and {_gib(error.available)} are available"
    ways = [f"set {KEEP} to off in {MODELS}"] if keep else []
    ways.append(
        f"choose the int8 file ({pull.gigabytes(pull.PINNED[_INT8].size)}) or the dynamic GGUF"
        f" file ({pull.gigabytes(pull.PINNED[_GGUF].size)}) in {MODELS}"
    )
    return (
        f"The parts of the model that wait in RAM while the graphics card works need{asked} your"
        f" computer's RAM{has}. You can {', or '.join(ways)}."
    )


def _not_in_folder(file: str, node: str) -> str:
    return (
        f"{file} is not in the folder wired to {folder_input(node)}: put the file there under"
        " that name, or unwire the input to use Hugging Face's cache, where seedvr2x fetches its"
        " files"
    )


def not_a_folder(value: str, node: str = MODELS) -> Refused:
    """A value of the model folder input that names no folder that exists."""
    return Refused(
        f'"{value}" is not a folder: wire a folder that holds seedvr2x\'s model files to'
        f" {folder_input(node)}, or unwire the input to use Hugging Face's cache"
    )


def _again(error: ModelError, model_dir: bool, node: str = MODELS) -> str:
    """The library's message of a damaged or changed file, which ends "fetch it again", with the
    way to do it in the nodes' terms where the file is one of seedvr2x's own (pinned, in its own
    role): the library's parts carry no advice of their own (weights.check_models adds the
    command line's to the whole). A file seedvr2x doesn't pin has no source to name: the
    library's words stand."""
    text = error.words(lambda setting: say(setting, node))
    ending = f"; {AGAIN}"
    pin = pull.PINNED.get(error.file)
    if not text.endswith(ending) or pin is None or (error.role and pin.role != error.role):
        return text
    if model_dir:
        way = (
            f"replace it with seedvr2x's own file ({pull.url(error.file)}), or unwire"
            f" {folder_input(node)} to use Hugging Face's cache"
        )
    else:
        way = (
            f"delete it from Hugging Face's cache ({pull.cache()}) and the next run fetches it"
            " again"
        )
    return f"{text.removesuffix(ending)}; {way}"


def _model_files(error: ModelError, model_dir: bool, node: str = MODELS) -> str:
    """A model file's refusal in the nodes' words: by its kind where the nodes have a way out of
    their own to say, else the library's message with the settings it names worded by `say`.
    `node` is the node whose model folder the files are looked for in."""
    if isinstance(error, ModelsRefused):
        return "\n".join(_model_files(part, model_dir, node) for part in error.parts)
    if isinstance(error, ModelMissing):
        return _not_in_folder(error.file or "a model file", node)
    if isinstance(error, ModelNotPinned):
        if error.figure == "size" and isinstance(error.found, int):
            found = f"size {error.found:,} bytes, where seedvr2x's has {error.expected:,} bytes"
        else:
            found = f"{error.figure} {error.found}, where seedvr2x's has {error.expected}"
        if error.command or not model_dir:
            # From Hugging Face's cache: seedvr2x fetches the file again once it is deleted.
            return (
                f"{error.file} is not seedvr2x's own file ({found}). Delete it from Hugging Face's"
                f" cache ({pull.cache()}) and the next run fetches it again."
            )
        source = f" ({pull.url(error.file)})" if error.file in pull.PINNED else ""
        return (
            f"{error.file} in the {FOLDER} is not seedvr2x's own file ({found}): replace it with"
            f" seedvr2x's file{source}, or unwire {folder_input(node)} to use Hugging Face's"
            " cache"
        )
    if isinstance(error, ModelUnreachable) and error.why == "offline":
        return (
            f"{error.file or 'a model file'} is not in Hugging Face's cache ({pull.cache()}) and"
            " Hugging Face's offline mode is on (HF_HUB_OFFLINE is set). seedvr2x fetches its"
            " files once, at the first run: unset HF_HUB_OFFLINE and restart ComfyUI, or"
            f" wire a folder that has them to {folder_input(node)}"
        )
    if isinstance(error, ModelDamaged | ModelChanged):
        return _again(error, model_dir, node)
    return error.words(lambda setting: say(setting, node))


# ---- The whole -----------------------------------------------------------------------------------


def refusal(
    error: Refusal,
    *,
    smaller: Sequence[pull.Smaller] = (),
    model_dir: bool = False,
    keep: bool = True,
    clip: ClipSize | None = None,
    node: str = MODELS,
) -> Refused:
    """A library refusal in the nodes' words (Refused): a plan's by its kind and facts with the
    way out in the nodes' terms; the probe's; a target too small for the padding; the model
    files' and the host's by their kind and facts; any other by its words. `smaller` are the
    pinned files smaller than the DiT file that plan the run (pull.smaller_planning), `model_dir`
    whether a model folder was set, `keep` whether the models are kept in RAM between runs, `clip`
    the clip's size where the run is a clip's, `node` the node whose model folder holds the files
    a model file's refusal names (seedvr2x Cuts has its own)."""
    if isinstance(error, PlanRefused):
        return Refused(_plan(error, smaller, model_dir, keep, clip))
    if isinstance(error, ProbeRefusal):
        return Refused(_probe(error))
    if isinstance(error, TargetTooSmall):
        return Refused(_too_small(error))
    if isinstance(error, ModelError):
        return Refused(_model_files(error, model_dir, node))
    if isinstance(error, HostMemoryError):
        return Refused(_host_memory(error, keep))
    return Refused(error.words(lambda setting: say(setting, node)))


def _too_small(error: TargetTooSmall) -> str:
    """A resolution that makes an image too small to work on, from the refusal's facts: the size
    it makes and the least the short side takes (the rows when the image is wider than tall, the
    columns otherwise), with the setting named as the node has it, whatever linked it."""
    height, width = error.size
    least = error.least[0] if height <= width else error.least[1]
    asked = say(Setting("resolution", str(error.resolution)))
    return (
        f"{asked} makes a {width}x{height} image, which is too small: seedvr2x needs {least} px on"
        f" the short side at least; raise resolution on this node"
    )


# ---- What the nodes refuse themselves ------------------------------------------------------------


def no_cuda(reason: str) -> Refused:
    return Refused(
        f"seedvr2x's nodes run on an NVIDIA GPU with CUDA, and {reason}: start ComfyUI on the GPU,"
        " without its CPU option"
    )


def not_freed(*, need: int, there: int, comfy: int) -> Refused:
    """The card still short after ComfyUI was asked to free its models: what the plan counts on,
    what is there, and what ComfyUI's models that stay still hold."""
    left, unload = (
        ("other programs' memory is what is left", "")
        if comfy == 0
        else (
            f"the {_gib(comfy)} that ComfyUI's models still use, and other programs' memory,"
            " are what is left",
            " use Unload models in ComfyUI's menu, or",
        )
    )
    return Refused(
        f"the card has {_gib(there)} free for seedvr2x's plan, which counts on {_gib(need)}, after"
        f" ComfyUI was asked to free its models; {left}. Close the programs that use the card,"
        f" or{unload} choose a smaller model file in {MODELS} or a lower resolution on this node"
    )


def card_busy() -> Refused:
    """A run refused at its start: the work an earlier run's release left (a second Cancel) is
    still pending on the card, asked without waiting (waiting.settle)."""
    return Refused(
        "the graphics card is still busy with the last work of an earlier seedvr2x run: wait for it"
        " to finish and run again, or restart ComfyUI if it does not. Nothing was read from the"
        " card and nothing was run",
        line=STOPPED,
    )


def card_failed(error: str, earlier: bool = False) -> Refused:
    """The card failed with a CUDA error other than out of memory (`error`, its first line;
    waiting.fail): nothing on it is freed or run again in ComfyUI's process, since under ComfyUI's
    cudaMallocAsync a free there reports the error again and ends the process at once. Said by the
    run the error stopped, and by every later run at its start (`earlier`)."""
    if earlier:
        return Refused(
            f"the graphics card failed during an earlier seedvr2x run ({error}), and nothing on it"
            " can be freed or run again inside this ComfyUI: restart ComfyUI, then run again."
            " Nothing was read from the card and nothing was run",
            line=FAILED,
        )
    return Refused(
        f"the graphics card failed: {error}. seedvr2x leaves everything on the card as it is,"
        " since freeing memory there could end ComfyUI at once: restart ComfyUI, then run again",
        line=FAILED,
    )


def transparent() -> Refused:
    return Refused(
        "the image has transparent pixels, which seedvr2x refuses: the model would redraw the"
        " colours hidden under them and leave coloured edges. Put the image on a background"
        " first (with an image composite node, for one) and connect the result here"
    )


def channels(count: int) -> Refused:
    return Refused(
        f"the image has {count} channels, and seedvr2x takes a colour image of 3, or 4 with an"
        " opaque alpha channel"
    )


def frames_missing() -> Refused:
    return Refused("the images input is empty: the node got no image")


def non_finite(message: str) -> Refused:
    return Refused(f"{message}, so nothing is output; run it again, or with a different seed")


def cuts_wrong(covered: int, frames: int) -> Refused:
    """A cut list that does not cover the frames it is given: it was made for another clip."""
    return Refused(
        f"the cut list covers {covered} frames and the clip has {frames}: connect the images that"
        " seedvr2x Cuts was given to this node too, or its output is for another clip"
    )


def seed_past_the_cuts(seed: int, start: int, highest: int) -> Refused:
    """A seed that some shot of the cut list seeds past numz's range: each shot is seeded with the
    seed plus its first frame, its encode with passes.ENCODE_SEED_OFFSET more, and NumPy's
    generator takes seeds under 2**32 only. `start` is the last shot's first frame, `highest` the
    largest seed that works with this cut list."""
    return Refused(
        f"seed {seed:,} is too high for this cut list: each shot is seeded with the seed plus its"
        f" first frame, the last starts at frame {start:,}, and the seed can be {highest:,} at"
        " most: lower the seed"
    )


def cuts_not_a_list() -> Refused:
    return Refused(f"seedvr2x Upscale Video takes the cuts output of {CUTS}")

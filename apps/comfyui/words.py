"""The nodes' words (DESIGN.md, Repository and apps, the ComfyUI nodes, "Refusals by kind"): what
each refusal of the library, and each the nodes make themselves, says to a ComfyUI user, with its
way out in ComfyUI's terms ("choose the int8 file (8.3 GB) in seedvr2x Models", "decode tile 512
on this node"), never an option of the command line.

The library raises a plan's refusal as a class of its kind carrying its facts (errors.PlanRefused:
MemoryShort, HostShort, InputTooLarge, PinRefused, CompileRefused) and a message that is a
function of how an application words a setting (errors.Say, Refusal.words). `say` is the nodes'
words for a setting: the widget as the user sees it and the node it is in. A refusal may be
worded long after it was raised, by whoever holds it: nothing here reads a state of the run, only
the refusal and the facts handed in (the DiT file, whether a model directory was set, whether the
models are kept in RAM, the smaller files of the same 7B that would plan the run).

The node raises the result, `Refused`, an exception ComfyUI shows with its message."""

from collections.abc import Sequence

from seedvr2x import pull
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
    PlanRefused,
    Refusal,
    Setting,
    TargetTooSmall,
)
from seedvr2x.memory import GIB
from seedvr2x.probe import ProbeRefusal
from seedvr2x.weights import AGAIN

from . import plans

MODELS = "seedvr2x Models"
KEEP = "keep in RAM between runs"


class Refused(Exception):
    """A run seedvr2x's nodes refuse, or one that can't go on: the message says what and the way
    out, in the nodes' terms. ComfyUI shows it on the node."""


# ---- A setting, as the nodes call it -------------------------------------------------------------


def say(setting: Setting) -> str:
    """The nodes' words for a setting a library refusal names (errors.Say): the widget by the name
    it has (its label), its value, and that it is on this node. A tile the library pins at 0 is
    untiled, which the widget says with plans.UNTILED_TILE (0 being its auto)."""
    name, value = setting.name, setting.value
    if name in ("window", "swap_blocks", "encode_tile", "decode_tile"):
        label = name.replace("_", " ")
        if name.endswith("_tile") and value == "0":
            value = f"untiled ({plans.UNTILED_TILE})"
        return f"{label} {value} on this node" if value else f"{label} on this node"
    if name == "resolution":
        return f"resolution {value} on this node" if value else "resolution on this node"
    if name == "dit_model":
        label = plans.DIT_LABELS.get(value)
        return f"the model {label} in {MODELS}" if label else f"the model in {MODELS}"
    if name == "vae_model":
        return "the VAE file"
    if name == "model_dir":
        return f"model_dir in {MODELS}"
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
    the file by its kind and size, the node it is chosen in; with a model directory set, the file
    must be there under its name."""
    if not smaller:
        return ""
    named = [
        f"{_FILES.get(each.label, each.label)} ({pull.gigabytes(each.size)})" for each in smaller
    ]
    way = f"choose {' or '.join(named)} in {MODELS}, which plans this run"
    if model_dir:
        names = ", ".join(each.name for each in smaller)
        way += f" (with a model directory set there, the file must be in it: {names})"
    return way


def _host_way(keep: bool) -> str:
    """The way out of a host short of memory: a smaller file pins less, and a model not kept
    between runs pins only what leaves the card for the run."""
    return (
        f"set {KEEP} to off in {MODELS}, so that the weights are read for the run and only what"
        " leaves the card is pinned in system memory"
        if keep
        else "close programs that hold system memory"
    )


def _plan(error: PlanRefused, smaller: Sequence[pull.Smaller], model_dir: bool, keep: bool) -> str:
    text = error.words(say)
    if isinstance(error, CompileRefused):
        # The nodes never compile the VAE (a plan's compile flag is False), so this refusal is
        # one they can't reach; worded all the same, the library's "command" being a run here.
        text = text.replace("The same command", "The same run")
    ways: list[str] = []
    if isinstance(error, MemoryShort | InputTooLarge):
        # The message names the settings of the run that fit (a smaller resolution, a tile, a
        # window); a smaller file of the same 7B that plans it is the way out apart from them.
        way = _smaller(smaller, model_dir)
        if way:
            ways.append(way)
    elif isinstance(error, HostShort):
        ways.append(_host_way(keep))
        way = _smaller(smaller, model_dir)
        if way:
            ways.append(way)
    if not ways:
        return text
    return f"{text}\nOr {'; or '.join(ways)}."


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
    """The host that can't pin the weights: what was asked, what it has, and what pins less."""
    asked = "" if error.need is None else f" (asking {_gib(error.need)})"
    ways = [f"set {KEEP} to off in {MODELS}"] if keep else []
    ways.append(
        f"choose the int8 file ({pull.gigabytes(pull.PINNED[_INT8].size)}) or the dynamic GGUF"
        f" file ({pull.gigabytes(pull.PINNED[_GGUF].size)}) in {MODELS}"
    )
    return f"{error.words(say)}{asked}. You can {', or '.join(ways)}."


def _again(error: ModelError, model_dir: bool) -> str:
    """The library's message of a damaged or changed file, which ends "fetch it again", with the
    way to do it in the nodes' terms where the file is one of seedvr2x's own (pinned, in its own
    role): the library's parts carry no advice of their own (weights.check_models adds the
    command line's to the whole). A file seedvr2x doesn't pin has no source to name: the
    library's words stand."""
    text = error.words(say)
    ending = f"; {AGAIN}"
    pin = pull.PINNED.get(error.file)
    if not text.endswith(ending) or pin is None or (error.role and pin.role != error.role):
        return text
    if model_dir:
        way = (
            f"replace it with the pinned file ({pull.url(error.file)}), or leave model_dir empty"
            f" in {MODELS} to use Hugging Face's cache"
        )
    else:
        way = (
            f"delete it from Hugging Face's cache ({pull.cache()}) and the next run fetches it"
            " again"
        )
    return f"{text.removesuffix(ending)}; {way}"


def _model_files(error: ModelError, model_dir: bool) -> str:
    """A model file's refusal in the nodes' words: by its kind where the nodes have a way out of
    their own to say, else the library's message with the settings it names worded by `say`."""
    if isinstance(error, ModelsRefused):
        return "\n".join(_model_files(part, model_dir) for part in error.parts)
    if isinstance(error, ModelMissing):
        return (
            f"{error.file or 'a model file'} is not in the directory set as model_dir in"
            f" {MODELS}: put the file there under that name, or leave model_dir empty to use"
            " Hugging Face's cache, where seedvr2x fetches its files"
        )
    if isinstance(error, ModelNotPinned):
        found = f"{error.figure} {error.found}, where the pin has {error.expected}"
        if error.command or not model_dir:
            # From Hugging Face's cache: seedvr2x fetches the file again once it is deleted.
            return (
                f"{error.file} is not the file seedvr2x pins ({found}). Delete it from Hugging"
                f" Face's cache ({pull.cache()}) and the next run fetches it again."
            )
        return (
            f"{error.file} in the model directory is not the file seedvr2x pins ({found}): replace"
            f" it with the pinned file, or leave model_dir empty in {MODELS} to use Hugging Face's"
            " cache"
        )
    if isinstance(error, ModelUnreachable) and error.why == "offline":
        return (
            f"{error.file or 'a model file'} is not in Hugging Face's cache ({pull.cache()}) and"
            " the environment keeps the network out (HF_HUB_OFFLINE is set). seedvr2x fetches its"
            " files once, at the first run: unset HF_HUB_OFFLINE for it, or set model_dir in"
            f" {MODELS} to a directory that holds them"
        )
    if isinstance(error, ModelDamaged | ModelChanged):
        return _again(error, model_dir)
    return error.words(say)


# ---- The whole -----------------------------------------------------------------------------------


def refusal(
    error: Refusal,
    *,
    smaller: Sequence[pull.Smaller] = (),
    model_dir: bool = False,
    keep: bool = True,
) -> Refused:
    """A library refusal in the nodes' words (Refused): a plan's by its kind and facts with the
    way out in the nodes' terms; the probe's; a target too small for the padding; the model
    files' and the host's by their kind and facts; any other by its words. `smaller` are the
    pinned files smaller than the DiT file that plan the run (pull.smaller_planning), `model_dir`
    whether a model directory was set, `keep` whether the models are kept in RAM between runs."""
    if isinstance(error, PlanRefused):
        return Refused(_plan(error, smaller, model_dir, keep))
    if isinstance(error, ProbeRefusal):
        return Refused(_probe(error))
    if isinstance(error, TargetTooSmall):
        return Refused(f"{error.words(say)}; raise resolution on this node")
    if isinstance(error, ModelError):
        return Refused(_model_files(error, model_dir))
    if isinstance(error, HostMemoryError):
        return Refused(_host_memory(error, keep))
    return Refused(error.words(say))


# ---- What the nodes refuse themselves ------------------------------------------------------------


def no_cuda(reason: str) -> Refused:
    return Refused(
        f"seedvr2x's nodes run on an NVIDIA GPU with CUDA, and {reason}: start ComfyUI on the GPU,"
        " without its CPU option"
    )


def attention_missing() -> Refused:
    return Refused(
        "attention is set to flash_attn_2 on this node, and FlashAttention 2 isn't installed in"
        " ComfyUI's environment: choose auto, which runs it where it is installed and PyTorch's"
        " SDPA where it isn't, or sdpa"
    )


def not_freed(*, need: int, there: int, comfy: int) -> Refused:
    """The card still short after ComfyUI was asked to free its models: what the plan counts on,
    what is there, and what ComfyUI's models that stay still hold."""
    left = (
        "other programs' memory is what is left"
        if comfy == 0
        else f"the {_gib(comfy)} that ComfyUI's models still hold, and other programs' memory,"
        " are what is left"
    )
    return Refused(
        f"the card has {_gib(there)} free for seedvr2x's plan, which counts on {_gib(need)}, after"
        f" ComfyUI was asked to free its models; {left}. Close the programs that use the card, or"
        f" use Unload models in ComfyUI's menu, or choose a smaller DiT file in {MODELS} or a"
        " lower resolution on this node"
    )


def transparent() -> Refused:
    return Refused(
        "the image has transparent pixels, and seedvr2x refuses it: the model would redraw the"
        " colours hidden under them and fringe the edges. Flatten it onto a background before this"
        " node (a node that composites it onto a colour or another image)"
    )


def channels(count: int) -> Refused:
    return Refused(
        f"the image has {count} channels, and seedvr2x takes a colour image of 3, or 4 with an"
        " opaque alpha channel"
    )


def frames_missing() -> Refused:
    return Refused("the node got no image: its batch is empty")


def non_finite(message: str) -> Refused:
    return Refused(f"{message}, so nothing is output; run it again, or with a different seed")

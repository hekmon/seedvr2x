"""The nodes' words for every refusal of the library and every refusal of their own
(words.refusal and the rest): the refusal and its way out in ComfyUI's terms, from the class's
facts and the refusal's own words with the nodes' `say` for a setting, never an option of the
command line."""

import ast
import importlib
import re
from pathlib import Path

import comfyui
import pytest
import torch
from comfyui import models, nodes, plans, words
from standins import GIB, execute, frames_video

from seedvr2x import clip, frames, peaks, pull, still_plan, weights, yuv
from seedvr2x.errors import (
    CompileRefused,
    HostMemoryError,
    HostShort,
    InputTooLarge,
    MemoryShort,
    ModelChanged,
    ModelDamaged,
    ModelMissing,
    ModelNoRoom,
    ModelNotPinned,
    ModelsRefused,
    ModelUnknown,
    ModelUnreachable,
    ModelUnreadable,
    PictureRefused,
    PinRefused,
    PlanRefused,
    Refusal,
    Setting,
    TargetTooSmall,
)
from seedvr2x.memory import Reading
from seedvr2x.probe import ProbeRefusal, Result
from seedvr2x.shot_plan import Pins as ShotPins

TARGET = (1080, 1920)
BOUND = (1920, 1080)


def picture(
    free: int, *, host: int | None = 64 * GIB, input_pixels: int = 1920 * 1080, **pins: int
) -> PlanRefused:
    """The refusal still_plan.plan gives a picture of 1080p in `free` bytes."""
    try:
        still_plan.plan(
            BOUND,
            Reading(free, None, free, False),
            host,
            process=2 * GIB,
            input_pixels=input_pixels,
            picture=TARGET,
            pins=still_plan.Pins(**pins),
        )
    except PlanRefused as refused:
        return refused
    raise AssertionError("the plan was not refused")


def clip_refusal(free: int, frames_: int, host: int | None, **pins: int) -> PlanRefused:
    try:
        clip.plan(
            Reading(free, None, free, False),
            TARGET,
            frames_,
            ShotPins(**pins),
            host,
            correct=True,
        )
    except PlanRefused as refused:
        return refused
    raise AssertionError("the plan was not refused")


def models_refused() -> dict[str, Refusal]:
    """One refusal of each kind of errors.py's model files, as the library's own calls word them
    (settings named by `Setting`, so that an application's `say` words them)."""
    dir_ = Setting("model_dir")
    dit = Setting("dit_model", "x.safetensors")
    pinned = ModelNotPinned(
        lambda say: (
            f"/c/a.safetensors: 5 bytes, where the pin has 9: not the file seedvr2x pins;"
            f" remove it, or leave out {say(dir_)}"
        ),
        role="dit",
        file="a.safetensors",
        figure="size",
        expected=9,
        found=5,
        settings=(dir_,),
        command="hf download hekmon/seedvr2x a.safetensors --revision c14a2bc4 --force-download",
    )
    unknown = ModelUnknown(
        lambda say: (
            f"{say(dit)}: not one of seedvr2x's own model files; give {say(dir_)} with a directory"
        ),
        role="dit",
        file="x.safetensors",
        settings=(dit, dir_),
    )
    unreachable = lambda why: ModelUnreachable(  # noqa: E731
        lambda say: f"a.safetensors: Hugging Face is out of reach ({why}); give {say(dir_)}",
        why=why,
        file="a.safetensors",
        settings=(dir_,),
    )
    return {
        "model: missing": ModelMissing("/m/a.safetensors: no such file", file="a.safetensors"),
        "model: unknown": unknown,
        "model: damaged": ModelDamaged("/m/a: cut short, 5 of 9 bytes; fetch it again", file="a"),
        "model: changed": ModelChanged("/m/a: changed while it loaded", file="a", figure="sha256"),
        "model: not pinned": pinned,
        "model: not pinned, directory": ModelNotPinned(
            "/m/a: 5 bytes, where the pin has 9",
            file="a.safetensors",
            figure="sha256",
            expected="b",
            found="a",
        ),
        "model: offline": unreachable("offline"),
        "model: unreachable": unreachable("unreachable"),
        "model: failed": unreachable("failed"),
        "model: no room": ModelNoRoom("no room for 17 GB", need=17 * GIB, free=GIB),
        "model: unreadable": ModelUnreadable(
            lambda say: f"/c: permission denied; give {say(dir_)}", settings=(dir_,)
        ),
        "model: several": ModelsRefused([ModelMissing("a: no such file", file="a"), unknown]),
        "picture": PictureRefused(
            "a picture has transparent pixels", kind="transparent", reason="r", way="w"
        ),
    }


def refusals() -> dict[str, Refusal]:
    probe = ProbeRefusal(
        Result("int8 linear", 0.5, 0.01, "bfloat16 spacings", False, "2026-11-05T10:00:00"),
        {"gpu": "a GPU", "torch": "2.9.1", "cuda": "12.8"},
        Path("probe.json"),
        cached=True,
    )
    too_small = Refusal("")
    try:
        frames.check_target((10, 16), 10)
    except TargetTooSmall as small:
        too_small = small
    source = Refusal("")
    try:
        yuv.reading("yuva420p", 64, 16)
    except Refusal as refused:
        source = refused
    compile_ = CompileRefused(
        lambda say: (
            f"{say(Setting('compile'))} compiles only the shapes measured\n"
            f"The same command with {say(Setting('decode_tile', '512'))} fits"
        ),
        phase="decode",
        within=(Setting("decode_tile", "512"),),
    )
    return {
        "pin: range": picture(24 * GIB, swap_blocks=99),
        "pin: covers": picture(24 * GIB, decode_tile=4096),
        "memory": picture(1 * GIB),
        "memory, video": clip_refusal(2 * GIB, 33, 64 * GIB),
        "host": picture(24 * GIB, host=1 * GIB),
        "host, video": clip_refusal(24 * GIB, 33, 1 * GIB),
        "host, the output array": clip_refusal(24 * GIB, 100_000, 8 * GIB),
        "input": picture(8 * GIB, input_pixels=10**9),
        # A window of 9 latents takes 20.56 GiB since the lean attention (2026-11-06), over 24
        # before.
        "pin: window": clip_refusal(
            19 * GIB, 33, 64 * GIB, window=9, swap_blocks=0, encode_tile=512
        ),
        "compile": compile_,
        "probe": probe,
        "too small": too_small,
        "source": source,
        **models_refused(),
        "host copies": HostMemoryError(
            "the weights that leave the GPU need 15.81 GiB of host memory, pinned, and the host has"
            " 4.00 GiB available",
            need=15 * GIB,
            available=4 * GIB,
        ),
    }


@pytest.mark.parametrize("name", list(refusals()))
def test_every_refusal_is_worded_without_an_option_of_the_command_line(name: str) -> None:
    error = refusals()[name]
    # Worded long after it was raised and its handler left: nothing is read but the refusal.
    said = words.refusal(error, model_dir=True)
    assert isinstance(said, words.Refused) and isinstance(said, Exception)
    text = str(said)
    assert text and "--" not in text and "command" not in text.lower().replace("command-line", "")
    assert "hf download" not in text


def test_a_setting_is_named_as_the_widget_the_user_sees() -> None:
    say = words.say
    # Each in the unit its value node sets it in: a window in frames (6 latents, 21 frames).
    assert say(Setting("window", "6")) == "window length 21 frames on this node"
    assert say(Setting("swap_blocks", "30")) == "blocks swapped 30 on this node"
    assert say(Setting("decode_tile", "512")) == "decode tile 512 px on this node"
    # The library's pin of the untiled pass, 0, is seedvr2x Tile's whole image at once.
    assert say(Setting("encode_tile", "0")) == (
        "encode tile set to the whole image at once on this node"
    )
    assert say(Setting("resolution", "720")) == "resolution 720 on this node"
    assert say(Setting("resolution")) == "resolution on this node"
    assert say(Setting("dit_model", "seedvr2x_ema_7b_sharp_int8_convrot.safetensors")) == (
        "the model Sharp 7B, int8 (8.3 GB, 0.86%) in seedvr2x Models"
    )
    assert "seedvr2x Models" in say(Setting("dit_model"))


def test_a_pin_that_is_no_pin_names_the_widget_and_the_value_node() -> None:
    text = str(words.refusal(picture(24 * GIB, swap_blocks=99)))
    assert text.startswith("Blocks swapped 99 is more than the model has: 36 blocks.")
    assert "connect a seedvr2x Blocks Swapped of 0 to 36 to blocks swapped" in text
    covers = str(words.refusal(picture(24 * GIB, decode_tile=4096)))
    assert covers.startswith("Decode tile 4096 px is as large as the whole image")
    assert "seedvr2x Tile with whole image at once on to decode tile" in covers
    assert "a smaller seedvr2x Tile to decode tile" in covers


def test_a_pin_that_does_not_fit_the_card_is_worded_with_the_value_that_does() -> None:
    """The value nodes' units: a tile in px, blocks as a count, a window in frames; the figures
    are the refusal's facts (need, there), the way out the node and the socket that take it."""
    tile = picture(6 * GIB, decode_tile=1024)
    assert isinstance(tile, PinRefused) and tile.why == "memory"
    said = str(words.refusal(tile))
    assert said.startswith("Decode tile 1024 px needs ")
    assert f"{tile.need / GIB:.2f} GiB free on the graphics card" in said  # type: ignore[operator]
    assert f"the card has {tile.there / GIB:.2f} GiB free" in said  # type: ignore[operator]
    assert "to turn the model's result back into images" in said
    assert "connect a seedvr2x Tile of " in said and " px or less to decode tile" in said
    assert "leave decode tile unconnected, and the plan chooses one that fits" in said
    swap = picture(6 * GIB, swap_blocks=0)
    assert isinstance(swap, PinRefused) and swap.why == "memory"
    said = str(words.refusal(swap))
    assert said.startswith("Blocks swapped 0 needs ") and "while the model works" in said
    assert "connect a seedvr2x Blocks Swapped of " in said and " or more to blocks swapped" in said
    window = clip_refusal(20 * GIB, 33, 64 * GIB, window=9, swap_blocks=0, encode_tile=512)
    said = str(words.refusal(window))
    assert said.startswith("Window length 33 frames needs ")
    assert "connect a seedvr2x Window Length of 21 frames or less to window length" in said


def test_a_memory_refusal_names_the_smaller_files_that_plan_it_and_the_resolution() -> None:
    error = picture(1 * GIB)
    smaller = pull.smaller_files("seedvr2x_ema_7b_sharp_fp16.safetensors", weights.DIT_7B)
    assert [each.label for each in smaller] == ["int8", "the dynamic GGUF"]
    text = str(words.refusal(error, smaller=smaller))
    assert (
        "choose the int8 file (8.3 GB) or the dynamic GGUF file (4.8 GB) in seedvr2x Models" in text
    )
    # The facts, worded: what the picture needs of the card, what the card has, and the phase.
    assert isinstance(error, MemoryShort) and error.need is not None and error.there is not None
    assert text.startswith(
        f"This picture needs {error.need / GIB:.2f} GiB free on the graphics card to turn the"
        f" model's result back into images, and the card has {error.there / GIB:.2f} GiB free for"
        " it"
    )
    assert "no resolution fits" in text and "close programs that use the graphics card" in text
    with_dir = str(words.refusal(error, smaller=smaller[1:], model_dir=True))
    assert "must be in it: seedvr2x_ema_7b_sharp_dyn.gguf" in with_dir
    assert 'with a folder wired to the "model folder" input of seedvr2x Models' in with_dir
    assert "choose" not in str(words.refusal(error))


def test_a_host_short_of_memory_names_what_pins_less() -> None:
    error = picture(24 * GIB, host=1 * GIB)
    assert isinstance(error, HostShort)
    text = str(words.refusal(error, keep=True))
    assert "set keep in RAM between runs to off in seedvr2x Models" in text
    assert "keep in RAM" not in str(words.refusal(error, keep=False))
    small = pull.smaller_files("seedvr2x_ema_7b_sharp_fp16.safetensors", weights.DIT_7B)
    assert "choose the int8 file" in str(words.refusal(error, smaller=small))


def test_a_videos_frames_that_the_host_cant_hold_are_worded_with_the_hosts_figures() -> None:
    # The frames on their way through the run alone are over what the computer has: said as such.
    error = clip_refusal(24 * GIB, 100_000, 8 * GIB)
    assert isinstance(error, HostShort) and error.need is not None
    size = words.ClipSize(100_000, error.need)
    text = str(words.refusal(error, clip=size))
    assert text.startswith(
        f"The frames on their way through this run take {error.need / GIB:.2f} GiB of RAM, and"
        " the computer has 8.00 GiB available"
    )
    assert "--" not in text


def test_a_video_the_host_cant_hold_is_told_the_ways_out_of_a_streamed_video_or_of_frames() -> None:
    """A video's plan the host can't hold: the run's own work and the frames on their way summed.
    A video read from its file (streamed) is never held whole, so no way out is the command
    line's and none promises a RAM figure; frames ComfyUI holds whole have the way of loading the
    video from its file first."""
    error = clip_refusal(24 * GIB, 33, 1 * GIB)
    assert isinstance(error, HostShort) and error.need is not None and error.available is not None
    on_the_way = clip.output_bytes(33, TARGET)
    assert on_the_way != error.need
    for streamed in (True, False):
        size = words.ClipSize(33, on_the_way, streamed)
        text = str(words.refusal(error, clip=size))
        total = error.need + on_the_way
        assert text.startswith(
            f"This video needs about {total / GIB:.2f} GiB of your computer's RAM"
        )
        assert f"the frames on their way, {on_the_way / GIB:.2f} GiB)" in text
        assert f"{(error.available + on_the_way) / GIB:.2f} GiB are available" in text
        assert f"{(total - error.available - on_the_way) / GIB:.2f} GiB short" in text
        _, ways = text.split("\nYou can ", 1)
        first, *others = ways.split("; or ")
        # No way out is the command line's stream, whichever the video is.
        assert "command line" not in text and "apps/seedvr2x" not in text
        assert "any length fits" not in text and "kept whole" not in text
        assert "lower resolution on this node" in ways and "film" not in text.lower()
        assert "http" not in text and "--" not in text
        if streamed:
            assert "Load Video" not in text
            assert first.startswith("close programs") or first.startswith("set keep in RAM")
        else:
            assert first == (
                "load the video from its file with Load Video: this node then reads it shot by"
                " shot, never all in memory at once"
            )
            assert others[0].startswith("close programs") or others[0].startswith("set keep in")
    # A picture's host refusal says none of it.
    picture_text = str(words.refusal(picture(24 * GIB, host=1 * GIB)))
    assert "command line" not in picture_text and "give it fewer images" in picture_text


# The library's own words (the DiT's name is the one case-sensitive: "edit" holds "dit").
LIBRARY_TERMS = ("vram", "pinned", "latent", "slot", "phase", "gpu", "--")


def test_no_library_term_reaches_the_user_in_the_four_kinds_a_plan_refuses() -> None:
    """The library's own words (item 19): memory, host, input and pin refusals, each from the
    real planner, a picture's and a clip's, worded from their facts."""
    size = words.ClipSize(33, clip.output_bytes(33, TARGET))
    made = {
        "pin range": (picture(24 * GIB, swap_blocks=99), None),
        "pin covers": (picture(24 * GIB, decode_tile=4096), None),
        "pin tile": (picture(6 * GIB, decode_tile=1024), None),
        "pin swap": (picture(6 * GIB, swap_blocks=0), None),
        "pin window": (clip_refusal(20 * GIB, 33, 64 * GIB, window=9, swap_blocks=0), size),
        "memory": (picture(1 * GIB), None),
        "memory clip": (clip_refusal(2 * GIB, 33, 64 * GIB), size),
        "host": (picture(24 * GIB, host=1 * GIB), None),
        "host clip": (clip_refusal(24 * GIB, 33, 1 * GIB), size),
        "host output": (clip_refusal(24 * GIB, 100_000, 8 * GIB), size),
        "input": (picture(8 * GIB, input_pixels=10**9), None),
    }
    kinds = set()
    for name, (error, clip_size) in made.items():
        kinds.add(error.kind)
        for keep in (True, False):
            text = str(words.refusal(error, keep=keep, clip=clip_size))
            assert "DiT" not in text, (name, text)
            for term in LIBRARY_TERMS:
                assert term not in text.lower(), (name, term, text)
    assert kinds == {"memory", "host", "input", "pin"}


def test_an_input_too_large_says_the_resolution_that_lowers_the_outputs_part() -> None:
    error = picture(8 * GIB, input_pixels=10**9)
    assert isinstance(error, InputTooLarge)
    text = str(words.refusal(error))
    assert error.fits_input is not None
    assert f"no larger than about {error.fits_input / 1_000_000:.1f} megapixels" in text
    assert "lower resolution on this node" in text
    assert "megapixels" in text and "1000.0 megapixels" in text


def test_a_memory_refusal_without_a_smaller_file_names_none() -> None:
    error = picture(1 * GIB)
    assert isinstance(error, MemoryShort)
    assert "choose" not in str(words.refusal(error)) and "You can close programs" in str(
        words.refusal(error)
    )


def test_a_memory_refusal_names_the_resolution_that_fits() -> None:
    video = clip_refusal(11 * GIB // 2, 33, 64 * GIB)
    # (A driver that refuses takes no room rung, the segments' alone: the 224 the tail alone fits,
    # where the room lowered to 0.5 GiB gave 320.)
    assert isinstance(video, MemoryShort) and video.instead == Setting("resolution", "224")
    assert "set resolution to 224 or lower on this node" in str(words.refusal(video))
    still = picture(3 * GIB)
    assert isinstance(still, MemoryShort) and still.fits == (340, 192)
    said = str(words.refusal(still))
    assert "set resolution to 192 or lower on this node" in said and "no resolution" not in said


def test_the_probes_refusal_says_what_failed_and_what_to_do() -> None:
    text = str(words.refusal(refusals()["probe"]))
    assert "int8 linear gives the wrong numbers on a GPU" in text
    assert "ComfyUI's environment" in text and "measured before, at 2026-11-05T10:00:00 UTC" in text
    assert "delete that file to measure again" in text


def test_the_model_files_refusals_name_the_nodes_settings_and_ways() -> None:
    said = {
        name: str(words.refusal(error, model_dir=name != "model: not pinned"))
        for name, error in models_refused().items()
    }
    assert (
        'is not in the folder wired to the "model folder" input of seedvr2x Models'
        in said["model: missing"]
    )
    assert "unwire the input to use Hugging Face's cache" in said["model: missing"]
    assert "leave" not in said["model: missing"] and "empty" not in said["model: missing"]
    assert "the model in seedvr2x Models: not one of" in said["model: unknown"]
    assert (
        'give the "model folder" input of seedvr2x Models with a directory'
        in said["model: unknown"]
    )
    # From Hugging Face's cache: the file is fetched again once deleted from it, the cache named.
    assert "Delete it from Hugging Face's cache (" in said["model: not pinned"]
    assert "the next run fetches it again" in said["model: not pinned"]
    assert "hf download" not in said["model: not pinned"]
    # From a folder: the copy there is not the one seedvr2x pins.
    assert "in the model folder is not seedvr2x's own file" in said["model: not pinned, directory"]
    assert (
        'unwire the "model folder" input of seedvr2x Models to use Hugging Face\'s cache'
        in said["model: not pinned, directory"]
    )
    assert (
        "HF_HUB_OFFLINE" in said["model: offline"]
        and "fetches its files once" in said["model: offline"]
    )
    assert "HF_HUB_OFFLINE" not in said["model: unreachable"]
    assert 'give the "model folder" input of seedvr2x Models' in said["model: unreachable"]
    assert said["model: several"].count("\n") == 1
    host = str(words.refusal(refusals()["host copies"], keep=True))
    assert "need 15.00 GiB of your computer's RAM" in host and "4.00 GiB are available" in host
    assert "set keep in RAM between runs to off in seedvr2x Models" in host
    assert (
        "choose the int8 file (8.3 GB) or the dynamic GGUF file (4.8 GB) in seedvr2x Models" in host
    )
    assert "to off" not in str(words.refusal(refusals()["host copies"], keep=False))


def test_every_class_of_errors_py_is_worded_without_an_option() -> None:
    """The library's refusals by class: each one the test makes is a class of errors.py, and the
    classes it makes cover every concrete class there."""
    from seedvr2x import errors

    made = {type(error) for error in {**refusals(), **models_refused()}.values()}
    concrete = {
        each
        for each in vars(errors).values()
        if isinstance(each, type)
        and issubclass(each, Refusal)
        and each.__module__ == errors.__name__
    }
    # The bases carry no kind of their own: the generic ones are worded as any refusal.
    assert concrete - made <= {Refusal, errors.JobError, errors.PlanRefused, errors.ModelError}, (
        concrete - made
    )
    for error in {**refusals(), **models_refused()}.values():
        assert "--" not in str(words.refusal(error))


def test_what_the_nodes_refuse_themselves() -> None:
    for refused in (
        words.no_cuda("ComfyUI is running on the CPU"),
        words.not_freed(need=20 * GIB, there=7 * GIB, comfy=2 * GIB),
        words.not_freed(need=20 * GIB, there=7 * GIB, comfy=0),
        words.transparent(),
        words.channels(2),
        words.frames_missing(),
        words.non_finite("the picture's decode holds NaN or inf"),
        words.cuts_wrong(100, 120),
        words.crop_unread(),
        words.stream_unholdable("subtitle stream 2 (hdmv_pgs_subtitle)", "no Matroska mapping"),
    ):
        assert isinstance(refused, words.Refused) and "--" not in str(refused)
    assert "the 2.00 GiB that ComfyUI's models still use" in str(
        words.not_freed(need=20 * GIB, there=7 * GIB, comfy=2 * GIB)
    )
    assert "other programs' memory is what is left" in str(
        words.not_freed(need=20 * GIB, there=7 * GIB, comfy=0)
    )
    assert peaks.BLOCKS == 36 and plans.DIT_LABELS
    # The cut list made for another video: the words speak of the video, not of a clip's images.
    wrong = str(words.cuts_wrong(100, 120))
    assert "the cut list covers 100 frames and the video has 120" in wrong
    assert "images" not in wrong and "clip" not in wrong
    assert "connect the video that seedvr2x Cuts was given to this node too" in wrong
    # A crop this ComfyUI keeps out of reach is refused, never ignored, with the way out.
    unread = str(words.crop_unread())
    assert "Crop Video's" in unread and "refused rather than upscaled uncropped" in unread
    assert "crop the frames after this node" in unread
    # A stream Matroska can't hold: what it is, why, and the way out in ComfyUI's terms.
    unholdable = str(
        words.stream_unholdable("subtitle stream 2 (hdmv_pgs_subtitle)", "no Matroska mapping")
    )
    assert "the source's subtitle stream 2 (hdmv_pgs_subtitle) can't go into the upscaled file" in (
        unholdable
    )
    assert "(no Matroska mapping)" in unholdable and "load that file instead" in unholdable


# ---- The library's own refusals, from its real code paths ----------------------------------------
# Files written in a temporary directory and in a temporary Hugging Face cache, the plan with tiny
# figures and the target check, each refusal raised by the library as a run raises it and worded
# long after, as a node does: what the library's real messages say is what the nodes word.

DIT, VAE = "seedvr2x_ema_7b_sharp_fp16.safetensors", "seedvr2x_ema_vae_fp16.safetensors"


@pytest.fixture
def snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Hugging Face's cache in a temporary directory, as the library lays it out
    (huggingface_hub reads its constant when pull.cache() and the lookup ask), and the pinned
    revision's folder in it, empty."""
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    folder = tmp_path / "hub" / "models--hekmon--seedvr2x" / "snapshots" / pull.REVISION
    folder.mkdir(parents=True)
    return folder


def refused_by(where: Path | None, dit: str = DIT, vae: str = VAE) -> Refusal:
    """The refusal of the nodes' own sequence (models._prepare): the files found, their headers
    checked with the way to fetch each again, then each pinned file against its pin."""
    try:
        files = pull.resolve(where, dit, vae)
        weights.check_models(files.dit.path, files.vae.path, pull.advice(files), None)
        pull.check_pinned(files, load=True)
    except Refusal as error:
        return error
    raise AssertionError("the files were not refused")


def test_a_damaged_pinned_file_in_the_cache_keeps_the_way_to_fetch_it_again(
    snapshot: Path,
) -> None:
    (snapshot / DIT).write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00{}")
    (snapshot / VAE).write_bytes(b"short")
    error = refused_by(None)
    assert isinstance(error, ModelsRefused) and [type(each) for each in error.parts] == [
        ModelDamaged,
        ModelDamaged,
    ]
    # The library's parts end "fetch it again" with no way; the nodes say the way, and no command.
    lines = str(words.refusal(error)).splitlines()
    assert len(lines) == 2
    for line, name in zip(lines, (DIT, VAE), strict=True):
        assert name in line and "; fetch it again" not in line
        assert (
            f"delete it from Hugging Face's cache ({pull.cache()}) and the next run fetches it"
            in (line)
        )
        assert "hf download" not in line and " --" not in line and "remove" not in line


def test_a_damaged_pinned_file_in_a_directory_keeps_the_way_too(tmp_path: Path) -> None:
    (tmp_path / DIT).write_bytes(b"12345")
    (tmp_path / VAE).write_bytes(b"12345")
    error = refused_by(tmp_path)
    assert isinstance(error, ModelsRefused)
    assert all(isinstance(each, ModelDamaged) for each in error.parts)
    text = str(words.refusal(error, model_dir=True))
    assert text.count("replace it with seedvr2x's own file (https://huggingface.co/") == 2
    assert 'unwire the "model folder" input of seedvr2x Models to use Hugging Face\'s cache' in text
    assert "; fetch it again" not in text and " --" not in text


def test_a_file_seedvr2x_does_not_pin_keeps_the_librarys_words_when_damaged(
    tmp_path: Path,
) -> None:
    (tmp_path / "mine.safetensors").write_bytes(b"12345")
    (tmp_path / VAE).write_bytes(b"12345")
    error = refused_by(tmp_path, "mine.safetensors")
    assert isinstance(error, ModelsRefused)
    lines = str(words.refusal(error, model_dir=True)).splitlines()
    assert lines[0].endswith("; fetch it again") and "mine.safetensors" in lines[0]
    assert "replace it with seedvr2x's own file" in lines[1]


def test_files_missing_from_a_folder_are_worded_for_the_model_folder(tmp_path: Path) -> None:
    error = refused_by(tmp_path)
    assert isinstance(error, ModelsRefused) and all(
        isinstance(e, ModelMissing) for e in error.parts
    )
    lines = str(words.refusal(error, model_dir=True)).splitlines()
    assert len(lines) == 2
    assert all(
        'is not in the folder wired to the "model folder" input of seedvr2x Models' in x
        for x in lines
    )
    assert " --" not in "".join(lines)


def test_a_name_that_is_not_a_model_of_seedvr2x_is_worded_as_a_setting_of_models(
    snapshot: Path,
) -> None:
    error = refused_by(None, "x.safetensors")
    assert isinstance(error, ModelsRefused) and isinstance(error.parts[0], ModelUnknown)
    text = str(words.refusal(error))
    assert "the model in seedvr2x Models" in text and " --" not in text and "Setting(" not in text


def test_a_pinned_name_with_other_bytes_is_not_the_pinned_file_in_either_place(
    snapshot: Path, tmp_path: Path
) -> None:
    # Header-valid enough to pass the header check is not needed: the pin's size comes first in
    # check_pinned, which the nodes run after the header check, so here it is the one called.
    for where in (None, tmp_path / "dir"):
        folder = snapshot if where is None else where
        folder.mkdir(exist_ok=True)
        (folder / DIT).write_bytes(b"12345")
        (folder / VAE).write_bytes(b"12345")
        files = pull.resolve(where, DIT, VAE)
        with pytest.raises(ModelsRefused) as refused:
            pull.check_pinned(files)
        text = str(words.refusal(refused.value, model_dir=where is not None))
        assert "is not seedvr2x's own file" in text and " --" not in text
        assert ("in the model folder" in text) == (where is not None)
        assert ("Delete it from Hugging Face's cache" in text) == (where is None)
        assert "hf download" not in text


def test_a_cache_the_network_stays_out_of_is_worded_with_the_way_to_set_the_model_folder(
    snapshot: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    error = refused_by(None)
    assert isinstance(error, ModelUnreachable) and error.why == "offline"
    text = str(words.refusal(error))
    assert "HF_HUB_OFFLINE" in text
    assert 'wire a folder that has them to the "model folder" input of seedvr2x Models' in text
    assert " --" not in text


def test_a_target_too_small_for_the_padding_is_worded_from_the_real_check() -> None:
    from fractions import Fraction

    target = frames.target_size(40, 24, Fraction(1), 10)
    with pytest.raises(TargetTooSmall) as small:
        frames.check_target(target, 10)
    text = str(words.refusal(small.value))
    assert text.endswith("raise resolution on this node") and " --" not in text


def test_a_plan_with_tiny_figures_is_refused_by_the_real_planner_in_the_nodes_words() -> None:
    for error in (picture(1 * GIB), picture(24 * GIB, swap_blocks=99)):
        text = str(words.refusal(error))
        assert text and " --" not in text and "command" not in text.lower()
    # A window pin the planner can't take for a clip, and a host too small for its output.
    clip_error = clip_refusal(19 * GIB, 33, 64 * GIB, window=9, swap_blocks=0, encode_tile=512)
    assert "--" not in str(words.refusal(clip_error))


def full_type_name(klass: type) -> str:
    """The name ComfyUI's error report gives an exception by its class (v0.44.0's execution.py,
    full_type_name, read, not copied: GPL-3.0): the class's module, a dot and its qualified name;
    a builtin's qualified name alone."""
    module = klass.__module__
    return klass.__qualname__ if module == "builtins" else f"{module}.{klass.__qualname__}"


def test_comfyuis_error_report_names_the_nodes_exceptions_seedvr2x_never_their_folder() -> None:
    """A custom node's module is its folder's path in ComfyUI, so the report named a refusal
    "<their path>\\custom_nodes\\<folder>.words.Refused" (the user's fourth test, 2026-11-08):
    every exception class the nodes define, found in their source (their module no longer names
    their file), is seedvr2x.<Name> there."""
    folder = Path(comfyui.__file__).parent
    found: dict[str, type] = {}
    for path in sorted(folder.glob("*.py")):
        name = "comfyui" if path.stem == "__init__" else f"comfyui.{path.stem}"
        module = importlib.import_module(name)
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.ClassDef):
                value = getattr(module, node.name)
                if isinstance(value, type) and issubclass(value, BaseException):
                    found[node.name] = value
    assert set(found) == {"Refused", "BadSetting"}
    for name, klass in found.items():
        assert full_type_name(klass) == f"seedvr2x.{name}"
    # As a node raises it to ComfyUI: a value no Cuts takes.
    with pytest.raises(words.Refused) as refused:
        execute(
            nodes.Cuts,
            "7",
            video=frames_video(torch.zeros(2, 8, 8, 3)),
            models=models.Handle(plans.DEFAULT_DIT, ""),
            threshold=2.0,
        )
    named = full_type_name(type(refused.value))
    assert named == "seedvr2x.Refused" and "comfyui" not in named and "/" not in named


def _source_refusals() -> dict[str, Refusal]:
    """The library's real refusals of a video's frames, one of each kind of yuv.KINDS."""
    made: dict[str, Refusal] = {}
    cases = {
        "pixel-format, alpha": lambda: yuv.reading("yuva420p", 64, 16),
        "pixel-format, palette": lambda: yuv.reading("pal8", 64, 16),
        "pixel-format, grey": lambda: yuv.reading("gray", 64, 16),
        "matrix": lambda: yuv.reading("yuv420p", 64, 16, color_space="bt2020c"),
        "rgb-limited": lambda: yuv.reading("gbrp", 64, 16, color_range="tv"),
        "jpeg-limited": lambda: yuv.reading("yuvj420p", 64, 16, color_range="tv"),
        "rgb-matrix": lambda: yuv.reading("gbrp", 64, 16, matrix="bt709"),
        "size": lambda: yuv.check_size("yuv420p", 63, 16),
    }
    for name, make in cases.items():
        try:
            make()
        except Refusal as refused:
            made[name] = refused
    return made


SOURCE_BANNED = re.compile(
    r"\b(pins?|pinned|caps?|held|holds?|phases?|latents?|untiled|host|film)\b|--", re.IGNORECASE
)
SOURCE_KINDS = {
    "pixel-format, alpha": "pixel-format",
    "pixel-format, palette": "pixel-format",
    "pixel-format, grey": "pixel-format",
    "matrix": "matrix",
    "rgb-limited": "rgb-limited",
    "jpeg-limited": "jpeg-limited",
    "rgb-matrix": "rgb-matrix",
    "size": "size",
}


def test_every_kind_of_yuv_is_refused_by_the_library_and_covered_here() -> None:
    made = _source_refusals()
    assert set(made) == set(SOURCE_KINDS)
    assert {made[name].kind for name in made} == set(yuv.KINDS)
    assert all(made[name].kind == SOURCE_KINDS[name] for name in made)


@pytest.mark.parametrize("name", list(SOURCE_KINDS))
def test_a_source_refused_says_its_reason_then_both_ways_out_in_comfyuis_terms(name: str) -> None:
    error = _source_refusals()[name]
    said = words.refusal(error)
    text = str(said)
    assert type(said).__module__ == words.ERRORS_MODULE
    assert text.startswith("seedvr2x can't read this video file: ")
    assert error.reason in text
    assert not SOURCE_BANNED.search(text), text
    # The way every kind has: ComfyUI's own decode, its cost and its colours said.
    assert "Get Video Components" in text and "Create Video" in text
    assert "seedvr2x Upscale Video" in text
    assert "whole in RAM" in text and "BT.601" in text
    assert "ffmpeg" not in text and "command" not in text.lower()
    # And a file made again, in plain words: no command.
    assert "make the file again" in text.lower() or name == "rgb-matrix"


def test_each_kind_of_source_has_its_own_way_of_making_the_file_again() -> None:
    said = {name: str(words.refusal(error)) for name, error in _source_refusals().items()}
    assert "cropped to the multiples it needs" in said["size"]
    assert "63x16 in yuv420p" in said["size"]
    assert "has an alpha channel" in said["pixel-format, alpha"]
    assert "no alpha" in said["pixel-format, alpha"]
    assert "gives the alpha apart" in said["pixel-format, alpha"]
    for name in ("pixel-format, palette", "pixel-format, grey"):
        assert "ordinary colour format" in said[name]
        assert "alpha" not in said[name]
    assert "bt2020c" in said["matrix"] and "BT.709 is the usual one" in said["matrix"]
    for name in ("rgb-limited", "jpeg-limited"):
        assert "right range tag, without re-encoding" in said[name]
    assert "limited range" in said["jpeg-limited"] and "yuvj420p" in said["jpeg-limited"]
    assert "has no matrix" in said["rgb-matrix"]
    assert "a fault in seedvr2x" in said["rgb-matrix"]
    # The line the node shows is the nodes' own, not the message.
    assert words.refusal(_source_refusals()["size"]).line == words.REFUSED

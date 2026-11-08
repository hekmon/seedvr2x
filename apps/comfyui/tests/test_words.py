"""The nodes' words for every refusal of the library and every refusal of their own
(words.refusal and the rest): the refusal and its way out in ComfyUI's terms, from the class's
facts and the refusal's own words with the nodes' `say` for a setting, never an option of the
command line."""

from pathlib import Path

import pytest
from comfyui import plans, words
from standins import GIB

from seedvr2x import clip, frames, peaks, pull, still_plan, weights
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


def test_a_clips_output_that_the_host_cant_hold_is_worded_with_the_hosts_figures() -> None:
    error = clip_refusal(24 * GIB, 100_000, 8 * GIB)
    assert isinstance(error, HostShort)
    size = words.ClipSize(100_000, clip.output_bytes(100_000, TARGET))
    text = str(words.refusal(error, clip=size))
    assert text.startswith("The output of this clip, 100000 frames kept whole, takes 2317.43 GiB")
    assert "the computer has 8.00 GiB available" in text
    assert "--" not in text


def test_a_clip_the_host_cant_hold_is_told_what_the_command_line_does() -> None:
    """The clip's output and the run's own work summed, the input not counted (ComfyUI holds it
    already, so it is out of the RAM available), and what seedvr2x's command line does instead."""
    error = clip_refusal(24 * GIB, 33, 1 * GIB)
    assert isinstance(error, HostShort) and error.need is not None and error.available is not None
    size = words.ClipSize(33, clip.output_bytes(33, TARGET))
    text = str(words.refusal(error, clip=size))
    total = error.need + size.output
    assert text.startswith(f"This clip needs about {total / GIB:.2f} GiB of your computer's RAM")
    assert f"{size.output / GIB:.2f} GiB for the output, 33 frames kept whole" in text
    assert f"{(error.available + size.output) / GIB:.2f} GiB are available" in text
    # What the command line does instead, by its place beside the nodes and not by an address.
    for part in (
        "seedvr2x's command line",
        "apps/seedvr2x",
        "shot by shot and never keeps it whole",
        "picks up again after a stop",
        "writes lossless files",
    ):
        assert part in text, part
    assert "http" not in text and "give it fewer frames" in text
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
        words.attention_missing(),
        words.not_freed(need=20 * GIB, there=7 * GIB, comfy=2 * GIB),
        words.not_freed(need=20 * GIB, there=7 * GIB, comfy=0),
        words.transparent(),
        words.channels(2),
        words.frames_missing(),
        words.non_finite("the picture's decode holds NaN or inf"),
    ):
        assert isinstance(refused, words.Refused) and "--" not in str(refused)
    assert "the 2.00 GiB that ComfyUI's models still use" in str(
        words.not_freed(need=20 * GIB, there=7 * GIB, comfy=2 * GIB)
    )
    assert "other programs' memory is what is left" in str(
        words.not_freed(need=20 * GIB, there=7 * GIB, comfy=0)
    )
    assert peaks.BLOCKS == 36 and plans.DIT_LABELS


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

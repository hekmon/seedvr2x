"""The still path's command-line side. It uses the library's tests' helpers (core/tests):
test_still.py's stand-in models and DiT, stand_ins.py's VAE; the library's half is
tests/test_still.py there.

A picture through seedvr2x.still gives, bit for bit, what the shot's pass (seedvr2x/shot_pass.py),
as the command line runs it, gives for a shot of one frame: encode_shot, sample_windows over the one
window [(0, 1)], merge_windows, decode_shot, with its Correction when the picture is corrected,
tiled or not.

Then the command (stills.py, main._run's branch by content): a file told a picture by its content
and not its name, every option classified as taken or refused on one, -o's rules, the PNG written
at 16 or 8 bits with the picture's profile and orientation, --plan loading nothing, and no ffmpeg
asked for. The machine is stood in as test_cli_run.py stands it in (no GPU, no weights): the
command's run on the CPU, the pass itself the library's own on a VAE of random weights and a DiT
of arithmetic (test_still.py), or a cheap stand-in where the pass is not what is held."""

import hashlib
import logging
import os
import shlex
import shutil
import struct
import subprocess
import sys
import zlib
from collections.abc import Callable, Generator
from contextlib import contextmanager
from fractions import Fraction
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import torch
from PIL import Image, ImageCms
from posix_only import SYMLINKS
from stand_ins import cpu_models, shipped_cpu_models  # noqa: F401
from test_picture import exif_chunk, header_only_png, inserted
from test_record import read_xmp
from test_still import PICTURE, SEED, TARGET, TILES, pipeline_models, pixels_of, stand_in_dit

import main as cli
import shot
import stills
from media import ffmpeg
from media.ffmpeg import MediaError
from media.probe import IMAGE_BRANDS, IMAGE_DEMUXERS, PICTURES, image_kind
from seedvr2x import model, picture, png, record, still, webp, weights
from seedvr2x.errors import Refusal
from seedvr2x.frames import output_size
from seedvr2x.passes import Tiles
from seedvr2x.peaks import GIB
from seedvr2x.shot_pass import decode_shot, encode_shot, merge_windows, sample_windows
from shot import Correction

LAYOUT = [(0, 1)]


def shots_pipeline(
    models: model.Models,
    pixels: npt.NDArray[np.float32],
    seed: int,
    encode_tiles: Tiles | None,
    decode_tiles: Tiles | None,
    correction: Correction | None,
    target: tuple[int, int] = TARGET,
) -> npt.NDArray[np.float32]:
    """The picture as the command line runs it, a shot of one frame at frame 0: its frame."""
    frames = pixels[None]
    # The change of phase before each unit, as run.py makes it (Models.enter).
    models.enter("encode")
    latent = encode_shot(models, lambda count: frames, 1, target, seed, tiles=encode_tiles)
    models.enter("dit")
    sampled = list(sample_windows(models, latent, LAYOUT, seed))
    models.enter("decode")
    written: list[npt.NDArray[np.float32]] = []
    decode_shot(
        models,
        merge_windows(sampled, LAYOUT),
        1,
        target,
        written.append,
        reference=correction,
        tiles=decode_tiles,
    )
    assert len(written) == 1 and written[0].shape[0] == 1
    return written[0][0]


@pytest.mark.parametrize(
    ("encode_tiles", "decode_tiles", "correct"),
    [
        (None, None, False),
        (TILES, None, False),
        (None, TILES, False),
        (TILES, TILES, False),
        (None, None, True),
        (TILES, TILES, True),
    ],
)
def test_a_picture_is_a_shot_of_one_frame(
    monkeypatch: pytest.MonkeyPatch,
    encode_tiles: Tiles | None,
    decode_tiles: Tiles | None,
    correct: bool,
    cpu_models: model.Models,  # noqa: F811
) -> None:
    # The shot's input copy, which the correction re-reads through ffmpeg for a job, is read from
    # memory here (test_shot_cli.py does the same): the frame the encode read.
    pixels = pixels_of(7)

    @contextmanager
    def copy_reader(
        correction: Correction, count: int
    ) -> Generator[Callable[[int], npt.NDArray[np.float32]]]:
        yield lambda n: pixels[None][:n]

    monkeypatch.setattr(shot, "_copy_reader", copy_reader)
    stand_in_dit(monkeypatch)
    models = pipeline_models(cpu_models)
    correction = (
        Correction(Path("input.mkv"), Path("input.crc32"), PICTURE[1], PICTURE[0])
        if correct
        else None
    )
    expected = shots_pipeline(models, pixels, SEED, encode_tiles, decode_tiles, correction)
    ours = still.upscale(
        models,
        pixels,
        TARGET,
        SEED,
        encode_tiles=encode_tiles,
        decode_tiles=decode_tiles,
        correct=correct,
    )
    assert ours.shape == expected.shape == (48, 80, 3) and ours.dtype == expected.dtype
    assert ours.tobytes() == expected.tobytes()
    # Not a constant, nor the other correction's.
    assert ours.std() > 0.01
    other = shots_pipeline(
        models,
        pixels,
        SEED,
        encode_tiles,
        decode_tiles,
        None if correct else Correction(Path("i.mkv"), Path("i.crc32"), PICTURE[1], PICTURE[0]),
    )
    assert other.tobytes() != expected.tobytes()


def test_a_picture_is_a_shot_of_one_frame_at_the_vaes_shipped_widths(
    monkeypatch: pytest.MonkeyPatch,
    shipped_cpu_models: model.Models,  # noqa: F811
) -> None:
    # The cases above run the narrow stand-in VAE (stand_ins.NARROW); one runs it at the shipped
    # widths: both passes tiled, corrected.
    test_a_picture_is_a_shot_of_one_frame(monkeypatch, TILES, TILES, True, shipped_cpu_models)


# The picture on the GPU ----------------------------------------------------------------------

# The real models, the sharp 7B fp16 and the VAE seedvr2x ships (the command line's defaults), in
# SEEDVR2X_MODEL_DIR as the other GPU tests read it.
GPU_MODELS = os.environ.get("SEEDVR2X_MODEL_DIR")
GPU_DIT, GPU_VAE = "seedvr2x_ema_7b_sharp_fp16.safetensors", "seedvr2x_ema_vae_fp16.safetensors"
# The cases, the picture's size (height, width), the resolution it is upscaled to (x2) and the
# tiles pinned (encode, decode; None: the plan's own):
# - 720p, untiled in both passes by the plan;
# - 1080p, the plan's own passes, whatever it chooses (untiled on the RTX 5090 since 1b96b25,
#   one frame's figures: 21.39 GiB allocated against the decode's 21.52 predicted, impl-gate's
#   probe, 2026-11-04);
# - 1080p, both passes tiled: pinned to the tiles the plan chose when it counted the video
#   slice's VAE figures (an encode tile of 1728, a decode tile of 1024 at 20 GiB). On one frame's
#   figures (peaks.ONE_FRAME_UNTILED, 1b96b25) the plan takes 1080p untiled in 24 GiB, so the case
#   pins its tiles, the plan still placing the models around them.
GPU_CASES: dict[str, tuple[tuple[int, int], int, tuple[int, int] | None]] = {
    "720p-untiled": ((360, 640), 720, None),
    "1080p-plan": ((540, 960), 1080, None),
    "1080p-tiled": ((540, 960), 1080, (1728, 1024)),
}
# The cases whose passes are the plan's own, untiled or tiled as it chooses.
GPU_PLANS = {"1080p-plan"}
# The test plans as if this much of the card's free memory (the smaller of main._free_memory and
# NVML's) were taken by someone else: the card here drives a desktop as well.
GPU_SPARE = 4 * GIB
# Torch is held to the free memory less this, so that a plan that is wrong is an out-of-memory
# error of the test's, never the desktop's memory taken or a driver that spills.
GPU_HELD_BACK = 3 * GIB


def scene(height: int, width: int, seed: int) -> npt.NDArray[np.float32]:
    """A picture that looks like one, (H, W, 3) float32 in [0, 1]: smooth colour gradients, hard
    edged discs and bars, a fine texture and a little noise, from a seeded generator (nothing of
    it read from a file)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    y /= height
    x /= width
    pixels = np.stack(
        [0.2 + 0.6 * x, 0.3 + 0.4 * y + 0.2 * np.sin(6 * x), 0.8 - 0.5 * x * y], axis=-1
    ).astype(np.float32)
    for _ in range(12):
        cy, cx, radius = rng.random(), rng.random(), 0.03 + 0.12 * rng.random()
        disc = (y - cy) ** 2 + (x * width / height - cx * width / height) ** 2 < radius**2
        pixels[disc] = rng.random(3, dtype=np.float32)
    for _ in range(6):
        top, left = int(rng.integers(0, height - 8)), int(rng.integers(0, width - 8))
        pixels[top : top + 3 + int(rng.integers(2, 30)), left : left + 3] = rng.random(
            3, dtype=np.float32
        )
    pixels += 0.04 * np.sin(120 * x + 40 * y)[..., None]
    pixels += 0.02 * rng.standard_normal(pixels.shape, dtype=np.float32)
    return np.clip(pixels, 0, 1).astype(np.float32)


@pytest.mark.gpu
@pytest.mark.skipif(not GPU_MODELS, reason="needs SEEDVR2X_MODEL_DIR")
@pytest.mark.parametrize("case", GPU_CASES)
def test_a_picture_is_a_shot_of_one_frame_on_the_gpu(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """still.upscale with the real models on CUDA, with the colour correction, equals the shot of
    one frame of the command line bit for bit (float32 output), under the placement and the tiles
    still_plan gives in the free memory less GPU_SPARE, the models loaded once for both."""
    import gc
    from fractions import Fraction

    from seedvr2x import memory, peaks, still_plan
    from seedvr2x.frames import target_size

    assert GPU_MODELS is not None
    directory = Path(GPU_MODELS)
    dit, vae = directory / GPU_DIT, directory / GPU_VAE
    if not (dit.is_file() and vae.is_file()):
        pytest.skip(f"needs {GPU_DIT} and {GPU_VAE} in SEEDVR2X_MODEL_DIR")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    size, resolution, pinned = GPU_CASES[case]
    pixels = scene(*size, seed=11)
    target = target_size(size[1], size[0], Fraction(1), resolution)
    out_height, out_width = output_size(target)
    device = torch.device("cuda", 0)
    # The smaller of the driver's figure and the whole card's by NVML (nvidia-smi's): the WSL2
    # driver reports free memory the card doesn't have, as it lends system memory (memory.reading).
    reported = cli._free_memory(device)  # pyright: ignore[reportPrivateUsage]
    card = memory.card_free(0)  # read with this process's context on the card
    if card is None:
        pytest.skip("the card's free memory can't be read through NVML")
    free = min(reported, card)
    counted = free - GPU_SPARE
    host = cli._host_available()  # pyright: ignore[reportPrivateUsage]
    # The plan for this picture's output, as the command makes it (stills._run).
    plan = still_plan.plan(
        (max(out_height, out_width), min(out_height, out_width)),
        counted,
        host,
        process=stills._process(size, (out_height, out_width), 16),  # pyright: ignore[reportPrivateUsage]
        input_pixels=size[0] * size[1],
        dit_model=peaks.FP16,
        split=True,
        pins=still_plan.NO_PINS
        if pinned is None
        else still_plan.Pins(encode_tile=pinned[0], decode_tile=pinned[1]),
    )
    tiles = still_plan.tiles(plan, target)
    print(
        f"\n{case}: {size[1]}x{size[0]} to {out_width}x{out_height}; free {free / GIB:.2f} GiB,"
        f" planned in {counted / GIB:.2f} (budget {plan.budget / GIB:.2f}); {plan.models};"
        f" encode {tiles.encode} ({tiles.encode_peak / GIB:.2f} GiB), decode {tiles.decode}"
        f" ({tiles.decode_peak / GIB:.2f} GiB)"
    )
    if pinned is not None:
        assert tiles.encode is not None and tiles.decode is not None, "the case tiles both passes"
    elif case not in GPU_PLANS and (tiles.encode is not None or tiles.decode is not None):
        pytest.skip(f"the memory free plans {tiles}, not the untiled passes this case holds")

    torch.cuda.set_per_process_memory_fraction(
        (free - GPU_HELD_BACK) / torch.cuda.get_device_properties(device).total_memory, device
    )
    models = None
    try:
        models = model.load_models(dit, vae, device, None, plan.models)

        # The input copy the correction re-reads for a job is read from memory here, as the CPU
        # test does: the frame the encode read.
        @contextmanager
        def copy_reader(
            correction: Correction, count: int
        ) -> Generator[Callable[[int], npt.NDArray[np.float32]]]:
            yield lambda n: pixels[None][:n]

        monkeypatch.setattr(shot, "_copy_reader", copy_reader)
        ours = still.upscale(
            models,
            pixels,
            target,
            SEED,
            encode_tiles=tiles.encode,
            decode_tiles=tiles.decode,
            correct=True,
        )
        expected = shots_pipeline(
            models,
            pixels,
            SEED,
            tiles.encode,
            tiles.decode,
            Correction(Path("input.mkv"), Path("input.crc32"), size[1], size[0]),
            target,
        )
        print(f"peak allocated {torch.cuda.max_memory_allocated(device) / GIB:.2f} GiB")
    finally:
        del models
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(1.0, device)
    assert ours.shape == expected.shape == (out_height, out_width, 3)
    assert ours.dtype == expected.dtype == np.float32
    assert ours.std() > 0.05
    if not np.array_equal(ours, expected):
        difference = np.abs(ours - expected)
        place = np.unravel_index(int(difference.argmax()), difference.shape)
        raise AssertionError(
            f"{int((ours != expected).sum())} of {ours.size} values differ, at most"
            f" {float(difference.max())} at {place}: {ours[place]} against {expected[place]}"
        )


# The command ----------------------------------------------------------------------------------

# The GPU the stand-in machine has free and in all, as its driver reports them
# (torch.cuda.mem_get_info), in bytes: the 96 GB card's, which holds every plan a stand-in
# picture asks for (test_cli_run.py's FREE and TOTAL); and the system memory its host has.
FREE, TOTAL = round(94.5 * GIB), 96 * GIB
HOST = 64 * GIB


@pytest.fixture(autouse=True)
def log_everything(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)


class Machine:
    """The stand-in machine and what it was asked: the GPU's free memory read (`asked`, the device
    each time), the models loaded (`loads`, load_models's arguments) and the picture's passes
    (`passes`, still.upscale's arguments, with the keywords)."""

    def __init__(self, directory: Path) -> None:
        self.models_dir = directory / "models"
        self.models_dir.mkdir()
        (self.models_dir / "w.safetensors").write_bytes(b"")
        self.asked: list[torch.device] = []
        self.loads: list[tuple[Any, ...]] = []
        self.passes: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.free = FREE
        self.library: Any = None  # still.upscale, as the library has it
        self.models: Any = SimpleNamespace(device=torch.device("cpu"), enter=lambda phase: None)


def nearest(
    models: object,
    pixels: npt.NDArray[np.float32],
    target: tuple[int, int],
    seed: int,
    *,
    encode_tiles: Tiles | None,
    decode_tiles: Tiles | None,
    correct: bool,
) -> npt.NDArray[np.float32]:
    """A pass that takes no time: the picture's nearest-neighbour resize to the target's output
    size, so that each output says which picture it comes from."""
    height, width = output_size(target)
    rows = np.arange(height) * pixels.shape[0] // height
    columns = np.arange(width) * pixels.shape[1] // width
    return np.ascontiguousarray(pixels[rows][:, columns])


@pytest.fixture
def machine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Machine:
    """The command's machine stood in as test_cli_run.stand_in_model does for a video's: no GPU,
    no weights, the memory read through the same function (main._free_memory) on the CPU, the
    models' load a stand-in, and the picture's pass cheap (nearest); `use_library` puts the
    library's own back."""
    from seedvr2x import peaks

    made = Machine(tmp_path)
    made.library = still.upscale
    free_memory = cli._free_memory  # pyright: ignore[reportPrivateUsage]

    def read_free(device: torch.device) -> int:
        made.asked.append(device)
        return free_memory(torch.device("cpu"))

    def load(*arguments: Any) -> Any:
        # As the library's load: each file hashed as it is read, its SHA-256 proved or given at
        # its end (pull.Hashes.proved; DESIGN.md, Weights, decided on 2026-11-03).
        made.loads.append(arguments)
        hashes: Any = arguments[3] if len(arguments) > 3 else None
        for path in arguments[:2]:
            if hashes is not None:
                hashes.proved(path, hashlib.sha256(path.read_bytes()).hexdigest(), path.stat())
        return made.models

    def upscale(*arguments: Any, **keywords: Any) -> npt.NDArray[np.float32]:
        made.passes.append((arguments, keywords))
        return nearest(*arguments, **keywords)

    monkeypatch.setattr(weights, "check_models", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "a stand-in")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (made.free, TOTAL))
    monkeypatch.setattr(cli, "_free_memory", read_free)
    monkeypatch.setattr(cli, "_host_available", lambda: HOST)
    monkeypatch.setattr(cli, "_dit_model", lambda path: peaks.FP16)
    monkeypatch.setattr(model, "load_models", load)
    monkeypatch.setattr(still, "upscale", upscale)
    return made


def use_library(made: Machine, monkeypatch: pytest.MonkeyPatch, models: model.Models) -> None:
    """The library's own pass in the command, on `models` (a VAE of random weights and a DiT of
    arithmetic: test_still.py's), each call recorded."""
    real = made.library
    stand_in_dit(monkeypatch)
    made.models = models

    def upscale(*arguments: Any, **keywords: Any) -> npt.NDArray[np.float32]:
        made.passes.append((arguments, keywords))
        return real(*arguments, **keywords)

    monkeypatch.setattr(still, "upscale", upscale)


def make_picture(
    path: Path,
    *,
    bits: int = 16,
    grey: bool = False,
    icc: bytes | None = None,
    orientation: int = 1,
    seed: int = 0,
) -> Path:
    """A PNG of test_still.py's picture, 24 x 40, written by the library's writer, an eXIf chunk
    saying orientation put in after it when it is not 1 (the writer writes none)."""
    data = png.write(pixels_of(seed), bits=bits, grey=grey, icc=icc, chunks=())
    path.write_bytes(data if orientation == 1 else inserted(data, exif_chunk(orientation)))
    return path


def command(made: Machine, source: Path, output: Path, *options: str) -> list[str]:
    """The picture's command line: its short side 48, so that 24 x 40 is TARGET, 48 x 80."""
    return [
        *(str(source), "-o", str(output)),
        *("--model-dir", str(made.models_dir), "--dit-model", "w.safetensors"),
        *("--vae-model", "w.safetensors", "--resolution", "48", "--seed", str(SEED)),
        *(() if "--color-correction" in options else ("--color-correction", "none")),
        *options,
    ]


def run(made: Machine, source: Path, output: Path, *options: str) -> int:
    return cli.main(command(made, source, output, *options))


def read_back(path: Path) -> picture.Picture:
    return picture.read(path.read_bytes(), str(path))


# The stand-in machine's model file, w.safetensors, empty: the SHA-256 of nothing.
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def recorded(
    output: tuple[int, int], keywords: dict[str, Any], colour: str = "none"
) -> record.Record:
    """The record the command writes on the stand-in machine (DESIGN.md, The record in each
    output): this command's name and version, make_picture's 40 x 24 source, `output` (width,
    height) as written, command()'s --resolution 48 and square pixels, w.safetensors for both
    models, SEED, the colour correction, the tiles of the pass it ran (still.upscale's keywords),
    and the attention kernel, always SDPA."""

    def side(tiles: Tiles | None) -> int:
        return 0 if tiles is None else tiles.size

    model_file = record.Model("w.safetensors", EMPTY_SHA256)
    return record.Record(
        "seedvr2x",
        version("seedvr2x"),
        (40, 24),
        output,
        48,
        Fraction(1),
        model_file,
        model_file,
        SEED,
        colour,
        side(keywords["encode_tiles"]),
        side(keywords["decode_tiles"]),
    )


def png_written(pixels: npt.NDArray[np.float32], bits: int, made: record.Record) -> bytes:
    """png.write of pixels, colour, no profile or chunk, with the record made."""
    return png.write(
        pixels,
        bits=bits,
        grey=False,
        icc=None,
        chunks=(),
        xmp=made.xmp(),
        software=made.creator,
    )


def srgb_profile() -> bytes:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


# What is a picture ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "filename"),
    [
        ("PNG", "p.mkv"),
        ("JPEG", "p.mp4"),
        ("WEBP", "p.avi"),
        ("TIFF", "p.png"),
        ("BMP", "p.mov"),
        ("GIF", "p"),
    ],
)
def test_a_picture_is_told_by_its_content_not_its_name(
    machine: Machine, tmp_path: Path, name: str, filename: str
) -> None:
    # Every format the library reads, in a file named like a video, a picture's other suffix or
    # none: the command goes the picture's way (the GPU asked once, one pass, one PNG written).
    source = tmp_path / filename
    Image.fromarray((pixels_of() * 255).astype(np.uint8)).save(source, format=name)
    output = tmp_path / "out.png"
    assert run(machine, source, output) == 0
    assert len(machine.passes) == 1 and len(machine.loads) == 1
    written = read_back(output)
    assert written.pixels.shape == (48, 80, 3) and written.bits == 16 and not written.grey


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg to make a video")
def test_a_video_named_png_goes_to_the_video_path(tmp_path: Path) -> None:
    # The converse of the above, through test_cli_run's stand-in model: ffprobe reads it as the
    # video it is, whatever its name.
    from test_cli_run import source, stand_in_model, upscale

    mp = pytest.MonkeyPatch()
    try:
        stand_in_model(mp.setattr)
        renamed = source(tmp_path / "in.mkv").rename(tmp_path / "in.png")
        assert upscale(tmp_path, renamed, "out.mkv") == 0
        assert (tmp_path / "out.mkv").is_file()
        assert cli.main(["verify", str(tmp_path / "out.mkv")]) == 0
    finally:
        mp.undo()


def test_a_file_that_is_no_picture_goes_the_videos_way(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Not there, empty, text, a directory: none is identified (picture.identify), so each is the
    # video path's to refuse, as before; the picture's path says nothing of it.
    (tmp_path / "text.png").write_text("not a picture")
    (tmp_path / "empty.png").write_bytes(b"")
    for name in ("missing.png", "text.png", "empty.png"):
        caplog.clear()
        status = cli.main([str(tmp_path / name), "-o", str(tmp_path / "out"), "--format", "png"])
        assert status == 1
        assert "cli.stills" not in {record.name for record in caplog.records}


# The options ----------------------------------------------------------------------------------


def test_every_option_is_taken_or_refused_on_a_picture() -> None:
    # A walk of the parser: an option in neither list is one a picture would ignore, which
    # DESIGN.md forbids (Still images: "never ignored"), so a new option must be classified.
    actions = [
        action
        for action in cli._parser()._actions  # pyright: ignore[reportPrivateUsage]
        if action.dest not in ("help", "version")
    ]
    dests = {action.dest for action in actions}
    assert dests == stills.TAKEN | set(stills.REFUSED)
    assert not stills.TAKEN & set(stills.REFUSED)
    for action in actions:
        if action.dest in stills.REFUSED:
            flag, why = stills.REFUSED[action.dest]
            # The flag as the refusal says it is the option's own, and "given" (a value that is
            # not None or False) is what the parser's default is not.
            assert flag in action.option_strings and why
            assert action.default in (None, False), action.dest


def test_format_quality_min_segment_and_resolution_have_no_default_in_the_parser() -> None:
    # So that one given is told from its default; the video path takes its defaults at its start
    # (main._video_arguments), a picture takes --resolution's in stills.py, and each help says
    # them.
    parsed = cli._parser().parse_args(["in.mkv", "-o", "out"])  # pyright: ignore[reportPrivateUsage]
    assert (parsed.format, parsed.quality, parsed.min_segment, parsed.plan) == (None,) * 4
    assert parsed.resolution is None
    cli._video_arguments(parsed)  # pyright: ignore[reportPrivateUsage]
    assert (parsed.format, parsed.quality) == (cli.DEFAULT_FORMAT, cli.DEFAULT_QUALITY)
    assert parsed.min_segment == cli.MIN_SEGMENT and parsed.resolution == 1080
    assert (cli.DEFAULT_FORMAT, cli.DEFAULT_QUALITY, cli.MIN_SEGMENT) == (
        "yuv420p10le",
        "standard",
        5,
    )
    given = cli._parser().parse_args(["in.mkv", "-o", "out", "--resolution", "720"])  # pyright: ignore[reportPrivateUsage]
    cli._video_arguments(given)  # pyright: ignore[reportPrivateUsage]
    assert given.resolution == 720


# Each refused option with a value that parses, and a word of its reason.
GIVEN: dict[str, tuple[list[str], str]] = {
    "cuts": (["--cuts", "cuts.txt"], "no shots"),
    "cut_threshold": (["--cut-threshold", "0.5"], "no shots"),
    "segment_cmd": (["--segment-cmd", "cat > {out}"], "no video segments"),
    "quality": (["--quality", "maximum"], "encoded alone"),
    "min_segment": (["--min-segment", "2"], "no output segments"),
    "input_matrix": (["--input-matrix", "bt709"], "no YUV matrix"),
    "frame_rate": (["--frame-rate", "25"], "no frame rate"),
    "window": (["--window", "6"], "no windows"),
    "compile": (["--compile"], "not over one picture"),
    "until": (["--until", "07:00"], "one unit"),
    "accept_env_change": (["--accept-env-change"], "no job to resume"),
    "dump_frames": (["--dump-frames", "frames"], "the output file itself"),
}


def test_the_refused_options_are_all_given_here() -> None:
    assert set(GIVEN) == set(stills.REFUSED)


@pytest.mark.parametrize("dest", sorted(GIVEN))
def test_a_refused_option_is_refused_saying_why(
    machine: Machine,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    dest: str,
) -> None:
    options, why = GIVEN[dest]
    output = tmp_path / "out.png"
    assert run(machine, make_picture(tmp_path / "in.png"), output, *options) == 1
    flag = stills.REFUSED[dest][0]
    assert (
        f"{flag}: " in caplog.text and why in caplog.text and "refused for a picture" in caplog.text
    )
    # Before anything else: no GPU asked, nothing loaded or written.
    assert not machine.asked and not machine.loads and not machine.passes
    assert not output.exists() and sorted(p.name for p in tmp_path.iterdir()) == [
        "in.png",
        "models",
    ]


def test_several_refused_options_are_all_named(
    machine: Machine, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    status = run(
        machine, make_picture(tmp_path / "in.png"), tmp_path / "o.png", "--cuts", "c", "--compile"
    )
    assert status == 1
    assert "2 options refused for a picture: (1) --cuts: " in caplog.text
    assert "(2) --compile: " in caplog.text


@pytest.mark.parametrize("given", ["gbrp16le", "yuv420p10le"])
def test_a_videos_format_is_refused_on_a_picture(
    machine: Machine, tmp_path: Path, caplog: pytest.LogCaptureFixture, given: str
) -> None:
    status = run(machine, make_picture(tmp_path / "in.png"), tmp_path / "o.png", "--format", given)
    assert status == 1
    assert f"--format {given}: a video's format; a picture is written as a PNG" in caplog.text
    assert not machine.asked


def test_plan_with_a_file_is_refused_on_a_picture(
    machine: Machine, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    status = run(
        machine, make_picture(tmp_path / "in.png"), tmp_path / "o.png", "--plan", "cuts.txt"
    )
    assert status == 1
    assert "--plan cuts.txt: a picture has no cut list to write; --plan alone says its plan" in (
        caplog.text
    )
    assert not (tmp_path / "cuts.txt").exists() and not machine.asked


@pytest.mark.parametrize("seed", ["4293967296", "4294967295", "-1"])
def test_a_seed_out_of_range_is_refused_in_a_pictures_words(
    machine: Machine, tmp_path: Path, caplog: pytest.LogCaptureFixture, seed: str
) -> None:
    status = run(machine, make_picture(tmp_path / "in.png"), tmp_path / "o.png", "--seed", seed)
    assert status == 1 and not machine.asked
    assert (
        f"--seed {seed}: a picture's noise is drawn from its seed and its encode seeded with"
        " 1,000,000 more, so the seed is from 0 up to 4,293,967,295"
    ) in caplog.text
    assert "shot" not in caplog.text and "source" not in caplog.text


@pytest.mark.parametrize("seed", ["0", "4293967295"])
def test_the_seeds_at_the_picture_bounds_are_taken(
    machine: Machine, tmp_path: Path, seed: str
) -> None:
    # The same bound as a shot at frame 0 has (job.check_seed): the seed plus the encode's offset
    # under numz's limit.
    from job import check_seed
    from seedvr2x.shots import Shot

    check_seed(int(seed), [Shot(0, 1)])
    assert run(machine, make_picture(tmp_path / "in.png"), tmp_path / "o.png", "--seed", seed) == 0


def test_a_resolution_too_small_for_the_padding_is_refused(
    machine: Machine, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    status = run(
        machine, make_picture(tmp_path / "in.png"), tmp_path / "o.png", "--resolution", "8"
    )
    assert status == 1 and "too small for the padding" in caplog.text and not machine.asked


def test_a_resolution_whose_even_crop_is_too_small_is_refused_before_anything_is_asked(
    machine: Machine, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # 17 on the 24 x 40 picture is a 28 x 17 target and a 28 x 16 output, which the plan's bound
    # refuses with a ValueError: the picture's own check refuses it first, in the picture's words.
    status = run(
        machine, make_picture(tmp_path / "in.png"), tmp_path / "o.png", "--resolution", "17"
    )
    assert status == 1 and not machine.asked and not (tmp_path / "o.png").exists()
    assert "--resolution 17: the output 28x16, the 28x17 it is resized to cropped" in caplog.text
    assert "it takes 18 rows and 16 columns at least" in caplog.text
    assert "ValueError" not in caplog.text and "bound" not in caplog.text


# The size line: a default that makes the output smaller than the source ----------------------


def large_picture(path: Path, width: int, height: int) -> Path:
    Image.new("RGB", (width, height), (40, 90, 160)).save(path)
    return path


def without_resolution(made: Machine, source: Path, output: Path, *options: str) -> int:
    """The command with no --resolution unless `options` give one (command() always gives 48): its
    status."""
    arguments = [
        *(str(source), "-o", str(output)),
        *("--model-dir", str(made.models_dir), "--dit-model", "w.safetensors"),
        *("--vae-model", "w.safetensors", "--color-correction", "none", *options),
    ]
    return cli.main(arguments)


def size_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if "--resolution" in r.getMessage()]


def test_a_picture_larger_than_the_default_says_what_keeps_its_size(
    machine: Machine, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # 1100 x 1300 at the default 1080: 1080 x 1276 out, its short side 1100 as stored.
    source = large_picture(tmp_path / "photo.png", 1100, 1300)
    assert without_resolution(machine, source, tmp_path / "o.png") == 0
    assert size_lines(caplog) == [
        f"{source}: no --resolution given, so the output is 1080x1276, its short side the default"
        " 1080, smaller than its source's 1100x1300: --resolution 1100 keeps the source's size"
    ]
    assert read_back(tmp_path / "o.png").pixels.shape[:2] == (1276, 1080)


def test_the_size_line_comes_beside_the_pictures_line_and_in_plan_too(
    machine: Machine, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    source = large_picture(tmp_path / "photo.png", 1100, 1300)
    assert without_resolution(machine, source, tmp_path / "o.png", "--plan") == 0
    messages = [r.getMessage() for r in caplog.records]
    said = next(n for n, m in enumerate(messages) if "no --resolution given" in m)
    assert (
        "a PNG picture, 1100x1300" in messages[said - 1]
        and "output 1080x1276" in messages[said - 1]
    )
    assert not (tmp_path / "o.png").exists()


@pytest.mark.parametrize("given", ["1080", "540", "2160", "1100"])
def test_a_size_given_says_nothing_more(
    machine: Machine, tmp_path: Path, caplog: pytest.LogCaptureFixture, given: str
) -> None:
    source = large_picture(tmp_path / "photo.png", 1100, 1300)
    assert without_resolution(machine, source, tmp_path / "o.png", "--resolution", given) == 0
    assert size_lines(caplog) == []


@pytest.mark.parametrize("size", [(24, 40), (1080, 1500), (1500, 1080), (1000, 1080)])
def test_a_picture_the_default_does_not_make_smaller_says_nothing(
    machine: Machine, tmp_path: Path, caplog: pytest.LogCaptureFixture, size: tuple[int, int]
) -> None:
    # Smaller than 1080 on its short side (made larger by the default), or exactly 1080.
    source = large_picture(tmp_path / "photo.png", *size)
    assert without_resolution(machine, source, tmp_path / "o.png") == 0
    assert size_lines(caplog) == []


def test_the_short_side_is_the_one_as_displayed_with_a_sample_aspect(
    machine: Machine, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # 1100 x 1300 pixels, each twice as wide as tall: 2200 x 1300 displayed, short side 1300
    # (target_size's), output 1080 x 1826 (1080 * 2200 / 1300 = 1827.7, floored, even).
    source = large_picture(tmp_path / "photo.png", 1100, 1300)
    status = without_resolution(machine, source, tmp_path / "o.png", "--input-sar", "2/1")
    assert status == 0
    assert size_lines(caplog) == [
        f"{source}: no --resolution given, so the output is 1826x1080, its short side the default"
        " 1080, smaller than its source's 2200x1300: --resolution 1300 keeps the source's size"
    ]


def test_the_resolution_the_line_names_keeps_the_size(machine: Machine, tmp_path: Path) -> None:
    source = large_picture(tmp_path / "photo.png", 1100, 1300)
    status = without_resolution(machine, source, tmp_path / "o.png", "--resolution", "1100")
    assert status == 0
    assert read_back(tmp_path / "o.png").pixels.shape[:2] == (1300, 1100)


# -o -----------------------------------------------------------------------------------------------


def test_o_must_be_a_png_file_and_a_conversion_is_given(
    machine: Machine, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    source = make_picture(tmp_path / "in.png")
    assert run(machine, source, tmp_path / "photo.jpg") == 1
    assert (
        f"{tmp_path / 'photo.jpg'}: seedvr2x writes a picture as a PNG or a lossless WebP only:"
        f" write {tmp_path / 'photo.png'} (or {tmp_path / 'photo.webp'}), then convert it, e.g."
        f" ffmpeg -i {tmp_path / 'photo.png'} {tmp_path / 'photo.jpg'}"
    ) in caplog.text
    caplog.clear()
    # No suffix at all: the example names a format.
    assert run(machine, source, tmp_path / "photo") == 1
    assert f"write {tmp_path / 'photo.png'} (or {tmp_path / 'photo.webp'}), then" in caplog.text
    assert f"{tmp_path / 'photo.jpg'}" in caplog.text
    assert not machine.asked and sorted(p.name for p in tmp_path.iterdir()) == ["in.png", "models"]


def test_o_can_be_a_png_in_any_case(machine: Machine, tmp_path: Path) -> None:
    assert run(machine, make_picture(tmp_path / "in.png"), tmp_path / "OUT.PNG") == 0
    assert (tmp_path / "OUT.PNG").is_file()


def test_o_can_be_a_webp_in_any_case(machine: Machine, tmp_path: Path) -> None:
    source = make_picture(tmp_path / "in.png", bits=8)
    for place, name in enumerate(("out.webp", "OUT.WEBP", "Out.WebP")):
        # A directory each: Windows' names are one file whatever their case.
        output = tmp_path / str(place) / name
        output.parent.mkdir()
        assert run(machine, source, output) == 0
        assert read_back(output).format == "WEBP"


@SYMLINKS
def test_o_is_never_a_directory_the_source_or_a_file_there(
    machine: Machine, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    source = make_picture(tmp_path / "in.png")
    kept = source.read_bytes()
    (tmp_path / "dir.png").mkdir()
    (tmp_path / "there.png").write_bytes(b"mine")
    (tmp_path / "empty.png").write_bytes(b"")
    (tmp_path / "link.png").symlink_to(source)
    (tmp_path / "dangling.png").symlink_to(tmp_path / "nowhere")
    cases = {
        "dir.png": "a directory; a picture goes to a new .png or .webp file",
        "in.png": "the source itself; choose another -o",
        "link.png": "the source itself; choose another -o",
        "there.png": "a file already there, not replaced; remove it, or choose another -o",
        "empty.png": "a file already there, not replaced",
        "dangling.png": "a file already there, not replaced",
        "missing/out.png": "not a directory, so out.png can't be written there",
    }
    for name, said in cases.items():
        caplog.clear()
        assert run(machine, source, tmp_path / name) == 1, name
        assert said in caplog.text, name
    # Nothing replaced, nothing asked.
    assert source.read_bytes() == kept
    assert (tmp_path / "there.png").read_bytes() == b"mine" and not (
        tmp_path / "empty.png"
    ).stat().st_size
    assert (tmp_path / "dangling.png").is_symlink() and not (tmp_path / "nowhere").exists()
    assert not machine.asked and not list(tmp_path.glob("*.partial"))


def test_a_png_is_written_whole_and_nothing_is_left_on_a_failure(
    machine: Machine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import errno

    from media import files

    source = make_picture(tmp_path / "in.png")
    output = tmp_path / "out.png"
    seen: list[bool] = []
    real = files.replace_whole

    def replace(partial: Path, path: Path, stale: Path | None = None) -> None:
        # The data is in a partial file beside the name, not yet at it.
        seen.append(partial.is_file() and not path.exists())
        real(partial, path, stale)

    monkeypatch.setattr(files, "replace_whole", replace)
    assert run(machine, source, output) == 0 and seen == [True]
    assert not list(tmp_path.glob("*.partial"))
    output.unlink()

    def full(partial: Path, path: Path, stale: Path | None = None) -> None:
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    monkeypatch.setattr(files, "replace_whole", full)
    assert run(machine, source, output) == 1
    assert f"{output}: not written: No space left on device" in caplog.text
    assert not output.exists() and not list(tmp_path.glob("*.partial"))

    def stopped(partial: Path, path: Path, stale: Path | None = None) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(files, "replace_whole", stopped)
    assert run(machine, source, output) == 130
    assert not output.exists() and not list(tmp_path.glob("*.partial"))


def test_a_file_that_came_during_the_run_is_not_replaced(
    machine: Machine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    output = tmp_path / "out.png"

    def upscale(*arguments: Any, **keywords: Any) -> npt.NDArray[np.float32]:
        output.write_bytes(b"another program's")
        return nearest(*arguments, **keywords)

    monkeypatch.setattr(still, "upscale", upscale)
    assert run(machine, make_picture(tmp_path / "in.png"), output) == 1
    assert "written to since it was checked, by another program: not replaced" in caplog.text
    assert output.read_bytes() == b"another program's"


# The video path's new refusals ----------------------------------------------------------------


def test_a_video_with_a_webp_o_is_refused(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    status = cli.main([str(tmp_path / "film.mkv"), "-o", str(tmp_path / "X.WebP")])
    assert status == 1
    assert (
        f"{tmp_path / 'X.WebP'}: a .webp file is a picture's output, and a video's output is a"
        " .mkv file or a directory"
    ) in caplog.text
    assert not (tmp_path / "X.WebP").exists()


@pytest.mark.parametrize("name", ["x.png", "X.PNG", "frames.Png"])
def test_a_video_with_a_png_o_is_refused_naming_format_png_and_a_directory(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, name: str
) -> None:
    # Before ffmpeg and the file are looked at: the source need not exist.
    status = cli.main([str(tmp_path / "film.mkv"), "-o", str(tmp_path / name)])
    assert status == 1
    assert (
        f"{tmp_path / name}: a .png file is a picture's output, and a video's PNG frames go in a"
        " directory, one per frame: --format png, with -o naming a new or empty directory"
    ) in caplog.text
    assert not (tmp_path / name).exists()


def test_png8_is_refused_on_a_video(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    (tmp_path / "film.mkv").write_bytes(b"not a picture")
    status = cli.main([str(tmp_path / "film.mkv"), "-o", str(tmp_path / "out"), "--format", "png8"])
    assert status == 1
    assert (
        "--format png8: a video's PNG segments are 16-bit; png8, an 8-bit PNG, is for a still"
        " picture"
    ) in caplog.text


def test_plan_alone_is_refused_on_a_video(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    status = cli.main([str(tmp_path / "film.mkv"), "-o", str(tmp_path / "out"), "--plan"])
    assert status == 1
    assert "--plan alone: a video's plan writes its cut list, to the FILE it needs" in caplog.text


def test_the_formats_read_as_pictures_are_the_librarys() -> None:
    names = {"WEBP": "WebP"}
    said = [names.get(name, name) for name in picture.FORMATS]
    assert PICTURES == f"{', '.join(said[:-1])} and {said[-1]}"


@pytest.mark.parametrize(
    ("container", "kind"),
    [
        ({"format_name": "png_pipe"}, "read by ffprobe as png_pipe"),
        ({"format_name": "jpegxl_pipe"}, "read by ffprobe as jpegxl_pipe"),
        ({"format_name": "exr_pipe"}, "read by ffprobe as exr_pipe"),
        ({"format_name": "image2"}, "read by ffprobe as image2"),
        ({"format_name": "gif"}, "read by ffprobe as gif"),
        (
            {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "tags": {"major_brand": "avif", "compatible_brands": "avifmif1miafMA1A"},
            },
            "an MP4-family file of brand avif",
        ),
        (
            {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "tags": {"major_brand": "mif1", "compatible_brands": "mif1miafheic"},
            },
            "an MP4-family file of brand mif1",
        ),
        # A video whose writer lists a sequence or image brand among its compatible ones is a
        # video: only the major brand tells an image (media/probe.image_kind).
        (
            {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "tags": {"major_brand": "isom", "compatible_brands": "isomiso2mif1"},
            },
            "",
        ),
        (
            {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "tags": {"major_brand": "mp42", "compatible_brands": "mp42isomavc1msf1"},
            },
            "",
        ),
        (
            {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "tags": {"major_brand": "isom", "compatible_brands": "isomhevchevx"},
            },
            "",
        ),
        (
            {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "tags": {"major_brand": "heic", "compatible_brands": "mif1heic"},
            },
            "an MP4-family file of brand heic",
        ),
        ({"format_name": "matroska,webm"}, ""),
        ({"format_name": "yuv4mpegpipe"}, ""),
        ({"format_name": "mpegts"}, ""),
        ({"format_name": "avi"}, ""),
        (
            {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "tags": {"major_brand": "isom", "compatible_brands": "isomiso2avc1mp41"},
            },
            "",
        ),
        (
            {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "tags": {"major_brand": "qt  ", "compatible_brands": ""},
            },
            "",
        ),
    ],
)
def test_the_image_formats_ffprobe_reads_are_told_from_videos(
    container: dict[str, Any], kind: str
) -> None:
    assert image_kind(container) == kind
    assert all(len(brand) == 4 for brand in IMAGE_BRANDS)
    assert not any(name.endswith("_pipe") for name in IMAGE_DEMUXERS)


def encoded(tmp_path: Path, name: str, *options: str) -> Path:
    """A one-frame file made by ffmpeg with `options`, or a skip where this build can't."""
    path = tmp_path / name
    made = subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=64x48:r=25"),
            *("-frames:v", "2", *options, str(path)),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if made.returncode != 0 or not path.is_file():
        pytest.skip(f"this ffmpeg can't make {name}: {made.stderr.strip()[:200]}")
    return path


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg to make the files")
@pytest.mark.parametrize(
    ("name", "options", "said"),
    [
        ("p.jxl", ("-frames:v", "1", "-c:v", "libjxl"), "jpegxl_pipe"),
        ("p.avif", ("-frames:v", "1", "-c:v", "libaom-av1", "-still-picture", "1"), "brand avif"),
        ("avif.mp4", ("-c:v", "mpeg4", "-brand", "avif"), "brand avif"),
        ("heic.mp4", ("-c:v", "mpeg4", "-brand", "heic"), "brand heic"),
    ],
)
def test_an_image_format_that_is_no_picture_is_refused_by_the_video_path(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    name: str,
    options: tuple[str, ...],
    said: str,
) -> None:
    # Made by ffmpeg where this build can, or a file whose brand says so: refused naming the
    # formats read as pictures, before ffmpeg could run it as a video of one frame.
    if not _has_ffmpeg():
        pytest.skip("needs ffmpeg 7.1 or later with the filters seedvr2x checks")
    path = encoded(tmp_path, name, *options)
    assert cli.main([str(path), "-o", str(tmp_path / "out")]) == 1
    assert "an image file" in caplog.text and said in caplog.text
    assert "it reads PNG, JPEG, WebP, TIFF, BMP and GIF as pictures" in caplog.text
    assert not (tmp_path / "out").exists()


def _has_ffmpeg() -> bool:
    try:
        ffmpeg.check(("png",))
    except MediaError:
        return False
    return True


def test_the_animated_refusals_command_makes_a_file_the_video_path_accepts(tmp_path: Path) -> None:
    # A GIF of three frames with uneven delays (10, 250 and 40 hundredths of a second ...), the
    # case the command's fixed rate is for: the picture path refuses it with an ffmpeg command,
    # and the file that makes is a video the video path's probe and declaration take as it is.
    from media.source import declare

    if not _has_ffmpeg():
        pytest.skip("needs ffmpeg 7.1 or later with the filters seedvr2x checks")
    shown = [Image.fromarray((pixels_of(k) * 255).astype(np.uint8)) for k in range(3)]
    source = tmp_path / "an imated.gif"
    shown[0].save(source, save_all=True, append_images=shown[1:], duration=[100, 2500, 400])
    with pytest.raises(Refusal) as caught:
        picture.read(source.read_bytes(), source.name)
    message = str(caught.value)
    command = shlex.split(message[message.index("ffmpeg -i ") :])
    assert command[:3] == ["ffmpeg", "-i", source.name] and command[-1] == f"{source.name}.mkv"
    subprocess.run([*command[:1], "-v", "error", *command[1:]], cwd=tmp_path, check=True)
    video = tmp_path / command[-1]
    declared = declare(video)
    assert declared.stream.codec_name == "ffv1" and declared.stream.frame_rate == 10
    # fps=10,format=rgb24 into FFV1: packed RGB with an unused byte, ffmpeg 9.0's choice; the
    # video path takes the RGB formats of FFV1 whichever it picks.
    assert declared.stream.pix_fmt in ("bgr0", "rgb24", "bgra", "gbrp")
    assert (declared.stream.width, declared.stream.height) == (PICTURE[1], PICTURE[0])


# The run --------------------------------------------------------------------------------------


def test_the_library_makes_the_png_the_command_writes(
    machine: Machine,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    cpu_models: model.Models,  # noqa: F811
) -> None:
    # The command on the library's own pass (a VAE of random weights, a DiT of arithmetic): the
    # file is png.write of what still.upscale gives for the plan's tiles, at 16 bits, the
    # colour correction on, and the pass is the one the command planned for the picture.
    models = pipeline_models(cpu_models)
    use_library(machine, monkeypatch, models)
    source = make_picture(tmp_path / "in.png", seed=3)
    output = tmp_path / "out.png"
    assert run(machine, source, output, "--color-correction", "split") == 0
    ((arguments, keywords),) = machine.passes
    pixels, target, seed = arguments[1:4]
    assert target == TARGET and seed == SEED and keywords["correct"] is True
    again = still.upscale(models, pixels, target, seed, **keywords)
    expected = png_written(again, 16, recorded((80, 48), keywords, "split"))
    assert output.read_bytes() == expected
    # The pixels are the model's, not the picture's.
    assert not np.array_equal(
        read_back(output).pixels, nearest(models, pixels, target, seed, **keywords)
    )
    assert read_back(output).pixels.shape == (48, 80, 3)
    # One load, as the plan placed the models, the shot detector's file never asked for.
    (loaded,) = machine.loads
    assert loaded[0].name == "w.safetensors" and loaded[1].name == "w.safetensors"
    assert "transnetv2" not in caplog.text.lower()
    # The lines a user reads.
    lines = [r.getMessage() for r in caplog.records if r.levelno >= logging.INFO]
    assert any(line.startswith("plan: picture: 80x48, padded to 80x80") for line in lines)
    assert any(
        line.startswith("upscaled in ") and f"wrote {output}: 80x48, 16-bit RGB" in line
        for line in lines
    )


def test_png8_writes_8_bits_rounded_to_the_nearest_code(machine: Machine, tmp_path: Path) -> None:
    source = make_picture(tmp_path / "in.png", seed=4)
    sixteen, eight = tmp_path / "16.png", tmp_path / "8.png"
    assert run(machine, source, sixteen) == 0
    assert run(machine, source, eight, "--format", "png8") == 0
    assert (read_back(sixteen).bits, read_back(eight).bits) == (16, 8)
    # The same pixels, the 8-bit one the nearest code of the 16-bit one's (what the pass made).
    made = nearest(
        None,
        read_back(source).pixels,
        TARGET,
        SEED,
        encode_tiles=None,
        decode_tiles=None,
        correct=False,
    )
    keywords = machine.passes[-1][1]
    assert eight.read_bytes() == png_written(made, 8, recorded((80, 48), keywords))
    with Image.open(eight) as image:
        assert image.mode == "RGB" and image.size == (80, 48)


def test_an_8_bit_source_is_written_at_16_bits_by_default(machine: Machine, tmp_path: Path) -> None:
    # All the model made (DESIGN.md, Still images, Output), whatever the source's depth.
    source = tmp_path / "in.jpg"
    Image.fromarray((pixels_of() * 255).astype(np.uint8)).save(source, quality=95)
    assert run(machine, source, tmp_path / "out.png") == 0
    assert read_back(tmp_path / "out.png").bits == 16


def test_a_grey_picture_comes_back_grey(machine: Machine, tmp_path: Path) -> None:
    source = tmp_path / "grey.png"
    Image.fromarray((pixels_of()[..., 0] * 255).astype(np.uint8), "L").save(source)
    assert read_back(source).grey
    for options, bits, mode in (((), 16, "I;16"), (("--format", "png8"), 8, "L")):
        output = tmp_path / f"out{bits}.png"
        assert run(machine, source, output, *options) == 0
        written = read_back(output)
        assert written.grey and written.bits == bits
        with Image.open(output) as image:
            assert image.mode == mode and image.size == (80, 48)


def test_the_profile_is_carried_and_the_picture_written_upright(
    machine: Machine, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    profile = srgb_profile()
    source = make_picture(tmp_path / "in.png", icc=profile, orientation=6)
    output = tmp_path / "out.png"
    assert run(machine, source, output) == 0
    written = read_back(output)
    # Processed as stored (24 x 40 in, 80 x 48 out), then turned upright: 48 wide, 80 high, with
    # no orientation written.
    assert written.icc == profile and written.orientation == 1
    assert written.pixels.shape == (80, 48, 3)
    with Image.open(output) as image:
        assert image.info["icc_profile"] == profile and 0x0112 not in image.getexif()
        assert image.size == (48, 80)
    assert b"eXIf" not in output.read_bytes()
    lines = [r.getMessage() for r in caplog.records]
    assert any(
        line.startswith(f"{source}: a PNG picture")
        and line.endswith(
            "with its ICC profile and its orientation, 6; output 48x80 (x2), a 16-bit PNG"
        )
        for line in lines
    ), lines
    line = next(line for line in lines if line.startswith("upscaled in"))
    assert line.endswith(
        f"wrote {output}: 48x80, 16-bit RGB, with its ICC profile, turned upright from its"
        " orientation 6"
    )


@pytest.mark.parametrize("suffix", ["png", "webp"])
@pytest.mark.parametrize("value", range(1, 9))
def test_every_orientation_is_written_upright_with_no_tag(
    machine: Machine,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    suffix: str,
    value: int,
) -> None:
    """The 8 orientations, in PNG and in lossless WebP: the output is the output of the same
    picture stored upright, turned by picture.upright, with no orientation tag whatever (no eXIf
    chunk, no EXIF chunk), and the result line says it was turned, nothing at 1."""
    upright_source = make_picture(tmp_path / "up.png", bits=8)
    stored = make_picture(tmp_path / "stored.png", bits=8, orientation=value)
    plain = tmp_path / f"plain.{suffix}"
    output = tmp_path / f"out.{suffix}"
    assert run(machine, upright_source, plain) == 0
    caplog.clear()
    assert run(machine, stored, output) == 0
    turned = picture.upright(read_back(plain).pixels, value)
    written = read_back(output)
    assert written.orientation == 1 and written.pixels.shape == turned.shape
    assert turned.shape == ((80, 48, 3) if value >= 5 else (48, 80, 3))
    assert np.array_equal(written.pixels, turned)
    with Image.open(output) as image:
        assert 0x0112 not in image.getexif() and "exif" not in image.info
    assert b"eXIf" not in output.read_bytes() and b"EXIF" not in output.read_bytes()
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("upscaled in"))
    size = "48x80" if value >= 5 else "80x48"
    assert f"wrote {output}: {size}, " in line
    if value == 1:
        assert "turned" not in line and ", its orientation" not in line
    else:
        assert line.endswith(f", turned upright from its orientation {value}")


def written_record(path: Path) -> tuple[str, str, list[tuple[str, str]], str | None]:
    """The record a written picture holds, by test_record.read_xmp: its XMP packet, from the PNG's
    iTXt chunk or the WebP's XMP chunk as Pillow reads them, and the PNG's Software text (None for
    a WebP)."""
    with Image.open(path) as image:
        image.load()
        if image.format == "PNG":
            packet = image.info["XML:com.adobe.xmp"].encode()
            software = image.info["Software"]
        else:
            packet, software = image.info["xmp"], None
    return (*read_xmp(packet), software)


@pytest.mark.parametrize("suffix", ["png", "webp"])
@pytest.mark.parametrize("value", [1, 6])
def test_the_file_carries_seedvr2xs_record(
    machine: Machine,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    suffix: str,
    value: int,
) -> None:
    """DESIGN.md, The record in each output: the version and the address, the mark, the source's
    size as stored and the output's as written (upright: 48 x 80 at orientation 6), the model
    files by name and SHA-256, the seed, the colour correction, the tiles (0 untiled) and the
    attention kernel, written out here; the model files hashed by their load alone, which reads
    each once (DESIGN.md, Weights, decided on 2026-11-03): no whole read of a file before it."""
    source = make_picture(tmp_path / "in.png", bits=8, orientation=value)
    output = tmp_path / f"out.{suffix}"
    assert run(machine, source, output, "--color-correction", "split") == 0
    tool, mark, fields, software = written_record(output)
    creator = f"seedvr2x {version('seedvr2x')} (https://github.com/hekmon/seedvr2x)"
    assert tool == creator
    assert software == (creator if suffix == "png" else None)
    assert mark == (
        "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia"
    )
    assert fields == [
        ("source_size", "40x24"),
        ("output_size", "48x80" if value == 6 else "80x48"),
        ("resolution", "48"),
        ("sample_aspect", "1:1"),
        ("dit_model", "w.safetensors"),
        ("dit_sha256", EMPTY_SHA256),
        ("vae_model", "w.safetensors"),
        ("vae_sha256", EMPTY_SHA256),
        ("seed", str(SEED)),
        ("color_correction", "split"),
        ("encode_tile", "0"),
        ("decode_tile", "0"),
        ("attention", "sdpa"),
    ]
    messages = [r.getMessage() for r in caplog.records]
    assert f"w.safetensors: SHA-256 {EMPTY_SHA256}, read as it loaded" in messages
    assert not [m for m in messages if m.startswith("w.safetensors: SHA-256") and " in " in m]


def test_the_record_names_the_tiles_of_the_pass(
    machine: Machine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A picture's tiles, as the plan chose them for it, are the record's: the plan's choice stood
    # in by tiles of 64 and 48 px, with TILES' overlap, which the 80 x 80 padded frame takes.
    from seedvr2x import still_plan

    def tiled(plan: Any, target: tuple[int, int]) -> still_plan.PictureTiles:
        return still_plan.PictureTiles(Tiles(64, TILES.overlap), Tiles(48, TILES.overlap), 0, 0)

    monkeypatch.setattr(still_plan, "tiles", tiled)
    output = tmp_path / "out.png"
    assert run(machine, make_picture(tmp_path / "in.png"), output) == 0
    ((_, keywords),) = machine.passes
    assert keywords["encode_tiles"].size == 64 and keywords["decode_tiles"].size == 48
    fields = dict(written_record(output)[2])
    assert (fields["encode_tile"], fields["decode_tile"]) == ("64", "48")
    assert fields["color_correction"] == "none"


def test_the_record_names_the_resolution_asked_and_the_sample_aspect(
    machine: Machine, tmp_path: Path
) -> None:
    # The short side asked and the sample aspect used, which the sizes alone don't pin
    # (DESIGN.md, The record in each output): --resolution 48 and 49 both give 80x48 from the
    # 40 x 24 picture (the even crop), with other bits; --input-sar is recorded as given.
    source = make_picture(tmp_path / "in.png")
    asked = {}
    for name, options in (("48", ()), ("49", ("--resolution", "49", "--input-sar", "4:3"))):
        output = tmp_path / f"{name}.png"
        assert run(machine, source, output, *options) == 0
        fields = dict(written_record(output)[2])
        asked[name] = (fields["output_size"], fields["resolution"], fields["sample_aspect"])
    assert asked["48"] == ("80x48", "48", "1:1")
    assert asked["49"][1:] == ("49", "4:3")


def test_plan_hashes_no_model_file(
    machine: Machine, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # The record is made for a file written: --plan, which writes none, hashes no model file for
    # it (the stand-in's file is no pinned one, which pull.check_pinned would hash before).
    assert run(machine, make_picture(tmp_path / "in.png"), tmp_path / "o.png", "--plan") == 0
    assert "SHA-256" not in caplog.text


@pytest.mark.parametrize("suffix", ["png", "webp"])
def test_the_library_makes_the_turned_file_the_command_writes(
    machine: Machine,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cpu_models: model.Models,  # noqa: F811
    suffix: str,
) -> None:
    # The file is the writer's of what still.upscale gives, turned by picture.upright, byte for
    # byte: the writers turn nothing and tag nothing.
    models = pipeline_models(cpu_models)
    use_library(machine, monkeypatch, models)
    source = make_picture(tmp_path / "in.png", seed=3, bits=8, orientation=6)
    output = tmp_path / f"out.{suffix}"
    assert run(machine, source, output, "--color-correction", "split") == 0
    ((arguments, keywords),) = machine.passes
    pixels, target, seed = arguments[1:4]
    turned = picture.upright(still.upscale(models, pixels, target, seed, **keywords), 6)
    # Written upright: 48 x 80.
    made = recorded((48, 80), keywords, "split")
    if suffix == "png":
        expected = png_written(turned, 16, made)
    else:
        expected = webp.write(turned, grey=False, icc=None, chunks=(), xmp=made.xmp())
    assert output.read_bytes() == expected


def test_an_opaque_alpha_plane_is_dropped_and_said(
    machine: Machine, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    source = tmp_path / "rgba.png"
    rgba = np.dstack([(pixels_of() * 255).astype(np.uint8), np.full(PICTURE, 255, np.uint8)])
    Image.fromarray(rgba, "RGBA").save(source)
    assert run(machine, source, tmp_path / "out.png") == 0
    assert (
        f"{source}: its alpha plane is opaque everywhere, left out: the picture is unchanged"
        in (caplog.text)
    )
    rgba[3, 3, 3] = 254
    Image.fromarray(rgba, "RGBA").save(source)
    caplog.clear()
    (tmp_path / "out.png").unlink()
    assert run(machine, source, tmp_path / "out.png") == 1
    assert "has transparent pixels, and seedvr2x refuses it" in caplog.text
    assert not (tmp_path / "out.png").exists()


@pytest.mark.parametrize("frames", [2])
def test_a_picture_the_library_refuses_is_refused_as_a_job_is(
    machine: Machine, tmp_path: Path, caplog: pytest.LogCaptureFixture, frames: int
) -> None:
    source = tmp_path / "anim.gif"
    shown = [Image.fromarray((pixels_of(k) * 255).astype(np.uint8)) for k in range(frames)]
    shown[0].save(source, save_all=True, append_images=shown[1:])
    assert run(machine, source, tmp_path / "out.png") == 1
    assert f"{source} holds {frames} frames, and seedvr2x reads one picture" in caplog.text
    assert not machine.asked and not (tmp_path / "out.png").exists()


def test_a_second_image_in_a_jpeg_is_read_as_the_first_and_said(
    machine: Machine, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    source = tmp_path / "pair.jpg"
    first = Image.fromarray((pixels_of() * 255).astype(np.uint8))
    first.save(source, format="MPO", save_all=True, append_images=[first.copy()])
    assert run(machine, source, tmp_path / "out.png") == 0
    assert "2 images in the file" in caplog.text and "the first is read, 1 left out" in caplog.text


def test_the_librarys_info_lines_reach_the_user_on_stderr(tmp_path: Path) -> None:
    # Through main._logging in a process of its own (pytest's own handler would stand in the way
    # in this one): the library's INFO line from seedvr2x.picture, said before the model files
    # (an empty directory here) are asked for.
    source = tmp_path / "rgba.png"
    rgba = np.dstack([(pixels_of() * 255).astype(np.uint8), np.full(PICTURE, 255, np.uint8)])
    Image.fromarray(rgba, "RGBA").save(source)
    (tmp_path / "models").mkdir()
    done = subprocess.run(
        [
            Path(sys.executable).with_name("seedvr2x"),
            *(
                str(source),
                "-o",
                str(tmp_path / "out.png"),
                "--model-dir",
                str(tmp_path / "models"),
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "HF_HOME": str(tmp_path / "hf")},
    )
    assert f"{source}: its alpha plane is opaque everywhere, left out" in done.stderr
    assert done.returncode == 1 and "Traceback" not in done.stderr


def test_pillows_size_warning_is_replaced_by_a_line_of_seedvr2xs_own(tmp_path: Path) -> None:
    # In a process of its own (the filter is installed by main.main, once, for the process): a PNG
    # whose IHDR says 10,000 x 10,000, over Pillow's check (89,478,485 pixels) and under its
    # refusal, with image data that is empty, so that the run stops at the read, refused as
    # damaged, with no 100-megapixel decode.
    source = tmp_path / "big.png"
    source.write_bytes(header_only_png(10_000, 10_000))
    (tmp_path / "models").mkdir()
    done = subprocess.run(
        [
            Path(sys.executable).with_name("seedvr2x"),
            *(str(source), "-o", str(tmp_path / "out.png")),
            *("--model-dir", str(tmp_path / "models")),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "HF_HOME": str(tmp_path / "hf")},
    )
    assert f"{source}: 10000x10000, 100.0 megapixels, over the 89.5 at which Pillow" in done.stderr
    assert "damaged or truncated" in done.stderr and done.returncode == 1
    assert "decompression bomb" not in done.stderr.lower()
    assert "DecompressionBombWarning" not in done.stderr and "Traceback" not in done.stderr


def test_the_filter_is_installed_once_for_the_process_by_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Not the library's, per read: warnings.catch_warnings is not thread-safe. main.main installs
    # the filter before it runs anything, and a read of such a picture then raises no warning.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("default")
        cli._pillow_size_warning_off()  # pyright: ignore[reportPrivateUsage]
        assert warnings.filters[0][2] is Image.DecompressionBombWarning  # pyright: ignore[reportUnknownMemberType]
        assert warnings.filters[0][0] == "ignore"
        with warnings.catch_warnings(record=True) as seen:
            warnings.warn("size", Image.DecompressionBombWarning, stacklevel=1)
            warnings.warn("other", RuntimeWarning, stacklevel=1)
        assert [str(w.message) for w in seen] == ["other"]


def test_no_ffmpeg_is_needed(
    machine: Machine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "empty-path"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    assert shutil.which("ffmpeg") is None and shutil.which("ffprobe") is None
    source = make_picture(tmp_path / "in.png")
    assert run(machine, source, tmp_path / "out.png") == 0
    # And the video path, on the same PATH, is refused for the build it needs.
    (tmp_path / "film.mkv").write_bytes(b"not a picture")
    assert cli.main([str(tmp_path / "film.mkv"), "-o", str(tmp_path / "v")]) == 1


# --plan ---------------------------------------------------------------------------------------


def test_plan_says_the_pictures_plan_and_loads_and_writes_nothing(
    machine: Machine, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    source = make_picture(tmp_path / "in.png")
    output = tmp_path / "out.png"
    assert run(machine, source, output, "--plan") == 0
    assert not output.exists() and sorted(p.name for p in tmp_path.iterdir()) == [
        "in.png",
        "models",
    ]
    assert not machine.loads and not machine.passes
    # The memory was read, as a video's --plan reads it, to plan.
    assert machine.asked
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("plan: ")]
    assert lines[0] == "plan: card: a stand-in"
    for opening in (
        "memory: ",
        "bound: the picture alone, 80x48 as stored",
        "blocks swapped: ",
        "the DiT's phase: ",
        "system memory: ",
        "picture: 80x48, padded to 80x80",
        "encode: ",
        "decode: ",
    ):
        assert any(line.startswith(f"plan: {opening}") for line in lines), opening
    assert not [line for line in lines if line.startswith("plan: attention")]
    # The same plan as a run's: the run logs the lines --plan does, then loads.
    caplog.clear()
    assert run(machine, source, output) == 0
    again = [r.getMessage() for r in caplog.records if r.getMessage().startswith("plan: ")]
    assert again == lines and len(machine.loads) == 1


def test_a_plan_refused_for_the_memory_is_a_refusal(
    machine: Machine, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    machine.free = 2 * GIB
    source = make_picture(tmp_path / "in.png")
    assert run(machine, source, tmp_path / "out.png") == 1
    assert "GiB" in caplog.text and not machine.loads and not (tmp_path / "out.png").exists()
    caplog.clear()
    assert run(machine, source, tmp_path / "out.png", "--plan") == 1
    assert not machine.loads


def test_the_pins_are_the_pictures(
    machine: Machine, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    source = make_picture(tmp_path / "in.png")
    # A swap count above the DiT's blocks is a pin that is no pin (still_plan.PlanRefusal).
    assert run(machine, source, tmp_path / "o.png", "--swap-blocks", "99") == 1
    assert "--swap-blocks 99: the DiT has 36 blocks" in caplog.text
    assert run(machine, source, tmp_path / "o.png", "--swap-blocks", "36", "--plan") == 0
    assert "blocks swapped: 36 of the DiT's 36, pinned by --swap-blocks" in caplog.text


@pytest.mark.parametrize("phase", ["encode", "decode"])
def test_a_tile_covering_the_pictures_padded_frame_is_refused_in_the_videos_words(
    machine: Machine, tmp_path: Path, caplog: pytest.LogCaptureFixture, phase: str
) -> None:
    # A 1920x1080 output pads to 1920x1104 (frames.padded_size): a tile of 1920 px covers it. The
    # bound's frame (the odd sides' rounding up) is larger, so still_plan.plan alone took the pin
    # and the pass ran untiled without saying so.
    source = tmp_path / "hd.png"
    Image.new("RGB", (1920, 1080), (30, 60, 90)).save(source)
    option = f"--{phase}-tile"
    assert run(machine, source, tmp_path / "o.png", option, "1920", "--resolution", "1080") == 1
    assert (
        f"{option} 1920: a tile of 1920 px covers the whole padded frame, 1920x1104:"
        f" {option} 0 pins an untiled {phase}"
    ) in caplog.text
    assert not machine.loads and not machine.passes and not (tmp_path / "o.png").exists()
    # A smaller tile and the untiled pin are taken.
    caplog.clear()
    for pin in ("0", "1024"):
        assert (
            run(machine, source, tmp_path / "o.png", option, pin, "--resolution", "1080", "--plan")
            == 0
        )
    assert "covers the whole padded frame" not in caplog.text


def test_a_landscape_1080p_picture_pinned_untiled_is_taken_where_its_plan_is_untiled(
    machine: Machine, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # The plan is the picture's alone, as stored (still_plan.plan's `picture`): with the bound in
    # either orientation, --encode-tile 0 --decode-tile 0 was refused on 24 GiB (2026-11-04, the
    # RTX 5090), the portrait frame past the one measured untiled, where the plan unpinned took
    # the picture untiled.
    machine.free = 24 * GIB
    source = tmp_path / "hd.png"
    Image.new("RGB", (1920, 1080), (30, 60, 90)).save(source)
    sized = ("--resolution", "1080")
    assert run(machine, source, tmp_path / "o.png", *sized, "--plan") == 0
    for phase in ("encode", "decode"):
        assert f"plan: {phase}: untiled, the whole frame fitting the budget" in caplog.text
    caplog.clear()
    pins = ("--encode-tile", "0", "--decode-tile", "0")
    assert run(machine, source, tmp_path / "o.png", *sized, *pins) == 0
    for phase in ("encode", "decode"):
        assert f"plan: {phase}: untiled, pinned by --{phase}-tile" in caplog.text
    assert "bound: the picture alone, 1920x1080 as stored" in caplog.text
    ((_, keywords),) = machine.passes
    assert keywords["encode_tiles"] is None and keywords["decode_tiles"] is None
    assert read_back(tmp_path / "o.png").pixels.shape[:2] == (1080, 1920)


def test_the_host_figure_counts_the_pictures_own_arrays(
    machine: Machine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An 8K output (7680x4320) from a small picture; --format png8 and the default 16 bits.
    from seedvr2x import still_plan
    from seedvr2x.shot_plan import HOST_RESIDENT

    source = tmp_path / "small.png"
    Image.new("RGB", (160, 90), (30, 60, 90)).save(source)
    processes: list[int] = []
    real = still_plan.plan

    def spy(*arguments: Any, **keywords: Any) -> Any:
        processes.append(keywords["process"])
        return real(*arguments, **keywords)

    monkeypatch.setattr(still_plan, "plan", spy)
    out_h, out_w = 4320, 7680
    samples = out_h * out_w * 3
    for options, width in (((), 2), (("--format", "png8"), 1)):
        raw = samples * width
        expected = HOST_RESIDENT + 90 * 160 * 3 * 4 + 2 * samples * 4 + raw + 2 * raw
        assert (
            run(machine, source, tmp_path / "o.png", "--resolution", "4320", "--plan", *options)
            == 0
        )
        assert processes[-1] == expected, options


def test_the_plan_is_given_the_pictures_stored_pixels(
    machine: Machine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The picture goes to the GPU at its stored size and is resized there: the plan counts its
    # copies (still_plan.COPIES) from H x W as stored, whatever the output's size.
    from seedvr2x import still_plan

    inputs: list[int] = []
    real = still_plan.plan

    def spy(*arguments: Any, **keywords: Any) -> Any:
        inputs.append(keywords["input_pixels"])
        return real(*arguments, **keywords)

    monkeypatch.setattr(still_plan, "plan", spy)
    for width, height in ((160, 90), (90, 160), (301, 111)):
        source = tmp_path / f"{width}x{height}.png"
        Image.new("RGB", (width, height), (30, 60, 90)).save(source)
        assert run(machine, source, tmp_path / "o.png", "--resolution", "1080", "--plan") == 0
        assert inputs[-1] == width * height, (width, height)
    # Not the output's: a 24 x 40 picture upscaled to 1080 rows.
    assert run(machine, make_picture(tmp_path / "in.png"), tmp_path / "o.png", "--plan") == 0
    assert inputs[-1] == 24 * 40


# What goes wrong in the pass ------------------------------------------------------------------


def test_nan_in_the_decode_writes_nothing(
    machine: Machine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def upscale(*arguments: Any, **keywords: Any) -> npt.NDArray[np.float32]:
        raise still.NonFinite("the picture's decode holds NaN or inf: 3 of 11520 values")

    monkeypatch.setattr(still, "upscale", upscale)
    assert run(machine, make_picture(tmp_path / "in.png"), tmp_path / "out.png") == 1
    assert "holds NaN or inf: 3 of 11520 values, so no picture is written" in caplog.text
    assert not (tmp_path / "out.png").exists() and not list(tmp_path.glob("*.partial"))


def test_the_gpus_memory_refused_in_the_pass_writes_nothing(
    machine: Machine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def upscale(*arguments: Any, **keywords: Any) -> npt.NDArray[np.float32]:
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB.")

    monkeypatch.setattr(still, "upscale", upscale)
    assert run(machine, make_picture(tmp_path / "in.png"), tmp_path / "out.png") == 1
    assert (
        "the GPU's memory was refused: CUDA out of memory. Tried to allocate 2.00 GiB"
        in caplog.text
    )
    assert "nothing is written" in caplog.text and not (tmp_path / "out.png").exists()


def test_a_stop_in_the_pass_exits_as_a_jobs_does(
    machine: Machine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def upscale(*arguments: Any, **keywords: Any) -> npt.NDArray[np.float32]:
        raise KeyboardInterrupt

    monkeypatch.setattr(still, "upscale", upscale)
    assert run(machine, make_picture(tmp_path / "in.png"), tmp_path / "out.png") == 130
    assert not (tmp_path / "out.png").exists()


def test_the_default_models_are_asked_for_without_the_shot_detector(
    machine: Machine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from seedvr2x import pull

    asked: list[dict[str, Any]] = []
    real = pull.resolve

    def resolve(*arguments: Any, **keywords: Any) -> Any:
        asked.append(keywords)
        return real(*arguments, **keywords)

    monkeypatch.setattr(pull, "resolve", resolve)
    assert run(machine, make_picture(tmp_path / "in.png"), tmp_path / "out.png") == 0
    assert asked == [{"detector": False}]


# Lossless WebP ----------------------------------------------------------------------------------


def cicp() -> bytes:
    """A cICP chunk (BT.709 primaries, the sRGB transfer, identity, full range), whole."""
    body = bytes([1, 13, 0, 1])
    return struct.pack(">I", 4) + b"cICP" + body + struct.pack(">I", zlib.crc32(b"cICP" + body))


def png_with(path: Path, *, chunks: tuple[bytes, ...] = (), **options: Any) -> Path:
    """A PNG of test_still.py's picture holding colour chunks, 8 bits."""
    settings: dict[str, Any] = {"grey": False, "icc": None, **options}
    path.write_bytes(png.write(pixels_of(), bits=8, chunks=chunks, **settings))
    return path


def test_the_library_makes_the_webp_the_command_writes(
    machine: Machine,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    cpu_models: model.Models,  # noqa: F811
) -> None:
    # As the PNG's test: the file is webp.write of what still.upscale gives for the plan's tiles,
    # byte for byte, in either case of the suffix, 8 bits whatever the source's.
    models = pipeline_models(cpu_models)
    use_library(machine, monkeypatch, models)
    source = make_picture(tmp_path / "in.png", seed=3, bits=8)
    for place, name in enumerate(("out.webp", "OUT.WEBP")):
        machine.passes.clear()
        caplog.clear()
        # A directory each: Windows' names are one file whatever their case.
        output = tmp_path / str(place) / name
        output.parent.mkdir()
        assert run(machine, source, output, "--color-correction", "split") == 0
        ((arguments, keywords),) = machine.passes
        pixels, target, seed = arguments[1:4]
        assert target == TARGET and seed == SEED and keywords["correct"] is True
        again = still.upscale(models, pixels, target, seed, **keywords)
        made = recorded((80, 48), keywords, "split")
        expected = webp.write(again, grey=False, icc=None, chunks=(), xmp=made.xmp())
        assert output.read_bytes() == expected
        written = read_back(output)
        assert (
            written.format == "WEBP" and written.bits == 8 and written.pixels.shape == (48, 80, 3)
        )
        lines = [r.getMessage() for r in caplog.records if r.levelno >= logging.INFO]
        assert any(line.endswith("output 80x48 (x2), a lossless WebP, 8-bit") for line in lines), (
            lines
        )
        assert any(
            line.startswith("upscaled in ")
            and line.endswith(f"wrote {output}: 80x48, 8-bit lossless WebP, RGB")
            for line in lines
        ), lines


def test_a_webp_is_the_codes_of_png8(machine: Machine, tmp_path: Path) -> None:
    # The same codes as --format png8 on the same picture.
    source = make_picture(tmp_path / "in.png", bits=8, seed=5)
    assert run(machine, source, tmp_path / "a.webp") == 0
    assert run(machine, source, tmp_path / "a.png", "--format", "png8") == 0
    with Image.open(tmp_path / "a.webp") as webp_image, Image.open(tmp_path / "a.png") as png_image:
        assert webp_image.mode == "RGB" and np.array_equal(
            np.asarray(webp_image), np.asarray(png_image)
        )


@pytest.mark.parametrize("given", ["png", "png8", "gbrp16le"])
def test_format_with_a_webp_output_is_refused(
    machine: Machine, tmp_path: Path, caplog: pytest.LogCaptureFixture, given: str
) -> None:
    source = make_picture(tmp_path / "in.png", bits=8)
    assert run(machine, source, tmp_path / "out.webp", "--format", given) == 1
    assert (
        f"--format {given}: with a .webp output the format is the file's, a lossless WebP at 8"
        " bits; --format is for a .png output (png, 16 bits, or png8, 8 bits)"
    ) in caplog.text
    assert not machine.asked and not (tmp_path / "out.webp").exists()


def test_a_source_stored_at_16_bits_is_refused_for_a_webp_naming_png(
    machine: Machine, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    source = make_picture(tmp_path / "in.png", bits=16)
    assert run(machine, source, tmp_path / "out.webp") == 1
    assert (
        f"{source}: a lossless WebP holds 8 bits per sample and this picture is stored at 16,"
        " which it would round away: write a PNG (a .png file), which holds them"
    ) in caplog.text
    # Refused before the card is read and the models load, and nothing is written.
    assert not machine.asked and not machine.loads and not machine.passes
    assert sorted(p.name for p in tmp_path.iterdir()) == ["in.png", "models"]
    # The same picture, as a PNG, is taken.
    assert run(machine, source, tmp_path / "out.png") == 0


def test_colour_chunks_without_a_profile_are_refused_for_a_webp_and_taken_for_a_png(
    machine: Machine, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    source = png_with(tmp_path / "in.png", chunks=(cicp(),))
    assert run(machine, source, tmp_path / "out.webp") == 1
    assert (
        f"{source}: a lossless WebP holds no cICP colour chunk and this picture has no ICC profile"
        " to carry its colour instead, so its colour would change: write a PNG (a .png file),"
        " which holds them"
    ) in caplog.text
    assert not machine.asked and not machine.loads and not (tmp_path / "out.webp").exists()
    assert run(machine, source, tmp_path / "out.png", "--format", "png8") == 0
    assert read_back(tmp_path / "out.png").chunks == read_back(source).chunks != ()


def test_an_srgb_chunk_alone_or_a_profile_beside_the_chunks_is_taken_for_a_webp(
    machine: Machine, tmp_path: Path
) -> None:
    srgb = (struct.pack(">I", 1) + b"sRGB\x00" + struct.pack(">I", zlib.crc32(b"sRGB\x00")),)
    assert run(machine, png_with(tmp_path / "a.png", chunks=srgb), tmp_path / "a.webp") == 0
    both = png_with(tmp_path / "b.png", chunks=(cicp(),), icc=srgb_profile())
    assert run(machine, both, tmp_path / "b.webp") == 0
    assert read_back(tmp_path / "b.webp").icc == srgb_profile()


def test_the_profile_is_carried_into_a_webp_and_the_picture_written_upright(
    machine: Machine, tmp_path: Path
) -> None:
    profile = srgb_profile()
    source = make_picture(tmp_path / "in.png", bits=8, icc=profile, orientation=6)
    output = tmp_path / "out.webp"
    assert run(machine, source, output) == 0
    written = read_back(output)
    # Processed as stored, then turned upright: 48 wide, 80 high, no orientation written.
    assert written.icc == profile and written.orientation == 1
    assert written.pixels.shape == (80, 48, 3)
    with Image.open(output) as image:
        assert image.info["icc_profile"] == profile and 0x0112 not in image.getexif()
        assert "exif" not in image.info
    # None of it when the source had none.
    plain = make_picture(tmp_path / "plain.png", bits=8)
    assert run(machine, plain, tmp_path / "plain.webp") == 0
    with Image.open(tmp_path / "plain.webp") as image:
        assert "icc_profile" not in image.info and "exif" not in image.info


def test_a_grey_picture_is_written_to_a_webp_as_three_equal_planes(
    machine: Machine, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    source = tmp_path / "grey.png"
    Image.fromarray((pixels_of()[..., 0] * 255).astype(np.uint8), "L").save(source)
    output = tmp_path / "out.webp"
    assert run(machine, source, output) == 0
    with Image.open(output) as image:
        planes = np.asarray(image)
    assert planes.shape == (48, 80, 3)
    assert (planes[..., 0] == planes[..., 1]).all() and (planes[..., 0] == planes[..., 2]).all()
    assert any(
        r.getMessage().endswith(
            f"wrote {output}: 80x48, 8-bit lossless WebP, grey as three equal planes"
        )
        for r in caplog.records
    )


def test_plan_alone_with_a_webp_output_writes_nothing(machine: Machine, tmp_path: Path) -> None:
    source = make_picture(tmp_path / "in.png", bits=8)
    assert run(machine, source, tmp_path / "out.webp", "--plan") == 0
    assert not (tmp_path / "out.webp").exists() and not machine.loads

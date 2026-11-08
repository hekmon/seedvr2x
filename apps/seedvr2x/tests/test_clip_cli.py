"""A clip in memory through seedvr2x.clip, held to the shot's pass as the command line runs it. It
uses the library's tests' helpers (core/tests): test_still.py's stand-in models and DiT,
test_clip.py's clips, stand_ins.py's VAE; the library's half is tests/test_clip.py there.

At `quality` enhanced and maximum the clip is the shot of 2 or 4 runs as the command line makes
it (run.py: each run the shot fed from its first frame, shot.Correction's `first`, the shot's seed,
the layout of its own frames; each frame taken from the run job.run_of names), held to the same
passes run one run at a time.

A clip through clip.upscale gives, bit for bit, what the shot's pass (seedvr2x/shot_pass.py), as
the command line runs it, gives for the same frames as one shot at frame 0: encode_shot, the DiT's
windows over shots.shot_layout (sample_windows), merge_windows, decode_shot with shot.py's
Correction when the clip is corrected, tiled or not, in one window or in several. The input copy
the command line's correction re-reads through ffmpeg is read from memory here, the frames the
encode read, as test_still_cli.py and test_shot_cli.py read it."""

import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import torch
from stand_ins import cpu_models, shipped_cpu_models  # noqa: F401
from test_clip import LONG, MIDDLE, SHORT, clip_of
from test_still import SEED, TARGET, TILES, pipeline_models, stand_in_dit

import job
import shot
from seedvr2x import clip, model, shotruns
from seedvr2x.frames import output_size
from seedvr2x.passes import Tiles
from seedvr2x.peaks import GIB
from seedvr2x.shot_pass import decode_shot, encode_shot, merge_windows, sample_windows
from seedvr2x.shots import Shot, shot_layout
from shot import Correction


def shots_pipeline(
    models: model.Models,
    frames: npt.NDArray[np.float32],
    seed: int,
    window: int | None,
    encode_tiles: Tiles | None,
    decode_tiles: Tiles | None,
    correction: Correction | None,
    target: tuple[int, int] = TARGET,
) -> npt.NDArray[np.float32]:
    """The clip as the command line runs it, one shot at frame 0: its frames, in order."""
    count = frames.shape[0]
    layout = shot_layout(count, window)
    position = 0

    def read(n: int) -> npt.NDArray[np.float32]:
        nonlocal position
        position += n
        return frames[position - n : position]

    # The change of phase before each unit, as run.py makes it (Models.enter).
    models.enter("encode")
    latent = encode_shot(models, read, count, target, seed, tiles=encode_tiles)
    windows = sample_windows(models, latent, layout, seed)
    sampled: list[torch.Tensor] = []
    for _ in layout:
        models.enter("dit")
        sampled.append(next(windows).to("cpu"))
    models.enter("decode")
    written: list[npt.NDArray[np.float32]] = []
    decode_shot(
        models,
        merge_windows(sampled, layout),
        count,
        target,
        written.append,
        reference=correction,
        tiles=decode_tiles,
    )
    return np.concatenate(written)


def from_memory(monkeypatch: pytest.MonkeyPatch, frames: npt.NDArray[np.float32]) -> None:
    """shot.py's reader of the input copy replaced by the frames themselves."""

    @contextmanager
    def copy_reader(
        correction: Correction, count: int
    ) -> Generator[Callable[[int], npt.NDArray[np.float32]]]:
        position = 0

        def read(n: int) -> npt.NDArray[np.float32]:
            nonlocal position
            position += n
            return frames[position - n : position]

        yield read

    monkeypatch.setattr(shot, "_copy_reader", copy_reader)


def correction_of(frames: npt.NDArray[np.float32], first: int = 0) -> Correction:
    return Correction(
        Path("input.mkv"), Path("input.crc32"), frames.shape[2], frames.shape[1], first
    )


def runs_pipeline(
    models: model.Models,
    frames: npt.NDArray[np.float32],
    seed: int,
    window: int | None,
    encode_tiles: Tiles | None,
    decode_tiles: Tiles | None,
    corrected: bool,
    level: str,
    target: tuple[int, int] = TARGET,
) -> npt.NDArray[np.float32]:
    """The clip as the command line runs it at `level`, one shot at frame 0 of job.QUALITY[level]
    runs: each run the shot from its first frame (job.shot_runs) with the shot's seed, its own
    windows and its correction against the shot's frames from there (Correction.first), the shot
    pass of shots_pipeline; the output's frames those job.run_of takes from each run. The reader
    of the input copy (from_memory) is the caller's, over the whole clip."""
    made = runs_made(
        models, frames, seed, window, encode_tiles, decode_tiles, corrected, level, target
    )
    return composed(made, frames.shape[0], level)


def runs_made(
    models: model.Models,
    frames: npt.NDArray[np.float32],
    seed: int,
    window: int | None,
    encode_tiles: Tiles | None,
    decode_tiles: Tiles | None,
    corrected: bool,
    level: str,
    target: tuple[int, int] = TARGET,
) -> dict[int, npt.NDArray[np.float32]]:
    """Each run of runs_pipeline, by its first frame: run 0 is shots_pipeline over the whole
    clip with the shot's seed (the clip's own, at frame 0) and its correction from frame 0, the
    standard clip's very pass."""
    count = frames.shape[0]
    runs = job.QUALITY[level]
    return {
        each.offset: shots_pipeline(
            models,
            frames[each.offset :],
            each.seed(seed),
            window,
            encode_tiles,
            decode_tiles,
            correction_of(frames, each.offset) if corrected else None,
            target,
        )
        for each in job.shot_runs(Shot(0, count), runs)
    }


def composed(
    made: dict[int, npt.NDArray[np.float32]], count: int, level: str
) -> npt.NDArray[np.float32]:
    """The clip of `count` frames at `level` from its runs (runs_made): each frame from the run
    job.run_of names."""
    runs = job.QUALITY[level]
    taken = [job.run_of(runs, frame) for frame in range(count)]
    return np.stack([made[run][frame - run] for frame, run in enumerate(taken)])


# (frames, the plan's window cap, encode tiles, decode tiles, corrected): one, two and three
# windows, each tiled pass alone and both, with the correction and without.
CASES = [
    (SHORT, None, None, None, False),
    (MIDDLE, 5, None, None, False),
    (LONG, 5, None, None, False),
    (LONG, 5, TILES, None, False),
    (LONG, 5, None, TILES, False),
    (LONG, 5, TILES, TILES, False),
    (SHORT, None, TILES, TILES, True),
    (MIDDLE, 5, None, None, True),
    (LONG, 5, None, None, True),
    (LONG, 5, TILES, TILES, True),
]


@pytest.mark.parametrize(
    ("count", "window", "encode_tiles", "decode_tiles", "correct"),
    CASES,
    ids=[
        f"{count}f-{len(shot_layout(count, window))}w-{'e' if enc else ''}{'d' if dec else ''}"
        f"{'-corrected' if correct else ''}"
        for count, window, enc, dec, correct in CASES
    ],
)
def test_a_clip_is_a_shot_at_frame_0(
    monkeypatch: pytest.MonkeyPatch,
    count: int,
    window: int | None,
    encode_tiles: Tiles | None,
    decode_tiles: Tiles | None,
    correct: bool,
    cpu_models: model.Models,  # noqa: F811
) -> None:
    frames = clip_of(count, 7)
    from_memory(monkeypatch, frames)
    stand_in_dit(monkeypatch)
    models = pipeline_models(cpu_models)
    correction = correction_of(frames) if correct else None
    expected = shots_pipeline(models, frames, SEED, window, encode_tiles, decode_tiles, correction)
    ours = clip.upscale(
        models,
        frames,
        TARGET,
        SEED,
        window=window,
        encode_tiles=encode_tiles,
        decode_tiles=decode_tiles,
        correct=correct,
    )
    assert ours.shape == expected.shape == (count, 48, 80, 3) and ours.dtype == expected.dtype
    assert ours.tobytes() == expected.tobytes()
    # Not a constant, nor the other correction's.
    assert ours.std() > 0.01
    other = shots_pipeline(
        models,
        frames,
        SEED,
        window,
        encode_tiles,
        decode_tiles,
        None if correct else correction_of(frames),
    )
    assert other.tobytes() != expected.tobytes()


def test_a_clip_is_a_shot_at_frame_0_at_the_vaes_shipped_widths(
    monkeypatch: pytest.MonkeyPatch,
    shipped_cpu_models: model.Models,  # noqa: F811
) -> None:
    # The cases above run the narrow stand-in VAE (stand_ins.NARROW); one runs it at the shipped
    # widths: 33 frames in 3 windows, both passes tiled, corrected.
    test_a_clip_is_a_shot_at_frame_0(monkeypatch, LONG, 5, TILES, TILES, True, shipped_cpu_models)


# One, two and three windows, each tiled pass alone and both, with the correction and without,
# at each level above standard.
LEVEL_CASES = [
    (SHORT, None, None, None, False),
    (MIDDLE, 5, None, None, False),
    (LONG, 5, None, None, False),
    (LONG, 5, TILES, TILES, False),
    (SHORT, None, None, None, True),
    (MIDDLE, 5, None, None, True),
    (LONG, 5, None, None, True),
    (MIDDLE, 5, TILES, TILES, True),
]


@pytest.mark.parametrize("level", ["enhanced", "maximum"])
@pytest.mark.parametrize(
    ("count", "window", "encode_tiles", "decode_tiles", "correct"),
    LEVEL_CASES,
    ids=[
        f"{count}f-{len(shot_layout(count, window))}w-{'e' if enc else ''}{'d' if dec else ''}"
        f"{'-corrected' if correct else ''}"
        for count, window, enc, dec, correct in LEVEL_CASES
    ],
)
def test_a_clip_at_a_level_is_the_shots_runs_as_the_command_line_makes_them(
    monkeypatch: pytest.MonkeyPatch,
    level: str,
    count: int,
    window: int | None,
    encode_tiles: Tiles | None,
    decode_tiles: Tiles | None,
    correct: bool,
    cpu_models: model.Models,  # noqa: F811
) -> None:
    frames = clip_of(count, 7)
    from_memory(monkeypatch, frames)
    stand_in_dit(monkeypatch)
    models = pipeline_models(cpu_models)
    made = runs_made(models, frames, SEED, window, encode_tiles, decode_tiles, correct, level)
    expected = composed(made, count, level)
    ours = clip.upscale(
        models,
        frames,
        TARGET,
        SEED,
        window=window,
        encode_tiles=encode_tiles,
        decode_tiles=decode_tiles,
        correct=correct,
        quality=level,
    )
    assert ours.shape == expected.shape == (count, 48, 80, 3) and ours.dtype == expected.dtype
    assert ours.tobytes() == expected.tobytes()
    # The runs matter: not the standard run's frames, run 0's (runs_made), the shot's pass over
    # the whole clip at the clip's seed, corrected from frame 0.
    assert ours.tobytes() != made[0].tobytes()


def test_the_library_chooses_the_runs_the_command_line_does() -> None:
    # job.py imports the choice from seedvr2x.shotruns, the one clip.py runs by: the very objects.
    for name in (
        "QUALITY",
        "RUN_OFFSETS",
        "PLACES_BEST_FIRST",
        "KEPT_SAMPLE_BYTES",
        "Run",
        "place",
        "run_of",
        "shot_runs",
        "kept_frames",
        "kept_bytes",
    ):
        assert getattr(job, name) is getattr(shotruns, name), name


def test_the_windows_cut_the_output_where_the_layout_does(
    monkeypatch: pytest.MonkeyPatch,
    cpu_models: model.Models,  # noqa: F811
) -> None:
    # The cap shapes the output (DESIGN.md, Memory planner: a plan shapes the output a little):
    # the same clip in windows of 5 latents is not the clip in one window, and a cap no shorter
    # than the clip's latents is the one window.
    frames = clip_of(LONG, 3)
    stand_in_dit(monkeypatch)

    # The stand-in DiT is blind to its window, and windows over the same slices of the noise and
    # the latent would merge to the one window's output; this one adds the window's length.
    def sample(models: model.Models, noise: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return cond[..., :16] * 0.7 + noise * 0.3 + 0.01 * noise.shape[0]

    monkeypatch.setattr(model, "sample", sample)
    models = pipeline_models(cpu_models)

    def run(window: int | None) -> npt.NDArray[np.float32]:
        return clip.upscale(
            models,
            frames,
            TARGET,
            SEED,
            window=window,
            encode_tiles=None,
            decode_tiles=None,
            correct=False,
        )

    five, whole = run(5), run(None)
    assert five.tobytes() != whole.tobytes()
    assert run(9).tobytes() == whole.tobytes() == run(40).tobytes()
    height, width = output_size(TARGET)
    assert five.shape == (LONG, height, width, 3)


# The clip on the GPU --------------------------------------------------------------------------

GPU_MODELS = os.environ.get("SEEDVR2X_MODEL_DIR")
GPU_DIT, GPU_VAE = "seedvr2x_ema_7b_sharp_fp16.safetensors", "seedvr2x_ema_vae_fp16.safetensors"
# 25 frames of 640x360 to 720p: 7 latents, two windows under a cap of 5.
GPU_FRAMES, GPU_SIZE, GPU_RESOLUTION, GPU_WINDOW = 25, (360, 640), 720, 5
# The card's free memory less this is planned as if someone else had it (test_still_cli.py).
GPU_SPARE = 4 * GIB
GPU_HELD_BACK = 3 * GIB


@pytest.mark.gpu
@pytest.mark.skipif(not GPU_MODELS, reason="needs SEEDVR2X_MODEL_DIR")
@pytest.mark.parametrize("level", ["standard", "maximum"])
def test_a_clip_is_a_shot_at_frame_0_on_the_gpu(
    monkeypatch: pytest.MonkeyPatch, level: str
) -> None:
    """clip.upscale with the real models on CUDA, with the colour correction, in two windows,
    equals the shot of the command line bit for bit (float32 output), at standard's one run and at
    maximum's four (runs_pipeline), under the plan clip.plan gives in the card's free memory less
    GPU_SPARE, the models loaded once for both."""
    import gc
    from fractions import Fraction

    from test_still_cli import scene

    import main as cli
    from seedvr2x import memory, peaks
    from seedvr2x.frames import target_size
    from seedvr2x.shot_plan import Pins

    assert GPU_MODELS is not None
    directory = Path(GPU_MODELS)
    dit, vae = directory / GPU_DIT, directory / GPU_VAE
    if not (dit.is_file() and vae.is_file()):
        pytest.skip(f"needs {GPU_DIT} and {GPU_VAE} in SEEDVR2X_MODEL_DIR")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    frames = np.stack([scene(*GPU_SIZE, seed=11 + index // 5) for index in range(GPU_FRAMES)])
    target = target_size(GPU_SIZE[1], GPU_SIZE[0], Fraction(1), GPU_RESOLUTION)
    device = torch.device("cuda", 0)
    reported = cli._free_memory(device)  # pyright: ignore[reportPrivateUsage]
    card = memory.card_free(0)
    if card is None:
        pytest.skip("the card's free memory can't be read through NVML")
    free = min(reported, card)
    host = cli._host_available()  # pyright: ignore[reportPrivateUsage]
    made, predicted = clip.plan(
        free - GPU_SPARE,
        target,
        GPU_FRAMES,
        Pins(window=GPU_WINDOW),
        host,
        correct=True,
        dit_model=peaks.FP16,
    )
    print(
        f"\nclip: {GPU_FRAMES} frames of {GPU_SIZE[1]}x{GPU_SIZE[0]} to {target[1]}x{target[0]};"
        f" free {free / GIB:.2f} GiB; window {made.window}, {made.swapped} blocks swapped,"
        f" encode {made.encode_tiles}, decode {made.decode_tiles}; host {predicted.host / GIB:.2f}"
    )
    torch.cuda.set_per_process_memory_fraction(
        (free - GPU_HELD_BACK) / torch.cuda.get_device_properties(device).total_memory, device
    )
    models = None
    try:
        models = model.load_models(dit, vae, device, None, made.model_plan)
        from_memory(monkeypatch, frames)
        told: list[tuple[str, int, int]] = []
        ours = clip.upscale(
            models,
            frames,
            target,
            SEED,
            window=made.window,
            encode_tiles=made.encode_tiles,
            decode_tiles=made.decode_tiles,
            correct=True,
            step=lambda stage, done, total: told.append((stage, done, total)),
            quality=level,
        )
        expected = runs_pipeline(
            models,
            frames,
            SEED,
            made.window,
            made.encode_tiles,
            made.decode_tiles,
            True,
            level,
            target,
        )
        print(f"peak allocated {torch.cuda.max_memory_allocated(device) / GIB:.2f} GiB")
    finally:
        del models
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(1.0, device)
    # Each run in two windows (a run from frame 3 has 22 frames: 7 latents still), the last run
    # the shortest.
    runs = job.QUALITY[level]
    assert [each for each in told if each[0] == "window"] == [
        ("window", 1, 2),
        ("window", 2, 2),
    ] * runs
    assert told[-1] == ("decode", GPU_FRAMES - runs + 1, GPU_FRAMES - runs + 1)
    assert ours.shape == expected.shape and ours.dtype == expected.dtype == np.float32
    assert ours.std() > 0.05
    if not np.array_equal(ours, expected):
        difference = np.abs(ours - expected)
        place = np.unravel_index(int(difference.argmax()), difference.shape)
        raise AssertionError(
            f"{int((ours != expected).sum())} of {ours.size} values differ, at most"
            f" {float(difference.max())} at {place}: {ours[place]} against {expected[place]}"
        )

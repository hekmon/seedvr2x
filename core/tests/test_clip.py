"""One clip through the models (seedvr2x/clip.py) on the CPU: the VAE of cpu_models (the real
configs, random weights) in bfloat16, as the pipeline runs it, and test_still.py's stand-in DiT, so
that what is held is clip.py's own steps: the phases entered in order and nothing loaded, the
clip's seeds, the windows laid out and merged, the steps told in order with their counts, a step
that raises (an Exception or a BaseException) leaving no decode open and nothing of the call alive,
the refusals, and the plan's count of the host (the output array, the windows and the scratch kept
whole in system memory). That the output is the shot's pass's, bit for bit, as the command line
runs it, is the CLI's tests/test_clip_cli.py's."""

import weakref
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import torch
from stand_ins import cpu_models  # noqa: F401
from test_held import strays
from test_still import PICTURE, SEED, TARGET, TILES, pipeline_models, stand_in_dit
from torch import Tensor

from seedvr2x import clip, model, sampling, shot_plan, shotruns, tiling
from seedvr2x.frames import output_size
from seedvr2x.peaks import FP16, GIB
from seedvr2x.shot_plan import Pins, PlanError
from seedvr2x.shots import Shot, shot_layout

Events = list[Any]

# Clips of 33 frames: 9 latents, which windows of at most 5 latents cut in 3, 25 frames (7
# latents) in 2, 5 in 1 whatever the cap.
LONG, MIDDLE, SHORT = 33, 25, 5


def clip_of(count: int, seed: int = 0) -> npt.NDArray[np.float32]:
    """A clip as stored, (count, 24, 40, 3) float32 in [0, 1]."""
    return np.random.default_rng(seed).random((count, *PICTURE, 3), dtype=np.float32)


@pytest.fixture
def events() -> Events:
    return []


@pytest.fixture
def models(cpu_models: model.Models, events: Events) -> model.Models:  # noqa: F811
    """cpu_models' VAE in bfloat16, with the DiT's place and the embeddings test_held.strays
    looks at (empty and zero: the DiT is stood in), and a Models.enter that records the phase."""
    made = pipeline_models(cpu_models, events)
    made.runner.dit = torch.nn.ModuleList()  # pyright: ignore[reportAttributeAccessIssue]
    made.text_pos = made.text_neg = torch.zeros(2)  # pyright: ignore[reportAttributeAccessIssue]
    return made


def upscale(
    models: model.Models,
    frames: npt.NDArray[np.float32] | None = None,
    seed: int = SEED,
    *,
    window: int | None = None,
    encode_tiles: Any = None,
    decode_tiles: Any = None,
    correct: bool = False,
    step: clip.Step | None = None,
    quality: str = "standard",
) -> npt.NDArray[np.float32]:
    return clip.upscale(
        models,
        clip_of(SHORT) if frames is None else frames,
        TARGET,
        seed,
        window=window,
        encode_tiles=encode_tiles,
        decode_tiles=decode_tiles,
        correct=correct,
        step=step,
        quality=quality,
    )


def test_the_output_is_the_targets_frames_and_the_input_is_left_alone(
    models: model.Models, monkeypatch: pytest.MonkeyPatch
) -> None:
    stand_in_dit(monkeypatch)
    for count in (1, 2, SHORT):
        frames = clip_of(count, count)
        before = frames.copy()
        for correct in (False, True):
            out = upscale(models, frames, correct=correct)
            assert isinstance(out, np.ndarray) and out.dtype == np.float32
            assert out.shape == (count, *output_size(TARGET), 3) == (count, 48, 80, 3)
            assert 0.0 <= out.min() and out.max() <= 1.0 and np.isfinite(out).all()
            assert out.std() > 0.01
            assert frames.tobytes() == before.tobytes()
        # Each frame its own: a clip is not one frame repeated.
        if count > 1:
            assert out[0].tobytes() != out[-1].tobytes()
    # An odd target is cropped to even sides, the padding dropped.
    odd = clip.upscale(
        models,
        clip_of(2),
        (47, 79),
        SEED,
        window=None,
        encode_tiles=None,
        decode_tiles=None,
        correct=False,
    )
    assert odd.shape == (2, *output_size((47, 79)), 3) == (2, 46, 78, 3)


def test_the_clip_is_the_shot_at_frame_0_seeded_as_such(
    models: model.Models, monkeypatch: pytest.MonkeyPatch, events: Events
) -> None:
    noises = stand_in_dit(monkeypatch, events)
    seeded: list[int] = []
    drawn: list[tuple[int, tuple[int, ...]]] = []
    set_seed, draw = model.set_seed, sampling.drawn

    def spy_seed(seed: int) -> None:
        seeded.append(seed)
        set_seed(seed)

    def spy_drawn(models: model.Models, latent: Tensor, seed: int) -> tuple[Tensor, Tensor]:
        drawn.append((seed, tuple(latent.shape)))
        return draw(models, latent, seed)

    monkeypatch.setattr(model, "set_seed", spy_seed)
    monkeypatch.setattr(sampling, "drawn", spy_drawn)
    frames = clip_of(LONG)
    first = upscale(models, frames, window=5)
    assert Shot(0, LONG).seed(SEED) == SEED
    # The encode seeded with the seed plus the offset, nothing drawing; the noise drawn once, at
    # the seed, for the whole clip (sampling.drawn seeds again, as numz's draw is).
    assert seeded == [SEED + sampling.ENCODE_SEED_OFFSET, SEED]
    assert drawn == [(SEED, (9, 10, 10, 16))]
    # The windows of the layout, each the DiT's call on its slice of the noise.
    layout = shot_layout(LONG, 5)
    assert layout == [(0, 5), (3, 7), (5, 9)]
    assert [tuple(noise.shape) for noise in noises] == [
        (end - start, 10, 10, 16) for start, end in layout
    ]
    assert upscale(models, frames, window=5).tobytes() == first.tobytes()
    assert upscale(models, frames, seed=SEED + 1, window=5).tobytes() != first.tobytes()


def test_a_shot_of_a_longer_clip_is_seeded_by_its_first_frame_and_written_into_its_array(
    models: model.Models, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`start` is the shot's first frame in its source: its seed the job's plus it (Shot.seed, the
    command line's rule), so a clip cut into shots is the shots run one after the other; and
    `output`, a slice of the clip's array, takes the frames, the array returned."""
    stand_in_dit(monkeypatch)
    frames = clip_of(LONG, 3)
    shot = Shot(11, 11 + LONG)
    assert shot.seed(SEED) == SEED + 11
    plain = clip.upscale(
        models, frames, TARGET, SEED + 11, window=5, encode_tiles=None, decode_tiles=None,
        correct=True,
    )  # fmt: skip
    clip_array = np.zeros((11 + LONG + 4, *output_size(TARGET), 3), np.float32)
    returned = clip.upscale(
        models, frames, TARGET, SEED, window=5, encode_tiles=None, decode_tiles=None,
        correct=True, start=shot.start, output=clip_array[shot.start : shot.end],
    )  # fmt: skip
    assert np.shares_memory(returned, clip_array) and returned.tobytes() == plain.tobytes()
    assert clip_array[11 : 11 + LONG].tobytes() == plain.tobytes()
    assert not clip_array[:11].any() and not clip_array[11 + LONG :].any()
    # Not the plain seed: the start decides it.
    at_zero = clip.upscale(
        models, frames, TARGET, SEED, window=5, encode_tiles=None, decode_tiles=None,
        correct=True,
    )  # fmt: skip
    assert at_zero.tobytes() != plain.tobytes()


def test_an_output_array_of_another_shape_is_refused(models: model.Models) -> None:
    wrong: list[Any] = [
        np.zeros((SHORT + 1, *output_size(TARGET), 3), np.float32),
        np.zeros((SHORT, *output_size(TARGET), 3), np.float64),
    ]
    for output in wrong:
        with pytest.raises(ValueError, match="output"):
            clip.upscale(
                models, clip_of(SHORT), TARGET, SEED, window=None, encode_tiles=None,
                decode_tiles=None, correct=False, output=output,
            )  # fmt: skip


@pytest.mark.parametrize(
    ("count", "window", "windows"), [(SHORT, 5, 1), (MIDDLE, 5, 2), (LONG, 5, 3)]
)
def test_the_phases_are_entered_in_order_and_nothing_is_loaded(
    models: model.Models,
    monkeypatch: pytest.MonkeyPatch,
    events: Events,
    count: int,
    window: int,
    windows: int,
) -> None:
    stand_in_dit(monkeypatch, events)

    def loading(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("clip.upscale loaded models")

    monkeypatch.setattr(model, "load_models", loading)
    assert len(shot_layout(count, window)) == windows
    expected: Events = [("enter", "encode")]
    expected += [("enter", "dit"), "sample"] * windows
    expected += [("enter", "decode")]
    upscale(models, clip_of(count), window=window, correct=True, encode_tiles=TILES)
    assert events == expected
    # A cap none of the layouts needs: one window.
    events.clear()
    upscale(models, clip_of(count), window=None)
    assert events == [("enter", "encode"), ("enter", "dit"), "sample", ("enter", "decode")]


def test_the_steps_are_told_in_order_with_their_counts(
    models: model.Models, monkeypatch: pytest.MonkeyPatch
) -> None:
    stand_in_dit(monkeypatch)
    told: list[tuple[str, int, int]] = []

    def record(stage: str, done: int, total: int) -> None:
        told.append((stage, done, total))

    for count, window, windows in ((SHORT, 5, 1), (MIDDLE, 5, 2), (LONG, 5, 3)):
        for correct, tiles in ((False, None), (True, TILES)):
            told.clear()
            out = upscale(
                models,
                clip_of(count),
                window=window,
                correct=correct,
                encode_tiles=tiles,
                decode_tiles=tiles,
                step=record,
            )
            assert told[0] == ("encode", 1, 1)
            assert told[1 : 1 + windows] == [("window", n + 1, windows) for n in range(windows)]
            decoding = told[1 + windows :]
            assert decoding and all(stage == "decode" for stage, _, _ in decoding)
            # Each slice of the decode once written, the frames written so far counted up to all.
            done = [each for _, each, _ in decoding]
            assert done == sorted(set(done)) and done[-1] == count
            assert {total for _, _, total in decoding} == {count}
            assert out.shape[0] == count
    # Without a step, the same bits.
    told.clear()
    again = upscale(models, clip_of(MIDDLE), window=5, step=record)
    assert told and again.tobytes() == upscale(models, clip_of(MIDDLE), window=5).tobytes()


class Stop(BaseException):
    """What a node's Cancel raises: not an Exception, as KeyboardInterrupt is not."""


class Watched:
    """The decode's generators and slices, seen from outside: model.decode_stream and
    tiling.decode_tiled replaced by wrappers that remember each generator they hand out and a
    weak reference to each slice they yield."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.generators: list[Any] = []
        self.slices: list[weakref.ref[Tensor]] = []
        for owner, name in ((model, "decode_stream"), (tiling, "decode_tiled")):
            monkeypatch.setattr(owner, name, self.watching(getattr(owner, name)))

    def watching(self, make: Any) -> Any:
        def watched(*args: Any, **kwargs: Any) -> Any:
            inner = make(*args, **kwargs)

            def each() -> Any:
                # Closing what is handed out closes the decode behind it, as it does for the
                # caller that holds the generator itself (shot_pass._slices).
                try:
                    for chunk in inner:
                        self.slices.append(weakref.ref(chunk))
                        yield chunk
                        del chunk
                finally:
                    inner.close()

            generator = each()
            self.generators.append((generator, inner))
            return generator

        return watched

    def closed(self) -> bool:
        """Every generator handed out is finished or closed, and no slice of it is alive."""
        return all(
            outer.gi_frame is None and inner.gi_frame is None for outer, inner in self.generators
        ) and all(each() is None for each in self.slices)


@pytest.mark.parametrize(
    "error", [RuntimeError("cancelled"), Stop()], ids=["Exception", "BaseException"]
)
@pytest.mark.parametrize("stage", ["encode", "window", "decode"])
@pytest.mark.parametrize("correct", [False, True])
def test_a_step_that_raises_leaves_no_decode_open_and_nothing_alive(
    models: model.Models,
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
    stage: str,
    correct: bool,
) -> None:
    stand_in_dit(monkeypatch)
    watched = Watched(monkeypatch)
    calls = 0

    def step(each: str, done: int, total: int) -> None:
        nonlocal calls
        # The second window's, the second slice of the decode: not the first, so that a stop comes
        # with the work half done.
        if each == stage and (stage == "encode" or done >= 2 or total == 1):
            calls += 1
            raise error

    with pytest.raises(type(error)) as caught:
        upscale(
            models,
            clip_of(LONG),
            window=5,
            correct=correct,
            decode_tiles=TILES if stage == "decode" else None,
            step=step,
        )
    # The exception is still held, with its traceback: what its frames kept is cleared.
    assert caught.value is error and calls == 1
    assert watched.closed(), "a decode was left open, or a slice of it kept"
    # Nothing but what the models hold: no stray on a device, no RoPE cache, the VAE's causal
    # memory cleared (model._forget).
    assert strays(models) == []
    memories = [
        module.memory
        for module in models.runner.vae.modules()
        if getattr(module, "memory", None) is not None
    ]
    assert memories == []
    # The models serve the next clip as if nothing had happened.
    clean = upscale(models, clip_of(LONG), window=5, correct=correct)
    assert clean.shape == (LONG, 48, 80, 3)


def test_a_window_stopped_by_its_step_is_not_kept(
    models: model.Models, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A window's output leaves the device before its step is told: nothing of it is bound in the
    # call's frames by then (the stop's traceback is cleared too), only its copy in the windows
    # kept, which the call gives up with it.
    stand_in_dit(monkeypatch)
    window = sampling.window
    outputs: list[weakref.ref[Tensor]] = []

    def seen(models: model.Models, noise: Tensor, latent: Tensor) -> Tensor:
        out = window(models, noise, latent)
        # On the CPU the "device" is the system's, so the copy to it is the output itself: the
        # reference tells whether the clip kept anything once it was stopped.
        outputs.append(weakref.ref(out))
        return out

    monkeypatch.setattr(sampling, "window", seen)

    def step(stage: str, done: int, total: int) -> None:
        if stage == "window":
            raise Stop()

    with pytest.raises(Stop):
        upscale(models, clip_of(LONG), window=5, step=step)
    assert len(outputs) == 1 and outputs[0]() is None


@pytest.mark.parametrize(
    "bad",
    [
        clip_of(2).astype(np.float64),
        clip_of(2)[..., :1],
        np.concatenate([clip_of(2), clip_of(2)[..., :1]], axis=-1),
        clip_of(2)[0],
        clip_of(2)[None],
    ],
    ids=["float64", "one channel", "four channels", "3 dimensions", "5 dimensions"],
)
def test_frames_that_are_not_a_float32_clip_are_refused(
    models: model.Models, bad: npt.NDArray[Any]
) -> None:
    with pytest.raises(ValueError, match=r"not \(T, H, W, 3\) float32"):
        upscale(models, bad)


def test_a_clip_of_no_frame_is_refused(models: model.Models) -> None:
    with pytest.raises(ValueError, match="no frame"):
        upscale(models, clip_of(0))


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_a_window_holding_nan_or_inf_is_refused(
    models: model.Models, monkeypatch: pytest.MonkeyPatch, value: float
) -> None:
    stand_in_dit(monkeypatch)
    sample = model.sample
    calls = 0

    def poisoned(models: model.Models, noise: Tensor, cond: Tensor) -> Tensor:
        nonlocal calls
        calls += 1
        out = sample(models, noise, cond)
        if calls == 2:
            out[1, 2, 3, 4] = value
        return out

    monkeypatch.setattr(model, "sample", poisoned)
    with pytest.raises(
        clip.NonFinite, match=r"the clip's window 2/3 \(frames 0 to 32\): 1 of its"
    ) as caught:
        upscale(models, clip_of(LONG), window=5)
    assert isinstance(caught.value, RuntimeError)
    assert calls == 2


def test_a_window_of_a_later_shot_holding_nan_names_the_shot_by_its_frames(
    models: model.Models, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A clip cut into shots is one call a shot: the refusal says which, as the decode's does.
    stand_in_dit(monkeypatch)
    sample = model.sample

    def poisoned(models: model.Models, noise: Tensor, cond: Tensor) -> Tensor:
        out = sample(models, noise, cond)
        out[1, 2, 3, 4] = float("nan")
        return out

    monkeypatch.setattr(model, "sample", poisoned)
    with pytest.raises(clip.NonFinite, match=r"window 1/\d+ \(frames 9 to 29\)"):
        clip.upscale(
            models,
            clip_of(21),
            TARGET,
            SEED,
            window=5,
            encode_tiles=None,
            decode_tiles=None,
            correct=False,
            start=9,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
@pytest.mark.parametrize("correct", [False, True])
def test_a_decode_holding_nan_or_inf_is_refused_and_closed(
    models: model.Models, monkeypatch: pytest.MonkeyPatch, value: float, correct: bool
) -> None:
    stand_in_dit(monkeypatch)
    decode_stream = model.decode_stream

    def poisoned(models: model.Models, latent: Tensor) -> Any:
        for chunk in decode_stream(models, latent):
            chunk[0, 0, 3, 5] = value
            yield chunk

    monkeypatch.setattr(model, "decode_stream", poisoned)
    with pytest.raises(clip.NonFinite, match=r"the clip's decode, frames 0 to") as caught:
        upscale(models, clip_of(SHORT), correct=correct)
    assert isinstance(caught.value, RuntimeError)


def test_an_encode_holding_nan_is_refused(
    models: model.Models, monkeypatch: pytest.MonkeyPatch
) -> None:
    stand_in_dit(monkeypatch)
    frames = clip_of(SHORT)
    frames[1, 2, 3, 0] = np.nan
    with pytest.raises(clip.NonFinite, match="the clip's encode"):
        upscale(models, frames)


def test_the_correction_is_split_against_the_clips_own_frames(
    models: model.Models, monkeypatch: pytest.MonkeyPatch
) -> None:
    from seedvr2x import colour

    stand_in_dit(monkeypatch)
    frames = clip_of(SHORT, 3)
    seen: list[tuple[Tensor, Tensor, int]] = []
    split = colour.split

    def spy(content: Tensor, reference: Tensor, stages: int) -> Tensor:
        seen.append((content, reference, stages))
        return split(content, reference, stages)

    monkeypatch.setattr(colour, "split", spy)
    plain = upscale(models, frames)
    assert seen == []
    corrected = upscale(models, frames, correct=True)
    height, width = output_size(TARGET)
    # Every frame of the decode against its own: the clip's frames through the encoder's
    # transform in float32, cropped as the decode is, in the order of the decode's slices.
    assert sum(content.shape[0] for content, _, _ in seen) == SHORT
    reference = torch.cat([each for _, each, _ in seen])
    sources = torch.as_tensor(frames).permute(0, 3, 1, 2).to(torch.float32)
    expected = model.input_transform(TARGET)(sources).permute(1, 0, 2, 3)[:, :, :height, :width]
    assert reference.dtype == torch.float32 and torch.equal(reference, expected)
    assert {stages for _, _, stages in seen} == {colour.colour_stages(PICTURE, TARGET)}
    assert corrected.shape == plain.shape and corrected.tobytes() != plain.tobytes()


def test_tiled_passes_keep_their_scratch_in_memory(
    models: model.Models, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    import tempfile

    stand_in_dit(monkeypatch)
    stores: list[tiling.InMemory] = []

    class Counted(tiling.InMemory):
        def __init__(self) -> None:
            super().__init__()
            stores.append(self)

    monkeypatch.setattr(tiling, "InMemory", Counted)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    frames = clip_of(MIDDLE)
    plain = upscale(models, frames, window=5)
    both = upscale(models, frames, window=5, encode_tiles=TILES, decode_tiles=TILES)
    # One store made a pass, the encode's and the decode's, and no file anywhere.
    assert len(stores) == 4
    assert list(tmp_path.iterdir()) == []
    assert both.shape == plain.shape and both.tobytes() != plain.tobytes()


# The quality levels: a shot run 2 or 4 times --------------------------------------------------


def run_in_pieces(
    models: model.Models, frames: npt.NDArray[np.float32], level: str, **options: Any
) -> npt.NDArray[np.float32]:
    """The composite of a level made from standard clips: frame i of the output is frame
    i - offset of a standard clip of the clip's frames from `offset`, offset being the first frame
    of the run shotruns.run_of takes frame i from (a run is a clip of its own, seeded as the
    clip, its correction against its own frames: clip.upscale's runs made by hand)."""
    count = frames.shape[0]
    levels = shotruns.QUALITY[level]
    made = {
        offset: upscale(models, frames[offset:], **options)
        for offset in {shotruns.run_of(levels, i) for i in range(count)}
    }
    out = np.empty((count, *made[0].shape[1:]), np.float32)
    for i in range(count):
        offset = shotruns.run_of(levels, i)
        out[i] = made[offset][i - offset]
    return out


@pytest.mark.parametrize("level", ["enhanced", "maximum"])
@pytest.mark.parametrize(("count", "window"), [(SHORT, None), (MIDDLE, 5), (LONG, 5)])
@pytest.mark.parametrize("correct", [False, True])
def test_a_level_takes_each_frame_from_its_run_and_each_run_is_a_clip_from_its_first_frame(
    models: model.Models,
    monkeypatch: pytest.MonkeyPatch,
    level: str,
    count: int,
    window: int,
    correct: bool,
) -> None:
    stand_in_dit(monkeypatch)
    frames = clip_of(count, 5)
    before = frames.copy()
    out = upscale(models, frames, window=window, correct=correct, quality=level)
    expected = run_in_pieces(models, frames, level, window=window, correct=correct)
    assert out.shape == (count, 48, 80, 3) and out.tobytes() == expected.tobytes()
    assert frames.tobytes() == before.tobytes()
    # Not the standard run's frames: the runs differ, as the levels do.
    assert out.tobytes() != upscale(models, frames, window=window, correct=correct).tobytes()
    # Standard is the default, and the same as asked.
    plain = upscale(models, frames, window=window, correct=correct)
    assert (
        plain.tobytes()
        == upscale(models, frames, window=window, correct=correct, quality="standard").tobytes()
    )


def test_a_clip_too_short_for_a_runs_first_frame_has_no_such_run(
    models: model.Models, monkeypatch: pytest.MonkeyPatch
) -> None:
    stand_in_dit(monkeypatch)
    told: list[tuple[str, int, int]] = []

    def record(stage: str, done: int, total: int) -> None:
        told.append((stage, done, total))

    one = clip_of(1)
    assert upscale(models, one, quality="maximum").tobytes() == upscale(models, one).tobytes()
    two = clip_of(2)
    out = upscale(models, two, quality="maximum", step=record)
    # Runs from frames 0 and 1: frame 1 is its run's own first frame.
    assert [each for each in told if each[0] == "run"] == [("run", 1, 2), ("run", 2, 2)]
    assert out.tobytes() == run_in_pieces(models, two, "maximum").tobytes()
    # Enhanced's second run starts at frame 2: a clip of two has one run, and told none.
    told.clear()
    upscale(models, two, quality="enhanced", step=record)
    assert "run" not in {stage for stage, _, _ in told}


def test_the_steps_of_the_runs_are_told_run_by_run_and_standards_are_as_they_were(
    models: model.Models, monkeypatch: pytest.MonkeyPatch, events: Events
) -> None:
    stand_in_dit(monkeypatch, events)
    told: list[tuple[str, int, int]] = []

    def record(stage: str, done: int, total: int) -> None:
        told.append((stage, done, total))

    upscale(models, clip_of(MIDDLE), window=5, step=record)
    assert "run" not in {stage for stage, _, _ in told} and told[0] == ("encode", 1, 1)
    for level, offsets in (("enhanced", (0, 2)), ("maximum", (0, 1, 2, 3))):
        told.clear()
        events.clear()
        upscale(models, clip_of(MIDDLE), window=5, quality=level, step=record)
        runs = len(offsets)
        # Each run: told its place, then its encode, its windows (25 frames: 7 latents, 2 windows
        # of at most 5; the shorter runs of 23 to 25 frames: 6 or 7 latents) and its decode, the
        # frames counted from the run's own first.
        starts = [i for i, each in enumerate(told) if each[0] == "run"]
        assert [told[i] for i in starts] == [("run", n + 1, runs) for n in range(runs)]
        for n, offset in enumerate(offsets):
            part = told[starts[n] + 1 : starts[n + 1] if n + 1 < runs else None]
            assert part[0] == ("encode", 1, 1)
            # The run's windows, counted from its own first, and its decode's frames from its own.
            windows = [(done, total) for stage, done, total in part if stage == "window"]
            assert windows == [(i + 1, len(windows)) for i in range(len(windows))]
            assert len(windows) == len(shot_layout(MIDDLE - offset, 5))
            decoding = [each for each in part if each[0] == "decode"]
            assert decoding and {total for _, _, total in decoding} == {MIDDLE - offset}
            assert decoding[-1][1] == MIDDLE - offset
        # The phases entered in order, run after run: 25 frames and shorter, 2 windows of at most 5
        # latents each time (7 latents, 6 or 7 for the shorter runs).
        phases = [each for each in events if each != "sample"]
        assert (
            phases
            == [("enter", "encode"), ("enter", "dit"), ("enter", "dit"), ("enter", "decode")] * runs
        )


def test_the_output_is_allocated_once_and_the_runs_write_into_it(
    models: model.Models, monkeypatch: pytest.MonkeyPatch
) -> None:
    stand_in_dit(monkeypatch)
    frames = clip_of(MIDDLE)
    made: list[tuple[int, ...]] = []
    empty = np.empty

    def counting(shape: Any, *args: Any, **kwargs: Any) -> Any:
        made.append(tuple(shape) if isinstance(shape, tuple) else (shape,))
        return empty(shape, *args, **kwargs)

    monkeypatch.setattr(np, "empty", counting)
    out = upscale(models, frames, window=5, quality="maximum")
    # One array of the output's shape, not one a run: the frames of the runs go into it.
    assert made.count((MIDDLE, *output_size(TARGET), 3)) == 1
    # A given array is the one written and returned, outside the shot's frames untouched.
    big = np.full((MIDDLE + 6, *output_size(TARGET), 3), -1.0, np.float32)
    returned = clip.upscale(
        models, frames, TARGET, SEED, window=5, encode_tiles=None, decode_tiles=None,
        correct=False, output=big[3 : 3 + MIDDLE], quality="maximum",
    )  # fmt: skip
    assert np.shares_memory(returned, big) and returned.tobytes() == out.tobytes()
    assert (big[:3] == -1).all() and (big[3 + MIDDLE :] == -1).all()


def test_the_runs_share_the_clips_seed_and_the_correction_reads_the_same_frames_from_each_first(
    models: model.Models, monkeypatch: pytest.MonkeyPatch
) -> None:
    from seedvr2x import colour

    stand_in_dit(monkeypatch)
    seeded: list[int] = []
    set_seed = model.set_seed

    def spy_seed(seed: int) -> None:
        seeded.append(seed)
        set_seed(seed)

    monkeypatch.setattr(model, "set_seed", spy_seed)
    run = 0
    seen: list[list[Tensor]] = [[], []]
    split = colour.split

    def spy(content: Tensor, reference: Tensor, stages: int) -> Tensor:
        seen[run - 1].append(reference)
        return split(content, reference, stages)

    def step(stage: str, done: int, total: int) -> None:
        nonlocal run
        if stage == "run":
            run = done

    monkeypatch.setattr(colour, "split", spy)
    frames = clip_of(SHORT, 3)
    clip.upscale(
        models, frames, TARGET, SEED, window=None, encode_tiles=None, decode_tiles=None,
        correct=True, start=11, quality="enhanced", step=step,
    )  # fmt: skip
    # Each run: the encode seeded with the offset seed, the noise with the clip's (the shot's
    # seed, 11 more than the job's), the same for the two runs.
    assert seeded == [SEED + 11 + sampling.ENCODE_SEED_OFFSET, SEED + 11] * 2
    height, width = output_size(TARGET)
    transform = model.input_transform(TARGET)
    for offset, got in zip((0, 2), seen, strict=True):
        sources = torch.as_tensor(frames[offset:]).permute(0, 3, 1, 2).to(torch.float32)
        expected = transform(sources).permute(1, 0, 2, 3)[:, :, :height, :width]
        assert torch.equal(torch.cat(got), expected)


def test_a_cancel_between_runs_stops_the_clip_with_nothing_alive(
    models: model.Models, monkeypatch: pytest.MonkeyPatch, events: Events
) -> None:
    stand_in_dit(monkeypatch, events)
    watched = Watched(monkeypatch)

    def step(stage: str, done: int, total: int) -> None:
        if stage == "run" and done == 2:
            raise Stop()

    with pytest.raises(Stop):
        upscale(models, clip_of(LONG), window=5, quality="maximum", step=step, correct=True)
    # The first run was made whole, the second begun: told, not encoded.
    assert events.count(("enter", "encode")) == 1 and events.count(("enter", "decode")) == 1
    assert watched.closed()
    assert strays(models) == []
    clean = upscale(models, clip_of(LONG), window=5, quality="maximum")
    assert clean.shape == (LONG, 48, 80, 3)


def test_a_level_that_is_not_one_is_refused(models: model.Models) -> None:
    with pytest.raises(ValueError, match="quality 'ultra', not one of standard, enhanced, maximum"):
        upscale(models, clip_of(2), quality="ultra")


# The plan -------------------------------------------------------------------------------------

FREE = 30 * GIB
UHD = (2160, 3840)
TILED = Pins(encode_tile=1024, decode_tile=1024)


def our_plan(frames: int, target: tuple[int, int], pins: Pins, host: int | None, **options: Any):
    return clip.plan(FREE, target, frames, pins, host, correct=True, **options)


def test_the_output_array_is_counted_and_taken_off_the_host() -> None:
    assert clip.output_bytes(100, (1080, 1920)) == 100 * 1080 * 1920 * 3 * 4
    assert clip.output_bytes(7, (1081, 1921)) == 7 * 1080 * 1920 * 12
    host = 20 * GIB
    taken = clip.output_bytes(400, (1080, 1920))
    made = our_plan(400, (1080, 1920), Pins(window=9), host, held=True)
    # The plan of the one shot with the host less the array, the scratch and the windows whole.
    expected = shot_plan.plan(
        FREE, (1080, 1920), [Shot(0, 400)], Pins(window=9), host - taken,
        split=True, held=True, in_memory=True,
    )  # fmt: skip
    assert made == expected
    # Unchecked stays unchecked.
    assert our_plan(400, (1080, 1920), Pins(window=9), None, held=True) == shot_plan.plan(
        FREE, (1080, 1920), [Shot(0, 400)], Pins(window=9), None,
        split=True, held=True, in_memory=True,
    )  # fmt: skip
    # The weights not held: their pinned copies are in the count, as the planner has them.
    unheld = our_plan(400, (1080, 1920), Pins(window=9), host)
    assert unheld[1].host - made[1].host == shot_plan._pinned(unheld[0], FP16)  # pyright: ignore[reportPrivateUsage]
    # No colour correction, no split in the plan's peaks.
    plain, _ = clip.plan(FREE, (1080, 1920), 400, Pins(window=9), host, correct=False, held=True)
    assert plain.window == made[0].window == 9


def test_a_clip_of_shots_is_planned_as_its_shots_with_the_whole_output_counted() -> None:
    """The shots are the planner's (the longest decides), a number of frames one shot, and the
    output array of all of them is what is taken off the host."""
    cuts = [Shot(0, 100), Shot(100, 400), Shot(400, 450)]
    host = 20 * GIB
    made = clip.plan(FREE, (1080, 1920), cuts, Pins(window=9), host, correct=True, held=True)
    taken = clip.output_bytes(450, (1080, 1920))
    assert made == shot_plan.plan(
        FREE, (1080, 1920), cuts, Pins(window=9), host - taken,
        split=True, held=True, in_memory=True,
    )  # fmt: skip
    one = clip.plan(FREE, (1080, 1920), 450, Pins(window=9), host, correct=True, held=True)
    assert one == clip.plan(
        FREE, (1080, 1920), [Shot(0, 450)], Pins(window=9), host, correct=True, held=True
    )
    for none in ([], [Shot(0, 0)]):
        with pytest.raises(ValueError, match="no frame"):
            clip.plan(FREE, UHD, none, Pins(), None, correct=True)


def test_a_host_short_of_the_output_is_refused_and_says_so() -> None:
    taken = clip.output_bytes(150, UHD)
    with pytest.raises(PlanError, match=r"^the clip's output, 150 frames of 3840x2160") as caught:
        our_plan(150, UHD, Pins(), taken - 1)
    assert caught.value.phase == "host"
    assert f"takes {taken / GIB:.2f} GiB of system memory" in str(caught.value)
    # Just over it: the process's own memory is what the host is short of, in the planner's words
    # with the array said.
    with pytest.raises(PlanError, match="for the process") as caught:
        our_plan(150, UHD, Pins(), taken + GIB, held=True)
    assert caught.value.phase == "host"
    assert (
        f"the output array's {taken / GIB:.2f} GiB are taken off the {(taken + GIB) / GIB:.2f}"
        in (str(caught.value))
    )
    with pytest.raises(ValueError, match="no frame"):
        our_plan(0, UHD, Pins(), None)


def test_a_tiled_pass_keeps_its_whole_scratch_in_system_memory() -> None:
    # 150 frames at 4K, both passes tiled: a scratch of 7.20 GiB. The command line's plan gives a
    # host that is short of it an allowance below it, the rest going to a file; the clip has no
    # file: the allowance is the scratch whole, and the host is judged with it.
    shots = [Shot(0, 150)]
    scratch = shot_plan.peaks.scratch(shot_plan.padded_size(UHD), 150)
    assert scratch / GIB == pytest.approx(7.20, abs=0.005)
    for host in (12 * GIB, 30 * GIB, None):
        files, _ = shot_plan.plan(FREE, UHD, shots, TILED, host, split=True, held=True)
        whole, peaks_ = shot_plan.plan(
            FREE, UHD, shots, TILED, host, split=True, held=True, in_memory=True
        )
        assert whole.scratch_allowance == scratch == peaks_.scratch
        assert files.scratch_allowance <= scratch
        assert replace_allowance(files, scratch) == whole
    short, _ = shot_plan.plan(FREE, UHD, shots, TILED, 12 * GIB, split=True, held=True)
    assert short.scratch_allowance < scratch
    # A host that holds the file's plan and not the whole scratch: refused, the scratch in the
    # words, no shorter window helping it.
    host = round(9.8 * GIB)
    shot_plan.plan(FREE, UHD, shots, TILED, host, split=True, held=True)
    with pytest.raises(PlanError, match=r"7\.20 for a tiled shot's scratch") as caught:
        shot_plan.plan(FREE, UHD, shots, TILED, host, split=True, held=True, in_memory=True)
    assert caught.value.phase == "host"
    # Untiled passes have none.
    flat, _ = shot_plan.plan(FREE, (720, 1280), shots, Pins(), None, split=True, in_memory=True)
    assert flat.scratch_allowance == 0


def replace_allowance(plan: Any, allowance: int) -> Any:
    from dataclasses import replace

    return replace(plan, scratch_allowance=allowance)


def test_the_windows_are_all_in_memory_at_the_dits_last_window() -> None:
    padded = shot_plan.padded_size((1080, 1920))
    latent = (
        shot_plan.LATENT_CHANNELS * shot_plan.LATENT_VALUE * (padded[0] // 8) * (padded[1] // 8)
    )
    for latents, cap in ((9, 5), (101, 9), (39, 6), (1000, 20)):
        layout = shot_plan.window_layout(latents, cap)
        kept = sum(end - start for start, end in layout)
        assert kept == latents + shot_plan.SHARED * (len(layout) - 1)
        files = shot_plan.host_moments(padded, latents, cap)
        memory = shot_plan.host_moments(padded, latents, cap, in_memory=True)
        # The latent and the noise, and the windows all together instead of the longest alone.
        assert files["windows"] == 2 * latent * latents + latent * max(
            end - start for start, end in layout
        )
        assert memory["windows"] == 2 * latent * latents + latent * kept
        # The latent is alive through the merge (clip.upscale holds it until its windows are
        # merged): L latents more than a file's merge, which has read the windows back and
        # given the latent up (review L2).
        assert memory["merge"] == files["merge"] + latent * latents
        assert {k: v for k, v in memory.items() if k not in ("windows", "merge")} == {
            k: v for k, v in files.items() if k not in ("windows", "merge")
        }
        # The merge's count is the windows kept, the merged latent, a join's mix and the latent.
        assert memory["merge"] == latent * (
            kept + latents + shot_plan.MIXING * shot_plan.SHARED + latents
        )
    # A clip in one window holds the same either way but at the merge, where the latent is there
    # beside the window (which is the merged latent).
    for cap in (None, 20):
        one = shot_plan.host_moments(padded, 12, cap, True)
        plain = shot_plan.host_moments(padded, 12, cap)
        assert {k: v for k, v in one.items() if k != "merge"} == {
            k: v for k, v in plain.items() if k != "merge"
        }
        assert one["merge"] == plain["merge"] + latent * 12 == 2 * latent * 12
    # What the plan counts of the shot, at the DiT's phase, for 400 frames in windows of 9.
    _, in_memory = our_plan(400, (1080, 1920), Pins(window=9), None, held=True)
    _, files = shot_plan.plan(
        FREE, (1080, 1920), [Shot(0, 400)], Pins(window=9), None, split=True, held=True
    )
    assert in_memory.moment == "merge" and files.moment == "merge"
    assert in_memory.shot > files.shot


def test_in_memory_false_plans_every_job_as_before() -> None:
    jobs: list[tuple[tuple[int, int], list[Shot], Pins, int]] = [
        (UHD, [Shot(0, 150)], TILED, 12 * GIB),
        ((1080, 1920), [Shot(0, 400)], Pins(window=9), 16 * GIB),
        ((720, 1280), [Shot(0, 12), Shot(12, 145)], Pins(), 9 * GIB),
    ]
    for target, film, pins, host in jobs:
        for held in (False, True):
            for available in (host, None):
                job = (FREE, target, film, pins, available)
                try:
                    before = shot_plan.plan(*job, split=True, held=held)
                except PlanError as error:
                    with pytest.raises(PlanError) as again:
                        shot_plan.plan(*job, split=True, held=held, in_memory=False)
                    assert str(again.value) == str(error)
                    continue
                assert shot_plan.plan(*job, split=True, held=held, in_memory=False) == before

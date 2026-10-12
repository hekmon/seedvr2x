"""The shot pipeline's parts on the CPU: windows, mixing weights, padding, the encoder's input
and split's reference, streamed encode and decode; the DiT's windows over a stand-in, their one
draw of noise and what waits where between them; a shot's tiled encode and decode
(runtime/tiling.py) through encode_shot and decode_shot, over test_tiling.py's stand-in VAE; and
what a shot's steps on the CPU allocate in system memory, which the planner counts
(planner.host_moments)."""

import logging
import weakref
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
import torch
from omegaconf import OmegaConf
from test_tiling import MEMORY, Failing, StandIn, failing_at, latent_of, mapped, models_of, same
from test_units import channel_major, job

from seedvr2x.media.ffmpeg import MediaError
from seedvr2x.runtime import model, planner, shot, tiling, units
from seedvr2x.runtime.job import OutputSegment, Shot, output_size, padded_size
from seedvr2x.runtime.plan import Tiles
from seedvr2x.runtime.shot import (
    Correction,
    decode_shot,
    encode_shot,
    encoder_inputs,
    finite,
    input_slices,
    merge_windows,
    mix_weights,
    pad_4n1,
    padded_length,
    reference_inputs,
    sample_windows,
    shot_layout,
    window_layout,
)
from seedvr2x.runtime.tiling import Scratch, ScratchError
from seedvr2x.runtime.units import DiskUnits, Units

# The models as far as the encoder's input and split's reference use them: the CPU and the VAE's
# slicing (model.encode_slices).
SLICING = cast(
    model.Models,
    SimpleNamespace(
        device=torch.device("cpu"),
        runner=SimpleNamespace(vae=SimpleNamespace(use_slicing=True, slicing_sample_min_size=4)),
    ),
)


def test_one_window_when_the_shot_fits() -> None:
    assert window_layout(21, 21) == [(0, 21)]
    assert window_layout(5, 6) == [(0, 5)]


def test_study_layout() -> None:
    # Clip B of the stitching study: 81 frames, 21 latents, windows of 6 sharing 2.
    assert window_layout(21, 6) == [(0, 6), (4, 10), (8, 14), (12, 18), (16, 21)]


@pytest.mark.parametrize("window", [5, 6, 7, 11, 30])
def test_balanced_layouts(window: int) -> None:
    for latents in range(window + 1, 200):
        layout = window_layout(latents, window)
        lengths = [e - s for s, e in layout]
        assert layout[0][0] == 0 and layout[-1][1] == latents
        assert max(lengths) <= window and max(lengths) - min(lengths) <= 1
        assert lengths == sorted(lengths, reverse=True)
        assert all(e0 - s == 2 for (_, e0), (s, _) in pairwise(layout))
        # no latent in three windows
        assert all(s2 >= e0 for (_, e0), (s2, _) in zip(layout, layout[2:], strict=False))
        # as few windows as the cap allows: one fewer would need a longer one
        fewer = len(layout) - 1
        assert fewer * window - (fewer - 1) * 2 < latents


def test_window_too_short() -> None:
    with pytest.raises(ValueError):
        window_layout(21, 4)


def test_cosine_weights_exact_in_bfloat16() -> None:
    assert mix_weights(2) == pytest.approx([0.75, 0.25])
    as_bf16 = torch.tensor(mix_weights(2)).to(torch.bfloat16).float().tolist()
    assert as_bf16 == [0.75, 0.25]


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (1, [0]),
        (2, [0, 1, 1, 1, 1]),
        (3, [0, 1, 2, 1, 0]),
        (5, [0, 1, 2, 3, 4]),
        (10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6]),
    ],
)
def test_padding_as_numz(count: int, expected: list[int]) -> None:
    frames = torch.arange(count).view(count, 1, 1, 1).expand(count, 2, 2, 3)
    assert pad_4n1(frames)[:, 0, 0, 0].tolist() == expected


class Reads:
    """A shot's frames, read as encode_shot reads them: n at a time, each read recorded."""

    def __init__(self, frames: npt.NDArray[np.float32]) -> None:
        self.frames, self.read, self.counts = frames, 0, list[int]()

    def __call__(self, count: int) -> npt.NDArray[np.float32]:
        self.counts.append(count)
        self.read += count
        return self.frames[self.read - count : self.read]


def vae_slices(frames: int) -> list[int]:
    return model.encode_slices(SLICING, frames)


@pytest.mark.parametrize("count", range(1, 22))
def test_streamed_input_is_the_padded_shot(count: int) -> None:
    # Read in the VAE's slices (5 frames, then 4), padded at the end as pad_4n1 pads the whole.
    frames = np.random.default_rng(count).random((count, 2, 3, 3), dtype=np.float32)
    reads = Reads(frames)
    sizes = vae_slices(padded_length(count))
    slices = list(input_slices(reads, count, sizes))
    assert [s.shape[0] for s in slices] == sizes
    assert sum(reads.counts) == count and all(n <= 5 for n in reads.counts)
    assert torch.equal(torch.cat(slices), pad_4n1(model.to_input(frames)))


def test_vae_slices() -> None:
    assert vae_slices(1) == [1]
    assert vae_slices(5) == [5]
    assert vae_slices(9) == [5, 4]
    assert vae_slices(45) == [5, *[4] * 10]


# Frames resized 2x, as milestone 1's are, to a height the transform pads (24 rows to 48: 8
# reflected, 16 black).
SOURCE, TARGET = (12, 16), (24, 32)


@pytest.mark.parametrize("count", [1, 6, 13])
def test_reference_is_the_transform_in_float32(count: int) -> None:
    # split's reference (DESIGN.md, Colour correction, Numerics): the frames read, through the
    # encoder's transform in float32, with no float16 or bfloat16 step, which random values would
    # show; slice by slice as the encoder's input.
    frames = np.random.default_rng(count).random((count, *SOURCE, 3), dtype=np.float32)
    # From a copy, before the reference is built: on the CPU, the reference's slices hold the
    # frames read without a copy, so a step altering them in place would alter both sides alike.
    video = pad_4n1(torch.from_numpy(frames.copy())).permute(0, 3, 1, 2)
    expected = model.input_transform(TARGET)(video)
    slices = list(reference_inputs(SLICING, Reads(frames), count, TARGET))
    assert [s.shape[1] for s in slices] == vae_slices(padded_length(count))
    assert all(s.dtype == torch.float32 for s in slices)
    assert torch.equal(torch.cat(slices, dim=1), expected)


@pytest.mark.parametrize("count", [1, 6, 13])
def test_encoder_input_is_numz(count: int) -> None:
    # The encoder's input stays numz's (milestone 1): float16 frames, cast to bfloat16 before the
    # transform (generation_phases.py:380-413).
    frames = np.random.default_rng(count).random((count, *SOURCE, 3), dtype=np.float32)
    slices = list(encoder_inputs(SLICING, Reads(frames), count, TARGET))
    assert [s.shape[1] for s in slices] == vae_slices(padded_length(count))
    assert all(s.dtype == model.COMPUTE_DTYPE == torch.bfloat16 for s in slices)
    video = pad_4n1(model.to_input(frames)).permute(0, 3, 1, 2).to(torch.bfloat16)
    assert torch.equal(torch.cat(slices, dim=1), model.input_transform(TARGET)(video))


@pytest.fixture(scope="module")
def cpu_models() -> model.Models:
    """A VAE built from the configs with random weights, on the CPU, in a runner."""
    from seedvr2x.vendor.common.config import create_object, load_config
    from seedvr2x.vendor.core.infer import VideoDiffusionInfer

    config: Any = load_config(str(model.VENDOR / "configs_7b" / "main.yaml"))
    vae_yaml = model.VENDOR / "models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml"
    config.vae.model = OmegaConf.merge(config.vae.model, load_config(str(vae_yaml)))
    config.vae.dtype = "float32"
    torch.manual_seed(0)
    vae_config: Any = config.vae.model
    vae = create_object(vae_config).eval()
    vae.set_causal_slicing(**config.vae.slicing)
    runner: Any = VideoDiffusionInfer(config, cast(Any, SimpleNamespace(log=lambda *a, **k: None)))
    runner.vae = vae
    return cast(model.Models, SimpleNamespace(runner=runner, device=torch.device("cpu")))


@pytest.mark.parametrize("count", [1, 2, 6, 13])
def test_streamed_encode_is_the_one_pass_encode(cpu_models: model.Models, count: int) -> None:
    # The shot read in slices, each prepared and encoded as it comes, must give the latent of the
    # whole shot prepared and encoded at once, bit for bit, in the same memory layout (the noise
    # drawn from it depends on it). The 32 rows are padded to 64 (16 reflected, 16 black).
    frames = np.random.default_rng(count).random((count, 32, 48, 3), dtype=np.float32)
    transform = model.input_transform((32, 48))
    padded = padded_length(count)
    with torch.no_grad():
        video = pad_4n1(model.to_input(frames)).permute(0, 3, 1, 2).to(torch.float32)
        one_pass = model.encode(cpu_models, transform(video))
        slices = (
            transform(part.permute(0, 3, 1, 2).to(torch.float32))
            for part in input_slices(Reads(frames), count, model.encode_slices(cpu_models, padded))
        )
        streamed = model.encode_stream(cpu_models, slices, padded)
    assert streamed.shape == one_pass.shape == ((padded - 1) // 4 + 1, 8, 6, 16)
    assert torch.equal(streamed, one_pass)
    assert streamed.stride() == one_pass.stride()


def test_streamed_decode_is_the_one_pass_decode(cpu_models: model.Models) -> None:
    # The streamed decode must give the one-pass decode's frames bit for bit, through the same
    # slices and causal memory.
    models, runner = cpu_models, cast(Any, cpu_models).runner
    latent = torch.randn(4, 4, 6, 16)  # (T', h, w, C): 13 frames of 32x48
    with torch.no_grad():
        one_pass = runner.vae_decode([latent])[0]
        streamed = torch.cat(list(model.decode_stream(models, latent)), dim=1)
    assert streamed.shape == one_pass.shape == (3, 13, 32, 48)
    assert torch.equal(streamed, one_pass)


def test_noise_is_drawn_on_the_models_device(monkeypatch: pytest.MonkeyPatch) -> None:
    # The encode gives its latent in system memory, and a resumed shot's is read there: the noise
    # is drawn from it on the models' device, where numz draws it, the CPU's generator giving
    # other numbers than CUDA's; in the latent's own layout, which the draw follows.
    device = torch.device("cpu")  # no other here: what is drawn from, then, not where
    models = cast(model.Models, SimpleNamespace(device=device))
    monkeypatch.setattr(model, "condition", lambda models, noise, latent: latent)
    monkeypatch.setattr(model, "sample", lambda models, noise, cond: noise)
    randn_like = torch.randn_like
    drawn_from: list[torch.Tensor] = []

    def recorded(tensor: torch.Tensor, **options: Any) -> torch.Tensor:
        drawn_from.append(tensor)
        return randn_like(tensor, **options)

    monkeypatch.setattr(torch, "randn_like", recorded)
    encoded = latent_of(5, (4, 6)).permute(3, 0, 1, 2).contiguous().permute(1, 2, 3, 0)
    assert not encoded.is_contiguous()  # channel-major memory, as an encode's
    on_device = encoded.to(device, copy=True)
    assert on_device.stride() == encoded.stride()
    moves: list[tuple[Any, ...]] = []

    class InSystemMemory(torch.Tensor):
        def to(self, *args: Any, **kwargs: Any) -> torch.Tensor:
            moves.append((*args, kwargs))
            return on_device

    layout = [(0, 3), (2, 5)]
    given = encoded.as_subclass(InSystemMemory)
    noise = list(shot.sample_windows(models, given, layout, 7))
    # Moved first, to the models' device, and the draw is from what came back; then back to
    # system memory, where it waits through the windows (the one device here).
    assert moves == [(device, {}), (device, {})]
    assert len(drawn_from) == 1 and drawn_from[0] is on_device
    assert [window.shape for window in noise] == [(3, 4, 6, 16), (3, 4, 6, 16)]
    # A latent on the device already gives the same draw, as does the one in system memory.
    for latent in (on_device, encoded):
        again = list(shot.sample_windows(models, latent, layout, 7))
        assert all(same(one, other) for one, other in zip(again, noise, strict=True))
    assert drawn_from[1] is on_device and drawn_from[2] is encoded
    # The draw follows the layout: the same values in contiguous memory draw other noise.
    other = list(shot.sample_windows(models, encoded.contiguous(), layout, 7))
    assert not same(other[0], noise[0])
    # A window from which the run resumes draws the shot's noise again.
    resumed = list(shot.sample_windows(models, encoded, layout, 7, 1))
    assert len(resumed) == 1 and same(resumed[0], noise[1])
    with pytest.raises(ValueError, match="over a latent of 5"):
        list(shot.sample_windows(models, encoded, [(0, 3), (2, 6)], 7))


# The DiT's windows (sample_windows) over a stand-in for the DiT. Its device is the CPU, where a
# copy to the device or back is the tensor itself: what is held here is the order of things and
# the values each window is given.
DEVICE = cast(model.Models, SimpleNamespace(device=torch.device("cpu")))
# A shot of 33 frames, 9 latents, in windows of 5.
LAYOUT = [(0, 5), (3, 7), (5, 9)]


class Dit:
    """A stand-in for the DiT, its output a window's noise and latent combined. Recorded: what
    leads to a window, in order (`events`: the seed set, the draw with the shape, the strides
    and the options of what is drawn, each window), the noise and the latent each window is
    given (`given`), and whether gradients were enabled at the draw and in each window
    (`grad`)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.events: list[object] = []
        self.given: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.grad: list[bool] = []
        set_seed, randn_like = model.set_seed, torch.randn_like

        def seeded(seed: int) -> None:
            self.events.append(("seed", seed))
            set_seed(seed)

        def drawn(tensor: torch.Tensor, **options: Any) -> torch.Tensor:
            self.events.append(("draw", tuple(tensor.shape), tensor.stride(), options))
            self.grad.append(torch.is_grad_enabled())
            return randn_like(tensor, **options)

        def sample(models: object, noise: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
            self.events.append("window")
            self.given.append((noise, cond))
            self.grad.append(torch.is_grad_enabled())
            return noise * 0.5 + cond

        monkeypatch.setattr(model, "set_seed", seeded)
        monkeypatch.setattr(torch, "randn_like", drawn)
        monkeypatch.setattr(model, "condition", lambda models, noise, latent: latent)
        monkeypatch.setattr(model, "sample", sample)


@pytest.mark.parametrize("start", [0, 1, 2])
def test_windows_are_slices_of_one_draw(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, start: int
) -> None:
    # The noise is drawn once for the shot: torch.randn_like of its whole latent, in the layout
    # its encode gave it, right after the seed is set, when the first window is asked for and
    # not before, whichever window that is: a shot resumed at its second or third window draws
    # the noise an uninterrupted one drew. Each window is given its slices' values of the noise
    # and of the latent, and the steps that compute run without gradients.
    assert shot_layout(33, 5) == LAYOUT
    caplog.set_level(logging.DEBUG)
    latent = channel_major(9)
    given = latent.clone()
    model.set_seed(1234)
    noise = torch.randn_like(latent, dtype=torch.bfloat16)
    dit = Dit(monkeypatch)
    torch.manual_seed(0)  # the generators as something else left them
    sampled = sample_windows(DEVICE, latent, LAYOUT, 1234, start)
    assert dit.events == []
    outputs = [next(sampled)]
    draw = ("draw", (9, 3, 5, 16), (15, 5, 1, 135), {"dtype": model.COMPUTE_DTYPE})
    assert dit.events == [("seed", 1234), draw, "window"] and latent.stride() == draw[2]
    assert torch.is_grad_enabled()
    outputs += list(sampled)
    assert dit.events[3:] == ["window"] * (len(LAYOUT) - start - 1)
    assert len(dit.grad) == 1 + len(LAYOUT) - start and not any(dit.grad)
    windows = zip(LAYOUT[start:], dit.given, outputs, strict=True)
    for (s, e), (window_noise, window_latent), output in windows:
        assert window_noise.dtype == model.COMPUTE_DTYPE == torch.bfloat16
        assert torch.equal(window_noise, noise[s:e]) and torch.equal(window_latent, latent[s:e])
        assert torch.equal(output, noise[s:e] * 0.5 + latent[s:e])
    assert torch.equal(latent, given) and latent.stride() == given.stride()
    assert "noise drawn" not in caplog.text  # the draw's memory is a GPU's: nothing on the CPU
    with pytest.raises(ValueError, match="over a latent of 9"):
        next(sample_windows(DEVICE, latent, [(0, 5), (3, 8)], 1234))


def test_the_draw_is_logged_on_a_gpu(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # With -v, a line at each shot's draw: its latents, what the GPU has allocated then, its
    # latent and its noise there, and the unit's peak so far, read and never reset, in GiB to 3
    # decimals: the moment the planner bounds (planner.dit_draw), which a run on a card shows.
    # No GPU here: a device by its name alone, the copies to it and back left out, and the
    # driver's two figures given.
    gpu = cast(model.Models, SimpleNamespace(device=torch.device("cuda")))
    asked: list[str] = []

    def figure(name: str, count: float) -> Callable[[torch.device], int]:
        def read(device: torch.device) -> int:
            asked.append(name)
            return round(count * 2**30)

        return read

    monkeypatch.setattr(shot, "moved", lambda tensor, device, what: tensor)
    monkeypatch.setattr(torch.cuda, "memory_allocated", figure("allocated", 1.5))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", figure("peak", 16.25))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", figure("reset", 0))
    caplog.set_level(logging.DEBUG, logger="seedvr2x.runtime.shot")
    latent = channel_major(9)
    model.set_seed(1234)
    expected = torch.randn_like(latent, dtype=model.COMPUTE_DTYPE)
    noise, kept = shot._drawn(gpu, latent, 1234)  # pyright: ignore[reportPrivateUsage]
    assert torch.equal(noise, expected) and kept is latent
    assert caplog.messages == [
        "noise drawn for the shot's 9 latents: 1.500 GiB allocated on the GPU, its latent and its"
        " noise there; the unit's peak so far 16.250 GiB"
    ]
    assert asked == ["allocated", "peak"]


def test_windows_from_the_units_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The latent as the units give it, the copy kept in memory, the one written for a resume, or
    # the one a resume reads back: the windows the encode's own latent gives. And that copy is
    # the one the windows are cut from, no other being made of the latent: on the CPU, where a
    # slice sent to the device is the slice, each window's latent is a view of the copy given.
    latent = channel_major(9)
    dit = Dit(monkeypatch)
    expected = list(sample_windows(DEVICE, latent, LAYOUT, 1234))
    disk = DiskUnits(tmp_path, job(tmp_path, [Shot(0, 33)], [OutputSegment("a", 0, 33)]))
    in_memory = Units()
    copies = [in_memory.save_latent(0, latent), disk.save_latent(0, latent), disk.latent(0)]
    assert copies[0] is in_memory.latent(0) and copies[2] is not latent
    for copy in copies:
        assert copy is not None and copy.device.type == "cpu"
        dit.given.clear()
        windows = list(sample_windows(DEVICE, copy, LAYOUT, 1234))
        for ours, theirs in zip(windows, expected, strict=True):
            assert torch.equal(ours, theirs) and ours.stride() == theirs.stride()
        whole = copy.untyped_storage().data_ptr()
        assert len(dit.given) == len(LAYOUT)
        assert all(cut.untyped_storage().data_ptr() == whole for _, cut in dit.given)


def test_no_output_is_kept_once_given(monkeypatch: pytest.MonkeyPatch) -> None:
    # A window's output is the caller's alone once given: while the next window runs, nothing of
    # sample_windows refers to it, so that it leaves the device as soon as the run has kept its
    # copy, before the next window's pass. Decorated with torch.no_grad, the generator would
    # keep it through that pass (shot.sample_windows).
    given: list[weakref.ref[torch.Tensor]] = []
    alive: list[bool] = []

    def sample(models: object, noise: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        alive.append(any(output() is not None for output in given))
        return noise * 0.5 + cond

    monkeypatch.setattr(model, "condition", lambda models, noise, latent: latent)
    monkeypatch.setattr(model, "sample", sample)
    sampled = sample_windows(DEVICE, channel_major(9), LAYOUT, 1234)
    while (output := next(sampled, None)) is not None:
        given.append(weakref.ref(output))
        del output
    assert len(given) == len(LAYOUT) and alive == [False] * len(LAYOUT)
    assert all(output() is None for output in given)


def test_the_device_is_given_the_shot_for_its_draw_then_each_window_s_slices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # What goes to the device and what leaves it: the shot's latent whole for the noise's draw,
    # the noise and the latent back to system memory after it, then each window's slices alone,
    # so that the device holds nothing of the shot beside a window (shot.sample_windows). On the
    # CPU a move is the tensor itself: the tensors here record the moves they are asked for, the
    # shape moved and its way, by the device named: shot.CPU itself for system memory.
    moves: list[tuple[str, tuple[int, ...]]] = []

    class Recorded(torch.Tensor):
        def to(self, *args: Any, **kwargs: Any) -> torch.Tensor:
            home = bool(args) and args[0] is shot.CPU
            moves.append(("system memory" if home else "device", tuple(self.shape)))
            return super().to(*args, **kwargs)

    monkeypatch.setattr(model, "condition", lambda models, noise, latent: latent)
    monkeypatch.setattr(model, "sample", lambda models, noise, cond: noise * 0.5 + cond)
    assert DEVICE.device is not shot.CPU
    latent = channel_major(9).as_subclass(Recorded)
    whole = tuple(latent.shape)

    def slices(start: int, end: int) -> list[tuple[str, tuple[int, ...]]]:
        # A window's noise, then its latent.
        return [("device", (end - start, *whole[1:]))] * 2

    sampled = sample_windows(DEVICE, latent, LAYOUT, 1234)
    assert moves == []  # nothing moves before the first window is asked for
    next(sampled)
    # The latent to the device, the noise drawn there, then both to system memory.
    drawn = [("device", whole), ("system memory", whole), ("system memory", whole)]
    assert moves == [*drawn, *slices(*LAYOUT[0])]
    assert len(list(sampled)) == len(LAYOUT) - 1
    assert moves[len(drawn) :] == [move for window in LAYOUT for move in slices(*window)]


# A shot's tiled passes through encode_shot and decode_shot, over test_tiling.py's stand-in VAE.

# Frames of 28x20 resized 2x to 56x40, padded to 80x64 (8 rows and 8 columns reflected, 16 black
# each way): 8x10 latents, which tiles of 32 px sharing 8 cover in 9, 3 rows of 3.
SMALL, LARGE = (20, 28), (40, 56)
TILES = Tiles(32, 8)
# One tile holds the picture: the untiled pass.
WHOLE = Tiles(1024)


@pytest.fixture
def stand_in() -> model.Models:
    return models_of(StandIn())


def shot_frames(count: int) -> npt.NDArray[np.float32]:
    """A shot's `count` frames as the encode reads them, (count, 20, 28, 3) float32 in [0, 1]."""
    return np.random.default_rng(count).random((count, *SMALL, 3), dtype=np.float32)


def shot_latent(count: int) -> torch.Tensor:
    """The merged latents of a shot of `count` frames at LARGE, (T', 8, 10, 16) bfloat16, small
    enough for the stand-in's frames to stay within [-1, 1] but for a few, which the clamp
    takes."""
    padded = padded_size(LARGE)
    assert padded == (64, 80) and len(tiling.grid((8, 10), TILES)) == 9
    latents = (padded_length(count) - 1) // 4 + 1
    return (latent_of(latents, (padded[0] // 8, padded[1] // 8), seed=count).float() * 0.3).to(
        torch.bfloat16
    )


def written_by(
    models: model.Models, latent: torch.Tensor, count: int, **options: Any
) -> list[npt.NDArray[np.float32]]:
    """The frames decode_shot writes for a shot of `count` frames at LARGE, write by write."""
    written: list[npt.NDArray[np.float32]] = []
    decode_shot(models, latent, count, LARGE, written.append, **options)
    assert sum(frames.shape[0] for frames in written) == count
    assert all(frames.shape[1:] == (*output_size(LARGE), 3) for frames in written)
    return written


def same_writes(ours: list[npt.NDArray[np.float32]], theirs: list[npt.NDArray[np.float32]]) -> bool:
    """Whether two decodes wrote the same frames in the same writes, bit for bit."""
    return len(ours) == len(theirs) and all(
        one.shape == other.shape and one.tobytes() == other.tobytes()
        for one, other in zip(ours, theirs, strict=True)
    )


@pytest.mark.parametrize("count", [1, 6, 13])
def test_tiled_encode_is_tilings(stand_in: model.Models, count: int) -> None:
    # encode_shot with tiles: tiling's encode of the encoder's own input, bit for bit, in the
    # same memory layout, the frames read once; with one tile, the untiled pass.
    frames = shot_frames(count)
    reads = Reads(frames)
    tiled = encode_shot(stand_in, reads, count, LARGE, 42, tiles=TILES)
    assert sum(reads.counts) == count and all(n <= 5 for n in reads.counts)
    inputs = encoder_inputs(stand_in, Reads(frames), count, LARGE)
    direct = tiling.encode_tiled(
        stand_in, inputs, padded_length(count), TILES, Scratch(None, MEMORY)
    )
    assert tiled.shape == ((padded_length(count) - 1) // 4 + 1, 8, 10, 16)
    assert same(tiled, direct) and tiled.stride() == direct.stride()
    untiled = encode_shot(stand_in, Reads(frames), count, LARGE, 42)
    assert untiled.stride() == tiled.stride() and not same(tiled, untiled)
    one = encode_shot(stand_in, Reads(frames), count, LARGE, 42, tiles=WHOLE)
    assert same(one, untiled) and one.stride() == untiled.stride()


def tiled_slices(models: model.Models, latent: torch.Tensor) -> Iterator[torch.Tensor]:
    """tiling's decode of latent under TILES, in model.decode_stream's place: the untiled pass
    then takes its slices, and gives what a tiled one must."""
    return tiling.decode_tiled(models, latent, TILES, Scratch(None, MEMORY))


@pytest.mark.parametrize("count", [1, 6, 13])
def test_tiled_decode_is_tilings(
    stand_in: model.Models, monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    # decode_shot with tiles: tiling's slices, checked, cropped and written as a streamed
    # decode's are, write by write; with one tile, the untiled pass.
    latent = shot_latent(count)
    given = latent.clone()
    tiled = written_by(stand_in, latent, count, tiles=TILES)
    untiled = written_by(stand_in, latent, count)
    with monkeypatch.context() as patch:
        patch.setattr(model, "decode_stream", tiled_slices)
        expected = written_by(stand_in, latent, count)
    assert same_writes(tiled, expected) and same(latent, given)
    assert [frames.shape[0] for frames in tiled] == [frames.shape[0] for frames in untiled]
    assert not same_writes(tiled, untiled)  # blended
    # Few values at the clamp's ends: the comparison is of the decode's own values.
    values = np.concatenate(tiled)
    assert ((values > 0) & (values < 1)).mean() > 0.99
    assert same_writes(written_by(stand_in, latent, count, tiles=WHOLE), untiled)


def test_tiled_decode_is_corrected_as_a_streamed_one(
    stand_in: model.Models, monkeypatch: pytest.MonkeyPatch
) -> None:
    # With split, each slice a tiled decode hands on is corrected against its own frames of the
    # reference and written, as a streamed decode's is (_correct), the last one once the copy is
    # read whole. The shot's input copy is read from memory here, without ffmpeg.
    count = 13
    frames, latent = shot_frames(count), shot_latent(count)
    copies: list[Reads] = []

    @contextmanager
    def copy_reader(
        correction: Correction, count: int
    ) -> Generator[Callable[[int], npt.NDArray[np.float32]]]:
        copies.append(Reads(frames))
        yield copies[-1]

    monkeypatch.setattr(shot, "_copy_reader", copy_reader)
    correction = Correction(Path("input.mkv"), Path("input.crc32"), SMALL[1], SMALL[0])
    tiled = written_by(stand_in, latent, count, correction=correction, tiles=TILES)
    with monkeypatch.context() as patch:
        patch.setattr(model, "decode_stream", tiled_slices)
        expected = written_by(stand_in, latent, count, correction=correction)
    assert same_writes(tiled, expected)
    assert [written.shape[0] for written in tiled] == [5, 4, 4]
    assert [sum(copy.counts) for copy in copies] == [count, count]
    # Corrected, and tiled: neither the tiled decode alone nor the untiled one corrected.
    assert not same_writes(tiled, written_by(stand_in, latent, count, tiles=TILES))
    assert not same_writes(tiled, written_by(stand_in, latent, count, correction=correction))


def test_tiled_passes_keep_their_scratch_where_told(stand_in: model.Models, tmp_path: Path) -> None:
    count = 13
    frames, latent = shot_frames(count), shot_latent(count)
    vae = cast(Any, stand_in).runner.vae
    directory = tmp_path / "resume" / "shot_000000"  # not there yet
    path = directory / tiling.SCRATCH
    seen: list[bool] = []
    vae.before = lambda: seen.append(path.is_file() and mapped(path))

    # Within its allowance, the scratch is in system memory: nothing on disk, no directory made.
    # The allowance is the pass's to be given, as a run gives it its plan's; without one, here
    # for the encode, a default of 8 GiB (tiling.DEFAULT_ALLOWANCE).
    needed = Scratch.needed(count, padded_size(LARGE))
    assert needed < tiling.DEFAULT_ALLOWANCE == 8 * 2**30
    in_memory = encode_shot(
        stand_in, Reads(frames), count, LARGE, 42, tiles=TILES, scratch=directory
    )
    within: dict[str, Any] = {"tiles": TILES, "scratch": directory, "allowance": needed}
    frames_in_memory = written_by(stand_in, latent, count, **within)
    assert len(seen) == 9 * 3 * 2 and not any(seen) and not list(tmp_path.iterdir())

    # Beyond it, here an allowance one byte short, then of 0: the file in the directory given,
    # made with it, there through every tile's pass and every write, gone after; the directory
    # stays, the run's to remove with the shot's units. The same bits either way.
    seen.clear()
    beyond: dict[str, Any] = {"tiles": TILES, "scratch": directory, "allowance": needed - 1}
    on_disk = encode_shot(stand_in, Reads(frames), count, LARGE, 42, **beyond)
    assert len(seen) == 9 * 3 and all(seen) and same(on_disk, in_memory)
    assert not list(directory.iterdir()) and not mapped(path)
    seen.clear()
    there: list[bool] = []
    frames_on_disk: list[npt.NDArray[np.float32]] = []

    def write(written: npt.NDArray[np.float32]) -> None:
        there.append(path.is_file())
        frames_on_disk.append(written)

    decode_shot(stand_in, latent, count, LARGE, write, tiles=TILES, scratch=directory, allowance=0)
    assert len(seen) == 9 * 3 and all(seen) and there == [True] * 3
    assert same_writes(frames_on_disk, frames_in_memory)
    assert not list(directory.iterdir()) and not mapped(path)

    # The picture within one tile: the untiled pass, which asks for no scratch, and so runs
    # with nowhere to put one, whatever its allowance.
    seen.clear()
    encode_shot(stand_in, Reads(frames), count, LARGE, 42, tiles=WHOLE, allowance=0)
    written_by(stand_in, latent, count, tiles=WHOLE, allowance=0)
    assert len(seen) == 3 * 2 and not any(seen)

    # With nowhere to put its file, a scratch beyond the allowance is refused, as a file the run
    # can't write (cli._run reports a MediaError and exits 1).
    assert issubclass(ScratchError, MediaError)
    with pytest.raises(ScratchError, match="no directory for its file"):
        encode_shot(stand_in, Reads(frames), count, LARGE, 42, tiles=TILES, allowance=0)
    with pytest.raises(ScratchError, match="no directory for its file"):
        written_by(stand_in, latent, count, tiles=TILES, allowance=0)
    assert vae.carry.memory is None


@pytest.mark.parametrize("failure", ["writer", "tile", "correction"])
def test_tiled_decode_failing_leaves_nothing(
    stand_in: model.Models, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    # A tiled decode whose consumer fails, the writer or the correction, or one of whose tiles
    # does: its scratch's file goes at once, while the failure is still held with its traceback,
    # and the VAE's causal memory is cleared.
    count = 13
    latent, vae = shot_latent(count), cast(Any, stand_in).runner.vae
    path = tmp_path / tiling.SCRATCH
    there: list[bool] = []
    options: dict[str, Any] = {"tiles": TILES, "scratch": tmp_path, "allowance": 0}

    def write(frames: npt.NDArray[np.float32]) -> None:
        there.append(path.is_file() and mapped(path))
        raise Failing("the writer")

    if failure == "tile":
        failing_at(vae, 14)  # the fifth tile's last slice
    if failure == "correction":

        @contextmanager
        def copy_reader(
            correction: Correction, count: int
        ) -> Generator[Callable[[int], npt.NDArray[np.float32]]]:
            def read(n: int) -> npt.NDArray[np.float32]:
                there.append(path.is_file() and mapped(path))
                raise Failing("the copy")

            yield read

        monkeypatch.setattr(shot, "_copy_reader", copy_reader)
        options["correction"] = Correction(tmp_path / "input.mkv", tmp_path / "input.crc32", 28, 20)
    with pytest.raises(Failing) as raised:
        decode_shot(stand_in, latent, count, LARGE, write, **options)
    assert there == ([] if failure == "tile" else [True])
    assert not path.exists() and vae.carry.memory is None
    assert not list(tmp_path.iterdir())
    del raised
    assert not mapped(path)


def test_untiled_decode_failing_clears_its_memory(stand_in: model.Models) -> None:
    # An untiled decode whose consumer fails is closed at once, as a tiled one is: the VAE's
    # causal memory, on the device, is cleared while the failure is still held with its
    # traceback, whose frames hold the pass.
    count = 13
    latent, vae = shot_latent(count), cast(Any, stand_in).runner.vae
    kept: list[bool] = []

    def write(frames: npt.NDArray[np.float32]) -> None:
        kept.append(vae.carry.memory is not None)
        raise Failing("the writer")

    with pytest.raises(Failing) as raised:
        decode_shot(stand_in, latent, count, LARGE, write)
    assert kept == [True]  # in the middle of the pass, its memory kept
    assert vae.carry.memory is None
    del raised


def test_tiled_encode_failing_leaves_nothing(stand_in: model.Models, tmp_path: Path) -> None:
    # A tiled encode whose source fails while its slices are kept, or one of whose tiles does:
    # its scratch's file goes, and the VAE's causal memory is cleared.
    on_disk: dict[str, Any] = {"tiles": TILES, "scratch": tmp_path, "allowance": 0}
    count = 13
    frames, vae = shot_frames(count), cast(Any, stand_in).runner.vae
    path = tmp_path / tiling.SCRATCH
    there: list[bool] = []

    def read(n: int) -> npt.NDArray[np.float32]:
        if n != 5:
            there.append(path.is_file())
            raise Failing("the source")
        return frames[:5]

    with pytest.raises(Failing, match="the source"):
        encode_shot(stand_in, read, count, LARGE, 42, **on_disk)
    assert there == [True] and not path.exists()
    failing_at(vae, 4)  # the second tile's second slice
    with pytest.raises(Failing, match="pass 4"):
        encode_shot(stand_in, Reads(frames), count, LARGE, 42, **on_disk)
    assert not path.exists() and vae.carry.memory is None
    assert not list(tmp_path.iterdir())


def allocated(run: Callable[[], object]) -> tuple[int, int]:
    """(most, left): the most bytes torch allocated in system memory while run() ran, over what
    was allocated before it, and what was left of them at its end, its result still held. By
    torch's memory profiler, every allocation and release of its CPU allocator in order."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    with torch.profiler.profile(activities=activities, profile_memory=True) as profiled:
        result = run()
    assert profiled.profiler is not None and profiled.profiler.kineto_results is not None
    # torch's stubs don't give a memory event's bytes.
    recorded: list[Any] = list(profiled.profiler.kineto_results.events())
    events = [event for event in recorded if event.name() == "[memory]"]
    most = left = 0
    for event in sorted(events, key=lambda event: event.start_ns()):
        left += event.nbytes()
        most = max(most, left)
    del result
    return most, left


# The padded frame the planner's count is held to below, and its latents: (16, 24) positions.
SMALL_PADDED = (128, 192)


def contiguous(latents: int, seed: int = 0) -> torch.Tensor:
    """A window as the DiT gives it, (t, 16, 24, 16) bfloat16, contiguous."""
    return latent_of(latents, (16, 24), seed)


def test_the_check_allocates_two_and_a_half_latents() -> None:
    # The check of an encode's latent for NaN and inf, on the CPU, where the encode leaves it
    # (run._sample): torch allocates the latent's absolute values, as large, and three masks of
    # a byte a value, 2.5 times a bfloat16 latent's bytes, all given back; with the latent, the
    # planner's "check" (planner.host_moments), whatever the latent's length and its memory.
    for latents in (40, 200):
        latent = contiguous(latents).permute(3, 0, 1, 2).contiguous().permute(1, 2, 3, 0)
        assert not latent.is_contiguous() and latent.dtype == torch.bfloat16
        most, left = allocated(lambda latent=latent: finite(latent, "a latent"))
        assert (most, left) == (latent.nbytes + planner.CHECK_MASKS * latent.numel(), 0)
        assert most == 2.5 * latent.nbytes
        counted = planner.host_moments(SMALL_PADDED, latents, None)
        assert counted["check"] == latent.nbytes + most == 3.5 * latent.nbytes


@pytest.mark.parametrize(("latents", "cap"), [(13, 13), (48, 30), (48, 8), (100, 7), (100, 5)])
def test_the_merge_allocates_what_the_planner_counts(latents: int, cap: int) -> None:
    # A shot's windows merged at its decode's start (run._decode): beside the windows, torch
    # allocates the merged latent, the 2 latents mixed at a join, and from three windows on the
    # latent merged before the last window, made anew at each join; the planner's "merge" with
    # the windows (planner.host_moments), to the few bytes of the mix's weights. One window is
    # the merged latent itself: nothing.
    layout = window_layout(latents, cap)
    windows = [contiguous(end - start, seed) for seed, (start, end) in enumerate(layout)]
    kept = sum(window.nbytes for window in windows)
    one = windows[0].nbytes // (layout[0][1] - layout[0][0])
    most, left = allocated(lambda: merge_windows(windows, layout))
    counted = planner.host_moments(SMALL_PADDED, latents, cap)
    assert counted["decode"] == latents * one
    assert 0 <= kept + most - counted["merge"] <= 64
    assert left == (0 if len(layout) == 1 else counted["decode"])
    # And with one file, the last window holds what it counts: the latent, the noise and every
    # window kept.
    assert planner.host_moments(SMALL_PADDED, latents, cap, True)["windows"] == (
        2 * latents * one + kept
    )


def test_a_window_s_file_allocates_the_window_alone(tmp_path: Path) -> None:
    # A window written to its file from its copy in system memory, and read back for the decode
    # (units.DiskUnits): torch allocates nothing to write it, and the tensor alone to read it.
    save = units._save  # pyright: ignore[reportPrivateUsage]
    load = units._load  # pyright: ignore[reportPrivateUsage]
    window, path = contiguous(8), tmp_path / "window_0000.pt"
    assert allocated(lambda: save(window, path)) == (0, 0)
    assert allocated(lambda: load(path)) == (window.nbytes, window.nbytes)


def test_the_encode_s_end_allocates_the_latent_beside_its_buffer(stand_in: model.Models) -> None:
    # An encode's end (model.to_latent): the latent's own memory beside the buffer it is made
    # from, as large, the planner's "encode" (planner.host_moments); what its slices allocate on
    # their way, 2 latents at a time at the most, is the device's in a run, on the CPU here.
    latents = 40
    means = contiguous(latents).permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    one = means.nbytes // latents
    most, left = allocated(lambda: model.to_latent(stand_in, means))
    counted = planner.host_moments(SMALL_PADDED, latents, None)
    assert counted["encode"] == means.nbytes + left == 2 * latents * one
    assert left <= most <= left + 8 * one

"""The shot pipeline's parts on the CPU: windows, mixing weights, padding, the encoder's input
and split's reference, streamed encode and decode; the DiT's windows over a stand-in, their one
draw of noise, held to its digest, and what waits where between them; a shot's tiled encode and
decode
(runtime/tiling.py) through encode_shot and decode_shot, over test_tiling.py's stand-in VAE; and
what a shot's steps on the CPU allocate in system memory, which the planner counts
(planner.host_moments).

One test needs a GPU (pytest -m gpu) with the 7B whole on it and a window beside (20 GiB free),
and the weights (SEEDVR2X_MODEL_DIR: a 7B DiT in fp16 and the VAE, as tests/test_placement.py
takes them): a shot's windows through the real DiT, held to a one-window shot's on the device."""

import gc
import logging
import math
import re
import weakref
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
import torch
from omegaconf import OmegaConf
from test_tiling import (
    DITS,
    MEMORY,
    VAES,
    Failing,
    StandIn,
    failing_at,
    found,
    latent_of,
    mapped,
    models_of,
    same,
)
from test_units import channel_major, job

from seedvr2x.media.ffmpeg import MediaError
from seedvr2x.runtime import model, planner, shot, tiling, units
from seedvr2x.runtime.job import OutputSegment, Shot, output_size, padded_size
from seedvr2x.runtime.plan import Plan, Tiles
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
from seedvr2x.runtime.units import Units

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


# The frames a shot of `count` frames (value = index) ends with once padded to 4n + 1, by its
# remainder modulo 4 and for the short shots (1 to 4 frames): the last frame repeated by default
# (DESIGN.md, Pipeline step 1, decided on 2026-10-19; ByteDance's cut_videos), the frames before
# it mirrored in numz's tail (cli.NUMZ_TAIL; generation_utils.py:635-656 at 4490bd1), the last one
# repeated there only beyond a full mirror (a shot of 2 frames).
REPEATED = {
    1: [0],
    2: [0, 1, 1, 1, 1],
    3: [0, 1, 2, 2, 2],
    4: [0, 1, 2, 3, 3],
    5: [0, 1, 2, 3, 4],
    6: [0, 1, 2, 3, 4, 5, 5, 5, 5],
    7: [0, 1, 2, 3, 4, 5, 6, 6, 6],
    8: [0, 1, 2, 3, 4, 5, 6, 7, 7],
    9: [0, 1, 2, 3, 4, 5, 6, 7, 8],
    10: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9],
    11: [*range(11), 10, 10],
    12: [*range(12), 11],
    13: list(range(13)),
}
MIRRORED = {
    **REPEATED,
    3: [0, 1, 2, 1, 0],
    4: [0, 1, 2, 3, 2],
    6: [0, 1, 2, 3, 4, 5, 4, 3, 2],
    7: [0, 1, 2, 3, 4, 5, 6, 5, 4],
    8: [0, 1, 2, 3, 4, 5, 6, 7, 6],
    10: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6],
    11: [*range(11), 9, 8],
    12: [*range(12), 10],
}


@pytest.mark.parametrize("numz_tail", [False, True], ids=["repeated", "numz-tail"])
@pytest.mark.parametrize("count", range(1, 14))
def test_shot_s_end_padded(count: int, numz_tail: bool) -> None:
    # A shot of 4n + 1 or 1 frame is left as it is; of 2, 3 or 4 frames, padded to 5; else by 1
    # to 3 frames (padded_length). Frames of (T, H, W, 3), as a shot's, and of one dimension.
    expected = (MIRRORED if numz_tail else REPEATED)[count]
    frames = torch.arange(count).view(count, 1, 1, 1).expand(count, 2, 2, 3)
    padded = pad_4n1(frames, numz_tail=numz_tail)
    assert padded[:, 0, 0, 0].tolist() == expected
    assert len(expected) == padded_length(count) and padded.shape[1:] == (2, 2, 3)
    assert pad_4n1(torch.arange(count), numz_tail=numz_tail).tolist() == expected
    assert torch.equal(padded[:count], frames)


def test_shot_s_end_is_repeated_by_default() -> None:
    # The default is the repeat, numz's mirror a mode of the tests' own: pad_4n1's default is the
    # repeat, and the last frame is the one repeated, whole (not a value of it).
    frames = torch.rand(6, 2, 3, 3, generator=torch.Generator().manual_seed(0))
    padded = pad_4n1(frames)
    assert padded.shape[0] == 9 and all(torch.equal(padded[i], frames[-1]) for i in (6, 7, 8))
    mirrored = pad_4n1(frames, numz_tail=True)
    assert [torch.equal(mirrored[6 + i], frames[4 - i]) for i in range(3)] == [True] * 3


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


@pytest.mark.parametrize("numz_tail", [False, True], ids=["repeated", "numz-tail"])
@pytest.mark.parametrize("count", range(1, 22))
def test_streamed_input_is_the_padded_shot(count: int, numz_tail: bool) -> None:
    # Read in the VAE's slices (5 frames, then 4), padded at the end as pad_4n1 pads the whole,
    # the last frame repeated or, in numz's tail, the frames before it mirrored.
    frames = np.random.default_rng(count).random((count, 2, 3, 3), dtype=np.float32)
    reads = Reads(frames)
    sizes = vae_slices(padded_length(count))
    slices = list(input_slices(reads, count, sizes, numz_tail=numz_tail))
    assert [s.shape[0] for s in slices] == sizes
    assert sum(reads.counts) == count and all(n <= 5 for n in reads.counts)
    assert torch.equal(torch.cat(slices), pad_4n1(model.to_input(frames), numz_tail=numz_tail))


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
    # The encoder's input keeps numz's numerics (milestone 1): float16 frames, cast to bfloat16
    # before the transform (generation_phases.py:380-413); its padded frames are seedvr2x's, the
    # last frame repeated (at 6 frames numz mirrors 4, 3 and 2: test_encoder_input_in_numz_s_tail).
    frames = np.random.default_rng(count).random((count, *SOURCE, 3), dtype=np.float32)
    slices = list(encoder_inputs(SLICING, Reads(frames), count, TARGET))
    assert [s.shape[1] for s in slices] == vae_slices(padded_length(count))
    assert all(s.dtype == model.COMPUTE_DTYPE == torch.bfloat16 for s in slices)
    video = pad_4n1(model.to_input(frames)).permute(0, 3, 1, 2).to(torch.bfloat16)
    assert torch.equal(torch.cat(slices, dim=1), model.input_transform(TARGET)(video))


@pytest.mark.parametrize("count", [2, 3, 6, 13])
def test_encoder_input_in_numz_s_tail(count: int) -> None:
    # The encoder's input with the tests' numz tail (cli.NUMZ_TAIL): the shot's end mirrored as
    # pad_4n1 does, a different input from the default's for these shots but the 13-frame one,
    # which needs no padding, and the 2-frame one, padded alike by both (DESIGN.md, Pipeline
    # step 1).
    frames = np.random.default_rng(count).random((count, *SOURCE, 3), dtype=np.float32)
    mirrored = list(encoder_inputs(SLICING, Reads(frames), count, TARGET, numz_tail=True))
    repeated = list(encoder_inputs(SLICING, Reads(frames), count, TARGET))
    video = pad_4n1(model.to_input(frames), numz_tail=True).permute(0, 3, 1, 2)
    expected = model.input_transform(TARGET)(video.to(torch.bfloat16))
    assert torch.equal(torch.cat(mirrored, dim=1), expected)
    assert torch.equal(torch.cat(repeated, dim=1), torch.cat(mirrored, dim=1)) == (count in (2, 13))


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


def test_noise_is_drawn_in_system_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    # The encode gives its latent in system memory, and a resumed shot's is read there: the noise
    # is drawn from it there, by torch's CPU generator, whatever the models' device, so that it is
    # the same on every card (DESIGN.md, Pipeline step 2); in the latent's own layout, which the
    # draw follows. A latent given on the device is copied to system memory for it. numz's draw,
    # for the tests (cli.NUMZ_DRAW), is from the latent sent to the models' device.
    device = torch.device("cpu")  # no other here: what is drawn from, then, not where
    assert device is not shot.CPU
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
    elsewhere = encoded.to(device, copy=True)
    assert elsewhere.stride() == encoded.stride()
    moves: list[str] = []

    class OnTheDevice(torch.Tensor):
        def to(self, *args: Any, **kwargs: Any) -> torch.Tensor:
            moves.append("system memory" if args[0] is shot.CPU else "device")
            return elsewhere

    layout = [(0, 3), (2, 5)]
    given = encoded.as_subclass(OnTheDevice)
    noise = list(shot.sample_windows(models, given, layout, 7))
    # Copied to system memory first, and the draw is from that copy, where it waits through the
    # windows; nothing sent to the device.
    assert moves == ["system memory"]
    assert len(drawn_from) == 1 and drawn_from[0] is elsewhere
    assert [window.shape for window in noise] == [(3, 4, 6, 16), (3, 4, 6, 16)]
    # The latent in system memory gives the same draw, drawn from it as it is.
    again = list(shot.sample_windows(models, encoded, layout, 7))
    assert all(same(one, other) for one, other in zip(again, noise, strict=True))
    assert drawn_from[1] is encoded
    # The draw follows the layout: the same values in contiguous memory draw other noise.
    other = list(shot.sample_windows(models, encoded.contiguous(), layout, 7))
    assert not same(other[0], noise[0])
    # A window from which the run resumes draws the shot's noise again.
    resumed = list(shot.sample_windows(models, encoded, layout, 7, 1))
    assert len(resumed) == 1 and same(resumed[0], noise[1])
    with pytest.raises(ValueError, match="over a latent of 5"):
        list(shot.sample_windows(models, encoded, [(0, 3), (2, 6)], 7))
    # numz's draw: the latent sent to the models' device and drawn from there; then the noise
    # and the latent back to system memory. Here the one device is the CPU, whose generator
    # draws the same values from the same layout.
    moves.clear()
    numz = list(shot.sample_windows(models, given, layout, 7, numz_draw=True))
    assert moves == ["device", "system memory"]
    assert drawn_from[-1] is elsewhere
    assert all(same(one, other) for one, other in zip(numz, noise, strict=True))


# The noise of a shot of 21 frames at 1080p, 6 latents of 138 x 240 positions (1104 x 1920
# padded) in the layout an encode gives them, from seed 42: the SHA-256 of the bfloat16 values'
# bits, in the latent's index order (test_the_draw_is_held_to_its_digest).
DRAW_DIGEST = "90d8e6bcc5b463868eed13bf2f09493f998a771f4fc347d3e0f95492065f5f1e"


def test_the_draw_is_held_to_its_digest(cpu_models: model.Models) -> None:
    # The noise drawn on the CPU is the same on every card and every machine (DESIGN.md,
    # Pipeline step 2): held here to its digest for a fixed seed and shape, so that another
    # generator, another seeding or memory layout fails, and to its generator's being its own.
    # The layout is the encode's:
    # channel-major, a permuted view, which a 1-frame encode through the VAE gives here, its
    # strides those of a latent so built. The noise keeps the latent's shape, dtype and strides.
    import hashlib

    frames = np.zeros((1, 32, 48, 3), dtype=np.float32)
    with torch.no_grad():
        video = pad_4n1(model.to_input(frames)).permute(0, 3, 1, 2).to(torch.float32)
        encoded = model.encode(cpu_models, model.input_transform((32, 48))(video))

    def channel_major_of(shape: tuple[int, ...]) -> torch.Tensor:
        latents, height, width, channels = shape
        empty = torch.zeros(channels, latents, height, width, dtype=model.COMPUTE_DTYPE)
        return empty.permute(1, 2, 3, 0)

    assert encoded.stride() == channel_major_of(tuple(encoded.shape)).stride()
    latent = channel_major_of((6, 138, 240, 16))
    noise, kept = shot._drawn(DEVICE, latent, 42)  # pyright: ignore[reportPrivateUsage]
    assert kept is latent and noise.device.type == "cpu"
    assert noise.shape == latent.shape and noise.stride() == latent.stride()
    assert noise.dtype == model.COMPUTE_DTYPE == torch.bfloat16

    def digest(noise: torch.Tensor) -> str:
        return hashlib.sha256(noise.contiguous().view(torch.int16).numpy().tobytes()).hexdigest()

    assert digest(noise) == DRAW_DIGEST
    # The draw's generator is its own: torch's global one drawn from between the seeding and
    # the draw, as anything else in the process could, moves no value.
    set_seed = model.set_seed

    def seeded_then_drawn_from(seed: int) -> None:
        set_seed(seed)
        torch.rand(3)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(model, "set_seed", seeded_then_drawn_from)
        again, _ = shot._drawn(DEVICE, latent, 42)  # pyright: ignore[reportPrivateUsage]
    assert digest(again) == DRAW_DIGEST


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
            # A generator by its device and the seed it was given.
            said = {
                name: (value.device.type, value.initial_seed())
                if isinstance(value, torch.Generator)
                else value
                for name, value in options.items()
            }
            self.events.append(("draw", tuple(tensor.shape), tensor.stride(), said))
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
    # its encode gave it, from a CPU generator of its own seeded with the shot's seed, right
    # after the global generators are seeded as numz seeds them, when the first window is asked
    # for and not before, whichever window that is: a shot resumed at its second or third
    # window draws the noise an uninterrupted one drew. Each window is given its slices' values
    # of the noise and of the latent, and the steps that compute run without gradients. With -v,
    # the draw is logged: where, the shot's latents, its time and the noise's size.
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
    options = {"dtype": model.COMPUTE_DTYPE, "generator": ("cpu", 1234)}
    draw = ("draw", (9, 3, 5, 16), (15, 5, 1, 135), options)
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
    drawn = [message for message in caplog.messages if message.startswith("noise drawn")]
    assert len(drawn) == 1
    assert re.fullmatch(
        r"noise drawn on the CPU for the shot's 9 latents in \d+\.\d\d s: 0\.000 GiB", drawn[0]
    )
    with pytest.raises(ValueError, match="over a latent of 9"):
        next(sample_windows(DEVICE, latent, [(0, 5), (3, 8)], 1234))


def test_nothing_of_the_draw_goes_to_a_gpu_but_numz_s(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # On a GPU the noise is drawn in system memory, nothing of the shot sent there for it, and
    # its values are those the CPU draws; numz's draw, for the tests (cli.NUMZ_DRAW), sends the
    # latent to the GPU and draws there, as numz does (generation_phases.py:663-680), said in
    # the -v line. No GPU here: a device by its name alone, the copies to it and back recorded
    # and left out.
    gpu = cast(model.Models, SimpleNamespace(device=torch.device("cuda")))
    sent: list[str] = []

    def moving(tensor: torch.Tensor, device: torch.device, what: str) -> torch.Tensor:
        sent.append(f"{what} to {device.type}")
        return tensor

    monkeypatch.setattr(shot, "moved", moving)
    caplog.set_level(logging.DEBUG, logger="seedvr2x.runtime.shot")
    latent = channel_major(9)
    model.set_seed(1234)
    expected = torch.randn_like(latent, dtype=model.COMPUTE_DTYPE)
    noise, kept = shot._drawn(gpu, latent, 1234)  # pyright: ignore[reportPrivateUsage]
    assert torch.equal(noise, expected) and kept is latent
    assert sent == ["the shot's latent to cpu"]
    assert re.fullmatch(
        r"noise drawn on the CPU for the shot's 9 latents in \d+\.\d\d s: 0\.000 GiB",
        caplog.messages[-1],
    )
    sent.clear()
    noise, kept = shot._drawn(gpu, latent, 1234, numz_draw=True)  # pyright: ignore[reportPrivateUsage]
    assert torch.equal(noise, expected) and kept is latent
    assert sent == [
        "the shot's latent to cuda",
        "the shot's noise to cpu",
        "the shot's latent to cpu",
    ]
    assert re.fullmatch(
        r"noise drawn on the device \(numz's draw\) for the shot's 9 latents in \d+\.\d\d s:"
        r" 0\.000 GiB",
        caplog.messages[-1],
    )


def test_windows_from_the_units_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The latent as the units give it, the one written for a resume or the one a resume reads
    # back: the windows the encode's own latent gives. And that copy is the one the windows are
    # cut from, no other being made of the latent: on the CPU, where a slice sent to the device
    # is the slice, each window's latent is a view of the copy given.
    latent = channel_major(9)
    dit = Dit(monkeypatch)
    expected = list(sample_windows(DEVICE, latent, LAYOUT, 1234))
    disk = Units(tmp_path, job(tmp_path, [Shot(0, 33)], [OutputSegment("a", 0, 33)]))
    copies = [disk.save_latent(0, latent), disk.latent(0)]
    assert copies[0] is latent and copies[1] is not latent
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


@pytest.mark.parametrize("numz_draw", [False, True])
def test_the_device_is_given_each_window_s_slices_alone(
    monkeypatch: pytest.MonkeyPatch, numz_draw: bool
) -> None:
    # What goes to the device and what leaves it: each window's slices alone, the noise drawn in
    # system memory, so that the device holds nothing of the shot whole, nor beside a window
    # (shot.sample_windows); in numz's draw (cli.NUMZ_DRAW), the shot's latent whole for the
    # draw, then the noise and the latent back to system memory. On the CPU a move is the tensor
    # itself: the tensors here record the moves they are asked for, the shape moved and its way,
    # by the device named: shot.CPU itself for system memory.
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

    sampled = sample_windows(DEVICE, latent, LAYOUT, 1234, numz_draw=numz_draw)
    assert moves == []  # nothing moves before the first window is asked for
    next(sampled)
    # The latent in system memory already, where the noise is drawn; in numz's, the latent to
    # the device, the noise drawn there, then both to system memory.
    drawn = [("system memory", whole)]
    if numz_draw:
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


def whole_finite(tensor: torch.Tensor, what: str) -> None:
    """shot.finite as it was until 2026-10-12, the tensor read whole: what the check by slices
    is held to."""
    if not bool(torch.isfinite(tensor).all()):
        bad = int((~torch.isfinite(tensor)).sum())
        raise shot.NonFinite(f"{what}: {bad} of its {tensor.numel()} values NaN or inf")


def study_merge(sampled: list[torch.Tensor], layout: list[tuple[int, int]]) -> torch.Tensor:
    """shot._merge as it was until 2026-10-12, the study's own writing (blend_patch.py,
    latent_upscale): the latents merged so far, the mixed ones and the window's rest
    concatenated at each join. What the merge into one tensor is held to, to the bit."""
    merged = sampled[0].cpu()
    for (_, previous_end), (start, _), current in zip(
        layout, layout[1:], sampled[1:], strict=False
    ):
        shared = previous_end - start
        current = current.cpu()
        weights = torch.tensor(mix_weights(shared), dtype=torch.float32).to(current.dtype)
        weights = weights.view(shared, *([1] * (current.dim() - 1)))
        mixed = merged[start:previous_end] * weights + current[:shared] * (1.0 - weights)
        merged = torch.cat([merged[:start], mixed, current[shared:]])
    return merged


def test_the_check_allocates_two_and_a_half_times_one_latent() -> None:
    # The check of an encode's latent for NaN and inf, on the CPU, where the encode leaves it
    # (run._sample), reads it a latent at a time: torch allocates that latent's absolute values,
    # as large, and three masks of a byte a value, 2.5 times one bfloat16 latent's bytes, all
    # given back before the next; with the shot's latent, the planner's "check"
    # (planner.host_moments), whatever the shot's length and its memory. Read whole, as it was
    # until 2026-10-12, it allocated 2.5 times the shot's.
    for latents in (40, 200):
        latent = contiguous(latents).permute(3, 0, 1, 2).contiguous().permute(1, 2, 3, 0)
        assert not latent.is_contiguous() and latent.dtype == torch.bfloat16
        one = latent.nbytes // latents
        most, left = allocated(lambda latent=latent: finite(latent, "a latent"))
        assert (most, left) == (one + planner.CHECK_MASKS * one // 2, 0)
        assert most == 2.5 * one
        counted = planner.host_moments(SMALL_PADDED, latents, None)
        assert counted["check"] == latent.nbytes + most
        whole = allocated(lambda latent=latent: whole_finite(latent, "a latent"))
        assert whole == (2.5 * latent.nbytes, 0)


def test_the_check_by_slices_says_what_the_whole_one_said() -> None:
    # The check read slice by slice gives the verdict, and the words, of the check that read
    # the tensor whole: nothing for finite values, and the count of every NaN and inf, wherever
    # they are, in the first slice, in the last alone, in several, in all; over a latent in the
    # encode's layout, a window, a decode's slice in float32, and tensors of one dimension, of
    # none and of no value.
    def said(check: Callable[[torch.Tensor, str], None], tensor: torch.Tensor) -> str | None:
        try:
            check(tensor, "a unit")
        except shot.NonFinite as error:
            return str(error)
        return None

    latent = contiguous(9).permute(3, 0, 1, 2).contiguous().permute(1, 2, 3, 0)
    frames = torch.rand(5, 6, 8, 3)
    line, scalar = torch.rand(7), torch.tensor(1.5)
    none = torch.empty(0, 4, 4, 16, dtype=torch.bfloat16)
    cases: list[tuple[torch.Tensor, list[tuple[tuple[int, ...], float]]]] = [
        (latent, []),
        (latent, [((0, 0, 0, 0), math.nan)]),
        (latent, [((8, 15, 23, 15), math.inf)]),
        (latent, [((2, 1, 1, 1), math.nan), ((2, 3, 4, 5), -math.inf), ((7, 0, 0, 3), math.inf)]),
        (contiguous(4), [((3, 0, 0, 0), -math.inf), ((3, 0, 0, 1), math.nan)]),
        (frames, []),
        (frames, [((4, 5, 7, 2), math.nan)]),
        (line, []),
        (line, [((6,), math.inf)]),
        (scalar, []),
        (scalar, [((), math.nan)]),
        (none, []),
    ]
    refused = 0
    for tensor, bad in cases:
        tensor = tensor.clone()
        for place, value in bad:
            tensor[place] = value
        ours = said(finite, tensor)
        assert ours == said(whole_finite, tensor)
        assert (ours is None) == (not bad)
        if bad:
            assert ours == f"a unit: {len(bad)} of its {tensor.numel()} values NaN or inf"
            refused += 1
    assert refused == 7
    everything = torch.full((3, 2, 2), math.nan)
    assert said(finite, everything) == "a unit: 12 of its 12 values NaN or inf"


@pytest.mark.parametrize(("latents", "cap"), [(13, 13), (48, 30), (48, 8), (100, 7), (100, 5)])
def test_the_merge_allocates_what_the_planner_counts(latents: int, cap: int) -> None:
    # A shot's windows merged at its decode's start (run._decode): beside the windows, torch
    # allocates the merged latent, once, and at a join the two products and the sum of the 2
    # latents mixed, given back before the next join; the planner's "merge" with the windows
    # (planner.host_moments), to the few bytes of the mix's weights. One window is the merged
    # latent itself: nothing. The study's own writing, the merge until 2026-10-12, made the
    # shot's latents anew at each join: from three windows on it held them twice over.
    layout = window_layout(latents, cap)
    windows = [contiguous(end - start, seed) for seed, (start, end) in enumerate(layout)]
    kept = sum(window.nbytes for window in windows)
    one = windows[0].nbytes // (layout[0][1] - layout[0][0])
    most, left = allocated(lambda: merge_windows(windows, layout))
    counted = planner.host_moments(SMALL_PADDED, latents, cap)
    assert counted["decode"] == latents * one
    assert 0 <= kept + most - counted["merge"] <= 64
    assert left == (0 if len(layout) == 1 else counted["decode"])
    if len(layout) > 1:
        assert 0 <= most - (latents + planner.MIXING * 2) * one <= 64
    if len(layout) > 2:
        before, _ = allocated(lambda: study_merge(windows, layout))
        assert before >= (2 * latents - (layout[-1][1] - layout[-1][0])) * one > most


@pytest.mark.parametrize(("latents", "cap"), [(13, 12), (48, 30), (48, 8), (100, 7), (100, 5)])
def test_the_merge_into_one_tensor_is_the_study_s_to_the_bit(latents: int, cap: int) -> None:
    # The merge written into one tensor allocated for the shot (shot._merge) gives the latents
    # of the study's own writing, a concatenation at each join (study_merge, the merge until
    # 2026-10-12): the same bits, inf and negative zeros among the shared latents included, the
    # same dtype, on the CPU, dense in the windows' order of dimensions; from windows as the DiT
    # gives them, dense, and from windows in another layout, channel-major. The windows aren't
    # written to.
    # A NaN mixed is a NaN in both, at the same places; its own bits, sign and payload, are the
    # study's from dense windows, and follow the loop torch takes over the operands from others
    # (0x7FC0 here for the study's 0xFFFF at the first join of channel-major windows). No NaN
    # reaches a merge in a run: a window's output is checked before it is kept (run._sample).
    layout = window_layout(latents, cap)
    assert len(layout) > 1

    def bits(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.contiguous().view(torch.int16)

    for major in (False, True):
        windows = [contiguous(end - start, seed) for seed, (start, end) in enumerate(layout)]
        windows[0][-1, 0, 0, :4] = torch.tensor([math.nan, math.inf, -math.inf, -0.0])
        windows[1][0, 1, 1, :4] = torch.tensor([-0.0, math.nan, 1e-30, math.inf])
        windows[-1][1, 2, 3, 4] = -0.0
        if major:
            windows = [w.permute(3, 0, 1, 2).contiguous().permute(1, 2, 3, 0) for w in windows]
            assert not windows[0].is_contiguous()
        given = [bits(window).clone() for window in windows]
        ours, theirs = merge_windows(windows, layout), study_merge(windows, layout)
        assert ours.dtype == theirs.dtype == torch.bfloat16
        assert ours.device == theirs.device == torch.device("cpu")
        assert ours.shape == theirs.shape == (latents, 16, 24, 16)
        assert ours.stride() == theirs.stride() and ours.is_contiguous()
        nan = torch.isnan(theirs)
        assert torch.equal(torch.isnan(ours), nan) and int(nan.sum()) == 2
        assert torch.equal(bits(ours)[~nan], bits(theirs)[~nan])
        assert major or torch.equal(bits(ours), bits(theirs))
        assert all(torch.equal(bits(w), was) for w, was in zip(windows, given, strict=True))
        # The shared latents were mixed, not copied: neither window's.
        (_, first_end), (second_start, _) = layout[0], layout[1]
        shared = bits(ours[second_start:first_end])
        assert not torch.equal(shared, bits(windows[0][second_start:]))
    # One window is the merged latent itself.
    alone = contiguous(7)
    assert merge_windows([alone], [(0, 7)]) is alone


def test_the_merge_refuses_windows_that_aren_t_the_layout_s() -> None:
    # The merged latents are written into a tensor allocated for the layout (shot._merge):
    # with a window missing, the latents no window wrote would be whatever the memory held, and
    # a window of a single latent would be repeated over its place, where the concatenation
    # gave a short latent that the decode's frame count refused. So the windows are held to the
    # layout first, each to its length: one missing, one more, one of another length, the
    # first, a later one or a shot's only one, are refused, naming both.
    layout = window_layout(48, 8)
    windows = [contiguous(end - start, seed) for seed, (start, end) in enumerate(layout)]
    assert merge_windows(windows, layout).shape[0] == 48
    lengths = [end - start for start, end in layout]
    assert lengths == [8] * 6 + [7] * 2
    for given in (
        windows[:-1],
        windows[1:],
        [*windows, contiguous(7)],
        [*windows[:3], contiguous(1), *windows[4:]],
        [contiguous(1), *windows[1:]],
        [contiguous(7), *windows[1:]],
    ):
        said = f"windows of {[window.shape[0] for window in given]} latents for a layout of"
        with pytest.raises(ValueError, match=re.escape(f"{said} {lengths}")):
            merge_windows(given, layout)
    with pytest.raises(ValueError, match=re.escape("windows of [8] latents for a layout of [7]")):
        merge_windows([contiguous(8)], [(0, 7)])
    with pytest.raises(ValueError, match=re.escape("windows of [8] latents for a layout of [")):
        merge_windows(windows[:1], layout)
    # And the merge itself takes no fewer windows than its layout has, whoever calls it.
    with pytest.raises(ValueError, match=re.escape("zip() argument 2 is shorter")):
        shot._merge(windows[:-1], layout)  # pyright: ignore[reportPrivateUsage]


def test_a_window_s_file_allocates_the_window_alone(tmp_path: Path) -> None:
    # A window written to its file from its copy in system memory, and read back for the decode
    # (units.Units): torch allocates nothing to write it, and the tensor alone to read it.
    save = units._save  # pyright: ignore[reportPrivateUsage]
    load = units._load  # pyright: ignore[reportPrivateUsage]
    window, path = contiguous(8), tmp_path / "window_0000.pt"
    assert allocated(lambda: save(window, path)) == (0, 0)
    assert allocated(lambda: load(path)) == (window.nbytes, window.nbytes)


def test_the_encode_s_end_allocates_the_latent_beside_its_buffer(stand_in: model.Models) -> None:
    # An encode's end (model.to_latent): the latent's own memory beside the buffer it is made
    # from, as large, the planner's "encode" (planner.host_moments); what its slices allocate on
    # their way, 2 latents at a time at the most, is the device's in a run, on the CPU here. So
    # this holds the end to its slices too: the buffer taken whole to the device, its copy
    # there, the shifted one and the scaled one are 3 times the latent more, which would grow
    # there with the shot (160 latents' worth allocated for these 40, with the loop taken
    # whole).
    latents = 40
    means = contiguous(latents).permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    one = means.nbytes // latents
    most, left = allocated(lambda: model.to_latent(stand_in, means))
    counted = planner.host_moments(SMALL_PADDED, latents, None)
    assert counted["encode"] == means.nbytes + left == 2 * latents * one
    assert left <= most <= left + 8 * one


def test_the_tiled_encode_s_sums_are_divided_slice_by_slice(
    stand_in: model.Models, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A tiled encode's end before its latent is made (tiling.encode_tiled): the tiles' sums go
    # to the device to be divided by the summed weights a slice at a time, 2 latents then 1
    # (model.latent_slices), and come back to their place: nothing there grows with the shot.
    # The sums given whole, 40 latents', what the tiles' passes would have summed (their part
    # is tests/test_tiling.py's), and the latent's making left out (the test above): torch
    # allocates 3 latents' worth at the most, a slice's copy while the slice before's is let go,
    # and the divisor, on the CPU here, the device's in a run, where the sums taken whole are
    # 40.
    latents, frames = 40, 157
    height, width = SMALL_PADDED[0] // 8, SMALL_PADDED[1] // 8
    sums = contiguous(latents).permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    expected = sums.clone()
    one = sums.nbytes // latents
    count = torch.full((height, width), 2.0, dtype=model.COMPUTE_DTYPE)
    given: list[torch.Tensor] = []

    def summed(*arguments: object) -> tuple[torch.Tensor, torch.Tensor]:
        return sums, count

    def made(models: model.Models, mean: torch.Tensor) -> torch.Tensor:
        given.append(mean)
        return mean

    monkeypatch.setattr(tiling, "_means", summed)
    monkeypatch.setattr(model, "to_latent", made)
    sizes = model.encode_slices(stand_in, frames)
    fed = [torch.zeros((3, size, *SMALL_PADDED), dtype=model.COMPUTE_DTYPE) for size in sizes]
    tiles = Tiles(64, 16)
    assert len(tiling.grid((height, width), tiles)) > 1
    most, left = allocated(
        lambda: tiling.encode_tiled(stand_in, iter(fed), frames, tiles, Scratch(None, MEMORY))
    )
    assert len(given) == 1 and given[0] is sums and torch.equal(sums, expected / 2)
    assert model.latent_slices(stand_in, latents)[:2] == [2, 1]
    assert 2 * one <= most <= 3 * one + count.nbytes and left == 0


def test_a_tiled_decode_hands_its_frames_on_slice_by_slice(
    stand_in: model.Models, tmp_path: Path
) -> None:
    # A tiled decode's end (tiling.decode_tiled): once its last tile is in, each slice of the
    # sums goes to the device to be divided and handed on, 5 frames at the most, never the
    # scratch whole, which grows with the shot. The scratch in its file, mapped, torch
    # allocating nothing for it: what torch allocates at the most over the pass, on the CPU
    # here, the device's in a run, is then the same to the byte for shots of 21, 81 and 161
    # frames, under 11 frames' worth (a tile's slice as it is weighted and summed, the divisor,
    # a slice handed on). The scratch taken whole to the device after the last tile would be 161
    # frames' more for the longest, which no figure read as a call of the VAE starts can see
    # (tests/test_tiling.py's test on a card): no call starts after it.
    tiles, size = Tiles(64, 16), (11, 13)
    frame = 3 * 88 * 104 * 2
    assert len(tiling.grid(size, tiles)) == 4
    taken: list[int] = []
    for latents in (6, 21, 41):
        latent = latent_of(latents, size)
        frames: list[int] = []

        def decoded(latent: torch.Tensor = latent, frames: list[int] = frames) -> None:
            passed = tiling.decode_tiled(stand_in, latent, tiles, Scratch(tmp_path, 0))
            frames.extend(handed.shape[1] for handed in passed)

        most, left = allocated(decoded)
        assert sum(frames) == 4 * latents - 3 and max(frames) == 5 and left == 0
        assert not (tmp_path / tiling.SCRATCH).exists()
        taken.append(most)
    assert taken[0] == taken[1] == taken[2] and 5 * frame < taken[0] < 11 * frame


# The shots of the windows' test on a card: 960x540, padded to 960x576, 276,480 bytes a latent in
# bfloat16 (16 channels over 72 x 120 positions). One shot of ONE_WINDOW latents in one window,
# and one of WINDOWED latents, 389 frames, which a cap of ONE_WINDOW lays out in MANY windows of
# that length, sharing 2 latents with each neighbour.
CARD_TARGET = (540, 960)
ONE_WINDOW, MANY = 8, 16
WINDOWED = MANY * (ONE_WINDOW - 2) + 2
# What two measures of the device may differ by, and still be the same, in the bytes asked of
# torch's allocator (tests/test_tiling.py says why those: SETTLED): 1 MiB. The least a run could
# keep there of a shot from window to window is a window's output, 8 x 276,480 = 2,211,840
# bytes, twice this; the shot's noise or its latent whole, 98 x 276,480 = 27,095,040. Where
# torch doesn't give those bytes, the blocks its allocator handed out are compared instead,
# which the same tensors fill by megabytes more or less (up to 9,765,504 bytes over what was
# asked, in the VAE's tiled passes on the card): to 12 MiB then, under the noise or the latent
# alone.
ASKED_TOLERANCE = 1 << 20
BLOCKS_TOLERANCE = 12 << 20


@dataclass(frozen=True)
class OnDevice:
    """What the device holds over a shot's windows as the run makes them (on_the_device), in
    bytes, each figure by torch's count of the blocks its allocator handed out
    (memory_allocated) and by its count of the bytes asked of it (memory_stats'
    requested_bytes; None where torch doesn't give it): as each window's unit starts; at each
    window's peak, its output still there; and once the shot's windows are done. reset: the
    two counts' peaks right after each unit's reset of them, which are what is held then when
    the reset took."""

    starts: tuple[tuple[int, int | None], ...]
    peaks: tuple[tuple[int, int | None], ...]
    left: tuple[int, int | None]
    reset: tuple[tuple[int, int | None], ...]

    def doubt(self) -> str:
        """Why the bytes asked of the allocator can't be compared, the blocks then being; empty
        when they can: torch gives them at every figure, never over the blocks handed out, and
        their peak was brought back to what was asked as each unit started."""
        figures = (*self.starts, *self.peaks, self.left, *self.reset)
        if any(asked is None for _, asked in figures):
            return "torch gives no requested_bytes"
        if any(asked is not None and asked > blocks for blocks, asked in figures):
            return "the requested bytes were over the allocated ones"
        if any(reset[1] != start[1] for reset, start in zip(self.reset, self.starts, strict=True)):
            return "resetting the peaks left the requested one above what was requested then"
        return ""


def on_the_device(
    models: model.Models,
    latents: int,
    cap: int,
    monkeypatch: pytest.MonkeyPatch,
    directory: Path,
) -> OnDevice:
    """A shot of `latents` latents at CARD_TARGET through the run's own steps for its windows
    (run._sample over the units a run keeps, units.Units in `directory`, made here, its windows
    capped at `cap` latents), its encode's latent given, random, to the units, in an encode's
    layout, which the steps read back as a resumed run does: what the device holds over them
    (OnDevice). The figures are read where the run's own are: as a unit starts,
    before the run resets the peak it logs (run._started), and as the window's output is given
    to the units, where the run reads that peak."""
    from seedvr2x.runtime import run

    device = models.device
    started = run._started  # pyright: ignore[reportPrivateUsage]
    sample = run._sample  # pyright: ignore[reportPrivateUsage]
    starts: list[tuple[int, int | None]] = []
    peaks: list[tuple[int, int | None]] = []
    reset: list[tuple[int, int | None]] = []

    def asked(what: str = "current") -> int | None:
        return torch.cuda.memory_stats(device).get(f"requested_bytes.all.{what}")

    def starting(models: model.Models) -> float:
        torch.cuda.synchronize(device)
        starts.append((torch.cuda.memory_allocated(device), asked()))
        at = started(models)
        reset.append((torch.cuda.max_memory_allocated(device), asked("peak")))
        return at

    class Kept(Units):
        def save_window(self, run: int, window: int, sampled: torch.Tensor) -> None:
            torch.cuda.synchronize(device)
            peaks.append((torch.cuda.max_memory_allocated(device), asked("peak")))
            super().save_window(run, window, sampled)

    height, width = padded_size(CARD_TARGET)
    generator = torch.Generator().manual_seed(latents)
    values = torch.randn((16, latents, height // 8, width // 8), generator=generator)
    frames = 4 * latents - 3
    shots = [Shot(0, frames)]
    directory.mkdir()
    kept = Kept(directory, job(directory, shots, [OutputSegment("a", 0, frames)], cap))
    kept.save_latent(0, values.to(model.COMPUTE_DTYPE).permute(1, 2, 3, 0))
    del values
    # In seedvr2x's padding, its tail and its draw, on the CPU (run._sample's numz_padding,
    # numz_draw and numz_tail).
    job_ = (shots, CARD_TARGET, 42, Plan(cap), kept, None, None, False, False, False)
    with monkeypatch.context() as patch:
        patch.setattr(run, "_started", starting)
        sample(0, models, cast(Any, None), *job_)
    torch.cuda.synchronize(device)
    left = (torch.cuda.memory_allocated(device), asked())
    done = kept.windows_done(0)
    assert done == len(shot_layout(frames, cap)) == len(starts) == len(peaks) == len(reset)
    assert not (kept.shot_directory(0) / "latent.pt").exists()
    return OnDevice(tuple(starts), tuple(peaks), left, tuple(reset))


def test_the_card_s_test_reads_the_run_s_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # What the windows' test on a card reads the device by (on_the_device), on the CPU, over a
    # stand-in DiT and the device's counts replaced: the run's own steps make every window of
    # the shot, a figure is read as each starts, before the run's reset, and one as each is
    # kept, and the shot's latent is given up. The figures are whatever the counts say: here 3
    # more bytes held each time the device is asked, and a window's pass 100 over that.
    Dit(monkeypatch)
    count = iter(range(0, 10_000, 3))
    held: dict[str, int] = {"now": 0, "peak": 0}
    sample = model.sample

    def now(device: torch.device) -> int:
        held["now"] = next(count)
        return held["now"]

    def sampled(models: object, noise: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        held["peak"] = held["now"] + 100
        return sample(cast(model.Models, models), noise, cond)

    def reset(device: torch.device) -> None:
        held["peak"] = held["now"]

    def stats(device: torch.device) -> dict[str, int]:
        return {
            "requested_bytes.all.current": held["now"],
            "requested_bytes.all.peak": held["peak"],
        }

    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", now)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: held["peak"])
    monkeypatch.setattr(torch.cuda, "memory_stats", stats)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset)
    monkeypatch.setattr(model, "sample", sampled)
    entered: list[str] = []
    models = cast(model.Models, SimpleNamespace(device=torch.device("cpu"), enter=entered.append))
    read = on_the_device(models, WINDOWED, ONE_WINDOW, monkeypatch, tmp_path / "windowed")
    assert entered == ["dit"] * MANY
    assert [start for start, _ in read.starts] == list(range(0, 3 * MANY, 3))
    assert read.starts == read.reset and read.doubt() == ""
    assert read.peaks == tuple((start + 100, start + 100) for start, _ in read.starts)
    assert read.left == (3 * MANY, 3 * MANY)
    layout = shot_layout(4 * WINDOWED - 3, ONE_WINDOW)
    assert [end - start for start, end in layout] == [ONE_WINDOW] * MANY
    alone = on_the_device(models, ONE_WINDOW, ONE_WINDOW, monkeypatch, tmp_path / "alone")
    assert len(alone.peaks) == 1


@pytest.mark.gpu
def test_a_shot_s_windows_hold_the_device_as_one_window_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The run's own steps for a shot's windows, over the real DiT on a card: a shot of 98
    # latents in 16 windows of 8 against a shot of 8 latents in one window. Between two windows
    # the device holds what it held before the shot, and each window peaks where the one-window
    # shot's does: nothing of the shot waits there but the window being made, neither its
    # latent, its noise nor a window's output (DESIGN.md, Memory planner, Per-phase peaks). A
    # first one-window shot runs unmeasured: what a process's first window leaves on the device
    # for good, the tables the DiT's blocks cache for a window's size, is no shot's.
    models = model.load_models(found(DITS), found(VAES), torch.device("cuda:0"))
    try:
        on_the_device(models, ONE_WINDOW, ONE_WINDOW, monkeypatch, tmp_path / "first")
        alone = on_the_device(models, ONE_WINDOW, ONE_WINDOW, monkeypatch, tmp_path / "alone")
        windowed = on_the_device(models, WINDOWED, ONE_WINDOW, monkeypatch, tmp_path / "windowed")
        again = on_the_device(models, ONE_WINDOW, ONE_WINDOW, monkeypatch, tmp_path / "again")
    finally:
        models.runner.dit = models.runner.vae = None
        gc.collect()
        torch.cuda.empty_cache()
    layout = shot_layout(4 * WINDOWED - 3, ONE_WINDOW)
    assert [end - start for start, end in layout] == [ONE_WINDOW] * MANY
    assert (len(alone.peaks), len(windowed.peaks), len(again.peaks)) == (1, MANY, 1)
    # What the figures are compared on, for the whole run, and to what.
    doubts = [read.doubt() for read in (alone, windowed, again) if read.doubt()]
    on, column, tolerance = ("requested", 1, ASKED_TOLERANCE)
    if doubts:
        on, column, tolerance = (f"allocated ({doubts[0]})", 0, BLOCKS_TOLERANCE)

    def figures(read: tuple[tuple[int, int | None], ...]) -> list[int]:
        return [cast(int, figure[column]) for figure in read]

    # Said before anything is held of them: pytest shows the output of a test that fails, and of
    # one that passes under -rP or -s.
    for name, read in (("one window", alone), ("16 windows", windowed), ("one window", again)):
        print(
            f"{name}, compared on the {on} bytes: held as each window starts"
            f" {figures(read.starts)}, left {figures((read.left,))[0]}; peaks"
            f" {figures(read.peaks)}; allocated: starts {[blocks for blocks, _ in read.starts]},"
            f" peaks {[blocks for blocks, _ in read.peaks]}"
        )
    before, peak = figures(alone.starts)[0], figures(alone.peaks)[0]
    said = f"on the {on} bytes, {before} held before the one-window shot, its peak {peak}"
    # A window takes gigabytes over the weights, at this size.
    assert peak - before > 256 << 20, said
    for name, read in (("one window", alone), ("16 windows", windowed), ("one window", again)):
        # Between two windows, and after the last, what the device held before the shot.
        for window, start in enumerate(figures(read.starts)):
            held = f"{name}: {start} held as window {window} starts: {said}"
            assert abs(start - before) <= tolerance, held
        left = figures((read.left,))[0]
        assert abs(left - before) <= tolerance, f"{name}: {left} left: {said}"
        # Each window's peak, the one-window shot's.
        for window, top in enumerate(figures(read.peaks)):
            reached = f"{name}: window {window} peaks at {top}: {said}"
            assert abs(top - peak) <= tolerance, reached

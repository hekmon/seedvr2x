"""The vendored RoPE's tables cleared where the DiT's work on each shot starts and where it ends
(model.forget_rope_tables; DESIGN.md, Allocator, "The vendored RoPE's tables"), on the CPU: the
real vendored rotary module (NaRotaryEmbedding3d, whose get_axial_freqs rope.py:44 memoises per
window shape) asked for the tables of each window by a stand-in DiT, run through the library's own
entries (shot_pass.sample_windows, still.sampled, still.upscale, clip.upscale). A long run of many
shot lengths holds at most one shot's tables while its windows run and none after them, a raise
included, the entries call the clearing, and no output bit moves with it."""

from collections.abc import Generator, Iterator
from typing import Any

import numpy as np
import pytest
import torch
from stand_ins import cpu_models  # noqa: F401
from test_clip import LONG, SHORT, clip_of
from test_still import SEED, TARGET, pipeline_models, pixels_of
from torch import Tensor

from seedvr2x import clip, model, shot_pass, still
from seedvr2x.shots import shot_layout
from seedvr2x.vendor.models.dit_7b.rope import NaRotaryEmbedding3d

SIZE = (2, 3)  # a latent's rows and columns in the tests that call the passes directly


@pytest.fixture(autouse=True)
def clean_cache() -> Iterator[None]:
    """The cache is the class's, shared by the whole process: empty before and after each test,
    so that no test sees the tables of another (test_held's count its own)."""
    model.forget_rope_tables()
    yield
    model.forget_rope_tables()


def tables() -> Any:
    """The vendored memoised function itself, for cache_info."""
    return NaRotaryEmbedding3d.get_axial_freqs


def stand_in_dit(monkeypatch: pytest.MonkeyPatch) -> NaRotaryEmbedding3d:
    """A DiT of arithmetic that asks the real rotary module for the table of each window's shape
    (frames, rows, columns) and mixes its mean into the output, so that a table that came out
    other than the first time's would move a bit. Returns the module."""
    rope = NaRotaryEmbedding3d(dim=120)

    def condition(models: model.Models, noise: Tensor, latent: Tensor) -> Tensor:
        return torch.cat([latent, noise[..., :1]], dim=-1)

    def sample(models: model.Models, noise: Tensor, cond: Tensor) -> Tensor:
        freqs = rope.get_axial_freqs(*noise.shape[:3])
        return cond[..., :16] * 0.7 + noise * 0.3 + freqs.float().mean() * 0.01

    monkeypatch.setattr(model, "condition", condition)
    monkeypatch.setattr(model, "sample", sample)
    return rope


def latent_of(frames: int, rows: int, columns: int) -> Tensor:
    """The latent of a shot of `frames` frames, (T', rows, columns, 16) bfloat16 in the encode's
    channel-major memory (model.encode)."""
    count = (shot_pass.padded_length(frames) - 1) // 4 + 1
    generator = torch.Generator().manual_seed(frames)
    latent = torch.randn(16, count, rows, columns, generator=generator)
    return latent.permute(1, 2, 3, 0).to(torch.bfloat16)


def run_shot(
    models: model.Models, frames: int, window: int | None, rows: int = SIZE[0]
) -> tuple[list[Tensor], set[tuple[int, int, int]], list[int]]:
    """One shot's windows through shot_pass.sample_windows: their outputs, the shapes they asked
    the tables for, and the tables held after each window."""
    latent = latent_of(frames, rows, SIZE[1])
    layout = shot_layout(frames, window)
    held: list[int] = []
    outputs: list[Tensor] = []
    for output in shot_pass.sample_windows(models, latent, layout, frames):
        outputs.append(output)
        held.append(tables().cache_info().currsize)
    shapes = {(end - start, rows, SIZE[1]) for start, end in layout}
    return outputs, shapes, held


# 40 shots of 1 to 40 latents (frames 4n - 3), in windows of at most 5, 7 or none, with rows that
# change: the shapes asked over the run are many more than any one shot's.
SHOTS = [
    (4 * latents - 3, window, 2 + latents % 11) for latents in range(1, 41) for window in (5, 7)
]


def test_a_long_run_of_many_shot_lengths_holds_at_most_one_shots_tables(
    cpu_models: model.Models,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stand_in_dit(monkeypatch)
    models = pipeline_models(cpu_models)
    asked: set[tuple[int, int, int]] = set()
    for frames, window, rows in SHOTS:
        _, shapes, held = run_shot(models, frames, window, rows)
        asked |= shapes
        # Nothing of the shots before is still held, and the shot's own windows share their
        # shapes: a table per distinct shape, no more; and none once its windows are done.
        assert max(held) <= len(shapes)
        assert held[-1] == len(shapes)
        assert tables().cache_info().currsize == 0
    assert len(asked) > 40  # the run asked for many more than one shot holds


def test_without_the_clearing_the_same_run_grows_past_one_shot(
    cpu_models: model.Models,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The control: the test above is not vacuous. The cache, unbounded by shots, holds every
    # shape asked, up to its 128.
    stand_in_dit(monkeypatch)
    monkeypatch.setattr(model, "forget_rope_tables", lambda: None)
    models = pipeline_models(cpu_models)
    most, asked = 0, set[tuple[int, int, int]]()
    for frames, window, rows in SHOTS:
        _, shapes, held = run_shot(models, frames, window, rows)
        asked |= shapes
        most = max(most, max(held))
    assert most == len(asked) > 40


def test_a_picture_run_step_by_step_holds_one_table_while_it_samples_and_none_after(
    cpu_models: model.Models,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The image server calls still.sampled for each picture: one window, one shape, its table
    # made in the window and gone before the picture's decode.
    rope = stand_in_dit(monkeypatch)
    models = pipeline_models(cpu_models)
    sample = model.sample
    during: list[int] = []

    def counted(models: model.Models, noise: Tensor, cond: Tensor) -> Tensor:
        output = sample(models, noise, cond)
        during.append(tables().cache_info().currsize)
        return output

    monkeypatch.setattr(model, "sample", counted)
    for rows in range(1, 12):
        rope.get_axial_freqs(9, rows, 9)  # a table of some other window, held before
        latent = latent_of(1, rows, 3)
        still.sampled(models, latent, SEED)
        assert during[-1] == 1
        assert tables().cache_info().currsize == 0


def test_the_tables_are_gone_once_a_shots_windows_end_however_they_end(
    cpu_models: model.Models,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The DiT's work on a shot ends with its generator: exhausted (a for loop), closed before its
    # last window (clip.py closes it in a finally, a step's stop included), dropped (run.py's del,
    # the calibration's one window taken by next), or left by a raise in a window; the tables are
    # cleared at each, before any VAE pass.
    rope = stand_in_dit(monkeypatch)
    models = pipeline_models(cpu_models)
    layout = shot_layout(LONG, 5)
    assert len(layout) == 3

    def windows() -> Iterator[Tensor]:
        return shot_pass.sample_windows(models, latent_of(LONG, *SIZE), layout, 1)

    for _ in windows():
        assert tables().cache_info().currsize > 0
    assert tables().cache_info().currsize == 0
    started = windows()
    assert isinstance(started, Generator)
    next(started)
    assert tables().cache_info().currsize > 0
    started.close()
    assert tables().cache_info().currsize == 0
    started = windows()
    next(started)
    next(started)
    del started
    assert tables().cache_info().currsize == 0
    next(windows())
    assert tables().cache_info().currsize == 0
    sample = model.sample
    calls: list[int] = []

    def raising(models: model.Models, noise: Tensor, cond: Tensor) -> Tensor:
        output = sample(models, noise, cond)
        calls.append(tables().cache_info().currsize)
        if len(calls) == 2:
            raise RuntimeError("the second window fails")
        return output

    monkeypatch.setattr(model, "sample", raising)
    with pytest.raises(RuntimeError, match="the second window fails"):
        for _ in windows():
            pass
    assert calls[-1] > 0 and tables().cache_info().currsize == 0
    # A picture's window that raises leaves none either.
    calls.clear()
    rope.get_axial_freqs(9, 9, 9)

    def raises(models: model.Models, noise: Tensor, cond: Tensor) -> Tensor:
        sample(models, noise, cond)
        raise RuntimeError("the picture's window fails")

    monkeypatch.setattr(model, "sample", raises)
    with pytest.raises(RuntimeError, match="the picture's window fails"):
        still.sampled(models, latent_of(1, *SIZE), SEED)
    assert tables().cache_info().currsize == 0


def test_every_entry_clears_the_tables_once_before_a_shots_first_window_and_once_after_its_last(
    cpu_models: model.Models,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rope = stand_in_dit(monkeypatch)
    models = pipeline_models(cpu_models)
    cleared: list[int] = []
    clear = model.forget_rope_tables

    def recorded() -> None:
        cleared.append(tables().cache_info().currsize)
        clear()

    monkeypatch.setattr(model, "forget_rope_tables", recorded)
    # A shot in three windows: cleared when the first is asked for and not before, then once its
    # last is done, the shot's tables with it.
    layout = shot_layout(LONG, 5)
    assert len(layout) == 3
    shapes = len({e - s for s, e in layout})
    windows = shot_pass.sample_windows(models, latent_of(LONG, *SIZE), layout, 1)
    assert cleared == []
    next(windows)
    next(windows)
    next(windows)
    assert cleared == [0]  # nothing held before it
    assert tables().cache_info().currsize == shapes
    with pytest.raises(StopIteration):
        next(windows)
    assert cleared == [0, shapes]
    # Another shot: what was held before it is dropped at its start, its own at its end.
    cleared.clear()
    rope.get_axial_freqs(9, 9, 9)
    list(shot_pass.sample_windows(models, latent_of(SHORT, *SIZE), shot_layout(SHORT, 5), 1))
    assert cleared == [1, len({e - s for s, e in shot_layout(SHORT, 5)})]
    # A picture's DiT step, and the whole picture and clip through their own entries.
    cleared.clear()
    still.sampled(models, latent_of(1, *SIZE), SEED)
    assert cleared == [0, 1]
    cleared.clear()
    still.upscale(
        models, pixels_of(), TARGET, SEED, encode_tiles=None, decode_tiles=None, correct=False
    )
    assert cleared == [0, 1]
    cleared.clear()
    clip.upscale(
        models,
        clip_of(LONG),
        TARGET,
        SEED,
        window=5,
        encode_tiles=None,
        decode_tiles=None,
        correct=False,
    )
    # A clip is one shot: its three windows share the clearing before them and the one after.
    assert cleared == [0, shapes]


def test_clearing_moves_no_bit(
    cpu_models: model.Models,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The tables are made again from the module's buffer to the same bits, so a pass that clears
    # and one that never does give the same output, for a picture, a clip and a shot's windows.
    stand_in_dit(monkeypatch)
    models = pipeline_models(cpu_models)

    def passes() -> tuple[Any, ...]:
        picture = still.upscale(
            models, pixels_of(), TARGET, SEED, encode_tiles=None, decode_tiles=None, correct=True
        )
        video = clip.upscale(
            models,
            clip_of(LONG),
            TARGET,
            SEED,
            window=5,
            encode_tiles=None,
            decode_tiles=None,
            correct=True,
        )
        # Several shots after one another, so that the tables are cleared between them.
        shots = [run_shot(models, frames, 5)[0] for frames in (LONG, SHORT, LONG)]
        return picture, video, shots

    cleared = passes()
    model.forget_rope_tables()
    monkeypatch.setattr(model, "forget_rope_tables", lambda: None)
    kept = passes()
    assert np.array_equal(cleared[0], kept[0]) and cleared[0].shape == (*TARGET, 3)
    assert np.array_equal(cleared[1], kept[1]) and cleared[1].shape[1:3] == TARGET
    for a, b in zip(cleared[2], kept[2], strict=True):
        assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))

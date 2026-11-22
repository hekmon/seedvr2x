"""The frees on an error's way out on a card a wait was abandoned on (model.unless_abandoned;
DESIGN.md, Allocator, "A stall at `deghost`, and the stop's cleaning", decided on 2026-11-22), on
the CPU: a wait in a pass that never ends, past its bound, ended by a stop, marks the card
abandoned (waiting.abandoned), and nothing on the way out polls the card again or frees on it,
as on a card a CUDA error failed: the RoPE tables (shot_pass.sample_windows, still.sampled), the
VAE's causal memory (model.encode_sliced, decode_sliced), the decode's slices (shot_pass._slices)
and the frames the error went through (clip.upscale). A stop before the wait's bound leaves the
card answering, and the way out polls and frees as before.

The card's poll is model.synchronize replaced (Stall): from where a test arms it, the next poll,
or the DiT's forward (model.sample, whose poll is model._paced's on a card), is a wait on an event
that never completes, its unit named with an estimate, on a clock whose sleep advances it and
raises the stop at a time the test chooses, before the bound or past it. The models are
test_clip.py's: cpu_models' VAE in bfloat16 and test_still.py's stand-in DiT."""

import logging
from typing import Any

import numpy as np
import pytest
import torch
from stand_ins import cpu_models  # noqa: F401
from test_clip import LONG, Watched, clip_of
from test_still import TARGET, TILES, pipeline_models, pixels_of, stand_in_dit

from seedvr2x import clip, memory, model, still, waiting

ILLEGAL = "CUDA error: an illegal memory access was encountered"

# A unit's estimate, whose bound is waiting.FLOOR, 60 s.
ESTIMATE = 1.0


class Stop(BaseException):
    """A stop at once, raised in the wait's sleep as a signal's handler raises it there."""


class Stall:
    """The card's poll and the DiT's forward, replaced: once armed, the next of either waits on an
    event that never completes (waiting.wait), polled every second on a clock that raises Stop
    at `stop_at` seconds, past the bound (60 s) or before it, or raises a CUDA error at once
    (`how` "failed"). Every poll after that is counted, none waiting."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, how: str) -> None:
        self.how = how
        self.armed = False
        self.stalled = False
        self.after = 0  # the polls made once the stall has ended
        self.now = 0.0
        self.forgotten = 0  # the RoPE tables cleared
        monkeypatch.setattr(waiting, "monotonic", lambda: self.now)
        monkeypatch.setattr(waiting, "sleep", self.sleep)
        monkeypatch.setattr(waiting, "_stacks", lambda: None)
        monkeypatch.setattr(waiting, "_ended", None)
        monkeypatch.setattr(waiting, "_running", {})
        monkeypatch.setattr(memory, "activity", lambda index=0: None)
        sample = model.sample
        forget = model.forget_rope_tables

        def synchronize(device: torch.device) -> None:
            self.poll()

        def sampled(*args: Any) -> torch.Tensor:
            self.poll()
            return sample(*args)

        def forgetting() -> None:
            self.forgotten += 1
            forget()

        monkeypatch.setattr(model, "synchronize", synchronize)
        monkeypatch.setattr(model, "sample", sampled)
        monkeypatch.setattr(model, "forget_rope_tables", forgetting)

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        if self.now >= (100.0 if self.how == "abandoned" else 30.0):
            raise Stop

    def poll(self) -> None:
        if self.stalled:
            self.after += 1
            return
        if not self.armed:
            return
        self.stalled = True
        if self.how == "failed":
            with model.marking(torch.device("cpu")):
                raise RuntimeError(ILLEGAL)
        with waiting.named("the stalled unit", ESTIMATE):
            waiting.wait(lambda: False, 1.0)

    def arm(self, *args: Any) -> None:
        self.armed = True


WAYS = ["abandoned", "answering", "failed"]


def raised(how: str) -> type[BaseException]:
    return RuntimeError if how == "failed" else Stop


def memories(models: model.Models) -> list[Any]:
    return [
        module.memory
        for module in models.runner.vae.modules()
        if getattr(module, "memory", None) is not None
    ]


def held_as_it_was(stall: Stall, how: str) -> bool:
    """Whether the way out left the card alone: no poll after the stall, as on a failed card,
    on a card abandoned or failed; else whether it polled, as before."""
    if how == "answering":
        assert not waiting.abandoned(0)
        return stall.after >= 1
    assert waiting.abandoned(0) and (how != "failed" or model.failed(torch.device("cpu")))
    return stall.after == 0


@pytest.fixture
def clip_models(cpu_models: model.Models) -> model.Models:  # noqa: F811
    """test_clip.py's models: cpu_models' VAE in bfloat16, the DiT stood in."""
    made = pipeline_models(cpu_models)
    made.runner.dit = torch.nn.ModuleList()  # pyright: ignore[reportAttributeAccessIssue]
    made.text_pos = made.text_neg = torch.zeros(2)  # pyright: ignore[reportAttributeAccessIssue]
    return made


@pytest.fixture(autouse=True)
def lines(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)


# ---- unless_abandoned ---------------------------------------------------------------------------


@pytest.mark.parametrize("card", ["answering", "abandoned", "failed", "another abandoned"])
def test_a_free_on_an_errors_way_out_polls_and_frees_only_a_card_that_answers(
    monkeypatch: pytest.MonkeyPatch, card: str
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(model, "synchronize", lambda device: calls.append("poll"))
    if card == "abandoned":
        waiting._abandoned.add(0)  # pyright: ignore[reportPrivateUsage]
    if card == "another abandoned":
        waiting._abandoned.add(1)  # pyright: ignore[reportPrivateUsage]
    if card == "failed":
        waiting.fail(0, RuntimeError(ILLEGAL))
    model.unless_abandoned(torch.device("cpu"), lambda: calls.append("free"))
    answers = card in ("answering", "another abandoned")
    assert calls == (["poll", "free"] if answers else [])


@pytest.mark.parametrize("how", WAYS)
def test_a_poll_of_the_free_that_a_stop_ends_frees_unless_it_was_past_its_bound(
    monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    # The free's own poll stalls: past its bound, the stop leaves the card abandoned, and nothing
    # is freed on it; before its bound, the free is made, then the stop goes on, as before; a CUDA
    # error there marks the card failed, frees nothing and is not raised (the error being raised
    # is the caller's, on whose way out this is).
    stall = Stall(monkeypatch, how)
    stall.arm()
    freed: list[int] = []
    if how == "failed":
        model.unless_abandoned(torch.device("cpu"), lambda: freed.append(1))
        assert model.failed(torch.device("cpu"))
    else:
        with pytest.raises(Stop):
            model.unless_abandoned(torch.device("cpu"), lambda: freed.append(1))
    assert waiting.abandoned(0) == (how != "answering")
    assert freed == ([1] if how == "answering" else [])


# ---- The passes' frees ------------------------------------------------------------------------


@pytest.mark.parametrize("how", WAYS)
def test_a_window_stalled_in_the_dits_forward_leaves_its_tables_and_polls_no_more(
    clip_models: model.Models,
    monkeypatch: pytest.MonkeyPatch,
    how: str,
) -> None:
    # The night's stall (the RTX 5090 under WSL2, 2026-11-22): the DiT's forward of the clip's
    # second window waits for ever, a stop ends the wait past its bound, and the windows'
    # generator's `finally` cleared the RoPE tables behind a poll that waited again. Now nothing
    # is polled or cleared on the way out but where the card answers.
    stand_in_dit(monkeypatch)
    stall = Stall(monkeypatch, how)

    def step(each: str, done: int, total: int) -> None:
        if each == "window" and done == 1:
            stall.arm()

    with pytest.raises(raised(how)):
        clip.upscale(
            *(clip_models, clip_of(LONG), TARGET, 5),
            window=5,
            encode_tiles=None,
            decode_tiles=None,
            correct=False,
            step=step,
        )
    assert stall.stalled and held_as_it_was(stall, how)
    # Once as the shot's first window is asked for, again on the way out where the card answers.
    assert stall.forgotten == (2 if how == "answering" else 1)


@pytest.mark.parametrize("how", WAYS)
@pytest.mark.parametrize("tiled", [False, True])
def test_a_decode_stalled_in_its_poll_is_left_open_holding_what_it_holds(
    clip_models: model.Models,
    monkeypatch: pytest.MonkeyPatch,
    how: str,
    tiled: bool,
) -> None:
    # Untiled, the stall at the poll of the decode's second slice, the decode not closed; tiled,
    # inside its second tile's pass, which the stop leaves through the tiled decode, finished by
    # it: the tile's causal memory and the frames the stop went through kept, as on a failed card;
    # where the card answers, closed with nothing alive, as before.
    stand_in_dit(monkeypatch)
    stall = Stall(monkeypatch, how)
    watched = Watched(monkeypatch)

    def step(each: str, done: int, total: int) -> None:
        if each == "decode" and done >= 1 and not tiled:
            stall.arm()

    def tile(each: str, done: int, total: int) -> None:
        if each == "decode" and done == 1:
            stall.arm()

    with pytest.raises(raised(how)) as caught:
        clip.upscale(
            *(clip_models, clip_of(LONG), TARGET, 5),
            window=5,
            encode_tiles=None,
            decode_tiles=TILES if tiled else None,
            correct=True,
            step=step,
            tile=tile,
        )
    assert stall.stalled and held_as_it_was(stall, how)
    if how == "answering":
        assert watched.closed() and memories(clip_models) == []
    else:
        assert tiled or not watched.closed(), "a decode closed on a card that has not answered"
        assert memories(clip_models), "the causal memory of the decode is kept"
        frames = []
        traceback = caught.value.__traceback__
        while traceback is not None:
            frames.append(traceback.tb_frame)
            traceback = traceback.tb_next
        assert any(frame.f_locals for frame in frames[1:]), "the frames' locals are kept"
    del caught


@pytest.mark.parametrize("how", WAYS)
@pytest.mark.parametrize("tiled", [False, True])
def test_an_encode_stalled_in_its_poll_keeps_its_causal_memory(
    clip_models: model.Models,
    monkeypatch: pytest.MonkeyPatch,
    how: str,
    tiled: bool,
) -> None:
    # The encode's first poll inside a pass of the VAE (its first slice's copy to system memory,
    # or one inside its second tile's pass): its causal memory left as it is on a card that has
    # not answered, cleared where it answers.
    stand_in_dit(monkeypatch)
    stall = Stall(monkeypatch, how)
    if not tiled:
        stall.arm()

    def tile(each: str, done: int, total: int) -> None:
        if each == "encode" and done == 1:
            stall.arm()

    with pytest.raises(raised(how)):
        clip.upscale(
            *(clip_models, clip_of(LONG), TARGET, 5),
            window=5,
            encode_tiles=TILES if tiled else None,
            decode_tiles=None,
            correct=False,
            tile=tile,
        )
    assert stall.stalled and held_as_it_was(stall, how)
    assert bool(memories(clip_models)) == (how != "answering")


@pytest.mark.parametrize("how", WAYS)
@pytest.mark.parametrize("where", ["window", "decode"])
def test_a_picture_stalled_in_its_window_or_its_decode_polls_no_more(
    clip_models: model.Models,
    monkeypatch: pytest.MonkeyPatch,
    how: str,
    where: str,
) -> None:
    # still.sampled's tables and still.decoded's slices (tiled: the stall inside its second
    # tile's pass), as the clip's.
    stand_in_dit(monkeypatch)
    stall = Stall(monkeypatch, how)
    latent = still.encoded(clip_models, pixels_of(), TARGET, 5, tiles=None)
    if where == "window":
        stall.arm()
        with pytest.raises(raised(how)):
            still.sampled(clip_models, latent, 5)
        assert stall.forgotten == (2 if how == "answering" else 1)
    else:
        out = still.sampled(clip_models, latent, 5)
        watched = Watched(monkeypatch)

        def tile(done: int, total: int) -> None:
            if done == 1:
                stall.arm()

        with pytest.raises(raised(how)):
            still.decoded(
                clip_models, out, pixels_of(), TARGET, tiles=TILES, correct=False, tile=tile
            )
        # The stop went through the tiled decode, which it finished; a picture's tiles, one latent
        # each, are a slice each, with no causal memory to keep (model.decode_sliced).
        assert watched.closed() and memories(clip_models) == []
    assert stall.stalled and held_as_it_was(stall, how)


def test_the_passes_on_a_card_that_answers_are_as_before(
    clip_models: model.Models,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Nothing armed: the clip's output the same with the polls replaced as without.
    stand_in_dit(monkeypatch)
    before = clip.upscale(
        *(clip_models, clip_of(LONG), TARGET, 5),
        window=5,
        encode_tiles=TILES,
        decode_tiles=TILES,
        correct=True,
    )
    Stall(monkeypatch, "abandoned")
    after = clip.upscale(
        *(clip_models, clip_of(LONG), TARGET, 5),
        window=5,
        encode_tiles=TILES,
        decode_tiles=TILES,
        correct=True,
    )
    assert np.array_equal(before, after) and not waiting.abandoned(0)

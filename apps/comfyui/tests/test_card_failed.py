"""A CUDA error other than out of memory on the nodes (comfyui/models.py's session,
words.card_failed; seedvr2x/waiting.py's fail): the card is marked before anything is freed, the
models are held and the error kept with its frames, nothing given back to the card or emptied from
torch's cache, the error's text on the node, in the console and in ComfyUI's error with the way
out (restart ComfyUI), and every later run refused at its start with nothing read. Under
cudaMallocAsync a free on such a card reports the error again inside torch's deleter, which aborts
the process (the nodes on Windows, 2026-11-07). Out of memory frees as before."""

import gc
import logging
import weakref
from typing import Any

import comfy.model_management as management
import pytest
import server
import torch
from comfyui import console, models, plans, progress, words
from standins import RESOLUTION, StandIn, handle
from standins import upscale_video as run_video
from test_nodes import pictures

from seedvr2x import model, waiting

ILLEGAL = "CUDA error: an illegal memory access was encountered"


def failing_window(monkeypatch: pytest.MonkeyPatch, error: BaseException) -> dict[str, Any]:
    """model.sample raising `error` at its second window; torch's cache emptied so far counted
    then ("empties"), and the stand-in it replaced kept ("sample") for the runs after."""
    real = model.sample
    calls: list[int] = []
    happened: dict[str, Any] = {"sample": real}

    def sample(*args: Any) -> Any:
        calls.append(1)
        if len(calls) == 2:
            happened["empties"] = len(management.empties)
            raise error
        return real(*args)

    monkeypatch.setattr(model, "sample", sample)
    return happened


@pytest.mark.parametrize("keep", [True, False])
def test_a_cuda_error_holds_everything_and_says_restart_comfyui(
    machine: StandIn,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    keep: bool,
) -> None:
    caplog.set_level(logging.INFO)
    happened = failing_window(monkeypatch, RuntimeError(f"{ILLEGAL}\nmore of torch's words"))
    with pytest.raises(words.Refused) as refused:
        run_video(
            handle(tmp_path, keep=keep), plans.Settings(window=5), pictures(33), RESOLUTION, "9"
        )
    text = str(refused.value)
    assert f"the graphics card failed: {ILLEGAL}" in text and "restart ComfyUI" in text
    assert isinstance(refused.value.__cause__, RuntimeError)
    assert waiting.failed(0) == ILLEGAL and waiting.abandoned(0)
    # Nothing freed: the error kept with its frames, the models held, torch's cache left alone.
    kept = models._failures  # pyright: ignore[reportPrivateUsage]
    assert len(kept) == 1 and kept[0] is refused.value.__cause__
    assert kept[0].__traceback__ is not None
    assert len(management.empties) == happened["empties"]
    loaded = machine.made[0]
    alive = weakref.ref(loaded)
    machine.made.clear()
    del loaded
    gc.collect()
    assert alive() is not None
    # Said on the node and in ComfyUI's console.
    instance = server.PromptServer.instance
    assert instance is not None and instance.texts[-1][0] == progress.FAILED
    assert any(r.levelno == logging.ERROR and ILLEGAL in r.getMessage() for r in caplog.records)
    assert progress.FAILED == "GPU failed: restart ComfyUI"
    # The error is in the console once, at ERROR: not again as a refusal at WARNING.
    assert not any("refused" in r.getMessage() for r in caplog.records)
    # The run's generators are left open on a failed card, in the frames the error keeps: the
    # process is for restarting, so nothing here lets them go.
    monkeypatch.setattr(model, "sample", happened["sample"])
    del refused, kept
    models.reset()


@pytest.mark.parametrize("plan_only", [False, True])
def test_every_later_run_is_refused_at_its_start_with_nothing_read(
    machine: StandIn,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    plan_only: bool,
) -> None:
    h = handle(tmp_path, keep=True)
    happened = failing_window(monkeypatch, RuntimeError(ILLEGAL))
    with pytest.raises(words.Refused):
        run_video(h, plans.Settings(window=5), pictures(33), RESOLUTION, "9")
    monkeypatch.setattr(model, "sample", happened["sample"])
    reads: list[int] = []
    free = machine.free
    machine.free = lambda: reads.append(1) or free()  # type: ignore[method-assign]
    loads, emptied = len(machine.loads), len(management.empties)
    with pytest.raises(words.Refused, match="failed during an earlier seedvr2x run") as refused:
        run_video(h, plans.Settings(window=5), pictures(5), RESOLUTION, "9", plan_only=plan_only)
    assert ILLEGAL in str(refused.value) and "restart ComfyUI" in str(refused.value)
    assert reads == [] and len(machine.loads) == loads and len(management.empties) == emptied
    # The node says the card failed in one line, and the refusal's text is in the console once.
    instance = server.PromptServer.instance
    assert instance is not None and instance.texts[-1][0] == progress.FAILED
    said = [r for r in caplog.records if "failed during an earlier" in r.getMessage()]
    assert len(said) == 1 and said[0].levelno == logging.WARNING
    assert said[0].getMessage().removeprefix(console.PREFIX).startswith("refused: ")
    models.reset()


@pytest.mark.parametrize("keep", [True, False])
def test_out_of_memory_frees_as_before_and_marks_nothing(
    machine: StandIn, tmp_path: Any, monkeypatch: pytest.MonkeyPatch, keep: bool
) -> None:
    h = handle(tmp_path, keep=keep)
    happened = failing_window(monkeypatch, torch.OutOfMemoryError("CUDA out of memory"))
    with pytest.raises(torch.OutOfMemoryError):
        run_video(h, plans.Settings(window=5), pictures(33), RESOLUTION, "9")
    assert waiting.failed(0) is None and not waiting.abandoned(0)
    assert models._failures == [] and models._left_on_card == []  # pyright: ignore[reportPrivateUsage]
    loaded = machine.made[0]
    assert not loaded.on_card  # type: ignore[attr-defined]
    monkeypatch.setattr(model, "sample", happened["sample"])
    run_video(h, plans.Settings(window=5), pictures(5), RESOLUTION, "9")
    models.reset()


def test_a_cuda_error_at_the_run_s_end_poll_holds_the_models_not_kept(
    machine: StandIn, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A run's normal end polls the card before models not kept are dropped (models._session): a
    # pending CUDA error comes back there, marks the card, and the models are held, not freed.
    h = handle(tmp_path, keep=False)
    polls: list[int] = []
    monkeypatch.setattr(model, "synchronize", lambda device: polls.append(1))
    run_video(h, plans.Settings(window=5), pictures(5), RESOLUTION, "9")
    last, polls[:] = len(polls), []

    emptied: list[int] = []

    def synchronize(device: Any) -> None:
        polls.append(1)
        if len(polls) >= last:
            emptied.append(len(management.empties))
            raise RuntimeError(ILLEGAL)

    monkeypatch.setattr(model, "synchronize", synchronize)
    with pytest.raises(words.Refused, match="the graphics card failed"):
        run_video(h, plans.Settings(window=5), pictures(5), RESOLUTION, "9")
    assert waiting.failed(0) == ILLEGAL
    assert len(models._left_on_card) == 1  # pyright: ignore[reportPrivateUsage]
    assert len(management.empties) == emptied[0]
    models.reset()

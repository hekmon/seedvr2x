"""The shot detector's close (seedvr2x/detector.py, Detector.close) on a card, on the CPU: the
detector is built on the CPU with the untrained model's weights and its device then said to be
CUDA, torch's CUDA calls replaced by stand-ins that record what is called. What is held: the
wait for the card's work is the poll (model.synchronize, waiting.wait), never torch.cuda.
synchronize, so that a stop or an app's check (ComfyUI's Cancel) reaches it; a card a CUDA error
failed, found so or met by the poll, has nothing freed and the model kept; an exception from the
wait that is no CUDA failure keeps the model too and goes through; a healthy close frees as
before. The memory given back and torch's settings on the GPU are test_detector.py's (-m gpu)."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from test_detector import untrained_weights  # pyright: ignore[reportUnknownVariableType]

from seedvr2x import detector, waiting
from seedvr2x.detector import Detector

ILLEGAL = "CUDA error: an illegal memory access was encountered"


class Stop(BaseException):
    """What an app's check raises to end a wait: ComfyUI's Cancel, a stop."""


class Card:
    """torch's CUDA calls the close makes, replaced: each recorded by name, the event's query
    answering `done` (a list of answers, the last repeated) or raising `error`."""

    def __init__(self) -> None:
        self.called: list[str] = []
        self.answers: list[bool] = [True]
        self.error: BaseException | None = None

    def event(self) -> Any:
        card = self

        class Event:
            def record(self) -> None:
                card.called.append("record")

            def query(self) -> bool:
                card.called.append("query")
                if card.error is not None:
                    raise card.error
                return card.answers.pop(0) if len(card.answers) > 1 else card.answers[0]

        return Event()

    def says(self, name: str, result: Any = None) -> Any:
        def call(*args: Any, **kwargs: Any) -> Any:
            self.called.append(name)
            return result

        return call


@pytest.fixture
def card(monkeypatch: pytest.MonkeyPatch) -> Card:
    found = Card()
    monkeypatch.setattr(torch.cuda, "Event", found.event)
    monkeypatch.setattr(torch.cuda, "synchronize", found.says("torch.cuda.synchronize"))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", found.says("peak", 2**30))
    monkeypatch.setattr(torch.cuda, "memory_reserved", found.says("reserved", 2**20))
    monkeypatch.setattr(torch.cuda, "empty_cache", found.says("empty_cache"))
    monkeypatch.setattr(torch._C, "_cuda_clearCublasWorkspaces", found.says("cublas"))
    monkeypatch.setattr(waiting, "sleep", lambda seconds: None)
    monkeypatch.setattr(detector, "_left", [])
    return found


@pytest.fixture
def closing(tmp_path: Path, card: Card) -> Detector:
    """A detector whose device is CUDA's index 0, built on the CPU."""
    found = Detector(untrained_weights(tmp_path), torch.device("cpu"))
    found._device = SimpleNamespace(type="cuda", index=0)  # type: ignore[assignment]  # pyright: ignore[reportPrivateUsage]
    return found


def kept() -> list[torch.nn.Module]:
    return detector._left  # pyright: ignore[reportPrivateUsage]


def test_a_healthy_close_waits_in_the_poll_then_frees_as_before(
    closing: Detector, card: Card
) -> None:
    card.answers = [False, False, True]
    closing.close()
    # The event recorded, queried until done (the poll), and only then the memory given back;
    # torch.cuda.synchronize, which holds the main thread in C, is never called.
    assert card.called == [
        "record",
        "query",
        "query",
        "query",
        "peak",
        "cublas",
        "empty_cache",
        "reserved",
    ]
    assert kept() == [] and waiting.failed(0) is None
    with pytest.raises(ValueError, match="closed"):
        closing.push(None)  # type: ignore[arg-type]


def test_a_card_found_failed_keeps_the_model_and_frees_and_waits_for_nothing(
    closing: Detector, card: Card
) -> None:
    model = closing._model  # pyright: ignore[reportPrivateUsage]
    waiting.fail(0, RuntimeError(ILLEGAL))
    closing.close()
    assert card.called == []
    assert kept() == [model]
    closing.close()  # closed: nothing more
    assert card.called == [] and kept() == [model]


def test_a_cuda_error_met_by_the_poll_marks_the_card_and_keeps_the_model(
    closing: Detector, card: Card
) -> None:
    model = closing._model  # pyright: ignore[reportPrivateUsage]
    card.error = RuntimeError(f"{ILLEGAL}\nCUDA kernel errors might be asynchronously reported")
    closing.close()
    assert waiting.failed(0) == ILLEGAL and waiting.abandoned(0)
    assert kept() == [model]
    # Nothing was freed and torch's blocking wait was not made.
    assert card.called == ["record", "query"]


def test_out_of_memory_in_the_poll_is_no_failed_card_and_goes_through(
    closing: Detector, card: Card
) -> None:
    card.error = torch.OutOfMemoryError("CUDA out of memory")
    with pytest.raises(torch.OutOfMemoryError):
        closing.close()
    assert waiting.failed(0) is None
    # The close did not free: its model is kept, as for any exception that ends the wait.
    assert len(kept()) == 1 and "empty_cache" not in card.called


def test_a_check_that_raises_during_the_close_keeps_the_model_and_the_exception_goes_through(
    closing: Detector, card: Card
) -> None:
    model = closing._model  # pyright: ignore[reportPrivateUsage]
    card.answers = [False]  # the card never answers: only the check ends the wait

    def cancel() -> None:
        raise Stop

    with waiting.checking(cancel):
        with pytest.raises(Stop):
            closing.close()
    assert kept() == [model]
    assert waiting.failed(0) is None
    assert not {"peak", "cublas", "empty_cache", "torch.cuda.synchronize"} & set(card.called)
    closing.close()  # the detector is closed: the second close does nothing
    assert kept() == [model]


def test_the_cpu_close_makes_no_wait_on_a_card(tmp_path: Path, card: Card) -> None:
    found = Detector(untrained_weights(tmp_path), torch.device("cpu"))
    found.close()
    assert card.called == [] and kept() == []

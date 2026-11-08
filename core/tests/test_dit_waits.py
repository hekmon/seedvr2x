"""A DiT forward that waits on the card only in the poll (DESIGN.md, Allocator, "Out of the stop's
reach"): the vendored Cache's held values (vendor/common/cache.py), made once a forward after a
poll whatever `disable`, and model._paced, which holds the host to PACED blocks ahead of the card,
each wait in waiting.wait. On the CPU, the card's events and the poll replaced by stand-ins that
record their calls; the bits of a block under held values are tests/test_lean_attention.py's,
and the card's own syncs are counted on the GPU (impl-stop-2's census, torch's sync debug mode)."""

import threading
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from seedvr2x import model, waiting
from seedvr2x.vendor.common import cache as vendor_cache
from seedvr2x.vendor.common.cache import Cache


def missed() -> Any:
    """The calling thread's hook before a held value's first making, None for none."""
    return getattr(vendor_cache.hooks, "missed", None)


def test_a_held_value_is_made_once_whatever_disable() -> None:
    for disable in (True, False):
        cache = Cache(disable=disable)
        made: list[int] = []

        def make(made: list[int] = made) -> int:
            made.append(1)
            return 7

        for _ in range(3):
            assert cache.held("lengths", make) == 7
        assert made == [1]


def test_a_namespace_holds_its_own_values_in_the_forwards_cache() -> None:
    cache = Cache(disable=True)
    shifted, plain = cache.namespace("shifted"), cache.namespace("plain")
    assert shifted.held("win", lambda: "a") == "a"
    assert plain.held("win", lambda: "b") == "b"
    assert cache.namespace("shifted").held("win", lambda: "c") == "a"
    # The cache disabled makes its own values again every time, as the pipeline's forward did.
    assert cache("rope", lambda: "d") == "d" and cache("rope", lambda: "e") == "e"


def test_the_poll_comes_before_each_first_making_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(vendor_cache.hooks, "missed", lambda: calls.append("poll"), raising=False)
    cache = Cache(disable=True)
    for _ in range(36):  # the 7B's blocks: one making a key, the poll before it
        cache.held("all_len", lambda: calls.append("make") or 1)
        cache.namespace("win").held("vid_len", lambda: calls.append("make") or 2)
    assert calls == ["poll", "make", "poll", "make"]


class Events:
    """torch.cuda.Event's stand-in: each recorded in order, done once `finished` says so."""

    def __init__(self) -> None:
        self.recorded: list[Any] = []
        self.finished = True

    def __call__(self) -> Any:
        events = self

        class Event:
            def record(self) -> None:
                self.number = len(events.recorded)
                events.recorded.append(self)

            def query(self) -> bool:
                return events.finished

        return Event()


class Blocks(nn.Module):
    """A DiT's stand-in: `count` blocks run in order, each saying when it runs."""

    def __init__(self, count: int, said: list[str]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(nn.Identity() for _ in range(count))
        self.vid_out = nn.Identity()
        self.said = said

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for number, block in enumerate(self.blocks):
            self.said.append(f"block {number}")
            x = block(x)
        self.said.append("vid_out")
        return self.vid_out(x)


@pytest.fixture
def card(monkeypatch: pytest.MonkeyPatch) -> tuple[Events, list[str]]:
    events, said = Events(), []

    def wait(done: Any, poll: float, index: int = 0) -> None:
        # The event the wait is on, by the order it was recorded in, and the poll's period.
        owner = done.__self__
        said.append(f"wait {owner.number} every {poll}")
        assert done()

    monkeypatch.setattr(torch.cuda, "Event", events)
    monkeypatch.setattr(waiting, "wait", wait)
    return events, said


DEVICE: Any = SimpleNamespace(type="cuda", index=0)


def test_each_block_waits_in_the_poll_for_the_one_paced_before(
    card: tuple[Events, list[str]],
) -> None:
    events, said = card
    dit = Blocks(5, said)
    with model._paced(dit, DEVICE):  # pyright: ignore[reportPrivateUsage]
        dit(torch.zeros(1))
    assert model.PACED == 2
    # The card polled as the forward begins (event 0), an event recorded after each block (1 to
    # 5), the one two before waited for before each block from the third on, every 1 ms as
    # blockswap's waits, and the card polled before the output's patching (event 6) and once the
    # forward has ended (event 7).
    assert said == [
        "wait 0 every 0.0002",
        "block 0",
        "block 1",
        "block 2",
        "wait 1 every 0.001",
        "block 3",
        "wait 2 every 0.001",
        "block 4",
        "wait 3 every 0.001",
        "vid_out",
        "wait 6 every 0.0002",
        "wait 7 every 0.0002",
    ]
    assert len(events.recorded) == 8


def test_the_hooks_and_the_poll_go_with_the_forward(card: tuple[Events, list[str]]) -> None:
    events, said = card
    dit = Blocks(3, said)
    before = missed()
    with model._paced(dit, DEVICE):  # pyright: ignore[reportPrivateUsage]
        assert missed() is not None and missed() is not before
        # The first making's poll: an event of its own, polled finer than a block's.
        said.clear()
        vendor_cache.hooks.missed()
        assert said == [f"wait {len(events.recorded) - 1} every 0.0002"]
    assert missed() is before
    said.clear()
    dit(torch.zeros(1))
    assert said == ["block 0", "block 1", "block 2", "vid_out"]
    assert not dit.vid_out._forward_pre_hooks
    assert all(not block._forward_pre_hooks and not block._forward_hooks for block in dit.blocks)
    assert not dit._forward_hooks


def test_the_hook_is_the_forwards_threads_alone(card: tuple[Events, list[str]]) -> None:
    # Two DiTs' forwards on two threads poll their own cards: the hook is per thread.
    seen: list[Any] = []
    with model._paced(Blocks(1, []), DEVICE):  # pyright: ignore[reportPrivateUsage]
        other = threading.Thread(target=lambda: seen.append(missed()))
        other.start()
        other.join()
        assert missed() is not None
    assert seen == [None]


def test_a_forward_that_raises_leaves_no_hook(card: tuple[Events, list[str]]) -> None:
    _, said = card
    dit = Blocks(3, said)
    with pytest.raises(RuntimeError), model._paced(dit, DEVICE):  # pyright: ignore[reportPrivateUsage]
        raise RuntimeError("a window that failed")
    assert missed() is None
    assert all(not block._forward_pre_hooks and not block._forward_hooks for block in dit.blocks)


def test_nothing_is_paced_off_the_card(card: tuple[Events, list[str]]) -> None:
    events, said = card
    dit = Blocks(4, said)
    with model._paced(dit, torch.device("cpu")):  # pyright: ignore[reportPrivateUsage]
        assert missed() is None
        dit(torch.zeros(1))
    assert said == ["block 0", "block 1", "block 2", "block 3", "vid_out"]
    assert not events.recorded

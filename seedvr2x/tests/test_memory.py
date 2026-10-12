"""The driver that gives memory the card doesn't have (runtime/memory.py): the probe's reading
of the driver's answers, a stand-in in CUDA's place. The probe on a real driver, its ctypes
calls and its verdict: tests/test_placement.py's test_the_driver_is_asked (GPU)."""

import ctypes
import logging
import os
from typing import Any

import pytest
import torch

from seedvr2x.runtime import memory
from seedvr2x.runtime.memory import GIB, OUT_OF_MEMORY, OVER, SUCCESS, Driver

TOTAL = 32 * GIB


class Card:
    """A stand-in for libcuda: a card of TOTAL bytes with `free` of them free, whose driver
    refuses what it can't hold, or gives it (spills). Another process gives back `once` bytes
    between the probe's first reading and its request, and `frees` more at every request, as a
    card in use does under the probe."""

    def __init__(
        self, free: int, spills: bool, once: int = 0, frees: int = 0, failing: str | None = None
    ) -> None:
        self.free, self.spills, self.once, self.frees = free, spills, once, frees
        self.failing = failing
        self.held = 0
        self.calls: list[str] = []

    def call(self, name: str) -> int | None:
        self.calls.append(name)
        return 999 if name == self.failing else None

    def cuInit(self, flags: int) -> int:
        return self.call("cuInit") or SUCCESS

    def cuDeviceGet(self, device: Any, index: int) -> int:
        return self.call("cuDeviceGet") or (SUCCESS if index == 0 else 101)

    def cuCtxCreate_v2(self, context: Any, flags: int, device: Any) -> int:
        return self.call("cuCtxCreate") or SUCCESS

    def cuCtxDestroy_v2(self, context: Any) -> int:
        return self.call("cuCtxDestroy") or SUCCESS

    def cuMemGetInfo_v2(self, free: Any, total: Any) -> int:
        free._obj.value = max(self.free - self.held, 0)
        total._obj.value = TOTAL
        return self.call("cuMemGetInfo") or SUCCESS

    def cuMemAlloc_v2(self, pointer: Any, size: ctypes.c_size_t) -> int:
        failed = self.call("cuMemAlloc")
        if failed:
            return failed
        self.free += self.once + self.frees
        self.once = 0
        if size.value > self.free and not self.spills:
            return OUT_OF_MEMORY
        self.held = size.value
        pointer._obj.value = 0xC0DA
        return SUCCESS

    def cuMemFree_v2(self, pointer: ctypes.c_uint64) -> int:
        assert pointer.value == 0xC0DA
        self.held = 0
        return self.call("cuMemFree") or SUCCESS


def probe(monkeypatch: pytest.MonkeyPatch, card: Card) -> Driver | None:
    monkeypatch.setattr(memory.ctypes, "CDLL", lambda name: card)
    return memory.probe()


def test_a_driver_that_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    card = Card(free=30 * GIB, spills=False)
    found = probe(monkeypatch, card)
    assert found == Driver(False, 30 * GIB + OVER, 30 * GIB, TOTAL)
    assert found is not None and found.said() == (
        "the GPU's driver refuses what the card can't hold (30.25 GiB asked with 30.00 free:"
        " refused)"
    )
    # One request, nothing held, the probe's context gone.
    made = ["cuInit", "cuDeviceGet", "cuCtxCreate"]
    assert card.calls == [*made, "cuMemGetInfo", "cuMemAlloc", "cuCtxDestroy"]


def test_a_driver_that_gives_what_the_card_does_not_have(monkeypatch: pytest.MonkeyPatch) -> None:
    card = Card(free=30 * GIB, spills=True)
    found = probe(monkeypatch, card)
    assert found == Driver(True, 30 * GIB + OVER, 30 * GIB, TOTAL)
    assert found is not None and found.said().startswith(
        "the GPU's driver gives memory the card doesn't have (30.25 GiB asked with 30.00 free:"
        " given): system memory in its place"
    )
    # Freed at once, the free memory read again, then the context gone.
    assert card.calls[-3:] == ["cuMemFree", "cuMemGetInfo", "cuCtxDestroy"] and card.held == 0


@pytest.mark.parametrize("freed", [OVER, OVER + (44 << 20), 2 * OVER - 1, 2 * OVER, GIB])
def test_memory_freed_under_the_probe_is_no_spill(
    monkeypatch: pytest.MonkeyPatch, freed: int
) -> None:
    # Another process frees memory between the reading and the request: a driver that refuses
    # then gives it, the card having room. That is no spill, however little over the request's
    # extra was freed: the card has that much more free afterwards, and the question is asked
    # again, of the card as it now is.
    card = Card(free=20 * GIB, spills=False, once=freed)
    found = probe(monkeypatch, card)
    assert found == Driver(False, 20 * GIB + freed + OVER, 20 * GIB + freed, TOTAL)
    assert card.calls.count("cuMemAlloc") == 2 and card.held == 0


def test_under_the_request_s_extra_freed_changes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Less freed than the request asks beyond the free memory: a driver that refuses still
    # refuses, and one that spills is still seen to.
    refusing = Card(20 * GIB, spills=False, once=OVER - 1)
    assert probe(monkeypatch, refusing) == Driver(False, 20 * GIB + OVER, 20 * GIB, TOTAL)
    spilling = Card(20 * GIB, spills=True, once=OVER - 1)
    assert probe(monkeypatch, spilling) == Driver(True, 20 * GIB + OVER, 20 * GIB, TOTAL)


def test_a_card_that_keeps_freeing_gives_no_answer(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    card = Card(free=20 * GIB, spills=False, frees=GIB)
    assert probe(monkeypatch, card) is None
    assert card.calls.count("cuMemAlloc") == 2 and card.held == 0
    assert "whether the driver refuses isn't known" in caplog.text


@pytest.mark.parametrize(
    "failing", ["cuInit", "cuDeviceGet", "cuCtxCreate", "cuMemGetInfo", "cuMemAlloc"]
)
def test_a_driver_that_cannot_be_asked(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, failing: str
) -> None:
    card = Card(free=30 * GIB, spills=False, failing=failing)
    assert probe(monkeypatch, card) is None
    assert f"can't be asked whether it refuses: {failing}: 999" in caplog.text
    # A context made is destroyed, whatever failed after it.
    assert ("cuCtxDestroy" in card.calls) == (failing in ("cuMemGetInfo", "cuMemAlloc"))


def test_no_cuda_library(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def missing(name: str) -> Any:
        raise OSError(f"{name}: cannot open shared object file")

    monkeypatch.setattr(memory.ctypes, "CDLL", missing)
    caplog.set_level(logging.WARNING)
    assert memory.probe() is None
    assert "can't be asked whether it refuses: libcuda.so.1: cannot open" in caplog.text


def test_wsl_s_gpu_device(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    monkeypatch.undo()  # the tests' own stand-in for it (conftest.py)
    monkeypatch.setattr(memory, "PARAVIRTUAL", str(tmp_path / "dxg"))
    assert not memory.paravirtual()
    (tmp_path / "dxg").write_bytes(b"")
    assert memory.paravirtual()


def startup(monkeypatch: pytest.MonkeyPatch, card: Card | None) -> str:
    """memory.startup with WSL2's device there and `card` behind it, or without the device."""
    monkeypatch.setattr(memory, "paravirtual", lambda: card is not None)
    monkeypatch.setattr(memory.ctypes, "CDLL", lambda name: card)
    return memory.startup()


def test_a_linux_driver_is_asked_nothing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # No WSL2 device: the driver refuses, as Linux's does: not asked, and DESIGN.md's allocator.
    caplog.set_level(logging.INFO)
    asked: list[str] = []
    monkeypatch.setattr(memory, "probe", lambda: asked.append("probe"))
    assert startup(monkeypatch, None) == "backend:cudaMallocAsync"
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "backend:cudaMallocAsync"
    assert not asked and not memory.spills() and not caplog.messages
    assert memory.usable(30 * GIB) == 30 * GIB and memory.hold("cuda:0") is None
    memory.check("cuda:0")  # nothing held: nothing read, torch's stand-in none
    assert not caplog.messages


def test_a_driver_that_refuses_behind_wsl_changes_nothing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    assert startup(monkeypatch, Card(30 * GIB, spills=False)) == "backend:cudaMallocAsync"
    assert not memory.spills() and memory.usable(30 * GIB) == 30 * GIB
    assert [record.levelname for record in caplog.records] == ["INFO"]
    assert caplog.messages[0].startswith("the GPU's driver refuses what the card can't hold")


def test_on_a_driver_that_spills_torch_is_held(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    # torch's own allocator, which counts all it holds; and said, twice a warning.
    assert startup(monkeypatch, Card(30 * GIB, spills=True)) == "backend:native"
    assert memory.spills() and os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "backend:native"
    assert [record.levelname for record in caplog.records] == ["WARNING", "WARNING"]
    assert "gives memory the card doesn't have" in caplog.messages[0]
    assert caplog.messages[1] == (
        "seedvr2x holds itself to the card there: torch's own allocator, limited to the memory"
        " free less 3.0 GiB, and plans in 2.3 GiB less; a run that passes the limit is refused"
    )
    # The planner counts its peaks in what is free less what is held back and what torch's
    # allocator needs beyond a peak; never under nothing. Said: the plan's lines count in less
    # than the driver reports.
    assert memory.usable(30 * GIB) == round(24.7 * GIB) and memory.usable(5 * GIB) == 0
    assert caplog.messages[2] == (
        "the plan counts on 24.70 GiB of the GPU's memory, of the 30.00 GiB its driver reports"
        " free: 3.0 GiB are left to what torch doesn't count on the card, 2.3 to what torch's"
        " own allocator holds beyond a peak"
    )
    # torch is limited to what it holds and that: a fraction of the card's memory.
    limits: list[tuple[float, Any]] = []
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (20 * GIB, 32 * GIB))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: GIB)
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        lambda fraction, device: limits.append((fraction, device)),
    )
    assert memory.hold("cuda:0") == 18 * GIB
    assert limits == [(18 / 32, "cuda:0")]
    assert caplog.messages[-1] == (
        "torch held to 18.00 GiB on the GPU: the 20.00 GiB free less 3.0, and what it holds"
    )


def held(monkeypatch: pytest.MonkeyPatch, free: int, reserved: int = 0) -> list[float]:
    """torch's stand-in for hold and check: a 32 GiB card with `free` bytes free, torch holding
    `reserved`; returns the limits torch is given, as fractions."""
    import torch

    limits: list[float] = []
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (free, 32 * GIB))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: reserved)
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        lambda fraction, device: limits.append(fraction),
    )
    return limits


def test_the_limit_stays_within_the_card(monkeypatch: pytest.MonkeyPatch) -> None:
    # Less free than is held back: torch keeps what it holds, and gets no more. A reading above
    # the card's memory: the whole card, the fraction torch takes at the most.
    startup(monkeypatch, Card(30 * GIB, spills=True))
    limits = held(monkeypatch, 2 * GIB, reserved=GIB)
    assert memory.hold("cuda:0") == GIB and limits == [1 / 32]
    limits = held(monkeypatch, 31 * GIB, reserved=5 * GIB)
    assert memory.hold("cuda:0") == 33 * GIB and limits == [1.0]


def test_the_reading_names_its_figures(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # What a plan is made in, for the planner's words and the manifest (memory.Reading): what
    # the driver reports free to the process, what the whole card has free, which nothing reads
    # here, what the plan may count on, usable's, and whether the run is held. A driver that
    # refuses: all of what it reports, nothing said.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    startup(monkeypatch, None)
    caplog.clear()
    assert memory.reading(30 * GIB) == memory.Reading(30 * GIB, None, 30 * GIB, False)
    assert caplog.text == ""
    # One that spills: HELD_BACK and RESERVE less, said once, as usable says it.
    startup(monkeypatch, Card(30 * GIB, spills=True))
    caplog.clear()
    counted = 30 * GIB - memory.HELD_BACK - memory.RESERVE
    assert memory.reading(30 * GIB) == memory.Reading(30 * GIB, None, counted, True)
    assert memory.usable(30 * GIB) == counted
    [said, again] = caplog.messages
    assert said == again and said.startswith("the plan counts on 24.70 GiB of the GPU's memory")


def test_no_answer_behind_wsl_is_taken_for_a_driver_that_spills(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The driver couldn't be asked, said by the probe: behind WSL2's device it is Windows',
    # which gave what the card didn't have whenever it was asked. Held: a run held for nothing
    # loses memory, one not held where it should be loses its frames.
    assert startup(monkeypatch, Card(30 * GIB, spills=True, failing="cuInit")) == "backend:native"
    assert memory.spills() and os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "backend:native"
    assert "can't be asked whether it refuses: cuInit" in caplog.messages[0]
    assert caplog.messages[1] == (
        "the GPU is Windows', reached through WSL2, and its driver didn't say whether it refuses"
        " what the card can't hold: taken for one that gives system memory in its place, as"
        " that driver did whenever it was asked"
    )
    assert caplog.messages[2].startswith("seedvr2x holds itself to the card there")
    assert memory.usable(30 * GIB) == round(24.7 * GIB)
    limits = held(monkeypatch, 20 * GIB)
    assert memory.hold("cuda:0") == 17 * GIB and limits == [17 / 32]


@pytest.mark.parametrize(
    ("variable", "given", "warned"),
    [
        ("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync", True),
        ("PYTORCH_ALLOC_CONF", "backend:cudaMallocAsync", True),
        ("PYTORCH_CUDA_ALLOC_CONF", "backend:native,max_split_size_mb:512", False),
        ("PYTORCH_ALLOC_CONF", "max_split_size_mb:512", False),  # torch's default: its own
    ],
)
def test_the_environment_s_allocator_stands(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    variable: str,
    given: str,
    warned: bool,
) -> None:
    # The user's setting is left as it is, under either of the names torch reads, and nothing is
    # set beside it: with both set torch takes PYTORCH_CUDA_ALLOC_CONF, seedvr2x's default
    # written there would silence a user's PYTORCH_ALLOC_CONF. On a driver that spills, another
    # allocator than torch's own is warned of, since torch's limit holds less there, and the
    # card isn't checked before each phase: that allocator's slack would pass for another
    # program's memory.
    monkeypatch.setenv(variable, given)
    assert startup(monkeypatch, Card(30 * GIB, spills=True)) == "backend:native"
    assert {name: os.environ.get(name) for name in memory.VARIABLES} == {
        name: given if name == variable else None for name in memory.VARIABLES
    }
    said = f"the allocator is the environment's, {given}: torch's limit counts what torch has"
    assert (said in caplog.text) == warned and len(caplog.messages) == 2 + warned
    limits = held(monkeypatch, 20 * GIB)
    assert memory.hold("cuda:0") == 17 * GIB and limits == [17 / 32]
    held(monkeypatch, GIB)  # 1 GiB free where torch may take 17 more
    if warned:
        memory.check("cuda:0")
    else:
        with pytest.raises(torch.OutOfMemoryError):
            memory.check("cuda:0")
    # On a driver that refuses the setting stands as well, and nothing is said of it.
    caplog.clear()
    assert startup(monkeypatch, Card(30 * GIB, spills=False)) == "backend:cudaMallocAsync"
    assert os.environ[variable] == given and len(caplog.messages) == 0


def test_other_programs_filling_the_card_stop_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # torch's limit is worth the memory free when it was set: 20 GiB free, 1 held, so 18 GiB
    # in all. Before each unit the card's free memory is read again: while what torch may
    # still take fits there, nothing; once it doesn't, the run is refused as torch refuses
    # what passes its limit, saying why.
    startup(monkeypatch, Card(30 * GIB, spills=True))
    memory.check("cuda:0")  # not held yet: nothing read
    held(monkeypatch, 20 * GIB, reserved=GIB)
    assert memory.hold("cuda:0") == 18 * GIB
    memory.check("cuda:0")  # 20 GiB free where torch may take 17 more
    held(monkeypatch, 3 * GIB, reserved=15 * GIB)
    memory.check("cuda:0")  # torch's own work: 3 GiB free, and 3 more to take
    held(monkeypatch, 3 * GIB - 1, reserved=15 * GIB)
    with pytest.raises(torch.OutOfMemoryError) as refusal:
        memory.check("cuda:0")
    assert str(refusal.value) == (
        "the GPU has 3.00 GiB free where the run may still take 3.00: other programs hold the"
        " rest of the card's memory, or the run's own work took more than the 3.0 GiB left to it"
        " outside torch's allocator"
    )
    # A new run asks again, and starts unheld.
    startup(monkeypatch, Card(30 * GIB, spills=True))
    memory.check("cuda:0")


def test_the_card_is_checked_before_every_unit(monkeypatch: pytest.MonkeyPatch) -> None:
    # The run's models ask before each unit's phase, the one already entered too, and before
    # anything moves: a refusal there leaves the phase as it was.
    from types import SimpleNamespace

    from seedvr2x.runtime import model
    from seedvr2x.runtime.plan import Plan

    checked: list[str] = []
    monkeypatch.setattr(memory, "check", checked.append)
    runner = SimpleNamespace(dit=None, vae=None)
    models = model.Models(runner, torch.empty(0), torch.empty(0), "none", "the card", Plan(None))  # pyright: ignore[reportArgumentType]
    models.enter("encode")
    models.enter("encode")
    assert checked == ["the card"] * 2 and models.phase == "encode"

    def refused(device: object) -> None:
        raise torch.OutOfMemoryError("other programs hold the rest of the card's memory")

    monkeypatch.setattr(memory, "check", refused)
    with pytest.raises(torch.OutOfMemoryError):
        models.enter("dit")
    assert models.phase == "encode"

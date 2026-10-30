"""The driver that gives memory the card doesn't have (runtime/memory.py): the probe's reading
of the driver's answers, a stand-in in CUDA's place, and the whole card's free memory, a
stand-in in NVML's. The probe on a real driver, its ctypes calls and its verdict:
tests/test_placement.py's test_the_driver_is_asked (GPU); the card's free memory on a real one:
test_the_card_s_free_memory_is_read, here (GPU)."""

import ctypes
import logging
import os
import shutil
import subprocess
import sys
from typing import Any

import pytest
import torch

from seedvr2x.runtime import memory
from seedvr2x.runtime.memory import GIB, OUT_OF_MEMORY, OVER, SUCCESS, Driver

TOTAL = 32 * GIB
# The allocator's setting a held run takes by default (memory.EXPANDABLE), spelt out: what torch
# reads, held here to the letter.
HELD = "backend:native,expandable_segments:True"
# The two figures a held run plans under, as measured (runtime/memory.py's comments), read
# before the tests put their own in their place (round_figures).
MEASURED = (memory.HELD_BACK, memory.RESERVE, memory.HELD_BACK_UNREAD)


@pytest.fixture(autouse=True)
def round_figures(monkeypatch: pytest.MonkeyPatch) -> None:
    """What a held run leaves outside torch's limit and what it plans under it again, at
    figures of the tests' own, 3 GiB and 2.3: the tests hold what is done with them, whatever
    a card's measure makes them; test_the_measured_figures holds the measured ones, and
    test_a_card_unread_is_held_under_a_margin_of_its_own what each is taken under. Under these
    a check short of room is always past what torch held again may lose (more than 3 GiB grown
    outside its limit, 2.3 to lose): the re-hold's tests take the measured ones (measured)."""
    monkeypatch.setattr(memory, "HELD_BACK", 3 * GIB)
    monkeypatch.setattr(memory, "HELD_BACK_UNREAD", 3 * GIB)
    monkeypatch.setattr(memory, "RESERVE", round(2.3 * GIB))


def test_the_measured_figures() -> None:
    # 0.5 GiB left outside torch's limit under the card's own figure, design's figure where the
    # measure's own was 0.2 (DESIGN.md, Allocator, 2026-10-12: a desktop that wakes at night
    # mustn't stop a job); 3.0 under the driver's where the card's isn't read, and 4.5 planned
    # under the limit again: an RTX 5090's under WSL2 (runtime/memory.py, HELD_BACK,
    # HELD_BACK_UNREAD and RESERVE, their rows there). Changed with a new measure or a new
    # decision, and the comments with them.
    assert MEASURED == (round(0.5 * GIB), round(4.5 * GIB), 3 * GIB)


def test_a_card_unread_is_held_under_a_margin_of_its_own(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The module's figures, not the tests' own: the 0.5 GiB are left under the card's own
    # figure; the driver's stood 1.98 GiB over it at rest (30.20 GiB reported to a process where
    # the whole card had 28.22). Where NVML doesn't answer, 0.5 GiB under the driver's figure
    # would be a limit 1.48 GiB over what the card has: the plan, the limit and the check take
    # the margin measured under that figure alone, 3.0 GiB.
    for name, value in zip(("HELD_BACK", "RESERVE", "HELD_BACK_UNREAD"), MEASURED, strict=True):
        monkeypatch.setattr(memory, name, value)
    caplog.set_level(logging.DEBUG, logger="seedvr2x")  # the whole warning is said at debug
    reported, card = round(30.2 * GIB), round(28.22 * GIB)
    startup(monkeypatch, Card(reported, spills=True))
    assert caplog.messages[-1].endswith(
        "read through NVML; where NVML doesn't answer, the driver's figure less 3.0 GiB"
    )
    caplog.set_level(logging.INFO, logger="seedvr2x")
    counted = reported - 3 * GIB - round(4.5 * GIB)
    assert memory.reading(reported) == memory.Reading(reported, None, counted, True, 3 * GIB)
    assert caplog.messages[-1].endswith(
        "the plan counts on that figure, torch is held 3.0 GiB under it and the run's units are"
        " checked on it"
    )
    limits = held(monkeypatch, reported)
    limit = memory.hold("cuda:0")
    assert limit == reported - 3 * GIB and limits == [(reported - 3 * GIB) / TOTAL]
    assert reported - 3 * GIB < card  # under what the card had, where 0.5 under stood over it
    assert reported - round(0.5 * GIB) > card
    assert (
        "the 30.20 GiB free as its driver reports it to this process, less 3.0 GiB, and what it"
        " holds"
    ) in caplog.text
    # The run alone on its card, torch holding 22 GiB, the driver's figure 2.23 GiB under
    # torch's own count (the desktop's memory showing as the card fills): no stop.
    held(monkeypatch, reported - 22 * GIB - round(2.23 * GIB), reserved=22 * GIB)
    memory.check("cuda:0")
    # 0.78 GiB more gone from the figure, another program's: stopped, the margin named.
    held(monkeypatch, reported - 22 * GIB - round(3.01 * GIB), reserved=22 * GIB)
    with pytest.raises(torch.OutOfMemoryError, match=r"more than the 3\.0 GiB left to it"):
        memory.check("cuda:0")
    # The card read: the plan made and the limit set 0.5 GiB under the smaller of the two
    # figures, the card's or the driver's; and each unit checked against the card's, torch
    # holding 22 GiB, whatever the driver reports to the process by then: nothing, here.
    for seen, taken in ((card, card), (round(31 * GIB), reported)):
        nvml = Nvml({NAMED: seen})
        startup(monkeypatch, Card(reported, spills=True), nvml)
        left = round(0.5 * GIB)
        counted = taken - left - round(4.5 * GIB)
        assert memory.reading(reported) == memory.Reading(reported, seen, counted, True, left)
        held(monkeypatch, reported)
        limit = taken - left
        assert memory.hold("cuda:0") == limit
        held(monkeypatch, 0, reserved=22 * GIB)
        nvml.cards[NAMED] = limit - 22 * GIB
        memory.check("cuda:0")
        # Short by more than torch held again may lose under the expandable segments, the 4.5
        # GiB (test_a_check_short_of_room_holds_torch_again_under_the_segments): stopped, the
        # margin named.
        nvml.cards[NAMED] = limit - 22 * GIB - 4 * GIB - 1
        with pytest.raises(torch.OutOfMemoryError, match=r"more than the 0\.5 GiB left to it"):
            memory.check("cuda:0")


# A card's identifier as CUDA gives it, 16 bytes, and as NVML writes it; and another card's.
UUID, NAMED = bytes(range(16)), b"GPU-00010203-0405-0607-0809-0a0b0c0d0e0f"
OTHER = b"GPU-ffffffff-ffff-ffff-ffff-ffffffffffff"


class Card:
    """A stand-in for libcuda: a card of TOTAL bytes with `free` of them free, whose driver
    refuses what it can't hold, or gives it (spills). Another process gives back `once` bytes
    between the probe's first reading and its request, and `frees` more at every request, as a
    card in use does under the probe. Its device 0 is the card UUID names."""

    def __init__(
        self, free: int, spills: bool, once: int = 0, frees: int = 0, failing: str | None = None
    ) -> None:
        self.free, self.spills, self.once, self.frees = free, spills, once, frees
        self.failing = failing
        self.held = 0
        self.calls: list[str] = []

    def cuDeviceGetUuid_v2(self, identifier: Any, device: Any) -> int:
        ctypes.memmove(identifier._obj, UUID, len(UUID))
        return self.call("cuDeviceGetUuid") or SUCCESS

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


class Nvml:
    """A stand-in for libnvidia-ml: the cards it lists, each under its identifier as NVML
    writes it with the bytes free on it; `failing` names a call that fails, every time, or
    `times` times then no more. `open` counts its initialisations not yet shut down."""

    def __init__(self, cards: dict[bytes, int], failing: str | None = None) -> None:
        self.cards, self.failing = cards, failing
        self.times: int | None = None
        self.calls: list[str] = []
        self.open = 0

    def call(self, name: str) -> int | None:
        self.calls.append(name)
        if name != self.failing or self.times == 0:
            return None
        if self.times is not None:
            self.times -= 1
        return 999

    def nvmlInit_v2(self) -> int:
        failed = self.call("nvmlInit")
        self.open += not failed
        return failed or SUCCESS

    def nvmlDeviceGetHandleByUUID(self, named: bytes, handle: Any) -> int:
        failed = self.call("nvmlDeviceGetHandleByUUID")
        if failed or named not in self.cards:
            return failed or 6  # NVML_ERROR_NOT_FOUND
        handle._obj.value = 1 + list(self.cards).index(named)
        return SUCCESS

    def nvmlDeviceGetMemoryInfo_v2(self, handle: ctypes.c_void_p, found: Any) -> int:
        failed = self.call("nvmlDeviceGetMemoryInfo")
        # nvmlMemory_v2_t's own version word: 40 bytes, version 2. Any other is refused.
        if failed or found._obj.version != 40 | 2 << 24 or self.open != 1:
            return failed or 25  # NVML_ERROR_ARGUMENT_VERSION_MISMATCH
        assert handle.value is not None
        free = list(self.cards.values())[handle.value - 1]
        # What the driver keeps of a card is counted apart from what programs hold, and from
        # what is free: the fields in nvml.h's order.
        found._obj.total, found._obj.reserved = TOTAL, GIB // 2
        found._obj.free, found._obj.used = free, TOTAL - GIB // 2 - free
        return SUCCESS

    def nvmlShutdown(self) -> int:
        self.open -= 1
        return self.call("nvmlShutdown") or SUCCESS


def libraries(monkeypatch: pytest.MonkeyPatch, card: Card | None, nvml: Any = None) -> list[str]:
    """The stand-ins in the driver's libraries' place, `card` as CUDA's and `nvml` as NVML's;
    one not given is missing, as on a machine without it. Returns the names asked for, in
    order, as they come."""
    asked: list[str] = []

    def load(name: str) -> Any:
        asked.append(name)
        found = {memory.CUDA: card, memory.NVML: nvml}[name]
        if found is None:
            raise OSError(f"{name}: cannot open shared object file")
        return found

    monkeypatch.setattr(memory.ctypes, "CDLL", load)
    return asked


def probe(monkeypatch: pytest.MonkeyPatch, card: Card) -> Driver | None:
    libraries(monkeypatch, card)
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


def test_the_whole_card_s_free_memory(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # CUDA names its device 0, NVML finds the card of that name and says what is free on it:
    # NVML's figure for the whole card, the driver's count for the process not asked.
    card, nvml = Card(30 * GIB, spills=True), Nvml({NAMED: 28 * GIB})
    loaded = libraries(monkeypatch, card, nvml)
    assert memory.card_free() == 28 * GIB
    # NVML's library first, then CUDA's; CUDA asked for no context and no memory; NVML
    # initialised once the card is named, and shut down.
    assert loaded == [memory.NVML, memory.CUDA]
    assert card.calls == ["cuInit", "cuDeviceGet", "cuDeviceGetUuid"]
    read = ["nvmlInit", "nvmlDeviceGetHandleByUUID", "nvmlDeviceGetMemoryInfo", "nvmlShutdown"]
    assert nvml.calls == read and nvml.open == 0
    assert memory.card_free(0) == 28 * GIB and nvml.calls == read * 2 and nvml.open == 0
    assert not caplog.messages


def test_the_card_is_cuda_s_device_not_nvml_s_first(monkeypatch: pytest.MonkeyPatch) -> None:
    # Two cards, and CUDA's device 0 is the second NVML lists (CUDA_VISIBLE_DEVICES, or
    # CUDA's own order, fastest first): the figure is that card's, found by its identifier,
    # never the card NVML has at the same index.
    nvml = Nvml({OTHER: 10 * GIB, NAMED: 20 * GIB})
    libraries(monkeypatch, Card(30 * GIB, spills=True), nvml)
    assert memory.card_free() == 20 * GIB
    nvml.cards = {NAMED: 20 * GIB, OTHER: 10 * GIB}
    assert memory.card_free() == 20 * GIB


def test_a_card_nvml_does_not_list_gives_no_figure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # NVML answers, but for other cards than CUDA's: no figure, rather than another card's. The
    # reason is said at debug level: what a run does without the figure is its readings' to say
    # (reading, hold, check), each after asking again.
    caplog.set_level(logging.DEBUG, logger="seedvr2x")
    nvml = Nvml({OTHER: 10 * GIB})
    libraries(monkeypatch, Card(30 * GIB, spills=True), nvml)
    assert memory.card_free() is None
    assert [record.levelname for record in caplog.records] == ["DEBUG"]
    assert caplog.messages == [
        "the whole card's free memory can't be read through NVML (nvmlDeviceGetHandleByUUID,"
        f" {NAMED.decode()}: 6)"
    ]
    assert nvml.calls == ["nvmlInit", "nvmlDeviceGetHandleByUUID", "nvmlShutdown"]
    assert nvml.open == 0
    # CUDA's device 1, which the stand-in's CUDA doesn't have: NVML isn't even initialised.
    assert memory.card_free(1) is None and len(nvml.calls) == 3


CUDA_S = ["cuInit", "cuDeviceGet", "cuDeviceGetUuid"]
NVML_S = ["nvmlInit", "nvmlDeviceGetHandleByUUID", "nvmlDeviceGetMemoryInfo", "nvmlShutdown"]


@pytest.mark.parametrize("failing", [*CUDA_S, *NVML_S])
def test_a_call_that_fails_gives_no_figure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, failing: str
) -> None:
    # Whichever call fails, of CUDA's naming the card or of NVML's reading it: None, the call
    # named at debug level, nothing raised, and NVML shut down exactly when it was initialised.
    caplog.set_level(logging.DEBUG, logger="seedvr2x")
    card = Card(30 * GIB, spills=True, failing=failing if failing in CUDA_S else None)
    nvml = Nvml({NAMED: 28 * GIB}, failing=failing if failing in NVML_S else None)
    libraries(monkeypatch, card, nvml)
    if failing == "nvmlShutdown":
        # The figure was read: a shutdown that fails takes nothing from it.
        assert memory.card_free() == 28 * GIB and not caplog.messages
        return
    assert memory.card_free() is None
    assert [record.levelname for record in caplog.records] == ["DEBUG"]
    assert f"can't be read through NVML ({failing}" in caplog.text and caplog.text.endswith(
        ": 999)\n"
    )
    initialised = failing in NVML_S[1:]
    assert nvml.calls.count("nvmlShutdown") == initialised and nvml.open == 0
    assert ("nvmlInit" in nvml.calls) == (failing in NVML_S)


def test_nvml_s_structure_is_its_second_version_s() -> None:
    # nvmlMemory_v2_t as nvml.h lays it out: a version word, then four 64-bit counts, the
    # total, the driver's reserve, the free memory and the used, 40 bytes. The stand-in refuses
    # any other version word, as NVML does; the real NVML's answer through it is the GPU
    # test's, against nvidia-smi.
    layout = memory._Memory  # pyright: ignore[reportPrivateUsage]
    assert ctypes.sizeof(layout) == 40
    assert [(field[0], getattr(layout, field[0]).offset) for field in layout._fields_] == [
        *(("version", 0), ("total", 8), ("reserved", 16), ("free", 24), ("used", 32))
    ]


def test_no_nvml_library_or_an_older_one(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A machine without NVML: no figure, and CUDA's library not even loaded for it.
    caplog.set_level(logging.DEBUG, logger="seedvr2x")
    loaded = libraries(monkeypatch, Card(30 * GIB, spills=True))
    assert memory.card_free() is None and loaded == [memory.NVML]
    assert f"through NVML ({memory.NVML}: cannot open shared object file)" in caplog.text

    class Older:
        """An NVML from before nvmlDeviceGetMemoryInfo_v2, as ctypes shows one: no such
        attribute."""

        def __getattr__(self, name: str) -> Any:
            if name == "nvmlDeviceGetMemoryInfo_v2":
                raise AttributeError(f"undefined symbol: {name}")
            return getattr(nvml, name)

    # No figure, nothing raised, and NVML, initialised by then, shut down.
    nvml = Nvml({NAMED: 28 * GIB})
    libraries(monkeypatch, Card(30 * GIB, spills=True), Older())
    assert memory.card_free() is None and nvml.open == 0
    assert nvml.calls == ["nvmlInit", "nvmlDeviceGetHandleByUUID", "nvmlShutdown"]
    assert "(undefined symbol: nvmlDeviceGetMemoryInfo_v2)" in caplog.messages[-1]


def test_a_cpu_test_finds_no_nvml() -> None:
    # A test that stands in for a driver that spills, here or through the command line, would
    # read this machine's real card: for the tests not marked gpu, NVML's library has a name
    # no loader finds (conftest.py).
    with pytest.raises(OSError):
        ctypes.CDLL(memory.NVML)


def startup(monkeypatch: pytest.MonkeyPatch, card: Card | None, nvml: Nvml | None = None) -> str:
    """memory.startup with WSL2's device there and `card` behind it, or without the device;
    `nvml` as the machine's NVML, or none."""
    monkeypatch.setattr(memory, "paravirtual", lambda: card is not None)
    libraries(monkeypatch, card, nvml)
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
    # The whole free memory is the plan's, and the card's own figure is read nowhere: neither
    # of the driver's libraries is loaded, NVML's no more than CUDA's.
    loaded = libraries(monkeypatch, Card(30 * GIB, spills=False), Nvml({NAMED: 20 * GIB}))
    assert memory.usable(30 * GIB) == 30 * GIB and memory.hold("cuda:0") is None
    assert memory.reading(30 * GIB) == memory.Reading(30 * GIB, None, 30 * GIB, False)
    memory.check("cuda:0")  # nothing held: nothing read, torch's stand-in none
    assert not caplog.messages and not loaded


def test_a_driver_that_refuses_behind_wsl_changes_nothing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    # Behind WSL2's device, a driver whose figure is the card's own (the whole card has what it
    # reports, to the request's extra), and which refuses: asked, and nothing held. NVML was
    # read once, by the probe, and is nobody's business after: the plan, the limit and the
    # check load no library.
    nvml = Nvml({NAMED: 30 * GIB - OVER})
    monkeypatch.setattr(memory, "paravirtual", lambda: True)
    card = Card(30 * GIB, spills=False)
    loaded = libraries(monkeypatch, card, nvml)
    assert memory.startup() == "backend:cudaMallocAsync"
    assert "cuMemAlloc" in card.calls and nvml.calls.count("nvmlInit") == 1
    assert loaded == [memory.CUDA, memory.NVML, memory.CUDA]
    assert not memory.spills() and memory.usable(30 * GIB) == 30 * GIB
    assert memory.reading(30 * GIB) == memory.Reading(30 * GIB, None, 30 * GIB, False)
    assert memory.hold("cuda:0") is None
    memory.check("cuda:0")
    assert len(loaded) == 3 and nvml.calls.count("nvmlInit") == 1
    assert [record.levelname for record in caplog.records] == ["INFO"]
    assert caplog.messages[0].startswith("the GPU's driver refuses what the card can't hold")


def test_a_driver_that_leaves_other_programs_out_is_not_asked(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The driver reports 30 GiB free to the process where the whole card has 20, another
    # program holding the rest: a request for what it reports and more would take the card from
    # that program (it left the card, then came back under the run's limit: runtime/memory.py,
    # probe). Its own figure says what it is: taken for one that spills, and not asked.
    card, nvml = Card(30 * GIB, spills=True), Nvml({NAMED: 20 * GIB})
    libraries(monkeypatch, card, nvml)
    caplog.set_level(logging.DEBUG, logger="seedvr2x")
    found = memory.probe()
    assert found == Driver(True, 0, 30 * GIB, TOTAL, 20 * GIB)
    assert "cuMemAlloc" not in card.calls and card.calls[-1] == "cuCtxDestroy"
    assert nvml.calls.count("nvmlInit") == 1 and nvml.open == 0
    assert found is not None and found.said().startswith(
        "the GPU's driver gives memory the card doesn't have (it reports 30.00 GiB free to this"
        " process where the whole card has 20.00 by NVML, other programs' memory left out: not"
        " asked for more, which would push theirs off the card): system memory in its place"
    )
    # A run there is held, whether such a driver would have refused or not: held for nothing,
    # it loses memory; not held where it should be, its frames.
    assert startup(monkeypatch, Card(30 * GIB, spills=False), nvml) == HELD
    # The warning is the line (memory.LENDS); the finding whole, with that, is the debug line.
    assert memory.spills() and caplog.messages[0] == memory.LENDS
    assert "not asked for more" in caplog.messages[1] and caplog.records[1].levelname == "DEBUG"
    # The two figures within the request's extra of each other, or NVML silent: asked, as
    # before, by one request of what the driver reports and that extra.
    for free in (30 * GIB - OVER, 30 * GIB, 31 * GIB, None):
        card = Card(30 * GIB, spills=True)
        libraries(monkeypatch, card, None if free is None else Nvml({NAMED: free}))
        assert memory.probe() == Driver(True, 30 * GIB + OVER, 30 * GIB, TOTAL)
        assert card.calls.count("cuMemAlloc") == 1
    card = Card(30 * GIB, spills=True)
    libraries(monkeypatch, card, Nvml({NAMED: 30 * GIB - OVER - 1}))
    assert memory.probe() == Driver(True, 0, 30 * GIB, TOTAL, 30 * GIB - OVER - 1)


def test_on_a_driver_that_spills_torch_is_held(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="seedvr2x")
    # torch's own allocator, which counts all it holds; and said, twice a warning. The whole
    # card has more free than the driver reports to the process: the driver's figure is taken.
    nvml = Nvml({NAMED: 31 * GIB})
    assert startup(monkeypatch, Card(30 * GIB, spills=True), nvml) == HELD
    assert memory.spills() and os.environ["PYTORCH_CUDA_ALLOC_CONF"] == HELD
    # Each warning is a line, and the whole text of each is said at debug after it (decided on
    # 2026-10-28, built on 2026-10-29).
    assert [record.levelname for record in caplog.records] == [
        *("WARNING", "DEBUG", "WARNING", "DEBUG")
    ]
    assert caplog.messages[0] == memory.LENDS
    assert "gives memory the card doesn't have" in caplog.messages[1]
    assert caplog.messages[2] == (
        "seedvr2x holds itself to the card's own memory there: torch limited to the memory free"
        ' less 3.0 GiB, and the plan in 2.3 GiB less; the manual\'s "Under WSL2" section says why'
    )
    assert caplog.messages[3] == (
        "seedvr2x holds itself to the card there: torch's own allocator, its segments expandable,"
        " limited to the memory free less 3.0 GiB, and plans in 2.3 GiB less; a run that passes"
        " the limit is refused."
        " The memory free is the smaller of two figures: what the driver reports free to this"
        " process, which leaves other programs' memory out while the card has room, and what"
        " the whole card has free, read through NVML; where NVML doesn't answer, the driver's"
        " figure less 3.0 GiB"
    )
    # The planner counts its peaks in what is free less what is held back and what it leaves
    # torch's allocator beyond a peak; never under nothing. The reading names each figure for
    # the plan's own first line (runtime/planner.py); here it is said at debug level only.
    assert nvml.calls.count("nvmlInit") == 1  # the probe's, which then asked the driver
    held_under = memory.Reading(30 * GIB, 31 * GIB, round(24.7 * GIB), True, 3 * GIB)
    assert memory.reading(30 * GIB) == held_under
    assert caplog.records[4].levelname == "DEBUG" and caplog.messages[4] == (
        "the plan counts on 24.70 GiB of the GPU's memory, of the 30.00 GiB free as its driver"
        " reports it to this process (the whole card has 31.00 GiB free by NVML): 3.0 GiB are"
        " left to what torch doesn't count on the card, 2.3 to what the plan leaves torch's"
        " allocator beyond a peak"
    )
    assert memory.reading(5 * GIB) == memory.Reading(5 * GIB, 31 * GIB, 0, True, 3 * GIB)
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
    # The two figures come first, under --verbose (DESIGN.md, What it prints, 2026-10-29).
    assert caplog.messages[-2:] == [
        "free memory read: the driver reports 20.00 GiB free to this process, the whole card"
        " 31.00 GiB by NVML: the smaller is taken",
        "torch held to 18.00 GiB on the GPU: the 20.00 GiB free as its driver reports it to this"
        " process, less 3.0 GiB, and what it holds",
    ]
    # The two warnings' whole texts (debug) are two of the records.
    assert len(caplog.messages) == 5 + 2 + 1 and nvml.open == 0


def test_the_limit_is_said_without_the_drivers_figure_but_under_verbose(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # DESIGN.md, Architecture, What it prints (decided on 2026-10-29): the line says what torch is
    # held to and why (the free memory less the margin); the driver's own figure, and NVML's, are
    # a debug line before it.
    nvml = Nvml({NAMED: 28 * GIB})
    startup(monkeypatch, Card(30 * GIB, spills=True), nvml)
    held(monkeypatch, 30 * GIB)
    caplog.clear()
    caplog.set_level(logging.INFO, logger="seedvr2x")
    memory.hold("cuda:0")
    assert caplog.messages == [
        "torch held to 25.00 GiB on the GPU: the 28.00 GiB free on the whole card, less 3.0 GiB,"
        " and what it holds"
    ]
    caplog.clear()
    caplog.set_level(logging.DEBUG, logger="seedvr2x")
    memory.hold("cuda:0")
    assert caplog.messages == [
        "free memory read: the driver reports 30.00 GiB free to this process, the whole card"
        " 28.00 GiB by NVML: the smaller is taken",
        "torch held to 25.00 GiB on the GPU: the 28.00 GiB free on the whole card, less 3.0 GiB,"
        " and what it holds",
    ]


def test_the_whole_card_s_free_memory_is_taken_where_it_is_the_smaller(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A desktop and another program hold part of the card, which the driver's figure leaves
    # out while the card has room (the module's docstring): the plan, torch's limit and the
    # check before each unit all go by what the whole card has free, NVML's figure; the check
    # by that one alone, the driver's being under it near the full card with nothing behind
    # it (memory.check).
    caplog.set_level(logging.DEBUG, logger="seedvr2x")
    nvml = Nvml({NAMED: 28 * GIB})
    startup(monkeypatch, Card(30 * GIB, spills=True), nvml)
    counted = memory.Reading(30 * GIB, 28 * GIB, round(22.7 * GIB), True, 3 * GIB)
    assert memory.reading(30 * GIB) == counted
    assert caplog.messages[-1] == (
        "the plan counts on 22.70 GiB of the GPU's memory, of the 28.00 GiB free on the whole"
        " card by NVML (its driver reports 30.00 GiB free to this process): 3.0 GiB are left to"
        " what torch doesn't count on the card, 2.3 to what the plan leaves torch's allocator"
        " beyond a peak"
    )
    # The two figures the same: the card's, which is the one that names what is there.
    nvml.cards[NAMED] = 30 * GIB
    assert memory.usable(30 * GIB) == round(24.7 * GIB)
    assert "of the 30.00 GiB free on the whole card by NVML (its driver" in caplog.messages[-1]
    # torch's limit: the driver reports 20 GiB free to the process, the card has 18.
    nvml.cards[NAMED] = 18 * GIB
    limits = held(monkeypatch, 20 * GIB, reserved=GIB)
    assert memory.hold("cuda:0") == 16 * GIB and limits == [16 / 32]
    assert caplog.messages[-2:] == [
        "free memory read: the driver reports 20.00 GiB free to this process, the whole card"
        " 18.00 GiB by NVML: the smaller is taken",
        "torch held to 16.00 GiB on the GPU: the 18.00 GiB free on the whole card, less 3.0 GiB,"
        " and what it holds",
    ]
    # Before a unit, torch holding 15 GiB of its 16: the driver still reports 20 GiB free to
    # the process, and the card has 1 GiB left, as much as torch may still take: nothing. One
    # byte less on the card, another program's: refused, on the card's figure.
    held(monkeypatch, 20 * GIB, reserved=15 * GIB)
    nvml.cards[NAMED] = GIB
    memory.check("cuda:0")
    nvml.cards[NAMED] = GIB - 1
    with pytest.raises(torch.OutOfMemoryError) as refusal:
        memory.check("cuda:0")
    # The two figures a byte apart, and said apart: what is free rounded down, what the run
    # may take rounded up, where "1.00 GiB free ... may still take 1.00" gave no reason to stop;
    # and what the memory outside torch's allocator grew, rounded up: 3 GiB and a byte, the card
    # under the 3.0 GiB these figures leave the rest of it (round_figures), where torch held
    # again would stand under what it holds.
    assert str(refusal.value) == (
        "the GPU has 0.99 GiB free on the whole card by NVML (its driver reports 20.00 GiB free"
        " to this process) where the run may still take 1.00: other programs hold the rest of"
        " the card's memory, or the run's own work took more than the 3.0 GiB left to it"
        " outside torch's allocator; the memory outside that allocator grew 3.01 GiB on the card"
        " since torch's limit was set: the card has less than the 3.0 GiB left to the rest of it,"
        " where torch held again would stand under what it holds"
    )
    # The driver's figure the smaller, as it is near the full card: it stops nothing. Nothing
    # reported free to the process, and 1 GiB on the card, which torch may still take: the
    # unit runs.
    held(monkeypatch, 0, reserved=15 * GIB)
    nvml.cards[NAMED] = GIB
    memory.check("cuda:0")
    # One byte less on the card: refused, on the card's figure, named beside the driver's.
    nvml.cards[NAMED] = GIB - 1
    with pytest.raises(torch.OutOfMemoryError) as refusal:
        memory.check("cuda:0")
    assert str(refusal.value).startswith(
        "the GPU has 0.99 GiB free on the whole card by NVML (its driver reports 0.00 GiB free"
        " to this process) where the run may still take 1.00: other programs hold"
    )
    # Every reading opened NVML and shut it down, and none was warned of. Each check that found
    # the card short read it twice, torch's cache given back between the two (memory.check).
    assert nvml.open == 0 and nvml.calls.count("nvmlInit") == 10  # the probe's first
    assert [record.levelname for record in caplog.records].count("WARNING") == 2


def test_torch_is_held_a_first_time_before_the_first_pass(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A run holds torch twice (cli._run): a first time before its first pass, whose shot
    # detector and the warm-up after it worked on the GPU before any limit existed; then once
    # its plan is made, from the memory free then, the limit its units are checked against. The
    # first is said at debug level, as a first one; the second is the run's.
    caplog.set_level(logging.DEBUG, logger="seedvr2x")
    nvml = Nvml({NAMED: 20 * GIB})
    startup(monkeypatch, Card(30 * GIB, spills=True), nvml)
    limits = held(monkeypatch, 22 * GIB)
    caplog.clear()
    assert memory.hold("cuda:0", first=True) == 17 * GIB and limits == [17 / 32]
    assert [record.levelname for record in caplog.records] == ["DEBUG", "DEBUG"]
    assert caplog.messages == [
        "free memory read: the driver reports 22.00 GiB free to this process, the whole card"
        " 20.00 GiB by NVML: the smaller is taken",
        "torch held to 17.00 GiB on the GPU, a first time, before the first pass: the 20.00 GiB"
        " free on the whole card, less 3.0 GiB, and what it holds",
    ]
    # The first pass done, torch keeps 1 GiB of it in its cache, which the card has less of, and
    # another program has taken 2 GiB: the limit set again, on the memory free now.
    limits = held(monkeypatch, 19 * GIB, reserved=GIB)
    nvml.cards[NAMED] = 17 * GIB
    assert memory.hold("cuda:0") == 15 * GIB and limits == [15 / 32]
    assert caplog.records[-1].levelname == "INFO" and caplog.messages[-1].startswith(
        "torch held to 15.00 GiB on the GPU: the 17.00 GiB free on the whole card, less 3.0 GiB,"
        " and what it holds"
    )
    # Which is the limit a unit is checked against.
    held(monkeypatch, 19 * GIB, reserved=GIB)
    nvml.cards[NAMED] = 14 * GIB
    memory.check("cuda:0")
    nvml.cards[NAMED] = 14 * GIB - 1
    with pytest.raises(torch.OutOfMemoryError, match=r"where the run may still take 14\.00"):
        memory.check("cuda:0")
    # On a driver that refuses neither does anything.
    startup(monkeypatch, None)
    assert memory.hold("cuda:0", first=True) is None and memory.hold("cuda:0") is None


def test_the_card_read_is_the_device_s(monkeypatch: pytest.MonkeyPatch) -> None:
    # The limit and the check read the card of the device they are given, by CUDA's index for
    # it; a device that names none is CUDA's first, the one the plan's figure is read on.
    startup(monkeypatch, Card(30 * GIB, spills=True), Nvml({NAMED: 28 * GIB}))
    asked: list[int] = []
    monkeypatch.setattr(memory, "_read", lambda index: asked.append(index) or 28 * GIB)
    held(monkeypatch, 20 * GIB)
    memory.usable(30 * GIB)
    for device in ("cuda:1", "cuda", torch.device("cuda", 0)):
        memory.hold(device)
        memory.check(device)
    assert asked == [0, 1, 1, 0, 0, 0, 0]


def test_without_nvml_the_driver_s_figure_is_taken_and_said_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A machine without NVML's library, or whose NVML fails: no figure for the whole card, so
    # the plan, the limit and the check go by what the driver reports free to the process, as
    # before the card was read, and say which. Warned of once a run, as the plan is made, with
    # the reason, NVML asked twice: the limit and the checks after it stand on the plan's
    # footing and ask NVML nothing (DESIGN.md, Allocator, decided on 2026-10-13).
    caplog.set_level(logging.DEBUG, logger="seedvr2x")
    startup(monkeypatch, Card(30 * GIB, spills=True))
    caplog.clear()
    unread_reading = memory.Reading(30 * GIB, None, round(24.7 * GIB), True, 3 * GIB)
    assert memory.reading(30 * GIB) == unread_reading
    failed = f"{memory.NVML}: cannot open shared object file"
    assert caplog.messages == [
        "the whole card's free memory couldn't be read through NVML as the plan was made, asked"
        f" twice ({failed}; then {failed}): what the driver reports free to this process is"
        " taken alone, which leaves other programs' memory out while the card has room: the"
        " plan counts on that figure, torch is held 3.0 GiB under it and the run's units are"
        " checked on it",
        "the plan counts on 24.70 GiB of the GPU's memory, of the 30.00 GiB free as its driver"
        " reports it to this process (the whole card's, NVML's, wasn't read): 3.0 GiB are left"
        " to what torch doesn't count on the card, 2.3 to what the plan leaves torch's allocator"
        " beyond a peak",
    ]
    limits = held(monkeypatch, 20 * GIB, reserved=GIB)
    assert memory.hold("cuda:0") == 18 * GIB and limits == [18 / 32]
    assert caplog.messages[2:] == [
        "free memory read: the driver reports 20.00 GiB free to this process; the whole card's,"
        " by NVML, wasn't read",
        "torch held to 18.00 GiB on the GPU: the 20.00 GiB free as its driver reports it to this"
        " process, less 3.0 GiB, and what it holds",
    ]
    held(monkeypatch, 3 * GIB, reserved=15 * GIB)
    memory.check("cuda:0")
    assert [record.levelname for record in caplog.records] == ["WARNING", "DEBUG", "DEBUG", "INFO"]
    # An NVML that answers later in the run isn't asked: the run stays on its plan's footing,
    # the driver's figure, on which a unit is refused, a byte short.
    nvml = Nvml({NAMED: 2 * GIB})
    libraries(monkeypatch, Card(30 * GIB, spills=True), nvml)
    memory.check("cuda:0")
    held(monkeypatch, 3 * GIB - 1, reserved=15 * GIB)
    with pytest.raises(torch.OutOfMemoryError, match=r"2\.99 GiB free as its driver reports it"):
        memory.check("cuda:0")
    assert nvml.calls == []
    # A new run asks again, and is warned again.
    startup(monkeypatch, Card(30 * GIB, spills=True))
    caplog.clear()
    memory.usable(30 * GIB)
    assert [record.levelname for record in caplog.records] == ["WARNING", "DEBUG"]


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
    # the driver reports free to the process, what the whole card has free, what the plan may
    # count on, and whether the run is held. A driver that refuses: all of what it reports, the
    # card not read, nothing said.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    startup(monkeypatch, None)
    caplog.clear()
    assert memory.reading(30 * GIB) == memory.Reading(30 * GIB, None, 30 * GIB, False)
    assert caplog.text == ""
    # One that spills: the smaller figure, the card's here, less HELD_BACK and RESERVE; and
    # nothing said at the level a user reads, where the plan's own first line says it
    # (runtime/planner.py): once in a run's log, not twice. usable is the number alone. Each start
    # a process of its own, which finds no allocator set by the one before.
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    startup(monkeypatch, Card(30 * GIB, spills=True), Nvml({NAMED: 28 * GIB}))
    caplog.clear()
    counted = 28 * GIB - memory.HELD_BACK - memory.RESERVE
    left = memory.HELD_BACK
    assert memory.reading(30 * GIB) == memory.Reading(30 * GIB, 28 * GIB, counted, True, left)
    assert memory.usable(30 * GIB) == counted
    assert caplog.text == ""
    # What torch's limit leaves is the reading's own: the larger margin where the card's figure
    # isn't read, for the plan's first line to say (runtime/planner.py).
    startup(monkeypatch, Card(30 * GIB, spills=True))
    monkeypatch.setattr(memory, "HELD_BACK_UNREAD", 4 * GIB)
    counted = 30 * GIB - 4 * GIB - memory.RESERVE
    assert memory.reading(30 * GIB) == memory.Reading(30 * GIB, None, counted, True, 4 * GIB)


def test_no_answer_behind_wsl_is_taken_for_a_driver_that_spills(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The driver couldn't be asked, said by the probe: behind WSL2's device it is Windows',
    # which gave what the card didn't have whenever it was asked. Held: a run held for nothing
    # loses memory, one not held where it should be loses its frames.
    assert startup(monkeypatch, Card(30 * GIB, spills=True, failing="cuInit")) == HELD
    assert memory.spills() and os.environ["PYTORCH_CUDA_ALLOC_CONF"] == HELD
    assert "can't be asked whether it refuses: cuInit" in caplog.messages[0]
    assert caplog.messages[1] == (
        "the GPU is Windows', reached through WSL2, and its driver didn't say whether it refuses"
        " what the card can't hold: taken for one that gives system memory in its place, as"
        " that driver did whenever it was asked"
    )
    assert caplog.messages[2].startswith("seedvr2x holds itself to the card's own memory there")
    assert memory.usable(30 * GIB) == round(24.7 * GIB)
    limits = held(monkeypatch, 20 * GIB)
    assert memory.hold("cuda:0") == 17 * GIB and limits == [17 / 32]


# What torch's allocator takes of its two variables, each case in a process of its own, which
# needs no card: the variables set, and the backend torch then names.
TORCH_READS = (
    ({"PYTORCH_CUDA_ALLOC_CONF": ""}, "native"),
    # A held run's own default: torch's own allocator still, the setting taken whole.
    ({"PYTORCH_CUDA_ALLOC_CONF": HELD}, "native"),
    ({"PYTORCH_CUDA_ALLOC_CONF": "", "PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync"}, "native"),
    (
        {"PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync", "PYTORCH_ALLOC_CONF": ""},
        "cudaMallocAsync",
    ),
    # Blanks dropped, a tab too; the first backend standing, cudaMallocAsync to the letter; a
    # bracketed list one value; no backend named, torch's own (memory._allocator reads them so:
    # test_the_allocator_is_read_as_torch_reads_it).
    ({"PYTORCH_CUDA_ALLOC_CONF": "  "}, "native"),
    ({"PYTORCH_CUDA_ALLOC_CONF": "backend: native"}, "native"),
    ({"PYTORCH_CUDA_ALLOC_CONF": " backend : cudaMallocAsync "}, "cudaMallocAsync"),
    ({"PYTORCH_CUDA_ALLOC_CONF": "backend:\tcudaMallocAsync"}, "cudaMallocAsync"),
    ({"PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync,backend:native"}, "cudaMallocAsync"),
    ({"PYTORCH_CUDA_ALLOC_CONF": "backend:native,backend:cudaMallocAsync"}, "native"),
    ({"PYTORCH_CUDA_ALLOC_CONF": "backend:cudamallocasync,backend:cudaMallocAsync"}, "native"),
    (
        {
            "PYTORCH_CUDA_ALLOC_CONF": "roundup_power2_divisions:[256:1,512:2],backend:"
            "cudaMallocAsync"
        },
        "cudaMallocAsync",
    ),
    ({"PYTORCH_CUDA_ALLOC_CONF": "roundup_power2_divisions:[backend:cudaMallocAsync]"}, "native"),
    ({"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"}, "native"),
)


def test_the_allocator_is_read_as_torch_reads_it() -> None:
    # One reading of a setting, as torch's: whether it takes its own allocator, held to what
    # torch itself took from each of TORCH_READS (test_torch_takes_the_first_variable_set_even_
    # empty), and whether that allocator's segments are expandable, the last entry standing, as
    # torch's own allocator took them on an RTX 5090 (2026-10-16). A setting read by substrings
    # took "backend: native" for another allocator, the held run then unchecked, and
    # "backend:cudaMallocAsync,backend:native" for torch's own (a review, 2026-10-16).
    allocator = memory._allocator  # pyright: ignore[reportPrivateUsage]
    for variables, backend in TORCH_READS:
        setting = next((value for value in variables.values() if value), "")
        if (
            setting
            and "PYTORCH_CUDA_ALLOC_CONF" in variables
            and not variables["PYTORCH_CUDA_ALLOC_CONF"]
        ):
            continue  # the empty first variable hides the second: startup's own case
        assert allocator(setting)[0] == (backend == "native"), (setting, backend)
    assert allocator(memory.EXPANDABLE) == (True, True)
    assert allocator(memory.ASYNC) == (False, False)
    for setting, segments in (
        ("expandable_segments:True,expandable_segments:False", False),
        ("expandable_segments:False,expandable_segments:True", True),
        ("max_split_size_mb:512, expandable_segments : True", True),
        ("backend: native,expandable_segments:\tTrue", True),
        ("roundup_power2_divisions:[256:1,512:2],expandable_segments:True", True),
        ("roundup_power2_divisions:[expandable_segments:True]", False),
        ("expandable_segments:true", False),
        ("backend:cudaMallocAsync,expandable_segments:True", False),
    ):
        assert allocator(setting)[1] == segments, setting
    # A key or a value missing at a colon stops torch as it is imported (torch 2.14.1: "Index
    # out of bounds in ConfigTokenizer"): left to it, read as nothing.
    for setting in ("backend:", ":native", "expandable_segments:"):
        assert allocator(setting) == (True, False), setting


def test_torch_takes_the_first_variable_set_even_empty() -> None:
    # What startup's handling of an empty setting rests on (runtime/memory.py), asked of torch
    # itself: an empty setting names no allocator, torch's own then; and the first variable set
    # is the one read, empty or not, so that an empty PYTORCH_CUDA_ALLOC_CONF hides a
    # PYTORCH_ALLOC_CONF that names cudaMallocAsync.
    names = "import torch; print(torch.cuda.get_allocator_backend())"
    kept = {name: value for name, value in os.environ.items() if name not in memory.VARIABLES}
    for variables, backend in TORCH_READS:
        environment = {**kept, **variables, "CUDA_VISIBLE_DEVICES": ""}
        read = subprocess.run(
            [sys.executable, "-c", names],
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        assert read.stdout.strip() == backend, (variables, read.stdout, read.stderr[-500:])


@pytest.mark.parametrize(
    ("variables", "left", "warned"),
    [
        # Either variable empty, alone or both: no setting, torch's own allocator by default.
        ({"PYTORCH_CUDA_ALLOC_CONF": ""}, {"PYTORCH_CUDA_ALLOC_CONF": HELD}, False),
        ({"PYTORCH_ALLOC_CONF": ""}, {"PYTORCH_CUDA_ALLOC_CONF": HELD}, False),
        (
            {"PYTORCH_CUDA_ALLOC_CONF": "", "PYTORCH_ALLOC_CONF": ""},
            {"PYTORCH_CUDA_ALLOC_CONF": HELD},
            False,
        ),
        # An empty one beside the user's own: the user's stands, where torch can read it.
        (
            {"PYTORCH_CUDA_ALLOC_CONF": "", "PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync"},
            {"PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync"},
            True,
        ),
        (
            {"PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync", "PYTORCH_ALLOC_CONF": ""},
            {"PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync"},
            True,
        ),
    ],
)
def test_an_empty_setting_is_no_setting(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    variables: dict[str, str],
    left: dict[str, str],
    warned: bool,
) -> None:
    # An empty PYTORCH_CUDA_ALLOC_CONF or PYTORCH_ALLOC_CONF is no setting: it goes from the
    # environment, and the run takes the other variable's allocator, or its default. On a driver
    # that spills, the default is torch's own, held and checked before each phase; it passed
    # for a user's own allocator, warned of by an empty name, the check off. Left there, an
    # empty first variable would also hide the second from torch
    # (test_torch_takes_the_first_variable_set_even_empty).
    for name, value in variables.items():
        monkeypatch.setenv(name, value)
    assert startup(monkeypatch, Card(30 * GIB, spills=True)) == HELD
    assert {name: os.environ[name] for name in memory.VARIABLES if name in os.environ} == left
    assert ("the allocator is the environment's, " in caplog.text) == warned
    assert "the allocator is the environment's, :" not in caplog.text
    assert len(caplog.messages) == 2 + warned
    held(monkeypatch, 20 * GIB)
    memory.hold("cuda:0")
    held(monkeypatch, GIB)  # 1 GiB free where torch may take 17 more
    if warned:
        memory.check("cuda:0")
    else:
        with pytest.raises(torch.OutOfMemoryError):
            memory.check("cuda:0")
    # On a driver that refuses: the default there, or the user's own, and nothing said.
    for name in memory.VARIABLES:
        monkeypatch.delenv(name, raising=False)
    for name, value in variables.items():
        monkeypatch.setenv(name, value)
    caplog.clear()
    assert startup(monkeypatch, Card(30 * GIB, spills=False)) == "backend:cudaMallocAsync"
    assert len(set(left.values())) == 1 and len(caplog.messages) == 0
    there = {name: os.environ[name] for name in memory.VARIABLES if name in os.environ}
    assert there == dict.fromkeys(left, "backend:cudaMallocAsync")


@pytest.mark.parametrize(
    ("variable", "given", "kind"),
    [
        ("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync", "another"),
        ("PYTORCH_ALLOC_CONF", "backend:cudaMallocAsync", "another"),
        ("PYTORCH_CUDA_ALLOC_CONF", "backend:native,max_split_size_mb:512", "plain"),
        ("PYTORCH_ALLOC_CONF", "max_split_size_mb:512", "plain"),  # torch's default: its own
        ("PYTORCH_CUDA_ALLOC_CONF", "backend:native,expandable_segments:False", "plain"),
        # The last of a key stands, blanks around it ignored, as torch reads it.
        ("PYTORCH_ALLOC_CONF", "expandable_segments:True,expandable_segments:False", "plain"),
        ("PYTORCH_ALLOC_CONF", "max_split_size_mb:512, expandable_segments : True", "segments"),
        ("PYTORCH_CUDA_ALLOC_CONF", "backend:native,expandable_segments:True", "segments"),
        # As torch reads them: blanks dropped, the first backend standing (a review, 2026-10-16).
        ("PYTORCH_CUDA_ALLOC_CONF", "backend: native,expandable_segments:True", "segments"),
        ("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync,backend:native", "another"),
        ("PYTORCH_ALLOC_CONF", "backend:native,backend:cudaMallocAsync", "plain"),
        ("PYTORCH_CUDA_ALLOC_CONF", " backend : native ", "plain"),
        ("PYTORCH_CUDA_ALLOC_CONF", "backend:cudamallocasync", "plain"),
        ("PYTORCH_CUDA_ALLOC_CONF", "  ", "plain"),
    ],
)
def test_the_environment_s_allocator_stands(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    variable: str,
    given: str,
    kind: str,
) -> None:
    # The user's setting is left as it is, under either of the names torch reads, and nothing is
    # set beside it: with both set torch takes PYTORCH_CUDA_ALLOC_CONF, seedvr2x's default
    # written there would silence a user's PYTORCH_ALLOC_CONF. On a driver that spills, another
    # allocator than torch's own is warned of, since torch's limit holds less there, and the
    # card isn't checked before each phase: that allocator's slack would pass for another
    # program's memory. torch's own allocator without expandable segments, a held run's default
    # (memory.EXPANDABLE), is warned of too, a step short of room refused under it where they
    # slow it down and run it (DESIGN.md, Allocator, decided on 2026-10-13), and checked as the
    # default is.
    monkeypatch.setenv(variable, given)
    caplog.set_level(logging.DEBUG, logger="seedvr2x")  # the whole texts are debug lines
    assert startup(monkeypatch, Card(30 * GIB, spills=True)) == HELD
    assert {name: os.environ.get(name) for name in memory.VARIABLES} == {
        name: given if name == variable else None for name in memory.VARIABLES
    }
    warned = kind != "segments"
    said = {
        "another": f"the allocator is the environment's, {given}: torch's limit counts what torch",
        "plain": (
            f"the allocator is the environment's, {given}: torch's own without expandable"
            " segments, under which a step short of room is refused where they would slow it down"
            " and run it; add expandable_segments:True to it, or unset it, for a held run's"
            " default, backend:native,expandable_segments:True"
        ),
        "segments": "the allocator is the environment's",
    }[kind]
    warnings_said = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert (said in caplog.text) == warned and len(warnings_said) == 2 + warned
    # The line before it names torch's own allocator only where the run takes it, and whether
    # its segments are expandable.
    held_by = {
        "another": "torch limited to",
        "plain": "torch's own allocator without expandable segments, limited to",
        "segments": "torch's own allocator, its segments expandable, limited to",
    }[kind]
    [whole] = [
        m for m in caplog.messages if m.startswith("seedvr2x holds itself to the card there")
    ]
    assert whole.startswith(f"seedvr2x holds itself to the card there: {held_by}")
    assert warnings_said[1].startswith("seedvr2x holds itself to the card's own memory there")
    limits = held(monkeypatch, 20 * GIB)
    assert memory.hold("cuda:0") == 17 * GIB and limits == [17 / 32]
    held(monkeypatch, GIB)  # 1 GiB free where torch may take 17 more
    if kind == "another":
        loaded = libraries(monkeypatch, Card(30 * GIB, spills=True), Nvml({NAMED: GIB}))
        memory.check("cuda:0")
        assert not loaded  # nothing checked, nothing read
    else:
        with pytest.raises(torch.OutOfMemoryError):
            memory.check("cuda:0")
    # On a driver that refuses the setting stands as well, and nothing is said of it.
    caplog.clear()
    assert startup(monkeypatch, Card(30 * GIB, spills=False)) == "backend:cudaMallocAsync"
    assert os.environ[variable] == given
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]


def test_other_programs_filling_the_card_stop_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # torch's limit is worth the memory free when it was set: 20 GiB free, 1 held, so 18 GiB
    # in all. Before each unit the card's free memory is read again: while what torch may
    # still take fits there, nothing; once it doesn't, the run is refused as torch refuses
    # what passes its limit, saying why. The whole card's figure is the one read then: the
    # driver reports 1 GiB less to the process here, which stops nothing.
    nvml = Nvml({NAMED: 20 * GIB})
    startup(monkeypatch, Card(30 * GIB, spills=True), nvml)
    memory.check("cuda:0")  # not held yet: nothing read since the probe
    assert nvml.calls.count("nvmlInit") == 1
    held(monkeypatch, 20 * GIB, reserved=GIB)
    assert memory.hold("cuda:0") == 18 * GIB
    memory.check("cuda:0")  # 20 GiB free where torch may take 17 more
    held(monkeypatch, 2 * GIB, reserved=15 * GIB)
    nvml.cards[NAMED] = 3 * GIB
    memory.check("cuda:0")  # torch's own work: 3 GiB free on the card, and 3 more to take
    nvml.cards[NAMED] = 3 * GIB - 1
    with pytest.raises(torch.OutOfMemoryError) as refusal:
        memory.check("cuda:0")
    assert str(refusal.value) == (
        "the GPU has 2.99 GiB free on the whole card by NVML (its driver reports 2.00 GiB free"
        " to this process) where the run may still take 3.00: other programs hold the rest of"
        " the card's memory, or the run's own work took more than the 3.0 GiB left to it"
        " outside torch's allocator; the memory outside that allocator grew 3.01 GiB on the card"
        " since torch's limit was set: the card has less than the 3.0 GiB left to the rest of it,"
        " where torch held again would stand under what it holds"
    )
    # A new run asks again, and starts unheld: nothing read, of the card either, after its
    # probe.
    startup(monkeypatch, Card(30 * GIB, spills=True), nvml)
    read = len(nvml.calls)
    memory.check("cuda:0")
    assert len(nvml.calls) == read


def test_nvml_failing_at_a_check_is_asked_again_then_that_check_skipped(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # DESIGN.md, Allocator (2026-10-12): "NVML failing at one check is asked again, then that
    # check is skipped with a warning, never judged on the driver's figure." Near a full card
    # the driver's figure stands under the card's with nothing behind it, and judged on it six
    # runs of six stopped for nothing (memory.check): here it reports nothing free at all.
    caplog.set_level(logging.DEBUG, logger="seedvr2x")
    nvml = Nvml({NAMED: 20 * GIB})
    startup(monkeypatch, Card(30 * GIB, spills=True), nvml)
    held(monkeypatch, 20 * GIB, reserved=GIB)
    assert memory.hold("cuda:0") == 18 * GIB
    held(monkeypatch, 0, reserved=15 * GIB)  # torch may take 3 GiB more
    nvml.cards[NAMED] = 3 * GIB
    read, said = nvml.calls.count("nvmlInit"), len(caplog.messages)
    memory.check("cuda:0")
    assert nvml.calls.count("nvmlInit") == read + 1 and len(caplog.messages) == said
    # NVML fails once: asked again, and judged on that second reading, here a card that has the
    # room, then one that hasn't, by a byte.
    nvml.failing, nvml.times = "nvmlDeviceGetMemoryInfo", 1
    memory.check("cuda:0")
    assert nvml.calls.count("nvmlInit") == read + 3
    assert [record.levelname for record in caplog.records[said:]] == ["DEBUG"]
    assert caplog.messages[-1] == (
        "the whole card's free memory wasn't read through NVML at first"
        " (nvmlDeviceGetMemoryInfo: 999): read again"
    )
    nvml.times, nvml.cards[NAMED] = 1, 3 * GIB - 1
    with pytest.raises(torch.OutOfMemoryError, match=r"2\.99 GiB free on the whole card by NVML"):
        memory.check("cuda:0")
    # It fails twice: that check is skipped, with a warning naming both failures, and nothing is
    # refused, on the card's figure, which isn't there, nor on the driver's, which is 0.
    nvml.times = None
    read, said = nvml.calls.count("nvmlInit"), len(caplog.messages)
    memory.check("cuda:0")
    assert nvml.calls.count("nvmlInit") == read + 2 and nvml.open == 0
    assert [record.levelname for record in caplog.records[said:]] == ["WARNING"]
    assert caplog.messages[-1] == (
        "the whole card's free memory couldn't be read through NVML before this unit, asked"
        " twice (nvmlDeviceGetMemoryInfo: 999; then nvmlDeviceGetMemoryInfo: 999): the unit"
        " isn't checked against the card, and runs under torch's limit alone; what the driver"
        " reports free to this process isn't taken in its place, which stands under the card's"
        " near a full card and would stop the run for nothing"
    )
    # That check alone: the next one asks again, is warned again while NVML fails, and is
    # judged on the card's figure once it answers.
    memory.check("cuda:0")
    assert [record.levelname for record in caplog.records[said:]] == ["WARNING"] * 2
    nvml.failing = None
    with pytest.raises(torch.OutOfMemoryError, match=r"2\.99 GiB free on the whole card by NVML"):
        memory.check("cuda:0")
    nvml.cards[NAMED] = 3 * GIB
    memory.check("cuda:0")
    # NVML silent since the run's start is another case, and stays as it was: the limit is set
    # HELD_BACK_UNREAD under the driver's figure, and the check judged on that figure. A new
    # run starts there: what NVML said to the run before is nothing to it.
    startup(monkeypatch, Card(30 * GIB, spills=True))
    held(monkeypatch, 20 * GIB, reserved=GIB)
    assert memory.hold("cuda:0") == 18 * GIB
    held(monkeypatch, 3 * GIB - 1, reserved=15 * GIB)
    with pytest.raises(torch.OutOfMemoryError, match=r"2\.99 GiB free as its driver reports it"):
        memory.check("cuda:0")
    # NVML answering later in that run isn't asked: its limit went by the driver's figure, and
    # its checks go by it too (DESIGN.md, Allocator, decided on 2026-10-13: "a run whose limit
    # had to fall back on the driver's figure is checked on that figure").
    nvml = Nvml({NAMED: 3 * GIB})
    libraries(monkeypatch, Card(30 * GIB, spills=True), nvml)
    with pytest.raises(torch.OutOfMemoryError, match=r"2\.99 GiB free as its driver reports it"):
        memory.check("cuda:0")
    held(monkeypatch, 3 * GIB, reserved=15 * GIB)
    memory.check("cuda:0")
    assert nvml.calls == []


def test_nvml_failing_as_the_limit_is_set_leaves_the_plan_its_reserve(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # NVML gives the card's figure to the plan, then fails as torch's limit is set. The plan
    # counts on the card's figure less HELD_BACK and RESERVE; a limit set HELD_BACK_UNREAD under
    # the driver's figure alone, 1.98 GiB over the card's here, would leave the plan 3.98 GiB
    # under it where RESERVE is 4.5 (a review, 2026-10-12). Once NVML has answered in the run,
    # the limit's reading is asked again, as a check's is, and where it fails twice the limit is
    # set from the card's figure the plan was made in, with a warning: never from the driver's
    # figure alone.
    caplog.set_level(logging.DEBUG, logger="seedvr2x")
    card, reported = 20 * GIB, 20 * GIB + round(1.98 * GIB)
    nvml = Nvml({NAMED: card})
    startup(monkeypatch, Card(30 * GIB, spills=True), nvml)
    held(monkeypatch, reported)
    found = memory.reading(reported)
    assert (found.card, found.left) == (card, memory.HELD_BACK)
    assert found.counted == card - memory.HELD_BACK - memory.RESERVE
    # It fails once: asked again, and the limit set from that second reading.
    nvml.failing, nvml.times = "nvmlDeviceGetMemoryInfo", 1
    read, said = nvml.calls.count("nvmlInit"), len(caplog.messages)
    assert memory.hold("cuda:0") == card - memory.HELD_BACK
    assert nvml.calls.count("nvmlInit") == read + 2
    assert [record.levelname for record in caplog.records[said:]] == ["DEBUG", "DEBUG", "INFO"]
    # It fails twice: the plan's own figure of the card, and the plan keeps its reserve.
    nvml.times = None
    read, said = nvml.calls.count("nvmlInit"), len(caplog.messages)
    limit = memory.hold("cuda:0")
    assert limit == card - memory.HELD_BACK == found.counted + memory.RESERVE
    assert limit != reported - memory.HELD_BACK_UNREAD
    assert nvml.calls.count("nvmlInit") == read + 2 and nvml.open == 0
    assert [record.levelname for record in caplog.records[said:]] == ["WARNING", "DEBUG", "INFO"]
    assert caplog.messages[said] == (
        "as torch's limit was set, the whole card's free memory couldn't be read through NVML,"
        " asked twice (nvmlDeviceGetMemoryInfo: 999; then nvmlDeviceGetMemoryInfo: 999): the"
        " limit is set from the 20.00 GiB the card had free when the plan was made, not from"
        " what the driver reports free to this process alone"
    )
    # The driver's figure is still the one taken where it is the smaller of the two.
    held(monkeypatch, 19 * GIB)
    assert memory.hold("cuda:0") == 19 * GIB - memory.HELD_BACK
    # A plan that never had the card's figure, NVML having answered before it alone, at the
    # first hold: the plan counted on the driver's figure less HELD_BACK_UNREAD and RESERVE, as
    # its warning says, and the limit is set under that figure, as the plan was, NVML not asked.
    nvml = Nvml({NAMED: card})
    startup(monkeypatch, Card(30 * GIB, spills=True), nvml)
    held(monkeypatch, reported)
    assert memory.hold("cuda:0", first=True) == card - memory.HELD_BACK
    nvml.failing = "nvmlDeviceGetMemoryInfo"
    # A limit set again with no plan made (a run never does): nothing is taken of the run before
    # in the same process, whose plan had the card's figure, and the limit goes by the driver's.
    said = len(caplog.messages)
    assert memory.hold("cuda:0") == reported - memory.HELD_BACK_UNREAD
    assert caplog.messages[said] == (
        "as torch's limit was set, the whole card's free memory couldn't be read through NVML,"
        " asked twice (nvmlDeviceGetMemoryInfo: 999; then nvmlDeviceGetMemoryInfo: 999), nor"
        " when the plan was made: the limit is set 3.0 GiB under what the driver reports free to"
        " this process, and the run's units are checked on that figure"
    )
    found = memory.reading(reported)
    assert found.card is None
    assert found.counted == reported - memory.HELD_BACK_UNREAD - memory.RESERVE
    assert "the plan counts on that figure, torch is held 3.0 GiB under it" in caplog.text
    said, read = len(caplog.messages), nvml.calls.count("nvmlInit")
    limit = memory.hold("cuda:0")
    assert limit == reported - memory.HELD_BACK_UNREAD == found.counted + memory.RESERVE
    assert nvml.calls.count("nvmlInit") == read
    assert [record.levelname for record in caplog.records[said:]] == ["DEBUG", "INFO"]


def test_nvml_failing_at_the_plan_s_reading_is_asked_again(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # DESIGN.md, Allocator (decided on 2026-10-13): "NVML failing at the plan's reading is asked
    # again as well". Until then one failure there put the plan on the driver's figure, less
    # HELD_BACK_UNREAD and RESERVE, while the limit and the checks read the card again: the
    # plan stood 7.5 GiB under the driver's figure and the limit 0.5 under the card's, and the
    # warning said the run was held 3.0 GiB under the driver's figure (design's check).
    caplog.set_level(logging.DEBUG, logger="seedvr2x")
    card, reported = 20 * GIB, 20 * GIB + round(1.98 * GIB)
    nvml = Nvml({NAMED: card})
    startup(monkeypatch, Card(30 * GIB, spills=True), nvml)
    nvml.failing, nvml.times = "nvmlDeviceGetMemoryInfo", 1
    said, read = len(caplog.messages), nvml.calls.count("nvmlInit")
    found = memory.reading(reported)
    assert (found.card, found.left) == (card, memory.HELD_BACK)
    assert found.counted == card - memory.HELD_BACK - memory.RESERVE
    assert nvml.calls.count("nvmlInit") == read + 2
    assert [record.levelname for record in caplog.records[said:]] == ["DEBUG", "DEBUG"]
    assert caplog.messages[said] == (
        "the whole card's free memory wasn't read through NVML at first"
        " (nvmlDeviceGetMemoryInfo: 999): read again"
    )


# What NVML does at each of a run's readings, in turn (test_the_plan_the_limit_and_the_checks_
# stand_on_one_footing): answers, or fails twice, each reading asking again where it fails.
ANSWERS = ("answers", "fails")


@pytest.mark.parametrize("at_a_check", ANSWERS)
@pytest.mark.parametrize("at_the_limit", ANSWERS)
@pytest.mark.parametrize("at_the_plan", ANSWERS)
@pytest.mark.parametrize("at_the_first_hold", ANSWERS)
def test_the_plan_the_limit_and_the_checks_stand_on_one_footing(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    at_the_first_hold: str,
    at_the_plan: str,
    at_the_limit: str,
    at_a_check: str,
) -> None:
    # DESIGN.md, Allocator (decided on 2026-10-13): "a run whose limit had to fall back on the
    # driver's figure is checked on that figure ...: the plan, the limit and the checks stand on
    # one footing". NVML answering or not at each of a run's readings (cli._run): the first
    # hold, before the first pass; the plan's; the limit set for the run's units; a check
    # before a unit. The footing is the plan's: the card's where NVML answered it, the driver's
    # where it didn't. On it the limit is RESERVE over what the plan counts on, and the check
    # judges what is free. Until then NVML answering the first hold alone left the plan and the
    # limit on the driver's figure and every check skipped: nothing held that run to the card;
    # and NVML failing the plan alone put the limit and the checks on the card's figure under a
    # plan made on the driver's. The card has 20 GiB free, the driver reports 1.98 more to the
    # process, as at rest beside a desktop (the module's docstring).
    caplog.set_level(logging.DEBUG, logger="seedvr2x")
    card, reported = 20 * GIB, 20 * GIB + round(1.98 * GIB)
    nvml = Nvml({NAMED: card})
    startup(monkeypatch, Card(30 * GIB, spills=True), nvml)
    said = len(caplog.messages)

    def nvml_now(answers: str) -> int:
        """NVML from here on answering, or failing at every call; what it was asked so far."""
        nvml.failing = None if answers == "answers" else "nvmlDeviceGetMemoryInfo"
        return nvml.calls.count("nvmlInit")

    held_back, unread = memory.HELD_BACK, memory.HELD_BACK_UNREAD
    nvml_now(at_the_first_hold)
    held(monkeypatch, reported)
    first = memory.hold("cuda:0", first=True)
    assert first == (card - held_back if at_the_first_hold == "answers" else reported - unread)
    nvml_now(at_the_plan)
    found = memory.reading(reported)
    on_the_card = at_the_plan == "answers"
    if on_the_card:
        assert (found.card, found.left) == (card, held_back)
        assert found.counted == card - held_back - memory.RESERVE
    else:
        assert (found.card, found.left) == (None, unread)
        assert found.counted == reported - unread - memory.RESERVE
    # The run's limit, the figures as the plan had them: RESERVE over what the plan counts on,
    # NVML not asked where the plan went by the driver's figure.
    asked = nvml_now(at_the_limit)
    limit = memory.hold("cuda:0")
    assert limit is not None and limit == found.counted + memory.RESERVE
    readings = 0 if not on_the_card else 1 if at_the_limit == "answers" else 2
    assert nvml.calls.count("nvmlInit") - asked == readings
    # A check before a unit, torch holding 15 GiB of it: on the card's figure, the driver's
    # reporting nothing free at all (near a full card), refused a byte short of what torch may
    # still take, or skipped where NVML fails, never judged on the driver's; on the driver's,
    # NVML not asked, the card's figure, which would stop it, unread.
    room = limit - 15 * GIB
    asked = nvml_now(at_a_check)
    if on_the_card:
        held(monkeypatch, 0, reserved=15 * GIB)
        nvml.cards[NAMED] = room
        memory.check("cuda:0")
        nvml.cards[NAMED] = room - 1
        if at_a_check == "answers":
            with pytest.raises(torch.OutOfMemoryError, match="free on the whole card by NVML"):
                memory.check("cuda:0")
        else:
            memory.check("cuda:0")
        # The check refused read the card twice, torch's cache given back between the two
        # (memory.check); one NVML fails at is skipped at its first reading.
        assert nvml.calls.count("nvmlInit") - asked == (3 if at_a_check == "answers" else 4)
    else:
        nvml.cards[NAMED] = 0
        held(monkeypatch, room, reserved=15 * GIB)
        memory.check("cuda:0")
        held(monkeypatch, room - 1, reserved=15 * GIB)
        with pytest.raises(
            torch.OutOfMemoryError, match="as its driver reports it to this process"
        ):
            memory.check("cuda:0")
        assert nvml.calls.count("nvmlInit") == asked
    # What was said: a warning where NVML failed twice, each true of what was done, and none
    # where it answered or wasn't asked; at the first hold, whose limit is said at debug level,
    # at debug level too.
    failed = "asked twice (nvmlDeviceGetMemoryInfo: 999; then nvmlDeviceGetMemoryInfo: 999)"
    first_said = f"as torch was held a first time, {failed}: the first pass is held 3.0 GiB"
    debug = [record.getMessage() for record in caplog.records[said:] if record.levelname == "DEBUG"]
    assert any(first_said in message for message in debug) == (at_the_first_hold == "fails")
    expected = []
    if not on_the_card:
        expected.append(
            f"as the plan was made, {failed}: what the driver reports free to this process is"
            " taken alone, which leaves other programs' memory out while the card has room: the"
            " plan counts on that figure, torch is held 3.0 GiB under it and the run's units are"
            " checked on it"
        )
    elif at_the_limit == "fails":
        expected.append(f"{failed}: the limit is set from the 20.00 GiB the card had free when")
    if on_the_card and at_a_check == "fails":
        expected += [f"before this unit, {failed}: the unit isn't checked against the card"] * 2
    warned = [
        record.getMessage() for record in caplog.records[said:] if record.levelname == "WARNING"
    ]
    assert len(warned) == len(expected) and all(
        text in message for text, message in zip(expected, warned, strict=True)
    ), warned


def measured(monkeypatch: pytest.MonkeyPatch) -> None:
    """The module's figures as measured (MEASURED: 0.5 GiB left outside torch's limit, 4.5 that a
    plan leaves torch's allocator beyond a step's peak, 3.0 under the driver's figure alone), in
    the tests' own place (round_figures), under which torch is never held again."""
    for name, value in zip(("HELD_BACK", "RESERVE", "HELD_BACK_UNREAD"), MEASURED, strict=True):
        monkeypatch.setattr(memory, name, value)


def held_run(
    monkeypatch: pytest.MonkeyPatch, nvml: Nvml | None, reserve: int | None = None
) -> list[float]:
    """A run held to its card under the measured figures, its allocator the environment's or the
    default: torch held a first time with 28 GiB free on the card, then its plan made and torch
    held for its units with 26 GiB free on the card (`nvml` set so, or None: NVML silent, the
    driver's figure alone), the driver reporting 30 GiB and torch holding nothing, the plan's
    reserve `reserve` (None: memory.RESERVE); returns the limits torch is given from there on,
    as fractions of a 32 GiB card."""
    measured(monkeypatch)
    startup(monkeypatch, Card(30 * GIB, spills=True), nvml)
    held(monkeypatch, 30 * GIB)
    if nvml is not None:
        nvml.cards[NAMED] = 28 * GIB
    memory.hold("cuda:0", first=True)
    if nvml is not None:
        nvml.cards[NAMED] = 26 * GIB
    memory.reading(30 * GIB)
    limits = held(monkeypatch, 30 * GIB)
    memory.hold("cuda:0", reserve=reserve)
    return limits


def test_a_check_short_of_room_holds_torch_again_under_the_segments(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # DESIGN.md, Allocator (decided on 2026-10-16): "under the segments, a check that finds the
    # card short re-holds torch to the card's free memory less 0.5 GiB and goes on, said in a
    # warning, as long as what the limit has lost since the plan stays within the 4.5 GiB". A
    # desktop in use had stopped 3 runs of 6 under them, the memory outside torch grown 0.50 to
    # 0.59 GiB past its level at the plan. Here the plan's limit is 25.5 GiB, the card's 26 GiB
    # free less 0.5, torch holding nothing: 6 GiB of the card outside torch.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    nvml = Nvml({NAMED: 26 * GIB})
    limits = held_run(monkeypatch, nvml)
    assert limits == [25.5 / 32]
    # torch holding 20 GiB, the card 6 free: torch may still take 5.5, 0.5 to spare. Nothing.
    limits = held(monkeypatch, 0, reserved=20 * GIB)
    nvml.cards[NAMED] = 6 * GIB
    said = len(caplog.messages)
    memory.check("cuda:0")
    assert limits == [] and len(caplog.messages) == said
    # Another program takes 0.75 GiB: the card has 5.25 free where torch may still take 5.5.
    # Held again to what torch holds and the card's free memory less 0.5, 24.75 GiB: 0.75
    # under the plan's limit, the growth; and said, the run going on.
    nvml.cards[NAMED] = round(5.25 * GIB)
    memory.check("cuda:0")
    assert limits == [24.75 / 32]
    assert [record.levelname for record in caplog.records[said:]] == ["WARNING"]
    assert caplog.messages[-1] == (
        "the GPU has 5.25 GiB free on the whole card by NVML (its driver reports 0.00 GiB free to"
        " this process) where the run may still take 5.50: the memory outside torch's allocator,"
        " other programs' or the run's own, grew 0.75 GiB on the card since torch's limit was"
        " set, past the 0.5 GiB left to it; torch held again to 24.75 GiB, what it holds and the"
        " card's free memory less 0.5: 0.75 GiB under the limit set once the plan was made,"
        " within the 4.5 GiB that plan leaves torch's allocator beyond a step's peak, and the run"
        " goes on, a step left that much less beyond its peak"
    )
    # The next unit judged against that limit: torch holding 22 GiB of it may take 2.75 more,
    # the card has 3.25, 0.5 to spare: nothing. The program gone, 4 GiB free on the card: the
    # limit stays where it is, never raised again by a check.
    limits = held(monkeypatch, 0, reserved=22 * GIB)
    for free in (round(3.25 * GIB), 4 * GIB):
        nvml.cards[NAMED] = free
        memory.check("cuda:0")
    assert limits == [] and len(caplog.messages) == said + 1
    # A byte short of that limit: held again, 0.5 GiB and a byte grown since it was set, 1.25
    # and a byte lost since the plan's.
    nvml.cards[NAMED] = round(2.75 * GIB) - 1
    memory.check("cuda:0")
    assert limits == [(round(24.25 * GIB) - 1) / (32 * GIB)]
    assert caplog.messages[-1].startswith(
        "the GPU has 2.74 GiB free on the whole card by NVML (its driver reports 0.00 GiB free to"
        " this process) where the run may still take 2.75: the memory outside torch's allocator,"
        " other programs' or the run's own, grew 0.51 GiB on the card since torch's limit was"
        " set, past the 0.5 GiB left to it; torch held again to 24.25 GiB, what it holds and the"
        " card's free memory less 0.5: 1.26 GiB under the limit set once the plan was made,"
    )
    # Where the card has less than 0.5 GiB free, torch held again would stand under what it
    # holds: the run stops (test_a_re_hold_needs_the_0_5_left_to_the_rest_of_the_card).
    nvml.cards[NAMED] = GIB // 4
    with pytest.raises(torch.OutOfMemoryError, match=r"grew 2\.50 GiB on the card since"):
        memory.check("cuda:0")
    assert len(limits) == 1


def test_re_holds_summing_past_what_the_plan_leaves_stop_the_run(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # What the limit loses is counted from the one set once the plan was made, 25.5 GiB here,
    # not from the first, set before the first pass (27.5), nor from the last re-hold: a run
    # may lose 4.5 GiB of it in all, the reserve its plan left torch's allocator beyond a step's
    # peak, and stops past that, the stop saying it.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    nvml = Nvml({NAMED: 26 * GIB})
    held_run(monkeypatch, nvml)
    limits = held(monkeypatch, 0, reserved=20 * GIB)
    # 1 GiB taken by another program: held again to 24.5 GiB, 1 under the plan's limit.
    nvml.cards[NAMED] = 5 * GIB
    memory.check("cuda:0")
    # 3.5 more, the card at 1.5 where torch may still take 4.5: held again to 21.0, 4.5 under
    # the plan's limit in all, as much as the plan leaves: taken.
    nvml.cards[NAMED] = round(1.5 * GIB)
    memory.check("cuda:0")
    assert limits == [24.5 / 32, 21 / 32]
    assert "4.50 GiB under the limit set once the plan was made, within" in caplog.messages[-1]
    # A byte more: 0.5 GiB and a byte grown since that limit, a byte past the 0.5 left to it,
    # which held again would stand 5 GiB and a byte under the plan's: stopped, the limit as it
    # was.
    said = len(caplog.messages)
    nvml.cards[NAMED] = GIB - 1
    with pytest.raises(torch.OutOfMemoryError) as refusal:
        memory.check("cuda:0")
    assert limits == [24.5 / 32, 21 / 32] and len(caplog.messages) == said
    assert str(refusal.value) == (
        "the GPU has 0.99 GiB free on the whole card by NVML (its driver reports 0.00 GiB free to"
        " this process) where the run may still take 1.00: other programs hold the rest of the"
        " card's memory, or the run's own work took more than the 0.5 GiB left to it outside"
        " torch's allocator; the memory outside that allocator grew 0.51 GiB on the card since"
        " torch's limit was set: torch held again to what it holds and the card's free memory"
        " less 0.5 would stand 5.01 GiB under the limit set once the plan was made, past the 4.5"
        " GiB that plan leaves torch's allocator beyond a step's peak"
    )
    # A program taking 5 GiB at once, from the plan's limit: past it at the first check.
    nvml = Nvml({NAMED: 26 * GIB})
    held_run(monkeypatch, nvml)
    limits = held(monkeypatch, 0, reserved=20 * GIB)
    nvml.cards[NAMED] = GIB
    with pytest.raises(torch.OutOfMemoryError, match=r"grew 5\.00 GiB on the card .* 5\.00 GiB"):
        memory.check("cuda:0")
    assert limits == []


def test_a_plan_made_with_2_5_loses_that_much_at_the_most(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A plan made with the reserve of 2.5 GiB, the one with 4.5 refused (planner.plan; DESIGN.md,
    # Allocator): a check may take from the limit what that plan leaves torch's allocator beyond
    # a step's peak, 2.5, and the run stops past it (PROVISIONAL, implementation, 2026-10-21).
    # The plan's limit 25.5 GiB, torch holding 20.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    nvml = Nvml({NAMED: 26 * GIB})
    held_run(monkeypatch, nvml, reserve=memory.RESERVE_TIGHT)
    limits = held(monkeypatch, 0, reserved=20 * GIB)
    # 2.5 GiB taken by another program: held again to 23.0, 2.5 under the plan's limit.
    nvml.cards[NAMED] = round(3.5 * GIB)
    memory.check("cuda:0")
    assert limits == [23 / 32]
    assert caplog.messages[-1].endswith(
        "2.50 GiB under the limit set once the plan was made, within the 2.5 GiB that plan leaves"
        " torch's allocator beyond a step's peak, and the run goes on, a step left that much less"
        " beyond its peak"
    )
    # A byte more from the plan's limit: stopped, where the plan with 4.5 holds torch again
    # (test_re_holds_summing_past_what_the_plan_leaves_stop_the_run).
    for reserve, stops in ((memory.RESERVE_TIGHT, True), (None, False)):
        held_run(monkeypatch, nvml, reserve=reserve)
        limits = held(monkeypatch, 0, reserved=20 * GIB)
        nvml.cards[NAMED] = round(3.5 * GIB) - 1
        if not stops:
            memory.check("cuda:0")
            assert limits == [(23 * GIB - 1) / (32 * GIB)]
            continue
        with pytest.raises(torch.OutOfMemoryError) as refusal:
            memory.check("cuda:0")
        assert limits == [] and str(refusal.value).endswith(
            "would stand 2.51 GiB under the limit set once the plan was made, past the 2.5 GiB"
            " that plan leaves torch's allocator beyond a step's peak"
        )
    # A new run starts from the plan with 4.5 again (memory.startup).
    startup(monkeypatch, Card(30 * GIB, spills=True), nvml)
    assert memory._reserve == memory.RESERVE  # pyright: ignore[reportPrivateUsage]


def test_tightened_moves_the_reserve_alone() -> None:
    # The plan's last resort (planner.plan): the same figures, the reserve 2.5 GiB where 4.5,
    # under the card's figure or the driver's alone.
    for card, left in ((26 * GIB, memory.HELD_BACK), (None, memory.HELD_BACK_UNREAD)):
        taken = 30 * GIB if card is None else card
        reading = memory.Reading(30 * GIB, card, taken - left - memory.RESERVE, True, left)
        assert not reading.tight and reading.reserve == memory.RESERVE
        tight = memory.tightened(reading)
        assert tight == memory.Reading(
            30 * GIB, card, taken - left - memory.RESERVE_TIGHT, True, left, True
        )
        assert tight.tight and tight.reserve == memory.RESERVE_TIGHT == round(2.5 * GIB)
    # Never under nothing.
    assert memory.tightened(memory.Reading(GIB, None, 0, True, 3 * GIB)).counted == 0


def test_the_early_reading_counts_off_this_process_s_context(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A resume's first look at the card (memory.early; DESIGN.md, Memory planner, A plan shapes
    # the output a little, decided on 2026-10-28), through NVML before torch and so before this
    # process's CUDA context is on the card: that context off the figure, which the later reading
    # has counted; the module's state, startup's, untouched, and nothing logged.
    caplog.set_level(logging.DEBUG, logger="seedvr2x")
    # 0.67 GiB measured on an RTX 5090 under WSL2 (2026-10-29) plus 0.08 (DESIGN.md, Memory
    # planner, A resume whose recorded shape no longer fits); 1.23 until then.
    assert memory.CONTEXT == round(0.75 * GIB)
    state = (memory._spills, memory._native, memory._segments, memory._limit)  # pyright: ignore[reportPrivateUsage]
    free = 28 * GIB - memory.CONTEXT
    # A driver that refuses (no WSL2 device): its figure is the card's, nothing held.
    refusing = memory.Reading(free, None, free, False, nvml=28 * GIB)
    assert memory.early(28 * GIB) == refusing
    # The reading says it is early, and what was counted off NVML's figure.
    assert (
        refusing.context == memory.CONTEXT and memory.Reading(free, None, free, False).context == 0
    )
    # Behind WSL2's device, taken for held as startup takes a driver that doesn't answer: the card's
    # figure less what torch is held back from and the reserve, torch's own allocator with
    # its segments by default.
    monkeypatch.setattr(memory, "paravirtual", lambda: True)
    counted = free - memory.HELD_BACK - memory.RESERVE
    held = memory.Reading(
        free, free, counted, True, memory.HELD_BACK, expandable=True, nvml=28 * GIB
    )
    assert memory.early(28 * GIB) == held
    # The environment's allocator stands, as startup reads it: cudaMallocAsync or torch's own
    # without the segments are not expandable; an empty setting is none.
    for setting, expandable in (
        ("backend:cudaMallocAsync", False),
        ("backend:native", False),
        ("backend:native,expandable_segments:True", True),
        ("", True),
    ):
        monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", setting)
        assert memory.early(28 * GIB).expandable is expandable, setting
    # What programs hold of the card by NVML rides on the reading (card_state).
    assert memory.early(28 * GIB, 4 * GIB).nvml_used == 4 * GIB and refusing.nvml_used is None
    # Never under nothing.
    # A card with less free than the context counted (1 GiB while CONTEXT was 1.23).
    least = memory.Reading(0, 0, 0, True, memory.HELD_BACK, expandable=True, nvml=GIB // 2)
    assert memory.early(GIB // 2) == least and least.context == GIB // 2
    assert state == (memory._spills, memory._native, memory._segments, memory._limit)  # pyright: ignore[reportPrivateUsage]
    assert caplog.messages == []


def test_card_state_reads_what_is_free_and_what_programs_hold(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # card_free's reading with nvmlMemory_v2_t's `used` beside it (the driver's own `reserved`
    # not in it), for a resume's early reading; card_free itself is the free memory alone.
    caplog.set_level(logging.DEBUG, logger="seedvr2x")
    monkeypatch.setattr(memory, "_read_all", lambda index: (28 * GIB, 3 * GIB))
    assert memory.card_state() == (28 * GIB, 3 * GIB) and memory.card_free() == 28 * GIB
    monkeypatch.setattr(memory, "_read_all", lambda index: "no such library")
    caplog.clear()
    assert memory.card_state() is None and memory.card_free() is None
    assert len([m for m in caplog.messages if "no such library" in m]) == 2


def cached(
    monkeypatch: pytest.MonkeyPatch, nvml: Nvml, reserved: int, allocated: int, freed: int = 0
) -> dict[str, int]:
    """torch's stand-in for a check, holding `reserved` bytes of which `allocated` are in use:
    giving its cache back (empty_cache) gives the rest to the card, by NVML's figure, and
    `freed` more besides, another program's, meanwhile; returns its state, what it holds and how
    many times its cache was given back."""
    import torch

    state = {"reserved": reserved, "emptied": 0}

    def empty_cache() -> None:
        nvml.cards[NAMED] += state["reserved"] - allocated + freed
        state["reserved"] = allocated
        state["emptied"] += 1

    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: state["reserved"])
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    return state


def test_a_check_short_gives_torch_s_cache_back_first(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # DESIGN.md, Allocator (decided on 2026-10-18): a check that finds the card short gives
    # torch's cache back first and judges the card again, so that a growth larger than torch's
    # room, within the 4.5 GiB, is held again, not stopped. The plan's limit 25.5 GiB, torch
    # holding 25 of it, 20 in use: a program taking 0.75 GiB leaves the card 0.25 free, under the
    # 0.5 a re-hold needs, which stopped the run (test_a_re_hold_needs_the_0_5_left_to_the_rest_
    # of_the_card). Its cache given back, the card has 5.25 where torch may still take 5.5: held
    # again to 24.75, the 0.75 grown.
    caplog.set_level(logging.DEBUG, logger="seedvr2x")
    nvml = Nvml({NAMED: 26 * GIB})
    held_run(monkeypatch, nvml)
    limits = held(monkeypatch, 0)
    state = cached(monkeypatch, nvml, 25 * GIB, 20 * GIB)
    # With room, nothing given back.
    nvml.cards[NAMED] = GIB
    memory.check("cuda:0")
    assert state["emptied"] == 0 and limits == []
    nvml.cards[NAMED] = GIB // 4
    memory.check("cuda:0")
    assert state["emptied"] == 1 and limits == [24.75 / 32]
    assert caplog.records[-1].levelname == "WARNING"
    assert caplog.messages[-1] == (
        "the GPU has 5.25 GiB free on the whole card by NVML (its driver reports 0.00 GiB free to"
        " this process) where the run may still take 5.50, torch's cache of 5.00 GiB given back"
        " first: the memory outside torch's allocator, other programs' or the run's own, grew"
        " 0.75 GiB on the card since torch's limit was set, past the 0.5 GiB left to it; torch"
        " held again to 24.75 GiB, what it holds and the card's free memory less 0.5: 0.75 GiB"
        " under the limit set once the plan was made, within the 4.5 GiB that plan leaves"
        " torch's allocator beyond a step's peak, and the run goes on, a step left that much less"
        " beyond its peak"
    )
    # The card judged again with room, another program having given 1 GiB back meanwhile:
    # nothing held again, nor warned of; said at debug level.
    held_run(monkeypatch, nvml)
    limits = held(monkeypatch, 0)
    state = cached(monkeypatch, nvml, 25 * GIB, 20 * GIB, freed=GIB)
    said = len(caplog.records)
    nvml.cards[NAMED] = GIB // 4
    memory.check("cuda:0")
    assert state["emptied"] == 1 and limits == []
    assert [record.levelname for record in caplog.records[said:]] == ["DEBUG"]
    assert caplog.messages[-1].startswith(
        "the card short of what the run may still take before this unit, torch's cache given back"
        " (5.00 GiB), and judged again: 6.25 GiB free on the whole card by NVML"
    )
    # NVML failing at the second reading, the first having found the card short: judged on that
    # reading and what the cache gave back, never skipped: held again as above.
    held_run(monkeypatch, nvml)
    limits = held(monkeypatch, 0)
    cached(monkeypatch, nvml, 25 * GIB, 20 * GIB)
    emptied = torch.cuda.empty_cache

    def fails_after() -> None:
        emptied()
        nvml.failing = "nvmlDeviceGetMemoryInfo"

    monkeypatch.setattr(torch.cuda, "empty_cache", fails_after)
    nvml.cards[NAMED] = GIB // 4
    memory.check("cuda:0")
    assert limits == [24.75 / 32] and "torch held again to 24.75 GiB" in caplog.messages[-1]
    assert "couldn't be read through NVML again" in caplog.messages[-2]
    nvml.failing = None
    # Past the reserve, its cache given back all the same: stopped, saying so. Torch holding 20
    # GiB, 15 in use, a program taking the card's last 6.
    held_run(monkeypatch, nvml)
    limits = held(monkeypatch, 0)
    cached(monkeypatch, nvml, 20 * GIB, 15 * GIB)
    nvml.cards[NAMED] = 0
    with pytest.raises(torch.OutOfMemoryError) as refusal:
        memory.check("cuda:0")
    assert limits == [] and str(refusal.value) == (
        "the GPU has 5.00 GiB free on the whole card by NVML (its driver reports 0.00 GiB free to"
        " this process) where the run may still take 10.50, torch's cache of 5.00 GiB given back"
        " first: other programs hold the rest of the card's memory, or the run's own work took"
        " more than the 0.5 GiB left to it outside torch's allocator; the memory outside that"
        " allocator grew 6.00 GiB on the card since torch's limit was set: torch held again to"
        " what it holds and the card's free memory less 0.5 would stand 6.00 GiB under the limit"
        " set once the plan was made, past the 4.5 GiB that plan leaves torch's allocator beyond"
        " a step's peak"
    )
    # Nothing to give back: stopped where the card has under the 0.5 GiB, as before.
    held_run(monkeypatch, nvml)
    limits = held(monkeypatch, 0)
    state = cached(monkeypatch, nvml, 25 * GIB, 25 * GIB)
    nvml.cards[NAMED] = GIB // 4
    with pytest.raises(torch.OutOfMemoryError, match=r"less than the 0\.5 GiB left") as refusal:
        memory.check("cuda:0")
    assert state["emptied"] == 1 and "given back" not in str(refusal.value)
    # Under torch's allocator without the segments, which holds nothing again: given back, then
    # stopped, the figures those judged after it.
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "backend:native")
    held_run(monkeypatch, nvml)
    limits = held(monkeypatch, 0)
    state = cached(monkeypatch, nvml, 25 * GIB, 20 * GIB)
    nvml.cards[NAMED] = GIB // 4
    with pytest.raises(torch.OutOfMemoryError) as refusal:
        memory.check("cuda:0")
    assert state["emptied"] == 1 and limits == []
    assert str(refusal.value).startswith(
        "the GPU has 5.25 GiB free on the whole card by NVML (its driver reports 0.00 GiB free to"
        " this process) where the run may still take 5.50, torch's cache of 5.00 GiB given back"
        " first: other programs hold"
    )


def test_a_held_run_s_vae_unit_is_watched_for_refused_allocations(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # DESIGN.md, Allocator (decided on 2026-10-18): torch's count of refused allocations
    # (num_ooms) grows by 19 to 38 on every VAE pass that lost its frames under the segments,
    # cuDNN's workspaces refused, and stays 0 on every one that kept them. A held run under
    # torch's own allocator warns where it grows across a unit; nothing where it doesn't, nor
    # under another allocator, which counts none, nor on a driver that refuses, nor where the
    # unit raises.
    import torch

    caplog.set_level(logging.INFO, logger="seedvr2x")
    count = {"num_ooms": 3}
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda device: dict(count))

    def unit(grows: int) -> None:
        with memory.watched("cuda:0", "shot 2/5's decode", "frames"):
            count["num_ooms"] += grows

    for setting, held in (("backend:cudaMallocAsync", True), (None, False), (None, True)):
        if setting is None:
            monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
        else:
            monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", setting)
        startup(monkeypatch, Card(30 * GIB, spills=held))
        caplog.clear()
        count["num_ooms"] = 3
        unit(0)
        unit(19)
        warned = [record for record in caplog.records if "refused" in record.message]
        if setting is not None or not held:
            assert warned == []
            continue
        assert [record.levelname for record in warned] == ["WARNING"]
        assert warned[0].message == (
            "shot 2/5's decode: torch's allocator refused 19 of its requests during it (3 to 22 in"
            " all): where cuDNN's workspaces are refused it runs other kernels, which give other"
            " bits: its frames may not be those of a run with more room on the card"
        )
    with pytest.raises(torch.OutOfMemoryError):
        with memory.watched("cuda:0", "shot 2/5's decode", "frames"):
            count["num_ooms"] += 1
            raise torch.OutOfMemoryError("refused")
    assert len([record for record in caplog.records if "refused" in record.message]) == 1


def test_a_re_hold_needs_the_0_5_left_to_the_rest_of_the_card(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A review (2026-10-16): held again with the card under 0.5 GiB free, torch's limit stood
    # under what torch held, every later check passed unjudged and the bound went: the plan's
    # limit 25.5 GiB, torch holding 25, a program taking 0.75 left the card 0.25 free and torch
    # was held again to 24.75; the program taking 4.75 more went unsaid. The run stops there.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    nvml = Nvml({NAMED: 26 * GIB})
    held_run(monkeypatch, nvml)
    limits = held(monkeypatch, 0, reserved=25 * GIB)
    nvml.cards[NAMED] = GIB // 4
    with pytest.raises(torch.OutOfMemoryError) as refusal:
        memory.check("cuda:0")
    assert limits == []
    assert str(refusal.value) == (
        "the GPU has 0.25 GiB free on the whole card by NVML (its driver reports 0.00 GiB free to"
        " this process) where the run may still take 0.50: other programs hold the rest of the"
        " card's memory, or the run's own work took more than the 0.5 GiB left to it outside"
        " torch's allocator; the memory outside that allocator grew 0.75 GiB on the card since"
        " torch's limit was set: the card has less than the 0.5 GiB left to the rest of it, where"
        " torch held again would stand under what it holds"
    )
    # Held again where the card has the 0.5 GiB, to what torch holds at the least, and each
    # growth after it judged: held again or stopped, never passed unsaid. Torch holding 24.5
    # GiB, a program's 0.75 leave the card 0.75 free: held again to 24.75; 0.25 more, the card
    # at 0.5 where torch may still take 0.25: nothing; the card's last 0.5: stopped, 0.75 grown
    # since that limit.
    held_run(monkeypatch, nvml)
    limits = held(monkeypatch, 0, reserved=round(24.5 * GIB))
    nvml.cards[NAMED] = round(0.75 * GIB)
    memory.check("cuda:0")
    assert limits == [24.75 / 32]
    nvml.cards[NAMED] = GIB // 2
    memory.check("cuda:0")
    nvml.cards[NAMED] = 0
    with pytest.raises(
        torch.OutOfMemoryError, match=r"grew 0\.75 GiB on the card .* less than the 0\.5 GiB"
    ):
        memory.check("cuda:0")
    assert limits == [24.75 / 32]
    # The card with the 0.5 GiB to the byte: held again to what torch holds, nothing over it.
    held_run(monkeypatch, nvml)
    limits = held(monkeypatch, 0, reserved=24 * GIB)
    nvml.cards[NAMED] = GIB // 2
    memory.check("cuda:0")
    assert limits == [24 / 32] and "torch held again to 24.00 GiB" in caplog.messages[-1]


def test_the_growth_is_counted_from_the_card_s_figure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The limit set under the driver's figure, the smaller then: 24 GiB reported, 26 on the
    # card, the limit 23.5, the card left 2.5 GiB outside it. The check goes by that limit; the
    # growth it says is the card's own, from the card's figure and what torch held as the limit
    # was set: 3 GiB here, where counted from the limit it read 1 (a review, 2026-10-16). Under
    # torch's allocator without the segments the run stops; under them torch is held again, 1
    # GiB under that limit for 3 grown.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    measured(monkeypatch)
    for setting in ("backend:native", None):
        if setting is None:
            monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF")
        else:
            monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", setting)
        nvml = Nvml({NAMED: 26 * GIB})
        startup(monkeypatch, Card(30 * GIB, spills=True), nvml)
        memory.reading(30 * GIB)
        held(monkeypatch, 24 * GIB)
        assert memory.hold("cuda:0") == round(23.5 * GIB)
        held(monkeypatch, 0, reserved=20 * GIB)
        nvml.cards[NAMED] = 3 * GIB
        if setting is not None:
            with pytest.raises(torch.OutOfMemoryError, match=r"grew 3\.00 GiB on the card since"):
                memory.check("cuda:0")
            continue
        memory.check("cuda:0")
        assert "grew 3.00 GiB on the card since" in caplog.messages[-1]
        assert "torch held again to 22.50 GiB" in caplog.messages[-1]
        assert "1.00 GiB under the limit set once the plan was made" in caplog.messages[-1]


@pytest.mark.parametrize(
    ("setting", "again"),
    [
        ("backend:native", False),
        ("backend:native,expandable_segments:False", False),
        # torch's own allocator with the segments, as the user may write it (_allocator): as the
        # default.
        (" backend: native , expandable_segments:True", True),
    ],
)
def test_without_the_segments_a_check_short_of_room_stops_the_run(
    monkeypatch: pytest.MonkeyPatch, setting: str, again: bool
) -> None:
    # Torch's allocator without expandable segments refuses a step short of room where they
    # slow it down and run it (memory.EXPANDABLE): torch isn't held again there, and the run
    # stops as before, the growth said. The allocator as the run reads it, the user's own
    # setting standing.
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", setting)
    nvml = Nvml({NAMED: 26 * GIB})
    held_run(monkeypatch, nvml)
    limits = held(monkeypatch, 0, reserved=20 * GIB)
    nvml.cards[NAMED] = round(5.25 * GIB)
    if again:
        memory.check("cuda:0")
        assert limits == [24.75 / 32]
        return
    with pytest.raises(torch.OutOfMemoryError) as refusal:
        memory.check("cuda:0")
    assert limits == []
    assert str(refusal.value) == (
        "the GPU has 5.25 GiB free on the whole card by NVML (its driver reports 0.00 GiB free to"
        " this process) where the run may still take 5.50: other programs hold the rest of the"
        " card's memory, or the run's own work took more than the 0.5 GiB left to it outside"
        " torch's allocator; the memory outside that allocator grew 0.75 GiB on the card since"
        " torch's limit was set"
    )


def test_on_the_driver_s_figure_a_check_short_of_room_stops_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # NVML silent at the plan's reading: the plan, the limit and the checks on the driver's
    # figure, 3 GiB under it (DESIGN.md, Allocator: "Past it, under the plain allocator, or on
    # the driver's figure, the run stops as today"). The limit 27 GiB, torch holding 20 of it,
    # the driver reporting 6.25 free: 3.75 GiB grown by that figure, within what torch held
    # again could lose on the card's: stopped all the same, the growth said.
    limits = held_run(monkeypatch, None)
    assert limits == [27 / 32]
    limits = held(monkeypatch, 7 * GIB, reserved=20 * GIB)
    memory.check("cuda:0")
    limits = held(monkeypatch, round(6.25 * GIB), reserved=20 * GIB)
    with pytest.raises(torch.OutOfMemoryError) as refusal:
        memory.check("cuda:0")
    assert limits == []
    assert str(refusal.value) == (
        "the GPU has 6.25 GiB free as its driver reports it to this process (the whole card's,"
        " NVML's, wasn't read) where the run may still take 7.00: other programs hold the rest of"
        " the card's memory, or the run's own work took more than the 3.0 GiB left to it outside"
        " torch's allocator; the memory outside that allocator grew 3.75 GiB by the driver's"
        " figure since torch's limit was set"
    )


def test_a_check_nvml_fails_at_holds_nothing_again(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # NVML failing twice at a check under the segments: that check skipped with its warning, as
    # before (test_nvml_failing_at_a_check_is_asked_again_then_that_check_skipped), the limit
    # untouched, on the driver's figure no more than on the card's; the next check that reads
    # the card short holds torch again.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    nvml = Nvml({NAMED: 26 * GIB})
    held_run(monkeypatch, nvml)
    limits = held(monkeypatch, 0, reserved=20 * GIB)
    nvml.cards[NAMED] = round(5.25 * GIB)
    nvml.failing = "nvmlDeviceGetMemoryInfo"
    said = len(caplog.messages)
    memory.check("cuda:0")
    assert limits == [] and [record.levelname for record in caplog.records[said:]] == ["WARNING"]
    assert caplog.messages[-1].startswith(
        "the whole card's free memory couldn't be read through NVML before this unit"
    )
    nvml.failing = None
    memory.check("cuda:0")
    assert limits == [24.75 / 32] and "torch held again to 24.75 GiB" in caplog.messages[-1]


def test_a_check_before_the_plan_s_limit_holds_nothing_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Torch held once only, before the first pass (a run checks no unit before its second
    # limit): no limit set for the run's units to count a loss from, and a check short of room
    # stops, as before.
    measured(monkeypatch)
    nvml = Nvml({NAMED: 26 * GIB})
    startup(monkeypatch, Card(30 * GIB, spills=True), nvml)
    held(monkeypatch, 30 * GIB)
    memory.hold("cuda:0", first=True)
    limits = held(monkeypatch, 0, reserved=20 * GIB)
    nvml.cards[NAMED] = round(5.25 * GIB)
    with pytest.raises(torch.OutOfMemoryError, match=r"grew 0\.75 GiB on the card since"):
        memory.check("cuda:0")
    assert limits == []


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


@pytest.mark.gpu
def test_the_card_s_free_memory_is_read() -> None:
    # The real libraries: NVML finds the card CUDA's device 0 is and gives what the whole card
    # has free, nvidia-smi's figure for it, this process's memory counted in it: a GiB taken
    # by torch is a GiB less. On a driver that refuses too, where a run never asks: there the
    # same GiB is expected, torch's own allocator asking the driver for the request as it is,
    # and cudaMallocAsync's pool for a little more (32 MiB at its first reservation:
    # cli._free_memory); not run on one yet. Hence the bounds: nothing under the GiB itself,
    # and room above it for an allocator's rounding and a first use's libraries.
    device = torch.device("cuda", 0)
    # The context, the allocator's first block and what a first kernel loads, before the
    # first reading: they are the process's too. On an RTX 5090 under WSL2 the GiB taken cost
    # the card 1.0000 GiB so, and 1.1250 with the first kernel run after the reading.
    first = torch.ones(1, device=device)
    torch.cuda.synchronize(device)
    reported = torch.cuda.mem_get_info(device)[0]
    before = memory.card_free(0)
    assert before is not None and before > 0
    lines = [f"the driver reports {reported / GIB:.4f} GiB free to this process"]
    lines.append(f"the whole card has {before / GIB:.4f} GiB free by NVML")
    smi = shutil.which("nvidia-smi")
    if smi is not None:
        named = f"GPU-{torch.cuda.get_device_properties(device).uuid}"
        query = [smi, f"--id={named}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
        said = subprocess.run(query, capture_output=True, text=True, check=True).stdout
        lines.append(f"nvidia-smi: {said.strip()} MiB free")
        # Read a moment apart, on a card whose other programs may move: within 64 MiB.
        assert abs(int(said) * (1 << 20) - before) < 64 << 20, lines
    taken = torch.empty(GIB, dtype=torch.uint8, device=device).fill_(1)
    torch.cuda.synchronize(device)
    after = memory.card_free(0)
    assert after is not None
    lines.append(f"{(before - after) / GIB:.4f} GiB less with a GiB taken")
    print("; ".join(lines))
    assert 0.95 * GIB < before - after < 1.5 * GIB, lines
    assert memory.card_free(torch.cuda.device_count()) is None  # no such device of CUDA's
    del first, taken


HELD_RUN = """
import torch
GIB = 1 << 30
device = torch.device("cuda", 0)
first = torch.ones(1, device=device)
torch.cuda.synchronize(device)
reserved, total = torch.cuda.memory_reserved(device), torch.cuda.mem_get_info(device)[1]
limit = reserved + GIB
torch.cuda.set_per_process_memory_fraction(limit / total, device)
fits = torch.empty(GIB // 2, dtype=torch.uint8, device=device).fill_(1)
try:
    torch.empty(GIB, dtype=torch.uint8, device=device)
    refused = False
except torch.OutOfMemoryError:
    refused = True
del fits
after = torch.empty(3 * GIB // 4, dtype=torch.uint8, device=device).fill_(2)
torch.cuda.synchronize(device)
expandable = [segment["is_expandable"] for segment in torch.cuda.memory_snapshot()]
whole = int(after[::4096].min()) == 2 == int(after[::4096].max())  # no copy of its size
under = torch.cuda.max_memory_reserved(device) <= limit
backend = torch.cuda.get_allocator_backend()
print(backend, refused, int(after[-1]), whole, under, all(expandable), len(expandable))
"""


@pytest.mark.gpu
def test_a_held_run_s_allocator_keeps_torch_s_limit() -> None:
    # The allocator a held run takes by default (memory.EXPANDABLE), in a process of its own as
    # torch reads it when imported: torch's own, every segment of its pool expandable; a
    # request that passes torch's limit is refused, an error Python catches, what torch holds
    # never over the limit, and the work after the refusal is right. On a driver that refuses
    # too: the setting is torch's, and the limit its own count.
    environment = {
        name: value for name, value in os.environ.items() if name not in memory.VARIABLES
    }
    environment[memory.VARIABLES[0]] = memory.EXPANDABLE
    ran = subprocess.run(
        [sys.executable, "-c", HELD_RUN],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert ran.returncode == 0, ran.stderr[-800:]
    print(ran.stdout.strip())
    backend, refused, last, whole, under, expandable, segments = ran.stdout.split()
    assert (backend, refused, last, whole, under) == ("native", "True", "2", "True", "True")
    assert expandable == "True" and int(segments) >= 1


def test_a_driver_that_spills_is_warned_of_in_a_line_each_at_every_start(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # DESIGN.md, Architecture, What it prints (decided on 2026-10-28, built on 2026-10-29): the
    # two warnings (501 and 521 characters at every start, resumes included) are a line each, short
    # and plain, and point to the manual; without --verbose nothing else is said of them.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    assert startup(monkeypatch, Card(30 * GIB, spills=True), Nvml({NAMED: 20 * GIB})) == HELD
    assert [r.levelname for r in caplog.records] == ["WARNING", "WARNING"]
    assert caplog.messages[0] == memory.LENDS
    assert "Under WSL2" in caplog.messages[1]
    assert all(len(message) < 220 for message in caplog.messages)

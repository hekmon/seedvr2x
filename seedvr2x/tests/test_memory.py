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
from typing import Any

import pytest
import torch

from seedvr2x.runtime import memory
from seedvr2x.runtime.memory import GIB, OUT_OF_MEMORY, OVER, SUCCESS, Driver

TOTAL = 32 * GIB
# The two figures a held run plans under, as measured (runtime/memory.py's comments), read
# before the tests put their own in their place (round_figures).
MEASURED = (memory.HELD_BACK, memory.RESERVE, memory.HELD_BACK_UNREAD)


@pytest.fixture(autouse=True)
def round_figures(monkeypatch: pytest.MonkeyPatch) -> None:
    """What a held run leaves outside torch's limit and what it plans under it again, at
    figures of the tests' own, 3 GiB and 2.3: the tests hold what is done with them, whatever
    a card's measure makes them; test_the_measured_figures holds the measured ones, and
    test_a_card_unread_is_held_under_a_margin_of_its_own what each is taken under."""
    monkeypatch.setattr(memory, "HELD_BACK", 3 * GIB)
    monkeypatch.setattr(memory, "HELD_BACK_UNREAD", 3 * GIB)
    monkeypatch.setattr(memory, "RESERVE", round(2.3 * GIB))


def test_the_measured_figures() -> None:
    # 0.2 GiB left outside torch's limit under the card's own figure, 3.0 under the driver's
    # where the card's isn't read, and 4.5 planned under the limit again: an RTX 5090's under
    # WSL2 (runtime/memory.py, HELD_BACK, HELD_BACK_UNREAD and RESERVE, their rows there).
    # Changed with a new measure, and the comments with them.
    assert MEASURED == (round(0.2 * GIB), round(4.5 * GIB), 3 * GIB)


def test_a_card_unread_is_held_under_a_margin_of_its_own(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The measured figures, not the tests' own: the 0.2 GiB were measured under the card's own
    # figure; the driver's stood 1.98 GiB over it at rest (30.20 GiB reported to a process where
    # the whole card had 28.22). Where NVML doesn't answer, 0.2 GiB under the driver's figure
    # would be a limit 1.78 GiB over what the card has: the plan, the limit and the check take
    # the margin measured under that figure alone, 3.0 GiB.
    for name, value in zip(("HELD_BACK", "RESERVE", "HELD_BACK_UNREAD"), MEASURED, strict=True):
        monkeypatch.setattr(memory, name, value)
    caplog.set_level(logging.INFO, logger="seedvr2x")
    reported, card = round(30.2 * GIB), round(28.22 * GIB)
    startup(monkeypatch, Card(reported, spills=True))
    assert caplog.messages[-1].endswith(
        "read through NVML; where NVML doesn't answer, the driver's figure less 3.0 GiB"
    )
    counted = reported - 3 * GIB - round(4.5 * GIB)
    assert memory.reading(reported) == memory.Reading(reported, None, counted, True)
    assert caplog.messages[-1].endswith("and the run is held 3.0 GiB under it")
    limits = held(monkeypatch, reported)
    limit = memory.hold("cuda:0")
    assert limit == reported - 3 * GIB and limits == [(reported - 3 * GIB) / TOTAL]
    assert reported - 3 * GIB < card  # under what the card had, where 0.2 under stood over it
    assert "the 30.20 GiB free less 3.0, and what it holds; free as its driver" in caplog.text
    # The run alone on its card, torch holding 22 GiB, the driver's figure 2.23 GiB under
    # torch's own count (the desktop's memory showing as the card fills): no stop.
    held(monkeypatch, reported - 22 * GIB - round(2.23 * GIB), reserved=22 * GIB)
    memory.check("cuda:0")
    # 0.78 GiB more gone from the figure, another program's: stopped, the margin named.
    held(monkeypatch, reported - 22 * GIB - round(3.01 * GIB), reserved=22 * GIB)
    with pytest.raises(torch.OutOfMemoryError, match=r"more than the 3\.0 GiB left to it"):
        memory.check("cuda:0")
    # The card read: the plan made and the limit set 0.2 GiB under the smaller of the two
    # figures, the card's or the driver's; and each unit checked against the card's, torch
    # holding 22 GiB, whatever the driver reports to the process by then: nothing, here.
    for seen, taken in ((card, card), (round(31 * GIB), reported)):
        nvml = Nvml({NAMED: seen})
        startup(monkeypatch, Card(reported, spills=True), nvml)
        counted = taken - round(0.2 * GIB) - round(4.5 * GIB)
        assert memory.reading(reported) == memory.Reading(reported, seen, counted, True)
        held(monkeypatch, reported)
        limit = taken - round(0.2 * GIB)
        assert memory.hold("cuda:0") == limit
        held(monkeypatch, 0, reserved=22 * GIB)
        nvml.cards[NAMED] = limit - 22 * GIB
        memory.check("cuda:0")
        nvml.cards[NAMED] = limit - 22 * GIB - 1
        with pytest.raises(torch.OutOfMemoryError, match=r"more than the 0\.2 GiB left to it"):
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
    writes it with the bytes free on it; `failing` names a call that fails. `open` counts its
    initialisations not yet shut down."""

    def __init__(self, cards: dict[bytes, int], failing: str | None = None) -> None:
        self.cards, self.failing = cards, failing
        self.calls: list[str] = []
        self.open = 0

    def call(self, name: str) -> int | None:
        self.calls.append(name)
        return 999 if name == self.failing else None

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
    # NVML answers, but for other cards than CUDA's: no figure, rather than another card's.
    nvml = Nvml({OTHER: 10 * GIB})
    libraries(monkeypatch, Card(30 * GIB, spills=True), nvml)
    assert memory.card_free() is None
    assert caplog.messages == [
        "the whole card's free memory can't be read through NVML (nvmlDeviceGetHandleByUUID,"
        f" {NAMED.decode()}: 6): what the driver reports free to this process is taken alone,"
        " which leaves other programs' memory out while the card has room, and the run is held"
        " 3.0 GiB under it"
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
    # named in a warning, nothing raised, and NVML shut down exactly when it was initialised.
    card = Card(30 * GIB, spills=True, failing=failing if failing in CUDA_S else None)
    nvml = Nvml({NAMED: 28 * GIB}, failing=failing if failing in NVML_S else None)
    libraries(monkeypatch, card, nvml)
    if failing == "nvmlShutdown":
        # The figure was read: a shutdown that fails takes nothing from it.
        assert memory.card_free() == 28 * GIB and not caplog.messages
        return
    assert memory.card_free() is None
    assert [record.levelname for record in caplog.records] == ["WARNING"]
    assert f"can't be read through NVML ({failing}" in caplog.text
    assert ": 999): what the driver reports free to this process is taken alone" in caplog.text
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
    assert startup(monkeypatch, Card(30 * GIB, spills=False), nvml) == "backend:native"
    assert memory.spills() and "not asked for more" in caplog.messages[0]
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
    assert startup(monkeypatch, Card(30 * GIB, spills=True), nvml) == "backend:native"
    assert memory.spills() and os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "backend:native"
    assert [record.levelname for record in caplog.records] == ["WARNING", "WARNING"]
    assert "gives memory the card doesn't have" in caplog.messages[0]
    assert caplog.messages[1] == (
        "seedvr2x holds itself to the card there: torch's own allocator, limited to the memory"
        " free less 3.0 GiB, and plans in 2.3 GiB less; a run that passes the limit is refused."
        " The memory free is the smaller of two figures: what the driver reports free to this"
        " process, which leaves other programs' memory out while the card has room, and what"
        " the whole card has free, read through NVML; where NVML doesn't answer, the driver's"
        " figure less 3.0 GiB"
    )
    # The planner counts its peaks in what is free less what is held back and what torch's
    # allocator needs beyond a peak; never under nothing. The reading names each figure for
    # the plan's own first line (runtime/planner.py); here it is said at debug level only.
    assert nvml.calls.count("nvmlInit") == 1  # the probe's, which then asked the driver
    held_under = memory.Reading(30 * GIB, 31 * GIB, round(24.7 * GIB), True)
    assert memory.reading(30 * GIB) == held_under
    assert caplog.records[2].levelname == "DEBUG" and caplog.messages[2] == (
        "the plan counts on 24.70 GiB of the GPU's memory, of the 30.00 GiB free as its driver"
        " reports it to this process (the whole card has 31.00 GiB free by NVML): 3.0 GiB are"
        " left to what torch doesn't count on the card, 2.3 to what torch's own allocator holds"
        " beyond a peak"
    )
    assert memory.reading(5 * GIB) == memory.Reading(5 * GIB, 31 * GIB, 0, True)
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
        "torch held to 18.00 GiB on the GPU: the 20.00 GiB free less 3.0, and what it holds;"
        " free as its driver reports it to this process (the whole card has 31.00 GiB free by"
        " NVML)"
    )
    assert len(caplog.messages) == 5 and nvml.open == 0


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
    assert memory.reading(30 * GIB) == memory.Reading(30 * GIB, 28 * GIB, round(22.7 * GIB), True)
    assert caplog.messages[-1] == (
        "the plan counts on 22.70 GiB of the GPU's memory, of the 28.00 GiB free on the whole"
        " card by NVML (its driver reports 30.00 GiB free to this process): 3.0 GiB are left to"
        " what torch doesn't count on the card, 2.3 to what torch's own allocator holds beyond"
        " a peak"
    )
    # The two figures the same: the card's, which is the one that names what is there.
    nvml.cards[NAMED] = 30 * GIB
    assert memory.usable(30 * GIB) == round(24.7 * GIB)
    assert "of the 30.00 GiB free on the whole card by NVML (its driver" in caplog.messages[-1]
    # torch's limit: the driver reports 20 GiB free to the process, the card has 18.
    nvml.cards[NAMED] = 18 * GIB
    limits = held(monkeypatch, 20 * GIB, reserved=GIB)
    assert memory.hold("cuda:0") == 16 * GIB and limits == [16 / 32]
    assert caplog.messages[-1] == (
        "torch held to 16.00 GiB on the GPU: the 18.00 GiB free less 3.0, and what it holds;"
        " free on the whole card by NVML (its driver reports 20.00 GiB free to this process)"
    )
    # Before a unit, torch holding 15 GiB of its 16: the driver still reports 20 GiB free to
    # the process, and the card has 1 GiB left, as much as torch may still take: nothing. One
    # byte less on the card, another program's: refused, on the card's figure.
    held(monkeypatch, 20 * GIB, reserved=15 * GIB)
    nvml.cards[NAMED] = GIB
    memory.check("cuda:0")
    nvml.cards[NAMED] = GIB - 1
    with pytest.raises(torch.OutOfMemoryError) as refusal:
        memory.check("cuda:0")
    assert str(refusal.value) == (
        "the GPU has 1.00 GiB free on the whole card by NVML (its driver reports 20.00 GiB free"
        " to this process) where the run may still take 1.00: other programs hold the rest of"
        " the card's memory, or the run's own work took more than the 3.0 GiB left to it"
        " outside torch's allocator"
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
        "the GPU has 1.00 GiB free on the whole card by NVML (its driver reports 0.00 GiB free"
        " to this process) where the run may still take 1.00: other programs hold"
    )
    # Every reading opened NVML and shut it down, and none was warned of.
    assert nvml.open == 0 and nvml.calls.count("nvmlInit") == 8  # the probe's first
    assert [record.levelname for record in caplog.records].count("WARNING") == 2


def test_the_card_read_is_the_device_s(monkeypatch: pytest.MonkeyPatch) -> None:
    # The limit and the check read the card of the device they are given, by CUDA's index for
    # it; a device that names none is CUDA's first, the one the plan's figure is read on.
    startup(monkeypatch, Card(30 * GIB, spills=True), Nvml({NAMED: 28 * GIB}))
    asked: list[int] = []
    monkeypatch.setattr(memory, "card_free", lambda index: asked.append(index) or 28 * GIB)
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
    # before the card was read, and say which. Warned of once a run, with the reason; the
    # readings after it, one before every unit, say it at debug level.
    caplog.set_level(logging.DEBUG, logger="seedvr2x")
    startup(monkeypatch, Card(30 * GIB, spills=True))
    caplog.clear()
    assert memory.reading(30 * GIB) == memory.Reading(30 * GIB, None, round(24.7 * GIB), True)
    unread = (
        f"the whole card's free memory can't be read through NVML ({memory.NVML}: cannot open"
        " shared object file): what the driver reports free to this process is taken alone,"
        " which leaves other programs' memory out while the card has room, and the run is held"
        " 3.0 GiB under it"
    )
    assert caplog.messages == [
        unread,
        "the plan counts on 24.70 GiB of the GPU's memory, of the 30.00 GiB free as its driver"
        " reports it to this process (the whole card's, NVML's, wasn't read): 3.0 GiB are left"
        " to what torch doesn't count on the card, 2.3 to what torch's own allocator holds"
        " beyond a peak",
    ]
    limits = held(monkeypatch, 20 * GIB, reserved=GIB)
    assert memory.hold("cuda:0") == 18 * GIB and limits == [18 / 32]
    assert caplog.messages[2:] == [
        unread,
        "torch held to 18.00 GiB on the GPU: the 20.00 GiB free less 3.0, and what it holds;"
        " free as its driver reports it to this process (the whole card's, NVML's, wasn't"
        " read)",
    ]
    held(monkeypatch, 3 * GIB, reserved=15 * GIB)
    memory.check("cuda:0")
    assert [record.levelname for record in caplog.records] == [
        *("WARNING", "DEBUG", "DEBUG", "INFO", "DEBUG")
    ]
    # An NVML that answers later in the run is taken from then on: the check asks every time.
    nvml = Nvml({NAMED: 2 * GIB})
    libraries(monkeypatch, Card(30 * GIB, spills=True), nvml)
    with pytest.raises(torch.OutOfMemoryError, match=r"2\.00 GiB free on the whole card by NVML"):
        memory.check("cuda:0")
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
    # (runtime/planner.py): once in a run's log, not twice. usable is the number alone.
    startup(monkeypatch, Card(30 * GIB, spills=True), Nvml({NAMED: 28 * GIB}))
    caplog.clear()
    counted = 28 * GIB - memory.HELD_BACK - memory.RESERVE
    assert memory.reading(30 * GIB) == memory.Reading(30 * GIB, 28 * GIB, counted, True)
    assert memory.usable(30 * GIB) == counted
    assert caplog.text == ""


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


# What torch's allocator takes of its two variables, each case in a process of its own, which
# needs no card: the variables set, and the backend torch then names.
TORCH_READS = (
    ({"PYTORCH_CUDA_ALLOC_CONF": ""}, "native"),
    ({"PYTORCH_CUDA_ALLOC_CONF": "", "PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync"}, "native"),
    (
        {"PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync", "PYTORCH_ALLOC_CONF": ""},
        "cudaMallocAsync",
    ),
)


def test_torch_takes_the_first_variable_set_even_empty() -> None:
    # What startup's handling of an empty setting rests on (runtime/memory.py), asked of torch
    # itself: an empty setting names no allocator, torch's own then; and the first variable set
    # is the one read, empty or not, so that an empty PYTORCH_CUDA_ALLOC_CONF hides a
    # PYTORCH_ALLOC_CONF that names cudaMallocAsync.
    import subprocess
    import sys

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
        ({"PYTORCH_CUDA_ALLOC_CONF": ""}, {"PYTORCH_CUDA_ALLOC_CONF": "backend:native"}, False),
        ({"PYTORCH_ALLOC_CONF": ""}, {"PYTORCH_CUDA_ALLOC_CONF": "backend:native"}, False),
        (
            {"PYTORCH_CUDA_ALLOC_CONF": "", "PYTORCH_ALLOC_CONF": ""},
            {"PYTORCH_CUDA_ALLOC_CONF": "backend:native"},
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
    assert startup(monkeypatch, Card(30 * GIB, spills=True)) == "backend:native"
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
    # The line before it names torch's own allocator only where the run takes it.
    held_by = "torch limited to" if warned else "torch's own allocator, limited to"
    assert caplog.messages[1].startswith(f"seedvr2x holds itself to the card there: {held_by}")
    limits = held(monkeypatch, 20 * GIB)
    assert memory.hold("cuda:0") == 17 * GIB and limits == [17 / 32]
    held(monkeypatch, GIB)  # 1 GiB free where torch may take 17 more
    if warned:
        loaded = libraries(monkeypatch, Card(30 * GIB, spills=True), Nvml({NAMED: GIB}))
        memory.check("cuda:0")
        assert not loaded  # nothing checked, nothing read
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
        "the GPU has 3.00 GiB free on the whole card by NVML (its driver reports 2.00 GiB free"
        " to this process) where the run may still take 3.00: other programs hold the rest of"
        " the card's memory, or the run's own work took more than the 3.0 GiB left to it"
        " outside torch's allocator"
    )
    # A new run asks again, and starts unheld: nothing read, of the card either, after its
    # probe.
    startup(monkeypatch, Card(30 * GIB, spills=True), nvml)
    read = len(nvml.calls)
    memory.check("cuda:0")
    assert len(nvml.calls) == read


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

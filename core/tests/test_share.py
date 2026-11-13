"""A card another program uses under Linux's driver, held as under WSL2 (seedvr2x/memory.py:
share, startup, early; DESIGN.md, Allocator, decided on 2026-11-11): NVML's answers stood in
(processes listed on the card, compute or graphics, both, none, NVML unread; a display active on
it, which is no share since 2026-11-12 and never asked), and what the start, the plan's reading and
a resume's early reading make of them. The stand-ins
for CUDA's and NVML's libraries are tests/test_memory.py's."""

import ctypes
import logging
import os
from typing import Any

import pytest
from test_memory import NAMED, Card, Nvml, libraries

from seedvr2x import memory
from seedvr2x.memory import GIB, SUCCESS

# memory.share as the module has it, before library_fixtures.no_driver_asked puts a card no
# other program uses in its place for every test.
SHARE = memory.share
HELD = "backend:native,expandable_segments:True"
# The warnings of a shared card, written out whole (not built from memory's constants: a change
# of wording must fail here): the hold's, the shipped figures', then the share's.
HOLDS_LINE = (
    "seedvr2x holds itself to the card's own memory: torch limited to the memory free"
    " less 0.5 GiB, and each step planned with a reserve by its own peak"
)
SHARED_BY_A_SERVER = (
    "the GPU is shared: another program on it (Xorg); held so that the rest of the card keeps room"
    " to grow into"
)
ALONE = (
    "no other program on the GPU at the start (none listed by NVML): not held, its driver refusing"
    " what the card can't hold; torch's own allocator with expandable segments"
)


class Listing(Nvml):
    """NVML with the card's display and its processes: `display`, whether one is active, which
    seedvr2x never asks (nvmlDeviceGetDisplayActive fails the test); `compute` and `graphics`, the
    processes' identifiers each list names; `grow`, identifiers added to the compute list once
    its count has been asked, as a process starting between NVML's two calls; `without`, a
    function this NVML lacks (an older library)."""

    def __init__(
        self,
        display: bool = False,
        compute: tuple[int, ...] = (),
        graphics: tuple[int, ...] = (),
        grow: tuple[int, ...] = (),
        without: str | None = None,
        failing: str | None = None,
    ) -> None:
        super().__init__({NAMED: 20 * GIB}, failing)
        self.display = display
        self.lists = {"compute": list(compute), "graphics": list(graphics)}
        self.grow, self.without = list(grow), without
        self.filled: list[tuple[str, int]] = []

    def __getattribute__(self, name: str) -> Any:
        if name == object.__getattribute__(self, "without"):
            raise AttributeError(f"undefined symbol: {name}")
        return object.__getattribute__(self, name)

    def nvmlDeviceGetDisplayActive(self, handle: ctypes.c_void_p, active: Any) -> int:
        pytest.fail("whether a display is active is asked")

    def nvmlDeviceGetComputeRunningProcesses_v3(self, handle: Any, count: Any, infos: Any) -> int:
        return self._listed("compute", count, infos)

    def nvmlDeviceGetGraphicsRunningProcesses_v3(self, handle: Any, count: Any, infos: Any) -> int:
        return self._listed("graphics", count, infos)

    def _listed(self, kind: str, count: Any, infos: Any) -> int:
        """As NVML answers (nvml.h): with no buffer or one too small, the count needed and
        NVML_ERROR_INSUFFICIENT_SIZE (SUCCESS where none is listed); else the processes."""
        failed = self.call(f"{kind} processes")
        if failed:
            return failed
        pids = self.lists[kind]
        if infos is None:
            count._obj.value = len(pids)
            if kind == "compute" and self.grow:
                pids += self.grow
                self.grow = []
            return SUCCESS if not count._obj.value else memory.INSUFFICIENT_SIZE
        self.filled.append((kind, len(infos)))
        if count._obj.value < len(pids):
            count._obj.value = len(pids)
            return memory.INSUFFICIENT_SIZE
        for number, pid in enumerate(pids):
            infos[number].pid, infos[number].usedGpuMemory = pid, 300 << 20
        count._obj.value = len(pids)
        return SUCCESS


@pytest.fixture
def named(monkeypatch: pytest.MonkeyPatch) -> None:
    """Processes named by their identifiers alone: 101 Xorg, 102 gnome-shell, 103 firefox,
    104 python3; others unnamed, as a process of another PID namespace is."""
    names = {101: "Xorg", 102: "gnome-shell", 103: "firefox", 104: "python3"}
    monkeypatch.setattr(memory, "_program", lambda pid: names.get(pid, ""))


def read(monkeypatch: pytest.MonkeyPatch, nvml: Any) -> memory.Share:
    libraries(monkeypatch, Card(30 * GIB, spills=False), nvml)
    return SHARE()


def test_a_card_no_other_program_uses(monkeypatch: pytest.MonkeyPatch, named: None) -> None:
    nvml, card = Listing(), Card(30 * GIB, spills=False)
    libraries(monkeypatch, card, nvml)
    found = SHARE()
    assert found == memory.Share() and not found.shared
    assert found.said() == "no other program on it"
    # NVML initialised once and shut down; no CUDA context made.
    assert nvml.open == 0 and nvml.calls[-1] == "nvmlShutdown"
    assert card.calls == ["cuInit", "cuDeviceGet", "cuDeviceGetUuid"]


def test_a_display_active_with_no_program_listed_is_no_share(
    monkeypatch: pytest.MonkeyPatch, named: None
) -> None:
    # The 96 GB card with no program on it and no display server running read a display active
    # (its text console, inferred: DESIGN.md, Allocator, decided on 2026-11-12): not asked, the
    # card no other program's (Listing's nvmlDeviceGetDisplayActive fails a test that asks).
    nvml = Listing(display=True)
    found = read(monkeypatch, nvml)
    assert found == memory.Share() and not found.shared
    assert found.said() == "no other program on it"
    # A desktop is seen by its display server's graphics context.
    found = read(monkeypatch, Listing(display=True, graphics=(101,)))
    assert found.shared and found.said() == "another program on it (Xorg)"


def test_shared_by_a_process_compute_or_graphics(
    monkeypatch: pytest.MonkeyPatch, named: None
) -> None:
    found = read(monkeypatch, Listing(compute=(104,)))
    assert found.shared and found.processes == ((104, "python3"),)
    assert found.said() == "another program on it (python3)"
    found = read(monkeypatch, Listing(graphics=(101, 102)))
    assert found.said() == "2 other programs on it (Xorg, gnome-shell)"


def test_shared_by_both_each_process_once_this_one_left_out(
    monkeypatch: pytest.MonkeyPatch, named: None
) -> None:
    # A process in both lists is one; this process, where it has a context already (the nodes'
    # ComfyUI), is no other program; past NAMED, the rest counted; an unnamed one by identifier.
    own = os.getpid()
    nvml = Listing(display=True, compute=(103, own, 900), graphics=(101, 102, 103))
    found = read(monkeypatch, nvml)
    assert [pid for pid, _ in found.processes] == [101, 102, 103, 900]
    assert found.said() == "4 other programs on it (Xorg, gnome-shell, firefox, 1 more)"
    found = read(monkeypatch, Listing(compute=(own, 900)))
    assert found.said() == "another program on it (process 900)"
    assert not read(monkeypatch, Listing(compute=(own,), graphics=(own,))).shared


def test_a_list_asked_for_its_count_first_and_again_where_it_grew(
    monkeypatch: pytest.MonkeyPatch, named: None
) -> None:
    nvml = Listing(compute=(101,), grow=tuple(range(200, 212)))
    found = read(monkeypatch, nvml)
    assert len(found.processes) == 13
    # Room for the count and 8 more (9): too little for 13, asked again with room for 21.
    assert nvml.filled == [("compute", 9), ("compute", 21)]
    assert nvml.calls.count("compute processes") == 3


def test_a_list_that_keeps_growing_or_fails_is_unread(
    monkeypatch: pytest.MonkeyPatch, named: None
) -> None:
    # A list grown at every call, past the four asks, and a list failing with a status of its
    # own: NVML unread, the status said, taken for shared.
    class Growing(Listing):
        def _listed(self, kind: str, count: Any, infos: Any) -> int:
            if kind == "compute":
                self.lists[kind] += list(
                    range(1000 + len(self.lists[kind]), 1030 + len(self.lists[kind]))
                )
            return super()._listed(kind, count, infos)

    nvml = Growing(compute=(101,))
    found = read(monkeypatch, nvml)
    assert found.unread == "nvmlDeviceGetComputeRunningProcesses_v3: 7"
    assert nvml.calls.count("compute processes") == 2 * 5 and nvml.open == 0

    class Refusing(Listing):
        def _listed(self, kind: str, count: Any, infos: Any) -> int:
            return 15 if kind == "graphics" else super()._listed(kind, count, infos)

    found = read(monkeypatch, Refusing())
    assert found.unread == "nvmlDeviceGetGraphicsRunningProcesses_v3: 15" and found.shared


@pytest.mark.parametrize(
    ("nvml", "reason"),
    [
        (None, "cannot open shared object file"),
        (Listing(failing="nvmlInit"), "nvmlInit: 999"),
        (Listing(failing="nvmlDeviceGetHandleByUUID"), "nvmlDeviceGetHandleByUUID, GPU-"),
        (
            Listing(failing="graphics processes"),
            "nvmlDeviceGetGraphicsRunningProcesses_v3: 999",
        ),
        (
            Listing(without="nvmlDeviceGetComputeRunningProcesses_v3"),
            "undefined symbol: nvmlDeviceGetComputeRunningProcesses_v3",
        ),
    ],
)
def test_nvml_unread_is_taken_for_shared(
    monkeypatch: pytest.MonkeyPatch, named: None, nvml: Any, reason: str
) -> None:
    found = read(monkeypatch, nvml)
    assert found.shared and found.unread is not None and reason in found.unread
    assert found.said().startswith("maybe, NVML unread (")
    if nvml is not None:
        # Asked twice, NVML shut down each time it was initialised.
        assert nvml.calls.count("nvmlInit") == 2 and nvml.open == 0


def test_nvml_failing_once_is_asked_again(monkeypatch: pytest.MonkeyPatch, named: None) -> None:
    nvml = Listing(graphics=(101,), failing="graphics processes")
    nvml.times = 1
    found = read(monkeypatch, nvml)
    assert found == memory.Share(processes=((101, "Xorg"),))


def start(monkeypatch: pytest.MonkeyPatch, nvml: Any) -> str:
    """memory.startup under Linux's driver (no WSL2 device), NVML `nvml` reading the card."""
    monkeypatch.setattr(memory, "share", SHARE)
    monkeypatch.setattr(memory, "probe", lambda *args: pytest.fail("Linux's driver is asked"))
    libraries(monkeypatch, Card(30 * GIB, spills=False), nvml)
    return memory.startup()


@pytest.mark.parametrize(
    ("nvml", "said"),
    [
        (Listing(graphics=(101,)), "another program on it (Xorg)"),
        (Listing(compute=(104,)), "another program on it (python3)"),
        (
            Listing(display=True, graphics=(101, 102)),
            "2 other programs on it (Xorg, gnome-shell)",
        ),
    ],
)
def test_a_shared_card_is_held_as_under_wsl2(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    named: None,
    nvml: Any,
    said: str,
) -> None:
    caplog.set_level(logging.DEBUG, logger=memory.logger.name)
    assert start(monkeypatch, nvml) == HELD
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == HELD
    assert memory.spills() and memory.shared() == said
    # What seedvr2x does, then why, as a lending driver's lines are said; the finding at debug.
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warned == [
        HOLDS_LINE,
        f"the GPU is shared: {said}; held so that the rest of the card keeps room to grow into",
    ]
    finding = caplog.records[-1]
    assert finding.levelno == logging.DEBUG and finding.getMessage().startswith(
        "the GPU's share by NVML: processes other than this one"
    )
    # The plan's reading: held, by the card's figure (NVML's 20 GiB, the smaller), less 0.5 GiB,
    # torch's own allocator with its segments, the share carried for the card line and out of
    # the repr (the plan matrix's text).
    found = memory.reading(30 * GIB)
    assert found.held and found.card == 20 * GIB and found.left == memory.HELD_BACK
    assert found.allocator == memory.SEGMENTS and found.counted == 20 * GIB - memory.HELD_BACK
    assert found.shared == said and "shared" not in repr(found)


def test_the_display_server_s_line_whole(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, named: None
) -> None:
    caplog.set_level(logging.INFO, logger=memory.logger.name)
    start(monkeypatch, Listing(display=True, graphics=(101,)))
    assert caplog.messages == [HOLDS_LINE, SHARED_BY_A_SERVER]


def test_nvml_unread_at_the_start_holds_the_run_saying_why(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, named: None
) -> None:
    caplog.set_level(logging.INFO, logger=memory.logger.name)
    assert start(monkeypatch, Listing(failing="nvmlInit")) == HELD
    assert memory.spills()
    assert caplog.messages[1] == (
        "the GPU is shared or may be: NVML can't say whether another program uses it"
        " (nvmlInit: 999), so it is taken for shared, and held"
    )
    # Held 0.5 GiB under the driver's figure, the card's own under Linux's driver, where WSL2's
    # unread card takes 3 (memory.HELD_BACK_UNREAD): the reading and its words.
    monkeypatch.setattr(memory, "_twice", lambda index: "nvmlInit: 999")
    found = memory.reading(30 * GIB)
    assert found.held and found.card is None and found.left == memory.HELD_BACK
    assert found.shared == "maybe, NVML unread (nvmlInit: 999)"
    assert (
        "what the driver reports free to this process is taken alone, which is the whole card's"
        " under Linux's driver: the plan counts on that figure, torch is held 0.5 GiB under it"
    ) in caplog.messages[-1]


def test_a_card_no_other_program_uses_takes_the_segments_unheld_and_says_so(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, named: None
) -> None:
    # torch's own allocator with expandable segments where the driver refuses too, in
    # cudaMallocAsync's place (DESIGN.md, Allocator, decided on 2026-11-13), the run
    # neither limited nor checked, the start's line naming the allocator.
    caplog.set_level(logging.DEBUG, logger=memory.logger.name)
    assert start(monkeypatch, Listing()) == HELD
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == HELD
    assert not memory.spills() and memory.shared() is None
    assert [(r.levelno, r.getMessage()) for r in caplog.records] == [(logging.INFO, ALONE)]
    # The plan of a driver that refuses, as before: nothing held, nothing read of the card, no
    # reserve beyond the planner's margin (PLAIN on a reading not held).
    assert memory.reading(30 * GIB) == memory.Reading(30 * GIB, None, 30 * GIB, False)
    assert memory.hold("cuda:0") is None
    memory.check("cuda:0")  # nothing held: nothing checked


@pytest.mark.parametrize(
    ("setting", "words"),
    [
        ("backend:cudaMallocAsync", "torch on cudaMallocAsync"),
        ("backend:native", "torch's own allocator without expandable segments"),
        ("expandable_segments:True", "torch's own allocator with expandable segments"),
    ],
)
def test_the_user_s_allocator_stands_on_a_card_no_other_program_uses(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    named: None,
    setting: str,
    words: str,
) -> None:
    # The environment's setting is torch's, as on a shared card; the start's one line names the
    # allocator it gives, and nothing more is said: the run isn't held, its plan a refusing
    # driver's whatever the allocator.
    caplog.set_level(logging.DEBUG, logger=memory.logger.name)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", setting)
    start(monkeypatch, Listing())
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == setting
    assert caplog.messages == [
        ALONE.replace("torch's own allocator with expandable segments", words)
    ]
    assert memory.reading(30 * GIB) == memory.Reading(30 * GIB, None, 30 * GIB, False)


@pytest.mark.parametrize(
    ("setting", "allocator", "said"),
    [
        ("backend:cudaMallocAsync", memory.POOL, "torch's limit counts what torch has allocated"),
        ("backend:native", memory.PLAIN, "torch's own without expandable segments"),
        ("expandable_segments:True", memory.SEGMENTS, None),
    ],
)
def test_the_user_s_allocator_stands_on_a_shared_card(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    named: None,
    setting: str,
    allocator: str,
    said: str | None,
) -> None:
    # As under WSL2: the environment's setting is torch's, the run held all the same, the plan
    # made for that allocator, and a setting that holds less said.
    caplog.set_level(logging.INFO, logger=memory.logger.name)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", setting)
    start(monkeypatch, Listing(graphics=(101,)))
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == setting
    assert memory.spills() and memory.reading(30 * GIB).allocator == allocator
    environment = [m for m in caplog.messages if m.startswith("the allocator is the environment")]
    assert len(environment) == (said is not None)
    if said is not None:
        assert said in environment[0]


def test_a_resume_s_early_reading_takes_the_share(
    monkeypatch: pytest.MonkeyPatch, named: None
) -> None:
    # Before torch and before startup, NVML alone: a shared card's reading is held, as the run's
    # will be; one no other program uses is a refusing driver's, as before.
    monkeypatch.setattr(memory, "share", SHARE)
    libraries(monkeypatch, Card(30 * GIB, spills=False), Listing(graphics=(101,)))
    found = memory.early(20 * GIB, used=3 * GIB)
    assert found.held and found.shared == "another program on it (Xorg)"
    assert found.nvml == 20 * GIB and found.reported == 20 * GIB - memory.CONTEXT
    assert found.counted == found.reported - memory.HELD_BACK
    libraries(monkeypatch, Card(30 * GIB, spills=False), Listing(display=True))
    found = memory.early(20 * GIB, used=3 * GIB)
    assert not found.held and found.shared is None
    assert found == memory.Reading(
        *(20 * GIB - memory.CONTEXT, None, 20 * GIB - memory.CONTEXT, False),
        nvml=20 * GIB,
        nvml_used=3 * GIB,
    )


def test_behind_wsl2_and_on_windows_the_share_is_not_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The driver that lends decides there; the share is Linux's driver's question.
    monkeypatch.setattr(memory, "share", lambda index=0: pytest.fail("the share is asked"))
    monkeypatch.setattr(memory, "paravirtual", lambda: True)
    monkeypatch.setattr(memory, "probe", lambda *args: None)
    memory.startup()
    assert memory.spills() and memory.shared() is None
    assert memory.early(20 * GIB).shared is None
    monkeypatch.setattr(memory, "paravirtual", lambda: False)
    monkeypatch.setattr(memory.sys, "platform", "win32")
    monkeypatch.setattr(memory.ctypes, "CDLL", lambda name: Nvml({NAMED: 28 * GIB}))
    memory.startup()
    assert memory.spills() and memory.shared() is None


def test_a_platform_other_than_linux_asks_nothing_and_is_not_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The share is Linux's driver's question: elsewhere (macOS, a port to come) nothing is asked
    # and nothing held, as before; the default allocator is Windows itself's alone to differ
    # (memory._default; macOS has no CUDA to read it).
    monkeypatch.setattr(memory, "share", lambda index=0: pytest.fail("the share is asked"))
    monkeypatch.setattr(memory.sys, "platform", "darwin")
    assert memory.startup() == HELD
    assert not memory.spills() and memory.shared() is None
    assert not memory.early(20 * GIB).held


def test_a_new_start_forgets_the_share(monkeypatch: pytest.MonkeyPatch, named: None) -> None:
    start(monkeypatch, Listing(graphics=(101,)))
    assert memory.shared() is not None
    start(monkeypatch, Listing())
    assert memory.shared() is None and not memory.spills()

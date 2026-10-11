"""A job's plan as the run and the manifest take it (runtime/plan.py; DESIGN.md, Memory planner),
a resume's reading of it, the pins the command line reads and the memory it finds free for the
planner (runtime/planner.py, whose own tests are test_planner.py's)."""

import argparse
import datetime
import logging
import weakref
from pathlib import Path

import pytest
import torch

from seedvr2x import cli
from seedvr2x.runtime import resume
from seedvr2x.runtime.job import JobError
from seedvr2x.runtime.plan import OVERLAP, Plan, Tiles
from seedvr2x.runtime.planner import PlanError


def test_record() -> None:
    # What shapes the output at the top, what doesn't under memory, which a resume doesn't
    # compare, nor the timings the time estimates will record.
    assert OVERLAP == 64 and Tiles(1024) == Tiles(1024, 64)
    assert Plan(None).record() == {
        "window": None,
        "encode_tiles": None,
        "decode_tiles": None,
        "compile_dit": False,
        "compile_vae": False,
        "memory": {
            "swapped_blocks": 0,
            "resident_encode": None,
            "resident_decode": None,
            "vae_in_dit": True,
        },
    }
    plan = Plan(12, Tiles(1536), Tiles(1024), 20, 0, 4, vae_in_dit=False)
    record = plan.record()
    assert record["window"] == 12
    assert record["encode_tiles"] == {"size": 1536, "overlap": 64}
    assert record["decode_tiles"] == {"size": 1024, "overlap": 64}
    assert record["memory"] == {
        "swapped_blocks": 20,
        "resident_encode": 0,
        "resident_decode": 4,
        "vae_in_dit": False,
    }
    assert resume.UNCOMPARED["plan"] == ("memory", "timings")
    one = {"plan": plan.record()}
    other = {"plan": {**Plan(12, Tiles(1536), Tiles(1024)).record(), "timings": {"encode": 1.5}}}
    assert resume.differences(one, other) == []
    assert resume.differences(one, {"plan": Plan(12, Tiles(1536), Tiles(512)).record()}) == [
        "plan.decode_tiles.size: 1024 -> 512"
    ]
    assert resume.differences(one, {"plan": Plan(11, Tiles(1536), Tiles(1024)).record()}) == [
        "plan.window: 12 -> 11"
    ]


def test_recorded() -> None:
    # A resume takes what shapes the output as the manifest records it, and plans the rest
    # again: where the weights were isn't read, nor the timings.
    assert issubclass(PlanError, JobError)  # refused as a job is, before the run
    plan = Plan(12, Tiles(1536), Tiles(1024, 128), 20, 0, 4, vae_in_dit=False)
    assert Plan.recorded(plan.record()) == Plan(12, Tiles(1536), Tiles(1024, 128))
    assert Plan.recorded(Plan(None).record()) == Plan(None)
    record = {**Plan(7).record(), "memory": {"swapped_blocks": "all"}, "timings": {"window": 1.5}}
    assert Plan.recorded(record) == Plan(7)
    del record["memory"]
    assert Plan.recorded(record) == Plan(7)
    compiled = {**Plan(5).record(), "compile_dit": True}
    assert Plan.recorded(compiled) == Plan(5, compile_dit=True)


@pytest.mark.parametrize(
    ("change", "said"),
    [
        ({"window": "six"}, "its window, 'six'"),
        ({"window": 0}, "its window, 0"),
        ({"window": True}, "its window, True"),
        ({"window": 5.0}, "its window, 5.0"),
        ({"encode_tiles": 1024}, "its encode_tiles, 1024"),
        ({"decode_tiles": {"size": 1024}}, "its decode_tiles, {'size': 1024}"),
        (
            {"decode_tiles": {"size": 64, "overlap": 64}},
            "its decode_tiles, {'size': 64, 'overlap': 64}",
        ),
        ({"compile_vae": None}, "its compile_vae, None"),
        ({"compile_dit": 0}, "its compile_dit, 0"),
    ],
)
def test_not_a_recorded_plan(change: dict[str, object], said: str) -> None:
    with pytest.raises(JobError) as refused:
        Plan.recorded({**Plan(5).record(), **change})
    assert str(refused.value) == f"not a plan as seedvr2x records one: {said}"


def test_no_recorded_plan() -> None:
    for record in (None, [5], "plan"):
        with pytest.raises(JobError, match="not a plan as seedvr2x records one"):
            Plan.recorded(record)
    with pytest.raises(JobError, match="its compile_dit, None"):
        Plan.recorded({"window": 5})


def test_free_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    # The memory free on the job's GPU as the driver reports it, read once a first tensor has
    # been allocated on the device and freed, the allocator's cache emptied: in that order, so
    # that no reading comes before the process's first allocation, nor counts the tensor. The
    # tensor is gone when the cache is emptied, which gives back nothing of one still held.
    did: list[object] = []
    made: list[weakref.ref[torch.Tensor]] = []
    empty = torch.empty

    def allocate(*size: int, device: torch.device) -> torch.Tensor:
        did.append(("tensor", device))
        first = empty(*size, device=device)
        made.append(weakref.ref(first))
        return first

    def read(device: torch.device) -> tuple[int, int]:
        did.append(("read", device))
        return 94_000_000_000, 96_000_000_000

    monkeypatch.setattr(torch, "empty", allocate)
    monkeypatch.setattr(
        torch.cuda, "empty_cache", lambda: did.append(("emptied", [ref() for ref in made]))
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", read)
    device = torch.device("cpu")
    assert cli._free_memory(device) == 94_000_000_000  # pyright: ignore[reportPrivateUsage]
    assert did == [("tensor", device), ("emptied", [None]), ("read", device)]


def test_host_available(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # The kernel's MemAvailable, in bytes; None, said at debug level, when it can't be read.
    caplog.set_level(logging.DEBUG, logger="seedvr2x")
    available = cli._host_available  # pyright: ignore[reportPrivateUsage]
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       65806340 kB\nMemAvailable:   41234567 kB\nCached: 1 kB\n")
    assert available(meminfo) == 41_234_567 * 1024
    found = available()  # this machine's own
    assert found is not None and found > 0
    assert caplog.text == ""
    for text, reason in (
        ("MemTotal:       65806340 kB\n", "no MemAvailable in it"),
        ("MemAvailable:   lots kB\n", "invalid literal for int"),
        ("MemAvailable:\n", "list index out of range"),
    ):
        caplog.clear()
        meminfo.write_text(text)
        assert available(meminfo) is None
        assert f"{meminfo}: {reason}" in caplog.text
        assert "the plan's system memory isn't checked" in caplog.text
    caplog.clear()
    assert available(tmp_path / "none") is None
    assert "No such file or directory" in caplog.text


@pytest.mark.parametrize("text", ["0", "512", "576", "1024", "4096"])
def test_tile(text: str) -> None:
    assert cli._tile(text) == int(text)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("text", ["64", "448", "500", "1000", "-512", "1024.0", "1k", ""])
def test_not_a_tile(text: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="multiple of 64 from 512 up"):
        cli._tile(text)  # pyright: ignore[reportPrivateUsage]


def test_count_and_until() -> None:
    assert [cli._count(text) for text in ("0", "7", "36")] == [0, 7, 36]  # pyright: ignore[reportPrivateUsage]
    for text in ("-1", "1.5", "all", ""):
        with pytest.raises(argparse.ArgumentTypeError):
            cli._count(text)  # pyright: ignore[reportPrivateUsage]
    assert cli._until("07:30") == datetime.time(7, 30)  # pyright: ignore[reportPrivateUsage]
    assert cli._until("7:05") == datetime.time(7, 5)  # pyright: ignore[reportPrivateUsage]
    assert cli._until("23:59") == datetime.time(23, 59)  # pyright: ignore[reportPrivateUsage]
    for text in ("24:00", "12:60", "7", "07:3", "7h30", "07:30:00", ""):
        with pytest.raises(argparse.ArgumentTypeError, match="not a time of day"):
            cli._until(text)  # pyright: ignore[reportPrivateUsage]

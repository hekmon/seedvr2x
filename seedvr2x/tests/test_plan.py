"""A job's plan as the run and the manifest take it (runtime/plan.py; DESIGN.md, Memory planner),
the pins the command line reads, and milestone 3's seam: the stand-in planner, which takes the
pins as the plan and refuses those whose work isn't built yet (runtime/planner.py)."""

import argparse
import datetime

import pytest

from seedvr2x import cli
from seedvr2x.runtime import planner, resume
from seedvr2x.runtime.job import JobError
from seedvr2x.runtime.plan import OVERLAP, Plan, Tiles
from seedvr2x.runtime.planner import Pins, PlanError


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


def test_stand_in_planner() -> None:
    # The pins as the plan; untiled and unswapped pins (0) are what the run does already.
    assert planner.plan(Pins()) == Plan(None)
    assert planner.plan(Pins(window=7)) == Plan(7)
    assert planner.plan(Pins(5, 0, 0, 0)) == Plan(5)
    assert issubclass(PlanError, JobError)  # refused as a job is, before the run
    for pins, message in [
        (Pins(encode_tile=1024), "--encode-tile 1024: the tiled passes aren't built yet"),
        (Pins(decode_tile=512), "--decode-tile 512: the tiled passes aren't built yet"),
        (Pins(swap_blocks=36), "--swap-blocks 36: BlockSwap isn't built yet"),
    ]:
        with pytest.raises(PlanError) as refused:
            planner.plan(pins)
        assert str(refused.value) == message


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

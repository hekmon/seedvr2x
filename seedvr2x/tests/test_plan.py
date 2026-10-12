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
from seedvr2x.runtime.planner import GIB, PlanError


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
            "scratch_allowance": 0,
        },
    }
    plan = Plan(12, Tiles(1536), Tiles(1024), 20, 0, 4, vae_in_dit=False, scratch_allowance=3 * GIB)
    record = plan.record()
    assert record["window"] == 12
    assert record["encode_tiles"] == {"size": 1536, "overlap": 64}
    assert record["decode_tiles"] == {"size": 1024, "overlap": 64}
    # What a tiled shot's scratch may take of system memory, in bytes, with where the weights
    # are: for information, as they are (DESIGN.md, Memory planner, Host RAM: "it shapes no bit").
    assert record["memory"] == {
        "swapped_blocks": 20,
        "resident_encode": 0,
        "resident_decode": 4,
        "vae_in_dit": False,
        "scratch_allowance": 3 * GIB,
    }
    assert resume.UNCOMPARED["plan"] == ("memory", "timings")
    one = {"plan": plan.record()}
    other = {"plan": {**Plan(12, Tiles(1536), Tiles(1024)).record(), "timings": {"encode": 1.5}}}
    assert resume.differences(one, other) == []
    # Nor what the plan was made in, recorded there with its predicted peaks (cli._run;
    # planner.Peaks.record): the memory it counted on, what the driver reported free and what
    # the whole card had, read or not, and the attention backend of that day's run.
    read = {"free": 22 * GIB, "driver_free": 30 * GIB, "card_free": 28 * GIB, "attention": "sdpa"}
    unread = {"free": 30 * GIB, "driver_free": 30 * GIB, "card_free": None}
    unread["attention"] = "flash_attn_2"
    for memory in (read, unread):
        made = {"plan": plan.record()}
        made["plan"]["memory"].update(memory)
        assert resume.differences(one, made) == resume.differences(made, one) == []
    assert resume.differences(one, {"plan": Plan(12, Tiles(1536), Tiles(512)).record()}) == [
        "plan.decode_tiles.size: 1024 -> 512"
    ]
    assert resume.differences(one, {"plan": Plan(11, Tiles(1536), Tiles(1024)).record()}) == [
        "plan.window: 12 -> 11"
    ]


def test_recorded() -> None:
    # A resume takes what shapes the output as the manifest records it, and plans the rest
    # again: where the weights were isn't read, nor the scratch's allowance, nor the timings.
    assert issubclass(PlanError, JobError)  # refused as a job is, before the run
    plan = Plan(12, Tiles(1536), Tiles(1024, 128), 20, 0, 4, vae_in_dit=False, scratch_allowance=7)
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
    # The memory free on the job's GPU as the driver reports it, read on one footing whatever
    # ran before. While a first tensor is held on the device, so that no reading comes before
    # the process's first allocation. With the results of a tiny convolution and of a tiny
    # matrix product held beside it, both in float32 on the device, so that a run without a shot
    # detector has loaded the libraries a detector's run leaves loaded (with the tensor alone, a
    # card emulated at 24 GB read 22.54 GiB free after the detector and 22.56 on a resume). And
    # the allocator's cache is never emptied, which would give back the pool's first reservation
    # for the next allocation to take again (94.4308 GiB read that way on the bare 96 GB card,
    # against 94.2609 with the tensor held). All three go once the reading is taken. On the CPU
    # here, as the stand-in's runs take it (test_cli_run.py).
    did: list[object] = []
    made: list[weakref.ref[torch.Tensor]] = []
    empty, conv3d, mm = torch.empty, torch.nn.functional.conv3d, torch.mm

    def held(what: str, result: torch.Tensor) -> torch.Tensor:
        did.append((what, result.device, result.dtype))
        made.append(weakref.ref(result))
        return result

    def allocate(*size: int, device: torch.device) -> torch.Tensor:
        return held("tensor", empty(*size, device=device))

    def convolve(volume: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        assert volume.dim() == kernel.dim() == 5
        return held("convolution", conv3d(volume, kernel))

    def multiply(one: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        return held("product", mm(one, other))

    def read(device: torch.device) -> tuple[int, int]:
        did.append(("read", device, [ref() is not None for ref in made]))
        return 94_000_000_000, 96_000_000_000

    monkeypatch.setattr(torch, "empty", allocate)
    monkeypatch.setattr(torch.nn.functional, "conv3d", convolve)
    monkeypatch.setattr(torch, "mm", multiply)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: did.append("emptied"))
    monkeypatch.setattr(torch.cuda, "mem_get_info", read)
    device = torch.device("cpu")
    assert cli._free_memory(device) == 94_000_000_000  # pyright: ignore[reportPrivateUsage]
    assert did == [
        ("tensor", device, torch.float32),
        ("convolution", device, torch.float32),
        ("product", device, torch.float32),
        ("read", device, [True, True, True]),
    ]
    assert [ref() for ref in made] == [None, None, None]


def test_host_available(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # What the placement reads before it pins a copy (blockswap.available, which
    # tests/test_blockswap.py holds to the kernel's files): the kernel's MemAvailable, in bytes,
    # or less under a container's limit; None when nothing can be read, said as a warning, once:
    # the plan then can't check the system memory it takes, nor size the scratch's allowance
    # from the host's (planner.ALLOWANCE_UNKNOWN_HOST then bounds it).
    from seedvr2x.runtime import blockswap

    caplog.set_level(logging.DEBUG, logger="seedvr2x")
    available = cli._host_available  # pyright: ignore[reportPrivateUsage]
    proc, cgroups = tmp_path / "proc", tmp_path / "cgroup"
    (proc / "self").mkdir(parents=True)
    (cgroups / "job").mkdir(parents=True)
    meminfo = proc / "meminfo"
    meminfo.write_text("MemTotal:       65806340 kB\nMemAvailable:   41234567 kB\nCached: 1 kB\n")
    assert available(proc, cgroups) == 41_234_567 * 1024 == blockswap.available(proc, cgroups)
    # In a container held to a limit, what is left under it: 8 GiB allowed, 3 held, 1 of them
    # file cache, which the kernel gives back.
    (proc / "self" / "cgroup").write_text("0::/job\n")
    (cgroups / "job" / "memory.max").write_text(f"{8 * 1024**3}\n")
    (cgroups / "job" / "memory.current").write_text(f"{3 * 1024**3}\n")
    (cgroups / "job" / "memory.stat").write_text(f"active_file {1024**3}\ninactive_file 0\n")
    assert available(proc, cgroups) == 6 * 1024**3 == blockswap.available(proc, cgroups)
    found = available()  # this machine's own
    assert found is not None and found == pytest.approx(blockswap.available(), rel=0.2)
    assert caplog.text == ""
    # Nothing to read: no MemAvailable, and no limit.
    for text in ("MemTotal:       65806340 kB\n", "MemAvailable:   lots kB\n", "MemAvailable:\n"):
        caplog.clear()
        meminfo.write_text(text)
        (cgroups / "job" / "memory.max").write_text("max\n")
        assert available(proc, cgroups) is None
        [said] = caplog.records
        assert said.levelno == logging.WARNING
        assert said.getMessage() == (
            f"{meminfo}: no MemAvailable read there, nor a control group's limit under"
            f" {cgroups}: the system memory available isn't known, so the plan can't check what"
            " it takes of it, nor size from it what a tiled shot's scratch may take, which is"
            " then the shot's whole scratch, 8 GiB at most"
        )
    caplog.clear()
    assert available(tmp_path / "none", tmp_path / "none") is None
    [said] = caplog.records
    assert said.levelno == logging.WARNING


def test_attention_is_the_loader_s() -> None:
    # The backend a plan is made for is the one the loader sets on every attention module of
    # the DiT (model.attention_backend, model.load_models): FlashAttention 2 where it imports,
    # else SDPA, by this machine's own.
    from seedvr2x.runtime import model

    attention = cli._attention  # pyright: ignore[reportPrivateUsage]
    assert attention() == model.attention_backend()
    assert attention() in ("flash_attn_2", "sdpa")


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

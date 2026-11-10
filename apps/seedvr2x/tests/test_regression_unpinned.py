"""The helpers of tests/test_regression.py's unpinned case and its back-to-back reading, on the
CPU with synthetic inputs: a reading found in a log or a manifest, the plan the planner makes
for it rebuilt and compared with a call's lines, the run stopped once it has planned."""

import logging
import sys
import textwrap
from typing import Any

import pytest
import test_regression as reg
from levels import levels

import disk
import planner
from seedvr2x import memory, peaks
from seedvr2x.shots import Shot

GIB = memory.GIB
HD = (1080, 1920)
# Milestone 1's job: 45 frames of 960x540, one shot, at 1080.
SHOTS = (Shot(0, 45),)


def inputs() -> reg.Inputs:
    return reg.Inputs(HD, SHOTS)


def plan_log(
    free: reg.Free,
    monkeypatch: pytest.MonkeyPatch,
    host: int | None = None,
    room: str | None = None,
) -> str:
    """The lines a run logs for a reading: memory.reading's debug line and the plan's lines
    (planner.said), as the command's --verbose log has them, the plan's with a time and a
    logger in front, and `room`, if given, the run's disk line."""
    reading = reg.readings(free, monkeypatch)[0]
    made, predicted = planner.plan(reading, HD, SHOTS, planner.Pins(), host)
    said = planner.said(made, predicted, "a card", HD, SHOTS, planner.Pins(), disk=room)
    return "\n".join(f"2026-10-30 12:00:00 INFO cli.planner: {line}" for line in said)


def logged(
    free: reg.Free, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> str:
    """What memory.reading says at debug for `free`, then its plan's lines."""
    caplog.clear()
    levels(caplog, logging.DEBUG)
    reg.readings(free, monkeypatch)
    return "\n".join(caplog.messages) + "\n" + plan_log(free, monkeypatch)


# The readings a held run makes: the card smaller than the driver, the driver smaller, the card
# not read; and a driver that refuses.
READINGS = {
    "card": reg.Free(30 * GIB, round(24.8 * GIB), True),
    "driver": reg.Free(round(24.8 * GIB), 30 * GIB, True),
    "unread": reg.Free(round(24.8 * GIB), None, True),
    "refuses": reg.Free(round(24.8 * GIB), None, False),
}


@pytest.mark.parametrize("name", list(READINGS))
def test_a_reading_is_found_in_a_verbose_log(
    name: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    free = READINGS[name]
    found = reg.logged_free(logged(free, monkeypatch, caplog))
    assert found is not None
    # To the hundredth of a GiB, as the log says it.
    assert found.held is free.held and not found.exact
    assert found.reported == pytest.approx(free.reported, abs=0.005 * GIB)
    if free.card is None:
        assert found.card is None
    else:
        assert found.card == pytest.approx(free.card, abs=0.005 * GIB)


def test_a_log_without_a_reading_gives_none() -> None:
    assert reg.logged_free("12:00:00 INFO a log of no plan\n") is None


def test_a_reading_and_the_inputs_are_read_from_a_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    # The record the planner makes (Peaks.record) in the manifest's shape (cli._run).
    free = READINGS["card"]
    reading = reg.readings(free, monkeypatch)[0]
    made, predicted = planner.plan(reading, HD, SHOTS, planner.Pins(), None)
    record = made.record()
    record["memory"].update(predicted.record())
    content: dict[str, Any] = {
        "input": {"size": [960, 540], "sample_aspect": "1"},
        "settings": {"resolution": 1080},
        "shots": [{"start": 0, "end": 45, "seed": 42}],
        "plan": record,
    }
    found, read = reg.recorded_inputs(content)
    assert found == reg.Inputs(HD, SHOTS)
    # Exact: the bytes the driver and the card gave, and held, the reserve being recorded.
    assert read == free
    # A driver that refuses leaves no reserve.
    refusing = reg.readings(READINGS["refuses"], monkeypatch)[0]
    again, predicted = planner.plan(refusing, HD, SHOTS, planner.Pins(), None)
    content["plan"] = again.record() | {"memory": again.record()["memory"] | predicted.record()}
    assert reg.recorded_inputs(content)[1] == READINGS["refuses"]


def test_the_dit_s_format_is_read_from_a_manifest_and_fp16_where_none_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The DiT's format the plan was predicted for (Peaks.record's dit_format): the plan made
    # again for it, int8's by int8's figures; a manifest from before int8, which records none,
    # read as fp16, whose plans it held.
    reading = reg.readings(READINGS["card"], monkeypatch)[0]
    base: dict[str, Any] = {
        "input": {"size": [960, 540], "sample_aspect": "1"},
        "settings": {"resolution": 1080},
        "shots": [{"start": 0, "end": 45, "seed": 42}],
    }
    for dit_model, name in ((peaks.FP16, "fp16"), (peaks.INT8, "int8")):
        made, predicted = planner.plan(
            *(reading, HD, SHOTS, planner.Pins(), None),
            dit_model=dit_model,
        )
        record = made.record()
        record["memory"].update(predicted.record())
        assert record["memory"]["dit_format"] == name
        found, _ = reg.recorded_inputs(base | {"plan": record})
        assert found == reg.Inputs(HD, SHOTS, dit_model=dit_model)
        said = planner.said(made, predicted, "a card", HD, SHOTS, planner.Pins())
        assert reg.planner_says(reading, found, None) == reg.compared(
            reg.plan_lines("\n".join(said))
        )
    del record["memory"]["dit_format"]
    assert reg.recorded_inputs(base | {"plan": record})[0].dit_model == peaks.FP16


def test_the_readings_are_those_memory_makes(monkeypatch: pytest.MonkeyPatch) -> None:
    # memory.reading's own arithmetic, the tests' stand-ins for the driver and NVML restored.
    held = reg.readings(READINGS["card"], monkeypatch)
    assert held == [
        memory.Reading(
            30 * GIB,
            round(24.8 * GIB),
            round(24.8 * GIB) - memory.HELD_BACK,
            True,
            memory.HELD_BACK,
            # The platform's allocator, as readings takes it (tests/test_regression.py).
            memory.POOL if sys.platform == "win32" else memory.SEGMENTS,
        )
    ]
    unread = reg.readings(READINGS["unread"], monkeypatch)[0]
    assert unread.card is None and unread.left == memory.HELD_BACK_UNREAD
    refusing = reg.readings(READINGS["refuses"], monkeypatch)[0]
    assert refusing == memory.Reading(round(24.8 * GIB), None, round(24.8 * GIB), False)
    # A log's reading takes its hundredths' rounding both ways, each figure: nine readings.
    from_log = reg.Free(30 * GIB, round(24.8 * GIB), True, exact=False)
    options = reg.readings(from_log, monkeypatch)
    assert len(options) == 9 and options[0] == held[0]
    # The driver's figure is the larger here: only the card's decides what is counted.
    assert len({option.counted for option in options}) == 3


def test_a_call_that_planned_as_the_planner_does_agrees(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    for free in READINGS.values():
        log = logged(free, monkeypatch, caplog)
        verdict = reg.compare_with_planner("--plan", log, free, inputs(), None, monkeypatch)
        assert verdict.agrees, verdict
        reg.assert_planned_as_read([verdict])


def test_a_plan_made_in_the_desktops_other_memory_agrees_with_its_own_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The failure that made the test: two calls 0.3 GiB apart plan 5 and 4 blocks swapped. Each
    # is right for its own reading; the planner here makes both.
    blocks: dict[float, str] = {}
    for gib in (22.0, 22.3, 22.6, 22.9, 23.2, 23.5, 23.8, 24.1, 24.4, 24.7, 25.0, 25.3):
        free = reg.Free(round(gib * GIB), round(gib * GIB), True)
        lines = reg.planner_says(reg.readings(free, monkeypatch)[0], inputs(), None)
        blocks[gib] = reg.line_of(lines, "blocks swapped: ")
    assert len(set(blocks.values())) > 1, blocks
    low, high = reg.Free(round(22.0 * GIB), None, True), reg.Free(round(25.3 * GIB), None, True)
    calls = [(low, "--plan"), (high, "the run")]
    verdicts = [
        reg.compare_with_planner(
            call,
            plan_log(free, monkeypatch),
            free,
            inputs(),
            None,
            monkeypatch,
        )
        for free, call in calls
    ]
    assert all(verdict.agrees for verdict in verdicts)
    assert verdicts[0].logged[1] != verdicts[1].logged[1]


def test_a_plan_that_is_not_the_planners_for_its_reading_fails_saying_both_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A log that claims 25 GiB free and plans as for 22: the planner, for 25, plans otherwise.
    small, big = reg.Free(22 * GIB, None, True), reg.Free(30 * GIB, None, True)
    mismatched = plan_log(small, monkeypatch)
    right = plan_log(big, monkeypatch)
    verdicts = [
        reg.compare_with_planner("--plan", right, big, inputs(), None, monkeypatch),
        reg.compare_with_planner("the run", mismatched, big, inputs(), None, monkeypatch),
    ]
    assert [verdict.agrees for verdict in verdicts] == [True, False]
    with pytest.raises(pytest.fail.Exception) as failed:
        reg.assert_planned_as_read(verdicts)
    words = str(failed.value)
    assert (
        "--plan: as the planner makes it; read reported 30.00 GiB, card not read, counted" in words
    )
    assert "the run: NOT; read reported 30.00 GiB" in words
    for line in (*verdicts[1].logged, *verdicts[1].planner):
        assert line in words
    assert verdicts[1].logged != verdicts[1].planner


def test_a_reading_within_the_logs_hundredths_agrees(monkeypatch: pytest.MonkeyPatch) -> None:
    # The run's reading is known to 0.01 GiB; its plan, made in the exact bytes, may fall on the
    # other side of a step of the planner's than the figure said. Found: the smallest free
    # memory with a plan other than the one below it, to the byte.
    def blocks(driver: int) -> str:
        free = reg.Free(driver, None, False)
        return reg.line_of(
            reg.planner_says(reg.readings(free, monkeypatch)[0], inputs(), None), "blocks"
        )

    low, high = round(20 * GIB), round(60 * GIB)
    assert blocks(low) != blocks(high)
    while high - low > 1:
        middle = (low + high) // 2
        low, high = (middle, high) if blocks(middle) == blocks(low) else (low, middle)
    step = high
    assert blocks(step - 1) != blocks(step)
    said = round(step / GIB, 2)
    for exact in (step - 1, step):
        log = plan_log(reg.Free(exact, None, False), monkeypatch)
        from_log = reg.Free(round(said * GIB), None, False, exact=False)
        assert reg.compare_with_planner("run", log, from_log, inputs(), None, monkeypatch).agrees
    # Further than a hundredth from the step, it does not.
    far = reg.Free(round((said + 0.02) * GIB), None, False, exact=False)
    assert not reg.compare_with_planner(
        "run",
        plan_log(reg.Free(step - 1, None, False), monkeypatch),
        far,
        inputs(),
        None,
        monkeypatch,
    ).agrees


def test_a_run_is_stopped_once_it_has_planned() -> None:
    script = textwrap.dedent(
        """
        import sys, time
        print("12:00:00 INFO plan: card: x", file=sys.stderr, flush=True)
        print("12:00:00 INFO plan: pinned: nothing", file=sys.stderr, flush=True)
        time.sleep(300)
        """
    )
    log, status = reg.run_until_planned([sys.executable, "-c", script], {}, wait=60)
    assert status is None and log.endswith("plan: pinned: nothing\n")
    assert "plan: card: x" in log


def test_a_run_that_ends_before_its_plan_says_its_status() -> None:
    script = "import sys; print('12:00:00 ERROR refused', file=sys.stderr); sys.exit(1)"
    log, status = reg.run_until_planned([sys.executable, "-c", script], {}, wait=60)
    assert status == 1 and "ERROR refused" in log


def test_the_other_programs_memory_is_read_once_it_settles(monkeypatch: pytest.MonkeyPatch) -> None:
    states = iter([(0, 5 * GIB), (0, 7 * GIB), (0, 6 * GIB), (0, 6 * GIB + 1)])
    monkeypatch.setattr(memory, "card_state", lambda index=0: next(states))
    monkeypatch.setattr(reg.time, "sleep", lambda seconds: None)
    assert reg.other_programs() == 6 * GIB + 1
    # NVML silent: the test skips, a steady card can't be told.
    monkeypatch.setattr(memory, "card_state", lambda index=0: None)
    with pytest.raises(pytest.skip.Exception):
        reg.other_programs()


def test_unpinned_takes_the_figure_it_is_given() -> None:
    # From the call's own record, not from the log's words: under 40 GiB skips, over fails.
    with pytest.raises(pytest.skip.Exception):
        reg.unpinned("a log that says nothing", "why", 24.8)
    with pytest.raises(pytest.fail.Exception, match=r"reports 45.00 GiB free"):
        reg.unpinned("a log that says nothing", "why", 45.0)


# ------------------------------------------------------------------- the disk line's masters

OUT = (2160, 3840)


def masters_line(
    free: int,
    *,
    sizes: tuple[int | None, ...] = (None, None),
    assembled: bool = True,
    commanded: bool = False,
    format: str = "gbrp16le",
) -> str:
    """The disk line a run logs (disk.Disk.line) for a job of two segments of 20 and 25 frames,
    `sizes` the bytes of those finished, with `free` bytes free on its disk."""
    found = disk.masters([20, 25], list(sizes), format, OUT, commanded)
    return disk.Disk(found, free, assembled).line()


def manifest(**more: Any) -> dict[str, Any]:
    """The part of a manifest the disk line is made from (cli._disk), with the job's other
    records of test_a_reading_and_the_inputs_are_read_from_a_manifest's shape."""
    return {
        "output": {"format": "gbrp16le", "size": [OUT[1], OUT[0]]},
        "settings": {"resolution": 1080},
        "segments": [
            {"start": 0, "end": 20, "finished": False, "bytes": None},
            {"start": 20, "end": 45, "finished": False, "bytes": None},
        ],
    } | more


@pytest.mark.parametrize(
    "line",
    [
        masters_line(500 * GIB),
        masters_line(1 << 20),  # short: the verdict follows the free space
        masters_line(500 * GIB, assembled=False),
        masters_line(500 * GIB, commanded=True),
        masters_line(500 * GIB, sizes=(3 * GIB, None)),
        masters_line(500 * GIB, sizes=(3 * GIB, 4 * GIB), assembled=False),
    ],
)
def test_a_disk_line_is_compared_by_its_masters_alone(line: str) -> None:
    kept = reg.compared([line])
    [masters] = kept
    assert masters.startswith("disk: the masters")
    assert "free on its disk" not in masters and "short" not in masters and "enough" not in masters
    # Nothing but the free space and the verdict is gone: the same job on another disk compares
    # equal, whatever is free there.
    assert line.startswith(masters)
    other = line.replace("500 GiB free", "9.0 GiB free").replace("1 MiB free", "700 GiB free")
    assert reg.compared([other]) == kept


def test_a_disk_line_with_other_masters_compares_otherwise() -> None:
    # The masters' size, whether the file is assembled and a command's files are the job's.
    lines = [
        masters_line(500 * GIB),
        masters_line(500 * GIB, assembled=False),
        masters_line(500 * GIB, commanded=True),
        masters_line(500 * GIB, format="yuv420p10le"),
        masters_line(500 * GIB, sizes=(3 * GIB, None)),
    ]
    compared = [reg.compared([line])[0] for line in lines]
    assert len(set(compared)) == len(lines), compared
    # Other lines than the disk's are as choices leaves them.
    others = ["windows: 1 window", "card: 24 GiB free", "pinned: nothing"]
    assert reg.compared(others) == ["windows: 1 window", "pinned: nothing"]


def test_the_disk_line_a_manifest_records_is_the_runs_without_its_free_space() -> None:
    recorded = reg.recorded_disk(manifest())
    assert recorded is not None
    assert reg.compared([recorded]) == reg.compared([masters_line(500 * GIB)])
    # A segment command's files, and the bytes of the segments a resume finished.
    commanded = reg.recorded_disk(manifest(settings={"segment_cmd": "x"}))
    assert commanded is not None and "not known before one is written" in commanded
    done = manifest()
    done["segments"][0] |= {"finished": True, "bytes": 3 * GIB}
    resumed = reg.recorded_disk(done)
    assert resumed is not None
    assert reg.compared([resumed]) == reg.compared([masters_line(1 << 30, sizes=(3 * GIB, None))])
    # No record of an output: no line (the other tests' synthetic content).
    assert reg.recorded_disk({"settings": {}}) is None


def test_the_inputs_carry_the_disk_line_of_the_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    free = READINGS["card"]
    reading = reg.readings(free, monkeypatch)[0]
    made, predicted = planner.plan(reading, HD, SHOTS, planner.Pins(), None)
    content: dict[str, Any] = {
        "input": {"size": [960, 540], "sample_aspect": "1"},
        "shots": [{"start": 0, "end": 45, "seed": 42}],
        "plan": made.record() | {"memory": made.record()["memory"] | predicted.record()},
    } | manifest()
    found, _ = reg.recorded_inputs(content)
    assert found.disk == reg.recorded_disk(content) and found.disk is not None
    assert found.disk.startswith("disk: the masters")


def test_a_plan_with_the_jobs_masters_agrees_on_whatever_disk_it_ran(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    free = READINGS["card"]
    wanted = disk.Disk(disk.masters([45], [None], "gbrp16le", OUT, False), 0, True).line()
    for room in (500 << 30, 2 << 20):  # the disk has room, and is short
        log = plan_log(free, monkeypatch, room=masters_line(room, sizes=(None, None)))
        # The log's job is two segments of 20 and 25 frames: 45 frames in all, as `wanted`'s.
        with_disk = reg.Inputs(HD, SHOTS, wanted)
        verdict = reg.compare_with_planner("run", log, free, with_disk, None, monkeypatch)
        assert verdict.agrees, verdict


def test_a_plan_with_other_masters_than_the_jobs_fails_saying_the_disk_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    free = READINGS["card"]
    wanted = disk.Disk(disk.masters([45], [None], "gbrp16le", OUT, False), 0, True).line()
    for log_room in (
        masters_line(500 * GIB, format="yuv420p10le"),  # another format
        masters_line(500 * GIB, assembled=False),  # another assembly
        None,  # no disk line at all
    ):
        log = plan_log(free, monkeypatch, room=log_room)
        with_disk = reg.Inputs(HD, SHOTS, wanted)
        verdict = reg.compare_with_planner("the run", log, free, with_disk, None, monkeypatch)
        assert not verdict.agrees
        with pytest.raises(pytest.fail.Exception) as failed:
            reg.assert_planned_as_read([verdict])
        assert "disk: the masters" in str(failed.value) or "(no line 'disk: ')" in str(failed.value)


# ------------------------------------------------------------ the host's reading, from the log

DEBUG_PREFIX = "2026-10-31 12:00:00,123 DEBUG cli.main: "


def test_the_host_reading_is_found_in_a_verbose_log() -> None:
    # The line cli._run logs (tests/test_cli_run.py holds its words), in the logger's format.
    log = host_line(20164055531) + "\n"
    assert "18.78 GiB (20164055531 bytes) of system memory available" in log
    assert reg.logged_host(log) == 20164055531
    unread = f"{DEBUG_PREFIX}{reg.HOST_UNREAD}\n"
    assert reg.logged_host(unread) is None
    # A log with neither can't have its plan rebuilt: the call fails, saying so.
    with pytest.raises(pytest.fail.Exception, match="no system memory reading"):
        reg.logged_host("12:00:00 INFO a log of no plan\n")


def host_line(host: int) -> str:
    return (
        f"{DEBUG_PREFIX}the plan counts on {host / GIB:.2f} GiB ({host} bytes) of system memory"
        " available"
    )


def test_a_plan_bound_by_the_system_memory_is_rebuilt_from_the_logged_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A plan bound by the system memory moves at steps of a few MiB, and its lines say the host's
    # figure to 0.01 GiB: two decimals of GiB rebuilt it wrongly near a step (found by a sweep of
    # +-0.005 GiB that missed one). Found: the smallest host with a plan other than the one below
    # it, to the byte; the bytes of the log rebuild each side of it. The card's figure picked
    # where the plan swaps blocks (20 GiB: 11 to 12 of them by the host; the segments' reserve by
    # step, which takes no flat 4.5 off the figure, holds the DiT whole at 24.8 GiB, where the
    # host could bind nothing).
    free = reg.Free(round(20 * GIB), 30 * GIB, True)

    def blocks(host: int) -> str:
        lines = reg.planner_says(reg.readings(free, monkeypatch)[0], inputs(), host)
        return reg.line_of(lines, "blocks")

    low, high = 10 * GIB, 60 * GIB
    assert blocks(low) != blocks(high)
    while high - low > 1:
        middle = (low + high) // 2
        low, high = (middle, high) if blocks(middle) == blocks(low) else (low, middle)
    step = high
    assert blocks(step - 1) != blocks(step)
    for exact in (step - 1, step):
        log = host_line(exact) + "\n" + plan_log(free, monkeypatch, exact)
        host = reg.logged_host(log)
        assert host == exact
        assert reg.compare_with_planner("run", log, free, inputs(), host, monkeypatch).agrees
    # A host the other side of the step than the one the plan was made with does not agree.
    log = plan_log(free, monkeypatch, step - 1)
    assert not reg.compare_with_planner("run", log, free, inputs(), step, monkeypatch).agrees
    # A host not read is rebuilt as the planner makes it without one.
    assert reg.compare_with_planner(
        "run", plan_log(free, monkeypatch, None), free, inputs(), None, monkeypatch
    ).agrees


def test_errors_are_found_with_and_without_verbose() -> None:
    plain = "12:00:00 INFO a line\n12:00:01 ERROR the plan takes 40 GiB of free VRAM\n"
    verbose = (
        "2026-10-31 12:00:00,001 INFO cli.main: a line\n"
        "2026-10-31 12:00:01,002 ERROR cli.main: the plan takes 40 GiB of free VRAM\n"
        "2026-10-31 12:00:01,003 ERROR cli.media.assembly: and: a second\n"
    )
    assert reg.errors_of(plain) == ["the plan takes 40 GiB of free VRAM"]
    assert reg.errors_of(verbose) == ["the plan takes 40 GiB of free VRAM", "and: a second"]
    assert reg.errors_of("12:00:00 INFO a line\n") == []


def test_a_verbose_plan_log_still_parses(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The --plan call runs with -v now: its log carries the debug lines and the logger's format,
    # as the run's has. Its first pass, plan lines, free memory and host are all read.
    free = READINGS["card"]
    log = "\n".join(
        [
            "2026-10-31 12:00:00,001 INFO cli.main: the shot detector scored 45 frames",
            "2026-10-31 12:00:00,002 INFO cli.main: shot detector: VRAM peak 0.30 GiB",
            "2026-10-31 12:00:00,003 INFO cli.main: 1 shot: 0 cuts detected at --cut-threshold 0.5",
            host_line(round(31.5 * GIB)),
            logged(free, monkeypatch, caplog),
        ]
    )
    reg.detected(log)
    assert reg.logged_host(log) == round(31.5 * GIB)
    assert reg.logged_free(log) is not None
    assert reg.reported(log) == pytest.approx(30.0, abs=0.01)
    lines = reg.plan_lines(log)
    assert any(line.startswith("encode: ") for line in lines)
    assert any(line.startswith("pinned: ") for line in lines)
    # The rebuilt plan agrees with it, host and all.
    verdict = reg.compare_with_planner(
        "--plan",
        log,
        free,
        inputs(),
        reg.logged_host(log),
        monkeypatch,
    )
    assert verdict.agrees, verdict

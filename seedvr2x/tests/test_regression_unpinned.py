"""The helpers of tests/test_regression.py's unpinned case and its back-to-back reading, on the
CPU with synthetic inputs: a reading found in a log or a manifest, the plan the planner makes
for it rebuilt and compared with a call's lines, the run stopped once it has planned."""

import logging
import sys
import textwrap
from typing import Any

import pytest
import test_regression as reg

from seedvr2x.runtime import memory, planner
from seedvr2x.runtime.job import Shot

GIB = memory.GIB
HD = (1080, 1920)
# Milestone 1's job: 45 frames of 960x540, one shot, at 1080.
SHOTS = (Shot(0, 45),)


def inputs() -> reg.Inputs:
    return reg.Inputs(HD, SHOTS, planner.FLASH_ATTN_2)


def plan_log(free: reg.Free, monkeypatch: pytest.MonkeyPatch, host: int | None = None) -> str:
    """The lines a run logs for a reading: memory.reading's debug line and the plan's lines
    (planner.said), as the command's --verbose log has them, the plan's with a time and a
    logger in front."""
    reading = reg.readings(free, monkeypatch)[0]
    made, peaks = planner.plan(
        reading, HD, SHOTS, planner.Pins(), host, numz_padding=True, attention=planner.FLASH_ATTN_2
    )
    said = planner.said(made, peaks, "a card", HD, SHOTS, planner.Pins(), numz_padding=True)
    return "\n".join(f"2026-10-30 12:00:00 INFO seedvr2x: {line}" for line in said)


def logged(
    free: reg.Free, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> str:
    """What memory.reading says at debug for `free`, then its plan's lines."""
    caplog.clear()
    caplog.set_level(logging.DEBUG, logger="seedvr2x")
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
    made, peaks = planner.plan(
        reading, HD, SHOTS, planner.Pins(), None, numz_padding=True, attention=planner.SDPA
    )
    record = made.record()
    record["memory"].update(peaks.record())
    content: dict[str, Any] = {
        "input": {"size": [960, 540], "sample_aspect": "1"},
        "settings": {"resolution": 1080},
        "shots": [{"start": 0, "end": 45, "seed": 42}],
        "plan": record,
    }
    found, read = reg.recorded_inputs(content)
    assert found == reg.Inputs(HD, SHOTS, planner.SDPA)
    # Exact: the bytes the driver and the card gave, and held, the reserve being recorded.
    assert read == free
    # A driver that refuses leaves no reserve.
    refusing = reg.readings(READINGS["refuses"], monkeypatch)[0]
    again, peaks = planner.plan(
        refusing, HD, SHOTS, planner.Pins(), None, numz_padding=True, attention=planner.SDPA
    )
    content["plan"] = again.record() | {"memory": again.record()["memory"] | peaks.record()}
    assert reg.recorded_inputs(content)[1] == READINGS["refuses"]


def test_the_readings_are_those_memory_makes(monkeypatch: pytest.MonkeyPatch) -> None:
    # memory.reading's own arithmetic, the tests' stand-ins for the driver and NVML restored.
    held = reg.readings(READINGS["card"], monkeypatch)
    assert held == [
        memory.Reading(
            30 * GIB,
            round(24.8 * GIB),
            round(24.8 * GIB) - memory.HELD_BACK - memory.RESERVE,
            True,
            memory.HELD_BACK,
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

"""One clock (clock.py; DESIGN.md, Time estimates, "One clock", decided on 2026-10-28): the
run's own clock against the wall's, on fake readers, nothing sleeping. A tick of the test moves
both: the run's clock by `mono` seconds, the wall by `mono` times the ratio the test sets, plus
whatever step of the wall it adds."""

import logging
import os
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from seedvr2x import clock
from seedvr2x.clock import MIN_SPAN, Pace
from seedvr2x.media import assembly
from seedvr2x.runtime import run, timing
from seedvr2x.runtime import stop as stop_module
from seedvr2x.runtime.stop import Deadline, Stop, late
from seedvr2x.runtime.timing import Clock, Kind, Model, Progress, Unit, would_end

START = datetime(2026, 10, 29, 22, 0, tzinfo=UTC)
# A day's epoch seconds for the wall's reader: any, only its differences count.
EPOCH = 1_793_000_000.0


class Readers:
    """The two clocks of a test: `tick` moves the run's by `mono` seconds and the wall's by
    `mono * drift`, `step` moves the wall's alone."""

    def __init__(self, drift: float) -> None:
        self.mono, self.wall, self.drift = 5_000.0, EPOCH, drift

    def tick(self, mono: float) -> None:
        self.mono += mono
        self.wall += mono * self.drift

    def step(self, seconds: float) -> None:
        self.wall += seconds

    def pace(self) -> Pace:
        return Pace(lambda: self.mono, lambda: self.wall)


def measured(drift: float, over: float = 1_200.0, every: float = 100.0) -> tuple[Readers, Pace]:
    """A Pace that has watched `over` seconds of the run's clock, a reading every `every`, the
    wall's running at `drift` of it."""
    readers = Readers(drift)
    pace = readers.pace()
    for _ in range(round(over / every)):
        readers.tick(every)
        pace.ratio()
    return readers, pace


def test_the_ratio_is_one_before_the_minimum_span() -> None:
    readers = Readers(0.9)
    pace = readers.pace()
    assert pace.ratio() == 1.0
    readers.tick(MIN_SPAN - 1)
    assert pace.ratio() == 1.0
    assert pace.wall(42.0) == 42.0
    readers.tick(1)
    assert pace.ratio() == pytest.approx(0.9)


@pytest.mark.parametrize("drift", [0.9, 0.91, 1.0, 1 / 1.1, 1.05])
def test_the_ratio_is_the_walls_over_the_runs(drift: float) -> None:
    # 0.90 to 0.91: milestone 7's nights (a 300 s sleep took 333 s); 1 / 1.1: 2026-10-29's, 1,692
    # s of the run's clock to 1,539 s of the wall's.
    _, pace = measured(drift)
    assert pace.ratio() == pytest.approx(drift)
    assert pace.wall(100.0) == pytest.approx(100.0 * drift)


def test_a_duration_said_at_150_s_after_the_import_is_converted() -> None:
    # The RTX 5090 run of 2026-10-29: "SHA-256 ... in 99.8 s" in the run's clock where the log's
    # times gave 91 s, the ratio held at 1.0 for the first 600 s. At MIN_SPAN (120 s) it is
    # measured: 150 s in, a duration is in the wall's seconds already.
    readers = Readers(0.91)
    pace = readers.pace()
    readers.tick(75.0)
    assert pace.wall(99.8) == 99.8  # 75 s in: too early
    readers.tick(75.0)
    assert pace.wall(99.8) == pytest.approx(99.8 * 0.91)
    assert pace.wall(99.8) == pytest.approx(90.8, abs=0.1)


@pytest.mark.parametrize("drift", [0.9, 1.1])
def test_a_first_reading_after_a_long_first_pass_is_no_step(
    drift: float, caplog: pytest.LogCaptureFixture
) -> None:
    # The review of 2026-10-29: a first reading 1,500 s after the import (a 4K first pass), the
    # run's clock at `drift` of the wall's, was 167 s off 1.0 and taken for a step, the anchor
    # moved and 1.0 kept. No ratio measured yet, the gap's own ratio (1.11 or 0.91) is inside
    # BAND: no step, the ratio measured.
    caplog.set_level(logging.DEBUG, logger=clock.__name__)
    readers = Readers(1 / drift)
    pace = readers.pace()
    readers.tick(1_500.0)
    assert pace.ratio() == pytest.approx(1 / drift)
    assert "stepped" not in caplog.text


@pytest.mark.parametrize("hour", [3600.0, -3600.0])
def test_an_hour_is_a_step_before_any_ratio_is_measured(
    hour: float, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=clock.__name__)
    readers = Readers(1.0)
    pace = readers.pace()
    readers.tick(100.0)
    readers.step(hour)
    assert pace.ratio() == 1.0
    assert f"the wall clock stepped by {hour:+.0f} s" in caplog.text
    # Re-anchored there: the hour is in no ratio, and the measure that follows is the clocks'.
    readers.drift = 0.9
    readers.tick(150.0)
    assert pace.ratio() == pytest.approx(0.9)


def test_the_measure_of_2026_10_29() -> None:
    readers = Readers(1_539 / 1_692)
    pace = readers.pace()
    for _ in range(18):
        readers.tick(94.0)
        pace.ratio()
    assert pace.ratio() == pytest.approx(1_539 / 1_692)
    assert 1 / pace.ratio() == pytest.approx(1.0994, abs=1e-4)


def test_the_ratio_is_said_at_debug_when_it_moves(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger=clock.__name__)
    readers = Readers(1 / 1.1)
    pace = readers.pace()
    for _ in range(14):
        readers.tick(100.0)
        pace.ratio()
    said = [r.getMessage() for r in caplog.records if r.getMessage().startswith("the clocks:")]
    assert said == ["the clocks: the run's own clock at 1.100 of the wall's, over 200 s"]
    # Every later reading is within 1% of it: said no more.
    for _ in range(5):
        readers.tick(100.0)
        pace.ratio()
    assert len([r for r in caplog.records if r.getMessage().startswith("the clocks:")]) == 1
    # The drift changes by 5%: said again, with the span since the anchor.
    readers.drift = 1 / 1.1 * 0.95
    for _ in range(80):
        readers.tick(100.0)
        pace.ratio()
    assert any(r.getMessage().startswith("the clocks:") for r in caplog.records[1:])


def test_a_ratio_out_of_the_band_is_not_used(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger=clock.__name__)
    # Read every 20 s, so that no reading is a step (60 s over what the ratio held predicts).
    readers = Readers(2.5)
    pace = readers.pace()
    for _ in range(40):
        readers.tick(20.0)
        pace.ratio()
    assert pace.ratio() == 1.0
    assert "outside 0.5 to 2.0: not used" in caplog.text


def test_the_wall_stepped_by_two_seconds_either_way_barely_moves_the_ratio() -> None:
    readers, pace = measured(0.91)
    before = pace.ratio()
    for seconds in (2.0, -2.0, -1.5, 2.0, -2.0, 1.0):
        readers.tick(100.0)
        readers.step(seconds)
        # The duration the run says is positive and the run's own, whatever the wall did.
        assert pace.wall(100.0) > 0
        assert pace.ratio() == pytest.approx(before, abs=0.004)
    # Stepped back at the very reading, the wall behind the last reading's: still no negative.
    readers.tick(5.0)
    readers.step(-2.0)
    assert pace.wall(5.0) > 0 and pace.wall(0.5) > 0


@pytest.mark.parametrize("hour", [3600.0, -3600.0])
def test_the_wall_set_by_an_hour_measures_again_and_keeps_the_ratio(
    hour: float, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger=clock.__name__)
    readers, pace = measured(0.91)
    before = pace.ratio()
    readers.tick(100.0)
    readers.step(hour)
    assert pace.ratio() == pytest.approx(before)
    assert f"the wall clock stepped by {hour:+.0f} s" in caplog.text
    # Re-anchored at that reading: the hour is in no ratio, now or when the span is long again.
    for _ in range(3):
        readers.tick(100.0)
        assert pace.ratio() == pytest.approx(before)
    for _ in range(6):
        readers.tick(100.0)
        pace.ratio()
    assert pace.ratio() == pytest.approx(0.91)


def test_a_step_of_the_wall_is_said_at_info(caplog: pytest.LogCaptureFixture) -> None:
    # DESIGN.md, Architecture, What it prints (decided on 2026-10-28, built on 2026-10-29): the
    # clock's own re-anchor is a line of the default output, not a debug line: a time of day
    # below it follows the stepped clock.
    caplog.set_level(logging.INFO, logger=clock.__name__)
    readers, pace = measured(1.0)
    readers.tick(100.0)
    readers.step(3600.0)
    pace.ratio()
    [record] = [r for r in caplog.records if "stepped" in r.getMessage()]
    assert record.levelname == "INFO"
    assert record.getMessage().startswith("the wall clock stepped by +3600 s (over 100 s")
    # A step of two seconds either way is no line: it is the wall clock's own small steps.
    caplog.clear()
    readers.tick(100.0)
    readers.step(2.0)
    pace.ratio()
    assert not caplog.records


def test_a_vm_asleep_is_a_step_of_the_wall(caplog: pytest.LogCaptureFixture) -> None:
    # The VM resumed from sleep: the run's clock stood still, the wall went on for 8 hours.
    caplog.set_level(logging.DEBUG, logger=clock.__name__)
    readers, pace = measured(1.1)
    readers.step(8 * 3600.0)
    readers.tick(10.0)
    assert pace.ratio() == pytest.approx(1.1)
    assert "the wall clock stepped by +28800 s" in caplog.text


def test_a_drift_over_a_long_unit_is_no_step(caplog: pytest.LogCaptureFixture) -> None:
    # Held against the ratio measured, not against 1.0: a unit of an hour at 0.9 is 6 minutes
    # off a ratio of 1.0, and is no step once the ratio is known.
    caplog.set_level(logging.DEBUG, logger=clock.__name__)
    readers, pace = measured(0.9)
    readers.tick(3600.0)
    assert pace.ratio() == pytest.approx(0.9)
    assert "stepped" not in caplog.text


def at(ratio: float) -> Pace:
    """A Pace that measured `ratio`."""
    return measured(ratio)[1]


@pytest.fixture(autouse=True)
def zone_utc() -> Iterator[None]:
    """The local zone, which a job's end is said in (timing.ending), set to UTC here, the
    process's given back."""
    before = os.environ.get("TZ")
    os.environ["TZ"] = "UTC"
    time.tzset()
    try:
        yield
    finally:
        if before is None:
            del os.environ["TZ"]
        else:
            os.environ["TZ"] = before
        time.tzset()


@pytest.fixture
def paced(monkeypatch: pytest.MonkeyPatch) -> Callable[[float], Pace]:
    """A way to set the ratio the code says its time by (the conftest's is 1.0)."""

    def setting(ratio: float) -> Pace:
        pace = at(ratio)
        monkeypatch.setattr(clock, "pace", pace)
        return pace

    return setting


# A deadline 100 s after the unit begins, a unit estimated at 90 s of the run's clock: 1.15 times
# it is 103.5 s of the run's clock, and 93.2 s of the wall's at 0.9, 113.9 s at 1.1.
DEADLINE = START + timedelta(seconds=100)


@pytest.mark.parametrize(
    ("ratio", "stops"),
    [(1.0, True), (0.9, False), (1 / 0.9, True), (1.1, True), (0.95, False)],
)
def test_until_is_held_by_the_wall_clock(
    ratio: float, stops: bool, paced, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The unit fits by the wall when the run's clock is slow against it (ratio under 1: the wall
    # advances less per second of the run's), the reverse when it is fast.
    paced(ratio)
    monkeypatch.setattr(stop_module, "now", lambda: START)
    with Stop(DEADLINE) as stop:
        if stops:
            with pytest.raises(Deadline) as raised:
                stop.check(90.0, "the unit")
            # The estimate it carries is said in the wall's seconds.
            assert raised.value.cost == pytest.approx(90.0 * ratio)
        else:
            stop.check(90.0, "the unit")


def test_until_flips_with_the_ratio_where_the_wall_says(
    paced, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The same unit and the same time: begun at 1.0 as the run's clock says (103.5 s of 100 s
    # left: late), and not late once the wall's seconds are 0.9 of them (93.15 s).
    monkeypatch.setattr(stop_module, "now", lambda: START)
    assert late(START, 90.0, DEADLINE)
    paced(0.9)
    assert not late(START, clock.wall(90.0), DEADLINE)
    paced(1 / 0.9)
    assert late(START, clock.wall(90.0), DEADLINE)


def test_a_unit_nothing_estimates_is_held_to_the_time_alone(
    paced, monkeypatch: pytest.MonkeyPatch
) -> None:
    paced(0.9)
    monkeypatch.setattr(stop_module, "now", lambda: START)
    with Stop(DEADLINE) as stop:
        stop.check(None, "the unit")
    monkeypatch.setattr(stop_module, "now", lambda: DEADLINE + timedelta(seconds=1))
    with Stop(DEADLINE) as stop, pytest.raises(Deadline) as raised:
        stop.check(None, "the unit")
    assert raised.value.cost is None


MODELS: dict[Kind, Model] = {
    "encode": Model(600.0, 0.0),
    "window": Model(60.0, 0.0),
    "decode": Model(60.0, 0.0),
}
UNITS = [Unit(("encode", 0), "the encode", (("encode", 1),))]  # 600 s of the run's clock


@pytest.mark.parametrize(
    ("ratio", "left", "end"),
    [(1.0, "10 min 0 s", "22:10"), (0.9, "9 min 0 s", "22:09"), (1.1, "11 min 0 s", "22:11")],
)
def test_the_time_left_and_the_end_are_the_walls(ratio: float, left: str, end: str, paced) -> None:
    paced(ratio)
    progress = Progress(Clock(MODELS), UNITS, lambda: START)
    assert progress.said() == f"about {left} left, the job ending about {end}"
    # The estimates themselves stay in the run's clock: the models learn in it.
    assert progress.left() == pytest.approx(600.0)
    assert progress.cost(("encode", 0)) == pytest.approx(600.0)


def test_the_assembly_left_is_said_in_the_walls_seconds(paced) -> None:
    paced(0.9)
    progress = Progress(Clock(MODELS), [], lambda: START, lambda: 600.0)
    assert progress.said() == "about 9 min 0 s left, the job ending about 22:09"


def test_would_end_by_the_walls_seconds(paced) -> None:
    # Three units of 60 s of the run's clock, --until 200 s from START: by the run's clock the
    # third begins at 120 s and ends at 189 s, so fits; at 1.2 it ends at 120 * 1.2 + 69 * 1.2.
    units = [Unit(("window", i), f"encode {i}", (("window", 1),)) for i in range(3)]
    clock_ = Clock(MODELS)
    deadline = START + timedelta(seconds=200)
    found = would_end(clock_, units, START, deadline)
    assert found.before is None and (found.done, found.total) == (180.0, 180.0)
    paced(0.9)
    found = would_end(clock_, units, START, deadline)
    assert found.before is None
    assert (found.done, found.total) == pytest.approx((162.0, 162.0))
    assert found.at == START + timedelta(seconds=162)
    paced(1 / 0.9)
    found = would_end(clock_, units, START, deadline)
    assert found.before == "encode 2"
    assert found.done == pytest.approx(120 / 0.9)
    assert found.total == pytest.approx(180 / 0.9)
    # The assembly after the units, its estimate in the run's clock too.
    paced(0.9)
    found = would_end(clock_, units, START, START + timedelta(seconds=250), ("the assembly", 100.0))
    assert found.before == "the assembly" and found.done == pytest.approx(162.0)


def test_nothing_measures_a_unit_by_the_wall(paced, monkeypatch: pytest.MonkeyPatch) -> None:
    # A unit's time given to the models is the run's clock's, whatever the ratio.
    paced(0.5)
    clock_ = Clock(MODELS)
    progress = Progress(clock_, UNITS, lambda: START)
    seen: list[float] = []
    monkeypatch.setattr(clock_, "observe", lambda kind, size, seconds: seen.append(seconds))
    progress.made(("encode", 0), "encode", 1, 123.0)
    assert seen == [123.0]


def test_the_spent_time_is_said_in_the_walls_seconds(
    paced, monkeypatch: pytest.MonkeyPatch
) -> None:
    paced(0.9)
    monkeypatch.setattr(run, "time", SimpleNamespace(monotonic=lambda: 110.0))
    monkeypatch.setattr(run.torch.cuda, "max_memory_allocated", lambda device: 2 * 1024**3)
    assert run._spent(SimpleNamespace(device="cpu"), 0.0) == (  # pyright: ignore[reportPrivateUsage,reportArgumentType]
        "99.0 s, VRAM peak 2.00 GiB"
    )


def test_the_time_of_day_is_the_walls_across_a_step() -> None:
    # Progress says the end from the wall's `now` at the moment it is asked: a step of the wall
    # clock moves the end it says with it, as the clock on the wall does.
    stepped = START + timedelta(hours=1)
    progress = Progress(Clock(MODELS), UNITS, lambda: stepped)
    assert progress.said() == "about 10 min 0 s left, the job ending about 23:10"
    assert timing.ending(stepped, 600) == "23:10"


def test_the_assemblys_reading_time_is_said_in_the_walls_seconds(
    paced, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # media/ says through the same clock: a read of 60 s of the run's clock is 54 s of the wall's
    # at 0.9, said to 5 s.
    source = tmp_path / "in.mkv"
    source.write_bytes(b"")
    monkeypatch.setattr(assembly, "_read_seconds", lambda *given: 60.0)
    head = (
        f"{source}: the source's other streams (audio, subtitles, a cover) copied once ahead, so"
        f" that one the finished file can't take is found now, not after the night's work;"
        f" reading"
    )
    assert assembly.reading_said(source) == f"{head} its 0 KiB, at least about 60 s"
    paced(0.9)
    assert assembly.reading_said(source) == f"{head} its 0 KiB, at least about 55 s"


def sampled(readers: Readers, pace: Pace, *spans: float) -> None:
    """The run's clock moving through `spans`, a reading of both at the end of each, as a line
    logged makes one (clock.sample)."""
    for span in spans:
        readers.tick(span)
        pace.sample()


def test_a_span_said_before_the_ratio_is_measured_is_the_walls_own() -> None:
    # "models loaded in 88.2 s" was said for 79 s by the lines' times (2026-10-29): the run's
    # clock 10% fast against the wall's, 90 s in, no ratio yet (MIN_SPAN is 120 s).
    readers = Readers(0.9)
    pace = readers.pace()
    sampled(readers, pace, 10.0)  # a line logged where the span begins
    sampled(readers, pace, 80.0)
    assert pace.wall(80.0) == 80.0  # the ratio is held at 1.0
    assert pace.spent(80.0) == pytest.approx(72.0)  # the wall's seconds over the span
    # The start between two readings: the clocks run evenly between them.
    assert pace.spent(85.0) == pytest.approx(85.0 * 0.9)


def test_a_span_said_after_the_ratio_is_measured_is_converted_by_it() -> None:
    readers, pace = measured(0.9)
    assert pace.spent(80.0) == pytest.approx(72.0) == pace.wall(80.0)
    # The ratio, not the span's own seconds: a step inside the span doesn't count in it.
    readers.tick(10.0)
    readers.step(2.0)
    assert pace.spent(10.0) == pytest.approx(9.0, abs=0.05)  # not 11


def test_a_span_that_began_before_every_reading_is_said_in_the_runs_own() -> None:
    readers = Readers(0.9)
    pace = readers.pace()
    sampled(readers, pace, 50.0)
    assert pace.spent(60.0) == 60.0  # began 10 s before the anchor


@pytest.mark.parametrize(("step", "said"), [(2.0, 74.0), (-3.0, 69.0), (-45.0, 80.0)])
def test_a_step_of_the_wall_inside_a_span_never_makes_it_negative_or_absurd(
    step: float, said: float, caplog: pytest.LogCaptureFixture
) -> None:
    # A step of 2 s goes in the span's seconds whole (within SPAN_BAND); one that leaves the wall's
    # seconds outside it (-45 s of 80) is no measure of the span: its own seconds, said at debug.
    caplog.set_level(logging.DEBUG, logger=clock.__name__)
    readers = Readers(0.9)
    pace = readers.pace()
    sampled(readers, pace, 10.0, 40.0)
    readers.step(step)
    sampled(readers, pace, 40.0)
    assert pace.spent(80.0) == pytest.approx(said)
    assert (said == 80.0) == ("outside 0.8 to 1.25 of it" in caplog.text)


@pytest.mark.parametrize("hour", [3600.0, -3600.0])
def test_an_hour_set_inside_a_span_leaves_the_runs_own_seconds(
    hour: float, caplog: pytest.LogCaptureFixture
) -> None:
    # Seen as a step (ratio): the readings from before it are dropped, so the span is said in the
    # run's own seconds, not in an hour more or less.
    caplog.set_level(logging.INFO, logger=clock.__name__)
    readers = Readers(1.0)
    pace = readers.pace()
    sampled(readers, pace, 10.0, 30.0)
    readers.step(hour)
    readers.tick(40.0)
    assert pace.spent(70.0) == 70.0
    assert "the wall clock stepped" in caplog.text
    assert 0 < pace.spent(30.0) <= 60.0


def test_a_duration_is_never_negative_whatever_the_wall_does() -> None:
    readers = Readers(1.0)
    pace = readers.pace()
    sampled(readers, pace, 10.0)
    readers.step(-5.0)
    sampled(readers, pace, 3.0)
    assert pace.spent(3.0) == 3.0
    assert pace.spent(0.0) == 0.0


def test_every_line_logged_is_a_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    # The logging handler's filter (cli._logging) lets the line through and reads both clocks,
    # so that the start of a span is among the readings.
    from seedvr2x import cli

    readers = Readers(0.9)
    pace = readers.pace()
    monkeypatch.setattr(clock, "pace", pace)
    record = logging.LogRecord("seedvr2x", logging.INFO, __file__, 1, "a line", (), None)
    readers.tick(30.0)
    assert cli._sampled(record)  # pyright: ignore[reportPrivateUsage]
    readers.tick(60.0)
    assert clock.spent(60.0) == pytest.approx(54.0)
    handler_filters = logging.StreamHandler()
    handler_filters.addFilter(cli._sampled)  # pyright: ignore[reportPrivateUsage]
    assert handler_filters.filter(record)


@pytest.mark.parametrize("verbose", [True, False])
def test_the_handler_the_cli_installs_samples_the_clocks(
    verbose: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The real cli._logging, as the CLI calls it with and without --verbose: the handler it puts
    # on the root logger carries the sampling filter, and a line through it is a reading.
    from seedvr2x import cli

    readers = Readers(0.9)
    pace = readers.pace()
    monkeypatch.setattr(clock, "pace", pace)
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    root.handlers.clear()  # basicConfig does nothing where the root has handlers (pytest's own)
    try:
        cli._logging(verbose)  # pyright: ignore[reportPrivateUsage]
        installed = list(root.handlers)
        assert len(installed) == 1
        assert cli._sampled in installed[0].filters  # pyright: ignore[reportPrivateUsage]
        assert root.level == (logging.DEBUG if verbose else logging.INFO)
        record = logging.LogRecord("seedvr2x", logging.INFO, __file__, 1, "a line", (), None)
        readers.tick(30.0)
        assert installed[0].filter(record)
        readers.tick(60.0)
        assert clock.spent(60.0) == pytest.approx(54.0)
    finally:
        root.handlers[:] = handlers
        root.setLevel(level)


def test_a_wall_step_of_59_s_inside_a_60_s_span_is_not_the_spans_seconds(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Under STEP_FLOOR, so no step is spotted, but 119 s of a 60 s span is outside SPAN_BAND: said
    # in the run's own seconds, at debug.
    caplog.set_level(logging.DEBUG, logger=clock.__name__)
    readers = Readers(1.0)
    pace = readers.pace()
    sampled(readers, pace, 10.0, 25.0)
    readers.step(59.0)
    sampled(readers, pace, 25.0)
    assert pace.spent(60.0) == 60.0
    assert "outside 0.8 to 1.25 of it" in caplog.text

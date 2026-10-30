"""One clock (DESIGN.md, Time estimates, "One clock", decided on 2026-10-28): the run times its
units by Python's monotonic clock, which is steady, and says durations, the time left, the time
of day a job should end, and checks --until, in the wall clock's time, by the ratio of the two
that it measures as it goes. Under WSL2 the VM's monotonic clock ran at 0.90 to 0.91 of the wall
clock on milestone 7's nights (a 5-min sleep took 333 s by the wall) and, on 2026-10-29, 10% fast
over 1,692 s (the wall advanced about 1,539 s), the wall clock itself, kept on Windows' by small
steps, stepping up to about 2 s either way, now and then back.

So the ratio is measured both ways, and a step of the wall clock is never a unit's time: nothing
measures a unit by the wall, the models learn in the run's own clock, and only what is said
converts (wall). A step of the wall clock so large that it is no measure of the run's time (the
time was set, or the VM resumed from sleep) is spotted and the measure begun again from there.

Until the ratio is measured (MIN_SPAN of the run's clock), a duration the run has just measured
is said by the wall clock's own seconds over that span (spent), as the times of day of the lines
around it are: "models loaded in 88.2 s" was said for 79 s by the lines' times (decided on
2026-10-29, DESIGN.md, Architecture, What it prints). The span's start is found among the
readings made as the run said things (sample), at the least one per line logged.

At the top level of the package, importing nothing of seedvr2x, so that the media layer and the
runtime layer both say their durations through it. Waits and timeouts that are not said (the
reader's and the writer's) stay on the monotonic clock.

PROVISIONAL (impl-v010, 2026-10-29: DESIGN.md says what is said in the wall's time; the models'
own figures at debug stay in the clock they are learnt in): the debug lines that hold a unit
against its model's estimate (runtime/run._made), the noise draw's and the phase's times, and
the calibration's fitted constants are in the run's own clock, both sides consistent."""

import logging
import time
from bisect import bisect_right
from collections.abc import Callable

logger = logging.getLogger(__name__)

# The monotonic time that must pass before the ratio is trusted: a step of the wall clock of
# 2.6 s moves it then by 2%, against the 10% drift left uncorrected in every duration said before
# (the first 600 s of a run were said in the run's own clock: "SHA-256 ... in 99.8 s" where the
# log's times gave 91 s). Before it, and after a wall-clock step until it is passed again, the
# last ratio is held (1.0 at first).
# PROVISIONAL (impl-v010, 2026-10-29: DESIGN.md leaves the span open; measured on 2026-10-29 on
# the RTX 5090 host: the wall clock steps by at most 2.6 s against Windows' clock between readings
# 60 s apart, so 2% at 120 s).
MIN_SPAN = 120.0
# The ratios that are believed. Outside it the clocks are not just drifting: the measure is
# said at debug and not used (1.0). The VM's drift measured was 0.90 to 1.10.
# PROVISIONAL (impl-v010, 2026-10-29: DESIGN.md leaves the band open).
BAND = (0.5, 2.0)
# A step of the wall clock between two readings is spotted when the wall advanced by more than
# STEP_FLOOR seconds and STEP_SHARE of the reading's span from what the ratio held predicts (the
# 2 s steps seen on 2026-10-29 are far under the floor; an hour's set of the clock is far over).
# PROVISIONAL (impl-v010, 2026-10-29: DESIGN.md leaves the bound open). Against the ratio held
# and not against 1.0, so that a drift of 10% over a long unit (20 min and more) isn't taken
# for a step.
STEP_FLOOR = 60.0
STEP_SHARE = 0.05
# The ratio is said at debug each time it moves by more than this share from the last said.
SAID_MOVE = 0.01
# The readings kept while no ratio is measured, to find where a span began (Pace.spent): the
# oldest dropped past this many. A line logged is a reading; a run's first MIN_SPAN seconds say
# a few hundred at the most.
# PROVISIONAL (impl-v010, 2026-10-29: DESIGN.md leaves the bound open).
HISTORY = 4_096


class Pace:
    """The run's clock against the wall's: wall(seconds) says a duration the run measured, in the
    wall clock's seconds. `monotonic` and `wall_clock` (the epoch time in seconds) are the two
    readers, injected by the tests. Every reading is made when something is said, which a unit's
    end does at the least (cli, runtime/run.py); no thread."""

    def __init__(
        self,
        monotonic: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self._monotonic, self._wall = monotonic, wall_clock
        self._anchor = (monotonic(), wall_clock())
        self._last = self._anchor
        self._ratio = 1.0
        self._measured = False  # a ratio inside BAND has been measured
        self._said = 1.0
        # The readings (run's clock, wall's) since the anchor, in order, kept while no ratio is
        # measured: where a span began, for spent.
        self._history: list[tuple[float, float]] = [self._anchor]

    def sample(self) -> None:
        """A reading of both clocks, kept to find where a later span began (spent), and nothing
        else: no ratio is judged and nothing logged, so that a logging filter may call it."""
        if self._measured:
            return
        mono, wall = self._monotonic(), self._wall()
        if self._history and mono < self._history[-1][0]:
            return  # a run's clock going back is ratio()'s to deal with
        self._history.append((mono, wall))
        if len(self._history) > HISTORY:
            del self._history[: len(self._history) - HISTORY]

    def ratio(self) -> float:
        """The wall clock's seconds in one of the run's, measured since the anchor, from the
        readings of now. 1.0 until MIN_SPAN of the run's clock has passed since the anchor (a
        re-anchor, after a step of the wall clock, holds the ratio until then), and where the
        measure is outside BAND."""
        mono, wall = self._monotonic(), self._wall()
        last_mono, last_wall = self._last
        span = mono - last_mono
        if span > 0:
            expected = span * self._ratio
            step = (wall - last_wall) - expected
            # While no ratio is measured (1.0 held), a long first gap is no step unless its own
            # ratio is outside BAND: a first reading 20 min and more after the import (a 4K
            # first pass) is held against 1.0 and a 10% drift is 2 min off it; read for a step,
            # it was re-anchored and the ratio kept at 1.0 ("the wall clock stepped by +167 s",
            # found by the review of 2026-10-29). A change of an hour is outside BAND for
            # any gap shorter than the change itself (an hour forward, two back).
            gap_ratio = (wall - last_wall) / span
            unmeasured_drift = not self._measured and BAND[0] <= gap_ratio <= BAND[1]
            if abs(step) > STEP_FLOOR + STEP_SHARE * expected and not unmeasured_drift:
                # Said at INFO: a step of a minute and more is the time set or the machine
                # resumed from sleep, which a time of day below it must be read with (DESIGN.md,
                # Architecture, What it prints, decided on 2026-10-28; built on 2026-10-29).
                logger.info(
                    "the wall clock stepped by %+.0f s (over %.0f s of the run's clock), the time"
                    " set or the machine resumed from sleep: the times of day below follow it,"
                    " and the clocks are measured against each other again from here",
                    step,
                    span,
                )
                self._anchor = (mono, wall)
                self._history.clear()
            self._last = (mono, wall)
        elif span < 0:
            # The run's clock steady, this can't be: measured again rather than trusted.
            logger.debug("the run's clock went back by %.3f s: measured again", -span)
            self._anchor = self._last = (mono, wall)
            self._history.clear()
        if not self._measured and (not self._history or mono >= self._history[-1][0]):
            self._history.append((mono, wall))
            del self._history[: max(len(self._history) - HISTORY, 0)]
        since = mono - self._anchor[0]
        if since >= MIN_SPAN:
            measured = (wall - self._anchor[1]) / since
            if BAND[0] <= measured <= BAND[1]:
                self._ratio = measured
                self._measured = True
            else:
                logger.debug(
                    "the clocks measured %.3f over %.0f s, outside %.1f to %.1f: not used",
                    measured,
                    since,
                    *BAND,
                )
                self._ratio = 1.0
                self._measured = False
            if abs(self._ratio - self._said) > SAID_MOVE * self._said:
                self._said = self._ratio
                logger.debug(
                    "the clocks: the run's own clock at %.3f of the wall's, over %s s",
                    1 / self._ratio,
                    f"{since:,.0f}",
                )
        return self._ratio

    def wall(self, seconds: float) -> float:
        """A duration of the run's clock, in the wall clock's seconds."""
        return seconds * self.ratio()

    def spent(self, seconds: float) -> float:
        """A span the run has just measured, `seconds` of its clock ending now, in the wall
        clock's seconds. With the ratio measured, wall; before, the span's own seconds by the wall
        clock: now's reading less the wall's at the span's start, found between the readings kept
        (sample, ratio), the clocks running evenly between two. Where the start is before every
        reading kept (the span began before the anchor, or the wall clock stepped since), or the
        wall's seconds are outside BAND of the run's (a step inside the span, absurd or
        negative), the run's own seconds, said at debug.

        PROVISIONAL (impl-v010, 2026-10-29: DESIGN.md says a duration comes from the wall clock
        before the ratio is measured and leaves the guard open): BAND, as for the ratio; a step of
        the wall clock inside a span, under STEP_FLOOR, goes into its seconds whole (2 s at the
        most, seen on 2026-10-29)."""
        ratio = self.ratio()
        if self._measured or seconds <= 0:
            return seconds * ratio
        mono, wall = self._history[-1] if self._history else (self._monotonic(), self._wall())
        started = mono - seconds
        after = bisect_right([reading[0] for reading in self._history], started)
        if after == 0 or after == len(self._history):
            logger.debug(
                "no reading of the clocks from before a span of %.1f s: said in the run's own",
                seconds,
            )
            return seconds
        (before_mono, before_wall), (next_mono, next_wall) = self._history[after - 1 : after + 1]
        began = before_wall + (next_wall - before_wall) * (started - before_mono) / (
            next_mono - before_mono
        )
        walled = wall - began
        if not BAND[0] * seconds <= walled <= BAND[1] * seconds:
            logger.debug(
                "a span of %.1f s of the run's clock was %.1f s by the wall's, outside %.1f to %.1f"
                " of it: said in the run's own",
                seconds,
                walled,
                *BAND,
            )
            return seconds
        return walled


# The process's clock, anchored when seedvr2x imports (a run's start); tests replace it.
pace = Pace()


def wall(seconds: float) -> float:
    """A duration the run measured, said in the wall clock's seconds (Pace.wall)."""
    return pace.wall(seconds)


def spent(seconds: float) -> float:
    """A span the run has just measured, ending now, in the wall clock's seconds (Pace.spent)."""
    return pace.spent(seconds)


def sample() -> None:
    """A reading of both clocks for the spans to come (Pace.sample)."""
    pace.sample()

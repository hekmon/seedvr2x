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

At the top level of the package, importing nothing of seedvr2x, so that the media layer and the
runtime layer both say their durations through it. Waits and timeouts that are not said (the
reader's and the writer's) stay on the monotonic clock.

PROVISIONAL (impl-v010, 2026-10-29: DESIGN.md says what is said in the wall's time; the models'
own figures at debug stay in the clock they are learnt in): the debug lines that hold a unit
against its model's estimate (runtime/run._made), the noise draw's and the phase's times, and
the calibration's fitted constants are in the run's own clock, both sides consistent."""

import logging
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

# The monotonic time that must pass before the ratio is trusted: a step of the wall clock of 2 s
# moves it then by 0.33%, under the 0.4% that a say of "about" hides. Before it, and after a
# wall-clock step until it is passed again, the last ratio is held (1.0 at first).
# PROVISIONAL (impl-v010, 2026-10-29: DESIGN.md leaves the span open; the figure keeps a
# 2 s step of the wall clock, as seen on 2026-10-29, under 0.4% of the ratio).
MIN_SPAN = 600.0
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
        self._said = 1.0

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
            if abs(step) > STEP_FLOOR + STEP_SHARE * expected:
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
            self._last = (mono, wall)
        elif span < 0:
            # The run's clock steady, this can't be: measured again rather than trusted.
            logger.debug("the run's clock went back by %.3f s: measured again", -span)
            self._anchor = self._last = (mono, wall)
        since = mono - self._anchor[0]
        if since >= MIN_SPAN:
            measured = (wall - self._anchor[1]) / since
            if BAND[0] <= measured <= BAND[1]:
                self._ratio = measured
            else:
                logger.debug(
                    "the clocks measured %.3f over %.0f s, outside %.1f to %.1f: not used",
                    measured,
                    since,
                    *BAND,
                )
                self._ratio = 1.0
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


# The process's clock, anchored when seedvr2x imports (a run's start); tests replace it.
pace = Pace()


def wall(seconds: float) -> float:
    """A duration the run measured, said in the wall clock's seconds (Pace.wall)."""
    return pace.wall(seconds)

"""A job's times (DESIGN.md, Memory planner, Time estimates): what each of its units should take,
for --until, which stops before a unit that wouldn't finish in time (runtime/stop.py), for --plan,
which prints the job's, and for each unit's progress line, which says the job's time left and
when it should end. Measured on the machine, never constants: the same GPU ran a token in 0.23 to
0.45 ms from one session to the next (research/docs/planner-limits.md, The DiT's time per token).

One model per kind of unit (Model), a constant plus a rate: a run's encode and its decode by the
frame, a window by the token; and the terms a unit pays besides it, measured like the rest: the
change of phase a window or a decode opens, a constant per phase (ENTER), and a run's first
window's noise draw, a rate per token (DESIGN.md, Memory planner, Time estimates, decided on
2026-10-21). An encode's change of phase is in its own model, its constant: a run's encodes
always follow one (decided on 2026-10-22). A calibration
sets them (runtime/calibration.py), which every run makes, --plan included. After every unit
(Clock.observe), a kind's model is corrected by the median of its last RATIOS ratios of measured
to predicted time, which follows a card whose speed drifts and forgets a unit that went another
way.

A job's units (units_of) are listed in the order the run makes them (run.run_job): each run's
encode and windows, a segment's runs' in turn, then the segment's decode and write, one unit
made of its runs' decodes; and a one-file output's assembly, the job's last. Progress holds what
is left of them as the run goes."""

import logging
import math
import os
import statistics
import tempfile
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from seedvr2x.runtime.job import OutputSegment, Shot, padded_size, shot_runs
from seedvr2x.runtime.plan import Phase
from seedvr2x.runtime.shot import padded_length, shot_layout
from seedvr2x.runtime.stop import late

logger = logging.getLogger(__name__)

Kind = Literal["encode", "window", "decode", "enter-dit", "enter-decode", "draw"]
# The kinds of unit, each a model of its own (DESIGN.md, Memory planner, Time estimates).
KINDS: tuple[Kind, ...] = ("encode", "window", "decode")
# The terms a unit pays besides its kind's model, measured like the rest (DESIGN.md, the same
# section, decided on 2026-10-21): the change of phase a unit opens (Models.enter), a constant per
# phase entered, and a run's first window's noise draw, a rate per token of the run's latent. None
# for the encode's phase: a run's encodes always follow a change of phase, which the encode's
# model holds, calibrated so (decided on 2026-10-22: calibrated after an encode, the encode's
# constant came out 0.10 s for 0.23 to 0.66 s calibrated after a window, and a job's short
# encodes ran 0.5 to 0.7 s over their estimates; 1080p, RTX 5090).
ENTER: dict[Phase, Kind] = {"dit": "enter-dit", "decode": "enter-decode"}
TERMS: tuple[Kind, ...] = (*ENTER.values(), "draw")
# The phase of each kind of unit (Models.enter).
PHASES: dict[Kind, Phase] = {"encode": "encode", "window": "dit", "decode": "decode"}

# The ratios of measured to predicted time whose median corrects a kind's model (DESIGN.md, Memory
# planner, Time estimates): the last 5.
RATIOS = 5

# The pixels a token's side spans: the VAE's 8 (video_vae_v3/s8_c16_t4_inflation_sd3.yaml,
# spatial_downsample_factor), then the DiT's patches of 2 latent pixels (configs_7b/main.yaml,
# patch_size [1, 2, 2]; the 3B's alike).
TOKEN = 16


@dataclass(frozen=True)
class Model:
    """What a kind of unit takes: constant + rate x size seconds, size in frames, padded to
    4n + 1 as the VAE takes them, for an encode and a decode, in tokens for a window."""

    constant: float
    rate: float

    def at(self, size: int) -> float:
        return self.constant + self.rate * size


class Clock:
    """What a job's units take, in seconds (the module's docstring): estimate gives a unit's,
    None while nothing says what its kind takes; observe takes each unit's time once it is made.
    `models` are a calibration's (calibration.calibrate), each kind's then counted as a first
    ratio of 1 (DESIGN.md: "It counts as a first ratio of 1 among the last 5")."""

    def __init__(self, models: Mapping[Kind, Model] | None = None) -> None:
        self._models: dict[Kind, Model] = dict(models or {})
        self._ratios: dict[Kind, deque[float]] = {
            kind: deque(maxlen=RATIOS) for kind in (*KINDS, *TERMS)
        }
        # A calibration is a kind's first measure, at a ratio of 1 to its own model: counted among
        # the last ratios, so that the first unit alone doesn't set the correction. On a job of 78
        # units (RTX 5090, 432p, 2026-10-10), whose first decode took 1.5 times its estimate, the
        # next one's came out 21% over with it, 42% without, and the decodes' mean error 6.4%
        # against 8.3%.
        for kind in self._models:
            self._ratios[kind].append(1.0)
        self.calibrated = bool(self._models)
        self._units: dict[Kind, int] = dict.fromkeys((*KINDS, *TERMS), 0)
        self._record: dict[str, object] = {}
        for kind in self._models:
            self._said(kind)

    def estimate(self, kind: Kind, size: int) -> float | None:
        """A unit of `kind` of `size`, frames for an encode or a decode (padded here, as the VAE
        takes them), tokens for a window (tokens): its model's time, corrected."""
        model = self._models.get(kind)
        if model is None:
            # A term nothing measured yet adds nothing: a kind's model unknown, no estimate.
            return 0.0 if kind in TERMS else None
        return model.at(_sized(kind, size)) * self._correction(kind)

    def observe(self, kind: Kind, size: int, seconds: float) -> None:
        """A unit made in `seconds`, of `size` as estimate takes it. Its ratio to what the kind's
        model predicts, uncorrected, joins the last RATIOS; the first unit of a kind nothing
        calibrated is its model, by its rate alone."""
        size = _sized(kind, size)
        self._units[kind] += 1
        model = self._models.get(kind)
        if model is None:
            self._models[kind] = Model(0.0, seconds / size)
        elif model.at(size) > 0:
            self._ratios[kind].append(seconds / model.at(size))
        self._said(kind)

    def knows(self, kind: Kind) -> bool:
        return kind in self._models

    def record(self) -> dict[str, object]:
        """The models as the manifest records them, for information (DESIGN.md, Memory planner):
        per kind that has one, its constant in seconds, its rate in seconds per frame or per
        token, its correction, the median of its last ratios (1 without any), and the units
        observed. One dict, kept current as units are observed, so that the manifest holding it
        says the latest at each of its writes."""
        return self._record

    def _said(self, kind: Kind) -> None:
        model = self._models[kind]
        self._record[kind] = {
            "constant": model.constant,
            "rate": model.rate,
            "correction": self._correction(kind),
            "units": self._units[kind],
        }

    def _correction(self, kind: Kind) -> float:
        ratios = self._ratios[kind]
        return statistics.median(ratios) if ratios else 1.0


def _sized(kind: Kind, size: int) -> int:
    """size as a kind's model takes it: an encode's or a decode's frames padded to 4n + 1."""
    return padded_length(size) if kind in ("encode", "decode") else size


def tokens(target: tuple[int, int], latents: int, *, numz_padding: bool = False) -> int:
    """The tokens of a window of `latents` latents for frames resized to target, (height, width):
    a token per TOKEN x TOKEN pixels of the padded frame (job.padded_size), per latent.
    numz_padding is for tests only (cli.NUMZ_PADDING): numz's padding, up to multiples of 16."""
    height, width = target
    if numz_padding:
        height, width = -(-height // TOKEN) * TOKEN, -(-width // TOKEN) * TOKEN
    else:
        height, width = padded_size(target)
    return latents * (height // TOKEN) * (width // TOKEN)


def run_name(shot: int, shots: int, position: int, runs: int, offset: int, per_shot: int) -> str:
    """A run's name as the logs give it: its shot's, "shot 3/12"; with several runs a shot
    (`per_shot`), the run's too, "shot 3/12, run 2/4 (from its frame 1)", `position` its place
    among the shot's own `runs` (a shot too short for some has fewer than per_shot)."""
    name = f"shot {shot + 1}/{shots}"
    if per_shot == 1:
        return name
    return f"{name}, run {position + 1}/{runs} (from its frame {offset})"


# A unit's key (units_of): ("encode", run), ("window", run, window) or ("decode", segment).
Key = tuple[str | int, ...]


@dataclass(frozen=True)
class Unit:
    """A unit of the job (DESIGN.md, Pause and resume), named as the run names it, and what it is
    made of: each part a kind and a size, one for an encode or a window, a run's decode each
    for a segment's decode and write, its shots' runs in turn."""

    key: Key
    name: str
    parts: tuple[tuple[Kind, int], ...]

    @property
    def phase(self) -> Kind:
        """The kind the unit is of, its terms aside: a segment's decode and write is a decode."""
        return next(kind for kind, _ in self.parts if kind in KINDS)


def units_of(
    shots: Sequence[Shot],
    segments: Sequence[OutputSegment],
    per_shot: int,
    window: int | None,
    per_latent: int,
    *,
    encoded: Sequence[bool] | None = None,
    windows_done: Sequence[int] | None = None,
    finished: Sequence[bool] | None = None,
    start: Phase | None = None,
    layouts: Sequence[Sequence[tuple[int, int]]] | None = None,
) -> list[Unit]:
    """The units a job has left, in the order the run makes them (run.run_job): its shots, each
    run `per_shot` times (job.shot_runs), their windows capped at `window` latents
    (shot.shot_layout) of per_latent tokens each (tokens); each segment's runs' encodes and
    windows in turn, then its decode and write. What the job's record says is made already is
    left out, as the run skips it: a run's encode (encoded, run by run, as the manifest's), its
    windows done (windows_done), a finished segment (finished).

    Each unit but an encode, whose model holds it (ENTER), pays, as parts of its own, the change
    of phase it opens, where the unit before it is of another phase (the first unit's against
    `start`, the phase the models are in, None for one unknown), and a run's first window left,
    its run's noise draw, by the tokens of the run's whole latent (shot.sample_windows draws it
    with that window).

    layouts: each run's windows as the job records them, in the order of runs, where a resume
    planned again for the memory free (--accept-env-change) left runs by the cap they were made
    under; else they are laid out by `window`."""
    units: list[Unit] = []
    run = 0  # the runs' index across the job, as the manifest's
    for number, segment in enumerate(segments):
        group = [i for i, shot in enumerate(shots) if segment.start <= shot.start < segment.end]
        first_run = run
        decodes: list[tuple[Kind, int]] = []
        for index in group:
            runs = shot_runs(shots[index], per_shot)
            for position, each in enumerate(runs):
                name = run_name(index, len(shots), position, len(runs), each.offset, per_shot)
                layout = shot_layout(each.frames, window) if layouts is None else layouts[run]
                done = 0 if windows_done is None else windows_done[run]
                if done < len(layout) and not (encoded is not None and encoded[run]):
                    units.append(
                        Unit(("encode", run), f"{name}'s encode", (("encode", each.frames),))
                    )
                for k in range(done, len(layout)):
                    first, end = layout[k]
                    parts: tuple[tuple[Kind, int], ...] = (("window", (end - first) * per_latent),)
                    if k == done:
                        drawn: tuple[Kind, int] = ("draw", layout[-1][1] * per_latent)
                        parts += (drawn,)
                    said = f"{name}'s window {k + 1}/{len(layout)}"
                    units.append(Unit(("window", run, k), said, parts))
                decodes.append(("decode", each.frames))
                run += 1
        if finished is not None and finished[number]:
            # Its runs' encodes and windows were made too: a finished segment's are kept by none.
            del units[len(units) - _made_since(units, first_run) :]
            continue
        said = f"segment {number + 1}/{len(segments)}'s decode and write"
        units.append(Unit(("decode", number), said, tuple(decodes)))
    return _entering(units, start)


def _entering(units: Sequence[Unit], start: Phase | None) -> list[Unit]:
    """units, each that opens a change of phase given its part for it (units_of), but an encode,
    whose model holds its change (ENTER)."""
    entering: list[Unit] = []
    current = start
    for unit in units:
        phase = PHASES[unit.phase]
        if phase != current and phase in ENTER:
            unit = Unit(unit.key, unit.name, ((ENTER[phase], 1), *unit.parts))
        current = phase
        entering.append(unit)
    return entering


def _made_since(units: Sequence[Unit], first_run: int) -> int:
    """How many of units' last entries are encodes or windows of runs from first_run on."""
    count = 0
    for unit in reversed(units):
        if unit.phase == "decode" or int(unit.key[1]) < first_run:
            break
        count += 1
    return count


@dataclass(frozen=True)
class Times:
    """A job's estimate, in seconds: its encodes, its windows, its decodes with their writes, and
    its assembly; and its longest unit, named as the run names it, which is how early --until may
    stop (DESIGN.md, Pause and resume). None where nothing estimates a kind."""

    encode: float
    windows: float
    decode: float
    longest: float
    unit: str

    @property
    def total(self) -> float:
        return self.encode + self.windows + self.decode


def estimate(clock: Clock, unit: Unit) -> float | None:
    """unit's time by clock: its parts', None if a kind of them has no model yet."""
    total = 0.0
    for kind, size in unit.parts:
        seconds = clock.estimate(kind, size)
        if seconds is None:
            return None
        total += seconds
    return total


def job_times(clock: Clock, units: Sequence[Unit]) -> Times | None:
    """The estimate of units by clock, phase by phase (Times); None while the clock lacks a kind's
    model."""
    phases = dict.fromkeys(KINDS, 0.0)
    longest, name = 0.0, ""
    for unit in units:
        seconds = estimate(clock, unit)
        if seconds is None:
            return None
        phases[unit.phase] += seconds
        if seconds > longest:
            longest, name = seconds, unit.name
    return Times(phases["encode"], phases["window"], phases["decode"], longest, name)


class Progress:
    """What is left of a job's units as the run makes them (units_of), and the clock that says
    what they take: cost gives a unit's estimate before it begins (for --until), made takes each
    of its parts' time once done, which the clock is corrected by, and left says the job's time
    left, which each unit's progress line gives with the time of day it should end (DESIGN.md,
    Memory planner, Time estimates: "Each unit's progress line says the job's time left and the
    time of day it should end"). `extra` is what a one-file output's assembly adds after the last
    unit, in seconds, None while nothing estimates it."""

    def __init__(
        self,
        clock: Clock,
        units: Sequence[Unit],
        now: Callable[[], datetime],
        extra: Callable[[], float | None] | None = None,
    ) -> None:
        self.clock, self.now = clock, now
        self._parts: dict[Key, list[tuple[Kind, int]]] = {}
        self._order: list[Key] = []
        for unit in units:
            self._parts[unit.key] = list(unit.parts)
            self._order.append(unit.key)
        self._listed = set(self._order)
        self._extra = extra
        # Whether the run's next encode is its first (run._sample): a resumed run's first seek
        # isn't counted.
        self.first_encode = True

    def cost(self, key: Key) -> float | None:
        """The estimate of what is left of unit `key`, None without one, or for a unit not listed:
        one the run makes that units_of doesn't count, a seedvr2x bug, warned of under a
        calibrated clock, where --until then holds that unit to its time alone."""
        parts = self._parts.get(key)
        if not parts:
            if self.clock.calibrated and key not in self._listed:
                logger.warning("%s: a unit the time estimates don't list, a seedvr2x bug", key)
            return None
        return estimate(self.clock, Unit(key, "", tuple(parts)))

    def lists(self, key: Key, kind: Kind) -> bool:
        """Whether unit `key` has a part of `kind` left."""
        return any(part == kind for part, _ in self._parts.get(key, ()))

    def made(self, key: Key, kind: Kind, size: int, seconds: float) -> None:
        """A part of unit `key` made in `seconds`: given to the clock (Clock.observe), and no
        longer left."""
        self.clock.observe(kind, size, seconds)
        self.skipped(key, kind, size)

    def skipped(self, key: Key, kind: Kind, size: int) -> None:
        """A part of unit `key` made, its time not given to the clock: no longer left."""
        parts = self._parts.get(key)
        if parts and (kind, size) in parts:
            parts.remove((kind, size))
            if not parts:
                del self._parts[key]

    def left(self) -> float | None:
        """The job's time left, in seconds: its units' left, then the assembly; None while a kind
        of them has no model."""
        total = 0.0
        for key in self._order:
            if key in self._parts:
                seconds = self.cost(key)
                if seconds is None:
                    return None
                total += seconds
        if self._extra is not None:
            extra = self._extra()
            if extra is None:
                return None
            total += extra
        return total

    def said(self) -> str:
        """The job's time left and its end, in plain words, for a unit's progress line: "about 1
        h 12 min left, the job ending about 07:42"; while a kind has no model yet (a run not
        calibrated, DESIGN.md: "any other run takes its first units' times"), the kind awaited."""
        if not self._parts and self._extra is None:
            return "the job's last unit made"
        seconds = self.left()
        if seconds is None:
            missing = [kind for kind in KINDS if not self.clock.knows(kind)]
            what = {"encode": "an encode", "window": "a window", "decode": "a decode"}
            first = what[missing[0]] if missing else "the assembly"
            return f"the job's time left known once {first} is timed"
        now = self.now()
        return f"about {spoken(seconds)} left, the job ending about {ending(now, seconds)}"


def spoken(seconds: float) -> str:
    """A duration as the logs say it: "42 s", "12 min 5 s", "3 h 7 min", "2 d 4 h": its two
    largest units, rounded to the smaller."""
    total = round(seconds)
    if total < 60:
        return f"{total} s"
    if total < 3600:
        return f"{total // 60} min {total % 60} s"
    minutes = round(seconds / 60)
    if minutes < 24 * 60:
        return f"{minutes // 60} h {minutes % 60} min"
    hours = round(seconds / 3600)
    return f"{hours // 24} d {hours % 24} h"


def ending(now: datetime, seconds: float) -> str:
    """The time of day `seconds` after now, as the logs say a job's end: "07:42" today, "07:42
    tomorrow", "07:42 on Fri 23 Oct" later; to the minute, rounded up; in the local zone, as its
    clock will show it, across a change of the clock too."""
    # In the local zone's own time: an aware time plus a duration keeps its offset, an hour off
    # across a clock change.
    end = (now + timedelta(seconds=math.ceil(seconds))).astimezone()
    if end.second or end.microsecond:
        end = end.replace(second=0, microsecond=0) + timedelta(minutes=1)
    days = (end.date() - now.astimezone().date()).days
    if days == 0:
        return f"{end:%H:%M}"
    if days == 1:
        return f"{end:%H:%M} tomorrow"
    return f"{end:%H:%M} on {end:%a %d %b}"


# The bytes the disk's rate is measured by (disk_rate): written and synced, beside the output.
PROBE = 256 * 1024**2
# The probe's writes, a block at a time.
BLOCK = 8 * 1024**2


def disk_rate(directory: Path) -> float:
    """The rate the disk of `directory` writes at, in bytes per second: PROBE bytes written to a
    temporary file there, then synced (os.fsync), the file removed; for a one-file output's
    assembly (assembly_seconds), whose bytes go through the disk its masters are on. A file
    beside the output, not in the job's directory, which holds nothing but what its manifest
    names: a kill during the probe leaves it there, named .seedvr2x-disk-rate-*.
    PROVISIONAL (implementation, 2026-10-21): DESIGN.md says "estimated from the masters' size",
    not at what rate; the probe's size, a second or so on a hard disk."""
    block = os.urandom(BLOCK)  # data a filesystem can't compress
    begun = time.monotonic()
    with tempfile.NamedTemporaryFile(dir=directory, prefix=".seedvr2x-disk-rate-") as file:
        for _ in range(PROBE // BLOCK):
            file.write(block)
        file.flush()
        os.fsync(file.fileno())
        seconds = time.monotonic() - begun
    return PROBE / max(seconds, 1e-6)


def assembly_seconds(rate: float, masters: int, source: int) -> float:
    """A one-file output's assembly (media/assembly.assemble), in seconds, at the disk's rate
    (disk_rate, bytes per second), from the bytes its masters take, `masters`, and the source's,
    `source`: the join reads the masters and writes them out, a stream copy, and the check
    reads the file back, three times the masters' bytes; the source is read twice, once for its
    other streams alone (the reference the result is checked against), once by the join. Every
    byte counted at the write's rate, a disk reading no slower than it writes.
    PROVISIONAL (implementation, 2026-10-21)."""
    return (3 * masters + 2 * source) / rate


@dataclass(frozen=True)
class Ending:
    """When a job run from `start` would end by its estimate (would_end), or the unit it would
    stop before under --until's rule (stop.late), `before`, None if none: at `at`, its work
    done until then `done` seconds of `total`."""

    before: str | None
    at: datetime
    done: float
    total: float


def would_end(
    clock: Clock,
    units: Sequence[Unit],
    start: datetime,
    deadline: datetime,
    last: tuple[str, float] | None = None,
) -> Ending:
    """A job of units run from start, its units taking their estimates, under --until's rule:
    each unit begun only if it ends by deadline at MARGIN times its estimate (stop.late); `last`,
    a one-file output's assembly, its name and estimate, after them. For --plan given --until
    (DESIGN.md, Pause and resume, Stopping: "`--plan` given `--until` says whether the job ends by
    then and, if not, the unit it would stop before")."""
    costs: list[tuple[str, float]] = []
    for unit in units:
        seconds = estimate(clock, unit)
        if seconds is None:
            raise ValueError(f"{unit.name}: no estimate")
        costs.append((unit.name, seconds))
    if last is not None:
        costs.append(last)
    total = sum(seconds for _, seconds in costs)
    at, done = start, 0.0
    for name, seconds in costs:
        if late(at, seconds, deadline):
            return Ending(name, at, done, total)
        at += timedelta(seconds=seconds)
        done += seconds
    return Ending(None, at, done, total)

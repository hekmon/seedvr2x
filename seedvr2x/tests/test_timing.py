"""A job's times (runtime/timing.py, runtime/calibration.py; DESIGN.md, Memory planner, Time
estimates), on the CPU: the models and their correction, the job's units and their estimate, the
progress line's words, --plan's ending, and the calibration on steps that move a clock of their
own."""

import logging
import os
import tempfile
import time
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest
import torch

from seedvr2x.media.checksums import read_checksums
from seedvr2x.runtime import calibration, planner, run, timing
from seedvr2x.runtime.calibration import FIRST, FRAMES, SHORT, Calibrated, calibrate
from seedvr2x.runtime.job import OutputSegment, Shot, padded_size, shot_runs
from seedvr2x.runtime.plan import Plan, Tiles
from seedvr2x.runtime.shot import Correction, shot_layout
from seedvr2x.runtime.timing import (
    RATIOS,
    Clock,
    Kind,
    Model,
    Progress,
    Unit,
    assembly_seconds,
    disk_rate,
    ending,
    job_times,
    run_name,
    spoken,
    tokens,
    units_of,
    would_end,
)

# An encode at 2 s and 0.5 s a frame, a window at 1 s and 0.25 ms a token, a decode at 3 s and
# 1.5 s a frame.
MODELS: dict[Kind, Model] = {
    "encode": Model(2.0, 0.5),
    "window": Model(1.0, 0.00025),
    "decode": Model(3.0, 1.5),
}

# The terms a unit pays besides its kind: entering the DiT's phase and the decode's, a constant,
# and the noise's draw, a rate per token. None for the encode's phase: its change is in the
# encode's own model (timing.ENTER).
TERMS: dict[Kind, Model] = {
    "enter-dit": Model(1.5, 0.0),
    "enter-decode": Model(0.7, 0.0),
    "draw": Model(0.0, 0.001),
}

START = datetime(2026, 10, 21, 10, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def zone_utc() -> Generator[None]:
    """The local zone, which a job's end is said in (timing.ending), set to UTC for every test
    here, unless a test sets another; the process's given back."""
    yield from zone("UTC")


def zone(name: str) -> Generator[None]:
    before = os.environ.get("TZ")
    os.environ["TZ"] = name
    time.tzset()
    try:
        yield
    finally:
        if before is None:
            del os.environ["TZ"]
        else:
            os.environ["TZ"] = before
        time.tzset()


def test_ending_across_a_change_of_the_clock() -> None:
    # Paris's clock goes back from 03:00 CEST to 02:00 CET on 2026-10-25: two hours after 01:30
    # CEST, it shows 02:30 (a review's finding: the offset of the start kept, it said 03:30).
    for _ in zone("Europe/Paris"):
        start = datetime(2026, 10, 25, 1, 30).astimezone()
        assert timing.ending(start, 2 * 3600) == "02:30"


def test_model_at() -> None:
    assert Model(2.0, 0.5).at(13) == 8.5
    assert Model(0.0, 0.25).at(0) == 0.0


def test_a_constant_plus_a_rate() -> None:
    clock = Clock(MODELS)
    assert clock.estimate("encode", 13) == 2.0 + 0.5 * 13
    assert clock.estimate("window", 16_560) == 1.0 + 0.00025 * 16_560
    assert clock.estimate("decode", 13) == 3.0 + 1.5 * 13


@pytest.mark.parametrize(("frames", "padded"), [(1, 1), (2, 5), (6, 9), (13, 13), (14, 17)])
def test_frames_as_the_vae_takes_them(frames: int, padded: int) -> None:
    # The VAE encodes and decodes a shot padded to 4n + 1 frames (shot.padded_length): a shot of
    # 6 frames takes a shot of 9's time. A window's size is in tokens, taken as it is.
    clock = Clock(MODELS)
    assert clock.estimate("encode", frames) == 2.0 + 0.5 * padded
    assert clock.estimate("decode", frames) == 3.0 + 1.5 * padded
    assert clock.estimate("window", frames) == 1.0 + 0.00025 * frames


def test_nothing_known_nothing_estimated() -> None:
    clock = Clock()
    assert [clock.estimate(kind, 13) for kind in timing.KINDS] == [None, None, None]
    assert not clock.calibrated and not clock.knows("encode")
    assert clock.record() == {}


def test_a_term_nothing_measured_adds_nothing() -> None:
    # A kind unknown has no estimate; a term unknown costs 0 s, so that a clock without it
    # estimates a unit by its kind alone.
    clock = Clock()
    assert clock.estimate("window", 1_000) is None
    assert [clock.estimate(term, 1_000) for term in timing.TERMS] == [0.0] * 3
    assert not clock.knows("draw")
    known = Clock(TERMS)
    assert known.estimate("enter-dit", 1) == 1.5
    # No term for the encode's phase: its change is in the encode's model.
    assert "enter-encode" not in timing.TERMS
    assert known.estimate("draw", 900) == pytest.approx(0.9)
    # Terms are not padded as frames are.
    assert known.estimate("enter-decode", 7) == pytest.approx(0.7)
    assert known.estimate("window", 1_000) is None
    assert known.record()["draw"] == {"constant": 0.0, "rate": 0.001, "correction": 1.0, "units": 0}


def test_a_kind_nothing_calibrated_takes_its_first_unit_by_its_rate() -> None:
    # "any other run takes its first units' times".
    clock = Clock()
    clock.observe("encode", 6, 4.5)  # 6 frames are 9 to the VAE: 0.5 s a frame, no constant
    assert clock.knows("encode") and not clock.knows("window")
    assert clock.estimate("encode", 13) == pytest.approx(6.5)
    assert clock.estimate("window", 4_320) is None
    clock.observe("window", 4_320, 2.16)
    assert clock.estimate("window", 8_640) == pytest.approx(4.32)
    # Its first unit isn't a ratio: the second is the first one.
    assert clock.record()["encode"] == {"constant": 0.0, "rate": 0.5, "correction": 1.0, "units": 1}
    clock.observe("encode", 9, 2 * 4.5)
    assert clock.estimate("encode", 13) == pytest.approx(2 * 6.5)


def test_a_calibration_counts_as_a_first_ratio_of_one() -> None:
    clock = Clock(MODELS)
    assert clock.calibrated
    predicted = 8.5
    # One unit twice as slow: the median of 1 and 2, so that one unit alone doesn't set the
    # correction.
    clock.observe("encode", 13, 2 * predicted)
    assert clock.estimate("encode", 13) == pytest.approx(1.5 * predicted)
    assert clock.estimate("encode", 25) == pytest.approx(1.5 * (2.0 + 0.5 * 25))
    # The other kinds keep theirs.
    assert clock.estimate("decode", 13) == 3.0 + 1.5 * 13


def test_only_the_last_ratios_count() -> None:
    assert RATIOS == 5
    clock = Clock(MODELS)
    # Each ratio is against the model uncorrected, whatever the correction in force.
    for ratio, median in [(10, 5.5), (20, 10), (30, 15), (40, 20)]:
        clock.observe("encode", 13, ratio * 8.5)
        assert clock.estimate("encode", 13) == pytest.approx(median * 8.5), ratio
    # Now 1, 10, 20, 30, 40: the calibration's 1 is still among the last 5; a sixth ratio pushes
    # it out (otherwise the median of six would be 25).
    clock.observe("encode", 13, 50 * 8.5)
    assert clock.estimate("encode", 13) == pytest.approx(30 * 8.5)


def test_a_card_20_percent_slower_stays_20_percent_slower() -> None:
    clock = Clock(MODELS)
    for _ in range(12):
        clock.observe("window", 16_560, 1.2 * (1.0 + 0.00025 * 16_560))
    assert clock.estimate("window", 33_120) == pytest.approx(1.2 * (1.0 + 0.00025 * 33_120))


def test_the_observed_size_is_padded_too() -> None:
    clock = Clock(MODELS)
    # 2 frames are 5 to the VAE: a decode of 2 frames in the time of 5 is on the model.
    clock.observe("decode", 2, 3.0 + 1.5 * 5)
    assert clock.estimate("decode", 5) == 3.0 + 1.5 * 5


def test_record() -> None:
    clock = Clock(MODELS)
    record = clock.record()
    assert record["encode"] == {"constant": 2.0, "rate": 0.5, "correction": 1.0, "units": 0}
    assert record["window"] == {"constant": 1.0, "rate": 0.00025, "correction": 1.0, "units": 0}
    # One dict, kept current: the manifest holding it says the latest at each of its writes.
    for _ in range(2):
        clock.observe("window", 16_560, 1.1 * (1.0 + 0.00025 * 16_560))
    assert clock.record() is record
    assert record["window"] == {
        "constant": 1.0,
        "rate": 0.00025,
        "correction": pytest.approx(1.1),
        "units": 2,
    }
    # A kind nothing calibrated appears with its first unit.
    learnt = Clock()
    record = learnt.record()
    assert record == {}
    learnt.observe("decode", 13, 26.0)
    assert record == {"decode": {"constant": 0.0, "rate": 2.0, "correction": 1.0, "units": 1}}


def test_tokens() -> None:
    # 1080x1920 is padded to 1104x1920 (job.padded_size): 69 x 120 tokens a latent; numz pads it
    # to 1088 rows, 68 x 120, the 48,960 tokens of planner-limits.md's 6 latents.
    assert tokens((1080, 1920), 1) == 69 * 120
    assert tokens((1080, 1920), 6, numz_padding=True) == 6 * 68 * 120 == 48_960
    assert tokens((1080, 1920), 6) == 6 * 69 * 120
    assert tokens((540, 960), 2) == 2 * 36 * 60
    # Columns padded too when the width isn't a multiple of 16: 1060x1912 to 1088x1936; numz
    # pads it to 1072x1912's multiples, 1072x1920.
    assert tokens((1060, 1912), 3) == 3 * 68 * 121
    assert tokens((1060, 1912), 3, numz_padding=True) == 3 * 67 * 120


# Shots of 3, 1 and 21 frames in two segments, as test_cli_run's JOB makes them; windows capped
# at 5 latents, of 100 tokens each.
SHOTS = [Shot(0, 3), Shot(3, 4), Shot(4, 25)]
SEGMENTS = [OutputSegment("seg_000000", 0, 4), OutputSegment("seg_000001", 4, 25)]
LONG = shot_layout(21, 5)  # 6 latents in two windows


def test_the_layout_is_what_the_tests_expect() -> None:
    assert [shot_layout(shot.frames, 5) for shot in SHOTS[:2]] == [[(0, 2)], [(0, 1)]]
    assert len(LONG) == 2 and LONG[-1][1] == 6


def test_units_of_one_run_a_shot() -> None:
    long_windows = [(end - start) * 100 for start, end in LONG]
    expected: list[tuple[timing.Key, str, tuple[tuple[Kind, int], ...]]] = [
        # Each window or decode that changes the phase pays for it, an encode not (its change is in
        # its model); a run's first window, its run's draw (the tokens of its whole latent).
        (("encode", 0), "shot 1/3's encode", (("encode", 3),)),
        (
            ("window", 0, 0),
            "shot 1/3's window 1/1",
            (("enter-dit", 1), ("window", 200), ("draw", 200)),
        ),
        (("encode", 1), "shot 2/3's encode", (("encode", 1),)),
        (
            ("window", 1, 0),
            "shot 2/3's window 1/1",
            (("enter-dit", 1), ("window", 100), ("draw", 100)),
        ),
        (
            ("decode", 0),
            "segment 1/2's decode and write",
            (("enter-decode", 1), ("decode", 3), ("decode", 1)),
        ),
        (("encode", 2), "shot 3/3's encode", (("encode", 21),)),
        (
            ("window", 2, 0),
            "shot 3/3's window 1/2",
            (("enter-dit", 1), ("window", long_windows[0]), ("draw", 600)),
        ),
        # The second window is of the phase the first left the models in.
        (("window", 2, 1), "shot 3/3's window 2/2", (("window", long_windows[1]),)),
        (
            ("decode", 1),
            "segment 2/2's decode and write",
            (("enter-decode", 1), ("decode", 21)),
        ),
    ]
    units = units_of(SHOTS, SEGMENTS, 1, 5, 100)
    assert [(unit.key, unit.name, unit.parts) for unit in units] == expected
    # A unit's phase is its kind's, its terms aside.
    assert [unit.phase for unit in units] == [
        *("encode", "window", "encode", "window", "decode", "encode", "window", "window"),
        "decode",
    ]
    # A segment's decode pays one change of phase, however many runs it holds.
    assert [part for part in units[4].parts if part[0] == "enter-decode"] == [("enter-decode", 1)]
    # Named as the run names a shot.
    assert units[0].name == f"{run_name(0, 3, 0, 1, 0, 1)}'s encode"


def test_units_of_takes_each_run_s_windows_from_the_layouts_given() -> None:
    # A job whose shape was planned again for the memory free keeps runs by the windows they
    # were made under (manifest.layouts): the units are listed by them, not by the plan's cap.
    layouts = [shot_layout(shot.frames, 5) for shot in SHOTS]
    assert units_of(SHOTS, SEGMENTS, 1, 5, 100, layouts=layouts) == units_of(
        SHOTS, SEGMENTS, 1, 5, 100
    )
    layouts[2] = [(0, 6)]
    units = units_of(SHOTS, SEGMENTS, 1, 99, 100, layouts=layouts)
    windows = [unit for unit in units if unit.key[0] == "window" and unit.key[1] == 2]
    assert [unit.name for unit in windows] == ["shot 3/3's window 1/1"]
    assert ("window", 600) in windows[0].parts


def test_units_of_the_first_unit_pays_against_the_start_phase() -> None:
    # The models' phase as the run finds them: a first unit pays unless they are in its phase.
    # The purpose was the first encode's term; an encode has none now (its change is in its
    # model), so it is kept with the first window and the first decode, the units that do.
    for start in (None, "encode", "decode", "dit"):
        assert units_of(SHOTS, SEGMENTS, 1, 5, 100, start=start)[0].parts == (("encode", 3),)
    window_first = units_of(SHOTS[:1], SEGMENTS[:1], 1, 5, 100, encoded=[True], start="encode")
    assert window_first[0].parts[0] == ("enter-dit", 1)
    for start in (None, "encode", "decode"):
        parts = units_of(SHOTS[:1], SEGMENTS[:1], 1, 5, 100, encoded=[True], start=start)[0].parts
        assert parts[0] == ("enter-dit", 1)
    in_dit = units_of(SHOTS[:1], SEGMENTS[:1], 1, 5, 100, encoded=[True], start="dit")
    assert in_dit[0].parts[0] == ("window", 200)
    decode_first = units_of(
        SHOTS[:1], SEGMENTS[:1], 1, 5, 100, encoded=[True], windows_done=[1], start="dit"
    )
    assert decode_first[0].parts[0] == ("enter-decode", 1)
    in_decode = units_of(
        SHOTS[:1], SEGMENTS[:1], 1, 5, 100, encoded=[True], windows_done=[1], start="decode"
    )
    assert in_decode[0].parts[0] == ("decode", 3)
    # And the units after the first, the same.
    after = units_of(SHOTS, SEGMENTS, 1, 5, 100, start="encode")
    assert after[1].parts[0] == ("enter-dit", 1)


def test_units_of_a_resumed_run_s_first_window_draws_the_whole_run() -> None:
    # Run 2's first window is done: its second is the first left, which draws the noise of all
    # 6 latents (shot.sample_windows draws it with the first window made).
    units = units_of(
        SHOTS, SEGMENTS, 1, 5, 100, encoded=[True] * 3, windows_done=[1, 1, 1], start="dit"
    )
    second = next(unit for unit in units if unit.key == ("window", 2, 1))
    assert second.parts == (
        ("enter-dit", 1),  # after the first segment's decode
        ("window", (LONG[1][1] - LONG[1][0]) * 100),
        ("draw", 600),
    )
    # Without a window done, the first window carries the draw and the second doesn't.
    whole = units_of(SHOTS, SEGMENTS, 1, 5, 100, encoded=[True] * 3)
    draws = [unit.key for unit in whole if any(kind == "draw" for kind, _ in unit.parts)]
    assert draws == [("window", 0, 0), ("window", 1, 0), ("window", 2, 0)]


def test_units_of_several_runs_a_shot() -> None:
    units = units_of(SHOTS, SEGMENTS, 4, 5, 100)
    # A shot of 3 frames has runs from its frames 0, 1 and 2; one of 1 frame has one; one of 21
    # has four (job.shot_runs).
    counts = [len(shot_runs(shot, 4)) for shot in SHOTS]
    assert counts == [3, 1, 4]
    encodes = [unit for unit in units if unit.phase == "encode"]
    assert [unit.key for unit in encodes] == [("encode", run) for run in range(8)]
    names = [
        f"{run_name(shot, 3, position, counts[shot], offset, 4)}'s encode"
        for shot, offset, position in [
            *((0, offset, offset) for offset in range(3)),
            (1, 0, 0),
            *((2, offset, offset) for offset in range(4)),
        ]
    ]
    assert [unit.name for unit in encodes] == names
    assert names[1] == "shot 1/3, run 2/3 (from its frame 1)'s encode"
    assert names[3] == "shot 2/3, run 1/1 (from its frame 0)'s encode"
    # Each follows a window or the start: no term, its change of phase being in its model.
    assert [unit.parts for unit in encodes] == [
        *((("encode", frames),) for frames in (3, 2, 1, 1, 21, 20, 19, 18))
    ]
    # Each run's first window draws its run's latent: 6 latents of 100 tokens for a 21-frame run.
    draws = [part for unit in units for part in unit.parts if part[0] == "draw"]
    assert draws == [("draw", n * 100) for n in (2, 2, 1, 1, 6, 6, 6, 6)]
    # Each segment's decode, one part to a run, in the order of the runs.
    decodes = [unit for unit in units if unit.phase == "decode"]
    assert [unit.parts for unit in decodes] == [
        (("enter-decode", 1), ("decode", 3), ("decode", 2), ("decode", 1), ("decode", 1)),
        (("enter-decode", 1), ("decode", 21), ("decode", 20), ("decode", 19), ("decode", 18)),
    ]
    # A segment's units come before the next one's, its decode last.
    assert [unit.key for unit in units].index(("decode", 0)) < [unit.key for unit in units].index(
        ("encode", 4)
    )
    assert units[-1].key == ("decode", 1)


def keys(units: list[Unit]) -> list[timing.Key]:
    return [unit.key for unit in units]


def test_units_of_skips_what_a_resume_has_made() -> None:
    everything = keys(units_of(SHOTS, SEGMENTS, 1, 5, 100))
    # A finished segment's units are all gone.
    first = keys(units_of(SHOTS, SEGMENTS, 1, 5, 100, finished=[True, False]))
    assert first == everything[5:]
    second = keys(units_of(SHOTS, SEGMENTS, 1, 5, 100, finished=[False, True]))
    assert second == everything[:5]
    assert units_of(SHOTS, SEGMENTS, 1, 5, 100, finished=[True, True]) == []
    # A run encoded keeps its windows; one with 1 window done keeps only its later one.
    encoded = keys(units_of(SHOTS, SEGMENTS, 1, 5, 100, encoded=[True, False, True]))
    assert encoded == [k for k in everything if k not in {("encode", 0), ("encode", 2)}]
    partly = keys(
        units_of(SHOTS, SEGMENTS, 1, 5, 100, encoded=[True, True, True], windows_done=[0, 0, 1])
    )
    assert partly == [
        *(("window", 0, 0), ("window", 1, 0), ("decode", 0)),
        *(("window", 2, 1), ("decode", 1)),
    ]
    # A run whose windows are all done keeps nothing, its segment's decode still to make.
    done = keys(units_of(SHOTS, SEGMENTS, 1, 5, 100, encoded=[True] * 3, windows_done=[1, 1, 2]))
    assert done == [("decode", 0), ("decode", 1)]
    # A finished segment whose runs' windows are done already takes nothing else with it.
    mixed = keys(
        units_of(SHOTS, SEGMENTS, 1, 5, 100, windows_done=[0, 0, 2], finished=[False, True])
    )
    assert mixed == everything[:5]


def test_job_times() -> None:
    clock = Clock(MODELS)
    units = units_of(SHOTS, SEGMENTS, 1, 5, 100)
    times = job_times(clock, units)
    assert times is not None
    # Runs of 3, 1 and 21 frames, padded to 5, 1 and 21.
    assert times.encode == pytest.approx((2 + 2.5) + (2 + 0.5) + (2 + 10.5))
    windows = sum(1.0 + 0.00025 * 100 * n for n in (2, 1, *(e - s for s, e in LONG)))
    assert times.windows == pytest.approx(windows)
    assert times.decode == pytest.approx((3 + 7.5) + (3 + 1.5) + (3 + 31.5))
    assert times.total == pytest.approx(times.encode + times.windows + times.decode)
    # The longest unit, the second segment's decode and write: how early --until may stop.
    assert times.longest == pytest.approx(34.5)
    assert times.unit == "segment 2/2's decode and write"


def test_job_times_counts_the_terms_in_each_unit_s_phase() -> None:
    units = units_of(SHOTS, SEGMENTS, 1, 5, 100)
    base = job_times(Clock(MODELS), units)
    times = job_times(Clock({**MODELS, **TERMS}), units)
    assert base is not None and times is not None
    # The encodes pay no term (their change is in their model); two decodes each enter their
    # phase, three of the four windows do, and the draws are those of 2, 1 and 6 latents of 100
    # tokens.
    assert times.encode == pytest.approx(base.encode)
    assert times.windows == pytest.approx(base.windows + 3 * 1.5 + 0.001 * 900)
    assert times.decode == pytest.approx(base.decode + 2 * 0.7)
    # The longest unit is still the second segment's decode, with its change of phase.
    assert times.longest == pytest.approx(34.5 + 0.7)
    assert times.unit == "segment 2/2's decode and write"
    # The encodes pay nothing whatever phase the models start in.
    for start in ("encode", "dit", "decode"):
        resumed = units_of(SHOTS, SEGMENTS, 1, 5, 100, start=start)
        after = job_times(Clock({**MODELS, **TERMS}), resumed)
        assert after is not None and after.encode == pytest.approx(base.encode)
    # A job resumed with the models in the DiT's phase, its encodes made, pays two changes into
    # it less: the first window's and the second's, which follow no encode.
    made = units_of(SHOTS, SEGMENTS, 1, 5, 100, start="dit", encoded=[True] * 3)
    held = job_times(Clock({**MODELS, **TERMS}), made)
    assert held is not None and held.encode == 0.0
    assert held.windows == pytest.approx(times.windows - 2 * 1.5)


def test_job_times_need_every_kind() -> None:
    units = units_of(SHOTS, SEGMENTS, 1, 5, 100)
    clock = Clock({"encode": MODELS["encode"], "window": MODELS["window"]})
    assert job_times(clock, units) is None
    clock.observe("decode", 13, 20.0)
    assert job_times(clock, units) is not None
    assert job_times(Clock(), []) == timing.Times(0.0, 0.0, 0.0, 0.0, "")


UNITS = [
    Unit(("encode", 0), "the encode", (("encode", 13),)),  # 8.5 s
    Unit(("window", 0, 0), "the window", (("window", 4_000),)),  # 2 s
    Unit(("decode", 0), "the decode", (("decode", 13), ("decode", 5))),  # 22.5 s + 10.5 s
]


def test_a_unit_s_cost_is_its_parts_left() -> None:
    progress = Progress(Clock(MODELS), UNITS, lambda: START)
    assert progress.cost(("encode", 0)) == 8.5
    assert progress.cost(("decode", 0)) == 33.0
    assert progress.cost(("encode", 7)) is None  # not a unit of the job
    progress.skipped(("decode", 0), "decode", 13)
    assert progress.cost(("decode", 0)) == 10.5
    # A part not in the unit leaves it as it is.
    progress.skipped(("decode", 0), "decode", 99)
    assert progress.cost(("decode", 0)) == 10.5
    progress.skipped(("decode", 0), "decode", 5)
    assert progress.cost(("decode", 0)) is None


def test_made_feeds_the_clock_and_skipped_does_not() -> None:
    clock = Clock(MODELS)
    progress = Progress(clock, UNITS, lambda: START)
    progress.made(("decode", 0), "decode", 13, 2 * 22.5)
    record = clock.record()["decode"]
    assert isinstance(record, dict) and record["units"] == 1
    # What is left of the unit is estimated by the clock as corrected (median of 1 and 2).
    assert progress.cost(("decode", 0)) == pytest.approx(1.5 * 10.5)
    progress.skipped(("decode", 0), "decode", 5)
    assert record["units"] == 1 and progress.cost(("decode", 0)) is None


def test_a_unit_s_terms_are_parts_like_the_rest() -> None:
    key = ("window", 0, 0)
    unit = Unit(key, "the window", (("enter-dit", 1), ("window", 4_000), ("draw", 4_000)))
    clock = Clock({**MODELS, **TERMS})
    progress = Progress(clock, [unit], lambda: START)
    assert progress.cost(key) == pytest.approx(1.5 + 2.0 + 4.0)
    # A term made is fed to the clock, its correction that of its own ratios.
    progress.made(key, "enter-dit", 1, 3.0)
    assert progress.cost(key) == pytest.approx(2.0 + 4.0)
    record = clock.record()["enter-dit"]
    assert isinstance(record, dict) and record["units"] == 1
    assert record["correction"] == pytest.approx(1.5)  # the median of 1 and 2
    progress.skipped(key, "draw", 4_000)
    assert progress.cost(key) == pytest.approx(2.0)


def test_left_sums_what_is_left_and_the_extra() -> None:
    extra: list[float | None] = [None]
    progress = Progress(Clock(MODELS), UNITS, lambda: START, lambda: extra[0])
    assert progress.left() is None  # nothing estimates the assembly yet
    extra[0] = 100.0
    assert progress.left() == 8.5 + 2.0 + 33.0 + 100.0
    progress.skipped(("encode", 0), "encode", 13)
    assert progress.left() == 2.0 + 33.0 + 100.0
    assert Progress(Clock(MODELS), UNITS, lambda: START).left() == 43.5
    assert Progress(Clock(), UNITS, lambda: START).left() is None
    assert Progress(Clock(MODELS), [], lambda: START).left() == 0.0


def test_said_waits_for_the_first_kind_missing() -> None:
    first = Progress(Clock(), UNITS, lambda: START)
    assert first.said() == "the job's time left known once an encode is timed"
    clock = Clock({"encode": MODELS["encode"]})
    assert Progress(clock, UNITS, lambda: START).said() == (
        "the job's time left known once a window is timed"
    )
    clock = Clock({"encode": MODELS["encode"], "window": MODELS["window"]})
    assert Progress(clock, UNITS, lambda: START).said() == (
        "the job's time left known once a decode is timed"
    )
    # Every kind known, the assembly not.
    waiting = Progress(Clock(MODELS), UNITS, lambda: START, lambda: None)
    assert waiting.said() == "the job's time left known once the assembly is timed"


def test_said_the_time_left_and_the_end() -> None:
    now = datetime(2026, 10, 21, 6, 59, 30, tzinfo=UTC)
    progress = Progress(Clock(MODELS), UNITS, lambda: now)
    # 8.5 + 2 + 33 s from 06:59:30 is 07:00:14 (43.5 rounds to 44): the minute after.
    assert progress.said() == "about 44 s left, the job ending about 07:01"
    progress.skipped(("encode", 0), "encode", 13)
    assert progress.said() == "about 35 s left, the job ending about 07:01"


def test_said_the_last_unit_made() -> None:
    progress = Progress(Clock(MODELS), UNITS[:1], lambda: START)
    progress.made(("encode", 0), "encode", 13, 8.5)
    assert progress.said() == "the job's last unit made"
    # With an assembly to come, the time left is its own.
    after = Progress(Clock(MODELS), UNITS[:1], lambda: START, lambda: 90.0)
    after.made(("encode", 0), "encode", 13, 8.5)
    assert after.said() == "about 1 min 30 s left, the job ending about 10:02"


def test_the_first_encode_is_the_first() -> None:
    assert Progress(Clock(), [], lambda: START).first_encode is True


@pytest.mark.parametrize(
    ("seconds", "said"),
    [
        (0.4, "0 s"),
        (42.4, "42 s"),
        (59.4, "59 s"),
        (59.6, "1 min 0 s"),
        (725, "12 min 5 s"),
        (3599.4, "59 min 59 s"),
        (3599.6, "1 h 0 min"),
        (3 * 3600 + 7 * 60 + 20, "3 h 7 min"),
        (86_399, "1 d 0 h"),
        (2 * 86_400 + 4 * 3600 + 100, "2 d 4 h"),
    ],
)
def test_spoken(seconds: float, said: str) -> None:
    assert spoken(seconds) == said


@pytest.mark.parametrize(
    ("seconds", "said"),
    [
        (0, "10:00"),
        (0.2, "10:01"),  # to the minute, rounded up
        (60, "10:01"),
        (61, "10:02"),
        (3600, "11:00"),
        (13 * 3600 + 30 * 60, "23:30"),
        (14 * 3600, "00:00 tomorrow"),
        (15 * 3600 + 5, "01:01 tomorrow"),
        (38 * 3600, "00:00 on Fri 23 Oct"),
        (2 * 86_400 + 600, "10:10 on Fri 23 Oct"),
    ],
)
def test_ending(seconds: float, said: str) -> None:
    assert ending(START, seconds) == said


def test_ending_in_the_local_zone() -> None:
    now = datetime(2026, 10, 21, 12, 0).astimezone()
    assert ending(now, 90 * 60) == "13:30"
    assert ending(now, 13 * 3600) == "01:00 tomorrow"


def test_would_end_when_it_fits() -> None:
    clock = Clock(MODELS)
    found = would_end(clock, UNITS, START, START + timedelta(seconds=50))
    assert found.before is None
    assert found.at == START + timedelta(seconds=43.5)
    assert (found.done, found.total) == (43.5, 43.5)
    with_last = would_end(
        clock, UNITS, START, START + timedelta(seconds=200), ("the assembly", 100.0)
    )
    assert with_last.before is None and with_last.at == START + timedelta(seconds=143.5)
    assert (with_last.done, with_last.total) == (143.5, 143.5)


@pytest.mark.parametrize(
    ("offset", "before", "done"),
    [
        # The estimate alone fits, 1.15 times it does not.
        (9.0, "the encode", 0.0),
        (10.0, "the window", 8.5),
        (20.0, "the decode", 10.5),
        (10.5 + 1.15 * 33.0 - 0.01, "the decode", 10.5),
    ],
)
def test_would_end_stops_before_the_first_unit_that_is_late(
    offset: float, before: str, done: float
) -> None:
    found = would_end(Clock(MODELS), UNITS, START, START + timedelta(seconds=offset))
    assert found.before == before
    assert found.done == done and found.total == 43.5
    assert found.at == START + timedelta(seconds=done)


def test_would_end_counts_the_assembly_after_the_units() -> None:
    deadline = START + timedelta(seconds=43.5 + 1.15 * 10.0 - 0.01)
    found = would_end(Clock(MODELS), UNITS, START, deadline, ("the assembly", 10.0))
    assert found.before == "the assembly"
    assert found.at == START + timedelta(seconds=43.5)
    assert (found.done, found.total) == (43.5, 53.5)


def test_would_end_needs_an_estimate_for_every_unit() -> None:
    with pytest.raises(ValueError, match="the encode: no estimate"):
        would_end(Clock(), UNITS, START, START + timedelta(hours=1))


def test_assembly_seconds() -> None:
    # The masters hashed, read, written and read back, the source read twice:
    # (4 x 1000 + 2 x 500) / 100.
    assert assembly_seconds(100.0, 1000, 500) == 50.0
    assert assembly_seconds(50.0, 0, 0) == 0.0


def test_disk_rate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The probe is a mebibyte under conftest; it leaves nothing behind.
    rate = disk_rate(tmp_path)
    assert isinstance(rate, float) and rate > 0
    assert list(tmp_path.iterdir()) == []
    # And it divides the bytes written by the seconds taken.
    ticks = iter([5.0, 7.0])
    monkeypatch.setattr(timing, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    assert disk_rate(tmp_path) == timing.PROBE / 2.0
    assert list(tmp_path.iterdir()) == []


def test_disk_rate_leaves_nothing_if_the_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(descriptor: int) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(timing.os, "fsync", broken)
    with pytest.raises(OSError, match="disk full"):
        disk_rate(tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("measures", "constant", "rate"),
    [
        ([(5, 3.5), (13, 7.5)], 1.0, 0.5),
        ([(5, 2.5), (13, 6.5)], 0.0, 0.5),
        # A line that would start under zero, or run flat or down: the two times' scatter, no
        # constant; the longest alone, by its rate.
        ([(5, 2.0), (13, 6.5)], 0.0, 0.5),
        ([(5, 7.0), (13, 6.5)], 0.0, 0.5),
        ([(5, 6.5), (13, 6.5)], 0.0, 0.5),
        ([(200, 1.2)], 0.0, 0.006),
    ],
)
def test_fit(measures: list[tuple[int, float]], constant: float, rate: float) -> None:
    model = calibration._fit(measures)  # pyright: ignore[reportPrivateUsage]
    assert (model.constant, model.rate) == (pytest.approx(constant), pytest.approx(rate))


TARGET = (48, 64)
PER_LATENT = tokens(TARGET, 1)
# What the fake models take to enter a phase the first time, the noise's draw a token.
ENTERING = {"encode": 0.4, "dit": 1.5, "decode": 0.7}
DRAWING = 0.0002


class Steady:
    """The calibration's steps, run.encode_shot, run.sample_windows and run.decode_shot, replaced
    by ones that move a clock of their own: an encode takes 1 s and 0.5 s a frame, a decode 3 s
    and 2 s a frame, a window 1 s and 0.1 ms a token plus the noise's draw it makes, which
    shot.draw takes 0.2 ms a token alone; the first of each kind 100 s more, as a process's first
    does. Models.enter takes a time of its own for each change of phase: ENTERING[phase] for
    the first change into it, and 0.5 s more for each one after. `called` lists the steps with
    their arguments, `entered` the phases the models were told."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.start = self.at = 1000.0
        self.first = {"encode", "window", "decode"}
        self.called: list[tuple[Any, ...]] = []
        self.entered: list[str] = []
        self.corrections: list[tuple[Correction, bool, list[int]]] = []
        self.models: Any = SimpleNamespace(device=torch.device("cpu"), enter=self.enter)
        self.phase: str | None = None
        self.changes = dict.fromkeys(ENTERING, 0)
        monkeypatch.setattr(calibration, "draw", self.draw)
        monkeypatch.setattr(run, "encode_shot", self.encode)
        monkeypatch.setattr(run, "sample_windows", self.windows)
        monkeypatch.setattr(run, "decode_shot", self.decode)
        monkeypatch.setattr(calibration, "time", SimpleNamespace(monotonic=lambda: self.at))

    def enter(self, phase: str) -> None:
        self.entered.append(phase)
        if phase != self.phase:
            self.at += ENTERING[phase] + 0.5 * self.changes[phase]
            self.changes[phase] += 1
        self.phase = phase

    def draw(self, models: Any, latent: torch.Tensor, seed: int, numz_draw: bool = False) -> Any:
        assert models is self.models and not numz_draw
        self.called.append(("draw", latent.shape[0]))
        self.at += DRAWING * latent.shape[0] * PER_LATENT
        return latent

    def encoding(self, count: int) -> float:
        """What an encode of `count` frames takes, in seconds, the first's extra apart."""
        return 1.0 + 0.5 * count

    def took(self, kind: str, seconds: float) -> None:
        self.at += seconds + (100.0 if kind in self.first else 0.0)
        self.first.discard(kind)

    def encode(
        self, models: Any, read: Any, count: int, target: tuple[int, int], seed: int, **o: Any
    ) -> torch.Tensor:
        assert models is self.models
        frames = read(count)
        assert frames.dtype == np.float32 and frames.shape[0] == count
        self.called.append(("encode", count, frames.shape[1:3], target, o))
        self.took("encode", self.encoding(count))
        return torch.zeros((count - 1) // 4 + 1, 2, 3, 16)

    def windows(
        self, models: Any, latent: torch.Tensor, layout: list[tuple[int, int]], seed: int, **o: Any
    ) -> Any:
        assert models is self.models
        self.called.append(("window", layout))
        tokens_made = (layout[0][1] - layout[0][0]) * PER_LATENT
        # The first window of a run draws its noise (shot.sample_windows).
        self.took("window", 1.0 + (0.0001 + DRAWING) * tokens_made)
        yield latent

    def decode(
        self,
        models: Any,
        latent: torch.Tensor,
        count: int,
        target: tuple[int, int],
        write: Any,
        name: str,
        first: int,
        correction: Correction | None,
        **o: Any,
    ) -> None:
        assert models is self.models and target == TARGET
        self.called.append(("decode", latent.shape[0], count, o))
        if correction is not None:
            # The shot's own copy and its checksums, as a run's decode finds them.
            sums = read_checksums(correction.checksums)
            self.corrections.append((correction, correction.copy.exists(), sums))
        write(np.zeros((count, *TARGET, 3), np.float32))
        self.took("decode", 3.0 + 2.0 * count)


class Quick(Steady):
    """A Steady whose encode is quick: 0.1 s and 0.03 s a frame, so 0.05 s a frame at 5 frames,
    where 3 s (calibration.SECONDS) are 60 frames, and whose change into the encode's phase
    costs nothing."""

    def encoding(self, count: int) -> float:
        return 0.1 + 0.03 * count

    def enter(self, phase: str) -> None:
        # The change into the encode's phase is part of an encode's time now (calibration.measure)
        # and costs nothing here: these tests are of the second length's rule, whose figures
        # (61, 29, 37) come of an encode's 0.25 s at 5 frames, not of the change.
        before = self.at
        super().enter(phase)
        if phase == "encode":
            self.at = before


def untiled(**more: Any) -> dict[str, Any]:
    return {"numz_padding": False, "numz_tail": False, "tiles": None, "scratch": None, **more}


def test_calibration(monkeypatch: pytest.MonkeyPatch) -> None:
    steady = Steady(monkeypatch)
    plan = Plan(9)
    found = calibrate(steady.models, TARGET, plan, 9)
    assert isinstance(found, Calibrated)
    assert (FIRST, FRAMES, SHORT) == (5, 13, 2)
    size = TARGET
    rest = {"allowance": plan.scratch_allowance}
    decoding = {"tiles": None, "scratch": None, **rest}
    assert steady.called == [
        # Each encode's two lengths right after a window (the warm-up's first, at the encode's
        # largest shape, unmeasured), as a run's encodes come; the second length: 0.88 s a frame
        # at 5 frames, change of phase included, so 3 s is about 3 frames, and never fewer
        # than 13.
        ("encode", 13, size, TARGET, untiled(**rest)),
        ("window", [(0, 2)]),
        ("encode", 5, size, TARGET, untiled(**rest)),
        ("window", [(0, 2)]),
        ("encode", 13, size, TARGET, untiled(**rest)),
        # Then the windows and the decodes, each after a step of its own kind: the longest's
        # warm-up, the draw alone, the two lengths.
        ("window", [(0, 9)]),
        # The noise's draw, alone, on the longest window's latent.
        ("draw", 9),
        ("window", [(0, 2)]),
        ("window", [(0, 9)]),
        # The warm-up's shot decoded twice: its 13 frames are the decode's largest.
        ("decode", 4, 13, decoding),
        ("decode", 2, 5, decoding),
        ("decode", 4, 13, decoding),
    ]
    assert steady.entered == [
        # Each encode right after a window, the first after nothing; the windows, the first of
        # them after an encode.
        *("encode", "dit", "encode", "dit", "encode", "dit", "dit", "dit"),
        *("decode", "decode", "decode"),
        # Then the changes made alone: the DiT's from the decode's, which no run makes, and the
        # decode's from the DiT's.
        *("dit", "decode"),
    ]
    # Each kind's constant and rate: the warm-ups' 100 s, on the calls at the largest sizes, and
    # the draw, which the window pays and the units count apart, left out.
    record = found.clock.record()
    # The encode's change of phase is in its time: Steady's change into the encode costs 0.4 s
    # and 0.5 s more for each one before it, so the encode of 5 frames, the second change, takes
    # 0.9 + 3.5 = 4.4 s and that of 13, the third, 1.4 + 7.5 = 8.9 s: the line through the two.
    rate = (8.9 - 4.4) / (13 - 5)
    assert record["encode"] == {
        "constant": pytest.approx(4.4 - 5 * rate),
        "rate": pytest.approx(rate),
        "correction": 1.0,
        "units": 0,
    }
    assert record["window"] == {
        "constant": pytest.approx(1.0),
        "rate": pytest.approx(0.0001),
        "correction": 1.0,
        "units": 0,
    }
    assert record["decode"] == {
        "constant": pytest.approx(3.0),
        "rate": pytest.approx(2.0),
        "correction": 1.0,
        "units": 0,
    }
    # The draw by the token of the latent it was timed on.
    assert record["draw"] == {
        "constant": 0.0,
        "rate": pytest.approx(DRAWING),
        "correction": 1.0,
        "units": 0,
    }
    # A phase's change, the median of those seen: the DiT's three made by the steps, each from the
    # encode's (1.5, 2.0, 2.5 s: 2.0), its fourth, from the decode's (3.0 s), made alone and not
    # counted; the decode twice, the warm-up's and the one made alone (0.7 and 1.2 s: 0.95). The
    # encode's three are in the encode's own model, and no term of the record.
    assert steady.changes == {"encode": 3, "dit": 4, "decode": 2}
    for phase, seconds in {"dit": 2.0, "decode": 0.95}.items():
        assert record[f"enter-{phase}"] == {
            "constant": pytest.approx(seconds),
            "rate": 0.0,
            "correction": 1.0,
            "units": 0,
        }
    assert "enter-encode" not in record
    assert found.clock.calibrated
    assert found.clock.estimate("encode", 101) == pytest.approx(4.4 - 5 * rate + rate * 101)
    assert found.clock.estimate("window", 30 * PER_LATENT) == pytest.approx(
        1.0 + 0.0001 * 30 * PER_LATENT
    )
    assert found.clock.estimate("draw", 1_000) == pytest.approx(1_000 * DRAWING)
    # The calibration's time is the clock's, warm-ups and phase changes and all.
    assert found.seconds == pytest.approx(steady.at - steady.start)
    assert found.seconds > 300


def test_an_encode_s_change_of_phase_is_in_its_own_time(monkeypatch: pytest.MonkeyPatch) -> None:
    # In the units: an encode opening a change of phase carries no term part, the window after it
    # carries the DiT's and a segment's decode the decode's.
    units = units_of(SHOTS, SEGMENTS, 1, 5, 100, start="decode")
    opening = [(unit.phase, unit.parts[0][0]) for unit in units if unit.phase != "window"]
    assert opening[0] == ("encode", "encode")
    assert {part[0] for unit in units if unit.phase == "encode" for part in unit.parts} == {
        "encode"
    }
    assert [unit.parts[0] for unit in units if unit.key == ("window", 0, 0)] == [("enter-dit", 1)]
    assert [unit.parts[0] for unit in units if unit.key == ("decode", 0)] == [("enter-decode", 1)]
    # In the calibration: the encode is measured from before its change into the encode's phase,
    # which Steady charges 0.4 s and 0.5 s more for each one before it, so that the encode of 5
    # frames (the second change, 0.9 s) takes 0.9 + 3.5 = 4.4 s and that of 13 frames (the third,
    # 1.4 s) 1.4 + 7.5 = 8.9 s.
    steady = Steady(monkeypatch)
    found = calibrate(steady.models, TARGET, Plan(9), 9)
    assert steady.changes["encode"] == 3
    five, thirteen = found.clock.estimate("encode", 5), found.clock.estimate("encode", 13)
    assert five is not None and thirteen is not None
    assert five == pytest.approx(4.4) and thirteen == pytest.approx(8.9)
    # The encode's own steps take 3.5 s and 7.5 s: the changes are the 0.9 s and 1.4 s more, and
    # no term of the clock holds them.
    assert five - steady.encoding(5) == pytest.approx(0.9)
    assert thirteen - steady.encoding(13) == pytest.approx(1.4)
    assert "enter-encode" not in found.clock.record()


def test_calibration_of_short_windows_only(monkeypatch: pytest.MonkeyPatch) -> None:
    # A job whose longest window is the short one: one length gives a rate alone.
    steady = Steady(monkeypatch)
    found = calibrate(steady.models, TARGET, Plan(9), 2)
    # The two windows between the encodes, the warm-up's and the one measured, the two lengths
    # being one.
    assert [call for call in steady.called if call[0] == "window"] == [("window", [(0, 2)])] * 4
    expected = 1.0 + 0.0001 * 2 * PER_LATENT
    assert found.clock.record()["window"] == {
        "constant": 0.0,
        "rate": pytest.approx(expected / (2 * PER_LATENT)),
        "correction": 1.0,
        "units": 0,
    }
    # And of single frames' latents: one latent, the windows between the encodes too.
    steady.called.clear()
    calibrate(steady.models, TARGET, Plan(9), 1)
    assert [call for call in steady.called if call[0] == "window"] == [("window", [(0, 1)])] * 4


def encode_counts(steady: Steady) -> list[int]:
    return [call[1] for call in steady.called if call[0] == "encode"]


def second_length_said(caplog: pytest.LogCaptureFixture, count: int) -> bool:
    return any(f"encodes of 5 and {count} frames" in line for line in caplog.messages)


def test_the_encode_s_second_length_by_time(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A quick encode (0.25 s for 5 frames, 0.05 s a frame) is measured again at about 3 s: 60
    # frames, 61 on the VAE's grid of 4n + 1, which the fit then takes with the 5.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    steady = Quick(monkeypatch)
    found = calibrate(steady.models, TARGET, Plan(9), 9)
    assert encode_counts(steady) == [13, 5, 61]
    assert found.clock.record()["encode"] == {
        "constant": pytest.approx(0.1),
        "rate": pytest.approx(0.03),
        "correction": 1.0,
        "units": 0,
    }
    assert second_length_said(caplog, 61)


def test_the_encode_s_second_length_within_the_job_s_longest_run(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The plan was made for the job's longest run: no encode longer, 30 frames giving 29; and
    # never fewer than 13, whatever a run of 10 frames would allow.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    steady = Quick(monkeypatch)
    calibrate(steady.models, TARGET, Plan(9), 9, most=30)
    assert encode_counts(steady) == [13, 5, 29]
    assert second_length_said(caplog, 29)
    steady.called.clear()
    calibrate(steady.models, TARGET, Plan(9), 9, most=10)
    assert encode_counts(steady) == [13, 5, 13]
    steady.called.clear()
    calibrate(steady.models, TARGET, Plan(9), 9, most=200)
    assert encode_counts(steady) == [13, 5, 61]


def test_the_encode_s_second_length_within_a_tiled_plan_s_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A tiled encode keeps its frames' scratch in system memory up to the plan's allowance
    # (tiling.Scratch): a second length over it is shortened by 4 frames until it fits.
    steady = Quick(monkeypatch)
    padded = padded_size(TARGET)
    holding = planner.scratch(padded, 37)
    assert planner.scratch(padded, 41) > holding > planner.scratch(padded, 33)
    plan = Plan(9, encode_tiles=Tiles(512), scratch_allowance=holding)
    calibrate(steady.models, TARGET, plan, 9)
    assert encode_counts(steady) == [13, 5, 37]
    # The allowance holding 61 frames' scratch, or more: the time's length stays.
    steady.called.clear()
    roomy = Plan(9, encode_tiles=Tiles(512), scratch_allowance=planner.scratch(padded, 61))
    calibrate(steady.models, TARGET, roomy, 9)
    assert encode_counts(steady) == [13, 5, 61]
    # Under 13 frames' scratch: never fewer than 13, the run's own refusal to come if it must.
    steady.called.clear()
    tight = Plan(9, encode_tiles=Tiles(512), scratch_allowance=planner.scratch(padded, 13) - 1)
    calibrate(steady.models, TARGET, tight, 9)
    assert encode_counts(steady) == [13, 5, 13]


@pytest.mark.parametrize(
    ("per_frame", "most", "fits", "count"),
    [
        # About 3 s at the rate, on the grid of 4n + 1: 19.5 frames, 27.3, 10, and no rate.
        (0.154, None, None, 21),
        (0.11, None, None, 29),
        (0.3, None, None, 13),
        (0.0, None, None, 13),
        # 300 frames at 0.01 s, then within the job's longest run, never fewer than 13.
        (0.01, None, None, 301),
        (0.01, 100, None, 97),
        (0.01, 1, None, 13),
        # Shrunk by 4 frames until it fits, 41 being the first over 40.
        (0.01, None, 40, 37),
        (0.01, None, 5, 13),
    ],
)
def test_second_length(per_frame: float, most: int | None, fits: int | None, count: int) -> None:
    def fitting(frames: int) -> bool:
        return fits is None or frames <= fits

    assert calibration._second_length(per_frame, most, fitting) == count  # pyright: ignore[reportPrivateUsage]


def test_calibration_reads_frames_of_the_source_s_size(monkeypatch: pytest.MonkeyPatch) -> None:
    steady = Steady(monkeypatch)
    calibrate(steady.models, TARGET, Plan(9), 9)
    assert {call[2] for call in steady.called if call[0] == "encode"} == {TARGET}
    steady.called.clear()
    calibrate(steady.models, TARGET, Plan(9), 9, source=(24, 32))
    assert [call[2] for call in steady.called if call[0] == "encode"] == [(24, 32)] * 3
    assert all(call[3] == TARGET for call in steady.called if call[0] == "encode")


def test_calibration_passes_a_tiled_plans_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The run's rule (run._scratch): a scratch directory for a tiled pass alone.
    steady = Steady(monkeypatch)
    plan = Plan(9, encode_tiles=Tiles(512), scratch_allowance=1 << 30)
    calibrate(steady.models, TARGET, plan, 9)
    encodes = [call[4] for call in steady.called if call[0] == "encode"]
    decodes = [call[3] for call in steady.called if call[0] == "decode"]
    assert all(o["tiles"] == Tiles(512) and isinstance(o["scratch"], Path) for o in encodes)
    assert all(o["tiles"] is None and o["scratch"] is None for o in decodes)
    assert all(o["allowance"] == 1 << 30 for o in [*encodes, *decodes])
    steady.called.clear()
    both = Plan(9, encode_tiles=Tiles(512), decode_tiles=Tiles(256))
    calibrate(steady.models, TARGET, both, 9)
    decodes = [call[3] for call in steady.called if call[0] == "decode"]
    assert all(o["tiles"] == Tiles(256) and isinstance(o["scratch"], Path) for o in decodes)
    # The tests' options are the run's.
    steady.called.clear()
    calibrate(steady.models, TARGET, Plan(9), 9, numz_padding=True, numz_tail=True)
    option = next(call[4] for call in steady.called if call[0] == "encode")
    assert option["numz_padding"] is True and option["numz_tail"] is True


class FakeCopy:
    """FFV1Writer for the input copy, ffmpeg left out: a checksum a frame, its file touched."""

    opened: ClassVar[list[tuple[Any, ...]]] = []

    def __init__(self, path: Path, *rest: Any) -> None:
        FakeCopy.opened.append((path.name, *rest))
        self.checksums: list[int] = []
        path.write_bytes(b"copy")

    def __enter__(self) -> "FakeCopy":
        return self

    def __exit__(self, *raised: object) -> None:
        pass

    def write(self, frames: Any) -> None:
        self.checksums.extend(range(len(self.checksums), len(self.checksums) + len(frames)))


class Output:
    """The job's writer, to which the calibration's decodes go."""

    def __init__(self) -> None:
        self.written: list[int] = []
        self.closed: list[bool] = []

    def __enter__(self) -> "Output":
        return self

    def __exit__(self, *raised: object) -> None:
        self.closed.append(raised[0] is None)

    def write(self, frames: Any) -> None:
        assert not self.closed
        self.written.append(len(frames))


def test_calibration_runs_the_job_s_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # As the run makes them: with split, each encode's frames copied and its decode corrected
    # against its own copy; the decodes' frames written to the job's kind of output, one for them
    # all; and nothing left behind.
    steady = Steady(monkeypatch)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(calibration, "FFV1Writer", FakeCopy)
    FakeCopy.opened.clear()
    output, opened = Output(), []

    def writer(path: Path) -> Any:
        opened.append(path)
        return output

    found = calibrate(steady.models, TARGET, Plan(9), 9, source=(24, 32), split=True, writer=writer)
    assert found.clock.estimate("decode", 13) == pytest.approx(3.0 + 2.0 * 13)
    # A copy an encode, three, at the source's size (width 32, height 24) and in 16 bits.
    assert [(name, fmt, width, height) for name, fmt, width, height, *_ in FakeCopy.opened] == [
        ("input.mkv", "gbrp16le", 32, 24)
    ] * 3
    # Each decode found its shot's copy, of the frames its encode read, and their checksums: three
    # decodes, the warm-up's shot twice (the second encode, of 13 frames, is shot 2's, not decoded).
    assert len(steady.corrections) == 3
    assert [
        (c.copy.name, c.checksums.name, c.width, c.height) for c, _, _ in steady.corrections
    ] == [("input.mkv", "input.crc32", 32, 24)] * 3
    assert [(exists, len(sums)) for _, exists, sums in steady.corrections] == [
        (True, 13),  # the warm-up's
        (True, 5),
        (True, 13),  # the warm-up's shot again
    ]
    assert [c.copy.parent.name for c, _, _ in steady.corrections] == [
        "shot_0",
        "shot_1",
        "shot_0",
    ]
    assert all(c.first == 0 for c, _, _ in steady.corrections)
    assert output.written == [13, 5, 13] and output.closed == [True]
    # In a directory of its own under the system's, gone with the calibration.
    assert len(opened) == 1 and opened[0].parent.parent == tmp_path
    assert list(tmp_path.iterdir()) == []


def test_calibration_without_split_copies_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    steady = Steady(monkeypatch)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(calibration, "FFV1Writer", FakeCopy)
    FakeCopy.opened.clear()
    calibrate(steady.models, TARGET, Plan(9), 9)
    assert FakeCopy.opened == [] and steady.corrections == []
    assert list(tmp_path.iterdir()) == []


def test_calibration_removes_its_directory_when_a_step_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    steady = Steady(monkeypatch)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    def broken(*args: Any, **options: Any) -> None:
        raise RuntimeError("out of memory")

    monkeypatch.setattr(run, "decode_shot", broken)
    with pytest.raises(RuntimeError, match="out of memory"):
        calibrate(steady.models, TARGET, Plan(9), 9)
    assert list(tmp_path.iterdir()) == []

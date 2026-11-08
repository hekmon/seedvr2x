"""What the nodes show while they run (comfyui/progress.py): the bar as the upscale's expected time,
spread again once a kind of unit is measured and never going back; the text of each form (a shot of
a clip, a clip of one shot, a picture of a batch; the time so far and the time left once a window
is measured; the plan and the result at the end); the previews, one a decode slice or a picture, at
ComfyUI's preview size; the download and the reading of the files on one bar; and the nodes' runs
over the stand-in models, tiled, equal bit for bit to the library's, with ComfyUI's Cancel honoured
after a tile.

The clock is a fake one the tests advance (Clock), the bar and the text box ComfyUI's stand-ins
(tests/comfy_stand_in), recording."""

import dataclasses
import threading
from typing import Any

import comfy.model_management as management
import numpy as np
import pytest
import server
from comfy.cli_args import args
from comfy.utils import ProgressBar
from comfyui import console, models, nodes, plans, progress
from standins import READ_PIECE, RESOLUTION, StandIn, handle, not_a_status, upscale_image, where
from standins import upscale_video as run_video
from test_nodes import TARGET, fresh, pictures
from test_still import TILES

from seedvr2x import clip, model, pull, still
from seedvr2x.clock import wall
from seedvr2x.passes import Tiles
from seedvr2x.shots import Shot


class Clock:
    """The progress's clock, in seconds, moved by the test."""

    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def shots_of(count: int, frames: int = 120) -> list[Shot]:
    return [Shot(index * frames, (index + 1) * frames) for index in range(count)]


def make(
    passes: list[progress.Pass],
    costs: progress.Costs = progress.CLIP,
) -> tuple[progress.Progress, Clock, list[str], ProgressBar]:
    clock, shown = Clock(), []
    bar = ProgressBar(progress.BAR)
    return progress.Progress(passes, costs, bar, shown.append, clock), clock, shown, bar


def clip_passes(
    shots: list[Shot],
    window: int | None = 5,
    encode: Tiles | None = None,
    decode: Tiles | None = None,
) -> list[progress.Pass]:
    return progress.clip_passes(shots, TARGET, window, encode, decode)


def play(
    p: progress.Progress,
    clock: Clock,
    index: int,
    *,
    encode: float,
    window: float,
    decode: float,
) -> None:
    """The events pass `index` tells the progress as the library tells them, each unit taking
    the given seconds (a decode's, shared by its slices of 4 frames)."""
    current = p.passes[index]
    p.begin(index)
    for kind, seconds, units in (
        ("encode", encode, current.encode),
        ("decode", decode, current.decode),
    ):
        if kind == "decode":
            for number in range(1, len(current.windows) + 1):
                clock.t += window
                p.window(number)
        if len(units) > 1:
            p.tile(kind, 0, len(units))  # type: ignore[arg-type]
            for done in range(1, len(units)):
                clock.t += seconds
                p.tile(kind, done, len(units))  # type: ignore[arg-type]
            p.tile(kind, len(units), len(units))  # type: ignore[arg-type]
        if kind == "encode":
            clock.t += seconds
            p.encoded()
            continue
        slices = -(-current.frames // 4)
        written = 0
        for _ in range(slices):
            clock.t += seconds / slices if len(units) == 1 else 0.01
            written = min(written + 4, current.frames)
            p.decoded(written, current.frames)


# ---- The bar ------------------------------------------------------------------------------------


def test_the_bar_is_monotone_ends_full_and_is_full_at_the_last_unit_only() -> None:
    p, clock, _, bar = make(clip_passes(shots_of(3, 33), 5, TILES, TILES))
    for index in range(3):
        play(p, clock, index, encode=2.0, window=3.0, decode=5.0)
    assert bar.updates == sorted(bar.updates) and bar.updates[0] == 0
    assert bar.updates[-1] == progress.BAR
    assert all(each < progress.BAR for each in bar.updates[:-1])
    assert set(bar.totals) == {progress.BAR}
    # Every pass moved it: its tiles, its windows, its slices.
    assert len(set(bar.updates)) > 3 * 8


def test_a_kinds_cost_is_the_runs_own_once_its_first_unit_is_measured() -> None:
    passes = clip_passes(shots_of(2, 33))
    costs = progress.Costs(encode=1e-6, window=1e-4, decode=2e-6)
    p, clock, _, bar = make(passes, costs)
    first = passes[0]
    for kind in ("encode", "window", "decode"):
        assert p.rate(kind) == getattr(costs, kind)  # type: ignore[arg-type]
    p.begin(0)
    # The encode takes 7 times what the card's figure says.
    seconds = 7 * costs.encode * first.encode[0].weight
    clock.t += seconds
    p.encoded()
    assert p.rate("encode") == pytest.approx(7 * costs.encode)
    assert p.rate("window") == costs.window, "a kind not measured yet keeps the card's figure"
    # The bar is the share of the expected time done: the measured encode over it and what is left,
    # the second shot's encode at the measured rate and the rest at the figures.
    others = sum(each.weight for each in first.windows) * costs.window
    others += first.decode[0].weight * costs.decode
    second = passes[1]
    left = others + 7 * costs.encode * second.encode[0].weight
    left += sum(each.weight for each in second.windows) * costs.window
    left += second.decode[0].weight * costs.decode
    assert bar.updates[-1] == int(progress.BAR * seconds / (seconds + left))


def test_the_bar_never_goes_back_when_a_kind_turns_out_slower_than_expected() -> None:
    passes = clip_passes(shots_of(3, 33))
    costs = progress.Costs(encode=1e-6, window=1e-4, decode=2e-6)
    p, clock, _, bar = make(passes, costs)
    play(p, clock, 0, encode=1.0, window=1.0, decode=1.0)
    before = bar.updates[-1]
    # The second shot's windows take 50 times the first's: the rest of the bar is spread again.
    p.begin(1)
    clock.t += 1.0
    p.encoded()
    for number in range(1, len(passes[1].windows) + 1):
        clock.t += 50.0
        p.window(number)
    assert bar.updates == sorted(bar.updates) and bar.updates[-1] >= before
    assert p.rate("window") > 10 * costs.window, "the windows' own rate, measured"
    p.decoded(33, 33)
    play(p, clock, 2, encode=1.0, window=1.0, decode=1.0)
    assert bar.updates == sorted(bar.updates) and bar.updates[-1] == progress.BAR


def test_an_untiled_decode_moves_the_bar_by_its_frames_written() -> None:
    p, clock, _, bar = make(clip_passes(shots_of(1, 33)))
    p.begin(0)
    clock.t += 1.0
    p.encoded()
    for number in range(1, len(p.passes[0].windows) + 1):
        clock.t += 1.0
        p.window(number)
    start = len(bar.updates)
    for written in (8, 16, 24):
        clock.t += 1.0
        p.decoded(written, 33)
    moved = bar.updates[start:]
    assert moved == sorted(moved) and len(set(moved)) == 3 and moved[-1] < progress.BAR
    p.decoded(33, 33)
    assert bar.updates[-1] == progress.BAR


# ---- The text -----------------------------------------------------------------------------------


def test_the_status_of_a_shot_of_a_clip() -> None:
    passes = clip_passes(shots_of(4), 5, TILES, None)
    p, _, shown, _ = make(passes)
    tiles, windows = len(passes[1].encode), len(passes[1].windows)
    assert tiles == 4
    p.begin(1)
    assert shown[-1] == "shot 2/4 \u00b7 encoding\n0 s elapsed \u00b7 estimating time left"
    p.tile("encode", 0, tiles)
    p.tile("encode", 2, tiles)
    assert where(shown[-1]) == f"shot 2/4 \u00b7 tile 3/{tiles} \u00b7 encoding"
    p.tile("encode", tiles, tiles)
    assert where(shown[-1]) == f"shot 2/4 \u00b7 tile {tiles}/{tiles} \u00b7 encoding"
    p.encoded()
    assert where(shown[-1]) == f"shot 2/4 \u00b7 window 1/{windows} \u00b7 upscaling"
    p.window(1)
    assert where(shown[-1]) == f"shot 2/4 \u00b7 window 2/{windows} \u00b7 upscaling"
    for number in range(2, windows + 1):
        p.window(number)
    assert where(shown[-1]) == "shot 2/4 \u00b7 frames 0/120 \u00b7 decoding"
    p.decoded(60, 120)
    assert where(shown[-1]) == "shot 2/4 \u00b7 frames 60/120 \u00b7 decoding"
    # The box is replaced, not appended to: every message is two lines, with no name.
    assert all(each.count("\n") == 1 and "seedvr2x" not in each for each in shown)


def test_the_status_of_a_tiled_decode_and_of_a_clip_of_one_shot() -> None:
    passes = clip_passes([Shot(0, 480)], 5, None, TILES)
    p, _, shown, _ = make(passes)
    windows, tiles = len(passes[0].windows), len(passes[0].decode)
    p.begin(0)
    p.encoded()
    p.window(4)
    # A clip of one shot has no shot part.
    assert where(shown[-1]) == f"window 5/{windows} \u00b7 upscaling"
    for number in range(5, windows + 1):
        p.window(number)
    assert where(shown[-1]) == f"tile 1/{tiles} \u00b7 decoding"
    p.tile("decode", 1, tiles)
    assert where(shown[-1]) == f"tile 2/{tiles} \u00b7 decoding"


def test_the_status_of_a_picture_of_a_batch_and_of_one_picture() -> None:
    passes = progress.picture_passes(8, TARGET, TILES, TILES)
    p, _, shown, _ = make(passes, progress.PICTURE)
    p.begin(2)
    assert where(shown[-1]) == "picture 3/8 \u00b7 encoding"
    p.tile("decode", 1, 4)
    assert where(shown[-1]) == "picture 3/8 \u00b7 tile 2/4 \u00b7 decoding"
    p.encoded()
    assert where(shown[-1]) == "picture 3/8 \u00b7 upscaling"
    alone, _, shown, _ = make(progress.picture_passes(1, TARGET, None, None), progress.PICTURE)
    alone.begin(0)
    alone.encoded()
    alone.window(1)
    assert where(shown[-1]) == "decoding", "one picture has no picture part"


def test_the_time_part_says_the_time_so_far_then_the_time_left_once_a_window_is_measured() -> None:
    p, clock, shown, _ = make(clip_passes(shots_of(2, 33)))
    p.begin(0)
    clock.t += 72.0
    p.encoded()
    second = shown[-1].split("\n")[1]
    assert second == "1 min elapsed \u00b7 estimating time left", "no window measured yet"
    clock.t += 10.0
    p.window(1)
    left = shown[-1].split("\n")[1]
    assert left == progress.time_left(wall(p.left())) and left.startswith("~")
    assert left.endswith(" left")
    # The time left is the units left at the rates: the window's own now, the others the card's.
    assert p.left() == pytest.approx(
        sum(u.weight for u in p.passes[0].windows[1:]) * p.rate("window")
        + sum(u.weight for u in p.passes[1].windows) * p.rate("window")
        + sum(p.rate(u.kind) * u.weight for u in p.passes[0].decode + p.passes[1].decode)
        + p.rate("encode") * p.passes[1].encode[0].weight
    )


@pytest.mark.parametrize(
    ("seconds", "said"),
    [
        (0.2, "~1 s left"),
        (42, "~42 s left"),
        (59.4, "~59 s left"),
        (60, "~1 min left"),
        (11.6 * 60, "~12 min left"),
        (3599, "~59 min left"),
        (3600, "~1 h 0 min left"),
        (4020, "~1 h 7 min left"),
        (30 * 3600, "~1 d 6 h left"),
    ],
)
def test_the_time_left_is_rough_seconds_then_whole_minutes_then_two_units(
    seconds: float, said: str
) -> None:
    assert progress.time_left(seconds) == said


def test_the_end_says_what_was_made_in_how_long_and_no_plan() -> None:
    p, clock, shown, bar = make(clip_passes(shots_of(1, 33)))
    p.begin(0)
    clock.t += 272.0
    p.finish("480 frames")
    assert shown[-1] == "480 frames in 4 min 32 s"
    assert bar.updates[-1] == progress.BAR
    p.finish("1 frame")
    assert shown[-1] == "1 frame in 4 min 32 s"


# ---- The runs of a shot (quality) ---------------------------------------------------------------


def test_every_run_of_a_shot_is_a_pass_of_its_own_with_its_own_units() -> None:
    passes = progress.clip_passes(shots_of(2, 33), TARGET, 5, TILES, TILES, runs=4)
    assert len(passes) == 8
    first = [each for each in passes if each.shot == 1]
    # A run is a pass over the shot from its first frame: the frames it has, its windows and its
    # padded length follow (the VAE takes 4n + 1).
    assert [each.frames for each in first] == [33, 32, 31, 30]
    assert [each.run for each in first] == [1, 2, 3, 4]
    assert [len(each.windows) for each in first] == [3, 3, 3, 3]
    assert [len(each.encode) for each in first] == [4, 4, 4, 4]
    standard = progress.clip_passes(shots_of(2, 33), TARGET, 5, TILES, TILES)
    assert len(standard) == 2 and [each.frames for each in standard] == [33, 33]
    # A shot too short for a run has no such run, as the library's.
    short = progress.clip_passes([Shot(0, 3)], TARGET, 5, None, None, runs=4)
    assert [each.frames for each in short] == [3, 2, 1]
    one = progress.clip_passes([Shot(0, 1)], TARGET, 5, None, None, runs=4)
    assert [each.label for each in one] == ["Frames 1 to 1"]


def test_the_status_of_a_run_of_a_shot_and_of_a_clip_of_one_shot() -> None:
    passes = progress.clip_passes(shots_of(4), TARGET, 5, None, None, runs=4)
    p, _, shown, _ = make(passes)
    windows = len(passes[1 * 4 + 2].windows)  # shot 2's third run: 120 - 2 frames
    p.run(1, 3)
    assert where(shown[-1]) == "shot 2/4 \u00b7 run 3/4 \u00b7 encoding"
    p.encoded()
    p.window(1)
    # The first line is never trimmed: the shot, the run, the unit and the action.
    assert where(shown[-1]) == f"shot 2/4 \u00b7 run 3/4 \u00b7 window 2/{windows} \u00b7 upscaling"
    alone = progress.clip_passes([Shot(0, 480)], TARGET, 5, None, None, runs=2)
    p, _, shown, _ = make(alone)
    p.run(0, 2)
    p.encoded()
    p.window(4)
    count = len(alone[1].windows)
    assert where(shown[-1]) == f"run 2/2 \u00b7 window 5/{count} \u00b7 upscaling"
    # Standard has no run part, and a clip of one shot no shot part.
    p, _, shown, _ = make(progress.clip_passes([Shot(0, 480)], TARGET, 5, None, None))
    p.run(0, 1)
    assert where(shown[-1]) == "encoding"
    with pytest.raises(ValueError, match="no run 3"):
        p.run(0, 3)


def test_the_bar_counts_every_runs_units_and_ends_full_at_the_last_run() -> None:
    passes = progress.clip_passes(shots_of(2, 33), TARGET, 5, TILES, TILES, runs=2)
    p, clock, _, bar = make(passes)
    halfway = 0
    for index in range(len(passes)):
        play(p, clock, index, encode=2.0, window=3.0, decode=5.0)
        if index == 1:
            halfway = bar.updates[-1]
    assert bar.updates == sorted(bar.updates) and bar.updates[-1] == progress.BAR
    assert all(each < progress.BAR for each in bar.updates[:-1])
    # Four runs of the same units: the first shot's two end at half the bar, give or take the
    # rates re-measured as the first run goes.
    assert abs(halfway - progress.BAR // 2) <= progress.BAR // 10


def test_a_later_run_names_its_own_frames_and_counts_its_decode_in_them() -> None:
    passes = progress.clip_passes(shots_of(4), TARGET, 5, None, None, runs=4)
    p, _, shown, _ = make(passes)
    # Run 3 of shot 2 starts at the shot's third frame (clip frame 123, counted from 1) and has
    # 118 of its 120 frames.
    assert passes[1 * 4 + 2].label == "Shot 2 of 4, run 3 of 4 (frames 123 to 240)"
    assert passes[1 * 4 + 2].frames == 118
    assert [each.label for each in passes[4:8]] == [
        "Shot 2 of 4, run 1 of 4 (frames 121 to 240)",
        "Shot 2 of 4, run 2 of 4 (frames 122 to 240)",
        "Shot 2 of 4, run 3 of 4 (frames 123 to 240)",
        "Shot 2 of 4, run 4 of 4 (frames 124 to 240)",
    ]
    p.run(1, 3)
    p.encoded()
    for number in range(1, len(passes[6].windows) + 1):
        p.window(number)
    assert where(shown[-1]) == "shot 2/4 \u00b7 run 3/4 \u00b7 frames 0/118 \u00b7 decoding"
    p.decoded(60, 118)
    assert where(shown[-1]) == "shot 2/4 \u00b7 run 3/4 \u00b7 frames 60/118 \u00b7 decoding"
    p.decoded(118, 118)
    assert where(shown[-1]) == "shot 2/4 \u00b7 run 3/4 \u00b7 frames 118/118 \u00b7 decoding"
    # A clip of one shot, the same: its second run begins at frame 3.
    alone = progress.clip_passes([Shot(0, 480)], TARGET, 5, None, None, runs=2)
    assert [each.label for each in alone] == [
        "Frames 1 to 480, run 1 of 2",
        "Frames 3 to 480, run 2 of 2",
    ]
    # A clip that is a shot cut from a longer film starts where the shot does.
    inside = progress.clip_passes([Shot(100, 130)], TARGET, 5, None, None, runs=2)
    assert [each.label for each in inside] == [
        "Frames 101 to 130, run 1 of 2",
        "Frames 103 to 130, run 2 of 2",
    ]


def test_a_run_at_maximum_counts_its_decode_in_its_own_frames_on_the_node(
    machine: StandIn, tmp_path
) -> None:
    run_video(
        handle(tmp_path), plans.Settings(window=5), pictures(33), RESOLUTION, "7", quality="maximum"
    )
    said = [where(each) for each in node_texts("7") if progress.SEPARATOR in each]
    # Run 3 of the clip of one shot begins at its third frame and has 31 frames.
    assert "run 3/4 \u00b7 encoding" in said
    assert "run 3/4 \u00b7 frames 31/31 \u00b7 decoding" in said
    assert "run 2/4 \u00b7 frames 32/32 \u00b7 decoding" in said
    assert not any("run 3/4" in each and "/33" in each for each in said)


# ---- The time between events --------------------------------------------------------------------

THREAD = "seedvr2x-progress-text"


def ticker_threads() -> list[threading.Thread]:
    return [each for each in threading.enumerate() if each.name == THREAD]


class Cue:
    """The ticker's pause, driven by the test: the thread waits at `pause` until `fire` lets one
    tick through, and `fire` returns once that tick is made and the thread is back at the pause
    (or has ended), so that a tick is awaited by what it does, not by a sleep."""

    def __init__(self) -> None:
        self.go = threading.Semaphore(0)
        self.back = threading.Semaphore(0)
        self.pauses = 0

    def pause(self, stopped: threading.Event, period: float) -> None:
        self.pauses += 1
        self.back.release()
        while not stopped.is_set():
            if self.go.acquire(timeout=0.02):
                return

    def settle(self) -> None:
        assert self.back.acquire(timeout=5), "the ticker never reached its pause"

    def fire(self) -> None:
        self.go.release()
        self.settle()


def test_the_time_is_said_between_events_by_a_thread_that_leaves_the_bar_alone() -> None:
    p, clock, shown, bar = make(clip_passes(shots_of(2, 33)))
    cue = Cue()
    with p.running(pause=cue.pause):
        cue.settle()
        assert [each.name for each in ticker_threads()] == [THREAD]
        assert ticker_threads()[0].daemon
        # Nothing is said before the first event.
        cue.fire()
        assert shown == []
        p.begin(0)
        updates, count = list(bar.updates), len(shown)
        text = shown[-1]
        assert text == "shot 1/2 \u00b7 encoding\n0 s elapsed \u00b7 estimating time left"
        # The same first line as the event's, the time moved on, once a second.
        clock.t += 5.0
        cue.fire()
        assert len(shown) == count + 1
        assert where(shown[-1]) == where(text)
        assert shown[-1].endswith("\n5 s elapsed \u00b7 estimating time left")
        clock.t += 1.0
        cue.fire()
        assert shown[-1].endswith("\n6 s elapsed \u00b7 estimating time left")
        assert bar.updates == updates, "a tick never moves the bar"
        # The next event says the time as well, from the same clock.
        clock.t += 2.0
        p.encoded()
        assert shown[-1].endswith("\n8 s elapsed \u00b7 estimating time left")
    assert ticker_threads() == []


def test_the_time_left_is_in_the_tick_once_a_window_is_measured() -> None:
    p, clock, shown, _ = make(clip_passes(shots_of(2, 33)))
    cue = Cue()
    with p.running(pause=cue.pause):
        cue.settle()
        p.begin(0)
        clock.t += 10.0
        p.encoded()
        clock.t += 20.0
        p.window(1)
        clock.t += 3.0
        cue.fire()
        assert shown[-1].startswith("shot 1/2 \u00b7 window 2/")
        assert shown[-1].endswith("\n" + progress.time_left(wall(p.left())))


def test_no_tick_after_the_end_and_no_thread_alive() -> None:
    p, clock, shown, bar = make(clip_passes(shots_of(1, 33)))
    cue = Cue()
    with p.running(pause=cue.pause):
        cue.settle()
        p.begin(0)
        clock.t += 4.0
        p.finish("33 frames")
        summary = shown[-1]
        clock.t += 9.0
        cue.fire()
        assert shown[-1] == summary == "33 frames in 4 s"
    assert ticker_threads() == [] and shown[-1] == summary
    assert bar.updates[-1] == progress.BAR


def test_the_thread_ends_on_an_error_and_on_cancel() -> None:
    for error in (RuntimeError("boom"), management.InterruptProcessingException()):
        p, _, shown, _ = make(clip_passes(shots_of(1, 33)))
        cue = Cue()
        with pytest.raises(type(error)):
            with p.running(pause=cue.pause):
                cue.settle()
                p.begin(0)
                raise error
        assert ticker_threads() == []
        count = len(shown)
        cue.go.release()  # a tick after the stop would be said now
        assert len(shown) == count


def test_a_text_that_cannot_be_sent_ends_the_ticker_not_the_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock, calls = Clock(), []
    bar = ProgressBar(progress.BAR)

    def show(text: str) -> None:
        calls.append(text)
        if threading.current_thread().name == THREAD:
            raise OSError("the server went away")

    p = progress.Progress(clip_passes(shots_of(1, 33)), progress.CLIP, bar, show, clock)
    cue = Cue()
    with p.running(pause=cue.pause):
        cue.settle()
        p.begin(0)
        thread = ticker_threads()[0]
        cue.go.release()
        thread.join(5)
        assert not thread.is_alive() and len(calls) == 2
        # The ticker has ended; the node's thread goes on with its own events.
        p.encoded()
        assert len(calls) == 3
    assert "can't be said between events" in caplog.text
    assert ticker_threads() == []


def test_no_tick_is_said_once_the_block_has_ended_even_one_on_its_way() -> None:
    p, _, shown, _ = make(clip_passes(shots_of(1, 33)))
    cue = Cue()
    with pytest.raises(RuntimeError):
        with p.running(pause=cue.pause):
            cue.settle()
            p.begin(0)
            raise RuntimeError("boom")
    count = len(shown)
    p.tick()  # a tick that was on its way when the error came
    assert len(shown) == count


def test_the_join_returns_at_the_stop_with_no_time_cap_and_the_thread_is_gone() -> None:
    """The default pause on a real event, a short period: the stop sets the event and joins with
    no timeout, and the thread is not alive right after stop returns, however long the period."""
    assert not hasattr(progress.Ticker, "JOIN")
    p, _, _, _ = make(clip_passes(shots_of(1, 33)))
    for period in (0.01, 3600.0):
        ticker = progress.Ticker(p, period)
        ticker.start()
        ticker.stop()
        assert not ticker.thread.is_alive()
        ticker.stop()  # idempotent
    assert ticker_threads() == []


def test_the_default_pause_waits_a_second_and_a_stop_does_not_wait_for_it() -> None:
    assert progress.TICK == 1.0
    p, _, _, _ = make(clip_passes(shots_of(1, 33)))
    with p.running():
        assert len(ticker_threads()) == 1
    assert ticker_threads() == []


@pytest.mark.parametrize("how", ["end", "cancel", "error"])
def test_a_node_leaves_no_thread_running(
    machine: StandIn, tmp_path, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    h = handle(tmp_path)
    real = model.sample
    if how == "cancel":
        stop_at(3)
    elif how == "error":
        monkeypatch.setattr(model, "sample", lambda *args: (_ for _ in ()).throw(KeyError("x")))
    expected: Any = {
        "end": None,
        "cancel": management.InterruptProcessingException,
        "error": KeyError,
    }[how]
    if expected is None:
        run_video(h, plans.Settings(window=5), pictures(9), RESOLUTION, "6")
    else:
        with pytest.raises(expected):
            run_video(h, plans.Settings(window=5), pictures(9), RESOLUTION, "6")
    assert ticker_threads() == []
    # The next run goes through and leaves none either.
    ProgressBar.on_update = None
    monkeypatch.setattr(model, "sample", real)
    upscale_image(h, plans.Settings(), pictures(1), RESOLUTION, "6")
    assert ticker_threads() == []


# ---- The previews -------------------------------------------------------------------------------


def test_a_preview_is_the_newest_frame_at_comfyuis_preview_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(args, "preview_size", 16)
    p, _, _, bar = make(clip_passes(shots_of(1, 33)))
    p.begin(0)
    p.encoded()
    for number in range(1, len(p.passes[0].windows) + 1):
        p.window(number)
    assert bar.previews == [], "none before the decode"
    for written, level in ((8, 0.2), (16, 0.4), (33, 0.6)):
        p.decoded(written, 33, np.full((48, 80, 3), level, np.float32))
    assert len(bar.previews) == 3
    for (_, preview), level in zip(bar.previews, (0.2, 0.4, 0.6), strict=True):
        kind, image, side = preview  # type: ignore[misc]
        # The format ComfyUI's samplers send, shrunk to the preview size and not past it, the
        # size limit kept.
        assert (kind, side) == ("JPEG", 16) and image.size == (16, 10)
        assert image.getpixel((8, 5)) == (round(level * 255),) * 3


def test_a_frame_within_the_preview_size_is_sent_at_its_own_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ComfyUI's server fits a preview to the size with ImageOps.contain, which enlarges a small
    # one: the node sends it as it is, and says the size it is to be fitted to.
    monkeypatch.setattr(args, "preview_size", 512)
    kind, image, side = progress.preview_of(np.zeros((24, 40, 3), np.float32))
    assert image.size == (40, 24) and side == 512 and kind == "JPEG"


# ---- Before the upscale -------------------------------------------------------------------------


def preparing() -> tuple[progress.Preparing, Clock, list[str]]:
    clock, shown = Clock(), []
    return progress.Preparing(shown.append, plans.file_label, clock), clock, shown


def test_the_download_and_the_reading_share_one_bar_of_one_total_with_their_time_under_them(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO", "seedvr2x_comfyui")
    p, clock, shown = preparing()
    dit, vae = plans.DEFAULT_DIT, plans.VAE_FILE
    # The reading's bytes are announced before the first download, and every download's size is
    # told before its first byte (pull.fetch): the bar is made at the first byte, its total whole.
    p.will_read(17_000_000_000)
    p.download(dit, 0, 16_500_000_000)
    p.download(vae, 0, 500_000_000)
    # All the downloads' bytes together, in the total's unit, over the time: no rate yet.
    estimating = "0 s elapsed \u00b7 estimating time left"
    assert ProgressBar.created == [] and shown[-1] == f"downloading 0.0/17.0 GB\n{estimating}"
    clock.t += 30.0
    # The first bytes give no rate: the time before them is the connection's.
    p.download(dit, 4_200_000_000, 16_500_000_000)
    assert shown[-1] == "downloading 4.2/17.0 GB\n30 s elapsed \u00b7 estimating time left"
    clock.t += 30.0
    p.download(dit, 8_400_000_000, 16_500_000_000)
    # 4.2 GB in 30 s since the first bytes: the 8.6 GB left at that rate, 61 s.
    assert shown[-1] == "downloading 8.4/17.0 GB\n~1 min left"
    assert progress.time_left(8.6e9 * 30 / 4.2e9) == "~1 min left"
    p.download(dit, 16_500_000_000, 16_500_000_000)
    p.download(vae, 500_000_000, 500_000_000)
    # The reading's rate is its own, from its first bytes.
    p.read(vae, 100_000_000, 500_000_000)
    assert shown[-1] == "reading 0.1/17.0 GB\n1 min elapsed \u00b7 estimating time left", (
        "of the bytes the load announced"
    )
    clock.t += 10.0
    p.read(vae, 500_000_000, 500_000_000)
    p.read(dit, 100_000_000, 16_500_000_000)
    clock.t += 10.0
    p.read(dit, 9_000_000_000, 16_500_000_000)
    # 0.4 GB of the VAE and 8.9 of the DiT in 20 s since the first bytes, each file's first
    # left out: the 7.5 GB left in 16 s.
    assert shown[-1] == "reading 9.5/17.0 GB\n~16 s left"
    p.read(dit, 16_500_000_000, 16_500_000_000)
    assert all(not not_a_status([each]) and each.count("\n") == 1 for each in shown)
    # The files are named in the console, once each when its first byte is told.
    assert [each.removeprefix(console.PREFIX) for each in caplog.messages] == [
        "downloading the sharp 7B (float16)",
        "downloading the VAE",
        "reading the VAE into memory",
        "reading the sharp 7B (float16) into memory",
    ]
    (bar,) = ProgressBar.created
    assert bar.updates == sorted(bar.updates) and bar.updates[0] == 4_200_000_000
    # One total throughout, the downloads' and the reading's: the fraction never falls, and the
    # bar ends full.
    assert set(bar.totals) == {17_000_000_000 + 17_000_000_000}
    assert bar.updates[-1] == bar.totals[-1]


def test_a_small_file_is_said_in_megabytes() -> None:
    p, _, shown = preparing()
    p.download("transnetv2.safetensors", 12_100_000, 30_500_000)
    assert shown[-1] == "downloading 12.1/30.5 MB\n0 s elapsed \u00b7 estimating time left"


def test_nothing_to_download_or_read_makes_no_bar() -> None:
    preparing()
    assert ProgressBar.created == []


def test_cancel_stops_a_reading_and_a_download_reported_on_the_nodes_thread() -> None:
    p, _, _ = preparing()
    management.interrupt_current_processing()
    with pytest.raises(management.InterruptProcessingException):
        p.download(plans.DEFAULT_DIT, 1, 10)
    management.interrupt_current_processing()
    with pytest.raises(management.InterruptProcessingException):
        p.read(plans.DEFAULT_DIT, 1, 10)


def report_elsewhere(p: progress.Preparing, done: int = 1) -> list[BaseException]:
    errors: list[BaseException] = []

    def report() -> None:
        try:
            p.download(plans.DEFAULT_DIT, done, 10)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=report)
    thread.start()
    thread.join()
    return errors


def test_a_download_reported_on_another_thread_leaves_a_pressed_cancel_to_the_nodes_thread() -> (
    None
):
    # huggingface_hub's Xet downloads report from a thread of hf_xet's own, which catches what a
    # report raises. ComfyUI's hook would clear the flag and raise there, and the node's thread
    # would never see Cancel: the report holds the flag's lock, touches neither the bar nor the
    # text while Cancel is pressed, and leaves the flag.
    p, _, shown = preparing()
    p.download(plans.DEFAULT_DIT, 0, 10)
    assert report_elsewhere(p, 1) == [] and len(ProgressBar.created) == 1
    (bar,) = ProgressBar.created
    assert bar.updates == [1] and len(shown) == 2
    management.interrupt_current_processing()
    assert report_elsewhere(p, 2) == []
    assert bar.updates == [1] and len(shown) == 2, "nothing sent while Cancel is pressed"
    assert management.processing_interrupted(), "the flag is left for the nodes' thread"
    # The node's thread then finds it: a report on it, or the check when the download returns.
    with pytest.raises(management.InterruptProcessingException):
        management.throw_exception_if_processing_interrupted()


def test_a_cancel_pressed_during_a_download_is_honoured_when_it_returns(
    machine: StandIn, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from seedvr2x import pull

    real = pull.resolve

    def resolve(*arguments: Any, **options: Any) -> Any:
        # The downloader's thread reports while Cancel is pressed, then the download returns.
        management.interrupt_current_processing()
        assert report_elsewhere_to(options["progress"]) == []
        return real(*arguments, **options)

    monkeypatch.setattr(pull, "resolve", resolve)
    with pytest.raises(management.InterruptProcessingException):
        upscale_image(handle(tmp_path), plans.Settings(), pictures(1), RESOLUTION, "9")
    assert machine.loads == [], "nothing was loaded after Cancel"


def report_elsewhere_to(progress_: Any) -> list[BaseException]:
    errors: list[BaseException] = []

    def report() -> None:
        try:
            progress_("w.safetensors", 5, 10)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=report)
    thread.start()
    thread.join()
    return errors


def test_the_scan_of_the_cuts_node_moves_its_bar_by_frames_and_honours_cancel() -> None:
    shown: list[str] = []
    bar = ProgressBar(450)
    tell = progress.scanned(bar, shown.append)
    tell(200, 450)
    assert bar.updates == [200] and shown[-1] == "scanning 200/450 frames"
    management.interrupt_current_processing()
    with pytest.raises(management.InterruptProcessingException):
        tell(400, 450)


# ---- The nodes' runs ----------------------------------------------------------------------------


def tiled_plans(monkeypatch: pytest.MonkeyPatch) -> None:
    """The nodes' planners giving the stand-in size's tiles (the planner takes none that small):
    4 tiles to each VAE pass, as the library's own tests of the tiles use."""
    clip_planner, picture_planner = nodes.clip_planner, nodes.picture_planner

    def clips(*arguments: Any) -> Any:
        planner = clip_planner(*arguments)

        def tiled(*rest: Any) -> Any:
            planned = planner(*rest)
            made, expected = planned.plan
            both = dataclasses.replace(made, encode_tiles=TILES, decode_tiles=TILES)
            return dataclasses.replace(planned, plan=(both, expected))

        return tiled

    def pictures_(*arguments: Any) -> Any:
        planner = picture_planner(*arguments)

        def tiled(*rest: Any) -> Any:
            planned = planner(*rest)
            made, tiles = planned.plan
            both = dataclasses.replace(tiles, encode=TILES, decode=TILES)
            return dataclasses.replace(planned, plan=(made, both))

        return tiled

    monkeypatch.setattr(nodes, "clip_planner", clips)
    monkeypatch.setattr(nodes, "picture_planner", pictures_)


def node_texts(node: str) -> list[str]:
    instance = server.PromptServer.instance
    assert instance is not None
    return [text for text, each in instance.texts if each == node]


def status(texts: list[str]) -> list[str]:
    """The status lines of a run's texts without their time: the node's texts less the first
    ("planning\u2026") and the last (what was made, and in how long)."""
    return [where(each) for each in texts[1:-1]]


def test_a_tiled_clips_bar_text_and_previews_and_the_bits_of_the_library(
    machine: StandIn, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tiled_plans(monkeypatch)
    frames = pictures(33)
    out, _ = run_video(handle(tmp_path), plans.Settings(window=5), frames, RESOLUTION, "4")
    expected = clip.upscale(
        fresh(machine),
        frames.numpy(),
        TARGET,
        plans.SEED,
        window=5,
        encode_tiles=TILES,
        decode_tiles=TILES,
        correct=True,
    )
    assert out.numpy().tobytes() == expected.tobytes(), "the callbacks only observe"
    (bar,) = ProgressBar.created
    assert bar.updates == sorted(bar.updates) and bar.updates[-1] == progress.BAR
    said = status(node_texts("4"))
    encoding = [each for each in said if "encod" in each]
    assert encoding[0] == "encoding"
    assert encoding[1:] == [f"tile {n}/4 \u00b7 encoding" for n in (1, 2, 3, 4, 4)]
    windows = [each for each in said if "window" in each]
    assert windows == [f"window {n}/3 \u00b7 upscaling" for n in (1, 2, 3)]
    decoding = [each for each in said if "decod" in each]
    # The last window says the decode is next (tile 1), and its first tile then tells 0 of 4.
    assert decoding[:6] == [f"tile {n}/4 \u00b7 decoding" for n in (1, 1, 2, 3, 4, 4)]
    written = [int(each.split(" ")[1].split("/")[0]) for each in decoding[6:]]
    assert all(each.startswith("frames ") for each in decoding[6:])
    assert written and written == sorted(written) and written[-1] == 33
    # One preview a decode slice, the newest frame, at ComfyUI's size.
    assert len(bar.previews) == len(written) and all(
        preview[2] == args.preview_size and preview[1].size == (80, 48)  # type: ignore[index]
        for _, preview in bar.previews
    )
    # The first message is "planning\u2026", the last the result alone.
    texts = node_texts("4")
    assert texts[0] == "planning\u2026" and texts[-1].startswith("33 frames in ")


def test_each_picture_of_a_tiled_batch_is_a_preview_and_the_bits_are_the_librarys(
    machine: StandIn, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tiled_plans(monkeypatch)
    batch = pictures(3)
    out, _ = upscale_image(handle(tmp_path), plans.Settings(seed=7), batch, RESOLUTION, "5")
    alone = fresh(machine)
    for index in range(3):
        expected = still.upscale(
            alone,
            batch[index].numpy(),
            TARGET,
            7,
            encode_tiles=TILES,
            decode_tiles=TILES,
            correct=True,
        )
        assert out[index].numpy().tobytes() == expected.tobytes()
    (bar,) = ProgressBar.created
    assert bar.updates == sorted(bar.updates) and bar.updates[-1] == progress.BAR
    assert len(bar.previews) == 3
    said = status(node_texts("5"))
    assert "picture 2/3 \u00b7 tile 3/4 \u00b7 encoding" in said
    assert "picture 3/3 \u00b7 tile 2/4 \u00b7 decoding" in said
    assert node_texts("5")[-1].startswith("3 pictures in ")


def test_at_maximum_each_runs_previews_are_frames_the_output_then_holds(
    machine: StandIn, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tiled_plans(monkeypatch)
    told: list[tuple[int, int, np.ndarray[Any, Any] | None]] = []
    real = progress.Progress.decoded

    def spy(self: progress.Progress, written: int, total: int, frame: Any = None) -> None:
        told.append((written, total, None if frame is None else np.array(frame)))
        real(self, written, total, frame)

    monkeypatch.setattr(progress.Progress, "decoded", spy)
    out, _ = run_video(
        handle(tmp_path), plans.Settings(window=5), pictures(33), RESOLUTION, "7", quality="maximum"
    )
    rows = [each for each in out.numpy()]
    sent = [frame for _, _, frame in told if frame is not None]
    # A frame sent is one the output holds in the end: a run's dropped frames are never shown.
    assert sent and all(any(np.array_equal(frame, row) for row in rows) for frame in sent)
    # Every slice sends one, the newest frame its run keeps so far, the last of each of the four
    # runs among them.
    assert all(frame is not None for _, _, frame in told)
    assert len([1 for written, total, _ in told if written == total]) == 4
    (bar,) = ProgressBar.created
    assert len(bar.previews) == len(sent)


def stop_at(update: int) -> None:
    seen: list[float] = []

    def hook(value: float) -> None:
        seen.append(value)
        if len(seen) == update:
            management.interrupt_current_processing()

    ProgressBar.on_update = hook


@pytest.mark.parametrize("update", [3, 5, 6, 9])
def test_cancel_is_honoured_after_a_tile_and_the_card_is_freed(
    machine: StandIn, tmp_path, monkeypatch: pytest.MonkeyPatch, update: int
) -> None:
    tiled_plans(monkeypatch)
    h = handle(tmp_path)
    stop_at(update)
    with pytest.raises(management.InterruptProcessingException):
        run_video(h, plans.Settings(window=5), pictures(33), RESOLUTION, "6")
    (bar,) = ProgressBar.created
    # The update 1 is the pass beginning, 2 its first encode tile told (0 of 4), and so on: the
    # run stops at the update that was asked for, between two tiles, with nothing run on after.
    assert len(bar.updates) == update
    entry = models.kept(h)
    assert entry is not None and not entry.busy
    assert not machine.made[0].on_card  # type: ignore[attr-defined]
    # The flag was cleared as it raised: the next run goes through.
    ProgressBar.on_update = None
    out, _ = run_video(h, plans.Settings(window=5), pictures(5), RESOLUTION, "6")
    assert out.shape[0] == 5


def test_the_models_load_is_told_on_a_bar_of_its_own_before_the_upscales(
    machine: StandIn, tmp_path
) -> None:
    machine.reads = True
    h = handle(tmp_path)
    upscale_image(h, plans.Settings(), pictures(1), RESOLUTION, "8")
    first, second = ProgressBar.created
    # The reading's bar: both files' pieces, 2 x 2 x READ_PIECE bytes in all.
    assert first.updates == [READ_PIECE, 2 * READ_PIECE, 3 * READ_PIECE, 4 * READ_PIECE]
    assert first.totals[-1] >= 4 * READ_PIECE
    # The upscale's bar starts again at 0.
    assert second.updates[0] == 0 and second.updates[-1] == progress.BAR
    reading = [where(each) for each in node_texts("8") if each.startswith("reading ")]
    assert reading and reading[0].endswith(" GB")
    # Kept in memory, the next run reads nothing and has no bar for it.
    ProgressBar.created.clear()
    upscale_image(h, plans.Settings(), pictures(1), RESOLUTION, "8")
    assert len(ProgressBar.created) == 1


def test_a_download_is_told_to_the_session_through_pull(
    machine: StandIn, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    told: list[tuple[str, int, int]] = []
    real = pull.resolve

    def resolve(*arguments: Any, **options: Any) -> Any:
        options["progress"]("w.safetensors", 0, 100)
        options["progress"]("w.safetensors", 100, 100)
        told.append(("resolved", 0, 0))
        return real(*arguments, **options)

    monkeypatch.setattr(pull, "resolve", resolve)
    upscale_image(handle(tmp_path), plans.Settings(), pictures(1), RESOLUTION, "9")
    assert told == [("resolved", 0, 0)]
    (first, _) = ProgressBar.created
    # The total is the download's and the reading announced before it: the VAE's pinned size (the
    # stand-in DiT's name is no pin's).
    assert first.updates == [100] and first.totals == [100 + pull.PINNED[plans.VAE_FILE].size]
    assert node_texts("9")[:2] == [
        "planning\u2026",
        "downloading 0.0/0.0 MB\n0 s elapsed \u00b7 estimating time left",
    ]

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
from comfyui import models, nodes, plans, progress
from standins import READ_PIECE, RESOLUTION, StandIn, handle, upscale_image
from standins import upscale_video as run_video
from test_nodes import TARGET, fresh, pictures
from test_still import TILES

from seedvr2x import clip, model, pull, still
from seedvr2x.clock import taking, wall
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
    plan: str = "the plan",
) -> tuple[progress.Progress, Clock, list[str], ProgressBar]:
    clock, shown = Clock(), []
    bar = ProgressBar(progress.BAR)
    return progress.Progress(passes, costs, bar, shown.append, plan, clock), clock, shown, bar


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


def lines(text: str) -> list[str]:
    return text.splitlines()


def test_the_status_of_a_shot_of_a_clip() -> None:
    passes = clip_passes(shots_of(4), 5, TILES, None)
    p, _, shown, _ = make(passes)
    tiles, windows = len(passes[1].encode), len(passes[1].windows)
    assert tiles == 4
    label = "Shot 2 of 4 (frames 121 to 240)"
    p.begin(1)
    assert lines(shown[-1])[:2] == ["the plan", f"{label}: encoding"]
    p.tile("encode", 0, tiles)
    p.tile("encode", 2, tiles)
    assert lines(shown[-1])[1] == f"{label}: encoding, tile 3 of {tiles}"
    p.tile("encode", tiles, tiles)
    assert lines(shown[-1])[1] == f"{label}: encoding, tile {tiles} of {tiles}"
    p.encoded()
    assert lines(shown[-1])[1] == f"{label}: upscaling, window 1 of {windows}"
    p.window(1)
    assert lines(shown[-1])[1] == f"{label}: upscaling, window 2 of {windows}"
    for number in range(2, windows + 1):
        p.window(number)
    assert lines(shown[-1])[1] == f"{label}: decoding, 0 of 120 frames"
    p.decoded(60, 120)
    assert lines(shown[-1])[1] == f"{label}: decoding, 60 of 120 frames"
    # Every message carries the plan, and the box is replaced, not appended to.
    assert all(lines(each)[0] == "the plan" and len(lines(each)) == 3 for each in shown)


def test_the_status_of_a_tiled_decode_and_of_a_clip_of_one_shot() -> None:
    passes = clip_passes([Shot(0, 480)], 5, None, TILES)
    p, _, shown, _ = make(passes)
    windows, tiles = len(passes[0].windows), len(passes[0].decode)
    p.begin(0)
    p.encoded()
    p.window(4)
    assert lines(shown[-1])[1] == f"Frames 1 to 480: upscaling, window 5 of {windows}"
    for number in range(5, windows + 1):
        p.window(number)
    assert lines(shown[-1])[1] == f"Frames 1 to 480: decoding, tile 1 of {tiles}"
    p.tile("decode", 1, tiles)
    assert lines(shown[-1])[1] == f"Frames 1 to 480: decoding, tile 2 of {tiles}"


def test_the_status_of_a_picture_of_a_batch_and_of_one_picture() -> None:
    passes = progress.picture_passes(8, TARGET, TILES, TILES)
    p, _, shown, _ = make(passes, progress.PICTURE)
    p.begin(2)
    p.tile("decode", 1, 4)
    assert lines(shown[-1])[1] == "Picture 3 of 8: decoding, tile 2 of 4"
    alone, _, shown, _ = make(progress.picture_passes(1, TARGET, None, None), progress.PICTURE)
    alone.begin(0)
    alone.encoded()
    alone.window(1)
    assert lines(shown[-1])[1] == "Picture: decoding"


def test_the_time_line_says_the_time_left_once_the_first_window_is_measured() -> None:
    p, clock, shown, _ = make(clip_passes(shots_of(2, 33)))
    p.begin(0)
    clock.t += 72.0
    p.encoded()
    assert lines(shown[-1])[2] == "1 min 12 s so far", "no window measured yet: no time left"
    clock.t += 10.0
    p.window(1)
    assert lines(shown[-1])[2] == f"1 min 22 s so far, {taking(wall(p.left()))} left"
    assert lines(shown[-1])[2].split(", ")[1].startswith("about ")
    # The time left is the units left at the rates: the window's own now, the others the card's.
    assert p.left() == pytest.approx(
        sum(u.weight for u in p.passes[0].windows[1:]) * p.rate("window")
        + sum(u.weight for u in p.passes[1].windows) * p.rate("window")
        + sum(p.rate(u.kind) * u.weight for u in p.passes[0].decode + p.passes[1].decode)
        + p.rate("encode") * p.passes[1].encode[0].weight
    )


def test_the_end_says_the_plan_and_what_was_made_in_how_long() -> None:
    p, clock, shown, bar = make(clip_passes(shots_of(1, 33)), plan="line one\nline two")
    p.begin(0)
    clock.t += 272.0
    p.finish("480 frames")
    assert shown[-1] == "line one\nline two\n480 frames in 4 min 32 s"
    assert bar.updates[-1] == progress.BAR


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
    # The run starts at its third frame of the shot, 121 + 2 of the clip: its own frames.
    assert lines(shown[-1])[1] == "Shot 2 of 4, run 3 of 4 (frames 123 to 240): encoding"
    p.encoded()
    p.window(1)
    assert lines(shown[-1])[1] == (
        f"Shot 2 of 4, run 3 of 4 (frames 123 to 240): upscaling, window 2 of {windows}"
    )
    alone = progress.clip_passes([Shot(0, 480)], TARGET, 5, None, None, runs=2)
    p, _, shown, _ = make(alone)
    p.run(0, 2)
    p.encoded()
    p.window(4)
    count = len(alone[1].windows)
    assert lines(shown[-1])[1] == f"Frames 3 to 480, run 2 of 2: upscaling, window 5 of {count}"
    # Standard has no run part.
    p, _, shown, _ = make(progress.clip_passes([Shot(0, 480)], TARGET, 5, None, None))
    p.run(0, 1)
    assert lines(shown[-1])[1] == "Frames 1 to 480: encoding"
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
    label = "Shot 2 of 4, run 3 of 4 (frames 123 to 240)"
    assert lines(shown[-1])[1] == f"{label}: decoding, 0 of 118 frames"
    p.decoded(60, 118)
    assert lines(shown[-1])[1] == f"{label}: decoding, 60 of 118 frames"
    p.decoded(118, 118)
    assert lines(shown[-1])[1] == f"{label}: decoding, 118 of 118 frames"
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
    said = [each.splitlines()[-2] for each in node_texts("7") if len(each.splitlines()) >= 3]
    # Run 3 of the clip of one shot begins at its third frame and has 31 frames.
    assert "Frames 3 to 33, run 3 of 4: encoding" in said
    assert "Frames 3 to 33, run 3 of 4: decoding, 31 of 31 frames" in said
    assert "Frames 2 to 33, run 2 of 4: decoding, 32 of 32 frames" in said
    assert not any("run 3 of 4" in each and "of 33 frames" in each for each in said)


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
        assert lines(text)[2] == "0 s so far"
        # The same text as the event's, the time moved on, once a second.
        clock.t += 5.0
        cue.fire()
        assert len(shown) == count + 1
        assert lines(shown[-1])[:2] == lines(text)[:2]
        assert lines(shown[-1])[2] == "5 s so far"
        clock.t += 1.0
        cue.fire()
        assert lines(shown[-1])[2] == "6 s so far"
        assert bar.updates == updates, "a tick never moves the bar"
        # The next event says the time as well, from the same clock.
        clock.t += 2.0
        p.encoded()
        assert lines(shown[-1])[2] == "8 s so far"
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
        assert lines(shown[-1])[2] == f"33 s so far, {taking(wall(p.left()))} left"


def test_no_tick_after_the_end_and_no_thread_alive() -> None:
    p, clock, shown, bar = make(clip_passes(shots_of(1, 33)), plan="the plan")
    cue = Cue()
    with p.running(pause=cue.pause):
        cue.settle()
        p.begin(0)
        clock.t += 4.0
        p.finish("33 frames")
        summary = shown[-1]
        clock.t += 9.0
        cue.fire()
        assert shown[-1] == summary == "the plan\n33 frames in 4 s"
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

    p = progress.Progress(clip_passes(shots_of(1, 33)), progress.CLIP, bar, show, "", clock)
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


def test_the_download_and_the_reading_share_one_bar_of_one_total_each_with_its_line() -> None:
    p, clock, shown = preparing()
    dit, vae = plans.DEFAULT_DIT, plans.VAE_FILE
    # The reading's bytes are announced before the first download, and every download's size is
    # told before its first byte (pull.fetch): the bar is made at the first byte, its total whole.
    p.will_read(17_000_000_000)
    p.download(dit, 0, 16_500_000_000)
    p.download(vae, 0, 500_000_000)
    assert ProgressBar.created == [] and shown[-1].startswith("downloading the VAE: 0.0 of 0.5 GB")
    clock.t += 30.0
    p.download(dit, 4_200_000_000, 16_500_000_000)
    assert shown[-1] == "downloading the sharp 7B (float16): 4.2 of 16.5 GB\n30 s so far"
    p.download(dit, 16_500_000_000, 16_500_000_000)
    p.download(vae, 500_000_000, 500_000_000)
    p.read(vae, 100_000_000, 500_000_000)
    assert shown[-1].splitlines()[0] == "reading the VAE into memory: 0.1 of 0.5 GB"
    p.read(vae, 500_000_000, 500_000_000)
    p.read(dit, 9_000_000_000, 16_500_000_000)
    assert shown[-1].splitlines()[0] == "reading the sharp 7B (float16) into memory: 9.0 of 16.5 GB"
    p.read(dit, 16_500_000_000, 16_500_000_000)
    (bar,) = ProgressBar.created
    assert bar.updates == sorted(bar.updates) and bar.updates[0] == 4_200_000_000
    # One total throughout, the downloads' and the reading's: the fraction never falls, and the
    # bar ends full.
    assert set(bar.totals) == {17_000_000_000 + 17_000_000_000}
    assert bar.updates[-1] == bar.totals[-1]


def test_a_small_file_is_said_in_megabytes() -> None:
    p, _, shown = preparing()
    p.download("transnetv2.safetensors", 12_100_000, 30_500_000)
    assert shown[-1].splitlines()[0] == "downloading the shot detector: 12.1 of 30.5 MB"


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
    clock, shown = Clock(), []
    bar = ProgressBar(450)
    tell = progress.scanned(bar, shown.append, clock)
    clock.t += 5.0
    tell(200, 450)
    assert bar.updates == [200] and shown[-1] == "looking for cuts: 200 of 450 frames\n5 s so far"
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
    return [each.splitlines()[-2] for each in texts if len(each.splitlines()) >= 3]


def test_a_tiled_clips_bar_text_and_previews_and_the_bits_of_the_library(
    machine: StandIn, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tiled_plans(monkeypatch)
    frames = pictures(33)
    out, plan = run_video(handle(tmp_path), plans.Settings(window=5), frames, RESOLUTION, "4")
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
    encoding = [each for each in said if "encoding" in each]
    assert encoding[0] == "Frames 1 to 33: encoding"
    assert [each.rsplit(", ", 1)[-1] for each in encoding[1:]] == [
        "tile 1 of 4",
        "tile 2 of 4",
        "tile 3 of 4",
        "tile 4 of 4",
        "tile 4 of 4",
    ]
    assert [each for each in said if "upscaling" in each] == [
        f"Frames 1 to 33: upscaling, window {n} of 3" for n in (1, 2, 3)
    ]
    decoding = [each.split(": ")[1] for each in said if "decoding" in each]
    # The last window says the decode is next (tile 1), and its first tile then tells 0 of 4.
    assert decoding[:6] == ["decoding, tile 1 of 4"] + [
        f"decoding, tile {n} of 4" for n in (1, 2, 3, 4, 4)
    ]
    written = [int(each.split(", ")[1].split(" ")[0]) for each in decoding[6:]]
    assert written and written == sorted(written) and written[-1] == 33
    # One preview a decode slice, the newest frame, at ComfyUI's size.
    assert len(bar.previews) == len(written) and all(
        preview[2] == args.preview_size and preview[1].size == (80, 48)  # type: ignore[index]
        for _, preview in bar.previews
    )
    # The last message: the plan and the result, no status line.
    last = node_texts("4")[-1]
    assert last.startswith(plan) and last.splitlines()[-1].startswith("33 frames in ")


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
    assert "Picture 2 of 3: encoding, tile 3 of 4" in said
    assert "Picture 3 of 3: decoding, tile 2 of 4" in said
    assert node_texts("5")[-1].splitlines()[-1].startswith("3 pictures in ")


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
    reading = [each for each in node_texts("8") if "into memory" in each]
    assert reading and reading[0].startswith("reading ")
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
    assert any(text.startswith("downloading w.safetensors: ") for text in node_texts("9"))

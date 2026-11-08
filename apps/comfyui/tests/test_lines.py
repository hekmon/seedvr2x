"""The one line each node shows (comfyui/progress.py, plans.py): a node's text box is a progress
hint of at most 40 characters on one line with no name, never a log (the user, 2026-11-08), and the
console has the rest. Every line a node can send is held here in each of its states: its exact
form with ordinary numbers, and its length with numbers far past any real run (thousands of
shots, hours left, a byte count of 16 digits), a line that does not fit dropping its lowest
priority parts and never passing the limit. tests/conftest.py checks every text any test of the
nodes sent, so that no path is missed."""

import logging
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from comfy.utils import ProgressBar
from comfyui import nodes, plans, progress, words
from standins import not_one_short_line, where
from test_progress import Clock

from seedvr2x import waiting
from seedvr2x.passes import Plan, Tiles
from seedvr2x.still_plan import PictureTiles

DOT = progress.SEPARATOR


def one_short_line(text: str) -> None:
    assert not not_one_short_line([text]), text
    assert len(text) <= 40 and "\n" not in text and "seedvr2x" not in text


def run(
    *,
    shot: int = 2,
    of: int = 4,
    run_: int = 1,
    runs: int = 1,
    noun: str = "shot",
    encode: int = 6,
    windows: int = 3,
    decode: int = 4,
    frames: int = 118,
) -> tuple[progress.Progress, Clock, list[str]]:
    """A progress over one pass the line counts as given, its units unweighted: the shot (or
    picture) `shot` of `of`, the run `run_` of `runs`, with `encode` and `decode` tiles."""
    unit = progress.Unit
    one = progress.Pass(
        "label",
        frames,
        [unit("encode", 1.0) for _ in range(encode)],
        [unit("window", 1.0) for _ in range(windows)],
        [unit("decode", 1.0) for _ in range(decode)],
        shot=shot,
        run=run_,
        of=of,
        runs=runs,
        noun=noun,
    )
    clock, shown = Clock(), []
    bar = ProgressBar(progress.BAR)
    p = progress.Progress([one], progress.CLIP, bar, shown.append, clock)
    p.begin(0)
    return p, clock, shown


def left_in(p: progress.Progress, seconds: float) -> None:
    """The progress has measured a window and expects `seconds` more."""
    p.made["window"] = [1.0, 1.0]
    p.left = lambda: seconds  # type: ignore[method-assign]


# ---- The exact forms, with ordinary numbers -----------------------------------------------------


def test_upscale_video_shows_the_shot_the_step_and_the_time() -> None:
    p, clock, shown = run()
    assert shown[-1] == "shot 2/4 · encoding · 0 s"
    p.tile("encode", 2, 6)
    assert shown[-1] == "shot 2/4 · encode tile 3/6 · 0 s"
    clock.t += 65.0
    p.encoded()
    assert shown[-1] == "shot 2/4 · window 1/3 · 1 min 5 s"
    left_in(p, 180.0)
    p.window(1)
    assert shown[-1] == "shot 2/4 · window 2/3 · ~3 min left"
    p.window(2)
    p.window(3)
    # 40 characters with the time: "decode tile", not "decoding, tile", so that it fits.
    assert shown[-1] == "shot 2/4 · decode tile 1/4 · ~3 min left"
    p.tile("decode", 1, 4)
    assert where(shown[-1]) == "shot 2/4 · decode tile 2/4"
    quick, _, shown = run(decode=1)
    quick.window(3)
    # A window is made: the time left, a second at the stand-in's rates.
    assert shown[-1] == "shot 2/4 · decoding 0/118 · ~1 s left"
    quick.decoded(60, 118)
    assert shown[-1] == "shot 2/4 · decoding 60/118 · ~1 s left"
    quick.finish("480 frames")
    assert shown[-1] == "480 frames in 0 s"


def test_a_clip_of_one_shot_has_no_shot_part_and_several_runs_have_a_run_part() -> None:
    p, _, shown = run(of=1, shot=1)
    assert shown[-1] == "encoding · 0 s"
    p, _, shown = run(of=1, shot=1, run_=3, runs=4)
    assert shown[-1] == "run 3/4 · encoding · 0 s"
    left_in(p, 180.0)
    p.encoded()
    assert shown[-1] == "run 3/4 · window 1/3 · ~3 min left"
    p, _, shown = run(run_=3, runs=4)
    assert shown[-1] == "shot 2/4 · run 3/4 · encoding · 0 s"
    left_in(p, 180.0)
    p.encoded()
    # 45 characters whole: the word of the place goes first, and the run and the time both fit.
    assert shown[-1] == "2/4 · run 3/4 · window 1/3 · ~3 min left"
    # Where both can't, the run goes (the bar counts the runs) and the time left stays.
    p.tile("decode", 0, 456)
    assert shown[-1] == "2/4 · decode tile 1/456 · ~3 min left"


def test_the_end_of_a_clip_and_of_a_batch() -> None:
    for what, said in (
        ("480 frames", "480 frames in 4 min 32 s"),
        ("1 frame", "1 frame in 4 min 32 s"),
    ):
        p, clock, shown = run()
        clock.t += 272.0
        p.finish(what)
        assert shown[-1] == said
    for what, said in (("8 pictures", "8 pictures in 41 s"), ("1 picture", "1 picture in 41 s")):
        p, clock, shown = run(noun="picture", of=8)
        clock.t += 41.0
        p.finish(what)
        assert shown[-1] == said


def test_upscale_image_shows_the_picture_the_step_and_the_time() -> None:
    p, clock, shown = run(noun="picture", shot=3, of=8, windows=1)
    assert shown[-1] == "picture 3/8 · encoding · 0 s"
    p.tile("encode", 1, 6)
    assert shown[-1] == "picture 3/8 · encode tile 2/6 · 0 s"
    clock.t += 6.0
    p.encoded()
    assert shown[-1] == "picture 3/8 · upscaling · 6 s"
    p.window(1)
    # A window is measured: the time is the time left, which the word "picture" makes room for.
    assert shown[-1] == "3/8 · decode tile 1/4 · ~1 s left"
    p.tile("decode", 1, 4)
    assert where(shown[-1]) == "3/8 · decode tile 2/4"
    one, _, shown = run(noun="picture", shot=1, of=1, windows=1, decode=1, frames=1)
    one.encoded()
    one.window(1)
    assert shown[-1] == "decoding · ~1 s left"


def test_the_time_so_far_and_the_time_left_are_rough_and_short() -> None:
    assert progress.time_left(42) == "~42 s left"
    assert progress.time_left(12 * 60) == "~12 min left"
    assert progress.time_left(4020) == "~1 h 7 min left"
    for seconds in (1, 59, 60, 3599, 3600, 86_400, 10**7):
        one_short_line(progress.time_left(seconds))


def test_the_states_of_the_card() -> None:
    assert progress.FREEING == "freeing the GPU… Cancel again to leave"
    assert progress.STOPPED == "GPU busy: run again or restart ComfyUI"
    assert progress.FAILED == "GPU failed: restart ComfyUI"
    assert progress.PLANNING == "planning…"
    for line in (progress.FREEING, progress.STOPPED, progress.FAILED, progress.PLANNING):
        one_short_line(line)
    # The full sentences are the console's.
    assert "Cancel again to leave it" in progress.FREEING_SAID
    assert "restart ComfyUI" in progress.STOPPED_SAID and "restart ComfyUI" in progress.WAY


def test_a_wait_on_the_gpu_past_its_bound_shows_the_minutes_it_has_lasted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p, _, shown = run()
    for lasted, said in (
        (75.0, "GPU not answering for 1 min"),
        (3 * 60.0, "GPU not answering for 3 min"),
    ):
        monkeypatch.setattr(waiting, "stalled", lambda: True)
        monkeypatch.setattr(waiting, "longest", lambda lasted=lasted: lasted)
        p.tick()
        assert shown[-1] == said
    monkeypatch.setattr(waiting, "longest", lambda: 10.0**9)
    p.tick()
    one_short_line(shown[-1])
    monkeypatch.setattr(waiting, "stalled", lambda: False)
    p.tick()
    assert shown[-1] == "shot 2/4 · encoding · 0 s"


def test_a_refusal_shows_a_line_the_cards_states_their_own() -> None:
    assert words.Refused("any").line == "refused: see the error" == progress.REFUSED
    assert words.card_busy().line == progress.STOPPED
    assert words.card_failed("boom").line == progress.FAILED
    assert words.card_failed("boom", earlier=True).line == progress.FAILED
    assert (
        str(words.Refused("x", line="short")) == "x"
        and words.Refused("x", line="short").line == "short"
    )
    for line in (progress.REFUSED, progress.STOPPED, progress.FAILED):
        one_short_line(line)


# ---- The files before the upscale ----------------------------------------------------------


def test_the_downloads_and_the_reading_are_one_line_each() -> None:
    shown: list[str] = []
    p = progress.Preparing(shown.append, plans.file_label)
    p.will_read(8_800_000_000)
    p.download(plans.DEFAULT_DIT, 3_100_000_000, 8_300_000_000)
    assert shown[-1] == "downloading 3.1/8.3 GB"
    p.read(plans.VAE_FILE, 2_000_000_000, 8_800_000_000)
    assert shown[-1] == "reading 2.0/8.8 GB"
    detector = progress.Preparing(shown.append, plans.file_label)
    detector.download("transnetv2.safetensors", 12_100_000, 30_600_000)
    assert shown[-1] == "downloading 12.1/30.6 MB"


def test_the_scan_of_the_cuts_is_one_line() -> None:
    shown: list[str] = []
    progress.scanned(ProgressBar(120), shown.append)(64, 120)
    assert shown[-1] == "scanning: 64/120 frames"


# ---- The plan only and the cuts ------------------------------------------------------------


def clip_plan(**given: object) -> Plan:
    base: dict[str, object] = {
        "window": 11,
        "encode_tiles": Tiles(1024),
        "decode_tiles": Tiles(1088),
        "swapped": 0,
    }
    return Plan(**{**base, **given})  # type: ignore[arg-type]


def test_a_plan_only_shows_the_main_numbers_and_drops_from_the_end() -> None:
    assert plans.clip_line(clip_plan()) == "plan: window 41 frames · decode 1088 px"
    assert plans.clip_line(clip_plan(window=None)) == "plan: one window · decode 1088 px"
    assert plans.clip_line(clip_plan(decode_tiles=None)) == "plan: window 41 frames · decode whole"
    tiles = PictureTiles(Tiles(1024), Tiles(1088), 0, 0)
    assert plans.picture_line(tiles, 0) == "plan: encode 1024 px · decode 1088 px"
    whole = PictureTiles(None, None, 0, 0)
    assert plans.picture_line(whole, 0) == "plan: encode whole · decode whole"
    # Whatever fits is said, in the order of importance; the rest is dropped from the end.
    assert plans.picture_line(whole, 12) == "plan: encode whole · decode whole"
    short = plans.clip_line(clip_plan(window=None, decode_tiles=None, encode_tiles=None, swapped=7))
    assert short == "plan: one window · decode whole"
    for each in (clip_plan(window=2**20), clip_plan(decode_tiles=Tiles(10**9), swapped=10**6)):
        one_short_line(plans.clip_line(each))
    one_short_line(plans.picture_line(PictureTiles(Tiles(10**12), Tiles(10**12), 0, 0), 10**9))


def test_the_cuts_node_shows_the_shots_and_the_possible_cuts() -> None:
    assert plans.cuts_line(2, 1, 0.3) == "2 shots · 1 possible cut below 0.30"
    assert plans.cuts_line(1, 0, 0.3) == "1 shot · no possible cut"
    assert plans.cuts_line(3, 2, 0.3) == "3 shots · 2 possible cuts below 0.30"
    assert plans.cuts_line(3, 2, 0.05) == "3 shots · 2 possible cuts below 0.05"
    assert plans.one_shot_line(120) == "one shot: 120 frames"
    for line in (
        plans.cuts_line(10**9, 10**9, 0.35),
        plans.cuts_line(10**15, 0, 0.35),
        plans.one_shot_line(10**30),
    ):
        one_short_line(line)
    # Past the limit the lowest priority parts go: the threshold, then the possible cuts.
    assert plans.cuts_line(1000, 100_000, 0.3) == "1000 shots · 100000 possible cuts"
    assert plans.cuts_line(10**8, 10**8, 0.3) == "100000000 shots"


# ---- Numbers far past any real run ------------------------------------------------------------


def test_no_line_of_a_run_passes_the_limit_whatever_the_numbers() -> None:
    """A clip of thousands of shots, runs, windows and tiles, frames written in the tens of
    thousands, hours of time left, days so far: every line is one short line."""
    for noun in ("shot", "picture"):
        p, clock, shown = run(
            shot=1234, of=5678, run_=4, runs=4, noun=noun, encode=456, windows=456, decode=456
        )
        left_in(p, 10 * 3600.0 + 7 * 60)
        clock.t += 4 * 86_400.0
        p.tile("encode", 122, 456)
        p.tile("encode", 456, 456)
        p.encoded()
        for number in range(1, 457):
            p.window(number)
        p.tile("decode", 122, 456)
        big, clock, shown_big = run(
            shot=1234, of=5678, run_=4, runs=4, noun=noun, decode=1, frames=23_456, windows=456
        )
        clock.t += 4 * 86_400.0
        left_in(big, 10 * 3600.0 + 7 * 60)
        big.window(456)
        big.decoded(12_345, 23_456)
        big.decoded(23_456, 23_456)
        big.finish("23456 frames")
        assert len(shown) > 450 and len(shown_big) == 5
        for text in (*shown, *shown_big):
            one_short_line(text)
        assert shown_big[-1] == "23456 frames in 4 d 0 h"
        assert shown_big[-2].endswith("decoding 23456/23456")
        # Days so far, before a window is measured, are said too.
        early, clock, shown_early = run(shot=1, of=1)
        clock.t += 4 * 86_400.0
        early.tile("encode", 1, 6)
        assert shown_early[-1] == "encode tile 2/6 · 4 d 0 h"


def test_the_longest_status_keeps_its_step_and_drops_the_rest_in_order() -> None:
    """The run first, then the time (the run back where the time can't fit), then the word of the
    place; the step is never dropped."""

    def decoding(*, shot: int = 2, of: int = 4, run_: int = 1, runs: int = 1) -> str:
        p, _, shown = run(
            encode=1, windows=1, decode=1, frames=23_456, shot=shot, of=of, run_=run_, runs=runs
        )
        left_in(p, 10 * 3600.0)
        p.encoded()
        p.window(1)
        p.decoded(12_345, 23_456)
        return shown[-1]

    whole = "shot 1234/5678 · run 4/4 · decoding 12345/23456 · ~10 h 0 min left"
    assert len(whole) > 40
    assert decoding(shot=1234, of=5678, run_=4, runs=4) == "shot 1234/5678 · decoding 12345/23456"
    # Without the run it is 43 characters: the word of the place goes next.
    assert decoding(shot=1234567, of=7654321) == "1234567/7654321 · decoding 12345/23456"
    # And past that the step alone.
    assert decoding(shot=10**12, of=10**13) == "decoding 12345/23456"
    # A step that is itself over the limit is cut to it, with an ellipsis.
    p, _, shown = run(encode=1, windows=1, decode=1, frames=10**18)
    p.encoded()
    p.window(1)
    p.decoded(10**17, 10**18)
    assert len(shown[-1]) == 40 and shown[-1].endswith("…")


def test_a_download_of_any_size_is_one_short_line() -> None:
    shown: list[str] = []
    p = progress.Preparing(shown.append, plans.file_label)
    p.will_read(10**18)
    for done, total in ((0, 10**18), (10**15, 10**16), (10**17, 10**18)):
        p.download("w.safetensors", done, total)
        p.read("v.safetensors", done, total)
    assert shown and all(len(each) <= 40 and "\n" not in each for each in shown)
    detector = progress.Preparing(shown.append, plans.file_label)
    detector.download("transnetv2.safetensors", 10**15, 10**16)
    progress.scanned(ProgressBar(10**18), shown.append)(10**17, 10**18)
    assert all(not not_one_short_line([each]) for each in shown)


# ---- The nodes' own runs: the lines every run sends, and the console has the rest ---------------


def test_the_server_check_finds_a_long_text_a_second_line_and_a_name() -> None:
    assert not_one_short_line(["x" * 41]) and not_one_short_line(["a\nb"])
    assert not_one_short_line(["seedvr2x"]) and not not_one_short_line(["x" * 40])


def test_a_refusal_is_in_the_console_once_and_its_line_is_on_the_node(
    caplog: pytest.LogCaptureFixture,
) -> None:
    shown: list[str] = []
    with caplog.at_level(logging.INFO, logger="seedvr2x_comfyui"):
        for _ in range(2):
            with pytest.raises(words.Refused):
                with nodes._refusing(shown.append):  # pyright: ignore[reportPrivateUsage]
                    raise words.Refused("the text of the refusal")
    assert shown == [progress.REFUSED] * 2
    said = [r for r in caplog.records if "the text of the refusal" in r.getMessage()]
    assert len(said) == 2 and all(r.levelno == logging.WARNING for r in said)
    assert all(r.getMessage().endswith("refused: the text of the refusal") for r in said)
    # One refusal passing through two blocks is logged once.
    caplog.clear()
    error = words.Refused("again")
    with pytest.raises(words.Refused):
        with nodes._refusing(shown.append):  # pyright: ignore[reportPrivateUsage]
            with nodes._refusing(shown.append):  # pyright: ignore[reportPrivateUsage]
                raise error
    assert len([r for r in caplog.records if "again" in r.getMessage()]) == 1


def test_an_error_before_the_runs_first_line_leaves_no_planning_on_the_node(
    machine: object, tmp_path: Path
) -> None:
    # Not a refusal (those show their own line): a TypeError, as for a node given another node's
    # output for its models, raised after "planning…": the box is emptied, not left planning.
    import server
    from standins import upscale_image

    with pytest.raises(TypeError):
        upscale_image(
            cast(Any, object()), plans.Settings(), torch.zeros(1, 24, 40, 3), 48, unique_id="9"
        )
    instance = server.PromptServer.instance
    assert instance is not None
    assert [text for text, node in instance.texts if node == "9"] == [progress.PLANNING, ""]

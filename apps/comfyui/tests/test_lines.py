"""The lines each node shows (comfyui/progress.py, plans.py): a node's text box is a progress hint
of two lines at most with no name, never a log (the user, 2026-11-08: ComfyUI's box holds two
lines, a longer text must be scrolled), and the console has the rest. A running node's first line
says where the run is and what it does, never trimmed (a user who wants it whole widens the
node); its second the time, at most 40 characters. The end, a refusal, the card's states, the
Models node's end and the Cuts node's lines are one line. Every line a node can send is held here
in each of its states: its exact form with ordinary numbers, and its form with numbers far past
any real run (thousands of shots, hours left, a byte count of 16 digits): the first line whole,
the second within the limit. tests/conftest.py checks every text any test of the nodes sent, so
that no path is missed."""

import logging
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from comfy.utils import ProgressBar
from comfyui import nodes, plans, progress, words
from standins import not_a_status, where
from test_progress import Clock

from seedvr2x import waiting
from seedvr2x.passes import Plan, Tiles
from seedvr2x.still_plan import PictureTiles

DOT = progress.SEPARATOR
ESTIMATING = f"elapsed{DOT}estimating time left"


def two_lines(text: str) -> tuple[str, str]:
    """A running node's text: two lines, the second within the limit, no name."""
    assert not not_a_status([text]), text
    first, second = text.split("\n")
    assert first and len(second) <= 40 and "seedvr2x" not in text
    return first, second


def one_line(text: str) -> None:
    assert not not_a_status([text]), text
    assert "\n" not in text and "seedvr2x" not in text


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


def test_upscale_video_shows_the_shot_the_unit_and_the_action_over_the_time() -> None:
    p, clock, shown = run()
    assert shown[-1] == f"shot 2/4{DOT}encoding\n0 s {ESTIMATING}"
    p.tile("encode", 2, 6)
    assert shown[-1] == f"shot 2/4{DOT}tile 3/6{DOT}encoding\n0 s {ESTIMATING}"
    clock.t += 65.0
    p.encoded()
    # Past a minute, whole minutes, so that the second line keeps within 40 characters.
    assert shown[-1] == f"shot 2/4{DOT}window 1/3{DOT}upscaling\n1 min {ESTIMATING}"
    left_in(p, 38 * 60.0)
    p.window(1)
    assert shown[-1] == f"shot 2/4{DOT}window 2/3{DOT}upscaling\n~38 min left"
    p.window(2)
    p.window(3)
    assert shown[-1] == f"shot 2/4{DOT}tile 1/4{DOT}decoding\n~38 min left"
    p.tile("decode", 1, 4)
    assert where(shown[-1]) == f"shot 2/4{DOT}tile 2/4{DOT}decoding"
    quick, _, shown = run(decode=1)
    quick.window(3)
    # A window is made: the time left, a second at the stand-in's rates. A decode without tiles
    # counts its frames.
    assert shown[-1] == f"shot 2/4{DOT}frames 0/118{DOT}decoding\n~1 s left"
    quick.decoded(60, 118)
    assert shown[-1] == f"shot 2/4{DOT}frames 60/118{DOT}decoding\n~1 s left"
    quick.finish("480 frames")
    assert shown[-1] == "480 frames in 0 s"


def test_a_phase_done_in_one_piece_says_the_action_alone() -> None:
    p, _, shown = run(encode=1, windows=1)
    assert where(shown[-1]) == f"shot 2/4{DOT}encoding"
    p.encoded()
    assert where(shown[-1]) == f"shot 2/4{DOT}upscaling"


def test_a_clip_of_one_shot_has_no_shot_part_and_several_runs_have_a_run_part() -> None:
    p, _, shown = run(of=1, shot=1)
    assert shown[-1] == f"encoding\n0 s {ESTIMATING}"
    p, _, shown = run(of=1, shot=1, run_=3, runs=4)
    assert shown[-1] == f"run 3/4{DOT}encoding\n0 s {ESTIMATING}"
    left_in(p, 180.0)
    p.encoded()
    assert shown[-1] == f"run 3/4{DOT}window 1/3{DOT}upscaling\n~3 min left"
    p, _, shown = run(run_=3, runs=4)
    assert shown[-1] == f"shot 2/4{DOT}run 3/4{DOT}encoding\n0 s {ESTIMATING}"
    left_in(p, 180.0)
    p.encoded()
    # The first line is never trimmed: the shot, the run, the unit and the action, whatever its
    # length (44 characters here).
    assert shown[-1] == f"shot 2/4{DOT}run 3/4{DOT}window 1/3{DOT}upscaling\n~3 min left"
    p.tile("decode", 0, 456)
    assert shown[-1] == f"shot 2/4{DOT}run 3/4{DOT}tile 1/456{DOT}decoding\n~3 min left"


def test_the_end_of_a_clip_and_of_a_batch_is_one_line() -> None:
    for what, said in (
        ("480 frames", "480 frames in 4 min 32 s"),
        ("1 frame", "1 frame in 4 min 32 s"),
    ):
        p, clock, shown = run()
        clock.t += 272.0
        p.finish(what)
        assert shown[-1] == said
        one_line(said)
    for what, said in (("8 pictures", "8 pictures in 41 s"), ("1 picture", "1 picture in 41 s")):
        p, clock, shown = run(noun="picture", of=8)
        clock.t += 41.0
        p.finish(what)
        assert shown[-1] == said
        one_line(said)


def test_upscale_image_shows_the_picture_the_unit_and_the_action_over_the_time() -> None:
    p, clock, shown = run(noun="picture", shot=3, of=8, windows=1)
    assert shown[-1] == f"picture 3/8{DOT}encoding\n0 s {ESTIMATING}"
    p.tile("encode", 1, 6)
    assert shown[-1] == f"picture 3/8{DOT}tile 2/6{DOT}encoding\n0 s {ESTIMATING}"
    clock.t += 6.0
    p.encoded()
    assert shown[-1] == f"picture 3/8{DOT}upscaling\n6 s {ESTIMATING}"
    p.window(1)
    # A window is measured: the second line is the time left.
    assert shown[-1] == f"picture 3/8{DOT}tile 1/4{DOT}decoding\n~1 s left"
    p.tile("decode", 1, 4)
    assert where(shown[-1]) == f"picture 3/8{DOT}tile 2/4{DOT}decoding"
    one, _, shown = run(noun="picture", shot=1, of=1, windows=1, decode=1, frames=1)
    one.encoded()
    one.window(1)
    assert shown[-1] == "decoding\n~1 s left"


def test_the_time_so_far_and_the_time_left_are_rough_and_short() -> None:
    assert progress.time_left(42) == "~42 s left"
    assert progress.time_left(45) == "~45 s left"
    assert progress.time_left(38 * 60) == "~38 min left"
    assert progress.time_left(72 * 60) == "~1 h 12 min left"
    assert progress.times(6.0, None) == f"6 s {ESTIMATING}"
    assert progress.times(59.4, None) == f"59 s {ESTIMATING}"
    assert progress.times(4 * 60 + 32, None) == f"4 min {ESTIMATING}"
    assert progress.times(72 * 60 + 5, None) == f"72 min {ESTIMATING}"
    assert progress.times(6.0, 38 * 60.0) == "~38 min left"
    assert progress.times(999 * 60 + 59, None) == f"999 min {ESTIMATING}"
    assert progress.times(1000 * 60, None) == f"16 h {ESTIMATING}"
    assert progress.times(1000 * 3600, None) == f"41 d {ESTIMATING}"
    for seconds in (0, 1, 59, 60, 3599, 3600, 86_400, 10**7, 99_999 * 86_400):
        for left in (None, float(seconds)):
            assert len(progress.times(seconds, left)) <= 40


def test_the_states_of_the_card_are_one_line() -> None:
    assert progress.FREEING == "freeing the GPU… Cancel again to leave"
    assert progress.STOPPED == "GPU busy: run again or restart ComfyUI"
    assert progress.FAILED == "GPU failed: restart ComfyUI"
    assert progress.PLANNING == "planning…"
    for each in (progress.FREEING, progress.STOPPED, progress.FAILED, progress.PLANNING):
        one_line(each)
    # The full sentences are the console's.
    assert "Cancel again to leave it" in progress.FREEING_SAID
    assert "restart ComfyUI" in progress.STOPPED_SAID and "restart ComfyUI" in progress.WAY


def test_a_wait_on_the_gpu_past_its_bound_keeps_the_first_line_and_says_its_minutes(
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
        assert shown[-1] == f"shot 2/4{DOT}encoding\n{said}"
    monkeypatch.setattr(waiting, "longest", lambda: 10.0**9)
    p.tick()
    two_lines(shown[-1])
    monkeypatch.setattr(waiting, "stalled", lambda: False)
    p.tick()
    assert shown[-1] == f"shot 2/4{DOT}encoding\n0 s {ESTIMATING}"


def test_a_refusal_shows_a_line_the_cards_states_their_own() -> None:
    assert words.Refused("any").line == "refused: see the error" == progress.REFUSED
    assert words.card_busy().line == progress.STOPPED
    assert words.card_failed("boom").line == progress.FAILED
    assert words.card_failed("boom", earlier=True).line == progress.FAILED
    assert (
        str(words.Refused("x", line="short")) == "x"
        and words.Refused("x", line="short").line == "short"
    )
    for each in (progress.REFUSED, progress.STOPPED, progress.FAILED):
        one_line(each)


# ---- The files before the upscale ----------------------------------------------------------


def test_the_downloads_and_the_reading_are_their_bytes_over_their_time() -> None:
    shown: list[str] = []
    clock = Clock()
    p = progress.Preparing(shown.append, plans.file_label, clock)
    p.will_read(8_800_000_000)
    p.download(plans.DEFAULT_DIT, 0, 8_300_000_000)
    assert shown[-1] == f"downloading 0.0/8.3 GB\n0 s {ESTIMATING}"
    clock.t += 5.0
    # The first bytes give no rate: the time before them is the connection's.
    p.download(plans.DEFAULT_DIT, 100_000_000, 8_300_000_000)
    assert shown[-1] == f"downloading 0.1/8.3 GB\n5 s {ESTIMATING}"
    clock.t += 60.0
    # 3.0 GB in a minute since: the 5.2 GB left at that rate, 104 s.
    p.download(plans.DEFAULT_DIT, 3_100_000_000, 8_300_000_000)
    assert shown[-1] == "downloading 3.1/8.3 GB\n~2 min left"
    p.read(plans.VAE_FILE, 2_000_000_000, 8_800_000_000)
    assert shown[-1] == f"reading 2.0/8.8 GB\n1 min {ESTIMATING}"
    detector = progress.Preparing(shown.append, plans.file_label, clock)
    detector.download("transnetv2.safetensors", 12_100_000, 30_600_000)
    assert shown[-1] == f"downloading 12.1/30.6 MB\n0 s {ESTIMATING}"
    clock.t += 2.0
    detector.download("transnetv2.safetensors", 24_100_000, 30_600_000)
    assert shown[-1] == "downloading 24.1/30.6 MB\n~1 s left"


def test_a_resumed_downloads_bytes_on_disk_are_not_counted_as_come_at_its_rate() -> None:
    # huggingface_hub's bar of a resumed download starts at the bytes already on disk, told at
    # once (review M1 of impl-nodes-6): 6.0 of 8.3 GB there, then 50 MB/s, 36 s left.
    shown: list[str] = []
    clock = Clock()
    p = progress.Preparing(shown.append, plans.file_label, clock)
    p.download(plans.DEFAULT_DIT, 0, 8_300_000_000)
    clock.t += 0.5
    p.download(plans.DEFAULT_DIT, 6_000_000_000, 8_300_000_000)
    assert shown[-1] == f"downloading 6.0/8.3 GB\n0 s {ESTIMATING}"
    clock.t += 10.0
    p.download(plans.DEFAULT_DIT, 6_500_000_000, 8_300_000_000)
    assert shown[-1] == "downloading 6.5/8.3 GB\n~36 s left"
    # A second file's first bytes, a resume of its own, are left out the same way: 0.3 GB on disk
    # told at once moves no rate, and 1 s later the 50 MB/s holds.
    p.download(plans.VAE_FILE, 0, 500_000_000)
    p.download(plans.VAE_FILE, 300_000_000, 500_000_000)
    clock.t += 1.0
    p.download(plans.DEFAULT_DIT, 6_550_000_000, 8_300_000_000)
    # 0.55 GB in 11 s since the first bytes: the 1.95 GB left of both files, 39 s.
    assert where(shown[-1]) == "downloading 6.8/8.8 GB" and shown[-1].endswith("\n~39 s left")


def test_the_models_nodes_end_is_one_line() -> None:
    for name in plans.DIT_LABELS:
        for kept in ("kept", "not kept"):
            one_line(f"{plans.short_label(name)}{DOT}{kept} in RAM")


def test_the_scan_of_the_cuts_is_one_line() -> None:
    shown: list[str] = []
    progress.scanned(ProgressBar(120), shown.append)(64, 120)
    assert shown[-1] == "scanning 64/120 frames"
    one_line(shown[-1])


# ---- The plan only and the cuts ------------------------------------------------------------


def clip_plan(**given: object) -> Plan:
    base: dict[str, object] = {
        "window": 11,
        "encode_tiles": None,
        "decode_tiles": Tiles(1088),
        "swapped": 0,
    }
    return Plan(**{**base, **given})  # type: ignore[arg-type]


def test_a_plan_only_shows_the_plan_over_where_the_rest_is() -> None:
    whole_encode = f"plan: window 41 frames{DOT}decode 1088 px{DOT}encode whole"
    assert plans.clip_line(clip_plan()) == whole_encode
    tiled = plans.clip_line(clip_plan(encode_tiles=Tiles(1024)))
    assert tiled == f"plan: window 41 frames{DOT}decode 1088 px{DOT}encode 1024 px"
    assert plans.clip_line(clip_plan(window=None)).startswith(f"plan: one window{DOT}decode")
    tiles = PictureTiles(Tiles(1024), Tiles(1088), 0, 0)
    assert plans.picture_line(tiles, 0) == f"plan: encode 1024 px{DOT}decode 1088 px"
    whole = PictureTiles(None, None, 0, 0)
    # Never trimmed: every part is said, in the order of importance.
    swapped = plans.picture_line(whole, 12)
    assert swapped == f"plan: encode whole{DOT}decode whole{DOT}12 blocks swapped"
    shown = progress.status(plans.clip_line(clip_plan()), progress.IN_THE_CONSOLE)
    assert two_lines(shown) == (whole_encode, "whole plan in the console")
    for each in (clip_plan(window=2**20), clip_plan(decode_tiles=Tiles(10**9), swapped=10**6)):
        two_lines(progress.status(plans.clip_line(each), progress.IN_THE_CONSOLE))
    big = plans.picture_line(PictureTiles(Tiles(10**12), Tiles(10**12), 0, 0), 10**9)
    assert big.endswith(f"{DOT}1000000000 blocks swapped")


def test_the_cuts_node_shows_the_shots_and_the_possible_cuts_in_one_line() -> None:
    assert plans.cuts_line(2, 1, 0.3) == f"2 shots{DOT}1 possible cut below 0.30"
    assert plans.cuts_line(1, 0, 0.3) == f"1 shot{DOT}no possible cut"
    assert plans.cuts_line(3, 2, 0.3) == f"3 shots{DOT}2 possible cuts below 0.30"
    assert plans.cuts_line(3, 2, 0.05) == f"3 shots{DOT}2 possible cuts below 0.05"
    # Whole at any count: one line, never trimmed.
    many = plans.cuts_line(10**8, 10**8, 0.3)
    assert many == f"100000000 shots{DOT}100000000 possible cuts below 0.30"
    for each in (many, plans.cuts_line(10**9, 10**9, 0.35), plans.cuts_line(10**15, 0, 0.35)):
        one_line(each)


# ---- Numbers far past any real run ------------------------------------------------------------


def test_no_status_of_a_run_passes_two_lines_whatever_the_numbers() -> None:
    """A clip of thousands of shots, runs, windows and tiles, frames written in the tens of
    thousands, hours of time left, days so far: every text is two lines, the first whole and
    the second within the limit, or one line at the end."""
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
        for text in (*shown, *shown_big[:-1]):
            two_lines(text)
        assert shown_big[-1] == "23456 frames in 4 d 0 h"
        one_line(shown_big[-1])
        first, second = two_lines(shown_big[-2])
        assert first == f"{noun} 1234/5678{DOT}run 4/4{DOT}frames 23456/23456{DOT}decoding"
        assert second == "~10 h 7 min left"
    # Days so far, before a window is measured, are said in one whole unit.
    early, clock, shown_early = run(shot=1, of=1)
    clock.t += 4 * 86_400.0
    early.tile("encode", 1, 6)
    assert shown_early[-1] == f"tile 2/6{DOT}encoding\n96 h {ESTIMATING}"


def test_the_first_line_is_never_trimmed() -> None:
    p, _, shown = run(encode=1, windows=1, decode=1, frames=10**18, shot=10**12, of=10**13)
    left_in(p, 10 * 3600.0)
    p.encoded()
    p.window(1)
    p.decoded(10**17, 10**18)
    first, second = two_lines(shown[-1])
    assert first == f"shot {10**12}/{10**13}{DOT}frames {10**17}/{10**18}{DOT}decoding"
    assert second == "~10 h 0 min left"


def test_a_download_of_any_size_is_two_lines_within_the_limit() -> None:
    shown: list[str] = []
    clock = Clock()
    p = progress.Preparing(shown.append, plans.file_label, clock)
    p.will_read(10**18)
    for done, total in ((0, 10**18), (10**15, 10**16), (10**17, 10**18)):
        clock.t += 1.0
        p.download("w.safetensors", done, total)
        p.read("v.safetensors", done, total)
    assert shown
    for each in shown:
        two_lines(each)
    detector = progress.Preparing(shown.append, plans.file_label, clock)
    detector.download("transnetv2.safetensors", 10**15, 10**16)
    two_lines(shown[-1])
    progress.scanned(ProgressBar(10**18), shown.append)(10**17, 10**18)
    one_line(shown[-1])


# ---- The nodes' own runs: the lines every run sends, and the console has the rest ---------------


def test_the_server_check_finds_three_lines_a_long_second_line_an_empty_one_and_a_name() -> None:
    assert not_a_status(["a\nb\nc"]) and not_a_status(["a\n" + "x" * 41])
    assert not_a_status(["a\n"]) and not_a_status(["seedvr2x"])
    assert not not_a_status(["x" * 400, "a\n" + "x" * 40, ""]), "a first line has no limit"


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

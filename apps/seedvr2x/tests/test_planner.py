"""The command line's side of the memory planner (planner.py; DESIGN.md, Memory planner), on
plans the library's planner makes (seedvr2x/shot_plan.py): the plan's lines (said, card_detail),
the tiles beyond the shapes measured (beyond_measured), the smaller DiT files named (plan_naming,
smaller_swaps), a resume's record checked or planned again (check_recorded, plan_again), and the
plan's choices as those lines say them, on cards and hosts of every size. The plan itself is the
library's core/tests/test_shot_plan.py's, whose helpers these use. On the CPU, no weights."""

import time
from dataclasses import replace
from typing import Any

import pytest
import test_regression
from library_fixtures import LEAN_WINDOWS
from test_peaks import (
    OURS_4K,
    OURS_1080,
    gib,
)
from test_shot_plan import (  # noqa: F401
    HD,
    KEPT,
    LEFT,
    NO_PINS,
    UHD,
    bare,
    fits,
    held,
    host_need,
    in_one_window,
    numz_free,
    our_free,
    paged,
    refused,
    shot_held,
    shots,
)

import planner
from seedvr2x import memory, peaks, pull, shot_plan, still_plan, weights
from seedvr2x.frames import output_size, padded_length, padded_size
from seedvr2x.memory import Reading
from seedvr2x.passes import Plan, Tiles
from seedvr2x.peaks import BLOCK, GIB, MARGIN
from seedvr2x.shot_plan import Peaks, Pins, PlanError
from seedvr2x.shots import Shot, shot_layout


def test_an_early_reading_refusal_says_nvml_s_figure_and_the_context_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A resume's early look at the card (memory.early; DESIGN.md, Memory planner, A resume whose
    # recorded shape no longer fits, 2026-10-29) is NVML's figure less the CUDA context this
    # process is yet to make: its refusal, said at debug level, names those two figures, never
    # "its driver reports" a figure that was computed, and no driver was asked. Behind WSL2's
    # device (held) and where the driver refuses (not held). A later reading's words are as ever.
    shot = [Shot(0, 300)]
    said = "NVML's 5.00 GiB free on the whole card, less 0.75 GiB counted for this process's CUDA"
    for behind in (False, True):
        monkeypatch.setattr(memory, "paravirtual", lambda behind=behind: behind)
        with pytest.raises(PlanError) as refused:
            planner.plan(memory.early(5 * GIB), HD, shot, Pins(), None, recorded=Plan(8))
        words = str(refused.value)
        assert said in words and "driver" not in words, words
        # Quitting programs gives the shape its memory back: an early reading cannot tell otherwise
        # (shot_plan._freeable), the driver's figure unread.
        assert refused.value.shortfall is not None and refused.value.shortfall.freeable
        # What programs hold by NVML is the most quitting them gives back (memory.card_state):
        # the shape 18 to 21 GiB short with 1 GiB held is out of their reach, with 40 GiB not.
        for used, cure in ((None, True), (GIB, False), (40 * GIB, True)):
            with pytest.raises(PlanError) as cured:
                planner.plan(memory.early(5 * GIB, used), HD, shot, Pins(), None, recorded=Plan(8))
            assert cured.value.shortfall is not None
            assert cured.value.shortfall.freeable is cure, (behind, used)
    # The plan's own lines, were they said, name them too (_card_said).
    for behind in (False, True):
        monkeypatch.setattr(memory, "paravirtual", lambda behind=behind: behind)
        made, peaks = planner.plan(memory.early(96 * GIB), HD, shot, Pins(), None, recorded=Plan(8))
        first = planner.said(made, peaks, "a card", HD, shot, Pins())[0]
        assert "NVML's 96.00 GiB free on the whole card, less 0.75 GiB" in first
        assert "driver reports" not in first
    # A run's own reading keeps the words it had.
    with pytest.raises(PlanError) as later:
        planner.plan(Reading(5 * GIB, None, 5 * GIB, False), HD, shot, Pins(), None)
    assert "the driver reports 5.00 GiB free" in str(later.value)


def test_recorded_tiles_without_a_grid_refused() -> None:
    # A record's tiles are whole numbers, the overlap under the side (Plan.recorded); the tiled
    # passes lay a grid of multiples of 8 px alone (tilegrid.latent_tiles), and the planner
    # counts a tile's peak over that grid: a record with other tiles was edited, and is refused
    # as one that seedvr2x never recorded, on a resume's check and where its plan is made again.
    for tiles in (Tiles(1001), Tiles(1024, 60)):
        said = f"tiles of {tiles.size} px sharing {tiles.overlap}"
        for recorded in (Plan(12, encode_tiles=tiles), Plan(12, None, tiles)):
            name = "encode_tiles" if recorded.encode_tiles else "decode_tiles"
            match = f"not a plan as seedvr2x records one: its {name}: {said}: a tile's side"
            with pytest.raises(PlanError, match=match):
                planner.check_recorded(recorded, shots(12), HD)
            with pytest.raises(PlanError, match=match):
                planner.plan(our_free(24), HD, shots(12), Pins(), None, recorded=recorded)


def said_line(lines: list[str], name: str) -> str:
    """The one line of the plan's lines (planner.said) that opens with `name`, "plan: <name>: "."""
    found = [line for line in lines if line.startswith(f"plan: {name}: ")]
    assert len(found) == 1, (name, lines)
    return found[0]


def test_the_longest_window_run_sets_the_peak() -> None:
    # A shot of 80 latents under a cap of 76 runs two windows of 41: the DiT holds 41 latents,
    # not 76, and no block is swapped for it. Its peak is that window's: nothing of the shot's
    # 39 other latents is on the GPU beside it (shot_pass.sample_windows).
    free = our_free(96)
    made, predicted = planner.plan(free, HD, shots(80), Pins(window=76), None)
    assert made == Plan(76)
    assert predicted.window == 41 and predicted.dit == peaks.dit_peak(OURS_1080, 41)
    assert shot_layout(shots(80)[0].frames, 76) == [(0, 41), (39, 80)]
    # So with its job: the cap sought is the largest whose longest window fits, which a shot's
    # length may leave shorter than it. A 123-latent shot on a 24 GB card, every block swapped: a
    # cap of 42, the longest that fits there, lays it out in 4 windows of 33 at most, and the
    # fewest blocks that hold 33 latents are swapped, 25. The plan's cap is those 33, which lay
    # the shot out the same, not a 42 the job never runs. (The lean attention's figures: the
    # shot of 76 latents that stood here now runs two windows of 39, and 123 is the first length
    # to lay out in 4 windows of fewer than the 42 that fit.)
    made, predicted = planner.plan(our_free(24), HD, shots(123), Pins(), None)
    assert (made.window, predicted.window, made.swapped) == (33, 33, 25)
    assert in_one_window(our_free(24), HD, Pins()) == 42
    assert shot_layout(489, 33) == shot_layout(489, 42) and len(shot_layout(489, 42)) == 4
    # A shot of 60 latents there runs two windows of 31 under every cap from 31 to 59: 31, not
    # the 59 the search stops at, with the fewest blocks that hold a window of 31, 23 of them.
    made, predicted = planner.plan(our_free(24), HD, shots(60), Pins(), None)
    assert (made.window, predicted.window, made.swapped) == (31, 31, 23)
    held = [peaks.dit_peak(OURS_1080, 31, blocks, vae_in_dit=False) for blocks in (22, 23)]
    assert held[1] <= our_free(24) - MARGIN < held[0] and predicted.dit == held[1]
    assert shot_layout(237, 59) == shot_layout(237, 31) == [(0, 31), (29, 60)]
    # A shot of 400 latents on the 96 GB card, no block swapped: 3 windows of 135 at most, under
    # every cap from 135 to 200, which 2 windows of 201 don't fit (160 latents are the most).
    made, predicted = planner.plan(our_free(96), HD, shots(400), Pins(swap_blocks=0), None)
    assert (made.window, predicted.window) == (135, 135)
    assert shot_layout(1597, 135) == shot_layout(1597, 200) and len(shot_layout(1597, 200)) == 3
    assert len(shot_layout(1597, 201)) == 2
    assert in_one_window(our_free(96), HD, Pins(swap_blocks=0)) == 160
    # A split shot's windows can be 4 latents long, under the 5 a cap is at least: a shot of 6
    # latents where 5 fit and not 6 runs two windows of 4, and its plan's cap is 5. At 4K, where
    # the DiT's smallest window is the size's floor (at 1080p the VAE's is, with its room).
    free = peaks.dit_peak(OURS_4K, 5, 36, vae_in_dit=False) + MARGIN
    made, predicted = planner.plan(free, UHD, shots(6, 2), Pins(), None)
    assert (made.window, predicted.window) == (5, 4)
    assert shot_layout(21, 5) == [(0, 4), (2, 6)]
    planner.check_recorded(made, shots(6, 2), UHD)
    free = our_free(96)
    # The pin is the cap exactly, even above the longest shot, each shot then one window.
    made, predicted = planner.plan(free, HD, shots(3, 40, 2), Pins(window=60), None)
    assert (made.window, predicted.window) == (60, 40)
    # A job whose shots are all shorter than the smallest cap has its longest shot's length.
    made, predicted = planner.plan(free, HD, shots(3, 2), Pins(), None)
    assert (made.window, predicted.window) == (3, 3)
    made, predicted = planner.plan(free, HD, shots(3, 2), Pins(window=5), None)
    assert (made.window, predicted.window) == (5, 3)


def test_a_film_is_planned_at_once() -> None:
    # Thousands of shots, windows of hundreds of latents: well under a second, the caps bisected
    # by their tokens before the model's window functions are asked. The bound is 10 s, for a
    # machine busy with something else: what it must catch is the bisection lost, not a slow day.
    lengths = [1 + (7919 * index) % 700 for index in range(3000)]
    film = shots(*lengths)
    for card in (12, 24, 96):
        started = time.perf_counter()
        made, predicted = planner.plan(our_free(card), HD, film, Pins(), 64 * GIB)
        lines = planner.said(made, predicted, "a card", HD, film, Pins())
        assert time.perf_counter() - started < 10
        assert predicted.window == made.window and len(lines) >= 9  # the planner's own cap
    # The 189 latents the card holds of a shot in one window, however long the film's longest
    # shot, 700 latents: nothing of it is on the GPU beside one of its windows.
    made, predicted = planner.plan(our_free(96), HD, film, Pins(), None)
    assert (made.window, made.swapped, in_one_window(our_free(96), HD, Pins())) == (189, 36, 189)
    assert predicted.dit == peaks.dit_peak(OURS_1080, 189, 36, vae_in_dit=False)


def test_a_window_is_all_the_phase_holds_of_a_shot() -> None:
    # The DiT's phase holds nothing of a shot beside the window it runs, the shot's noise drawn
    # in system memory (shot_pass.sample_windows): its peak is its longest window's, whatever the
    # shot's length.
    # 4K on the 96 GB card, one shot of 601 frames, 151 latents: windows of 40 with no block
    # swapped (the lean attention's: it was 21 with 18 swapped), 93.35 GiB, 0.32 under the budget.
    free, film = our_free(96), shots(151)
    made, predicted = planner.plan(free, UHD, film, Pins(), None)
    assert bare(made) == Plan(40, None, Tiles(2048), 0, vae_in_dit=True)
    assert predicted.dit == peaks.dit_peak(OURS_4K, 40, 0, True) <= free - MARGIN
    assert (round(gib(predicted.dit), 2), round(gib(free - MARGIN - predicted.dit), 2)) == (
        93.35,
        0.32,
    )
    assert peaks.dit_peak(OURS_4K, 41, 0, True) > free - MARGIN
    # The same shot on a card emulated at 24 GB, 22.51 GiB free: windows of 10 latents with 34
    # blocks swapped (on the lean attention; the plain one's were 5, the smallest, every block
    # swapped), 21.57 GiB, 0.34 under the budget; its VAE passes in the tiles of a shot of 10
    # latents, and the encode, 20.38 GiB, which now bounds the plan (the DiT's phase did).
    free = round(22.51 * GIB)
    made, predicted = planner.plan(free, UHD, film, Pins(), None)
    assert bare(made) == Plan(10, Tiles(1408), Tiles(1088), 34, 2, 0, vae_in_dit=False)
    assert predicted.dit == peaks.dit_peak(OURS_4K, 10, 34, vae_in_dit=False)
    assert (round(gib(predicted.dit), 2), round(gib(free - MARGIN - predicted.dit), 2)) == (
        21.57,
        0.34,
    )
    assert (predicted.bound, round(gib(predicted.encode), 2)) == ("encode", 20.38)
    assert fits(made, free, UHD, film, None) is None
    # 1080p on a 24 GB card, one shot of 1,197 frames, 300 latents: windows of 40 with 33
    # blocks swapped, 21.92 GiB, under the budget's 21.97.
    free, film = our_free(24), shots(300)
    made, predicted = planner.plan(free, HD, film, Pins(), None)
    assert bare(made) == Plan(40, Tiles(1856), Tiles(1088), 33, 1, 0, vae_in_dit=False)
    assert predicted.dit == peaks.dit_peak(OURS_1080, 40, 33, vae_in_dit=False) <= free - MARGIN
    assert (round(gib(predicted.dit), 2), round(gib(free - MARGIN), 2)) == (21.92, 21.97)
    # Where the DiT's phase bounds a plan by a window, its line says no more.
    assert said_line(
        planner.said(made, predicted, "a card", HD, film, Pins()), "predicted peaks"
    ) == (
        "plan: predicted peaks: encode 20.55 GiB, DiT 21.92 GiB, decode 20.17 GiB; the DiT's phase"
        " bounds the plan, 0.05 GiB under the budget"
    )
    # A job of many short shots, each in one window: the longest's.
    for card, target, padded, film, plan in (
        (24, HD, OURS_1080, shots(*range(1, 18)), Plan(17, Tiles(1856), Tiles(1088), 7, 1, 0)),
        (96, UHD, OURS_4K, shots(*range(1, 20)), Plan(19, None, Tiles(2048), vae_in_dit=True)),
    ):
        plan = replace(plan, vae_in_dit=plan.swapped == 0)
        made, predicted = planner.plan(our_free(card), target, film, Pins(), None)
        assert bare(made) == plan
        assert predicted.dit == peaks.dit_peak(
            padded, predicted.window, made.swapped, made.vae_in_dit
        )


def test_tile_pins() -> None:
    # A tile pin is the tile exactly, or refused, naming the largest tile that fits; 0 pins an
    # untiled pass, refused likewise. A pass pinned is held to the room the planner's own is.
    free = our_free(24)
    made, _ = planner.plan(free, UHD, shots(5), Pins(), None)
    assert (made.encode_tiles, made.decode_tiles) == (Tiles(1472), Tiles(1088))
    pins = Pins(encode_tile=1024, decode_tile=1024)
    made, predicted = planner.plan(free, UHD, shots(5), pins, None)
    assert (made.encode_tiles, made.decode_tiles) == (Tiles(1024), Tiles(1024))
    assert refused(free, UHD, shots(5), Pins(decode_tile=2048), None) == (
        "--decode-tile 2048: the decode in tiles of 2048 px takes 65.73 GiB of free VRAM, the"
        " DiT off the GPU, 1.25 of them the room a VAE pass is held to, and the driver reports"
        " 22.57 GiB free: the largest tile that fits is --decode-tile 1088"
    )
    assert refused(free, UHD, shots(5), Pins(encode_tile=0), None) == (
        "--encode-tile 0: the encode untiled takes 78.05 GiB of free VRAM, the DiT off the GPU,"
        " 1.25 of them the room a VAE pass is held to, and the driver reports 22.57 GiB free: the"
        " largest tile that fits is --encode-tile 1472"
    )
    # At 1080p an untiled encode fits a 24 GB card by its peak, 21.38 GiB of free VRAM with the
    # margin, and is refused for its room.
    assert round(gib(peaks.encode_peak(OURS_1080, None, 0) + MARGIN), 2) == 21.38
    assert refused(free, HD, shots(12), Pins(encode_tile=0), None) == (
        "--encode-tile 0: the encode untiled takes 22.63 GiB of free VRAM, the DiT off the GPU,"
        " 1.25 of them the room a VAE pass is held to, and the driver reports 22.57 GiB free: the"
        " largest tile that fits is --encode-tile 1856"
    )
    # The planner chooses no tile above 2048 px, the largest measured: on the 96 GB card a 4K
    # decode is in tiles of 2048 px. A larger one pinned is taken where it fits by the fit, and a
    # refusal names none. The log is given a warning for each such pass, the encode's then the
    # decode's (planner.beyond_measured), for a plan taken up from a record as for one made, and
    # none for the planner's own tiles.
    made, _ = planner.plan(our_free(96), UHD, shots(5), Pins(), None)
    assert bare(made) == Plan(5, None, Tiles(2048))
    assert planner.beyond_measured(made) == []
    made, _ = planner.plan(our_free(96), UHD, shots(5), Pins(encode_tile=2304), None)
    assert bare(made) == Plan(5, Tiles(2304), Tiles(2048))
    assert planner.beyond_measured(made) == [
        "the encode in tiles of 2304 px: its predicted peak is the fit's, beyond the sizes"
        " measured (up to 2048 px), and the pass may take more"
    ]
    recorded = Plan(5, Tiles(2112), Tiles(2304))
    again, _ = planner.plan(our_free(96), UHD, shots(5), Pins(), None, recorded=recorded)
    assert (again.encode_tiles, again.decode_tiles) == (Tiles(2112), Tiles(2304))
    assert planner.beyond_measured(again) == [
        "the encode in tiles of 2112 px: its predicted peak is the fit's, beyond the sizes"
        " measured (up to 2048 px), and the pass may take more",
        "the decode in tiles of 2304 px: its predicted peak is the fit's, beyond the sizes"
        " measured (up to 2048 px), and the pass may take more",
    ]
    assert planner.beyond_measured(Plan(5, Tiles(2048), Tiles(512))) == []
    assert refused(our_free(48), UHD, shots(5), Pins(decode_tile=2304), None) == (
        "--decode-tile 2304: the decode in tiles of 2304 px takes 85.09 GiB of free VRAM, the"
        " DiT off the GPU, 1.25 of them the room a VAE pass is held to, and the driver reports"
        " 46.57 GiB free: the largest tile that fits is --decode-tile 1664"
    )
    # A tile pinned where the untiled pass would fit is taken as pinned; and 0 where it fits.
    pins = Pins(encode_tile=0, decode_tile=1024)
    made, predicted = planner.plan(our_free(96), HD, shots(40), pins, None)
    assert bare(made) == Plan(40, None, Tiles(1024))
    assert predicted.scratch == peaks.scratch(OURS_1080, 157)
    # The largest tile under the padded frame's longer side is one: 1856 px at 1080p.
    made, _ = planner.plan(our_free(96), HD, shots(40), Pins(decode_tile=1856), None)
    grid = peaks.grid_shapes  # pyright: ignore[reportPrivateUsage]
    assert made.decode_tiles == Tiles(1856) and grid(OURS_1080, Tiles(1856)) == (1, 2)


def test_tile_pin_that_covers_the_frame_refused() -> None:
    # DESIGN.md, Pipeline, per shot, step 3: "A tile that covers the whole padded frame is no
    # tiling: such a pin is refused, naming 0, the untiled pass, and the planner never plans
    # one." Whatever the memory, on the 96 GB card, where the untiled pass it names fits; the
    # encode's first.
    free = our_free(96)
    pins = Pins(encode_tile=1920, decode_tile=4096)
    assert refused(free, HD, shots(40), pins, 64 * GIB) == (
        "--encode-tile 1920: a tile of 1920 px covers the whole padded frame, 1920x1104:"
        " --encode-tile 0 pins an untiled encode"
    )
    assert refused(free, HD, shots(40), Pins(encode_tile=0, decode_tile=4096), None) == (
        "--decode-tile 4096: a tile of 4096 px covers the whole padded frame, 1920x1104:"
        " --decode-tile 0 pins an untiled decode"
    )
    assert peaks.covers(OURS_1080, Tiles(1920)) and not peaks.covers(OURS_1080, Tiles(1856))
    made, predicted = planner.plan(free, HD, shots(40), Pins(encode_tile=0, decode_tile=0), None)
    assert made == Plan(40) and predicted.scratch == 0
    made, _ = planner.plan(free, HD, shots(40), Pins(encode_tile=1856, decode_tile=1856), None)
    assert bare(made) == Plan(40, Tiles(1856), Tiles(1856))
    # Before the size's own refusal, where it doesn't fit, and after --swap-blocks' range.
    assert refused(numz_free(16), UHD, shots(40), Pins(decode_tile=3840), None) == (
        "--decode-tile 3840: a tile of 3840 px covers the whole padded frame, 3840x2192:"
        " --decode-tile 0 pins an untiled decode"
    )
    pins = Pins(swap_blocks=37, decode_tile=3840)
    assert refused(free, UHD, shots(40), pins, None).startswith("--swap-blocks 37: the DiT has")
    # On a frame too small for a tile, every tile covers it: 0 alone is taken, where the size
    # fits, by its untiled encode and the room it is held to (test_floor).
    tiny = peaks.encode_peak((128, 128), None, 0) + peaks.VAE_ROOM + MARGIN
    for pin in (512, 1024):
        assert refused(tiny, (96, 128), shots(6), Pins(decode_tile=pin), None) == (
            f"--decode-tile {pin}: a tile of {pin} px covers the whole padded frame, 128x128:"
            " --decode-tile 0 pins an untiled decode"
        )
    made, _ = planner.plan(tiny, (96, 128), shots(6), Pins(encode_tile=0, decode_tile=0), None)
    assert (made.encode_tiles, made.decode_tiles) == (None, None)
    # A recorded plan with such tiles is none that seedvr2x records: its job, resumed, is
    # refused, by the planner and by the check the command line names the manifest with.
    for recorded, said in (
        (Plan(40, Tiles(1920), Tiles(4096)), "its encode_tiles, of 1920 px"),
        (Plan(40, None, Tiles(4096)), "its decode_tiles, of 4096 px"),
    ):
        untiled = "encode" if recorded.encode_tiles else "decode"
        whole = (
            f"not a plan as seedvr2x records one: {said}, with frames padded to 1920x1104: a"
            f" tile that covers the whole frame is no tiling, and an untiled {untiled} records none"
        )
        assert refused(free, HD, shots(40), Pins(), None, recorded=recorded) == whole
        with pytest.raises(PlanError) as error:
            planner.check_recorded(recorded, shots(40), HD)
        assert str(error.value) == whole
    # 4K is padded to 2192x3840: tiles of 3776 px leave it a grid, tiles of 3840 cover it.
    planner.check_recorded(Plan(5, Tiles(3776), Tiles(1024)), shots(40), UHD)
    with pytest.raises(PlanError, match="with frames padded to 3840x2192"):
        planner.check_recorded(Plan(5, Tiles(3840)), shots(40), UHD)
    # Nor has such a plan peaks, or a sentence of fits: no pass is run in such tiles.
    with pytest.raises(ValueError, match="tiles of 1920 px over frames padded to 1920x1104"):
        fits(Plan(12, Tiles(1920)), free, HD, shots(12), None)


def test_host_memory() -> None:
    # The system memory the plan takes: the process's own 2.4 GiB, a pinned copy of each part of
    # the weights that leaves the GPU in some phase, what the run holds of a shot at the moment
    # it holds the most (test_host_moments), and what a tiled shot's scratch may take
    # of it, the plan's allowance (test_scratch_allowance): here the longest shot's scratch
    # whole. The first three over what is available: refused, naming each.
    free = our_free(24)
    made, predicted = planner.plan(free, HD, shots(76), Pins(), 64 * GIB)
    assert bare(made) == Plan(39, Tiles(1856), Tiles(1088), 32, 1, 0, vae_in_dit=False)
    kept = peaks.scratch(OURS_1080, 301)
    assert predicted.scratch == made.scratch_allowance == kept == 301 * 12_718_080
    # Every block leaves in the VAE's phases, the DiT's other parameters with them, and the VAE
    # in the DiT's phase: each part's bytes rounded up to a page, 15.81 GiB (measured, milestone
    # 3).
    pinned = 36 * paged(BLOCK) + paged(peaks.DIT_OTHER) + paged(peaks.VAE_WEIGHTS)
    # The shot's 76 latents in windows of 39: its latent and its noise beside a window's output,
    # 191 latents' worth of 1,059,840 bytes at 1080p: 0.19 GiB.
    held = 191 * 1_059_840
    assert (predicted.shot, predicted.moment, predicted.holder) == (held, "windows", 76)
    assert predicted.host == shot_plan.HOST_RESIDENT + pinned + held + kept
    assert f"{gib(pinned):.2f}" == "15.81" and pinned - 36 * BLOCK < 501_500_000 + 166_500_000
    # The refusal says each part as the plan's line does, to the nearest hundredth, and its
    # total is their sum; the shot's part with its length and the moment that takes it. Of the
    # plan that takes the least of it at the card's window: the VAE's weights kept on the GPU
    # in the DiT's phase, one block more swapped for them (test_the_vae_stays_where_the_host_is_
    # short), then the DiT's 3 other blocks kept on the GPU in the VAE's phases, their tiles
    # shrunk for it, the 33 swapped alone pinned, 13.93 GiB, which the refusal says
    # (test_on_a_short_host_the_tiles_shrink_before_the_window). No shorter window's plan fits
    # 3 GiB either; 4 GiB run windows of 13 with 3 blocks swapped, the VAE kept, in tiles of 640
    # and 512 px with 33 of the DiT's blocks beside each; 8 GiB, refused until then, run windows
    # of 21 (that test).
    four, _ = planner.plan(free, HD, shots(76), Pins(), 4 * GIB)
    assert bare(four) == Plan(13, Tiles(640), Tiles(512), 3, 33, 33)
    assert refused(free, HD, shots(76), Pins(), 3 * GIB) == (
        "the plan takes 16.52 GiB of system memory (2.40 for the process, 13.93 pinned for 33"
        " blocks that leave the GPU, 0.19 for the longest shot, 301 frames (76 latents), at its"
        " longest window, its latent and its noise beside the window's output) and 3.00 GiB are"
        " available, 13.52 GiB short, its VAE's tiles shrunk to keep the DiT on the GPU: the"
        " encode in tiles of 1728 px, with 3 of the DiT's blocks on the GPU; the decode in tiles"
        " of 1024 px, with the DiT's 3 unswapped blocks and its 2 slots on the GPU"
    )
    said = planner.said(made, predicted, "a card", HD, shots(76), Pins())
    assert said_line(said, "system memory") == (
        "plan: system memory: 21.97 GiB in all; 15.81 GiB pinned for 36 blocks, the DiT's other"
        " parameters and the VAE; 0.19 GiB for the longest shot at the most, at its longest window"
    )
    # A job is refused for those three alone: with no more than they take, its scratch is
    # allowed nothing, and every shot's goes to its file.
    least = shot_plan.HOST_RESIDENT + pinned + held
    scarce, taken = planner.plan(free, HD, shots(76), Pins(), least)
    assert scarce == bare(made) and (taken.host, taken.scratch) == (least, kept)
    # One byte less than a plan takes, with --swap-blocks and --window pinned, which are taken
    # as they are, and its tiles too, which no host shrinks then, is refused: 18.3813 GiB
    # taken, said 18.40 by its parts, where a byte less is there: what is there is said a
    # hundredth under what is taken, which the two rounded alike would hide. It names the
    # longest window whose plan fits, the --window that plans there.
    assert f"{gib(least):.4f}" == "18.4030"
    tiled = Pins(encode_tile=1856, decode_tile=1088)
    assert refused(free, HD, shots(76), replace(tiled, swap_blocks=32, window=39), least - 1) == (
        "the plan takes 18.40 GiB of system memory (2.40 for the process, 15.81 pinned for 36"
        " blocks, the DiT's other parameters and the VAE's weights that leave the GPU, 0.19 for"
        " the longest shot, 301 frames (76 latents), at its longest window, its latent and its"
        " noise beside the window's output) and 18.39 GiB are available, 0.01 GiB short: the"
        " longest window whose plan fits it is --window 27"
    )
    # Without --window, that window is the planner's own: under it the VAE's weights stay
    # beside the 32 blocks' windows, unpinned (test_the_planner_takes_the_longest_window_the_
    # host_holds).
    kept, taken = planner.plan(free, HD, shots(76), replace(tiled, swap_blocks=32), least - 1)
    assert (kept.window, kept.swapped, kept.vae_in_dit) == (27, 32, True)
    assert taken.host <= least - 1 and taken.shortened == shot_plan.Shortened(39, 1840, 1839)
    pinned_so = planner.plan(
        free, HD, shots(76), replace(tiled, swap_blocks=32, window=27), least - 1
    )
    assert pinned_so == (kept, replace(taken, shortened=None))
    assert planner.plan(free, HD, shots(76), Pins(), None) == (made, predicted)  # not checked
    # Nothing leaves the GPU, nothing tiled: the process and the shot, 40 latents in one window,
    # 120 latents' worth through it, 0.12 GiB.
    made, predicted = planner.plan(our_free(96), HD, shots(40), Pins(), 4 * GIB)
    assert made == Plan(40) and predicted.scratch == 0
    assert predicted.host == shot_plan.HOST_RESIDENT + 120 * 1_059_840
    assert refused(our_free(96), HD, shots(40), Pins(), 2 * GIB) == (
        "the plan takes 2.52 GiB of system memory (2.40 for the process, 0.12 for the longest"
        " shot, 157 frames (40 latents), at its longest window, its latent and its noise beside"
        " the window's output) and 2.00 GiB are available, 0.52 GiB short"
    )
    # A long tiled shot's scratch counts up to its allowance: 4K, 601 frames, 28.27 GiB, over
    # half of what a host with 56 GiB available has left of the pinned copies and of the shot,
    # 151 latents in windows of 40, 342 latents' worth of 4,208,640 bytes, 1.34 GiB: 27.33, so
    # that it goes whole to its memory-mapped file (tiling.Scratch). Nothing moves between the
    # phases and nothing is swapped on the lean attention, so nothing is pinned. (The host was
    # 64 GiB, half of which, 31.3, passes the scratch, when the 18 swapped blocks and the VAE
    # were pinned; 56 GiB is the same case on the same side of the boundary.)
    made, predicted = planner.plan(our_free(96), UHD, shots(151), Pins(), 56 * GIB)
    assert bare(made) == Plan(40, None, Tiles(2048), 0, vae_in_dit=True)
    pinned = 0
    held = (2 * 151 + 40) * 4_208_640
    left = 56 * GIB - pinned - held
    assert predicted.scratch == 601 * 50_503_680 > made.scratch_allowance == left // 2
    assert (f"{gib(held):.2f}", f"{gib(left // 2):.2f}") == ("1.34", "27.33")
    assert predicted.host == shot_plan.HOST_RESIDENT + pinned + held + made.scratch_allowance
    # Two swapped blocks save nothing in the DiT's phase, their slots there, and wait in pinned
    # copies all the same, with the 13 more the untiled decode sends away, which frees their
    # slots: 21 blocks stay beside it, as with none swapped.
    made, predicted = planner.plan(our_free(48), HD, shots(12), Pins(swap_blocks=2), None)
    assert made == Plan(12, swapped=2, resident_decode=21)
    assert predicted.host == shot_plan.HOST_RESIDENT + 15 * paged(BLOCK) + 36 * 1_059_840
    unswapped, _ = planner.plan(our_free(48), HD, shots(12), Pins(swap_blocks=0), None)
    assert unswapped == Plan(12, resident_decode=21)


def test_the_vae_stays_where_the_host_is_short() -> None:
    # DESIGN.md has the VAE's weights leave the DiT's phase before a block is swapped (Memory
    # planner, Choices, 1), and this in the paragraph after 3. 1080p on a 24 GB card, a shot of
    # 301 frames: 32 blocks swapped and the VAE away. Away, its weights are pinned, 0.47 GiB of
    # the 15.81 pinned there, every block leaving in the VAE's phases anyway. Where that plan
    # takes more system memory than is available, the same shape with one block more swapped
    # and the VAE staying takes 0.46 GiB less, and is the plan (shot_plan._vae_kept): no bit
    # moves, nor any plan on a host that holds the first.
    free, job = our_free(24), (HD, shots(76), Pins())
    pinned = 36 * paged(BLOCK) + paged(peaks.DIT_OTHER) + paged(peaks.VAE_WEIGHTS)
    held = 191 * 1_059_840
    first = shot_plan.HOST_RESIDENT + pinned + held
    made, predicted = planner.plan(free, *job, first)
    assert (made.swapped, made.vae_in_dit, predicted.host) == (32, False, first)
    kept, predicted = planner.plan(free, *job, first - 1)
    assert kept == replace(
        made, swapped=33, vae_in_dit=True, scratch_allowance=first - 1 - least_of(kept)
    )
    assert predicted.host == least_of(kept) + kept.scratch_allowance == first - 1
    # 33 is the fewest that hold the window with the VAE there, and the DiT's peak fits.
    assert predicted.dit == peaks.dit_peak(OURS_1080, 39, 33) <= free - MARGIN
    assert peaks.dit_peak(OURS_1080, 39, 32) > free - MARGIN
    said = planner.said(kept, predicted, "a card", *job)
    assert said_line(said, "blocks swapped") == (
        "plan: blocks swapped: 33 of the DiT's 36, more than the window needs: the VAE stays on"
        " the GPU, which spares system memory: slower, not worse; --swap-blocks N"
    )
    assert said_line(said, "the VAE") == (
        "plan: the VAE: on the GPU in the DiT's phase, the window fitting beside it"
    )
    assert said_line(said, "the DiT in the VAE's passes") == (
        "plan: the DiT in the VAE's passes: which model stays on the GPU in which phase, as much"
        " of the DiT as fits beside each: in the encode, 1 of the DiT's blocks on the GPU; in"
        " the decode, the DiT off the GPU"
    )
    assert said_line(said, "system memory") == (
        "plan: system memory: 18.41 GiB in all; 15.35 GiB pinned for 36 blocks and the DiT's other"
        " parameters; 0.19 GiB for the longest shot at the most, at its longest window"
    )
    # (The line's total is its parts' as said, the scratch's allowance, 0.47 GiB, among them.)
    # Down to what that plan takes, 17.94 GiB; a byte under, its window pinned or not, the DiT
    # stays on the GPU in the VAE's phases and their tiles shrink, the next step
    # (test_on_a_short_host_the_tiles_shrink_before_the_window).
    least = least_of(kept)
    assert f"{gib(least):.2f}" == "17.94" and first - least == paged(peaks.VAE_WEIGHTS)
    scarce, predicted = planner.plan(free, *job, least)
    assert scarce == replace(kept, scratch_allowance=0) and predicted.host == least
    for pins in (Pins(), Pins(window=39)):
        shrunk, predicted = planner.plan(free, HD, shots(76), pins, least - 1)
        assert (shrunk.encode_tiles, shrunk.decode_tiles) == (Tiles(1856), Tiles(1024))
        assert (shrunk.swapped, shrunk.vae_in_dit) == (33, True)
        assert predicted.shrunk == planner.Shrunk(Tiles(1856), Tiles(1088), 1794, 1793)
    # --swap-blocks is taken exactly, the VAE's place around it as on any host; the tiles still
    # shrink around it.
    pins = Pins(swap_blocks=32, window=39)
    shrunk, predicted = planner.plan(free, HD, shots(76), pins, first - 1)
    assert (shrunk.swapped, shrunk.vae_in_dit, shrunk.decode_tiles) == (32, False, Tiles(1024))
    assert predicted.shrunk == planner.Shrunk(Tiles(1856), Tiles(1088), 1840, 1839)
    # A resume plans the weights' places the same way around its recorded shape.
    shape = Plan.recorded(made.record())
    again, _ = planner.plan(free, *job, first - 1, recorded=shape)
    assert bare(again) == bare(kept)
    assert bare(planner.plan(free, *job, first, recorded=shape)[0]) == bare(made)
    # A plan the GPU's memory refuses is refused for it, and one whose window no count of
    # swapped blocks holds with the VAE there keeps the VAE away, refused for the system
    # memory it takes: 4K's smallest window on a card of 11.8 GiB free, every block swapped (the
    # 24 GB card's, 22.6 GiB, holds more than the smallest on the lean attention).
    assert "of free VRAM" in refused(round(6.5 * GIB), *job, 8 * GIB)
    small = round(11.8 * GIB)
    made, predicted = planner.plan(small, UHD, shots(5), Pins(), None)
    assert (made.swapped, made.vae_in_dit) == (36, False)
    assert refused(
        small, UHD, shots(5), Pins(), predicted.host - made.scratch_allowance - 1
    ).startswith(
        "the plan takes 18.27 GiB of system memory (2.40 for the process, 15.81 pinned for 36"
        " blocks, the DiT's other parameters and the VAE's weights that leave the GPU, 0.06 for"
    )


def least_of(made: Plan) -> int:
    """What a plan takes of system memory without its scratch's allowance, for the 301-frame
    shot of test_the_vae_stays_where_the_host_is_short, in bytes."""
    return host_need(replace(made, scratch_allowance=0), HD, shots(76))


def one_cap_per_layout(film: list[Shot], caps: range) -> list[int]:
    """The first of `caps` (window caps, in latents) for each distinct layout of the film's DiT
    windows under them (shot_layout, shot by shot), in order.

    Caps under which every shot runs the same windows plan the same: the planner reads a pinned
    cap only through the windows the shots run under it (shot_plan._longest, the shots' layouts
    for the host's figure), the cap itself echoed in the plan and its refusal alone, and its own
    walk down the caps skips one whose longest window is the last one tried's ("plans the same",
    shot_plan._window_that_runs), a coarser grouping than this one. Planned cap by cap, the
    grid's 2,685 calls (2026-11-07) gave the outcome the test asserts the same within each of
    their 764 distinct layouts (2026-11-08): one cap a layout proves of every window what every
    cap did, in 764 calls, each refused pin costing the planner a walk down every shorter window
    to word its advice."""
    first: dict[tuple[tuple[tuple[int, int], ...], ...], int] = {}
    for cap in caps:
        layout = tuple(tuple(shot_layout(shot.frames, cap)) for shot in film)
        first.setdefault(layout, cap)
    return list(first.values())


def test_the_planner_takes_the_longest_window_the_host_holds() -> None:
    # The choices are made for the GPU's memory, the longest window first, and the system memory
    # is checked after them: more free VRAM turned a job that was planned into one refused for
    # its pinned blocks. 1080p, a shot of 301 frames, 16 GiB of system memory available:
    # planned with 26, 30 and 34 GiB of free VRAM (46.5, 60 and 70 before the lean attention),
    # in two windows of 39 latents, and with 55 (94.2), in one window with nothing swapped; with
    # 40 (80), one window of 76 with 35 blocks pinned, 17.40 GiB, was refused. The planner takes
    # by itself the longest window whose plan the host holds too (DESIGN.md, Memory planner,
    # Choices; design, 2026-10-12): the plan --window 39 makes, and its lines say which window
    # the card alone holds, and why not.
    job, host = (HD, shots(76)), 16 * GIB
    for free in (26, 30, 34):
        made, peaks = planner.plan(round(free * GIB), *job, Pins(), host, split=True)
        assert made.window == 39 and peaks.shortened is None
    alone, _ = planner.plan(round(55 * GIB), *job, Pins(), host, split=True)
    assert (alone.window, alone.swapped) == (76, 0)
    made, peaks = planner.plan(40 * GIB, *job, Pins(), host, split=True)
    assert (made.window, peaks.window, made.swapped) == (39, 39, 0) and peaks.host <= host
    assert peaks.shortened == shot_plan.Shortened(76, 1740, 1600)
    pinned = planner.plan(40 * GIB, *job, Pins(window=39), host, split=True)
    assert pinned == (made, replace(peaks, shortened=None))
    assert said_line(planner.said(made, peaks, "a card", *job, Pins()), "windows") == (
        "plan: windows: 39 latents at most (153 frames), shortened from 76 latents for the host's"
        " memory (their plan 17.40 GiB of 16.00 available): 0 of 1 shot in one window, 1 join; a"
        " longer window, fewer joins; --window N caps them"
    )
    said = planner.said(pinned[0], pinned[1], "a card", *job, Pins(window=39))
    assert said_line(said, "windows") == (
        "plan: windows: 39 latents at most (153 frames), pinned by --window: 0 of 1 shot in one"
        " window, 1 join; a longer window, fewer joins; --window N caps them"
    )
    # The host's memory unread, nothing is shortened: the card's own window.
    unread, peaks = planner.plan(40 * GIB, *job, Pins(), None, split=True)
    assert (unread.window, unread.swapped, peaks.shortened) == (76, 34, None)
    # Never under --window: the pin is refused, naming that window, the --window that plans.
    assert refused(40 * GIB, *job, Pins(window=76), host, split=True) == (
        "the plan takes 17.40 GiB of system memory (2.40 for the process, 14.77 pinned for 35"
        " blocks that leave the GPU, 0.23 for the longest shot, 301 frames (76 latents), at its"
        " longest window, its latent and its noise beside the window's output) and 16.00 GiB are"
        " available, 1.40 GiB short: the longest window whose plan fits it is --window 39"
    )
    # With 22 GiB free (30 before the lean attention) every window's plan sent the DiT away in
    # the VAE's phases, 36 blocks pinned whatever the cap, and the job was refused: the DiT kept
    # on the GPU there, their tiles shrunk, before the window shortens (test_on_a_short_host_the_
    # tiles_shrink_before_the_window), it runs windows of 27, the encode in tiles of 1600 px for
    # the card's 1856 and the decode in tiles of 960 for 1024.
    made, peaks = planner.plan(22 * GIB, *job, Pins(), host, split=True)
    assert (made.window, made.encode_tiles, made.decode_tiles) == (27, Tiles(1600), Tiles(960))
    assert peaks.shortened == shot_plan.Shortened(39, 1736, 1600)
    assert peaks.shrunk == planner.Shrunk(Tiles(1856), Tiles(1024), 1793, 1600)
    # Where no window's plan fits both, the job is refused, and nothing more is said of the
    # window: under the process's own 2.4 GiB.
    assert "and 2.00 GiB are available, " in refused(22 * GIB, *job, Pins(), 2 * GIB, split=True)
    # Nor on a resume, whose window is the recorded one; nor for the GPU's memory.
    shape = Plan.recorded(planner.plan(40 * GIB, *job, Pins(), None, split=True)[0].record())
    again = refused(40 * GIB, *job, Pins(), host, split=True, recorded=shape)
    assert again.endswith("and 16.00 GiB are available, 1.40 GiB short")
    assert "--window" not in refused(round(6.5 * GIB), *job, Pins(), host)
    # Over a grid of jobs, memories and hosts: a plan with a window shortened for the host is
    # the plan that window pinned makes, every longer window refused there for the system
    # memory, up to the one the card alone gives; and a job refused for the system memory has
    # no window that plans. "Every window" is every distinct layout of the film's windows, one
    # cap planned for each (one_cap_per_layout says why that proves the same).
    shortened = unheld = 0
    films = (shots(76), shots(40), shots(12, 133), shots(5))
    frees = (12.5, 15, 18.5, 24, 31, 40, 41.5)
    longest = shot_plan._longest  # pyright: ignore[reportPrivateUsage]
    for target in ((540, 960), (720, 1280), HD, UHD):
        for film in films:
            lengths = tuple(sorted({(padded_length(shot.frames) - 1) // 4 + 1 for shot in film}))
            for free in (round(count * GIB) for count in frees):
                for available in (round(5.3 * GIB), round(13.11 * GIB), 16 * GIB, 18 * GIB):
                    case = (free, target, film)

                    def plans(
                        cap: int,
                        free: int = free,
                        target: tuple[int, int] = target,
                        film: list[Shot] = film,
                        available: int = available,
                    ) -> str | None:
                        try:
                            planner.plan(
                                free, target, film, Pins(window=cap), available, split=True
                            )
                        except PlanError as error:
                            return str(error)
                        return None

                    window = planner.plan(*case, Pins(), None, split=True)[0].window
                    assert window is not None
                    try:
                        made, peaks = planner.plan(*case, Pins(), available, split=True)
                    except PlanError as error:
                        if "of system memory" in str(error):
                            said = str(error)
                            shrunk = ", its VAE's tiles shrunk to keep the DiT on the GPU: "
                            assert said.endswith(" GiB short") or shrunk in said, said
                            caps = one_cap_per_layout(film, range(5, window + 1))
                            assert all(plans(cap) is not None for cap in caps)
                            unheld += 1
                        continue
                    if peaks.shortened is None:
                        assert made.window == window
                        continue
                    assert made.window is not None and made.window < window
                    assert peaks.shortened.window == window and peaks.host <= available
                    assert peaks.shortened.taken > peaks.shortened.available
                    again = planner.plan(*case, Pins(window=made.window), available, split=True)
                    assert again == (made, replace(peaks, shortened=None))
                    for cap in one_cap_per_layout(film, range(made.window + 1, window + 1)):
                        said = plans(cap)
                        if longest(lengths, cap) == peaks.window:
                            assert said is None  # the same windows, under a larger cap
                        else:
                            assert said is not None and "of system memory" in said, (cap, said)
                    shortened += 1
    # 87 and 20 (2026-10-16): fewer jobs are refused for the system memory since the DiT stays
    # on the GPU in the VAE's phases before a window shortens. 82 and 79 on the lean attention's
    # grid of free memories.
    assert shortened > 20 and unheld >= 20


def test_on_a_short_host_the_tiles_shrink_before_the_window() -> None:
    # DESIGN.md, Memory planner, Choices (decided on 2026-10-13): where the plan takes more
    # system memory than the host has, "before the window shortens, the DiT stays on the GPU for
    # the VAE's phases and the tiles shrink, the window being the larger lever", after the VAE's
    # weights kept for the DiT's phase (test_the_vae_stays_where_the_host_is_short). The largest
    # tiles first send the most of the DiT away, each block in a pinned copy: more free VRAM
    # pinned more and shortened the window, or refused the job: refusals that smaller tiles run,
    # and windows that fell as the free VRAM grew (DESIGN.md, Memory planner, Choices; To
    # measure, The planner on a short host). One that fell, as a grid of plans made it: 1440p,
    # shots of 5 and 39 latents (17 and 152 frames there), split, 17.338 GiB of system memory.
    film, host = shots(5, 39), 18_616_624_832
    qhd = (1440, 2560)
    # 44.5 and 45.5 GiB: the lean attention moved the grid (it was 43.0 and 45.5 with windows of
    # 21), the window of 39 now being the card's from 37 GiB, and 1 GiB apart is the pair.
    below, made = round(44.5 * GIB), round(45.5 * GIB)
    planned, peaks = planner.plan(below, qhd, film, Pins(), host, split=True)
    assert (planned.window, planned.decode_tiles, planned.resident_decode) == (39, Tiles(1664), 3)
    assert peaks.shrunk is None and peaks.shortened is None
    # With 1 GiB more, the card alone takes the decode in tiles of 1728 px, which keep 5 blocks
    # of the DiT on the GPU, 14 pinned with the VAE's weights kept: 17.38 GiB, 0.05 over. The
    # window fell to 15 for 0.005 GiB of the shot's latents; the tiles shrink instead, 1664 px,
    # and the window stays.
    planned, peaks = planner.plan(made, qhd, film, Pins(), host, split=True)
    assert (planned.window, planned.swapped, planned.vae_in_dit) == (39, 14, True)
    assert (planned.encode_tiles, planned.decode_tiles) == (None, Tiles(1664))
    assert (planned.resident_encode, planned.resident_decode) == (19, 5)
    assert peaks.shortened is None and peaks.host <= host
    assert peaks.shrunk == planner.Shrunk(None, Tiles(1728), 1738, 1733)
    said = planner.said(planned, peaks, "a card", qhd, film, Pins())
    assert said_line(said, "encode") == (
        "plan: encode: untiled, the whole frame fitting the budget; --encode-tile PX tiles it"
    )
    assert said_line(said, "decode") == (
        "plan: decode: tiles of 1664 px overlapping by 64, 2 across, 1 down, shrunk from 1728 px"
        " for the host's memory (that plan 17.38 GiB of 17.33 available): a slight colour drift,"
        " which split corrects; --decode-tile PX"
    )
    # The plan --window 39 makes, which a pinned window's never shortens; nor a pass pinned in
    # its tiles: under --decode-tile 1728 the window falls to 15, as it did, the decode's tiles
    # taken as asked.
    pinned = planner.plan(made, qhd, film, Pins(window=39), host, split=True)
    assert pinned == (planned, peaks)
    planned, peaks = planner.plan(made, qhd, film, Pins(decode_tile=1728), host, split=True)
    assert (planned.window, planned.decode_tiles, peaks.shrunk) == (15, Tiles(1728), None)
    assert peaks.shortened == shot_plan.Shortened(39, 1738, 1733)
    # One of the 828: 4K, a shot of 5 frames, 13.11 GiB of system memory. With 78 GiB free the
    # decode in tiles of 2048 px keeps 28 blocks of the DiT on the GPU (15 before the one-output
    # convolution lowered its peak by 5.16 GiB), and the plan fits; with 80.5 the card alone
    # takes the encode untiled, which keeps none of the DiT, 15.92 GiB, and the job was refused,
    # no window being shorter; the encode in tiles of 2048 px keeps it all, the decode 34
    # blocks (21 before).
    film, host = shots(2), round(13.11 * GIB)
    planned, peaks = planner.plan(round(78 * GIB), UHD, film, Pins(), host)
    assert (planned.encode_tiles, planned.decode_tiles, peaks.shrunk) == (
        Tiles(2048),
        Tiles(2048),
        None,
    )
    planned, peaks = planner.plan(round(80.5 * GIB), UHD, film, Pins(), host)
    assert (planned.encode_tiles, planned.decode_tiles) == (Tiles(2048), Tiles(2048))
    assert (planned.resident_encode, planned.resident_decode, planned.swapped) == (None, 34, 0)
    assert peaks.shrunk == planner.Shrunk(None, Tiles(2048), 1550, 1311)
    # 1080p on a 24 GB card, a shot of 301 frames, 8 GiB available: refused until then, every
    # window's VAE passes in tiles of 1856 and 1088 px sending the DiT away, its 36 blocks
    # pinned (test_host_memory); windows of 21 run, the VAE away, 25 of the DiT's blocks kept on
    # the GPU in the encode and all its unswapped ones in the decode, the encode in tiles of
    # 896 px, the decode of 640 (test_the_vae_kept_first_costs_no_window).
    planned, peaks = planner.plan(our_free(24), HD, shots(76), Pins(), 8 * GIB)
    assert (planned.window, planned.swapped, planned.vae_in_dit) == (21, 11, False)
    assert (planned.encode_tiles, planned.decode_tiles) == (Tiles(896), Tiles(640))
    assert (planned.resident_encode, planned.resident_decode) == (25, None)
    assert peaks.shrunk == planner.Shrunk(Tiles(1856), Tiles(1088), 1838, 800)
    assert peaks.shortened == shot_plan.Shortened(39, 1652, 800)


def test_scratch_allowance() -> None:
    # DESIGN.md, Memory planner, Host RAM: "The planner sets the scratch's allowance per job:
    # half of what the host has available once the pinned copies are counted, and no more than
    # the longest tiled shot needs"; none where no pass is tiled. Once what the run holds of the
    # longest shot is counted with the copies, ahead of the allowance. 1080p on a 24 GB card,
    # both passes tiled, every part of the weights pinned, 15.81 GiB: the longest shot's 301
    # frames take 3.57 GiB of scratch, and its 76 latents 0.19 GiB through its windows of 39.
    free, film = our_free(24), shots(5, 76, 17)
    pinned = 36 * paged(BLOCK) + paged(peaks.DIT_OTHER) + paged(peaks.VAE_WEIGHTS)
    kept = peaks.scratch(OURS_1080, 301)
    held = 191 * 1_059_840

    def allowed(host: int | None) -> int:
        made, predicted = planner.plan(free, HD, film, Pins(), host)
        assert bare(made) == Plan(39, Tiles(1856), Tiles(1088), 32, 1, 0, vae_in_dit=False)
        assert predicted.scratch == kept and predicted.shot == held
        assert predicted.host == shot_plan.HOST_RESIDENT + pinned + held + made.scratch_allowance
        assert host is None or predicted.host <= host
        return made.scratch_allowance

    assert allowed(64 * GIB) == allowed(24 * GIB) == kept
    assert allowed(23 * GIB) == (23 * GIB - pinned - held) // 2 < kept
    assert allowed(22 * GIB) == (22 * GIB - pinned - held) // 2
    assert (f"{gib(allowed(23 * GIB)):.2f}", f"{gib(allowed(22 * GIB)):.2f}") == ("3.50", "3.00")
    # No more than what the process's own 2.4 GiB, the pinned copies and the shot leave, which
    # half of what the last two leave passes under 4.8 GiB (PROVISIONAL): down to nothing, the
    # job never refused for its allowance.
    left = 19 * GIB - pinned - held
    assert allowed(19 * GIB) == left - shot_plan.HOST_RESIDENT < left // 2
    assert f"{gib(allowed(19 * GIB)):.2f}" == "0.60"
    assert allowed(shot_plan.HOST_RESIDENT + pinned + held) == 0
    # What the host has available unknown: the longest shot's scratch, 8 GiB at most
    # (PROVISIONAL), reached by 601 frames at 4K.
    assert allowed(None) == kept and shot_plan.ALLOWANCE_UNKNOWN_HOST == 8 * GIB
    made, predicted = planner.plan(our_free(96), UHD, shots(151), Pins(), None)
    assert (made.scratch_allowance, predicted.scratch) == (8 * GIB, 601 * 50_503_680)
    # Nothing tiled, nothing allowed, whatever the host has.
    made, _ = planner.plan(our_free(96), HD, film, Pins(), 64 * GIB)
    assert made == Plan(76) and made.scratch_allowance == 0
    # It shapes nothing, and is recorded with the memory, for information; a resume plans it
    # again, from what the host has that day, whatever was recorded.
    made, _ = planner.plan(free, HD, film, Pins(), 22 * GIB)
    assert made.record()["memory"]["scratch_allowance"] == (22 * GIB - pinned - held) // 2
    assert Plan.recorded(made.record()) == Plan(39, Tiles(1856), Tiles(1088))
    again, _ = planner.plan(free, HD, film, Pins(), 64 * GIB, recorded=made)
    assert again == replace(made, scratch_allowance=kept)
    # The plan's lines say it, and the shots whose scratch goes to its file instead, whole: of
    # 5, 76 and 17 latents, the one over 3.00 GiB.
    made, predicted = planner.plan(free, HD, film, Pins(), 64 * GIB)
    assert said_line(planner.said(made, predicted, "a card", HD, film, Pins()), "scratch") == (
        "plan: scratch: what a tiled pass keeps of the shot's frames between its tiles: 3.57 GiB"
        " for the longest tiled shot; in system memory up to an allowance of"
        " 3.57 GiB, which holds every shot's"
    )
    made, predicted = planner.plan(free, HD, film, Pins(), 22 * GIB)
    assert said_line(planner.said(made, predicted, "a card", HD, film, Pins()), "scratch") == (
        "plan: scratch: what a tiled pass keeps of the shot's frames between its tiles: 3.57 GiB"
        " for the longest tiled shot; in system memory up to an allowance of"
        " 3.00 GiB, over it, the scratch of 1 of 3"
        " shots goes whole to a memory-mapped file"
    )
    made, predicted = planner.plan(free, HD, film, Pins(), shot_plan.HOST_RESIDENT + pinned + held)
    assert said_line(planner.said(made, predicted, "a card", HD, film, Pins()), "scratch") == (
        "plan: scratch: what a tiled pass keeps of the shot's frames between its tiles: 3.57 GiB"
        " for the longest tiled shot; in system memory up to an allowance of"
        " 0.00 GiB, over it, the scratch of 3 of 3"
        " shots goes whole to a memory-mapped file"
    )
    # A plan from elsewhere is judged with the allowance it has (fits), said with the rest.
    large = replace(made, scratch_allowance=6 * GIB)
    assert fits(large, free, HD, film, 24 * GIB) == (
        "the plan takes 24.40 GiB of system memory (2.40 for the process, 15.81 pinned for 36"
        " blocks, the DiT's other parameters and the VAE's weights that leave the GPU, 0.19 for"
        " the longest shot, 301 frames (76 latents), at its longest window, its latent and its"
        " noise beside the window's output, 6.00 for a tiled shot's scratch) and 24.00 GiB are"
        " available, 0.40 GiB short"
    )
    assert fits(large, free, HD, film, 25 * GIB) is None
    # The plan's lines say the same figures, and "in all" the refusal's total, their sum.
    said = planner.said(large, predicted, "a card", HD, film, Pins())
    assert said_line(said, "system memory") == (
        "plan: system memory: 24.40 GiB in all; 15.81 GiB pinned for 36 blocks, the DiT's other"
        " parameters and the VAE; 0.19 GiB for the longest shot at the most, at its longest window"
    )
    assert said_line(said, "scratch") == (
        "plan: scratch: what a tiled pass keeps of the shot's frames between its tiles: 3.57 GiB"
        " for the longest tiled shot; in system memory up to an allowance of"
        " 6.00 GiB, which holds every shot's"
    )


def test_the_shot_that_holds_the_most() -> None:
    # The plan counts the system memory of one shot, the one that holds the most
    # (shot_plan._shot_held): the longest, but where a shorter one in fewer windows holds more.
    # Under a cap, the most a shot holds can fall as the shot grows into one window more: 12
    # latents in one window keep 36 at the window's, 13 in two windows of 8 and 7 keep 34, at
    # the window's (26 and 8) as at the merge (15, 13 and 6). The moment that holds the most:
    # the check for a shot of 1 latent; then the windows'; and the merge once a shot's n windows
    # are shorter than 2 n + 4 latents.
    def most(latents: int, cap: int | None) -> tuple[int, str]:
        counted = shot_plan.host_moments(OURS_1080, latents, cap)
        moment = max(shot_plan.MOMENTS, key=lambda moment: counted[moment])
        return counted[moment], moment

    one, two = (shot_plan.host_moments(OURS_1080, count, 12) for count in (12, 13))
    assert (one["windows"] // 1_059_840, two["windows"] // 1_059_840) == (36, 34)
    assert two["merge"] // 1_059_840 == 34 and most(13, 12)[0] < most(12, 12)[0]
    for cap in (None, 5, 6, 7, 8, 9, 12, 20, 44, 91):
        held = [most(latents, cap) for latents in range(1, 600)]
        moments = [moment for _, moment in held]
        assert moments[0] == "check" and moments[1] == "windows"
        for latents, moment in enumerate(moments[1:], 2):
            windows = shot_layout(4 * latents - 3, cap)
            longest = max(end - start for start, end in windows)
            merges = len(windows) > 1 and 2 * len(windows) + 4 > longest
            assert moment == ("merge" if merges else "windows"), (cap, latents)
        # It falls only where a shot takes one window more than the shot a latent shorter: 38
        # latents under a cap of 20 hold 96 at a window's, two of 20, and 39, in three of 15,
        # 14 and 14, hold 93.
        for (low, _), (high, _), latents in zip(held, held[1:], range(2, 600), strict=False):
            more = len(shot_layout(4 * latents - 3, cap)) > len(shot_layout(4 * latents - 7, cap))
            assert high >= low or more, (cap, latents)
        assert cap != 20 or [count // 1_059_840 for count, _ in held[37:39]] == [96, 93]
    # Through the planner: a film's plan counts the shot that holds the most under its cap.
    # 4K on a 24 GB card, windows of 10: the longest shot's merge, 2,254 latents of the shot of
    # 1,000; at 1080p, windows of 42, its merge too, 2,054.
    film = shots(40, 1000, 7)
    made, peaks = planner.plan(our_free(24), UHD, film, Pins(), None)
    assert made.window == 10 and (peaks.shot, peaks.moment) == (2254 * 4_208_640, "merge")
    made, peaks = planner.plan(our_free(24), HD, film, Pins(), None)
    assert made.window == 42 and (peaks.shot, peaks.moment) == (2054 * 1_059_840, "merge")
    assert peaks.record()["host_shot"] == peaks.shot and peaks.holder == 1000
    # A shot of the cap's length beside a longer one holds more than it: 1080p on a 96 GB
    # card, shots of 76 and 77 latents under --window 76: the first holds 228 latents' worth
    # through its one window, the second, in two windows of 40 and 39, 194. The plan's line and
    # a refusal name that shot.
    film = shots(76, 77)
    made, peaks = planner.plan(our_free(96), HD, film, Pins(window=76), None)
    assert (made.window, peaks.holder, peaks.moment) == (76, 76, "windows")
    assert peaks.shot == 228 * 1_059_840 > max(shot_plan.host_moments(OURS_1080, 77, 76).values())
    assert max(shot_plan.host_moments(OURS_1080, 77, 76).values()) == 194 * 1_059_840
    said = planner.said(made, peaks, "a card", HD, film, Pins(window=76))
    assert said_line(said, "system memory") == (
        "plan: system memory: 2.63 GiB in all; nothing pinned; 0.23 GiB for a shot of 76 latents"
        " at the most, more than the longest, at its longest window"
    )
    assert refused(our_free(96), HD, film, Pins(window=76), 2 * GIB) == (
        "the plan takes 2.63 GiB of system memory (2.40 for the process, 0.23 for a shot of 76"
        " latents, which holds more than the longest, at its longest window, its latent and its"
        " noise beside the window's output) and 2.00 GiB are available, 0.63 GiB short"
    )
    # Of two shots that hold as much, the longer is named: under a cap of 12, a shot of 12
    # latents in one window and one of 14 in two windows of 8 hold 36 latents' worth each, and
    # the plan's line says the longest shot's. One shot's is the longest's.
    most_held = shot_plan._most_held  # pyright: ignore[reportPrivateUsage]
    assert most_held(OURS_1080, (12, 13), 12) == (36 * 1_059_840, "windows", 12)
    assert most_held(OURS_1080, (13,), 12) == (34 * 1_059_840, "windows", 13)
    twelve, fourteen = (shot_plan.host_moments(OURS_1080, count, 12) for count in (12, 14))
    assert max(twelve.values()) == max(fourteen.values()) == 36 * 1_059_840
    assert most_held(OURS_1080, (12, 14), 12) == (36 * 1_059_840, "windows", 14)
    film = shots(12, 14)
    made, peaks = planner.plan(our_free(96), HD, film, Pins(window=12), None)
    assert (peaks.shot, peaks.holder) == (36 * 1_059_840, 14)
    said = planner.said(made, peaks, "a card", HD, film, Pins(window=12))
    assert " for the longest shot at the most, at its longest window" in said_line(
        said, "system memory"
    )
    # The moment that holds the most is one of three: the encode's end and the decode never
    # are, and have no sentence (the loop above reads shots of 1 to 599 latents under ten caps).
    assert set(shot_plan._MOMENT_SAID) == {"check", "windows", "merge"}  # pyright: ignore[reportPrivateUsage]
    assert set(shot_plan.MOMENTS) - set(shot_plan._MOMENT_SAID) == {"encode", "decode"}  # pyright: ignore[reportPrivateUsage]
    # Every length is read that could hold the most: over many films, the most of every shot's
    # moments, counted here.
    for cap in (None, 5, 9, 20, 44):
        for lengths in ((1,), (3, 4), (5, 6, 7), (9, 10, 11, 40), (19, 20, 21, 22, 300), (44, 45)):
            counted = [shot_plan.host_moments(OURS_1080, count, cap) for count in lengths]
            assert most_held(OURS_1080, lengths, cap)[0] == max(
                max(moments.values()) for moments in counted
            )


def test_recorded_cap_checked() -> None:
    # A cap under 5 latents is the planner's own for a job whose longest shot is as short, each
    # shot one window; with a longer shot no layout exists for it (shots.window_layout), and a
    # record that has one was edited: refused, where the layout would raise.
    made, _ = planner.plan(our_free(96), HD, shots(3, 2), Pins(), None)
    assert made.window == 3
    planner.check_recorded(made, shots(3, 2), HD)
    planner.check_recorded(Plan(4), shots(4, 1), HD)
    planner.check_recorded(Plan(5), shots(76), HD)
    planner.check_recorded(Plan(None), shots(76), HD)
    with pytest.raises(PlanError) as error:
        planner.check_recorded(Plan(3), shots(3, 6), HD)
    assert str(error.value) == (
        "not a plan as seedvr2x records one: its window, 3, with a shot of 6 latents: a cap that"
        " a shot is longer than is 5 latents at least"
    )
    with pytest.raises(ValueError, match="windows of 3 latents leave none of their own"):
        shot_layout(21, 3)
    with pytest.raises(PlanError, match="its window, 4, with a shot of 5 latents"):
        planner.check_recorded(Plan(4), shots(5), HD)


def test_an_int8_dit_plans_by_its_own_figures() -> None:
    # The DiT of seedvr2x's int8 files (peaks.INT8): its phase by its own weights and the W8A8
    # path's buffers (peaks.dit_peak), its blocks' pinned copies and its VAE phases' share by its
    # blocks' bytes; on cards of 16 to 48 GB at 1080p, a shot of 301 frames, a window as long as
    # fp16's or longer, with fewer blocks swapped or as few, every predicted peak under the
    # budget, and its line saying int8. The int8 figures beyond the weights are provisional
    # (peaks.INT8_*).
    job = (HD, shots(76), Pins(), 64 * GIB)
    for card in (16, 24, 32, 48):
        fp16, fp16_peaks = planner.plan(our_free(card), *job, split=True)
        int8, int8_peaks = planner.plan(our_free(card), *job, split=True, dit_model=peaks.INT8)
        assert int8_peaks.dit_model == peaks.INT8 and fp16_peaks.dit_model == peaks.FP16
        # The record keeps the format by its name, read back as the format itself; a record
        # from before int8, with none, as fp16 (planner.recorded_dit_model).
        for predicted, name in ((fp16_peaks, "fp16"), (int8_peaks, "int8")):
            record = predicted.record()
            assert record["dit_format"] == name
            assert planner.recorded_dit_model(record) == predicted.dit_model
        assert int8_peaks.window >= fp16_peaks.window and int8.swapped <= fp16.swapped, card
        assert int8_peaks.dit == peaks.dit_peak(
            OURS_1080,
            int8_peaks.window,
            int8.swapped,
            int8.vae_in_dit,
            dit_model=peaks.INT8,
        )
        assert max(int8_peaks.need(phase) for phase in ("encode", "dit", "decode")) <= (
            int8_peaks.budget
        )
        pinned = shot_plan._pinned(int8, peaks.INT8)  # pyright: ignore[reportPrivateUsage]
        assert int8_peaks.host == (
            shot_plan.HOST_RESIDENT + pinned + int8_peaks.shot + int8.scratch_allowance
        )
    assert planner.recorded_dit_model({"attention": "sdpa"}) == peaks.FP16
    with pytest.raises(KeyError):
        planner.recorded_dit_model({"dit_format": "fp8"})
    # On 24 GB, windows of 39 latents either way (half of the 76-latent shot, the card holding
    # up to 42), every block swapped holding no longer one: the DiT's phase then holds the slots
    # and the tokens, which int8 shrinks little. fp16 swaps 32 blocks for them and keeps one in
    # the encode, none in the decode, 15.81 GiB pinned for every part of the weights; int8 swaps
    # 28 and keeps 2 and 1, 7.86 GiB pinned for 35 blocks and the VAE.
    fp16_24, fp16_24_peaks = planner.plan(our_free(24), *job, split=True)
    int8_24, int8_24_peaks = planner.plan(our_free(24), *job, split=True, dit_model=peaks.INT8)
    assert (fp16_24_peaks.window, int8_24_peaks.window) == (39, 39)
    assert (fp16_24.swapped, int8_24.swapped) == (32, 28)
    blocks = (
        shot_plan._pinned(fp16_24, peaks.FP16),  # pyright: ignore[reportPrivateUsage]
        shot_plan._pinned(int8_24, peaks.INT8),  # pyright: ignore[reportPrivateUsage]
    )
    assert [f"{gib(pinned):.2f}" for pinned in blocks] == ["15.81", "7.86"]
    assert (int8_24.resident_encode, int8_24.resident_decode) == (2, 1)
    said = planner.said(int8_24, int8_24_peaks, "a card", *job[:3])
    assert said_line(said, "DiT") == (
        "plan: DiT: its weights in int8, 7.76 GiB, half of fp16's; a window's memory counted as"
        " fp16's plus 0.38 GiB for int8's buffers until int8's own is measured; --dit-model names"
        " the file"
    )
    # A recorded plan planned again around its shape, and a resume's refusal, in int8's figures.
    recorded = replace(bare(int8_24), swapped=0, resident_encode=None, resident_decode=None)
    again, _ = planner.plan(our_free(24), *job, split=True, recorded=recorded, dit_model=peaks.INT8)
    assert bare(again) == bare(int8_24)
    # Another architecture has no figures to plan by.
    with pytest.raises(
        ValueError, match="the DiT's figures are for its fp16, int8 and dynamic GGUF files"
    ):
        planner.plan(our_free(24), *job, dit_model=weights.DIT_3B)


def test_a_gguf_dit_plans_by_its_own_figures() -> None:
    # The DiT of seedvr2x's dynamic GGUF files (peaks.GGUF_7B, the regular 7B's, and
    # peaks.GGUF_7B_SHARP): its phase by its own weights and the dequantisation's buffers
    # (peaks.dit_peak), its blocks' pinned copies by their own bytes, with a window as long as
    # int8's on every card of 16 to 48 GB; its format recorded by the name DIT_FORMATS gives it,
    # read back as itself, and its line saying so, in the words of the picture's (still_plan.said).
    job = (HD, shots(76), Pins(), 64 * GIB)
    assert planner.DIT_FORMATS["gguf_dyn"] == peaks.GGUF_7B
    assert planner.DIT_FORMATS["gguf_sharp_dyn"] == peaks.GGUF_7B_SHARP
    for dit_model, name in ((peaks.GGUF_7B, "gguf_dyn"), (peaks.GGUF_7B_SHARP, "gguf_sharp_dyn")):
        for card in (16, 24, 32, 48):
            _, int8_peaks = planner.plan(our_free(card), *job, split=True, dit_model=peaks.INT8)
            made, predicted = planner.plan(our_free(card), *job, split=True, dit_model=dit_model)
            assert predicted.dit_model == dit_model
            record = predicted.record()
            assert record["dit_format"] == name
            assert planner.recorded_dit_model(record) == dit_model
            # A window as long as int8's or longer (the longer window may swap more blocks).
            assert predicted.window >= int8_peaks.window, (name, card)
            assert predicted.dit == peaks.dit_peak(
                OURS_1080, predicted.window, made.swapped, made.vae_in_dit, dit_model=dit_model
            )
            assert max(predicted.need(phase) for phase in ("encode", "dit", "decode")) <= (
                predicted.budget
            )
            pinned = shot_plan._pinned(made, dit_model)  # pyright: ignore[reportPrivateUsage]
            assert predicted.host == (
                shot_plan.HOST_RESIDENT + pinned + predicted.shot + made.scratch_allowance
            )
            said = planner.said(made, predicted, "a card", *job[:3])
            weights_gib = f"{gib(peaks.weights_bytes(dit_model)):.2f}"
            assert said_line(said, "DiT") == (
                f"plan: DiT: its weights in seedvr2x's dynamic GGUF, {weights_gib} GiB, under a"
                " third of fp16's; a window's memory counted as fp16's plus"
                f" {gib(peaks.GGUF_BUFFERS):.2f} GiB for the dequantisation's buffers;"
                " --dit-model names the file"
            )
            # The same as the picture's plan says it.
            assert said_line(said, "DiT").removeprefix("plan: ") in [
                line.removeprefix("plan: ")
                for line in still_plan.said(still_for(dit_model), say=planner.option)
            ]
    # A record from before the GGUF, with no format, is fp16's.
    assert planner.recorded_dit_model({"attention": "sdpa"}) == peaks.FP16
    # A recorded plan planned again around its shape, in the GGUF's figures.
    made, _ = planner.plan(our_free(24), *job, split=True, dit_model=peaks.GGUF_7B)
    recorded = replace(bare(made), swapped=0, resident_encode=None, resident_decode=None)
    again, _ = planner.plan(
        our_free(24), *job, split=True, recorded=recorded, dit_model=peaks.GGUF_7B
    )
    assert bare(again) == bare(made)


def still_for(dit_model: Any) -> Any:
    """A picture's plan (still_plan) for 1080p on 24 GB in the DiT's format, for its DiT line."""
    return still_plan.plan(
        (1920, 1080),
        our_free(24),
        64 * GIB,
        process=shot_plan.HOST_RESIDENT,
        input_pixels=1920 * 1080,
        dit_model=dit_model,
    )


# A card of 16 GB and a shot of 301 frames: the host's memory decides which files plan. fp16's
# 15 GiB of pinned copies are refused at 5 GiB and under, int8's at 4 and under, the GGUF's at 2
# (shot_plan.plan, measured by the plan itself: the tests below ask it).
SHARP_FILES = {
    "fp16": "seedvr2x_ema_7b_sharp_fp16.safetensors",
    "int8": "seedvr2x_ema_7b_sharp_int8_convrot.safetensors",
    "gguf": "seedvr2x_ema_7b_sharp_dyn.gguf",
}


INT8_WAY = (
    "int8 (--dit-model seedvr2x_ema_7b_sharp_int8_convrot.safetensors, 8.3 GB)",
    "the dynamic GGUF (--dit-model seedvr2x_ema_7b_sharp_dyn.gguf, 4.8 GB)",
)


def naming(host: int, dit_model: Any, dit_file: str, **options: Any) -> PlanError:
    """The refusal of the job on a card of 16 GB with `host` GiB of system memory, naming files."""
    with pytest.raises(PlanError) as refused:
        planner.plan_naming(
            our_free(16), HD, shots(76), Pins(), host * GIB, dit_file=dit_file,
            dit_model=dit_model, split=True, **options,
        )  # fmt: skip
    return refused.value


def plain(host: int, dit_model: Any, **options: Any) -> PlanError:
    with pytest.raises(PlanError) as refused:
        planner.plan(
            our_free(16), HD, shots(76), Pins(), host * GIB, dit_model=dit_model, split=True,
            **options,
        )  # fmt: skip
    return refused.value


def test_the_floor_refusal_names_the_smaller_files_as_it_did() -> None:
    # An output too large for the card (shot_plan's floor refusal) ends with the resolution that
    # fits, and the smaller DiT files still serve it: its Setting("resolution") is no pin's
    # (PlanError.instead is a window, blocks or a tile there), so plan_naming adds its way out
    # as it did when that refusal carried no instead (a7ffd27; review H1).
    # 4K on 11.5 GiB free: the lean attention's smallest window fits 1440p on 8.3 GiB, where the
    # VAE's floor is (no DiT file helps there), so the case that was 1440p on 10.5 is 4K now.
    args = (round(11.5 * GIB), UHD, shots(9), Pins(), 64 * GIB)
    with pytest.raises(PlanError) as plain_refusal:
        planner.plan(*args, dit_model=peaks.FP16, split=True)
    assert plain_refusal.value.instead is not None
    assert plain_refusal.value.instead.name == "resolution"
    with pytest.raises(PlanError) as named:
        planner.plan_naming(*args, dit_file=SHARP_FILES["fp16"], dit_model=peaks.FP16, split=True)
    said = planner.worded(named.value)
    assert said.startswith(planner.worded(plain_refusal.value) + "; or ")
    assert "is --resolution 2128, about 3782x2128; or " in said
    # Only the dynamic GGUF plans it on 11.5 GiB, as 1440p on 10.5 did on main.
    assert said.endswith("a smaller DiT file that plans this job: " + INT8_WAY[1])


def test_the_smaller_files_are_named_with_the_command_lines_options() -> None:
    # The sentences of the smaller DiT files (planner.smaller_way, planner.smaller_line), which
    # the library's pull.smaller_files and smaller_planning leave to the apps: moved here from
    # tests/test_pull.py (core) with the expectations unchanged.
    sharp_name = "seedvr2x_ema_7b_sharp_fp16.safetensors"
    planned = pull.smaller_planning(
        "seedvr2_ema_7b_fp16.safetensors",
        weights.DIT_7B,
        lambda each: each.name if each == weights.DIT_7B_SHARP_DYN else None,
    )
    int8_sharp = "int8 (--dit-model seedvr2x_ema_7b_sharp_int8_convrot.safetensors, 8.3 GB)"
    gguf_sharp = "the dynamic GGUF (--dit-model seedvr2x_ema_7b_sharp_dyn.gguf, 4.8 GB)"
    int8_regular = "int8 (--dit-model seedvr2x_ema_7b_int8_convrot.safetensors, 8.3 GB)"
    gguf_regular = "the dynamic GGUF (--dit-model seedvr2x_ema_7b_dyn.gguf, 4.8 GB)"
    way = "a smaller DiT file that plans this job: "
    assert (
        planner.smaller_way([one for one, _ in planned]) == f"{way}for the sharp 7B, {gguf_sharp}"
    )
    assert (
        planner.smaller_way(pull.smaller_files(sharp_name, weights.DIT_7B))
        == f"{way}{int8_sharp} or {gguf_sharp}"
    )
    # Where the run was given --model-dir, the way out says so once, after "this job".
    assert planner.smaller_way(pull.smaller_files(sharp_name, weights.DIT_7B), True) == (
        f"a smaller DiT file that plans this job, put in --model-dir: {int8_sharp} or {gguf_sharp}"
    )
    # The plan's line of those that plan the job better lists them the same way, with the
    # figures of the plan and of each file.
    assert planner.smaller_line(
        pull.Swaps(11, 10),
        [
            (one, pull.Swaps(19, 17))
            for one in pull.smaller_files("seedvr2x_ema_7b_sharp_fp16.safetensors", weights.DIT_7B)
        ],
    ) == (
        "smaller DiT files plan this job better: this plan swaps 11 blocks with windows of 10"
        " latents; int8 19 blocks, windows of 17; the dynamic GGUF 19 blocks, windows of 17;"
        " --dit-model names them"
    )
    # A picture is one latent: blocks alone.
    assert planner.smaller_line(
        pull.Swaps(3), [(one, pull.Swaps(0)) for one in pull.smaller_files("x", weights.DIT_7B)]
    ) == (
        "smaller DiT files plan this job better: this plan swaps 3 blocks; for the sharp 7B"
        " (int8 0 blocks; the dynamic GGUF 0 blocks); for the 7B (int8 0 blocks; the dynamic GGUF"
        " 0 blocks); --dit-model names them"
    )
    assert planner.smaller_way(pull.smaller_files("x", weights.DIT_7B)) == (
        f"{way}for the sharp 7B, {int8_sharp} or {gguf_sharp};"
        f" for the 7B, {int8_regular} or {gguf_regular}"
    )


def test_a_refusal_for_memory_names_the_smaller_files_that_plan() -> None:
    # DESIGN.md, Weights, "The dynamic GGUF before v0.1.0 too": where the host's memory is short
    # for float16, the refusal's way out names the smaller files that plan the same job, found
    # by planning it again in their figures; the file is never changed by itself. The refusal's
    # own words stay, the way out after them.
    # 5 GiB: int8 plans (12 blocks swapped on 6 GiB, 7 on 5), the GGUF too: both named.
    refused = naming(5, peaks.FP16, SHARP_FILES["fp16"])
    assert str(refused).startswith(str(plain(5, peaks.FP16)) + "; or ")
    assert str(refused).endswith("a smaller DiT file that plans this job: " + " or ".join(INT8_WAY))
    assert (refused.phase, refused.shortfall, refused.instead) == ("host", None, None)
    # 3 GiB: int8 is refused as fp16 is; only the GGUF plans, and only it is named.
    only = naming(3, peaks.FP16, SHARP_FILES["fp16"])
    assert str(only).endswith("a smaller DiT file that plans this job: " + INT8_WAY[1])
    assert "int8" not in str(only).removeprefix(str(plain(3, peaks.FP16)))
    # Where the run was given --model-dir, the way out says once that the file goes there.
    here = naming(5, peaks.FP16, SHARP_FILES["fp16"], model_dir=True)
    assert str(here).endswith(
        "a smaller DiT file that plans this job, put in --model-dir: " + " or ".join(INT8_WAY)
    )
    assert str(here).count("--model-dir") == 1
    assert str(here).replace(", put in --model-dir", "") == str(refused)
    # The regular 7B's pinned file gets the regular 7B's smaller files, unmarked.
    regular = str(naming(5, peaks.FP16, "seedvr2x_ema_7b_fp16.safetensors"))
    assert regular.endswith(
        "a smaller DiT file that plans this job: int8 (--dit-model"
        " seedvr2x_ema_7b_int8_convrot.safetensors, 8.3 GB) or the dynamic GGUF (--dit-model"
        " seedvr2x_ema_7b_dyn.gguf, 4.8 GB)"
    )
    # A file that is not a pinned one (numz's, a renamed one) is never taken for a 7B by its
    # name: both 7Bs' files are named, each said as its model's, those that plan.
    for name in ("seedvr2_ema_7b_fp16.safetensors", "seedvr2_ema_7b_sharp_fp16.safetensors", "x"):
        assert str(naming(5, peaks.FP16, name)).endswith(
            "a smaller DiT file that plans this job: for the sharp 7B, "
            + " or ".join(INT8_WAY)
            + "; for the 7B, int8 (--dit-model seedvr2x_ema_7b_int8_convrot.safetensors, 8.3 GB)"
            " or the dynamic GGUF (--dit-model seedvr2x_ema_7b_dyn.gguf, 4.8 GB)"
        )
    # An int8 DiT is offered the GGUF alone; a GGUF DiT nothing: its own refusal, as it was.
    assert str(naming(3, peaks.INT8, SHARP_FILES["int8"])).endswith(
        "a smaller DiT file that plans this job: " + INT8_WAY[1]
    )
    assert str(naming(1, peaks.GGUF_7B_SHARP, SHARP_FILES["gguf"])) == str(
        plain(1, peaks.GGUF_7B_SHARP)
    )
    # Nothing plans: the refusal is the plan's, as it was.
    assert str(naming(1, peaks.FP16, SHARP_FILES["fp16"])) == str(plain(1, peaks.FP16))
    # A resume's recorded shape (a smaller file is another job) and a pin's refusal (it ends with
    # the pin that fits) are left as they are, and so is the card's: the VAE's floor binds the
    # same in every format.
    made, _ = planner.plan(our_free(24), HD, shots(76), Pins(), 64 * GIB, split=True)
    with pytest.raises(PlanError) as small:
        planner.plan_naming(
            round(6 * GIB), HD, shots(76), Pins(), 64 * GIB, dit_file=SHARP_FILES["fp16"],
            split=True, recorded=made,
        )  # fmt: skip
    assert "smaller DiT file" not in str(small.value)
    with pytest.raises(PlanError) as pin:
        planner.plan_naming(
            our_free(16), HD, shots(76), Pins(window=60), 5 * GIB, dit_file=SHARP_FILES["fp16"],
            split=True,
        )  # fmt: skip
    assert pin.value.instead and "smaller DiT file" not in str(pin.value)
    with pytest.raises(PlanError) as card:
        planner.plan_naming(
            round(5 * GIB), HD, shots(76), Pins(), 64 * GIB, dit_file=SHARP_FILES["fp16"],
            split=True,
        )  # fmt: skip
    assert "smaller DiT file" not in str(card.value)
    # The files the refusal names are the pinned ones, with the pins' sizes.
    from seedvr2x import pull

    for each in pull.smaller_files(SHARP_FILES["fp16"], peaks.FP16):
        assert pull.PINNED[each.name].size == each.size


def test_the_plan_names_the_smaller_files_that_plan_the_job_better() -> None:
    # The same job planned in int8's and the GGUF's figures: the plan's line names those that swap
    # fewer blocks than the plan's own or run a longer window, saying both figures for the plan
    # and for each file, never for a recorded plan or a GGUF DiT (DESIGN.md, Weights).
    job = (HD, shots(76), Pins(), 64 * GIB)
    arguments = (our_free(24), *job)
    options: dict[str, Any] = dict(split=True, dit_model=peaks.FP16)
    made, predicted = planner.plan(*arguments, split=True)
    swapped = {
        name: planner.plan(*arguments, split=True, dit_model=model)[0].swapped
        for name, model in (("int8", peaks.INT8), ("gguf", peaks.GGUF_7B_SHARP))
    }
    assert made.swapped == 32 and swapped == {"int8": 28, "gguf": 17}
    windows = {
        name: planner.plan(*arguments, split=True, dit_model=model)[1].window
        for name, model in (("int8", peaks.INT8), ("gguf", peaks.GGUF_7B_SHARP))
    }
    line = planner.smaller_swaps(made, predicted, arguments, options, SHARP_FILES["fp16"])
    assert line == (
        "smaller DiT files plan this job better: this plan swaps 32 blocks with windows of"
        f" {predicted.window} latents; int8 28 blocks, windows of {windows['int8']}; the dynamic"
        f" GGUF 17 blocks, windows of {windows['gguf']}; --dit-model names them"
    )
    lines = planner.said(made, predicted, "a card", *job[:3], smaller=line)
    at = next(
        number for number, each in enumerate(lines) if each.startswith("plan: blocks swapped")
    )
    assert lines[at + 1] == f"plan: {line}"
    assert planner.said(made, predicted, "a card", *job[:3]) == [
        each for each in lines if each != f"plan: {line}"
    ]
    # An int8 DiT is offered the GGUF alone, only if that plans better.
    int8_made, int8_predicted = planner.plan(*arguments, split=True, dit_model=peaks.INT8)
    assert planner.smaller_swaps(
        int8_made,
        int8_predicted,
        arguments,
        dict(split=True, dit_model=peaks.INT8),
        SHARP_FILES["int8"],
    ) == (
        "smaller DiT files plan this job better: this plan swaps 28 blocks with windows of"
        f" {int8_predicted.window} latents; the dynamic GGUF 17 blocks, windows of"
        f" {windows['gguf']}; --dit-model names them"
    )
    gguf_made, gguf_predicted = planner.plan(*arguments, split=True, dit_model=peaks.GGUF_7B_SHARP)
    gguf = dict(split=True, dit_model=peaks.GGUF_7B_SHARP)
    assert (
        planner.smaller_swaps(gguf_made, gguf_predicted, arguments, gguf, SHARP_FILES["gguf"])
        is None
    )
    # The DiT whole on the card, the longest window the shot has: nothing better, nothing named.
    whole = (our_free(48), HD, shots(76), Pins(window=5), 64 * GIB)
    whole_made, whole_predicted = planner.plan(*whole, split=True)
    assert whole_made.swapped == 0
    assert (
        planner.smaller_swaps(whole_made, whole_predicted, whole, options, SHARP_FILES["fp16"])
        is None
    )
    # A recorded plan keeps its file.
    recorded = dict(options, recorded=made)
    assert planner.smaller_swaps(made, predicted, arguments, recorded, "x") is None


def test_the_plan_names_a_smaller_file_that_runs_a_longer_window_where_the_host_is_short() -> None:
    # The review's case (impl-gguf, 2026-11-04): 1080p, one shot of 101 latents, SDPA, 24 GiB free
    # and 8 GiB of system memory. The host cuts float16's window (its pinned copies are 15 GiB,
    # the window shortened for them: shot_plan.plan, "host"), and the smaller files swap more blocks
    # yet run longer windows: the line names them by the window, as DESIGN.md has it, "where the
    # card or the host's memory is short for float16".
    arguments = (24 * GIB, HD, shots(101), Pins(), 8 * GIB)
    options: dict[str, Any] = dict(split=True, dit_model=peaks.FP16)
    made, predicted = planner.plan(*arguments, **options)
    found = {
        name: planner.plan(*arguments, **dict(options, dit_model=model))
        for name, model in (("int8", peaks.INT8), ("gguf", peaks.GGUF_7B_SHARP))
    }
    assert (made.swapped, predicted.window) == (10, 22)
    assert [(each.swapped, each_peaks.window) for each, each_peaks in found.values()] == [
        (14, 35),
        (0, 35),
    ]
    line = planner.smaller_swaps(made, predicted, arguments, options, SHARP_FILES["fp16"])
    assert line == (
        "smaller DiT files plan this job better: this plan swaps 10 blocks with windows of 22"
        " latents; int8 14 blocks, windows of 35; the dynamic GGUF 0 blocks, windows of 35;"
        " --dit-model names them"
    )
    # A file that is not a pinned one: each 7B's files in brackets.
    other = planner.smaller_swaps(made, predicted, arguments, options, "x")
    assert other is not None and other.startswith(
        "smaller DiT files plan this job better: this plan swaps 10 blocks with windows of 22"
        " latents; for the sharp 7B (int8 14 blocks, windows of 35; the dynamic GGUF 0 blocks,"
        " windows of 35); for the 7B (int8 14 blocks, windows of 35; the dynamic GGUF 0 blocks,"
        " windows of 35); --dit-model"
    )


def test_said() -> None:
    # The plan's lines, as --plan prints them and every run logs them (DESIGN.md, Memory planner,
    # --plan): 1080p on a 24 GB card, a shot of 301 frames, with split.
    job = (HD, shots(76), Pins())
    made, peaks = planner.plan(numz_free(24), *job, 64 * GIB, split=True)
    assert planner.said(made, peaks, "NVIDIA RTX PRO 6000 Blackwell", *job) == [
        "plan: card: NVIDIA RTX PRO 6000 Blackwell, 22.56 GiB free, a budget of 21.96 GiB: the"
        " margin of 0.60 GiB left to the desktop and other programs; no reserve, the driver"
        " refusing memory the card doesn't have",
        "plan: frames: 1920x1080 padded to 1920x1104 for the model, 8,280 tokens a latent",
        "plan: shots: 1, the longest of 301 frames (76 latents)",
        "plan: windows: 39 latents at most (153 frames), the longest the budget holds: 0 of 1 shot"
        " in one window, 1 join; a longer window, fewer joins; --window N caps them",
        "plan: quality: standard, the default, one run a shot; --quality enhanced or maximum runs"
        " 2 or 4 and takes each frame from the one that rebuilds it best, for 2 or 4 times the GPU"
        " time",
        "plan: blocks swapped: 32 of the DiT's 36, the fewest that hold the window: slower, not"
        " worse; --swap-blocks N",
        "plan: the VAE: off the GPU in the DiT's phase, its 0.47 GiB given to the window",
        "plan: the DiT in the VAE's passes: which model stays on the GPU in which phase, as much"
        " of the DiT as fits beside each: in the encode, 1 of the DiT's blocks on the GPU; in"
        " the decode, the DiT off the GPU",
        "plan: system memory: 21.97 GiB in all; 15.81 GiB pinned for 36 blocks, the DiT's other"
        " parameters and the VAE; 0.19 GiB for the longest shot at the most, at its longest window",
        "plan: encode: tiles of 1856 px overlapping by 64, 2 across, 1 down, the whole frame not"
        " fitting the budget, the largest that do, up to 2048 px: a slight colour drift, which"
        " split corrects; --encode-tile PX",
        "plan: decode: tiles of 1088 px overlapping by 64, 2 across, 2 down, the whole frame not"
        " fitting the budget, the largest that do, up to 2048 px: a slight colour drift, which"
        " split corrects; --decode-tile PX",
        "plan: the VAE's room: what each VAE pass leaves free, for the large pieces of memory a"
        " pass asks for at once: 1.25 GiB of the budget",
        "plan: scratch: what a tiled pass keeps of the shot's frames between its tiles: 3.57 GiB"
        " for the longest tiled shot; in system memory up to an allowance of"
        " 3.57 GiB, which holds every shot's",
        "plan: predicted peaks: encode 20.55 GiB, DiT 21.86 GiB, decode 20.17 GiB; the DiT's phase"
        " bounds the plan, 0.10 GiB under the budget",
        "plan: time: measured, not guessed: a calibration of the job's own steps once the models"
        " are loaded, its cost said then; it gives the job's time and each unit's time left and"
        " end; --until HH:MM stops before a unit that wouldn't finish by then",
        "plan: pinned: nothing; --window, --swap-blocks, --encode-tile and --decode-tile each fix"
        " a choice",
    ]
    # 4K on the 96 GB card as DESIGN.md first read it, 94.34 GiB free, the same shot, pinned:
    # the untiled encode bounds the plan with the room a VAE pass is held to, which the line
    # says; the peaks are the predicted ones, without the room. Nothing leaves the GPU for it,
    # whatever the shot's length.
    pins = Pins(window=19, swap_blocks=0, encode_tile=0, decode_tile=2048)
    made, peaks = planner.plan(round(94.34 * GIB), UHD, shots(76), pins, 64 * GIB, split=True)
    said = planner.said(made, peaks, "a card", UHD, shots(76), pins)
    # From the windows' line on: the card's, the frames', the attention's and the shots' before it.
    windows = next(i for i, line in enumerate(said) if line.startswith("plan: windows:"))
    assert said[windows:] == [
        "plan: windows: 19 latents at most (73 frames), the longest run 17 latents, pinned by"
        " --window: 0 of 1 shot in one window, 4 joins; a longer window, fewer joins; --window N"
        " caps them",
        "plan: quality: standard, the default, one run a shot; --quality enhanced or maximum runs"
        " 2 or 4 and takes each frame from the one that rebuilds it best, for 2 or 4 times the GPU"
        " time",
        "plan: blocks swapped: none, pinned by --swap-blocks; --swap-blocks N swaps some",
        "plan: the VAE: on the GPU in the DiT's phase, the window fitting beside it",
        "plan: the DiT in the VAE's passes: which model stays on the GPU in which phase, as much"
        " of the DiT as fits beside each: in the encode, the DiT's 36 blocks on the GPU; in the"
        " decode, the DiT's 36 blocks on the GPU",
        "plan: system memory: 17.22 GiB in all; nothing pinned; 0.66 GiB for the longest shot at"
        " the most, at its longest window",
        "plan: encode: untiled, pinned by --encode-tile; --encode-tile PX tiles it",
        "plan: decode: tiles of 2048 px overlapping by 64, 2 across, 2 down, pinned by"
        " --decode-tile: a slight colour drift, which split corrects; --decode-tile PX",
        "plan: the VAE's room: what each VAE pass leaves free, for the large pieces of memory a"
        " pass asks for at once: 1.25 GiB of the budget",
        "plan: scratch: what a tiled pass keeps of the shot's frames between its tiles: 14.16 GiB"
        " for the longest tiled shot; in system memory up to an allowance of"
        " 14.16 GiB, which holds every shot's",
        "plan: predicted peaks: encode 91.54 GiB, DiT 48.99 GiB, decode 79.22 GiB; the encode"
        " bounds the plan, its room counted, 0.95 GiB under the budget",
        "plan: time: measured, not guessed: a calibration of the job's own steps once the models"
        " are loaded, its cost said then; it gives the job's time and each unit's time left and"
        " end; --until HH:MM stops before a unit that wouldn't finish by then",
        "plan: pinned: --window 19, --swap-blocks 0, --encode-tile 0, --decode-tile 2048, taken"
        " exactly",
    ]
    # A recorded plan without a cap, each shot in one window; one block left unswapped, and
    # none; what leaves the GPU in some phase, pinned; and a job of one frame, its cap one
    # latent.
    job = (HD, shots(40), Pins())
    made, peaks = planner.plan(our_free(96), *job, None, recorded=Plan(None))
    assert said_line(planner.said(made, peaks, "a card", *job), "windows") == (
        "plan: windows: no cap, the longest run 40 latents, the longest the budget holds: 1 of 1"
        " shot in one window, 0 joins; --window N caps them"
    )
    made, peaks = planner.plan(our_free(96), *job, None, recorded=Plan(40))
    said = planner.said(replace(made, swapped=35), peaks, "a card", *job)
    assert said_line(said, "the DiT in the VAE's passes").endswith(
        "in the encode, the DiT's 1 unswapped block and its 2 slots on the GPU; in the decode,"
        " the DiT's 1 unswapped block and its 2 slots on the GPU"
    )
    said = planner.said(replace(made, swapped=36), peaks, "a card", *job)
    assert said_line(said, "the DiT in the VAE's passes").endswith(
        "in the encode, the DiT's 2 slots on the GPU; in the decode, the DiT's 2 slots on the GPU"
    )
    away = replace(made, swapped=20, resident_encode=16, resident_decode=0)
    said = planner.said(away, peaks, "a card", *job)
    assert said_line(said, "the VAE") == (
        "plan: the VAE: on the GPU in the DiT's phase, the window fitting beside it"
    )
    assert said_line(said, "the DiT in the VAE's passes") == (
        "plan: the DiT in the VAE's passes: which model stays on the GPU in which phase, as much"
        " of the DiT as fits beside each: in the encode, 16 of the DiT's blocks on the GPU; in"
        " the decode, the DiT off the GPU"
    )
    assert said_line(said, "system memory") == (
        "plan: system memory: 17.87 GiB in all; 15.35 GiB pinned for 36 blocks and the DiT's other"
        " parameters; 0.12 GiB for the longest shot at the most, at its longest window"
    )
    job = (HD, shots(1), Pins())
    made, peaks = planner.plan(our_free(96), *job, None)
    said = planner.said(made, peaks, "a card", *job)
    assert said_line(said, "shots") == "plan: shots: 1, the longest of 1 frame (1 latent)"
    assert said_line(said, "windows") == (
        "plan: windows: 1 latent at most (1 frame), the longest shot whole, which the budget"
        " holds: 1 of 1 shot in one window, 0 joins; --window N caps them"
    )
    # A cap no shot reaches: each in one window, no join.
    job = (HD, shots(3, 40, 2), Pins())
    made, peaks = planner.plan(our_free(96), *job, None)
    said = planner.said(made, peaks, "a card", *job)
    assert said_line(said, "shots") == "plan: shots: 3, the longest of 157 frames (40 latents)"
    assert said_line(said, "windows") == (
        "plan: windows: 40 latents at most (157 frames), the longest shot whole, which the budget"
        " holds: 3 of 3 shots in one window, 0 joins; --window N caps them"
    )


def test_a_resume_says_what_it_keeps_of_the_record() -> None:
    # A job resumed that has made units takes its window cap and its tiles from the job's record
    # (shot_plan.plan's `recorded`): its lines say so in place of the rule, unless the value is
    # pinned, which is what the line says then; the other lines are the same.
    job = (HD, shots(76), Pins())
    made, peaks = planner.plan(numz_free(24), *job, 64 * GIB, split=True)
    fresh = planner.said(made, peaks, "a card", *job)
    kept = planner.said(made, peaks, "a card", *job, recorded=True)
    words = "kept from the job's record, so that the output stays bit for bit"
    named = ("windows", "encode", "decode")
    for name in named:
        assert words in said_line(kept, name) and words not in said_line(fresh, name)
    assert len(kept) == len(fresh)
    assert [line for line in kept if words not in line] == [
        line for line in fresh if not line.startswith(tuple(f"plan: {name}: " for name in named))
    ]
    # A pin says itself, a record or not.
    pins = Pins(window=17, encode_tile=1856)
    made, peaks = planner.plan(numz_free(24), HD, shots(76), pins, 64 * GIB, split=True)
    said = planner.said(made, peaks, "a card", HD, shots(76), pins, recorded=True)
    assert "pinned by --window" in said_line(said, "windows")
    assert "pinned by --encode-tile" in said_line(said, "encode")
    assert words in said_line(said, "decode")


# The memory a plan is made in, each time a budget of 22.15 GiB, and how its words name it: the
# plan's first line after the card's name, and what a refusal says is there. A driver that
# refuses: one figure, what it reports free, 22.75 GiB, the budget the planner's margin under
# it. A run held to the card: what the plan counts on, 22.15 GiB, which is its budget, no margin
# on top (DESIGN.md, Allocator: the margin is counted inside what torch's allocator is left);
# of which figure, the driver's or the whole card's, with the other, in seedvr2x/memory.py's own
# words; the whole card's wasn't read, is the smaller of the two, or the larger; and what
# torch's limit leaves under it apart from what the plan leaves again, two figures that were
# given as one. Then what the driver reported, in GiB, as both say it; and what memory freed on
# the card could give the plan, in GiB: nothing on a driver that refuses, nor where the card's
# figure isn't the one the plan is held under.
HELD_LINE = ": the run is held to the card's own memory, its driver giving more than the card has"


# The second line of the plan's card on a held run: torch's limit and the reserve, by name.
BUDGET_LINE = (
    "plan: budget: 22.15 GiB, the 28.05 GiB free less torch's limit, 0.80 GiB under it, and the"
    " reserve of 5.10 GiB left to torch's allocator beyond a step's peak; the margin of 0.60 GiB"
    " is in the reserve"
)


# What a refusal under torch's own allocator without the segments adds where the segments would
# run the job: the library's words, which name no option.
PLAIN_HINT = (
    "; torch's own allocator with its expandable segments, a held run's default, which the"
    " allocator's setting in the environment replaces here, would run it"
)


# The same on a driver that refuses: the margin, no reserve.
REFUSING_LINE = (
    ": the margin of 0.60 GiB left to the desktop and other programs; no reserve, the driver"
    " refusing memory the card doesn't have"
)


WORDS = [
    (
        round(22.75 * GIB),
        f"22.75 GiB free, a budget of 22.15 GiB{REFUSING_LINE}",
        "the driver reports 22.75 GiB free",
        22.75,
        0.0,
    ),
    (
        held(28.05, None),
        "28.05 GiB free as its driver reports it to this process (the whole card's, NVML's,"
        f" wasn't read){HELD_LINE}",
        "the plan counts on 22.15 GiB of the 28.05 GiB free as its driver reports it to this"
        " process (the whole card's, NVML's, wasn't read), the run held to the card's own memory",
        28.05,
        0.0,
    ),
    (
        held(30.03, 28.05),
        f"28.05 GiB free on the whole card by NVML{HELD_LINE}",
        "the plan counts on 22.15 GiB of the 28.05 GiB free on the whole card by NVML (its"
        " driver reports 30.03 GiB free to this process), the run held to the card's own memory",
        30.03,
        1.98,
    ),
    (
        held(28.05, 29.20),
        "28.05 GiB free as its driver reports it to this process (the whole card has 29.20 GiB"
        f" free by NVML){HELD_LINE}",
        "the plan counts on 22.15 GiB of the 28.05 GiB free as its driver reports it to this"
        " process (the whole card has 29.20 GiB free by NVML), the run held to the card's own"
        " memory",
        28.05,
        0.0,
    ),
]


def test_the_card_line_leaves_the_drivers_figure_to_its_debug_record() -> None:
    # Design's ruling, baton 3, 2026-10-30: the card line keeps NVML's free memory and why the
    # run is held; "its driver reports N GiB free to this process" is a debug record
    # (card_detail), under --verbose. Held with NVML's figure the smaller: the line lacks the
    # driver's, the record has it. Elsewhere the line is the free memory itself and no record
    # is: the driver's figure the smaller (it stays in the line), the card's not read, a driver
    # that refuses.
    film = shots(31)
    held_to_card = {
        "smaller": held(30.03, 28.05),
        "driver": held(28.05, 29.20),
        "unread": held(28.05, None),
    }
    said: dict[str, str] = {}
    detail: dict[str, str | None] = {}
    for name, reading in held_to_card.items():
        made, peaks = planner.plan(reading, HD, film, Pins(), None)
        said[name] = planner.said(made, peaks, "a card", HD, film, Pins())[0]
        detail[name] = planner.card_detail(peaks)
    assert (
        said["smaller"]
        == f"plan: card: a card, 28.05 GiB free on the whole card by NVML{HELD_LINE}"
    )
    assert detail["smaller"] == (
        "plan: card: the GPU's driver reports 30.03 GiB free to this process, which the run isn't"
        " held to: the whole card has 28.05 GiB free by NVML"
    )
    # The other branch, the driver's figure the smaller, is the free memory: unchanged.
    assert said["driver"] == (
        "plan: card: a card, 28.05 GiB free as its driver reports it to this process (the whole"
        f" card has 29.20 GiB free by NVML){HELD_LINE}"
    )
    assert said["unread"] == (
        "plan: card: a card, 28.05 GiB free as its driver reports it to this process (the whole"
        f" card's, NVML's, wasn't read){HELD_LINE}"
    )
    assert detail["driver"] is None and detail["unread"] is None
    # A card shared under Linux's driver: held, the share the reason the line gives
    # (memory.Reading.shared; DESIGN.md, Allocator, decided on 2026-11-11).
    shared = replace(held_to_card["smaller"], shared="a display active on it")
    made, peaks = planner.plan(shared, HD, film, Pins(), None)
    assert planner.said(made, peaks, "a card", HD, film, Pins())[0] == (
        "plan: card: a card, 28.05 GiB free on the whole card by NVML: the run is held to the"
        " card's own memory, the card shared: a display active on it"
    )
    # A driver that refuses has no such record, nor does an early reading.
    made, peaks = planner.plan(round(22.75 * GIB), HD, film, Pins(), None)
    assert planner.card_detail(peaks) is None
    made, peaks = planner.plan(memory.early(96 * GIB), HD, film, Pins(), None, recorded=Plan(8))
    assert planner.card_detail(peaks) is None


@pytest.mark.parametrize(("free", "line", "there", "driver", "freeable"), WORDS)
def test_each_figure_of_the_free_memory_is_named(
    free: int | Reading, line: str, there: str, driver: float, freeable: float
) -> None:
    # DESIGN.md, Memory planner, Allocator: on a driver that gives memory the card doesn't have
    # the plan counts on less than the driver reports, and "the driver reports X GiB free" of
    # that figure would be false: the plan's first line and its five refusals name each figure
    # for what it is, and a refusal says what would have to be freed on the card for what was
    # asked, where memory freed there can give it, and what it is short of where it can't (the
    # card's figure not read, the driver's the smaller, or more than the two are apart). On a
    # driver that refuses they say what they said. What something takes is said without the
    # planner's margin on a held run, 0.60 GiB less: the plan has none there.
    is_held = isinstance(free, Reading)
    film = shots(31)

    def takes(refusing: str) -> str:
        return refusing if not is_held else f"{(round(float(refusing) * 100) - 60) / 100:.2f}"

    def to_free(short: str) -> str:
        if not is_held:
            return ""
        if float(short) > freeable:
            return f"{short} GiB short; "
        return f"{short} GiB more would have to be freed on the card for it; "

    def freed(short: str) -> str:
        return ", which would have to be freed on the card" if float(short) <= freeable else ""

    margin = "" if is_held else ", and 0.60 of margin"
    # A held reading here is torch's own allocator without the segments (test_shot_plan.held):
    # a refusal says where the segments' plan would run the job (PLAIN_HINT), where it does.
    hint = PLAIN_HINT if is_held else ""
    # The plan is the one of a budget of 22.15 GiB, whatever names the figure; its record keeps
    # what the plan was made in as `free`, what the driver reported and what the card had
    # beside it.
    made, peaks = planner.plan(free, HD, film, Pins(), None)
    assert (made, peaks.budget) == (
        planner.plan(round(22.75 * GIB), HD, film, Pins(), None)[0],
        24427626496 - MARGIN,
    )
    assert peaks.free == peaks.budget + (0 if is_held else MARGIN)
    said = planner.said(made, peaks, "NVIDIA GeForce RTX 5090", HD, film, Pins())
    assert said[0] == f"plan: card: NVIDIA GeForce RTX 5090, {line}"
    # A held run says its budget on a line of its own, torch's limit and the reserve by name;
    # the rest of the plan is the refusing driver's, line for line.
    assert said[1] == BUDGET_LINE if is_held else said[1].startswith("plan: frames: ")
    assert (
        said[1 + is_held :]
        == planner.said(
            *planner.plan(round(22.75 * GIB), HD, film, Pins(), None), "a card", HD, film, Pins()
        )[1:]
    )
    record = peaks.record()
    if isinstance(free, Reading):
        assert (record["free"], record["driver_free"], record["card_free"]) == (
            free.counted,
            free.reported,
            free.card,
        )
        assert record["budget"] == free.counted
    else:
        assert (record["free"], record["driver_free"], record["card_free"]) == (free, free, None)
        # A number is a driver that refuses, which reports the card's free memory.
        assert planner.plan(Reading(free, None, free, False), HD, film, Pins(), None) == (
            made,
            peaks,
        )
    # --window refused, here with --swap-blocks pinned beside it.
    window_refused = refused(free, HD, film, Pins(swap_blocks=0, window=30), None)
    assert window_refused == (
        f"--window 30: its longest window, 17 latents, takes {takes('24.43')} GiB of free VRAM"
        f" with --swap-blocks 0, and {there}: {to_free('1.68')}the longest window that fits is"
        f" --window 12{hint}"
    )
    # What the driver reported is read from either, the plan's lines as a run logs them and a
    # refusal, by the regression's unpinned case, which a small card alone skips
    # (tests/test_regression.py, reported).
    # The card line leaves the driver's figure out where NVML's is the smaller (design's ruling,
    # baton 3, 2026-10-30): the debug record next to it (card_detail) carries it.
    detail = planner.card_detail(peaks)
    logged = "".join(f"12:00:00 {row}\n" for row in said)
    assert test_regression.reported(logged) == (None if detail is not None else driver)
    if detail is not None:
        assert test_regression.reported(f"12:00:00 DEBUG {detail}\n") == driver
    assert test_regression.reported(f"12:00:00 ERROR {window_refused}\n") == driver
    # --swap-blocks refused, on a resume whose recorded window the blocks pinned don't hold:
    # 1.68 GiB short, which memory freed on the card gives where the card's figure is 1.98 GiB
    # under the driver's. (A window of 17 latents: 8 fitted with no block swapped, and a card of
    # 22 GiB holds up to 12 now.)
    recorded = Plan(17, Tiles(1856), Tiles(1088))
    assert refused(free, HD, film, Pins(swap_blocks=0), None, recorded=recorded) == (
        f"--swap-blocks 0: a window of 17 latents, the recorded plan's, takes {takes('24.43')} GiB"
        f" of free VRAM with 0 blocks swapped, and {there}: {to_free('1.68')}the fewest blocks"
        f" that hold it are --swap-blocks 6{hint}"
    )
    # A tile pin refused.
    assert refused(free, HD, film, Pins(decode_tile=1856), None) == (
        f"--decode-tile 1856: the decode in tiles of 1856 px takes {takes('34.76')} GiB of free"
        f" VRAM, the DiT off the GPU, 1.25 of them the room a VAE pass is held to, and {there}:"
        f" {to_free('12.01')}the largest tile that fits is --decode-tile 1088"
    )
    # A size that doesn't fit.
    assert refused(free, (4320, 7680), shots(5), Pins(), None) == (
        f"an output of 7680x4320 takes {takes('40.92')} GiB of free VRAM (its smallest window, 5"
        " latents, with every block of the DiT swapped and the VAE off the GPU: a peak of 40.32"
        f" GiB{margin}), and {there}: {to_free('18.17')}the largest output that fits there is"
        " --resolution 3136, about 5574x3136"
    )
    # A plan that doesn't fit, and a recorded one that no longer does: the shortfall is what
    # would have to be freed.
    assert fits(Plan(12), free, HD, shots(12), None) == (
        f"the encode, untiled, with the DiT's 36 blocks on the GPU, takes {takes('37.98')} GiB of"
        f" free VRAM, 1.25 of them the room a VAE pass is held to, and {there}, 15.23 GiB short"
        f"{freed('15.23')}"
    )
    recorded = Plan(45, Tiles(1856), Tiles(1088))
    assert refused(free, HD, shots(45), Pins(), None, recorded=recorded) == (
        "the recorded plan no longer fits the memory free now: the DiT's phase, a window of 45"
        f" latents with 36 blocks swapped and the VAE off the GPU, takes {takes('23.71')} GiB of"
        f" free VRAM and {there}, 0.96 GiB short{freed('0.96')}"
    )


def test_a_held_run_takes_no_margin_on_top() -> None:
    # DESIGN.md, Allocator (2026-10-12): "On such a driver the planner's 0.6 GiB margin is
    # counted inside the 4.5, which measure the same need: with both, 4K's smallest window is
    # refused on that card beside its usual desktop." What a held run counts on is its budget:
    # a phase that takes it to the byte is planned, and one byte more refused; on a driver that
    # refuses the margin stands over the same peak.
    film = shots(5)
    peak = peaks.dit_peak(padded_size(HD), 5, 0, False)
    pins = Pins(window=5, swap_blocks=0)

    def on_card(counted: int) -> Reading:
        # Torch's own allocator without the segments (PLAIN): no reserve beyond a window's peak, a
        # byte short is refused; this test holds the margin's absence, not the step's reserve.
        return Reading(
            counted + round(5.0 * GIB), None, counted, True, round(0.5 * GIB), memory.PLAIN
        )

    made, predicted = planner.plan(on_card(peak), HD, film, pins, None)
    assert (predicted.dit, predicted.free, predicted.budget) == (peak, peak, peak)
    lines = planner.said(made, predicted, "a card", HD, film, pins)
    assert sum(
        "the DiT's phase bounds the plan, 0.00 GiB under the budget" in line for line in lines
    )
    assert refused(on_card(peak - 1), HD, film, pins, None).startswith("--swap-blocks 0: a window")
    # The same peak on a driver that refuses: its margin over it, to the byte.
    made, predicted = planner.plan(peak + MARGIN, HD, film, pins, None)
    assert (predicted.dit, predicted.free, predicted.budget) == (peak, peak + MARGIN, peak)
    assert refused(peak + MARGIN - 1, HD, film, pins, None).startswith("--swap-blocks 0: a window")
    # A VAE pass keeps its room on both (peaks.VAE_ROOM), which is another need: an untiled
    # decode pinned, the DiT off the GPU.
    tight = Pins(window=5, swap_blocks=36, decode_tile=0)
    decode = peaks.decode_peak(padded_size(HD), output_size(HD), None, 0, 36)
    need = max(decode + peaks.VAE_ROOM, peaks.dit_peak(padded_size(HD), 5, 36, False))
    assert need == decode + peaks.VAE_ROOM
    planner.plan(on_card(need), HD, film, tight, None)
    assert refused(on_card(need - 1), HD, film, tight, None).startswith("--decode-tile 0: ")
    planner.plan(need + MARGIN, HD, film, tight, None)
    assert refused(need + MARGIN - 1, HD, film, tight, None).startswith("--decode-tile 0: ")


def _segments_card(card: float, allocator: str = memory.SEGMENTS) -> Reading:
    """A held run on a card of `card` GiB free (the driver reporting 2 GiB more, as under WSL2),
    counted by memory.counted for `allocator`: under the segments and the pool the figure less
    what torch's limit leaves, nothing else."""
    whole = round(card * GIB)
    plain = Reading(whole + 2 * GIB, whole, whole - memory.HELD_BACK, True, memory.HELD_BACK)
    made = replace(plain, allocator=allocator)
    return replace(made, counted=memory.counted(whole, made))


def test_a_held_plan_says_the_reserve_of_each_step() -> None:
    # DESIGN.md, Allocator, Small cards (decided 2026-11-06): the reserve of 4.5 and the 2.5 retire
    # under the segments, a step is left a reserve by its own peak, max(0.6, 0.33 x P) for the
    # DiT's window and max(1.25, 0.2 x P) for a VAE pass; under cudaMallocAsync the pool's,
    # max(0.85, 0.24 x P) and max(1.25, 0.35 x P). The budget line says the three, in the
    # plan's own figures (Peaks.reserves); the "up to 1.5 times as long" line is gone.
    padded = padded_size(HD)
    # A shot of one latent at 1080p is under each rule's share: the floors both sides.
    small, one = Pins(swap_blocks=36), shots(1)
    for allocator, name, dit_floor in (
        (memory.SEGMENTS, "expandable segments", 0.6),
        (memory.POOL, "cudaMallocAsync's pool", 0.85),
    ):
        reading = _segments_card(30, allocator)
        made, predicted = planner.plan(reading, HD, one, small, None)
        assert predicted.free == predicted.budget == reading.counted
        reserves = predicted.reserves
        assert reserves["dit"] == round(dit_floor * GIB) and reserves["encode"] >= peaks.VAE_ROOM
        lines = planner.said(made, predicted, "a card", HD, one, small)
        assert lines[1] == (
            f"plan: budget: {reading.counted / GIB:.2f} GiB, the 30.00 GiB free less torch's"
            f" limit, {memory.HELD_BACK / GIB:.2f} GiB under it; reserve beyond each step's peak:"
            f" {reserves['dit'] / GIB:.2f} GiB for the DiT's window,"
            f" {reserves['encode'] / GIB:.2f} for the encode, {reserves['decode'] / GIB:.2f} for"
            f" the decode ({name})"
        )
        assert "1.5 times" not in "".join(lines) and "slow" not in "".join(lines[:3])
        assert "the margin of" not in lines[1] and "the reserve of" not in lines[1]
        assert predicted.record()["reserve"] == {
            phase: reserves[phase] for phase in ("encode", "dit", "decode")
        }
    # A long window of a tall shot is over the DiT's floor: 0.33 x P (0.24 under the pool), P the
    # window's peak over the weights it keeps (peaks.dit_working); the segments' is capped at
    # 4.5 GiB, from P of 13.6 GiB.
    long = Pins(window=17, swap_blocks=0)
    for allocator, share in ((memory.SEGMENTS, 0.33), (memory.POOL, 0.24)):
        reading = _segments_card(60, allocator)
        made, predicted = planner.plan(reading, HD, shots(17), long, None)
        working = peaks.dit_working(padded, 17)
        assert predicted.reserves["dit"] == round(share * working) > round(0.85 * GIB)
        text = "".join(planner.said(made, predicted, "a card", HD, shots(17), long))
        assert f"{predicted.reserves['dit'] / GIB:.2f} GiB for the DiT's window" in text
    # Torch's own allocator without the segments keeps the flat 4.5, its margin in the reserve,
    # one line, as before: nothing of the step's rule, nothing slow.
    plain = held(28.05, None)
    made, predicted = planner.plan(plain, HD, one, small, None)
    lines = planner.said(made, predicted, "a card", HD, one, small)
    assert lines[1] == BUDGET_LINE and "1.5 times" not in "".join(lines)
    assert predicted.record()["reserve"] == dict.fromkeys(
        ("encode", "dit", "decode"), memory.RESERVE
    )
    assert predicted.reserve_taken() == dict.fromkeys(("encode", "dit", "decode"), memory.RESERVE)
    # A driver that refuses leaves none.
    made, predicted = planner.plan(round(22.75 * GIB), HD, one, small, None)
    assert predicted.reserve_taken() is None and predicted.record()["reserve"] is None


def test_a_plan_refused_under_torch_s_own_allocator_says_where_the_segments_would_run_it() -> None:
    # The 2.5 GiB reserve and its advice retired with the step's reserve (2026-11-06): a plan
    # refused under torch's own allocator without the segments, a user's setting, says that the
    # default with them would run it, only where its plan does, and words no reserve of 2.50 nor
    # an option the library has no name for. 1080p, a decode in tiles of 1088 px pinned with the DiT
    # off the GPU, refused by a byte with the flat 4.5 (an untiled one, 36.7 GiB, is reserved
    # 0.2 x P past the 4.5 the segments give back: the segments do not run it, and nothing is said).
    film = shots(5)
    pins = Pins(window=5, swap_blocks=36, decode_tile=1088)
    decode = peaks.decode_peak(padded_size(HD), output_size(HD), Tiles(1088), 0, 36)
    need = decode + peaks.VAE_ROOM

    def on_card(counted: int) -> Reading:
        whole = counted + memory.HELD_BACK
        return Reading(whole + 2 * GIB, whole, counted, True, memory.HELD_BACK, memory.PLAIN)

    planner.plan(on_card(need), HD, film, pins, None)
    said = refused(on_card(need - 1), HD, film, pins, None)
    assert said.startswith("--decode-tile 1088: the decode in tiles of 1088 px takes")
    assert said.endswith(PLAIN_HINT) and "2.50" not in said and "reserve of" not in said
    # Where the segments would not run it either, nothing is said of them.
    hopeless = refused(on_card(2 * GIB), HD, film, pins, None)
    assert "expandable segments" not in hopeless and "reserve of" not in hopeless
    untiled = Pins(window=5, swap_blocks=36, decode_tile=0)
    whole = peaks.decode_peak(padded_size(HD), output_size(HD), None, 0, 36) + peaks.VAE_ROOM
    nothing = refused(on_card(whole - 1), HD, film, untiled, None)
    assert nothing.startswith("--decode-tile 0: ") and "expandable segments" not in nothing
    # The segments' own plan of that card is the one the hint speaks of.
    segments = replace(on_card(need - 1), allocator=memory.SEGMENTS)
    segments = replace(segments, counted=segments.counted + memory.RESERVE)
    planner.plan(segments, HD, film, pins, None)


def test_a_recorded_shape_planned_again_releases_the_part_the_refusal_names() -> None:
    # plan_again (DESIGN.md, Memory planner, A plan shapes the output a little, decided on
    # 2026-10-28, with --accept-env-change): what still fits of the recorded shape is kept; the
    # part of the phase that doesn't fit is released to the planner, one at a time, at most three
    # rounds; the command line's own pins stay pinned.
    def again(
        free: int | Reading, recorded: Plan, pins: Pins = NO_PINS, host: int | None = None
    ) -> tuple[Plan, Peaks]:
        with pytest.raises(PlanError) as refused:
            planner.plan(free, HD, shots(76), pins, host, recorded=recorded)
        return planner.plan_again(
            free, HD, shots(76), pins, host, recorded=recorded, refused=refused.value
        )

    def shape(made: Plan) -> tuple[int | None, Tiles | None, Tiles | None]:
        return made.window, made.encode_tiles, made.decode_tiles

    # The encode alone doesn't fit, a little short untiled (on 23 GiB free: the lean attention and
    # the reserve by the step let the untiled encode in at 24): its tiles are the planner's, the
    # window and the decode's tiles as recorded.
    untiled = Plan(17, None, Tiles(1088))
    made, _ = again(our_free(23), untiled)
    # (A driver that refuses takes no room rung, which was measured under the segments alone: the
    # tiles at the room of 1.25 GiB, 1792 px for the encode and, the two together no longer fitting
    # the recorded decode's 1088, 1024 px for the decode: not the 1856 and 1088 the room lowered
    # to 0.5 gave.)
    assert shape(made) == (17, Tiles(1792), Tiles(1024))
    # The DiT's phase doesn't fit: the cap is the planner's, the tiles recorded kept.
    # (A window of 45: the lean attention's 16 GB card holds 20 and more, the cap the planner
    # gives back is 21.)
    made, _ = again(our_free(16), Plan(45, Tiles(768), Tiles(640)))
    assert shape(made) == (21, Tiles(768), Tiles(640))
    # The encode refused first, then the DiT's phase, then the decode, one part each round.
    made, _ = again(our_free(16), Plan(17, Tiles(1856), Tiles(1088)))
    assert shape(made) == (17, Tiles(1024), Tiles(768))
    # The system's memory: the tiles go before the window (the planner's own order).
    made, _ = again(our_free(24), Plan(17, Tiles(1856), Tiles(1088)), host=8 * GIB)
    assert shape(made) == (17, Tiles(896), Tiles(704))
    # What the command line pins is kept: the refusal given back, the shape untouched, whatever
    # else would fit.
    for pins in (Pins(window=45), Pins(swap_blocks=36, window=45)):
        with pytest.raises(PlanError) as pinned:
            again(our_free(16), Plan(45, Tiles(768), Tiles(640)), pins)
        assert pinned.value.phase == "dit" and pinned.value.shortfall is not None
    with pytest.raises(PlanError) as tiles:
        again(our_free(16), Plan(17, Tiles(1856), Tiles(1088)), Pins(encode_tile=1856))
    assert tiles.value.phase == "encode" and tiles.value.shortfall is not None
    # A pin on one part leaves the others to be released: here the window is the user's, the
    # encode's tiles are released and the decode's are not needed.
    made, _ = again(our_free(23), untiled, Pins(window=17))
    assert shape(made) == (17, Tiles(1792), Tiles(1024))
    # Nothing fits whatever is released: the last round's refusal, a new job's (no shortfall: it
    # has no recorded shape to keep).
    with pytest.raises(PlanError) as nothing:
        again(our_free(8), Plan(17, None, None))
    assert nothing.value.shortfall is None and nothing.value.phase == "decode"


def test_the_unpinned_case_skips_a_small_card_alone() -> None:
    # tests/test_regression.py's unpinned case, on a card whose plan can't make the untiled
    # master (test_regression.unpinned): skipped where the driver reports under 40 GiB free,
    # failed where it reports 40 or more, a plan that tiles or is refused there being wrong;
    # and failed where the log doesn't say, a refusal for the system memory alone. By the
    # planner's own lines and refusals, as a --plan run logs them (without --verbose: the time
    # of day, a warning's or an error's level, the message).
    def logged(lines: list[str], level: str = "") -> str:
        return "".join(f"12:00:00 {level + ' ' if level else ''}{line}\n" for line in lines)

    def passes(lines: list[str]) -> list[str]:
        return [line for line in lines if line.startswith(("plan: encode: ", "plan: decode: "))]

    job = (HD, shots(12), Pins())
    made, peaks = planner.plan(our_free(24), *job, None, split=True)
    lines = planner.said(made, peaks, "a card", *job)
    tiled = [line for line in passes(lines) if "tiles of" in line]
    assert tiled and test_regression.reported(logged(lines)) == 22.57
    with pytest.raises(pytest.skip.Exception) as skipped:
        test_regression.unpinned(logged(lines), f"the plan tiles a pass: {tiled[0]}")
    assert str(skipped.value).startswith(
        "the driver reports 22.57 GiB free, under the 40 a card makes the untiled master"
        " from: the plan tiles a pass: plan: encode: tiles of "
    )
    # 48 GB: 46.57 GiB reported, and both passes untiled, as numz ran them: no verdict to give.
    made, peaks = planner.plan(our_free(48), *job, None, split=True)
    assert (made.encode_tiles, made.decode_tiles) == (None, None)
    lines = planner.said(made, peaks, "a card", *job)
    assert test_regression.reported(logged(lines)) == 46.57
    assert [line.split(",")[0] for line in passes(lines)] == [
        "plan: encode: untiled",
        "plan: decode: untiled",
    ]
    # A plan that tiles where 46.57 GiB are reported, a larger job's here: failed.
    larger = (UHD, shots(12), Pins())
    made, peaks = planner.plan(our_free(48), *larger, None)
    lines = planner.said(made, peaks, "a card", *larger)
    tiled = [line for line in passes(lines) if "tiles of" in line]
    assert tiled
    with pytest.raises(pytest.fail.Exception) as failed:
        test_regression.unpinned(logged(lines), f"the plan tiles a pass: {tiled[0]}")
    assert str(failed.value).startswith(
        "the driver reports 46.57 GiB free, 40 or more, where the plan must run both passes"
        " untiled: the plan tiles a pass: plan: "
    )
    # Refused: skipped on a small card, failed on a large one, by the refusal's own figure.
    small = refused(round(6.5 * GIB), *job, None)
    with pytest.raises(pytest.skip.Exception, match=r"the driver reports 6\.50 GiB free, under"):
        test_regression.unpinned(logged([small], "ERROR"), f"the plan is refused: {small}")
    # (10240x5760: the 7680x4320 output that was refused on this card takes 40.92 GiB now.)
    large = refused(our_free(48), (5760, 10240), shots(5), Pins(), None)
    with pytest.raises(pytest.fail.Exception, match=r"the driver reports 46\.57 GiB free, 40 or"):
        test_regression.unpinned(logged([large], "ERROR"), f"the plan is refused: {large}")
    # A refusal for the system memory names no figure of the driver's: failed, saying so.
    host = refused(our_free(48), *job, 2 * GIB)
    assert test_regression.reported(logged([host], "ERROR")) is None
    with pytest.raises(pytest.fail.Exception, match="what the driver reports free isn't in the"):
        test_regression.unpinned(logged([host], "ERROR"), f"the plan is refused: {host}")


def test_the_dit_s_fit_is_the_lean_attention_s() -> None:
    # DESIGN.md, Memory planner, Per-phase peaks: the planner holds one fit for the DiT, the lean
    # attention's under PyTorch's SDPA (FlashAttention 2, which peaked the same to the MiB, is no
    # longer run): 61.243 KiB a token, 0.996 MiB a window and the 0.081 GiB that bounds every
    # window, the 1-latent 1080p one's. Each prediction is at or over its measure and within
    # 0.081 GiB of it.
    assert peaks.DIT_ATTENTION == round(0.081 * GIB)
    padded = {1104: OURS_1080, 2192: OURS_4K}
    assert len(LEAN_WINDOWS) == 19
    margins: list[tuple[float, int, int, int]] = []
    for rows, latents, swapped, vae, peak in LEAN_WINDOWS:
        fitted = peaks.dit_peak(padded[rows], latents, swapped, vae)
        assert fitted >= peak, (rows, latents, swapped)
        margins.append((gib(fitted - peak), rows, latents, swapped))
    # The least margin is 0.0012 GiB, at 1 latent of 1080p (whose peak is elsewhere in the step,
    # 0.073 GiB over the line, the 0.081 of the constant bounding it); every other window has
    # 0.0739 to 0.0745 GiB left, the constant less the fit's residual.
    least = min(margins)
    assert round(least[0], 4) == 0.0012 and least[1:3] == (1104, 1)
    others = [m[0] for m in margins if m[2] > 1 or m[1] == 2192]
    assert 0.0739 <= round(min(others), 4) and round(max(others), 4) <= 0.0745
    # 12 latents at 1080p, no block swapped, the VAE's weights there: 21.90 GiB.
    assert peaks.dit_peak(OURS_1080, 12) == 23_516_133_819
    # The VAE's weights away.
    apart = peaks.dit_peak(OURS_1080, 6, 0, False)
    assert peaks.dit_peak(OURS_1080, 6, 0, True) - apart == peaks.VAE_WEIGHTS
    # 1080p on a 24 GB card, a shot of 301 frames: windows of 39 latents with 32 blocks swapped;
    # the plan's record names no kernel and its lines none.
    free, job = our_free(24), (HD, shots(76), Pins(), None)
    made, predicted = planner.plan(free, *job)
    assert bare(made) == Plan(39, Tiles(1856), Tiles(1088), 32, 1, 0, vae_in_dit=False)
    assert predicted.dit == peaks.dit_peak(OURS_1080, 39, 32, False)
    assert "attention" not in predicted.record()
    said = planner.said(made, predicted, "a card", HD, shots(76), Pins())
    assert not [line for line in said if "attention" in line or "FlashAttention" in line]
    # The same window where the card was full: a shot of 1,197 frames in windows of 40 latents
    # with 33 blocks swapped (20 with 35 and 19 with 36 before the lean attention).
    made, _ = planner.plan(free, HD, shots(300), Pins(), None)
    assert (made.window, made.swapped) == (40, 33)
    # 4K's smallest window, 5 latents, on a card emulated at 24 GB, 22.51 GiB free: planned.
    emulated = 24_168_824_832
    plan, _ = planner.plan(emulated, UHD, shots(5), Pins(), None)
    assert plan.window == 5


# ---- The floor, last rungs (DESIGN.md, Allocator, Small cards): the plan's lines ---------------

TAIL_LINE = (
    "plan: the decoder's tail: its last blocks a frame at a time, taken because the plain decode"
    " does not fit this card: its peak about a quarter lower, the decode about 1.1 times as long,"
    " its frames other than the plain decoder's at 512, 576 and 960 px tiles (the same in every"
    " run)"
)
ROOM_LINE = (
    "plan: the VAE's least room: taken because the plan does not fit this card without it: the"
    " VAE passes' reserve floor lowered to 0.50 GiB (a pass whose fifth of its working memory is"
    " more keeps that), the decode about 1.2 times as long"
)


def test_a_plan_that_takes_no_rung_says_none() -> None:
    made, predicted = planner.plan(_segments_card(9.0), HD, shots(19), NO_PINS, None)
    assert not made.tail_frames and made.vae_floor is None
    lines = planner.said(made, predicted, "a card", HD, shots(19), NO_PINS)
    assert not [line for line in lines if "decoder's tail" in line or "least room" in line]
    assert any(
        line.startswith("plan: the VAE's room: ")
        and "at least 1.25 GiB of the budget, the reserve of each pass above" in line
        for line in lines
    )


def test_a_plan_says_the_tail_it_takes_and_what_it_costs() -> None:
    made, predicted = planner.plan(_segments_card(8.0), HD, shots(19), NO_PINS, None)
    assert made.tail_frames and made.vae_floor is None
    lines = planner.said(made, predicted, "a card", HD, shots(19), NO_PINS)
    place = next(
        number for number, line in enumerate(lines) if line.startswith("plan: the VAE's room")
    )
    assert lines[place + 1] == TAIL_LINE
    assert "at least 1.25 GiB of the budget" in lines[place]
    assert "least room" not in "".join(lines)


def test_a_plan_says_the_room_it_takes_besides_the_tail() -> None:
    made, predicted = planner.plan(_segments_card(6.25), HD, shots(19), NO_PINS, None)
    assert made.tail_frames and made.vae_floor == peaks.VAE_ROOM_LEAST
    lines = planner.said(made, predicted, "a card", HD, shots(19), NO_PINS)
    place = next(
        number for number, line in enumerate(lines) if line.startswith("plan: the VAE's room")
    )
    assert "at least 0.50 GiB of the budget" in lines[place]
    assert lines[place + 1 : place + 3] == [TAIL_LINE, ROOM_LINE]
    # The record holds both: the tail with the shape, the floor under memory.
    record = made.record()
    assert record["tail_frames"] is True
    assert record["memory"]["vae_floor"] == peaks.VAE_ROOM_LEAST


def test_a_recorded_plan_with_the_tail_planned_again_keeps_it() -> None:
    # plan_again (a resume's shape that no longer fits, --accept-env-change) keeps what shapes the
    # output: the tail as recorded, whatever tiles are released.
    # (A window of 7 latents: the plan's own on this card, the DiT's phase fits and the decode
    # is what refuses; a window of 19 was refused in the DiT's phase first. A card of 6.4 GiB:
    # on 6.25, under the DiT's share of 0.33 of P, the window of 7 is refused there first.)
    recorded = Plan(7, Tiles(512), Tiles(768), tail_frames=True)
    card = _segments_card(6.4)
    with pytest.raises(PlanError) as refused:
        planner.plan(card, HD, shots(19), NO_PINS, None, recorded=recorded)
    assert refused.value.phase == "decode"
    made, _ = planner.plan_again(
        card, HD, shots(19), NO_PINS, None, recorded=recorded, refused=refused.value
    )
    assert made.tail_frames and made.decode_tiles != Tiles(768)
    # A recorded plan without it is planned again without it.
    # (On a card of 9 GiB: a plain decode does not fit the 6.4 GiB card at all, only the tail
    # does, so that card would refuse the plain shape whatever is released.)
    plain = Plan(7, Tiles(512), Tiles(768))
    card = _segments_card(9.0)
    with pytest.raises(PlanError) as refused:
        planner.plan(card, HD, shots(19), NO_PINS, None, recorded=plain)
    made, _ = planner.plan_again(
        card, HD, shots(19), NO_PINS, None, recorded=plain, refused=refused.value
    )
    assert not made.tail_frames

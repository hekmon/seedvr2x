"""--compile, the VAE compiled (runtime/compiling.py, planner.py's compiled passes; DESIGN.md,
Memory planner, "torch.compile: the VAE, opt-in, in v0.1.0; the DiT, not", decided on 2026-10-24):
the planner holding a compiled pass to 1.1 times its tile's plain peak and keeping to the shapes
measured, the shapes warmed, inductor's deterministic mode kept on through every trace, the
causal convolutions left out of the graph, the order of the load, the cache in the job's
directory, and the plan's words. On the CPU: nothing is compiled on a GPU here; a tiny module is
compiled on the CPU where a C++ compiler is there."""

import json
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf
from test_planner import HD, UHD, our_free, shots

from seedvr2x.runtime import compiling, model, planner, resume
from seedvr2x.runtime.job import JobError, Shot, output_size, padded_size
from seedvr2x.runtime.model import COMPUTE_DTYPE, Models
from seedvr2x.runtime.plan import Compilation, Plan, Shapes, Tiles
from seedvr2x.runtime.planner import GIB, Pins, PlanError
from seedvr2x.vendor.models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d
from seedvr2x.vendor.models.video_vae_v3.modules.types import MemoryState

# The tile peaks the planner holds (planner.MEASURED, planner-limits.md "seedvr2x's tiled passes:
# every tile's peak"), above what is on the GPU before the pass, in GiB: 1024-px tiles and the
# largest the compiled VAE may meet, as the study measured them (compile.md): the encode's
# 1856-px tile and the decode's 1152-px one.
ENCODE_1024, DECODE_1024 = 10.47, 17.46
ENCODE_1856, DECODE_1152 = 30.46, 21.99


def plain_and_compiled(
    free: int,
    target: tuple[int, int],
    lengths: tuple[int, ...] = (76,),
    pins: Pins | None = None,
) -> tuple[tuple[Plan, planner.Peaks], tuple[Plan, planner.Peaks]]:
    """The same job planned without and with the VAE compiled, on a host with memory to spare."""
    job = (free, target, shots(*lengths), pins or Pins(), None)
    return planner.plan(*job), planner.plan(*job, compile_vae=True)


def test_constants() -> None:
    # DESIGN.md, same section: a compiled pass 1.04 to 1.10 times plain's peak, held to 1.1; encode
    # tiles up to 1856 px or the 1080p frame whole, decode tiles up to 1152 px; compilation 4 to 6
    # min per process, counted at the upper end. The 1080p frame padded is 1104x1920.
    assert planner.COMPILE_PEAK == 1.1
    assert (planner.COMPILE_ENCODE_TILE_MAX, planner.COMPILE_DECODE_TILE_MAX) == (1856, 1152)
    assert planner.COMPILE_ENCODE_WHOLE == padded_size(HD) == (1104, 1920)
    assert (planner.COMPILE_MIN, planner.COMPILE_MAX) == (4 * 60, 6 * 60)


def test_a_compiled_pass_peaks_at_1_1_times_the_tiles_plain_peak() -> None:
    # The tile's peak above what is on the GPU before the pass is 1.1 times its plain one, the
    # table's (planner.MEASURED); the weights and the encode's 0.4 GiB of margin (ENCODE_EXTRA)
    # are not scaled, as the study's run scaled the fit alone (research/scripts/compile_run.py,
    # --vae-peak-factor). Every tile, at 1080p and 4K, with the DiT's part on the GPU counted
    # as for a plain pass.
    fixed_encode = planner.ENCODE_EXTRA + planner.VAE_WEIGHTS
    for target in (HD, UHD):
        padded, output = padded_size(target), output_size(target)
        for size in range(512, 1857, 64):
            tiles = Tiles(size)
            for resident, swapped in ((None, 0), (5, 28), (0, 36)):
                kept = planner.dit_on_gpu(resident, swapped)
                plain = planner.encode_peak(padded, tiles, resident, swapped) - fixed_encode - kept
                compiled = planner.encode_peak(padded, tiles, resident, swapped, True)
                assert compiled - fixed_encode - kept == pytest.approx(1.1 * plain, abs=1)
                plain = planner.decode_peak(padded, output, tiles, resident, swapped) - kept
                compiled = planner.decode_peak(
                    padded, output, tiles, resident, swapped, compiled=True
                )
                assert compiled - planner.VAE_WEIGHTS - kept == pytest.approx(
                    1.1 * (plain - planner.VAE_WEIGHTS), abs=1
                )
    # Untiled passes too (a small frame): the fit scaled, the correction's term not.
    small = (576, 960)
    plain = planner.decode_peak(small, small, None, split=True)
    compiled = planner.decode_peak(small, small, None, split=True, compiled=True)
    assert compiled - plain == pytest.approx(
        0.1 * (planner._fit(small, None, 1)),  # pyright: ignore[reportPrivateUsage]
        abs=1,
    )


def test_a_plan_with_the_vae_compiled_chooses_smaller_tiles_where_1_1_times_doesnt_fit() -> None:
    # On cards where the plain tiles fit at their peak and not at 1.1 times it, the compiled plan
    # takes smaller ones; never larger, never beyond the measured shapes, and every pass of it
    # fits the budget at the compiled peak, its room counted.
    smaller = 0
    for card in range(18, 49, 2):
        (plain, _), (compiled, peaks) = plain_and_compiled(our_free(card), HD)
        assert compiled.compile_vae and not plain.compile_vae
        assert planner.in_compile_range(padded_size(HD), "encode", compiled.encode_tiles)
        assert planner.in_compile_range(padded_size(HD), "decode", compiled.decode_tiles)
        for phase in ("encode", "decode"):
            assert peaks.need(phase) + peaks.free - peaks.budget <= peaks.free
        for ours, theirs in (
            (compiled.encode_tiles, plain.encode_tiles),
            (compiled.decode_tiles, plain.decode_tiles),
        ):
            if theirs is not None and ours is not None:
                assert ours.size <= theirs.size
                smaller += ours.size < theirs.size
    assert smaller > 0


def test_the_compiled_plans_predicted_peaks_are_its_passes_compiled_peaks() -> None:
    # What the plan records and judges (Peaks.encode, Peaks.decode) is the compiled pass at the
    # tiles it chose and the DiT it keeps on the GPU there; the plain plan's are the plain ones.
    for card in (24, 32, 48, 96):
        for compile_vae in (False, True):
            made, peaks = planner.plan(
                our_free(card), HD, shots(76), Pins(), None, compile_vae=compile_vae
            )
            padded, output = padded_size(HD), output_size(HD)
            assert peaks.encode == planner.encode_peak(
                padded, made.encode_tiles, made.resident_encode, made.swapped, compile_vae
            )
            assert peaks.decode == planner.decode_peak(
                padded,
                output,
                made.decode_tiles,
                made.resident_decode,
                made.swapped,
                False,
                compile_vae,
            )


def test_a_compiled_decode_is_always_tiled_and_in_the_measured_range() -> None:
    # On the 96 GB card a plain 1080p job decodes untiled and encodes untiled: compiled, the
    # encode stays untiled (the 1080p frame whole is measured) and the decode takes tiles of
    # 1152 px or less, the largest that fit.
    (plain, _), (compiled, _) = plain_and_compiled(our_free(96), HD)
    assert plain.encode_tiles is None and plain.decode_tiles is None
    assert compiled.encode_tiles is None
    assert compiled.decode_tiles is not None and compiled.decode_tiles.size == 1152
    # 4K is beyond the frame measured: the encode is tiled too, up to 1856 px.
    (plain, _), (compiled, _) = plain_and_compiled(our_free(96), UHD)
    assert compiled.compile_vae
    assert compiled.encode_tiles is not None and compiled.encode_tiles.size <= 1856
    assert compiled.decode_tiles is not None and compiled.decode_tiles.size <= 1152
    assert plain.decode_tiles is None or plain.decode_tiles.size > compiled.decode_tiles.size


def test_the_range_is_the_1080p_frame_as_a_whole_not_its_area() -> None:
    # The more conservative reading of "the 1080p frame whole" (PROVISIONAL, planner.COMPILE_
    # ENCODE_WHOLE): rows and columns each no larger than the padded 1080p frame's. A portrait
    # frame of the same area is another shape, and so is 1440p.
    inside = planner.in_compile_range
    assert inside((1104, 1920), "encode", None) and inside((752, 1280), "encode", None)
    assert not inside((1920, 1104), "encode", None) and not inside((1456, 2576), "encode", None)
    assert inside((1920, 1104), "encode", Tiles(1856))
    assert not inside((1920, 1104), "encode", Tiles(1920))
    assert inside((2192, 3840), "decode", Tiles(1152)) and inside(
        (2192, 3840), "decode", Tiles(512)
    )
    assert not inside((2192, 3840), "decode", Tiles(1216))
    assert not inside((1104, 1920), "decode", None)
    assert inside((1104, 1920), "dit", None)


def refusal(
    free: int, target: tuple[int, int], pins: Pins, lengths: tuple[int, ...] = (76,)
) -> str:
    """The message of a job refused compiled where the plain plan is made."""
    planner.plan(free, target, shots(*lengths), pins, None)
    with pytest.raises(PlanError) as refused:
        planner.plan(free, target, shots(*lengths), pins, None, compile_vae=True)
    return str(refused.value)


def test_a_pass_pinned_beyond_the_measured_range_is_refused_with_its_advice() -> None:
    # A pin is taken exactly: a decode tile of 1280 px, an untiled decode, an encode tile of 1920,
    # an untiled encode at 4K. The job is refused, never run plain (DESIGN.md, decided on
    # 2026-10-29): the pass, its tiles and the shapes measured, then the advice, each on a line of
    # its own: the same command without --compile, and the pin within the shapes.
    free = our_free(96)
    for pins, target, phase, pin, where, within in (
        (Pins(decode_tile=1280), HD, "decode", 1280, "in tiles of 1280 px", 1152),
        (Pins(decode_tile=0), HD, "decode", 0, "untiled", 1152),
        (Pins(encode_tile=1920), UHD, "encode", 1920, "in tiles of 1920 px", 1856),
        (Pins(encode_tile=0), UHD, "encode", 0, "untiled", 1856),
    ):
        lines = refusal(free, target, pins).split("\n")
        assert lines[0].startswith(
            f"--{phase}-tile {pin}: the VAE compiled has the {phase} {where} beyond the shapes"
        )
        assert planner._compile_range_said(phase) in lines[0]  # pyright: ignore[reportPrivateUsage]
        assert lines[1:] == [
            "The same command without --compile fits",
            f"The same command with --{phase}-tile {within} fits, within the shapes measured",
        ]
    # Both passes pinned beyond them: the plan stops at the first it meets (the decode), and the
    # refusal names the other pass too, on a line of its own, then gives the advice for both pins.
    lines = refusal(free, UHD, Pins(encode_tile=2048, decode_tile=1280)).split("\n")
    assert lines[0].startswith("--decode-tile 1280: the VAE compiled has the decode in tiles of")
    assert lines[1] == (
        "--encode-tile 2048: the VAE compiled has the encode in tiles of 2048 px beyond the shapes"
        f" measured ({planner._compile_range_said('encode')})"  # pyright: ignore[reportPrivateUsage]
    )
    assert lines[2:] == [
        "The same command without --compile fits",
        "The same command with --encode-tile 1856 fits, within the shapes measured",
        "The same command with --decode-tile 1152 fits, within the shapes measured",
    ]
    lines = refusal(free, UHD, Pins(encode_tile=1920, decode_tile=1280)).split("\n")
    assert lines[-2:] == [
        "The same command with --encode-tile 1856 fits, within the shapes measured",
        "The same command with --decode-tile 1152 fits, within the shapes measured",
    ]
    # A pass pinned within the range is taken, compiled.
    made, _ = planner.plan(
        free, HD, shots(76), Pins(encode_tile=1024, decode_tile=1024), None, compile_vae=True
    )
    assert made.compile_vae
    assert (made.encode_tiles, made.decode_tiles) == (Tiles(1024), Tiles(1024))


def test_a_4k_job_pinned_untiled_is_refused_on_every_card_that_holds_it() -> None:
    # 4K is beyond the 1080p frame whole, so an untiled encode is a pin beyond the shapes
    # whatever the card; the pin advised fits that card, compiled.
    for card in (96, 88):
        lines = refusal(our_free(card), UHD, Pins(encode_tile=0)).split("\n")
        assert lines[0].startswith("--encode-tile 0: the VAE compiled has the encode untiled")
        advised = int(lines[-1].split("--encode-tile ")[1].split(" ")[0])
        made, _ = planner.plan(
            our_free(card), UHD, shots(76), Pins(encode_tile=advised), None, compile_vae=True
        )
        assert made.compile_vae and made.encode_tiles == Tiles(advised)


def test_a_pin_in_the_range_that_fits_plain_and_not_compiled_is_refused_with_its_advice() -> None:
    # The compiled pass is held to 1.1 times the plain peak: on this card a pinned decode tile
    # fits plain and not compiled (the HD job, 25 GB). The refusal gives the plan's own sentence,
    # the shapes measured (which it doesn't say), the same command without --compile, and the
    # largest tile that fits compiled.
    lines = refusal(our_free(25), HD, Pins(decode_tile=1152)).split("\n")
    assert lines[0].startswith("--decode-tile 1152: the decode in tiles of 1152 px takes ")
    measured = [planner._compile_range_said(phase) for phase in ("encode", "decode")]  # pyright: ignore[reportPrivateUsage]
    assert lines[1] == "--compile compiles only the shapes measured: " + "; ".join(measured)
    assert lines[2:] == [
        "The same command without --compile fits",
        "The same command with --decode-tile 1024 fits, within the shapes measured",
    ]


def test_a_frame_too_small_for_a_tile_has_no_compiled_decode_and_is_refused() -> None:
    # A decode can't be untiled compiled, and a frame under 576 px a side has no tile to make
    # (planner._tile_sizes): refused, with no pin to advise.
    lines = refusal(our_free(96), (96, 128), Pins(), (3,)).split("\n")
    assert lines[0].startswith("the VAE compiled has no decode pass within the shapes measured")
    assert lines[1:] == ["The same command without --compile fits"]


def test_a_job_that_fits_no_card_plain_is_refused_as_it_was() -> None:
    # --compile adds no refusal of its own: the plain job's is the one given.
    free = round(3 * GIB)
    with pytest.raises(PlanError) as plain:
        planner.plan(free, HD, shots(76), Pins(), None)
    with pytest.raises(PlanError) as compiled:
        planner.plan(free, HD, shots(76), Pins(), None, compile_vae=True)
    assert str(compiled.value) == str(plain.value)


def test_a_recorded_plan_keeps_its_own_compile_flag() -> None:
    # The units made are what they are: a record with compile_vae is planned compiled whatever
    # the command line says, one without never is. The recorded tiles are in the range.
    free = our_free(48)
    recorded = Plan(17, Tiles(1088), Tiles(1024), compile_vae=True)
    for asked in (False, True):
        made, _ = planner.plan(
            free, HD, shots(76), Pins(), None, recorded=recorded, compile_vae=asked
        )
        assert made.compile_vae
        assert (made.window, made.encode_tiles, made.decode_tiles) == (17, Tiles(1088), Tiles(1024))
    plain = replace(recorded, compile_vae=False)
    for asked in (False, True):
        made, _ = planner.plan(free, HD, shots(76), Pins(), None, recorded=plain, compile_vae=asked)
        assert not made.compile_vae
    # Its peaks are the compiled ones: a card the plain shape fits and the compiled doesn't.
    assert planner.decode_peak(
        padded_size(HD), output_size(HD), Tiles(1024), compiled=True
    ) > planner.decode_peak(padded_size(HD), output_size(HD), Tiles(1024))


def test_a_recorded_compiled_plan_beyond_the_range_is_no_plans() -> None:
    # check_recorded: a record with compile_vae and a pass the planner never makes compiled
    # (edited) is refused, naming it; a plain record of the same passes is a plan.
    for recorded, target, where in (
        (Plan(17, Tiles(1088), Tiles(1280), compile_vae=True), HD, "decode"),
        (Plan(17, Tiles(1088), None, compile_vae=True), HD, "decode"),
        (Plan(17, Tiles(1920), Tiles(1024), compile_vae=True), UHD, "encode"),
        (Plan(17, None, Tiles(1024), compile_vae=True), UHD, "encode"),
    ):
        with pytest.raises(PlanError, match=rf"compile_vae is true and its {where}"):
            planner.check_recorded(recorded, shots(40), target)
        planner.check_recorded(replace(recorded, compile_vae=False), shots(40), target)
    planner.check_recorded(Plan(17, None, Tiles(1152), compile_vae=True), shots(40), HD)
    planner.check_recorded(Plan(17, Tiles(1856), Tiles(512), compile_vae=True), shots(40), UHD)


def test_a_plan_planned_again_keeps_the_recorded_compile_flag() -> None:
    # plan_again (--accept-env-change's re-plan) pins the recorded tiles and keeps compile_vae: a
    # job whose units were made compiled is never planned plain around them.
    recorded = Plan(17, Tiles(1088), Tiles(1024), compile_vae=True)
    job = (our_free(48), HD, shots(76), Pins(), None)
    with pytest.raises(PlanError) as refused:
        planner.plan(round(12 * GIB), HD, shots(76), Pins(), None, recorded=recorded)
    made, _ = planner.plan_again(
        round(26 * GIB), *job[1:], recorded=recorded, refused=refused.value
    )
    assert made.compile_vae


def test_the_calibrations_passes_are_among_the_kinds_warmed() -> None:
    # The calibration times a pass of FIRST frames and one of FRAMES, in the plan's tiles: their
    # slice kinds are warmed whatever the job's shots (planner.COMPILE_CALIBRATED), or the first
    # would time a compilation.
    from seedvr2x.runtime import calibration

    kinds = {
        min((frames - 1) // 4 + 1, planner.COMPILE_LATENTS)
        for frames in (calibration.FIRST, calibration.FRAMES)
    }
    assert kinds == set(planner.COMPILE_CALIBRATED)
    plan = Plan(17, None, Tiles(1088), compile_vae=True)
    assert set(planner.compile_shapes(plan, HD, shots(76), 1).latents) >= kinds


def test_the_shapes_warmed_are_the_plans_passes_at_the_slice_kinds_of_the_job() -> None:
    # One shape per distinct tile of a tiled pass (the grid's: the tiles cut off at the picture's
    # edges) or the padded frame; the shots' lengths shown by their shortest of each kind: 1
    # latent is one slice of 1 frame, 2 one of 5 frames, 3 or more a first slice of 5 and the
    # next ones of 4 (model.encode_slices, model.latent_slices). The runs of --quality's level
    # count, each fed from its own frame.
    plan = Plan(17, None, Tiles(1088), compile_vae=True)
    found = planner.compile_shapes(plan, HD, shots(1, 2, 3, 76), 1)
    assert found.padded == (1104, 1920) and found.encode == ((1104, 1920),)
    assert found.latents == (1, 2, 3)
    # 1104 rows by 1920 columns in tiles of 1088 px sharing 64, a stride of 1024: two tiles each
    # way, 1088 then what the stride leaves, 80 rows and 896 columns; the largest first.
    assert found.decode == ((1088, 1088), (1088, 896), (80, 1088), (80, 896))
    # Only the shortest kinds the job has, and the calibration's (2 and 3 latents: a pass of 5
    # frames, and of 13).
    assert planner.compile_shapes(plan, HD, shots(40, 76), 1).latents == (2, 3)
    assert planner.compile_shapes(plan, HD, shots(1, 76), 1).latents == (1, 2, 3)
    # A shot of 1 frame, its only run: 1 latent. One of 8 frames, run at "maximum" (4 runs, from
    # its frames 0, 1, 2 and 3): 8, 7, 6 and 5 frames, padded to 9, 9, 9 and 5: 3, 3, 3 and 2.
    assert planner.compile_shapes(plan, HD, [Shot(0, 1)], 4).latents == (1, 2, 3)
    assert planner.compile_shapes(plan, HD, [Shot(0, 8)], 4).latents == (2, 3)
    # An untiled pass is the whole frame.
    whole = planner.compile_shapes(Plan(17, None, None), UHD, shots(40), 1)
    assert whole.encode == whole.decode == ((2192, 3840),) and whole.padded == (2192, 3840)


# ----------------------------------------------------------------------------- the plan's words


def lines(plan: Plan, peaks: planner.Peaks, *, target: tuple[int, int] = HD) -> str:
    return "\n".join(
        planner.said(
            plan,
            peaks,
            "a card",
            target,
            shots(76),
            Pins(),
            numz_padding=False,
        )
    )


def test_the_plan_says_the_compilation_and_its_time() -> None:
    free = our_free(96)
    made, peaks = planner.plan(free, HD, shots(76), Pins(), None, compile_vae=True)
    said = lines(made, peaks)
    assert (
        "plan: compile: the VAE compiled (--compile), each pass held to 1.1 times its plain peak,"
        " in the shapes measured only; compiling it takes about 4 to 6 min per process, paid"
        " before the DiT loads, at every start and resume, its cache kept in the job's directory"
        in said
    )
    # No line without the option, and none where the plan is plain and nothing was asked.
    plain, plain_peaks = planner.plan(free, HD, shots(76), Pins(), None)
    assert "compile" not in lines(plain, plain_peaks)


# ----------------------------------------------------------------------------- keep_deterministic


@pytest.fixture
def inductor_state() -> Iterator[Any]:
    """torch's use_deterministic_algorithms and inductor's flag as they were, put back."""
    config: Any = pytest.importorskip("torch._inductor.config")
    function, flag, mode = (
        torch.use_deterministic_algorithms,
        config.deterministic,
        torch.are_deterministic_algorithms_enabled(),
    )
    config.deterministic = False
    yield config
    torch.use_deterministic_algorithms = function
    config.deterministic = flag
    torch.use_deterministic_algorithms(mode)


def test_inductors_deterministic_mode_stays_on_through_torchs_resets(inductor_state: Any) -> None:
    # torch 2.14.1 sets it again at each trace's end, through torch.use_deterministic_algorithms
    # (_dynamo/convert_frame.py:375, with the state before the trace), whose call sets inductor's
    # flag too (torch/__init__.py:1852). Kept: every such call leaves it on, torch's eager mode as
    # asked.
    torch.use_deterministic_algorithms(False)
    assert inductor_state.deterministic is False  # torch's own: the flag follows the call
    restore = compiling.keep_deterministic()
    assert inductor_state.deterministic is True
    torch.use_deterministic_algorithms(False)  # dynamo's reset after a trace
    assert inductor_state.deterministic is True
    assert not torch.are_deterministic_algorithms_enabled()  # the eager operations unchanged
    torch.use_deterministic_algorithms(False, warn_only=True)
    assert inductor_state.deterministic is True
    # Idempotent: a second call wraps nothing more, and its restore does nothing.
    wrapped = torch.use_deterministic_algorithms
    again = compiling.keep_deterministic()
    assert torch.use_deterministic_algorithms is wrapped
    again()
    assert torch.use_deterministic_algorithms is wrapped
    restore()
    assert torch.use_deterministic_algorithms is not wrapped
    torch.use_deterministic_algorithms(False)
    assert inductor_state.deterministic is False


# ----------------------------------------------------------------------------- the exclusion


def tiny_convolution() -> InflatedCausalConv3d:
    return InflatedCausalConv3d(2, 2, kernel_size=1, inflation_mode="tail")


def test_the_causal_convolutions_are_what_leaves_the_graph() -> None:
    # The predicate takes the VAE's causal convolutions and nothing else, a plain Conv3d among the
    # rest; exclusion wraps their forward on the instance and no other module's, and the class's
    # is untouched (numz's attribute, _dynamo_disable, is read by no one: bug 15).
    tree = torch.nn.Sequential(
        torch.nn.Linear(2, 2),
        tiny_convolution(),
        torch.nn.Sequential(torch.nn.Conv3d(2, 2, 1), tiny_convolution()),
    )
    found = [compiling.is_causal_convolution(module) for module in tree.modules()]
    assert (
        found.count(True) == 2
        and sum(isinstance(m, InflatedCausalConv3d) for m in tree.modules()) == 2
    )
    assert not compiling.is_causal_convolution(torch.nn.Conv3d(2, 2, 1))
    original = InflatedCausalConv3d.forward
    assert compiling.exclude_causal_convolutions(tree) == 2
    wrapped = [name for name, m in tree.named_modules() if "forward" in vars(m)]
    assert wrapped == ["1", "2.1"]
    assert InflatedCausalConv3d.forward is original


# A tiny module around a causal convolution, compiled on the CPU in a process of its own: a
# compilation imports jinja2, networkx and others into the process (environment.import_inductor),
# which tests of the environment, run after in this one, must not find there. argv: whether the
# convolution is excluded (1 or 0). Prints the values of torch.compiler.is_compiling() seen
# inside the convolution's forward, and how far the compiled output is from plain code's.
COMPILED = """
import json
import sys

import torch
import torch._inductor.config as inductor_config

from seedvr2x.runtime import compiling
from seedvr2x.vendor.models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d
from seedvr2x.vendor.models.video_vae_v3.modules.types import MemoryState

inductor_config.compile_threads = 1


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = torch.nn.Linear(4, 4)
        self.conv = InflatedCausalConv3d(2, 2, kernel_size=1, inflation_mode="tail")

    def forward(self, x):
        y = torch.relu(self.pre(x)).reshape(1, 2, 1, 2, 4)
        return self.conv(y, MemoryState.DISABLED) * 2


torch.manual_seed(0)
module = Tiny().eval()
x = torch.randn(4, 4)
expected = module(x)
seen = []
probed = module.conv.forward


def probe(*arguments, **options):
    seen.append(torch.compiler.is_compiling())
    return probed(*arguments, **options)


module.conv.forward = probe
if sys.argv[1] == "1":
    assert compiling.exclude_causal_convolutions(module) == 1
with torch.no_grad():
    got = torch.compile(module, **compiling.SETTINGS)(x)
compiling.shutdown_workers()
print(json.dumps({"seen": seen, "far": float((got - expected).abs().max())}))
"""


@pytest.mark.parametrize("excluded", [True, False], ids=["excluded", "traced"])
def test_the_causal_convolutions_run_as_plain_code_inside_a_compiled_module(
    tmp_path: Path, excluded: bool
) -> None:
    # Compiled on the CPU, a tiny module around a causal convolution: its forward, wrapped by
    # exclusion, runs eagerly (torch.compiler.is_compiling() is False inside it, where dynamo
    # would have traced it, as it does without the exclusion), and the compiled module gives plain
    # code's values. Skipped without a C++ compiler.
    if shutil.which("g++") is None and shutil.which("c++") is None:
        pytest.skip("no C++ compiler for inductor on the CPU")
    result = subprocess.run(
        [sys.executable, "-c", COMPILED, "1" if excluded else "0"],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "cache")},
    )
    found = json.loads(result.stdout.strip().splitlines()[-1])
    assert found["seen"] == ([False] if excluded else [True])
    assert found["far"] < 1e-5


# ----------------------------------------------------------------------------- workers and cache


def test_the_compile_workers_are_shut_through_torchs_own_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # torch 2.14.1: torch._inductor.async_compile.shutdown_compile_workers, the one its atexit
    # hook calls; it exists, and is the one called.
    module: Any = pytest.importorskip("torch._inductor.async_compile")
    assert callable(module.shutdown_compile_workers)
    called: list[str] = []
    config: Any = pytest.importorskip("torch._inductor.config")
    monkeypatch.setattr(config, "compile_threads", 8)  # put back by monkeypatch
    monkeypatch.setattr(module, "shutdown_compile_workers", lambda: called.append("shut"))
    compiling.shutdown_workers()
    assert called == ["shut"]
    # Any compile after it runs in this process, no pool started again (the review of
    # 2026-10-29, LOW 5): torch asks for a pool when compile_threads is over 1.
    assert config.compile_threads == 1
    monkeypatch.undo()
    monkeypatch.setattr(config, "compile_threads", 8)
    compiling.shutdown_workers()  # no pool started: nothing to do
    assert config.compile_threads == 1


def test_the_compile_threads_are_one_after_the_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, undo_global_state: None
) -> None:
    # A compile after the warm-up must not bring the workers back (compile_vae's own shutdown is
    # replaced by the Recorder's, so the real one is called here, with torch's shut replaced).
    config: Any = pytest.importorskip("torch._inductor.config")
    monkeypatch.setattr(config, "compile_threads", 8)
    module: Any = pytest.importorskip("torch._inductor.async_compile")
    monkeypatch.setattr(module, "shutdown_compile_workers", lambda: None)
    compiling.shutdown_workers()
    from torch._inductor.async_compile import get_compile_threads  # pyright: ignore

    assert get_compile_threads() == 1


@pytest.fixture
def frames(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """torch._dynamo's count of frames compiled as the test sets it (put back), and the
    compilation module's warm-up mark cleared."""
    utils: Any = pytest.importorskip("torch._dynamo.utils")
    monkeypatch.setitem(utils.counters["frames"], "total", 10)
    monkeypatch.setattr(compiling, "_warmed", [None])
    yield utils.counters["frames"]


def test_a_compilation_after_the_warm_up_is_said_at_warning_naming_the_pass(
    frames: Any, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger=compiling.__name__)
    # No warm-up (the VAE not compiled): nothing said, whatever dynamo counts.
    frames["total"] = 99
    assert compiling.report_late("shot 0's encode") == 0 and not caplog.records
    # The warm-up ends at 10 frames compiled.
    frames["total"] = 10
    compiling._warmed[0] = compiling.frames_compiled()  # pyright: ignore[reportPrivateUsage]
    assert compiling.report_late("shot 0's encode") == 0 and not caplog.records
    frames["total"] = 12
    assert compiling.report_late("shot 1's decode") == 2
    [record] = caplog.records
    assert record.levelname == "WARNING"
    assert record.getMessage().startswith("shot 1's decode: the compiler compiled 2 more frames")
    # Said once: the next pass compiles nothing more, the one after one frame.
    caplog.clear()
    assert compiling.report_late("shot 2's encode") == 0 and not caplog.records
    frames["total"] = 13
    assert compiling.report_late("shot 2's decode") == 1
    assert "compiled 1 more frame after" in caplog.text


def test_the_run_names_the_unit_that_compiled_after_the_warm_up(
    frames: Any, caplog: pytest.LogCaptureFixture
) -> None:
    # run._made, after each part of a unit, checks (and calibration's end does): through the real
    # function with a progress stand-in.
    from seedvr2x.runtime import run

    caplog.set_level(logging.WARNING, logger=compiling.__name__)
    compiling._warmed[0] = 10  # pyright: ignore[reportPrivateUsage]
    progress: Any = SimpleNamespace(
        clock=SimpleNamespace(estimate=lambda kind, size: None),
        made=lambda *given: None,
        skipped=lambda *given: None,
        said=lambda: "left",
    )
    run._made(progress, ("encode", 0), "encode", 5, 0.0, "shot 0's encode")  # pyright: ignore
    assert not caplog.records
    frames["total"] = 11
    run._made(progress, ("decode", 0), "decode", 5, 0.0, "shot 0's decode")  # pyright: ignore
    [record] = caplog.records
    assert record.getMessage().startswith("shot 0's decode: the compiler compiled 1 more frame")


def test_the_cache_is_the_jobs_own(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "users"))
    cache = tmp_path / "job" / "compile-cache"
    compiling.use_cache(cache)
    assert cache.is_dir()
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(cache)
    # torch's own reading of it, which exports the directory for Triton as well.
    from torch._inductor.runtime.cache_dir_utils import cache_dir  # pyright: ignore

    assert cache_dir() == str(cache)


# ----------------------------------------------------------------------------- the warm calls


class FakeVAE(torch.nn.Module):
    """The VAE as the run's own passes (model.encode_sliced, model.decode_sliced) call it, with
    the slicing of the config; _encode and _decode record the shape and memory state of each
    call. Its encoder and decoder are plain modules, compiled by nothing here."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, dtype=COMPUTE_DTYPE))
        self.encoder = torch.nn.Identity()
        self.decoder = torch.nn.Identity()
        self.use_slicing = True
        self.slicing_sample_min_size = 4
        self.slicing_latent_min_size = 1
        self.temporal_downsample_factor = 4
        self.calls: list[tuple[str, tuple[int, ...], str]] = []

    def _encode(self, x: torch.Tensor, memory_state: MemoryState) -> torch.Tensor:
        self.calls.append(("encode", tuple(x.shape), memory_state.name))
        return torch.zeros(1, 32, 1, 1, 1, dtype=x.dtype)

    def _decode(
        self, z: torch.Tensor, memory_state: MemoryState = MemoryState.DISABLED
    ) -> list[torch.Tensor]:
        self.calls.append(("decode", tuple(z.shape), memory_state.name))
        return [torch.zeros(1, dtype=z.dtype)]


def fake_models(vae: FakeVAE | None = None) -> Models:
    config = OmegaConf.create({"vae": {"scaling_factor": 0.9152}})
    runner = SimpleNamespace(vae=vae or FakeVAE(), config=config)
    return Models(runner, torch.zeros(1), torch.zeros(1), "sdpa", torch.device("cpu"))


def test_every_shape_and_slice_kind_is_warmed() -> None:
    # The kinds as the run meets them: a shot of 1 latent is one slice of 1 frame, of 2 one of 5
    # frames (neither with a memory), of 3 a first slice of 5 frames and a next one of 4; the
    # decode's the first 2 latents then 1 (DESIGN.md: "5 then 4 a slice"; "2 latents then 1").
    fake = FakeVAE()
    shapes = Shapes((128, 192), ((64, 128), (128, 128)), ((64, 64),), (1, 2, 3))
    compiling.warm(fake_models(fake), shapes)
    disabled, initial, active = "DISABLED", "INITIALIZING", "ACTIVE"
    expected: list[tuple[str, tuple[int, ...], str]] = []
    for rows, columns in shapes.encode:
        expected += [("encode", (1, 3, 1, rows, columns), disabled)]
        expected += [("encode", (1, 3, 5, rows, columns), disabled)]
        expected += [("encode", (1, 3, 5, rows, columns), initial)]
        expected += [("encode", (1, 3, 4, rows, columns), active)]
    # The shapes of a shot of 1 latent come first, then 2, then 3: the encode's order within.
    ordered = sorted(
        (call for call in fake.calls if call[0] == "encode"), key=lambda call: call[1:2]
    )
    assert sorted(call for call in fake.calls if call[0] == "encode") == sorted(expected)
    assert ordered
    decodes = [call for call in fake.calls if call[0] == "decode"]
    # 1 latent: one call of 1; 2 latents: one of 2; 3: the first 2 latents (a first slice of 1 +
    # 1) then 1; a latent 8 px per side: the tile's 64x64 is 8x8 latents.
    assert decodes == [
        ("decode", (1, 16, 1, 8, 8), "DISABLED"),
        ("decode", (1, 16, 2, 8, 8), "DISABLED"),
        ("decode", (1, 16, 2, 8, 8), "INITIALIZING"),
        ("decode", (1, 16, 1, 8, 8), "ACTIVE"),
    ]


def test_only_the_kinds_the_job_has_are_warmed() -> None:
    fake = FakeVAE()
    shapes = Shapes((128, 128), ((128, 128),), ((64, 64),), (3,))
    compiling.warm(fake_models(fake), shapes)
    assert [call[1:] for call in fake.calls] == [
        ((1, 3, 5, 128, 128), "INITIALIZING"),
        ((1, 3, 4, 128, 128), "ACTIVE"),
        ((1, 16, 2, 8, 8), "INITIALIZING"),
        ((1, 16, 1, 8, 8), "ACTIVE"),
    ]


# ----------------------------------------------------------------------------- the order


class Recorder:
    """What compile_vae and load_models do, in order, with torch's compiler and the weights'
    reading replaced."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.events: list[str] = []
        self.deterministic_at_compile: list[bool] = []
        self.cache_at_compile: list[str | None] = []
        self.excluded_at_compile: list[bool] = []

        def compile_module(module: Any, **settings: Any) -> Any:
            self.events.append(f"compile {settings['backend']}")
            self.deterministic_at_compile.append(
                getattr(torch.use_deterministic_algorithms, "keeps_inductor_deterministic", False)
            )
            self.cache_at_compile.append(os.environ.get("TORCHINDUCTOR_CACHE_DIR"))
            return module

        monkeypatch.setattr(torch, "compile", compile_module)
        monkeypatch.setattr(compiling, "warm", lambda models, shapes: self.events.append("warm"))
        monkeypatch.setattr(
            compiling, "shutdown_workers", lambda: self.events.append("workers shut")
        )

        def load_weights(module: Any, path: Path, *rest: Any) -> dict[str, Any]:
            self.events.append(f"weights {path.name}")
            return {}

        monkeypatch.setattr(model, "_load_weights", load_weights)
        monkeypatch.setattr(
            model.weights, "check", lambda path, kind: SimpleNamespace(config="configs_7b")
        )


@pytest.fixture
def undo_global_state(inductor_state: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "unset")
    monkeypatch.setattr(
        torch._dynamo.config, "recompile_limit", torch._dynamo.config.recompile_limit
    )


def test_the_vae_is_compiled_before_the_dit_loads_and_the_workers_shut_before_its_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, undo_global_state: None
) -> None:
    # DESIGN.md: "the VAE compiled right after the plan, before the DiT loads: the VAE alone on the
    # card, its shapes compiled with torch's own workers, then the workers shut at once, before
    # the DiT's blocks are pinned". The loads recorded by their files; the compilation's calls
    # by name; deterministic mode, the cache and the exclusion in place at the first compile.
    recorder = Recorder(monkeypatch)
    compilation = Compilation(Shapes((128, 128), (), (), ()), tmp_path / "compile-cache")
    plan = Plan(5, compile_vae=True)
    models = model.load_models(
        Path("dit.safetensors"),
        Path("vae.safetensors"),
        torch.device("cpu"),
        None,
        plan,
        compilation,
    )
    assert recorder.events == [
        "weights vae.safetensors",
        "compile inductor",
        "compile inductor",
        "warm",
        "workers shut",
        "weights dit.safetensors",
    ]
    assert recorder.deterministic_at_compile == [True, True]
    assert recorder.cache_at_compile == [str(tmp_path / "compile-cache")] * 2
    assert (tmp_path / "compile-cache").is_dir()
    assert compilation.seconds >= 0 and models.plan is plan
    # The causal convolutions left out before anything compiled.
    convolutions = [m for m in models.runner.vae.modules() if isinstance(m, InflatedCausalConv3d)]
    assert convolutions and all("forward" in vars(m) for m in convolutions)


def test_a_plan_without_compile_loads_as_it_did(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, undo_global_state: None
) -> None:
    recorder = Recorder(monkeypatch)
    model.load_models(Path("dit.safetensors"), Path("vae.safetensors"), torch.device("cpu"))
    model.load_models(
        Path("dit.safetensors"), Path("vae.safetensors"), torch.device("cpu"), None, Plan(5)
    )
    assert recorder.events == ["weights vae.safetensors", "weights dit.safetensors"] * 2
    with pytest.raises(ValueError, match="nothing to compile the VAE with"):
        model.load_models(
            Path("dit.safetensors"),
            Path("vae.safetensors"),
            torch.device("cpu"),
            None,
            Plan(5, compile_vae=True),
        )


def test_the_workers_shut_whatever_happens_and_a_failure_is_the_jobs_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, undo_global_state: None
) -> None:
    recorder = Recorder(monkeypatch)

    def failing(models: Models, shapes: Shapes) -> None:
        recorder.events.append("warm fails")
        raise RuntimeError("InductorError: no compiler\nmore")

    monkeypatch.setattr(compiling, "warm", failing)
    compilation = Compilation(Shapes((128, 128), (), (), ()), tmp_path / "cache")
    with pytest.raises(compiling.CompileError) as error:
        model.load_models(
            Path("dit.safetensors"),
            Path("vae.safetensors"),
            torch.device("cpu"),
            None,
            Plan(5, compile_vae=True),
            compilation,
        )
    assert isinstance(error.value, JobError)
    assert str(error.value) == (
        "--compile: the VAE did not compile (RuntimeError: InductorError: no compiler); nothing"
        " was run, the same command without --compile runs the plain VAE"
    )
    assert recorder.events[-2:] == ["warm fails", "workers shut"]  # the DiT never loaded
    assert "weights dit.safetensors" not in recorder.events
    # The card's own refusal is not hidden under it.
    monkeypatch.setattr(
        compiling,
        "warm",
        lambda models, shapes: (_ for _ in ()).throw(torch.OutOfMemoryError("CUDA out of memory")),
    )
    with pytest.raises(torch.OutOfMemoryError):
        model.load_models(
            Path("dit.safetensors"),
            Path("vae.safetensors"),
            torch.device("cpu"),
            None,
            Plan(5, compile_vae=True),
            compilation,
        )
    assert recorder.events[-1] == "workers shut"


def test_the_vae_leaving_the_card_comes_for_the_warm_calls_and_goes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A plan with the VAE's weights away from the card in the DiT's phase has them in their
    # pinned copy at the load: brought to the device for the warm calls as Models.enter brings
    # them, and sent away after the device's work, the allocator trimmed (blockswap.Cuda).
    events: list[str] = []

    class Part:
        device = None

        def enter(self, backend: Any) -> None:
            events.append("enter")
            self.device = "on"

        def leave(self) -> None:
            events.append("leave")
            self.device = None

    class Backend:
        def __init__(self, device: Any) -> None:
            pass

        def compute(self) -> str:
            return "stream"

        def wait(self, *streams: Any) -> None:
            events.append("wait")

        def trim(self) -> None:
            events.append("trim")

    monkeypatch.setattr(compiling.blockswap, "Cuda", Backend)
    models = fake_models()
    models.vae = Part()  # pyright: ignore[reportAttributeAccessIssue]
    models.device = SimpleNamespace(type="cuda")  # pyright: ignore[reportAttributeAccessIssue]
    with compiling._on_the_device(models):  # pyright: ignore[reportPrivateUsage]
        events.append("work")
    assert events == ["enter", "work", "wait", "leave", "trim"]
    # Weights that never leave need nothing.
    events.clear()
    models.vae = None
    with compiling._on_the_device(models):  # pyright: ignore[reportPrivateUsage]
        events.append("work")
    assert events == ["work"]


# ----------------------------------------------------------------------------- the environment


def test_inductors_modules_are_imported_before_the_environment_is_recorded() -> None:
    # What a compilation imports and the run otherwise doesn't (measured on a small module
    # compiled on the CPU, torch 2.14.1): the packages recorded with --compile are the packages
    # the compilation will have imported, at a start and at a resume alike.
    code = (
        "from seedvr2x.runtime import environment as e\n"
        "plain = e.versions()\n"
        "e.import_inductor()\n"
        "after = e.versions()\n"
        "import json\nprint(json.dumps(sorted(set(after) - set(plain))))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    added = json.loads(result.stdout.strip())
    assert set(added) == resume.INDUCTOR_PACKAGES


def test_nothing_a_compilation_imports_is_missing_from_the_record(
    tmp_path: Path,
) -> None:
    # After a real compilation of a small module on the CPU, in a process of its own, no
    # distribution of the run is imported that environment.current(compiling=True) didn't record
    # (cli._run says it as a seedvr2x bug otherwise). Skipped without a C++ compiler.
    if shutil.which("g++") is None and shutil.which("c++") is None:
        pytest.skip("no C++ compiler for inductor on the CPU")
    code = (
        "import os, torch\n"
        "from seedvr2x.runtime import environment as e\n"
        "recorded = e.current(None, 'ffmpeg', 'conv', {'gpu': 'g', 'driver': 'd', 'cudnn': 1},"
        " compiling=True)['packages']\n"
        "import torch._inductor.config as c\n"
        "c.compile_threads = 1\n"
        "m = torch.compile(torch.nn.Linear(4, 4), **{'backend': 'inductor', 'dynamic': False})\n"
        "with torch.no_grad():\n"
        "    m(torch.randn(2, 4))\n"
        "print(sorted(e.imported() - set(recorded)))\n"
    )
    environment = {**os.environ, "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "cache")}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env=environment,
    )
    assert result.stdout.strip().splitlines()[-1] == "[]", result.stdout


# ----------------------------------------------------------------------------- the command line

from test_cli_run import (  # noqa: E402
    JOB,
    LARGER,
    LARGER_PADDED,
    STAND_IN_MODELS,
    Steps,
    job,
    refused,
    stand_in,  # noqa: F401
    steps,  # noqa: F401
    stopped,
    upscale,
)

from seedvr2x import cli  # noqa: E402
from seedvr2x.runtime import environment, timing  # noqa: E402


@pytest.fixture(autouse=True)
def no_inductor_imported_here(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run with --compile imports inductor's modules before it records the environment
    (environment.import_inductor), jinja2 and networkx among them, which stay in the process: the
    environment's tests, run after in it, count on the run's packages alone
    (tests/test_environment.py, test_torch_runtime_libraries). Replaced here, a no-op; a process
    of its own holds the import itself (test_inductors_modules_are_imported_before_the_
    environment_is_recorded)."""
    monkeypatch.setattr(environment, "import_inductor", lambda: None)


class Loads:
    """The stand-in's load of the models (model.load_models): what it was given, and, as the
    compilation does, the seconds it took and a file in the job's cache."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, seconds: float = 300.0) -> None:
        self.given: list[tuple[Any, ...]] = []
        self.seconds = seconds
        monkeypatch.setattr(model, "load_models", self)

    def __call__(self, *arguments: Any) -> Any:
        self.given.append(arguments)
        compilation = arguments[5]
        if compilation is not None:
            compilation.cache.mkdir(parents=True, exist_ok=True)
            (compilation.cache / "entry.bin").write_bytes(b"compiled")
            compilation.seconds = self.seconds
        return STAND_IN_MODELS


def accepting_tiles(monkeypatch: pytest.MonkeyPatch, steps: Steps) -> None:  # noqa: F811
    """The stand-in's encode and decode (steps) taking the plan's tiles, scratch and allowance
    as the run gives them, around their own, untiled, pass: what these tests ask of them is the
    plan's, not the tiled passes' own (tests/test_tiling.py, tests/test_shot.py)."""
    from seedvr2x.runtime import run

    def encode(*arguments: Any, tiles: Any, scratch: Any, allowance: Any, **options: Any) -> Any:
        return steps.encode(*arguments, **options)

    def decode(*arguments: Any, tiles: Any, scratch: Any, allowance: Any) -> None:
        steps.decode(*arguments)

    monkeypatch.setattr(run, "encode_shot", encode)
    monkeypatch.setattr(run, "decode_shot", decode)


def manifest_of(directory: Path) -> dict[str, Any]:
    return json.loads((directory / "manifest.json").read_text())


def test_the_flag_is_recorded_and_the_models_load_with_the_compilation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    # --compile: a setting (compile_vae), the plan's record compile_vae true and compile_dit
    # false, the shapes of the plan's passes and the cache in the job's directory given to the
    # load, the plan's line of it said.
    loads = Loads(monkeypatch)
    accepting_tiles(monkeypatch, steps)
    caplog.set_level(logging.INFO)
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB, *LARGER, "--compile") == 0
    out = manifest_of(tmp_path / "out")
    assert out["settings"]["compile_vae"] is True
    assert (out["plan"]["compile_vae"], out["plan"]["compile_dit"]) == (True, False)
    [(*_, plan, compilation)] = loads.given
    assert plan.compile_vae and not plan.compile_dit
    assert compilation.cache == tmp_path / "out" / "compile-cache"
    # 704x560 frames: the decode in tiles of 640 px (the frame whole is no compiled decode), the
    # encode the frame whole, which the 1080p frame measured holds; JOB's shots are 2, 1 and 6
    # latents.
    assert plan.encode_tiles is None and plan.decode_tiles == Tiles(640)
    assert compilation.shapes.padded == LARGER_PADDED
    assert compilation.shapes.encode == (LARGER_PADDED,)
    # 704 columns in tiles of 640 px sharing 64, a stride of 576: a second tile of 128.
    assert compilation.shapes.decode == ((560, 640), (560, 128))
    assert compilation.shapes.latents == (1, 2, 3)
    assert "plan: compile: the VAE compiled (--compile)" in caplog.text
    assert "--compile is not used" not in caplog.text


def test_without_the_flag_nothing_is_compiled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    loads = Loads(monkeypatch)
    caplog.set_level(logging.INFO)
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB, *LARGER) == 0
    out = manifest_of(tmp_path / "out")
    assert out["settings"]["compile_vae"] is False
    assert (out["plan"]["compile_vae"], out["plan"]["compile_dit"]) == (False, False)
    [(*_, plan, compilation)] = loads.given
    assert compilation is None and not plan.compile_vae
    assert "compile" not in caplog.text.replace("compile_vae", "")
    assert not (tmp_path / "out" / "compile-cache").exists()


def test_a_resume_with_the_other_value_is_another_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The compiled VAE's frames are not plain code's: a job is never resumed with the other
    # setting, and says so as any other setting's difference.
    Loads(monkeypatch)
    accepting_tiles(monkeypatch, steps)
    source_path = job(tmp_path, monkeypatch)
    for first, second in (("--compile", None), (None, "--compile")):
        output = f"out{first}{second}".replace("-", "")
        steps.stop = "window 4:1"
        stopped(tmp_path, source_path, output, *JOB, *LARGER, *([first] if first else []))
        text = refused(
            tmp_path,
            source_path,
            caplog,
            *JOB,
            *LARGER,
            *([second] if second else []),
            output=output,
        )
        assert "another job than the one asked" in text
        expected = "true -> false" if first else "false -> true"
        assert f"settings.compile_vae: {expected}" in text
        # The same setting resumes it.
        steps.stop = None
        assert (
            upscale(tmp_path, source_path, output, *JOB, *LARGER, *([first] if first else [])) == 0
        )


def test_a_job_outside_the_shapes_measured_is_refused_never_run_plain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The stand-in's frames, 128x96 padded, are too small for a decode tile, and an untiled
    # decode is no compiled decode: the job is refused (DESIGN.md, Memory planner, decided on
    # 2026-10-29), nothing loaded or run, the first pass kept as after any refusal, with the
    # advice; --plan gives the same refusal, and the same command without --compile takes the
    # first pass up.
    loads = Loads(monkeypatch)
    caplog.set_level(logging.INFO)
    source_path = job(tmp_path, monkeypatch)
    text = refused(tmp_path, source_path, caplog, *JOB, "--compile")
    assert "the VAE compiled has no decode pass within the shapes measured" in text
    assert "\nThe same command without --compile fits; the first pass is kept in " in text
    assert "--compile is not used" not in text
    assert loads.given == [] and steps.calls == []
    out = tmp_path / "out"
    assert (out / "manifest.json").is_file() and manifest_of(out)["settings"]["compile_vae"] is True
    assert manifest_of(out).get("plan") is None
    # The same refusal for --plan, whose first pass is the one kept.
    asked = refused(tmp_path, source_path, caplog, *JOB, "--compile", "--plan", "p.txt")
    assert "the VAE compiled has no decode pass within the shapes measured" in asked
    assert "\nThe same command without --compile fits; " in asked
    assert "the first pass is kept in " in asked and loads.given == []
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert "the first pass as recorded" in caplog.text
    assert manifest_of(out)["plan"]["compile_vae"] is False


def test_a_pin_beyond_the_shapes_measured_is_refused_by_the_command_with_its_advice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    # An untiled decode pinned is beyond the shapes whatever the frame (a decode compiled is
    # always tiled): on the stand-in's 704x560 frames, refused naming the pass, the tiles and the
    # shapes, with the advice on a line of its own each.
    loads = Loads(monkeypatch)
    source_path = job(tmp_path, monkeypatch)
    text = refused(tmp_path, source_path, caplog, *JOB, *LARGER, "--compile", "--decode-tile", "0")
    assert "--decode-tile 0: the VAE compiled has the decode untiled beyond the shapes" in text
    assert "(decode tiles up to 1152 px, never untiled)\n" in text
    assert "\nThe same command without --compile fits\n" in text
    assert "\nThe same command with --decode-tile 640 fits, within the shapes measured;" in text
    assert loads.given == []


def test_plan_says_the_compilations_time_and_the_estimate_counts_it_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    # --plan loads the models for its calibration, so compiles the VAE as a run does: the plan's
    # line says 4 to 6 min per process, and the time of the load, which the compilation is part
    # of, says its share of it, once, in the load's own seconds.
    Loads(monkeypatch, seconds=300.0)
    accepting_tiles(monkeypatch, steps)
    caplog.set_level(logging.INFO)
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB, *LARGER, "--compile", "--plan", "p.txt") == 0
    assert "compiling it takes about 4 to 6 min per process, paid before the DiT loads" in (
        caplog.text
    )
    [load] = [r.getMessage() for r in caplog.records if "the models loaded in" in r.getMessage()]
    assert (
        ", 5 min 0 s of it compiling the VAE, which the run pays again before its first unit"
        in load
    )


def test_the_load_time_names_the_compilation_once_and_adds_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from seedvr2x.runtime.calibration import Calibrated

    clock = timing.Clock(dict.fromkeys((*timing.KINDS, *timing.TERMS), timing.Model(1.0, 0.0)))
    unit = timing.Unit((0, 0), "encode 0", (("encode", 5),))
    caplog.set_level(logging.INFO)
    compilation = Compilation(Shapes((128, 128), (), (), ()), Path("c"), seconds=300.0)
    for given, expected in (
        (compilation, "the models loaded in 6 min 40 s, 5 min 0 s of it compiling the VAE"),
        (None, "the models loaded in 6 min 40 s,"),
    ):
        caplog.clear()
        cli._say_times(  # pyright: ignore[reportPrivateUsage]
            Calibrated(clock, 20.0), [unit], "standard", None, 400.0, None, given, planning=False
        )
        [text] = [
            r.getMessage() for r in caplog.records if "the models loaded in" in r.getMessage()
        ]
        assert expected.rstrip(",") in text
        if given is None:
            assert "compiling" not in text
        # Nothing summed: no 11 min (400 + 300 + 20 s) anywhere.
        assert "12 min" not in caplog.text and "11 min" not in caplog.text


def test_the_cache_is_the_jobs_own_and_stays_where_the_job_does(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
) -> None:
    # The compile cache, in the job's directory, is its own name to the checks of a resume (a
    # stop leaves it, the next run finds it, nothing foreign); a link by its name is not. An
    # output directory keeps it (PROVISIONAL, manifest.COMPILE_CACHE); a .mkv path's work
    # directory goes with it when the file is assembled.
    Loads(monkeypatch)
    accepting_tiles(monkeypatch, steps)
    source_path = job(tmp_path, monkeypatch)
    steps.stop = "window 4:1"
    stopped(tmp_path, source_path, "out", *JOB, *LARGER, "--compile")
    cache = tmp_path / "out" / "compile-cache"
    assert (cache / "entry.bin").read_bytes() == b"compiled"
    steps.stop = None
    assert upscale(tmp_path, source_path, "out", *JOB, *LARGER, "--compile") == 0
    assert (cache / "entry.bin").is_file()
    assert upscale(tmp_path, source_path, "out", *JOB, *LARGER, "--compile") == 0  # finished
    # Another directory's: a link is refused, nothing removed.
    steps.stop = "window 4:1"
    stopped(tmp_path, source_path, "linked", *JOB, *LARGER, "--compile")
    shutil.rmtree(tmp_path / "linked" / "compile-cache")
    (tmp_path / "linked" / "compile-cache").symlink_to(tmp_path)
    steps.stop = None
    assert upscale(tmp_path, source_path, "linked", *JOB, *LARGER, "--compile") == 1
    assert (tmp_path / "linked" / "compile-cache").is_symlink()
    # A one-file output: its work directory, cache included, is removed with the assembly.
    assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4], *LARGER, "--compile") == 0
    assert (tmp_path / "one.mkv").is_file() and not (tmp_path / "one.mkv.work").exists()


def test_a_compile_cache_is_the_jobs_own_only_when_the_job_asked_for_the_compile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The review of 2026-10-29 (LOW 6): a directory called compile-cache in a job that asked for
    # no --compile (settings.compile_vae false) is foreign as any unknown name: refused at a
    # resume and left in a .mkv path's work directory when it is cleaned, never deleted; the
    # asked job's own is kept and removed (the test above).
    Loads(monkeypatch)
    source_path = job(tmp_path, monkeypatch)
    steps.stop = "window 4:1"
    stopped(tmp_path, source_path, "plain", *JOB)
    foreign = tmp_path / "plain" / "compile-cache"
    foreign.mkdir()
    (foreign / "mine.txt").write_text("someone else's")
    steps.stop = None
    text = refused(tmp_path, source_path, caplog, *JOB, output="plain")
    assert "not this job's: compile-cache" in text
    assert (foreign / "mine.txt").read_text() == "someone else's"
    # The clean-up of a work directory (resume.clear), on the manifest of the stopped job.
    content = json.loads((tmp_path / "plain" / "manifest.json").read_text())
    assert content["settings"]["compile_vae"] is False
    assert resume.clear(tmp_path / "plain", content) == ["compile-cache"]
    assert (foreign / "mine.txt").read_text() == "someone else's"
    # The same directory in a job that asked for --compile is its own cache, removed whole.
    content["settings"]["compile_vae"] = True
    assert resume.clear(tmp_path / "plain", content) == []
    assert not foreign.exists()


def test_a_vae_that_does_not_compile_refuses_the_job_before_anything_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    said = "--compile: the VAE did not compile (RuntimeError: no compiler); nothing was run"

    def failing(*arguments: Any) -> Any:
        raise compiling.CompileError(said + ", the same command without --compile runs it plain")

    monkeypatch.setattr(model, "load_models", failing)
    source_path = job(tmp_path, monkeypatch)
    text = refused(tmp_path, source_path, caplog, *JOB, *LARGER, "--compile")
    assert said in text
    assert steps.calls == []


def test_inductors_modules_are_imported_before_the_environment_is_read_with_the_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
) -> None:
    events: list[str] = []
    versions = environment.versions
    monkeypatch.setattr(environment, "import_inductor", lambda: events.append("inductor"))
    monkeypatch.setattr(environment, "versions", lambda: (events.append("versions"), versions())[1])
    Loads(monkeypatch)
    accepting_tiles(monkeypatch, steps)
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB, *LARGER, "--compile") == 0
    assert events[:2] == ["inductor", "versions"]
    events.clear()
    assert upscale(tmp_path, source_path, "plain", *JOB, *LARGER) == 0
    assert "inductor" not in events


@pytest.fixture
def inductor_modules(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Inductor's modules stood in for: what a process has imported depends on the tests run
    before in it, and a real process starts with none of them, so a run with --compile adds one
    to the versions recorded (one of resume.INDUCTOR_PACKAGES, networkx) and a plain one doesn't.
    The list is the process's imports: cleared before each run a test makes, a process of its
    own."""
    imported: list[str] = []
    versions = environment.versions
    monkeypatch.setattr(environment, "import_inductor", lambda: imported.append("inductor"))
    monkeypatch.setattr(
        environment,
        "versions",
        lambda: {**versions(), **({"networkx": "1.0"} if imported else {})},
    )
    return imported


def test_a_plan_with_no_unit_made_is_planned_again_for_the_other_value_with_no_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
    inductor_modules: list[str],
) -> None:
    # --compile is a setting a plan takes up changed (resume.FIRST_PASS_SETTINGS): a job that
    # made no unit yet, only planned (--plan), is planned again with it, the first pass kept. The
    # environment recorded with the flag has the packages a compilation imports (inductor's
    # modules, imported before it is recorded), which the plain one hasn't, and they count only
    # between two runs that both compile (DESIGN.md, decided on 2026-10-29): no
    # --accept-env-change is asked, to or from --compile, and none is recorded.
    loads = Loads(monkeypatch)
    accepting_tiles(monkeypatch, steps)
    source_path = job(tmp_path, monkeypatch)
    out_dir = tmp_path / "out"
    options = (*JOB, *LARGER)
    inductor_modules.clear()
    assert upscale(tmp_path, source_path, "out", *options, "--plan", "p.txt") == 0
    assert manifest_of(out_dir)["plan"]["compile_vae"] is False
    assert "networkx" not in manifest_of(out_dir)["environment"]["packages"]
    # Plain to compiled: a plan with no unit made, planned again.
    inductor_modules.clear()
    assert upscale(tmp_path, source_path, "out", *options, "--compile", "--plan", "p.txt") == 0
    planned = manifest_of(out_dir)
    assert (planned["settings"]["compile_vae"], planned["plan"]["compile_vae"]) == (True, True)
    assert planned["environment"]["packages"]["networkx"] == "1.0"
    assert planned.get("environment_changes", []) == []
    assert (out_dir / "compile-cache" / "entry.bin").is_file()  # the calibration compiled
    # Compiled to plain, again with no flag, and to compiled once more, then run.
    inductor_modules.clear()
    assert upscale(tmp_path, source_path, "out", *options, "--plan", "p.txt") == 0
    again = manifest_of(out_dir)
    assert (again["settings"]["compile_vae"], again["plan"]["compile_vae"]) == (False, False)
    assert "networkx" not in again["environment"]["packages"]
    assert again.get("environment_changes", []) == []
    # The cache belonged to the plan that compiled: gone with it, not a foreign name in a plain
    # job's directory (cli._replan, PROVISIONAL).
    assert not (out_dir / "compile-cache").exists()
    inductor_modules.clear()
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *options, "--compile") == 0
    out = manifest_of(out_dir)
    assert (out["settings"]["compile_vae"], out["plan"]["compile_vae"]) == (True, True)
    assert out.get("environment_changes", []) == []
    assert "another environment" not in caplog.text and "--accept-env-change" not in caplog.text
    assert loads.given[-1][5] is not None


def test_a_job_with_units_made_toggled_is_another_jobs_settings_not_its_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
    inductor_modules: list[str],
) -> None:
    # Units made keep the setting they have: --compile toggled is another job's refusal as for any
    # setting, listing the setting alone, not the packages a compilation imports.
    Loads(monkeypatch)
    accepting_tiles(monkeypatch, steps)
    source_path = job(tmp_path, monkeypatch)
    for first, second in (("--compile", None), (None, "--compile")):
        output = f"out{first}{second}".replace("-", "")
        steps.stop = "window 4:1"
        inductor_modules.clear()
        stopped(tmp_path, source_path, output, *JOB, *LARGER, *([first] if first else []))
        inductor_modules.clear()
        text = refused(
            tmp_path,
            source_path,
            caplog,
            *JOB,
            *LARGER,
            *([second] if second else []),
            output=output,
        )
        assert "another job than the one asked" in text and "settings.compile_vae" in text
        assert "environment.packages" not in text and "networkx" not in text
        assert "Only its environment differs" not in text


def test_the_environments_comparison_counts_inductors_modules_only_between_compiled_jobs() -> None:
    # resume.differences and resume.environment_change: the packages a compilation imports
    # (INDUCTOR_PACKAGES) are left out of both sides unless both jobs have the VAE compiled; any
    # other package, and those packages between two compiled jobs, count.
    def content(compiled: bool, packages: dict[str, str]) -> dict[str, Any]:
        return {
            "settings": {"compile_vae": compiled, "seed": 1},
            "environment": {"torch": "2.14.1", "packages": packages},
            "segments": [],
            "shots": [],
        }

    base = {"torch": "2.14.1", "numpy": "2.0"}
    brought = {name: "1.0" for name in resume.INDUCTOR_PACKAGES}
    plain, compiled = content(False, base), content(True, {**base, **brought})
    # Plain to compiled or back: the setting alone differs, never the packages.
    assert resume.differences(plain, compiled) == ["settings.compile_vae: false -> true"]
    assert resume.differences(compiled, plain) == ["settings.compile_vae: true -> false"]
    assert resume.differences(plain, content(False, base)) == []
    # Between two compiled jobs they count, one changed or one missing.
    for name in sorted(resume.INDUCTOR_PACKAGES):
        changed = content(True, {**base, **brought, name: "2.0"})
        assert resume.differences(compiled, changed) == [
            f'environment.packages.{name}: "1.0" -> "2.0"'
        ]
        missing = content(True, {**base, **{n: v for n, v in brought.items() if n != name}})
        assert resume.differences(compiled, missing) == [
            f'environment.packages.{name}: "1.0" -> null'
        ]
    # Any other package counts whatever the settings.
    other = content(False, {**base, "numpy": "2.1"})
    assert resume.differences(plain, other) == ['environment.packages.numpy: "2.0" -> "2.1"']
    # The record of an accepted change leaves them out in the same cases.
    mixed = content(True, {**base, **brought, "numpy": "2.1"})
    change = resume.environment_change(plain, mixed["environment"], mixed["settings"])
    assert change["before"] == {"packages": {"numpy": "2.0"}}
    assert change["after"] == {"packages": {"numpy": "2.1"}}
    changed = content(True, {**base, **brought, "jinja2": "2.0"})
    shown = resume.environment_change(compiled, changed["environment"], changed["settings"])
    assert shown["before"] == {"packages": {"jinja2": "1.0"}}

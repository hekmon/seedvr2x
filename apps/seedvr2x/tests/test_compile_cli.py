"""--compile through the CLI (seedvr2x/shot_plan.py's compiled passes, planner.py's shapes, the
plan, the manifest and a resume; DESIGN.md, Memory planner, "torch.compile: the VAE, opt-in, in
v0.1.0; the DiT, not", decided on 2026-10-24): the planner holding a compiled pass to 1.1 times its
tile's plain peak and keeping to the shapes measured, the flag recorded and a resume with the other
value another job, the compile cache in the job's directory, the environment's modules of inductor
counted between two compiled jobs only, and the plan's words. On the CPU. The library's side (the
shapes warmed, the causal convolutions left out of the graph, the order of the load) is
test_compile.py's (core/tests/), whose fixtures these tests use."""

import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from test_compile import (
    FakeVAE,
    fake_models,
    frames,  # noqa: F401
    inductor_state,  # noqa: F401
    undo_global_state,  # noqa: F401
)
from test_shot_plan import HD, UHD, our_free, segments_at, shots
from torch._dynamo.exc import FailOnRecompileLimitHit

import planner
import resume
import run
from seedvr2x import compiling, model, peaks
from seedvr2x.errors import Refusal
from seedvr2x.frames import output_size, padded_size
from seedvr2x.model import COMPUTE_DTYPE, Models
from seedvr2x.passes import Compilation, Plan, Shapes, Tiles
from seedvr2x.peaks import GIB
from seedvr2x.shot_plan import Pins, PlanError
from seedvr2x.shots import Shot
from seedvr2x.vendor.models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d


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
    # tiles up to 1856 px or the 1080p frame whole, decode tiles up to 1152 px; compilation,
    # measured on 2026-10-31 and fitted a shape (1080p's 6 shapes and 4K's 8 on one host): 40 s
    # a new job and 15 s a resume on Linux's disk, 65 s and 20 s on a Windows drive. The 1080p
    # frame padded is 1104x1920.
    assert peaks.COMPILE_PEAK == 1.1
    assert (peaks.COMPILE_ENCODE_TILE_MAX, peaks.COMPILE_DECODE_TILE_MAX) == (1856, 1152)
    assert peaks.COMPILE_ENCODE_WHOLE == padded_size(HD) == (1104, 1920)
    assert (planner.COMPILE_SHAPE_NEW, planner.COMPILE_SHAPE_RESUME) == (40, 15)
    assert (planner.COMPILE_SHAPE_NEW_DRIVE, planner.COMPILE_SHAPE_RESUME_DRIVE) == (65, 20)


def test_a_compiled_pass_peaks_at_1_1_times_the_tiles_plain_peak() -> None:
    # The tile's peak above what is on the GPU before the pass is 1.1 times its plain one, the
    # table's (peaks.MEASURED); the weights and the encode's 0.4 GiB of margin (ENCODE_EXTRA)
    # are not scaled, as the study's run scaled the fit alone (research/scripts/compile_run.py,
    # --vae-peak-factor). Every tile, at 1080p and 4K, with the DiT's part on the GPU counted
    # as for a plain pass.
    fixed_encode = peaks.ENCODE_EXTRA + peaks.VAE_WEIGHTS
    for target in (HD, UHD):
        padded, output = padded_size(target), output_size(target)
        for size in range(512, 1857, 64):
            tiles = Tiles(size)
            for resident, swapped in ((None, 0), (5, 28), (0, 36)):
                kept = peaks.dit_on_gpu(resident, swapped)
                plain = peaks.encode_peak(padded, tiles, resident, swapped) - fixed_encode - kept
                compiled = peaks.encode_peak(padded, tiles, resident, swapped, True)
                assert compiled - fixed_encode - kept == pytest.approx(1.1 * plain, abs=1)
                plain = peaks.decode_peak(padded, output, tiles, resident, swapped) - kept
                compiled = peaks.decode_peak(
                    padded, output, tiles, resident, swapped, compiled=True
                )
                assert compiled - peaks.VAE_WEIGHTS - kept == pytest.approx(
                    1.1 * (plain - peaks.VAE_WEIGHTS), abs=1
                )
    # Untiled passes too (a small frame): the fit scaled, the correction's term not.
    small = (576, 960)
    plain = peaks.decode_peak(small, small, None, split=True)
    compiled = peaks.decode_peak(small, small, None, split=True, compiled=True)
    assert compiled - plain == pytest.approx(
        0.1 * (peaks._fit(small, None, 1)),  # pyright: ignore[reportPrivateUsage]
        abs=1,
    )


def test_a_plan_with_the_vae_compiled_chooses_smaller_tiles_where_1_1_times_doesnt_fit() -> None:
    # On cards where the plain tiles fit at their peak and not at 1.1 times it, the compiled plan
    # takes smaller ones; never larger, never beyond the measured shapes, and every pass of it
    # fits the budget at the compiled peak, its room counted.
    smaller = 0
    for card in range(18, 49, 2):
        (plain, _), (compiled, predicted) = plain_and_compiled(our_free(card), HD)
        assert compiled.compile_vae and not plain.compile_vae
        assert peaks.in_compile_range(padded_size(HD), "encode", compiled.encode_tiles)
        assert peaks.in_compile_range(padded_size(HD), "decode", compiled.decode_tiles)
        for phase in ("encode", "decode"):
            assert predicted.need(phase) + predicted.free - predicted.budget <= predicted.free
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
            made, predicted = planner.plan(
                our_free(card), HD, shots(76), Pins(), None, compile_vae=compile_vae
            )
            padded, output = padded_size(HD), output_size(HD)
            assert predicted.encode == peaks.encode_peak(
                padded, made.encode_tiles, made.resident_encode, made.swapped, compile_vae
            )
            assert predicted.decode == peaks.decode_peak(
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


def refusal(
    free: int, target: tuple[int, int], pins: Pins, lengths: tuple[int, ...] = (76,)
) -> str:
    """The message of a job refused compiled where the plain plan is made."""
    planner.plan(free, target, shots(*lengths), pins, None)
    with pytest.raises(PlanError) as refused:
        planner.plan(free, target, shots(*lengths), pins, None, compile_vae=True)
    return planner.worded(refused.value)


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
        assert peaks.compile_range_said(phase) in lines[0]  # pyright: ignore[reportPrivateUsage]
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
        f" measured ({peaks.compile_range_said('encode')})"  # pyright: ignore[reportPrivateUsage]
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
    measured = [peaks.compile_range_said(phase) for phase in ("encode", "decode")]  # pyright: ignore[reportPrivateUsage]
    assert lines[1] == "--compile compiles only the shapes measured: " + "; ".join(measured)
    assert lines[2:] == [
        "The same command without --compile fits",
        "The same command with --decode-tile 1024 fits, within the shapes measured",
    ]


def test_a_frame_too_small_for_a_tile_has_no_compiled_decode_and_is_refused() -> None:
    # A decode can't be untiled compiled, and a frame under 576 px a side has no tile to make
    # (shot_plan._tile_sizes): refused, with no pin to advise.
    lines = refusal(our_free(96), (96, 128), Pins(), (3,)).split("\n")
    assert lines[0].startswith("the VAE compiled has no decode pass within the shapes measured")
    assert lines[1:] == ["The same command without --compile fits"]


def test_a_job_that_fits_no_card_plain_is_refused_as_it_was() -> None:
    # --compile adds no refusal of its own: the plain job's is the one given, as it was before
    # the last rungs (no rung under --compile: a compiled decode with the tail is not measured),
    # so that the plain job is judged without them (with them it is refused too, but in the last
    # rung's figures and words, which a compiled job's refusal does not carry).
    free = round(3 * GIB)
    with pytest.raises(PlanError) as plain:
        planner.plan(free, HD, shots(76), Pins(), None, rungs=False)
    with pytest.raises(PlanError) as compiled:
        planner.plan(free, HD, shots(76), Pins(), None, compile_vae=True)
    assert str(compiled.value) == str(plain.value)


def test_a_job_only_a_rung_fits_is_refused_saying_it_runs_without_compile() -> None:
    # No rung is taken compiled (DESIGN.md, Allocator, Small cards, decided on 2026-11-06): a job
    # the plain plan fits only with the decoder's tail, or with it and the VAE's least room, is
    # refused under --compile, the command line's words saying that it runs without it and with
    # which rungs, named as the plan's lines name them (_rungs_said; the library's lines are held
    # in core's test_shot_plan.py, here worded with options).
    tail = "the decoder's tail (its last blocks a frame at a time)"
    room = "the VAE's least room (the VAE passes' reserve floor lowered to 0.50 GiB)"
    for free, rungs in ((8.0, tail), (6.5, f"{tail} and {room}")):
        made, _ = planner.plan(segments_at(free), HD, shots(19), Pins(), None)
        assert made.tail_frames and not made.compile_vae
        with pytest.raises(PlanError) as refused:
            planner.plan(segments_at(free), HD, shots(19), Pins(), None, compile_vae=True)
        lines = planner.worded(refused.value).split("\n")
        # The compiled plan's own refusal first, for its decode.
        assert lines[0].startswith("an output of 1920x1080 takes "), lines[0]
        assert "(its decode in tiles of " in lines[0], lines[0]
        assert lines[1:] == [
            "--compile takes none of the floor's last rungs, not measured under --compile",
            f"The same command without --compile runs, with {rungs}",
        ]
        # The plan's own lines name the rungs in the same words.
        said = "\n".join(planner._rungs_said(made))  # pyright: ignore[reportPrivateUsage]
        assert "the decoder's tail: its last blocks a frame at a time" in said


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
    assert peaks.decode_peak(
        padded_size(HD), output_size(HD), Tiles(1024), compiled=True
    ) > peaks.decode_peak(padded_size(HD), output_size(HD), Tiles(1024))


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
    import calibration

    kinds = {
        min((frames - 1) // 4 + 1, planner.COMPILE_LATENTS)
        for frames in (calibration.FIRST, calibration.FRAMES)  # noqa: F811
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
        )
    )


def test_the_plan_says_the_compilation_and_its_time() -> None:
    free = our_free(96)
    made, peaks = planner.plan(free, HD, shots(76), Pins(), None, compile_vae=True)
    said = lines(made, peaks)
    assert (
        "plan: compile: the VAE compiled (--compile), each pass held to 1.1 times its plain peak,"
        " in the shapes measured only; compiling its 3 shapes takes about 2 min for this new job,"
        " about 45 s on a resume, its cache warm (measured on an RTX 5090's host), paid before the"
        " DiT loads, at every start and resume, its cache kept in the job's directory" in said
    )
    # No line without the option, and none where the plan is plain and nothing was asked.
    plain, plain_peaks = planner.plan(free, HD, shots(76), Pins(), None)
    assert "compile" not in lines(plain, plain_peaks)


# What /proc/self/mountinfo says (proc(5)): id, parent, device, root, mount point, options,
# optional fields, " - ", fstype, source, super options. A host with Linux's own disk as /, a
# Windows drive mounted in /mnt/d by WSL2 (a 9p mount whose super options say aname=drvfs), a
# drive mounted by the older drvfs type, a Linux disk nested in the drive, and a space escaped.
MOUNTINFO = r"""22 1 8:16 / / rw,relatime - ext4 /dev/sdd rw,discard
23 22 0:20 / /mnt/c rw,noatime - 9p C:\ rw,dirsync,aname=drvfs;path=C:\;uid=1000
24 22 0:21 / /mnt/d rw,noatime - 9p D:\ rw,dirsync,aname=drvfs;path=D:\;uid=1000
25 24 8:32 / /mnt/d/linux rw,relatime - ext4 /dev/sde rw
26 22 0:22 / /mnt/old rw,noatime - drvfs E:\ rw,uid=1000
27 22 0:23 / /mnt/with\040space rw,noatime - 9p F:\ rw,aname=drvfs;path=F:\
28 22 0:24 / /mnt/plain9p rw,noatime - 9p share rw,trans=virtio
"""


@pytest.mark.parametrize(
    ("path", "drive"),
    [
        ("/home/job/compile-cache", False),
        ("/mnt/d/Video/job/compile-cache", True),
        ("/mnt/c", True),
        ("/mnt/d/linux/job/compile-cache", False),
        ("/mnt/dd/job", False),
        ("/mnt/old/job", True),
        ("/mnt/with space/job", True),
        ("/mnt/plain9p/job", False),
    ],
)
def test_a_windows_drive_is_the_longest_mount_point_that_says_drvfs(path: str, drive: bool) -> None:
    # The longest mount point that is a prefix of the path decides (a Linux disk nested in the
    # drive is Linux's; /mnt/dd is not under /mnt/d); a 9p mount counts only with aname=drvfs.
    assert planner.on_windows_drive(Path(path), MOUNTINFO) is drive


def test_an_unreadable_mountinfo_is_linuxs_disk_said_at_debug(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def refuse(self: Path, *args: Any, **kwargs: Any) -> str:
        raise PermissionError("no")

    monkeypatch.setattr(Path, "read_text", refuse)
    caplog.set_level(logging.DEBUG)
    assert planner.on_windows_drive(Path("/mnt/d/job")) is False
    assert "mountinfo unreadable" in caplog.text
    # Lines that are not mounts are passed over.
    assert planner.on_windows_drive(Path("/mnt/d/job"), "garbage\n\n - \n") is False


# The tiles of the 1080p jobs the compilation was measured on (the RTX 5090, 2026-10-31: encode
# tiles of 1728, decode tiles of 1024): the encode's 2 shapes and the decode's 4, 6 in all.
MEASURED_1080P = (Tiles(1728), Tiles(1024))


def compile_line(
    cache: Path | None,
    monkeypatch: pytest.MonkeyPatch,
    *,
    drive: bool,
    recorded: bool = False,
    target: tuple[int, int] = HD,
    tiles: tuple[Tiles | None, Tiles | None] | None = None,
) -> str:
    """The plan's compile line of a 96 GB card's plan of `target`, its tiles those of `tiles`
    where given (the plan's own otherwise)."""
    monkeypatch.setattr(planner, "on_windows_drive", lambda path: drive)
    made, peaks = planner.plan(our_free(96), target, shots(76), Pins(), None, compile_vae=True)
    if tiles is not None:
        made = replace(made, encode_tiles=tiles[0], decode_tiles=tiles[1])
    said = planner.said(
        made,
        peaks,
        "a card",
        target,
        shots(76),
        Pins(),
        compile_cache=cache,
        recorded=recorded,
    )
    [line] = [line for line in said if line.startswith("plan: compile: ")]
    return line


@pytest.mark.parametrize(
    ("warm", "drive", "words"),
    [
        (
            False,
            False,
            "about 4 min for this new job, about 1 min 30 s on a resume, its cache warm",
        ),
        (
            True,
            False,
            "about 1 min 30 s for this resume, its cache warm, about 4 min for a new job",
        ),
        (False, True, "about 6 min 30 s for this new job, about 2 min on a resume, its cache warm"),
        (True, True, "about 2 min for this resume, its cache warm, about 6 min 30 s for a new job"),
    ],
)
def test_the_plans_line_says_the_time_of_this_job_on_this_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, warm: bool, drive: bool, words: str
) -> None:
    # 1080p as measured: 6 shapes, 240 s and 90 s a job and a resume on Linux's disk, 390 s and
    # 120 s on a Windows drive.
    cache = tmp_path / "compile-cache"
    if warm:
        (cache / "inductor").mkdir(parents=True)
        (cache / "inductor" / "kernel").write_bytes(b"x")
    line = compile_line(cache, monkeypatch, drive=drive, recorded=warm, tiles=MEASURED_1080P)
    why = ", on a Windows drive"
    assert (
        f"compiling its 6 shapes takes {words}"
        f" (measured on an RTX 5090's host{why if drive else ''}), paid" in line
    )
    assert (
        "before the DiT loads, at every start and resume, its cache kept in the job's directory"
        in line
    )
    assert "\n" not in line


@pytest.mark.parametrize(
    ("drive", "new", "resume"),
    [(False, "5 min 20 s", "2 min"), (True, "8 min 40 s", "2 min 40 s")],
)
def test_the_plans_line_at_4k_counts_its_8_shapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drive: bool, new: str, resume: str
) -> None:
    # 4K: the 96 GB card's own plan, 4 encode shapes and 4 decode, as the job measured on the RTX
    # 5090 had with its tiles of 1344 and 1024,
    # 8 x 40 s and 8 x 15 s on Linux's disk, 8 x 65 s (520 s, 8 min 40 s; 521 s measured) and
    # 8 x 20 s on a Windows drive.
    cache = tmp_path / "compile-cache"
    line = compile_line(cache, monkeypatch, drive=drive, target=UHD)
    assert f"compiling its 8 shapes takes about {new} for this new job, about {resume} on a" in line
    cache.mkdir()
    (cache / "kernel").write_bytes(b"x")
    line = compile_line(cache, monkeypatch, drive=drive, recorded=True, target=UHD)
    assert f"about {resume} for this resume, its cache warm, about {new} for a new job" in line


def test_the_plans_line_counts_the_shapes_the_warm_up_compiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An untiled pass is one shape, the frame: 2 in all for both passes untiled (80 s new, 30 s a
    # resume, "30 s" under a minute), one tiled and one not 1 + 2. Never the number of tiles.
    cache = tmp_path / "compile-cache"
    line = compile_line(cache, monkeypatch, drive=False, tiles=(None, None))
    assert "compiling its 2 shapes takes about 1 min 20 s for this new job, about 30 s on a" in line
    line = compile_line(cache, monkeypatch, drive=False, tiles=(None, Tiles(1088)))
    assert (
        "compiling its 5 shapes takes about 3 min 20 s for this new job, about 1 min 15 s" in line
    )
    one = replace(Plan(17, None, None), compile_vae=True)
    assert len(planner.compile_shapes(one, HD, shots(76), 1).encode) == 1
    assert planner._minutes_said(15) == "15 s"  # pyright: ignore[reportPrivateUsage]
    assert planner._minutes_said(60) == "1 min"  # pyright: ignore[reportPrivateUsage]
    assert planner._minutes_said(75) == "1 min 15 s"  # pyright: ignore[reportPrivateUsage]


def test_the_windows_drive_line_is_short_and_its_reason_is_said_at_debug(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Review of baton 3 (2026-10-31): the line grew to 418 characters; the figures and "on a
    # Windows drive" stay, the cache's many small files go to --verbose.
    caplog.set_level(logging.DEBUG, logger=planner.logger.name)
    cache = tmp_path / "compile-cache"
    on_linux = compile_line(cache, monkeypatch, drive=False)
    on_drive = compile_line(cache, monkeypatch, drive=True)
    assert "many small files" not in on_drive and "on a Windows drive" in on_drive
    assert len(on_drive) - len(on_linux) < 40
    # Under 360 until the line said its shapes' count ("its 3 shapes", baton 4 item a).
    assert len(on_drive) < 380
    assert "the compile cache's many small files" in caplog.text


def test_a_cache_only_a_recorded_plan_kept_as_it_is_finds_warm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A resume re-planned (other tiles, other graphs) and a first start killed while compiling,
    # with no plan recorded, find a non-empty cache that does not hold their graphs: the new
    # job's figure (review of baton 3, 2026-10-31). The recorded plan kept as it is: the
    # resume's.
    cache = tmp_path / "compile-cache"
    (cache / "inductor").mkdir(parents=True)
    (cache / "inductor" / "kernel").write_bytes(b"x")
    kept = compile_line(cache, monkeypatch, drive=False, recorded=True, tiles=MEASURED_1080P)
    assert "about 1 min 30 s for this resume, its cache warm" in kept
    for drive in (False, True):
        other = compile_line(cache, monkeypatch, drive=drive, recorded=False)
        assert "for this new job" in other and "for this resume" not in other


def test_a_cache_missing_or_empty_is_a_new_job_and_none_asked_is_linuxs_new_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "empty").mkdir()
    for cache in (tmp_path / "absent", tmp_path / "empty", None):
        assert "for this new job" in compile_line(cache, monkeypatch, drive=False, recorded=True)


# A target whose tiles are small; its padded frame (frames.padded_size) is 256x384.
SMALL = (256, 384)


def small_shapes(*lengths: int) -> Shapes:
    plan = Plan(17, None, Tiles(192), compile_vae=True)
    return planner.compile_shapes(plan, SMALL, shots(*lengths), 1)


def runs_calls(vae_models: Models, shapes: Shapes, latents: int) -> None:
    """What the run's own passes call for a shot of `latents` latents: the encode over each tile's
    crops, slice by slice (model.encode_sliced), the decode over each tile's crop of a latent
    laid out as the DiT gives it (channels last, dense), as the calibration's decodes take it
    too (calibration.calibrate)."""
    rows, columns = shapes.padded
    frames = 4 * (latents - 1) + 1  # noqa: F811
    for tile_rows, tile_columns in shapes.encode:
        crops = (
            torch.zeros((3, size, tile_rows, tile_columns), dtype=COMPUTE_DTYPE)
            for size in model.encode_slices(vae_models, frames)
        )
        for _ in model.encode_sliced(vae_models, crops, frames):
            pass
    height, width = rows // 8, columns // 8
    latent = torch.zeros((latents, height, width, 16), dtype=COMPUTE_DTYPE)
    for tile_rows, tile_columns in shapes.decode:
        top, left = slice(0, tile_rows // 8), slice(0, tile_columns // 8)
        for _ in model.decode_sliced(vae_models, latent, top, left):
            pass


def test_a_shot_of_one_latent_is_warmed_with_the_kinds_of_a_job_that_has_one() -> None:
    # The job of 2026-10-30's finding, shots of 1 latent among longer ones: a shot of 1 latent
    # is one slice of 1 frame, and its decode one of 1 latent, both warmed.
    shapes = small_shapes(1, 2, 12)
    assert shapes.latents == (1, 2, 3)
    fake = FakeVAE()
    compiling.warm(fake_models(fake), shapes)
    first = shapes.encode[0]
    assert ("encode", (1, 3, 1, *first), "DISABLED") in fake.calls
    side = tuple(each // 8 for each in shapes.decode[0])
    assert ((1, 16, 1, *side), "DISABLED", True) in fake.decoded


def test_every_kind_the_run_calls_is_warmed_for_every_shot_length() -> None:
    # Against the run's own slicing (model.encode_sliced, model.decode_sliced), the shot lengths
    # 1 to 64 latents, each alone in a job: whatever the run calls of the compiled VAE, the
    # warm-up has called, the decoder's latent as the DiT gives it. The calibration's lengths (2
    # and 3 latents) the same, their decodes taking the latent in the DiT's layout too
    # (calibration.calibrate). No length shows a kind of its own at the end of a shot
    # (planner.compile_shapes).
    for latents in range(1, 65):
        shapes = small_shapes(latents)
        warm = FakeVAE()
        compiling.warm(fake_models(warm), shapes)
        ran = FakeVAE()
        runs_calls(fake_models(ran), shapes, latents)
        assert set(ran.calls) <= set(warm.calls), latents
        assert set(ran.decoded) <= set(warm.decoded), latents
    for latents in planner.COMPILE_CALIBRATED:
        shapes = small_shapes(40)
        warm = FakeVAE()
        compiling.warm(fake_models(warm), shapes)
        calibration = FakeVAE()
        runs_calls(fake_models(calibration), shapes, latents)
        assert set(calibration.calls) <= set(warm.calls), latents
        assert set(calibration.decoded) <= set(warm.decoded), latents


def test_the_decoders_latent_is_warmed_in_the_layout_of_the_dit_alone() -> None:
    # The finding of 2026-10-30: 12 recompiles of the decoder in the run, the guards on the
    # latent's stride, after a warm-up with the encode's channel-major layout alone; the ruling of
    # baton 3 after it: one layout, the DiT's, for the run's decodes and the calibration's.
    shapes = small_shapes(40)
    assert shapes.latents == planner.COMPILE_CALIBRATED == (2, 3)
    fake = FakeVAE()
    compiling.warm(fake_models(fake), shapes)
    assert fake.decoded and all(dense for _, _, dense in fake.decoded)
    # The shots' own (3 latents: a first slice of 2, the next ones of 1) and the calibration's:
    # its 5-frame pass (2 latents, one slice) and its 13-frame one (3 and 4 latents' slices).
    assert {(shape[2], state) for shape, state, _ in fake.decoded} == {
        (2, "DISABLED"),
        (2, "INITIALIZING"),
        (1, "ACTIVE"),
    }
    # Each (shape, kind) once: a second layout would call the decoder again with the same ones.
    assert len(fake.decoded) == len(set(fake.decoded))


LIMITS_SAID = (
    "recompile limit is %d for a code object's entries guarding on the same objects and %d for"
    " all of a code object's entries across its regions"
)


def test_the_recompile_limits_are_raised_past_any_warm_up_and_never_lowered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    undo_global_state: None,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The review of baton 3 (2026-10-31): torch 2.14.1 applies recompile_limit per code object,
    # and the VAE's blocks share theirs (one reached 125 against 128 on a 1080p job of 6 shapes);
    # a limit that no warm-up reaches, 1024 and 8192 in all, where torch's own are 8 and 256.
    caplog.set_level(logging.DEBUG, logger=compiling.__name__)
    dynamo = torch._dynamo.config
    assert (compiling.RECOMPILE_LIMIT, compiling.ACCUMULATED_RECOMPILE_LIMIT) == (1024, 8192)

    def load() -> tuple[int, int]:
        Recorder(monkeypatch)
        model.load_models(
            Path("dit.safetensors"),
            Path("vae.safetensors"),
            torch.device("cpu"),
            None,
            Plan(5, compile_vae=True).model_plan,
            Compilation(small_shapes(12), tmp_path / "cache"),
        )
        return dynamo.recompile_limit, dynamo.accumulated_recompile_limit

    dynamo.recompile_limit, dynamo.accumulated_recompile_limit = 8, 256
    assert load() == (1024, 8192)
    assert LIMITS_SAID % (1024, 8192) in caplog.text
    # A limit the process already set higher is kept.
    dynamo.recompile_limit, dynamo.accumulated_recompile_limit = 4096, 100000
    assert load() == (4096, 100000)
    assert LIMITS_SAID % (4096, 100000) in caplog.text


@pytest.mark.parametrize("suppressing", [False, True])
@pytest.mark.parametrize("before", [False, True])
def test_a_limit_hit_in_the_warm_up_is_a_refusal_and_the_setting_goes_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    undo_global_state: None,  # noqa: F811
    before: bool,
    suppressing: bool,
) -> None:
    # fail_on_recompile_limit_hit is on during the warm-up, where torch would run the frame
    # eagerly and say nothing; torch's FailOnRecompileLimitHit is an Exception, not a
    # RuntimeError, and becomes the job's refusal naming the limit and the way out. Afterwards the
    # setting is as it was (a hit in the run is torch's warning, report_late reports the graphs).
    # suppress_errors (TORCHDYNAMO_SUPPRESS_ERRORS) set beforehand would swallow the exception in
    # convert_frame and torch calls the two incompatible: it is off during the warm-up, and as it
    # was after.
    recorder = Recorder(monkeypatch)
    dynamo = torch._dynamo.config
    seen: list[tuple[bool, bool]] = []
    dynamo.fail_on_recompile_limit_hit = before
    dynamo.suppress_errors = suppressing

    def warm(models: Models, shapes: Shapes) -> None:
        seen.append((dynamo.fail_on_recompile_limit_hit, dynamo.suppress_errors))

    monkeypatch.setattr(compiling, "warm", warm)

    def load() -> None:
        model.load_models(
            Path("dit.safetensors"),
            Path("vae.safetensors"),
            torch.device("cpu"),
            None,
            Plan(5, compile_vae=True).model_plan,
            Compilation(Shapes((128, 128), (), (), ()), tmp_path / "cache"),
        )

    load()
    assert seen == [(True, False)]
    assert dynamo.fail_on_recompile_limit_hit is before and dynamo.suppress_errors is suppressing

    def hit(models: Models, shapes: Shapes) -> None:
        raise FailOnRecompileLimitHit("Hard failure due to fail_on_recompile")

    assert not issubclass(FailOnRecompileLimitHit, RuntimeError)
    monkeypatch.setattr(compiling, "warm", hit)
    with pytest.raises(compiling.CompileError) as error:
        load()
    assert isinstance(error.value, Refusal)
    assert "recompile limit" in str(error.value)
    assert "the same command without --compile" in str(error.value)
    assert dynamo.fail_on_recompile_limit_hit is before and dynamo.suppress_errors is suppressing
    assert recorder.events[-1] == "workers shut"


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

        def load_weights(module: Any, path: Path, *rest: Any, **more: Any) -> dict[str, Any]:
            self.events.append(f"weights {path.name}")
            return {}

        monkeypatch.setattr(model, "_load_weights", load_weights)
        # What the check finds in the files, as it says it: the 7B in fp16 and the VAE.
        monkeypatch.setattr(
            model.weights,
            "check",
            lambda path, kind: model.weights.DIT_7B if kind == "dit" else model.weights.VAE,
        )


def test_the_vae_is_compiled_before_the_dit_loads_and_the_workers_shut_before_its_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    undo_global_state: None,  # noqa: F811
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
        plan.model_plan,
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
    assert compilation.seconds >= 0 and models.model_plan == plan.model_plan
    # The causal convolutions left out before anything compiled.
    convolutions = [m for m in models.runner.vae.modules() if isinstance(m, InflatedCausalConv3d)]
    assert convolutions and all("forward" in vars(m) for m in convolutions)


def test_a_plan_without_compile_loads_as_it_did(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    undo_global_state: None,  # noqa: F811
) -> None:
    recorder = Recorder(monkeypatch)
    model.load_models(Path("dit.safetensors"), Path("vae.safetensors"), torch.device("cpu"))
    model.load_models(
        Path("dit.safetensors"),
        Path("vae.safetensors"),
        torch.device("cpu"),
        None,
        Plan(5).model_plan,
    )
    assert recorder.events == ["weights vae.safetensors", "weights dit.safetensors"] * 2
    with pytest.raises(ValueError, match="nothing to compile the VAE with"):
        model.load_models(
            Path("dit.safetensors"),
            Path("vae.safetensors"),
            torch.device("cpu"),
            None,
            Plan(5, compile_vae=True).model_plan,
        )


def test_the_workers_shut_whatever_happens_and_a_failure_is_the_jobs_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    undo_global_state: None,  # noqa: F811
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
            Plan(5, compile_vae=True).model_plan,
            compilation,
        )
    assert isinstance(error.value, Refusal)
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
            Plan(5, compile_vae=True).model_plan,
            compilation,
        )
    assert recorder.events[-1] == "workers shut"


# ----------------------------------------------------------------------------- the environment


def test_inductors_modules_are_imported_before_the_environment_is_recorded() -> None:
    # What a compilation imports and the run otherwise doesn't (measured on a small module
    # compiled on the CPU, torch 2.14.1): the packages recorded with --compile are the packages
    # the compilation will have imported, at a start and at a resume alike.
    code = (
        "import environment as e\n"
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
        "import environment as e\n"
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

import environment  # noqa: E402
import main as cli  # noqa: E402
import timing  # noqa: E402


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
    """The stand-in's encode and decode (steps) taking the plan's tiles and scratch as the run
    gives them, around their own, untiled, pass: what these tests ask of them is the
    plan's, not the tiled passes' own (tests/test_tiling.py, tests/test_shot_pass.py)."""
    import run

    def encode(*arguments: Any, tiles: Any, store: Any, **options: Any) -> Any:
        return steps.encode(*arguments, **options)

    def decode(*arguments: Any, tiles: Any, store: Any, **options: Any) -> None:
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
    assert plan.compile_vae
    assert compilation.cache == tmp_path / "out" / "compile-cache"
    # 704x560 frames: the decode in tiles of 640 px (the frame whole is no compiled decode), the
    # encode the frame whole, which the 1080p frame measured holds; JOB's shots are 2, 1 and 6
    # latents.
    # The tiles are the plan's, not the loaded models': the manifest records them.
    assert out["plan"]["encode_tiles"] is None
    assert out["plan"]["decode_tiles"] == {"size": 640, "overlap": Tiles(640).overlap}
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
    # line says the new job's time from its shapes (3 here, 40 s each: 2 min on Linux's disk,
    # whatever disk the test's directory is on), and the time of the load, which the compilation
    # is part of, says its share of it, once, in the load's own seconds.
    monkeypatch.setattr(planner, "on_windows_drive", lambda path: False)
    Loads(monkeypatch, seconds=300.0)
    accepting_tiles(monkeypatch, steps)
    caplog.set_level(logging.INFO)
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB, *LARGER, "--compile", "--plan", "p.txt") == 0
    assert "compiling its 3 shapes takes about 2 min for this new job, about 45 s on a resume" in (
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
    from calibration import Calibrated

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


def test_the_run_names_the_unit_that_compiled_after_the_warm_up(
    frames: Any,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    # run._made, after each part of a unit, checks (and calibration's end does): through the real
    # function with a progress stand-in.
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
    assert record.getMessage().startswith(
        "shot 0's decode: the compiler compiled, or tried to compile, 1 more graph"
    )


# ---- --compile on Windows ----------------------------------------------------------------------

WINDOWS_REFUSAL = (
    "--compile: torch.compile makes the VAE's GPU code with Triton, which PyTorch's Windows build"
    " doesn't have; on Windows, run without --compile"
)


@pytest.mark.parametrize(("platform", "refused"), [("win32", True), ("linux", False)])
def test_compile_is_refused_on_windows_with_a_video_s_arguments(
    monkeypatch: pytest.MonkeyPatch, platform: str, refused: bool
) -> None:
    # A video's argument, refused with the others of a video's, before the source is looked at
    # (main._run); the platform set around that check alone, so that no module imported meanwhile
    # takes it for its own. On Linux the arguments pass, the video's defaults taken.
    from argparse import Namespace

    from seedvr2x.errors import JobError

    args = Namespace(
        compile=True,
        plan=None,
        format=None,
        output=Path("out"),
        resolution=None,
        quality=None,
        min_segment=None,
    )
    with monkeypatch.context() as windows:
        windows.setattr(sys, "platform", platform)
        if refused:
            with pytest.raises(JobError) as error:
                cli._video_arguments(args)  # pyright: ignore[reportPrivateUsage]
            assert str(error.value) == WINDOWS_REFUSAL
        else:
            cli._video_arguments(args)  # pyright: ignore[reportPrivateUsage]
    assert (args.resolution is None) is refused


def test_the_lock_s_torch_needs_triton_on_linux_alone() -> None:
    # The refusal's reason, as the lock resolves torch 2.14.1 for both platforms.
    import tomllib

    lock = tomllib.loads((Path(__file__).resolve().parent.parent / "uv.lock").read_text())
    torch_entry = next((each for each in lock["package"] if each["name"] == "torch"), None)
    assert torch_entry is not None, "no torch in the lock"
    triton = next((each for each in torch_entry["dependencies"] if each["name"] == "triton"), None)
    assert triton is not None, "torch's entry in the lock lists no triton"
    assert triton.get("marker") == "sys_platform != 'win32'"

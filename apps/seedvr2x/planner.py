"""The command line's side of the memory planner (DESIGN.md, Memory planner). The plan itself, one
per job, is the library's (seedvr2x/shot_plan.py, plan); here, what a job does with it: the plan's
lines (said, card_detail), the smaller DiT files that would plan the job (plan_naming,
smaller_swaps), a resume's plan made again or checked against its record (plan_again,
check_recorded), the shapes a compiled VAE is warmed with and the time they take
(compile_shapes, on_windows_drive), and where a tiled shot's scratch goes (scratch_file). Nor
torch at import, as the library's half: the command line imports this module before its first
refusals, which don't wait for torch (tests/test_cli.py)."""

import logging
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any

from job import QUALITY, kept_bytes, shot_runs
from seedvr2x import memory, peaks, pull
from seedvr2x.errors import PinRefused, Refusal, Setting
from seedvr2x.frames import output_size, padded_size
from seedvr2x.memory import Reading
from seedvr2x.passes import Plan, Shapes, Tiles
from seedvr2x.peaks import MARGIN, VAE_ROOM
from seedvr2x.shot_plan import (
    DIT_FORMATS,
    Peaks,
    Pins,
    PlanError,
    Shrunk,
    _check_tiles,  # pyright: ignore[reportPrivateUsage]
    _count,  # pyright: ignore[reportPrivateUsage]
    _early_said,  # pyright: ignore[reportPrivateUsage]
    _gib,  # pyright: ignore[reportPrivateUsage]
    _held_under,  # pyright: ignore[reportPrivateUsage]
    _host_hundredths,  # pyright: ignore[reportPrivateUsage]
    _Job,  # pyright: ignore[reportPrivateUsage]
    _job,  # pyright: ignore[reportPrivateUsage]
    _latents,  # pyright: ignore[reportPrivateUsage]
    _left,  # pyright: ignore[reportPrivateUsage]
    _named,  # pyright: ignore[reportPrivateUsage]
    _resident_said,  # pyright: ignore[reportPrivateUsage]
    held_note,
    plan,
)
from seedvr2x.shots import SHARED, Shot
from seedvr2x.weights import Architecture

logger = logging.getLogger("cli.planner")


def option(setting: Setting) -> str:
    """The command line's option for a setting a library refusal names as its way out
    (errors.Setting): "--" and the setting's name with its underscores as hyphens, then its value
    ("--window 6", "--decode-tile 512", "--resolution" for a setting named with none). The say
    that every library refusal's words (Refusal.words) reach the user through."""
    return "--" + setting.name.replace("_", "-") + (f" {setting.value}" if setting.value else "")


# The compilation's time, said by --plan and the plan's line: a new job's and a resume's (its
# cache warm), for the job's directory on Linux's own disk and on a Windows drive under WSL2, paid
# before the DiT loads (DESIGN.md, Memory planner, --compile, decided on 2026-10-30: "the plan's
# line says the times measured, a new job's and a resume's"), counted in seconds a shape: the
# shapes the warm-up compiles, the encode's and the decode's tile shapes, interior and edges, each
# at every slice kind (compile_shapes; design's ruling of baton 4, 2026-10-31, after the 4K job's
# 8 shapes took 8 min 42 s where the 1080p job's 6 shapes, 6 min 23 s to 6 min 28 s, had set the
# total at 6 min 30 s for any job). Measured on 2026-10-31 on an RTX 5090 with an AMD Ryzen 9
# 9950X3D under WSL2, 1080p (encode tiles of 2 shapes, decode tiles of 4) and 4K
# (4 and 4): with the job's directory on a Windows drive (D:, a 9p mount, "aname=drvfs"; code
# subset b514b336) a new job took 383 to 388 s at 1080p (6 shapes) and 521 s at 4K (8): 64.5 to
# 65.1 s a shape, and a resume 109 s at 1080p: 18 s a shape; on ext4 (code subset 113640f7,
# whose warm-up made both memory layouts of the decoder's latent, where the one-layout warm-up
# that followed drops the channel-major pass, about 5 s and its graphs, and the GPU gate measures
# these figures again) a new job took 240 s at 1080p: 40 s a shape, and a resume 91 s: 15 s a
# shape. The same graphs on both disks (535 on 113640f7), the difference the compile cache's
# small files (8,273, 359.5 MB, in the final 1080p jobs' caches) written to and read back from
# the drive. Each shape is warmed at the slice kinds of shots of 2 and of 3 latents or more, as
# every job measured was; a job with a shot of 1 latent adds a third kind, which these figures
# don't count. A resume's floor is dynamo tracing every
# graph again (about 77 s of its 84 s warm-up), which no cache keeps. The figures here are
# whole seconds a shape, fitted on 1080p's 6 shapes and 4K's 8 on one host: 1080p says 4 min and
# 1 min 30 s on ext4, 6 min 30 s and 2 min on the drive; 4K 5 min 20 s and 2 min on ext4, 8 min
# 40 s and 2 min 40 s on the drive. The estimates that follow count the seconds
# the load measured, the compilation's among them, once (main._say_times): there is no figure to
# sum before the load. PROVISIONAL (impl-v010, 2026-10-31: fitted on one host; the 96 GB box
# measures next, inductor being CPU work).
COMPILE_SHAPE_NEW = 40
COMPILE_SHAPE_RESUME = 15
COMPILE_SHAPE_NEW_DRIVE = 65
COMPILE_SHAPE_RESUME_DRIVE = 20
# The slices a compiled shape is warmed with are the shot's: one of 1 or 2 latents is a single
# slice, the encode's and the decode's memory state off, one of 3 or more has a first slice and
# the next ones (model.encode_slices, model.latent_slices), so that 3 latents show every kind
# (compile_shapes).
COMPILE_LATENTS = 3
# The kinds the time estimates' calibration meets besides the job's own (calibration.py,
# FIRST and FRAMES): an encode and a decode of 5 frames, 2 latents, one slice, and of 13 frames,
# 4 latents, a first slice and next ones (kind 3). Compiled with the rest, so that the
# calibration, which times its passes, never pays for a compilation: it would put it into the
# models of the encode and the decode (tests/test_compile.py holds them to the calibration's).
COMPILE_CALIBRATED = (2, 3)


def compile_shapes(
    plan: Plan,
    target: tuple[int, int],
    shots: Sequence[Shot],
    runs: int,
) -> Shapes:
    """The shapes the plan's VAE passes meet, for the compiled VAE to be warmed with before the
    DiT loads (seedvr2x/compiling.py): one per distinct tile of each pass (tile_shapes), the whole
    padded frame untiled, and the slice kinds of the job's runs, `runs` a shot (job.QUALITY):
    a run of 1 latent is a slice of 1 frame, of 2 a slice of 5, of 3 or more a first slice of 5
    and the next ones of 4 (model.encode_slices, model.latent_slices), so that the shortest of
    each kind, up to COMPILE_LATENTS, shows them all. A compiled pass compiles once for each
    shape, for each of these (DESIGN.md: "a warm call per shape the plan uses ... at the job's
    slice lengths"), so that no unit of the run compiles; and the calibration's own
    (COMPILE_CALIBRATED), which times the passes right after, a job whose shots are all long
    meeting the 2 latents of its 5-frame passes there only.

    A shot's last slice is no kind of its own: its frames are 4n + 1 and its latents whole, so
    that the cut leaves no rest (model.encode_slices, model.latent_slices), and every slice
    after the first is of one size, whatever the shot's length (tests/test_compile.py holds
    lengths 1 to 64 to the run's own slicing). The decoder's latent is warmed in one memory
    layout, the DiT's, at all these lengths: the calibration's decodes take it in that layout
    too (calibration.calibrate; design's ruling of 2026-10-30)."""
    padded = padded_size(target)

    def cut(tiles: Tiles | None) -> tuple[tuple[int, int], ...]:
        return (padded,) if tiles is None else peaks.tile_shapes(padded, tiles)

    latents = sorted(
        {
            *COMPILE_CALIBRATED,
            *(
                min(_latents(run.frames), COMPILE_LATENTS)
                for shot in shots
                for run in shot_runs(shot, runs)
            ),
        }
    )
    return Shapes(padded, cut(plan.encode_tiles), cut(plan.decode_tiles), tuple(latents))


def scratch_file(plan: Plan, target: tuple[int, int], frames: int) -> int:
    """What the scratch of a shot of `frames` frames takes on disk under plan, in bytes: its
    memory-mapped file in the shot's unit directory (scratch.Scratch) where the plan tiles a pass
    and the scratch is over the plan's allowance, as scratch.Scratch.where decides; 0 where no pass
    is tiled or the scratch is held in system memory. The encode's and the decode's are the same
    store, one at a time (scratch); a shot's first run is its longest (job.shot_runs)."""
    if plan.encode_tiles is None and plan.decode_tiles is None:
        return 0
    size = peaks.scratch(padded_size(target), frames)
    return size if size > plan.scratch_allowance else 0


def recorded_dit_model(memory_record: dict[str, Any]) -> Architecture:
    """The DiT's format a plan's record says its peaks were predicted for (Peaks.record's
    dit_format), for whoever makes the plan again from a manifest: fp16 where the record has
    none, a manifest from before int8 and the GGUF, whose plans were all fp16's. KeyError for a name
    DIT_FORMATS doesn't know."""
    return DIT_FORMATS[memory_record.get("dit_format", "fp16")]


def worded(error: BaseException) -> str:
    """An error's text as the command line says it: a library refusal's message with the
    settings it names worded as options (Refusal.words, option), any other error's own."""
    return error.words(option) if isinstance(error, Refusal) else str(error)


def pin_named(refused: PlanError) -> bool:
    """Whether a refusal ends with the pin that fits (PlanError.instead is a window, a block
    count or a tile). The floor refusal's instead is the resolution that fits, which a smaller
    DiT file can still serve, so plan_naming adds its way out to it as it did when that refusal
    carried none."""
    return refused.instead is not None and refused.instead.name != "resolution"


def plan_naming(
    free: int | Reading,
    target: tuple[int, int],
    shots: Sequence[Shot],
    pins: Pins,
    host_available: int | None,
    *,
    dit_file: str,
    model_dir: bool = False,
    **options: Any,
) -> tuple[Plan, Peaks]:
    """`plan` for a new job whose DiT file is dit_file (`model_dir`: the run was given
    --model-dir, which the way out then says, smaller_way), a refusal for memory adding the
    way out of a smaller DiT file: the same job planned again in int8's figures and in the GGUF's
    (the smaller formats of the DiT's, pull.smaller_files), and those that plan it named with
    their option and size (smaller_way), the file never changed by itself (DESIGN.md, Weights,
    "The dynamic GGUF before v0.1.0 too"). Left as plan has it: a pin's refusal, which ends with
    the pin that fits (PlanError.instead), a refusal that is no memory's (no phase), the
    compiled VAE's, whose advice has lines of its own, and a recorded shape's (a resume's job
    has its file, and a smaller one is another job); and a DiT in the smallest format, or one
    that no smaller file plans for."""
    try:
        return plan(free, target, shots, pins, host_available, **options)
    except PlanError as error:
        refused = error
    dit_model = options.get("dit_model", peaks.FP16)
    if (
        options.get("recorded") is not None
        or not refused.phase
        or pin_named(refused)
        or "\n" in refused.words(option)
    ):
        raise refused

    def plans(each: Architecture) -> bool | None:
        again: dict[str, Any] = {**options, "dit_model": each}
        try:
            plan(free, target, shots, pins, host_available, **again)
        except PlanError:
            return None
        return True

    found = pull.smaller_planning(dit_file, dit_model, plans)
    if not found:
        raise refused
    way = smaller_way([each for each, _ in found], model_dir)
    # The way out sits beside the refusal's own (a pin's, the fewest blocks that hold it), before
    # a held run's note on the reserves its figures were found with (held_note).
    said, note = refused.words(option), held_note()
    if said.endswith(note):
        said = said[: -len(note)]
    else:
        note = ""
    raise refused.reworded(f"{said}; or {way}{note}") from None


def smaller_swaps(
    made: Plan,
    predicted: Peaks,
    arguments: tuple[int | Reading, tuple[int, int], Sequence[Shot], Pins, int | None],
    options: dict[str, Any],
    dit_file: str,
) -> str | None:
    """The plan's line of the smaller DiT files that plan the job better than `made`, the plan of
    a new job with the DiT file dit_file (said's `smaller`), `predicted` its Peaks: each smaller
    format (pull.smaller_files) planned for the same job, `arguments` as plan takes them and
    `options` its keywords, and named where it swaps fewer blocks than made.swapped or runs a
    longer window (Peaks.window, in latents: the host's memory cuts float16's window where its
    pinned copies don't fit, DESIGN.md, Weights, "The dynamic GGUF before v0.1.0 too"). The line
    says both figures for this plan and for each file named. None where no smaller format is
    better by either, or for a recorded plan (a resume's file is its job's). The file is never
    changed by itself; --dit-model names them."""
    if options.get("recorded") is not None:
        return None

    def better(each: Architecture) -> pull.Swaps | None:
        again: dict[str, Any] = {**options, "dit_model": each}
        try:
            other, other_peaks = plan(*arguments, **again)
        except PlanError:
            return None
        if other.swapped < made.swapped or other_peaks.window > predicted.window:
            return pull.Swaps(other.swapped, other_peaks.window)
        return None

    found = pull.smaller_planning(dit_file, options.get("dit_model", peaks.FP16), better)
    if not found:
        return None
    return smaller_line(pull.Swaps(made.swapped, predicted.window), found)


def swaps_said(swaps: pull.Swaps, join: str = "") -> str:
    """The blocks a plan swaps and the longest window it runs, in latents, as the line of the
    smaller files says them: "11 blocks with windows of 10 latents" for the plan itself (`join`
    "with"), "19 blocks, windows of 17" for a file; the blocks alone for a picture."""
    blocks = f"{swaps.swapped} block{'' if swaps.swapped == 1 else 's'}"
    if swaps.window is None:
        return blocks
    if join:
        return f"{blocks} {join} windows of {swaps.window} latent{'' if swaps.window == 1 else 's'}"
    return f"{blocks}, windows of {swaps.window}"


def smaller_way(found: Sequence[pull.Smaller], model_dir: bool = False) -> str:
    """The way out a refusal for memory adds when the DiT is not in the smallest format and a
    smaller file plans the job (pull.smaller_planning), as the command line says it, with its
    options: "a smaller DiT file that plans this job: int8 (--dit-model F, 8.3 GB) or the dynamic
    GGUF (--dit-model G, 4.8 GB)"; for a file that is not a pinned one, each 7B's files after "for
    the sharp 7B," and "for the 7B,", separated by semicolons. Where the run was given --model-dir
    (`model_dir`), the way out says so once, after "this job": the pinned files are named, but
    only a file put in that directory is read there, nothing being fetched. The job's file is
    never changed by itself."""
    groups: dict[str | None, list[str]] = {}
    for one in found:
        file = option(Setting("dit_model", one.name))
        groups.setdefault(one.model, []).append(f"{one.label} ({file}, {pull.gigabytes(one.size)})")
    said = [
        (f"for {model}, " if model else "") + " or ".join(each) for model, each in groups.items()
    ]
    here = ", put in --model-dir" if model_dir else ""
    return f"a smaller DiT file that plans this job{here}: " + "; ".join(said)


def smaller_line(this: pull.Swaps, found: Sequence[tuple[pull.Smaller, pull.Swaps]]) -> str:
    """The plan's line of the smaller DiT files that plan the job better than the plan's own
    (pull.smaller_planning's pairs, each file with what its plan makes of the job, `this` being
    what the plan itself makes), as the command line says it: "smaller DiT files plan this job
    better: this plan swaps 11 blocks with windows of 10 latents; int8 19 blocks, windows of 17;
    the dynamic GGUF 17 blocks, windows of 19; --dit-model names them". For a file that is not a
    pinned one, each 7B's files after "for the sharp 7B" and "for the 7B", in brackets, separated
    by semicolons. The file is never changed by itself."""
    groups: dict[str | None, list[str]] = {}
    for one, swaps in found:
        groups.setdefault(one.model, []).append(f"{one.label} {swaps_said(swaps)}")
    listed = "; ".join(
        f"for {model} ({'; '.join(each)})" if model else "; ".join(each)
        for model, each in groups.items()
    )
    return (
        f"smaller DiT files plan this job better: this plan swaps {swaps_said(this, 'with')};"
        f" {listed}; {option(Setting('dit_model'))} names them"
    )


# The parts of a shape that plan_again releases for a refused phase, in order (Pins' fields): the
# phase's own, and for the system's memory, which no phase names, the tiles before the window
# (the planner's own order for the host, _dit_kept before _window_that_runs: DESIGN.md, Memory
# planner, Choices, decided on 2026-10-13, "before the window shortens, the DiT stays on the GPU
# for the VAE's phases and the tiles shrink"). PROVISIONAL (impl-v010, 2026-10-28): DESIGN.md
# names the phase's part ("the component of the phase the refusal names", the brief's design),
# not what the system's memory releases.
RELEASED = {
    "encode": ("encode_tile",),
    "dit": ("window",),
    "decode": ("decode_tile",),
    "host": ("decode_tile", "encode_tile", "window"),
}


def plan_again(
    free: int | Reading,
    target: tuple[int, int],
    shots: Sequence[Shot],
    pins: Pins,
    host_available: int | None,
    *,
    recorded: Plan,
    refused: PlanError,
    split: bool = False,
    attention: str = peaks.FLASH_ATTN_2,
    dit_model: Architecture = peaks.FP16,
) -> tuple[Plan, Peaks]:
    """The plan of a job resumed whose recorded shape no longer fits the memory free (`refused`,
    plan's refusal of `recorded`), planned again for what is left (DESIGN.md, Memory planner, A
    plan shapes the output a little, decided on 2026-10-28, with --accept-env-change): what still
    fits of the shape is kept, the rest is planned as a new job's is.

    The recorded cap and tiles are pinned (a pin is taken exactly or refused: `Pins`), the
    command line's own pins as given. While the plan is refused, the part of the shape the
    refusal's phase names is released (PlanError.phase; RELEASED: the encode's tiles, the decode's
    tiles, the window cap, the system's memory taking the tiles first), unless the command line
    pins it, and the plan made again; at most three rounds, one part each. Refused with nothing
    left to release, `refused` itself when none was released, else the last round's refusal, a new
    job's.

    The result is a plan whose parts are the recorded ones or the planner's own (Plan.window,
    Plan.encode_tiles, Plan.decode_tiles): the caller compares them. The planning is plan's, the
    reserves and the tight reserve included, so that what is refused here is refused as any new job
    is."""
    held = Pins(
        pins.window if pins.window is not None else recorded.window,
        pins.encode_tile if pins.encode_tile is not None else _tile_pin(recorded.encode_tiles),
        pins.decode_tile if pins.decode_tile is not None else _tile_pin(recorded.decode_tiles),
        pins.swap_blocks,
    )
    # The parts the command line leaves to the planner, which are the ones that can be released.
    open_to = {
        "window": pins.window is None and recorded.window is not None,
        "encode_tile": pins.encode_tile is None,
        "decode_tile": pins.decode_tile is None,
    }
    error = refused
    for _ in range(len(open_to)):
        part = next((part for part in RELEASED.get(error.phase, ()) if open_to.get(part)), None)
        if part is None:
            raise error
        open_to[part] = False
        held = replace(held, **{part: None})
        try:
            return plan(
                free,
                target,
                shots,
                held,
                host_available,
                split=split,
                attention=attention,
                compile_vae=recorded.compile_vae,
                compile_fixed=True,
                dit_model=dit_model,
            )
        except PlanError as again:
            error = again
    raise error


def _tile_pin(tiles: Tiles | None) -> int:
    """The pin that makes a pass's tiles as recorded: their side, 0 for the untiled pass."""
    return 0 if tiles is None else tiles.size


def check_recorded(
    recorded: Plan,
    shots: Sequence[Shot],
    target: tuple[int, int],
) -> None:
    """Refuse (PlanError) a recorded plan that the planner makes for no job of `shots` resized to
    target (frames.target_size), which Plan.recorded, with neither, can't tell from one it does:
    - a cap no layout exists for (shots.window_layout): one under 2 * SHARED + 1 latents that a
      shot is longer than. The planner makes a cap that short only for a job whose longest shot
      is as short, each shot then one window (_Job.smallest), so that record was edited;
    - tiles of which one covers the whole padded frame, which is no tiling (covers): the planner
      makes none, and refuses the pin of one (plan); or tiles the tiled passes lay no grid of
      (_check_tiles)."""
    longest = max(_latents(shot.frames) for shot in shots)
    if recorded.window is not None and recorded.window < min(2 * SHARED + 1, longest):
        raise PinRefused(
            f"not a plan as seedvr2x records one: its window, {recorded.window}, with a shot of"
            f" {_count(longest, 'latent')}: a cap that a shot is longer than is"
            f" {2 * SHARED + 1} latents at least",
            why="recorded",
        )
    _check_tiles(padded_size(target), recorded)


def _quality_said(shots: Sequence[Shot], target: tuple[int, int], quality: str) -> str:
    """The plan's line of the --quality level (said), at every level, standard's too (DESIGN.md,
    Memory planner, Every choice of the plan is said): the runs a shot and in all, how a frame
    is taken, the GPU time against standard's one run, the memory one run's, the disk the frames
    kept between a shot's runs take, and the levels that give or cost more. The time here is a
    multiple of one run's, about runs_per_shot times, a later run being shorter by its first frame
    (DESIGN.md, Several runs of a shot: "4x the GPU time"): each level's time in seconds needs the
    calibration, which comes after the plan's lines, the models loaded (main._say_times)."""
    runs_per_shot = QUALITY[quality]
    if runs_per_shot == 1:
        return (
            "quality: standard, the default, one run a shot; --quality enhanced or maximum runs 2"
            " or 4 and takes each frame from the one that rebuilds it best, for 2 or 4 times the"
            " GPU time"
        )
    runs = sum(len(shot_runs(shot, runs_per_shot)) for shot in shots)
    # The frames a shot's earlier runs give wait on disk, at 16 bits, in the shot's unit directory
    # until its last run's decode (run.Kept): the most a shot keeps, said so that a user sees the
    # room it takes before anything runs.
    kept = max(kept_bytes(shot, runs_per_shot, output_size(target)) for shot in shots)
    levels = [(level, count) for level, count in QUALITY.items() if level != quality]
    others = (
        f"{' or '.join(level for level, _ in levels)}:"
        f" {' or '.join(str(count) for _, count in levels)} runs"
    )
    return (
        f"quality: {quality}, {runs_per_shot} runs a shot, {runs} in all, each frame from its best"
        f" run: about {runs_per_shot} times standard's GPU time, one run's memory;"
        f" {kept / 2**30:.2f} GiB of frames on disk for the longest shot; --quality {others}"
    )


# What a run holds of a shot at the moment it holds the most, in the plan's line of the system
# memory (said): _MOMENT_SAID's moments, shorter, the line giving every figure beside them.
_MOMENT_SHORT = {
    "check": "as its latent is checked",
    "windows": "at its longest window",
    "merge": "as its windows are merged",
}


def _escaped(field: str) -> str:
    """A mountinfo field with its octal escapes (\\040 for a space) undone (proc(5))."""
    return re.sub(r"\\([0-7]{3})", lambda found: chr(int(found.group(1), 8)), field)


def on_windows_drive(path: Path, mountinfo: str | None = None) -> bool:
    """Whether `path` lies on a Windows drive under WSL2: the longest mount point that is a
    prefix of its resolved path, in /proc/self/mountinfo (or the text given), is a 9p mount whose
    super options say "aname=drvfs", or a drvfs one. Never fails the run: an unreadable
    mountinfo says no, with the reason at debug, so that the figures are Linux's disk's
    (DESIGN.md, Memory planner, --compile, decided on 2026-10-30; the ruling of baton 3,
    2026-10-30)."""
    if mountinfo is None:
        try:
            mountinfo = Path("/proc/self/mountinfo").read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            logger.debug(
                "mountinfo unreadable (%s): the compile cache taken as on Linux's disk", error
            )
            return False
    try:
        where = PurePosixPath(path.resolve())
    except (OSError, RuntimeError) as error:
        logger.debug(
            "%s not resolved (%s): the compile cache taken as on Linux's disk", path, error
        )
        return False
    best: tuple[int, bool] | None = None
    for line in mountinfo.splitlines():
        head, separator, tail = line.partition(" - ")
        fields, rest = head.split(), tail.split()
        # Mount point is the fifth field; after the separator: fstype, source, super options.
        if not separator or len(fields) < 5 or len(rest) < 3:
            continue
        mount = PurePosixPath(_escaped(fields[4]))
        if mount != where and mount not in where.parents:
            continue
        # WSL2 writes the options as "aname=drvfs;path=C:\;uid=1000,...": split at both.
        drive = rest[0] == "drvfs" or (
            rest[0] == "9p" and "aname=drvfs" in re.split(r"[,;]", rest[2])
        )
        if best is None or len(mount.parts) >= best[0]:
            best = (len(mount.parts), drive)
    return best is not None and best[1]


def _minutes_said(seconds: int) -> str:
    """A duration in whole seconds as the plan's line says it, to the second as it is given (the
    compile times are multiples of 5 s): "4 min", "1 min 30 s", "45 s" under a minute."""
    minutes, rest = divmod(seconds, 60)
    if not minutes:
        return f"{rest} s"
    return f"{minutes} min" + (f" {rest} s" if rest else "")


def _compile_time_said(cache: Path | None, recorded: bool, shapes: int) -> str:
    """The compilation's time for this job, counted from the `shapes` its warm-up compiles
    (COMPILE_SHAPE_NEW and the others, in whole seconds a shape, said to the second by
    _minutes_said: 8 shapes on a drive, 8 min 40 s): a resume's where the job is a resume whose
    recorded plan is kept as it is (`recorded`: same tiles, same shapes) and its compile cache is
    there and holds something (the cache is warm), a new job's otherwise,
    and the other case's after it; the Windows drive's figures where the cache's directory is on
    one (on_windows_drive), why they are longer at debug. A resume planned again (other tiles,
    other graphs) or a first start killed while compiling, whose plan was never recorded, finds
    a cache that doesn't hold its graphs: the new job's figure."""
    drive = cache is not None and on_windows_drive(cache)
    new, resume = (
        (COMPILE_SHAPE_NEW_DRIVE, COMPILE_SHAPE_RESUME_DRIVE)
        if drive
        else (COMPILE_SHAPE_NEW, COMPILE_SHAPE_RESUME)
    )
    new, resume = new * shapes, resume * shapes
    try:
        warm = recorded and cache is not None and cache.is_dir() and any(cache.iterdir())
    except OSError as error:
        logger.debug("%s not listed (%s): the compile cache taken as cold", cache, error)
        warm = False
    if drive:
        logger.debug(
            "the job's directory is on a Windows drive: the compile cache's many small files,"
            " written to and read back from it, make the compilation slower"
        )
    why = ", on a Windows drive" if drive else ""
    figures = (
        f"about {_minutes_said(resume)} for this resume, its cache warm, about"
        f" {_minutes_said(new)} for a new job"
        if warm
        else f"about {_minutes_said(new)} for this new job, about {_minutes_said(resume)} on a"
        " resume, its cache warm"
    )
    return (
        f"compiling its {_count(shapes, 'shape')} takes {figures}"
        f" (measured on an RTX 5090's host{why})"
    )


def said(
    plan: Plan,
    predicted: Peaks,
    card: str,
    target: tuple[int, int],
    shots: Sequence[Shot],
    pins: Pins,
    *,
    quality: str = "standard",
    recorded: bool = False,
    layouts: Sequence[Sequence[tuple[int, int]]] | None = None,
    disk: str | None = None,
    compile_cache: Path | None = None,
    smaller: str | None = None,
) -> list[str]:
    """The plan as --plan prints it and every run and resume logs it, before the first unit,
    a line per choice in plain words (DESIGN.md, Memory planner, Every choice of the plan is
    said), each opening with what it says, "plan: windows: ...": the value with its unit; why it
    was taken, the rule and its figure; what it gives or costs; the option that changes it.
    Every value the planner sets is in it: the card's free memory and the budget, with the
    margin, and on a run held to the card the torch's limit and the reserve, by their names and
    figures (_card_said, _budget_said); the padded frames and the attention backend; the shots;
    the windows; the --quality level, at every level (_quality_said); the swapped blocks, where
    the VAE is in the DiT's phase and what of the DiT stays in the VAE's; the system memory;
    each VAE pass, untiled or its tiles, their overlap and grid; the room a VAE pass is held to
    (VAE_ROOM); the scratch of the longest tiled shot; the disk the job takes (disk.py); the
    predicted peaks and which bounds the plan; with --compile, the VAE compiled, its passes held
    to COMPILE_PEAK times their plain peaks, and the compilation's time, paid before the DiT
    loads; the time, estimated once the models are loaded, by
    the calibration every run makes (DESIGN.md, Memory planner, Time estimates); what was
    pinned. recorded: the plan of a job resumed that has made units, whose window cap and tiles
    were kept from the job's record so that the output stays bit for bit (plan's `recorded`):
    their lines say so in place of the rule. A window the planner shortened for the host's
    memory, tiles it shrank for it, say what the card alone gives and why it isn't taken
    (Peaks.shortened, Peaks.shrunk). The lines stay short, one terminal line where they can, two
    at most; the manual's words.

    layouts: each shot's windows (latents [start, end)) where a job holds runs of its own window
    cap, made under a cap the plan no longer lays out (manifest.WINDOW_CAP; a shape planned again
    for the memory free, DESIGN.md, Memory planner, decided on 2026-10-28): the line's shots in
    one window and its joins are counted from them, not by the plan's cap, which they would
    otherwise misstate (review of the plan's lines, 2026-10-28). None for every other job, whose
    line is the plan's alone.

    disk: the line of the disk the job takes beside its output against what is free (
    disk.py, DESIGN.md, Memory planner, decided on 2026-10-28), after the scratch's and before the
    predicted peaks, as DESIGN.md lists it; None says none.

    compile_cache: the compile cache's directory (manifest.COMPILE_CACHE in the job's), whose
    state and disk the compilation's time depends on (_compile_time_said, which counts the
    plan's compile_shapes), warm only where
    `recorded` too; None says a new job on Linux's own disk.

    smaller: the line of the smaller DiT files (smaller_swaps) that plan the job better than this
    plan's, after the swapped blocks' line; None says none.

    The line of the compiled VAE is the plan's own (plan.compile_vae, the setting's, always: a
    job outside the shapes measured is refused, plan)."""
    from seedvr2x.shots import shot_layout

    job = _job(
        *(predicted.reading, target, shots, False, predicted.attention),
        plan.compile_vae,
        predicted.dit_model,
    )
    height, width = job.padded
    out_height, out_width = output_size(target)
    frames = Counter(shot.frames for shot in shots)
    if layouts is None:
        windows = {count: len(shot_layout(count, plan.window)) for count in frames}
        whole = sum(number for count, number in frames.items() if windows[count] == 1)
        joins = sum((windows[count] - 1) * number for count, number in frames.items())
    else:
        whole = sum(1 for layout in layouts if len(layout) == 1)
        joins = sum(len(layout) - 1 for layout in layouts)
    lines = [*_card_said(card, predicted)]
    lines.append(
        f"frames: {out_width}x{out_height} padded to {width}x{height} for the model,"
        f" {predicted.tokens:,} tokens a latent"
    )
    if predicted.attention == peaks.FLASH_ATTN_2:
        lines.append("attention: FlashAttention 2, which is installed")
    else:
        lines.append(
            "attention: PyTorch's SDPA, FlashAttention 2 not being installed: a window takes"
            " more memory under it"
        )
    if job.dit_model == peaks.INT8:
        lines.append(_int8_said())
    elif job.dit_model in (peaks.GGUF_7B, peaks.GGUF_7B_SHARP):
        lines.append(_gguf_said(job.dit_model))
    lines.append(
        f"shots: {len(shots)}, the longest of {_count(job.frames, 'frame')}"
        f" ({_count(job.lengths[-1], 'latent')})"
    )
    lines.append(_windows_said(job, plan, predicted, pins, recorded, whole, joins, len(shots)))
    lines.append(_quality_said(shots, target, quality))
    lines.append(_swap_said(plan, pins))
    if smaller is not None:
        lines.append(smaller)
    lines.append(_vae_said(plan))
    # The three lines below say what they are for first, in the manual's words, then their
    # figures (DESIGN.md, What it prints, decided on 2026-10-29: "each in the manual's words,
    # what it is for first, its figure after").
    lines.append(
        "the DiT in the VAE's passes: which model stays on the GPU in which phase, as much of the"
        f" DiT as fits beside each: in the encode,"
        f" {_resident_said(plan.resident_encode, plan.swapped)}; in the decode,"
        f" {_resident_said(plan.resident_decode, plan.swapped)}"
    )
    process, pinned, held, allowance = _host_hundredths(plan, predicted.shot, job.dit_model)
    left = _left(plan, "the DiT's other parameters", "the VAE")
    copies = f"{pinned / 100:.2f} GiB pinned for {left}" if left else "nothing pinned"
    in_all = process + pinned + held + allowance
    # The shot that holds the most: the longest, but where a shorter one in fewer windows does.
    if predicted.holder == job.lengths[-1]:
        whose = "the longest shot at the most"
    else:
        whose = f"a shot of {_count(predicted.holder, 'latent')} at the most, more than the longest"
    lines.append(
        f"system memory: {in_all / 100:.2f} GiB in all; {copies}; {held / 100:.2f} GiB for"
        f" {whose}, {_MOMENT_SHORT[predicted.moment]}"
    )
    for phase, tiles, pin in (
        ("encode", plan.encode_tiles, pins.encode_tile),
        ("decode", plan.decode_tiles, pins.decode_tile),
    ):
        lines.append(_vae_pass_said(job, phase, tiles, pin, recorded, predicted.shrunk))
    lines.append(
        "the VAE's room: what each VAE pass leaves free, for the large pieces of memory a pass"
        f" asks for at once: {_gib(VAE_ROOM)} GiB of the budget"
    )
    if predicted.scratch:
        # A scratch over the allowance goes whole to its file (scratch.Scratch).
        allowed = plan.scratch_allowance
        over = sum(
            number for count, number in frames.items() if peaks.scratch(job.padded, count) > allowed
        )
        where = (
            f"over it, the scratch of {over} of {_count(len(shots), 'shot')} goes whole to a"
            " memory-mapped file"
            if over
            else "which holds every shot's"
        )
        lines.append(
            "scratch: what a tiled pass keeps of the shot's frames between its tiles:"
            f" {_gib(predicted.scratch)} GiB for the longest tiled shot; in system memory up to an"
            f" allowance of {allowance / 100:.2f} GiB, {where}"
        )
    if disk is not None:
        lines.append(disk)
    names = {"encode": "encode", "dit": "DiT's phase", "decode": "decode"}
    lines.append(
        f"predicted peaks: encode {_gib(predicted.encode)} GiB, DiT {_gib(predicted.dit)} GiB,"
        f" decode {_gib(predicted.decode)} GiB; the {names[predicted.bound]} bounds the plan,"
        f"{'' if predicted.bound == 'dit' else ' its room counted,'}"
        f" {_gib(predicted.budget - predicted.need(predicted.bound))} GiB under the budget"
    )
    if plan.compile_vae:
        shapes = compile_shapes(plan, target, shots, QUALITY[quality])
        compiling = _compile_time_said(
            compile_cache, recorded, len(shapes.encode) + len(shapes.decode)
        )
        lines.append(
            f"compile: the VAE compiled (--compile), each pass held to {peaks.COMPILE_PEAK} times"
            f" its plain peak, in the shapes measured only; {compiling}, paid"
            " before the DiT loads, at every start and resume, its cache kept in the job's"
            " directory"
        )
    lines.append(
        "time: measured, not guessed: a calibration of the job's own steps once the models are"
        " loaded, its cost said then; it gives the job's time and each unit's time left and end;"
        " --until HH:MM stops before a unit that wouldn't finish by then"
    )
    options = (
        ("--window", pins.window),
        ("--swap-blocks", pins.swap_blocks),
        ("--encode-tile", pins.encode_tile),
        ("--decode-tile", pins.decode_tile),
    )
    given = [f"{option} {value}" for option, value in options if value is not None]
    lines.append(
        f"pinned: {', '.join(given)}, taken exactly"
        if given
        else "pinned: nothing; --window, --swap-blocks, --encode-tile and --decode-tile each fix"
        " a choice"
    )
    return [f"plan: {line}" for line in lines]


def _int8_said() -> str:
    """The plan's line of a DiT in int8 (said): its weights' bytes, by its header (peaks.INT8),
    and that a window's memory beside them is counted as fp16's plus the int8 path's buffers
    (peaks.INT8_BUFFERS), provisional until int8's own is measured (peaks.INT8_*). A DiT in fp16,
    the default, has no line of its own: weights.check_models' line names it."""
    return (
        f"DiT: its weights in int8, {_gib(peaks.weights_bytes(peaks.INT8))} GiB, half of fp16's;"
        f" a window's memory counted as fp16's plus {_gib(peaks.INT8_BUFFERS)} GiB for int8's"
        " buffers until int8's own is measured; --dit-model names the file"
    )


def _gguf_said(dit_model: Architecture) -> str:
    """The plan's line of a DiT in a dynamic GGUF file (said), in the words of the picture's
    (still_plan.said): its weights' bytes by its header, and that a window's memory beside them
    is fp16's plus the dequantisation's buffers (peaks.GGUF_BUFFERS, measured on the RTX 5090)."""
    return (
        f"DiT: its weights in seedvr2x's dynamic GGUF, {_gib(peaks.weights_bytes(dit_model))} GiB,"
        " under a third of fp16's; a window's memory counted as fp16's plus"
        f" {_gib(peaks.GGUF_BUFFERS)} GiB for the dequantisation's buffers;"
        " --dit-model names the file"
    )


# What the plan's lines say of a value a resume takes from the job's record (said).
_KEPT = "kept from the job's record, so that the output stays bit for bit"


def _windows_said(
    job: _Job,
    plan: Plan,
    peaks: Peaks,
    pins: Pins,
    recorded: bool,
    whole: int,
    joins: int,
    shots: int,
) -> str:
    """The plan's line of the windows (said): the cap, in latents and frames, and the longest
    window run where the cap doesn't say it, a cap pinned or recorded above it; why: pinned,
    kept from the record, shortened for the host's memory (Peaks.shortened), the longest shot
    whole, or the longest window the budget holds (DESIGN.md, Memory planner, Choices, 1); the
    shots in one window and the joins; what a longer window gives; --window."""
    if plan.window is None:
        cap = "no cap"
    else:
        cap = f"{_count(plan.window, 'latent')} at most ({_count(_frames(plan.window), 'frame')})"
    # The longest window run, where the cap doesn't say it: a cap pinned or recorded above it.
    if peaks.window != plan.window:
        cap += f", the longest run {_count(peaks.window, 'latent')}"
    if pins.window is not None:
        why = "pinned by --window"
    elif recorded:
        why = _KEPT
    elif peaks.shortened is not None:
        short = peaks.shortened
        why = (
            f"shortened from {_count(short.window, 'latent')} for the host's memory (their plan"
            f" {short.taken / 100:.2f} GiB of {short.available / 100:.2f} available)"
        )
    elif plan.window is not None and plan.window >= job.lengths[-1]:
        why = "the longest shot whole, which the budget holds"
    else:
        why = "the longest the budget holds"
    gives = "; a longer window, fewer joins" if joins else ""
    return (
        f"windows: {cap}, {why}: {whole} of {_count(shots, 'shot')} in one window,"
        f" {_count(joins, 'join')}{gives}; --window N caps them"
    )


def _frames(latents: int) -> int:
    """The frames `latents` latents hold: 1 in the first, 4 in each next (DESIGN.md, Terms)."""
    return 4 * latents - 3


def _swap_said(plan: Plan, pins: Pins) -> str:
    """The plan's line of the swapped blocks (said): how many; why: pinned, none needed, the VAE
    kept on the GPU for the host's memory (_vae_kept), or the fewest that hold the window
    (DESIGN.md, Memory planner, Choices, 2); what it costs; --swap-blocks."""
    if pins.swap_blocks is not None:
        why = "pinned by --swap-blocks"
    elif not plan.swapped:
        why = "the window fitting with the DiT whole on the GPU"
    elif plan.vae_in_dit:
        why = "more than the window needs: the VAE stays on the GPU, which spares system memory"
    else:
        why = "the fewest that hold the window"
    if not plan.swapped:
        return f"blocks swapped: none, {why}; --swap-blocks N swaps some"
    return (
        f"blocks swapped: {plan.swapped} of the DiT's {peaks.BLOCKS}, {why}: slower, not worse;"
        " --swap-blocks N"
    )


def _vae_said(plan: Plan) -> str:
    """The plan's line of the VAE's weights in the DiT's phase (said): off the GPU, given to the
    window, before one more block is swapped; or on it, beside the window."""
    if plan.vae_in_dit:
        return "the VAE: on the GPU in the DiT's phase, the window fitting beside it"
    return (
        f"the VAE: off the GPU in the DiT's phase, its {_gib(peaks.VAE_WEIGHTS)} GiB given to the"
        " window"
    )


def _vae_pass_said(
    job: _Job,
    phase: str,
    tiles: Tiles | None,
    pin: int | None,
    recorded: bool,
    shrunk: Shrunk | None,
) -> str:
    """The plan's line of a VAE pass, the encode or the decode (said): untiled, or its tiles,
    their overlap and grid; why: pinned, kept from the record, shrunk for the host's memory
    (Peaks.shrunk), the whole frame fitting, or the largest tiles that fit (DESIGN.md, Memory
    planner, Choices, 3); what tiles cost; --encode-tile or --decode-tile."""
    option = f"--{phase}-tile"
    alone = None if shrunk is None else (shrunk.encode if phase == "encode" else shrunk.decode)
    if pin is not None:
        why = f"pinned by {option}"
    elif recorded:
        why = _KEPT
    elif shrunk is not None and alone != tiles:
        why = (
            f"shrunk from {'untiled' if alone is None else f'{alone.size} px'} for the host's"
            f" memory (that plan {shrunk.taken / 100:.2f} GiB of {shrunk.available / 100:.2f}"
            " available)"
        )
    elif tiles is None:
        why = "the whole frame fitting the budget"
    else:
        why = (
            "the whole frame not fitting the budget, the largest that do, up to"
            f" {peaks.TILE_MAX} px"
        )
    if tiles is None:
        return f"{phase}: untiled, {why}; {option} PX tiles it"
    rows, columns = peaks.grid_shapes(job.padded, tiles)
    return (
        f"{phase}: tiles of {tiles.size} px overlapping by {tiles.overlap}, {columns} across,"
        f" {rows} down, {why}: a slight colour drift, which split corrects; {option} PX"
    )


def beyond_measured(plan: Plan) -> list[str]:
    """What the log warns of once the plan is made, a line per pass, the encode's then the
    decode's (main._run): a pass in tiles larger than TILE_MAX, the largest measured, which only a
    pin gives it (_pass), or the record of a job made with one. Its predicted peak is the fit's
    alone there, and the one such pass run, a 4K encode in tiles of 2304 px, took more than its
    fit (TILE_MAX). None for a plan of the planner's own choices."""
    passes = (("encode", plan.encode_tiles), ("decode", plan.decode_tiles))
    return [
        f"the {phase} in tiles of {tiles.size} px: its predicted peak is the fit's, beyond the"
        f" sizes measured (up to {peaks.TILE_MAX} px), and the pass may take more"
        for phase, tiles in passes
        if tiles is not None and tiles.size > peaks.TILE_MAX
    ]


# The attention backends as the plan's lines name them (model.load_models' own words for SDPA).
_ATTENTION_SAID = {peaks.FLASH_ATTN_2: "FlashAttention 2", peaks.SDPA: "PyTorch's SDPA"}


def card_detail(peaks: Peaks) -> str | None:
    """The driver's figure the plan's card line no longer carries (design's ruling, baton 3,
    2026-10-30: under --verbose), as the debug record the caller logs right after that line
    (cli, "plan: " in front, as said gives the lines): on a held run whose whole-card figure,
    NVML's, is the smaller, the one the line names, what the driver reports free to this process
    and that the run isn't held to it. None elsewhere: where the driver's figure is the free
    memory the line gives, the card's not read, or an early reading (memory.early), which asked
    the driver nothing."""
    reading = peaks.reading
    if not reading.held or reading.nvml is not None:
        return None
    if reading.card is None or reading.card > reading.reported:
        return None
    return (
        f"plan: card: the GPU's driver reports {_gib(reading.reported)} GiB free to this process,"
        f" which the run isn't held to: the whole card has {_gib(reading.card)} GiB free by NVML"
    )


def _card_said(card: str, peaks: Peaks) -> list[str]:
    """The plan's first lines (said): the card, its free memory and the budget, with the margin
    and the reserve by their names and figures. On a driver that refuses, one line: what it
    reports free, the budget that less the margin (MARGIN), and no reserve. On a run held to the
    card (memory.Reading.held), two: the free memory the run is held under, the driver's figure
    or the whole card's, with the other (_named); then the budget, what the plan counts on: that
    less what torch's limit leaves under it (Reading.left) and the reserve the plan leaves
    torch's allocator beyond a step's peak (memory.RESERVE), two figures that were once given as
    one (a review's finding, 2026-10-12), the margin counted in the reserve (_Job.margin); where
    the plan took memory.RESERVE_TIGHT, the plan with memory.RESERVE refused, that refusal and
    the cost, slow, in a second line (Peaks.tightened)."""
    reading = peaks.reading
    # An early reading (memory.early) says what NVML read and what was counted off it, never
    # "its driver reports", which it did not ask.
    early = None if reading.nvml is None else _early_said(reading, _gib)
    if not reading.held:
        free_said = (
            f"{_gib(peaks.free)} GiB free"
            if early is None
            else f"{early}, {_gib(peaks.free)} GiB left"
        )
        return [
            f"card: {card}, {free_said}, a budget of {_gib(peaks.budget)} GiB:"
            f" the margin of {_gib(MARGIN)} GiB left to the desktop and other programs; no"
            " reserve, the driver refusing memory the card doesn't have"
        ]
    free = _held_under(reading)
    budget = (
        f"budget: {_gib(peaks.budget)} GiB, the {_gib(free)} GiB free less torch's limit,"
        f" {_gib(reading.left)} GiB under it, and the reserve of"
        f" {_gib(free - reading.left - peaks.free)} GiB left to torch's allocator beyond a step's"
        f" peak; the margin of {_gib(MARGIN)} GiB is in the reserve"
    )
    named = _named(reading, _gib, driver=False) if early is None else f"({early})"
    lines = [
        f"card: {card}, {_gib(free)} GiB free {named}: the run is held to the card's own memory,"
        " its driver giving more than the card has",
        budget,
    ]
    if peaks.tightened is not None:
        # The plan's last resort (plan; DESIGN.md, Allocator): said slow, on a line of its own,
        # the budget's second (DESIGN.md, Architecture, What it prints, decided on 2026-10-28:
        # the budget's line took 851 characters with the usual reserve's refusal inside it, which
        # plan() now says at debug). PROVISIONAL (impl-v010, 2026-10-29: DESIGN.md says "the two
        # lines a plan's line may take" and leaves the split open): the figures in the first, the
        # reserve of 2.5, its cost and the pin that runs at full speed in the second, so that the
        # first reads as any held plan's does.
        tight, usual = _gib(memory.RESERVE_TIGHT), _gib(memory.RESERVE)
        reserve = (
            f"reserve: {tight} GiB and not {usual}, the plan with {usual} having been refused"
            " (--verbose says why); slow: a step short of room beyond its peak slows down where"
            f" it would be refused, and a job that fits with {usual} runs up to 1.5 times as"
            f" long with {tight}; memory freed on the card gives the {usual} back"
        )
        if peaks.instead is not None:
            # The pin that runs with the usual reserve, at full speed (design, 2026-10-21).
            reserve += f"; {option(peaks.instead)} would run with the full reserve, at full speed"
        lines.append(reserve)
    return lines

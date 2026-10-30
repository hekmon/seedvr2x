"""The calibration of a job's time estimates (DESIGN.md, Memory planner, Time estimates): each kind
of unit run on frames of its own making, at the job's size and tiles, through the job's own
passes, before the job's first unit: every run makes it, --plan included (decided on 2026-10-21),
so that --plan says the job's time, each unit's line its time left from the first unit, and
--until holds each unit's against its time.

The steps are the run's own, taken through runtime/run.py's names (run.encode_shot,
run.sample_windows, run.decode_shot), as the run takes them."""

import logging
import statistics
import tempfile
import time
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch

from seedvr2x.clock import wall
from seedvr2x.media.checksums import write_checksums
from seedvr2x.media.decode import to_float32
from seedvr2x.media.writer import FFV1Writer, Tags, Writer
from seedvr2x.runtime import planner, run
from seedvr2x.runtime.job import padded_size
from seedvr2x.runtime.model import Models, synchronize
from seedvr2x.runtime.plan import Phase, Plan
from seedvr2x.runtime.shot import Correction, draw, finite
from seedvr2x.runtime.timing import ENTER, Clock, Kind, Model, tokens

logger = logging.getLogger(__name__)

# What a decode's frames are written by: (n, H', W', 3) float32 in [0, 1] (shot.decode_shot).
Write = Callable[[npt.NDArray[np.float32]], None]

# What a calibration encodes and decodes, in frames (DESIGN.md, Memory planner, Time estimates:
# "5 and 13 frames encoded and decoded"): three of the VAE's slices each way, of 5, 4 and 4
# frames; and the first slice alone, FIRST frames, so that two lengths tell a kind's constant
# from its rate. The encode's second length is FRAMES or more, about SECONDS of it where it is
# quick (_second_length; decided on 2026-10-22).
FRAMES = 13
FIRST = 5

# What the encode's second length takes, about, in seconds at the rate its first length was
# measured at (decided on 2026-10-22): two encodes of 5 and 13 frames, each under a second and a
# half at 1080p on an RTX 5090, put encodes 11.8% under and 15.3% over in two runs of one job.
SECONDS = 3.0

# Its short window, in latents (DESIGN.md: "windows of 2 latents and of the job's longest").
SHORT = 2

# The rate its input copies are written at, which is no part of their frames.
RATE = Fraction(25)


@dataclass(frozen=True)
class Calibrated:
    """A calibration's clock (calibrate), and what the calibration took, in seconds."""

    clock: Clock
    seconds: float


@torch.no_grad()
def calibrate(
    models: Models,
    target: tuple[int, int],
    plan: Plan,
    longest: int,
    *,
    most: int | None = None,
    source: tuple[int, int] | None = None,
    frames: Callable[[int], npt.NDArray[np.uint16]] | None = None,
    split: bool = False,
    writer: Callable[[Path], Writer] | None = None,
    directory: Path | None = None,
    numz_padding: bool = False,
    numz_draw: bool = False,
    numz_tail: bool = False,
) -> Calibrated:
    """A clock for a job whose frames are resized to target, (height, width), run by plan, whose
    longest window is `longest` latents (shot.shot_layout): the models of its three kinds of
    unit, measured on the job's own steps at the job's size and tiles, on frames of its own
    making (DESIGN.md, Memory planner, Time estimates).

    The steps as the run makes them (run.run_job): the source's frames, frames(n) giving its next n,
    (n, H, W, 3) uint16 as the job's reader gives them, read for each encode and timed with it, the
    reader's start paid by the warm-up, through the reader's conversion to float (DESIGN.md:
    "through the job's own passes (the reader, ...)"); without `frames`, black frames of the
    source's size, `source`, (height, width) as stored, the target's without it, for the tests; with
    `split`, the encode's frames copied and the decode corrected against the copy (DESIGN.md,
    Colour correction); the decode's frames written by `writer`, which opens the job's own kind
    of output at a path, dropped without it; the plan's tiles, their scratch where the run's
    goes, in system memory up to the plan's allowance. A decode alone, uncorrected and
    unwritten, came out 9% under a job's decodes at 432p, and that job's first decode took 1.5
    times its estimate (RTX 5090, 2026-10-10). The copies, the scratch and the output go to a
    temporary directory in `directory`, the disk the job's own go to (the system's temporary
    directory without it), removed.

    Each kind at two lengths, which tell its constant from its rate (_fit): FIRST frames then
    about SECONDS' worth encoded (_second_length), at least FRAMES and no more than `most`, the
    job's longest run in frames, which its plan was made for; FIRST then FRAMES decoded; a
    window of SHORT latents and one of `longest`. One length spreads the constant over its every
    frame or token: a window's is a third of a 2-latent window at 540p (0.2 s of 0.6), which
    would put 30% on a 16-latent one, and an encode's is a frame and a half's time at 432p
    (0.05 s; RTX 5090, 2026-10-10), 13% on a long shot's.

    Each kind's steps are taken together, the first of them once at the kind's largest shapes, a
    warm-up not measured (an encode longer than its warm-up's 13 frames runs more slices of the
    same shapes, 4 frames each): a process's first encode takes seconds more than its next ones, and
    its first window too (3.4 s and 1.5 s at 540p on that card), which no unit of the run pays
    again, and a first step at a size allocates for it. Each kind is measured as most of its units
    come in a run (decided on 2026-10-22): a window or a decode after a step of its own kind, the
    first unit after a change of phase paying the change's term besides (timing.units_of; measured
    each after a change, at 720p on an RTX 5090, a decode came out at 0.247 s a frame, where the
    job's own took 0.205, and --plan said 18% over); an encode right after a window, as a run's
    encodes always follow a change of phase, that change timed with it, in its model (timing.ENTER;
    measured after an encode, its constant came out 0.10 s for 0.23 to 0.66 s, and a job's short
    encodes ran 0.5 to 0.7 s over). The terms a unit pays besides its kind's model are measured
    apart (DESIGN.md, Memory planner, Time estimates, decided on 2026-10-21): each change into the
    DiT's phase or the decode's (Models.enter), a constant per phase, the median of those seen,
    three into the DiT's and two into the decode's, one made alone after the steps; the noise's
    draw (shot.draw), a rate per token, timed alone on the longest window's latent, and taken out of
    the windows' times. The models are left in the decode's phase. numz_padding, numz_draw and
    numz_tail are for tests only, as run.run_job takes them."""
    started = time.monotonic()
    height, width = target if source is None else source
    if frames is None:

        def blank(count: int) -> npt.NDArray[np.uint16]:
            return np.zeros((count, height, width, 3), np.uint16)

        frames = blank
    read_source = frames
    per_latent = tokens(target, 1, numz_padding=numz_padding)

    def fits(count: int) -> bool:
        """Whether a tiled encode's scratch of `count` frames stays in the plan's system memory,
        as the run's does up to its allowance (tiling.Scratch); an untiled encode keeps nothing
        of its frames."""
        if plan.encode_tiles is None:
            return True
        padded = padded_size(target)
        if numz_padding:
            padded = (-(-target[0] // 16) * 16, -(-target[1] // 16) * 16)
        return planner.scratch(padded, count) <= plan.scratch_allowance

    with tempfile.TemporaryDirectory(prefix=".seedvr2x-calibration-", dir=directory) as made:
        scratch = Path(made)

        # Each change of phase timed on its own (Models.enter), by the phase entered: the run's
        # units pay it apart from their kind's model (timing.units_of).
        entered: dict[Phase, list[float]] = {phase: [] for phase in ENTER}
        left: list[Phase | None] = [None]  # the phase the models are in
        lengths: list[int] = []  # the encode's second length, for the log

        def enter(phase: Phase, counted: bool = True) -> None:
            begun = time.monotonic()
            models.enter(phase)
            if left[0] != phase and phase in ENTER:
                # Its copies waited for, as the run's _enter waits: the term holds the transfer.
                synchronize(models.device)
                if counted:
                    entered[phase].append(time.monotonic() - begun)
            left[0] = phase

        def encode(shot: int, count: int) -> tuple[torch.Tensor, float, float]:
            """The latent of `count` frames of the source, in its own scratch directory; the
            encode's seconds, its change of phase timed with it, as the run's encode times its own
            (timing.ENTER); and the seconds of its steps alone, after the change."""
            begun = time.monotonic()
            enter("encode")
            stepped = time.monotonic()
            read = _reader(read_source)
            where = scratch / f"shot_{shot}"
            where.mkdir()

            def encoded(read: Callable[[int], npt.NDArray[np.float32]]) -> torch.Tensor:
                return run.encode_shot(
                    *(models, read, count, target, 0),
                    numz_padding=numz_padding,
                    numz_tail=numz_tail,
                    tiles=plan.encode_tiles,
                    # The run's: a tiled pass's alone (run._scratch).
                    scratch=None if plan.encode_tiles is None else where,
                    allowance=plan.scratch_allowance,
                )

            if split:
                # The copy's rate is no part of its frames: any will do.
                with _copied(where, width, height) as copy:

                    def read_and_copy(count: int) -> npt.NDArray[np.float32]:
                        frames = read(count)
                        copy.write(frames)
                        return frames

                    latent = encoded(read_and_copy)
                write_checksums(where / "input.crc32", copy.checksums)
            else:
                latent = encoded(read)
            finite(latent, "the calibration's encode")
            ended = time.monotonic()
            return latent, ended - begun, ended - stepped

        def window(latent: torch.Tensor) -> float:
            enter("dit")
            begun = time.monotonic()
            layout = [(0, latent.shape[0])]
            output = next(run.sample_windows(models, latent, layout, 0, numz_draw=numz_draw))
            finite(output, "the calibration's window")
            del output
            return time.monotonic() - begun

        def decode(shot: int, latent: torch.Tensor, count: int, write: Write) -> float:
            enter("decode")
            begun = time.monotonic()
            where = scratch / f"shot_{shot}"
            correction = None
            if split:
                correction = Correction(where / "input.mkv", where / "input.crc32", width, height)
            run.decode_shot(
                *(models, latent, count, target, write, "the calibration", 0, correction),
                tiles=plan.decode_tiles,
                scratch=None if plan.decode_tiles is None else where,
                allowance=plan.scratch_allowance,
            )
            return time.monotonic() - begun

        def measure(write: Write) -> dict[Kind, Model]:
            # The warm-up, at the largest shapes, so that no measured step is the first at its
            # size: on an RTX 5090 at 1080p (2026-10-21), a job's windows came out 10-15% under
            # the estimate of a calibration warmed up on its 2-latent window alone. Each kind's
            # warm-up right before its measured steps.
            warm, _, _ = encode(0, FRAMES)
            short = min(SHORT, longest)
            between = warm[:short].contiguous()
            # Each encode measured after a window, its change of phase timed with it, as most of
            # the run's encodes come: after a shot's windows, a segment's first shot's after the
            # decode before it (decided on 2026-10-22; on the cut lists of 12 sources, 61% of the
            # encodes after a window, 36 to 68% by source, a cartoon's the fewest, segments merging
            # shots up to 5 s). A window of SHORT latents, the cheapest. The second length by the
            # first's own rate, its change left out (_second_length).
            window(between)
            first, encoded_first, stepped_first = encode(1, FIRST)
            second = _second_length(stepped_first / FIRST, most, fits)
            lengths.append(second)
            window(between)
            _, encoded, _ = encode(2, second)
            # The windows and the decodes each after a step of their own kind, as most of the
            # run's are, the first of a kind after a change of phase paying its term besides
            # (timing.units_of; decided on 2026-10-22: measured each after a change of phase, at
            # 720p on an RTX 5090, a decode came out at 0.247 s a frame, where the job's own took
            # 0.205, and --plan said 18% over).
            repeated = torch.cat([warm] * -(-longest // warm.shape[0]))
            window(repeated[:longest].contiguous())
            # The noise's draw, timed alone on the longest window's latent: a run's first window
            # draws its whole run's (shot.sample_windows), about 1 us a token on a desktop's CPU
            # (shot._drawn's figure), which a window of the calibration pays for its own tokens.
            begun = time.monotonic()
            draw(models, repeated[:longest].contiguous(), 0, numz_draw)
            drawing = (time.monotonic() - begun) / (longest * per_latent)
            sizes = [short]
            timed = [window(repeated[:short].contiguous())]
            if longest != short:
                sizes.append(longest)
                timed.append(window(repeated[:longest].contiguous()))
            # The warm-up's shot decoded twice: its 13 frames are the decode's largest.
            decode(0, warm, FRAMES, write)
            decoded_first = decode(1, first, FIRST, write)
            decoded = decode(0, warm, FRAMES, write)
            # The DiT's phase entered three times by the steps, each from the encode's, as a run's
            # windows enter it but a resumed run's first; the decode's once: entered again from the
            # DiT's, alone, the change into the DiT's phase from the decode's not counted. The
            # models left in the decode's phase, where the job's units start.
            enter("dit", counted=False)
            enter("decode")
            # A window less its own draw, which the units count apart.
            windows = [
                (size * per_latent, seconds - drawing * size * per_latent)
                for size, seconds in zip(sizes, timed, strict=True)
            ]
            fitted: dict[Kind, Model] = {
                "encode": _fit([(FIRST, encoded_first), (second, encoded)], "the encode"),
                "window": _fit(windows, "the window"),
                "decode": _fit([(FIRST, decoded_first), (FRAMES, decoded)], "the decode"),
                "draw": Model(0.0, drawing),
            }
            for phase, seconds in entered.items():
                fitted[ENTER[phase]] = Model(statistics.median(seconds) if seconds else 0.0, 0.0)
            return fitted

        if writer is None:
            fitted = measure(_dropped)
        else:
            # One output for the calibration's decodes, as a segment's shots share theirs.
            with writer(scratch / "output") as output:
                fitted = measure(output.write)
    seconds = time.monotonic() - started
    said: tuple[tuple[str, Kind, int, str], ...] = (
        ("an encode, its change of phase in its constant,", "encode", 1, "s a frame"),
        ("a window", "window", 1000, "ms a token"),
        ("a decode", "decode", 1, "s a frame"),
    )
    logger.info(
        "time estimates calibrated in %.1f s, encodes of %d and %d frames: %s; entering the DiT's"
        " phase and the decode %.2f and %.2f s; the noise's draw %.2f us a token",
        wall(seconds),
        FIRST,
        lengths[0],
        "; ".join(
            f"{name} {fitted[kind].constant:.2f} s and {fitted[kind].rate * scale:.3f} {unit}"
            for name, kind, scale, unit in said
        ),
        *(fitted[ENTER[phase]].constant for phase in ("dit", "decode")),
        fitted["draw"].rate * 1e6,
    )
    return Calibrated(Clock(fitted), seconds)


def _dropped(frames: npt.NDArray[np.float32]) -> None:
    """A write that keeps nothing: a calibration given no writer."""


def _reader(
    source: Callable[[int], npt.NDArray[np.uint16]],
) -> Callable[[int], npt.NDArray[np.float32]]:
    """read(n), giving the source's next n frames as the encode asks for them, as run.Inputs gives
    a shot's: through the reader's conversion to float (decode.to_float32), never more of them
    held than the encode's slice (a review's finding: an encode's whole length read first held
    12.4 MB a frame at 1080p beyond what the plan counts, the second length up to 3 s of them)."""

    def read(count: int) -> npt.NDArray[np.float32]:
        return to_float32(source(count))

    return read


@contextmanager
def _copied(where: Path, width: int, height: int) -> Generator[FFV1Writer]:
    """The writer of a calibration's input copy in `where`, as run._copying writes a shot's."""
    with FFV1Writer(where / "input.mkv", "gbrp16le", width, height, RATE, Tags()) as copy:
        yield copy


def _fit(measures: Sequence[tuple[int, float]], kind: str = "a kind") -> Model:
    """The model through a calibration's measures of one kind, each a size and its seconds, the
    longest last: the line through the shortest and the longest. One size alone gives a rate
    alone; so do two whose line would start under zero or run flat, the times' own scatter
    showing in place of a constant, the longest then taken alone."""
    size, seconds = measures[-1]
    if len(measures) > 1:
        short, quick = measures[0]
        rate = (seconds - quick) / (size - short)
        constant = quick - rate * short
        if rate > 0 and constant >= 0:
            return Model(constant, rate)
        logger.info(
            "time estimates: %s's two lengths, %s, give no line with a constant of 0 or more and"
            " a rate above 0: its longest alone taken, a rate without a constant",
            kind,
            ", ".join(f"{size} in {seconds:.3f} s" for size, seconds in measures),
        )
    return Model(0.0, seconds / size)


def _second_length(per_frame: float, most: int | None, fits: Callable[[int], bool]) -> int:
    """The encode's second length, in frames (calibrate): about SECONDS at per_frame, its first
    length's seconds a frame, on the VAE's grid of 4n + 1 frames (shot.padded_length); no more than
    `most`, the job's longest run, nor than what `fits`, a tiled encode's scratch in the plan's
    memory; never fewer than FRAMES (decided on 2026-10-22: at 4K, an encode's 13 frames take
    more than SECONDS)."""
    wanted = FRAMES if per_frame <= 0 else SECONDS / per_frame
    count = 4 * max(round((wanted - 1) / 4), 0) + 1
    if most is not None:
        count = min(count, 4 * ((max(most, 1) - 1) // 4) + 1)
    while count > FRAMES and not fits(count):
        count -= 4
    return max(count, FRAMES)

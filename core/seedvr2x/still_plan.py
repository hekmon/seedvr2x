"""The plan for one picture (DESIGN.md, Still images, "The plan for one picture"; Repository and
apps, step 3): where the models are and how the VAE's passes are tiled, made once for a bound, the
output's long and short sides in either orientation, and holding for every picture within it. The
CLI's bound is its picture's output, the image server's its largest.

An addition beside the CLI's planner (apps/seedvr2x/planner.py), not a move: it imports nothing of
the CLI, and no torch, the command line planning before torch is imported
(tests/test_still_plan.py). Its margins and its order of choices are the planner's (DESIGN.md,
Memory planner, Choices), and the comments name the planner's function each choice mirrors. What
differs is what a plan is for: a job's plan is made for the one frame size it runs, so that its
tiles are chosen at it; a bound's plan holds for every size within it, and "the largest picture's
tiles don't hold for every smaller one" (DESIGN.md: a cut-off tile is counted at its rule plus 0.3
GiB, and measured steps aren't monotone). So the placement of the DiT and the VAE
(peaks.ModelPlan) is fixed for the bound, once, by a proof that every picture within it has a pass
that fits (_worst); the tiles are chosen per picture under that placement (tiles), never refused
within the bound. Sizes are (height, width) in pixels, and a bound (long, short), unless said;
`padded` is the frame the model takes (frames.padded_size).

A picture of one frame is one latent in the DiT, one frame in the VAE's passes: its passes take
one frame's figures (peaks.picture_encode_peak, peaks.picture_decode_peak), measured on the RTX
5090 against the video slice's, of which they are 0.18 to 0.34 (DESIGN.md, Still images,
decided on 2026-11-03), provisional until the card's batch.

The plan also counts the picture's own copies on the GPU, at its stored size, which the output's
size doesn't bound: the resize that still.upscale makes on the device (COPIES, input_copies), by
the rule measured on the RTX 5090.

What a phase takes for the image server's reservation, its predicted peak with the room it is
held to and the share torch's allocator reserves over it (Peak), is said at the bound
(bound_peaks) and for one picture within it (picture_peaks)."""

import math
import mmap
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from functools import cache

from seedvr2x import memory, peaks, tilegrid
from seedvr2x.errors import Refusal
from seedvr2x.frames import MULTIPLE, output_size, padded_size, padding
from seedvr2x.memory import Reading
from seedvr2x.passes import ModelPlan, Phase, Tiles
from seedvr2x.peaks import MARGIN, VAE_ROOM
from seedvr2x.weights import Architecture

# The output's long and short sides, in pixels: the bound a plan is made for.
Bound = tuple[int, int]

# A picture's pixels, one frame (peaks.decode_peak's `frames`): the colour correction's term is one
# frame's, not a video slice's.
FRAMES = 1

PHASES: tuple[Phase, ...] = ("encode", "dit", "decode")


class PlanRefusal(Refusal):
    """A bound the plan can't take: a pin it can't take exactly, or a size that doesn't fit the
    memory free, or the system memory. The message counts in free VRAM, never in card sizes, and
    names what is there for what it is (_there), as the planner's refusals do (PlanError).

    `phase`: the part that doesn't fit, "encode", "dit" or "decode" for the GPU's memory, "host"
    for the system's, "pin" for a pin that is no pin (swap count above the DiT's blocks, a tile
    covering the whole padded frame, a tile that fits no picture within the bound); a swap pin
    that doesn't hold the DiT's phase is the "dit" phase's. `fits`: for a floor refusal, the
    largest bound (long, short) that fits the memory, searched as the planner searches the
    largest output (_too_large); None elsewhere, and where no output size fits. `fits_input`:
    for the refusal of a picture's copies (_input_refused, phase "encode" or "decode"), the
    largest input in pixels that fits; None elsewhere, and where no input does (the output's own
    copies are over the memory)."""

    def __init__(
        self,
        message: str,
        phase: str = "",
        fits: Bound | None = None,
        fits_input: int | None = None,
    ) -> None:
        super().__init__(message)
        self.phase = phase
        self.fits = fits
        self.fits_input = fits_input


@dataclass(frozen=True)
class Pins:
    """A user's pins, as the CLI's planner.Pins have them (the picture has no window to pin):
    each taken exactly or refused, None leaving a choice to the plan."""

    encode_tile: int | None = None  # --encode-tile: a tile's side in output pixels; 0: untiled
    decode_tile: int | None = None  # --decode-tile, likewise
    swap_blocks: int | None = None  # --swap-blocks: the DiT's blocks swapped


NO_PINS = Pins()


def _margin(reading: Reading) -> int:
    """What a plan leaves of the memory it may count on, in bytes: MARGIN on a driver that
    refuses; nothing on a run held to the card, where memory.RESERVE, taken off already, measures
    the same need (the planner's _Job.margin; DESIGN.md, Allocator)."""
    return 0 if reading.held else MARGIN


@dataclass(frozen=True)
class StillPlan:
    """The plan for one picture, made for a bound (plan): everything that holds for every picture
    within it. The tiles are the picture's own (tiles)."""

    bound: Bound  # (long, short): the output's sides, either orientation
    input_pixels: int  # the largest input, H x W as stored: its copies on the GPU are counted
    models: ModelPlan  # swapped, resident_encode, resident_decode, vae_in_dit; compile_vae False
    reading: Reading  # as made: tightened where the tight retry took it
    attention: str  # peaks.DIT_ATTENTION's
    dit_model: Architecture  # peaks.DITS'
    split: bool  # whether the decode corrects colours: its term is counted in the decode's phase
    pins: Pins
    process: int  # bytes: the process's own resident system memory, the caller's figure
    dit_peak: int  # bytes: the DiT's phase, the most over the shapes within the bound
    windows: int  # the most attention windows any picture within the bound takes (windows_most)
    pinned: int  # bytes: the pinned copies of what leaves the GPU in some phase (_pinned)
    host: int  # bytes: process + pinned + the tiled passes' scratch, at the bound
    host_available: (
        int | None
    )  # bytes of system memory available as the plan was made; None: unread
    tightened: str = ""  # the refusal with the usual reserve, where the tight reserve was taken

    @property
    def margin(self) -> int:
        """What the plan leaves of the memory it counts on, in bytes (_margin)."""
        return _margin(self.reading)

    @property
    def budget(self) -> int:
        """What the plan counts its peaks in, in bytes: the reading's `counted` less its margin."""
        return self.reading.counted - self.margin


@dataclass(frozen=True)
class PictureTiles:
    """A picture's VAE passes (tiles): None is untiled; the peaks are predicted, in bytes, under
    the plan's placement, without the room every VAE pass is held to (VAE_ROOM)."""

    encode: Tiles | None
    decode: Tiles | None
    encode_peak: int
    decode_peak: int


@dataclass(frozen=True)
class _Job:
    """What the plan is made by: the bound and the memory it may count on (the reading, whose
    `counted` the planner's _Job calls `free`), the DiT, and the host."""

    bound: Bound
    input_pixels: int
    reading: Reading
    attention: str
    dit_model: Architecture
    split: bool
    pins: Pins
    process: int
    host_available: int | None

    @property
    def free(self) -> int:
        return self.reading.counted

    @property
    def margin(self) -> int:
        return _margin(self.reading)

    @property
    def budget(self) -> int:
        return self.free - self.margin

    def pin(self, phase: Phase) -> int | None:
        return self.pins.encode_tile if phase == "encode" else self.pins.decode_tile


def plan(
    bound: Bound,
    free: int | Reading,
    host_available: int | None,
    *,
    process: int,
    input_pixels: int,
    attention: str = peaks.FLASH_ATTN_2,
    dit_model: Architecture = peaks.FP16,
    split: bool = True,
    pins: Pins = NO_PINS,
) -> StillPlan:
    """The plan for every picture whose output is within `bound`, (long, short) in pixels, each
    side even (frames.output_size) and either orientation, in the GPU memory it may count on
    (`free`: bytes, what a driver that refuses reports free; or the reading the run made,
    memory.Reading, whose `counted` is the memory the plan counts on, as the planner's plan has
    it), with `host_available` bytes of system memory (None: unchecked) and the process's own
    `process` bytes of it. PlanRefusal when a pin can't be taken exactly, when no plan fits, or
    when the plan takes more system memory than there is.

    `input_pixels` is the largest input, in pixels (height x width as stored, before the resize):
    the CLI's picture, the server's largest. The picture goes to the GPU at that size and is
    resized there, in the encode's phase and, with the colour correction, in the decode's, which
    the output's size doesn't bound (COPIES); ValueError below one pixel.

    The choices, in the planner's order (DESIGN.md, Memory planner, Choices), each made once for
    the bound:
    1. the floor (the planner's _floor): the least each phase takes over every picture within the
       bound, the DiT's with every block swapped and the VAE away, the VAE's passes with the DiT
       away; a phase over the budget is refused, naming the largest bound that fits;
    2. the DiT on one latent, the fewest swapped blocks, the VAE leaving the DiT's phase before
       one more block is swapped (_fewest); --swap-blocks exactly (_swap_refused);
    3. the VAE's placement (_place): what of the DiT stays on the GPU in each VAE phase, as much
       as the bound's own picture allows (the planner's _pass and _resident), lowered until every
       picture within the bound has a pass that fits (_worst); a pin proved for its own tiles;
    4. the system memory (the planner's _unfit): where the plan takes more than is available,
       the VAE's weights kept for the DiT's phase with more blocks swapped (_vae_kept), then more
       of the DiT kept on the GPU in the VAE's phases (_dit_kept), as long as the proof holds.
    The input's copies (step 3, _holds) are judged with the proof, and refused at the floor
    (_input_refused).
    Then, on a held run, a refusal is planned again with memory.RESERVE_TIGHT in
    memory.RESERVE's place, as the planner's plan does (memory.tightened; DESIGN.md, Allocator),
    only under torch's own allocator with expandable segments (Reading.expandable)."""
    _check_bound(bound)
    if input_pixels < 1:
        raise ValueError(f"an input of {input_pixels} pixels")
    if attention not in peaks.DIT_ATTENTION:
        raise ValueError(
            f"attention {attention!r}: the DiT's fit is {peaks.FLASH_ATTN_2}'s or {peaks.SDPA}'s"
        )
    if dit_model not in peaks.DITS:
        raise ValueError(f"{dit_model.name}: the DiT's figures are for its fp16 and int8 files")
    reading = free if isinstance(free, Reading) else Reading(free, None, free, False)
    job = _Job(
        bound, input_pixels, reading, attention, dit_model, split, pins, process, host_available
    )
    if pins.swap_blocks is not None and pins.swap_blocks > peaks.BLOCKS:
        raise PlanRefusal(
            f"--swap-blocks {pins.swap_blocks}: the DiT has {peaks.BLOCKS} blocks, so from 0 to"
            f" {peaks.BLOCKS}",
            phase="pin",
        )
    # The planner's refusal of a tile pin that covers the frame, before any memory is counted: for
    # a bound, the larger of its two orientations' frames; a smaller picture takes the untiled pass
    # then (tiles), which is no tiling.
    largest = max(_frames(job.bound), key=max)
    for phase, pin in (("encode", pins.encode_tile), ("decode", pins.decode_tile)):
        if pin and peaks.covers(largest, Tiles(pin)):
            height, width = largest
            raise PlanRefusal(
                f"--{phase}-tile {pin}: a tile of {pin} px covers the whole padded frame,"
                f" {width}x{height}: --{phase}-tile 0 pins an untiled {phase}",
                phase="pin",
            )
    try:
        return _planned(job)
    except PlanRefusal as error:
        if not reading.held or reading.tight or memory.RESERVE_TIGHT >= reading.reserve:
            raise
        refused = error
    tight = replace(job, reading=memory.tightened(reading))
    if not reading.expandable:
        # Refused with the usual reserve, as the planner does, saying what torch's default
        # setting would do where it would run the picture.
        try:
            _planned(tight)
        except PlanRefusal:
            raise PlanRefusal(
                str(refused), refused.phase, refused.fits, refused.fits_input
            ) from None
        raise PlanRefusal(
            f"{refused}; torch's own allocator with its expandable segments, a held run's"
            " default, which the allocator's setting in the environment replaces here, would run"
            f" it with the reserve of {_gib(memory.RESERVE_TIGHT)} GiB, slower",
            refused.phase,
            refused.fits,
            refused.fits_input,
        ) from None
    try:
        made = _planned(tight)
    except PlanRefusal as again:
        raise PlanRefusal(
            f"{again}; refused with the reserve of {_gib(memory.RESERVE)} GiB that a held run's"
            " plan leaves torch's allocator beyond a step's peak, and with"
            f" {_gib(memory.RESERVE_TIGHT)}, the one it takes where that is refused, whose"
            " figures these are",
            again.phase,
            again.fits,
            again.fits_input,
        ) from None
    return replace(made, tightened=str(refused))


def tiles(plan: StillPlan, target: tuple[int, int], budget: int | None = None) -> PictureTiles:
    """The VAE passes of a picture resized to `target`, (height, width) as frames.target_size
    gives it, under the plan's placement: a pinned pass as pinned, or the first that fits of the
    planner's order (_options: untiled, then the tiles from the largest down), each judged with
    its room (VAE_ROOM). A pin that covers the picture's padded frame gives the untiled pass:
    a covering tile is no tiling (peaks.covers). ValueError for a target whose output is not
    within the plan's bound in one orientation or the other, or too small for the padding
    (frames.check_target's rule). Never a refusal within the bound: the proof of the plan holds for
    every picture in it.

    `budget`, bytes: what the picture's passes may count on in place of the plan's budget, for a
    server that holds each picture to what the bound's own pictures take (the image server's
    reservation); at least the proof's worst pass with its room in each phase (bound_peaks), so
    that every picture still has a pass, and at most the plan's budget (ValueError over it)."""
    long, short = plan.bound
    height, width = output_size(target)
    if min(height, width) > short or max(height, width) > long:
        raise ValueError(
            f"a target of {target[1]}x{target[0]} pixels, output {width}x{height}, is not within"
            f" the plan's bound, {long}x{short} in either orientation"
        )
    if not _paddable(target):
        raise ValueError(f"a target of {target[1]}x{target[0]} pixels is too small for the padding")
    job = _job_of(plan)
    if budget is not None:
        if budget > plan.budget:
            raise ValueError(f"a budget of {budget} bytes, over the plan's {plan.budget}")
        job = replace(job, reading=replace(job.reading, counted=budget + job.margin))
    padded, output = padded_size(target), (height, width)
    models = plan.models
    encode = _first(job, "encode", padded, output, models.swapped, models.resident_encode)
    decode = _first(job, "decode", padded, output, models.swapped, models.resident_decode)
    assert encode is not None and decode is not None, (
        f"{target}: no pass fits within {plan.bound}: the plan's proof is wrong"
    )

    def peak(phase: Phase, chosen: Tiles | None, resident: int | None) -> int:
        return _bare(
            phase, padded, output, chosen, resident, models.swapped, plan.split, plan.dit_model
        )

    return PictureTiles(
        encode[0],
        decode[0],
        peak("encode", encode[0], models.resident_encode),
        peak("decode", decode[0], models.resident_decode),
    )


def _job_of(plan: StillPlan) -> _Job:
    """The job a made plan was made by, for the figures that follow from it."""
    return _Job(
        plan.bound,
        plan.input_pixels,
        plan.reading,
        plan.attention,
        plan.dit_model,
        plan.split,
        plan.pins,
        plan.process,
        plan.host_available,
    )


# ---- What a picture within the bound is --------------------------------------------------------


def _paddable(target: tuple[int, int]) -> bool:
    """Whether the padding can be made on a target, (height, width): frames.check_target's rule,
    17 rows and 16 columns at least (torch reflects fewer rows than the picture has)."""
    rows, columns = padding(target)
    return rows < target[0] and columns < target[1]


def _check_bound(bound: Bound) -> None:
    """ValueError for a bound that is no output size: sides that aren't even (frames.output_size
    crops them so), the long under the short, or a short side too small for the padding."""
    long, short = bound
    if long % 2 or short % 2 or long < short or not _paddable((short, long)):
        raise ValueError(
            f"a bound of {long}x{short}: even sides, the long one first, 17 rows and 16"
            " columns at least"
        )


@cache
def _rows(side: int) -> tuple[int, ...]:
    """The padded row counts, ascending, of the pictures whose height gives an output of `side`
    rows or fewer (frames.output_size crops to even, so a target of side + 1 rows is within an
    even side), as frames.padded_size and frames.check_target take them. Padding is separable
    (frames.padding: the rows reflected depend on the height alone, the columns on the width
    alone, and padded_size adds each to its own side), so a width of one multiple of 16 stands
    for the picture's columns here, as _columns' height does; tests/test_still_plan.py holds it."""
    heights = (
        height
        for height in range(1, side + 2)
        if height - height % 2 <= side and padding((height, MULTIPLE))[0] < height
    )
    return tuple(sorted({padded_size((height, MULTIPLE))[0] for height in heights}))


@cache
def _columns(side: int) -> tuple[int, ...]:
    """The padded column counts, ascending, of the pictures whose width gives an output of `side`
    columns or fewer (_rows)."""
    widths = (
        width
        for width in range(1, side + 2)
        if width - width % 2 <= side and padding((2 * MULTIPLE, width))[1] < width
    )
    return tuple(sorted({padded_size((2 * MULTIPLE, width))[1] for width in widths}))


@cache
def _orientations(bound: Bound) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """The padded rows and columns of the pictures within the bound, for each orientation: the
    short side along the rows (landscape) or along the columns (portrait). Rows and columns are
    independent (_rows), so every pair is a picture."""
    long, short = bound
    return ((_rows(short), _columns(long)), (_rows(long), _columns(short)))


def _frames(bound: Bound) -> tuple[tuple[int, int], tuple[int, int]]:
    """The largest padded frame of each orientation (_orientations): every picture within the
    bound is padded to a frame no larger in either side."""
    (rows_a, columns_a), (rows_b, columns_b) = _orientations(bound)
    return (rows_a[-1], columns_a[-1]), (rows_b[-1], columns_b[-1])


def _most(bound: Bound) -> tuple[int, int]:
    """The padded frame of the most tokens within the bound, the DiT's tokens growing with each
    side: the larger of the two orientations' largest frames."""
    return max(_frames(bound), key=lambda frame: frame[0] * frame[1])


# ---- The DiT on one latent ---------------------------------------------------------------------


def _axis_regular(length: int, window: int) -> int:
    """The windows of the regular layout (vendor/models/dit_7b/window.py,
    make_720Pwindows_bysize) along one axis: ceil(length / window), none of which the layout's
    filter (min((i + 1) * window, length) > i * window) drops, i running to that count."""
    return -(-length // window)


def _axis_shifted(length: int, window: int) -> int:
    """The windows of the shifted layout (make_shifted_720Pwindows_bysize) along one axis, as its
    list comprehension keeps them. A window not shorter than the axis is one window, unshifted.
    Else the layout shifts by half a window and has n = ceil((length - 1/2) / window) + 1
    positions, i, of which it keeps those with end > start, start = max(int((i - 1/2) * window), 0)
    and end = min(int((i + 1/2) * window), length). The positions between the first and the last
    always keep: their start is under length (i <= n - 2 puts it under (n - 2) * window, itself
    under length - 1/2) and their end is a whole window past it. The first is empty for a window
    of 1 (end = int(1/2) = 0), the last when its start has reached the axis."""
    if window >= length:
        return 1
    count = math.ceil((length - 0.5) / window) + 1
    first = int(0.5 * window) > 0
    last = min(int((count - 0.5) * window), length) > int((count - 1.5) * window)
    return count - 2 + first + last


def _windows(rows: int, columns: int) -> int:
    """The attention windows the DiT cuts one latent of rows x columns tokens into: the larger of
    its two layouts' counts (peaks.attention_windows), by the window functions' own arithmetic
    without building their lists. The time axis, one latent, is one window in both (a window of
    ceil(min(1, 30) / 4) = 1 latent, no shift), so the layouts' counts are the products of their
    two other axes'."""
    scale = math.sqrt((45 * 80) / (rows * columns))
    _, windows_rows, windows_columns = peaks.NUM_WINDOWS
    window_rows = -(-round(rows * scale) // windows_rows)
    window_columns = -(-round(columns * scale) // windows_columns)
    return max(
        _axis_regular(rows, window_rows) * _axis_regular(columns, window_columns),
        _axis_shifted(rows, window_rows) * _axis_shifted(columns, window_columns),
    )


@cache
def windows_most(bound: Bound) -> int:
    """The most attention windows the DiT cuts a picture within the bound into, one latent: the
    larger of its two layouts' counts (peaks.attention_windows) at every token grid (the padded
    frame / 16) of every picture within the bound, both orientations.

    Not the bound's own count: the windows are not monotone in the picture's size (vendor/models/
    dit_7b/window.py): over token grids up to 300 x 560 the count reaches 441, and neighbouring
    shapes differ by tens (174 x 558 tokens take 240 windows, 174 x 559 take 289; 2.726 MiB a
    window, peaks.PER_WINDOW), so a smaller picture can take more windows than the bound does (a
    3000x2000 bound: 72 for its own, 81 for 2970x1898). The DiT's peak adds this to the tokens of
    the largest frame (_dit_need), each the most it can be: a bound of every picture's. Plain
    Python on the 187,000 token grids of an 8K bound in 0.10 s, 125,000 at 6000x4000 in 0.07 s
    (measured on one CPU core, 2026-11-03)."""
    token_columns: dict[int, set[int]] = {}
    for rows, columns in _orientations(bound):
        for row in rows:
            token_columns.setdefault(row // 16, set()).update(column // 16 for column in columns)
    return max(
        _windows(row, column) for row, columns in token_columns.items() for column in columns
    )


def _per_window(job: _Job) -> float:
    return peaks.INT8_PER_WINDOW if job.dit_model == peaks.INT8 else peaks.PER_WINDOW


def _dit_need(job: _Job, swapped: int, vae_in_dit: bool) -> int:
    """The predicted peak of the DiT's phase over every picture within the bound, in bytes, one
    latent, `swapped` blocks swapped and the VAE's weights there or not: peaks.dit_peak's, with
    the tokens of the largest padded frame (_most) and the most windows any picture takes
    (windows_most), the larger of each, which no one picture need take together."""
    least = peaks.dit_least(_most(job.bound), 1, swapped, vae_in_dit, job.attention, job.dit_model)
    return round(least + _per_window(job) * windows_most(job.bound))


def _fewest(job: _Job) -> tuple[int, bool] | None:
    """The fewest swapped blocks with which the DiT's phase holds the bound, and whether the
    VAE's weights stay on the GPU meanwhile: they leave before one more block is swapped. One and
    two swapped blocks save nothing (peaks.dit_peak), so are never chosen. None when every block
    swapped and the VAE away don't hold it. The planner's _fewest."""
    for swapped in (0, *range(peaks.SLOTS + 1, peaks.BLOCKS + 1)):
        for vae_in_dit in (True, False):
            if _dit_need(job, swapped, vae_in_dit) <= job.budget:
                return swapped, vae_in_dit
    return None


def _swap_choice(job: _Job) -> tuple[int, bool]:
    """The swapped blocks and whether the VAE stays in the DiT's phase: the fewest that hold it
    (_fewest), or --swap-blocks exactly, the VAE staying if it fits and leaving if only that fits
    (the planner's _window), refused when neither holds: the DiT's phase, naming the fewest
    blocks that do (the planner's _swap_refused). The floor holds, so the fewest exist."""
    fewest = _fewest(job)
    assert fewest is not None
    swapped = job.pins.swap_blocks
    if swapped is None:
        return fewest
    if _dit_need(job, swapped, vae_in_dit=False) > job.budget:
        need = _dit_need(job, swapped, vae_in_dit=False) + job.margin
        raise PlanRefusal(
            f"--swap-blocks {swapped}: the DiT's phase, one latent of the largest picture within"
            f" {_bound_said(job.bound)}, takes {_over(need)} GiB of free VRAM with"
            f" {_count(swapped, 'block')} swapped, and {_there(job.reading)}:"
            f" {_to_free(job.reading, need)}the fewest blocks that hold it are --swap-blocks"
            f" {fewest[0]}",
            phase="dit",
        )
    return swapped, _dit_need(job, swapped, vae_in_dit=True) <= job.budget


# ---- The VAE's passes --------------------------------------------------------------------------


def _tile_sizes(padded: tuple[int, int]) -> range:
    """The sides of the tiles the plan makes for padded frames, the largest first: the planner's
    _tile_sizes. Multiples of TILE_STEP from the largest under the frame's longer side, a tile
    that covers the frame being no tiling (peaks.covers), or from TILE_MAX on a frame longer than
    it, down to TILE_MIN; none for a frame that small."""
    largest = min(peaks.TILE_MAX, (max(padded) - 1) // peaks.TILE_STEP * peaks.TILE_STEP)
    return range(largest, peaks.TILE_MIN - 1, -peaks.TILE_STEP)


def _untiled_measured(phase: Phase, padded: tuple[int, int]) -> bool:
    """Whether one frame's untiled pass was measured on a frame as large as `padded`, either
    orientation (peaks.ONE_FRAME_UNTILED_MEASURED): the plan chooses no untiled pass past it, a
    pin alone taking one there, at the video slice's figure (DESIGN.md, Still images: the planner
    keeping no blind spot; the review of 2026-11-03)."""
    rows, columns = peaks.ONE_FRAME_UNTILED_MEASURED[0 if phase == "encode" else 1]
    return max(padded) <= max(rows, columns) and min(padded) <= min(rows, columns)


def _options(padded: tuple[int, int], pin: int | None, phase: Phase) -> Iterator[Tiles | None]:
    """The passes a picture of padded frame `padded` may take in the phase, in order: a pin
    exactly (0: untiled; a tile that covers the frame is no tiling, the untiled pass), else the
    planner's order (_fitting): untiled where one frame's untiled pass was measured on such a
    frame (_untiled_measured), then the tiles from the largest down."""
    if pin is None:
        if _untiled_measured(phase, padded):
            yield None
        yield from (Tiles(size) for size in _tile_sizes(padded))
    elif pin == 0 or peaks.covers(padded, Tiles(pin)):
        yield None
    else:
        yield Tiles(pin)


def _bare(
    phase: Phase,
    padded: tuple[int, int],
    output: tuple[int, int],
    tiles: Tiles | None,
    resident: int | None,
    swapped: int,
    split: bool,
    dit_model: Architecture,
) -> int:
    """The predicted peak of a picture's encode or decode in `tiles`, in bytes, with `resident`
    blocks of the DiT on the GPU and `swapped` swapped (peaks.picture_encode_peak,
    peaks.picture_decode_peak: one frame, the colour correction at the output's size when the
    decode splits)."""
    if phase == "encode":
        return peaks.picture_encode_peak(padded, tiles, resident, swapped, dit_model)
    return peaks.picture_decode_peak(padded, output, tiles, resident, swapped, split, dit_model)


def _first(
    job: _Job,
    phase: Phase,
    padded: tuple[int, int],
    output: tuple[int, int],
    swapped: int,
    resident: int | None,
) -> tuple[Tiles | None] | None:
    """The pass a picture takes: the first of its options (_options) that fits the budget with
    its room (VAE_ROOM), the planner's _pass, in a 1-tuple (None is the untiled pass); None when
    none does."""
    for chosen in _options(padded, job.pin(phase), phase):
        peak = _bare(phase, padded, output, chosen, resident, swapped, job.split, job.dit_model)
        if peak + VAE_ROOM <= job.budget:
            return (chosen,)
    return None


def _lengths(padded_side: int, size: int) -> frozenset[int]:
    """The lengths, in latents, of the spans a side padded to `padded_side` px is cut into by tiles
    of `size` px: a tile's rows are a span of the rows' side and its columns a span of the
    columns' (tilegrid.grid lays the product of the two), so one side's spans are the grid of a
    frame one latent wide."""
    laid = tilegrid.grid((padded_side // tilegrid.SCALE, 1), Tiles(size))
    return frozenset(tile.rows for tile in laid)


@cache
def _worst(bound: Bound, phase: Phase, size: int | None, split: bool) -> int:
    """The most any picture within the bound needs of a VAE pass, in bytes, with the DiT away
    (resident 0) and without the room (VAE_ROOM): every picture has a pass that fits when this is
    within the budget (the floor proof). `size` is the pass the pictures are proved for:
    - a pin: 0, the untiled pass for all; else tiles of that side for the pictures whose padded
      frame is longer than it, and the untiled pass for those it covers (_options: peaks.covers
      is size >= the longer side);
    - None, the unpinned pictures: the less of the untiled pass (where one frame's was measured
      on such a frame, _untiled_measured; else the tiles alone) and of tiles of TILE_MIN, the
      smallest the planner makes (a larger tile may take less on a strip or a cut-off shape, and
      the untiled pass can, on a frame little larger than a tile: the planner's _least_pass).
      The picture takes the first pass of the planner's order that fits (_first), so it has one
      when either does; the pictures not over TILE_MIN take the untiled pass alone.

    The tiles: a tiled pass peaks at the most one tile of its grid takes (peaks._fit), and a
    grid's tile shapes are the product of its rows' span lengths and its columns' (_lengths). So
    the pass depends on the padded frame by the two sets of lengths alone, and one padded frame
    per pair of sets stands for all of them (at most about 60 sets a side), the pair taken through
    peaks.picture_encode_peak and peaks.picture_decode_peak: the largest frame of the pair (the
    rows and columns ascend, and the key keeps the last), since their terms beside the tile's,
    the picture's frame in the encode and the colour correction in the decode, grow with the
    frame. A side's set includes whether it is longer than the tile: {64} is both 512 px whole
    and 960 px in two spans, and only the latter is cut. The colour correction, one frame, is at
    the largest output the frame can give within the bound (a picture's decode takes the larger
    of its pass and the correction, peaks.picture_decode_peak).

    The untiled passes: the pictures too small for the tiles are the padded frames whose sides
    are both within the tile. The untiled pass takes more of the GPU the larger the frame: both
    of its fits (peaks.ENCODE_UNTILED, peaks.DECODE_UNTILED and the small tiles' fit it is the
    larger of) have positive slopes in the area, taken at one ratio whatever the size
    (peaks.ONE_FRAME_UNTILED), and so do the frame's and the correction's terms, so the largest
    frame, in area, is the worst. The
    same holds in a pair of sets of a cut picture: its members all take the same tile, and the
    untiled pass of the largest of them (the key's last row and column) is the most of theirs, so
    the worst member needs the less of the two."""
    long, short = bound
    tile = peaks.TILE_MIN if size is None else size
    worst = 0
    for (rows, columns), (high, wide) in zip(
        _orientations(bound), ((short, long), (long, short)), strict=True
    ):
        # The output of a picture is no larger than its padded frame, nor than the bound's side
        # along that axis: the colour correction's term at the most it can be.
        def pass_of(
            frame: tuple[int, int], tiles: Tiles | None, high: int = high, wide: int = wide
        ) -> int:
            output = (min(frame[0], high), min(frame[1], wide))
            return _bare(phase, frame, output, tiles, 0, 0, split, peaks.FP16)

        if tile:
            keyed_rows = {(_lengths(row, tile), row > tile): row for row in rows}
            keyed_columns = {(_lengths(column, tile), column > tile): column for column in columns}
            for (_, cut_row), row in keyed_rows.items():
                for (_, cut_column), column in keyed_columns.items():
                    if cut_row or cut_column:
                        need = pass_of((row, column), Tiles(tile))
                        if size is None and _untiled_measured(phase, (row, column)):
                            need = min(need, pass_of((row, column), None))
                        worst = max(worst, need)
        small_rows = [row for row in rows if not tile or row <= tile]
        small_columns = [column for column in columns if not tile or column <= tile]
        if small_rows and small_columns:
            worst = max(worst, pass_of((small_rows[-1], small_columns[-1]), None))
    return worst


def _proof_need(job: _Job, phase: Phase, resident: int | None, swapped: int) -> int:
    """What the phase takes of the free VRAM at its worst picture within the bound with `resident`
    blocks of the DiT on the GPU, in bytes, its room with it: the proof's worst pass at resident
    0 (_worst), whose peak is the pass's own and what of the DiT is on the GPU, plus what the
    placement keeps there beyond it (peaks.dit_on_gpu; the rest of a pass's peak doesn't depend
    on it)."""
    base = _worst(job.bound, phase, job.pin(phase), job.split)
    kept = peaks.dit_on_gpu(resident, swapped, job.dit_model)
    return base + kept - peaks.dit_on_gpu(0, swapped, job.dit_model) + VAE_ROOM


def _resident(
    job: _Job, phase: Phase, padded: tuple[int, int], output: tuple[int, int], swapped: int
) -> int | None:
    """What of the DiT is left on the GPU in the phase, for the picture of padded frame `padded`
    in the pass it takes with the DiT away: the planner's _resident. None, the DiT staying as its
    own phase has it, when that fits; else as many of the blocks that aren't swapped as fit, from
    all of them down, the slots freed; 0 when one block and the other parameters don't fit."""
    found = _first(job, phase, padded, output, swapped, 0)
    assert found is not None, "the floor holds: some pass fits with the DiT away"

    def holds(resident: int | None) -> bool:
        peak = _bare(phase, padded, output, found[0], resident, swapped, job.split, job.dit_model)
        return peak + VAE_ROOM <= job.budget

    if holds(None):
        return None
    for resident in range(peaks.BLOCKS - swapped, 0, -1):
        if holds(resident):
            return resident
    return 0


def _level(resident: int | None, swapped: int) -> int:
    """The blocks of the DiT a phase has on the GPU that aren't swapped: all of them where
    nothing moves (None)."""
    return peaks.BLOCKS - swapped if resident is None else resident


def _residents(swapped: int) -> list[int | None]:
    """Every resident count a phase may have, from the most to the least, in the order the plan
    lowers them: None, the blocks that aren't swapped down to 0. With no block swapped, all of
    them is what None has."""
    first = peaks.BLOCKS - swapped - (swapped == 0)
    return [None, *range(first, -1, -1)]


def _place(job: _Job, swapped: int, vae_in_dit: bool) -> ModelPlan:
    """Where the models are for the bound (step 3 of plan): for each VAE phase, decode then
    encode as the planner does, what of the DiT stays on the GPU for the bound's own picture in
    each orientation, the less of the two (None the most); then lowered, in the order of
    _residents, until every picture within the bound has a pass that fits at it (_proof_need) and
    the input's copies fit beside the models (_copies_need): _holds. Both hold at 0 (the floor,
    the pins', and _input_refused), so it ends."""
    chosen: dict[Phase, int | None] = {}
    for phase in ("decode", "encode"):
        output = (job.bound[1], job.bound[0])
        found = [_resident(job, phase, frame, output, swapped) for frame in _frames(job.bound)]
        resident = min(found, key=lambda each: (_level(each, swapped), each is None))
        order = _residents(swapped)
        for each in order[order.index(resident) :]:
            if _holds(job, phase, each, swapped):
                chosen[phase] = each
                break
    return ModelPlan(swapped, chosen["encode"], chosen["decode"], vae_in_dit)


# ---- The floor ---------------------------------------------------------------------------------


def _floor(job: _Job) -> tuple[Phase, int] | None:
    """The phase that takes the most where every phase takes its least over the pictures within
    the bound, and what it takes there, when that is over the budget, no plan then fitting
    (the planner's _floor); else None. The DiT's: one latent, every block swapped and the VAE
    away. The VAE's: the unpinned proof (_worst), the DiT away, the room with it."""
    least: dict[Phase, int] = {
        "encode": _worst(job.bound, "encode", None, job.split) + VAE_ROOM,
        "dit": _dit_need(job, peaks.BLOCKS, vae_in_dit=False),
        "decode": _worst(job.bound, "decode", None, job.split) + VAE_ROOM,
    }
    phase = max(PHASES, key=lambda each: least[each])
    return (phase, least[phase]) if least[phase] > job.budget else None


def _fits(job: _Job) -> Bound | None:
    """The largest bound that fits the job's memory, below the job's, as the planner's _too_large
    searches the output: the short side down in steps of 16 from the multiple of 16 under it, the
    long side in proportion (to an even number), while the padding can be made (_paddable), the
    floor checked at each; None when none does. The floor of a smaller bound is never above a larger
    one's (the pictures within it are some of theirs, and each figure is the most over them), so
    the search is bisected, and finds what the planner's walk down would."""
    long, short = job.bound
    candidates: list[Bound] = []
    for side in range((short - 1) // 16 * 16, 0, -16):
        smaller = (side * long // short // 2 * 2, side)
        if not _paddable((side, smaller[0])):
            break  # too small for the padding, as every smaller one is
        candidates.append(smaller)
    low, high = 0, len(candidates)
    while low < high:
        middle = (low + high) // 2
        if _floor(replace(job, bound=candidates[middle])) is None:
            high = middle
        else:
            low = middle + 1
    return candidates[low] if low < len(candidates) else None


def _too_large(job: _Job, phase: Phase, least: int) -> PlanRefusal:
    """The floor's refusal (_floor), counted in free VRAM, never in card sizes: the free VRAM
    the bound takes, what is there for it (_there), and the largest bound that fits in it
    (_fits); the planner's _too_large."""
    if phase == "dit":
        how = (
            "one latent, with every block of the DiT swapped and the VAE off the GPU, at the"
            " most any picture within it takes"
        )
        room = 0
    else:
        how = (
            f"its {phase} in tiles of {peaks.TILE_MIN} px or untiled, with the DiT off the GPU,"
            " at the most any picture within it takes"
        )
        room = VAE_ROOM
    # The room as what is left of the total once the peak and the margin are said, each rounded
    # as the message has it, so that its figures add up.
    total, peak = _hundredths(least + job.margin, up=True), _hundredths(least - room, up=True)
    rest = (total - peak - _hundredths(job.margin, up=True)) / 100
    held = f", {rest:.2f} of room a VAE pass is held to" if room else ""
    margin = f", and {_gib(job.margin)} of margin" if job.margin else ""
    fits = _fits(job)
    fitting = (
        "no output size fits there"
        if fits is None
        else f"the largest bound that fits there is {_bound_said(fits)}"
    )
    return PlanRefusal(
        f"pictures within {_bound_said(job.bound)} take {total / 100:.2f} GiB of free VRAM"
        f" ({how}: a peak of {peak / 100:.2f} GiB{held}{margin}), and {_there(job.reading)}:"
        f" {_to_free(job.reading, least + job.margin)}{fitting}",
        phase=phase,
        fits=fits,
    )


def _pin_refused(job: _Job, phase: Phase) -> PlanRefusal | None:
    """The refusal of the phase's tile pin, None when it holds: the pin is taken exactly or
    refused, and it holds when every picture within the bound has a pass that fits with it, the
    DiT away (_worst at the pin; the planner's _pass). The refusal names the largest tile that
    fits, in the planner's sizes at the bound's largest frame."""
    pin = job.pin(phase)
    if pin is None:
        return None
    need = _worst(job.bound, phase, pin, job.split) + VAE_ROOM
    if need <= job.budget:
        return None
    sizes = _tile_sizes(max(_frames(job.bound), key=max))
    largest = next(
        (
            size
            for size in sizes
            if _worst(job.bound, phase, size, job.split) + VAE_ROOM <= job.budget
        ),
        None,
    )
    fits = (
        f"no tile fits, the untiled {phase} does, --{phase}-tile 0"
        if largest is None
        else f"the largest tile that fits is --{phase}-tile {largest}"
    )
    said = "untiled" if pin == 0 else f"in tiles of {pin} px"
    need += job.margin
    return PlanRefusal(
        f"--{phase}-tile {pin}: the {phase} {said}, for the pictures within"
        f" {_bound_said(job.bound)}, takes {_over(need)} GiB of free VRAM, the DiT off the GPU,"
        f" {VAE_ROOM / peaks.GIB:.2f} of them the room a VAE pass is held to, and"
        f" {_there(job.reading)}: {_to_free(job.reading, need)}{fits}",
        phase="pin",
    )


# ---- The picture's own copies on the GPU -------------------------------------------------------

# still.upscale moves the picture to the device at its STORED size and resizes it there
# (still.py _transformed: `.permute(0, 3, 1, 2).to(device, dtype)`, then model.input_transform's
# Compose), so a 100 MP photo takes memory the output's size doesn't bound. The tensors the
# transform makes on the device from a (1, 3, H, W) input, in the Compose's order
# (model.py:428-434), torchvision 0.29.1 (the locked version), each alive while the next is made
# (the Compose's `img = t(img)`), and the stored tensor alive through all of them
# (_transformed's `frames`):
#
#   1. the stored picture, (1, 3, H, W) in the encode's dtype, bfloat16 (still.py _transformed
#      with model.COMPUTE_DTYPE, model.py:52; model.to_input made it float16 on the CPU,
#      model.py:406-410). The colour correction's reference is float32 (torch.as_tensor of the
#      float32 pixels).
#   2. TVF.resize(BICUBIC, antialias=True) (model.py:430): torchvision/transforms/
#      _functional_tensor.py:441-474. _cast_squeeze_in(img, [float32, float64]) UPCASTS a bfloat16
#      input to float32 (:516-529), a float32 copy of the stored picture; interpolate makes the
#      float32 output at the target's size (<= the padded frame); _cast_squeeze_out casts it back
#      to bfloat16 (:532-542). torch's antialiased bicubic on CUDA, given the channels-last view
#      `frames` is, copies its float32 input to a contiguous tensor and writes its output twice,
#      contiguous then in the input's layout. Measured on the RTX 5090 (2026-11-03, design's
#      agent, 8 shapes from 960x540 to 12000x6750, matched to 0.0001 GiB): the resize holds 30
#      bytes a stored pixel in the encode (the stored picture 6, its float32 copy 12, the
#      contiguous copy 12) and 24 in the decode (the float32 stored picture and its contiguous
#      copy), and 24 a pixel of its output in both (the float32 output twice), the bfloat16 cast
#      after it within that. One moment, RESIZE.
#   3. torch.clamp(x, 0, 1) (model.py:431): a new tensor beside the resize's.
#   4. pad (model.py:439-450, Lambda(pad) at :432): the clamped tensor, `reflected` and the
#      result of the zeros' second F.pad alive at once, three tensors of the padded frame's size
#      at most.
#   5. Normalize(0.5, 0.5) (:433; _functional_tensor.py:916-928 clones its input, then sub_ and
#      div_ in place): the padded tensor and its clone.
#   6. permute(1, 0, 2, 3) (:434): a view. After the call returns only the padded tensor stays,
#      in the encode's dtype: the encode's input, which the encode's measured peak holds already
#      (peaks.encode_peak's docstring: the frames the encode takes are part of what it measures,
#      seedvr2x feeding it slice by slice), so no moment is counted past this one.
#
# Every target-size tensor is counted at the bound's largest padded frame (the resize's output is
# at most the target, itself within the padded frame), the stored picture at `input_pixels`.
# The host-to-device `.to()` of the permuted view stages the stored picture once more, 12 bytes a
# stored pixel at the most, under the resize's moment. cuBLAS's workspace (peaks.CUBLAS_WORKSPACE)
# is held beside every moment: allocated at the first matrix product, kept for the process, so
# there from an image server's second picture on.
#
# The decode's phase (colour correction only): the decoded frame is alive while the reference is
# made. It is bfloat16 (the VAE's dtype, model.COMPUTE_DTYPE; tiling.decode_tiled sums in the
# same dtype), and `chunk` is a view of the slice the decode gave (still.upscale), so its storage
# is the PADDED frame's, 3 channels. The decode's own working memory is gone: the generator is
# closed first (still.upscale's `finally`).
#
# The moments are judged in their phase like a phase, but without VAE_ROOM and with the margin:
# no VAE pass runs, and the room is a pass's (a convolution whose workspace the allocator refuses
# runs on another engine and may give other frames: peaks.VAE_ROOM), where here are ATen's resize
# and copies, which allocate the exact sizes above. The margin stays: it is the desktop's and the
# first allocation's, whatever runs. PROVISIONAL until the card's batch. A contiguous input to the
# resize, which would drop a copy and a write, waits for its bits to be checked against the
# video path's (DESIGN.md, Still images, decided on 2026-11-03).


@dataclass(frozen=True)
class Moment:
    """A moment of the picture's copies on the GPU: what is alive, in bytes per pixel of the
    stored picture and per pixel of the bound's largest padded frame, and `held` bytes beside
    them whatever the sizes (cuBLAS's workspace)."""

    what: str
    per_input: int
    per_padded: int
    held: int = 0

    def at(self, input_pixels: int, padded_pixels: int) -> int:
        return self.per_input * input_pixels + self.per_padded * padded_pixels + self.held


CHANNELS = 3
FLOAT32 = 4  # bytes: the resize's working dtype
BFLOAT16 = 2  # bytes: the encode's dtype (model.COMPUTE_DTYPE)


# The resize's moment as measured (step 2 above), in bytes a stored pixel and a pixel of the
# output, the encode's then the decode's.
RESIZE: dict[Phase, tuple[int, int]] = {"encode": (30, 24), "decode": (24, 24)}


def _moments(phase: Phase, itemsize: int, held: int) -> tuple[Moment, ...]:
    """The moments of the transform in the phase, in a dtype of `itemsize` bytes (the stored
    picture's, step 1 above), `held` bytes per padded pixel alive besides (the decoded frame), and
    cuBLAS's workspace beside each."""
    sample = CHANNELS * itemsize
    per_input, per_output = RESIZE[phase]
    workspace = peaks.CUBLAS_WORKSPACE
    return (
        Moment(
            "the resize: the stored picture, its float32 and contiguous copies, and its output"
            " written twice",
            per_input,
            per_output + held,
            workspace,
        ),
        Moment(
            "the stored picture, the resize's output and the clamped one",
            sample,
            2 * sample + held,
            workspace,
        ),
        Moment("the stored picture and the padding's three", sample, 3 * sample + held, workspace),
        Moment(
            "the stored picture, the padded tensor and its normalised clone",
            sample,
            2 * sample + held,
            workspace,
        ),
    )


# The encode's phase: bfloat16. The decode's, with the colour correction: float32, and the
# decoded frame (bfloat16, 3 channels, the padded frame's storage) beside.
COPIES: dict[Phase, tuple[Moment, ...]] = {
    "encode": _moments("encode", BFLOAT16, 0),
    "decode": _moments("decode", FLOAT32, CHANNELS * BFLOAT16),
}


def copies_peak(phase: Phase, input_pixels: int, padded_pixels: int) -> int:
    """The most the picture's copies and the transform's tensors take of the GPU at once in the
    encode's or the decode's phase, in bytes, above what the models take there (COPIES), by the
    rule measured on the RTX 5090, provisional until the card's batch."""
    return max(moment.at(input_pixels, padded_pixels) for moment in COPIES[phase])


def _padded_pixels(bound: Bound) -> int:
    """The area of the bound's largest padded frame (_most), in pixels."""
    rows, columns = _most(bound)
    return rows * columns


def _copy_phases(job: _Job) -> tuple[Phase, ...]:
    """The phases in which the picture is copied and resized: the encode's, and the decode's where
    the colour correction compares it with the picture."""
    return ("encode", "decode") if job.split else ("encode",)


def _on_gpu(job: _Job, resident: int | None, swapped: int) -> int:
    """What is on the GPU in a VAE phase before anything of the picture's: the VAE's weights and
    what of the DiT stays there, as peaks.encode_peak counts them."""
    return peaks.VAE_WEIGHTS + peaks.dit_on_gpu(resident, swapped, job.dit_model)


def _copies_need(job: _Job, phase: Phase, resident: int | None, swapped: int) -> int:
    """What the phase takes of the free VRAM at the moment of its copies, in bytes: the models on
    the GPU there and the copies at the bound's largest padded frame (copies_peak). Judged
    against the budget without VAE_ROOM."""
    return _on_gpu(job, resident, swapped) + copies_peak(
        phase, job.input_pixels, _padded_pixels(job.bound)
    )


def _holds(job: _Job, phase: Phase, resident: int | None, swapped: int) -> bool:
    """Whether the phase holds at `resident` blocks of the DiT on the GPU: every picture within
    the bound has a pass that fits (_proof_need) and the input's copies fit beside the models
    (_copies_need)."""
    if _proof_need(job, phase, resident, swapped) > job.budget:
        return False
    return (
        phase not in _copy_phases(job) or _copies_need(job, phase, resident, swapped) <= job.budget
    )


def _largest_input(job: _Job) -> int | None:
    """The largest input, in pixels, whose copies fit at resident 0 in every phase they are made
    in, the output's size as it is: each moment is linear in the input's pixels, so the least
    over the moments of what is left of the room after the padded frame's share, divided by the
    bytes per pixel. None where the padded frame's share alone is over the room, or the input
    would be under a pixel."""
    padded = _padded_pixels(job.bound)
    room = job.budget - _on_gpu(job, 0, 0)
    fits: int | None = None
    for phase in _copy_phases(job):
        for moment in COPIES[phase]:
            left = room - moment.per_padded * padded - moment.held
            if left < moment.per_input:
                return None
            most = left // moment.per_input
            fits = most if fits is None else min(fits, most)
    return fits


def _input_refused(job: _Job) -> PlanRefusal | None:
    """The refusal of a picture whose copies don't fit even with the DiT off the GPU (resident 0,
    where the placement ends), None when they do: the phase that is over by the most, what its
    copies take, and the largest input that fits, in megapixels and as about WxH at the bound's
    ratio, the way out being a smaller copy of the picture, or a smaller --resolution where the
    output's own part is what is over."""
    over: dict[Phase, int] = {
        phase: _copies_need(job, phase, 0, 0) - job.budget
        for phase in _copy_phases(job)
        if _copies_need(job, phase, 0, 0) > job.budget
    }
    if not over:
        return None
    phase = max(over, key=lambda each: over[each])
    padded = _padded_pixels(job.bound)
    copies = copies_peak(phase, job.input_pixels, padded)
    held = _on_gpu(job, 0, 0)
    need = held + copies + job.margin
    fits = _largest_input(job)
    if fits is None:
        way = (
            "no input fits there, the output's own copies being over the memory: a smaller"
            " --resolution"
        )
    else:
        long, short = job.bound
        wide = math.isqrt(fits * long // short)
        way = (
            f"the largest input that fits is {_megapixels(fits, up=False)} MP, about"
            f" {wide}x{fits // wide} at the bound's ratio: a copy of the picture no larger, or a"
            " smaller --resolution, which lowers the output's part of it"
        )
    which = max(COPIES[phase], key=lambda each: each.at(job.input_pixels, padded))
    return PlanRefusal(
        f"the picture's input, {_megapixels(job.input_pixels, up=True)} MP at its stored size, is"
        f" copied to the GPU and resized there in the {phase}'s phase: its copies take"
        f" {_over(copies)} GiB at the peak ({which.what}, at the largest padded frame within"
        f" {_bound_said(job.bound)}) and {_short(need, copies)} GiB more for the models on the GPU"
        f" and the margin, {_over(need)} GiB of free VRAM, and {_there(job.reading)}:"
        f" {_to_free(job.reading, need)}{way}",
        phase=phase,
        fits_input=fits,
    )


def input_copies(plan: StillPlan) -> dict[Phase, int]:
    """The picture's copies at their peak in each phase they are made in (the encode's, and the
    decode's with the colour correction), in bytes, above the models (copies_peak)."""
    padded = _padded_pixels(plan.bound)
    phases: tuple[Phase, ...] = ("encode", "decode") if plan.split else ("encode",)
    return {phase: copies_peak(phase, plan.input_pixels, padded) for phase in phases}


# ---- What a phase takes, for a reservation -----------------------------------------------------


@dataclass(frozen=True)
class Peak:
    """What a phase of one picture takes of the GPU, in bytes: its predicted allocated peak,
    `models` of it what the models have on the GPU in that phase, and the room it is held to
    beyond the peak (VAE_ROOM for a VAE pass, none for the DiT's)."""

    allocated: int
    models: int
    room: int

    @property
    def above(self) -> int:
        """What the picture itself takes: the peak above the models."""
        return self.allocated - self.models

    @property
    def reserved(self) -> int:
        """What torch's allocator is to hold for the phase: the peak, its room, and the share it
        reserves over what it allocates above the models (peaks.RESERVED_OVER)."""
        return self.allocated + self.room + math.ceil(peaks.RESERVED_OVER * self.above)


def _dit_models(plan: StillPlan) -> int:
    """What the models have on the GPU in the DiT's phase: the DiT as its placement keeps it,
    and the VAE's weights where they stay (peaks.dit_on_gpu)."""
    models = plan.models
    vae = peaks.VAE_WEIGHTS if models.vae_in_dit else 0
    return peaks.dit_on_gpu(None, models.swapped, plan.dit_model) + vae


def bound_peaks(plan: StillPlan) -> dict[Phase, Peak]:
    """The most each phase takes for a picture within the plan's bound from an input of its
    input_pixels, under its placement: the DiT's phase as the plan counts it (dit_peak), each
    VAE phase the larger of the worst picture's pass (_proof_need, its room apart) and the
    input's copies (_copies_need, the picture's frame in the decode's included)."""
    job, models = _job_of(plan), plan.models
    found: dict[Phase, Peak] = {"dit": Peak(plan.dit_peak, _dit_models(plan), 0)}
    pairs: tuple[tuple[Phase, int | None], ...] = (
        ("encode", models.resident_encode),
        ("decode", models.resident_decode),
    )
    for phase, resident in pairs:
        allocated = _proof_need(job, phase, resident, models.swapped) - VAE_ROOM
        if phase in _copy_phases(job):
            allocated = max(allocated, _copies_need(job, phase, resident, models.swapped))
        found[phase] = Peak(allocated, _on_gpu(job, resident, models.swapped), VAE_ROOM)
    return {phase: found[phase] for phase in PHASES}


def picture_peaks(
    plan: StillPlan, target: tuple[int, int], input_pixels: int, budget: int | None = None
) -> dict[Phase, Peak]:
    """What each phase takes for one picture under the plan, resized to `target` (frames.
    target_size's) from an input of `input_pixels` pixels as stored: the DiT's phase at its own
    padded frame and windows (peaks.dit_peak), each VAE phase the larger of its passes (tiles)
    and its copies at its own sizes (copies_peak), its tiles chosen within `budget` as tiles has
    it. ValueError as tiles, and for an input over the plan's input_pixels or under a pixel."""
    if not 1 <= input_pixels <= plan.input_pixels:
        raise ValueError(f"an input of {input_pixels} pixels, the plan's {plan.input_pixels}")
    found_tiles = tiles(plan, target, budget)
    job, models = _job_of(plan), plan.models
    padded = padded_size(target)
    dit = peaks.dit_peak(
        padded, 1, models.swapped, models.vae_in_dit, plan.attention, plan.dit_model
    )
    found: dict[Phase, Peak] = {"dit": Peak(dit, _dit_models(plan), 0)}
    passes: tuple[tuple[Phase, int | None, int], ...] = (
        ("encode", models.resident_encode, found_tiles.encode_peak),
        ("decode", models.resident_decode, found_tiles.decode_peak),
    )
    for phase, resident, peak in passes:
        on_gpu = _on_gpu(job, resident, models.swapped)
        if phase in _copy_phases(job):
            copies = copies_peak(phase, input_pixels, padded[0] * padded[1])
            peak = max(peak, on_gpu + copies)
        found[phase] = Peak(peak, on_gpu, VAE_ROOM)
    return {phase: found[phase] for phase in PHASES}


def _megapixels(pixels: int, *, up: bool) -> str:
    """Pixels in megapixels (10^6), one decimal, rounded up or down."""
    tenths = pixels / 1e5
    return f"{(math.ceil(tenths - 1e-6) if up else math.floor(tenths + 1e-6)) / 10:.1f}"


# ---- The system memory -------------------------------------------------------------------------


def _leaving(models: ModelPlan) -> int:
    """The DiT's blocks that leave the GPU in some phase of the plan, each then waiting in a pinned
    copy: its swapped ones, and those the VAE's phases send away (the planner's _leaving)."""
    stay = peaks.BLOCKS - models.swapped
    levels = (_level(each, models.swapped) for each in _both(models))
    return peaks.BLOCKS - min(stay, *levels)


def _both(models: ModelPlan) -> tuple[int | None, int | None]:
    return models.resident_encode, models.resident_decode


def _pinned(models: ModelPlan, dit_model: Architecture) -> int:
    """The pinned system memory of the plan, in bytes: a copy of each part of the weights that
    leaves the GPU in some phase, of its own size rounded up to a page (blockswap.HostCopy), as
    the planner's _pinned counts them: each block of the DiT that is swapped or that a VAE phase
    doesn't keep; the DiT's other parameters when a VAE phase keeps nothing of it; the VAE's
    weights when they leave the DiT's phase; the DiT in the format of its file."""
    parts = [peaks.block_bytes(dit_model)] * _leaving(models)
    parts += [peaks.other_bytes(dit_model)] if 0 in _both(models) else []
    parts += [] if models.vae_in_dit else [peaks.VAE_WEIGHTS]
    return sum(-(-part // mmap.PAGESIZE) * mmap.PAGESIZE for part in parts)


def _account(job: _Job, models: ModelPlan) -> StillPlan:
    """The plan for the placement, with its system memory: the process's own, the pinned copies,
    and the scratch of a tiled pass of the largest picture, one frame (peaks.scratch), where any
    picture within the bound may be tiled. One may be where the bound's own picture, which takes
    the most of the GPU in either pass, is tiled under the placement: a smaller one that is has
    the bound's tiles or smaller ones, and a pin tiles the pictures it doesn't cover."""
    made = StillPlan(
        job.bound,
        job.input_pixels,
        models,
        job.reading,
        job.attention,
        job.dit_model,
        job.split,
        job.pins,
        job.process,
        _dit_need(job, models.swapped, models.vae_in_dit),
        windows_most(job.bound),
        _pinned(models, job.dit_model),
        0,
        job.host_available,
    )
    output = (job.bound[1], job.bound[0])
    tiled = any(
        found is not None and found[0] is not None
        for frame in _frames(job.bound)
        for found in (
            _first(job, "encode", frame, output, models.swapped, models.resident_encode),
            _first(job, "decode", frame, output, models.swapped, models.resident_decode),
        )
    )
    scratch = max(peaks.scratch(frame, FRAMES) for frame in _frames(job.bound)) if tiled else 0
    return replace(made, host=job.process + made.pinned + scratch)


def _vae_kept(job: _Job, made: StillPlan) -> StillPlan | None:
    """The plan with the VAE's weights left on the GPU in the DiT's phase and as many more blocks
    swapped as that takes, the fewest, where the plan as made fits the GPU's memory and not the
    system's; None where it doesn't apply: the VAE's weights are there already, --swap-blocks is
    pinned (taken exactly), or no count of swapped blocks holds the DiT's phase with them. The
    planner's _vae_kept: the VAE's weights away are 0.47 GiB more of pinned memory, and a swapped
    block costs none where a VAE phase sends it away anyway."""
    if made.models.vae_in_dit or job.pins.swap_blocks is not None:
        return None
    counts = range(max(made.models.swapped, peaks.SLOTS) + 1, peaks.BLOCKS + 1)
    swapped = next(
        (count for count in counts if _dit_need(job, count, vae_in_dit=True) <= job.budget), None
    )
    if swapped is None:
        return None
    return _account(job, _place(job, swapped, vae_in_dit=True))


def _dit_kept(job: _Job, made: StillPlan) -> StillPlan | None:
    """The plan with more of the DiT kept on the GPU in the VAE's phases, one block more in both
    at a time (a block fewer pinned), as long as every picture within the bound still has a pass
    that fits (_proof_need), until the plan fits the host; where none does, the last, which pins
    the least, for the refusal's figures. None where no block more can be kept. The planner's
    _dit_kept shrinks the tiles of the one frame size of its job to keep them; a bound's tiles
    are chosen per picture, so here the placement is what rises, to the most the proof of the
    smallest tiles (or the pin's) allows."""
    models, kept = made.models, None
    fixed = peaks.BLOCKS - models.swapped
    while True:
        keep = peaks.BLOCKS - _leaving(models) + 1
        if keep > fixed:
            return kept
        encode, decode = (
            resident if _level(resident, models.swapped) >= keep else keep
            for resident in (models.resident_encode, models.resident_decode)
        )
        if not (
            _holds(job, "encode", encode, models.swapped)
            and _holds(job, "decode", decode, models.swapped)
        ):
            return kept
        models = replace(models, resident_encode=encode, resident_decode=decode)
        kept = _account(job, models)
        if job.host_available is None or kept.host <= job.host_available:
            return kept


def _host_hundredths(made: StillPlan) -> tuple[int, int, int]:
    """What the plan takes of system memory, in hundredths of a GiB to the nearest: the process's
    own, the pinned copies, and the scratch; the total either says is their sum, so that its
    figures add up (the planner's _host_hundredths)."""
    scratch = made.host - made.process - made.pinned
    return _nearest(made.process), _nearest(made.pinned), _nearest(scratch)


def _unfit(job: _Job, made: StillPlan) -> PlanRefusal:
    """The refusal for the system memory, each part named with what it takes where that is a
    hundredth of a GiB or more, the total their sum (the planner's _unfit)."""
    assert job.host_available is not None
    process, pinned, scratch = _host_hundredths(made)
    taken = [(process, "for the process")]
    if pinned:
        left = _left(made.models, "the DiT's other parameters", "the VAE's weights")
        taken.append((pinned, f"pinned for {left} that leave the GPU"))
    if scratch:
        taken.append((scratch, "for a tiled pass's scratch, one frame"))
    total = process + pinned + scratch
    # What is there is rounded down, and said a hundredth under the total at least.
    there = min(_hundredths(job.host_available, up=False), total - 1)
    said = ", ".join(f"{count / 100:.2f} {what}" for count, what in taken)
    return PlanRefusal(
        f"the plan takes {total / 100:.2f} GiB of system memory ({said}) and"
        f" {there / 100:.2f} GiB are available, {(total - there) / 100:.2f} GiB short",
        phase="host",
    )


def _planned(job: _Job) -> StillPlan:
    """plan's work, once the pins that need no memory are checked: the floor, the DiT, the pins'
    proofs (the decode's then the encode's, the planner's order), the placement, the system
    memory; PlanRefusal as plan says."""
    floor = _floor(job)
    if floor is not None:
        raise _too_large(job, *floor)
    refused = _input_refused(job)
    if refused is not None:
        raise refused
    swapped, vae_in_dit = _swap_choice(job)
    for phase in ("decode", "encode"):
        pin_refused = _pin_refused(job, phase)
        if pin_refused is not None:
            raise pin_refused
    made = _account(job, _place(job, swapped, vae_in_dit))
    if job.host_available is None or made.host <= job.host_available:
        return made
    # The planner's order (_made): the VAE kept with more blocks swapped, when it fits the host or
    # pins less; then the DiT kept from there and from the plan as made, the first that fits the
    # host, else the one that takes the least.
    ordered = made
    spared = _vae_kept(job, made)
    if spared is not None and (spared.host <= job.host_available or spared.host < made.host):
        made = spared
    for start in dict.fromkeys((made, ordered)):
        kept = _dit_kept(job, start)
        if kept is not None and kept.host < made.host:
            made = kept
        if made.host <= job.host_available:
            return made
    raise _unfit(job, made)


# ---- The plan said -----------------------------------------------------------------------------

# The attention backends as the plan's lines name them (model.load_models' own words for SDPA).
_ATTENTION_SAID = {peaks.FLASH_ATTN_2: "FlashAttention 2", peaks.SDPA: "PyTorch's SDPA"}


def said(plan: StillPlan, target: tuple[int, int] | None = None) -> list[str]:
    """The plan as --plan prints it and a run logs it, a line per choice in plain words, each
    opening with what it says, "plan: memory: ...", as the planner's lines do (planner.said):
    the memory (what was read, the margin or the reserve held, the budget, the tight reserve when
    taken), the picture's size, the attention backend and the DiT's format, the blocks swapped and
    where the VAE is in the DiT's phase with the DiT's peak, what of the DiT stays on the GPU in
    each VAE phase, the system memory; and with `target` (frames.target_size's), the picture's
    encode and decode, untiled or in tiles with the grid and the predicted peaks."""
    frame = _most(plan.bound)
    height, width = frame
    lines = _memory_said(plan)
    lines.append(
        f"bound: pictures within {_bound_said(plan.bound)} in either orientation, padded to at"
        f" most {width}x{height} for the model, {peaks.tokens(frame, 1):,} tokens, one latent"
    )
    if plan.attention == peaks.FLASH_ATTN_2:
        lines.append(f"attention: {_ATTENTION_SAID[plan.attention]}, which is installed")
    else:
        lines.append(
            f"attention: {_ATTENTION_SAID[plan.attention]}, FlashAttention 2 not being"
            " installed: a window takes more memory under it"
        )
    if plan.dit_model == peaks.INT8:
        lines.append(
            f"DiT: its weights in int8, {_gib(peaks.weights_bytes(peaks.INT8))} GiB, half of"
            f" fp16's; a window's memory counted as fp16's plus {_gib(peaks.INT8_BUFFERS)} GiB for"
            " int8's buffers until int8's own is measured; --dit-model names the file"
        )
    models = plan.models
    if plan.pins.swap_blocks is not None:
        why = "pinned by --swap-blocks"
    elif not models.swapped:
        why = "the DiT fitting whole on the GPU"
    elif models.vae_in_dit:
        why = "more than the DiT needs: the VAE stays on the GPU, which spares system memory"
    else:
        why = "the fewest that hold the DiT"
    if models.swapped:
        lines.append(
            f"blocks swapped: {models.swapped} of the DiT's {peaks.BLOCKS}, {why}: slower, not"
            " worse; --swap-blocks N"
        )
    else:
        lines.append(f"blocks swapped: none, {why}; --swap-blocks N swaps some")
    if models.vae_in_dit:
        lines.append("the VAE: on the GPU in the DiT's phase, the DiT fitting beside it")
    else:
        lines.append(
            f"the VAE: off the GPU in the DiT's phase, its {_gib(peaks.VAE_WEIGHTS)} GiB given to"
            " the DiT"
        )
    lines.append(
        f"the DiT's phase: {_gib(plan.dit_peak)} GiB at the most for a picture within the bound,"
        f" one latent, the tokens of the largest and the most windows any takes,"
        f" {plan.windows}, {_gib(plan.budget - plan.dit_peak)} GiB under the budget"
    )
    lines.append(
        "the DiT in the VAE's passes: which model stays on the GPU in which phase, as much of the"
        f" DiT as fits beside the worst picture's pass: in the encode,"
        f" {_resident_said(models.resident_encode, models.swapped)}; in the decode,"
        f" {_resident_said(models.resident_decode, models.swapped)}"
    )
    lines.append(_copies_said(plan))
    process, pinned, scratch = _host_hundredths(plan)
    left = _left(models, "the DiT's other parameters", "the VAE")
    copies = f"{pinned / 100:.2f} GiB pinned for {left}" if left else "nothing pinned"
    scratch_said = (
        f"; {scratch / 100:.2f} GiB for the scratch of a tiled pass of one frame" if scratch else ""
    )
    lines.append(
        f"system memory: {(process + pinned + scratch) / 100:.2f} GiB in all; {process / 100:.2f}"
        f" GiB for the process; {copies}{scratch_said}"
    )
    if target is not None:
        found = tiles(plan, target)
        padded, (out_height, out_width) = padded_size(target), output_size(target)
        lines.append(f"picture: {out_width}x{out_height}, padded to {padded[1]}x{padded[0]}")
        for phase, chosen, peak, pin in (
            ("encode", found.encode, found.encode_peak, plan.pins.encode_tile),
            ("decode", found.decode, found.decode_peak, plan.pins.decode_tile),
        ):
            lines.append(_pass_said(phase, chosen, peak, pin, padded, plan))
        lines.append(
            "the VAE's room: what each VAE pass leaves free, for the large pieces of memory a"
            f" pass asks for at once: {_gib(VAE_ROOM)} GiB of the budget, beside each peak"
        )
    return [f"plan: {line}" for line in lines]


def _copies_said(plan: StillPlan) -> str:
    """The plan's line of the picture's copies (said): the input's size, and for each phase it is
    resized in the copies at their peak, by the measured rule (COPIES), and what
    is left under the budget at the placement's residents (_copies_need)."""
    job, models = _job_of(plan), plan.models
    figures: list[str] = []
    for phase, resident in (("encode", models.resident_encode), ("decode", models.resident_decode)):
        if phase not in input_copies(plan):
            continue
        copies = input_copies(plan)[phase]
        room = plan.budget - _copies_need(job, phase, resident, models.swapped)
        figures.append(f"{_gib(copies)} GiB in the {phase} ({_gib(room)} GiB under the budget)")
    return (
        f"the input: {_megapixels(plan.input_pixels, up=True)} MP at its stored size, copied to"
        " the GPU and resized there, at the bound's largest padded frame, its copies"
        f" {', '.join(figures)}; by the rule measured on the RTX 5090"
    )


def _memory_said(plan: StillPlan) -> list[str]:
    """The plan's first lines (said): the free memory and the budget, with the margin and the
    reserve by their names and figures, as the planner's _card_said has them without the card's
    name. On a driver that refuses, one line: what it reports free, the budget that less the
    margin (MARGIN), and no reserve. On a run held to the card (memory.Reading.held), two: the
    free memory the run is held under, with the other figure (_named); then the budget, what the
    plan counts on; where the plan took memory.RESERVE_TIGHT, the plan with memory.RESERVE
    refused, a line more."""
    reading = plan.reading
    if not reading.held:
        return [
            f"memory: {_gib(plan.reading.counted)} GiB free, a budget of {_gib(plan.budget)} GiB:"
            f" the margin of {_gib(MARGIN)} GiB left to the desktop and other programs; no"
            " reserve, the driver refusing memory the card doesn't have"
        ]
    free = _held_under(reading)
    lines = [
        f"memory: {_gib(free)} GiB free {_named(reading, _gib, driver=False)}: the run is held"
        " to the card's own memory, its driver giving more than the card has",
        f"budget: {_gib(plan.budget)} GiB, the {_gib(free)} GiB free less torch's limit,"
        f" {_gib(reading.left)} GiB under it, and the reserve of"
        f" {_gib(free - reading.left - reading.counted)} GiB left to torch's allocator beyond a"
        f" step's peak; the margin of {_gib(MARGIN)} GiB is in the reserve",
    ]
    if plan.tightened:
        tight, usual = _gib(memory.RESERVE_TIGHT), _gib(memory.RESERVE)
        lines.append(
            f"reserve: {tight} GiB and not {usual}, the plan with {usual} having been refused"
            " (--verbose says why); slow: a step short of room beyond its peak slows down where"
            f" it would be refused, and a picture that fits with {usual} runs up to 1.5 times as"
            f" long with {tight}; memory freed on the card gives the {usual} back"
        )
    return lines


def _pass_said(
    phase: str,
    chosen: Tiles | None,
    peak: int,
    pin: int | None,
    padded: tuple[int, int],
    plan: StillPlan,
) -> str:
    """The plan's line of a picture's VAE pass (said): untiled, or its tiles, their overlap and
    grid; why: pinned, the whole frame fitting, or the largest tiles that fit under the plan's
    placement (the planner's _vae_pass_said); the predicted peak; --encode-tile or
    --decode-tile."""
    option = f"--{phase}-tile"
    if pin is not None:
        why = f"pinned by {option}" if chosen is not None or pin == 0 else "the pin covering it"
    elif chosen is None:
        why = "the whole frame fitting the budget"
    else:
        why = (
            "the whole frame not fitting the budget, the largest tiles that do, up to"
            f" {peaks.TILE_MAX} px"
        )
    peak_said = f"peak {_gib(peak)} GiB, {_gib(plan.budget - peak - VAE_ROOM)} GiB under the budget"
    if chosen is None:
        return f"{phase}: untiled, {why}; {peak_said}; {option} PX tiles it"
    rows, columns = peaks.grid_shapes(padded, chosen)
    return (
        f"{phase}: tiles of {chosen.size} px overlapping by {chosen.overlap}, {columns} across,"
        f" {rows} down, {why}; {peak_said}; {option} PX"
    )


def _resident_said(resident: int | None, swapped: int) -> str:
    """What of the DiT a VAE phase has on the GPU, in words, as peaks.dit_on_gpu counts it: nothing;
    a number of its blocks, with its other parameters; or, nothing moving, what its own phase has,
    the blocks that aren't swapped and the slots (the planner's _resident_said)."""
    stay = peaks.BLOCKS - swapped
    if resident == 0:
        return "the DiT off the GPU"
    if resident is not None:
        return f"{resident} of the DiT's blocks on the GPU"
    if not swapped:
        return f"the DiT's {peaks.BLOCKS} blocks on the GPU"
    slots = _count(min(swapped, peaks.SLOTS), "slot")
    if not stay:
        return f"the DiT's {slots} on the GPU"
    return f"the DiT's {_count(stay, 'unswapped block')} and its {slots} on the GPU"


def _left(models: ModelPlan, other: str, vae: str) -> str:
    """What of the weights leaves the GPU in some phase of the plan, pinned, in words: the DiT's
    blocks counted, its other parameters and the VAE named as given; empty when nothing leaves."""
    leaving = _leaving(models)
    left = [_count(leaving, "block")] if leaving else []
    left += [other] if 0 in _both(models) else []
    left += [] if models.vae_in_dit else [vae]
    return " and ".join(filter(None, (", ".join(left[:-1]), *left[-1:])))


def _bound_said(bound: Bound) -> str:
    long, short = bound
    return f"{long}x{short}"


# ---- Words, the planner's (planner.py), for a reading without the early look at the card -------


def _held_under(reading: Reading) -> int:
    """The free memory a held run is held under, in bytes: the smaller of what its driver reports
    free to the process and of what the whole card has free; the driver's figure where the card's
    wasn't read (the planner's _held_under)."""
    if reading.card is None:
        return reading.reported
    return min(reading.reported, reading.card)


def _named(reading: Reading, said_: Callable[[float], str], driver: bool = True) -> str:
    """Which of a held reading's figures is the free memory, in words for a line that has just
    given it as "N GiB free", with the other figure (the planner's _named)."""
    if reading.card is None:
        return "as its driver reports it to this process (the whole card's, NVML's, wasn't read)"
    if reading.card <= reading.reported:
        if not driver:
            return "on the whole card by NVML"
        reported = said_(reading.reported)
        return f"on the whole card by NVML (its driver reports {reported} GiB free to this process)"
    return (
        "as its driver reports it to this process (the whole card has"
        f" {said_(reading.card)} GiB free by NVML)"
    )


def _there(reading: Reading) -> str:
    """What is there for the plan, as a refusal says it after what something takes: on a driver
    that refuses, what it reports free; on a run held to the card, what the plan counts on, of
    which figure, with the other (the planner's _there). Rounded down."""
    if not reading.held:
        return f"the driver reports {_under(reading.counted)} GiB free"
    return (
        f"the plan counts on {_under(reading.counted)} GiB of the"
        f" {_under(_held_under(reading))} GiB free {_named(reading, _under)}, the run held to the"
        " card's own memory"
    )


def _to_free(reading: Reading, need: float) -> str:
    """What a held run's refusal says before what fits: the memory that would have to be freed on
    the card for `need` bytes of free VRAM, by the message's own figures (_short); where no
    memory freed on the card gives it (_freeable), what it is short of. Nothing on a driver that
    refuses (the planner's _to_free)."""
    if not reading.held:
        return ""
    if not _freeable(reading, need):
        return f"{_short(need, reading.counted)} GiB short; "
    return f"{_short(need, reading.counted)} GiB more would have to be freed on the card for it; "


def _freeable(reading: Reading, need: float) -> bool:
    """Whether memory freed on the card, other programs closed, can give a held run's plan the
    `need` bytes of free VRAM: only where the whole card's figure is the one the plan was made
    under, the smaller of the two, and only up to the driver's (the planner's _freeable)."""
    if not reading.held or reading.card is None or reading.card >= reading.reported:
        return False
    return need - reading.counted <= reading.reported - reading.card


def _count(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _gib(count: float) -> str:
    """Bytes in GiB, to 2 decimals."""
    return f"{count / peaks.GIB:.2f}"


def _nearest(count: float) -> int:
    """Bytes in hundredths of a GiB, to the nearest, as _gib says them."""
    return round(float(_gib(count)) * 100)


def _over(count: float) -> str:
    """Bytes in GiB, to 2 decimals, rounded up: what something takes of the GPU's memory."""
    return f"{_hundredths(count, up=True) / 100:.2f}"


def _under(count: float) -> str:
    """Bytes in GiB, to 2 decimals, rounded down: what is there, so that what takes more than is
    there never reads as taking the same."""
    return f"{_hundredths(count, up=False) / 100:.2f}"


def _short(need: float, there: float) -> str:
    """What `need` bytes are short of in `there`, in GiB to 2 decimals: the difference of the two as
    _over and _under say them, so that a message's figures add up."""
    return f"{(_hundredths(need, up=True) - _hundredths(there, up=False)) / 100:.2f}"


def _hundredths(count: float, *, up: bool) -> int:
    """Bytes in hundredths of a GiB, rounded up or down, a figure that is one already kept."""
    exact = count / peaks.GIB * 100
    return math.ceil(exact - 1e-6) if up else math.floor(exact + 1e-6)

"""The server's reservation (DESIGN.md, The image server, Its start): what torch's allocator holds
on the GPU for the process, made once at the start and capped there: the models, --max-concurrent
pictures at --max-size from inputs of --max-input, by the library's plan for one picture
(still_plan) and the figures it gives each phase on reserved memory (still_plan.Peak.reserved).
Plain Python, no torch: the start plans before torch is imported, and the tests hold it alone.

Each picture's passes are held to one budget, what the bound's own pictures take under the plan
(picture_budget): the plan's tiles are the largest that fit its budget, so a reservation counted
on the plan's floor would be short, and one counted on its budget would take the whole card
whatever the bound (the review, 2026-11-03).

One picture at a time may swap blocks, as the plan chooses. Several at once need the whole DiT on
the card: the pictures share the models, which a placement that moves them per phase can't serve
to two pictures in two phases. The plan, made for one picture, lowers the DiT beside a VAE pass
before it tiles that pass smaller (still_plan._place); several pictures take the VAE's passes in
tiles pinned smaller instead, the largest that keep the DiT whole with a reservation in the memory
counted (_made; an RTX 5090 under WSL2, 2026-11-04: two pictures within 1920x1080 on the int8 DiT
were refused where tiles of 1088 px fit). The server runs one picture in each model at a time
(service.py: the VAE keeps its causal memory in its modules), so several pictures at once take the
models, one picture's share in the VAE and one in the DiT, whatever --max-concurrent: the pictures
past two wait on the CPU, between the models."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cache

from seedvr2x import frames, peaks, pull, still_plan
from seedvr2x.errors import Refusal, Setting
from seedvr2x.memory import Reading
from seedvr2x.passes import Phase
from seedvr2x.still_plan import Bound, Peak, Pins, StillPlan
from seedvr2x.weights import Architecture

GIB = peaks.GIB

# A video job's process in system memory, the CLI's provisional figure for a still's too
# (apps/seedvr2x/planner.py HOST_RESIDENT; DESIGN.md, Still images: "the server's is a parameter
# of the plan"), until the card's batch measures a still's: the server's own, with no job.
PROCESS = round(2.4 * GIB)
# What a picture holds in system memory, in bytes a pixel: read, its float32 pixels (12 a stored
# pixel); running, those and its output, float32, twice (the decode's and the turned view's copy,
# 24 an output pixel), and its file being written, 16-bit, three times at the most (the samples,
# the filtered rows, the compressed bytes: 18), as the CLI counts a still (stills._process). A
# picture waiting holds its body too, at most the body limit (body_limit).
INPUT_BYTES = 12
OUTPUT_BYTES = 24 + 18
# The most bytes a picture's file can take for its pixels: 16-bit RGBA uncompressed, the largest
# any format read stores (a PNG of stored blocks or a TIFF without compression), and the room for
# its metadata (an ICC profile, EXIF, XMP), which no format bounds.
BODY_BYTES = 8
BODY_EXTRA = 16 * 1024**2
# The tries at a plan that leaves the reserved share beside each phase (_made): each lowers what
# the plan counts on by what the last one was short over the reservation's rate, and a MiB more,
# so that the tries end; past them the start is refused (tests/test_reservation.py counts the
# tries its cases take: five at the most over 2 to 100 GiB free and 1 to 4 pictures).
TRIES = 8


def body_limit(input_pixels: int) -> int:
    """The largest body /upscale reads, in bytes, for pictures of `input_pixels` at the most:
    a larger one is answered 413 before it is read (BODY_BYTES, BODY_EXTRA)."""
    return BODY_BYTES * input_pixels + BODY_EXTRA


def host_process(count: int, queue: int, input_pixels: int, bound: Bound) -> int:
    """The server's system memory beside the models' pinned copies, in bytes, the plan's
    `process`: its own (PROCESS), `count` pictures running from inputs of `input_pixels` to
    outputs of the bound, and `queue` waiting, each with its pixels and its body."""
    output = bound[0] * bound[1]
    running = count * (INPUT_BYTES * input_pixels + OUTPUT_BYTES * output)
    waiting = queue * (INPUT_BYTES * input_pixels + body_limit(input_pixels))
    return PROCESS + running + waiting


def share(peaks_of: Mapping[Phase, Peak]) -> int:
    """A picture's share of the reservation above the models, in bytes: its largest phase's
    reserved figure less what the models have on the GPU in it."""
    return max(peak.reserved - peak.models for peak in peaks_of.values())


def picture_budget(plan: StillPlan, input_pixels: int) -> int:
    """What a picture's passes may count on, in bytes, models included (still_plan.tiles'
    `budget`): the most the bound's own pictures, both orientations, from inputs of
    `input_pixels`, take in a VAE phase with the plan's tiles, their room with it, and at least the
    plan's proof's worst pass with its room (still_plan.bound_peaks), so that every picture within
    the bound has a pass. Never over the plan's budget."""
    long, short = plan.bound
    most = 0
    for phase, peak in still_plan.bound_peaks(plan).items():
        if phase != "dit":
            most = max(most, peak.allocated + peak.room)
    for target in ((short, long), (long, short)):
        for phase, peak in still_plan.picture_peaks(plan, target, input_pixels).items():
            if phase != "dit":
                most = max(most, peak.allocated + peak.room)
    return min(most, plan.budget)


def ceilings(plan: StillPlan, budget: int) -> dict[Phase, Peak]:
    """The most each phase of any picture within the plan's bound takes, under its placement, its
    passes held to `budget` (picture_budget): the DiT's phase the plan's (dit_peak, the most over
    the bound's pictures), each VAE phase the budget less its room."""
    found = still_plan.bound_peaks(plan)
    return {
        phase: peak
        if phase == "dit"
        else Peak(max(peak.allocated, budget - peak.room), peak.models, peak.room)
        for phase, peak in found.items()
    }


def reserved(peaks_of: Mapping[Phase, Peak], count: int) -> int:
    """The reservation for `count` pictures at once, in bytes, each at `peaks_of` (ceilings):
    one picture, its largest phase's reserved figure, the models as the plan places them in each
    phase; several, the models, all on the card, one picture's share in the VAE (its larger
    phase) and one in the DiT, the server running one picture in each model at a time."""
    if count < 1:
        raise ValueError(f"{count} pictures at once")
    if count == 1:
        return max(peak.reserved for peak in peaks_of.values())
    vae = max(peaks_of[phase].reserved - peaks_of[phase].models for phase in ("encode", "decode"))
    dit = peaks_of["dit"].reserved - peaks_of["dit"].models
    return peaks_of["dit"].models + vae + dit


def whole(plan: StillPlan) -> bool:
    """Whether the plan keeps every model on the card in every phase: no block swapped, the VAE
    in the DiT's phase, the DiT whole in the VAE's."""
    models = plan.models
    return (
        models.swapped == 0
        and models.vae_in_dit
        and models.resident_encode is None
        and models.resident_decode is None
    )


@dataclass(frozen=True)
class Made:
    """The reservation made for a start: its plan, the pictures at once, what torch is held to
    (`reservation`, bytes), what each picture's passes may count on (`budget`, picture_budget),
    the admission's pool (`pool`: `count` shares) and the share of the bound's own picture
    (`share`, the most any picture's is)."""

    plan: StillPlan
    count: int
    reservation: int
    budget: int
    pool: int
    share: int

    @property
    def models(self) -> int:
        """What the models take of the reservation with several pictures at once, all on the
        card, in bytes: the DiT's phase's models."""
        return still_plan.bound_peaks(self.plan)["dit"].models


class ReservationRefused(Refusal):
    """The start refused: the reservation doesn't fit the card's free memory, or no plan does. The
    message names the most pictures at --max-size and the largest size at --max-concurrent that
    would fit (`most`, `largest`: None where none would)."""

    def __init__(self, message: str, most: int, largest: Bound | None) -> None:
        super().__init__(message)
        self.most = most
        self.largest = largest


@dataclass(frozen=True)
class Asked:
    """What the start asks of the plan: the bound (long, short) of --max-size, the largest input
    in pixels, the pictures at once and waiting, the reading of the card and the host's memory
    available (None: unread), the attention kernel and the DiT's format, and the pins (one
    picture at a time only); `dit_file`, the DiT file's name ("" says unknown), whose smaller
    files a refused start names where they would fit (make); `model_dir`, whether the server was
    given --model-dir, which that way out then says (smaller_way)."""

    bound: Bound
    input_pixels: int
    count: int
    queue: int
    reading: Reading
    host_available: int | None
    attention: str
    dit_model: Architecture
    pins: Pins = still_plan.NO_PINS
    dit_file: str = ""
    model_dir: bool = False


class _Moves(ReservationRefused):
    """Several pictures at once refused for a plan that moves the models (_tried)."""


@cache
def tile_sides(bound: Bound) -> range:
    """The tiles' sides the pictures at once are pinned to, the largest first: still_plan's own
    sizes for the bound (its _tile_sizes at the bound's largest padded frame), multiples of
    TILE_STEP from the largest under the longest padded side of any picture within the bound, or
    from TILE_MAX, down to TILE_MIN. A tile covering a smaller picture's frame gives that picture
    the untiled pass (still_plan._options), which the plan's proof counts."""
    long = bound[0]
    sides = range(1, long + 2)
    within = [side for side in sides if side - side % 2 <= long]
    padded = max(
        max(frames.padded_size((side, frames.MULTIPLE))[0] for side in within),
        max(frames.padded_size((2 * frames.MULTIPLE, side))[1] for side in within),
    )
    largest = min(peaks.TILE_MAX, (padded - 1) // peaks.TILE_STEP * peaks.TILE_STEP)
    return range(largest, peaks.TILE_MIN - 1, -peaks.TILE_STEP)


def _several(bound: Bound) -> list[Pins]:
    """The pins several pictures at once within `bound` are tried with, in order: no block
    swapped and the plan's own tiles, then both VAE phases pinned to each of tile_sides."""
    tiled = [Pins(side, side, swap_blocks=0) for side in tile_sides(bound)]
    return [Pins(swap_blocks=0), *tiled]


def _plan(asked: Asked, bound: Bound, count: int, reading: Reading, pins: Pins) -> StillPlan:
    return still_plan.plan(
        bound,
        reading,
        asked.host_available,
        process=host_process(count, asked.queue, asked.input_pixels, bound),
        input_pixels=asked.input_pixels,
        attention=asked.attention,
        dit_model=asked.dit_model,
        split=True,
        pins=pins,
    )


def _made(asked: Asked, bound: Bound, count: int) -> Made:
    """The reservation for `count` pictures within `bound`, tried with each of _pins in turn
    (_tried), the first that fits taken: for one picture the asked pins alone; for several, the
    plan's own tiles first, then the VAE's passes in tiles pinned smaller until a plan keeps the
    DiT whole with a reservation that fits. The plan places what of the DiT stays beside a VAE
    pass for one picture, lowering the DiT before it tiles smaller (still_plan._place), which
    several pictures can't take: the int8 DiT's 8.25 GiB within 1920x1080 on 18.45 GiB counted
    kept the DiT whole at the plan's own tiles but over the reservation, and lowered it on the
    memory lowered (the RTX 5090, 2026-11-04), where both passes in tiles of 1088 px keep it whole
    and reserve 18.20. still_plan.PlanRefusal where no plan fits one picture; ReservationRefused
    (with no counts) where the reservation doesn't."""
    if count == 1:
        return _tried(asked, bound, count, asked.pins)
    for pins in _several(bound):
        try:
            return _tried(asked, bound, count, pins)
        except ReservationRefused:
            # The plan moves the models (_Moves), or its reservation doesn't fit: smaller tiles
            # next.
            continue
    raise ReservationRefused(
        f"{count} pictures at once need the whole DiT on the card, every block of it in every"
        " phase and the VAE beside it, and no plan for pictures within"
        f" {_bound_said(bound)} keeps it there with their reservation in the memory the plan"
        f" counts on, the VAE's passes tiled down to {peaks.TILE_MIN} px",
        0,
        None,
    )


def _tried(asked: Asked, bound: Bound, count: int, pins: Pins) -> Made:
    """The reservation for `count` pictures within `bound` under `pins`: the plan made for one
    picture, then made again on less memory while `count` pictures at its ceilings, with the share
    torch reserves beside each phase, take more than the reading's budget, so that the plan swaps
    more blocks or tiles smaller; each try lowers the memory one picture counts on by what the last
    was short over the reservation's rate. still_plan.PlanRefusal where no plan fits one picture;
    _Moves where several pictures' plan moves the models or is refused; ReservationRefused (with
    no counts) where the reservation doesn't fit."""
    reading, short = asked.reading, 0
    for _ in range(TRIES):
        try:
            made = _plan(asked, bound, count, reading, pins)
        except still_plan.PlanRefusal:
            if count == 1:
                raise
            # The plan's own refusal names the pins it was made with, --swap-blocks 0 and the
            # tiles tried, which the server takes from no user at several pictures at once
            # (main.main): a pin it refuses is one the next tries smaller.
            made = None
        if made is None or (count > 1 and not whole(made)):
            raise _Moves(f"{count} pictures at once: the plan moves the models", 0, None)
        budget_of = picture_budget(made, asked.input_pixels)
        peaks_of = ceilings(made, budget_of)
        reservation = reserved(peaks_of, count)
        # The budget the original reading gives the plan's reserve: the tight reserve, taken by
        # the plan where the usual one is refused, counts on more.
        budget = made.budget + short
        if reservation <= budget:
            # The admission bounds the pictures in flight, `count` of them, each within the
            # share; the GPU holds two at the most, one in each model (reserved).
            each = share(peaks_of)
            return Made(made, count, reservation, budget_of, count * each, each)
        # What the plan counts on next: under what the phase that binds takes now, so that the
        # plan chooses less there (smaller tiles, or more blocks swapped): the budget a picture
        # takes moves in steps, the tiles' sides, and a reading lowered by less than a step would
        # change nothing; and at least the shortfall over the rate the reservation grows at with
        # the budget, 1 + RESERVED_OVER at the most (one picture in the VAE whatever the count).
        # Several pictures at once keep the DiT whole, so only their VAE share is lowered.
        rate = 1 + peaks.RESERVED_OVER
        dit = peaks_of["dit"]
        vae = max(peaks_of[phase].reserved for phase in ("encode", "decode"))
        binding = made.dit_peak if count == 1 and dit.reserved >= vae else budget_of
        lowered = min(
            made.budget - math.ceil((reservation - budget) / rate),
            binding - peaks.MIB,
        )
        short = asked.reading.counted - made.margin - lowered
        reading = replace(reading, counted=asked.reading.counted - short)
        if reading.counted <= 0:
            break
    raise ReservationRefused(
        f"the reservation for {_count(count)} within {_bound_said(bound)} doesn't fit the memory"
        " the plan counts on, with the share torch's allocator reserves over each phase's peak",
        0,
        None,
    )


def fits(asked: Asked, bound: Bound, count: int) -> bool:
    """Whether a reservation for `count` pictures within `bound` fits."""
    try:
        _made(asked, bound, count)
    except Refusal:
        return False
    return True


def most(asked: Asked) -> int:
    """The most pictures at once within the asked bound whose reservation fits, up to the asked
    count; 0 where not one does."""
    return next((n for n in range(asked.count, 0, -1) if fits(asked, asked.bound, n)), 0)


def largest(asked: Asked) -> Bound | None:
    """The largest bound at the asked bound's ratio whose reservation fits the asked count,
    searched as still_plan's floor refusal searches its own (the short side down in steps of 16
    from the multiple of 16 under it, the long side in proportion, to an even number), bisected:
    a smaller bound never takes more. None where none does."""
    long, short = asked.bound
    candidates: list[Bound] = []
    for side in range((short - 1) // 16 * 16, 0, -16):
        smaller = (side * long // short // 2 * 2, side)
        if side < 2 * frames.MULTIPLE:
            break  # too small for the padding's 17 rows (frames.check_target)
        candidates.append(smaller)
    low, high = 0, len(candidates)
    while low < high:
        middle = (low + high) // 2
        if fits(asked, candidates[middle], asked.count):
            high = middle
        else:
            low = middle + 1
    return candidates[low] if low < len(candidates) else None


def make(asked: Asked) -> Made:
    """The reservation for the start: ReservationRefused where it doesn't fit, its message saying
    why and naming the most pictures at the asked size, and the largest size at the asked count,
    that would; and the smaller DiT files that would fit the asked bound and count as they are
    (smaller; DESIGN.md, Weights, "The dynamic GGUF before v0.1.0 too": the file is never changed
    by itself)."""
    try:
        return _made(asked, asked.bound, asked.count)
    except Refusal as error:
        refused = error
    at_most = most(asked)
    found = largest(asked)
    ways: list[str] = []
    if at_most and at_most < asked.count:
        ways.append(
            f"{_count(at_most)} at once within {_bound_said(asked.bound)} (--max-concurrent)"
        )
    if found is not None:
        ways.append(f"{_count(asked.count)} at once within {_bound_said(found)} (--max-size)")
    way = (
        "; what would fit: " + ", or ".join(ways)
        if ways
        else "; no picture fits there at any size the plan takes"
    )
    raise ReservationRefused(
        f"{refused.words(option)}{way}{smaller(asked)}", at_most, found
    ) from None


def option(setting: Setting) -> str:
    """The server's option for a setting a library refusal names as its way out (errors.Setting):
    "--" and the setting's name with its underscores as hyphens, then its value, as the command
    line words them. The server has --encode-tile, --decode-tile, --swap-blocks and --dit-model
    of its own; the library's plan refusals name two settings it has no option for, the window
    (a video's) and the resolution (a picture's refusal for its input: "a smaller --resolution",
    the server's bound being --max-size), worded all the same."""
    return "--" + setting.name.replace("_", "-") + (f" {setting.value}" if setting.value else "")


def worded(error: BaseException) -> str:
    """An error's text as the server says it: a library refusal's message with the settings it
    names worded as options (Refusal.words, option), any other error's own. The model files'
    refusals reach the operator at the start through it."""
    return error.words(option) if isinstance(error, Refusal) else str(error)


def smaller_way(found: Sequence[pull.Smaller], model_dir: bool = False) -> str:
    """The way out a refused start adds when the DiT is not in the smallest format and a smaller
    file fits the asked bound and count (pull.smaller_planning): "a smaller DiT file that plans
    this bound: int8 (--dit-model F, 8.3 GB) or the dynamic GGUF (--dit-model G, 4.8 GB)"; for a
    file that is not a pinned one, each 7B's files after "for the sharp 7B," and "for the 7B,",
    separated by semicolons. Where the server was given --model-dir (`model_dir`), it says so
    once, after "this bound": the pinned files are named, but only a file put in that directory
    is read there, nothing being fetched. The server's file is never changed by itself."""
    groups: dict[str | None, list[str]] = {}
    for one in found:
        file = option(Setting("dit_model", one.name))
        groups.setdefault(one.model, []).append(f"{one.label} ({file}, {pull.gigabytes(one.size)})")
    said = [
        (f"for {model}, " if model else "") + " or ".join(each) for model, each in groups.items()
    ]
    here = ", put in --model-dir" if model_dir else ""
    return f"a smaller DiT file that plans this bound{here}: " + "; ".join(said)


def smaller_files_line(this: int, found: Sequence[tuple[pull.Smaller, pull.Swaps]]) -> str:
    """The start's line of the smaller DiT files that swap fewer blocks than the asked file's
    plan, which swaps `this` (pull.smaller_planning's pairs): "smaller DiT files plan this bound
    better: this plan swaps 5 blocks; int8 0 blocks; the dynamic GGUF 0 blocks; --dit-model names
    them". A picture is one latent, so blocks alone, no window. For a file that is not a pinned
    one, each 7B's files after "for the sharp 7B" and "for the 7B", in brackets, separated by
    semicolons. The file is never changed by itself."""
    groups: dict[str | None, list[str]] = {}
    for one, swaps in found:
        groups.setdefault(one.model, []).append(f"{one.label} {_blocks_said(swaps.swapped)}")
    listed = "; ".join(
        f"for {model} ({'; '.join(each)})" if model else "; ".join(each)
        for model, each in groups.items()
    )
    return (
        f"smaller DiT files plan this bound better: this plan swaps {_blocks_said(this)};"
        f" {listed}; {option(Setting('dit_model'))} names them"
    )


def smaller(asked: Asked) -> str:
    """The way out of a refused start that names the smaller DiT files of the asked file
    (pull.smaller_files) whose reservation fits the asked bound and count (smaller_way); ""
    where none does, the file is in the smallest format, or its name is unknown."""
    if not asked.dit_file:
        return ""

    def plans(each: Architecture) -> bool | None:
        return True if fits(replace(asked, dit_model=each), asked.bound, asked.count) else None

    found = pull.smaller_planning(asked.dit_file, asked.dit_model, plans)
    if not found:
        return ""
    return f"; or {smaller_way([each for each, _ in found], asked.model_dir)}"


def smaller_line(asked: Asked, made: Made) -> str | None:
    """The start's line of the smaller DiT files of the asked file (pull.smaller_files) that swap
    fewer blocks than `made`, the plan of the asked file: each planned again for the same bound
    and count as `make` does (smaller_files_line; a picture is one latent, so blocks alone, no
    window: DESIGN.md, Weights, "The dynamic GGUF before v0.1.0 too"). None where the plan swaps
    none, the file is in the smallest format, its name is unknown, or no smaller file swaps
    fewer. The file is never changed by itself."""
    swapped = made.plan.models.swapped
    if not swapped or not asked.dit_file:
        return None

    def fewer(each: Architecture) -> pull.Swaps | None:
        try:
            other = _made(replace(asked, dit_model=each), asked.bound, asked.count)
        except Refusal:
            return None
        return (
            pull.Swaps(other.plan.models.swapped) if other.plan.models.swapped < swapped else None
        )

    found = pull.smaller_planning(asked.dit_file, asked.dit_model, fewer)
    return smaller_files_line(swapped, found) if found else None


def _blocks_said(swapped: int) -> str:
    """The blocks a plan swaps, as the line of the smaller files says them."""
    return f"{swapped} block{'' if swapped == 1 else 's'}"


def _count(count: int) -> str:
    return "one picture" if count == 1 else f"{count} pictures"


def _bound_said(bound: Bound) -> str:
    return f"{bound[0]}x{bound[1]}"


def gib(count: float) -> str:
    """Bytes in GiB, 2 decimals."""
    return f"{count / GIB:.2f}"


def megapixels(pixels: int) -> str:
    """Pixels in megapixels, 1 decimal, rounded up."""
    return f"{math.ceil(pixels / 1e5 - 1e-9) / 10:.1f}"

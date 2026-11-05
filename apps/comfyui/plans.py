"""What the nodes choose and what they say of it (DESIGN.md, Repository and apps, the ComfyUI
nodes): the DiT files by the names a user reads, the settings of an upscale node with the "auto"
the plan makes its own choice for, those settings as the library's pins, and the plan in the
nodes' words.

Plain Python and the library's planning modules, no torch and nothing of ComfyUI's: the node
classes (nodes.py) and the run (models.py) build on it, and the tests hold it alone. The nodes'
words never name an option of the command line: a setting is named as the widget the user sees,
on the node that has it (words.say)."""

from dataclasses import dataclass

from seedvr2x import peaks, pull, shots
from seedvr2x.memory import GIB, Reading
from seedvr2x.passes import Plan, Tiles
from seedvr2x.shot_plan import Peaks
from seedvr2x.shot_plan import Pins as ShotPins
from seedvr2x.still_plan import PictureTiles, StillPlan
from seedvr2x.still_plan import Pins as StillPins

AUTO = "auto"

# The DiT files of v0.1.0 the nodes run, by the name a user reads: the model, the file's format,
# its size and the model card's typical error per weight (models/hf/README.md, "Error per weight:
# typical": float16 0.02%, int8 rotated 0.86%, the dynamic GGUF 7.55%), the sharp 7B first, the
# command line's default. The sizes are the pinned files' (pull.PINNED), said as pull.gigabytes
# says them.
_FORMATS = (
    ("float16", "seedvr2x_ema_{}7b{}_fp16.safetensors", "0.02%"),
    ("int8", "seedvr2x_ema_{}7b{}_int8_convrot.safetensors", "0.86%"),
    ("dynamic GGUF", "seedvr2x_ema_{}7b{}_dyn.gguf", "7.55%"),
)


def _file(pattern: str, sharp: bool) -> str:
    """The pinned file's name: the sharp 7B's carries `sharp` after the 7B, the regular one's not
    ("seedvr2x_ema_7b_sharp_fp16.safetensors", "seedvr2x_ema_7b_fp16.safetensors")."""
    return pattern.format("", "_sharp" if sharp else "")


DIT_LABELS: dict[str, str] = {
    _file(
        pattern, sharp
    ): f"{model}, {kind} ({pull.gigabytes(pull.PINNED[_file(pattern, sharp)].size)}, {error})"
    for sharp, model in ((True, "Sharp 7B"), (False, "7B"))
    for kind, pattern, error in _FORMATS
}
DEFAULT_DIT = "seedvr2x_ema_7b_sharp_fp16.safetensors"
VAE_FILE = "seedvr2x_ema_vae_fp16.safetensors"

# The settings of an upscale node, each an INT widget (or a combo) so that any node that outputs
# an INT can drive it. "Auto" is a value no pin can take: 0 for a tile's side and the window cap,
# -1 for the blocks swapped (0 is a real pin there: swap none). What the plan tries is the grid
# of the widget (min, max, step), so that a typed or dragged value lands on it (the frontend
# rounds an INT to the nearest multiple of its step from its min, ComfyUI_frontend 1.57's
# useIntWidget): a tile a multiple of peaks.TILE_STEP from peaks.TILE_MIN to peaks.TILE_MAX, the
# tiles the plan makes; blocks from 0 to the DiT's; a window from 5 latents (shots.SHARED makes
# shorter ones leave no latent of their own) to 128.
#
# Untiled is a tile pin of its own in the library (0), which an INT with 0 for auto cannot
# say, and no side says it: a side that covers the frame is refused there (shot_plan's
# _covering), the planner taking a covering tile for no tiling. So the widget's one value past the
# largest tile, TILE_MAX + TILE_STEP, on the same grid, is untiled; the frontend shows no word for
# a special value of an INT (its widget has a min, a max, a step and a display type), so the label
# and the tooltip say it.
AUTO_TILE = 0
AUTO_WINDOW = 0
AUTO_SWAP = -1
UNTILED_TILE = peaks.TILE_MAX + peaks.TILE_STEP
WINDOW_MIN = 2 * shots.SHARED + 1
WINDOW_MAX = 128
ATTENTIONS = (AUTO, peaks.SDPA, peaks.FLASH_ATTN_2)
CORRECTIONS = ("split", "none")
# The command line's default seed, and the highest seed it takes for a shot at frame 0, which a
# picture and a clip both are: the seed plus the encode's offset must stay under numz's 2**32
# (job.SEED_LIMIT, passes.ENCODE_SEED_OFFSET).
SEED = 42
SEED_MAX = 2**32 - 1 - 1_000_000
# The command line's default resolution (job.DEFAULT_RESOLUTION), and the bounds of the widget: a
# short side of 18 pixels is the least the padding can be made on once the output's sides are
# cropped to even ones (frames.check_target, frames.output_size: 17 gives 16 rows), 16,384
# the most a lossless WebP can write, a bound no node needs but that keeps the widget finite.
RESOLUTION = 1080
RESOLUTION_MIN = 18
RESOLUTION_MAX = 16_384
# Keep in RAM between runs: auto keeps the weights where they fit under ComfyUI's own cap on
# pinned memory.
KEEP_MODES = ("auto", "on", "off")


class BadSetting(ValueError):
    """A setting no widget of the node takes, worded for the node's user (it reaches the user as
    the node's refusal, at queue time when the value is typed, at the node's start when linked)."""


def tile_problem(label: str, value: int) -> str | None:
    """Why `value` is no tile side of the widget `label`, or None: 0 is auto, UNTILED_TILE one
    pass, else a multiple of peaks.TILE_STEP from peaks.TILE_MIN to peaks.TILE_MAX."""
    if value in (AUTO_TILE, UNTILED_TILE):
        return None
    if peaks.TILE_MIN <= value <= peaks.TILE_MAX and value % peaks.TILE_STEP == 0:
        return None
    return (
        f"{label} {value}: a tile the plan makes is a multiple of {peaks.TILE_STEP} from"
        f" {peaks.TILE_MIN} to {peaks.TILE_MAX} px; 0 is auto, {UNTILED_TILE} is one pass"
        " (untiled)"
    )


def swap_problem(value: int) -> str | None:
    if AUTO_SWAP <= value <= peaks.BLOCKS:
        return None
    return f"swap blocks {value}: from 0 to {peaks.BLOCKS} blocks, or {AUTO_SWAP} for auto"


def window_problem(value: int) -> str | None:
    if value == AUTO_WINDOW or WINDOW_MIN <= value <= WINDOW_MAX:
        return None
    return (
        f"window {value}: from {WINDOW_MIN} to {WINDOW_MAX} latents (4 frames each), or 0 for auto"
    )


def seed_problem(value: int) -> str | None:
    if 0 <= value <= SEED_MAX:
        return None
    return f"seed {value}: from 0 up to {SEED_MAX:,}"


def problems(
    *,
    window: int | None = None,
    swap_blocks: int | None = None,
    encode_tile: int | None = None,
    decode_tile: int | None = None,
    seed: int | None = None,
) -> list[str]:
    """What is wrong with the numbers given (a number left None is not checked): each reason as
    the node words it. The widgets' own min, max and step cannot refuse a typed or linked value
    that is on the grid but not a tile (64 to 448 px), nor a window of 1 to 4 latents."""
    found = [
        None if encode_tile is None else tile_problem("encode tile", encode_tile),
        None if decode_tile is None else tile_problem("decode tile", decode_tile),
        None if swap_blocks is None else swap_problem(swap_blocks),
        None if window is None else window_problem(window),
        None if seed is None else seed_problem(seed),
    ]
    return [each for each in found if each is not None]


def _tile(value: int) -> int | None:
    """A tile widget's value as the library's pin: None for auto, 0 for untiled."""
    return None if value == AUTO_TILE else 0 if value == UNTILED_TILE else value


@dataclass(frozen=True)
class Settings:
    """The choices of an upscale node, and nothing of a plan: the pins the plan takes exactly
    (None: left to the plan), the attention kernel (None: whichever the environment runs), the
    colour correction and the seed. Held by value: a workflow saved and run again gives the same."""

    window: int | None = None
    swap_blocks: int | None = None
    encode_tile: int | None = None  # 0: untiled
    decode_tile: int | None = None  # 0: untiled
    attention: str | None = None
    color_correction: str = "split"
    seed: int = SEED

    @classmethod
    def of(
        cls,
        *,
        window: int = AUTO_WINDOW,
        swap_blocks: int = AUTO_SWAP,
        encode_tile: int = AUTO_TILE,
        decode_tile: int = AUTO_TILE,
        attention: str = AUTO,
        color_correction: str = "split",
        seed: int = SEED,
    ) -> "Settings":
        """The settings of the widgets' values. BadSetting, worded for the node, for a value the
        widget does not take."""
        found = problems(
            window=window,
            swap_blocks=swap_blocks,
            encode_tile=encode_tile,
            decode_tile=decode_tile,
            seed=seed,
        )
        if attention not in ATTENTIONS:
            found.append(f"attention {attention!r}: one of {', '.join(ATTENTIONS)}")
        if color_correction not in CORRECTIONS:
            found.append(f"colour correction {color_correction!r}: one of {', '.join(CORRECTIONS)}")
        if found:
            raise BadSetting("; ".join(found))
        return cls(
            None if window == AUTO_WINDOW else window,
            None if swap_blocks == AUTO_SWAP else swap_blocks,
            _tile(encode_tile),
            _tile(decode_tile),
            None if attention == AUTO else attention,
            color_correction,
            seed,
        )

    @property
    def split(self) -> bool:
        return self.color_correction == "split"

    def clip_pins(self) -> ShotPins:
        """The clip's pins (shot_plan.Pins)."""
        return ShotPins(self.window, self.encode_tile, self.decode_tile, self.swap_blocks)

    def picture_pins(self) -> StillPins:
        """The picture's pins (still_plan.Pins): a picture is one latent and has no window."""
        return StillPins(self.encode_tile, self.decode_tile, self.swap_blocks)


# ---- Keeping the weights in RAM ------------------------------------------------------------------


@dataclass(frozen=True)
class Keeping:
    """What "keep in RAM between runs" came to for a run: the mode asked (KEEP_MODES), whether the
    weights are kept, the bytes the kept set pins (the DiT's and the VAE's files), ComfyUI's cap on
    pinned memory in bytes (0 or less: ComfyUI pins none on this machine) and where the cap was
    read from. `because` is set where another reason decided (ComfyUI runs without smart memory:
    it unloads every model after each prompt)."""

    mode: str
    kept: bool
    size: int
    cap: int
    cap_from: str
    because: str = ""

    @property
    def text(self) -> str:
        """The decision in the node's words, for the plan and the console."""
        if self.because:
            return f"not kept: {self.because}"
        if self.mode == "off":
            return "not kept, as asked: the files are read for each run and dropped after it"
        size, cap = gib(self.size), gib(max(self.cap, 0))
        if self.cap <= 0:
            if self.mode == "on":
                return (
                    f"kept, {size} (ComfyUI pins no memory on this machine, its cap is 0: kept"
                    " as asked)"
                )
            return (
                f"not kept: ComfyUI pins no memory on this machine, its cap is 0 ({size} to keep)"
            )
        if self.size <= self.cap:
            return f"kept, {size} (cap {cap})"
        if self.mode == "on":
            return f"kept, {size} (over ComfyUI's cap of {cap} on pinned memory: kept as asked)"
        return f"not kept: {size} is over the cap of {cap}"


# ---- The plan in the nodes' words ----------------------------------------------------------------


@dataclass(frozen=True)
class Room:
    """What the run asked of the card and of ComfyUI (models.session), in bytes: the card's name,
    the driver's free memory when the run began (`free`), what ComfyUI's own models held of the
    card (`comfy`), the free memory the plan needs (`needed`, the largest phase with the margin and
    what the reading holds back), whether ComfyUI was asked to free it (`asked`), the driver's
    free memory once it had (`after`; `free` where nothing was asked), the system memory the host
    had available when the plan was made (`host`, None: unread), whether the run is a plan
    only, which asks ComfyUI for nothing and loads nothing, and the bytes asked of ComfyUI over
    `needed` where it was asked (`margin`, models.ASK_MARGIN)."""

    card: str
    free: int
    comfy: int
    needed: int
    asked: bool
    after: int
    host: int | None = None
    plan_only: bool = False
    margin: int = 0


def gib(count: float) -> str:
    return f"{count / GIB:.2f} GiB"


def _room(room: Room, reading: Reading) -> list[str]:
    lines = [
        f"card: {room.card}, {gib(room.free)} free; ComfyUI's own models hold {gib(room.comfy)}"
        f" of the card, and the plan is made as if they were gone: it counts on"
        f" {gib(reading.counted)}"
    ]
    if room.plan_only:
        lines.append(
            f"plan only: nothing is loaded and ComfyUI is asked to free nothing; a run would need"
            f" {gib(room.needed)} of free memory for the plan's largest phase"
        )
    elif room.asked:
        lines.append(
            f"ComfyUI was asked to free {gib(room.needed + room.margin)} of free memory (the plan's"
            f" {gib(room.needed)} for its largest phase, and {gib(room.margin)} over it for"
            f" ComfyUI's partial unload); the card then had {gib(room.after)} free"
        )
    else:
        lines.append(
            f"ComfyUI was asked to free nothing: the card has the {gib(room.needed)} the plan's"
            " largest phase needs, so its models stay where they are"
        )
    return lines


def _became(label: str, pinned: bool, outcome: str) -> str:
    """One setting and what it came to: "encode tile: auto -> tiles of 1024 px", or the outcome
    "as set" for a pin the plan took exactly."""
    return f"{label}: {outcome}, as set" if pinned else f"{label}: auto -> {outcome}"


def _tiles(label: str, pinned: bool, tiles: Tiles | None, peak: int | None) -> str:
    how = "one pass, untiled" if tiles is None else f"tiles of {tiles.size} px"
    said = "" if peak is None else f", predicted peak {gib(peak)}"
    return _became(label, pinned, f"{how}{said}")


def _swap(pinned: bool, swapped: int) -> str:
    if not swapped:
        outcome = "none, the DiT stays on the card in its phase"
    else:
        outcome = (
            f"{swapped} of the DiT's {peaks.BLOCKS}, which wait in system memory and stream"
            " through the card"
        )
    return _became("swap blocks", pinned, outcome)


_ATTENTION = {peaks.FLASH_ATTN_2: "FlashAttention 2", peaks.SDPA: "PyTorch's SDPA"}


def _common(
    dit: str,
    attention: str,
    swapped: int,
    settings: Settings,
    keeping: Keeping,
) -> list[str]:
    return [
        f"DiT: {DIT_LABELS.get(dit, dit)}",
        _became(
            "attention",
            settings.attention is not None,
            _ATTENTION.get(attention, attention),
        ),
        _swap(settings.swap_blocks is not None, swapped),
        f"colour correction {settings.color_correction}, seed {settings.seed}",
    ]


def _host(
    available: int | None,
    keeping: Keeping,
    held: int,
    plan_counts: list[str],
) -> str:
    """The system memory the plan counts, beside what the host has: `plan_counts` the parts of
    the run's own, `held` the bytes of the weights already pinned (not in the available figure)."""
    has = "the host's available memory was not read, so the plan does not check it"
    if available is not None:
        has = f"the host has {gib(available)} available"
    pinned = (
        f"the weights kept in RAM, {gib(held)} pinned, are not in that figure"
        if keeping.kept and held
        else f"the weights kept in RAM, {gib(keeping.size)}, will be pinned out of it"
        if keeping.kept
        else "the weights are not kept: only what leaves the card is pinned, in the plan's count"
    )
    return f"system memory: {'; '.join(plan_counts)}; {has}; {pinned}"


def picture_text(
    *,
    count: int,
    source: tuple[int, int],
    output: tuple[int, int],
    resolution: int,
    dit: str,
    attention: str,
    plan: StillPlan,
    tiles: PictureTiles,
    room: Room,
    settings: Settings,
    keeping: Keeping,
    notes: tuple[str, ...] = (),
    held: int = 0,
) -> str:
    """The plan of a batch of pictures, each upscaled on its own: what they are, the card and what
    ComfyUI was asked, each setting and what it came to, the system memory. `source` and `output`
    are (width, height) in pixels; `held` the bytes of the weights already pinned."""
    what = "1 picture" if count == 1 else f"{count} pictures, each on its own,"
    inputs = count * source[0] * source[1] * 3 * 4
    outputs = count * output[0] * output[1] * 3 * 4
    lines = [
        f"seedvr2x plan{' (plan only)' if room.plan_only else ''}: {what} {source[0]}x{source[1]}"
        f" to {output[0]}x{output[1]} (resolution {resolution})",
        *notes,
        *_room(room, plan.reading),
        *_common(dit, attention, plan.models.swapped, settings, keeping),
        _tiles("encode tile", settings.encode_tile is not None, tiles.encode, tiles.encode_peak),
        _tiles("decode tile", settings.decode_tile is not None, tiles.decode, tiles.decode_peak),
        f"models: {keeping.text}",
        _host(
            room.host,
            keeping,
            held,
            [
                f"the plan counts {gib(plan.host)}, of which {gib(plan.process)} for this process"
                f" (the input's {gib(inputs)} as stored, the batch's output {gib(outputs)} and the"
                " decode's copy of one picture)"
            ],
        ),
    ]
    return "\n".join(lines)


def clip_text(
    *,
    frames: int,
    source: tuple[int, int],
    output: tuple[int, int],
    resolution: int,
    dit: str,
    attention: str,
    plan: Plan,
    expected: Peaks,
    room: Room,
    settings: Settings,
    keeping: Keeping,
    notes: tuple[str, ...] = (),
    held: int = 0,
) -> str:
    """The plan of a clip, the frames of one shot in sliding windows, in the same words."""
    windows = len(shots.shot_layout(frames, plan.window))
    cap = "one window" if plan.window is None else f"windows of at most {plan.window} latents"
    inputs = frames * source[0] * source[1] * 3 * 4
    outputs = frames * output[0] * output[1] * 3 * 4
    counts = [
        f"the plan counts {gib(expected.host)}"
        + (
            ""
            if expected.scratch == 0
            else f", {gib(expected.scratch)} of it a tiled pass's scratch"
        ),
        f"the clip's output, {gib(outputs)}, is taken off what the host has",
        f"the input's frames, {gib(inputs)}, are already in memory",
    ]
    lines = [
        f"seedvr2x plan{' (plan only)' if room.plan_only else ''}: a clip of {frames} frames"
        f" {source[0]}x{source[1]} to {output[0]}x{output[1]} (resolution {resolution}): {windows}"
        f" DiT {'window' if windows == 1 else 'windows'}, {cap}; the longest takes"
        f" {expected.window} latents",
        *notes,
        *_room(room, expected.reading),
        *_common(dit, attention, plan.swapped, settings, keeping),
        _became(
            "window cap",
            settings.window is not None,
            "one window" if plan.window is None else f"at most {plan.window} latents",
        ),
        _tiles("encode tile", settings.encode_tile is not None, plan.encode_tiles, expected.encode),
        _tiles("decode tile", settings.decode_tile is not None, plan.decode_tiles, expected.decode),
        f"models: {keeping.text}",
        _host(room.host, keeping, held, counts),
    ]
    return "\n".join(lines)

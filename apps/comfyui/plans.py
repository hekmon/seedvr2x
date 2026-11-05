"""What the nodes choose and what they say of it (DESIGN.md, Repository and apps, the ComfyUI
nodes): the DiT files by the names a user reads, the settings of an upscale node with the "auto"
the plan makes its own choice for, those settings as the library's pins, and the plan in the
nodes' words.

Plain Python and the library's planning modules, no torch and nothing of ComfyUI's: the node
classes (nodes.py) and the run (models.py) build on it, and the tests hold it alone. The nodes'
words never name an option of the command line: a setting is named by the label the user sees
on the node that has it (LABELS, words.say)."""

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

# The settings of an upscale node. The numbers the plan makes (each pass's tile, the blocks
# swapped, a clip's window length) are optional INT inputs with no widget (the user's Desktop
# test, 2026-11-05: a widget whose 0 meant auto let a tile of 64 px be chosen): unwired, the plan
# chooses; wired, the value is taken exactly. Their min, max and step are the planner's own
# candidates, so that a value on the grid is one the plan could take: a tile a multiple of
# peaks.TILE_STEP from peaks.TILE_MIN to peaks.TILE_MAX, the tiles the plan makes; blocks from 0
# (none swapped) to the DiT's; a window from 5 latents (shots.SHARED makes shorter ones leave no
# latent of their own) to 128.
#
# Untiled is a tile pin of its own in the library (0), and no side says it: a side that covers
# the frame is refused there (shot_plan's _covering), the planner taking a covering tile for no
# tiling. So the input's one value past the largest tile, TILE_MAX + TILE_STEP, on the same grid,
# is untiled, which the tooltip says (ComfyUI_frontend 1.57 has no word for a special value of an
# INT: a min, a max, a step and a display type).
UNTILED_TILE = peaks.TILE_MAX + peaks.TILE_STEP
WINDOW_MIN = 2 * shots.SHARED + 1
WINDOW_MAX = 128
ATTENTIONS = (AUTO, peaks.SDPA, peaks.FLASH_ATTN_2)
# The colour correction, a BOOLEAN: on is split, off none (the user, 2026-11-05).
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

# The labels the nodes show for their inputs, which the plan's text and the refusals name.
LABELS = {
    "window": "window length",
    "swap_blocks": "blocks swapped",
    "encode_tile": "encode tile",
    "decode_tile": "decode tile",
    "resolution": "resolution",
    "attention": "attention",
    "color_correction": "colour correction",
    "seed": "seed",
    "plan_only": "plan only",
    "model": "model",
    "model_dir": "model folder",
    "keep_in_ram_between_runs": "keep in RAM between runs",
}


class BadSetting(ValueError):
    """A value an input of the node does not take, worded for the node's user (it reaches the user
    as the node's refusal: at queue time for a value the prompt holds, at the node's start for one
    a link gives)."""


def tile_problem(name: str, value: int) -> str | None:
    """Why `value` is no tile side for the input `name` (encode_tile, decode_tile), or None: a
    multiple of peaks.TILE_STEP from peaks.TILE_MIN to peaks.TILE_MAX, or UNTILED_TILE."""
    if value == UNTILED_TILE:
        return None
    if peaks.TILE_MIN <= value <= peaks.TILE_MAX and value % peaks.TILE_STEP == 0:
        return None
    return (
        f"{LABELS[name]} {value}: must be {peaks.TILE_MIN} to {peaks.TILE_MAX} px in steps of"
        f" {peaks.TILE_STEP}, or {UNTILED_TILE} for no tiles"
    )


def swap_problem(value: int) -> str | None:
    if 0 <= value <= peaks.BLOCKS:
        return None
    return f"{LABELS['swap_blocks']} {value}: must be from 0 to {peaks.BLOCKS}"


def window_problem(value: int) -> str | None:
    if WINDOW_MIN <= value <= WINDOW_MAX:
        return None
    return f"{LABELS['window']} {value}: must be from {WINDOW_MIN} to {WINDOW_MAX}"


def seed_problem(value: int) -> str | None:
    if 0 <= value <= SEED_MAX:
        return None
    return f"seed {value}: must be from 0 to {SEED_MAX:,}"


def _problem(name: str, value: object) -> str | None:
    """Why `value`, given for the number `name`, is not one the input takes, or None. A link may
    carry anything: what is no whole number is refused first."""
    if isinstance(value, bool) or not isinstance(value, int):
        return f"{LABELS[name]} {value!r}: must be a whole number"
    if name in ("encode_tile", "decode_tile"):
        return tile_problem(name, value)
    if name == "swap_blocks":
        return swap_problem(value)
    if name == "window":
        return window_problem(value)
    return seed_problem(value)


def problems(
    *,
    window: object = None,
    swap_blocks: object = None,
    encode_tile: object = None,
    decode_tile: object = None,
    seed: object = None,
) -> list[str]:
    """What is wrong with the numbers given (a number left None is not given, and not checked):
    each reason as the node words it. The inputs' own min, max and step hold neither a value that
    a link gives nor one on the grid but outside the tiles the plan makes."""
    given = {
        "encode_tile": encode_tile,
        "decode_tile": decode_tile,
        "swap_blocks": swap_blocks,
        "window": window,
        "seed": seed,
    }
    found = [_problem(name, value) for name, value in given.items() if value is not None]
    return [each for each in found if each is not None]


def _tile(value: int | None) -> int | None:
    """A tile input's value as the library's pin: None unwired (the plan's choice), 0 untiled."""
    return 0 if value == UNTILED_TILE else value


@dataclass(frozen=True)
class Settings:
    """The choices of an upscale node, and nothing of a plan: the pins the plan takes exactly
    (None: left to the plan), the attention kernel (None: whichever the environment runs), the
    colour correction (CORRECTIONS) and the seed. Held by value: a workflow saved and run again
    gives the same."""

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
        window: int | None = None,
        swap_blocks: int | None = None,
        encode_tile: int | None = None,
        decode_tile: int | None = None,
        attention: str = AUTO,
        color_correction: bool = True,
        seed: int = SEED,
    ) -> "Settings":
        """The settings of the node's inputs: a number None where its input is unwired, the
        colour correction's on or off. BadSetting, worded for the node, for a value an input does
        not take."""
        found = problems(
            window=window,
            swap_blocks=swap_blocks,
            encode_tile=encode_tile,
            decode_tile=decode_tile,
            seed=seed,
        )
        if attention not in ATTENTIONS:
            found.append(f"attention {attention!r}: must be one of {', '.join(ATTENTIONS)}")
        if not isinstance(color_correction, bool):  # pyright: ignore[reportUnnecessaryIsInstance]
            found.append(f"colour correction {color_correction!r}: must be on or off")
        if found:
            raise BadSetting("; ".join(found))
        return cls(
            window,
            swap_blocks,
            _tile(encode_tile),
            _tile(decode_tile),
            None if attention == AUTO else attention,
            CORRECTIONS[0] if color_correction else CORRECTIONS[1],
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
        """The decision in the node's words, after the mode chosen (`line`)."""
        if self.because:
            return f"not kept: {self.because}"
        if self.mode == "off":
            return "not kept: the files are read again for each run"
        size, cap = gib(self.size), gib(max(self.cap, 0))
        if self.cap <= 0:
            if self.mode == "on":
                return f"kept, {size}, though ComfyUI allows no RAM for this on this computer"
            return f"not kept: ComfyUI allows no RAM for this on this computer ({size} to keep)"
        if self.size <= self.cap:
            return f"kept, {size} (ComfyUI allows {cap})"
        if self.mode == "on":
            return f"kept, {size}, over the {cap} ComfyUI allows"
        return f"not kept: {size} is over the {cap} ComfyUI allows"

    @property
    def line(self) -> str:
        """The plan's line, also ComfyUI's console's: the mode chosen and what it came to."""
        return f"{LABELS['keep_in_ram_between_runs']}: {self.mode} -> {self.text}"


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
        f"graphics card: {room.card}, {gib(room.free)} free; ComfyUI's own models use"
        f" {gib(room.comfy)} of it, and seedvr2x plans as if they were unloaded, with"
        f" {gib(reading.counted)}"
    ]
    if room.plan_only:
        lines.append(
            "plan only: nothing is loaded and ComfyUI's models stay as they are; a run would need"
            f" up to {gib(room.needed)} free on the card, with a safety margin"
        )
    elif room.asked:
        lines.append(
            f"ComfyUI was asked to unload models until {gib(room.needed + room.margin)} was free"
            f" on the card: the {gib(room.needed)} this run needs, with a safety margin, and"
            f" {gib(room.margin)} more, as ComfyUI leaves some memory unfreed when it unloads;"
            f" the card then had {gib(room.after)} free"
        )
    else:
        lines.append(
            f"no need to unload ComfyUI's models: this run needs up to {gib(room.needed)} free on"
            " the card, with a safety margin, and the card has it"
        )
    return lines


def _became(name: str, given: bool, outcome: str) -> str:
    """One setting by its label and what it came to: "encode tile: auto -> 1024 px", or the
    outcome "as you set it" for a value given, which the plan took exactly."""
    label = LABELS[name]
    return f"{label}: {outcome}, as you set it" if given else f"{label}: auto -> {outcome}"


def _tiles(name: str, given: bool, tiles: Tiles | None, peak: int | None) -> str:
    how = "no tiles, the whole image at once" if tiles is None else f"{tiles.size} px"
    said = "" if peak is None else f" (needs up to {gib(peak)})"
    return _became(name, given, f"{how}{said}")


def _swap(given: bool, swapped: int) -> str:
    if not swapped:
        outcome = "none, the whole model on the card while it works"
    else:
        outcome = (
            f"{swapped} of the model's {peaks.BLOCKS}, which wait in RAM and pass through the card"
            " in turn"
        )
    return _became("swap_blocks", given, outcome)


def _frames(length: int) -> str:
    """A window's length as the node's input counts it, in groups of 4 frames, and in frames."""
    return f"{length} groups of 4 frames (about {4 * length} frames)"


_ATTENTION = {peaks.FLASH_ATTN_2: "FlashAttention 2", peaks.SDPA: "PyTorch's SDPA"}


def _common(dit: str, attention: str, swapped: int, settings: Settings) -> list[str]:
    return [
        f"model: {DIT_LABELS.get(dit, dit)}",
        _became(
            "attention",
            settings.attention is not None,
            _ATTENTION.get(attention, attention),
        ),
        _swap(settings.swap_blocks is not None, swapped),
        f"{LABELS['color_correction']}: {'on' if settings.split else 'off'}"
        f" ({settings.color_correction})",
        f"seed: {settings.seed}",
    ]


def _host(
    available: int | None,
    keeping: Keeping,
    held: int,
    plan_counts: list[str],
) -> str:
    """The RAM the plan counts, beside what the computer has: `plan_counts` the parts of the
    run's own, `held` the bytes of the model already kept (not in the available figure)."""
    has = "the RAM available was not read, so it is not checked"
    if available is not None:
        has = f"the computer has {gib(available)} available"
    kept = (
        f"the model already kept, {gib(held)}, is not counted in that"
        if keeping.kept and held
        else f"keeping the model takes {gib(keeping.size)} of that"
        if keeping.kept
        else "the model is not kept: only the part that does not fit on the card is in RAM"
        " during the run, counted above"
    )
    return f"RAM: {'; '.join(plan_counts)}; {has}; {kept}"


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
    ComfyUI was asked, each setting and what it came to, the RAM. `source` and `output` are
    (width, height) in pixels; `held` the bytes of the model already kept in RAM."""
    what = "1 image" if count == 1 else f"{count} images, each on its own,"
    inputs = count * source[0] * source[1] * 3 * 4
    outputs = count * output[0] * output[1] * 3 * 4
    lines = [
        f"seedvr2x plan{' (plan only)' if room.plan_only else ''}: {what} {source[0]}x{source[1]}"
        f" to {output[0]}x{output[1]} (resolution {resolution})",
        *notes,
        *_room(room, plan.reading),
        *_common(dit, attention, plan.models.swapped, settings),
        _tiles("encode_tile", settings.encode_tile is not None, tiles.encode, tiles.encode_peak),
        _tiles("decode_tile", settings.decode_tile is not None, tiles.decode, tiles.decode_peak),
        keeping.line,
        _host(
            room.host,
            keeping,
            held,
            [
                f"this run needs about {gib(plan.host)}, {gib(plan.process)} of it for ComfyUI"
                f" itself and the images (the input's {gib(inputs)}, the output's {gib(outputs)}"
                " and a working copy of one image)"
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
    inputs = frames * source[0] * source[1] * 3 * 4
    outputs = frames * output[0] * output[1] * 3 * 4
    counts = [
        f"this run needs about {gib(expected.host)}"
        + (
            ""
            if expected.scratch == 0
            else f", {gib(expected.scratch)} of it for a pass made in tiles"
        ),
        f"the clip's output, {gib(outputs)}, is taken from what the computer has",
        f"the input's frames, {gib(inputs)}, are already in memory",
    ]
    lines = [
        f"seedvr2x plan{' (plan only)' if room.plan_only else ''}: a clip of {frames} frames"
        f" {source[0]}x{source[1]} to {output[0]}x{output[1]} (resolution {resolution}): "
        + (
            "the whole clip in one window of the model"
            if windows == 1
            else f"{windows} windows of the model, the longest {_frames(expected.window)}"
        ),
        *notes,
        *_room(room, expected.reading),
        *_common(dit, attention, plan.swapped, settings),
        _became(
            "window",
            settings.window is not None,
            "the whole clip in one window"
            if plan.window is None
            else f"at most {_frames(plan.window)}",
        ),
        _tiles("encode_tile", settings.encode_tile is not None, plan.encode_tiles, expected.encode),
        _tiles("decode_tile", settings.decode_tile is not None, plan.decode_tiles, expected.decode),
        keeping.line,
        _host(room.host, keeping, held, counts),
    ]
    return "\n".join(lines)

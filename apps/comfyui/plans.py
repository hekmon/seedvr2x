"""What the nodes choose and what they say of it (DESIGN.md, Repository and apps, the ComfyUI
nodes): the DiT files by the names a user reads, the settings of an upscale node with the "auto"
the plan makes its own choice for, those settings as the library's pins, and the plan in the
nodes' words.

Plain Python and the library's planning modules, no torch and nothing of ComfyUI's: the node
classes (nodes.py) and the run (models.py) build on it, and the tests hold it alone. The nodes'
words never name an option of the command line: a setting is named by the label the user sees
on the node that has it (LABELS, words.say)."""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal

from seedvr2x import peaks, pull, shotcuts, shotruns, shots
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


def file_label(name: str) -> str:
    """A model file as the node's progress text names it: "the sharp 7B (float16)", "the VAE",
    "the shot detector"; any other file by its name."""
    if name == VAE_FILE:
        return "the VAE"
    if name == pull.DETECTOR:
        return "the shot detector"
    label = DIT_LABELS.get(name)
    if label is None:
        return name
    model, kind = label.split(" (")[0].split(", ")
    return f"the {model.replace('Sharp', 'sharp')} ({kind})"


# The settings of an upscale node. The numbers the plan makes (each pass's tile, the blocks
# swapped, a clip's window length) are optional sockets of seedvr2x's own types, each given by a
# value node of its own (the user's Desktop test, 2026-11-05: a widget whose 0 meant auto let a
# tile of 64 px be chosen; a socket of its own type lets ComfyUI propose its node): unconnected,
# the plan chooses; connected, the value is taken exactly. A socket carries the library's pin
# (the tile's side in px, 0 for untiled; the blocks; the window in latents), which the value nodes
# make from what the user sets on their grid, the planner's own candidates: a tile a multiple of
# peaks.TILE_STEP from peaks.TILE_MIN to peaks.TILE_MAX, the tiles the plan makes, or the whole
# image at once; blocks from 0 (none swapped) to the DiT's; a window from 5 latents (shots.SHARED
# makes shorter ones leave no latent of their own) to 128, set in frames.
WINDOW_MIN = 2 * shots.SHARED + 1
WINDOW_MAX = 128
# The value nodes' defaults: a tile of 1024 px (design, 2026-11-05), no block swapped, and a
# window of 41 frames (11 latents, the 73-frame clip's on the RTX 5090 in impl-nodes' GPU run).
TILE_DEFAULT = 1024
SWAP_DEFAULT = 0
WINDOW_FRAMES_DEFAULT = 41


def window_frames(latents: int) -> int:
    """A window of `latents` latents in frames, the model's grid: its first latent is one frame,
    each other four (the VAE's temporal compression), 4(L - 1) + 1."""
    return 4 * (latents - 1) + 1


def window_latents(frames: int) -> int:
    """The latents of a window of `frames` frames on the model's grid (window_frames' inverse)."""
    return (frames - 1) // 4 + 1


WINDOW_FRAMES_MIN = window_frames(WINDOW_MIN)
WINDOW_FRAMES_MAX = window_frames(WINDOW_MAX)
WINDOW_FRAMES_STEP = 4
ATTENTIONS = (AUTO, peaks.SDPA, peaks.FLASH_ATTN_2)
# Upscale Video's quality, the library's levels (shotruns.QUALITY): standard, the default, runs each
# shot once, as the command line's --quality does.
QUALITY_DEFAULT = "standard"
# The colour correction, a BOOLEAN: on is split, off none (the user, 2026-11-05).
CORRECTIONS = ("split", "none")
# The command line's default seed, and the highest seed it takes for a shot at frame 0, which a
# picture is: the seed plus the encode's offset must stay under numz's 2**32 (job.SEED_LIMIT,
# passes.ENCODE_SEED_OFFSET). A clip's later shots add their first frame (nodes._check_seed).
SEED = 42
SEED_MAX = 2**32 - 1 - 1_000_000
# The command line's default resolution (job.DEFAULT_RESOLUTION), and the bounds of the widget: a
# short side of 18 pixels is the least a picture's padding can be made on once the output's sides
# are cropped to even ones (frames.check_picture_target: 17 gives 16 rows), 16,384
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
    "quality": "quality",
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


# ---- The value nodes' grids, in the user's units -------------------------------------------------


def _whole(label: str, value: object) -> str | None:
    """Why a value given for a number is none (a link may carry anything), or None."""
    if isinstance(value, bool) or not isinstance(value, int):
        return f"{label} {value!r}: must be a whole number"
    return None


def tile_size_problem(value: object) -> str | None:
    """Why `value` is no tile size seedvr2x Tile takes, or None."""
    if (problem := _whole("tile size", value)) is not None:
        return problem
    assert isinstance(value, int)
    if peaks.TILE_MIN <= value <= peaks.TILE_MAX and value % peaks.TILE_STEP == 0:
        return None
    return (
        f"tile size {value}: must be {peaks.TILE_MIN} to {peaks.TILE_MAX} px in steps of"
        f" {peaks.TILE_STEP}"
    )


def blocks_problem(value: object) -> str | None:
    """Why `value` is no count seedvr2x Blocks Swapped takes, or None."""
    if (problem := _whole(LABELS["swap_blocks"], value)) is not None:
        return problem
    assert isinstance(value, int)
    if 0 <= value <= peaks.BLOCKS:
        return None
    return f"{LABELS['swap_blocks']} {value}: must be from 0 to {peaks.BLOCKS}"


def window_frames_problem(value: object) -> str | None:
    """Why `value` is no window length seedvr2x Window Length takes, in frames, or None."""
    if (problem := _whole(LABELS["window"], value)) is not None:
        return problem
    assert isinstance(value, int)
    if WINDOW_FRAMES_MIN <= value <= WINDOW_FRAMES_MAX and (value - 1) % WINDOW_FRAMES_STEP == 0:
        return None
    return (
        f"{LABELS['window']} {value}: must be {WINDOW_FRAMES_MIN} to {WINDOW_FRAMES_MAX} frames"
        f" in steps of {WINDOW_FRAMES_STEP} ({WINDOW_FRAMES_MIN}, {WINDOW_FRAMES_MIN + 4},"
        f" {WINDOW_FRAMES_MIN + 8}...)"
    )


def tile_pin(size: int, whole_image: bool) -> int:
    """seedvr2x Tile's value as the library's pin: the side in px, or 0 for the whole image."""
    return 0 if whole_image else size


# ---- What an upscale node takes, in the library's units ------------------------------------------

# The value node that gives each pin, named in a refusal of a value no such node gives.
VALUE_NODES = {
    "encode_tile": "seedvr2x Tile",
    "decode_tile": "seedvr2x Tile",
    "swap_blocks": "seedvr2x Blocks Swapped",
    "window": "seedvr2x Window Length",
}


def _pin_problem(name: str, value: object) -> str | None:
    """Why `value`, given to the upscale node's socket `name` in the library's units, is no value
    its value node gives, or None (another node of the same type could give anything)."""
    if isinstance(value, bool) or not isinstance(value, int):
        valid = False
    elif name in ("encode_tile", "decode_tile"):
        valid = value == 0 or tile_size_problem(value) is None
    elif name == "swap_blocks":
        valid = blocks_problem(value) is None
    else:
        valid = WINDOW_MIN <= value <= WINDOW_MAX
    if valid:
        return None
    return f"{LABELS[name]} {value!r}: not a value {VALUE_NODES[name]} gives; connect one to it"


def seed_problem(value: object) -> str | None:
    if (problem := _whole("seed", value)) is not None:
        return problem
    assert isinstance(value, int)
    if 0 <= value <= SEED_MAX:
        return None
    return f"seed {value}: must be from 0 to {SEED_MAX:,}"


def problems(
    *,
    window: object = None,
    swap_blocks: object = None,
    encode_tile: object = None,
    decode_tile: object = None,
    seed: object = None,
) -> list[str]:
    """What is wrong with the values an upscale node is given (a value left None is not given,
    and not checked): each reason as the node words it."""
    given = {
        "encode_tile": encode_tile,
        "decode_tile": decode_tile,
        "swap_blocks": swap_blocks,
        "window": window,
    }
    found = [_pin_problem(name, value) for name, value in given.items() if value is not None]
    if seed is not None:
        found.append(seed_problem(seed))
    return [each for each in found if each is not None]


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
        """The settings of the node's inputs: a pin in the library's units, None where its socket
        is unconnected; the colour correction's on or off. BadSetting, worded for the node, for a
        value an input does not take."""
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
            encode_tile,
            decode_tile,
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
        f" {gib(room.comfy)} of it, and seedvr2x counts on {gib(reading.counted)}"
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
    how = "the whole image at once, no tiles" if tiles is None else f"{tiles.size} px"
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
    """A window's length in latents, said in frames as seedvr2x Window Length sets it."""
    return f"{window_frames(length)} frames"


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


def _shot_line(number: int, shot: shots.Shot, window: int | None) -> str:
    """One shot of a clip in the plan's text: where it is in the clip (frames counted from 0, the
    first and the last of it) and its windows of the model, by the layout the pass runs
    (shots.shot_layout, in frames as seedvr2x Window Length sets them)."""
    layout = shots.shot_layout(shot.frames, window)
    longest = max(end - start for start, end in layout)
    how = (
        "the whole shot in one window of the model"
        if len(layout) == 1
        else f"{len(layout)} windows of the model, the longest {_frames(longest)}"
    )
    return f"shot {number}: frames {shot.start} to {shot.end - 1} ({shot.frames} frames): {how}"


def _quality(cuts: Sequence[shots.Shot], quality: str) -> str:
    """The quality level's line in a clip's plan: the level, the runs of each shot and in all (a
    shot too short for a run's first frame has no such run, shotruns.shot_runs), and what they
    cost, which is time alone (the runs are made one after the other, the plan the same:
    clip.plan)."""
    label = LABELS["quality"]
    per_shot = shotruns.QUALITY[quality]
    if per_shot == 1:
        return f"{label}: {quality}, the model runs once over each shot"
    runs = sum(len(shotruns.shot_runs(each, per_shot)) for each in cuts)
    return (
        f"{label}: {quality}, the model runs {per_shot} times over each shot, {runs} runs in all,"
        " each frame kept from the run that rebuilds it best: about"
        f" {per_shot} times standard's time, and no more memory"
    )


def clip_text(
    *,
    cuts: Sequence[shots.Shot],
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
    quality: str = QUALITY_DEFAULT,
) -> str:
    """The plan of a clip, the frames of its shots each in sliding windows, in the same words: a
    clip of one shot says its windows in its first line, a clip of several a line for each shot.
    `cuts` are the shots, frames [start, end) covering the clip; `quality` the level
    (shotruns.QUALITY), said with the runs it makes."""
    frames = sum(each.frames for each in cuts)
    inputs = frames * source[0] * source[1] * 3 * 4
    outputs = frames * output[0] * output[1] * 3 * 4
    counts = [
        f"this run needs about {gib(expected.host + outputs)}: {gib(outputs)} for the clip's"
        f" output, kept whole, and {gib(expected.host)} for seedvr2x's own work"
        + (
            ""
            if expected.scratch == 0
            else f" ({gib(expected.scratch)} of it for a pass made in tiles)"
        ),
        f"the input's frames, {gib(inputs)}, are not in that: ComfyUI has them in memory already,"
        " so they are out of the RAM available",
    ]
    first = f"seedvr2x plan{' (plan only)' if room.plan_only else ''}: a clip of {frames} frames"
    first += f" {source[0]}x{source[1]} to {output[0]}x{output[1]} (resolution {resolution})"
    if len(cuts) == 1:
        windows = len(shots.shot_layout(cuts[0].frames, plan.window))
        first += ": " + (
            "the whole clip in one window of the model"
            if windows == 1
            else f"{windows} windows of the model, the longest {_frames(expected.window)}"
        )
        per_shot: list[str] = []
    else:
        first += f", in {len(cuts)} shots, each upscaled on its own"
        per_shot = [_shot_line(number, each, plan.window) for number, each in enumerate(cuts, 1)]
    lines = [
        first,
        *per_shot,
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
        _quality(cuts, quality),
        _tiles("encode_tile", settings.encode_tile is not None, plan.encode_tiles, expected.encode),
        _tiles("decode_tile", settings.decode_tile is not None, plan.decode_tiles, expected.decode),
        keeping.line,
        _host(room.host, keeping, held, counts),
    ]
    return "\n".join(lines)


# ---- The cuts in the nodes' words ----------------------------------------------------------------


def probability(value: float) -> str:
    """A cut's probability to 3 decimals, the float32 value's exact decimal cut and never rounded
    up (as the command line's cut list shows it), so that a possible cut never shows at the
    threshold it stays under, nor a cut under the one it reaches."""
    return str(Decimal(float(value)).quantize(Decimal("0.001"), rounding=ROUND_DOWN))


def one_shot_text(frames: int) -> str:
    """The Cuts node's text for "one shot": the clip taken whole, no detection run."""
    return f"seedvr2x Cuts: one shot, the whole clip of {frames} frames; no detection was run"


def cuts_text(
    frames: int,
    threshold: float,
    found: Sequence[shotcuts.Cut],
    possible: Sequence[shotcuts.Cut],
) -> str:
    """The Cuts node's text for "detect": the shots found at `threshold`, each cut by the first
    frame of the shot it starts (frames counted from 0, as the command line's cut list counts
    them) with the probability of the detection, then the possible cuts, which the list does not
    hold: runs that peak from shotcuts.POSSIBLE up to the threshold, for the eyes to check."""
    shots_found = len(found) + 1
    lines = [
        f"seedvr2x Cuts: {frames} frames, {len(found)} cut{'' if len(found) == 1 else 's'} found at"
        f" threshold {threshold:g}: {shots_found} shot{'' if shots_found == 1 else 's'}"
    ]
    lines += [
        f"cut at frame {each.frame} (probability {probability(each.probability or 0.0)})"
        for each in found
    ]
    if possible:
        lines.append(
            f"{len(possible)} possible cut{'' if len(possible) == 1 else 's'}, not in the list:"
            f" scenes the detector found less sure than {threshold:g} (from"
            f" {shotcuts.POSSIBLE:g}), to check by eye; a lower threshold makes them cuts"
        )
        lines += [
            f"possible cut at frame {each.frame}"
            f" (probability {probability(each.probability or 0.0)})"
            for each in possible
        ]
    else:
        if threshold <= shotcuts.POSSIBLE:
            # The possible cuts are the scenes between POSSIBLE and the threshold: none to find.
            lines.append(
                f"no possible cuts: at a threshold of {shotcuts.POSSIBLE:g} or under, every scene"
                " change the detector sees is a cut"
            )
        else:
            lines.append(
                f"no possible cuts: nothing between {shotcuts.POSSIBLE:g} and {threshold:g}"
            )
    return "\n".join(lines)

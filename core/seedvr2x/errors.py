"""What a brick of the library refuses to run (DESIGN.md, Repository and apps): the library's
own base for its refusals, so that it imports nothing of the command line's job. Plain Python, no
torch.

A refusal's message may name a setting as the way out ("window 6", "decode tile 512"). The library
does not know how an application calls its settings (the command line's options, the image
server's, a node's widgets): it names each by a `Setting`, and the message is a function of how
the application words one (`Say`), which `Refusal.words` calls; `str()` of the refusal is the
library's own words (`plain`), for its logs, and names no option of any application."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

# The settings a refusal names as its way out, by the library's name for each, which an
# application words as its own (Setting). The library's names, not an application's spelling: the
# command line's option for each is "--" and the name with its underscores as hyphens.
SETTINGS: dict[str, str] = {
    "window": "the cap on the DiT's windows, in latents (a video's)",
    "swap_blocks": "how many of the DiT's blocks are swapped between the GPU and system memory",
    "encode_tile": "the side in output pixels of the encode's tiles, 0 for the untiled pass",
    "decode_tile": "the side in output pixels of the decode's tiles, 0 for the untiled pass",
    "resolution": "the output's short side in pixels, which sets its size",
    "dit_model": "the DiT's model file",
    "vae_model": "the VAE's model file",
    "model_dir": "the directory that holds the model files, read there instead of fetched",
    "cuts": "the cut list that replaces the shot detection (a video's)",
    "compile": "whether the VAE is compiled (torch.compile)",
}

# What `plain` says of each setting: a short noun phrase, the value after it. A setting that
# names a whole feature (compile) says it as a gerund, so that "the same job without compiling"
# reads.
_PLAIN: dict[str, str] = {
    "window": "window",
    "swap_blocks": "swapped blocks",
    "encode_tile": "encode tile",
    "decode_tile": "decode tile",
    "resolution": "resolution",
    "dit_model": "DiT file",
    "vae_model": "VAE file",
    "model_dir": "the model directory",
    "cuts": "the cut list",
    "compile": "compiling",
}


@dataclass(frozen=True)
class Setting:
    """A setting a refusal names as its way out, by the library's name for it (SETTINGS) and the
    value it names ("" where the refusal names the setting alone: "a smaller resolution"). An
    application words it as its own (Say)."""

    name: str
    value: str = ""

    def __post_init__(self) -> None:
        if self.name not in SETTINGS:
            raise ValueError(f"{self.name!r} is not one of the settings a refusal names")


Say = Callable[[Setting], str]
Message = str | Callable[[Say], str]


def worded(message: Message) -> Callable[[Say], str]:
    """`message` as a function of Say: a plain string gives itself, whatever the words."""
    if callable(message):
        return message
    text: str = message
    return lambda _say: text


def plain(setting: Setting) -> str:
    """The library's own words for a setting, for its logs: a short noun phrase with the value
    ("window 6", "swapped blocks 30", "decode tile 512"), naming no application's option."""
    return f"{_PLAIN[setting.name]} {setting.value}" if setting.value else _PLAIN[setting.name]


def aside(say: Say, apps: str, plain_text: str = "") -> str:
    """`apps` where an application words the settings, `plain_text` for the library's own words
    (`plain`): a clause that only points at an application's option, "(--dit-model)" after "given
    as the DiT", which the library's log has no use for."""
    return plain_text if say is plain else apps


class Refusal(ValueError):
    """What a brick refuses to run: the message says what, for the user. The command line reports
    it as it reports a job it refuses.

    The message is a string, or a function of `Say` (Message) where it names a setting: str() is
    the message in the library's own words (`plain`), `words(say)` the message in an
    application's.

    A refusal that an application other than the command line must word for itself (the image
    server answers a refused picture with the reason and a way out of its own, DESIGN.md, The image
    server, Errors) carries its message in parts as well, all keyword-only and "" where a
    refusal has none; str() is the message, whole, whatever the parts:

    - kind: a stable lower-case slug naming the refusal ("raw", "too-large"); an application
      words its own way out by it. The kinds of a brick are listed in the brick (picture.KINDS).
    - reason: what is refused and why, a complete clause naming no file, no command and no option
      of the command line ("the picture has transparent pixels, and seedvr2x refuses it: ...").
    - way: the way out, "" where the refusal has none. A picture's (PictureRefused) is the command
      line's, as its message has it, which may name the file and give an ffmpeg command; a
      setting's (TargetTooSmall) is the library's plain words for it ("resolution 16",
      errors.plain), which an application words by its own option."""

    def __init__(
        self, message: Message, *, kind: str = "", reason: str = "", way: str = ""
    ) -> None:
        self._message = worded(message)
        super().__init__(self._message(plain))
        self.kind = kind
        self.reason = reason
        self.way = way

    def words(self, say: Say) -> str:
        """The message, the settings it names worded by `say` (an application's words for them)."""
        return self._message(say)


class JobError(Refusal):
    """A job seedvr2x can't lay out: a bad cut list; the message says what."""


@dataclass(frozen=True)
class Shortfall:
    """What a recorded shape is short of, as a refusal says it first (PlanRefused.shortfall):
    - what: the phase that doesn't fit, in words ("the encode", "the DiT's phase", "the
      decode"), or "" where it is the system's memory that falls short (host);
    - short: how much, in GiB to 2 decimals as the message's figures say it, for the GPU's memory;
    - host: the system's memory is what falls short, not the GPU's;
    - freeable: whether programs closed on the card give the shape its memory back, or a driver
      that refuses reports what they leave free; False where the card, whole, has no more to give
      a run held to it."""

    what: str
    short: str
    host: bool = False
    freeable: bool = True


class PlanRefused(JobError):
    """What a plan (shot_plan.plan for a video, still_plan.plan for a picture) refuses before the
    run, the base of the kinds below. Each carries the facts apart, so that an application words
    the refusal and its way out in its own terms; memory is in bytes, as ints, and a fact that
    doesn't apply is None. The message counts in free VRAM, never in card sizes.

    - phase: where the refusal is for want of memory, the phase that doesn't fit ("encode", "dit"
      or "decode") or "host" for the system's, for planner.plan_again to know which part of a shape
      to release; "pin" for a picture's pin that is no pin; else empty.
    - instead: the setting, with its value, that fits where this one doesn't (the window, the
      swapped blocks, the tile, the resolution); None where the refusal names none, and for a
      plan made with the tight reserve to name in its budget line.
    - shortfall: where a resume's recorded shape no longer fits the memory free, what it is short
      of (Shortfall), for whoever words that refusal with the figures apart; else None.
    - fits: a picture's refusal of a bound, the largest bound (long, short) that fits, in pixels;
      None elsewhere, and where none does.
    - fits_input: a picture's refusal of its input, the largest input in pixels that fits; None
      elsewhere, and where none does."""

    def __init__(
        self,
        message: Message,
        *,
        kind: str = "",
        phase: str = "",
        instead: Setting | None = None,
        shortfall: Shortfall | None = None,
        fits: tuple[int, int] | None = None,
        fits_input: int | None = None,
    ) -> None:
        super().__init__(message, kind=kind)
        self.phase = phase
        self.instead = instead
        self.shortfall = shortfall
        self.fits = fits
        self.fits_input = fits_input

    def reworded(self, message: Message, **facts: object) -> "PlanRefused":
        """This refusal with another message, of the same kind and facts but for those given: what
        the planner makes of a refusal it takes again (the tight reserve's, the compiled plan's)."""
        again = type(self).__new__(type(self))
        again.__dict__.update(self.__dict__)
        again.__dict__.update(facts)
        again._message = worded(message)
        again.args = (again._message(plain),)
        return again


class PinRefused(PlanRefused):
    """A pin the plan can't take exactly (kind "pin"): the setting asked and why.

    - pin: the pin asked, as a Setting (swap_blocks, encode_tile, decode_tile, window); None for a
      recorded plan's.
    - why: "range" (above the DiT's blocks), "covers" (a tile that covers the whole padded
      frame), "memory" (the pin doesn't fit the memory free; `instead` the pin that does),
      "compile" (a tile beyond the shapes measured compiled) or "recorded" (not a plan as
      seedvr2x records one).
    - need, there: for why "memory", the free VRAM the pin takes and what is there, in bytes."""

    def __init__(
        self,
        message: Message,
        *,
        pin: Setting | None = None,
        why: str = "",
        need: int | None = None,
        there: int | None = None,
        phase: str = "",
        instead: Setting | None = None,
        shortfall: Shortfall | None = None,
        fits: tuple[int, int] | None = None,
        fits_input: int | None = None,
    ) -> None:
        super().__init__(
            message,
            kind="pin",
            phase=phase,
            instead=instead,
            shortfall=shortfall,
            fits=fits,
            fits_input=fits_input,
        )
        self.pin = pin
        self.why = why
        self.need = need
        self.there = there


class MemoryShort(PlanRefused):
    """A job or a picture that doesn't fit the free VRAM (kind "memory"): the size, or the part
    of the plan, whose peak is over what the plan counts on, with no pin to blame.

    - need: the free VRAM the phase takes, in bytes, the margin counted.
    - there: what the plan counts on, in bytes.
    - short: need less there, in bytes.
    - freeable: whether programs closed on the card give the memory back; None where the plan
      does not say (a driver that refuses says what it reports).
    - reserve: on a run held to the card, the reserve the plan leaves torch's allocator beyond a
      step's peak, in bytes; None on a driver that refuses.
    - `instead`: the largest resolution, window or tile that fits, as a Setting; None where
      none does."""

    def __init__(
        self,
        message: Message,
        *,
        need: int | None = None,
        there: int | None = None,
        short: int | None = None,
        freeable: bool | None = None,
        reserve: int | None = None,
        phase: str = "",
        instead: Setting | None = None,
        shortfall: Shortfall | None = None,
        fits: tuple[int, int] | None = None,
        fits_input: int | None = None,
    ) -> None:
        super().__init__(
            message,
            kind="memory",
            phase=phase,
            instead=instead,
            shortfall=shortfall,
            fits=fits,
            fits_input=fits_input,
        )
        self.need = need
        self.there = there
        self.short = short
        self.freeable = freeable
        self.reserve = reserve


class HostShort(PlanRefused):
    """A plan that takes more system memory than the host has (kind "host", phase "host").

    - need: what the plan takes, in bytes.
    - available: what the host has available, in bytes.
    - short: need less available, in bytes.
    - `instead`: a shorter window whose plan fits, as a Setting; None where none does."""

    def __init__(
        self,
        message: Message,
        *,
        need: int | None = None,
        available: int | None = None,
        short: int | None = None,
        instead: Setting | None = None,
        shortfall: Shortfall | None = None,
    ) -> None:
        super().__init__(message, kind="host", phase="host", instead=instead, shortfall=shortfall)
        self.need = need
        self.available = available
        self.short = short


class InputTooLarge(PlanRefused):
    """A picture whose own copies on the GPU don't fit (kind "input"), the input being copied
    there and resized (still_plan).

    - pixels: the input's pixels, as stored.
    - need, there: the copies' peak with the models and the margin, and what is there, in bytes.
    - `fits_input`: the largest input that fits, in pixels; None where the output's own copies
      are over the memory.
    - `phase`: the pass whose phase the copies are counted in ("encode" or "decode").
    - `instead`: the resolution that lowers the output's part of it, as a Setting; None where
      the message names none."""

    def __init__(
        self,
        message: Message,
        *,
        pixels: int | None = None,
        need: int | None = None,
        there: int | None = None,
        phase: str = "",
        instead: Setting | None = None,
        fits_input: int | None = None,
    ) -> None:
        super().__init__(message, kind="input", phase=phase, instead=instead, fits_input=fits_input)
        self.pixels = pixels
        self.need = need
        self.there = there


class CompileRefused(PlanRefused):
    """A job --compile can't run, within the shapes measured compiled (kind "compile"), where the
    plain plan of the same job fits.

    - `phase`: the VAE pass that has none ("encode" or "decode"), or the one the compiled plan
      first refuses.
    - `instead`: what the compiled plan's own refusal ends with, a tile that fits there; None
      where it names none.
    - within: the tile pins within the shapes measured that the compiled plan takes in place of
      the ones refused (encode_tile, decode_tile), as Settings, in the order of the passes; empty
      where the refusal advises none (a pass the pins don't give)."""

    def __init__(
        self,
        message: Message,
        *,
        phase: str = "",
        instead: Setting | None = None,
        within: tuple[Setting, ...] = (),
    ) -> None:
        super().__init__(message, kind="compile", phase=phase, instead=instead)
        self.within = within


class ModelError(Refusal):
    """A model file seedvr2x refuses: the message says what the file is, or what is wrong with
    it. The base of the kinds below, each of which a node or an application may word on its own;
    str() is the library's own words (`plain`), for its logs, and names no option, though the
    path of the file stays in it. The facts, all keyword-only and empty or None where a refusal
    has none:

    - role: what the file was asked to be, "dit", "vae" or "detector", "" if no role applies;
    - file: the file's name, never its path;
    - figure: what `expected` and `found` count, "size" (bytes) or "sha256" (hex);
    - expected, found: the figure the pin, a record or the file's own header gives, and the one
      read;
    - settings: the settings the message names as the way out (model_dir, dit_model...);
    - command: the shell command that fetches the file again, where the message gives one (the
      library's own tool, `hf download ...`, not a setting)."""

    KIND = ""

    def __init__(
        self,
        message: Message,
        *,
        kind: str = "",
        role: str = "",
        file: str = "",
        figure: str = "",
        expected: int | str | None = None,
        found: int | str | None = None,
        settings: tuple[Setting, ...] = (),
        command: str = "",
    ) -> None:
        super().__init__(message, kind=kind or self.KIND)
        self.role = role
        self.file = file
        self.figure = figure
        self.expected = expected
        self.found = found
        self.settings = settings
        self.command = command


class ModelMissing(ModelError):
    """A model file that is not there (kind "missing"): no such file in the directory given."""

    KIND = "missing"


class ModelUnknown(ModelError):
    """A file that is not one of seedvr2x's own, or not in the role it was given (kind
    "unknown"): `found` says what it is, in words, where the header tells; `role` is the role it
    was given."""

    KIND = "unknown"


class ModelDamaged(ModelError):
    """A model file cut short, empty, or whose header or bytes don't parse (kind "damaged"):
    `figure` "size", `found` the bytes it has and `expected` the bytes it needs, where it is cut
    short and the header says how many."""

    KIND = "damaged"


class ModelChanged(ModelError):
    """A model file that changed after its SHA-256 was read or while it loaded (kind "changed"):
    `figure` "sha256", `expected` the one read before or recorded, `found` the one its load
    read, "" where only its stat changed."""

    KIND = "changed"


class ModelNotPinned(ModelError):
    """A file named like one of seedvr2x's own whose size or SHA-256 is not its pin's (kind
    "not-pinned"): `figure` says which, `expected` the pin's, `found` the file's. `command` is
    the command that fetches it again into Hugging Face's cache, "" for a directory given."""

    KIND = "not-pinned"


class ModelUnreachable(ModelError):
    """A model file that can't be had from the Hub (kind "unreachable"): `why` says what stops
    it, "offline" (the environment keeps the network out), "unreachable" (the Hub didn't
    answer), "refused" (the Hub refuses the file to anyone), "not-the-hub" (the answer isn't a
    Hub's), "not-found" (the pinned revision lacks it) or "failed" (the download failed)."""

    KIND = "unreachable"

    def __init__(self, message: Message, *, why: str = "", **facts: Any) -> None:
        super().__init__(message, **facts)
        self.why = why


class ModelNoRoom(ModelError):
    """A download that doesn't fit the disk of Hugging Face's cache (kind "no-room"): `need` the
    bytes the downloads take, `free` the bytes free there, None where the cache can't be written
    for another reason."""

    KIND = "no-room"

    def __init__(
        self, message: Message, *, need: int | None = None, free: int | None = None, **facts: Any
    ) -> None:
        super().__init__(message, **facts)
        self.need = need
        self.free = free


class ModelUnreadable(ModelError):
    """A model file or the cache that holds it that the operating system won't read or write
    (kind "unreadable"): permissions, an I/O error, a path that is no regular file."""

    KIND = "unreadable"


class ModelsRefused(ModelError):
    """Several model files refused at once (kind "several"): `parts`, each refusal in the order
    of the files, whose words are the message's lines."""

    KIND = "several"

    def __init__(self, parts: Sequence[ModelError], message: Message | None = None) -> None:
        self.parts = tuple(parts)

        def lines(say: Say) -> str:
            return "\n".join(part.words(say) for part in self.parts)

        super().__init__(lines if message is None else message)


class HostMemoryError(Refusal):
    """The host can't hold the pinned copies a plan needs (kind "host-memory"): the message names
    the RAM they take. `need`: what was asked to be pinned, `available`: what the host has
    available, both in bytes, None where the host refused the mapping or the pinning of one
    buffer with memory to spare."""

    def __init__(
        self, message: Message, *, need: int | None = None, available: int | None = None
    ) -> None:
        super().__init__(message, kind="host-memory")
        self.need = need
        self.available = available


class PictureRefused(Refusal):
    """A picture seedvr2x refuses to read (kind in picture.KINDS), with the parts an application
    words its own way out by (Refusal.kind, reason, way). Facts besides, None where a kind has
    none: `pixels`, a picture over the limit's pixel count ("too-large"); `frames`, the frames of
    a picture that is several ("frames")."""

    def __init__(
        self,
        message: Message,
        *,
        kind: str = "",
        reason: str = "",
        way: str = "",
        pixels: int | None = None,
        frames: int | None = None,
    ) -> None:
        super().__init__(message, kind=kind, reason=reason, way=way)
        self.pixels = pixels
        self.frames = frames


class TargetTooSmall(Refusal):
    """The frames resized to the target are too small for the padding (kind "too-small"): the
    resolution asked, the size of the frames it gives and the least the padding takes; the
    resolution is the way out, as a Setting (`resolution`) the message words as an option.

    - resolution: the short side asked, in pixels;
    - size: (height, width) of the resized frames, in pixels;
    - least: (rows, columns) the padding takes at least, in pixels."""

    def __init__(
        self,
        message: Message,
        *,
        reason: str,
        resolution: int,
        size: tuple[int, int],
        least: tuple[int, int],
    ) -> None:
        super().__init__(
            message,
            kind="too-small",
            reason=reason,
            way=plain(Setting("resolution", str(resolution))),
        )
        self.resolution = resolution
        self.size = size
        self.least = least

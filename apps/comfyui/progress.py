"""What the nodes show while they run (DESIGN.md, Repository and apps, the ComfyUI nodes), in what
ComfyUI v0.44.0 (frontend 1.57.6) gives a node, read in its sources:

- One bar a node: comfy.utils.ProgressBar(total) and update_absolute(value, total, preview). The
  hook ComfyUI installs (main.py) sends the value to the frontend, which draws the node's bar; the
  value may go up to the total and the total may change. A bar that gets a value lower than its
  last is throttled out (utils.py: a 0.5% step and 100 ms are needed), so a bar that starts again
  at 0 is a new ProgressBar.
- One text box on the node: PromptServer.send_progress_text(text, node_id), replaced at each
  message and kept until the job ends. The queue's overall bar counts nodes, which nothing here
  drives.
- One preview a node, the newest: the third argument of update_absolute, a tuple (format, a PIL
  image, the largest side), as the sampler nodes send it (latent_preview.py), which the server
  shrinks to fit that side (server.py, send_image). The side is ComfyUI's --preview-size
  (comfy.cli_args, 512 by default): the limit is kept, never lifted.
(Described, not copied: ComfyUI is GPL-3.0.)

The bar is the upscale's expected time, not a count of steps (Progress): the units of the plan,
every tile of the encode, every window of the DiT and every tile of the decode of every shot or
picture, each weighing what it should take by its size. Before the upscale, the download of
seedvr2x's files and their reading into memory share a bar of their own, in bytes (Preparing), and
the Cuts node's by frames scanned (Preparing for the detector's file, then a ProgressBar by frame).

Everything here observes the library's callbacks and says what they tell: it moves no value of the
upscale. The things it does besides are ComfyUI's Cancel, checked at each event and at each poll of
a wait on the GPU, the time said each second between events (Ticker), and the units named for the
waits on the GPU, so that one that lasts past its bound is said on the node (Progress.running)."""

import logging
import threading
import time
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, override

import comfy.model_management as management
import numpy as np
import numpy.typing as npt
from comfy.utils import ProgressBar

from seedvr2x import frames as frame_sizes
from seedvr2x import pull, waiting
from seedvr2x.clock import spoken, taking, wall
from seedvr2x.passes import Tiles
from seedvr2x.shotruns import shot_runs
from seedvr2x.shots import Shot, shot_layout
from seedvr2x.tilegrid import SCALE, grid, latent_size

logger = logging.getLogger("seedvr2x_comfyui.progress")

# The upscale's bar is this many thousandths of its expected time.
BAR = 1000

Kind = Literal["encode", "window", "decode"]

# The way out a wait on the GPU that lasts is said with, in a ComfyUI user's words (waiting.py;
# DESIGN.md, Allocator, the hang of 2026-11-06). Cancel leaves the wait, but work queued on a card
# that stopped answering can't be undone in the process: the card finishes it or not, and the next
# run asks it without waiting (models._settle).
WAY = (
    "Cancel leaves the wait; if the card does not answer after that, restart ComfyUI, since work"
    " queued on a stopped GPU can't be undone inside its process (a card that finishes it lets the"
    " next run go on)"
)

# What the node says while the models are given back to the card after a run an exception ended
# (models._freeing): that wait ends by itself on a card that answers, and a second Cancel leaves it
# (waiting.releasing) where it does not.
FREEING = (
    "freeing the card: waiting for the GPU to finish its last work; Cancel again to leave it,"
    " then run again once the card has finished, or restart ComfyUI"
)

# What the node says once the release was left by a second Cancel or ended in an error: the card
# is marked (waiting.abandoned): the next run asks the card, without waiting, whether it finished
# (models._settle), and is refused (words.card_busy) while it has not.
STOPPED = "the card has not finished its last work: run again once it has, or restart ComfyUI"

# What the node says once a CUDA error other than out of memory failed the card (waiting.fail):
# nothing on it is freed or run again in the process (models.session).
FAILED = "the graphics card failed: restart ComfyUI"

# How often the time on the node is said between two events, in seconds (the user, 2026-11-06).
TICK = 1.0

# The pixels a DiT token's side spans: the VAE's 8, then the DiT's patches of 2 latent pixels
# (vendor/configs_7b/main.yaml, patch_size [1, 2, 2]; apps/seedvr2x/timing.py's TOKEN).
TOKEN = 16

# ---- What a unit should take, to start with ------------------------------------------------------


@dataclass(frozen=True)
class Costs:
    """What a unit of each kind takes before the run has measured one, in seconds: an encode or a
    decode by the pixel of one frame (a tile's pixels, with its overlap, times the frames of the
    shot padded to 4n + 1), a window by the token. Each kind's own, measured once its first unit is
    made (Progress)."""

    encode: float
    window: float
    decode: float


# A clip on the RTX 5090, 1080p output (research/docs/compile.md, the VAE's passes, plain, the
# tiles of 1024 px and more: 0.202 to 0.230 s a frame to encode and 0.437 to 0.444 to decode; at
# 4K, 0.781 and 1.738). A unit is weighed by the padded frame (frames.padded_size), 1104 x 1920 =
# 2.12 million pixels at 1080p and 2192 x 3840 = 8.42 million at 4K, so those are 0.095 to 0.109
# and 0.206 to 0.209 s per million pixels at 1080p, 0.093 and 0.206 at 4K; the figures below, 0.10
# (inside them) and 0.21 (rounding them) are the figures kept. The DiT's 0.083 ms a token is flat
# across window lengths ("What a job gains"; planner-limits.md). Together they put a 1080p job's
# time at 26% in encodes, 21% in windows and 54% in decodes, where seedvr2x's own 90-s excerpt on
# that card spent 25.6, 20.2 and 54.2% (compile.md). The DiT's token is 0.23 to 0.45 ms on the
# 96 GB card in other sessions (planner-limits.md): the figures are this card's, and each run
# measures its own.
CLIP = Costs(encode=0.10e-6, window=0.083e-3, decode=0.21e-6)
# A single picture on the same card, 1920x1080 to 3840x2160, warm, FlashAttention 2 (research/docs/
# small-cards.md, "The 4K picture's time, phase by phase"): the encode 1.52 s, the DiT's step
# 2.60 s, the decode 3.10 s, over the padded frame of 2192 x 3840 pixels and its 137 x 240 tokens.
# A picture's VAE passes cost about 1.8 times a clip's frame per pixel (a frame alone, no causal
# memory to share), so they are their own.
PICTURE = Costs(
    encode=1.52 / (2192 * 3840),
    window=2.60 / ((2192 // TOKEN) * (3840 // TOKEN)),
    decode=3.10 / (2192 * 3840),
)


@dataclass
class Unit:
    """One unit of the plan: a tile of a VAE pass (its pixels times the frames), or a window of
    the DiT (its tokens). `seconds` is what it took, once it is made."""

    kind: Kind
    weight: float
    seconds: float | None = None


@dataclass
class Pass:
    """What one shot's run or one picture is made of, in the order the library runs it: its label
    ("Shot 2 of 4 (frames 121 to 240)"), its frames, and its units of each kind. Tile-less
    passes (the untiled ones) have one unit of their kind."""

    label: str
    frames: int
    encode: list[Unit]
    windows: list[Unit]
    decode: list[Unit]
    # A clip's pass is a run of a shot (shotruns): the shot's place in the clip and the run's in
    # the shot, both from 1 as a user counts, and what Progress.run looks it up by.
    shot: int = 1
    run: int = 1

    @property
    def units(self) -> list[Unit]:
        return [*self.encode, *self.windows, *self.decode]


def _vae_units(kind: Kind, tiles: Tiles | None, padded: tuple[int, int], frames: int) -> list[Unit]:
    """The units of a VAE pass over `frames` frames (padded to 4n + 1) of the padded size
    (height, width): a tile each of the grid the library lays (tilegrid.grid), by its pixels with
    its overlap; one unit, the whole frame, for an untiled pass or a grid of one tile."""
    cells = [] if tiles is None else grid(latent_size(padded), tiles)
    if len(cells) < 2:
        return [Unit(kind, padded[0] * padded[1] * frames)]
    return [Unit(kind, cell.rows * cell.columns * SCALE**2 * frames) for cell in cells]


def make_pass(
    label: str,
    frames: int,
    target: tuple[int, int],
    window: int | None,
    encode_tiles: Tiles | None,
    decode_tiles: Tiles | None,
) -> Pass:
    """The units of a shot of `frames` frames resized to `target` (height, width), in DiT windows
    of at most `window` latents (None: one window), its VAE passes in the given tiles: the plan's
    (clip.upscale's arguments, or a picture's, one frame)."""
    padded = frame_sizes.padded_size(target)
    length = frame_sizes.padded_length(frames)
    tokens = (padded[0] // TOKEN) * (padded[1] // TOKEN)
    return Pass(
        label,
        frames,
        _vae_units("encode", encode_tiles, padded, length),
        [Unit("window", (end - start) * tokens) for start, end in shot_layout(frames, window)],
        _vae_units("decode", decode_tiles, padded, length),
    )


def clip_passes(
    shots: Sequence[Shot],
    target: tuple[int, int],
    window: int | None,
    encode_tiles: Tiles | None,
    decode_tiles: Tiles | None,
    runs: int = 1,
) -> list[Pass]:
    """The passes of a clip cut into `shots`, a pass a run of a shot (`runs` a shot, shotruns:
    1, 2 or 4; a shot too short for a run has no such run): a run is a pass over the shot from its
    first frame, so it has its own windows and tiles and its frames are the shot's less its offset.
    Labelled "Shot 2 of 4 (frames 121 to 240)", the frames counted from 1 as a user counts them,
    or "Frames 1 to 480" for a clip of one shot; with several runs the frames are the run's own,
    from its first: "Shot 2 of 4, run 3 of 4 (frames 123 to 240)" and "Frames 3 to 480, run 2 of
    2". A shot of one run has no run part."""
    passes: list[Pass] = []
    for number, shot in enumerate(shots, 1):
        each_run = shot_runs(Shot(0, shot.frames), runs)
        for index, each in enumerate(each_run, 1):
            part = f", run {index} of {len(each_run)}" if len(each_run) > 1 else ""
            # The run's own frames: it starts at its offset in the shot, and so does its decode.
            first = shot.start + each.offset + 1
            if len(shots) > 1:
                label = f"Shot {number} of {len(shots)}{part} (frames {first} to {shot.end})"
            else:
                label = f"Frames {first} to {shot.end}{part}"
            made = make_pass(
                label, shot.frames - each.offset, target, window, encode_tiles, decode_tiles
            )
            made.shot, made.run = number, index
            passes.append(made)
    return passes


def picture_passes(
    count: int, target: tuple[int, int], encode_tiles: Tiles | None, decode_tiles: Tiles | None
) -> list[Pass]:
    """The passes of a batch of `count` pictures, each of one frame and one window:
    "Picture 3 of 8", or "Picture" for one."""
    return [
        make_pass(
            f"Picture {number} of {count}" if count > 1 else "Picture",
            1,
            target,
            None,
            encode_tiles,
            decode_tiles,
        )
        for number in range(1, count + 1)
    ]


# ---- A preview -----------------------------------------------------------------------------------


def preview_of(frame: npt.NDArray[np.float32]) -> tuple[str, Any, int]:
    """The preview ComfyUI sends for `frame`, (H, W, 3) float32 in [0, 1]: its format, a PIL image
    of its 8-bit codes shrunk to fit ComfyUI's preview size (comfy.cli_args' preview_size), and that
    size, which the server fits it to again. The frame is shrunk here, before it is made a JPEG:
    averaged in whole blocks down to between one and two times the size, then fitted by PIL's
    bilinear filter, so that the server's event loop gets no 25 MB image from a 4K frame. A frame
    already within the size is sent at its own (the server's fit would enlarge it). The block
    average reads a view of the frame where its sides are multiples of the block; where they are
    not, NumPy copies the cropped frame (float32, up to the frame's size) to reshape it."""
    from comfy.cli_args import args
    from PIL import Image, ImageOps

    side = int(args.preview_size)
    height, width = frame.shape[0], frame.shape[1]
    block = max(1, max(height, width) // (2 * side))
    if block > 1:
        rows, columns = height // block * block, width // block * block
        frame = frame[:rows, :columns].reshape(rows // block, block, columns // block, block, 3)
        frame = frame.mean(axis=(1, 3), dtype=np.float32)
    codes = np.rint(np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
    image = Image.fromarray(codes)
    if max(image.size) > side:
        image = ImageOps.contain(image, (side, side), Image.Resampling.BILINEAR)
    return ("JPEG", image, side)


# ---- The upscale's bar, text and previews --------------------------------------------------------


class Progress:
    """An upscale's bar, text box and previews over its `passes` (make_pass), moved by the library's
    events: the node tells it where the run is (begin, tile, encoded, window, decoded) and it says.

    The bar. Its value is the share of the run's expected time that is done, in thousandths of
    BAR: the seconds the units made took, over those and the estimate of the units left. A unit's
    estimate is its weight at its kind's rate, which starts at `costs` and is the run's own once a
    unit of the kind is made: the total of its units' seconds over the total of their weights.
    So the rest of the bar is spread again over what remains when a kind turns out slower or
    faster than the card's figures, and the bar never goes back: a rate that moves the estimate
    up leaves the bar where it is until the units catch up. It is full at the last unit, and only
    there. A unit's time is the time since the one before ended (the first from `begin`), the
    changes of phase and the copies between units included in the unit that follows them.

    A tiled pass tells its tiles (done of total); the last tile is made when the pass is: the
    pass's tail (the sums divided, the slices written and corrected) belongs to it. An untiled
    decode moves the bar by the frames written, the others by their unit.

    The text, replaced at each message: the plan's lines (`plan`), a status line saying which pass
    and how far it has got, a time line (so far, and the time left once the first window has been
    measured, which says the DiT's rate too). The preview: the newest frame decoded, once per
    slice (`decoded`).

    Between events the time moves on: `running` starts a Ticker, a thread that says the text again
    each second (`tick`), so that a long window's text never stands still. The node's thread and
    the ticker's share the state under a lock; only the node's thread moves the bar.

    The waits on the GPU. Each event names the unit that follows it for the library's waits
    (waiting.unit), with the estimate that unit should take once a unit of its kind is measured,
    and none before: a wait of a unit with no estimate says nothing. A wait past its bound is
    said by the library on ComfyUI's console, and by the text here (`tick`) with the way out
    (WAY). `running` also gives the waits ComfyUI's Cancel as their check, so that a wait is left
    by Cancel.

    `now` is the clock, injected for the tests; `show(text)` puts the text on the node."""

    def __init__(
        self,
        passes: Sequence[Pass],
        costs: Costs,
        bar: ProgressBar,
        show: Callable[[str], None],
        plan: str = "",
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.passes = list(passes)
        self.costs = costs
        self.bar = bar
        self.show = show
        self.plan = plan
        self.now = now
        self.started = self.mark = now()
        self.value = 0
        self.pass_ = 0
        self.what = ""
        # The untiled decode's frames written, as a share of its unit.
        self.partial = 0.0
        self.units = [unit for each in self.passes for unit in each.units]
        # (seconds, weight) of the units made, by kind.
        self.made: dict[Kind, list[float]] = {"encode": [0, 0], "window": [0, 0], "decode": [0, 0]}
        self.mismatch = False
        # Held by an event and by a tick: the units' times and rates, the pass and the words are
        # read by the ticker's thread while the node's moves them.
        self._lock = threading.RLock()
        self.finished = False
        # Set when the run's block ends (running), however it ends: a tick still on its way, or
        # one after the join gave up, says nothing.
        self.stopped = False

    # -- The estimate ----------------------------------------------------------------------------

    def rate(self, kind: Kind) -> float:
        """Seconds per unit of weight for `kind`: the run's own once a unit is made, else
        `costs`'."""
        seconds, weight = self.made[kind]
        return seconds / weight if weight > 0 else getattr(self.costs, kind)

    def left(self) -> float:
        """The expected seconds of the units not made yet, the untiled decode's frames written
        taken off."""
        total = 0.0
        for each in self.units:
            if each.seconds is None:
                total += each.weight * self.rate(each.kind)
        return total - self._partial_seconds()

    def _partial_seconds(self) -> float:
        current = self._current_decode()
        return 0.0 if current is None else self.partial * current.weight * self.rate("decode")

    def _current_decode(self) -> Unit | None:
        """The untiled decode being written, a pass whose decode is one unit not made."""
        if self.pass_ >= len(self.passes):
            return None
        decode = self.passes[self.pass_].decode
        return decode[0] if len(decode) == 1 and decode[0].seconds is None else None

    def done(self) -> float:
        return sum(each.seconds or 0.0 for each in self.units) + self._partial_seconds()

    def position(self) -> int:
        """The bar's value, thousandths of BAR: never less than it was, BAR only when every unit
        is made."""
        if all(each.seconds is not None for each in self.units):
            self.value = BAR
            return BAR
        done, left = self.done(), self.left()
        value = int(BAR * done / (done + left)) if done + left > 0 else 0
        self.value = max(self.value, min(value, BAR - 1))
        return self.value

    # -- The run's events ------------------------------------------------------------------------

    def begin(self, index: int) -> None:
        """Pass `index` (0-based) starts: its encode."""
        with self._lock:
            self.pass_ = index
            self.partial = 0.0
            self._say("encoding")

    def run(self, shot: int, run: int) -> None:
        """Run `run` (1-based) of shot `shot` (0-based) starts: its encode. A shot of one run
        begins by this too, its run 1."""
        for index, each in enumerate(self.passes):
            if (each.shot, each.run) == (shot + 1, run):
                self.begin(index)
                return
        raise ValueError(f"no run {run} of shot {shot + 1} in the progress's passes")

    def tile(self, pass_: Literal["encode", "decode"], done: int, total: int) -> None:
        """A tiled pass has made `done` of its `total` tiles (0 before the first). The last is
        made with the pass (encoded, decoded)."""
        with self._lock:
            self._tile(pass_, done, total)

    def _tile(self, pass_: Literal["encode", "decode"], done: int, total: int) -> None:
        units = getattr(self.passes[self.pass_], "encode" if pass_ == "encode" else "decode")
        if total != len(units) and not self.mismatch:
            self.mismatch = True
            logger.warning(
                "the %s has %d tiles, the progress counted %d: its bar is less exact",
                pass_,
                total,
                len(units),
            )
        self._finish(units, min(done, len(units) - 1) if done < total else 0)
        word = "encoding" if pass_ == "encode" else "decoding"
        self._say(f"{word}, tile {min(done + 1, total)} of {total}")

    def encoded(self) -> None:
        """The pass's encode is made."""
        with self._lock:
            current = self.passes[self.pass_]
            self._finish(current.encode, len(current.encode))
            self._say(self._window_text(0))

    def window(self, number: int) -> None:
        """Window `number` (1-based) of the pass is made."""
        with self._lock:
            self._window(number)

    def _window(self, number: int) -> None:
        current = self.passes[self.pass_]
        self._finish(current.windows, number)
        if number < len(current.windows):
            self._say(self._window_text(number))
            return
        decode = current.decode
        tiled = len(decode) > 1
        self._say(
            f"decoding, tile 1 of {len(decode)}" if tiled else self._decoding(0, current.frames)
        )

    def decoded(
        self, written: int, total: int, frame: npt.NDArray[np.float32] | None = None
    ) -> None:
        """`written` of the pass's `total` frames are decoded and written (a slice has been);
        `frame` is the newest, the preview. The last makes the pass."""
        with self._lock:
            self._decoded(written, total, frame)

    def _decoded(
        self, written: int, total: int, frame: npt.NDArray[np.float32] | None = None
    ) -> None:
        current = self.passes[self.pass_]
        if written >= total:
            self.partial = 0.0
            self._finish(current.decode, len(current.decode))
            self._say(self._decoding(total, total), frame)
            return
        if len(current.decode) == 1:
            self.partial = written / total
        self._say(self._decoding(written, total), frame)

    @staticmethod
    def _decoding(written: int, total: int) -> str:
        """The decode's status: the frames written of the pass's, none said for a pass of one
        frame (a picture)."""
        return "decoding" if total == 1 else f"decoding, {written} of {total} frames"

    def finish(self, what: str) -> None:
        """The run is over: the bar full, and the plan with `what` made and the time it took
        ("480 frames in 4 min 32 s") on the node. No tick says anything after it."""
        with self._lock:
            self.finished = True
            self.bar.update_absolute(BAR, BAR)
            summary = f"{what} in {spoken(wall(self.now() - self.started))}"
            self.show(f"{self.plan}\n{summary}" if self.plan else summary)

    # -- Words -----------------------------------------------------------------------------------

    def _window_text(self, made: int) -> str:
        count = len(self.passes[self.pass_].windows)
        return f"upscaling, window {made + 1} of {count}"

    def _finish(self, units: list[Unit], through: int) -> None:
        """The first `through` units of the list are made now, the unfinished ones among them
        taking the time since the last was."""
        pending = [each for each in units[:through] if each.seconds is None]
        if not pending:
            return
        moment = self.now()
        # One event can make several (a pass told its tail at once): the time is shared by weight.
        weight = sum(each.weight for each in pending)
        for each in pending:
            each.seconds = (moment - self.mark) * each.weight / weight if weight > 0 else 0.0
            made = self.made[each.kind]
            made[0] += each.seconds
            made[1] += each.weight
        self.mark = moment

    def _say(self, what: str, frame: npt.NDArray[np.float32] | None = None) -> None:
        """The status, the bar and the preview: sent, then ComfyUI's Cancel honoured."""
        self.what = what
        self.show(self._text())
        self.bar.update_absolute(self.position(), BAR, None if frame is None else preview_of(frame))
        self._name_unit()
        management.throw_exception_if_processing_interrupted()

    def _text(self) -> str:
        """The node's text: the plan, the pass and where it is, the time, and a wait on the GPU
        past its bound where there is one."""
        label = self.passes[self.pass_].label
        lines = [*([self.plan] if self.plan else []), f"{label}: {self.what}", self._time()]
        lasted = waiting.longest() if waiting.stalled() else None
        if lasted is not None:
            lines.append(
                f"A wait on the GPU has lasted {spoken(wall(lasted))}, much longer than this step"
                f" should take: {WAY}."
            )
        return "\n".join(lines)

    def tick(self) -> None:
        """The text again, the time moved on: said by the Ticker between two events. Nothing
        before the first event, none after the end, and the bar is never touched."""
        with self._lock:
            if self.finished or self.stopped or not self.what:
                return
            self.show(self._text())

    def _name_unit(self) -> None:
        """Name the unit that follows the event for the waits on the GPU of the node's thread
        (waiting.unit): the pass's first unit not made, with the time it should take at its
        kind's rate once a unit of the kind is measured (rate), none before: a wait of a unit
        with no estimate says nothing. The unit is a tile, a window or an untiled decode, the
        wait a part of it (waiting.py)."""
        current = self.passes[self.pass_]
        name = f"{current.label}: {self.what}"
        coming = next((each for each in current.units if each.seconds is None), None)
        estimate = None
        if coming is not None and self.made[coming.kind][1] > 0:
            estimate = coming.weight * self.rate(coming.kind)
        waiting.unit(name, estimate, WAY)

    @contextmanager
    def running(
        self, period: float = TICK, pause: Callable[[threading.Event, float], None] | None = None
    ) -> Generator[None]:
        """The upscale's run, in the with block: the Ticker saying the time each `period` seconds
        (`pause(stopped, period)` waits that long, or until `stopped` is set: the tests'
        hook), ComfyUI's Cancel the check of every wait on the GPU the node's thread makes
        (waiting.check), and the units named for those waits (_name_unit). At the block's end,
        however it ends, the ticker is stopped and joined and the unit forgotten."""
        ticker = Ticker(self, period, pause)
        self.stopped = False
        ticker.start()
        try:
            with waiting.checking(management.throw_exception_if_processing_interrupted):
                yield
        finally:
            with self._lock:
                self.stopped = True
            ticker.stop()
            waiting.clear()

    def _time(self) -> str:
        """The time line: the time so far, and the time left once a window has been measured."""
        text = f"{spoken(wall(self.now() - self.started))} so far"
        if self.made["window"][1] > 0:
            text += f", {taking(wall(self.left()))} left"
        return text


def _wait(stopped: threading.Event, seconds: float) -> None:
    stopped.wait(seconds)


class Ticker:
    """The thread that says the time on the node between the library's events: every `period`
    seconds it has the Progress say its text again (tick), nothing else. It never touches the bar
    (which only the node's thread moves, event by event), and it reads the Progress's clock, so
    that its text is the events' own. Started and stopped by Progress.running, at the run's end,
    on Cancel and on an error alike.

    A daemon thread, joined at the stop with no time cap: the join is what leaves nothing
    running. It ends at once at the stop, since it waits on an event the stop sets, not on a
    sleep, and it sends nothing that can block: ComfyUI's text send only queues the message for
    its event loop, which the loop sends later. A cap on the join would be a timeout standing in
    for that method. It stays a daemon because it holds no model, no tensor and no GPU call, so
    that it has nothing to finish at a shutdown, whatever else ends the process.

    `pause(stopped, period)` waits for the period or until `stopped` is set; the tests replace it
    to tick on their own cue, the default waits on the event."""

    def __init__(
        self,
        progress: Progress,
        period: float = TICK,
        pause: Callable[[threading.Event, float], None] | None = None,
    ) -> None:
        self.progress = progress
        self.period = period
        self.pause = pause or _wait
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, name="seedvr2x-progress-text", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        """No tick from now on, and the thread gone (idempotent)."""
        self.stopped.set()
        if not self.thread.is_alive():
            return
        self.thread.join()

    def _run(self) -> None:
        while True:
            self.pause(self.stopped, self.period)
            if self.stopped.is_set():
                return
            try:
                self.progress.tick()
            except Exception as error:
                # Only the text: the run goes on without the seconds between events.
                logger.warning("the time can't be said between events: %r", error)
                return


# ---- Before the upscale: files downloaded and read ----------------------------------------------


def amount(done: int, total: int) -> str:
    """Bytes done of total, in the unit of the total: "4.2 of 16.5 GB", "12.1 of 30.5 MB"."""
    if total >= 100_000_000:
        return f"{done / 1e9:.1f} of {total / 1e9:.1f} GB"
    return f"{done / 1e6:.1f} of {total / 1e6:.1f} MB"


def amount_short(done: int, total: int) -> str:
    """The same in the form the Models node's one short line takes: "3.1/8.3 GB", "12.1/30.5 MB"."""
    if total >= 100_000_000:
        return f"{done / 1e9:.1f}/{total / 1e9:.1f} GB"
    return f"{done / 1e6:.1f}/{total / 1e6:.1f} MB"


@dataclass
class Preparing:
    """The download of seedvr2x's files and their reading into memory on one bar, in bytes, each
    with its line ("downloading the sharp 7B (float16): 4.2 of 16.5 GB", "reading the VAE into
    memory: 0.1 of 0.5 GB"). It is told by pull.fetch (download) and by the load (read). The bar's
    total is the downloads' and the reading's together and does not change once the first byte
    is told: will_read announces the reading's before the first download (the session does, from
    the pins' sizes), and fetch tells every download's size before its first byte. It makes its
    bar when the first byte is told, so that a run with nothing to download and models kept in
    memory has none; the upscale's bar starts again at 0 and is another (Progress).

    Cancel. The hook ComfyUI installs on its bars checks the Cancel flag first, clears it and
    raises (main.py:467, model_management.py:2224-2230), so the bar's update raises on the node's
    thread, at each read and each download report that comes on it. huggingface_hub's Xet
    downloads report from a thread of hf_xet's own, which catches what a report raises and goes
    on: a report from another thread holds the flag's lock and moves the bar and the text only if
    Cancel is not pressed, so that the flag stays for the node's thread, which checks it when the
    download returns (models._prepare)."""

    show: Callable[[str], None]
    labels: Callable[[str], str]
    now: Callable[[], float] = time.monotonic
    bar: ProgressBar | None = None
    downloads: dict[str, tuple[int, int]] = field(default_factory=dict[str, tuple[int, int]])
    reads: dict[str, tuple[int, int]] = field(default_factory=dict[str, tuple[int, int]])
    announced: int = 0
    started: float = field(init=False)
    thread: int = field(init=False)

    def __post_init__(self) -> None:
        self.started = self.now()
        self.thread = threading.get_ident()

    def will_read(self, total: int) -> None:
        """The bytes the load is to read, all its files'."""
        self.announced = total

    def download(self, name: str, done: int, total: int) -> None:
        self.downloads[name] = (done, total)
        self._tell(f"downloading {self.labels(name)}: {amount(done, total)}")

    def read(self, name: str, done: int, total: int) -> None:
        self.reads[name] = (done, total)
        self._tell(f"reading {self.labels(name)} into memory: {amount(done, total)}")

    def _tell(self, line: str) -> None:
        if threading.get_ident() == self.thread:
            self._send(line)
            management.throw_exception_if_processing_interrupted()
            return
        with management.interrupt_processing_mutex:
            if not management.processing_interrupted():
                self._send(line)

    def _send(self, line: str) -> None:
        self.show(self._text(line))
        self._move()

    def _text(self, line: str) -> str:
        return f"{line}\n{spoken(wall(self.now() - self.started))} so far"

    def _move(self) -> None:
        total = sum(each[1] for each in self.downloads.values())
        total += max(self.announced, sum(each[1] for each in self.reads.values()))
        done = sum(each[0] for each in self.downloads.values())
        done += sum(each[0] for each in self.reads.values())
        if self.bar is None:
            # Made at the first byte, when every download's size has been told (fetch tells them
            # all before the first) and the reading's is announced: the total holds from there.
            if done == 0:
                return
            self.bar = ProgressBar(total)
        self.bar.update_absolute(done, total)


@dataclass
class Downloading(Preparing):
    """The Models node's downloads: Preparing's bar and Cancel, in bytes, with the one short line
    the node shows ("downloading 3.1/8.3 GB": every download together, in the total's unit) in
    place of a line for each file and the time, which the node's line has no room for (about 40
    characters, the user, 2026-11-08). Nothing is read into memory here. `expected`: the sizes of
    the files about to be downloaded, by name (their pins'), so that the line's total is whole
    from the first, before pull.fetch has told every size."""

    expected: dict[str, int] = field(default_factory=dict[str, int])

    @override
    def download(self, name: str, done: int, total: int) -> None:
        self.downloads[name] = (done, total)
        sizes = {**self.expected, **{each: size for each, (_, size) in self.downloads.items()}}
        self._tell(
            "downloading "
            + amount_short(sum(each[0] for each in self.downloads.values()), sum(sizes.values()))
        )

    @override
    def _text(self, line: str) -> str:
        return line


def scanned(
    bar: ProgressBar, show: Callable[[str], None], now: Callable[[], float] = time.monotonic
) -> Callable[[int, int], None]:
    """The Cuts node's progress, a callback (frames scanned, frames) moving `bar` by frame and
    saying it on the node, ComfyUI's Cancel honoured after each batch."""
    started = now()

    def tell(done: int, total: int) -> None:
        bar.update_absolute(done, total)
        show(f"looking for cuts: {done} of {total} frames\n{spoken(wall(now() - started))} so far")
        management.throw_exception_if_processing_interrupted()

    return tell


def file_size(path: Path) -> int:
    """A model file's size for will_read: its pin's where it is one of seedvr2x's own (the cache
    holds nothing else under those names), else the file's, 0 where it isn't there (the total
    is then the sizes the load tells)."""
    if path.name in pull.PINNED:
        return pull.PINNED[path.name].size
    return path.stat().st_size if path.is_file() else 0

"""The server's GPU side (DESIGN.md, The image server): its start, from the model files' check to
the warm-up, and a picture's run, from its bytes read in its request's thread to its file
written there. The models are loaded once and kept; the library's bricks do the work
(picture, still_plan, still, png, webp, record), assembled here as the CLI's stills.py assembles
them for one picture.

Several pictures at once share the models. The DiT takes several samples in one pass
(vendor/models/dit_7b/nadit.py packs the samples of the lists runner.inference is given, each its
own shape), but the samples' bits in a pass of several are unchecked against their pass alone,
and a picture's answer must be the one its seed makes again whatever else runs. CUDA streams
would run two pictures in one model at once, and the VAE's causal convolutions keep their memory
in the modules (vendor/models/video_vae_v3/modules/causal_inflation_lib.py, `memory`), which two
passes at once would mix. So each model takes one picture at a time, a lock each (the VAE's for
the encode and the decode, the DiT's for its pass), and the pictures admitted at once overlap
across the models and their work on the CPU: reading, the tiles' blending in system memory, the
copies, the file's writing. Their bits are those of the picture alone: the same kernels on the
same shapes, nothing shared written (tests/test_service.py on the CPU; the GPU window's run on
the card). What several at once gain is measured on the card at 1, 2 and 4 before a default
other than 1 is set (DESIGN.md, Several pictures at once)."""

import io
import logging
import math
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import numpy.typing as npt

import admission
import answers
import reservation
from metrics import Gauges, Metrics
from seedvr2x import blockswap, frames, memory, peaks, picture, png, pull, still_plan, webp
from seedvr2x import record as records
from seedvr2x.errors import Refusal
from seedvr2x.memory import Reading
from seedvr2x.passes import ModelPlan
from seedvr2x.pull import ModelFiles
from seedvr2x.still_plan import Bound, Pins
from seedvr2x.weights import Architecture

if TYPE_CHECKING:
    from seedvr2x.model import Models

logger = logging.getLogger("imgserver.service")

APP = "seedvr2x-imgserver"
GIB = peaks.GIB


class State(Enum):
    """The server's state, as /health says it."""

    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    FAILED = "failed"


class Unprocessable(Exception):
    """A picture refused or over the server's limits: answered 422 with the message. `kind` is
    the library's kind of the refusal (picture.KINDS) where there is one, which the answer's
    headers follow: Accept-Post for "not-a-picture"."""

    def __init__(self, message: str, kind: str | None = None) -> None:
        super().__init__(message)
        self.kind = kind


class Failed(Exception):
    """A picture that failed on the GPU: answered 500 with the message, the server still running
    unless `fatal` (a CUDA error other than out of memory, the card's state no longer trusted)."""

    def __init__(self, message: str, fatal: bool = False) -> None:
        super().__init__(message)
        self.fatal = fatal


class Unavailable(Exception):
    """The server not taking pictures: starting, stopping or failed, or its queue full. Answered
    503, with Retry-After where waiting helps."""

    def __init__(self, message: str, retry: bool = True) -> None:
        super().__init__(message)
        self.retry = retry


class Machine(Protocol):
    """What the start reads of the machine and does on its card: the GPU's, the real one (Card);
    the tests' stand-in on the CPU."""

    def device_name(self) -> str: ...
    def free(self) -> int: ...
    def reading(self, free: int) -> Reading: ...
    def host(self) -> int | None: ...
    def attention(self) -> str: ...
    def dit_model(self, path: Path) -> Architecture: ...
    def load(self, files: ModelFiles, plan: ModelPlan, reserve: int) -> "Models": ...
    def reserve(self, size: int) -> None: ...
    def reserved(self) -> int: ...
    def reserved_peak(self) -> int: ...
    def out_of_memory(self) -> tuple[type[BaseException], ...]: ...
    def fatal(self, error: BaseException) -> bool: ...
    def watched(self, models: "Models") -> Any: ...


class Card:
    """The machine's first CUDA device, through torch: imported here, after main set the
    allocator (memory.startup)."""

    def __init__(self) -> None:
        import torch

        if not torch.cuda.is_available():
            raise Refusal("no CUDA device: the image server runs on an NVIDIA GPU")
        if not torch.cuda.is_bf16_supported():
            raise Refusal("the GPU doesn't compute in bfloat16, which the models run in")
        self.torch = torch
        self.device = torch.device("cuda", 0)
        # Where the driver spills, torch is held to the card first (DESIGN.md, Allocator).
        memory.hold(self.device, first=True)

    def device_name(self) -> str:
        return str(self.torch.cuda.get_device_name(self.device))

    def free(self) -> int:
        """The driver's free memory with a first allocation and the convolution's and the matrix
        product's libraries loaded, as the CLI reads it (apps/seedvr2x/main.py, _free_memory: the
        planner's margin was bisected on such readings)."""
        torch = self.torch
        first = torch.empty(1, device=self.device)
        volume = torch.zeros(1, 1, 3, 3, 3, dtype=torch.float32, device=self.device)
        convolved = torch.nn.functional.conv3d(volume, volume)
        matrix = torch.zeros(2, 2, dtype=torch.float32, device=self.device)
        product = torch.mm(matrix, matrix)
        free = int(torch.cuda.mem_get_info(self.device)[0])
        del first, convolved, product
        return free

    def reading(self, free: int) -> Reading:
        return memory.reading(free)

    def host(self) -> int | None:
        found = blockswap.available()
        if found is None:
            logger.warning("the system memory available is unread: the plan doesn't check it")
        return found

    def attention(self) -> str:
        from seedvr2x import model

        return model.attention_backend()

    def dit_model(self, path: Path) -> Architecture:
        from seedvr2x import weights

        return weights.check(path, "dit")

    def load(self, files: ModelFiles, plan: ModelPlan, reserve: int) -> "Models":
        from seedvr2x import model

        memory.hold(self.device, reserve=reserve)
        return model.load_models(files.dit.path, files.vae.path, self.device, files.hashes, plan)

    def reserve(self, size: int) -> None:
        """torch's allocator made to reserve `size` bytes in all, models included, kept there for
        the process and capped there, the card judged first where the driver spills
        (memory.keep): torch.OutOfMemoryError where the card can't give it. Its peak counted
        from there."""
        memory.keep(self.device, size)
        self.torch.cuda.reset_peak_memory_stats(self.device)

    def reserved(self) -> int:
        return int(self.torch.cuda.memory_reserved(self.device))

    def reserved_peak(self) -> int:
        return int(self.torch.cuda.max_memory_reserved(self.device))

    def out_of_memory(self) -> tuple[type[BaseException], ...]:
        return (self.torch.OutOfMemoryError,)

    def fatal(self, error: BaseException) -> bool:
        """A CUDA error other than out of memory: torch's AcceleratorError, or a RuntimeError
        CUDA, cuBLAS or cuDNN reported, after which the card's state is untrusted."""
        if isinstance(error, self.torch.OutOfMemoryError):
            return False
        accelerator = getattr(self.torch, "AcceleratorError", None)
        if accelerator is not None and isinstance(error, accelerator):
            return True
        text = str(error)
        # cuBLAS's and cuDNN's refused allocations are out of memory, not a broken card
        # (CUBLAS_STATUS_ALLOC_FAILED, CUDNN_STATUS_ALLOC_FAILED): the picture is answered 500 and
        # the server goes on (_upscaled).
        if "_ALLOC_FAILED" in text:
            return False
        return isinstance(error, RuntimeError) and any(
            word in text for word in ("CUDA error", "CUBLAS_STATUS", "cuDNN error")
        )

    def watched(self, models: "Models") -> Any:
        return memory.watched(models.device, "the picture's passes", "pixels")


@dataclass(frozen=True)
class Options:
    """The start's settings (main.py's options): the bound (long, short) of --max-size, the
    largest input (pixels), the pictures at once and waiting, the model files and the pins."""

    bound: Bound
    input_pixels: int
    count: int = 1
    queue: int = 8
    model_dir: Path | None = None
    dit: str = "seedvr2x_ema_7b_sharp_fp16.safetensors"
    vae: str = "seedvr2x_ema_vae_fp16.safetensors"
    pins: Pins = still_plan.NO_PINS


@dataclass(frozen=True)
class Prepared:
    """A picture read and checked in its request's thread, ready to wait for the GPU."""

    picture: picture.Picture
    target: tuple[int, int]
    seed: int
    resolution: int
    format: str  # answers.PNG or answers.WEBP
    need: int  # its share of the pool, bytes (reservation.share of its peaks)
    tiles: still_plan.PictureTiles


@dataclass(frozen=True)
class Made:
    """A picture made: its file, its type, the seed it was made with, the source's size as
    stored and the output's as written, (width, height), and its wait and upscale, in seconds."""

    data: bytes
    content_type: str
    seed: int
    source: tuple[int, int]
    output: tuple[int, int]
    waited: float
    seconds: float


@dataclass
class Service:
    """The server's models, plan, reservation and state; `machine` the card's reads (Card)."""

    options: Options
    # The card's reads, made once the server listens (main.bring_up): its making imports torch
    # and opens the CUDA context, which takes seconds (DESIGN.md: it listens at once).
    machine: Machine | None
    metrics: Metrics = field(default_factory=Metrics)
    state: State = State.STARTING
    models: "Models | None" = None
    files: ModelFiles | None = None
    made: reservation.Made | None = None
    admitted: admission.Admission | None = None
    record_models: tuple[records.Model, records.Model] | None = None
    _vae: threading.Lock = field(default_factory=threading.Lock)
    _dit: threading.Lock = field(default_factory=threading.Lock)
    _state: threading.Lock = field(default_factory=threading.Lock)
    # Set where the state goes failed (a CUDA error other than out of memory): main.serve waits
    # on it beside the stop, then ends the process with main.FAILED_EXIT, so that a restart
    # policy acts (DESIGN.md, The image server, /health).
    failed: threading.Event = field(default_factory=threading.Event)
    # The error that set it, for main's last line.
    failure: str = ""

    # ---- The start ------------------------------------------------------------------------

    @property
    def card(self) -> Machine:
        """The card's reads: RuntimeError before they are made."""
        if self.machine is None:
            raise RuntimeError("the card isn't read yet: the server is starting")
        return self.machine

    def check_files(self) -> ModelFiles:
        """The model files found, pulled at the pinned revision where they are missing, and
        checked as the CLI checks them: their headers, then each pinned file's size and SHA-256,
        read once for the server's life (pull.Hashes); an unpinned file's SHA-256, which the
        record holds, read now too. ModelError, a Refusal, where one is refused."""
        from seedvr2x import weights

        options = self.options
        files = pull.resolve(options.model_dir, options.dit, options.vae, detector=False)
        dit, vae = (None if each.absent else each.path for each in (files.dit, files.vae))
        weights.check_models(dit, vae, pull.advice(files), None)
        pull.check_pinned(files)
        self.files = files
        self.record_models = (
            records.Model(options.dit, files.hashes.sha256(files.dit.path)),
            records.Model(options.vae, files.hashes.sha256(files.vae.path)),
        )
        return files

    def plan(self) -> reservation.Made:
        """The card read, the plan made for the bound and the reservation counted on it:
        ReservationRefused where it doesn't fit, naming what would."""
        assert self.files is not None
        options, machine = self.options, self.card
        logger.info("card: %s", machine.device_name())
        reading = machine.reading(machine.free())
        asked = reservation.Asked(
            options.bound,
            options.input_pixels,
            options.count,
            options.queue,
            reading,
            machine.host(),
            machine.attention(),
            machine.dit_model(self.files.dit.path),
            options.pins,
        )
        made = reservation.make(asked)
        for line in still_plan.said(made.plan):
            logger.info("%s", line)
        self.made = made
        return made

    def load(self) -> "Models":
        """The models loaded as the plan places them, then the reservation made: the share torch
        doesn't hold yet allocated and freed, and torch held there."""
        assert self.files is not None and self.made is not None
        started = time.monotonic()
        made = self.made
        self.models = self.card.load(self.files, made.plan.models, made.plan.reading.reserve)
        logger.info(
            "models loaded in %.1f s, attention: %s",
            time.monotonic() - started,
            self.models.attention,
        )
        try:
            self.card.reserve(made.reservation)
        except self.card.out_of_memory() as error:
            raise reservation.ReservationRefused(
                f"the card couldn't give the reservation of {reservation.gib(made.reservation)} GiB"
                f" once the models were loaded ({error}): another program has taken memory since"
                " the card was read; free it, or start with a smaller --max-size, --max-input or"
                " --max-concurrent",
                0,
                None,
            ) from None
        self.admitted = admission.Admission(made.pool, made.count, self.options.queue)
        self._say_reservation()
        return self.models

    def _say_reservation(self) -> None:
        made, options = self.made, self.options
        assert made is not None
        long, short = options.bound
        each = (
            f"from inputs of up to {reservation.megapixels(options.input_pixels)} MP, each"
            f" picture's passes held to {reservation.gib(made.budget)} GiB, what the bound's own"
            " take with the plan's tiles"
        )
        if made.count == 1:
            holds = (
                f"one picture at a time within {long}x{short} {each}, the models placed by the plan"
            )
            fits = (
                f"what fits: one picture at a time within {long}x{short}; larger ones are refused"
                " (--max-size)"
            )
        else:
            holds = (
                f"the models' {reservation.gib(made.models)} GiB, all on the card, one picture in"
                f" the VAE and one in the DiT at a time, within {long}x{short} {each}"
            )
            pins = made.plan.pins
            if pins.encode_tile is not None and pins.encode_tile == pins.decode_tile:
                # The tiles reservation._made pinned for the DiT to stay whole: said, a picture's
                # passes taking them where the plan's own for one picture would be larger.
                holds += (
                    f", the VAE's passes in tiles of {pins.encode_tile} px (untiled where a"
                    " picture's frame is no larger), so that the DiT stays whole beside them"
                )
            fits = (
                f"what fits: {made.count} pictures in flight within {long}x{short}"
                " (--max-concurrent), two on the GPU at a time, one in each model, the others"
                " waiting between them on the CPU"
            )
        logger.info(
            "reservation: %s GiB held for torch on the GPU and capped there: %s; the CUDA context,"
            " outside torch's allocator, takes about %s GiB more",
            reservation.gib(made.reservation),
            holds,
            reservation.gib(memory.CONTEXT),
        )
        logger.info("%s", fits)

    def warm_up(self) -> None:
        """--max-concurrent pictures at the bound at once, from inputs of --max-input, through the
        models and the queue, as requests go: the plan proved on the card before the first
        request. A picture that fails stops the start (Failed)."""
        options = self.options
        long, short = options.bound
        # The input at the bound's own ratio, as large as --max-input allows, so that the output
        # is the bound itself (frames.target_size of long x short times a whole factor); a bound
        # whose ratio's smallest picture is over --max-input takes the largest at about it.
        unit = math.gcd(long, short)
        wide, high = long // unit, short // unit
        factor = math.isqrt(options.input_pixels // (wide * high))
        if factor:
            width, height = wide * factor, high * factor
        else:
            width = math.isqrt(options.input_pixels * long // short)
            height = -(-width * short // long)
            while width * height > options.input_pixels:
                width -= 1
                height = -(-width * short // long)
        pixels = _warm_picture(height, width)
        # Both orientations, each at the bound: a portrait picture has other padded rows and
        # columns, so other tiles.
        sources = [
            picture.Picture(pixels, "PNG", 8, False, None, (), 1, False),
            picture.Picture(
                np.ascontiguousarray(pixels.transpose(1, 0, 2)), "PNG", 8, False, None, (), 1, False
            ),
        ]
        started = time.monotonic()
        for source in sources:
            prepared = [
                self._prepared(source, answers.Settings(short, index), answers.PNG)
                for index in range(options.count)
            ]
            failures: list[BaseException] = []

            def one(each: Prepared, failures: list[BaseException] = failures) -> None:
                try:
                    self.run(each, warm=True)
                except BaseException as error:  # handed to the start, which refuses
                    failures.append(error)

            threads = [threading.Thread(target=one, args=(each,)) for each in prepared]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            if failures:
                raise failures[0]
        logger.info(
            "warm-up: %d %s within %dx%d, landscape and portrait, from inputs of %dx%d"
            " (%s MP) in %.1f s; torch reserved %s GiB at its peak and holds %s GiB now, of the"
            " reservation's %s",
            options.count,
            "picture" if options.count == 1 else "pictures at once",
            long,
            short,
            width,
            height,
            reservation.megapixels(width * height),
            time.monotonic() - started,
            reservation.gib(self.card.reserved_peak()),
            reservation.gib(self.card.reserved()),
            reservation.gib(self._reservation()),
        )
        self._held()

    def ready(self) -> None:
        with self._state:
            if self.state is State.STARTING:
                self.state = State.READY

    def stop(self) -> None:
        """No new picture: those waiting are answered 503, those running finish."""
        with self._state:
            if self.state is not State.FAILED:
                self.state = State.STOPPING
        if self.admitted is not None:
            self.admitted.stop()

    def _reservation(self) -> int:
        return 0 if self.made is None else self.made.reservation

    def gauges(self) -> Gauges:
        admitted = self.admitted
        loaded = self.models is not None
        return Gauges(
            running=0 if admitted is None else admitted.running,
            waiting=0 if admitted is None else admitted.waiting,
            reserved=self.card.reserved() if loaded else 0,
            reserved_peak=self.card.reserved_peak() if loaded else 0,
            reservation=self._reservation(),
            ready=self.state is State.READY,
        )

    # ---- A picture --------------------------------------------------------------------------

    def taking(self) -> None:
        """Unavailable where the server takes no picture now: starting, stopping or failed."""
        state = self.state
        if state is State.STARTING:
            raise Unavailable("the server is starting: its models load, then its warm-up runs")
        if state is State.STOPPING:
            raise Unavailable("the server is stopping")
        if state is State.FAILED:
            raise Unavailable(
                "the server has failed: a CUDA error left the GPU's state untrusted, and it takes"
                " no picture until it is restarted",
                retry=False,
            )

    def prepare(
        self, data: bytes, settings: answers.Settings, formats: answers.Formats
    ) -> Prepared:
        """A request's picture read and checked, in its request's thread: Unprocessable for a
        picture refused or over the server's limits, answers.NotAcceptable where the client
        takes no format that holds it."""
        size = _stored_size(data)
        limit = self.options.input_pixels
        if size is not None and size[0] * size[1] > limit:
            raise Unprocessable(
                f"The picture has {size[0] * size[1]:,} pixels ({size[0]}x{size[1]}), over the"
                f" server's largest input, {limit:,}. Send a smaller copy of it."
            )
        try:
            read = picture.read(data, "the picture")
        except Refusal as error:
            raise Unprocessable(answers.refused(error), error.kind) from None
        chosen = formats.first
        if chosen == answers.WEBP:
            why = webp.why_not(read.bits, read.icc, read.chunks)
            if why is not None:
                if formats.fallback is None:
                    raise answers.NotAcceptable(
                        f"The picture can't be a lossless WebP ({why}), and the request's Accept"
                        f" takes no PNG ({answers.PNG}), which holds it."
                    )
                chosen = formats.fallback
        return self._prepared(read, settings, chosen, formats.fallback)

    def _prepared(
        self,
        read: picture.Picture,
        settings: answers.Settings,
        chosen: str,
        fallback: str | None = None,
    ) -> Prepared:
        made = self.made
        assert made is not None
        height, width = read.pixels.shape[:2]
        if width * height > self.options.input_pixels:
            raise Unprocessable(
                f"The picture has {width * height:,} pixels ({width}x{height}), over the"
                f" server's largest input, {self.options.input_pixels:,}. Send a smaller copy of"
                " it."
            )
        target = frames.target_size(width, height, Fraction(1), settings.resolution)
        try:
            frames.check_target(target, settings.resolution)
        except Refusal as error:
            raise Unprocessable(answers.refused(error), error.kind) from None
        out_height, out_width = frames.output_size(target)
        long, short = made.plan.bound
        if max(out_height, out_width) > long or min(out_height, out_width) > short:
            raise Unprocessable(
                f"Seedvr2x-Resolution {settings.resolution} makes a {out_width}x{out_height}"
                f" output, over the server's largest, {long}x{short} in either orientation. Ask"
                f" for Seedvr2x-Resolution {_largest_resolution(width, height, long, short)} at"
                " the most for this picture."
            )
        if chosen == answers.WEBP and max(out_height, out_width) > webp.MAX_SIDE:
            # WebP's side limit, checked before the GPU's work (webp.write refuses it after).
            if fallback is None:
                raise answers.NotAcceptable(
                    f"A {out_width}x{out_height} output can't be a WebP, whose sides stop at"
                    f" {webp.MAX_SIDE} px, and the request's Accept takes no PNG ({answers.PNG})."
                )
            chosen = fallback
        peaks_of = still_plan.picture_peaks(made.plan, target, width * height, made.budget)
        seed = secrets.randbelow(answers.SEED_MAX + 1) if settings.seed is None else settings.seed
        return Prepared(
            read,
            target,
            seed,
            settings.resolution,
            chosen,
            reservation.share(peaks_of),
            still_plan.tiles(made.plan, target, made.budget),
        )

    def run(
        self, prepared: Prepared, gone: Callable[[], bool] = lambda: False, warm: bool = False
    ) -> Made:
        """A prepared picture through the queue and the models, its file written: Unavailable
        for the queue full or the server stopping, admission.Gone for a client gone while it
        waited, Failed for a picture that failed on the GPU. `warm`: the warm-up's, taken while
        the server starts."""
        if not warm:
            self.taking()
        assert self.admitted is not None
        if prepared.need > self.admitted.pool:
            # The plan's figures say no picture within the bound takes more: a planning error,
            # never run past the reservation.
            logger.error(
                "a picture's share of %d bytes passes the pool's %d: the plan is wrong; the picture"
                " is answered 500",
                prepared.need,
                self.admitted.pool,
            )
            raise Failed("The picture failed: the server's plan doesn't hold it; its log says why.")
        try:
            ticket = self.admitted.enter(prepared.need, gone)
        except admission.Full:
            raise Unavailable(
                f"the server's queue is full: {self.options.queue} pictures wait already"
            ) from None
        except admission.Stopping:
            self.taking()
            raise Unavailable("the server is stopping") from None
        started = time.monotonic()
        try:
            out = self._upscaled(prepared)
        finally:
            self.admitted.leave(ticket)
            self._held()
        seconds = time.monotonic() - started
        made = self._written(prepared, out, ticket.waited, seconds)
        # The warm-up's picture is no request's: counted, the upscale and wait counts would pass
        # the 200s of requests_total by the pictures at once.
        if not warm:
            self.metrics.upscaled(ticket.waited, seconds, made.output[0] * made.output[1])
        return made

    def _upscaled(self, prepared: Prepared) -> npt.NDArray[np.float32]:
        from seedvr2x import still

        models = self.models
        assert models is not None
        pixels, target, seed = prepared.picture.pixels, prepared.target, prepared.seed
        try:
            with self.card.watched(models):
                with self._vae:
                    latent = still.encoded(
                        models, pixels, target, seed, tiles=prepared.tiles.encode
                    )
                with self._dit:
                    sampled = still.sampled(models, latent, seed)
                del latent
                with self._vae:
                    return still.decoded(
                        models, sampled, pixels, target, tiles=prepared.tiles.decode, correct=True
                    )
        except still.NonFinite as error:
            logger.error("%s: no picture is answered", error)
            raise Failed(f"The picture failed: {error}.") from None
        except Exception as error:
            if self.card.fatal(error):
                with self._state:
                    self.state = State.FAILED
                if self.admitted is not None:
                    self.admitted.stop()
                self.failure = str(error)
                self.failed.set()
                logger.critical(
                    "%s: the GPU's state is untrusted, the server takes no picture any longer"
                    " (/health says failed); it exits once the pictures running are answered",
                    error,
                )
                raise Failed(
                    "The picture failed on the GPU with a CUDA error, which leaves its state"
                    " untrusted: the server takes no picture until it is restarted; its log says"
                    " which error.",
                    fatal=True,
                ) from None
            if isinstance(error, self.card.out_of_memory()) or "_ALLOC_FAILED" in str(error):
                # cuBLAS's and cuDNN's refused allocations are out of memory too, not a broken
                # card (CUBLAS_STATUS_ALLOC_FAILED, CUDNN_STATUS_ALLOC_FAILED).
                logger.error("%s: the picture is answered 500", error)
                raise Failed(
                    "The picture ran out of the GPU's memory, which the plan should have"
                    " prevented; the server goes on."
                ) from None
            raise

    def _held(self) -> None:
        """The reservation held again where torch holds less of it than reserved, with no picture
        running (Admission.when_idle): torch gives its cached memory back to the card where an
        allocation can't otherwise be served under its limit (an allocation refused, or
        fragmentation at the cap), and the hold is made again from what the card has free, or
        said lost in a warning.

        Not at every picture: the models' moves between phases leave the reservation alone
        (memory.keep; before, every picture of 12 gave it back where the plan moved the DiT's
        blocks, an RTX 5090 under WSL2, 2026-11-04). A held run's check gives it back too where
        it finds the card short (memory.check), and stops the picture or says why; what is left
        is torch's own release at its cap, where its cache has no block for an allocation (a
        picture's buffers met in another order), which takes nothing from the bound: said at
        info level, as it is done. PROVISIONAL (impl-imgserver, 2026-11-04): how often that is
        on a card is for its GPU run to count."""
        admitted, made = self.admitted, self.made
        # torch reserves in steps (memory.keep takes the reservation in allocations of
        # memory.MIN_LARGE and more, which torch rounds to memory.ROUND_LARGE), so it holds up to
        # a step under a reservation that is no multiple of one: on the RTX 5090 (2026-11-04,
        # int8, 18.43 GiB) 0.2 to 2.2 MiB under it after every picture, which made this line say
        # the hold was lost each time. Within a step of it, the hold is whole.
        if admitted is None or made is None:
            return
        if self.card.reserved() >= made.reservation - memory.MIN_LARGE - memory.ROUND_LARGE:
            return

        def again() -> None:
            try:
                self.card.reserve(made.reservation)
            except Exception as error:
                logger.warning(
                    "torch holds %s GiB of the reservation's %s and the card couldn't give the"
                    " rest back (%s): another program may hold it, and a picture may run out of"
                    " memory",
                    reservation.gib(self.card.reserved()),
                    reservation.gib(made.reservation),
                    error,
                )
            else:
                logger.info(
                    "the reservation held again: torch had given part of it back to the card"
                )

        admitted.when_idle(again)

    def _written(
        self, prepared: Prepared, out: npt.NDArray[np.float32], waited: float, seconds: float
    ) -> Made:
        """The picture's file, upright, with its profile and colour chunks and seedvr2x's record,
        in its request's thread (DESIGN.md, Still images, Output)."""
        read = prepared.picture
        assert self.record_models is not None and self.made is not None
        turned = picture.upright(out, read.orientation)
        written = (turned.shape[1], turned.shape[0])
        height, width = read.pixels.shape[:2]
        tiles = prepared.tiles
        made = records.Record(
            app=APP,
            version=metadata.version(APP),
            source=(width, height),
            output=written,
            resolution=prepared.resolution,
            sample_aspect=Fraction(1),
            dit=self.record_models[0],
            vae=self.record_models[1],
            seed=prepared.seed,
            color_correction="split",
            encode_tile=0 if tiles.encode is None else tiles.encode.size,
            decode_tile=0 if tiles.decode is None else tiles.decode.size,
            attention=self.made.plan.attention,
        )
        if prepared.format == answers.WEBP:
            data = webp.write(
                turned, grey=read.grey, icc=read.icc, chunks=read.chunks, xmp=made.xmp()
            )
        else:
            data = png.write(
                turned,
                bits=read.bits,
                grey=read.grey,
                icc=read.icc,
                chunks=read.chunks,
                xmp=made.xmp(),
                software=made.creator,
            )
        return Made(data, prepared.format, prepared.seed, (width, height), written, waited, seconds)


def _stored_size(data: bytes) -> tuple[int, int] | None:
    """A picture's size as stored, (width, height), from its header alone, before any pixel is
    decoded, so that a picture over the server's limit is refused unread; None where Pillow can't
    say, the read then refusing it in its own words."""
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data), formats=picture.FORMATS) as image:
            return image.size
    except Exception:
        return None


def _largest_resolution(width: int, height: int, long: int, short: int) -> int:
    """The largest Seedvr2x-Resolution whose output of a picture of width x height, square
    pixels, is within the bound in one orientation or the other (frames.target_size,
    output_size), 1 where none is."""
    low, high = 1, max(long, short)
    while low < high:
        middle = (low + high + 1) // 2
        rows, columns = frames.output_size(frames.target_size(width, height, Fraction(1), middle))
        if max(rows, columns) <= long and min(rows, columns) <= short:
            low = middle
        else:
            high = middle - 1
    return low


def _warm_picture(height: int, width: int) -> npt.NDArray[np.float32]:
    """The warm-up's picture, (H, W, 3) float32 in [0, 1]: gradients and seeded noise, so that no
    pass runs on a constant."""
    rows = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    columns = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    noise = np.random.default_rng(0).random((height, width, 3), dtype=np.float32)
    pixels = 0.4 * noise
    pixels[..., 0] += 0.6 * rows
    pixels[..., 1] += 0.6 * columns
    pixels[..., 2] += 0.3 * (rows + columns)
    return np.ascontiguousarray(np.clip(pixels, 0.0, 1.0))

"""The server on the CPU for the tests, over a real socket on localhost: the stand-in machine in
service.Card's place (StandIn), the library's stand-in models (cpu_models' VAE in bfloat16, a DiT
of arithmetic: core/tests/test_still.py's pipeline_models and stand_in_dit), and a server
listening on a port of its own, started and stopped as main.serve does (main.launch, bring_up,
shut), without its signals, which only the main thread may take."""

import contextlib
import http.client
import io
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from fractions import Fraction
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from test_still import pipeline_models, stand_in_dit

import main
import service as services
from metrics import Metrics
from seedvr2x import frames, memory, model, peaks, picture, png, still, still_plan, webp, weights
from seedvr2x import record as records
from seedvr2x.memory import Reading
from seedvr2x.passes import ModelPlan
from seedvr2x.pull import ModelFiles
from seedvr2x.weights import Architecture

GIB = peaks.GIB
# A card of 32 GiB with 24 free, as the RTX 5090 had beside the desktop (DESIGN.md, Still images).
FREE = 24 * GIB
HOST = 64 * GIB
# Pictures small enough for the stand-in VAE on the CPU: within 128x96, from inputs of 16,000 px.
BOUND = (128, 96)
INPUT = 16_000
NAME = "w.safetensors"


@dataclass
class StandIn:
    """service.Machine on the CPU: the stand-in models, the card's figures chosen, the
    reservation recorded, and failures put in on purpose (`fail`, called before each pass)."""

    vae_models: model.Models
    free_bytes: int = FREE
    host_bytes: int | None = HOST
    attention_name: str = peaks.FLASH_ATTN_2
    reserves: list[int] = field(default_factory=list[int])
    # Bytes of the last reservation torch has given back to the card: reserved() is that less,
    # until reserve() is called again.
    lost: int = 0
    # What reserve() raises where set (MemoryError: the card refusing, as the real one does with
    # torch.OutOfMemoryError); reserve() is then not recorded.
    refuse: BaseException | None = None
    fail: Callable[[], None] | None = None
    plans: list[ModelPlan] = field(default_factory=list[ModelPlan])
    # The pictures inside a pass now, and the most there were at once (watched).
    inside: int = 0
    most_inside: int = 0
    _counting: threading.Lock = field(default_factory=threading.Lock)

    def device_name(self) -> str:
        return "a stand-in"

    def free(self) -> int:
        return self.free_bytes

    def reading(self, free: int) -> Reading:
        return memory.reading(free)

    def host(self) -> int | None:
        return self.host_bytes

    def attention(self) -> str:
        return self.attention_name

    def dit_model(self, path: Path) -> Architecture:
        return peaks.FP16

    def load(self, files: ModelFiles, plan: ModelPlan, reserve: int) -> model.Models:
        self.plans.append(plan)
        loaded: Any = pipeline_models(self.vae_models)
        loaded.attention = self.attention_name
        return loaded

    def reserve(self, size: int) -> None:
        if self.refuse is not None:
            raise self.refuse
        self.reserves.append(size)
        self.lost = 0

    def reserved(self) -> int:
        return self.reserves[-1] - self.lost if self.reserves else 0

    def reserved_peak(self) -> int:
        return self.reserved()

    def out_of_memory(self) -> tuple[type[BaseException], ...]:
        return (MemoryError,)

    def fatal(self, error: BaseException) -> bool:
        return isinstance(error, RuntimeError) and "CUDA error" in str(error)

    @contextlib.contextmanager
    def watched(self, models: model.Models) -> Iterator[None]:
        """The picture's passes: `fail` is called on entering (raising there is a pass that
        fails), and the pictures inside at once are counted."""
        if self.fail is not None:
            self.fail()
        with self._counting:
            self.inside += 1
            self.most_inside = max(self.most_inside, self.inside)
        try:
            yield
        finally:
            with self._counting:
                self.inside -= 1


@dataclass
class HeldStart(StandIn):
    """A machine whose load waits for `go`, `reached` set once it is there: a server that is
    starting for as long as the test wants."""

    reached: threading.Event = field(default_factory=threading.Event)
    go: threading.Event = field(default_factory=threading.Event)

    def load(self, files: ModelFiles, plan: ModelPlan, reserve: int) -> model.Models:
        self.reached.set()
        assert self.go.wait(120), "the start was never let go"
        return super().load(files, plan, reserve)


@dataclass
class Live:
    """A server listening, ready or not: its address, service, machine and the start's thread."""

    running: main.Running
    machine: StandIn
    starting: threading.Thread
    errors: list[BaseException]

    @property
    def service(self) -> services.Service:
        return self.running.service

    def connect(self, timeout: float = 60.0) -> http.client.HTTPConnection:
        host, port = self.running.address
        return http.client.HTTPConnection(host, port, timeout=timeout)

    def request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        """One request on a connection of its own: the status, the headers (lower-case names)
        and the body."""
        connection = self.connect()
        try:
            connection.request(method, path, body=body, headers=headers or {})
            answer = connection.getresponse()
            return (
                answer.status,
                {name.lower(): value for name, value in answer.getheaders()},
                answer.read(),
            )
        finally:
            connection.close()

    def upscale(
        self,
        body: bytes,
        resolution: int | str = 48,
        seed: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        """POST /upscale of `body` with Seedvr2x-Resolution (and -Seed where given) and `headers`
        besides."""
        sent = {"Seedvr2x-Resolution": str(resolution)}
        if seed is not None:
            sent["Seedvr2x-Seed"] = str(seed)
        sent.update(headers or {})
        return self.request("POST", "/upscale", body, sent)

    def wait_ready(self, timeout: float = 300.0) -> None:
        self.starting.join(timeout)
        if self.errors:
            raise self.errors[0]
        assert not self.starting.is_alive(), "the start didn't end"


@dataclass
class Gate:
    """A `StandIn.fail` hook that stops the first `blocking` pictures it sees at the start of their
    passes until `release` is set, and counts every call (the warm-up's included, so it is set
    after the start)."""

    blocking: int = 1
    calls: int = 0
    entered: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __call__(self) -> None:
        with self._lock:
            self.calls += 1
            blocked = self.calls <= self.blocking
        if blocked:
            self.entered.set()
            assert self.release.wait(60), "the gate was never released"


def until(condition: Callable[[], bool], what: str, timeout: float = 30.0) -> None:
    """Wait for `condition` to hold, failing with `what` after `timeout` seconds."""
    end = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < end, what
        time.sleep(0.01)


def png_of(
    width: int,
    height: int,
    seed: int = 0,
    mode: str = "RGB",
    orientation: int | None = None,
) -> bytes:
    """A PNG of noise, 8 bits, in the Pillow `mode` given (RGB, L, RGBA: opaque), with an EXIF
    orientation where given."""
    channels = {"L": 1, "RGB": 3, "RGBA": 4}[mode]
    shape = (height, width) if channels == 1 else (height, width, channels)
    pixels = np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)
    if mode == "RGBA":
        pixels[..., 3] = 255
    image = Image.fromarray(pixels, mode)
    out = io.BytesIO()
    if orientation is None:
        image.save(out, "PNG")
    else:
        exif = Image.Exif()
        exif[0x0112] = orientation
        image.save(out, "PNG", exif=exif.tobytes())
    return out.getvalue()


def model_dir(tmp_path: Path) -> Path:
    """A model directory holding an empty file for both models, which the stand-in loads past:
    unpinned, so it is only hashed, as the CLI's tests have it (tests/test_still_cli.py)."""
    directory = tmp_path / "models"
    directory.mkdir(exist_ok=True)
    (directory / NAME).write_bytes(b"")
    return directory


def options(tmp_path: Path, **given: Any) -> services.Options:
    settings: dict[str, Any] = {
        "bound": BOUND,
        "input_pixels": INPUT,
        "model_dir": model_dir(tmp_path),
        "dit": NAME,
        "vae": NAME,
    }
    settings.update(given)
    return services.Options(**settings)


def quiet_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    """The autouse fixture no_driver_asked's patches (core/tests/library_fixtures.py), for a server
    started in a fixture of a wider scope than a test: the driver is asked nothing."""
    monkeypatch.setattr(memory, "paravirtual", lambda: False)
    monkeypatch.setattr(memory, "NVML", "libnvidia-ml.so.none-in-a-cpu-test")
    for name in memory.VARIABLES:
        monkeypatch.delenv(name, raising=False)
    state = (("_spills", False), ("_native", False), ("_limit", None), ("_footing", None))
    held = (("_segments", False), ("_plan_s_limit", None), ("_base", None), ("_kept", None))
    for name, value in (*state, *held, ("_planned", None), ("_left", memory.HELD_BACK)):
        monkeypatch.setattr(memory, name, value)


@contextlib.contextmanager
def live(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    vae_models: model.Models,
    *,
    key: str | None = None,
    ready: bool = True,
    machine: StandIn | None = None,
    **given: Any,
) -> Iterator[Live]:
    """A server on 127.0.0.1, a port the system gives, its start run in a thread of its own
    (main.bring_up), ready once it has passed where `ready`; shut on leaving (main.shut)."""
    monkeypatch.setattr(weights, "check_models", lambda *args: None)
    stand_in_dit(monkeypatch)
    found = machine or StandIn(vae_models)
    running = main.launch(options(tmp_path, **given), key, ("127.0.0.1", 0), found, Metrics())
    errors: list[BaseException] = []

    def start() -> None:
        try:
            main.bring_up(running)
            running.service.ready()
        except BaseException as error:
            errors.append(error)

    starting = threading.Thread(target=start, name="start")
    if ready:
        starting.start()
    made = Live(running, found, starting, errors)
    try:
        if ready:
            made.wait_ready()
        yield made
    finally:
        if starting.is_alive() or (not ready and starting.ident is not None):
            starting.join(300)
        main.shut(running)


def library_file(
    server: "Live",
    vae_models: model.Models,
    data: bytes,
    resolution: int,
    seed: int,
    *,
    as_webp: bool = False,
) -> bytes:
    """The file the library makes of a picture with the server's settings: still.upscale on the
    stand-in models, with the plan's tiles, turned upright, written with the record the server
    writes (service._written), the stand-in DiT being the one patched in."""
    service = server.service
    made, models = service.made, service.record_models
    assert made is not None and models is not None
    read = picture.read(data, "the picture")
    height, width = read.pixels.shape[:2]
    target = frames.target_size(width, height, Fraction(1), resolution)
    tiles = still_plan.tiles(made.plan, target)
    out = still.upscale(
        pipeline_models(vae_models),
        read.pixels,
        target,
        seed,
        encode_tiles=tiles.encode,
        decode_tiles=tiles.decode,
        correct=True,
    )
    turned = picture.upright(out, read.orientation)
    written = records.Record(
        app=services.APP,
        version=metadata.version(services.APP),
        source=(width, height),
        output=(turned.shape[1], turned.shape[0]),
        resolution=resolution,
        sample_aspect=Fraction(1),
        dit=models[0],
        vae=models[1],
        seed=seed,
        color_correction="split",
        encode_tile=0 if tiles.encode is None else tiles.encode.size,
        decode_tile=0 if tiles.decode is None else tiles.decode.size,
        attention=made.plan.attention,
    )
    if as_webp:
        return webp.write(
            turned, grey=read.grey, icc=read.icc, chunks=read.chunks, xmp=written.xmp()
        )
    return png.write(
        turned,
        bits=read.bits,
        grey=read.grey,
        icc=read.icc,
        chunks=read.chunks,
        xmp=written.xmp(),
        software=written.creator,
    )

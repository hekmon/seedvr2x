"""seedvr2x-imgserver, seedvr2x's image server (DESIGN.md, The image server): its options, its
environment set before torch is imported, its log, and its life: it listens at once, checks its
model files, reads the card, plans, loads the models, makes its reservation, warms up, then
serves until SIGTERM or Ctrl-C, or until a CUDA error (other than out of memory) fails it, which
ends the process with FAILED_EXIT."""

import argparse
import logging
import os
import re
import signal
import sys
import threading
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import web
    from metrics import Metrics
    from service import Machine, Options, Service

logger = logging.getLogger("imgserver.main")

# 127.0.0.1 by default (DESIGN.md, The image server, Access), on a port of its own, not ComfyUI's
# 8188 nor its neighbours.
HOST = "127.0.0.1"
PORT = 8390
# --max-size's default, either orientation (DESIGN.md).
MAX_SIZE = (3840, 2160)
# --max-input's default, in pixels: 50 MP, a 48 MP phone's photo (8064x6048 is 48.8 MP) with room,
# under Pillow's own warning at 89.5 MP; its copies on the GPU take 1.4 GiB in the encode at 30
# bytes a pixel (still_plan.RESIZE).
MAX_INPUT = 50_000_000
QUEUE = 8
KEY_VARIABLE = "SEEDVR2X_API_KEY"
# The listening address as a variable, for a container: the image sets it to 0.0.0.0:8390 and its
# ENTRYPOINT is the bare command, so that an option given to `docker run` appends and never takes
# the address's place (apps/imgserver/Dockerfile).
LISTEN_VARIABLE = "SEEDVR2X_LISTEN"
# The exit code of a server that a CUDA error failed (DESIGN.md, /health): the process ends so
# that a restart policy (Docker's --restart on-failure or unless-stopped, Kubernetes) starts it
# again on a card in a state it trusts. 3 because the others are taken: 0 stopped, 1 a refused
# start, 2 argparse's for a wrong option, 128+N a second signal; and a code of its own tells an
# orchestrator's log, and a user, a failed card from a refused start, which no restart cures.
FAILED_EXIT = 3
# How often serve looks at the service's failed event beside the stop's: a signal handler may
# set an Event but not wait on two; 0.2 s of delay costs nothing before an exit.
WATCH = 0.2
DIT = "seedvr2x_ema_7b_sharp_fp16.safetensors"
VAE = "seedvr2x_ema_vae_fp16.safetensors"


def _size(text: str) -> tuple[int, int]:
    found = re.fullmatch(r"([0-9]{1,6})x([0-9]{1,6})", text.strip())
    if found is None:
        raise argparse.ArgumentTypeError(f"{text!r}: WIDTHxHEIGHT, as 3840x2160")
    width, height = int(found[1]), int(found[2])
    if width % 2 or height % 2 or min(width, height) < 32:
        raise argparse.ArgumentTypeError(
            f"{text}: even sides of 32 px at least, as an output is (frames.output_size)"
        )
    return width, height


def _positive(text: str) -> int:
    if not text.isascii() or not text.isdigit() or int(text) < 1:
        raise argparse.ArgumentTypeError(f"{text!r}: a whole number from 1")
    return int(text)


def _count(text: str) -> int:
    if not text.isascii() or not text.isdigit():
        raise argparse.ArgumentTypeError(f"{text!r}: a whole number from 0")
    return int(text)


def _tile(text: str) -> int:
    value = _count(text)
    if value and (value < 512 or value % 64):
        raise argparse.ArgumentTypeError(f"{text}: 0 (untiled), or a multiple of 64 from 512")
    return value


def _listen(text: str) -> tuple[str, int]:
    host, colon, port = text.rpartition(":")
    if not colon or not port.isascii() or not port.isdigit() or not 0 < int(port) < 65536:
        raise argparse.ArgumentTypeError(f"{text!r}: HOST:PORT, as {HOST}:{PORT}")
    return host.strip("[]") or HOST, int(port)


def parser() -> argparse.ArgumentParser:
    found = argparse.ArgumentParser(
        prog="seedvr2x-imgserver",
        description=(
            "seedvr2x's image server: POST /upscale takes a picture's bytes and answers it"
            " upscaled, the models loaded once and kept on the GPU. GET /health and GET /metrics"
            " for a container's probes and Prometheus."
        ),
    )
    found.add_argument(
        "--listen",
        type=_listen,
        metavar="HOST:PORT",
        help=f"where it listens, {HOST}:{PORT} by default; {LISTEN_VARIABLE} sets it too, and the"
        f" image sets it to 0.0.0.0:{PORT}; the option wins over the variable",
    )
    found.add_argument(
        "--max-size",
        type=_size,
        default=MAX_SIZE,
        metavar="WxH",
        help="the largest output, in either orientation, 3840x2160 by default: the reservation"
        " holds --max-concurrent pictures of it",
    )
    found.add_argument(
        "--max-input",
        type=_positive,
        default=MAX_INPUT,
        metavar="PIXELS",
        help=f"the largest picture taken, in pixels as stored, {MAX_INPUT:,} by default: it is"
        " copied and resized on the GPU, so the reservation holds it",
    )
    found.add_argument(
        "--max-concurrent",
        type=_positive,
        default=1,
        metavar="N",
        help="the pictures upscaled at once, 1 by default; more than one needs the whole DiT on"
        " the card",
    )
    found.add_argument(
        "--queue",
        type=_count,
        default=QUEUE,
        metavar="N",
        help=f"the pictures that may wait, {QUEUE} by default; past them, 503",
    )
    found.add_argument(
        "--model-dir",
        type=Path,
        metavar="DIR",
        help="read the model files from DIR, nothing downloaded; by default seedvr2x's own"
        " files, from Hugging Face's cache, pulled at the pinned revision where missing",
    )
    found.add_argument(
        "--dit-model",
        default=DIT,
        metavar="FILE",
        help=f"the DiT, {DIT} (the default, fp16); smaller files of the same 7B: seedvr2x's int8"
        " files (half the weights) and its dynamic GGUF files (under a third: about 4.5 GiB of"
        " pinned system memory in place of 15, a little slower, no less VRAM where the VAE's"
        " floor binds), recognised by their tensors",
    )
    found.add_argument("--vae-model", default=VAE, metavar="FILE", help=f"the VAE, {VAE}")
    found.add_argument(
        "--encode-tile", type=_tile, metavar="PX", help="pin the encode's tiles (0: untiled)"
    )
    found.add_argument(
        "--decode-tile", type=_tile, metavar="PX", help="pin the decode's tiles (0: untiled)"
    )
    found.add_argument(
        "--swap-blocks", type=_count, metavar="N", help="pin the DiT's blocks swapped"
    )
    found.add_argument(
        "--api-key",
        metavar="KEY",
        help=f"require Authorization: Bearer KEY on /upscale; {KEY_VARIABLE} sets it too, which"
        " the host's process list doesn't show, so prefer it",
    )
    found.add_argument("-v", "--verbose", action="store_true", help="debug lines")
    return found


def _logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    if not verbose:
        # Libraries' own debug and info, the server's lines alone.
        for name in ("httpx", "huggingface_hub", "urllib3", "filelock"):
            logging.getLogger(name).setLevel(logging.WARNING)


def _environment(verbose: bool) -> None:
    """What must be set before huggingface_hub and torch are imported: Hugging Face's telemetry
    off, torch's C++ warnings off but under -v, as the CLI sets them (apps/seedvr2x/main.py), and
    torch's own allocator with expandable segments, the user's setting standing: the reservation
    is allocated once and kept by it, and its limit refuses what passes it, where
    cudaMallocAsync's went 2.06 GiB past (DESIGN.md, Allocator)."""
    from seedvr2x import memory

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if not verbose:
        os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
    for name in memory.VARIABLES:
        if os.environ.get(name) == "":
            del os.environ[name]
    given = next((os.environ[name] for name in memory.VARIABLES if name in os.environ), None)
    if given is None:
        os.environ[memory.VARIABLES[0]] = memory.EXPANDABLE
    elif "cudaMallocAsync" in given:
        logger.warning(
            "the allocator's setting in the environment, %s, takes cudaMallocAsync, whose limit"
            " doesn't hold the reservation: a picture may take more of the card than reserved",
            given,
        )
    memory.startup()
    # Pillow's warning for a picture over half its limit, which seedvr2x replaces by a line of
    # its own (seedvr2x/picture.py, _say_large), off for the whole process, before any thread:
    # warnings.catch_warnings in the library would not be thread-safe (the CLI's
    # _pillow_size_warning_off).
    from PIL import Image

    warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)


def _key(given: str | None) -> str | None:
    """The API key, --api-key winning over SEEDVR2X_API_KEY, its source logged, never the key:
    SystemExit for an empty one, one with a blank at either end (never stripped silently: the
    client would have to send what the server took), or one that isn't printable ASCII with no
    space (HTTP carries a header's value as latin-1: a UTF-8 key would never match)."""
    from_variable = os.environ.get(KEY_VARIABLE)
    key, source = (given, "--api-key") if given is not None else (from_variable, KEY_VARIABLE)
    if key is None:
        logger.warning(
            "no API key: /upscale answers any client that reaches the server; --api-key or %s"
            " sets one",
            KEY_VARIABLE,
        )
        return None
    if not key.strip():
        logger.error("the API key from %s is empty: give one, or none", source)
        raise SystemExit(1)
    if key != key.strip():
        logger.error("the API key from %s has a blank at its start or end: remove it", source)
        raise SystemExit(1)
    if not (key.isascii() and key.isprintable()) or " " in key:
        logger.error(
            "the API key from %s must be printable ASCII with no space: a header carries it as"
            " latin-1, so any other key would never match",
            source,
        )
        raise SystemExit(1)
    if source == "--api-key":
        logger.info(
            "API key from --api-key: the host's process list shows it; %s doesn't", KEY_VARIABLE
        )
    else:
        logger.info("API key from %s", KEY_VARIABLE)
    return key


def _address(given: tuple[str, int] | None) -> tuple[str, int]:
    """The listening address, --listen winning over SEEDVR2X_LISTEN, which is read by --listen's
    own rule (`_listen`), over the default; its source logged. SystemExit for a variable that
    isn't HOST:PORT, naming it (an empty one too, as an empty key is refused)."""
    if given is not None:
        logger.info("listening address from --listen")
        return given
    text = os.environ.get(LISTEN_VARIABLE)
    if text is None:
        logger.info("listening address: the default (--listen or %s sets it)", LISTEN_VARIABLE)
        return HOST, PORT
    try:
        found = _listen(text)
    except argparse.ArgumentTypeError as error:
        logger.error("the listening address from %s is wrong: %s", LISTEN_VARIABLE, error)
        raise SystemExit(1) from None
    logger.info("listening address from %s", LISTEN_VARIABLE)
    return found


class Stop:
    """SIGTERM and Ctrl-C: the first stops the server as it should, its pictures running
    finished; a second ends the process at once."""

    def __init__(self) -> None:
        self.asked = threading.Event()
        self.during_start = False

    def handle(self, number: int, frame: FrameType | None) -> None:
        if self.asked.is_set():
            logger.warning("asked to stop again: the process ends now")
            os._exit(128 + number)
        self.asked.set()
        if self.during_start:
            raise KeyboardInterrupt


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    _logging(args.verbose)
    _environment(args.verbose)
    key = _key(args.api_key)
    listen = _address(args.listen)

    # The server's modules import the library, and torch with it: after the allocator is set.
    import service as services
    from metrics import Metrics
    from seedvr2x.still_plan import Pins

    width, height = args.max_size
    pins = Pins(args.encode_tile, args.decode_tile, args.swap_blocks)
    if args.max_concurrent > 1 and pins != Pins():
        logger.error(
            "--encode-tile, --decode-tile and --swap-blocks pin one picture at a time's plan:"
            " with --max-concurrent %d the DiT is whole on the card and the tiles are each"
            " picture's",
            args.max_concurrent,
        )
        return 1
    from PIL import Image

    pillow_limit = 2 * (Image.MAX_IMAGE_PIXELS or 0)
    if pillow_limit and args.max_input > pillow_limit:
        logger.error(
            "--max-input %d: over the %d pixels Pillow reads, which refuses a larger picture",
            args.max_input,
            pillow_limit,
        )
        return 1
    options = services.Options(
        (max(width, height), min(width, height)),
        args.max_input,
        args.max_concurrent,
        args.queue,
        args.model_dir,
        args.dit_model,
        args.vae_model,
        pins,
    )
    return serve(options, key, listen, services.Card, Metrics())


@dataclass
class Running:
    """A server listening: its service, its HTTP server and the thread serving it."""

    service: "Service"
    httpd: "web.Server"
    thread: threading.Thread

    @property
    def address(self) -> tuple[str, int]:
        host, port = self.httpd.server_address[:2]
        return str(host), int(port)


def launch(
    options: "Options",
    key: str | None,
    listen: tuple[str, int],
    machine: "Machine | None",
    found_metrics: "Metrics",
) -> Running:
    """The server listening at once on `listen`, /health saying starting: OSError where it
    can't. `machine` None: made by bring_up, once the server listens."""
    import service as services
    import web as webs

    running = services.Service(options, machine, found_metrics)
    connections = options.count + options.queue + webs.EXTRA_CONNECTIONS
    httpd = webs.Server(listen, running, key, connections)
    thread = threading.Thread(target=httpd.serve_forever, name="http", daemon=True)
    thread.start()
    found = Running(running, httpd, thread)
    logger.info("listening on http://%s:%d, /health says starting", *found.address)
    return found


def bring_up(running: Running, machine: "Callable[[], Machine] | None" = None) -> None:
    """The start, listening: the model files checked, the card's reads made (`machine`, where
    the server was launched without them), the card read, the plan and the reservation made, the
    models loaded, the warm-up run. Refusal or service.Failed where it is refused."""
    import service as services

    service = running.service
    files = service.check_files()
    if service.machine is None:
        assert machine is not None
        service.machine = machine()
    service.metrics.info.update(
        version=services.metadata.version(services.APP),
        dit_model=files.dit.name,
        vae_model=files.vae.name,
    )
    made = service.plan()
    service.metrics.info["attention"] = made.plan.attention
    service.load()
    service.warm_up()


def wait_for_end(stop: "Stop", service: "Service") -> None:
    """Returns once the stop was asked (SIGTERM, Ctrl-C) or the service failed."""
    while not (stop.asked.is_set() or service.failed.is_set()):
        stop.asked.wait(WATCH)


def shut(running: Running) -> None:
    """The stop: no new picture, those waiting answered 503, those running finished and
    answered, then the listening socket closed."""
    logger.info("stopping: no new picture; those waiting are answered 503")
    running.service.stop()
    running.httpd.shutdown()
    # The handlers' threads joined (ThreadingMixIn's block_on_close): the pictures running
    # finish and are answered.
    running.httpd.server_close()
    running.thread.join()
    logger.info("stopped")


def serve(
    options: "Options",
    key: str | None,
    listen: tuple[str, int],
    machine: "Callable[[], Machine]",
    found_metrics: "Metrics",
) -> int:
    """The server's life on `options`, `machine` making the card's reads (service.Card, or the
    tests' stand-in): listening, the start, serving until SIGTERM or Ctrl-C, the stop. Exit 0
    once stopped, 1 for a start refused, FAILED_EXIT once a CUDA error failed the server (in the
    warm-up too), after the pictures running are answered."""
    import reservation
    import service as services
    from seedvr2x.errors import Refusal

    stop = Stop()
    signal.signal(signal.SIGTERM, stop.handle)
    signal.signal(signal.SIGINT, stop.handle)
    try:
        running = launch(options, key, listen, None, found_metrics)
    except OSError as error:
        logger.error("can't listen on %s:%d: %s", *listen, error)
        return 1
    code = 0
    stop.during_start = True
    try:
        bring_up(running, machine)
        stop.during_start = False
        if stop.asked.is_set():
            raise KeyboardInterrupt
        running.service.ready()
        logger.info("ready: /health says ready")
        wait_for_end(stop, running.service)
        if running.service.failed.is_set():
            code = FAILED_EXIT
            logger.critical(
                "the server failed (%s): the waiting pictures are answered 503; it exits with"
                " code %d once the pictures running are answered, for a restart policy to start"
                " it again",
                running.service.failure,
                FAILED_EXIT,
            )
            # The pictures waiting were turned away when the state went failed, and no new one is
            # taken: this waits for those running.
            assert running.service.admitted is not None
            running.service.admitted.idle()
    except KeyboardInterrupt:
        if stop.during_start:
            logger.info("asked to stop while starting: nothing was served")
    except (Refusal, services.Failed) as error:
        logger.error("%s; the server stops", reservation.worded(error))
        # A CUDA error in the warm-up is the card's, not a refusal that a restart would repeat.
        code = FAILED_EXIT if isinstance(error, services.Failed) and error.fatal else 1
    except Exception as error:
        card = running.service.machine
        if card is None or (not card.fatal(error) and not isinstance(error, card.out_of_memory())):
            raise
        logger.error("%s, as the server started; the server stops", error)
        code = 1
    finally:
        stop.during_start = False
        shut(running)
    return code


if __name__ == "__main__":
    sys.exit(main())

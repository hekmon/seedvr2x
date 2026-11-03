"""The server's /metrics (DESIGN.md, The image server): Prometheus' text exposition format, version
0.0.4, written here with no dependency, its tests reading it back with prometheus_client's parser:
requests by status, the upscale's and the wait's times, the pictures running and waiting, the
output's pixels, the VRAM reserved and its peak, ready, and the version and model files."""

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

PREFIX = "seedvr2x_imgserver_"
CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

# The histograms' upper bounds, in seconds. The upscale: a 4K picture from 1080p took 14 s on the
# RTX 5090, the desktop live (DESIGN.md, Still images), a 1080p one about a second; 8K and the
# slower cards past a minute. The wait: none at one picture at a time with an empty queue, up to
# --queue pictures' upscales.
UPSCALE_BUCKETS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0)
WAIT_BUCKETS = (0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)


@dataclass
class Histogram:
    """A histogram's counts: per bucket (not cumulative; cumulated when written), the sum and the
    count."""

    bounds: Sequence[float]
    counts: list[int] = field(default_factory=list[int])
    total: float = 0.0
    count: int = 0

    def __post_init__(self) -> None:
        self.counts = [0] * len(self.bounds)

    def observe(self, value: float) -> None:
        for index, bound in enumerate(self.bounds):
            if value <= bound:
                self.counts[index] += 1
                break
        self.total += value
        self.count += 1


@dataclass(frozen=True)
class Gauges:
    """What /metrics reads at each scrape: the pictures running and waiting, torch's reserved
    memory and its peak (bytes; 0 before the models load), the reservation (bytes), and ready."""

    running: int
    waiting: int
    reserved: int
    reserved_peak: int
    reservation: int
    ready: bool


class Metrics:
    """The counters the server keeps, thread-safe, and their text. `info` labels the info metric:
    the version, the model files by name, the attention kernel."""

    def __init__(self, info: dict[str, str] | None = None) -> None:
        self._lock = threading.Lock()
        self.info: dict[str, str] = dict(info or {})
        self._requests: dict[str, int] = {}
        self._upscale = Histogram(UPSCALE_BUCKETS)
        self._wait = Histogram(WAIT_BUCKETS)
        self._pixels = 0

    def answered(self, status: int) -> None:
        with self._lock:
            key = str(status)
            self._requests[key] = self._requests.get(key, 0) + 1

    def upscaled(self, wait: float, seconds: float, pixels: int) -> None:
        """A picture made: its wait in the queue and its upscale, in seconds, and its output's
        pixels."""
        with self._lock:
            self._wait.observe(wait)
            self._upscale.observe(seconds)
            self._pixels += pixels

    def text(self, gauges: Callable[[], Gauges]) -> str:
        """The exposition, the gauges read now."""
        read = gauges()
        with self._lock:
            lines: list[str] = []
            _family(lines, "requests_total", "counter", "Requests answered, by HTTP status.")
            for status in sorted(self._requests):
                lines.append(_sample("requests_total", {"status": status}, self._requests[status]))
            _histogram(lines, "upscale_seconds", "A picture's upscale on the GPU.", self._upscale)
            _histogram(lines, "wait_seconds", "A picture's wait in the queue.", self._wait)
            _family(lines, "output_pixels_total", "counter", "The pixels of the pictures made.")
            lines.append(_sample("output_pixels_total", {}, self._pixels))
            info = dict(self.info)
        for name, help_, value in (
            ("pictures_running", "The pictures upscaling now.", read.running),
            ("pictures_waiting", "The pictures waiting in the queue.", read.waiting),
            ("vram_reserved_bytes", "torch's memory reserved on the GPU now.", read.reserved),
            (
                "vram_reserved_peak_bytes",
                "torch's reserved memory at its peak.",
                read.reserved_peak,
            ),
            ("vram_reservation_bytes", "The reservation torch is held to.", read.reservation),
            (
                "ready",
                "1 once the warm-up has passed, until the server stops or fails.",
                read.ready,
            ),
        ):
            _family(lines, name, "gauge", help_)
            lines.append(_sample(name, {}, int(value)))
        _family(lines, "info", "gauge", "The server's version and model files.")
        lines.append(_sample("info", info, 1))
        return "\n".join(lines) + "\n"


def _family(lines: list[str], name: str, kind: str, help_: str) -> None:
    lines.append(f"# HELP {PREFIX}{name} {_help(help_)}")
    lines.append(f"# TYPE {PREFIX}{name} {kind}")


def _histogram(lines: list[str], name: str, help_: str, histogram: Histogram) -> None:
    _family(lines, name, "histogram", help_)
    cumulated = 0
    for bound, count in zip(histogram.bounds, histogram.counts, strict=True):
        cumulated += count
        lines.append(_sample(f"{name}_bucket", {"le": _number(bound)}, cumulated))
    lines.append(_sample(f"{name}_bucket", {"le": "+Inf"}, histogram.count))
    lines.append(_sample(f"{name}_sum", {}, histogram.total))
    lines.append(_sample(f"{name}_count", {}, histogram.count))


def _sample(name: str, labels: dict[str, str], value: float) -> str:
    said = ",".join(f'{key}="{_label(text)}"' for key, text in labels.items())
    return (
        f"{PREFIX}{name}{{{said}}} {_number(value)}" if said else f"{PREFIX}{name} {_number(value)}"
    )


def _number(value: float) -> str:
    if isinstance(value, int):
        return str(value)
    return repr(float(value))


def _label(text: str) -> str:
    """A label's value escaped as the format has it: backslash, double quote and line feed."""
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _help(text: str) -> str:
    """A HELP line's text escaped: backslash and line feed."""
    return text.replace("\\", "\\\\").replace("\n", "\\n")

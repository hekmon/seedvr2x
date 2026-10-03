"""The manifest of a job's output (DESIGN.md, Pause and resume): what the job is, which a resume
must find again (its settings, models by hash, environment, input files, output, shots with
their windows, output segments), and how far it went: each shot encoded, its windows done, each
segment finished. A unit is recorded once its file is whole, so the manifest only names whole
files. Written once the models load, then again after every unit, always whole (a temporary
file, synced, renamed)."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from seedvr2x.media.files import write_whole
from seedvr2x.runtime.job import OutputSegment, Part, Shot

NAME = "manifest.json"
VERSION = 2
# The directory of the units kept for a resume, beside the manifest (units.DiskUnits).
STATE = "resume"


@dataclass
class Manifest:
    """A job's manifest, at path."""

    path: Path
    settings: dict[str, Any]
    environment: dict[str, Any]  # what the output's bits depend on besides the settings
    parts: Sequence[Part]
    shots: Sequence[Shot]
    layouts: Sequence[Sequence[tuple[int, int]]]  # each shot's DiT windows, latents [start, end)
    segments: Sequence[OutputSegment]
    files: Sequence[str]  # each segment's file (FFV1) or directory (PNG), in the output
    output: dict[str, Any]
    # The environment changes a resume accepted (--accept-env-change), oldest first: the units
    # made before each are the environment's before it (resume.environment_change).
    environment_changes: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    # How far the job went.
    encoded: list[bool] = field(default_factory=list[bool])  # each shot's latent kept
    windows_done: list[int] = field(default_factory=list[int])  # each shot's windows kept
    finished: list[bool] = field(default_factory=list[bool])  # each segment whole
    sizes: list[int | None] = field(default_factory=list[int | None])  # finished ones' bytes

    def __post_init__(self) -> None:
        if not self.encoded:
            self.encoded = [False] * len(self.shots)
        if not self.windows_done:
            self.windows_done = [0] * len(self.shots)
        if not self.finished:
            self.finished = [False] * len(self.segments)
        if not self.sizes:
            self.sizes = [None] * len(self.segments)
        self._inputs = [_input(part) for part in self.parts]

    def shot_encoded(self, index: int) -> None:
        """Record shot `index`'s latent as kept, and write the manifest."""
        self.encoded[index] = True
        self.write()

    def window_done(self, index: int, window: int) -> None:
        """Record window `window` of shot `index` as kept, the windows before it being kept
        already, and write the manifest."""
        if window != self.windows_done[index]:
            raise ValueError(f"shot {index}: window {window} after {self.windows_done[index]}")
        self.windows_done[index] = window + 1
        self.write()

    def segment_finished(self, index: int, size: int) -> None:
        """Record segment `index` as finished, whole and checked, `size` bytes, and write the
        manifest."""
        self.finished[index] = True
        self.sizes[index] = size
        self.write()

    def segment_shots(self, index: int) -> list[int]:
        """The shots of segment `index`: a segment holds whole shots."""
        segment = self.segments[index]
        return [i for i, shot in enumerate(self.shots) if segment.start <= shot.start < segment.end]

    def write(self) -> None:
        """Write the manifest, whole or not at all, even after a power cut."""
        write_whole(self.path, (json.dumps(self.content(), indent=2) + "\n").encode())

    def content(self) -> dict[str, Any]:
        seed = self.settings["seed"]
        return {
            "seedvr2x_manifest": VERSION,
            "settings": self.settings,
            "environment": self.environment,
            "environment_changes": self.environment_changes,
            "input": self._inputs,
            "output": self.output,
            "shots": [
                {
                    "start": shot.start,
                    "end": shot.end,
                    "seed": shot.seed(seed),
                    "latents": layout[-1][1],
                    "windows": [list(window) for window in layout],
                    "encoded": encoded,
                    "windows_done": done,
                }
                for shot, layout, encoded, done in zip(
                    self.shots, self.layouts, self.encoded, self.windows_done, strict=True
                )
            ],
            "segments": [
                {
                    "name": name,
                    "start": segment.start,
                    "end": segment.end,
                    "finished": done,
                    "bytes": size,
                }
                for segment, name, done, size in zip(
                    self.segments, self.files, self.finished, self.sizes, strict=True
                )
            ],
        }


def _input(part: Part) -> dict[str, Any]:
    """An input file as the manifest records it: the file itself (where it is, its size and
    modification time, read when the job starts) and what the first pass found in it."""
    path = part.source.path.resolve()
    status = path.stat()
    return {
        "path": str(path),
        "bytes": status.st_size,
        "modified_ns": status.st_mtime_ns,
        "start": part.start,
        "frames": part.source.frames,
        "frame_rate": str(part.source.stream.frame_rate),
        "size": [part.source.stream.width, part.source.stream.height],
        "sample_aspect": str(part.source.sample_aspect),
        "read_as": part.source.conversion.describe(),
        # The format decoded, and the tags the output copies (writer.Tags).
        "pix_fmt": part.source.stream.pix_fmt,
        "primaries": part.source.stream.color_primaries,
        "transfer": part.source.stream.color_transfer,
    }


def code_sha256() -> str:
    """The SHA-256 of seedvr2x's own files, the vendored model code included: the code a job's
    bits come from, which a resume must find again; a version number doesn't change with it."""
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            digest.update(str(path.relative_to(package)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def sha256(path: Path) -> str:
    """The SHA-256 of a file, in hex: models are identified by hash (DESIGN.md, Options kept and
    dropped)."""
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()

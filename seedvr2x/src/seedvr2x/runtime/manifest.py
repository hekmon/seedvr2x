"""The manifest of a job's output (DESIGN.md, Pause and resume): what the job is, which a resume
must find again (its settings, models by hash, environment, input file, output, shots with their
windows, output segments), and how far it went: each shot encoded, its windows done, each
segment finished. A unit is recorded once its file is whole, so the manifest only names whole
files: the frame index too, written before the manifest's first write (INDEX). Written once the
models load, then again after every unit, and with the index of a first pass a resume ran again
(cli._index_made_again), always whole (a temporary file, synced, renamed)."""

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from seedvr2x.media.files import write_whole
from seedvr2x.media.source import Source
from seedvr2x.runtime.job import JobError, OutputSegment, Shot

NAME = "manifest.json"
# 3 since the input is one record, the source's, where it was a list: a change of shape bumps the
# version (DESIGN.md, Pause and resume), so that older code, which reads its own version only,
# refuses a newer manifest cleanly. At version 2, the record (3ec48b9) made the code before it
# fail on a resume, its resume.identity taking the record for a list (AttributeError).
VERSION = 3
# The versions read: this code's, and 2, older code's, its input a list of one per input file or
# already the source's record, so that a resume refuses its job as another job, its differences
# listed (settings.code always among them: the code changed), as it did before version 3, rather
# than for its version alone; and so that verify still checks its segments, recorded as they
# are now. Version 1, from before resume (7c7c90d), is refused.
READ = (2, VERSION)
# The directory of the units kept for a resume, beside the manifest (units.DiskUnits).
STATE = "resume"
# The directory of the output segments' checksums, beside them, which outlives the job
# (media/checksums.py; DESIGN.md, Output, Checksums).
SUMS = "checksums"
# The source's frame index, beside the manifest, which names it in its input record (_input):
# written once with the first pass's record, kept with the manifest, which a finished job's run
# again reads too (DESIGN.md, Input; media/index.py). PROVISIONAL: the name, and the record's
# fields, name, bytes and sha256, DESIGN.md's "names it by its SHA-256".
INDEX = "frame_index.bin"


def checksums_file(segment: str, output_format: str) -> str:
    """The name, in SUMS, of the checksums of the output segment named `segment`: an FFV1 file's
    name without the .mkv seedvr2x gives it, or a PNG directory's name, as it is; then .crc32.
    Unique as the segments' own names are."""
    stem = segment if output_format == "png" else segment.removesuffix(".mkv")
    return f"{stem}.crc32"


def read(path: Path) -> dict[str, Any]:
    """The manifest at path, as written; refused unless of a version this code reads (READ)."""
    try:
        content: Any = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise JobError(f"{path}: not readable as a manifest: {error}") from None
    if not isinstance(content, dict):
        raise JobError(f"{path}: not a manifest")
    manifest = cast(dict[str, Any], content)
    found = manifest.get("seedvr2x_manifest")
    if found not in READ:
        # Said as written: a number as it is, anything else as JSON writes it ("3", a string).
        said = "no manifest version" if found is None else f"manifest version {json.dumps(found)}"
        raise JobError(f"{path}: {said}, where this seedvr2x writes {VERSION}")
    return manifest


@dataclass
class Manifest:
    """A job's manifest, at path."""

    path: Path
    settings: dict[str, Any]
    environment: dict[str, Any]  # what the output's bits depend on besides the settings
    source: Source  # the job's only input (DESIGN.md, Input)
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
    # Called once, before the manifest's next write, then dropped: it writes the frame index that
    # write names, made again by a resume in another environment (cli._index_made_again).
    before_write: Callable[[], None] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.encoded:
            self.encoded = [False] * len(self.shots)
        if not self.windows_done:
            self.windows_done = [0] * len(self.shots)
        if not self.finished:
            self.finished = [False] * len(self.segments)
        if not self.sizes:
            self.sizes = [None] * len(self.segments)
        self._input = _input(self.source)

    @property
    def split(self) -> bool:
        """Whether the job corrects colours with `split`, each shot then keeping a copy of its
        input frames from its encode (units.COPY)."""
        return self.settings.get("color_correction") == "split"

    def shot_encoded(self, index: int) -> None:
        """Record shot `index`'s latent as kept, and its input copy with `split`, and write the
        manifest."""
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
        if self.before_write is not None:
            self.before_write()
            self.before_write = None
        write_whole(self.path, (json.dumps(self.content(), indent=2) + "\n").encode())

    def content(self) -> dict[str, Any]:
        seed = self.settings["seed"]
        return {
            "seedvr2x_manifest": VERSION,
            "settings": self.settings,
            "environment": self.environment,
            "environment_changes": self.environment_changes,
            "input": self._input,
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


def _input(source: Source) -> dict[str, Any]:
    """The source as the manifest records it: the file itself (where it is, its size and
    modification time, read when the job starts, and its content's SHA-256) and what the first
    pass found in it; with its frame index, the index's file, INDEX, by its name, size and
    SHA-256. Its content says which file it is; where it is and when it was modified are
    information (resume.UNCOMPARED), as the index is: derived data, which a resume checks against
    this record before trusting it, and makes again when missing or damaged (cli._prior)."""
    path = source.path.resolve()
    status = path.stat()
    record: dict[str, Any] = {
        "path": str(path),
        "bytes": status.st_size,
        "modified_ns": status.st_mtime_ns,
        "sha256": source.sha256,
        "frames": source.frames,
        "frame_rate": str(source.stream.frame_rate),
        "size": [source.stream.width, source.stream.height],
        "sample_aspect": str(source.sample_aspect),
        "read_as": source.conversion.describe(),
        # The format decoded, and the tags the output copies (writer.Tags).
        "pix_fmt": source.stream.pix_fmt,
        "primaries": source.stream.color_primaries,
        "transfer": source.stream.color_transfer,
    }
    if source.index is not None:
        data = source.index.to_bytes()
        digest = hashlib.sha256(data).hexdigest()
        record["index"] = {"name": INDEX, "bytes": len(data), "sha256": digest}
    return record


def code_sha256() -> str:
    """The SHA-256 of seedvr2x's own files, the vendored model code included: the code a job's
    bits come from, which a resume must find again; a version number doesn't change with it."""
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            digest.update(str(path.relative_to(package)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()

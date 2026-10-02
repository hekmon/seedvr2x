"""The manifest of a job's output: its settings, its input, its shots and its output segments, and
which segments are finished (DESIGN.md, Pause and resume). A skeleton: written beside the
segments before any GPU work, then again as each segment is finished; resume will read it back,
and refuse to go on with other settings.

Provisional: its fields are this code's, until resume settles what it needs (model hashes
included: only the weights' names and sizes are recorded yet)."""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from seedvr2x.runtime.job import OutputSegment, Part, Shot

NAME = "manifest.json"
VERSION = 1


@dataclass
class Manifest:
    """A job's manifest, at path."""

    path: Path
    settings: dict[str, Any]
    parts: Sequence[Part]
    shots: Sequence[Shot]
    segments: Sequence[OutputSegment]
    files: Sequence[str]  # each segment's file (FFV1) or directory (PNG), in the output
    output: dict[str, Any]
    finished: list[bool] = field(default_factory=list[bool])

    def __post_init__(self) -> None:
        if not self.finished:
            self.finished = [False] * len(self.segments)

    def segment_finished(self, index: int) -> None:
        """Record segment `index` as finished, whole and checked, and write the manifest."""
        self.finished[index] = True
        self.write()

    def write(self) -> None:
        """Write the manifest, whole or not at all: to a temporary file, then renamed."""
        partial = self.path.with_name(f"{self.path.name}.partial")
        partial.write_text(json.dumps(self.content(), indent=2) + "\n")
        partial.replace(self.path)

    def content(self) -> dict[str, Any]:
        seed = self.settings["seed"]
        return {
            "seedvr2x_manifest": VERSION,
            "settings": self.settings,
            "input": [
                {
                    "path": str(part.source.path),
                    "start": part.start,
                    "frames": part.source.frames,
                    "frame_rate": str(part.source.stream.frame_rate),
                    "size": [part.source.stream.width, part.source.stream.height],
                    "sample_aspect": str(part.source.sample_aspect),
                    "read_as": part.source.conversion.describe(),
                }
                for part in self.parts
            ],
            "output": self.output,
            "shots": [
                {"start": shot.start, "end": shot.end, "seed": shot.seed(seed)}
                for shot in self.shots
            ],
            "segments": [
                {"name": name, "start": segment.start, "end": segment.end, "finished": done}
                for segment, name, done in zip(
                    self.segments, self.files, self.finished, strict=True
                )
            ],
        }

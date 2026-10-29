"""The manifest of a job's output (DESIGN.md, Pause and resume): what the job is, which a resume
must find again (its settings, models by hash, environment, input file, output, plan, shots with
their windows, output segments), and how far it went: each shot encoded, its windows done, each
segment finished, and, for a .mkv path, the finished file once assembled (ASSEMBLED). It sits in
the output directory, or in a .mkv path's work directory (cli._output). A unit is recorded once its
file is whole, so the manifest only names whole files: the frame index too, written before the
manifest's first write (INDEX). Written once the models load, or before any does when the job is
planned (--plan: its plan, no unit made), then again after every unit; and, before a unit, with the
index of a first pass a resume ran again, or for a plan taken up or planned again (cli._taken_up,
cli._replan); always whole (a temporary file, synced, renamed)."""

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from seedvr2x.media.files import write_whole
from seedvr2x.media.source import Source
from seedvr2x.runtime.job import (
    QUALITY,
    JobError,
    OutputSegment,
    Run,
    Shot,
    shot_runs,
    window_layout,
)

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
# The directory of the units kept for a resume, beside the manifest (units.Units).
STATE = "resume"
# The directory of the output segments' checksums, beside them, which outlives the job
# (media/checksums.py; DESIGN.md, Output, Checksums).
SUMS = "checksums"
# The source's frame index, beside the manifest, which names it in its input record (_input):
# written once with the first pass's record, kept with the manifest, which a finished job's run
# again reads too (DESIGN.md, Input; media/index.py). The record's fields, name, bytes and
# sha256, are DESIGN.md's (Pause and resume: "the frame index by its name, size and SHA-256");
# the name itself is the code's, accepted by design as built (2026-10-10).
INDEX = "frame_index.bin"
# What the first pass depends on in the environment: it runs on ffmpeg's decode alone, so that
# its record stands through any other change accepted (DESIGN.md, Pause and resume). The shot
# detector's probabilities, which torch computes on the GPU in that pass, stand with it: they are
# the record's, so that a job's cuts stay its own (DESIGN.md, Pause and resume: "The shot
# detector's probabilities are part of that record, though torch computes them: a torch, CUDA,
# GPU or attention change doesn't run the detector again"). So an index made again under such a
# change accepted, its ffmpeg the same, is written with the manifest as recorded, the change
# with the next unit (cli._taken_up; kept by design, 2026-10-10). Here, not in runtime/resume.py,
# which imports torch (runtime/units.py): the command line reads it before it checks the model
# files, which comes before torch (cli._trusted; DESIGN.md, Weights, Recognised by content).
FIRST_PASS = ("ffmpeg", "conversions")
# The setting of the level, --quality's, which says the runs a shot (job.QUALITY; cli._settings;
# DESIGN.md, Several runs of a shot): standard, one run, for a manifest recorded before it, which
# code before it wrote (its settings.code another job's anyway).
LEVEL = "quality"
# A shot's record of its runs after the first, with more than one run a shot: their own windows
# and how far each went, the first run being the shot's own fields (Manifest.content). With one
# run a shot, the shots' record is what it was before the option, byte for byte (the settings
# gain the level, standard).
SHOT_RUNS = "runs"
# The record of the finished file a .mkv path's segments were assembled into (cli._assemble): its
# name and bytes, written once the file is in place, before its work directory, this manifest's,
# is removed. How far the job went, never compared on a resume (resume.differences): it stands
# for a stop between the two, so that the next run finishes the removal instead of refusing a
# directory half removed (cli._assembled). An additive field, absent until then.
ASSEMBLED = "assembled"
# The input's record of the remux rehearsal a .mkv path's job makes of the source's other streams
# before its first pass (assembly.preflight; cli._copied), once it passed: {"passed": true,
# "ffmpeg": the version it ran under, "conversions": the fingerprint}. A resume that finds the
# same content (cli._prior compares the input's size and SHA-256) and the same two trusts it and
# rehearses nothing (DESIGN.md, Output, Assembly, "Met on 2026-10-23", decided on 2026-10-28: "its
# result is recorded with the first pass's, a resume trusting it for the same content and
# `ffmpeg`"). Additive, no version bump: a record without it is rehearsed again; information for
# a resume's comparison (resume.UNCOMPARED), the content it stands for being compared as the input.
REHEARSAL = "rehearsal"
# A run's own window cap, recorded only where its windows are not those of the job's plan: a run
# kept as it was made when a resume planned the rest again for the memory free (--accept-env-change;
# DESIGN.md, Memory planner, A plan shapes the output a little, decided on 2026-10-28), its windows
# laid out by a cap the plan no longer holds. Additive, no version bump: a reader of version 3
# takes the field for an unknown one and, laying the run out by the job's plan, finds the windows
# recorded differ, and refuses the job as another, as it does for a plan changed (the version
# bumps only where older code would misread a field, which this one it can't: DESIGN.md, Pause and
# resume, Manifest). A job never planned again records none, byte for byte as before.
WINDOW_CAP = "window_cap"


def write_content(path: Path, content: dict[str, Any]) -> None:
    """Write a manifest's content to path, whole or not at all, even after a power cut: a
    temporary file, fsync, rename, then a directory fsync (media.files.write_whole). The job's
    manifest after every unit (Manifest.write), and its copy beside a finished file (cli.
    _assemble; DESIGN.md, Output, Assembly, decided on 2026-10-12)."""
    write_whole(path, (json.dumps(content, indent=2) + "\n").encode())


def checksums_file(segment: str, output_format: str) -> str:
    """The name, in SUMS, of the checksums of the output segment named `segment`: an FFV1 file's
    name without the .mkv seedvr2x gives it, or a PNG directory's name, as it is; then .crc32.
    Unique as the segments' own names are."""
    stem = segment if output_format == "png" else segment.removesuffix(".mkv")
    return f"{stem}.crc32"


def planned(content: dict[str, Any]) -> bool:
    """Whether the job a manifest's content records has made no unit yet: no shot encoded, no
    window kept, no segment finished. Such a job is a plan: written by --plan, or by a run
    stopped before its first unit. False for a record this code can't read so: another job's
    anyway (cli._prior). Here, not in runtime/resume.py, which imports torch: the command line
    asks before it checks the model files (cli._run), which comes before torch."""
    shots: Any = content.get("shots")
    segments: Any = content.get("segments")
    try:
        made = any(
            run["encoded"] or run["windows_done"] for shot in shots for run in recorded_runs(shot)
        )
        return not made and not any(segment["finished"] for segment in segments)
    except (KeyError, TypeError):
        return False


def complete(content: dict[str, Any]) -> bool:
    """Whether the job a manifest's content records has finished every segment, a run of it then
    making nothing (cli._run). False for a record this code can't read so, as planned is; here
    for the same reason."""
    segments: Any = content.get("segments")
    try:
        return bool(segments) and all(segment["finished"] for segment in segments)
    except (KeyError, TypeError):
        return False


def recorded_runs(shot: Any) -> list[Any]:
    """Of a shot's record in a manifest's content, its runs' records, each with its windows and how
    far it went: the shot's own, the first run's, then those of SHOT_RUNS, which only a job of
    more than one run a shot records."""
    return [shot, *shot.get(SHOT_RUNS, [])]


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
    # Each run's DiT windows, latents [start, end), in the order of runs: with one run a shot,
    # each shot's.
    layouts: Sequence[Sequence[tuple[int, int]]]
    segments: Sequence[OutputSegment]
    files: Sequence[str]  # each segment's file (FFV1) or directory (PNG), in the output
    output: dict[str, Any]
    # The job's plan (Plan.record; DESIGN.md, Memory planner): what shapes the output, which a
    # resume must find again, and for information what doesn't (resume.UNCOMPARED). A field added
    # to the record, whose shape stays: VERSION too.
    # None, before the plan is made: a run refused after its first pass leaves the record of the
    # pass, a plan with no plan's record (cli._run).
    plan: dict[str, Any] | None
    # The environment changes a resume accepted (--accept-env-change), oldest first: the units
    # made before each are the environment's before it (resume.environment_change).
    environment_changes: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    # How far the job went, run by run (runs): with one run a shot, shot by shot.
    encoded: list[bool] = field(default_factory=list[bool])  # each run's latent kept
    windows_done: list[int] = field(default_factory=list[int])  # each run's windows kept
    finished: list[bool] = field(default_factory=list[bool])  # each segment whole
    sizes: list[int | None] = field(default_factory=list[int | None])  # finished ones' bytes
    # Finished ones' SHA-256 (units.digest), read back at the assembly (cli._damaged): None for a
    # segment not finished, or recorded by code before the field, which a code change refuses
    # to resume (settings.code) anyway.
    hashes: list[str | None] = field(default_factory=list[str | None])
    assembled: dict[str, Any] | None = None  # the finished file, once in place (ASSEMBLED)
    # The remux rehearsal of a .mkv path's job, passed (REHEARSAL), recorded with the first pass's
    # in the input's record; None for a directory's job, which makes none.
    rehearsal: dict[str, Any] | None = None
    # Each run's own window cap (WINDOW_CAP), None where its windows are the job's plan's: with
    # layouts, those windows laid out by the cap (job.window_layout), and by the plan's window
    # when it is None. In the order of runs.
    window_caps: list[int | None] = field(default_factory=list[int | None])
    # Called once, before the manifest's next write, then dropped: it writes the frame index that
    # write names, made again by a resume in another environment (cli._taken_up).
    before_write: Callable[[], None] | None = field(default=None, repr=False, compare=False)
    # Called once, after the manifest's next write, then dropped: it discards the files the
    # manifest before that write named and this one doesn't, the windows of a shape planned again
    # (cli._reshape), which only then are unnamed. Units writes the manifest before the first file
    # it keeps (Units.settle), so that no file of the new shape is ever under a manifest that
    # names the old.
    after_write: Callable[[], None] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        level: Any = self.settings.get(LEVEL, "standard")
        # A hand-edited record's level may be any JSON value: a list or an object, unhashable,
        # refused as another value is.
        if not isinstance(level, str) or level not in QUALITY:
            *others, last = QUALITY
            levels = f"{', '.join(others)} and {last}"
            raise JobError(f"{self.path}: quality {json.dumps(level)}, where seedvr2x has {levels}")
        count = self.runs_per_shot
        # The job's runs, the units its encodes and windows are (DESIGN.md, Several runs of a
        # shot): each shot's in turn, and the shot each is of.
        self.runs: list[Run] = [run for shot in self.shots for run in shot_runs(shot, count)]
        self.run_shots: list[int] = [
            index for index, shot in enumerate(self.shots) for _ in shot_runs(shot, count)
        ]
        if len(self.layouts) != len(self.runs):
            raise ValueError(f"{len(self.layouts)} layouts for {len(self.runs)} runs")
        if not self.window_caps:
            self.window_caps = [None] * len(self.runs)
        if not self.encoded:
            self.encoded = [False] * len(self.runs)
        if not self.windows_done:
            self.windows_done = [0] * len(self.runs)
        if not self.finished:
            self.finished = [False] * len(self.segments)
        if not self.sizes:
            self.sizes = [None] * len(self.segments)
        if not self.hashes:
            self.hashes = [None] * len(self.segments)
        self._input = _input(self.source, self.rehearsal)

    @property
    def runs_per_shot(self) -> int:
        """The runs of each shot, its level's (LEVEL, job.QUALITY): 1, 2 or 4."""
        return QUALITY[self.settings.get(LEVEL, "standard")]

    def shot_of(self, run: int) -> int:
        """The index of the shot run `run` is of."""
        return self.run_shots[run]

    @property
    def split(self) -> bool:
        """Whether the job corrects colours with `split`, each shot then keeping a copy of its
        input frames from its encode (units.COPY)."""
        return self.settings.get("color_correction") == "split"

    def shot_encoded(self, index: int) -> None:
        """Record run `index`'s latent as kept, and with `split` the input copy of its shot
        with a shot's first run, which makes it, and write the manifest."""
        self.encoded[index] = True
        self.write()

    def window_done(self, index: int, window: int) -> None:
        """Record window `window` of run `index` as kept, the windows before it being kept
        already, and write the manifest."""
        if window != self.windows_done[index]:
            raise ValueError(f"run {index}: window {window} after {self.windows_done[index]}")
        self.windows_done[index] = window + 1
        self.write()

    def segment_finished(self, index: int, size: int, sha256: str) -> None:
        """Record segment `index` as finished, whole and checked, `size` bytes of SHA-256
        `sha256`, and write the manifest."""
        self.finished[index] = True
        self.sizes[index] = size
        self.hashes[index] = sha256
        self.write()

    def segment_damaged(self, index: int) -> None:
        """Record segment `index`, finished, as not finished again, its bytes no longer its
        record's (cli._damaged; DESIGN.md, Output, Assembly, decided on 2026-10-12: "a damaged one
        made again"), and write the manifest. Its shots' units went when it was finished, so
        its runs are recorded as not begun: the next run encodes, samples and decodes them
        again."""
        self.finished[index] = False
        self.sizes[index] = None
        self.hashes[index] = None
        window: Any = (self.plan or {}).get("window")
        for shot in self.segment_shots(index):
            for run in self.shot_runs(shot):
                self.encoded[run] = False
                self.windows_done[run] = 0
                if self.window_caps[run] is not None:
                    # Made again under the job's plan, not the cap it was made under before the
                    # plan was made again for the memory free: that cap may no longer fit.
                    latents = self.layouts[run][-1][1]
                    layouts = list(self.layouts)
                    layouts[run] = window_layout(latents, window or latents)
                    self.layouts = layouts
                    self.window_caps[run] = None
        self.write()

    def job_assembled(self, name: str, size: int) -> None:
        """Record the job's segments as assembled into the file `name`, `size` bytes, whole and
        in place beside the manifest's directory, and write the manifest."""
        self.assembled = {"name": name, "bytes": size}
        self.write()

    def segment_shots(self, index: int) -> list[int]:
        """The shots of segment `index`: a segment holds whole shots."""
        segment = self.segments[index]
        return [i for i, shot in enumerate(self.shots) if segment.start <= shot.start < segment.end]

    def shot_runs(self, index: int) -> list[int]:
        """The runs of shot `index`, in order: their indices in runs."""
        return [run for run, shot in enumerate(self.run_shots) if shot == index]

    def write(self) -> None:
        """Write the manifest, whole or not at all, even after a power cut."""
        if self.before_write is not None:
            self.before_write()
            self.before_write = None
        write_content(self.path, self.content())
        if self.after_write is not None:
            self.after_write()
            self.after_write = None

    def content(self) -> dict[str, Any]:
        seed = self.settings["seed"]
        shots: list[dict[str, Any]] = []
        for index, shot in enumerate(self.shots):
            first, *others = self.shot_runs(index)
            entry: dict[str, Any] = {"start": shot.start, "end": shot.end, "seed": shot.seed(seed)}
            entry.update(self._run(first))
            if self.runs_per_shot > 1:
                entry[SHOT_RUNS] = [
                    {"from": self.runs[run].offset, **self._run(run)} for run in others
                ]
            shots.append(entry)
        return {
            "seedvr2x_manifest": VERSION,
            "settings": self.settings,
            "environment": self.environment,
            "environment_changes": self.environment_changes,
            "input": self._input,
            "output": self.output,
            "plan": self.plan,
            "shots": shots,
            "segments": [
                {
                    "name": name,
                    "start": segment.start,
                    "end": segment.end,
                    "finished": done,
                    "bytes": size,
                    "sha256": digest,
                }
                for segment, name, done, size, digest in zip(
                    self.segments,
                    self.files,
                    self.finished,
                    self.sizes,
                    self.hashes,
                    strict=True,
                )
            ],
            **({} if self.assembled is None else {ASSEMBLED: self.assembled}),
        }

    def _run(self, run: int) -> dict[str, Any]:
        """Run `run`'s record: its latents, its windows and how far it went."""
        layout = self.layouts[run]
        record: dict[str, Any] = {
            "latents": layout[-1][1],
            "windows": [list(window) for window in layout],
            "encoded": self.encoded[run],
            "windows_done": self.windows_done[run],
        }
        if self.window_caps[run] is not None:
            record[WINDOW_CAP] = self.window_caps[run]
        return record


def _input(source: Source, rehearsal: dict[str, Any] | None = None) -> dict[str, Any]:
    """The source as the manifest records it: the file itself (where it is, its size and
    modification time, read when the job starts, and its content's SHA-256) and what the first
    pass found in it, idet's counts included; with its frame index, the index's file, INDEX, by
    its name, size and SHA-256. Its content says which file it is; where it is and when it was
    modified are information (resume.UNCOMPARED), as the index is: derived data, which a resume
    checks against this record before trusting it, and makes again when missing or damaged
    (cli._prior); and as idet's counts are."""
    path = source.path.resolve()
    status = path.stat()
    record: dict[str, Any] = {
        "path": str(path),
        "bytes": status.st_size,
        "modified_ns": status.st_mtime_ns,
        "sha256": source.sha256,
        "frames": source.frames,
        # The rate it declares; the job's, --frame-rate's when given, is the output's (cli._run).
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
        filled = source.filled
        if filled is not None and filled.first is not None:
            # Only for a source with places left empty on its rate's grid, each filled by the
            # frame before it, held (DESIGN.md, Input: "says so ... and records it"): "frames"
            # above counts the places; here the frames ffmpeg decodes, the holes, the places
            # they fill, the first one's first place with its length, and the longest's, in
            # frames. Information drawn from the index, whose SHA-256 above holds which rows are
            # held; a resume compares it as it compares "frames", its index's rows too. A
            # source without a hole has the record of before the places, byte for byte.
            record["filled"] = {
                "decoded": filled.decoded,
                "holes": filled.holes,
                "places": filled.filled,
                "first": {"frame": filled.first.first, "length": filled.first.length},
                "longest": filled.longest,
            }
    if rehearsal is not None:
        record[REHEARSAL] = rehearsal
    if source.idet is not None:
        # For information, not compared (resume.UNCOMPARED): what idet found, the warning's
        # figures (source.combing), said again by a resume trusting this record.
        record["idet"] = source.idet.record()
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

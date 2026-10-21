"""A job resumed from its output directory, or from a .mkv path's work directory (cli._output),
which is one (DESIGN.md, Pause and resume), the manifest there being the truth: the job asked
must be the one recorded, its settings, models, environment, input,
output and layout alike, but for an environment change the user accepts (--accept-env-change),
which the manifest records, and for a plan's settings, those its first pass doesn't read
(FIRST_PASS_SETTINGS); the directory must hold what the manifest says, and nothing else of
anyone's; what a stop left that the manifest doesn't name is discarded."""

import json
import os
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from seedvr2x.media.assembly import work_file
from seedvr2x.media.files import command_partial, partial_path
from seedvr2x.media.source import Source
from seedvr2x.runtime.job import JobError, OutputSegment, Shot
from seedvr2x.runtime.manifest import (
    ASSEMBLED,
    INDEX,
    NAME,
    SHOT_RUNS,
    STATE,
    SUMS,
    Manifest,
    checksums_file,
    recorded_runs,
)
from seedvr2x.runtime.tiling import SCRATCH
from seedvr2x.runtime.units import CHECKSUMS, COPY, KEPT, run_file, size

# The fields a resume reads and doesn't compare: how far the job went, in its shots and segments;
# what is recorded for information only (DESIGN.md, Pause and resume): where an input is and when
# it was modified, an input being its content, so that a moved input is the same; the NVIDIA
# driver, since the math kernels ship with torch. And the frame index's file ("the frame index by
# its name, size and SHA-256, and idet's counts, for information"; "The frame index is derived
# data: missing or damaged on a resume, the first pass runs again and writes it"), checked
# against its record when the first pass's record is trusted, made again with the pass otherwise
# (cli._prior), whose new decode may hash some frames otherwise after an ffmpeg change accepted,
# which is no other job. idet's counts likewise, information (manifest._input). And of the plan,
# what doesn't shape the output (DESIGN.md, Memory planner): where the weights are (Plan.record's
# memory), and its timings, which the time estimates will record.
UNCOMPARED = {
    "shots": ("encoded", "windows_done"),
    "segments": ("finished", "bytes"),
    "input": ("path", "modified_ns", "index", "idet"),
    "environment": ("driver",),
    "plan": ("memory", "timings"),
}

# The manifest's record of the environment changes a resume accepted (Manifest.content).
CHANGES = "environment_changes"

# The names units.Units gives a shot's files, a later run's units named for it (units.run_file),
# and the scratch a tiled pass keeps beside them (tiling.SCRATCH) and the frames a shot's runs
# keep for its decode (units.KEPT): derived data, never recorded, so a leftover wherever a kill
# left it.
UNIT_FILE = re.compile(
    rf"(?:run[1-3]_)?(?:latent\.pt|window_\d{{4}}\.pt)"
    rf"|{'|'.join(map(re.escape, (COPY, CHECKSUMS, SCRATCH, KEPT)))}"
)

# The settings that say how a job's source is cut (cli._settings): the cut list given, or the
# detection's threshold and the shot detector's file, which a cut list drops. A plan, a job that
# has made no unit yet (manifest.planned), is taken up with them changed, its cuts derived again
# from the first pass's record, or taken from the new list (cli._prior, cli._replan), as it is
# with any other setting its first pass doesn't read (FIRST_PASS_SETTINGS). DESIGN.md, Pause and
# resume, A plan: "A run whose settings differ only in its cuts (the cut list, the threshold, the
# detector's file) plans it again from its record, with no decode". Named here for that rule,
# which FIRST_PASS_SETTINGS's covers.
CUT_SETTINGS = frozenset({"cuts", "cut_threshold", "detector_model"})

# The settings a plan can't change (DESIGN.md, Pause and resume, A plan, decided on 2026-10-21: "A
# difference in what the first pass reads (the source, the frame rate), in the code, or a unit
# made, is another job"): --frame-rate's rate, which the pass judges the source's timing by and
# fills its grid at (media/source.py: timing_refusal, in_places), and the code, seedvr2x's
# version and its sources' hash. The pass reads nothing else of the settings: the source's
# decode (media/scan.py's scan) takes its path alone, and the shot detector its file, a cut
# setting (CUT_SETTINGS), pinned (pull.check_pinned). The source is the input's record, compared
# as itself (cli._prior). Any other setting differing, a plan, a job that has made no unit yet
# (manifest.planned), is planned again from its record, with no decode (cli._prior,
# cli._replan): the cuts, the quality level, the pins, the resolution, the models, the colour
# correction, the output's format, the segments' command and their least length, the input's
# matrix and sample aspect, the seed, and the tests' numz modes.
FIRST_PASS_SETTINGS = frozenset({"seedvr2x", "code", "frame_rate"})


def replannable(difference: str) -> bool:
    """Whether a difference between a plan's record and the job asked is one a plan takes,
    planned again (FIRST_PASS_SETTINGS): a setting, any but those the first pass reads."""
    name = setting(difference)
    return name is not None and name not in FIRST_PASS_SETTINGS


def identity(content: dict[str, Any]) -> dict[str, Any]:
    """Of a job's manifest content, what a resume compares before the first pass: the settings,
    the environment, and the source's content, its size and SHA-256. The first pass's record is
    trusted when they are the same, the pass then not run again (DESIGN.md, Pause and resume)."""
    recorded: Any = content.get("input")
    # One input, the source (DESIGN.md, Input). A manifest of version 2 (manifest.READ) may hold a
    # list, as older code recorded, of a directory's segments or of the one file: compared as it
    # is, it differs, another job's, as the manifest's settings.code makes it already.
    source = recorded
    if isinstance(recorded, dict):
        entry = cast(dict[str, Any], recorded)
        source = {"bytes": entry.get("bytes"), "sha256": entry.get("sha256")}
    return {
        "settings": content.get("settings"),
        "environment": content.get("environment"),
        "input": source,
    }


def differences(recorded: dict[str, Any], asked: dict[str, Any]) -> list[str]:
    """What differs between the job recorded and the job asked, both as Manifest.content gives
    them, how far they went aside: one line each, `where: recorded -> now`."""
    found: list[str] = []
    _compare(_job(recorded), _job(json.loads(json.dumps(asked))), "", found)
    return found


def adopt(manifest: Manifest, recorded: dict[str, Any]) -> None:
    """Take how far the recorded job went into manifest, the same job (differences: none, but
    the environment's when accepted), and the environment changes accepted before."""
    runs = [run for shot in recorded["shots"] for run in recorded_runs(shot)]
    if len(runs) != len(manifest.runs):
        raise JobError(
            f"{manifest.path}: {len(runs)} runs recorded, for the {len(manifest.runs)} of its"
            f" {len(manifest.shots)} shots at {manifest.runs_per_shot} a shot"
        )
    manifest.encoded = [run["encoded"] for run in runs]
    manifest.windows_done = [run["windows_done"] for run in runs]
    manifest.finished = [segment["finished"] for segment in recorded["segments"]]
    manifest.sizes = [segment["bytes"] for segment in recorded["segments"]]
    manifest.environment_changes = list(recorded.get(CHANGES, []))


def section(difference: str) -> str:
    """The manifest's field a difference is in: environment, for `environment.gpu: ...`."""
    return re.split(r"[.\[:]", difference, maxsplit=1)[0]


def setting(difference: str) -> str | None:
    """The setting a difference is in: seed, for `settings.seed: 42 -> 7`, cuts for
    `settings.cuts[0]: 3 -> 4`; None for a difference outside the settings."""
    found = re.match(r"settings\.(\w+)", difference)
    return None if found is None else found[1]


def recorded_job(path: Path, content: dict[str, Any], source: Source) -> Manifest:
    """The job the manifest at path records, its content `content`, as Manifest holds it, with
    how far it went, `source` its input: so that what a stop left of a plan is found by the
    plan's own layout (leftovers) when it is re-planned (cli._replan)."""
    shots: list[Any] = content["shots"]
    entries: list[Any] = content["segments"]
    files = [str(entry["name"]) for entry in entries]
    # The plan as recorded, which nothing reads here: the leftovers go by the layout. One
    # deleted by hand, or no record of a plan's, is none: its job, which has made no unit, is
    # planned again whole whatever it recorded (cli._resume).
    plan: Any = content.get("plan")
    job = Manifest(
        path,
        content["settings"],
        content["environment"],
        source,
        [Shot(int(shot["start"]), int(shot["end"])) for shot in shots],
        [
            [(int(first), int(end)) for first, end in run["windows"]]
            for shot in shots
            for run in recorded_runs(shot)
        ],
        [
            OutputSegment(name.removesuffix(".mkv"), int(entry["start"]), int(entry["end"]))
            for name, entry in zip(files, entries, strict=True)
        ],
        files,
        content["output"],
        cast(dict[str, Any], plan) if isinstance(plan, dict) else {},
    )
    adopt(job, content)
    return job


def clear(directory: Path, content: dict[str, Any]) -> list[str]:
    """Remove from directory, a .mkv path's work directory, the files of the job its manifest's
    content records, once its segments are assembled into the finished file (cli._assemble): its
    segments, their checksums, what a stop left of its units and of its partial files, the frame
    index; then the manifest, last, so that a removal stopped half way finds it again and is
    finished by the next run (cli._assembled); then the directory itself. Names missing are fine.
    Returns what is there that isn't this job's, as paths relative to directory, sorted: left as
    it is, never deleted, the manifest and the directory then left with it, the manifest saying
    whose it is."""
    files = [str(entry["name"]) for entry in content["segments"]]
    shots = {f"shot_{int(shot['start']):06d}" for shot in content["shots"]}
    sums = {checksums_file(name, content["output"]["format"]) for name in files}
    ours = {INDEX, partial_path(directory / INDEX).name, partial_path(directory / NAME).name}
    for name in files:
        # A segment command's partial file too, whether the job had one or not: the name is no
        # one else's in a work directory.
        partials = (partial_path(directory / name), command_partial(directory / name))
        ours.update((name, *(partial.name for partial in partials)))
    foreign: list[str] = []

    def plain(entry: Path) -> bool:
        return entry.is_file() and not entry.is_symlink()

    def emptied(entry: Path, mine: Callable[[str], bool]) -> None:
        """Remove from entry, a directory of this job's, the plain files `mine` knows by name,
        their partial files too, then entry itself when nothing else is left; what else it
        holds is foreign, as entry is when it isn't a directory."""
        if entry.is_symlink() or not entry.is_dir():
            foreign.append(str(entry.relative_to(directory)))
            return
        for file in sorted(entry.iterdir()):
            if mine(file.name.removesuffix(".partial")) and plain(file):
                file.unlink()
            else:
                foreign.append(str(file.relative_to(directory)))
        if not any(entry.iterdir()):
            entry.rmdir()

    for entry in sorted(directory.iterdir()):
        if entry.name == NAME:
            continue
        if entry.name == SUMS:
            emptied(entry, sums.__contains__)
        elif entry.name == STATE and entry.is_dir() and not entry.is_symlink():
            for shot in sorted(entry.iterdir()):
                if shot.name in shots:
                    emptied(shot, lambda name: UNIT_FILE.fullmatch(name) is not None)
                else:
                    foreign.append(str(shot.relative_to(directory)))
            if not any(entry.iterdir()):
                entry.rmdir()
        elif (entry.name in ours or work_file(entry.name)) and plain(entry):
            # An assembly's own files too (assembly.work_file), which it removes but when killed.
            entry.unlink()
        else:
            foreign.append(entry.name)
    if not foreign:
        (directory / NAME).unlink(missing_ok=True)
        directory.rmdir()
    return sorted(foreign)


def environment_change(recorded: dict[str, Any], environment: dict[str, Any]) -> dict[str, Any]:
    """The record of a resume in another environment, accepted (--accept-env-change): when, what
    changed (the driver too, for information; of the packages, those changed), and how far the
    job had gone, in units made in the environment before (a run takes them in one order:
    segments, then their shots, then each shot's windows)."""
    before, after = _changes(recorded["environment"], json.loads(json.dumps(environment)))
    return {
        "accepted": datetime.now().astimezone().isoformat(timespec="seconds"),
        "before": before,
        "after": after,
        "segments_finished": sum(bool(segment["finished"]) for segment in recorded["segments"]),
        # With more than one run a shot, runs encoded and their windows, under the fields' names
        # of before the option: a shot's first run is the shot's own record (manifest.SHOT_RUNS).
        "shots_encoded": sum(
            bool(run["encoded"]) for shot in recorded["shots"] for run in recorded_runs(shot)
        ),
        "windows_done": sum(
            run["windows_done"] for shot in recorded["shots"] for run in recorded_runs(shot)
        ),
    }


def _changes(
    before: dict[str, Any], after: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The fields of two records that differ, with their values in each; those of a record
    within, such as the packages' versions, by its own fields."""
    old: dict[str, Any] = {}
    new: dict[str, Any] = {}
    for key in sorted({*before, *after}):
        one: Any = before.get(key)
        other: Any = after.get(key)
        if isinstance(one, dict) and isinstance(other, dict):
            inner = _changes(cast(dict[str, Any], one), cast(dict[str, Any], other))
            if inner != ({}, {}):
                old[key], new[key] = inner
        elif one != other:
            old[key], new[key] = one, other
    return old, new


def leftovers(manifest: Manifest) -> list[Path]:
    """What a stop left in the output directory that the manifest doesn't name, to discard: the
    partial files and directories, a segment command's partial file among them
    (files.command_partial), an unfinished segment's file and checksums, a shot's unit files not
    recorded, the units of finished segments. Refused (JobError): a finished segment missing, of
    another size or without its checksums, which a segment command's has none of
    (settings.segment_cmd), a unit recorded but missing, and anything in the directory that isn't
    this job's, which is never deleted. A shot's input copy or its checksums missing isn't refused:
    the copy is derived data, which the run makes again (run_job); nor is the frame index (INDEX),
    made again with the first pass (cli._prior)."""
    directory = manifest.path.parent
    found: list[Path] = []
    indexed = partial_path(directory / INDEX)
    names = {NAME, partial_path(manifest.path).name, STATE, SUMS, INDEX, indexed.name}
    if indexed.exists():
        found.append(indexed)
    commanded = manifest.settings.get("segment_cmd") is not None
    for index, name in enumerate(manifest.files):
        path = directory / name
        names.update((name, partial_path(path).name))
        if commanded:
            names.add(command_partial(path).name)
            if command_partial(path).exists():
                found.append(command_partial(path))
        if manifest.finished[index]:
            if not path.exists():
                raise JobError(f"{path}: finished, the manifest says, but missing")
            if size(path) != manifest.sizes[index]:
                raise JobError(
                    f"{path}: {size(path)} bytes, where the manifest recorded"
                    f" {manifest.sizes[index]}"
                )
        elif path.exists():
            found.append(path)
        if partial_path(path).exists():
            found.append(partial_path(path))
    if partial_path(manifest.path).exists():
        found.append(partial_path(manifest.path))
    foreign = sorted(entry.name for entry in directory.iterdir() if entry.name not in names)
    if foreign:
        raise JobError(f"{directory}: not this job's: {', '.join(foreign[:5])}")
    return found + _checksum_leftovers(manifest) + _unit_leftovers(manifest)


def _checksum_leftovers(manifest: Manifest) -> list[Path]:
    """In SUMS: an unfinished segment's checksums and partial files, to discard. A finished
    segment's are written before it is recorded, so their absence is refused, as anything there
    that isn't this job's is."""
    sums = manifest.path.parent / SUMS
    if os.path.lexists(sums) and (sums.is_symlink() or not sums.is_dir()):
        raise JobError(f"{sums}: not this job's")
    output_format = manifest.output["format"]
    finished = {
        checksums_file(name, output_format): done
        for name, done in zip(manifest.files, manifest.finished, strict=True)
    }
    found: list[Path] = []
    for entry in sorted(sums.iterdir()) if sums.is_dir() else []:
        name = entry.name.removesuffix(".partial")
        if name not in finished or not entry.is_file() or entry.is_symlink():
            raise JobError(f"{entry}: not this job's")
        if entry.name != name or not finished[name]:
            found.append(entry)
    # A segment command's segments have no checksums: none written, none expected.
    summed = manifest.settings.get("segment_cmd") is None
    for name, done in finished.items():
        if summed and done and not (sums / name).is_file():
            raise JobError(
                f"{sums / name}: a finished segment's checksums, the manifest says, but missing"
            )
    return found


def _unit_leftovers(manifest: Manifest) -> list[Path]:
    root = manifest.path.parent / STATE
    shots = {f"shot_{shot.start:06d}": index for index, shot in enumerate(manifest.shots)}
    finished = {
        shot
        for segment, done in enumerate(manifest.finished)
        if done
        for shot in manifest.segment_shots(segment)
    }
    found: list[Path] = []
    entries = sorted(root.iterdir()) if root.is_dir() else []
    for entry in entries:
        index = shots.get(entry.name)
        if index is None or not entry.is_dir():
            raise JobError(f"{entry}: not this job's")
        if index in finished:
            found.append(entry)
            continue
        kept = _kept(manifest, index)
        for file in sorted(entry.iterdir()):
            if file.name in kept:
                continue
            ours = UNIT_FILE.fullmatch(file.name.removesuffix(".partial"))
            if ours is None or not file.is_file():
                raise JobError(f"{file}: not this job's")
            found.append(file)
    for index in range(len(manifest.shots)):
        if index in finished:
            continue
        directory = root / f"shot_{manifest.shots[index].start:06d}"
        for name in sorted(_kept(manifest, index) - {COPY, CHECKSUMS}):
            if not (directory / name).is_file():
                raise JobError(f"{directory / name}: kept, the manifest says, but missing")
    if all(manifest.finished) and root.is_dir():
        found.append(root)  # left by a stop before the last segment's units went
    return found


def _kept(manifest: Manifest, index: int) -> set[str]:
    """The unit files of shot `index` the manifest names, of each of its runs: a run's latent
    until its windows are all done, its windows done (units.run_file); and with `split` the
    shot's input copy and the copy's checksums once its first run is encoded, which makes
    them."""
    kept: set[str] = set()
    for run in manifest.shot_runs(index):
        offset = manifest.runs[run].offset
        done = manifest.windows_done[run]
        kept.update(run_file(offset, f"window_{window:04d}.pt") for window in range(done))
        if manifest.encoded[run] and done < len(manifest.layouts[run]):
            kept.add(run_file(offset, "latent.pt"))
        if manifest.encoded[run] and manifest.split and offset == 0:
            kept.update((COPY, CHECKSUMS))
    return kept


def _job(content: dict[str, Any]) -> dict[str, Any]:
    """content without the fields a resume doesn't compare (UNCOMPARED), nor the environment
    changes accepted, which a resume carries on, nor the record of its assembly, how far it went
    (manifest.ASSEMBLED)."""
    job = dict(content)
    job.pop(CHANGES, None)
    job.pop(ASSEMBLED, None)
    for key, fields in UNCOMPARED.items():
        value: Any = content.get(key)
        if isinstance(value, dict):
            job[key] = _without(cast(dict[str, Any], value), fields)
        elif isinstance(value, list):
            job[key] = [_without(entry, fields) for entry in cast(list[dict[str, Any]], value)]
    # A shot's later runs, how far each went likewise (manifest.SHOT_RUNS).
    shots: Any = job.get("shots")
    if isinstance(shots, list):
        job["shots"] = [_runs_without(entry) for entry in cast(list[Any], shots)]
    return job


def _runs_without(shot: Any) -> Any:
    """A shot's record, its later runs' without how far they went (UNCOMPARED's shots fields)."""
    if not isinstance(shot, dict):
        return shot
    entry = dict(cast(dict[str, Any], shot))
    runs: Any = entry.get(SHOT_RUNS)
    if isinstance(runs, list):
        entry[SHOT_RUNS] = [
            _without(cast(dict[str, Any], run), UNCOMPARED["shots"])
            if isinstance(run, dict)
            else run
            for run in cast(list[Any], runs)
        ]
    return entry


def _without(entry: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {name: value for name, value in entry.items() if name not in fields}


def _compare(recorded: Any, asked: Any, where: str, found: list[str]) -> None:
    if isinstance(recorded, dict) and isinstance(asked, dict):
        one, other = cast(dict[str, Any], recorded), cast(dict[str, Any], asked)
        for key in sorted({*one, *other}):
            inner = f"{where}.{key}" if where else key
            if key not in one or key not in other:
                found.append(f"{inner}: {_brief(one.get(key))} -> {_brief(other.get(key))}")
            else:
                _compare(one[key], other[key], inner, found)
    elif isinstance(recorded, list) and isinstance(asked, list):
        one, other = cast(list[Any], recorded), cast(list[Any], asked)
        if len(one) != len(other):
            found.append(f"{where}: {len(one)} recorded -> {len(other)} now")
            return
        for index, (first, second) in enumerate(zip(one, other, strict=True)):
            _compare(first, second, f"{where}[{index}]", found)
    elif recorded != asked:
        found.append(f"{where}: {_brief(recorded)} -> {_brief(asked)}")


def _brief(value: Any) -> str:
    text = json.dumps(value)
    return text if len(text) <= 70 else f"{text[:67]}..."

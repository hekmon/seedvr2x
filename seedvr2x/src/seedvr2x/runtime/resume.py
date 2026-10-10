"""A job resumed from its output directory (DESIGN.md, Pause and resume), the manifest there being
the truth: the job asked must be the one recorded, its settings, models, environment, input,
output and layout alike, but for an environment change the user accepts (--accept-env-change),
which the manifest records; the directory must hold what the manifest says, and nothing else of
anyone's; what a stop left that the manifest doesn't name is discarded."""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from seedvr2x.media.files import partial_path
from seedvr2x.runtime.job import JobError
from seedvr2x.runtime.manifest import INDEX, NAME, STATE, SUMS, Manifest, checksums_file
from seedvr2x.runtime.units import CHECKSUMS, COPY, size

# The fields a resume reads and doesn't compare: how far the job went, in its shots and segments;
# what is recorded for information only (DESIGN.md, Pause and resume): where an input is and when
# it was modified, an input being its content, so that a moved input is the same; the NVIDIA
# driver, since the math kernels ship with torch. And the frame index's file, PROVISIONAL
# (DESIGN.md says only that the manifest names it): derived data, as an input copy is, checked
# against its record when the first pass's record is trusted, made again with the pass otherwise
# (cli._prior), whose new decode may hash some frames otherwise after an ffmpeg change accepted,
# which is no other job. idet's counts likewise, information (manifest._input).
UNCOMPARED = {
    "shots": ("encoded", "windows_done"),
    "segments": ("finished", "bytes"),
    "input": ("path", "modified_ns", "index", "idet"),
    "environment": ("driver",),
}

# The manifest's record of the environment changes a resume accepted (Manifest.content).
CHANGES = "environment_changes"

# The names DiskUnits gives a shot's files.
UNIT_FILE = re.compile(
    rf"latent\.pt|window_\d{{4}}\.pt|{'|'.join(map(re.escape, (COPY, CHECKSUMS)))}"
)


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
    manifest.encoded = [shot["encoded"] for shot in recorded["shots"]]
    manifest.windows_done = [shot["windows_done"] for shot in recorded["shots"]]
    manifest.finished = [segment["finished"] for segment in recorded["segments"]]
    manifest.sizes = [segment["bytes"] for segment in recorded["segments"]]
    manifest.environment_changes = list(recorded.get(CHANGES, []))


def section(difference: str) -> str:
    """The manifest's field a difference is in: environment, for `environment.gpu: ...`."""
    return re.split(r"[.\[:]", difference, maxsplit=1)[0]


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
        "shots_encoded": sum(bool(shot["encoded"]) for shot in recorded["shots"]),
        "windows_done": sum(shot["windows_done"] for shot in recorded["shots"]),
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
    partial files and directories, an unfinished segment's file and checksums, a shot's unit files
    not recorded, the units of finished segments. Refused (JobError): a finished segment missing,
    of another size or without its checksums, a unit recorded but missing, and anything in the
    directory that isn't this job's, which is never deleted. A shot's input copy or its checksums
    missing isn't refused: the copy is derived data, which the run makes again (run_job); nor is
    the frame index (INDEX), made again with the first pass (cli._prior)."""
    directory = manifest.path.parent
    found: list[Path] = []
    indexed = partial_path(directory / INDEX)
    names = {NAME, partial_path(manifest.path).name, STATE, SUMS, INDEX, indexed.name}
    if indexed.exists():
        found.append(indexed)
    for index, name in enumerate(manifest.files):
        path = directory / name
        names.update((name, partial_path(path).name))
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
    for name, done in finished.items():
        if done and not (sums / name).is_file():
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
    """The unit files of shot `index` the manifest names: its latent until its windows are all
    done, its windows done, and with `split` its input copy and the copy's checksums once
    encoded."""
    done = manifest.windows_done[index]
    kept = {f"window_{window:04d}.pt" for window in range(done)}
    if manifest.encoded[index] and done < len(manifest.layouts[index]):
        kept.add("latent.pt")
    if manifest.encoded[index] and manifest.split:
        kept.update((COPY, CHECKSUMS))
    return kept


def _job(content: dict[str, Any]) -> dict[str, Any]:
    """content without the fields a resume doesn't compare (UNCOMPARED), nor the environment
    changes accepted, which a resume carries on."""
    job = dict(content)
    job.pop(CHANGES, None)
    for key, fields in UNCOMPARED.items():
        value: Any = content.get(key)
        if isinstance(value, dict):
            job[key] = _without(cast(dict[str, Any], value), fields)
        elif isinstance(value, list):
            job[key] = [_without(entry, fields) for entry in cast(list[dict[str, Any]], value)]
    return job


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

"""A job resumed from its output directory (DESIGN.md, Pause and resume), the manifest there being
the truth: the job asked must be the one recorded, its settings, models, environment, input,
output and layout alike, but for an environment change the user accepts (--accept-env-change),
which the manifest records; the directory must hold what the manifest says, and nothing else of
anyone's; what a stop left that the manifest doesn't name is discarded."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from seedvr2x.media.files import partial_path
from seedvr2x.runtime.job import JobError
from seedvr2x.runtime.manifest import NAME, STATE, VERSION, Manifest
from seedvr2x.runtime.units import size

# The fields a resume reads and doesn't compare: how far the job went, in its shots and segments;
# what is recorded for information only (DESIGN.md, Pause and resume): where an input is and when
# it was modified, an input being its content, so that a moved input is the same; the NVIDIA
# driver, since the math kernels ship with torch.
UNCOMPARED = {
    "shots": ("encoded", "windows_done"),
    "segments": ("finished", "bytes"),
    "input": ("path", "modified_ns"),
    "environment": ("driver",),
}

# The manifest's record of the environment changes a resume accepted (Manifest.content).
CHANGES = "environment_changes"

# What the first pass depends on in the environment: it runs on ffmpeg's decode alone, so that
# its record stands through any other change accepted (DESIGN.md, Pause and resume).
FIRST_PASS = ("ffmpeg", "conversions")

# The names DiskUnits gives a shot's files.
UNIT_FILE = re.compile(r"latent\.pt|window_\d{4}\.pt")


def read(path: Path) -> dict[str, Any]:
    """The manifest at path, as written; refused unless written by this manifest version."""
    try:
        content: Any = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise JobError(f"{path}: not readable as a manifest: {error}") from None
    if not isinstance(content, dict):
        raise JobError(f"{path}: not a manifest")
    manifest = cast(dict[str, Any], content)
    found = manifest.get("seedvr2x_manifest")
    if found != VERSION:
        raise JobError(
            f"{path}: manifest version {found}, where this seedvr2x writes {VERSION}: not resumable"
        )
    return manifest


def identity(content: dict[str, Any]) -> dict[str, Any]:
    """Of a job's manifest content, what a resume compares before the first pass: the settings,
    the environment, and the inputs' content, their size and SHA-256. The first pass's record is
    trusted when they are the same, the pass then not run again (DESIGN.md, Pause and resume)."""
    inputs = cast(list[dict[str, Any]], content.get("input", []))
    return {
        "settings": content.get("settings"),
        "environment": content.get("environment"),
        "input": [{"bytes": entry.get("bytes"), "sha256": entry.get("sha256")} for entry in inputs],
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
    partial files and directories, an unfinished segment's file, a shot's unit files not recorded,
    the units of finished segments. Refused (JobError): a finished segment missing or of another
    size, a unit recorded but missing, and anything in the directory that isn't this job's, which
    is never deleted."""
    directory = manifest.path.parent
    found: list[Path] = []
    names = {NAME, partial_path(manifest.path).name, STATE}
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
    return found + _unit_leftovers(manifest)


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
        for name in sorted(_kept(manifest, index)):
            if not (directory / name).is_file():
                raise JobError(f"{directory / name}: kept, the manifest says, but missing")
    if all(manifest.finished) and root.is_dir():
        found.append(root)  # left by a stop before the last segment's units went
    return found


def _kept(manifest: Manifest, index: int) -> set[str]:
    """The unit files of shot `index` the manifest names: its latent until its windows are all
    done, and its windows done."""
    done = manifest.windows_done[index]
    kept = {f"window_{window:04d}.pt" for window in range(done)}
    if manifest.encoded[index] and done < len(manifest.layouts[index]):
        kept.add("latent.pt")
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

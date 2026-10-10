"""Measurement's real seek sources read through the frame index (DESIGN.md, Input): seedvr2x's
first pass and reader on research/docs/seeking.md's sources (S1-S11 and the synthetic files,
on the GPU box), held to what measurement's research/scripts/seek_test.py recorded of them. CPU
only, opt-in: skipped without SEEDVR2X_SEEK_SOURCES.

SEEDVR2X_SEEK_SOURCES names a file, outside the repository, listing the cases, one per line, its
fields separated by tabs (a path may hold spaces):

    label <TAB> source file <TAB> seek_test.py's output directory for it [<TAB> targets directory]

`#` starts a comment line; blank lines are skipped. The output directory holds seek_test.py's
ref.framemd5, the source's full decode (framemd5 version 2: `#tb 0: num/den`, then a line per
frame in display order, `0, dts, pts, duration, size, md5`, the pts in the stream's time base, the
MD5 of the frame as decoded, in its own pixel format) and targets.json (`{"targets": [{"n": frame,
"kind": ...}, ...]}`, the frames measurement read, 8 from each). The optional fourth field is a
directory whose targets.json is read in its place, ref.framemd5 still the third field's: S1's
dense-IDR passage, S11's leading pictures, a source of several lines indexed once. The label names
the case, so that pytest -k chooses some: -k "S3 or S11".

Each case: the first pass's ffmpeg over the source (media/scan.py; not the pipeline's refusals,
which take S6, interlaced, and S9, its rate contradicted); its index's count, time base and pts
those of ref.framemd5, frame for frame; then each target read with seedvr2x's reader, 8 frames,
fewer at the end: every frame's CRC-32 the index's (the reader's own check) and its MD5, as decoded
(Reader's md5), ref.framemd5's: an independent check. Frames are compared by place, never found by
their MD5: every source holds one picture at several places, up to 4,346 in a 2-hour film. Frames
in the GOP of a frame decoded with an error (S9's last ones), which a decode may conceal otherwise
than the reference's, are counted and printed, not held to the reference; a frame the reader took
as decoded from the start, its CRC-32 not the index's (media/reader.py, Reader.otherwise), outside
such a GOP fails the case: a clean source decodes as the first pass decoded it. A read that took
more than one attempt fails it too where ffmpeg itself, seeking as the read's first attempt did
(the same -ss and select: seek_test.py's c_gop), gives the reference's frames, no frame flagged
from that keyframe on: only the reader's own check failed that seek, as it failed every seek of a
packed source (2026-10-09). Each read's attempts and time are printed, and the attempts by kind of
target (pytest -rP).

On the box: CUDA_VISIBLE_DEVICES= SEEDVR2X_SEEK_SOURCES=<list> uv run pytest
tests/test_seek_sources.py -rP [-k S3]"""

import json
import os
import statistics
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from seedvr2x.media.conversion import conversion_for
from seedvr2x.media.ffmpeg import input_args
from seedvr2x.media.index import FrameIndex
from seedvr2x.media.probe import probe
from seedvr2x.media.reader import Reader, seek_time
from seedvr2x.media.scan import Scan, scan
from seedvr2x.media.source import resolved

LISTED = os.environ.get("SEEDVR2X_SEEK_SOURCES")
WHY = "needs SEEDVR2X_SEEK_SOURCES, a list of measurement's seek sources (this module's docstring)"
FRAMES = 8  # read from each target, as measurement did (seek_test.py --frames)


@dataclass(frozen=True)
class Case:
    label: str
    source: Path
    measured: Path  # seek_test.py's output directory: ref.framemd5
    targets: Path  # the directory of targets.json


def cases(listed: str) -> list[Case]:
    found: list[Case] = []
    for number, line in enumerate(Path(listed).read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) not in (3, 4):
            raise ValueError(f"{listed}:{number}: not label, source, directory[, targets]")
        label, source, measured = (field.strip() for field in fields[:3])
        targets = fields[3].strip() if len(fields) == 4 else measured
        found.append(Case(label, Path(source), Path(measured), Path(targets)))
    return found


CASES = cases(LISTED) if LISTED else []


@dataclass(frozen=True)
class Reference:
    time_base: Fraction
    pts: list[int]
    md5: list[str]


def reference(path: Path) -> Reference:
    """seek_test.py's ref.framemd5 (its parse_framemd5)."""
    time_base, pts, md5 = Fraction(0), list[int](), list[str]()
    with path.open() as lines:
        for line in lines:
            if line.startswith("#tb 0:"):
                time_base = Fraction(line.split(":", 1)[1].strip())
            elif line[:1] != "#" and line.strip():
                fields = [field.strip() for field in line.split(",")]
                if len(fields) >= 6 and fields[0] == "0":
                    pts.append(int(fields[2]))
                    md5.append(fields[5])
    return Reference(time_base, pts, md5)


def damaged_gops(index: FrameIndex) -> set[int]:
    """The frames in the GOP of a frame the first pass flagged: from the first frame of its own
    keyframe to the next keyframe's, leading pictures included, which may reference it."""
    found: set[int] = set()
    for frame in np.flatnonzero(index.error).tolist():
        own = index.own_keyframe(frame)
        first = index.first_frame_from(own) if own >= 0 else 0
        after = own + 1
        end = index.first_frame_from(after) if after < len(index.keyframes) else index.frames
        found.update(range(min(first, frame), max(end, frame + 1)))
    return found


# The first pass of each source, its time, once for every case naming it.
_SCANNED: dict[Path, tuple[Scan, float]] = {}


def scanned(source: Path) -> tuple[Scan, float]:
    key = source.resolve()
    if key not in _SCANNED:
        started = time.monotonic()
        found = scan(source)
        _SCANNED[key] = (found, time.monotonic() - started)
    return _SCANNED[key]


def ffmpeg_seek(path: Path, seek: str, select: int, count: int) -> list[str]:
    """The MD5s of the `count` frames ffmpeg itself gives, seeking as a read's attempt does
    (media/reader.py: -noaccurate_seek -ss `seek`, kept from pts `select` on): seek_test.py's
    c_gop, with the reader's seek."""
    output = subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-nostdin", "-copyts", "-noaccurate_seek", "-ss", seek),
            *input_args(path),
            *("-map", "0:v:0", "-vf", f"select=gte(pts\\,{select})", "-fps_mode", "passthrough"),
            *("-enc_time_base:v", "demux", "-frames:v", str(count), "-f", "framemd5", "-"),
        ],
        capture_output=True,
        check=True,
    ).stdout.decode()
    return [line.split(",")[5].strip() for line in output.splitlines() if line[:1] == "0"]


@pytest.mark.parametrize(
    "case",
    CASES or [pytest.param(None, id="none", marks=pytest.mark.skip(reason=WHY))],
    ids=[case.label for case in CASES] or None,
)
def test_seek_source(case: Case) -> None:
    found, seconds = scanned(case.source)
    index = found.index
    assert index is not None
    full = reference(case.measured / "ref.framemd5")
    print(
        f"{case.label}: {case.source.name}: first pass {found.frames} frames in {seconds:.1f} s,"
        f" {found.frames / seconds:.0f} fps; {len(index.keyframes)} keyframes, start time"
        f" {index.start_time} us, {int(index.error.sum())} frames decoded with an error"
    )
    assert found.frames == len(full.pts)
    assert index.time_base == full.time_base
    pairs = zip(index.pts.tolist(), full.pts, strict=True)
    differing = [n for n, (a, b) in enumerate(pairs) if a != b]
    assert not differing, f"pts differ from frame {differing[0]} on, {len(differing)} frames"
    stream = resolved(probe(case.source))
    conversion = conversion_for(stream)
    listed: Any = json.loads((case.targets / "targets.json").read_text())
    chosen = [(int(target["n"]), str(target.get("kind", ""))) for target in listed["targets"]]
    damaged = damaged_gops(index)
    wrong: list[int] = []
    otherwise: list[int] = []
    retried: list[int] = []
    taken = 0
    times: list[float] = []
    attempts: list[int] = []
    by_kind: dict[str, Counter[int]] = {}
    for n, kind in chosen:
        count = min(FRAMES, found.frames - n)
        started = time.monotonic()
        reader = Reader(*(case.source, conversion, stream.width, stream.height, index, n), md5=True)
        try:
            reader.read(count)
        finally:
            reader.stop()
        spent = time.monotonic() - started
        times.append(spent)
        attempts.append(len(reader.attempts))
        by_kind.setdefault(kind, Counter())[len(reader.attempts)] += 1
        places = range(n, n + len(reader.md5s))
        loose = [k for k in places if k in damaged]
        exact = len(reader.md5s) == count and all(
            reader.md5s[k - n] == full.md5[k] for k in places if k not in damaged
        )
        taken += sum(reader.md5s[k - n] != full.md5[k] for k in loose)
        otherwise += [k for k in reader.otherwise if k not in damaged]
        first = reader.attempts[0].keyframe
        if (
            first is not None
            and len(reader.attempts) > 1
            and index.first_error(index.first_frame_from(first), n + count - 1) is None
            and ffmpeg_seek(case.source, seek_time(index, first), int(index.pts[n]), count)
            == full.md5[n : n + count]
        ):
            retried.append(n)
        if not exact:
            wrong.append(n)
        steps = ", ".join(
            "start" if a.keyframe is None else f"keyframe {a.keyframe} (-{a.back})"
            for a in reader.attempts
        )
        print(
            f"  {n} {kind}: {'exact' if exact else 'WRONG'} in {spent:.3f} s, from {steps}"
            + (f"; {len(loose)} in a damaged GOP" if loose else "")
        )
    print(
        f"{case.label}: {len(chosen)} targets, {len(chosen) - len(wrong)} exact, attempts"
        f" {dict(sorted((a, attempts.count(a)) for a in set(attempts)))}, read in"
        f" {statistics.median(times):.3f} s median, {max(times):.3f} s worst; {taken} frames of"
        " damaged GOPs otherwise than the reference"
    )
    print(
        f"{case.label}: attempts by kind of target: "
        + "; ".join(f"{kind} {dict(sorted(c.items()))}" for kind, c in sorted(by_kind.items()))
    )
    assert not wrong, f"not exact at {wrong}"
    assert not otherwise, f"decoded otherwise than in the first pass, no GOP damaged: {otherwise}"
    assert not retried, f"read again where ffmpeg's own seek gives the reference: {retried}"

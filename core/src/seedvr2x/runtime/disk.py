"""The disk a job takes beside its output, said by --plan and at every run's start, and warned of
where it won't fit (DESIGN.md, Memory planner: "the disk the job takes beside its output: its
masters, about 100-135 GiB an hour of 1080p in yuv420p10le, twice that while a file is assembled,
against what its disk has free; a run says it too, warning where they won't fit, since the disk
may be cleared as the job runs"; decided on 2026-10-28, built on 2026-10-29).

A reading of the masters (`masters`) and of the room (`reading`), a line (`Disk.line`) and a
warning (`Disk.warning`), never a refusal: the assembly's own stop for want of room, and the
frames a --quality's runs keep (cli._room_on_disk), are separate and stay as they are."""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

# A master's bytes a frame before any is written, at most: its samples uncompressed, FFV1 being
# lossless, at the bytes a sample each format takes (media/writer.py: yuv420p10le's 1.5 samples a
# pixel at 2 bytes, gbrp16le's 3 at 2, PNG's rgb48be 3 at 2, a PNG compressing 16-bit noise
# little). For a one-file output's assembly's estimate (cli._masters) and a PNG directory's disk.
RAW_BYTES = {"yuv420p10le": 3.0, "gbrp16le": 6.0, "png": 6.0}

_GIB = 2**30

# DESIGN.md, Output, Master sizes (milestone 5, milestone 7): per hour of 1080p (1920x1080) at
# 24000/1001 "about 100-135 GiB in yuv420p10le, the default, and 540-690 GiB in gbrp16le with
# colour correction"; the upper end of each, the disk being short rather than wasted, scaled by
# a master's pixels and frames, not by the rate: bytes a pixel of a frame. PNG has no figure of
# its own: RAW_BYTES.
_HOUR = 3600 * Fraction(24000, 1001) * 1920 * 1080
FIGURE = {
    "yuv420p10le": Fraction(135 * _GIB) / _HOUR,
    "gbrp16le": Fraction(690 * _GIB) / _HOUR,
}


@dataclass(frozen=True)
class Masters:
    """The bytes of a job's masters (the output segments, FFV1 files, a segment command's files
    or PNG directories): `written` of them are on the disk, `left` still to write; None for both
    of the others where unknown. `bound`: `left` is an upper bound, from the figure or the raw
    size, not from segments written."""

    written: int
    left: int | None
    bound: bool

    @property
    def total(self) -> int | None:
        return None if self.left is None else self.written + self.left


def masters(
    frames: Sequence[int],
    sizes: Sequence[int | None],
    output_format: str,
    size: tuple[int, int],
    commanded: bool,
) -> Masters:
    """The masters of the segments of `frames` frames each, `sizes` the bytes of those finished
    (None for the others), the output's `size` (height, width). What is left to write is the
    finished ones' bytes a frame (the rate measured on this job) times the frames left; with none
    written, DESIGN's figure of the format at the output's pixels (FIGURE), a PNG's raw size at
    most (RAW_BYTES), and for a segment command's files unknown: its encoder decides, the
    content less than the format does (DESIGN.md, Output, Writer)."""
    height, width = size
    done = [(count, taken) for count, taken in zip(frames, sizes, strict=True) if taken is not None]
    written = sum(taken for _, taken in done)
    counted = sum(count for count, _ in done)
    remaining = sum(frames) - counted
    if remaining == 0:
        return Masters(written, 0, False)
    if counted:
        return Masters(written, round(written / counted * remaining), False)
    if commanded:
        return Masters(0, None, True)
    if output_format in FIGURE:
        return Masters(0, round(FIGURE[output_format] * height * width * remaining), True)
    return Masters(0, round(RAW_BYTES.get(output_format, 6.0) * height * width * remaining), True)


def _size(count: int) -> str:
    """Bytes in binary units, to three figures at most: 186 GiB, 1.2 GiB, 512 MiB."""
    if count >= 10 * _GIB:
        return f"{count / _GIB:.0f} GiB"
    if count >= _GIB:
        return f"{count / _GIB:.1f} GiB"
    return f"{count / 2**20:.0f} MiB"


def free_on(directory: Path) -> int:
    """The bytes free on the disk of `directory`, or of its nearest parent that exists."""
    place = directory
    while not place.exists():
        place = place.parent
    # As the assembly's own check reads it (media/assembly.py, os.statvfs): the blocks free to a
    # user, shutil.disk_usage's `free`.
    stats = os.statvfs(place)
    return stats.f_bavail * stats.f_frsize


@dataclass(frozen=True)
class Disk:
    """What the job takes on its disk and what is free there (`reading`)."""

    masters: Masters
    free: int
    assembled: bool  # a one-file output: the file is made beside the masters

    @property
    def need(self) -> int | None:
        """The bytes the job still writes: the masters left, and for a one-file output the file,
        as many as the masters in all (DESIGN.md, Output, Assembly: "at the end the segments and
        the finished file sit side by side, about twice the master"). None where unknown. The
        source's other streams are left out: their bytes are not known without reading the
        source's container (PROVISIONAL, impl-v010, 2026-10-29: DESIGN.md says them counted
        "twice while they are compared" and gives no figure), so the line says it in words."""
        total = self.masters.total
        if total is None or self.masters.left is None:
            return None
        return self.masters.left + (total if self.assembled else 0)

    @property
    def short(self) -> int:
        """How many bytes the need exceeds the free room by, 0 where it fits or isn't known."""
        need = self.need
        return 0 if need is None else max(0, need - self.free)

    def line(self) -> str:
        """The plan's line for the list (planner.said gives it its "plan: "): the masters, the
        room they take with the file beside them where one is made, what is free on the disk, and
        by how much it falls short. PROVISIONAL (impl-v010, 2026-10-29: DESIGN.md names the
        content, not the words)."""
        masters = self.masters
        free = f"{_size(self.free)} free on its disk"
        total = masters.total
        if total is None:
            return (
                "disk: the masters' size is not known before one is written, the segment"
                f" command's encoder deciding it; {free}"
            )
        verdict = f"{_size(self.short)} short" if self.short else "enough"
        about = "at most" if masters.bound else "about"
        other = ", and the source's other streams" if self.assembled else ""
        if masters.written:
            where = (
                f"the file assembled beside them{other}"
                if self.assembled
                else "in the output directory"
            )
            return (
                f"disk: the masters {about} {_size(total)}, {_size(masters.written)} of them"
                f" written; the job still takes {_size(self.need or 0)} on its disk, {where};"
                f" {free}, {verdict}"
            )
        if self.assembled:
            return (
                f"disk: the masters {about} {_size(total)}, twice that, {_size(2 * total)}, while"
                f" the file is assembled{other}; {free}, {verdict}"
            )
        return (
            f"disk: the masters {about} {_size(total)} in the output directory; {free}, {verdict}"
        )

    def warning(self, output: Path) -> str | None:
        """The warning of a job whose disk is short, after the plan's lines: what is free, what
        the job still writes, and what comes of it; None where it fits or is unknown. Never a
        refusal: the disk may be cleared as the job runs (DESIGN.md, Memory planner, decided on
        2026-10-28). PROVISIONAL (impl-v010, 2026-10-29: the words; for a directory, where the
        run stops when the disk fills, is the writers' own failure, every finished segment kept
        and the same command resuming)."""
        need = self.need
        if need is None or not self.short:
            return None
        if self.assembled:
            then = "free some before the assembly, or the run stops there, everything kept"
        else:
            then = "free some as the run goes, or it stops where the disk is full, the finished"
            then += " segments kept"
        return (
            f"{output}: its disk has {_size(self.free)} free for about {_size(need)} this job"
            f" still writes: {then}"
        )


def reading(
    directory: Path,
    segments: Sequence[tuple[int, int, int | None]],
    output_format: str,
    size: tuple[int, int],
    *,
    commanded: bool,
    assembled: bool,
) -> Disk:
    """The disk of a job whose work is in `directory`: `segments` each (start, end, bytes) with
    the bytes of the finished ones, None for the others (the manifest's), `size` the output's
    (height, width). `commanded`: the masters are a segment command's files; `assembled`: a
    .mkv path, whose file is made beside them."""
    return Disk(
        masters(
            [end - start for start, end, _ in segments],
            [taken for _, _, taken in segments],
            output_format,
            size,
            commanded,
        ),
        free_on(directory),
        assembled,
    )

"""The frame index (DESIGN.md, Input): what the first pass records of each frame of a source, so
that any frame is read again exactly, through a seek (media/reader.py). For each frame, in the
order the decoder gives them, counted as they come: its timestamp, a CRC-32 of its decoded picture
in the decoder's own pixel format, and whether the decoder reported an error on it; and the
keyframes' timestamps, from a packet scan (media/scan.py). When the shot detector ran in the first
pass, each frame's probability too (DESIGN.md, Shot detection: the first pass's record keeps
TransNetV2's per-frame probabilities).

One file (-o x.mkv) keeps it in memory. An output directory holds it beside the manifest, written
once with the first pass's record, which names it by its size and SHA-256 (runtime/manifest.py);
a resume reads it back as it trusts that record. Its format is exact and versioned: a magic line, a
JSON line saying the version, the time base, the file's start time and each table's columns, then
the columns' values, little-endian, column after column. The probabilities are a column of the
frames' table, PROBABILITIES, named in the header when the detector ran: a reader takes the columns
it knows by name and skips the others by their size, so the version stays."""

import json
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from seedvr2x.media.ffmpeg import MediaError
from seedvr2x.media.files import write_whole

# libavutil's AV_NOPTS_VALUE: a frame without a timestamp, which no select can single out. ffmpeg
# n9.0.2 gives every decoded frame one, extrapolated from the frame before when the decoder has
# none (fftools/ffmpeg_dec.c:409-411), so the first pass doesn't meet it: kept for a build that
# gives none all the same, whose hash line would say this very number (media/scan.py, HASHED).
NOPTS = -(2**63)

# PROVISIONAL (DESIGN.md names neither the index's file nor its format): this one, exact, versioned
# and open to more columns (the module's docstring).
MAGIC = b"seedvr2x frame index\n"
VERSION = 1
# Each table's columns, (name, numpy's dtype), the order they are written in.
FRAME_COLUMNS = (("pts", "<i8"), ("crc32", "<u4"), ("error", "|u1"))
KEYFRAME_COLUMNS = (("pts", "<i8"),)
# The shot detector's probability of each frame, the sigmoid of TransNetV2's single-frame head
# (runtime/detector.py), float32 as it computes them: a column of the frames' table, after the
# others, only when the detector ran (absent with --cuts, whose list replaces the detection).
# PROVISIONAL, as the format is: its name and place (DESIGN.md: "the first pass's record keeps"
# them).
PROBABILITIES = ("transnetv2", "<f4")


@dataclass(frozen=True, eq=False)
class FrameIndex:
    """A source's frame index. The timestamps are the source's own, integers in its stream's time
    base (ffmpeg's -copyts, the encoder's time base the demuxer's), never rounded."""

    time_base: Fraction  # the stream's, in seconds per unit of pts
    # The file's start, in microseconds (ffmpeg's AV_TIME_BASE), as ffprobe reads it: ffmpeg's -ss
    # counts from it (fftools/ffmpeg_demux.c:2244-2246). None when the file declares none.
    start_time: int | None
    pts: npt.NDArray[np.int64]  # each frame's, in the order the decoder gives them
    crc32: npt.NDArray[np.uint32]  # each frame's decoded picture, as stored in its pixel format
    error: npt.NDArray[np.bool_]  # whether the decoder reported an error on the frame
    keyframes: npt.NDArray[np.int64]  # the keyframe packets' pts, sorted, each once
    # Each frame's shot probability, float32 in [0, 1] (PROBABILITIES); None when the shot
    # detector didn't run.
    probabilities: npt.NDArray[np.float32] | None = None

    @property
    def frames(self) -> int:
        return len(self.pts)

    def own_keyframe(self, frame: int) -> int:
        """The keyframe frame `frame` belongs to, its position in keyframes: the last whose pts is
        at most the frame's, -1 when none is (or the frame has no pts). Defined on display order,
        the only order the index holds: a leading picture, displayed before its keyframe and
        decoded after it, belongs to the keyframe before (media/reader.py says why that holds).
        PROVISIONAL (DESIGN.md says "n's own keyframe" and defines it not): seek_test.py's c_gop,
        as measured."""
        pts = int(self.pts[frame])
        if pts == NOPTS:
            return -1
        return int(np.searchsorted(self.keyframes, pts, side="right")) - 1

    def first_frame_from(self, keyframe: int) -> int:
        """The first frame whose pts is at least that of keyframe `keyframe` (its position in
        keyframes): where a read decoding from that keyframe starts giving frames."""
        # The highest pts so far, which a search needs ascending: the frames' own, for a source
        # the first pass accepts, whose frame durations are all positive (scan.timing_error).
        pts = int(self.keyframes[keyframe])
        return int(np.searchsorted(self._highest_before, pts, side="left"))

    def selects(self, frame: int) -> bool:
        """Whether ffmpeg's select on frame `frame`'s pts, gte(pts, pts_n), passes exactly that
        frame and every one after it: the frame has a pts, above every one before it, and none
        after it has a lower one or none. Two frames sharing one pts, or a frame without one,
        can't be told apart by it (DESIGN.md, Input): reads through them decode from the start
        and count (media/reader.py)."""
        pts = int(self.pts[frame])
        if pts == NOPTS or bool(self._undefined_after[frame]):
            return False
        if frame and int(self._highest_before[frame - 1]) >= pts:
            return False
        return int(self._lowest_after[frame]) == pts

    def lowest_from(self, frame: int) -> int | None:
        """The lowest pts of frame `frame` and those after it, None when one of them has none: a
        select passing them all, for a read that decodes from the start and counts."""
        if bool(self._undefined_after[frame]):
            return None
        return int(self._lowest_after[frame])

    def first_error(self, first: int, last: int) -> int | None:
        """The first frame from `first` to `last` (inclusive) the decoder reported an error on in
        the first pass, or None: a read that decoded it goes to the start at once
        (media/reader.py)."""
        found = np.flatnonzero(self.error[max(first, 0) : last + 1])
        return None if not len(found) else max(first, 0) + int(found[0])

    @cached_property
    def _highest_before(self) -> npt.NDArray[np.int64]:
        # NOPTS, the lowest int64, never wins a maximum.
        return np.maximum.accumulate(self.pts)

    @cached_property
    def _lowest_after(self) -> npt.NDArray[np.int64]:
        defined = np.where(self.pts == NOPTS, np.iinfo(np.int64).max, self.pts)
        return np.minimum.accumulate(defined[::-1])[::-1]

    @cached_property
    def _undefined_after(self) -> npt.NDArray[np.bool_]:
        return np.logical_or.accumulate((self.pts == NOPTS)[::-1])[::-1]

    def to_bytes(self) -> bytes:
        """The index as its file holds it (the module's docstring): the same bytes for the same
        index, so that one made again from the same decode has the SHA-256 recorded (a damaged
        source's need not, its decode not the same each time: media/reader.py). ValueError when
        the probabilities aren't one per frame."""
        frame_columns = FRAME_COLUMNS
        columns = [self.pts.astype("<i8"), self.crc32.astype("<u4"), self.error.astype("|u1")]
        if self.probabilities is not None:
            if len(self.probabilities) != self.frames:
                raise ValueError(
                    f"{len(self.probabilities)} shot probabilities for {self.frames} frames"
                )
            frame_columns = (*FRAME_COLUMNS, PROBABILITIES)
            columns.append(self.probabilities.astype(PROBABILITIES[1]))
        header = {
            "version": VERSION,
            "time_base": f"{self.time_base.numerator}/{self.time_base.denominator}",
            "start_time": self.start_time,
            "frames": {"rows": self.frames, "columns": [list(c) for c in frame_columns]},
            "keyframes": {
                "rows": len(self.keyframes),
                "columns": [list(c) for c in KEYFRAME_COLUMNS],
            },
        }
        line = json.dumps(header, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        columns.append(self.keyframes.astype("<i8"))
        return MAGIC + line + b"".join(column.tobytes() for column in columns)

    @classmethod
    def from_bytes(cls, data: bytes) -> "FrameIndex":
        """The index a file holds (to_bytes), its probabilities with it when it has them.
        Raises ValueError, saying why, for anything else: another format or version, a column
        missing or of another type, a probability outside [0, 1], the values cut short or
        followed by more."""
        if not data.startswith(MAGIC):
            raise ValueError("not a frame index of seedvr2x")
        end = data.find(b"\n", len(MAGIC))
        if end < 0:
            raise ValueError("its header unfinished")
        try:
            header: Any = json.loads(data[len(MAGIC) : end])
        except ValueError:
            raise ValueError("its header not JSON") from None
        if not isinstance(header, dict):
            raise ValueError("its header not a JSON object")
        fields = cast(dict[str, Any], header)
        version: Any = fields.get("version")
        if version != VERSION:
            raise ValueError(f"version {json.dumps(version)}, where {VERSION} is read")
        time_base = _fraction(fields.get("time_base"))
        start_time: Any = fields.get("start_time")
        if start_time is not None and type(start_time) is not int:
            raise ValueError(f"start time {json.dumps(start_time)}, not microseconds")
        offset = end + 1
        tables: dict[str, dict[str, npt.NDArray[Any]]] = {}
        for name, known, optional in (
            ("frames", FRAME_COLUMNS, (PROBABILITIES,)),
            ("keyframes", KEYFRAME_COLUMNS, ()),
        ):
            tables[name], offset = _table(
                data, offset, name, fields.get(name), dict(known), dict(optional)
            )
        if offset != len(data):
            raise ValueError(f"{len(data) - offset} bytes after its last column")
        frames, keyframes = tables["frames"], tables["keyframes"]
        if frames["error"].size and int(frames["error"].max()) > 1:
            raise ValueError("an error flag neither 0 nor 1")
        if np.any(np.diff(keyframes["pts"]) <= 0):
            raise ValueError("its keyframes not in order, each once")
        found = frames.get(PROBABILITIES[0])
        probabilities = None
        if found is not None:
            # A NaN fails both bounds: the detector gives a sigmoid's values.
            if not bool(np.all((found >= 0) & (found <= 1))):
                raise ValueError("a shot probability outside [0, 1]")
            probabilities = found.astype(np.float32)
        return cls(
            time_base,
            start_time,
            frames["pts"].astype(np.int64),
            frames["crc32"].astype(np.uint32),
            frames["error"].astype(np.bool_),
            keyframes["pts"].astype(np.int64),
            probabilities,
        )

    def write(self, path: Path) -> None:
        """Write the index to path, whole even after a power cut (files.write_whole)."""
        write_whole(path, self.to_bytes())

    @classmethod
    def read(cls, path: Path) -> "FrameIndex":
        """The index at path. Raises MediaError when it is missing or not an index (from_bytes)."""
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            raise MediaError(f"{path}: missing") from None
        except OSError as error:
            raise MediaError(f"{path}: not readable: {error}") from None
        try:
            return cls.from_bytes(data)
        except ValueError as error:
            raise MediaError(f"{path}: not readable as a frame index: {error}") from None


def _fraction(value: object) -> Fraction:
    """A positive time base written num/den."""
    try:
        found = Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        raise ValueError(f"time base {json.dumps(value)}, not num/den") from None
    if found <= 0:
        raise ValueError(f"time base {found}, not positive")
    return found


def _table(
    data: bytes,
    offset: int,
    name: str,
    declared: object,
    known: dict[str, str],
    optional: dict[str, str] | None = None,
) -> tuple[dict[str, npt.NDArray[Any]], int]:
    """The columns of table `name` its header declares, those of `known` with their dtypes, and
    those of `optional` it declares, from data at offset; and the offset after them, every column
    declared skipped by its size."""
    read = {**known, **(optional or {})}
    if not isinstance(declared, dict):
        raise ValueError(f"no {name} table")
    table = cast(dict[str, Any], declared)
    rows: Any = table.get("rows")
    columns: Any = table.get("columns")
    if type(rows) is not int or rows < 0 or not isinstance(columns, list):
        raise ValueError(f"its {name} table's rows or columns unreadable")
    found: dict[str, npt.NDArray[Any]] = {}
    for column in cast(list[Any], columns):
        if not isinstance(column, list) or len(cast(list[Any], column)) != 2:
            raise ValueError(f"a {name} column unreadable: {json.dumps(column)}")
        label, dtype = (str(part) for part in cast(list[Any], column))
        try:
            kind = np.dtype(dtype)
        except TypeError:
            raise ValueError(f"{name} column {label}: unknown type {dtype}") from None
        size = rows * kind.itemsize
        if offset + size > len(data):
            raise ValueError(f"cut short in its {name} column {label}")
        if label in read:
            if dtype != read[label]:
                raise ValueError(f"{name} column {label} of type {dtype}, not {read[label]}")
            found[label] = np.frombuffer(data, kind, rows, offset)
        offset += size
    missing = sorted(set(known) - set(found))
    if missing:
        raise ValueError(f"no {name} column {', '.join(missing)}")
    return found, offset

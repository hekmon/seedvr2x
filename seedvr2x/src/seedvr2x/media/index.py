"""The frame index (DESIGN.md, Input): what the first pass records of each frame of a source, so
that any frame is read again exactly, through a seek (media/reader.py). For each frame, in the
order the decoder gives them, counted as they come: its timestamp, a CRC-32 of its decoded picture
in the decoder's own pixel format, and whether the decoder reported an error on it; and the
keyframes' timestamps, from a packet scan (media/scan.py). When the shot detector ran in the first
pass, each frame's probability too (DESIGN.md, Shot detection: the first pass's record keeps
TransNetV2's per-frame probabilities).

A source whose frames lie on its rate's grid with places left empty has them filled (DESIGN.md,
Input: "Missing frames are filled, not refused"; media/grid.py): its index is then one in PLACES,
a row a place, which everything after the first pass counts its frames by. A row is a decoded
frame's, as the first pass recorded it, or a HELD one, a place left empty, which shows the frame
before it again (FrameIndex.held). The decoded frames' own index stays within reach (decoded),
with the maps both ways (shown, placed): the reads check the decoded frames, each once
(media/reader.py). A source without an empty place has no held row, and its index is the first
pass's, its file the same bytes as before the places.

A job's directory, the output's or a .mkv path's work directory, holds it beside the manifest,
written once with the first pass's record, which names it by its size and SHA-256
(runtime/manifest.py); a resume reads it back as it trusts that record, and the assembly of a .mkv
path takes the source's first timestamp and time base from it (media/assembly.py). Its format is
exact and versioned: a magic line, a JSON line saying the version, the time base, the file's start
time and each table's columns, then the columns' values, little-endian, column after column. The
probabilities are a column of the frames' table, PROBABILITIES, named in the header when the
detector ran: a reader takes the columns it knows by name and skips the others by their size, so
the version stays. The held rows are a column too, HELD, and a file that has it says another
version, FILLED: a reader skipping that column would take every held row for a decoded frame."""

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

# The index's file (DESIGN.md, Input: "written once with the first pass's record, beside the
# manifest, which names it by its SHA-256"), in a format DESIGN.md leaves to the code, accepted by
# design as built (2026-10-10): this one, exact, versioned and open to more columns (the module's
# docstring).
MAGIC = b"seedvr2x frame index\n"
VERSION = 1
# Each table's columns, (name, numpy's dtype), the order they are written in.
FRAME_COLUMNS = (("pts", "<i8"), ("crc32", "<u4"), ("error", "|u1"))
KEYFRAME_COLUMNS = (("pts", "<i8"),)
# The shot detector's probability of each frame, the sigmoid of TransNetV2's single-frame head
# (runtime/detector.py), float32 as it computes them: a column of the frames' table, after the
# others, only when the detector ran (absent with --cuts, whose list replaces the detection).
# DESIGN.md, Shot detection: "The first pass's record keeps TransNetV2's per-frame probabilities
# (its single-frame head), a column of the frame index"; the column's name and place are the
# format's, accepted with it.
PROBABILITIES = ("transnetv2", "<f4")
# Whether each row is a held one, a place left empty that shows the frame before it again
# (FrameIndex.held): a column of the frames' table, after the first pass's own and before the
# probabilities, only when a place is filled, the file then at version FILLED. Another version,
# not a column a reader may skip: a reader of VERSION alone, skipping it, would take each held
# row for a decoded frame, its count one more than ffmpeg's frames at every filled place, and
# must refuse the file instead. It does, by the version ("version 2, where 1 is read"), a resume
# then running the first pass again (cli._recorded_index). PROVISIONAL (implementation,
# 2026-10-10/11, a question for design: the index's format is seedvr2x's own, DESIGN.md naming
# neither its file nor its versions): two versions written, by what the index holds, rather than
# every file moved to the new one. An index without a held row is then written as before the
# places, byte for byte: read back from a job of before, it is recorded again by the size and
# SHA-256 its file has (runtime/manifest.py), and nothing changes for the sources read so far.
FILLED = 2
HELD = ("held", "|u1")


@dataclass(frozen=True, eq=False)
class FrameIndex:
    """A source's frame index, a row a place (the module's docstring): a decoded frame's, or a
    held one's, which shows the frame before it again. The timestamps are the source's own,
    integers in its stream's time base (ffmpeg's -copyts, the encoder's time base the demuxer's),
    never rounded; a held row's is its place's own time (media/grid.py), for the times shown to
    a user alone (runtime/cuts.py), its CRC-32 and error flag those of the frame it shows.

    What is asked of the frames ffmpeg gives (own_keyframe, first_frame_from, selects,
    lowest_from, first_error) is asked of the decoded frames' index, decoded(): an index with
    held rows refuses it (ValueError), a place being no frame's number there."""

    time_base: Fraction  # the stream's, in seconds per unit of pts
    # The file's start, in microseconds (ffmpeg's AV_TIME_BASE), as ffprobe reads it: ffmpeg's -ss
    # counts from it (fftools/ffmpeg_demux.c:2244-2246). None when the file declares none.
    start_time: int | None
    pts: npt.NDArray[np.int64]  # each frame's, in the order the decoder gives them
    crc32: npt.NDArray[np.uint32]  # each frame's decoded picture, as stored in its pixel format
    error: npt.NDArray[np.bool_]  # whether the decoder reported an error on the frame
    keyframes: npt.NDArray[np.int64]  # the keyframe packets' pts, sorted, each once
    # Each frame's shot probability, float32 in [0, 1] (PROBABILITIES); None when the shot
    # detector didn't run. With held rows, the decoded frames' spread over the places so that the
    # cuts fall where they fell (media/grid.py, spread), each decoded frame's own on the last
    # place that shows it (decoded).
    probabilities: npt.NDArray[np.float32] | None = None
    # Whether each row is a held one (HELD); None when no place is filled, never all False: one
    # form for an index without an empty place, the first pass's (__post_init__).
    held: npt.NDArray[np.bool_] | None = None

    def __post_init__(self) -> None:
        held = self.held
        if held is None:
            return
        if len(held) != len(self.pts):
            raise ValueError(f"{len(held)} held flags for {len(self.pts)} rows")
        if not held.any():
            object.__setattr__(self, "held", None)  # frozen: the one form of "none held"
        elif held[0]:
            raise ValueError("its first row held, with no frame before it to show")

    @property
    def frames(self) -> int:
        """The index's rows: the places, the frames everything after the first pass counts; the
        decoded frames when none is held."""
        return len(self.pts)

    @cached_property
    def shown(self) -> npt.NDArray[np.int64]:
        """The decoded frame each place shows, its number among the decoded frames, decoded()'s
        rows: (places,) int64, a held place's the frame before its hole."""
        if self.held is None:
            return np.arange(self.frames, dtype=np.int64)
        return np.cumsum(~self.held, dtype=np.int64) - 1

    @cached_property
    def placed(self) -> npt.NDArray[np.int64]:
        """Each decoded frame's own place, the first that shows it: (decoded frames,) int64,
        strictly increasing from 0."""
        if self.held is None:
            return np.arange(self.frames, dtype=np.int64)
        return np.flatnonzero(~self.held).astype(np.int64)

    def decoded(self) -> "FrameIndex":
        """The decoded frames' own index, the first pass's exactly: the rows that aren't held,
        their pts, CRC-32s and error flags as recorded, each frame's shot probability the one of
        the last place that shows it (media/grid.py, spread), bit for bit what the detector gave.
        The index itself when no row is held. ValueError when the probabilities aren't one per
        row."""
        return self._decoded

    @cached_property
    def _decoded(self) -> "FrameIndex":
        if self.held is None:
            return self
        own = self.placed
        probabilities = None
        if self.probabilities is not None:
            self._one_per_row(self.probabilities)
            probabilities = self.probabilities[np.append(own[1:], self.frames) - 1]
        return FrameIndex(
            *(self.time_base, self.start_time, self.pts[own], self.crc32[own], self.error[own]),
            self.keyframes,
            probabilities,
        )

    def _one_per_row(self, probabilities: npt.NDArray[np.float32]) -> None:
        if len(probabilities) != self.frames:
            raise ValueError(f"{len(probabilities)} shot probabilities for {self.frames} frames")

    def _of_frames(self) -> None:
        """Refuse what is asked of the frames ffmpeg gives when rows are held: its answer would
        take a place for a decoded frame's number, silently."""
        if self.held is not None:
            raise ValueError(
                "asked of a frame index with held rows, by place: a seedvr2x bug, its decoded"
                " frames' index answers it (FrameIndex.decoded)"
            )

    def own_keyframe(self, frame: int) -> int:
        """The keyframe frame `frame` belongs to, its position in keyframes: the last whose pts is
        at most the frame's, -1 when none is (or the frame has no pts). Defined on display order,
        the only order the index holds: a leading picture, displayed before its keyframe and
        decoded after it, belongs to the keyframe before (media/reader.py says why that holds).
        DESIGN.md, Input: "n's own keyframe is the last whose timestamp is at most n's":
        seek_test.py's c_gop, as measured."""
        self._of_frames()
        pts = int(self.pts[frame])
        if pts == NOPTS:
            return -1
        return int(np.searchsorted(self.keyframes, pts, side="right")) - 1

    def first_frame_from(self, keyframe: int) -> int:
        """The first frame whose pts is at least that of keyframe `keyframe` (its position in
        keyframes): where a read decoding from that keyframe starts giving frames."""
        self._of_frames()
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
        self._of_frames()
        pts = int(self.pts[frame])
        if pts == NOPTS or bool(self._undefined_after[frame]):
            return False
        if frame and int(self._highest_before[frame - 1]) >= pts:
            return False
        return int(self._lowest_after[frame]) == pts

    def lowest_from(self, frame: int) -> int | None:
        """The lowest pts of frame `frame` and those after it, None when one of them has none: a
        select passing them all, for a read that decodes from the start and counts."""
        self._of_frames()
        if bool(self._undefined_after[frame]):
            return None
        return int(self._lowest_after[frame])

    def first_error(self, first: int, last: int) -> int | None:
        """The first frame from `first` to `last` (inclusive) the decoder reported an error on in
        the first pass, or None: a read that decoded it goes to the start at once
        (media/reader.py)."""
        self._of_frames()
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
        source's need not, its decode not the same each time: media/reader.py). Without a held
        row, version VERSION, the bytes of before the places; with, version FILLED and the held
        column (HELD). ValueError when the probabilities aren't one per frame."""
        version, frame_columns = VERSION, FRAME_COLUMNS
        columns = [self.pts.astype("<i8"), self.crc32.astype("<u4"), self.error.astype("|u1")]
        if self.held is not None:
            version, frame_columns = FILLED, (*frame_columns, HELD)
            columns.append(self.held.astype(HELD[1]))
        if self.probabilities is not None:
            self._one_per_row(self.probabilities)
            frame_columns = (*frame_columns, PROBABILITIES)
            columns.append(self.probabilities.astype(PROBABILITIES[1]))
        header = {
            "version": version,
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
        """The index a file holds (to_bytes), its probabilities with it when it has them, and its
        held rows (version FILLED). Raises ValueError, saying why, for anything else: another
        format or version, a column missing or of another type, a probability outside [0, 1], the
        values cut short or followed by more; a held column in a file of version VERSION, or none
        or no held row in one of version FILLED, which to_bytes never writes; a first row held, or
        a held row whose CRC-32 or error flag isn't the frame's it shows."""
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
        if type(version) is not int or version not in (VERSION, FILLED):
            raise ValueError(
                f"version {json.dumps(version)}, where {VERSION} and {FILLED} are read"
            )
        time_base = _fraction(fields.get("time_base"))
        start_time: Any = fields.get("start_time")
        if start_time is not None and type(start_time) is not int:
            raise ValueError(f"start time {json.dumps(start_time)}, not microseconds")
        offset = end + 1
        tables: dict[str, dict[str, npt.NDArray[Any]]] = {}
        # The held column is version FILLED's own, known there; looked for in a file of version
        # VERSION too, where it is refused, never skipped as a column unknown is.
        filled = version == FILLED
        for name, known, optional in (
            (
                "frames",
                (*FRAME_COLUMNS, HELD) if filled else FRAME_COLUMNS,
                (PROBABILITIES,) if filled else (PROBABILITIES, HELD),
            ),
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
        held = _held(frames, filled)
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
            held,
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


def _held(frames: dict[str, npt.NDArray[Any]], filled: bool) -> npt.NDArray[np.bool_] | None:
    """The held rows of a file's frames table, None for a file of version VERSION, which has none
    (`filled` says the file's version is FILLED). ValueError for what to_bytes never writes, and
    for a held row that isn't the copy of the frame before its hole."""
    found = frames.get(HELD[0])
    if not filled:
        if found is not None:
            raise ValueError(f"a held column in version {VERSION}, which has none")
        return None
    assert found is not None, "known in version FILLED (_table)"
    if found.size and int(found.max()) > 1:
        raise ValueError("a held flag neither 0 nor 1")
    held = found.astype(np.bool_)
    if not held.any():
        # Written at version VERSION (to_bytes): read, it would be written back as other bytes.
        raise ValueError(f"version {FILLED} without a held row, which version {VERSION} holds")
    if held[0]:
        raise ValueError("its first row held, with no frame before it to show")
    shows = np.flatnonzero(~held)[np.cumsum(~held) - 1]  # the place of the frame each row shows
    for name in ("crc32", "error"):
        if np.any(frames[name] != frames[name][shows]):
            raise ValueError(f"a held row's {name} not that of the frame it shows")
    return held


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

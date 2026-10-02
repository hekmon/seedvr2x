"""What seedvr2x reads of a video stream, through ffprobe."""

import json
import re
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

from seedvr2x.media.ffmpeg import MediaError

# ffprobe's words for an absent tag.
UNTAGGED = frozenset({"", "unknown", "unspecified"})


@dataclass(frozen=True)
class VideoStream:
    """The first video stream of a file, as declared. An absent tag reads "", whatever word
    ffprobe has for it ("unknown", "unspecified")."""

    width: int
    height: int
    pix_fmt: str
    frame_rate: Fraction  # r_frame_rate, exact (24000/1001, not 23.976)
    avg_frame_rate: Fraction | None  # None when ffprobe has none (0/0)
    field_order: str  # "progressive", "tt", "bb", "tb", "bt", or "" (unknown)
    sample_aspect: Fraction | None  # None when undeclared (0:1)
    color_space: str  # the matrix: "bt709", "smpte170m", "gbr"...
    color_range: str  # "tv" or "pc"
    color_primaries: str
    color_transfer: str
    chroma_location: str  # "left", "center", "topleft", "top", "bottomleft", "bottom"
    display_matrix: tuple[int, ...] | None  # the 9 values of a display matrix, if declared
    cropped: bool  # a crop declared by the container (MP4 clap, Matroska PixelCrop)


def probe(path: Path) -> VideoStream:
    result = subprocess.run(
        [
            *("ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams"),
            *("-of", "json", str(path)),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise MediaError(f"{path}: ffprobe failed: {result.stderr.strip()}")
    streams = cast(list[dict[str, Any]], json.loads(result.stdout).get("streams", []))
    if not streams:
        raise MediaError(f"{path}: no video stream")
    stream = streams[0]
    side_data = cast(list[dict[str, Any]], stream.get("side_data_list") or [])
    matrices = [d for d in side_data if d.get("side_data_type") == "Display Matrix"]
    frame_rate = _fraction(stream.get("r_frame_rate"))
    if frame_rate is None:
        raise MediaError(f"{path}: no frame rate declared")
    return VideoStream(
        width=int(stream["width"]),
        height=int(stream["height"]),
        pix_fmt=str(stream["pix_fmt"]),
        frame_rate=frame_rate,
        avg_frame_rate=_fraction(stream.get("avg_frame_rate")),
        field_order=_tag(stream.get("field_order")),
        sample_aspect=_fraction(stream.get("sample_aspect_ratio")),
        color_space=_tag(stream.get("color_space")),
        color_range=_tag(stream.get("color_range")),
        color_primaries=_tag(stream.get("color_primaries")),
        color_transfer=_tag(stream.get("color_transfer")),
        chroma_location=_tag(stream.get("chroma_location")),
        display_matrix=_display_matrix(str(matrices[0].get("displaymatrix"))) if matrices else None,
        cropped=any(d.get("side_data_type") == "Frame Cropping" for d in side_data),
    )


def _tag(value: object) -> str:
    text = "" if value is None else str(value)
    return "" if text in UNTAGGED else text


def _fraction(value: object) -> Fraction | None:
    """A positive ratio written "num/den" (frame rates) or "num:den" (sample aspect), else None:
    ffprobe writes 0/0 and 0:1 for none."""
    match = re.fullmatch(r"(\d+)[/:](\d+)", "" if value is None else str(value))
    if match is None or int(match[1]) == 0 or int(match[2]) == 0:
        return None
    return Fraction(int(match[1]), int(match[2]))


def _display_matrix(dump: str) -> tuple[int, ...]:
    """The 9 values of ffprobe's display matrix dump, three rows "0000000N: a b c"."""
    values = tuple(int(v) for line in dump.splitlines() if ":" in line for v in line.split()[1:])
    if len(values) != 9:
        raise MediaError(f"unreadable display matrix: {dump!r}")
    return values

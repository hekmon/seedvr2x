"""What seedvr2x reads of a video stream, through ffprobe."""

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

# RGB pixel formats by bits per component, read at that depth: ffmpeg's swscale doesn't expand
# 8-bit RGB to 16 bits exactly (255 comes out at 65283, not 65535).
RGB_8BIT = frozenset("rgb24 bgr24 rgb0 bgr0 0rgb 0bgr rgba bgra argb abgr gbrp gbrap".split())
RGB_16BIT = frozenset(
    "rgb48le rgb48be bgr48le bgr48be rgba64le rgba64be bgra64le bgra64be gbrp16le gbrp16be"
    " gbrap16le gbrap16be".split()
)


@dataclass(frozen=True)
class VideoStream:
    """The first video stream of a file."""

    width: int
    height: int
    pix_fmt: str
    frame_rate: Fraction  # r_frame_rate, exact (24000/1001, not 23.976)

    @property
    def rgb_bits(self) -> int | None:
        """8 or 16 for the RGB formats read at their depth, else None."""
        if self.pix_fmt in RGB_8BIT:
            return 8
        if self.pix_fmt in RGB_16BIT:
            return 16
        return None


def probe(path: Path) -> VideoStream:
    result = subprocess.run(
        [
            *("ffprobe", "-v", "error", "-select_streams", "v:0", "-of", "json"),
            *("-show_entries", "stream=width,height,pix_fmt,r_frame_rate", str(path)),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    streams = cast(list[dict[str, Any]], json.loads(result.stdout).get("streams", []))
    if not streams:
        raise ValueError(f"{path}: no video stream")
    stream = streams[0]
    return VideoStream(
        width=int(stream["width"]),
        height=int(stream["height"]),
        pix_fmt=str(stream["pix_fmt"]),
        frame_rate=Fraction(str(stream["r_frame_rate"])),
    )

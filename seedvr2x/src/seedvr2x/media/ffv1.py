"""Lossless FFV1 masters from float frames.

The format and the quantisation are those of research/scripts/ffv1_out.py, the master validated
in research/docs/output.md: 16-bit planar RGB rounded to nearest, FFV1 level 3, every frame a
keyframe, 16 slices with CRCs, the exact rational frame rate. Its tags (BT.709 primaries and
transfer, full range) are ffv1_out.py's; DESIGN.md's (copied from the source) come with the
Output milestone.
"""

import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import numpy.typing as npt

# Frame properties only, no pixel conversion: without them the muxer writes no primaries and no
# transfer (ffv1_out.py).
SETPARAMS = "setparams=color_primaries=bt709:color_trc=bt709:colorspace=gbr:range=pc"
RGB_TAGS = [
    *("-colorspace", "rgb", "-color_primaries", "bt709", "-color_trc", "bt709"),
    *("-color_range", "pc"),
]


def to_planar16(frame: npt.NDArray[np.float32]) -> npt.NDArray[np.uint16]:
    """(H, W, 3) float in [0, 1] to the G, B, R planes of gbrp16le, round(x * 65535)."""
    q = np.clip(frame, 0.0, 1.0)
    q *= 65535
    np.rint(q, out=q)
    return np.ascontiguousarray(q.astype("<u2").transpose(2, 0, 1)[[1, 2, 0]])


def write_gbrp16(
    path: Path, frames: npt.NDArray[np.float32], frame_rate: Fraction, slices: int = 16
) -> None:
    """Write frames (T, H, W, 3) float32 in [0, 1] to an FFV1 gbrp16le master at path."""
    _, height, width, _ = frames.shape
    command = ["ffmpeg", "-hide_banner", "-nostdin", "-nostats", "-loglevel", "error", "-y"]
    command += ["-f", "rawvideo", "-pix_fmt", "gbrp16le", "-s", f"{width}x{height}"]
    command += ["-framerate", str(frame_rate), "-i", "-", "-map", "0:v:0"]
    command += ["-fps_mode", "passthrough", "-vf", SETPARAMS, *RGB_TAGS]
    command += ["-c:v", "ffv1", "-level", "3", "-g", "1", "-slices", str(slices)]
    command += ["-slicecrc", "1", "-pix_fmt", "gbrp16le", str(path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None and process.stderr is not None
    try:
        for frame in frames:
            process.stdin.write(memoryview(to_planar16(frame)).cast("B"))
    finally:
        process.stdin.close()
        errors = process.stderr.read().decode(errors="replace")
        process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}: {errors.strip()}")

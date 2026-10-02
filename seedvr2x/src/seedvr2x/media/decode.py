"""Decoding to RGB frames through an ffmpeg pipe."""

import subprocess
from pathlib import Path

import numpy as np
import numpy.typing as npt

from seedvr2x.media.probe import VideoStream


def read_rgb(path: Path, stream: VideoStream) -> npt.NDArray[np.float32]:
    """Every frame of an 8- or 16-bit RGB video: (T, H, W, 3) float32 in [0, 1].

    The frames come at the source's depth, so ffmpeg converts no value, and are counted as
    decoded, not as the container declares (bug 11). An 8-bit frame is numz's own read (OpenCV,
    uint8 / 255 in float32): milestone 1 feeds both the same RGB file. YUV sources and other
    depths need the explicit conversions of DESIGN.md (Input), not yet here.
    """
    bits = stream.rgb_bits
    if bits is None:
        raise ValueError(f"{path}: {stream.pix_fmt}: only 8- and 16-bit RGB sources are read yet")
    pix_fmt, dtype, peak = ("rgb24", np.uint8, 255) if bits == 8 else ("rgb48le", "<u2", 65535)
    result = subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:v:0"),
            *("-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"),
        ],
        capture_output=True,
        check=True,
    )
    frame_bytes = stream.width * stream.height * 3 * bits // 8
    if len(result.stdout) % frame_bytes:
        raise ValueError(f"{path}: {len(result.stdout)} bytes decoded, not whole frames")
    frames = np.frombuffer(result.stdout, dtype=dtype).reshape(-1, stream.height, stream.width, 3)
    return frames.astype(np.float32) / np.float32(peak)

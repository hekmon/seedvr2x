"""ffprobe and the ffmpeg pipes, against the real ffmpeg (skipped without it)."""

import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from seedvr2x.media.decode import read_rgb
from seedvr2x.media.ffv1 import to_planar16, write_gbrp16
from seedvr2x.media.probe import probe

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="no ffmpeg"
)


def encode_rgb24(path: Path, frames: np.ndarray, rate: str = "24000/1001") -> None:
    """Frames (T, H, W, 3) uint8 to a lossless 8-bit RGB FFV1 file."""
    _, height, width, _ = frames.shape
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24"),
            *("-s", f"{width}x{height}", "-framerate", rate, "-i", "-", "-c:v", "ffv1", str(path)),
        ],
        input=frames.tobytes(),
        check=True,
    )


def test_8bit_rgb_reads_as_numz(tmp_path: Path) -> None:
    # Every 8-bit value on every channel must read as numz's uint8 / 255 in float32, bit for
    # bit (media/decode.py).
    values = np.arange(256, dtype=np.uint8)
    frame = np.stack([values, values[::-1], np.roll(values, 85)], axis=-1)[None, None]
    frames = np.repeat(np.repeat(frame, 2, axis=1), 5, axis=0)  # (5, 2, 256, 3)
    path = tmp_path / "rgb.mkv"
    encode_rgb24(path, frames)
    stream = probe(path)
    assert stream.rgb_bits == 8 and stream.frame_rate == Fraction(24000, 1001)
    decoded = read_rgb(path, stream)
    expected = frames.astype(np.float32) / 255.0
    assert decoded.shape == expected.shape
    assert np.array_equal(decoded.view(np.uint32), expected.view(np.uint32))


def test_ffv1_master_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    frames = rng.random((3, 8, 16, 3), dtype=np.float32)
    frames[0, 0, 0] = (-0.5, 0.0, 1.5)  # out of range: clipped
    path = tmp_path / "master.mkv"
    write_gbrp16(path, frames, Fraction(24000, 1001))
    stream = probe(path)
    assert stream.pix_fmt == "gbrp16le" and stream.frame_rate == Fraction(24000, 1001)
    decoded = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gbrp16le", "-"],
        capture_output=True,
        check=True,
    ).stdout
    planes = np.frombuffer(decoded, dtype="<u2").reshape(3, 3, 8, 16)
    expected = np.stack([to_planar16(frame) for frame in frames])
    assert np.array_equal(planes, expected)

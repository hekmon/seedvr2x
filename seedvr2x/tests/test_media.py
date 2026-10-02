"""The FFV1 writer, against the real ffmpeg (skipped without it)."""

import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from seedvr2x.media.ffv1 import to_planar16, write_gbrp16
from seedvr2x.media.probe import probe

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="no ffmpeg"
)


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

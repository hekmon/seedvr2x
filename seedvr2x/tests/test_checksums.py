"""Per-frame checksum files (media/checksums.py): read back as written, refused unless whole."""

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from seedvr2x.media import ffmpeg
from seedvr2x.media.checksums import check_frames, read_checksums, write_checksums
from seedvr2x.media.decode import to_float32
from seedvr2x.media.ffmpeg import MediaError, input_args
from seedvr2x.media.writer import FFV1Writer, Tags


def _usable() -> bool:
    try:
        ffmpeg.check()
    except MediaError:
        return False
    return True


def test_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "a.crc32"
    write_checksums(path, [0, 1, 0xDEADBEEF, 0xFFFFFFFF])
    assert path.read_text() == "00000000\n00000001\ndeadbeef\nffffffff\n"
    assert read_checksums(path) == [0, 1, 0xDEADBEEF, 0xFFFFFFFF]
    write_checksums(path, [])
    assert read_checksums(path) == []


@pytest.mark.parametrize(
    ("content", "said"),
    [
        (None, "missing"),
        (b"00000000\n0000001", "cut short, its last line unfinished"),
        (b"00000000\n0000000g\n", "line 2 not a CRC-32: '0000000g'"),
        (b"00000000\n\n", "line 2 not a CRC-32: ''"),
        (b"DEADBEEF\n", "line 1 not a CRC-32"),
        (b"\xff\n", "not readable"),
    ],
)
def test_refused_unless_whole(tmp_path: Path, content: bytes | None, said: str) -> None:
    path = tmp_path / "a.crc32"
    if content is not None:
        path.write_bytes(content)
    with pytest.raises(MediaError, match=f"^{path}: {said}"):
        read_checksums(path)


@pytest.mark.skipif(not _usable(), reason="needs ffmpeg 7.1 or later with zscale, scdet and ffv1")
def test_check_frames(tmp_path: Path) -> None:
    # Every frame decoded as stored and checked, to the end: those not as written, a count other
    # than the checksums', what ffmpeg reports.
    frames = np.random.default_rng(0).integers(0, 65536, (5, 16, 64, 3), dtype=np.uint16)
    path = tmp_path / "m.mkv"
    with FFV1Writer(path, "gbrp16le", 64, 16, Fraction(25), Tags()) as writer:
        writer.write(to_float32(frames))
    checksums = writer.checksums
    args = input_args(path)
    assert check_frames(args, "gbrp16le", 64, 16, checksums) == []
    altered = [*checksums[:1], checksums[1] ^ 1, *checksums[2:3], checksums[3] ^ 1, checksums[4]]
    assert check_frames(args, "gbrp16le", 64, 16, altered) == ["2 frames not as written: 1, 3"]
    assert check_frames(args, "gbrp16le", 64, 16, checksums[:4]) == [
        "5 frames, where 4 were written"
    ]
    cut = tmp_path / "cut.mkv"
    cut.write_bytes(path.read_bytes()[: path.stat().st_size // 2])
    found = check_frames(input_args(cut), "gbrp16le", 64, 16, checksums)
    assert any(line.startswith("ffmpeg reported: ") for line in found)
    assert any(line.endswith(", where 5 were written") for line in found)

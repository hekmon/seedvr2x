"""Per-frame checksum files (media/checksums.py): read back as written, refused unless whole."""

from pathlib import Path

import pytest

from seedvr2x.media.checksums import read_checksums, write_checksums
from seedvr2x.media.ffmpeg import MediaError


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

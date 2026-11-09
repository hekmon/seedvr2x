"""media/files.py: a file written whole on Windows, against the C runtime's and Python's own
limits there (files._sync), and Linux's way unchanged; no ffmpeg needed."""

from pathlib import Path

import pytest

from media import files


def test_on_windows_a_file_is_synced_writable_and_a_directory_is_left_said_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Measured on native Windows (2026-11-09): fsync of a read-only descriptor is "Bad file
    # descriptor" and os.open of a directory "Permission denied", so a picture's write failed at
    # its end. There a file is opened for writing, and a directory not opened at all.
    opened: list[tuple[str, int]] = []
    real_open = files.os.open

    def recording(path: Path, flags: int) -> int:
        opened.append((Path(path).name, flags))
        return real_open(path, flags)

    monkeypatch.setattr(files.sys, "platform", "win32")
    # Windows' own flag, which Linux's os lacks: its value there (0x8000).
    monkeypatch.setattr(files.os, "O_BINARY", 0x8000, raising=False)
    monkeypatch.setattr(files.os, "open", recording)
    monkeypatch.setattr(files, "_UNSYNCED", [])
    caplog.set_level("DEBUG", logger=files.logger.name)
    files.write_whole(tmp_path / "a.png", b"picture")
    files.write_whole(tmp_path / "b.png", b"picture")
    assert (tmp_path / "a.png").read_bytes() == b"picture"
    assert [name for name, _ in opened] == ["a.png.partial", "b.png.partial"]
    assert all(flags & files.os.O_RDWR and flags & 0x8000 for _, flags in opened)
    said = [r for r in caplog.records if "a directory isn't synced on Windows" in r.getMessage()]
    assert len(said) == 1 and said[0].levelname == "DEBUG"
    assert not list(tmp_path.glob("*.partial"))


def test_on_linux_a_file_and_its_directory_are_synced_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[tuple[str, int]] = []
    real_open = files.os.open

    def recording(path: Path, flags: int) -> int:
        opened.append((Path(path).name, flags))
        return real_open(path, flags)

    monkeypatch.setattr(files.sys, "platform", "linux")
    monkeypatch.setattr(files.os, "open", recording)
    files.write_whole(tmp_path / "a.png", b"picture")
    assert opened == [("a.png.partial", files.os.O_RDONLY), (tmp_path.name, files.os.O_RDONLY)]

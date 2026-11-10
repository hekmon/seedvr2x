"""media/files.py: a file written whole on Windows, against the C runtime's and Python's own
limits there (files._sync), and Linux's way unchanged; a file Windows won't replace while another
program holds it refused in its words (files.held), and a video's directory lock there
(files.lock_directory, main._lock); no ffmpeg needed. The cases named on_windows run natively."""

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import main
from media import files
from media.ffmpeg import MediaError
from seedvr2x.errors import JobError

on_windows = pytest.mark.skipif(sys.platform != "win32", reason="Windows' own")


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


@pytest.mark.skipif(sys.platform == "win32", reason="a read-only fsync is Linux's")
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


def windows_error(winerror: int, path: Path, target: Path | None = None) -> PermissionError:
    """A PermissionError as Windows raises it, its winerror set by hand where Linux sets none."""
    error = PermissionError(13, "The process cannot access the file", str(path), None, None)
    if target is not None:
        error.filename2 = str(target)
    error.winerror = winerror  # type: ignore[misc]  # Windows' attribute, read by getattr
    return error


def test_a_held_file_is_said_with_its_name_and_windows_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = windows_error(32, tmp_path / "a.partial", tmp_path / "a.mkv")
    if sys.platform != "win32":
        assert files.held(error) is None  # Linux's: nothing of Windows'
    monkeypatch.setattr(files.sys, "platform", "win32")
    assert files.held(error) == (
        f"{tmp_path / 'a.mkv'}: Windows refused to replace it with a.partial, beside it: The"
        " process cannot access the file (WinError 32); another program has it open (a player,"
        " an editor, a file manager's preview): close it there, then run the same command again"
    )
    removal = windows_error(32, tmp_path / "a.mkv")
    said = files.held(removal)
    assert said is not None and said.startswith(
        f"{tmp_path / 'a.mkv'}: Windows refused to change or remove it: "
    )
    # Access denied is Windows' answer to more than a hold: each cause said, none promised.
    denied = files.held(windows_error(5, tmp_path / "a.mkv"))
    assert denied == (
        f"{tmp_path / 'a.mkv'}: Windows refused to change or remove it: The process cannot access"
        " the file (WinError 5); another program may have it or the file beside it open (a"
        " player, an antivirus scanning it), it may be read-only, or its folder one Windows"
        " protects from this program (Controlled folder access): mend that, then run the same"
        " command again"
    )
    assert files.held(windows_error(2, tmp_path / "a.mkv")) is None
    # Still an OSError where a failed write's was caught, and a MediaError where a run's is.
    assert issubclass(files.HeldFile, PermissionError) and issubclass(files.HeldFile, MediaError)


def test_a_directory_where_a_file_goes_is_said_as_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Windows refuses a replace onto a directory and its unlink with access denied (WinError 5),
    # and its opening with errno 13 alone (measured natively, impl-win-5, 2026-11-10): said as a
    # directory in the file's place, never as a file held, read-only or protected.
    target = tmp_path / "a.mkv"
    target.mkdir()
    replace = windows_error(5, tmp_path / "a.partial", target)
    opening = PermissionError(13, "Permission denied", str(target))
    if sys.platform != "win32":
        assert files.held(replace) is None and files.refused(opening) is None
    monkeypatch.setattr(files.sys, "platform", "win32")
    assert files.held(replace) == (
        f"{target}: a directory, where seedvr2x needs a file: Windows refused to replace it with"
        " a.partial, beside it: The process cannot access the file (WinError 5); move or rename"
        " that directory, or give another name, then run the same command again"
    )
    assert files.refused(replace) == files.held(replace)
    removal = files.held(windows_error(5, target))
    assert removal is not None and removal.startswith(
        f"{target}: a directory, where seedvr2x needs a file: Windows refused to change or remove"
        " it: "
    )
    assert files.refused(opening) == (
        f"{target}: a directory, where seedvr2x needs a file: Windows refused to open it:"
        " Permission denied (errno 13); move or rename that directory, or give another name,"
        " then run the same command again"
    )
    # A file there, or a directory put onto a directory (replace_whole's PNG segments): the
    # words of before; a file's opening refused with no winerror is no refusal of Windows'.
    assert "a directory, where" not in str(files.held(windows_error(5, tmp_path / "b.mkv")))
    partial = tmp_path / "a.partial"
    partial.mkdir()
    assert "a directory, where" not in str(files.held(windows_error(5, partial, target)))
    assert files.refused(PermissionError(13, "Permission denied", str(tmp_path / "b"))) is None
    # A directory seedvr2x means to remove (rmdir, rmtree) keeps held's words: read-only or
    # protected, it is no directory in a file's place.
    (target / "inside").touch()
    for remove in (target.rmdir, lambda: shutil.rmtree(tmp_path / "missing")):
        with pytest.raises(OSError) as raised:
            remove()
        error = raised.value  # raised in the removal's own frames, given the directory's name
        error.filename, error.winerror = str(target), 5  # type: ignore[attr-defined]
        assert "a directory, where" not in str(files.held(error))
        assert "Windows refused to change or remove it" in str(files.refused(error))
    # Sharing violations are a holder's, whatever the path.
    assert "another program has it open" in str(files.held(windows_error(32, partial, target)))


@on_windows
def test_on_windows_a_directory_where_a_file_goes_is_said_as_one(tmp_path: Path) -> None:
    target = tmp_path / "out.mkv"
    target.mkdir()
    source = tmp_path / "out.partial"
    source.write_bytes(b"x")
    for call in (lambda: os.replace(source, target), lambda: os.unlink(target)):
        with pytest.raises(OSError) as raised:
            call()
        said = files.refused(raised.value)
        assert said is not None and said.startswith(f"{target}: a directory, where seedvr2x")
    with pytest.raises(PermissionError) as opened:
        open(target, "wb")
    said = files.refused(opened.value)
    assert said is not None and "Windows refused to open it" in said and "(errno 13)" in said


def raised_by(call: str, winerror: int, path: Path) -> OSError:
    """The error `call` raises on a file that isn't there, its winerror set by hand: its
    innermost frame is the line of the call, as a C call's error shows it."""
    try:
        if call == "unlink":
            os.unlink(path)
        elif call == "remove":
            path.unlink()
        else:
            open(path, "rb").close()
    except OSError as error:
        error.winerror = winerror  # type: ignore[attr-defined]  # Windows' attribute
        return error
    raise AssertionError("the file exists")


def test_a_denied_opening_is_said_as_one_and_a_denied_removal_as_a_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # main's net words a WinError 5 or 32 by what was being done: a source Windows would not let
    # seedvr2x open, to read it, is no refused change (impl-win-4, 2026-11-09).
    missing = tmp_path / "source.mkv"
    read = raised_by("read", 5, missing)
    if sys.platform != "win32":
        assert files.refused(read) is None  # Linux's: nothing of Windows'
    monkeypatch.setattr(files.sys, "platform", "win32")
    said = files.refused(read)
    assert said is not None
    assert said.startswith(f"{missing}: Windows refused to open it: ")
    assert "(WinError 5)" in said and "exclusively" in said and "permission" in said
    assert "change or remove" not in said
    assert not files.changing(read)
    sharing = files.refused(raised_by("read", 32, missing))
    assert sharing is not None and "refused to open it" in sharing and "(WinError 32)" in sharing
    for call in ("unlink", "remove"):
        removal = raised_by(call, 32, missing)
        assert files.changing(removal)
        assert files.refused(removal) == files.held(removal)
        assert "Windows refused to change or remove it" in str(files.refused(removal))
    # A rename or a replace carries its second name: a change whatever the frame.
    assert files.changing(windows_error(5, tmp_path / "a.partial", missing))
    assert files.refused(windows_error(2, missing)) is None


def test_a_replace_windows_refuses_is_held_file_and_write_whole_leaves_no_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(files.sys, "platform", "win32")
    monkeypatch.setattr(files.os, "O_BINARY", 0x8000, raising=False)
    target = tmp_path / "plan.txt"
    target.write_bytes(b"before")

    def refused(self: Path, other: Path) -> Path:
        raise windows_error(5, self, Path(other))

    monkeypatch.setattr(Path, "replace", refused)
    with pytest.raises(files.HeldFile) as stopped:
        files.write_whole(target, b"after")
    assert str(stopped.value).startswith(f"{target}: Windows refused to replace it with ")
    assert target.read_bytes() == b"before" and not list(tmp_path.glob("*.partial"))


def test_a_second_lock_on_windows_is_refused_in_linuxs_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(main.sys, "platform", "win32")
    monkeypatch.setattr(main, "_LOCKED", [])
    monkeypatch.setattr(main, "_WINDOWS_LOCKS", [])

    def taken(directory: Path) -> int:
        raise BlockingIOError(11, "held by another process")

    monkeypatch.setattr(files, "lock_directory", taken)
    with pytest.raises(JobError) as refused:
        main._lock(tmp_path / "out")  # pyright: ignore[reportPrivateUsage]
    assert str(refused.value) == f"{tmp_path / 'out'}: another seedvr2x is writing it"

    def unidentified(directory: Path) -> int:
        raise OSError(95, "its filesystem gives it no file id to lock it by")

    monkeypatch.setattr(files, "lock_directory", unidentified)
    main._lock(tmp_path / "other")  # pyright: ignore[reportPrivateUsage]
    assert f"{tmp_path / 'other'}: not locked (" in caplog.text
    assert main._LOCKED == [tmp_path / "other"]  # pyright: ignore[reportPrivateUsage]


# A process holding a path open, as another program does: it says "held" once it holds it, and
# holds it until it is killed.
HOLDER = """
import sys, time
held = open(sys.argv[1], "rb")
print("held", flush=True)
time.sleep(60)
"""

LOCKER = """
import sys, time
from pathlib import Path
from media import files
mutex = files.lock_directory(Path(sys.argv[1]))
print("locked", flush=True)
time.sleep(60)
"""


def holding(code: str, path: Path, said: str) -> "subprocess.Popen[str]":
    # The interpreter itself: a venv's python.exe on Windows is a launcher whose Python is its
    # child, still holding the path a moment after the launcher is killed.
    interpreter = getattr(sys, "_base_executable", sys.executable)
    holder = subprocess.Popen(
        [interpreter, "-c", code, str(path)],
        stdout=subprocess.PIPE,
        text=True,
        cwd=Path(main.__file__).parent,
    )
    assert holder.stdout is not None and holder.stdout.readline() == f"{said}\n"
    return holder


@on_windows
def test_on_windows_a_file_another_program_holds_is_refused_not_retried(tmp_path: Path) -> None:
    target = tmp_path / "seg_000001.mkv"
    target.write_bytes(b"old")
    holder = holding(HOLDER, target, "held")
    try:
        with pytest.raises(files.HeldFile) as stopped:
            files.write_whole(target, b"new")
        assert str(stopped.value).startswith(f"{target}: Windows refused to replace it with ")
        assert "(WinError 5)" in str(stopped.value)
        assert target.read_bytes() == b"old" and not list(tmp_path.glob("*.partial"))
    finally:
        holder.kill()
        holder.wait()
    files.write_whole(target, b"new")  # the holder gone, the same write goes through
    assert target.read_bytes() == b"new"


@on_windows
def test_on_windows_a_directory_another_process_locked_is_refused_until_it_ends(
    tmp_path: Path,
) -> None:
    locker = holding(LOCKER, tmp_path, "locked")
    try:
        with pytest.raises(BlockingIOError):
            files.lock_directory(tmp_path)
    finally:
        locker.kill()
        locker.wait()
    # Killed, its mutex abandoned: the next taker has it, and a second take by the same process
    # is main's to refuse (_LOCKED), as flock's of one descriptor.
    mutex = files.lock_directory(tmp_path)
    files.unlock_directory(mutex)
    again = files.lock_directory(tmp_path)
    files.unlock_directory(again)


def test_free_space_and_allocate_are_statvfs_and_posix_fallocate_on_linux(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if sys.platform != "win32":
        stats = files.os.statvfs(tmp_path)
        assert abs(files.free_space(tmp_path) - stats.f_bavail * stats.f_frsize) < 1 << 26
        calls: list[tuple[int, int, int]] = []
        monkeypatch.setattr(files.os, "posix_fallocate", lambda *a: calls.append(a))
        files.allocate(7, 4096)
        assert calls == [(7, 0, 4096)]
    # Windows' twins: shutil.disk_usage's free bytes, and the file's room reserved by its handle
    # (files._reserve, natively below).
    monkeypatch.setattr(files.sys, "platform", "win32")
    import shutil

    assert files.free_space(tmp_path) == shutil.disk_usage(tmp_path).free
    reserved: list[tuple[int, int]] = []
    monkeypatch.setattr(files, "_reserve", lambda *a: reserved.append(a))
    files.allocate(7, 1 << 20)
    assert reserved == [(7, 1 << 20)]


@on_windows
def test_on_windows_a_scratch_is_allocated_without_writing_its_zeros(tmp_path: Path) -> None:
    # 1 GiB reserved and its end set at once, where the C runtime's _chsize_s wrote the zeros at
    # 0.84 s a GiB (impl-win-4, 2026-11-09); what was never written reads as zeros.
    import time

    path = tmp_path / "scratch"
    with path.open("x+b") as file:
        start = time.monotonic()
        files.allocate(file.fileno(), 1 << 30)
        assert time.monotonic() - start < 1.0
        file.seek((1 << 30) - 4096)
        assert file.read() == bytes(4096)
    assert path.stat().st_size == 1 << 30


def test_on_windows_a_disk_without_the_room_is_refused_by_its_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ERROR_DISK_FULL (112) from FileAllocationInfo: an OSError carrying it, as posix_fallocate's
    # ENOSPC, nothing extended.
    monkeypatch.setattr(files.sys, "platform", "win32")
    asked: list[int] = []

    def full(descriptor: int, information: int, size: int) -> int:
        asked.append(information)
        return 112

    monkeypatch.setattr(files, "_file_information", full)
    path = tmp_path / "scratch"
    with path.open("x+b") as file, pytest.raises(OSError) as refused:
        files.allocate(file.fileno(), 1 << 20)
    assert "112" in str(refused.value)  # Windows' WinError, its words or its number
    assert asked == [files.FILE_ALLOCATION_INFO] and path.stat().st_size == 0


@pytest.mark.parametrize("error", files.UNSUPPORTED)
def test_on_windows_a_file_system_without_the_reservation_extends_the_file_saying_so(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error: int,
) -> None:
    # A share, or a driver other than NTFS's: the file extended as before the reservation, by
    # the C runtime's writing its zeros, said at INFO.
    monkeypatch.setattr(files.sys, "platform", "win32")
    monkeypatch.setattr(files, "_file_information", lambda *a: error)
    path = tmp_path / "scratch"
    with caplog.at_level(logging.INFO, logger="cli.media.files"), path.open("x+b") as file:
        files.allocate(file.fileno(), 1 << 20)
    assert path.stat().st_size == 1 << 20
    assert f"doesn't reserve a file's room ahead (WinError {error})" in caplog.text


@on_windows
def test_on_windows_the_ram_line_reads_the_working_set() -> None:
    import run

    resident, peak = run._working_set()  # pyright: ignore[reportPrivateUsage]
    assert 0 < resident <= peak
    assert run._resident() == pytest.approx(resident, rel=0.5)  # pyright: ignore[reportPrivateUsage]
    assert run._peak() >= resident  # pyright: ignore[reportPrivateUsage]

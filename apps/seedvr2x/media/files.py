"""Files written whole: each written beside its place, then renamed into it (DESIGN.md, Pause and
resume: the manifest and the units a resume trusts)."""

import contextlib
import errno
import logging
import os
import re
import sys
import traceback
from pathlib import Path

from media.ffmpeg import MediaError

logger = logging.getLogger("cli.media.files")

# Windows' refusals of a rename over a file, or of its removal (winerror.h): ERROR_ACCESS_DENIED
# for os.replace onto a file another program holds open, ERROR_SHARING_VIOLATION for os.unlink of
# one. Measured natively (impl-win-2, 2026-11-09): a file another process held by open() or by a
# map refused both until the holder ended, 2.96 s later in the test; 400 replaces and removals of
# files nothing held met neither. Windows' own reason is said, never retried: how long another
# program holds a file is its own. ERROR_ACCESS_DENIED is also Windows' answer to a read-only
# file, a directory in the way and a folder it protects from this program (Controlled folder
# access), so it is said with each (impl-win-3's review, 2026-11-09).
SHARING_VIOLATION, ACCESS_DENIED = 32, 5
HELD = (ACCESS_DENIED, SHARING_VIOLATION)


class HeldFile(MediaError, PermissionError):
    """A replace or a removal Windows refused (held): the file, what was refused and Windows'
    reason. A PermissionError too, so that what caught a failed write's OSError still does."""


def held(error: OSError, what: str = "") -> str | None:
    """The words of a Windows refusal of a file's replace or removal (HELD), None for any other
    error: the file (a rename's target, the file put in its place named too), `what` was refused
    (else told by the error), Windows' reason, its likely causes, then the way out."""
    winerror = getattr(error, "winerror", None)
    if sys.platform != "win32" or winerror not in HELD:
        return None
    path = error.filename2 if error.filename2 is not None else error.filename
    action = what or ("replace" if error.filename2 is not None else "change or remove")
    if error.filename2 is not None and error.filename is not None:
        action = f"{action} it with {os.path.basename(error.filename)}, beside it"
    else:
        action = f"{action} it"
    causes = (
        "another program has it open (a player, an editor, a file manager's preview): close it"
        " there"
        if winerror == SHARING_VIOLATION
        else "another program may have it or the file beside it open (a player, an antivirus"
        " scanning it), it may be read-only, or its folder one Windows protects from this program"
        " (Controlled folder access): mend that"
    )
    return (
        f"{path}: Windows refused to {action}: {error.strerror} (WinError {winerror}); {causes},"
        " then run the same command again"
    )


# The calls that change or remove a file, as the source line of an error's innermost frame shows
# them, or as the name of a frame on its way (pathlib's unlink, shutil's rmtree and its helpers):
# a Windows refusal (WinError 5 or 32) names the same file, and no other attribute, whether open()
# or unlink() raised it, so the call that raised it is what tells a refused change from a refused
# opening (impl-win-4, 2026-11-09; os.replace and os.rename also carry a second name, filename2).
_CHANGES = re.compile(
    r"\b(?:os|shutil)\.(?:unlink|remove|removedirs|rmdir|rmtree|rename|renames|replace|truncate)\("
)
_CHANGING_FRAMES = re.compile(
    r"^(?:unlink|remove|removedirs|rmdir|rmtree|_rmtree_unsafe|_rmtree_safe_fd|rename|replace)$"
)


def changing(error: OSError) -> bool:
    """Whether `error` was met changing or removing a file, not reading one: a second name
    (a rename's or a replace's) or, in the innermost frame of its traceback, the line of a call
    that changes or removes (os.unlink, Path.unlink, os.replace, shutil.rmtree and the rest)."""
    if error.filename2 is not None:
        return True
    if error.__traceback__ is None:
        return False
    frames = traceback.extract_tb(error.__traceback__)
    if any(_CHANGING_FRAMES.match(frame.name) for frame in frames[1:]):
        return True
    return bool(frames) and _CHANGES.search(frames[-1].line or "") is not None


def refused(error: OSError) -> str | None:
    """The words of a Windows refusal (HELD) of a file, None for any other error: a change or a
    removal said by `held`; anything else said as the file's opening refused, to read it or to
    write it (open() raises both, the same way), with Windows' reason and its likely causes
    (another program holding the file exclusively, or permissions), never as a change."""
    if sys.platform != "win32" or getattr(error, "winerror", None) not in HELD:
        return None
    if changing(error):
        return held(error)
    return (
        f"{error.filename}: Windows refused to open it: {error.strerror} (WinError"
        f" {error.winerror}); another program may have it open exclusively (a player, an editor,"
        " an antivirus scanning it), or this account may lack the permission to open it (or its"
        " folder is one Windows protects from this program): close it there or mend that, then"
        " run the same command again"
    )


# The first directory whose sync failed, said once.
_UNSYNCED: list[Path] = []


def partial_path(path: Path) -> Path:
    """Where path is written before it is whole: beside it, named path.partial."""
    return path.with_name(f"{path.name}.partial")


def command_partial(path: Path) -> Path:
    """Where a segment command writes path before it is whole (media/writer.py, CommandWriter):
    beside it, `.partial` before its suffix, seg_000001.partial.mkv. The command's muxer picks
    its format by the suffix: ffmpeg n9.0.2 refuses partial_path's seg_000001.mkv.partial,
    "Unable to choose an output format" (2026-10-10)."""
    return path.with_name(f"{path.stem}.partial{path.suffix}")


def replace_whole(partial: Path, path: Path, stale: Path | None = None) -> None:
    """Put partial, a file or a directory of files, in path's place, whole even after a power cut:
    its data reaches the disk before the rename, and the rename before this returns. A rename
    alone can reach the disk before the data it names, which a later resume would trust.

    stale, when given, is the file of the checksums of what path held before: removed once
    partial's data is on the disk, just before the rename, so that a stop during the sync, which
    lasts as long as the file is large, leaves the old file with its own (DESIGN.md, Output,
    Checksums: "a run stopped before keeps the old file with its own")."""
    if partial.is_dir():
        for entry in partial.iterdir():
            _sync(entry)
    _sync(partial)
    try:
        if stale is not None:
            stale.unlink(missing_ok=True)
    except OSError as error:
        refused = held(error, "remove")
        if refused is None:
            raise
        raise HeldFile(refused) from error
    try:
        partial.replace(path)
    except OSError as error:
        refused = held(error, "replace")
        if refused is None:
            raise
        raise HeldFile(refused) from error
    _sync(path.parent)


def make_directories(path: Path) -> None:
    """Create path and its missing parents, each new entry synced in its parent, so that a new
    directory survives a power cut with the files the manifest names in it."""
    missing: list[Path] = []
    while not path.exists():
        missing.append(path)
        path = path.parent
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True)
        _sync(directory.parent)


def write_whole(path: Path, data: bytes) -> None:
    """Write data to path, whole even after a power cut (replace_whole). A failure to write its
    partial file or to put it in place (OSError, or HeldFile where Windows refused it while
    another program held the file) is raised as it is, the partial file removed
    first where there is one, so that nothing of a write that failed stays beside path (design,
    2026-10-12; a review's finding of that day: a playlist that couldn't be put in place, a
    directory at its name, left plan.txt.cuts.m3u8.partial beside the cut list)."""
    partial = partial_path(path)
    try:
        partial.write_bytes(data)
        replace_whole(partial, path)
    except (OSError, HeldFile):
        # The failure raised is the write's own, whatever the removal meets.
        with contextlib.suppress(OSError):
            partial.unlink()
        raise


def _sync(path: Path) -> None:
    flags = os.O_RDONLY
    if sys.platform == "win32":
        # Windows syncs a file through a descriptor open for writing (fsync is the C runtime's
        # _commit: "Bad file descriptor" from a read-only one), and Python's os opens no
        # directory there ("Permission denied"; both on native Windows, 2026-11-09): a directory
        # is left to the filesystem, said once at debug.
        if path.is_dir():
            if not _UNSYNCED:
                _UNSYNCED.append(path)
                logger.debug(
                    "%s: a directory isn't synced on Windows (os opens none): a power cut may"
                    " lose what was last renamed into it",
                    path,
                )
            return
        flags = os.O_RDWR | os.O_BINARY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    except OSError as error:
        # Some filesystems (network, FUSE) can't sync a directory: said once, the run goes on.
        if error.errno not in (errno.EINVAL, errno.ENOTSUP, errno.ENOSYS) or not path.is_dir():
            raise
        if not _UNSYNCED:
            _UNSYNCED.append(path)
            logger.warning(
                "%s: its filesystem can't sync a directory (%s): a power cut may lose what was"
                " last renamed into it",
                path,
                error.strerror,
            )
    finally:
        os.close(descriptor)


def free_space(directory: Path) -> int:
    """The bytes free to this user on the disk of `directory`: statvfs's blocks available to a
    user, as shutil.disk_usage's `free` reads them; on Windows, which has no statvfs,
    shutil.disk_usage itself (GetDiskFreeSpaceExW's free bytes of the whole volume, a user's
    quota not counted)."""
    if sys.platform == "win32":
        import shutil

        return shutil.disk_usage(directory).free
    stats = os.statvfs(directory)
    return stats.f_bavail * stats.f_frsize


def allocate(descriptor: int, size: int) -> None:
    """Give the new, empty file of `descriptor` its `size` bytes on the disk now, so that a full
    disk fails here rather than later, as a fault on a page of its mapping: posix_fallocate; on
    Windows, which has none, its clusters reserved (SetFileInformationByHandle's
    FileAllocationInfo, refused where the disk hasn't the room), then its end set there
    (FileEndOfFileInfo), neither writing a byte: NTFS reads what lies past the data written as
    zeros. Not os.ftruncate there: the C runtime's _chsize_s writes the zeros, 0.84 s a GiB on
    an NTFS disk (impl-win-4, 2026-11-09), seconds to a minute before a large tiled shot."""
    if sys.platform == "win32":
        _reserve(descriptor, size)
        return
    os.posix_fallocate(descriptor, 0, size)


# The errors of a file system that has no FileAllocationInfo or FileEndOfFileInfo (a share, or
# another driver than NTFS's): ERROR_INVALID_FUNCTION, ERROR_NOT_SUPPORTED,
# ERROR_INVALID_PARAMETER (winerror.h). There the C runtime's extension, which writes the zeros,
# as before.
UNSUPPORTED = (1, 50, 87)


def _reserve(descriptor: int, size: int) -> None:
    """allocate's Windows half: the clusters reserved, then the end set, by the file's handle
    (_file_information); where the file system has neither, the file extended by os.ftruncate,
    said; any other refusal (ERROR_DISK_FULL, 112) an OSError, as posix_fallocate's ENOSPC."""
    for information in (FILE_ALLOCATION_INFO, FILE_END_OF_FILE_INFO):
        error = _file_information(descriptor, information, size)
        if error in UNSUPPORTED:
            logger.info(
                "this file system doesn't reserve a file's room ahead (WinError %d): the scratch"
                " file is extended by writing its zeros instead, which takes longer",
                error,
            )
            os.ftruncate(descriptor, size)
            return
        if error:
            raise OSError(errno.EIO, _said(error), None, error)


# FILE_INFO_BY_HANDLE_CLASS (minwinbase.h): FileAllocationInfo, FileEndOfFileInfo, each
# structure one LARGE_INTEGER.
FILE_ALLOCATION_INFO = 5
FILE_END_OF_FILE_INFO = 6


def _file_information(descriptor: int, information: int, size: int) -> int:
    """SetFileInformationByHandle of class `information`, `size` bytes, on the file of
    `descriptor`: 0 once set, else Windows' error."""
    if sys.platform != "win32":
        raise AssertionError("Windows' allocation: allocate takes posix_fallocate elsewhere")
    import ctypes
    import msvcrt

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.SetFileInformationByHandle.argtypes = (
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    )
    value = ctypes.c_int64(size)
    if kernel32.SetFileInformationByHandle(
        msvcrt.get_osfhandle(descriptor), information, ctypes.byref(value), ctypes.sizeof(value)
    ):
        return 0
    return ctypes.get_last_error()


def _said(error: int) -> str:
    """Windows' words for its error `error` (its number alone where ctypes has no FormatError:
    off Windows, where the tests fake it)."""
    import ctypes

    format_error = getattr(ctypes, "FormatError", None)
    return f"WinError {error}" if format_error is None else str(format_error(error)).strip()


def lock_directory(directory: Path) -> int:
    """Windows' twin of flock on a video's output directory (main._lock): a named mutex of the
    directory's identity, its volume's serial number and its file id (os.stat's st_dev and
    st_ino), whatever path names it, taken by this process at once or refused, BlockingIOError,
    where another process holds it. No file is added to the directory, which resume and the
    manifest's checks read whole, and the kernel releases it however the process ends: a
    holder killed leaves it abandoned, which the next taker gets (measured natively on
    2026-11-09: a second process's wait timed out, and after the holder's TerminateProcess the
    next wait took it). In Global\\, so that another session's seedvr2x sees it too; there
    another user's is refused to this one's opening (access denied), held too. OSError where
    the filesystem gives the directory no identity (st_ino 0): main says it isn't locked. The
    handle is the caller's, for unlock_directory. Machine-wide only: a directory on a share that
    another machine's seedvr2x writes is not guarded, as flock isn't across NFS clients that
    don't relay it."""
    if sys.platform != "win32":
        raise AssertionError("Windows' lock: main._lock takes flock elsewhere")
    import _winapi

    identity = os.stat(directory)
    if not identity.st_ino:
        raise OSError(errno.ENOTSUP, "its filesystem gives it no file id to lock it by")
    name = f"Global\\seedvr2x-directory-{identity.st_dev:x}-{identity.st_ino:x}"
    try:
        mutex = _winapi.CreateMutexW(_winapi.NULL, False, name)
    except PermissionError:
        raise BlockingIOError(errno.EAGAIN, "held by another user's process") from None
    if _winapi.WaitForSingleObject(mutex, 0) in (_winapi.WAIT_OBJECT_0, _winapi.WAIT_ABANDONED_0):
        return mutex
    _winapi.CloseHandle(mutex)
    raise BlockingIOError(errno.EAGAIN, "held by another process")


def unlock_directory(mutex: int) -> None:
    """Release lock_directory's mutex and close it. From a thread other than the one that took
    it (the stop's watcher, main._flush) the release is refused, ERROR_NOT_OWNER: the close of its
    last handle frees it all the same."""
    if sys.platform != "win32":
        raise AssertionError("Windows' lock")
    import _winapi

    with contextlib.suppress(OSError):
        _winapi.ReleaseMutex(mutex)
    _winapi.CloseHandle(mutex)

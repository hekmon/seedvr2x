"""Files written whole: each written beside its place, then renamed into it (DESIGN.md, Pause and
resume: the manifest and the units a resume trusts)."""

import contextlib
import errno
import logging
import os
import sys
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

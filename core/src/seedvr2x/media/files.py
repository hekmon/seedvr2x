"""Files written whole: each written beside its place, then renamed into it (DESIGN.md, Pause and
resume: the manifest and the units a resume trusts)."""

import contextlib
import errno
import hashlib
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# The first directory whose sync failed, said once.
_UNSYNCED: list[Path] = []


# Under it, a size is said in MiB (size_said): 0.1 GiB.
TINY = 2**30 // 10


def size_said(count: int, places: int = 1) -> str:
    """A size as the logs say it before a long read: "16.5 GiB" to `places` decimals, and "41 MiB"
    under 0.1 GiB, where "0.0 GiB" says nothing (decided on 2026-10-29); "12 KiB" under 1 MiB."""
    if count < 2**20:
        return f"{count / 2**10:.0f} KiB"
    if count < TINY:
        return f"{count / 2**20:.0f} MiB"
    return f"{count / 2**30:.{places}f} GiB"


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
    if stale is not None:
        stale.unlink(missing_ok=True)
    partial.replace(path)
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


def sha256(path: Path) -> str:
    """The SHA-256 of a file's content, in hex: models and inputs are identified by their content
    (DESIGN.md, Pause and resume)."""
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def write_whole(path: Path, data: bytes) -> None:
    """Write data to path, whole even after a power cut (replace_whole). A failure to write its
    partial file or to put it in place (OSError) is raised as it is, the partial file removed
    first where there is one, so that nothing of a write that failed stays beside path (design,
    2026-10-12; a review's finding of that day: a playlist that couldn't be put in place, a
    directory at its name, left plan.txt.cuts.m3u8.partial beside the cut list)."""
    partial = partial_path(path)
    try:
        partial.write_bytes(data)
        replace_whole(partial, path)
    except OSError:
        # The failure raised is the write's own, whatever the removal meets.
        with contextlib.suppress(OSError):
            partial.unlink()
        raise


def _sync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
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

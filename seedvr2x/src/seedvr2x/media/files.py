"""Files written whole: each written beside its place, then renamed into it (DESIGN.md, Pause and
resume: the manifest and the units a resume trusts)."""

import errno
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# The first directory whose sync failed, said once.
_UNSYNCED: list[Path] = []


def partial_path(path: Path) -> Path:
    """Where path is written before it is whole: beside it, named path.partial."""
    return path.with_name(f"{path.name}.partial")


def replace_whole(partial: Path, path: Path) -> None:
    """Put partial, a file or a directory of files, in path's place, whole even after a power cut:
    its data reaches the disk before the rename, and the rename before this returns. A rename
    alone can reach the disk before the data it names, which a later resume would trust."""
    if partial.is_dir():
        for entry in partial.iterdir():
            _sync(entry)
    _sync(partial)
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


def write_whole(path: Path, data: bytes) -> None:
    """Write data to path, whole even after a power cut (replace_whole)."""
    partial = partial_path(path)
    partial.write_bytes(data)
    replace_whole(partial, path)


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

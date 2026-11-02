"""Where a tiled shot's frames wait, in the command line (DESIGN.md, Pipeline step 3): system
memory up to the plan's allowance, else a memory-mapped file in the shot's unit directory (an
8K still's tiled decode is 0.2 GB a frame, a long video's a good deal more). It is handed to the
library's tiled passes as their store (tiling.Store), which the library holds in system memory
alone otherwise (tiling.InMemory); a store that can't be made is a ScratchError, a MediaError,
so that the command line reports it as a file the run can't write."""

import contextlib
import logging
import mmap
import os
from collections.abc import Generator
from pathlib import Path

import torch
from torch import Tensor

from media.ffmpeg import MediaError
from media.files import make_directories
from seedvr2x.model import COMPUTE_DTYPE
from seedvr2x.tilegrid import CHANNELS, SCRATCH, scratch_bytes

logger = logging.getLogger("cli.scratch")

# The bytes of a scratch kept in system memory where its pass is given no allowance (Scratch), a
# larger one going to its file: for a test or a script that calls shot.encode_shot or
# shot.decode_shot without one. 8 GiB, which holds 675 frames at 1080p (padded to 1104x1920,
# 12,718,080 bytes each) and 170 at 4K (2192x3840, 50,503,680). A run gives each pass its plan's
# allowance, which the planner sets from the host's memory (planner.plan; DESIGN.md, Memory
# planner, Host RAM).
DEFAULT_ALLOWANCE = 8 * 2**30


class ScratchError(MediaError):
    """A scratch that can't be made: no directory to put its file in, no room for it there, or no
    mapping it; the message says which, with the bytes it takes.

    PROVISIONAL: a MediaError, a file the run can't write: the command line reports its message
    and what the run keeps, the units made before it, with exit status 1 (main._run): a full disk
    is the user's to fix, and the same command then resumes. DESIGN.md doesn't say how it is
    reported."""


class Scratch:
    """Where a tiled shot's frames wait (DESIGN.md, Pipeline step 3): a store of (3, frames, H, W)
    bfloat16 on the CPU, zeros at first. In system memory when its bytes (needed) are within
    `allowance` (None: DEFAULT_ALLOWANCE), else a memory-mapped file in `directory`, the shot's
    unit directory (SCRATCH).

    The file is derived data, never recorded: it goes when its store is closed, on success and on
    failure alike, a mapping the system refuses included, and nothing stays open on it. A run
    killed leaves it, for the resume to discard."""

    def __init__(self, directory: Path | None, allowance: int | None = None) -> None:
        self.directory = directory
        self.allowance = DEFAULT_ALLOWANCE if allowance is None else allowance
        self._open = False

    @staticmethod
    def needed(frames: int, padded: tuple[int, int]) -> int:
        """The bytes of a store of `frames` frames padded to (height, width) pixels: 12.7 MB per
        1080p frame, padded to 1104x1920, 50.5 MB per 4K frame, padded to 2192x3840
        (tilegrid.scratch_bytes, there without torch for the planner to count with)."""
        return scratch_bytes(frames, padded)

    def where(self, frames: int, padded: tuple[int, int]) -> str:
        """Where a store of that size is while it is open, for the logs: system memory, or its
        file's path."""
        if self.directory is None or self.needed(frames, padded) <= self.allowance:
            return "system memory"
        return str(self.directory / SCRATCH)

    @contextlib.contextmanager
    def store(self, frames: int, padded: tuple[int, int]) -> Generator[Tensor]:
        """A store of `frames` frames padded to (height, width) pixels, open while the with block
        runs: (3, frames, height, width) bfloat16 on the CPU, zeros. One at a time.

        ScratchError when it takes a file and there is no directory, no room for it there, or no
        mapping it. The file is removed on leaving the block; its mapping goes with the last
        tensor on it, so nothing taken from the store is kept past the block but copies."""
        if self._open:
            raise RuntimeError("a scratch holds one store at a time")
        size = self.needed(frames, padded)
        shape = (CHANNELS, frames, *padded)
        self._open = True
        try:
            if size <= self.allowance:
                yield torch.zeros(shape, dtype=COMPUTE_DTYPE)
                return
            with self._file(size) as flat:
                yield flat.view(shape)
        finally:
            self._open = False

    @contextlib.contextmanager
    def _file(self, size: int) -> Generator[Tensor]:
        """The scratch's file, new, `size` bytes of zeros, mapped (_mapped): a flat tensor, the
        file there while the with block runs."""
        if self.directory is None:
            # PROVISIONAL: the planner refuses such a job before the run (DESIGN.md, Memory
            # planner); DESIGN.md doesn't say what the run itself does.
            raise ScratchError(
                f"a scratch of {size} bytes, over the {self.allowance} allowed in system memory,"
                " and no directory for its file"
            )
        path = self.directory / SCRATCH
        made = False
        try:
            try:
                make_directories(self.directory)
                # PROVISIONAL: a killed run's file, if the resume left it there, is replaced, not
                # refused: it is derived data. Unlinked, never truncated, so that a process still
                # mapping it keeps its own.
                path.unlink(missing_ok=True)
                # Read and written: the mapping is of this descriptor, and writes.
                with path.open("x+b") as file:
                    made = True
                    # Allocated whole: a full disk fails here, not later as a SIGBUS on a page of
                    # the mapping.
                    os.posix_fallocate(file.fileno(), 0, size)
                    # Mapped while the file allocated is still open, through its descriptor.
                    flat = _mapped(file.fileno(), path, size)
            except OSError as error:
                raise ScratchError(f"{path}: the shot's scratch, {size} bytes: {error}") from error
            yield flat
        finally:
            # Whatever ended the block, an allocation interrupted included. A file that stays
            # changes no result: said, and the run goes on.
            try:
                if made:
                    path.unlink(missing_ok=True)
            except OSError as error:
                logger.warning("%s: the shot's scratch, not removed: %s", path, error)


def _mapped(descriptor: int, path: Path, size: int) -> Tensor:
    """The file open as `descriptor`, read and written, `size` bytes, as a flat tensor in
    COMPUTE_DTYPE: a shared mapping, so that what is written through the tensor is the file's,
    which the system writes out and drops from memory as it needs to.

    Mapped through the descriptor itself (mmap), which is the file that was allocated, whatever
    its name has become: by its name, path, a link put there since the allocation would be mapped
    in its place, and another file written (tests/test_scratch.py). path is for the message.

    The mapping is never closed here: the tensor's storage keeps the mapping object, which goes
    with the last tensor on it, as does the duplicate of the descriptor it holds, the descriptor
    given closed or not. Closed under a tensor, it would leave the tensor on memory that is no
    longer mapped, without an error. A mapping the system refuses is a ScratchError, and leaves
    nothing open on the file: torch.from_file, which maps a name, kept its own descriptor on it
    then."""
    try:
        # Shared, read and written: mmap's defaults.
        mapping = mmap.mmap(descriptor, size)
    except OSError as error:
        raise ScratchError(
            f"{path}: the shot's scratch, {size} bytes, not mapped: {error}"
        ) from error
    return torch.frombuffer(mapping, dtype=COMPUTE_DTYPE)

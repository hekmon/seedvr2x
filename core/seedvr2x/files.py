"""What the library says and checks of a file: its SHA-256, by which the models and the inputs are
identified (DESIGN.md, Pause and resume), and its size as the logs say it before a long read.
The library's (DESIGN.md, Repository and apps), plain Python: the pull hashes and says the model
files with them, and the command line's inputs and outputs likewise (media/files.py has the
rest: the files written whole)."""

import hashlib
from pathlib import Path

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


def sha256(path: Path) -> str:
    """The SHA-256 of a file's content, in hex: models and inputs are identified by their content
    (DESIGN.md, Pause and resume)."""
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()

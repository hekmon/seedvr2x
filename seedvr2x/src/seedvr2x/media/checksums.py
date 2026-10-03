"""Per-frame checksums, kept with what seedvr2x writes so that it can be read back checked end to
end (DESIGN.md: Colour correction, input copy; Output, Checksums): a CRC-32 of each frame as
written, one per line in frame order, as 8 lowercase hex digits. FFV1's slice CRCs alone let
damage through, which ffmpeg decodes without a word."""

import re
from collections.abc import Sequence
from pathlib import Path

from seedvr2x.media.ffmpeg import MediaError
from seedvr2x.media.files import write_whole

_LINE = re.compile(r"[0-9a-f]{8}")


def write_checksums(path: Path, checksums: Sequence[int]) -> None:
    """Write the checksums of a file's frames to path, whole (files.write_whole)."""
    write_whole(path, "".join(f"{checksum:08x}\n" for checksum in checksums).encode())


def read_checksums(path: Path) -> list[int]:
    """The checksums at path, in frame order. Raises MediaError if the file is missing or not
    entirely checksums."""
    try:
        lines = path.read_bytes().decode("ascii").split("\n")
    except FileNotFoundError:
        raise MediaError(f"{path}: missing") from None
    except (OSError, UnicodeDecodeError) as error:
        raise MediaError(f"{path}: not readable: {error}") from None
    if lines[-1] != "":
        raise MediaError(f"{path}: cut short, its last line unfinished")
    for number, line in enumerate(lines[:-1], 1):
        if _LINE.fullmatch(line) is None:
            raise MediaError(f"{path}: line {number} not a CRC-32: {line[:20]!r}")
    return [int(line, 16) for line in lines[:-1]]

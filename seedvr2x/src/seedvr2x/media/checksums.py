"""Per-frame checksums, kept with what seedvr2x writes so that it can be read back checked end to
end (DESIGN.md: Colour correction, input copy; Output, Checksums): a CRC-32 of each frame as
written, one per line in frame order, as 8 lowercase hex digits. FFV1's slice CRCs alone let
damage through, which ffmpeg decodes without a word."""

import re
import subprocess
import tempfile
import zlib
from collections import deque
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


def frame_bytes(pix_fmt: str, width: int, height: int) -> int:
    """The bytes of a raw frame of pix_fmt, its planes packed as ffmpeg's rawvideo lays them out:
    those the checksums of an output are of (Writer.checksums)."""
    if pix_fmt in ("gbrp16le", "rgb48be"):
        return width * height * 3 * 2
    if pix_fmt == "yuv420p10le":
        return (width * height + 2 * ((width + 1) // 2) * ((height + 1) // 2)) * 2
    raise ValueError(f"no checksums of {pix_fmt} frames")


def check_frames(
    input_args: Sequence[str], pix_fmt: str, width: int, height: int, checksums: Sequence[int]
) -> list[str]:
    """Decode the frames of what ffmpeg reads with input_args as they are stored, pix_fmt, and
    check each against its checksum: what is wrong, said in a few words each (the frames failing
    theirs, a count other than the checksums', ffmpeg's errors); nothing when every frame is as
    written. Read to the end, whatever it finds."""
    size = frame_bytes(pix_fmt, width, height)
    command = [
        *("ffmpeg", "-hide_banner", "-nostdin", "-nostats", "-loglevel", "error", *input_args),
        *("-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"),
    ]
    wrong: list[int] = []
    frames = 0
    found: list[str] = []
    # ffmpeg's errors go to a file, which it never waits on; what it reports fails the check.
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log, process_group=0)
        assert process.stdout is not None
        with process.stdout as frames_read:
            while frame := frames_read.read(size):
                if len(frame) != size:
                    found.append(f"its last frame cut short, {len(frame)} of {size} bytes")
                    break
                if frames < len(checksums) and zlib.crc32(frame) != checksums[frames]:
                    wrong.append(frames)
                frames += 1
        status = process.wait()
        log.seek(0)
        said = " / ".join(line.decode(errors="replace").strip() for line in deque(log, maxlen=5))
    if wrong:
        listed = ", ".join(map(str, wrong[:10])) + (", ..." if len(wrong) > 10 else "")
        found.append(f"{len(wrong)} frames not as written: {listed}")
    if frames != len(checksums):
        found.append(f"{frames} frames, where {len(checksums)} were written")
    if status != 0:
        found.append(f"ffmpeg's exit status {status}")
    if said:
        found.append(f"ffmpeg reported: {said}")
    return found

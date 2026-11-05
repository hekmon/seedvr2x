"""A shot's input copy, read back for the colour correction of its decode (DESIGN.md, Colour
correction): the command line's reference for the library's decode_shot (seedvr2x/shot_pass.py,
shot_pass.Reference), which corrects each slice it decodes against the slice's own frames of the
copy. The pass itself, the encode, the DiT's windows and the decode, is the library's
(seedvr2x/shot_pass.py); run.py's run_job runs it, unit by unit (units.py).
"""

from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from media.checksums import read_checksums
from media.conversion import Conversion
from media.decode import Decoder, to_float32
from media.ffmpeg import MediaError, input_args

# A shot's input copy read back: gbrp16le to gbrp16le, which zscale passes through unchanged.
COPY_READ = Conversion("gbrp16le", "gbr", "full", "left")


@dataclass(frozen=True)
class Correction:
    """What a shot's decode with `split` needs besides its latents (DESIGN.md, Colour correction):
    its input copy, the frames its encode read, width x height as the input stores them
    (units.COPY), and their checksums (units.CHECKSUMS); for a later run of a shot (DESIGN.md,
    Several runs of a shot), `first`, the run's first frame in the shot, the copy being the
    shot's, its first run's: the run is corrected against the copy's frames from there, the
    frames a single run fed the shot from that frame would have copied."""

    copy: Path
    checksums: Path
    width: int
    height: int
    first: int = 0

    def reader(
        self, count: int
    ) -> AbstractContextManager[Callable[[int], npt.NDArray[np.float32]]]:
        """read(n), the next n of the copy's `count` first frames (_copy_reader): the reference
        the decode corrects against (shot_pass.Reference)."""
        return _copy_reader(self, count)


class CopyError(MediaError):
    """A shot's input copy that its decode can't read whole and intact: missing, cut short,
    failing FFV1's slice CRCs, a frame failing its checksum, or its checksums missing or damaged.
    It is derived data: the run stops, and the next one makes it again from the input, keeping the
    shot's latent and windows (DESIGN.md, Colour correction)."""

    def __init__(self, copy: Path, error: MediaError) -> None:
        super().__init__(f"{copy}: the shot's input copy, not readable whole and intact: {error}")
        self.copy = copy


@contextmanager
def _copy_reader(
    correction: Correction, count: int
) -> Generator[Callable[[int], npt.NDArray[np.float32]]]:
    """read(n), the next n frames of the shot's input copy, (n, H, W, 3) float32 in [0, 1] as the
    encode read them, each checked against its checksum before it is given (Decoder), `count` of
    them. The copy is read strictly too, so that what ffmpeg reports names the cause: the copy
    missing, a slice failing its CRC, the file cut short. A copy that fails either check, or whose
    checksums are missing or damaged, raises CopyError."""
    copy = correction.copy
    with _copy_failing(copy):
        checksums = read_checksums(correction.checksums)
        if len(checksums) != count:
            raise MediaError(
                f"{correction.checksums}: {len(checksums)} checksums, for {count} frames"
            )
    args = (correction.width, correction.height, count)
    decoder = Decoder(input_args(copy), COPY_READ, *args, strict=True, checksums=checksums)

    def read(n: int) -> npt.NDArray[np.float32]:
        with _copy_failing(copy):
            frames = decoder.read(n)
            if frames.shape[0] != n:
                raise decoder.failure(f"it ended after {decoder.decoded} of its {count} frames")
        return to_float32(frames)

    try:
        yield read
    except BaseException:
        decoder.stop()
        raise
    with _copy_failing(copy):
        decoder.finish()


@contextmanager
def _copy_failing(copy: Path) -> Generator[None]:
    """A failure to read the copy, raised as CopyError; what the reader's user raises isn't."""
    try:
        yield
    except MediaError as error:
        raise CopyError(copy, error) from error

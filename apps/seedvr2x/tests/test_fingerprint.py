"""The conversion chain's fingerprint (media/fingerprint.py): the same on every run, and another
when a conversion gives other values, as a zimg upgrade would."""

import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from media import conversion, ffmpeg, writer
from media import fingerprint as digests
from media.conversion import Conversion
from media.decode import decode_output
from media.ffmpeg import MediaError
from media.fingerprint import deband_fingerprint, fingerprint


def _usable() -> bool:
    try:
        ffmpeg.check()
    except MediaError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _usable(), reason="needs ffmpeg 7.1 or later with zscale, idet and ffv1"
)


def test_the_same_every_run() -> None:
    first = fingerprint()
    assert re.fullmatch(r"[0-9a-f]{64}", first)
    assert fingerprint() == first


@pytest.mark.parametrize(
    ("module", "name", "value"),
    [
        # The decode's chroma kernel, Catmull-Rom, the one before bilinear (2026-10-20): a job's
        # environment read with it differs.
        (conversion, "KERNEL", "f=bicubic:param_a=0:param_b=0.5"),
        # The yuv420p10le master's chroma kernel.
        (writer, "CHROMA_KERNEL", "f=bicubic"),
    ],
)
def test_another_when_a_conversion_changes(
    monkeypatch: pytest.MonkeyPatch, module: object, name: str, value: str
) -> None:
    first = fingerprint()
    monkeypatch.setattr(module, name, value)
    assert fingerprint() != first


def test_failed_conversions_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(conversion, "KERNEL", "f=none_such")
    with pytest.raises(MediaError, match="ffmpeg failed the startup check's test conversions"):
        fingerprint()


def test_the_deband_chains_have_a_digest_of_their_own() -> None:
    # --deband's chains are no part of the conversions' digest (resume.DEBAND_KEY): a job that
    # never debanded is not refused when ffmpeg's deband alone changes. Here the filter's
    # string stands for the change.
    plain, banded = fingerprint(), deband_fingerprint()
    assert re.fullmatch(r"[0-9a-f]{64}", banded) and banded != plain
    assert deband_fingerprint() == banded


def test_the_deband_digest_moves_with_the_deband_chain_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plain, banded = fingerprint(), deband_fingerprint()
    monkeypatch.setattr(conversion, "DEBAND", "deband=blur=0:1thr=0.2:2thr=0.2:3thr=0.2")
    assert fingerprint() == plain
    assert deband_fingerprint() != banded


def test_the_deband_digest_moves_with_zscale_too(monkeypatch: pytest.MonkeyPatch) -> None:
    banded = deband_fingerprint()
    monkeypatch.setattr(conversion, "KERNEL", "f=bicubic:param_a=0:param_b=0.5")
    assert deband_fingerprint() != banded


def _share(pix_fmt: str, matrix: str, color_range: str, location: str, frame: bytes) -> float:
    """The share of the output samples that differ between a chain with deband and the same
    without, ffmpeg alone, on `frame`."""
    outputs: list[np.ndarray] = []
    for deband in (True, False):
        chain = Conversion(pix_fmt, matrix, color_range, location, deband=deband)
        outputs.append(_run(chain, frame))
    return float((outputs[0] != outputs[1]).mean())


def _run(chain: Conversion, frame: bytes) -> "np.ndarray":
    import tempfile

    with tempfile.TemporaryDirectory(prefix="seedvr2x-") as name:
        directory = Path(name)
        (directory / "in").write_bytes(frame)
        size = f"{digests.WIDTH}x{digests.HEIGHT}"
        command = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y"]
        command += [
            "-f",
            "rawvideo",
            "-pix_fmt",
            chain.planar,
            "-s",
            size,
            "-i",
            str(directory / "in"),
        ]
        command += ["-map", "0:v:0", *decode_output(chain), str(directory / "out")]
        subprocess.run(command, check=True)
        return np.fromfile(directory / "out", dtype="<u2")


@pytest.mark.parametrize(
    ("pix_fmt", "matrix", "color_range", "location"),
    [("yuv420p", "709", "limited", "left"), ("yuvj420p", "470bg", "full", "center")],
)
def test_the_gradient_frame_is_one_deband_changes(
    pix_fmt: str, matrix: str, color_range: str, location: str
) -> None:
    # Measured 2026-11-11 (ffmpeg 9.0.2): 77% of the samples of the limited range's chain
    # and 88% of the full range's, against 0.32% and 2.3% on the noise frame. The floor is well
    # under the measure and far over the noise's.
    gradient = digests._gradient(pix_fmt)  # pyright: ignore[reportPrivateUsage]
    assert len(gradient) == digests.WIDTH * digests.HEIGHT * 3 // 2
    assert _share(pix_fmt, matrix, color_range, location, gradient) > 0.5
    noise = digests._frame(pix_fmt)  # pyright: ignore[reportPrivateUsage]
    assert _share(pix_fmt, matrix, color_range, location, noise) < 0.05


def test_the_gradient_is_the_same_every_time_and_only_for_8_bit_420() -> None:
    first = digests._gradient("yuv420p")  # pyright: ignore[reportPrivateUsage]
    assert digests._gradient("yuv420p") == first  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(ValueError, match="no gradient frame in yuv444p"):
        digests._gradient("yuv444p")  # pyright: ignore[reportPrivateUsage]

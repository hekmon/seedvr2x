"""The conversion chain's fingerprint (media/fingerprint.py): the same on every run, and another
when a conversion gives other values, as a zimg upgrade would."""

import re

import pytest

from seedvr2x.media import conversion, ffmpeg, writer
from seedvr2x.media.ffmpeg import MediaError
from seedvr2x.media.fingerprint import fingerprint


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
        # The decode's chroma kernel, slightly off Catmull-Rom.
        (conversion, "KERNEL", "f=bicubic:param_a=0:param_b=0.55"),
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

"""A source whose sides its chroma subsampling doesn't divide, refused at the start (DESIGN.md,
Input, zimg, decided on 2026-11-13): zimg refuses such frames, and a run failed on the reader's
"ffmpeg's frames end before frame 0", which named no cause. The rule is the library's
(`yuv.check_size`, held in its test_yuv.py); this file holds the command line's words, the
remake they give and the order: nothing planned or loaded before the refusal."""

import shlex
import subprocess
from pathlib import Path

import pytest
from test_rate import needs_ffmpeg

import main as cli
from media import rate
from media.ffmpeg import MediaError
from media.source import declare, examine

pytestmark = needs_ffmpeg


def made(path: Path, size: str, pix_fmt: str, *extra: str) -> Path:
    """An FFV1 file of 3 frames of `size` in `pix_fmt`, with `extra` input options first."""
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            *extra,
            "-f",
            "lavfi",
            "-i",
            f"testsrc=s={size}:r=25",
            "-frames:v",
            "3",
            "-c:v",
            "ffv1",
            "-pix_fmt",
            pix_fmt,
            str(path),
        ],
        check=True,
    )
    return path


def command_of(message: str) -> list[str]:
    """The remake's words, from its ffmpeg to its output."""
    line = next(line for line in message.splitlines() if line.startswith("ffmpeg"))
    return shlex.split(line)


@pytest.mark.parametrize(
    ("size", "pix_fmt", "chroma", "layout", "cropped", "dropped"),
    [
        ("66x48", "yuv420p", "4:2:0", "2 columns by 2 rows", "66x48", None),
        ("65x48", "yuv420p", "4:2:0", "2 columns by 2 rows", "64x48", "the last column"),
        ("64x47", "yuv420p10le", "4:2:0", "2 columns by 2 rows", "64x46", "the last row"),
        ("67x49", "yuv420p", "4:2:0", "2 columns by 2 rows", "66x48", "the last column and"),
        ("66x48", "yuv411p", "4:1:1", "4 columns by 1 row", "64x48", "the last 2 columns"),
        ("64x47", "yuv440p", "4:4:0", "1 column by 2 rows", "64x46", "the last row"),
        ("66x50", "yuv410p", "4:1:0", "4 columns by 4 rows", "64x48", "the last 2 columns and"),
    ],
)
def test_a_source_the_subsampling_doesnt_divide_is_refused_with_its_remake(
    tmp_path: Path,
    size: str,
    pix_fmt: str,
    chroma: str,
    layout: str,
    cropped: str,
    dropped: str | None,
) -> None:
    path = made(tmp_path / "odd.mkv", size, pix_fmt)
    if dropped is None:
        # 66x48 divides 4:2:0: a control, accepted.
        assert examine(path).frames == 3
        return
    with pytest.raises(MediaError) as refused:
        declare(path)
    message = str(refused.value)
    assert message.startswith(f"{path}: ")
    width, height = size.split("x")
    assert f"frames of {width}x{height} in {pix_fmt}" in message
    assert f"{chroma}, {layout}" in message
    assert f"crop them to {cropped}" in message
    assert dropped is None or dropped in message
    words = command_of(message)
    assert words[words.index("-filter:v:0") + 1] == f"crop={cropped.replace('x', ':')}:0:0"
    assert words[-1] == "remade.mkv" and words[words.index("-c:v:0") + 1] == "ffv1"


@pytest.mark.parametrize(
    ("size", "pix_fmt"),
    [("65x47", "yuv420p"), ("66x48", "yuv411p")],
)
def test_the_remake_gives_a_file_the_command_line_accepts(
    tmp_path: Path, size: str, pix_fmt: str
) -> None:
    path = made(tmp_path / "odd.mkv", size, pix_fmt)
    with pytest.raises(MediaError) as refused:
        examine(path)
    words = command_of(str(refused.value))
    words[words.index("SOURCE")] = str(path)
    words[-1] = str(tmp_path / "remade.mkv")
    subprocess.run([*words[:1], "-v", "error", "-y", *words[1:]], check=True)
    remade = examine(tmp_path / "remade.mkv")
    assert remade.frames == 3
    assert remade.stream.width % 4 == 0 and remade.stream.height % 2 == 0


def test_the_remake_keeps_the_other_streams(tmp_path: Path) -> None:
    path = tmp_path / "odd.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=s=65x47:r=25",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-frames:v",
            "3",
            "-c:v",
            "ffv1",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "flac",
            str(path),
        ],
        check=True,
    )
    with pytest.raises(MediaError) as refused:
        declare(path)
    words = command_of(str(refused.value))
    words[words.index("SOURCE")] = str(path)
    words[-1] = str(tmp_path / "remade.mkv")
    subprocess.run([*words[:1], "-v", "error", "-y", *words[1:]], check=True)
    kinds = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(tmp_path / "remade.mkv"),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert kinds == ["video", "audio"]


def test_the_command_is_the_rate_remakes_without_its_rate() -> None:
    assert rate.crop_remake(64, 46) == (
        "ffmpeg -i SOURCE -map 0 -map -0:d -c copy -c:v:0 ffv1 -level:v:0 3 -g:v:0 1"
        " -slicecrc:v:0 1 -fps_mode:v:0 passthrough -filter:v:0 crop=64:46:0:0 remade.mkv"
    )


def test_the_run_is_refused_before_anything_is_planned_or_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from seedvr2x import memory, pull

    monkeypatch.setattr(memory, "startup", lambda *a, **k: pytest.fail("the card was read"))
    monkeypatch.setattr(pull, "fetch", lambda *a, **k: pytest.fail("a model was fetched"))
    path = made(tmp_path / "odd.mkv", "65x48", "yuv420p")
    out = tmp_path / "out"
    status = cli.main([str(path), "-o", str(out), "--model-dir", str(tmp_path)])
    assert status == 1
    assert not out.exists()
    error = caplog.text
    assert "frames of 65x48 in yuv420p" in error and "crop=64:48:0:0" in error

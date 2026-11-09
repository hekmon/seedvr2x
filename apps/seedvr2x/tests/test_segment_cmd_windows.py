"""The segment command on Windows (media/writer.py, CommandWriter): cmd.exe (COMSPEC, Popen's
shell) in a Job Object of its own (ffmpeg.contained), {out} quoted for cmd. The words run
anywhere, Windows faked; the commands themselves natively, with an ffmpeg on PATH."""

import shutil
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from media import writer as writer_module
from media.ffmpeg import MediaError
from media.writer import CommandWriter, Tags, check_command, count_frames

on_windows = pytest.mark.skipif(
    sys.platform != "win32" or shutil.which("ffmpeg") is None,
    reason="Windows' cmd.exe and Job Objects, with an ffmpeg on PATH",
)
RATE = Fraction(24000, 1001)
BT709 = Tags("bt709", "bt709")


def test_on_windows_out_is_a_double_quoted_word_and_a_percent_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(writer_module.sys, "platform", "win32")
    partial = tmp_path / "seedvr2x trial '$ x" / "seg_000001.partial.mkv"
    assert writer_module._word(partial) == f'"{partial}"'  # pyright: ignore[reportPrivateUsage]
    held = tmp_path / "100%" / "seg_000001.partial.mkv"
    with pytest.raises(MediaError) as refused:
        writer_module._word(held)  # pyright: ignore[reportPrivateUsage]
    assert str(refused.value) == (
        f"{held}: a name holding %, which cmd.exe, the segment command's shell on Windows, would"
        " expand: write the output elsewhere"
    )


def marked(marker: str) -> list[str]:
    """The names of the processes whose command line holds marker."""
    query = (
        "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*"
        + marker
        + "*' -and $_.Name -ne 'powershell.exe' } | ForEach-Object { $_.Name }"
    )
    out = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", query],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=True,
    ).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def test_on_windows_an_output_under_a_percent_is_refused_at_the_jobs_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Before the first pass, the models and the first shot, not at the first segment's writer.
    monkeypatch.setattr(writer_module.sys, "platform", "win32")
    with pytest.raises(MediaError, match="a name holding %"):
        check_command("ffmpeg -i - {out}", Path("/films/100%/film.mkv"))
    check_command("ffmpeg -i - {out}", Path("/films/x/film.mkv"))
    monkeypatch.setattr(writer_module.sys, "platform", "linux")
    check_command("ffmpeg -i - {out}", Path("/films/100%/film.mkv"))


def test_on_windows_cmd_runs_without_autorun_or_delayed_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COMSPEC", "C:\\Windows\\system32\\cmd.exe")
    assert writer_module._cmd('ffmpeg -i - "x.mkv"') == (  # pyright: ignore[reportPrivateUsage]
        '"C:\\Windows\\system32\\cmd.exe" /d /v:off /s /c "ffmpeg -i - "x.mkv""'
    )


@on_windows
def test_on_windows_a_command_writes_its_segment_through_cmd(tmp_path: Path) -> None:
    # A directory whose name holds a space, a quote and a dollar, as the trial's: {out} quoted.
    path = tmp_path / "seedvr2x trial '$ x" / "seg_000001.mkv"
    command = "ffmpeg -nostdin -v error -i - -c copy {out}"
    with CommandWriter(command, path, "yuv420p10le", 64, 48, RATE, BT709) as writer:
        writer.write(np.random.default_rng(0).random((3, 48, 64, 3), dtype=np.float32))
    assert count_frames(path) == 3 and writer.written == 3
    assert writer_module._JOBS == set()  # pyright: ignore[reportPrivateUsage]  # closed


@on_windows
def test_on_windows_an_abort_ends_the_commands_whole_tree(tmp_path: Path) -> None:
    # cmd.exe starts a program that would run a minute, marked by its own argument, then ffmpeg;
    # abort ends the command's job, the shell and what it started, at once, and nothing is left.
    marker = f"seedvr2x-test-{tmp_path.name}"
    interpreter = getattr(sys, "_base_executable", sys.executable)
    sleeper = f'"{interpreter}" -c "import time; time.sleep(60)" {marker}'
    path = tmp_path / "seg_000001.mkv"
    held = CommandWriter(
        f"{sleeper} & ffmpeg -i - -c copy {{out}}", path, "yuv420p10le", 64, 48, RATE, BT709
    )
    deadline = time.monotonic() + 15
    while "python.exe" not in marked(marker):
        assert time.monotonic() < deadline, "the command's sleeper never started"
        time.sleep(0.1)
    held.abort()
    deadline = time.monotonic() + 15
    while marked(marker):
        assert time.monotonic() < deadline, f"left after the abort: {marked(marker)}"
        time.sleep(0.1)
    assert not path.exists()
    assert writer_module._JOBS == set()  # pyright: ignore[reportPrivateUsage]

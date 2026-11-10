"""The segment command on Windows (media/writer.py, CommandWriter): cmd.exe (COMSPEC, Popen's
shell) in a Job Object of its own (ffmpeg.contained), {out} quoted for cmd. The words run
anywhere, Windows faked; the commands themselves natively, with an ffmpeg on PATH."""

import contextlib
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_segment_cmd import TELL, gone, members, started

from media import ffmpeg as ffmpeg_module
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


@pytest.mark.skipif(sys.platform == "win32", reason="Windows faked over a POSIX shell")
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs an ffmpeg on PATH")
def test_on_windows_a_shell_that_could_not_be_contained_is_killed_and_reaped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ffmpeg.contained raising after the shell was started (it ends the process only after some
    # of its failures, OpenProcess's not among them): the Popen handle was left unwaited, the
    # shell alive if suspended. Windows faked, a POSIX shell in its place.
    monkeypatch.setattr(writer_module.sys, "platform", "win32")
    monkeypatch.setattr(writer_module, "CREATE_SUSPENDED", 0)
    monkeypatch.setattr(writer_module, "_cmd", lambda command: ["/bin/sh", "-c", "sleep 60"])

    def uncontained(pid: int) -> int:
        raise MediaError("the shell can't be put in a job: OpenProcess failed")

    monkeypatch.setattr(writer_module, "contained", uncontained)
    # seedvr2x's own Job Object (ffmpeg._in_a_job), which the writer's ffmpeg would ask for.
    monkeypatch.setattr(ffmpeg_module, "_in_a_job", lambda: None)
    shells: list[subprocess.Popen[Any]] = []
    real = subprocess.Popen

    def recording(args: list[str], *more: object, **named: object) -> subprocess.Popen[Any]:
        process = real(args, *more, **named)  # pyright: ignore[reportCallIssue, reportArgumentType]
        if args[:1] == ["/bin/sh"]:
            shells.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", recording)
    path = tmp_path / "seg_000001.mkv"
    with pytest.raises(MediaError, match="can't be put in a job"):
        CommandWriter("sleep 60 {out}", path, "yuv420p10le", 64, 48, RATE, BT709)
    assert len(shells) == 1 and shells[0].returncode is not None  # waited, not left to the GC


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


# The stop's grace on Windows (CommandWriter._stop): Ctrl-Break to the command's console group,
# STOP_GRACE seconds, then its job ended. Here with Windows faked over a POSIX shell: Ctrl-Break
# stood in by SIGTERM to the command's group, the job's end by SIGKILL to it.
WINDOWS_NOTICE = re.compile(
    r"\n\d\d:\d\d:\d\d the segment command was sent Ctrl-Break: it has 5 s to end before its"
    r" job is ended; Ctrl-C again ends it at once\n"
)


class FakedJob:
    """ffmpeg.break_group and end_job over a POSIX process group, recording their calls."""

    def __init__(self, refuse: str | None = None) -> None:
        self.calls: list[tuple[str, int]] = []
        self.refuse = refuse
        self.group = 0  # the shell's, once the writer is faked

    def break_group(self, group: int) -> str | None:
        self.calls.append(("break", group))
        if self.refuse is None:
            os.kill(-group, signal.SIGTERM)
        return self.refuse

    def end_job(self, job: int) -> None:
        self.calls.append(("end", job))
        posix_signal: Any = signal  # POSIX's SIGKILL: the test's own POSIX shell
        with contextlib.suppress(ProcessLookupError):  # ended already, by its trap
            os.kill(-self.group, posix_signal.SIGKILL)


@pytest.fixture
def faked(monkeypatch: pytest.MonkeyPatch) -> Callable[[CommandWriter, FakedJob], None]:
    """Windows faked for a writer started on POSIX: its job a number, the job's calls the fake's."""

    def fake(writer: CommandWriter, job: FakedJob) -> None:
        job.group = writer._shell.pid  # pyright: ignore[reportPrivateUsage]
        writer._job = 7  # pyright: ignore[reportPrivateUsage]
        monkeypatch.setattr(writer_module, "break_group", job.break_group)
        monkeypatch.setattr(writer_module, "end_job", job.end_job)
        monkeypatch.setattr(writer_module, "close_job", lambda job: None)
        monkeypatch.setattr(writer_module.sys, "platform", "win32")

    return fake


posix_shell = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("ffmpeg") is None,
    reason="Windows faked over a POSIX shell, with an ffmpeg on PATH",
)


@posix_shell
def test_on_windows_a_stop_sends_ctrl_break_and_waits_no_longer_than_the_command(
    tmp_path: Path,
    faked: Callable[[CommandWriter, FakedJob], None],
    capfd: pytest.CaptureFixture[str],
) -> None:
    undone = tmp_path / "undone"
    writer, shell, _ = started(
        tmp_path, f"trap 'echo yes > {undone}; exit 143' TERM; {TELL}; wait # {{out}}"
    )
    job = FakedJob()
    faked(writer, job)
    capfd.readouterr()
    start = time.monotonic()
    writer.abort()
    assert time.monotonic() - start < writer_module.STOP_NOTICE_AFTER + 1
    assert undone.read_text() == "yes\n"
    assert job.calls == [("break", shell), ("end", 7)]
    assert gone(shell)
    assert capfd.readouterr().err == ""


@posix_shell
def test_on_windows_a_command_that_ignores_ctrl_break_has_its_job_ended_after_the_grace(
    tmp_path: Path,
    faked: Callable[[CommandWriter, FakedJob], None],
    capfd: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer, shell, child = started(tmp_path, f"trap '' TERM; {TELL}; cat > {{out}}; wait")
    job = FakedJob()
    faked(writer, job)
    capfd.readouterr()
    start = time.monotonic()
    writer.abort()
    elapsed = time.monotonic() - start
    assert 4.5 <= elapsed < 8
    assert job.calls == [("break", shell), ("end", 7)]
    assert gone(shell) and child not in members(shell)
    assert list(writer.path.parent.iterdir()) == []
    assert WINDOWS_NOTICE.fullmatch(capfd.readouterr().err)


@posix_shell
def test_on_windows_a_command_ctrl_break_cant_reach_has_its_job_ended_at_once(
    tmp_path: Path,
    faked: Callable[[CommandWriter, FakedJob], None],
    capfd: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    writer, shell, _ = started(tmp_path, f"trap '' TERM; {TELL}; cat > {{out}}; wait")
    job = FakedJob(refuse="The handle is invalid, WinError 6")
    faked(writer, job)
    capfd.readouterr()
    start = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="cli.media.writer"):
        writer.abort()
    assert time.monotonic() - start < 2
    assert job.calls == [("break", shell), ("end", 7)]
    assert gone(shell)
    assert (
        "the segment command can't be sent Ctrl-Break (The handle is invalid, WinError 6): its"
        " job is ended at once"
    ) in caplog.text
    assert capfd.readouterr().err == ""


@posix_shell
def test_on_windows_a_command_reading_its_pipe_ends_with_ffmpeg_before_any_ctrl_break(
    tmp_path: Path,
    faked: Callable[[CommandWriter, FakedJob], None],
    capfd: pytest.CaptureFixture[str],
) -> None:
    # On Windows Ctrl-Break, then ffmpeg ended at once: a command blocked reading its pipe,
    # which Ctrl-Break doesn't wake there, finds its stdin at its end and ends within the notice,
    # no line said; then its job is ended, whatever it left.
    writer, shell, _ = started(tmp_path, "echo $$ $$ > '{pids}'; exec cat > {out}")
    job = FakedJob()
    faked(writer, job)
    capfd.readouterr()
    start = time.monotonic()
    writer.abort()
    assert time.monotonic() - start < 2
    assert job.calls == [("break", shell), ("end", 7)]
    assert gone(shell)
    assert list(writer.path.parent.iterdir()) == []
    assert capfd.readouterr().err == ""


@posix_shell
def test_on_windows_a_command_whose_shell_has_ended_gets_no_ctrl_break(
    tmp_path: Path,
    faked: Callable[[CommandWriter, FakedJob], None],
    capfd: pytest.CaptureFixture[str],
) -> None:
    # Its shell gone, the group it led has no leader to send Ctrl-Break by: its job is ended at
    # once, what it left running with it. Its child reads stdin through descriptor 3: a shell
    # gives its background jobs /dev/null.
    writer, shell, child = started(
        tmp_path, "exec 3<&0; cat <&3 > {out} & echo $$ $! > '{pids}'; exit 0"
    )
    job = FakedJob()
    faked(writer, job)
    deadline = time.monotonic() + 10
    while writer._shell.poll() is None:  # pyright: ignore[reportPrivateUsage]
        assert time.monotonic() < deadline, "the shell never ended"
        time.sleep(0.01)
    capfd.readouterr()
    writer.abort()
    assert job.calls == [("end", 7)]
    assert gone(shell) and child not in members(shell)
    assert capfd.readouterr().err == ""


@on_windows
def test_on_windows_an_ffmpeg_reading_its_pipe_ends_at_once_on_an_abort(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    # Natively: the command's ffmpeg blocked reading the stream ends once ffmpeg's end closes
    # it, not 5 s later.
    path = tmp_path / "seg_000001.mkv"
    held = CommandWriter(
        "ffmpeg -v error -y -i - -c copy {out}", path, "yuv420p10le", 64, 48, RATE, BT709
    )
    held.write(np.zeros((3, 48, 64, 3), np.float32))
    time.sleep(1)
    capfd.readouterr()
    start = time.monotonic()
    held.abort()
    assert time.monotonic() - start < 2
    assert held._shell.returncode is not None  # pyright: ignore[reportPrivateUsage]
    assert not path.exists() and list(tmp_path.iterdir()) == []
    assert capfd.readouterr().err == ""


# Natively: a command that answers Ctrl-Break (Python's SIGBREAK handler, writing a file) and one
# that ignores it, run by cmd.exe in its job, {out} an argument it never writes.
# Short sleeps: Python's sleep on Windows wakes for SIGINT alone, so a long one would hold
# SIGBREAK's handler to its end.
ANSWERS = (
    "import signal, sys, time;"
    " signal.signal(signal.SIGBREAK,"
    " lambda n, f: (open(sys.argv[1], 'w').write('yes'), sys.exit(3)));"
    " [time.sleep(0.05) for _ in range(1200)]"
)
IGNORES = "import signal, time; signal.signal(signal.SIGBREAK, signal.SIG_IGN); time.sleep(60)"


@on_windows
@pytest.mark.parametrize("answers", [True, False])
def test_on_windows_a_stop_gives_the_command_ctrl_break_and_5_s(
    tmp_path: Path, capfd: pytest.CaptureFixture[str], answers: bool
) -> None:
    marker = f"seedvr2x-test-{tmp_path.name}"
    undone = tmp_path / "undone"
    interpreter = getattr(sys, "_base_executable", sys.executable)
    script = ANSWERS if answers else IGNORES
    program = f'"{interpreter}" -c "{script}" "{undone}" {marker}'
    path = tmp_path / "seg_000001.mkv"
    held = CommandWriter(f"{program} {{out}}", path, "yuv420p10le", 64, 48, RATE, BT709)
    deadline = time.monotonic() + 15
    while "python.exe" not in marked(marker):
        assert time.monotonic() < deadline, "the command's program never started"
        time.sleep(0.1)
    capfd.readouterr()
    start = time.monotonic()
    held.abort()
    elapsed = time.monotonic() - start
    deadline = time.monotonic() + 15
    while marked(marker):
        assert time.monotonic() < deadline, f"left after the abort: {marked(marker)}"
        time.sleep(0.1)
    assert not path.exists()
    if answers:
        assert elapsed < 4 and undone.read_text() == "yes"
        assert capfd.readouterr().err == ""
    else:
        assert 4.5 <= elapsed < 8 and not undone.exists()
        assert WINDOWS_NOTICE.fullmatch(capfd.readouterr().err)

"""Every child seedvr2x starts in a process group of its own is out of a console's Ctrl-C on
Windows too (media/ffmpeg.py, NEW_GROUP): the sites that pass `process_group=0` pass
CreateProcess's CREATE_NEW_PROCESS_GROUP as well, so that a first Ctrl-C reaches seedvr2x alone
and the unit in progress finishes (stop.py). The scan holds every site on any platform; the
Windows case sends a real Ctrl-C to a console of its own (on_windows)."""

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from test_ends_with_us import sources

from media.ffmpeg import NEW_GROUP

# CreateProcess's CREATE_NEW_PROCESS_GROUP (winbase.h).
CREATE_NEW_PROCESS_GROUP = 0x00000200


def grouped_starts() -> list[tuple[str, int, ast.expr | None]]:
    """Each call of the CLI's files given `process_group=0`: its file, its line and the
    expression given as `creationflags` (None without one)."""
    found: list[tuple[str, int, ast.expr | None]] = []
    for path in sources():
        for node in ast.walk(ast.parse(path.read_text(), str(path))):
            if not isinstance(node, ast.Call):
                continue
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            group = keywords.get("process_group")
            if isinstance(group, ast.Constant) and group.value == 0:
                found.append((path.name, node.lineno, keywords.get("creationflags")))
    return found


def names(expression: ast.expr) -> set[str]:
    return {node.id for node in ast.walk(expression) if isinstance(node, ast.Name)}


def test_every_child_in_a_group_of_its_own_is_one_on_windows_too() -> None:
    starts = grouped_starts()
    # The readers, decoders, writers, checksums' and the assembly's ffmpeg and ffprobe, and the
    # segment command's shell: a new one is counted on purpose.
    assert len(starts) == 12, starts
    missing = [(name, line) for name, line, flags in starts if flags is None]
    assert not missing, f"process_group=0 without creationflags: {missing}"
    other = [
        (name, line) for name, line, flags in starts if flags and "NEW_GROUP" not in names(flags)
    ]
    assert not other, f"creationflags without NEW_GROUP: {other}"


def test_the_flag_is_windows_own_and_nothing_elsewhere() -> None:
    assert NEW_GROUP == (CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0)
    # What Popen takes on POSIX: creationflags 0 alone, beside the group.
    done = subprocess.run(
        [sys.executable, "-c", "pass"],
        process_group=0 if sys.platform != "win32" else None,
        creationflags=NEW_GROUP,
        check=False,
    )
    assert done.returncode == 0


# A console of its own (hidden), so that the Ctrl-C sent to it reaches nothing of pytest's: the
# parent starts a child with NEW_GROUP, then sends CTRL_C_EVENT to every process of the console
# (GenerateConsoleCtrlEvent, group 0), as a keyboard's Ctrl-C does, and reports whether its own
# handler ran and whether the child still runs a second later.
PARENT = textwrap.dedent(
    """
    import ctypes, signal, subprocess, sys, time
    # Ctrl-C as a user's console has it, whatever this process inherited (a process started in a
    # group of its own has it disabled, its children too).
    ctypes.WinDLL("kernel32").SetConsoleCtrlHandler(None, False)
    got = []
    signal.signal(signal.SIGINT, lambda number, frame: got.append(number))
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], creationflags=int(sys.argv[1])
    )
    time.sleep(1.0)
    ok = ctypes.WinDLL("kernel32", use_last_error=True).GenerateConsoleCtrlEvent(0, 0)
    deadline = time.monotonic() + 5
    while not got and time.monotonic() < deadline:
        time.sleep(0.05)
    time.sleep(1.0)
    print(f"sent={ok} handled={bool(got)} child={child.poll()}", flush=True)
    child.kill()
    child.wait()
    """
)


@pytest.mark.skipif(sys.platform != "win32", reason="a console's Ctrl-C is Windows' own here")
@pytest.mark.parametrize("flags", ["NEW_GROUP", "none"])
def test_on_windows_a_consoles_ctrl_c_reaches_seedvr2x_alone(tmp_path: Path, flags: str) -> None:
    assert sys.platform == "win32"  # for the type checker: the skip above says it
    script = tmp_path / "parent.py"
    script.write_text(PARENT)
    out = tmp_path / "out.txt"
    info = subprocess.STARTUPINFO()
    info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    info.wShowWindow = 0  # SW_HIDE: no window on the desktop
    with out.open("w") as stream:
        subprocess.run(
            [sys.executable, "-I", str(script), str(NEW_GROUP if flags == "NEW_GROUP" else 0)],
            stdout=stream,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            startupinfo=info,
            timeout=60,
            check=True,
        )
    said = out.read_text().strip()
    assert "sent=1 handled=True" in said, said
    if flags == "NEW_GROUP":
        assert said.endswith("child=None"), said  # still running: the Ctrl-C didn't reach it
    else:
        # The control: in the console's own group the child gets it too (Python's default
        # handler: KeyboardInterrupt, its status 0xC000013A).
        assert not said.endswith("child=None"), said

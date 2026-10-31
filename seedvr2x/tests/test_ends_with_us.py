"""Every ffmpeg and ffprobe seedvr2x starts ends with seedvr2x (media/ffmpeg.py, ends_with_us;
DESIGN.md, Output, Assembly, decided on 2026-10-29): killed, seedvr2x leaves none of them
running, the user's --segment-cmd aside. The assembly's own are held by tests/test_assembled.py.

Three ways, each on the CPU. A static scan of the source: every process it starts passes the
hook but the segment command's shell. A child Python process that starts one at each kind of
site, with a fake ffmpeg and ffprobe on PATH that only wait (a site that can't be killed cheaply
with the real one: a probe lasts milliseconds), SIGKILLed: none of the processes it started is
left, where the control, the hook replaced by none in that site's module, leaves them. And the
real ffmpeg of a writer and of a reader, as the assembly's test does (skipped without a build
seedvr2x accepts). Linux's /proc tells the processes left."""

import ast
import contextlib
import ctypes
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from seedvr2x.media import ffmpeg
from seedvr2x.media.ffmpeg import MediaError

SOURCE = Path(__file__).parent.parent / "src" / "seedvr2x"


def _usable() -> bool:
    try:
        ffmpeg.check()
    except MediaError:
        return False
    return True


needs_ffmpeg = pytest.mark.skipif(
    not _usable(), reason="needs ffmpeg 7.1 or later with zscale, idet and ffv1"
)

# Every call that starts a process, in the standard library: seedvr2x uses subprocess alone.
SPAWNS = {"Popen", "run", "call", "check_call", "check_output"}
# subprocess's other starters, which the scan refuses to import by name (a call on the module is
# refused whatever its name: see scan).
OTHER_SUBPROCESS = {"getoutput", "getstatusoutput"}
# os's, by family so that a name added later is caught too: spawn* (spawnl ... spawnvpe), exec*
# (execl ... execvpe), posix_spawn and posix_spawnp, fork and forkpty, and system and popen.
OS_FAMILIES = ("spawn", "exec", "posix_spawn", "fork")
OS_NAMES = {"system", "popen"}


def starts_a_process(name: str) -> bool:
    """Whether os.<name> starts a process or replaces the one running."""
    return name in OS_NAMES or name.startswith(OS_FAMILIES)


class Scan:
    """What `scan` found in one source: the starts that pass the hook, those that don't, and the
    sites that break a rule (each as 'file:line: why')."""

    def __init__(self) -> None:
        self.hooked: list[str] = []
        self.unhooked: list[str] = []
        self.violations: list[str] = []


def scan(tree: ast.AST, name: str) -> Scan:
    """The scan of the source `tree`, read from the file `name`, for the process it starts.
    subprocess is imported plainly, and used as `subprocess.<starter>(...)`, each with the
    hook; os starts nothing. An alias (`import os as x`) or a name imported from either would
    hide a start from this scan, so each is refused."""
    found = Scan()
    for node in ast.walk(tree):
        where = f"{name}:{getattr(node, 'lineno', 0)}"
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"subprocess", "os"} and alias.asname is not None:
                    found.violations.append(f"{where}: import {alias.name} as {alias.asname}")
        elif isinstance(node, ast.ImportFrom) and node.module in {"subprocess", "os"}:
            for alias in node.names:
                if alias.name == "*" or (
                    alias.name in SPAWNS | OTHER_SUBPROCESS
                    if node.module == "subprocess"
                    else starts_a_process(alias.name)
                ):
                    found.violations.append(f"{where}: from {node.module} import {alias.name}")
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "os"
            and starts_a_process(node.attr)
        ):
            found.violations.append(f"{where}: os.{node.attr}")
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        owner = node.func.value
        if not (isinstance(owner, ast.Name) and owner.id == "subprocess"):
            continue
        if node.func.attr not in SPAWNS:
            found.violations.append(f"{where}: subprocess.{node.func.attr}")
            continue
        hooked = any(
            keyword.arg == "preexec_fn"
            and isinstance(keyword.value, ast.Call)
            and isinstance(keyword.value.func, ast.Name)
            and keyword.value.func.id == "ends_with_us"
            for keyword in node.keywords
        )
        (found.hooked if hooked else found.unhooked).append(where)
    return found


def sources() -> list[Path]:
    """seedvr2x's own files: the vendored model code and TransNetV2 start no process."""
    return [
        path
        for path in sorted(SOURCE.rglob("*.py"))
        if not {"vendor", "transnetv2"} & set(path.relative_to(SOURCE).parts)
    ]


def test_every_process_started_passes_the_hook_but_the_segment_commands_shell() -> None:
    # DESIGN.md, Output, Assembly (2026-10-29): "every ffmpeg and ffprobe seedvr2x starts ends
    # with it the same way, the user's --segment-cmd aside". A scan of the source, so that a
    # site added later without the hook fails here.
    hooked: list[str] = []
    without: list[str] = []
    violations: list[str] = []
    for path in sources():
        found = scan(ast.parse(path.read_text()), str(path.relative_to(SOURCE)))
        hooked += found.hooked
        without += found.unhooked
        violations += found.violations
    assert not violations, violations
    # 16 sites pass the hook today; a new one is counted here, so that it is added on purpose.
    assert len(hooked) == 16, hooked
    assert len(without) == 1 and without[0].startswith("media/writer.py:"), without
    # That one is the shell the user's command runs in.
    shell = int(without[0].split(":")[1])
    lines = (SOURCE / "media" / "writer.py").read_text().splitlines()
    assert '"/bin/sh"' in " ".join(lines[shell - 1 : shell + 3]), lines[shell - 1 : shell + 3]


# One snippet per rule of the scan: each is flagged (design, baton 3, 2026-10-30: the scan
# catches an aliased import and os's starters).
FLAGGED = {
    "subprocess aliased": "import subprocess as sp\nsp.run(['x'])",
    "os aliased": "import os as operating\noperating.execv('/bin/x', ['x'])",
    "from subprocess import run": "from subprocess import run\nrun(['x'])",
    "from subprocess import Popen as P": "from subprocess import Popen as P\nP(['x'])",
    "from subprocess import getoutput": "from subprocess import getoutput",
    "from subprocess import *": "from subprocess import *",
    "from os import spawnvp": "from os import spawnvp",
    "from os import execlp": "from os import execlp",
    "from os import posix_spawnp": "from os import posix_spawnp",
    "from os import system": "from os import system",
    "from os import a future exec name": "from os import execvz",
    "os.spawnlpe": "import os\nos.spawnlpe(os.P_WAIT, 'x', 'x', {})",
    "os.spawnv": "import os\nos.spawnv(os.P_WAIT, '/bin/x', ['x'])",
    "os.execve": "import os\nos.execve('/bin/x', ['x'], {})",
    "os.execvpe": "import os\nos.execvpe('x', ['x'], {})",
    "os.posix_spawn": "import os\nos.posix_spawn('/bin/x', ['x'], {})",
    "os.posix_spawnp": "import os\nos.posix_spawnp('x', ['x'], {})",
    "os.system": "import os\nos.system('x')",
    "os.popen": "import os\nos.popen('x')",
    "os.fork": "import os\nos.fork()",
    "os.forkpty": "import os\nos.forkpty()",
    "subprocess.getoutput": "import subprocess\nsubprocess.getoutput('x')",
}


@pytest.mark.parametrize("rule", FLAGGED)
def test_the_scan_flags_what_could_hide_a_start(rule: str) -> None:
    # Each rule shown on a small source: without it, the real scan above would pass a site
    # that starts a process without the hook.
    found = scan(ast.parse(FLAGGED[rule]), "synthetic.py")
    assert found.violations, rule
    assert all(violation.startswith("synthetic.py:") for violation in found.violations)


def test_the_scan_counts_the_hooked_and_the_unhooked_starts_and_leaves_plain_uses_alone() -> None:
    source = textwrap.dedent(
        """
        import os
        import subprocess
        from subprocess import PIPE
        from os import getpid
        subprocess.Popen(["x"], preexec_fn=ends_with_us())
        subprocess.run(["x"], preexec_fn=other())
        subprocess.check_output(["x"])
        os.getpid()
        os._exit(1)
        """
    )
    found = scan(ast.parse(source), "synthetic.py")
    assert found.violations == []
    assert found.hooked == ["synthetic.py:6"]
    assert found.unhooked == ["synthetic.py:7", "synthetic.py:8"]


class Recorder:
    """A stand-in for the C library that notes every attribute looked up on it."""

    def __init__(self) -> None:
        self.looked_up: list[str] = []
        self.calls: list[tuple[int, int]] = []

    def __getattr__(self, name: str) -> object:
        self.looked_up.append(name)
        return self

    def __call__(self, option: int, argument: int) -> int:
        self.calls.append((option, argument))
        return 0


def test_prctl_is_resolved_in_the_parent_before_any_fork(monkeypatch: pytest.MonkeyPatch) -> None:
    # Design, baton 3 (2026-10-30): ctypes' lazy attribute lookup allocates, which the child of
    # a multi-threaded process must not do between its fork and its exec. ends_with_us looks
    # prctl up, once, itself; the hook it returns calls the function it holds and looks nothing
    # up on the library.
    recorder = Recorder()
    monkeypatch.setattr(ffmpeg, "_libc", lambda: recorder)
    # Run in the parent, the hook would see its own parent (the shell) as the one that ended.
    monkeypatch.setattr(os, "getppid", os.getpid)
    ffmpeg._prctl.cache_clear()  # pyright: ignore[reportPrivateUsage]
    try:
        prepare = ffmpeg.ends_with_us()
        assert recorder.looked_up == ["prctl"]
        prepare()  # the child's part, run here
        assert recorder.looked_up == ["prctl"]
        assert recorder.calls == [(ffmpeg.PR_SET_PDEATHSIG, int(signal.SIGKILL))]
    finally:
        ffmpeg._prctl.cache_clear()  # pyright: ignore[reportPrivateUsage]


def test_the_hook_holds_the_typed_c_function_not_the_library() -> None:
    # The real library: the function cached, typed, and the only ctypes object the hook closes
    # over (a CDLL there would look its attributes up again in the child).
    ffmpeg._prctl.cache_clear()  # pyright: ignore[reportPrivateUsage]
    prepare = ffmpeg.ends_with_us()
    assert ffmpeg._prctl.cache_info().currsize == 1  # pyright: ignore[reportPrivateUsage]
    function = ffmpeg._prctl()  # pyright: ignore[reportPrivateUsage]
    assert function.argtypes == (ctypes.c_int, ctypes.c_ulong)  # pyright: ignore[reportFunctionMemberAccess]
    assert function.restype is ctypes.c_int  # pyright: ignore[reportFunctionMemberAccess]
    cells = [cell.cell_contents for cell in prepare.__closure__ or ()]
    assert function in cells
    assert not any(isinstance(cell, ctypes.CDLL) for cell in cells), cells


def alive(pid: int) -> bool:
    """Whether the process `pid` runs: a zombie, ended and not yet reaped, does not."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def children_of(parent: int, name: str) -> list[int]:
    """The processes whose parent is `parent` and whose command is `name` (Linux's /proc)."""
    found: list[int] = []
    for entry in Path("/proc").glob("[0-9]*/stat"):
        try:
            # After the command's name, in brackets: state, then the parent's pid.
            fields = entry.read_text().rsplit(")", 1)[1].split()
            if int(fields[1]) != parent:
                continue
            command = (entry.parent / "cmdline").read_bytes().split(b"\0")[0]
        except (OSError, IndexError, ValueError):  # gone meanwhile
            continue
        if Path(command.decode(errors="replace")).name == name:
            found.append(int(entry.parent.name))
    return found


def waited(condition: object, seconds: float = 15.0) -> bool:
    """Whether the callable `condition` holds within `seconds`."""
    assert callable(condition)
    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)
    return True


def control_patch(module: str, held: bool) -> str:
    """The code that, in the child, replaces the hook by none in `module` (the control)."""
    if held:
        return ""
    return f"import seedvr2x.media.{module} as module\nmodule.ends_with_us = lambda: None\n"


# A fake ffmpeg and ffprobe: each says its pid and waits. exec keeps the pid, and the parent
# death signal is kept across it.
FAKE = """#!/bin/sh
echo $$ >> "$FAKE_PIDS"
exec sleep 1000
"""

# What the child runs at each site, with the module whose hook the control replaces, and how many
# processes the site starts. Run in the child's main thread, which the fake keeps waiting.
SITES = {
    "startup check": ("ffmpeg", "from seedvr2x.media import ffmpeg\nffmpeg.check()", 1),
    "probe": (
        "probe",
        "from seedvr2x.media.probe import probe\nprobe(Path('f.mkv'), first_frame=False)",
        1,
    ),
    "fingerprint": (
        "fingerprint",
        "from seedvr2x.media.fingerprint import fingerprint\nfingerprint()",
        1,
    ),
    "count packets": (
        "writer",
        "from seedvr2x.media.writer import count_packets\ncount_packets(Path('f.mkv'))",
        1,
    ),
    "count frames": (
        "writer",
        "from seedvr2x.media.writer import count_frames\ncount_frames(Path('f.mkv'))",
        1,
    ),
    "checksums": (
        "checksums",
        "from seedvr2x.media.checksums import check_frames\n"
        "check_frames(['-i', 'f.mkv'], 'gbrp16le', 4, 4, [1])",
        1,
    ),
    "packets": (
        "scan",
        "from seedvr2x.media.scan import packets\npackets(Path('f.mkv'))",
        1,
    ),
    # The decode and the packets scanned beside it: two processes, the second from a thread of
    # the first pass that waits for its child.
    "first pass": (
        "scan",
        "from seedvr2x.media.scan import scan\nscan(Path('f.mkv'))",
        2,
    ),
    "decoder": (
        "decode",
        "import time\nfrom seedvr2x.media.conversion import Conversion\n"
        "from seedvr2x.media.decode import Decoder\n"
        "Decoder(['-i', 'f.mkv'], Conversion('yuv420p', '709', 'limited', 'left'), 4, 4)\n"
        "time.sleep(1000)",
        1,
    ),
}


@pytest.fixture
def fakes(tmp_path: Path) -> dict[str, str]:
    """The environment of a child that finds a fake ffmpeg and ffprobe first on PATH."""
    bin_directory = tmp_path / "bin"
    bin_directory.mkdir()
    for name in ("ffmpeg", "ffprobe"):
        (bin_directory / name).write_text(FAKE)
        (bin_directory / name).chmod(0o755)
    (tmp_path / "f.mkv").write_bytes(b"")
    return {
        **os.environ,
        "PATH": f"{bin_directory}{os.pathsep}{os.environ['PATH']}",
        "FAKE_PIDS": str(tmp_path / "pids"),
        "HF_HUB_OFFLINE": "1",
    }


def pids_of(path: Path) -> list[int]:
    return [int(word) for word in path.read_text().split()] if path.exists() else []


@pytest.mark.parametrize("held", [True, False], ids=["hooked", "control_without"])
@pytest.mark.parametrize("site", SITES)
def test_a_killed_run_leaves_none_of_the_ffmpeg_and_ffprobe_it_started(
    tmp_path: Path, fakes: dict[str, str], site: str, held: bool
) -> None:
    # A child Python process starts the process at this kind of site, and is SIGKILLed alone
    # (not its group: what it started has groups of its own, where the terminal's Ctrl-C and
    # a killpg don't reach): with the hook, what it started is gone within seconds; the control,
    # the hook replaced by none in the site's module, shows the test would see one left.
    module, code, wanted = SITES[site]
    script = "from pathlib import Path\n" + control_patch(module, held) + code + "\n"
    child = subprocess.Popen(
        [sys.executable, "-c", script], env=fakes, cwd=tmp_path, start_new_session=True
    )
    pids: list[int] = []
    try:
        assert waited(lambda: len(pids_of(tmp_path / "pids")) >= wanted), pids_of(tmp_path / "pids")
        pids = pids_of(tmp_path / "pids")
        assert len(pids) == wanted and all(alive(pid) for pid in pids), pids
        child.kill()
        child.wait(timeout=30)
        if held:
            assert waited(lambda: not any(alive(pid) for pid in pids)), [
                pid for pid in pids if alive(pid)
            ]
        else:
            time.sleep(1.5)
            assert all(alive(pid) for pid in pids), pids
    finally:
        for pid in pids:
            # It may end between the look and the kill: the test's own finding, not this one's.
            if alive(pid):
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)
        if child.poll() is None:
            child.kill()
            child.wait()


# The real ffmpeg of a reader and of the writers.

SOURCE_CLIP = [
    *("ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "lavfi"),
    *("-i", "testsrc2=size=64x16:rate=25:duration=12", "-c:v", "ffv1", "-pix_fmt", "yuv420p"),
    "src.mkv",
]

REAL = {
    # 300 frames, far more than a pipe holds: ffmpeg waits for the reader to take them.
    "reader": (
        "reader",
        """
        from seedvr2x.media.conversion import Conversion
        from seedvr2x.media.reader import Reader
        from seedvr2x.media.scan import scan

        index = scan(Path("src.mkv")).index
        assert index is not None
        reader = Reader(
            Path("src.mkv"), Conversion("yuv420p", "709", "limited", "left"), 64, 16, index
        )
        reader.read(1)
        """,
        "ffmpeg",
    ),
    "ffv1 writer": (
        "writer",
        """
        import numpy as np
        from seedvr2x.media.writer import FFV1Writer, Tags

        writer = FFV1Writer(Path("s.mkv"), "gbrp16le", 32, 16, Fraction(24), Tags("bt709", "bt709"))
        writer.write(np.zeros((1, 16, 32, 3), np.float32))
        """,
        "ffmpeg",
    ),
}


@needs_ffmpeg
@pytest.mark.parametrize("site", REAL)
def test_a_killed_run_leaves_none_of_its_real_ffmpeg(tmp_path: Path, site: str) -> None:
    # As the assembly's test (tests/test_assembled.py): the child at work, SIGKILLed alone. No
    # control here: these ffmpeg end on their pipes whatever the hook does (the reader's at its
    # next write, the writer's at its input's end, DESIGN.md, Output, Assembly), where the join
    # survives, writing a file. The control is the fake ffmpeg's above, at every kind of site;
    # this is the same with the real one.
    _, code, name = REAL[site]
    subprocess.run(SOURCE_CLIP, cwd=tmp_path, check=True)
    script = (
        "import time\nfrom fractions import Fraction\nfrom pathlib import Path\n"
        + textwrap.dedent(code)
        + "\nprint('ready', flush=True)\ntime.sleep(1000)\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
        env={**os.environ, "HF_HOME": str(tmp_path / "hf"), "HF_HUB_OFFLINE": "1"},
    )
    assert child.stdout is not None
    left: list[int] = []
    try:
        assert child.stdout.readline() == "ready\n"
        left = children_of(child.pid, name)
        assert len(left) == 1 and alive(left[0]), left
        child.kill()
        child.wait(timeout=30)
        assert waited(lambda: not alive(left[0])), left
    finally:
        for pid in left:
            # It may end between the look and the kill: the test's own finding, not this one's.
            if alive(pid):
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)
        child.stdout.close()
        if child.poll() is None:
            child.kill()
            child.wait()


@needs_ffmpeg
def test_the_segment_commands_shell_is_not_hooked(tmp_path: Path) -> None:
    # DESIGN.md, Output, Assembly (2026-10-29): the user's --segment-cmd "ends at its input's end".
    # A command that never reads its input, killed with seedvr2x's ffmpeg, goes on: the shell has
    # no hook (the static scan above holds it for the source; this holds it for the behaviour).
    script = (
        "import time\nfrom fractions import Fraction\nfrom pathlib import Path\n"
        "import numpy as np\n"
        "from seedvr2x.media.writer import CommandWriter, Tags\n"
        "writer = CommandWriter('exec sleep 1000 # {out}', Path('s.mkv'), 'yuv420p10le', 64, 48,"
        " Fraction(24), Tags('bt709', 'bt709'))\n"
        "print('ready', flush=True)\ntime.sleep(1000)\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
        env={**os.environ, "HF_HOME": str(tmp_path / "hf"), "HF_HUB_OFFLINE": "1"},
    )
    assert child.stdout is not None
    shells: list[int] = []
    feeders: list[int] = []
    try:
        assert child.stdout.readline() == "ready\n"
        # The shell (/bin/sh) was replaced by sleep (exec); ffmpeg is the writer's.
        assert waited(lambda: bool(children_of(child.pid, "sleep")))
        shells = children_of(child.pid, "sleep")
        feeders = children_of(child.pid, "ffmpeg")
        assert len(shells) == 1 and len(feeders) == 1
        child.kill()
        child.wait(timeout=30)
        assert waited(lambda: not alive(feeders[0]))  # ffmpeg, hooked
        time.sleep(1.5)
        assert alive(shells[0])  # the command, not
    finally:
        for pid in shells + feeders:
            # It may end between the look and the kill: the test's own finding, not this one's.
            if alive(pid):
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)
        child.stdout.close()
        if child.poll() is None:
            child.kill()
            child.wait()

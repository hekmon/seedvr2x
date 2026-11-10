"""The CLI's tests that need POSIX (not named posix.py: the builtin module of that name wins the
import), one skip marker per reason, so that a native Windows run
counts what it skips by reason (`pytest -rs` groups the skips by their reason's text). Every
marker is false on Linux: nothing is skipped there, and the collected ids are the same.

A test that can run on Windows with a cheap change that keeps its assertion is made portable
instead; one whose subject is a POSIX call, program or file system is marked here."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

WINDOWS = sys.platform == "win32"

SHELL_FAKE = pytest.mark.skipif(
    WINDOWS,
    reason="a fake program written as a POSIX shell script (#!/bin/sh), which Windows doesn't run"
    " by its bare name",
)

SIGNALS = pytest.mark.skipif(
    WINDOWS,
    reason="POSIX signals and process groups (killpg, SIGKILL, SIGUSR1, SIGALRM, the signal wakeup"
    " pipe, a signal sent to the test's own process)",
)

PROC = pytest.mark.skipif(WINDOWS, reason="Linux's own interfaces (/proc, prctl)")

PROGRAM = pytest.mark.skipif(
    WINDOWS,
    reason="a POSIX program the test runs (sh, cat, sleep, true, false), or a shell's script",
)

PERMISSIONS = pytest.mark.skipif(
    WINDOWS, reason="POSIX file permissions (a mode that refuses its owner, root's exemption)"
)

FLOCK = pytest.mark.skipif(WINDOWS, reason="POSIX's fcntl.flock, which Windows doesn't have")

TZSET = pytest.mark.skipif(WINDOWS, reason="time.tzset, which Windows doesn't have")

POSIX_CALLS = pytest.mark.skipif(
    WINDOWS,
    reason="POSIX's resource limits and file calls (resource, posix_fallocate, pread, st_blocks,"
    " statvfs)",
)


def _can_symlink() -> bool:
    """Whether this process may make a symbolic link: Windows asks for a privilege (Developer
    Mode, or an administrator), which it may not have; Linux always may."""
    if not WINDOWS:
        return True
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory, "target")
        target.write_text("")
        try:
            os.symlink(target, Path(directory, "link"))
        except OSError:
            return False
    return True


SYMLINKS = pytest.mark.skipif(
    not _can_symlink(),
    reason="symbolic links, which Windows makes only with a privilege this process lacks",
)

COMPILE = pytest.mark.skipif(
    WINDOWS,
    reason="--compile, which the command refuses on Windows (PyTorch's Windows build has no"
    " Triton: main.py)",
)

NAME = pytest.mark.skipif(
    WINDOWS,
    reason="a file name Windows refuses (a colon, a question mark, a line break, bytes no encoding"
    " holds) or a POSIX path's form",
)

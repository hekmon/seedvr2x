"""The CLI's tests: the library's autouse fixtures (core/tests/library_fixtures.py, on this
suite's pythonpath), and the one that is the CLI's own.

A one-file output measures the rate its disk writes at (timing.disk_rate), 256 MiB written and
synced: for every test, a mebibyte, so that the suite's many one-file runs don't write gigabytes.
tests/test_timing.py holds the probe itself.

ffmpeg's listings and the conversions' fingerprint, which every run of the command line asks at
its start, are asked once per build (below): tests/test_ffmpeg_check.py, test_fingerprint.py and
test_ends_with_us.py, whose subject they are, ask them every time."""

import hashlib
import os
import shutil
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

import pytest
from library_fixtures import no_driver_asked, one_to_one_clock  # noqa: F401

import timing
from media import ffmpeg, fingerprint


@pytest.fixture(autouse=True)
def small_disk_probe(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(timing, "PROBE", 1024**2)
    monkeypatch.setattr(timing, "BLOCK", 256 * 1024)
    yield


# ---- ffmpeg's listings and the conversions' fingerprint, asked once per build -------------------
#
# A run of the command line checks the build (media/ffmpeg.py's check: six or eight ffmpeg
# processes, its listings) and takes the conversions' fingerprint (media/fingerprint.py: one
# ffmpeg of 19 conversions) at its start. Counted on 2026-11-07, tests/test_cli_run.py alone
# made 814 checks and 834 fingerprints, 177 s of its 459, about 6,300 of its 8,898 ffmpeg
# processes. Here an unchanged build is asked each listing and the fingerprint once: what
# ffmpeg prints is kept by the program PATH finds (its path, inode, size and times, and its
# bytes where it is a script of under a mebibyte: the fake builds of the tests), the
# environment its loader reads, and the options asked; the fingerprint by the same program
# and the conversions' own description (fingerprint._conversions, WIDTH, HEIGHT), so that a
# conversion changed is asked again. What the check makes of the listings, and its refusals,
# run at every call. A failure is never kept. Off where the listings or the fingerprint are
# the subject (MEMO_OFF); a process of its own (the command started by a test) loads no
# conftest and asks ffmpeg for real.
MEMO_OFF = {"test_ffmpeg_check.py", "test_fingerprint.py", "test_ends_with_us.py"}
SCRIPT_BYTES = 1024**2


class _Memo:
    on = True
    listings: ClassVar[dict[tuple[object, ...], str]] = {}
    fingerprints: ClassVar[dict[tuple[object, ...], str]] = {}
    saved: ClassVar[Counter[str]] = Counter()


def _program(name: str) -> tuple[object, ...] | None:
    """What identifies the program PATH finds for `name`, None if none."""
    path = shutil.which(name)
    if path is None:
        return None
    found = Path(path)
    stat = found.stat()
    script = hashlib.sha256(found.read_bytes()).hexdigest() if stat.st_size < SCRIPT_BYTES else None
    loader = tuple(os.environ.get(each) for each in ("LD_LIBRARY_PATH", "LD_PRELOAD"))
    return (
        str(found.resolve()),
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
        script,
        loader,
    )


_listing = ffmpeg._run  # pyright: ignore[reportPrivateUsage]
_fingerprint = fingerprint.fingerprint


def _remembered_listing(*options: str) -> str:
    program = _program("ffmpeg") if _Memo.on else None
    if program is None:
        return _listing(*options)
    key = (program, options)
    if key in _Memo.listings:
        _Memo.saved["listings"] += 1
    else:
        _Memo.listings[key] = _listing(*options)
    return _Memo.listings[key]


def _remembered_fingerprint() -> str:
    program = _program("ffmpeg") if _Memo.on else None
    if program is None:
        return _fingerprint()
    conversions = fingerprint._conversions()  # pyright: ignore[reportPrivateUsage]
    key = (program, repr(conversions), fingerprint.WIDTH, fingerprint.HEIGHT)
    if key in _Memo.fingerprints:
        _Memo.saved["fingerprints"] += 1
    else:
        _Memo.fingerprints[key] = _fingerprint()
    return _Memo.fingerprints[key]


# At import, before the test modules: their skip marks ask for the check while collected.
ffmpeg._run = _remembered_listing  # pyright: ignore[reportPrivateUsage]
fingerprint.fingerprint = _remembered_fingerprint


@pytest.fixture(autouse=True)
def ffmpeg_memo(request: pytest.FixtureRequest) -> Iterator[None]:
    """The memo above, off for the modules of MEMO_OFF."""
    _Memo.on = request.path.name not in MEMO_OFF
    yield
    _Memo.on = True


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    terminalreporter.write_line(
        f"ffmpeg not started again for the same build: {_Memo.saved['listings']} listings,"
        f" {_Memo.saved['fingerprints']} fingerprints (tests/conftest.py)"
    )

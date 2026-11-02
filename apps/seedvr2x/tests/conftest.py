"""The CLI's tests: the library's autouse fixtures (core/tests/library_fixtures.py, on this
suite's pythonpath), and the one that is the CLI's own.

A one-file output measures the rate its disk writes at (timing.disk_rate), 256 MiB written and
synced: for every test, a mebibyte, so that the suite's many one-file runs don't write gigabytes.
tests/test_timing.py holds the probe itself."""

from collections.abc import Iterator

import pytest
from library_fixtures import no_driver_asked, one_to_one_clock  # noqa: F401

import timing


@pytest.fixture(autouse=True)
def small_disk_probe(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(timing, "PROBE", 1024**2)
    monkeypatch.setattr(timing, "BLOCK", 256 * 1024)
    yield

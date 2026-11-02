"""The level of the lines a test captures, for the library's loggers and the CLI's alike."""

import pytest


def levels(caplog: pytest.LogCaptureFixture, level: int) -> None:
    """Set `level` on both loggers the lines reach caplog by: "seedvr2x", the library's modules
    (seedvr2x.model, seedvr2x.tiling, ...), and "cli", the CLI's (cli.main, cli.run,
    cli.media.assembly, ...). The CLI's loggers sit beside the library's, not under it
    (DESIGN.md, Repository and apps), so a test of a run that names only "seedvr2x" would capture
    the library's lines alone. The tests of library modules alone keep their own call."""
    caplog.set_level(level, logger="seedvr2x")
    caplog.set_level(level, logger="cli")

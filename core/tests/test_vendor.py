"""The vendored model code: marked, headers kept, no numz runtime (tools/vendor.py check)."""

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
NUMZ = PROJECT.parent / "upstream" / "seedvr2-numz"


@pytest.mark.skipif(not (NUMZ / ".git").exists(), reason="numz submodule not checked out")
def test_vendor_check() -> None:
    result = subprocess.run(
        [sys.executable, str(PROJECT / "tools" / "vendor.py"), "check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

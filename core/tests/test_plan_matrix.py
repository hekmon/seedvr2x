"""tools/plan_matrix.py kept running: its --quick part of the matrix for each DiT format, with no
crash and no torch imported, and its diff on two outputs made to differ. The full matrix is run
by hand, before and after a change to the plan (the tool's docstring)."""

import importlib.util
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

PROJECT = Path(__file__).resolve().parents[1]
TOOL = PROJECT / "tools" / "plan_matrix.py"
DITS = ("fp16", "int8", "gguf7", "gguf7sharp")
# The tool run as its script runs, then whether torch was imported, said on stderr.
DRIVER = """
import runpy, sys
sys.argv = sys.argv[1:]
code = 0
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
except SystemExit as stop:
    code = stop.code
print("torch imported:", "torch" in sys.modules, file=sys.stderr)
sys.exit(code)
"""


@pytest.fixture(scope="module")
def quick() -> Callable[[str], subprocess.CompletedProcess[str]]:
    """--quick's run for a DiT format, once per format: in a process of its own, from core/, so
    that the seedvr2x it plans with is this tree's and no torch another test imported is there."""
    runs: dict[str, subprocess.CompletedProcess[str]] = {}

    def run(dit: str) -> subprocess.CompletedProcess[str]:
        if dit not in runs:
            runs[dit] = subprocess.run(
                [sys.executable, "-c", DRIVER, str(TOOL), "run", "--quick", "--dit", dit],
                cwd=PROJECT,
                capture_output=True,
                text=True,
                timeout=120,
            )
        return runs[dit]

    return run


def tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("plan_matrix_under_test", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dit", DITS)
def test_the_quick_matrix_runs_with_no_crash_and_no_torch(
    quick: Callable[[str], subprocess.CompletedProcess[str]], dit: str
) -> None:
    done = quick(dit)
    assert done.returncode == 0, done.stderr
    assert "torch imported: False" in done.stderr
    assert f"--dit {dit} --quick: " in done.stderr and " 0 crashes, " in done.stderr
    lines = done.stdout.splitlines()
    assert lines[0].startswith("## case 1: card=8 ")
    assert any(line.startswith("plan Plan(") for line in lines)
    # A plan's refusal is of a kind (errors.PlanRefused), the library's own words with its facts.
    assert any(line.startswith("REFUSED MemoryShort: ") for line in lines)
    assert any(line.startswith("## recorded ") for line in lines)
    assert any(line.startswith(f"dit_on_gpu{None, 36} -> ") for line in lines)
    assert not any("CRASH" in line for line in lines)
    assert lines[-1].startswith("## total: ") and lines[-1].endswith(" brick items")


def test_the_formats_plan_apart(quick: Callable[[str], subprocess.CompletedProcess[str]]) -> None:
    # --dit reaches the plans: each format's text is its own, the two GGUF files' too, their
    # blocks being of other sizes (peaks.block_sizes).
    assert len({quick(dit).stdout for dit in DITS}) == len(DITS)


def test_diff_lists_the_cases_and_items_that_differ(
    quick: Callable[[str], subprocess.CompletedProcess[str]],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = tool()
    before = quick("fp16").stdout
    (tmp_path / "before.txt").write_text(before)
    parsed_cases, parsed_items, _ = module.parsed(tmp_path / "before.txt")
    cases, items = len(parsed_cases), len(parsed_items)
    lines = before.splitlines()
    # One case's plan moved, the last case of the grid gone, one brick item moved, and the
    # recorded cases numbered one lower, as a refused plan on a larger card numbers them: compared
    # by what they are, they move nothing.
    planned = next(i for i, line in enumerate(lines) if line.startswith("plan Plan("))
    gone = max(i for i, line in enumerate(lines) if line.startswith("## case "))
    item = next(i for i, line in enumerate(lines) if line.startswith("dit_peak("))
    header = next(line for line in reversed(lines[:planned]) if line.startswith("## case "))
    after = list(lines)
    after[planned] = after[planned].replace("Plan(", "Plan(moved, ", 1)
    after[item] = after[item].partition(" -> ")[0] + " -> 1"
    end = next(i for i in range(gone + 1, len(after)) if after[i].startswith("## "))
    del after[gone:end]
    for i, line in enumerate(after):
        if line.startswith("## recorded ") and line.split()[2][0].isdigit():
            number = line.split()[2].rstrip(":")
            after[i] = line.replace(f"## recorded {number}:", f"## recorded {int(number) - 1}:")
    (tmp_path / "after.txt").write_text("\n".join(after) + "\n")
    assert module.main(["diff", str(tmp_path / "before.txt"), str(tmp_path / "before.txt")]) == 0
    assert capsys.readouterr().out == (
        f"0 of {cases} cases differ, 0 of {items} brick items differ\n"
    )
    assert module.main(["diff", str(tmp_path / "before.txt"), str(tmp_path / "after.txt")]) == 1
    said = capsys.readouterr().out.splitlines()
    assert said[-1] == f"2 of {cases} cases differ, 1 of {items} brick items differ"
    moved = said.index(header.partition(":")[0][3:])
    assert said[moved + 1 :].count(f"  - {lines[planned]}") == 1
    assert f"  + {after[planned]}" in said
    assert lines[gone].partition(":")[0][3:] in said
    assert not any(line.startswith("recorded ") for line in said)
    assert "  (not in AFTER)" in said
    assert f"item {lines[item].partition(' -> ')[0]}" in said
    assert "  + 1" in said
    # An output that can't be read, or one cut short of its total line, is compared with nothing.
    cut = "\n".join(line for line in after if not line.startswith("## total: ")) + "\n"
    (tmp_path / "cut.txt").write_text(cut)
    for broken in ("cut.txt", "missing.txt"):
        assert module.main(["diff", str(tmp_path / "before.txt"), str(tmp_path / broken)]) == 2
        assert capsys.readouterr().out == ""

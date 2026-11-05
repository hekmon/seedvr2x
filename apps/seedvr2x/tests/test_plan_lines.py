"""tools/plan_lines.py, the CLI's half of the plan matrix, kept running: its --quick part for each
DiT format, with no crash and no torch imported; core's text in it unchanged (core's
tools/plan_matrix.py's, its own test core/tests/test_plan_matrix.py), the CLI's lines added at
their places; --without-smaller leaving out the smaller DiT files' line alone; and diff comparing
the CLI's checks of a record by what they are. The full matrix is run by hand, before and after a
change to the plan (the tool's docstring)."""

import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1]
TOOL = APP / "tools" / "plan_lines.py"
CORE_TOOL = APP.parents[1] / "core" / "tools" / "plan_matrix.py"
DITS = ("fp16", "int8", "gguf7", "gguf7sharp")
SMALLER = "  | plan: smaller DiT files plan this job better: "
# The CLI's lines, by their start: what core's text doesn't have.
CLI_LINES = (
    "  | ",
    "card_detail ",
    "beyond_measured ",
    "check_recorded ",
    "plan_again",
    "## check ",
    "## checks: ",
    "compile_shapes#",
    "on_windows_drive",
)
# The lines of core's text that the CLI's half words again: a refusal, its parts, the Peaks.
REWORDED = ("REFUSED ", "plan REFUSED ", "peaks ", "  ")
# A tool run as its script runs, then whether torch was imported, said on stderr.
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

Run = Callable[..., subprocess.CompletedProcess[str]]


@pytest.fixture(scope="module")
def quick() -> Run:
    """A tool's --quick run with its options, once each: in a process of its own, from the app's
    root, so that no torch another test imported is there."""
    runs: dict[tuple[str, ...], subprocess.CompletedProcess[str]] = {}

    def run(tool: Path, *options: str) -> subprocess.CompletedProcess[str]:
        key = (str(tool), *options)
        if key not in runs:
            runs[key] = subprocess.run(
                [sys.executable, "-c", DRIVER, str(tool), "run", "--quick", *options],
                cwd=APP,
                capture_output=True,
                text=True,
                timeout=120,
            )
        return runs[key]

    return run


@pytest.mark.parametrize("dit", DITS)
def test_the_quick_matrix_with_the_clis_lines_runs_with_no_crash_and_no_torch(
    quick: Run, dit: str
) -> None:
    done = quick(TOOL, "--dit", dit)
    assert done.returncode == 0, done.stderr
    assert "torch imported: False" in done.stderr
    assert f"--dit {dit} --quick: " in done.stderr and " 0 crashes, " in done.stderr
    lines = done.stdout.splitlines()
    assert lines[0].startswith("## case 1: card=8 ") and lines[0].endswith(" quality=standard")
    for start in CLI_LINES:
        assert any(line.startswith(start) for line in lines), start
    assert not any("CRASH" in line for line in lines)
    assert lines[-1].startswith("## total: ") and lines[-1].endswith(" brick items")


@pytest.mark.parametrize("dit", DITS)
def test_cores_text_is_in_it_unchanged(quick: Run, dit: str) -> None:
    # The CLI's lines taken out, and the quality from the headers, what is left is core's text
    # but its two totals: the CLI's half adds, never changes, but for the lines that hold a
    # library refusal, which this half words as the command line does (the CLI's options, under
    # the class name and the parts a plan's refusal had before the library's had kinds: the
    # proof that the command line's words did not move) where core's gives the library's own
    # words and the kind's facts. Those lines are the same in number, each refusal at its place.
    cli = quick(TOOL, "--dit", dit).stdout.splitlines()
    core = quick(CORE_TOOL, "--dit", dit).stdout.splitlines()
    left = [
        line.removesuffix(f" quality={line.rpartition(' quality=')[2]}")
        if line.startswith("## case ")
        else line
        for line in cli
        if not line.startswith((*CLI_LINES, "## bricks: ", "## total: "))
    ]
    core = [line for line in core if not line.startswith(("## bricks: ", "## total: "))]

    def unrefused(lines: list[str]) -> list[str]:
        return [line for line in lines if not line.startswith(REWORDED)]

    assert unrefused(left) == unrefused(core)
    kinds = re.compile(r"(REFUSED )(MemoryShort|HostShort|PinRefused|CompileRefused)\b")

    def refusals(lines: list[str]) -> list[str]:
        return [
            kinds.sub(r"\1PlanError", line.split(":")[0])
            for line in lines
            if line.startswith(("REFUSED ", "plan REFUSED "))
        ]

    assert refusals(left) == refusals(core)


def test_without_smaller_leaves_out_the_smaller_files_line_alone(quick: Run) -> None:
    with_line = quick(TOOL, "--dit", "fp16").stdout.splitlines()
    without = quick(TOOL, "--dit", "fp16", "--without-smaller").stdout.splitlines()
    said = [line for line in with_line if line.startswith(SMALLER)]
    assert said and [line for line in with_line if not line.startswith(SMALLER)] == without


def test_diff_compares_a_check_by_what_it_is(quick: Run, tmp_path: Path) -> None:
    # One check's result moved, and every check numbered one higher, as one more recorded case
    # numbers them: the moved one alone differs, said by what it is.
    lines = quick(TOOL, "--dit", "fp16").stdout.splitlines()
    checks = [i for i, line in enumerate(lines) if line.startswith("## check ")]
    after = list(lines)
    for i in checks:
        words = after[i].split(maxsplit=3)
        after[i] = f"## check {int(words[2].rstrip(':')) + 1}: {words[3]}"
    moved = checks[len(checks) // 2]
    after[moved] = after[moved].rpartition(": ")[0] + ": 'moved'"
    (tmp_path / "before.txt").write_text("\n".join(lines) + "\n")
    (tmp_path / "after.txt").write_text("\n".join(after) + "\n")
    done = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "diff",
            str(tmp_path / "before.txt"),
            str(tmp_path / "after.txt"),
        ],
        cwd=APP,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 1, done.stderr
    said = done.stdout.splitlines()
    assert re.fullmatch(r"1 of \d+ cases differ, 0 of \d+ brick items differ", said[-1])
    described = lines[moved].split(maxsplit=3)[3]
    label, _, shots = described.partition(" shots=")
    assert said[:3] == [
        f"check {label} shots={shots.partition(':')[0]}",
        f"  - ## check {described}",
        "  + ## check " + after[moved].split(maxsplit=3)[3],
    ]

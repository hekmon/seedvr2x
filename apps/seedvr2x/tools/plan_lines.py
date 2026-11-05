#!/usr/bin/env python3
"""The CLI's half of the plan matrix: core's tools/plan_matrix.py, its cases and brick items, with
what the command line does with each plan, which core's rule leaves out (core imports nothing of
the CLI). The two texts are compared by core's diff.

Usage, from apps/seedvr2x/, in the CLI's environment:
  uv run tools/plan_lines.py run [--dit FORMAT] [--quick] [--without-smaller] > OUT
      core's matrix for FORMAT (fp16, the default, int8, gguf7 or gguf7sharp) with the CLI's
      lines; --quick its small part, in a second
  uv run tools/plan_lines.py diff BEFORE AFTER
      core's diff: the cases and brick items that differ, before and after, and a count

As core's tool, on the tree before a change and on the tree after it, the same FORMAT each time:
a change to the CLI's planner, or to anything that moves a plan, then shows the cases whose plan
or whose lines it moves. tests/test_plan_lines.py keeps it running.

What it adds to core's text, the planner's (planner.py) calls on the plan each case makes:
- after a grid case's Plan and Peaks: the header's quality, the plan's lines as said gives them
  to the log (`  | `, on a card named "Test Card 9000", the disk's line on one case in 3, as
  recorded on one in 5), with the smaller DiT files' line said passes it where smaller_swaps
  finds one (as main does for a new job), then card_detail and beyond_measured;
- a recorded case: check_recorded on its record before the plan, and plan_again after a plan
  refused (PlanError);
- 360 checks of a record (`## check N: ...: result`): check_recorded on records no planner makes,
  for each target and three sets of shots;
- brick items: compile_shapes on one plan of the grid's in 25, run 1, 2 and 4 times, and
  on_windows_drive on paths of a mountinfo text given (never /proc).
--without-smaller leaves the smaller DiT files' line out of said, as the matrix's copies before
it (until 2026-11-05) did: the text they made, which the tool's proof compares.
"""

import argparse
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

# This tree's CLI, and core's tool beside the library it plans with (a folder, not a package).
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "core" / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plan_matrix
from plan_matrix import Case, Grid, Matrix

import planner
from seedvr2x.errors import PlanRefused
from seedvr2x.passes import Plan, Tiles
from seedvr2x.shot_plan import Peaks, PlanError
from seedvr2x.weights import Architecture

APP = Path(__file__).resolve().parents[1]
# The DiT file of each format, which smaller_swaps names the smaller files from (pull.PINNED).
DIT_FILES = {
    "fp16": "seedvr2x_ema_7b_fp16.safetensors",
    "int8": "seedvr2x_ema_7b_int8_convrot.safetensors",
    "gguf7": "seedvr2x_ema_7b_dyn.gguf",
    "gguf7sharp": "seedvr2x_ema_7b_sharp_dyn.gguf",
}
CARD = "Test Card 9000"
DISK = "the masters take 10.0 GiB of the 100.0 GiB free"
RECORDS = {
    "window3": Plan(3),
    "window1": Plan(1),
    "window8": Plan(8),
    "window9": Plan(9),
    "none": Plan(None),
    "enc4096": Plan(None, Tiles(4096)),
    "dec2048": Plan(8, None, Tiles(2048)),
    "enc500": Plan(None, Tiles(500)),
    "enc512ov60": Plan(None, Tiles(512, 60)),
    "both1024": Plan(20, Tiles(1024), Tiles(1024)),
    "both1920": Plan(20, Tiles(1920), Tiles(1920)),
    "compiled": Plan(20, Tiles(1024), Tiles(768), compile_vae=True),
}
CHECKED_SHOTS = ("short17", "long300", "very-long1000")
# A WSL2 machine's mounts: the root on ext4, a Windows drive by 9p with drvfs, a tmpfs.
MOUNTINFO = (
    "22 1 8:1 / / rw - ext4 /dev/sda1 rw\n"
    "30 22 0:27 / /mnt/d rw - 9p D:\\134 rw,dirsync,aname=drvfs;path=D:\\;uid=1000\n"
    "31 22 0:28 / /mnt/wsl rw - tmpfs tmpfs rw\n"
)
MOUNTED = ("/mnt/d/x", "/mnt/wsl/a", "/", "/home/x/y", "/mnt")


class Lines(Matrix):
    """Core's matrix with the CLI's lines, through its hooks."""

    def __init__(self, grid: Grid, dit: Architecture, dit_file: str, smaller: bool) -> None:
        super().__init__(grid, dit)
        self.dit_file = dit_file
        self.smaller = smaller
        self.kept: list[tuple[Plan, tuple[int, int], list[Any]]] = []

    def described(self, case: Case) -> str:
        return f" quality={case.quality}"

    def refusal(self, error: Exception) -> tuple[str, str, list[tuple[str, str]]]:
        """A refusal as the matrix said it before the library's refusals had kinds: a plan's is a
        PlanError (the library names the kind), its message the command line's words, its parts
        the three PlanError had, `instead` as the option string ("" for none). Kept so that this
        half's text proves the CLI's words unchanged."""
        if not isinstance(error, PlanRefused):
            name, _, parts = super().refusal(error)
            return name, planner.worded(error), parts
        parts = [("instead", repr(self.legacy_instead(error))), ("phase", repr(error.phase))]
        return "PlanError", planner.worded(error), [*parts, ("shortfall", repr(error.shortfall))]

    @staticmethod
    def legacy_instead(error: PlanRefused) -> str:
        """The pin a plan's refusal ended with as PlanError.instead said it, an option string. The
        floor's refusal (MemoryShort.instead the resolution that fits) said none then."""
        if error.instead is None or error.instead.name == "resolution":
            return ""
        return planner.option(error.instead)

    def peaks_text(self, made_peaks: Peaks) -> str:
        """Peaks as they were: the refusal of the usual reserve in the command line's words, the
        pin that runs with it as an option string ("" for none)."""
        tightened = made_peaks.tightened
        words = None if tightened is None else planner.worded(tightened)
        instead = "" if made_peaks.instead is None else planner.option(made_peaks.instead)
        # The two fields are strings in what is printed, as they were.
        return repr(replace(made_peaks, tightened=words, instead=instead))  # type: ignore[arg-type]

    def planned(self, case: Case, made: Plan, made_peaks: Peaks) -> None:
        self.kept.append((made, case.target, case.shots))
        smaller = None
        if self.smaller:
            options = {"split": case.split, "recorded": None, "attention": case.attention}
            options |= {"compile_vae": case.compile_vae, "dit_model": self.dit}
            arguments = (case.free, case.target, case.shots, case.pins, case.host)
            ok, smaller = self.attempt(
                lambda: planner.smaller_swaps(made, made_peaks, arguments, options, self.dit_file)
            )
            if not ok:
                self.emit(f"smaller_swaps {smaller}")
                return
        ok, lines = self.attempt(
            lambda: planner.said(
                *(made, made_peaks, CARD, case.target, case.shots, case.pins),
                quality=case.quality,
                recorded=case.index % 5 == 0,
                layouts=None,
                disk=None if case.index % 3 else DISK,
                compile_cache=None,
                smaller=smaller,
            )
        )
        if not ok:
            self.emit(lines)
            return
        for line in lines:
            self.emit(f"  | {line}")
        self.emit(f"card_detail {self.show(lambda: planner.card_detail(made_peaks))}")
        self.emit(f"beyond_measured {self.show(lambda: planner.beyond_measured(made))}")

    def recording(self, kept: Plan, case: Case) -> None:
        checked = self.show(lambda: planner.check_recorded(kept, case.shots, case.target))
        self.emit(f"check_recorded {checked}")

    def refused(self, error: Exception, kept: Plan, case: Case) -> None:
        if not isinstance(error, PlanError):
            return
        ok, again = self.attempt(
            lambda: planner.plan_again(
                *(case.free, case.target, case.shots, case.pins, case.host),
                recorded=kept,
                refused=error,
                split=case.split,
                attention=case.attention,
                dit_model=self.dit,
            )
        )
        if ok:
            self.emit(f"plan_again {again[0]!r}")
            self.emit(f"plan_again peaks {self.peaks_text(again[1])}")
        else:
            self.emit(f"plan_again {again}")

    def checks(self, index: int) -> int:
        start = index
        for target_label, target in self.targets:
            for shots_label in (s for s in CHECKED_SHOTS if s in self.grid.shot_sets):
                shots = self.shots(shots_label)
                for label, record in RECORDS.items():
                    index += 1
                    checked = self.show(partial(planner.check_recorded, record, shots, target))
                    self.emit(
                        f"## check {index}: {label} target={target_label}={target}"
                        f" shots={shots_label}: {checked}"
                    )
        self.emit(f"## checks: {index - start} cases")
        return index

    def more_bricks(self) -> None:
        for number, (made, target, shots) in enumerate(self.kept[::25]):
            for runs in (1, 2, 4):
                self.item(
                    f"compile_shapes#{number} {made!r} {target} {shots!r} runs={runs}",
                    planner.compile_shapes,
                    *(made, target, shots, runs),
                )
        for path in MOUNTED:
            self.item(f"on_windows_drive{path}", planner.on_windows_drive, Path(path), MOUNTINFO)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="The plan matrix with the CLI's lines, and two such texts compared."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    running = commands.add_parser("run", help="the matrix's text with the CLI's lines, on stdout")
    running.add_argument("--dit", choices=list(DIT_FILES), default="fp16", help="the DiT's figures")
    running.add_argument("--quick", action="store_true", help="a small part, in a second")
    running.add_argument(
        "--without-smaller", action="store_true", help="said without the smaller DiT files' line"
    )
    comparing = commands.add_parser("diff", help="the cases and items two outputs differ in")
    comparing.add_argument("before", type=Path)
    comparing.add_argument("after", type=Path)
    arguments = parser.parse_args(argv)
    if Path(planner.__file__).resolve().parent != APP:
        raise SystemExit(
            f"plan_lines: the planner imported, {planner.__file__}, is not this tree's ({APP})"
        )
    if arguments.command == "diff":
        return plan_matrix.diff(arguments.before, arguments.after)
    dit_file, smaller = DIT_FILES[arguments.dit], not arguments.without_smaller
    return plan_matrix.run(
        arguments.dit,
        arguments.quick,
        lambda grid, dit: Lines(grid, dit, dit_file, smaller),
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

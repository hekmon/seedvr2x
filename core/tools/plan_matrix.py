#!/usr/bin/env python3
"""The memory planner's plans over a fixed matrix of jobs, as text, and two such texts compared:
what a change to the plan moves, case by case.

Usage, from core/, in the CLI's environment:
  uv run --project ../apps/seedvr2x tools/plan_matrix.py run [--dit FORMAT] [--quick] > OUT
      the matrix's cases and brick items, for the DiT's figures of FORMAT: fp16 (the default),
      int8, gguf7 or gguf7sharp
  uv run --project ../apps/seedvr2x tools/plan_matrix.py diff BEFORE AFTER
      the cases and brick items that differ between two outputs, before and after, and a count;
      exit status 0 where none does, 1 where some do, 2 where an output can't be read or is cut
      short (no total line)

A change to the planner or to the peaks is shown by `run` on the tree before the change and on
the tree after it, the same FORMAT each time (the four, for a change that may move any), then
`diff`: a change meant to move no plan shows none; one that moves plans on purpose lists them. It
compares a grid's case by its number, a recorded case by what it is (its header less its number
and the record, which a plan refused on the larger card renumbers or moves) and a brick item by
its call. `--quick` runs a small part of the matrix in a second, for tests/test_plan_matrix.py,
which keeps the tool running; the full matrix takes under a minute per format.

What is covered, all of it the library's (core imports nothing of the CLI):
- shot_plan.plan over a grid of 11 cards (the free memory seedvr2x finds on each,
  tests/test_shot_plan.py's our_free) x 10 targets (540p to 4K from 1920x1080 and from a
  1916x1080 crop, as frames.target_size gives them) x 5 sets of shots x 4 figures of system
  memory, then a stride through the product of every dimension: these, and pins, split, attention
  kernel, compiled VAE and the kind of free figure (a plain int, memory.reading's, a held
  Reading, a held Reading on a card with less free than the driver says, the card's whole size
  or what seedvr2x finds of it); each case its Plan and its Peaks, or the refusal with its
  instead, phase and shortfall;
- recorded plans: a plan made on a 96 or a 32 GB card planned again with `recorded` on smaller
  ones, its Plan and Peaks or its refusal;
- the bricks over grids: peaks.tokens, dit_peak, attention_layouts and attention_windows,
  encode_peak, decode_peak, covers, in_compile_range, scratch, dit_on_gpu, and
  shot_plan.host_moments, one line each.
What the CLI does with a plan (its lines, a plan made again after a refusal, a record checked,
the compiled shapes, the cache's disk, the smaller DiT files) is not here: the CLI's half of the
matrix, apps/seedvr2x/tools/plan_lines.py, adds it through Matrix's hooks.

Deterministic: no time, randomness, environment or network reaches the text; memory.reading runs
on the module's state as imported (no driver asked), and the held readings are made from the
memory module's constants. Each case is a block, its header `## case N: ...` or `## recorded N:
...` then its lines; each brick item one line, `call -> result`; a message on several lines is
written on one, its line breaks as `\\n`. stderr says which seedvr2x ran and the counts. Torch is
never imported (planning mustn't), checked at the end; a crash, an exception no refusal raises, is
written in the text and makes the exit status 2.
"""

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import product
from pathlib import Path
from typing import Any

import seedvr2x
from seedvr2x import frames, memory, peaks
from seedvr2x.errors import JobError, Refusal
from seedvr2x.passes import Phase, Plan, Tiles
from seedvr2x.shot_plan import Peaks, Pins, PlanError, host_moments, plan
from seedvr2x.shots import Shot
from seedvr2x.weights import Architecture

PROJECT = Path(__file__).resolve().parents[1]
GIB = 1024**3

DITS = {
    "fp16": peaks.FP16,
    "int8": peaks.INT8,
    "gguf7": peaks.GGUF_7B,
    "gguf7sharp": peaks.GGUF_7B_SHARP,
}
EXPECTED = (PlanError, JobError, Refusal, ValueError)
# What a refusal is printed with, those it has: the kind and the facts of errors.PlanRefused and
# its kinds, each as `part=repr`.
REFUSAL_PARTS = (
    *("kind", "instead", "phase", "shortfall", "pin", "why", "need", "there", "short"),
    *("freeable", "reserve", "available", "pixels", "fits", "fits_input", "within"),
)
# The facts printed only where set: a compiled job's refusal where the plain plan fits only with
# the floor's last rungs (errors.CompileRefused, 2026-11-07), so that every other refusal's text
# is the same as before them.
REFUSAL_PARTS_SET = ("tail_frames", "vae_floor")


def gib(count: float) -> int:
    return round(count * GIB)


def our_free(card: int) -> int:
    """What seedvr2x finds free on a card of `card` GB (tests/test_shot_plan.py's our_free)."""
    return round((94.2746 if card == 96 else card - 1.4285) * GIB)


class Grid:
    """The matrix's dimensions: the full one, or --quick's small part of it."""

    def __init__(self, quick: bool) -> None:
        self.cards = (8, 24, 96) if quick else (6, 8, 10, 12, 16, 20, 24, 32, 48, 80, 96)
        self.hosts = (None, gib(16)) if quick else (None, gib(6), gib(16), gib(64))
        self.sources = ((1920, 1080), (1916, 1080))
        self.resolutions = (720, 2160) if quick else (540, 720, 1080, 1440, 2160)
        shot_sets = {
            "short17": (17,),
            "long300": (300,),
            "mixed17-300-81": (17, 300, 81),
            "very-long1000": (1000,),
            "five": (5, 45, 120, 300, 700),
        }
        kept = ("short17", "mixed17-300-81") if quick else tuple(shot_sets)
        self.shot_sets = {label: shot_sets[label] for label in kept}
        self.sampled = 24 if quick else 1600


PIN_SETS = {
    "none": Pins(),
    "window4": Pins(window=4),
    "window12": Pins(window=12),
    "window40": Pins(window=40),
    "enc512": Pins(encode_tile=512),
    "dec768": Pins(decode_tile=768),
    "untiled": Pins(encode_tile=0, decode_tile=0),
    "tiles1024": Pins(encode_tile=1024, decode_tile=1024),
    "enc2048": Pins(encode_tile=2048),
    "swap12": Pins(swap_blocks=12),
    "w8+swap20+dec512": Pins(window=8, decode_tile=512, swap_blocks=20),
}
# The targets plans recorded are taken from, by label: 720p, 1080p and 4K, and the crop at 1080p.
RECORDED_TARGETS = ("1920x1080@720", "1920x1080@1080", "1920x1080@2160", "1916x1080@1080")
RECORDED_SHOTS = ("long300", "mixed17-300-81", "very-long1000")
TILE_SIZES = (None, 0, 256, 512, 576, 1024, 1152, 1216, 1856, 2048, 4096)
# The output's quality, a digit of the stride's: only the CLI's lines say it.
QUALITIES = ("standard", "enhanced", "maximum")


@dataclass(frozen=True)
class Case:
    """A case's job as plan takes it, for the hooks: its number, the free memory, the target,
    shots and pins, the system memory, split, the attention kernel, the compiled VAE and the
    quality (a recorded case's is the default's)."""

    index: int
    free: int | memory.Reading
    target: tuple[int, int]
    shots: list[Shot]
    pins: Pins
    host: int | None
    split: bool
    attention: str
    compile_vae: bool
    quality: str = QUALITIES[0]


class Matrix:
    """One run's text and counts. The hooks (described, planned, recording, refused, checks and
    more_bricks) add nothing here: the CLI's half of the matrix, apps/seedvr2x/tools/
    plan_lines.py, adds the CLI's lines through them, which core can't import."""

    def __init__(self, grid: Grid, dit: Architecture) -> None:
        self.grid = grid
        self.dit = dit
        self.lines: list[str] = []
        self.crashes = 0
        self.items = 0
        self.targets: list[tuple[str, tuple[int, int]]] = []
        for width, height in grid.sources:
            for resolution in grid.resolutions:
                target = frames.target_size(width, height, Fraction(1), resolution)
                frames.check_target(target, resolution)
                self.targets.append((f"{width}x{height}@{resolution}", target))

    def emit(self, text: str) -> None:
        self.lines.append(text.replace("\n", "\\n"))

    def refusal(self, error: Exception) -> tuple[str, str, list[tuple[str, str]]]:
        """A refusal as the text says it: its class's name, its message and its parts, each as
        (name, repr). The library's own words (str), whatever the application words (a hook:
        the CLI's half words them as its options)."""
        parts = [
            (part, repr(getattr(error, part))) for part in REFUSAL_PARTS if hasattr(error, part)
        ]
        parts += [
            (part, repr(getattr(error, part)))
            for part in REFUSAL_PARTS_SET
            if getattr(error, part, None) not in (None, False)
        ]
        return type(error).__name__, str(error), parts

    def peaks_text(self, made_peaks: Peaks) -> str:
        """A plan's Peaks as the text says them. The two fields the 2.5 GiB reserve had
        (`tightened`, `instead`) retired from Peaks with it (DESIGN.md, Allocator, Small cards,
        2026-11-06); the text keeps them, None and '', where they stood, so that two runs of the
        matrix, before and after, differ in the plans alone."""
        return repr(made_peaks).replace(
            ", dit_model=", ", tightened=None, instead='', dit_model=", 1
        )

    def attempt(self, call: Callable[[], Any]) -> tuple[bool, Any]:
        """(True, the result), or (False, the line saying the refusal or the crash)."""
        try:
            return True, call()
        except EXPECTED as error:
            name, message, parts = self.refusal(error)
            said = "".join(f" [{part}={value}]" for part, value in parts)
            return False, f"REFUSED {name}: {message}{said}"
        except Exception as error:  # written, so that a tree that crashes differs
            self.crashes += 1
            return False, f"CRASH {type(error).__name__}: {error}"

    def show(self, call: Callable[[], Any]) -> str:
        ok, result = self.attempt(call)
        return repr(result) if ok else result

    def shots(self, label: str) -> list[Shot]:
        made: list[Shot] = []
        for count in self.grid.shot_sets[label]:
            start = made[-1].end if made else 0
            made.append(Shot(start, start + count))
        return made

    def free_figure(self, kind: int, card: int, exact: bool) -> tuple[Any, str]:
        """plan's free memory: kind 0 an int (a driver that refuses), 1 memory.reading's of it
        (the module's state as imported: the same), 2 a held Reading (a driver that spills:
        WSL2: torch's own allocator with expandable segments), 3 a held one on a card with 1.5 GiB
        less free than the driver reports, its allocator without the segments (memory.PLAIN)."""
        free = gib(card) if exact else our_free(card)
        label = f"{free}"
        if kind == 0:
            return free, label + " int"
        if kind == 1:
            return memory.reading(free), label + " reading()"
        if kind == 2:
            held = memory.Reading(free, free, 0, True, memory.HELD_BACK, memory.SEGMENTS)
            return replace(held, counted=memory.counted(free, held)), label + " held"
        card_free = max(free - gib(1.5), 0)
        plain = memory.Reading(free, card_free, 0, True, memory.HELD_BACK, memory.PLAIN)
        reading = replace(plain, counted=memory.counted(card_free, plain))
        return reading, label + f" held-card={card_free}"

    def item(self, text: str, function: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """A brick item: `text -> ` and what function(*args, **kwargs) gives, or its refusal."""
        self.items += 1
        self.emit(f"{text} -> {self.show(lambda: function(*args, **kwargs))}")

    def described(self, case: Case) -> str:
        """More of a grid case's header, after its own words (a hook)."""
        return ""

    def planned(self, case: Case, made: Plan, made_peaks: Peaks) -> None:
        """Lines after a grid case's Plan and Peaks (a hook)."""

    def recording(self, kept: Plan, case: Case) -> None:
        """Lines after a recorded case's header, before its plan (a hook)."""

    def refused(self, error: Exception, kept: Plan, case: Case) -> None:
        """Lines after a recorded case's refusal (a hook)."""

    def checks(self, index: int) -> int:
        """Cases after the recorded ones, numbered on from `index`; the last number (a hook)."""
        return index

    def more_bricks(self) -> None:
        """Brick items after the peaks' (a hook)."""

    def plan_case(
        self,
        index: int,
        free_kind: int,
        card: int,
        exact: bool,
        target: tuple[str, tuple[int, int]],
        shots_label: str,
        pins_label: str,
        host: int | None,
        split: bool,
        attention: str,
        compile_vae: bool,
        quality: str,
    ) -> None:
        free, free_label = self.free_figure(free_kind, card, exact)
        shots, pins = self.shots(shots_label), PIN_SETS[pins_label]
        case = Case(
            *(index, free, target[1], shots, pins, host, split, attention, compile_vae, quality)
        )
        self.emit(
            f"## case {index}: card={card} free={free_label} target={target[0]}={target[1]}"
            f" shots={shots_label} pins={pins_label} host={host} split={split}"
            f" attention={attention} compile={compile_vae}{self.described(case)}"
        )
        ok, result = self.attempt(
            lambda: plan(
                *(free, target[1], shots, pins, host),
                split=split,
                attention=attention,
                compile_vae=compile_vae,
                dit_model=self.dit,
            )
        )
        if not ok:
            self.emit(result)
            return
        made, made_peaks = result
        self.emit(f"plan {made!r}")
        self.emit(f"peaks {self.peaks_text(made_peaks)}")
        self.planned(case, made, made_peaks)

    def main_grid(self) -> int:
        """The full grid on the main dimensions, then a stride through the product of all."""
        grid, index = self.grid, 0
        for card in grid.cards:
            for target in self.targets:
                for shots_label in grid.shot_sets:
                    for host in grid.hosts:
                        index += 1
                        self.plan_case(
                            *(index, 0, card, False, target, shots_label, "none", host),
                            *(False, peaks.FLASH_ATTN_2, False, QUALITIES[0]),
                        )
        self.emit(f"## main grid: {index} cases")
        attentions = (peaks.FLASH_ATTN_2, peaks.SDPA)
        # Mixed radix, the last digit varying fastest. Two digits change no plan: numz's padding
        # (retired on 2026-11-04), whose cases are skipped, kept so that the stride samples the
        # cases it sampled before; and the quality, which only the CLI's lines say.
        radices = [len(grid.cards), 2, len(self.targets), len(grid.shot_sets), len(PIN_SETS)]
        radices += [len(grid.hosts), 2, 2, 2, 2, 4, len(QUALITIES)]
        total = 1
        for radix in radices:
            total *= radix
        step, sampled = -1, 0
        while sampled < grid.sampled:
            step += 1
            number = (13 + step * 1000003) % total
            digits: list[int] = []
            for radix in reversed(radices):
                number, digit = divmod(number, radix)
                digits.append(digit)
            c, e, t, s, p, h, sp, at, co, retired, k, q = reversed(digits)
            if retired:
                continue
            sampled += 1
            index += 1
            self.plan_case(
                *(index, k, grid.cards[c], bool(e), self.targets[t], list(grid.shot_sets)[s]),
                *(list(PIN_SETS)[p], grid.hosts[h], bool(sp), attentions[at], bool(co)),
                QUALITIES[q],
            )
        self.emit(f"## sampled: {grid.sampled} cases of {total}")
        return index

    def recorded_cases(self, index: int) -> int:
        """A plan from one card planned again, `recorded`, on a smaller one."""
        start = index
        chosen = [target for target in self.targets if target[0] in RECORDED_TARGETS]
        for source in (96, 32):
            for target in chosen:
                for shots_label in (s for s in RECORDED_SHOTS if s in self.grid.shot_sets):
                    for host in (None, gib(16)):
                        for split in (False, True):
                            for compile_vae in (False, True):
                                index = self.recorded_plans(
                                    *(index, source, target, shots_label, host, split),
                                    compile_vae,
                                )
        self.emit(f"## recorded: {index - start} cases")
        return index

    def recorded_plans(
        self,
        index: int,
        source: int,
        target: tuple[str, tuple[int, int]],
        shots_label: str,
        host: int | None,
        split: bool,
        compile_vae: bool,
    ) -> int:
        shots = self.shots(shots_label)
        options: dict[str, Any] = {"split": split, "dit_model": self.dit}
        options["attention"] = peaks.FLASH_ATTN_2
        ok, first = self.attempt(
            lambda: plan(
                *(our_free(source), target[1], shots, Pins(), host),
                compile_vae=compile_vae,
                **options,
            )
        )
        if not ok:
            return index
        kept = first[0]
        for card in (8, 12, 16, 24, 32, 48):
            if card >= source:
                continue
            index += 1
            self.emit(
                f"## recorded {index}: from={source} to={card} target={target[0]}={target[1]}"
                f" shots={shots_label} host={host} split={split} compile={compile_vae}"
                f" record={kept!r}"
            )
            case = Case(
                *(index, our_free(card), target[1], shots, Pins(), host, split),
                *(options["attention"], kept.compile_vae),
            )
            self.recording(kept, case)
            try:
                again, again_peaks = plan(
                    *(our_free(card), target[1], shots, Pins(), host),
                    recorded=kept,
                    compile_vae=kept.compile_vae,
                    **options,
                )
            except EXPECTED as error:
                name, message, parts = self.refusal(error)
                self.emit(f"plan REFUSED {name}: {message}")
                for part, value in parts:
                    self.emit(f"  {part}={value}")
                self.refused(error, kept, case)
                continue
            except Exception as error:  # written, as attempt's
                self.crashes += 1
                self.emit(f"plan CRASH {type(error).__name__}: {error}")
                continue
            self.emit(f"plan {again!r}")
            self.emit(f"peaks {self.peaks_text(again_peaks)}")
        return index

    def bricks(self) -> int:
        """The peaks' functions over grids, one line each, then more_bricks'; their count."""
        dit = self.dit
        line = self.item
        paddeds = sorted({frames.padded_size(target) for _, target in self.targets})

        def tiles_of(size: int | None) -> Tiles | None:
            return None if size is None else Tiles(size)

        def tokens(padded: tuple[int, int]) -> list[int]:
            return [peaks.tokens(padded, latents) for latents in (1, 2, 13, 41)]

        attentions = (peaks.FLASH_ATTN_2, peaks.SDPA)
        phases: tuple[Phase, ...] = ("encode", "dit", "decode")
        for padded in paddeds:
            line(f"tokens{padded}", tokens, padded)
            for latents, swapped, vae_in_dit, attention in product(
                (1, 2, 3, 6, 12, 20, 41, 80), (0, 3, 12, 36), (True, False), attentions
            ):
                line(
                    f"dit_peak{padded, latents, swapped, vae_in_dit, attention}",
                    peaks.dit_peak,
                    *(padded, latents, swapped, vae_in_dit, attention),
                    dit_model=dit,
                )
            line(
                f"dit_peak{padded, 5, 0, True, 'bogus'}",
                peaks.dit_peak,
                *(padded, 5, 0, True, "bogus"),
                dit_model=dit,
            )
            for name, function in (
                ("attention_layouts", peaks.attention_layouts),
                ("attention_windows", peaks.attention_windows),
            ):
                for latents in (1, 3, 12, 41, 80):
                    line(f"{name}{padded, latents}", function, padded, latents)
            for size in TILE_SIZES:
                for resident, swapped, compiled in product(
                    (None, 0, 10, 36), (0, 10), (False, True)
                ):
                    line(
                        f"encode_peak{padded, size, resident, swapped, compiled}",
                        peaks.encode_peak,
                        *(padded, tiles_of(size), resident, swapped, compiled),
                        dit_model=dit,
                    )
                    for split in (False, True):
                        line(
                            f"decode_peak{padded, size, resident, swapped, split, compiled}",
                            peaks.decode_peak,
                            *(padded, frames.output_size(padded), tiles_of(size), resident),
                            *(swapped, split, compiled),
                            dit_model=dit,
                        )
                if size:
                    line(f"covers{padded, size}", peaks.covers, padded, Tiles(size))
            for phase in phases:
                for size in TILE_SIZES:
                    line(
                        f"in_compile_range{padded, phase, size}",
                        peaks.in_compile_range,
                        *(padded, phase, tiles_of(size)),
                    )
            for latents in (1, 2, 5, 13, 41, 76, 200):
                for cap in (None, 1, 5, 9, 20, 41):
                    line(f"host_moments{padded, latents, cap}", host_moments, padded, latents, cap)
            for count_of_frames in (1, 17, 300, 1000):
                line(f"scratch{padded, count_of_frames}", peaks.scratch, padded, count_of_frames)
        for resident in (None, 0, 1, 10, 36):
            for swapped in (0, 1, 2, 3, 12, 36):
                line(
                    f"dit_on_gpu{resident, swapped}",
                    peaks.dit_on_gpu,
                    *(resident, swapped),
                    dit_model=dit,
                )
        self.more_bricks()
        return self.items


def run(dit_name: str, quick: bool, made: Callable[[Grid, Architecture], Matrix] = Matrix) -> int:
    """The matrix's text on stdout, the Matrix `made` of the grid and the DiT's figures."""
    if Path(seedvr2x.__file__).resolve().parents[1] != PROJECT:
        raise SystemExit(
            f"plan_matrix: the seedvr2x imported, {seedvr2x.__file__}, is not this tree's"
            f" ({PROJECT}): run it from this tree's core/ in its own environment"
        )
    started = time.perf_counter()
    matrix = made(Grid(quick), DITS[dit_name])
    index = matrix.main_grid()
    index = matrix.recorded_cases(index)
    index = matrix.checks(index)
    items = matrix.bricks()
    matrix.emit(f"## bricks: {items} items")
    matrix.emit(f"## total: {index} cases, {items} brick items")
    if "torch" in sys.modules:
        raise SystemExit("plan_matrix: torch was imported, which planning mustn't")
    sys.stdout.write("\n".join(matrix.lines) + "\n")
    sys.stderr.write(
        f"plan_matrix: seedvr2x from {Path(seedvr2x.__file__).parent}, --dit {dit_name}"
        f"{' --quick' if quick else ''}: {index} cases, {items} brick items,"
        f" {matrix.crashes} crashes, torch never imported, {time.perf_counter() - started:.1f} s\n"
    )
    return 2 if matrix.crashes else 0


def case_key(header: str) -> str | None:
    """The key a case is compared by, from its header line; None for a line no case header is.
    A grid's case by its number, which no plan moves; a recorded one by what it is, the record
    left out: a plan refused on the larger card makes no recorded cases, and numbers the next
    ones lower; the CLI's check of a record (plan_lines.py) by what it is too, its result, which
    its header ends with, left out."""
    words = header.split(maxsplit=3)
    if len(words) < 3 or words[0] != "##" or not words[2].rstrip(":").isdigit():
        return None
    if words[1] == "case":
        return f"case {words[2].rstrip(':')}"
    if words[1] == "recorded" and len(words) == 4:
        return "recorded " + words[3].partition(" record=")[0]
    if words[1] == "check" and len(words) == 4:
        described, _, shots = words[3].partition(" shots=")
        return f"check {described} shots={shots.partition(':')[0]}"
    return None


def parsed(path: Path) -> tuple[dict[str, list[str]], dict[str, str], dict[str, str]]:
    """An output's cases (key: header and lines), brick items (call: result) and counts (label:
    line). A recorded case's header, or a check's, is kept without its number."""
    cases: dict[str, list[str]] = {}
    items: dict[str, str] = {}
    counts: dict[str, str] = {}
    case: list[str] | None = None
    for line in path.read_text().splitlines():
        key = case_key(line)
        if key is not None:
            case = cases.setdefault(key, [])
            # A recorded case's header, or a check's, compared without its number, as its key.
            words = line.split(maxsplit=3)
            case.append(line if key.startswith("case ") else f"## {words[1]} {words[3]}")
        elif line.startswith("## "):
            case = None
            counts[line.partition(":")[0]] = line
        elif case is not None:
            case.append(line)
        else:
            call, _, result = line.partition(" -> ")
            key, seen = call, 1
            while key in items:
                seen += 1
                key = f"{call} #{seen}"
            items[key] = result
    return cases, items, counts


def diff(before_path: Path, after_path: Path) -> int:
    """The cases, brick items and counts that differ between two outputs, before (-) and after
    (+), and a count; 1 where any does, else 0. 2, nothing compared, where an output can't be
    read or has no total line, its run cut short: a difference it would show is no change's."""
    try:
        before, before_items, before_counts = parsed(before_path)
        after, after_items, after_counts = parsed(after_path)
    except OSError as error:
        sys.stderr.write(f"plan_matrix: {error}\n")
        return 2
    for path, counts in ((before_path, before_counts), (after_path, after_counts)):
        if "## total" not in counts:
            sys.stderr.write(f"plan_matrix: {path} has no total line: a run cut short?\n")
            return 2
    out: list[str] = []
    cases = 0
    for key in [*before, *(k for k in after if k not in before)]:
        old, new = before.get(key), after.get(key)
        if old == new:
            continue
        cases += 1
        out.append(key)
        out += ["  (not in BEFORE)"] if old is None else [f"  - {line}" for line in old]
        out += ["  (not in AFTER)"] if new is None else [f"  + {line}" for line in new]
    items = 0
    for key in [*before_items, *(k for k in after_items if k not in before_items)]:
        old_item, new_item = before_items.get(key), after_items.get(key)
        if old_item == new_item:
            continue
        items += 1
        out.append(f"item {key}")
        out.append("  (not in BEFORE)" if old_item is None else f"  - {old_item}")
        out.append("  (not in AFTER)" if new_item is None else f"  + {new_item}")
    counts = 0
    for label in [*before_counts, *(k for k in after_counts if k not in before_counts)]:
        if before_counts.get(label) != after_counts.get(label):
            counts += 1
            out += [f"count  - {before_counts.get(label)}", f"       + {after_counts.get(label)}"]
    cases_count = len(set(before) | set(after))
    items_count = len(set(before_items) | set(after_items))
    out.append(
        f"{cases} of {cases_count} cases differ, {items} of {items_count} brick items differ"
    )
    sys.stdout.write("\n".join(out) + "\n")
    return 1 if cases or items or counts else 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="The memory planner's plans over a fixed matrix of jobs, and two compared."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    running = commands.add_parser("run", help="the matrix's text, on stdout")
    running.add_argument("--dit", choices=list(DITS), default="fp16", help="the DiT's figures")
    running.add_argument("--quick", action="store_true", help="a small part, in seconds")
    comparing = commands.add_parser("diff", help="the cases and items two outputs differ in")
    comparing.add_argument("before", type=Path)
    comparing.add_argument("after", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.command == "run":
        return run(arguments.dit, arguments.quick)
    return diff(arguments.before, arguments.after)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

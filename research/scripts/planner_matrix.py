#!/usr/bin/env python3
"""The memory planner's matrix (seedvr2x, milestone 3): seedvr2x run on cards of N GB emulated by
vram_cap.py's ballast, the planner's own plan each time; each phase's predicted peak (the
manifest's plan.memory) against the measured one (the units' "VRAM peak" log lines); then the same
plan pinned on a card one notch under, which the planner must refuse, saying what the driver
expects it to take and to find free; and, with --resume-check, a job stopped after its first
encode, resumed on a smaller card, where its recorded shape must no longer fit, then on a larger
one, where it must finish with the frames of the uninterrupted run.

Run with the python of seedvr2x's environment (it imports seedvr2x's planner to choose each
clip's length), one GPU job at a time:
  python planner_matrix.py --out DIR --source CLIP --vram-cap vram_cap.py --model-dir DIR \
      --case 1080:24:2w --case 2160:bare:w --resume-check 1080:16:2w:10:24 \
      --references REFS/references.json
  python planner_matrix.py --out REFS --source CLIP --vram-cap vram_cap.py --model-dir DIR \
      --make-reference 1080:24:2w --tree TREE      (and --again 2160:bare:w, --self-check NAME)
A case is RES:CARD:MODE. RES is the output's short side, the source taken as 16:9. CARD is a size
in GB, or `bare` (no emulation). MODE picks the clip's length from W, the longest shot the card's
plan runs in one window: 2w = 2W - 2 latents (two windows at the cap), w1 = W + 1 (two windows of
about half), w = W (one window at the cap). The clips are cut from CLIP, from its frame --first
on, one shot each (--cut-threshold 1). A case whose JSON exists in DIR is skipped: a batch can be
run again after an interruption. DIR/extra-cases.txt, when there, names more cases, run last.
Results: DIR/<case>.json and .log, DIR/table.md. A case fails, which its JSON records
("failed"), its table row says and the exit status tells, 1, when its own run fails, when its
run one notch under isn't refused, or is refused with other figures than the driver expects; a
resume check, when its job isn't stopped as asked, isn't refused on the lower card with its
manifest unchanged, or doesn't finish on the higher one with the uninterrupted run's frames. A
case or a check that raises is a failure too. The last line counts what was compared: a batch
that compared nothing says so, and doesn't pass for one where everything was equal.

What the refusal's comparisons are, and aren't. "free" has a side of its own: the constants
below, measured on the card. "takes" is first the sentence seedvr2x logged against this script's
own call of the same planner for the lower card: it holds that the command line gave the planner
the job this script describes, and that the sentence says the planner's figure. That one can't
catch a planner whose constants are wrong: both sides would say the same wrong figure.

So "takes" is also held to what the card showed (measured_takes): the peak the case's own run
MEASURED for the phase the refusal names, the largest of its units' "VRAM peak" log lines, with
the margin and, for a VAE pass, the room, both written here as DESIGN.md has them (MARGIN, ROOM),
not read from the planner; no margin on a machine held to its card, whose plan takes none.
Only where the two are the same thing: the refusal counts a VAE pass with the DiT off the GPU
and the DiT's phase with the VAE's weights away, at the window the run ran; a plan that kept
some of them there isn't compared, and is said so. Within a tolerance by the kind of phase,
said with each comparison (TOLERANCE): a predicted peak is an upper bound that stands as far
over the measure as its fit is loose. The DiT's phase and a tiled decode, whose tile's peak is
a measured one, are held closely, to 0.10 and 0.05 GiB over; an untiled decode by its fit, and
every encode, which the planner takes 0.4 GiB over its peak, only to 0.5 to 1.0 GiB: a constant
wrong by less passes there, and each such line says "loosely".

EMULATED and BARE are what seedvr2x found free on the RTX PRO 6000 this was written on
(research/docs/planner-limits.md's N - 1.44 is numz's): they choose the clips' lengths, and are
what a refusal one notch under is expected to say is free.

On a machine whose driver gives memory the card doesn't have (behind WSL2's GPU device), where
seedvr2x holds itself to the card, only `bare` cases run: the card's reading is taken once, at
the start, by a process of its own, as a run takes it (seedvr2x's runtime/memory.py: what the
driver reports, what the whole card has free, what the plan counts on), with the attention
backend the machine runs; the clips' lengths and the predicted peaks follow them. The run one
notch under isn't made there: vram_cap.py's ballast emulates a card for a driver that refuses,
and a held run's refusal is another program's memory to show (the case's JSON says so).

What other programs hold of the card is taken off the ballast (compensation): the constants
above were measured with nothing else on the card, and vram_cap.py's ballast is a fixed size, so
a program holding 662 MiB beside the runs (2026-10-16) left every emulated card 0.652 GiB short
of its constant, and the planner made another plan in each of the nine cases (2160:24:w
refused). At each emulated run's start, vram_cap's "other" is set from what nvidia-smi gives the
card's compute processes, then corrected by what a probe of seedvr2x's own reading under that
ballast still finds off the constant; the case's log and record say the foreign memory, the
probe, the ballast and what the run then read beside the constant. The foreign memory is read
again at the run's end: where it moved by more than FOREIGN_TOLERANCE_MIB, the case's line says
so and its room is said with both readings. --as-found runs without the compensation. A bare
card can't be compensated: its run reads what the card has, and its record says so.

BY ITS FRAMES (DESIGN.md, Memory planner: a plan is judged by its frames, 2026-10-13). With
--references, each case's frames are compared with its shape's reference: the same clip, the case's
plan on its card as this script makes it, pinned (--window, --encode-tile, --decode-tile,
--swap-blocks) on the bare card with room to spare; no option pins the VAE's placement or the blocks
kept for the VAE's phases, which the bare card's planner chooses. A bare case's plan is the bare
card's own: its reference swaps every block instead, the room a pin gives on that card.
--make-reference makes one (its frames kept, its record in REFS/references.json, with the room each
phase had); one that holds is kept, but made again for other pins or another clip, and one made on
another tree is said; --again runs a reference's shape once more against it, and nothing for a shape
whose reference is missing, failed, or recorded for other pins or another clip: the command's other
shapes are made or run all the same, and the command fails at its end, naming it. A reference
recorded as failed is never compared with: a case judged by it fails, its reference to make again.
Compared by each
frame decoded: ffmpeg's md5 of it (framehash), and seedvr2x's own checksums (checksums/*.crc32, each
frame's CRC-32 as the lossless master stores it), which ffmpeg's CRC-32 of the decoded frame must
equal; by the CRC-32s alone where the segments are gone. A case's shape is its clip's frames (their
digest), its window and its tiles: the swapped blocks and the VAE's placement change no bit
(tests/test_placement.py) and are recorded, not compared. A case whose shape isn't its reference's
isn't compared, and fails: its reference is made again. A case whose frames differ fails, said with
how many, the worst PSNR-Y against the reference's (ffmpeg's psnr) and its room: what its plan's
free memory left beyond each phase's measured peak, beside its reference's (the case's line and the
table say both).
--compare-only DIR compares the outputs an earlier batch left in DIR, running nothing;
--self-check NAME compares a reference with itself and with a copy of it one frame altered."""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

GIB = 1024**3
# What seedvr2x reads free on a card emulated at N GB, N - EMULATED, and on the bare 96 GB card,
# in GiB, a first allocation and its warm-up in place (seedvr2x's cli._free_memory; measured
# 2026-10-11: 24,168,824,832 bytes on a card emulated at 24 GB and 101,159,469,056 on the bare
# one, the same to the byte for a new job, the same command again and a job given a cut list):
# N - 1.49 and 94.21. To choose the clip's length, and for the free memory a refusal one notch
# under must say: kept to the byte, since the planner says what is free rounded down to the
# hundredth, where N - 1.49 itself would stand 0.01 over the 15.5 GB card's 14.00. The matrices
# of 2026-10-11 chose their clips with earlier readings, 1.4285 and 94.2746.
EMULATED = (24 * GIB - 24_168_824_832) / GIB
BARE = 101_159_469_056 / GIB
UNIT = {
    "encode": re.compile(r"encoded in ([\d.]+) s, VRAM peak ([\d.]+) GiB"),
    "dit": re.compile(r"window \d+/\d+, latents \d+ to \d+, in ([\d.]+) s, VRAM peak ([\d.]+) GiB"),
    "decode": re.compile(
        r"decoded in ([\d.]+) s, VRAM peak ([\d.]+) GiB; RAM ([\d.]+) GiB, peak ([\d.]+) GiB"
    ),
}


# What a refusal of the planner's says the plan takes, and what it says is there, in GiB: of
# the GPU's memory or of the system's, whichever refusal it is (seedvr2x's runtime/planner.py:
# _too_large, _window, _swap_refused, _pass, _unfit).
FIGURES = {
    "takes": re.compile(r"takes ([\d.]+) GiB of (?:free VRAM|system memory)"),
    # On a run held to the card, what the plan counts on, which the sentence gives before the
    # driver's own figure ("the plan counts on 22.75 GiB of the 28.05 GiB free on the whole card
    # by NVML (its driver reports 30.03 GiB free to this process), the run held ...").
    "free": re.compile(
        r"the plan counts on ([\d.]+) GiB of the [\d.]+ GiB free"
        r"|reports ([\d.]+) GiB free|and ([\d.]+) GiB are available"
    ),
}
# The planner says its figures to the hundredth of a GiB, what is taken rounded up and what is
# there rounded down: two that differ by more aren't the same figure.
HUNDREDTH = 0.01 + 1e-9

# What the plan leaves under the free memory, and the room a VAE pass is held to besides, in
# GiB, as DESIGN.md has them (Memory planner, Budget: 0.6, and 1.25 further under the budget for
# a VAE pass, decided on 2026-10-18; 1.5 before): written here, not read from seedvr2x's planner,
# so that a planner whose own moved doesn't agree with itself (measured_takes). On a machine
# held to its card the plan takes no margin on top, the 4.5 GiB it leaves under torch's limit
# measuring the same need (DESIGN.md, Allocator): margin_taken.
MARGIN = 0.6
ROOM = 1.25
# How far what a refusal says a phase takes may stand from the peak the case's run measured for
# it with the margin and the room, in GiB, (under, over), by the kind of phase (measured_takes).
# Under: both figures are said to the hundredth, the one rounded up, the other to the nearest,
# and a tiled decode read 0.01 over its prediction in three cases of nine; over: by how much
# the predicted peaks stood over the measured ones in the nine-case matrix on the 96 GB card
# (2026-10-12, on an earlier tree: the planner as 7cc874f has it), and a little more:
# - the DiT's phase: 0.00 to 0.07 GiB over, nine cases;
# - a tiled decode: 0.01 under to 0.02 over, eight cases, its tile's peak a measured one;
# - an untiled decode: 0.67 over, one case, its peak a fit's;
# - a tiled encode: 0.39 to 0.41 over, six cases, the 0.4 GiB the planner takes every encode
#   over its peak (ENCODE_EXTRA, provisional);
# - an untiled encode: 0.30, 0.91 and 0.91 over, three cases, that and its fit.
# The last three hold little: a constant wrong by under 0.5 to 1.0 GiB passes them.
TOLERANCE = {
    "DiT": (0.02, 0.10),
    "tiled decode": (0.03, 0.05),
    "untiled decode": (0.03, 0.75),
    "tiled encode": (0.03, 0.50),
    "untiled encode": (0.03, 1.00),
}
LOOSE = ("untiled decode", "tiled encode", "untiled encode")

# vram_cap.py's own "other" (VRAM_CAP_OTHER_GIB's default), with which the constants above were
# measured, and the probe of what seedvr2x then reads free (compensation).
OTHER = 0.3
PROBE = (
    "import json, torch\n"
    "from seedvr2x import cli\n"
    "print(json.dumps(cli._free_memory(torch.device('cuda', 0))))\n"
)
REFERENCES = "references.json"
MIB = 1 << 20
# How far what other programs hold of the card may move between a run's start and its end, in
# MiB, before the case's line says it and its room is said with both readings: the planner says
# what is free to the hundredth of a GiB, 10 MiB (PROVISIONAL).
FOREIGN_TOLERANCE_MIB = 10


# On a machine held to its card (machine): the bare card's reading there, seedvr2x's
# memory.Reading, in place of BARE; and the attention backend the planner's fit is taken for,
# FlashAttention 2 unless the machine says otherwise.
HELD = None
ATTENTION = "flash_attn_2"
READING = """
import json, torch
from seedvr2x import cli
from seedvr2x.runtime import memory, model
memory.startup()
found = memory.reading(cli._free_memory(torch.device("cuda", 0)))
figures = [found.reported, found.card, found.counted, found.held, found.left]
print(json.dumps([*figures, model.attention_backend()]))
"""


def machine() -> None:
    """Where seedvr2x would hold itself to the card (behind WSL2's GPU device), take the bare
    card's reading and the attention backend as a run does, in a process of its own, which
    leaves nothing on the card: HELD and ATTENTION."""
    global HELD, ATTENTION
    from seedvr2x.runtime import memory

    if not memory.paravirtual():
        return
    said = subprocess.run(
        [sys.executable, "-c", READING], capture_output=True, text=True, check=True
    )
    reading = json.loads(said.stdout.strip().splitlines()[-1])
    reported, card, counted, held, left, attention = reading
    if held:
        # With what torch's limit leaves of the free memory (Reading.left), which the plan's
        # first line says apart from what the plan leaves again.
        HELD, ATTENTION = memory.Reading(reported, card, counted, held, left), attention
        print(
            f"held to the card: {reported / GIB:.2f} GiB reported, "
            f"{'no figure' if card is None else f'{card / GIB:.2f} GiB'} on the card,"
            f" {counted / GIB:.2f} counted on; attention {attention}",
            flush=True,
        )


def free_of(card: str):
    """What a plan is made in on `card`: bytes, or the held machine's own reading of its bare
    card (machine)."""
    if card == "bare" and HELD is not None:
        return HELD
    return round((BARE if card == "bare" else float(card) - EMULATED) * GIB)


def figures(message: str) -> dict:
    """What a refusal says the plan takes and what it says is there, in GiB (FIGURES); None for
    one the message doesn't say."""
    found: dict = {}
    for name, pattern in FIGURES.items():
        hit = pattern.search(message)
        said = [group for group in hit.groups() if group] if hit else []
        found[name] = float(said[0]) if said else None
    return found


def compared(said: list[str], expected: str, free: int) -> dict:
    """The figures of the refusal seedvr2x logged (said, its log's error lines) against the
    driver's: what the driver's own call of the planner says the pinned plan takes (expected,
    its refusal, from the predicted peaks), and the memory the card is expected to have free
    (free, bytes), rounded down as the planner says what is there. Each to the hundredth of a
    GiB the messages are rounded to; "mismatch" names the figures that differ, or that
    seedvr2x's log doesn't say."""
    ours = figures(expected)
    ours["free"] = math.floor(free / GIB * 100 + 1e-6) / 100
    refusals = [figures(line) for line in said]
    theirs = next((found for found in refusals if found["takes"] is not None), figures(""))
    mismatch = [
        name
        for name in FIGURES
        if ours[name] is None or theirs[name] is None or abs(theirs[name] - ours[name]) > HUNDREDTH
    ]
    return {"said": theirs, "expected": ours, "mismatch": mismatch}


# The phase a refusal for a pin names, by its first words (seedvr2x's runtime/planner.py:
# _pass's for a tile pin, _window's and _swap_refused's for the DiT's phase).
REFUSED_PHASE = (
    (re.compile(r"--encode-tile \d+: the encode "), "encode"),
    (re.compile(r"--decode-tile \d+: the decode "), "decode"),
    (re.compile(r"--window \d+: its longest window|--swap-blocks \d+: a window of"), "dit"),
)


def margin_taken() -> float:
    """The margin a plan takes on this machine, in GiB: MARGIN, or none where the run is held
    to its card (machine)."""
    return 0.0 if HELD is not None else MARGIN


# The window a refusal for the DiT's phase counts, in latents (seedvr2x's runtime/planner.py:
# _window's "its longest window, N latents,", _swap_refused's "a window of N latents,",
# _too_large's "its smallest window, N latents,"), and the floor's words for the DiT's phase,
# which counts every block swapped (_too_large).
REFUSED_WINDOW = re.compile(
    r"(?:its longest window,|its smallest window,|a window of) (\d+) latents?,"
)
FLOOR = "with every block of the DiT swapped and the VAE off the GPU"
EVERY_BLOCK = 36


def measured_takes(said: list[str], plan: dict, units: dict) -> dict:
    """What the refusal one notch under says the plan takes (said, its log's error lines)
    against the peak the case's own run measured for the phase it names (units, the largest of
    the phase's "VRAM peak" log lines, to the log's hundredth of a GiB), with the margin the
    plan took (margin_taken: none on a machine held to its card) and, for a VAE pass, the room
    (ROOM): {"phase", "kind", "said", "peak", "margin", "expected", "over", "tolerance",
    "within", "loose"}; or {"skipped": why} where the two aren't the same
    thing: the refusal counts a VAE pass with the DiT off the GPU, and the DiT's phase with the
    VAE's weights away at the window it names, a pin's or the size's smallest, and the plan
    that ran (plan, its manifest's) kept some of them there or ran another window; or the
    refusal names no phase the run measured (the system memory)."""
    line = next((line for line in said if figures(line)["takes"] is not None), "")
    phase = next((phase for pattern, phase in REFUSED_PHASE if pattern.search(line)), None)
    memory = plan["memory"]
    if phase is None and "its smallest window" in line and FLOOR in line:
        # The size's floor, the DiT's smallest window: the run's own where it swapped every
        # block and ran that window.
        if memory["swapped_blocks"] != EVERY_BLOCK:
            return {"skipped": "the refusal counts every block swapped, the run swapped fewer"}
        phase = "dit"
    if phase is None:
        return {"skipped": "the refusal names no phase the run measured"}
    if phase == "dit":
        window = REFUSED_WINDOW.search(line)
        if memory["vae_in_dit"]:
            return {"skipped": "the run kept the VAE's weights on the GPU in the DiT's phase"}
        if window is None or int(window[1]) != memory["longest_window"]:
            return {"skipped": "the refusal counts another window than the run's longest"}
        kind = "DiT"
    else:
        if memory[f"resident_{phase}"] != 0:
            return {"skipped": f"the run kept some of the DiT on the GPU in its {phase}"}
        kind = f"{'untiled' if plan[f'{phase}_tiles'] is None else 'tiled'} {phase}"
    peak = units[phase]["peak"]
    if peak is None:
        return {"skipped": f"no measured peak of the {phase} in the run's log"}
    margin = margin_taken()
    expected = round(peak + margin + (0 if phase == "dit" else ROOM), 2)
    takes = figures(line)["takes"]
    over = round(takes - expected, 2)
    under, most = TOLERANCE[kind]
    return {
        "phase": phase,
        "kind": kind,
        "said": takes,
        "peak": peak,
        "margin": margin,
        "expected": expected,
        "over": over,
        "tolerance": [under, most],
        "within": -under - 1e-9 <= over <= most + 1e-9,
        "loose": kind in LOOSE,
    }


def measure_said(found: dict) -> str:
    """measured_takes' result in a clause, for the case's line."""
    if "skipped" in found:
        return f"not held to a measured peak: {found['skipped']}"
    room = "" if found["phase"] == "dit" else f" and the room, {ROOM}"
    margin = f"the margin, {found['margin']}" if found["margin"] else "no margin (a held run)"
    return (
        f"takes {found['said']} GiB said for {found['expected']} by the {found['kind']}'s peak"
        f" this case's run measured, {found['peak']}, with {margin}{room}:"
        f" {found['over']:+.2f}, {'within' if found['within'] else 'OUTSIDE'}"
        f" -{found['tolerance'][0]} to +{found['tolerance'][1]}"
        f"{', held loosely (the docstring)' if found['loose'] else ''}"
    )


def planned(free, target: tuple[int, int], latents: int, pins=None):
    from seedvr2x.runtime import planner
    from seedvr2x.runtime.job import Shot

    shots = [Shot(0, 4 * latents - 3)]
    pins = pins or planner.Pins()
    return planner.plan(free, target, shots, pins, None, split=True, attention=ATTENTION)


def one_window(free, target: tuple[int, int]) -> int:
    """The longest shot, in latents, the planner runs in one window with `free` bytes free."""
    from seedvr2x.runtime.planner import PlanError

    for latents in range(160, 0, -1):
        try:
            plan, peaks = planned(free, target, latents)
        except PlanError:
            continue
        if peaks.record()["longest_window"] == latents:
            return latents
    raise SystemExit(f"nothing fits {free} at {target}")


def clip(args, frames: int) -> Path:
    path = args.out / "clips" / f"clip-{frames}.mkv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        trim = f"trim=start_frame={args.first}:end_frame={args.first + frames},setpts=PTS-STARTPTS"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(args.source), "-vf", trim, "-an"]
            + ["-c:v", "ffv1", str(path)],
            check=True,
        )
    return path


def foreign() -> tuple[int, list[str]]:
    """What the card's compute processes hold now, in MiB by nvidia-smi, and each one's pid and
    MiB: none of them this script's, which makes no CUDA context."""
    said = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    held = [[int(field) for field in line.split(",")] for line in said.splitlines() if line.strip()]
    return sum(mib for _, mib in held), [f"{pid}: {mib} MiB" for pid, mib in held]


def probe(args, card: str, other: float) -> int:
    """What seedvr2x reads free (cli._free_memory), in bytes, on `card` emulated with vram_cap's
    other at `other` GiB, in a process of its own, which leaves nothing on the card."""
    path = args.out / "free_probe.py"
    path.write_text(PROBE)
    env = dict(os.environ, HF_HUB_OFFLINE="1", VRAM_CAP_GIB=card, VRAM_CAP_OTHER_GIB=repr(other))
    said = subprocess.run(
        [sys.executable, str(args.vram_cap), str(path)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return int(json.loads(said.stdout.strip().splitlines()[-1]))


def compensation(args, card: str) -> dict:
    """vram_cap's other for a run on the emulated `card` now (the docstring): OTHER less what the
    card's other processes hold by nvidia-smi, then less what a probe of seedvr2x's own reading
    under that ballast still finds under the constant (free_of), so that the run reads the
    constant: 5.31 MiB on 2026-10-16, beside a program nvidia-smi gave 662 MiB, the same on
    cards of 12, 15.5, 16, 24, 32 and 48 GB."""
    mib, held = foreign()
    other = OTHER - mib / 1024
    constant = free_of(card)
    first = probe(args, card, other)
    taken = {"foreign_mib": mib, "foreign": held, "constant": constant, "probe": first}
    return taken | {"probe_other": other, "other": other - (constant - first) / GIB}


def compensation_said(taken: dict) -> str:
    """A run's compensation (run), in a line of its log."""
    held = f"foreign {taken['foreign_mib']} MiB ({', '.join(taken['foreign']) or 'nothing else'})"
    if "other" not in taken:
        return f"{held}: {taken['why']}"
    return (
        f"{held}; with vram_cap's other at {taken['probe_other']:.6f} GiB seedvr2x read"
        f" {taken['probe']} bytes free, {(taken['probe'] - taken['constant']) / MIB:+.2f} MiB off"
        f" the constant's {taken['constant']}: other {taken['other']:.6f} GiB"
    )


def run(
    args, name: str, card: str, options: list[str], stop_at: str | None = None
) -> tuple[int, float, str, dict]:
    """seedvr2x run with options, under the emulation unless the card is bare: its exit status,
    its duration in seconds, its log's text and what else held the card, with the compensation
    made for it (compensation; none on a bare card, which can't be, nor with --as-found). With
    stop_at, a Ctrl-C (SIGINT, once) is sent when the log first matches it: the run ends after
    the unit in progress (exit 130)."""
    entry = Path(sys.executable).parent / "seedvr2x"
    command = [sys.executable]
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    if card != "bare":
        command.append(str(args.vram_cap))
        env["VRAM_CAP_GIB"] = card
    if card != "bare" and not args.as_found:
        taken = compensation(args, card)
        env["VRAM_CAP_OTHER_GIB"] = repr(taken["other"])
    else:
        mib, held = foreign()
        why = "a bare card, not compensated" if card == "bare" else "--as-found, not compensated"
        taken = {"foreign_mib": mib, "foreign": held, "why": why}
    command += [str(entry), *options, "--model-dir", str(args.model_dir)]
    log = args.out / f"{name}.log"
    started = time.monotonic()
    with log.open("w") as out:
        out.write(
            f"# VRAM_CAP_GIB={env.get('VRAM_CAP_GIB', 'none')}"
            f" VRAM_CAP_OTHER_GIB={env.get('VRAM_CAP_OTHER_GIB', 'unset')} {' '.join(command)}\n"
        )
        out.write(f"# {compensation_said(taken)}\n")
        out.flush()
        process = subprocess.Popen(command, env=env, stdout=out, stderr=subprocess.STDOUT)
        asked = stop_at is None
        while process.poll() is None:
            time.sleep(0.5)
            if not asked and re.search(stop_at, log.read_text(errors="replace")):
                process.send_signal(signal.SIGINT)
                asked = True
        status = process.returncode
    # And again at its end: a program that freed or took memory meanwhile gave the run another
    # card than the one compensated for (moved).
    taken["foreign_mib_end"], taken["foreign_end"] = foreign()
    if (move := moved(taken)) is not None:
        with log.open("a") as out:
            out.write(
                f"# foreign {taken['foreign_mib']} MiB at the start, {taken['foreign_mib_end']} at"
                f" the end: {move:+d} MiB, past the {FOREIGN_TOLERANCE_MIB} MiB tolerance\n"
            )
    return status, time.monotonic() - started, log.read_text(errors="replace"), taken


def moved(taken: dict) -> int | None:
    """How far what other programs hold of the card moved through a run (run's taken), in MiB,
    where it moved past FOREIGN_TOLERANCE_MIB; else None."""
    move = taken["foreign_mib_end"] - taken["foreign_mib"]
    return move if abs(move) > FOREIGN_TOLERANCE_MIB else None


def measured(text: str) -> dict:
    found: dict = {"units": {}, "plan_lines": re.findall(r"plan: (.*)", text)}
    for phase, pattern in UNIT.items():
        hits = pattern.findall(text)
        found["units"][phase] = {
            "count": len(hits),
            "peak": max((float(hit[1]) for hit in hits), default=None),
            "seconds": round(sum(float(hit[0]) for hit in hits), 1),
        }
        if phase == "decode" and hits:
            found["ram"] = [float(hits[-1][2]), float(hits[-1][3])]
    found["trims"] = text.count("recovered from an allocation failure")
    found["oom"] = [line for line in text.splitlines() if re.search(r"out of memory|OOM", line)][:3]
    found["errors"] = [line for line in text.splitlines() if " ERROR " in line][:5]
    found["vram_cap"] = re.findall(r"vram_cap: (.*)", text)
    return found


def hashes(path: Path, kind: str) -> list[str]:
    """Each frame of the video at path decoded by ffmpeg, as it is stored: its hash (framehash's
    md5 or crc32), in frame order."""
    said = subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0"),
            *("-fps_mode", "passthrough", "-f", "framehash", "-hash", kind, "-"),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [line.rsplit(",", 1)[1].strip() for line in said.splitlines() if line[:1] not in "#"]


def digest(items: list[str]) -> str:
    """One md5 over a list of frames' hashes."""
    return hashlib.md5("\n".join(items).encode()).hexdigest()


CLIPS: dict[Path, str] = {}


def clip_digest(path: Path) -> str:
    """The digest of a clip's frames, each decoded (hashes): what makes two clips the same."""
    if path not in CLIPS:
        CLIPS[path] = digest(hashes(path, "md5"))
    return CLIPS[path]


def frames_of(out: Path) -> dict:
    """The frames of the directory job at out, in segment and frame order (the docstring):
    seedvr2x's own checksums ("crc32"); where the segments are still there, ffmpeg's md5 of each
    frame decoded ("md5") and whether ffmpeg's CRC-32 of each is seedvr2x's ("agree"), else
    None; one md5 over the frames' md5s, or over the CRC-32s without them ("digest")."""
    crc32: list[str] = []
    for path in sorted((out / "checksums").glob("seg_*.crc32")):
        crc32 += path.read_text().split()
    segments = sorted(out.glob("seg_*.mkv"))
    md5: list[str] = []
    decoded: list[str] = []
    for path in segments:
        md5 += hashes(path, "md5")
        decoded += hashes(path, "crc32")
    kept = bool(segments)
    return {
        "segments": [path.name for path in segments],
        "crc32": crc32,
        "md5": md5 if kept else None,
        "agree": decoded == crc32 if kept else None,
        "digest": digest(md5 if kept else crc32),
    }


def shape_key(clip: str, frames: int, plan: dict) -> dict:
    """What decides a run's frames (the docstring): its clip's frames (clip_digest), their count,
    its window and its tiles, from its manifest's plan."""
    tiles = [plan[key]["size"] if plan[key] else 0 for key in ("encode_tiles", "decode_tiles")]
    key = {"clip": clip, "frames": frames, "window": plan["window"]}
    return key | {"encode_tile": tiles[0], "decode_tile": tiles[1]}


def room_of(plan: dict, units: dict) -> dict:
    """What the plan's free memory left beyond each phase's measured peak, in GiB: the least
    left to its VAE passes and to its DiT phase."""
    free = plan["memory"]["free"] / GIB
    return {
        phase: None if units[phase]["peak"] is None else round(free - units[phase]["peak"], 2)
        for phase in ("encode", "dit", "decode")
    }


def against(ours: dict, theirs: dict) -> dict:
    """Two runs' frames (frames_of), frame by frame: by md5 where both have them, by CRC-32
    otherwise; the frames that differ, those one run has and the other hasn't among them."""
    by = "md5" if ours["md5"] is not None and theirs["md5"] is not None else "crc32"
    mine, other = ours[by], theirs[by]
    # zip() to the shorter: the frames past it differ, counted below.
    differ = [
        index for index, pair in enumerate(zip(mine, other, strict=False)) if pair[0] != pair[1]
    ]
    differ += list(range(min(len(mine), len(other)), max(len(mine), len(other))))
    return {"by": by, "frames": len(other), "differ": differ}


def psnr_y(ours: Path, theirs: Path, stats: Path) -> list[float]:
    """Each frame's PSNR-Y of the video ours against theirs, in frame order, by ffmpeg's psnr
    filter, its stats kept at stats; inf where the luma is the same."""
    assert "'" not in str(stats), stats
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-i", str(ours), "-i", str(theirs), "-lavfi"),
            *(f"[0:v][1:v]psnr=stats_file='{stats}'", "-f", "null", "-"),
        ],
        check=True,
    )
    found = [re.search(r"psnr_y:(\S+)", line) for line in stats.read_text().splitlines()]
    return [float(hit[1]) for hit in found if hit]


def worst_psnr(ours: Path, theirs: Path, differ: list[int], stats: Path) -> float | None:
    """The least PSNR-Y over the frames that differ, the segments of the job at ours against
    those of the same names at theirs, in order; None where either's segments are gone."""
    segments = sorted(path.name for path in ours.glob("seg_*.mkv"))
    if not segments or segments != sorted(path.name for path in theirs.glob("seg_*.mkv")):
        return None
    values: list[float] = []
    for segment in segments:
        values += psnr_y(ours / segment, theirs / segment, stats.with_suffix(f".{segment}.psnr"))
    return min(values[index] for index in differ if index < len(values))


def judged(
    args,
    name: str,
    out: Path,
    clip: str,
    frames: int,
    plan: dict,
    units: dict,
    move: int | None = None,
) -> dict:
    """The frames of the case `name`'s run, its output at out, against its shape's reference in
    --references (the docstring): what was compared ("key", "by", "frames", "differ", "agree",
    "worst_psnr_y"), the run's "room" (room_of; "room_at_end" too, the room less what other
    programs took of the card through the run, `move` MiB, where it moved), the reference's
    ("reference_room", "reference_tree") and, for a case that fails, "failed"."""
    found: dict = {"key": shape_key(clip, frames, plan), "room": room_of(plan, units)}
    if move is not None:
        found["room_at_end"] = {
            phase: None if left is None else round(left - move / 1024, 2)
            for phase, left in found["room"].items()
        }
    reference = args.refs["cases"].get(name)
    if reference is None:
        return found | {"failed": f"its frames: no reference of its shape in {args.references}"}
    if "failed" in reference:
        # A reference whose run failed holds no frames to judge by, or ones that aren't its
        # shape's: its record's digests and key, where it has them, aren't compared with.
        return found | {
            "failed": f"its frames not compared: its shape's reference in {args.references} is"
            f" recorded as failed ({'; '.join(reference['failed'])}): that reference is to make"
            " again"
        }
    # A reference made before its room was recorded has it from its plan and its peaks.
    room = reference.get("room")
    if room is None and "plan" in reference and "units" in reference:
        room = room_of(reference["plan"], reference["units"])
    found |= {"reference_room": room, "reference_tree": reference.get("tree")}
    if found["key"] != reference["key"]:
        return found | {
            "failed": f"its frames not compared: its shape {found['key']} isn't its reference's"
            f" {reference['key']}: that reference is to make again"
        }
    ours = frames_of(out)
    found["agree"] = ours["agree"]
    if ours["agree"] is False:
        return found | {"failed": "its frames: seedvr2x's checksums aren't its decoded frames'"}
    found |= against(ours, reference["digests"])
    if not found["differ"]:
        return found
    theirs = args.references.parent / reference["output"]
    stats = args.out / f"{name}-frames"
    found["worst_psnr_y"] = worst = worst_psnr(out, theirs, found["differ"], stats)
    return found | {
        "failed": f"its frames: {len(found['differ'])} of {found['frames']} differ from its"
        f" reference's (by {found['by']}), the worst PSNR-Y"
        f" {'not measured, the segments gone' if worst is None else f'{worst:.2f} dB'};"
        f" {rooms_said(found)}"
    }


def room_said(room: dict | None) -> str:
    """A room (room_of) in a few words: what was left to the encode, the DiT's phase, the decode."""
    if room is None:
        return "(not recorded)"
    return " / ".join("-" if room[phase] is None else f"{room[phase]}" for phase in room)


def rooms_said(found: dict) -> str:
    """The case's room and its reference's, GiB left beyond each phase's measured peak (encode /
    DiT / decode): the case's also at the run's end where other programs' memory moved."""
    said = f"room encode / DiT / decode, the case {room_said(found['room'])}"
    if "room_at_end" in found:
        said += f" ({room_said(found['room_at_end'])} by the card's other programs at its end)"
    if "reference_room" in found:
        said += f", its reference {room_said(found['reference_room'])}"
    return said + " GiB"


def frames_said(found: dict | None) -> str:
    """judged's result in a few words, for the table and the case's line: the frames, then the
    rooms (rooms_said)."""
    if found is None:
        return ""
    if "failed" in found and "differ" not in found:
        return "not compared"
    # A run again against its reference (make_reference) has no room of a case to say.
    rooms = f"; {rooms_said(found)}" if "room" in found else ""
    if found["differ"]:
        worst = found.get("worst_psnr_y")
        psnr = "" if worst is None else f", {worst:.2f} dB"
        return f"{len(found['differ'])} of {found['frames']} differ{psnr}{rooms}"
    return f"the same {found['frames']} (by {found['by']}){rooms}"


def shape_of(spec: str) -> tuple[str, str, tuple[int, int], int, int]:
    """The case spec's name, card, output size, and its clip's latents and frames, as case()
    takes them."""
    resolution, card, mode = spec.split(":")
    target = (int(resolution), int(resolution) * 16 // 9)
    cap = one_window(free_of(card), target)
    latents = {"2w": 2 * cap - 2, "w1": cap + 1, "w": cap}[mode]
    return f"{resolution}-{card}-{mode}", card, target, latents, 4 * latents - 3


def references_of(directory: Path) -> dict:
    """The references' record in directory (REFERENCES), or a new one."""
    path = directory / REFERENCES
    return json.loads(path.read_text()) if path.exists() else {"cases": {}}


def keep_references(directory: Path, refs: dict) -> None:
    """The references' record written whole, in place of the last."""
    path = directory / REFERENCES
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(refs, indent=1))
    partial.replace(path)


def not_again(name: str, spec: str, why: str) -> dict:
    """What --again gives for a shape it runs nothing for (make_reference): said, and a record of
    the failure, not written, so that the command's other shapes are made or run all the same and
    the command fails at its end, naming it (main). Before, such a shape stopped the command at
    once, and no --again after it ran either (a review's LOW-5, 2026-10-16)."""
    print(f"{name}: nothing run again: {why}", flush=True)
    return {"case": name, "spec": spec, "failed": [why]}


def make_reference(args, spec: str, again: bool = False) -> dict:
    """The reference of the case spec's shape (the docstring): its clip, its plan on its card as
    case() has the planner make it, pinned on the bare card; the output kept in --out, the
    record in --out/references.json. With again, the reference's shape once more, its frames
    against the reference's, recorded with it ("again")."""
    refs = references_of(args.out)
    name, card, target, latents, frames = shape_of(spec)
    entry = refs["cases"].get(name)
    if again and entry is None:
        return not_again(name, spec, f"no reference recorded in {args.out / REFERENCES}")
    if again and "failed" in entry:
        failure = "; ".join(entry["failed"])
        where = args.out / REFERENCES
        return not_again(name, spec, f"its reference in {where} is recorded as failed ({failure})")
    plan, _ = planned(free_of(card), target, latents)
    tiles = [tile.size if tile else 0 for tile in (plan.encode_tiles, plan.decode_tiles)]
    pins = {"window": plan.window, "encode_tile": tiles[0], "decode_tile": tiles[1]}
    # The blocks swapped too. An emulated case's own: the bare card leaves its reference room
    # to spare in every phase. A bare case's plan is the bare card's own, packed as tightly:
    # pinned so, its reference was the case again and showed determinism alone (the review's
    # F2, 2026-10-16). It gets the room a pin can give on that card: every block swapped, the
    # DiT's blocks off the GPU in the VAE's phases (EVERY_BLOCK), the window and the tiles kept,
    # the frames' key. No option pins the VAE's weights away in the DiT's phase: the bare
    # card's planner places them. The room each phase had is recorded with the reference.
    pins["swap_blocks"] = EVERY_BLOCK if card == "bare" else plan.swapped
    source = clip(args, frames)
    if entry is not None and "failed" not in entry:
        # One that holds is kept and run again, but not one made for other pins or another
        # clip: that one is made again. One made on another tree is said. --again runs a shape
        # once more against its reference only where the reference is of the shape asked, as a
        # fresh run keeps it: else it runs nothing (the day's review's F9, 2026-10-16: run on a
        # reference made for other pins, it compared the frames of another shape).
        asked = {"clip": clip_digest(source), "frames": frames, "window": pins["window"]}
        asked |= {"encode_tile": pins["encode_tile"], "decode_tile": pins["decode_tile"]}
        stale = [
            what
            for what, was in (("pins", entry["pins"] != pins), ("key", entry.get("key") != asked))
            if was
        ]
        if again and stale:
            recorded = {"pins": (entry["pins"], pins), "key": (entry.get("key"), asked)}
            differ = "; ".join(
                f"its {what} {recorded[what][0]} for {recorded[what][1]}" for what in stale
            )
            return not_again(
                name,
                spec,
                f"its reference isn't of the shape asked ({differ}); --make-reference {spec} makes"
                " it again",
            )
        if not again and not stale:
            tree = entry.get("tree")
            other = "" if tree == args.tree else f" on tree {tree}, where this run's is {args.tree}"
            print(f"{name}: its reference made already{other}: kept", flush=True)
            return entry
        if not again:
            print(
                f"{name}: its reference's {' and '.join(stale)} aren't the ones asked"
                f" ({entry['pins']} for {pins}): made again",
                flush=True,
            )
    run_name = f"ref-{name}-again-{len(entry.get('again', [])) + 1}" if again else f"ref-{name}"
    out = args.out / run_name
    # What a reference made before left, failed or stale, goes: seedvr2x would take it for a job
    # to resume, finished or another.
    shutil.rmtree(out, ignore_errors=True)
    options = [str(source), "-o", str(out), "--resolution", str(target[0]), "--cut-threshold", "1"]
    names = ("--window", "--encode-tile", "--decode-tile", "--swap-blocks")
    for option, pin in zip(names, pins, strict=True):
        options += [] if pins[pin] is None else [option, str(pins[pin])]
    status, seconds, text, taken = run(args, run_name, "bare", options)
    record = {"case": name, "spec": spec, "frames": frames, "clip": source.name, "pins": pins}
    record |= {"planned": {"vae_in_dit": plan.vae_in_dit, "resident_encode": plan.resident_encode}}
    record["planned"]["resident_decode"] = plan.resident_decode
    record |= {"exit": status, "seconds": round(seconds), "output": run_name, "tree": args.tree}
    record |= {"ran": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "card_held": taken}
    record |= measured(text)
    manifest = out / "manifest.json"
    failed = [] if status == 0 and manifest.exists() else [f"its run: exit {status}"]
    if not failed:
        record["plan"] = ran = json.loads(manifest.read_text())["plan"]
        took = shape_key("", frames, ran)
        took_pins = [took["window"], took["encode_tile"], took["decode_tile"]]
        took_pins.append(ran["memory"]["swapped_blocks"])
        pairs = zip(pins.values(), took_pins, strict=True)
        if any(pin is not None and pin != got for pin, got in pairs):
            failed.append(f"its run didn't take the pins {pins}: {took_pins}")
        record["digests"] = frames_of(out)
        record["key"] = shape_key(clip_digest(source), frames, ran)
        record["room"] = room_of(ran, record["units"])
        if record["digests"]["agree"] is False:
            failed.append("seedvr2x's checksums aren't its decoded frames'")
    if failed:
        record["failed"] = failed
    if again and not failed:
        found = against(record["digests"], entry["digests"])
        reference = args.out / entry["output"]
        stats = args.out / f"{run_name}-frames"
        worst = worst_psnr(out, reference, found["differ"], stats) if found["differ"] else None
        record["against"] = found | {"worst_psnr_y": worst}
    # The record read again before it is written: what another run wrote meanwhile stays.
    refs = references_of(args.out)
    if again:
        refs["cases"][name].setdefault("again", []).append(record)
    else:
        refs["cases"][name] = record
    refs |= {"tree": args.tree, "first": args.first}
    keep_references(args.out, refs)
    said = "; ".join(failed) if failed else frames_said(record.get("against")) or "made"
    if "room" in record:
        said += f"; room encode / DiT / decode {room_said(record['room'])} GiB"
    print(f"{run_name}: exit {status} in {seconds:.0f} s, {frames} frames: {said}", flush=True)
    return record


def compare_only(args, spec: str) -> dict:
    """The output an earlier batch left in --compare-only for the case spec (its manifest, its
    checksums, its clip, its record's measured peaks) against its reference, running nothing."""
    name, _, _, _, frames = shape_of(spec)
    out = args.compare_only / name
    plan = json.loads((out / "manifest.json").read_text())["plan"]
    record = args.compare_only / f"{name}.json"
    units = json.loads(record.read_text())["units"] if record.exists() else None
    source = args.compare_only / "clips" / f"clip-{frames}.mkv"
    units = units or {phase: {"peak": None} for phase in ("encode", "dit", "decode")}
    found = judged(args, name, out, clip_digest(source), frames, plan, units)
    print(
        f"{name} (in {args.compare_only}): {found.get('failed') or frames_said(found)}", flush=True
    )
    return {"case": name} | found


def self_check(args, name: str) -> dict:
    """The comparison held to what it must find (the docstring): the reference `name` of
    --references against itself, no frame; then against a copy of it, its middle frame's luma
    inverted (ffmpeg's lutyuv, FFV1 again, its checksums ffmpeg's), that frame alone, with a
    PSNR-Y under any a run gives (lower than 20 dB)."""
    entry = args.refs["cases"][name]
    if "failed" in entry:
        raise SystemExit(
            f"{name}: its reference in {args.references} is recorded as failed"
            f" ({'; '.join(entry['failed'])}): nothing to check"
        )
    out = args.references.parent / entry["output"]
    itself = against(frames_of(out), entry["digests"])
    copy = args.out / f"self-check-{name}"
    (copy / "checksums").mkdir(parents=True, exist_ok=True)
    middle = entry["frames"] // 2
    # The middle frame of the whole output, in whichever segment holds it, by its index there;
    # every other segment copied as it is.
    first = 0
    for segment in sorted(out.glob("seg_*.mkv")):
        altered = copy / segment.name
        count = len(hashes(segment, "md5"))
        if first <= middle < first + count:
            alter = f"lutyuv=y=negval:enable='eq(n,{middle - first})'"
            command = ["ffmpeg", "-v", "error", "-y", "-i", str(segment), "-vf", alter]
            subprocess.run([*command, "-c:v", "ffv1", str(altered)], check=True)
        else:
            shutil.copyfile(segment, altered)
        first += count
        crc32 = "".join(f"{value}\n" for value in hashes(altered, "crc32"))
        (copy / "checksums" / f"{segment.stem}.crc32").write_text(crc32)
    found = against(frames_of(copy), entry["digests"])
    worst = worst_psnr(copy, out, found["differ"], args.out / f"self-check-{name}-frames")
    passed = not itself["differ"] and found["differ"] == [middle] and worst is not None
    passed = passed and worst < 20
    result = {"reference": name, "itself": itself, "altered": found, "frame": middle}
    result |= {"worst_psnr_y": worst, "passed": passed}
    print(
        f"self-check on {name}: itself {len(itself['differ'])} of {itself['frames']} differ;"
        f" frame {middle} altered: {found['differ']} differ (by {found['by']}), PSNR-Y {worst}:"
        f" {'PASSED' if passed else 'FAILED'}",
        flush=True,
    )
    return result


def case(args, spec: str) -> dict:
    from seedvr2x.runtime import planner

    resolution, card, mode = spec.split(":")
    name = f"{resolution}-{card}-{mode}"
    record_path = args.out / f"{name}.json"
    if record_path.exists():
        print(f"{name}: done already", flush=True)
        return json.loads(record_path.read_text())
    target = (int(resolution), int(resolution) * 16 // 9)
    free = free_of(card)
    cap = one_window(free, target)
    latents = {"2w": 2 * cap - 2, "w1": cap + 1, "w": cap}[mode]
    frames = 4 * latents - 3
    result: dict = {"case": name, "card": card, "target": target, "one_window": cap}
    result |= {"latents": latents, "frames": frames}
    print(
        f"{name}: one window holds {cap} latents; clip of {latents} latents, {frames} frames",
        flush=True,
    )
    source = clip(args, frames)
    out = args.out / name
    options = [str(source), "-o", str(out), "--resolution", resolution, "--cut-threshold", "1"]
    status, seconds, text, taken = run(args, name, card, options)
    result |= {"exit": status, "seconds": round(seconds), **measured(text), "card_held": taken}
    manifest = out / "manifest.json"
    if manifest.exists():
        result["plan"] = json.loads(manifest.read_text())["plan"]
    if "plan" in result and HELD is None:
        # What the run read free, beside what the constants say this card has (compensation).
        read, constant = result["plan"]["memory"]["free"], free_of(card)
        result["free_read"] = {"read": read, "constant": constant}
        result["free_read"]["off_mib"] = off = round((read - constant) / MIB, 2)
        with (args.out / f"{name}.log").open("a") as log:
            log.write(f"# the plan's free memory: {read} bytes, the constant's {constant}: ")
            log.write(f"{off:+.2f} MiB\n")
    print(f"{name}: exit {status} in {seconds:.0f} s", flush=True)
    if (move := moved(taken)) is not None:
        result["foreign_moved_mib"] = move
        print(
            f"{name}: the card's other programs held {taken['foreign_mib']} MiB at its start and"
            f" {taken['foreign_mib_end']} at its end ({move:+d} MiB): its room is said with both",
            flush=True,
        )
    # The case's own run must finish: nothing is compared of one that doesn't.
    failed = [] if status == 0 else [f"its own run: exit {status}"]
    # Its frames against its shape's reference, while its segments are there.
    if args.references is not None and status == 0 and "plan" in result:
        plan, units = result["plan"], result["units"]
        result["frames_judged"] = found = judged(
            args, name, out, clip_digest(source), frames, plan, units, move
        )
        if "failed" in found:
            failed.append(found["failed"])
        tree = found.get("reference_tree")
        made = "" if tree in (None, args.tree) else f" (its reference made on tree {tree})"
        said = found.get("failed") or frames_said(found)
        print(f"{name}: its frames: {said}{made}", flush=True)
    # The same plan, pinned, on a card one notch under: the planner must refuse it, with the
    # figures the driver's own call of the planner gives for that card. Planned for one file,
    # for which --plan writes its cut list alone: one file and a directory plan alike.
    if status == 0 and "plan" in result and HELD is not None:
        result["under"] = None
        result["under_skipped"] = (
            "a run held to its card: vram_cap.py's ballast emulates a card for a driver that"
            " refuses; no run one notch under"
        )
    elif status == 0 and "plan" in result:
        plan = result["plan"]
        tiles = [plan[key]["size"] if plan[key] else 0 for key in ("encode_tiles", "decode_tiles")]
        pins = planner.Pins(plan["window"], tiles[0], tiles[1], plan["memory"]["swapped_blocks"])
        size = float(96 if card == "bare" else card)
        under = None
        for step in range(1, 400):
            lower = size - 0.25 * step
            try:
                planned(free_of(str(lower)), target, latents, pins)
            except planner.PlanError as error:
                under = (lower, str(error))
                break
        if under is not None:
            pinned = ["--window", str(pins.window), "--swap-blocks", str(pins.swap_blocks)]
            pinned += ["--encode-tile", str(tiles[0]), "--decode-tile", str(tiles[1])]
            plan_file = args.out / f"{name}-under.cuts"
            options = [
                str(source),
                "-o",
                str(args.out / f"{name}-under.mkv"),
                "--resolution",
                resolution,
            ]
            options += ["--cut-threshold", "1", "--plan", str(plan_file), *pinned]
            status, seconds, text, taken = run(args, f"{name}-under", f"{under[0]:g}", options)
            errors = [line for line in text.splitlines() if " ERROR " in line]
            result["under"] = {"card": under[0], "notch": size - under[0], "expected": under[1]}
            result["under"] |= {"exit": status, "said": errors[:2], "seconds": round(seconds)}
            result["under"]["card_held"] = taken
            checked = compared(errors, under[1], free_of(str(under[0])))
            result["under"]["figures"] = checked
            if status != 1:
                failed.append(f"one notch under: exit {status}, not a refusal's 1")
            failed += [
                f"one notch under: {figure} {checked['said'][figure]} GiB said,"
                f" {checked['expected'][figure]} expected"
                for figure in checked["mismatch"]
            ]
            # And against the peak this case's own run measured for the phase refused, where
            # the two are the same thing (measured_takes).
            above = measured_takes(errors, plan, result["units"])
            result["under"]["measured"] = above
            if not above.get("within", True):
                failed.append(
                    f"one notch under: takes {above['said']} GiB said, {above['expected']} by"
                    f" the {above['kind']}'s peak this case's run measured: {above['over']:+.2f},"
                    f" outside -{above['tolerance'][0]} to +{above['tolerance'][1]}"
                )
            print(
                f"{name}: one notch under ({under[0]:g} GB): exit {status};"
                f" {'; '.join(failed) if failed else 'its figures as the planner gives them'};"
                f" {measure_said(above)}",
                flush=True,
            )
        else:
            failed.append("no card one notch under refuses the pinned plan")
    if failed:
        result["failed"] = failed
    # The masters are big, and only their figures matter here.
    if out.is_dir():
        for path in out.glob("seg_*.mkv"):
            path.unlink()
    record_path.write_text(json.dumps(result, indent=1))
    return result


def resume_check(args, spec: str) -> dict:
    """RES:CARD:MODE:LOWER:HIGHER: the case RES:CARD:MODE run again, stopped by a Ctrl-C after
    its first encode; resumed on a card of LOWER GB, where its recorded shape must no longer fit
    (refused, nothing changed); then on a card of HIGHER GB, where it must finish with the same
    shape and other memory, its frames' checksums those of the case's own uninterrupted run."""
    resolution, card, mode, lower, higher = spec.split(":")
    name = f"{resolution}-{card}-{mode}"
    record_path = args.out / f"{name}-resumed.json"
    if record_path.exists():
        print(f"{name}-resumed: done already", flush=True)
        return json.loads(record_path.read_text())
    base = json.loads((args.out / f"{name}.json").read_text())
    source = clip(args, base["frames"])
    out = args.out / f"{name}-resumed"
    options = [str(source), "-o", str(out), "--resolution", resolution, "--cut-threshold", "1"]
    record: dict = {"case": f"{name}-resumed", "base_plan": base.get("plan")}
    status, seconds, text, _ = run(args, f"{name}-resumed-1", card, options, r"encoded in [\d.]+ s")
    said = [line for line in text.splitlines() if "stopp" in line.lower()][-2:]
    record["stopped"] = {"exit": status, "seconds": round(seconds), "said": said}
    record["stopped"]["units"] = measured(text)["units"]
    manifest = out / "manifest.json"
    before = manifest.read_bytes() if manifest.exists() else None
    status, seconds, text, _ = run(args, f"{name}-resumed-2", lower, options)
    errors = [line for line in text.splitlines() if " ERROR " in line][:2]
    unchanged = before is not None and manifest.read_bytes() == before
    record["lower"] = {
        "card": lower,
        "exit": status,
        "said": errors,
        "manifest_unchanged": unchanged,
    }
    status, seconds, text, _ = run(args, f"{name}-resumed-3", higher, options)
    record["higher"] = {"card": higher, "exit": status, "seconds": round(seconds), **measured(text)}
    if manifest.exists():
        record["plan"] = json.loads(manifest.read_text())["plan"]
    ours = {p.name: p.read_bytes() for p in sorted((out / "checksums").glob("*.crc32"))}
    theirs = {
        p.name: p.read_bytes() for p in sorted((args.out / name / "checksums").glob("*.crc32"))
    }
    record["checksums"] = {"files": len(ours), "identical": bool(ours) and ours == theirs}
    # What the check holds: stopped as asked (a Ctrl-C's 130), refused on the lower card as a
    # recorded plan that no longer fits, its manifest unchanged, finished on the higher one with
    # the uninterrupted run's frames.
    failed = []
    if record["stopped"]["exit"] != 130:
        failed.append(f"stopped: exit {record['stopped']['exit']}, not a Ctrl-C's 130")
    lower_said = " ".join(errors)
    if record["lower"]["exit"] != 1 or "no longer fits the memory free now" not in lower_said:
        failed.append(f"lower: exit {record['lower']['exit']}, not refused as no longer fitting")
    if not unchanged:
        failed.append("lower: the manifest changed")
    if record["higher"]["exit"] != 0:
        failed.append(f"higher: exit {record['higher']['exit']}")
    if not record["checksums"]["identical"]:
        failed.append("higher: not the uninterrupted run's frames")
    if failed:
        record["failed"] = failed
    if out.is_dir():
        for path in out.glob("seg_*.mkv"):
            path.unlink()
    record_path.write_text(json.dumps(record, indent=1))
    print(
        f"{name}-resumed: stopped {record['stopped']['exit']}, lower {record['lower']['exit']},"
        f" higher {record['higher']['exit']}, identical {record['checksums']['identical']}",
        flush=True,
    )
    return record


def table(results: list[dict]) -> str:
    lines = [
        "| case | frames (latents) | cap / longest | swapped, VAE | tiles e / d | residents e / d |"
        " free / budget | encode pred / meas | DiT pred / meas | decode pred / meas | bound | trims |"
        " exit, s | under: card, exit, figures | frames |",
        "|" + "---|" * 15,
    ]
    for r in results:
        plan = r.get("plan")
        if plan is None:
            lines.append(
                f"| {r['case']} | {r['frames']} ({r['latents']}) | no plan | | | | | | | | | | {r['exit']}, {r['seconds']} | | |"
            )
            continue
        memory = plan["memory"]
        gib = lambda key: f"{memory[key] / GIB:.2f}"  # noqa: E731
        tiles = " / ".join(
            str(plan[k]["size"]) if plan[k] else "none" for k in ("encode_tiles", "decode_tiles")
        )
        units = r["units"]
        under = r.get("under")
        notch = ""
        if under is not None:
            # The refusal's figures against the driver's (compared), where the case's JSON
            # holds them.
            checked = under.get("figures")
            said = "" if checked is None else ", ok"
            if checked is not None and checked["mismatch"]:
                said = ", " + ", ".join(
                    f"{figure} {checked['said'][figure]} for {checked['expected'][figure]}"
                    for figure in checked["mismatch"]
                )
            notch = f"{under['card']:g}, {under['exit']}{said}"
        if r.get("failed"):
            notch += " FAILED"
        lines.append(
            f"| {r['case']} | {r['frames']} ({r['latents']}) | {plan['window']} / {memory['longest_window']}"
            f" | {memory['swapped_blocks']}, {'in' if memory['vae_in_dit'] else 'away'} | {tiles}"
            f" | {memory['resident_encode']} / {memory['resident_decode']} | {gib('free')} / {gib('budget')}"
            f" | {gib('encode_peak')} / {units['encode']['peak']} | {gib('dit_peak')} / {units['dit']['peak']}"
            f" | {gib('decode_peak')} / {units['decode']['peak']} | {memory['bound']} | {r['trims']}"
            f" | {r['exit']}, {r['seconds']} | {notch} | {frames_said(r.get('frames_judged'))} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--vram-cap", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--first", type=int, default=0, help="the first frame taken of the source")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--resume-check", action="append", default=[])
    parser.add_argument("--dry", action="store_true", help="say each case's clip, run nothing")
    parser.add_argument("--references", type=Path, help="the references' record to judge by")
    parser.add_argument("--make-reference", action="append", default=[], metavar="SPEC")
    parser.add_argument("--again", action="append", default=[], metavar="SPEC")
    parser.add_argument("--compare-only", type=Path, metavar="DIR")
    parser.add_argument("--self-check", action="append", default=[], metavar="NAME")
    parser.add_argument("--tree", help="the tree a reference runs on, for its record")
    parser.add_argument(
        "--as-found", action="store_true", help="no compensation of what else holds the card"
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.refs = None
    if args.references is not None:
        args.refs = json.loads(args.references.read_text())
    machine()
    if HELD is not None and any(spec.split(":")[1] != "bare" for spec in args.case):
        raise SystemExit("a machine held to its card runs `bare` cases only (the docstring)")
    if args.make_reference or args.again:
        if args.tree is None:
            raise SystemExit("--make-reference and --again name the tree they run on: --tree")
        made = [make_reference(args, spec) for spec in args.make_reference]
        made += [make_reference(args, spec, again=True) for spec in args.again]
        if any("failed" in record for record in made):
            raise SystemExit("FAILED: " + "; ".join(r["case"] for r in made if "failed" in r))
        return
    if args.self_check or args.compare_only:
        if args.refs is None:
            raise SystemExit("--self-check and --compare-only judge by --references")
        checked = [self_check(args, name) for name in args.self_check]
        found = [compare_only(args, spec) for spec in args.case] if args.compare_only else []
        name = "self-check.json" if checked else "compared.json"
        (args.out / name).write_text(json.dumps(checked + found, indent=1))
        if any(not result["passed"] for result in checked) or any("failed" in f for f in found):
            raise SystemExit("FAILED")
        return
    if args.dry:
        for spec in args.case:
            resolution, card, mode = spec.split(":")
            target = (int(resolution), int(resolution) * 16 // 9)
            cap = one_window(free_of(card), target)
            latents = {"2w": 2 * cap - 2, "w1": cap + 1, "w": cap}[mode]
            plan, peaks = planned(free_of(card), target, latents)
            print(
                spec,
                "one window",
                cap,
                "clip latents",
                latents,
                "frames",
                4 * latents - 3,
                plan,
                peaks.record(),
            )
        return
    results = []
    raised: list[str] = []
    for spec in args.case:
        try:
            results.append(case(args, spec))
        except Exception as error:  # a case's failure mustn't end the batch, and fails it
            raised.append(f"{spec}: raised {error!r}")
            print(f"{spec}: FAILED: {error!r}", flush=True)
    for spec in args.resume_check:
        try:
            resume_check(args, spec)
        except Exception as error:
            raised.append(f"resume check {spec}: raised {error!r}")
            print(f"{spec}: resume check FAILED: {error!r}", flush=True)
    # More cases, one a line in <out>/extra-cases.txt, run after the others in the same run: the
    # file may be written while the first ones run.
    extra = args.out / "extra-cases.txt"
    if extra.exists():
        for spec in extra.read_text().split():
            try:
                case(args, spec)
            except Exception as error:
                raised.append(f"{spec}: raised {error!r}")
                print(f"{spec}: FAILED: {error!r}", flush=True)
    every = [json.loads(path.read_text()) for path in sorted(args.out.glob("*.json"))]
    done = [result for result in every if "units" in result]
    resumed = [result for result in every if "stopped" in result]
    (args.out / "table.md").write_text(table(done) + "\n")
    print(table(done), flush=True)
    failed = raised + [
        f"{result['case']}: {reason}"
        for result in (*done, *resumed)
        for reason in result.get("failed", [])
    ]
    print(counted(done, resumed), flush=True)
    if failed:
        raise SystemExit("FAILED: " + "; ".join(failed))


def counted(done: list[dict], resumed: list[dict]) -> str:
    """What the batch compared, in a line, over every result in the output directory: the
    cases, the refusals one notch under held to the planner's figures (their figures counted)
    and, of those, to the peak the case's run measured, closely or loosely (TOLERANCE); the
    cases without such a run; the phases' predicted peaks set against measured ones in the
    table; the resume checks."""
    under = [result["under"] for result in done if result.get("under")]
    figures_held = sum(len(notch["figures"]["expected"]) for notch in under if "figures" in notch)
    held = [notch["measured"] for notch in under if "within" in notch.get("measured", {})]
    loose = sum(1 for found in held if found["loose"])
    phases = sum(
        1
        for result in done
        if "plan" in result
        for unit in result["units"].values()
        if unit["peak"] is not None
    )
    judged_ = [result["frames_judged"] for result in done if "frames_judged" in result]
    compared_frames = [found for found in judged_ if "differ" in found]
    differing = sum(1 for found in compared_frames if found["differ"])
    return (
        f"compared: {len(done)} cases; {len(compared_frames)} cases' frames against their"
        f" shape's reference, {differing} of them differing, {len(judged_) - len(compared_frames)}"
        f" not compared; {len(under)} refusals one notch under against the"
        f" planner's own figures ({figures_held} figures), {len(held)} of them against the peak"
        f" the run above measured too, {loose} of those loosely; {len(done) - len(under)} cases"
        f" with no run one notch under; {phases} phases' predicted peaks beside measured ones in"
        f" the table; {len(resumed)} resume checks"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The frames' floor by tile size (seedvr2x, milestone 3): how little memory a VAE pass in tiles of
a given size can be left beyond its peak before its frames leave those of the same pass run with
room to spare. Judged by the frames, never by the exit status: a pass short of memory can run on
with other frames and say nothing (vae-room.md).

For a VAE pass in tile T, a ladder of cards emulated by vram_cap.py's ballast, under the pinned
shape, the pass alone, until its frames (an encode: its latent) leave those of the tile's
reference rung, the same pass on the bare card. Each rung is one run of seedvr2x's own command
line by vae_room.py (beside this file), in its own process, under the GPU's lock, the planner's
room set in that process low enough (ROOM) that the plan neither refuses nor moves the pinned
pass, and the DiT's phase unjudged (vae_room.py run --dit-unjudged: a pass alone runs no
window), so that every rung of a ladder runs the same plan on less and less free memory:

  - a decode in tile T: seedvr2x's resume of the shape's units (seeds/seed-RES, the job stopped
    before its decode), copied, the copy's manifest's decode tile set to T (settings.decode_tile,
    the pin a resume repeats, and plan.decode_tiles, the shape it takes): seedvr2x's code
    unchanged, the decode exactly seedvr2x's in tile T;
  - an encode in tile T: a new job of the shape, stopped after its encode (--stop-after encode),
    its latent compared bit for bit.

The shapes are two of planner_matrix.py's clips: 1080 = clip-53 (53 frames, window 8, tiles
768 / 576), for the squares up to 1,088 px and the tiles cut off at 1080p's 1,104 rows; 2160 =
clip-17 (17 frames, window 5, tiles 1408 / 1088), for the squares from 1,152 px; 36 blocks
swapped (the fewest on the GPU in every phase); an empty cut list (one shot, no shot detector).

A rung's room is the free memory its plan was made in (the run's own plan, vae_room.py's
"planned") less the pass's measured peak; "usual", that free memory less the reference rung's
peak (vae-room.md's "left"), on which the ladder steps. The ladder: the reference rung, then
START, stepping down while the frames hold or up while they don't (STEP, doubled after two cards
one way), then the bracket halved down to FINE, then the two deciding rungs again (the least
room that kept the frames on every run, the most that lost them on any run), each on the same
card. A rung is kept (its frames, or latent, the reference's), moved (other frames), refused
(the run out of memory, or another memory error) or failed (anything else, which ends the
ladder).

Every rung also records what the run could see of a pass short of room: torch's allocator
counters across the pass (its count of refused allocations), the cuDNN frontend's execution
plans built in the pass (its log on, CUDNN_FRONTEND_LOG_INFO=1, in the rung's log: torch builds
a plan for a convolution's shape at its first call, keeps the first that runs for the life of
the process, and on a workspace refused builds the next ones without a word), torch's "recovered
from an allocation failure" lines; and the pass's seconds, for information only (on a
power-capped card a pass's time drifts from run to run: no signal there).

  seed RES      the shape's units, on the bare card (once).
  rung LADDER   one rung by hand: --card CARD (bare: the reference) or --room GIB.
  queue         the ladders in LADDERS' order, each rung decided from the earlier ones, until the
                UTC time in DIR/deadline, DIR/skip (ends after the rung running) or DIR/pause
                (waits before the next); a run is never interrupted. DIR/done at the end.
  table         per ladder the least room kept and the most lost, with their runs; every rung;
                the ladders not reached; the floors against the tile's area and against the
                largest workspace its reference's plans asked.
  detector      per rung, each signal that doesn't depend on the clock against its reference
                rung's: torch's count of refused allocations, the cuDNN frontend's plans built
                (and the slice where the extra ones were built, with the memory free before
                it), the "recovered" lines; which separates the moved rungs from the kept ones,
                and where; the refused rungs apart.

The study's directory (--dir): records/ (vae_room.py's record and log of each rung), work/,
refs/ (each tile's reference output), seeds/, rungs.jsonl (a record a rung), progress (a line a
rung) and the queue's control files. seed, rung and queue also need --python (seedvr2x's
environment's python), --copy and --want (the frozen copy of seedvr2x under study and its
tree, checked before every rung), --models, --clips (planner_matrix.py's clip-N.mkv) and --lock
(the GPU's lock file). ffmpeg on PATH."""

import argparse
import calendar
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHAPES = {
    1080: {"clip": "clip-53.mkv", "window": 8, "encode": 768, "decode": 576, "frames": 53},
    2160: {"clip": "clip-17.mkv", "window": 5, "encode": 1408, "decode": 1088, "frames": 17},
}
PADDED = {1080: (1104, 1920), 2160: (2192, 3840)}
SWAPPED = 36
ROOM = -50.0  # the planner's room in the rungs' process: no plan refused, none moved
GIB = 1024**3
START, STEP, FINE = 1.1, 0.2, 0.1  # GiB: PROVISIONAL
MOST_RUNGS = 12  # a ladder's rungs at the most, its reference included
TOP, BOTTOM = 4.0, -1.0  # the rooms a ladder goes no further than
OFFSET = 1.49  # card - free (GiB) before any rung measured it (DESIGN.md: N - 1.49)
FIRST = ["dec1080-640", "dec1080-1024", "dec2160-1408", "dec2160-1792", "dec2160-2048"]
FIRST += ["dec1080-1152", "dec1080-1536", "dec1080-1856"]
FIRST += ["enc1080-512", "enc1080-1024", "enc2160-1472", "enc2160-2048"]
SECOND = ["dec1080-832", "dec2160-1216", "dec2160-1600", "dec2160-1920"]
SECOND += ["dec1080-1344", "dec1080-1728"]
THIRD = [f"dec1080-{t}" for t in (704, 768, 896, 960, 1088)]
THIRD += [f"dec2160-{t}" for t in (1152, 1280, 1344, 1472, 1536, 1664, 1728, 1856, 1984)]
THIRD += [f"dec1080-{t}" for t in (1216, 1280, 1408, 1472, 1600, 1664, 1792)]
LADDERS = FIRST + SECOND + THIRD
PLANS = "EXECUTION_PLAN_DESCRIPTOR"  # the cuDNN frontend's line for a plan built
PLAN_LINE = re.compile(r"EXECUTION_PLAN_DESCRIPTOR : (\S+), .*workSpaceSize: (\d+)")
SLICE_LINE = re.compile(
    r"vae_room: slice (?:encode|decode) (\d+): [\d.]+ s, before: allocated ([\d.]+)"
    r" reserved ([\d.]+) driver free ([\d.]+)"
)
TRIM = "recovered from an allocation failure"
# What says a run ended short of memory, besides torch's OutOfMemoryError (run_rung).
MEMORY_ERROR = (
    r"out of memory|unable to find an engine|CUDNN_STATUS_ALLOC_FAILED"
    r"|CUDNN_STATUS_EXECUTION_FAILED|CUBLAS_STATUS_ALLOC_FAILED|cudaErrorMemoryAllocation"
    r"|CUDA_ERROR_OUT_OF_MEMORY"
)

# The study's places, from the arguments (configure).
DIR = RECORDS = WORK = REFS = SEEDS = CUTS = Path()
PROGRESS = RUNGS = DEADLINE = PAUSE = SKIP = DONE = Path()
PYTHON = WANT = MODELS = LOCK = ""
COPY = CLIPS = Path()


def configure(args: argparse.Namespace) -> None:
    global DIR, RECORDS, WORK, REFS, SEEDS, CUTS, PROGRESS, RUNGS, DEADLINE, PAUSE, SKIP, DONE
    global PYTHON, WANT, MODELS, LOCK, COPY, CLIPS
    DIR = args.dir.resolve()
    RECORDS, WORK, REFS, SEEDS = DIR / "records", DIR / "work", DIR / "refs", DIR / "seeds"
    CUTS = DIR / "no-cuts.txt"
    PROGRESS, RUNGS = DIR / "progress", DIR / "rungs.jsonl"
    DEADLINE, PAUSE, SKIP, DONE = DIR / "deadline", DIR / "pause", DIR / "skip", DIR / "done"
    PYTHON, WANT, MODELS, LOCK = args.python, args.want, args.models, args.lock
    COPY = Path(args.copy) if args.copy else Path()
    CLIPS = Path(args.clips) if args.clips else Path()


def needs_gpu_places(args: argparse.Namespace) -> None:
    missing = [
        name
        for name in ("python", "copy", "want", "models", "clips", "lock")
        if not getattr(args, name)
    ]
    if missing:
        raise SystemExit(f"{args.command}: needs {', '.join('--' + name for name in missing)}")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def say(text: str) -> None:
    print(f"{now()} floor: {text}", flush=True)


def parse(ladder: str) -> tuple[str, int, int]:
    found = re.fullmatch(r"(dec|enc)(1080|2160)-(\d+)", ladder)
    if not found:
        raise SystemExit(f"{ladder}: not a ladder (decRES-TILE or encRES-TILE)")
    return found[1], int(found[2]), int(found[3])


def tile_shape(res: int, tile: int) -> str:
    """The tile's full shape as the grid lays it (the edge tiles aside): a square, or cut off at
    the padded frame's rows."""
    rows, _ = PADDED[res]
    return f"{min(tile, rows)}x{tile}"


def gib(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


# ------------------------------------------------------------------ the records


def rungs_of(ladder: str | None = None) -> list[dict]:
    if not RUNGS.exists():
        return []
    found = [json.loads(line) for line in RUNGS.read_text().splitlines() if line.strip()]
    return [r for r in found if ladder is None or r["ladder"] == ladder]


def offset() -> float:
    """card - free in GiB, the median of the emulated rungs run so far (OFFSET before any)."""
    seen = [r["card_offset"] for r in rungs_of() if r.get("card_offset") is not None]
    return round(statistics.median(seen[-9:]), 4) if seen else OFFSET


def within(log: Path, unit: int | None) -> list[str]:
    """A rung's log, between its unit's begin and end lines (the whole log when unit is None)."""
    if not log.exists():
        return []
    found, inside = [], unit is None
    for line in log.read_text(errors="replace").splitlines():
        if unit is not None:
            if re.match(rf"vae_room: unit {unit} begins", line):
                inside = True
                continue
            if re.match(rf"vae_room: unit {unit} \S+ (ends|FAILED)", line):
                inside = False
                continue
        if inside:
            found.append(line)
    return found


def counted(log: Path, unit: int | None) -> dict:
    """Within the unit: the cuDNN frontend's plans built, by engine, and torch's "recovered from
    an allocation failure" lines."""
    found: dict = {"plans": 0, "recovered": 0, "engines": {}}
    for line in within(log, unit):
        if PLANS in line:
            found["plans"] += 1
            engine = re.search(r": (\w+?_eng\d+)", line)
            if engine:
                found["engines"][engine[1]] = found["engines"].get(engine[1], 0) + 1
        elif TRIM in line:
            found["recovered"] += 1
    return found


def by_slice(log: Path, unit: int | None) -> dict:
    """Within the unit, slice by slice: the plans built (tag, workspace in bytes), those logged
    before a slice's own line being its (vae_room.py prints it at the slice's end), the
    "recovered" lines likewise, and the memory before the slice (allocated, reserved and the
    driver's free, GiB); what follows the last slice's line (a slice refused), the next's."""
    plans: dict[int, list[tuple[str, int]]] = {}
    trims: dict[int, int] = {}
    before: dict[int, tuple[float, float, float]] = {}
    pending: list[tuple[str, int]] = []
    pending_trims, last = 0, 0
    for line in within(log, unit):
        hit = PLAN_LINE.search(line)
        if hit:
            pending.append((hit[1], int(hit[2])))
            continue
        if TRIM in line:
            pending_trims += 1
            continue
        hit = SLICE_LINE.search(line)
        if hit:
            last = int(hit[1])
            plans[last], trims[last] = pending, pending_trims
            before[last] = (float(hit[2]), float(hit[3]), float(hit[4]))
            pending, pending_trims = [], 0
    if pending or pending_trims:
        plans[last + 1], trims[last + 1] = pending, pending_trims
    return {"plans": plans, "trims": trims, "before": before, "slices": last}


def foreign() -> list[str]:
    said = subprocess.run(
        [
            *("nvidia-smi", "--query-compute-apps=pid,process_name,used_memory"),
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    ).stdout
    return [line.strip() for line in said.splitlines() if line.strip()]


# ------------------------------------------------------------------ one rung


def seed_for(res: int, tile: int) -> Path:
    """The shape's units (seeds/seed-RES), copied once per decode tile with the manifest's decode
    tile set to it: settings.decode_tile, the pin a resume must repeat, and plan.decode_tiles,
    the recorded shape a resume takes."""
    base = SEEDS / f"seed-{res}"
    path = SEEDS / f"seed-{res}-d{tile}"
    if path.exists():
        return path
    if not (base / "manifest.json").exists():
        raise SystemExit(f"{base}: no seed: run `floor.py seed {res}` first")
    partial = SEEDS / f"seed-{res}-d{tile}.partial"
    shutil.rmtree(partial, ignore_errors=True)
    shutil.copytree(base, partial)
    manifest = partial / "manifest.json"
    content = json.loads(manifest.read_text())
    content["settings"]["decode_tile"] = tile
    content["plan"]["decode_tiles"] = {"size": tile, "overlap": 64}
    manifest.write_text(json.dumps(content, indent=2) + "\n")
    partial.rename(path)
    return path


def command(name: str, kind: str, res: int, tile: int, card: str) -> list[str]:
    shape = SHAPES[res]
    encode = tile if kind == "enc" else shape["encode"]
    decode = tile if kind == "dec" else shape["decode"]
    ours = ["--name", name, "--records", str(RECORDS), "--work", str(WORK), "--tree", WANT]
    ours += ["--vram-cap", str(HERE / "vram_cap.py"), "--card", card, "--room", str(ROOM)]
    ours += ["--dit-unjudged", "--keep"]
    if kind == "dec":
        ours += ["--seed-dir", str(seed_for(res, tile))]
    else:
        ours += ["--stop-after", "encode"]
    theirs = [str(CLIPS / shape["clip"]), "--resolution", str(res), "--cuts", str(CUTS)]
    theirs += ["--window", str(shape["window"]), "--encode-tile", str(encode)]
    theirs += ["--decode-tile", str(decode), "--swap-blocks", str(SWAPPED), "--model-dir", MODELS]
    return ["flock", LOCK, PYTHON, str(HERE / "vae_room.py"), "run", *ours, "--", *theirs]


def environment() -> dict:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1")
    env |= {"CUDNN_FRONTEND_LOG_INFO": "1", "CUDNN_FRONTEND_LOG_FILE": "stderr"}
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def tree() -> str:
    sys.path.insert(0, str(HERE))
    import vae_room

    return vae_room.tree_of(COPY, DIR / "tree-check")


def crcs(out: Path) -> list[str]:
    found: list[str] = []
    for path in sorted((out / "checksums").glob("*.crc32")):
        found += path.read_text().split()
    return found


def psnr_y(ours: Path, theirs: Path, stats: Path) -> list[float]:
    """Each frame's PSNR-Y of ours against theirs, by ffmpeg's psnr filter (planner_matrix's)."""
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-i", str(ours), "-i", str(theirs), "-lavfi"),
            *(f"[0:v][1:v]psnr=stats_file='{stats}'", "-f", "null", "-"),
        ],
        check=True,
    )
    found = [re.search(r"psnr_y:(\S+)", line) for line in stats.read_text().splitlines()]
    return [float(hit[1]) for hit in found if hit]


def latent_of(out: Path):
    import torch

    path = out / "resume" / "shot_000000" / "latent.pt"
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=True)


def latent_md5(latent) -> str:
    import torch

    return hashlib.md5(latent.contiguous().view(torch.int16).numpy().tobytes()).hexdigest()


def latents_compared(ours, theirs) -> dict:
    """Two latents (T', h, w, 16) bfloat16, latent frame by latent frame: those that differ, the
    values that differ, the largest difference, and each differing frame's PSNR over the
    reference's range (the worst)."""
    import torch

    if ours.shape != theirs.shape:
        return {"shape": [list(ours.shape), list(theirs.shape)], "differ": None}
    a, b = ours.float(), theirs.float()
    span = float(b.max() - b.min()) or 1.0
    differ, values, worst, largest = [], 0, None, 0.0
    for index in range(a.shape[0]):
        delta = (a[index] - b[index]).abs()
        count = int((delta > 0).sum())
        if count:
            differ.append(index)
            values += count
            largest = max(largest, float(delta.max()))
            mse = float((delta**2).mean())
            psnr = 10 * math.log10(span**2 / mse)
            worst = psnr if worst is None else min(worst, psnr)
    return {
        "latents": int(a.shape[0]),
        "differ": differ,
        "values": values,
        "of": int(a.numel()),
        "largest": largest,
        "worst_psnr": worst,
        "equal": bool(torch.equal(ours, theirs)),
    }


def run_rung(ladder: str, card: str, target: float | None, name: str) -> dict:
    """One rung: run, read its record, compare it with the ladder's reference, record it."""
    kind, res, tile = parse(ladder)
    phase = "decode" if kind == "dec" else "encode"
    for directory in (RECORDS, WORK, REFS, SEEDS):
        directory.mkdir(parents=True, exist_ok=True)
    CUTS.touch()
    if (RECORDS / f"{name}.json").exists():
        raise SystemExit(f"{name}: its record exists already")
    out = WORK / name
    shutil.rmtree(out, ignore_errors=True)
    held = foreign()
    argv = command(name, kind, res, tile, card)
    log = RECORDS / f"{name}.log"
    began, started = now(), time.monotonic()
    with log.open("w") as file:
        file.write(f"# {began} floor rung {name}: {' '.join(argv)}\n# foreign: {held}\n")
        file.flush()
        status = subprocess.run(
            argv, stdout=file, stderr=subprocess.STDOUT, env=environment()
        ).returncode
    seconds = round(time.monotonic() - started, 1)
    record_path = RECORDS / f"{name}.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else None
    found: dict = {
        "name": name,
        "ladder": ladder,
        "kind": kind,
        "res": res,
        "tile": tile,
        "tile_shape": tile_shape(res, tile),
        "card": card,
        "target": target,
        "began": began,
        "ended": None,
        "exit": status,
        "seconds": seconds,
        "foreign": held,
        "record": str(record_path),
        "log": str(log),
    }
    unit = None
    if record is not None:
        # The run's own plan (vae_room's "planned"), else the manifest's; and the free memory
        # its first plan line says, to the hundredth, where the manifest's plan isn't the run's:
        # a decode resumed and refused before its segment is finished leaves the seed's plan in
        # its manifest.
        plan = record.get("planned") or record.get("plan") or {}
        found["plan_source"] = "planned" if record.get("planned") else "manifest"
        memory = plan.get("memory") or {}
        free = memory.get("free")
        said = re.search(
            r"plan: [^\n]*?, ([\d.]+) GiB free, a budget", log.read_text(errors="replace")
        )
        if said is not None and (free is None or abs(free / GIB - float(said[1])) > 0.006):
            manifest = "no plan" if free is None else f"{free / GIB:.3f}"
            found["plan_source"] = f"its log ({manifest} in the manifest)"
            free = round(float(said[1]) * GIB)
        found["free"] = None if free is None else free / GIB
        found["plan"] = {
            "window": plan.get("window"),
            "encode_tiles": plan.get("encode_tiles"),
            "decode_tiles": plan.get("decode_tiles"),
            "swapped": memory.get("swapped_blocks"),
            "resident_encode": memory.get("resident_encode"),
            "resident_decode": memory.get("resident_decode"),
            "vae_in_dit": memory.get("vae_in_dit"),
            f"{phase}_peak_predicted": None
            if memory.get(f"{phase}_peak") is None
            else memory[f"{phase}_peak"] / GIB,
        }
        units = [u for u in record["units"] if u.get("phase") == phase]
        if units:
            unit = units[0]
            found["peak"] = unit.get("peak_allocated", 0) / GIB
            found["pool_peak"] = unit.get("peak_reserved", 0) / GIB
            found["pass_seconds"] = unit.get("seconds")
            found["slices"] = len(unit.get("slices", []))
            found["slice_seconds"] = [s["seconds"] for s in unit.get("slices", [])]
            found["card_free_least"] = unit.get("card_free_least", 0) / GIB
            found["counters"] = [unit.get("counters_before"), unit.get("counters_after")]
            found["unit_failed"] = unit.get("failed")
        if record.get("oom"):
            oom = record["oom"]
            found["oom"] = {
                "site": oom.get("caught_at"),
                "request": oom.get("request"),
                "allocated": oom.get("allocated"),
                "reserved": oom.get("reserved"),
                "slice": oom.get("slice"),
            }
        if card != "bare" and free is not None:
            found["card_offset"] = float(card) - free / GIB
    numbered = None if unit is None else unit.get("number")
    found |= {"detector": counted(log, numbered)}
    if found.get("free") is not None and "peak" in found:
        found["room"] = found["free"] - found["peak"]
    # The outcome, against the ladder's reference.
    reference = next((r for r in rungs_of(ladder) if r["card"] == "bare" and r.get("ok")), None)
    expected = 0 if kind == "dec" else 130
    ran = record is not None and status == expected and unit is not None and not unit.get("failed")
    refused = record is not None and record.get("oom") is not None
    if record is not None and not ran and not refused:
        # A memory error torch didn't raise as OutOfMemoryError (vae_room records the exception
        # as "raised ..."): refused all the same, said with what said it.
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-80:])
        memory = re.search(MEMORY_ERROR, f"{record.get('exit')}\n{tail}")
        if memory is not None:
            refused = True
            found["refused_by"] = memory[0]
    if card == "bare":
        found["ok"] = ran
        found["outcome"] = "reference" if ran else ("refused" if refused else "failed")
    elif refused:
        found["outcome"] = "refused"
    elif not ran:
        found["outcome"] = "failed"
    if reference is not None and found.get("free") is not None:
        # The reference's peak: this rung's own may be missing (refused before its pass).
        found["room_usual"] = found["free"] - reference["peak"]
    if reference is not None:
        found["reference_seconds"] = reference.get("pass_seconds")
        found["reference_plans"] = reference["detector"]["plans"]
        found["reference_peak"] = reference.get("peak")
    if ran and kind == "dec":
        mine = crcs(out)
        found["frames"] = len(mine)
        found["crc_md5"] = hashlib.md5("\n".join(mine).encode()).hexdigest()
        if card == "bare":
            keep = REFS / ladder
            shutil.rmtree(keep, ignore_errors=True)
            keep.mkdir(parents=True)
            for path in sorted(out.glob("seg_*.mkv")):
                shutil.copy2(path, keep / path.name)
            shutil.copytree(out / "checksums", keep / "checksums")
        elif reference is not None:
            theirs = crcs(REFS / ladder)
            differ = [
                i for i, pair in enumerate(zip(mine, theirs, strict=False)) if pair[0] != pair[1]
            ]
            differ += list(range(min(len(mine), len(theirs)), max(len(mine), len(theirs))))
            found["differ"] = differ
            found["outcome"] = "moved" if differ else "kept"
            if differ:
                found["first_differing"] = differ[0]
                values: list[float] = []
                try:
                    for path in sorted(out.glob("seg_*.mkv")):
                        stats = WORK / f"{name}.{path.stem}.psnr"
                        values += psnr_y(path, REFS / ladder / path.name, stats)
                        stats.unlink(missing_ok=True)
                    found["worst_psnr_y"] = min(values[i] for i in differ if i < len(values))
                except (OSError, ValueError, subprocess.CalledProcessError) as error:
                    found["worst_psnr_y"] = None
                    found["psnr_error"] = repr(error)
    if ran and kind == "enc":
        latent = latent_of(out)
        if latent is None:
            found["outcome"] = "failed"
            found["why"] = "no latent kept"
        else:
            found["latent_md5"] = latent_md5(latent)
            if card == "bare":
                keep = REFS / ladder
                shutil.rmtree(keep, ignore_errors=True)
                keep.mkdir(parents=True)
                shutil.copy2(out / "resume" / "shot_000000" / "latent.pt", keep / "latent.pt")
            elif reference is not None:
                import torch

                theirs = torch.load(
                    REFS / ladder / "latent.pt", map_location="cpu", weights_only=True
                )
                compared = latents_compared(latent, theirs)
                found["latent"] = compared
                found["frames"] = compared.get("latents")
                found["differ"] = compared["differ"]
                same = compared["differ"] == [] and compared.get("equal")
                found["outcome"] = "kept" if same else "moved"
    if reference is None and card != "bare":
        found["outcome"] = found.get("outcome", "uncompared")
    shutil.rmtree(out, ignore_errors=True)
    found["ended"] = now()
    with RUNGS.open("a") as file:
        file.write(json.dumps(found) + "\n")
    progress(found)
    return found


def progress(r: dict) -> None:
    differ = r.get("differ")
    frames = "-" if differ is None else f"{len(differ)}/{r.get('frames')}"
    psnr = r.get("worst_psnr_y") or (r.get("latent") or {}).get("worst_psnr")
    detector = r.get("detector") or {}
    line = (
        f"{r['ended']} | {r['ladder']} ({r['tile_shape']}) | {r['name']} | card {r['card']} |"
        f" free {gib(r.get('free'))} peak {gib(r.get('peak'))} room {gib(r.get('room'))}"
        f" (usual {gib(r.get('room_usual'))}) | differ {frames}"
        f" psnr {'-' if psnr is None else f'{psnr:.2f}'} | plans {detector.get('plans')}"
        f" (ref {r.get('reference_plans', '-')}) recovered {detector.get('recovered')} counters"
        f" {(r.get('counters') or [None, None])[1]} | pass {r.get('pass_seconds')} s"
        f" (ref {r.get('reference_seconds', '-')}) | exit {r['exit']} | {r.get('outcome')}"
    )
    with PROGRESS.open("a") as file:
        file.write(line + "\n")
    say(line)


# ------------------------------------------------------------------ the ladder's next rung


def groups(rungs: list[dict]) -> list[dict]:
    """The emulated rungs by card: each card's room (usual), its outcomes, its runs."""
    found: dict[str, dict] = {}
    for r in rungs:
        if r["card"] == "bare" or r.get("outcome") not in ("kept", "moved", "refused"):
            continue
        group = found.setdefault(r["card"], {"card": r["card"], "rooms": [], "outcomes": []})
        if r.get("room_usual") is not None:
            group["rooms"].append(r["room_usual"])
        group["outcomes"].append(r["outcome"])
    for group in found.values():
        group["room"] = statistics.mean(group["rooms"]) if group["rooms"] else None
        group["kept"] = all(o == "kept" for o in group["outcomes"])
    return sorted((g for g in found.values() if g["room"] is not None), key=lambda g: g["room"])


def bracket(rungs: list[dict]) -> tuple[dict | None, dict | None, list[dict]]:
    """The least room kept on every run above every loss (K), the most room lost on any run (L),
    and every card's group."""
    every = groups(rungs)
    lost = [g for g in every if not g["kept"]]
    least_lost = max(lost, key=lambda g: g["room"]) if lost else None
    above = [
        g for g in every if g["kept"] and (least_lost is None or g["room"] > least_lost["room"])
    ]
    kept = min(above, key=lambda g: g["room"]) if above else None
    return kept, least_lost, every


def next_rung(ladder: str) -> tuple[str, float | None, str] | None:
    """The ladder's next rung: (card, target room, why), or None when the ladder is done."""
    rungs = rungs_of(ladder)
    if any(r.get("outcome") == "failed" for r in rungs):
        return None
    reference = next((r for r in rungs if r["card"] == "bare" and r.get("ok")), None)
    if reference is None:
        if any(r["card"] == "bare" for r in rungs):
            return None  # its reference failed
        return "bare", None, "the reference, room to spare"
    if len(rungs) >= MOST_RUNGS:
        return None
    kept, lost, every = bracket(rungs)
    peak = reference["peak"]

    def card_for(target: float) -> str:
        return f"{peak + target + offset():.3f}"

    if not every:
        start = start_for(ladder)
        return card_for(start), start, "the first rung"
    # While the ladder goes one way, the step doubles after two cards: 0.2, 0.2, 0.4, then 0.8
    # GiB (PROVISIONAL): a floor far from the start is bracketed in fewer rungs, then halved.
    step = STEP * 2 ** max(0, min(len(every) - 2, 2))
    if lost is None:
        target = round(kept["room"] - step, 2)
        if target < BOTTOM:
            return None
        return card_for(target), target, f"kept down to {kept['room']:.3f}: lower"
    if kept is None:
        target = round(lost["room"] + step, 2)
        if target > TOP:
            return None
        return card_for(target), target, f"lost up to {lost['room']:.3f}: higher"
    if kept["room"] - lost["room"] > FINE + 0.005:
        target = round((kept["room"] + lost["room"]) / 2, 2)
        return card_for(target), target, f"bracket {lost['room']:.3f}..{kept['room']:.3f}: halved"
    if len(kept["outcomes"]) < 2:
        return kept["card"], None, f"the least room kept ({kept['room']:.3f}) again"
    if len(lost["outcomes"]) < 2:
        return lost["card"], None, f"the most room lost ({lost['room']:.3f}) again"
    return None


def start_for(ladder: str) -> float:
    """Where a ladder starts: START, or near the floors found already for its kind."""
    kind, _, _ = parse(ladder)
    floors = []
    for other in LADDERS:
        if other == ladder or not other.startswith(kind):
            continue
        kept, lost, _ = bracket(rungs_of(other))
        if kept is not None and lost is not None:
            floors.append(kept["room"])
    if not floors:
        return START
    return round(max(statistics.mean(floors) + 0.05, 0.3), 2)


def name_for(ladder: str, card: str) -> str:
    if card == "bare":
        return f"{ladder}-ref"
    taken = {r["name"] for r in rungs_of(ladder)}
    for count in range(1, 10):
        name = f"{ladder}-c{card}-{count}"
        if name not in taken and not (RECORDS / f"{name}.json").exists():
            return name
    raise SystemExit(f"{ladder}: card {card} run 9 times")


# ------------------------------------------------------------------ the queue


def deadline() -> float | None:
    if not DEADLINE.exists():
        return None
    text = DEADLINE.read_text().strip()
    return calendar.timegm(time.strptime(text, "%Y-%m-%dT%H:%M:%SZ"))


def stop_asked() -> str | None:
    while PAUSE.exists():
        time.sleep(30)
    if SKIP.exists():
        return f"{SKIP.name} exists"
    limit = deadline()
    if limit is not None and time.time() >= limit:
        return f"the deadline {DEADLINE.read_text().strip()} is past"
    return None


def queue(args: argparse.Namespace) -> int:
    say(f"queue started; ladders {LADDERS}")
    ended = "every ladder done"
    for ladder in LADDERS:
        kind, res, tile = parse(ladder)
        if not (SEEDS / f"seed-{res}" / "manifest.json").exists() and kind == "dec":
            say(f"{ladder}: no seed for {res}: skipped")
            continue
        free_disk = shutil.disk_usage(DIR).free / 1e9
        say(f"{ladder}: begins ({tile_shape(res, tile)}); {free_disk:.0f} GB free on the disk")
        if free_disk < 30:
            ended = f"the disk has {free_disk:.0f} GB free"
            break
        while True:
            why = stop_asked()
            if why is not None:
                ended = why
                break
            found = tree()
            if found != WANT:
                ended = f"WRONG TREE {found} in {COPY}, wanted {WANT}"
                break
            step = next_rung(ladder)
            if step is None:
                break
            card, target, reason = step
            name = name_for(ladder, card)
            say(f"{name}: {reason}; card {card}" + ("" if target is None else f", target {target}"))
            try:
                run_rung(ladder, card, target, name)
            except Exception as error:  # the queue goes on to the next ladder
                say(f"{name}: THE DRIVER FAILED: {error!r}")
                failed = {"name": name, "ladder": ladder, "card": card, "outcome": "failed"}
                with RUNGS.open("a") as file:
                    file.write(json.dumps(failed | {"why": repr(error)}) + "\n")
                break
        if ended != "every ladder done":
            break
        # The ladder's own seed copy goes once it is done.
        if kind == "dec":
            shutil.rmtree(SEEDS / f"seed-{res}-d{tile}", ignore_errors=True)
        say(f"{ladder}: done")
    DONE.write_text(f"{now()} {ended}\n")
    say(f"queue ended: {ended}")
    return 0


# ------------------------------------------------------------------ seeds, a rung by hand


def seed(args: argparse.Namespace) -> int:
    res = args.res
    shape = SHAPES[res]
    name = f"seed-{res}"
    for directory in (RECORDS, WORK, SEEDS):
        directory.mkdir(parents=True, exist_ok=True)
    CUTS.touch()
    if (SEEDS / name / "manifest.json").exists():
        say(f"{name}: there already")
        return 0
    ours = ["--name", name, "--records", str(RECORDS), "--work", str(WORK), "--tree", WANT]
    ours += ["--card", "bare", "--room", "1.5", "--stop-after", "windows", "--keep"]
    theirs = [str(CLIPS / shape["clip"]), "--resolution", str(res), "--cuts", str(CUTS)]
    theirs += ["--window", str(shape["window"]), "--encode-tile", str(shape["encode"])]
    theirs += ["--decode-tile", str(shape["decode"]), "--swap-blocks", str(SWAPPED)]
    theirs += ["--model-dir", MODELS]
    argv = ["flock", LOCK, PYTHON, str(HERE / "vae_room.py"), "run", *ours, "--", *theirs]
    log = RECORDS / f"{name}.log"
    with log.open("w") as file:
        file.write(f"# {now()} {' '.join(argv)}\n")
        file.flush()
        status = subprocess.run(
            argv, stdout=file, stderr=subprocess.STDOUT, env=environment()
        ).returncode
    content = json.loads((WORK / name / "manifest.json").read_text())
    shots = content["shots"]
    say(f"{name}: exit {status}; shots {[(s['encoded'], s['windows_done']) for s in shots]}")
    if status != 130 or not all(s["windows_done"] == len(s["windows"]) for s in shots):
        raise SystemExit(f"{name}: not stopped before its decode with its windows done")
    shutil.rmtree(SEEDS / name, ignore_errors=True)
    shutil.move(str(WORK / name), str(SEEDS / name))
    say(f"{name}: kept in {SEEDS / name}")
    return 0


def rung(args: argparse.Namespace) -> int:
    if args.card is not None:
        card, target = args.card, None
    else:
        reference = next(
            (r for r in rungs_of(args.ladder) if r["card"] == "bare" and r.get("ok")), None
        )
        if reference is None:
            raise SystemExit(f"{args.ladder}: no reference rung yet (--card bare first)")
        target = args.room
        card = f"{reference['peak'] + target + offset():.3f}"
    run_rung(args.ladder, card, target, name_for(args.ladder, card))
    return 0


# ------------------------------------------------------------------ the tables


def log_of(r: dict) -> Path:
    """A rung's log in the study's directory (its record names where it was written)."""
    return RECORDS / f"{r['name']}.log"


def unit_of(r: dict) -> int | None:
    """The number of the rung's pass in its run (vae_room's record), None if it has none."""
    path = RECORDS / f"{r['name']}.json"
    if not path.exists():
        return None
    phase = "decode" if r.get("kind") == "dec" else "encode"
    units = [u for u in json.loads(path.read_text())["units"] if u.get("phase") == phase]
    return units[0].get("number") if units else None


def largest_workspace(r: dict) -> int | None:
    """The largest workspace of the plans built in the rung's pass (bytes)."""
    found = [
        int(hit[2]) for line in within(log_of(r), unit_of(r)) if (hit := PLAN_LINE.search(line))
    ]
    return max(found) if found else None


def line_fit(points: list[tuple[float, float]]) -> tuple[float, float, float] | None:
    """The least-squares line y = a + b x through the points, and its worst miss."""
    if len(points) < 2 or len({x for x, _ in points}) < 2:
        return None
    xs, ys = [x for x, _ in points], [y for _, y in points]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    return a, b, max(abs(a + b * x - y) for x, y in points)


def table(args: argparse.Namespace) -> int:
    every = rungs_of()
    ladders = list(dict.fromkeys(r["ladder"] for r in every))
    print(
        "| ladder | tile | ref peak GiB | ref pass s | ref plans | least room kept (runs) |"
        " most room lost (runs) | done |\n|---|---|---|---|---|---|---|---|"
    )
    floors = []
    for ladder in ladders:
        rungs = rungs_of(ladder)
        reference = next((r for r in rungs if r["card"] == "bare" and r.get("ok")), None)
        kept, lost, _ = bracket(rungs)

        def said(group: dict | None, rungs: list[dict] = rungs) -> str:
            if group is None:
                return "-"
            runs = [r for r in rungs if r.get("card") == group["card"]]
            parts = [f"{r['name'].rsplit('-', 1)[-1]}: {r.get('outcome')}" for r in runs]
            return f"{group['room']:.3f} ({len(runs)}: {', '.join(parts)})"

        done = next_rung(ladder) is None
        print(
            f"| {ladder} | {rungs[0].get('tile_shape', '-') if rungs else '-'}"
            f" | {gib(None if reference is None else reference.get('peak'))}"
            f" | {'-' if reference is None else reference.get('pass_seconds')}"
            f" | {'-' if reference is None else reference['detector']['plans']}"
            f" | {said(kept)} | {said(lost)} | {'yes' if done else 'no'} |"
        )
        if reference is not None and kept is not None:
            rows, columns = (int(x) for x in rungs[0]["tile_shape"].split("x"))
            space = largest_workspace(reference)
            floors.append(
                {
                    "ladder": ladder,
                    "area": rows * columns / 1e6,
                    "workspace": None if space is None else space / GIB,
                    "kept": kept["room"],
                    "lost": None if lost is None else lost["room"],
                    "lost_how": None if lost is None else "/".join(sorted(set(lost["outcomes"]))),
                }
            )
    reached = set(ladders)
    print(f"\nNot reached: {', '.join(x for x in LADDERS if x not in reached) or 'none'}")
    print(
        "\n| rung | card | free | peak | room | room usual | outcome | frames differing | worst"
        " PSNR | plans (ref) | recovered | pass s (ref) | exit |"
        "\n|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    for r in every:
        differ = r.get("differ")
        psnr = r.get("worst_psnr_y") or (r.get("latent") or {}).get("worst_psnr")
        detector = r.get("detector") or {}
        print(
            f"| {r.get('name')} | {r.get('card')} | {gib(r.get('free'))} | {gib(r.get('peak'))}"
            f" | {gib(r.get('room'))} | {gib(r.get('room_usual'))} | {r.get('outcome')}"
            f" | {'-' if differ is None else str(len(differ)) + ' of ' + str(r.get('frames'))}"
            f" | {'-' if psnr is None else f'{psnr:.2f}'}"
            f" | {detector.get('plans')} ({r.get('reference_plans', '-')})"
            f" | {detector.get('recovered')}"
            f" | {r.get('pass_seconds')} ({r.get('reference_seconds', '-')}) | {r.get('exit')} |"
        )
    print(
        "\nThe floor against the tile (ladders with a room kept: the least kept, the most lost):"
        "\n\n| ladder | tile area Mpx | reference's largest workspace GiB | least kept |"
        " most lost |\n|---|---|---|---|---|"
    )
    for f in floors:
        lost = "-" if f["lost"] is None else f"{f['lost']:.3f} ({f['lost_how']})"
        print(
            f"| {f['ladder']} | {f['area']:.3f} | {gib(f['workspace'])} | {f['kept']:.3f}"
            f" | {lost} |"
        )
    # A line through the bracketed ladders' least room kept, by kind, then by kind and
    # resolution, against the tile's area (Mpx) and its plans' largest workspace (GiB), with its
    # worst miss; and the step that bounds them, the most room any of them needed.
    for kind in ("dec", "dec1080", "dec2160", "enc", "enc1080", "enc2160"):
        mine = [f for f in floors if f["ladder"].startswith(kind) and f["lost"] is not None]
        if mine:
            most = max(mine, key=lambda f: f["kept"])
            print(
                f"{kind}: every bracketed ladder kept its frames with {most['kept']:.3f} GiB"
                f" ({most['ladder']}) or less, over {len(mine)} ladders"
            )
        for name in ("area", "workspace"):
            points = [(f[name], f["kept"]) for f in mine if f[name] is not None]
            fit = line_fit(points)
            if fit is None:
                print(f"{kind}, least kept against the {name}: fewer than two points")
                continue
            a, b, worst = fit
            print(
                f"{kind}, least kept against the {name}: {a:.3f} + {b:.4f} x, worst miss"
                f" {worst:.3f} GiB over {len(points)} ladders"
            )
    return 0


REFUSALS = ("num_ooms", "num_alloc_retries", "num_oom_rejections")  # torch's, per allocator


def detector(args: argparse.Namespace) -> int:
    """Each rung's signals that don't depend on the clock, against its reference rung's, and which
    separate the rungs whose frames moved from those that kept them: torch's count of refused
    allocations across the pass (memory_stats), the cuDNN frontend's plans built in the pass (its
    log, not torch's), torch's "recovered from an allocation failure" lines. The pass's time is
    no signal here: on a power-capped card it drifts from run to run (benchmarking.md)."""
    every = rungs_of()
    refs = {r["ladder"]: r for r in every if r.get("card") == "bare" and r.get("ok")}
    slices: dict[str, dict] = {}

    def sliced(r: dict) -> dict:
        if r["name"] not in slices:
            slices[r["name"]] = by_slice(log_of(r), unit_of(r))
        return slices[r["name"]]

    print(
        "| rung | outcome | room usual | frames differing | torch's refused allocations (counters"
        " moved) | cuDNN's plans built - the reference's | where the extra plans were built |"
        " torch's recovered lines |\n|---|---|---|---|---|---|---|---|"
    )
    signals: dict[str, list[tuple[str, int | None, float | None, int]]] = {}
    refused = []
    for r in every:
        ref = refs.get(r.get("ladder"))
        if ref is None or r is ref or r.get("outcome") not in ("kept", "moved", "refused"):
            continue
        plans = (r.get("detector") or {}).get("plans")
        extra = None if plans is None else plans - ref["detector"]["plans"]
        mine, theirs = sliced(r), sliced(ref)
        places = []
        for number in sorted(mine["plans"]):
            more = Counter(mine["plans"][number]) - Counter(theirs["plans"].get(number, []))
            if not more:
                continue
            built = [p for p in mine["plans"][number] if p in more]
            free = mine["before"].get(number)
            engines = Counter(tag.split("_k")[0] for tag, _ in more.elements())
            places.append(
                f"slice {number} (+{sum(more.values())}"
                + ("" if free is None else f", {free[2]:.3f} GiB free before it")
                + f"): {', '.join(f'{e} x{c}' for e, c in engines.items())};"
                f" first {built[0][0]} ({built[0][1] / GIB:.3f} GiB),"
                f" last {built[-1][0]} ({built[-1][1] / GIB:.3f} GiB)"
            )
        before, after = r.get("counters") or [None, None]
        moved = (
            None
            if not before or not after
            else {k: after[k] - before[k] for k in after if after[k] != before[k]}
        )
        counted_refusals = (
            None
            if not before or not after
            else sum((after.get(k) or 0) - (before.get(k) or 0) for k in REFUSALS)
        )
        differ = r.get("differ")
        recovered = (r.get("detector") or {}).get("recovered")
        if r["outcome"] == "refused":
            oom = r.get("oom") or {}
            refused.append((r, extra, places, oom, moved))
            continue
        print(
            f"| {r['name']} | {r['outcome']} | {gib(r.get('room_usual'))}"
            f" | {'-' if differ is None else len(differ)}"
            f"{'' if not differ else f' (from {differ[0]})'}"
            f" | {'-' if counted_refusals is None else counted_refusals} ({moved or 'none'})"
            f" | {extra} | {'; '.join(places) or '-'} | {recovered} |"
        )
        signals.setdefault(r["outcome"], []).append(
            (
                r["name"],
                counted_refusals,
                None if extra is None else float(extra),
                recovered or 0,
            )
        )
    print(
        "\nRefused rungs (out of memory, or another memory error), apart:\n"
        "\n| rung | room usual | where | asked GiB | allocated / pool GiB | torch's counters"
        " moved | cuDNN's plans built - the reference's | extra plans before the refusal |"
        "\n|---|---|---|---|---|---|---|---|"
    )
    for r, extra, places, oom, moved in refused:
        asked, held, pool = oom.get("request"), oom.get("allocated"), oom.get("reserved")
        print(
            f"| {r['name']} | {gib(r.get('room_usual'))} | {oom.get('site') or r.get('refused_by')}"
            f" | {'-' if asked is None else f'{asked / GIB:.2f}'}"
            f" | {'-' if held is None or pool is None else f'{held / GIB:.2f} / {pool / GIB:.2f}'}"
            f" | {moved or 'none'} | {extra} | {'; '.join(places) or 'none'} |"
        )
    print("\nWhat separates the moved rungs from the kept ones (a higher figure saying moved):\n")
    kept, moved_rungs = signals.get("kept", []), signals.get("moved", [])
    for index, name in (
        (1, f"torch's count of refused allocations across the pass ({' + '.join(REFUSALS)})"),
        (2, "the cuDNN frontend's plans built in the pass over the reference's (cuDNN's log)"),
        (3, 'torch\'s "recovered from an allocation failure" lines in the pass'),
    ):
        a = [x[index] for x in kept if x[index] is not None]
        b = [x[index] for x in moved_rungs if x[index] is not None]
        if not a or not b:
            print(f"- {name}: kept {sorted(a)}, moved {sorted(b)}: not both seen")
            continue
        if max(a) < min(b):
            threshold = (max(a) + min(b)) / 2
            print(
                f"- {name}: kept {min(a):g} to {max(a):g}, moved {min(b):g} to {max(b):g}:"
                f" SEPARATES, over {threshold:g} said moved: {len(b)} true positives,"
                f" 0 false; {len(a)} true negatives, 0 false"
            )
        else:
            threshold = (min(b) + max(a)) / 2
            false_positives = sum(x > threshold for x in a)
            false_negatives = sum(x <= threshold for x in b)
            print(
                f"- {name}: kept {min(a):g} to {max(a):g}, moved {min(b):g} to {max(b):g}:"
                f" does NOT separate; over {threshold:g}: {len(b) - false_negatives} true"
                f" positives, {false_positives} false; {len(a) - false_positives} true"
                f" negatives, {false_negatives} false"
            )
    print(
        "- the pass's time: not a signal here (its seconds are in every rung's record, for"
        " information): a power-capped card's speed drifts from run to run"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--dir", type=Path, required=True, help="the study's directory")
    parser.add_argument("--python", help="seedvr2x's environment's python, for vae_room.py")
    parser.add_argument("--copy", help="the frozen copy of seedvr2x under study")
    parser.add_argument("--want", help="its tree (git's), checked before every rung")
    parser.add_argument("--models", help="seedvr2x's --model-dir")
    parser.add_argument("--clips", help="the directory of planner_matrix.py's clip-N.mkv")
    parser.add_argument("--lock", help="the GPU's lock file (flock)")
    commands = parser.add_subparsers(dest="command", required=True)
    seeding = commands.add_parser("seed")
    seeding.add_argument("res", type=int, choices=sorted(SHAPES))
    one = commands.add_parser("rung")
    one.add_argument("ladder")
    one.add_argument("--card")
    one.add_argument("--room", type=float)
    commands.add_parser("queue")
    commands.add_parser("table")
    commands.add_parser("detector")
    args = parser.parse_args()
    configure(args)
    if args.command in ("seed", "rung", "queue"):
        needs_gpu_places(args)
    if args.command == "seed":
        return seed(args)
    if args.command == "rung":
        if (args.card is None) == (args.room is None):
            parser.error("rung: --card or --room, one of them")
        return rung(args)
    if args.command == "queue":
        return queue(args)
    if args.command == "detector":
        return detector(args)
    return table(args)


if __name__ == "__main__":
    sys.exit(main())

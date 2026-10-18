#!/usr/bin/env python3
"""The frames' floor of a VAE decode pass by tile size where the GPU's driver gives memory the card
doesn't have (Windows', WSL2 included), so that seedvr2x holds torch to the card under torch's own
allocator with expandable segments (runtime/memory.py: hold, EXPANDABLE). floor.py emulates
smaller cards with vram_cap.py's ballast on a driver that refuses; on a driver that gives memory
the card doesn't have no ballast is refused, so the room is set by torch's limit instead
(vae_room.py --limit): torch held to the pass's peak and a room, the card itself left as it is,
never passed. Each card's frames are judged against that card's own reference only: frames
follow the card.

  seed          the shape (--clip at --resolution, one shot (an empty cut list), --window,
                --encode-tile, --decode-tile, --swap-blocks; floor.py's 1080p shape by default)
                run under seedvr2x's own limit and stopped before its decode (vae_room.py's
                --stop-after windows): DIR/seeds/seed.
  rung T NAME [--room R]
                one decode in tiles of T px, alone: seedvr2x's resume of a copy of the seed whose
                manifest says decode tile T (settings.decode_tile, the pin a resume repeats, and
                plan.decode_tiles' size, the shape it takes), in a process of its own (torch keeps
                the cuDNN engine a convolution fell back to for the life of its process), the
                planner's room at -50 GiB and the DiT's phase unjudged (the plan the same on every
                rung), the cuDNN frontend's log on. Without --room, the tile's reference: seedvr2x's
                own limit (the card's free memory less 0.5 GiB). With it, torch's limit set to the
                reference's decode peak (max_memory_allocated in the unit) and R GiB. Recorded: the
                frames (seedvr2x's CRC-32 of each) against the reference's and the worst PSNR-Y
                where they differ, a refusal and where, the plans cuDNN's frontend built in the
                unit, torch's counters across it, the pass's time (information only), nvidia-smi
                before and after.
  ladder T      the tile's reference, then rungs until the least room that kept the frames and
                the most that lost them, 0.1 GiB apart, have each run twice: from START down by
                STEP while kept, then halved to FINE; a room lost in any run counts as lost; at
                most MOST rungs.
  whole T [R ...] [--plan-room P]
                whole runs (the encode, the windows and the decode in one process, not stopped):
                their reference under seedvr2x's own limit, then one run a room given, torch's limit
                the pass alone's reference peak and R from the plan on; the planner's room P (1.5,
                as shipped, by default; less where 1.5 refuses the pinned tile beside the card's
                other programs).
  table         every run, a table a tile, then what separates the runs that lost their frames
                from those that kept them, by the signals that don't depend on the clock.

Places, all arguments: --dir (records/, work/, seeds/, rungs.jsonl, progress, a file `skip` that
ends a ladder after its running rung); --python (seedvr2x's environment's), vae_room.py beside
this file; --clip; --model-dir (else the models of Hugging Face's cache, as the environment sets
it); --lock (each run under `flock LOCK`). Sizes in GiB (2**30). PROVISIONAL: the ladder's
constants; the room against the reference's peak, as floor.py's "room usual"."""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

GIB = 1024**3
HERE = Path(__file__).resolve().parent
SEGMENTS = "backend:native,expandable_segments:True"  # seedvr2x's memory.EXPANDABLE
ROOM = -50.0  # the planner's room in a rung's process: no plan refused, none moved
START, STEP, FINE = 1.0, 0.2, 0.1  # GiB: PROVISIONAL
MOST = 12  # rungs a ladder takes at the most, its reference apart
BOTTOM, TOP = -0.5, 3.0
PLANS = "EXECUTION_PLAN_DESCRIPTOR"  # the cuDNN frontend's line for a plan built
PLAN_LINE = re.compile(r"EXECUTION_PLAN_DESCRIPTOR : (\S+), .*workSpaceSize: (\d+)")
HELD = re.compile(r"torch held to ([\d.]+) GiB on the GPU: the ([\d.]+) GiB free")
COUNTERS = (
    "num_ooms",
    "num_alloc_retries",
    "num_oom_rejections",
    "num_sync_all_streams",
    "num_device_alloc",
    "num_device_free",
)
MEMORY_ERROR = ("out of memory", "OutOfMemoryError", "CUDNN_STATUS_ALLOC_FAILED", "unable to find")

ARGS = argparse.Namespace()


def place(name: str) -> Path:
    return ARGS.dir / name


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def say(text: str) -> None:
    print(f"held_floor {time.strftime('%H:%M:%S')}: {text}", flush=True)


def smi() -> dict:
    def query(what: str) -> str:
        found = subprocess.run(
            ["nvidia-smi", what, "--format=csv,noheader,nounits"], capture_output=True, text=True
        )
        return found.stdout.strip()

    used = query("--query-gpu=memory.used,memory.free")
    parts = [int(p) for p in used.split(",")] if used else [-1, -1]
    apps = query("--query-compute-apps=pid,process_name,used_memory")
    return {"at": now(), "used_mib": parts[0], "free_mib": parts[1], "apps": apps}


def environment() -> dict:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1")
    env |= {"CUDNN_FRONTEND_LOG_INFO": "1", "CUDNN_FRONTEND_LOG_FILE": "stderr"}
    for name in ("CUDA_VISIBLE_DEVICES", "PYTORCH_ALLOC_CONF"):
        env.pop(name, None)
    for name in ("SEEDVR2X_TESTS_NUMZ_PADDING", "SEEDVR2X_TESTS_NUMZ_DRAW"):
        env.pop(name, None)
    # vae_room.py takes cudaMallocAsync where the environment names no allocator: the setting a
    # held run takes by default is given.
    env["PYTORCH_CUDA_ALLOC_CONF"] = ARGS.allocator
    return env


def theirs(tile: int) -> list[str]:
    found = [str(ARGS.clip), "--resolution", str(ARGS.resolution), "--cuts", str(place("cuts"))]
    found += ["--window", str(ARGS.window), "--encode-tile", str(ARGS.encode_tile)]
    found += ["--decode-tile", str(tile), "--swap-blocks", str(ARGS.swap_blocks)]
    if ARGS.model_dir is not None:
        found += ["--model-dir", str(ARGS.model_dir)]
    return found


def launch(name: str, ours: list[str], tile: int) -> tuple[int, float]:
    argv = [ARGS.python, str(HERE / "vae_room.py"), "run", "--name", name]
    argv += ["--records", str(place("records")), "--work", str(place("work"))]
    argv += ["--tree", ARGS.tree, *ours, "--", *theirs(tile)]
    if ARGS.lock is not None:
        argv = ["flock", str(ARGS.lock), *argv]
    log = place("records") / f"{name}.log"
    started = time.monotonic()
    with log.open("w") as file:
        file.write(f"# {now()} held_floor {name}: {' '.join(argv)}\n")
        file.flush()
        status = subprocess.run(
            argv, stdout=file, stderr=subprocess.STDOUT, env=environment()
        ).returncode
    return status, round(time.monotonic() - started, 1)


def ready() -> None:
    for name in ("records", "work", "seeds"):
        place(name).mkdir(parents=True, exist_ok=True)
    place("cuts").touch()  # an empty cut list: one shot, no shot detector


# ------------------------------------------------------------------ the seed


def seed(_: argparse.Namespace) -> int:
    ready()
    if (place("seeds") / "seed" / "manifest.json").exists():
        say("the seed: there already")
        return 0
    before = smi()
    ours = ["--card", "bare", "--room", "1.5", "--stop-after", "windows", "--keep"]
    status, seconds = launch("seed", ours, ARGS.decode_tile)
    shots = json.loads((place("work") / "seed" / "manifest.json").read_text())["shots"]
    done = [(s["encoded"], s["windows_done"], len(s["windows"])) for s in shots]
    say(f"seed: exit {status} in {seconds} s; shots {done}; nvidia-smi {before} -> {smi()}")
    if status != 130 or not all(s["windows_done"] == len(s["windows"]) for s in shots):
        raise SystemExit("seed: not stopped before its decode with its windows done")
    shutil.rmtree(place("seeds") / "seed", ignore_errors=True)
    shutil.move(str(place("work") / "seed"), str(place("seeds") / "seed"))
    return 0


def seed_for(tile: int) -> Path:
    """The seed copied once a decode tile, its manifest's decode tile set to it: the pin a resume
    repeats (settings.decode_tile) and the shape it takes (plan.decode_tiles' size, its overlap
    the seed's plan's)."""
    path = place("seeds") / f"seed-d{tile}"
    if path.exists():
        return path
    partial = place("seeds") / f"seed-d{tile}.partial"
    shutil.rmtree(partial, ignore_errors=True)
    shutil.copytree(place("seeds") / "seed", partial)
    manifest = partial / "manifest.json"
    content = json.loads(manifest.read_text())
    content["settings"]["decode_tile"] = tile
    content["plan"]["decode_tiles"] = dict(content["plan"]["decode_tiles"], size=tile)
    manifest.write_text(json.dumps(content, indent=2) + "\n")
    partial.rename(path)
    return path


# ------------------------------------------------------------------ a run


def crcs(out: Path) -> list[str]:
    found: list[str] = []
    for path in sorted((out / "checksums").glob("*.crc32")):
        found += path.read_text().split()
    return found


def psnr_y(ours: Path, reference: Path, stats: Path) -> list[float]:
    """Each frame's PSNR-Y of ours against the reference's, by ffmpeg's psnr filter."""
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-i", str(ours), "-i", str(reference), "-lavfi"),
            *(f"[0:v][1:v]psnr=stats_file='{stats}'", "-f", "null", "-"),
        ],
        check=True,
    )
    found = [re.search(r"psnr_y:(\S+)", line) for line in stats.read_text().splitlines()]
    return [math.inf if hit[1] == "inf" else float(hit[1]) for hit in found if hit]


def within(log: Path, unit: int | None) -> list[str]:
    """A run's log between its unit's begin and end lines (the whole log when unit is None)."""
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


def plans_in(log: Path, unit: int | None) -> dict:
    """Within the unit: the plans cuDNN's frontend built, by engine, the largest workspace, and
    the slices that built any (a plan built before a slice's own line, which vae_room.py prints
    at the slice's end, is that slice's)."""
    found: dict = {"plans": 0, "engines": {}, "largest_workspace": 0, "slices_with_plans": []}
    slices, pending = 0, 0
    for line in within(log, unit):
        hit = PLAN_LINE.search(line)
        if hit or PLANS in line:
            found["plans"] += 1
            pending += 1
            if hit:
                engine = re.search(r"_eng(\d+)", hit[1])
                key = f"eng{engine[1]}" if engine else hit[1]
                found["engines"][key] = found["engines"].get(key, 0) + 1
                found["largest_workspace"] = max(found["largest_workspace"], int(hit[2]))
        elif line.startswith("vae_room: slice decode "):
            slices += 1
            if pending:
                found["slices_with_plans"].append([slices, pending])
            pending = 0
    if pending:
        found["slices_with_plans"].append([slices + 1, pending])
    return found


def runs_of(tile: int | None = None) -> list[dict]:
    path = place("rungs.jsonl")
    if not path.exists():
        return []
    found = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in found if tile is None or r["tile"] == tile]


def references(kind: str = "alone") -> dict:
    return {
        r["tile"]: r
        for r in runs_of()
        if r["room"] is None and r["outcome"] == "kept" and r.get("kind", "alone") == kind
    }


def run(tile: int, room: float | None, name: str, kind: str = "alone", plan: float = 1.5) -> dict:
    """One run, its record compared with its reference's, appended to rungs.jsonl. The limit is
    always the pass alone's reference peak and the room; a whole run's frames are judged against
    the whole run's own reference."""
    ready()
    if (place("records") / f"{name}.json").exists():
        raise SystemExit(f"{name}: its record exists already")
    alone = references().get(tile)
    ref = references(kind).get(tile) if kind == "whole" else alone
    if room is not None and (alone is None or ref is None):
        raise SystemExit(f"tile {tile}: no reference yet ({kind})")
    limit = None if room is None or alone is None else alone["peak_allocated"] / GIB + room
    if kind == "whole":
        ours = ["--card", "bare", "--room", str(plan), "--keep"]
    else:
        ours = ["--card", "bare", "--room", str(ROOM), "--dit-unjudged", "--keep"]
        ours += ["--seed-dir", str(seed_for(tile))]
    if limit is not None:
        ours += ["--limit", f"{limit:.4f}"]
    before = smi()
    status, seconds = launch(name, ours, tile)
    after = smi()
    log = place("records") / f"{name}.log"
    path = place("records") / f"{name}.json"
    record = json.loads(path.read_text()) if path.exists() else {}
    units = [u for u in record.get("units", []) if u.get("phase") == "decode"]
    unit = units[-1] if units else None
    text = log.read_text(errors="replace")
    held = HELD.search(text)
    limits = record.get("limits") or {}
    found: dict = {
        "name": name,
        "kind": kind,
        "plan_room": plan if kind == "whole" else ROOM,
        "tile": tile,
        "room": room,
        "limit": limit,
        "seedvr2x_limit": limits["held"] / GIB if limits else (float(held[1]) if held else None),
        "began": before["at"],
        "exit": status,
        "seconds": seconds,
        "smi_before": before,
        "smi_after": after,
        "record": str(path),
        "log": str(log),
        "tree": ARGS.tree,
    }
    if unit is not None:
        b, a = unit.get("counters_before") or {}, unit.get("counters_after") or {}
        found |= {
            "unit": unit["number"],
            "peak_allocated": unit.get("peak_allocated"),
            "peak_reserved": unit.get("peak_reserved"),
            "pass_seconds": unit.get("seconds"),
            "card_free_least": unit.get("card_free_least"),
            "slices": len(unit.get("slices", [])),
            "counters": {
                k: a[k] - b[k] if a.get(k) is not None and b.get(k) is not None else None
                for k in COUNTERS
            },
        }
    found |= plans_in(log, found.get("unit"))
    oom = record.get("oom")
    if oom:
        found["oom"] = {k: oom.get(k) for k in ("caught_at", "request", "slice")}
        found["oom"] |= {k: oom.get(k) for k in ("allocated", "reserved")}
        found["oom"]["message"] = (oom.get("message") or "")[:300]
    out = place("work") / name
    ours_crc = crcs(out)
    found["frames"] = len(ours_crc)
    if room is None:
        found["outcome"] = "kept" if status == 0 and ours_crc else "failed"
        found["crc"] = ours_crc
    else:
        assert ref is not None
        if oom or (status != 0 and any(m in text for m in MEMORY_ERROR)):
            found["outcome"] = "refused"
        elif status == 0 and len(ours_crc) == len(ref["crc"]):
            differ = [
                i for i, (x, y) in enumerate(zip(ours_crc, ref["crc"], strict=True)) if x != y
            ]
            found |= {"differ": len(differ), "first_differ": differ[0] if differ else None}
            found["outcome"] = "moved" if differ else "kept"
            if differ:
                values: list[float] = []
                pairs = zip(
                    sorted(out.glob("seg_*.mkv")),
                    sorted((place("work") / ref["name"]).glob("seg_*.mkv")),
                    strict=True,
                )
                for ours_seg, ref_seg in pairs:
                    values += psnr_y(ours_seg, ref_seg, place("records") / f"{name}.psnr")
                finite = [v for v in values if math.isfinite(v)]
                found["worst_psnr_y"] = min(finite) if finite else None
                found["crc"] = ours_crc
        else:
            found["outcome"] = "failed"
        if found["outcome"] != "moved":
            shutil.rmtree(out, ignore_errors=True)
        found["plans_over_ref"] = found["plans"] - ref["plans"]
        if found.get("pass_seconds") and ref.get("pass_seconds"):
            found["pass_over_ref"] = round(found["pass_seconds"] / ref["pass_seconds"], 3)
    with place("rungs.jsonl").open("a") as file:
        file.write(json.dumps(found) + "\n")
    line = (
        f"{found['began']} {name} ({kind}) tile {tile} room {room} limit {limit} (seedvr2x's"
        f" {found['seedvr2x_limit']}) -> {found['outcome']}; frames {found['frames']} differ"
        f" {found.get('differ')} from {found.get('first_differ')}, worst PSNR-Y"
        f" {found.get('worst_psnr_y')}; plans {found['plans']} ({found.get('plans_over_ref')} over"
        f" the reference's) {found['engines']}; counters {found.get('counters')}; pass"
        f" {found.get('pass_seconds')} s ({found.get('pass_over_ref')} of the reference's); exit"
        f" {status} in {seconds} s; nvidia-smi used {before['used_mib']} -> {after['used_mib']}"
        f" MiB; refused {found.get('oom')}"
    )
    with place("progress").open("a") as file:
        file.write(line + "\n")
    say(line)
    return found


def rung(args: argparse.Namespace) -> int:
    run(args.tile, args.room, args.name)
    return 0


# ------------------------------------------------------------------ the ladder


def decided(tile: int) -> tuple[float | None, float | None, dict]:
    """The least room kept in every run of it, the most lost in any, and the runs by room (the
    pass alone's)."""
    runs: dict = {}
    for r in runs_of(tile):
        if r["room"] is None or r["outcome"] == "failed" or r.get("kind", "alone") != "alone":
            continue
        runs.setdefault(round(r["room"], 2), []).append(r["outcome"])
    kept = [room for room, seen in runs.items() if all(o == "kept" for o in seen)]
    lost = [room for room, seen in runs.items() if any(o != "kept" for o in seen)]
    return (min(kept) if kept else None), (max(lost) if lost else None), runs


def next_room(tile: int) -> float | None:
    least, most, runs = decided(tile)
    if sum(len(v) for v in runs.values()) >= MOST:
        return None
    if least is None and most is None:
        return START
    if most is None:
        assert least is not None
        room = round(least - STEP, 2)
        return room if room >= BOTTOM else None
    if least is not None and least < most:
        # A room kept under one lost: the bracket is the lost one and the next kept above it.
        above = [room for room in runs if room > most and all(o == "kept" for o in runs[room])]
        least = min(above) if above else None
    if least is None:
        room = round(most + STEP, 2)
        return room if room <= TOP else None
    if round(least - most, 2) > FINE + 1e-9:
        middle = round(math.floor((least + most) / 2 / FINE + 1e-9) * FINE, 2)
        return middle if most < middle < least else round(most + FINE, 2)
    for room in (least, most):
        if len(runs[room]) < 2:
            return room
    return None


def ladder(args: argparse.Namespace) -> int:
    tile = args.tile
    if tile not in references():
        if run(tile, None, f"ref-d{tile}")["outcome"] != "kept":
            raise SystemExit(f"ref-d{tile}: the reference failed")
    while not place("skip").exists():
        room = next_room(tile)
        if room is None:
            least, most, runs = decided(tile)
            say(f"tile {tile} done: least kept {least}, most lost {most}; runs {runs}")
            return 0
        count = sum(
            1
            for r in runs_of(tile)
            if r.get("kind", "alone") == "alone"
            and r["room"] is not None
            and round(r["room"], 2) == room
        )
        run(tile, room, f"d{tile}-r{room:.2f}-{count + 1}")
    say(f"{place('skip')} exists: the ladder ends")
    return 0


def whole(args: argparse.Namespace) -> int:
    tile = args.tile
    if tile not in references("whole"):
        found = run(tile, None, f"whole-ref-d{tile}", "whole", args.plan_room)
        alone = references().get(tile)
        if alone is not None:
            say(f"whole-ref-d{tile}: the pass alone's frames: {found.get('crc') == alone['crc']}")
    for room in args.rooms:
        if place("skip").exists():
            say(f"{place('skip')} exists: the whole runs end")
            return 0
        count = sum(
            1
            for r in runs_of(tile)
            if r.get("kind") == "whole" and r["room"] is not None and round(r["room"], 2) == room
        )
        run(tile, room, f"whole-d{tile}-r{room:.2f}-{count + 1}", "whole", args.plan_room)
    return 0


# ------------------------------------------------------------------ tables


def gib(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value / GIB:.{digits}f}"


def table(_: argparse.Namespace) -> int:
    head = "| run | room | limit | outcome | frames off | first | worst PSNR-Y | plans (over ref)"
    head += " | eng28 | num_ooms | num_alloc_retries | pass s (of ref's) | peak / reserved |"
    for tile in sorted({r["tile"] for r in runs_of()}):
        least, most, runs = decided(tile)
        print(f"\n### decode {tile} px: least kept {least}, most lost {most}; runs {runs}\n")
        print(head)
        print("|" + " --- |" * 13)
        for r in runs_of(tile):
            c = r.get("counters") or {}
            limit = "-" if r["limit"] is None else f"{r['limit']:.3f}"
            print(
                f"| {r['name']} | {r['room']} | {limit} | {r['outcome']} |"
                f" {r.get('differ', '-')}/{r['frames']} | {r.get('first_differ', '-')} |"
                f" {r.get('worst_psnr_y', '-')} | {r['plans']} ({r.get('plans_over_ref', '-')}) |"
                f" {r['engines'].get('eng28', 0)} | {c.get('num_ooms')} |"
                f" {c.get('num_alloc_retries')} | {r.get('pass_seconds')}"
                f" ({r.get('pass_over_ref', '-')}) | {gib(r.get('peak_allocated'))} /"
                f" {gib(r.get('peak_reserved'))} |"
            )
    print("\nWhat separates the runs that lost their frames from those that kept them (rooms):")
    groups: dict = {}
    for r in runs_of():
        if r["room"] is None or r["outcome"] == "failed":
            continue
        c = r.get("counters") or {}
        seen = groups.setdefault(r["outcome"], {"runs": 0, "ooms": [], "retries": [], "plans": []})
        seen["runs"] += 1
        seen["ooms"].append(c.get("num_ooms"))
        seen["retries"].append(c.get("num_alloc_retries"))
        seen["plans"].append(r.get("plans_over_ref"))
    for outcome, seen in groups.items():
        print(
            f"- {outcome}: {seen['runs']} runs; torch's num_ooms {sorted(seen['ooms'])};"
            f" num_alloc_retries {sorted(seen['retries'])}; cuDNN's plans built over the"
            f" reference's {sorted(seen['plans'])}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable, help="seedvr2x's environment's")
    parser.add_argument("--clip", type=Path, help="the shape's source (floor.py's clip-53)")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--lock", type=Path, help="each run under flock LOCK")
    parser.add_argument("--tree", default="not told", help="the code's tree, recorded")
    parser.add_argument("--allocator", default=SEGMENTS, help="PYTORCH_CUDA_ALLOC_CONF")
    parser.add_argument("--resolution", type=int, default=1080)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--encode-tile", type=int, default=768)
    parser.add_argument("--decode-tile", type=int, default=576, help="the seed's")
    parser.add_argument("--swap-blocks", type=int, default=36)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("seed")
    one = commands.add_parser("rung")
    one.add_argument("tile", type=int)
    one.add_argument("name")
    one.add_argument("--room", type=float)
    many = commands.add_parser("ladder")
    many.add_argument("tile", type=int)
    entire = commands.add_parser("whole")
    entire.add_argument("tile", type=int)
    entire.add_argument("rooms", type=float, nargs="*")
    entire.add_argument("--plan-room", type=float, default=1.5)
    commands.add_parser("table")
    global ARGS
    ARGS = parser.parse_args()
    if ARGS.command in ("seed", "rung", "ladder", "whole") and ARGS.clip is None:
        parser.error("--clip is required to run")
    commands_ = {"seed": seed, "rung": rung, "ladder": ladder, "whole": whole, "table": table}
    return commands_[ARGS.command](ARGS)


if __name__ == "__main__":
    sys.exit(main())

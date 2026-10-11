#!/usr/bin/env python3
"""The memory planner's matrix (seedvr2x, milestone 3): seedvr2x run on cards of N GB emulated by
vram_cap.py's ballast, the planner's own plan each time; each phase's predicted peak (the
manifest's plan.memory) against the measured one (the units' "VRAM peak" log lines); then the same
plan pinned on a card one notch under, which the planner must refuse; and, with --resume-check, a
job stopped after its first encode, resumed on a smaller card, where its recorded shape must no
longer fit, then on a larger one, where it must finish with the frames of the uninterrupted run.

Run with the python of seedvr2x's environment (it imports seedvr2x's planner to choose each
clip's length), one GPU job at a time:
  python planner_matrix.py --out DIR --source CLIP --vram-cap vram_cap.py --model-dir DIR \
      --case 1080:24:2w --case 2160:bare:w --resume-check 1080:16:2w:10:24
A case is RES:CARD:MODE. RES is the output's short side, the source taken as 16:9. CARD is a size
in GB, or `bare` (no emulation). MODE picks the clip's length from W, the longest shot the card's
plan runs in one window: 2w = 2W - 2 latents (two windows at the cap), w1 = W + 1 (two windows of
about half), w = W (one window at the cap). The clips are cut from CLIP, from its frame --first
on, one shot each (--cut-threshold 1). A case whose JSON exists in DIR is skipped: a batch can be
run again after an interruption. DIR/extra-cases.txt, when there, names more cases, run last.
Results: DIR/<case>.json and .log, DIR/table.md.

EMULATED and BARE are what seedvr2x found free on the RTX PRO 6000 this was written on
(research/docs/planner-limits.md's N - 1.44 is numz's): they only choose the clips' lengths."""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

GIB = 1024**3
# What seedvr2x reads free on a card emulated at N GB, N - EMULATED, and on the bare 96 GB card,
# in GiB, the shot detector having run and a first allocation in place (seedvr2x/DESIGN.md,
# Memory planner, Budget; measured 2026-10-11): only to choose the clip's length. The matrices of
# 2026-10-11 chose theirs with the day before's readings, 1.4285 and 94.2746.
EMULATED = 1.46
BARE = 94.24
UNIT = {
    "encode": re.compile(r"encoded in ([\d.]+) s, VRAM peak ([\d.]+) GiB"),
    "dit": re.compile(r"window \d+/\d+, latents \d+ to \d+, in ([\d.]+) s, VRAM peak ([\d.]+) GiB"),
    "decode": re.compile(
        r"decoded in ([\d.]+) s, VRAM peak ([\d.]+) GiB; RAM ([\d.]+) GiB, peak ([\d.]+) GiB"
    ),
}


def free_of(card: str) -> int:
    return round((BARE if card == "bare" else float(card) - EMULATED) * GIB)


def planned(free: int, target: tuple[int, int], latents: int, pins=None):
    from seedvr2x.runtime import planner
    from seedvr2x.runtime.job import Shot

    shots = [Shot(0, 4 * latents - 3)]
    return planner.plan(free, target, shots, pins or planner.Pins(), None, split=True)


def one_window(free: int, target: tuple[int, int]) -> int:
    """The longest shot, in latents, the planner runs in one window with `free` bytes free."""
    from seedvr2x.runtime.planner import PlanError

    for latents in range(160, 0, -1):
        try:
            plan, peaks = planned(free, target, latents)
        except PlanError:
            continue
        if peaks.record()["longest_window"] == latents:
            return latents
    raise SystemExit(f"nothing fits {free / GIB:.2f} GiB at {target}")


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


def run(
    args, name: str, card: str, options: list[str], stop_at: str | None = None
) -> tuple[int, float, str]:
    """seedvr2x run with options, under the emulation unless the card is bare: its exit status,
    its duration in seconds and its log's text. With stop_at, a Ctrl-C (SIGINT, once) is sent
    when the log first matches it: the run ends after the unit in progress (exit 130)."""
    entry = Path(sys.executable).parent / "seedvr2x"
    command = [sys.executable]
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    if card != "bare":
        command.append(str(args.vram_cap))
        env["VRAM_CAP_GIB"] = card
    command += [str(entry), *options, "--model-dir", str(args.model_dir)]
    log = args.out / f"{name}.log"
    started = time.monotonic()
    with log.open("w") as out:
        out.write(f"# VRAM_CAP_GIB={env.get('VRAM_CAP_GIB', 'none')} {' '.join(command)}\n")
        out.flush()
        process = subprocess.Popen(command, env=env, stdout=out, stderr=subprocess.STDOUT)
        asked = stop_at is None
        while process.poll() is None:
            time.sleep(0.5)
            if not asked and re.search(stop_at, log.read_text(errors="replace")):
                process.send_signal(signal.SIGINT)
                asked = True
        status = process.returncode
    return status, time.monotonic() - started, log.read_text(errors="replace")


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
    status, seconds, text = run(args, name, card, options)
    result |= {"exit": status, "seconds": round(seconds), **measured(text)}
    manifest = out / "manifest.json"
    if manifest.exists():
        result["plan"] = json.loads(manifest.read_text())["plan"]
    print(f"{name}: exit {status} in {seconds:.0f} s", flush=True)
    # The same plan, pinned, on a card one notch under: the planner must refuse it.
    if status == 0 and "plan" in result:
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
            status, seconds, text = run(args, f"{name}-under", f"{under[0]:g}", options)
            errors = [line for line in text.splitlines() if " ERROR " in line]
            result["under"] = {"card": under[0], "notch": size - under[0], "expected": under[1]}
            result["under"] |= {"exit": status, "said": errors[:2], "seconds": round(seconds)}
            print(f"{name}: one notch under ({under[0]:g} GB): exit {status}", flush=True)
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
    status, seconds, text = run(args, f"{name}-resumed-1", card, options, r"encoded in [\d.]+ s")
    said = [line for line in text.splitlines() if "stopp" in line.lower()][-2:]
    record["stopped"] = {"exit": status, "seconds": round(seconds), "said": said}
    record["stopped"]["units"] = measured(text)["units"]
    manifest = out / "manifest.json"
    before = manifest.read_bytes() if manifest.exists() else None
    status, seconds, text = run(args, f"{name}-resumed-2", lower, options)
    errors = [line for line in text.splitlines() if " ERROR " in line][:2]
    unchanged = before is not None and manifest.read_bytes() == before
    record["lower"] = {
        "card": lower,
        "exit": status,
        "said": errors,
        "manifest_unchanged": unchanged,
    }
    status, seconds, text = run(args, f"{name}-resumed-3", higher, options)
    record["higher"] = {"card": higher, "exit": status, "seconds": round(seconds), **measured(text)}
    if manifest.exists():
        record["plan"] = json.loads(manifest.read_text())["plan"]
    ours = {p.name: p.read_bytes() for p in sorted((out / "checksums").glob("*.crc32"))}
    theirs = {
        p.name: p.read_bytes() for p in sorted((args.out / name / "checksums").glob("*.crc32"))
    }
    record["checksums"] = {"files": len(ours), "identical": bool(ours) and ours == theirs}
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
        " exit, s | under: card, exit |",
        "|" + "---|" * 14,
    ]
    for r in results:
        plan = r.get("plan")
        if plan is None:
            lines.append(
                f"| {r['case']} | {r['frames']} ({r['latents']}) | no plan | | | | | | | | | | {r['exit']}, {r['seconds']} | |"
            )
            continue
        memory = plan["memory"]
        gib = lambda key: f"{memory[key] / GIB:.2f}"  # noqa: E731
        tiles = " / ".join(
            str(plan[k]["size"]) if plan[k] else "none" for k in ("encode_tiles", "decode_tiles")
        )
        units = r["units"]
        under = r.get("under")
        lines.append(
            f"| {r['case']} | {r['frames']} ({r['latents']}) | {plan['window']} / {memory['longest_window']}"
            f" | {memory['swapped_blocks']}, {'in' if memory['vae_in_dit'] else 'away'} | {tiles}"
            f" | {memory['resident_encode']} / {memory['resident_decode']} | {gib('free')} / {gib('budget')}"
            f" | {gib('encode_peak')} / {units['encode']['peak']} | {gib('dit_peak')} / {units['dit']['peak']}"
            f" | {gib('decode_peak')} / {units['decode']['peak']} | {memory['bound']} | {r['trims']}"
            f" | {r['exit']}, {r['seconds']} | {'' if under is None else f'{under["card"]:g}, {under["exit"]}'} |"
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
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
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
    for spec in args.case:
        try:
            results.append(case(args, spec))
        except Exception as error:  # a case's failure mustn't end the batch
            print(f"{spec}: FAILED: {error!r}", flush=True)
    for spec in args.resume_check:
        try:
            resume_check(args, spec)
        except Exception as error:
            print(f"{spec}: resume check FAILED: {error!r}", flush=True)
    # More cases, one a line in <out>/extra-cases.txt, run after the others in the same run: the
    # file may be written while the first ones run.
    extra = args.out / "extra-cases.txt"
    if extra.exists():
        for spec in extra.read_text().split():
            try:
                case(args, spec)
            except Exception as error:
                print(f"{spec}: FAILED: {error!r}", flush=True)
    done = [json.loads(path.read_text()) for path in sorted(args.out.glob("*.json"))]
    done = [result for result in done if "units" in result]
    (args.out / "table.md").write_text(table(done) + "\n")
    print(table(done), flush=True)


if __name__ == "__main__":
    main()

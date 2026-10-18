#!/usr/bin/env python3
"""The frames' floor of whole runs (seedvr2x, milestone 3): how little memory a whole job's decode
can be left beyond its peak before its frames leave those of the same job run with room to spare,
beside the floor of the same decode run alone (floor.py). A pass alone is a process of its own
that enters its decode with the models only; a whole run enters it after its encode and its
windows, with what the allocator's pool kept of them (vae-room.md).

A rung is one WHOLE job (the encode, the DiT's windows and the decode in one process): seedvr2x's
own command line by vae_room.py run (beside this file), without its stop or a seed, under
vram_cap.py's ballast (--card N), in its own process, one hold of the GPU's lock. As floor.py's
passes alone run: the planner's room set low in that process (ROOM: no plan refused, the pinned
tiles never moved) and the DiT's phase unjudged (vae_room.py run --dit-unjudged), so that every
rung of a ladder runs the reference's plan (checked: the window, the tiles, the swapped blocks,
what of the DiT the VAE's phases keep, the VAE in the DiT's phase); the decode tile pinned; the
cuDNN frontend's log on (CUDNN_FRONTEND_LOG_INFO=1: the execution plans built, by unit). In the
rung's process this script adds two probes to vae_room.py's (`run`): the md5 of the shot's
latent once its encode has kept it, and of each window the DiT gave once kept, read back on the
CPU from the files seedvr2x writes (runtime/units.py, Units.save_latent and save_window), which a
finished job removes: a rung whose latent and windows are the reference's moved no frame but by
its decode.

A ladder is named wholeRES-TILE: 1080 or 2160, the decode tile's side in pixels (a square, cut
off at 1080p's 1,104 rows where larger) or `untiled` (--decode-tile 0: every rung checked to have
run the untiled decode, its plan with no decode tiles and the decode in one stream's slices, as
many as a tiled pass runs a tile; else the rung fails). Its pass alone is a ladder of floor.py's,
given as LADDER=PASS in --ladders and read from the --alone studies (a floor.py study's
directory, its rungs.jsonl and refs/; or a rungs file, refs/ beside it), read only.

The shapes are floor.py's: 1080 = clip-53 (53 frames, window 8, encode tiles 768); 2160 = clip-17
(17 frames, window 5, encode tiles 1408); 36 blocks swapped; an empty cut list (one shot).

A rung's room is the free memory its plan was made in (the run's own plan, vae_room.py's
"planned") less its decode's measured peak (torch's max allocated in the decode unit); "usual",
that free memory less the reference's decode peak, on which the ladder steps; "by the pass
alone", that free memory less the pass alone's reference peak, so that both floors stand on one
measure. Judged by every frame's CRC-32 (seedvr2x's checksums) against the tile's reference, the
same whole job on the bare card: kept, moved (how many, from which, the worst PSNR-Y by ffmpeg's
psnr), refused (out of memory or another memory error: a loss, counted apart) or failed (anything
else: the ladder ends). The ladder: the reference, then the pass alone's least room kept (START
where none is read); down by FINE while the frames hold, up by FINE while they don't (after four
steps one way the step doubles, then the bracket is halved back to FINE); then the least room
kept on every run and the most room lost on any run, each run a second time on the same card. A
rung's card is the reference's peak, the target room and the card's offset (card less free
memory: the median of the study's rungs so far, else OFFSET). At most MOST_RUNGS rungs a ladder.

Also recorded each rung: the plans cuDNN's frontend built in each unit against the reference's
same unit, torch's allocator counters by unit, torch's "recovered from an allocation failure"
lines, the pool at the decode's start (what torch had allocated and its pool reserved, the
driver's free memory), nvidia-smi's compute processes before it; the seconds, for information
only (on a power-capped card a run's time drifts).

  run ...       in a rung's process: vae_room.py run's arguments, its probes and these two.
  plan          no GPU: the code's tree, each ladder's pass alone and its next rung's command.
  rung LADDER --card CARD
                one rung by hand (CARD bare: the reference); its pass alone from --pass.
  queue         the ladders of --ladders in their order, each rung decided from the records;
                DIR/pause holds it before the next rung, DIR/skip ends it after the running one,
                a run never interrupted; --wait-for FILE: no rung before FILE exists; DIR/done (or
                --done's name) at the end, with why.
  table         per ladder the whole run's least room kept and most room lost beside the pass
                alone's, the references' peaks and frames, the pool at the decode's start; every
                rung; the pairs of runs at one card. --compact: each ladder's rungs on one line;
                --label LADDER=TEXT: a tile's name in the tables.

--rungs FILE (several, or a glob) reads other rungs' files besides the study's own: the records
of 2026-10-17's whole runs, made by this script's first form, as they are (the refusal's site
under `site`, the reference's frames against the pass alone's under `pass_reference_differ`, no
pool in the rung's record: read from its vae_room.py record, as for every rung). --alone
PATH=NOTE says NOTE beside a reference's frames against that study's pass alone (another code's).

The study's directory (--dir) holds rungs.jsonl (a record a rung), progress, records/ (vae_room's
record and log of each rung), refs/ (each reference's segments and checksums), work/ and the tree
check's scratch. --python is seedvr2x's environment's python, --copy the frozen copy of seedvr2x
under study (its tree --want, checked before every rung, nothing written in it: vae_room.tree_of),
--models seedvr2x's --model-dir, --clips the directory of planner_matrix.py's clip-N.mkv, --lock
the GPU's lock (flock). Run with any python 3.11+ (stdlib only) for the driver, ffmpeg on PATH,
CUDA_VISIBLE_DEVICES empty: a rung's own process gets the card."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
GIB = 1024**3
SHAPES = {
    1080: {"clip": "clip-53.mkv", "frames": 53, "window": 8, "encode": 768},
    2160: {"clip": "clip-17.mkv", "frames": 17, "window": 5, "encode": 1408},
}
PADDED = {1080: (1104, 1920), 2160: (2192, 3840)}
SWAPPED = 36
ROOM = -50.0  # the planner's room in the rungs' process: no plan refused, none moved
FINE = 0.1  # GiB, the ladder's step
MOST_RUNGS = 16  # a ladder's rungs at the most, its reference included
TOP, BOTTOM = 4.0, -1.0  # the rooms a ladder goes no further than
OFFSET = 1.49  # card less free memory, GiB, before a rung measured it (floor.py's)
START = 1.0  # the first room where no pass alone's floor is read
POLL = 30  # seconds between two looks at the pause file or at --wait-for's
PLANS = "EXECUTION_PLAN_DESCRIPTOR"  # the cuDNN frontend's line for a plan built
TRIM = "recovered from an allocation failure"
MEMORY_ERROR = (
    r"out of memory|unable to find an engine|CUDNN_STATUS_ALLOC_FAILED"
    r"|CUDNN_STATUS_EXECUTION_FAILED|CUBLAS_STATUS_ALLOC_FAILED|cudaErrorMemoryAllocation"
    r"|CUDA_ERROR_OUT_OF_MEMORY"
)
SAME_PLAN = (
    "window",
    "encode_tiles",
    "decode_tiles",
    "swapped",
    "resident_encode",
    "resident_decode",
    "vae_in_dit",
)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def gib(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def vae_room():
    """vae_room.py, beside this file."""
    spec = importlib.util.spec_from_file_location("vae_room", HERE / "vae_room.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["vae_room"] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ a rung's process


def probed_run(argv: list[str]) -> int:
    """vae_room.py run with ARGV, two probes more, installed after vae_room's own (install): torch
    and seedvr2x imported in vae_room's order, vram_cap's ballast first."""
    room = vae_room()
    installed = room.install

    def digest(path: Path) -> tuple[str, list[int], str]:
        import torch

        tensor = torch.load(path, map_location="cpu", weights_only=True)
        data = tensor.contiguous().view(torch.uint8).numpy().tobytes()
        return hashlib.md5(data).hexdigest(), list(tensor.shape), str(tensor.dtype)

    def tell(what: str, path: Path) -> None:
        try:
            md5, shape, dtype = digest(path)
            room.say(f"whole_floor: {what} md5 {md5} shape {shape} {dtype}")
        except Exception as error:  # a probe never ends the run it watches
            room.say(f"whole_floor: {what} NOT READ ({error!r})")

    def install(probes, args) -> list[str]:
        done = installed(probes, args)
        from seedvr2x.runtime import units

        save_latent, save_window = units.Units.save_latent, units.Units.save_window

        def latent_kept(self, shot, *given, **named):
            kept = save_latent(self, shot, *given, **named)
            tell(f"shot {shot} latent", self.shot_directory(shot) / "latent.pt")
            return kept

        def window_kept(self, shot, window, *given, **named):
            result = save_window(self, shot, window, *given, **named)
            kept = self.shot_directory(shot) / f"window_{window:04d}.pt"
            tell(f"shot {shot} window {window}", kept)
            return result

        units.Units.save_latent, units.Units.save_window = latent_kept, window_kept
        return [*done, "the md5 of each latent and window kept, read back on the CPU"]

    room.install = install
    sys.argv = [str(HERE / "vae_room.py"), "run", *argv]
    return room.main()


# ------------------------------------------------------------------ the study


class Study:
    """The study's files and settings, from the command line."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.dir: Path = args.dir
        self.records, self.work, self.refs = (
            self.dir / "records",
            self.dir / "work",
            self.dir / "refs",
        )
        self.rungs_file, self.progress_file = self.dir / "rungs.jsonl", self.dir / "progress"
        self.pause, self.skip = self.dir / "pause", self.dir / "skip"
        self.cuts = self.dir / "no-cuts.txt"
        self.passes = parse_ladders(getattr(args, "ladders", None) or "")
        if getattr(args, "pass_ladder", None):
            self.passes[args.ladder] = args.pass_ladder
        self.labels = dict(item.partition("=")[::2] for item in getattr(args, "label", None) or [])
        self._pools: dict[str, dict | None] = {}

    def need(self, *names: str) -> None:
        missing = [f"--{name}" for name in names if not getattr(self.args, name)]
        if missing:
            raise SystemExit(f"{' '.join(missing)} needed here")

    def note(self, line: str) -> None:
        """A line in the progress file ('#' lines: what the queue did) and on the screen."""
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.progress_file.open("a") as file:
            file.write(line + "\n")
        print(f"{now()} whole_floor: {line}", flush=True)

    # -- records

    def rung_files(self) -> list[Path]:
        """The rungs' files read: --rungs's paths or globs in their order, then the study's own
        rungs.jsonl (which a queue writes), each once."""
        found: list[Path] = []
        for given in getattr(self.args, "rungs", None) or []:
            path = Path(given)
            matches = (
                sorted(path.parent.glob(path.name)) if any(c in given for c in "*?[") else [path]
            )
            found += [
                match for match in matches if match.resolve() not in {f.resolve() for f in found}
            ]
        if self.rungs_file.resolve() not in {f.resolve() for f in found}:
            found.append(self.rungs_file)
        return found

    def rungs(self, ladder: str | None = None) -> list[dict]:
        """Every rung of the files read, of one ladder or all, in the files' order, each as this
        script records it whatever wrote it (normal)."""
        rows = [normal(r, path) for path in self.rung_files() for r in read(path)]
        return [r for r in rows if ladder is None or r.get("ladder") == ladder]

    def reference(self, ladder: str) -> dict | None:
        return next((r for r in self.rungs(ladder) if r["card"] == "bare" and r.get("ok")), None)

    def offset(self) -> float:
        seen = [r["card_offset"] for r in self.rungs() if r.get("card_offset") is not None]
        return round(statistics.median(seen[-9:]), 4) if seen else OFFSET

    def alone(self, ladder: str) -> dict:
        """The ladder's pass alone (read only): its reference's peak and checksums' place, the
        least room kept and the most lost, from the first --alone study that holds it."""
        name = self.passes.get(ladder)
        found: dict = {"ladder": name, "study": None, "peak": None, "crcs": None}
        found |= {"kept": None, "lost": None, "rungs": 0, "note": None, "reference": None}
        for given in self.args.alone or []:
            # PATH[=NOTE]: NOTE said beside a reference's frames against this study's pass alone
            # (another code's, its frames other).
            text, _, note = given.partition("=")
            path = Path(text)
            rungs, refs = (
                (path, path.parent / "refs")
                if path.is_file()
                else (path / "rungs.jsonl", path / "refs")
            )
            rows = [r for r in read(rungs) if name is not None and r.get("ladder") == name]
            if not rows:
                continue
            reference = next((r for r in rows if r.get("card") == "bare" and r.get("ok")), None)
            kept, lost, _ = bracket(rows)
            found |= {
                "study": str(rungs),
                "note": note or None,
                "reference": None if reference is None else normal(reference, rungs),
                "peak": None if reference is None else reference.get("peak"),
                "crcs": str(refs / name / "checksums"),
                "kept": None
                if kept is None
                else {k: kept[k] for k in ("card", "room", "outcomes")},
                "lost": None
                if lost is None
                else {k: lost[k] for k in ("card", "room", "outcomes")},
                "rungs": len(rows),
            }
            break
        return found

    def tree(self) -> str:
        """The copy's tree by vae_room.tree_of, in a scratch of this process's own (a plan beside a
        queue never shares an index), removed after."""
        scratch = self.dir / "tree-check" / str(os.getpid())
        try:
            return vae_room().tree_of(Path(self.args.copy), scratch)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def fit(self) -> float | None:
        """The planner's untiled decode fit of 1080p's padded frame (planner._fit), GiB, read once
        by the code's own python on the CPU (fits.json)."""
        path = self.dir / "fits.json"
        if not path.exists():
            script = (
                "import json; from seedvr2x.runtime import planner; "
                f"print(json.dumps(planner._fit({PADDED[1080]}, None, 1)))"
            )
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
            said = subprocess.run(
                [self.args.python, "-c", script], capture_output=True, text=True, env=env
            )
            try:
                path.write_text(json.dumps({"decode": float(said.stdout.split()[-1]) / GIB}) + "\n")
            except (IndexError, ValueError):
                return None
        return json.loads(path.read_text()).get("decode")

    # -- one rung

    def command(self, ladder: str, name: str, card: str) -> list[str]:
        res, tile = parse(ladder)
        shape, args = SHAPES[res], self.args
        ours = ["--name", name, "--records", str(self.records), "--work", str(self.work)]
        ours += ["--tree", args.want, "--vram-cap", str(HERE / "vram_cap.py"), "--card", card]
        ours += ["--room", str(ROOM), "--dit-unjudged", "--keep"]
        theirs = [str(Path(args.clips) / shape["clip"]), "--resolution", str(res)]
        theirs += ["--cuts", str(self.cuts), "--window", str(shape["window"])]
        theirs += ["--encode-tile", str(shape["encode"]), "--decode-tile", str(tile)]
        theirs += ["--swap-blocks", str(SWAPPED), "--model-dir", args.models]
        own = [args.python, str(Path(__file__).resolve()), "run", *ours, "--", *theirs]
        return ["flock", args.lock, *own]

    def run_rung(self, ladder: str, card: str, target: float | None, why: str, name: str) -> dict:
        """One rung: run, read its record, compare it with the ladder's reference, record it."""
        res, tile = parse(ladder)
        frames = SHAPES[res]["frames"]
        for directory in (self.records, self.work, self.refs):
            directory.mkdir(parents=True, exist_ok=True)
        self.cuts.touch()
        if (self.records / f"{name}.json").exists():
            raise SystemExit(f"{name}: its record exists already")
        out = self.work / name
        shutil.rmtree(out, ignore_errors=True)
        held = foreign()
        argv = self.command(ladder, name, card)
        log = self.records / f"{name}.log"
        began, started = now(), time.monotonic()
        with log.open("w") as file:
            file.write(f"# {began} whole_floor rung {name} ({why}): {' '.join(argv)}\n")
            file.write(f"# foreign: {held}\n")
            file.flush()
            status = subprocess.run(
                argv, stdout=file, stderr=subprocess.STDOUT, env=environment()
            ).returncode
        record_path = self.records / f"{name}.json"
        record = json.loads(record_path.read_text()) if record_path.exists() else None
        found: dict = {
            "name": name,
            "ladder": ladder,
            "res": res,
            "tile": tile,
            "card": card,
            "target": target,
            "why": why,
            "copy": self.args.copy,
            "want": self.args.want,
            "began": began,
            "exit": status,
            "seconds": round(time.monotonic() - started, 1),
            "foreign": held,
            "record": str(record_path),
            "log": str(log),
            "plans_whole": counted(log, None),
            "digests": digests(log),
        }
        decode = None
        if record is not None:
            found["plan"], free, found["plan_source"] = plan_of(record, log)
            found["free"] = None if free is None else free / GIB
            found["units"] = units_of(record, log)
            decodes = [u for u in found["units"] if u["phase"] == "decode"]
            decode = decodes[-1] if decodes else None
            if decode is not None:
                found |= {k: decode[k] for k in ("peak", "pool_peak", "pass_peak", "pool_start")}
            if record.get("oom"):
                oom = record["oom"]
                keys = ("phase", "unit", "caught_at", "request", "allocated", "reserved", "slice")
                found["oom"] = {key: oom.get(key) for key in keys}
            if card != "bare" and free is not None:
                found["card_offset"] = float(card) - free / GIB
        if found.get("free") is not None and found.get("peak") is not None:
            found["room"] = found["free"] - found["peak"]
        alone = self.alone(ladder)
        if found.get("free") is not None and alone["peak"] is not None:
            found["room_pass"] = found["free"] - alone["peak"]
        units = found.get("units", [])
        ran = (
            record is not None
            and status == 0
            and decode is not None
            and not any(u["failed"] for u in units)
        )
        refused = record is not None and record.get("oom") is not None
        if record is not None and not ran and not refused:
            tail = "\n".join(log.read_text(errors="replace").splitlines()[-80:])
            memory_error = re.search(MEMORY_ERROR, f"{record.get('exit')}\n{tail}")
            if memory_error is not None:
                refused = True
                found["refused_by"] = memory_error[0]
        if tile == 0 and found.get("plan") is not None:
            # The decode pinned untiled: whether the untiled pass is what ran (its plan has no
            # decode tiles, and the decode ran one stream's slices, a tiled pass as many a tile).
            tiles = found["plan"].get("decode_tiles", "absent")
            stream = 1 + math.ceil((frames - 5) / 4)
            slices = None if decode is None else decode["slices"]
            fit = self.fit()
            pass_peak = found.get("pass_peak")
            found["untiled"] = {
                "plan_tiles": tiles,
                "slices": slices,
                "one_stream": stream,
                "ran_untiled": tiles is None and slices == stream,
                "fit": fit,
                "pass_peak": pass_peak,
                "over_fit": None if fit is None or pass_peak is None else pass_peak - fit,
            }
            if ran and not found["untiled"]["ran_untiled"]:
                ran = False
                found["why_failed"] = (
                    f"the decode pinned untiled ran with tiles {tiles}, {slices} slices"
                )
        if card == "bare":
            found["outcome"] = "reference" if ran else ("refused" if refused else "failed")
        elif refused:
            found["outcome"] = "refused"
        elif not ran:
            found["outcome"] = "failed"
        reference = self.reference(ladder)
        if reference is not None:
            compare_with(found, reference)
        if ran:
            mine = crcs(out / "checksums")
            found["frames"] = len(mine)
            found["crc_md5"] = hashlib.md5("\n".join(mine).encode()).hexdigest()
            if len(mine) != frames:
                found["outcome"] = "failed"
                found["why_failed"] = f"{len(mine)} frames' checksums, {frames} expected"
            elif card == "bare":
                found["ok"] = True
                keep = self.refs / ladder
                shutil.rmtree(keep, ignore_errors=True)
                keep.mkdir(parents=True)
                for path in sorted(out.glob("seg_*.mkv")):
                    shutil.copy2(path, keep / path.name)
                shutil.copytree(out / "checksums", keep / "checksums")
                # The whole run's reference beside the pass alone's, frame for frame.
                if alone["crcs"] is not None:
                    found["alone_reference_differ"] = differing(mine, crcs(Path(alone["crcs"])))
            elif reference is not None:
                theirs = crcs(self.refs / ladder / "checksums")
                differ = differing(mine, theirs)
                found["differ"] = differ
                found["outcome"] = "moved" if differ else "kept"
                if not theirs:
                    found["outcome"] = "failed"
                    found["why_failed"] = f"no reference checksums in {self.refs / ladder}"
                if differ and theirs:
                    found["first_differing"] = differ[0]
                    found["worst_psnr_y"] = self.worst_psnr(out, ladder, name, differ, found)
        if reference is None and card != "bare":
            found["outcome"] = found.get("outcome", "uncompared")
        shutil.rmtree(out, ignore_errors=True)
        found["ended"] = now()
        with self.rungs_file.open("a") as file:
            file.write(json.dumps(found) + "\n")
        self.note(said(found))
        return found

    def worst_psnr(self, out: Path, ladder: str, name: str, differ: list, found: dict):
        values: list[float] = []
        try:
            for path in sorted(out.glob("seg_*.mkv")):
                stats = self.work / f"{name}.{path.stem}.psnr"
                try:
                    values += psnr_y(path, self.refs / ladder / path.name, stats)
                finally:
                    stats.unlink(missing_ok=True)
            return min(values[i] for i in differ if i < len(values))
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            found["psnr_error"] = repr(error)
            return None

    # -- the ladder

    def next_rung(self, ladder: str) -> tuple[str, float | None, str] | None:
        """The ladder's next rung: (card, target room, why), or None when it is done."""
        rungs = self.rungs(ladder)
        if any(r.get("outcome") in ("failed", "uncompared") for r in rungs):
            return None
        reference = self.reference(ladder)
        if reference is None:
            if any(r["card"] == "bare" for r in rungs):
                return None  # its reference failed
            return "bare", None, "the reference: the whole job on the bare card"
        if len(rungs) >= MOST_RUNGS:
            return None
        kept, lost, every = bracket(rungs)

        def card_for(target: float) -> str:
            return f"{reference['peak'] + target + self.offset():.3f}"

        if not every:
            alone = self.alone(ladder)
            if alone["kept"] is None:
                return card_for(START), START, "START (no pass alone's least room kept read)"
            start = round(alone["kept"]["room"], 3)
            return card_for(start), start, f"the pass alone's least room kept ({alone['ladder']})"
        # One way so far: FINE a step, doubling after four.
        step = FINE * 2 ** max(0, min(len(every) - 4, 3))
        if lost is None:
            target = round(kept["room"] - step, 3)
            if target < BOTTOM:
                return None
            return card_for(target), target, f"kept down to {kept['room']:.3f}: {step:.1f} lower"
        if kept is None:
            target = round(lost["room"] + step, 3)
            if target > TOP:
                return None
            return card_for(target), target, f"lost up to {lost['room']:.3f}: {step:.1f} higher"
        if kept["room"] - lost["room"] > FINE + 0.005:
            target = round((kept["room"] + lost["room"]) / 2, 3)
            bounds = f"{lost['room']:.3f}..{kept['room']:.3f}"
            return card_for(target), target, f"bracket {bounds}: halved"
        if len(kept["outcomes"]) < 2:
            return kept["card"], None, f"the least room kept ({kept['room']:.3f}) again"
        if len(lost["outcomes"]) < 2:
            return lost["card"], None, f"the most room lost ({lost['room']:.3f}) again"
        return None

    def name_for(self, ladder: str, card: str) -> str:
        if card == "bare":
            return f"{ladder}-ref"
        taken = {r["name"] for r in self.rungs(ladder)}
        for count in range(1, 10):
            name = f"{ladder}-c{card}-{count}"
            if name not in taken and not (self.records / f"{name}.json").exists():
                return name
        raise SystemExit(f"{ladder}: card {card} run 9 times")


def compare_with(found: dict, reference: dict) -> None:
    """A rung against its ladder's reference: its room usual, its plan, the plans built by unit,
    the latent's and the windows' digests."""
    found["reference"] = reference["name"]
    found["reference_peak"] = reference.get("peak")
    if found.get("free") is not None and reference.get("peak") is not None:
        found["room_usual"] = found["free"] - reference["peak"]
    if found.get("plan") is not None:
        differs = [k for k in SAME_PLAN if found["plan"].get(k) != reference["plan"].get(k)]
        found["plan_same"], found["plan_differs"] = not differs, differs
    theirs = {u["number"]: u for u in reference.get("units", [])}
    found["plans_over_ref"] = {
        f"{u['number']}:{u['phase']}": u["plans"] - theirs[u["number"]]["plans"]
        for u in found.get("units", [])
        if u["number"] in theirs
    }
    ours, their = found["digests"], reference.get("digests") or {}
    found["latent_same"] = bool(ours["latent"]) and ours["latent"] == their.get("latent")
    found["windows_same"] = bool(ours["windows"]) and ours["windows"] == their.get("windows")


# ------------------------------------------------------------------ reading


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normal(r: dict, path: Path) -> dict:
    """A rung's record as this script writes it, whatever wrote it (the night of 2026-10-17's
    driver among them): the refusal's site under `caught_at` (that driver wrote `site`), the
    reference's frames against the pass alone's under `alone_reference_differ` (it wrote
    `pass_reference_differ`), and where the run's vae_room.py record lies (`_record`: the path the
    driver recorded, else records/ beside the rungs' file), for the pool at the decode's start."""
    r = dict(r)
    if r.get("oom"):
        oom = dict(r["oom"])
        oom.setdefault("caught_at", oom.get("site"))
        r["oom"] = oom
    if "alone_reference_differ" not in r and "pass_reference_differ" in r:
        r["alone_reference_differ"] = r["pass_reference_differ"]
    candidates = [Path(r["record"])] if r.get("record") else []
    candidates.append(path.parent / "records" / f"{r.get('name')}.json")
    found = next((candidate for candidate in candidates if candidate.exists()), None)
    r["_record"] = None if found is None else str(found)
    return r


def pool_at_decode(r: dict) -> dict | None:
    """The pool as the rung's decode began, GiB (what torch had allocated, what its pool held, the
    driver's free memory): the rung's own record's (pool_start), else its vae_room.py record's
    first decode unit's (`entered`, once the phase was entered)."""
    if r.get("pool_start"):
        return r["pool_start"]
    if r.get("_record") is None:
        return None
    for unit in json.loads(Path(r["_record"]).read_text()).get("units", []):
        if unit.get("phase") == "decode":
            entered = unit.get("entered") or {}
            keys = ("allocated", "reserved", "driver_free")
            return {key: entered[key] / GIB for key in keys if key in entered} or None
    return None


def parse(ladder: str) -> tuple[int, int]:
    found = re.fullmatch(r"whole(1080|2160)-(\d+|untiled)", ladder)
    if not found:
        raise SystemExit(f"{ladder}: not a ladder (wholeRES-TILE, TILE untiled for 0)")
    return int(found[1]), 0 if found[2] == "untiled" else int(found[2])


def parse_ladders(text: str) -> dict[str, str | None]:
    """--ladders: LADDER[=PASS], comma-separated, in their order."""
    found: dict[str, str | None] = {}
    for item in filter(None, (part.strip() for part in text.split(","))):
        ladder, _, alone = item.partition("=")
        parse(ladder)
        found[ladder] = alone or None
    return found


def groups(rows: list[dict]) -> list[dict]:
    """The emulated rungs by card: each card's room (usual), its outcomes, whether all kept."""
    found: dict[str, dict] = {}
    for r in rows:
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


def bracket(rows: list[dict]) -> tuple[dict | None, dict | None, list[dict]]:
    """The least room kept on every run above every loss, the most room lost on any run, and
    every card's group (floor.py's rule)."""
    every = groups(rows)
    lost = [g for g in every if not g["kept"]]
    most_lost = max(lost, key=lambda g: g["room"]) if lost else None
    above = [g for g in every if g["kept"] and (most_lost is None or g["room"] > most_lost["room"])]
    kept = min(above, key=lambda g: g["room"]) if above else None
    return kept, most_lost, every


def counted(log: Path, unit: int | None) -> dict:
    """In a rung's log, between its unit's begin and end lines (the whole log when unit is None):
    the cuDNN frontend's plans built (by engine), and torch's "recovered" lines."""
    found: dict = {"plans": 0, "recovered": 0, "engines": {}}
    if not log.exists():
        return found
    inside = unit is None
    for line in log.read_text(errors="replace").splitlines():
        if unit is not None:
            if re.match(rf"vae_room: unit {unit} begins", line):
                inside = True
                continue
            if re.match(rf"vae_room: unit {unit} \S+ (ends|FAILED)", line):
                inside = False
                continue
        if not inside:
            continue
        if PLANS in line:
            found["plans"] += 1
            engine = re.search(r": (\w+_eng\d+)", line)
            if engine:
                found["engines"][engine[1]] = found["engines"].get(engine[1], 0) + 1
        elif TRIM in line:
            found["recovered"] += 1
    return found


def digests(log: Path) -> dict:
    """The latent's and the windows' md5, as the probes say them, by shot."""
    found: dict = {"latent": {}, "windows": {}, "unread": 0}
    if not log.exists():
        return found
    for line in log.read_text(errors="replace").splitlines():
        hit = re.match(r"vae_room: whole_floor: shot (\d+) latent md5 (\w+)", line)
        if hit:
            found["latent"][hit[1]] = hit[2]
            continue
        hit = re.match(r"vae_room: whole_floor: shot (\d+) window (\d+) md5 (\w+)", line)
        if hit:
            found["windows"][f"{hit[1]}:{hit[2]}"] = hit[3]
            continue
        if line.startswith("vae_room: whole_floor:") and "NOT READ" in line:
            found["unread"] += 1
    return found


def plan_of(record: dict, log: Path) -> tuple[dict, float | None, str]:
    """The run's own plan (vae_room's "planned", else the manifest's), its free memory in bytes and
    where it came from; the plan line's figure where the record's isn't the run's."""
    plan = record.get("planned") or record.get("plan") or {}
    source = "planned" if record.get("planned") else "manifest"
    memory = plan.get("memory") or {}
    free = memory.get("free")
    text = log.read_text(errors="replace")
    line = re.search(r"plan: [^\n]*?, ([\d.]+) GiB free, a budget", text)
    if line is not None and (free is None or abs(free / GIB - float(line[1])) > 0.006):
        source = f"its log ({'no plan' if free is None else f'{free / GIB:.3f}'} in the record)"
        free = round(float(line[1]) * GIB)

    def peak(phase: str) -> float | None:
        value = memory.get(f"{phase}_peak")
        return None if value is None else value / GIB

    summary = {
        "window": plan.get("window"),
        "encode_tiles": plan.get("encode_tiles", "absent"),
        "decode_tiles": plan.get("decode_tiles", "absent"),
        "swapped": memory.get("swapped_blocks"),
        "resident_encode": memory.get("resident_encode"),
        "resident_decode": memory.get("resident_decode"),
        "vae_in_dit": memory.get("vae_in_dit"),
        "budget": None if memory.get("budget") is None else memory["budget"] / GIB,
        **{f"{phase}_peak_predicted": peak(phase) for phase in ("encode", "dit", "decode")},
    }
    return summary, free, source


def units_of(record: dict, log: Path) -> list[dict]:
    """Each unit of the run: its phase, peaks, the pool at its start, torch's counters, the plans
    built in it. GiB."""

    def gb(value: int | None) -> float | None:
        return None if value is None else value / GIB

    found = []
    for unit in record.get("units", []):
        entered = unit.get("entered") or {}
        peak = unit.get("peak_allocated")
        start = {k: gb(entered.get(k)) for k in ("allocated", "reserved", "driver_free")}
        found.append(
            {
                "number": unit.get("number"),
                "phase": unit.get("phase"),
                "seconds": unit.get("seconds"),
                "slices": len(unit.get("slices", [])),
                "peak": gb(peak),
                "pool_peak": gb(unit.get("peak_reserved")),
                "pool_start": start if entered else None,
                "pass_peak": None
                if peak is None or entered.get("allocated") is None
                else gb(peak - entered["allocated"]),
                "card_free_least": gb(unit.get("card_free_least")),
                "failed": unit.get("failed"),
                "counters": [unit.get("counters_before"), unit.get("counters_after")],
                **counted(log, unit.get("number")),
            }
        )
    return found


def foreign() -> list[str]:
    query = "--query-compute-apps=pid,process_name,used_memory"
    found = subprocess.run(
        ["nvidia-smi", query, "--format=csv,noheader"], capture_output=True, text=True
    )
    return [line.strip() for line in found.stdout.splitlines() if line.strip()]


def environment() -> dict:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1")
    env |= {"CUDNN_FRONTEND_LOG_INFO": "1", "CUDNN_FRONTEND_LOG_FILE": "stderr"}
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def crcs(directory: Path) -> list[str]:
    found: list[str] = []
    for path in sorted(directory.glob("*.crc32")):
        found += path.read_text().split()
    return found


def differing(mine: list[str], theirs: list[str]) -> list[int]:
    differ = [i for i, pair in enumerate(zip(mine, theirs, strict=False)) if pair[0] != pair[1]]
    return differ + list(range(min(len(mine), len(theirs)), max(len(mine), len(theirs))))


def psnr_y(ours: Path, theirs: Path, stats: Path) -> list[float]:
    """Each frame's PSNR-Y of ours against theirs, by ffmpeg's psnr filter (planner_matrix's)."""
    lavfi = f"[0:v][1:v]psnr=stats_file='{stats}'"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(ours),
            "-i",
            str(theirs),
            "-lavfi",
            lavfi,
            "-f",
            "null",
            "-",
        ],
        check=True,
    )
    found = [re.search(r"psnr_y:(\S+)", line) for line in stats.read_text().splitlines()]
    return [float(hit[1]) for hit in found if hit]


def plans_said(r: dict, phase: str) -> str:
    """The plans built in a rung's units of a phase ('/' between units), and in brackets how many
    more than the reference's same units."""
    units = [u for u in r.get("units", []) if u["phase"] == phase]
    if not units:
        return "-"
    over = r.get("plans_over_ref") or {}
    deltas = [over.get(f"{u['number']}:{u['phase']}") for u in units]
    built = "/".join(str(u["plans"]) for u in units)
    if all(d is None for d in deltas):
        return built
    return f"{built} ({'/'.join(f'{d:+d}' for d in deltas if d is not None)})"


def said(r: dict) -> str:
    """A rung in one line (the progress file's)."""
    differ = r.get("differ")
    frames = "-" if differ is None else f"{len(differ)}/{r.get('frames')}"
    psnr = r.get("worst_psnr_y")
    decode = ([u for u in r.get("units", []) if u["phase"] == "decode"] or [{}])[-1]
    counters = (decode.get("counters") or [None, None])[1]
    pool = pool_at_decode(r) or {}
    line = (
        f"{r['ended']} | {r['ladder']} | {r['name']} | card {r['card']} | free {gib(r.get('free'))}"
        f" decode peak {gib(r.get('peak'))} room {gib(r.get('room'))} (usual"
        f" {gib(r.get('room_usual'))}, by the pass alone's peak {gib(r.get('room_pass'))}) | differ"
        f" {frames} psnr {'-' if psnr is None else f'{psnr:.2f}'} | plans: encode"
        f" {plans_said(r, 'encode')} windows {plans_said(r, 'dit')} decode"
        f" {plans_said(r, 'decode')}"
        f" | counters (decode) {counters} | pool at the decode's start: allocated"
        f" {gib(pool.get('allocated'))} reserved {gib(pool.get('reserved'))}"
    )
    if "latent_same" in r:
        line += f" | latent {'same' if r['latent_same'] else 'OTHER'}"
        line += f" windows {'same' if r['windows_same'] else 'OTHER'}"
        if r.get("plan_same") is False:
            line += f" | PLAN DIFFERS: {r.get('plan_differs')}"
    if r.get("alone_reference_differ") is not None:
        line += f" | against the pass alone's reference: {len(r['alone_reference_differ'])} differ"
    untiled = r.get("untiled")
    if untiled is not None:
        line += (
            f" | untiled ran {untiled['ran_untiled']} (plan tiles {untiled['plan_tiles']}, slices"
            f" {untiled['slices']}/{untiled['one_stream']}), pass peak {gib(untiled['pass_peak'])}"
            f" fit {gib(untiled['fit'])}"
        )
    if r.get("oom"):
        oom = r["oom"]
        site = oom.get("caught_at") or oom.get("site")
        line += f" | REFUSED in the {oom['phase']}, slice {oom['slice']}, at {site}"
    if r.get("why_failed"):
        line += f" | FAILED: {r['why_failed']}"
    if r.get("foreign"):
        line += f" | FOREIGN {r['foreign']}"
    times = "/".join(str(u.get("seconds")) for u in r.get("units", []))
    return line + f" | units s {times} | {r['seconds']} s | exit {r['exit']} | {r.get('outcome')}"


# ------------------------------------------------------------------ commands


def stop_asked(study: Study) -> str | None:
    if study.pause.exists():
        study.note(f"# {now()} {study.pause.name} exists: held before the next rung")
        while study.pause.exists():
            time.sleep(POLL)
        study.note(f"# {now()} {study.pause.name} gone: on")
    if study.skip.exists():
        return f"{study.skip.name} exists"
    return None


def queue(study: Study) -> int:
    study.need("python", "copy", "want", "models", "clips", "lock", "ladders")
    done = study.dir / study.args.done
    Path(study.args.lock).parent.mkdir(parents=True, exist_ok=True)
    done.unlink(missing_ok=True)
    order = list(study.passes)
    study.note(f"# {now()} queue started (pid {os.getpid()}): {' '.join(order)}")
    if study.args.wait_for is not None:
        study.note(f"# {now()} waiting for {study.args.wait_for}")
        while not study.args.wait_for.exists():
            time.sleep(POLL)
        study.note(f"# {now()} {study.args.wait_for} is there")
    for ladder in order:
        study.note(f"# {now()} {ladder}: the pass alone {json.dumps(study.alone(ladder))}")
    ended = "every ladder done"
    for ladder in order:
        study.note(f"# {now()} {ladder} begins or goes on")
        while (why := stop_asked(study)) is None:
            nxt = study.next_rung(ladder)
            if nxt is None:
                kept, lost, _ = bracket(study.rungs(ladder))
                least = (
                    "-" if kept is None else f"{kept['room']:.3f} ({', '.join(kept['outcomes'])})"
                )
                most = (
                    "-" if lost is None else f"{lost['room']:.3f} ({', '.join(lost['outcomes'])})"
                )
                study.note(f"# {now()} {ladder} done: least room kept {least}, most lost {most}")
                break
            tree = study.tree()
            if tree != study.args.want:
                why = f"WRONG TREE {tree} in {study.args.copy} (wanted {study.args.want})"
                break
            free_disk = shutil.disk_usage(study.dir).free / 1e9
            if free_disk < 30:
                why = f"{free_disk:.0f} GB free on the study's disk"
                break
            card, target, reason = nxt
            name = study.name_for(ladder, card)
            study.note(f"# {now()} next: {name} (target room {gib(target)}): {reason}")
            study.run_rung(ladder, card, target, reason, name)
        if why is not None:
            ended = why
            break
    done.write_text(f"{now()} {ended}\n")
    study.note(f"# {now()} queue ended: {ended}")
    return 0


def rung(study: Study) -> int:
    study.need("python", "copy", "want", "models", "clips", "lock")
    parse(study.args.ladder)
    tree = study.tree()
    if tree != study.args.want:
        raise SystemExit(f"WRONG TREE {tree} in {study.args.copy} (wanted {study.args.want})")
    card = study.args.card
    study.run_rung(
        study.args.ladder, card, None, "by hand", study.name_for(study.args.ladder, card)
    )
    return 0


def plan(study: Study) -> int:
    study.need("python", "copy", "want", "models", "clips", "lock", "ladders")
    print(
        f"code: {study.args.copy} tree {study.tree()} (wanted {study.args.want}); offset now"
        f" {study.offset()}"
    )
    for ladder in study.passes:
        print(f"{ladder}: the pass alone {json.dumps(study.alone(ladder))}")
        nxt = study.next_rung(ladder)
        print(f"  next rung: {nxt}")
        if nxt is not None:
            line = study.command(ladder, study.name_for(ladder, nxt[0]), nxt[0])
            print(f"  command: {' '.join(line)}")
    return 0


def table(study: Study) -> int:
    """The tables, from the rungs' files as they are: per ladder the whole run beside its pass
    alone; every rung; the pairs of runs at one card. --compact: each ladder's rungs on one line."""
    rows = study.rungs()
    ladders = list(study.passes) or list(dict.fromkeys(r["ladder"] for r in rows))
    if study.args.compact:
        print("| ladder | rungs |")
        print("|---|---|")
        for ladder in ladders:
            parts = []
            for r in [r for r in rows if r["ladder"] == ladder]:
                if r["card"] == "bare":
                    parts.append(f"ref {r['peak']:.3f}")
                    continue
                decode = decode_over(r)
                text = f"{r['room_usual']:.2f} {r['outcome']}"
                if r["outcome"] == "moved":
                    text += f" {len(r['differ'])}/{r['frames']}"
                if r["outcome"] == "refused":
                    text += f" (slice {r['oom']['slice']})"
                if decode is not None and r["outcome"] != "refused":
                    text += f" ({decode:+d})"
                parts.append(text)
            print(f"| {ladder} | {'; '.join(parts)} |")
        return 0

    def group(g: dict | None) -> str:
        return "-" if g is None else f"{g['room']:.3f} ({', '.join(g['outcomes'])})"

    print(
        "## Per tile: the whole run beside the pass alone (room usual, GiB: the free memory the"
        " plan was made in less the reference's decode peak)\n"
    )
    print(
        "| tile | whole run: least room kept (runs) | most room lost (runs) | pass alone: least"
        " kept | most lost | gap, least kept (whole - alone) | reference's decode peak: whole /"
        " alone | whole reference's frames against the pass alone's | pool at the decode's start,"
        " whole reference: allocated / reserved (alone's) | moved: frames, worst PSNR-Y | decode"
        " plans over the reference's: lost / kept | torch's counters |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for ladder in ladders:
        mine = [r for r in rows if r["ladder"] == ladder]
        reference = next((r for r in mine if r["card"] == "bare"), None)
        if reference is None:
            continue
        alone = study.alone(ladder)
        kept, lost, _ = bracket(mine)
        gap = None
        if kept is not None and alone["kept"] is not None:
            gap = kept["room"] - alone["kept"]["room"]
        pool = pool_at_decode(reference) or {}
        their = None if alone["reference"] is None else pool_at_decode(alone["reference"])
        theirs = "-" if not their else f"{their['allocated']:.3f} / {their['reserved']:.3f}"
        moved = [r for r in mine if r.get("outcome") == "moved"]
        counts = ", ".join(str(n) for n in sorted({len(r["differ"]) for r in moved}))
        psnrs = [r["worst_psnr_y"] for r in moved if r.get("worst_psnr_y") is not None]
        psnr = f"{min(psnrs):.2f}-{max(psnrs):.2f} dB" if psnrs else "-"
        lost_over = sorted(
            {
                decode_over(r)
                for r in mine
                if r["card"] != "bare" and r["outcome"] in ("moved", "refused")
            }
            - {None}
        )
        kept_over = sorted(
            {decode_over(r) for r in mine if r["card"] != "bare" and r["outcome"] == "kept"}
            - {None}
        )
        counters = sorted(
            {
                sum(v or 0 for v in ((u.get("counters") or [None, None])[1] or {}).values())
                for r in mine
                for u in r.get("units", [])
            }
        )
        differ = reference.get("alone_reference_differ")
        against = "-" if differ is None else f"{len(differ)} of {reference['frames']} differ"
        if differ is not None and alone["note"]:
            against += f" ({alone['note']})"
        print(
            f"| {study.labels.get(ladder, ladder)} | {group(kept)} | {group(lost)} |"
            f" {group(alone['kept'])} | {group(alone['lost'])} | {gib(gap)} |"
            f" {gib(reference['peak'])} / {gib(alone['peak'])} | {against} |"
            f" {gib(pool.get('allocated'))} / {gib(pool.get('reserved'))} ({theirs}) |"
            f" {counts or '-'} of {reference['frames']};"
            f" {psnr} | {'/'.join(f'{d:+d}' for d in lost_over)} /"
            f" {'/'.join(f'{d:+d}' for d in kept_over)} | sum {counters} |"
        )
    print("\n## Every whole-run rung\n")
    print(
        "| rung | card | free | decode peak | room | room usual | by the pass alone's peak |"
        " outcome | frames off (from) | worst PSNR-Y | cuDNN plans: encode / windows / decode (over"
        " the ref's) | latent, windows | pool at the decode's start: allocated / reserved, driver"
        " free | units s (information) | whole s |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in [r for ladder in ladders for r in rows if r["ladder"] == ladder]:
        outcome = r["outcome"]
        if r.get("oom"):
            oom = r["oom"]
            request = "?" if oom.get("request") is None else f"{oom['request'] / GIB:.2f} GiB"
            outcome += (
                f" ({oom['phase']}, slice {oom['slice']}, {oom['caught_at']}, {request} asked)"
            )
        if r.get("untiled"):
            outcome += f" [untiled: {r['untiled']['ran_untiled']}, {r['untiled']['slices']} slices]"
        differ = r.get("differ")
        off = "-" if differ is None else f"{len(differ)} of {r.get('frames')}"
        if differ is not None and r.get("first_differing") is not None:
            off += f" (from {r['first_differing']})"
        same = "-"
        if "latent_same" in r:
            latent, windows = r["latent_same"], r["windows_same"]
            same = f"{'same' if latent else 'OTHER'}, {'same' if windows else 'OTHER'}"
        psnr = r.get("worst_psnr_y")
        pool = pool_at_decode(r) or {}
        times = "/".join(str(u.get("seconds")) for u in r.get("units", []))
        print(
            f"| {r['name']} | {r['card']} | {gib(r.get('free'))} | {gib(r.get('peak'))} |"
            f" {gib(r.get('room'))} | {gib(r.get('room_usual'))} | {gib(r.get('room_pass'))} |"
            f" {outcome} | {off} | {'-' if psnr is None else f'{psnr:.2f}'} |"
            f" {plans_said(r, 'encode')} / {plans_said(r, 'dit')} / {plans_said(r, 'decode')} |"
            f" {same} | {gib(pool.get('allocated'))} / {gib(pool.get('reserved'))},"
            f" {gib(pool.get('driver_free'))} | {times} | {r['seconds']} |"
        )
    print("\n## Pairs of runs at one card\n")
    print("| ladder | card | room usual | runs | agree |")
    print("|---|---|---|---|---|")
    for ladder in ladders:
        by_card: dict[str, list[dict]] = {}
        for r in rows:
            if r["ladder"] == ladder and r["card"] != "bare":
                by_card.setdefault(r["card"], []).append(r)
        for card, runs in by_card.items():
            if len(runs) < 2:
                continue
            each = []
            for r in runs:
                text = r["outcome"]
                if r.get("differ"):
                    text += f" {len(r['differ'])}/{r['frames']} {r.get('worst_psnr_y')} dB,"
                    text += f" plans {decode_over(r):+d}"
                each.append(text)
            outcomes, frames = {r["outcome"] for r in runs}, {r.get("crc_md5") for r in runs}
            agree = "yes" if len(outcomes) == 1 and len(frames) == 1 else "NO"
            if len(outcomes) == 1 and len(frames) > 1:
                agree = "outcome yes, frames no"
            room = gib(runs[0].get("room_usual"))
            print(f"| {ladder} | {card} | {room} | {'; '.join(each)} | {agree} |")
    return 0


def decode_over(r: dict) -> int | None:
    """The decode's plans built over the reference's decode."""
    over = r.get("plans_over_ref") or {}
    return next((value for key, value in over.items() if key.endswith(":decode")), None)


def main() -> int:
    if sys.argv[1:2] == ["run"]:
        return probed_run(sys.argv[2:])
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--dir", type=Path, required=True, help="the study's directory")
    parser.add_argument("--python", help="seedvr2x's environment's python, for vae_room.py")
    parser.add_argument("--copy", help="the frozen copy of seedvr2x under study")
    parser.add_argument("--want", help="its tree (git's), checked before every rung")
    parser.add_argument("--models", help="seedvr2x's --model-dir")
    parser.add_argument("--clips", help="the directory of planner_matrix.py's clip-N.mkv")
    parser.add_argument("--lock", help="the GPU's lock file (flock)")
    parser.add_argument(
        "--alone",
        action="append",
        help="PATH[=NOTE]: a floor.py study (its directory, or its rungs file), NOTE said beside"
        " a reference's frames against its pass alone",
    )
    parser.add_argument("--ladders", help="LADDER[=PASS], comma-separated, in their order")
    parser.add_argument(
        "--rungs", action="append", help="a rungs file or a glob, read (the study's own besides)"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan")
    one = commands.add_parser("rung")
    one.add_argument("ladder")
    one.add_argument("--card", required=True)
    one.add_argument("--pass", dest="pass_ladder", help="its pass alone's ladder (floor.py's)")
    queued = commands.add_parser("queue")
    queued.add_argument("--wait-for", type=Path, help="no rung before this file exists")
    queued.add_argument("--done", default="done", help="the done file's name in the directory")
    tabled = commands.add_parser("table")
    tabled.add_argument("--compact", action="store_true", help="each ladder's rungs on one line")
    tabled.add_argument(
        "--label", action="append", help="LADDER=TEXT, the tile's name in the table"
    )
    args = parser.parse_args()
    study = Study(args)
    return {"plan": plan, "rung": rung, "queue": queue, "table": table}[args.command](study)


if __name__ == "__main__":
    sys.exit(main())

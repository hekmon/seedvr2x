#!/usr/bin/env python3
"""Tile probe for seedvr2x: what each tile of its tiled VAE passes takes of the GPU's memory.

seedvr2x's tiled encode and decode (runtime/tiling.py) run one tile at a time over the whole
shot, the frames and the latents waiting in system memory: a pass peaks at its largest tile, and
the memory planner takes that tile's measured peak where it has one (runtime/planner.py,
MEASURED). This probe measures those peaks on seedvr2x's own runtime, the two models loaded whole
(runtime/model.py, load_models, no plan), over a shot of 21 frames, the tiles sharing 64 px, the
scratch in system memory.

A pass's peak is torch's max_memory_allocated above memory_allocated before the pass, the peak
statistics reset first: the DiT's and the VAE's weights sit on the card before it and aren't
counted. The encode is tiling.encode_tiled, its latent returned in system memory; the decode is
tiling.decode_tiled of that latent, each slice dropped as it comes.

  run    an encode and a decode for each tile, the squares then the tiles cut off:
         - a square of T px a side, 512 to 2048 px in steps of 64: the source's frames resized
           with torch (bicubic, antialias) to T rows by T + 128 columns, normalised to [-1, 1],
           bfloat16, fed in the VAE's slices: the grid is 1 x 2, a whole T x T tile and one of
           T x 192;
         - a tile cut off at the picture's edge: the frames through seedvr2x's own transform and
           padding (shot.encoder_inputs), 1080p padded to 1104 x 1920 in tiles of 1152 to 1856
           px, 720p to 752 x 1280 in tiles of 768 to 1216: the grid is 1 x 2, its largest tile
           the padded rows by T.
         One JSON line per tile, sizes in bytes, appended to --out and written at once; every
         tile in its own try, so that a tile the card refuses has its line and the next one
         runs. A tile whose fits pass the memory free less --reserve is skipped, and said: a
         driver that gives system memory in the card's place (Windows', WSL2 included) would run
         it past the card instead of refusing it. --tiny takes sizes of a few dozen pixels and 9
         frames, for a debug run on the CPU.
  plan   the tiles `run` measures, the grid seedvr2x lays over each one's frames and the fits'
         peaks, with no GPU and no model: whether today's grids are the ones measured
  table  RESULTS.jsonl [...]   the Markdown tables of research/docs/planner-limits.md
         ("seedvr2x's tiled passes: every tile's peak") from run's lines, a tile measured twice
         taken at its last line: each tile's peaks against the tiled fits; a cut-off tile's also
         against the straight line, by area, between the two measured squares around its own,
         and against the larger of the two, the rule the planner gives a cut-off tile nobody
         measured. Standard library only: any python runs it.

Sizes are rows x columns; a GiB is 2**30 bytes, a megapixel 10**6 pixels. The fits are
planner-limits.md's and vram.md's, at a tile's area A in megapixels: from tiles of 1024 px a
side, encode 1.36 + 8.66 A and decode 0.44 + 16.29 A; below, encode 1.7 + 8.4 A and decode
1.6 + 15.6 A. A cut-off tile takes the fit of its side T.

The peaks in planner-limits.md were measured on 2026-10-11 by `run` as it then was, a driver
with its paths written in, on seedvr2x at e6450d8 with the VAE passes that 2adcea7 brought,
before that commit; `table` gives that section's tables from its lines. The passes it calls are
the same in the seedvr2x this file came with (runtime/model.py's VAE passes, tilegrid.py's grid,
tiling.py's two passes, shot.py's encoder inputs, job.py's padding). The allocator is
cudaMallocAsync unless the environment names one (PYTORCH_CUDA_ALLOC_CONF), set before torch is
imported: the one those peaks were measured under, and seedvr2x's own on a driver that refuses.

Usage (seedvr2x's venv python for run and plan; run under the GPU's lock, ffmpeg and ffprobe in
the PATH):
  python tile_peaks.py run --dit DIT.safetensors --vae VAE.safetensors --source CLIP.mkv \
      --first 48 --out tiles.jsonl
  python tile_peaks.py plan
  python3 tile_peaks.py table tiles.jsonl
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from functools import partial
from itertools import pairwise
from pathlib import Path

GIB = 2**30
OVERLAP = 64  # the pixels two tiles share: seedvr2x's default (runtime/plan.py, OVERLAP)
FRAMES = 21
SQUARES = list(range(512, 2049, 64))
# The tiles cut off at the picture's edge: a name, the output's (rows, columns), the tiles' sides.
CUTOFFS = [
    ("1080p", (1080, 1920), list(range(1152, 1857, 64))),
    ("720p", (720, 1280), list(range(768, 1217, 64))),
]
# --tiny's, for a debug run on the CPU.
TINY_OVERLAP, TINY_FRAMES = 16, 9
TINY_SQUARES = [64, 96, 128]
TINY_CUTOFFS = [("1080p", (48, 160), [96, 128]), ("720p", (32, 128), [80, 96])]
# The tiled fits, each (GiB, GiB per megapixel of the tile), the encode's then the decode's: from
# tiles of REFITTED px a side (research/docs/planner-limits.md, Tiled VAE at 4K), and below
# (research/docs/vram.md, Tiling).
REFITTED = 1024
FITS = ((1.36, 8.66), (0.44, 16.29))
FITS_SMALL = ((1.7, 8.4), (1.6, 15.6))
PASSES = ("encode", "decode")


def fits(side: int, pixels: int) -> tuple[float, float]:
    """The encode's and the decode's fitted peaks for a tile of `side` px a side and of `pixels`
    pixels, in GiB."""
    (encode, encode_slope), (decode, decode_slope) = FITS if side >= REFITTED else FITS_SMALL
    area = pixels / 1e6
    return encode + encode_slope * area, decode + decode_slope * area


# ---------------------------------------------------------------- run


def run(args: argparse.Namespace) -> int:
    # Before torch is imported; the environment's own setting stands.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

    import numpy as np
    import torch
    from seedvr2x.runtime import job, model, shot, tiling
    from seedvr2x.runtime.plan import Tiles

    torch.set_grad_enabled(False)
    started = time.time()
    cuda = torch.cuda.is_available()
    device = torch.device("cuda", 0) if cuda else torch.device("cpu")
    overlap, count = (TINY_OVERLAP, TINY_FRAMES) if args.tiny else (OVERLAP, FRAMES)
    squares = TINY_SQUARES if args.tiny else SQUARES
    cutoffs = TINY_CUTOFFS if args.tiny else CUTOFFS
    allowance = 2**40  # a scratch's allowance: always in system memory
    failed = 0

    def emit(**record) -> None:
        line = json.dumps({"t": round(time.time() - started, 1), **record}, default=str)
        with args.out.open("a") as file:
            file.write(line + "\n")
            file.flush()
            os.fsync(file.fileno())
        print(line, flush=True)

    def clean() -> None:
        gc.collect()
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def resident_peak() -> int:
        """The process's resident peak, in bytes, since it was last reset (measure)."""
        with open("/proc/self/status") as file:
            for line in file:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
        return 0

    def failure() -> str:
        """The current exception: its traceback on stderr, its last lines returned."""
        text = traceback.format_exc()
        print(text, file=sys.stderr, flush=True)
        return " | ".join(line.strip() for line in text.strip().splitlines()[-5:])[-700:]

    def measure(function, frames: int):
        """function() and its measures, in bytes: the device's peak above what is allocated
        before the call, the reserved peak, what is allocated right after it returns, the
        seconds, the process's resident peak during it."""
        clean()
        try:
            with open("/proc/self/clear_refs", "w") as file:
                file.write("5")  # the resident peak starts again from here
        except OSError:
            pass
        before = reserved = 0
        if cuda:
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
        began = time.time()
        result = function()
        if cuda:
            torch.cuda.synchronize()
        seconds = time.time() - began
        after = peak = before
        if cuda:
            after = torch.cuda.memory_allocated()
            peak = torch.cuda.max_memory_allocated()
            try:
                reserved = torch.cuda.max_memory_reserved()
            except RuntimeError:
                pass  # an allocator that doesn't count it: 0
        return result, {
            "before": before,
            "peak": peak,
            "above": peak - before,
            "above_gib": round((peak - before) / GIB, 4),
            "reserved_peak": reserved,
            "after": after,
            "left": after - before,
            "seconds": round(seconds, 2),
            "s_per_frame": round(seconds / frames, 3),
            "rss_peak": resident_peak(),
        }

    def passes(record: dict, encode, decode) -> None:
        """The encode pass, then the decode pass of its latent, each measured: one line."""
        latent, record["encode"] = measure(encode, count)
        record["latent"] = {
            "shape": list(latent.shape),
            "dtype": str(latent.dtype),
            "device": str(latent.device),
        }

        def decoded() -> int:
            got = 0
            for piece in decode(latent):
                got += piece.shape[1]
                del piece
            return got

        try:
            record["decoded_frames"], record["decode"] = measure(decoded, count)
        except Exception:
            record["decode_error"] = failure()
        latent = None  # given up before the next tile
        emit(**record)
        if "decode_error" in record:
            raise RuntimeError("the decode failed (the encode's line is written)")

    def fits_the_card(side: int, pixels: int) -> bool:
        """Whether a tile's fitted peaks leave --reserve of the memory free now; said if not."""
        if not cuda:
            return True
        clean()
        free = torch.cuda.mem_get_info()[0]
        need = max(fits(side, pixels)) * GIB
        if need <= free - args.reserve * GIB:
            return True
        emit(
            skipped=True,
            tile=side,
            largest_px=pixels,
            fit_gib=round(need / GIB, 3),
            free_gib=round(free / GIB, 3),
            reserve_gib=args.reserve,
        )
        return False

    def read_frames() -> np.ndarray:
        """`count` frames of the source from --first on, whole in memory: (count, H, W, 3) uint8."""
        probe = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
        probe += ["-show_entries", "stream=width,height", "-of", "csv=p=0", str(args.source)]
        said = subprocess.run(probe, check=True, stdout=subprocess.PIPE, text=True).stdout
        width, height = (int(value) for value in said.split()[0].split(",")[:2])
        first = args.first
        command = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(args.source)]
        command += [
            "-vf",
            f"trim=start_frame={first}:end_frame={first + count},setpts=PTS-STARTPTS",
        ]
        command += ["-fps_mode", "passthrough", "-frames:v", str(count)]
        command += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        raw = subprocess.run(command, check=True, stdout=subprocess.PIPE).stdout
        if len(raw) != count * height * width * 3:
            raise RuntimeError(f"{len(raw)} bytes read, not {count} frames of {height}x{width}")
        return np.frombuffer(raw, np.uint8).reshape(count, height, width, 3).copy()

    def reader(frames: np.ndarray):
        """read(n) over frames in memory, as seedvr2x's reader gives them: float32 in [0, 1]."""
        position = 0

        def read(n: int):
            nonlocal position
            out = frames[position : position + n].astype(np.float32) / 255
            position += n
            return out

        return read

    def square_slices(frames: np.ndarray, padded: tuple[int, int]) -> list:
        """The frames resized with torch (bicubic, antialias) to padded (rows, columns),
        normalised to [-1, 1], bfloat16, (C, t, rows, columns), in the slices model.encode_slices
        gives; made on the device, kept on the CPU, outside the measure."""
        out, start = [], 0
        for size in model.encode_slices(models, frames.shape[0]):
            x = torch.from_numpy(frames[start : start + size]).to(device).permute(0, 3, 1, 2)
            x = x.to(torch.float32).div_(255)
            x = torch.nn.functional.interpolate(x, size=padded, mode="bicubic", antialias=True)
            x = x.clamp_(0, 1).mul_(2).sub_(1).permute(1, 0, 2, 3)
            out.append(x.to(model.COMPUTE_DTYPE).contiguous().cpu())
            start += size
            del x
        return out

    def square(side: int) -> None:
        tiles = Tiles(side, overlap)
        padded = (side, side + 2 * overlap)
        record = {"part": "T2", "kind": "square", "frames": count, **geometry(padded, tiles)}
        record["as_expected"] = record["grid"] == "1x2" and record["largest_tile"] == [side, side]
        if not fits_the_card(side, record["largest_px"]):
            return
        slices = square_slices(frames, padded)
        record["input"] = {"shape": list(slices[0].shape), "dtype": str(slices[0].dtype)}

        def encode():
            inputs = (x.to(device) for x in slices)
            return tiling.encode_tiled(
                models, inputs, count, tiles, tiling.Scratch(None, allowance)
            )

        def decode(latent):
            return tiling.decode_tiled(models, latent, tiles, tiling.Scratch(None, allowance))

        passes(record, encode, decode)

    def cutoff(res: str, target: tuple[int, int], side: int) -> None:
        tiles = Tiles(side, overlap)
        padded = job.padded_size(target)
        record = {"part": "T2", "kind": "cutoff", "res": res, "target": list(target)}
        record.update(frames=count, **geometry(padded, tiles))
        record["as_expected"] = record["grid"] == "1x2" and record["largest_tile"] == [
            padded[0],
            side,
        ]
        if not fits_the_card(side, record["largest_px"]):
            return

        def encode():
            inputs = shot.encoder_inputs(models, reader(frames), count, target)
            scratch = tiling.Scratch(None, allowance)
            return tiling.encode_tiled(models, inputs, shot.padded_length(count), tiles, scratch)

        def decode(latent):
            return tiling.decode_tiled(models, latent, tiles, tiling.Scratch(None, allowance))

        passes(record, encode, decode)

    def step(name: str, function) -> None:
        """A step, in its own try."""
        nonlocal failed
        began = time.time()
        try:
            function()
            emit(step=name, done=True, seconds=round(time.time() - began, 1))
        except Exception:
            failed += 1
            emit(step=name, failed=True, seconds=round(time.time() - began, 1), error=failure())
        clean()

    emit(
        start=time.strftime("%Y-%m-%d %H:%M:%S %z"),
        epoch=round(started),
        parts=["T2"],
        tiny=args.tiny,
        cuda=cuda,
    )
    began = time.time()
    models = model.load_models(args.dit, args.vae, device)
    if cuda:
        torch.cuda.synchronize()
    emit(
        part="load",
        seconds=round(time.time() - began, 1),
        device=str(device),
        gpu=torch.cuda.get_device_name(0) if cuda else None,
        allocator=torch.cuda.get_allocator_backend() if cuda else None,
        allocated_gib=round(torch.cuda.memory_allocated() / GIB, 4) if cuda else None,
        card_free_total_gib=[round(x / GIB, 4) for x in torch.cuda.mem_get_info()]
        if cuda
        else None,
        vae_dtype=str(next(models.runner.vae.parameters()).dtype),
        attention=models.attention,
        torch=torch.__version__,
        overlap=overlap,
        encode_slices=model.encode_slices(models, count),
        latent_slices=model.latent_slices(models, 1 + (count - 1) // 4),
    )
    frames = read_frames()
    emit(part="T2", what="setup", frames=list(frames.shape), first_frame=args.first)
    for side in squares:
        step(f"T2-square-{side}", partial(square, side))
    for res, target, sides in cutoffs:
        for side in sides:
            step(f"T2-cutoff-{res}-{side}", partial(cutoff, res, target, side))
    emit(end=True, parts=["T2"], failed_steps=failed, total_seconds=round(time.time() - started, 1))
    return 1 if failed else 0


def geometry(padded: tuple[int, int], tiles) -> dict:
    """The grid seedvr2x lays over frames padded to (rows, columns) pixels (runtime/tilegrid.py)."""
    from seedvr2x.runtime import tilegrid

    layout = tilegrid.grid(tilegrid.latent_size(padded), tiles)
    rows = sorted({(t.top, t.bottom) for t in layout})
    columns = sorted({(t.left, t.right) for t in layout})
    big = max(layout, key=lambda t: t.rows * t.columns)
    return {
        "padded": list(padded),
        "tile": tiles.size,
        "overlap": tiles.overlap,
        "tiles": len(layout),
        "grid": f"{len(rows)}x{len(columns)}",
        "largest_px": tilegrid.largest(padded, tiles),
        "largest_tile": [big.rows * 8, big.columns * 8],
        "tile_sizes": [[t.rows * 8, t.columns * 8] for t in layout],
    }


# ---------------------------------------------------------------- plan


def plan(args: argparse.Namespace) -> int:
    from seedvr2x.runtime import job
    from seedvr2x.runtime.plan import Tiles

    cases = [("square", None, (side, side + 2 * OVERLAP), side, [side, side]) for side in SQUARES]
    for res, target, sides in CUTOFFS:
        padded = job.padded_size(target)
        cases += [(f"cut off, {res}", target, padded, side, [padded[0], side]) for side in sides]
    header = ["tile", "kind", "frames padded to", "grid", "tiles", "largest tile", "Mpx"]
    rows, odd = [], 0
    for kind, _, padded, side, expected in cases:
        found = geometry(padded, Tiles(side, OVERLAP))
        as_expected = found["grid"] == "1x2" and found["largest_tile"] == expected
        odd += not as_expected
        encode, decode = fits(side, found["largest_px"])
        rows.append(
            [
                f"{side}{'' if as_expected else ' (!)'}",
                kind,
                size(padded),
                found["grid"],
                ", ".join(size(tile) for tile in found["tile_sizes"]),
                size(found["largest_tile"]),
                f"{found['largest_px'] / 1e6:.4f}",
                f"{encode:.3f}",
                f"{decode:.3f}",
            ]
        )
    print("\n".join(table([*header, "encode fit", "decode fit"], rows)))
    print(f"\n{len(cases)} tiles, {FRAMES} frames, sharing {OVERLAP} px; fits in GiB")
    if odd:
        print(f"(!) {odd} grids are not the ones measured (1 x 2, the largest tile T wide)")
    return 1 if odd else 0


# ---------------------------------------------------------------- table


def size(pair) -> str:
    return f"{pair[0]} × {pair[1]}"


def gib(x: float | None) -> str:
    return "n/a" if x is None else f"{x / GIB:.3f}"


def signed(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.3f}".replace("-", "−")


def table(header: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]


def above(record: dict, name: str) -> int | None:
    return record[name]["above"] if name in record else None


def summary(args: argparse.Namespace) -> int:
    records, broken = [], 0
    for path in args.results:
        for line in path.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except ValueError:
                broken += 1
    # "T2" marks a tile's line, as in the lines the peaks on file were written in.
    kept: dict = {}
    for r in records:
        if r.get("part") == "T2" and "encode" in r:
            kept[(r["kind"], r.get("res"), r["tile"])] = r  # a tile measured again: the last
    squares = sorted((r for r in kept.values() if r["kind"] == "square"), key=lambda r: r["tile"])
    cutoffs = sorted(
        (r for r in kept.values() if r["kind"] == "cutoff"),
        key=lambda r: (-r["padded"][0], r["tile"]),
    )
    if not kept:
        print("no tile's line")
        return 1
    load = next((r for r in records if r.get("part") == "load"), None)
    if load:
        keys = ("gpu", "torch", "allocator", "attention")
        print("Measured on: " + ", ".join(f"{k} {load.get(k)}" for k in keys) + "\n")

    def flag(r: dict) -> str:
        return "" if r.get("as_expected") else " (!)"

    def line(pixels: int, name: str) -> tuple[float | None, str]:
        """A tile's peak by the straight line, in area, between the two measured squares whose
        areas are around its own, in bytes, and those squares."""
        points = sorted(
            (s["largest_px"], above(s, name), s["tile"])
            for s in squares
            if s.get("as_expected") and above(s, name) is not None
        )
        for (a0, v0, s0), (a1, v1, s1) in pairwise(points):
            if a0 <= pixels <= a1:
                return v0 + (pixels - a0) / (a1 - a0) * (v1 - v0), f"{s0} and {s1}"
        return None, "none around it"

    print(f"### The squares: a whole T × T tile ({len(squares)})\n")
    header = ["Tile", "Mpx", "Encode", "Encode's fit", "Encode − fit"]
    header += ["Decode", "Decode's fit", "Decode − fit"]
    rows = []
    worst = {name: (0.0, "") for name in PASSES}  # the most a measure stands over its fit
    for r in squares:
        row = [f"{r['tile']}{flag(r)}", f"{r['largest_px'] / 1e6:.4f}"]
        for index, name in enumerate(PASSES):
            now, fit = above(r, name), fits(r["tile"], r["largest_px"])[index]
            diff = None if now is None else now / GIB - fit
            row += ["FAILED" if now is None else gib(now), f"{fit:.3f}", signed(diff)]
            if diff is not None and diff > worst[name][0]:
                worst[name] = (diff, f"{r['tile']} px")
        rows.append(row)
    print("\n".join(table(header, rows)))

    over = {name: (0.0, "") for name in PASSES}  # the most one stands over fit and line both
    apart = {name: 0.0 for name in PASSES}  # the farthest one stands from the line
    for index, name in enumerate(PASSES):
        print(f"\n### The tiles cut off at the picture's edge: the {name} ({len(cutoffs)})\n")
        header = ["Tile", "Largest tile", "Mpx", name.capitalize(), "Fit", "Measured − fit"]
        header += ["Squares' line", "Between", "Measured − line", "Measured − the larger"]
        rows = []
        for r in cutoffs:
            row = [f"{r['tile']}{flag(r)}", size(r["largest_tile"]), f"{r['largest_px'] / 1e6:.4f}"]
            now, fit = above(r, name), fits(r["tile"], r["largest_px"])[index]
            if now is None:
                rows.append([*row, "FAILED", f"{fit:.3f}", *["n/a"] * 5])
                continue
            between, squares_named = line(r["largest_px"], name)
            gap = None if between is None else (now - between) / GIB
            rule = fit if between is None else max(fit, between / GIB)
            diff = now / GIB - fit
            row += [gib(now), f"{fit:.3f}", signed(diff), gib(between), squares_named]
            rows.append([*row, signed(gap), signed(now / GIB - rule)])
            tile = size(r["largest_tile"])
            if diff > worst[name][0]:
                worst[name] = (diff, tile)
            if now / GIB - rule > over[name][0]:
                over[name] = (now / GIB - rule, tile)
            if gap is not None:
                apart[name] = max(apart[name], abs(gap))
        print("\n".join(table(header, rows)) if rows else "no cut-off tile's line")

    def most(found: tuple[float, str]) -> str:
        return f"{found[0]:.3f} GiB ({found[1]})" if found[1] else "nothing"

    print()
    for name in PASSES:
        print(
            f"- the {name}: at the most {most(worst[name])} over its fit; a cut-off tile at the"
            f" most {apart[name]:.3f} GiB from the squares' line, and {most(over[name])} over the"
            " larger of its fit and that line"
        )
    odd = [r for r in kept.values() if not r.get("as_expected")]
    for r in odd:
        print(f"(!) {r['kind']} {r['tile']}: grid {r['grid']}, largest {size(r['largest_tile'])}")
    failed = [r for r in records if r.get("failed")]
    skipped = sum(1 for r in records if r.get("skipped"))
    print(
        f"\n{len(squares)} squares and {len(cutoffs)} cut-off tiles; {len(failed)} failed steps,"
        f" {skipped} skipped, {broken} unreadable lines"
    )
    for r in failed:
        print(f"  FAILED {r.get('step')}: {str(r.get('error'))[:300]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    sub = commands.add_parser("run", help="measure every tile's peak (GPU)")
    sub.add_argument("--dit", type=Path, required=True, help="the DiT's file (7B fp16)")
    sub.add_argument("--vae", type=Path, required=True, help="the VAE's file")
    sub.add_argument(
        "--source", type=Path, required=True, help="a clip, 1080p for the peaks on file"
    )
    sub.add_argument("--first", type=int, default=0, help="the first frame taken of the source")
    sub.add_argument("--out", type=Path, required=True, help="the JSON lines, appended")
    sub.add_argument("--reserve", type=float, default=1.5, help="GiB left free on the card")
    sub.add_argument("--tiny", action="store_true", help="tiny sizes, for a debug run on the CPU")
    sub.set_defaults(function=run)
    sub = commands.add_parser("plan", help="the tiles and their grids, no GPU")
    sub.set_defaults(function=plan)
    sub = commands.add_parser("table", help="the Markdown tables from run's lines")
    sub.add_argument("results", type=Path, nargs="+")
    sub.set_defaults(function=summary)
    args = parser.parse_args()
    return args.function(args)


if __name__ == "__main__":
    sys.exit(main())

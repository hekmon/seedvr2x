"""A VAE pass's time at each tile side the planner tries: seedvr2x's tiled encode and decode of
one shape, side by side, models loaded once.

  tile_times.py --dit DIT --vae VAE --out FILE.jsonl [--size 2160] [--frames 1] [--reps 3]
      [--pass encode,decode] [--sides 512,576,...] [--from 512] [--to PX]

The sides are the planner's own for the frame (shot_plan._tile_sizes: multiples of TILE_STEP from
TILE_MIN to the largest under the frame's longer side, TILE_MAX at most), smallest first, or
`--sides`. The models load as seedvr2x loads them (model.load_models), the DiT then taken off the
GPU, as a VAE phase with all its blocks swapped has it, and torch is held to the card as a run
holds it (memory.hold: the free memory less HELD_BACK; under small_card.py the desktop's share
taken off first). Each pass runs as the runtime runs it: tiling.encode_tiled of `--frames`
seeded frames (4n + 1) at `--size` through the encoder's own transform
(shot_pass.encoder_inputs), or tiling.decode_tiled of the seeded latent of those frames (their
padded_length's latents), the scratch in system memory.

At each side: one warm-up on a short shot of the same tile and slice shapes (9 frames for the
encode, its slices of 5 and 4; 3 latents for the decode, its slices of 2 and 1; the whole pass
for a picture of one frame), so that the convolutions' engines are chosen before the timing;
then `--reps` passes, each between two synchronisations, torch's peaks reset at its start. One
JSON line a pass: the side, its grid (tiles in all and across), the wall time, torch's allocated
and reserved peaks, and whether it is the warm-up; the limit on the first line. A pass refused
(OutOfMemoryError) is said and ends the sides of that pass: a larger tile needs more. The time
a side's plan spends moving the models between phases is not here: a plan's moves follow from
the room each side leaves the DiT (picture_phases.py and the command's own lines give them).

Written on 2026-11-05 for place-study (the night's Q10: a time per tile side, for the plan's tile
rule)."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(prog="tile_times.py")
    parser.add_argument("--dit", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", default="2160")
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--pass", dest="passes", default="encode,decode")
    parser.add_argument("--sides", default="")
    parser.add_argument("--from", dest="low", type=int, default=0)
    parser.add_argument("--to", dest="high", type=int, default=0)
    args = parser.parse_args()

    from seedvr2x import memory

    memory.startup()  # the allocator a held run takes, set before torch is imported
    import gc

    import numpy
    import torch

    from scratch import Scratch  # the CLI's store, as hold_probe.py hands it
    from seedvr2x import model, shot_pass, shot_plan, tilegrid, tiling
    from seedvr2x.frames import padded_length, padded_size
    from seedvr2x.passes import Tiles

    gib = 1024**3
    device = torch.device("cuda:0")
    out = open(args.out, "a")

    def say(record: dict) -> None:
        line = json.dumps({"time": time.strftime("%H:%M:%S"), **record})
        print(line, flush=True)
        out.write(line + "\n")
        out.flush()

    models = model.load_models(Path(args.dit), Path(args.vae), device)
    models.runner.dit.to("cpu")
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    limit = memory.hold(device)
    size = int(args.size)
    target = (size, round(size * 16 / 9))
    padded = padded_size(target)
    height, width = padded
    if args.sides:
        sides = [int(side) for side in args.sides.split(",")]
    else:
        sides = sorted(shot_plan._tile_sizes(padded))
    sides = [s for s in sides if s >= args.low and (not args.high or s <= args.high)]
    latents = padded_length(args.frames) // 4 + 1 if args.frames > 1 else 1
    say(
        {
            "start": True,
            "target": target,
            "padded": padded,
            "frames": args.frames,
            "latents": latents,
            "sides": sides,
            "limit_gib": None if limit is None else limit / gib,
            "loaded_gib": torch.cuda.memory_allocated(device) / gib,
        }
    )

    rng = numpy.random.default_rng(13)
    source = rng.random((args.frames, target[0], target[1], 3), numpy.float32)
    generator = torch.Generator().manual_seed(11)
    values = torch.randn((16, latents, height // 8, width // 8), generator=generator)
    latent = values.to(model.COMPUTE_DTYPE).permute(1, 2, 3, 0)

    def encode(frames: int, side: int) -> None:
        position = [0]

        def read(count: int) -> numpy.ndarray:
            position[0] += count
            return source[position[0] - count : position[0]]

        inputs = shot_pass.encoder_inputs(models, read, frames, target)
        scratch = Scratch(None, 2**40)  # always in system memory
        tiling.encode_tiled(models, inputs, padded_length(frames), Tiles(side), scratch)

    def decode(count: int, side: int) -> None:
        scratch = Scratch(None, 2**40)
        for piece in tiling.decode_tiled(models, latent[:count], Tiles(side), scratch):
            del piece

    for which in args.passes.split(","):
        times: dict[int, list[float]] = {}
        for side in sides:
            grid = tilegrid.grid(tilegrid.latent_size(padded), Tiles(side))
            across = len({tile.left for tile in grid})
            shape = {"pass": which, "side": side, "tiles": len(grid), "across": across}
            full = args.frames if which == "encode" else latents
            short = min(full, 9 if which == "encode" else 3)
            runs = [("warm-up", short)] + [(f"rep{n + 1}", full) for n in range(args.reps)]
            refused = False
            for name, amount in runs:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                try:
                    with torch.no_grad():
                        if which == "encode":
                            encode(amount, side)
                        else:
                            decode(amount, side)
                    torch.cuda.synchronize(device)
                except torch.OutOfMemoryError as error:
                    say({**shape, "run": name, "refused": str(error).splitlines()[0][:200]})
                    refused = True
                    break
                wall = time.perf_counter() - started
                record = {
                    **shape,
                    "run": name,
                    "amount": amount,
                    "wall_s": round(wall, 4),
                    "allocated_gib": round(torch.cuda.max_memory_allocated(device) / gib, 3),
                    "reserved_gib": round(torch.cuda.max_memory_reserved(device) / gib, 3),
                }
                say(record)
                if name != "warm-up":
                    times.setdefault(side, []).append(wall)
            gc.collect()
            torch.cuda.empty_cache()
            if refused:
                break
            say({**shape, "median_s": round(statistics.median(times[side]), 4)})
    return 0


if __name__ == "__main__":
    sys.exit(main())

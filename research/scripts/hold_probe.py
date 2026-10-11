#!/usr/bin/env python3
"""Hold probe for seedvr2x: how a process can hold itself to a card's real memory under a driver
that gives system memory instead of refusing (Windows' sysmem fallback, WSL2 included), and what
a window that passes the card does under each way.

One process, one way of holding, a sequence of steps; every step's line is printed and written
at once, since a step may kill the process (a sticky CUDA error aborts it in a destructor).

The ways (options, combined freely):
  --allocator async|native   PYTORCH_CUDA_ALLOC_CONF's backend (async: cudaMallocAsync, the
                             command line's default), set before torch is imported
  --gc FRACTION              native only: garbage_collection_threshold
  --cap GIB                  torch.cuda.set_per_process_memory_fraction, as GiB the process's
                             allocator may hold beyond what it holds once the models are loaded
                             as --placement says (so: the budget of a window's work); 0: none
  --under GIB                the same cap, given as GiB under the memory the driver reports
                             free at that moment (a margin)
  --trim                     torch.cuda.empty_cache() before every step

The steps (--steps, in order, comma-separated):
  L      a window of L latents at --size through the DiT (placement_bench.py's forward), seeds
         42 and 43: digests against --reference (a JSON this probe wrote with --write-reference,
         which adds the steps its file doesn't have yet and never replaces one it has)
  vF     a VAE decode of a latent of F latent frames at --size, streamed as the runtime does it
  trim   torch.cuda.empty_cache()
  host   the DiT's parameters to pinned host copies and back whole (blockswap.host, a placement
         with nothing swapped): the sequence after which other bits were seen once. Only with
         --placement 0: under a placement the swapped blocks' parameters are their slots', and
         copying them from there would give other bits by the probe's own doing. A host step
         refused half-way leaves the DiT split: the sequence ends there.
A line is written when a step starts, and one when it ends: torch's peak allocated and reserved,
the driver's free memory before and at the least, the time, the digests and whether they are the
reference's; or the refusal, its kind (an OutOfMemoryError Python caught, or another error) and
its time; then, apart, what the driver and torch say after it, which a CUDA error may refuse.

The cap is torch's: under the native allocator it bounds the memory torch has RESERVED from the
driver, its cache included; under cudaMallocAsync it bounds what torch has ALLOCATED, and the
pool's slack, which is what spills, is outside it: least_free tells what the card really held.

--placement N loads through nothing (0, the default: the models whole) or swaps the DiT's last N
blocks (a plan's placement, prefetched): N = 36 leaves the card to the windows, for references
of windows that pass it otherwise.

Usage (seedvr2x's venv python, under the GPU's lock; steps that pass the card fill it on purpose
and slow a desktop on it):
  python hold_probe.py --dit DIT --vae VAE --placement 36 --steps 1,6,11 --write-reference ref.json
  timeout 900 python hold_probe.py --dit DIT --vae VAE --allocator native --under 0.6 \
      --steps 6,12,6 --reference ref.json --out native-under0.6.jsonl
(under `timeout`: a step far past the card may run very long under such a driver)
"""

import argparse
import json
import os
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dit", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--allocator", choices=("async", "native"), default="async")
    parser.add_argument("--gc", type=float)
    parser.add_argument("--cap", type=float, default=0.0)
    parser.add_argument("--under", type=float)
    parser.add_argument("--trim", action="store_true")
    parser.add_argument("--placement", type=int, default=0)
    parser.add_argument("--size", default="1080")
    parser.add_argument("--steps", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--write-reference")
    parser.add_argument("--out")
    args = parser.parse_args()
    conf = "backend:cudaMallocAsync" if args.allocator == "async" else "backend:native"
    if args.gc is not None:
        conf += f",garbage_collection_threshold:{args.gc}"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = conf  # before torch, and before placement_bench

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import placement_bench as bench
    from pathlib import Path

    import torch

    from seedvr2x.runtime import blockswap, job, model

    gib = bench.GIB
    device = torch.device("cuda:0")
    out = open(args.out, "a") if args.out else None

    def say(record: dict) -> None:
        line = json.dumps(record)
        print(line, flush=True)
        if out:
            out.write(line + "\n")
            out.flush()
            os.fsync(out.fileno())

    def free() -> float:
        return round(torch.cuda.mem_get_info(device)[0] / gib, 3)

    def words(error: BaseException) -> str:
        """An error's first line: torch's own go on with advice, and some have no word."""
        return str(error).partition("\n")[0][:160]

    if "host" in args.steps.split(",") and args.placement:
        sys.exit("hold_probe: the host step needs --placement 0 (the docstring says why)")
    reference = {}
    if args.reference:
        with open(args.reference) as file:
            reference = json.load(file)
    say({"conf": conf, "cap": args.cap, "under": args.under, "trim": args.trim,
         "placement": args.placement, "steps": args.steps, "free_start": free(),
         "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__})
    models = model.load_models(Path(args.dit), Path(args.vae), device)
    dit = models.runner.dit
    backend = blockswap.Cuda(device)
    placement = None
    if args.placement:
        blocks = len(dit.blocks)
        fixed = blockswap.dit_parts(blocks, blocks - args.placement, False)
        parts = blockswap.host(dit, fixed, backend)
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        placement = blockswap.Placement(dit, device, args.placement, parts=parts, backend=backend)
        placement.enter("dit")
    torch.cuda.synchronize(device)
    held = torch.cuda.memory_reserved(device)
    total = torch.cuda.mem_get_info(device)[1]
    cap = args.cap * gib
    if args.under is not None:
        cap = torch.cuda.mem_get_info(device)[0] - args.under * gib
    if cap:
        torch.cuda.set_per_process_memory_fraction((held + cap) / total, device)
    say({"loaded": True, "allocated": round(torch.cuda.memory_allocated(device) / gib, 3),
         "reserved": round(held / gib, 3), "free": free(), "cap_gib": round(cap / gib, 3)})
    target = bench.parse_size(args.size)
    height, width = job.padded_size(target)
    # A reference being written keeps what its file has already: other sizes' steps, and this
    # size's from an earlier run, which a later run never replaces (a reference is what a run
    # known to be right gave; a step run again is compared with it instead).
    new_reference: dict = {}
    if args.write_reference and os.path.exists(args.write_reference):
        with open(args.write_reference) as file:
            new_reference = json.load(file)

    def key(step: str) -> str:
        """A step's name in a reference: its size with it."""
        return f"{args.size}:{step}"

    def keep(step: str, digests) -> None:
        """A step's digests, whole, into the reference being written, unless it has that step
        already: at once, since the next step may kill the process."""
        if args.write_reference and key(step) not in new_reference:
            new_reference[key(step)] = digests
            with open(args.write_reference + ".tmp", "w") as file:
                json.dump(new_reference, file, indent=1)
            os.replace(args.write_reference + ".tmp", args.write_reference)

    def window(length: int) -> dict:
        latent = bench.make_latent(torch, model.COMPUTE_DTYPE, length, height // 8, width // 8)
        digests, walls, peaks, reserved = [], [], [], []
        for seed in (42, 43):
            run = bench.one_forward(torch, model, models, latent, seed)  # resets the peaks
            digests.append(run["sha256"])
            walls.append(round(run["wall_s"], 3))
            peaks.append(round(run["peak_gib"], 3))
            reserved.append(round(torch.cuda.max_memory_reserved(device) / gib, 3))
        keep(str(length), digests)
        known = reference.get(key(str(length)), new_reference.get(key(str(length))))
        return {"digests": [d[:12] for d in digests], "walls": walls, "peak": max(peaks),
                "peak_reserved": max(reserved),
                "equal": None if known is None else digests == known}

    def vae(frames: int) -> dict:
        import hashlib

        generator = torch.Generator().manual_seed(11)
        values = torch.randn((16, frames, height // 8, width // 8), generator=generator)
        latent = values.to(model.COMPUTE_DTYPE).permute(1, 2, 3, 0).to(device)
        torch.cuda.reset_peak_memory_stats(device)
        digest = hashlib.sha256()
        with torch.no_grad():
            for piece in model.decode_stream(models, latent):
                digest.update(piece.float().cpu().numpy().tobytes())
        keep(f"v{frames}", digest.hexdigest())
        known = reference.get(key(f"v{frames}"), new_reference.get(key(f"v{frames}")))
        return {"digest": digest.hexdigest()[:12],
                "peak": round(torch.cuda.max_memory_allocated(device) / gib, 3),
                "peak_reserved": round(torch.cuda.max_memory_reserved(device) / gib, 3),
                "equal": None if known is None else digest.hexdigest() == known}

    def rehost() -> dict:
        blocks = len(dit.blocks)
        parts = blockswap.host(dit, blockswap.dit_parts(blocks, 0, other=True), backend)
        torch.cuda.synchronize(device)
        moved = blockswap.Placement(dit, device, 0, parts=parts, backend=backend)
        moved.enter("dit")
        torch.cuda.synchronize(device)
        return {"pinned_gib": round(moved.host_bytes / gib, 2)}

    for step in args.steps.split(","):
        record = {"step": step}
        ended = False
        started = time.perf_counter()
        sampler = None
        try:
            # Everything that asks CUDA is in here: after a sticky error any call may raise.
            if args.trim or step == "trim":
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
            record.update(free_before=free(),
                          reserved_before=round(torch.cuda.memory_reserved(device) / gib, 3))
            say({"starting": step, **record})
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            with bench.FreeSampler(torch, device) as sampler:
                if step == "trim":
                    result = {}
                elif step == "host":
                    result = rehost()
                elif step.startswith("v"):
                    result = vae(int(step[1:]))
                else:
                    result = window(int(step))
            record.update(result, ok=True)
        except torch.OutOfMemoryError as error:
            record.update(ok=False, refused="OutOfMemoryError", words=words(error))
            ended = step == "host"
        except Exception as error:  # the kind matters: reported, and the sequence goes on
            record.update(ok=False, refused=type(error).__name__, words=words(error))
            ended = step == "host"
        record.update(s=round(time.perf_counter() - started, 2))
        if sampler is not None:
            record.update(least_free=round(sampler.least, 3))
        say(record)
        try:
            say({"after": step, "free_after": free(),
                 "allocated": round(torch.cuda.memory_allocated(device) / gib, 3),
                 "reserved": round(torch.cuda.memory_reserved(device) / gib, 3)})
        except Exception as error:
            say({"after": step, "error": type(error).__name__, "words": words(error)})
        if ended:
            say({"ended": "the host step was refused: the DiT may be split, host and device"})
            break
    say({"done": True})
    return 0


if __name__ == "__main__":
    sys.exit(main())

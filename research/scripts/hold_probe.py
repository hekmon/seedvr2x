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
  --split MIB                native only: max_split_size_mb, the size from which the allocator
                             no longer splits a block it has cached for a smaller request
  --cap GIB                  torch.cuda.set_per_process_memory_fraction, as GiB the process's
                             allocator may hold beyond what it holds once the models are loaded
                             as --placement says (so: the budget of a window's work); 0: none
  --under GIB                the same cap, given as GiB under the memory free at that moment (a
                             margin): the smaller of what the driver reports free to the process
                             (CUDA's figure) and of what the whole card has free (NVML's, read
                             by seedvr2x's runtime/memory.py, card_free), as seedvr2x's own hold
                             takes it; CUDA's figure alone where NVML doesn't answer
  --trim                     torch.cuda.empty_cache() before every step

Two free figures are read wherever one is: `free`, CUDA's, which under such a driver is the
process's own while the card has room, and `card`, NVML's, the whole card's. A step's line has
the least of each while it ran (least_free, least_card), sampled from a thread.

The steps (--steps, in order, comma-separated):
  L      a window of L latents at --size through the DiT (placement_bench.py's forward), seeds
         42 and 43: digests against --reference (a JSON this probe wrote with --write-reference,
         which adds the steps its file doesn't have yet and never replaces one it has)
  vF     a VAE decode of a latent of F latent frames at --size, streamed as the runtime does it
  dF/T   the same decode in tiles of T pixels, as the runtime tiles it (tiling.decode_tiled),
         the latent in system memory and the scratch too
  eN/T   a VAE encode of N frames (4n + 1) at --size in tiles of T pixels (tiling.encode_tiled),
         the frames random ones at a quarter of the size, through the runtime's own transform
  trim   torch.cuda.empty_cache()
  under=GIB
         the cache emptied, then torch's limit set again, GIB under the memory free now, as
         --under sets it: a ladder of holds in one process, the safest first
  r:STEP[:FROM[:BY]]
         what the allocator needs beyond STEP's allocated peak: STEP once under the present
         limit, for its peak; then again and again, the cache emptied before each, under a
         limit of that peak and a margin, from FROM GiB (2.4) down by BY (0.1), until it is
         refused (a window's rungs run seed 42 alone); the present limit is set again after
  host   the DiT's parameters to pinned host copies and back whole (blockswap.host, a placement
         with nothing swapped): the sequence after which other bits were seen once. Only with
         --placement 0: under a placement the swapped blocks' parameters are their slots', and
         copying them from there would give other bits by the probe's own doing. A host step
         refused half-way leaves the DiT split: the sequence ends there.
A line is written when a step starts, and one when it ends: torch's peak allocated and reserved
(in GiB and in bytes), the limit it ran under, both free figures before and at the least, the
time, the digests and whether they are the reference's; or the refusal, its kind (an
OutOfMemoryError Python caught, or another error) and its time; then, apart, what the driver,
the card and torch say after it, which a CUDA error may refuse.

The cap is torch's: under the native allocator it bounds the memory torch has RESERVED from the
driver, its cache included; under cudaMallocAsync it bounds what torch has ALLOCATED, and the
pool's slack, which is what spills, is outside it: least_card tells what the card really held.

--placement N loads through nothing (0, the default: the models whole) or swaps the DiT's last N
blocks (a plan's placement, prefetched): N = 36 leaves the card to the windows, for references
of windows that pass it otherwise.

Before the models load, the free memory is read as seedvr2x's command line reads it for its plan
(cli._free_memory, then the card's): the `planned` line. What the process holds on the card
beyond torch's own count, from that reading on, is what a hold set then must leave outside
torch's limit: (card at the reading - card at a step's least) - (reserved at its peak -
reserved at the reading).

Usage (seedvr2x's venv python, under the GPU's lock; steps that pass the card fill it on purpose
and slow a desktop on it):
  python hold_probe.py --dit DIT --vae VAE --placement 36 --steps 1,6,11 --write-reference ref.json
  timeout 900 python hold_probe.py --dit DIT --vae VAE --allocator native --under 0.6 \
      --steps 6,12,6 --reference ref.json --out native-under0.6.jsonl
  timeout 900 python hold_probe.py --dit DIT --vae VAE --allocator native --under 1.0 \
      --steps 8,r:8,under=0.5,8,12,6 --reference ref.json --out ladder.jsonl
(under `timeout`: a step far past the card may run very long under such a driver)
"""

import argparse
import json
import os
import sys
import threading
import time

SEEDS = (42, 43)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dit", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--allocator", choices=("async", "native"), default="async")
    parser.add_argument("--gc", type=float)
    parser.add_argument("--split", type=int)
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
    if args.split is not None:
        conf += f",max_split_size_mb:{args.split}"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = conf  # before torch, and before placement_bench

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import hashlib
    from pathlib import Path

    import numpy
    import placement_bench as bench
    import torch

    from seedvr2x import cli
    from seedvr2x.runtime import blockswap, job, memory, model, shot, tiling
    from seedvr2x.runtime.plan import Tiles

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

    def card() -> float | None:
        """The whole card's free memory, GiB, by NVML; None where it doesn't answer."""
        seen = memory.card_free(0)
        return None if seen is None else round(seen / gib, 3)

    def words(error: BaseException) -> str:
        """An error's first line: torch's own go on with advice, and some have no word."""
        return str(error).partition("\n")[0][:160]

    class Sampler:
        """The least memory free while a step ran, sampled from a thread every 20 ms and the
        time a reading takes: what the driver reports to this process (least) and what the
        whole card has (card). A card whose driver spills into system memory went past its own
        memory when the second reaches 0."""

        def __init__(self) -> None:
            self.least, self.card = float("inf"), float("inf")
            self.stop = threading.Event()
            self.thread = threading.Thread(target=self.run, daemon=True)

        def read(self) -> None:
            self.least = min(self.least, torch.cuda.mem_get_info(device)[0] / gib)
            seen = memory.card_free(0)
            if seen is not None:
                self.card = min(self.card, seen / gib)

        def run(self) -> None:
            while not self.stop.is_set():
                self.read()
                self.stop.wait(0.02)

        def __enter__(self) -> "Sampler":
            self.thread.start()
            return self

        def __exit__(self, *exc: object) -> None:
            self.stop.set()
            self.thread.join()
            self.read()

    if "host" in args.steps.split(",") and args.placement:
        sys.exit("hold_probe: the host step needs --placement 0 (the docstring says why)")
    reference = {}
    if args.reference:
        with open(args.reference) as file:
            reference = json.load(file)
    say({"conf": conf, "cap": args.cap, "under": args.under, "trim": args.trim,
         "placement": args.placement, "steps": args.steps, "size": args.size,
         "free_start": free(), "card_start": card(),
         "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__})
    # The command line's own reading for its plan, on its footing, and the card's right after,
    # as memory.usable takes them: where seedvr2x sets its hold, before the models load.
    planned = cli._free_memory(device)
    seen = memory.card_free(0)
    say({"planned": True, "free": round(planned / gib, 4), "free_bytes": planned,
         "card": None if seen is None else round(seen / gib, 4), "card_bytes": seen,
         "reserved_bytes": torch.cuda.memory_reserved(device)})
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
    total = torch.cuda.mem_get_info(device)[1]
    # torch's limit now, in bytes of what it has reserved from the driver; None: none set.
    state: dict = {"limit": None}

    def limit_to(limit: float | None, **said: object) -> None:
        """torch's limit set to `limit` bytes reserved in all (None: the whole card's)."""
        fraction = 1.0 if limit is None else min(limit / total, 1.0)
        torch.cuda.set_per_process_memory_fraction(fraction, device)
        state["limit"] = limit
        say({"limit": None if limit is None else round(limit / gib, 3), **said})

    def hold(under: float) -> None:
        """torch's limit set `under` GiB under the memory free now, the smaller of the driver's
        figure for this process and of the whole card's, and what torch holds now."""
        torch.cuda.synchronize(device)
        reported = torch.cuda.mem_get_info(device)[0]
        seen = memory.card_free(0)
        taken = reported if seen is None else min(reported, seen)
        reserved = torch.cuda.memory_reserved(device)
        limit_to(reserved + taken - under * gib, under=under, free=round(reported / gib, 3),
                 card=None if seen is None else round(seen / gib, 3),
                 reserved=round(reserved / gib, 3))

    held = torch.cuda.memory_reserved(device)
    if args.under is not None:
        hold(args.under)
    elif args.cap:
        limit_to(held + args.cap * gib, cap=args.cap)
    say({"loaded": True, "allocated": round(torch.cuda.memory_allocated(device) / gib, 3),
         "reserved": round(held / gib, 3), "reserved_bytes": held, "free": free(),
         "card": card()})
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

    def known(step: str):
        return reference.get(key(step), new_reference.get(key(step)))

    def peaks() -> dict:
        allocated = torch.cuda.max_memory_allocated(device)
        reserved = torch.cuda.max_memory_reserved(device)
        return {"peak": round(allocated / gib, 3), "peak_reserved": round(reserved / gib, 3),
                "peak_bytes": allocated, "peak_reserved_bytes": reserved}

    def window(length: int, seeds=SEEDS) -> dict:
        latent = bench.make_latent(torch, model.COMPUTE_DTYPE, length, height // 8, width // 8)
        digests, walls, most = [], [], peaks()
        for seed in seeds:
            run = bench.one_forward(torch, model, models, latent, seed)  # resets the peaks
            digests.append(run["sha256"])
            walls.append(round(run["wall_s"], 3))
            now = peaks()
            if seed == seeds[0]:
                most = now
            for name in ("peak", "peak_reserved", "peak_bytes", "peak_reserved_bytes"):
                most[name] = max(most[name], now[name])
        if tuple(seeds) == SEEDS:
            keep(str(length), digests)
        both = known(str(length))
        equal = None if both is None else digests == [both[SEEDS.index(s)] for s in seeds]
        return {"digests": [d[:12] for d in digests], "walls": walls, **most, "equal": equal,
                "latents": length}

    def latent_of(frames: int):
        """A shot's latent of `frames` latent frames, (T', h, w, 16) in channel-major memory, in
        system memory, from a seeded generator."""
        generator = torch.Generator().manual_seed(11)
        values = torch.randn((16, frames, height // 8, width // 8), generator=generator)
        return values.to(model.COMPUTE_DTYPE).permute(1, 2, 3, 0)

    def decoded(step: str, pieces) -> dict:
        digest = hashlib.sha256()
        with torch.no_grad():
            for piece in pieces:
                digest.update(piece.float().cpu().numpy().tobytes())
        keep(step, digest.hexdigest())
        return {"digest": digest.hexdigest()[:12], **peaks(),
                "equal": None if known(step) is None else digest.hexdigest() == known(step)}

    def vae(step: str, frames: int) -> dict:
        latent = latent_of(frames).to(device)
        torch.cuda.reset_peak_memory_stats(device)
        return decoded(step, model.decode_stream(models, latent))

    def decode_tiled(step: str, frames: int, tile: int) -> dict:
        latent = latent_of(frames)
        scratch = tiling.Scratch(None, 2**40)  # always in system memory
        torch.cuda.reset_peak_memory_stats(device)
        return decoded(step, tiling.decode_tiled(models, latent, Tiles(tile), scratch))

    def encode_tiled(step: str, frames: int, tile: int) -> dict:
        generator = numpy.random.default_rng(13)
        source = generator.random((frames, target[0] // 4, target[1] // 4, 3), numpy.float32)
        position = [0]

        def read(count: int):
            position[0] += count
            return source[position[0] - count : position[0]]

        scratch = tiling.Scratch(None, 2**40)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            inputs = shot.encoder_inputs(models, read, frames, target)
            latent = tiling.encode_tiled(
                models, inputs, shot.padded_length(frames), Tiles(tile), scratch
            )
        digest = hashlib.sha256(latent.float().cpu().numpy().tobytes()).hexdigest()
        keep(step, digest)
        return {"digest": digest[:12], **peaks(),
                "equal": None if known(step) is None else digest == known(step)}

    def rehost() -> dict:
        blocks = len(dit.blocks)
        parts = blockswap.host(dit, blockswap.dit_parts(blocks, 0, other=True), backend)
        torch.cuda.synchronize(device)
        moved = blockswap.Placement(dit, device, 0, parts=parts, backend=backend)
        moved.enter("dit")
        torch.cuda.synchronize(device)
        return {"pinned_gib": round(moved.host_bytes / gib, 2)}

    def run(step: str, seeds) -> dict:
        if step == "trim":
            return {}
        if step == "host":
            return rehost()
        if step.startswith("v"):
            return vae(step, int(step[1:]))
        if step.startswith(("d", "e")):
            count, tile = step[1:].split("/")
            tiled = decode_tiled if step.startswith("d") else encode_tiled
            return tiled(step, int(count), int(tile))
        return window(int(step), seeds)

    def trim() -> None:
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()

    def attempt(step: str, seeds=SEEDS, **said: object) -> dict:
        """One step, its lines written; returns its record (ok: whether it ended)."""
        record: dict = {"step": step, **said}
        started = time.perf_counter()
        sampler = None
        try:
            # Everything that asks CUDA is in here: after a sticky error any call may raise.
            if args.trim or step == "trim":
                trim()
            record.update(free_before=free(), card_before=card(),
                          reserved_before=round(torch.cuda.memory_reserved(device) / gib, 3))
            say({"starting": step, **record})
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            with Sampler() as sampler:
                result = run(step, seeds)
            record.update(result, ok=True)
        except torch.OutOfMemoryError as error:
            record.update(ok=False, refused="OutOfMemoryError", words=words(error))
        except Exception as error:  # the kind matters: reported, and the sequence goes on
            record.update(ok=False, refused=type(error).__name__, words=words(error))
        record.update(s=round(time.perf_counter() - started, 2))
        if "peak_bytes" not in record and step not in ("trim", "host"):
            try:  # a refused step's peaks too: what torch held when it was refused
                record.update(peaks())
            except Exception as error:
                record.update(peaks_error=type(error).__name__)
        limit = state["limit"]
        record.update(limit=None if limit is None else round(limit / gib, 3))
        if sampler is not None:
            record.update(least_free=round(sampler.least, 3), least_card=round(sampler.card, 3))
        say(record)
        try:
            say({"after": step, "free_after": free(), "card_after": card(),
                 "allocated": round(torch.cuda.memory_allocated(device) / gib, 3),
                 "reserved": round(torch.cuda.memory_reserved(device) / gib, 3),
                 "reserved_bytes": torch.cuda.memory_reserved(device)})
        except Exception as error:
            say({"after": step, "error": type(error).__name__, "words": words(error)})
        return record

    def ladder(step: str, start: float, by: float) -> None:
        """What the allocator needs beyond `step`'s allocated peak: the step under the present
        limit, then under its peak and a margin going down, the cache emptied before each."""
        present = state["limit"]
        trim()
        first = attempt(step, ladder="peak")
        if not first.get("ok"):
            return
        peak, margin = first["peak_bytes"], start
        while margin > -1e-9:
            limit = peak + round(margin * gib)
            if present is None or limit <= present:
                trim()
                limit_to(limit, ladder=step, margin=round(margin, 2))
                if not attempt(step, SEEDS[:1], ladder=round(margin, 2)).get("ok"):
                    break
            margin -= by
        trim()
        limit_to(present, ladder=step, restored=True)

    for step in args.steps.split(","):
        if step.startswith("under="):
            trim()
            hold(float(step.partition("=")[2]))
            continue
        if step.startswith("r:"):
            of, start, by = (*step[2:].split(":"), "", "")[:3]
            ladder(of, float(start or 2.4), float(by or 0.1))
            continue
        record = attempt(step)
        if step == "host" and not record.get("ok"):
            say({"ended": "the host step was refused: the DiT may be split, host and device"})
            break
    say({"done": True})
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Placement probe for seedvr2x: what its BlockSwap (runtime/blockswap.py) changes, takes and costs.

seedvr2x swaps the DiT's last S blocks through two device buffers, from pinned host copies, with
or without a prefetch on a side stream, and sends the DiT and the VAE away between phases
(DESIGN.md, BlockSwap, rewritten). This probe measures it on seedvr2x's own runtime: whether the
DiT's output keeps its bits, the DiT's peak, the time of a forward, the memory each phase leaves on
the device, the host memory pinned.

One forward is what sample_windows does for one window, on random latents (the convention of
compile_dit.py): a latent (L, h, w, 16) bfloat16 in channel-major memory, as the encode gives it,
drawn from a seeded CPU generator, goes to the device; the seed is set, the noise drawn, the "sr"
condition built, then model.sample runs one Euler step under bfloat16 autocast. Its output is
known by the SHA-256 of its bytes; forward i of a step takes seed 42 + i.

  matrix  the models loaded whole (runtime/model.py, load_models, no plan), then, in one process:
          "loaded", the DiT as loaded; then every parameter of the DiT given a pinned host copy
          (blockswap.host) and, for each S of --swapped, the placement without the prefetch
          ("sync") and with it ("prefetch"), S = 0 being the DiT restored whole from its copies.
          Each runs --forwards forwards of every length of --lengths (a first one, untimed, warms
          up), at every size of --sizes. A step whose predicted peak passes the memory free at the
          start less --reserve is skipped, and said: on a desktop card the driver may spill into
          system memory rather than refuse. --rounds repeats the placements' passes in turn: on
          a card that drives a desktop, a time varies more from one minute to the next than
          between two placements, and the table gives each step's least time over the rounds.
          With --verify-slots, one more pass at the largest S compares every slot with its host
          copy before its block runs (Placement's verify).
  plan    the models loaded through a plan (load_models(plan=Plan(...))): S blocks swapped, the
          blocks a VAE phase keeps (--resident, 0: the DiT away), the VAE away in the DiT's phase
          (--vae-away). Measures the load (time, the device's peak, the host memory pinned), then
          each phase's memory and the time to enter it, over --cycles cycles of encode, dit,
          decode, with the forwards of --lengths in the DiT's phase, and a small VAE encode and
          decode in the VAE's. --reference names a matrix's results: the digests must be its own.
          --no-trim leaves what a phase frees to the allocator, for the comparison.
          Run it under vram_cap.py for a smaller card.
  table   RESULTS.json [...]   a compact Markdown summary per file

SIZE = 720 | 1080 | 2160 (16:9 output) or WxH, padded as seedvr2x pads it (runtime/job.py,
padded_size: 1920x1080 to 1920x1104, a latent of 138 x 240, 8,280 tokens).

The CLI's allocator default (PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync, cli.py) is applied
before torch is imported.

Usage (seedvr2x's venv python; under the GPU's lock):
  python placement_bench.py matrix --dit DIT.safetensors --vae VAE.safetensors \
      --sizes 1080 --lengths 1,6,12 --swapped 0,3,12,36 --forwards 5 --out matrix.json
  VRAM_CAP_GIB=16 python vram_cap.py placement_bench.py plan --dit ... --vae ... --swapped 36 \
      --resident 0 --vae-away --size 1080 --lengths 5 --reference matrix.json --out cap16.json
"""

import argparse
import hashlib
import json
import os
import resource
import statistics
import sys
import threading
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")  # the CLI's default

GIB = 1024**3
# The DiT's peak (7B fp16) by DESIGN.md's fit, for the guard only: 15.87 GiB with every weight on
# the device, 127.16 KiB per token (3% more stands for the attention windows' term).
FIT_GIB, TOKEN_GIB = 15.87, 1.03 * 127.16 / 1024**2
SIZES = {"720": (720, 1280), "1080": (1080, 1920), "2160": (2160, 3840)}


def log(message):
    print(f"placement_bench: {message}", flush=True)


def parse_size(text):
    if text in SIZES:
        return SIZES[text]
    width, height = text.lower().split("x")
    return int(height), int(width)


def numbers(text):
    return [int(part) for part in text.split(",") if part]


def rss_gib():
    with open("/proc/self/status") as file:
        for line in file:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024**2
    return 0.0


def available_gib():
    with open("/proc/meminfo") as file:
        for line in file:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024**2
    return 0.0


class FreeSampler:
    """The least device memory the driver reported free while it ran, sampled every 20 ms from a
    thread: on a card whose driver spills into system memory, a peak that reaches 0 free went
    past the card."""

    def __init__(self, torch, device):
        self.torch, self.device = torch, device
        self.least = float("inf")
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.is_set():
            self.least = min(self.least, self.torch.cuda.mem_get_info(self.device)[0] / GIB)
            self.stop.wait(0.02)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self.least = min(self.least, self.torch.cuda.mem_get_info(self.device)[0] / GIB)


def make_latent(torch, dtype, length, h, w):
    """The latent of a window of `length` latents: (length, h, w, 16) in channel-major memory, as
    the encode gives it (runtime/model.py, encode_stream), from a seeded CPU generator."""
    generator = torch.Generator().manual_seed(1000003 * length + 1009 * h + w)
    values = torch.randn((16, length, h, w), generator=generator, dtype=torch.float32)
    return values.to(dtype).permute(1, 2, 3, 0)


def digest(torch, out):
    """SHA-256 of a bfloat16 tensor's bytes, read as int16, on the CPU."""
    return hashlib.sha256(out.contiguous().view(torch.int16).numpy().tobytes()).hexdigest()


def one_forward(torch, model, models, latent_cpu, seed):
    """sample_windows' body for one window: the measurements and the output's digest."""
    device = models.device
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    with torch.no_grad():
        latent = latent_cpu.to(device)
        model.set_seed(seed)
        noise = torch.randn_like(latent, dtype=model.COMPUTE_DTYPE)
        cond = model.condition(models, noise, latent)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        events[0].record()
        out = model.sample(models, noise, cond)
        events[1].record()
        torch.cuda.synchronize(device)
        wall = time.perf_counter() - started
        gpu = events[0].elapsed_time(events[1]) / 1000
        result = out.to("cpu")
    peak = torch.cuda.max_memory_allocated(device) / GIB
    del latent, noise, cond, out
    return {"wall_s": wall, "gpu_s": gpu, "peak_gib": peak, "sha256": digest(torch, result)}


def run_step(torch, model, models, target, length, forwards, record):
    """`forwards` forwards of one length at one size, after a warm-up one: record gets the
    digests (the warm-up's first), the peak, the times."""
    from seedvr2x.runtime import job

    height, width = job.padded_size(target)
    latent = make_latent(torch, model.COMPUTE_DTYPE, length, height // 8, width // 8)
    with FreeSampler(torch, models.device) as sampler:
        runs = [one_forward(torch, model, models, latent, 42 + i) for i in range(forwards + 1)]
    timed = runs[1:]
    record.update(
        size=f"{target[1]}x{target[0]}",
        length=length,
        tokens=length * (height // 16) * (width // 16),
        forwards=forwards,
        digests=[run["sha256"] for run in runs],
        peak_gib=round(max(run["peak_gib"] for run in runs), 4),
        wall_s=round(statistics.median(run["wall_s"] for run in timed), 4),
        wall_min_s=round(min(run["wall_s"] for run in timed), 4),
        wall_max_s=round(max(run["wall_s"] for run in timed), 4),
        gpu_s=round(statistics.median(run["gpu_s"] for run in timed), 4),
        first_s=round(runs[0]["wall_s"], 4),
        least_free_gib=round(sampler.least, 3),
    )
    return record


def predicted(target, length, swapped, slots):
    """The DiT's peak for the guard, in GiB: the fit less the swapped blocks, but the slots."""
    from seedvr2x.runtime import job, weights

    height, width = job.padded_size(target)
    tokens = length * (height // 16) * (width // 16)
    saved = max(swapped - slots, 0) * weights.DIT_7B.blocks[0] / GIB
    return FIT_GIB - saved + TOKEN_GIB * tokens


def vae_digest(torch, model, models):
    """A small VAE round trip's digest: 5 frames of 256x256 from a seeded CPU generator,
    encoded and decoded as the runtime does a shot in one piece."""
    generator = torch.Generator().manual_seed(7)
    video = torch.rand((3, 5, 256, 256), generator=generator) * 2 - 1
    with torch.no_grad():
        latent = model.encode(models, video.to(models.device, model.COMPUTE_DTYPE))
        decoded = model.decode(models, latent)
        both = torch.cat([latent.reshape(-1).float(), decoded.reshape(-1).float()]).cpu()
    return hashlib.sha256(both.numpy().tobytes()).hexdigest()


def setup(args):
    import torch

    from seedvr2x.runtime import blockswap, model, weights

    device = torch.device("cuda:0")
    free, total = torch.cuda.mem_get_info(device)
    results = {
        "command": " ".join(sys.argv[1:]),
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "allocator": os.environ["PYTORCH_CUDA_ALLOC_CONF"],
        "cap_gib": os.environ.get("VRAM_CAP_GIB"),
        "free_start_gib": round(free / GIB, 3),
        "total_gib": round(total / GIB, 3),
        "host_available_start_gib": round(available_gib(), 2),
        "attention": model.attention_backend(),
        "block_bytes": weights.DIT_7B.blocks[0],
        "steps": [],
    }
    log(f"{results['gpu']}: {free / GIB:.2f} GiB free of {total / GIB:.2f}")
    return torch, blockswap, model, device, results


def write(args, results):
    if args.out:
        with open(args.out + ".tmp", "w") as file:
            json.dump(results, file, indent=1)
        os.replace(args.out + ".tmp", args.out)


def say(record):
    equal = record.get("equal")
    mark = "" if equal is None else (" bits EQUAL" if equal else " BITS DIFFER")
    log(
        f"{record['config']:<22} {record['size']} L={record['length']:<3}"
        f" peak {record['peak_gib']:.3f} GiB, wall {record['wall_s']:.3f} s"
        f" (gpu {record['gpu_s']:.3f}), least free {record['least_free_gib']:.2f}{mark}"
    )


def cmd_matrix(args):
    from pathlib import Path

    torch, blockswap, model, device, results = setup(args)
    budget = results["free_start_gib"] - args.reserve
    sizes = [parse_size(text) for text in args.sizes.split(",")]
    lengths, swaps = numbers(args.lengths), numbers(args.swapped)
    started = time.perf_counter()
    models = model.load_models(Path(args.dit), Path(args.vae), device)
    torch.cuda.synchronize(device)
    results["load_s"] = round(time.perf_counter() - started, 2)
    results["loaded_gib"] = round(torch.cuda.memory_allocated(device) / GIB, 4)
    results["vae_sha256"] = vae_digest(torch, model, models)
    dit = models.runner.dit
    blocks = len(dit.blocks)
    log(f"loaded in {results['load_s']} s: {results['loaded_gib']} GiB on the device")
    references = {}

    def steps(config, swapped, slots, extra, forwards=args.forwards):
        for target in sizes:
            for length in lengths:
                key = (target, length)
                need = predicted(target, length, swapped, slots)
                record = {"config": config, "swapped": swapped, **extra}
                if need > budget:
                    record.update(
                        size=f"{target[1]}x{target[0]}", length=length, skipped=round(need, 2)
                    )
                    log(f"{config}: {target[1]}x{target[0]} L={length} skipped: {need:.2f} GiB"
                        f" predicted, {budget:.2f} to use")
                else:
                    run_step(torch, model, models, target, length, forwards, record)
                    if key in references:
                        count = len(record["digests"])
                        record["equal"] = record["digests"] == references[key][1][:count]
                        record["reference"] = references[key][0]
                    else:
                        references[key] = (config, record["digests"])
                    say(record)
                results["steps"].append(record)
                write(args, results)

    steps("loaded", 0, 0, {})
    # Every parameter of the DiT to a pinned host copy, from the device: the DiT is away after.
    backend = blockswap.Cuda(device)
    before, started = rss_gib(), time.perf_counter()
    parts = blockswap.host(dit, blockswap.dit_parts(blocks, 0, other=True), backend)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()  # the loaded parameters' memory back to the driver
    results["host_s"] = round(time.perf_counter() - started, 2)
    results["host_gib"] = round(sum(p.host.bytes.numel() for p in parts.values()) / GIB, 4)
    results["host_rss_gib"] = round(rss_gib() - before, 3)
    results["away_gib"] = round(torch.cuda.memory_allocated(device) / GIB, 4)
    log(f"hosted in {results['host_s']} s: {results['host_gib']} GiB pinned (RSS"
        f" +{results['host_rss_gib']}), {results['away_gib']} GiB left on the device")
    passes = [(s, prefetch, False) for s in swaps for prefetch in (False, True)]
    passes = [each for each in passes if each[0] or not each[1]]  # S = 0: nothing to prefetch
    passes = [(*each, round_) for round_ in range(args.rounds) for each in passes]
    if args.verify_slots:
        passes.append((max(swaps), True, True, 0))
    for swapped, prefetch, verify, round_ in passes:
        config = "restored" if swapped == 0 else f"S={swapped} {'prefetch' if prefetch else 'sync'}"
        config += " verify" if verify else ""
        placement = blockswap.Placement(
            dit, device, swapped, prefetch, parts=parts, verify=verify, backend=backend
        )
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        placement.enter("dit")
        torch.cuda.synchronize(device)
        extra = {
            "prefetch": prefetch,
            "round": round_,
            "enter_s": round(time.perf_counter() - started, 4),
            "device_bytes_gib": round(placement.device_bytes() / GIB, 4),
            "allocated_gib": round(torch.cuda.memory_allocated(device) / GIB, 4),
        }
        # A verified forward reads every slot back, 0.2 s a block: two are enough.
        steps(config, swapped, blockswap.SLOTS, extra, min(args.forwards, 2) if verify else args.forwards)
        uploads, verified = placement.uploads, placement.verified
        placement.enter("encode", 0)
        gone = round(torch.cuda.memory_allocated(device) / GIB, 4)
        placement.close()
        log(f"{config}: entered in {extra['enter_s']} s, {extra['allocated_gib']} GiB allocated"
            f" ({extra['device_bytes_gib']} the DiT's), {uploads} uploads, {verified} verified;"
            f" away: {gone} GiB")
        results.setdefault("placements", []).append(
            {"config": config, **extra, "uploads": uploads, "verified": verified, "away_gib": gone}
        )
        write(args, results)
    differing = [s for s in results["steps"] if s.get("equal") is False]
    compared = [s for s in results["steps"] if "equal" in s]
    results["verdict"] = f"{len(compared) - len(differing)} of {len(compared)} steps bit-equal"
    log(results["verdict"] + (": BITS DIFFER" if differing else ""))
    write(args, results)
    return 1 if differing else 0


def cmd_plan(args):
    from pathlib import Path

    from seedvr2x.runtime.plan import Plan

    torch, blockswap, model, device, results = setup(args)
    if args.no_trim:
        blockswap.Cuda.trim = lambda self: None  # what leaves stays the allocator's
    target, lengths = parse_size(args.size), numbers(args.lengths)
    reference = None
    if args.reference:
        with open(args.reference) as file:
            reference = json.load(file)
    resident = None if args.resident < 0 else args.resident
    plan = Plan(
        None,
        swapped=args.swapped,
        resident_encode=resident,
        resident_decode=resident,
        vae_in_dit=not args.vae_away,
    )
    results["plan"] = plan.record()["memory"]
    torch.cuda.reset_peak_memory_stats(device)
    before, started = rss_gib(), time.perf_counter()
    with FreeSampler(torch, device) as sampler:
        models = model.load_models(Path(args.dit), Path(args.vae), device, plan=plan)
        torch.cuda.synchronize(device)
    placement = models.placement
    pinned = (placement.host_bytes if placement else 0) + (
        models.vae.host.bytes.numel() if models.vae else 0
    )
    results["load"] = {
        "s": round(time.perf_counter() - started, 2),
        "device_peak_gib": round(torch.cuda.max_memory_allocated(device) / GIB, 4),
        "device_after_gib": round(torch.cuda.memory_allocated(device) / GIB, 4),
        "least_free_gib": round(sampler.least, 3),
        "pinned_gib": round(pinned / GIB, 4),
        "rss_gib": round(rss_gib() - before, 3),
        # The process's peak so far, the load's: the mapped file's pages count in it.
        "rss_peak_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3),
        "host_available_gib": round(available_gib(), 2),
    }
    log(f"load: {results['load']}")
    vae_gib = (models.vae.size if models.vae else 0) / GIB
    results["vae_part_gib"] = round(vae_gib, 4)
    results["phases"] = []
    failed = False
    for cycle in range(args.cycles):
        for phase in ("encode", "dit", "decode"):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            models.enter(phase)
            torch.cuda.synchronize(device)
            entry = {
                "cycle": cycle,
                "phase": phase,
                "enter_s": round(time.perf_counter() - started, 4),
                "allocated_gib": round(torch.cuda.memory_allocated(device) / GIB, 4),
                # What the driver has free once the phase is entered: memory the allocator
                # keeps isn't in it.
                "free_gib": round(torch.cuda.mem_get_info(device)[0] / GIB, 3),
            }
            if placement is not None:
                kept = None if phase == "dit" else resident
                entry["dit_bytes_gib"] = round(placement.device_bytes(kept) / GIB, 4)
            if phase == "dit":
                for length in lengths:
                    record = {"config": f"plan S={args.swapped}", "swapped": args.swapped}
                    run_step(torch, model, models, target, length, args.forwards, record)
                    if reference is not None:
                        same = [
                            step for step in reference["steps"]
                            if step.get("size") == record["size"] and step.get("length") == length
                            and "digests" in step
                        ]
                        if same:
                            count = min(len(record["digests"]), len(same[0]["digests"]))
                            record["equal"] = record["digests"][:count] == same[0]["digests"][:count]
                            record["reference"] = same[0]["config"]
                            failed |= not record["equal"]
                    say(record)
                    results["steps"].append(record)
            else:
                entry["vae_sha256"] = vae_digest(torch, model, models)
                if reference is not None:
                    entry["vae_equal"] = entry["vae_sha256"] == reference["vae_sha256"]
                    failed |= not entry["vae_equal"]
                torch.cuda.synchronize(device)
                entry["allocated_after_gib"] = round(torch.cuda.memory_allocated(device) / GIB, 4)
            log(f"cycle {cycle} {phase}: {entry}")
            results["phases"].append(entry)
            write(args, results)
    if placement is not None:
        results["uploads"] = placement.uploads
    results["verdict"] = "BITS DIFFER" if failed else "bits equal where compared"
    log(results["verdict"])
    write(args, results)
    return 1 if failed else 0


def cmd_table(args):
    for path in args.results:
        with open(path) as file:
            results = json.load(file)
        print(f"\n### {path}\n")
        print(f"{results['gpu']}, {results['free_start_gib']} GiB free of {results['total_gib']}"
              f" at the start, attention {results['attention']}, cap {results['cap_gib']}:"
              f" {results.get('verdict')}\n")
        print("| Config | Size | L | Rounds | Peak GiB | Wall s: least, median (max) | Least free | Bits |")
        print("|---|---|---|---|---|---|---|---|")
        merged = {}
        for step in results["steps"]:
            merged.setdefault((step["config"], step["size"], step["length"]), []).append(step)
        for (config, size, length), steps in merged.items():
            if "skipped" in steps[0]:
                print(f"| {config} | {size} | {length} | | skipped: {steps[0]['skipped']} predicted"
                      " | | | |")
                continue
            equal = [step.get("equal") for step in steps]
            bits = "DIFFER" if False in equal else ("reference" if None in equal else "equal")
            print(f"| {config} | {size} | {length} | {len(steps)}"
                  f" | {max(step['peak_gib'] for step in steps):.3f}"
                  f" | {min(step['wall_min_s'] for step in steps):.3f},"
                  f" {statistics.median(step['wall_s'] for step in steps):.3f}"
                  f" ({max(step['wall_max_s'] for step in steps):.3f})"
                  f" | {min(step['least_free_gib'] for step in steps):.2f} | {bits} |")
        for placement in results.get("placements", []):
            print(f"- {placement['config']}: entered in {placement['enter_s']} s,"
                  f" {placement['allocated_gib']} GiB allocated ({placement['device_bytes_gib']}"
                  f" the DiT's), {placement['uploads']} uploads, away {placement['away_gib']} GiB")
        for phase in results.get("phases", []):
            print(f"- {phase}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("matrix", "plan"):
        sub = commands.add_parser(name)
        sub.add_argument("--dit", required=True)
        sub.add_argument("--vae", required=True)
        sub.add_argument("--lengths", default="1")
        sub.add_argument("--forwards", type=int, default=3)
        sub.add_argument("--out")
    matrix = commands.choices["matrix"]
    matrix.add_argument("--sizes", default="1080")
    matrix.add_argument("--swapped", default="0,3,12,36")
    matrix.add_argument("--reserve", type=float, default=1.5, help="GiB left free on the card")
    matrix.add_argument("--verify-slots", action="store_true")
    matrix.add_argument("--rounds", type=int, default=1)
    plan = commands.choices["plan"]
    plan.add_argument("--size", default="1080")
    plan.add_argument("--swapped", type=int, default=36)
    plan.add_argument("--resident", type=int, default=0, help="-1: the DiT stays (None)")
    plan.add_argument("--vae-away", action="store_true")
    plan.add_argument("--cycles", type=int, default=2)
    plan.add_argument("--no-trim", action="store_true", help="for the comparison only")
    plan.add_argument("--reference")
    table = commands.add_parser("table")
    table.add_argument("results", nargs="+")
    args = parser.parse_args()
    return {"matrix": cmd_matrix, "plan": cmd_plan, "table": cmd_table}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

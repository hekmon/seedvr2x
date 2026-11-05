"""Where a VAE pass's time goes: torch's profiler over seedvr2x's tiled decode or encode.

  vae_profile.py --dit DIT --vae VAE --out DIR [--size 2160] [--latents 1] [--tile 1024]
      [--pass decode|encode] [--frames 1]

The models load as seedvr2x loads them (model.load_models, no plan), the DiT then dropped from
the GPU, as a VAE phase with the DiT off it has them. A pass runs as the runtime runs it
(tiling.decode_tiled of a seeded latent of `--latents` latent frames, as hold_probe.py's dF/T
steps; or tiling.encode_tiled of `--frames` seeded frames, its eN/T), the scratch in system
memory: once to warm up, once timed between two synchronisations, once under torch.profiler
(CPU and CUDA). DIR/profile.txt holds the operators by their CUDA time, DIR/summary.json the wall
time, the CUDA time the profiler counts, the peak allocated, and the share of the wall the GPU
was busy (from the kernels' own intervals): a pass whose GPU idles waits on the host.

Written on 2026-11-05 for place-study (Q1: a 4K picture's decode took 3.0 to 3.6 s in
seedvr2x, 1.24 s in ComfyUI v0.44.0's tiled decode at 512 px, on the same card)."""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(prog="vae_profile.py")
    parser.add_argument("--dit", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", default="2160")
    parser.add_argument("--latents", type=int, default=1)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--pass", dest="which", choices=("decode", "encode"), default="decode")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from seedvr2x import memory

    memory.startup()  # the allocator a held run takes, set before torch is imported
    import gc

    import numpy
    import torch

    from scratch import Scratch  # the CLI's store, as hold_probe.py hands it
    from seedvr2x import model, shot_pass, tiling
    from seedvr2x.frames import padded_length, padded_size
    from seedvr2x.passes import Tiles

    device = torch.device("cuda:0")
    models = model.load_models(Path(args.dit), Path(args.vae), device)
    models.runner.dit.to("cpu")
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    size = int(args.size)
    target = (size, round(size * 16 / 9))
    height, width = padded_size(target)

    def run() -> None:
        scratch = Scratch(None, 2**40)  # always in system memory
        with torch.no_grad():
            if args.which == "decode":
                generator = torch.Generator().manual_seed(11)
                values = torch.randn(
                    (16, args.latents, height // 8, width // 8), generator=generator
                )
                latent = values.to(model.COMPUTE_DTYPE).permute(1, 2, 3, 0)
                for piece in tiling.decode_tiled(models, latent, Tiles(args.tile), scratch):
                    del piece
            else:
                rng = numpy.random.default_rng(13)
                source = rng.random((args.frames, target[0], target[1], 3), numpy.float32)
                position = [0]

                def read(count: int) -> numpy.ndarray:
                    position[0] += count
                    return source[position[0] - count : position[0]]

                inputs = shot_pass.encoder_inputs(models, read, args.frames, target)
                tiling.encode_tiled(
                    models, inputs, padded_length(args.frames), Tiles(args.tile), scratch
                )
        torch.cuda.synchronize(device)

    run()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    run()
    wall = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as profiler:
        run()
    table = profiler.key_averages().table(sort_by="cuda_time_total", row_limit=30)
    (Path(args.out) / "profile.txt").write_text(table)
    kernels = sorted(
        (e.time_range.start, e.time_range.end)
        for e in profiler.events()
        if e.device_type == torch.autograd.DeviceType.CUDA
    )
    busy, last = 0, None
    for start, end in kernels:
        if last is None or start > last:
            busy += end - start
            last = end
        elif end > last:
            busy += end - last
            last = end
    span = (kernels[-1][1] - kernels[0][0]) if kernels else 0
    summary = {
        "pass": args.which,
        "size": [height, width],
        "latents": args.latents,
        "frames": args.frames,
        "tile": args.tile,
        "wall_s": wall,
        "peak_gib": peak,
        "kernels_busy_s": busy / 1e6,
        "kernels_span_s": span / 1e6,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (Path(args.out) / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())

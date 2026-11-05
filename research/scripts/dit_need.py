"""What one DiT window holds at its peak, by the source line that allocated it.

  dit_need.py --dit DIT --vae VAE --out DIR [--size 2160] [--latents 1] [--placement 36]
      [--without-vae] [--mlp-chunk TOKENS] [--lean] [--sdpa-batched] [--lean-chunk TOKENS]
      [--gelu-inplace] [--reference FILE]

The models load as seedvr2x loads them (model.load_models), the DiT's last `--placement` blocks
swapped through seedvr2x's BlockSwap (blockswap.Placement, as hold_probe.py places them), the
VAE's weights off the GPU with --without-vae (a picture's plan on a small card has them so: "one
latent, with every block of the DiT swapped and the VAE off the GPU"). One window of `--latents`
latents at `--size` runs three times through placement_bench.one_forward (the same latent and
seeds as hold_probe.py's windows): a warm-up, one with torch's allocation history recorded
(torch.cuda.memory._record_memory_history, Python stacks), one plain for its time. The history is
replayed from what torch held when it started to find the allocated peak, and what is live there
is grouped by the innermost frame in seedvr2x's code (the vendored model's included): DIR/peak.json
and a table on stdout. The snapshot itself is kept (DIR/snapshot.pickle, readable by
torch's memory_viz).

--mlp-chunk N runs the 7B's feed-forward (vendor/models/dit_7b/mlp.py, MLP and SwiGLUMLP) on N
tokens at a time, its rows concatenated after: the memory a chunked feed-forward saves and
whether the output keeps its bits (the digests against --reference, a JSON this script writes when
absent: the unchunked run's).

--gelu-inplace runs the 7B's MLP with its GELU in place (torch's gelu_, the same kernel).

Written on 2026-11-05 for place-study (Q5: what the 5.64 GiB a 4K picture's DiT step needs on a
small card is made of)."""

import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(prog="dit_need.py")
    parser.add_argument("--dit", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", default="2160")
    parser.add_argument("--latents", type=int, default=1)
    parser.add_argument("--placement", type=int, default=36)
    parser.add_argument("--without-vae", action="store_true")
    parser.add_argument("--mlp-chunk", type=int, default=0)
    parser.add_argument("--reference", default="")
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument(
        "--gelu-inplace", action="store_true", help="the 7B MLP's GELU on proj_in's output"
    )
    parser.add_argument("--lean", action="store_true", help="dit_lean.py's attention")
    parser.add_argument("--sdpa-batched", action="store_true", help="dit_lean.py's batched SDPA")
    parser.add_argument("--lean-chunk", type=int, default=0, help="dit_lean.py's chunk")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from seedvr2x import memory

    memory.startup()  # the allocator a held run takes, set before torch is imported
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import placement_bench as bench
    import torch

    from seedvr2x import blockswap, model
    from seedvr2x.frames import padded_size

    device = torch.device("cuda:0")
    gib = 1024**3
    if args.lean or args.sdpa_batched:
        import dit_lean

        dit_lean.install(lean=args.lean, sdpa_batched=args.sdpa_batched, chunk=args.lean_chunk)
    if args.gelu_inplace:
        from seedvr2x.vendor.models.dit_7b import mlp as plain

        def inplace(self: Any, x: torch.Tensor) -> torch.Tensor:
            # MLP.forward with its GELU (tanh) in place on proj_in's output, which nothing
            # else reads: the same kernel's values, one tensor of 12,288 values a token fewer.
            x = self.proj_in(x)
            torch._C._nn.gelu_(x, approximate="tanh")
            return self.proj_out(x)

        plain.MLP.forward = inplace
    if args.mlp_chunk:
        from seedvr2x.vendor.models.dit_7b import mlp

        chunk = args.mlp_chunk

        def chunked_forward(forward: Any) -> Any:
            def chunked(self: object, x: torch.Tensor) -> torch.Tensor:
                if x.shape[0] <= chunk:
                    return forward(self, x)
                out = torch.cat([forward(self, part) for part in x.split(chunk, dim=0)], dim=0)
                return out

            return chunked

        # Both feed-forwards of the 7B's file (the 7B's blocks take the GELU MLP, mlp_type
        # "normal" in vendor/configs_7b/main.yaml; the SwiGLU one is the 3B's).
        mlp.MLP.forward = chunked_forward(mlp.MLP.forward)
        mlp.SwiGLUMLP.forward = chunked_forward(mlp.SwiGLUMLP.forward)
    models = model.load_models(Path(args.dit), Path(args.vae), device)
    dit = models.runner.dit
    if args.placement:
        blocks = len(dit.blocks)
        backend = blockswap.Cuda(device)
        fixed = blockswap.dit_parts(blocks, blocks - args.placement, False)
        parts = blockswap.host(dit, fixed, backend)
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        placement = blockswap.Placement(dit, device, args.placement, parts=parts, backend=backend)
        placement.enter("dit")
    if args.without_vae:
        models.runner.vae.to("cpu")
    import gc

    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    size = int(args.size) if args.size.isdigit() else None
    target = (size, round(size * 16 / 9)) if size else tuple(map(int, args.size.split("x")))[::-1]
    height, width = padded_size(target)  # type: ignore[arg-type]
    latent = bench.make_latent(torch, model.COMPUTE_DTYPE, args.latents, height // 8, width // 8)
    resident = torch.cuda.memory_allocated(device)
    said = {
        "size": [height, width],
        "latents": args.latents,
        "placement": args.placement,
        "without_vae": args.without_vae,
        "mlp_chunk": args.mlp_chunk,
        "lean": args.lean,
        "sdpa_batched": args.sdpa_batched,
        "lean_chunk": args.lean_chunk,
        "gelu_inplace": args.gelu_inplace,
        "attention": model.attention_backend(),
        "resident_gib": resident / gib,
    }
    runs = [bench.one_forward(torch, model, models, latent, 42)]
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    if not args.no_history:
        torch.cuda.memory._record_memory_history(max_entries=2_000_000, stacks="python")
    runs.append(bench.one_forward(torch, model, models, latent, 42))
    if not args.no_history:
        snapshot = torch.cuda.memory._snapshot()
        torch.cuda.memory._record_memory_history(enabled=None)
        with open(Path(args.out) / "snapshot.pickle", "wb") as handle:
            pickle.dump(snapshot, handle)
        said["live_at_peak"] = _live_at_peak(snapshot, baseline)
    runs.append(bench.one_forward(torch, model, models, latent, 43))
    said["runs"] = runs
    said["baseline_gib"] = baseline / gib
    reference = Path(args.reference) if args.reference else None
    if reference is not None:
        key = f"{height}x{width}:{args.latents}"
        known = json.loads(reference.read_text()) if reference.exists() else {}
        digests = [runs[1]["sha256"], runs[2]["sha256"]]
        if key in known:
            said["equal"] = known[key] == digests
        elif not (args.mlp_chunk or args.lean or args.sdpa_batched or args.gelu_inplace):
            known[key] = digests
            reference.write_text(json.dumps(known, indent=1))
            said["equal"] = "reference written"
    (Path(args.out) / "peak.json").write_text(json.dumps(said, indent=1))
    print(json.dumps({k: v for k, v in said.items() if k != "live_at_peak"}, indent=1))
    live = said.get("live_at_peak")
    if live:
        print(f"peak {live['peak_gib']:.3f} GiB allocated, {live['over_baseline_gib']:.3f} over")
        for row in live["by_line"][:25]:
            print(f"  {row['gib']:7.3f} GiB  {row['count']:4d}  {row['where']}")
    return 0


def _live_at_peak(snapshot: dict, baseline: int) -> dict:
    """The allocations live at the allocated peak of the recorded history, by their innermost
    frame in seedvr2x's code: replayed from `baseline` bytes, the history's own events (alloc,
    free_completed) in order."""
    events = [e for trace in snapshot["device_traces"] for e in trace]
    current, peak, at = baseline, baseline, -1
    for index, event in enumerate(events):
        if event["action"] == "alloc":
            current += event["size"]
        elif event["action"] == "free_completed":
            current -= event["size"]
        if current > peak:
            peak, at = current, index
    live: dict[int, dict] = {}
    for event in events[: at + 1]:
        if event["action"] == "alloc":
            live[event["addr"]] = event
        elif event["action"] == "free_completed":
            live.pop(event["addr"], None)
    groups: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for event in live.values():
        where = "?"
        for frame in event.get("frames", []):
            name = frame.get("filename", "")
            if "seedvr2x" in name and "site-packages" not in name:
                where = f"{name.split('seedvr2x/')[-1]}:{frame.get('line')} {frame.get('name')}"
                break
        groups[where][0] += event["size"]
        groups[where][1] += 1
    rows = sorted(
        ({"where": k, "gib": v[0] / 1024**3, "count": v[1]} for k, v in groups.items()),
        key=lambda row: -row["gib"],
    )
    return {
        "peak_gib": peak / 1024**3,
        "over_baseline_gib": (peak - baseline) / 1024**3,
        "event": at,
        "by_line": rows,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


if __name__ == "__main__":
    sys.exit(main())

"""One picture's time, phase by phase, through seedvr2x's own command, models loaded once.

  picture_phases.py --out FILE.jsonl [--reps 3] [--png-variants] [--lean] [--sdpa-batched] \
      [--then E,D [--then-reps 2] [--then-out FILE.png]] -- CLI ARGS...

Run under small_card.py (`small_card.py run --keep 4 -- picture_phases.py ...`), so that the
command's plan is made in the card's free memory less the desktop's share. The command runs as a
user's would (its plan, its load, its record), with the library's steps wrapped in this process:
each step is timed between two CUDA synchronisations, so that a step's time is the work the step
queued, and its allocated and reserved peaks are torch's (reset at the step's start; leaf steps
alone, the totals' would be their last leaf's). still.upscale runs `--reps` times on the models
the command loaded (the first is the command's own: the models just loaded, as a user's run has
them; the others are a server's second picture), each output's SHA-256 recorded so that the reps
are shown to give the same bits. The PNG writer is timed as the command calls it, and with
`--png-variants` the same picture is also written with each filter, zlib levels 1, 3 and 6, and
at 8 bits, each timed and its size kept (png.write's own FILTER and LEVEL: measured on a 32-core
machine, png.py; this one has 32 threads, one used by zlib).

Phases (each a line of the output): read (picture.read), enter:<phase> (Models.enter, the
models moved for the phase), resize (the picture to the device and through the encoder's
transform), encode, noise (sampling.drawn), dit (sampling.window), encoded/sampled/decoded (the
three steps whole: sampled holds the DiT's output's copy to system memory, decoded the crop and
the finite check), decode (each slice of the decode's generator), reference (the picture through
the transform in float32, for the colour correction), colour (colour.split), upscale (the three
steps whole), png:<variant>.

`--then E,D` runs `--then-reps` more pictures after the reps on the same models with the
encode's and the decode's tiles E and D (0: untiled), outside the plan, held by torch's limit
alone, the first written as a 16-bit PNG (`--then-out`, png.write, no colour chunks): the
untiled 4K decode, which the plan refuses (no one-frame untiled decode measured past 1080p).

Written on 2026-11-05 for place-study (the 4K picture's 14 s against ComfyUI's 5 s); the steps it
wraps are those of seedvr2x/still.py at that date."""

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable, Generator, Iterator
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(prog="picture_phases.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--png-variants", action="store_true")
    parser.add_argument("--lean", action="store_true", help="dit_lean.py's attention")
    parser.add_argument("--sdpa-batched", action="store_true", help="dit_lean.py's batched SDPA")
    parser.add_argument("--lean-chunk", type=int, default=0, help="dit_lean.py's chunk")
    parser.add_argument("--then", default="", help="ENCODE,DECODE tiles (0: untiled) for more reps")
    parser.add_argument("--then-reps", type=int, default=2)
    parser.add_argument("--then-out", default="", help="the first of those reps as a 16-bit PNG")
    parser.add_argument("cli", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cli = args.cli[1:] if args.cli[:1] == ["--"] else args.cli

    import main as command  # the seedvr2x command's module; imports no torch at its top
    from seedvr2x import memory

    out = open(args.out, "a")
    state: dict[str, Any] = {"rep": 0, "depth": 0}

    def emit(record: dict[str, Any]) -> None:
        record["rep"] = state["rep"]
        out.write(json.dumps(record) + "\n")
        out.flush()

    def gpu() -> Any:
        import torch

        return torch

    def timed(name: str | Callable[..., str], function: Callable[..., Any], leaf: bool) -> Any:
        def wrapper(*a: Any, **k: Any) -> Any:
            # torch only once the command has imported it: picture.read runs before the
            # allocator is chosen, and an import here would come before it.
            torch = sys.modules.get("torch")
            label = name(*a, **k) if callable(name) else name
            cuda = torch is not None and torch.cuda.is_initialized()
            if cuda:
                torch.cuda.synchronize()
                if leaf:
                    torch.cuda.reset_peak_memory_stats()
                before = torch.cuda.memory_allocated()
            started = time.perf_counter()
            result = function(*a, **k)
            if cuda:
                torch.cuda.synchronize()  # type: ignore[union-attr]
            record = {"phase": label, "s": time.perf_counter() - started}
            if cuda and leaf:
                record["before_gib"] = before / 2**30
                record["alloc_peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
                record["reserved_peak_gib"] = torch.cuda.max_memory_reserved() / 2**30
            emit(record)
            return result

        return wrapper

    def timed_slices(name: str, function: Callable[..., Iterator[Any]]) -> Any:
        def wrapper(*a: Any, **k: Any) -> Generator[Any]:
            torch = gpu()
            slices = function(*a, **k)
            try:
                while True:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    try:
                        piece = next(slices)
                    except StopIteration:
                        return
                    torch.cuda.synchronize()
                    emit(
                        {
                            "phase": name,
                            "s": time.perf_counter() - started,
                            "alloc_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                            "reserved_peak_gib": torch.cuda.max_memory_reserved() / 2**30,
                        }
                    )
                    yield piece
            finally:
                if isinstance(slices, Generator):
                    slices.close()

        return wrapper

    # picture and png import no torch: wrapped now. The rest once memory.startup has chosen the
    # allocator, which must come before torch is imported (seedvr2x/memory.py).
    from seedvr2x import picture, png

    picture.read = timed("read", picture.read, leaf=False)
    startup = memory.startup

    def started_up() -> str:
        said = startup()
        torch_steps()
        return said

    memory.startup = started_up

    def torch_steps() -> None:
        # Wrapped where still.py looks them up at call time (module attributes, its globals).
        from seedvr2x import colour, model, sampling, still, tiling

        if args.lean or args.sdpa_batched:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import dit_lean

            dit_lean.install(lean=args.lean, sdpa_batched=args.sdpa_batched, chunk=args.lean_chunk)

        model.Models.enter = timed(
            lambda self, phase, *a, **k: f"enter:{phase}", model.Models.enter, leaf=True
        )
        still._transformed = timed(
            lambda models, frame, target, dtype: (
                "resize" if dtype != gpu().float32 else "reference"
            ),
            still._transformed,
            leaf=True,
        )
        model.encode_stream = timed("encode", model.encode_stream, leaf=True)
        tiling.encode_tiled = timed("encode", tiling.encode_tiled, leaf=True)
        sampling.drawn = timed("noise", sampling.drawn, leaf=True)
        sampling.window = timed("dit", sampling.window, leaf=True)
        model.decode_stream = timed_slices("decode", model.decode_stream)
        tiling.decode_tiled = timed_slices("decode", tiling.decode_tiled)
        colour.split = timed("colour", colour.split, leaf=True)
        still.encoded = timed("encoded", still.encoded, leaf=False)
        still.sampled = timed("sampled", still.sampled, leaf=False)
        still.decoded = timed("decoded", still.decoded, leaf=False)
        upscale = timed("upscale", still.upscale, leaf=False)

        def repeated(*a: Any, **k: Any) -> Any:
            result = None
            for rep in range(1, args.reps + 1):
                state["rep"] = rep
                result = upscale(*a, **k)
                emit({"phase": "sha256", "value": hashlib.sha256(result.tobytes()).hexdigest()})
            if args.then:
                # The same picture on the same models with other tiles, outside the plan: an
                # untiled pass the planner takes no figure for, held by torch's limit alone.
                from seedvr2x.passes import Tiles

                sides = [int(side) for side in args.then.split(",")]
                other = dict(k)
                other["encode_tiles"] = Tiles(sides[0]) if sides[0] else None
                other["decode_tiles"] = Tiles(sides[1]) if sides[1] else None
                emit({"phase": "then", "encode": sides[0], "decode": sides[1]})
                for rep in range(args.reps + 1, args.reps + 1 + args.then_reps):
                    state["rep"] = rep
                    again = upscale(*a, **other)
                    digest = hashlib.sha256(again.tobytes()).hexdigest()
                    emit({"phase": "sha256", "value": digest, "then": True})
                    if args.then_out and rep == args.reps + 1:
                        data = write(again, bits=16, grey=False, icc=None, chunks=[])  # untimed
                        with open(args.then_out, "wb") as handle:
                            handle.write(data)
            state["rep"] = 0
            return result

        still.upscale = repeated

    write = png.write

    def written(rgb: Any, **k: Any) -> bytes:
        started = time.perf_counter()
        data = write(rgb, **k)
        emit(
            {
                "phase": f"png:command:{k.get('bits')}",
                "s": time.perf_counter() - started,
                "bytes": len(data),
            }
        )
        if args.png_variants:
            variants: list[tuple[str, dict[str, Any]]] = [
                (f"png:filter{f}:level6:16", {"bits": 16, "filter_type": f}) for f in range(6)
            ]
            variants += [(f"png:adaptive:level{v}:16", {"bits": 16, "level": v}) for v in (1, 3)]
            variants += [
                ("png:adaptive:level6:8", {"bits": 8}),
                ("png:adaptive:level1:8", {"bits": 8, "level": 1}),
                ("png:filter0:level1:16", {"bits": 16, "filter_type": 0, "level": 1}),
            ]
            for label, extra in variants:
                started = time.perf_counter()
                size = len(write(rgb, **{**k, **extra}))
                emit({"phase": label, "s": time.perf_counter() - started, "bytes": size})
        return data

    png.write = written
    emit({"phase": "start", "argv": cli, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
    status = command.main(cli)
    emit({"phase": "end", "status": status, "allocator": memory.__name__})
    return status


if __name__ == "__main__":
    sys.exit(main())

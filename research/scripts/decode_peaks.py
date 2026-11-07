"""The decode's allocated peak in bytes, part by part, through seedvr2x's own command.

  decode_peaks.py --out FILE.jsonl [--sides 512,576,...,0] [--reps 2] -- CLI ARGS...

Run under small_card.py (`small_card.py run --keep 4 -- decode_peaks.py ...`), so that the
command's plan is made in the card's free memory less the desktop's share. The command runs as a
user's would, its plan, load and steps unchanged; this process only reads torch's allocated
counters, which move on the host as tensors are made and freed, so no synchronisation is added to
a video's units and no step is moved; a picture's encode, DiT step and decodes are each waited
for at their start and end (their times are not the command's).

Each record holds what was allocated before the measured step (`before`, bytes), its allocated
peak (`peak`, bytes, torch's max_memory_allocated reset at the step's start), and its marks: at
each of the step's parts, [label, allocated then, the step's peak so far], so that the part
during which the peak rose is the one that set it. Parts marked: every tile of a tiled encode
or decode (tiling.TileStep, told the tiles done), every slice a decode hands on, the start and
end of the colour correction (colour.split), the end of a picture's transform (still._transformed:
`resize` for the encoder's input, `reference` for the correction's).

- A video's units: from run._started (the reset) to run._spent, `before` read just after the
  reset, as the command logs a unit's peak; the record names the unit by its last mark's kind.
- A picture's decode (still.decoded): the command's own call, `before` read before it, in the
  DiT's phase (still.decoded enters the decode's: `entered` is what was allocated once it had);
  with `--sides`, the same latent, picture and target decoded again on the models the command
  loaded, at each tile side (0: untiled), `--reps` times each, outside
  the plan (torch's limit alone holds them), the phase entered before the reset so that `before`
  is the models in the decode's phase; each output's SHA-256 recorded. A side that fails (out of
  memory, or refused by the library's own check) is recorded with its error, and the sides after
  it still run.
- `--census`: at each phase's entry (Models.enter), the CUDA tensors alive, each storage once,
  their bytes against torch's allocated figure, the RoPE tables (rope.py's get_axial_freqs) apart,
  and the largest storages: what a phase holds before its first step.

Written on 2026-11-07 for impl-place-2: the 4K picture's one-frame decode at every tile side with
its copies (one ran 0.17 GiB over its prediction at 832 px on an emulated 8 GB card, 2026-11-06),
and a 1080p clip's decode in 768-px tiles (1.1 MiB over); the steps it wraps are those of
seedvr2x/still.py, shot_pass.py, tiling.py and the command's run.py at that date."""

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable, Generator, Iterator
from typing import Any


ROPE_VALUES = 60  # a RoPE table's values a position, the 7B's (vendor/models/dit_7b/rope.py)


def main() -> int:
    parser = argparse.ArgumentParser(prog="decode_peaks.py")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--sides", default="", help="a picture's decode again at each side (0: untiled)"
    )
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument(
        "--census", action="store_true", help="the CUDA tensors alive at each phase's entry"
    )
    parser.add_argument("cli", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cli = args.cli[1:] if args.cli[:1] == ["--"] else args.cli
    sides = [int(side) for side in args.sides.split(",") if side]

    import main as command  # the seedvr2x command's module; imports no torch at its top
    from seedvr2x import memory

    out = open(args.out, "a")
    state: dict[str, Any] = {"marks": None, "unit": None, "picture_done": False}

    def emit(record: dict[str, Any]) -> None:
        out.write(json.dumps(record) + "\n")
        out.flush()

    def torch() -> Any:
        return sys.modules["torch"]

    def mark(label: str) -> None:
        if state["marks"] is None:
            return
        cuda = torch().cuda
        state["marks"].append([label, cuda.memory_allocated(), cuda.max_memory_allocated()])

    def stepped(step: Callable[[int, int], None] | None, kind: str) -> Callable[[int, int], None]:
        def told(done: int, total: int) -> None:
            mark(f"{kind} tile {done}/{total}")
            if step is not None:
                step(done, total)

        return told

    def marked_slices(slices: Iterator[Any], kind: str) -> Generator[Any]:
        count = 0
        try:
            for piece in slices:
                count += 1
                mark(f"{kind} slice {count}")
                yield piece
        finally:
            if isinstance(slices, Generator):
                slices.close()

    def census(phase: str) -> None:
        # The CUDA tensors alive, each storage once: its bytes, and the shape and dtype of a tensor
        # viewing it; what torch has allocated beyond them is not a tensor's (a library's
        # workspace, cuBLAS's). The RoPE tables (rope.py's get_axial_freqs, rotary_embedding_torch's
        # axial frequencies: (frames, rows, columns, 60) float16 for the 7B) summed apart.
        import gc

        cuda = torch().cuda
        storages: dict[int, tuple[int, list[int], str]] = {}
        for each in gc.get_objects():
            try:
                if not isinstance(each, torch().Tensor) or not each.is_cuda:
                    continue
                storage = each.untyped_storage()
            except Exception:
                continue
            pointer = storage.data_ptr()
            if pointer not in storages or storage.nbytes() > storages[pointer][0]:
                storages[pointer] = (storage.nbytes(), list(each.shape), str(each.dtype))
        rope = [
            entry
            for entry in storages.values()
            if len(entry[1]) == 4 and entry[1][3] == ROPE_VALUES and entry[2] == "torch.float16"
        ]
        emit(
            {
                "step": "census",
                "phase": phase,
                "allocated": cuda.memory_allocated(),
                "tensors": sum(entry[0] for entry in storages.values()),
                "count": len(storages),
                "rope": sum(entry[0] for entry in rope),
                "rope_shapes": sorted(entry[1][:3] for entry in rope),
                "largest": sorted(storages.values(), reverse=True)[:12],
                "storages": sorted(storages.values(), reverse=True),
            }
        )

    startup = memory.startup

    def started_up() -> str:
        said = startup()
        hooks()
        return said

    memory.startup = started_up

    def hooks() -> None:
        # Wrapped where the library looks them up at call time (module attributes).
        from seedvr2x import colour, model, still, tiling

        decode_tiled, encode_tiled, decode_stream = (
            tiling.decode_tiled,
            tiling.encode_tiled,
            model.decode_stream,
        )

        def tiled_decode(
            models: Any, latent: Any, tiles: Any, scratch: Any, step: Any = None
        ) -> Any:
            return marked_slices(
                decode_tiled(models, latent, tiles, scratch, stepped(step, "decode")), "decode"
            )

        def tiled_encode(
            models: Any, slices: Any, frames: Any, tiles: Any, scratch: Any, step: Any = None
        ) -> Any:
            return encode_tiled(models, slices, frames, tiles, scratch, stepped(step, "encode"))

        def streamed_decode(*a: Any, **k: Any) -> Any:
            return marked_slices(decode_stream(*a, **k), "decode")

        tiling.decode_tiled = tiled_decode
        tiling.encode_tiled = tiled_encode
        model.decode_stream = streamed_decode

        split = colour.split

        def marked_split(*a: Any, **k: Any) -> Any:
            mark("split start")
            result = split(*a, **k)
            mark("split end")
            return result

        colour.split = marked_split

        transformed = still._transformed

        def marked_transform(models: Any, frame: Any, target: Any, dtype: Any) -> Any:
            result = transformed(models, frame, target, dtype)
            mark("reference" if dtype == torch().float32 else "resize")
            return result

        still._transformed = marked_transform

        enter = model.Models.enter

        def marked_enter(self: Any, phase: str, *a: Any, **k: Any) -> Any:
            result = enter(self, phase, *a, **k)
            if state["marks"] is not None:
                state["entered"] = torch().cuda.memory_allocated()
            mark(f"entered {phase}")
            if args.census:
                census(phase)
            return result

        model.Models.enter = marked_enter

        import planner

        planned = planner.plan

        def plan_said(*a: Any, **k: Any) -> Any:
            made, predicted = planned(*a, **k)
            emit(
                {
                    "step": "plan",
                    "encode": predicted.encode,
                    "dit": predicted.dit,
                    "decode": predicted.decode,
                    "window": predicted.window,
                    "budget": predicted.budget,
                    "reserves": dict(predicted.reserves),
                    "swapped": made.swapped,
                    "resident_encode": made.resident_encode,
                    "resident_decode": made.resident_decode,
                    "encode_tiles": None if made.encode_tiles is None else made.encode_tiles.size,
                    "decode_tiles": None if made.decode_tiles is None else made.decode_tiles.size,
                    "tail_frames": made.tail_frames,
                    "vae_in_dit": made.vae_in_dit,
                    "attention": predicted.attention,
                    "dit_model": predicted.dit_model.name,
                }
            )
            return made, predicted

        planner.plan = plan_said

        from seedvr2x import still_plan

        picture_peaks = still_plan.picture_peaks

        def picture_peaks_said(*a: Any, **k: Any) -> Any:
            found = picture_peaks(*a, **k)
            emit(
                {
                    "step": "picture plan",
                    **{phase: [peak.allocated, peak.models, peak.room] for phase, peak in found.items()},
                }
            )
            return found

        still_plan.picture_peaks = picture_peaks_said
        still_planned = still_plan.plan

        def still_plan_said(*a: Any, **k: Any) -> Any:
            made = still_planned(*a, **k)
            if made.picture is not None:  # the command's own picture: its phases as predicted
                picture_peaks_said(made, made.picture, made.input_pixels)
            return made

        still_plan.plan = still_plan_said

        for name in ("encoded", "sampled"):
            inner = getattr(still, name)

            def picture_step(*a: Any, inner: Any = inner, name: str = name, **k: Any) -> Any:
                cuda = torch().cuda
                cuda.synchronize()
                before = cuda.memory_allocated()
                cuda.reset_peak_memory_stats()
                result = inner(*a, **k)
                cuda.synchronize()
                emit({"step": f"picture {name}", "before": before, "peak": cuda.max_memory_allocated()})
                return result

            setattr(still, name, picture_step)

        decoded = still.decoded

        def measured(label: Any, models: Any, run: Callable[[], Any], facts: dict[str, Any]) -> Any:
            cuda = torch().cuda
            state["marks"], state["entered"] = [], None
            before = cuda.memory_allocated()
            cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            record: dict[str, Any] = {"step": "picture decode", "side": label, **facts}
            try:
                result = run()
            except Exception as error:  # recorded, and the next side runs
                said = (str(error).splitlines() or [""])[0]
                record |= {"error": type(error).__name__, "said": said}
                result = None
            finally:
                marks, state["marks"] = state["marks"], None
            cuda.synchronize()
            record |= {
                "before": before,
                "entered": state["entered"],
                "peak": cuda.max_memory_allocated(),
                "s": time.perf_counter() - started,
                "marks": marks,
            }
            if result is not None:
                record["sha256"] = hashlib.sha256(result.tobytes()).hexdigest()
            emit(record)
            return result

        def picture_decoded(
            models: Any,
            latent: Any,
            pixels: Any,
            target: Any,
            *,
            tiles: Any,
            correct: bool,
            tile: Any = None,
        ) -> Any:
            facts = {
                "target": list(target),
                "stored": list(pixels.shape),
                "latent": list(latent.shape),
                "correct": correct,
            }
            own = measured(
                f"command:{tiles.size if tiles is not None else 0}",
                models,
                lambda: decoded(
                    models, latent, pixels, target, tiles=tiles, correct=correct, tile=tile
                ),
                facts,
            )
            if own is None:
                raise RuntimeError("the command's own decode failed")
            if not state["picture_done"]:
                state["picture_done"] = True
                from seedvr2x.passes import Tiles

                for side in sides:
                    chosen = Tiles(side) if side else None
                    for _ in range(args.reps):
                        models.enter("decode")
                        measured(
                            side,
                            models,
                            lambda chosen=chosen: decoded(
                                models, latent, pixels, target, tiles=chosen, correct=correct
                            ),
                            facts,
                        )
            return own

        still.decoded = picture_decoded

        # A video's units, as the command measures them (run._started resets the peak, run._spent
        # reads it).
        import run as running

        started, spent = running._started, running._spent

        def unit_started(models: Any) -> Any:
            result = started(models)
            state["marks"], state["entered"] = [], None
            state["before"] = torch().cuda.memory_allocated(models.device)
            return result

        def unit_spent(models: Any, began: Any) -> Any:
            cuda = torch().cuda
            marks, state["marks"] = state["marks"], None
            if marks is not None:
                emit(
                    {
                        "step": "unit",
                        "before": state["before"],
                        "entered": state["entered"],
                        "peak": cuda.max_memory_allocated(models.device),
                        "marks": marks,
                    }
                )
            return spent(models, began)

        running._started, running._spent = unit_started, unit_spent

    emit({"step": "start", "argv": cli, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
    status = command.main(cli)
    emit({"step": "end", "status": status})
    return status


if __name__ == "__main__":
    sys.exit(main())

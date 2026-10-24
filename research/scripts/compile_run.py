#!/usr/bin/env python3
"""seedvr2x's own CLI, run with torch.compile on its VAE and/or DiT, and what compile did, per unit.

What it measures: a real seedvr2x job (its first pass, plan, calibration, encodes, DiT windows,
decodes and assembly) where the only difference is that, right after the models are loaded, the
VAE and the DiT are given to compile_vae.install_vae and compile_dit.install_dit. Per unit it
records the kind, the shot, the size, the wall seconds (the GPU synchronised at both ends), the
peak torch allocated during the unit (reset before) with the memory allocated before and after,
the compilations that happened during it and the seconds they took, the recompiles with their
reasons, and the time estimate seedvr2x had for the unit before it, so that the estimate's error
can be read per unit. With both variants "plain", nothing is installed and nothing is added but
the records and the GPU synchronisations at the unit's ends: the output is seedvr2x's own, bit
for bit.

How it hooks seedvr2x, never editing it (everything is patched at run time, in this process):
- seedvr2x's CLI sets the allocator (runtime/memory.py: startup) before torch is imported, so
  the wrapper imports nothing of torch before it: it wraps memory.startup, and installs its
  patches right after the original returns, then calls cli.main(argv) unchanged.
- model.load_models: the models returned, install_vae(models.runner.vae, ...) and
  install_dit(models.runner.dit, ...) are applied to them (the CLI imports load_models at the call,
  from the module, so the module's attribute is the only binding that matters; every other
  binding found in seedvr2x's modules is replaced too).
- run._started and run._made, which every encode, DiT window and decode part of a run calls
  around its work (run.py: _sample, _decode): the unit's record begins in the first (peak reset by
  seedvr2x itself, memory and compile counters snapshotted) and ends in the second, which has the
  unit's key, kind, size, seconds' terms and the clock, so that the estimate is read before the
  clock corrects its model with the unit (Progress.made). A decode part of a shot with several runs
  gives one record per run, under the same segment key.
- planner.plan (with --record-flags): the plan returned has compile_dit and compile_vae true for
  what is compiled (runtime/plan.py records them, false, "kept so building it later reshapes
  nothing"). Nothing else in seedvr2x reads them.
- run.encode_shot, run.sample_windows and run.decode_shot, which the calibration calls through the
  module (calibration.py: run.encode_shot...) and the run calls as its own globals: inside
  calibration.calibrate only, each call is a record of kind "calibration" (a window's is made
  around each step of its generator). The calibration's changes of phase (Models.enter) are not in
  them. Their order is the calibration's: encode warm-up (13 frames), window, encode (FIRST), window,
  encode (second length), window (longest), draw (not recorded), windows (short, longest), three
  decodes. Outside it the wrappers pass straight through.

Compile figures come from torch._dynamo.utils: get_compilation_metrics() (one record per frame
compiled: entire_frame_compile_time_s, backend_compile_time_s, inductor_compile_time_s, the
recompile_reason and fail_reason), counters["stats"]["unique_graphs"], counters["frames"],
counters["graph_break"] (summed) and counters["inductor"] (the FX-graph cache's hits and
misses). A warm Inductor or Triton cache makes the compile seconds far smaller than a cold one's:
point TORCHINDUCTOR_CACHE_DIR and TRITON_CACHE_DIR at a new directory for a cold measure.

Output (--units-json): JSON Lines, one object per line, "record" being "install" (the models'
loading and the installers' seconds and compilations), "unit", "calibration" (one step), "plan"
(the flags set) or "run" (the whole run: exit status, wall seconds, totals).

Usage (python is seedvr2x's venv's; everything after -- is seedvr2x's own command line):
    python compile_run.py [--vae VARIANT] [--vae-fullgraph] [--dit VARIANT] [--dit-dynamic]
        [--recompile-limit N] [--mode MODE] [--record-flags] [--units-json FILE]
        -- INPUT -o OUTPUT --model-dir DIR --dit-model FILE --vae-model FILE ...
VAE variants: plain, numz, excl, int. DiT variants: plain, whole, blocks. Defaults: plain (the
wrapper is then a no-op).
"""

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
VAE_VARIANTS = ("plain", "numz", "excl", "int")
DIT_VARIANTS = ("plain", "whole", "blocks")
REASON_CHARS = 240


class Recorder:
    """The records, appended to the JSON Lines file as they are made, and the compile counters."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.file = None if path is None else path.open("a", buffering=1)
        self.records: list[dict[str, Any]] = []
        self.layout: list[tuple[int, int]] = []  # the last sample_windows' layout
        self.begins: dict[
            float, dict[str, Any]
        ] = {}  # a unit's start (run._started) -> snapshot
        self.calibrating = False
        self.steps = 0
        self.dynamo: Any = None

    def write(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        if self.file is not None:
            self.file.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def snapshot(self) -> dict[str, Any]:
        """The compile counters now, to be subtracted from (delta)."""
        utils = self.dynamo
        counters = utils.counters
        return {
            "metrics": len(utils.get_compilation_metrics()),
            "unique_graphs": counters["stats"]["unique_graphs"],
            "frames_ok": counters["frames"]["ok"],
            "frames_total": counters["frames"]["total"],
            "graph_breaks": sum(counters["graph_break"].values()),
            "inductor": dict(counters["inductor"]),
            "inner_s": sum(
                utils.compilation_time_metrics.get("_compile.compile_inner", [])
            ),
        }

    def delta(self, before: dict[str, Any]) -> dict[str, Any]:
        """What compiled since `before`: the compilations, their seconds, the recompiles."""
        now = self.snapshot()
        made = list(self.dynamo.get_compilation_metrics())[before["metrics"] :]

        def total(field: str) -> float:
            return round(sum(getattr(each, field, None) or 0.0 for each in made), 4)

        reasons = [each.recompile_reason for each in made if each.recompile_reason]
        failed = [each.fail_reason for each in made if each.fail_reason]
        cache = {
            name: now["inductor"].get(name, 0) - before["inductor"].get(name, 0)
            for name in sorted(now["inductor"])
            if now["inductor"].get(name, 0) != before["inductor"].get(name, 0)
        }
        return {
            "compilations": len(made),
            "unique_graphs": now["unique_graphs"] - before["unique_graphs"],
            "frames_ok": now["frames_ok"] - before["frames_ok"],
            "graph_breaks": now["graph_breaks"] - before["graph_breaks"],
            "frame_compile_s": total("entire_frame_compile_time_s"),
            "backend_compile_s": total("backend_compile_time_s"),
            "inductor_compile_s": total("inductor_compile_time_s"),
            "compile_inner_s": round(now["inner_s"] - before["inner_s"], 4),
            "recompiles": len(reasons),
            "recompile_reasons": [reason[:REASON_CHARS] for reason in reasons[:3]],
            "failures": [reason[:REASON_CHARS] for reason in failed[:3]],
            "inductor_counters": cache,
        }


def device_of(models: Any) -> Any:
    return models.device


def install(options: argparse.Namespace, recorder: Recorder) -> None:
    """Patch seedvr2x's modules (called once torch may be imported)."""
    import torch
    import torch._dynamo.utils as dynamo
    from seedvr2x.runtime import calibration, model, planner, run

    dynamo.set_compilation_metrics_limit(1_000_000)
    recorder.dynamo = dynamo
    flags = (options.dit != "plain", options.vae != "plain")

    def synchronised(device: Any) -> None:
        torch.cuda.synchronize(device)

    def memory(device: Any) -> tuple[int, int]:
        return torch.cuda.memory_allocated(device), torch.cuda.max_memory_allocated(
            device
        )

    def use(runner: Any, name: str, handle: Any) -> None:
        """The handle's module runs as the runner's (install_dit's "whole" is a wrapper around the
        DiT, which the runner must call); a variant patched in place has none of its own."""
        module = getattr(handle, "module", None)
        if module is not None and module is not getattr(runner, name):
            setattr(runner, name, module)

    # The models loaded, then compiled.
    loaded = model.load_models

    def load_models(*args: Any, **kwargs: Any) -> Any:
        started = time.monotonic()
        models = loaded(*args, **kwargs)
        record: dict[str, Any] = {
            "record": "install",
            "load_s": round(time.monotonic() - started, 3),
            "vae": options.vae,
            "dit": options.dit,
        }
        if options.vae != "plain" or options.dit != "plain":
            sys.path.insert(0, str(HERE))
            before = recorder.snapshot()
            begun = time.monotonic()
            if options.keep_deterministic:
                # Inductor's deterministic mode, kept on through dynamo's resets of it, so that
                # two processes compiling cold give the same kernels' configs, hence bits
                # (compile_dit.keep_deterministic).
                import torch
                from compile_dit import keep_deterministic  # type: ignore[import-not-found]

                keep_deterministic(torch, "inductor")
            if options.vae != "plain":
                from compile_vae import install_vae  # type: ignore[import-not-found]

                handle = install_vae(
                    models.runner.vae,
                    options.vae,
                    mode=options.mode,
                    fullgraph=options.vae_fullgraph,
                    dynamic=False,
                )
                use(models.runner, "vae", handle)
                record["vae_info"] = getattr(handle, "info", None)
            if options.dit != "plain":
                from compile_dit import install_dit  # type: ignore[import-not-found]

                handle = install_dit(
                    models.runner.dit,
                    options.dit,
                    dynamic=True if options.dit_dynamic else None,
                    mode=options.mode,
                    recompile_limit=options.recompile_limit,
                )
                use(models.runner, "dit", handle)
                record["dit_info"] = getattr(handle, "info", None)
            record["install_s"] = round(time.monotonic() - begun, 3)
            record.update(recorder.delta(before))
        recorder.write(record)
        return models

    for name, module in list(sys.modules.items()):
        if (
            name.startswith("seedvr2x")
            and getattr(module, "load_models", None) is loaded
        ):
            module.load_models = load_models

    # The plan's flags (--record-flags).
    original_plan = planner.plan

    def plan(*args: Any, **kwargs: Any) -> Any:
        made, peaks = original_plan(*args, **kwargs)
        if options.record_flags:
            made = replace(made, compile_dit=flags[0], compile_vae=flags[1])
            recorder.write(
                {"record": "plan", "compile_dit": flags[0], "compile_vae": flags[1]}
            )
        return made, peaks

    planner.plan = plan

    # The VAE's predicted peaks scaled (--vae-peak-factor): a compiled VAE's peak above what is on
    # the GPU before its pass, held to F times plain code's (docs/compile.md: excl 1.1, int 1.6),
    # so that the planner chooses its tiles, and judges every VAE pass, by the compiled peak.
    if options.vae_peak_factor != 1.0:
        fit = planner._fit
        factor = options.vae_peak_factor

        def scaled(padded: Any, tiles: Any, column: int) -> float:
            return factor * fit(padded, tiles, column)

        planner._fit = scaled
        recorder.write({"record": "vae_peak_factor", "factor": factor})

    # The run's units.
    started = run._started
    made = run._made

    def begin(models: Any) -> float:
        device = device_of(models)
        synchronised(device)
        stamp = started(models)  # resets the peak, takes the time
        recorder.begins[stamp] = {
            "allocated_before": memory(device)[0],
            "snapshot": recorder.snapshot(),
            "device": device,
        }
        return stamp

    def end(
        progress: Any,
        key: Any,
        kind: str,
        size: int,
        stamp: float,
        unit: str,
        counted: bool = True,
        *,
        entering: float = 0.0,
        term: float = 0.0,
        drawn: int | None = None,
    ) -> str:
        begun = recorder.begins.pop(stamp, None)
        if begun is not None:
            device = begun["device"]
            synchronised(device)
            whole = time.monotonic() - stamp
            allocated, peak = memory(device)
            clock = progress.clock
            drawing = (
                (clock.estimate("draw", drawn) or 0.0) if drawn is not None else 0.0
            )
            estimate = clock.estimate(kind, size)
            record: dict[str, Any] = {
                "record": "unit",
                "kind": kind,
                "unit": unit,
                "key": list(key),
                "run": key[1],
                "size": size,
                "counted": counted,
                "wall_s": round(whole, 4),
                "kind_s": round(whole - entering - drawing, 4),
                "entering_s": round(entering, 4),
                "term_est_s": round(term, 4),
                "draw_est_s": round(drawing, 4),
                "estimate_kind_s": None if estimate is None else round(estimate, 4),
                "estimate_whole_s": (
                    None if estimate is None else round(estimate + term + drawing, 4)
                ),
                "peak_allocated": peak,
                "allocated_before": begun["allocated_before"],
                "allocated_after": allocated,
            }
            if kind == "window":
                number = key[2]
                if number < len(recorder.layout):
                    first, last = recorder.layout[number]
                    record["latents"] = last - first
            else:
                record["frames"] = size
            record.update(recorder.delta(begun["snapshot"]))
            recorder.write(record)
        return made(
            progress,
            key,
            kind,
            size,
            stamp,
            unit,
            counted,
            entering=entering,
            term=term,
            drawn=drawn,
        )

    run._started = begin
    run._made = end

    # The calibration's own steps.
    def stepped(kind: str, size: int, models: Any, work: Any) -> Any:
        device = device_of(models)
        synchronised(device)
        before = recorder.snapshot()
        torch.cuda.reset_peak_memory_stats(device)
        allocated = memory(device)[0]
        begun = time.monotonic()
        try:
            result = work()
        except BaseException as error:
            step_record(kind, size, device, before, allocated, begun, failed=error)
            raise
        return result, device, before, allocated, begun

    def step_record(
        kind: str,
        size: int,
        device: Any,
        before: Any,
        allocated: int,
        begun: float,
        failed: BaseException | None = None,
    ) -> None:
        synchronised(device)
        wall = time.monotonic() - begun
        after, peak = memory(device)
        recorder.steps += 1
        record: dict[str, Any] = {
            "record": "calibration",
            "step": recorder.steps,
            "kind": kind,
            "frames" if kind != "window" else "latents": size,
            "wall_s": round(wall, 4),
            "peak_allocated": peak,
            "allocated_before": allocated,
            "allocated_after": after,
        }
        if failed is not None:
            record["failed"] = f"{type(failed).__name__}: {failed}"[:300]
        record.update(recorder.delta(before))
        recorder.write(record)

    encode_shot, decode_shot, sample_windows = (
        run.encode_shot,
        run.decode_shot,
        run.sample_windows,
    )

    def counted_call(kind: str, original: Any) -> Any:
        def call(models: Any, data: Any, count: int, *args: Any, **kwargs: Any) -> Any:
            if not recorder.calibrating:
                return original(models, data, count, *args, **kwargs)
            result, device, before, allocated, begun = stepped(
                kind,
                count,
                models,
                lambda: original(models, data, count, *args, **kwargs),
            )
            step_record(kind, count, device, before, allocated, begun)
            return result

        return call

    def windows(
        models: Any, latent: Any, layout: Any, *args: Any, **kwargs: Any
    ) -> Any:
        recorder.layout = list(layout)
        inner = sample_windows(models, latent, layout, *args, **kwargs)
        if not recorder.calibrating:
            return inner
        return stepping(models, inner, layout)

    def stepping(models: Any, inner: Any, layout: Any) -> Any:
        number = 0
        while True:
            first, last = layout[number] if number < len(layout) else (0, 0)
            device = device_of(models)
            synchronised(device)
            before = recorder.snapshot()
            torch.cuda.reset_peak_memory_stats(device)
            allocated = memory(device)[0]
            begun = time.monotonic()
            try:
                output = next(inner)
            except StopIteration:
                return
            except BaseException as error:
                step_record(
                    "window",
                    last - first,
                    device,
                    before,
                    allocated,
                    begun,
                    failed=error,
                )
                raise
            step_record("window", last - first, device, before, allocated, begun)
            number += 1
            yield output

    run.encode_shot = counted_call("encode", encode_shot)
    run.decode_shot = counted_call("decode", decode_shot)
    run.sample_windows = windows

    calibrate = calibration.calibrate

    def calibrating(*args: Any, **kwargs: Any) -> Any:
        recorder.calibrating = True
        before = recorder.snapshot()
        begun = time.monotonic()
        try:
            return calibrate(*args, **kwargs)
        finally:
            recorder.calibrating = False
            record = {
                "record": "calibration_total",
                "wall_s": round(time.monotonic() - begun, 3),
            }
            record.update(recorder.delta(before))
            recorder.write(record)

    calibration.calibrate = calibrating


def parse(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    split = argv.index("--") if "--" in argv else len(argv)
    own, rest = argv[:split], argv[split + 1 :]
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0], prog="compile_run.py"
    )
    parser.add_argument("--vae", choices=VAE_VARIANTS, default="plain")
    parser.add_argument("--vae-fullgraph", action="store_true")
    parser.add_argument("--dit", choices=DIT_VARIANTS, default="plain")
    parser.add_argument("--dit-dynamic", action="store_true")
    parser.add_argument("--recompile-limit", type=int, default=None)
    parser.add_argument(
        "--vae-peak-factor",
        type=float,
        default=1.0,
        help="the planner's VAE peaks (planner._fit) times F: a compiled VAE planned for",
    )
    parser.add_argument(
        "--keep-deterministic",
        action="store_true",
        help="inductor's deterministic mode on, kept through dynamo's resets (compile_dit.py)",
    )
    parser.add_argument(
        "--mode", default="default", help="torch.compile's mode, both models"
    )
    parser.add_argument(
        "--record-flags",
        action="store_true",
        help="the job's plan records compile_dit and compile_vae true for what is compiled",
    )
    parser.add_argument(
        "--units-json", type=Path, default=None, help="JSON Lines, appended"
    )
    return parser.parse_args(own), rest


def main() -> int:
    options, rest = parse(sys.argv[1:])
    recorder = Recorder(options.units_json)
    # No torch yet: seedvr2x sets the allocator in memory.startup before importing it; the
    # patches follow the original startup, whatever it returns.
    from seedvr2x.runtime import memory

    startup = memory.startup

    def patched() -> Any:
        found = startup()
        install(options, recorder)
        return found

    memory.startup = patched
    from seedvr2x import cli

    begun = time.monotonic()
    status = 1
    try:
        status = cli.main(rest)
        return status
    finally:
        for stamp, begun_unit in list(recorder.begins.items()):
            # A unit that began and never ended: the run stopped inside it (out of memory...).
            import torch

            recorder.write(
                {
                    "record": "unit_unfinished",
                    "wall_s": round(time.monotonic() - stamp, 4),
                    "peak_allocated": torch.cuda.max_memory_allocated(
                        begun_unit["device"]
                    ),
                    "allocated_before": begun_unit["allocated_before"],
                    **recorder.delta(begun_unit["snapshot"]),
                }
            )
        units = [r for r in recorder.records if r["record"] == "unit"]
        steps = [r for r in recorder.records if r["record"] == "calibration"]
        recorder.write(
            {
                "record": "run",
                "status": status,
                "wall_s": round(time.monotonic() - begun, 3),
                "units": len(units),
                "calibration_steps": len(steps),
                "unit_compile_s": round(sum(r["frame_compile_s"] for r in units), 3),
                "calibration_compile_s": round(
                    sum(r["frame_compile_s"] for r in steps), 3
                ),
                "unit_compilations": sum(r["compilations"] for r in units),
                "max_peak_allocated": max(
                    (r["peak_allocated"] for r in units), default=None
                ),
            }
        )


if __name__ == "__main__":
    sys.exit(main())

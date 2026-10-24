#!/usr/bin/env python3
"""torch.compile on seedvr2x's VAE: memory, time, compilations and bits, against plain code.

seedvr2x doesn't compile its models. numz's CLI offers --compile_vae (torch.compile on the VAE's
encoder and decoder), which costs about twice the VAE's working memory
(research/bugs/14-compile-vae-doubles-memory.md). This harness measures, on seedvr2x's own VAE
paths, untiled and tiled, what compiling the VAE costs and gives, and where the memory goes.

What runs is seedvr2x's runtime, imported from its venv and never edited: model.load_models with
the 7B fp16 DiT and the fp16 VAE, as the CLI loads them, then the DiT dropped (runtime.dit = None,
the allocator emptied: load_models builds both and has no VAE-only load): what stays allocated is
the VAE's weights and the text embeddings (1.2 MiB), both figures recorded. The encode is
shot.encode_shot, as the pipeline calls it: the VAE called slice by slice (5 frames, then 4), the
posterior's mode and the latent's shift and scale as model.encode_stream does them, or with
--encode-tile PX tiling.encode_tiled (a tile's crop of the same slices per tile). The decode is
model.decode_stream (2 latents, then 1), or with --decode-tile PX tiling.decode_tiled, each slice
clamped and moved to the CPU as it comes, as shot.decode_shot does; the latent waits in system
memory, as the pipeline's does. The decode takes the first plain encode's latent, with no DiT in
between: a VAE round trip has the same shapes, memory and time as the pipeline's decode, and every
decode of a run has the same input, so that its frames compare bit for bit. The tiles' scratch is
in RAM, or in --scratch DIR (a file there).

The input is a clip read with ffmpeg (--clip), resized to --size as the pipeline resizes it
(shot.encoder_inputs, model.input_transform), or without one random frames at --source-size: the
content changes no shape, memory or compilation, nor does it make bits comparable differently (the
same frames go through every pass), but only a clip gives the pipeline's own values, so use a clip
for figures that are meant to be the pipeline's.

One process = one variant, compared with plain code in the same process, back to back (the GPU's
speed drifts: times are valid as ratios of neighbouring passes only): --passes ABAB runs plain (A)
and compiled (B) passes in turn, switching by installing and restoring the variant, no reload.
The first B compiles: its time and its own peak are recorded apart. Variants (--variant), each
installed by install_vae(vae, variant, ...), which other scripts import:
  plain  nothing compiled
  numz   vae.encoder and vae.decoder each through torch.compile with numz's settings (inductor,
         mode default, fullgraph False, dynamic False; recompile limit 128), the convolutions as
         the code stands: the graph breaks at each convolution call, which branches on a tensor's
         value (memory_limit_conv)
  excl   the same, each InflatedCausalConv3d.forward wrapped by torch.compiler.disable first, as
         numz meant its exclusion (research/bugs/15-compile-vae-exclusion-dead-code.md)
  int    memory_limit_conv's size arithmetic on Python integers, so that Dynamo traces it (the
         vendored code's torch.tensor arithmetic is what breaks the graph), the rounding of the
         float32 tensors kept, so the same splits, and the head, whole and whole_output behaviour
         of the output written in one piece reproduced; then compiled as numz; --fullgraph turns
         fullgraph on for any compiled variant
The patch's bits against the vendored method are checked by `selftest`.

Per phase of a pass: wall time, GPU time between two events, time per frame, torch's peak
allocated (reset before) and that above what was allocated before, allocated before and after
(outputs on the CPU, references dropped, gc.collect()), per slice (vae._encode and vae._decode
wrapped on the instance) the same, and per tile when tiled. Compilations are counted from torch:
torch._dynamo.utils.counters, its compilation metrics, and the recompiles and graph_breaks log
records, counted with their distinct reasons, not printed. After each B the latent and the frames
are compared with the first A's, bit for bit; when they differ: the largest and mean absolute
difference (8-bit levels on the clamped frames) and PSNR. --shots N,N,... then runs shots of those
lengths in turn (plain then compiled), compilations counted after each. --leave-gpu ends the run
with: what stays allocated after a compiled encode and decode and what frees it; the VAE moved to
the CPU and back (what stays on the GPU meanwhile, recompilations and bits at the next call);
torch._dynamo.reset(). One self-describing JSON per run: arguments, versions, card, attention,
allocator, cache directories, every measurement.

Subcommands:
  run      --model-dir DIR --out RESULTS.json [--clip FILE | --source-size HxW] [--size HxW]
           [--frames N] [--variant V] [--fullgraph] [--passes ABAB] [--phases encode,decode]
           [--encode-tile PX] [--decode-tile PX] [--overlap PX] [--scratch DIR] [--mode M]
           [--inductor NAME=VALUE ...] [--torch-logs NAME,... --code-log FILE]
           [--shots N,N,...] [--leave-gpu] [--reset-call]
  table    RESULTS.json [...] [--slices]   a Markdown summary of one or several runs
  selftest CPU only (CUDA_VISIBLE_DEVICES=): the int patch against the vendored
           memory_limit_conv bit for bit on real causal convolutions with random weights (splits
           along H and W, prev caches, the three memory states, both sides of the one-output
           threshold), its arithmetic against torch's, and the int variant compiled with
           backend eager and fullgraph on a small VAE with no graph break

The allocator is chosen as seedvr2x's CLI chooses it, before torch is imported
(runtime/memory.startup: cudaMallocAsync, or expandable segments where the driver spills, as
WSL2's does), unless PYTORCH_CUDA_ALLOC_CONF is set. ffmpeg and ffprobe on PATH for --clip. The
GPU is one job's at a time: run under the shared GPU lock.

Usage (seedvr2x's venv's python):
  python compile_vae.py run --model-dir MODELS --clip CLIP.mkv --size 540x960 --frames 9 \\
      --variant int --passes ABAB --out smoke.json
  python compile_vae.py run --model-dir MODELS --clip CLIP.mkv --size 1080x1920 --frames 21 \\
      --variant int --encode-tile 1024 --decode-tile 1024 --passes AB --out tiled.json
  python compile_vae.py table smoke.json tiled.json
  CUDA_VISIBLE_DEVICES= python compile_vae.py selftest
"""

import argparse
import contextlib
import gc
import hashlib
import itertools
import json
import logging
import math
import os
import statistics
import subprocess
import sys
import time
import types
from collections import Counter
from pathlib import Path


def _choose_allocator():
    """The allocator seedvr2x's CLI would choose, set before torch is imported: its own startup
    (it asks the driver whether it refuses what the card can't hold), unless the environment has
    a setting; cudaMallocAsync, the CLI's default, where seedvr2x can't be imported yet."""
    try:
        from seedvr2x.runtime import memory

        memory.startup()
    except Exception as error:  # noqa: BLE001 - the harness runs anywhere the venv does
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
        print(f"[compile_vae] memory.startup() unavailable ({error!r}): cudaMallocAsync",
              file=sys.stderr)


if __name__ == "__main__" and sys.argv[1:2] == ["run"]:
    _choose_allocator()

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError:  # `table` needs neither
    np = torch = F = nn = None

GIB = 1024**3
DIT = "seedvr2_ema_7b_fp16.safetensors"
VAE = "ema_vae_fp16.safetensors"
VARIANTS = ("plain", "numz", "excl", "int")
# numz's defaults (src/core/model_configuration.py:1307-1315 at 4490bd1, inference_cli.py:1455-1466):
# it sets cache_size_limit 64 then recompile_limit 128, one setting in this torch, so 128 holds.
NUMZ_SETTINGS = {"backend": "inductor", "mode": "default", "fullgraph": False, "dynamic": False}
NUMZ_RECOMPILE_LIMIT = 128
REASON_CHARS = 400
LEFT_THRESHOLD = 16 * 1024**2  # bytes left on the GPU worth listing
RAM = 2**62  # a scratch allowance no store exceeds: the tiles' scratch in system memory


def log(message):
    print(f"[compile_vae {time.strftime('%H:%M:%S')}] {message}", flush=True)


def gib(count):
    return round(count / GIB, 3)


# ---------------------------------------------------------------------------------------------
# The clip


def probe_clip(clip):
    """(width, height, frames) of the clip's video stream, its packets counted."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
         "stream=width,height,nb_read_packets", "-of", "json", str(clip)],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(out.stdout)["streams"][0]
    return int(stream["width"]), int(stream["height"]), int(stream["nb_read_packets"])


def read_clip(clip, count):
    """The clip's first `count` frames: (count, H, W, 3) uint8, RGB."""
    width, height, total = probe_clip(clip)
    if total < count:
        sys.exit(f"compile_vae: {clip} has {total} frames, {count} asked for")
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", str(clip), "-frames:v", str(count),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(count, height, width, 3)


def reader(frames):
    """read(n) as shot.encode_shot takes it: the next n frames, (n, H, W, 3) float32 in [0, 1]."""
    position = 0

    def read(n):
        nonlocal position
        part = frames[position : position + n].astype(np.float32) / 255.0
        position += n
        return part

    return read


def random_frames(count, height, width):
    """count random frames, (count, H, W, 3) uint8: a source when no clip is given."""
    return np.random.default_rng(0).integers(0, 256, (count, height, width, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------------------------
# Compilations, counted


def reason(message):
    """A log record's reason, on one line, without its traceback."""
    text = message.split("User code traceback", 1)[0]
    text = text.replace("Graph break (user stack suppressed due to duplicate graph break) in",
                        "Graph break in")
    return " ".join(text.split())[:REASON_CHARS]


class Tally(logging.Filter):
    """Counts torch's recompiles and graph_breaks log records, with their distinct reasons, in
    place of their printing; the other artefacts asked for (--torch-logs) go to a file."""

    def __init__(self, code_log):
        super().__init__()
        self.recompiles = Counter()
        self.graph_breaks = Counter()
        self.code = open(code_log, "w") if code_log else None  # noqa: SIM115
        self.last = None

    def filter(self, record):
        name = record.name
        if name.endswith((".__recompiles", ".__recompiles_verbose")):
            kind = self.recompiles
        elif name.endswith(".__graph_breaks"):
            kind = self.graph_breaks
        elif ".__" in name and self.code is not None:
            if record is not self.last:
                self.code.write(f"##### {name}\n{record.getMessage()}\n")
                self.code.flush()
            self.last = record
            return False
        else:
            return True
        if record is not self.last:
            kind[reason(record.getMessage())] += 1
        self.last = record
        return False


def install_tally(extra, code_log):
    """torch's recompiles and graph_breaks logs on (and `extra` artefacts), into a Tally."""
    import torch._logging._internal as internal

    torch._logging.set_logs(recompiles=True, graph_breaks=True, **{name: True for name in extra})
    tally = Tally(code_log)
    for handler in list(internal.handlers):
        handler.addFilter(tally)
    torch._dynamo.utils.set_compilation_metrics_limit(1_000_000)
    return tally


def compile_counts(tally):
    """Counters of what has been compiled so far in this process."""
    counters = torch._dynamo.utils.counters
    metrics = torch._dynamo.utils.get_compilation_metrics()
    return {
        "frames_ok": counters["frames"]["ok"],
        "frames_total": counters["frames"]["total"],
        "unique_graphs": counters["stats"]["unique_graphs"],
        "resume_functions": sum(counters["resumes"].values()),
        "compilations": len(metrics),
        "compile_seconds": round(sum(m.entire_frame_compile_time_s or 0.0 for m in metrics), 2),
        "recompiles_logged": sum(tally.recompiles.values()),
        "graph_breaks_logged": sum(tally.graph_breaks.values()),
    }


def delta(after, before):
    return {key: round(after[key] - before[key], 2) for key in after}


def compile_summary(tally):
    """The whole process's compilations: counts, times, and the distinct reasons (top 10)."""
    counters = torch._dynamo.utils.counters
    metrics = torch._dynamo.utils.get_compilation_metrics()
    by_function = Counter()
    seconds = Counter()
    recompile_reasons = Counter()
    for m in metrics:
        key = f"{m.co_name} ({Path(m.co_filename or '?').name}:{m.co_firstlineno})"
        by_function[key] += 1
        seconds[key] += m.entire_frame_compile_time_s or 0.0
        if m.recompile_reason:
            recompile_reasons[" ".join(str(m.recompile_reason).split())[:REASON_CHARS]] += 1

    def total(field):
        return round(sum(getattr(m, field, None) or 0.0 for m in metrics), 2)

    return {
        "counts": compile_counts(tally),
        "seconds": {
            "entire_frame": total("entire_frame_compile_time_s"),
            "backend": total("backend_compile_time_s"),
            "inductor": total("inductor_compile_time_s"),
            "code_gen": total("code_gen_time_s"),
        },
        "graph_ops": sum(m.graph_op_count or 0 for m in metrics),
        "functions_compiled": [
            {"function": key, "compilations": count, "seconds": round(seconds[key], 2)}
            for key, count in by_function.most_common(15)
        ],
        "distinct_functions": len(by_function),
        "recompile_reasons_metrics": recompile_reasons.most_common(10),
        "distinct_recompile_reasons_metrics": len(recompile_reasons),
        "recompile_reasons_logged": tally.recompiles.most_common(10),
        "distinct_recompile_reasons_logged": len(tally.recompiles),
        "graph_break_reasons_logged": tally.graph_breaks.most_common(10),
        "distinct_graph_break_reasons_logged": len(tally.graph_breaks),
        "unimplemented": Counter(
            {" ".join(k.split())[:REASON_CHARS]: v for k, v in counters["unimplemented"].items()}
        ).most_common(10),
        "resume_functions_distinct": len(counters["resumes"]),
    }


# ---------------------------------------------------------------------------------------------
# Memory


def allocated():
    return torch.cuda.memory_allocated()


def device_memory():
    free, total = torch.cuda.mem_get_info()
    return {"free": free, "total": total, "used": total - free, "allocated": allocated()}


class Meter:
    """The peak of a phase across its slices' own peaks, and each slice's record. A slice that
    doesn't continue the causal memory (INITIALIZING, or DISABLED: a pass in one slice) starts a
    new pass, a tile's when the phase is tiled: `tile` counts them."""

    def __init__(self):
        self.slices = []
        self.peak = 0
        self.tile = -1

    def roll(self):
        """Fold torch's peak since the last reset into the phase's, and reset it."""
        self.peak = max(self.peak, torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats()

    def begin(self):
        self.slices = []
        self.tile = -1
        torch.cuda.reset_peak_memory_stats()
        self.peak = 0

    def wrap(self, vae, name, memory_states):
        """vae._encode or vae._decode, wrapped on the instance: each call's memory and time."""
        original = getattr(vae, name)

        def call(x, memory_state=memory_states.DISABLED):
            torch.cuda.synchronize()
            self.roll()
            if memory_state != memory_states.ACTIVE:
                self.tile += 1
            before = allocated()
            start = time.perf_counter()
            out = original(x, memory_state=memory_state)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            peak = torch.cuda.max_memory_allocated()
            self.slices.append({
                "call": name, "state": memory_state.name, "tile": self.tile,
                "input": list(x.shape), "allocated_before": before, "peak": peak,
                "allocated_after": allocated(), "seconds": round(seconds, 4),
            })
            return out

        setattr(vae, name, call)

    def per_tile(self):
        """The slices' records gathered by pass: peak, time, slices, the input's last two sides."""
        tiles = []
        for entry in self.slices:
            while len(tiles) <= entry["tile"]:
                tiles.append({"peak": 0, "seconds": 0.0, "slices": 0, "allocated_before": None,
                              "input_hw": entry["input"][-2:]})
            tile = tiles[entry["tile"]]
            if tile["allocated_before"] is None:
                tile["allocated_before"] = entry["allocated_before"]
            tile["peak"] = max(tile["peak"], entry["peak"])
            tile["seconds"] = round(tile["seconds"] + entry["seconds"], 4)
            tile["slices"] += 1
        return tiles


def tensors_in(value, depth=2):
    """The tensors in value: itself, or in a list, tuple or dict (two levels)."""
    if isinstance(value, torch.Tensor):
        yield value
    elif depth and isinstance(value, list | tuple):
        for item in value:
            yield from tensors_in(item, depth - 1)
    elif depth and isinstance(value, dict):
        for item in value.values():
            yield from tensors_in(item, depth - 1)


def census(known, top=12, referrers=False):
    """The CUDA tensors alive, one per storage, those of `known` storages left out: their count
    and bytes, how much the running frames' variables hold (by kind of frame: those of
    torch.compile's resume functions apart), and the `top` largest with what holds them."""
    alive = {}
    objects = {}
    for obj in gc.get_objects():
        try:
            if not (isinstance(obj, torch.Tensor) and obj.is_cuda):
                continue
            storage = obj.untyped_storage()
            pointer = storage.data_ptr()
        except Exception:  # noqa: BLE001, S112 - anything gc holds may refuse inspection
            continue
        if pointer == 0 or pointer in known:
            continue
        alive.setdefault(pointer, {
            "bytes": storage.nbytes(), "shape": list(obj.shape), "dtype": str(obj.dtype),
            "holders": [],
        })
        objects.setdefault(pointer, []).append(obj)
    frame = sys._getframe(1)
    depth = 0
    while frame is not None:
        name = frame.f_code.co_name
        try:
            variables = list(frame.f_locals.items())
        except Exception:  # noqa: BLE001
            variables = []
        for variable, value in variables:
            for tensor in tensors_in(value):
                try:
                    pointer = tensor.untyped_storage().data_ptr() if tensor.is_cuda else 0
                except Exception:  # noqa: BLE001
                    pointer = 0
                if pointer in alive:
                    holder = f"{name}#{depth}:{variable}"
                    if holder not in alive[pointer]["holders"]:
                        alive[pointer]["holders"].append(holder)
        frame = frame.f_back
        depth += 1
    if referrers:
        for pointer, entry in alive.items():
            kinds = Counter()
            for obj in objects[pointer]:
                for referrer in gc.get_referrers(obj):
                    if referrer is objects[pointer] or referrer is objects:
                        continue
                    kind = type(referrer).__name__
                    if kind == "frame":
                        kind = f"frame {referrer.f_code.co_name}"
                    elif kind == "dict":
                        owner = [type(o).__name__ for o in gc.get_referrers(referrer)[:3]]
                        kind = f"dict of {'/'.join(owner)}"
                    kinds[kind] += 1
            entry["referrers"] = dict(kinds.most_common(6))

    def held(test):
        return sum(e["bytes"] for e in alive.values() if any(test(h) for h in e["holders"]))

    largest = sorted(alive.values(), key=lambda e: -e["bytes"])[:top]
    return {
        "tensors": len(alive),
        "bytes": sum(e["bytes"] for e in alive.values()),
        "held_by_frames": held(lambda h: True),
        "held_by_resume_frames": held(lambda h: h.startswith("torch_dynamo_resume")),
        "held_by_no_frame": sum(e["bytes"] for e in alive.values() if not e["holders"]),
        "largest": [{**e, "holders": e["holders"][:6]} for e in largest],
    }


def known_storages(vae, *tensors):
    """The storages not to count as left over: the VAE's weights and the run's own tensors."""
    known = set()
    for tensor in [*vae.parameters(), *vae.buffers(), *tensors]:
        if tensor is not None and tensor.is_cuda:
            known.add(tensor.untyped_storage().data_ptr())
    return known


# ---------------------------------------------------------------------------------------------
# Variants


MISSING = object()
_SPLIT = 2.0**29 + 1.0  # Veltkamp's splitting constant for 53 - 24 = 29 bits


class Handle:
    """An installed variant: the attributes it set on the VAE and its modules. restore() sets
    them back (the original modules and methods are kept); apply() sets them again, with the
    compiled wrappers made at install time, so that their caches carry over between passes."""

    def __init__(self, variant, settings):
        self.variant = variant
        self.settings = settings
        self.items = []
        self.applied = False
        self.info = {"variant": variant, "compiled_units": [], "causal_convs": 0}
        self._limit = MISSING

    def add(self, owner, name, value):
        own = name in vars(owner) or (isinstance(owner, nn.Module) and name in owner._modules)
        self.items.append((owner, name, getattr(owner, name) if own else MISSING, value))

    def apply(self):
        """Set the variant's attributes (and, for a compiled variant, torch's recompile limit,
        which restore() sets back)."""
        if self.applied:
            return
        for owner, name, _, value in self.items:
            setattr(owner, name, value)
        if self.info["compiled_units"]:
            self._limit = torch._dynamo.config.recompile_limit
            torch._dynamo.config.recompile_limit = NUMZ_RECOMPILE_LIMIT
        self.applied = True

    def restore(self):
        """Set every attribute back to what it was: the VAE is plain code again."""
        if not self.applied:
            return
        for owner, name, original, _ in reversed(self.items):
            if original is MISSING:
                delattr(owner, name)
            else:
                setattr(owner, name, original)
        if self._limit is not MISSING:
            torch._dynamo.config.recompile_limit = self._limit
            self._limit = MISSING
        self.applied = False


def float32(count):
    """The float32 nearest to a non-negative integer (ties to even), as a Python float: what
    torch's int64 to float32 conversion gives, with no tensor."""
    bits = count.bit_length()
    if bits <= 24:
        return float(count)
    shift = bits - 24
    quotient, rest = count >> shift, count & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    if rest > half or (rest == half and quotient & 1):
        quotient += 1
    return float(quotient << shift)


def float32_of_float(value):
    """A positive Python float rounded to the nearest float32 (24 bits), as the float32 tensors
    the vendored arithmetic works in round it: Veltkamp's splitting, plain float arithmetic that
    Dynamo traces (tests against numpy's cast: selftest)."""
    scaled = value * _SPLIT
    return scaled - (scaled - value)


def int_limit_conv(get_cache_size):
    """InflatedCausalConv3d.memory_limit_conv with its size arithmetic on Python numbers.

    The vendored method (causal_inflation_lib.py, ByteDance's as numz modified it, Apache-2.0,
    then seedvr2x's head and whole) computes the padded input's size as a tensor
    (torch.tensor(x.size()) ... / 1024**3, a float32 0-dim tensor) and branches on it, which
    Dynamo can't trace: a graph break at every call. Here the same value comes from integers,
    rounded to float32 as the tensor's division rounds it, and compared with the limit as torch
    compares a float32 tensor with a Python float (the limit rounded to float32), the number of
    splits rounded to float32 before its ceiling, so that the same splits are made. The rest is
    the vendored method's, line by line: the convolution is called as Conv3d.forward calls it with
    its padding ignored (F.conv3d, zero padding, without the context manager), and whole_output,
    the vendored method, chooses the output's path (head, whole)."""

    def memory_limit_conv(
        self,
        x,
        *,
        split_dim=3,
        padding=(0, 0, 0, 0, 0, 0),
        prev_cache=None,
        head=0,
        whole=None,
    ):
        if math.isinf(self.memory_limit):
            if prev_cache is not None:
                x = torch.cat([prev_cache, x], dim=split_dim - 1)
            return F.conv3d(
                x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
            )

        shape = list(x.size())
        if prev_cache is not None:
            shape[split_dim - 1] += prev_cache.size(split_dim - 1)
        # padding is (left, right) for W, then H, then T: the vendored code's
        # torch.tensor(padding).view(3, 2).sum(-1).flip(0) added to shape[-3:].
        shape[-3] += padding[4] + padding[5]
        shape[-2] += padding[2] + padding[3]
        shape[-1] += padding[0] + padding[1]
        memory_occupy = float32(math.prod(shape) * x.element_size()) / 1024**3  # GiB
        limit = float32_of_float(self.memory_limit)
        if memory_occupy < limit or split_dim == x.ndim:
            x_concat = x
            if prev_cache is not None:
                x_concat = torch.cat([prev_cache, x], dim=split_dim - 1)
            padded = F.pad(x_concat, padding, mode="constant", value=0.0)
            return F.conv3d(
                padded, self.weight, self.bias, self.stride, (0, 0, 0), self.dilation, self.groups
            )

        num_splits = math.ceil(float32_of_float(memory_occupy / limit))
        size_per_split = x.size(split_dim) // num_splits
        split_sizes = [size_per_split] * (num_splits - 1)
        split_sizes += [x.size(split_dim) - sum(split_sizes)]

        x = list(x.split(split_sizes, dim=split_dim))
        if prev_cache is not None:
            prev_cache = list(prev_cache.split(split_sizes, dim=split_dim))
        output = self.whole_output(x[0], shape, head, whole)
        written = 0
        cache = None
        for idx in range(len(x)):
            if prev_cache is not None:
                x[idx] = torch.cat([prev_cache[idx], x[idx]], dim=split_dim - 1)

            lpad_dim = (x[idx].ndim - split_dim - 1) * 2
            rpad_dim = lpad_dim + 1
            padding = list(padding)
            padding[lpad_dim] = self.padding[split_dim - 2] if idx == 0 else 0
            padding[rpad_dim] = self.padding[split_dim - 2] if idx == len(x) - 1 else 0
            pad_len = padding[lpad_dim] + padding[rpad_dim]
            padding = tuple(padding)

            next_cache = None
            cache_len = cache.size(split_dim) if cache is not None else 0
            next_catch_size = get_cache_size(
                conv_module=self,
                input_len=x[idx].size(split_dim) + cache_len,
                pad_len=pad_len,
                dim=split_dim - 2,
            )
            if next_catch_size != 0:
                assert next_catch_size <= x[idx].size(split_dim)
                next_cache = (
                    x[idx].transpose(0, split_dim)[-next_catch_size:].transpose(0, split_dim)
                )

            piece = self.memory_limit_conv(
                x[idx],
                split_dim=split_dim + 1,
                padding=padding,
                prev_cache=cache,
                whole=output is not None,
            )
            if output is None:
                x[idx] = piece
            else:
                x[idx] = None
                length = piece.size(split_dim)
                place = list(output.shape)
                place[split_dim] = length
                if (
                    list(piece.shape) != place
                    or piece.dtype != output.dtype
                    or written + length > output.size(split_dim)
                ):
                    raise RuntimeError(
                        f"a piece of {tuple(piece.shape)} {piece.dtype} at {written} along"
                        f" dimension {split_dim} of an output of {tuple(output.shape)}"
                        f" {output.dtype}"
                    )
                output.narrow(split_dim, written, length).copy_(piece)
                written += length
            del piece

            cache = next_cache

        if output is None:
            return torch.cat(x, split_dim)
        if written != output.size(split_dim):
            raise RuntimeError(
                f"{written} of {output.size(split_dim)} written along dimension {split_dim}"
            )
        return output

    return memory_limit_conv


def install_vae(vae, variant, *, mode="default", fullgraph=False, dynamic=False,
                backend="inductor"):
    """Install a variant on a loaded seedvr2x VAE (vae, the model object: runner.vae), applied at
    once; returns a Handle: handle.restore() puts the plain code back, handle.apply() the variant
    again (the same compiled wrappers). No state is kept outside the handle but torch's recompile
    limit, set to numz's 128 while a compiled variant is applied. Nothing is read from argparse.

      plain  nothing
      numz   vae.encoder and vae.decoder through torch.compile(backend, mode, fullgraph, dynamic):
             numz's, inductor, "default", False, False; the convolutions as they are
      excl   numz, each InflatedCausalConv3d.forward first wrapped by torch.compiler.disable
      int    the convolutions' memory_limit_conv replaced on each instance by int_limit_conv's,
             then compiled as numz

    ValueError for another variant, or the int one on a convolution with another padding mode
    than zeros."""
    from seedvr2x.vendor.models.video_vae_v3.modules import (
        causal_inflation_lib,
        context_parallel_lib,
    )

    if variant not in VARIANTS:
        raise ValueError(f"variant {variant!r}, one of {VARIANTS}")
    settings = {"backend": backend, "mode": mode, "fullgraph": fullgraph, "dynamic": dynamic}
    handle = Handle(variant, settings)
    convs = [m for m in vae.modules() if isinstance(m, causal_inflation_lib.InflatedCausalConv3d)]
    handle.info["causal_convs"] = len(convs)
    if variant == "int":
        if any(conv.padding_mode != "zeros" for conv in convs):
            raise ValueError("the int patch is for zero-padded convolutions")
        patch = int_limit_conv(context_parallel_lib.get_cache_size)
        for conv in convs:
            handle.add(conv, "memory_limit_conv", types.MethodType(patch, conv))
    if variant == "excl":
        for conv in convs:
            handle.add(conv, "forward", torch.compiler.disable(conv.forward))
    if variant != "plain":
        for attribute in ("encoder", "decoder"):
            handle.add(vae, attribute, torch.compile(getattr(vae, attribute), **settings))
            handle.info["compiled_units"].append(attribute)
    handle.apply()
    return handle


# ---------------------------------------------------------------------------------------------
# Phases and passes


class Bench:
    """One loaded VAE and what a phase needs: the seedvr2x functions, the meter, the source
    frames, the tiles of each phase (None: untiled) and where their scratch is."""

    def __init__(self, models, frames, target, meter, encode_tiles=None, decode_tiles=None,
                 scratch=None):
        from seedvr2x.runtime import job, model, shot, tilegrid, tiling

        self.model, self.shot, self.tiling, self.tilegrid = model, shot, tiling, tilegrid
        self.models = models
        self.vae = models.runner.vae
        self.frames = frames
        self.target = target
        self.output = job.output_size(target)
        self.padded = job.padded_size(target)
        self.meter = meter
        self.encode_tiles = encode_tiles
        self.decode_tiles = decode_tiles
        self.scratch = scratch  # a directory for the scratch's file; None: system memory
        self.allowance = RAM if scratch is None else 0
        self.decode_latent = None  # the latent every decode of a run takes

    def grid(self, tiles):
        """The tiles of a tiled phase over the padded frames: their count, and the largest tile's
        (height, width) in pixels; None for an untiled phase."""
        if tiles is None:
            return None
        layout = self.tilegrid.grid(self.tilegrid.latent_size(self.padded), tiles)
        biggest = max(layout, key=lambda tile: tile.rows * tile.columns)
        return {"size": tiles.size, "overlap": tiles.overlap, "tiles": len(layout),
                "largest_px": [biggest.rows * 8, biggest.columns * 8],
                "scratch": "system memory" if self.scratch is None else str(self.scratch)}

    def phase(self, name, count, body, keep):
        """Run body() as a phase over `count` frames; keep(result) returns what stays (on the
        CPU). The phase's record: wall time, GPU time between two events, peak, allocated before
        and after."""
        gc.collect()
        torch.cuda.synchronize()
        self.meter.begin()
        before = allocated()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start = time.perf_counter()
        begin.record()
        result = body()
        end.record()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        self.meter.roll()
        peak = self.meter.peak
        kept = keep(result)
        del result
        raw = allocated()
        gc.collect()
        record = {
            "phase": name, "frames": count, "seconds": round(seconds, 3),
            "gpu_seconds": round(begin.elapsed_time(end) / 1000, 3),
            "seconds_per_frame": round(seconds / count, 4),
            "allocated_before": before, "peak": peak, "peak_above": peak - before,
            "allocated_after_before_gc": raw, "allocated_after": allocated(),
            "slices": self.meter.slices,
        }
        tiles = self.meter.per_tile()
        if len(tiles) > 1:
            record["per_tile"] = tiles
        return kept, record

    def encode(self, count):
        """The latent of the clip's first `count` frames, on the CPU as shot.encode_shot gave
        it (tiled with encode_tiles)."""
        padded = self.shot.padded_length(count)
        read = reader(self.frames[:count])
        return self.phase(
            "encode", padded,
            lambda: self.shot.encode_shot(
                self.models, read, count, self.target, 0,
                tiles=self.encode_tiles, scratch=self.scratch, allowance=self.allowance,
            ),
            lambda latent: latent.cpu(),
        )

    def decode(self, latent, count):
        """The frames decoded from latent (on the CPU, as the pipeline's waits), each slice
        clamped to [0, 1] and moved to the CPU as it comes, as shot.decode_shot does: (count, H,
        W, 3) float32. Tiled with decode_tiles."""
        height, width = self.output
        padded = self.shot.padded_length(count)

        def body():
            if self.decode_tiles is None:
                slices = self.model.decode_stream(self.models, latent)
            else:
                store = self.tiling.Scratch(self.scratch, self.allowance)
                slices = self.tiling.decode_tiled(self.models, latent, self.decode_tiles, store)
            chunks = []
            written = 0
            with contextlib.closing(slices):
                for decoded in slices:
                    chunk = decoded.permute(1, 2, 3, 0)[: count - written, :height, :width]
                    if chunk.shape[0]:
                        chunk.clamp_(-1, 1).mul_(0.5).add_(0.5)
                        self.model.synchronize(self.models.device)
                        chunks.append(chunk.to("cpu", torch.float32))
                        written += chunk.shape[0]
                    del decoded, chunk
            return torch.cat(chunks)

        return self.phase("decode", padded, body, lambda frames: frames)


def describe(record):
    """One line per phase of a pass, for the log."""
    parts = []
    for name in ("encode", "decode"):
        phase = record["phases"].get(name)
        if phase:
            parts.append(
                f"{name} {phase['seconds']:.1f} s ({phase['seconds_per_frame']:.3f} s/frame)"
                f" peak +{gib(phase['peak_above'])} left {gib(phase['allocated_after'])} GiB"
            )
    return "; ".join(parts)


def run_pass(bench, tally, kind, phases, count, reference):
    """One pass, plain (A) or compiled (B): the phases asked for, its compilations, its bits
    against the reference (latent on the CPU, frames). Returns the record and (latent, frames)."""
    counts = compile_counts(tally)
    record = {"kind": kind, "frames": count, "phases": {}}
    latent = frames = None
    if "encode" in phases:
        latent, record["phases"]["encode"] = bench.encode(count)
        if bench.decode_latent is None:
            bench.decode_latent = latent
    if "decode" in phases:
        frames, record["phases"]["decode"] = bench.decode(bench.decode_latent, count)
    record["compiled"] = delta(compile_counts(tally), counts)
    # The metrics' count: counters["frames"] stays at 0 under fullgraph.
    record["compiling"] = record["compiled"]["compilations"] > 0
    if reference:
        compare = {}
        if latent is not None:
            compare["latent"] = compare_latents(latent, reference[0])
        if frames is not None and reference[1] is not None:
            compare["frames"] = compare_frames(frames, reference[1])
        record["against_first_plain"] = compare
    # SHA-256 of the pass's outputs, to compare bits across processes (a cold compile against
    # another, or a warm cache).
    record["sha256"] = {
        name: sha256_of(value)
        for name, value in (("latent", latent), ("frames", frames))
        if value is not None
    }
    record["device"] = device_memory()
    return record, (latent, frames)


def sha256_of(tensor):
    """SHA-256 of a tensor's values, in its dtype, C-contiguous on the CPU."""
    data = tensor.detach().to("cpu").contiguous()
    return hashlib.sha256(data.view(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def compare_latents(latent, reference):
    if latent.shape != reference.shape:
        return {"equal": False, "shapes": [list(latent.shape), list(reference.shape)]}
    if torch.equal(latent, reference):
        return {"equal": True}
    a, b = latent.float(), reference.float()
    difference = (a - b).abs()
    return {
        "equal": False,
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "relative_max": float(difference.max() / b.abs().max()),
        "values_differing": float((a != b).float().mean()),
    }


def compare_frames(frames, reference):
    if frames.shape != reference.shape:
        return {"equal": False, "shapes": [list(frames.shape), list(reference.shape)]}
    if torch.equal(frames, reference):
        return {"equal": True}
    difference = (frames - reference).abs()
    mse = float((difference.double() ** 2).mean())
    return {
        "equal": False,
        "max_abs_levels": round(float(difference.max()) * 255, 4),
        "mean_abs_levels": round(float(difference.mean()) * 255, 5),
        "psnr_db": round(10 * math.log10(1.0 / mse), 2) if mse > 0 else None,
        "values_differing": round(float((frames != reference).float().mean()), 5),
        "non_finite": int((~torch.isfinite(frames)).sum()),
    }


def leave_gpu(bench, tally, handle, count, weights, last, results, reset_call):
    """What a compiled VAE leaves on the GPU, and what the VAE's trip to the CPU costs."""
    vae, device = bench.vae, bench.models.device
    conv_class = type(vae.encoder.conv_in)
    out = {"weights_allocated": weights}
    results["leave_gpu"] = out
    handle.apply()

    def steps_after(name):
        """Allocated beyond the VAE's weights after a phase, and after each freeing step."""
        known = known_storages(vae)
        steps = [["the phase's end, references dropped", allocated() - weights]]
        listing = None
        if steps[0][1] > LEFT_THRESHOLD:
            listing = census(known, referrers=True)
        gc.collect()
        steps.append(["gc.collect()", allocated() - weights])
        torch.cuda.empty_cache()
        steps.append(["empty_cache()", allocated() - weights])
        for module in vae.modules():
            if isinstance(module, conv_class):
                module.memory = None
        steps.append(["every conv's memory cleared", allocated() - weights])
        if steps[-1][1] > LEFT_THRESHOLD:
            listing = census(known, referrers=True)
        out[f"after_{name}"] = {"steps": steps, "tensors": listing, "device": device_memory()}
        log(f"leave-gpu: after the compiled {name}: "
            + ", ".join(f"{label}: {gib(value)} GiB" for label, value in steps))

    latent, _ = bench.encode(count)
    steps_after("encode")
    bench.decode_latent = latent
    frames, _ = bench.decode(latent, count)
    steps_after("decode")

    # The trip: parameters keep their identity, and come back to the same device and dtype.
    before = [(id(p), p.device, p.dtype, tuple(p.shape)) for p in vae.parameters()]
    counts = compile_counts(tally)
    start = time.perf_counter()
    vae.to("cpu")
    torch.cuda.synchronize()
    to_cpu_seconds = time.perf_counter() - start
    gc.collect()
    away_before_empty = allocated()
    torch.cuda.empty_cache()
    out["away"] = {
        "to_cpu_seconds": round(to_cpu_seconds, 3),
        "allocated_before_empty_cache": away_before_empty,
        "allocated": allocated(),
        "device": device_memory(),
        "tensors": census(set(), referrers=True),
    }
    log(f"leave-gpu: VAE on the CPU: {gib(allocated())} GiB allocated on the GPU,"
        f" device used {gib(out['away']['device']['used'])} GiB")
    start = time.perf_counter()
    vae.to(device)
    torch.cuda.synchronize()
    out["back"] = {
        "to_gpu_seconds": round(time.perf_counter() - start, 3),
        "parameters_unchanged":
            before == [(id(p), p.device, p.dtype, tuple(p.shape)) for p in vae.parameters()],
        "allocated": allocated(),
    }
    record, (latent2, frames2) = run_pass(
        bench, tally, "B", ("encode", "decode"), count, (latent, frames)
    )
    record["compiled_since_the_trip"] = delta(compile_counts(tally), counts)
    out["call_after_trip"] = record
    if last is not None:
        out["call_after_trip"]["against_last_compiled"] = {
            "latent": compare_latents(latent2, last[0]),
            "frames": compare_frames(frames2, last[1]),
        }
    log(f"leave-gpu: the call after the trip: {describe(record)};"
        f" {record['compiled_since_the_trip']['compilations']} compilations; bits against the call"
        f" before: {record['against_first_plain']}")
    del latent, frames, latent2, frames2

    # torch._dynamo.reset(): the compiled code dropped; what it frees, what the next call costs.
    out["compile_before_reset"] = compile_summary(tally)
    before_reset = allocated()
    used_before = device_memory()["used"]
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()
    out["reset"] = {
        "allocated_before": before_reset, "allocated_after": allocated(),
        "device_used_before": used_before, "device_used_after": device_memory()["used"],
    }
    log(f"leave-gpu: torch._dynamo.reset(): allocated {gib(before_reset)} -> {gib(allocated())}"
        f" GiB, device used {gib(used_before)} -> {gib(out['reset']['device_used_after'])} GiB")
    if reset_call:
        record, _ = run_pass(bench, tally, "B", ("encode", "decode"), count, None)
        out["call_after_reset"] = record
        log(f"leave-gpu: the call after the reset: {describe(record)}")
    handle.restore()


def parse_size(text, what):
    try:
        height, width = (int(v) for v in text.lower().split("x"))
    except ValueError:
        sys.exit(f"compile_vae: {what} is HEIGHTxWIDTH, not {text!r}")
    return height, width


def versions():
    out = {"torch": torch.__version__, "cuda": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version(), "python": sys.version.split()[0]}
    try:
        import triton

        out["triton"] = triton.__version__
    except ImportError:
        out["triton"] = None
    return out


def run(args):
    if torch is None:
        sys.exit("compile_vae: run needs torch (seedvr2x's venv)")
    from seedvr2x.runtime import plan as plan_module

    target = parse_size(args.size, "--size")
    shots = [int(v) for v in args.shots.split(",")] if args.shots else []
    phases = tuple(args.phases.split(","))
    passes = args.passes.upper()
    if not passes.startswith("A") or set(passes) - {"A", "B"} or set(phases) - {"encode", "decode"}:
        sys.exit("compile_vae: --passes is A's and B's starting with A; --phases encode,decode")
    if args.frames % 4 != 1 or any(n % 4 != 1 for n in shots):
        sys.exit("compile_vae: --frames and --shots are 4n + 1")
    tiles = {
        name: plan_module.Tiles(size, args.overlap) if size else None
        for name, size in (("encode", args.encode_tile), ("decode", args.decode_tile))
    }

    torch._dynamo.config.recompile_limit = NUMZ_RECOMPILE_LIMIT
    inductor = {}
    for item in args.inductor:
        name, _, value = item.partition("=")
        inductor[name] = json.loads(value)
        setattr(torch._inductor.config, name, inductor[name])
    if args.keep_deterministic:
        # Inductor's deterministic mode, kept on through dynamo's resets of it, so that two
        # processes compiling cold give the same kernels' configs, hence bits
        # (compile_dit.keep_deterministic, which says where torch resets it).
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from compile_dit import keep_deterministic

        keep_deterministic(torch, "inductor")
    extra_logs = [v for v in args.torch_logs.split(",") if v]
    tally = install_tally(extra_logs, args.code_log)

    results = {
        "script": "compile_vae.py",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "argv": sys.argv,
        "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                      if k != "command"},
        "versions": {**versions(), "allocator": os.environ.get("PYTORCH_CUDA_ALLOC_CONF")},
        "caches": {name: os.environ.get(name) for name in
                   ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "TORCH_COMPILE_DEBUG_DIR")},
        "settings": {
            **NUMZ_SETTINGS, "mode": args.mode, "fullgraph": args.fullgraph,
            "recompile_limit": NUMZ_RECOMPILE_LIMIT, "inductor": inductor,
            "inline_inbuilt_nn_modules": torch._dynamo.config.inline_inbuilt_nn_modules,
        },
        "variant": args.variant, "fullgraph": args.fullgraph,
        "memory": {}, "passes": [], "shots": [],
    }

    def save():
        Path(args.out).write_text(json.dumps(results, indent=1) + "\n")

    count = args.frames
    wanted = max([count, *shots])
    if args.clip:
        frames = read_clip(args.clip, wanted)
        results["input"] = "clip"
    else:
        frames = random_frames(wanted, *parse_size(args.source_size, "--source-size"))
        results["input"] = "random frames"
    results["source"] = {"width": frames.shape[2], "height": frames.shape[1],
                         "read": frames.shape[0]}
    results["target"] = list(target)

    device = torch.device("cuda", 0)
    torch.cuda.init()
    torch.zeros(1, device=device)
    results["versions"]["gpu"] = torch.cuda.get_device_name(0)
    results["memory"]["at_start"] = device_memory()
    log(f"CUDA context: free {gib(results['memory']['at_start']['free'])} of"
        f" {gib(results['memory']['at_start']['total'])} GiB; allocator"
        f" {results['versions']['allocator']}")

    from seedvr2x.runtime import job, memory, model
    from seedvr2x.vendor.models.video_vae_v3.modules.types import MemoryState

    results["versions"]["driver_spills"] = memory.spills()
    started = time.perf_counter()
    models = model.load_models(args.model_dir / args.dit, args.model_dir / args.vae, device)
    torch.cuda.synchronize()
    results["memory"]["after_load"] = device_memory()
    results["attention"] = models.attention
    # The DiT off the GPU, as milestone 3 has it in the VAE's phases (load_models loads both, and
    # has no VAE-only load): what stays is the VAE's weights and the text embeddings.
    models.runner.dit = None
    gc.collect()
    torch.cuda.empty_cache()
    results["memory"]["after_dit_drop"] = device_memory()
    weights = allocated()
    vae = models.runner.vae
    vae_bytes = sum(t.numel() * t.element_size() for t in [*vae.parameters(), *vae.buffers()])
    results["memory"]["weights_allocated"] = weights
    results["memory"]["vae_bytes"] = vae_bytes
    results["memory"]["other_allocated"] = weights - vae_bytes
    results["padded"] = list(job.padded_size(target))
    log(f"models loaded in {time.perf_counter() - started:.1f} s; DiT dropped: {gib(weights)} GiB"
        f" allocated (the VAE's weights {gib(vae_bytes)}, the rest {weights - vae_bytes} bytes),"
        f" free {gib(device_memory()['free'])} GiB; target {target}, padded {results['padded']},"
        f" {count} frames")

    meter = Meter()
    meter.wrap(vae, "_encode", MemoryState)
    meter.wrap(vae, "_decode", MemoryState)
    scratch = args.scratch
    if scratch is not None:
        scratch.mkdir(parents=True, exist_ok=True)
    bench = Bench(models, frames, target, meter, tiles["encode"], tiles["decode"], scratch)
    results["tiles"] = {name: bench.grid(value) for name, value in tiles.items()}
    handle = install_vae(vae, args.variant, mode=args.mode, fullgraph=args.fullgraph)
    handle.restore()
    results["setup"] = handle.info
    log(f"variant {args.variant}{' fullgraph' if args.fullgraph else ''}:"
        f" {len(handle.info['compiled_units'])} compiled units,"
        f" {handle.info['causal_convs']} causal convolutions; tiles {results['tiles']}")

    if "encode" not in phases:
        # The decode's latent, from a plain encode apart from the passes (not recorded as one).
        bench.decode_latent, prep = bench.encode(count)
        results["prep_encode"] = prep
    reference = None
    last_compiled = None
    for index, kind in enumerate(passes):
        if kind == "B":
            handle.apply()
        try:
            record, outputs = run_pass(bench, tally, kind, phases, count, reference)
        finally:
            if kind == "B":
                handle.restore()
        record["index"] = index
        results["passes"].append(record)
        if reference is None:
            reference = outputs
        elif kind == "B" and outputs[1] is not None:
            last_compiled = outputs
        log(f"pass {index} {kind}{' (compiling)' if record['compiling'] else ''}:"
            f" {describe(record)}; {record['compiled']['compilations']} compilations in"
            f" {record['compiled']['compile_seconds']} s; {record.get('against_first_plain', '')}")
        save()

    for length in shots:
        entry = {"frames": length, "padded": bench.shot.padded_length(length)}
        bench.decode_latent = None
        plain_outputs = None
        for kind in "AB":
            if kind == "B":
                handle.apply()
            try:
                record, outputs = run_pass(
                    bench, tally, kind, ("encode", "decode"), length, plain_outputs
                )
            finally:
                if kind == "B":
                    handle.restore()
            if kind == "A":
                plain_outputs = outputs
                entry["plain"] = record
            else:
                entry["compiled"] = record
        del plain_outputs
        entry["counts_after"] = compile_counts(tally)
        results["shots"].append(entry)
        log(f"shot of {length} frames: plain {describe(entry['plain'])}; compiled"
            f" {describe(entry['compiled'])}; {entry['compiled']['compiled']['compilations']}"
            f" compilations, {entry['compiled']['compiled']['unique_graphs']} graphs;"
            f" {entry['compiled'].get('against_first_plain')}")
        save()

    results["compile"] = compile_summary(tally)
    save()
    if args.leave_gpu:
        leave_gpu(bench, tally, handle, count, weights, last_compiled, results, args.reset_call)
    results["memory"]["at_end"] = device_memory()
    results["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save()
    counts = results["compile"]["counts"]
    log(f"done: {counts['compilations']} compilations, {counts['unique_graphs']} graphs,"
        f" {counts['resume_functions']} resume functions, {counts['graph_breaks_logged']} graph"
        f" breaks and {counts['recompiles_logged']} recompiles logged,"
        f" {counts['compile_seconds']} s compiling; results in {args.out}")


# ---------------------------------------------------------------------------------------------
# Self-test (CPU only)


def check_arithmetic():
    """The patch's float32 arithmetic against torch's: the integer rounding, the float rounding
    against numpy's cast, and the splitting decision and count against the vendored expressions
    (a float32 tensor compared with and divided by a Python float), at random sizes and limits
    and at limits one float32 step either side of a size."""
    import random

    rng = random.Random(0)
    failures = []
    for count in [rng.randrange(1, 2**40) for _ in range(20000)] + [
        2**24 + k for k in range(4)] + [2**29 - 17, 2**29 - 16, 2**29 + 32, 2**29 + 33, 2**33 - 1]:
        if float32(count) != float(torch.tensor(count).to(torch.float32)):
            failures.append(("float32(int)", count))
    values = np.exp(np.random.default_rng(0).uniform(math.log(1e-6), math.log(1e4), 200000))
    rounded = np.array([float32_of_float(float(v)) for v in values[:20000]])
    if not np.array_equal(rounded, values[:20000].astype(np.float32).astype(np.float64)):
        failures.append(("float32_of_float", "20000 random values"))
    decided = 0
    for _ in range(30000):
        dims = [rng.randrange(1, 4) if i < 2 else rng.randrange(1, 2200) for i in range(5)]
        element = rng.choice([2, 4])
        size = torch.tensor(dims).prod() * element / 1024**3
        occupy = float32(math.prod(dims) * element) / 1024**3
        if float(size) != occupy:
            failures.append(("occupy", dims, element))
        limit = rng.choice([0.5, 0.1, 0.01, 2.5, rng.uniform(0.001, 8)])
        if rng.random() < 0.3:
            limit = float(torch.nextafter(size.reshape(1), torch.tensor([rng.choice([0.0, 99.0])]))[0])
        mine = float32_of_float(limit)
        if bool(size < limit) != (occupy < mine):
            failures.append(("compare", dims, element, limit))
        if not size < limit and math.ceil(size / limit) != math.ceil(
            float32_of_float(occupy / mine)
        ):
            failures.append(("ceil", dims, element, limit))
        decided += 1
    return {"checked": decided, "failures": failures[:10], "failed": len(failures)}


def causal_conv_cases():
    """(channels, kernel, padding, spatial size, limit in MB) of the convolution cases."""
    for (cin, cout), kernel, (h, w), limit_mb in itertools.product(
        [(4, 8), (8, 4)], [3, 1, (1, 3, 3)], [(33, 47), (64, 64), (17, 130)],
        [0.01, 0.03, 0.11, 0.5, 1000.0, math.inf],
    ):
        pad = 1 if kernel == 3 else (0 if kernel == 1 else (0, 1, 1))
        yield cin, cout, kernel, pad, h, w, limit_mb


def check_convolutions():
    """int_limit_conv's method against the vendored one, bit for bit, on real
    InflatedCausalConv3d modules with random weights, and the sequence of calls to whole_output
    (the paths taken and the shapes of the outputs allocated) the same: a first slice of 5 frames INITIALIZING,
    two of 4 ACTIVE, one DISABLED, with the output threshold ONE_OUTPUT_FROM at values putting
    some outputs in one piece and some concatenated, in float32 (and bfloat16 at the two
    extremes). Splits along H, then W, prev caches included, are what the small limits make."""
    from seedvr2x.vendor.models.video_vae_v3.modules import (
        causal_inflation_lib as lib,
    )
    from seedvr2x.vendor.models.video_vae_v3.modules import context_parallel_lib
    from seedvr2x.vendor.models.video_vae_v3.modules.types import MemoryState

    patch = int_limit_conv(context_parallel_lib.get_cache_size)
    conv_class = lib.InflatedCausalConv3d
    default = lib.ONE_OUTPUT_FROM
    stats = Counter()
    original_whole = conv_class.whole_output

    trace = []

    def counted(self, like, padded, head=0, whole=None):
        out = original_whole(self, like, padded, head, whole)
        trace.append((tuple(like.shape), tuple(int(v) for v in padded), head, whole,
                      None if out is None else tuple(out.shape)))
        stats["whole_output calls"] += 1
        stats["  of them in one piece" if out is not None else "  of them concatenated"] += 1
        if whole is not None:
            stats["  of them a piece cut again"] += 1
        return out

    conv_class.whole_output = counted
    bad = []
    cases = 0
    refused = 0

    def outputs(conv, cin, dtype):
        conv.memory = None
        trace.clear()
        try:
            out = []
            for seed, frames, state in ((1, 5, MemoryState.INITIALIZING),
                                        (2, 4, MemoryState.ACTIVE), (3, 4, MemoryState.ACTIVE),
                                        (9, 5, MemoryState.DISABLED)):
                data = torch.randn(
                    1, cin, frames, *conv.shape_hw, generator=torch.Generator().manual_seed(seed)
                ).to(dtype)
                out.append(conv(data, memory_state=state))
            return out
        except AssertionError:
            return None
        finally:
            conv.memory = None

    try:
        with torch.no_grad():
            for dtype, thresholds in (
                (torch.float32, [0, 150_000, 250_000, 600_000, default]),
                (torch.bfloat16, [0, default]),
            ):
                for cin, cout, kernel, pad, h, w, limit_mb in causal_conv_cases():
                    torch.manual_seed(cin * 31 + cout)
                    conv = conv_class(cin, cout, kernel, padding=pad, inflation_mode="none")
                    conv = conv.to(dtype).eval()
                    conv.shape_hw = (h, w)
                    conv.set_memory_limit(limit_mb / 1024)
                    for threshold in thresholds:
                        lib.ONE_OUTPUT_FROM = threshold
                        original = outputs(conv, cin, dtype)
                        original_trace = list(trace)
                        conv.memory_limit_conv = types.MethodType(patch, conv)
                        patched = outputs(conv, cin, dtype)
                        del conv.memory_limit_conv
                        cases += 1
                        # The paths leave no trace in the bits (the pieces written into one
                        # output or concatenated are the same values): the calls to whole_output,
                        # with their arguments and what they chose, are compared.
                        if original_trace != trace:
                            bad.append(("different path", str(dtype), cin, cout, kernel, h, w,
                                        limit_mb, threshold))
                        if original is None or patched is None:
                            refused += 1
                            if (original is None) != (patched is None):
                                bad.append(("one refuses", str(dtype), cin, cout, kernel, h, w,
                                            limit_mb, threshold))
                        elif not all(torch.equal(a, b) for a, b in zip(original, patched)):
                            bad.append(("different", str(dtype), cin, cout, kernel, h, w,
                                        limit_mb, threshold))
    finally:
        lib.ONE_OUTPUT_FROM = default
        conv_class.whole_output = original_whole
    return {"cases": cases, "different": len(bad), "refused_by_both": refused,
            "examples": bad[:10], "paths": dict(stats)}


def small_vae():
    """The VAE as load_models builds it, random weights, float32, on the CPU, its limits low
    enough for a small input to be split as a 1080p one is at the shipped 0.5 GiB."""
    from omegaconf import OmegaConf
    from seedvr2x.runtime import model
    from seedvr2x.vendor.common.config import create_object, load_config

    torch.manual_seed(0)
    config = load_config(str(model.VENDOR / "configs_7b" / "main.yaml"))
    OmegaConf.set_readonly(config, False)
    vae_config = load_config(str(model.VENDOR / "models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml"))
    vae_config.spatial_downsample_factor = 8
    vae_config.temporal_downsample_factor = 4
    config.vae.model = OmegaConf.merge(config.vae.model, vae_config)
    vae = create_object(config.vae.model)
    vae.requires_grad_(False).eval()
    vae.set_causal_slicing(**config.vae.slicing)
    vae.set_memory_limit(conv_max_mem=0.001, norm_max_mem=0.001)
    return vae


def check_dynamo(variants):
    """Each variant (name, fullgraph) compiled with backend eager on a small VAE: an encode of
    5, 4 and 4 frames and a decode of 2, 1 and 1 latents, twice, with ONE_OUTPUT_FROM lowered so
    that part of the outputs are written in one piece; the graph breaks, graphs and resume
    functions counted, and the outputs compared bit for bit with plain code's. Under fullgraph
    a break is an error."""
    from seedvr2x.vendor.models.video_vae_v3.modules import causal_inflation_lib as lib
    from seedvr2x.vendor.models.video_vae_v3.modules.types import MemoryState

    vae = small_vae()
    x = torch.randn(1, 3, 13, 48, 64)
    z = torch.randn(1, 16, 4, 6, 8)

    def clear():
        for module in vae.modules():
            if isinstance(module, lib.InflatedCausalConv3d):
                module.memory = None

    def round_trip():
        clear()
        out = [vae._encode(x[:, :, :5], memory_state=MemoryState.INITIALIZING),
               vae._encode(x[:, :, 5:9], memory_state=MemoryState.ACTIVE),
               vae._encode(x[:, :, 9:13], memory_state=MemoryState.ACTIVE)]
        clear()
        out += [vae._decode(z[:, :, :2], memory_state=MemoryState.INITIALIZING),
                vae._decode(z[:, :, 2:3], memory_state=MemoryState.ACTIVE),
                vae._decode(z[:, :, 3:4], memory_state=MemoryState.ACTIVE)]
        clear()
        return out

    default = lib.ONE_OUTPUT_FROM
    lib.ONE_OUTPUT_FROM = 2**20
    tally = install_tally([], None)
    report = []
    try:
        with torch.no_grad():
            plain = round_trip()
            for name, fullgraph in variants:
                torch._dynamo.reset()
                before = compile_counts(tally)
                row = {"variant": name, "fullgraph": fullgraph}
                handle = install_vae(vae, name, fullgraph=fullgraph, backend="eager")
                try:
                    for attempt in range(2):
                        out = round_trip()
                        row[f"call {attempt} bits equal"] = all(
                            torch.equal(a, b) for a, b in zip(plain, out))
                except Exception as error:  # noqa: BLE001 - a fullgraph break is an error
                    row["error"] = " ".join(str(error).split())[:300]
                finally:
                    handle.restore()
                row["counts"] = delta(compile_counts(tally), before)
                report.append(row)
    finally:
        lib.ONE_OUTPUT_FROM = default
    return report


def selftest(args):
    if torch is None:
        sys.exit("compile_vae: selftest needs torch (seedvr2x's venv)")
    if torch.cuda.is_available():
        log("WARNING: CUDA is visible; the selftest uses the CPU only (CUDA_VISIBLE_DEVICES=)")
    failed = False
    log("arithmetic ...")
    arithmetic = check_arithmetic()
    log(f"arithmetic: {json.dumps(arithmetic)}")
    failed |= bool(arithmetic["failed"])
    log("causal convolutions, vendored against int ...")
    convolutions = check_convolutions()
    log(f"convolutions: {json.dumps(convolutions)}")
    paths = convolutions["paths"]
    failed |= bool(convolutions["different"])
    failed |= not (paths.get("  of them in one piece") and paths.get("  of them concatenated")
                   and paths.get("  of them a piece cut again"))
    log("dynamo (backend eager) on a small VAE ...")
    report = check_dynamo([("int", True), ("int", False), ("numz", False)])
    for row in report:
        log(f"dynamo: {json.dumps(row)}")
    for row in report:
        if row["variant"] == "int":
            counts = row["counts"]
            failed |= (
                "error" in row or counts["graph_breaks_logged"] != 0
                or counts["resume_functions"] != 0
                or not (row.get("call 0 bits equal") and row.get("call 1 bits equal"))
            )
    log("selftest " + ("FAILED" if failed else "passed"))
    sys.exit(1 if failed else 0)


# ---------------------------------------------------------------------------------------------
# Table


def spread(values):
    if not values:
        return "–"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{statistics.median(values):.3f} [{min(values):.3f}–{max(values):.3f}]"


def phase_summary(results, name):
    """A phase over a run's passes: peaks above what was allocated before, what is left, s/frame,
    the ratio of each steady compiled pass to its neighbouring plain passes."""
    passes = [p for p in results["passes"] if name in p["phases"]]
    plain = [p for p in passes if p["kind"] == "A"]
    steady = [p for p in passes if p["kind"] == "B" and not p["compiling"]]
    compiling = [p for p in passes if p["kind"] == "B" and p["compiling"]]

    def above(group):
        return max((p["phases"][name]["peak_above"] for p in group), default=None)

    def left(group):
        return max((p["phases"][name]["allocated_after"] - p["phases"][name]["allocated_before"]
                    for p in group), default=None)

    def rate(group):
        return [p["phases"][name]["seconds_per_frame"] for p in group]

    def tile_peak(group):
        peaks = [t["peak"] - t["allocated_before"] for p in group
                 for t in p["phases"][name].get("per_tile", [])]
        return max(peaks, default=None)

    ratios = []
    for p in steady:
        near = [q["phases"][name]["seconds_per_frame"] for q in plain
                if abs(q["index"] - p["index"]) == 1]
        if near:
            ratios.append(p["phases"][name]["seconds_per_frame"] / statistics.mean(near))
    return {
        "plain_above": above(plain), "compiled_above": above(steady),
        "compiling_above": above(compiling), "plain_left": left(plain),
        "compiled_left": left(steady or compiling),
        "plain_rate": rate(plain), "compiled_rate": rate(steady), "ratios": ratios,
        "compiling_seconds": [p["phases"][name]["seconds"] for p in compiling],
        "plain_tile": tile_peak(plain), "compiled_tile": tile_peak(steady or compiling),
    }


def amount(value):
    return "–" if value is None else f"{value / GIB:.2f}"


def bits(results):
    """The compiled passes' bits against the first plain pass, worst case."""
    latents, frames = [], []
    for p in results["passes"]:
        compare = p.get("against_first_plain") or {}
        if p["kind"] == "B":
            if "latent" in compare:
                latents.append(compare["latent"])
            if "frames" in compare:
                frames.append(compare["frames"])

    def one(items, key, unit):
        if not items:
            return "–"
        if all(c["equal"] for c in items):
            return "equal"
        return f"max {max(c.get(key, 0) for c in items if not c['equal']):.3g}{unit}"

    psnr = [c["psnr_db"] for c in frames if not c["equal"] and c.get("psnr_db")]
    text = f"latent {one(latents, 'max_abs', '')}, frames {one(frames, 'max_abs_levels', ' lv')}"
    if psnr:
        mean = max(c.get("mean_abs_levels", 0) for c in frames if not c["equal"])
        text += f" (mean {mean:.3g} lv, PSNR {min(psnr):.1f} dB)"
    return text


def tiles_label(results):
    labels = []
    for name in ("encode", "decode"):
        grid = (results.get("tiles") or {}).get(name)
        labels.append(f"{grid['size']} ({grid['tiles']})" if grid else "–")
    return " / ".join(labels)


def table(args):
    print("| Run | Variant | Target | Tiles enc / dec (count) | Frames |"
          " Enc peak above, plain / comp. / compiling (GiB) | Enc left |"
          " Dec peak above, plain / comp. / compiling | Dec left |"
          " Enc s/frame plain / comp. (ratio) | Dec s/frame plain / comp. (ratio) |"
          " Compilations / graphs / resumes / breaks / recompiles | Compile s |"
          " Bits (comp. against plain) |")
    print("|" + "---|" * 14)
    for path in args.results:
        results = json.loads(Path(path).read_text())
        variant = results["variant"] + (" fullgraph" if results.get("fullgraph") else "")
        cells = [Path(path).stem, variant, "x".join(str(v) for v in results["target"]),
                 tiles_label(results),
                 str(results["passes"][0]["frames"]) if results["passes"] else "–"]
        rates = []
        summaries = {name: phase_summary(results, name) for name in ("encode", "decode")}
        for name, s in summaries.items():
            cells.append(f"{amount(s['plain_above'])} / {amount(s['compiled_above'])}"
                         f" / {amount(s['compiling_above'])}")
            left = s["compiled_left"]
            cells.append("–" if left is None else f"{left / GIB:.3f}")
            plain = statistics.median(s["plain_rate"]) if s["plain_rate"] else float("nan")
            comp = statistics.median(s["compiled_rate"]) if s["compiled_rate"] else float("nan")
            rates.append(f"{plain:.3f} / {comp:.3f} ({spread(s['ratios'])})")
        counts = results.get("compile", {}).get("counts")
        cells += rates
        if counts:
            cells.append(f"{counts['compilations']} / {counts['unique_graphs']} /"
                         f" {counts['resume_functions']} / {counts['graph_breaks_logged']} /"
                         f" {counts['recompiles_logged']}")
            cells.append(f"{counts['compile_seconds']:.0f}")
        else:
            cells += ["–", "–"]
        cells.append(bits(results))
        print("| " + " | ".join(cells) + " |")
        memory = results["memory"]
        notes = [
            "weights {:.3f} GiB allocated before a pass (the VAE's {:.3f})".format(
                memory["weights_allocated"] / GIB, memory.get("vae_bytes", 0) / GIB
            )
        ]
        for name, s in summaries.items():
            if s["plain_tile"] is not None:
                notes.append(f"{name} largest tile's peak above, plain {amount(s['plain_tile'])}"
                             f" compiled {amount(s['compiled_tile'])} GiB")
            if s["compiling_seconds"]:
                notes.append(f"{name} compiling pass {s['compiling_seconds']} s")
        print(f"\n{Path(path).stem}: " + "; ".join(notes) + "\n")
        if args.slices:
            for p in results["passes"]:
                for name, phase in p["phases"].items():
                    line = ", ".join(
                        f"{s['state'][:4]} t{s['tile']} {s['input'][2]}:"
                        f" +{(s['peak'] - s['allocated_before']) / GIB:.2f} ({s['seconds']:.2f} s)"
                        for s in phase["slices"])
                    print(f"  pass {p['index']} {p['kind']}{'*' if p['compiling'] else ''}"
                          f" {name}: {line}")
            print()
        for entry in results.get("shots", []):
            c = entry["compiled"]
            print(f"  shot of {entry['frames']} frames: {c['compiled']['compilations']}"
                  f" compilations ({c['compiled']['unique_graphs']} graphs,"
                  f" {c['compiled']['compile_seconds']} s); "
                  + "; ".join(
                      f"{name} peak above plain {amount(entry['plain']['phases'][name]['peak_above'])}"
                      f" compiled {amount(c['phases'][name]['peak_above'])} GiB"
                      for name in ("encode", "decode"))
                  + f"; {c.get('against_first_plain')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    r = commands.add_parser("run", help="measure one variant against plain code")
    r.add_argument("--model-dir", type=Path, required=True)
    r.add_argument("--dit", default=DIT)
    r.add_argument("--vae", default=VAE)
    r.add_argument("--clip", type=Path, help="the input clip; without one, random frames")
    r.add_argument("--source-size", default="540x960",
                   help="without --clip: the random frames' HEIGHTxWIDTH")
    r.add_argument("--size", default="1080x1920", help="the target, HEIGHTxWIDTH")
    r.add_argument("--frames", type=int, default=21, help="frames per pass (4n + 1)")
    r.add_argument("--variant", choices=VARIANTS, default="numz")
    r.add_argument("--fullgraph", action="store_true", help="fullgraph=True for a compiled variant")
    r.add_argument("--passes", default="ABAB")
    r.add_argument("--phases", default="encode,decode")
    r.add_argument("--encode-tile", type=int, default=0, metavar="PX",
                   help="tiling.encode_tiled with tiles of PX pixels (0: untiled)")
    r.add_argument("--decode-tile", type=int, default=0, metavar="PX",
                   help="tiling.decode_tiled with tiles of PX pixels (0: untiled)")
    r.add_argument("--overlap", type=int, default=None, metavar="PX",
                   help="the tiles' overlap (default: runtime/plan.py's OVERLAP)")
    r.add_argument("--scratch", type=Path, metavar="DIR",
                   help="the tiled passes' scratch in a file there (default: system memory)")
    r.add_argument("--mode", default="default")
    r.add_argument("--keep-deterministic", action="store_true",
                   help="inductor's deterministic mode on, kept through dynamo's resets")
    r.add_argument("--inductor", action="append", default=[], metavar="NAME=VALUE",
                   help="a torch._inductor.config setting (JSON value)")
    r.add_argument("--torch-logs", default="", help="more torch log artefacts, say output_code")
    r.add_argument("--code-log", help="the file those artefacts are written to")
    r.add_argument("--shots", default="", help="shot lengths to run in turn after the passes")
    r.add_argument("--leave-gpu", action="store_true")
    r.add_argument("--reset-call", action="store_true",
                   help="with --leave-gpu: one more compiled pass after torch._dynamo.reset()")
    r.add_argument("--out", required=True)
    t = commands.add_parser("table", help="a Markdown summary of results")
    t.add_argument("results", nargs="+")
    t.add_argument("--slices", action="store_true", help="each pass's slices too")
    commands.add_parser("selftest", help="CPU only: the int patch against the vendored code")
    args = parser.parse_args()
    if args.command == "run":
        if args.overlap is None:
            from seedvr2x.runtime.plan import OVERLAP

            args.overlap = OVERLAP
        run(args)
    elif args.command == "selftest":
        selftest(args)
    else:
        table(args)


if __name__ == "__main__":
    main()

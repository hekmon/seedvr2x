#!/usr/bin/env python3
"""Window layouts and torch.compile strategies for seedvr2x: arithmetic on real cut lists (no GPU).

seedvr2x runs its DiT on windows of each shot's latents (seedvr2x/DESIGN.md, Pipeline step 2).
torch.compile compiles the DiT once per window LENGTH, in latents: a new length is a new
compilation, tens of seconds, after which every window of that length runs faster. The first pass
knows every shot before the DiT starts, so a job can tell beforehand which lengths it will meet,
how many windows run at each, and which compilations pay. This script counts that on real cut
lists, for three window layouts, and costs four compiling strategies with a cost model whose
every parameter is a flag or a key of a JSON file, so that the tables are made again when a
parameter is measured.

A shot of n frames is padded to 4k + 1 frames and makes (padded - 1) / 4 + 1 latents
(runtime/shot.py: padded_length, shot_layout). A shot no longer than the cap, the most latents a
window may hold, is ONE window of its own length, in every layout. A longer shot is cut into
ceil((latents - SHARED) / (cap - SHARED)) windows, the same count in the three layouts:

  balanced         seedvr2x's own (runtime/shot.py, window_layout, copied here): the windows'
                   lengths balanced, differing by at most 1, consecutive windows sharing SHARED
                   latents (2). The lengths depend on the shot's.
  full-short-last  windows of exactly `cap` latents every cap - SHARED, the last one holding what
                   is left, SHARED + 1 latents (1 new latent) to cap. The same latents computed as
                   balanced (every join shares SHARED), one length but for each long shot's last
                   window.
  slid-back        every window of a long shot exactly `cap` latents, the last one slid back to
                   end with the shot: one length for every long shot, and cap minus the last
                   window's length of full-short-last computed twice at the last join, 0 to
                   cap - SHARED - 1 latents per long shot. (When that last window would hold
                   fewer than 2 x SHARED latents, a latent falls in three windows.)

Lengths and compute only: what a layout does to the joins is another study.

The cost model (defaults: placeholders from numz's figures, research/docs/vram.md, torch.compile,
until they are measured on seedvr2x; the dynamic ones are nobody's measurement):

  plain time of a window of L latents   t_token x tokens_per_latent x L / gpu_speedup
                                        (0.30 ms a token on the throttled card; 8280 tokens a
                                        latent = 1080p padded to 1104x1920, 69 x 120)
  compiled                              plain x (1 - gain(L)); gain 0.29, or a table per L,
                                        interpolated between its lengths, held beyond them
  a compilation of a new length         cold(L) = a + b x L seconds (30) the first time the job
                                        compiles it; warm(L) (10) when an earlier process of the
                                        job compiled it (inductor's on-disk cache); compile_first
                                        more seconds for the first compilation of each process (0)
  recompile_limit                       PyTorch stops compiling after N lengths in one process
                                        (torch._dynamo.config.recompile_limit, 8 unless raised)
                                        and runs the others plain; 0 here: no limit
  dynamic shapes                        one compilation, dynamic_compile seconds (60 cold, 20
                                        warm), serving every length with gain dynamic_gain (0.20)
  processes                             the job run in N processes (resumed night after night),
                                        each an equal share of its windows in order: a process
                                        compiles again, warm, the lengths it meets
  gpu_speedup                           a healthier card runs the forwards that much faster and
                                        compiles no faster (compiling is CPU work): it divides
                                        the forward times only
  dit_share                             the DiT's share of a whole job's time (0.20), for the
                                        gain as % of the job; give the share at the speed costed

Strategies: `every` compiles every length a process meets, in the order met, up to the limit;
`pay` compiles only the lengths whose windows save more than their compilation costs, process by
process, the best ones first under a limit; `dynamic` compiles once for every length; `none`.
Every net gain is against the same baseline, the plain DiT time of the BALANCED layout: slid-back
without compilation shows its extra compute as a loss.

Subcommands:
  cuts    LABEL=FILE [...] --out CUTS.json [--threshold 0.3] [--key single]
          [--range LABEL=START:END ...] [--fps LABEL=N/D ...]
          the cut lists: per source, the frames of each shot, cut by seedvr2x's rule
          (runtime/cuts.py, reimplemented here) from per-frame probability files (.npz, its
          array --key; or .npy). --range keeps frames START to END - 1 of a file holding more;
          --fps is only for the durations printed.
  tables  CUTS.json [...] [--caps 5,13,19,27,76] [--model MODEL.json] [model flags]
          [--top N] [--no-details]
          Markdown on stdout: the sources, the break-even windows per length, then per cap a
          summary across sources, the layouts' lengths and the strategies' times.
          MODEL.json holds any of the model's keys (the flags' names, underscored); a flag wins.
  check   LABEL=FILE [...] [--threshold 0.3] [--key single] [--range ...] [--caps ...]
          the copies against seedvr2x's own functions on the same data: the cuts against
          runtime/cuts.py's detected, padded_length and the balanced layout against
          runtime/shot.py's, on every shot and cap and on every length up to the longest shot.
          Needs seedvr2x importable: run it with the python of seedvr2x's own environment.

A cut list file is JSON: {"threshold": 0.3, "sources": {LABEL: {"frames": N, "fps": "N/D",
"cuts": [first frame of each shot but the first], "shots": [frames of each shot]}}}; tables reads
"shots" (and "fps" when there). Sources are named by neutral labels only.

Usage (numpy is the only dependency):
  python compile_layouts.py cuts anime=probabilities.npz film=film.npz --range film=0:79200 \\
      --fps anime=24000/1001 --out cuts.json
  python compile_layouts.py tables cuts.json > tables.md
  python compile_layouts.py tables cuts.json --gpu-speedup 2.5 --gain 5:0.26,21:0.30 \\
      --compile-cold 25,1.2 --compile-warm 8 --recompile-limit 8 --processes 3 --no-details
  seedvr2x/.venv/bin/python research/scripts/compile_layouts.py check anime=probabilities.npz
"""
import argparse
import json
import math
import sys
from collections import Counter
from fractions import Fraction
from itertools import pairwise

import numpy as np

SHARED = 2  # seedvr2x/src/seedvr2x/runtime/job.py: the latents two consecutive windows share
LAYOUTS = ("balanced", "full-short-last", "slid-back")
STRATEGIES = ("every", "pay", "dynamic", "none")
CAPS = "5,13,19,27,76"  # what 24 GB to 96 GB cards fit at 1080p
BREAK_EVEN = (1, 2, 3, 4, 5, 7, 10, 13, 16, 19, 23, 27, 38, 51, 76)
TORCH_LIMIT = 8  # torch._dynamo.config.recompile_limit's default (torch 2.14.1)
MODEL = {
    "t_token_ms": 0.30,
    "tokens_per_latent": 8280,
    "gain": 0.29,
    "compile_cold": [30.0, 0.0],
    "compile_warm": [10.0, 0.0],
    "compile_first": 0.0,
    "recompile_limit": 0,
    "dynamic_compile": 60.0,
    "dynamic_compile_warm": 20.0,
    "dynamic_gain": 0.20,
    "processes": 1,
    "gpu_speedup": 1.0,
    "dit_share": 0.20,
}


# ---------------------------------------------------------------- seedvr2x's rules, copied

def detected_cuts(probabilities, threshold):
    """The first frame of each shot but the first, by seedvr2x's rule
    (seedvr2x/src/seedvr2x/runtime/cuts.py: detected, _exact, _runs): one detection per run of
    frames whose probability reaches the threshold, at the first of its highest frames, the cut
    being the frame after; none after a peak on the last frame. The values are compared as
    recorded, float32, each widened to the double of its exact value."""
    values = np.asarray(probabilities, dtype=np.float32).astype(np.float64)
    if np.isnan(values).any():
        raise ValueError("a NaN probability")
    reached = np.concatenate(([0], (values >= threshold).astype(np.int8), [0]))
    edges = np.diff(reached)
    firsts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    peaks = (int(a) + int(np.argmax(values[a:b])) for a, b in zip(firsts, ends))
    return [peak + 1 for peak in peaks if peak + 1 < len(values)]


def padded_length(count):
    """The frames of a shot of `count` frames once padded to 4n + 1 (runtime/shot.py,
    padded_length)."""
    return count + (-count + 1) % 4


def latents_of(count):
    """The latents of a shot of `count` frames (runtime/shot.py, shot_layout)."""
    return (padded_length(count) - 1) // 4 + 1


def balanced(latents, window, shared=SHARED):
    """seedvr2x's layout, [start, end) of each window: a copy of
    seedvr2x/src/seedvr2x/runtime/shot.py's window_layout (`check` holds it to the original)."""
    if latents <= window:
        return [(0, latents)]
    if window < 2 * shared + 1:
        raise ValueError(f"windows of {window} latents leave none of their own with {shared} shared")
    count = math.ceil((latents - shared) / (window - shared))
    base, longer = divmod(latents + (count - 1) * shared, count)
    layout = []
    start = 0
    for index in range(count):
        length = base + 1 if index < longer else base
        layout.append((start, start + length))
        start += length - shared
    assert layout[-1][1] == latents and base >= 2 * shared
    return layout


# ---------------------------------------------------------------- the other two layouts

def full_short_last(latents, window, shared=SHARED):
    """Windows of exactly `window` latents every window - shared, the last one what is left:
    shared + 1 to `window` latents."""
    if latents <= window:
        return [(0, latents)]
    if window < 2 * shared + 1:
        raise ValueError(f"windows of {window} latents leave none of their own with {shared} shared")
    layout = []
    start = 0
    while start + window < latents:
        layout.append((start, start + window))
        start += window - shared
    layout.append((start, latents))
    return layout


def slid_back(latents, window, shared=SHARED):
    """full_short_last with its last window slid back to `window` latents ending with the shot."""
    layout = full_short_last(latents, window, shared)
    if len(layout) > 1:
        layout[-1] = (latents - window, latents)
    return layout


LAYOUT = {"balanced": balanced, "full-short-last": full_short_last, "slid-back": slid_back}


def job_lengths(shots, cap, layout):
    """The length in latents of every window of a job, in the order it runs them: shot after
    shot (their frames in `shots`), window after window."""
    lengths = []
    for count in shots:
        lengths.extend(end - start for start, end in LAYOUT[layout](latents_of(count), cap))
    return lengths


# ---------------------------------------------------------------- the cost model

class Model:
    """The cost model's parameters (the module docstring says what each is), read from MODEL,
    then a JSON file, then the flags."""

    def __init__(self, args):
        values = dict(MODEL)
        if args.model:
            with open(args.model) as f:
                given = json.load(f)
            unknown = sorted(set(given) - set(MODEL))
            if unknown:
                sys.exit(f"{args.model}: unknown keys {', '.join(unknown)} (known: {', '.join(MODEL)})")
            values.update(given)
        for key in MODEL:
            flag = getattr(args, key)
            if flag is not None:
                values[key] = flag
        self.values = values
        self.t_token = values["t_token_ms"] / 1000
        self.tokens = values["tokens_per_latent"]
        gain = values["gain"]
        self.gain_table = (
            sorted((int(length), float(g)) for length, g in gain.items()) if isinstance(gain, dict) else None
        )
        self.gain_flat = None if self.gain_table else float(gain)
        self.cold_ab = affine(values["compile_cold"])
        self.warm_ab = affine(values["compile_warm"])
        self.first = float(values["compile_first"])
        self.limit = int(values["recompile_limit"])
        self.dynamic_cold = float(values["dynamic_compile"])
        self.dynamic_warm = float(values["dynamic_compile_warm"])
        self.dynamic_gain = float(values["dynamic_gain"])
        self.processes = max(1, int(values["processes"]))
        self.speedup = float(values["gpu_speedup"])
        self.dit_share = float(values["dit_share"])

    def plain(self, length):
        """Seconds of one plain forward on a window of `length` latents."""
        return self.t_token * self.tokens * length / self.speedup

    def gain(self, length):
        """The share of a window's plain time a compiled forward saves."""
        if self.gain_table is None:
            return self.gain_flat
        lengths, gains = zip(*self.gain_table)
        return float(np.interp(length, lengths, gains))

    def cold(self, length):
        return self.cold_ab[0] + self.cold_ab[1] * length

    def warm(self, length):
        return self.warm_ab[0] + self.warm_ab[1] * length

    def describe(self):
        v = self.values
        gain = (
            f"{self.gain_flat:g}" if self.gain_table is None
            else "table " + ", ".join(f"L={length}: {g:g}" for length, g in self.gain_table)
        )
        limit = f"{self.limit} lengths per process" if self.limit else "none"
        return [
            (f"- plain forward: {v['t_token_ms']:g} ms a token, {self.tokens} tokens a latent, GPU speed-up "
             f"{self.speedup:g} (forwards only): {self.plain(1):.3f} s a latent"),
            f"- compiled gain: {gain}",
            (f"- a compilation: cold {self.cold_ab[0]:g} + {self.cold_ab[1]:g} x L s, warm {self.warm_ab[0]:g} + "
             f"{self.warm_ab[1]:g} x L s, {self.first:g} s more for a process's first; recompile limit: {limit}"),
            (f"- dynamic shapes: one compilation of {self.dynamic_cold:g} s (warm {self.dynamic_warm:g} s), gain "
             f"{self.dynamic_gain:g}"),
            f"- processes: {self.processes}; the DiT's share of a job: {self.dit_share:g}",
        ]


def affine(value):
    """(a, b) of a + b x L, from a number or a pair."""
    if isinstance(value, (int, float)):
        return float(value), 0.0
    pair = [float(x) for x in value]
    return pair[0], (pair[1] if len(pair) > 1 else 0.0)


def processes_of(lengths, count):
    """A job's windows, in order, in `count` processes: each an equal share of its latents."""
    total = sum(lengths)
    parts = [[] for _ in range(count)]
    done = 0
    for length in lengths:
        parts[min(count - 1, done * count // total)].append(length)
        done += length
    return [part for part in parts if part]


def run(model, lengths, strategy):
    """A job's times under a strategy: {"dit": seconds of forwards, "compile": seconds compiling,
    "compiled": the distinct lengths compiled, "compilations": how many compilations}."""
    dit = compiling = 0.0
    compiled, compilations = set(), 0
    for index, part in enumerate(processes_of(lengths, model.processes)):
        plain = sum(model.plain(length) for length in part)
        if strategy == "none":
            dit += plain
            continue
        if strategy == "dynamic":
            dit += plain * (1 - model.dynamic_gain)
            compiling += model.dynamic_warm if index else model.dynamic_cold
            compilations += 1
            continue
        counts = Counter(part)  # in the order the process meets them
        cost = {length: model.warm(length) if length in compiled else model.cold(length) for length in counts}
        saving = {length: model.gain(length) * model.plain(length) * n for length, n in counts.items()}
        if strategy == "every":
            chosen = list(counts)[: model.limit] if model.limit else list(counts)
        else:
            chosen = [length for length in counts if saving[length] > cost[length]]
            if model.limit and len(chosen) > model.limit:
                chosen.sort(key=lambda length: saving[length] - cost[length], reverse=True)
                chosen = chosen[: model.limit]
            if chosen and sum(saving[length] - cost[length] for length in chosen) <= model.first:
                chosen = []
        dit += plain - sum(saving[length] for length in chosen)
        compiling += sum(cost[length] for length in chosen) + (model.first if chosen else 0.0)
        compiled.update(chosen)
        compilations += len(chosen)
    return {"dit": dit, "compile": compiling, "compiled": compiled, "compilations": compilations}


# ---------------------------------------------------------------- cuts

def pairs(items, what):
    """{LABEL: value} of LABEL=value arguments."""
    found = {}
    for item in items or ():
        label, sep, value = item.partition("=")
        if not sep or not label or not value:
            sys.exit(f"{what}: LABEL=VALUE expected, not {item!r}")
        if label in found:
            sys.exit(f"{what}: {label} given twice")
        found[label] = value
    return found


def load_probabilities(path, key):
    if path.endswith(".npz"):
        with np.load(path) as z:
            if key not in z:
                sys.exit(f"{path}: no array {key!r} (it holds {', '.join(z.files)})")
            values = z[key]
    else:
        values = np.load(path)
    if values.ndim != 1 or not len(values):
        sys.exit(f"{path}: one probability per frame expected, not an array of shape {values.shape}")
    return values


def read_sources(args):
    """{LABEL: (probabilities of the frames kept, [start, end) kept or None)} of the arguments."""
    files, ranges = pairs(args.sources, "sources"), pairs(args.range, "--range")
    for label in ranges:
        if label not in files:
            sys.exit(f"--range: no source {label}")
    sources = {}
    for label, path in files.items():
        values = load_probabilities(path, args.key)
        kept = None
        if label in ranges:
            start, _, end = ranges[label].partition(":")
            kept = [int(start), int(end)]
            if not 0 <= kept[0] < kept[1] <= len(values):
                sys.exit(f"--range {label}: {kept[0]}:{kept[1]} outside the file's {len(values)} frames")
            values = values[kept[0]:kept[1]]
        sources[label] = (values, kept)
    return sources


def shots_of(frames, cuts):
    """The frames of each shot of a source of `frames` frames cut at `cuts`."""
    bounds = [0, *cuts, frames]
    return [end - start for start, end in pairwise(bounds)]


def cmd_cuts(args):
    rates = pairs(args.fps, "--fps")
    out = {"format": "compile_layouts cut lists", "threshold": args.threshold,
           "rule": "one cut after the peak of each run of frames reaching the threshold", "sources": {}}
    for label, (values, kept) in read_sources(args).items():
        cuts = detected_cuts(values, args.threshold)
        entry = {"frames": len(values), "cuts": cuts, "shots": shots_of(len(values), cuts)}
        if label in rates:
            entry["fps"] = str(Fraction(rates[label]))
        if kept:
            entry["range"] = kept
        out["sources"][label] = entry
        print(f"{label}: {len(values)} frames, {len(cuts) + 1} shots at {args.threshold:g}", file=sys.stderr)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"wrote {args.out}", file=sys.stderr)


# ---------------------------------------------------------------- check

def cmd_check(args):
    try:
        from seedvr2x.runtime import cuts as their_cuts
        from seedvr2x.runtime import job as their_job
        from seedvr2x.runtime import shot as their_shot
    except ImportError as error:
        sys.exit(f"seedvr2x isn't importable ({error}): run check with seedvr2x's own environment")
    caps = [int(x) for x in args.caps.split(",")]
    failures = 0
    if their_job.SHARED != SHARED:
        failures += 1
        print(f"FAIL SHARED: {SHARED} here, {their_job.SHARED} in runtime/job.py")
    longest = 0
    for label, (values, _) in read_sources(args).items():
        mine = detected_cuts(values, args.threshold)
        theirs = [cut.frame for cut in their_cuts.detected(values, args.threshold)]
        shots = shots_of(len(values), mine)
        longest = max(longest, max(latents_of(count) for count in shots))
        padded = sum(padded_length(count) != their_shot.padded_length(count) for count in set(shots))
        layouts = sum(
            balanced(latents_of(count), cap) != their_shot.shot_layout(count, cap)
            for count in set(shots) for cap in caps
        )
        ok = mine == theirs and not padded and not layouts
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {label}: {len(values)} frames, cuts {len(mine)} here, {len(theirs)} "
              f"by runtime/cuts.py, {'identical' if mine == theirs else 'DIFFERENT'}; {len(set(shots))} distinct "
              f"shot lengths x {len(caps)} caps: padded_length differs on {padded}, the layout on {layouts}")
    differing = sum(
        balanced(latents, cap) != their_shot.window_layout(latents, cap)
        for latents in range(1, longest + 1) for cap in caps
    )
    failures += bool(differing)
    print(f"{'FAIL' if differing else 'ok  '} window_layout: every length from 1 to {longest} latents x caps "
          f"{args.caps}: {differing} of {longest * len(caps)} pairs differ")
    sys.exit(1 if failures else 0)


# ---------------------------------------------------------------- tables

def read_cut_lists(paths):
    """{LABEL: {"shots": [...], "fps": Fraction or None}} of the cut list files, and their
    thresholds."""
    sources, thresholds = {}, set()
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        thresholds.add(data.get("threshold"))
        for label, entry in data["sources"].items():
            if label in sources:
                sys.exit(f"{path}: {label} is in another cut list too")
            shots = [int(count) for count in entry["shots"]]
            if not shots or min(shots) < 1:
                sys.exit(f"{path}: {label}: shots of 1 frame or more expected")
            sources[label] = {"shots": shots, "fps": Fraction(entry["fps"]) if entry.get("fps") else None}
    return sources, thresholds


def quantiles(values):
    """min / 10% / median / 90% / 99% / max, each a value met."""
    picks = np.quantile(np.asarray(values), (0.0, 0.1, 0.5, 0.9, 0.99, 1.0), method="inverted_cdf")
    return " / ".join(f"{int(x):,}" for x in picks)


def layout_stats(shots, cap, layout, top):
    """A layout's figures on a job, and its window lengths in order."""
    lengths = job_lengths(shots, cap, layout)
    shot_latents = [latents_of(count) for count in shots]
    windows = Counter(lengths)
    by_latents = sorted(windows.items(), key=lambda item: (-item[0] * item[1], -item[0]))
    computed = sum(lengths)
    covered = np.cumsum([length * n for length, n in by_latents]) / computed
    return {
        "lengths": lengths,
        "shots": len(shots),
        "windows": len(lengths),
        "joins": len(lengths) - len(shots),
        "shot_latents": sum(shot_latents),
        "computed": computed,
        "one_window": sum(n for n in shot_latents if n <= cap) / sum(shot_latents),
        "one_window_shots": sum(n <= cap for n in shot_latents) / len(shots),
        "distinct": len(windows),
        "rare": sum(n <= 2 for n in windows.values()),
        "at_cap": windows.get(cap, 0) * cap / computed,
        "cover": [float(covered[min(k, len(covered)) - 1]) for k in (1, 3, TORCH_LIMIT)],
        "top": ", ".join(f"{length}x{n} ({100 * length * n / computed:.0f}%)" for length, n in by_latents[:top]),
    }


def pct(x, digits=1):
    return f"{100 * x:+.{digits}f}"


def row(cells):
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def table(header, rows, left=1):
    """Markdown lines: the first `left` columns left-aligned, the others right-aligned."""
    marks = ["---"] * left + ["---:"] * (len(header) - left)
    return [row(header), row(marks), *(row(r) for r in rows), ""]


def cmd_tables(args):
    model = Model(args)
    caps = [int(x) for x in args.caps.split(",")]
    if min(caps) < 2 * SHARED + 1:
        sys.exit(f"--caps: a window holds {2 * SHARED + 1} latents at least, {SHARED} shared on each side")
    sources, thresholds = read_cut_lists(args.cut_lists)
    out = ["# Window layouts and compile strategies", ""]
    cut_at = ", ".join(str(t) for t in sorted(thresholds, key=str))
    out += [(f"Cut lists: {len(sources)} sources, threshold {cut_at}; caps {args.caps} latents; windows share "
             f"{SHARED} latents. Cost model:"), ""]
    out += [*model.describe(), ""]
    out += [("Every net gain is against the plain DiT time of the balanced layout, compilations paid; "
             "% job = % DiT x the DiT's share."), ""]

    out += ["## Sources", ""]
    rows = []
    for label, src in sources.items():
        shots = src["shots"]
        frames = sum(shots)
        minutes = f"{frames / src['fps'] / 60:.1f}" if src["fps"] else "-"
        per_minute = f"{len(shots) / (frames / src['fps'] / 60):.1f}" if src["fps"] else "-"
        rows.append([label, f"{frames:,}", f"{float(src['fps']):.3f}" if src["fps"] else "-", minutes,
                     f"{len(shots):,}", per_minute, quantiles(shots),
                     quantiles([latents_of(count) for count in shots]),
                     f"{sum(latents_of(count) for count in shots):,}"])
    out += table(["Source", "Frames", "fps", "Minutes", "Shots", "Shots a minute",
                  "Shot frames: min / 10% / median / 90% / 99% / max", "Shot latents: the same", "Latents"], rows)

    out += ["## Break-even: the windows a length needs for its compilation to pay", ""]
    rows = []
    for length in sorted(set(BREAK_EVEN) | set(caps)):
        saved = model.gain(length) * model.plain(length)
        rows.append([length, f"{model.tokens * length:,}", f"{model.plain(length):.2f}", f"{saved:.2f}",
                     f"{model.cold(length):.1f}", f"{model.cold(length) / saved:.1f}",
                     f"{model.warm(length):.1f}", f"{model.warm(length) / saved:.1f}"])
    out += table(["L (latents)", "Tokens", "Plain s a window", "Saved s a window", "Cold compile s",
                  "Windows to pay it", "Warm compile s", "Windows to pay it"], rows, left=0)
    out += [(f"One dynamic compilation ({model.dynamic_cold:g} s) pays from "
             f"{model.dynamic_cold / (model.dynamic_gain * model.plain(1)):,.0f} latents computed."), ""]

    for cap in caps:
        out += [f"## Cap {cap} latents ({4 * cap - 3} frames)", ""]
        stats, results = {}, {}
        for label, src in sources.items():
            for layout in LAYOUTS:
                stats[label, layout] = layout_stats(src["shots"], cap, layout, args.top)
                for strategy in STRATEGIES:
                    results[label, layout, strategy] = run(model, stats[label, layout]["lengths"], strategy)
        base = {label: results[label, "balanced", "none"]["dit"] for label in sources}
        net = {key: base[key[0]] - r["dit"] - r["compile"] for key, r in results.items()}

        out += ["### Summary", ""]
        out += [("Lengths: distinct window lengths met / those `pay` compiles. Net gains in % of the plain DiT "
                 "time (balanced), for `every`, `pay`, `dynamic`, and `none` on slid-back (its extra compute)."), ""]
        rows = []
        for label in [*sources, None]:
            labels = [label] if label else list(sources)
            total = sum(base[name] for name in labels)
            balanced_stats = [stats[name, "balanced"] for name in labels]
            in_one = sum(s["one_window"] * s["shot_latents"] for s in balanced_stats)
            cells = [label or "all (each its own job)",
                     f"{sum(s['computed'] for s in balanced_stats):,}",
                     f"{100 * in_one / sum(s['shot_latents'] for s in balanced_stats):.1f}"]
            for layout in LAYOUTS:
                met = sum(stats[name, layout]["distinct"] for name in labels)
                paying = sum(len(results[name, layout, "pay"]["compiled"]) for name in labels)
                cells.append(f"{met} / {paying}")
                # full-short-last computes what balanced does: its dynamic and none are balanced's
                picks = {"balanced": STRATEGIES[:3], "full-short-last": STRATEGIES[:2], "slid-back": STRATEGIES}
                cells += [pct(sum(net[name, layout, s] for name in labels) / total) for s in picks[layout]]
            rows.append(cells)
        out += table(["Source", "Latents computed (balanced)", "% of latents in one-window shots",
                      "balanced: lengths", "every", "pay", "dynamic",
                      "full-short-last: lengths", "every", "pay",
                      "slid-back: lengths", "every", "pay", "dynamic", "none"], rows)
        if args.no_details:
            continue

        out += ["### Layouts", ""]
        out += [(f"Joins = windows - shots. Excess: latents computed over balanced's. Rare: lengths met once or "
                 f"twice. Cover: the share of the latents computed that the 1, 3 and {TORCH_LIMIT} lengths "
                 f"computing most hold. Top: length x windows (share of the latents computed)."), ""]
        rows = []
        for label in sources:
            for layout in LAYOUTS:
                s, b = stats[label, layout], stats[label, "balanced"]
                rows.append([label, layout, f"{s['shots']:,}", f"{100 * s['one_window_shots']:.1f}",
                             f"{100 * s['one_window']:.1f}", f"{s['windows']:,}", f"{s['joins']:,}",
                             f"{s['computed']:,}", f"{s['computed'] - b['computed']:,}",
                             pct(s["computed"] / b["computed"] - 1, 2), s["distinct"], s["rare"],
                             f"{100 * s['at_cap']:.1f}", " / ".join(f"{100 * c:.0f}" for c in s["cover"]), s["top"]])
        out += table(["Source", "Layout", "Shots", "% of shots in one window", "% of latents in one-window shots",
                      "Windows", "Joins", "Latents computed", "Excess", "Excess %", "Distinct lengths", "Rare",
                      "% of latents at the cap's length", f"Cover 1 / 3 / {TORCH_LIMIT} %", f"Top {args.top}"],
                     rows, left=2)

        out += ["### Strategies", ""]
        out += [("Times in seconds. Net = the balanced layout's plain DiT time - (DiT + compile). The last "
                 "column: the gain a dynamic compilation needs to match `pay` on that layout."), ""]
        rows = []
        for label in sources:
            for layout in LAYOUTS:
                plain = results[label, layout, "none"]["dit"]
                dynamic_compile = results[label, layout, "dynamic"]["compile"]
                match = (net[label, layout, "pay"] - base[label] + plain + dynamic_compile) / plain
                for strategy in STRATEGIES:
                    r = results[label, layout, strategy]
                    gain = net[label, layout, strategy]
                    rows.append([label, layout, strategy, f"{base[label]:,.0f}",
                                 len(r["compiled"]) if strategy in ("every", "pay") else "-", r["compilations"],
                                 f"{r['dit']:,.0f}", f"{r['compile']:,.0f}", f"{gain:+,.0f}",
                                 pct(gain / base[label]), pct(gain / base[label] * model.dit_share, 2),
                                 f"{match:.3f}" if strategy == "dynamic" else ""])
        out += table(["Source", "Layout", "Strategy", "Plain DiT s (balanced)", "Lengths compiled", "Compilations",
                      "DiT s", "Compile s", "Net s", "Net % DiT", "Net % job", "Dynamic gain to match pay"],
                     rows, left=3)
    print("\n".join(out))


# ---------------------------------------------------------------- main

def parse_gain(text):
    """A flat gain, or a table L:gain,L:gain."""
    if ":" not in text:
        return float(text)
    return {int(length): float(gain) for length, gain in (item.split(":") for item in text.split(","))}


def parse_affine(text):
    """A or A,B: a + b x L seconds."""
    return [float(x) for x in text.split(",")]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def source_opts(p):
        p.add_argument("sources", nargs="+", metavar="LABEL=FILE", help="per-frame probabilities, .npz or .npy")
        p.add_argument("--threshold", type=float, default=0.3, help="seedvr2x's --cut-threshold (default 0.3)")
        p.add_argument("--key", default="single", help="the array of a .npz (default: single)")
        p.add_argument("--range", action="append", metavar="LABEL=START:END",
                       help="keep frames START to END - 1 of a file holding more")

    p = sub.add_parser("cuts", help="cut lists from per-frame probabilities, by seedvr2x's rule")
    source_opts(p)
    p.add_argument("--fps", action="append", metavar="LABEL=N/D", help="the source's frame rate, for durations")
    p.add_argument("--out", required=True, metavar="CUTS.json")

    p = sub.add_parser("check", help="the copied rules against seedvr2x's own, on the same data")
    source_opts(p)
    p.add_argument("--caps", default=CAPS)

    p = sub.add_parser("tables", help="Markdown tables: lengths per layout, net gain per strategy")
    p.add_argument("cut_lists", nargs="+", metavar="CUTS.json")
    p.add_argument("--caps", default=CAPS, help=f"window caps in latents (default {CAPS})")
    p.add_argument("--top", type=int, default=5, help="lengths shown per layout, by latents (default 5)")
    p.add_argument("--no-details", action="store_true", help="per cap, the summary across sources only")
    p.add_argument("--model", metavar="MODEL.json", help="cost model keys (the flags below win)")
    p.add_argument("--t-token-ms", type=float, help=f"plain ms per token (default {MODEL['t_token_ms']})")
    p.add_argument("--tokens-per-latent", type=int, help=f"default {MODEL['tokens_per_latent']} (1080p padded)")
    p.add_argument("--gain", type=parse_gain, metavar="G | L:G,L:G,...",
                   help=f"share of the plain time compiled saves, flat or per length (default {MODEL['gain']})")
    p.add_argument("--compile-cold", type=parse_affine, metavar="A[,B]",
                   help="seconds A + B x L for a length's first compilation (default 30)")
    p.add_argument("--compile-warm", type=parse_affine, metavar="A[,B]",
                   help="the same when an earlier process compiled it (default 10)")
    p.add_argument("--compile-first", type=float, help="more seconds for a process's first compilation (default 0)")
    p.add_argument("--recompile-limit", type=int,
                   help=f"lengths a process compiles at most, 0 for no limit (default 0; torch's own: {TORCH_LIMIT})")
    p.add_argument("--dynamic-compile", type=float, help="seconds of the one dynamic compilation (default 60)")
    p.add_argument("--dynamic-compile-warm", type=float, help="the same in a later process (default 20)")
    p.add_argument("--dynamic-gain", type=float, help=f"its gain (default {MODEL['dynamic_gain']})")
    p.add_argument("--processes", type=int, help="processes the job is run in (default 1)")
    p.add_argument("--gpu-speedup", type=float, help="divides the forward times, not the compilations (default 1)")
    p.add_argument("--dit-share", type=float, help=f"the DiT's share of a job (default {MODEL['dit_share']})")

    args = ap.parse_args()
    {"cuts": cmd_cuts, "check": cmd_check, "tables": cmd_tables}[args.cmd](args)


if __name__ == "__main__":
    main()

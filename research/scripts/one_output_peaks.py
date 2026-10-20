"""Tile probe for seedvr2x's one-output convolution: each tile's peak with it and without it.

seedvr2x's VAE splits a convolution whose output is large into pieces (the frames' slices, then
the pieces of a slice) and used to concatenate them, the pieces and the concatenation standing on
the card together. With the one-output convolution (seedvr2x's vendored causal_inflation_lib.py,
InflatedCausalConv3d.whole_output) a convolution allocates its whole output ahead and writes each
piece into it: it takes that path where the output a slice of four frames gives is 2.5 GiB or
more (ONE_OUTPUT_FROM), judged at each convolution's own call. This script runs tile_peaks.py's
measure, unchanged, on both versions of the code and sets the two side by side, with the planner's
figure for each pass (runtime/planner.py, MEASURED, and _fit for an untiled pass). The measure is
tile_peaks.py's: torch's max_memory_allocated above memory_allocated before the pass, the peak
statistics reset first, the DiT's and the VAE's weights on the card before it and not counted;
tiling.encode_tiled, its latent returned in system memory, then tiling.decode_tiled of that
latent; 21 frames of the source from --first, the tiles sharing 64 px; the two models loaded whole
(runtime/model.py, load_models, no plan); cudaMallocAsync unless the environment names an
allocator (PYTORCH_CUDA_ALLOC_CONF), set before torch is imported. See tile_peaks.py's docstring
for the squares (a T x T tile and a T x 192 one, fed on frames of T rows by T + 128 columns) and
the tiles cut off at the picture's edge (1080p padded to 1104 x 1920, 720p to 752 x 1280).

Beyond tile_peaks.py, `run` takes:
  --steps STEP,...   only those steps, in that order. STEP is square-T, cutoff-1080p-T,
         cutoff-720p-T, untiled-1080p or untiled-2160p. An untiled step is the untiled encode and
         decode as a run calls them (shot.encode_shot and shot.decode_shot without tiles):
         model.encode_stream of shot.encoder_inputs, then model.decode_stream of its latent. Each
         untiled pass is measured only where the planner's own fit for it (planner._fit: the
         larger of the untiled and the small tiles' fits at the padded frame's area) leaves
         --reserve GiB of the free memory; else it is skipped and said. Without --steps, the
         order of tile_peaks.py: every square, then every cut-off tile.
  untiled-2160p moves the DiT's weights to the CPU before its passes; put it last in its
         process. The untiled 4K encode's fit is 75.27 GiB and the 96 GB card has 78.45 GiB free
         once the two models are loaded: with the DiT away 93.68 GiB are free. The peaks count
         from what is allocated before each pass either way.
  --tiny-limits CONV_GIB ONE_OUTPUT_BYTES   with --tiny only, for a debug run on the CPU: the
         VAE's convolution limit and the floor of the one-output path lowered, so that tiny tiles
         split and take it.
Each tile's line also holds the SHA-256 of its outputs (the encode's latent, in its logical
order; each decoded slice, copied to the CPU inside the measure, which allocates nothing on the
device for a dense slice, and one digest over all the slices in order), and, where the code has
the one-output convolution, which convolutions took it (the records' keys a4, a4_paths and
a4_paths_by_conv: for each convolution of the VAE, how many of its calls took the one output and
how many concatenated, the calls that judged the path, whole=None). Run the same steps on the
code with it and on the code without it: the two codes differ in that convolution alone.

Other subcommands:
  measured   --steps STEP,... --out measured.json: the planner's figure for each step's two
         passes, from the seedvr2x it imports (MEASURED where a tile has one, planner._fit for
         an untiled pass), and whether the pass is one the one-output convolution takes by the
         rule above (the records' key a4_list): decodes in squares from 1152 px, in 1104-row
         tiles from 1216 px and untiled at 1080p and 4K; encodes in squares from 1664 px and
         untiled at 4K. No GPU.
  compare    --with W.jsonl... --without N.jsonl... --measured measured.json [--order A,B,...]
         [--differ FILE] [--compact]: a table of the two runs against the planner's figures,
         then a summary (passes over the figure with it, outputs not equal, the path never seen
         taken); --differ writes the steps whose outputs differ or that a code lacks, to be run
         again. Standard library only: any python runs it. It reads each pass's peak in GiB as
         `run` rounded it to three decimals; `table` reads the bytes.
  table      --with W.jsonl... --without N.jsonl... --measured measured.json [--logs LOG...]:
         the final table, from the records' bytes, one row per step and pass: the planner's
         figure, the peak without and with the one output, both differences, whether every
         digest is equal across every run of both codes, the convolutions that took the path
         (whole calls / judged calls); then the controls (the passes off the list: peak bytes
         equal with and without), the warm-up's peaks and digests across its processes, the
         untiled 4K step's DiT move, and, of each --logs file (a run's log: the lines of the
         wrapper that holds the GPU's lock, nvidia-smi before and the release, if any), the
         number of times cudaMallocAsync recovered from an allocation failure. Standard library
         only.

Sizes are rows x columns; a GiB is 2**30 bytes.

Reproduce (seedvr2x's venv python for run and measured; run under the GPU's lock, ffmpeg and
ffprobe in the PATH; one group of steps in one process, the code with the one output first, then
the code without it, each in a tree of its own):
  python one_output_peaks.py measured --steps square-512,square-2048,untiled-1080p \
      --out measured.json
  python one_output_peaks.py run --dit DIT.safetensors --vae VAE.safetensors --source CLIP.mkv \
      --first 48 --out runs/g1-with.jsonl \
      --steps square-512,square-2048,square-1600,untiled-1080p     # the tree with it
  python one_output_peaks.py run ... --out runs/g1-without.jsonl ...   # the tree without it
  python3 one_output_peaks.py compare --with runs/*-with.jsonl --without runs/*-without.jsonl \
      --measured measured.json
  python3 one_output_peaks.py table --with runs/*-with.jsonl --without runs/*-without.jsonl \
      --measured measured.json --logs runs/*.log
Begin every process with square-512, a warm-up that is also the check of run-to-run determinism
(2026-10-11's table took it as its process's first pass).
"""

import argparse
import gc
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from functools import partial
from pathlib import Path

GIB = 2**30
OVERLAP = 64  # the pixels two tiles share: seedvr2x's default (runtime/plan.py, OVERLAP)
FRAMES = 21
SQUARES = list(range(512, 2049, 64))
# The tiles cut off at the picture's edge: a name, the output's (rows, columns), the tiles' sides.
CUTOFFS = [
    ("1080p", (1080, 1920), list(range(1152, 1857, 64))),
    ("720p", (720, 1280), list(range(768, 1217, 64))),
]
UNTILED = {"1080p": (1080, 1920), "2160p": (2160, 3840)}
# --tiny's, for a debug run on the CPU.
TINY_OVERLAP, TINY_FRAMES = 16, 9
TINY_SQUARES = [64, 96, 128]
TINY_CUTOFFS = [("1080p", (48, 160), [96, 128]), ("720p", (32, 128), [80, 96])]
TINY_UNTILED = {"1080p": (48, 160), "2160p": (96, 160)}
# The tiled fits, each (GiB, GiB per megapixel of the tile), the encode's then the decode's: from
# tiles of REFITTED px a side (research/docs/planner-limits.md, Tiled VAE at 4K), and below
# (research/docs/vram.md, Tiling).
REFITTED = 1024
FITS = ((1.36, 8.66), (0.44, 16.29))
FITS_SMALL = ((1.7, 8.4), (1.6, 15.6))
# The untiled fits, the planner's (runtime/planner.py, ENCODE_UNTILED, DECODE_UNTILED): an untiled
# pass takes the larger of its own and of the small tiles' at the padded frame's area (_fit).
FITS_UNTILED = ((1.2, 8.8), (0.8, 16.1))
PASSES = ("encode", "decode")


def fits(side: int, pixels: int) -> tuple[float, float]:
    """The encode's and the decode's fitted peaks for a tile of `side` px a side and of `pixels`
    pixels, in GiB."""
    (encode, encode_slope), (decode, decode_slope) = FITS if side >= REFITTED else FITS_SMALL
    area = pixels / 1e6
    return encode + encode_slope * area, decode + decode_slope * area


def untiled_fits(pixels: int) -> tuple[float, float]:
    """The untiled encode's and decode's fitted peaks for padded frames of `pixels` pixels, in
    GiB, as the planner takes them (planner._fit with no tiles)."""
    area = pixels / 1e6
    return tuple(max(c + s * area for c, s in (FITS_UNTILED[i], FITS_SMALL[i])) for i in range(2))  # type: ignore[return-value]


def parse_steps(text: str) -> list[tuple[str, str | None, int | None]]:
    steps: list[tuple[str, str | None, int | None]] = []
    for item in (part.strip() for part in text.split(",")):
        if not item:
            continue
        bits = item.split("-")
        if bits[0] == "square" and len(bits) == 2:
            steps.append(("square", None, int(bits[1])))
        elif bits[0] == "cutoff" and len(bits) == 3 and bits[1] in ("1080p", "720p"):
            steps.append(("cutoff", bits[1], int(bits[2])))
        elif bits[0] == "untiled" and len(bits) == 2 and bits[1] in UNTILED:
            steps.append(("untiled", bits[1], None))
        else:
            raise SystemExit(f"a step this script doesn't know: {item!r}")
    return steps


def step_name(kind: str, res: str | None, side: int | None) -> str:
    if kind == "square":
        return f"square-{side}"
    if kind == "cutoff":
        return f"cutoff-{res}-{side}"
    return f"untiled-{res}"


# ---------------------------------------------------------------- run


def run(args: argparse.Namespace) -> int:
    # Before torch is imported; the environment's own setting stands.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

    import numpy as np
    import torch

    import seedvr2x
    from seedvr2x.runtime import job, model, shot, tiling
    from seedvr2x.runtime.plan import Tiles
    from seedvr2x.vendor.models.video_vae_v3.modules import causal_inflation_lib as cil

    torch.set_grad_enabled(False)
    started = time.time()
    cuda = torch.cuda.is_available()
    device = torch.device("cuda", 0) if cuda else torch.device("cpu")
    overlap, count = (TINY_OVERLAP, TINY_FRAMES) if args.tiny else (OVERLAP, FRAMES)
    squares = TINY_SQUARES if args.tiny else SQUARES
    cutoffs = TINY_CUTOFFS if args.tiny else CUTOFFS
    untiled_targets = TINY_UNTILED if args.tiny else UNTILED
    allowance = 2**40  # a scratch's allowance: always in system memory
    failed = 0
    if args.tiny_limits and not args.tiny:
        raise SystemExit("--tiny-limits is for --tiny's CPU dry run only")

    def emit(**record) -> None:
        line = json.dumps({"t": round(time.time() - started, 1), **record}, default=str)
        with args.out.open("a") as file:
            file.write(line + "\n")
            file.flush()
            os.fsync(file.fileno())
        print(line, flush=True)

    def clean() -> None:
        gc.collect()
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def resident_peak() -> int:
        """The process's resident peak, in bytes, since it was last reset (measure)."""
        with open("/proc/self/status") as file:
            for line in file:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
        return 0

    def failure() -> str:
        """The current exception: its traceback on stderr, its last lines returned."""
        text = traceback.format_exc()
        print(text, file=sys.stderr, flush=True)
        return " | ".join(line.strip() for line in text.strip().splitlines()[-5:])[-700:]

    def free_gib() -> float | None:
        if not cuda:
            return None
        clean()
        return round(torch.cuda.mem_get_info()[0] / GIB, 4)

    # The one-output path, counted where the code has it (the records call it a4): for each
    # convolution (by its name in the VAE), the calls that judged it, [took the one output,
    # concatenated]. A wrapper that calls the method and returns what it returns: it allocates
    # nothing.
    a4 = hasattr(cil.InflatedCausalConv3d, "whole_output")
    names: dict[int, str] = {}
    paths: dict[str, list[int]] = {}
    if a4:
        original = cil.InflatedCausalConv3d.whole_output

        def counted(self, like, padded, head=0, whole=None):
            output = original(self, like, padded, head, whole)
            if whole is None:
                taken = paths.setdefault(names.get(id(self), "?"), [0, 0])
                taken[0 if output is not None else 1] += 1
            return output

        cil.InflatedCausalConv3d.whole_output = counted

    def path_summary() -> dict:
        took = sorted(name for name, (yes, _) in paths.items() if yes)
        mixed = sorted(name for name, (yes, no) in paths.items() if yes and no)
        return {
            "judged_convs": len(paths),
            "judged_calls": sum(yes + no for yes, no in paths.values()),
            "one_output_calls": sum(yes for yes, _ in paths.values()),
            "one_output_convs": took,
            "mixed_convs": mixed,
        }

    def measure(function, frames: int):
        """function() and its measures, in bytes: the device's peak above what is allocated
        before the call, the reserved peak, what is allocated right after it returns, the
        seconds, the process's resident peak during it."""
        clean()
        try:
            with open("/proc/self/clear_refs", "w") as file:
                file.write("5")  # the resident peak starts again from here
        except OSError:
            pass
        before = reserved = 0
        if cuda:
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
        paths.clear()
        began = time.time()
        result = function()
        if cuda:
            torch.cuda.synchronize()
        seconds = time.time() - began
        after = peak = before
        if cuda:
            after = torch.cuda.memory_allocated()
            peak = torch.cuda.max_memory_allocated()
            try:
                reserved = torch.cuda.max_memory_reserved()
            except RuntimeError:
                pass  # an allocator that doesn't count it: 0
        measured = {
            "before": before,
            "peak": peak,
            "above": peak - before,
            "above_gib": round((peak - before) / GIB, 4),
            "reserved_peak": reserved,
            "after": after,
            "left": after - before,
            "seconds": round(seconds, 2),
            "s_per_frame": round(seconds / frames, 3),
            "rss_peak": resident_peak(),
        }
        if a4:
            measured["a4_paths"] = path_summary()
            measured["a4_paths_by_conv"] = {name: list(v) for name, v in sorted(paths.items())}
        return result, measured

    def tensor_digest(t, also=None) -> str:
        """SHA-256 of a CPU tensor's values' bytes in their logical order (C-contiguous); also,
        when given, a running SHA-256 fed the same bytes."""
        flat = t.contiguous().reshape(-1).view(torch.uint8).numpy()
        if also is not None:
            also.update(flat)
        return hashlib.sha256(flat).hexdigest()

    def passes(record: dict, encode, decode, decode_fits=None) -> None:
        """The encode pass, then the decode pass of its latent, each measured, their outputs'
        digests: one line. decode_fits, where given, says whether the decode may run (a pass
        whose fit passes the free memory is skipped, and said)."""
        latent, record["encode"] = measure(encode, count)
        record["latent"] = {
            "shape": list(latent.shape),
            "stride": list(latent.stride()),
            "dtype": str(latent.dtype),
            "device": str(latent.device),
        }
        record["digests"] = {"latent": tensor_digest(latent.to("cpu"))}
        if decode_fits is not None:
            fit, why = decode_fits()
            if not fit:
                record["decode_skipped"] = why
                latent = None
                emit(**record)
                return
        slices: list = []
        whole = hashlib.sha256()

        def decoded() -> int:
            got = 0
            for piece in decode(latent):
                got += piece.shape[1]
                dense = piece.is_contiguous()
                host = piece.to("cpu")
                del piece
                slices.append(
                    [list(host.shape), str(host.dtype), dense, tensor_digest(host, whole)]
                )
                del host
            return got

        try:
            record["decoded_frames"], record["decode"] = measure(decoded, count)
        except Exception:
            record["decode_error"] = failure()
        record["digests"]["decode"] = whole.hexdigest() if slices else None
        record["digests"]["decode_slices"] = slices
        latent = None  # given up before the next tile
        emit(**record)
        if "decode_error" in record:
            raise RuntimeError("the decode failed (the encode's line is written)")

    def fits_the_card(side: int, pixels: int) -> bool:
        """Whether a tile's fitted peaks leave --reserve of the memory free now; said if not."""
        if not cuda:
            return True
        clean()
        free = torch.cuda.mem_get_info()[0]
        need = max(fits(side, pixels)) * GIB
        if need <= free - args.reserve * GIB:
            return True
        emit(
            skipped=True,
            tile=side,
            largest_px=pixels,
            fit_gib=round(need / GIB, 3),
            free_gib=round(free / GIB, 3),
            reserve_gib=args.reserve,
        )
        return False

    def read_frames() -> np.ndarray:
        """`count` frames of the source from --first on, whole in memory: (count, H, W, 3) uint8."""
        probe = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
        probe += ["-show_entries", "stream=width,height", "-of", "csv=p=0", str(args.source)]
        said = subprocess.run(probe, check=True, stdout=subprocess.PIPE, text=True).stdout
        width, height = (int(value) for value in said.split()[0].split(",")[:2])
        first = args.first
        command = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(args.source)]
        command += [
            "-vf",
            f"trim=start_frame={first}:end_frame={first + count},setpts=PTS-STARTPTS",
        ]
        command += ["-fps_mode", "passthrough", "-frames:v", str(count)]
        command += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        raw = subprocess.run(command, check=True, stdout=subprocess.PIPE).stdout
        if len(raw) != count * height * width * 3:
            raise RuntimeError(f"{len(raw)} bytes read, not {count} frames of {height}x{width}")
        return np.frombuffer(raw, np.uint8).reshape(count, height, width, 3).copy()

    def reader(frames: np.ndarray):
        """read(n) over frames in memory, as seedvr2x's reader gives them: float32 in [0, 1]."""
        position = 0

        def read(n: int):
            nonlocal position
            out = frames[position : position + n].astype(np.float32) / 255
            position += n
            return out

        return read

    def square_slices(frames: np.ndarray, padded: tuple[int, int]) -> list:
        """The frames resized with torch (bicubic, antialias) to padded (rows, columns),
        normalised to [-1, 1], bfloat16, (C, t, rows, columns), in the slices model.encode_slices
        gives; made on the device, kept on the CPU, outside the measure."""
        out, start = [], 0
        for size in model.encode_slices(models, frames.shape[0]):
            x = torch.from_numpy(frames[start : start + size]).to(device).permute(0, 3, 1, 2)
            x = x.to(torch.float32).div_(255)
            x = torch.nn.functional.interpolate(x, size=padded, mode="bicubic", antialias=True)
            x = x.clamp_(0, 1).mul_(2).sub_(1).permute(1, 0, 2, 3)
            out.append(x.to(model.COMPUTE_DTYPE).contiguous().cpu())
            start += size
            del x
        return out

    def square(side: int) -> None:
        tiles = Tiles(side, overlap)
        padded = (side, side + 2 * overlap)
        record = {"part": "T2", "kind": "square", "frames": count, **geometry(padded, tiles)}
        record["as_expected"] = record["grid"] == "1x2" and record["largest_tile"] == [side, side]
        record["free_gib"] = free_gib()
        if not fits_the_card(side, record["largest_px"]):
            return
        slices = square_slices(frames, padded)
        record["input"] = {"shape": list(slices[0].shape), "dtype": str(slices[0].dtype)}

        def encode():
            inputs = (x.to(device) for x in slices)
            return tiling.encode_tiled(
                models, inputs, count, tiles, tiling.Scratch(None, allowance)
            )

        def decode(latent):
            return tiling.decode_tiled(models, latent, tiles, tiling.Scratch(None, allowance))

        passes(record, encode, decode)

    def cutoff(res: str, target: tuple[int, int], side: int) -> None:
        tiles = Tiles(side, overlap)
        padded = job.padded_size(target)
        record = {"part": "T2", "kind": "cutoff", "res": res, "target": list(target)}
        record.update(frames=count, **geometry(padded, tiles))
        record["as_expected"] = record["grid"] == "1x2" and record["largest_tile"] == [
            padded[0],
            side,
        ]
        record["free_gib"] = free_gib()
        if not fits_the_card(side, record["largest_px"]):
            return

        def encode():
            inputs = shot.encoder_inputs(models, reader(frames), count, target)
            scratch = tiling.Scratch(None, allowance)
            return tiling.encode_tiled(models, inputs, shot.padded_length(count), tiles, scratch)

        def decode(latent):
            return tiling.decode_tiled(models, latent, tiles, tiling.Scratch(None, allowance))

        passes(record, encode, decode)

    def untiled(res: str, target: tuple[int, int]) -> None:
        padded = job.padded_size(target)
        pixels = padded[0] * padded[1]
        record = {"part": "T2", "kind": "untiled", "res": res, "target": list(target)}
        record.update(frames=count, padded=list(padded), tile=0, overlap=0, tiles=1, grid="1x1")
        record.update(largest_px=pixels, largest_tile=list(padded), tile_sizes=[list(padded)])
        record["as_expected"] = True
        encode_fit, decode_fit = untiled_fits(pixels)
        record["fit_gib"] = [round(encode_fit, 3), round(decode_fit, 3)]
        if cuda and res == "2160p" and not args.tiny:
            # The DiT's weights leave the card for the untiled 4K passes: the encode's fit leaves
            # 3.1 GiB of the 78.37 free beside them on the 96 GB card; the peaks count from what is
            # allocated before each pass either way.
            before = free_gib()
            models.runner.dit.to("cpu")
            record["dit_to_cpu"] = {"free_gib_before": before, "free_gib_after": free_gib()}
        record["free_gib"] = free_gib()
        if cuda and encode_fit * GIB > torch.cuda.mem_get_info()[0] - args.reserve * GIB:
            emit(skipped=True, step=step_name("untiled", res, None), **record)
            return

        def decode_fits():
            if not cuda:
                return True, None
            free = torch.cuda.mem_get_info()[0]
            if decode_fit * GIB <= free - args.reserve * GIB:
                return True, None
            return False, {
                "fit_gib": round(decode_fit, 3),
                "free_gib": round(free / GIB, 3),
                "reserve_gib": args.reserve,
            }

        def encode():
            inputs = shot.encoder_inputs(models, reader(frames), count, target)
            return model.encode_stream(models, inputs, shot.padded_length(count))

        def decode(latent):
            return model.decode_stream(models, latent)

        passes(record, encode, decode, decode_fits)

    def step(name: str, function) -> None:
        """A step, in its own try."""
        nonlocal failed
        began = time.time()
        try:
            function()
            emit(step=name, done=True, seconds=round(time.time() - began, 1))
        except Exception:
            failed += 1
            emit(step=name, failed=True, seconds=round(time.time() - began, 1), error=failure())
        clean()

    library = Path(cil.__file__)
    emit(
        start=time.strftime("%Y-%m-%d %H:%M:%S %z"),
        epoch=round(started),
        parts=["T2"],
        tiny=args.tiny,
        cuda=cuda,
        argv=sys.argv[1:],
        seedvr2x=seedvr2x.__file__,
        causal_inflation_lib=str(library),
        causal_inflation_lib_md5=hashlib.md5(library.read_bytes()).hexdigest(),
        a4=a4,
        one_output_from=getattr(cil, "ONE_OUTPUT_FROM", None),
        alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    )
    began = time.time()
    models = model.load_models(args.dit, args.vae, device)
    if cuda:
        torch.cuda.synchronize()
    vae = models.runner.vae
    names.update(
        (id(module), name)
        for name, module in vae.named_modules()
        if isinstance(module, cil.InflatedCausalConv3d)
    )
    if args.tiny_limits:
        conv_gib, floor = float(args.tiny_limits[0]), int(float(args.tiny_limits[1]))
        vae.set_memory_limit(conv_max_mem=conv_gib, norm_max_mem=0.5)
        if a4:
            cil.ONE_OUTPUT_FROM = floor
    emit(
        part="load",
        seconds=round(time.time() - began, 1),
        device=str(device),
        gpu=torch.cuda.get_device_name(0) if cuda else None,
        allocator=torch.cuda.get_allocator_backend() if cuda else None,
        allocated_gib=round(torch.cuda.memory_allocated() / GIB, 4) if cuda else None,
        card_free_total_gib=[round(x / GIB, 4) for x in torch.cuda.mem_get_info()]
        if cuda
        else None,
        vae_dtype=str(next(models.runner.vae.parameters()).dtype),
        attention=models.attention,
        torch=torch.__version__,
        cudnn=torch.backends.cudnn.version(),
        overlap=overlap,
        encode_slices=model.encode_slices(models, count),
        latent_slices=model.latent_slices(models, 1 + (count - 1) // 4),
        convs=len(names),
        tiny_limits=args.tiny_limits,
        one_output_from=getattr(cil, "ONE_OUTPUT_FROM", None),
    )
    frames = read_frames()
    emit(part="T2", what="setup", frames=list(frames.shape), first_frame=args.first)
    if args.steps:
        for kind, res, side in parse_steps(args.steps):
            name = f"T2-{step_name(kind, res, side)}"
            if kind == "square":
                step(name, partial(square, side))
            elif kind == "cutoff":
                target = dict((r, t) for r, t, _ in cutoffs)[res]
                step(name, partial(cutoff, res, target, side))
            else:
                step(name, partial(untiled, res, untiled_targets[res]))
    else:
        for side in squares:
            step(f"T2-square-{side}", partial(square, side))
        for res, target, sides in cutoffs:
            for side in sides:
                step(f"T2-cutoff-{res}-{side}", partial(cutoff, res, target, side))
    emit(end=True, parts=["T2"], failed_steps=failed, total_seconds=round(time.time() - started, 1))
    return 1 if failed else 0


def geometry(padded: tuple[int, int], tiles) -> dict:
    """The grid seedvr2x lays over frames padded to (rows, columns) pixels (runtime/tilegrid.py)."""
    from seedvr2x.runtime import tilegrid

    layout = tilegrid.grid(tilegrid.latent_size(padded), tiles)
    rows = sorted({(t.top, t.bottom) for t in layout})
    columns = sorted({(t.left, t.right) for t in layout})
    big = max(layout, key=lambda t: t.rows * t.columns)
    return {
        "padded": list(padded),
        "tile": tiles.size,
        "overlap": tiles.overlap,
        "tiles": len(layout),
        "grid": f"{len(rows)}x{len(columns)}",
        "largest_px": tilegrid.largest(padded, tiles),
        "largest_tile": [big.rows * 8, big.columns * 8],
        "tile_sizes": [[t.rows * 8, t.columns * 8] for t in layout],
    }


# ---------------------------------------------------------------- measured


def measured(args: argparse.Namespace) -> int:
    """The planner's figure for each step's two passes, in GiB, from the tree it imports: a
    measured tile's MEASURED, an untiled pass's _fit (no tiles); and whether the pass is one the
    one-output convolution takes."""
    from seedvr2x.runtime import job, planner

    out = {"planner": planner.__file__, "steps": {}}
    for kind, res, side in parse_steps(args.steps):
        name = step_name(kind, res, side)
        if kind == "square":
            key = (side, side)
        elif kind == "cutoff":
            key = (job.padded_size(dict((r, t) for r, t, _ in CUTOFFS)[res])[0], side)
        else:
            key = None
        if key is not None:
            figures = list(planner.MEASURED[key]) if key in planner.MEASURED else [None, None]
            source = "MEASURED" if key in planner.MEASURED else "none"
            padded = list(key)
        else:
            padded = list(job.padded_size(UNTILED[res]))
            figures = [
                round(planner._fit(tuple(padded), None, column) / GIB, 4) for column in (0, 1)
            ]
            source = "planner._fit untiled"
        out["steps"][name] = {
            "key": padded,
            "figures": figures,
            "source": source,
            "a4_list": [on_a4_list(kind, res, side, column) for column in (0, 1)],
        }
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


def on_a4_list(kind: str, res: str | None, side: int | None, column: int) -> bool:
    """Whether the one-output convolution takes this pass by its rule (a convolution's output of
    2.5 GiB or more for a slice of four frames): decodes in squares from 1152 px, in 1104-row
    tiles from 1216 px and untiled at 1080p and 4K; encodes in squares from 1664 px and untiled
    at 4K."""
    if column == 1:
        if kind == "square":
            return side >= 1152
        if kind == "cutoff":
            return res == "1080p" and side >= 1216
        return True
    if kind == "square":
        return side >= 1664
    return kind == "untiled" and res == "2160p"


# ---------------------------------------------------------------- compare


def load_lines(paths: list[Path]) -> tuple[dict, list[dict]]:
    """Every tile line of each step (a list by its step name), and every other line, from run's
    files."""
    tiles: dict = {}
    other: list[dict] = []
    for path in paths:
        if not path.exists():
            continue
        for text in path.read_text().splitlines():
            try:
                r = json.loads(text)
            except ValueError:
                continue
            if r.get("part") == "T2" and "encode" in r:
                if r["kind"] == "square":
                    name = f"square-{r['tile']}"
                elif r["kind"] == "cutoff":
                    name = f"cutoff-{r['res']}-{r['tile']}"
                else:
                    name = f"untiled-{r['res']}"
                tiles.setdefault(name, []).append(r)
            else:
                other.append(r)
    return tiles, other


def compare(args: argparse.Namespace) -> int:
    figures = json.loads(args.measured.read_text())["steps"]
    with_one, with_other = load_lines(args.with_one)
    without, without_other = load_lines(args.without)
    order = [n for n in args.order.split(",") if n] if args.order else list(figures)
    rows, verdicts, differ = [], [], []
    header = [
        "step",
        "pass",
        "on the list",
        "one output (convs/calls)",
        "planner's figure",
        "without",
        "with",
        "with - planner",
        "with - without",
        "outputs equal",
        "s with/without",
    ]
    for name in order:
        w_runs, n_runs = with_one.get(name, []), without.get(name, [])
        fig = figures.get(name, {})
        for column, pass_ in enumerate(PASSES):
            planner_gib = (fig.get("figures") or [None, None])[column]
            listed = (fig.get("a4_list") or [False, False])[column]
            w = w_runs[-1].get(pass_) if w_runs else None
            n = n_runs[-1].get(pass_) if n_runs else None
            # Every measure of the step (a step run again, or the warm-up in every process).
            w_all = [r[pass_]["above_gib"] for r in w_runs if pass_ in r]
            n_all = [r[pass_]["above_gib"] for r in n_runs if pass_ in r]
            key = "latent" if column == 0 else "decode"
            w_dig = [r.get("digests", {}).get(key) for r in w_runs if pass_ in r]
            n_dig = [r.get("digests", {}).get(key) for r in n_runs if pass_ in r]
            digests = set(w_dig) | set(n_dig)
            if not w_dig or not n_dig:
                equal = "n/a"
            elif None in digests:
                equal = "MISSING"
            else:
                equal = "yes" if len(digests) == 1 else f"NO ({len(digests)} digests)"
            took = "n/a"
            if w is not None and "a4_paths" in w:
                p = w["a4_paths"]
                took = f"{len(p['one_output_convs'])}/{p['one_output_calls']}"
                if p["mixed_convs"]:
                    # Both paths, across the grid's tiles: a square's T x T tile takes it
                    # and its T x 192 tile concatenates; never both within one tile's pass.
                    took += f" (both paths across the tiles: {p['mixed_convs']})"
            wg = max(w_all) if w_all else None
            ng = max(n_all) if n_all else None
            over = None if wg is None or planner_gib is None else wg - planner_gib
            diff = None if wg is None or ng is None else wg - ng
            seconds = "/".join("n/a" if x is None else f"{x['seconds']:.0f}" for x in (w, n))
            skipped = ""
            if w_runs and pass_ == "decode" and "decode_skipped" in w_runs[-1]:
                skipped = f" (skipped: fit {w_runs[-1]['decode_skipped']['fit_gib']} GiB)"
            rows.append(
                [
                    name,
                    pass_ + skipped,
                    "yes" if listed else "no",
                    took,
                    "n/a" if planner_gib is None else f"{planner_gib:.3f}",
                    "n/a"
                    if ng is None
                    else f"{ng:.3f}" + (f" ({len(n_all)})" if len(n_all) > 1 else ""),
                    "n/a"
                    if wg is None
                    else f"{wg:.3f}" + (f" ({len(w_all)})" if len(w_all) > 1 else ""),
                    "n/a" if over is None else f"{over:+.3f}",
                    "n/a" if diff is None else f"{diff:+.3f}",
                    equal,
                    seconds,
                ]
            )
            # Run again: a step one code has no line of, or whose outputs differ.
            if not w_runs or not n_runs or equal.startswith("NO") or equal == "MISSING":
                differ.append(name)
            if listed:
                verdicts.append((name, pass_, wg, planner_gib, over, equal, took))
    if args.compact:
        # One line a step: each pass's planner / without / with (with - planner, with - without),
        # its outputs equal, what of the one-output path the run with it saw taken.
        for i in range(0, len(rows), 2):
            parts = [rows[i][0]]
            for row in rows[i : i + 2]:
                parts.append(
                    f"{row[1]}{' [list]' if row[2] == 'yes' else ''}: planner {row[4]},"
                    f" without {row[5]}, with {row[6]} ({row[7]} planner, {row[8]} without),"
                    f" equal {row[9]}, one output {row[3]}, s {row[10]}"
                )
            print(" | ".join(parts))
        return 0
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(str(c) for c in row) + " |")
    print()
    missing = [(n, p) for n, p, wg, *_ in verdicts if wg is None]
    over_rows = [(n, p, o) for n, p, wg, pg, o, *_ in verdicts if o is not None and o > 0]
    unequal = [(n, p, e) for n, p, wg, pg, o, e, t in verdicts if e != "yes"]
    not_taken = [
        (n, p) for n, p, wg, pg, o, e, t in verdicts if t in ("n/a",) or t.startswith("0/")
    ]
    print(f"passes on the list: {len(verdicts)}; measured with it: {len(verdicts) - len(missing)}")
    print(
        f"  over the planner's figure with it: {len(over_rows)}"
        + "".join(f"\n    {n} {p} {o:+.3f} GiB" for n, p, o in over_rows)
    )
    print(
        f"  outputs not equal (or unmeasured): {len(unequal)}"
        + "".join(f"\n    {n} {p}: {e}" for n, p, e in unequal)
    )
    print(
        f"  the one-output path not seen taken: {len(not_taken)}"
        + "".join(f"\n    {n} {p}" for n, p in not_taken)
    )
    failures = [r for r in with_other + without_other if r.get("failed") or r.get("skipped")]
    for r in failures:
        print(f"  FAILED or SKIPPED: {json.dumps(r)[:400]}")
    if args.differ:
        args.differ.write_text("".join(f"{n}\n" for n in dict.fromkeys(differ)))
    return 0


# ---------------------------------------------------------------- table

SHORT_CONVS = {
    "decoder.up_blocks.1.upsamplers.0.conv": "d.ub1.up",
    "decoder.up_blocks.2.upsamplers.0.conv": "d.ub2.up",
}


def short_conv(name: str) -> str:
    """A convolution's name in the VAE, shortened for a table's cell."""
    if name in SHORT_CONVS:
        return SHORT_CONVS[name]
    name = name.replace("decoder.up_blocks.", "d.ub").replace("encoder.down_blocks.", "e.db")
    return (
        name.replace(".resnets.", ".r")
        .replace(".conv_shortcut", ".sc")
        .replace(".downsamplers.0.conv", ".down")
    )


def read_runs(paths: list[Path]) -> dict:
    """Every tile's line by step name, each with the file it came from, in the order read."""
    out: dict = {}
    for path in paths:
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get("part") == "T2" and "encode" in r:
                if r["kind"] == "square":
                    name = f"square-{r['tile']}"
                elif r["kind"] == "cutoff":
                    name = f"cutoff-{r['res']}-{r['tile']}"
                else:
                    name = f"untiled-{r['res']}"
                out.setdefault(name, []).append((path.name, r))
    return out


def final_table(args: argparse.Namespace) -> int:
    """The final table from the records' bytes (the peaks from `above`, not the rounded GiB)."""
    measured_steps = json.loads(args.measured.read_text())["steps"]
    order = list(measured_steps)
    W, N = read_runs(sorted(args.with_one)), read_runs(sorted(args.without))

    print(
        "| step | pass | list | planner | without | with | with-planner | with-without |"
        " outputs equal | one output: convs (whole/judged calls per conv) |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    controls = []
    for name in order:
        fig = measured_steps[name]
        for col, p in enumerate(PASSES):
            key = "latent" if p == "encode" else "decode"
            w = [r for _, r in W.get(name, []) if p in r]
            n = [r for _, r in N.get(name, []) if p in r]
            if not w and not n:
                skipped = W.get(name, [[None, {}]])[-1][1].get("decode_skipped")
                print(
                    f"| {name} | {p} | {fig['a4_list'][col]} | {fig['figures'][col]} | not run"
                    f" | not run | | | | skipped: {skipped} |"
                )
                continue
            wb = max(r[p]["above"] for r in w)
            nb = max(r[p]["above"] for r in n)
            dig = {r["digests"].get(key) for r in w + n}
            slices = set()
            if p == "decode":
                for r in w + n:
                    slices.add(tuple(s[3] for s in r["digests"]["decode_slices"]))
            equal = (
                "yes"
                if len(dig) == 1 and None not in dig and len(slices) <= 1
                else f"NO {len(dig)}"
            )
            took = w[-1][p].get("a4_paths_by_conv", {})
            whole = {c: v for c, v in took.items() if v[0]}
            convs = (
                ", ".join(f"{short_conv(c)} {v[0]}/{v[0] + v[1]}" for c, v in sorted(whole.items()))
                or "none"
            )
            f = fig["figures"][col]
            print(
                f"| {name} | {p} | {'yes' if fig['a4_list'][col] else 'no'} | {f:.3f}"
                f" | {nb / GIB:.3f}{' x' + str(len(n)) if len(n) > 1 else ''}"
                f" | {wb / GIB:.3f}{' x' + str(len(w)) if len(w) > 1 else ''}"
                f" | {wb / GIB - f:+.3f} | {(wb - nb) / GIB:+.3f}"
                f" | {equal} ({len(w)}+{len(n)} runs) | {convs} |"
            )
            if not fig["a4_list"][col]:
                controls.append(
                    (
                        name,
                        p,
                        wb == nb,
                        min(r[p]["above"] for r in w) == max(r[p]["above"] for r in n),
                    )
                )

    print()
    print(
        "controls (passes off the list): peak bytes equal with/without:",
        sum(c[2] for c in controls),
        "of",
        len(controls),
        [c[:2] for c in controls if not c[2]],
    )
    warm_w, warm_n = list(W["square-512"]), list(N["square-512"])
    print("warm-up square-512 runs:", len(warm_w), "with,", len(warm_n), "without")
    for p, key in (("encode", "latent"), ("decode", "decode")):
        peaks = {r[p]["above"] for _, r in warm_w + warm_n}
        digs = {r["digests"][key] for _, r in warm_w + warm_n}
        found = sorted(d[:16] for d in digs)
        print(f"  {p}: peak bytes {sorted(peaks)}; digests {len(digs)} distinct: {found}")
    slice_sets = {tuple(s[3] for s in r["digests"]["decode_slices"]) for _, r in warm_w + warm_n}
    print("  decode slices digest sets distinct:", len(slice_sets))
    for code, rr in (("with", W), ("without", N)):
        for f, r in rr.get("untiled-2160p", []):
            print(
                "untiled-2160p",
                code,
                f,
                "dit_to_cpu",
                r.get("dit_to_cpu"),
                "free",
                r.get("free_gib"),
                "fit",
                r.get("fit_gib"),
                "decode_skipped",
                r.get("decode_skipped"),
                "encode above",
                r["encode"]["above"],
                "s",
                r["encode"]["seconds"],
            )
    if "untiled-2160p" in W:
        print(
            "untiled-2160p encode, one output (with):",
            W["untiled-2160p"][-1][1]["encode"].get("a4_paths_by_conv"),
        )
    for log in sorted(args.logs):
        text = log.read_text()
        recovered = len(re.findall(r"recovered from an allocation failure", text))
        apps = re.findall(r"^compute app before: .*$", text, re.M)
        gpu = re.findall(r"^gpu before: .*$", text, re.M)
        released = re.findall(r"^released .*$", text, re.M)
        print(
            log.name,
            "recovered",
            recovered,
            "apps before",
            apps or "none",
            gpu[0] if gpu else "",
            released[0] if released else "",
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    sub = commands.add_parser("run", help="measure the tiles' peaks (GPU)")
    sub.add_argument("--dit", type=Path, required=True, help="the DiT's file (7B fp16)")
    sub.add_argument("--vae", type=Path, required=True, help="the VAE's file")
    sub.add_argument("--source", type=Path, required=True, help="a clip")
    sub.add_argument("--first", type=int, default=0, help="the first frame taken of the source")
    sub.add_argument("--out", type=Path, required=True, help="the JSON lines, appended")
    sub.add_argument("--reserve", type=float, default=1.5, help="GiB left free on the card")
    sub.add_argument("--tiny", action="store_true", help="tiny sizes, for a debug run on the CPU")
    sub.add_argument("--steps", help="STEP,... only those, in that order (the docstring)")
    sub.add_argument("--tiny-limits", nargs=2, metavar=("CONV_GIB", "ONE_OUTPUT_BYTES"))
    sub.set_defaults(function=run)
    sub = commands.add_parser("measured", help="the planner's figures for the steps, no GPU")
    sub.add_argument("--steps", required=True)
    sub.add_argument("--out", type=Path, required=True)
    sub.set_defaults(function=measured)
    sub = commands.add_parser("compare", help="the table of runs with and without the one output")
    sub.add_argument("--with", dest="with_one", type=Path, nargs="+", required=True)
    sub.add_argument("--without", type=Path, nargs="+", required=True)
    sub.add_argument("--measured", type=Path, required=True)
    sub.add_argument("--order", help="the steps' names in the table's order")
    sub.add_argument("--differ", type=Path, help="write the steps whose outputs differ here")
    sub.add_argument("--compact", action="store_true", help="one line a step, no summary")
    sub.set_defaults(function=compare)
    sub = commands.add_parser("table", help="the final table from the records' bytes")
    sub.add_argument("--with", dest="with_one", type=Path, nargs="+", required=True)
    sub.add_argument("--without", type=Path, nargs="+", required=True)
    sub.add_argument("--measured", type=Path, required=True)
    sub.add_argument("--logs", type=Path, nargs="*", default=[], help="the runs' log files")
    sub.set_defaults(function=final_table)
    args = parser.parse_args()
    return args.function(args)


if __name__ == "__main__":
    sys.exit(main())

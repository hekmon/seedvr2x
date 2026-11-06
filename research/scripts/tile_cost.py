#!/usr/bin/env python3
"""What VAE tiles cost a picture or a clip: a tiled output against the untiled one, the tiles' seams
looked for where they are.

  tile_cost.py --reference REF --out FILE.json [--side PX | --encode-side PX --decode-side PX]
               [--map DIR] [--label NAME] TEST

REF and TEST are each one PNG (a picture, 8 or 16 bits) or a directory of PNG frames (a clip's
output; seedvr2x's --format png writes seg_NNNNNN/ directories of them, all of the directory's
PNGs are taken, in the order of their relative names). A count that differs is refused, and so
is a size. REF is the untiled output, TEST the tiled one, both through the same correction.
Pixels are read exactly (seedvr2x.picture.read: TorchCodec for 16 bits, which Pillow cuts to 8)
and scaled to 8-bit levels, 0 to 255, in float; no rounding.

--side gives both passes' tiles, --encode-side and --decode-side each pass's own; 0 or absent is
an untiled pass, which has no seam. The seam bands are the tiles' overlap zones in the output's
pixel space: seedvr2x's grid (seedvr2x/tilegrid.py, tiles of N px every N - 64 from the top left,
over the padded frame of frames.padded_size), the zone between two consecutive tiles being
[start of the next, end of the first), each widened by BAND (16 px) on each side and cut to the
picture; both passes' bands are united. Both passes' tiles sit in the same pixels: the padding is
reflected and black rows at the bottom and columns at the right, and the decode is cropped to
the target from its top left (shot_pass._decoded, still.decoded), so a tile's pixel is the
output's own, with no offset. The padded frame is the output's size padded (a target with an odd
side, cropped to even, is padded as its even size).

Figures, per frame (8-bit levels; luma is BT.709 Y' from R'G'B', colour_diag.luma8's formula,
0.2126 R + 0.7152 G + 0.0722 B), then the mean, the min and the max over the frames, and the
per-frame lists:
- drift: the largest |mean luma difference| of a 60x60 block, tiled minus untiled, over all
  blocks (colour.md, "VAE tiles at 1080p": colour_eval.py's BLOCK, block_means and tiles_table's
  "max |block dY|"; the blocks tile the frame from the top left, the edge remainder left out)
- drift_flat: the same over the flattest third of blocks, by the mean luma gradient of each
  60x60 block of the REFERENCE (the colour study used the ground truth's; the untiled output
  here, there being none). The gradient is colour_eval.py's edge_mask: the Sobel (3x3, edges
  reflected without repeating the border, as OpenCV's default) magnitude of the luma; the third
  is its 33rd percentile of the frame's blocks (tiles_table), blocks at or under it
- psnr_y: PSNR of the luma over the whole frame, peak 255 (colour_eval.py's psnr_y); identical
  frames give 100 dB
- psnr_y_seams, psnr_y_inside: the same inside the seam bands and outside them;
  seam_error_share: the share of the frame's total squared luma error that sits in the bands,
  to compare with seam_pixel_share, the share of pixels they are (a share of error well over
  the share of pixels is a seam; equal, the error is spread evenly)
- step: colour.md's step across an overlap zone ("Through seedvr2x's split, for the eyes": across
  one overlap zone, the flattest non-black one, on the frame where the difference steps most
  across it). colour_eval.py holds no code for it; it is made here from those words. A candidate
  is one zone cut into segments of 60 px along it; its step in a frame is |mean of the luma
  difference over the BAND px after the zone - mean of it over the BAND px before it| on the
  segment (levels of difference, tiled minus untiled; the zone is not in either strip). Flat
  and non-black are judged on the reference, over the segment's strip (BAND before, the zone,
  BAND after), by the mean gradient above and the mean luma (black: under BLACK_LEVEL = 8
  levels), both taken over the frames. step_zone gives the flattest non-black candidate, step
  its figure per frame (its mean over the frames, and max: the frame where it steps most);
  step_worst is the largest step over all the non-black candidates, per frame. Only zones
  whose strips lie in the picture count
- max_abs, p999: the largest |luma difference| of a pixel, and its 99.9th percentile

--map DIR writes, named after --label, the maps of the frame with the largest drift (a picture's
own): LABEL_diff.png, the |luma difference| averaged over 8x8 blocks, 2 levels white, and
LABEL_blocks.png, the 60x60 blocks' signed mean difference, mid grey (128) zero, 32 grey levels a
level of difference, clipped. Both are a quarter of the frame's size, 8-bit grey; the seam
bands' edges are drawn at grey 64 where the map is darker than that. The JSON holds the figures,
the sides, the frame count, the size, the seam bands and the seconds taken; one line is printed.

Written on 2026-11-05 for place-study (the night's Q11: the tiles' cost to the picture at 2048,
1024, 768 and 512 px)."""

import argparse
import json
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path

import numpy as np
import numpy.typing as npt

from seedvr2x import frames, picture, tilegrid
from seedvr2x.passes import OVERLAP, Tiles

BLOCK = 60  # colour_eval.py's BLOCK
FLAT_PERCENTILE = 33  # tiles_table's flattest third
BAND = 16  # px each side of an overlap zone: the seam band, and the step's strips
BLACK_LEVEL = 8.0  # a segment whose reference luma averages under this is black
MAX_PSNR = 100.0
LOOKAHEAD = 4  # frame pairs read ahead of the one being measured
FigureArray = npt.NDArray[np.float32]


def luma8(rgb: FigureArray) -> FigureArray:
    """BT.709 luma of (H, W, 3) in [0, 1] in 8-bit levels: colour_diag.luma8."""
    y = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return (y * np.float32(255)).astype(np.float32, copy=False)


def block_means(x: FigureArray, block: int = BLOCK) -> FigureArray:
    """Means of the block x block blocks from the top left: colour_eval.block_means."""
    h, w = x.shape[:2]
    hb, wb = h // block, w // block
    cut = x[: hb * block, : wb * block]
    return cut.reshape(hb, block, wb, block).mean(axis=(1, 3), dtype=np.float64).astype(np.float32)


def gradient(y: FigureArray) -> FigureArray:
    """The Sobel (3x3) gradient magnitude of the luma, edges reflected without repeating the
    border (OpenCV's default), as colour_eval.edge_mask takes it."""
    p = np.pad(y, 1, mode="reflect")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    return np.hypot(gx, gy)


def psnr(sse: float, count: int) -> float | None:
    if count == 0:
        return None
    if sse == 0:
        return MAX_PSNR
    return min(MAX_PSNR, float(10 * np.log10(255.0**2 * count / sse)))


# ------------------------------------------------------------------------------ the seams


Zone = tuple[str, int, int]  # ("x": columns / "y": rows, start px, end px)


def zones(size: tuple[int, int], side: int, overlap: int = OVERLAP) -> list[Zone]:
    """The overlap zones, in pixels, of tiles of `side` px over a picture of size (height, width)
    padded as the model takes it: between two consecutive tiles, from the start of the next to
    the end of the first. No zones for side 0 (untiled), nor for a tile covering the frame."""
    if side <= 0:
        return []
    padded = frames.padded_size(size)
    tiles = tilegrid.grid(tilegrid.latent_size(padded), Tiles(side, overlap))
    rows = sorted({(t.top, t.bottom) for t in tiles})
    columns = sorted({(t.left, t.right) for t in tiles})
    found: list[Zone] = []
    for axis, spans in (("y", rows), ("x", columns)):
        for first, following in pairwise(spans):
            found.append((axis, following[0] * tilegrid.SCALE, first[1] * tilegrid.SCALE))
    return found


def band_mask(size: tuple[int, int], found: list[Zone]) -> npt.NDArray[np.bool_]:
    """(H, W) True inside the zones widened by BAND px each side, cut to the picture."""
    height, width = size
    mask = np.zeros((height, width), dtype=bool)
    for axis, start, end in found:
        lo, hi = max(0, start - BAND), end + BAND
        if axis == "x":
            mask[:, lo : min(hi, width)] = True
        else:
            mask[lo : min(hi, height), :] = True
    return mask


def inside(size: tuple[int, int], zone: Zone) -> bool:
    """Whether the strips of BAND px either side of the zone lie in the picture."""
    axis, start, end = zone
    extent = size[1] if axis == "x" else size[0]
    return start - BAND >= 0 and end + BAND <= extent and end - start > 0


def segments(arr: FigureArray, zone: Zone, lo: int, hi: int) -> FigureArray:
    """Means, over rows (columns) lo to hi of `arr` across the zone's axis, of each BLOCK px
    segment along the zone, from the picture's top (left): (segments,)."""
    view = arr[:, lo:hi] if zone[0] == "x" else arr[lo:hi, :].T
    count = view.shape[0] // BLOCK
    cut = view[: count * BLOCK]
    return cut.reshape(count, BLOCK, -1).mean(axis=(1, 2), dtype=np.float64).astype(np.float32)


def strip_steps(diff: FigureArray, found: list[Zone]) -> FigureArray:
    """|mean of diff over the BAND px after each zone - mean over the BAND px before| for every
    segment of every zone in `found`, in order: (candidates,)."""
    out = []
    for zone in found:
        _, start, end = zone
        after = segments(diff, zone, end, end + BAND)
        before = segments(diff, zone, start - BAND, start)
        out.append(np.abs(after - before))
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def strip_reference(arr: FigureArray, found: list[Zone]) -> FigureArray:
    """The mean of `arr` over each segment's whole strip (BAND, the zone, BAND), as strip_steps'
    candidates."""
    out = [segments(arr, zone, zone[1] - BAND, zone[2] + BAND) for zone in found]
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def candidate_names(found: list[Zone], size: tuple[int, int]) -> list[tuple[str, int, int, int]]:
    """(axis, zone start, zone end, segment's first px along the zone) of each candidate."""
    names = []
    for zone in found:
        length = size[0] if zone[0] == "x" else size[1]
        names += [(zone[0], zone[1], zone[2], n * BLOCK) for n in range(length // BLOCK)]
    return names


# ------------------------------------------------------------------------------ reading


def listing(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise SystemExit(f"tile_cost: {path} is neither a file nor a directory")
    files = sorted(
        (p for p in path.rglob("*") if p.suffix.lower() == ".png" and p.is_file()),
        key=lambda p: p.relative_to(path).as_posix(),
    )
    if not files:
        raise SystemExit(f"tile_cost: no PNG in {path}")
    return files


def luma_of(path: Path) -> FigureArray:
    return luma8(picture.read(path.read_bytes(), path.name).pixels)


# ------------------------------------------------------------------------------ the maps


def outline(grey: npt.NDArray[np.uint8], found: list[Zone], scale: int) -> None:
    """Draws the bands' edges, at `scale` px of the picture to a pixel of the map, in grey 64 where
    the map is darker."""
    height, width = grey.shape
    for axis, start, end in found:
        for edge in (start - BAND, end + BAND):
            at = edge // scale
            if axis == "x" and 0 <= at < width:
                grey[:, at] = np.maximum(grey[:, at], 64)
            elif axis == "y" and 0 <= at < height:
                grey[at, :] = np.maximum(grey[at, :], 64)


def write_maps(
    directory: Path, label: str, ref: FigureArray, test: FigureArray, found: list[Zone]
) -> list[str]:
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    diff = test - ref
    # |difference| in 8x8 blocks, each shown 2 px square: a quarter of the frame
    h8, w8 = diff.shape[0] // 8, diff.shape[1] // 8
    small = np.abs(diff[: h8 * 8, : w8 * 8]).reshape(h8, 8, w8, 8).mean(axis=(1, 3))
    grey = np.clip(small / 2 * 255, 0, 255).round().astype(np.uint8)
    grey = np.repeat(np.repeat(grey, 2, axis=0), 2, axis=1)
    outline(grey, found, 4)
    first = directory / f"{label}_diff.png"
    Image.fromarray(grey, "L").save(first)
    means = block_means(diff)
    signed = np.clip(128 + means * 32, 0, 255).round().astype(np.uint8)
    signed = np.repeat(np.repeat(signed, BLOCK // 4, axis=0), BLOCK // 4, axis=1)
    outline(signed, found, 4)
    second = directory / f"{label}_blocks.png"
    Image.fromarray(signed, "L").save(second)
    return [first.name, second.name]


# ------------------------------------------------------------------------------ the figures


def summary(values: list[float | None]) -> dict[str, object]:
    held = [v for v in values if v is not None]
    return {
        "mean": float(np.mean(held)) if held else None,
        "min": float(np.min(held)) if held else None,
        "max": float(np.max(held)) if held else None,
        "per_frame": [None if v is None else round(float(v), 6) for v in values],
    }


def measure(
    reference: Path,
    test: Path,
    encode_side: int,
    decode_side: int,
    map_directory: Path | None,
    label: str,
) -> dict[str, object]:
    started = time.monotonic()
    ref_files, test_files = listing(reference), listing(test)
    if len(ref_files) != len(test_files):
        raise SystemExit(
            f"tile_cost: {len(ref_files)} reference frames and {len(test_files)} test frames"
        )
    count = len(ref_files)

    found: list[Zone] = []
    size: tuple[int, int] | None = None
    mask: npt.NDArray[np.bool_] | None = None
    candidates: list[tuple[str, int, int, int]] = []
    keep: list[Zone] = []

    series: dict[str, list[float | None]] = {
        k: []
        for k in (
            "drift",
            "drift_flat",
            "psnr_y",
            "psnr_y_seams",
            "psnr_y_inside",
            "seam_error_share",
            "max_abs",
            "p999",
        )
    }
    grad_rows: list[FigureArray] = []
    luma_rows: list[FigureArray] = []
    step_rows: list[FigureArray] = []
    worst: tuple[float, int] = (-1.0, 0)
    held: tuple[FigureArray, FigureArray] | None = None

    with ThreadPoolExecutor(max_workers=2) as pool:
        pending: deque[tuple[Future[FigureArray], Future[FigureArray]]] = deque()
        queue = iter(zip(ref_files, test_files, strict=True))

        def fill() -> None:
            while len(pending) < LOOKAHEAD:
                pair = next(queue, None)
                if pair is None:
                    return
                pending.append((pool.submit(luma_of, pair[0]), pool.submit(luma_of, pair[1])))

        fill()
        for index in range(count):
            ref_future, test_future = pending.popleft()
            fill()
            ref, tst = ref_future.result(), test_future.result()
            if ref.shape != tst.shape:
                raise SystemExit(
                    f"tile_cost: frame {index} is {ref.shape[1]}x{ref.shape[0]} in the reference"
                    f" and {tst.shape[1]}x{tst.shape[0]} in the test"
                )
            if size is None:
                size = (ref.shape[0], ref.shape[1])
                found = sorted(set(zones(size, encode_side) + zones(size, decode_side)))
                mask = band_mask(size, found)
                keep = [z for z in found if inside(size, z)]
                candidates = candidate_names(keep, size)
            elif size != ref.shape:
                raise SystemExit(f"tile_cost: frame {index} changes the size")
            assert mask is not None

            diff = tst - ref
            absolute = np.abs(diff)
            blocks = block_means(diff)
            drift_blocks = np.abs(blocks)
            grad = block_means(gradient(ref))
            flat = grad <= np.percentile(grad, FLAT_PERCENTILE)
            drift = float(drift_blocks.max())
            series["drift"].append(drift)
            series["drift_flat"].append(float(drift_blocks[flat].max()))

            squares = absolute.astype(np.float64) ** 2
            total = float(squares.sum())
            series["psnr_y"].append(psnr(total, squares.size))
            if mask.any():
                banded = float(squares[mask].sum())
                series["psnr_y_seams"].append(psnr(banded, int(mask.sum())))
                series["psnr_y_inside"].append(psnr(total - banded, int((~mask).sum())))
                series["seam_error_share"].append(banded / total if total > 0 else None)
            else:
                for k in ("psnr_y_seams", "psnr_y_inside", "seam_error_share"):
                    series[k].append(None)
            series["max_abs"].append(float(absolute.max()))
            series["p999"].append(float(np.percentile(absolute, 99.9)))

            if keep:
                step_rows.append(strip_steps(diff, keep))
                grad_rows.append(strip_reference(gradient(ref), keep))
                luma_rows.append(strip_reference(ref, keep))
            if drift > worst[0]:
                worst = (drift, index)
                held = (ref, tst) if map_directory is not None else None

    assert size is not None and mask is not None
    out: dict[str, object] = {
        "label": label,
        "reference": str(reference),
        "test": str(test),
        "encode_side": encode_side,
        "decode_side": decode_side,
        "overlap": OVERLAP,
        "frames": count,
        "size": [size[1], size[0]],
        "padded": list(frames.padded_size(size)[::-1]),
        "seam_band_px": BAND,
        "seam_pixel_share": float(mask.mean()),
        "zones": [{"axis": a, "start": s, "end": e} for a, s, e in found],
    }
    for key, values in series.items():
        out[key] = summary(values)

    if step_rows:
        steps = np.stack(step_rows)  # (frames, candidates)
        grads = np.stack(grad_rows).mean(axis=0)
        lumas = np.stack(luma_rows).mean(axis=0)
        usable = lumas >= BLACK_LEVEL
        if usable.any():
            chosen = int(np.flatnonzero(usable)[np.argmin(grads[usable])])
            axis, start, end, along = candidates[chosen]
            step = summary(list(steps[:, chosen]))
            step["zone"] = {"axis": axis, "start": start, "end": end, "segment_from": along}
            step["reference_gradient"] = float(grads[chosen])
            step["frame_of_max"] = int(np.argmax(steps[:, chosen]))
            out["step"] = step
            out["step_worst"] = summary(list(steps[:, usable].max(axis=1)))
        else:
            out["step"] = out["step_worst"] = None
    else:
        out["step"] = out["step_worst"] = None

    if map_directory is not None and held is not None:
        out["maps"] = write_maps(map_directory, label, held[0], held[1], found)
        out["maps_frame"] = worst[1]
    out["seconds"] = round(time.monotonic() - started, 2)
    return out


def line(result: dict[str, object]) -> str:
    def mean(key: str) -> float | None:
        value = result[key]
        return value["mean"] if isinstance(value, dict) else None

    def show(value: float | None, digits: int = 2) -> str:
        return "-" if value is None else f"{value:.{digits}f}"

    step = result["step"]
    step_max = step["max"] if isinstance(step, dict) else None
    return (
        f"{result['label']}: {result['frames']} frames"
        f" {result['size'][0]}x{result['size'][1]}"  # type: ignore[index]
        f", drift {show(mean('drift'))} (flat {show(mean('drift_flat'))}) levels"
        f", PSNR-Y {show(mean('psnr_y'))} dB (seams {show(mean('psnr_y_seams'))}"
        f", inside {show(mean('psnr_y_inside'))})"
        f", seams hold {show(mean('seam_error_share'), 3)} of the error in"
        f" {show(result['seam_pixel_share'], 3)} of the pixels"  # type: ignore[arg-type]
        f", step max {show(step_max)}, max |dY| {show(mean('max_abs'))}"
        f", {result['seconds']} s"
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="tile_cost.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("test", metavar="TEST", type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--side", type=int, default=0)
    parser.add_argument("--encode-side", type=int, default=None)
    parser.add_argument("--decode-side", type=int, default=None)
    parser.add_argument("--map", type=Path, default=None)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()
    encode = args.side if args.encode_side is None else args.encode_side
    decode = args.side if args.decode_side is None else args.decode_side
    label = args.label or args.test.resolve().name
    try:
        result = measure(args.reference, args.test, encode, decode, args.map, label)
    except ValueError as error:
        raise SystemExit(f"tile_cost: {error}") from error
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(line(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Command line: options and logging."""

import argparse
import datetime
import fcntl
import logging
import math
import os
import re
import shlex
import shutil
import sys
import time
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from importlib.metadata import version
from itertools import accumulate
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from seedvr2x.media.conversion import MATRICES
from seedvr2x.media.ffmpeg import MediaError
from seedvr2x.media.writer import FORMATS
from seedvr2x.runtime.cuts import PLAYLISTS, POSSIBLE, THRESHOLD, playlist_path, write_playlists
from seedvr2x.runtime.job import MIN_SEGMENT
from seedvr2x.runtime.pull import DETECTOR, REPO
from seedvr2x.runtime.stop import Stop, Stopped, Terminated, terminable

if TYPE_CHECKING:
    import torch

    from seedvr2x.media.index import FrameIndex
    from seedvr2x.media.source import Declared, FirstPass, Source
    from seedvr2x.runtime.cuts import Cut
    from seedvr2x.runtime.job import JobError, OutputSegment, Shot
    from seedvr2x.runtime.manifest import Manifest
    from seedvr2x.runtime.plan import Plan
    from seedvr2x.runtime.pull import ModelFile, ModelFiles

# Video file names, other than Matroska's, that -o refuses: the finished file is a .mkv path, the
# one container the output segments are assembled into (_output), and anything else names the
# directory of the output segments, which such a name would only hide.
VIDEO_SUFFIXES = frozenset(
    {".mp4", ".mov", ".m4v", ".avi", ".webm", ".ts", ".m2ts", ".mts", ".mxf", ".nut", ".mpg"}
)

logger = logging.getLogger("seedvr2x")

# numz's padding of the frames before the model, DivisiblePad((16, 16)), zeros up to multiples of
# 16, in place of seedvr2x's (DESIGN.md, Pipeline step 0), for seedvr2x's GPU tests only, never a
# user's: milestone 1's regression holds the output to numz's own with it. It runs the CLI in a
# process of its own, the allocator being set before torch is imported (runtime/memory.py's
# startup, from _run), so an environment variable reaches it, and no option shows it to users.
# Read once (_run), passed down as a parameter, and recorded in an output directory's settings,
# so that a job is never resumed in the other padding.
NUMZ_PADDING = "SEEDVR2X_TESTS_NUMZ_PADDING"
# numz's draw of each shot's noise, on the GPU, whose generator follows the card (DESIGN.md,
# Pipeline step 2), in place of seedvr2x's, on the CPU: for seedvr2x's GPU tests only, never a
# user's, as NUMZ_PADDING, and read, passed down and recorded the same way, so that a job is
# never resumed in the other draw. Milestone 1's regression holds the output to numz's own with
# it, and every GPU test held to a reference numz made. PROVISIONAL (implementation,
# 2026-10-16): the planner counts nothing of that draw, the shot's latent and noise whole on the
# GPU before its first window, beside the DiT's phase's weights: 2,088,960 bytes a latent in
# numz's padding at 1080p, 25,067,520 for milestone 1's 12 latents, by the tensors' sizes.
NUMZ_DRAW = "SEEDVR2X_TESTS_NUMZ_DRAW"
# numz's padding of a shot's end, the frames before its last one mirrored (generation_utils.py:
# 651-656 at 4490bd1), in place of seedvr2x's, the last frame repeated as ByteDance does
# (DESIGN.md, Pipeline step 1, decided on 2026-10-19): for seedvr2x's GPU tests only, never a
# user's, as NUMZ_PADDING, and read, passed down and recorded the same way, so that a job is
# never resumed in the other tail. Milestone 1's regression runs with it, so that its cases stay
# numz's whatever length a reference is made again with: today one shot of 45 frames, 4n + 1,
# which neither pads.
NUMZ_TAIL = "SEEDVR2X_TESTS_NUMZ_TAIL"


def main(argv: list[str] | None = None) -> int:
    """Run seedvr2x with argv (sys.argv[1:] when None) and return the exit status."""
    # Hugging Face's telemetry off unless the user set it (DESIGN.md, Weights). huggingface_hub
    # reads the variable once, when imported (constants.py:244-248 in 1.33.0), by runtime/pull.py's
    # lookup in the cache or by diffusers, which the vendored model imports: both after this line,
    # where every entry starts (seedvr2x, seedvr2x verify, python -m seedvr2x). Without it, every
    # request the library makes names torch's version and the AI agent it detects running it, from a
    # registry it fetches from the Hub's /api/agent-harnesses at most once a day and writes to
    # HF_HOME/.agent_harnesses.json (utils/_headers.py:183-189, utils/_detect_agent.py:143-163,
    # 180-203).
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["verify"]:
        return verify(argv[1:])
    parser = argparse.ArgumentParser(
        prog="seedvr2x",
        description="SeedVR2 video upscaler for long runs.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {version('seedvr2x')}")
    parser.add_argument("input", type=Path, help="the source, one video file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="a .mkv path: the finished file, the upscaled video with the source's other streams"
        " (audio, subtitles, chapters, attachments), assembled from the output segments once"
        " they are all done; they and the job's other files wait in PATH.work beside it, removed"
        " once the file is whole, and the same command resumes a job stopped before. Else a new"
        " or empty directory: the output segments (FFV1 files, or PNG directories) and their"
        " manifest, the source cut at its shots, merged to --min-segment, never assembled",
    )
    # yuv420p10le by default (DESIGN.md, Output): what every encoder takes, converted here by
    # zscale, exactly, rather than later by sptenc's swscale (16-bit white at 943, not 940). The
    # sizes per hour of 1080p at 24000/1001: DESIGN.md, Output, Master sizes (milestone 5).
    parser.add_argument(
        "--format",
        choices=FORMATS,
        default="yuv420p10le",
        help="output format (default: %(default)s): yuv420p10le, FFV1 master in 10-bit YUV 4:2:0"
        " (BT.709 at HD sizes, limited range), what encoders take, converted here exactly by"
        " zscale (white at 940), about 100-135 GiB per hour of 1080p; gbrp16le, FFV1 master in"
        " 16-bit RGB, for precision work, tests and scoring on short samples, 540-690 GiB per hour"
        " of 1080p with colour correction; png, 16-bit PNG",
    )
    # DESIGN.md, Output, Writer: seedvr2x owns no encoder flags; how the output is compressed
    # during the run is the user's own command, one run per output segment.
    parser.add_argument(
        "--segment-cmd",
        metavar="CMD",
        help="write each output segment with the command CMD, run by /bin/sh once per segment, in"
        " place of the FFV1 writer: it reads the segment on stdin, lossless and tagged, in the"
        " master's format (the FFV1 stream --format names, in Matroska: yuv420p10le by default,"
        " so that an encoder converts nothing), and writes the file seedvr2x names, which {out}"
        " stands for, quoted for the shell: --segment-cmd 'ffmpeg -i - -c:v libx265 -crf 16"
        " {out}'. seedvr2x never parses CMD: it tries it on a few black frames before any work,"
        " then checks its exit status and the frames its file holds, and keeps no checksums of"
        " them. A directory -o then holds what CMD wrote, and a .mkv path is assembled from it;"
        " a failing CMD stops the run at its segment, which the same command resumes. A setting:"
        " another CMD is another job. Refused with --format png",
    )
    # Optional (DESIGN.md, Weights, decided 2026-10-09): without it, seedvr2x's own files come
    # from Hugging Face's cache, pulled from its repo when missing (runtime/pull.py).
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="a directory holding the model files, each read there by its name, the shot"
        f" detector's {DETECTOR} among them unless --cuts is given; nothing is downloaded."
        " Without it, --dit-model and --vae-model name seedvr2x's own files, taken from Hugging"
        f" Face's cache (HF_HOME moves it) with the shot detector's, downloaded into it from {REPO}"
        " at a pinned revision when missing (HF_HUB_OFFLINE keeps the network out). Either way,"
        " seedvr2x's own files are checked by their pinned size and SHA-256",
    )
    # The sharp 7B by default, on the user's eyes: preferred or alike on 67 of 75 windows
    # (DESIGN.md, Weights). A file is recognised by its tensors, not its name (runtime/weights.py).
    parser.add_argument(
        "--dit-model",
        default="seedvr2x_ema_7b_sharp_fp16.safetensors",
        help="the DiT's file: SeedVR2's 7B DiT, regular or sharp, in fp16, recognised by its"
        " tensors; without --model-dir, seedvr2x_ema_7b_fp16.safetensors or the default"
        " (default: %(default)s, the sharp 7B)",
    )
    parser.add_argument(
        "--vae-model",
        default="seedvr2x_ema_vae_fp16.safetensors",
        help="the VAE's file: SeedVR2's VAE in fp16, recognised by its tensors (default:"
        " %(default)s)",
    )
    parser.add_argument(
        "--resolution",
        type=_positive,
        default=1080,
        help="short side of the output, square pixels at the source's display aspect",
    )
    # Without it, the shot detector runs in the first pass, its probabilities recorded, and the
    # cuts derive from them and --cut-threshold (DESIGN.md, Shot detection).
    parser.add_argument(
        "--cuts",
        type=Path,
        metavar="FILE",
        help="cut list: the first frame of each shot but the first, one frame number per line,"
        " counted from 0, # comments, the fields after the number ignored, as --plan writes it;"
        " frame numbers count the source's frames as seedvr2x counts them, a missing frame's"
        " filled place among them;"
        " in place of the shot detector (default: the cuts TransNetV2 detects, see"
        " --cut-threshold)",
    )
    # The user's proposal, 2026-10-06 (DESIGN.md, Shot detection): erring lower is the cheaper
    # mistake, a miss costing fidelity on four cuts of six, a false cut at most a low-frequency
    # step on a continuous shot (research/docs/cuts.md, What it means for shot detection).
    parser.add_argument(
        "--cut-threshold",
        type=_threshold,
        metavar="P",
        help="the shot detector's threshold, a probability above 0 and at most 1: a cut after the"
        " peak of each run of frames reaching it; lower where cuts go missing, higher where false"
        f" ones show (default: {THRESHOLD}); refused with --cuts, whose list replaces the"
        " detection",
    )
    # DESIGN.md, Shot detection: "`--plan FILE` writes the detected one to FILE in the same
    # format": to a path given, so that --plan works with a one-file output as with a directory,
    # and the list is the user's to edit and keep, never among the job's own files.
    parser.add_argument(
        "--plan",
        type=Path,
        metavar="FILE",
        help="plan the job and stop before the upscale: the first pass, then the cut list"
        f" written to FILE, for editing and --cuts, the possible cuts (runs peaking from {POSSIBLE}"
        " up to the threshold) commented with their probabilities and times, and two playlists"
        " for VLC beside it when the shots are detected, FILE.cuts.m3u8 of the detected cuts and"
        " FILE.possible.m3u8 of the possible ones (none with --cuts); and the counts"
        " printed. The job's manifest and frame index are written too, in the output directory"
        " (PATH.work for a .mkv path): the same command without --plan then runs it, another"
        " --cut-threshold or --cuts plans it again, with no second decode",
    )
    parser.add_argument("--seed", type=int, default=42)
    # split, the colour study's winner, in place of numz's lab (DESIGN.md, Colour correction).
    parser.add_argument(
        "--color-correction",
        choices=("split", "none"),
        default="split",
        help="split: the input's coarse lightness and colour under the model's details, frame by"
        " frame; none: the model's colours, which drift (default: %(default)s)",
    )
    parser.add_argument(
        "--min-segment",
        type=_seconds,
        default=MIN_SEGMENT,
        metavar="SECONDS",
        help="output segments last this long at least, each shorter one merged into its shorter"
        " neighbour, as sptenc's -L (default: %(default)s; 0 keeps every cut)",
    )
    parser.add_argument(
        "--input-matrix",
        choices=sorted(MATRICES),
        help="matrix of a YUV source, when its tag is missing or wrong",
    )
    parser.add_argument(
        "--input-sar",
        type=_ratio,
        metavar="N:D",
        help="sample aspect ratio of the source, when its tag is missing or wrong",
    )
    # An override like the two above (DESIGN.md, Input: exact rational frame rate), for a file
    # whose timestamps contradict the rate it declares; the refusal of such a file says which.
    parser.add_argument(
        "--frame-rate",
        type=_rate,
        metavar="N/D",
        help="frame rate of the source, when its timestamps contradict the one it declares, N/D or"
        " a whole number (24000/1001, 25): its frames taken at that rate, only if every one lies"
        " within half a frame of that rate's timeline",
    )
    parser.add_argument(
        "--window",
        type=int,
        metavar="LATENTS",
        help="pin the cap on the DiT windows, in latents of 4 frames, 5 at least; taken exactly"
        " or refused (default: the planner's, the longest that fits the GPU's free memory)",
    )
    # The planner's other pins (DESIGN.md, Memory planner, Pins), each fixing one choice, taken
    # exactly or refused, never adjusted (planner.plan). Then --until (DESIGN.md, Pause and
    # resume, Stopping), the one option still refused until its work is built (_run).
    parser.add_argument(
        "--encode-tile",
        type=_tile,
        metavar="PX",
        help="pin the VAE encode's tiles: squares of PX output pixels a side, a multiple of 64"
        " from 512 up, or 0 for an encode in one pass, untiled; taken exactly or refused",
    )
    parser.add_argument(
        "--decode-tile",
        type=_tile,
        metavar="PX",
        help="pin the VAE decode's tiles, as --encode-tile pins the encode's; 0 for a decode in"
        " one stream, untiled",
    )
    parser.add_argument(
        "--swap-blocks",
        type=_count,
        metavar="N",
        help="pin the DiT's swapped blocks: its last N wait in system memory and stream through"
        " the GPU, their memory left to the windows; 0 keeps every block on the GPU; taken"
        " exactly or refused",
    )
    parser.add_argument(
        "--until",
        type=_until,
        metavar="HH:MM",
        help="stop before the first unit that wouldn't finish by the next time the local clock"
        " shows HH:MM, what is done kept (exit status 75): the same command resumes",
    )
    parser.add_argument(
        "--accept-env-change",
        action="store_true",
        help="resume a job whose environment changed (the GPU, Python, torch or another package,"
        " ffmpeg or its conversions), recorded in its manifest: the output then differs from an"
        " uninterrupted run's",
    )
    parser.add_argument(
        "--dump-frames",
        type=Path,
        metavar="DIR",
        help="also save every output frame as float32 frame_NNNNNN.npy, as ffv1_out.py does",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # A stop at once, a second Ctrl-C or SIGTERM during the run (runtime/stop.py), SIGTERM during
    # the model files' fetch (stop.terminable), or Ctrl-C before the run: what is kept is what the
    # manifest says, and the same command resumes.
    try:
        return _run(args)
    except KeyboardInterrupt:
        logger.error("stopped at once (Ctrl-C)")
        return 130
    except Terminated:
        logger.error("stopped at once (SIGTERM)")
        return 143
    finally:
        _unmake()
        _unlock()


def verify(argv: list[str]) -> int:
    """seedvr2x verify: decode a job's output and check every frame against the checksums written
    with it (DESIGN.md, Output, Checksums): an output directory's segments, one of them, or a
    one-file output. Exit status 1 when a frame isn't as written, checksums are missing, or a
    segment isn't finished: the output isn't to be trusted whole yet."""
    parser = argparse.ArgumentParser(
        prog="seedvr2x verify",
        description="Check every frame of a job's output against the checksums written with it.",
    )
    parser.add_argument(
        "output", type=Path, help="an output directory, one of its segments, or a one-file output"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        return _verify(args.output)
    except KeyboardInterrupt:
        logger.error("stopped (Ctrl-C)")
        return 130


@dataclass(frozen=True)
class _Checked:
    """What verify checks: an output segment's file or PNG directory, or a one-file output; the
    pixel format of the frames as it holds them; their size; its checksums, None for a segment not
    finished; frames, for a segment command's segment (--segment-cmd), which has no checksums:
    the frames it must hold, counted, where the others' are checked."""

    path: Path
    pix_fmt: str
    size: tuple[int, int]
    checksums: Path | None
    png: bool = False
    frames: int | None = None


def _verify(output: Path) -> int:
    from seedvr2x.media import ffmpeg
    from seedvr2x.media.checksums import check_frames, read_checksums
    from seedvr2x.media.ffmpeg import MediaError, input_args
    from seedvr2x.media.writer import count_frames
    from seedvr2x.runtime.job import JobError

    try:
        ffmpeg.found()
        checked = _checked(output)
    except (MediaError, JobError) as error:
        logger.error("%s", error)
        return 1
    wrong = unfinished = 0
    # A segment command's output has no checksums (DESIGN.md, Output, Checksums: "`--segment-cmd`
    # outputs only get their frame count checked"): each finished segment's frames are counted,
    # as the writer counted them (writer.count_frames), against the manifest's. PROVISIONAL
    # (implementation, 2026-10-10: DESIGN.md doesn't say what verify does of one).
    counted = any(item.frames is not None for item in checked)
    for item in checked:
        if item.checksums is None and item.frames is None:
            unfinished += 1
            logger.warning("%s: unfinished, not checked", item.path)
            continue
        if item.checksums is None:
            try:
                # Strictly: anything ffprobe reports while decoding fails the check (DESIGN.md,
                # Output, Checksums).
                frames = count_frames(item.path, strict=True)
            except MediaError as error:
                wrong += 1
                logger.error("%s", error)
                continue
            if frames != item.frames:
                wrong += 1
                logger.error("%s: %d frames, where %d were written", item.path, frames, item.frames)
            else:
                logger.info("%s: its %d frames there, counted", item.path, frames)
            continue
        try:
            checksums = read_checksums(item.checksums)
            found = _png_listing(item.path, len(checksums)) if item.png else []
            if not found:
                # PNG frames numbered from 0 (writer.PNGWriter), the first one required.
                pattern = ["-start_number", "0", "-start_number_range", "1"]
                reading = (
                    ["-f", "image2", *pattern, "-i", str(item.path / "%06d.png")]
                    if item.png
                    else input_args(item.path)
                )
                found = check_frames(reading, item.pix_fmt, *item.size, checksums)
        except MediaError as error:
            found = [str(error)]
        if found:
            wrong += 1
            logger.error("%s: %s", item.path, "; ".join(found))
        else:
            logger.info("%s: every frame as written", item.path)
    if wrong or unfinished:
        counts = [f"{wrong} not as written"] if wrong else []
        counts += [f"{unfinished} unfinished"] if unfinished else []
        logger.error("%s: of %d, %s", output, len(checked), " and ".join(counts))
        return 1
    if counted:
        logger.info(
            "%s: %d checked, by their frame counts alone: a --segment-cmd output has no checksums,"
            " its frames being its encoder's",
            output,
            len(checked),
        )
        return 0
    logger.info("%s: %d checked, every frame as written", output, len(checked))
    return 0


def _checked(output: Path) -> list[_Checked]:
    """What verify checks of output: the segments its manifest names, or the one it is, or the
    one-file output it is."""
    from seedvr2x.media.ffmpeg import MediaError
    from seedvr2x.media.probe import probe
    from seedvr2x.runtime.job import JobError
    from seedvr2x.runtime.manifest import NAME, SUMS, checksums_file, read

    directory, only = (output, None) if output.is_dir() else (output.parent, output.name)
    if not (directory / NAME).is_file():
        if output.is_dir():
            raise JobError(f"{output}: no {NAME}, so not an output directory of seedvr2x")
        # A finished file without its checksums beside it has nothing to be checked against: a
        # --segment-cmd job's has none, and no manifest once its work directory is removed.
        sums = _checksums_beside(output)
        if not sums.is_file():
            raise MediaError(
                f"{sums}: missing, so nothing to check the frames of {output} against; a"
                " --segment-cmd output has no checksums, its frames being its encoder's"
            )
        # No frame read: verify decodes every one, and its report says what doesn't decode.
        stream = probe(output, first_frame=False)
        if stream.pix_fmt not in ("gbrp16le", "yuv420p10le"):
            raise MediaError(f"{output}: {stream.pix_fmt}, not a master seedvr2x writes")
        return [_Checked(output, stream.pix_fmt, (stream.width, stream.height), sums)]
    content = read(directory / NAME)
    try:
        output_format = str(content["output"]["format"])
        width, height = (int(side) for side in content["output"]["size"])
        segments = [
            (str(entry["name"]), bool(entry["finished"]), int(entry["end"]) - int(entry["start"]))
            for entry in content["segments"]
        ]
        commanded = content["settings"].get("segment_cmd") is not None
    except (KeyError, TypeError, ValueError, AttributeError):
        raise JobError(f"{directory / NAME}: not a manifest seedvr2x wrote") from None
    if only is not None:
        segments = [segment for segment in segments if segment[0] == only]
        if not segments:
            raise JobError(f"{output}: not one of the segments of {directory}")
    png = output_format == "png"
    return [
        _Checked(
            directory / name,
            "rgb48be" if png else output_format,
            (width, height),
            None
            if commanded or not finished
            else directory / SUMS / checksums_file(name, output_format),
            png,
            frames if commanded and finished else None,
        )
        for name, finished, frames in segments
    ]


def _png_listing(directory: Path, count: int) -> list[str]:
    """What is wrong with a PNG segment's files, before its frames are decoded: each of its
    `count` frames there, NNNNNN.png from 0, and nothing else."""
    if not directory.is_dir():
        return ["missing"]
    names = {entry.name for entry in directory.iterdir()}
    expected = {f"{index:06d}.png" for index in range(count)}
    missing, foreign = sorted(expected - names), sorted(names - expected)
    found = [f"{len(missing)} PNG missing, from {missing[0]}"] if missing else []
    found += [f"not its frames: {', '.join(foreign[:5])}"] if foreign else []
    return found


def _run(args: argparse.Namespace) -> int:
    from seedvr2x.media import ffmpeg
    from seedvr2x.media.ffmpeg import MediaError
    from seedvr2x.media.fingerprint import fingerprint
    from seedvr2x.media.source import counted, declare
    from seedvr2x.media.writer import check_command
    from seedvr2x.runtime import memory, planner, pull
    from seedvr2x.runtime.job import (
        SHARED,
        JobError,
        Shot,
        check_seed,
        check_target,
        output_size,
        read_cuts,
        shots_from_cuts,
        target_size,
    )
    from seedvr2x.runtime.manifest import NAME, complete, planned, read
    from seedvr2x.runtime.planner import Pins
    from seedvr2x.runtime.weights import ModelError, check_models

    # The build, the source, the target, the cut list and the model files are checked before
    # anything touches the GPU, but for a job resumed, which is first checked against its record
    # (its settings, environment, the GPU's included, and input), before its first pass, which
    # isn't run again when nothing it depends on changed (DESIGN.md, Pause and resume). The job's
    # GPU is checked before its first pass, whose shot detector runs on it.
    prior: _Prior | None = None
    try:
        # The source is the only input (DESIGN.md, Input): a directory is refused before anything
        # is done, the build's check included, so that nothing is made.
        if args.input.is_dir():
            raise JobError(_directory_refused(args.input))
        numz_padding = _numz_padding()
        numz_draw = _numz_draw()
        numz_tail = _numz_tail()
        if args.window is not None and args.window < 2 * SHARED + 1:
            raise JobError(
                f"--window {args.window}: windows share {SHARED} latents with each neighbour, so"
                f" a window needs at least {2 * SHARED + 1}"
            )
        pins = Pins(args.window, args.encode_tile, args.decode_tile, args.swap_blocks)
        if args.until is not None:
            # Leaves with --until's commit (run.run_job's clock).
            raise JobError(f"--until {args.until:%H:%M}: stopping by the clock isn't built yet")
        # DESIGN.md, Shot detection: --cut-threshold "is refused with --cuts".
        if args.cuts is not None and args.cut_threshold is not None:
            raise JobError(
                f"--cut-threshold {args.cut_threshold}: refused with --cuts, whose list replaces"
                " the detection"
            )
        png = args.format == "png"
        if args.segment_cmd is not None:
            if png:
                raise JobError(
                    "--segment-cmd: refused with --format png: the command reads an FFV1 master's"
                    " stream, and PNG output has none"
                )
            # No {out} in it, or nothing at all: it could write nowhere seedvr2x looks.
            check_command(args.segment_cmd)
        # A .mkv path is assembled at the job's end (_assemble): what the join needs of the build
        # is asked now, not once every segment is made.
        final = _final(args.output)
        ffmpeg_version = ffmpeg.check(
            ("png",) if png else (), ("image2",) if png else (), assembly=final is not None
        )
        conversions = fingerprint()
        logger.info("ffmpeg %s, its conversions' fingerprint %s", ffmpeg_version, conversions[:16])
        declared = declare(args.input, args.input_matrix, args.input_sar, args.frame_rate)
        # The size the frames are resized to, from what the source declares; refused, before the
        # first pass, when too small to pad (job.check_target).
        stream = declared.stream
        target = target_size(stream.width, stream.height, declared.sample_aspect, args.resolution)
        check_target(target, args.resolution)
        cuts = read_cuts(args.cuts) if args.cuts else []
        # The seed against the last shot known before the first pass, the cut list's, so that a
        # seed out of range is refused at once, before the model files and the first pass, whose
        # shot detector imports torch; against every shot after it (check_seed, below).
        # Accepted by design, 2026-10-12 (DESIGN.md doesn't order the checks): "what the command
        # line alone can refuse is refused before the model files and the first pass".
        last = cuts[-1] if cuts else 0
        check_seed(args.seed, [Shot(last, last + 1)])
        directory = _output(args)
        if args.plan is not None:
            _plan_file(args, directory)
        # A job resumed: its record read first, and whether its first pass's would be trusted
        # (_trusted), so that the shot detector's file is fetched and checked only when the
        # detector runs: without --cuts, whose list replaces the detection, and in a first pass
        # run (DESIGN.md, Weights: TransNetV2's weights with the shot detector). Nor for a job
        # recorded with a cut list and asked without one, another job, refused below (_prior)
        # before any first pass: its refusal neither fetches nor hashes a file it wouldn't run.
        recorded: dict[str, Any] | None = None
        kept: FrameIndex | None = None
        untrusted = ""
        if (directory / NAME).is_file():
            _lock(directory)
            recorded = read(directory / NAME)
            # A run stopped between its file put in place and its work directory removed: the
            # removal finished, nothing started (_assembled).
            asked = (args, cuts, numz_padding, numz_draw, numz_tail)
            if final is not None and _assembled(*asked, final, directory, recorded):
                return 0
            kept, untrusted = _trusted(directory, recorded, ffmpeg_version, conversions)
        if final is not None:
            # Before any work, on a new job and on a resume: what the assembly can't take of the
            # source's other streams would otherwise be found at the job's end, every segment
            # made (_copied). SIGTERM unwinds it as Ctrl-C does, its ffmpeg stopped with it
            # (stop.terminable): the run's handlers come later.
            with terminable():
                _copied(args.input, directory)
        if args.segment_cmd is not None:
            _trial(args, declared, output_size(target))
        # A job detecting its shots needs the record's probabilities: a plan cut by a list has
        # none, its first pass run again with the shot detector when it is planned again without
        # one (_prior), which gives the detector its frames (DESIGN.md, Pause and resume, A plan:
        # "A plan cut by a list and planned again without one costs a first pass, no probabilities
        # having been kept"; said in the log). A cut list takes the record with or without them.
        reprobed = args.cuts is None and kept is not None and kept.probabilities is None
        trusted = None if reprobed else kept
        # The detector's file isn't asked for a job recorded with a cut list that has made units,
        # asked without one: another job, which no plan's change takes (_prior).
        detects = (
            args.cuts is None
            and trusted is None
            # A directory's finished job runs no pass (_finished_already); a .mkv path's does.
            and not (final is None and recorded is not None and complete(recorded))
            and (_detecting(recorded) or (recorded is not None and planned(recorded)))
        )
        # A job with every segment made takes no GPU, and is known for one here, before anything
        # asks for it (DESIGN.md, Pause and resume: "the finished state is known before anything
        # asks for the GPU"; Allocator: "A finished job run again asks the driver nothing"): a
        # directory's, finished, which makes nothing; a .mkv path's, whose assembly alone is
        # left, ffmpeg's work. Its driver isn't asked, no NVML is read, no CUDA device is
        # wanted, and its environment is compared without the GPU (_identity). It is held to its
        # model files as any job, checked and hashed below, none of which takes a GPU; but it
        # loads none, and one of seedvr2x's own missing from Hugging Face's cache is held to its
        # pin, not downloaded (pull.resolve; DESIGN.md, Pause and resume, decided on 2026-10-13).
        # But a .mkv path's whose first pass must run again with the shot detector, its frame
        # index gone and its shots detected (`detects`): that pass takes the GPU, as any run, and
        # the detector's file, which it loads, the job still loading no model of its own (`made`;
        # decided on 2026-10-16: "a job that loads no model fetches none even where its first
        # pass must run again", where such a job downloaded the DiT's 16.5 GB for its assembly).
        made = recorded is not None and complete(recorded)
        idle = made and not detects
        # The model files, from --model-dir, or seedvr2x's own from Hugging Face's cache, pulled
        # into it when missing, in one fetch, but for a job with every segment made (`made`),
        # which fetches the shot detector's alone, where its first pass runs with it;
        # checked by their headers, in a second, then seedvr2x's own by their pinned size and
        # SHA-256 (16.5 GB to read for the DiT), which the manifest's record takes: before the
        # resume's comparison and the first pass, which decodes the whole source (DESIGN.md,
        # Weights). The header check comes first, as DESIGN.md has it (Weights, Recognised by
        # content); its refusal of one of seedvr2x's own files, given in its own role, says how
        # to fetch it again, as the pin's does (pull.advice). SIGTERM, which kills the process at
        # once until the run installs its handlers (runtime/stop.py), unwinds a download as
        # Ctrl-C does, so that the library removes its partial file, up to 16.5 GB that no later
        # download takes up (runtime/pull.py, at its end).
        with terminable():
            model_files = pull.resolve(
                *(args.model_dir, args.dit_model, args.vae_model),
                detector=detects,
                download=not made,
            )
        detector = model_files.detector
        dit, vae = (
            None if file.absent else file.path for file in (model_files.dit, model_files.vae)
        )
        check_models(
            *(dit, vae, pull.advice(model_files)), None if detector is None else detector.path
        )
        pull.check_pinned(model_files)
        # The allocator, set before torch is imported, which the job's identity does and, for a
        # job recorded, the modules that compare it (_prior): numz's, which DESIGN.md keeps
        # (Allocator), or torch's own where the GPU's driver gives memory the card doesn't have,
        # which is asked here (runtime/memory.py), where a run first takes the GPU: one refused
        # before, for its source, its settings or its model files, asks the card nothing.
        if not idle:
            memory.startup()
        if recorded is not None:
            prior = _prior(
                *(args, directory, recorded, trusted, cuts, declared, ffmpeg_version),
                *(conversions, numz_padding, numz_draw, numz_tail, model_files),
                idle=idle,
            )
            identity = prior.identity
            # A finished job with no first pass's record to trust: said finished, nothing made.
            done = _finished_already(args, directory, prior, declared, target, untrusted)
            if done is not None:
                return done
        else:
            identity = _identity(
                *(args, cuts, ffmpeg_version, conversions),
                *(numz_padding, numz_draw, numz_tail, model_files),
            )
        if not idle:
            # Where the driver spills, torch is held a first time here, before anything of the
            # run works on the GPU: the shot detector in the first pass, then the first
            # allocation and the warm-up the free memory is read with (_free_memory), which ran
            # before any limit existed. Held again once the plan is made, from the memory free
            # then (below).
            memory.hold(identity.device, first=True)
        if prior is not None and prior.known is not None:
            found = prior.known
        else:
            if untrusted:
                # Said once the job is known for the one recorded, never before a refusal.
                logger.warning("%s; the first pass runs again, which makes it", untrusted)
            elif reprobed:
                logger.info(
                    "%s: no shot probabilities in its record, its cuts given by a list: the first"
                    " pass runs again, the shot detector with it",
                    directory,
                )
            # The source's content hashed meanwhile, for the manifest.
            found = _first_pass(declared, model_files, identity.device)
        source = counted(declared, found)
        if final is not None:
            _grid_start(source)
        found_cuts = _source_cuts(args, cuts, source)
        shots = shots_from_cuts([cut.frame for cut in found_cuts.cuts], source.frames)
        logger.info("%s: %s", _plural(len(shots), "shot"), _said(found_cuts))
        check_seed(args.seed, shots)
        segments, paths = _segments(args, source, shots, directory)
        out_height, out_width = output_size(target)
        logger.info(
            "%s; output %dx%d, square pixels",
            _plural(len(segments), "output segment"),
            out_width,
            out_height,
        )
        # The job's plan (DESIGN.md, Memory planner): one for the job, made after the first pass,
        # whose shots are its input, and before the models load, from the memory free then. A job
        # resumed that has made a unit keeps its plan's shape, where its joins fall and its
        # picture is cut, and where the weights are is planned again around it; a plan, which
        # has made none, is planned again whole (_resume).
        split = args.color_correction == "split"
        kept_plan = None
        plan_record: dict[str, Any] | None = None
        if prior is not None and not planned(prior.recorded):
            kept_plan = _recorded_plan(
                *(directory / NAME, prior.recorded, shots, target), numz_padding=numz_padding
            )
            if complete(prior.recorded):
                # A finished job makes nothing again, and its manifest isn't rewritten
                # (DESIGN.md, Pause and resume), its plan no more than the rest: it stays as
                # recorded, the memory free now none of its business, nor read. Refused for the
                # memory another program holds today, it would no longer say that it is
                # finished.
                plan_record = prior.recorded["plan"]
        if kept_plan is not None and plan_record is not None:
            plan = kept_plan
        else:
            import torch

            # What the driver reports free, and what the plan may count on of it: all of it,
            # but on a driver that gives memory the card doesn't have, where the run is held to
            # less (runtime/memory.py); the plan's lines and its refusals name each figure. The
            # DiT's peak by the fit of the attention backend this run takes, today's on a
            # resume too (_attention).
            free = memory.reading(_free_memory(identity.device))
            plan, peaks = planner.plan(
                *(free, target, shots, pins, _host_available()),
                split=split,
                numz_padding=numz_padding,
                recorded=kept_plan,
                attention=_attention(),
            )
            card = torch.cuda.get_device_name(identity.device)
            said = planner.said(plan, peaks, card, target, shots, pins, numz_padding=numz_padding)
            for line in said:
                logger.info("%s", line)
            # A pass in tiles larger than any measured, which only a pin gives it: warned of,
            # once per pass (planner.beyond_measured).
            for line in planner.beyond_measured(plan):
                logger.warning("%s", line)
            # What shapes the output, compared on a resume, and under memory what doesn't, the
            # predicted peaks with where the weights are and the scratch's allowance, for
            # information (resume.UNCOMPARED).
            plan_record = plan.record()
            plan_record["memory"].update(peaks.record())
    except (MediaError, JobError) as error:
        logger.error("%s", error)
        return 1
    except _out_of_memory() as error:
        # The GPU's memory refused before the models load, where a run first works on the
        # device: the shot detector in the first pass, or the first allocation and the warm-up
        # the free memory is read with (_free_memory), another program holding the card. One
        # line and status 1, as at the load and in a unit.
        logger.error("%s, before the models loaded: nothing was run", _refused(error))
        return 1

    import numpy as np
    import numpy.typing as npt

    from seedvr2x.media.checksums import write_checksums
    from seedvr2x.media.files import make_directories
    from seedvr2x.media.reader import FramesLeft
    from seedvr2x.media.writer import SegmentWriter, Tags, Writer, open_writer
    from seedvr2x.runtime import manifest
    from seedvr2x.runtime.blockswap import HostMemoryError
    from seedvr2x.runtime.model import load_models
    from seedvr2x.runtime.run import run_job
    from seedvr2x.runtime.shot import CopyError, NonFinite, shot_layout
    from seedvr2x.runtime.units import Units

    remaining = list(range(len(segments)))  # the segments to write, those not finished
    try:
        if prior is None:
            _lock(directory)
    except JobError as error:
        logger.error("%s", error)
        return 1
    record = manifest.Manifest(
        directory / NAME,
        identity.settings,
        identity.environment,
        source,
        shots,
        [shot_layout(shot.frames, plan.window) for shot in shots],
        segments,
        [path.name for path in paths],
        {
            "format": args.format,
            "size": [out_width, out_height],
            "frame_rate": str(source.frame_rate),
        },
        plan_record,
    )
    if prior is None and not _empty(directory, final is not None):
        logger.error("%s: written to since it was checked, by another program", directory)
        return 1
    if prior is not None:
        # The job recorded there, checked whole and taken up before the models load: a plan
        # whose cuts changed, or whose shape did with the memory free, planned again
        # (_replan).
        try:
            replanned = _resume(record, prior, final is not None)
        except JobError as error:
            logger.error("%s", error)
            return 1
        remaining = [index for index, done in enumerate(record.finished) if not done]
        if not replanned:
            # The job as recorded: what of it is written before a unit, its first pass run
            # again or its plan taken up in another environment (a plan planned again wrote
            # its own, _replan).
            _taken_up(record, prior)
        if not remaining:
            # Said for --plan too, which still writes the finished job's cut list, its plan.
            logger.info("%s: finished already, its %d segments", directory, len(segments))
            if args.plan is None:
                # A .mkv path's segments, every one finished by an earlier run, are assembled
                # now, the models not loaded (_assemble).
                return 0 if final is None else _assemble(final, record, paths)
    elif args.plan is not None:
        # A new job's directory made a plan (DESIGN.md, Memory planner): its frame index and
        # manifest, no unit; the same command without --plan runs it.
        _first_write(record, source)
    units = Units(directory, record)
    if args.plan is not None:
        # No upscale: --plan prints the plan "without running the job" (DESIGN.md, Memory
        # planner), the models not loaded.
        return _write_plan(args, source, found_cuts, len(shots), model_files)
    started = time.monotonic()
    import torch  # loaded since the job's identity: named here for the refusals below

    # Where the driver spills, torch is held again, to the memory free now that the first pass
    # and the plan are made, and refuses from here on what passes it.
    try:
        memory.hold(identity.device)
    except torch.OutOfMemoryError as error:
        logger.error("%s, before the models loaded: nothing was run", _refused(error))
        return 1
    try:
        models = load_models(
            model_files.dit.path, model_files.vae.path, identity.device, model_files.hashes, plan
        )
    except (ModelError, HostMemoryError) as error:
        # A model file replaced since its check, during the first pass: refused as the check
        # refuses it, or as changed since its hash. Or a host that can't hold the pinned copies
        # of the weights the plan has away from the GPU: refused before the run, naming the RAM
        # they take (DESIGN.md, BlockSwap, rewritten).
        logger.error("%s", error)
        return 1
    except torch.OutOfMemoryError as error:
        # The weights the plan has on the GPU don't fit there after all: other programs hold
        # the card's memory, or the plan is wrong.
        logger.error("%s, as the models loaded: nothing was run", _refused(error))
        return 1
    logger.info(
        "models loaded in %.1f s, attention: %s", time.monotonic() - started, models.attention
    )
    if args.dump_frames is not None:
        args.dump_frames.mkdir(parents=True, exist_ok=True)
    if prior is None:
        _first_write(record, source)
    started = time.monotonic()

    tags = Tags.of(stream, source.conversion.matrix_tag)

    def open_segment(path: Path) -> Writer:
        size = (out_width, out_height, source.frame_rate)
        if args.segment_cmd is not None:
            # No checksums of what the user's command writes, so none of a file before it.
            return open_writer(args.format, path, *size, tags, command=args.segment_cmd)
        stale = _checksums_path(directory, path, args.format)
        return open_writer(args.format, path, *size, tags, stale=stale)

    def finished(which: int, checksums: list[int]) -> None:
        """A segment whole: its checksums written whole, then the segment recorded (DESIGN.md,
        Output, Checksums); none for a segment command's, whose frames are its encoder's."""
        if args.segment_cmd is None:
            path = _checksums_path(directory, paths[remaining[which]], args.format)
            make_directories(path.parent)
            write_checksums(path, checksums)
        elif final is not None:
            # What the assembly needs of a file the command wrote, asked now, not at the job's
            # end (_joinable); against the first segment finished before it.
            done = [index for index in range(len(segments)) if units.finished(index)]
            _joinable(
                *(final, paths[remaining[which]], segments[remaining[which]].frames),
                source.frame_rate,
                (paths[done[0]], segments[done[0]].frames) if done else None,
            )
        units.segment_finished(remaining[which])

    # The n-th frame written is the job's frame number(n): the segments left, in order.
    ends = list(accumulate(segments[index].frames for index in remaining))

    def number(written: int) -> int:
        which = bisect_right(ends, written)
        return segments[remaining[which]].start + written - (ends[which - 1] if which else 0)

    outputs = [(paths[index], segments[index].frames) for index in remaining]
    with Stop() as stop:
        try:
            with SegmentWriter(outputs, open_segment, finished) as writer:

                def write(frames: npt.NDArray[np.float32]) -> None:
                    if args.dump_frames is not None:
                        for written, frame in enumerate(frames, writer.written):
                            path = args.dump_frames / f"frame_{number(written):06d}.npy"
                            np.save(path, np.ascontiguousarray(frame))
                    writer.write(frames)

                run_job(
                    *(models, source, shots, segments, target, args.seed, plan, units),
                    *(write, stop, args.color_correction == "split"),
                    numz_padding=numz_padding,
                    numz_draw=numz_draw,
                    numz_tail=numz_tail,
                )
        except CopyError as error:
            # Derived data, removed (run_job): a resume makes it again.
            again = "; removed, made again from the input on resuming"
            logger.error("%s%s; stopped: %s", error, again, _kept(record, len(segments)))
            return 1
        except _Unjoinable as error:
            # A segment the assembly couldn't join (_joinable): kept, the units; the same
            # command would decode it again and write the same file.
            done = sum(units.finished(index) for index in range(len(segments)))
            logger.error(
                "%s; stopped: %d of %d segments finished, and the units of the next ones kept:"
                " the same command decodes that segment again, and meets the same refusal"
                " unless its cause is gone",
                error,
                done,
                len(segments),
            )
            return 1
        except FramesLeft as error:
            # The source's end, with a frame more than the job's record counts (run.Inputs): the
            # same command would read the last shot again and stop here again, so its line says
            # what is kept and not that it resumes; the way out is the error's own, another -o
            # (design, 2026-10-12). Before MediaError, whose kind it is.
            logger.error("%s; stopped: %s", error, _kept(record, len(segments), resumes=False))
            return 1
        except MediaError as error:
            # A segment command failing among them (--segment-cmd): its segment unfinished, its
            # shots' windows kept, which a resume decodes and writes again. And a tiled pass's
            # scratch that can't be made, its disk full: the user's to fix (tiling.ScratchError).
            # Derived data, its file gone; the units made before it are kept, which the message
            # says as a stop's does.
            logger.error("%s; stopped: %s", error, _kept(record, len(segments)))
            return 1
        except NonFinite as error:
            logger.error("%s, so not recorded; stopped: %s", error, _kept(record, len(segments)))
            return 1
        except torch.OutOfMemoryError as error:
            # The plan passed the GPU's memory: refused by the driver, or by torch where
            # seedvr2x holds it under a driver that would give system memory instead
            # (runtime/memory.py), which also refuses when other programs fill the card.
            # Nothing of the unit is recorded.
            logger.error("%s; stopped: %s", _refused(error), _kept(record, len(segments)))
            return 1
        except Stopped:
            logger.warning(
                "stopped after %s, as asked; %s", stop.unit, _kept(record, len(segments))
            )
            return 130
    from seedvr2x.runtime.environment import imported

    # The versions recorded were derived before the models loaded: what the run has
    # imported since must be among them, or a resume wouldn't see it change. imported()
    # leaves out what isn't the run's, pytest or a profiler, as the record does.
    recorded = cast(dict[str, str], identity.environment["packages"])
    unrecorded = sorted(imported() - set(recorded))
    if unrecorded:
        logger.warning("not in the manifest's environment, a seedvr2x bug: %s", unrecorded)
    logger.info(
        "upscaled in %.1f s; wrote %s: %d frames, %s",
        time.monotonic() - started,
        directory,
        writer.written,
        args.format,
    )
    if final is not None:
        # The job's last unit. A stop asked during the last segment's (Ctrl-C once) stops before
        # it.
        return _assemble(final, record, paths, asked=stop.asked)
    return 0


def _out_of_memory() -> tuple[type[BaseException], ...]:
    """torch's OutOfMemoryError, for an `except` that torch's import may come after (the
    allocator is chosen before torch is imported: memory.startup): nothing to catch while torch
    isn't loaded, no code of its own having run."""
    torch = sys.modules.get("torch")
    return () if torch is None else (getattr(torch, "OutOfMemoryError"),)  # noqa: B009


# torch's out-of-memory message, one line (c10's CUDACachingAllocator): what was asked, then the
# card's figures as torch reads them, its own count, its limit when one is set, and advice.
_OOM_ASKED = re.compile(r"CUDA out of memory\. Tried to allocate (\S+ \w+)\.")
_OOM_HELD = re.compile(
    r"Of the allocated memory (\S+ \w+) is allocated by PyTorch, and (\S+ \w+) is reserved by"
    r" PyTorch but unallocated"
)
_OOM_ALLOWED = re.compile(r" (\S+ \w+) allowed;")
_OOM_CARD = re.compile(r"GPU \d+ has a total capacity of (\S+ \w+) of which (\S+ \w+) is free")
# cudaMallocAsync's message, several lines (c10's CUDAMallocAsyncAllocator): its first says no
# figure; the request, torch's own count, the card's memory and what CUDA has free of it each
# have a line after it, and torch's limit, when one is set, two.
_OOM_ASYNC = re.compile(
    r"Currently allocated *: (?P<held>.+)\nRequested *: (?P<asked>.+)\n"
    r"Device limit *: (?P<card>.+)\nFree \(according to CUDA\): (?P<free>.+)"
)


def _refused(error: BaseException) -> str:
    """The GPU's memory refused, in words, and whether the run was held to the card by seedvr2x
    (runtime/memory.py). Of torch's own message, what happened: the request, what torch held
    and, where it is limited, its limit, each read by its own sentence. The rest of it goes: its
    advice on the allocator's settings, which are seedvr2x's here, with its link, on the same
    line; and "this process has 17179869184.00 GiB memory in use", which torch said of a
    process holding 20.7 GiB under WSL2 (an RTX 5090, 2026-10-12). A message that doesn't start
    as torch's does is given by its first line, whole: the per-unit check's (memory.check).

    The card's memory and what the driver had free of it, torch's "GPU 0 has a total capacity
    of 31.82 GiB of which 9.36 GiB is free", are kept on a driver that refuses, where they are
    the card's; they go on a run held to its card, whose driver's figure isn't (it leaves other
    programs' memory out, and stands under the card's near a full card: runtime/memory.py).
    There too, cudaMallocAsync's message, DESIGN.md's allocator on such a driver, is read
    beyond its first line, which says no figure: what was asked, what torch had allocated, and
    the same two figures, from its own lines (torch 2.14.1's words under each allocator, its
    limit refusing a request of 4 GiB: tests/test_cli_run.py)."""
    from seedvr2x.runtime import memory

    held = " (held to the card's own memory: its driver gives more)" * memory.spills()
    whole = str(error)
    said = whole.partition(chr(10))[0]
    asked = _OOM_ASKED.match(said)
    if asked is not None:
        words = f"CUDA out of memory. Tried to allocate {asked[1]}"
        card = None if memory.spills() else _OOM_CARD.search(said)
        if card is not None:
            words += f", {card[2]} free of the GPU's {card[1]}"
        counted, allowed = _OOM_HELD.search(said), _OOM_ALLOWED.search(said)
        if counted is not None:
            words += f", torch holding {counted[1]} allocated and {counted[2]} more reserved"
        said = words + (f", {allowed[1]} allowed" if allowed is not None else "")
    elif not memory.spills():
        lines = _OOM_ASYNC.search(whole)
        if lines is not None:
            said += (
                f": {lines['asked']} asked, {lines['free']} free of the GPU's {lines['card']},"
                f" torch holding {lines['held']} allocated"
            )
    return f"the GPU's memory was refused{held}: {said}"


def _directory_refused(path: Path) -> str:
    """Why a directory is no input, and what to give instead (DESIGN.md, Input, the user's
    decision of 2026-10-05; "The advice names a join by ffmpeg alone that holds the grid, tested
    with that rule"): the file its segments were split from, or their join on the frame grid,
    each way with its command (rate.grid_join), the directory in them as a shell takes it."""
    from seedvr2x.media import rate

    # The source is the only gateway, so that every cut comes from seedvr2x's own detection or a
    # cut list, and every output segment from its rules, never from an outside splitter's joins:
    # of an earlier split's 411 joins, 63 scored under scdet's threshold of 10 and 14 under 4
    # (research/docs/scene-detection.md).
    #
    # The joins, measured with ffmpeg n9.0.2 (2026-10-10) on 320x180 masters whose pictures
    # carry their number: 5,199 frames in 128 segments of 12 to 102 at 24000/1001, 25 and
    # 30000/1001, and 2,517 in 120 segments of 1 to 300, 45 of them one frame, at 24000/1001.
    # The segments: FFV1 split by stream copy, as sptenc's split cuts (the segment muxer,
    # timestamps reset), and each encoded on its own, FFV1, H.264 (libx264: its default
    # B-frames, preset slow, none), HEVC (libx265) and MPEG-2 (two B-frames) in Matroska, H.264
    # and HEVC in MP4, MPEG-2 in a program and a transport stream, H.264 with AAC audio in both
    # containers; at 25 and 30000/1001, FFV1 and H.264 alone. Each join checked for every
    # picture in order (framemd5, and the numbers read back), every timestamp against its
    # frame's time at the rate rounded to the time base, sptenc's rule, the timestamps' order,
    # ffmpeg's warnings, and seedvr2x's own first pass.
    #
    # The plain join, ffmpeg's concat demuxer and a stream copy, starts each segment where the
    # one before ends by its declared duration, which Matroska keeps in milliseconds: every
    # picture in order, but at 24000/1001 0.77 ms early a join for split segments, half a frame
    # off after 24 joins, and 0.31 ms late for segments encoded on their own, whatever the
    # codec, after 68; at 30000/1001, 0.35 and 0.37 ms early, after 41 and 45; none at 25, a
    # frame lasting whole milliseconds, nor from MP4 segments, whose durations are exact. A
    # 2,400-frame FFV1 master at 24000/1001 split into 156 segments came back 1 ms early at each
    # of the 155 joins, the last frame 155 ms early, and a default read of it by ffmpeg (vfr)
    # dropped 3 frames. Such a join was read until 2026-10-10 and is refused since, once a frame
    # lies half a frame off (source._leaves). The demuxer's input option -r changes nothing in
    # a stream copy, and it takes no -framerate.
    #
    # The join advised, a duration line a segment and every timestamp snapped (sptenc's method,
    # its ffmpeg/concat.go): every picture in order and every timestamp the master's, to the
    # tick, on each Matroska and MP4 material, both sets of segments, the three rates, and on
    # the transport streams of 12 frames and more; in MP4, across segments of one or two frames,
    # ffmpeg warned 39 times of a decoding timestamp it then set right. Durations without the
    # snap leave the frames of segments encoded on their own 41 to 43 ms apart at 24000/1001,
    # and those of split ones 33 to 35 at 30000/1001, a variable rate, refused; the snap without
    # durations, on the plain join's timestamps, puts a frame that drifted half a frame on its
    # neighbour's place. A second stream copy of the plain join that counts each packet's
    # distance from the one before in frames held as widely, without the durations, for a whole
    # second copy of the film (rate.grid_join). Typed as the message first gave it, the list by
    # echo from *.mkv, on the GPU machine (2026-10-10), at 24000/1001 on the FFV1, H.264 (with
    # B-frames), HEVC and MPEG-2 Matroska segments, 128 of each, and on H.264's at 25 and
    # 30000/1001: the same, and read by the first pass, where the plain join of each of those
    # at 24000/1001 and 30000/1001 is refused. As it gives it today, the list by printf from
    # ./*.mkv (rate.LISTED), it wasn't typed on those 128 but on the development machine, under
    # dash 0.5.12 (2026-10-12): on 30 segments of unequal lengths each of six materials, FFV1
    # split at 24000/1001, 25 and 30000/1001, and H.264, HEVC and MPEG-2 each encoded on its own
    # in Matroska at 24000/1001, 813 frames, and on 29 MPEG-2 transport streams, 812: every
    # picture in order, each frame at its place in the master (tests/test_rate.py, which types
    # it at each run, its guard against an apostrophe and its cd -- in sh, dash and bash).
    #
    # No stream copy held on Matroska segments with audio, the video's timestamps coming out 0
    # to 84 ms apart (in MP4 with audio the video held and the audio's timestamps were changed
    # at 18 joins); nor on transport stream segments of a single frame, whose timestamps ffmpeg
    # doesn't find (of two, three and four frames, on the grid: 64x48 pictures, 2026-10-12);
    # nor on program streams, whose packets without a timestamp stop Matroska's muxer, 567 of
    # 5,199 frames written, ffmpeg failing: -fflags +genpts joined those right, and 29 smaller
    # ones with 1 frame of 812 out of place, so it isn't advised. The remake held on all 31
    # materials, typed as the message gives it on the segments with audio and the single-frame
    # transport streams, the audio left out.
    return (
        f"{path}: a directory; seedvr2x takes one video file, the source, so that every cut comes"
        " from its own detection or a cut list (--cuts), and every output segment from its rules."
        " Give it the file the segments were split from, or join the segments back into one file"
        " first so that every frame keeps its place on the frame grid,"
        f" {rate.grid_join(shlex.quote(str(path)))}. A plain join by ffmpeg's concat demuxer, -f"
        " concat -c copy alone, can drift off the grid, up to 1 ms a join, and is refused once a"
        " frame lies half a frame off its place"
    )


def _kept(record: "Manifest", segments: int, resumes: bool = True) -> str:
    """What a run stopped before its end keeps, of a job of `segments` output segments; without
    `resumes`, for a failure the same command would meet again, whose own message gives the way
    out (reader.FramesLeft), what is kept and no more."""
    kept = (
        f"{sum(record.finished)} of {segments} segments finished, and the units of the next ones"
        " kept"
    )
    return f"{kept}: the same command resumes" if resumes else kept


def _final(output: Path) -> Path | None:
    """The finished file a .mkv path names, assembled from the job's output segments once they are
    all done (_assemble); None for a directory, whose segments are the output, never assembled
    (DESIGN.md, Two kinds of users)."""
    return output if output.suffix.lower() == ".mkv" else None


def _work_directory(output: Path) -> Path:
    """Where the job of the .mkv path `output` keeps its files until the file is assembled: beside
    it, its name and .work (DESIGN.md, Colour correction: `<output>.work`)."""
    return output.with_name(f"{output.name}.work")


# The black frames a segment command is tried on (_trial): a few, quick to write and to count.
TRIAL_FRAMES = 5
# What the trial's temporary directory's name begins with: a space, a single quote and a dollar.
TRIAL_PREFIX = "seedvr2x trial '$ "


def _trial(args: argparse.Namespace, declared: "Declared", size: tuple[int, int]) -> None:
    """Try --segment-cmd before any work, at every start: the command run once on TRIAL_FRAMES
    black frames at the output's `size` (height, width), format, rate and tags, through the very
    writer the run uses (writer.CommandWriter), into a temporary directory, the output's own left
    as it is; the file named as a segment is, its suffix the muxer's cue; its frames counted, as a
    segment's are, and, for a .mkv path, what the assembly needs of it asked (_joinable). A
    failure refuses the job (JobError) with the writer's message.

    PROVISIONAL (implementation, 2026-10-10: DESIGN.md has the command run once per segment and
    says nothing of a trial): a command with a typo or a missing encoder would otherwise fail
    once the first segment's shots are encoded and sampled, GPU minutes to hours in, and the
    command mended would then be another job (settings.segment_cmd). It costs 0.26 to 0.29 s
    with DESIGN.md's own example, libx265 at -crf 16, at 1920x1080 in yuv420p10le, the
    assembly's check of the file among it (measured on 2026-10-12, ffmpeg n9.0.2, 32 threads)."""
    import tempfile

    import numpy as np

    from seedvr2x.media.ffmpeg import MediaError
    from seedvr2x.media.writer import Tags, open_writer
    from seedvr2x.runtime.job import JobError

    height, width = size
    stream = declared.stream
    rate = stream.frame_rate if declared.rate_override is None else declared.rate_override
    tags = Tags.of(stream, declared.conversion.matrix_tag)
    black = np.zeros((1, height, width, 3), np.float32)
    started = time.monotonic()
    # SIGTERM unwinds it as Ctrl-C does, the command stopped and the directory removed
    # (stop.terminable): the run's handlers come later.
    # In a directory whose name the shell can't take bare, a space, a quote and a dollar in it: a
    # command that quotes {out} itself, "{out}", where seedvr2x quotes the name it stands for,
    # then fails here, as it would at the first segment whose path holds a space, a parenthesis
    # or an accent (a review's finding, 2026-10-12: `cat > "{out}"` wrote in a directory named
    # plain and was refused, "Directory nonexistent", in one named `with space`).
    with terminable(), tempfile.TemporaryDirectory(prefix=TRIAL_PREFIX) as scratch:
        path = Path(scratch) / "seg_000000.mkv"
        try:
            writing = open_writer(
                args.format, path, width, height, rate, tags, command=args.segment_cmd
            )
            with writing as writer:
                for _ in range(TRIAL_FRAMES):
                    writer.write(black)
            final = _final(args.output)
            if final is not None:
                _joinable(final, path, TRIAL_FRAMES, rate, None)
        except MediaError as error:
            # The count decodes what the command wrote (writer.count_frames): a codec this
            # ffmpeg can't decode fails it, as a wrong count does.
            raise JobError(
                f"--segment-cmd, tried on {TRIAL_FRAMES} black frames before any work: {error}"
                " (its file's frames are counted by decoding them, so this ffmpeg must decode"
                " what it writes; and {out} is to be written bare, seedvr2x quoting the name"
                " it stands for, which holds a space, a quote and a dollar here, on purpose)"
            ) from None
    logger.info(
        "--segment-cmd: tried on %d black frames, %dx%d %s at %s fps, in %.1f s",
        TRIAL_FRAMES,
        width,
        height,
        args.format,
        rate,
        time.monotonic() - started,
    )


class _Unjoinable(MediaError):
    """What a segment command wrote can't be assembled (_joinable): a failure of its own kind,
    which the same command would meet again."""


def _joinable(
    final: Path,
    path: Path,
    frames: int,
    frame_rate: Fraction,
    before: tuple[Path, int] | None,
) -> None:
    """Refuse (MediaError) the file at path, `frames` frames a segment command just wrote for a
    .mkv path's job, when the assembly couldn't join it into `final`: not Matroska at its
    millisecond time base, a packet short or without a timestamp, frames off the rate's grid
    (assembly.segment_refusal); or, `before` being the first segment finished before it, its
    file and frames, another codec, size or pixel format than that one's, or codec headers
    that don't decode it as its own do (assembly.joins, the assembly's own verdict): the concat
    demuxer gives every packet the first file's. A read of its packets, and a decode of the two
    only when their headers differ.

    PROVISIONAL (implementation, 2026-10-10: DESIGN.md has a --segment-cmd output's frame count
    checked alone): asked of the trial's file and of each segment as it is finished, so that a
    command whose files can't be joined is refused at its first one, not by the assembly's own
    check once every segment is made. A directory's segments, never assembled, are asked
    nothing."""
    from seedvr2x.media import assembly

    refusal = assembly.segment_refusal(path, frames, frame_rate)
    if not refusal and before is not None:
        # In the segment's own directory, the job's, where the assembly's files go.
        differing = assembly.joins(before, (path, frames), frame_rate, path.parent)
        refusal = differing and f"{differing}, as {before[0].name} has them"
    if refusal:
        raise _Unjoinable(
            f"{path}: what the segment command wrote can't be assembled into {final}: {refusal};"
            " give a command whose files are Matroska holding every frame at its own time, alike"
            " from one segment to the next (another command is another job), or give -o a"
            " directory, whose segments are joined by your own tools"
        )


def _copied(source: Path, directory: Path) -> None:
    """Refuse (JobError) a source whose other streams the assembly can't take, before any work
    (assembly.preflight): a stream Matroska's muxer refuses as it is, each named with ffmpeg's
    own reason and the ways out; else anything ffmpeg says of their remux, the very one the
    assembly makes, rehearsed whole in directory, the job's work directory, which is made and
    locked for it (_lock), as an output directory is, from the start: a second run to the same
    file is refused before its first pass. What an assembly or a rehearsal killed there left
    (assembly.work_file) goes first. A run refused before it writes anything there, by this
    check or a later one, leaves nothing: the directory, when this made it, goes as main returns
    if it is still empty (_unmake).

    PROVISIONAL (implementation, 2026-10-10: DESIGN.md has every other stream copied and says
    nothing of one that can't be): refused before any work, the fix being the user's, a remux of
    the source or a directory output, rather than copied without it, or converted."""
    from seedvr2x.media import assembly
    from seedvr2x.runtime.job import JobError

    # Its own to remove (_unmake) only once the lock says so: another seedvr2x may have made
    # it meanwhile.
    made = not directory.exists()
    _lock(directory)
    if made:
        _MADE.append(directory)
    for left in sorted(directory.iterdir()):
        if assembly.work_file(left.name) and left.is_file() and not left.is_symlink():
            left.unlink()
            logger.info("%s: left by an assembly that was stopped, removed", left)
    refused = assembly.preflight(source, directory)
    if refused:
        raise JobError(refused)


def _grid_start(source: "Source") -> None:
    """Refuse (JobError), once the first pass has timed its frames, a source whose first video
    frame comes before its file's start time (assembly.start): the assembly can't place its
    frames, and refuses it (assembly.assemble); said here, before any unit is made, not after
    the last. PROVISIONAL, as the join's refusal is (no source measured has one)."""
    from seedvr2x.media import assembly
    from seedvr2x.runtime.job import JobError

    if source.index is None:
        raise ValueError(f"{source.path}: no frame index to time the assembly by: a seedvr2x bug")
    first = assembly.start(source.index)
    if first < 0:
        raise JobError(
            f"{source.path}: its first video frame is {float(-first) * 1000:.3f} ms before the"
            " file's start time, where the assembly of a .mkv path can't place it; give -o a"
            " directory, whose segments are then muxed with the source's streams by your own"
            " tools"
        )


def _assemble(final: Path, record: "Manifest", paths: Sequence[Path], asked: bool = False) -> int:
    """Assemble the finished file `final` from the job's segments, every one finished, at `paths`
    in record's directory, with the source's other streams (DESIGN.md, Output, Assembly), and
    return the exit status. One unit, the job's last, run by the run that finished the last
    segment or by a later one, which loads no model: stopped before it when a stop is asked
    (`asked`: during the last segment's unit), finished when one is asked during it; a stop at
    once unwinds through the join, which kills its ffmpeg and removes its partial file
    (assembly.assemble), every segment kept for the same command to assemble.

    The join proves its result before putting it in place (assembly.assemble), an old file's
    checksums going only as it is replaced. Then, in this order, each step safe to stop after:
    the file's checksums, the segments' one after the other, the file holding their frames as
    they are (stream copy), written whole beside it (DESIGN.md, Output, Checksums); the
    manifest's record of the assembly (manifest.ASSEMBLED); the work directory's removal
    (_clear), which a run stopped during it leaves to the next (_assembled)."""
    from seedvr2x.media import assembly
    from seedvr2x.media.checksums import read_checksums, write_checksums
    from seedvr2x.media.ffmpeg import MediaError

    directory, source = record.path.parent, record.source
    if source.index is None:
        raise ValueError(f"{source.path}: no frame index to time the assembly by: a seedvr2x bug")
    output_format = str(record.output["format"])
    segments = [(path, part.frames) for path, part in zip(paths, record.segments, strict=True)]
    sums = _checksums_beside(final)
    # What the same command does then: its checks of the job first, as at any start, no model
    # loaded and no GPU work.
    kept = (
        f"every segment finished, {len(segments)} in {directory}: the same command assembles"
        f" {final}, once it has checked the job again (its model files, its environment, the"
        " source's other streams), no model loaded"
    )
    # A segment command's segments have no checksums (--segment-cmd), nor the file they make.
    summed = record.settings.get("segment_cmd") is None
    with Stop() as stop:
        stop.asked = asked
        try:
            stop.check()
            stop.unit = f"the assembly of {final}"
            started = time.monotonic()
            checksums = [
                checksum
                for path in (paths if summed else ())
                for checksum in read_checksums(_checksums_path(directory, path, output_format))
            ]
            if summed and len(checksums) != source.frames:
                raise MediaError(
                    f"{directory}: {len(checksums)} checksums for its segments' {source.frames}"
                    " frames"
                )
            joined = assembly.assemble(
                *(final, segments, source.frame_rate, source.path, source.index),
                work=directory,
                stale=sums,
            )
        except Stopped:
            logger.warning("stopped before the assembly, as asked; %s", kept)
            return 130
        except MediaError as error:
            logger.error("%s; not assembled: %s", error, kept)
            return 1
        if summed:
            write_checksums(sums, checksums)
        record.job_assembled(final.name, final.stat().st_size)
        logger.info(
            "wrote %s: %d frames from %s, with the source's %s and %s, assembled in %.1f s",
            final,
            joined.frames,
            _plural(len(segments), "segment"),
            _plural(joined.streams - 1, "other stream"),
            _plural(joined.chapters, "chapter"),
            time.monotonic() - started,
        )
        _clear(directory, record.content())
    return 0


def _assembled(
    args: argparse.Namespace,
    cuts: list[int],
    numz_padding: bool,
    numz_draw: bool,
    numz_tail: bool,
    final: Path,
    directory: Path,
    recorded: dict[str, Any],
) -> bool:
    """Whether the job recorded in directory, the work directory of `final`, was assembled
    already, its manifest saying so (manifest.ASSEMBLED) and the file there with the size
    recorded: a run stopped, or killed, between the file put in place and the work directory
    removed, or one that left the directory for what it held of someone else's. The removal is
    then finished (_clear), and the run starts nothing: no first pass, no model. Else, the
    record's file missing or of another size, the record is dropped and the job assembled again
    from its segments, which a removal begun then makes a resume refuse, naming the first one
    missing (resume.leftovers).

    Only for the job asked: refused (JobError), nothing removed, when the job recorded is
    another, as far as is told here without a hash: the source's size, and every setting the
    command line gives (_settings without the model files; HASHED). The file there is then
    another job's, which exit status 0 would pass for this one's (a review's finding,
    2026-10-12: another source and seed given the same -o).

    PROVISIONAL (implementation, 2026-10-10: DESIGN.md has the work directory removed and says
    nothing of a stop during it): the record is the cheap way to make a kill during the removal
    of 100 GiB harmless, a directory half removed being otherwise refused by the next run. With
    --plan, FILE isn't written, which is said: the job's first pass's record may be gone with
    the directory, and the same command then plans a new job."""
    from seedvr2x.runtime.job import JobError
    from seedvr2x.runtime.manifest import ASSEMBLED

    entry: Any = recorded.get(ASSEMBLED)
    if not isinstance(entry, dict):
        return False
    record = cast(dict[str, Any], entry)
    size = final.stat().st_size if final.is_file() else None
    if record.get("name") != final.name or size != record.get("bytes"):
        logger.warning(
            "%s: assembled as %s, %s bytes, its manifest says, but %s: assembled again",
            directory,
            record.get("name"),
            record.get("bytes"),
            f"{final} is missing" if size is None else f"{final} holds {size}",
        )
        return False
    from seedvr2x.runtime import resume

    settings: Any = recorded.get("settings")
    source: Any = recorded.get("input")
    was = {
        "settings": {
            name: value
            for name, value in cast(dict[str, Any], settings).items()
            if name not in HASHED
        }
        if isinstance(settings, dict)
        else settings,
        "input": {"bytes": cast(dict[str, Any], source).get("bytes")}
        if isinstance(source, dict)
        else source,
    }
    now = {
        "settings": _settings(args, cuts, numz_padding, numz_draw, numz_tail, None),
        "input": {"bytes": args.input.stat().st_size},
    }
    found = resume.differences(was, now)
    if found:
        raise JobError(
            f"{directory}: the work directory of another job than the one asked, which {final}"
            " was assembled from already; it differs in:\n  "
            + "\n  ".join(found[:20])
            + (f"\n  and {len(found) - 20} more" if len(found) > 20 else "")
            + f"\nNothing was removed: remove {directory}, or choose another -o"
        )
    logger.info("%s: assembled already, %d bytes", final, size)
    _clear(directory, recorded)
    if args.plan is not None:
        logger.warning(
            "--plan %s: not written, the job being finished and its record removed: the same"
            " command plans a new job",
            args.plan,
        )
    return True


def _clear(directory: Path, content: dict[str, Any]) -> None:
    """Remove the work directory of a .mkv path, its job assembled, `content` its manifest's: this
    job's files, then the directory (resume.clear); what else is there is left, and said, as is
    a removal the file system refuses: the file is in place, and nothing here fails the run."""
    from seedvr2x.runtime import resume

    try:
        foreign = resume.clear(directory, content)
    except OSError as error:
        logger.warning(
            "%s: not removed: %s; its job is assembled, and it is left as it is: remove it by hand",
            directory,
            error,
        )
        return
    if foreign:
        logger.warning(
            "%s: not removed, holding what isn't this job's: %s%s; it is left as it is, with the"
            " job's manifest: remove it by hand",
            directory,
            ", ".join(foreign[:5]),
            f" and {len(foreign) - 5} more" if len(foreign) > 5 else "",
        )
    else:
        logger.info("%s: removed, its job assembled", directory)


def _output(args: argparse.Namespace) -> Path:
    """The directory the job's files go in: its output segments, manifest, frame index, units and
    checksums. For a .mkv path, the finished file's (_final), its work directory beside it
    (_work_directory), where the segments wait for their assembly; else the output itself, a
    directory of segments (DESIGN.md, Output). Either is new or empty, which keeps another run's
    files out of what sptenc reads or the assembly joins, or holds a job's manifest, which resume
    reads back (resume.leftovers checks the rest); anything else is refused, never deleted. A
    .mkv path's job runs in its work directory as a directory's does in its own: every job keeps
    its units in files (units.Units), none in memory."""
    from seedvr2x.runtime.job import JobError
    from seedvr2x.runtime.manifest import NAME

    output: Path = args.output
    # The output naming the source itself: its file, once assembled, would replace the source,
    # the one thing a job can't make again, and its checksums' file likewise. The same file
    # when the name exists, by whatever path (a relative name, a linked directory); else the
    # same place. PROVISIONAL (implementation, 2026-10-12: DESIGN.md has no such sentence;
    # found by a review, `seedvr2x film.mkv -o film.mkv` leaving the upscale under the
    # source's name).
    sums = [_checksums_beside(output)] if _final(output) is not None else []
    for written in (output, *sums):
        if _same(written, args.input):
            said = "the output" if written == output else f"the output's checksums, {written},"
            raise JobError(f"{output}: {said} is the source itself; choose another -o")
    # PROVISIONAL (implementation, 2026-10-10: DESIGN.md's Assembly names one file and no
    # container): Matroska is the one container assembled, the segments' own (media/writer.py);
    # the other video suffixes stay refused, as before the assembly.
    if output.suffix.lower() in VIDEO_SUFFIXES:
        raise JobError(
            f"{output}: a video file name; the finished file is a .mkv path, and the output"
            " segments go in a directory"
        )
    directory, what = output, "the output directory"
    if _final(output) is not None:
        if output.is_dir():
            raise JobError(f"{output}: a directory; a .mkv path names the finished file")
        if args.format == "png":
            raise JobError(f"{output}: PNG output goes to a directory")
        # A file already at the path, with no work directory beside it, is no obstacle: a new
        # job, whose file replaces it once whole (assembly.assemble), with its checksums.
        # PROVISIONAL (implementation, 2026-10-10: DESIGN.md doesn't say; what one FFV1 master
        # did before the assembly, kept).
        directory, what = _work_directory(output), f"the work directory of {output}"
        if os.path.lexists(directory) and (directory.is_symlink() or not directory.is_dir()):
            raise JobError(
                f"{directory}: not a directory; {output}'s job keeps its files there until the"
                " file is assembled: remove it, or choose another -o"
            )
    elif output.is_file():
        raise JobError(f"{output}: a file; the output segments need a directory")
    dumps: Path | None = args.dump_frames
    if dumps is not None and dumps.resolve().is_relative_to(directory.resolve()):
        raise JobError(f"--dump-frames {dumps}: in {what}, which holds the job's own files only")
    work = directory != output
    if directory.is_dir() and not _empty(directory, work) and not (directory / NAME).is_file():
        raise JobError(
            f"{directory}: not empty, and no {NAME} to resume from; "
            + (
                f"{output}'s job keeps its files there: remove it, or choose another -o"
                if work
                else "the output segments need a new or empty directory"
            )
        )
    return directory


def _segments(
    args: argparse.Namespace,
    source: "Source",
    shots: "Sequence[Shot]",
    directory: Path,
) -> "tuple[list[OutputSegment], list[Path]]":
    """The output's segments and each one's path in the job's directory (_output): the source's
    shots, merged to --min-segment, and named as sptenc's split names its own (DESIGN.md,
    Output)."""
    from seedvr2x.runtime.job import merged_segments

    segments = merged_segments(shots, source.frames, source.frame_rate, args.min_segment)
    suffix = "" if args.format == "png" else ".mkv"
    return segments, [directory / f"{segment.name}{suffix}" for segment in segments]


def _checksums_path(directory: Path, output: Path, output_format: str) -> Path:
    """Where the checksums of the output segment at path `output` go: in SUMS, in the job's
    directory (DESIGN.md, Output, Checksums); a finished file's own go beside it
    (_checksums_beside)."""
    from seedvr2x.runtime.manifest import SUMS, checksums_file

    return directory / SUMS / checksums_file(output.name, output_format)


def _same(path: Path, other: Path) -> bool:
    """Whether path and other name one file: the same file when both exist, however each is
    spelled (os.path.samefile); else the same place, links followed."""
    try:
        return os.path.samefile(path, other)
    except OSError:
        return path.resolve() == other.resolve()


def _checksums_beside(output: Path) -> Path:
    """Where the checksums of the finished file `output` go: beside it, its name and .crc32
    (DESIGN.md, Output, Checksums)."""
    return output.with_name(f"{output.name}.crc32")


def _empty(directory: Path, work: bool = False) -> bool:
    """Whether directory holds nothing, but what an interrupted first write of the manifest left,
    which the next write replaces: the manifest's partial file, the frame index written whole
    before it (manifest.INDEX), or the index's partial file; and, in a .mkv path's work
    directory (`work`), what a rehearsal of its assembly left when killed (_copied), which the
    next start removes."""
    from seedvr2x.media.assembly import work_file
    from seedvr2x.runtime.manifest import INDEX, NAME

    left = {f"{NAME}.partial", INDEX, f"{INDEX}.partial"}
    return all(
        entry.name in left or (work and work_file(entry.name)) for entry in directory.iterdir()
    )


def _write_index(directory: Path, source: "Source") -> None:
    """Write the source's frame index beside the manifest, whole (manifest.INDEX), the shot
    detector's probabilities with it when it ran."""
    from seedvr2x.runtime.manifest import INDEX

    if source.index is None:
        raise ValueError(f"{source.path}: no frame index to write")
    source.index.write(directory / INDEX)


def _first_write(record: "Manifest", source: "Source") -> None:
    """A new job's first write in its directory: the frame index, whole, then the manifest naming
    it (manifest.INDEX)."""
    record.path.parent.mkdir(parents=True, exist_ok=True)
    _write_index(record.path.parent, source)
    record.write()


def _first_pass(
    declared: "Declared", model_files: "ModelFiles", device: "torch.device"
) -> "FirstPass":
    """The first pass over the source, its content hashed meanwhile, for the manifest
    (source.first_pass); with the shot detector when its file was resolved (no --cuts):
    TransNetV2 on the job's device, fed every frame the pass decodes, its memory given back
    before the models load (DESIGN.md, Shot detection; Memory planner, Budget)."""
    from seedvr2x.media.source import first_pass
    from seedvr2x.runtime.detector import Detector

    if model_files.detector is None:
        return first_pass(declared, hashed=True)
    with Detector(model_files.detector.path, device, hashes=model_files.hashes) as detector:
        return first_pass(declared, hashed=True, detector=detector)


def _free_memory(device: "torch.device") -> int:
    """The memory free on the job's GPU, in bytes, as the driver reports it (mem_get_info): the
    planner's budget, less its margin (DESIGN.md, Memory planner, Budget). Read once a run, after
    the first pass, whose shot detector has given its memory back, and before the models load.

    Read with a first allocation in place, a small tensor held on the device, and no cache
    emptied (DESIGN.md, Memory planner, Budget): a process's first allocation on the GPU takes
    0.2363 GiB outside torch's counters (253,755,392 bytes with cudaMallocAsync, 32 MiB of them
    the pool's first reservation), which no reading before it shows, and a run that builds no
    shot detector, one given a cut list or resumed from a record it trusts, has allocated
    nothing yet when it plans. Emptying the cache after it gives the pool's first reservation
    back, which the next allocation takes again. Measured on the bare 96 GB card (2026-10-10 and
    11): 94.4973 GiB free with the bare context; 94.4308 with a tensor allocated and freed and
    the cache emptied, as this read at first; 94.2609 with a first tensor held; 94.2746 after the
    shot detector ran and closed. The margin was bisected against readings taken with a first
    allocation in place (research/docs/planner-limits.md, The margin over free memory;
    vram_cap.py allocates a tensor and never empties the cache). The freed tensor left the two
    kinds of run 160 MiB apart, a run with a cut list planning on memory its first allocation
    then takes. With the tensor held and no more, a run read after its shot detector 94.24 GiB
    on the 96 GB card, and N - 1.46 on a card emulated at N GB (2026-10-11).

    The tensor alone still left the two kinds of run apart: on a card emulated at 24 GB, a new
    job, whose shot detector ran in the first pass, read 22.54 GiB free, and a resume of it, its
    record trusted and no detector built, 22.56 (2026-10-11). The 0.02 GiB are what the
    detector's run leaves loaded in the process, the libraries of its convolutions and of its
    linear layers (cuDNN's and cuBLAS's handles and kernels, which stay until the process ends:
    detector.Detector.close). So both read on one footing: besides the tensor, a tiny
    convolution (conv3d) and a tiny matrix product run on the device, in float32, as the
    detector computes, and their results are held with the tensor while the driver is read. A
    run without a detector has then loaded what a run with one has, which the models' own load
    brings anyway. On the CPU, the device of the tests' stand-in, they run and load nothing.

    With that warm-up, as this reads (measured 2026-10-11): 94.21 GiB on the 96 GB card
    (101,159,469,056 bytes), and N - 1.49 on a card emulated at N GB (24,168,824,832 bytes at
    24), the same to the byte for a new job, whose shot detector ran, for the same command
    again, its record trusted and no detector built, and for a job given a cut list."""
    import torch

    first = torch.empty(1, device=device)
    # The warm-up: one 3x3x3 kernel over one 3x3x3 volume, and a 2x2 matrix by itself.
    volume = torch.zeros(1, 1, 3, 3, 3, dtype=torch.float32, device=device)
    convolved = torch.nn.functional.conv3d(volume, volume)
    matrix = torch.zeros(2, 2, dtype=torch.float32, device=device)
    product = torch.mm(matrix, matrix)
    free = torch.cuda.mem_get_info(device)[0]
    del first, convolved, product
    return free


def _attention() -> str:
    """The attention backend this run's DiT takes: FlashAttention 2 where it imports, else
    PyTorch's SDPA, as the loader decides it for every attention module, and warns of
    (model.attention_backend, model.load_models). The plan's DiT peak is predicted by that
    backend's fit (planner.DIT_ATTENTION): today's, never a recorded environment's, which a
    resume in another one would plan by."""
    from seedvr2x.runtime import model

    return model.attention_backend()


def _host_available(
    proc: Path = Path("/proc"), cgroups: Path = Path("/sys/fs/cgroup")
) -> int | None:
    """The system memory this process can still take, in bytes, as the placement reads it before
    it pins a copy (blockswap.available): the kernel's MemAvailable, what new work can take
    without swapping, or less in a container held to a limit, what is left under its control
    group's. The plan's pinned copies and scratch are counted against it, and the scratch's
    allowance sized from it (DESIGN.md, Memory planner, Host RAM). None when nothing can be
    read: the plan then can't check the system memory it takes, nor size the allowance from the
    host's, a tiled shot's scratch allowed planner.ALLOWANCE_UNKNOWN_HOST at most. Said as a
    warning, once a run, which reads it once (_run)."""
    from seedvr2x.runtime import blockswap
    from seedvr2x.runtime.planner import ALLOWANCE_UNKNOWN_HOST, GIB

    available = blockswap.available(proc, cgroups)
    if available is None:
        logger.warning(
            "%s: no MemAvailable read there, nor a control group's limit under %s: the system"
            " memory available isn't known, so the plan can't check what it takes of it, nor"
            " size from it what a tiled shot's scratch may take, which is then the shot's whole"
            " scratch, %d GiB at most",
            proc / "meminfo",
            cgroups,
            ALLOWANCE_UNKNOWN_HOST // GIB,
        )
    return available


def _recorded_plan(
    path: Path,
    recorded: dict[str, Any],
    shots: "Sequence[Shot]",
    target: tuple[int, int],
    *,
    numz_padding: bool,
) -> "Plan":
    """The plan the manifest at path records, as a resume takes it (Plan.recorded): refused
    (JobError), naming the manifest, when its record isn't a plan's, or is one the planner makes
    for no job of these shots at this target: a cap they can't be laid out by, or a tile that
    covers the whole padded frame (planner.check_recorded)."""
    from seedvr2x.runtime import planner
    from seedvr2x.runtime.job import JobError
    from seedvr2x.runtime.plan import Plan

    try:
        plan = Plan.recorded(recorded.get("plan"))
        planner.check_recorded(plan, shots, target, numz_padding=numz_padding)
    except JobError as error:
        raise JobError(f"{path}: {error}") from None
    return plan


# The settings that record a hash, of seedvr2x's code or of a model file (_settings): those a
# job can't be told by before its files are read.
HASHED = ("code", "dit_model", "vae_model", "detector_model")


def _settings(
    args: argparse.Namespace,
    cuts: list[int],
    numz_padding: bool,
    numz_draw: bool,
    numz_tail: bool,
    model_files: "ModelFiles | None",
) -> dict[str, object]:
    """The settings a manifest records, those a resume must find again; numz's padding, numz's
    draw and numz's tail each only when the tests ask for it (NUMZ_PADDING, NUMZ_DRAW, NUMZ_TAIL),
    so that a user's manifest never names them. Without model_files, those the command line
    alone gives, the records that take a hash left out (HASHED): what tells one job from another
    before any file is read (_assembled)."""
    from seedvr2x.runtime.manifest import code_sha256

    detects = args.cuts is None
    hashed: dict[str, object] = {}
    if model_files is not None:
        hashed = {
            "code": code_sha256(),
            "dit_model": _model(model_files.dit, model_files),
            "vae_model": _model(model_files.vae, model_files),
            # The shot detector's file, as the DiT's and the VAE's, when the job detects shots:
            # a resume with another is another job.
            **({"detector_model": _detector_model(model_files)} if detects else {}),
        }
        if not set(hashed) <= set(HASHED):
            raise ValueError(f"{sorted(set(hashed) - set(HASHED))}: not in HASHED: a seedvr2x bug")
    settings: dict[str, object] = {
        "seedvr2x": version("seedvr2x"),
        **hashed,
        "resolution": args.resolution,
        "seed": args.seed,
        "color_correction": args.color_correction,
        # The pins that shape the output, as asked, None when not given: the same command
        # resumes (DESIGN.md, Pause and resume), so a resume finds them again. The tile pins
        # are settings, which a resume must repeat; --swap-blocks isn't, shaping no bit, and a
        # resume may change it (DESIGN.md, Memory planner, Pins).
        "window": args.window,
        "encode_tile": args.encode_tile,
        "decode_tile": args.decode_tile,
        "format": args.format,
        # The segments' writer (DESIGN.md, Output, Writer): the user's command, as given, or
        # None, seedvr2x's FFV1 or PNG writer. PROVISIONAL (implementation, 2026-10-10: DESIGN.md
        # doesn't say whether a command may change): a setting like the others, another command
        # another job, even before any segment is finished, where the units kept don't depend
        # on it.
        "segment_cmd": args.segment_cmd,
        # How the source is cut (resume.CUT_SETTINGS): the cut list given, or none and the
        # detection's threshold, "a setting the manifest records" (DESIGN.md, Shot detection),
        # as Python writes the float, which reads back exactly; the cuts it gives are the
        # layout's shots, compared after the first pass.
        "cuts": None if detects else cuts,
        **({"cut_threshold": repr(_cut_threshold(args))} if detects else {}),
        "min_segment": str(args.min_segment),
        "input_matrix": args.input_matrix,
        "input_sar": None if args.input_sar is None else str(args.input_sar),
        # --frame-rate's R, "recorded as a setting" (DESIGN.md, Input), as Fraction writes it:
        # "24" for a whole rate, "24000/1001" otherwise, either read back exactly (kept by
        # design, 2026-10-10).
        "frame_rate": None if args.frame_rate is None else str(args.frame_rate),
    }
    if numz_padding:
        settings["numz_padding"] = True
    if numz_draw:
        settings["numz_draw"] = True
    if numz_tail:
        settings["numz_tail"] = True
    return settings


def _numz_padding() -> bool:
    """Whether seedvr2x's tests ask for numz's padding (NUMZ_PADDING=1), said when they do;
    refused (JobError) when the variable is set to anything else."""
    from seedvr2x.runtime.job import JobError

    value = os.environ.get(NUMZ_PADDING)
    if value is None:
        return False
    if value != "1":
        raise JobError(f"{NUMZ_PADDING}={value!r}: for seedvr2x's tests only, 1 or unset")
    logger.warning("%s=1: numz's padding, for seedvr2x's tests only", NUMZ_PADDING)
    return True


def _numz_draw() -> bool:
    """Whether seedvr2x's tests ask for numz's draw of the noise, on the GPU (NUMZ_DRAW=1), said
    when they do; refused (JobError) when the variable is set to anything else."""
    from seedvr2x.runtime.job import JobError

    value = os.environ.get(NUMZ_DRAW)
    if value is None:
        return False
    if value != "1":
        raise JobError(f"{NUMZ_DRAW}={value!r}: for seedvr2x's tests only, 1 or unset")
    logger.warning(
        "%s=1: numz's draw of the noise, on the GPU, for seedvr2x's tests only", NUMZ_DRAW
    )
    return True


def _numz_tail() -> bool:
    """Whether seedvr2x's tests ask for numz's padding of a shot's end, mirrored (NUMZ_TAIL=1),
    said when they do; refused (JobError) when the variable is set to anything else."""
    from seedvr2x.runtime.job import JobError

    value = os.environ.get(NUMZ_TAIL)
    if value is None:
        return False
    if value != "1":
        raise JobError(f"{NUMZ_TAIL}={value!r}: for seedvr2x's tests only, 1 or unset")
    logger.warning("%s=1: numz's padding of a shot's end, for seedvr2x's tests only", NUMZ_TAIL)
    return True


# The output directories this process holds locked (_lock), until main returns: their
# descriptors, and the directories themselves.
_LOCKS: list[int] = []
_LOCKED: list[Path] = []


def _lock(directory: Path) -> None:
    """Lock directory for this process until main returns: one seedvr2x at a time writes an
    output directory, since two would race on the manifest and discard each other's files in
    progress (resume.leftovers). The lock is the kernel's (flock), on the directory itself, so
    no file is added, and it goes with the process, however it ends. Once per directory: a
    .mkv path's work directory is locked from its start (_copied), an output directory once its
    job is known (_run), and a second lock on the same directory, even this process's own,
    would be refused, flock's locks being its descriptors'."""
    from seedvr2x.media.files import make_directories
    from seedvr2x.runtime.job import JobError

    if directory in _LOCKED:
        return
    make_directories(directory)
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        raise JobError(f"{directory}: another seedvr2x is writing it") from None
    except OSError as error:
        # A filesystem without locks: logged, the run goes on (AGENTS.md: no silent fallback).
        logger.warning("%s: not locked (%s); let one seedvr2x at a time write it", directory, error)
    _LOCKS.append(descriptor)
    _LOCKED.append(directory)


# The work directories this run made for a .mkv path's job (_copied), removed as main returns
# when still empty: a run refused before its first write leaves nothing.
_MADE: list[Path] = []


def _unmake() -> None:
    import contextlib

    while _MADE:
        with contextlib.suppress(OSError):  # the job's files are there, or the directory gone
            _MADE.pop().rmdir()


def _unlock() -> None:
    _LOCKED.clear()
    while _LOCKS:
        os.close(_LOCKS.pop())


@dataclass(frozen=True)
class _Identity:
    """What a job's output depends on besides its frames (DESIGN.md, Pause and resume): its
    settings, the models by hash, and the environment, both recorded in the job's manifest; and
    the GPU it runs on."""

    settings: dict[str, object]
    device: "torch.device"
    environment: dict[str, object]


@dataclass(frozen=True)
class _Prior:
    """A job resumed, as checked against its record before its first pass (_prior)."""

    recorded: dict[str, Any]  # its manifest, as written
    identity: _Identity
    # Its environment's differences: accepted (--accept-env-change), or a finished directory
    # job's, which makes nothing and asks no acceptance (_prior).
    changed: list[str]
    known: "FirstPass | None"  # the first pass's record, unless ffmpeg or conversions changed
    # A plan planned again: its cut settings' differences (resume.CUT_SETTINGS); else none.
    replanned: list[str] = field(default_factory=list[str])


def _identity(
    args: argparse.Namespace,
    cuts: list[int],
    ffmpeg_version: str,
    conversions: str,
    numz_padding: bool,
    numz_draw: bool,
    numz_tail: bool,
    model_files: "ModelFiles",
    made_in: dict[str, Any] | None = None,
) -> _Identity:
    """The job's identity, refused (JobError) without a CUDA GPU computing in bfloat16; its model
    files are checked before (weights.check_models, pull.check_pinned). The models are hashed
    before any GPU work, each once a run (pull.Hashes).

    made_in: for a job recorded with every segment made, which takes no GPU (_run), the
    environment its manifest records. Nothing is then asked of the GPU: no CUDA device is
    wanted, no torch.cuda called by seedvr2x (a library asks once, as it is imported, whether
    CUDA is there: environment.current), and the environment is this run's but for the GPU, its
    driver and cuDNN's version, which are the record's own, the card that made its segments
    (environment.current).
    The device is still named, and used by nothing."""
    from seedvr2x.runtime.job import JobError

    settings = _settings(args, cuts, numz_padding, numz_draw, numz_tail, model_files)
    import torch

    if made_in is not None:
        from seedvr2x.runtime import environment

        found = environment.current(None, ffmpeg_version, conversions, made_in)
        return _Identity(settings, torch.device("cuda", 0), found)
    if not torch.cuda.is_available():
        raise JobError("no CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise JobError("the GPU doesn't compute in bfloat16, numz's pipeline dtype")
    device = torch.device("cuda", 0)
    from seedvr2x.runtime import environment

    return _Identity(settings, device, environment.current(device, ffmpeg_version, conversions))


def _prior(
    args: argparse.Namespace,
    directory: Path,
    recorded: dict[str, Any],
    trusted: "FrameIndex | None",
    cuts: list[int],
    declared: "Declared",
    ffmpeg_version: str,
    conversions: str,
    numz_padding: bool,
    numz_draw: bool,
    numz_tail: bool,
    model_files: "ModelFiles",
    idle: bool = False,
) -> _Prior:
    """The job recorded in directory, its manifest's content `recorded`, checked against the one
    asked before its first pass: the same settings, environment and source, the source being its
    content (resume.identity), or refused (JobError), but for an environment change accepted
    (--accept-env-change), and for a plan's cuts. The record of the first pass is trusted, and
    the pass isn't run again, with its index `trusted` (_trusted): unless ffmpeg or its
    conversions changed, the same bytes, decoded by the same build, give the same frames, and
    nothing else takes part in the pass's decode (DESIGN.md, Pause and resume); the shot
    detector's probabilities are the record's then, never made again.

    A plan, a job that has made no unit yet (manifest.planned), is planned again when the settings
    asked differ from its own in how its source is cut alone (resume.CUT_SETTINGS): another
    --cut-threshold, a cut list given or another, or none, its cuts derived again from its first
    pass's record (DESIGN.md, Shot detection: another threshold gives its cuts without a second
    decode or detection), the layout and settings rewritten (_replan): DESIGN.md, Pause and
    resume, A plan, "A run whose settings differ only in its cuts", as resume.CUT_SETTINGS says.

    A finished directory job, every segment made, is taken whatever its environment, with no
    --accept-env-change (design, 2026-10-12: "A finished directory job is said finished whatever
    the environment, nothing being made: no refusal, no --accept-env-change asked"): nothing is
    made in this one, so no output can differ, and the difference is said (ENVIRONMENT_STAYS),
    by _finished_already or by _resume. Its environment alone: other settings, another source
    or, after this, another layout or plan (_resume) are another job, as before. Not a .mkv
    path's job with every segment made and no file, which isn't finished, its assembly the next
    unit, made in this environment: refused, or accepted and recorded, as any resume.
    With --plan FILE too, which writes the job's cut list and asks no flag either (design,
    2026-10-12; refused without one until then): under another stack the list is written, the
    manifest untouched; under another ffmpeg or other conversions it isn't, the frame index it
    is written from not trusted there (_finished_already).

    idle: the job has every segment made and takes no GPU (_run): its identity is made without
    one (_identity), so that its GPU's name is no difference, not being asked."""
    from seedvr2x.media.scan import Idet
    from seedvr2x.media.source import FirstPass
    from seedvr2x.runtime import resume
    from seedvr2x.runtime.manifest import NAME, complete, planned

    path = directory / NAME
    made_in: dict[str, Any] | None = None
    if idle:
        # Not a record of this code's, without an environment: refused below, as another job.
        was: Any = recorded.get("environment")
        made_in = cast(dict[str, Any], was) if isinstance(was, dict) else {}
    identity = _identity(
        *(args, cuts, ffmpeg_version, conversions),
        *(numz_padding, numz_draw, numz_tail, model_files, made_in),
    )
    asked = {
        "settings": identity.settings,
        "environment": identity.environment,
        "input": _content(declared.path),
    }
    found = resume.differences(resume.identity(recorded), resume.identity(asked))
    changed = [line for line in found if resume.section(line) == "environment"]
    recut = [line for line in found if resume.setting(line) in resume.CUT_SETTINGS]
    other = len(found) > len(changed) + len(recut)
    replanned = recut if recut and not other and planned(recorded) else []
    finished = _final(args.output) is None and complete(recorded)
    if other or recut != replanned or (changed and not args.accept_env_change and not finished):
        raise _another_job(
            path,
            found,
            only_environment=len(changed) == len(found),
            units_made=bool(recut) and not other and not replanned,
            plan_elsewhere=bool(replanned) and bool(changed),
        )
    entry: dict[str, Any] = recorded["input"]
    if entry["path"] != str(declared.path.resolve()):
        logger.info(
            "%s: the input recorded at %s, moved: the same content", declared.path, entry["path"]
        )
    if trusted is None:
        return _Prior(recorded, identity, changed, None, replanned)
    logger.info(
        "%s: the same input and ffmpeg, the first pass as recorded%s",
        directory,
        "" if trusted.probabilities is None else ", the shot detector's probabilities with it",
    )
    found = FirstPass(
        entry["frames"], entry["sha256"], trusted, Idet.from_record(entry.get("idet"))
    )
    return _Prior(recorded, identity, changed, found, replanned)


def _detecting(recorded: dict[str, Any] | None) -> bool:
    """Whether the job a manifest records detects its shots, its settings naming the shot
    detector's file (_settings); true without a record, a new job detecting unless a cut list is
    given."""
    if recorded is None:
        return True
    settings: Any = recorded.get("settings")
    return isinstance(settings, dict) and "detector_model" in settings


def _trusted(
    directory: Path,
    recorded: dict[str, Any],
    ffmpeg_version: str,
    conversions: str,
) -> "tuple[FrameIndex | None, str]":
    """The frame index of the first pass the manifest `recorded` in directory records, when a
    resume may trust that record (DESIGN.md, Pause and resume): made by the same ffmpeg and
    conversions (manifest.FIRST_PASS), and there as recorded (_recorded_index); else None, the
    first pass then run again, but a finished job's (_finished_already), and why, when the index
    is the reason, for the run to say. Read before the model files are, so that a resume trusting
    it neither fetches nor reads the shot detector's file, which it doesn't run, its probabilities
    there (cli._run); whether the job is the one recorded is _prior's to say. Nothing here imports
    torch, which the model files' check comes before (DESIGN.md, Weights, Recognised by content):
    runtime/resume.py does, and isn't read."""
    from seedvr2x.runtime.manifest import FIRST_PASS

    environment: Any = recorded.get("environment")
    entry: Any = recorded.get("input")
    if not isinstance(environment, dict) or not isinstance(entry, dict):
        return None, ""  # not this code's record: _prior refuses the job
    # The environment's fields the first pass depends on, as runtime/environment.py records them.
    current = {"ffmpeg": ffmpeg_version, "conversions": conversions}
    before = cast(dict[str, Any], environment)
    if any(before.get(key) != current[key] for key in FIRST_PASS):
        return None, ""
    # DESIGN.md, Pause and resume: "The frame index is derived data: missing or damaged on a
    # resume, the first pass runs again and writes it": as an input copy is, never trusted
    # unchecked, nor a reason to refuse the job.
    return _recorded_index(directory, cast(dict[str, Any], entry))


def _recorded_index(directory: Path, entry: dict[str, Any]) -> "tuple[FrameIndex | None, str]":
    """The frame index the input's record names, read from directory and checked against the
    record (manifest.INDEX): its size, SHA-256 and frames; or None, and why it isn't trusted. Its
    shot detector's probabilities, when there, come with it: a record this code wrote has them
    when its job detected its shots, which its settings say (_detecting), or when it is a
    detected plan planned again by a cut list, which keeps them (_replan), so that a later
    threshold needs no second decode."""
    import hashlib

    from seedvr2x.media.index import FrameIndex
    from seedvr2x.runtime.manifest import INDEX

    path = directory / INDEX
    recorded: Any = entry.get("index")
    if not isinstance(recorded, dict):
        return None, f"{path}: not in the manifest's record of the input"
    named = cast(dict[str, Any], recorded)
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None, f"{path}: the frame index missing"
    except OSError as error:
        return None, f"{path}: the frame index not readable: {error}"
    if len(data) != named.get("bytes"):
        return None, f"{path}: {len(data)} bytes, where the manifest recorded {named.get('bytes')}"
    if hashlib.sha256(data).hexdigest() != named.get("sha256"):
        return None, f"{path}: not the frame index the manifest recorded, by its SHA-256"
    try:
        index = FrameIndex.from_bytes(data)
    except ValueError as error:
        return None, f"{path}: not readable as a frame index: {error}"
    if index.frames != entry.get("frames"):
        return None, f"{path}: {index.frames} frames, where the record has {entry.get('frames')}"
    return index, ""


def _content(path: Path) -> dict[str, object]:
    """The source's content, as a resume compares it: its size and SHA-256."""
    from seedvr2x.media.files import sha256

    started = time.monotonic()
    digest = sha256(path)
    logger.info("%s: SHA-256 %s, in %.1f s", path, digest, time.monotonic() - started)
    return {"bytes": path.stat().st_size, "sha256": digest}


def _resume(record: "Manifest", prior: _Prior, assembles: bool = False) -> bool:
    """Take up the job recorded beside record, checked before its first pass (_prior): refused
    (JobError) unless the rest is the job asked too, what the first pass found and the layout
    following from it; its directory must be as the manifest says. Then record the environment
    change accepted, and discard what a stop left that the manifest doesn't name
    (resume.leftovers). The manifest is written by the next unit made: a resume stopped
    before one keeps the record as it was, but for the index of a first pass run again and for
    a plan's (_taken_up).

    A finished job makes no unit, so nothing records an environment change for it, accepted or
    not, as it needs no acceptance (_prior), and that is what is said (ENVIRONMENT_STAYS, as
    _finished_already says it of one without its first pass's record): a directory's job with
    every segment finished. A .mkv path's (`assembles`) with every segment finished isn't
    finished, its file not in place (DESIGN.md, Pause and resume: "A finished file's job is
    finished once the file is in place"): its assembly is the next unit made, which writes the
    manifest (_assemble).

    A plan, a job that has made no unit yet (manifest.planned), is planned again whole from the
    memory free now (_run; DESIGN.md, Memory planner: "A plan recorded with no unit made yet is
    made again whole"): its plan's shape may differ from the one recorded, and its shots' windows
    with it, which is no other job. Such a plan, and one planned again for its cuts (_prior),
    takes the new plan and layout at once (_replan): True then, the manifest rewritten; False for
    a job taken up as it is recorded."""
    from seedvr2x.runtime import resume
    from seedvr2x.runtime.manifest import complete, planned

    plan = planned(prior.recorded)

    def reshaped(line: str) -> bool:
        # A plan's shape, made again, and the windows it gives each shot.
        return plan and (
            resume.section(line) == "plan" or re.match(r"shots\[\d+\]\.windows\b", line) is not None
        )

    def compared(line: str) -> bool:
        # The environment compared before the first pass; a plan's cuts, accepted there.
        if resume.section(line) == "environment" or reshaped(line):
            return False
        recut = resume.setting(line) in resume.CUT_SETTINGS
        return not prior.replanned or not (recut or resume.section(line) in ("shots", "segments"))

    differences = resume.differences(prior.recorded, record.content())
    found = [line for line in differences if compared(line)]
    if found:
        raise _another_job(record.path, found, only_environment=False)
    shape = [line for line in differences if reshaped(line)]
    if prior.replanned or shape:
        _replan(record, prior, shape)
        return True
    resume.adopt(record, prior.recorded)
    if prior.changed and complete(prior.recorded) and not assembles:
        logger.info(ENVIRONMENT_STAYS, record.path.parent, "; ".join(prior.changed))
    elif prior.changed:
        change = resume.environment_change(prior.recorded, record.environment)
        record.environment_changes.append(change)
        logger.warning(
            "%s: resumed in another environment, as accepted, which its manifest records %s: %s",
            record.path.parent,
            # A plan's at once (_taken_up).
            "at once, a plan having made no unit in the one before"
            if planned(prior.recorded)
            else "with the next unit made",
            "; ".join(prior.changed),
        )
    discarded = resume.leftovers(record)
    for path in discarded:
        logger.debug("discarded %s", path)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    if complete(prior.recorded) and not assembles:
        # A finished job resumes nothing (design, 2026-10-12): no "resuming DIR: 2 of 2 segments
        # finished" between its environment's line and "finished already"; what was discarded
        # is said as _finished_already says it.
        if discarded:
            logger.info("%s: %s discarded", record.path.parent, _plural(len(discarded), "leftover"))
        return False
    logger.info(
        "resuming %s: %d of %d segments finished, %d of %d shots encoded, %d windows kept; %s"
        " discarded",
        record.path.parent,
        sum(record.finished),
        len(record.finished),
        sum(record.encoded),
        len(record.encoded),
        sum(record.windows_done),
        _plural(len(discarded), "leftover"),
    )
    return False


def _taken_up(record: "Manifest", prior: _Prior) -> None:
    """What a job taken up as it is recorded (_resume) writes before any unit: nothing, unless its
    first pass ran again or it is a plan taken up in another environment. The frame index is
    written only with a manifest that names it and records the environment it was made in
    (manifest.INDEX).

    A plan, a job that has made no unit yet (manifest.planned), whose first pass ran again or
    whose environment changed, accepted (--accept-env-change): its index, when made again, then
    its manifest, at once, in this run's environment, the change recorded
    (environment_changes). No unit of it was made in the environment recorded, so nothing is
    left to resume there, and its record then stands for what this run made and found: written
    as recorded, a plan whose pass ran again under another ffmpeg named that ffmpeg's index
    beside the first one's name, and the first ffmpeg then trusted it (a review's finding,
    2026-10-10). DESIGN.md, Pause and resume: "Two things rewrite it at once: a plan taken up or
    planned again (below), and a frame index made again".

    A job that has made units and isn't finished, its first pass run again (_prior; a finished
    job's isn't, and nothing of it is written: _finished_already):

    Its index made in the environment recorded, the recorded one being missing or damaged: at
    once, then the manifest as recorded, naming it, so that the next run trusts the record, a unit
    made after it or not. Left to the next unit, a job stopped before one kept index bytes its
    manifest didn't name when the pass gave others (a damaged source decodes otherwise from one
    pass to the next: media/reader.py), and every later run warned of them and ran the pass
    again.

    Made in another environment, accepted (ffmpeg or its conversions changed): with the next unit
    made, whose manifest records that environment (DESIGN.md, Pause and resume), the directory
    keeping the recorded index until then: a resume stopped before a unit "still resumes in its
    old environment", its record whole.

    So DESIGN.md's rule, a resume stopped before a unit "leaves the manifest as it was", holds
    but for its two exceptions, "a plan taken up or planned again (below), and a frame index
    made again": the index made again in the environment recorded, as above. One made again
    under another ffmpeg, accepted, waits for the next unit with that environment, by the rule
    itself; DESIGN.md's sentence doesn't tell the two apart (said to design, 2026-10-12)."""
    from dataclasses import replace

    from seedvr2x.runtime import resume
    from seedvr2x.runtime.manifest import FIRST_PASS, planned

    directory = record.path.parent
    source = record.source
    if planned(prior.recorded):
        if prior.known is None:
            _write_index(directory, source)
        if prior.known is None or prior.changed:
            record.write()
        return
    if prior.known is not None:
        return
    before: dict[str, Any] = prior.recorded["environment"]
    if any(before.get(key) != record.environment.get(key) for key in FIRST_PASS):
        record.before_write = lambda: _write_index(directory, source)
        return
    _write_index(directory, source)
    # The environment as recorded, the first pass's own fields being the same (FIRST_PASS): a
    # torch or GPU change accepted by this run is no part of the index's record, and is written
    # "with the next unit made" (DESIGN.md, Pause and resume), as any resume's is (kept by
    # design, 2026-10-10).
    as_recorded = replace(
        record,
        environment=before,
        environment_changes=list(prior.recorded.get(resume.CHANGES, [])),
    )
    as_recorded.write()


# The input record's fields a source declares, read without a decode (media/source.py, declare;
# manifest._input): what a finished job run again is still compared by (_finished_already).
DECLARED = ("frame_rate", "size", "sample_aspect", "read_as", "pix_fmt", "primaries", "transfer")
# What is said of a finished job run again in another environment: its directory, then the
# differences. No unit is made in it, so no manifest written records it (_finished_already;
# _resume, for a finished job whose first pass's record is trusted). The same line with
# --accept-env-change and without, which such a job doesn't need (design, 2026-10-12; _prior):
# "as accepted" stood after "this one" until then, the flag being asked.
ENVIRONMENT_STAYS = (
    "%s: its environment differs from this one and stays its manifest's, nothing being made in"
    " this one: %s"
)


def _finished_already(
    args: argparse.Namespace,
    directory: Path,
    prior: _Prior,
    declared: "Declared",
    target: tuple[int, int],
    untrusted: str,
) -> int | None:
    """A finished job run again, the record of its first pass not one to trust (_trusted, why in
    `untrusted`): its frame index missing or damaged, or made by another ffmpeg, which needs no
    acceptance (_prior; design, 2026-10-12). It is said finished and nothing is made again
    (DESIGN.md, Pause and resume: "Re-running a finished job doesn't rewrite its manifest, and
    makes nothing again, its frame index and its plan included: no memory is read"): no first
    pass, no shot detector nor its file (cli._run), no index, no plan nor the GPU's free memory
    read for one, no manifest written; its exit status, 0. None for any other job, which the run
    goes on with: one unfinished, or a finished one whose record is trusted, for which no pass
    runs either (_prior) and no plan is made (cli._run), said finished once it is compared whole
    (_resume).

    A directory's job alone. A .mkv path's job with every segment finished isn't finished: "A
    finished file's job is finished once the file is in place: with every segment made and no
    file yet, the same command assembles" (DESIGN.md, Pause and resume), and the assembly takes
    the source's first timestamp and time base from the frame index (assembly.start). None for
    it then, whatever became of its index: the run goes on as for any job resumed, its first
    pass run again when its record isn't trusted, with the shot detector for a job that detects
    its shots, the index written again (_taken_up), then the file assembled (cli._run). One
    whose file is in place never comes here: its record of the assembly answers first
    (_assembled), before the index is looked at.

    The job is checked before this as any job resumed is before its first pass (_prior): the
    same settings, the pins that shape its plan among them, and source by content, or refused,
    its environment said, never refused, with --plan or without (_prior). Of what a resume compares
    after its first pass (_resume), it is held to what needs no decode: what the source declares
    (DECLARED) and the output's format, size (`target`, its (height, width) before the output's
    rounding) and rate, against its record, another job when they differ (JobError), as an
    accepted ffmpeg that reads the source otherwise makes a resume. Its frame count, the places
    filled among them, its shots, windows, segments and plan are the record's own: the pass that
    counts the frames and the probabilities that cut them are what isn't made again, and a
    finished job's plan stays as recorded with an index too (cli._run). Its directory is held to
    its manifest as a resume's is (resume.leftovers): a finished segment missing or of another
    size, its checksums missing, or anything not the job's, refused; what a stop left, discarded.

    PROVISIONAL (implementation, 2026-10-11, a question for design: DESIGN.md says a finished job
    makes nothing again, not what it is still compared by without its index): the list above; the
    recorded plan isn't read, where a finished job with its index is refused for one the planner
    never makes (_recorded_plan), a manifest edited by hand.

    PROVISIONAL (implementation, 2026-10-10, a question for design: DESIGN.md has --plan FILE
    write a job's cut list, and a finished job make nothing again): --plan FILE on such a job
    writes nothing and fails, 1, saying why: the list is derived from the frame index and timed
    by it, and no first pass is run to make one for a job that needs none."""
    from seedvr2x.media.scan import Idet
    from seedvr2x.media.source import Source
    from seedvr2x.runtime import resume
    from seedvr2x.runtime.job import output_size
    from seedvr2x.runtime.manifest import NAME, complete

    recorded = prior.recorded
    if _final(args.output) is not None or prior.known is not None or not complete(recorded):
        return None
    entry: dict[str, Any] = recorded["input"]
    # The source as declared now, its frames the record's count, in places, and no index: nothing
    # reads a frame, and what the index alone says, the places filled (Source.filled), is the
    # record's, not compared.
    source = Source(
        path=declared.path,
        stream=declared.stream,
        conversion=declared.conversion,
        frames=entry["frames"],
        sample_aspect=declared.sample_aspect,
        sha256=entry["sha256"],
        index=None,
        rate_override=declared.rate_override,
        idet=Idet.from_record(entry.get("idet")),
    )
    job = resume.recorded_job(directory / NAME, recorded, source)
    height, width = output_size(target)
    output = {"format": args.format, "size": [width, height], "frame_rate": str(source.frame_rate)}
    asked = {
        "input": {name: job.content()["input"][name] for name in DECLARED},
        "output": output,
    }
    before = {
        "input": {name: entry.get(name) for name in DECLARED},
        "output": {name: recorded["output"].get(name) for name in output},
    }
    found = resume.differences(before, asked)
    if found:
        raise _another_job(job.path, found, only_environment=False)
    discarded = resume.leftovers(job)
    for path in discarded:
        logger.debug("discarded %s", path)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    if untrusted:
        # Said, the directory holding less than its manifest names; left as it is.
        logger.warning("%s: a finished job needs none, and makes nothing again", untrusted)
    if prior.changed:
        logger.info(ENVIRONMENT_STAYS, directory, "; ".join(prior.changed))
    if discarded:
        logger.info("%s: %s discarded", directory, _plural(len(discarded), "leftover"))
    logger.info("%s: finished already, its %d segments", directory, len(job.files))
    if args.plan is None:
        return 0
    logger.error(
        "--plan %s: not written: %s is finished already, and a job's cut list is written from its"
        " frame index, which %s and which a finished job doesn't make again: plan the source in"
        " another output directory (-o) for its cut list",
        args.plan,
        directory,
        "isn't there as its manifest records it"
        if untrusted
        else "another ffmpeg or other conversions than this run's made",
    )
    return 1


def _replan(record: "Manifest", prior: _Prior, shape: list[str]) -> None:
    """Take up a plan with other cuts (_prior), or whose plan the memory free now shapes
    otherwise (_resume), `shape` the differences in its plan and its shots' windows: what a stop
    left of it, found by its own layout (resume.recorded_job, resume.leftovers: anything not its
    own refused, never deleted), is discarded, its units' directories with it, none of them the
    new layout's; its first pass's index replaced when the pass ran again (a plan cut by a list,
    planned again without one, or one planned again under another ffmpeg, accepted); then its
    manifest rewritten with the new settings, plan and layout, at once, so that the plan is the
    new one whether a unit follows or not, in this run's environment, a change accepted recorded
    with it, as a plan taken up is (_taken_up). DESIGN.md, Pause and resume, A plan: "the cuts
    are derived again and the manifest rewritten at once, in the run's environment"
    (resume.CUT_SETTINGS)."""
    from seedvr2x.runtime import resume
    from seedvr2x.runtime.manifest import STATE

    directory = record.path.parent
    planned = resume.recorded_job(record.path, prior.recorded, record.source)
    discarded = resume.leftovers(planned)
    for path in discarded:
        logger.debug("discarded %s", path)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    # Only the plan's units' directories are left there, emptied (leftovers): rmdir, which
    # refuses a directory not empty, removes nothing else.
    state = directory / STATE
    if state.is_dir():
        for shot in state.iterdir():
            shot.rmdir()
        state.rmdir()
    record.environment_changes = list(prior.recorded.get(resume.CHANGES, []))
    if prior.changed:
        record.environment_changes.append(
            resume.environment_change(prior.recorded, record.environment)
        )
        logger.warning(
            "%s: planned again in another environment, as accepted, which its manifest records"
            " at once, a plan having made no unit in the one before: %s",
            directory,
            "; ".join(prior.changed),
        )
    if prior.known is None:
        _write_index(directory, record.source)
    record.write()
    # What changed, in one line: the cuts' settings, the plan's shape, and, the shots being the
    # same ones, how many of them it lays out otherwise.
    changed = [*prior.replanned, *(line for line in shape if resume.section(line) == "plan")]
    laid_out = {line.partition("]")[0] for line in shape if resume.section(line) == "shots"}
    if laid_out and not prior.replanned:
        changed.append(f"the windows of {_plural(len(laid_out), 'shot')}")
    logger.info(
        "%s: re-planned, its job having made no unit yet: %s; %s discarded",
        directory,
        "; ".join(changed),
        _plural(len(discarded), "leftover"),
    )


def _another_job(
    path: Path,
    found: list[str],
    only_environment: bool,
    units_made: bool = False,
    plan_elsewhere: bool = False,
) -> "JobError":
    """A resume refused, each difference listed, `where: recorded -> now`; `units_made` when the
    cuts differ alone, which a plan would take (_prior), but the job has made units;
    `plan_elsewhere` when a plan's cuts differ, which it takes, and its environment too, which
    --accept-env-change takes."""
    from seedvr2x.runtime.job import JobError

    return JobError(
        f"{path}: another job than the one asked, which differs in:\n  "
        + "\n  ".join(found[:20])
        + (f"\n  and {len(found) - 20} more" if len(found) > 20 else "")
        + (
            "\nOnly its environment differs: --accept-env-change resumes it anyway, though"
            " its output then differs from an uninterrupted run's"
            if only_environment
            else ""
        )
        + (
            "\nIts cuts change only while it is a plan, before its first unit, and it has made"
            " some: its cuts are its own; another output directory takes new ones"
            if units_made
            else ""
        )
        + (
            "\nA plan, which has made no unit yet, is planned again with other cuts; its"
            " environment differs too: --accept-env-change takes it up anyway, its first pass"
            " run again where ffmpeg or its conversions changed"
            if plan_elsewhere
            else ""
        )
    )


def _model(file: "ModelFile", model_files: "ModelFiles") -> dict[str, object]:
    """A model as the manifest records it: its file's name, as given, its size and SHA-256, models
    being identified by hash (DESIGN.md, Options kept and dropped). The 7B fp16 DiT is 16.5 GB to
    read, once a run: a file seedvr2x pins was hashed by its check already (pull.Hashes). One of
    seedvr2x's own that a job with every segment made doesn't have, held to its pin
    (pull.ModelFile.absent): the pin's size and SHA-256, those its record has where the file
    passed its pin check, as the shot detector's is taken on a trusted resume
    (_detector_model)."""
    from seedvr2x.runtime import pull

    if file.absent:
        pin = pull.PINNED[file.name]
        return {"name": file.name, "size": pin.size, "sha256": pin.sha256}
    digest = model_files.hashes.sha256(file.path)
    return {"name": file.name, "size": file.path.stat().st_size, "sha256": digest}


def _detector_model(model_files: "ModelFiles") -> dict[str, object]:
    """The shot detector as the manifest records it (_model): its file, when the run resolved it;
    else, on a resume whose first pass's record is trusted (_trusted), which runs no detector and
    so neither fetches nor reads its file, the file by its pin: the only one seedvr2x runs in that
    role (pull.check_pinned refuses any other), so the one any run of this code recorded.
    Accepted by design, 2026-10-12 (DESIGN.md records the models by hash, and says no more of
    this one): settings' detector_model is recorded by the pin on a resume whose record is
    trusted."""
    from seedvr2x.runtime import pull

    if model_files.detector is not None:
        return _model(model_files.detector, model_files)
    pin = pull.PINNED[pull.DETECTOR]
    return {"name": pull.DETECTOR, "size": pin.size, "sha256": pin.sha256}


def _cut_threshold(args: argparse.Namespace) -> float:
    """The detection's threshold: --cut-threshold's, else THRESHOLD."""
    return THRESHOLD if args.cut_threshold is None else args.cut_threshold


@dataclass(frozen=True)
class _Cuts:
    """How a job's source is cut (_source_cuts): its cuts, and the possible ones a detection
    found (--plan's list); the threshold of a detection, as the manifest records it, or the cut
    list given."""

    cuts: "list[Cut]"
    possible: "list[Cut]"
    threshold: str | None  # the detection's, None for a cut list's
    listed: Path | None  # the cut list given, None for a detection's


def _source_cuts(args: argparse.Namespace, listed: list[int], source: "Source") -> _Cuts:
    """The cuts of the job's source: the cut list's, `listed`, as read (--cuts); else detected
    at --cut-threshold from the shot detector's probabilities, its first pass's or its record's
    (DESIGN.md, Shot detection: "the cuts derive from them and the threshold"), with the
    possible ones."""
    from seedvr2x.runtime.cuts import Cut, detected, possible

    if args.cuts is not None:
        return _Cuts([Cut(frame) for frame in listed], [], None, args.cuts)
    probabilities = None if source.index is None else source.index.probabilities
    if probabilities is None:
        # The detector runs in every first pass of a job detecting its shots, and a record
        # without them isn't trusted then (_run).
        raise ValueError(f"{source.path}: no shot probabilities to cut it by: a seedvr2x bug")
    threshold = _cut_threshold(args)
    found = detected(probabilities, threshold), possible(probabilities, threshold)
    return _Cuts(*found, repr(threshold), None)


def _said(found: _Cuts) -> str:
    """How the shots were found, as a run's log says it: "61 cuts detected at 0.3, and 39
    possible cuts from 0.1 up to 0.3, which --plan lists", or "3 cuts from the cut list
    cuts.txt"."""
    if found.threshold is None:
        return f"{_plural(len(found.cuts), 'cut')} from the cut list {found.listed}"
    said = f"{_plural(len(found.cuts), 'cut')} detected at {found.threshold}"
    if float(found.threshold) <= POSSIBLE:
        return said
    possible = _plural(len(found.possible), "possible cut")
    return f"{said}, and {possible} from {POSSIBLE} up to {found.threshold}, which --plan lists"


def _counts(found: _Cuts, shots: int) -> str:
    """The counts --plan prints (DESIGN.md, Shot detection): "61 cuts at 0.3 (62 shots), 39
    possible cuts from 0.1 up to 0.3"; "3 cuts from the cut list cuts.txt (4 shots)"."""
    cuts, shots_said = _plural(len(found.cuts), "cut"), _plural(shots, "shot")
    if found.threshold is None:
        return f"{cuts} from the cut list {found.listed} ({shots_said})"
    said = f"{cuts} at {found.threshold} ({shots_said})"
    if float(found.threshold) <= POSSIBLE:
        # No run from POSSIBLE up peaks under such a threshold (cuts.possible).
        return f"{said}, no possible cuts at a threshold of {POSSIBLE} or under"
    possible = _plural(len(found.possible), "possible cut")
    return f"{said}, {possible} from {POSSIBLE} up to {found.threshold}"


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _taken(args: argparse.Namespace, directory: Path, where: Path) -> str | None:
    """What of the job's the resolved path `where` would take the place of, as a refusal says it
    (_plan_file), or None: a .mkv path's finished file or its checksums, or a place in the job's
    directory, the output's or a .mkv path's work directory, which holds the job's own files
    only (resume.leftovers). One test for --plan FILE and for each playlist written beside it
    (cuts.playlist_path), so that what refuses the one refuses the others (design, 2026-10-12:
    "the playlist is never the output file, its checksums beside it, its work directory, nor the
    source"; the source is each one's own line in _plan_file, the cut list given being FILE's
    alone, between the two)."""
    final = _final(args.output)
    # The same file, not the same name (_same, as _output holds the output to the source): see
    # _plan_file's first test.
    if final is not None and _same(where, final):
        return "the output"
    if final is not None and _same(where, _checksums_beside(final)):
        return "the output's checksums"
    if where.is_relative_to(directory.resolve()):
        what = "the output directory" if final is None else f"the work directory of {final}"
        return f"in {what}, which holds the job's own files only"
    return None


def _plan_file(args: argparse.Namespace, directory: Path) -> None:
    """Refuse a --plan FILE (JobError) that would take the place of what the job reads or writes:
    the source, the cut list given (--cuts), a .mkv path's finished file or its checksums, or
    anything in the job's directory, the output's or a .mkv path's work directory, which holds
    the job's own files only (resume.leftovers); or that can't be
    written: a directory, in none, or in one that takes no new file. Before the first pass, which
    may take an hour. DESIGN.md has --plan "writes the detected cut list to FILE" (Memory planner)
    and names no refusal of FILE: these, accepted by design as built (2026-10-10).

    Likewise the two playlists, of its detected cuts and of its possible cuts, written beside it
    under its name with .cuts.m3u8 and .possible.m3u8 added (cuts.playlist_path), when the shots
    are detected: a cut list's plan has neither, and no playlist. Each one's place is held to
    FILE's own tests (_taken), for a directory's job and a .mkv path's alike: it is reached by a
    link at the playlist's name, or is the output directory itself, one named FILE.cuts.m3u8."""
    import tempfile

    from seedvr2x.runtime.job import JobError

    plan: Path = args.plan
    where = plan.resolve()
    # The same FILE as the source, by whatever name (_same), not the same resolved name alone: a
    # file system that takes a name in any case (macOS's default, a Windows disk under WSL) has
    # FILM.MKV for film.mkv, two names no resolved path makes equal, and the cut list renamed
    # into its place (files.write_whole) would replace the source, the one thing a job can't
    # make again. Found by a review (2026-10-12), read-only on such a disk: os.path.samefile
    # true of two spellings whose resolved paths differ. A FILE that doesn't exist yet is no
    # file to compare: its resolved name then, as before. The same for the output's files
    # (_taken) and for each playlist's place, below.
    if _same(plan, args.input):
        raise JobError(f"--plan {plan}: the source; it names the cut list's file")
    if args.cuts is not None and _same(plan, args.cuts):
        # The plan of a cut list is that list normalised: written over it, the list's own
        # comments went, the possible cuts it kept commented and their probabilities with them.
        # The same file by whatever name, as the source above (design, 2026-10-12: "the --cuts
        # list compared with FILE as a file"): --cuts list.txt --plan LIST.TXT, on a disk that
        # takes a name in any case, is that list too.
        raise JobError(
            f"--plan {plan}: the cut list given (--cuts), which it would replace, its comments"
            " lost; it names another file"
        )
    taken = _taken(args, directory, where)
    if taken is not None:
        # One of the output's files, which FILE isn't to name; or a place in the job's directory.
        names = "" if taken.startswith("in ") else "; it names the cut list's file"
        raise JobError(f"--plan {plan}: {taken}{names}")
    if plan.is_dir():
        raise JobError(f"--plan {plan}: a directory; it names the cut list's file")
    if not plan.parent.is_dir():
        raise JobError(f"--plan {plan}: no directory {plan.parent} to write it in")
    # A cut list's plan writes no playlist.
    for which in PLAYLISTS if args.cuts is None else ():
        playlist = playlist_path(plan, which)
        there = playlist.resolve()
        if _same(playlist, args.input):
            taken = "the source"
        else:
            taken = _taken(args, directory, there) or ("a directory" if playlist.is_dir() else None)
        if taken is not None:
            raise JobError(
                f"--plan {plan}: the playlist of its {which.cut}s, {playlist.name} beside it, is"
                f" {taken}; it names another file"
            )
    # The list is written beside its place, then renamed into it (files.write_whole): a file made
    # there and removed tells now, not once the first pass has run, whether the directory takes
    # one.
    try:
        with tempfile.NamedTemporaryFile(dir=plan.parent, prefix=f"{plan.name}."):
            pass
    except OSError as error:
        raise JobError(
            f"--plan {plan}: no file can be written in {plan.parent}: {error.strerror}"
        ) from None


def _write_plan(
    args: argparse.Namespace,
    source: "Source",
    found: _Cuts,
    shots: int,
    model_files: "ModelFiles",
) -> int:
    """--plan's end: the cut list written to FILE, whole, an existing one replaced (files.
    write_whole), in the cut list format (cuts.cut_list), and the counts said; 0, or 1 when FILE
    can't be written. After each cut, its probability and time (DESIGN.md, Shot detection: "A
    line holds the cut's frame, its probability, and its time from the file's start"), which
    --cuts ignores (DESIGN.md, Input); the header's wording is the code's, the list accepted by
    design as built (2026-10-10)."""
    import textwrap

    from seedvr2x.media.files import write_whole
    from seedvr2x.runtime.cuts import cut_list, places_note

    if source.index is None:
        raise ValueError(f"{source.path}: no frame index to time its cuts by: a seedvr2x bug")
    rate = source.frame_rate
    # No line of the header begins with a digit, nor holds a line break (cuts.cut_list).
    header = [f"seedvr2x cut list of {source.path.name}: {source.frames} frames at {rate} fps"]
    if found.threshold is not None:
        model = _detector_model(model_files)
        header.append(
            f"Shot detector: TransNetV2, {model['name']} (SHA-256 {str(model['sha256'])[:8]}),"
            f" threshold {found.threshold}"
        )
    header += [f"Found: {_counts(found, shots)}", ""]
    detected = found.threshold is not None
    peak = "the shot detector's peak probability, on the frame before it; " if detected else ""
    possible = " A possible cut is a commented line: checking it is a jump to its time in a player."
    lines = (
        "A line per cut: the frame it cuts at, the first frame of a shot, counted from zero;"
        f" {peak}the time of that frame from the source's start, H:MM:SS.mmm, as a player shows it."
        f" seedvr2x reads the frame number alone.{possible if detected else ''}"
    )
    usage = (
        "Uncomment a possible cut to keep it, delete a line to drop a cut"
        if detected
        else "Delete a line to drop a cut, add one to cut at its frame"
    )
    header += textwrap.wrap(lines, 86)
    header += places_note(source.index)
    header += textwrap.wrap(f"{usage}, then run with --cuts FILE, this file.", 86)
    text = cut_list(header, source.index, found.cuts, found.possible)
    try:
        write_whole(args.plan, text.encode())
    except OSError as error:
        logger.error("--plan %s: not written: %s", args.plan, error)
        return 1
    logger.info("%s: the cut list in %s", _counts(found, shots), args.plan)
    # The playlists of its cuts and of its possible cuts beside it, said there
    # (cuts.write_playlists); none for a cut list's plan, which detected neither.
    cuts = found.cuts if detected else ()
    return 0 if write_playlists(args.plan, source.index, cuts, found.possible, source.path) else 1


def _threshold(text: str) -> float:
    """--cut-threshold's P: a probability above 0 and at most 1 (DESIGN.md, Shot detection)."""
    try:
        value = float(text)
    except ValueError:
        value = math.nan
    if not 0 < value <= 1:  # NaN fails it too
        raise argparse.ArgumentTypeError(
            f"{text!r}: not a threshold; give a probability above 0 and at most 1, such as"
            f" {THRESHOLD}, the default"
        )
    return value


def _count(text: str) -> int:
    """A whole number, 0 or more."""
    if not re.fullmatch(r"\d+", text):
        raise argparse.ArgumentTypeError(f"{text!r}: not a whole number, 0 or more")
    return int(text)


def _tile(text: str) -> int:
    """A VAE tile's side in output pixels (DESIGN.md, Memory planner): a multiple of 64 from 512
    up, the smallest tile the planner makes; or 0, untiled."""
    if not re.fullmatch(r"\d+", text) or (int(text) and (int(text) < 512 or int(text) % 64)):
        raise argparse.ArgumentTypeError(
            f"{text!r}: not a tile's side: a multiple of 64 from 512 up, in output pixels, or 0"
            " for an untiled pass"
        )
    return int(text)


def _until(text: str) -> datetime.time:
    """A time of day on the local clock, HH:MM, of 24 hours."""
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", text)
    if match is None or int(match[1]) > 23 or int(match[2]) > 59:
        raise argparse.ArgumentTypeError(f"{text!r}: not a time of day, HH:MM, from 00:00 to 23:59")
    return datetime.time(int(match[1]), int(match[2]))


def _positive(text: str) -> int:
    """A whole number above 0."""
    if not re.fullmatch(r"\d+", text) or int(text) == 0:
        raise argparse.ArgumentTypeError(f"{text!r}: not a whole number above 0")
    return int(text)


def _seconds(text: str) -> Fraction:
    """A duration in seconds, 0 or more, exact: 4.99 is 499/100."""
    try:
        seconds = Fraction(text)
    except (ValueError, ZeroDivisionError):
        raise argparse.ArgumentTypeError(f"{text!r}: not a number of seconds") from None
    if seconds < 0:
        raise argparse.ArgumentTypeError(f"{text!r}: negative")
    return seconds


def _rate(text: str) -> Fraction:
    """A positive frame rate, exact, written N/D or as a whole number, as ffprobe's r_frame_rate
    gives it: 24000/1001, not 23.976."""
    match = re.fullmatch(r"(\d+)(?:/(\d+))?", text)
    if match is None or int(match[1]) == 0 or (match[2] is not None and int(match[2]) == 0):
        raise argparse.ArgumentTypeError(
            f"{text!r}: not a frame rate; write it N/D or as a whole number, exact: 24000/1001 for"
            " 23.976, 30000/1001 for 29.97, 25"
        )
    return Fraction(int(match[1]), int(match[2] or 1))


def _ratio(text: str) -> Fraction:
    """A positive ratio written N:D or N/D, as the sample aspect ratio option takes it."""
    match = re.fullmatch(r"(\d+)[:/](\d+)", text)
    if match is None or int(match[1]) == 0 or int(match[2]) == 0:
        raise argparse.ArgumentTypeError(f"{text!r}: not a positive ratio N:D")
    return Fraction(int(match[1]), int(match[2]))

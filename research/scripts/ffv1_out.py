#!/usr/bin/env python3
"""Lossless FFV1 master output for the SeedVR2 CLI, taken from the float frames.

The CLI's outputs are lossy (x264/x265 CRF 12, or mp4v) or 8-bit and truncated (PNG, bug 09),
so comparing two configurations through them mixes the encoder's error into the difference.
This wrapper runs inference_cli.py in-process and, without touching the checkout, also writes
every output frame to a Matroska/FFV1 file:

- the frames are taken where the CLI saves them (save_frames_to_video / save_frames_to_image:
  float32 [T, H, W, C] in [0, 1], converted from the bfloat16 final_video), before the uint8
  cast, and quantised by rounding: round(x * 65535) for 16 bits, round(x * 1023) for 10 bits
- RGB is stored as RGB (planar gbrp, piped as is: ffmpeg converts nothing), FFV1 level 3,
  intra only (-g 1), 16 slices with CRCs
- colour tags: matrix GBR (identity), BT.709 primaries and transfer, full range; the YUV
  variants are converted with an explicit BT.709 matrix to limited range (zscale) and tagged so
- the frame rate is the input's exact rational (ffprobe r_frame_rate, e.g. 24000/1001), given
  on the rawvideo input side
- the frames --prepend_frames adds are dropped (the CLI keeps them on one GPU, bug 05)

The CLI still writes its own output (mp4 or png) unless FFV1_OUT_KEEP=0. The .mkv goes next to
it: <output>.mkv for an mp4 (same stem), <png dir>.mkv for a PNG sequence; with bench.py,
<output dir>/<run name>.mkv. Single images are not handled (one frame: use the CLI's PNG).

Environment:
  FFV1_OUT_PIXFMT  gbrp16le (default) | gbrp10le | yuv444p10le | yuv420p10le; a comma list
                   writes one file per format, named <stem>.<pix_fmt>.mkv
  FFV1_OUT_PATH    output .mkv (a single input), or a directory for <stem>.mkv
  FFV1_OUT_KEEP    1 (default): the CLI writes its own output too; 0: skip it (faster, no
                   lossy/8-bit copy; the CLI's "Output saved to" line then names a missing file)
  FFV1_OUT_DROP    frames to drop at the start (default: --prepend_frames on one GPU, else 0)
  FFV1_OUT_FPS     frame rate override, rational ("24000/1001") or decimal
  FFV1_OUT_SLICES  FFV1 slices (default 16)
  FFV1_OUT_FFMPEG, FFV1_OUT_FFPROBE   binaries (default: ffmpeg / ffprobe on PATH)
  FFV1_OUT_DUMP    directory: also save some frames as float32 .npy, exactly as received
  FFV1_OUT_DUMP_FRAMES  comma list of .mkv frame indices to dump (default 0,1,2)

Usage (cwd = the SeedVR2 checkout, its venv's python):
  python ffv1_out.py inference_cli.py <CLI args>
  python3 bench.py run NAME --wrap ffv1_out.py --env FFV1_OUT_PIXFMT=gbrp10le -- <CLI args>
  python3 bench.py run NAME --wrap attn_probe.py --wrap ffv1_out.py -- <CLI args>   # chained
  python ffv1_out.py --selftest                  # synthetic frames through every pixel format
  python ffv1_out.py --verify OUT.mkv DUMP_DIR   # decoded .mkv vs the dumped float frames
  python ffv1_out.py --diff A.mkv B.mkv          # PSNR / max difference of two RGB masters

Needs numpy, and ffmpeg/ffprobe with ffv1 (and zscale for the YUV variants).
"""
import atexit
import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

# pix_fmt -> (bits of the RGB values piped to ffmpeg, ffmpeg output options)
RGB_TAGS = ["-colorspace", "rgb", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "pc"]
YUV_TAGS = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
# frame properties only, no pixel conversion: without them the muxer writes no primaries/transfer
SETPARAMS = "setparams=color_primaries=bt709:color_trc=bt709:colorspace=gbr:range=pc"
# RGB in, BT.709 matrix, limited range out; same primaries/transfer on both sides (no conversion)
ZSCALE = "zscale=rin=full:pin=709:tin=709:m=709:r=limited:p=709:t=709:d=none"
FORMATS = {
    "gbrp16le": (16, ["-vf", SETPARAMS] + RGB_TAGS),
    "gbrp10le": (10, ["-vf", SETPARAMS] + RGB_TAGS),
    "yuv444p10le": (16, ["-vf", f"{ZSCALE},format=yuv444p10le"] + YUV_TAGS),
    "yuv420p10le": (16, ["-vf", f"{ZSCALE}:c=left,format=yuv420p10le"] + YUV_TAGS
                    + ["-chroma_sample_location", "left"]),
}


def log(msg):
    print(f"ffv1_out: {msg}", file=sys.stderr, flush=True)


def ffmpeg_bin():
    return os.environ.get("FFV1_OUT_FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"


def ffprobe_bin():
    return os.environ.get("FFV1_OUT_FFPROBE") or shutil.which("ffprobe") or "ffprobe"


def to_planar(frame, bits):
    """float [H, W, 3] in [0, 1] -> uint16 G, B, R planes (gbrpNNle), rounded to nearest."""
    scale = (1 << bits) - 1
    q = np.clip(frame[..., :3], 0.0, 1.0)
    q *= scale
    np.rint(q, out=q)
    return np.ascontiguousarray(q.astype("<u2").transpose(2, 0, 1)[[1, 2, 0]])


def probe_fps(path):
    try:
        out = subprocess.run([ffprobe_bin(), "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=r_frame_rate", "-of", "csv=p=0", path],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        fps = Fraction(out)
        return fps if fps > 0 else None
    except Exception:
        return None


class Writer:
    """One ffmpeg process per output pixel format, fed planar 16-bit RGB on stdin."""

    def __init__(self, path, pix_fmt, width, height, fps, slices):
        if pix_fmt not in FORMATS:
            raise SystemExit(f"ffv1_out: unknown pixel format {pix_fmt!r} (choose from {', '.join(FORMATS)})")
        self.path, self.pix_fmt = str(path), pix_fmt
        self.bits, opts = FORMATS[pix_fmt]
        in_fmt = "gbrp10le" if self.bits == 10 else "gbrp16le"
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.err = tempfile.TemporaryFile()
        self.cmd = [ffmpeg_bin(), "-hide_banner", "-nostdin", "-nostats", "-loglevel", "error", "-y",
                    "-f", "rawvideo", "-pix_fmt", in_fmt, "-s", f"{width}x{height}",
                    "-framerate", str(fps), "-i", "-",
                    "-map", "0:v:0", "-fps_mode", "passthrough", *opts,
                    "-c:v", "ffv1", "-level", "3", "-g", "1", "-slices", str(slices), "-slicecrc", "1",
                    "-pix_fmt", pix_fmt, self.path]
        self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.err)
        self.frames = 0

    def write(self, planes):
        try:
            self.proc.stdin.write(memoryview(planes).cast("B"))
        except BrokenPipeError:
            self.close()
            raise RuntimeError(f"ffv1_out: ffmpeg died writing {self.path}")
        self.frames += 1

    def close(self):
        if self.proc is None:
            return 0
        try:
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        rc = self.proc.wait()
        self.proc = None
        self.err.seek(0)
        msg = self.err.read().decode(errors="replace").strip()
        self.err.close()
        if rc or msg:
            log(f"ffmpeg ({self.path}) exited with {rc}: {msg}" if rc else f"ffmpeg messages ({self.path}): {msg}")
        if rc:
            log(f"command: {' '.join(self.cmd)}")
        return rc


# ---------------------------------------------------------------- the wrapper

class State:
    def __init__(self):
        fmts = os.environ.get("FFV1_OUT_PIXFMT", "gbrp16le")
        self.pix_fmts = [f.strip() for f in fmts.split(",") if f.strip()]
        for f in self.pix_fmts:
            if f not in FORMATS:
                sys.exit(f"ffv1_out: unknown FFV1_OUT_PIXFMT {f!r} (choose from {', '.join(FORMATS)})")
        self.keep = os.environ.get("FFV1_OUT_KEEP", "1") != "0"
        self.slices = int(os.environ.get("FFV1_OUT_SLICES", "16"))
        self.dump = os.environ.get("FFV1_OUT_DUMP")
        self.dump_frames = {int(x) for x in os.environ.get("FFV1_OUT_DUMP_FRAMES", "0,1,2").split(",") if x.strip()}
        self.patched = False
        self.file = None          # per input file: dict(input, out_paths, drop, fps, ...)
        self.writers = []
        self.files_done = 0
        self.t_ffv1 = 0.0         # conversion + pipe writes + final flush
        self.t_cli = 0.0          # the CLI's own save_frames_* calls and writer release
        self.frames_in = 0        # frames received (before the drop)


S = State()


def out_paths(input_path, cli_output):
    stem = Path(input_path).stem
    explicit = os.environ.get("FFV1_OUT_PATH")
    if explicit and (Path(explicit).is_dir() or explicit.endswith(os.sep) or not Path(explicit).suffix):
        base = Path(explicit) / stem
    elif explicit:
        base = Path(explicit).with_suffix("")
    else:
        cli = Path(cli_output)
        run = os.environ.get("BENCH_RUN_NAME")
        parent = cli.parent
        if run:  # bench.py: one file per run (a directory input adds the stem after the first file)
            base = parent / (run if not S.files_done else f"{run}.{stem}")
        else:
            base = cli.with_suffix("") if cli.suffix else cli
    if len(S.pix_fmts) == 1:
        return [(S.pix_fmts[0], base.with_name(base.name + ".mkv"))]
    return [(f, base.with_name(f"{base.name}.{f}.mkv")) for f in S.pix_fmts]


def begin_file(input_path, args, device_list, cli_output):
    drop_env = os.environ.get("FFV1_OUT_DROP")
    prepend = getattr(args, "prepend_frames", 0) or 0
    if drop_env is not None:
        drop, why = int(drop_env), "FFV1_OUT_DROP"
    elif prepend and len(device_list) <= 1:
        drop, why = prepend, f"--prepend_frames {prepend} on one GPU: the CLI keeps them (bug 05)"
    else:
        drop, why = 0, ("--prepend_frames removed by the CLI (multi-GPU)" if prepend else "")
    fps_env = os.environ.get("FFV1_OUT_FPS")
    fps = Fraction(fps_env).limit_denominator(1000000) if fps_env else probe_fps(input_path)
    S.file = {"input": input_path, "cli_output": cli_output, "drop": drop, "drop_why": why, "fps": fps,
              "received": 0, "written": 0, "paths": None}
    msg = f"input {Path(input_path).name}: dropping the first {drop} output frames ({why})" if drop else \
        f"input {Path(input_path).name}: no frames dropped" + (f" ({why})" if why else "")
    log(msg)


def handle(frames_tensor, cli_fps):
    f = S.file
    if f is None:
        return
    import torch
    t0 = time.perf_counter()
    if frames_tensor.dtype != torch.float32:
        frames_tensor = frames_tensor.float()
    arr = frames_tensor.cpu().numpy()
    T, H, W, C = arr.shape
    if not S.writers:
        if C == 4:
            log("RGBA frames: the alpha channel is not written")
        fps = f["fps"]
        if fps is None:
            fps = Fraction(cli_fps).limit_denominator(1001)
            log(f"ffprobe gave no frame rate, using the CLI's {cli_fps} as {fps}")
        f["fps"] = fps
        f["paths"] = out_paths(f["input"], f["cli_output"])
        for fmt, path in f["paths"]:
            S.writers.append(Writer(path, fmt, W, H, fps, S.slices))
            log(f"writing {path} (ffv1 {fmt}, {W}x{H}, {fps} fps)")
        if S.dump:
            os.makedirs(S.dump, exist_ok=True)
    for i in range(T):
        idx = f["received"] - f["drop"]
        f["received"] += 1
        S.frames_in += 1
        if idx < 0:
            continue
        frame = arr[i]
        if S.dump and idx in S.dump_frames:
            np.save(os.path.join(S.dump, f"frame_{idx:06d}.npy"), np.ascontiguousarray(frame[..., :3]))
        planes = {}
        for w in S.writers:
            if w.bits not in planes:
                planes[w.bits] = to_planar(frame, w.bits)
            w.write(planes[w.bits])
        f["written"] += 1
    S.t_ffv1 += time.perf_counter() - t0


def end_file():
    f = S.file
    if f is None:
        return
    t0 = time.perf_counter()
    rcs = [w.close() for w in S.writers]
    S.t_ffv1 += time.perf_counter() - t0
    S.writers = []
    S.file = None
    S.files_done += 1
    if not f["paths"]:
        log(f"input {Path(f['input']).name}: no video frames received (images are not handled)")
        return
    for (fmt, path), rc in zip(f["paths"], rcs):
        size = os.path.getsize(path) if os.path.exists(path) else 0
        n = f["written"]
        log(f"{'FAILED ' if rc else ''}{path}: {n} frames ({f['received']} received, {f['drop']} dropped), "
            f"{size / 2**20:.1f} MiB, {size / max(n, 1) / 2**20:.2f} MiB/frame, {fmt}, {f['fps']} fps")
    n = max(f["written"], 1)
    log(f"time: ffv1 {S.t_ffv1:.2f} s ({S.t_ffv1 / n * 1000:.0f} ms/frame), "
        f"CLI's own save {S.t_cli:.2f} s ({S.t_cli / max(f['received'], 1) * 1000:.0f} ms/frame"
        f"{', skipped: FFV1_OUT_KEEP=0' if not S.keep else ''})")
    S.t_ffv1 = S.t_cli = 0.0


def patch_cli(g):
    """g: the globals of the running inference_cli.py."""
    orig_psf = g["process_single_file"]
    orig_video = g["save_frames_to_video"]
    orig_image = g["save_frames_to_image"]

    def process_single_file(input_path, args, device_list, output_path=None, *a, **k):
        input_type = g["get_input_type"](input_path)
        if input_type == "video":
            out = output_path
            # same resolution of the output path as process_single_file
            if out is None:
                out = g["generate_output_path"](input_path, args.output_format, input_type=input_type)
            elif not Path(out).suffix or (args.output_format == "png" and input_type != "image"):
                out = g["generate_output_path"](input_path, args.output_format, output_dir=out,
                                                input_type=input_type)
            begin_file(input_path, args, device_list, out)
        else:
            log(f"{Path(input_path).name}: {input_type} input, no FFV1 output")
        try:
            return orig_psf(input_path, args, device_list, output_path, *a, **k)
        finally:
            end_file()

    def save_frames_to_video(frames_tensor, output_path, fps=30.0, writer=None, *a, **k):
        handle(frames_tensor, fps)
        if not S.keep:
            return None
        t0 = time.perf_counter()
        try:
            return orig_video(frames_tensor, output_path, fps, writer, *a, **k)
        finally:
            S.t_cli += time.perf_counter() - t0

    def save_frames_to_image(frames_tensor, output_dir, base_name, start_index=0):
        handle(frames_tensor, None)
        if not S.keep:
            return frames_tensor.shape[0]
        t0 = time.perf_counter()
        try:
            return orig_image(frames_tensor, output_dir, base_name, start_index)
        finally:
            S.t_cli += time.perf_counter() - t0

    cls = g.get("FFMPEGVideoWriter")
    if cls is not None:
        orig_release = cls.release

        def release(self):
            t0 = time.perf_counter()
            try:
                return orig_release(self)
            finally:
                S.t_cli += time.perf_counter() - t0
        cls.release = release

    g["process_single_file"] = process_single_file
    g["save_frames_to_video"] = save_frames_to_video
    g["save_frames_to_image"] = save_frames_to_image
    S.patched = True
    log(f"patched the CLI's savers: {', '.join(S.pix_fmts)}, "
        f"{'CLI output kept' if S.keep else 'CLI output skipped (FFV1_OUT_KEEP=0)'}")


def install():
    """Patch the CLI's module globals when it parses its arguments: by then every function of
    inference_cli.py is defined, and main() looks them up at call time. Works whatever the
    position of this wrapper in a chain of wrappers."""
    import argparse
    orig = argparse.ArgumentParser.parse_args

    def parse_args(self, *a, **k):
        if not S.patched:
            fr = sys._getframe(1)
            while fr is not None:
                g = fr.f_globals
                if "save_frames_to_video" in g and "process_single_file" in g:
                    patch_cli(g)
                    break
                fr = fr.f_back
        return orig(self, *a, **k)

    argparse.ArgumentParser.parse_args = parse_args


def at_exit():
    if S.file is not None:  # the CLI died mid-file: close what was written
        end_file()
    if not S.patched:
        log("the CLI's savers were never found: no FFV1 output")


# ---------------------------------------------------------------- checks (no GPU)

def probe(path):
    return json.loads(subprocess.run(
        [ffprobe_bin(), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,pix_fmt",
         "-of", "json", path], capture_output=True, text=True, check=True).stdout)["streams"][0]


def decode(path):
    """Yield the frames of an RGB master as uint16 [H, W, 3] RGB, in its own bit depth (no
    conversion: gbrp10le/gbrp16le are decoded as stored). Returns (bits, pix_fmt, generator)."""
    info = probe(path)
    fmt = info["pix_fmt"]
    if fmt not in ("gbrp16le", "gbrp10le"):
        raise SystemExit(f"{path}: {fmt}, only the RGB (gbrp) masters can be compared bit-exactly")
    W, H = info["width"], info["height"]
    size = 3 * H * W * 2

    def frames():
        proc = subprocess.Popen([ffmpeg_bin(), "-v", "error", "-i", path, "-map", "0:v:0", "-f", "rawvideo",
                                 "-pix_fmt", fmt, "-"], stdout=subprocess.PIPE)
        try:
            while True:
                buf = proc.stdout.read(size)
                if len(buf) < size:
                    break
                planes = np.frombuffer(buf, dtype="<u2").reshape(3, H, W)
                yield planes[[2, 0, 1]].transpose(1, 2, 0)  # G, B, R -> R, G, B
        finally:
            proc.stdout.close()
            proc.wait()
    return (10 if fmt == "gbrp10le" else 16), fmt, frames()


def cmd_verify(mkv, dump_dir):
    bits, fmt, gen = decode(mkv)
    scale = (1 << bits) - 1
    frames = list(gen)
    files = sorted(Path(dump_dir).glob("frame_*.npy"))
    if not files:
        raise SystemExit(f"no frame_*.npy in {dump_dir}")
    print(f"{mkv}: {fmt}, {len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]}")
    for p in files:
        idx = int(p.stem.split("_")[1])
        x = np.load(p).astype(np.float64)
        want = np.rint(np.clip(x, 0, 1) * scale).astype(np.int64)
        got = frames[idx].astype(np.int64)
        d = np.abs(got - want)
        err = np.abs(got / scale - x)
        print(f"  frame {idx}: {'bit-exact' if not d.any() else 'DIFFERS'} vs round(x*{scale}) "
              f"(max {d.max()}, {np.count_nonzero(d)} samples); |decoded - x| max {err.max() * 255:.4f}, "
              f"mean {err.mean() * 255:.4f} (8-bit levels)")


def cmd_diff(a, b):
    bits_a, fmt_a, ga = decode(a)
    bits_b, fmt_b, gb = decode(b)
    sa, sb = (1 << bits_a) - 1, (1 << bits_b) - 1
    mses, maxd, ndiff, tot, n = [], 0.0, 0, 0, 0
    for fa, fb in zip(ga, gb):
        d = fa / sa - fb / sb
        mses.append(float(np.mean(d * d)))
        maxd = max(maxd, float(np.abs(d).max()))
        ndiff += int(np.count_nonzero(d))
        tot += d.size
        n += 1
    rest_a, rest_b = sum(1 for _ in ga), sum(1 for _ in gb)
    if rest_a or rest_b:
        print(f"frame counts differ: {n + rest_a} vs {n + rest_b}, compared the first {n}")

    def psnr(m):
        return float("inf") if m == 0 else 10 * np.log10(1.0 / m)
    worst = int(np.argmax(mses))
    print(f"{a} ({fmt_a}) vs {b} ({fmt_b}): {n} frames")
    print(f"  PSNR (RGB, all frames) {psnr(np.mean(mses)):.2f} dB; worst frame {worst} {psnr(mses[worst]):.2f} dB, "
          f"best {psnr(min(mses)):.2f} dB")
    print(f"  max |difference| {maxd * 255:.2f} 8-bit levels; samples that differ {ndiff / tot:.2%}")


def cmd_selftest():
    rng = np.random.default_rng(0)
    H, W, T = 64, 96, 3
    # bf16-like values: 8 significant bits, plus a fine gradient, clipped to [0, 1]
    x = rng.random((T, H, W, 3), dtype=np.float32)
    x = (x.view(np.uint32) & 0xFFFF0000).view(np.float32)
    x[0, :, :, 0] = np.linspace(0, 1, W, dtype=np.float32)
    ok = True
    with tempfile.TemporaryDirectory() as d:
        for fmt in FORMATS:
            path = os.path.join(d, f"t.{fmt}.mkv")
            w = Writer(path, fmt, W, H, Fraction(24000, 1001), 4)
            for f in x:
                w.write(to_planar(f, w.bits))
            rc = w.close()
            tags = subprocess.run([ffprobe_bin(), "-v", "error", "-select_streams", "v:0", "-count_frames",
                                   "-show_entries", "stream=codec_name,pix_fmt,color_space,color_range,"
                                   "color_primaries,color_transfer,r_frame_rate,nb_read_frames",
                                   "-of", "compact=p=0", path], capture_output=True, text=True).stdout.strip()
            line = f"{fmt}: rc {rc}, {tags}"
            if fmt.startswith("gbrp") and rc == 0:
                bits, _, gen = decode(path)
                got = np.stack(list(gen))
                want = np.rint(x.astype(np.float64) * ((1 << bits) - 1))
                exact = got.shape == want.shape and np.array_equal(got, want)
                ok &= exact
                line += f"; round trip {'bit-exact' if exact else 'DIFFERS'}"
            ok &= rc == 0
            print(line)
    print("selftest", "ok" if ok else "FAILED")
    sys.exit(0 if ok else 1)


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        sys.exit(__doc__)
    if sys.argv[1] == "--selftest":
        return cmd_selftest()
    if sys.argv[1] == "--verify" and len(sys.argv) == 4:
        return cmd_verify(sys.argv[2], sys.argv[3])
    if sys.argv[1] == "--diff" and len(sys.argv) == 4:
        return cmd_diff(sys.argv[2], sys.argv[3])
    if sys.argv[1].startswith("--"):
        sys.exit(__doc__)
    script = os.path.abspath(sys.argv[1])
    sys.argv = [script, *sys.argv[2:]]
    sys.path[0] = os.path.dirname(script)
    install()
    atexit.register(at_exit)
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()

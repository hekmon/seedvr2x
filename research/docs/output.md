# Lossless output: an FFV1 master from the float frames

> Status: **implemented and measured** ([`scripts/ffv1_out.py`](../scripts/ffv1_out.py), SeedVR2
> `4490bd1`, 7B fp16, 1080p). The master is bit-exact to `round(x × 65535)` of the frames the CLI
> saves. Measured with it: the four attention backends change the output by 55–56 dB PSNR
> (`sageattn_3` = `sageattn_2` bit for bit, `flash_attn_2` is deterministic), which VMAF does not
> see; through the CLI's own mp4 the same comparison mostly measures x264.

## Why

Comparing two configurations through the CLI's output mixes the output path's error into the
difference:

- **mp4** is lossy: x264 (or x265 with `--10bit`) at CRF 12, 8-bit 4:2:0 with an untagged BT.601
  matrix ([10](../bugs/10-ffmpeg-untagged-bt601.md)), or OpenCV's `mp4v`. Two encodings of nearly
  identical frames differ by the encoder's noise.
- **PNG** is lossless but 8-bit, and truncated instead of rounded
  ([09](../bugs/09-uint8-truncation.md)). The frames are a bfloat16 tensor, whose grid is finer
  than 8 bits below 0.5 ([19](../bugs/19-10bit-output-is-8-bit.md)): dark content loses levels.
- On one GPU, `--prepend_frames` frames stay in both ([05](../bugs/05-prepend-frames-not-removed.md)),
  which shifts every frame.

## Usage

`ffv1_out.py` is a `--wrap` script like the probes: it runs `inference_cli.py` in-process and
patches it without touching the checkout.

```bash
# with bench.py: the master is <output dir>/<run name>.mkv
python3 scripts/bench.py run fa2 --wrap scripts/ffv1_out.py -- input.mkv --output out/fa2 \
  --output_format mp4 --video_backend ffmpeg --model_dir /path/to/models ...
# several formats in one run, and chained with another wrapper
python3 scripts/bench.py run fa2 --wrap scripts/attn_probe.py --wrap scripts/ffv1_out.py \
  --env FFV1_OUT_PIXFMT=gbrp16le,yuv420p10le -- input.mkv ...
# without bench.py: the master is the CLI's output path with .mkv
python ffv1_out.py inference_cli.py input.mkv --output out/ ...

python ffv1_out.py --diff a.mkv b.mkv           # PSNR, max difference, share of samples that differ
python ffv1_out.py --verify out.mkv dump_dir/   # a master against frames dumped with FFV1_OUT_DUMP
python ffv1_out.py --selftest                   # synthetic frames through every format (no GPU)
```

| Variable | Default | Meaning |
|---|---|---|
| `FFV1_OUT_PIXFMT` | `gbrp16le` | `gbrp16le`, `gbrp10le`, `yuv444p10le` or `yuv420p10le`. A comma list writes one file per format, `<name>.<pix_fmt>.mkv` |
| `FFV1_OUT_PATH` | next to the CLI's output | An `.mkv` path (one input), or a directory for `<input stem>.mkv`. Default: the CLI's mp4 path with `.mkv`, the PNG directory + `.mkv`, or `<output dir>/<run name>.mkv` under bench.py |
| `FFV1_OUT_KEEP` | `1` | `0` skips the CLI's own output (its save functions are not called) |
| `FFV1_OUT_DROP` | `--prepend_frames` on one GPU, else 0 | Frames dropped at the start of the master. The log says how many and why |
| `FFV1_OUT_FPS` | the input's `r_frame_rate` | Frame rate override (`24000/1001` or a decimal) |
| `FFV1_OUT_SLICES` | 16 | FFV1 slices |
| `FFV1_OUT_FFMPEG`, `FFV1_OUT_FFPROBE` | from `PATH` | Binaries. `zscale` is needed for the YUV formats |
| `FFV1_OUT_DUMP`, `FFV1_OUT_DUMP_FRAMES` | off, `0,1,2` | Directory where the listed master frames are also saved as float32 `.npy`, exactly as received |

Which `--output_format` to give the CLI: either works. `mp4` with `--video_backend ffmpeg` costs
the least; `png` when PNG-based scripts ([`quality_metrics.py`](../scripts/quality_metrics.py),
[`frame_diff.py`](../scripts/frame_diff.py)) are needed as well. The CLI's output is kept by
default, so its log, its output path and bench.py's parsing stay as they are; set
`FFV1_OUT_KEEP=0` when only the master matters. Skipping it is safe: the CLI only releases a video
writer it got back, and a PNG save returns the frame count it expects. Its "Output saved to" line
then names a file that doesn't exist. Single images are not handled (one frame: use the PNG).

## How it works, and the format choices

- **Where the frames are taken.** `save_frames_to_video` and `save_frames_to_image` receive the
  final frames, `[T, H, W, C]` float32 on the CPU in [0, 1]: Phase 4 clamps them after colour
  correction (`src/core/generation_phases.py:1348`), and the core converts the bfloat16
  `final_video` to float32 (`inference_cli.py:1004-1010`). The wrapper
  replaces both functions in the script's globals (and `process_single_file`, to open and close one
  master per input file), so every chunk of `--chunk_size` streaming and the multi-GPU result go
  through it. It patches when the CLI parses its arguments: by then every function of the script
  is defined, and `main()` looks them up at call time. That works at any position in a chain of
  wrappers (checked: `--wrap ffv1_out.py --wrap attn_probe.py` with `FFV1_OUT_KEEP=0` gave the
  master, the probe's JSON and no mp4).
- **Quantisation: rounded, from the float tensor.** `round(x × 65535)` for 16 bits,
  `round(x × 1023)` for 10 bits (NumPy `rint`, ties to even), instead of the CLI's
  `floor(x × 255)`.
- **RGB, not 4:2:0.** The model's output is RGB. Storing it as planar RGB (`gbrp16le`, FFV1 level
  3) keeps it exactly; any YUV conversion is a choice (matrix, range, chroma siting) that belongs
  to whoever encodes the final file. The frames are piped as `gbrp16le` / `gbrp10le` already, so
  ffmpeg converts nothing: bit-exact by construction, not by trusting swscale.
- **FFV1 options:** `-c:v ffv1 -level 3 -g 1 -slices 16 -slicecrc 1`: intra only (every frame can
  be cut or seeked to), slice CRCs to detect corruption, slices for threading.
- **Colour tags.** RGB: matrix GBR (Matroska `MatrixCoefficients` 0), BT.709 primaries and
  transfer, full range. Output options alone left primaries and transfer unwritten with this
  ffmpeg (n9.0.2): the frames get them from a `setparams` filter, which changes no pixel. YUV
  variants: converted with `zscale` (RGB full range → BT.709 matrix, limited range, same primaries
  and transfer on both sides, no dither; left chroma siting for 4:2:0) and tagged BT.709 / tv,
  so nothing falls back to swscale's BT.601 default ([10](../bugs/10-ffmpeg-untagged-bt601.md)).
- **Frame rate:** the input's exact `r_frame_rate` from `ffprobe` (`24000/1001`), given as the
  rawvideo input's `-framerate`: the frames come from a pipe, so there are no timestamps to pass
  through. The CLI's own writer uses OpenCV's float (`23.976023…`).
- **Prepended frames:** by default the first `--prepend_frames` frames are dropped on one GPU,
  where the CLI keeps them, and nothing on several GPUs, where it removes them.

Limits: the tags describe the output RGB as BT.709, which is right for HD sources. The input
side is the CLI's: OpenCV decodes the source to 8-bit RGB with its own matrix choice and drops
its colour metadata ([cli-flags.md](cli-flags.md#input-handling)). Alpha is not written.

## Validation

1080p, 7B fp16, batch 5, `--color_correction lab`, `flash_attn_2`, `--skip_first_frames 48`
(anime, 1920×1080, 24000/1001).

**Bit-exactness** (run `ffv1-lossless`: `--load_cap 10 --prepend_frames 2 --output_format png`,
three formats in one run, frames 0, 4 and 9 dumped as float32):

| Master | Decoded vs `round(x × scale)` | \|decoded − x\|, 8-bit levels: max / mean |
|---|---|---|
| `gbrp16le` | bit-exact (3 frames) | 0.0019 / 0.0004–0.0008 |
| `gbrp10le` | bit-exact (3 frames) | 0.125 / 0.03–0.05 |

The CLI's PNGs of the same frames (`_000002`, `_000006`, `_000011`: the CLI kept the 2 prepended
frames, the master dropped them and has 10 frames) are exactly `floor(x × 255)`, i.e.
[09](../bugs/09-uint8-truncation.md) on real frames:

| Frame | PNG − round(x × 255): mean, samples one level low | PNG − x × 255: mean | Samples < 0.5 | Distinct values (R): float / 16-bit / PNG |
|---|---|---|---|---|
| 0 | −0.556, 55.6% | −0.667 | 99.9% | 263 / 263 / 134 |
| 4 | −0.555, 55.5% | −0.629 | 97.6% | 332 / 332 / 203 |
| 9 | −0.532, 53.2% | −0.583 | 91.5% | 347 / 347 / 218 |

On these dark frames the bf16 output has 1.6–2× more levels per channel than 8 bits can hold;
the 16-bit master keeps all of them.

**ffprobe** (`-count_frames`):

| File | codec, pix_fmt | color_space / range / primaries / transfer | Chroma | Frame rate | Frames |
|---|---|---|---|---|---|
| `.gbrp16le.mkv` | ffv1, gbrp16le | gbr / pc / bt709 / bt709 | – | 24000/1001 | 10 |
| `.gbrp10le.mkv` | ffv1, gbrp10le | gbr / pc / bt709 / bt709 | – | 24000/1001 | 10 |
| `.yuv420p10le.mkv` | ffv1, yuv420p10le | bt709 / tv / bt709 / bt709 | left | 24000/1001 | 10 |
| CLI mp4 | h264, yuv420p | unknown | – | 24000/1001 | 21 (other run) |

`mkvinfo` shows the same in the Matroska `Colour` element (matrix 0 or 1, range 2 or 1, primaries
1, transfer 1). Frames: 10 processed + 2 prepended = 12 received, 2 dropped.

**Size and time per 1080p frame** (anime, 21-frame runs; times in the CLI process, encoder end
included):

| Output | Size per frame | Write time per frame |
|---|---|---|
| FFV1 `gbrp16le` | 2.97 MiB | 15–16 ms for `gbrp16le` + `yuv420p10le` together |
| FFV1 `gbrp10le` | 2.43 MiB (10-frame run, against 2.86 for 16-bit) | 28 ms for three formats (10 frames) |
| FFV1 `yuv420p10le` | 1.07 MiB | (with `gbrp16le`, above) |
| CLI mp4, x264 CRF 12 | 0.11 MiB | 21–24 ms (save + writer release) |
| CLI PNG, 8-bit | – | 44 ms (12 frames) |

Against ≈ 4 s of processing per frame, either is noise. A 16-bit master is 27× the CLI's mp4:
about 70 MiB per second of 1080p video. That is numz's output with `lab`, whose values sit on
bf16's steps ([numerics.md](numerics.md#where-the-precision-goes)). seedvr2x's `lab` keeps a
float32 output: its `gbrp16le` master takes 2.0–2.5× that (the implementation's milestone 5
masters), while its `yuv420p10le` master stays the same size.

## Do attention backends change the output?

The question that motivated the master. Same input, seed 42, 21 frames (`--load_cap 21`), 1080p,
batch 5, `lab`; runs `ffv1-attn-{fa2,sdpa,sa2,sa3,fa2-r2}`. PSNR on the 16-bit RGB masters
(`ffv1_out.py --diff`), VMAF with `sptenc vmaf` (VMAF v1 `vmaf_v1.0.16_3d0h`, "fidelity" =
without its CAMBI feature, then the banding CAMBI says the second file adds).

| Pair | PSNR (RGB, 16-bit) | Worst frame | Max \|difference\| | Samples that differ | VMAF fidelity | CAMBI added (mean / worst frame) |
|---|---|---|---|---|---|---|
| `flash_attn_2` vs its rerun | ∞ | ∞ | 0 | 0% | 100 | 0 |
| `sageattn_2` vs `sageattn_3` | ∞ | ∞ | 0 | 0% | – | – |
| `flash_attn_2` vs `sdpa` | 56.20 dB | 55.33 dB | 10.0 levels | 34% | 100 | +0.0002 / +0.002 |
| `flash_attn_2` vs `sageattn_2` | 55.08 dB | 54.21 dB | 14.4 levels | 40% | 100 | +0.0002 / +0.003 |
| `sdpa` vs `sageattn_2` | 55.02 dB | 54.10 dB | 12.5 levels | 41% | – | – |

- **`flash_attn_2` is deterministic** (two runs, bit-identical masters), so any difference below
  is the backend's.
- **`sageattn_3` gives exactly `sageattn_2`'s output**: SA3 never runs, every call falls back to
  SA2 ([attention.md](attention.md#consequences-for---attention_mode)).
- **The backends do change the output, slightly.** 55–56 dB on average; a third of the samples
  move; the largest differences reach 10–14 levels. `sdpa` and
  `flash_attn_2` are the closest pair; SageAttention's INT8 QK is a little further from both.
- **VMAF can't tell them apart**: fidelity 100 at every percentile, and no added banding. The
  yuv420p10le BT.709 variants give the same scores as the RGB masters.

Through the CLI's mp4s, the same comparison measures x264 more than attention:

| Pair | PSNR (RGB, 8-bit decode) | VMAF fidelity (min / harmonic mean) | CAMBI added (mean / worst) |
|---|---|---|---|
| `flash_attn_2` master vs its own CLI mp4 | 44.62 dB (worst 43.59) | 100 / 100 | **+1.19 / +3.62** |
| `flash_attn_2` mp4 vs `sdpa` mp4 | 47.31 dB | 100 / 100 | +0.03 / +0.09 |
| `flash_attn_2` mp4 vs `sageattn_2` mp4 | 47.14 dB | 99.82 / 99.99 | +0.02 / +0.06 |

- The mp4 is 44.6 dB from the frames it encodes (decoded with the BT.601 matrix it was encoded
  with), against 55–56 dB between backends: the encoder's error is ≈ 10× (in MSE) the backends'.
  Between two mp4s, PSNR falls to 47 dB and the backend difference is hidden in x264's noise.
- **The CLI's mp4 adds banding**: CAMBI rates it 1.22 on average against 0.04 for the master
  (converted by ffmpeg to the mp4's 8-bit 4:2:0 for the comparison), up to +3.6 on one frame
  (CAMBI: ≈ 5 is "slightly annoying"). The CLI truncates to 8-bit RGB, then x264 quantises again
  at CRF 12, on dark gradients. An encode made from the master starts without the first step.

`sptenc vmaf` takes the RGB masters as they are: libvmaf only accepts YUV, so ffmpeg
auto-inserts a conversion to `yuv444p10le` on both inputs, with swscale's default matrix (BT.601;
the log shows `csp:unknown`). Both sides get the same conversion, and the fidelity scores matched
the BT.709 `yuv420p10le` variants (100). For a reference against an encode, write the YUV variant
(`FFV1_OUT_PIXFMT=gbrp16le,yuv420p10le`) or encode from the master with explicit tags, so that
both inputs carry the same matrix.

## Reproduce

```bash
# bit-exactness: three formats, float dumps, PNG for the bug-09 comparison
python3 scripts/bench.py run ffv1-lossless --wrap scripts/ffv1_out.py \
  --env FFV1_OUT_PIXFMT=gbrp16le,gbrp10le,yuv420p10le --env FFV1_OUT_DUMP=out/ffv1-lossless/npy \
  --env FFV1_OUT_DUMP_FRAMES=0,4,9 -- input.mkv --output out/ffv1-lossless --output_format png \
  --video_backend ffmpeg --model_dir /path/to/models --dit_model seedvr2_ema_7b_fp16.safetensors \
  --resolution 1080 --batch_size 5 --skip_first_frames 48 --color_correction lab \
  --load_cap 10 --prepend_frames 2 --attention_mode flash_attn_2
python scripts/ffv1_out.py --verify out/ffv1-lossless/ffv1-lossless.gbrp16le.mkv out/ffv1-lossless/npy

# backends: one run per --attention_mode, then
python scripts/ffv1_out.py --diff out/fa2/fa2.gbrp16le.mkv out/sdpa/sdpa.gbrp16le.mkv
sptenc vmaf out/fa2/fa2.gbrp16le.mkv out/sdpa/sdpa.gbrp16le.mkv
```

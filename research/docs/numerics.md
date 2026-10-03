# Numerics and input preparation, against a ground truth

> Status: **measured** with [`scripts/numerics_patch.py`](../scripts/numerics_patch.py) (one switch
> per numerics or input-preparation choice, patched in at import; with no switch set, the output
> is bit-identical to numz's), [`scripts/ffv1_out.py`](../scripts/ffv1_out.py) (16-bit RGB
> masters), [`scripts/fr_clips.py`](../scripts/fr_clips.py) (ground-truth clips and degraded
> inputs) and [`scripts/fr_metrics.py`](../scripts/fr_metrics.py) (full-reference metrics and
> paired statistics), for the "Numerics" question in
> [DESIGN.md](../../seedvr2x/DESIGN.md#to-measure). SeedVR2 `4490bd1`, 7B fp16, `flash_attn_2`,
> one batch of 45 frames, 1080p (one 720p test), `--color_correction none`, with `lab` rendered
> from the same run. Four animated clips; live action is not measured yet.

In short (4 animated clips of 45 frames, a ×2 upscale of a mildly degraded input to 1080p; every
variant paired with numz's default at the same seed, and judged against the spread of 3 seeds):

- **SeedVR2 re-renders the picture: a plain bicubic upscale is closer to the source on every
  fidelity metric.** numz's default lands at PSNR-Y 27.1–32.1 dB, VMAF 51–65 and a
  low-frequency ΔE00 of 1.8–2.6, where a Catmull-Rom upscale of the same input gets 32.2–43.2 dB,
  83–92 and 0.6–0.7; LPIPS is worse on 3 of 4 clips. Given the ground truth itself as input, the
  model moves it as far (25.4–30.4 dB, ΔE00 2.4–3.5). Registration is not the cause (shifts
  ≤ 0.12 px, every frame matches its own): the model redraws texture, and its mean brightness is
  off by −0.7 to +6 levels, changing from frame to frame (−6.9 to +7.6 on the ground truth
  itself). These metrics compare variants, paired and against the seed spread; they don't rate
  the model.
- **The seed is the noise floor:** 3 seeds of the default spread by 0.09–0.47 dB PSNR-Y,
  0.0002–0.018 LPIPS and 0.3–3.3 VMAF, depending on the clip.
- **ByteDance's numerics change nothing measurable:** float32 RoPE angles, VAE posterior
  sampling, a float32 input chain, the float32 weights, and all of them together stay within the
  seed spread on all 4 clips for PSNR, SSIM, LPIPS, DISTS, VMAF and temporal error
  (|ΔPSNR-Y| ≤ 0.14 dB, |ΔVMAF| ≤ 0.8), with or without `lab`. So does float16 attention, the
  path of GPUs without bfloat16 (|ΔPSNR-Y| ≤ 0.06 dB).
- **The resize kernel is second order:** zimg spline36 stays within the spread everywhere,
  lanczos is better on the grainy clip only (+0.24 dB, LPIPS −0.005), and torchvision's bicubic
  without antialiasing adds a little low-frequency colour error and flicker on 4 of 4 clips.
- **numz's black padding is both a defect and an anchor.** At 1080p numz pads 8 black rows: the
  bottom 16 rows lose 3–10 dB against the rest of the frame. Padding without black (reflect,
  replicate, mid grey) repairs that band, but it, and no padding at all (a crop to 1072 rows),
  make the whole frame worse on every clip without black areas of its own: −0.6 to −1.1 dB
  PSNR-Y, −5.5 to −16 VMAF, +0.8 to +1.3 temporal error, far beyond the seed spread. Without black
  rows the model's frame-to-frame brightness drift grows (its sd by up to 41%).
- **Recommended: always reflect at least 8 rows, then 16 black rows**
  (`NUM_PAD=reflect>=8+black+16`). At 1080p this is exactly the measured `reflect+black+16`: the
  bottom band gains 7–13 dB and the frame stays as good or gets better (PSNR-Y +0.29 to +0.40 dB
  on 3 of 4 clips, the letterboxed one included). At 720p, where numz pads nothing, 16 black rows
  still steady the model (VMAF +1.8 to +6.4, the rest of the frame +0.2 to +0.6 dB, 4 of 4 clips)
  but, right under the picture, damage its bottom band again; with 16 reflected rows in between
  (the same mode at 720p) the test is pending. A stronger degradation (area downscale, CRF 26,
  2 clips) gives the same picture, the resize kernels gaining a little more.

## Why it matters for seedvr2x

seedvr2x vendors the model code and rewrites the runtime around it
([DESIGN.md](../../seedvr2x/DESIGN.md#vendored-model-code)): every numerics choice and the whole
input preparation become ours. numz and ByteDance's reference differ in RoPE precision, the
casts of the input frames, the VAE's posterior mode against a sample, the weights' dtype, the
resize kernel, and padding (numz) against cropping (ByteDance) to multiples of 16. DESIGN.md
asks which of these matter. A difference only counts here if it moves the output closer to the
source than a change of seed does.

## The full-reference protocol

### Clips

Single shots of 45 frames from 1080p 8-bit 4:2:0 sources (BT.709, limited range, 23.976 fps),
extracted by decoding from the start and counting frames. Statistics of the ground truth (Y on
the 8-bit scale):

| Clip | Content | Mean Y | Near black (Y < 10) | Motion (mean \|ΔY\| per frame) | Detail (Laplacian variance) |
|---|---|---|---|---|---|
| anime-clean | clean digital anime | 115 | 0.0% | 12.3 | 37 |
| anime-grain | grainy anime, letterboxed (23 black rows at the top, 21 at the bottom) | 76 | 8.9% | 10.5 | 59 |
| anime-dark | dark anime | 53 | 6.8% | 12.3 | 12 |
| cartoon-bright | bright cartoon | 163 | 0.1% | 17.1 | 277 |

### Ground truth, inputs and runs

- **Ground truth (GT):** the source frames as decoded, converted to 16-bit RGB (zscale, BT.709
  matrix, limited to full range), FFV1.
- **d1, the main degradation:** ×1/2 in YUV with zimg bicubic b = c = 1/3 (Mitchell), x264
  `-preset slow -crf 20`, decoded to 8-bit RGB stored as FFV1, which the CLI's reader (cv2) gets
  bit-exactly. **d2** (anime-clean, anime-grain): an area downscale and CRF 26.
- **Bicubic baseline:** the d1 input upscaled ×2 with zimg Catmull-Rom (b = 0, c = 1/2) to 16-bit
  RGB.
- **Crop control:** rows 4–1075 of the GT (1920×1072) and rows 2–537 of the input (960×536), so
  that numz pads nothing.
- **Runs:** the CLI at `--resolution 1080` (crop: 1072), `--batch_size 45` (one batch per clip),
  `--color_correction none`, wrapped by `numerics_patch.py` (the variant, and `NUM_CC_EXTRA=lab`:
  Phase 4 also runs `lab` on a copy of the same frames) and `ffv1_out.py` (16-bit RGB masters).
  The default runs at seeds 42, 43 and 1234, every variant at 42 (`reflect` also at 43 and 1234
  on two clips).

### Metrics and verdicts

`fr_metrics.py` scores each output against the GT, frame by frame (per transition for the
temporal errors): PSNR-Y and SSIM-Y (BT.709 luma, 8-bit scale), LPIPS (AlexNet) and DISTS at full
resolution, VMAF as `sptenc vmaf` measures it (v1 1080p model, CAMBI clipped) and VMAF NEG, ΔE00
after a 4 px Gaussian blur (low-frequency colour and brightness), the temporal error |Δout − ΔGT|
between consecutive frames (and on 16×16 block means: low-frequency flicker), and PSNR-Y of the
bottom 16 rows and of the rest. A variant's figure is its paired difference to the default at
the same seed, frame by frame, with a 95% interval from a block bootstrap (blocks of 8 frames).
**Better / worse** means the interval excludes 0 *and* the difference exceeds the clip's seed
spread (max − min of the default's 3 per-seed means); anything else is **within**. Each
variant is scored twice, as rendered (`none`) and after `lab`.

### Validation

- **Determinism:** the default re-run with the final `numerics_patch.py` reproduced the first
  default run bit for bit (45 of 45 frames, `none` and `lab`). Where a variant's input is
  identical to the default's (reflect or replicate padding over the letterboxed clip's black
  rows), its output is bit-identical too.
- **Metrics on known cases** (`--make-test` on anime-clean): Gaussian noise σ = 2 levels per
  channel gives PSNR-Y 44.62 dB, the exact value for σ_Y = 2 × 0.749; the GT against itself gives
  infinite PSNR, LPIPS and DISTS 0, VMAF 100; a 1 px blur gives PSNR-Y 41.0, LPIPS 0.062,
  VMAF 93.1. DISTS from cached GT features equals the reference implementation (≤ 6e−8).
- **Inputs:** the crop GT is rows 4–1075 of the GT (framemd5). cv2 does *not* read the 16-bit GT
  as its 8-bit rounding (36–79% of values equal, +0.2 to +0.64 levels): the `ident` control below
  reads an 8-bit rounding of the GT stored like the d1 inputs, which cv2 reads bit-exactly.

## The model re-renders

**Registration.** Phase correlation on luma, per frame, against the GT (median over 45 frames),
the mean luma offset (output − GT, 8-bit levels, per-frame range) and the GT frame each output
frame matches best:

| Clip | Default: shift dx, dy (px) | Default: offset | Bicubic: shift, offset | `ident`: shift, offset |
|---|---|---|---|---|
| anime-clean | −0.10, +0.06 | +3.26 (+1.85..+5.11) | ≤ 0.01, +0.26 | −0.08, +0.07; +2.25 (+0.73..+4.14) |
| anime-grain | −0.05, −0.05 | +2.00 (+1.01..+2.78) | ≤ 0.01, +0.10 | −0.00, −0.04; −0.61 (−1.82..+0.52) |
| anime-dark | −0.07, −0.06 | +3.10 (−0.72..+5.96) | ≤ 0.01, +0.01 | −0.02, −0.04; −1.95 (−6.87..+1.36) |
| cartoon-bright | −0.07, −0.10 | +1.95 (−0.04..+3.60) | ≤ 0.01, +0.17 | −0.01, −0.12; +4.10 (+1.96..+7.63) |

Every output frame matches its own GT frame (45 of 45, never t ± 1). Shifting the default back by
its estimated shift gains nothing (anime-clean: 28.28 against 28.39 dB on frame 5). The offset
changes from frame to frame and changes sign between clips while the bicubic baseline, made from
the same input files, stays at +0.01 to +0.26: it is the model's, not the pipeline's. `lab` brings
it under 0.43 levels on average.

**Against the GT** (`none`; the default as the mean of its 3 seeds):

| Clip | | PSNR-Y | SSIM-Y | LPIPS | DISTS | VMAF | ΔE00 lf | Temporal error |
|---|---|---|---|---|---|---|---|---|
| anime-clean | default | 27.72 | 0.868 | 0.212 | 0.098 | 65.3 | 2.31 | 7.72 |
| | `ident` | 27.72 | 0.878 | 0.205 | 0.106 | 62.4 | 2.76 | 7.47 |
| | bicubic | 39.69 | 0.970 | 0.097 | 0.059 | 92.1 | 0.65 | 2.40 |
| anime-grain | default | 29.27 | 0.916 | 0.114 | 0.069 | 65.2 | 1.84 | 5.97 |
| | `ident` | 27.99 | 0.892 | 0.096 | 0.067 | 59.0 | 2.44 | 6.19 |
| | bicubic | 37.38 | 0.954 | 0.098 | 0.060 | 86.6 | 0.68 | 2.96 |
| anime-dark | default | 32.12 | 0.940 | 0.213 | 0.136 | 51.1 | 2.36 | 5.87 |
| | `ident` | 30.42 | 0.904 | 0.150 | 0.110 | 45.3 | 2.89 | 6.69 |
| | bicubic | 43.16 | 0.970 | 0.171 | 0.131 | 83.5 | 0.74 | 1.61 |
| cartoon-bright | default | 27.13 | 0.931 | 0.080 | 0.100 | 65.3 | 2.59 | 7.42 |
| | `ident` | 25.36 | 0.920 | 0.100 | 0.107 | 55.1 | 3.50 | 9.01 |
| | bicubic | 32.23 | 0.955 | 0.118 | 0.085 | 82.7 | 0.63 | 3.92 |

- **The model is further from the source than bicubic** on every clip and metric but LPIPS on
  cartoon-bright: 5.1 to 12.0 dB of PSNR-Y, 17 to 32 points of VMAF, ΔE00 2.7 to 4.1 times
  higher, temporal error 1.9 to 3.6 times. The windows where it departs most from the baseline show redrawn
  lines and texture (review crops `bicubic-<clip>-max-*`: mean |ΔY| 12–16 levels against the GT
  where the baseline has 1.4–6.5).
- **`ident`**, the GT itself (8-bit) at `--resolution 1080`, is as far or further: the degradation
  is not what limits fidelity, the model's rendering is. It is closer on LPIPS on 3 clips
  (anime-dark: 0.150 against 0.213) and further on colour on all 4.
- **`lab` brings the default closer to the GT:** PSNR-Y +0.9 to +4.0 dB (anime-dark 36.09),
  ΔE00 lf down 37–55% (1.07–1.58), VMAF +3.9 to +6.6, LPIPS slightly lower on all 4 clips.

## Seed bands

Spread (max − min) of the default's 3 per-seed means, `none` (`lab` within ±0.15 dB and ±0.4
VMAF of these):

| Clip | PSNR-Y | SSIM-Y | LPIPS | DISTS | VMAF | ΔE00 lf | Temporal error lf | Temporal error | Bottom 16 rows |
|---|---|---|---|---|---|---|---|---|---|
| anime-clean | 0.47 | 0.0124 | 0.0183 | 0.0099 | 3.30 | 0.026 | 0.042 | 0.563 | 0.56 |
| anime-grain | 0.19 | 0.0023 | 0.0034 | 0.0020 | 0.95 | 0.034 | 0.026 | 0.169 | – |
| anime-dark | 0.09 | 0.0012 | 0.0074 | 0.0049 | 2.18 | 0.001 | 0.050 | 0.088 | 1.18 |
| cartoon-bright | 0.25 | 0.0007 | 0.0002 | 0.0008 | 0.34 | 0.013 | 0.034 | 0.072 | 2.35 |

A seed moves the whole clip together (one batch): seed 1234 is +0.37 dB on 100% of
anime-clean's frames. A variant measured at one seed is compared with this spread for that
reason. anime-grain's bottom rows are black in the GT and nearly so in every output; a few
frames match exactly, so their mean PSNR is infinite and the spread undefined.

## Numerics

Paired difference to the default, `none`: the range over the 4 clips, and how many clips are
better (B) or worse (W); "–" = within everywhere.

| Variant | Switch | PSNR-Y | LPIPS | DISTS | VMAF | ΔE00 lf | Temporal error |
|---|---|---|---|---|---|---|---|
| `rope` | `NUM_ROPE=fp32` | −0.05..+0.03 (–) | −0.0004..+0.0016 (–) | −0.0002..+0.0010 (–) | −0.78..+0.06 (–) | +0.002..+0.016 (1 W) | −0.014..+0.044 (–) |
| `vaes` | `NUM_VAE=sample` | −0.04..+0.02 (–) | −0.0004..+0.0012 (–) | −0.0003..+0.0008 (–) | −0.64..+0.03 (–) | +0.000..+0.016 (–) | −0.009..+0.037 (–) |
| `prep` | `NUM_PREP=fp32` | −0.06..+0.02 (–) | −0.0001..+0.0030 (–) | −0.0007..+0.0014 (–) | −0.58..+0.19 (–) | −0.035..−0.015 (3 B) | −0.065..+0.101 (–) |
| `w32` | the fp32 `.pth` weights | −0.09..+0.00 (–) | −0.0009..+0.0015 (–) | −0.0006..+0.0006 (–) | −0.60..−0.08 (–) | +0.006..+0.050 (3 W) | −0.005..+0.081 (–) |
| `parity` | all of the above | −0.09..+0.03 (–) | −0.0003..+0.0038 (–) | −0.0005..+0.0018 (–) | −0.80..+0.16 (–) | −0.019..−0.000 (1 B) | −0.074..+0.131 (–) |
| `attn16` | `NUM_ATTN=fp16` | −0.03..+0.01 (–) | −0.0004..+0.0007 (–) | −0.0003..+0.0004 (–) | −0.44..+0.04 (–) | −0.004..+0.004 (1 B) | −0.006..+0.035 (–) |

- **Nothing reaches the seed spread on PSNR, SSIM, LPIPS, DISTS, VMAF or temporal error,** on any
  clip, `none` or `lab` (`lab`: |ΔPSNR-Y| ≤ 0.14 dB, |ΔVMAF| ≤ 0.72).
- The only verdicts are low-frequency colour shifts of 0.002–0.05 ΔE00 against values of 1.8–2.6,
  and low-frequency flicker 0.08 lower on anime-dark (`prep`, `parity`): they pass only because
  those spreads are tiny (ΔE00 0.001–0.034). After `lab` they vanish, but for DISTS changes of
  ±0.0004 on cartoon-bright, whose DISTS spread is 0.0003.
- `w32` loads ByteDance's float32 checkpoint, whose RoPE table is already float32, so it includes
  `rope`; both give the same verdicts. The float16 file numz ships costs nothing measurable.

## Resize kernels

The input (960×540) is upscaled ×2 to the output size before encoding. numz: torchvision bicubic
with `antialias=True` (a = −0.5).

| Variant | Kernel | PSNR-Y | LPIPS | VMAF | ΔE00 lf | Temporal error lf | Temporal error |
|---|---|---|---|---|---|---|---|
| `noaa` | torchvision bicubic, no antialias (a = −0.75) | −0.09..+0.12 (–) | −0.0052..+0.0005 (1 B) | −1.38..+0.10 (–) | +0.010..+0.088 (4 W) | +0.037..+0.149 (4 W) | −0.150..+0.161 (1 W) |
| `spline36` | zimg spline36 | +0.01..+0.16 (–) | −0.0040..−0.0001 (–) | −0.89..+0.77 (–) | −0.009..+0.010 (1 B) | −0.014..+0.051 (1 W) | −0.104..+0.061 (–) |
| `lanczos` | zimg lanczos (3 taps) | −0.02..+0.24 (1 B) | −0.0082..−0.0001 (1 B) | −0.93..+0.79 (–) | −0.004..+0.030 (1 B, 1 W) | −0.004..+0.108 (1 W) | −0.163..+0.110 (1 W) |

The kernels change the output by 0.6–1.7 levels (mean |ΔY|) in the windows where they differ
most from the default (review crops). The sharper zimg kernels lean slightly toward the GT on
detail (LPIPS lower on all 4 clips, DISTS on 3), significantly so only on anime-grain
(lanczos: +0.24 dB, LPIPS −0.005, DISTS −0.0023; `lab`: +0.30 dB). Not worth a CPU resize through
ffmpeg on its own; spline36 or lanczos are safe if the pipeline resizes with zimg anyway. On the
softer d2 input all three kernels gain a little more ([Degradation d2](#degradation-d2)).

## Padding

### numz's black band

numz pads the resized frames to multiples of 16, bottom and right, with zeros (black), and crops
the decoded output back. At 1080p that is 8 black rows (1088), no columns. The bottom 16 rows of
the default are 3.4–10.3 dB below the rest of its own frame (PSNR-Y 18.0 against 28.3 on
anime-clean, 28.8 against 32.2 on anime-dark, 19.1 against 27.6 on cartoon-bright), where the
bicubic baseline's bottom rows are 5.3 dB below to 7.7 dB above its rest: the black rows bleed
into the picture above them.

### Alternatives without black rows

| Variant | PSNR-Y | LPIPS | VMAF | ΔE00 lf | Temporal error | Bottom 16 rows | Rest of the frame |
|---|---|---|---|---|---|---|---|
| `reflect` | −0.80..−0.67 (3 W) | +0.0095..+0.0304 (3 W) | −14.45..−6.13 (3 W) | +0.135..+0.591 (3 W) | +0.898..+1.214 (3 W) | +6.45..+11.63 (3 B) | −1.35..−0.88 (3 W) |
| `replicate` | −0.89..−0.58 (3 W) | +0.0090..+0.0319 (3 W) | −14.69..−5.73 (3 W) | +0.132..+0.556 (3 W) | +0.884..+1.330 (3 W) | +6.65..+12.45 (3 B) | −1.44..−0.89 (3 W) |
| `grey` (0.5) | −0.78..−0.64 (3 W) | +0.0087..+0.0259 (3 W) | −9.99..−5.71 (3 W) | +0.186..+0.581 (3 W) | +0.895..+1.152 (3 W) | +4.29..+11.89 (3 B) | −1.34..−0.82 (3 W) |
| `crop` (rows 4–1075) | −1.13..−0.98 (3 W) | +0.0093..+0.0199 (2 W) | −16.10..−5.49 (3 W) | +0.199..+0.564 (3 W) | +0.825..+1.154 (3 W) | −4.32..−0.04 (2 W) | −1.13..−0.97 (3 W) |

The three clips without black areas (anime-grain's letterbox is discussed below); `reflect`
pooled over 3 seeds on anime-clean and cartoon-bright, 1 elsewhere; `crop` paired with the default
on the same rows (4–1075).

- **Reflect, replicate and grey repair the bottom band; all four spoil the rest,** on 80–100% of
  the frames (`reflect`, `replicate`, `crop`, frame by frame). Cropped, the picture's own bottom
  edge is no better than numz's padded one.
- **`reflect` over 3 seeds:** anime-clean PSNR-Y 26.92 (spread 0.33) against the default's 27.72
  (0.47), VMAF 57.2 (2.1) against 65.3 (3.3); cartoon-bright 26.46 (0.03) against 27.13 (0.25),
  VMAF 59.2 (0.11) against 65.3 (0.34). The seed ranges don't overlap; `lab` keeps the verdicts
  but on cartoon-bright's PSNR (−0.19, within).
- **Mid grey fails like reflect:** what matters is black, not a constant fill.
- **The letterboxed clip is the control:** anime-grain's 21 bottom rows are black in the input,
  so reflect and replicate pad black anyway and give a bit-identical output; cropped (still with
  17 black rows at the bottom), it is *better* than the default (+0.38 dB, LPIPS −0.0085).

### Black as an anchor

Mean luma offset of the output against the GT (8-bit levels, mean over frames, and its standard
deviation over frames), `none`:

| Clip | Default (3 seeds) | `reflect` | `grey` | `black+16` | `reflect+black+16` |
|---|---|---|---|---|---|
| anime-clean | +3.26..+3.35, sd 1.01 | +3.72..+3.75, sd 1.22–1.27 | +3.78, sd 1.25 | +3.54, sd 0.96 | +3.61, sd 0.97 |
| anime-dark | +3.06..+3.12, sd 1.76 | +2.33, sd 2.30 | +2.04, sd 2.49 | +2.98, sd 1.79 | +2.74, sd 1.78 |
| cartoon-bright | +1.95, sd 1.04 | +3.22..+3.31, sd 1.13–1.15 | +3.28, sd 1.17 | +2.26, sd 1.05 | +2.36, sd 1.05 |

The offset barely depends on the seed (±0.1) but on the padding: without black rows the
frame-to-frame drift grows by 9–41% and the level moves (cartoon-bright +1.3); with 8 to 24
black rows, right under the picture or 8 reflected rows below it, the drift stays at the
default's and the level within 0.4 of it. `lab` brings every variant's mean offset
under 0.43 levels, yet the rendering differences remain (the verdicts above hold after `lab`).
Why the model needs a black reference is not established; the measurements only show that it
does.

### reflect+black+16

`NUM_PAD=reflect+black+16` reflects to the multiple of 16 (the 8 rows above the edge continue the
picture), then adds 16 black rows below; `black+16` adds them to numz's zeros (24 black rows).
Both are trimmed after decoding like numz's padding. `NUM_PAD=reflect>=8+black+16` reflects at
least 8 rows (the fewest that reach a multiple of 16) before the black rows: at 1080p that is 8,
so it is exactly `reflect+black+16` (checked by the selftest) and the 1080p figures below are its
own; at 720p it reflects 16.

| Variant | PSNR-Y | LPIPS | VMAF | ΔE00 lf | Temporal error lf | Temporal error | Bottom 16 rows | Rest of the frame |
|---|---|---|---|---|---|---|---|---|
| `black+16`, `none` | +0.06..+0.09 (–) | −0.0070..−0.0010 (1 B) | −0.02..+0.83 (–) | +0.016..+0.152 (3 W) | −0.030..+0.089 (1 W) | −0.054..+0.052 (–) | +0.35..+0.90 (–) | +0.01..+0.07 (–) |
| `black+16`, `lab` | +0.15..+0.21 (–) | −0.0070..−0.0009 (1 B) | +0.29..+1.16 (1 B) | −0.017..+0.018 (1 B) | +0.003..+0.013 (–) | −0.050..−0.024 (–) | +0.45..+0.93 (–) | +0.08..+0.18 (–) |
| `reflect+black+16`, `none` | +0.29..+0.40 (3 B) | −0.0069..−0.0016 (2 B) | −1.03..+1.15 (1 B) | −0.011..+0.149 (1 B, 2 W) | +0.005..+0.090 (2 W) | −0.133..+0.051 (–) | +7.07..+13.38 (3 B); letterbox −5.03 (W) | −0.13..+0.29 (2 B) |
| `reflect+black+16`, `lab` | +0.33..+0.54 (3 B) | −0.0068..−0.0010 (2 B) | −0.45..+1.00 (1 B) | −0.026..+0.003 (1 B) | +0.003..+0.041 (1 W) | −0.134..+0.059 (–) | +2.10..+17.36 (4 B) | −0.01..+0.33 (1 B) |

`black+16` on the 3 clips without black areas; `reflect+black+16` on all 4.

- **`reflect+black+16` keeps the anchor and repairs the band:** the bottom 16 rows gain 7–17 dB
  (`none`: anime-clean 25.0 dB, 3.2 below its rest where the bicubic baseline's bottom is 5.3
  below; anime-dark 36.0 and cartoon-bright 32.4, above their rest) and the rest is unchanged;
  the frame as a whole gains on PSNR-Y and LPIPS. Its cost is a low-frequency colour shift of up
  to +0.15 ΔE00 on anime-clean and a little more low-frequency flicker (+0.09 at most) in `none`,
  both gone after `lab`.
- **More black alone changes little at 1080p:** `black+16` stays within the spread but for a
  low-frequency colour shift in `none` (+0.02 to +0.15 ΔE00), and its band gains only 0.4–0.9 dB:
  the picture's edge still touches black.
- **On the letterboxed clip** the reflected rows are black (its own bars), so `reflect+black+16`
  amounts to 24 more black rows: the frame is better (+0.29 dB, LPIPS −0.0069, VMAF +1.15;
  `lab`: +0.33 dB) and its mean luma offset is unchanged (+1.99 against +2.00, sd 0.42 both).
  Its bottom 16 rows are black in the GT and stay black within 0.2–0.4 levels RMS in both (PSNR
  55–62 dB, 5–6 frames exact): the −5.03 dB is a difference between very high PSNRs (`lab`:
  +2.10).
- Cost: 16 more rows, +1.5% of the DiT tokens and of the VAE's work at 1080p (138 latent rows
  instead of 136).

### 720p: no padding at all

1280×720 is a multiple of 16: numz pads nothing, so there are no black rows. Same d1 inputs at
`--resolution 720` (×1.33), scored against the GT downscaled to 1280×720 (zimg spline36 on the
16-bit GT, frame-exact; baseline: the input upscaled with Catmull-Rom). `black+16` adds 16 black
rows under the picture (736 rows, trimmed after decoding). One seed: there is no 720p spread, the
1080p spreads give the scale. The default lands at PSNR-Y 25.2–31.1 dB (bicubic 33.4–43.8).

| `black+16` − default | PSNR-Y | LPIPS | VMAF | ΔE00 lf | Temporal error lf | Temporal error | Bottom 16 rows | Rest of the frame |
|---|---|---|---|---|---|---|---|---|
| `none` | +0.22..+0.42 | −0.0121..−0.0012 | +1.83..+6.40 | −0.397..−0.068 | −0.307..−0.061 | −0.554..−0.124 | −10.35..−2.54 | +0.23..+0.61 |
| `lab` | +0.03..+0.42 | −0.0123..−0.0012 | +1.05..+5.43 | −0.234..−0.052 | −0.153..−0.048 | −0.535..−0.110 | −12.03..−0.23 | +0.19..+0.70 |

All 4 clips move the same way on every column; only cartoon-bright's PSNR-Y under `lab` (+0.03)
and anime-grain's black bottom rows under `lab` (−0.23) are not significant.

- **Black rows help where numz adds none:** the rest of the frame gains 0.23–0.61 dB, VMAF
  1.8–6.4, and the frame is steadier (temporal error −0.12 to −0.55, low-frequency flicker −0.06
  to −0.31), beyond the 1080p spreads for VMAF and low-frequency flicker on all 4 clips
  (anime-clean's VMAF at the edge) and for colour on 3. The anchor is not an artefact of 1080p's
  8 padded rows.
- **Black right under the picture damages its bottom band** (−2.5 to −10.4 dB on the bottom 16
  rows), as numz's padding does at 1080p. At 720p `reflect+black+16` would put the black rows
  directly under the picture too (there is nothing to reflect up to the multiple of 16): the
  protection needs a reflected margin of its own (8 rows at 1080p).
- **Pending:** `NUM_PAD=reflect>=8+black+16` adds that margin (at 720p, 16 reflected rows, then
  16 black: 752 rows, trimmed after decoding). It is implemented and selftested, but its 720p runs
  did not get GPU time before the deadline: whether it keeps this frame-wide gain while sparing
  the bottom band at 720p is not measured.

## Degradation d2

anime-clean and anime-grain with a stronger degradation (area downscale, CRF 26): the default
lands at PSNR-Y 27.18 and 28.27 dB (bicubic 38.09 and 36.26); seed spreads 0.47 and 0.13 dB
PSNR-Y, 3.27 and 0.67 VMAF, 0.021 and 0.0027 LPIPS.

| Variant, `none` | PSNR-Y | LPIPS | DISTS | VMAF | ΔE00 lf | Temporal error | Bottom 16 rows | Rest of the frame |
|---|---|---|---|---|---|---|---|---|
| `reflect+black+16` | +0.26..+0.27 (1 B) | −0.0078..−0.0028 (1 B) | −0.0049..−0.0024 (1 B) | −1.66..+1.54 (1 B) | −0.025..+0.122 (1 W) | −0.150..+0.191 (1 B) | +6.53 (B); letterbox −5.63 (W) | −0.19..+0.27 (1 B) |
| `lanczos` | +0.17..+0.22 (1 B) | −0.0105..−0.0044 (1 B) | −0.0057..−0.0025 (1 B) | +0.74..+1.07 (1 B) | +0.008..+0.017 (1 W) | −0.235..−0.124 (–) | −0.04..+0.07 (–) | +0.17..+0.25 (1 B) |
| `spline36` | +0.16..+0.17 (1 B) | −0.0060..−0.0026 (–) | −0.0033..−0.0020 (1 B) | +0.88..+1.10 (1 B) | −0.012..−0.009 (–) | −0.153..−0.113 (–) | −0.45..+0.06 (–) | +0.17..+0.18 (1 B) |
| `noaa` | +0.16..+0.26 (1 B) | −0.0078..−0.0051 (1 B) | −0.0033..−0.0029 (1 B) | +0.72..+0.84 (1 B) | +0.063 (2 W) | −0.235..−0.222 (1 B) | +0.05..+4.82 (1 B) | +0.18..+0.26 (1 B) |

The "B" verdicts are anime-grain's: anime-clean moves the same way within its 3.6 times wider
spread. `lab` gives the same pattern (`reflect+black+16`: +0.31 to +0.36 dB, bottom rows +1.7 and
+7.1 dB).

- **`reflect+black+16` holds at d2:** the band repaired on anime-clean (+6.5 dB), the frame
  unchanged or better.
- **The kernels matter a little more on a softer input:** all three move both clips toward the
  GT (PSNR-Y +0.16 to +0.26 dB, LPIPS −0.003 to −0.011, VMAF +0.7 to +1.1), beyond the spread on
  anime-grain only, and torchvision without antialias again adds low-frequency colour error
  (+0.063 ΔE00 on both clips).

## Float16 attention

`NUM_ATTN=fp16` casts q, k and v to float16 instead of bfloat16 for `flash_attn_2`, with autocast
off around the kernel: the attention of GPUs without bfloat16, where numz also runs autocast, the
VAE and the text embeddings in float16 (not emulated here). Within the seed spread on all 4 clips
and every metric (row `attn16` in [Numerics](#numerics); `lab`: PSNR-Y −0.06 to +0.01 dB, VMAF
−0.44 to +0.04, no verdict): float16's three extra mantissa bits and narrower range change
nothing measurable here.

## Visual review

Side-by-side crops, 1:1, GT | default | variant, 480×270 each, where the variant differs most from
the default (`max`, any frame) and, for padding, where its bottom 16 rows differ most
(`bottom`); each with a `.diff.png` (|variant − default| on luma, ×8). Names:
`<variant>-<clip>-<max|bottom>-f<frame>-x<x>-y<y>.png` (crop: y in the 1072-row picture).

| File | What to look at |
|---|---|
| `bicubic-anime-clean-max-f6-x392-y632.png` | GT \| default \| bicubic: what the model redraws (default 12.2 levels from the GT in this window, bicubic 2.8) |
| `reflect-anime-dark-max-f3-x696-y168.png`, `grey-anime-dark-max-f3-x688-y160.png`, `crop-anime-dark-max-f3-x696-y160.png` | frame 3 re-rendered far from the bottom edge by every padding without black (16–19 levels from the GT where the default has 6.6–6.8) |
| `reflect-cartoon-bright-bottom-f3-x1312-y810.png`, `replicate-anime-clean-bottom-f29-x136-y810.png` | the bottom band repaired: bottom 16 rows 18.7 → 9.7 and 24.1 → 16.1 levels from the GT |
| `reflblack16-<clip>-bottom-*.png`, `reflblack16-<clip>-max-*.png` | the recommended padding: band repaired, largest change elsewhere 2.1–4.1 levels, as close to the GT there as the default (±0.4) |
| `noaa-`, `spline36-`, `lanczos-anime-clean-max-f1-*.png` | resize kernels: 1.2–1.7 levels at most |

## Caveats

- **Four animated clips**, 45 frames each, from 1080p sources; live action, film grain at 4K and
  other content are not measured. anime-grain is letterboxed, which changes the padding results.
- **One model and one size:** 7B fp16 at 1080p (one 720p test), `flash_attn_2`; the 3B, fp8 and
  GGUF weights and 4K are not covered.
- **One batch of 45 frames per clip:** batch boundaries are measured in
  [stitching.md](stitching.md); the 4n + 1 tail padding (mirror against repeat) is not.
- **Single seed per variant** (but `reflect`), judged against a 3-seed spread: a variant whose
  effect equals one seed's would not show; the padding effects reach 1.7 to 8 times the spread
  on PSNR-Y, more on VMAF and LPIPS. The 720p test has no spread of its own.
- **Fidelity is not quality:** a variant closer to the GT is not necessarily nicer to watch, and
  the model is far from the GT anyway. The [review crops](#visual-review) show the largest
  differences for a visual check.
- **Paired differences only:** VMAF and DISTS are not read as absolute quality here; VMAF's model
  assumes 1080p viewing, and the 720p scores use it too.

## Reproduce

```bash
S=scripts; C=/path/to/clips; OUT=/path/to/out
# clips: ground truth, d1 and d2 inputs, bicubic baselines, the crop control
python3 $S/fr_clips.py scan /path/to/source.mkv --json scan.json     # candidate single shots
python3 $S/fr_clips.py make /path/to/source.mkv --start FIRST_FRAME --frames 45 --name anime-clean \
  --out $C --degrade d1,d2 --crop
python3 $S/fr_clips.py verify $C/anime-clean --cv2-python /path/to/seedvr2/.venv/bin/python --source
# one run: the default (no NUM_ switch) or a variant, with its lab rendering, 16-bit RGB masters
python3 $S/bench.py run q1-anime-clean-d1-reflblack16-s42 --wrap $S/numerics_patch.py \
  --wrap $S/ffv1_out.py --env FFV1_OUT_KEEP=0 --env NUM_CC_EXTRA=lab \
  --env NUM_PAD=reflect+black+16 -- $C/anime-clean.d1.lr.mkv --output $OUT/anime-clean-d1/ \
  --model_dir /path/to/models --dit_model seedvr2_ema_7b_fp16.safetensors --resolution 1080 \
  --attention_mode flash_attn_2 --batch_size 45 --load_cap 45 --color_correction none --seed 42
# variants: NUM_ROPE=fp32, NUM_VAE=sample, NUM_PREP=fp32, NUM_DIT_WEIGHTS=/path/to/seedvr2_ema_7b.pth
#   NUM_VAE_WEIGHTS=/path/to/ema_vae.pth, NUM_ATTN=fp16, NUM_RESIZE=tv-noaa|zimg-spline36|zimg-lanczos,
#   NUM_PAD=reflect|replicate|grey|black+16|reflect+black+16|reflect>=8+black+16;
#   crop: the .d1.crop.lr.mkv input at 1072; 720p: --resolution 720, scored against the GT
#   downscaled to 1280x720 (ffmpeg -vf zscale=w=1280:h=720:filter=spline36:dither=none)
# metrics: all outputs of a clip in one call (the GT decoded once); the lab renderings as VARIANT+lab
python3 $S/fr_metrics.py $C/anime-clean.gt.mkv --clip anime-clean-d1 --json-dir m/d1-none \
  --out def 42 $OUT/anime-clean-d1/q1-anime-clean-d1-def-s42.mkv \
  --out reflblack16 42 $OUT/anime-clean-d1/q1-anime-clean-d1-reflblack16-s42.mkv \
  --out bicubic 0 $C/anime-clean.d1.bicubic.mkv
# crop pairs: GT crop.gt.mkv with --rows 4:1076 (1080-row outputs are cut to rows 4..1075)
python3 $S/fr_metrics.py --summary m/d1-none --default def --reference bicubic,ident > d1-none.md
python3 $S/fr_metrics.py --summary m/d1-lab --default def+lab --reference bicubic,ident+lab > d1-lab.md
python3 $S/fr_metrics.py --make-test $C/anime-clean.gt.mkv --work val --blur 1 --noise 2   # known cases
python3 $S/numerics_patch.py --selftest     # zimg resize, 8-bit recovery, every padding mode (CPU)
```

<details>
<summary>Run names (7B fp16, <code>flash_attn_2</code>, one batch of 45 frames, <code>--color_correction
none</code> plus the <code>lab</code> rendering of the same run)</summary>

`q1-<clip>-<degradation>-<variant>-s<seed>`, clips `anime-clean`, `anime-grain`, `anime-dark`,
`cartoon-bright`. d1, 1080p: `def` (seeds 42, 43, 1234), `parity`, `rope`, `vaes`, `prep`, `w32`,
`attn16`, `noaa`, `spline36`, `lanczos`, `reflect`, `replicate`, `crop` (1072 rows), seed 42;
`reflect` also at 43 and 1234 on anime-clean and cartoon-bright; `grey`, `black16`, `reflblack16`
on anime-clean, anime-dark and cartoon-bright, `reflblack16` also on anime-grain; `defcheck`
(anime-clean, the default re-run, compared bit for bit). `ident`: `q1-<clip>-gt-ident-s42`, the
8-bit GT as input. 720p: `q1-<clip>-d1-720-{def,black16,refl8black16}-s42`. d2, anime-clean and anime-grain:
`def` (42, 43, 1234), `reflblack16`, `lanczos`, `noaa`, `spline36`. Metric labels:
`<clip>-d1`, `<clip>-d1-crop` (rows 4–1075), `<clip>-d1-720`, `<clip>-d2`; `lab` renderings as
`<variant>+lab`, the baselines as `bicubic`.

</details>

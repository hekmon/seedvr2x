# Numerics and input preparation, against a ground truth

> Status: **measured** with [`scripts/numerics_patch.py`](../scripts/numerics_patch.py) (one switch
> per numerics or input-preparation choice, patched in at import; with no switch set, the output
> is bit-identical to numz's), [`scripts/ffv1_out.py`](../scripts/ffv1_out.py) (16-bit RGB
> masters), [`scripts/fr_clips.py`](../scripts/fr_clips.py) (ground-truth clips and degraded
> inputs), [`scripts/fr_metrics.py`](../scripts/fr_metrics.py) (full-reference metrics and
> paired statistics), [`scripts/chroma_kernels.py`](../scripts/chroma_kernels.py) (the master's
> 4:2:0 conversion) and [`scripts/cut_metrics.py`](../scripts/cut_metrics.py) `phase` (the
> 4-frame latent grid), for the "Numerics" question in
> [DESIGN.md](../../seedvr2x/DESIGN.md#to-measure). SeedVR2 `4490bd1`, 7B fp16, `flash_attn_2`,
> one batch of 45 frames, 1080p (one 720p test), `--color_correction none`, with `lab` rendered
> from the same run. Four animated clips, a fifth (a bright sky) for the VAE decode precision,
> and three more for [live action and a bright anime](#live-action-and-a-bright-anime): two
> letterboxed live-action films with grain, and a bright anime.

In short (4 animated clips of 45 frames, and for parts of it 4 more: a bright sky, two
letterboxed live-action films with grain, a bright anime; a ×2 upscale of a mildly degraded input
to 1080p; every variant paired with numz's default at the same seed, and judged against the spread
of 3 seeds):

- **SeedVR2 re-renders the picture: a plain bicubic upscale is closer to the source on every
  pixel-wise metric.** numz's default lands at PSNR-Y 27.1–32.1 dB, VMAF 51–65 and a
  low-frequency ΔE00 of 1.8–2.6, where a Catmull-Rom upscale of the same input gets 32.2–43.2 dB,
  83–92 and 0.6–0.7; LPIPS is worse on 3 of 4 animated clips. On live action the perceptual
  metrics side with the model (LPIPS 0.17 against 0.27 and 0.087 against 0.107, DISTS likewise)
  while PSNR-Y (−6.8 to −7.1 dB), VMAF, colour and temporal error still side with bicubic: the
  model redraws grain and detail of the right kind at the wrong place. Given the ground truth
  itself as input (animated clips), the model moves it as far (25.4–30.4 dB, ΔE00 2.4–3.5).
  Registration is not the cause (shifts ≤ 0.12 px, every frame matches its own): the model
  redraws texture, and its mean brightness is off by −0.7 to +6 levels, changing from frame to
  frame (−6.9 to +7.6 on the ground truth itself). These metrics compare variants, paired and
  against the seed spread; they don't rate the model.
- **The seed is the noise floor:** 3 seeds of the default spread by 0.03–0.51 dB PSNR-Y,
  0.0001–0.018 LPIPS and 0.13–5.5 VMAF, depending on the clip (5.5: live-vfx's flickering light).
- **Every fourth frame is the model's best.** The causal VAE gives a unit's first frame a latent
  of its own, then packs 4 frames per latent. The last frame of each group is the closest to the
  GT on 8 of 8 clips by VMAF and LPIPS, 7 of 8 by PSNR-Y: against the group's second frame,
  PSNR-Y +0.8 to +5.0 dB (the smallest on the slow sky pan) and VMAF +2.2 to +19.3, the 3 seeds
  within 0.5 dB of each other; the bicubic baseline has no such pattern (0.2–0.8 dB, in no
  consistent direction). On fast motion the frames inside a group carry ghosts, doubled line art
  on the bright cartoon. `lab` keeps it
  ([Every fourth frame](#every-fourth-frame-the-latent-grid)).
- **ByteDance's numerics change little:** float32 RoPE angles, VAE posterior sampling, a
  float32 input chain, the float32 weights, bfloat16 DiT norms, a VAE decode under bfloat16
  autocast, and all of them together stay within the seed spread on all 4 animated clips for
  PSNR, SSIM, LPIPS, DISTS, VMAF and temporal error (|ΔPSNR-Y| ≤ 0.14 dB, |ΔVMAF| ≤ 1.1), with or
  without `lab`; on anime-clean they move the output by 0.1–0.5 levels on average, a seed by
  1.9. So does float16 attention, the path of GPUs without bfloat16 (|ΔPSNR-Y| ≤ 0.06 dB). On
  the live-action and bright-anime clips all of them together (`parity`) move slightly toward
  the GT: low-frequency colour on 3 of 3 (ΔE00 lf −0.05 to −0.09), temporal error on both
  live-action clips (−0.07 and −0.17), less after `lab`.
- **numz's bfloat16 decode costs colour, not banding.** The VAE decodes in bfloat16 and numz
  normalises in bfloat16: its 16-bit masters hold at most 129 codes per channel from mid-grey
  up, one 8-bit level apart. Yet CAMBI finds no banding in any output (≤ 0.004, a bright sky
  clip included): the model's rendering dithers the steps. Decoding in float16 (or float32, the
  same picture 0.02 levels apart) brings low-frequency colour closer to the source on 8 of 8
  clips (ΔE00 lf −0.05 to −0.09); float16 costs at most a little LPIPS (+0.0003 to +0.0024 on 3
  of 8 clips) and never overflowed (largest activation 19,088 of 65,504), float32 doubles the
  decode's memory. Recommended: float32 after the decoder (free) and a float16 decode with a
  non-finite check.
- **The `yuv420p10le` master: keep zscale's bilinear chroma, on one slice.** Through 4:2:0 and
  back (Catmull-Rom, as seedvr2x reads), bilinear keeps the model's output the closest to the GT
  of 6 kernels (8 of 8 clips by PSNR-Cb and ΔE00, 7 of 8 by PSNR-Cr); Catmull-Rom, spline16/36
  and lanczos lose 0.3–0.5 dB of PSNR-Cb/Cr and ring more. The round trip even brings the
  output's chroma closer to the GT than the unconverted output (PSNR-Cb +1.1 dB), the GT's own
  chroma being 4:2:0; the usual metrics don't move. But ffmpeg runs zscale in slices, one per CPU
  by default, and the bytes depend on them: the master's chroma on 1.7% of the samples (up to 13
  ten-bit codes), a 10-bit 4:2:0 source's RGB on nearly every sample (up to 0.57 level) from 4
  slices on. `threads=1` on zscale (libavfilter's generic per-filter option) restores the
  single-slice bytes for 2.5 ms per 1080p frame
  ([The master's chroma](#the-masters-chroma-420-kernels-and-zscales-slices)).
- **The resize kernel is second order:** torchvision's bicubic without antialiasing adds a
  little low-frequency colour error and flicker on 7 of 7 clips (live-slow: every metric worse).
  The sharper zimg kernels help a little on some grainy or detailed clips (lanczos on the
  grainy anime clip +0.24 dB, LPIPS −0.005; both on live-slow, VMAF +0.2 to +0.3) but add
  low-frequency colour error and flicker on the fast live-action clip (lanczos: temporal error
  +0.11, ΔE00 lf +0.05; spline36 half that). numz's antialiased bicubic stays a sound default;
  spline36 only where the pipeline resizes with zimg anyway.
- **Film grain comes back at half strength:** on the grainy live-action clip the output holds
  0.93–0.97 levels of grain against the source's 1.75 and the degraded input's 0.45 (std of the
  high-pass luma in the smoothest 30% of the picture), whatever the variant, `lab` or not. On the
  slow film, whose degraded input keeps most of its grain (1.29 of 1.59), the output holds a
  little more than the source (1.86–1.89) ([Grain](#grain), corrected on 2026-10-05).
- **numz's black padding is both a defect and an anchor.** At 1080p numz pads 8 black rows: the
  bottom 16 rows lose 3–10 dB against the rest of the frame. Padding without black (reflect,
  replicate, mid grey) repairs that band, but it, and no padding at all (a crop to 1072 rows),
  make the whole frame worse on every clip without black areas of its own: −0.6 to −1.1 dB
  PSNR-Y, −5.5 to −16 VMAF, +0.8 to +1.3 temporal error, far beyond the seed spread. Without black
  rows the model's frame-to-frame brightness drift grows (its sd by up to 41%).
- **Recommended: always reflect at least 8 rows, then 16 black rows**
  (`NUM_PAD=reflect>=8+black+16`). At 1080p this is exactly the measured `reflect+black+16`: the
  bottom band gains 7–13 dB and the frame stays as good or gets better (PSNR-Y +0.29 to +0.40 dB
  on 3 of 4 clips, the letterboxed one included). At 720p, where numz pads nothing, it reflects
  16 rows before the black ones and beats the default on all 4 clips (PSNR-Y +0.07 to +1.03 dB,
  VMAF +0.9 to +10.2, LPIPS and temporal error lower, the bottom band +1.2 to +2.0 dB on the 3
  clips without black bars); black rows right under the picture steady the model too but damage
  that band (−2.5 to −10.4 dB). A stronger degradation (area downscale, CRF 26, 2 clips) gives
  the same picture, the resize kernels gaining a little more.

## Why it matters for seedvr2x

seedvr2x vendors the model code and rewrites the runtime around it
([DESIGN.md](../../seedvr2x/DESIGN.md#vendored-model-code)): every numerics choice and the whole
input preparation become ours. numz and ByteDance's reference differ in RoPE precision, the
casts of the input frames, the VAE's posterior mode against a sample, the weights' dtype, the
DiT norms' output dtype, the VAE decode's autocast, the resize kernel, and padding (numz)
against cropping (ByteDance) to multiples of 16. DESIGN.md
asks which of these matter. A difference only counts here if it moves the output closer to the
source than a change of seed does.

## The full-reference protocol

### Clips

Single shots of 45 frames from 1080p 8-bit 4:2:0 sources (BT.709, limited range, 23.976 fps;
live-slow 24 fps, live-vfx's source untagged and read as BT.709), extracted by decoding from the
start and counting frames. Statistics of the ground truth (Y on the 8-bit scale):

| Clip | Content | Mean Y | Near black (Y < 10) | Motion (mean \|ΔY\| per frame) | Detail (Laplacian variance) |
|---|---|---|---|---|---|
| anime-clean | clean digital anime | 115 | 0.0% | 12.3 | 37 |
| anime-grain | grainy anime, letterboxed (23 black rows at the top, 21 at the bottom) | 76 | 8.9% | 10.5 | 59 |
| anime-dark | dark anime | 53 | 6.8% | 12.3 | 12 |
| cartoon-bright | bright cartoon | 163 | 0.1% | 17.1 | 277 |
| anime-sky | bright sky and clouds, digital anime, slow pan ([VAE decode precision](#vae-decode-precision) only) | 216 | 0.0% | 3.4 | 8 |
| live-vfx | live action with VFX, fast: a soldier's face, flying debris, flickering light; film grain; letterboxed (140 black rows at the top and the bottom) | 78 | 26.0% | 15.3 | 86 |
| live-slow | live action from a slow film: people before a sunlit window grille, camera move; film grain; letterboxed (131 black rows at the top, 132 at the bottom) | 88 | 24.4% | 20.1 | 100 |
| anime-bright | bright anime: a character on white, line art and credit text (an opening sequence) | 174 | 5.7% | 20.2 | 237 |

The last three are measured in [Live action and a bright anime](#live-action-and-a-bright-anime)
with a subset of the variants; their verdicts are also rows of the tables below.

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

- **Determinism:** the default re-run with `numerics_patch.py`, after the padding modes and again
  after the norm and VAE-autocast switches (`defcheck`, `defcheck2`), reproduced the first default
  run bit for bit (45 of 45 frames, `none` and `lab`). Where a variant's input is
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
| live-vfx | default | 31.75 | 0.927 | 0.173 | 0.157 | 67.6 | 2.06 | 5.48 |
| | bicubic | 38.55 | 0.951 | 0.266 | 0.202 | 85.4 | 0.61 | 2.07 |
| live-slow | default | 28.51 | 0.920 | 0.087 | 0.074 | 75.2 | 1.84 | 7.97 |
| | bicubic | 35.65 | 0.970 | 0.107 | 0.104 | 90.3 | 0.61 | 2.60 |
| anime-bright | default | 25.12 | 0.932 | 0.082 | 0.083 | 69.6 | 2.16 | 9.42 |
| | bicubic | 32.16 | 0.968 | 0.080 | 0.067 | 85.5 | 0.87 | 4.07 |

- **The model is further from the source than bicubic** on every animated clip and metric but
  LPIPS on cartoon-bright: 5.1 to 12.0 dB of PSNR-Y, 16 to 32 points of VMAF, ΔE00 2.5 to 4.1
  times higher, temporal error 1.9 to 3.6 times. The windows where it departs most from the
  baseline show redrawn lines and texture (review crops `bicubic-<clip>-max-*`: mean |ΔY| 12–16
  levels against the GT where the baseline has 1.4–6.5).
- **On live action the perceptual metrics side with the model:** LPIPS 0.173 against bicubic's
  0.266 on live-vfx, 0.087 against 0.107 on live-slow, DISTS 0.157 against 0.202 and 0.074 against
  0.104, far beyond the seed spreads (≤ 0.008), while PSNR-Y (−6.8 and −7.1 dB), VMAF (−18 and
  −15), colour (ΔE00 lf 3.0–3.4 times higher) and temporal error (2.7–3.1 times) still side with
  bicubic. The blurred upscale of a grainy, detailed picture is what LPIPS and DISTS penalise;
  the model redraws texture of the right kind, at the wrong place for the pixel metrics.
- **`ident`**, the GT itself (8-bit) at `--resolution 1080`, is as far or further: the degradation
  is not what limits fidelity, the model's rendering is. It is closer on LPIPS on 3 clips
  (anime-dark: 0.150 against 0.213) and further on colour on all 4.
- **`lab` brings the default closer to the GT:** PSNR-Y +0.9 to +4.0 dB (anime-dark 36.09),
  ΔE00 lf down 37–55% (1.07–1.58), VMAF +3.9 to +6.6, LPIPS slightly lower on all 4 clips; on
  live-vfx and live-slow PSNR-Y +3.5 and +1.4 dB, ΔE00 lf −55% and −40%, VMAF +5.2 and +2.9,
  LPIPS lower. **anime-bright is the exception:** `lab` gains 0.55 dB of PSNR-Y and 15% of ΔE00 lf
  but costs LPIPS +0.019, DISTS +0.012, SSIM −0.005 and VMAF −3.1, far beyond the seed spreads.

## Every fourth frame: the latent grid

SeedVR2's causal VAE encodes a unit's first frame alone, then 4 frames per latent, and its decoder
rebuilds each group of 4 from one latent. Per frame, against the bicubic baseline of the same clip
(which shows whether the input itself varies with the frame's place), numz's default without
`lab`, 3 seeds each, frames 1–44 by their place in their group (0 to 3, 3 = the group's last;
`cut_metrics.py phase`):

| Clip | PSNR-Y − bicubic, place 0 / 1 / 2 / 3 | Gap 3 − 1 (3 seeds) | VMAF gap 3 − 1 | Bicubic's own PSNR-Y, place 0 / 1 / 2 / 3 |
|---|---|---|---|---|
| anime-clean | −12.04 / −12.65 / −11.99 / −11.28 | +1.36 (+1.35…+1.39) | +8.9 | 39.84 / 39.63 / 39.66 / 39.51 |
| anime-grain | −8.42 / −8.59 / −8.30 / −7.44 | +1.15 (+1.14…+1.18) | +8.4 | 37.29 / 37.31 / 37.20 / 37.64 |
| anime-dark | −10.67 / −13.22 / −12.45 / −8.24 | +4.98 (+4.91…+5.03) | +19.2 | 42.79 / 43.11 / 43.08 / 43.62 |
| cartoon-bright | −7.19 / −5.64 / −5.28 / −2.60 | +3.04 (+2.91…+3.17) | +19.3 | 32.23 / 32.14 / 32.31 / 32.21 |
| anime-sky | −15.18 / −16.19 / −15.65 / −15.39 | +0.80 (+0.77…+0.84) | +2.2 | 44.77 / 44.80 / 44.72 / 44.46 |
| live-vfx | −7.27 / −8.52 / −7.34 / −4.36 | +4.17 (+3.96…+4.45) | +12.2 | 38.57 / 38.53 / 38.38 / 38.59 |
| live-slow | −8.33 / −7.98 / −7.53 / −5.01 | +2.97 (+2.94…+3.00) | +12.9 | 35.73 / 35.65 / 35.60 / 35.55 |
| anime-bright | −8.60 / −7.57 / −7.41 / −4.88 | +2.69 (+2.64…+2.78) | +13.0 | 32.17 / 32.10 / 32.01 / 32.26 |

- **The last frame of each group is the model's best:** the closest to the GT on 7 of 8 clips by
  PSNR-Y (anime-sky, a slow pan, is flat), 8 of 8 by VMAF and by LPIPS (place 3 − 1: −0.003 to
  −0.045); the gap between places 3 and 1 also favours place 3 on 8 of 8 clips by DISTS and ΔE00
  lf, 7 of 8 by SSIM. Place 1 trails place 3 by 0.8–5.0 dB of PSNR-Y. The seeds agree to 0.5 dB
  on the gap; the bicubic baseline's own range over the places is 0.2–0.8 dB, in no consistent
  direction, so the input's coding doesn't cause it.
- **It shows as ghosts on fast motion.** On the bright cartoon (a character walking past the
  camera, frames 34, 36 and 38, GT | bicubic | default with `lab`): frames 34 and 38, place 1,
  double the line art and let the background through the body; frame 36, place 3, is clean
  ([review crop](#visual-review)).
- **`lab` keeps it:** with `lab` the PSNR-Y gap is −0.1 to +3.1 dB and the VMAF gap +1.3 to
  +17.7.
- A shot's first frame, a latent of its own, is the same effect: it is the frame closest to the
  GT on 8 of 8 shots ([cuts.md](cuts.md)), and runs whose 4-frame grids start on different frames
  differ with a period of 4 for the same reason. The period-4 pattern is the model's; nothing
  measured here removes it. Taking each frame from the run where it ends its group would take 4
  runs with the grid shifted (4× the GPU time): not measured.

## Seed bands

Spread (max − min) of the default's 3 per-seed means, `none` (`lab` within ±0.15 dB and ±0.4
VMAF of these):

| Clip | PSNR-Y | SSIM-Y | LPIPS | DISTS | VMAF | ΔE00 lf | Temporal error lf | Temporal error | Bottom 16 rows |
|---|---|---|---|---|---|---|---|---|---|
| anime-clean | 0.47 | 0.0124 | 0.0183 | 0.0099 | 3.30 | 0.026 | 0.042 | 0.563 | 0.56 |
| anime-grain | 0.19 | 0.0023 | 0.0034 | 0.0020 | 0.95 | 0.034 | 0.026 | 0.169 | – |
| anime-dark | 0.09 | 0.0012 | 0.0074 | 0.0049 | 2.18 | 0.001 | 0.050 | 0.088 | 1.18 |
| cartoon-bright | 0.25 | 0.0007 | 0.0002 | 0.0008 | 0.34 | 0.013 | 0.034 | 0.072 | 2.35 |
| anime-sky | 0.51 (0.07 without the bottom rows) | 0.0007 | 0.0014 | 0.0024 | 0.41 | 0.017 | 0.017 | 0.017 | 2.19 |
| live-vfx | 0.51 (`lab`: 1.11) | 0.0023 | 0.0080 | 0.0047 | 5.53 | 0.019 | 0.019 | 0.018 | – |
| live-slow | 0.03 | 0.0020 | 0.0001 | 0.0002 | 0.13 | 0.019 | 0.068 | 0.069 | – |
| anime-bright | 0.30 | 0.0020 | 0.0017 | 0.0023 | 0.66 | 0.019 | 0.056 | 0.308 | 1.11 |

A seed moves the whole clip together (one batch): seed 1234 is +0.37 dB on 100% of
anime-clean's frames. A variant measured at one seed is compared with this spread for that
reason. anime-grain's bottom rows are black in the GT and nearly so in every output; a few
frames match exactly, so their mean PSNR is infinite and the spread undefined (the same for the
live-action clips' letterbox bars). The spreads differ by clip far more than the means do:
live-slow's seeds agree within 0.03 dB and 0.13 VMAF, so differences of a few hundredths reach a
verdict there, while live-vfx's VMAF moves by 5.5 points from seed to seed.

## Numerics

### What differs from ByteDance on the 7B fp16 path

Checked in the code before patching anything (numz `4490bd1` paths under `src/`, ByteDance
`e4de8c2` paths as in its repository), and confirmed by the `NUM_CHECK` probe on the full 7B
model in the CLI:

| Item | numz | ByteDance | Differs? | Run |
|---|---|---|---|---|
| Compute dtype, attention | bf16 when a bf16 matmul works (`optimization/compatibility.py:684-698`); the DiT under bf16 autocast (`core/generation_phases.py:718-724`); q, k, v cast to it before the kernel (`models/dit_7b/attention.py:117-121`). q and k arrive in fp32 (the norms and RoPE output fp32 under autocast), v in bf16 | explicit `.bfloat16()` (`models/dit/nablocks/mmsr_block.py:130-132`) under bf16 autocast (`projects/inference_seedvr2_7b.py:119`) | no: bf16 in both | `attn16` = GPUs without bf16 |
| RoPE | angle table from the checkpoint's fp16 `rope.freqs` (128π stored as 402.0), angles computed in fp16 by rotary_embedding_torch, then cast to q's dtype (fp32) for the rotation (`models/dit_7b/rope.py:84-89`) | fp32 table and rotation (`models/dit/rope.py:84-88`, fp32 checkpoint) | yes: the table's precision | `rope` |
| Input frames | 8-bit → fp32 → fp16 (`inference_cli.py:697`) → bf16, resize and normalisation in bf16 (`core/generation_phases.py:380-388`) | uint8 → fp32, resize and normalisation in fp32, bf16 at encode (`projects/inference_seedvr2_7b.py:228-244`) | yes | `prep` |
| Weights | fp16 files, ByteDance's fp32 `.pth` rounded to the nearest fp16 (below), cast to bf16 per layer under autocast | fp32 `.pth`, cast under autocast | yes | `w32` |
| VAE encode | posterior mode (`models/video_vae_v3/modules/attn_video_vae.py:1688`) | posterior sample (`models/video_vae_v3/modules/attn_video_vae.py:1305`) | yes | `vaes` |
| Multiples of 16 | black padding, bottom and right, trimmed after decode (`data/image/transforms/divisible_crop.py:61-72`) | centre crop (`data/image/transforms/divisible_crop.py:36-39`) | yes | [Padding](#padding) |
| TF32 | off (numz never calls the init that sets it) | on for matmul and cuDNN (`common/distributed/basic.py:66-68`) | yes, no effect: bit-identical | smoke test |
| Conv3d bias | added separately by the Conv3d workaround (`models/video_vae_v3/modules/causal_inflation_lib.py:94-107`) | inside the convolution | yes, no effect: bit-identical with the workaround off | smoke test |
| DiT norms | custom RMS/LayerNorm, fp32 output under autocast (`models/dit_7b/normalization.py:28-97`) | Apex fused norms, bf16 in and out | yes | `norm16` |
| VAE decode | no autocast, norm casts removed (`models/video_vae_v3/modules/causal_inflation_lib.py:354-409`) | bf16 autocast, norm outputs cast back (`models/video_vae_v3/modules/causal_inflation_lib.py:330-367`) | yes | `vaeac` |

numz's `seedvr2_ema_7b_fp16.safetensors` and `ema_vae_fp16.safetensors` are ByteDance's
`seedvr2_ema_7b.pth` and `ema_vae.pth` rounded to the nearest fp16, ties to even, element for
element (checked on the files: all 1,128 and 250 tensors, same names, no metadata, fp16
subnormals kept, no value above 65504; truncation would differ on 50% of the elements, a cast
through bf16 on 87%).

The sampler, CFG (off, one step), the timestep and the positive/negative text embeddings are the
same (the embeddings are byte-identical files). The resize kernel is the same, torchvision's
bicubic with antialias, and the targets agree for 16:9 sources.

### Results

Paired difference to the default, `none`: the range over the 4 animated clips (the "3 new clips"
row: over live-vfx, live-slow and anime-bright), and how many clips are better (B) or worse (W);
"–" = within everywhere.

| Variant | Switch | PSNR-Y | LPIPS | DISTS | VMAF | ΔE00 lf | Temporal error |
|---|---|---|---|---|---|---|---|
| `rope` | `NUM_ROPE=fp32` | −0.05..+0.03 (–) | −0.0004..+0.0016 (–) | −0.0002..+0.0010 (–) | −0.78..+0.06 (–) | +0.002..+0.016 (1 W) | −0.014..+0.044 (–) |
| `vaes` | `NUM_VAE=sample` | −0.04..+0.02 (–) | −0.0004..+0.0012 (–) | −0.0003..+0.0008 (–) | −0.64..+0.03 (–) | +0.000..+0.016 (–) | −0.009..+0.037 (–) |
| `prep` | `NUM_PREP=fp32` | −0.06..+0.02 (–) | −0.0001..+0.0030 (–) | −0.0007..+0.0014 (–) | −0.58..+0.19 (–) | −0.035..−0.015 (3 B) | −0.065..+0.101 (–) |
| `w32` | the fp32 `.pth` weights | −0.09..+0.00 (–) | −0.0009..+0.0015 (–) | −0.0006..+0.0006 (–) | −0.60..−0.08 (–) | +0.006..+0.050 (3 W) | −0.005..+0.081 (–) |
| `parity` | `rope` + `vaes` + `prep` + `w32` | −0.09..+0.03 (–) | −0.0003..+0.0038 (–) | −0.0005..+0.0018 (–) | −0.80..+0.16 (–) | −0.019..−0.000 (1 B) | −0.074..+0.131 (–) |
| `parity`, 3 new clips | the same, on live-vfx, live-slow, anime-bright | −0.01..+0.38 (1 B) | −0.0001..+0.0008 (1 B) | −0.0000..+0.0003 (–) | −0.31..+0.18 (1 B) | −0.088..−0.046 (3 B) | −0.166..+0.060 (2 B) |
| `norm16` | `NUM_NORM=bf16` | −0.03..+0.01 (–) | −0.0004..+0.0015 (–) | −0.0003..+0.0011 (–) | −0.70..+0.08 (–) | −0.006..+0.017 (–) | −0.011..+0.024 (–) |
| `vaeac` | `NUM_VAE_AUTOCAST=1` | ±0.00 (–) | ±0.0000 (–) | ±0.0000 (–) | ±0.00 (–) | ±0.000 (–) | ±0.000 (–) |
| `parity2` | `parity` + `norm16` + `vaeac` | −0.12..+0.03 (–) | −0.0002..+0.0030 (–) | −0.0002..+0.0015 (–) | −1.06..+0.09 (–) | −0.016..+0.013 (1 B) | −0.074..+0.138 (–) |
| `attn16` | `NUM_ATTN=fp16` | −0.03..+0.01 (–) | −0.0004..+0.0007 (–) | −0.0003..+0.0004 (–) | −0.44..+0.04 (–) | −0.004..+0.004 (1 B) | −0.006..+0.035 (–) |

`parity2` covers every difference listed in
[What differs](#what-differs-from-bytedance-on-the-7b-fp16-path) but the multiples of 16 (TF32
and the Conv3d bias change no bit): ByteDance's numerics on numz's code.

- **Nothing reaches the seed spread on PSNR, SSIM, LPIPS, DISTS, VMAF or temporal error,** on any
  of the 4 animated clips, `none` or `lab` (`lab`: |ΔPSNR-Y| ≤ 0.14 dB, |ΔVMAF| ≤ 0.82).
- **On the 3 new clips `parity` moves a little further, toward the GT:** low-frequency colour on
  all 3 (ΔE00 lf −0.05 to −0.09, 2.4 to 4.6 times their spreads, as much as a float16 decode),
  temporal error on both live-action clips (live-vfx −0.17 and its low-frequency flicker −0.18,
  nine times its spread; live-slow −0.07), and on live-slow, whose seeds agree within 0.03 dB,
  PSNR-Y +0.12 dB and VMAF +0.18. Small throughout (live-vfx's +0.38 dB stays within its
  0.51 dB spread), and smaller after `lab`: ΔE00 lf −0.008 on the live clips, temporal error
  −0.045 on live-vfx, PSNR-Y +0.03 dB on live-slow. Which of the four switches does it is not
  measured on these clips.
- On those 4, the only verdicts are low-frequency colour shifts of 0.002–0.05 ΔE00 against values of 1.8–2.6,
  and low-frequency flicker 0.08–0.09 lower on anime-dark (`prep`, `parity`, `parity2`): they pass
  only because those spreads are tiny (ΔE00 0.001–0.034). After `lab` they vanish, but for DISTS
  changes of ±0.0004 on cartoon-bright, whose DISTS spread is 0.0003.
- `w32` loads ByteDance's float32 checkpoint, whose RoPE table is already float32, so it includes
  `rope`; both give the same verdicts. The float16 file numz ships costs nothing measurable.
- **The norms and the VAE decode took effect** (`NUM_CHECK`, every run): with `NUM_NORM=bf16` the
  DiT's RMS norms return bfloat16 instead of float32, so q, k and v reach the attention in
  bfloat16 where numz passes float32 q and k; the RoPE rotates the bfloat16 q and k in float32,
  as ByteDance's does (numz's code would cast its angle table to bfloat16). With
  `NUM_VAE_AUTOCAST=1` the VAE decodes under bfloat16 autocast, its GroupNorms computing in
  float32 and cast back to bfloat16. `vaeac` moves no metric of the table by a printed digit
  (|ΔPSNR-Y| < 0.005 dB): it changes 11–16% of the output's samples, by 0.06–0.11 levels on
  average.
- **Scale:** on anime-clean the variants change the output by 0.11 (`vaeac`) to 0.52 (`parity2`)
  8-bit levels on average (`attn16` 0.23, `norm16` 0.25, `vaes` 0.28, `rope` 0.34, `w32` 0.38,
  `prep` 0.45, `parity` 0.51), another seed by 1.90.
- **The autocast decode costs memory:** it keeps the GroupNorms' float32 outputs; decoding
  anime-clean's 45 frames at 1080p peaks at 37.7 GiB allocated and 56.5 GiB on the device,
  against numz's 35.3 and 45.5.

## Resize kernels

The input (960×540) is upscaled ×2 to the output size before encoding. numz: torchvision bicubic
with `antialias=True` (a = −0.5).

| Variant | Kernel | PSNR-Y | LPIPS | VMAF | ΔE00 lf | Temporal error lf | Temporal error |
|---|---|---|---|---|---|---|---|
| `noaa` | torchvision bicubic, no antialias (a = −0.75) | −0.09..+0.12 (–) | −0.0052..+0.0005 (1 B) | −1.38..+0.10 (–) | +0.010..+0.088 (4 W) | +0.037..+0.149 (4 W) | −0.150..+0.161 (1 W) |
| `spline36` | zimg spline36 | +0.01..+0.16 (–) | −0.0040..−0.0001 (–) | −0.89..+0.77 (–) | −0.009..+0.010 (1 B) | −0.014..+0.051 (1 W) | −0.104..+0.061 (–) |
| `lanczos` | zimg lanczos (3 taps) | −0.02..+0.24 (1 B) | −0.0082..−0.0001 (1 B) | −0.93..+0.79 (–) | −0.004..+0.030 (1 B, 1 W) | −0.004..+0.108 (1 W) | −0.163..+0.110 (1 W) |
| `noaa`, 3 new clips | | −0.27..−0.10 (1 W) | +0.0002..+0.0008 (1 W) | −0.79..−0.29 (1 W) | +0.019..+0.093 (3 W) | +0.078..+0.292 (3 W) | +0.133..+0.279 (2 W) |
| `spline36`, 3 new clips | | −0.20..+0.06 (1 B) | −0.0007..+0.0003 (1 B) | −1.21..+0.29 (1 B) | −0.000..+0.023 (1 W) | −0.030..+0.055 (1 W) | −0.064..+0.061 (1 W) |
| `lanczos`, 3 new clips | | −0.34..−0.02 (–) | −0.0012..−0.0004 (1 B) | −1.57..+0.18 (1 B) | +0.003..+0.046 (2 W) | +0.010..+0.126 (1 W) | +0.002..+0.107 (1 W) |

The kernels change the output by 0.6–1.7 levels (mean |ΔY|) in the windows where they differ
most from the default (review crops). On the animated clips the sharper zimg kernels lean
slightly toward the GT on detail (LPIPS lower on all 4 clips, DISTS on 3), significantly so only
on anime-grain (lanczos: +0.24 dB, LPIPS −0.005, DISTS −0.0023; `lab`: +0.30 dB). On softer d2
input all three kernels gain a little more ([Degradation d2](#degradation-d2)).

On the 3 new clips the picture is mixed:

- **torchvision without antialiasing is worse on all 3** (low-frequency colour +0.02 to +0.09
  ΔE00, low-frequency flicker +0.08 to +0.29), and on live-slow on every metric (PSNR-Y −0.25 dB,
  LPIPS +0.0008, VMAF −0.48, temporal error +0.28); `lab` keeps it (live-slow −0.10 dB, temporal
  error +0.19). 7 clips of 7.
- **The zimg kernels help live-slow and cost live-vfx:** on live-slow (slow, grainy, detailed)
  spline36 gains +0.06 dB, LPIPS −0.0005, VMAF +0.29, lanczos LPIPS −0.0004 and VMAF +0.18 but
  low-frequency colour +0.030; on live-vfx (fast, grainy) both add low-frequency colour error
  (+0.023, +0.046) and flicker (temporal error +0.043, +0.107; low-frequency +0.055, +0.126:
  2 to 7 times its spreads, 1–3% of the values); anime-bright stays within with both.
  After `lab` the live-slow gains remain (+0.05 to +0.08 dB) and the live-vfx costs shrink to
  +0.004 to +0.008 ΔE00 lf and +0.01 to +0.03 temporal error.

numz's antialiased torchvision bicubic stays a sound default. A zimg resize is not worth a CPU
pass through ffmpeg on its own; if the pipeline resizes with zimg anyway, spline36 is the safer of
the two (half lanczos's colour and flicker cost on live-vfx, and no colour cost on live-slow).

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
own; at 720p it reflects 16 ([720p](#720p-no-padding-at-all)).

| Variant | PSNR-Y | LPIPS | VMAF | ΔE00 lf | Temporal error lf | Temporal error | Bottom 16 rows | Rest of the frame |
|---|---|---|---|---|---|---|---|---|
| `black+16`, `none` | +0.06..+0.09 (–) | −0.0070..−0.0010 (1 B) | −0.02..+0.83 (–) | +0.016..+0.152 (3 W) | −0.030..+0.089 (1 W) | −0.054..+0.052 (–) | +0.35..+0.90 (–) | +0.01..+0.07 (–) |
| `black+16`, `lab` | +0.15..+0.21 (–) | −0.0070..−0.0009 (1 B) | +0.29..+1.16 (1 B) | −0.017..+0.018 (1 B) | +0.003..+0.013 (–) | −0.050..−0.024 (–) | +0.45..+0.93 (–) | +0.08..+0.18 (–) |
| `reflect+black+16`, `none` | +0.29..+0.40 (3 B) | −0.0069..−0.0016 (2 B) | −1.03..+1.15 (1 B) | −0.011..+0.149 (1 B, 2 W) | +0.005..+0.090 (2 W) | −0.133..+0.051 (–) | +7.07..+13.38 (3 B); letterbox −5.03 (W) | −0.13..+0.29 (2 B) |
| `reflect+black+16`, `lab` | +0.33..+0.54 (3 B) | −0.0068..−0.0010 (2 B) | −0.45..+1.00 (1 B) | −0.026..+0.003 (1 B) | +0.003..+0.041 (1 W) | −0.134..+0.059 (–) | +2.10..+17.36 (4 B) | −0.01..+0.33 (1 B) |
| `reflect>=8+black+16`, `none`, 3 new clips | +0.10..+0.36 (2 B) | −0.0013..−0.0008 (1 B) | +0.15..+3.85 (1 B) | −0.001..+0.014 (–) | −0.026..+0.037 (1 W) | −0.144..+0.024 (1 B, 1 W) | +0.37..+8.94 (1 B) | +0.08..+0.23 (1 B) |
| `reflect>=8+black+16`, `lab`, 3 new clips | +0.11..+0.69 (2 B) | −0.0014..−0.0007 (1 B) | +0.20..+4.00 (1 B) | −0.031..−0.003 (1 B) | −0.024..+0.010 (–) | −0.088..−0.001 (1 B) | +1.91..+8.09 (3 B) | +0.11..+0.69 (1 B) |

`black+16` on the 3 clips without black areas; `reflect+black+16` on all 4 animated clips;
`reflect>=8+black+16` (the same padding at 1080p) on live-vfx, live-slow and anime-bright.

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
- **The new clips confirm it.** anime-bright, bright to its bottom edge, loses 6.9 dB in its
  bottom band with numz's padding (18.5 against 25.4 for the rest); `reflect>=8+black+16` brings
  the band to 27.4 dB (+8.9) and the frame +0.36 dB, SSIM +0.004. The live-action clips are
  letterboxed, so the reflected rows are their own black bars, 24 black rows in all as on
  anime-grain: live-slow is better on PSNR-Y (+0.10 dB), LPIPS, DISTS, VMAF (+0.31) and temporal
  error; live-vfx stays within its wide spread (PSNR-Y +0.23 dB, VMAF +3.9 against a spread of
  5.5) but for slightly more flicker in `none` (temporal error +0.024, low-frequency +0.037,
  twice its spread), gone after `lab` (−0.001, +0.010). Their bottom rows are bars, black in
  every output (PSNR-Y 64 dB or more, often exact).
- Cost: 16 more rows, +1.5% of the DiT tokens and of the VAE's work at 1080p (138 latent rows
  instead of 136).

### 720p: no padding at all

1280×720 is a multiple of 16: numz pads nothing, so there are no black rows. Same d1 inputs at
`--resolution 720` (×1.33), scored against the GT downscaled to 1280×720 (zimg spline36 on the
16-bit GT, frame-exact; baseline: the input upscaled with Catmull-Rom). `black+16` adds 16 black
rows under the picture (736 rows); `reflect>=8+black+16` reflects 16 rows first (there is
nothing to reflect up to the multiple of 16, so it takes the next one), then adds the 16 black
rows (752 rows); both are trimmed after decoding. One seed: there is no 720p spread, so B / W
only mean that the interval excludes 0; the 1080p spreads give the scale. The default lands at
PSNR-Y 25.2–31.1 dB (bicubic 33.4–43.8).

| Pair | PSNR-Y | LPIPS | VMAF | ΔE00 lf | Temporal error lf | Temporal error | Bottom 16 rows | Rest of the frame |
|---|---|---|---|---|---|---|---|---|
| `black+16` − default, `none` | +0.22..+0.42 (4 B) | −0.0121..−0.0012 (4 B) | +1.83..+6.40 (4 B) | −0.397..−0.068 (3 B) | −0.307..−0.061 (4 B) | −0.554..−0.124 (4 B) | −10.35..−2.54 (4 W) | +0.23..+0.61 (4 B) |
| `black+16` − default, `lab` | +0.03..+0.42 (3 B) | −0.0123..−0.0012 (4 B) | +1.05..+5.43 (4 B) | −0.234..−0.052 (4 B) | −0.153..−0.048 (4 B) | −0.535..−0.110 (4 B) | −12.03..−0.23 (3 W) | +0.19..+0.70 (4 B) |
| `reflect>=8+black+16` − default, `none` | +0.07..+1.03 (4 B) | −0.0301..−0.0039 (4 B) | +0.93..+10.19 (4 B) | −0.248..−0.005 (3 B) | −0.278..−0.008 (3 B) | −1.088..−0.053 (4 B) | −2.12..+2.04 (3 B) | +0.07..+1.02 (4 B) |
| `reflect>=8+black+16` − default, `lab` | +0.11..+1.30 (4 B) | −0.0312..−0.0033 (4 B) | +0.31..+9.23 (4 B) | −0.194..−0.052 (4 B) | −0.182..−0.012 (3 B) | −1.067..−0.049 (4 B) | +0.60..+2.19 (3 B) | +0.11..+1.29 (4 B) |
| `reflect>=8+black+16` − `black+16`, `none` | −0.15..+0.61 (2 B, 1 W) | −0.0180..−0.0001 (3 B) | −1.30..+3.79 (2 B, 2 W) | −0.020..+0.149 (1 B, 3 W) | −0.008..+0.122 (2 W) | −0.534..+0.113 (2 B, 2 W) | +3.78..+8.48 (4 B) | −0.15..+0.41 (2 B, 2 W) |
| `reflect>=8+black+16` − `black+16`, `lab` | −0.12..+0.88 (3 B, 1 W) | −0.0189..+0.0004 (3 B) | −0.73..+3.79 (2 B, 2 W) | −0.028..+0.041 (2 B, 2 W) | −0.029..+0.078 (1 B, 2 W) | −0.532..+0.077 (2 B, 2 W) | +1.45..+12.63 (4 B) | −0.13..+0.59 (2 B, 1 W) |

- **Black rows help where numz adds none:** with `black+16` the rest of the frame gains
  0.23–0.61 dB, VMAF 1.8–6.4, and the frame is steadier (temporal error −0.12 to −0.55,
  low-frequency flicker −0.06 to −0.31), beyond the 1080p spreads for VMAF and low-frequency
  flicker on all 4 clips (anime-clean's VMAF at the edge) and for colour on 3. The anchor is not
  an artefact of 1080p's 8 padded rows.
- **Black right under the picture damages its bottom band** (−2.5 to −10.4 dB on the bottom 16
  rows), as numz's padding does at 1080p.
- **The reflected margin keeps the gain and spares the band:** `reflect>=8+black+16` is better
  than the default on all 4 clips: PSNR-Y +0.07 to +1.03 dB, LPIPS −0.004 to −0.030, VMAF +0.9
  to +10.2, temporal error −0.05 to −1.09, the rest of the frame +0.07 to +1.02 dB, and the
  bottom 16 rows +1.2 to +2.0 dB (anime-grain's, black bars in the GT: −2.1, not significant).
  `lab` gives the same picture (PSNR-Y +0.11 to +1.30 dB, VMAF +0.3 to +9.2).
- **Against `black+16` it repairs the band** (+3.8 to +8.5 dB on all 4 clips) and the whole
  frame is better on anime-clean and anime-dark (PSNR-Y +0.52 and +0.61 dB, VMAF +0.6 and +3.8,
  LPIPS −0.018 and −0.015), a little worse on anime-grain (−0.15 dB, VMAF −0.9: within its 1080p
  spreads) and cartoon-bright (rest of the frame −0.12 dB, VMAF −1.3, ΔE00 lf +0.15,
  low-frequency flicker +0.12; `lab`: PSNR-Y +0.15 dB, VMAF −0.3). The 16 reflected rows give
  back part of plain black's frame-wide gain on two clips, never all of it: the frame stays better
  than the default's on every clip.

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

## VAE decode precision

The VAE decodes in bfloat16, and numz brings the decoded frames to [0, 1] in bfloat16: from 0.5
up, bfloat16 keeps 8 significant bits, steps of 1/256, one 8-bit level, which no 16-bit master
can undo. Smooth bright gradients could band.

### Where the precision goes

From the decoder to the frames `ffv1_out.py` receives (numz `4490bd1`; paths under `src/`, the
CLI at the root):

| Step | Code | dtype | What is lost |
|---|---|---|---|
| VAE weights | the VAE takes the compute dtype (`core/model_configuration.py:1128-1132`); the file's float16 weights are converted on load (`core/model_loader.py:583-584`) | bfloat16 | 3 mantissa bits of every weight |
| Decode | the latent is already bfloat16, so the decoder runs without autocast (`core/infer.py:245-266`) | bfloat16 activations and output | the decoder's arithmetic, and its output in [−1, 1]: steps of 1/256 above 0.5 in magnitude, half an 8-bit level near black and near white once in [0, 1] |
| Phase 3 | `final_video` allocated in the compute dtype (`core/generation_phases.py:879`), the decoded frames cast to it and written (`:1003-1017`) | bfloat16 | nothing more |
| Phase 4 | read back in bfloat16 (`:1239-1245`); `lab`: wavelet step in the frames' dtype (`utils/color_fix.py:280`, its blur kernel `:150`), LAB transfer in float32 (`:299`), cast back (`:361`); then `clamp_(-1, 1).mul_(0.5).add_(0.5)` in place (`core/generation_phases.py:1348`), written back (`:1358-1373`) | bfloat16 | **the coarsest step:** [0, 1] in bfloat16, values from 0.5 up on multiples of 1/256 (one 8-bit level), [0.25, 0.5) on 1/512 |
| CLI | `.to(torch.float32)` (`inference_cli.py:1010`) | float32 | nothing (exact), nothing to recover either |
| Writer | `ffv1_out.py`: rint(x × 65535) | 16 bits | nothing, but a channel holds at most 129 codes from 0.5 up |

### Variants

- `out32` (`NUM_OUT32=1`): numz's bfloat16 decode, then float32 up to the writer (`final_video`
  and Phase 4): the masters get the decoder's own output, without the [0, 1] rounding.
- `dec16` (`NUM_DECODE=fp16`): the decoder in float16, with the VAE file's own float16 weights,
  without autocast, float32 after it; the encode stays numz's. Forward hooks count the non-finite
  values of every decoder module's output.
- `dec32` (`NUM_DECODE=fp32`): float32 weights (the file's float16 values, exactly) and
  activations, TF32 off (cuDNN's included); `dec32tf` with TF32.

All five clips, seed 42, `lab` rendered too: the four above, and anime-sky, a bright sky with
clouds over 45 frames (73% of each frame bright and smooth: luma ≥ 140, 15×15 local deviation
≤ 2 levels), with its default at three seeds.

| | numz (bfloat16) | `out32` | `dec16` | `dec32` | `dec32tf` |
|---|---|---|---|---|---|
| Distinct 16-bit codes from 0.5 up, per channel | 75–129 | 549–1,044 | 4,229–6,056 | 15,979–32,768 | 15,981–32,768 |
| Share of those samples on the bfloat16 grid | 100% | 11–45% | 1.4–6.2% | 0.4% (chance) | 0.4% |
| CAMBI of the output / added to the GT | ≤ 0.004 / 0 | ≤ 0.001 / 0 | ≤ 0.004 / 0 | ≤ 0.004 / 0 | ≤ 0.004 / 0 |
| Mean \|ΔY\| to numz's (anime-clean, anime-sky) | – | 0.13–0.23 (max 0.5) | 0.26–0.31 (max 4.2–12.3) | 0.26–0.30 | 0.26–0.31 |
| Decode time, anime-clean, single runs (clocks drift between them) | 113 s | 116 s | 72 s (173 s with the hooks) | 462 s | 169 s |
| Decode peak: allocated / on the device | 35.3 / 45.5 GiB | 35.3 / 45.5 GiB | 35.3 / 45.5 GiB (47.7 / 78.5 with the hooks) | 70.0 / 88.9 GiB | 70.0 / 88.8 GiB |
| Non-finite values, largest \|activation\| | – | – | 0; 5,616–19,088 | – | – |

The codes are counted over 45 frames, `none`; numz's `lab` rendering has the same 129 at most
(it is cast back to bfloat16 before the normalisation); anime-dark barely reaches 0.5 (75–114).

Paired difference to the default, `none` (5 clips; the last row: `dec16` on the 3 new clips):

| Variant | PSNR-Y | LPIPS | DISTS | VMAF | ΔE00 lf | Temporal error |
|---|---|---|---|---|---|---|
| `out32` | +0.00..+0.01 (–) | −0.0000..+0.0011 (–) | −0.0022..+0.0004 (1 B) | +0.00..+0.13 (–) | −0.024..−0.008 (1 B) | −0.026..−0.001 (1 B) |
| `dec16` | +0.03..+0.12 (1 B) | +0.0000..+0.0014 (1 W) | −0.0025..+0.0011 (1 B) | −0.00..+0.12 (–) | −0.086..−0.046 (5 B) | −0.039..−0.012 (1 B) |
| `dec32` | +0.03..+0.11 (1 B) | +0.0000..+0.0014 (1 W) | −0.0025..+0.0010 (1 B) | −0.01..+0.13 (–) | −0.086..−0.047 (5 B) | −0.038..−0.015 (1 B) |
| `dec32tf` | +0.03..+0.12 (1 B) | +0.0000..+0.0014 (1 W) | −0.0025..+0.0010 (1 B) | −0.01..+0.13 (–) | −0.086..−0.047 (5 B) | −0.038..−0.015 (1 B) |
| `dec16`, 3 new clips | +0.02..+0.06 (1 B) | +0.0003..+0.0024 (2 W) | −0.0003..+0.0009 (1 W) | −0.01..+0.03 (–) | −0.092..−0.047 (3 B) | −0.055..−0.022 (1 B) |

- **numz's masters hold one 8-bit level of precision near white:** at most 129 codes per channel
  from 0.5 up, all on the bfloat16 grid. `out32` removes only the last rounding: the decoder's
  bfloat16 output reaches 549–1,044 codes there, the same sets on every clip (the format sets
  them, not the content). float16 decoding reaches 4,229–6,056, float32 every code the content
  spans.
- **No banding measured, in any variant.** CAMBI, as `fr_metrics.py` computes it (10-bit, as
  sptenc does), gives every output 0.004 at most and adds nothing to the GT's (0.02–0.22), the
  sky clip included. It does see steps of one 8-bit level on a clean gradient (a grey ramp
  0.55 → 0.95 rounded to 8 bits: 0.70); the model's rendering, several levels from the GT
  everywhere, dithers the bfloat16 steps away (the outputs score below the GT itself).
- **float16 or float32 decoding brings low-frequency colour closer to the GT on all 5 clips**
  (ΔE00 lf −0.05 to −0.09 against values of 1.8–2.6, beyond spreads of 0.001–0.034), PSNR-Y
  +0.03 to +0.12 dB (beyond the spread on anime-dark only); LPIPS +0.0014 on the sky clip (its
  spread); the rest within. After `lab` the verdicts left are small: ΔE00 lf −0.01 to −0.02 and
  SSIM +0.001 on 2 clips, LPIPS +0.002 on the sky. `out32` alone stays within but for single
  small verdicts. `dec16` does the same on the 3 new clips (ΔE00 lf −0.05 to −0.09, beyond
  their spreads on all 3): 8 clips of 8. Its cost there is a little LPIPS, +0.0003 on live-slow
  (whose spread is 0.0001) and +0.0024 on anime-bright (3% of its value), and DISTS +0.0008 on
  live-slow; after `lab`, LPIPS +0.0007 and DISTS +0.0008 on live-slow, DISTS −0.0028 on
  anime-bright, ΔE00 lf −0.01 on both live-action clips.
- **float16 and float32 give the same picture:** 0.02 levels apart on average (TF32: 0.006–0.009
  from strict float32), where they move numz's output by 0.26–0.31 levels and another seed moves
  it by 1.05 (anime-sky).
- **Cost:** float32 doubles the decode's memory (70.0 GiB allocated, 88.9 on the device for 45
  frames at 1080p, near this 96 GB card's limit) and takes 3.3–4 times as long without TF32
  (294–462 s against 80–116 s in bfloat16 for the same clips), 1.4–1.7 times with it. float16
  takes the same memory as bfloat16 and slightly more time: back to back in one lock hold, at the
  same clocks (`decode_resume.py dtype-ab`, 33 frames at 1080p, three alternating runs each, the
  first discarded), decode 54.7 s against 52.4 s (+4.4%), tiled decode (1024:128) 66.2 against
  63.3 s (+4.6%), encode 25.7 against 25.0 s (+2.9%), every run at the 600 W limit. The 72 s
  single run in the table came from a faster clock session, and the overflow hooks add passes
  over every module output. `out32` costs host memory only: the frames in float32, 1.1 GiB for
  45 frames at 1080p.
- **No float16 overflow:** no non-finite value in any of the 1,529 module outputs of a decode, on
  any clip; the largest activations are in the last up block (`up_blocks.3`): 5,616 to 19,088,
  the sky's 19,088 3.4 times under float16's 65,504. Brighter content runs closer to the limit.
  The 3 new clips stay under the sky: 10,416 (live-slow), 11,936 (live-vfx), 13,464
  (anime-bright, mean Y 174), no non-finite value.
- **Recommended:** keep everything after the decoder in float32 (free), and decode in float16
  with a non-finite check that redoes the batch in float32 if it trips: float32's output for
  3–5% more decode time than bfloat16. Not for banding, which nothing here shows, but for
  colour; numz's bfloat16 decode with float32 after it remains a sound default without the check.

## The master's chroma: 4:2:0 kernels and zscale's slices

seedvr2x's default master is FFV1 `yuv420p10le`, converted from the float RGB frames as
[`ffv1_out.py`](../scripts/ffv1_out.py) does it (zscale: BT.709 matrix, limited range, chroma
sited left, no dither, zscale's default chroma kernel, bilinear;
[DESIGN.md](../../seedvr2x/DESIGN.md#output)). Every default output goes through that chroma
downsampling. [`chroma_kernels.py`](../scripts/chroma_kernels.py) sends the 16-bit RGB output
of the 8 clips (numz's default with `lab`, seed 42) and their GTs through `yuv420p10le` with 6
chroma kernels and back to 16-bit RGB with one upsampler, seedvr2x's reader (zscale,
Catmull-Rom), then scores each round trip against the GT: PSNR on Y', Cb and Cr (BT.709, 8-bit
scale), CIEDE2000 after blurs of 0 to 4 px, CIEDE2000 on colour edges (the 10% of pixels with
the GT's largest chroma gradient), and ringing (Cb or Cr beyond the GT's local 5×5 range by more
than 2 levels). A `yuv444p10le` round trip is the control: 10 bits, no subsampling.

Checks: the bilinear master is ffv1_out.py's own conversion, 45 of 45 frames by framemd5 on all
16 files (an explicit `f=bilinear` too, same tags); a synthetic frame confirms the siting (left
across, centred down, the round trip symmetric); the σ = 4 ΔE00 equals fr_metrics' ΔE00 lf
exactly; no round trip shifts the mean by more than 0.03 level (point 0.10).

Model output, means over the 8 clips (wins: clips where the kernel beats bilinear on ΔE00):

| Round trip | PSNR-Cb | PSNR-Cr | ΔE00 | ΔE00 σ = 4 | ΔE00 on colour edges | Ringing | Wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| none (the output itself) | 42.61 | 42.02 | 2.319 | 1.190 | 5.35 | 14.0% | |
| bilinear (zscale's default) | **43.68** | **43.06** | **2.161** | **1.174** | **4.86** | **11.2%** | |
| Catmull-Rom | 43.35 | 42.75 | 2.207 | 1.181 | 5.00 | 12.3% | 0/8 |
| spline16 | 43.29 | 42.70 | 2.214 | 1.182 | 5.02 | 12.5% | 0/8 |
| spline36 | 43.24 | 42.65 | 2.221 | 1.181 | 5.05 | 12.5% | 0/8 |
| lanczos (3 taps) | 43.20 | 42.60 | 2.227 | 1.181 | 5.07 | 12.7% | 0/8 |
| point (aliasing control) | 42.69 | 42.13 | 2.293 | 1.205 | 5.33 | 13.4% | 0/8 |
| yuv444p10le (no subsampling) | 42.60 | 42.01 | 2.326 | 1.184 | 5.36 | 13.0% | 0/8 |

- **Bilinear wins on the model's output:** every sharper kernel loses 0.3–0.5 dB of PSNR-Cb and
  PSNR-Cr (one clip of 8 on Cr goes the other way), 0.05–0.07 of ΔE00, 0.14–0.21 on colour edges,
  and rings more, on anime and live action alike.
- **The 4:2:0 round trip brings the output closer to the GT:** PSNR-Cb +1.1 dB and colour edges
  −0.49 against the unconverted output, and no subsampling (`yuv444p10le`) is no better than the
  output itself. The GT's chroma comes from 4:2:0 sources; the model's chroma detail beyond that
  resolution is not in the GT, and a smooth 4:2:0 kernel takes it out.
- **The GT's own round trip prefers sharper kernels** (lanczos: ΔE00 0.118 against 0.188, colour
  edges 0.25 against 0.50, 8 of 8), the format's cost on perfect content. That GT's chroma was
  itself upsampled from 4:2:0 with zscale's default, bilinear
  ([Ground truth](#ground-truth-inputs-and-runs)), which a sharper
  downsampler partly undoes: against the source's own 4:2:0 samples, the sharper kernels land
  closer for the GT (16 of 16 planes) and further for the model's output (15 of 16). Both amounts
  are tiny next to the model's own error.
- **The usual metrics don't move:** fr_metrics on the round trips stays within 0.012 dB of
  PSNR-Y, 0.24 of VMAF and 0.007 of LPIPS of the unconverted output, CAMBI 0; the conversion lowers
  ΔE00 lf by 0.016 on average (8 of 8 clips).
- **Cost:** zscale alone takes 2.5 ms per 1080p frame for the bilinear conversion on one slice
  (lanczos 2.6, the Catmull-Rom read 2.3), against about 4 s of GPU time per frame.

**zscale's slices change the bytes.** ffmpeg runs zscale in `-filter_threads` slices, by
default one per CPU the process may use, and each slice is a separate zimg graph on its own rows:
the vertical chroma filter stops at the slice's edge. On the first 3 frames of anime-clean's
output, each count against one slice:

| Chain | 2 slices | 4 | 8 | 16 or more |
|---|---|---|---|---|
| RGB → `yuv420p10le` (the master): Cb / Cr samples that differ, largest difference (10-bit codes) | 0.12% / 0.11%, 6 | 0.33% / 0.29%, 10 | 0.79% / 0.73%, 10 | 1.70% / 1.58%, 13 |
| `yuv420p10le` → RGB (Catmull-Rom): samples that differ (16-bit codes) | 0 | 99.97%, up to 146 | same | same |

- Every slice count from 1 to 16 gives a different master (framemd5; 16 and 48 agree), its luma
  untouched. Against a float64 Catmull-Rom reference, the 10-bit read is exact on 1 to 3 slices
  (at most 1 sixteen-bit code off) and off from 4 on (rms 35, up to 146 codes, 0.57 level); an
  8-bit 4:2:0 source reads exactly at any count. Why 10-bit and not 8-bit is not known (ffmpeg
  n9.0.2).
- **`threads=1` on zscale fixes it:** with `-filter_threads 48`, both chains then give the
  one-slice bytes (framemd5). `threads` is libavfilter's generic per-filter option ("Allowed
  number of threads", beside `enable` and `thread_type`), not one of zscale's own: `ffmpeg -h
  filter=zscale` doesn't list it, `ffmpeg -h full` does among the generic filter options, and
  any build takes it. ffv1_out.py and fr_clips.py set it since 2026-10-05: on the 48-CPU box
  ffv1_out.py's master then equals the one-slice one, 45 of 45 frames.
- **The measurements so far are unaffected:** fr_clips.py's GT conversions of these 8-bit
  sources, its d1 downscale and its bicubic baselines give the same bytes on 1 and 48 slices (the
  stored files equal the one-slice output), and every score here reads 16-bit RGB masters, not
  `yuv420p10le` ones. A GT made from a 10-bit source (4K HEVC) would have gone through the
  faulty read.

So: keep bilinear for the master's chroma, and run every zscale conversion on one slice
(`threads=1`), the master's and the reader's alike: with ffmpeg's default, a master and its
checksum change with the machine's CPU count, and a 10-bit source reads off the exact
conversion. The review sheets (`chroma-<clip>.png`, [Visual review](#visual-review)) show 3
crops per clip where the GT's colour edges are sharpest.

## Live action and a bright anime

Three clips from three more sources ([Clips](#clips)): live-vfx and live-slow, letterboxed live
action with film grain (a fast VFX shot, a slow film), and anime-bright, line art on white. Same
d1 protocol: the default at seeds 42, 43 and 1234; at seed 42 `parity`, the three resize
kernels, `reflect>=8+black+16` and `dec16`, each with its `lab` rendering. Their rows are in the
tables above ([Seed bands](#seed-bands), [Results](#results), [Resize kernels](#resize-kernels),
[reflect+black+16](#reflectblack16), [VAE decode precision](#variants)). Per clip, `none`, the
metrics beyond the clip's seed spread (B better, W worse; the rest within):

| Variant | live-vfx | live-slow | anime-bright |
|---|---|---|---|
| `parity` | B: ΔE00 lf −0.088, temporal error −0.166 (low-frequency −0.181) | B: PSNR-Y +0.12 dB, VMAF +0.18, LPIPS −0.0001, ΔE00 lf −0.046, temporal error −0.074 | B: ΔE00 lf −0.072 |
| `reflect>=8+black+16` | W: temporal error +0.024 (low-frequency +0.037) | B: PSNR-Y +0.10 dB, LPIPS −0.0008, DISTS −0.0006, VMAF +0.31, temporal error −0.073 | B: PSNR-Y +0.36 dB, SSIM +0.004, bottom 16 rows +8.94 dB |
| `dec16` | B: ΔE00 lf −0.047, temporal error −0.023 | B: PSNR-Y +0.06 dB, SSIM +0.004, ΔE00 lf −0.092; W: LPIPS +0.0003, DISTS +0.0008 | B: ΔE00 lf −0.048; W: LPIPS +0.0024 |
| `noaa` | W: ΔE00 lf +0.039, temporal error +0.185 (low-frequency +0.205) | W on every metric: PSNR-Y −0.25 dB, LPIPS +0.0008, DISTS +0.0006, VMAF −0.48, ΔE00 lf +0.093, temporal error +0.279 | W: ΔE00 lf +0.019, low-frequency flicker +0.078 |
| `spline36` | W: ΔE00 lf +0.023, temporal error +0.043 (low-frequency +0.055) | B: PSNR-Y +0.06 dB, LPIPS −0.0005, VMAF +0.29 | within |
| `lanczos` | W: ΔE00 lf +0.046, temporal error +0.107 (low-frequency +0.126) | B: LPIPS −0.0004, VMAF +0.18; W: ΔE00 lf +0.030 | within |

The live-action clips' bottom rows are letterbox bars (PSNR-Y 64 dB or more in every output,
often exact), left out.
After `lab` the signs hold, smaller: anime-bright keeps only `reflect>=8+black+16`'s gains (+0.34
dB, bottom rows +8.1 dB) and a DISTS −0.003 for `dec16`; on live-vfx `reflect>=8+black+16`'s
flicker is gone (ΔE00 lf −0.003, B) and the zimg kernels' costs shrink to +0.004/+0.008 ΔE00 lf
and +0.01 to +0.03 temporal error; `noaa` stays worse on every metric of live-slow.

- **The spreads, not the means, decide the verdicts here.** live-slow's seeds agree within
  0.03 dB, 0.13 VMAF and 0.0001 LPIPS, so changes of a few hundredths of a dB pass; live-vfx's
  flickering light makes its VMAF move by 5.5 points from seed to seed (PSNR-Y 0.51 dB, 1.11
  after `lab`), and only colour and temporal error, whose spreads stay at 0.02, show anything.
- **The recommendations hold but one** ([In short](#numerics-and-input-preparation-against-a-ground-truth)):
  the padding, a float16 decode and float32 after the decoder are confirmed; ByteDance's numerics
  gain a little on these clips; the zimg kernels are no longer safe everywhere (live-vfx), so
  spline36 only, and only where zimg resizes anyway; torchvision's bicubic without antialiasing
  stays the one to avoid.
- **`lab` hurts anime-bright's perceptual metrics** (LPIPS +0.019, DISTS +0.012, VMAF −3.1 against
  the default) while it fixes the mean level: over the clip's flat white areas (13% of the frame)
  the default is 3.3 levels too bright, `lab` +0.1, with no drift through the batch (−0.08 from
  the first frame to the last, so not the tie order of `lab`'s histogram matching,
  [quality.md](quality.md)) and their local noise 0.71 levels against 0.54. The cause of the
  LPIPS cost is not established.

### Grain

The measure (`fr_clips.py grain`): the standard deviation of the full-range BT.709 luma minus its
2-pixel Gaussian blur, over the 30% of the picture where the blurred image's gradient is lowest
(and 16 < Y < 235), in 8-bit levels; per frame, the median over the frames (25–75% in brackets).
Black bars (rows or columns whose mean Y stays below 20) and the 8 pixels next to them, or next
to the frame's edge where there is no bar, are left out. A source is surveyed on 12 frames over
the middle 80% of its duration.

Corrected on 2026-10-05: the first version had no margin at the frame's own edges, where its
blur and gradient see a mirrored border. A dark edge line then counts among the smoothest areas
and dominates the measure: live-slow read 3.96 instead of 1.59 (its outermost 9 columns, 0.7% of
the mask, held 83% of the squared residual) and its film 1.64 instead of 1.02 (a 3-column fade to
black at both sides). live-vfx and its film have clean edges and keep their figures.

| | live-vfx | live-slow |
|---|---|---|
| source, 12 frames over the film | 1.65 | 1.02 |
| GT (the clip) | 1.75 (1.69–1.83) | 1.59 (1.52–1.70) |
| bicubic baseline (the d1 input upscaled) | 0.45 (0.41–0.50) | 1.29 (1.24–1.35) |
| default, 3 seeds | 0.93–0.97 | 1.86–1.89 |
| default + `lab`, 3 seeds | 0.92–0.95 | 1.95–1.97 |
| every variant, `none` / `lab` | 0.90–0.93 / 0.89–0.93 | 1.87–1.93 / 1.95–1.98 |

- **On live-vfx the model gives back about half of the grain.** The d1 input keeps a quarter of
  it (0.45 against 1.75); the output has 0.93–0.97, whatever the variant, `lab` or not. What it
  puts there behaves like grain, not like a fixed texture: on consecutive frames registered by
  phase correlation the residual changes at least as much as two independent fields would (1.19
  for the output, 1.80 for the GT, against 0.93 and 1.75 within a frame). At the finest scale
  (1-pixel blur) the output keeps less: 0.61–0.64 against 1.40 (the input 0.21).
- **On live-slow the input keeps most of the grain, and the output a little more than the GT.**
  The d1 input keeps 1.29 of the GT's 1.59; the output has 1.86–1.89 (+17 to +19%; with `lab`
  1.95–1.97), and the smoothest 10% reads the same way (GT 1.05, output 1.26–1.28). At the
  1-pixel scale it matches the GT (0.71–0.72 against 0.76; the input 0.41).
- So the output's grain follows its input: from a quarter of the source's grain (live-vfx) the
  model gives back about half; from most of it (live-slow) it draws a little more than the source
  at the 2-pixel scale and as much at 1 pixel. None of the switches measured here changes that
  (±0.05); keeping more of a grainy source's grain would need its own step (grain synthesis, or a
  lighter degradation of the input than d1).
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
| `dec32sky-anime-sky-f44-x136-y688.png`, `dec32-anime-sky-f44-x136-y808.png` | GT \| numz's bfloat16 decode \| float32 decode, on the sky, where the two decodes differ most (rows 0–959, and anywhere): 0.31–0.32 levels apart on average, both 7.4–9.8 levels from the GT; `.stretch.png`: the same crops with the GT window's 1st–99th luma percentiles stretched to the full range (one 8-bit level becomes 2.7–4.4), to look for the bfloat16 steps; `.diff.png`: bfloat16 − float32 on luma, ×16 around grey |
| `dec32lab-anime-sky-f44-x80-y808.png` | the same after `lab` (0.28 levels apart, 1.8 from the GT) |
| `chroma-<clip>.png` | the master's chroma: 3 windows of 96×96 px per clip at 4× (nearest), where the GT's colour edges are sharpest: GT \| the output \| its bilinear, spline36 and lanczos round trips through `yuv420p10le`, each with its ΔE00 to the GT over the window (cartoon-bright: bilinear the lowest of the three in all 3 windows) |
| `latent-cartoon-bright-f34-36-38-x0-y520.png` | the latent grid: GT \| bicubic \| default with `lab` (columns), frames 34, 36, 38 (rows), 640×400 px at x 0, y 520, 1:1: ghosts in 34 and 38 (place 1 of their latent group), none in 36 (place 3) |

## Caveats

- **Eight clips**, 45 frames each, from 1080p sources: four animated ones with every variant, the
  sky for the decode precision, two live-action films and a bright anime with a subset; 4K sources
  and other content are not measured. anime-grain and both live-action clips are letterboxed,
  which changes the padding results. Grain has one measure (high-pass residual in the smoothest
  30% of the picture), which also counts fine texture there.
- **One model and one size:** 7B fp16 at 1080p (one 720p test), `flash_attn_2`; the 3B, fp8 and
  GGUF weights and 4K are not covered (`NUM_NORM=bf16` patches the 3B's norms too, unmeasured).
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
- **The chroma kernels were scored on one seed** (42, with `lab`) at 1080p, against GTs whose
  chroma is itself 4:2:0 upsampled with bilinear: the model's chroma detail beyond 4:2:0 can't be
  checked against any of these sources. The conversion is deterministic once on one slice;
  zscale's slice behaviour was seen with ffmpeg n9.0.2 and may change with the version.
- **The latent grid was measured on single 45-frame batches** at 1080p from per-frame scores
  already computed; frames 1–44 (11 groups) per clip, no 4K, no other weights.
- **Banding has one measure, CAMBI,** whole-frame (on the sky clip, mostly bright gradients); it
  ignores steps of more than a few 10-bit levels (a ramp rounded to 7 or 6 bits scores 0). The
  float16 headroom (3.4 times on the brightest clip) is for these five SDR clips; brighter or
  more saturated content runs closer to 65,504. Decode times come from a power-capped GPU.

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
#   NUM_VAE_WEIGHTS=/path/to/ema_vae.pth, NUM_NORM=bf16, NUM_VAE_AUTOCAST=1, NUM_ATTN=fp16,
#   NUM_RESIZE=tv-noaa|zimg-spline36|zimg-lanczos (NUM_CHECK=1 logs what each changes),
#   NUM_PAD=reflect|replicate|grey|black+16|reflect+black+16|reflect>=8+black+16,
#   NUM_OUT32=1, NUM_DECODE=fp16|fp32 (with NUM_TF32=1: dec32tf; NUM_DECODE_HOOKS=0: fp16 without
#   the overflow hooks, for its timing);
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
python3 $S/numerics_patch.py --selftest     # zimg resize, 8-bit recovery, padding, norms, decode (CPU)
# decode precision: per channel, a histogram of the master's 16-bit codes (ffmpeg -pix_fmt rgb48le):
#   the codes used from 32768 up, and the share of those samples (below 65535) on rint(65535 k / 256),
#   k = 128..255; CAMBI: the JSONs' per-frame cambi_out and cambi_added. The sky clip: fr_clips.py scan
#   of a source, its calmest bright windows ranked by bright smooth area, then make --degrade d1 --crop
# long sources (the live-action clips): scan in segments (--ss, --duration), keep windows with every
#   scene score < 3, no fade, the most motion per shot; scan's indices come from timestamps, so confirm
#   the window frame-exactly before make (one source's timestamps ran 44 frames ahead of the decode
#   order 30 minutes in):
python3 $S/fr_clips.py scores /path/to/source.mkv --first FIRST_FRAME-10 --last FIRST_FRAME+54
# grain: every frame of an RGB file, or 12 frames over a source's middle 80% (seeking; taller than
#   1080 rows: area-downscaled to 1080); --sigma 1 and --share 10 for the other scales
python3 $S/fr_clips.py grain $C/live-vfx.gt.mkv --json grain-live-vfx-gt.json
python3 $S/fr_clips.py grain /path/to/source.mkv
# the master's chroma (CPU): per clip, round trips of the output (MODEL = its lab master) and of the GT,
#   scores, then one fr_metrics call on the model's round trips (every kernel; see its docstring)
python3 $S/chroma_kernels.py siting --json siting.json
python3 $S/chroma_kernels.py convert --clip anime-clean --model MODEL.mkv --gt $C/anime-clean.gt.mkv \
  --dir rt --filter-threads 1
python3 $S/chroma_kernels.py score --clip anime-clean --model MODEL.mkv --gt $C/anime-clean.gt.mkv \
  --dir rt --json scores/anime-clean.json
taskset -c 0 python3 $S/chroma_kernels.py proof --src MODEL.mkv \
  --yuv rt/anime-clean/model.bilinear.yuv.mkv --work proof --json proof/anime-clean.json
python3 $S/chroma_kernels.py srcchroma --clip anime-clean --src $C/anime-clean.src.mkv --dir rt \
  --json srcchroma/anime-clean.json
python3 $S/chroma_kernels.py review --clip anime-clean --model MODEL.mkv --gt $C/anime-clean.gt.mkv \
  --dir rt --out review
python3 $S/chroma_kernels.py threads --src MODEL.mkv --json threads.json   # slices against one
python3 $S/chroma_kernels.py upcheck --yuv rt/anime-clean/model.bilinear.yuv.mkv --json up.json
python3 $S/chroma_kernels.py gtcheck --gt $C/anime-clean.gt.mkv --src $C/anime-clean.src.mkv \
  --json gtcheck.json
python3 $S/chroma_kernels.py timing --src MODEL.mkv --json timing.json
python3 $S/chroma_kernels.py summary scores/*.json --frm frm --q1 m/d1-lab --proof proof/*.json \
  --siting siting.json --threads-json threads.json --gtcheck gtcheck.json --upcheck up.json \
  --srcchroma srcchroma/*.json --timing timing.json --md summary.md
# zscale's slices, standalone (ffmpeg + zimg only, e.g. for an FFmpeg report): one testsrc2 frame
DOWN="zscale=rin=full:pin=709:tin=709:m=709:r=limited:p=709:t=709:d=none:c=left,format=yuv420p10le"
UP="zscale=min=709:rin=limited:cin=left:pin=709:tin=709:m=gbr:r=full:p=709:t=709:d=none:f=bicubic"
UP="$UP:param_a=0:param_b=0.5,format=gbrp16le"
ffmpeg -f lavfi -i testsrc2=s=1920x1080:r=24:d=1 -frames:v 1 -vf format=gbrp16le -c:v ffv1 rgb.mkv
for t in 1 2 4 16; do ffmpeg -i rgb.mkv -filter_threads $t -vf "$DOWN" -f framemd5 - | tail -1; done
#   4 different hashes (n9.0.2); with -vf "zscale=threads=1:${DOWN#zscale=}" at 16: the 1-slice hash
ffmpeg -i rgb.mkv -filter_threads 1 -vf "$DOWN" -c:v ffv1 yuv10.mkv
for t in 1 3 4 16; do ffmpeg -i yuv10.mkv -filter_threads $t -vf "$UP" -f framemd5 - | tail -1; done
#   1 = 3, 4 = 16, the two differ; threads=1 restores 1's; an 8-bit yuv420p copy reads alike at any count
# the latent grid: per-frame scores by place in the 4-frame groups, against each clip's bicubic baseline
python3 $S/cut_metrics.py phase m/d1-none/*-d1.def.s{42,43,1234}.json \
  $(for f in m/d1-lab/*-d1.bicubic.s0.json; do echo --base $f; done) \
  --metrics psnr_y,ssim_y,lpips,dists,vmaf,de00_lf
# its review crop: frames 34, 36, 38 of cartoon-bright, GT | bicubic | default with lab
F="select='eq(n\,34)+eq(n\,36)+eq(n\,38)',crop=640:400:0:520,format=rgb24"
ffmpeg -i $C/cartoon-bright.gt.mkv -i $C/cartoon-bright.d1.bicubic.mkv -i LAB_MASTER.mkv \
  -filter_complex "[0]$F[a];[1]$F[b];[2]$F[c];[a][b][c]hstack=3,tile=1x3" \
  -frames:v 1 -update 1 latent-cartoon-bright-f34-36-38-x0-y520.png
```

<details>
<summary>Run names (7B fp16, <code>flash_attn_2</code>, one batch of 45 frames, <code>--color_correction
none</code> plus the <code>lab</code> rendering of the same run)</summary>

`q1-<clip>-<degradation>-<variant>-s<seed>`, clips `anime-clean`, `anime-grain`, `anime-dark`,
`cartoon-bright`. d1, 1080p: `def` (seeds 42, 43, 1234), `parity`, `rope`, `vaes`, `prep`, `w32`,
`norm16`, `vaeac`, `parity2`, `attn16`, `noaa`, `spline36`, `lanczos`, `reflect`, `replicate`,
`crop` (1072 rows), seed 42; `reflect` also at 43 and 1234 on anime-clean and cartoon-bright;
`grey`, `black16`, `reflblack16` on anime-clean, anime-dark and cartoon-bright, `reflblack16`
also on anime-grain; `defcheck`, `defcheck2`, `defcheck3` (anime-clean, the default re-run,
compared bit for bit). Decode precision: `out32`, `dec16`, `dec32`, `dec32tf` on the four clips
and `anime-sky` (with its `def` at 42, 43 and 1234), `dec16nh` (anime-clean, float16 without the
hooks: bit-identical to `dec16`, for the timing). Live action and a bright anime: `live-vfx`,
`live-slow`, `anime-bright` at d1, 1080p: `def` (42, 43, 1234), `refl8black16`
(`NUM_PAD=reflect>=8+black+16`, at 1080p the padding of `reflblack16`), `parity`, `noaa`,
`spline36`, `lanczos`, `dec16`. `ident`: `q1-<clip>-gt-ident-s42`, the 8-bit GT as input. 720p:
`q1-<clip>-d1-720-{def,black16,refl8black16}-s42`. d2, anime-clean and anime-grain: `def` (42,
43, 1234), `reflblack16`, `lanczos`, `noaa`, `spline36`. Metric labels:
`<clip>-d1`, `<clip>-d1-crop` (rows 4–1075), `<clip>-d1-720`, `<clip>-d2`; `lab` renderings as
`<variant>+lab`, the baselines as `bicubic`.

</details>

# Colour correction beyond numz's `lab`

> Status: **steps 0 to 3 measured; step 4, the eyes, pending.** Step 0 (the diagnosis) ran on the
> CPU from [numerics.md](numerics.md)'s masters; steps 1 to 3 on dumps of 7B fp16 runs (1080p,
> ×1.5 to ×4, the heavier degradation, VAE tiles, 4K), scored on the CPU; step 4's crops are made.
> The study answers "Beyond numz's `lab`" in
> [DESIGN.md](../../seedvr2x/DESIGN.md#beyond-numzs-lab); its decision brief is at the end.

In short:

- **Recommended: a split in Y'CbCr without numz's histogram step.** Lightness from the input
  below 6.5 px, colour below 3.2 px at ×2, the rest from the model: `split:ycc:4:3`. Against
  `lab` it lowers ΔE00 at every blur scale, raises PSNR-Y and lowers LPIPS and flicker, on most
  or all clips, at 1080p (mild and heavier degradation), at 4K and through the default master
  (VMAF +3.7 there); it leaves half of `lab`'s VAE tile drift; it costs 23 ms per 4K frame
  against 105; and with no statistics pooled over the shot, the decode can stream. DISTS is
  mixed: within on half the clips, worse by about 0.003 on compressed inputs.
- **The colour scale follows the upscale factor, the lightness scale doesn't:** colour comes from
  the input below about 1.6 source pixels, the input's own colour resolution (3 à-trous stages at
  ×1.5 and ×2, 4 at ×3 and ×4); colour finer than that costs LPIPS and DISTS, coarser costs
  colour fringes. Lightness at 6.5 px works at every factor tested, ×1.5 to ×4.
- **numz's histogram step is what costs `lab` colour** (step 0, confirmed): matching a\* and b\*
  to the blurred input's distribution moves colour at every scale. Its L\* blend only helps a
  split at 13 px, against flicker; at 6.5 px it adds nothing.
- **Lightness at 3.2 px too** (`ycc:3:3`) gives the most fidelity before detail goes (PSNR-Y +1.7
  dB, flicker −1.13 against `ycc:4:3`'s −0.44, a quarter of `lab`'s tile drift) at the price of
  faint fine texture (a sky loses 15% of its Laplacian variance). The crops put both beside `lab`
  for the eyes.
- **4K:** the correction closes the colour part of the gap to bicubic (with `ycc:3:3`,
  low-frequency ΔE00 below bicubic's on 4 of 4 shots). The rest, 2.6–7.4 dB of PSNR-Y, 10–20
  points of VMAF, DISTS on 4 of 4, is the model's 4K rendering, not colour.
- **A float16 decode adds nothing after the correction** (within the seed band): numz's
  bfloat16 decode can stay.

## Step 0: the diagnosis

CPU only, on numerics.md's masters (indicative: an oracle stands in for steps 1–3's exact dumps).
Its findings (10 clips: 8 at ×2 with the mild degradation d1, 2 with the heavier d2; numz's `lab`
at 3 seeds, against the ground truth (GT) and against the input's own Catmull-Rom upscale):

- **`lab` leaves 1.2 to 3.1 times the input's own colour error at every scale**, from the
  unblurred pixel to a 16 px blur. On animation the excess grows toward the coarse scales
  (cartoon-bright: 1.8× unblurred, 3.1× after a 16 px blur), which the wavelet split is meant
  to take from the input.
- **It is mostly colour, not lightness:** after a 4 px blur, the chroma and hue terms hold 69–84%
  of `lab`'s ΔE00² on 9 of 10 clips. The exception is a slow live-action clip, where lightness
  holds 62%.
- **The input is closer to the GT than the model in every band**, from finer than 1 px to coarser
  than 32 px, in lightness and in colour, on all 10 clips. Fidelity alone would take everything
  from the input. Where to split is therefore a trade-off with the model's detail, for the
  perceptual scores and the eyes to settle.
- **numz's histogram step causes `lab`'s coarse colour error.** The wavelet split alone beats
  `lab` after a 16 px blur on 10 of 10 clips, by 0.03 to 0.51. Adding numz's a\*b\* rank matching
  to the split reproduces `lab`'s figures (0.644 against 0.664, 1.152 against 1.167). `lab` also
  keeps less detail than the split alone on 8 of 10 clips: 92–102% of the uncorrected output's
  Laplacian variance, against 99–100% (the sky: 87% for both).
- **Colour at a finer scale than lightness is nearly free.** Lightness from the input below the
  split's 13 px and colour below 3.2 px brings ΔE00 after a 4 px blur down by 6–45% against
  numz's `lab`, better at every blur scale on 10 of 10 clips, with the luma detail unchanged. 3.2
  px is about the input's own colour resolution at ×2 (a 4:2:0 source has a chroma sample every
  4 output pixels). Colour taken finer than that gets worse again.
- **Lightness at a finer scale costs detail:** at 3.2 px for both, ΔE00 after a 4 px blur reaches
  the input's own on all 10 clips (within 0.004), for 2–4.5% of the Laplacian variance (27% on
  the sky clip, whose fine texture is faint).

### Method (step 0)

- **Masters:** numerics.md's full-reference runs (7B fp16, one batch of 45 frames, 1080p from a
  540p input): the run with `--color_correction none`, its `lab` rendering (the same decoded
  frames, numz's own Phase 4), the bicubic baseline (the d1 input upscaled ×2 with Catmull-Rom)
  and the GT, all 16-bit RGB. `none` and `lab` at seeds 42, 43 and 1234.
- **Clips** (numerics.md, [Clips](numerics.md#clips)): anime-clean, anime-grain, anime-dark,
  cartoon-bright, anime-sky, anime-bright, live-vfx and live-slow at d1; anime-clean and
  anime-grain at d2 (an area downscale and CRF 26).
- **ΔE00 by scale:** CIEDE2000 between the output and the GT after a Gaussian blur of σ = 0, 1,
  2, 4, 8 and 16 px of both CIELAB images (OpenCV's float conversion, sRGB, D65), on every
  second pixel (every σ/2-th from σ = 8). σ = 4 is fr_metrics.py's ΔE00 lf. It matches
  fr_metrics.py's per-frame values to 1e-6 on every clip, seed and output. ΔE00² splits exactly
  into a lightness term, (ΔL'/S_L)², and a colour term: the chroma and hue terms with their
  rotation.
- **Bands:** the CIELAB difference to the GT split by differences of Gaussians (σ < 1, 1–2, 2–4,
  4–8, 8–16, 16–32, > 32 px): the RMS of ΔL\* and of the a\*b\* distance in each band.
- **The picture** leaves out letterbox bars and the bottom 16 rows, which numz's padding damages
  ([numerics.md](numerics.md#numzs-black-band)).
- **Oracle:** what a split could reach, with the bicubic baseline standing in for the encoder's
  input. The `none` master takes the baseline's low band in Y'CbCr (BT.709 matrix on the
  gamma-encoded RGB). That transform is linear, so equal scales are the plain RGB split. Y' takes
  the low band of sL à-trous stages, Cb and Cr that of sC stages. The low band is
  `runtime/colour.py`'s: the 3×3 binomial kernel, taps 2^stage px apart, edges replicated. 5
  stages are σ = 13.1 px (numz's split), 4: 6.5, 3: 3.2, 2: 1.6, 1: 0.7. (5, 5) is numz's
  `wavelet`, `lab` without its histogram step. The detail kept is the luma Laplacian variance
  over the uncorrected output's, as fr_clips.py computes it.
- **The histogram step alone:** numz's matching (a\* and b\* get the reference's value of the
  same rank, L\* = 0.8 own + 0.2 matched; exact ranks over the 45 frames, as numz's one batch) on
  top of the oracle's split, in OpenCV's CIELAB, the reference being the bicubic baseline.

### Where `lab`'s error sits

ΔE00 after a blur of σ px, whole frame; `lab` the mean of 3 seeds (its spread in brackets):

| Clip | Output | σ 0 | σ 1 | σ 2 | σ 4 | σ 8 | σ 16 |
|---|---|---|---|---|---|---|---|
| anime-clean | bicubic | 1.57 | 1.26 | 0.98 | 0.65 | 0.41 | 0.30 |
| | `lab` | 2.97 (0.14) | 2.34 (0.09) | 1.79 (0.06) | 1.25 (0.04) | 0.84 (0.02) | 0.66 (0.02) |
| anime-grain | bicubic | 1.47 | 1.20 | 0.96 | 0.68 | 0.42 | 0.26 |
| | `lab` | 2.46 (0.05) | 2.02 (0.04) | 1.61 (0.03) | 1.16 (0.02) | 0.80 (0.01) | 0.62 (0.01) |
| anime-dark | bicubic | 1.32 | 1.12 | 0.95 | 0.74 | 0.50 | 0.30 |
| | `lab` | 1.86 (0.01) | 1.57 (0.01) | 1.35 (0.01) | 1.07 (0.01) | 0.73 (0.00) | 0.44 (0.00) |
| cartoon-bright | bicubic | 1.66 | 1.26 | 0.96 | 0.63 | 0.36 | 0.22 |
| | `lab` | 2.92 (0.02) | 2.45 (0.02) | 2.07 (0.01) | 1.58 (0.01) | 1.06 (0.01) | 0.69 (0.01) |
| anime-sky | bicubic | 0.84 | 0.68 | 0.58 | 0.47 | 0.36 | 0.25 |
| | `lab` | 1.18 (0.01) | 0.86 (0.02) | 0.71 (0.01) | 0.57 (0.01) | 0.44 (0.01) | 0.34 (0.01) |
| anime-bright | bicubic | 2.02 | 1.64 | 1.27 | 0.87 | 0.61 | 0.49 |
| | `lab` | 3.29 (0.07) | 2.68 (0.05) | 2.26 (0.05) | 1.84 (0.05) | 1.45 (0.05) | 1.14 (0.05) |
| live-vfx | bicubic | 1.24 | 0.97 | 0.80 | 0.61 | 0.41 | 0.25 |
| | `lab` | 1.71 (0.03) | 1.36 (0.01) | 1.15 (0.00) | 0.92 (0.00) | 0.63 (0.00) | 0.37 (0.00) |
| live-slow | bicubic | 1.33 | 1.07 | 0.86 | 0.61 | 0.38 | 0.24 |
| | `lab` | 2.06 (0.01) | 1.73 (0.01) | 1.44 (0.01) | 1.10 (0.01) | 0.72 (0.00) | 0.45 (0.00) |
| anime-clean d2 | bicubic | 1.98 | 1.69 | 1.42 | 1.07 | 0.79 | 0.69 |
| | `lab` | 3.09 (0.13) | 2.48 (0.08) | 1.96 (0.04) | 1.45 (0.02) | 1.08 (0.01) | 0.93 (0.01) |
| anime-grain d2 | bicubic | 1.76 | 1.51 | 1.28 | 0.99 | 0.70 | 0.50 |
| | `lab` | 2.64 (0.04) | 2.21 (0.03) | 1.81 (0.02) | 1.36 (0.01) | 1.01 (0.01) | 0.83 (0.01) |

- **Colour, not lightness.** After a 4 px blur, `lab`'s lightness term holds 16–31% of its ΔE00²
  over the picture on 9 clips (62% on live-slow). Unblurred, its mean colour part exceeds its
  lightness part on 9 of 10 clips (live-slow: 1.50 against 1.89).
- **By band** (RMS over the picture), the input is closer to the GT than the uncorrected model
  in every band and channel on all 10 clips, but in two mid colour bands of live-slow where they
  are equal. `lab` changes the bands finer than 8 px very little, as its split intends. Above
  32 px it leaves colour errors of 0.24–1.40 against the input's 0.21–0.83: anime-clean 0.97
  against 0.33, cartoon-bright 0.92 against 0.24. On anime-bright it is worse than the
  uncorrected output in the bands from 8 px up (1.40 against 1.27 above 32 px).
- **In the picture.** Strong edges (the GT's steepest tenth) hold 12–23% of `lab`'s error after
  a 4 px blur, the flattest half 24–47%. No hue stands out: blues are no worse than the rest.
  On the 6 clips without letterbox bars, the bottom 16 rows hold 1.6–8.5% of the error from 1.5%
  of the pixels: numz's padding.
- **Offsets are not the problem:** `lab`'s per-frame mean ΔL\* is at most 0.28, Δa\* and Δb\* at
  most 0.5, as for the input. The error left after a 4 px blur mostly changes from frame to
  frame: its mean over the clip holds 3–23% of its lightness energy (64% on the sky) and 4–38% of
  its colour energy.
- **The model's own drift, uncorrected:** chroma +0.6 to +1.7 C\* above the GT on 9 clips (the
  sky: +0.07), and b\* +0.45 to +1.1 (toward yellow, not blue) on 7. The drift toward blue that
  DESIGN.md quotes came from one dark, bluish clip
  ([quality.md](quality.md#colour-correction)).

### The split at two scales (oracle, seed 42)

ΔE00 after a 4 px blur (whole frame), and the detail kept (luma Laplacian variance over the
uncorrected output's):

| Clip | bicubic | `lab` | split (5, 5) | (5, 3) | (4, 2) | (3, 3) | detail: `lab` | (5, 5) | (5, 3) | (3, 3) |
|---|---|---|---|---|---|---|---|---|---|---|
| anime-clean | 0.651 | 1.249 | 1.180 | 0.876 | 0.764 | 0.640 | 0.976 | 0.994 | 0.994 | 0.966 |
| anime-grain | 0.681 | 1.155 | 1.013 | 0.790 | 0.741 | 0.633 | 0.976 | 1.000 | 0.999 | 0.980 |
| anime-dark | 0.738 | 1.072 | 1.067 | 0.905 | 0.796 | 0.706 | 1.022 | 0.995 | 0.995 | 0.955 |
| cartoon-bright | 0.627 | 1.589 | 1.504 | 0.978 | 0.790 | 0.631 | 0.958 | 0.988 | 0.988 | 0.963 |
| anime-sky | 0.471 | 0.570 | 0.559 | 0.535 | 0.501 | 0.471 | 0.865 | 0.871 | 0.870 | 0.728 |
| anime-bright | 0.874 | 1.869 | 1.503 | 1.020 | 0.894 | 0.748 | 0.922 | 0.998 | 0.997 | 0.980 |
| live-vfx | 0.610 | 0.915 | 0.905 | 0.759 | 0.651 | 0.580 | 1.005 | 0.991 | 0.991 | 0.977 |
| live-slow | 0.605 | 1.101 | 1.094 | 1.019 | 0.775 | 0.577 | 0.981 | 0.998 | 0.998 | 0.974 |
| anime-clean d2 | 1.067 | 1.459 | 1.393 | 1.232 | 1.152 | 1.029 | 0.975 | 0.994 | 0.994 | 0.967 |
| anime-grain d2 | 0.993 | 1.358 | 1.162 | 1.051 | 1.033 | 0.916 | 0.971 | 1.000 | 0.999 | 0.980 |

(sL, sC): lightness and colour below sL and sC à-trous stages, 5 = σ 13.1 px, 4 = 6.5, 3 = 3.2,
2 = 1.6.

- **Colour at 3.2 px, lightness at 13 px** (5, 3) closes 17–85% of the gap between `lab` and
  the input's own error after a 4 px blur, and beats `lab` at every blur scale on all 10 clips
  (live-slow unblurred: equal). Colour at 1.6 or 0.7 px gets worse again after a 4 px blur on 10
  of 10 clips: the input has no finer colour to give.
- **Lightness finer** takes the error to the input's own level, (3, 3) within 0.004 of the
  bicubic baseline or below it on all 10 clips, for a loss of detail that grows as the scale
  shrinks: up to 2% of the Laplacian variance at 6.5 px, 2–4.5% at 3.2 px, 8–17% at 1.6 px (the
  sky: 19%, 27%, 38%).

The histogram step on top of the split, against the split alone (ΔE00 after a blur of 4 px /
16 px, seed 42):

| Clip | (5, 5) alone | + numz's matching | (5, 3) alone | + numz's matching | `lab` |
|---|---|---|---|---|---|
| anime-clean | 1.187 / 0.430 | 1.236 / 0.644 | 0.887 / 0.388 | 0.927 / 0.463 | 1.249 / 0.664 |
| anime-grain | 1.012 / 0.319 | 1.152 / 0.629 | 0.791 / 0.291 | 0.864 / 0.421 | 1.155 / 0.618 |
| cartoon-bright | 1.502 / 0.455 | 1.577 / 0.680 | 0.977 / 0.313 | 1.040 / 0.437 | 1.589 / 0.695 |
| anime-sky | 0.559 / 0.304 | 0.584 / 0.373 | 0.535 / 0.307 | 0.556 / 0.362 | 0.570 / 0.340 |
| anime-bright | 1.508 / 0.671 | 1.850 / 1.152 | 1.027 / 0.447 | 1.496 / 0.772 | 1.869 / 1.167 |
| live-slow | 1.093 / 0.296 | 1.089 / 0.446 | 1.018 / 0.287 | 1.002 / 0.341 | 1.101 / 0.445 |

- The matching moves colour at every scale. The reference's a\*b\* distribution is the blurred
  input's, narrower than the output's, which carries the model's fine colour. Matching one onto
  the other compresses colour differences at every scale, the coarse ones included (a
  hypothesis that fits the bands; the a\*b\* part does it, the L\* blend changes little).
  Unblurred, it helps a little on 3 of 6 clips. On top of a finer colour split it still hurts at
  every blur from 2 px up on 5 of 6 clips. On live-slow, whose error is mostly lightness, it
  helps by 0.016 after a 4 px blur.
- If the scores and the eyes confirm it (step 3), dropping the histogram step would also remove
  what makes the decode take two passes: the histograms pooled over the shot
  ([DESIGN.md](../../seedvr2x/DESIGN.md#colour-correction)).

### Step 0's caveats

- **The oracle is indicative.** It works on numz's clamped bfloat16 masters, with the bicubic
  baseline instead of the encoder's input, at the à-trous scales only, in Y'CbCr only.
  Step 1 dumps the exact decode and encoder input. The detail measure is one number, the
  Laplacian variance. LPIPS, DISTS and the eyes come in step 3.
- **Fidelity is not quality** ([numerics.md](numerics.md#caveats)): the model is far from the GT
  everywhere, and an output closer to it is not necessarily nicer to watch.
- One model, one upscale factor (×2), 1080p, 45-frame single shots, one batch, untiled. Tiles,
  ×4 and 4K come in steps 1–3.

## Step 1: the dumps

[`scripts/colour_dump.py`](../scripts/colour_dump.py) is a `--wrap` script for numz's CLI, as
[`numerics_patch.py`](../scripts/numerics_patch.py) is. It changes nothing in the run and saves
what a colour correction starts from:

- `decode.pt`: the raw decode, numz's `final_video` as its Phase 4 receives it (bfloat16 in
  [−1, 1], unclamped);
- `enc_bf16.pt`: the encoder's exact input (numz's frames in bfloat16 through the model's
  transform: resize, clamp, padding, normalisation);
- `ref_f32.pt`: the same frames through the same transform in float32, seedvr2x's `lab` reference
  since milestone 5 ([DESIGN.md](../../seedvr2x/DESIGN.md#colour-correction));
- `latents.pt`: the DiT's output. `colour_dump.py decode` decodes it again through numz's own
  Phase 3 with other VAE tiles, or in float16.

Every variant then post-processes the same decode with the same reference: no model run per
variant, and two variants' scores differ by the variants alone.

- **The dumps are the runs measurement scored.** The seed-42 decodes of anime-clean and
  live-vfx, brought to [0, 1] as numz's Phase 4 does with `none` and quantised as `ffv1_out.py`
  does, equal numerics.md's masters of the same runs sample for sample (`colour_dump.py verify`).
- **The decode-only path is numz's decode:** run untiled from a run's latents, it gives that
  run's `decode.pt` bit for bit (`colour_dump.py decode --check`, anime-clean at d1, seed 42).
- **The bfloat16 input reads brighter**, as milestone 5 found: the encoder's input minus the
  float32 reference averages +0.03 to +0.16 levels on the 8 clips at d1 (anime-dark +0.026,
  anime-bright +0.165), −0.05 on clip B, +0.02 on the 4K close-up. Every variant below uses the
  float32 reference. `lab` with the bfloat16 one (`lab@bf16`) stays within `lab@f32`'s seed band
  on every score and clip.

The runs (7B fp16, numz `4490bd1`, `--color_correction none`, 45 frames in one batch, seeds 42,
43 and 1234 unless said):

| Set | Clips | Runs |
|---|---|---|
| d1 | anime-clean, anime-grain, anime-dark, cartoon-bright, anime-sky, anime-bright, live-vfx, live-slow | ×2 from 540p, the mild degradation ([numerics.md](numerics.md#clips)), untiled |
| d2 | anime-clean, anime-grain; live-slow, live-vfx, cartoon-bright, anime-sky, anime-dark | the heavier degradation (area downscale, CRF 26). The last five made by `colour_clips.py --degrade d2` with fr_clips.py's recipe (which remakes fr_clips.py's anime-clean d2 frame for frame), at seeds 42 and 43 |
| tiles, 1080p | anime-clean, anime-dark, cartoon-bright, anime-sky, live-slow | seed 42. Decode-only from the untiled run's latents, tiles of 1280, 1024, 768 and 512 px (overlap 64); runs with tiled encodes, encode / decode tiles 1344 / 1024, 1024 / 768 and 512 / 512, the last one decoded untiled too (the encode's tiles alone) |
| ×4 | anime-clean, cartoon-bright, live-slow; live-vfx, anime-sky | ×4 from 270p: fr_clips.py's d1 at ×1/4 (`colour_clips.py`); the last two at seeds 42 and 43 |
| ×1.5, ×3 | anime-clean, cartoon-bright, live-slow | seeds 42 and 43, from 720p and 360p: fr_clips.py's d1 at ×1/1.5 and ×1/3 (`colour_clips.py`) |
| clip B | milestone 2's clip | seed 42: its windowed run (windows of 6 latents, 2 shared, linear blend) and its one-batch run |
| 4K | 4 live-action shots: a close-up (face), an interior by lamp light (dark), a sunny street (street), a fast pan (motion) | ×2 from 1080p to 3840×2160, 45 frames from the shot's first ([numerics.md](numerics.md#clips)); seeds 42 and 43 (1234 too); the 96 GB card's plan: encode untiled, decode tiles of 2048 px with vram.md's overlap of 64 (measurement's 4K runs used 128); numz's padding, which adds nothing at 2160 rows |
| 4K tiles | the same 4 | seed 42. Decode-only from the 2048 run's latents, tiles of 1536, 1024, 768 and 512 px; dark and street also run with encode / decode tiles 1344 / 1024 and 1024 / 768 |
| float16 | the 8 clips at d1 | seed 42's latents decoded again with the VAE in float16, untiled |

A 1080p run takes about 4 minutes on the RTX PRO 6000. A 4K run takes 11.8: the untiled encode 3
minutes at a torch peak of 77.2 GiB, the DiT 2.3 at 64.4 GiB, the decode in 2048-px tiles 6.

## Step 2: the variants

[`scripts/colour_variants.py`](../scripts/colour_variants.py) builds every variant on seedvr2x's
own `runtime/colour.py`, as milestone 5 accepted it, so that a winner ports as it is. A variant
takes one shot's decode and reference and returns RGB in [0, 1], in float32:

- `lab`: seedvr2x's (milestone 5): numz's split in RGB at 5 à-trous stages, then numz's
  histogram step pooled over the shot. `@f32` and `@bf16` name the reference.
- `split:SPACE:SL:SC`: lightness below SL à-trous stages and colour below SC stages from the
  reference, the rest from the decode. The stages' σ: 1: 0.7 px, 2: 1.6, 3: 3.2, 4: 6.5, 5: 13.1
  (numz's split), 6: 26. SPACE: `ycc` (BT.709 Y'CbCr on the gamma-encoded RGB, linear: `ycc:S:S`
  is the RGB split), `lab` (CIELAB, colour.py's conversions), `ok` (OKLab). `rgb:5:5` is numz's
  `wavelet`: `lab` without its histogram step. The names below drop `split:` and `@f32`.
- `:hist` after a split: numz's histogram step on top (a\* and b\* matched, L\* = 0.8 own + 0.2
  matched); `:histY0.8`: its L\* blend alone, a\* and b\* kept.
- `guided:SPACE:SL:R:EPS`: lightness as the split; colour from a guided filter (He, Sun and Tang)
  of the reference-minus-decode colour difference, steered by the decode's lightness, radius R px.

Step 0's oracle used numz's clamped masters and the bicubic baseline as the reference; these use
the exact decode and the encoder's own input, in float32.

## Step 3: the scores

### Protocol

[`scripts/colour_eval.py`](../scripts/colour_eval.py) scores each variant's output against the
16-bit ground truth, frame by frame:
- ΔE00 after Gaussian blurs of σ = 0, 1, 2, 4, 8 and 16 px (colour_diag.py's; σ 4 is
  fr_metrics.py's ΔE00 lf), with its lightness and colour parts after 2 px over the picture;
- PSNR-Y (BT.709 luma, 8-bit scale), LPIPS (AlexNet), DISTS (every 9th frame);
- detail: the luma Laplacian variance (fr_clips.py's);
- flicker: fr_metrics.py's temporal errors, |Δoutput − ΔGT| between consecutive frames, at full
  resolution and on 16×16 block means (lf);
- colour fringes: the colour part of the unblurred ΔE00 on the GT's strongest edges (the top 5%
  of its luma gradient, widened by 2 px);
- on tiled runs, against the untiled decode through the same variant: the mean luma difference
  of 60×60 blocks in 8-bit levels, the largest per frame, over all blocks and over the GT's
  flattest third.

Verdicts as in numerics.md: per clip, each variant paired with `lab@f32` frame by frame, pooled
over the seeds, with a 95% interval by moving-block bootstrap (blocks of 8 frames, 2,000
resamples). Better (B) or worse (W) when the interval excludes 0 and the mean difference exceeds
`lab`'s seed spread (its largest per-seed mean minus its smallest), within otherwise. The tables
give the mean over the clips of the per-clip differences, then the counts B / W / within. Detail
gives the mean change of the Laplacian variance and the clips where it rose (↑) or fell (↓)
beyond the band.

### 1080p, mild degradation (d1)

8 clips, 3 seeds; variant − `lab`:

| Variant | ΔE00 σ 0 | σ 4 | σ 16 | PSNR-Y | LPIPS | DISTS | Detail | Fringes | Flicker | Flicker lf |
|---|---|---|---|---|---|---|---|---|---|---|
| `none` | +0.748 (0/8/0) | +0.969 (0/8/0) | +1.126 (0/8/0) | −2.29 (0/8/0) | +0.0021 (1/5/2) | +0.0008 (1/3/4) | +9.9 (2↑) | +0.493 (0/7/1) | +0.996 (0/7/1) | +1.374 (0/8/0) |
| `lab@bf16` | +0.001 (0/0/8) | +0.001 (0/0/8) | −0.001 (3/0/5) | −0.01 (0/0/8) | −0.0001 (0/0/8) | +0.0000 (0/0/8) | +0.0 | −0.000 (0/0/8) | +0.001 (0/0/8) | +0.001 (0/0/8) |
| `rgb:5:5` | +0.022 (0/3/5) | −0.071 (4/1/3) | −0.183 (8/0/0) | −0.12 (0/1/7) | −0.0021 (2/1/5) | −0.0004 (2/1/5) | +8.4 | +0.165 (0/8/0) | +0.163 (0/4/4) | +0.080 (0/5/3) |
| `ycc:5:4` | −0.095 (5/1/2) | −0.240 (8/0/0) | −0.225 (8/0/0) | −0.12 (0/1/7) | −0.0045 (3/0/5) | −0.0016 (2/0/6) | +8.4 | −0.088 (4/1/3) | +0.162 (0/4/4) | +0.080 (0/5/3) |
| `ycc:5:3` | −0.196 (6/1/1) | −0.317 (8/0/0) | −0.248 (8/0/0) | −0.12 (0/1/7) | −0.0070 (4/1/3) | −0.0018 (1/2/5) | +8.4 | −0.454 (7/0/1) | +0.160 (0/4/4) | +0.079 (0/5/3) |
| `ycc:5:3:hist` | −0.181 (8/0/0) | −0.242 (8/0/0) | −0.161 (7/0/1) | +0.03 (0/0/8) | −0.0049 (4/1/3) | −0.0006 (0/2/6) | +0.6 | −0.460 (8/0/0) | −0.011 (0/0/8) | −0.012 (1/0/7) |
| `ycc:5:3:histY0.8` | −0.212 (7/0/1) | −0.324 (8/0/0) | −0.236 (8/0/0) | +0.02 (0/0/8) | −0.0076 (4/0/4) | −0.0021 (1/1/6) | +2.9 | −0.454 (7/0/1) | +0.043 (0/1/7) | +0.008 (0/1/7) |
| `ycc:5:2` | −0.275 (7/0/1) | −0.295 (7/0/1) | −0.249 (8/0/0) | −0.11 (0/1/7) | −0.0023 (4/1/3) | −0.0017 (3/2/3) | +8.5 | −0.870 (8/0/0) | +0.155 (0/4/4) | +0.078 (0/5/3) |
| `ycc:4:3` | −0.297 (8/0/0) | −0.461 (8/0/0) | −0.286 (8/0/0) | +0.63 (4/0/4) | −0.0096 (7/0/1) | −0.0023 (2/2/4) | +7.3 | −0.459 (7/0/1) | −0.441 (6/0/2) | −0.755 (8/0/0) |
| `ycc:4:3:histY0.8` | −0.305 (8/0/0) | −0.460 (8/0/0) | −0.273 (8/0/0) | +0.73 (5/0/3) | −0.0101 (7/0/1) | −0.0024 (2/2/4) | +2.4 | −0.459 (7/0/1) | −0.514 (6/0/2) | −0.792 (8/0/0) |
| `ycc:4:2` | −0.377 (8/0/0) | −0.438 (8/0/0) | −0.287 (8/0/0) | +0.63 (4/0/4) | −0.0053 (5/1/2) | −0.0023 (3/2/3) | +7.3 | −0.872 (8/0/0) | −0.446 (6/0/2) | −0.757 (8/0/0) |
| `ycc:3:3` | −0.400 (8/0/0) | −0.554 (8/0/0) | −0.301 (8/0/0) | +1.69 (8/0/0) | −0.0166 (7/0/1) | −0.0063 (5/1/2) | +3.6 (1↓) | −0.464 (8/0/0) | −1.129 (8/0/0) | −1.250 (8/0/0) |
| `ycc:3:2` | −0.484 (8/0/0) | −0.532 (8/0/0) | −0.300 (8/0/0) | +1.69 (8/0/0) | −0.0135 (6/0/2) | −0.0066 (4/3/1) | +3.6 (1↓) | −0.874 (8/0/0) | −1.132 (8/0/0) | −1.251 (8/0/0) |
| `ycc:2:2` | −0.626 (8/0/0) | −0.556 (8/0/0) | −0.305 (8/0/0) | +3.64 (8/0/0) | −0.0253 (7/0/1) | −0.0126 (6/2/0) | −14.4 (5↓) | −0.873 (8/0/0) | −2.068 (8/0/0) | −1.418 (8/0/0) |
| `lab:5:3` | −0.200 (7/1/0) | −0.312 (7/0/1) | −0.239 (7/0/1) | −0.14 (0/1/7) | −0.0058 (4/1/3) | −0.0006 (1/3/4) | +7.4 | −0.455 (7/0/1) | +0.114 (0/4/4) | +0.052 (1/5/2) |
| `ok:5:3` | −0.215 (7/1/0) | −0.310 (7/0/1) | −0.231 (7/0/1) | −0.16 (0/2/6) | −0.0071 (5/0/3) | −0.0012 (1/2/5) | +7.9 | −0.509 (8/0/0) | +0.074 (0/2/6) | +0.027 (1/4/3) |
| `ok:4:3` | −0.316 (8/0/0) | −0.454 (8/0/0) | −0.272 (8/0/0) | +0.57 (4/0/4) | −0.0097 (6/0/2) | −0.0013 (2/3/3) | +7.7 | −0.513 (8/0/0) | −0.493 (6/0/2) | −0.766 (8/0/0) |
| `ok:3:3` | −0.419 (8/0/0) | −0.546 (8/0/0) | −0.289 (8/0/0) | +1.60 (8/0/0) | −0.0167 (6/0/2) | −0.0056 (5/1/2) | +5.5 (1↑ 1↓) | −0.518 (8/0/0) | −1.134 (8/0/0) | −1.223 (8/0/0) |
| `guided:ycc:5:4:0.001` | −0.213 (7/1/0) | −0.273 (8/0/0) | −0.231 (8/0/0) | −0.11 (0/1/7) | −0.0019 (4/1/3) | −0.0006 (1/2/5) | +8.1 | −0.723 (7/1/0) | +0.150 (0/4/4) | +0.077 (0/5/3) |

`lab`'s own means, for scale: ΔE00 σ 4 0.56–1.84, PSNR-Y 25.7–36.1 dB, LPIPS 0.07–0.21, DISTS
0.07–0.15, Laplacian variance 9 (anime-dark) to 494 (anime-bright).

- **Colour from the input below 3.2 px beats `lab` everywhere it measures colour.** Every split
  with SC = 3 lowers ΔE00 at every blur scale on 7 or 8 of the 8 clips, and colour fringes by
  0.45 (7–8 clips). numz's split alone (`rgb:5:5`) only wins after coarse blurs (σ 16, 8 clips)
  and worsens fringes on all 8: colour taken at 13 px bleeds across edges.
- **numz's histogram step costs colour,** as step 0 found: on top of `ycc:5:3` it gives back a
  quarter of the gain after 4 px (−0.24 against −0.32) and a third after 16 px (−0.16 against
  −0.25). Its L\* blend alone (`histY0.8`) keeps the whole colour gain and repairs what `ycc:5:3`
  alone does to flicker (+0.16, worse on 4 clips): at 13 px, the model's own lightness flickers more
  than `lab`'s blended one.
- **Lightness at 6.5 px needs no histogram:** `ycc:4:3` flickers less than `lab` on 6 to 8 clips
  (−0.44, lf −0.76) with no histogram step at all, and adds PSNR-Y (+0.63 dB, 4 clips) and LPIPS
  (−0.0096, 7 clips). The L\* blend on top (`4:3:histY0.8`) changes little: +0.1 dB, flicker
  −0.07 and −0.04, and less detail (Laplacian +2.4 instead of +7.3).
- **Lightness at 3.2 px** (`ycc:3:3`) gives the most before detail goes: PSNR-Y +1.69 dB, LPIPS
  −0.017 and flicker −1.13 / −1.25 on 7–8 clips, DISTS −0.006 (5 better, 1 worse), detail
  lower on one clip (the sky, whose fine texture is faint, as in step 0). At 1.6 px (`2:2`)
  detail falls on 5 clips (−14) and DISTS gets worse on 2.
- **Colour at 1.6 px** (SC = 2) halves the fringes again (−0.87 against −0.46, 8 clips) and helps
  unblurred ΔE00, but ΔE00 after 4 px is a little worse than at 3.2 px and LPIPS gains about half
  as much (`4:2` −0.0053 against `4:3` −0.0096): the input holds little colour finer than 3.2
  px at ×2, but edges.
- **The space hardly matters:** CIELAB and OKLab give what Y'CbCr gives, within a few
  hundredths, and Y'CbCr is a 3×3 matrix. The guided filter's fringe gain (−0.72) is SC = 2's at
  a higher cost, with `5:4`'s flicker.
- **The reference's precision doesn't matter after `lab`:** `lab@bf16` is within `lab@f32` on
  every score of every clip.

### The heavier degradation (d2)

The heavier degradation (area downscale, CRF 26) leaves the input blurrier and blockier: a
finer split takes more of that. 7 clips: anime-clean and anime-grain at 3 seeds, live-slow,
live-vfx, cartoon-bright, anime-sky and anime-dark at 2; variant − `lab`:

| Variant | ΔE00 σ 0 | σ 4 | PSNR-Y | LPIPS | DISTS | Detail | Fringes | Flicker lf |
|---|---|---|---|---|---|---|---|---|
| `none` | +0.781 (0/7/0) | +0.983 (0/7/0) | −2.66 (0/7/0) | +0.0049 (0/5/2) | +0.0025 (1/4/2) | +5.7 (3↑) | +0.473 (0/7/0) | +1.331 (0/7/0) |
| `ycc:5:3` | −0.107 (5/2/0) | −0.149 (5/2/0) | −0.14 (0/3/4) | −0.0018 (3/1/3) | +0.0013 (2/3/2) | +4.2 (2↑) | −0.281 (6/1/0) | +0.074 (0/5/2) |
| `ycc:5:3:histY0.8` | −0.125 (5/2/0) | −0.158 (6/0/1) | +0.01 (0/0/7) | −0.0024 (3/1/3) | +0.0009 (2/3/2) | +0.2 | −0.281 (6/1/0) | −0.001 (1/1/5) |
| `ycc:4:3` | −0.204 (7/0/0) | −0.273 (7/0/0) | +0.65 (5/0/2) | −0.0044 (4/1/2) | +0.0011 (1/4/2) | +3.0 (1↑ 1↓) | −0.287 (6/1/0) | −0.696 (7/0/0) |
| `ycc:4:3:histY0.8` | −0.213 (7/0/0) | −0.274 (7/0/0) | +0.75 (6/0/1) | −0.0049 (5/0/2) | +0.0009 (1/4/2) | −0.4 (1↓) | −0.286 (6/1/0) | −0.734 (7/0/0) |
| `ycc:3:3` | −0.298 (7/0/0) | −0.345 (7/0/0) | +1.65 (7/0/0) | −0.0108 (5/0/2) | −0.0010 (3/3/1) | −0.2 (3↓) | −0.294 (6/1/0) | −1.099 (7/0/0) |

- **The fidelity gains hold on compressed inputs:** `ycc:4:3` lowers ΔE00 on 7 of 7 clips, adds
  PSNR-Y on 5 and removes flicker on 7, as at d1.
- **DISTS is where they cost, a little:** every split is worse than `lab` on DISTS on 3 or 4 of the
  7 clips, by +0.003 to +0.005 for `ycc:4:3` (anime-grain, anime-sky, cartoon-bright, live-slow;
  `lab`'s own DISTS there 0.06–0.15), better on one (live-vfx, −0.008), within on two. `ycc:3:3` is
  no worse on average (−0.001: 3 better, 3 worse, the grainy anime clip +0.009) but loses detail on
  3 clips (the sky −15%: 36.6 against `lab`'s 42.8; the dark clip and live-slow less). The first
  two-clip d2 runs had put `ycc:3:3` at +0.006: the 5 more clips don't confirm that it is worse than
  `ycc:4:3` on compressed inputs.

### ×4

D1's recipe at ×1/4 (a 270p input, fr_clips.py's d1 at ×1/4 by `colour_clips.py`), upscaled to
1080p: anime-clean, cartoon-bright and live-slow at 3 seeds, live-vfx and anime-sky at 2;
variant − `lab`:

| Variant | ΔE00 σ 0 | σ 4 | σ 16 | PSNR-Y | LPIPS | DISTS | Detail | Fringes | Flicker lf |
|---|---|---|---|---|---|---|---|---|---|
| `ycc:5:3:histY0.8` | +0.027 (0/3/2) | −0.008 (1/3/1) | −0.102 (4/0/1) | −0.00 (0/0/5) | +0.0028 (1/2/2) | +0.0043 (1/2/2) | +0.3 | +0.020 (2/2/1) | −0.000 (0/0/5) |
| `ycc:5:4` | −0.021 (2/2/1) | −0.079 (4/0/1) | −0.125 (5/0/0) | −0.11 (0/1/4) | −0.0007 (1/0/4) | +0.0000 (1/1/3) | +5.4 | +0.002 (2/2/1) | +0.066 (0/3/2) |
| `ycc:4:3` | −0.040 (4/0/1) | −0.106 (5/0/0) | −0.144 (5/0/0) | +0.55 (3/0/2) | +0.0019 (2/2/1) | +0.0051 (0/3/2) | +3.5 (1↓) | +0.018 (1/2/2) | −0.583 (5/0/0) |
| `ycc:4:4` | −0.099 (3/0/2) | −0.181 (5/0/0) | −0.152 (5/0/0) | +0.55 (3/0/2) | −0.0020 (4/0/1) | +0.0009 (2/2/1) | +3.5 (1↓) | +0.001 (2/2/1) | −0.582 (5/0/0) |
| `ycc:3:3` | −0.119 (5/0/0) | −0.162 (5/0/0) | −0.156 (5/0/0) | +1.37 (5/0/0) | −0.0027 (2/0/3) | +0.0056 (1/3/1) | −1.4 (1↓) | +0.018 (1/2/2) | −0.894 (5/0/0) |
| `ycc:3:4` | −0.176 (5/0/0) | −0.238 (5/0/0) | −0.163 (5/0/0) | +1.37 (5/0/0) | −0.0059 (4/0/1) | +0.0020 (1/3/1) | −1.5 (1↓) | +0.003 (2/2/1) | −0.894 (5/0/0) |

On the 3 clips with seed 42 only: colour at 1.6 px (`4:2`, `3:2`, `5:2`) costs LPIPS (+0.018 to
+0.023) and DISTS (+0.015 to +0.017) on 2–3 of 3; lightness at 26 px (`6:4`, `6:5`) adds flicker
(+0.45, 3 of 3).

- **The colour scale follows the factor:** at ×4, colour from the input below 3.2 px costs
  LPIPS and DISTS (`ycc:4:3`: DISTS worse on 3 of 5), below 6.5 px it doesn't (`ycc:4:4`:
  LPIPS better on 4, DISTS 2 better, 2 worse) and gives more colour fidelity (σ 4 −0.18 against
  −0.11). Both are about 1.6 source pixels: the input's own colour resolution (a 4:2:0 source
  has a chroma sample every 2 source pixels). At ×2, 6.5 px was worse than 3.2 (`ycc:5:4`
  against `ycc:5:3`).
- **Lightness doesn't follow it:** 6.5 px of output keeps working at ×4 (flicker lf −0.58 on 5
  of 5, PSNR-Y +0.55 dB), where 13 px already does nothing for flicker (`ycc:5:4`) and 26 px
  adds it. The input's luma is sampled at every source pixel: there is real lightness in it down
  to 1.6 source px at ×4.
- So the correction's colour stages are 2 + log2(factor): 3 at ×2, 4 at ×4 (at other factors, the
  stage whose σ is nearest 1.6 source pixels: checked at ×1.5 and ×3 below), its lightness stages
  4 at both.
- `ycc:3:4` at ×4 is `ycc:3:3` at ×2 again: more fidelity and less flicker, DISTS worse on 3 of
  5 clips and detail lower on one.

### ×1.5 and ×3

The rule above, checked at two factors users meet: ×1.5 (720p to 1080p, a 1280×720 input) and ×3
(360p to 1080p, 640×360), fr_clips.py's d1 at ×1/F (`colour_clips.py --factor F`); anime-clean,
cartoon-bright and live-slow at seeds 42 and 43. The colour scale nearest 1.6 source pixels is 3
stages at ×1.5 (2.4 output pixels: σ 3.2 is the nearest) and 4 at ×3 (4.8 output pixels: σ 6.5
is the nearest, on a log scale). Variant − `lab`:

| Factor | Variant | ΔE00 σ 0 | σ 4 | PSNR-Y | LPIPS | DISTS | Detail | Fringes | Flicker lf |
|---|---|---|---|---|---|---|---|---|---|
| ×1.5 | `ycc:4:2` | −0.620 (3/0/0) | −0.680 (3/0/0) | +0.42 (3/0/0) | −0.0061 (2/1/0) | +0.0029 (1/2/0) | +6.6 | −1.277 (3/0/0) | −1.032 (3/0/0) |
| | `ycc:4:3` | −0.457 (3/0/0) | −0.661 (3/0/0) | +0.42 (3/0/0) | −0.0066 (3/0/0) | +0.0006 (1/1/1) | +6.5 | −0.532 (3/0/0) | −1.031 (3/0/0) |
| | `ycc:4:4` | −0.249 (3/0/0) | −0.477 (3/0/0) | +0.42 (3/0/0) | −0.0014 (2/1/0) | +0.0007 (1/2/0) | +6.4 | +0.063 (1/2/0) | −1.030 (3/0/0) |
| | `ycc:5:3` | −0.297 (2/0/1) | −0.432 (3/0/0) | −0.20 (0/3/0) | −0.0060 (3/0/0) | −0.0001 (1/0/2) | +8.1 | −0.524 (3/0/0) | +0.151 (0/3/0) |
| | `ycc:3:2` | −0.799 (3/0/0) | −0.850 (3/0/0) | +1.38 (3/0/0) | −0.0111 (3/0/0) | −0.0003 (2/1/0) | +1.6 (2↓) | −1.287 (3/0/0) | −1.828 (3/0/0) |
| | `ycc:3:3` | −0.629 (3/0/0) | −0.825 (3/0/0) | +1.38 (3/0/0) | −0.0101 (3/0/0) | −0.0023 (2/0/1) | +1.5 (2↓) | −0.540 (3/0/0) | −1.828 (3/0/0) |
| ×3 | `ycc:4:3` | −0.226 (3/0/0) | −0.362 (3/0/0) | +0.37 (3/0/0) | −0.0001 (1/1/1) | +0.0039 (0/2/1) | +5.8 | −0.265 (2/1/0) | −0.825 (3/0/0) |
| | `ycc:4:4` | −0.201 (3/0/0) | −0.363 (3/0/0) | +0.37 (3/0/0) | −0.0017 (2/0/1) | −0.0004 (1/1/1) | +5.8 | −0.028 (1/1/1) | −0.824 (3/0/0) |
| | `ycc:4:5` | −0.109 (3/0/0) | −0.227 (3/0/0) | +0.37 (3/0/0) | −0.0004 (2/0/1) | +0.0002 (1/1/1) | +5.7 | +0.191 (0/3/0) | −0.824 (3/0/0) |
| | `ycc:5:4` | −0.086 (2/1/0) | −0.206 (3/0/0) | −0.16 (0/2/1) | −0.0013 (2/0/1) | −0.0007 (1/0/2) | +7.6 | −0.024 (1/1/1) | +0.114 (0/3/0) |
| | `ycc:3:3` | −0.357 (3/0/0) | −0.466 (3/0/0) | +1.23 (3/0/0) | −0.0033 (2/0/1) | +0.0025 (1/2/0) | −0.1 | −0.268 (2/1/0) | −1.395 (3/0/0) |
| | `ycc:3:4` | −0.327 (3/0/0) | −0.464 (3/0/0) | +1.23 (3/0/0) | −0.0040 (2/0/1) | −0.0008 (1/0/2) | −0.1 | −0.028 (1/1/1) | −1.395 (3/0/0) |

- **The rule holds at both factors.** At ×1.5, colour at 3 stages is the balance: at 2 it
  costs DISTS on 2 of 3 clips (+0.0029), at 4 it gives back most of the LPIPS gain and the
  fringe gain. At ×3, colour at 4 stages is: at 3 it costs DISTS on 2 of 3 (+0.0039), as at ×4,
  at 5 colour fringes get worse on 3 of 3. So the colour stage is the one whose σ is nearest
  1.6 source pixels, on a log scale: 3 at ×1.5 and ×2, 4 at ×3 and ×4.
- **Lightness at 4 stages works at every factor tested,** ×1.5 to ×4 (flicker lf −0.82 to −1.03
  on 3 of 3 here); at 5 stages it flickers more than `lab` at ×1.5 and ×3 (+0.15, +0.11), as at
  ×4. At ×1.5, where the input holds the most, lightness at 3 stages adds the most (`ycc:3:3`:
  LPIPS −0.010, DISTS −0.0023, PSNR-Y +1.38 dB) and lowers detail on 2 of 3 clips.

### Through the default master (yuv420p10le)

What users get by default: each finalist rendered as seedvr2x's `yuv420p10le` master through
`ffv1_out.py`'s chain (zscale on one slice, BT.709, limited range, chroma sited left, bilinear
4:2:0), read back to RGB with zscale as a player shows it, and scored as a master: VMAF v1 (the
1080p model) and CAMBI at 10 bits, where the master quantises, LPIPS, DISTS on every 3rd frame.
8 clips at d1, 3 seeds; each finalist's master minus `lab`'s master:

| Variant | ΔE00 σ 0 | σ 4 | σ 16 | PSNR-Y | LPIPS | DISTS | VMAF | CAMBI added | Fringes | Flicker lf |
|---|---|---|---|---|---|---|---|---|---|---|
| `ycc:5:3:histY0.8` | −0.253 (8/0/0) | −0.311 (8/0/0) | −0.242 (8/0/0) | +0.02 (0/0/8) | −0.0061 (4/1/3) | −0.0024 (1/1/6) | +2.07 (4/0/4) | +0.000 (0/0/8) | −0.675 (8/0/0) | +0.008 (0/1/7) |
| `ycc:4:3` | −0.341 (8/0/0) | −0.447 (8/0/0) | −0.291 (8/0/0) | +0.63 (4/0/4) | −0.0083 (5/1/2) | −0.0026 (1/2/5) | +3.69 (7/0/1) | +0.000 (0/0/8) | −0.675 (8/0/0) | −0.756 (8/0/0) |
| `ycc:4:3:histY0.8` | −0.350 (8/0/0) | −0.447 (8/0/0) | −0.278 (8/0/0) | +0.73 (5/0/3) | −0.0088 (6/0/2) | −0.0027 (1/2/5) | +4.02 (7/0/1) | +0.000 (0/0/8) | −0.679 (8/0/0) | −0.792 (8/0/0) |
| `ycc:3:3` | −0.451 (8/0/0) | −0.541 (8/0/0) | −0.304 (8/0/0) | +1.69 (8/0/0) | −0.0160 (6/0/2) | −0.0075 (6/1/1) | +9.70 (8/0/0) | +0.000 (0/0/8) | −0.681 (8/0/0) | −1.250 (8/0/0) |

- **The gains survive 4:2:0 and 10 bits,** with VMAF on top: +3.7 for `ycc:4:3` (7 clips
  better), +9.7 for `ycc:3:3` (8). No variant brings banding back (CAMBI added within the band on
  every clip).
- **The master helps every variant, `lab` included:** `lab`'s master against `lab` in float RGB
  has 0.145 less ΔE00 unblurred (8 clips) and 0.58 less colour fringing, as numerics.md found for
  the chroma kernel (the round trip brings the output closer to a GT whose colour came from
  4:2:0); LPIPS is 0.0016 worse on 4 clips. The splits' lead on fringes even grows there:
  −0.68 against `lab`'s master, −0.46 in float RGB.

### Clip B's boundary steps

Milestone 2's measure ([stitch_metrics.py](../scripts/stitch_metrics.py) with its `--latent`
analysis): clip B's run with DiT windows of 6 latents (2 shared, linear blend) against its
one-batch run, both corrected by the same variant. The excess step is, at each window join, the
largest change the join adds over the one-batch run, averaged over the joins: on held frames
(luma flicker) and on the low-frequency brightness and colour of 16×16 blocks (lf).

| Variant | Excess step, held frames | Excess step, lf | PSNR to the one-batch run (dB) |
|---|---|---|---|
| `none` | 0.241 | 0.090 | 39.20 |
| `lab` | 0.221 | 0.046 | 39.97 |
| `ycc:5:3` | 0.221 | 0.049 | 40.06 |
| `ycc:5:3:histY0.8` | 0.220 | 0.048 | 40.09 |
| `ycc:4:3` | 0.209 | 0.028 | 40.39 |
| `ycc:4:3:histY0.8` | 0.208 | 0.027 | 40.42 |
| `ycc:3:3` | 0.189 | 0.013 | 40.92 |

- Lightness from the input at 6.5 px removes 40% of `lab`'s lf step at the joins, at 3.2 px 70%;
  the held-frame step falls by 5% and 14%. At 13 px nothing changes: what the joins add is
  lightness, at the 16-px block scale that a 13 px split still leaves partly to the model.
- The L\* blend adds nothing at 6.5 px (0.208 against 0.209, 0.027 against 0.028).

### The latent grid

DESIGN.md's open question "every fourth frame is the model's best": after a shot's first frame
the causal VAE packs 4 frames per latent, and the last frame of each group (place 3) comes out
closest to the GT. Place 3 minus place 1 (the group's second frame), frames 1–44 (11 groups),
mean over the 8 clips at d1 and the 3 seeds (`colour_eval.py summary --tables groups`):

| Variant | PSNR-Y (dB) | ΔE00 σ 4 | LPIPS |
|---|---|---|---|
| `none` | +2.72 | −0.503 | −0.0209 |
| `lab` | +1.87 | −0.286 | −0.0199 |
| `ycc:5:3:histY0.8` | +1.97 | −0.204 | −0.0190 |
| `ycc:4:3` | +1.59 | −0.115 | −0.0179 |
| `ycc:4:3:histY0.8` | +1.57 | −0.109 | −0.0176 |
| `ycc:3:3` | +1.13 | −0.048 | −0.0132 |
| `ycc:2:2` | +0.72 | −0.018 | −0.0079 |

A correction narrows the grid's gap as far as it takes lightness and colour from the input,
which has no grid: the colour gap from 0.50 to 0.29 with `lab` and 0.05 with `ycc:3:3`, the luma
gap from 2.7 dB to 1.9, 1.6 and 1.1. The perceptual gap (LPIPS) barely moves until 1.6 px: the
group's inner frames differ in fine texture (the ghosts of fast motion), which a split at 3.2 px
and coarser leaves to the model.

### VAE tiles at 1080p

5 clips (anime-clean, anime-dark, cartoon-bright, anime-sky, live-slow), seed 42. Each tiled
output against the untiled one through the same variant: the worst 60×60 block's mean luma
difference per frame (8-bit levels, mean over the frames), its range over the 5 clips, then the
same over the GT's flattest third of blocks in brackets:

| Encode / decode tiles | `none` | `lab` | `ycc:5:3:histY0.8` | `ycc:4:3` | `ycc:3:3` |
|---|---|---|---|---|---|
| untiled / 1280 | 0.85–3.09 (0.69–2.14) | 0.10–0.35 (0.06–0.16) | 0.10–0.34 (0.06–0.16) | 0.04–0.20 (0.02–0.14) | 0.02–0.14 (0.01–0.13) |
| untiled / 1024 | 3.05–6.71 (2.70–4.94) | 0.36–0.87 (0.12–0.45) | 0.35–0.68 (0.12–0.42) | 0.18–0.32 (0.05–0.17) | 0.07–0.22 (0.03–0.12) |
| untiled / 768 | 3.14–8.46 (2.81–8.08) | 0.34–0.81 (0.26–0.36) | 0.33–0.70 (0.26–0.36) | 0.15–0.33 (0.08–0.19) | 0.08–0.19 (0.03–0.12) |
| untiled / 512 | 6.60–10.65 (4.48–9.48) | 0.56–1.00 (0.27–0.49) | 0.56–1.00 (0.27–0.49) | 0.21–0.50 (0.09–0.22) | 0.08–0.22 (0.03–0.11) |
| 1344 / 1024 | 3.60–7.46 (3.18–5.08) | 0.41–0.81 (0.23–0.42) | 0.41–0.66 (0.23–0.41) | 0.16–0.35 (0.10–0.17) | 0.06–0.28 (0.04–0.12) |
| 1024 / 768 | 5.04–8.95 (3.53–7.57) | 0.49–0.72 (0.33–0.44) | 0.49–0.72 (0.33–0.43) | 0.20–0.37 (0.12–0.21) | 0.07–0.20 (0.05–0.12) |
| 512 / 512 | 3.78–6.88 (2.41–5.91) | 0.60–1.11 (0.38–0.49) | 0.60–1.11 (0.38–0.49) | 0.23–0.56 (0.13–0.23) | 0.09–0.42 (0.05–0.12) |
| 512 / untiled | 7.88–16.93 (5.76–15.22) | 0.74–1.39 (0.44–0.63) | 0.75–1.39 (0.44–0.62) | 0.32–0.77 (0.17–0.27) | 0.11–0.45 (0.05–0.15) |

- **A finer split leaves less of the tiles' drift:** `ycc:4:3` halves what `lab` leaves at every
  tile size, the encode's included (512-px decode tiles: 0.21–0.50 against 0.56–1.00; 512-px
  encode tiles: 0.32–0.77 against 0.74–1.39); `ycc:3:3` cuts it to a quarter. Against the
  untiled output, PSNR rises by 0.6–1.3 dB over `lab`'s (`ycc:4:3`) and 1.6–2.8 dB (`ycc:3:3`).
  The drift is uniform over each tile, a low band: the finer the lightness taken from the input,
  the less of it is left.
- `ycc:5:3:histY0.8` leaves what `lab` leaves: at 13 px the tiles' lightness drift stays the
  model's.
- The encode's tiles drift more than the decode's (512-px encode tiles alone, decoded untiled,
  leave the most), as vram.md measured before correction.

### 4K: the default runs against bicubic

DESIGN.md's open question "4K output's quality": measurement's first 4K runs, uncorrected, put
the model further from the ground truth than bicubic on every metric
([numerics.md](numerics.md#4k-no-padding-inside-the-letterbox)). Here the 4 shots in full (the
letterbox's 84 rows included, as for bicubic's figures), 45 frames, seeds 42, 43 and 1234, the
96 GB card's plan; each variant on the run's own decode; bicubic is the d1 input upscaled ×2 with
Catmull-Rom. VMAF v1 with sptenc's 2160p model, on the 16-bit RGB masters through fr_metrics.py's
chain (as bicubic's), on seeds 42 and 43. Means over the seeds:

| Clip | Output | ΔE00 σ 4 | PSNR-Y | LPIPS | DISTS | VMAF | Flicker lf |
|---|---|---|---|---|---|---|---|
| face (close-up) | bicubic | 0.607 | 41.03 | 0.1993 | 0.1455 | 74.3 | 0.316 |
|  | `none` | 2.016 | 33.90 | 0.2149 | 0.1830 | 52.9 | 0.504 |
|  | `lab` | 0.662 | 37.84 | 0.2057 | 0.1781 | 56.6 | 0.376 |
|  | `ycc:5:3:histY0.8` | 0.620 | 37.84 | 0.2003 | 0.1731 | 57.5 | 0.377 |
|  | `ycc:4:3` | 0.602 | 38.02 | 0.1999 | 0.1714 | 58.9 | 0.331 |
|  | `ycc:3:3` | 0.592 | 38.35 | 0.1994 | 0.1607 | 64.1 | 0.317 |
| dark (lamp-lit interior) | bicubic | 0.722 | 40.52 | 0.2230 | 0.1357 | 84.0 | 0.478 |
|  | `none` | 2.322 | 33.50 | 0.2810 | 0.1889 | 46.3 | 2.180 |
|  | `lab` | 1.055 | 34.74 | 0.2690 | 0.1865 | 52.3 | 1.490 |
|  | `ycc:5:3:histY0.8` | 0.820 | 34.77 | 0.2562 | 0.1899 | 55.2 | 1.479 |
|  | `ycc:4:3` | 0.745 | 35.40 | 0.2521 | 0.1903 | 57.8 | 0.909 |
|  | `ycc:3:3` | 0.700 | 36.47 | 0.2459 | 0.1748 | 69.1 | 0.562 |
| street (sunny) | bicubic | 0.788 | 38.22 | 0.1847 | 0.0725 | 84.8 | 0.481 |
|  | `none` | 2.147 | 28.43 | 0.2733 | 0.1220 | 47.1 | 1.480 |
|  | `lab` | 1.210 | 30.04 | 0.2697 | 0.1159 | 53.2 | 0.994 |
|  | `ycc:5:3:histY0.8` | 0.914 | 30.07 | 0.2672 | 0.1143 | 55.0 | 0.991 |
|  | `ycc:4:3` | 0.840 | 30.25 | 0.2669 | 0.1153 | 56.4 | 0.730 |
|  | `ycc:3:3` | 0.779 | 30.90 | 0.2670 | 0.1145 | 64.6 | 0.591 |
| motion (fast pan) | bicubic | 0.761 | 35.83 | 0.2748 | 0.1603 | 79.8 | 0.515 |
|  | `none` | 3.019 | 28.73 | 0.3768 | 0.2060 | 46.2 | 6.973 |
|  | `lab` | 1.158 | 31.49 | 0.3617 | 0.2054 | 52.2 | 2.172 |
|  | `ycc:5:3:histY0.8` | 0.987 | 31.49 | 0.3573 | 0.2057 | 54.0 | 2.167 |
|  | `ycc:4:3` | 0.840 | 32.03 | 0.3512 | 0.2034 | 57.2 | 1.209 |
|  | `ycc:3:3` | 0.752 | 32.80 | 0.3426 | 0.1904 | 66.3 | 0.713 |

Paired with `lab` (4 clips, 3 seeds): `ycc:4:3` is better on ΔE00 at every scale, PSNR-Y (+0.40
dB), colour fringes (−0.41) and both flickers (−0.26, lf −0.46) on 4 of 4 clips, LPIPS on 3
(−0.0090), VMAF +4.0 on 4; DISTS −0.0014 (1 better, 1 worse). `ycc:3:3`: +1.10 dB, LPIPS −0.013
(3), DISTS −0.011 (4), VMAF +12.5 (4), flicker −0.58 / −0.71 (4). `ycc:5:3:histY0.8` is within
on PSNR-Y and flicker, VMAF +1.8.

- **The colour part of the 4K gap is the correction's:** uncorrected, ΔE00 lf is 2.0–3.0
  against bicubic's 0.6–0.8. `lab` leaves 0.66–1.21; with lightness and colour from the input
  at 3.2 px it is below bicubic's on all 4 clips (0.59–0.78), and the dark and fast clips'
  low-frequency flicker falls from 1.5 and 2.2 (`lab`) to 0.56 and 0.71 (bicubic 0.48, 0.52).
- **The rest is the model's rendering, not colour:** even `ycc:3:3` stays 2.6–7.4 dB below
  bicubic on PSNR-Y, 10–20 VMAF points below (64–69 against 74–85), worse on DISTS on all 4 and
  on LPIPS on 3 (on par on the close-up). The sunny street is furthest (−7.4 dB, DISTS 0.115
  against 0.073): fine, regular detail the model redraws. More from the input closes the gap
  further, at the price of detail: what's left is a question about the model at 4K for design,
  not about the colour correction.
- **Banding:** every model output adds less banding than bicubic (CAMBI added 0.000 against
  0.026 on average, 4 of 4).

### 4K tiles

The 4 shots, seed 42: the latents of the default run decoded again with tiles of 1536, 1024, 768
and 512 px (overlap 64), and dark and street run with the consumer cards' encode / decode tiles,
scored on their first 21 frames. An untiled 4K decode doesn't fit the card (about 134 GiB), so
each is compared with the default run's 2048-px decode: the offsets below are between two
tilings, both corrected. The worst 60×60 block per frame, range over the clips (the GT's flattest
third in brackets):

| Encode / decode tiles | `none` | `lab` | `ycc:5:3:histY0.8` | `ycc:4:3` | `ycc:3:3` |
|---|---|---|---|---|---|
| untiled / 1536 | 4.43–5.74 (3.37–4.14) | 0.38–0.65 (0.28–0.40) | 0.38–0.65 (0.28–0.41) | 0.16–0.31 (0.11–0.19) | 0.09–0.13 (0.07–0.11) |
| untiled / 1024 | 4.34–7.51 (2.78–5.55) | 0.46–0.76 (0.17–0.45) | 0.45–0.74 (0.17–0.44) | 0.14–0.31 (0.11–0.20) | 0.08–0.14 (0.07–0.08) |
| untiled / 768 | 5.11–9.53 (3.61–7.52) | 0.59–0.79 (0.38–0.49) | 0.57–0.78 (0.36–0.47) | 0.25–0.33 (0.15–0.21) | 0.11–0.15 (0.07–0.12) |
| untiled / 512 | 6.04–11.27 (3.78–10.05) | 0.69–0.95 (0.31–0.61) | 0.66–0.93 (0.30–0.61) | 0.25–0.42 (0.17–0.28) | 0.11–0.21 (0.09–0.13) |
| 1344 / 1024 (2 clips) | 5.15–9.53 (3.23–8.89) | 0.77–0.89 (0.35–0.56) | 0.78–0.89 (0.35–0.55) | 0.35–0.37 (0.18–0.21) | 0.15–0.16 (0.09–0.10) |
| 1024 / 768 (2 clips) | 8.85–10.71 (5.97–9.14) | 1.08–1.10 (0.75–0.76) | 1.11–1.12 (0.75–0.75) | 0.48–0.49 (0.33–0.35) | 0.20–0.23 (0.14–0.15) |

- **At 4K too, a finer split leaves about half of `lab`'s drift** (`ycc:3:3` a fifth): after
  `ycc:4:3`, the worst block moves by 0.14–0.42 levels at decode tiles from 1536 down to 512 px,
  0.35–0.49 with the consumer cards' encode tiles; after `lab`, 0.38–0.95 and 0.77–1.10.
  Against the 2048 decode, PSNR rises 0.5–0.8 dB over `lab`'s.
- **For the planner:** after `ycc:4:3`, no tiling measured here leaves half a level in the worst
  block of a frame, on flat areas a third of a level, at 1080p (5 clips) or at 4K (4): colour
  drift needs no floor on tile size above 512 px; time and memory set it. After `lab`, 512-px
  tiles and the 1024 / 768 recipe leave about one level.
- `ycc:5:3:histY0.8` leaves what `lab` leaves, as at 1080p.

### A float16 decode

DESIGN.md left the VAE decode's precision to the colour study's winner: a float16 decode
(the VAE file's own float16 weights, no autocast, float32 after the decoder) brought
low-frequency colour closer to the GT on 8 of 8 clips uncorrected (ΔE00 lf −0.05 to −0.09),
but only −0.01 to −0.02 after `lab`, for 4.4% more decode time
([numerics.md](numerics.md)). Seed 42's latents of the 8 clips at d1 decoded again in float16
(`colour_dump.py decode --vae-dtype fp16`): no non-finite value, decoder output in
[−1.23, 1.44] on anime-clean, every bfloat16 latent value exact in float16; against the
bfloat16 decode, 0.21 levels apart on average. Each variant on the float16 decode minus the same
variant on the bfloat16 one (seed 42, against that variant's 3-seed band):

| Variant | ΔE00 σ 0 | σ 4 | σ 16 | PSNR-Y | LPIPS | DISTS | Flicker | Flicker lf |
|---|---|---|---|---|---|---|---|---|
| `none` | −0.047 (6/0/2) | −0.051 (8/0/0) | −0.059 (8/0/0) | +0.06 (2/0/6) | +0.0001 (1/0/7) | +0.0001 (0/0/8) | −0.020 (1/0/7) | −0.013 (0/0/8) |
| `lab` | −0.007 (1/0/7) | −0.001 (1/0/7) | −0.001 (2/0/6) | +0.01 (1/0/7) | +0.0003 (0/0/8) | −0.0000 (1/0/7) | −0.017 (1/0/7) | −0.008 (1/0/7) |
| `ycc:5:3:histY0.8` | −0.009 (2/0/6) | −0.002 (0/0/8) | −0.001 (2/0/6) | +0.01 (1/0/7) | +0.0003 (1/0/7) | −0.0000 (1/0/7) | −0.017 (1/0/7) | −0.008 (1/0/7) |
| `ycc:4:3` | −0.010 (2/0/6) | −0.002 (1/0/7) | −0.001 (2/0/6) | +0.01 (0/0/8) | +0.0002 (0/0/8) | −0.0001 (0/0/8) | −0.015 (2/0/6) | −0.005 (1/0/7) |
| `ycc:3:3` | −0.009 (3/0/5) | −0.001 (3/0/5) | −0.000 (2/0/6) | +0.01 (0/0/8) | +0.0003 (0/0/8) | +0.0001 (0/0/8) | −0.013 (3/0/5) | −0.003 (2/0/6) |

- **After any correction, float16 adds nothing that matters:** at most 0.01 of ΔE00 unblurred
  and 0.002 after 4 px, within the band on 5–8 clips of 8 for every score. The uncorrected gain
  is the low band, which every correction replaces with the input's. numz's bfloat16 decode can
  stay, without float16's 4.4% and its overflow fallback.

### Cost, and what streams

`colour_variants.py bench`, on the RTX PRO 6000, 4 frames of 3840×2160 at once in the decode's
bfloat16 (the correction runs in the decode's second pass, or as the frames stream out):

| Variant | ms per 4K frame | torch peak per frame (GiB) |
|---|---|---|
| `none` | 1.2 | 0.19 |
| `lab` | 104.6 | 0.84 |
| `ycc:5:3:histY0.8` | 73.2 | 0.96 |
| `ycc:4:3` | 22.8 | 0.56 |
| `ycc:4:4` | 25.0 | 0.56 |
| `ycc:3:3` | 21.6 | 0.56 |

A split alone is a per-frame operation: each frame needs only its own decode and its reference
frame. The histogram step, numz's or its L\* blend alone, needs statistics pooled over the shot
before the first frame can be mapped, which is what makes seedvr2x's decode take two passes over
a temporary buffer (DESIGN.md, Colour correction). A plain split lets the decode stream with the
correction on; the input copy stays, since the reference is the input's frames.

## Step 4: crops for the eyes

[`scripts/colour_crops.py`](../scripts/colour_crops.py) cuts windows from seed 42's frames, the
GT then `lab`, `ycc:5:3:histY0.8`, `ycc:4:3` and `ycc:3:3` side by side at 1:1, each labelled
with its mean ΔE00 to the GT there, and the same with every panel's luma stretched by the GT
window's 1st–99th percentiles (small level and colour shifts become visible). Per clip, the two
best windows of each kind, one per frame: where `lab` and `ycc:3:3` differ most, the most strong
edges, the flattest areas that aren't black, the most skin tones, the most sky. 1080p: anime-clean,
anime-dark, cartoon-bright, anime-sky, live-slow and live-vfx at d1, anime-clean, anime-sky and
live-slow at d2 (480×270 windows); 4K: the 4 shots (640×360). The user's verdict is pending.

## Decision brief

**Recommendation.** seedvr2x's colour correction becomes a split in BT.709 Y'CbCr without numz's
histogram step: lightness from the input below 4 à-trous stages (σ 6.5 px), colour below
the stage nearest 1.6 source pixels, 2 + log2(factor) rounded on a log scale (3 at ×1.5 and ×2,
σ 3.2 px; 4 at ×3 and ×4, σ 6.5 px), the rest from the model, with the float32 input copy as
the reference. At ×2 it is `split:ycc:4:3`. It replaces `lab` as
`--color-correction`'s default.

What it changes in DESIGN.md:
- **The decode streams.** Without a histogram there are no shot-pooled statistics: no second
  pass, no temporary bfloat16 buffer (17.9 GB per minute of 1080p shot, four times that at 4K),
  a shot's first frames out as they are decoded. The input copy stays: the reference needs the
  input frames at decode time.
- **Cost:** 22.8 ms and 0.56 GiB per 4K frame on the GPU, against `lab`'s 104.6 ms and 0.84 GiB.
- **The upscale factor sets the colour scale** (the correction needs it; the plan has it).
- **The tiles:** a split leaves half of what `lab` leaves of the VAE tiles' drift, at every tile
  size at 1080p and 4K: under half a level in the worst block of a frame down to 512-px tiles.
  Colour gives the planner no floor on tile size above 512 px.
- **A float16 decode isn't needed:** after the correction it changes nothing beyond the seed
  band; numz's bfloat16 decode stays.
- `test_lab.py`'s thresholds follow the new correction (DESIGN.md's order for a winner).

**Evidence** (each variant paired with `lab` frame by frame, verdicts beyond the seed band:
better / worse / within):
- 1080p, d1 (8 clips, 3 seeds): ΔE00 −0.30, −0.46, −0.29 after blurs of 0, 4 and 16 px (8 / 0 /
  0); PSNR-Y +0.63 dB (4 / 0 / 4); LPIPS −0.0096 (7 / 0 / 1); DISTS −0.0023 (2 / 2 / 4); flicker
  −0.44 and lf −0.76 (6–8 better); colour fringes −0.46 (7 better); detail kept (Laplacian
  variance +7, no clip lower).
- The heavier degradation (7 clips, 2–3 seeds): ΔE00 σ 4 −0.27 (7 / 0 / 0), PSNR-Y +0.65
  (5 / 0 / 2), LPIPS −0.0044 (4 / 1 / 2), flicker lf −0.70 (7 / 0 / 0); DISTS +0.0011 (1 / 4 / 2:
  +0.003 to +0.005 on the four worse).
- Through the default master, `yuv420p10le` (8 clips, 3 seeds): ΔE00 σ 4 −0.45 (8 / 0 / 0),
  PSNR-Y +0.63, LPIPS −0.008 (5 / 1 / 2), DISTS −0.003 (1 / 2 / 5), VMAF v1 +3.7 (7 / 0 / 1), no
  banding added (CAMBI within), fringes −0.68.
- Clip B's window joins: the low-frequency step 40% lower (0.046 → 0.028), the held-frame step 5%.
- Tiles at 1080p (5 clips, decode tiles 1280 to 512 px, encode tiles 1344 to 512): the worst
  60-px block left after correction 0.04–0.77 levels against `lab`'s 0.10–1.39; at 4K (4 shots,
  decode tiles 1536 to 512, 2 encode recipes, against the 2048-px decode) 0.14–0.49 against
  0.38–1.10.
- ×4 (5 clips, 2–3 seeds), at `ycc:4:4`: ΔE00 σ 4 −0.18 (5 / 0 / 0), PSNR-Y +0.55 (3 / 0 / 2),
  LPIPS −0.0020 (4 / 0 / 1), DISTS +0.0009 (2 / 2 / 1), flicker lf −0.58 (5 / 0 / 0). The ×2
  scale at ×4 (`ycc:4:3`) costs DISTS on 3 of 5.
- ×1.5 and ×3 (3 clips, 2 seeds), the rule's stage: at ×1.5 `ycc:4:3` LPIPS −0.0066 (3 / 0 / 0),
  DISTS +0.0006 (1 / 1 / 1); at ×3 `ycc:4:4` LPIPS −0.0017 (2 / 0 / 1), DISTS −0.0004 (1 / 1 / 1);
  the stage below costs DISTS on 2 of 3, the stage above colour fringes or LPIPS.
- 4K (4 live-action shots, 3 seeds, decode tiles of 2048): ΔE00 better at every scale on 4 of 4,
  PSNR-Y +0.40 dB (4 / 0 / 0), LPIPS −0.0090 (3 / 0 / 1), VMAF +4.0 (4 / 0 / 0, seeds 42 and 43),
  flicker −0.26 and lf −0.46 (4 / 0 / 0), DISTS −0.0014 (1 / 1 / 2).
- The histogram step is what costs `lab` colour (numz's a\*b\* matching moves colour at every
  scale); its L\* blend matters only with lightness at 13 px (flicker), and adds nothing at 6.5
  px (d1 +0.1 dB and flicker −0.04; clip B and the master the same).

**Confidence:** high on fidelity (colour, flicker, the joins, the tiles: better on 7–8 of 8
clips and every set: 1080p, ×4 with the scale rule, 4K, through the master). Moderate on the
perceptual side, where DISTS is mixed: within on half the clips at d1 (2 better, 2 worse), and
0.003–0.005 worse on 4 of 7 compressed inputs, as every split is.

**Caveats:**
- Fidelity to a ground truth is not quality: a correction that takes more from the input scores
  better on it. DISTS and the eyes are the guard on the model's detail.
- One model (7B fp16) with numz's numerics; single shots of 45 frames, one batch; 8 clips at
  1080p (animation and live action), 4 at 4K from one film; ×4 on 5.
- Factors tested: ×1.5, ×2, ×3 and ×4; the rule in between (×2.25 for 480p to 1080p, say) is
  an interpolation.
- The 4K gap to bicubic that remains is the model's (below), not the correction's.

**The finer option, for the eyes:** `split:ycc:3:3` (lightness at 3.2 px too) gives the most
fidelity before detail goes: PSNR-Y +1.69 dB, VMAF +9.7 in the master, flicker −1.13, the tiles'
drift a quarter of `lab`'s, at 4K VMAF +12.5 and DISTS −0.011. It costs 15% of the sky's fine
texture, and DISTS is mixed on compressed inputs (3 of 7 clips worse, 3 better). The crops put
`lab`, `ycc:5:3:histY0.8`, `ycc:4:3` and `ycc:3:3` side by side.

**4K's quality (for DESIGN.md's open question):** with the correction, the 4K output's
low-frequency colour reaches bicubic's (ΔE00 lf 0.59–0.78 with `ycc:3:3`, below bicubic's on 4
of 4; 0.60–0.84 with `ycc:4:3`, at most 0.08 above it; bicubic 0.61–0.79), and its flicker
comes close. The rest of the gap is the model's: PSNR-Y 2.6–7.4 dB below bicubic, VMAF 10–20
points below, DISTS worse on 4 of 4 and LPIPS on 3 of 4, even with `ycc:3:3`.

## Caveats

- **Fidelity is not quality** ([numerics.md](numerics.md#caveats)): every score but DISTS and
  the detail measure rewards taking more from the input, which the GT is closer to in every
  band (step 0). DISTS, the Laplacian variance and the eyes are the guard on the model's detail.
- One model (7B fp16) with numz's numerics; single shots of 45 frames in one batch; 8 clips at
  1080p (animation and live action), 7 at the heavier degradation, 5 at ×4, 4 at 4K from one
  film; 3 at ×1.5 and ×3. Seeds: 3 at d1, 2–3 elsewhere.
- Factors tested: ×1.5, ×2, ×3, ×4 (×1.5 and ×3 on 3 clips, 2 seeds).
- The variants are scored on the decode in float32, as seedvr2x keeps it; numz's own `lab`
  masters (step 0) were clamped bfloat16.

## Reproduce

Step 0 (CPU, numerics.md's masters):

```bash
S=scripts; C=/path/to/clips; Q=/path/to/numerics-runs
python3 $S/colour_diag.py selftest       # CIEDE2000 against published pairs and skimage, bands, oracle
python3 $S/colour_diag.py scan --clip anime-clean-d1 --gt $C/anime-clean.gt.mkv \
  --bicubic $C/anime-clean.d1.bicubic.mkv --oracle-seed 42 --json diag/anime-clean-d1.json \
  --run 42 $Q/q1-anime-clean-d1-def-s42.mkv $Q/q1-anime-clean-d1-def-s42.cc-lab.mkv \
  --run 43 $Q/q1-anime-clean-d1-def-s43.mkv $Q/q1-anime-clean-d1-def-s43.cc-lab.mkv \
  --run 1234 $Q/q1-anime-clean-d1-def-s1234.mkv $Q/q1-anime-clean-d1-def-s1234.cc-lab.mkv
python3 $S/colour_diag.py check diag/anime-clean-d1.json \
  --fr lab.s42=metrics/d1-lab/anime-clean-d1.def+lab.s42.json    # sigma 4 against fr_metrics.py
python3 $S/colour_diag.py report diag/*.json > report.md
python3 $S/colour_diag.py histtest --clip anime-clean-d1 --gt $C/anime-clean.gt.mkv \
  --none $Q/q1-anime-clean-d1-def-s42.mkv --bicubic $C/anime-clean.d1.bicubic.mkv
```

Steps 1 to 3. Inputs: `colour_clips.py` makes the ×4 inputs and the d2 ones fr_clips.py didn't
make; the dumps come from numz runs wrapped by `colour_dump.py` (numz's checkout as the working
directory, its venv; GPU work under the shared lock); the scoring runs on the CPU, with
`COLOUR_BASELINE` pointing at a copy of seedvr2x's `runtime/colour.py` and `MEAS_SCRIPTS` at
fr_metrics.py, ffv1_out.py and stitch_metrics.py:

```bash
S=scripts; C=/path/to/clips; D=/path/to/dumps; N=/path/to/numz
python3 $S/colour_clips.py make $C/anime-clean --factor 4 --out clips              # x4 input, d1 recipe
python3 $S/colour_clips.py make $C/live-slow --factor 2 --degrade d2 --out clips   # fr_clips.py's d2
# one dump per run (seed 42 with the encoder input and float32 reference, other seeds without)
python3 $S/bench.py run colour-anime-clean-d1-s42 --seedvr2-dir $N --wrap $S/colour_dump.py \
  --env COLOUR_DUMP=$D/anime-clean-d1/s42 --env COLOUR_DUMP_INPUTS=1 -- $C/anime-clean.d1.lr.mkv \
  --output out/ --dit_model seedvr2_ema_7b_fp16.safetensors --resolution 1080 --batch_size 45 \
  --load_cap 45 --color_correction none --seed 42
python3 $S/colour_dump.py verify $D/anime-clean-d1/s42 --master numerics/q1-anime-clean-d1-def-s42.mkv
(cd $N && python3 $S/colour_dump.py decode $D/anime-clean-d1/s42 --out check --check)  # = the CLI's decode
(cd $N && python3 $S/colour_dump.py decode $D/anime-clean-d1/s42 --out $D/anime-clean-d1/s42/dec \
  --tile 1280:64 --tile 1024:64 --tile 768:64 --tile 512:64)                          # other decode tiles
(cd $N && python3 $S/colour_dump.py decode $D/anime-clean-d1/s42 --out $D/anime-clean-d1/s42/dec16 \
  --untiled --vae-dtype fp16)                                                          # float16 decode
python3 $S/colour_variants.py selftest
python3 $S/colour_variants.py bench lab,split:ycc:5:3:histY0.8,split:ycc:4:3,split:ycc:3:3   # GPU, per 4K frame
# scores: one JSON per clip, content and variant
python3 $S/colour_eval.py score --clip anime-clean-d1 --gt $C/anime-clean.gt.mkv \
  --ref f32=$D/anime-clean-d1/s42/ref_f32.pt --ref bf16=$D/anime-clean-d1/s42/enc_bf16.pt \
  --content s42=$D/anime-clean-d1/s42/decode.pt \
  --variants none,lab,lab@bf16,split:ycc:5:3:histY0.8,split:ycc:4:3,split:ycc:3:3 \
  --out eval/anime-clean-d1 --lpips --dists-every 9
python3 $S/colour_eval.py score ... --content s42=decode.pt --content s42-d512=dec/decode-512-64.pt \
  --untiled s42 --out eval/anime-clean-d1-tiled                                        # tiles
python3 $S/colour_eval.py render --clip anime-clean-d1 --gt $C/anime-clean.gt.mkv --ref f32=... \
  --content s42=... --variants lab,split:ycc:4:3 --out masters --pix-fmt yuv420p10le,gbrp16le
python3 $S/colour_eval.py score --clip anime-clean-d1 --gt $C/anime-clean.gt.mkv \
  --master lab=masters/anime-clean-d1.s42.lab~f32.yuv420p10le.mkv --master-content s42 \
  --out eval-yuv/anime-clean-d1 --vmaf --lpips --dists-every 3                         # the default master
python3 $S/colour_eval.py vmaf --clip face-d1 --gt $C4/face.gt.mkv --master bicubic=$C4/face.d1.bicubic.mkv \
  --master-content s42,s43 --out eval-vmaf4k/face-d1                                   # VMAF v1 alone
python3 $S/colour_eval.py stitch --ref f32=$D/clipb/s42/ref_f32.pt --onebatch $D/clipb/s42/decode.pt \
  --windows $D/clipb/s42-lat6-2/decode.pt --input clipb-source.mkv --skip 20 --latent 6:2 \
  --variants lab,split:ycc:4:3 --out eval/clipb-stitch                                 # clip B's joins
python3 $S/colour_eval.py summary eval/*-d1 --baseline lab@f32                          # paired tables
python3 $S/colour_eval.py summary eval/*-d1-tiled --tables tiles
python3 $S/colour_eval.py summary eval/*-d1 --tables groups                            # the latent grid
python3 $S/colour_crops.py --clip anime-clean-d1 --gt $C/anime-clean.gt.mkv \
  --ref $D/anime-clean-d1/s42/ref_f32.pt --content $D/anime-clean-d1/s42/decode.pt \
  --variants lab,split:ycc:5:3:histY0.8,split:ycc:4:3,split:ycc:3:3 --out crops/anime-clean-d1
```

# Colour correction beyond numz's `lab`

> Status: **steps 0 to 5 done** (step 4: the user's eyes, 2026-10-06; step 5, baton 1 of the 4K
> relay, 2026-10-06, with the user's eyes on its crops given the same day: the 0.25 / 0.5
> recommendation is withdrawn); **step 6 done** (baton 2, the sharp 7B, night of
> 2026-10-06, with the user's eyes on its crops on 2026-10-07). Step 0 (the diagnosis) ran on the CPU from
> [numerics.md](numerics.md)'s masters; steps 1 to 3 and 5 on dumps of 7B fp16 runs (1080p, ×1.5
> to ×4, the heavier degradation, VAE tiles, 4K on 17 shots), scored on the CPU. The study answers
> "Beyond numz's `lab`" in [DESIGN.md](../../seedvr2x/DESIGN.md#beyond-numzs-lab) and, with steps
> 5 and 6, part of "4K output's quality" ([To measure](../../seedvr2x/DESIGN.md#to-measure)); its
> decision brief is at the end.

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
  for the eyes; the user can't tell `4:3` and `3:3` apart (step 4).
- **4K:** the correction closes the colour part of the gap to bicubic (with `ycc:3:3`,
  low-frequency ΔE00 below bicubic's on 4 of 4 shots of the first film, 9 of 13 in step 5). The
  rest, 2.7–7.3 dB of PSNR-Y, 10–20 points of VMAF, DISTS on 4 of 4, is the model's 4K
  rendering, not colour, and clean sources keep it (step 5, 13 more shots): with `split:ycc:4:3`
  the model trails bicubic on clean digital live action (PSNR-Y −5.8 dB, VMAF −19) and on
  animation made in 4K (−7.6 dB, −25) as on grain, while adding less banding. It over-renders:
  3.4–3.7 times the GT's fine detail on clean sources (Laplacian variance; bicubic 0.15–0.17).
- **A detail strength wins on the metrics, not by eye** (step 5): the split's lightness blended
  with the input's beats bicubic on LPIPS and DISTS (A = 0.25 at 4K, 9 of 9 clean shots; 0.5 at
  1080p, 7 of 8). The user's eyes want the model's redraw on structure, not its invented texture,
  by an amount that changes with the content: no single value fits, and the default stays the
  split alone. A user scale is the user's proposal, not a finding.
- **The sharp 7B over-renders less, and the eyes prefer it** (step 6): at 4K it keeps 0.53–0.88
  of the 7B's fine detail by kind of source (the same uncorrelated texture, 7–26% less energy in
  the finest band), closer to the GT by PSNR-Y and VMAF on 14 of 17 shots, with no guard worse.
  The user's eyes prefer it on 27 of 75 windows, the 7B on 8, and see no difference on the rest:
  the first film's skin loses its reptilian texture, light artefacts and blockiness go. The 7B
  keeps the thinnest cel lines sharper; a heavier degradation (d2) tames neither model, and from
  it the sharp over-renders clouds more. Recommended: the sharp 7B as the default model, with the
  split alone.
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
- **The rest is the model's rendering, not colour:** even `ycc:3:3` stays 2.7–7.3 dB below
  bicubic on PSNR-Y, 10–20 VMAF points below (64–69 against 74–85), worse on DISTS on all 4 and
  on LPIPS on 3 (on par on the close-up). The sunny street is furthest (−7.3 dB, DISTS 0.115
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
- **For the planner:** after `ycc:4:3`, decode tiles down to 512 px and the consumer cards' two
  recipes leave at most half a level in the worst block of a frame, about a third on flat areas,
  at 1080p (5 clips) and on the first film at 4K (4 shots). On painted 4K animation (step 5)
  512-px decode tiles leave up to 0.77 in textured blocks, the flattest third at most 0.15.
  512-px encode tiles leave more at 1080p: 0.23–0.56 with 512-px decode tiles and 0.32–0.77
  decoded untiled (the flat third 0.13–0.27), still about half of `lab`'s 0.60–1.39. Colour drift
  needs no floor on tile size down to 512 px; time and memory set it. With a detail strength
  (step 5) the lightness drift left scales by A. After `lab`, 512-px tiles and the 1024 / 768
  recipe leave about one level on the first film, up to 1.6 on painted animation.
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
live-slow at d2 (480×270 windows); 4K: the 4 shots (640×360).

The user's verdict (2026-10-06), on four crops:

- **`ycc:4:3` is never worse than `lab`, and better where `lab`'s colour noise shows:** on the
  cartoon in fast motion (`cartoon-bright-d1-diff-f37`), `lab` is the worst panel (blotches, an
  elbow too white); on the compressed live action (`live-slow-d2-diff-f41`), `ycc:4:3` is the
  cleanest and sharpest, sharper than the GT (the model redraws detail the camera frame never had).
- **`ycc:4:3` and `ycc:3:3` can't be told apart** on any of them: a slight leaning to `3:3` on the
  cartoon, to `4:3` on the sky (`anime-sky-d1-diff-f44`, a hair more detail). Nothing wrong seen in
  `lab` on the sky.
- **The 4K close-up's skin (its skin crop) is over-textured in every panel**, far beyond the GT:
  the model's own rendering at 4K, which no correction changes (see 4K above).

## Step 5: 4K on clean and grainy sources, and a detail strength

DESIGN.md's open question "4K output's quality": on the first film the model trails bicubic on
every metric but banding, whatever the correction (step 3's 4K, and
[numerics.md](numerics.md#4k-output-with-colour-correction-against-bicubic)). That film is a
grainy restoration. Step 5 asks whether the verdict holds on clean native 4K, and what the colour
side can close: lightness at 3.2 px (`ycc:3:3`) and a detail strength, which it then scores at
1080p too.

### The clips and runs

13 shots of measurement's batch A ([numerics.md](numerics.md#clips)), from three sources:
- `digital-*`, clean digital live action (an IMAX film's native 4K master through a fixed tone
  map), cropped to 3840×2016: `sunrise` (a smooth sky), `cockpit` (clean skin in a helmet),
  `space`, `jet` (29 frames), `dogfight` (motion blur);
- `sollevante-*`, animation made natively in 4K, *Sol Levante* (Netflix / Production I.G,
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)), 3840×2160: `painted`, `line` (line
  art on flat colour), `action`, `dark` (41 frames);
- `cel4k-*`, a 4K scan of a grainy cel film, cropped to 3840×2048: `detail`, `motion`, `night`,
  `flat`.

Each is ×2 from the d1 recipe (×½ Mitchell, x264 CRF 20, 8-bit), 45 frames unless said, with the
first film's setup: 7B fp16, numz `4490bd1`, encode untiled, decode tiles of 2048 px (overlap 64),
`--color_correction none` with the raw decode dumped and every variant applied to that decode.
Seed 42 on all 13, seed 43 too on the four `sollevante-*`, `sunrise` and `space` (VMAF and CAMBI
of `ds4_0.75` and `ds4_0.25` at seed 42 only); on the other 7 a verdict rests on one seed, where
the bootstrap interval alone decides. The first film's 4 shots come along with their seeds (42, 43
and 1234; the detail strengths at 42 and 43, or 42 alone).

Scores as in step 3 (`colour_eval.py score`): LPIPS on every frame, DISTS on every 9th (5 frames a
clip, so its figures compare variants here, never with numerics.md's), VMAF v1 with the 2160p
model, chosen by the GT's width too (3840-wide crops), and CAMBI added, both from `gbrp16le`
masters. Bicubic, the d1 input upscaled ×2 with Catmull-Rom, scored by `colour_eval.py` from its
master, reproduces measurement's figures exactly (`sunrise`: PSNR-Y 45.53, VMAF 68.62, CAMBI
added 1.651). Verdicts against bicubic come from `summary --versus bicubic` (paired by frame,
beyond the variant's own seed spread), against `lab` from `--baseline lab@f32` as before.
`ds…` below is short for `split:ycc:4:3:ds…`.

Against `lab`, the split's 4K verdicts hold on the 13 shots: ΔE00 lf −0.33, PSNR-Y +0.57 dB,
VMAF +5.3 and low-frequency flicker −0.58 (13 / 0 / 0 each), LPIPS −0.0086 (10 / 3 / 0); DISTS
−0.0012 (6 / 6 / 1).

### Against bicubic, by kind of source

Variant − bicubic, mean over the shots (better / worse / within); detail: the luma Laplacian
variance over the GT's, geometric mean over the shots (bicubic's in brackets):

| Source (shots) | Variant | PSNR-Y (dB) | VMAF | LPIPS | DISTS | CAMBI added | Detail ÷ GT's |
|---|---|---|---|---|---|---|---|
| digital live action (5) | `ycc:4:3` | −5.84 (0/5/0) | −19.26 (0/5/0) | +0.0195 (0/4/1) | +0.0215 (1/4/0) | −2.209 (5/0/0) | 3.42 (0.15) |
| | `ycc:3:3` | −5.03 (0/5/0) | −12.32 (0/5/0) | +0.0121 (1/3/1) | +0.0148 (1/4/0) | −2.227 (5/0/0) | 3.38 |
| Sol Levante (4) | `ycc:4:3` | −7.57 (0/4/0) | −25.32 (0/4/0) | +0.0224 (0/3/1) | −0.0051 (3/1/0) | −0.963 (4/0/0) | 3.70 (0.17) |
| | `ycc:3:3` | −6.67 (0/4/0) | −15.66 (0/4/0) | +0.0159 (1/2/1) | −0.0078 (3/1/0) | −0.963 (4/0/0) | 3.67 |
| cel scan (4) | `ycc:4:3` | −9.36 (0/4/0) | −25.30 (0/4/0) | +0.1563 (0/4/0) | +0.0910 (0/4/0) | −0.016 (3/0/1) | 7.30 (0.54) |
| | `ycc:3:3` | −8.06 (0/4/0) | −16.22 (0/4/0) | +0.1573 (0/4/0) | +0.0919 (0/4/0) | −0.016 (3/0/1) | 7.11 |
| first film (4) | `ycc:4:3` | −4.98 (0/4/0) | −23.15 (0/4/0) | +0.0471 (0/3/1) | +0.0416 (0/4/0) | −0.026 (4/0/0) | 1.96 (0.12) |
| | `ycc:3:3` | −4.27 (0/4/0) | −14.68 (0/4/0) | +0.0432 (0/3/1) | +0.0316 (0/4/0) | −0.027 (4/0/0) | 1.94 |

Bicubic's own, for scale: PSNR-Y 37.2–47.4 dB, VMAF 68.6–98.3 on the 13 shots.

- **Clean sources don't change the verdict.** With the split, the model trails bicubic on clean
  digital live action and on animation made in 4K as on grain: PSNR-Y and VMAF on 17 of 17 shots
  (VMAF 4.7 points below on `sollevante-dark` to 46 on `cel4k-motion`), LPIPS on 14 (within on
  `space`, `sollevante-line` and the close-up). DISTS is worse on 13, better on three of the four
  Sol Levante shots (`painted`, `line`, `action`) and on `sunrise`.
- **Less banding than bicubic** on 16 of 17 (within on `cel4k-detail`, where neither adds any),
  most where the 8-bit input bands skies and dark gradients: CAMBI added 0.00–0.16 against
  bicubic's 1.11–4.61 on `sunrise`, `cockpit`, `space`, `jet`, `sollevante-line` and `-dark`.
- **Low-frequency colour:** `ycc:3:3` is at bicubic's level (ΔE00 lf −0.009 to +0.018 on
  average, lower on 9 of 13), the split 0.02–0.11 above it (12 of 13).
- **The model over-renders.** The split keeps 3.4–3.7 times the GT's Laplacian variance on the
  clean sources and 7.3 on the grainy scan (the first film 2.0), where bicubic keeps 0.12–0.54:
  fine detail the GT doesn't have, as the user saw on the close-up's skin (step 4).
- **`ycc:3:3` closes a little of the gap, not the over-rendering:** against the split, PSNR-Y
  +0.99 dB and VMAF +8.4 (13 / 0 / 0), DISTS −0.0032 (9 / 2 / 2; on digital live action −0.0068,
  4 of 5), LPIPS −0.0045 (9 / 2 / 2), but the detail ratio falls by 1–3% only: the excess sits
  finer than 3.2 px.

### Where the model's excess sits

`colour_bands.py` splits the luma Y' into the à-trous bands (band k between k − 1 and k stages:
below 0.7, 0.7–1.6, 1.6–3.2, 3.2–6.5 and 6.5–13 px) and measures in each the RMS of the split's
output, the GT, bicubic and the reference, and their correlation with the GT's band, on every
third frame, over the picture (inside the bars) and over the GT's flattest third of 60-px blocks.
Scanned: the 13 batch A shots, the first film's 4 and 4 clips at 1080p (anime-clean,
cartoon-bright, live-slow and live-vfx at d1). Per kind, whole picture, means over the shots:

| Source (shots) | below 0.7 px | 0.7–1.6 px | 1.6–3.2 px | 3.2–6.5 px | Least-squares A (K = 4) | Energy-match A (K = 4) |
|---|---|---|---|---|---|---|
| digital live action (5) | 1.99 · 0.15 | 1.64 · 0.41 | 1.48 · 0.68 | 1.22 · 0.90 | 0.11 | 0.23–0.28 (`sunrise` 0.97) |
| Sol Levante (4) | 1.99 · 0.32 | 1.74 · 0.66 | 1.50 · 0.88 | 1.21 · 0.96 | 0.13 | 0.22–0.32 |
| cel scan (4) | 2.71 · 0.35 | 2.06 · 0.69 | 1.52 · 0.88 | 1.18 · 0.96 | 0.10 | 0.09–0.24 |
| first film (4) | 1.58 · 0.09 | 1.31 · 0.30 | 1.30 · 0.73 | 1.15 · 0.95 | 0.08 | 0.20–0.93 |

Each band: the split's RMS over the GT's · its correlation with the GT. The two A columns are the
detail strength's (below) at K = 4: the least-squares one (PSNR's optimum, the mean over the
shots), and the one at which the output's fine bands meet the GT's RMS (range over the shots).

- **The reference is the bicubic upscale, in effect:** its RMS ratio equals bicubic's to 2
  decimals in every band of all 21 clips scanned (its correlation with the GT too, on the 8
  measured in full), and the reference's lightness (K = 4 at A = 0, below) has bicubic's PSNR-Y
  within 0.02 dB. Lightness detail taken from the reference is bicubic's.
- **At 4K the model's two finest bands are mostly not the GT's:** they correlate 0.09–0.35 and
  0.30–0.69 with it on average per kind (bicubic's correlate more on the first film: 0.09–0.77
  against the model's 0.02–0.61), from 3.2 px up 0.90–0.96. At 1080p they correlate more on
  three of the four clips scanned: 0.42–0.54 and 0.76–0.84 (live-vfx 0.04 and 0.37).
- **They also carry more energy than the GT's:** 1.3–2.7 times its RMS in the three finest bands
  on average per kind, most on the grainy scan (2.7, 2.1 and 1.5); the sky of `sunrise` (0.89 and
  0.88 in the two finest bands) and the first film's fast pan are the exceptions. On the first
  film, the sunny street 2.4, 2.0 and 1.6 times the GT's RMS, the dark interior 1.9, 1.4 and 1.4;
  on the grainy close-up the model's texture is at about the grain's energy (1.05–1.17 times)
  but uncorrelated with it (0.02 and 0.12 below 1.6 px): texture as strong as the grain, in other
  places.
- **Energy matching can't see that kind of over-texture:** it keeps 0.09–0.32 of the model's
  detail on 12 of the 13 batch A shots, but 0.97 on `sunrise`, whose sky's fine bands are already
  at the GT's level, and 0.84 and 0.93 on the close-up and the fast pan, where the model's
  texture has the GT's energy without its pattern.
- **Fidelity wants almost none of it:** the least-squares A is 0 for K = 1 on all 21 clips,
  negative before clipping for K = 2 on average on every kind (0 after clipping on 19 of 21;
  `sollevante-dark` 0.06, cartoon-bright 0.14), and for K = 4 0.10–0.13 on average on every
  batch A kind (the first film 0.08; per shot 0.03–0.23; 0.10–0.25 at 1080p). How much of the
  model's detail to keep is for the perceptual scores and the eyes.

### The detail strength

`split:ycc:4:3:ds<K>_<A>` (`colour_variants.py`, ours) adds (1 − A) · hp_K(d) to the split's
lightness, with d = reference − decode in Y' and hp_K(x) = x − low_K(x), the part of x finer than
K à-trous stages (K ≤ 4). Colour stays the split's, and each frame is still corrected alone.
A = 1 is the split; A = 0 takes that band from the reference.

- **At K = 4 it is a blend of two lightnesses.** With c and r the decode's and the reference's Y',
  the split's lightness is L_split = hp_4(c) + low_4(r), so L_ref − L_split = hp_4(r) − hp_4(c) =
  hp_4(d), and `ds4_A` gives L_split + (1 − A) · hp_4(d) = A · L_split + (1 − A) · L_ref: one
  multiply-add per pixel, no filtering beyond the split's own. So the tiles' drift that the split
  leaves in the lightness scales by A exactly (before the clamp), the reference having none; and
  A = 0 is bicubic's lightness, in effect, with the split's colour.
- **K = 2 (below 1.6 px only),** the band diagnostic's first pick, does worse than K = 4 on DISTS:
  on digital live action `ds2_0.5` and `ds2_0` stay worse than bicubic (+0.0056 and +0.0078,
  worse on 3 of 5), where `ds4_0.5` and `ds4_0.25` are better (−0.0046, 3 of 5; −0.0112, 5 of
  5); on the first film `ds2_0` is worse than the split itself (+0.0063, 3 of 4). The excess
  reaches 1.6–3.2 px (above), which K = 2 leaves to the model.

Along K = 4: where LPIPS and DISTS are lowest per shot (seed 42, the mean over seeds 42 and 43
picking the same A on the 6 shots that have both; A = 1, 0.75, 0.5, 0.25, and bicubic standing in
for A = 0), then A = 0.25 and 0.5 against bicubic, as in the table above:

| Source (shots) | Lowest LPIPS at A | Lowest DISTS at A | Variant | PSNR-Y (dB) | VMAF | LPIPS | DISTS | CAMBI added | Detail ÷ GT's |
|---|---|---|---|---|---|---|---|---|---|
| digital live action (5) | 0.25 ×4, 0.5 | 0.25 ×3, 0.5, 0.75 | `ds4_0.25` | −0.24 (1/4/0) | +2.30 (4/1/0) | −0.0214 (5/0/0) | −0.0112 (5/0/0) | −1.617 (5/0/0) | 0.35 |
| | | | `ds4_0.5` | −1.98 (0/5/0) | −3.00 (1/3/1) | −0.0137 (5/0/0) | −0.0046 (3/2/0) | −2.048 (5/0/0) | 0.96 |
| Sol Levante (4) | 0.25 ×4 | 0.25 ×2, 0.5, 0.75 | `ds4_0.25` | +0.13 (1/3/0) | +4.14 (4/0/0) | −0.0406 (4/0/0) | −0.0378 (4/0/0) | −0.681 (4/0/0) | 0.42 |
| | | | `ds4_0.5` | −2.45 (0/4/0) | −2.79 (1/3/0) | −0.0310 (4/0/0) | −0.0419 (4/0/0) | −0.903 (4/0/0) | 1.09 |
| cel scan (4) | bicubic ×4 | bicubic ×3, 0.25 | `ds4_0.25` | −0.83 (2/2/0) | +2.16 (3/1/0) | +0.0230 (0/4/0) | +0.0153 (1/3/0) | −0.015 (3/0/1) | 1.03 |
| | | | `ds4_0.5` | −4.08 (0/4/0) | −5.13 (2/2/0) | +0.0588 (0/4/0) | +0.0405 (0/4/0) | −0.016 (3/0/1) | 2.33 |
| first film (4) | bicubic ×3, 0.25 | bicubic ×4 | `ds4_0.25` | −0.24 (0/4/0) | +2.04 (3/1/0) | +0.0030 (0/2/2) | +0.0092 (0/4/0) | −0.022 (4/0/0) | 0.22 |
| | | | `ds4_0.5` | −1.65 (0/4/0) | −4.35 (0/4/0) | +0.0189 (0/4/0) | +0.0209 (0/4/0) | −0.026 (4/0/0) | 0.56 |

- **The split (A = 1) is never the best point at 4K:** on all 17 shots LPIPS and DISTS are
  lowest at A = 0.75 or below, or at bicubic. Against the split on batch A, `ds4_0.25` gains
  PSNR-Y +7.2 dB, VMAF +25.8 and LPIPS −0.076 (13 / 0 / 0 each), DISTS −0.046 (11 / 2 / 0);
  `ds4_0.5` VMAF +19.4, LPIPS −0.059 and DISTS −0.037 (13 / 0 / 0 each).
- **Clean sources (9 shots): A = 0.25 is the lowest-LPIPS point on 8** (`space`: 0.5) **and
  beats bicubic on LPIPS and DISTS on 9 of 9** (−0.030 and −0.023 on average), with VMAF above
  bicubic's on 8 (+3.1 on average; `cockpit` 89.6 against 85.9, `sollevante-line` 87.4 against
  83.8; `dogfight` 0.1 below) at bicubic's PSNR-Y (−0.07 dB on average: above it on `sunrise` and
  `sollevante-dark`, within 0.6 dB on 6 more, the painted landscape 1.2 dB below). A = 0.5 beats
  bicubic on LPIPS on 9 of 9 and DISTS on 7 (−0.021 both), 2.2 dB and 2.9 VMAF points below it,
  and brings the detail to the GT's own level (0.96 and 1.09 times its Laplacian variance,
  against 0.35 and 0.42 at A = 0.25 and bicubic's 0.15 and 0.17).
- **Grain (the cel scan, the first film): bicubic stays ahead** on LPIPS (+0.013 at A = 0.25:
  worse on 6 of 8, within on 2) and DISTS (+0.012, worse on 7 of 8). A = 0.25 is the model's
  closest output, VMAF above bicubic's on 6 of 8 (+2.1); on `cel4k-detail` it beats bicubic on
  DISTS too (0.0390 against 0.0410; VMAF 91.1 against 87.3).
- **What it costs.** Banding comes back as A falls, still below bicubic's: CAMBI added at
  A = 0.25 on `cockpit` 1.63 (the split 0.14, bicubic 3.58), on `sollevante-line` 0.11 (0.00 and
  1.11), on `sunrise` 0.22 (0.0035 and 1.65); against the split, more on 9 of 13 shots at
  A = 0.25 and on 8 at A = 0.5. Line art softens toward bicubic's (the crops). Low-frequency
  flicker falls with A (`ds4_0.25` −0.41 against the split, 12 / 0 / 1).

### At 1080p

The same strengths on the 1080p dumps (×2 from 540p, d1 and d2), against bicubic, its master
scored as at 4K (the 8 clips at d1; at d2 only anime-clean and anime-grain have one), and against
the split; `ds4_0.75` and `ds4_0.25` at seed 42, `ds4_0.5` and the split at 2–3 seeds. Against
bicubic at d1, 8 clips:

| Variant | PSNR-Y (dB) | LPIPS | DISTS | ΔE00 lf | Flicker lf |
|---|---|---|---|---|---|
| `ycc:4:3` | −6.18 (0/8/0) | −0.0062 (4/4/0) | −0.0134 (4/4/0) | +0.061 (0/7/1) | +0.681 (0/8/0) |
| `ds4_0.75` | −4.21 (0/8/0) | −0.0173 (5/2/1) | −0.0206 (6/2/0) | +0.012 (2/4/2) | +0.436 (0/7/1) |
| `ds4_0.5` | −1.75 (0/6/2) | −0.0272 (7/0/1) | −0.0228 (7/0/1) | −0.028 (4/1/3) | +0.210 (1/7/0) |
| `ds4_0.25` | +0.28 (6/1/1) | −0.0285 (8/0/0) | −0.0203 (8/0/0) | −0.053 (8/0/0) | +0.038 (1/6/1) |

Against the split:

| Set (clips) | Variant | PSNR-Y (dB) | LPIPS | DISTS | Flicker lf | Detail |
|---|---|---|---|---|---|---|
| d1 (8) | `ds4_0.75` | +2.10 (7/0/1) | −0.0134 (7/1/0) | −0.0083 (6/1/1) | −0.248 (8/0/0) | 8↓ |
| | `ds4_0.5` | +4.43 (8/0/0) | −0.0210 (7/1/0) | −0.0094 (6/1/1) | −0.471 (8/0/0) | 8↓ |
| | `ds4_0.25` | +6.58 (8/0/0) | −0.0246 (6/2/0) | −0.0080 (6/2/0) | −0.646 (8/0/0) | 8↓ |
| d2 (7) | `ds4_0.75` | +1.95 (6/0/1) | −0.0119 (6/1/0) | −0.0058 (5/1/1) | −0.208 (7/0/0) | 7↓ |
| | `ds4_0.5` | +4.02 (7/0/0) | −0.0186 (6/1/0) | −0.0034 (4/3/0) | −0.387 (7/0/0) | 7↓ |
| | `ds4_0.25` | +5.85 (7/0/0) | −0.0201 (5/2/0) | +0.0042 (3/4/0) | −0.510 (7/0/0) | 7↓ |

- **Against bicubic, the split is level at 1080p and the strengths are ahead.** The split beats
  bicubic on LPIPS on half the clips and on DISTS on half (4 / 4 / 0 each), 6.2 dB below it, with
  more low-frequency flicker on 8 of 8. `ds4_0.5` beats bicubic on LPIPS and DISTS on 7
  of 8 (within on anime-clean), 1.75 dB below it; `ds4_0.25` on 8 of 8, at bicubic's PSNR-Y
  (+0.28 dB, above it on 6) with a lower ΔE00 lf on 8. At d2, on the two clips with a bicubic
  master, the split is worse than bicubic on LPIPS and DISTS on both, `ds4_0.25` better on both.
- **Against the split, A = 0.5 is better on LPIPS on 7 of 8 clips at d1 (6 of 7 at d2) and on
  DISTS on 6 (4 at d2; worse on anime-dark, the sky and live-vfx),** with less low-frequency
  flicker on 8 of 8 (−0.47). Live-vfx is the one clip where every strength is worse than the
  split (A = 0.5: LPIPS +0.042 and DISTS +0.034 at d1). A = 0.25 also costs LPIPS on
  cartoon-bright and DISTS on the sky at d1, and DISTS on compressed inputs: +0.0042 at d2, worse
  on 4 of 7.
- **The best A per clip** (lowest LPIPS and DISTS along A = 1, 0.75, 0.5, 0.25 and bicubic, seed
  42): at d1, LPIPS at 0.25 on 3 clips (anime-clean, -dark, -grain), 0.5 on 3 (anime-bright,
  anime-sky, live-slow), 0.75 on cartoon-bright and 1 on live-vfx; DISTS at 0.25 on 3, 0.5 on 2,
  0.75 on 2 and 1 on live-vfx; bicubic is never best. At d2, LPIPS picks 0.25 on 3, 0.5 on 3 and
  1 on live-vfx. A = 0.5 is within one step of every d1 clip's best on both scores but live-vfx's,
  whose finest detail is already below the GT's (0.64 times its RMS below 0.7 px); at d2,
  anime-dark's DISTS also prefers 1.
- **By the metrics, the detail strength is not a 4K-only setting, and its best value is higher
  at 1080p:** 0.5 against 4K's 0.25. The model's fine detail strays less from the GT at 1080p: its
  finest bands correlate 0.42–0.54 and 0.76–0.84 with the GT's on three of the four 1080p clips
  scanned, against 0.09–0.35 and 0.30–0.69 on average per kind at 4K. The user's eyes then
  overturned these values (below).

### 4K tiles on flat content

Seed 42's latents decoded again with tiles of 1024 and 512 px (overlap 64) on `sunrise`, `space`
and `sollevante-painted`, the smooth and flat shots, each against the 2048-px decode; and the
consumer cards' 1024 / 768 recipe as a full run on `sunrise`. The worst 60×60 block per frame,
range over the shots (the GT's flattest third in brackets):

| Encode / decode tiles | `none` | `lab` | `ycc:4:3` | `ycc:3:3` |
|---|---|---|---|---|
| untiled / 1024 | 5.23–12.85 (4.19–10.05) | 0.38–1.19 (0.11–0.30) | 0.11–0.58 (0.04–0.13) | 0.04–0.27 (0.02–0.08) |
| untiled / 512 | 7.78–15.29 (5.64–9.92) | 0.39–1.58 (0.28–0.44) | 0.12–0.77 (0.08–0.15) | 0.05–0.35 (0.02–0.10) |
| 1024 / 768 (`sunrise`) | 11.91 (11.24) | 0.74 (0.34) | 0.24 (0.12) | 0.09 (0.05) |

- **Still no floor from colour down to 512 px, on flat content too:** after the split the
  flattest third moves by 0.04–0.13 levels with 1024-px decode tiles and 0.08–0.15 with 512-px
  ones (the first film: 0.11–0.28), the worst block by 0.11–0.58 and 0.12–0.77: a third to a half
  of `lab`'s (0.38–1.19, 0.39–1.58), `ycc:3:3` a tenth to a quarter. The consumer recipe leaves
  0.24 on the sky (its flattest third 0.12), against `lab`'s 0.74.
- **The worst blocks sit in texture:** every maximum is on the painted landscape, whose worst
  block moves by 0.58 and 0.77, its flattest third by 0.13 and 0.15. That is more than the first
  film's worst (0.42 at 512 px), on the shot whose uncorrected decode drifts most (12.9–15.3
  levels).
- With a detail strength the drift left in the lightness scales by A (above): at A = 0.25, a
  quarter of the split's, at 0.5 half.

### Crops

`colour_crops.py` with `--bicubic` and windows placed by hand (`--at`), on seed 42's frames, each
panel at 1:1 labelled with its mean ΔE00 to the GT, and a `.stretch.png` twin with every panel's
luma stretched by the GT window's 1st–99th percentiles (the better view of the dark ones); each
folder's README.txt lists its crops.

- **4K:** 31 strips in `D:\Video\seedvr2x\colour\crops-b1\`, GT | bicubic | `ycc:4:3` |
  `ycc:3:3` | `ds2_0.5` | `ds4_0.5` | `ds4_0.25`, panels of 512×288: each strip is 3584 px wide,
  to be viewed at 1:1, scrolling. Windows by hand on the middle frame, plus the tool's pick where
  `ds4_0.25` differs most from the split, and on two cel shots its flattest window that isn't
  black:
  - skin: the first film's close-up (3), `cockpit` (3);
  - line art and flat colour: `sollevante-line` (3), `-painted` (3), `-action` (2);
  - skies and smooth gradients: `sunrise` (3), `space` (3);
  - grain and flat cel: `cel4k-detail`, `-flat` and `-night` (3 each);
  - `jet`'s stencilled text and `dogfight`'s motion blur (1 each).
- **1080p:** 10 crops of 5 clips in `D:\Video\seedvr2x\colour\crops-b1-1080p\` (anime-clean,
  anime-bright, cartoon-bright, live-slow, live-vfx, 2 each, at d1), GT | bicubic | `ycc:4:3` |
  `ds4_0.75` | `ds4_0.5` | `ds4_0.25`, panels of 384×216: line art and flat colour, credit text,
  faces and skin, film grain.

What to look at: at 4K, whether A = 0.25 or 0.5 looks right on skin, which the split
over-textures (step 4), and on line art, which A = 0.25 softens toward bicubic's; whether banding
shows in the skies at A = 0.25; on grain, whether any model panel beats bicubic. At 1080p, A = 0.5
against 1 (the split) and 0.75, live-vfx being the clip where the scores prefer 1. The user's
verdict follows.

### The user's eyes

The user judged the crops on 2026-10-06: at 4K at least one window per clip, mostly the first of
each folder (each crop sits next to its `.stretch.png` twin there, which kept the others out of
view); at 1080p both windows of the 5 clips. Their criterion, in their words: "Tu veux l'image
redessinée avec des traits plus francs, plus fins, des couleurs plus nettes, pas juste la même
image délavée et nettoyée". A diffusion restorer should give a redrawn picture, with crisper,
finer lines and cleaner colours, the style kept, not the original's washed-out picture made
bigger, as if passed under a sponge. `ycc:4:3` and `ycc:3:3` still look the same to them.

At 4K:

| Shot | Preferred | What the eyes saw |
|---|---|---|
| `cel4k-detail` | the split | far better: redrawn, crisper, the style kept; A = 0.25 too close to the washed-out GT |
| `cel4k-flat` | bicubic | the thread-thin fake writing is beyond the model, which comes out blocky; bicubic doesn't deform it |
| `cel4k-night` | the split; A = 0.25 a good compromise | the split makes the blurred city readable (the buildings show); A = 0.25 keeps the original's soft style |
| `digital-cockpit` | none | bicubic loses the GT's pores, the split over-emphasises them, the blends attenuate them without reaching the GT's sharpness |
| `digital-dogfight` | not a blend | the blends are the worst: they mix the GT's grain and the model's sharp lines into an incoherent picture |
| `digital-jet` | the split, `ycc:3:3` | they restore the stencilled text better than the GT: more legible, sharper, some of the original blur kept |
| `digital-space` | A = 0.25 | the split and `ycc:3:3` destroy the clouds with bright, over-sharpened artefacts |
| `digital-sunrise` | A = 0.5 | no GT grain, a clean, coherent gradient |
| the first film's close-up | A = 0.25 | the split and `ycc:3:3` make the skin "almost reptilian" |
| `sollevante-action`, `-line`, `-painted` | A = 0.25, the least bad | nothing satisfies against the GT |

At 1080p:

| Clip | Window | Preferred | What the eyes saw |
|---|---|---|---|
| anime-bright | credit text | none | the split blocky, bicubic washed out |
| | face | the split | detail beyond the GT, no artefacts |
| anime-clean | the facade's windows | A = 0.25 | – |
| | small characters | A = 0.25 | the split too sharp |
| cartoon-bright | flowers | A = 0.5 or 0.25 | – |
| | the dog's face | A = 0.25 | still below the GT |
| live-slow | the man, frontal | A = 0.75 | – |
| | the couple | A = 0.5 | the split slightly too sharp, degrading a little; bicubic poor |
| live-vfx | shoulder | a matter of taste | all fine but bicubic |
| | the soldier's face | the split | detail without over-sharpening or artefacts |

- **At 4K, eyes and metrics agree where the model invents isotropic texture:** skin, clouds, the
  sky, brush texture, and structure too thin for it (`cel4k-flat`). There a blend toward the
  input helps, preferred at A = 0.25 or 0.5, though on `cockpit` no strength reaches the GT's
  pores.
- **They disagree where the model redraws structure:** cel lines, text, buildings, motion. The
  eyes prefer the redraw; full-reference metrics count a redraw sharper than the GT as error,
  even where it beats the GT (`digital-jet`'s text).
- **At 1080p the metrics' best A (seed 42, at d1) was the eyes' choice on anime-clean (0.25),
  live-slow (0.5 and 0.75) and live-vfx (1),** within one step on cartoon-bright (metrics 0.5–0.75,
  eyes 0.25–0.5), and away from it on anime-bright (metrics 0.5; eyes the split on the face, no
  winner on the text).
- **The preferred amount of redraw varies with the content,** even between two windows of one
  clip (anime-bright: the text against the face). No single global strength fits, and the global
  blend has its own failure: it softens lines and, on `dogfight`, mixes two styles.
- **Limits:** one viewer; at 4K mostly one window per clip; one model (7B fp16), ×2 from the mild
  degradation (d1).

### What step 5 recommends

- **No fixed strength.** seedvr2x's default stays the split alone (A = 1), as DESIGN.md has it.
  The 0.25 (4K) and 0.5 (1080p) defaults this step first drew from the metrics are withdrawn after
  the user's eyes.
- **What the findings support:** the right amount of the model's redraw depends on the content,
  within a clip too; a global blend is a crude tool, which softens lines and can mix two styles.
- **What they don't settle:**
  - whether seedvr2x needs a user scale ("detail recreation", as commercial upscalers offer): the
    user's proposal, supported by the content dependence the eyes show, not a measured
    conclusion;
  - what such a scale should move: the global blend measured here, or a structure-aware strength
    that keeps the model's lines and attenuates its isotropic invented texture, untested;
  - whether the sharp 7B changes the picture (the decision brief below).
- **Next, for design to decide:** the sharp 7B on batch A at 4K first, with the same protocol and
  crops for the eyes; a scale study only if the over-texture remains.

## Step 6: the sharp 7B, for the eyes

DESIGN.md's "4K output's quality" ([To measure](../../seedvr2x/DESIGN.md#to-measure)) left two
questions after step 5: whether ByteDance's sharp 7B over-renders less than the 7B, and whether a
heavier degradation tames the over-rendering, since ByteDance's readme says its models "tend to
overly generate details on inputs with very light degradations"
([`readme.md:119`](../../upstream/seedvr-bytedance/readme.md)). Step 6 runs the sharp 7B on the
clips and shots of steps 3 and 5, at 4K and 1080p, then both models at 4K from the heavier
degradation, and cuts crops of the two side by side for the user's eyes.

### The runs

- **The model:** `seedvr2_ema_7b_sharp_fp16.safetensors`, numz's fp16 file of ByteDance's sharp
  7B (the 7B's architecture, 16.5 GB, [models.md](models.md)). Each run is the 7B's of steps 1
  and 5 with `--dit_model` alone changed (the job lines compared word by word): numz `4490bd1`; at
  4K, encode untiled and decode tiles of 2048 px (overlap 64); at 1080p, untiled; seed 42;
  `--color_correction none` with the raw decode dumped (`colour_dump.py`), every variant applied
  to that decode.
- **One reference for both models:** the sharp's encoder input and float32 reference are
  bit-identical to the 7B's (md5 of `enc_bf16.pt` and `ref_f32.pt` on `sollevante-painted`): they
  come before the model. Each sharp decode is corrected and scored with the 7B run's reference,
  and paired with the 7B's output of the same input frame by frame.
- **58 runs, none failed:** (a) the sharp at 4K from d1 on step 5's 17 shots (batch A's 13, the
  first film's 4); (b) the sharp at 1080p on 26 clips: ×2 from d1 (8) and from d2 (7), ×1.5 and
  ×3 (3 each, from d1), ×4 (5); (c) both models at 4K from d2 on 5 shots chosen by content:
  `digital-cockpit` and the first film's close-up (skin), `digital-space` (clouds),
  `sollevante-painted` (painted texture), `cel4k-detail` (lines); (d) the sharp at seed 43 on those
  5 shots, from d1.
- **The 4K d2 inputs are new:** fr_clips.py's d2 (×½ in Y'CbCr by swscale's area filter, zimg
  having none; x264 slow, CRF 26), made from the 4K clips' tone-mapped SDR sources by
  `colour_clips.py`, which now imports fr_clips.py's recipe, with their bicubic baselines
  (Catmull-Rom ×2). The chain is frame-exact against measurement's own files: remade, d1 at 4K
  (`digital-sunrise`, the first film's close-up) and d2 at 1080p (anime-clean) give frames
  identical to fr_clips.py's by framemd5 (the files' md5 differ only by Matroska's random UIDs).
  Bicubic from d2 is 0.96–2.24 dB below bicubic from d1 by PSNR-Y (`cockpit` 43.19 → 41.28, the
  close-up 41.01 → 40.05, `space` 43.36 → 41.93, `painted` 37.12 → 34.88, `cel4k-detail` 41.52 →
  39.29).
- **Scores as in step 5:** LPIPS on every frame, DISTS on every 9th (to compare outputs within
  these tables only), VMAF v1 with the model the GT's width picks (2160p at 4K, 1080p at 1080p)
  and CAMBI added, from `gbrp16le` masters; `colour_bands.py` on every new decode (every third
  frame). At 1080p both models and bicubic were scored again at seed 42 with these settings (the
  7B's scores equal steps 1–3's on all 52 outputs), bicubic on all 26 clips. Verdicts as in step
  5, paired by frame over the seeds both outputs hold, beyond the 95% block-bootstrap interval and
  a seed spread: against the 7B, the 7B's own (seeds 42 and 43 on six batch A shots, 42, 43 and
  1234 on the first film); against bicubic, the output's own (the sharp's on the 5 shots of part
  (d)); with one seed the interval alone decides.
- **Time:** 4K runs took 7.5–15.8 minutes, 1080p runs 3.4–3.9, under the card's known fault: an
  SM clock of 577 MHz at a reported 604 W and 38 °C (medians under load; the clock-event reason SW
  power cap on 985 of 1,020 samples), as NVIDIA's forum reports on other RTX PRO 6000 Blackwell
  boards. It slows the runs, not their values.

### At 4K, against the 7B

The sharp's split minus the 7B's (`ycc:4:3`, each on its own decode, the same reference), frame
by frame over the seeds both hold (42; 43 too on `space`, `sollevante-painted` and the close-up),
mean over the shots (better / worse / within for the sharp: closer to / further from the GT by
that score); detail: the luma Laplacian variance over the GT's, geometric mean over the shots, the
sharp's (the 7B's), and the sharp's over the 7B's:

| Source (shots) | PSNR-Y (dB) | VMAF | LPIPS | DISTS | CAMBI added | ΔE00 lf | Flicker lf | Detail ÷ GT's: sharp (7B) | Sharp ÷ 7B |
|---|---|---|---|---|---|---|---|---|---|
| digital live action (5) | +0.49 (4/0/1) | +0.74 (4/1/0) | −0.0082 (4/1/0) | +0.0024 (4/1/0) | −0.026 (3/1/1) | +0.003 (2/2/1) | −0.011 (3/1/1) | 2.58 (3.42) | 0.76 |
| Sol Levante (4) | +0.52 (3/1/0) | +1.92 (3/1/0) | −0.0108 (3/1/0) | +0.0025 (1/2/1) | −0.001 (0/0/4) | −0.004 (2/2/0) | −0.028 (1/0/3) | 3.29 (3.73) | 0.88 |
| cel scan (4) | +0.67 (3/1/0) | +2.13 (3/1/0) | −0.0279 (4/0/0) | −0.0105 (4/0/0) | +0.000 (0/0/4) | +0.012 (1/3/0) | +0.009 (2/2/0) | 5.68 (7.30) | 0.78 |
| first film (4) | +0.85 (4/0/0) | +2.47 (4/0/0) | +0.0017 (1/1/2) | +0.0054 (2/2/0) | −0.002 (1/0/3) | +0.005 (0/2/2) | −0.014 (4/0/0) | 1.04 (1.97) | 0.53 |

- **Closer to the GT by PSNR-Y and VMAF on every kind, with less detail.** Per shot, PSNR-Y is
  higher on 14 of 17 (+0.32 to +1.44 dB) and VMAF on 14 (+0.66 to +4.78), LPIPS lower on 12
  (−0.0064 to −0.0526) and DISTS on 11 (the digital mean, +0.0024, is `space`'s +0.0219). The
  sharp keeps 0.45–0.88 of the 7B's Laplacian variance on 15 shots: still 2.6–5.7 times the GT's
  on batch A's kinds, 1.04 on the first film.
- **Where step 5's eyes rejected the 7B's invented texture:** on the close-up's skin ("almost
  reptilian" with the 7B) the sharp keeps 0.66 of the GT's fine detail where the 7B keeps 1.47,
  with PSNR-Y +1.14 dB, VMAF +4.8, LPIPS within (+0.0007) and DISTS −0.0025; on `cockpit`'s pores
  3.23 times the GT's (the 7B 4.30), closer on all four scores. `space`'s clouds are the exception:
  the sharp keeps the 7B's detail there (0.96 of it, 3.65 times the GT's) and is further from the
  GT on LPIPS (+0.0112), DISTS (+0.0219) and VMAF (−3.0), PSNR-Y within.
- **Where they wanted the 7B's redraw:** on `digital-jet`'s stencilled text (0.79 of the 7B's
  detail) and `cel4k-night`'s city (0.73) the sharp is closer on all four scores; on
  `cel4k-detail` (0.88) LPIPS and DISTS are lower (−0.0104, −0.0060), PSNR-Y and VMAF further
  (−0.11 dB, −1.4).
- **`sollevante-line` is the one shot where the sharp has more fine detail than the 7B** (1.20
  times its Laplacian variance), and it is further from the GT on all four scores (PSNR-Y −0.26
  dB, VMAF −1.2, LPIPS +0.0065, DISTS +0.0081).
- **On the first film** the sharp keeps about half the 7B's detail (0.45–0.74; the fast pan 0.44
  of the GT's) and is closer by PSNR-Y and VMAF on 4 of 4, but LPIPS and DISTS are mixed: further
  on the fast pan (+0.0169, +0.0123) and on the dark interior's DISTS (+0.0170), closer on the
  street (both) and on the close-up's DISTS.
- **Colour is the correction's, not the model's:** uncorrected, the sharp's colour is closer to
  the GT's on Sol Levante and the first film (ΔE00 lf −0.18 and −0.25, 4 of 4 each), as
  models.md found at 1080p; after the split the two are within 0.012 of each other per kind.
  Colour fringes and full-resolution flicker are lower with the sharp on 15 and 13 of the 17
  shots (−0.10 to −0.13 and −0.15 to −0.29 per kind), along with its lower fine detail. Banding
  doesn't change (CAMBI added within on 12 shots, lower on 4, higher by 0.001 on `space`).

### Against bicubic

Each model's split − bicubic, as step 5's table (the 7B's rows are step 5's; the sharp's over
seeds 42 and 43 on the 5 shots of part (d); bicubic's detail in brackets):

| Source (shots) | Model | PSNR-Y (dB) | VMAF | LPIPS | DISTS | CAMBI added | Detail ÷ GT's |
|---|---|---|---|---|---|---|---|
| digital live action (5) | 7B | −5.84 (0/5/0) | −19.26 (0/5/0) | +0.0195 (0/4/1) | +0.0215 (1/4/0) | −2.209 (5/0/0) | 3.42 (0.15) |
| | sharp | −5.36 (0/5/0) | −18.56 (0/5/0) | +0.0114 (0/3/2) | +0.0241 (1/4/0) | −2.235 (5/0/0) | 2.58 |
| Sol Levante (4) | 7B | −7.57 (0/4/0) | −25.32 (0/4/0) | +0.0224 (0/3/1) | −0.0051 (3/1/0) | −0.963 (4/0/0) | 3.70 (0.17) |
| | sharp | −7.07 (0/4/0) | −23.39 (0/4/0) | +0.0121 (1/2/1) | −0.0024 (2/1/1) | −0.963 (4/0/0) | 3.29 |
| cel scan (4) | 7B | −9.36 (0/4/0) | −25.30 (0/4/0) | +0.1563 (0/4/0) | +0.0910 (0/4/0) | −0.016 (3/0/1) | 7.30 (0.54) |
| | sharp | −8.69 (0/4/0) | −23.16 (0/4/0) | +0.1283 (0/4/0) | +0.0806 (0/4/0) | −0.016 (3/0/1) | 5.66 |
| first film (4) | 7B | −4.98 (0/4/0) | −23.15 (0/4/0) | +0.0471 (0/3/1) | +0.0416 (0/4/0) | −0.026 (4/0/0) | 1.96 (0.12) |
| | sharp | −4.12 (0/4/0) | −20.60 (0/4/0) | +0.0477 (0/3/1) | +0.0463 (0/4/0) | −0.028 (4/0/0) | 1.04 |

- **The sharp doesn't change step 5's verdict.** Its split trails bicubic on PSNR-Y and VMAF on
  17 of 17 shots (VMAF 2.5 points below on `sollevante-dark` to 45 on `cel4k-motion`), by 0.5–0.9
  dB and 0.7–2.6 VMAF points less than the 7B's per kind. LPIPS is above bicubic's on 12 shots
  (the 7B 14), within on 4, below on `sollevante-action`; DISTS above on 13 (the 7B 13), below on
  `sunrise`, `sollevante-line` and `-action`, within on `sollevante-painted`, which the 7B has
  below bicubic's (−0.0063).
- **Less banding than bicubic** on 16 of 17, as the 7B (within on `cel4k-detail`, where neither
  adds any). Low-frequency colour and flicker against bicubic stay the 7B's (ΔE00 lf +0.019 to
  +0.118 per kind, the 7B +0.016 to +0.106).
- **It still over-renders:** 2.6–5.7 times the GT's Laplacian variance on batch A's kinds (the 7B
  3.4–7.3), where bicubic keeps 0.15–0.54; on the first film 1.04 (the 7B 1.96, bicubic 0.12).

### Band by band

`colour_bands.py` on the sharp's split as step 5 on the 7B's (seed 42, every third frame): each
band's RMS over the GT's · its correlation with the GT, whole picture, means over the shots:

| Source (shots) | Model | below 0.7 px | 0.7–1.6 px | 1.6–3.2 px | 3.2–6.5 px |
|---|---|---|---|---|---|
| digital live action (5) | 7B | 1.99 · 0.15 | 1.64 · 0.41 | 1.48 · 0.68 | 1.22 · 0.90 |
| | sharp | 1.72 · 0.16 | 1.52 · 0.42 | 1.42 · 0.68 | 1.22 · 0.90 |
| Sol Levante (4) | 7B | 1.99 · 0.32 | 1.74 · 0.66 | 1.50 · 0.88 | 1.21 · 0.96 |
| | sharp | 1.85 · 0.32 | 1.64 · 0.66 | 1.45 · 0.88 | 1.20 · 0.96 |
| cel scan (4) | 7B | 2.71 · 0.35 | 2.06 · 0.69 | 1.52 · 0.88 | 1.18 · 0.96 |
| | sharp | 2.37 · 0.36 | 1.86 · 0.71 | 1.46 · 0.89 | 1.18 · 0.96 |
| first film (4) | 7B | 1.58 · 0.09 | 1.31 · 0.30 | 1.30 · 0.73 | 1.15 · 0.95 |
| | sharp | 1.17 · 0.10 | 1.10 · 0.33 | 1.24 · 0.73 | 1.15 · 0.95 |

- **The same kind of invented texture, less of it.** The sharp's three finest bands carry less
  energy than the 7B's on every kind (below 0.7 px, 1.72–2.37 times the GT's RMS on batch A's
  kinds against 1.99–2.71; 1.17 on the first film against 1.58), with the same correlation with
  the GT within 0.03 (0.10–0.36 below 0.7 px): fine detail that is mostly not the GT's, as with
  the 7B. From 3.2 px up the two are the same. On the GT's flattest third likewise: below 0.7 px,
  0.93–2.30 times the GT's RMS per kind against 1.07–2.44.
- **Still above the GT's energy on batch A** in the three finest bands (1.42–2.37 per kind). On
  three shots the sharp's two finest bands sit below it: the close-up (0.78 and 0.75; the 7B 1.17
  and 1.05), the fast pan (0.60 and 0.62; 0.89 and 0.73) and `sunrise`'s sky (0.77 and 0.71; 0.89
  and 0.88).
- **`space` and `sollevante-line` keep the 7B's energy,** the two shots where the sharp's detail
  doesn't fall: `space` 2.07 times the GT's RMS below 0.7 px (the 7B 2.08) and more from 1.6 px
  up (1.48 and 1.15 against 1.42 and 1.09); `sollevante-line` more below 0.7 px (1.53 against
  1.40).

### At 1080p

The 26 clips, seed 42, both models' splits (`ycc:4:3`; `ycc:4:4` at ×3 and ×4, the factor rule)
scored again; the sharp − the 7B per set, as at 4K (one seed: the interval alone decides):

| Set (clips) | PSNR-Y (dB) | VMAF | LPIPS | DISTS | ΔE00 lf | Flicker lf | Detail ÷ GT's: sharp (7B) | Sharp ÷ 7B |
|---|---|---|---|---|---|---|---|---|
| d1 (8) | +0.64 (6/2/0) | +1.40 (5/3/0) | +0.0019 (4/2/2) | +0.0034 (3/5/0) | −0.004 (5/1/2) | −0.037 (5/2/1) | 1.86 (1.90) | 0.98 |
| d2 (7) | +0.68 (5/1/1) | +1.59 (4/2/1) | +0.0013 (3/2/2) | +0.0033 (2/5/0) | −0.007 (5/0/2) | −0.043 (6/0/1) | 1.76 (1.85) | 0.95 |
| ×1.5 (3) | +0.61 (3/0/0) | +2.22 (3/0/0) | −0.0112 (3/0/0) | −0.0034 (3/0/0) | −0.023 (3/0/0) | −0.082 (2/0/1) | 2.32 (2.60) | 0.89 |
| ×3 (3) | +0.35 (3/0/0) | +0.96 (2/0/1) | −0.0041 (3/0/0) | +0.0008 (1/2/0) | −0.003 (2/1/0) | −0.070 (3/0/0) | 2.12 (2.30) | 0.92 |
| ×4 (5) | −0.05 (4/1/0) | −1.02 (1/3/1) | +0.0078 (3/1/1) | +0.0055 (2/3/0) | +0.002 (3/2/0) | −0.030 (4/0/1) | 1.61 (1.96) | 0.82 |

Each model's split − bicubic (bicubic's detail in brackets):

| Set (clips) | Model | PSNR-Y (dB) | VMAF | LPIPS | DISTS | Detail ÷ GT's |
|---|---|---|---|---|---|---|
| d1 (8) | 7B | −6.30 (0/8/0) | −12.52 (0/8/0) | −0.0039 (4/4/0) | −0.0124 (4/4/0) | 1.90 (0.16) |
| | sharp | −5.67 (0/8/0) | −11.11 (0/8/0) | −0.0020 (4/4/0) | −0.0090 (5/3/0) | 1.86 |
| d2 (7) | 7B | −5.53 (0/7/0) | −11.56 (0/7/0) | −0.0202 (5/2/0) | −0.0327 (5/2/0) | 1.85 (0.18) |
| | sharp | −4.86 (0/7/0) | −9.97 (0/7/0) | −0.0189 (5/1/1) | −0.0294 (6/1/0) | 1.76 |
| ×1.5 (3) | 7B | −9.77 (0/3/0) | −20.98 (0/3/0) | +0.0635 (0/3/0) | +0.0348 (1/2/0) | 2.60 (0.37) |
| | sharp | −9.16 (0/3/0) | −18.77 (0/3/0) | +0.0523 (1/2/0) | +0.0314 (1/2/0) | 2.32 |
| ×3 (3) | 7B | −2.97 (0/3/0) | +2.65 (2/1/0) | −0.0891 (2/1/0) | −0.0466 (3/0/0) | 2.30 (0.09) |
| | sharp | −2.62 (0/3/0) | +3.61 (2/1/0) | −0.0932 (2/1/0) | −0.0458 (2/1/0) | 2.12 |
| ×4 (5) | 7B | −1.42 (3/2/0) | +15.16 (4/0/1) | −0.1660 (5/0/0) | −0.1051 (5/0/0) | 1.96 (0.03) |
| | sharp | −1.47 (2/3/0) | +14.14 (4/0/1) | −0.1583 (5/0/0) | −0.0996 (5/0/0) | 1.61 |

- **At ×2 the sharp keeps the 7B's detail on average** (0.98 and 0.95 of its Laplacian
  variance), and the scores split: closer to the GT by PSNR-Y and VMAF on most clips, LPIPS and
  DISTS a little higher on average (DISTS +0.0034 and +0.0033, higher on 5 of 8 and 5 of 7), as
  models.md found on 4 animated clips (PSNR-Y +0.40 to +0.67 dB, 2–10% less fine texture). Its
  finest band loses less energy than at 4K: at d1, 1.60 times the GT's RMS against the 7B's 1.68
  (whole picture, the same correlation: 0.38), 1.47 against 1.67 on the GT's flattest third.
- **The clips differ most:** on anime-bright (d1) the sharp has 1.58 times the 7B's detail and
  is further on all four scores (PSNR-Y −0.96 dB, VMAF −1.7, LPIPS +0.0119, DISTS +0.0131); on
  anime-sky 0.69 times, with PSNR-Y +3.10 dB but VMAF −0.9 and DISTS +0.0051; on live-vfx, at the
  7B's detail (0.97 and 1.04 times at d1 and d2), PSNR-Y +1.3 and +1.2 dB and VMAF +7.3 and +7.1,
  LPIPS +0.034 and +0.035, DISTS +0.017 and +0.020.
- **×1.5:** closer on all four scores on 3 of 3 clips (detail 0.89 of the 7B's). **×4:** detail
  0.82; live-vfx much further (PSNR-Y −3.46 dB, VMAF −7.2, LPIPS +0.055, DISTS +0.024), the other
  4 closer by PSNR-Y (+0.12 to +1.97 dB).
- **Against bicubic the sharp changes nothing:** at ×2 both splits trail bicubic on PSNR-Y and
  VMAF on every clip, split LPIPS and DISTS about evenly at d1 and are ahead on 5–6 of 7 at d2; at
  ×3 and ×4 both are ahead on LPIPS and DISTS (×4: 5 of 5) and on VMAF (×4: +14 to +15); at ×1.5
  both trail on all four on most clips. The 7B's rows are seed 42's (step 5's, over 2–3 seeds,
  give PSNR-Y −6.18 dB at d1).

### The heavier degradation at 4K

Both models from d2 on the 5 shots. At seed 42 throughout, each output's detail from d1 and from
d2, bicubic's of each input, and the sharp − the 7B from d1 and from d2:

| Shot | Detail ÷ GT's: 7B d1 → d2 | sharp | bicubic | Sharp − 7B: PSNR-Y (dB) d1 → d2 | VMAF | LPIPS | DISTS |
|---|---|---|---|---|---|---|---|
| `digital-cockpit` | 4.30 → 4.69 | 3.23 → 3.94 | 0.29 → 0.31 | +0.63 → +0.41 | +1.72 → +1.57 | −0.0107 → −0.0047 | −0.0012 → +0.0068 |
| the close-up | 1.49 → 0.69 | 0.69 → 0.67 | 0.07 → 0.03 | +1.13 → +0.18 | +4.82 → −0.16 | +0.0002 → +0.0205 | −0.0030 → +0.0155 |
| `digital-space` | 3.86 → 4.22 | 3.77 → 5.54 | 0.17 → 0.15 | −0.09 → −1.05 | −3.32 → −8.80 | +0.0133 → +0.0341 | +0.0220 → +0.0425 |
| `sollevante-painted` | 6.71 → 6.94 | 5.80 → 6.46 | 0.23 → 0.27 | +0.49 → +0.21 | +2.96 → +1.16 | −0.0142 → −0.0085 | +0.0065 → +0.0068 |
| `cel4k-detail` | 3.98 → 4.25 | 3.48 → 3.39 | 0.53 → 0.56 | −0.11 → +0.37 | −1.38 → +0.69 | −0.0104 → −0.0121 | −0.0060 → −0.0099 |
| the 5 shots | 3.66 → 3.32 | 2.79 → 3.17 | 0.21 → 0.19 | +0.41 → +0.02 | +0.96 → −1.11 | −0.0044 → +0.0058 | +0.0037 → +0.0124 |

Against bicubic of the d2 input, the means over the 5 shots:

| Model | PSNR-Y (dB) | VMAF | LPIPS | DISTS | CAMBI added | Detail ÷ GT's |
|---|---|---|---|---|---|---|
| 7B | −5.61 (0/5/0) | −19.48 (0/5/0) | −0.0136 (2/2/1) | −0.0075 (3/2/0) | −2.956 (5/0/0) | 3.32 (0.19) |
| sharp | −5.58 (0/5/0) | −20.59 (0/5/0) | −0.0078 (2/2/1) | +0.0049 (2/3/0) | −2.969 (5/0/0) | 3.17 |

- **The heavier degradation doesn't tame the over-rendering.** From d2 both models keep 3.2–3.3
  times the GT's Laplacian variance (geometric mean of the 5 shots; from d1 3.66 and 2.79), their
  finest band at about 1.9 times the GT's RMS with the same low correlation (the 7B 1.94 · 0.26 →
  1.93 · 0.25, the sharp 1.75 · 0.27 → 1.89 · 0.26). The detail rises from d1 to d2 on 4 of the 5
  shots with the 7B and 3 with the sharp. It falls clearly only on the close-up with the 7B (1.49
  → 0.69, the sharp's level from d1); the sharp's falls are small (the close-up 0.69 → 0.67,
  `cel4k-detail` 3.48 → 3.39).
- **The sharp's lead shrinks:** from d2 its detail is 0.95 of the 7B's (0.76 from d1 on these
  shots), and the paired means move against it: PSNR-Y +0.02 dB (4 / 1 / 0), VMAF −1.11 (3 / 1 /
  1), LPIPS +0.0058 (3 / 2 / 0), DISTS +0.0124 (1 / 4 / 0), against +0.41 dB, +0.96, −0.0044 and
  +0.0037 from d1.
- **`space` again:** from d2 the sharp's split has 5.54 times the GT's detail (the 7B 4.22) and
  2.50 times its RMS below 0.7 px (2.18), and is further from the GT on all four scores. On the
  other four shots the sharp stays closer by PSNR-Y (+0.18 to +0.41 dB).
- **Against bicubic, the gap narrows: bicubic loses more from d2 than the models** (but the
  sharp on `space`). PSNR-Y is 5.6 dB below bicubic's for both models from d2, against 6.9 (the
  7B) and 6.5 (the sharp) on the same 5 shots from d1. LPIPS and DISTS fall below bicubic's on 2–3
  of the 5 shots (the close-up: LPIPS −0.121 and −0.101, DISTS −0.043 and −0.027), where from d1
  the models' DISTS was above bicubic's on 4 and 5 of them.
- **d2 is only mildly heavier on these sources:** bicubic's two finest bands barely move (0.52 →
  0.51 and 0.71 → 0.70 times the GT's RMS) and its PSNR-Y falls by 0.96–2.24 dB. ByteDance's
  heavier degradations (blur, noise) are untested.

### The seed

The sharp at seed 43 on the 5 shots of part (d), from d1: its seed spread (seed 43's mean minus
seed 42's, per shot) against the sharp − the 7B at seed 42, over the 5 shots:

| Score | Seed spread, mean size | Sharp − 7B, mean size | Shots where the models differ by more than the spread |
|---|---|---|---|
| PSNR-Y (dB) | 0.11 | 0.49 | 4 (not `space`) |
| VMAF | 0.55 | 2.84 | 5 |
| LPIPS | 0.0026 | 0.0097 | 4 (not the close-up) |
| DISTS | 0.0016 | 0.0077 | 5 |
| Detail ÷ GT's | 0.19 | 0.67 | 4 (not `space`) |
| RMS ÷ GT's below 0.7 px | 0.04 | 0.20 | 4 (not `space`) |

- **The two models differ by 3.5–5.2 times the sharp's own seed spread on average,** on every
  score, and by more than the spread on 4 or 5 of the 5 shots. The exceptions are seed 42's
  near-ties: `space`'s PSNR-Y and detail (the sharp − the 7B: −0.09 dB and −0.09 of the GT's
  Laplacian variance; the seed: +0.18 dB and −0.24) and the close-up's LPIPS (+0.0002 against
  0.0037). `space`'s DISTS and VMAF gaps hold (7.1 and 4.7 spreads); on `sollevante-painted` the
  gaps are the narrowest, 1.6–3.6 spreads on every score.
- **The sharp's spread is of the 7B's order, a little wider,** on the three shots where the 7B has
  seed 43 too (`space`, `sollevante-painted`, the close-up): PSNR-Y 0.07–0.22 dB against
  0.05–0.12, VMAF 0.6–1.2 against 0.1–0.7, the detail ratio 0.05–0.55 against 0.05–0.49.
- **Pooled into the 4K tables above** (the sharp's rows against bicubic; the pairs with the 7B on
  `space`, `sollevante-painted` and the close-up), seed 43 moves the means per kind by at most
  0.03 dB of PSNR-Y, 0.15 VMAF point, 0.0005 of LPIPS, 0.0003 of DISTS and 0.04 of the detail
  ratio, and changes five per-shot verdicts at the margin: against bicubic, `sollevante-painted`'s
  DISTS (+0.0009, worse, to −0.0002, within) and `space`'s low-frequency flicker (better to
  within); against the 7B, `space`'s colour fringes (within to better, −0.085) and ΔE00 after a
  16 px blur (within to better, −0.001), and the close-up's ΔE00 lf (within to worse, +0.001).

### Crops

`colour_crops.py` now puts several decodes in one strip (`--content LABEL=DECODE.pt`, repeated:
every variant on each) and leaves out the luma-stretched twins (`--no-stretch`). Each strip is
GT | bicubic | 7B split | sharp split at 1:1, each panel labelled with its mean ΔE00 to the GT;
each folder's README.txt lists its windows, what each shows and the verdict recorded on it:

- **4K from d1:** 31 strips in `D:\Video\seedvr2x\colour\crops-b2\`, step 5's windows on 13
  shots (the first film's close-up included), panels of 512×288 (strips 2048 px wide); the 7B
  panel is step 5's split panel, pixel for pixel.
- **1080p:** 29 strips in `D:\Video\seedvr2x\colour\crops-b2-1080p\`: step 5's 10 windows at d1
  (384×216) and step 4's 19 windows of anime-clean, anime-sky and live-slow at d2 (480×270).
- **4K from d2:** 15 strips in `D:\Video\seedvr2x\colour\crops-b2-d2\`: step 5's windows on the
  5 d2 shots, both models run from d2, bicubic from the d2 input.

What to look at: on the shots where step 5's eyes rejected the 7B's invented texture (the
close-up's skin, `cockpit`'s pores, `space`'s clouds, Sol Levante's painted texture), how the
sharp's texture looks; where they wanted the 7B's redraw (`cel4k-detail`, `cel4k-night`,
`digital-jet`'s text), whether the sharp keeps it; whether the d2 strips differ from the d1 ones
on the same windows. The metrics part the two models most on the close-up (the sharp at under
half the 7B's detail), `space` (the sharp further from the GT, from d1 and from d2),
`sollevante-line` (the sharp with more detail) and anime-bright at 1080p (1.58 times the 7B's
detail). The user's verdict follows.

### The user's eyes

The user judged all 75 strips on 2026-10-07, window by window, the 7B's split against the
sharp's (with no stretched twins, every window was seen). The criterion is step 5's: a picture
redrawn, with crisper lines and cleaner colours, not the GT's washed-out picture made bigger.

At 4K from d1 (31 windows):

| Shot (windows) | Preferred | What the eyes saw |
|---|---|---|
| the first film's close-up (3) | the sharp, on all 3 | "way better": the reptilian texture is gone |
| `cockpit` (3) | the sharp (2), alike (1) | the sharp really reduces the light artefacts, "much better" on the first window |
| `dogfight` (1) | the sharp | cleaner |
| `jet` (1) | the sharp | its letters a bit more readable, the style kept |
| `sollevante-painted` (3) | the sharp (1), leaning to it (1), alike (1) | less blocky, fewer light artefacts |
| `cel4k-detail` (3) | alike (2), the sharp (1) | in the flat window a bit more noise, but more consistent across the picture |
| `cel4k-night` (3) | alike (2; the 7B maybe a bit sharper on one), the sharp (1) | the flat dark window cleaner with the sharp: fewer artefacts, more coherent |
| `sollevante-line` (3) | alike | very thin differences: the 7B smoothing a little more on one, the sharp a little better on the finest lines on another |
| `sollevante-action` (2) | alike | the source too detailed to tell |
| `space` (3) | neither (2), the sharp a little (1) | both bad, on a hard source |
| `sunrise` (3) | the sharp (1), the 7B a little (2) | the sharp more coherent on one; on another it fuses the colours too much |
| `cel4k-flat` (3) | the 7B (2), neither (1) | the 7B draws sharper lines, "funny (that the sharp model is not the sharpest)"; both fail on the thread-thin writing, as expected |

At 4K from d2 (15 windows):

| Shot (windows) | Preferred | What the eyes saw |
|---|---|---|
| the first film's close-up (3) | the sharp, on all 3 | a better skin texture ("way better" on one), the GT's light grain removed |
| `cockpit` (3) | the sharp (1), alike (2) | fewer light artefacts |
| `sollevante-painted` (3) | the sharp (2), alike (1) | fewer artefacts |
| `cel4k-detail` (3) | alike (2), the sharp (1) | the flat window smoother, less blocky |
| `space` (3) | the 7B, on all 3 | the sharp increases the light artefacts a bit too much |

At 1080p (29 windows: 10 at d1, step 4's 19 at d2):

| Clip (windows) | Preferred | What the eyes saw |
|---|---|---|
| anime-bright, d1 (2) | alike; a tie | the sharp sharper but with fewer details, the 7B more details but a bit blurrier, like the GT |
| anime-clean, d1 (2) | the sharp (1), alike (1) | the sharp keeps more of the existing detail |
| live-slow, d1 (2) | alike, leaning to the sharp on one | a more homogeneous skin texture |
| cartoon-bright and live-vfx, d1 (4) | alike | – |
| anime-clean, d2 (5) | alike, leaning to the sharp on one edge | the sharp recreates better a drawing flaw the GT itself has |
| anime-sky, d2 (8) | the sharp (2), alike (1), can't tell (5) | the sharp less noisy, a bit smoother; too bright to tell on five |
| live-slow, d2 (6) | alike | plants a bit better with the sharp on one, the 7B maybe a little smoother on another |

Over the 75 windows the eyes prefer or lean to the sharp on 27 (14 at 4K from d1, 7 from d2, 6
at 1080p), the 7B on 8 (`cel4k-flat`'s lines 2, `cel4k-night` 1, `sunrise` 2, `space` from d2
3), and see no difference, or can't tell, on 40.

- **The sharp wins where step 5's eyes rejected the 7B's invented texture,** from d1 and from
  d2: the close-up's skin on all 6 windows, `cockpit`'s light artefacts, Sol Levante's
  blockiness, the cel scan's flat areas; at 1080p, a more homogeneous skin. Eyes and metrics
  agree there: on the close-up the sharp keeps 0.66 of the GT's fine detail where the 7B keeps
  1.47, and on `cockpit` it is closer to the GT on all four scores.
- **On lines and text the two are mostly alike, and the sharp keeps the redraw step 5's eyes
  wanted:** `jet`'s stencilled text a bit more readable, the style kept; Sol Levante's line art
  and `cel4k-detail` alike. The exception is the cel scan's thinnest lines: on `cel4k-flat` the
  7B draws them sharper on both hand-placed windows, where the metrics put the sharp closer to
  the GT on all four scores (PSNR-Y +1.44 dB, DISTS −0.0129; 0.73 of the 7B's detail). As in
  step 5, the eyes want a redraw of structure that full-reference metrics count as error.
- **Smooth gradients split:** on `sunrise` the sharp is more coherent on one window, the 7B a
  bit better on two, where the sharp fuses the colours too much; the metrics put the sharp
  closer on all four scores there.
- **From d2, clouds turn against the sharp:** `space` from d2 is the one shot where the eyes
  prefer the 7B on every window, as the metrics found (5.54 times the GT's fine detail against
  4.22; LPIPS +0.034, DISTS +0.043). From d1 both models fail on that source.
- **At 1080p the two look alike** on most windows (17 of 29); where the eyes split, they lean
  to the sharp (6) and never to the 7B. On anime-bright, where the sharp has 1.58 times the 7B's
  Laplacian variance and is further from the GT on every score, the eyes see a tie: sharper
  lines, fewer details.
- **Limits:** one viewer; 1:1 crops of 1–3 windows per shot, one seed; no crops at ×1.5, ×3 or
  ×4 (at ×4 the metrics are mixed: live-vfx 3.46 dB further from the GT with the sharp).

### What step 6 recommends

- **The sharp 7B as seedvr2x's default model, at 1080p and 4K.** The eyes prefer it or see no
  difference on 67 of 75 windows, and it removes the defect that drove step 5's detail-strength
  idea, the over-textured skin, with no guard worse: banding the same, colour the same after the
  split, colour fringes and full-resolution flicker lower. It has the 7B's architecture, memory
  and time.
- **No detail strength by default, and no user scale needed in v1:** the split alone (A = 1)
  stays. What remains goes both ways with the content (the 7B's sharper thin cel lines, the
  sharp's fused gradient colours, its clouds from compressed input), where a global strength
  would mend one case by harming another (step 5). If a scale is wanted later, the candidate is
  a structure-aware strength (keep the lines, tame isotropic texture), to study on step 6's
  dumps, on the CPU, with `space` from d2 as its test case.
- **Before trusting the default at other factors:** crops at ×1.5, ×3 and ×4 from step 6's
  dumps (CPU only); the metrics are mixed at ×4.
- **ByteDance's warning is tested only mildly:** a heavier compression (d2) tames neither model;
  blurred or noisy inputs would need GPU runs.

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
- **The tiles:** a split leaves half or less of what `lab` leaves of the VAE tiles' drift, at
  every tile size at 1080p and 4K: at most half a level in the worst block of a frame with decode
  tiles down to 512 px and with the consumer cards' recipes, at 1080p and on the first film's 4K
  shots; on painted 4K animation, 512-px decode tiles leave up to 0.77 in textured blocks (the
  flattest third at most 0.15); 512-px encode tiles leave up to 0.77 at 1080p (`lab` 1.39).
  Colour gives the planner no floor on tile size down to 512 px; with the detail strength the
  lightness drift scales by A.
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
  0.38–1.10; on 3 smooth or painted 4K shots (step 5: decode tiles of 1024 and 512, the 1024 / 768
  recipe) 0.11–0.77 against 0.38–1.58.
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
  1080p (animation and live action), 4 at 4K from one film (13 more from three sources in step
  5); ×4 on 5.
- Factors tested: ×1.5, ×2, ×3 and ×4; the rule in between (×2.25 for 480p to 1080p, say) is
  an interpolation.
- The 4K gap to bicubic that remains is the model's (below), not the correction's.

**The finer option, for the eyes:** `split:ycc:3:3` (lightness at 3.2 px too) gives the most
fidelity before detail goes: PSNR-Y +1.69 dB, VMAF +9.7 in the master, flicker −1.13, the tiles'
drift a quarter of `lab`'s, at 4K VMAF +12.5 and DISTS −0.011. It costs 15% of the sky's fine
texture, and DISTS is mixed on compressed inputs (3 of 7 clips worse, 3 better). The crops put
`lab`, `ycc:5:3:histY0.8`, `ycc:4:3` and `ycc:3:3` side by side: the user finds `4:3` never worse
than `lab` and can't tell it from `3:3` (step 4).

**4K's quality (for DESIGN.md's open question):** with the correction, the 4K output's
low-frequency colour reaches bicubic's (ΔE00 lf 0.59–0.78 with `ycc:3:3`, below bicubic's on 4
of 4; 0.60–0.84 with `ycc:4:3`, at most 0.08 above it; bicubic 0.61–0.79), and its flicker
comes close. The rest of the gap is the model's: PSNR-Y 2.7–7.3 dB below bicubic, VMAF 10–20
points below, DISTS worse on 4 of 4 and LPIPS on 3 of 4, even with `ycc:3:3`.

**4K and the detail strength (step 5, baton 1, and the user's eyes, 2026-10-06).**
Recommendation: no fixed detail strength. seedvr2x's default stays the split alone (A = 1), as
DESIGN.md has it; the 0.25 (4K) and 0.5 (1080p) defaults drawn from the metrics are withdrawn
after the user's eyes.
- **What the metrics found:** blending the split's lightness with the input's,
  A · L_split + (1 − A) · L_input, one multiply-add per pixel, gains LPIPS and DISTS. At 4K,
  A = 0.25 beats bicubic on both on 9 of 9 clean shots and on VMAF on 8, at bicubic's PSNR-Y; at
  1080p, A = 0.5 beats the split on LPIPS on 7 of 8 clips and bicubic on LPIPS and DISTS on 7 of 8.
- **What the eyes found:** they want the model's redraw where it redraws structure (cel lines,
  text, buildings), which full-reference metrics count as error, and not the isotropic texture it
  invents (skin, clouds, skies), where eyes and metrics agree. The preferred amount varies with
  the content, even within a clip, and the global blend softens lines and can mix two styles
  (`dogfight`).
- **What the findings support:** the right amount of the model's redraw depends on the content;
  a global blend is a crude tool.
- **What they don't settle:**
  - (a) whether seedvr2x needs a user scale ("detail recreation", as commercial upscalers offer):
    the user's proposal, supported by the content dependence the eyes show, not a measured
    conclusion;
  - (b) what such a scale should move: the global blend measured here, or a structure-aware
    strength keeping the model's lines and attenuating its isotropic invented texture, untested;
  - (c) whether the sharp 7B changes the picture. [models.md](models.md) measured it on 4 animated
    1080p clips, by metrics only: the closest of all models to the source (PSNR-Y +0.40 to +0.67
    dB, VMAF +1.9 to +3.5) and, despite its name, no more fine texture than 7B fp16 (2–10% less);
    never at 4K, on live action or by eye. ByteDance's readme (Limitations,
    [`upstream/seedvr-bytedance/readme.md`](../../upstream/seedvr-bytedance/readme.md)) says its
    models "tend to overly generate details on inputs with very light degradations […] leading
    to oversharpened results occasionally". The user has read online that the sharp model does
    better, and expects it to redraw without the over-sharpening, artefacts and blockiness.
- **Next, for design to decide:** the sharp 7B on batch A at 4K first (the same protocol, crops
  for the eyes), then a scale study only if the over-texture remains.
- **For "4K output's quality":** by the metrics, the model's 4K output trails bicubic whatever
  the colour correction (with the split, PSNR-Y 5.8–7.6 dB and VMAF 19–25 points behind on clean
  sources); only a blend toward the input beats bicubic, on LPIPS and DISTS. By eye it depends on
  the content: better than bicubic on structure (lines, text), worse where the model invents
  texture.

**The sharp 7B (step 6, baton 2, and the user's eyes, 2026-10-07).** Recommendation: the sharp
7B (`seedvr2_ema_7b_sharp_fp16`) as seedvr2x's default model at 1080p and 4K, with the split
alone (A = 1); no detail scale in v1.
- **It answers (c) above:** it over-renders less (at 4K 0.53–0.88 of the 7B's fine detail by
  kind, the same texture with less energy), the eyes prefer it or see no difference on 67 of 75
  windows, and it removes the defect behind (a): the first film's skin is no longer reptilian.
  The 7B's architecture, memory and time; banding and colour after the split unchanged, fringes
  and flicker lower.
- **(a) and (b):** with the sharp model the eyes' main complaint is gone. What remains goes both
  ways with the content (the 7B's sharper thin cel lines, the sharp's fused gradient colours, its
  clouds from compressed input), where a global strength would mend one case by harming another;
  a structure-aware strength stays the candidate if a scale is wanted later.
- **ByteDance's warning, tested mildly:** a heavier degradation (d2: area ×½, x264 CRF 26) tames
  neither model at 4K; blur and noise are untested.
- **For "4K output's quality":** the sharp model still trails bicubic on PSNR-Y and VMAF on every
  shot; by eye it is the better of the two models on invented texture and level with the 7B on
  most lines.

## Caveats

- **Fidelity is not quality** ([numerics.md](numerics.md#caveats)): every score but DISTS and
  the detail measure rewards taking more from the input, which the GT is closer to in every
  band (step 0). DISTS, the Laplacian variance and the eyes are the guard on the model's detail.
- One model (7B fp16) with numz's numerics, the sharp 7B beside it in step 6; single shots of 45
  frames in one batch; 8 clips at 1080p (animation and live action), 7 at the heavier
  degradation, 5 at ×4, 4 at 4K from one film (13 more from three sources in step 5); 3 at ×1.5
  and ×3. Seeds: 3 at d1, 2–3 elsewhere.
- Factors tested: ×1.5, ×2, ×3, ×4 (×1.5 and ×3 on 3 clips, 2 seeds).
- The variants are scored on the decode in float32, as seedvr2x keeps it; numz's own `lab`
  masters (step 0) were clamped bfloat16.
- **Step 5 rests on one seed on 7 of its 13 shots:** seeds 42 and 43 on the four `sollevante-*`,
  `sunrise` and `space`, seed 42 alone on the other 7, whose verdicts rest on the bootstrap
  interval alone, with no seed band. VMAF and CAMBI of `ds4_0.75` and `ds4_0.25` are seed 42's,
  as are the first film's `ds4_0.25` and `ds4_0.75`, the 1080p `ds4_0.75` and `ds4_0.25`, and
  the sweep's lowest points (the two-seed mean picks the same A where it exists).
- **DISTS on 5 frames a clip** (every 9th, as in step 3): enough to compare variants on the same
  frames, not comparable with numerics.md's DISTS figures.
- **The Laplacian variance is a proxy for detail:** one number, blind to whether the detail is
  the GT's; the bands' correlations and the eyes are the check.
- **Bicubic stands in for A = 0 in the sweep:** the reference's lightness is bicubic's to 2
  decimals in every band, but bicubic also takes all of its colour from the input, where A = 0
  keeps the split's.
- **At 1080p, bicubic is scored on the 8 clips at d1 but on 2 of 7 at d2** (the other 5 have no
  bicubic master). The detail strength is tested at ×2 only (1080p to 4K, and 540p to 1080p):
  output sizes between 1080p and 4K, and other factors, are untested.
- **Full-reference metrics cannot reward a redraw better than the GT:** a line or a text drawn
  sharper than the GT counts as error. So on structure (cel lines, text, buildings, motion) they
  disagree with the eyes, who found `digital-jet`'s text better than the GT's; they agree where
  the model invents texture.
- **Step 5's eyes are one viewer's:** at 4K mostly one window per clip (the first of each
  folder; the `.stretch.png` twins kept the others out of view), at 1080p both windows of 5
  clips; one model (7B fp16), ×2 from the mild degradation (d1).
- **Step 6's limits:** the sharp 7B at one seed (42) on 12 of its 17 4K shots and on all 26
  1080p clips, at two only on the 5 shots of part (d); the comparison from d2 on 5 shots at one
  seed. d2 is only mildly heavier than d1 on these sources (bicubic's fine bands barely move), and
  ByteDance's heavier degradations (blur, noise) are untested. At 4K, ×2 only. The eyes on its
  crops are one viewer's, with no crops at ×1.5, ×3 or ×4.

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

Step 5. The batch A dumps come from runs made as the first film's 4K ones (step 1: encode
untiled, decode tiles of 2048, overlap 64), one per clip and seed; the rest runs on the CPU:

```bash
S=scripts; C=/path/to/clips; C4=/path/to/clips4k; D=/path/to/dumps; c=digital-sunrise
V=none,lab,split:ycc:4:3,split:ycc:3:3,split:ycc:4:3:ds2_0.5,split:ycc:4:3:ds2_0
V=$V,split:ycc:4:3:ds4_0.75,split:ycc:4:3:ds4_0.5,split:ycc:4:3:ds4_0.25
python3 $S/colour_eval.py score --clip $c-d1 --gt $C4/$c.gt.mkv --ref f32=$D/$c-d1/s42/ref_f32.pt \
  --content s42=$D/$c-d1/s42/decode.pt --variants $V --out eval-b1/$c-d1 --lpips --dists-every 9
python3 $S/colour_eval.py score --clip $c-d1 --gt $C4/$c.gt.mkv --master bicubic=$C4/$c.d1.bicubic.mkv \
  --master-content s42,s43 --out eval-b1/$c-d1 --lpips --dists-every 9      # bicubic, from its master
python3 $S/colour_eval.py render --clip $c-d1 --gt $C4/$c.gt.mkv --ref f32=$D/$c-d1/s42/ref_f32.pt \
  --content s42=$D/$c-d1/s42/decode.pt --variants ${V#none,} --out masters-b1 --pix-fmt gbrp16le  # all but none
python3 $S/colour_eval.py vmaf --clip $c-d1 --gt $C4/$c.gt.mkv --master bicubic=$C4/$c.d1.bicubic.mkv \
  --master-content s42,s43 --out vmaf-b1/$c-d1                                # VMAF and CAMBI
python3 $S/colour_eval.py vmaf --clip $c-d1 --gt $C4/$c.gt.mkv --master-content s42 --out vmaf-b1/$c-d1 \
  --master split:ycc:4:3:ds4_0.25@f32=masters-b1/$c-d1.s42.split_ycc_4_3_ds4_0.25~f32.gbrp16le.mkv  # one per variant
python3 $S/colour_eval.py summary eval-b1/digital-*-d1 vmaf-b1/digital-*-d1 --versus bicubic \
  --tables versus                                                    # against bicubic, per kind
python3 $S/colour_eval.py summary eval-b1/*-d1 vmaf-b1/*-d1 --baseline lab@f32      # against lab
python3 $S/colour_eval.py summary eval-b1/*-d1 vmaf-b1/*-d1 --baseline split:ycc:4:3@f32
python3 $S/colour_eval.py score --clip $c-d1 --gt $C4/$c.gt.mkv --ref f32=$D/$c-d1/s42/ref_f32.pt \
  --content s42=$D/$c-d1/s42/decode.pt --content s42-d512=$D/$c-d1/s42/dec/decode-512-64.pt \
  --untiled s42 --variants none,lab,split:ycc:4:3,split:ycc:3:3 --out eval-b1/$c-d1-tiled   # tiles
python3 $S/colour_bands.py selftest
python3 $S/colour_bands.py scan --clip face-d1 --gt $C4/face.gt.mkv --bicubic $C4/face.d1.bicubic.mkv \
  --ref $D/face-d1/s42/ref_f32.pt --content $D/face-d1/s42/decode.pt --rows 64:2096 --every 3 \
  --out bands/face-d1.json                               # --rows: inside the letterbox
python3 $S/colour_bands.py report bands/*.json > bands.md  # --detail: each clip's full tables too
python3 $S/colour_crops.py --clip digital-cockpit-d1 --gt $C4/digital-cockpit.gt.mkv \
  --bicubic $C4/digital-cockpit.d1.bicubic.mkv --ref $D/digital-cockpit-d1/s42/ref_f32.pt \
  --content $D/digital-cockpit-d1/s42/decode.pt \
  --variants split:ycc:4:3,split:ycc:3:3,split:ycc:4:3:ds2_0.5,split:ycc:4:3:ds4_0.5,split:ycc:4:3:ds4_0.25 \
  --at 1840,700 --at 2020,430 --at-frame 22 --size 512x288 --per-kind 1 --out crops-b1/digital-cockpit
# 1080p: the strengths on steps 1-3's dumps, bicubic from its master, then against bicubic
python3 $S/colour_eval.py score --clip anime-clean-d1 --gt $C/anime-clean.gt.mkv \
  --ref f32=$D/anime-clean-d1/s42/ref_f32.pt --content s42=$D/anime-clean-d1/s42/decode.pt \
  --variants split:ycc:4:3:ds4_0.75,split:ycc:4:3:ds4_0.5,split:ycc:4:3:ds4_0.25 \
  --out eval-ds/anime-clean-d1 --lpips --dists-every 9
python3 $S/colour_eval.py score --clip anime-clean-d1 --gt $C/anime-clean.gt.mkv \
  --master bicubic=$C/anime-clean.d1.bicubic.mkv --master-content s42 --out eval-ds/anime-clean-d1 \
  --lpips --dists-every 9
python3 $S/colour_eval.py summary eval/*-d1 eval-ds/*-d1 --versus bicubic --tables versus
python3 $S/colour_crops.py --clip anime-clean-d1 --gt $C/anime-clean.gt.mkv \
  --bicubic $C/anime-clean.d1.bicubic.mkv --ref $D/anime-clean-d1/s42/ref_f32.pt \
  --content $D/anime-clean-d1/s42/decode.pt \
  --variants split:ycc:4:3,split:ycc:4:3:ds4_0.75,split:ycc:4:3:ds4_0.5,split:ycc:4:3:ds4_0.25 \
  --at 800,864 --at 400,480 --at-frame 22 --size 384x216 --per-kind 0 --out crops-b1-1080p/anime-clean
```

Step 6. The sharp's runs are the 7B's job lines with `--dit_model` alone changed (its inputs
dumped once, `COLOUR_DUMP_INPUTS=1`, to check that they are the 7B's); `colour_clips.py` makes the
4K d2 inputs; each sharp decode is scored with the 7B run's reference, under the tag `sharp`, then
paired with the 7B's split:

```bash
S=scripts; C4=/path/to/clips4k; D=/path/to/dumps; N=/path/to/numz; c=digital-cockpit
python3 $S/colour_clips.py make $C4/$c --factor 2 --degrade d2 --out clips4k   # fr_clips.py's d2, at 4K
python3 $S/bench.py run colour-b2-$c-d1-sh42 --seedvr2-dir $N --wrap $S/colour_dump.py \
  --env COLOUR_DUMP=$D/$c-d1/sh42 --env COLOUR_DUMP_INPUTS=0 -- $C4/$c.d1.lr.mkv --output out/ \
  --dit_model seedvr2_ema_7b_sharp_fp16.safetensors --resolution 2016 --batch_size 45 --load_cap 45 \
  --color_correction none --seed 42 --vae_decode_tiled --vae_decode_tile_size 2048 \
  --vae_decode_tile_overlap 64                     # the 7B's line but --dit_model; 1080p: untiled
# from d2: both models on clips4k/$c.d2.lr.mkv, the 7B first (COLOUR_DUMP_INPUTS=1: its reference serves both)
python3 $S/colour_eval.py score --clip $c-d1 --gt $C4/$c.gt.mkv --bars 0:0 \
  --ref sharp=$D/$c-d1/s42/ref_f32.pt --content s42=$D/$c-d1/sh42/decode.pt \
  --variants none,split:ycc:4:3 --out eval-b2/$c-d1 --lpips --dists-every 9   # --bars: the 7B's own
python3 $S/colour_eval.py render --clip $c-d1 --gt $C4/$c.gt.mkv --ref sharp=$D/$c-d1/s42/ref_f32.pt \
  --content s42=$D/$c-d1/sh42/decode.pt --variants none,split:ycc:4:3 --out masters-b2 --pix-fmt gbrp16le
python3 $S/colour_eval.py vmaf --clip $c-d1 --gt $C4/$c.gt.mkv --master-content s42 --out vmaf-b2/$c-d1 \
  --master split:ycc:4:3@sharp=masters-b2/$c-d1.s42.split_ycc_4_3~sharp.gbrp16le.mkv   # one per variant
python3 $S/colour_eval.py summary eval-b1/$c-d1 vmaf-b1/$c-d1 eval-b2/$c-d1 vmaf-b2/$c-d1 \
  --baseline split:ycc:4:3@f32 --tables pairs      # the sharp - the 7B, frame by frame
python3 $S/colour_eval.py summary eval-b1/$c-d1 vmaf-b1/$c-d1 eval-b2/$c-d1 vmaf-b2/$c-d1 \
  --versus bicubic --tables versus                 # each model against bicubic
python3 $S/colour_bands.py scan --clip $c-d1 --gt $C4/$c.gt.mkv --bicubic $C4/$c.d1.bicubic.mkv \
  --ref $D/$c-d1/s42/ref_f32.pt --content $D/$c-d1/sh42/decode.pt --every 3 --out bands-b2/$c-d1-sharp.json
python3 $S/colour_crops.py --clip $c-d1 --gt $C4/$c.gt.mkv --bicubic $C4/$c.d1.bicubic.mkv \
  --ref $D/$c-d1/s42/ref_f32.pt --content 7B=$D/$c-d1/s42/decode.pt \
  --content sharp=$D/$c-d1/sh42/decode.pt --variants split:ycc:4:3 --no-stretch --per-kind 0 \
  --at 1840,700 --at 2020,430 --at-frame 22 --size 512x288 --out crops-b2/$c
```

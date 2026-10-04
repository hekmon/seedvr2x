# Colour correction beyond numz's `lab`

> Status: **in progress.** Step 0, the diagnosis, is measured with
> [`scripts/colour_diag.py`](../scripts/colour_diag.py) on the CPU, from the 16-bit masters of
> [numerics.md](numerics.md)'s full-reference runs. The study answers "Beyond numz's `lab`" in
> [DESIGN.md](../../seedvr2x/DESIGN.md#beyond-numzs-lab): steps 1 to 4 (dumps, variants, scores,
> crops and a brief) follow.

In short (10 clips: 8 at ×2 with the mild degradation d1, 2 with the heavier d2; numz's `lab`
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

## Method

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

## Where `lab`'s error sits

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

## The split at two scales (oracle, seed 42)

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

## Caveats

- **The oracle is indicative.** It works on numz's clamped bfloat16 masters, with the bicubic
  baseline instead of the encoder's input, at the à-trous scales only, in Y'CbCr only.
  Step 1 dumps the exact decode and encoder input. The detail measure is one number, the
  Laplacian variance. LPIPS, DISTS and the eyes come in step 3.
- **Fidelity is not quality** ([numerics.md](numerics.md#caveats)): the model is far from the GT
  everywhere, and an output closer to it is not necessarily nicer to watch.
- One model, one upscale factor (×2), 1080p, 45-frame single shots, one batch, untiled. Tiles,
  ×4 and 4K come in steps 1–3.

## Reproduce

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

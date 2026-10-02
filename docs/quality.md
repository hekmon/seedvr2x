# Quality options: colour correction, batches and overlap, noise, seed

> Status: **measured** on the reference stack (7B fp16, `flash_attn_2`, 1080p, PNG output),
> SeedVR2 `4490bd1`. What each option does in the code is in [cli-flags.md](cli-flags.md); this
> page summarizes it and measures the effect. There is no ground truth: every figure is a proxy,
> meaningful only against another configuration.

In short (1080p restoration of anime, 7B fp16):

- **The model alone drifts in colour** (+30% saturation, ±1–4 levels of brightness). The
  default `--color_correction lab` removes that drift without losing detail, and halves the
  low-frequency jumps at batch boundaries. `hsv` and `adain` are worse; `wavelet` is close.
- **Batches are independent, so each boundary is a visible jump** on still content: ≈ 2.5× the
  in-batch flicker at batch 21, growing with the batch size. One batch per shot avoids it.
- **`--temporal_overlap` only cross-fades with odd values ≥ 3:** 1, 2 and 4 cost compute and
  change nothing at the boundary; 3 cuts the jump by about a third.
- **`--prepend_frames` frames stay in the output on one GPU**, and the gain (the clip's first
  frame is otherwise less restored) is small. `--uniform_batch_size` helps a short last batch.
- **Both noise options degrade this content:** input noise turns into texture and grain, latent
  noise washes the image out (its timestep shift makes 0.1 already strong).
- **The seed barely matters** (42 dB between seeds) and runs are bit-reproducible.
- **VAE tiling** shifts whole tiles by up to ≈ 2 levels, visible on flat backgrounds; `lab`
  colour correction removes it (per-tile offsets ≤ 0.06 levels vs untiled).

## Method

Two clips from the same anime episode, both 1920×1080 in and out: at 1080p the model restores, it
doesn't upscale, so the output can be compared pixel for pixel with the input.

| Clip | Source | Content |
|---|---|---|
| A | `seg_0000`, from frame 48 (`--skip_first_frames 48`) | Fast motion and pans: input mean \|ΔY\| between frames 15–56 levels. Dark (mean L\* 20) |
| B | `seg_0001-0004`, from frame 20 | Animated on threes: two of every three frame pairs are **held** drawings (input mean \|ΔY\| 0.3–0.5, codec noise), the third changes (≈ 3). Dark (mean L\* 14) |

Unless a table says otherwise: `--color_correction none` (to see the model alone), seed 42,
`--batch_size 21`, 45 frames (21 for single-batch tests). The metrics come from
[`scripts/quality_metrics.py`](../scripts/quality_metrics.py), which reads the input like the CLI
(OpenCV, same seek):

| Metric | What it measures |
|---|---|
| PSNR in, SSIM in | Fidelity: output vs input (RGB PSNR; SSIM on luma). Higher = closer to the source, which is not "better": restoration is supposed to change the image |
| PSNR ref | Distance to a reference run (the same configuration with the option at its default) |
| ΔE lf | Mean CIEDE76 between output and input after a Gaussian blur (σ = 4 px): low-frequency colour and brightness error |
| Y shift | Mean luma of output − input, 8-bit levels |
| Lap var | Variance of the luma Laplacian: sharpness / high-frequency energy (input: 7.2 on A, 17.8 on B) |
| flat std | Luma std (9×9 window) inside flat input areas (input local std < 1.5): grain or noise the model adds (input: 0.74 on A, 0.55 on B) |
| hold ΔY | On clip B's held pairs, the output's mean \|ΔY\| between consecutive frames: since the drawing doesn't change, this is flicker. Split into transitions inside a batch and at batch boundaries (where the output switches DiT batch). Input: 0.45 |
| added lf | Mean \|Δ(output − input)\| between consecutive frames on 16×16-pixel block means: low-frequency change the input doesn't have (brightness/colour flicker), inside batches / at boundaries |

Times (DiT, Phase 4, total) are indicative only: the GPU is power-capped and its speed varied by
up to 40–60% between sessions ([benchmarking.md](benchmarking.md#caveats)), so compare them only
between runs made back to back.

## Colour correction

Phase 4, batch by batch, against the input batch rebuilt at the output size
([cli-flags.md](cli-flags.md#quality) has the details):

| Mode | What it does |
|---|---|
| `none` | Nothing: the decoded frames |
| `wavelet` | Output minus its 5-level blur pyramid (3×3 binomial kernel, dilations 1–16) + the input's low-pass: everything coarser than ≈ 30–60 px comes from the input, per frame and pixel |
| `lab` (default) | `wavelet`, then in CIELAB: a\* and b\* histogram-matched to the input, L\* = 0.8 × output + 0.2 × matched. The histograms pool every frame of the batch |
| `wavelet_adaptive` | `wavelet`, blended with `hsv` where the output is > 0.15 more saturated than the input |
| `hsv` | Saturation histogram matched per 30° hue bin; hue and value kept from the output. No wavelet step |
| `adain` | Per-frame, per-channel mean and std of the RGB output set to the input's |

Clip A, batch 21, 21 frames (one batch); Phase 4 figures are for one 21-frame 1080p batch:

| Mode | PSNR in | SSIM in | Y shift | ΔE lf | L\* a\* b\* mean | a\* b\* std | Lap var | flat std | Phase 4 time | Phase 4 torch peak |
|---|---|---|---|---|---|---|---|---|---|---|
| input | – | – | – | – | 19.0 5.7 −13.4 | 2.71 5.53 | 5.9 | 0.71 | – | – |
| `none` | 28.43 dB | 0.9502 | +1.39 | 4.44 | 19.6 6.3 −15.1 | 3.62 7.09 | 11.6 | 1.06 | 0.07 s | 0.27 GiB |
| `lab` | 38.02 dB | 0.9736 | −0.59 | 1.01 | 18.8 5.7 −13.4 | 2.78 5.63 | 11.3 | 1.02 | 0.6 s ¹ | 4.26 GiB |
| `wavelet` | 37.92 dB | 0.9734 | −0.59 | 1.04 | 18.8 5.7 −13.4 | 2.90 5.79 | 11.2 | 1.02 | 0.35 s | 1.75 GiB |
| `wavelet_adaptive` | 37.74 dB | 0.9724 | −0.53 | 1.05 | 18.8 5.7 −13.4 | 2.91 5.80 | 13.2 | 1.05 | 3.77 s | 5.58 GiB |
| `hsv` | 28.27 dB | 0.9545 | +2.89 | 4.13 | 20.3 5.8 −14.1 | 3.41 6.78 | 15.1 | 1.14 | 1.24 s | 4.12 GiB |
| `adain` | 32.46 dB | 0.9679 | −0.44 | 2.91 | 18.8 5.7 −13.3 | 3.04 5.88 | 9.9 | 0.98 | 0.30 s | 1.73 GiB |

¹ From the 45-frame run: 0.6 s per 21-frame batch, 1.46 s for the phase. Phase 4 adds nothing to the device memory
(NVML growth 0): its peak fits in the pool the VAE decode left.

Clip B (dark, held drawings), same settings:

| Mode | PSNR in | SSIM in | Y shift | ΔE lf | L\* a\* b\* mean | a\* b\* std | Lap var | flat std | Phase 4 time | Phase 4 torch peak |
|---|---|---|---|---|---|---|---|---|---|---|
| input | – | – | – | – | 13.4 1.6 −2.8 | 1.84 2.87 | 18.1 | 0.54 | – | – |
| `none` | 31.74 dB | 0.9022 | −3.66 | 2.03 | 12.0 1.4 −3.5 | 2.38 3.65 | 76.2 | 0.83 | 0.07 s | 0.27 GiB |
| `lab` | 35.10 dB | 0.9628 | −0.63 | 0.66 | 13.2 1.6 −2.8 | 1.86 2.90 | 75.7 | 0.87 | 0.5 s ¹ | 4.26 GiB |
| `wavelet` | 34.81 dB | 0.9612 | −0.64 | 0.67 | 13.2 1.6 −2.8 | 2.18 3.14 | 76.7 | 0.86 | 0.31 s | 1.75 GiB |
| `wavelet_adaptive` | 34.70 dB | 0.9493 | −0.75 | 0.69 | 13.2 1.5 −2.8 | 2.16 3.12 | 79.0 | 0.91 | 0.78 s | 5.34 GiB |
| `hsv` | 32.36 dB | 0.9226 | −2.11 | 1.58 | 12.7 1.1 −2.6 | 2.10 2.81 | 82.9 | 0.95 | 0.53 s | 3.88 GiB |
| `adain` | 34.46 dB | 0.9527 | −0.48 | 1.11 | 13.2 1.6 −2.8 | 2.13 3.18 | 68.4 | 0.82 | 0.29 s | 1.73 GiB |

¹ 1.24 s for the three batches of the 45-frame run.

And across batch boundaries (45 frames, batch 21, two boundaries):

| Run | Hold ΔY inside / at boundary | Added lf inside / at boundary |
|---|---|---|
| clip B, `none` | 0.96 / 2.31 | 0.73 / 1.54 |
| clip B, `lab` | 0.96 / 1.83 | 0.45 / 0.68 |
| clip A, `none` | – | 10.31 / 4.98 |
| clip A, `lab` | – | 1.90 / 1.00 |

- **The model alone shifts colours:** saturation +30% (a\*/b\* std 3.62/7.09 vs 2.71/5.53 on
  A, 2.38/3.65 vs 1.84/2.87 on B), b\* toward blue, and the brightness moves with the content
  (+1.4 levels on A, −3.7 on B). ΔE lf 4.4 on A.
- **`wavelet` and `lab` put the input's low frequencies back** without touching detail: ΔE lf
  4.4 → 1.0 (A) and 2.0 → 0.7 (B), L\*a\*b\* means within 0.2 of the input, PSNR in +9.5 dB (A)
  and +3.3 dB (B), same Laplacian variance and grain.
  The −0.59 level left is mostly the uint8 truncation (−0.5 on average, [cli-flags.md](cli-flags.md#output)).
  `lab`'s histogram step also matches the saturation: a\*/b\* std 1.86/2.90 vs 1.84/2.87 for
  the input on B, where `wavelet` leaves 2.18/3.14 (the pyramid only replaces what is coarser
  than ≈ 30–60 px, and the extra saturation is finer). On A the two are close.
- **Colour correction also halves the batch-boundary jumps:** it takes the low frequencies of
  every frame from the input, and the boundary jumps are mostly low-frequency (B: 1.54 → 0.68 on
  the low-frequency measure, 2.31 → 1.83 on held frames). The per-batch pooling of `lab`'s
  histograms adds no visible jump of its own.
- **`wavelet_adaptive`** is `wavelet` plus HSV in oversaturated areas: same figures, a little more
  high-frequency energy, 2.5–10× the time and 3× the memory of `wavelet`.
- **`hsv` doesn't fix brightness** and has no wavelet step: it lowers saturation at constant HSV
  value, which lifts the darker channels (A: +1.4 → +2.9 levels; B: −3.7 → −2.1). ΔE lf barely
  improves (4.4 → 4.1, 2.0 → 1.6), and it adds high-frequency energy (Lap var +9–30%).
- **`adain`** fixes the per-frame means but not their spatial distribution (ΔE lf 2.9 on A, 1.1
  on B), and its std matching lowers contrast (Lap var −10 to −15%).
- The cost is negligible next to the VAE and DiT (≤ 0.6 s per 21-frame batch, except
  `wavelet_adaptive` at 3.8 s), and Phase 4's ≤ 5.6 GiB fit in the pool the decode left.

## Batches and temporal consistency

Each batch goes through the VAE and the DiT on its own: nothing is shared between batches but
the seed (every batch gets the same diffusion noise). Inside a batch the DiT attends across
frames; at a boundary the output switches to an independent reconstruction.

Clip B, 45 frames, `--color_correction none`. "Hold ΔY" is the output's frame-to-frame change
on held drawings (input: 0.45), inside batches / at batch boundaries (number of held boundary
transitions); "added" is the change the input doesn't have, at full resolution and on 16-pixel
blocks:

| Batch | Batches | PSNR in | SSIM in | Lap var | flat std | Hold ΔY inside / boundary (n) | Added inside / boundary | Added lf inside / boundary | DiT time |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 45 | 30.69 dB | 0.8493 | 92.0 | 0.65 | – / 0.81 (29) | – / 1.07 | – / 0.46 | 133 s |
| 5 | 9 | 31.10 dB | 0.8634 | 66.4 | 0.65 | 0.91 / 1.71 (5) | 1.23 / 1.74 | 0.63 / 0.92 | 62 s |
| 21 | 3 | 31.69 dB | 0.8972 | 75.9 | 0.82 | 0.96 / 2.31 (2) | 1.42 / 2.26 | 0.73 / 1.54 | 50 s |
| 45 | 1 | 32.33 dB | 0.9081 | 62.6 | 0.87 | 0.97 / – | 1.39 / – | 0.72 / – | 39 s |
| 81 (first 45 of 81 frames) | 1 | 32.39 dB | 0.9104 | 67.5 | 0.93 | 0.94 / – | 1.40 / – | 0.72 / – | 61 s ² |

² For the 81 frames. Batch 81 vs batch 45 on the same 45 frames: 41.6 dB (worst frame 40.7).

- **Inside a batch, held frames flicker at ≈ 0.9–1.0, twice the input's codec noise, whatever
  the batch size.** The DiT doesn't freeze a held drawing; it re-renders it with small changes.
- **At a boundary the jump grows with the batch size:** 1.7 at batch 5, 2.3 at batch 21 (2.5×
  the in-batch flicker), 2.1× on the low-frequency measure. Larger batches give fewer but
  stronger discontinuities. One batch for the whole shot (batch 45 here) has none.
- **Batch 1 is the steadiest on held frames** (0.81, and the lowest low-frequency change, 0.46):
  every frame gets the same noise and a near-identical input, so a held drawing comes out
  near-identical. It has no temporal context, though: it is the least faithful (SSIM 0.85) and
  the sharpest (Lap var 92), and costs 2.1× the DiT time of batch 5 (3.4× batch 45).
  On clip A (fast motion), batch 1 is far from the batch-21 output (25.9 dB; batch 5: 26.1 dB),
  darker (Y shift −7.8 levels, batch 5 −4.0, batch 21 +1.4) and much less faithful (SSIM
  0.83, batch 5 0.91, batch 21 0.95): without temporal context the model handles motion
  differently. (The temporal measures are not usable there: the motion dominates them.)
- **Larger batches stay closer to the input** (PSNR in +1.6 dB and SSIM +0.06 from batch 1 to
  45), are less darkened (Y shift −5.6, −5.2, −3.8, −3.5 levels from batch 1 to 45, which
  colour correction removes anyway) and are faster per frame.
- **Beyond one batch per clip nothing changes:** batch 81 on the first 45 of 81 frames is
  41.6 dB from batch 45, with the same flicker and fidelity.
- Every batch's first frame is encoded alone by the causal VAE and comes out less restored
  (see [`--prepend_frames`](#--prepend_frames)).

## `--temporal_overlap`

Batch k starts at k × (batch − overlap), so each overlap frame is computed twice. In Phase 3 the
first `overlap` frames of a batch are cross-faded with the previous batch's last ones, with
weights that are linear below 3 and a raised cosine over the **middle third** of the overlap
from 3 ([cli-flags.md](cli-flags.md#temporal)):

| Overlap | Weight of the previous batch on the overlap frames | Frames really mixed |
|---|---|---|
| 1 | 1 | none: the new batch's first frame is discarded |
| 2 | 1, 0 | none: hard switch one frame later |
| 3 | 1, 0.5, 0 | one, 50/50 |
| 4 | 1, 1, 0, 0 | none: hard switch two frames later |
| 5 | 1, 1, 0.5, 0, 0 | one, 50/50 |

Measured on clip B (batch 5: 25 frames; batch 21: 45 frames). "Boundary" = the transitions
where the frame source really switches (from the weights above); the frames computed include
the 4n+1 padding of the last batch:

| Batch | Overlap | Batches | Frames computed | Hold ΔY inside / at boundary (n) | Added lf inside / at boundary | PSNR vs overlap 0 (worst frame) | Lap var |
|---|---|---|---|---|---|---|---|
| 5 | 0 | 5 | 25 | 0.93 / 1.75 (3) | 0.66 / 0.92 | – | 67.4 |
| 5 | 1 | 6 | 30 | 0.88 / 1.78 (4) | 0.66 / 1.03 | 38.0 dB (33.9) | 68.1 |
| 5 | 2 | 8 | 40 | 1.17 / – (0) ¹ | 0.72 / 1.47 | 36.5 dB (33.3) | 65.4 |
| 5 | 3 | 11 | 55 | 0.86 / 1.18 (13) | 0.58 / 0.71 | 38.2 dB (33.4) | 60.9 |
| 21 | 0 | 3 | 47 | 0.96 / 2.31 (2) | 0.73 / 1.54 | – | 75.9 |
| 21 | 1 | 3 | 47 | 0.95 / 2.36 (2) | 0.71 / 1.51 | 39.4 dB (35.1) | 74.9 |
| 21 | 2 | 3 | 51 | 0.93 / 2.40 (2) | 0.70 / 1.58 | 38.8 dB (31.7) | 74.6 |
| 21 | 3 | 3 | 51 | 1.00 / 1.48 (2) | 0.68 / 1.17 | 38.3 dB (33.8) | 75.0 |
| 21 | 4 | 3 | 55 | 0.96 / 2.97 (1) | 0.69 / 1.82 | 39.6 dB (31.4) | 74.6 |

¹ With batch 5 and overlap 2 every switch falls on a drawing change of the input (the clip
changes every third frame, batches start every third frame), where flicker can't be told from
motion.

- **Overlap 1, 2 and 4 do nothing for continuity**, as the weights predict: the boundary jump is
  the same as without overlap (2.31 → 2.36, 2.40, 2.97 at batch 21; 0.92 → 1.03, 1.47 on the
  low-frequency measure at batch 5), for 0–17% more compute at batch 21 and up to 60% at batch 5.
  The output still changes (38–40 dB from overlap 0): the batches start elsewhere.
- **Overlap 3 helps, partly:** the one frame mixed 50/50 splits the jump in two, about −35% per
  transition (2.31 → 1.48 at batch 21, 1.75 → 1.18 at batch 5). At batch 5 it costs 2.2× the
  compute and blurs (Lap var −10%, the 50/50 frames average two reconstructions); at batch 21 it
  costs +8%.
- Overlap 5 (one frame mixed) or 6 and more (two or more frames mixed) would follow the same
  logic; not measured.

## `--prepend_frames`

Prepends N frames mirrored around frame 0 (frames N…1, then 0…) so that the clip's first frame
is not the first frame of a batch. **On one GPU the prepended frames are not removed**
([cli-flags.md](cli-flags.md#temporal)): confirmed, 45 input frames with `--prepend_frames 4`
gave 49 PNGs, the first four computed from the mirrored frames. Drop them yourself (the metrics
below do), or the video starts with a back-and-forth twitch and is 4 frames longer.

Why one would want it: the first frame of every batch is encoded alone by the causal VAE (latent
frame 0) and comes out **less restored** than the others, i.e. closer to the input:

| Clip B | PSNR in, frame 0 of the clip | PSNR in, first frame of a batch (mean) | PSNR in, other frames | Lap var, first frame of a batch / others |
|---|---|---|---|---|
| batch 21 | 33.36 dB | 32.59 dB | 31.68 dB | 74.4 / 76.0 |
| batch 21, `--prepend_frames 4` | 32.21 dB | 32.17 dB | 31.73 dB | – |
| batch 5 | 31.42 dB | 31.41 dB | 31.04 dB | 65.1 / 66.8 |
| batch 5, `--prepend_frames 4` | 30.95 dB | 31.44 dB | 31.03 dB | – |

(On clip A, whose first frames move little, frame 0 is at 36.0 dB against 27–29 dB for the rest
of its batch.)

- With 4 prepended frames, frame 0 becomes an ordinary frame (32.2 dB instead of 33.4 dB at
  batch 21). The effect is small on this content; it may matter more where the first frame
  is visibly softer.
- Prepend only fixes the clip's first frame: every later batch still starts with a lone latent
  frame. All batch boundaries move by N frames; their jumps are unchanged (hold ΔY at
  boundaries 2.39 vs 2.31 at batch 21, 1.68 vs 1.75 at batch 5).
- Cost: N more frames to compute.

## `--uniform_batch_size`

Without it, a short last batch is only padded to the next 4n+1 with mirrored frames; with it, to
the full batch size. Clip B, 45 frames at batch 21: the last batch has 3 frames, computed as 5
or as 21.

| Clip B, batch 21 | Last 3 frames: PSNR in | Hold ΔY at the last boundary | Added lf at the last boundary | PSNR vs default, last 3 frames | DiT time | Total |
|---|---|---|---|---|---|---|
| default (3 → 5 frames) | 31.25 / 30.43 / 30.62 dB | 2.46 | 1.80 | – | 50.4 s | 212 s |
| `--uniform_batch_size` (3 → 21) | 32.65 / 31.78 / 32.14 dB | 1.94 | 1.10 | 37.8 / 35.3 / 35.5 dB | 58.9 s | 256 s |

- The first 42 frames are bit-identical; only the short batch changes. Its frames end up
  less altered (+1.4 dB vs input) and the jump into it is smaller (−20% on held frames, −40% on
  the low-frequency measure): the 3 real frames get 18 mirrored frames of context instead of 2.
- Cost: the short batch is computed at full size (+21% total here). Worth it when the last
  batch is much shorter than the others; useless when the frame count is a multiple of the batch
  size.

## Noise scales

- **`--input_noise_scale s`** adds Gaussian noise to the resized input before encoding:
  x + 0.025·s·N(0, 1) in [−1, 1], i.e. σ = 1.6 levels (8-bit) at 0.5 and 3.2 at 1.0. Drawn from
  the seed + 1 000 000 stream, so it is reproducible.
- **`--latent_noise_scale s`** pulls the encoded input latent (the DiT's condition) toward a weak
  noise (std ≈ 0.11): (1 − t)·latent + t·noise, with t = shift·s / (1 + (shift − 1)·s). The shift
  is computed from the latent's (h, w, c) instead of (frames, h, w)
  ([cli-flags.md](cli-flags.md#quality)): ≈ 5 at 1080p, so t = 0.36, 0.63 and 0.83 for s = 0.1,
  0.25 and 0.5. The condition keeps 64%, 37% and 17% of its amplitude.

Clip A, batch 21, 21 frames, `--color_correction none`; PSNR ref = vs the same run without noise:

| Option | PSNR in | SSIM in | PSNR ref (worst frame) | Y shift | ΔE lf | Lap var | flat std |
|---|---|---|---|---|---|---|---|
| none | 28.43 dB | 0.9502 | – | +1.39 | 4.44 | 11.6 | 1.06 |
| `--input_noise_scale 0.5` | 27.07 dB | 0.9062 | 33.3 dB (30.7) | −1.69 | 5.83 | 14.5 | 1.05 |
| `--input_noise_scale 1.0` | 26.28 dB | 0.8828 | 30.9 dB (27.9) | −2.44 | 6.36 | 30.0 | 1.45 |
| `--latent_noise_scale 0.1` | 29.49 dB | 0.9592 | 35.4 dB (32.5) | +2.94 | 3.91 | 8.5 | 1.01 |
| `--latent_noise_scale 0.25` | 25.47 dB | 0.8817 | 24.4 dB (22.3) | +8.63 | 7.69 | 5.2 | 0.93 |
| `--latent_noise_scale 0.5` | 17.17 dB | 0.7051 | 16.5 dB (15.0) | +23.87 | 19.87 | 8.4 | 0.78 |

- **Input noise is not subtle:** σ = 1.6–3.2 levels on the input change the output as much as
  a different model (31–33 dB). The model reads the noise as texture: at 1.0 the Laplacian
  variance is 2.6× higher and flat areas get 37% more grain, and SSIM against the input drops
  from 0.95 to 0.88. It also darkens (−3 to −4 levels vs no noise).
- **Latent noise washes the image out:** with less condition the DiT drifts toward a lighter,
  flatter image. 0.1 is already +1.5 levels brighter and 27% softer (Lap var 8.5 vs 11.6);
  0.25 is +7 levels and twice softer; 0.5 is +22 levels (ΔE 20), unusable. `--color_correction`
  would pull the low frequencies back to the input, not the lost detail.
- Neither is a "denoise" or "detail" knob on this content; keep both at their default, 0.

## Seed

`set_seed(seed + 1 000 000)` before Phase 1 (VAE posterior sampling and input noise), `set_seed(seed)`
before every DiT batch: all batches of a run get the same diffusion noise.

| Clip A, 21 frames | PSNR vs seed 42 (worst frame) | PSNR in | Lap var |
|---|---|---|---|
| seed 42 again (21-frame run vs the first 21 frames of the 45-frame run) | bit-identical | 28.43 dB | 11.6 |
| seed 43 | 42.4 dB (40.9) | 28.26 dB | 10.7 |
| seed 1234 | 42.4 dB (40.8) | 28.35 dB | 10.5 |

- **The seed matters little:** 42.4 dB between seeds, the same order as fp8 vs fp16 weights or
  1024-pixel decode tiling ([vram.md](vram.md#other-models)), far less than any option above.
  Changing the seed is not a way to get a visibly different restoration.
- **Runs are reproducible, and a batch doesn't depend on the clip length:** a 21-frame run is
  bit-identical to the first batch of the 45-frame run (same seed, untiled VAE).

## VAE tiling on flat areas

[vram.md](vram.md#quality-tiled-vs-untiled-1080p-batch-9-png-output) found no seams but a
per-tile brightness drift. Here on clip A (dark, mostly flat backgrounds), 21 frames, default
tiles (1024 px, 128 px overlap: a 2×2 grid at 1080p), vs untiled:

| Tiling | PSNR vs untiled (worst frame) | Mean signed diff | Per-tile mean offset, all pixels (min…max, std) | Same, flat input areas only | Boundary / interior abs. diff |
|---|---|---|---|---|---|
| decode | 37.9 dB (34.7) | −1.03 | −1.82 … +0.98 (1.12) | −2.05 … +0.55 (1.10) | 2.50 / 2.41 (x) |
| encode + decode | 37.4 dB (34.4) | −0.51 | −1.09 … +0.08 (0.45) | −1.14 … 0.00 (0.46) | 2.59 / 2.48 (x) |
| decode, both with `lab` | 48.9 dB (45.6) | +0.01 | −0.00 … +0.07 (0.03) | +0.03 … +0.06 (0.01) | 0.58 / 0.54 (x) |

- Same picture as on the earlier clip: no seam (the difference is not concentrated at tile
  edges), and whole tiles shifted by up to ≈ 2 levels, here mostly darker.
- **Flat areas see the same offsets as the rest, not larger ones**, but that is where a 2-level
  step between neighbouring tiles is visible: on a dark, flat background, a 2×2 grid of slightly
  different shades, blended over the 128-px overlap.
- **Colour correction removes the drift:** `lab` takes the low frequencies from the (untiled)
  input, so tiled and untiled outputs, both corrected, differ by 0.07 level per tile at most
  (48.9 dB). The drift only matters with `--color_correction none` (or `hsv`, which has no
  wavelet step).
- This clip is 4 tiles; smaller tiles mean more tiles and more drift (512 px: 2–4 levels,
  [vram.md](vram.md#quality-tiled-vs-untiled-1080p-batch-9-png-output)).

## Recommendations

For 1080p restoration of anime-like content with the 7B model (other content and resolutions
not measured):

| Option | Recommendation |
|---|---|
| `--color_correction` | Keep **`lab`** (default): it removes the model's colour, saturation and brightness drift without touching detail, and halves the low-frequency jumps at batch boundaries. `wavelet` is a cheaper second best (leaves some extra saturation). Avoid `hsv` (no brightness fix) and `adain` (flattens); `wavelet_adaptive` brings nothing over `lab` here. `none` keeps the model's +30% saturation and level shift |
| `--batch_size` | As large as VRAM allows, ideally **one batch per shot**: closer to the input, no boundary jumps inside the shot. With several batches per shot, larger batches mean fewer but stronger jumps. Batch 1 is steady on held frames but slow and least faithful |
| `--temporal_overlap` | 0, or **3** (or another odd value ≥ 3) if boundary jumps show: only odd overlaps from 3 up actually cross-fade a frame. 1, 2 and 4 cost compute for nothing |
| `--prepend_frames` | Optional (makes the clip's first frame an ordinary frame). On one GPU, **cut the first N output frames yourself** |
| `--uniform_batch_size` | On when the last batch is much shorter than the others: smaller jump into it, less altered frames, at the cost of computing a full batch |
| `--input_noise_scale`, `--latent_noise_scale` | 0. Input noise turns into texture and grain; latent noise (overly strong because of its timestep shift) washes the image out |
| `--seed` | Anything; it barely matters (42 dB between seeds), and a fixed seed reproduces bit for bit |
| VAE tiling | Untiled if it fits; otherwise the largest tiles that fit. With `lab`/`wavelet` colour correction the per-tile drift disappears |

## Reproduce

```bash
# a run (bench.py record + PNG frames)
python3 scripts/bench.py run q-b-bs21 --seedvr2-dir /path/to/seedvr2 --runs-dir runs -- \
  seg_0001-0004.mp4 --output out/q-b-bs21/ --output_format png --model_dir /path/to/models \
  --dit_model seedvr2_ema_7b_fp16.safetensors --resolution 1080 --attention_mode flash_attn_2 \
  --skip_first_frames 20 --load_cap 45 --batch_size 21 --color_correction none
# its metrics: input read like the CLI does, batch boundaries from --batch / --temporal-overlap / --prepend
python3 scripts/quality_metrics.py out/q-b-bs21/seg_0001-0004 --input seg_0001-0004.mp4 --skip 20 \
  --batch 21 --json metrics/q-b-bs21.json
# vs a reference run; prepended frames kept in the output: --drop-first N; tiles: --tile 1024 --tile-overlap 128
python3 scripts/quality_metrics.py out/q-b-bs21-pp4/seg_0001-0004 --input seg_0001-0004.mp4 --skip 20 \
  --batch 21 --prepend 4 --drop-first 4 --ref out/q-b-bs21/seg_0001-0004 --json metrics/q-b-bs21-pp4.json
python3 scripts/quality_metrics.py --summary metrics/*.json   # Markdown table
```

<details>
<summary>Run names (7B fp16, <code>flash_attn_2</code>, 1080p, PNG; clip A = <code>seg_0000</code>
from frame 48, clip B = <code>seg_0001-0004</code> from frame 20)</summary>

`q-a-bs21-none` (45 frames, the clip A reference), colour `q-a-cc-{lab (45 frames),wavelet,wavada,hsv,adain}`
and `q-b-cc-{lab (45 frames),wavelet,wavada,hsv,adain}` (21 frames), noise
`q-a-in{0.5,1}`, `q-a-ln{0.1,0.25,0.5}`, seed `q-a-seed{43,1234,42-r2}`, tiles `q-a-dt1024`,
`q-a-t1024`, `q-a-dt1024-lab`, batch size `q-b-bs{1,5,21,45}` (45 frames), `q-b-bs81` (81),
`q-a-bs{1,5}` (21), overlap `q-b-bs5-ov{1,2,3}` (25 frames), `q-b-bs21-ov{1,2,3,4}` (45),
prepend `q-b-bs5-pp4` (25), `q-b-bs21-pp4` (45), `q-b-bs21-uni` (45). All with
`--color_correction none` except the colour runs and `q-a-dt1024-lab`.

</details>

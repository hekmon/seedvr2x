# Scene cuts inside a processing unit

> Status: **measured** on the reference stack (7B fp16, `flash_attn_2`, 1080p output,
> `--color_correction none`, seed 42), SeedVR2 `4490bd1`, against a ground truth with the
> full-reference protocol of question 1 (degradation d1), for the open questions on shot
> boundaries in [DESIGN.md](../../seedvr2x/DESIGN.md#open-questions). Tools:
> [`scripts/cut_metrics.py`](../scripts/cut_metrics.py) (analysis),
> [`scripts/fr_clips.py`](../scripts/fr_clips.py) `slice` (frame-exact sub-clips),
> [`scripts/fr_metrics.py`](../scripts/fr_metrics.py) (per-frame metrics),
> [`scripts/blend_patch.py`](../scripts/blend_patch.py) `STITCH_WINDOWS` (explicit latent-window
> layouts). Three real hard cuts, all from a dark shot to a bright one or back.

In short (three hard cuts, 81-frame clips with the cut at 21 + k, k = 0…3, scored against the
ground truth; the reference is the two shots run separately, what correct shot detection gives):

- **A missed cut costs the first frames of the next shot.** In one batch across the cut, B's
  first frame loses 2.4–9.6 dB of PSNR-Y on the bright and dark cuts, its next three 1.5–4.7 dB,
  and VMAF drops 2–12 points over its first four frames; the clean cut loses far less (0.1–3.9 dB
  on B's first frame only).
  The loss is measurable for up to 13 frames after the cut in PSNR-Y (4–6 bright, 8–13 dark,
  0–2 clean), 1–4 in VMAF, at most 2 in DISTS. Up to 5% of the previous shot's last frame shows in B's first frame (ghost coefficient).
- **The offset k in the 4-frame latent group moves the damage, it doesn't remove it.** At k = 0
  (cut on a latent boundary) only B suffers; from k = 1 on, A's last frame shares a latent with B
  and loses up to 2.3 dB and 16 VMAF points, with 1–3% of B in it.
- **What carries the previous shot over is the causal VAE, not the DiT.** With one VAE pass and a
  hard DiT boundary exactly at the cut, A is bit-identical to A run alone and B is hit hardest
  of all (−7.9 / −16.8 dB on its first frame, 9–11% ghost). Letting the DiT see both shots halves
  that. **Latent stitching across a cut is as good as one batch** (mid-window or shared-zone
  layouts within 0.6 dB of it on B's first frame, or better), **and much worse with a window
  boundary at the cut.**
- **A false cut costs no fidelity:** splitting a continuous shot in two leaves every metric within
  the noise band next to the split, the frames after it are even 0.3–1.7 dB closer to the ground
  truth; what remains is a low-frequency temporal step at the split on 2 of 4 clips.
- **Very short shots are better alone than merged across their cut, from n = 1 frame on:** 3.6 /
  7.8 dB better on a single frame, still 1.9 / 2.8 dB at 9 frames, and within 0.6 dB (bright) or
  2 dB (dark) of the same frames inside a long run.
- **For detection, a miss costs far more than a false cut:** 7–31 dB·frames of PSNR-Y over the
  next shot's first 8 frames on the bright and dark cuts (about 0 on the clean one), against a
  small gain for a false cut ([costs per error](#what-it-means-for-shot-detection)).

## Why it matters for seedvr2x

seedvr2x processes a video shot by shot ([DESIGN.md](../../seedvr2x/DESIGN.md), "Pipeline, per
shot"), so its shot detection decides where every processing unit starts. Three mechanisms can carry
one shot into the next when a cut falls inside a unit:

- **The VAE packs 4 frames per latent:** latent 0 is frame 0 alone, latent j covers frames
  4j − 3 … 4j. A cut that is not on a latent boundary puts both shots in one latent.
- **The VAE is causal and streams:** one pass carries its convolution caches from slice to slice
  (decoder reach ≈ 37 latents, encoder ≈ 28, from the code), so a pass across a cut carries the
  previous shot into the next one, and never the other way.
- **The DiT attends in windows of up to 30 latents** (`wt = ceil(min(t, 30) / 4)` latent frames,
  shifted every other layer, [attention.md](attention.md#windowed-attention)): inside a batch or a
  latent window, both shots see each other, in both directions.

A detector can fail two ways. A **missed cut** leaves a cut inside a unit, at any offset in its
4-frame latent group. A **false cut** splits a continuous shot into two units. Merging very
**short shots** into a neighbour, which sptenc does for its 5-second segments, also puts a cut inside a unit.
This page measures the three costs against the ground truth, so that a detection threshold can be
weighed against them.

## Method

**Cuts** (0-based frame indices; scene score as sptenc's `scdet`; Y = mean BT.709 luma, 8-bit):

| Cut | Source | Score (largest other in the 132 frames around it) | Shot A → shot B mean Y | Motion A / B (mean \|ΔY\|) | Detail (Laplacian var) |
|---|---|---|---|---|---|
| bright | flat-colour cartoon, web H.264 1080p | 48.2 (4.8) | 49 → 188 | 2.5 / 3.3 | 308 |
| dark | dark anime, HEVC 1080p segment | 41.4 (0.25) | 156 → 33 | 0.8 / 0.8 | 5 |
| clean | clean digital anime, Blu-ray H.264 1080p | 32.0 (1.9) | 52 → 143 | 1.2 / 5.7 | 35 |

**Clips.** Per cut, one range of 132 frames, from 48 before the cut to 84 after it, was made once
with `fr_clips.py make` (ground truth: 16-bit RGB; input: half size, Mitchell, x264 CRF 20, fed
as 8-bit RGB), then cut into frame-exact slices with `fr_clips.py slice`. Every slice's framemd5
equals the range's, so every run below reads bit-identical input frames. The long range was
checked against the source by framemd5 (132/132), and cv2 reads every input bit-exactly as the
CLI does.

- **The 81-frame clip at offset k** (k = 0 … 3) puts the cut at clip frame 21 + k. Latent 6
  (frames 21 … 24) holds the cut: at k = 0 it is pure B; at k = 1, 2, 3 it holds k frames of A.
- **A** = its 21 + k frames before the cut; **B** = its 60 − k frames from the cut.
- **Short shots:** B's first n frames, n = 1, 2, 3, 5, 9, run alone. **Merged:** the same frames
  inside the 81-frame batch at k = 0 (after A, with the rest of B after them).
- **False cuts:** the four 45-frame single-shot clips of question 1, split 21 + 24.

**Runs** (one batch each, `--batch_size` = clip length, `--temporal_overlap 0`; lossless masters
through [`ffv1_out.py`](../scripts/ffv1_out.py)):

| Run | What it is |
|---|---|
| one | the 81 frames in one batch: the cut inside one VAE pass and one DiT batch |
| aligned (the reference) | A alone and B alone, their outputs joined losslessly: what correct shot detection gives |
| latent (iii) hard | one VAE pass, DiT windows `0-6,6-12,10-16,14-21` (latents): a hard DiT boundary at latent 6, 2 shared latents elsewhere: the VAE carries A into B, the DiT doesn't |
| latent (ii) shared | `0-7,5-11,9-15,13-19,17-21`: latents 5 and 6 shared by two windows, cross-faded (cosine, 0.75 / 0.25) |
| latent (i) mid | `0-5,3-9,7-13,11-17,15-21`: window 3–9 centred on the cut, latents 5 and 6 inside it |
| short alone | B's first n frames as their own batch (merged: the same frames inside "one" at k = 0) |
| split | a question-1 clip as two batches, 21 + 24 frames, joined |

The latent runs (bright and dark cuts, k = 0) stitch like [stitching.md](stitching.md#latent-space-stitching)'s
6-latent windows sharing 2 (cosine weights, as there), with the layout placed around the cut.

**Measures** ([`cut_metrics.py`](../scripts/cut_metrics.py) on [`fr_metrics.py`](../scripts/fr_metrics.py)'s
per-frame JSONs). Frame t sits at signed distance d = t − c from the cut (c = B's first frame:
d = −1 is A's last frame, d = 0 B's first). A temporal error belongs to its second frame (d = 0 is
the transition across the cut).

- **Deficit:** how much worse the run is than the aligned reference at each frame, positive =
  worse: PSNR-Y (dB), DISTS, VMAF, low-frequency temporal error (T-err lf: the frame-to-frame
  change the ground truth doesn't have, on 16×16 block means), plus SSIM-Y, LPIPS and CIEDE2000
  lf in the JSONs.
- **Noise band:** the CLI is deterministic, so a same-seed rerun is bit-identical and measures
  nothing; two seeds are two equally valid renderings. The band is the 0.95 quantile of per-frame
  |seed-to-seed| differences, pooled over question 1's default runs (4 clips × seeds 42, 43,
  1234: 12 pairs × 45 frames) and the aligned references at k = 0 run with seeds 42 and 43 (3
  pairs × 81 frames).
- **Latent phase:** B run alone starts its own 4-frame grid at B's first frame, one to three
  frames off the 81-frame run's grid (they agree only at k = 3). That puts a period-4 pattern in
  per-frame differences that has nothing to do with the cut. Deficits and seed differences are
  therefore averaged over 4 frames (one latent), never across the cut, before any comparison
  with the band.
- **Reach:** from the cut, the frames until 4 consecutive averaged deficits fall inside the band,
  after the cut and before it (0 = not measurably worse next to the cut). Isolated exceedances
  far from the cut are counted apart. One batch and a shot run alone also differ for reasons
  other than the cut: batch length, DiT windows, noise by position.
- **Ghost coefficient α:** the share of the other shot in the output,
  α_t = argmin_α ‖out_t − ((1 − α) GT_t + α GT_other)‖², on 8×8 block means of luma, with GT_other
  = the other shot's frame next to the cut. Synthetic mixes 0.7 GT_t + 0.3 GT_other give 0.3000
  (± 0.0001 with 2-level noise); a blurred ground truth (no ghost) gives 0.0016 on block means,
  0.008 at full resolution. What is reported is the **excess** over the aligned reference's own α
  (its errors correlate with the shots' difference, e.g. −0.02 on bright B frames).

## Results

### A missed cut: one batch across the cut, by offset k

The 81 frames in one batch against the aligned reference (A alone + B alone). Deficits by
distance d to the cut, positive = the batch across the cut is worse; reach = frames measurably
worse next to the cut (4-frame averages beyond the noise band), before / after it; α8 = the ghost
coefficient's excess (share of the other shot's frame next to the cut).

| k | Cut | PSNR-Y deficit, dB: d −1 / 0 / 1…3 / 4…7 / 8…15 | DISTS d 0…3 | VMAF d 0…3 | T-err lf at the cut | Reach before / after: PSNR-Y, DISTS, VMAF, T-err lf | α8 excess d −1 / 0 / 1…7 / 8…15 |
|---|---|---|---|---|---|---|---|
| 0 | bright | +0.44 / +4.14 / +3.42 / +1.11 / −0.37 | +0.020 | +11.6 | +4.82 | 0/6, 0/2, 0/4, 0/25 | +0.003 / +0.034 / −0.011 / +0.029 |
| 0 | dark | +0.53 / +9.64 / +3.67 / +2.59 / +1.64 | −0.001 | +4.7 | −0.78 | 0/13, 21/0, 0/2, 0/8 | +0.005 / +0.045 / +0.017 / +0.012 |
| 0 | clean | −2.21 / +0.10 / +0.53 / −0.96 / −0.93 | +0.009 | +3.7 | −0.43 | 0/0, 0/0, 0/2, 0/13 | +0.016 / −0.004 / −0.004 / +0.020 |
| 1 | bright | +2.08 / +3.02 / +2.35 / +0.45 / −0.34 | +0.004 | +7.3 | +0.72 | 2/5, 0/0, 2/3, 3/24 | +0.010 / −0.012 / −0.005 / +0.024 |
| 1 | dark | +0.46 / +4.01 / +1.50 / +1.84 / +1.08 | −0.006 | +3.4 | +0.18 | 0/12, 20/0, 0/2, 3/18 | +0.034 / −0.009 / +0.007 / +0.005 |
| 1 | clean | −2.01 / +1.54 / +0.09 / −0.66 / −0.63 | +0.008 | +2.2 | +1.70 | 0/1, 0/0, 0/1, 3/5 | −0.002 / −0.012 / −0.006 / +0.008 |
| 2 | bright | +1.80 / +4.85 / +2.97 / −0.30 / −0.62 | +0.004 | +6.4 | +2.68 | 0/4, 1/0, 0/3, 4/11 | +0.013 / −0.031 / −0.005 / +0.030 |
| 2 | dark | −1.14 / +4.49 / +4.72 / +1.68 / +0.57 | −0.013 | +1.9 | +0.37 | 0/10, 23/0, 0/1, 4/16 | +0.019 / −0.013 / +0.015 / +0.007 |
| 2 | clean | −2.61 / +3.04 / +0.01 / −0.82 / −0.96 | +0.002 | +2.5 | +4.96 | 0/1, 0/0, 0/1, 4/11 | +0.001 / −0.048 / −0.012 / +0.014 |
| 3 | bright | +0.43 / +2.35 / +1.93 / −0.37 / −0.33 | +0.006 | +2.7 | +2.40 | 0/4, 1/0, 3/1, 5/12 | −0.015 / −0.005 / +0.001 / +0.021 |
| 3 | dark | +2.32 / +8.84 / +3.46 / +1.01 / −0.46 | −0.005 | +2.8 | +6.89 | 24/8, 24/0, 1/1, 4/11 | −0.011 / −0.039 / +0.011 / +0.006 |
| 3 | clean | −0.24 / +3.89 / +1.05 / −0.77 / −0.84 | +0.013 | +4.3 | +5.87 | 0/2, 0/1, 0/2, 4/10 | −0.009 / −0.053 / −0.013 / +0.012 |

- **The cost lands on B's first frames.** B's first frame loses 2.4–9.6 dB of PSNR-Y on the
  bright and dark cuts at every offset, its next three frames 1.5–4.7 dB, and VMAF drops 2–12
  points over B's first four frames. DISTS hardly moves (within ±0.02). The clean cut suffers
  least: 0.1–3.9 dB on B's first frame, within the band from the next one on (k = 0, 1, 2).
- **How far it reaches:** PSNR-Y is measurably worse for 4–6 frames after the cut on the bright
  cut, 8–13 on the dark one, 0–2 on the clean one; VMAF for 1–4 frames, DISTS for at most 2. The
  low-frequency temporal error (T-err lf), whose band is narrow, stays different for 5–25 frames.
- **The offset moves the damage between the shots.** At k = 0 the cut sits on a latent boundary
  and A's last frames are no worse than the rest of A (the dark cut's A is worse throughout, see
  below). From k = 1 on, A's last frames share latent 6 with B: the last one loses up to 2.3 dB
  of PSNR-Y and up to 16 VMAF points (bright, k = 1), carries 1–3% of B (α8 at d = −1, k = 1 and
  2), and the low-frequency temporal error is measurably worse for A's last 3–5 frames, where it
  is clean at k = 0. B's first frame carries 3–5% of A at k = 0 on the bright and dark cuts (none
  on the clean one), and none at k ≥ 1.
- **No offset is cheap.** Summed over B's first 8 frames, the PSNR-Y cost falls with k on the
  bright cut (18.8, 11.9, 12.6, 6.7 dB·frames for k = 0…3) but not on the dark one (31.0, 15.9,
  25.4, 23.3); meanwhile A's last 4 frames go from −10 VMAF points·frames at k = 0 to +27 at
  k = 3 (bright). The clean cut stays within ±4 dB·frames at every k.
- On the dark cut, shot A is worse in one batch back to its first frame at every k (DISTS reach
  20–24 frames before the cut; +0.5 to +1.6 dB of PSNR-Y and +0.02 DISTS at k = 0), with no
  ghost: the DiT renders the bright shot differently with the dark one in view (the VAE, causal,
  can't carry B back into A).

### What carries one shot into the other: the VAE or the DiT

The three latent layouts, at k = 0 (latent 6 is pure B), against the aligned reference; one batch
for comparison; "one − (iii)" compares the one batch with layout (iii) directly. Deficits by
distance d to the cut, positive = worse; reach in frames, before / after the cut (on 4-frame
averages, beyond the noise band); α8 = the ghost coefficient's excess.

| Cut | Run | PSNR-Y deficit, dB: d 0 / 1…3 / 4…7 / 8…15 | DISTS d 0…3 | VMAF d 0…3 | Reach after: PSNR-Y / DISTS / VMAF / T-err lf | A's last 4 frames: PSNR-Y / DISTS | α8 excess d 0 / 1…7 / 8…15 |
|---|---|---|---|---|---|---|---|
| bright | one batch | +4.14 / +3.42 / +1.11 / −0.37 | +0.020 | +11.6 | 6 / 2 / 4 / 25 | +0.13 / −0.056 | +0.034 / −0.011 / +0.029 |
| bright | (i) mid | +4.68 / +2.69 / +0.60 / −0.76 | +0.023 | +13.8 | 5 / 2 / 5 / 13 | +1.57 / −0.022 | +0.041 / −0.000 / +0.024 |
| bright | (ii) shared | +4.61 / +2.54 / +0.61 / −0.60 | +0.020 | +13.4 | 5 / 2 / 6 / 25 | +0.79 / −0.051 | +0.038 / −0.002 / +0.028 |
| bright | (iii) hard | **+7.85 / +3.34 / +2.43 / +1.71** | **+0.061** | **+19.3** | **14 / 7 / 27 / 33** | 0 / 0 (bit-identical) | **+0.086 / +0.055 / +0.057** |
| bright | one − (iii) | −3.71 / +0.08 / −1.33 / −2.08 | −0.041 | −7.7 | 0 / 0 / 0 / 13 | +0.13 / −0.056 | −0.052 / −0.066 / −0.028 |
| dark | one batch | +9.64 / +3.67 / +2.59 / +1.64 | −0.001 | +4.7 | 13 / 0 / 2 / 8 | +0.59 / +0.018 | +0.045 / +0.017 / +0.012 |
| dark | (i) mid | +9.62 / +3.51 / +2.62 / +1.57 | +0.004 | +6.5 | 19 / 0 / 2 / 35 | −0.24 / +0.024 | +0.044 / +0.016 / −0.002 |
| dark | (ii) shared | +8.14 / +1.46 / +1.03 / +1.31 | −0.007 | +4.9 | 17 / 0 / 2 / 35 | −0.17 / −0.008 | +0.034 / +0.006 / −0.001 |
| dark | (iii) hard | **+16.81 / +12.71 / +7.20 / +2.81** | **+0.090** | **+33.3** | 18 / 4 / 6 / 26 | 0 / 0 (bit-identical) | **+0.110 / +0.047 / +0.008** |
| dark | one − (iii) | −7.17 / −9.04 / −4.62 / −1.16 | −0.091 | −28.6 | 0 / 0 / 0 / 0 | +0.59 / +0.018 | −0.065 / −0.030 / +0.004 |

- **The VAE alone carries the previous shot into the next.** With a hard DiT boundary exactly at the
  cut and one VAE pass (iii), A is untouched: its 21 frames are bit-identical to A run alone (the
  encoder and the decoder are causal, and the first window gets the same noise as A's own batch).
  B is hit hardest of all the runs: −7.9 and −16.8 dB on its first frame, a ghost of 9–11% of A's
  last frame (still 5–6% 8 to 15 frames later on the bright cut), and VMAF −19 and −33 on its
  first 4 frames.
- **Letting the DiT see across the cut makes it better, not worse.** The one batch, and the two
  layouts whose DiT window straddles the cut (i, ii), lose about half as much as (iii) on B's
  first frames (one − (iii): −3.7 and −7.2 dB on the first frame, −0.04 and −0.09 DISTS on the
  first 4). Part of (iii)'s loss is likely its B window starting on a 4-frame latent encoded with
  A in the causal cache, without the DiT context a normal batch has; [stitching.md](stitching.md#latent-space-stitching)
  found the same first-latent outlier on continuous content.
- **Latent stitching across a cut is as good as one batch**, provided the cut is not on a window
  boundary: (i) and (ii) within 0.6 dB of the one batch on B's first frame (dark (ii): 1.5 dB
  better). Both put A's last latents in a window with B: A's last 4 frames lose 1.6 dB (i) and
  0.8 dB (ii) on the bright cut, nothing on the dark one.

### The ghost: how much of the other shot shows

The ghost coefficient α (8×8 block means of luma, excess over the aligned reference) stays small
in one batch, at most +0.045 (4.5% of the other shot's frame next to the cut), and the cut's latent
offset decides where it shows:

- **k = 0:** in B's first frame, 3.4% (bright) and 4.5% (dark); none on the clean cut.
- **k ≥ 1:** none in B's first frame (excess −0.005 to −0.053); A's last frame takes 1.0–3.4% of B
  at k = 1 and 2, none at k = 3 (one B frame in the shared latent).
- After B's first frame the excess stays between −0.013 and +0.030 on every cut and offset,
  without a trend.

With a hard DiT boundary at the cut (layout iii), the VAE alone puts 9–11% of A into B's first
frame, still 5–6% 8 to 15 frames later on the bright cut. With the DiT seeing both shots (one
batch, layouts i and ii) the ghost is about half as large: attention across the cut does not add
to what the causal VAE carries over.

### Very short shots: alone or merged

B's first n frames run as a shot of their own ("alone"), against the same frames inside the one
batch across the cut at k = 0 ("merged": what a minimum shot length that folds them into the
previous shot does, with B's continuation in the same batch), and inside B run alone ("long", the
ideal when the shot goes on). Means over the n frames against the ground truth:

| Cut | n | PSNR-Y, dB: alone / merged / long | DISTS: alone / merged / long | VMAF: alone / merged / long |
|---|---|---|---|---|
| bright | 1 | **30.94** / 27.35 / 31.49 | **0.027** / 0.100 / 0.038 | **82.7** / 58.7 / 82.9 |
| bright | 2 | **31.44** / 27.52 / 31.32 | **0.023** / 0.077 / 0.037 | **83.4** / 64.9 / 82.8 |
| bright | 3 | **31.12** / 27.72 / 31.39 | **0.022** / 0.064 / 0.037 | **82.4** / 69.0 / 82.7 |
| bright | 5 | **31.13** / 27.93 / 31.25 | **0.022** / 0.053 / 0.038 | **83.2** / 72.7 / 82.5 |
| bright | 9 | **30.94** / 29.04 / 31.12 | **0.022** / 0.044 / 0.037 | **81.2** / 75.8 / 82.3 |
| dark | 1 | **37.84** / 30.07 / 39.71 | **0.181** / 0.213 / 0.178 | **70.0** / 57.4 / 67.7 |
| dark | 2 | **38.75** / 32.19 / 39.70 | **0.160** / 0.195 / 0.177 | **71.8** / 57.4 / 67.4 |
| dark | 3 | **38.88** / 33.52 / 39.68 | **0.156** / 0.182 / 0.177 | **70.7** / 59.4 / 66.5 |
| dark | 5 | **38.14** / 34.91 / 39.70 | **0.145** / 0.168 / 0.176 | **72.1** / 62.8 / 66.0 |
| dark | 9 | **38.72** / 35.94 / 39.67 | **0.157** / 0.158 / 0.179 | **70.8** / 66.6 / 65.1 |

- **Alone beats merged from n = 1 on, on every metric and both cuts:** a single frame run alone
  (a batch of 1, one latent) is 3.6 and 7.8 dB better than the same frame merged across the cut,
  DISTS 0.073 and 0.032 better, VMAF 24 and 13 points better. The gap narrows as n grows (the
  merged frames get further from the cut) but stays at n = 9: 1.9 / 2.8 dB.
- **A short shot alone loses little against the ideal:** at most 0.6 dB of PSNR-Y on the bright
  cut, 0.8–1.9 dB on the dark one, and its DISTS is *better* than the same frames inside the long
  run in 9 of 10 cases (VMAF within 1.1 points on the bright cut, 2–6 points better on the dark).
  A 1- to 9-frame batch is one to three latents, rendered with nothing else in view.

### False cuts: a continuous shot split in two

Question 1's four single-shot clips (45 frames), run as 21 + 24 frames and joined, against the
same clip in one batch (question 1's default run, seed 42), both against the ground truth. The
"cut" is the false boundary at frame 21:

| Clip | PSNR-Y deficit, dB: d −4…−1 / 0…3 / 4…7 / ≥ 8 | DISTS d −4…−1 / 0…3 | VMAF d −4…−1 / 0…3 | T-err lf deficit at the boundary | Reach after: PSNR-Y / DISTS / VMAF / T-err lf (before: 0 everywhere) |
|---|---|---|---|---|---|
| clean anime | −0.06 / −0.93 / −0.04 / −0.26 | +0.005 / +0.012 | −0.3 / −4.3 | +0.73 | 0 / 0 / 0 / 0 |
| grainy anime | +0.15 / −1.69 / −0.52 / −0.81 | −0.001 / −0.017 | −0.0 / −5.9 | +0.73 | 0 / 0 / 0 / 11 |
| dark anime | −0.13 / −1.60 / −0.07 / −0.35 | −0.007 / −0.011 | −2.2 / −6.7 | −1.50 | 0 / 0 / 0 / 0 |
| bright cartoon | +0.12 / −0.30 / +0.18 / −0.14 | −0.002 / −0.006 | +1.7 / −0.1 | −1.12 | 0 / 0 / 0 / 11 |

- **A false cut costs no fidelity.** No metric is measurably worse next to the false boundary (the
  largest deficit, DISTS +0.012 on the clean clip's first 4 frames after it, is inside the band,
  0.014). The frames right after the boundary are even *closer* to the ground truth than in one
  batch, on all four clips: 0.3–1.7 dB of PSNR-Y and 0.1–6.7 VMAF points over the first 4. The
  second part starts a batch of its own, as a shot would.
- **What it can cost is a step in time:** the low-frequency temporal error at the boundary changes
  by −1.5 to +0.7 (16×16 block means; seeds differ by 0.08), and stays measurably different for
  11 frames on two clips. Whether a step of that size is visible on a continuous shot is for the
  visual review.

## What it means for shot detection

The cost of each kind of detection error against correct boundaries, as sums of per-frame
deficits over a fixed window next to the boundary (positive = worse): B = the 8 frames after it,
A = the 4 before it; for a merged short shot, its own n frames. PSNR-Y in dB·frames, VMAF in
points·frames, DISTS in DISTS·frames:

| Error | Case | PSNR-Y: B / A | VMAF: B / A | DISTS: B |
|---|---|---|---|---|
| Missed cut, one batch | bright, k = 0 / 1 / 2 / 3 | 18.8 / 11.9 / 12.6 / 6.7 ; A 0.5 / 2.6 / 1.2 / −2.0 | 56.0 / 35.3 / 26.6 / 7.7 ; A −10.3 / 11.5 / 1.5 / 26.5 | 0.075 / 0.004 / −0.009 / 0.017 |
| | dark, k = 0 / 1 / 2 / 3 | 31.0 / 15.9 / 25.4 / 23.3 ; A 2.4 / 2.9 / 1.1 / 4.2 | −7.5 / −12.3 / −18.0 / −19.7 ; A −8.9 / −6.9 / −9.4 / 4.3 | −0.153 / −0.171 / −0.175 / −0.173 |
| | clean, k = 0 / 1 / 2 / 3 | −2.1 / −0.8 / −0.2 / 3.9 ; A −8.9 / −8.5 / −9.6 / −6.5 | −3.1 / −8.9 / −13.3 / −14.3 ; A −26.8 / −24.7 / −27.0 / −13.9 | 0.016 / 0.021 / −0.013 / 0.036 |
| Missed cut, latent windows (k = 0) | bright: (i) / (ii) / (iii) | 15.2 / 14.7 / 27.6 | 67.2 / 67.2 / 132.3 | 0.078 / 0.063 / 0.312 |
| | dark: (i) / (ii) / (iii) | 30.7 / 16.6 / 83.7 | −4.5 / −7.1 / 149.0 | −0.208 / −0.245 / 0.281 |
| False cut | clean / grainy / dark / bright | −3.9 / −8.8 / −6.6 / −0.5 ; A −0.2 / 0.6 / −0.5 / 0.5 | −20.9 / −32.8 / −9.6 / 4.5 | 0.086 / −0.125 / −0.022 / −0.016 |
| Short shot merged instead of alone | bright, n = 1 / 2 / 3 / 5 / 9 | 3.6 / 7.8 / 10.2 / 16.0 / 17.1 | 24.0 / 37.0 / 40.3 / 52.5 / 48.0 | 0.073 / 0.108 / 0.126 / 0.157 / 0.201 |
| | dark, n = 1 / 2 / 3 / 5 / 9 | 7.8 / 13.1 / 16.1 / 16.2 / 25.0 | 12.7 / 28.8 / 33.9 / 46.5 / 37.7 | 0.032 / 0.070 / 0.077 / 0.117 / 0.004 |

Read per event:

- **A missed cut** costs 7–31 dB·frames of PSNR-Y on the next shot's first 8 frames on two of the
  three cuts, and nothing measurable beyond B's first frame on the clean one. On the dark cut the
  metrics disagree past B's first frames (PSNR-Y worse, VMAF and DISTS better: see the caveats).
  A window boundary of latent stitching that falls on the missed cut (iii) costs 1.5–3× more.
- **A false cut** costs no fidelity: the PSNR-Y sums are gains on all four clips (−0.5 to −8.8
  dB·frames), the VMAF sums on three. What it can cost is a low-frequency temporal step,
  measurable for 0–11 frames on 2 of the 4 clips; whether it shows on a continuous shot is for the
  visual review. [stitching.md](stitching.md) measured the same kind of boundary (a batch
  boundary inside a shot, no overlap) on held drawings: a jump of 0.78 on top of an in-batch
  flicker of ≈ 1.0, which is visible. The fidelity "gain" fits
  [quality.md](quality.md#--prepend_frames)'s finding that a batch's first frame comes out less
  restored, i.e. closer to its input, which this protocol rewards because the model re-renders.
- **A short shot merged across its cut** costs 3.6–7.8 dB·frames for a single frame and 17–25 for
  9 frames, against running it alone: the merge is never the cheaper option here.

So, against these three cuts, the expected cost of a detector per hour of video is about
(misses per hour) × 7–31 dB·frames + (short shots merged per hour) × 4–25, plus (false cuts per
hour) × a temporal step of uncertain visibility and no fidelity loss. The error rates per threshold
come from question 3's labelled review.

## Caveats

- **Three cuts, all between a dark shot and a bright one** (mean luma 49 → 188, 156 → 33,
  52 → 143). A cut between shots of similar brightness and colour may leak less visibly, and the
  ghost coefficient is ill-conditioned there (it projects on the difference between the shots).
  No live action yet.
- **One rendering per configuration (seed 42).** The noise band comes from question 1's seed
  runs on other clips, pooled with the three cuts' aligned references at a second seed (15 pairs,
  783 values per metric). It is wide for PSNR-Y (0.82 dB on 4-frame averages) and very narrow for
  the low-frequency measures (CIEDE2000 lf 0.044, T-err lf 0.078), which flag almost any
  systematic change.
- **PSNR-Y on dark content is hypersensitive:** shot B of the dark cut sits at a mean luma of 33,
  where a one-level shift is worth several dB. Its PSNR-Y deficits far from the cut (+3.4 dB at
  d ≥ 32, where VMAF and DISTS say the one batch is as good or better) are a rendering
  difference, not the cut. DISTS and VMAF are the steadier measures there.
- **The aligned reference is not a perfect target:** B alone starts on a 1-frame latent and A
  alone ends a batch, and both differ from a longer batch for reasons that have nothing to do
  with the cut (DiT windows, noise by position). The one batch is *better* than the reference on
  A's last frames of the clean cut (−2.2 dB at k = 0) and on the whole of B far from the cut on
  the clean one; the dark cut's whole shot A is worse in one batch (+0.5–1.6 dB, DISTS +0.018,
  with no ghost: the DiT renders the bright shot differently when the dark one is in view).
- **The latent layouts were measured at k = 0 only** (latent 6 pure B); the k = 2 layouts and
  the "merged with the previous shot only" short-shot runs (A + the n frames, without B's
  continuation) were prepared but not run: the GPU was shared with two other jobs all night.
- `--color_correction none`, 1080p from half-size input, 7B fp16 only. With `lab`, low-frequency
  differences between batches shrink ([quality.md](quality.md#colour-correction)).

## Reproduce

```bash
F=scripts/fr_clips.py; M=scripts/fr_metrics.py; J=scripts/cut_metrics.py
# one degraded range around the cut (c = the cut frame), then frame-exact slices of it
python3 $F make SRC --start $((c - 48)) --frames 132 --name cut-long --out clips --degrade d1
python3 $F verify clips/cut-long --source --cv2-python /path/to/seedvr2/.venv/bin/python
python3 $F slice clips/cut-long --first $((27 - k)) --frames 81 --name cut-k$k            # cut at 21 + k
python3 $F slice clips/cut-long --first $((27 - k)) --frames $((21 + k)) --name cut-k$k-A --files gt,d1.lr
python3 $F slice clips/cut-long --first 48 --frames $((60 - k)) --name cut-k$k-B --files gt,d1.lr
# runs: one batch each (FFV1 master only); a latent layout through blend_patch.py
python3 scripts/bench.py run cut-k0-one --wrap scripts/ffv1_out.py --env FFV1_OUT_KEEP=0 -- \
  clips/cut-k0.d1.lr.mkv --output out/ --model_dir /path/to/models --dit_model seedvr2_ema_7b_fp16.safetensors \
  --resolution 1080 --attention_mode flash_attn_2 --batch_size 81 --load_cap 81 --temporal_overlap 0 \
  --color_correction none --seed 42                                  # likewise A (21 + k) and B (60 - k)
python3 scripts/bench.py run cut-k0-lat-hard --wrap scripts/blend_patch.py --wrap scripts/ffv1_out.py \
  --env STITCH_WINDOWS=0-6,6-12,10-16,14-21 --env STITCH_CURVE=cosine --env FFV1_OUT_KEEP=0 -- ...
# the aligned reference, metrics, analysis
python3 $J join out/cut-k0-A.mkv out/cut-k0-B.mkv --out out/cut-k0-aligned.mkv
python3 $M clips/cut-k0.gt.mkv --clip cut-k0 --json-dir m --out one 42 out/cut-k0-one.mkv \
  --out aligned 42 out/cut-k0-aligned.mkv --out lat-hard 42 out/cut-k0-lat-hard.mkv
python3 $J analyze m/cut-k0.one.s42.json --ref m/cut-k0.aligned.s42.json --cut $((21 + k)) --smooth 4 \
  --noise m/*.def.s*.json m/cut-k0.aligned.s43.json --json a/cut-k0-one.json > a/cut-k0-one.md
python3 $J compare --row alone:m/cut-B5.alone.s42.json:0 --row merged:m/cut-k0.one.s42.json:21 --frames 5
python3 $J selftest clips/cut-k0.gt.mkv --cut 21 --out out/cut-k0-aligned.mkv   # ghost coefficient check
python3 $J summary a/                                                           # one row per analysis
```

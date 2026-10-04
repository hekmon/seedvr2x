# Scene cuts inside a processing unit

> Status: **measured** on the reference stack (7B fp16, `flash_attn_2`, 1080p output,
> `--color_correction none`, seed 42), SeedVR2 `4490bd1`, against a ground truth with the
> full-reference protocol of question 1 (degradation d1), for the open questions on shot
> boundaries in [DESIGN.md](../../seedvr2x/DESIGN.md#open-questions). Tools:
> [`scripts/cut_metrics.py`](../scripts/cut_metrics.py) (analysis),
> [`scripts/fr_clips.py`](../scripts/fr_clips.py) `slice` (frame-exact sub-clips),
> [`scripts/fr_metrics.py`](../scripts/fr_metrics.py) (per-frame metrics),
> [`scripts/blend_patch.py`](../scripts/blend_patch.py) `STITCH_WINDOWS` (explicit latent-window
> layouts). Six real hard cuts, all between a darker and a brighter shot: three animated ones
> measured in full, then a colourful anime and two live-action films at k = 0 and 2.

In short (six hard cuts, 81-frame clips with the cut at 21 + k, k = 0…3 on three animated cuts,
k = 0 and 2 on a colourful anime and two live-action films, scored against the ground truth; the
reference is the two shots run separately, what correct shot detection gives):

- **A missed cut costs the first frames of the next shot.** In one batch across the cut, B's
  first frame loses 2.4–9.6 dB of PSNR-Y on the bright and dark cuts, its next three 1.5–4.7 dB,
  and VMAF drops 2–12 points over its first four frames; the clean cut loses far less (0.1–3.9 dB
  on B's first frame only). The three later cuts agree: 2.0–7.4 dB on B's first frame.
  The loss is measurable for up to 17 frames after the cut in PSNR-Y (0–2 on the clean and
  colourful cuts, 4–6 bright, 8–17 on the dark and live-action ones, where all of shot B comes out
  different in one batch), up to 6 in VMAF and 5 in DISTS. Up to 12% of the previous shot's last
  frame shows in B's first frame (ghost coefficient: 3–5% on the bright and dark cuts, 12% on the
  colourful anime, none on the clean and live-action ones).
- **The offset k in the 4-frame latent group moves the damage, it doesn't remove it.** At k = 0
  (cut on a latent boundary) only B suffers; from k = 1 on, A's last frame shares a latent with B
  and loses up to 2.3 dB and 16 VMAF points, with 1–4% of B in it.
- **What carries the previous shot over is the causal VAE, not the DiT.** With one VAE pass and a
  hard DiT boundary exactly at the cut (k = 0), A is bit-identical to A run alone and B is hit
  hardest of all (−7.9 / −16.8 dB on its first frame, 9–11% ghost). Letting the DiT see both
  shots halves that. **Latent stitching across a cut is as good as one batch** when a window
  straddles the cut's latent (mid-window or shared-zone layouts, k = 0 and 2: from 14 dB·frames
  less to 2 more over B's first 8 frames), **and worse with a window boundary at the cut's
  latent:** 1.5–3× the PSNR-Y cost at k = 0; at k = 2 the same PSNR-Y cost, but worse VMAF and
  DISTS, the longest reach (11–16 frames) and, on the bright cut, a 5% ghost of A lasting 15 frames.
- **A false cut costs no fidelity:** splitting a continuous shot in two leaves every metric within
  the noise band next to the split, the frames after it are even 0.3–1.7 dB closer to the ground
  truth; what remains is a low-frequency temporal step at the split on 2 of 4 clips.
- **Very short shots are better alone than merged into the previous shot, from n = 1 frame on:**
  13.5 / 16.4 dB better on a single frame, where padding the merged batch to 4n + 1 frames copies
  the previous shot into the short one's latent (15–20% of it shows), 6.9 / 7.1 dB at 2 frames,
  still 2.1–2.4 dB at 9. From 3 frames on, the merge costs about what a missed cut does. Alone, they are
  within 0.6 dB (bright) or 2 dB (dark) of the same frames inside a long run.
- **A shot's first frame, encoded alone, is the closest to the ground truth, not the worst:** on
  all eleven shots measured it beats the next eight frames by 0.04–5.5 dB of PSNR-Y, and it is
  re-rendered less (less sharp on nine of them, by 3–55%). Prepending 4 mirrored frames (one more
  latent per shot: about 13 s at 1080p, 2–8% more GPU time per hour of animation) mostly gives it
  the next frames' look but costs it 0.6–2.8 dB of that lead on seven of eleven shots
  ([details](#a-shots-first-frame-prepending-mirrored-frames)). Whether its sharpness step is
  visible right after a cut, which changes the picture about 40 to 120 times more, waits for the
  review.
- **For detection, a miss costs far more than a false cut:** 5–31 dB·frames of PSNR-Y over the
  next shot's first 8 frames on four of the six cuts (about 0 on the clean and colourful ones,
  whose next frames come out better in one batch, though B's first frame still loses up to
  7.4 dB), against a small gain for a false cut; folding a short shot into the previous one costs
  10–21 dB·frames over its own frames ([costs per error](#what-it-means-for-shot-detection)).

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
| colour | bright, colourful anime, H.264 1080p | 50.7 (3.1) | 52 → 167 | 0.5 / 10.1 | 103 |
| live fast | live action, fast, with effects, Blu-ray H.264 1080p (untagged: BT.709 assumed), letterboxed | 30.1 (6.3) | 140 → 56 | 8.9 / 15.4 | 100 |
| live slow | live action, slow, Blu-ray H.264 1080p, 24 fps, letterboxed | 36.7 (0.81) | 151 → 45 | 0.8 / 2.9 | 72 |

The last three cuts came later, from three more sources, and were run at k = 0 and 2 only. A
score of 30 or more with long shots on both sides is rare in live action: 1 such cut in the fast
film and 6 in the slow one (about 1 h 40 min each), against 26 in 22 minutes of the anime episode.
All of the films' were between a darker and a brighter shot; the anime's few cuts between shots of
similar brightness were all between still shots.

**Clips.** Per cut, one range of 132 frames, from 48 before the cut to 84 after it, was made once
with `fr_clips.py make` (ground truth: 16-bit RGB; input: half size, Mitchell, x264 CRF 20, fed
as 8-bit RGB), then cut into frame-exact slices with `fr_clips.py slice`. Every slice's framemd5
equals the range's, so every run below reads bit-identical input frames. The long range was
checked against the source by framemd5 (132/132), and cv2 reads every input bit-exactly as the
CLI does.

- **The 81-frame clip at offset k** (k = 0 … 3) puts the cut at clip frame 21 + k. Latent 6
  (frames 21 … 24) holds the cut: at k = 0 it is pure B; at k = 1, 2, 3 it holds k frames of A.
- **A** = its 21 + k frames before the cut; **B** = its 60 − k frames from the cut.
- **Short shots:** B's first n frames, n = 1, 2, 3, 5, 9, run alone. **Merged with A:** A's 21
  frames and the n frames as one batch of 21 + n, the cut on a latent boundary. **In one batch:**
  the same frames inside the 81-frame batch at k = 0 (after A, with the rest of B after them).
- **False cuts:** the four 45-frame single-shot clips of question 1, split 21 + 24.

**Runs** (one batch each, `--batch_size` = clip length, `--temporal_overlap 0`; lossless masters
through [`ffv1_out.py`](../scripts/ffv1_out.py)):

| Run | What it is |
|---|---|
| one | the 81 frames in one batch: the cut inside one VAE pass and one DiT batch |
| aligned (the reference) | A alone and B alone, their outputs joined losslessly: what correct shot detection gives |
| latent (iii) hard | one VAE pass, DiT windows `0-6,6-12,10-16,14-21` (latents): a hard DiT boundary at latent 6, 2 shared latents elsewhere: the VAE carries A into B, the DiT doesn't (at k = 2, A's last 2 frames share latent 6 with B and fall in B's window) |
| latent (ii) shared | `0-7,5-11,9-15,13-19,17-21`: latents 5 and 6 shared by two windows, cross-faded (cosine, 0.75 / 0.25) |
| latent (i) mid | `0-5,3-9,7-13,11-17,15-21`: window 3–9 centred on the cut, latents 5 and 6 inside it |
| short alone | B's first n frames as their own batch (in one batch: the same frames inside "one" at k = 0) |
| merged with A | A and B's first n frames as one batch: the short shot folded into the previous one, as a minimum shot length does, the next shot starting a unit of its own. A batch that isn't 4n + 1 frames is padded with its last frames mirrored ([cli-flags.md](cli-flags.md)): at n = 1 and 2 the padding reaches back across the cut, and the short shot's latent holds 3 and 1 of A's frames |
| split | a question-1 clip as two batches, 21 + 24 frames, joined |
| prepend P | B alone (k = 0, 60 frames) with `--prepend_frames` 4 and 8, and the five question-1 clips (45 frames) with 4; `--batch_size` = frames + P, so the shot stays one batch; the P mirrored frames dropped from the master |

The latent runs (bright and dark cuts, k = 0 and 2) stitch like [stitching.md](stitching.md#latent-space-stitching)'s
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

The 81 frames in one batch against the aligned reference (A alone + B alone); the colourful anime
and the two live-action cuts at k = 0 and 2 only. Deficits by distance d to the cut, positive =
the batch across the cut is worse; reach = frames measurably worse next to the cut (4-frame
averages beyond the noise band), before / after it; α8 = the ghost coefficient's excess (share of
the other shot's frame next to the cut).

| k | Cut | PSNR-Y deficit, dB: d −1 / 0 / 1…3 / 4…7 / 8…15 | DISTS d 0…3 | VMAF d 0…3 | T-err lf at the cut | Reach before / after: PSNR-Y, DISTS, VMAF, T-err lf | α8 excess d −1 / 0 / 1…7 / 8…15 |
|---|---|---|---|---|---|---|---|
| 0 | bright | +0.44 / +4.14 / +3.42 / +1.11 / −0.37 | +0.020 | +11.6 | +4.82 | 0/6, 0/2, 0/4, 0/25 | +0.003 / +0.034 / −0.011 / +0.029 |
| 0 | dark | +0.53 / +9.64 / +3.67 / +2.59 / +1.64 | −0.001 | +4.7 | −0.78 | 0/13, 21/0, 0/2, 0/8 | +0.005 / +0.045 / +0.017 / +0.012 |
| 0 | clean | −2.21 / +0.10 / +0.53 / −0.96 / −0.93 | +0.009 | +3.7 | −0.43 | 0/0, 0/0, 0/2, 0/13 | +0.016 / −0.004 / −0.004 / +0.020 |
| 0 | colour | +0.37 / +7.41 / −0.39 / −1.80 / −1.25 | +0.042 | +12.4 | +14.32 | 0/2, 0/5, 0/2, 0/5 | +0.008 / **+0.116** / +0.022 / +0.019 |
| 0 | live fast | −0.85 / +2.79 / +2.23 / +2.79 / +0.73 | +0.007 | −0.6 | +3.50 | 0/11, 0/1, 0/6, 0/8 | +0.003 / −0.041 / +0.030 / +0.015 |
| 0 | live slow | −1.69 / +6.18 / +3.43 / +2.77 / +2.01 | +0.008 | +7.2 | +0.73 | 0/17, 0/0, 0/3, 0/6 | +0.024 / −0.032 / +0.021 / +0.021 |
| 1 | bright | +2.08 / +3.02 / +2.35 / +0.45 / −0.34 | +0.004 | +7.3 | +0.72 | 2/5, 0/0, 2/3, 3/24 | +0.010 / −0.012 / −0.005 / +0.024 |
| 1 | dark | +0.46 / +4.01 / +1.50 / +1.84 / +1.08 | −0.006 | +3.4 | +0.18 | 0/12, 20/0, 0/2, 3/18 | +0.034 / −0.009 / +0.007 / +0.005 |
| 1 | clean | −2.01 / +1.54 / +0.09 / −0.66 / −0.63 | +0.008 | +2.2 | +1.70 | 0/1, 0/0, 0/1, 3/5 | −0.002 / −0.012 / −0.006 / +0.008 |
| 2 | bright | +1.80 / +4.85 / +2.97 / −0.30 / −0.62 | +0.004 | +6.4 | +2.68 | 0/4, 1/0, 0/3, 4/11 | +0.013 / −0.031 / −0.005 / +0.030 |
| 2 | dark | −1.14 / +4.49 / +4.72 / +1.68 / +0.57 | −0.013 | +1.9 | +0.37 | 0/10, 23/0, 0/1, 4/16 | +0.019 / −0.013 / +0.015 / +0.007 |
| 2 | clean | −2.61 / +3.04 / +0.01 / −0.82 / −0.96 | +0.002 | +2.5 | +4.96 | 0/1, 0/0, 0/1, 4/11 | +0.001 / −0.048 / −0.012 / +0.014 |
| 2 | colour | +1.59 / +2.02 / −1.59 / −1.87 / −1.15 | +0.016 | −3.2 | +1.12 | 1/0, 0/2, 0/0, 4/3 | +0.009 / +0.016 / +0.022 / +0.016 |
| 2 | live fast | −1.81 / +2.58 / +0.13 / +0.56 / +0.61 | +0.017 | +4.1 | +1.97 | 0/10, 0/2, 0/4, 6/12 | +0.008 / −0.033 / +0.005 / +0.004 |
| 2 | live slow | −2.80 / +4.11 / +1.26 / +0.69 / +1.00 | −0.001 | +4.2 | −1.55 | 0/12, 0/0, 0/2, 4/17 | +0.044 / −0.031 / +0.011 / +0.010 |
| 3 | bright | +0.43 / +2.35 / +1.93 / −0.37 / −0.33 | +0.006 | +2.7 | +2.40 | 0/4, 1/0, 3/1, 5/12 | −0.015 / −0.005 / +0.001 / +0.021 |
| 3 | dark | +2.32 / +8.84 / +3.46 / +1.01 / −0.46 | −0.005 | +2.8 | +6.89 | 24/8, 24/0, 1/1, 4/11 | −0.011 / −0.039 / +0.011 / +0.006 |
| 3 | clean | −0.24 / +3.89 / +1.05 / −0.77 / −0.84 | +0.013 | +4.3 | +5.87 | 0/2, 0/1, 0/2, 4/10 | −0.009 / −0.053 / −0.013 / +0.012 |

- **The cost lands on B's first frames.** B's first frame loses 2.4–9.6 dB of PSNR-Y on the
  bright and dark cuts at every offset, its next three frames 1.5–4.7 dB, and VMAF drops 2–12
  points over B's first four frames. DISTS hardly moves (within ±0.02). The clean cut suffers
  least: 0.1–3.9 dB on B's first frame, within the band from the next one on (k = 0, 1, 2). The
  three later cuts agree: B's first frame loses 2.0–7.4 dB (the colourful anime 7.4 at k = 0,
  with DISTS +0.042 over B's first 4 frames, the only DISTS loss beyond ±0.02; the live-action
  cuts 2.6–6.2). The colourful anime's next frames come out better in one batch than B alone, as
  on the clean cut; the live-action ones worse, by 0.1–3.4 dB.
- **How far it reaches:** PSNR-Y is measurably worse for 4–6 frames after the cut on the bright
  cut, 8–13 on the dark one, 0–2 on the clean and the colourful anime ones, 10–17 on the
  live-action ones; VMAF for up to 6 frames, DISTS for up to 5 (the colourful anime, k = 0), at
  most 2 elsewhere. On the dark and live-action cuts the far deficits are a rendering difference
  of the whole shot B in one batch (PSNR-Y still +0.5 to +4.3 dB 16 or more frames after the cut,
  with DISTS within ±0.011 there) rather than the cut's reach. The low-frequency
  temporal error (T-err lf), whose band is narrow, stays different for 3–25 frames.
- **The offset moves the damage between the shots.** At k = 0 the cut sits on a latent boundary
  and A's last frames are no worse than the rest of A (the dark cut's A is worse throughout, see
  below). From k = 1 on, A's last frames share latent 6 with B: the last one loses up to 2.3 dB
  of PSNR-Y and up to 16 VMAF points (bright, k = 1), carries 1–4% of B (α8 at d = −1, k = 1 and
  2), and the low-frequency temporal error is measurably worse for A's last 3–5 frames, where it
  is clean at k = 0. B's first frame carries 3–5% of A at k = 0 on the bright and dark cuts and
  12% on the colourful anime (none on the clean and live-action ones), and at most 1.6% at k ≥ 1.
- **No offset is cheap.** Summed over B's first 8 frames, the PSNR-Y cost falls with k on the
  bright cut (18.8, 11.9, 12.6, 6.7 dB·frames for k = 0…3) but not on the dark one (31.0, 15.9,
  25.4, 23.3); meanwhile A's last 4 frames go from −10 VMAF points·frames at k = 0 to +27 at
  k = 3 (bright). The clean cut stays within ±4 dB·frames at every k. Of the later cuts, the
  live-action ones fall from 20.7 and 27.5 dB·frames at k = 0 to 5.2 and 10.6 at k = 2, and the
  colourful anime's is −1.0 and −10.2 (its later frames come out better in one batch).
- On the dark cut, shot A is worse in one batch back to its first frame at every k (DISTS reach
  20–24 frames before the cut; +0.5 to +1.6 dB of PSNR-Y and +0.02 DISTS at k = 0), with no
  ghost: the DiT renders the bright shot differently with the dark one in view (the VAE, causal,
  can't carry B back into A). On the live-action cuts it goes the other way: shot A comes out
  0.6–1.7 dB closer to the ground truth in one batch.

### What carries one shot into the other: the VAE or the DiT

The three latent layouts against the aligned reference, at k = 0 (latent 6 is pure B) and k = 2
(latent 6 holds A's last 2 frames and B's first 2); one batch for comparison; "one − (iii)"
compares the one batch with layout (iii) directly. Deficits by distance d to the cut, positive =
worse; reach in frames after the cut (on 4-frame averages, beyond the noise band); α8 = the ghost
coefficient's excess.

| k | Cut | Run | PSNR-Y deficit, dB: d 0 / 1…3 / 4…7 / 8…15 | DISTS d 0…3 | VMAF d 0…3 | Reach after: PSNR-Y / DISTS / VMAF / T-err lf | A's last 4 frames: PSNR-Y / DISTS | α8 excess d −1 / 0 / 1…7 / 8…15 |
|---|---|---|---|---|---|---|---|---|
| 0 | bright | one batch | +4.14 / +3.42 / +1.11 / −0.37 | +0.020 | +11.6 | 6 / 2 / 4 / 25 | +0.13 / −0.056 | +0.003 / +0.034 / −0.011 / +0.029 |
| 0 | bright | (i) mid | +4.68 / +2.69 / +0.60 / −0.76 | +0.023 | +13.8 | 5 / 2 / 5 / 13 | +1.57 / −0.022 | +0.012 / +0.041 / −0.000 / +0.024 |
| 0 | bright | (ii) shared | +4.61 / +2.54 / +0.61 / −0.60 | +0.020 | +13.4 | 5 / 2 / 6 / 25 | +0.79 / −0.050 | +0.006 / +0.038 / −0.002 / +0.028 |
| 0 | bright | (iii) hard | **+7.85 / +3.34 / +2.43 / +1.71** | **+0.061** | **+19.3** | **14 / 7 / 27 / 33** | 0 / 0 (bit-identical) | 0 / **+0.086 / +0.055 / +0.057** |
| 0 | bright | one − (iii) | −3.71 / +0.08 / −1.33 / −2.08 | −0.041 | −7.7 | 0 / 0 / 0 / 13 | +0.13 / −0.056 | +0.003 / −0.052 / −0.066 / −0.028 |
| 0 | dark | one batch | +9.64 / +3.67 / +2.59 / +1.64 | −0.001 | +4.7 | 13 / 0 / 2 / 8 | +0.59 / +0.018 | +0.005 / +0.045 / +0.017 / +0.012 |
| 0 | dark | (i) mid | +9.62 / +3.51 / +2.62 / +1.57 | +0.004 | +6.5 | 19 / 0 / 2 / 35 | −0.24 / +0.024 | +0.008 / +0.044 / +0.016 / −0.002 |
| 0 | dark | (ii) shared | +8.14 / +1.46 / +1.03 / +1.31 | −0.007 | +4.9 | 17 / 0 / 2 / 35 | −0.17 / −0.008 | +0.007 / +0.034 / +0.006 / −0.001 |
| 0 | dark | (iii) hard | **+16.81 / +12.71 / +7.20 / +2.81** | **+0.090** | **+33.3** | 18 / 4 / 6 / 26 | 0 / 0 (bit-identical) | 0 / **+0.110 / +0.047** / +0.008 |
| 0 | dark | one − (iii) | −7.17 / −9.04 / −4.62 / −1.16 | −0.091 | −28.6 | 0 / 0 / 0 / 0 | +0.59 / +0.018 | +0.005 / −0.065 / −0.030 / +0.004 |
| 2 | bright | one batch | +4.85 / +2.97 / −0.30 / −0.62 | +0.004 | +6.4 | 4 / 0 / 3 / 11 | +0.31 / −0.005 | +0.013 / −0.031 / −0.005 / +0.030 |
| 2 | bright | (i) mid | +3.66 / +1.24 / −0.86 / −0.77 | +0.009 | +7.3 | 2 / 0 / 3 / 11 | +1.49 / +0.025 | +0.008 / −0.017 / +0.017 / +0.030 |
| 2 | bright | (ii) shared | +3.57 / +1.25 / −0.89 / −0.62 | +0.004 | +6.9 | 2 / 0 / 3 / 11 | +0.33 / −0.014 | +0.002 / −0.018 / +0.013 / +0.035 |
| 2 | bright | (iii) hard | +1.51 / +1.81 / +1.40 / +0.35 | **+0.041** | +6.6 | **11 / 4 / 8 / 13** | +0.92 / +0.046 | +0.014 / **+0.035 / +0.052 / +0.048** |
| 2 | bright | one − (iii) | +3.34 / +1.16 / −1.69 / −0.97 | −0.037 | −0.3 | 2 / 0 / 0 / 0 | −0.61 / −0.051 | −0.001 / −0.066 / −0.058 / −0.018 |
| 2 | dark | one batch | +4.49 / +4.72 / +1.68 / +0.57 | −0.013 | +1.9 | 10 / 0 / 1 / 16 | +0.29 / +0.024 | +0.019 / −0.013 / +0.015 / +0.007 |
| 2 | dark | (i) mid | +4.35 / +4.81 / +2.11 / +1.32 | −0.015 | +1.8 | 7 / 0 / 1 / 24 | −0.26 / +0.023 | +0.012 / −0.008 / +0.015 / −0.004 |
| 2 | dark | (ii) shared | +6.05 / +2.69 / +1.10 / +0.72 | −0.019 | +3.0 | 6 / 0 / 1 / 24 | −0.68 / −0.005 | +0.014 / −0.023 / +0.005 / −0.003 |
| 2 | dark | (iii) hard | +4.83 / +3.21 / +2.18 / +1.58 | **+0.032** | **+15.3** | **16 / 3 / 4 / 24** | −0.31 / +0.005 | **+0.040** / −0.015 / +0.005 / −0.003 |
| 2 | dark | one − (iii) | −0.34 / +1.51 / −0.50 / −1.00 | −0.045 | −13.4 | 3 / 0 / 0 / 6 | +0.59 / +0.019 | −0.021 / +0.003 / +0.009 / +0.009 |

- **The VAE alone carries the previous shot into the next.** With a hard DiT boundary exactly at the
  cut and one VAE pass (iii at k = 0), A is untouched: its 21 frames are bit-identical to A run
  alone (the encoder and the decoder are causal, and the first window gets the same noise as A's
  own batch). B is hit hardest of all the runs: −7.9 and −16.8 dB on its first frame, a ghost of
  9–11% of A's last frame (still 5–6% 8 to 15 frames later on the bright cut), and VMAF −19 and
  −33 on its first 4 frames.
- **Letting the DiT see across the cut makes it better, not worse.** At k = 0 the one batch, and
  the two layouts whose DiT window straddles the cut (i, ii), lose about half as much as (iii) on
  B's first frames (one − (iii): −3.7 and −7.2 dB on the first frame, −0.04 and −0.09 DISTS on the
  first 4). Part of (iii)'s loss is likely its B window starting on a 4-frame latent encoded with
  A in the causal cache, without the DiT context a normal batch has; [stitching.md](stitching.md#latent-space-stitching)
  found the same first-latent outlier on continuous content.
- **At k = 2 the hard boundary moves the first-frame loss, not the rest.** Latent 6, the first of
  B's window, now starts with A's last 2 frames. B's first frame loses less than in one batch on
  the bright cut (1.5 against 4.9 dB) and as much on the dark one (4.8 against 4.5 dB), and A's last
  frame more on the bright cut (2.9 against 1.8 dB, with DISTS +0.10 and VMAF +9.7 on that frame;
  on the dark one it takes 4% of B). But B stays worse for longer than in any other run: DISTS
  +0.03–0.04 on its first 4 frames where one batch and (i), (ii) are within ±0.02, VMAF +15 on the
  dark cut's first 4 (one batch +1.9), PSNR-Y measurably worse for 11 and 16 frames after the cut,
  DISTS for 3–4, and on the bright cut 3.5–5% of A stays in B for 15 frames, as at k = 0.
- **Latent stitching across a cut is as good as one batch**, provided the cut's latent is not at
  a window boundary. On B's first frame, (i) and (ii) are within 0.6 dB of the one batch at k = 0
  (dark (ii): 1.5 dB better) and 1.2–1.3 dB better on the bright cut at k = 2; on the dark cut at
  k = 2, (i) is within 0.2 dB and (ii) 1.6 dB worse on the first frame but 2.0 dB better on the
  next three. Over B's first 8 frames they cost from 14 dB·frames less to 2 more than one batch
  ([costs](#what-it-means-for-shot-detection)). Both put A's last latents in a window with B: A's
  last 4 frames lose 1.5–1.6 dB (i) and 0.3–0.8 dB (ii) on the bright cut at both offsets, nothing
  on the dark one.

### The ghost: how much of the other shot shows

The ghost coefficient α (8×8 block means of luma, excess over the aligned reference) stays small
in one batch on five of the six cuts, at most +0.045 (4.5% of the other shot's frame next to the
cut); the colourful anime's B takes 12% at k = 0. The cut's latent offset decides where it shows:

- **k = 0:** in B's first frame, 3.4% (bright), 4.5% (dark) and 11.6% (colourful anime, whose
  shot A is a nearly still, dark shot); none on the clean and live-action cuts.
- **k ≥ 1:** at most 1.6% in B's first frame (excess −0.053 to +0.016); A's last frame takes
  0.8–4.4% of B at k = 1 and 2, none at k = 3 (one B frame in the shared latent).
- After B's first frame the excess stays between −0.013 and +0.030 on every cut and offset,
  without a trend.

With a hard DiT boundary at the cut (layout iii, k = 0), the VAE alone puts 9–11% of A into B's
first frame, still 5–6% 8 to 15 frames later on the bright cut. With the DiT seeing both shots
(one batch, layouts i and ii) the ghost is about half as large: attention across the cut does not
add to what the causal VAE carries over. At k = 2, layout (iii) leaves 3.5% of A in B's first
frame on the bright cut and 5% 1 to 15 frames later, as at k = 0; on the dark cut none in B, but
4% of B in A's last frame. Layouts (i) and (ii) at k = 2 show none in B's first frame, like the
one batch.

### Very short shots: alone or merged

B's first n frames run as a shot of their own ("alone"), against the same frames in three other
runs: appended to shot A as one batch ("merged with A": what a minimum shot length that folds a
short shot into the previous one does, the next shot starting a unit of its own), inside the one
batch across the cut at k = 0 ("in one batch": B's continuation in the same batch), and inside B
run alone ("long", the ideal when the shot goes on). Means over the n frames against the ground
truth, the better of alone and merged with A in bold:

| Cut | n | PSNR-Y, dB: alone / merged with A / in one batch / long | DISTS: alone / merged with A / in one batch / long | VMAF: alone / merged with A / in one batch / long |
|---|---|---|---|---|
| bright | 1 | **30.94** / 17.45 / 27.35 / 31.49 | **0.027** / 0.211 / 0.100 / 0.038 | **82.7** / 27.0 / 58.7 / 82.9 |
| bright | 2 | **31.44** / 24.54 / 27.52 / 31.32 | **0.023** / 0.095 / 0.077 / 0.037 | **83.4** / 57.3 / 64.9 / 82.8 |
| bright | 3 | **31.12** / 27.18 / 27.72 / 31.39 | **0.022** / 0.069 / 0.064 / 0.037 | **82.4** / 65.7 / 69.0 / 82.7 |
| bright | 5 | **31.13** / 27.87 / 27.93 / 31.25 | **0.022** / 0.056 / 0.053 / 0.038 | **83.2** / 72.0 / 72.7 / 82.5 |
| bright | 9 | **30.94** / 28.56 / 29.04 / 31.12 | **0.022** / 0.046 / 0.044 / 0.037 | **81.2** / 75.8 / 75.8 / 82.3 |
| dark | 1 | **37.84** / 21.42 / 30.07 / 39.71 | **0.181** / 0.331 / 0.213 / 0.178 | **70.0** / 33.3 / 57.4 / 67.7 |
| dark | 2 | **38.75** / 31.66 / 32.19 / 39.70 | **0.160** / 0.220 / 0.195 / 0.177 | **71.8** / 51.8 / 57.4 / 67.4 |
| dark | 3 | **38.88** / 35.58 / 33.52 / 39.68 | **0.156** / 0.185 / 0.182 / 0.177 | **70.7** / 61.9 / 59.4 / 66.5 |
| dark | 5 | **38.14** / 36.06 / 34.91 / 39.70 | **0.145** / 0.170 / 0.168 / 0.176 | **72.1** / 63.0 / 62.8 / 66.0 |
| dark | 9 | **38.72** / 36.60 / 35.94 / 39.67 | 0.157 / **0.153** / 0.158 / 0.179 | **70.8** / 65.6 / 66.6 / 65.1 |

- **Alone beats merged with A at every n, on both cuts:** by 13.5 and 16.4 dB of PSNR-Y on a
  single frame, 6.9 and 7.1 dB at 2 frames, 2.1–3.9 dB from 3 frames on, and on DISTS and VMAF
  too (except DISTS on the dark cut at n = 9: 0.004 better merged). Alone also beats the frames
  in one batch across the cut, on every metric (3.6 and 7.8 dB on a single frame, 1.9 and 2.8 dB
  at 9).
- **Below 3 frames the merge is worse than a missed cut,** because of the padding: the merged
  batch of 22 or 23 frames is padded to 25 with its last frames mirrored, which reach back across
  the cut, so the short shot's latent holds 3 (n = 1) or 1 (n = 2) of A's frames. Its first frame
  then carries 20% / 15% (n = 1) and 12% / 7% (n = 2) of A's last frame (ghost excess against A
  alone + the n frames alone) and loses 9–16 dB. From the code, a 1- or 2-frame shot at the end
  of a batch shares a latent with frames of the previous shot at any offset in the 4-frame grid,
  through the grid itself or the padding. From n = 3 on (no padding frame from A here), the merged
  frames fare like the same frames in one batch across the cut: within 0.6 dB on the bright cut,
  0.7–2.1 dB better on the dark one, with 3% / 5–6% of A in the first frame.
- **Merging costs A nothing consistent:** A's last 4 frames stay within ±0.75 dB of PSNR-Y and
  +0.012 DISTS of A run alone (inside the noise band; the batch is rendered with other noise and
  windows), with DISTS even better on the bright cut (by up to 0.045), and VMAF 4–7 points worse
  throughout A at n = 3 on the bright cut only.
- **A short shot alone loses little against the ideal:** at most 0.6 dB of PSNR-Y on the bright
  cut, 0.8–1.9 dB on the dark one, and its DISTS is *better* than the same frames inside the long
  run in 9 of 10 cases (VMAF within 1.1 points on the bright cut, 2–6 points better on the dark).
  A 1- to 9-frame batch is one to three latents, rendered with nothing else in view.
- VMAF understates the merge at n = 1: the merged frame ends its clip right after the cut, where
  VMAF scores the same output 12–13 points higher than as a clip of its own (B's frame run alone,
  scored both ways).

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

### A shot's first frame: prepending mirrored frames

The causal VAE encodes a batch's first frame alone (latent 0), and it comes out less restored,
closer to its input ([quality.md](quality.md#--prepend_frames)). In seedvr2x that is every shot's
first frame, right after a cut. numz's `--prepend_frames P` puts frames P … 1, mirrored, before
frame 0 inside the batch and drops them after decoding: at P = 4, frame 0 shares a latent with
frames 3 … 1 and the rest of the 4-frame grid is unchanged (one more latent; P = 8: two). On one
GPU the CLI keeps them ([bug 05](../bugs/05-prepend-frames-not-removed.md));
[`ffv1_out.py`](../scripts/ffv1_out.py) drops them from the master by default. Every master
holds the shot's frame count, and each of its first frames matches the ground-truth frame of the
same index best, and the same frame of the run without prepending. Runs: the three first shots B
(60 frames from the cut, alone) with P = 4 and 8, the three later cuts' shots B with P = 4, the
five question-1 clips (45 frames) with P = 4, against the same runs without prepending (the first
shots B at seeds 42 and 43; question 1's default runs at three seeds, the noise of the deficit
below: q95 / max over 18 seed pairs).

Frame 0's deficit against the mean of frames 1…8, positive = frame 0 worse, without prepending →
with P = 4 (→ P = 8). Sharpness = the Laplacian variance of luma relative to the ground truth's
(0.3–6.9× on these shots), frame 0 against frames 1…8, and the jump from frame 0 to frame 1:

| Shot | PSNR-Y, dB | DISTS | VMAF | Sharpness of frame 0 vs 1…8 | Jump 0 → 1 |
|---|---|---|---|---|---|
| bright cut, B | −0.41 → **+0.18** (→ **+0.45**) | +0.001 → +0.001 (→ +0.002) | −1.1 → **+1.1** (→ **+1.1**) | −7% → −5% (→ −5%) | +4% → +6% (→ +6%) |
| dark cut, B | −0.04 → −0.30 (→ +0.04) | −0.002 → **−0.013** (→ −0.005) | −3.3 → −4.0 (→ −2.3) | +14% → **+5%** (→ **+6%**) | −9% → −7% (→ −5%) |
| clean cut, B | −1.03 → **−0.12** (→ **+0.17**) | −0.005 → +0.000 (→ **+0.003**) | −5.7 → **−1.3** (→ **+0.4**) | −14% → **−2%** (→ **+1%**) | +16% → **+7%** (→ **+6%**) |
| colour cut, B | −1.15 → −0.85 | +0.006 → +0.010 | −15.6 → **−9.9** | −55% → −53% | +9% → **−1%** |
| live fast cut, B | −1.78 → −1.71 | −0.012 → **+0.011** | −9.3 → −10.8 | −3% → **−14%** | −6% → **+2%** |
| live slow cut, B | −1.60 → **−0.89** | −0.017 → −0.011 | −9.6 → **−6.6** | −15% → **+0%** | +3% → +1% |
| anime-clean | −1.81 → **−0.48** | −0.001 → +0.002 | −13.3 → **−4.2** | −10% → **+7%** | +2% → 0% |
| anime-grain | −4.56 → **−1.76** | −0.018 → **−0.008** | −18.9 → **−9.1** | −47% → **−15%** | +39% → **+5%** |
| anime-dark | −5.53 → **−3.44** | −0.031 → **−0.018** | −27.7 → **−14.5** | +31% → +29% | +8% → +8% |
| cartoon-bright | −3.98 → **−1.66** | −0.076 → **−0.037** | −25.5 → **−14.4** | −29% → **−14%** | +24% → **+14%** |
| anime-sky | −0.58 → −0.55 | −0.047 → −0.051 | −3.5 → −3.3 | −16% → −12% | −49% → −51% |
| input (bicubic), 11 shots | −0.84 … +1.51 | −0.012 … +0.027 | −3.4 … +0.1 | | |
| seed noise of the deficit | 0.22 / 0.30 | 0.005 / 0.007 | 1.4 / 1.8 | 4% / 5% | 5% / 5% |

(Bold: changed by more than the seed noise's max.)

- **Without prepending, a shot's first frame is the one closest to the ground truth,** not the
  worst: on all eleven shots it beats the next eight frames by 0.04–5.5 dB of PSNR-Y and 1–28
  VMAF points, in DISTS by up to 0.076 (within ±0.006 on four shots). Only the dark cut's frame 0
  is worse in LPIPS (+0.012). The degraded input accounts for little of it: its own first frame,
  an x264 I-frame, is between 0.8 dB better and 1.5 dB worse than its next ones. The first
  latent is re-rendered less: on nine shots frame 0 is 3–55% less sharp than the next frames
  (relative to the ground truth; on the colourful anime the input's first frame already is, by
  43%), with a jump of up to +39% to frame 1 (anime-grain); on the two darkest shots it is sharper.
- **Prepending 4 frames mostly makes frame 0 an ordinary frame.** Its sharpness joins the next
  frames' (anime-grain −47% → −15%, cartoon-bright −29% → −14%, the clean cut −14% → −2%, the
  slow live-action cut −15% → 0%) and the jump to frame 1 shrinks where it was large (+39% → +5%,
  +24% → +14%, +16% → +7%); not on the colourful anime (−55% → −53%), and on the fast
  live-action cut frame 0 gets softer (−3% → −14%). But frame 0 also loses much of its lead in
  fidelity: 0.6–2.8 dB of PSNR-Y, 2–13 VMAF points and up to 0.039 DISTS relative to frames 1…8
  on seven of the eleven shots, beyond the seed noise (0.3 dB on the colourful anime, at its
  edge). The dark cut's frame 0, the one worse in LPIPS, gains (LPIPS 0.031, DISTS 0.011; the
  slow live-action cut LPIPS 0.011, anime-sky 0.005). Eight frames do no better than four.
- **The step at 0 → 1 is a change of look, not a larger error.** The temporal error of
  transition 0 → 1 against the ground truth (T-err) is above the next transitions' on four shots
  without prepending (by 1.2–1.9: three clips and the fast live-action cut; the input's own,
  I-frame then P-frames, by 0.1–0.3), and prepending leaves it there (anime-dark +1.4 → +1.8,
  cartoon-bright +1.9 → +2.1, live fast +1.7 → +2.2; anime-sky +1.2 → +0.7): it is not the lone
  first latent's doing.
- **The rest of the shot** moves by less than 0.4 dB of PSNR-Y on average over frames 1…end:
  within 0.1 dB of the spread between seeds on the three first shots B, within 0.11 dB on the
  three later ones, 0.08–0.35 dB better on the five clips, whose batch is 4 frames longer.
- **Masking:** on the six cuts, prepending changes frame 0 by 1.1–4.0 luma levels on average
  (mean |ΔY|), where the cut itself changes the picture by 97–152 levels; on the first three,
  frame 0's error against the ground truth is about that of frames 1…8 either way (1.6–5.9
  levels). Whether the sharpness jump right after a cut is visible is for the review: clips of
  each cut (A's last 12 frames, B's first 36) without and with prepending, side by side, and 1:1
  crops of frames 0 and 1 are prepared for it.
- **Cost:** one latent per shot, 4 more computed frames: about 13 s per shot at 1080p (3.0–3.3 s
  per computed frame on this GPU in this session, 3.9–5.1 s in earlier ones: 16–20 s; pairs of
  runs differ by −0.5 to +28 s for one latent, within the ±7% run-to-run spread). Per hour of
  video that is 4 frames over the mean shot length, at the shot counts of scdet at threshold 10
  ([question 3](../PROGRESS.md#3-scene-detection)): 2.2–3.8% more GPU time on the bright cartoon
  (469–830 shots per hour), 6.6% on the clean digital anime film (1430), 5.9% on the dark action
  anime episode (1278), 7.7% on the 720p anime episode (1657), 4.2% on the grainy cel film (912).

## What it means for shot detection

The cost of each kind of detection error against correct boundaries, as sums of per-frame
deficits over a fixed window next to the boundary (positive = worse): B = the 8 frames after it,
A = the 4 before it; for a short shot, its own n frames (and A's last 4 against A run alone).
PSNR-Y in dB·frames, VMAF in points·frames, DISTS in DISTS·frames:

| Error | Case | PSNR-Y: B / A | VMAF: B / A | DISTS: B |
|---|---|---|---|---|
| Missed cut, one batch | bright, k = 0 / 1 / 2 / 3 | 18.8 / 11.9 / 12.6 / 6.7 ; A 0.5 / 2.6 / 1.2 / −2.0 | 56.0 / 35.3 / 26.6 / 7.7 ; A −10.3 / 11.5 / 1.5 / 26.5 | 0.075 / 0.004 / −0.009 / 0.017 |
| | dark, k = 0 / 1 / 2 / 3 | 31.0 / 15.9 / 25.4 / 23.3 ; A 2.4 / 2.9 / 1.1 / 4.2 | −7.5 / −12.3 / −18.0 / −19.7 ; A −8.9 / −6.9 / −9.4 / 4.3 | −0.153 / −0.171 / −0.175 / −0.173 |
| | clean, k = 0 / 1 / 2 / 3 | −2.1 / −0.8 / −0.2 / 3.9 ; A −8.9 / −8.5 / −9.6 / −6.5 | −3.1 / −8.9 / −13.3 / −14.3 ; A −26.8 / −24.7 / −27.0 / −13.9 | 0.016 / 0.021 / −0.013 / 0.036 |
| | colour, k = 0 / 2 | −1.0 / −10.2 ; A 1.3 / 2.4 | 15.3 / −48.4 ; A −6.5 / −6.0 | 0.199 / 0.059 |
| | live fast, k = 0 / 2 | 20.7 / 5.2 ; A −3.0 / −1.1 | 16.1 / 17.0 ; A −28.1 / −13.9 | 0.010 / 0.054 |
| | live slow, k = 0 / 2 | 27.5 / 10.6 ; A −6.3 / −5.9 | 28.5 / 1.3 ; A −11.7 / −7.4 | −0.027 / −0.076 |
| Missed cut, latent windows (k = 0) | bright: (i) / (ii) / (iii) | 15.2 / 14.7 / 27.6 ; A 6.3 / 3.2 / 0 | 67.2 / 67.2 / 132.3 ; A 5.1 / 11.0 / 0 | 0.078 / 0.063 / 0.312 |
| | dark: (i) / (ii) / (iii) | 30.7 / 16.6 / 83.7 ; A −0.9 / −0.7 / 0 | −4.5 / −7.1 / 149.0 ; A 0.3 / −3.2 / 0 | −0.208 / −0.245 / 0.281 |
| Missed cut, latent windows (k = 2) | bright: (i) / (ii) / (iii) | 3.9 / 3.7 / 12.5 ; A 6.0 / 1.3 / 3.7 | 31.4 / 32.3 / 43.2 ; A 13.6 / 9.2 / 19.7 | 0.009 / −0.016 / 0.188 |
| | dark: (i) / (ii) / (iii) | 27.2 / 18.5 / 23.2 ; A −1.1 / −2.7 / −1.2 | −23.7 / −15.7 / 57.7 ; A −5.1 / −14.2 / 8.5 | −0.258 / −0.257 / 0.052 |
| False cut | clean / grainy / dark / bright | −3.9 / −8.8 / −6.6 / −0.5 ; A −0.2 / 0.6 / −0.5 / 0.5 | −20.9 / −32.8 / −9.6 / 4.5 | 0.086 / −0.125 / −0.022 / −0.016 |
| Short shot merged with A instead of alone | bright, n = 1 / 2 / 3 / 5 / 9 | 13.5 / 13.8 / 11.8 / 16.3 / 21.4 ; A −2.6 / −1.3 / 3.0 / 1.6 / 1.1 | 55.7 / 52.2 / 50.2 / 56.0 / 48.2 ; A −6.7 / −2.3 / 21.1 / 7.1 / −7.3 | 0.184 / 0.145 / 0.141 / 0.170 / 0.217 |
| | dark, n = 1 / 2 / 3 / 5 / 9 | 16.4 / 14.2 / 9.9 / 10.4 / 19.1 ; A −1.9 / −0.8 / −0.4 / −0.6 / 2.2 | 36.7 / 40.0 / 26.5 / 45.5 / 46.2 ; A −2.2 / −0.1 / 0.0 / −1.8 / −6.5 | 0.150 / 0.120 / 0.088 / 0.125 / −0.039 |
| Short shot in one batch across its cut instead of alone | bright, n = 1 / 2 / 3 / 5 / 9 | 3.6 / 7.8 / 10.2 / 16.0 / 17.1 | 24.0 / 37.0 / 40.3 / 52.5 / 48.0 | 0.073 / 0.108 / 0.126 / 0.157 / 0.201 |
| | dark, n = 1 / 2 / 3 / 5 / 9 | 7.8 / 13.1 / 16.1 / 16.2 / 25.0 | 12.7 / 28.8 / 33.9 / 46.5 / 37.7 | 0.032 / 0.070 / 0.077 / 0.117 / 0.004 |

Read per event:

- **A missed cut** costs 5–31 dB·frames of PSNR-Y on the next shot's first 8 frames on four of
  the six cuts (bright, dark, the two live-action ones), and nothing measurable beyond B's first
  frame on the clean and the colourful anime ones, whose next frames come out better in one batch
  (the colourful anime's first frame still loses 7.4 dB at k = 0, and its DISTS sum, 0.20, is the
  largest of all one-batch cases).
  On the dark and live-action cuts part of the sum is shot B rendered differently throughout, and
  the metrics disagree past B's first frames (PSNR-Y worse, DISTS level or better: see the caveats).
  Latent windows that straddle the missed cut (i, ii) cost about what one batch does (from
  14 dB·frames less to 2 more on B). A window boundary at the cut's latent (iii) costs 1.5–3× more
  PSNR-Y at k = 0; at k = 2 the same PSNR-Y as one batch, but 17–76 VMAF points·frames and about
  0.2 DISTS·frames more on B, and 18 VMAF points·frames more on A's last 4 frames.
- **A false cut** costs no fidelity: the PSNR-Y sums are gains on all four clips (−0.5 to −8.8
  dB·frames), the VMAF sums on three. What it can cost is a low-frequency temporal step,
  measurable for 0–11 frames on 2 of the 4 clips; whether it shows on a continuous shot is for the
  visual review. [stitching.md](stitching.md) measured the same kind of boundary (a batch
  boundary inside a shot, no overlap) on held drawings: a jump of 0.78 on top of an in-batch
  flicker of ≈ 1.0, which is visible. The fidelity "gain" fits
  [quality.md](quality.md#--prepend_frames)'s finding that a batch's first frame comes out less
  restored, i.e. closer to its input, which this protocol rewards because the model re-renders.
- **A short shot merged into the previous shot** costs 10–21 dB·frames of PSNR-Y over its own
  frames against running it alone, from 1 frame to 9, and the most per frame at 1 and 2 frames
  (13.5–16.4 dB on a single frame), where the padding copies the previous shot into it; the
  previous shot itself loses nothing consistent. Inside one batch across its cut (the shot going
  on after it) it costs 3.6–7.8 dB·frames for a single frame and 17–25 for 9. The merge is never
  the cheaper option here.

So, against these six cuts, the expected cost of a detector per hour of video is about
(misses per hour) × 5–31 dB·frames (about 0 on two of the six) + (short shots merged into the
previous shot per hour) × 10–21, plus (false cuts per hour) × a temporal step of uncertain
visibility and no fidelity loss.
The error rates per threshold come from question 3's labelled review.

## Caveats

- **Six cuts, all between a darker and a brighter shot** (mean luma 49 → 188, 156 → 33,
  52 → 143, 52 → 167, 140 → 56, 151 → 45): a scene score of 30 or more picks such cuts, and in
  live action it is rare. A cut between shots of similar brightness and colour may leak less
  visibly, and the ghost coefficient is ill-conditioned there (it projects on the difference
  between the shots). The live action is two cuts, from two films, both letterboxed: a quarter
  of the frame is black bars, which dilutes their full-frame deficits by about a quarter. The
  three later cuts were run at k = 0 and 2 only, without the latent layouts, short shots or a
  second seed.
- **One rendering per configuration (seed 42).** The noise band comes from question 1's seed
  runs on other clips, pooled with the three first cuts' aligned references at a second seed (15 pairs,
  783 values per metric). It is wide for PSNR-Y (0.82 dB on 4-frame averages) and very narrow for
  the low-frequency measures (CIEDE2000 lf 0.044, T-err lf 0.078), which flag almost any
  systematic change.
- **PSNR-Y on dark content is hypersensitive:** shot B of the dark cut sits at a mean luma of 33,
  where a one-level shift is worth several dB. Its PSNR-Y deficits far from the cut (+3.4 dB at
  d ≥ 32, where VMAF and DISTS say the one batch is as good or better) are a rendering
  difference, not the cut. DISTS and VMAF are the steadier measures there. The same goes for the
  slow live-action cut's shot B (mean luma 45: +4.2 dB at d ≥ 32, DISTS level).
- **The aligned reference is not a perfect target:** B alone starts on a 1-frame latent and A
  alone ends a batch, and both differ from a longer batch for reasons that have nothing to do
  with the cut (DiT windows, noise by position). The one batch is *better* than the reference on
  A's last frames of the clean cut (−2.2 dB at k = 0) and on the whole of B far from the cut on
  the clean one; the dark cut's whole shot A is worse in one batch (+0.5–1.6 dB, DISTS +0.018,
  with no ghost: the DiT renders the bright shot differently when the dark one is in view).
- **The latent layouts were measured at k = 0 and 2 only**, on the bright and dark cuts; the
  merged short shots at one offset only (the cut on a latent boundary, after a 21-frame shot A),
  which decides how much of A the padding brings in (see above).
- **A shot's first frame is judged against a ground truth the model doesn't aim at:** it
  re-renders ([numerics.md](numerics.md)), so the full-reference metrics favour the frame it
  re-renders least; whether its look or the next frames' is better is for the eye. Its input is
  an x264 I-frame, between 0.8 dB better and 1.5 dB worse than the P-frames after it. One seed
  per prepend run; P = 8 on the three first cuts only. Run times vary by ±7% between runs and by up to 1.7× between sessions on
  the same GPU, so the cost of one latent comes from the time per computed frame.
- `--color_correction none`, 1080p from half-size input, 7B fp16 only. With `lab`, low-frequency
  differences between batches shrink ([quality.md](quality.md#colour-correction)).

## Reproduce

```bash
F=scripts/fr_clips.py; M=scripts/fr_metrics.py; J=scripts/cut_metrics.py
# a hard cut: scdet as sptenc runs it (seeking: approximate indices), then frame-exact scores
python3 $F scan SRC --ss T --duration D --hard 30 --before 48 --after 84 --json scan.json
python3 $F scores SRC --first $((c - 60)) --last $((c + 100))    # c = the cut's exact frame index
# one degraded range around the cut (c = the cut frame), then frame-exact slices of it
python3 $F make SRC --start $((c - 48)) --frames 132 --name cut-long --out clips --degrade d1
python3 $F verify clips/cut-long --source --cv2-python /path/to/seedvr2/.venv/bin/python
python3 $F slice clips/cut-long --first $((27 - k)) --frames 81 --name cut-k$k            # cut at 21 + k
python3 $F slice clips/cut-long --first $((27 - k)) --frames $((21 + k)) --name cut-k$k-A --files gt,d1.lr
python3 $F slice clips/cut-long --first 48 --frames $((60 - k)) --name cut-k$k-B --files gt,d1.lr
python3 $F slice clips/cut-long --first 48 --frames $n --name cut-B$n --files gt,d1.lr            # short shot
python3 $F slice clips/cut-long --first 27 --frames $((21 + n)) --name cut-AB$n --files gt,d1.lr  # merged with A
# runs: one batch each (FFV1 master only); a latent layout through blend_patch.py
python3 scripts/bench.py run cut-k0-one --wrap scripts/ffv1_out.py --env FFV1_OUT_KEEP=0 -- \
  clips/cut-k0.d1.lr.mkv --output out/ --model_dir /path/to/models --dit_model seedvr2_ema_7b_fp16.safetensors \
  --resolution 1080 --attention_mode flash_attn_2 --batch_size 81 --load_cap 81 --temporal_overlap 0 \
  --color_correction none --seed 42                    # likewise A (21 + k), B (60 - k), B<n> (n), AB<n> (21 + n)
python3 scripts/bench.py run cut-k0-lat-hard --wrap scripts/blend_patch.py --wrap scripts/ffv1_out.py \
  --env STITCH_WINDOWS=0-6,6-12,10-16,14-21 --env STITCH_CURVE=cosine --env FFV1_OUT_KEEP=0 -- ...
# the aligned reference, metrics, analysis
python3 $J join out/cut-k0-A.mkv out/cut-k0-B.mkv --out out/cut-k0-aligned.mkv
python3 $M clips/cut-k0.gt.mkv --clip cut-k0 --json-dir m --out one 42 out/cut-k0-one.mkv \
  --out aligned 42 out/cut-k0-aligned.mkv --out lat-hard 42 out/cut-k0-lat-hard.mkv
python3 $J analyze m/cut-k0.one.s42.json --ref m/cut-k0.aligned.s42.json --cut $((21 + k)) --smooth 4 \
  --noise m/*.def.s*.json m/cut-k0.aligned.s43.json --json a/cut-k0-one.json > a/cut-k0-one.md
python3 $J compare --row alone:m/cut-B5.alone.s42.json:0 --row with-A:m/cut-AB5.merged.s42.json:21 \
  --row one-batch:m/cut-k0.one.s42.json:21 --frames 5                         # short shot, n = 5
python3 $J join out/cut-k0-A.mkv out/cut-B5.mkv --out out/cut-AB5-aligned.mkv  # A alone + B5 alone, scored
python3 $M clips/cut-AB5.gt.mkv --clip cut-AB5 --json-dir m --out merged 42 out/cut-AB5.mkv \
  --out aligned 42 out/cut-AB5-aligned.mkv                                      # like the k0 clip above
python3 $J analyze m/cut-AB5.merged.s42.json --ref m/cut-AB5.aligned.s42.json --cut 21 --smooth 4 \
  --noise m/*.def.s*.json m/cut-k0.aligned.s43.json --json a/cut-AB5.json > a/cut-AB5.md  # deficits, ghost
python3 $J selftest clips/cut-k0.gt.mkv --cut 21 --out out/cut-k0-aligned.mkv   # ghost coefficient check
python3 $J summary a/                                                           # one row per analysis
# a shot's first frame: B alone with 4 mirrored frames before it, in its batch (ffv1_out.py drops them)
python3 scripts/bench.py run cut-k0-B-pp4 --wrap scripts/ffv1_out.py --env FFV1_OUT_KEEP=0 -- \
  clips/cut-k0-B.d1.lr.mkv --output out/ --model_dir /path/to/models --dit_model seedvr2_ema_7b_fp16.safetensors \
  --resolution 1080 --attention_mode flash_attn_2 --batch_size 64 --load_cap 60 --temporal_overlap 0 \
  --color_correction none --seed 42 --prepend_frames 4                        # batch = 60 frames + 4
python3 $M clips/cut-k0-B.gt.mkv --clip cut-B60 --json-dir m --out def 42 out/cut-k0-B.mkv \
  --out def 43 out/cut-k0-B-s43.mkv --out pp4 42 out/cut-k0-B-pp4.mkv          # B scored as a clip of its own
python3 $J first m/cut-B60.def.s42.json m/cut-B60.def.s43.json m/cut-B60.pp4.s42.json \
  m/cut-k0.bicubic.s0.json:21           # frame 0 vs frames 1-8, sharpness, the seed noise of the def pair
```

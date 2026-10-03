# Models: 7B fp8, Q4_K_M, 3B and the sharp 7B against a ground truth

> Status: **measured** with the full-reference protocol of [numerics.md](numerics.md#the-full-reference-protocol)
> ([`scripts/fr_metrics.py`](../scripts/fr_metrics.py), 16-bit masters from
> [`scripts/ffv1_out.py`](../scripts/ffv1_out.py)), for the "Model scope for v1" question in
> [DESIGN.md](../../seedvr2x/DESIGN.md#scope); BlockSwap's power-cap explanation tested with
> [`scripts/swap_idle.py`](../scripts/swap_idle.py). SeedVR2 `4490bd1`, `flash_attn_2`, one batch
> of 45 frames, 1080p, `--color_correction none` with `lab` rendered from the same run. Four
> animated clips; live action is not measured.

In short (4 animated clips of 45 frames, a ×2 upscale of a mildly degraded input to 1080p; each
model paired at the same seed with its fp16 parent, or with 7B fp16, and judged against the
seed spreads):

- **7B Q4_K_M is as close to the source as 7B fp16, or closer:** within the seed band or better
  on every clip and metric but one (DISTS on the bright cartoon, +0.002): PSNR-Y +0.11 to +0.49 dB
  (beyond the band on 2 clips), low-frequency colour error lower on 4, flicker lower on 2–3.
- **7B fp8 is slightly but consistently further from the source:** worse in the same direction on
  all 4 clips, PSNR-Y −0.25 to −0.58 dB and VMAF −1.4 to −3.0, beyond the band on 2–3 clips
  (the other two at its edge); `lab` keeps most verdicts. It adds more high-frequency texture
  than fp16 (+5 to +14% luma Laplacian variance). Only 2% faster than fp16, and its file is
  1.8 times Q4_K_M's.
- **The 3B is a different and perceptually worse model:** LPIPS +0.002 to +0.033 and DISTS +0.009
  to +0.032 on all 4 clips beyond both models' seed spreads, VMAF −1.7 to −6.0 on 3; PSNR-Y and
  colour are better on 2 clips, worse on 2. Its DiT is 34% faster, but 15 s saved on a 215 s
  run is 7% of the time, and its DiT peak (20.3 GiB at batch 45) is above Q4_K_M's (17.2).
  **3B fp8** stays near 3B fp16 (closer on detail and flicker, +0.05 to +0.11 ΔE00 on colour).
- **The sharp 7B is the closest of all models to the source:** PSNR-Y +0.40 to +0.67 dB on all 4
  clips (the seed ranges don't overlap), VMAF +1.9 to +3.5, colour error −0.09 to −0.68 ΔE00,
  less flicker, LPIPS lower on 2 clips; with `lab` the gains shrink (+0.27 to +0.47 dB) but
  remain. Despite its name it adds no more high-frequency texture than 7B fp16 (2–10% less on
  average). Same architecture and cost as 7B fp16.
- **Quantization moves the output less than a change of seed:** 40.0–47.8 dB between a quantized
  model and its parent at the same seed, against 37.3–44.4 dB between two seeds of 7B fp16
  (about the same on one clip). Its error is small but systematic: the seed's averages out,
  fp8's doesn't.
- **Every model stays far further from the source than a bicubic upscale** (PSNR-Y −4.5 to
  −12.7 dB, VMAF −15 to −37), as numerics.md found for 7B fp16.
- **BlockSwap's "free" Q4_K_M swap is not a power-cap effect:** idle pauses as long as the moves
  cost exactly their duration (×1.00). Measured between synchronizations, back to back, swapping
  all 36 blocks costs +0.30 s per batch at 1080p batch 5 (+3.6%), +1.2 s on the first: from the
  second batch on, the copies back to the CPU run 4.4 times faster (0.46 s of moves per batch,
  not 1.32).

## Why it matters for seedvr2x

DESIGN.md's [scope](../../seedvr2x/DESIGN.md#scope) asks whether v1 supports the 7B fp16 model
only, or also the fp8, GGUF Q4_K_M and 3B weights from the start. Small GPUs need them:
[vram.md's recipe](vram.md#recipe-per-card-size-validated) runs 8–16 GB cards on Q4_K_M with
all 36 blocks swapped, and the 7B fp16 weights alone (15.35 GiB) don't fit a 16 GB card. Each
extra model costs code (GGUF dequantisation adapted from city96, the 3B DiT's cache quirk) and
tests. [vram.md](vram.md#other-models) measured their memory and speed, and how far their
outputs are from the 7B fp16 one (≈ 44 dB for fp8 and Q4_K_M, 32 dB for the 3B), but not
whether those differences matter: a quantized model is as good as 7B fp16 if it stays as close
to the source as another seed of 7B fp16 does.

## Method

### Models

| Model | `--dit_model` | File | On the GPU | Compared with |
|---|---|---|---|---|
| 7B fp16 | `seedvr2_ema_7b_fp16.safetensors` | 16.5 GB | fp16 weights, cast to bf16 per layer under autocast | – (the reference) |
| 7B fp8 | `seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors` | 8.5 GB | fp8 e4m3fn weights (the last block fp16), cast per layer | 7B fp16, its parent |
| 7B Q4_K_M | `seedvr2_ema_7b-Q4_K_M.gguf` | 4.8 GB | GGUF 4-bit, dequantized per layer | 7B fp16, its parent |
| 3B fp16 | `seedvr2_ema_3b_fp16.safetensors` | 6.8 GB | fp16, a smaller DiT (32 blocks) | 7B fp16: a different model |
| 3B fp8 | `seedvr2_ema_3b_fp8_e4m3fn.safetensors` | 3.4 GB | fp8 e4m3fn | 3B fp16, its parent |
| 7B sharp | `seedvr2_ema_7b_sharp_fp16.safetensors` | 16.5 GB | fp16; ByteDance's "sharp" 7B weights, same architecture | 7B fp16: different weights |

All use the same VAE (`ema_vae_fp16`). numz's CLI downloads each from its registry and checks
its sha256; the sharp file was fetched with the CLI's own `download_weight` (sha256 verified).

### Runs and metrics

The full-reference protocol of [numerics.md](numerics.md#the-full-reference-protocol),
unchanged: the four animated clips (45 frames, 1080p ground truth), the d1 input (×1/2 Mitchell,
x264 CRF 20, 8-bit RGB), one batch of 45 frames at `--resolution 1080`, numz's defaults as
question 1's `def` (`--attention_mode flash_attn_2 --batch_size 45 --load_cap 45
--color_correction none`, [`numerics_patch.py`](../scripts/numerics_patch.py) with only
`NUM_CC_EXTRA=lab`, so `lab` is rendered from the same run, and `ffv1_out.py` for 16-bit RGB
masters). Only `--dit_model` changes. 7B fp16 is question 1's `def`, seeds 42, 43 and 1234,
reused as it is. 36 runs, none failed.

- **Seeds:** 42 for the approximations of a parent (7B fp8, Q4_K_M, 3B fp8); 42, 43 and 1234
  for the two other models (3B fp16, 7B sharp), which need a seed band of their own.
- **Verdicts** (fr_metrics.py's): the per-frame difference to the reference at the same seed,
  pooled over the common seeds, with a 95% block-bootstrap interval; **better / worse** when the
  interval excludes 0 *and* the difference exceeds the band, else **within**. Band: the parent's
  3-seed spread for a quantized model (7B fp16's for 7B fp8 and Q4_K_M, 3B fp16's for 3B fp8);
  for 3B fp16 and the sharp 7B against 7B fp16, the larger of the two models' spreads. Each
  output is scored as rendered (`none`) and after `lab`.
- **Distances between masters:** RGB PSNR over the 45 frames between two 16-bit masters
  (`ffv1_out.py --diff`), each quantized model against its parent at seed 42, next to the
  distance between two seeds of the same model.
- **Detail:** the luma Laplacian variance (4-neighbour, 8-bit scale, as `fr_clips.py` measures
  the clips), mean over the 45 frames, against the ground truth's.

## Results

### Against the ground truth

`none`; 3-seed means for 7B fp16, 3B fp16 and the sharp 7B, seed 42 for the others:

| Clip | Model | PSNR-Y | SSIM-Y | LPIPS | DISTS | VMAF | ΔE00 lf | Temporal error |
|---|---|---|---|---|---|---|---|---|
| anime-clean | 7B fp16 | 27.72 | 0.868 | 0.212 | 0.098 | 65.3 | 2.31 | 7.72 |
| | 7B fp8 | 27.18 | 0.858 | 0.224 | 0.103 | 61.9 | 2.45 | 8.11 |
| | 7B Q4_K_M | 27.76 | 0.866 | 0.216 | 0.101 | 64.9 | 2.18 | 7.77 |
| | 3B fp16 | 27.04 | 0.853 | 0.241 | 0.116 | 59.4 | 2.61 | 8.33 |
| | 3B fp8 | 27.23 | 0.861 | 0.228 | 0.108 | 60.3 | 2.72 | 7.98 |
| | 7B sharp | 28.39 | 0.874 | 0.211 | 0.102 | 67.3 | 1.63 | 7.44 |
| | bicubic | 39.69 | 0.970 | 0.097 | 0.059 | 92.1 | 0.65 | 2.40 |
| anime-grain | 7B fp16 | 29.27 | 0.916 | 0.114 | 0.069 | 65.2 | 1.84 | 5.97 |
| | 7B fp8 | 28.64 | 0.910 | 0.117 | 0.070 | 62.3 | 1.95 | 6.31 |
| | 7B Q4_K_M | 29.71 | 0.918 | 0.112 | 0.068 | 67.4 | 1.69 | 5.77 |
| | 3B fp16 | 28.79 | 0.917 | 0.147 | 0.101 | 63.6 | 2.03 | 5.79 |
| | 3B fp8 | 29.07 | 0.921 | 0.140 | 0.097 | 65.0 | 2.13 | 5.52 |
| | 7B sharp | 29.90 | 0.912 | 0.095 | 0.058 | 68.2 | 1.49 | 5.73 |
| | bicubic | 37.38 | 0.954 | 0.098 | 0.060 | 86.6 | 0.68 | 2.96 |
| anime-dark | 7B fp16 | 32.12 | 0.940 | 0.213 | 0.136 | 51.1 | 2.36 | 5.87 |
| | 7B fp8 | 31.51 | 0.937 | 0.220 | 0.140 | 47.9 | 2.52 | 5.92 |
| | 7B Q4_K_M | 32.57 | 0.940 | 0.212 | 0.135 | 51.7 | 2.25 | 5.67 |
| | 3B fp16 | 32.59 | 0.938 | 0.230 | 0.148 | 46.9 | 2.19 | 5.77 |
| | 3B fp8 | 32.57 | 0.938 | 0.230 | 0.148 | 46.4 | 2.23 | 5.69 |
| | 7B sharp | 32.52 | 0.940 | 0.203 | 0.133 | 54.6 | 2.27 | 5.40 |
| | bicubic | 43.16 | 0.970 | 0.171 | 0.131 | 83.5 | 0.74 | 1.61 |
| cartoon-bright | 7B fp16 | 27.13 | 0.931 | 0.080 | 0.100 | 65.3 | 2.59 | 7.42 |
| | 7B fp8 | 26.83 | 0.929 | 0.082 | 0.104 | 63.7 | 2.69 | 7.61 |
| | 7B Q4_K_M | 27.19 | 0.930 | 0.080 | 0.102 | 65.2 | 2.55 | 7.39 |
| | 3B fp16 | 27.56 | 0.930 | 0.082 | 0.109 | 66.1 | 2.53 | 6.77 |
| | 3B fp8 | 27.53 | 0.930 | 0.081 | 0.113 | 66.3 | 2.64 | 6.67 |
| | 7B sharp | 27.78 | 0.934 | 0.074 | 0.104 | 67.5 | 2.34 | 7.01 |
| | bicubic | 32.23 | 0.955 | 0.118 | 0.085 | 82.7 | 0.63 | 3.92 |

- **Every model is further from the source than bicubic** on PSNR-Y (4.5 to 12.7 dB), SSIM-Y,
  VMAF (15 to 37 points), low-frequency colour and temporal error, on every clip. LPIPS is
  lower than bicubic's only on cartoon-bright (all models) and, for the sharp 7B, on
  anime-grain (−0.003, DISTS −0.002 too).
- **Seed spreads** (max − min of 3 per-seed means, `none`): PSNR-Y 0.09–0.47 dB for 7B fp16,
  0.10–0.24 for 3B fp16, 0.10–0.55 for the sharp 7B; VMAF 0.34–3.30, 0.23–1.89 and 0.19–2.50;
  LPIPS 0.0002–0.0183, 0.0001–0.0061 and 0.0004–0.0190 (per clip: `fr_metrics.py --summary`,
  see [Reproduce](#reproduce)). The 3B's PSNR-Y spread is the narrowest on 3 of 4 clips.

### Paired with the parent or with 7B fp16

Mean paired difference (model − reference): the range over the 4 clips, and how many clips are
better (B) or worse (W) beyond the band; "–" = within everywhere. Seed 42 but where noted.

`none`:

| Model − reference | PSNR-Y | SSIM-Y | LPIPS | DISTS | VMAF | ΔE00 lf | Temporal error |
|---|---|---|---|---|---|---|---|
| 7B fp8 − 7B fp16 | −0.58..−0.25 (2 W) | −0.0072..−0.0020 (3 W) | +0.0016..+0.0068 (1 W) | +0.0004..+0.0044 (1 W) | −2.97..−1.43 (3 W) | +0.092..+0.159 (4 W) | +0.025..+0.320 (2 W) |
| Q4_K_M − 7B fp16 | +0.11..+0.49 (2 B) | −0.0003..+0.0030 (1 B) | −0.0053..−0.0004 (1 B) | −0.0037..+0.0019 (1 W) | +0.01..+2.19 (1 B) | −0.129..−0.047 (4 B) | −0.224..−0.018 (2 B) |
| 3B fp8 − 3B fp16 | −0.04..+0.34 (2 B) | +0.0004..+0.0083 (2 B) | −0.0142..−0.0008 (3 B) | −0.0086..+0.0043 (2 B, 1 W) | +0.07..+1.59 (2 B) | +0.046..+0.106 (4 W) | −0.362..−0.090 (4 B) |
| 3B fp16 − 7B fp16 (3 seeds) | −0.69..+0.46 (2 B, 2 W) | −0.0144..+0.0015 (3 W) | +0.0016..+0.0328 (4 W) | +0.0091..+0.0318 (4 W) | −5.96..+0.76 (1 B, 3 W) | −0.176..+0.304 (2 B, 2 W) | −0.654..+0.607 (3 B, 1 W) |
| 7B sharp − 7B fp16 (3 seeds) | +0.40..+0.67 (4 B) | −0.0043..+0.0062 (1 B) | −0.0197..−0.0009 (2 B) | −0.0113..+0.0040 (1 B, 1 W) | +1.92..+3.48 (3 B) | −0.680..−0.091 (4 B) | −0.469..−0.238 (2 B) |
| 3B fp8 − 7B fp16 | −0.41..+0.49 (2 B) | −0.0043..+0.0057 (1 B) | +0.0007..+0.0246 (2 W) | +0.0066..+0.0276 (3 W) | −4.64..+1.14 (1 B, 2 W) | −0.130..+0.430 (1 B, 3 W) | −0.790..+0.194 (3 B) |

`lab`:

| Model − reference | PSNR-Y | SSIM-Y | LPIPS | DISTS | VMAF | ΔE00 lf | Temporal error |
|---|---|---|---|---|---|---|---|
| 7B fp8 − 7B fp16 | −0.47..−0.16 (2 W) | −0.0068..−0.0013 (2 W) | +0.0022..+0.0072 (1 W) | +0.0004..+0.0042 (1 W) | −2.26..−1.07 (2 W) | +0.027..+0.055 (4 W) | +0.079..+0.319 (2 W) |
| Q4_K_M − 7B fp16 | +0.04..+0.38 (1 B) | −0.0003..+0.0027 (1 B) | −0.0049..−0.0006 (1 B) | −0.0034..+0.0017 (1 W) | −0.05..+1.64 (1 B) | −0.032..−0.003 (2 B) | −0.208..−0.025 (3 B) |
| 3B fp8 − 3B fp16 | +0.06..+0.55 (2 B) | +0.0004..+0.0082 (2 B) | −0.0150..−0.0011 (3 B) | −0.0087..+0.0038 (2 B, 1 W) | +0.02..+1.48 (2 B) | −0.010..+0.003 (1 B) | −0.366..−0.043 (3 B) |
| 3B fp16 − 7B fp16 (3 seeds) | −0.55..+0.42 (1 B, 2 W) | −0.0139..+0.0018 (2 W) | +0.0011..+0.0321 (4 W) | +0.0067..+0.0297 (4 W) | −4.87..−0.26 (3 W) | −0.070..+0.220 (2 B, 2 W) | −0.546..+0.518 (1 B, 1 W) |
| 7B sharp − 7B fp16 (3 seeds) | +0.27..+0.47 (2 B) | −0.0042..+0.0058 (1 B) | −0.0194..−0.0006 (2 B) | −0.0102..+0.0036 (1 B, 1 W) | +1.57..+2.27 (2 B) | −0.124..+0.008 (3 B) | −0.381..−0.232 (2 B) |

- **7B fp8:** every mean difference points away from the source on every clip (but the
  low-frequency flicker on one); the misses are at the band's edge (PSNR-Y −0.46 against a band
  of 0.47 on anime-clean, −0.25 against 0.25 on cartoon-bright). The effect is about one seed
  band, 2–4 times numerics.md's largest numerics change (|ΔPSNR-Y| ≤ 0.14 dB).
- **Q4_K_M:** every mean difference points toward the source but three: DISTS on cartoon-bright
  (+0.0019 against a band of 0.0008, the one "worse"), SSIM-Y there (−0.0003) and the
  low-frequency flicker on anime-clean (+0.013), both within.
- **3B fp16, per clip** (all 3 seeds on one side of all 3 of 7B fp16's, `none`): further on
  every metric on anime-clean; further on PSNR-Y, LPIPS, DISTS, VMAF and colour on anime-grain;
  closer on PSNR-Y, colour and flicker on anime-dark and cartoon-bright, but further on SSIM-Y,
  LPIPS and DISTS there too (and VMAF on anime-dark). The perceptual metrics are worse
  everywhere.
- **The sharp 7B, per clip:** its 3 seeds are all above 7B fp16's on PSNR-Y and all below on
  colour error on the 4 clips, closer on LPIPS and VMAF on 3, overlapping on anime-clean's
  detail metrics; only DISTS on cartoon-bright (+0.004) and SSIM-Y on anime-grain (−0.004,
  within the band) go against it. After `lab`, which removes most colour error from both, the
  PSNR-Y gain falls to +0.27–0.47 dB (beyond the band on 2 clips) and the LPIPS gains stay
  (anime-grain −0.019).
- **3B fp8 − 3B fp16:** closer on detail and flicker on 2–4 clips, but its colour is off by
  +0.05 to +0.11 ΔE00 (4 W, gone after `lab`).

### Distances between masters

RGB PSNR between two 16-bit masters, dB (seed 42 unless noted):

| Pair | anime-clean | anime-grain | anime-dark | cartoon-bright |
|---|---|---|---|---|
| 7B fp8 − 7B fp16 | 40.25 | 42.33 | 44.98 | 39.95 |
| 7B Q4_K_M − 7B fp16 | 40.80 | 43.10 | 46.64 | 40.18 |
| 3B fp8 − 3B fp16 | 40.57 | 42.29 | 47.79 | 40.20 |
| 7B fp16, seed 43 − seed 42 | 37.31 | 40.51 | 44.37 | 39.81 |
| 7B fp16, seed 1234 − seed 42 | 37.71 | 40.52 | 44.32 | 39.90 |
| 3B fp16, seed 43 − seed 42 | 35.82 | 39.33 | 43.24 | 36.56 |
| 7B sharp, seed 43 − seed 42 | 36.55 | 38.84 | 42.92 | 38.75 |
| 7B sharp − 7B fp16 | 32.96 | 35.53 | 40.35 | 33.79 |
| 3B fp16 − 7B fp16 | 29.61 | 32.21 | 36.32 | 29.52 |

- **Quantization changes the output less than a seed does:** the 7B's by 0.6–3.5 dB less on
  three clips and as much on cartoon-bright (0.05–0.4 dB less), the 3B's by 3.0–4.8 dB less;
  Q4_K_M changes it less than fp8 on every clip. Yet fp8's change is a bias: a seed's change
  averages out over a frame's metrics (the seed spreads above), fp8's moves every clip the same
  way.
- **The sharp 7B and the 3B are other models:** 33–40 and 30–36 dB from 7B fp16, further than
  any two seeds of one model.

### Detail

Luma Laplacian variance, mean over 45 frames (8-bit scale), ratio to the ground truth in
parentheses; 3-seed mean [range] for the models with three:

| Clip | GT | bicubic | 7B fp16 | 7B fp8 | 7B Q4_K_M | 3B fp16 | 3B fp8 | 7B sharp |
|---|---|---|---|---|---|---|---|---|
| anime-clean | 37.0 | 13.5 (0.36) | 239.0 (6.46) [208–258] | 268.0 (7.25) | 252.0 (6.82) | 277.2 (7.50) [267–284] | 247.5 (6.69) | 228.3 (6.18) [197–246] |
| anime-grain | 59.4 | 20.7 (0.35) | 170.8 (2.88) [162–179] | 193.9 (3.27) | 164.5 (2.77) | 202.3 (3.41) [199–206] | 170.9 (2.88) | 153.4 (2.58) [141–165] |
| anime-dark | 11.6 | 0.9 (0.07) | 9.3 (0.80) [8.8–9.6] | 9.9 (0.85) | 9.4 (0.81) | 10.5 (0.91) [10.2–10.9] | 10.6 (0.92) | 9.1 (0.79) [8.7–9.7] |
| cartoon-bright | 276.9 | 44.7 (0.16) | 437.9 (1.58) [426–449] | 459.4 (1.66) | 432.3 (1.56) | 385.7 (1.39) [382–389] | 351.0 (1.27) | 414.6 (1.50) [400–423] |

- **The models add high-frequency energy the source doesn't have:** 6.2–7.5 times the source's
  on the clean digital clip (bicubic: a third of it), 2.6–3.4 times on the grainy one, 1.3–1.7
  on the bright cartoon; only on the dark clip do they stay below it (0.79–0.92).
- **7B fp8 adds the most of the 7B weights** (+5 to +14% over fp16), the 3B more than the 7B on
  the anime clips; Q4_K_M is within 7B fp16's seed range on all 4 clips.
- **"Sharp" is not sharper here:** the sharp 7B adds 2–10% less than 7B fp16 on average (the
  seed ranges separate on cartoon-bright only), in line with its higher PSNR. Its name may
  describe what it does to blurrier inputs than this mild ×2 upscale; that is not measured.

### Speed and memory

A by-product of the runs (45 frames at 1080p, one batch, untiled VAE). DiT inference time from
the CLI's log, in runs made at the same GPU clock (VAE encode 49–56 s; this session's GPU sat at
its power cap's lowest clock, see [the BlockSwap test](#blockswaps-free-q4_k_m-swap-the-power-cap-test)):

| Model | DiT inference | Against 7B fp16 | DiT torch peak |
|---|---|---|---|
| 7B fp16 | 43.0–43.2 s (4 runs) | – | 28.15 GiB |
| 7B sharp | 43.0–43.2 s (11 runs) | same | 28.15 GiB |
| 7B fp8 | 42.1 s (2 runs; 2 others ran at a higher clock) | −2% | 20.69 GiB |
| 7B Q4_K_M | 43.8 s (4 runs) | +2% | 17.23 GiB |
| 3B fp16 | 28.3 s (12 runs) | −34% | 20.34 GiB |
| 3B fp8 | 26.9 s (4 runs) | −38% | 17.18 GiB |

The VAE dominates these runs (encode ≈ 56 s and decode 84–127 s, whatever the model, against
27–44 s of DiT): the 3B's DiT saves 15 s of ≈ 215 s. The peaks match vram.md's per-model
formulas ([Practical rules](vram.md#practical-rules)) within 0.1 GiB.

## BlockSwap's free Q4_K_M swap: the power-cap test

[vram.md](vram.md#blockswap) found that swapping all 36 Q4_K_M blocks cost nothing end to end
(7.63 s per batch against 7.61 and 7.90 without swap) although
[`swap_probe.py`](../scripts/swap_probe.py) timed 0.88 s of synchronous moves per batch, and
guessed that the GPU, always at its 600 W cap, runs its kernels faster after the copies' idle
gaps (energy-bound compute).

### Method

7B Q4_K_M, 1080p, batch 5, 2 batches (10 frames of the 1080p anime segment of vram.md's
`swapctl` runs, from frame 48), `flash_attn_2`, `--color_correction none`,
`--dit_offload_device cpu`, one CLI run per arm, all wrapped by
[`swap_idle.py`](../scripts/swap_idle.py) (NVML at 20 Hz: power, energy counter, SM clock,
clock-event reasons; every DiT forward timed between two synchronizations), back to back
inside one acquisition of the GPU (12 runs, 13 min):

| Arm | Configuration |
|---|---|
| P | `--blocks_to_swap 36`, with `swap_probe.py`: the move time of every block, both ways |
| A | `--blocks_to_swap 0` |
| B | `--blocks_to_swap 36` |
| C | `--blocks_to_swap 0`, every block on the GPU, and where each block would move a synchronous idle pause as long as P's mean move of that block (`SWAP_IDLE_BLOCKS=36 SWAP_IDLE_FROM=P.swap.json`: 72 pauses, 0.890 s per forward) |
| D | as C with the pauses scaled to 0: the 72 synchronizations alone |

Order P, then A B C D A B C D A B C. The 12 outputs and a validation run (constant pauses of 6
and 18 ms) are bit-identical: pauses, synchronizations and swapping change no numerics.

### Results

DiT forward time between synchronizations, s (forward 2 is the steady batch; forward 1 also
pays the first moves of every block):

| Arm | Forward 2, each run | Mean | − A | Forward 1, each run | − A |
|---|---|---|---|---|---|
| A, no swap | 8.524, 8.525, 8.526 | 8.525 | – | 8.689, 8.690, 8.712 | – |
| B, swap 36 | 8.871, 8.805, 8.808 | 8.828 | **+0.303** (+3.6%) | 9.938, 9.893, 9.916 | +1.219 (+14%) |
| C, idle pauses of 0.890 s | 9.421, 9.418, 9.415 | 9.418 | **+0.893** (1.00 × the pauses) | 9.457, 9.588, 9.583 | +0.846 (0.95 ×) |
| D, synchronizations only | 8.526, 8.527 | 8.527 | +0.002 | 8.612, 8.713 | −0.027 |
| P, swap 36 + probe | 8.834 | 8.834 | +0.309 | 9.683 | +0.99 |

- **Idle time costs its full duration:** 72 pauses totalling 0.890 s per forward add 0.889 to
  0.896 s (three repeats); the synchronizations alone add 2 ms. Nothing after a pause runs
  faster. **The power-cap explanation does not hold.**
- **The GPU had no clock to gain:** in every steady forward of every arm, pauses included, the SM
  clock read 577 MHz in all 20 Hz samples, with the software power cap active in all of them.
  NVML reported 685–762 W of board power against the 600 W limit (its energy counter 514–679 W:
  the two disagree by up to a third); the power controller sat at its lowest clock, so an idle
  gap could not buy a higher one. The clock only rose (up to 2.7 GHz) during the second-long
  idle gaps between phases, and fell back to 577 MHz within a second of load.
- **BlockSwap is not free, it is cheap after the first batch:** +0.30 s per batch (+3.6% at
  1080p batch 5), +1.2 s (+14%) on the first. The probe splits the moves by batch:

  | Batch | In: CPU → GPU | Out: GPU → CPU | Moves | Block compute (36 blocks) |
  |---|---|---|---|---|
  | 1 | 0.268 s, 17 GB/s | 1.056 s, 4.4 GB/s | 1.324 s | 8.307 s |
  | 2 | 0.217 s, 21 GB/s | 0.239 s, 19 GB/s | 0.456 s | 8.354 s (232 ms per block) |

  The way back is 4.4 times faster from the second batch on, as expected if the first batch
  copies every block into fresh pageable memory and the following ones into heap pages glibc
  recycles (Q4_K_M's tensors, ≤ 20 MiB, stay under its 32 MiB mmap threshold, as vram.md
  guessed). vram.md's 0.88 s was the mean of the two batches. The swapped blocks also computed
  0.16 s faster per batch than resident ones (232 against ≈ 236.6 ms per block: A's forward less
  the 7 ms outside the blocks), at the same 577 MHz; the cause is not established.
  0.456 − 0.16 ≈ 0.30 s.
- **Why vram.md saw no cost:** single runs in a drifting session, whose two no-swap controls
  differed by 0.29 s, the size of the effect. Here, back to back at a pinned clock, repeats
  agree within 2 ms (A) and 66 ms (B).
- **For the recipe:** with Q4_K_M, all 36 blocks swapped cost ≈ 0.3 s per batch after the first
  on this PCIe 5.0 link, a fixed cost per forward (3.6% at batch 5, less with longer batches).
  fp8 and fp16 blocks hold tensors above 32 MiB (36 and 72 MiB), which glibc would map afresh on
  every copy back if the mechanism above holds (vram.md measured 4.5–4.8 GB/s for them); their
  per-block costs were not re-measured.

## Caveats

- **Four animated clips**, 45 frames each, a ×2 upscale of a mild degradation; live action, film
  grain, heavier degradations and 4K are not measured. The sharp 7B's name suggests a behaviour
  on blurrier inputs that this protocol doesn't probe.
- **One seed for each quantized model**, judged against its parent's 3-seed spread: a
  quantization effect that changed sign with the seed would not show. The consistency over 4
  clips (fp8 always further, Q4_K_M nearly always closer) argues against chance.
- **Fidelity is not quality.** The model re-renders ([numerics.md](numerics.md#the-model-re-renders));
  closer to the source can mean redrawing less. The metrics compare models, they don't rate
  them; no visual review was done for this question.
- **numz's implementations:** the fp8 file is numz's ("mixed", the last block in fp16), the GGUF
  path numz's city96-derived dequantisation. A seedvr2x reimplementation must reproduce them;
  its own fp8 or GGUF numerics would need their own check.
- **Speed in one clock state:** this session's GPU ran its DiT at 577 MHz under the power cap
  (see the BlockSwap test); the relative DiT times hold for this state; vram.md's other-models
  table gives another (fp8 −3%, Q4_K_M +3%, 3B −25% at batch 21).
- **The power-cap test ran with the controller at its floor:** with clock headroom, idle gaps
  might still buy a higher clock; not reproduced here, and not needed to explain vram.md's
  observation (the swap's 0.30 s is within its controls' 0.29 s spread). NVML's power fields
  and its energy counter disagree by up to a third on this card: the clock and the
  clock-event reasons are the unambiguous signals.

## Reproduce

```bash
S=scripts; C=/path/to/clips; OUT=/path/to/out
# one run per model and seed (as numerics.md's def, only --dit_model changes)
python3 $S/bench.py run q7-anime-clean-d1-7bq4-s42 --wrap $S/numerics_patch.py --wrap $S/ffv1_out.py \
  --env FFV1_OUT_KEEP=0 --env NUM_CC_EXTRA=lab -- $C/anime-clean.d1.lr.mkv --output $OUT/anime-clean-d1/ \
  --model_dir /path/to/models --dit_model seedvr2_ema_7b-Q4_K_M.gguf --resolution 1080 \
  --attention_mode flash_attn_2 --batch_size 45 --load_cap 45 --color_correction none --seed 42
# metrics: the master as the model's label, the lab extra as label+lab, one fr_metrics call per clip
python3 $S/fr_metrics.py $C/anime-clean.gt.mkv --clip anime-clean-d1 --json-dir m/d1-none \
  --out 7bq4 42 $OUT/anime-clean-d1/q7-anime-clean-d1-7bq4-s42.mkv \
  --out def 42 $OUT/anime-clean-d1/q1-anime-clean-d1-def-s42.mkv --out bicubic 0 $C/anime-clean.d1.bicubic.mkv
# summaries: paired with 7B fp16, then 3B fp8 with its parent (the band of the --default variant)
python3 $S/fr_metrics.py --summary m/d1-none --default def --reference bicubic > d1-none.md
python3 $S/fr_metrics.py --summary m/d1-none --default 3bfp16 --reference bicubic > d1-none-3b.md
# distance between masters
python3 $S/ffv1_out.py --diff $OUT/anime-clean-d1/q7-anime-clean-d1-7bq4-s42.mkv \
  $OUT/anime-clean-d1/q1-anime-clean-d1-def-s42.mkv
# BlockSwap test: probe, then the arms back to back (1080p, batch 5, 2 batches)
A="<input> --output out/ --model_dir /path/to/models --skip_first_frames 48 --attention_mode flash_attn_2 \
  --color_correction none --dit_model seedvr2_ema_7b-Q4_K_M.gguf --resolution 1080 --batch_size 5 \
  --load_cap 10 --dit_offload_device cpu"
python3 $S/bench.py run partb-P1 --wrap $S/swap_idle.py --wrap $S/swap_probe.py \
  --env SWAP_IDLE_NVML_HZ=20 -- $A --blocks_to_swap 36
python3 $S/bench.py run partb-A1 --wrap $S/swap_idle.py --env SWAP_IDLE_NVML_HZ=20 -- $A --blocks_to_swap 0
python3 $S/bench.py run partb-B1 --wrap $S/swap_idle.py --env SWAP_IDLE_NVML_HZ=20 -- $A --blocks_to_swap 36
python3 $S/bench.py run partb-C1 --wrap $S/swap_idle.py --env SWAP_IDLE_NVML_HZ=20 \
  --env SWAP_IDLE_BLOCKS=36 --env SWAP_IDLE_FROM=runs/partb-P1.swap.json -- $A --blocks_to_swap 0
python3 $S/bench.py run partb-D1 ... --env SWAP_IDLE_SCALE=0 ...     # C, synchronizations only
# -> runs/<name>.idle.json (per DiT forward: synchronized time, pauses, energy, SM clock, reasons)
python3 $S/swap_idle.py --selftest; python3 $S/swap_idle.py --nvml-test 10
```

<details>
<summary>Run names (<code>flash_attn_2</code>, one batch of 45 frames at 1080p, <code>--color_correction
none</code> plus the <code>lab</code> rendering of the same run)</summary>

`q7-<clip>-d1-<model>-s<seed>`, clips `anime-clean`, `anime-grain`, `anime-dark`,
`cartoon-bright`; models `7bfp8`, `7bq4`, `3bfp8` (seed 42), `3bfp16`, `7bsharp` (seeds 42, 43,
1234). 7B fp16: question 1's `q1-<clip>-d1-def-s{42,43,1234}`. Metric labels `<clip>-d1`, the
`lab` renderings as `<model>+lab`. BlockSwap test: `partb-P1`, `partb-{A,B,C}{1,2,3}`,
`partb-D{1,2}`, validation `partb-T1`.

</details>

# Measurement campaign: progress

Checklist of the measurements that answer the open questions of
[seedvr2x/DESIGN.md](../seedvr2x/DESIGN.md#open-questions), in priority order. Each question ends
with a decision brief (recommendation, evidence, confidence, caveats) for the design
conversation. Methods and results go in [docs/](docs/), scripts in [scripts/](scripts/).

`[x]` done · `[~]` in progress · `[ ]` to do · `[-]` dropped (reason given)

Last update: 2026-10-03 08:51 CEST

## 0. Setup

- [x] Plan and full-reference protocol proposed (2026-10-02)
- [x] ByteDance's original fp32 weights (DiT 33 GB, VAE 1 GB) downloaded, sha256 checked
- [x] `numerics_patch.py` (`--wrap`): numerics and input-preparation switches; all defaults
      bit-identical to the plain CLI; each switch checked to take effect. Smoke tests (9 frames,
      1080p): every numerics switch stays 50–58 dB from the default (seeds differ by 42 dB);
      padding changes the bottom 8 rows (27–29 dB); TF32 and the Conv3d workaround change nothing
- [x] Metrics environment, separate from the SeedVR2 venv (LPIPS, DISTS, SSIM; VMAF through ffmpeg)
- [x] Full-reference tooling: ground-truth clips and degraded inputs (frame-exact), metrics
      validated on known cases (`fr_clips.py`, `fr_metrics.py`); VMAF identical to `sptenc vmaf`
      frame for frame. Four clips so far: clean digital anime, grainy cel anime, dark anime,
      bright flat-colour cartoon (web source, more compressed)
- [ ] Samples from the user: live action (grainy film, clean digital), HEVC with open GOPs,
      MPEG-2, an anamorphic DVD and a telecined one

## 1. Numerics and input preparation

Protocol: ground truth = 45-frame single-shot clips from 1080p Blu-rays; input = half size
(Mitchell bicubic) + x264 CRF 20, fed to the CLI as 8-bit RGB; one batch, upscaled back to
1080p; colour correction off (`lab` re-rendered from the same runs). Metrics against the ground
truth: VMAF, PSNR-Y, SSIM, LPIPS, DISTS, low-frequency colour error, temporal error. A variant
wins only beyond the seed band (numz's default, 3 seeds), on at least 4 of 5 clips, with no
temporal regression.

- [x] Attention dtype: numz runs attention in bf16 on this GPU, like ByteDance (its compute
      dtype is bf16 wherever bf16 works): no run needed
- [x] Seed band: numz's default, 3 seeds, every clip (e.g. VMAF spread 0.3–3.3 points, LPIPS
      0.0002–0.018 depending on the clip)
- [x] ByteDance parity: fp32 original weights, fp32 RoPE table, posterior sample, fp32 input
      preparation: within the seed band on all 4 clips and every metric (colour and low-frequency
      temporal error slightly better on the dark clip only)
- [x] Protocol checks: the output lines up with the ground truth (sub-pixel shift ≤ 0.1 px,
      frame t matches t). numz's default is far below a plain bicubic upscale (e.g. PSNR-Y 27.7
      vs 39.7, VMAF 65 vs 92 on the clean anime clip) because the model re-renders: given the
      ground truth itself as input, its output is 25–30 dB PSNR-Y away from it, as far as from
      the degraded input. Comparisons between variants are unaffected
- [x] Single factors: RoPE table, VAE sample, fp32 input preparation, fp32 weights: all within
      the seed band on all 4 clips, with and without `lab` (largest change 0.14 dB PSNR-Y, 0.8
      VMAF); only tiny low-frequency colour effects
- [x] Resize kernel: Spline36 within the band everywhere; Lanczos better on the grainy clip only
      (+0.24 dB, LPIPS −0.005); antialias off within on the main metrics, slightly worse
      low-frequency colour and temporal error without `lab`
- [x] Padding: reflect and replicate fix the bottom rows (+6.5 to +12.5 dB there) but make the
      whole frame worse on 3 of 4 clips (PSNR-Y −0.6 to −0.9 dB, VMAF −5.7 to −14.7, beyond
      both seed spreads); centre crop and grey padding are worse too. Black rows anchor the
      model's tone (without them its luma drift grows, up to +40% frame to frame). **Reflect to
      the next multiple of 16, then 16 black rows** fixes the band (+7 to +13 dB) with the rest
      of the frame within the band (PSNR-Y +0.34 to +0.40 on 2 of 3 clips)
- [x] Does a black band help where numz pads nothing (720p, 4K)? At 720p, 16 black rows improve
      the rest of the frame on 4 of 4 clips (PSNR-Y +0.23 to +0.61, VMAF +1.8 to +6.4) but
      black right under the picture damages its bottom rows (−2.5 to −10.4 dB)
- [ ] A guaranteed reflected margin before the black rows (`NUM_PAD=reflect>=8+black+16`,
      ready; identical to the measured mode at 1080p) at 720p: 4 runs, not reached before the
      09:00 stop (the GPU lock was busy)
- [x] Forced fp16 attention (GPUs without bf16): within the seed band on all 4 clips
- [x] Second degradation (area, CRF 26) on 2 clips: same picture as d1; reflect then black
      +0.26 dB; the sharper kernels gain a little on the softer input (+0.16 to +0.26 dB),
      beyond the spread on the grainy clip only
- [x] Reflect then black on the letterboxed clip: +0.29 dB PSNR-Y, letterbox kept black
- [x] The combined choice, checked once: numz's numerics + bicubic with antialias + reflect then
      black is the `reflect+black+16` run above
- [~] Doc ([docs/numerics.md](docs/numerics.md), written) + decision brief (numerics brief given
      2026-10-03; input preparation after the 720p margin test)

## 2. Cuts

- [x] Three real hard cuts (dark, bright, clean anime; live action later): one range per cut
      degraded once, then frame-exact slices (81-frame clips with the cut at frame 21 + k,
      k = 0…3, each shot alone, short shots of 1–9 frames)
- [x] Tooling: explicit window layouts in `blend_patch.py` (unset = bit-identical to before),
      `cut_metrics.py` (deficit vs the aligned reference by distance to the cut, ghost
      coefficient, validated on synthetic mixes)
- [x] Smoke test, bright cut, DiT windows split exactly at the cut but one VAE pass: frames
      before the cut bit-identical to the first shot alone; after it, −7.9 dB PSNR-Y and −40
      VMAF on the first frame, back within 0.2 dB after 16–31 frames: the VAE alone carries the
      previous shot across
- [x] Matrix: 69 runs, no failure ([docs/cuts.md](docs/cuts.md)). A missed cut costs the next
      shot's first frames: −2.4 to −9.6 dB PSNR-Y on its first frame (bright and dark cuts),
      −1.5 to −4.7 dB on the next three, VMAF −2 to −12, measurable for up to 13 frames; the
      clean cut loses far less. From k = 1 on, the previous shot's last frame suffers too (up to
      −2.3 dB, −16 VMAF). The causal VAE is what carries a shot over: a hard DiT window boundary
      at the cut with one VAE pass is the worst case (−7.9 / −16.8 dB, 9–11% ghost). A false
      cut costs no per-frame fidelity, only a temporal step. Very short shots are better alone
      than merged, from 1 frame on
- [-] Latent layouts at k = 2 and the optional "merged with the previous shot" short-shot runs
      (16 runs): not reached, the GPU was shared all night
- [~] Decision brief: shared with question 3, waiting for its labels

## 3. Scene detection

- [x] Every frame's scdet score, computed as sptenc runs it (its detections reproduced exactly),
      on 7 animated sources: 3 bright cartoon episodes, a dark anime episode, 25 min of a grainy
      cel film, 25 min of a clean digital anime film, a 720p anime episode. Live action when it arrives
- [x] Second opinion (PySceneDetect, frame-aligned with scdet); review sheets: 61 pages, 848
      candidates, a weighted sample by score band so partial labels still give estimates
- [ ] Review of the sheets by the user (waiting)
- [~] Per threshold 8–14: hits, false positives by kind (flash, pan, fade…), misses, shot lengths.
      Unlabelled so far: on action anime, scdet fires in bursts on new drawings after held frames
      and on effects (at threshold 10, half the shots of the dark anime episode are under 0.5 s),
      and misses some dark cuts; the bright cartoon is clean
- [ ] Doc + decision brief: the threshold and a minimum shot length, with question 2's costs

## 4. Frame-exact access into long-GOP sources (CPU only)

- [x] Reference decode per file (framemd5 + pts of every frame); targets including open-GOP
      leading frames: two H.264 Blu-rays, an H.264 web episode, HEVC (a long-GOP master and a
      720p segment), plus synthetic open-GOP x264, x265 and MPEG-2 excerpts in MKV, MP4, TS and VOB
- [x] H.264 (Blu-ray, web), HEVC, MPEG-2: accurate seek at n/fps and at the exact pts vs
      decode-and-count: exactness and time. Decode-and-count is always exact (1.3–17 min per
      2-hour film); no fixed seek recipe is: exact-pts seek lands late on MP4, TS and VOB, and on
      one open-GOP Blu-ray returns wrong pictures that carry the right pts (keyframes without
      their parameter sets). Recommended: a first-pass index of pts and picture hashes, seek to a
      keyframe at least 1 GOP and 1 s early, select by pts, verify the hashes, fall back
      (≈ 0.15 s per access on Blu-ray, ≈ 1 s on long-GOP HEVC)
- [ ] Real DVD (VOB), broadcast TS and open-GOP HEVC files (waiting for samples)
- [x] Doc ([docs/seeking.md](docs/seeking.md)) + decision brief (2026-10-02)

## 5. Decode resume granularity

- [x] Standalone VAE script (`decode_resume.py`, numz's own encode/decode path): one-pass decode
      vs resumed decodes with 1–40 warm-up latents, untiled and tiled, 720p and 1080p.
      **Bit-identical from 37 warm-up latents on**, exactly the decoder's reach read from the code
      (at 36, two frames still differ, 87 dB); deterministic across processes; a resume must use
      the original tiling. Warm-up cost at 1080p ≈ 4.7 min per resume (148 frames decoded again).
      Exact alternative: snapshot the 33 causal-conv caches, 8.05 GiB per output megapixel (18 GB
      at 1080p), ≈ 13 s to save and 5 s to restore
- [x] Doc ([docs/decode-resume.md](docs/decode-resume.md)) + decision brief (2026-10-02)

## 6. 4K and long windows

- [x] DiT probe (`dit_probe.py`, numz's own Phase 2 path, random latents) at 4K, L = 1…19
      latents: L = 20 runs out of memory on 96 GB (L = 19 passes with 2.5 GiB spare). Peaks fit
      **15.87 GiB + 127.2 KiB per token + 2.72 MiB per attention window** with no residual over
      12 points (the text tokens are repeated in every window); the per-token model alone is up
      to 0.8 GiB low at 4K. Time per token stays constant (within 5%) from 1 to 19 latents
- [ ] DiT probe at 1080p, long windows up to the memory limit (≈ 50 min of GPU, ready)
- [ ] Tiled VAE at 4K: peaks against the formulas, time per frame, 3 tile sizes (≈ 25 min, ready)
- [x] Planner margin, by bisection on the emulated card size: runs fail with 0.37 GiB between the
      free memory seen and the phase's peak (decode-bound), 0.14 (encode-bound), 0.15
      (DiT-bound); they pass at 0.45 / 0.24 / 0.23. A fourth configuration (DiT without swap,
      ≈ 20 min) is optional
- [ ] Doc + decision brief (brief on the margin and the 4K DiT model given 2026-10-03)

## 7. Optional: models and power cap

- [ ] 7B fp8, Q4_K_M, 3B fp16 (and 3B fp8) through question 1's protocol
- [ ] Power cap: power, clock and throttle sampling with and without BlockSwap; idle-pause control
- [ ] Doc + decision brief

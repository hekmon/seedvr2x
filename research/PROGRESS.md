# Measurement campaign: progress

Checklist of the measurements that answer the open questions of
[seedvr2x/DESIGN.md](../seedvr2x/DESIGN.md#open-questions), in priority order. Each question ends
with a decision brief (recommendation, evidence, confidence, caveats) for the design
conversation. Methods and results go in [docs/](docs/), scripts in [scripts/](scripts/).

`[x]` done · `[~]` in progress · `[ ]` to do · `[-]` dropped (reason given)

Last update: 2026-10-06 10:15 CEST

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
- [~] Samples from the user: two live-action Blu-rays (film grain 1.65 and 1.02, the grainy cel
      film 1.91), a bright anime episode, an NTSC DVD episode (16:9 anamorphic, declared interlaced;
      an inverse telecine leaves half the frames combed, so interlaced video), the original of the
      dark anime episode; a 4K UHD Blu-ray remux (HEVC, HDR10 + Dolby Vision 7, closed GOPs), a
      second 4K remaster awaited. No broadcast TS,
      no clean digital live action yet. Running on them: full-reference clips (question 1), cuts
      (question 2), scdet passes (question 3), seek tests (question 4)

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
- [x] A guaranteed reflected margin before the black rows (`NUM_PAD=reflect>=8+black+16`,
      identical to the measured mode at 1080p): at 720p better than numz's default on all 4 clips
      (PSNR-Y +0.07 to +1.03 dB, VMAF +0.9 to +10.2) and the bottom band repaired against black
      alone (+3.8 to +8.5 dB). Recommended at every size
- [x] The two ByteDance differences left: DiT norm output precision and VAE decode autocast
      within the seed band, and everything together too (`parity2`); the decode autocast costs
      decode memory (37.7 vs 35.3 GiB allocated, 56.5 vs 45.5 GiB device peak at 1080p)
- [x] VAE decode precision (from the design conversation): numz's masters hold 8-bit steps from
      0.5 up (Phase 4 normalises in bf16), but **no banding in any variant**, even on a new clip
      with large bright smooth gradients (CAMBI ≤ 0.004; it scores 0.70 on a clean 8-bit ramp).
      fp16 and fp32 decodes bring colour slightly closer (ΔE00 −0.05 to −0.09 on 5 of 5 clips);
      fp16 never overflowed (largest activation 3.4× under its limit), takes bf16's memory and 3–5%
      more decode time;
      fp32 costs 2× the memory and 1.4–4× the time. Keeping everything after the decode in fp32
      is free
- [x] Forced fp16 attention (GPUs without bf16): within the seed band on all 4 clips
- [x] Second degradation (area, CRF 26) on 2 clips: same picture as d1; reflect then black
      +0.26 dB; the sharper kernels gain a little on the softer input (+0.16 to +0.26 dB),
      beyond the spread on the grainy clip only
- [x] Reflect then black on the letterboxed clip: +0.29 dB PSNR-Y, letterbox kept black
- [x] Three more clips (two grainy live-action films, a bright anime; 27 runs). On live action
      the perceptual metrics favour the model over bicubic (LPIPS 0.173 vs 0.266, DISTS 0.087 vs
      0.107), PSNR, VMAF, colour and temporal error still bicubic. ByteDance's numerics as a whole
      bring slightly better low-frequency colour on all 3 (ΔE00 −0.05 to −0.09), as the fp16 decode
      does on all 8 clips. Kernels: antialias off worse on 7 of 7 clips; the zimg kernels cost
      colour and flicker on the fast live-action clip: keep numz's bicubic. Padding confirmed. The
      model restores about half of the source's film grain (1.75 → 0.93–0.97 on the fast clip;
      the grain measure corrected on 2026-10-05: frame edges left out)
- [x] The combined choice, checked once: numz's numerics + bicubic with antialias + reflect then
      black is the `reflect+black+16` run above
- [x] The default `yuv420p10le` master's chroma kernel (from the design conversation, CPU only,
      8 clips): zscale's bilinear keeps the model's output the closest to the ground truth of 6
      kernels on 8 of 8 clips (sharper kernels −0.3 to −0.5 dB on Cb/Cr, more ringing); the 4:2:0
      round trip even brings it closer than the unconverted output; the usual metrics don't move.
      Found on the way: zscale's slice threading (ffmpeg's default, one slice per CPU) changes the
      master's chroma (1.7% of the samples, up to 13 codes) and a 10-bit source's read; zscale's
      `threads=1` (libavfilter's generic per-filter option) fixes it, now set in ffv1_out.py and
      fr_clips.py (earlier data unaffected)
- [x] 4K full-reference clips (2026-10-05, CPU): 4 shots of 45 frames from the 4K remaster, a
      16-bit RGB 3840×2160 ground truth and 1920×1080 d1 inputs; seedvr2x refuses HDR, so the
      ground truth is an SDR rendition through one fixed tone map (mobius; hable was too dark).
      For colour's 4K tile study, and for scoring 4K output later. Their bicubic baselines (CPU):
      PSNR-Y 35.8–41.0 dB, VMAF 74–85 with sptenc's 2160p model, which `fr_metrics.py` now picks
      from 2160 rows up as sptenc does (it used the 1080p one at every size). VMAF v1 alone from now
      on: v0's NEG models are dropped, NEG being built into v1 (VMAF unchanged without them)
- [x] Padding at 4K (2026-10-05, 12 GPU runs, ≈ 1.2 h): the four 4K clips cropped inside their
      letterbox to 3840×2048 (no black in the frame, no numz padding), 25 frames, default (2
      seeds) against `reflect>=8+black+16`. The bottom band's repair carries over (+1.9 to +4.8
      dB, 4 of 4); 720p's frame-wide gain doesn't (rest of the frame −0.94 and −0.38 dB on 2
      clips, within the seed band on 2; low-frequency colour slightly worse on 3, LPIPS better on
      3, VMAF better on 2). Also seen: at 4K the model is further from the ground truth than
      bicubic on every metric, DISTS included, without colour correction
- [x] VMAF's 2160p model for every picture 3840 columns wide (2026-10-06, the user's rule):
      sptenc picks its model by the height alone, which gave 4K pictures cropped inside their bars
      the 1080p model. The padding check's crops are rescored: the model's VMAF is 3.5–4 points
      lower (34–44 against bicubic's 74–86), every other metric and every verdict the same
- [x] More 4K clips (2026-10-06, CPU): 13 shots from three more sources, so that the 4K verdict
      can be read by grain and by kind. Five come from clean digital live action (IMAX scenes,
      cropped inside the picture to 3840×2016). Four come from *Sol Levante*, clean animation
      made natively in 4K (its SDR ProRes master, CC BY 4.0). Four come from a 4K scan of a
      grainy cel-animated film (cropped to 3840×2048). Each is frame-exact and cut-free, with
      its bicubic baseline (PSNR-Y 37.2–47.4 dB). New `fr_clips.py make --src-crop`
- [x] 4K output with colour correction against bicubic (2026-10-06, CPU): colour's default 4K
      runs on the first film (none, lab, its split:ycc:4:3; 2 seeds), scored on the picture's
      rows and paired with bicubic. The correction closes the colour gap: split:ycc:4:3 brings
      ΔE00 lf to bicubic's level. Everything else stays behind bicubic: PSNR-Y 3.0–8.6 dB lower,
      VMAF 57–59 against 75–85, DISTS worse on 4 of 4, LPIPS on 3 (level on the close-up),
      flicker on 4. The model adds less banding than bicubic. split:ycc:4:3 beats lab on 4 of 4
      (PSNR-Y, VMAF, colour, flicker)
- [ ] The same on the 13 new clips, once the colour study has run its default 4K runs on them
- [x] Every fourth frame is the model's best: the last frame of each 4-frame latent group is
      PSNR-Y +0.8 to +5.0 dB and VMAF +2 to +19 closer to the ground truth than the group's
      second, on every clip, with or without `lab`; the input has no such pattern. Fast motion
      shows ghosts inside a group (from question 1's per-frame scores)
- [x] Doc ([docs/numerics.md](docs/numerics.md)) + decision briefs (numerics and input preparation,
      2026-10-03)

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
- [x] Latent layouts at k = 2 and the "merged with the previous shot" short-shot runs (16 runs):
      windows straddling a cut still cost about what one batch does, a window boundary at the
      cut's latent is still the worst. A short shot merged into the previous one is worse than run
      alone at every length (10–21 dB·frames), and at 1–2 frames worse than a missed cut (7–20%
      ghost): numz pads a batch to 4n + 1 frames by mirroring its end, which reaches back across
      the cut
- [x] A shot's first frame (from the design conversation): by the metrics, frame 0 is already
      the frame closest to the ground truth (the model re-renders it least) on 8 of 8 shots;
      prepending 4 mirrored frames gives up 0.6–2.8 dB of that lead on 6 of 8 to match the next
      frames' sharpness (deficit −47% → −15%), P = 8 does no better; ≈ 13 s per shot, +2–8% of
      GPU time per hour. Whether the sharpness step after a cut shows: for the user's eyes (review
      clips ready)
- [x] Three more cuts (bright anime, fast and slow live action; 21 runs): the conclusions hold.
      A miss costs 5–31 dB·frames over the next shot's first 8 frames on 4 of 6 cuts (about 0 on
      the clean anime and colour cuts), up to 12% ghost in its first frame, measurable for up to 17
      frames; prepending loses 0.6–2.8 dB of frame 0's lead on 7 of 11 shots
- [x] Decision brief (2026-10-05), joint with question 3
      ([docs/scene-detection.md](docs/scene-detection.md#decision-brief))

## 3. Scene detection

- [x] Every frame's scdet score, computed as sptenc runs it (its detections reproduced exactly),
      on 7 animated sources: 3 bright cartoon episodes, a dark anime episode, 25 min of a grainy
      cel film, 25 min of a clean digital anime film, a 720p anime episode. Live action when it arrives
- [x] Second opinion (PySceneDetect, frame-aligned with scdet); review sheets: 61 pages, 848
      candidates, a weighted sample by score band so partial labels still give estimates
- [x] New sources: the original of the dark anime episode (identical scores to its segments,
      2 frames apart), a bright anime episode, 25 minutes of each live-action film, the DVD
      episode (for the record). Live action shows no bursts, but scdet misses about half the cuts
      PySceneDetect finds there (dark, low-contrast shots); the bright anime's very short shots
      are mostly real (text-card flash cuts). 45 more review pages
- [x] Pipeline finding: a filter-graph rebuild mid-stream (the DVD's colour description appears at
      frame 14) restarts the frame counter sptenc reads cut indices from, so every later cut comes
      out 14 frames early; `-reinit_filter 0` fixes it
- [x] TransNetV2, the neural shot-boundary detector DESIGN.md lists (official MIT code and weights;
      its PyTorch port matches TensorFlow within 3e-7 on real frames, same detections; 5.1e-7 on
      2026-10-05's excerpts): every
      episode scored. It marks a cut on the outgoing shot's last frame (on 98% of the sure cuts).
      Unlabelled so far: on action anime it hardly fires in bursts (dark anime episode at p = 0.5
      against scdet at T = 10: detections 1–3 frames apart 627 → 5, shots under 0.5 s 782 → 18);
      on live action it takes almost every cut PySceneDetect finds below scdet's threshold (185 of
      191 where both of PySceneDetect's detectors agree, on one film); it misses 5 of the 15 sure
      cuts of the grainy cel film. 258 more review rows: the candidates only TransNetV2 finds, at
      most 30 per episode
- [x] Costs measured back to back (2026-10-05, one 1080p Blu-ray episode, 16 threads, two runs):
      per hour of source, decoding alone 1.96 min, scdet 1.98, PySceneDetect 2.28, TransNetV2
      4.30 (half of it the network); over the decode seedvr2x already makes for its frame index,
      scdet costs about nothing. All three deterministic; TransNetV2's port checked against
      TensorFlow again, kept this time (5.1 × 10⁻⁷ at most, same detections)
- [x] Labels, round 1 (2026-10-05): 100 of the review's 1,734 rows, drawn by agreement group and
      kind, shown blind, labelled by the user. The DVD's 20 left out: its thumbnails sat 14 frames
      late (a rebuilt filter graph restarted the trim filter's count; fixed in `scd_review.py`).
      On the 80 others: live action, TransNetV2 at 0.5 finds 95% of the cuts and scdet at 10 24%,
      at the same precision; animation, the same recall (0.80 against 0.84) with precision 0.88
      against 0.69 (scdet's bursts); filtering scdet's bursts drops recall to 0.64
- [~] Per threshold 8–14: hits, false positives by kind (flash, pan, fade…), misses, shot lengths.
      Unlabelled so far: on action anime, scdet fires in bursts on new drawings after held frames
      and on effects (at threshold 10, half the shots of the dark anime episode are under 0.5 s),
      and misses some dark cuts; the bright cartoon is clean
- [x] Labels, round 2 (2026-10-05, `scd_review.py round --plan 2`): 50 more rows in refined cells
      (TransNetV2's band, the picture's change where it alone fires), labelled. Both rounds pooled
      (130 rows): TransNetV2 at 0.3 finds 90% of the animated cuts (scdet at 10: 84%, precision
      0.82 against 0.62) and 98% of the live-action ones (scdet: 25%); 0.3 confirmed over 0.5; no
      picture-change gate (MAFD >= 2 gains 0.02-0.03 precision, a 0.14 margin to real cuts); the
      blind spot: 14% of scdet's lone hits inside bursts are real cuts, about 60-70 per hour of
      action anime. Addendum to the brief written
- [x] Doc + decision brief ([docs/scene-detection.md](docs/scene-detection.md), 2026-10-05):
      TransNetV2 at p = 0.3, no burst handling, no minimum shot length
- [x] Native 4K animation (2026-10-06, *Sol Levante*): inside a fast camera flight, two real cuts
      that every detector missed. The user found them; TransNetV2 scored 0.21 and 0.16, scdet 1.8
      and 1.3. A one-shot stretch in the same flight scored higher (0.23), so no threshold
      separates them there. The user proposes setting the threshold at the plan stage

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
- [x] Five more real sources: a DVD episode (MPEG-2 in MKV, closed GOPs), two anime and two
      film Blu-rays. Seeking one GOP early and checking by hash still holds everywhere; three
      additions: flag frames the decoder reports errors on (damaged frames only match from the
      start), count frames in the reader, never with ffmpeg's `select n` (a filter-graph rebuild
      restarts it: 14 frames late on the DVD), and take output timing from the source's
      timestamps, never the declared rate (a film declared 24 fps runs at 23.976 by its
      timestamps, which match its audio: 5.9 s apart at the end). Invalid entry points on 2 of 7
      real H.264 sources, so the content check is required
- [x] Real open-GOP HEVC in MKV (2026-10-05, an x265 encode the user made: one IDR, 435 CRA, 54
      RASL pictures): every method exact on every target, all 54 leading pictures included; a
      decode started at a CRA drops its RASL pictures silently, never shows a wrong one. The first
      4K remaster has closed GOPs (an IDR every 24 frames) and decodes at 204 fps on 16 threads
- [ ] Real VOB, broadcast TS, open-GOP MPEG-2, open-GOP HEVC in MP4 and TS
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

- [x] DiT probe (`dit_probe.py`, numz's own Phase 2 path), 26 window lengths (1080p 1–78 latents,
      4K 1–19): peak = **15.87 GiB + 127.16 KiB per token + 2.726 MiB per attention window**
      (the text tokens repeated in every window), residual ≤ 0.005 GiB; the per-token model alone
      is off by −0.17 to +0.83 GiB. Time per token constant within ±5%. Real 96 GB card: 78
      latents at 1080p (309 frames), 19 at 4K (73 frames)
- [x] Tiled VAE at 4K (tiles 1024, 1536, 2048): vram.md's fits hold to ±0.45 GiB up to 1536 and
      fall short at 2048 (new fits for tiles ≥ 1024); time per frame doesn't depend on the tile
      size (5.3–5.6 s encode, 11.5–12.3 s decode per 4K frame)
- [x] Planner margin, by bisection on the emulated card size, 4 configurations: runs fail with up
      to 0.37 GiB between the free memory seen and the phase's peak (decode-bound) and pass from
      0.23–0.45; **0.6 GiB** keeps 0.23 GiB over the worst failure, and gives back the validated
      "N − 2 GiB" rule
- [x] Doc ([docs/planner-limits.md](docs/planner-limits.md)) + decision brief (2026-10-03)

## 7. Optional: models and power cap

- [x] 7B fp8, Q4_K_M, 3B fp16, 3B fp8 and the "sharp" 7B fp16 through question 1's protocol (36
      runs, [docs/models.md](docs/models.md)). Q4_K_M as close to the source as 7B fp16 or closer
      (PSNR-Y +0.11 to +0.49 dB); fp8 slightly but consistently further (−0.25 to −0.58 dB, VMAF
      −1.4 to −3.0) for 2% speed; 3B fp16 perceptually worse on every clip (LPIPS, DISTS); the
      sharp 7B the closest to the source of all (+0.40 to +0.67 dB on every clip, beyond the seed
      ranges), and not sharper
- [x] Power cap: refuted. Idle pauses as long as the moves cost their full duration; the GPU sat at
      577 MHz with the power cap active throughout. Q4_K_M swap 36 really costs +0.30 s per batch
      at 1080p batch 5 (+3.6%): from the second batch on, the copies back to the CPU run 4.4 times
      faster (host memory reused)
- [x] Doc + decision brief (2026-10-03)

## Other

- [x] numz's `lab` and tied values (from the design conversation): its unstable sort gives tied
      pixels different reference values (memory order on the GPU), but on real frames numz's `lab`
      and a tie-aware `lab` differ by at most 0.06 a*/b* units per pixel, with every metric within
      the seed band: not a bug worth filing; noted in [docs/quality.md](docs/quality.md)
- [x] numz bug candidates from the implementation, checked against the code: new
      [bugs 24](bugs/24-rope-wrapper-late-binding.md) (the RoPE wrapper's late-binding closure:
      every block uses the last block's tables, cache bypassed) and
      [25](bugs/25-naditupscaler-undefined-attention-mode.md) (an unbuildable DiT class);
      bugs 18, 21 and 22 amended (fallback details, the import-time CUDA context, unchecked
      checkpoint keys). 25 bugs in all

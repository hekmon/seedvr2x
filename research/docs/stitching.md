# Batch stitching: boundaries, overlap and cross-fades

> Status: **measured** on the reference stack (7B fp16, `flash_attn_2`, 1080p, `--color_correction
> lab` unless noted), SeedVR2 `4490bd1`. The corrected cross-fades and the latent-space variant
> run through [`scripts/blend_patch.py`](../scripts/blend_patch.py), a `--wrap` patch; the metrics
> come from [`scripts/stitch_metrics.py`](../scripts/stitch_metrics.py). Builds on
> [quality.md](quality.md#batches-and-temporal-consistency), which found the boundary jumps.

In short (1080p restoration of anime, 7B fp16, batch 21; the jump is the extra frame-to-frame
change a boundary adds on held drawings over a single-batch reference):

- **Every batch boundary is a jump** (0.78 on top of an in-batch flicker of ≈ 1.0): each batch
  is encoded, upscaled and decoded on its own. One batch per shot has none, and is the target.
- **The CLI's `--temporal_overlap` hardly blends** ([bug 06](../bugs/06-temporal-overlap-blend-weights.md)):
  K = 2 and 4 are hard switches, 3 and 8 mix one or two frames (−38% and −44%).
- **A linear ramp over every overlap frame works:** K = 4 cuts the jump by 67% (0.25) and the
  low-frequency jump by 80%, for +16% compute; K = 2 already −56% for +6%; K = 8 adds nothing.
  The cost is softness: −15 to −20% Laplacian variance on the mixed frames. Context frames
  without mixing (what `--chunk_size` does) remove only a quarter of the jump.
- **Stitching in latent space is better and cheaper:** one VAE pass for the whole clip, the DiT
  on overlapping latent windows mixed before decoding. Sharing 2 latents (8 frames): jump 0.15
  (−80%), no softening, output within 40.8 dB of the single batch (37 dB for pixel batches),
  for +9% compute (the VAE, 77% of the time, runs once).
- **Today:** shot-aligned runs with one batch per shot; otherwise `--temporal_overlap 4` with
  [`blend_patch.py`](../scripts/blend_patch.py) `STITCH_CURVE=linear`, or `--temporal_overlap 3`
  without it. **For seedvr2x:** scene-aligned, one window per shot when it fits, latent-space
  stitching with 2 shared latents otherwise ([Recommendations](#recommendations)).

## Why there are boundaries

How the CLI cuts a clip (`src/core/generation_phases.py`):

- **Layout.** Batch k starts at k × (batch − K), K = `--temporal_overlap` (`:271`, `:349-358`); a
  last batch no longer than K is dropped, and every batch is padded to 4n + 1 frames with
  mirrored frames (`:397-402`) because the VAE compresses time by 4 (latent 0 = frame 0 alone,
  latent j = frames 4j − 3 … 4j).
- **Each batch is independent.** Phase 1 encodes it in its own causal VAE pass, so its first
  frame is a lone latent and comes out less restored ([quality.md](quality.md#--prepend_frames));
  Phase 2 reseeds before every batch (`:663`), so every batch gets the same diffusion noise *by
  position in the batch*, not by frame (an overlap frame gets different noise in its two
  batches); Phase 3 decodes it in its own causal pass. Nothing is shared but the seed.
- **The overlap is blended in pixels, after decoding** (`:971-997`): the first K decoded frames
  of batch k are mixed with the last K frames already written, with `blend_overlapping_frames`
  (`src/core/generation_utils.py:284`). Its weights only ramp over the middle third and start
  and end at exactly 1 and 0 ([bug 06](../bugs/06-temporal-overlap-blend-weights.md)): K = 1, 2,
  4 never mix, 3 and 5 mix one frame 50/50, 8 mixes two.
- **Colour correction comes after the blend** (Phase 4, `:1254-1262`), per write slice: the
  blended frames belong to the previous batch's slice, and `lab` pools its a\*/b\* histograms
  over that slice, so the mix also nudges the rest of that batch slightly (≤ 3 levels measured).
- **ByteDance's original has no batching at all:** `projects/inference_seedvr2_7b.py` runs the
  whole clip as one sequence, split across GPUs by sequence parallelism if needed. Batches,
  overlap and blending are the CLI's addition, to bound memory and per-call time.
- **The VAE already streams:** inside a batch it encodes frame 0, then 4-frame slices, and
  decodes one latent per slice, carrying the causal convolutions' caches from slice to slice
  (`attn_video_vae.py:1254-1290`, [vram.md](vram.md#how-the-vae-processes-a-batch)). A batch
  boundary in the VAE is only a cache reset; a whole shot can be encoded or decoded in one pass
  at the memory cost of one slice.

## Method

Clips from [quality.md](quality.md#method), 1920×1080 in and out (restoration, no upscale):

| Clip | Frames | Content | Reference (no boundary) |
|---|---|---|---|
| B | 81, from frame 20 of `seg_0001-0004` | Anime on threes: two of every three transitions are held drawings, so any output change there is flicker. Dark | one batch of 81 |
| A | 45, from frame 48 of `seg_0000` | Fast motion and pans | one batch of 45 |

Stitched runs use `--batch_size 21`. Each run is compared, frame by frame, with the same clip
computed as a single batch, which has no boundary at all: the ideal a stitched output should
look like.

**Curves.** [`blend_patch.py`](../scripts/blend_patch.py) replaces `blend_overlapping_frames`
at import and, since the weights only change how the decoded overlap frames are mixed, renders
several curves from one run: Phase 4 runs again on a copy of the Phase 3 frames with the overlap
re-mixed (identical to separate runs, the CLI being deterministic). Weight of the previous batch
on overlap frame i = 1 … K:

| Curve | Weights | K = 4 |
|---|---|---|
| `numz` | the CLI's (bug 06) | 1, 1, 0, 0 |
| `linear` | 1 − i / (K + 1), end points excluded | 0.8, 0.6, 0.4, 0.2 |
| `cosine` | ½ + ½ cos(π i / (K + 1)), end points excluded | 0.90, 0.65, 0.35, 0.10 |
| `prev` | 1: the new batch's first K frames are context only, hard switch at s + K | 1, 1, 1, 1 |
| `cur` | 0: the previous batch's last K frames are context only, hard switch at s | 0, 0, 0, 0 |

`prev` and `cur` separate the two effects of an overlap: giving a batch context on one side,
and mixing.

**Metrics** ([`stitch_metrics.py`](../scripts/stitch_metrics.py), on the CLI's PNGs). The
*zone* of a boundary at batch start s is the K + 1 transitions s … s + K where the frame source
changes (transition t is between frames t − 1 and t). Per transition, on the definitions of
[quality.md](quality.md#method):

| Metric | What it measures |
|---|---|
| Hold ΔY | Output mean \|ΔY\| on held drawings (clip B): flicker. Inside batches (outside every zone), and the **step**: each boundary's largest zone value, averaged over boundaries, what the eye catches |
| Added lf | Mean \|Δ(output − input)\| on 16×16 block means, RGB, every transition: low-frequency brightness/colour change the input doesn't have |
| Excess | Output minus the single-batch reference, transition by transition: the change the boundary adds (≈ 0 inside batches). **Excess step** = the largest per boundary; **excess total** = its sum over the zone, the whole jump however it is spread |
| PSNR vs ref | Per frame, against the single-batch output: per batch interior (≥ 4 frames from any zone) and by offset to the boundary |
| Cost | Frames computed (each batch padded to 4n + 1) / frames output |

The reference itself, at the same transitions, gives the clip's own change there (a drawing
change on clip B, motion on clip A), which is why the excess, not the raw step, is the measure
of the boundary.

## Pixel-space results (clip B)

Clip B, 81 frames, batch 21, `lab`. One run per overlap K (its own batch layout), every curve
rendered from it. **Hold excess step**: the largest extra frame-to-frame change on held drawings
that a boundary adds over the single-batch reference, averaged over the boundaries (inside
batches the output already flickers at ≈ 1.0 per transition, input codec noise ≈ 0.45);
**lf**: the same on the low-frequency measure; **Lap var, mixed**: sharpness of the mixed frames
relative to the frames away from boundaries (1 = as sharp):

| K | Cost | Hold excess step: `numz` / `prev` / `cur` / `cosine` / `linear` | Hold excess total, `cur` → `linear` | Lf excess step: `numz` / `cosine` / `linear` | Lap var, mixed: `cosine` / `linear` |
|---|---|---|---|---|---|
| 0 | 1.04 | 0.78 (no overlap) | 0.78 | 0.31 | – |
| 2 | 1.10 | 0.79 / 0.81 / 0.63 / **0.34** / **0.34** | 0.63 → 0.40 | 0.44 / 0.15 / **0.11** | −17% / −20% |
| 3 | 1.15 | 0.48 / 1.00 / 1.05 / **0.38** / **0.37** | 1.73 → 0.61 | 0.17 / 0.13 / 0.13 | −12% / −16% |
| 4 | 1.20 | 0.72 / 0.58 / 0.74 / **0.32** / **0.25** | 0.79 → 0.08 | 0.29 / 0.09 / **0.06** | −15% / −19% |
| 8 | 1.51 | 0.44 / 0.50 / 0.52 / **0.30** / **0.25** | 0.68 → −0.07 | 0.14 / 0.11 / **0.08** | −11% / −14% |

(Per-run figures: in the `stitch_metrics.py --summary` tables of the runs listed in
[Reproduce](#reproduce). With 3–5 boundaries per run the hard-switch baseline varies between
layouts, 0.63–1.05: compare curves within a row first.)

- **The CLI's weights barely help:** `numz` equals a hard switch at K = 2 and 4 (bug 06);
  K = 3 (one frame 50/50) and K = 8 (two mixed frames) cut the step by 38% and 44%, against 53%
  and 68% for a linear ramp at the same K.
- **A ramp over every overlap frame removes most of the jump:** `linear` K = 4 brings the excess
  step from 0.78 to 0.25 (−67%) and the low-frequency step from 0.31 to 0.06 (−80%); the held
  step is then 1.12 against 1.06 for the reference at the same transitions, i.e. close to the
  ordinary in-batch flicker. The total added change over the zone drops to ≈ 0: the jump is not
  just spread, it disappears into the in-batch flicker.
- **`linear` ≥ `cosine`:** at equal K the raised cosine puts a steeper slope in the middle (π/2 ×
  the linear one), so its largest step is larger. The difference is small (0.25 vs 0.32 at K = 4).
- **Diminishing returns past 4 frames:** K = 2 already gets 0.34 for +6% compute over no overlap;
  K = 4 gets 0.25 for +16%; K = 8 nothing more for +45%.
- **Context alone (`prev`, `cur`) does little:** a hard switch with 2–8 frames of context on one
  side stays at 0.5–1.0. Giving a batch context is not what smooths a boundary; mixing is. This
  is what a `--chunk_size` boundary does today ([bug 07](../bugs/07-chunk-overlap-not-blended.md)).
- **The cost of mixing is softness:** a mixed frame averages two renderings whose fine detail
  disagrees, so it loses 15–20% of Laplacian variance (`linear` a bit more than `cosine`, which
  keeps the end frames closer to one rendering). Every boundary gets a few slightly softer frames.
  Both renderings are of the same input frame, so there is no ghosting, even on motion (clip A
  below). Whether the softer frames are visible is for the [visual review](#visual-review).
- **Distance to the single-batch output** (PSNR vs reference): batch 1 shares the reference's
  start, encoder state and noise and stays at 41 dB; later batches are independent renderings at
  36–37 dB, slightly lower right after a hard boundary (35.5–36.3 dB on the first 4 frames of a
  new batch at K = 0: its first latent is encoded alone, with no past context). Mixed frames are
  *closer* to the reference (38–39.5 dB): the average of two renderings is nearer to any third.

### Clip A (motion) and `--color_correction none`

Clip A has no held drawings, and the motion dominates the low-frequency measure (it is noisy:
two boundaries, 45 frames). At K = 4, batch 21, `lab`:

| Curve | Lf excess step | Lap var, mixed / inner (vs ref) | PSNR vs ref, mixed frames / next batch |
|---|---|---|---|
| `cur` (hard switch) | 0.71 | – | 35.3 dB / 35.1 dB |
| `numz` | 0.74 | – | 38.1–38.8, then 35.6 dB |
| `cosine` | 0.47 | 1.02 / 1.03 | 38.5 → 36.4 dB |
| `linear` | **0.37** | 0.96 / 1.03 | 38.8 → 37.0 dB |

- On motion the ramp still halves the boundary's low-frequency step, and the mixed frames lose
  little sharpness (−6% `linear`, −1% `cosine`): moving content has less fine texture for two
  renderings to disagree on.
- **No ghosting:** the mixed frames are *closer* to the single-batch output (38.5–38.8 dB) than
  the unmixed frames of the new batch (35 dB). Both batches render the same input frame, so a
  mix doesn't superimpose two positions of a moving object.

At K = 8 on clip A, same order: `linear` 0.50 against 1.05 for the hard switch (`cur`).

**Without colour correction** (clip B, K = 4, reference `q-b-bs81`, the same clip as one batch
with `--color_correction none`): the ramp works the same way, `linear` 0.32 against 1.04 for the
hard switch on held frames (−69%), 0.18 against 0.75 on the low-frequency measure (−77%), with the
same softening (−19%). `lab` divides the remaining low-frequency step by about 3 (0.06), so the
two combine.

## Latent-space stitching

The CLI stitches in pixels, after decoding. Stitching in latent space instead, before decoding,
is possible and changes both the cost and the boundary:

- **One VAE pass per shot.** The VAE already streams a batch slice by slice with causal caches
  ([above](#why-there-are-boundaries)), so a whole shot can be encoded in one pass and decoded in
  one pass, at the memory of one slice (only the latents grow: 1 MiB per latent frame at 1080p,
  136 × 240 × 16 in bf16). No batch restarts the encoder (no lone first latent mid-shot) or the
  decoder.
- **The DiT runs on overlapping windows of latents**, which are cross-faded over the shared
  latents before decoding; the causal decoder then spreads the change over neighbouring frames.
- **Cost:** only the DiT repeats the shared latents; in pixel space every overlap frame is
  encoded, upscaled and decoded twice, and the VAE is most of the per-frame time
  ([cost](#cost-model)).
- **Alignment:** latent j covers frames 4j − 3 … 4j (latent 0: frame 0 alone), so windows must
  share whole latents (4-frame steps). A pixel-space overlap between two CLI batches only lines
  up with latents when batch − K is a multiple of 4.
- **Risks:** a window that starts mid-shot starts with a 4-frame latent, where in a normal batch
  (and presumably in training, on clips encoded from their first frame) the DiT always sees a
  1-frame latent 0.
  And the CLI gives every batch the same noise by position: two windows give a shared latent
  different noise. Drawing one noise tensor for the shot and slicing it per window would remove
  that difference (not tested; the seed itself barely matters, [quality.md](quality.md#seed)).
- The DiT's own attention is already windowed in time (`wt = ceil(min(t, 30) / 4)` latent
  frames, shifted every other layer, [attention.md](attention.md#windowed-attention)): inside a
  batch, the model stitches its temporal windows by itself through the shifts. A batch boundary
  is the one place where no shifted window crosses.

Measured with [`blend_patch.py`](../scripts/blend_patch.py)'s experimental `STITCH_LATENT=W:M`
(the CLI runs one batch for the whole clip; Phase 2 is split into windows of W latents sharing M,
merged with `cosine` weights before the single decode). Clip B, 81 frames = 21 latents, W = 6
(the DiT load of a 21-frame batch), `lab`; the pixel rows for comparison:

| Stitching | Compute: VAE / DiT | Hold excess step / total | Lf excess step | Lap var, mixed / inner (vs ref) | PSNR vs ref (min) |
|---|---|---|---|---|---|
| Pixel, no overlap (`ob-k0`) | 1.04 / 1.04 | 0.78 / 0.78 | 0.31 | – / 1.13 | 37.1 dB (34.8) |
| Pixel, K = 4 `linear` | 1.20 / 1.20 | 0.25 / 0.08 | 0.06 | 0.89 / 1.09 | 37.5 dB (33.3) |
| Pixel, K = 8 `linear` | 1.51 / 1.51 | 0.25 / −0.07 | 0.08 | 0.92 / 1.08 | 38.2 dB (34.9) |
| Latent, M = 0 (`ob-lat6m0`) | 1 / 1 | 0.71 / 0.75 | 0.19 | – / 1.07 | 39.0 dB (33.5) |
| Latent, M = 1, 4 frames (`ob-lat6m1`) | 1 / 1.14 | 0.25 / 0.44 | 0.05 | 1.08 / 1.08 | 40.5 dB (39.0) |
| Latent, M = 2, 8 frames (`ob-lat6m2`) | 1 / 1.38 | **0.15** / 0.21 | **0.03** | 1.07 / 1.09 | **40.8 dB (39.9)** |

(The latent zone is 4M + 5 transitions wide, 2 more on each side for the decoder's spread, so
its "total" sums more transitions than the pixel one. DiT factors are for this short clip; on a
long shot they tend to W / (W − M) = 1.2 and 1.5.)

- **A continuous VAE alone doesn't remove the boundary:** with hard DiT windows (M = 0) the
  step is 0.71, as with pixel batches. The first latent of a window is rendered from no past
  context and is the outlier (35.1 dB from the reference, against 40.5 for the rest of its
  window).
- **Sharing latents does:** M = 1 matches the best pixel cross-fade (0.25), M = 2 beats it (0.15,
  −80% vs no overlap) with the smallest low-frequency step (0.03).
- **No softening:** the mixed frames are as sharp as the rest (1.07–1.08 vs 1.08–1.09): the decoder
  renders mixed latents as a sharp image, where a pixel mix averages two textures (−15 to −20%).
- **The whole output stays close to the single-batch one:** 40.5–40.8 dB (worst frame 39–40 dB)
  against 37–38 dB for pixel batches. Most of the difference between independent pixel batches
  comes from the VAE restarts (each batch encoded and decoded alone), not from the DiT: with one
  VAE pass, even hard DiT windows stay at 40–41 dB inside the windows.
- **The 4-frame first latent of a window is not a problem in practice:** it only matters at M = 0,
  and with M ≥ 1 it is mixed away (weight ≤ 0.5).
- Compute: the VAE runs once per frame, so the overhead is the DiT's share (≈ 23% of the
  compute at batch 21) times the shared fraction: ≈ +3% (M = 1) and +9% (M = 2) on this clip,
  against +16% and +45% for pixel overlaps of 4 and 8 frames.


## Cost model

Phase times per computed frame, 1080p, 7B fp16, batch 21 (run `ob-k0`, 84 frames computed for
81 output): encode 1.02 s, DiT 1.00 s, decode 2.37 s. **The VAE is ≈ 77% of the compute.** One
batch of 81 (`ob-ref`): 0.96, 0.89 and 2.31 s. Compare times only between back-to-back runs
([benchmarking.md](benchmarking.md#caveats)).

| Stitching | What is computed twice | Compute factor | Batch 21 (or 6-latent windows) |
|---|---|---|---|
| CLI overlap K (pixel space) | K frames per boundary, through encode, DiT and decode | B / (B − K) | K = 2: 1.11; 4: 1.24; 8: 1.62 |
| Latent windows, M shared latents | M latents per boundary, through the DiT only | 0.77 + 0.23 × W / (W − M) | M = 1 (4 frames): 1.05; M = 2 (8 frames): 1.12 |
| One batch per shot | nothing | 1 (and the DiT is ≈ 10% faster per frame at batch 81) | – |

Measured totals for clip B's 81 frames, back to back (model loading and the extra curves'
rendering included): one batch 346 s; batch 21 without overlap 378 s, K = 2 454 s, K = 4 491 s,
K = 8 627 s; latent windows M = 0, 1, 2: 406, 406 and 419 s. The latent runs' single 81-frame
decode took 208–221 s against 187 s for the reference (power-cap drift, see
[benchmarking.md](benchmarking.md#caveats)); their DiT phase was 81, 90 and 107 s against 84 s
for batch 21 without overlap.

The 4n + 1 padding of a short last batch comes on top (here 81 frames at batch 21: 84 computed
without overlap); `--uniform_batch_size` pads it to the full batch instead.

## Recommendations

### The numz CLI today

1. **Avoid boundaries inside a shot.** Cut the video at scene changes first (a jump at a cut is
   invisible) and run each shot with `--batch_size` ≥ its length: one batch, no boundary, and the
   DiT is even ≈ 10% faster per frame. At 1080p a 96 GB card runs 81 frames untiled (DiT 37 GiB,
   decode 36 GiB) and more with VAE tiling ([vram.md](vram.md#practical-rules)); on smaller cards,
   use the largest batch the recipe allows.
2. **When a shot needs several batches:** split it into equal batches (no short last batch, or
   `--uniform_batch_size`), and stitch with **`--temporal_overlap 4` plus
   [`blend_patch.py`](../scripts/blend_patch.py) with `STITCH_CURVE=linear`** (a `--wrap`, no
   checkout change): −67% of the held-frame jump and −80% of the low-frequency one for +16%
   compute at batch 21, at the price of 4 slightly softer frames per boundary. `--temporal_overlap
   2` with `linear` is the cheap variant (−56%, +6%).
3. **Without the patch:** `--temporal_overlap 3` (one frame 50/50), the best the CLI's own
   weights do, −38% on this clip; 1, 2 and 4 only cost compute (bug 06).
4. Keep `--color_correction lab`: it already halves the low-frequency part of every jump
   ([quality.md](quality.md#colour-correction)).
5. Don't rely on `--chunk_size` to join long videos inside a shot: its boundaries are hard cuts
   ([bug 07](../bugs/07-chunk-overlap-not-blended.md)); context frames alone remove only a quarter
   of the jump.

### Our own CLI (seedvr2x)

1. **Scene-aligned processing.** Detect cuts, process shot by shot; never let a window straddle a
   cut (a cut resets everything anyway).
2. **One window per shot when it fits** — the ideal, measured as the reference here.
3. **Longer shots: stitch in latent space.** Encode the whole shot in one causal VAE pass
   (streaming, memory of one slice), run the DiT on overlapping windows of latent frames, mix
   the shared latents with a ramp, decode the merged latents in one causal pass. Measured with
   6-latent windows (the DiT load of a 21-frame batch): sharing 2 latents (8 frames) gives the
   smallest jumps of all configurations (held-frame step 0.15 vs 0.78 without overlap and 0.25
   for the best pixel cross-fade; low-frequency step 0.03), **no softening** (the decoder renders
   the mixed latents sharply) and an output 40.8 dB from the single-batch result instead of 37 dB.
   Cost: DiT × W / (W − M), VAE × 1: ≈ +12% total for M = 2 on long shots (+5% for M = 1, step
   0.25), against +24% for a 4-frame pixel overlap.
4. **Window sizing:** equal windows (shot latents split evenly, no short tail), W as large as
   memory allows (fewer boundaries; DiT throughput already 91% of the maximum at 6 latents),
   M = 2 latents (8 frames; M = 1 if compute is tight). The ramp shape matters little over 1–2
   latents (`cosine` was measured; with M = 2 it is 0.75 / 0.25, the same as `linear` would be up
   to 0.08). Draw the diffusion noise once per shot and slice it per window,
   so shared latents get the same noise in both windows (the CLI reseeds per batch).
5. **Pixel-space fallback** (if a model or path can't do latent windows): linear ramp, K = 4,
   end points excluded, and keep batch − K a multiple of 4 so that overlaps line up with latents.
6. **Colour correction once per shot** (or per window before the merge if it must pool): the
   `lab` histogram pooling per batch slightly shifts a batch when its overlap is re-mixed.

## Visual review

Every run also wrote lossless FFV1 masters ([output.md](output.md)): the CLI's own output as
`<run>/<run>.gbrp16le.mkv` (16-bit RGB) and `<run>/<run>.yuv420p10le.mkv` (10-bit 4:2:0, lighter
to play), and each extra curve as `<run>/extra/<curve>.yuv420p10le.mkv`. Frame indices are
0-based output frames (frame 0 = input frame 20 of `seg_0001-0004` on clip B, 48 of `seg_0000`
on clip A). In mpv, `.` and `,` step one frame; `--osd-msg1='${estimated-frame-number}'` shows
the 0-based index. Look at the held drawings (clip B) around each boundary, and for a softer
stretch on the mixed frames. Suggested order: `ob-ref`, `ob-k0`, `ob-k3` (today's best),
`ob-k4/extra/linear` (the patch), `ob-lat6m2` (latent, the seedvr2x design).

| Run | File(s) | New batch starts at | Mixed frames (weight of the previous batch) |
|---|---|---|---|
| `ob-ref` | the reference: one batch of 81 | – | – |
| `ob-k0` | no overlap (CLI default) | 21, 42, 63 | none: hard cut between 20 → 21, 41 → 42, 62 → 63 |
| `ob-k3` | CLI's `--temporal_overlap 3` (`numz`); extras `linear`, `cosine`, `prev`, `cur` | 18, 36, 54, 72 | `numz`: 19, 37, 55, 73 at 0.5 (18 is the old batch, 20 the new); `linear`: 18–20, … at 0.75, 0.5, 0.25 |
| `ob-k4` | `cosine`; extras `numz`, `linear`, `prev`, `cur` | 17, 34, 51, 68 | 17–20, 34–37, 51–54, 68–71 (`linear` 0.8 … 0.2; `numz` switches hard at 19) |
| `ob-k8` | `cosine`; extras `numz`, `linear`, `prev`, `cur` | 13, 26, 39, 52, 65 | 13–20, 26–33, 39–46, 52–59, 65–72 |
| `ob-k2` | `cosine`; extras `numz`, `linear`, `prev`, `cur` | 19, 38, 57, 76 | 19–20, 38–39, 57–58, 76–77 |
| `ob-lat6m0` | latent windows, none shared (control) | 21, 45, 69 (DiT windows; one VAE pass) | none |
| `ob-lat6m1` | latent windows, 1 shared latent, `cosine` (0.5) | 17, 37, 57 | 17–20, 37–40, 57–60 (mixed as latents) |
| `ob-lat6m2` | latent windows, 2 shared latents, `cosine` (0.75, 0.25) | 13, 29, 45, 61 | 13–20, 29–36, 45–52, 61–68 (mixed as latents) |
| `ob-k4-none` | as `ob-k4`, `--color_correction none` (`ob-k4-none.mkv`, 10-bit 4:2:0 only; extras as PNG only) | 17, 34, 51, 68 | 17–20, … |
| `oa-ref` | clip A reference: one batch of 45 | – | – |
| `oa-k0` | clip A, no overlap | 21, 42 | none |
| `oa-k4`, `oa-k8` | clip A, `cosine`; extras as `ob-k4` | 17, 34 / 13, 26 | 17–20, 34–37 / 13–20, 26–33 |

## Reproduce

```bash
# a stitched run: the CLI's output with one curve, the other curves rendered from the same run
python3 scripts/bench.py run ob-k4 --wrap scripts/blend_patch.py --wrap scripts/ffv1_out.py \
  --env STITCH_CURVE=cosine --env STITCH_EXTRA=numz,linear,prev,cur \
  --env STITCH_EXTRA_FFV1=yuv420p10le --env FFV1_OUT_PIXFMT=gbrp16le,yuv420p10le -- \
  seg_0001-0004.mp4 --output out/ob-k4/ --output_format png --model_dir /path/to/models \
  --dit_model seedvr2_ema_7b_fp16.safetensors --resolution 1080 --attention_mode flash_attn_2 \
  --skip_first_frames 20 --load_cap 81 --batch_size 21 --temporal_overlap 4 --color_correction lab
# the reference: the same clip as one batch (--batch_size 81 --temporal_overlap 0)
# latent-space stitching: one batch for the VAE, DiT on windows of 6 latents sharing 1
python3 scripts/bench.py run ob-lat6m1 --wrap scripts/blend_patch.py \
  --env STITCH_LATENT=6:1 --env STITCH_CURVE=cosine -- seg_0001-0004.mp4 ... \
  --load_cap 81 --batch_size 81 --temporal_overlap 0
# metrics against the reference; --latent 6:1 for a latent run
python3 scripts/stitch_metrics.py out/ob-k4/extra/linear --input seg_0001-0004.mp4 --skip 20 \
  --ref out/ob-ref/seg_0001-0004 --batch 21 --overlap 4 --curve linear --json m/ob-k4-linear.json
python3 scripts/stitch_metrics.py --summary m/*.json    # Markdown table
python3 scripts/stitch_metrics.py --profile m/*.json    # PSNR vs reference by offset to the boundary
```

<details>
<summary>Run names (7B fp16, <code>flash_attn_2</code>, 1080p, PNG + FFV1; clip B = 81 frames of
<code>seg_0001-0004</code> from frame 20, clip A = 45 frames of <code>seg_0000</code> from frame 48)</summary>

Clip B, `lab`: `ob-ref` (batch 81), `ob-k0`, `ob-k2`, `ob-k4`, `ob-k8` (main curve `cosine`,
extras `numz`, `linear`, `prev`, `cur`), `ob-k3` (main `numz`, extras `linear`, `cosine`, `prev`,
`cur`), latent `ob-lat6m0`, `ob-lat6m1`, `ob-lat6m2`; clip B, `none`: `ob-k4-none` (reference:
`q-b-bs81` from [quality.md](quality.md)). Clip A, `lab`: `oa-ref` (batch 45), `oa-k0`, `oa-k4`,
`oa-k8`. Metric labels: `<run>-<curve>`.

</details>

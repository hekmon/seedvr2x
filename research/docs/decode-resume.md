# Resuming a VAE decode mid-shot: warm-up latents and cache snapshots

> Status: **measured** with [`scripts/decode_resume.py`](../scripts/decode_resume.py) through
> numz's own runner, encode and decode functions (`4490bd1`, 7B config, bf16), for the open
> question "decode resume granularity" of [DESIGN.md](../../seedvr2x/DESIGN.md#to-measure). One
> clip, one GPU (RTX PRO 6000 Blackwell, torch 2.14.1+cu130, cuDNN 9.24). Timings are
> indicative: the GPU is power-capped and drifts between sessions.

In short (201 frames = 51 latents of a dark 1080p anime segment, at 720p and 1080p; a resume at
latent s with w warm-up latents decodes latents s − w … s + 2 from scratch, drops the frames
before latent s and is compared with the uninterrupted one-pass decode):

- **Bit-identical from w = 37, never before,** at 720p (s = 45 and 41), 720p tiled and 1080p. At
  w = 36 frames 1–2 of latent s still differ (0.1% of the values, 87 dB) and frames 3–4 are
  exact: the decoder's [receptive field](#the-decoders-temporal-reach), to the frame.
- **Quantising doesn't help:** 16- and 10-bit outputs differ wherever bf16 does (at w = 36, 96%
  and 59% of the differing values change code). Below 37: 22 dB (w = 1) to 87 dB (w = 36).
- **The decode is deterministic,** within a process and across processes.
- **A warm-up costs 37 latents (148 frames) decoded and thrown away per resume:** ≈ 4.7 min at
  1080p (7.8 s per latent), ≈ 2.1 min at 720p.
- **Snapshotting the causal-conv caches is exact too,** even through a file: 33 tensors, 8,641
  bytes per output pixel (8.05 GiB/Mpx), 18.05 GB at 1080p, saved in ≈ 13 s, restored in ≈ 5 s.
- **For seedvr2x:** a sub-segment resume can be bit-identical, with a warm-up of min(s, 37)
  latents (none at a shot's start) or a cache snapshot per resume point, and only with the same
  tiling, memory limits and slice size as the run it continues ([Caveats](#caveats)).

## Why it matters for seedvr2x

[DESIGN.md](../../seedvr2x/DESIGN.md#pause-and-resume) saves the VAE decode as output segments and
resumes per output segment ("its decode and write restart, the DiT latents are kept"), "finer,
per sub-segment with warm-up latents, if that proves bit-identical"; a resume must be bit-identical
to an uninterrupted run. A restarted decode is a fresh causal decode: its convolutions start from
copies of their first input frame, and its first latent gives 1 frame instead of 4. How many
latents must be decoded and dropped first, at what cost, and is saving the caches affordable?

## The decoder's temporal reach

From the code (`AVV` = `src/models/video_vae_v3/modules/attn_video_vae.py`, `CIL` =
`causal_inflation_lib.py`, `CPL` = `context_parallel_lib.py` next to it):

- **Slices** (`slicing_decode`, AVV:1278-1297): latents 0–1 first (state INITIALIZING, 5
  frames), then one latent per slice (ACTIVE, 4 frames). Latent j gives output frames
  4j − 3 … 4j (latent 0: frame 0).
- **Caches:** every `InflatedCausalConv3d` of temporal kernel 3 keeps the last 2 frames of its
  input for the next slice (CIL:250-278), dropped whenever the state isn't ACTIVE (CIL:219-220). A
  fresh start pads with 2 copies of the first input frame (CPL:62-65), and outside ACTIVE the
  temporal upsamplers drop the duplicated head (AVV:152-153). Nothing else reaches across
  frames: GroupNorm (CIL:354-408) and the mid-block attention (AVV:656-668) work per frame;
  shortcuts and `upscale_conv` are 1×1×1.

The VAE is built with `time_receptive_field="full"` (AVV:1087): both convs of every resnet have
temporal kernel 3. In decode order:

| Rate | Kernel-3 convs | Reach |
|---|---|---|
| latents | `conv_in`, mid block (2 resnets × 2), `up_blocks[0]` (3 × 2): 11 | 22 latents |
| 2× (after `up_blocks[0]`'s temporal upsampler) | its conv, `up_blocks[1]` (3 × 2): 7 | 14 frames (7 latents) |
| 4× = output (after `up_blocks[1]`'s) | its conv, `up_blocks[2]` (6), its spatial upsampler's conv, `up_blocks[3]` (6), `conv_out`: 15 | 30 frames (7.5 latents) |

- **Reach:** the first frame of latent s (f = 4s − 3) depends on output-rate frames ≥ f − 30,
  2× frames ≥ ⌈(f − 30)/2⌉ − 14 = 2s − 30, latents ≥ ⌈(2s − 30)/2⌉ − 22 = s − 37.
- **Restart:** a fresh decode from latent a = s − w differs at every conv's first 2 outputs, and
  each conv carries the difference 2 frames further: latents ≤ a + 21 after the latent-rate
  layers, 2× frames ≤ 2(a + 21) + 14 = 2a + 56, output frames ≤ 2(2a + 56) + 30 = 4a + 142.
  Latent s's frames 4a + 4w − 3 … 4a + 4w are all past that iff 4w − 3 > 142: **w ≥ 37**. At
  w = 36 they are frames 4a + 141 … 4a + 144: frames 1–2 inside, 3–4 outside.

## Method

- **Clip:** the first 201 frames (51 latents) of a dark 1080p anime segment (HEVC, 23.976 fps;
  the segment of [stitching.md](stitching.md#method)'s clip B), decoded by ffmpeg to RGB24 at
  1280×720 (Lanczos, BT.709 matrix) and at 1920×1080 (numz pads it to 1088).
- **Numz's own path, as `inference_cli.py` runs it:** `setup_generation_context` +
  `prepare_runner` (CLI defaults) + `materialize_model` build the VAE (fp16 weights cast to bf16,
  eval, 1-latent slices, caches on the GPU, memory limits 0.5 / 0.5 GiB, tensor offload to the
  CPU); `prepare_video_transforms` (checked: pad and normalise only) and `vae_encode` encode once
  (posterior mode × 0.9152); `vae_decode` decodes contiguous [T, H, W, C] bf16 latents, the DiT
  output's layout. Conv3d workaround active (`torch.cudnn_convolution`), `cudnn.benchmark` off.
- **Runs:** the one-pass reference (51 latents), twice in one process, saved and reloaded by
  later processes. Warm-up resume: `vae_decode` of latents [s − w, s + 3), its frames from 4w − 3
  on against the reference's 4s − 3 … 4s + 8; s = 45 and 41, w = 1 … 40. Profile: one fresh
  decode from latent 3, every latent j at distance d = j − 3. Tiled: `--vae_decode_tiled`
  (`tiled_decode`, AVV:1470), 512:128 on 720p: 6 tiles (2 × 3 latent tiles of 64, stride 48).
- **Snapshot:** a wrapper on the VAE instance's `_decode` (one slice, no numerical change)
  deep-copies every `InflatedCausalConv3d.memory` before latent 45's slice; latents 45–47 are
  then decoded slice by slice in state ACTIVE from the restored caches, from the GPU copy and
  after a file round trip (`torch.save`, fsync, `posix_fadvise(DONTNEED)`, `torch.load`).
- **Metrics:** bf16 equality; max |Δ| on the [−1, 1] output; PSNR on x = clamp((y + 1)/2, 0, 1);
  equality after round(x · 65535) and round(x · 1023). Times are CUDA-synchronised, per slice
  and per decode. Sanity: decoded vs input frames, 30.5–40.5 dB.

## Results

**Determinism.** The one-pass decode repeated bit for bit in one process: 720p 172.7 / 174.8 s,
720p tiled 202.4 / 200.1 s, 1080p 394.8 / 393.4 s. Across processes, decodes against the saved
references were exact wherever the distance was ≥ 37 (720p s = 41 with w = 37, 38, 40; 1080p
latents 46–47 of the w = 36 run).

### Warm-up resume

720p, s = 45 / s = 41; max |Δ| and PSNR are the worst of the 3 latents compared, "10-bit" the
largest code difference:

| w | Latents decoded | Wall (s) | bf16 identical | Max \|Δ\| | PSNR min (dB) | 10-bit |
|---|---|---|---|---|---|---|
| 1 | 4 | 10.5 / 10.9 | no | 1.53 / 0.95 | 21.6 / 35.6 | 783 / 484 |
| 2 | 5 | 13.7 / 14.2 | no | 0.57 / 0.68 | 34.0 / 40.2 | 292 / 346 |
| 4 | 7 | 20.1 / 20.8 | no | 0.20 / 0.42 | 45.6 / 45.8 | 102 / 212 |
| 8 | 11 | 32.9 / 34.0 | no | 0.27 / 0.29 | 48.1 / 51.8 | 107 / 149 |
| 16 | 19 | 58.7 / 60.1 | no | 0.078 / 0.057 | 57.6 / 56.9 | 32 / 29 |
| 24 | 27 | 84.9 / 86.3 | no | 0.047 / 0.022 | 63.6 / 64.6 | 22 / 12 |
| 32 | 35 | 110.4 / 112.2 | no | 0.031 / 0.012 | 66.4 / 67.8 | 10 / 4 |
| 36 | 39 | 138.6 / 122.3 | latent s: frames 1–2 differ (0.13% / 0.10% of values); s + 1, s + 2 yes | 0.012 / 0.008 | 86.7 / 87.7 | 4 / 2 |
| 37 | 40 | 140.0 / 124.5 | **yes** | 0 | ∞ | 0 |
| 38 | 41 | 143.8 / 130.9 | yes | 0 | ∞ | 0 |
| 40 | 43 | 151.6 / 137.2 | yes | 0 | ∞ | 0 |

- **Tiled 512:128 and 1080p (s = 45):** w = 37 and 38 (and 40 tiled) bit-identical; w = 36
  differs only on frames 1–2 of latent s (86.4 and 87.1 dB); 1080p w = 32 differs on all 3
  latents (66.8 dB).
- 16-bit outputs are identical exactly where bf16 is (w = 36: up to 128–256 codes apart). Below
  37 the error depends on the content: w = 1 gives 21.6 dB at s = 45, 35.6 dB at s = 41.

### Error by distance from the restart

Every distance d at once (one fresh decode from latent 3, 720p); "values ≠" out of the 11,059,200
bf16 values of a latent, max |Δ| and 10-bit untiled:

| d | PSNR (dB), untiled / tiled | Max \|Δ\| | Values ≠ | 10-bit |
|---|---|---|---|---|
| 1 | 22.0 / 22.1 | 1.58 | 98.8% | 809 |
| 4 | 36.6 / 36.5 | 0.64 | 91.3% | 328 |
| 8 | 42.7 / 42.3 | 0.28 | 85.0% | 143 |
| 16 | 40.8 / 40.9 | 0.27 | 87.5% | 139 |
| 20 | 51.4 / 51.7 | 0.074 | 66.1% | 38 |
| 22 | 59.4 / 60.2 | 0.030 | 36.8% | 15 |
| 24 | 64.0 / 63.9 | 0.016 | 16.5% | 8 |
| 32 | 67.9 / 67.6 | 0.0078 | 7.5% | 2 |
| 35 | 75.1 / 74.7 | 0.0078 | 1.75% | 2 |
| 36 | 87.3 / 86.6 | 0.0078 | 0.12%, frames 1–2 only | 2 |
| ≥ 37 | ∞ / ∞ | 0 | 0 | 0 |

The error falls in steps: 22–31 dB at d = 1–2, 37–43 dB for d = 3–17, 56 dB at d = 21 where the
latent-rate layers' reach ends (11 × 2 − 1), 64–68 dB for d = 24–32 (only the 2× and output-rate
layers reach back), 87 dB at 36 (0.0078 = 2⁻⁷, one or two bf16 steps), exact from 37, tiled too.

### Snapshot alternative

- **Exact:** bit-identical from the GPU copy and after the file round trip, at 720p and 1080p
  (latents 45–47), and on a 41-frame 720p clip at latent 6 in an earlier session (whose 720p
  write took 7.3 s); `torch.save` / `torch.load` keep the caches' strides and dtype.
- **33 tensors**, one per kernel-3 conv (11 + 7 + 15), each 2 frames × 2 bytes × its input
  channels × its area. In channels per output pixel: latent rate (1/64 of the pixels) `conv_in`
  16 + mid block 4 × 512 + `up_blocks[0]` 6 × 512 = 5,136 → 80.25; 1/16: 512 + 6 × 512 → 224;
  1/4: 512 + 512 + 5 × 256 → 576; full: 256 + 256 + 5 × 128 + 128 = 1,280. Total 2,160.25 × 4
  bytes = **8,641 bytes per output pixel**, 8.05 GiB per Mpx. Measured: 7,963,545,600 B at 720p
  (0.9216 Mpx) and 18,050,703,360 B at 1080p (1088 × 1920), both exactly 8,641 × pixels: the
  8.05 × P of [vram.md](vram.md#untiled-vae-the-model)'s decode caches.

| Step (copy on the GPU: 0.02 / 0.05 s) | 720p (7.96 GB) | 1080p (18.05 GB) |
|---|---|---|
| GPU → CPU, then write + fsync | 1.9 + 3.2 s | 4.0 + 9.0 s |
| Read (page cache dropped), then CPU → GPU | 2.2 + 0.35 s | 4.4 + 0.71 s |
| **Save / restore** | **5.1 / 2.5 s** | **13.0 / 5.1 s** |

### Time per latent and the cost of a resume

| Decode | ACTIVE slice (1 latent) | First slice (2 latents) | One-pass, per latent |
|---|---|---|---|
| 720p | 3.43–3.48 s (3.17–3.66 over all runs) | 4.4 s (4.0–4.7) | 3.39–3.43 s |
| 720p tiled 512:128 | 6 tiles × 0.66 s | 4.9–5.1 s | 3.92–3.97 s |
| 1080p | 7.82–7.84 s | 10.0–10.4 s | 7.71–7.74 s |

- **Linear in pixels:** 7.74 / 3.39 = 2.28 for 2.27× the pixels, 0.93–0.94 s per output Mpx and
  frame ([vram.md](vram.md#untiled-vae-the-model): 0.83 in another session; the same 720p slices
  took 4.70 s in an earlier session). Tiling 512:128: +16% for 1.41× the area decoded.
- **A warm-up** is the first slice plus w − 2 ACTIVE slices. At w = 37, 1080p: 10.0 + 35 × 7.83 ≈
  284 s; the w = 37 decode took 307.2 s for 40 latents, 307.2 − 3 × 7.82 = 283.7 s. At 720p:
  140.0 − 3 × 3.56 = 129 s. Either way 148 output frames are decoded twice per resume, against
  13 s to save and 5 s to restore a snapshot, plus 18 GB of disk per resume point at 1080p.

## Caveats

- **One clip:** exactness at w ≥ 37 follows from the receptive field and held at both resume
  points, both resolutions and tiled; the size of the error below 37 depends on the content.
- **One GPU and software stack** (Blackwell, torch 2.14.1+cu130, cuDNN 9.24, Conv3d workaround on,
  `cudnn.benchmark` off): kernels are picked by cuDNN's heuristics per shape. Timings drift with
  the power cap: compare them within a session ([benchmarking.md](benchmarking.md#caveats)).
- **Memory pressure not exercised:** the workaround's `torch.cudnn_convolution` falls back to
  `F.conv3d` on any `RuntimeError` (CIL:94-113), `retry_on_oom` retries after emptying the caches
  (`src/optimization/memory_manager.py:361`); either, or a cuDNN algorithm skipped for lack of
  workspace, may switch kernels and break bit-identity. seedvr2x should make them errors.
- **The layout must match the original run:** tiling (tiled and untiled outputs differ: 38.6 vs
  40.3 dB to the input on latent 41), tile size and overlap, memory limits (they decide the conv
  and GroupNorm splits), one latent per slice, dtype. w ≥ 2 keeps latent s in an ACTIVE slice.

## Reproduce

```bash
# cwd = the numz checkout, its venv's python; SEEDVR2_DIR defaults to the cwd, FFMPEG to ffmpeg
S=/path/to/scripts/decode_resume.py; M=/path/to/models; O=out/dec; IN=/path/to/segment.mp4
python $S encode --model-dir $M --input $IN --frames 201 --size 1280x720 --out $O/lat720.pt
python $S encode --model-dir $M --input $IN --frames 201 --size 1920x1080 --out $O/lat1080.pt
python $S run --model-dir $M --latents $O/lat720.pt --ref-runs 2 --ref-save $O/ref720.pt \
  --snapshot 45 --profile 3 --resume 45 --warmup 37,38,40,36 --out $O/r720a.json
python $S run --model-dir $M --latents $O/lat720.pt --ref-runs 0 --ref-load $O/ref720.pt \
  --resume 45 --warmup 32,24,16,8,4,2,1 --out $O/r720c.json
python $S run --model-dir $M --latents $O/lat720.pt --ref-runs 0 --ref-load $O/ref720.pt \
  --resume 41 --warmup 37,36,38,40,32,24,16,8,4,2,1 --out $O/r720b.json
python $S run --model-dir $M --latents $O/lat720.pt --tiled 512:128 --ref-runs 2 \
  --ref-save $O/ref720t.pt --profile 3 --resume 45 --warmup 37,36,38,40 --out $O/r720t.json
python $S run --model-dir $M --latents $O/lat1080.pt --ref-runs 2 --ref-save $O/ref1080.pt \
  --snapshot 45 --resume 45 --warmup 38,37 --out $O/r1080a.json
python $S run --model-dir $M --latents $O/lat1080.pt --ref-runs 0 --ref-load $O/ref1080.pt \
  --resume 45 --warmup 36,32 --out $O/r1080b.json
python $S table $O/r*.json
```

Each `run` rewrites its JSON after every decode; the snapshot file (`--snapshot-dir`, default:
next to the JSON) is deleted after its round trip. Encoding took 101 s (720p) and 247 s (1080p).

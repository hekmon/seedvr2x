# seedvr2x design

> Status: **draft for review**, nothing implemented. It collects the decisions taken so far and
> the questions still open. Each decision links to the measurement it rests on in
> [../research/](../research/).

## Goal

A SeedVR2 video upscaler for **long runs** (whole episodes or films) that:
- produces a lossless, correctly tagged master that the next encoding step can trust
- has no visible seams inside a shot
- fits the GPU it runs on without the user tuning memory options
- can be paused and resumed (run at night, give the computer back in the morning)

It replaces numz's orchestration and I/O (about 14.6k lines, where nearly all of the
[23 bugs](../research/bugs/README.md) live). It keeps ByteDance's model code.

### Two kinds of users, both first-class
- **Standalone, the regular workflow:** a video file in, an upscaled file out, with ffmpeg as
  the only other tool. seedvr2x finds the cuts itself and assembles the finished file, the
  source's audio, subtitles and chapters included. File size matters here, hence
  `--segment-cmd` (see [Output](#output)).
- **With sptenc**, the first use case and the reason for this project. This is the manual
  workflow, pre-split in and pre-split out: `sptenc split` → seedvr2x →
  `sptenc encode <dir> -f original`, with a directory of segments as the hand-over. It stays
  lossless end to end, and its users bring the disk space.

sptenc shapes the interfaces (segment directories, cut lists, master formats), but seedvr2x
never requires it.

### Not in the first version
- Image inputs, ComfyUI nodes, macOS/MPS, AMD/ROCm
- Multi-GPU in one process (separate processes on separate shot ranges can come later, see
  [Pause and resume](#pause-and-resume))
- Options that only work around numz's own design (see [Options](#options-kept-and-dropped))
- Variable frame rate sources: refused with a clear message, as sptenc does
- Interlaced and telecined sources, in any version. Deinterlacing and inverse telecine are
  workflows of their own, with specialised tools and many ways to do them. Detection:
  - refused when the stream declares interlacing
  - a warning when ffmpeg's `idet`, run during the first pass, finds combed frames in a source
    declared progressive, as hard-telecined sources often are
- Windows: Linux first, since inference is mostly a Linux world. Windows is a best-effort port
  after the prototype, starting with an analysis of:
  - FlashAttention 2 and `torch.compile`'s Triton there (SDPA works everywhere)
  - the driver spilling VRAM into system memory instead of failing (numz enforces the physical
    limit for this), which the planner must prevent

## Architecture

| Layer | Contents | Origin |
|---|---|---|
| Model | DiT 7B/3B, VAE, windowed attention, RoPE, Euler sampler, configs, text embeddings | **vendored** from `upstream/seedvr2-numz` (ByteDance code, numz changes) |
| Runtime | shot pipeline, latent stitching, memory planner, BlockSwap, resume | ours |
| I/O | ffmpeg pipes in and out, FFV1/PNG writers, manifest | ours |
| CLI | options, logging | ours |

**One long-running process** handles a whole job: models are loaded (and compiled) once and
stay on the GPU when the plan allows it.

### Language: all Python
Every layer is Python, in one process per job. seedvr2x stands alone; with sptenc, the two
connect through files.

- **Why:** inference is Python territory. The model, the frames as tensors, the latents,
  stitching, the planner (`mem_get_info`), BlockSwap, colour correction and the resume manifest
  all need torch, so a Python environment is required whatever language orchestrates it, and
  Python imports the model code and its libraries directly. sptenc could be all Go because it
  only drives external programs (ffmpeg), which process calls handle completely.
- **With sptenc:** `sptenc split` → seedvr2x → `sptenc encode <dir> -f original`. sptenc's
  pre-split directories exist for this ("splitting a source, upscaling its segments and
  encoding them"). In that chain, sptenc does the scene detection, the encodes, VMAF, concat
  and the final mux.
- **Door left open:** the runtime is a library (job → shots → units, emitting events), so a
  protocol front end can be added if sptenc ever drives seedvr2x directly.
- **Rejected:**
  - Go orchestration around a persistent Python worker. The worker would still hold everything
    above, the manifest included (only it knows windows and latents). Go's share would mostly
    re-wrap sptenc's `ffmpeg` package (probing, frame counts, scdet, split, concat; ~1.9k
    lines), at the permanent cost of a versioned protocol, signal translation (sptenc cancels
    at once and kills its children), progress bridging, `--until` estimates crossing the
    boundary, and two packagings.
  - One Python process per segment. Each one reloads the model and recompiles: 10–50 s of DiT
    compilation, and 108.5 s for the first compiled VAE encode batch against 4.79 s steady
    ([vram.md](../research/docs/vram.md#torchcompile)). That's 27–136 min of DiT compilation
    alone over a 163-segment episode, and no stitching across joins.
  - An upscale stage inside sptenc: the same protocol, plus changes in sptenc's package `main`.

### Vendored model code
- Copied from `upstream/seedvr2-numz` at `4490bd1`: `src/models/{dit_7b,dit_3b,video_vae_v3}`,
  `src/common/diffusion`, the parts of `src/common` and `src/core/infer.py`
  (`VideoDiffusionInfer`) that the model needs, the configs and `pos_emb.pt`/`neg_emb.pt`.
  About 11k lines ([provenance](../research/docs/provenance.md)).
- The submodules stay untouched. A script diffs our copy against numz and ByteDance, so every
  change stays visible.
- Changes we expect to make in our copy:
  - remove the dependencies on numz's runtime: `retry_on_oom`, the attention dispatch, the
    Conv3d flag, MPS detection, the banners and import-time shims
  - attention: SDPA, or FlashAttention 2 when installed, nothing else
    ([why](../research/docs/attention.md))
  - drop the ~880 lines of sequence-parallel code
  - decide per change whether to keep numz's numerics or go back to ByteDance's: RoPE in half
    precision (ByteDance: fp32), attention in the pipeline dtype (ByteDance: bf16), VAE encode
    taking the mode instead of a sample. Each one is measured against the other before choosing.
- Licence: Apache-2.0. We keep the copyright headers, add a NOTICE, and mark modified files.
  The StableSR-derived colour code (`color_fix.py`, non-commercial licence) is **not**
  vendored: see [Colour correction](#colour-correction).

### Weights
7B fp16, 7B fp8, 7B Q4_K_M (GGUF), 3B fp16/fp8, all Apache-2.0
([memory and quality per model](../research/docs/vram.md)). GGUF needs the dequantisation code
adapted from city96/ComfyUI-GGUF (Apache-2.0, credited).

## Input

Two forms, one internal model (a list of shots):
1. **A video file**: a source as it is (any codec ffmpeg decodes, no intermediate master
   needed) or a lossless master. The cuts come from seedvr2x's own detection by default, run as
   a first pass before any GPU work so the planner knows every shot (time estimate, `--until`,
   manifest). Or they come from a cut list (frame numbers or timestamps): sptenc's once it
   exports one, or any other tool's.
2. **A directory of segments** (sptenc's split, or any other splitter), plus which joins are
   real cuts: from a cut list, else each join scored by the detector. Joins that aren't real
   cuts are stitched like a long shot. The output mirrors the input's segments (same frame
   ranges and names), so sptenc encodes them as it would its own split.

Cuts matter for quality, not only for the VAE context: see [Pipeline](#pipeline-per-shot).

Decoding goes through an ffmpeg pipe:
- frame-accurate, counting the frames actually decoded (never trusting the container's count,
  bug 11)
- 16-bit RGB, converted by ffmpeg with every parameter given (matrix, range, chroma location):
  zscale when the build has it, else swscale with its accuracy flags and a warning. The matrix
  is the tags', else BT.709 for HD and BT.601 for SD, with a warning and an override.
- exact rational frame rate

ffmpeg does every colour conversion, in and out: we pin its parameters rather than
reimplementing them. A startup check reports what the build offers (zscale, ffv1, scdet). Tests
verify its conversions: round trip, white at 940, black at 64, chroma siting.

### Colour and shape, SD sources included
The rule: the upscale must look like its source in any given player.
- **Matrix:** BT.709 for the `yuv420p10le` master at HD sizes and above, whatever the source's
  (BT.601 for SD). Players read the tag, or assume BT.709 at HD sizes, so this is what keeps
  the colours identical everywhere.
- **Primaries and transfer:** copied as the source declares them (untagged stays untagged),
  never converted.
  - Converting the gamut (SD's SMPTE 170M or BT.470 BG to BT.709) changes the RGB values
    themselves. Most players ignore the primaries tag, so they would show the upscale
    differently from its source.
  - A gamut conversion is a grading choice, for the user's own tools.
- **Shape:** the declared sample aspect is honoured (with an override for broken tags), and the
  output has square pixels at the source's display aspect.
  - Players honour the sample aspect. numz ignores it (OpenCV), so its upscale of a 16:9 DVD
    comes out 16% too narrow (NTSC) or 30% too narrow (PAL).
  - The ITU-R BT.601 question of 704 or 720 active pixels (≈ 2%) is left to the stream's
    declaration.
  - The correction is only a different target size for the input resize (step 0 of the
    [Pipeline](#pipeline-per-shot)).

## Pipeline, per shot

0. **Resize** of the input to the output size. The model restores a picture that is already at
   its final size: ByteDance's reference pipeline upsamples first ("Upsample image, model only
   trained for high res."), with torchvision's bicubic (`NaResize`). That is a compiled
   C++/CUDA kernel, not Python code, and we keep it as the default.
   - Other kernels (zimg's Spline36 or Lanczos, through ffmpeg) are a measured choice, not a
     default: a sharper kernel pre-sharpens and rings, and the model takes that for content.
   - To pin: torchvision's bicubic depends on its `antialias` flag (PIL-like a = −0.5 with
     it, OpenCV-like a = −0.75 without), and that flag's default changed in torchvision 0.17.
     Verified with torchvision 0.29.1: `TVF.resize` defaults to `antialias=True`, so NaResize
     runs a = −0.5.
1. **VAE encode** of the whole shot in one causal pass. The VAE already streams in 4-frame
   slices; resetting it at each cut is correct (no context should cross a cut). Shots must
   start at real cuts:
   - each latent packs 4 frames (latent frames = 1 + (frames − 1)/4), so a cut inside a group
     mixes both scenes in one latent
   - windows stitched across a cut would cross-fade the two scenes

   Expected to be visible; not measured yet.
2. **DiT** on windows:
   - one window per shot when it fits
   - otherwise equal windows sharing **M = 2 latents** (8 frames), the shared latents mixed
     before decoding
   - noise drawn once per shot from the seed, sliced per window

   [Measured](../research/docs/stitching.md): −80% boundary jump against independent batches,
   no softening, output 40.5–40.8 dB from the single-window result, about +3–9% compute
   (the VAE is most of the time).
3. **VAE decode** of the shot in one streaming pass, tiled when the planner says so.
4. **Colour correction** against the input (see below).
5. **Write** frames as they come out.

Fallback when latent stitching can't be used: a linear pixel cross-fade over K = 4 frames
(−67% boundary jump, mixed frames ~20% softer).

## Colour correction

`lab` is the recommended mode on quality grounds
([quality.md](../research/docs/quality.md)): it removes the model's colour drift (+30%
saturation, toward blue), halves boundary jumps and cancels the VAE tile drift.
Its first step is StableSR's wavelet split (non-commercial licence). We **rewrite** that split
(an iterated blur separating low and high frequencies, or a Gaussian low-pass), keep numz's LAB
histogram matching (Apache-2.0), and validate against numz's `lab` with the same metrics
(ΔE low-frequency, a*/b* spread, boundary jumps).

## Output

seedvr2x delivers the upscale losslessly and owns no encoder flags. How the output gets
compressed is the user's choice: afterwards, from the master, or during the run with
`--segment-cmd` (below).

- **FFV1 masters**, every frame a keyframe, per-slice CRCs, exact frame rate, from the float
  frames (no 8-bit step, bugs 09/19):
  - `gbrp16le`: research and archive master, closest to the model
  - `yuv420p10le`, BT.709, limited range, explicit conversion (zscale): the master handed to
    sptenc, which takes such a file as it is, so no implicit RGB→YUV happens downstream. It
    shares the pixel format and tags of `sptenc master` (chroma sited left included), not its
    conversion: sptenc's swscale puts 16-bit white at 943 instead of 940. The conversion tests
    (see [Input](#input)) check that zscale sites the chroma left, as tagged.

  Validated with the [`ffv1_out.py` wrap](../research/docs/output.md): bit-exact round trip,
  tags checked by ffprobe.
- **PNG** (16-bit) as an alternative.
- **Output segments**, the resume units of the output, listed in a manifest:
  - **Layout, by sptenc's rule.** The threshold picks the cuts, then the minimum segment
    length (5 s by default) merges each too-short segment into its shorter neighbour, on the
    frame grid. With scdet run as sptenc runs it, the same threshold and minimum give the same
    segments as sptenc. With a directory of segments as input, the output mirrors it instead.
  - **Shots, the model's units, keep every cut.** A segment can hold several shots, and the
    model still resets at each cut inside it. So the merge costs no quality. It saves what
    short segments cost an encoder: a forced keyframe each, and too few frames to amortise it
    (sptenc's MANUAL, "Too fine: many short segments").
  - **Writer.** FFV1 by default. With `--segment-cmd`, the user's command runs once per
    segment: it reads the segment, lossless and tagged, on stdin, and writes the file seedvr2x
    names, e.g. `--segment-cmd 'ffmpeg -i - -c:v libx265 -crf 16 {out}'`. seedvr2x never
    parses the command and only checks the result's frame count. Disk use is then the
    compressed size, against 90–250 GiB per hour of 1080p anime for masters
    ([output.md](../research/docs/output.md)), and a stop keeps every finished segment.
  - **Rejected:** `--stream`, the output on stdout for a single compressor process. That
    process can't be paused, so a stopped run would leave parts to join by hand.
- **Assembly** of the segments:
  - **standalone:** joined into one file by stream copy, with the source's other streams
    (audio, subtitles, chapters, attachments). The join carries sptenc's lessons: each
    segment's duration comes from its frame count, and timestamps are snapped to the frame
    grid. Otherwise the video drifts: 46 ms behind the audio over a 163-segment episode, in
    sptenc's measurements.
  - **with sptenc:** the directory as it is, for `sptenc encode <dir> -f original` (or
    `sptenc concat` for one master)

## Memory planner

Built into the CLI, from the validated models in [vram.md](../research/docs/vram.md):
- **Budget:** free memory reported by the driver after the CUDA context exists
  (`mem_get_info`), minus a margin. That covers the desktop and other programs without
  guessing. The validated rule was "torch peak ≤ card size − 2 GiB"; the margin over measured
  free memory is to be re-derived from the emulation data (≈ 0.6 GiB).
- **Per-phase peaks** (P = output megapixels, T = tile size in megapixels, L = latent frames
  per window):
  - VAE encode ≈ 1.2 + 8.8·P GiB, decode ≈ 0.8 + 16.1·P GiB (flat beyond 9 frames)
  - tiled: encode ≈ 1.7 + 8.4·T², decode ≈ 1.6 + 15.6·T²
  - DiT (7B fp16) ≈ 16.05 GiB + 128.5 KiB per token (≈ 0.48 GiB per Mpx per latent frame),
    constants per model in vram.md
- **Choices, in order:** model (if the user allows a smaller one), window length (the biggest
  quality lever: fewer boundaries), BlockSwap blocks, VAE tile sizes, then what's left goes to
  speed: `compile_dit` (−26 to −32% DiT time, +0.1 to +1.4 GiB), and `compile_vae` only when
  its memory fits (it about doubles VAE activation memory, for −16 to −19% VAE time).
- `--plan` prints the plan and the time estimate without running.
- Validated with `vram_cap.py` emulation of 8–48 GB cards before release.

### BlockSwap, rewritten
numz moves each block to the GPU and back to pageable memory at every forward pass,
synchronously ([measured](../research/docs/vram.md)). Weights never change, so:
- one **pinned** host copy per block, host→device only (57 GB/s pinned vs 4 GB/s back to
  pageable memory)
- **prefetch** the next block on a side stream while the current one computes

Expected: most of the swap cost hidden. To measure.

### Allocator
`cudaMallocAsync` stays the default (it trims its pool on a full card; `native` fragments and
fails). Set by us, documented, overridable.

## Pause and resume

Work is saved in resumable units; a stop loses only the unit in progress.

| Stage | Saved | Resume granularity |
|---|---|---|
| VAE encode | the shot's latents (~1 MB per 4 frames at 1080p) | per shot |
| DiT | each window's output latents | per window |
| VAE decode | output segments (FFV1, or the user's `--segment-cmd`) | per output segment: its decode and write restart, the DiT latents are kept (finer, per sub-segment with warm-up latents, if that proves bit-identical) |

- **Manifest:** settings, seed, model hashes, finished shots and windows. A resume with
  different settings is refused.
- **Stopping:** Ctrl-C once finishes the current unit and exits; twice stops at once.
  `--until HH:MM` stops cleanly before a unit that wouldn't finish in time, using the
  planner's time estimates.
- Exiting frees the GPU entirely.
- **Resume must be bit-identical** to an uninterrupted run: deterministic noise per shot, exact
  latents, deterministic attention (FA2 reruns are bit-identical). Automated test: run,
  interrupt, resume, compare the FFV1 masters bit for bit.
- The same manifest lets separate processes take separate shot ranges (several GPUs or
  machines) later.

## Options kept and dropped

| numz option | seedvr2x |
|---|---|
| `--attention_mode` | dropped: FA2 when installed, else SDPA |
| `--color_correction` | `lab` (rewritten) by default; `none` available |
| `--input_noise_scale` | dropped (numz's own addition, only degrades) |
| `--latent_noise_scale` | dropped from v1 (ByteDance uses 0; corrected version possibly later as an experiment, see bug 08) |
| `--batch_size`, `--temporal_overlap`, `--prepend_frames`, `--uniform_batch_size` | replaced by shots, windows and latent stitching, chosen by the planner; window length can be capped by the user |
| `--blocks_to_swap`, tile sizes, offload devices | chosen by the planner; overridable |
| `--compile_dit`, `--compile_vae` | chosen by the planner when memory allows; overridable |
| `--cache_dit`, `--cache_vae`, `--chunk_size` | dropped: one long-running process, streaming by design |
| `--seed` | kept (comparisons, reproducibility) |
| `--output_format`, `--video_backend`, `--10bit` | replaced by FFV1 (`gbrp16le` / `yuv420p10le`) and 16-bit PNG |
| `--resolution`, `--max_resolution` | kept |
| `--dit_model`, `--model_dir` | kept; models identified by hash, no silent deletion (bug 22) |
| `--cuda_device` | one device per process |

## Validation milestones

1. **Reproduce numz:** same settings (one batch, no tiling, `flash_attn_2`, same seed) →
   bit-identical or within quantisation of numz's float frames, checked with the FFV1 masters.
   Then each numerics choice (RoPE precision, attention dtype, VAE mode vs sample) measured
   separately.
2. **Stitching:** reproduce the latent-stitching results from the study.
3. **Planner:** every card size passes under `vram_cap.py` emulation; plan estimates within a
   few percent of measured time and memory.
4. **Resume:** interrupted and resumed runs bit-identical to uninterrupted ones.
5. **Colour correction:** rewritten `lab` matches numz's `lab` on the quality metrics.
6. **Assembly (standalone):** the finished file's video timestamps equal the source's, frame
   for frame, and every other stream is copied.
7. **Visual review** of long runs by the user.

## Open questions

### Input
- **Own scene detection:** three candidates.
  - ffmpeg's scdet, as sptenc runs it: the same scores, so both tools detect alike and, with
    the same merge, lay out the same output segments. No new dependency.
  - PySceneDetect: detects fades, but adds a dependency
  - a detector on our own decoded frames

  Output segments take sptenc's minimum length (see [Output](#output)). Shots are a different
  matter:
  - merging a real cut into a shot puts two scenes in one latent
  - keeping a false detection (a flash, a fast pan) splits continuous motion with a hard
    boundary

  To measure: the threshold, and whether shots need a minimum of their own.
- **Scene list format.** Known from sptenc's code:
  - it exports no scene list (`split --list-scenes` prints a table)
  - it never splits inside a shot (no maximum length, no fade handling)
  - but it merges scenes shorter than 5 s into their shorter neighbour, so real cuts can sit
    inside a segment unmarked
  - scdet also fires on pans and flashes, so a join isn't always a cut

  Open: the machine-readable list sptenc should export (frame indexes, scores, and the cuts
  the merge removed), and how seedvr2x treats cuts inside a segment and doubtful joins.
- **Frame-exact access into long-GOP sources**, to resume a shot and to read its input frames
  again for colour correction. Three ways:
  - ffmpeg's accurate seek: fast, but trusts timestamps
  - decoding from the start and counting: exact, but slow on a film
  - a lossless intermediate: exact and fast, but large

  Plan: measure accurate seek against decode-and-count on real long-GOP files.

### Scope
- **Model scope for v1:** 7B fp16 only, or fp8/Q4 and 3B from the start (needed for small
  GPUs).

### To measure
- **Numerics:** numz's or ByteDance's for RoPE, attention dtype and VAE encode, once measured.
  The input preparation too: the resize kernel and its `antialias` flag, and multiples of 16
  reached by padding (numz) or cropping (ByteDance).
- **Decode resume granularity:** whether sub-segment decoding with warm-up latents is
  bit-identical.
- **4K and long windows:** the planner's limits on large outputs, where the DiT window is the
  constraint.

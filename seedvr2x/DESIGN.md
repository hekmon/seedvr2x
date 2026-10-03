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
- Sources refused in v1, each with a clear message:
  - a rotation or flip in the display matrix (phone and camera files). The decode passes
    `-noautorotate`, so nothing rotates behind our back. The likely later way is to process the
    frames as stored and carry the rotation to the output.
  - declared cropping (MP4 `clap`, Matroska `PixelCrop`), which ffmpeg applies on decode. The
    likely later way is to accept the cropped picture and size everything from it.
  - any pixel format outside the tested list. Accepted:
    - planar YUV 4:2:0, 4:2:2 and 4:4:4 at 8 to 16 bits
    - 4:1:1, 4:1:0 and 4:4:0 at 8 bits
    - `yuvj` 4:2:0, 4:2:2 and 4:4:4
    - planar RGB at 8 to 16 bits
    - 26 packed or semi-planar formats, repacked exactly

    Everything else is refused: alpha (dropping it changes the picture wherever it isn't
    opaque), palette, grey, XYZ, Bayer, float, 5-6-5 RGB, 4:4:0 above 8 bits,
    `yuvj440p`/`yuvj411p`.
  - matrices other than BT.709, BT.601 (`bt470bg`, `smpte170m`) and BT.2020 non-constant
    luminance. The constant-luminance ones and ICtCp need the transfer function, which we
    never convert. FCC, SMPTE 240M and YCgCo are rare and would go untested.
  - RGB tagged limited range: rare, most likely a mis-tag, and either reading could be wrong
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
- Copied from `upstream/seedvr2-numz` at `4490bd1`:
  - `src/models/{dit_7b,dit_3b,video_vae_v3}` and `src/common/diffusion`
  - the parts of `src/common` and `src/core/infer.py` (`VideoDiffusionInfer`) that the model
    needs
  - `src/data/image/transforms/`, the input preparation of [step 0](#pipeline-per-shot):
    `NaResize`, plus `DivisiblePad` (numz) or `DivisibleCrop` (ByteDance) to reach multiples
    of 16. numz chains them at `src/core/generation_utils.py:73-81`.
  - the configs and `pos_emb.pt`/`neg_emb.pt`

  About 11k lines ([provenance](../research/docs/provenance.md)).
- The submodules stay untouched. A script (`tools/vendor.py diff`) diffs our copy against numz
  and ByteDance, so every change stays visible.
- Changes made in our copy, each one checked bit-identical to numz by the milestone-1
  regression test:
  - **numz's runtime dependencies removed:** `retry_on_oom`, the attention dispatch, MPS
    detection, the banners and import-time shims, the model registry
    (`src/utils/model_registry`), and the performance helpers
    (`src/optimization/performance`).
  - **Logging:** `VideoDiffusionInfer` logs through numz's `Debug` object, which our runtime
    supplies as a logging adapter.
  - **Attention:** FlashAttention 2 when installed, else SDPA; any other mode is an error
    ([why](../research/docs/attention.md)).
  - **Conv3d:** numz's workaround is removed. It called `torch.cudnn_convolution` directly,
    because plain Conv3d takes ≈ 3× the memory with torch ≥ 2.9 and cuDNN ≥ 9.10.2
    ([environment.md](../research/docs/environment.md)).
    - On the pinned stack (torch 2.14.1, cuDNN 9.24), the standard path gives bit-identical
      output, the same peaks (encode 19.03 GiB, decode 34.49 GiB on 21 frames at 1080p) and the
      same time. [vram.md](../research/docs/vram.md#other-knobs) found the same memory and time.
    - The bug depends on the torch and cuDNN versions. The stack therefore stays pinned
      (`uv.lock`), and any upgrade of torch or cuDNN re-runs the regression test and this
      memory check.
  - **Sequence parallelism:** dropped: `common/distributed/{advanced,ops}.py`, numz's stubs in
    the VAE, and the dead DDP and TF32 helpers, 804 lines in all. On one GPU every call was an
    identity.
- **Upstream quirks**, left as they are:
  - The 3B DiT only works with its cache on: its `vid_out_ada` gets its shape from a cache key
    that block 0 fills.
  - `na.unconcat_coalesce` fails on a batch whose samples have different window counts. Each
    DiT call therefore takes one window (see [Pipeline](#pipeline-per-shot)).
- **Numerics:** keep numz's or go back to ByteDance's, change by change, each measured against
  the other before choosing. What numz's 7B fp16 path actually does (implementation probe):
  - it computes in bf16: the pipeline dtype comes from a probe at import, the fp16 VAE weights
    are cast to bf16, the fp16 DiT runs under bf16 autocast, and attention computes in bf16
  - q and k reach RoPE and attention in fp32 (probed on a 2-layer model)
  - the RoPE frequencies are stored in fp16 in the checkpoint (36 equal tensors of 10 values),
    and `rotary_embedding_torch` computes the angles in that dtype
  - input frames are cast fp32 → fp16 → bf16: 26 of the 256 8-bit codes differ from a direct
    bf16 cast
  - VAE encode takes the posterior mode

  So two of the differences listed so far, "RoPE in half precision" and "attention in the
  pipeline dtype", may not differ from ByteDance on this path. This must be confirmed before
  any patch.
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
   manifest). Or they come from a cut list.
   - The cut list format: one frame number per line, the first frame of each shot except the
     first, `#` comments, no timestamps. Frame numbers are exact on the frame grid.
   - Fields after the frame number are ignored, so an export carrying scores (sptenc's, once
     it has one) stays readable.
2. **A directory of segments** (sptenc's split, or any other splitter), plus which joins are
   real cuts: from a cut list, else each join scored by the detector. Joins that aren't real
   cuts are stitched like a long shot. The output mirrors the input's segments (same frame
   ranges and names), so sptenc encodes them as it would its own split.
   - sptenc's split names its segments `seg_%06d.mkv`: FFV1 `yuv420p10le` in Matroska,
     timestamps reset.
   - Its `encode <dir>` takes the `.mkv` and `.mp4` files, any names, in byte-wise name order,
     and applies no minimum length to a directory. seedvr2x reads a directory the same way.
   - Every segment must share size, sample aspect, frame rate, conversion, pixel format,
     primaries and transfer, else the directory is refused.
   - Until the detector comes, every join is a cut, and `--cuts` is refused with a directory.

Cuts matter for quality, not only for the VAE context: see [Pipeline](#pipeline-per-shot).

Decoding goes through an ffmpeg pipe:
- frame-accurate, counting the frames actually decoded (never trusting the container's count,
  bug 11)
- 16-bit RGB, converted by ffmpeg's zscale with every parameter given (matrix, range, chroma
  location, chroma kernel). Each is the tags' value; untagged values are guessed with a
  warning, and the matrix has an override:
  - matrix: BT.709 when the width is at least 1280 or the height above 576, else BT.601.
    That's mpv's rule for untagged video, a player's explicit rule, as our rule (the upscale
    looks like its source in a player) wants. Sizes in between, such as 960×540, count as SD.
  - range: limited for YUV (the `yuvj` formats are full); RGB is read full range
  - chroma location: left, the H.264, HEVC and MPEG-2 default. The JPEG family is
    centre-sited, and ffmpeg declares it so (MJPEG in AVI, MKV and MOV, with ffmpeg 6.1.1 and
    n9.0.2), so the tags cover it.
  - chroma kernel: bicubic with b = 0, c = 0.5 (Catmull-Rom: zscale
    `filter=bicubic:param_a=0:param_b=0.5`). That is Keys' cubic with a = −0.5, the kernel of
    step 0's resize, so every interpolation feeding the model is the reference pipeline's.
    zscale's default, bilinear, blurs chroma edges. Sharper kernels (Spline36, Lanczos)
    pre-sharpen and ring, so they are measured options, as for the resize.
  - primaries and transfer: the same on both sides of zscale, so never converted; no dither
  - **zscale is required.** Without it, seedvr2x refuses to run and says how to get a build
    that has it.
  - swscale is no fallback. Measured with ffmpeg n9.0.2, even the bit depth alone differs:
    zscale expands 8-bit RGB to 16 bits exactly (v × 257, white at 65535), while swscale, its
    accuracy flags included, puts white at 65283 (0.4% low), 128 at 32767 and 1 at 256. A
    systematic error on every frame the model sees breaks the rule that the upscale looks like
    its source.
  - zscale expands 10-bit RGB exactly too. At 9, 12 and 14 bits, some values land one code off
    out of 65535, because zimg computes in float32. That is far below the fp16 cast that
    follows, which keeps 11 significant bits.
  - zscale must be fed planar RGB: given packed RGB, ffmpeg puts swscale in front of it, which
    then does the expansion.
- exact rational frame rate, checked against the frames themselves. The first pass (below)
  decodes every frame before any GPU work and refuses the source when its shortest and
  longest frame durations differ by more than 1 ms, sptenc's rule. The declared rate must
  also match the measured durations, since the output is written at that rate.
  - Comparing declared rates alone fails on Matroska: sptenc found a 24/30 fps mix declared
    24/1 for both.
  - The pass costs one software decode (171 fps on an HEVC master, over 1,000 fps on
    H.264). scdet and idet join it with automatic scene detection.
- interlacing: refused when the field order is neither progressive nor unknown (sptenc's
  rule)

ffmpeg does every colour conversion, in and out: we pin its parameters rather than
reimplementing them. A startup check refuses a build without zscale, ffv1 or scdet. Tests verify
its conversions: round trip, white at 940, black at 64, chroma siting.

### Colour and shape, SD sources included
The rule: the upscale must look like its source in any given player.
- **Matrix:** BT.709 for the `yuv420p10le` master at HD sizes and above, whatever the source's
  (BT.601 for SD). Players read the tag, or assume BT.709 at HD sizes, so this is what keeps
  the colours identical everywhere.
  - Below HD, by the same mpv rule (width under 1280 and height up to 576), the master is
    BT.601.
  - It is tagged as the source is when the source says `bt470bg` or `smpte170m`, which have
    the same coefficients; otherwise it is tagged `smpte170m`.
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
   - The target size follows the display aspect. At square pixels it is the size torchvision
     computes from NaResize's int, and the resize to it is the same call, bit for bit.
1. **VAE encode** of the whole shot in one causal pass. The VAE already streams in 4-frame
   slices; resetting it at each cut is correct (no context should cross a cut).
   - The frames are read from the decode as the VAE takes them: 5, then 4 at a time.
   - The padding to 4n + 1 is made from the last 4 frames read, so the latent is the one-pass
     encode's, bit for bit.
   - Only the latents are ever whole in memory.

   Shots must start at real cuts:
   - each latent packs 4 frames (latent frames = 1 + (frames − 1)/4), so a cut inside a group
     mixes both scenes in one latent
   - windows stitched across a cut would cross-fade the two scenes

   Expected to be visible; not measured yet.

   A shot that isn't 4n + 1 frames long is padded, then trimmed after decoding. numz mirrors
   the end (`generation_utils.py:642-654`); ByteDance repeats the last frame
   (`inference_seedvr2_7b.py:182-196`). We use numz's padding for now: the choice belongs to
   the numerics.
2. **DiT** on windows:
   - one window per shot when it fits
   - otherwise windows sharing **M = 2 latents** (8 frames). Their lengths are balanced,
     differing by at most 1, with as few windows as the cap allows: 21 latents under a cap of
     6 give 6, 6, 6, 6, 5. A grid of full windows plus a short last one can leave a window of
     3 latents with only 1 new.
   - the shared latents are mixed before decoding with cosine weights (0.75 then 0.25 for
     M = 2), the curve the study measured
   - noise drawn once per shot from the seed, sliced per window. Each latent gets the noise a
     one-window run of the shot would give it, and a shared latent gets the same noise in both
     of its windows. The study reseeded every window instead, so its shared latents mixed two
     different renderings. Milestone 2 kept the sliced scheme: it is 1.2–2.0 dB closer to the
     one-batch run at every offset to the boundaries, with the same boundary steps (see
     [Validation milestones](#validation-milestones)).
   - each shot's generator is seeded with the seed plus the shot's first frame index in the
     source.
     - A shot starting at frame 0 gets the plain seed, as numz's single batch does.
     - Shots never share a noise pattern.
     - A shot's output depends only on the seed, its frames and its place in the source, so a
       resumed or re-cut job reproduces the untouched shots exactly.
     - Its encode uses the seed plus its first frame plus 1,000,000, as numz does.
     - numz seeds NumPy, which takes seeds below 2^32, so the highest valid seed depends on
       the source's length. seedvr2x refuses an out-of-range seed before loading the model.
   - one window per DiT call. Batching windows breaks `na.unconcat_coalesce` when their
     window counts differ, and the planner sizes a window to fill the memory anyway.

   [Measured](../research/docs/stitching.md) with `lab`: −80% boundary jump against independent
   batches, no softening, output 40.5–40.8 dB from the single-window result, about +3–9%
   compute (the VAE is most of the time).
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
    - The chroma is downsampled with zscale's bilinear, pinned, as `ffv1_out.py` does. That is
      the right kernel on its own merits: decimation wants a low-pass, not a sharp
      interpolator, which is why the decode, which interpolates, uses Catmull-Rom.

  Validated with the [`ffv1_out.py` wrap](../research/docs/output.md): bit-exact round trip,
  tags checked by ffprobe. Our writers give a `yuv420p10le` bit-identical to its output. They
  set every tag both on the frames (`setparams`) and on the encoder: ffmpeg n9 converts frames
  whose tags differ from the encoder's.
- **PNG** (16-bit) as an alternative, one directory per segment, frames numbered from
  `000000`. ffmpeg's PNG encoder writes the primaries and transfer as cICP, cHRM and gAMA
  chunks from the frame tags, and none for an untagged frame. So PNG copies the tags as the
  masters do.
- **Until assembly** (milestone 6):
  - `-o x.mkv` writes one FFV1 master; that needs a video file as input.
  - Any other `-o` is a directory of segments plus `manifest.json`. It is either new or empty,
    or an unfinished job's directory, which the same command resumes (see
    [Pause and resume](#pause-and-resume)). One seedvr2x at a time writes it, and anything
    else found in it is refused, so another run's files never reach what sptenc reads.
  - An `-o` with another video suffix (`.mp4`, `.mov`…) is refused.
- **Output segments**, the resume units of the output, listed in a manifest:
  - **Layout, by sptenc's rule.** The threshold picks the cuts, then the minimum segment
    length (5 s by default) merges each too-short segment into its shorter neighbour, on the
    frame grid. With scdet run as sptenc runs it, the same threshold and minimum give the same
    segments as sptenc. With a directory of segments as input, the output mirrors it instead.
    - The rule counts in frames: the minimum is rounded up (5 s is 120 frames at 24000/1001),
      a segment exactly that long is kept, and ties merge left.
    - Ported, and checked against sptenc's `FilterShortScenes` on 20,000 random cases, run
      against its Go code.
    - One known difference: sptenc ends the last segment at the container's duration rounded
      to a frame, where seedvr2x counts frames. They agree unless that duration is off by
      half a frame or more.
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
- **Host RAM** too. The process's peak, 16.5 GiB with 7B fp16, comes while the weights load,
  not during the shots, which stay flat (2.3–2.4 GiB resident over 6 shots). It is probably the
  weights file mapped while it is copied to the GPU; not measured further. BlockSwap's pinned
  host copies add their size. Both matter on hosts with little RAM.
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

Work is saved in resumable units; a stop loses only the unit in progress. Milestone 4 passed on
2026-10-03.

| Stage | Saved | Resume granularity |
|---|---|---|
| VAE encode | the shot's latent | per shot |
| DiT | each window's output | per window: an encoded shot resumes at its first missing window, its noise drawn again from the seed |
| VAE decode | the output segment (FFV1, or the user's `--segment-cmd`) | per output segment: its decode and write restart whole |

- **Where:** a directory output keeps the units in `<out>/resume/shot_<start>/` (`latent.pt`,
  `window_NNNN.pt`).
  - They are `torch.save` copies on the CPU, values and memory layout kept, since the noise
    depends on the latent's layout. They are loaded with `weights_only=True`, so a tampered
    file can't run code.
  - Each file is written as `.partial`, fsync'd, renamed, and only then recorded.
  - A latent is deleted once its windows are done, a shot's directory once its segment is
    finished, and `resume/` once the job is.
  - `-o x.mkv` keeps its units in memory, so it isn't resumable until assembly gives it
    segments. Ctrl-C works, but nothing is kept.
- **Order:** a segment's shots are all encoded and sampled before its decode and write, which
  reads their windows back, so units never interleave. The first frames come out later.
- **Manifest** (`manifest.json`, version 2), the truth:
  - settings, with the SHA-256 of the models and of seedvr2x's own files. Any code change
    refuses a resume, even a comment.
  - environment: torch, CUDA, cuDNN, GPU, attention backend, FlashAttention, ffmpeg, and a
    fingerprint of the conversion chain (a hash of the startup check's test conversions,
    which catches a zimg upgrade that ffmpeg's version string doesn't show). The NVIDIA
    driver is recorded for information: the math kernels ship with torch.
  - inputs: path, bytes, mtime, SHA-256 of the content, and the first pass's facts
  - output, shots (windows, encoded, windows done), and segments (bytes when finished)

  It is rewritten whole after every unit: a temporary file, fsync, rename, then a directory
  fsync. A unit is recorded only once its file is whole.
- **Resuming:** the same command on the same `-o` directory, with no `--resume` flag.
  - Refused, with each difference listed: any difference in settings, inputs (by content),
    models or code, except progress. A moved input with the same content is accepted, since
    the path is information.
  - Environment differences are refused too, unless `--accept-env-change` is given, which the
    manifest records. A different GPU, stack or ffmpeg only breaks bit-identity across the
    resume, not the upscale's correctness, and refusing outright would throw away tens of
    GPU hours over an upgrade.
  - With the content, environment and settings unchanged, the first pass's record is trusted
    and the pass isn't run again. That saves a full decode per resume, about 17 min on a
    2-hour HEVC master.
  - The directory must hold what the manifest names and nothing of anyone else's, dotfiles
    included: anything else is refused, never deleted.
  - These are discarded: `.partial` files, an unfinished segment's file, unrecorded units, and
    the units of finished segments.
  - Finished segments and kept units are skipped. The input frames of skipped shots are
    decoded and dropped (decode-and-count), until frame-exact seeking is settled.
  - One seedvr2x at a time per directory (flock); a filesystem without locks is warned about.
    A filesystem that can't sync a directory is warned about once.
  - `--dump-frames` is refused inside the output directory.
- **Stopping:**
  - Ctrl-C once finishes the unit in progress and stops before the next (exit 130).
  - Twice, or SIGTERM, stops at once (exit 130, or 143 for SIGTERM).
  - ffmpeg runs in its own process groups, out of the terminal's reach.
  - The run polls the GPU (every 10 ms) before device-to-host copies: Python can't run a
    signal handler during a blocking CUDA copy (a decode slice takes ~8 s at 1080p), and two
    presses would merge into one.
  - `--until HH:MM` stops cleanly before a unit that wouldn't finish in time, using the
    planner's time estimates; it comes with the planner.
- Exiting frees the GPU entirely.
- **Resume is bit-identical** to an uninterrupted run: deterministic noise per shot, exact
  latents, deterministic attention (FA2 reruns are bit-identical). Checked on 2026-10-03 on
  milestone 1's input in 3 shots (one in 3 windows) and 2 segments:
  - stopped by a kill in an encode, a kill in a window, Ctrl-C once in a window, and twice, a
    kill and once in a segment's decode, then resumed after each
  - every segment equals the uninterrupted run's bit for bit, and every stopped process exits
    and leaves the GPU
- **Later:**
  - Resuming a segment's decode mid-way. Warm-up latents make it bit-identical from 37 on
    ([decode-resume.md](../research/docs/decode-resume.md)), but that's about 6 s of
    frames, so it only pays off on long segments. `--until` avoids most of the loss anyway.
  - The same manifest lets separate processes take separate shot ranges (several GPUs or
    machines).

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

The numbers name the checks, not the build order. After milestones 1 and 2 came the I/O layer:
- decode, writers, cut-list and directory input, output segments and the manifest
- checked by the conversion tests (see [Input](#input)), frame counts, and per-shot identity

Every later milestone needs it: resume needs segments and a manifest, assembly needs decode and
writers, and the planner needs real shot lengths.

**The I/O layer passed on 2026-10-03.** The milestone-1 regression is still bit-identical.
- Writers: `yuv420p10le` bit-identical to `ffv1_out.py`'s; white at 940, black at 64, chroma
  sited left; tags as declared.
- Shots: each shot of a cut-list job is identical to the same shot run alone, from its own
  frames, with its seed.
- Display aspect: 720×480 at 16:9 gives 1920×1080.
- Directory input: the output mirrors names and frame counts, and equals the one file cut at
  the joins, frame for frame.
- Segments: joined, they equal the one-file output, byte for byte. The merge rule is
  identical to sptenc's on 20,000 random cases.
- Long real segment: 377 frames at 1080p, 6 shots including 1- and 2-frame ones, run at 540p
  with window 8. Every frame came out, resident RAM stayed flat at 2.3–2.4 GiB, and VRAM
  peaked at 19–26 GiB.

**The build order:**
1. Resume (milestone 4): passed on 2026-10-03. Real episodes needed it: at 4.4 s per 1080p
   frame ([stitching.md](../research/docs/stitching.md#cost-model)), a 24-minute episode
   takes about 40 GPU hours.
2. The `lab` rewrite (milestone 5), next. Without it the output keeps the model's colour
   drift.
3. The planner, BlockSwap and tiling (milestone 3), then `--until`. On the 96 GB card at
   1080p, windows and the streamed decode already bound memory, and the planner's inputs (4K
   limits, the margin) come from the measurement campaign.
4. Assembly and `--segment-cmd` (milestone 6), for the regular workflow. The manual sptenc
   workflow already works without it.

1. **Reproduce numz.** Passed on 2026-10-02: 45 of 45 frames bit-identical, on the FFV1
   masters and on the float32 dumps. Same settings: one batch, no tiling, `flash_attn_2`, same
   seed, numz's input preparation, colour correction off (numz's `lab` runs the StableSR code
   we don't vendor).
   - Both sides read the same RGB frames, from an 8-bit RGB copy of the sample. numz decodes
     with OpenCV at 8 bits (`inference_cli.py:469`, `/ 255` at `:336`), which our 16-bit ffmpeg
     decode can't match bit for bit. Our decode is validated by the conversion tests instead
     (see [Input](#input)).
   - The result must be bit-identical, or within quantisation, to numz's float frames, checked
     with the FFV1 masters.
   - The numerics choices (RoPE precision, attention dtype, VAE mode vs sample, input
     preparation) are measured beforehand on numz, with patches, and applied once this
     milestone passes.
2. **Stitching.** Passed on 2026-10-02, with colour correction off. It isn't implemented yet,
   and `lab` cuts the remaining low-frequency jump about 3×
   ([stitching.md](../research/docs/stitching.md)), so the study's figures don't carry over.
   - Material: an 8-bit RGB copy of clip B's frames 20–100, against new numz runs: one batch
     of 81, independent batches of 21, and `STITCH_LATENT` 6:2. The windows are latents 0–6,
     4–10, 8–14, 12–18 and 16–21.
   - Our windowing, mixing and decode, run in an internal per-window reseed mode (tests only,
     not user-facing), reproduce numz + `STITCH_LATENT` bit for bit.
   - The sliced noise against that reseed mode, both measured against the one-batch run:

     | Metric | Reseed | Sliced |
     |---|---|---|
     | PSNR, mean | 40.14 dB | 41.91 dB |
     | PSNR, worst frame | 39.20 dB | 40.42 dB |
     | Hold excess step | 0.170 | 0.171 |
     | Low-frequency excess step | 0.050 | 0.069 |
     | Hold excess total | 0.185 | 0.297 |
     | Low-frequency excess total | −0.197 | 0.023 |
     | Against independent batches: hold step | −85% | −85% |
     | Against independent batches: low-frequency step | −93% | −91% |

     - Sliced's PSNR is better at every offset to the boundaries.
     - The excess figures are in 8-bit levels.
     - Steps read lower-is-better. Totals are signed sums and read closer-to-0-is-better: a
       negative total means the boundary zone changes less than the one-batch run's, which is
       a deviation too. The reseed mode is lower on the hold total partly through such a
       boundary.
     - Sliced is kept on the dB, which is the closeness to the seamless run; the differences
       left are fractions of a level. One clip and four boundaries: the visual review of
       milestone 7 covers the rest.
   - The single-window case stays bit-identical to milestone 1.
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

  seedvr2x's side is settled: it reads a cut list of frame numbers (see [Input](#input)), so
  sptenc's export can write that, with scores and the cuts the merge removed as extra fields.
  Open: how seedvr2x treats cuts inside a segment and doubtful joins.
- **Colour correction's input frames at decode.** `lab` compares each output frame with its
  input. The streamed pipeline has passed the input by the time a shot decodes: only the
  latents are kept. Three ways:
  - re-read the shot from the source, which needs the frame-exact access below
  - keep a temporary lossless copy of each shot's decoded input, written while the encode
    reads it: exact, sequential, and on disk at input resolution. A segment's decode comes
    after all its shots' windows, so the copies of a whole segment's shots coexist, which
    is acceptable.
  - keep only what the rewritten `lab` needs (its low frequencies and LAB statistics),
    computed at read time

  Recommendation: the temporary copy. It is the simplest, it's exact, and it doesn't depend on
  seeking. It stays part of the shot's resumable state until its decode is done. To confirm at
  the `lab` checkpoint, with the seeking results.
- **Frame-exact access into long-GOP sources**, to resume a shot and to read its input frames
  again for colour correction. Three ways:
  - ffmpeg's accurate seek: fast, but trusts timestamps
  - decoding from the start and counting: exact, but slow on a film. Resume uses it for now.
  - a lossless intermediate: exact and fast, but large

  Plan: measure accurate seek against decode-and-count on real long-GOP files.

### Scope
- **Model scope for v1:** 7B fp16 only, or fp8/Q4 and 3B from the start (needed for small
  GPUs).

### To measure
- **Numerics:** numz's or ByteDance's, change by change, once it's confirmed which ones really
  differ on the 7B fp16 path (see [Vendored model code](#vendored-model-code)):
  - the RoPE angle precision
  - the fp32 → fp16 → bf16 cast of input frames
  - the fp16 weights cast to bf16
  - VAE mode vs sample
  - the chroma kernels: Catmull-Rom upsampling at decode, bilinear downsampling for the
    `yuv420p10le` master
  - the input preparation: the resize kernel and its `antialias` flag, and multiples of 16
    reached by padding (numz) or cropping (ByteDance)
  - a shot padded to 4n + 1 frames by mirroring its end (numz) or repeating its last frame
    (ByteDance)
- **Decode resume granularity:** whether sub-segment decoding with warm-up latents is
  bit-identical.
- **4K and long windows:** the planner's limits on large outputs, where the DiT window is the
  constraint.

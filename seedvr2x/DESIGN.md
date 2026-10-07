# seedvr2x design

> Status: **being built.** It holds the decisions taken and the questions still open, and the
> code follows it: milestones 1, 2, 4 and 5 and the I/O layer have passed (see
> [Validation milestones](#validation-milestones)). Each decision links to the measurement it
> rests on in [../research/](../research/).

## Goal

A SeedVR2 video upscaler for **long runs** (whole episodes or films) that:
- produces a lossless, correctly tagged master that the next encoding step can trust
- looks redrawn, not merely enlarged: crisper lines and text, cleaner colours than a plain
  upscale, the source's style kept. The user's eyes judge the look. Full-reference metrics,
  which reward a copy of the ground truth's own blur and grain, guard against damage: colour
  drift, flicker, banding, artefacts
  ([colour.md](../research/docs/colour.md#the-users-eyes)).
- has no visible seams inside a shot
- fits the GPU it runs on without the user tuning memory options. It is developed on a 96 GB
  card, but made for consumer cards of 16–32 GB as well, where the planner, BlockSwap and
  tiling carry the run:
  - At 1080p, without BlockSwap, a 24 GB card holds windows of about 6 latents and a 32 GB one
    13, so many shots are split into windows.
  - A 16 GB card can't hold the 7B fp16's weights (15.35 GiB) at all: it swaps every block,
    and phase 2's smaller files lighten it (see [Weights](#weights)).
  - Below 48 GB, the 1080p VAE decode only fits tiled. At 4K, every card tiles the decode,
    the 96 GB one included, and that one's windows hold about 19 latents.
  - These cards also compute more slowly (an RTX 5080 has 84 SMs, the 96 GB card 188), so a
    job lasts longer there, and resume and `--until` matter most.
- can be paused and resumed (run at night, give the computer back in the morning)

v1 is for SDR sources, made as perfect as they can be: the user's priority, SDR being what gets
upscaled today. HDR comes in a later version (see
[Not in the first version](#not-in-the-first-version)).

It replaces numz's orchestration and I/O (about 14.6k lines, where nearly all of the
[26 bugs](../research/bugs/README.md) live). It keeps ByteDance's model code.

### Two kinds of users, both first-class
- **Standalone, the regular workflow:** a video file in, an upscaled file out, with ffmpeg as
  the only other tool. seedvr2x finds the cuts itself and assembles the finished file, the
  source's audio, subtitles and chapters included. File size matters here, hence
  `--segment-cmd` (see [Output](#output)).
- **With sptenc**, the first use case and the reason for this project. This is the manual
  workflow: the source in, a directory of segments out, then
  `sptenc encode <dir> <out> -f <source>`, which encodes the segments as they are and takes the
  audio and subtitles back from the source. It stays lossless up to sptenc's encode, and its
  users bring the disk space.
  - seedvr2x takes the source itself because shot detection is the biggest quality lever
    outside the model ([cuts.md](../research/docs/cuts.md)). sptenc's split merges scenes
    shorter than 5 s into their neighbours, which suits encoding but hides real cuts inside its
    segments, and a cut the model doesn't see is its costliest error. seedvr2x keeps every cut
    for the model's shots and merges only its output segments, by sptenc's rule.
  - The source is seedvr2x's only input, in both workflows (the user's decision, 2026-10-05):
    a directory of segments from an outside splitter is no input (see [Input](#input)).

Both workflows start from the source and differ only at the end: seedvr2x's assembly, or
sptenc's encode.

sptenc shapes the interfaces (segment directories, cut lists, master formats), but seedvr2x
never requires it.

### Not in the first version
- Image inputs, ComfyUI nodes, macOS/MPS, AMD/ROCm
- Multi-GPU in one process (separate processes on separate shot ranges can come later, see
  [Pause and resume](#pause-and-resume))
- Options that only work around numz's own design (see [Options](#options-kept-and-dropped))
- Models other than the 7B fp16: phase 2, after v1 (see [Weights](#weights))
- Variable frame rate sources: refused with a clear message, as sptenc does
- Sources refused in v1, each with a clear message:
  - HDR: the PQ (`smpte2084`) and HLG (`arib-std-b67`) transfers. The model was trained on SDR
    video: what PQ-coded pixels passed through untouched would give is unmeasured, and HDR10's
    metadata (mastering display, MaxCLL) isn't carried. The message says how to get an SDR
    upscale: tone-map the source to SDR first, a grading choice for the user's own tools, as a
    gamut conversion is.
    - Dolby Vision is read through its base layer: accepted when that layer is SDR, refused
      otherwise, profile 5's included, which isn't viewable without Dolby's processing.
    - The later version measures passthrough against tone mapping first (2 HDR clips at
      1080p, about 1 GPU hour) and carries HDR10's metadata.
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

## Terms

- **Frame:** one picture of the video.
- **Latent:** the VAE's compressed form of frames. The first latent of a sequence holds 1
  frame and every next one holds 4, so a sequence of 4n + 1 frames makes n + 1 latents.
- **Shot:** the frames from one cut to the next. Nothing crosses a cut: each shot is encoded,
  upscaled and decoded on its own.
- **Window** (DiT window): a run of consecutive latents of one shot that the DiT processes in
  one pass.
  - A shot that fits is one window. A longer one is split into windows of balanced lengths that
    overlap by 2 latents (8 frames), and the overlapping latents are mixed before decoding, so
    the joins don't show.
  - The length comes from the GPU's memory, through the per-token model in
    [vram.md](../research/docs/vram.md): 16 GiB of weights plus about 1 GiB per latent at
    1080p with the 7B fp16 model, and four times that per latent at 4K.
    - At 1080p: about 6 latents (21–24 frames) on a 24 GB card, 13 (49 frames) on a 32 GB
      card, 78 (309 frames) on a 96 GB card.
    - At 4K: 19 latents (73 frames) on a 96 GB card.
  - A window bounds the DiT only. The VAE's memory depends on the frame size, not on the
    window, since it streams in 4-frame slices (flat beyond ~9 frames). At 4K the untiled
    decode needs ≈ 134 GiB, so 4K needs tiled decoding, about 18 GiB with 1024-px tiles. The
    planner sizes the two separately: window length for the DiT, tile size for the VAE.
  - Not to be confused with the DiT's **attention windows**: the space-time tiles its
    attention works in inside one pass ([attention.md](../research/docs/attention.md)), which
    `na.unconcat_coalesce` counts.
- **Output segment:** an output file. It holds one or more whole shots, at least 5 s long by
  sptenc's rule. It is the unit of output and of decode resume.
- **Unit:** a piece of saved work for resume: a shot's encode, a window, a segment's decode.

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
- **With sptenc:** seedvr2x → `sptenc encode <dir> <out> -f <source>`. sptenc encodes a
  directory of segments as it is given, RGB ones included, "such as an upscaler delivers"
  (its MANUAL). In that chain, seedvr2x does the scene detection, and sptenc the encodes, VMAF,
  concat and the final mux.
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
    of 16. numz chains them at `src/core/generation_utils.py:73-81`. seedvr2x pads its own
    way (see step 0).
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

  Measured since ([numerics.md](../research/docs/numerics.md)): none of ByteDance's choices
  moves the output beyond the spread between seeds, alone or all together (float32 RoPE angles,
  a float32 input chain, the fp32 weights, posterior sampling, bf16 DiT norms, the decode under
  bf16 autocast, fp16 attention), on 4 animated clips and every metric (PSNR-Y within 0.14 dB,
  VMAF within 1.1). So numz's numerics stay, and with them bit-identity with milestone 1. The
  exception is the padding to multiples of 16, which measured better done otherwise (step 0 of
  the [Pipeline](#pipeline-per-shot)).

  The decode stays in bf16 too. bf16 bands nowhere: CAMBI gives every output 0.004 at most, the
  model's rendering dithering bf16's steps. A float16 decode brings the raw output's
  low-frequency colour closer to the ground truth (ΔE00 −0.05 to −0.09 on 8 of 8 clips, for
  4.4% more decode time), but colour correction replaces that band with the input's: after it,
  float16 moves ΔE00 by 0.01 at most, 0.002 after a 4 px blur, within the spread between seeds
  on 5 to 8 of the 8 clips for every score
  ([colour.md](../research/docs/colour.md#a-float16-decode)). So no float16 decode, and no
  overflow fallback to go with it.
- Licence: Apache-2.0. We keep the copyright headers, add a NOTICE, and mark modified files.
  The StableSR-derived colour code (`color_fix.py`, non-commercial licence) is **not**
  vendored: see [Colour correction](#colour-correction).

### Weights
v1 runs ByteDance's sharp 7B by default, and the regular 7B, both DiTs in fp16 with the fp16
VAE, Apache-2.0 (seedvr2x's own files: see the Hugging Face repo below). Every other model is
phase 2's (below).
- **The sharp 7B is the default, on the user's eyes** (2026-10-07): on 75 windows of 4K and
  1080p crops they preferred it or saw no difference on 67 (it 27, the 7B 8), the grainy first
  film's skin no longer "almost reptilian". At 4K it invents less of the fine texture the eyes
  rejected with the 7B: 2.6–5.7 times the ground truth's fine detail per kind of source on 13
  shots, against the 7B's 3.4–7.3, the same kind of texture, and on the first film 1.0 against
  2.0. No guard gets worse: the same banding and colour after the split, fewer colour fringes
  and less flicker. At ×2 to 1080p the metrics barely tell them apart
  ([colour.md](../research/docs/colour.md#step-6-the-sharp-7b-for-the-eyes)). It has the 7B's
  architecture, memory and time.
- **The regular 7B stays the reference** milestone 1 is checked with: numz's 7B fp16 file, and
  ours equal to it.
- **Recognised by content.** A model file is checked by its own tensors, not by its name, which
  is what numz goes by (`src/core/model_configuration.py:717-719`). Anything but the 7B in fp16
  is refused, saying what the file is.
- **Small cards** rely on BlockSwap and tiling (milestone 3): the 7B fp16 DiT's weights alone
  take 15.35 GiB. Milestone 3 measures how small a card that reaches, 16 GB being the aim:
  there every block is swapped, its weights in pinned host RAM (about 15 GiB of it). numz's
  Q4_K_M, every block swapped, ran 1080p in batches of 13 latents on an emulated 16 GB card,
  and of 5 on an 8 GB one ([vram.md](../research/docs/vram.md#recipe-per-card-size-validated)):
  phase 2's smaller files make small cards lighter on memory, host RAM and transfers.

#### Phase 2: the other models
A phase of its own, after v1, settles which models to keep, which to make again and how, how
they differ and by which metrics, and how users are guided to them. What is known already:
- **numz's registry**, Apache-2.0: the 3B and the 7B in fp16, fp8 (`e4m3fn`; the 7B's file
  keeps its last block in fp16) and GGUF (Q4_K_M, and Q8_0 for the 3B), and the sharp 7B in
  fp16, fp8 and Q4_K_M, all with the one VAE. Its default is the 3B fp8
  (`src/utils/model_registry.py:56`).
  - Running them takes numz's fp8 conversions for arithmetic (its `CompatibleDiT` wrapper,
    `src/optimization/compatibility.py`), its GGUF dequantisation (`gguf_dequant.py` and
    `gguf_ops.py` there, adapted from city96's ComfyUI-GGUF, Apache-2.0) with the `gguf`
    library (llama.cpp's, MIT), the 3B's cache quirk, and the planner's constants per model
    ([vram.md](../research/docs/vram.md)).
- **Comfy-Org's own repo** (`Comfy-Org/SeedVR2` on Hugging Face, apache-2.0, "repackaged
  model files for ComfyUI") ships the 3B, 7B and sharp 7B in fp16, fp8 (a plain cast, like
  numz's), MXFP8, INT8 W8A8 with a rotation, and NVFP4 (the 7B 4.76 GB); its VAE is numz's file.
  It is made by a GPL converter, never read.
- **numz's fp8 files are plain casts.** Every tensor of the 3B's, and all but the last block's
  in the 7B's, is rounded to `e4m3fn` (3 mantissa bits), with no scale, down to the biases,
  norms, modulation tables, input and output layers and RoPE's frequencies. The 7B's file is
  ByteDance's fp32 master cast straight to `e4m3fn`, bit for bit, not numz's fp16 file cast
  again, which would differ on 0.28% of the values. The 3B's is cast the same way, from
  ByteDance's first 3B weights: ByteDance replaced the 3B's master on 2025-06-22, and numz's 3B
  files, fp16 included, are the earlier one's (635 of 635 tensors). Every 3B figure in
  models.md ran those first weights ([FORMATS.md](../models/FORMATS.md)). The Q4_K_M file
  quantises only the 288 attention and MLP matrices of the blocks, 8 per block, 99% of the
  weights, with a scale every 32 weights. Its other 840 tensors, the 6 matrices outside the
  blocks included, are the fp16 file's byte for byte
  ([models.md](../research/docs/models.md)). That is why it measures closer to the source than
  fp8: a cast without a scale puts a median 36% of a matrix's weights in E4M3's subnormal range,
  all of the timestep embedding's (15.5% error per weight) and 70–100% of the last blocks' text
  matrices' (up to 16.7%), where one scale per tensor gives 2.65%. One of those 6,
  `emb_in.proj_out` (56.6M values), is larger than any matrix it quantises.
- **RoPE's frequencies are constants of the architecture,** never trained, yet the fp8 files
  hold them rounded: up to 6% off in the 7B's blocks 0–34, and in the 3B's, the 5 lowest of 21
  at zero and others up to 41% off. numz runs the 7B's file on block 35's fp16 values in every
  block anyway, through [bug 24](../research/bugs/24-rope-wrapper-late-binding.md). Any file
  seedvr2x runs takes the fp16 files' values.
- **What the measurements found**
  ([models.md](../research/docs/models.md): 4 animated clips, a ×2 upscale of a mild
  degradation, against the ground truth):
  - Q4_K_M is as close to the source as fp16, or closer (PSNR-Y +0.11 to +0.49 dB, one seed),
    with a DiT peak of 17.2 GiB against 28.2 at 45 frames of 1080p, for 2–3% more DiT time.
  - fp8 is further from the source on all 4 clips (PSNR-Y −0.25 to −0.58 dB), takes more memory
    than Q4_K_M (20.7 GiB) and saves only 2–3% of DiT time: its weights are widened for the
    arithmetic, so they save memory, not time.
  - The 3B is a different model. In fp16 it is perceptually worse than the 7B on all 4 clips
    (LPIPS and DISTS), and its fp8 file stays close to it. Its DiT is 25–38% faster, but the VAE
    takes most of the run (15 s saved out of 215), and it needs as much memory as the 7B's
    Q4_K_M (17.2 GiB in fp8) or more (20.3 GiB in fp16).
  - The sharp 7B is the closest of all to the source: PSNR-Y +0.40 to +0.67 dB on all 4 clips,
    its seeds apart from 7B fp16's, VMAF up by 1.9 to 3.5, and less colour error. Despite its
    name, it adds no more fine texture than 7B fp16. ByteDance publishes it beside the regular
    7B (`seedvr2_ema_7b_sharp.pth`), without a word about it in either readme.
  - A quantized file moves the output no more than a change of seed does: 40–48 dB from its
    parent at the same seed, against 37–44 dB between two seeds of the 7B fp16 (about equal on
    one clip). A distance to the 7B fp16 alone would overstate what a smaller file loses, so
    each figure is read against the spread between seeds.
  - Fidelity is not quality: the model re-renders, so closer to the source can mean redrawing
    less. Live action and blurrier inputs aren't measured.
- **Sources: ByteDance's fp32 releases,** the masters: `seedvr2_ema_7b.pth` and
  `seedvr2_ema_7b_sharp.pth` (33 GB each), `seedvr2_ema_3b.pth` (13.6 GB), `ema_vae.pth`.
  numz's files are 16-bit, so a file made from them is rounded twice.
  - The fp16 rounding is small, about 0.05%, and only matters where it tips a value over a
    rounding boundary of the smaller format. For the reference itself, the 7B's fp32 weights in
    place of numz's fp16 file move the output by 0.38 8-bit levels on average, against 1.90
    for another seed: within the seed spread on every metric
    ([numerics.md](../research/docs/numerics.md#results)).
  - Converting from the master costs nothing more, so every file seedvr2x makes starts there.
- **Files of our own,** made by `models/`'s scripts from the fp32 masters, the small tensors and
  RoPE's frequencies kept in 16 bits, as in the Q4_K_M file. They are prepared ahead of phase 2,
  on the CPU (the user's request, 2026-10-06), validated by GPU runs before implementation's
  next steps, and uploaded with v1's files only if within the 7B fp16's seed spread on every
  kind of source, the user's eyes agreeing:
  - Made, for the 7B and the sharp 7B, reproducible byte for byte, the 840 tensors outside the
    288 block matrices our fp16 file's ([FORMATS.md](../models/FORMATS.md); error per weight,
    median and worst):
  - fp8: the 288 block matrices in `e4m3fn` with one float32 scale per tensor (max |W| / 448),
    in comfy-kitchen's layout (weight and weight scale): 2.65% (2.67%); a scale per row gains
    nothing (2.64%). The activations get their scale at run time: multiplied in fp8 (W8A8) from
    the RTX 40 generation on, widened to 16 bits (W8A16) before it.
  - GGUF, made by ggml's own quantize function (llama.cpp's, MIT) and written by gguf-py in
    city96's conventions, which numz's loader reads: Q4_K on the 288 block matrices (numz's
    Q4_K_M layout, 7.35% for both) and Q8_0 (0.56%), everything else in 16 bits. numz decodes
    them in float16, which rounds Q4_K's values twice: seedvr2x decodes in float32. Then a
    dynamic file, each tensor's type chosen from a sensitivity scan on the GPU (as Unsloth's
    method chooses from a calibration set), kept only if it beats the static ones.
  - NVFP4 for Blackwell, 4-bit weights and activations, in comfy-kitchen's layout (a scale
    every 16 values), each block's scale chosen among 8 to minimise its error: 8.80% (8.94%),
    against 9.45% with comfy-kitchen's own scales, lower on every matrix. At 4 bits, Q4_K stays
    closer per weight: NVFP4's case is its 4-bit multiply on Blackwell, its 4-bit activations
    the risk.
  - INT8 W8A8 with comfy-kitchen's rotation, the fast path on the RTX 20 and 30 generations,
    where fp8 can't multiply: each 256 input columns of a matrix rotated by a Hadamard matrix,
    its own inverse, one int8 scale per row; at run time comfy-kitchen rotates and quantizes
    each input the same way. 0.86% (1.08%), against 1.01% unrotated.
  - The weights need no data: their scales come from their own values. Activations quantized
    to 4 bits are where quality can go, and on video DiTs mostly because their ranges drift
    across the denoising steps; SeedVR2 runs a single step. Should a calibration set be needed,
    the full-reference clips of each kind of source can serve.
  - Kernels: comfy-kitchen (Comfy-Org, Apache-2.0) quantizes and multiplies FP8 (natively from
    sm_89), NVFP4 (Blackwell), MXFP8 (from sm_100) and INT8 W8A8 (from sm_75), its activations
    quantized on the fly, but converts no model. ComfyUI's own code is GPL-3.0, never copied.
  - The gain is mostly memory. ComfyUI reports about 2× over fp8 or bf16 on Blackwell, but the
    DiT is about a fifth of a 1080p run here: a DiT twice as fast shortens a job by about 10%.
- **One Hugging Face repo, seedvr2x's own, from v1 on**,
  [`hekmon/seedvr2x`](https://huggingface.co/hekmon/seedvr2x) (the user's plan, 2026-10-05),
  holds every file seedvr2x runs, each made by a documented script: each file's origin can be
  checked, and nothing depends on another party's repo staying as it is. Apache-2.0 allows it,
  with the licence and a notice of the changes; TransNetV2's weights are MIT.
  - v1 seeds it with the 7B and sharp 7B fp16 DiTs and the fp16 VAE, rounded from ByteDance's
    fp32 masters (`ByteDance-Seed/SeedVR2-7B` at `eb0c428`), and TransNetV2's weights (see
    [Shot detection](#shot-detection)): `seedvr2x_ema_7b_fp16.safetensors`,
    `seedvr2x_ema_7b_sharp_fp16.safetensors`, `seedvr2x_ema_vae_fp16.safetensors` and
    `transnetv2.safetensors`. They go up in one upload with phase 2's own files, once these are
    validated, and the card's table of every file (the user's choice, 2026-10-06).
    - The names aren't numz's: numz's downloader deletes a file named like one of its own whose
      SHA-256 differs (`src/utils/downloads.py:216-235`), and ours differ by their header, so
      a model directory shared with numz would lose them.
  - `models/`, at the repository's root, holds the scripts that make them. Each records its
    inputs (URL, revision and SHA-256) and its outputs' SHA-256, and runs its check: our fp16
    files equal numz's element for element; TransNetV2 in PyTorch matches its TensorFlow
    original. A run from the masters gives the same bytes: the files are written in the
    safetensors layout by our own writer, since the library writes the metadata in a random
    order. The scripts run on
    their own, with inline dependencies (TensorFlow only in TransNetV2's conversion), so
    seedvr2x stays the repository's only uv project.
  - Every file is safetensors, TransNetV2's PyTorch conversion included: no pickle, and a
    header the model check reads without torch.
  - The files are made in `models/dist/`, which git ignores, and the user uploads that
    directory as it is (`hf upload`, from the GPU box), with a list of each file's SHA-256 and
    a model card giving the licences and how each file was made.
  - Licences: the repo is `apache-2.0`, as ByteDance's SeedVR2 repos and seedvr2x are, with
    the Apache-2.0 text and a NOTICE at its root. The NOTICE names each SeedVR2 file's origin
    (ByteDance's file and revision) and its change (rounded to fp16; later, the quantization),
    as Apache-2.0 asks of modified files, and each file's safetensors metadata says so too.
    TransNetV2's file keeps its MIT licence, the notice beside it. The card says the files are
    unofficial conversions, not ByteDance's.
  - seedvr2x pulls its files from that repo at a revision pinned in its code, each checked by a
    SHA-256 pinned there too, so a version always runs the same bytes. A local directory
    holding the same files works offline.
  - Our fp16 changes no bit of v1's output. numz's 7B and sharp 7B fp16 DiTs and its fp16 VAE
    are the masters rounded to the nearest fp16, ties to even: every element of 1,128, 1,128
    and 250 tensors, under the same names, the data sections byte for byte ours (the files
    differ in their headers' metadata alone). Truncation would differ on half the elements, a
    cast through bf16 on 87%. numz's 3B fp16 is ByteDance's first 3B master rounded the same
    way.
- **The VAE stays in 16 bits.** Quantizing a model means one of two things, and neither suits
  the VAE:
  - The weights alone, stored in 8 or 4 bits and widened back for the arithmetic. The VAE's
    take 0.47 GiB, so that saves nothing, and widening gains no time: it is why numz's fp8 DiT
    is only 2–3% faster than fp16.
  - The activations too, the values the model computes, so that the arithmetic itself runs in
    8 or 4 bits: that is where the speed comes from.
    - The VAE's work is 3D convolutions (128 to 512 channels, one attention block at the
      smallest size). PyTorch has no fp8 or fp4 convolution (`torch._scaled_mm` multiplies
      matrices), and comfy-kitchen's kernels are matrix multiplies too.
    - In the decoder's last stages, the activations are the picture taking shape, so their
      precision becomes the picture's. In bf16, 8 significant bits, near-white values already
      sit on 8-bit steps ([numerics.md](../research/docs/numerics.md#vae-decode-precision));
      fp8 keeps 4 significant bits, NVFP4 2. The latent itself is untouched: its rendering
      into pixels is what would lose precision. In the encoder, the latent itself would.
  - The VAE's memory is activations, about 34 GiB for an untiled 1080p decode: tiling is the
    lever there.
  - The VAE is about four fifths of a 1080p run, so its speed matters most. Its levers are
    `compile_vae` (−16 to −19% of VAE time, see [Memory planner](#memory-planner)) and,
    unmeasured, the convolutions' memory layout.
- **What a smaller DiT file buys** is VRAM, for longer windows (fewer joins) and, on small
  cards, less BlockSwap; time only through quantized activations. BlockSwap already gives the
  fp16 weights the same windows, for host RAM and transfers its prefetch should mostly hide.
  So a smaller file is weighed against the 7B fp16 with BlockSwap, at the same window length.
  On the 96 GB card at 1080p, windows are already longer than most shots: the gain is on small
  and mid-sized cards, and at 4K.
- **Checks:** a file kept from numz is bit-identical to numz on milestone 1's input, the 3B fp8
  against numz patched to use the fp16 RoPE values; a file of our own is within the 7B fp16's
  seed spread on every kind of source.
- **Guiding users:** a table of the files, on the Hugging Face card and in seedvr2x's README
  alike (the user's requirement, 2026-10-06: "to help users choose in their right mind"):
  each file's size, how it multiplies on each GPU generation (W8A8, W8A16, W4A4, W4A16), where
  it is faster and what that buys (the DiT is about a fifth of a 1080p run), and its measured
  quality against the 7B fp16's seed spread. A second table sets other repositories' files
  (numz's, Comfy-Org's) against ours, measured the same way. Then the docs, per kind of source
  (anime, dark, live action…), each figure read against the spread between seeds; `--plan`
  showing what each model gets on the user's own card (window length, BlockSwap, tiles, time);
  and whether the planner may pick a smaller file when the user allows it.

## Input

One input, a video file, made into a list of shots: a source as it is (any codec ffmpeg
decodes, no intermediate master needed) or a lossless master. The cuts come from seedvr2x's own
detection by default (see [Shot detection](#shot-detection)), run as a first pass before the
model's work so the planner knows every shot (time estimate, `--until`, manifest). Or they come
from a cut list.
- The cut list format: one frame number per line, the first frame of each shot except the
  first, `#` comments, no timestamps. Frame numbers are exact on the frame grid.
- Fields after the frame number are ignored, so an export carrying scores (sptenc's, once it
  has one) stays readable.
- A directory of segments is no input (the user's decision, 2026-10-05). The source is the only
  gateway, so every cut and every output segment comes from seedvr2x's own detection and
  rules, never from an outside splitter's joins: of an earlier split's 411 joins, 63 scored
  under scdet's threshold of 10 and 14 under 4
  ([scene-detection.md](../research/docs/scene-detection.md)). Segments split losslessly can
  be joined back into one file.

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
  decodes every frame before the model's work and refuses the source when its shortest and
  longest frame durations differ by more than 1 ms, sptenc's rule. The declared rate must
  also match the measured durations, since the output is written at that rate.
  - Comparing declared rates alone fails on Matroska: sptenc found a 24/30 fps mix declared
    24/1 for both.
  - A file whose timestamps contradict its declared rate is a bad file: refused like any
    other, never reinterpreted on seedvr2x's own (the user's decision). The case measured: a
    Blu-ray remux declared 24/1 whose frames step at 24 fps, with one frame in 500 held 62–63
    ms instead of 42, which keeps every frame within ±10.9 ms of a steady 24000/1001
    timeline, its audio's ([seeking.md](../research/docs/seeking.md), mechanism 7).
  - The fix is the user's, and seedvr2x guides it. When the timestamps follow a constant rate
    within half a frame, as there, the message gives that rate, the largest deviation, and
    two fixes:
    - Remake the file: an FFV1 master at that rate, the other streams copied with it, a clean
      file for every tool. ffmpeg keeps every frame doing so: on that file's first 3,000
      frames, `fps=24000/1001`, `-fps_mode cfr -r 24000/1001` and a plain retiming all gave
      the source's frames in order, none dropped or doubled. It costs about 80 GB for that
      film (0.53–0.55 MB per frame, against the source's 14.5 GB) and a full encode.
    - Or `--frame-rate R`, an override like the matrix's and the sample aspect's: the frames
      are taken at R, accepted only if every one lies within half a frame of R's timeline,
      so none is dropped or doubled, and R is recorded as a setting. The same frames on the
      same timeline, without the intermediate file.
  - The pass costs one software decode: 171 fps on an HEVC master, 730–2,360 on 1080p H.264
    (a 26 Mbit/s Blu-ray: 735 on 16 threads, about 4 min for a 2-hour film), and 204 fps on
    16 threads, 315–326 on 48, on a 4K UHD HEVC remux: at least 19–30 min for a 4-hour film
    ([seeking.md](../research/docs/seeking.md),
    [scene-detection.md](../research/docs/scene-detection.md)). The shot detector and idet
    join it (see [Shot detection](#shot-detection)).
  - Software, not NVDEC (the user asked). The decode is about 0.03% of a run (1.4 ms per
    1080p frame against about 4 s of GPU time), seen only as the first pass's wait. The frame
    index's CRC-32s must match every later read: H.264, HEVC, AV1 and VP9 decode bit-exactly
    by their specs, but MPEG-2 and MPEG-4 Part 2 allow IDCT differences, so a hardware first
    pass could disagree with a software read. NVDEC also lacks FFV1 and some profiles on some
    generations, so a software path stays anyway.
- interlacing: refused when the field order is neither progressive nor unknown (sptenc's
  rule)
- frame-exact reading from any frame, through an index the first pass builds
  ([seeking.md](../research/docs/seeking.md)):
  - The first pass, which decodes every frame anyway, records each frame's timestamp, a CRC-32
    of the decoded picture and whether the decoder reported an error on it, plus the keyframes'
    timestamps from a packet scan. Frames are counted as they arrive, never from ffmpeg's own
    counters, which restart when it rebuilds its filter graph mid-stream.
  - Reading frames n to m (a resumed job's first unfinished shot, later another process's
    range) seeks to the keyframe one GOP before n's own, selects by timestamp, and checks every
    frame against the index. A mismatch starts again one keyframe further back, doubling; the
    last resort decodes from the start and counts.
  - Why every part: seeking to n / fps missed on 6 of 10 real sources, by up to 142 frames on
    a Blu-ray declaring 24/1. The exact timestamp lands late on open-GOP leading pictures, and
    keyframes that aren't valid entry points (6.7% and 8.3% on two Blu-ray remuxes) return
    wrong pictures with the right timestamps. One GOP early and selected by timestamp was exact
    on 538 of 540 targets outside one passage of dense keyframes, where the CRC-32 caught what
    it missed.
  - Cost: 0.04–0.23 s per read on DVD, Blu-ray and web H.264, 1.3 s median and 16 s worst on a
    long-GOP HEVC master, against 1.2–17 min to decode a 2-hour source to its end. The CRC-32
    takes 0.47 ms per 1080p frame, which keeps up with the first pass on one thread.
  - The index is written once with the first pass's record, beside the manifest, which names
    it by its SHA-256; a resume trusts it as it trusts that record.
  - Real open-GOP HEVC in MKV is tested (an x265 encode: one IDR, then 435 CRA, 54 leading
    pictures): frame-exact on 136 targets, 96 of them around every leading picture, at 0.21 s
    per read. A decode started exactly at a CRA outputs none of its leading pictures and says
    nothing, but every later frame is right: frames go missing, never come out wrong. Starting
    one GOP early avoids it, and the CRC-32s would catch it.
  - Real VOB, broadcast TS, open-GOP MPEG-2, open-GOP HEVC in MP4 and TS, and BLA pictures
    aren't tested yet. The check covers them: a wrong or missing picture fails its CRC-32, and
    the read falls back.

ffmpeg does every colour conversion, in and out: we pin its parameters rather than
reimplementing them. A startup check refuses a build without zscale or ffv1. Tests verify its
conversions: round trip, white at 940, black at 64, chroma siting.

Every zscale runs on one slice, with libavfilter's per-filter option `threads=1`, in the decode
and in the `yuv420p10le` writer alike
([numerics.md](../research/docs/numerics.md#the-masters-chroma-420-kernels-and-zscales-slices)).
By default, ffmpeg cuts zscale into slices, one per CPU the process may use, and each slice is
a zimg graph of its own, whose vertical chroma filter stops at the slice's edge:
- A `yuv420p10le` master then changes with the slice count, so with the machine: up to 1.7% of
  its chroma samples, by up to 13 ten-bit codes. Every count from 1 to 16 gave other bytes.
- A 10-bit 4:2:0 source reads off the exact conversion from 4 slices on: nearly every sample,
  by up to 0.57 level. An 8-bit source reads exactly at any count, which is why nothing so far
  showed it.

One slice costs 2.3–2.6 ms per 1080p frame, against about 4 s of GPU time. Reproduced with
seedvr2x's own chains on the 48-CPU box: with `threads=1`, 48 slices give the one-slice bytes,
both ways. A test holds the chains to that.

### Shot detection
Shot detection is the biggest quality lever outside the model, and what both workflows start
from (see [Two kinds of users](#two-kinds-of-users-both-first-class)). The detector was chosen
on the user's labels: 130 candidates in two rounds, drawn where the detectors disagree, shown
blind ([scene-detection.md](../research/docs/scene-detection.md#decision-brief), jointly with
[cuts.md](../research/docs/cuts.md#what-it-means-for-shot-detection)).
- **The detector: TransNetV2**, the official model (MIT): its PyTorch code vendored, its
  weights converted from the official TensorFlow ones by its own `convert_weights.py` (PyTorch
  matches TensorFlow within 5.1e-7, with the same detections), saved as safetensors by
  `models/`'s script and pulled from seedvr2x's Hugging Face repo (see [Weights](#weights)).
  - It reads every frame of the first pass's decode, scaled to 48×27 as its official extraction
    scales them (ffmpeg's default scaler; another scaler is untested), on one thread, so that
    its input doesn't depend on the CPU count.
  - It runs on the GPU, idle during the first pass. Its cost is about 2.3 min of CPU per hour
    of 1080p source over the decode the frame index makes anyway (16 threads), deterministic;
    the GPU's is measured at the build step, the cut list checked equal to the CPU's on the
    labelled sources.
- **A cut:** a detection is a run of frames whose single-frame probability (the sigmoid of
  TransNetV2's single-frame head) reaches the threshold, one per run, at its peak. The cut is
  the frame after the peak: TransNetV2 marks the outgoing shot's last frame (offset −1 on 98%
  of the sure cuts). Every detection starts a shot, however short: no burst filter and no
  minimum shot length. A short shot is better run alone than merged into its neighbour, from 1
  frame on (10–21 dB·frames of PSNR-Y), and real cuts come 1–3 frames apart in action anime.
- **The threshold is 0.3 by default**, confirmed by the second round. Against 0.5 it misses an
  estimated 92 cuts per hour of animation instead of 122, and 19 per hour of live action
  instead of 70, for more false cuts (animation's precision 0.82 against 0.89). A miss costs
  fidelity (5–31 dB·frames of PSNR-Y on four of six cuts), a false cut none (at most a
  low-frequency step on a continuous shot).
  - `--cut-threshold P` sets it (the user's proposal, 2026-10-06), a setting the manifest
    records: lower where cuts go missing, higher where false ones show. Erring lower is the
    cheaper mistake. It is refused with `--cuts`, whose list replaces the detection.
- **No gate on a picture change.** Its measure, scdet's MAFD at the cut frame (the mean
  absolute difference between that frame's luma and the previous one's, at full resolution and
  native bit depth, in percent), at 2 keeps every labelled cut but adds only 0.02–0.03 of
  precision, and at 3 loses live-action cuts. The lowest real cuts sit at 2.14 and 2.93
  (low-contrast cuts in a slow film) and 2.45 (rolling credits); TransNetV2's false cuts on
  still pictures run from 0.42 to 3.87.
  - Should false cuts on held pictures show in the user's visual review (stitching found a
    unit boundary's step visible on held drawings), the fallback keeps a detection only at a
    MAFD of 1.5 or more, 0.6 under the lowest labelled cut.
- **Why TransNetV2:**
  - Live action: recall 0.98 at 0.3, against 0.25 for scdet as sptenc runs it (threshold 10),
    precision 0.84 against 0.89: scdet misses three quarters of the cuts.
  - Animation: recall 0.90 against scdet's 0.84, precision 0.82 against 0.62. scdet bursts by
    construction: its score is the smaller of the frame difference and that difference's change
    from the previous frame, so every new drawing after a held one scores its whole difference
    (351 bursts per hour against TransNetV2's 19). Filtering scdet's bursts loses a third of
    its cuts. PySceneDetect avoids bursts only through its 15-frame minimum scene length, which
    merges real flash cuts too.
  - Its blind spot is fast action anime, inside scdet's bursts: of 14 labelled lone scdet hits
    there, 2 were real cuts (14%, 5–35%) that TransNetV2 scored under 0.1. So it misses about
    60–70 cuts per hour of action anime (25–180 at the interval's ends). Taking them back with
    scdet would bring 6 false cuts for each real one, a burst again: not taken. Such content
    gets its missing cuts by hand, through a cut list.
  - Fast camera work in animation made natively in 4K too: in a fast camera flight, every
    frame differs from the last by as much as a cut does (MAFD 13–16 on every frame), so two
    real cuts between similar-coloured moving shots scored 0.207 and 0.155, while a shot in the
    same flight peaked at 0.232. No threshold separates them there: 0.2 finds 1 of the 2 and 1
    false cut, 0.15 both and 2 false ones
    ([scene-detection.md](../research/docs/scene-detection.md#native-4k-animation-cuts-inside-fast-camera-motion)).
    Full-resolution measures missed both cuts as well (scdet 1.8 and 1.3).
  - Unlabelled: the 548 animated candidates TransNetV2 alone scores 0.1–0.3 hold at most
    about 80 cuts, under 3% of animation's estimated 2,910.
- scdet and PySceneDetect aren't used, and the startup check doesn't require scdet. idet stays,
  for telecined sources declared progressive.
- The first pass's record keeps TransNetV2's per-frame probabilities (its single-frame head),
  and the cuts derive from them and the threshold: another threshold gives its cuts without a
  second decode or detection. A resume trusts the record.
- A cut list given with `--cuts` replaces the detection, and `--plan` writes the detected one
  in the same format, so that adding or removing a few cuts is an edit, not a rewrite.
  - The possible cuts go in it too, as comments: each run peaking from 0.1 up to the
    threshold, with its probability and timecode. Checking them is a jump to each timecode in
    a player, and keeping a real one is uncommenting its line. `--plan` prints the counts: a
    4.4-minute short made natively in 4K has 61 cuts at 0.3 and 39 possible cuts.

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
   - Measured, the default stays: without antialiasing, worse on 7 of 7 clips; zimg's Spline36
     and Lanczos gain a little on some grainy or detailed clips and cost colour and flicker on
     fast live action ([numerics.md](../research/docs/numerics.md)). No kernel option in v1.
   - The frame is then padded to multiples of 16: at least 8 rows reflected from the picture,
     then 16 black rows, all trimmed after decoding (`reflect>=8+black+16`,
     [numerics.md](../research/docs/numerics.md#reflectblack16)).
     - numz pads with black alone (8 rows at 1080p, none at 720p), which costs the bottom 16
       rows 3–10 dB. Padding without black, or cropping as ByteDance does, repairs that band
       but makes the whole frame worse on clips without black areas of their own (−0.6 to
       −1.1 dB PSNR-Y): the black rows anchor the model's tone.
     - Reflected rows, then black ones: the bottom band +7 to +13 dB at 1080p and the rest as
       good or better (PSNR-Y +0.29 to +0.40 dB on 3 of 4 clips); at 720p better on all 4
       clips (PSNR-Y +0.07 to +1.03 dB, VMAF +0.9 to +10.2). Effects 1.7–8 times the spread
       between seeds; through `lab` they hold, smaller.
     - At 4K, measured inside a letterboxed film's picture (2048 rows, where numz pads
       nothing, as at 720p): the bottom band +1.9 to +4.8 dB on 4 of 4 clips, but no gain
       across the frame. The rest is worse on 2 clips (−0.94 and −0.38 dB, beyond the seeds'
       band) and within it on 2; LPIPS is better on 3, VMAF on 2, low-frequency colour worse on
       3 (ΔE00 +0.07 to +0.29), which colour correction works on. The rule stays at every size,
       since a second rule would buy nothing measured (one film, 25 frames, colour correction
       off; the latent grid's change, 260 rows instead of 256, isn't separated from the black
       rows).
     - Columns, when the width isn't a multiple of 16, the same way (not measured: outputs are
       mostly 1920 or 3840 wide). It costs 24 rows at 1080p (1,104 for numz's 1,088, +1.5% of
       the tokens) and 32 at 720p.
     - That ends bit-identity with numz on the default path. The milestone-1 regression runs
       with numz's padding through a mode internal to the tests, and the new padding is checked
       bit-identical to numz with measurement's `NUM_PAD` patch, as milestone 2 checked the
       stitching.
1. **VAE encode** of the whole shot in one causal pass. The VAE already streams in 4-frame
   slices; resetting it at each cut is correct (no context should cross a cut).
   - The frames are read from ffmpeg's decoding of the source at the VAE's own pace: 5, then 4
     at a time. The first latent holds 1 frame and every next one 4, and the causal VAE works
     in those slices anyway. This is not a batch size: the shot is encoded in one pass.
   - The padding to 4n + 1 is made from the last 4 frames read, so the latent is the one-pass
     encode's, bit for bit.
   - Only the latents are ever whole in memory.

   Shots must start at real cuts:
   - each latent packs 4 frames (latent frames = 1 + (frames − 1)/4), so a cut inside a group
     mixes both scenes in one latent
   - windows stitched across a cut would cross-fade the two scenes

   Measured ([cuts.md](../research/docs/cuts.md)): a missed cut costs the next shot's first
   frames, mostly through the causal VAE, which carries the previous shot over. It is the
   costliest detection error (see [Open questions](#open-questions)).

   A shot's first frame is encoded alone, the first latent holding 1 frame. It comes out closer
   to the ground truth than the next 8 frames (by 0.04–5.5 dB of PSNR-Y, on 11 shots), but
   re-rendered less: less sharp on 9 of them, by 3–55%
   ([cuts.md](../research/docs/cuts.md#a-shots-first-frame-prepending-mirrored-frames)).
   Prepending 4 mirrored frames, as numz's `--prepend_frames` does, gives it the next frames'
   look for one more latent per shot (2–8% more GPU time per hour of animation), at the cost of
   0.6–2.8 dB of that lead on 7 of the 11: not done. Whether its sharpness step shows right
   after a cut, which changes the picture 40 to 120 times more, is for the user's visual review
   (milestone 7).

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

The model shifts colours, and the VAE's tiles shift each tile's level and colour. seedvr2x
corrects both against the input with `split`, the colour study's winner
([colour.md](../research/docs/colour.md)), which goes beyond numz's `lab` (see
[Beyond numz's `lab`](#beyond-numzs-lab)). `--color-correction {split,none}`, `split` by
default, is a setting, so a resume compares it. `lab`, which milestone 5 matched to numz's,
goes, and with it the decode's two passes and their buffer (the user's decision, 2026-10-06,
after looking at the study's crops).

What `split` does, frame by frame:
- **Two scales, in Y'CbCr.** The model's decode keeps its detail, and takes its coarse lightness
  and colour from the input: BT.709 Y'CbCr, the gamma-encoded RGB through BT.709's matrix, a
  linear map (so equal scales would be a plain RGB split). Each channel is moved by the
  difference of the low bands, then the frame goes back to RGB, clamped.
  - Lightness (Y') below 4 à-trous stages, σ 6.5 px of output, at every upscale factor.
  - Colour (Cb, Cr) below the stage whose σ is nearest 1.6 source pixels: round(2 + log2(f))
    stages, so 3 (σ 3.2 px) from ×1.41 to ×2.83, 4 (σ 6.5 px) from ×2.83 to ×5.66, and 2 (σ
    1.6 px) below ×1.41. f is the upscale factor; where the display aspect gives the two axes
    different factors (anamorphic SD), their geometric mean.
  - The stages are the wavelet split's: the 3×3 binomial kernel, its taps 2^s pixels apart at
    stage s, at most an eighth of the frame's smaller side, edges replicated at each stage.
- **Why these scales.**
  - Colour finer than about 1.6 source pixels, the input's own colour resolution (a 4:2:0 source
    has a chroma sample every 2 source pixels), costs LPIPS and DISTS; coarser leaves colour
    fringes. Checked at ×1.5, ×2, ×3 and ×4: at ×1.5 and ×3, the stage below cost DISTS on 2 of
    3 clips, the stage above colour fringes or LPIPS. Between those factors (×2.25, 480p to
    1080p) the rule is an interpolation, and outside them it is untested.
  - Lightness at 6.5 px works at every factor measured. At 13 px, numz's split, the model's own
    lightness flickers more than `lab`'s; at 3.2 px, fine texture starts to go.
  - No histogram step: numz's a\*b\* matching is what costs `lab` colour, moving it at every
    scale. With no statistics pooled over the shot, each frame needs only its own decode and
    its reference frame, so the decode streams: no second pass, no buffer, a shot's first
    frames out as they are decoded.
- **Against our `lab`** (colour.md: 7B fp16, shots of 45 frames, each verdict paired by frame
  and beyond `lab`'s seed spread, counted better / worse / within):
  - 1080p, ×2 from the mild degradation (8 clips, 3 seeds): ΔE00 −0.30, −0.46 and −0.29 after
    blurs of 0, 4 and 16 px (8 / 0 / 0); PSNR-Y +0.63 dB (4 / 0 / 4); LPIPS −0.0096 (7 / 0 / 1);
    DISTS −0.0023 (2 / 2 / 4); flicker −0.44 (6 / 0 / 2), at low frequency −0.76 (8 / 0 / 0);
    colour fringes −0.46 (7 / 0 / 1); no clip loses detail.
  - The heavier degradation (7 clips): ΔE00 after 4 px −0.27 (7 / 0 / 0), PSNR-Y +0.65 dB
    (5 / 0 / 2), low-frequency flicker −0.70 (7 / 0 / 0); DISTS +0.0011 (1 / 4 / 2), 0.003 to
    0.005 worse on four compressed inputs, as every split is.
  - Through the default `yuv420p10le` master: VMAF +3.7 (7 / 0 / 1), no banding added.
  - ×4, colour at 4 stages (5 clips): ΔE00 after 4 px −0.18 (5 / 0 / 0), LPIPS −0.0020
    (4 / 0 / 1), low-frequency flicker −0.58 (5 / 0 / 0).
  - 4K: on the first film's 4 shots, PSNR-Y +0.40 dB and VMAF +4.0 (4 / 0 / 0); on 13 more
    (clean live action, animation made in 4K, a grainy scan), low-frequency ΔE00 −0.33, PSNR-Y
    +0.57 dB, VMAF +5.3 and low-frequency flicker −0.58 (13 / 0 / 0 each), LPIPS better on 10
    of 13, DISTS 6 / 6 / 1.
  - Clip B's window joins: the low-frequency step 40% lower (0.046 to 0.028).
  - The user's eyes, on crops beside the ground truth: never worse than `lab`, and better where
    `lab`'s colour noise shows (a cartoon in fast motion, a compressed live-action shot).
  - Lightness at 3.2 px too (`ycc:3:3`) scores higher still on fidelity (PSNR-Y +1.69 dB, VMAF
    +9.7 in the master), but loses faint texture (15% of a sky's Laplacian variance) and is
    mixed on DISTS on compressed inputs; the user's eyes couldn't tell it from `split`.
- **Cost:** 22.8 ms and 0.56 GiB per 4K frame on the GPU, against `lab`'s 104.6 ms and 0.84
  GiB. It runs on each decoded slice as it comes out, so the planner counts it in the decode's
  phase.
- **Licences.** The wavelet split is StableSR's method, whose code is non-commercial: ours is
  rewritten from the method, clean room (another agent read StableSR's code and passed on only
  the method). No numz code is left in the correction: the CIELAB conversions and the histogram
  matching ported from numz (Apache-2.0) go with `lab`.
- **Numerics.**
  - The split runs in float32. numz's bf16 is 0.10 level off on average, 1.17 at most.
  - The low band is moved by adding the difference of the low bands: content +
    (low(reference) − low(content)). That is the same sum as high(content) + low(reference),
    without rounding a high band on its own. A float32 high band added back to its low band
    misses the image at 0.4–2.2% of values on test frames, whereas content moved onto itself
    comes back bit for bit.
  - In the pipeline, the correction runs on the GPU; the CPU path serves the tests.
  - The reference is the input copy's frames through the encoder's transform (the same resize,
    clamp, padding and normalisation) in float32, ByteDance's precision. It is not the
    encoder's own tensor, decided after milestone 5:
    - numz's numerics cast the frames to bf16 before the resize
      (`generation_phases.py:380-413`), and 89% of 8-bit codes round up in bf16: +0.114 level
      on average, +0.186 over codes 128–255, where the fp16 path averages 0.000.
    - A resize keeps the mean, so a reference built that way brightens the corrected output by
      the clip's own bias. As first built, our `lab` exceeded numz's Y shift on 5 of 8
      full-reference clips, by 0.10–0.16 level, as predicted from their codes.
    - numz's own `lab` never used that tensor: it transforms the fp16 frames without the cast
      (`:127-168`).
  - The corrected frames stay float32 for the writer. The low bands are computed in float32,
    and a bf16 cast would cut them back to 8 significant bits. The output then fills the 16
    bits: a `gbrp16le` master took 2.0–2.5 times numz's with `lab` (see [Output](#output)).
- **Input copy.** An FFV1 `gbrp16le` copy of the 16-bit frames the encode reads, at input
  resolution (about 32 MiB per second of 1080p input: 1.38 MB per frame on clip B). It is
  written to `resume/shot_<start>/input.mkv` while the encode reads the frames, recorded with
  the shot's latent, and kept until its segment is finished. It's exact, sequential, and
  independent of seeking.
  - It is derived data. It's a lossless decode of an input whose content is checked, so it can
    be remade bit for bit with the same ffmpeg and conversions.
    - After an accepted change of ffmpeg or its conversions, a remade copy holds the new
      decode's frames. The correction's reference may then differ from the encoder's input by
      what the change changed. Only bit-identity is given up, as with any accepted change.
  - **Integrity, checked end to end.**
    - Each frame gets a CRC-32 over its 16-bit planes as written, kept in
      `resume/shot_<start>/input.crc32`. That file is written whole with the copy, before the
      latent is recorded, and a missing checksum file counts as a missing copy.
    - Each frame read back is checked before the correction uses it. That costs about 2 ms per
      1080p frame. The checksums guard the file, not the decode, so a remade copy gets new
      ones.
    - The copy is also read strictly. ffmpeg's errors go to a file, and a read fails on
      anything ffmpeg reports, checked after every read before its frames are used. That
      names the cause.
  - Why both checks are needed:
    - No ffmpeg option makes a slice failing its CRC fail the decode. With
      `-err_detect crccheck+explode` or `-xerror`, ffmpeg n9.0.2 decodes through it and exits
      0, hiding the slice under the previous frame's. Its only signal is a line on stderr.
    - A damaged slice-size field in FFV1 v3 can make ffmpeg skip slices without a word. The
      skipped area keeps whatever its frame buffer held: another frame's picture, or zeros
      (black) in a newly allocated buffer, so a black frame can come back right by chance. The
      damage is silent on any flat frame whose slices are all one size.
      - A frame came back with 2,880 of its 3,072 pixels from another frame.
      - Fuzzing small copies gave 2 silent wrong frames in 17,585 byte positions on random
        content, and 2 in 13,160 on smooth content.
      - The test uses a flat frame at 255.
    - The Matroska demuxer silently drops a block whose ID is damaged.
    - A full temporary directory would silence ffmpeg's errors altogether.
  - **A missing or damaged copy.**
    - A copy missing when a run starts is remade from the input before the shot's windows or
      decode, reading the input in shot order.
    - A copy the decode finds missing, cut short or damaged stops the run. It is named and
      removed, and the next run remakes it, keeping the shot's latent and windows. A lost or
      damaged temporary file never ends a multi-hour job.
  - The decode writes a shot's last frames only once its copy has been read whole and
    checked, since they may finish the output segment, which is then recorded.
  - For `-o x.mkv`, the same files live in `<output>.work`, made and locked at the start as an
    output directory is, so a second run to the same file is refused. It is removed when the
    run ends, however it ends. What a killed run left there (only seedvr2x's files) is emptied
    at the next start; anything else is refused, never deleted.
- With `none`, nothing is copied.

`split` is validated against the colour study's own implementation, frame for frame, and the
study validated it against `lab` (see [Validation milestones](#validation-milestones), 5).

### Beyond numz's `lab`

Milestone 5 matched numz's `lab`: a floor, not the aim. The user counts colour correction among
seedvr2x's critical parts: it decides how faithful the output's colours are, it repairs what
tiling does to them, and the measurements left room. The colour study
([colour.md](../research/docs/colour.md)) went past it, to `split`.

numz's `lab`, per batch slice (numz `4490bd1`):
- **Inputs.** The content is the VAE's bf16 output, unclamped. The reference is the input
  rebuilt with the model's input transform, in fp16 on the CPU. That is not the tensor the
  encoder saw; ByteDance uses that one.
- **Wavelet split.** The output is the content's high band plus the reference's low band,
  clamped. The low band is 5 à-trous stages of the 3×3 binomial kernel, edges replicated at
  each stage: away from the edges, a separable 63-tap tent filter, σ ≈ 13 px. The high band
  is the image minus the low band.
  - The dilation at stage s is min(2^s, max(1, ⌊min(H, W) / 8⌋)): 1 to 16, capped only on
    frames under 128 px.
- **Matching**, in float32: RGB to CIELAB (sRGB, D65), then histogram matching per channel by
  exact rank over all the slice's frames (a* and b* fully, L* = 0.8 content + 0.2 matched),
  then back to RGB, clamped, bf16.

Milestone 5 built it pooled per shot, one mapping per shot through integer histograms, so the
decode made two passes over a bf16 buffer of the shot's frames (17.9 GB per minute of 1080p
shot). `split` needs neither.

- **The gap.** On the full-reference clips, `lab` brings the low-frequency colour error to the
  ground truth down by 37–55%, to a ΔE00 of 1.07–1.58 after a 4 px blur, but the input itself,
  upscaled with Catmull-Rom, is at 0.6–0.7 ([numerics.md](../research/docs/numerics.md)). The
  split replaces only what is coarser than about 30–60 px (σ ≈ 13 px), while the model's extra
  saturation and colour drift sit finer
  ([quality.md](../research/docs/quality.md#colour-correction)). The histogram matching then
  corrects their distribution over the shot, not where they are.
- **The aim:** colour and brightness as close to the source as the input holds them, at every
  scale where it does, with the model's detail kept: no detail lost, no flicker, halo or
  banding added.
- **What the study found** (step 0 on numerics.md's masters, CPU only; steps 1–3 on dumps of 7B
  fp16 runs, every variant post-processing the same decode and reference; step 4 the user's
  eyes):
  - Colour and brightness at different scales, in a luma-chroma space, chroma finer than luma:
    the winner, `split` (above).
  - numz's histogram step is what costs `lab` colour. Its L\* blend alone only helps a split
    whose lightness sits at 13 px, against flicker, and adds nothing at 6.5 px.
  - The scale following the upscale factor: yes for colour, no for lightness (above).
  - An edge-aware transfer, a guided filter steered by the decode's lightness: its gain on
    colour fringes is colour at 1.6 px's, at a higher cost.
  - CIELAB or OKLab rather than Y'CbCr: the same within a few hundredths, and Y'CbCr is a 3×3
    matrix.
  - The uncorrected model's drift: more saturation on 9 of 10 clips, b\* toward yellow on 7.
    "Toward blue" held on the dark anime clips alone.
- **Tiles are part of it.** The VAE runs tiled at 4K on every card, the 96 GB one included (an
  untiled 4K decode needs ≈ 134 GiB), and at 1080p below 48 GB. A tile's damage is a colour
  drift: each tile shifted uniformly, by up to 2 levels in the decode and 4 in the encode at
  512 px ([vram.md](../research/docs/vram.md#tiling)).
  - After `split`, the worst 60-px block of a frame moves by 0.04–0.50 8-bit levels at 1080p
    (5 clips: decode tiles from 1280 down to 512 px, and the consumer cards' encode / decode
    recipes 1344 / 1024 and 1024 / 768), and up to 0.77 with 512-px encode tiles, 0.27 at
    most on the flattest areas. At 4K it moves by 0.14–0.49 (4 shots: decode tiles from 1536
    down to 512 px and the two recipes, against the 2048-px decode). That is about half of what
    `lab` leaves (0.10–1.39 and 0.38–1.10).
  - So colour sets no floor on tile size down to 512 px, the smallest measured: the planner's
    tiles are the largest that fit (see [Memory planner](#memory-planner)), 512 px at the
    least. At 4K it holds on flat content too (a smooth sky, near space, painted animation):
    with 512-px decode tiles the flattest third moves by 0.08–0.15, the worst block by up to
    0.77, in painted texture
    ([colour.md](../research/docs/colour.md#4k-tiles-on-flat-content)).
- **Protocol:** no model run per variant. The raw decodes and the float32 reference are dumped
  once per clip and tiling, and each variant post-processes them, built on `runtime/colour.py`
  so that the winner ports as it is. Scores: ΔE00 after blurs of 0 to 16 px, PSNR-Y, LPIPS,
  DISTS, the Laplacian variance (detail kept), flicker, colour fringes at strong edges, CAMBI
  (banding), milestone 2's boundary steps on clip B, and the finalists through the default
  `yuv420p10le` master. Verdicts are paired against our `lab` and must exceed its seed spread.
  Then the user's eyes, on crops of edges, skin, skies and flat areas.
- **Constraints:** written from its own maths, never from StableSR's code (the clean room
  holds).

## Output

seedvr2x delivers the upscale losslessly and owns no encoder flags. How the output gets
compressed is the user's choice: afterwards, from the master, or during the run with
`--segment-cmd` (below).

- **FFV1 masters**, every frame a keyframe, per-slice CRCs, exact frame rate, from the float
  frames (no 8-bit step, bugs 09/19):
  - `yuv420p10le`, **the default** (the user's decision, 2026-10-04): BT.709, limited range,
    explicit conversion (zscale). It is what sptenc and every encoder take. sptenc takes such
    a file as it is, without any conversion, which its MANUAL recommends for tools that can
    write YUV. So the RGB→YUV conversion happens once, in the tool that requires zscale, and
    exactly (white at 940).
    - With `gbrp16le` the default, that conversion would fall to whatever reads the master:
      sptenc's swscale (white at 943), or the user's `--segment-cmd`. And a corrected `gbrp16le`
      master takes 540–690 GiB per hour of 1080p, five times as much: impossible to keep.
    - This is sptenc's intended input, not a workaround. sptenc converts RGB sources with
      swscale (16-bit white at 943 instead of 940), a documented trade-off: zscale isn't in
      every ffmpeg build, and sptenc ships for any build.
    - The master shares the pixel format and tags of `sptenc master`, chroma sited left
      included. The conversion tests (see [Input](#input)) check that zscale sites the chroma
      left, as tagged.
    - The chroma is downsampled with zscale's bilinear, pinned, as `ffv1_out.py` does, on one
      slice (see [Input](#input)). Decimation wants a low-pass, not a sharp interpolator (the
      decode, which interpolates, uses Catmull-Rom), and the measurement agrees
      ([numerics.md](../research/docs/numerics.md#the-masters-chroma-420-kernels-and-zscales-slices)):
      - On the 8 full-reference clips, bilinear gives the lowest ΔE00 on every clip, against
        Catmull-Rom, Spline16, Spline36 and Lanczos. These lose 0.3–0.5 dB on Cb and Cr on
        average, 0.05–0.07 of ΔE00, and ring more.
      - The round trip even brings the output closer to the ground truth (PSNR-Cb +1.1 dB),
        whose colour comes from 4:2:0 sources. The luma metrics don't move (PSNR-Y within
        0.012 dB, VMAF within 0.24).
  - `gbrp16le`, on request: closest to the model, the master for precision work and for
    measurements on short samples, where methods are searched and verified (the tests and the
    research scoring use it). At its full size (see Master sizes below), it isn't meant for
    whole films.

  Validated with the [`ffv1_out.py` wrap](../research/docs/output.md): bit-exact round trip,
  tags checked by ffprobe. Our writers give a `yuv420p10le` bit-identical to its output. They
  set every tag both on the frames (`setparams`) and on the encoder: ffmpeg n9 converts frames
  whose tags differ from the encoder's.
- **PNG** (16-bit) as an alternative, one directory per segment, frames numbered from
  `000000`. ffmpeg's PNG encoder writes the primaries and transfer as cICP, cHRM and gAMA
  chunks from the frame tags, and none for an untagged frame. So PNG copies the tags as the
  masters do.
- **Until assembly** (milestone 6):
  - `-o x.mkv` writes one FFV1 master.
  - Any other `-o` is a directory of segments plus `manifest.json`. It is either new or empty,
    or an unfinished job's directory, which the same command resumes (see
    [Pause and resume](#pause-and-resume)). One seedvr2x at a time writes it, and anything
    else found in it is refused, so another run's files never reach what sptenc reads.
  - An `-o` with another video suffix (`.mp4`, `.mov`…) is refused.
- **Output segments**, the resume units of the output, listed in a manifest:
  - **Layout, by sptenc's rule.** The detector picks the cuts, then the minimum segment length
    (5 s by default) merges each too-short segment into its shorter neighbour, on the frame
    grid.
    - `sptenc encode` takes any segments, so they needn't match sptenc's own split, whose
      cuts come from scdet (seedvr2x's come from TransNetV2, see
      [Shot detection](#shot-detection)).
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
  - **Checksums.** The segments are the product, and the silent FFV1 damage found on input
    copies could hit a finished master on disk too. So each frame of an FFV1 or PNG segment
    gets a CRC-32 over the planes the file holds, computed at write time, the only cheap
    moment (about 2 ms per 1080p frame).
    - `gbrp16le` and PNG hold the planes seedvr2x feeds ffmpeg, so their CRC-32 is computed
      on those planes.
    - A `yuv420p10le` master holds what ffmpeg converts. Its CRC-32 comes from ffmpeg itself:
      `framehash -hash crc32` on the converted frames, as a second output of the same process
      (through the `split` filter), checked equal to zlib's CRC-32 of the raw frames. The
      startup check therefore requires `split`, and `framehash` for a `yuv420p10le` output.
      The master's bytes are unchanged by the second output: decoded frames, size and metadata
      were compared with the previous writer's in 7 cases (BT.709, untagged SD, `smpte170m`,
      `bt470bg`, BT.2020 with PQ, an odd chroma width, an odd size).
    - They are kept in `<out>/checksums/`, one per frame in order, written whole before the
      segment is recorded. They outlive the job, unlike `resume/`, and sptenc only reads `.mkv`
      and `.mp4` files.
      - Each file is named after its segment, then `.crc32`: an FFV1 file's name without the
        `.mkv` seedvr2x gives it, a PNG directory's name as it is.
      - A resume refuses a finished segment whose checksums are missing, since they are
        written before it is recorded. It discards an unfinished segment's checksums and
        partial files, and refuses anything else there.
    - For `-o x.mkv`, they go in `<output>.crc32` beside it. The old ones are removed only as
      the new file replaces the old, so a run stopped before keeps the old file with its own.
    - A resume checks a finished segment's size against the manifest: cheap, and it catches a
      truncated file.
    - `seedvr2x verify` decodes and checks every frame, on demand, for instance before sptenc or
      an archive trusts the output.
      - It takes an output directory, one of its segments (checked against the directory's
        checksums), or a one-file output.
      - A PNG segment must hold exactly its frames, `NNNNNN.png` from 0, before any is decoded.
      - Anything ffmpeg reports while decoding fails the check.
      - It exits 1 when a frame isn't as written, when checksums are missing, and when a
        segment is unfinished: the output isn't to be trusted whole yet, so
        `seedvr2x verify out && sptenc encode out` never hands sptenc an incomplete directory.
        Its summary says which, e.g. "of 2, 1 unfinished".
      - It needs only ffmpeg and ffprobe, not zscale: the frames are decoded as stored.
    - `--segment-cmd` outputs only get their frame count checked: their frames are the user's
      encoder's.
  - **Writer.** FFV1 by default. With `--segment-cmd`, the user's command runs once per
    segment: it reads the segment, lossless and tagged, on stdin, in the master's format
    (`yuv420p10le` by default, so the encoder converts nothing), and writes the file seedvr2x
    names, e.g. `--segment-cmd 'ffmpeg -i - -c:v libx265 -crf 16 {out}'`. seedvr2x never
    parses the command and only checks the result's frame count. Disk use is then the
    compressed size, and a stop keeps every finished segment.
  - **Master sizes,** per hour of 1080p at 24000/1001, four times that at 4K: about 100–135
    GiB in `yuv420p10le`, the default, and 540–690 GiB in `gbrp16le` with colour correction
    (measured with `lab` in milestone 5). The corrected float32 output fills the 16 bits, tens
    of thousands of distinct codes in a frame against 300–400 in numz's bf16 output, whose
    `gbrp16le` masters took 250–350 GiB ([output.md](../research/docs/output.md) measured 2.97
    MiB per frame). Rounding ours to bf16 would save 34–43%, for 8 significant bits.
  - **Rejected:** `--stream`, the output on stdout for a single compressor process. That
    process can't be paused, so a stopped run would leave parts to join by hand.
- **Assembly** of the segments:
  - **standalone:** joined into one file by stream copy, with the source's other streams
    (audio, subtitles, chapters, attachments). The join carries sptenc's lessons: each
    segment's duration comes from its frame count, and timestamps are snapped to the frame
    grid. Otherwise the video drifts: 46 ms behind the audio over a 163-segment episode, in
    sptenc's measurements.
  - **with sptenc:** the directory as it is, for `sptenc encode <dir> <out> -f <source>` (or
    `sptenc concat` for one master)

## Memory planner

Built into the CLI, from the validated models in [vram.md](../research/docs/vram.md) and
[planner-limits.md](../research/docs/planner-limits.md):
- **Budget:** the free memory the driver reports once the CUDA context exists (`mem_get_info`,
  read before anything else allocates on the GPU), minus 0.6 GiB. That covers the desktop and
  other programs without guessing.
  - The margin was bisected on emulated cards: a run needs 0.15–0.45 GiB between the free
    memory it starts with and the torch peak of its bounding phase, the decode-bound 8 GB card
    the most (it failed with 0.37 GiB). 0.6 GiB keeps 0.23 over the worst failure, and gives
    back the validated "torch peak ≤ card size − 2 GiB" on those cards.
  - Below the margin a run fails in its bounding phase, after dozens of silent allocator
    retries.
- **Host RAM** too. The process's peak, 16.5 GiB with 7B fp16, comes while the weights load,
  not during the shots, which stay flat (2.3–2.4 GiB resident over 6 shots). It is probably the
  weights file mapped while it is copied to the GPU; not measured further. BlockSwap's pinned
  host copies add their size. Both matter on hosts with little RAM.
- **Per-phase peaks** (P = output megapixels, T² = a tile's area in megapixels):
  - VAE encode ≈ 1.2 + 8.8·P GiB, decode ≈ 0.8 + 16.1·P GiB (flat beyond 9 frames)
  - tiled, from 1024-px tiles: encode ≈ 1.36 + 8.66·T², decode ≈ 0.44 + 16.29·T², refitted
    at 4K up to 2048-px tiles, where vram.md's fits fell 1.7 GiB short. Below 1024 px,
    vram.md's: encode ≈ 1.7 + 8.4·T², decode ≈ 1.6 + 15.6·T². Each adds what is on the GPU
    before the call: the weights, not the shot's frames, since seedvr2x feeds the encode slice
    by slice (numz moves a batch's frames there first, 0.046 GiB per 4K frame).
  - the colour correction, counted in the decode's phase, where it runs on each decoded slice:
    0.56 GiB per 4K frame it corrects at once
    ([colour.md](../research/docs/colour.md#cost-and-what-streams)).
  - DiT (7B fp16) ≈ 15.87 GiB + 127.16 KiB per token + 2.726 MiB per attention window, the
    windows counted with the model's own window functions (the larger of its two layouts).
    That fits 26 window lengths, 1–78 latents at 1080p and 1–19 at 4K, to 0.005 GiB; tokens
    alone were up to 0.83 GiB off at 4K, since each attention window repeats the 58 text
    tokens. The other models' constants are in vram.md.
  - On the 96 GB card: windows of 78 latents (309 frames) at 1080p, 0.56 GiB to spare, and 19
    (73 frames) at 4K; one latent more fails within 3 s. vram.md's recipes per card size,
    validated on emulated cards, are milestone 3's starting points.
- **Choices, in order:** window length (the biggest quality lever: fewer boundaries),
  BlockSwap blocks, VAE tile sizes, then what's left goes to speed: `compile_dit` (−26 to −32%
  DiT time, +0.1 to +1.4 GiB), and `compile_vae` only when its memory fits (it about doubles
  VAE activation memory, for −16 to −19% VAE time). Nothing caps a large card: the memory a
  16–32 GB card spends on BlockSwap and tiles goes, on 96 GB and more, to longer windows and
  speed.
  - The tiles are the largest that fit. A frame takes the same time whatever the tile (a 4K
    frame 5.3–5.6 s to encode and 11.5–12.3 s to decode with 1024-, 1536- or 2048-px tiles),
    and smaller tiles drift more in colour, though after `split` too little to set a floor down
    to 512 px, the smallest tile the planner makes (see
    [Beyond numz's `lab`](#beyond-numzs-lab)).
- **Time estimates** (for `--until`, `--plan` and progress) are measured, never constants.
  The DiT's time is its tokens times the machine's time per token, flat within ±5% across
  window lengths and resolutions, but anywhere from 0.23 to 0.45 ms on this one GPU, with its
  clock. A run measures it on its first window, and the VAE's time per frame on its first
  slices; `--plan` measures both with a short calibration on the GPU.
- **A plan shapes the output a little.** Its window lengths set where a long shot's joins
  fall, and its VAE tiles where the picture is cut. A larger card gives fewer joins and larger
  tiles, closer to the one-window, untiled run:
  - Windows change the output about as much as a change of seed: milestone 2's came out
    41.9 dB from the one-window run, and two seeds are 37–44 dB apart
    ([models.md](../research/docs/models.md)).
  - Tiles drift in colour, uniformly across each tile, more as they get smaller: 1024-pixel
    decode tiles came out 37.9–43.6 dB from the untiled decode on two clips, 512-pixel encode
    tiles 33.4 ([vram.md](../research/docs/vram.md#tiling)). Colour correction removes most of
    it: tiled and untiled, both through `lab`, came out 48.9 dB apart
    ([quality.md](../research/docs/quality.md#vae-tiling-on-flat-areas)), and `split` leaves
    about half of what `lab` leaves
    ([colour.md](../research/docs/colour.md#vae-tiles-at-1080p)).
  - So the plan is part of the job. The manifest records it, and a resume reuses it rather
    than planning again, since free memory varies from one start to the next. A resume whose
    plan no longer fits stops and says so.
  - `--window` and the tile options pin a plan, for two runs that must match bit for bit.
- `--plan` prints the plan and the time estimate without running the job, after the first pass
  (the shots are the plan's input), and writes the detected cut list for editing (see
  [Shot detection](#shot-detection)).
- Validated before release with `vram_cap.py` emulation, from the smallest card the 7B fp16
  reaches up to 48 GB.

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
  - environment, compared on resume:
    - torch, CUDA, cuDNN, GPU, attention backend, FlashAttention and ffmpeg
    - a fingerprint of the conversion chain: the SHA-256 of 19 test conversions of fixed
      frames, run in one ffmpeg process at every start (40–100 ms). They cover the decode's
      zscale chain at every input depth, chroma subsampling, matrix, range and siting, and
      the writers' chains. The value is the same on two CPUs with one ffmpeg build, and with
      zscale on one slice it can't depend on the CPU count (see [Input](#input)). It catches
      a zimg upgrade that ffmpeg's version string doesn't show.
    - versions, derived rather than hand-kept, so a new dependency can't be missed:
      torchvision (the resize), diffusers (the VAE's blocks) and rotary-embedding-torch
      change the output bits as surely as torch does, and a venv can drift from its lock.
      The record holds:
      - the distributions imported once every module of seedvr2x is (the vendored models
        included), narrowed to the closure of seedvr2x's declared requirements, followed
        through extras. Dev-only or tooling installs therefore never refuse a resume.
        - cuda-bindings, which `import torch` imports, is among them.
        - setuptools' start-up hook is left out. Every Python process imports
          `_distutils_hack` from `distutils-precedence.pth`, unless
          `SETUPTOOLS_USE_DISTUTILS=stdlib`. The run itself imports no setuptools, and
          counting the hook would make the record depend on that variable.
      - flash-attn, which is not a declared dependency
      - torch's runtime libraries, loaded without being imported: the distributions it
        requires (through extras) that ship a shared library other than a Python extension
        module, since an extension module is loaded only by an import. On the pinned stack,
        that is its 15 nvidia-* wheels and triton.
      - Python's version

      On the GPU box that is 47 distributions, read in 1.75 s at start. A check at the end of
      every run with a manifest warns of any of the run's distributions it imported after the
      record. Known limits:
      - As for any version record, third-party code shadowed on `PYTHONPATH` or installed
        editable isn't seen changing.
      - An import from outside the closure, flash-attn aside, is neither recorded nor warned
        of, so a dependency the code imports must be declared. With uv's exact sync, only
        dev-group packages can be installed outside the closure.
      - Environment markers other than extras aren't evaluated: another platform's or Python
        version's requirement counts when it is installed, which only errs toward recording.

    The NVIDIA driver is recorded through NVML for information, not compared: the math kernels
    ship with torch.
  - inputs: path and mtime for information; the content, compared by size and SHA-256 (hashed
    by a thread while the first pass decodes); the first pass's facts. Also compared: the pixel
    format each input is decoded from, and the primaries and transfer the output copies. An
    accepted ffmpeg change must read and tag the source exactly as before.
  - output, shots (windows, encoded, windows done), and segments (bytes when finished)
  - `environment_changes`: each change accepted with `--accept-env-change`. It records when,
    each field that changed with its values before and after (the driver included, for
    information), and how many segments, shots and windows were already made.

  It is rewritten whole after every unit: a temporary file, fsync, rename, then a directory
  fsync. A unit is recorded only once its file is whole. The manifest stays at version 2: new
  fields are additive, and a manifest written by older code is refused anyway, since its
  `settings.code` differs.
- **Resuming:** the same command on the same `-o` directory, with no `--resume` flag.
  - Refused, with each difference listed: any difference in settings, inputs (by content),
    models or code, except progress. A moved or touched input with the same content is
    accepted, since path and mtime are information.
  - The comparison has two stages:
    - Before the first pass: settings (models hashed), environment and the inputs' content.
      Another job is refused without any decode, and the refusal lists only these
      differences, not the layout differences that would follow from them.
    - After the first pass, trusted or run again: everything else, the first pass's facts and
      the layout.
  - Environment differences are refused too, unless `--accept-env-change` is given. When they
    are the only differences, the refusal names the flag. A different GPU, stack or ffmpeg only
    breaks bit-identity across the resume, not the upscale's correctness, and refusing
    outright would throw away tens of GPU hours over an upgrade.
    - The new environment and its `environment_changes` record are written with the next unit
      made. A resume stopped before making one leaves the manifest as it was, and the job
      still resumes in its old environment.
    - Re-running a finished job doesn't rewrite its manifest.
  - The first pass's record is trusted, and the pass isn't run again, when the content and the
    settings are unchanged and neither `ffmpeg` nor `conversions` changed. The pass runs only
    on ffmpeg's decode, so a torch, CUDA, GPU or attention change doesn't touch it. That saves
    a full decode per resume, about 17 min on a 2-hour HEVC master.
  - The directory must hold what the manifest names and nothing of anyone else's, dotfiles
    included: anything else is refused, never deleted.
  - These are discarded: `.partial` files, an unfinished segment's file, unrecorded units, and
    the units of finished segments.
  - A recorded input copy that is missing or damaged is remade from the input, keeping the
    shot's latent and windows (see [Colour correction](#colour-correction)).
  - Finished segments and kept units are skipped. A finished segment's size is checked against
    the manifest, and its frames by `seedvr2x verify` (see [Output](#output)). The input
    frames of skipped shots are reached through the frame index (see [Input](#input)); until
    that step is built, they are decoded and dropped (decode-and-count).
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
- **Non-finite values** (NaN, inf) stop the run, in both colour correction modes, with a
  message naming the stage, shot, window or frames. numz writes NaN patches about 63 px wide
  through its `lab`, and with `none` a master would get whatever the VAE produced.
  - Every unit is checked before it is recorded: the encode's latent, each DiT window, and
    each decode slice. A unit holding a non-finite value is never recorded, so a resume can't
    reuse it.
  - Stopping at the first stage that produces one names the culprit and saves the decode.
  - A resume restarts that unit once the cause is fixed.
- **Resume is bit-identical** to an uninterrupted run: deterministic noise per shot, exact
  latents, deterministic attention (FA2 reruns are bit-identical). Checked on 2026-10-03 on
  milestone 1's input in 3 shots (one in 3 windows) and 2 segments:
  - stopped by a kill in an encode, a kill in a window, Ctrl-C once in a window, and twice, a
    kill and once in a segment's decode, then resumed after each
  - every segment equals the uninterrupted run's bit for bit, and every stopped process exits
    and leaves the GPU
- **Later:**
  - Resuming a segment's decode mid-way
    ([decode-resume.md](../research/docs/decode-resume.md)). Decoding a warm-up of min(s, 37)
    latents before the resume point and dropping them is bit-identical, never with fewer: 37
    is the decoder's receptive field, to the frame. A snapshot of its 33 causal-conv caches is
    the exact alternative, 8.05 GiB per output megapixel (18 GB at 1080p).
    - Either needs the run's own tiling, memory limits and slice size, which the recorded plan
      keeps.
    - A warm-up costs 4.7 min at 1080p (148 frames), so it only pays off on segments much
      longer than that. `--until` avoids most of the loss anyway.
  - The same manifest lets separate processes take separate shot ranges (several GPUs or
    machines).

## Options kept and dropped

| numz option | seedvr2x |
|---|---|
| `--attention_mode` | dropped: FA2 when installed, else SDPA |
| `--color_correction` | `split` (ours, beyond numz's `lab`) by default; `none` available; numz's `lab`, `wavelet`, `wavelet_adaptive`, `hsv` and `adain` dropped |
| `--input_noise_scale` | dropped (numz's own addition, only degrades) |
| `--latent_noise_scale` | dropped from v1 (ByteDance uses 0; corrected version possibly later as an experiment, see bug 08) |
| `--batch_size`, `--temporal_overlap`, `--prepend_frames`, `--uniform_batch_size` | replaced by shots, windows and latent stitching, chosen by the planner; window length can be capped by the user |
| `--blocks_to_swap`, tile sizes, offload devices | chosen by the planner; overridable |
| `--compile_dit`, `--compile_vae` | chosen by the planner when memory allows; overridable |
| `--cache_dit`, `--cache_vae`, `--chunk_size` | dropped: one long-running process, streaming by design |
| `--seed` | kept (comparisons, reproducibility) |
| `--output_format`, `--video_backend`, `--10bit` | replaced by FFV1 (`yuv420p10le` by default, `gbrp16le`) and 16-bit PNG |
| `--resolution`, `--max_resolution` | kept |
| `--dit_model`, `--model_dir` | kept, the sharp 7B by default; the files come from seedvr2x's own Hugging Face repo, pinned by revision and SHA-256, `--model_dir` holding a local copy; no silent deletion (bug 22) |
| `--cuda_device` | one device per process |

## Documentation for users

seedvr2x's README and manual explain what would otherwise surprise a newcomer, with the reason
and the measurement behind each point.

They open with the [Terms](#terms) in plain words. A window, for instance: "a stretch of a shot
that the model upscales in one go. A long shot is done in several windows that overlap a
little, blended where they overlap before the frames are rebuilt, so the joins don't show.
seedvr2x sizes the windows from your GPU's memory; you can cap them." The model's internal
attention windows stay out of the user docs; they matter only to developers.

**Colour correction, first and in full:**
- **The model itself shifts colours.** Raw SeedVR2 output is more saturated than its input,
  and its hue moves with the content. On a dark anime clip (clip A), with colour correction
  off, the saturation spread is +34% on a* and +28% on b*, toward blue, and the low-frequency
  colour error is ΔE 4.44 ([quality.md](../research/docs/quality.md#colour-correction)). On 7
  of 10 full-reference clips the shift is toward yellow
  ([colour.md](../research/docs/colour.md)).
- **ByteDance's own pipeline corrects it, with a fix it can't ship.** Its inference script
  runs StableSR's wavelet colour fix after the model (`projects/inference_seedvr2_7b.py:300`),
  but the file is under a non-commercial licence, so the readme asks users to download it
  themselves (`readme.md:142`). Without it, the script prints "Color fix is not avaliable" and
  writes the drifted output.
- **numz bundles that code** (uncredited) as `wavelet`, and adds `lab` on top, its default:
  the wavelet step, then CIELAB colour matching. The wavelet fix does most of the work (ΔE
  1.04); `lab` brings the saturation closer still (ΔE 1.01).
- **seedvr2x:** `split` by default: the input's colour below about 1.6 source pixels and its
  brightness below 6.5 px, the model's detail above, and no histogram matching, which is what
  moves `lab`'s colours. Against `lab` it gives a lower colour error at every scale, less
  flicker and half the tiles' drift, and the user's eyes found it never worse
  ([colour.md](../research/docs/colour.md)). Its wavelet split is rewritten from the method,
  with no non-commercial code. `none` gives the raw model's colours, for comparison or for
  users who grade themselves.

**The pipeline, for SeedVR2 users coming from numz.** numz's main knobs (`--batch_size`,
`--temporal_overlap`, `--chunk_size`…) are absent from seedvr2x, which would puzzle its users.
So the docs don't go knob by knob. They explain numz's pipeline with each knob where it acts,
then ours, then why ours has no use for them. The
[Options kept and dropped](#options-kept-and-dropped) table is the reference, not the
explanation.
- **numz works in phases over fixed batches.**
  - It reads the whole clip into RAM, or each `--chunk_size` chunk.
  - It cuts the clip into batches of `--batch_size` frames that ignore the cuts, each padded
    to 4n + 1 frames.
  - It then runs four phases, each over every batch: encode every batch, upscale every batch,
    decode every batch, colour-correct every batch.
  - The latents wait in RAM between phases (`--tensor_offload_device cpu`), and the decoded
    frames gather in RAM before they are written. Without the cache options, the DiT is
    loaded for each file or chunk ([cli-flags.md](../research/docs/cli-flags.md)).
  - So RAM grows with the clip, every batch boundary is a visible jump, and a stop loses
    everything. Most knobs patch what this design creates:
    - `--temporal_overlap` cross-fades batch boundaries in pixels, after decoding. Its
      weights only blend with odd values ≥ 3
      ([bug 06](../research/bugs/06-temporal-overlap-blend-weights.md)): 1, 2 and 4 are a
      hard switch that costs compute. Even corrected, a pixel cross-fade removes 67% of the
      jump, softens the mixed frames by about 20%, and costs +16%.
    - `--chunk_size` bounds the RAM, and leaves a hard seam at every chunk boundary.
    - `--cache_dit` and `--cache_vae` keep the models between files or chunks, since each
      chunk is a full run.
    - `--prepend_frames` makes the clip's first frame not a batch's first. The causal VAE
      encodes a batch's first frame alone, and it comes out less restored
      ([quality.md](../research/docs/quality.md#--prepend_frames)). On one GPU, numz doesn't
      remove the prepended frames ([bug 05](../research/bugs/05-prepend-frames-not-removed.md)).
    - `--uniform_batch_size` pads a short last batch to full size.
    - The offload devices, `--blocks_to_swap` and the tile sizes fit the GPU, by hand.
- **seedvr2x works per shot, from cut to cut, with no batch size.** A shot of any length is
  padded once to 4n + 1 frames, because the VAE's first latent holds 1 frame and every next
  one holds 4. Its DiT windows are as long as the memory allows.
  - Frames stream from ffmpeg's decoding of the source into the VAE encoder at the VAE's own
    pace: 5 frames, then 4 per latent. So the shot never has to sit in memory.
  - A long shot runs as DiT windows sharing 2 latents, mixed before one streamed decode, so
    no boundary is left inside a shot.
  - Frames are written into output segments as they come out.
  - Only the latents are ever whole in memory, so RAM stays flat with length (2.3–2.4 GiB
    over a 377-frame, 6-shot job).
  - The models stay loaded for the whole job.
  - Every unit is saved, so a job stops and resumes, bit for bit.
- **Why the knobs go:**
  - No batch boundary is left inside a shot, so there is nothing to cross-fade. Latent
    windows share whole latents, so there is no parity quirk. They remove 80% of a window
    boundary's jump, with no softening, for about +3–9%
    ([stitching.md](../research/docs/stitching.md)).
  - Streaming keeps RAM flat, so there are no chunks. One process runs the whole job, so
    there is no cache to keep.
  - Only a shot's first frame is encoded alone, right after its cut, where it is least
    visible. numz has one at every batch. Windows are balanced, so none is short.
  - The planner fits the GPU from its free memory, so offload, swap and tiling aren't tuned
    by hand, though they stay overridable.

**Also explained:**
- the two workflows: a file in and a finished file out, or the source in and a directory of
  segments out for sptenc's encode
- how shots are found: TransNetV2, why a missed cut costs more than a false one, and for content
  it gets wrong (fast action anime first, fast camera work) `--cut-threshold`, the possible
  cuts `--plan` lists, and a cut list (`--cuts`)
- the model: v1 runs the 7B fp16 alone, the reference every check is made against, and phase 2
  brings the others
- what the model does to live action ([numerics.md](../research/docs/numerics.md)): the
  perceptual metrics side with it over a plain upscale, the pixel ones with the plain upscale.
  It redraws grain and detail of the right kind at the wrong place, and from a heavily degraded
  grainy source it gives back about half of the film's grain.
- what the model does at 4K
  ([colour.md](../research/docs/colour.md#step-5-4k-on-clean-and-grainy-sources-and-a-detail-strength),
  step 6): ×2 from 1080p, it redraws rather than enlarges. Full-reference metrics, which reward
  a copy of the source's own blur and grain, put it behind a plain upscale on every score but
  banding, on clean sources as on grainy ones (17 shots, PSNR-Y 4–9 dB lower), while the eyes
  prefer its redraw of lines and text. The regular 7B invents fine texture the eyes rejected
  (skin "almost reptilian"); the sharp 7B, the default, invents less of it and keeps the
  redraw. A more compressed input changes neither. Clouds and smooth gradients from compressed
  sources can still show light artefacts.
- lossless delivery: why there are no encoder options, and how `--segment-cmd` compresses with
  the user's own command
- disk use: master sizes per hour, and the input copies during a run
- GPU memory, two points:
  - nvidia-smi can show the card nearly full while a phase needs much less: the allocator keeps
    freed memory reserved, and trims it only when an allocation would fail
  - which phase sets the peak: the DiT through the window length, the VAE through the frame
    size, which is why 4K needs tiled decoding
- why zscale is required, and how to get an ffmpeg build that has it
- the refused sources (VFR, HDR, interlaced, telecined, rotated, cropped, unusual pixel
  formats or matrices), and what to do with each; for a rate its timestamps contradict, the
  two fixes and what each costs (see [Input](#input))
- colour and shape: what the output is tagged with and why (BT.709 at HD, primaries and
  transfer kept, square pixels)
- resume: the same command resumes; what refuses a resume and why; `--accept-env-change`
- integrity: the per-frame checksums kept with the output, and `seedvr2x verify` to check a
  job's masters before trusting them
- seeds and reproducibility: the same settings and plan (window lengths and tiles, which
  `--plan` prints) give the same output, bit for bit

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
  the joins, frame for frame. (Dropped since, the source being the only input.)
- Segments: joined, they equal the one-file output, byte for byte. The merge rule is
  identical to sptenc's on 20,000 random cases.
- Long real segment: 377 frames at 1080p, 6 shots including 1- and 2-frame ones, run at 540p
  with window 8. Every frame came out, resident RAM stayed flat at 2.3–2.4 GiB, and VRAM
  peaked at 19–26 GiB.

**The build order:**
1. Resume (milestone 4): passed on 2026-10-03. Real episodes needed it: at 4.4 s per 1080p
   frame ([stitching.md](../research/docs/stitching.md#cost-model)), a 24-minute episode
   takes about 40 GPU hours.
2. The `lab` rewrite (milestone 5): passed on 2026-10-04. Then, as small steps of their own,
   next: zscale on one slice (see [Input](#input)) and the default master format
   (`yuv420p10le`, see [Output](#output)), then the model check of [Weights](#weights): today a
   3B or fp8 file is accepted by its name, then fails later or runs unchecked. Then the
   padding of step 0 (see [Pipeline](#pipeline-per-shot)), before the planner counts tokens,
   and the refusal of HDR sources (see [Not in the first version](#not-in-the-first-version)).
   Then `split` in place of `lab` (see [Colour correction](#colour-correction)): the decode
   streams, and its buffer, the histograms and the code ported from numz go, before the
   planner sizes the decode.
3. The model files from seedvr2x's own Hugging Face repo: `models/`'s scripts are done; once
   phase 2's own files are validated, one upload holds every file and the card's table, and
   seedvr2x's pull is pinned to its revision and the files' SHA-256s (see
   [Weights](#weights)). Then the first pass's new work,
   ahead of the planner, since both workflows start from it: the directory input goes, the
   source being the only input; then the shot detector (see [Shot detection](#shot-detection)),
   whose brief is in, and with it the frame index (see [Input](#input)), so a resume seeks
   instead of decoding from the start, and the frame-rate refusal's guidance. The detector's
   threshold is settled (0.3 by default, `--cut-threshold`), with no gate, and `--plan` lists
   the possible cuts. Until it is built, the cuts come from a cut list.
4. The planner, BlockSwap and tiling (milestone 3), then `--until`. On the 96 GB card at
   1080p, windows and the streamed decode already bound memory. The planner's inputs (budget,
   margin, the DiT's and the tiled VAE's peaks, 4K limits, measured times) are in
   [Memory planner](#memory-planner). Consumer cards need it to run 1080p at all.
5. Assembly and `--segment-cmd` (milestone 6), for the regular workflow. The manual sptenc
   workflow already works without it.

After v1, phase 2 brings the other models (see [Weights](#weights)).

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
5. **Colour correction:** our `lab` against numz's `lab`, one batch = one shot.
   - Material: milestone 1's input, clip B and the measurement campaign's full-reference
     clips, plus multi-window shots against numz + `STITCH_LATENT` with `lab` (clip B, 6:2).
   - Both sides are scored alike from 16-bit masters.
   - Accepted when ours is at least as good on every metric. Differences below the
     tolerance count as equal.

     | Metric | Better is | Tolerance |
     |---|---|---|
     | ΔE of the low frequencies to the input; ΔE00 to the ground truth on the full-reference clips | lower | 0.1 ΔE |
     | a*/b* spread | closer to the input's | 1% of the input's spread, or 0.1 unit if larger |
     | Y shift (signed) | closer to 0 | 0.1 level |
     | Hold and low-frequency boundary steps, as the excess over each run's own one-batch output (milestone 2's measure) | lower | 0.02 level |

   - PSNR between ours and numz's is reported for information.
   - A GPU test, `test_lab.py`, holds the result on milestone 1's input, with thresholds set
     from the first runs.
   - Two rules were sharpened after the first run, on 2026-10-04. On a near-neutral sky, 1% of
     the input's a\* spread is 0.0097 unit, far below anything else the milestone resolves,
     hence the 0.1-unit floor, the ΔE tolerance's unit. A raw boundary step also counts each
     pipeline's own motion there (ours' one-batch run 1.223, numz's 1.246): the excess
     isolates the boundary.
   - **Passed on 2026-10-04: 11 of 11 materials,** with the float32 reference (step 3b).
     - The first run, with the encoder's bf16 tensor as the reference, passed 4: ours came out
       brighter than numz's on 5 of 8 full-reference clips, by each clip's own bf16 bias (see
       [Colour correction](#colour-correction), Numerics).
     - ΔE to the input: equal everywhere (within 0.008). ΔE00 to the ground truth: lower than
       numz's on 7 of 8 clips (anime-bright +0.0014). PSNR to numz: 53.6–60.3 dB on the
       one-batch materials, 0.5–1.5 dB above the first run's.
     - Closest calls: anime-sky's a\* spread, 0.023 unit narrower than the input's against
       numz's 0.001 (inside the 0.1-unit floor, outside 1% alone); clip B 6:2's low-frequency
       excess, 0.036 against 0.033. Its hold excess: 0.152 against 0.156.
     - `test_lab.py` holds it on milestone 1's input: PSNR to numz 60.18 dB, floor 59.5.
   - **`split` replaces `lab` (2026-10-06).** The colour study measured it against `lab` on
     every set (see [Colour correction](#colour-correction)): milestone 5's floor, numz's
     `lab`, is cleared. Its step is accepted when:
     - its output equals the study's own `split()` (`colour_variants.py`, built on
       `runtime/colour.py`) on the same decode and reference, within float32 rounding, under
       one 16-bit code, at ×2 and ×4, on one window and on several;
     - the decode streams, with no buffer, and resume stays bit-identical (milestone 4's
       tests);
     - the GPU test that follows `test_lab.py` holds it on milestone 1's input, its thresholds
       set from the first runs.
6. **Assembly (standalone):** the finished file's video timestamps equal the source's, frame
   for frame, and every other stream is copied.
7. **Visual review** of long runs by the user: fast motion, where the frames inside a latent
   group can ghost (see [To measure](#to-measure)); held pictures, where a false cut's step
   would show (see [Shot detection](#shot-detection): the fallback gate); and shots' first
   frames, re-rendered less than the next ones (see [Pipeline](#pipeline-per-shot), step 1).

## Open questions

### To measure
- **Numerics:** settled but for two items ([numerics.md](../research/docs/numerics.md)).
  numz's choices stay (see [Vendored model code](#vendored-model-code)), its bf16 decode
  included, and so does its resize; the padding changes (step 0 of the
  [Pipeline](#pipeline-per-shot)). Left:
  - the chroma kernel at decode, Catmull-Rom upsampling, which shapes what the model sees
    (GPU runs). The master's downsampling is settled: bilinear (see [Output](#output)).
  - a shot padded to 4n + 1 frames by mirroring its end (numz) or repeating its last frame
    (ByteDance)
- **Every fourth frame is the model's best**
  ([numerics.md](../research/docs/numerics.md#every-fourth-frame-the-latent-grid)). After a
  shot's first frame, the causal VAE packs 4 frames per latent, and the last frame of each
  group comes out closest to the ground truth: on 8 of 8 clips by VMAF and LPIPS, 7 of 8 by
  PSNR-Y, 0.8–5.0 dB above the group's second frame, the seeds agreeing within 0.5 dB, with
  `lab` too. On fast motion, the frames inside a group carry ghosts (doubled line art). A
  shot's first frame is the same effect. Colour correction narrows the gap as far as it takes
  from the input, which has no grid: PSNR-Y's from 2.72 dB without it to 1.87 with `lab` and
  1.59 with `split`, colour's (ΔE00 after a 4 px blur) from 0.50 to 0.29 and 0.12. The
  perceptual gap barely moves (LPIPS from 0.021 to 0.018): the inner frames' texture, ghosts
  included, stays the model's ([colour.md](../research/docs/colour.md#the-latent-grid)).
  Nothing measured removes it.
  Taking each frame from a run whose grid ends a group there would take 4 runs per shot, the
  grid shifted by 0–3 frames: 4× the GPU time, so at most a quality mode after v1. Not
  measured (about 2 GPU h on the 8 clips).
- **The sharp 7B at other upscale factors.** It is the default on crops at ×2 (see
  [Weights](#weights)). At ×1.5, ×3 and ×4 the metrics are mixed (×4: PSNR-Y −0.05 dB, DISTS
  worse on 3 of 5 clips, one clip −3.5 dB) and no eyes have looked yet: crops from the colour
  study's runs, on the CPU.
- **A detail scale, not in v1.** The user proposed a "detail recreation" scale for the
  over-rendering at 4K. With the sharp 7B the eyes' main complaint, invented skin texture, is
  gone, and what remains goes both ways with the content: the 7B's sharper thin cel lines, the
  sharp's fused gradient colours and its clouds from a compressed input. A global strength
  would mend one case by harming another, and the global blend measured
  (A · split + (1 − A) · input lightness) washes lines out to the eyes. Should users need one,
  the candidate is a strength that keeps the model's lines and tames its isotropic invented
  texture, measured on the colour study's runs
  ([colour.md](../research/docs/colour.md#step-6-the-sharp-7b-for-the-eyes)).

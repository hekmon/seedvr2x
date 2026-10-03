# seedvr2x design

> Status: **draft for review**, nothing implemented. It collects the decisions taken so far and
> the questions still open. Each decision links to the measurement it rests on in
> [../research/](../research/).

## Goal

A SeedVR2 video upscaler for **long runs** (whole episodes or films) that:
- produces a lossless, correctly tagged master that the next encoding step can trust
- has no visible seams inside a shot
- fits the GPU it runs on without the user tuning memory options. It is developed on a 96 GB
  card, but made for consumer cards of 16–32 GB as well, where the planner, BlockSwap and
  tiling carry the run:
  - At 1080p, without BlockSwap, a 24 GB card holds windows of about 6 latents and a 32 GB one
    13, so many shots are split into windows.
  - A 16 GB card can't hold the 7B fp16's weights (15.35 GiB) at all: it swaps every block,
    and phase 2's smaller files lighten it (see [Weights](#weights)).
  - Below 48 GB, the 1080p VAE decode only fits tiled.
  - These cards also compute more slowly (an RTX 5080 has 84 SMs, the 96 GB card 188), so a
    job lasts longer there, and resume and `--until` matter most.
- can be paused and resumed (run at night, give the computer back in the morning)

It replaces numz's orchestration and I/O (about 14.6k lines, where nearly all of the
[23 bugs](../research/bugs/README.md) live). It keeps ByteDance's model code.

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
  - A directory of segments, such as an earlier `sptenc split`, is still accepted (see
    [Input](#input)).

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
      card, about 75 (≈ 300 frames) on a 96 GB card.
    - At 4K: about 19 latents (≈ 75 frames) on a 96 GB card.
  - A window bounds the DiT only. The VAE's memory depends on the frame size, not on the
    window, since it streams in 4-frame slices (flat beyond ~9 frames). At 4K the untiled
    decode needs ≈ 134 GiB, so 4K needs tiled decoding, about 18 GiB with 1024-px tiles. The
    planner sizes the two separately: window length for the DiT, tile size for the VAE.
  - Not to be confused with the DiT's **attention windows**: the space-time tiles its
    attention works in inside one pass ([attention.md](../research/docs/attention.md)), which
    `na.unconcat_coalesce` counts.
- **Output segment:** an output file. It holds one or more whole shots, at least 5 s long by
  sptenc's rule, or mirrors an input segment. It is the unit of output and of decode resume.
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
v1 runs the reference model, the one milestone 1 is checked with: numz's 7B fp16 DiT
(`seedvr2_ema_7b_fp16.safetensors`) and fp16 VAE (`ema_vae_fp16.safetensors`), Apache-2.0.
ByteDance's sharp 7B, in numz's fp16 file, has the same architecture and format, so it runs
too. Every other model is phase 2's (below).
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
- **numz's fp8 files are plain casts.** Every tensor of the 3B's, and all but the last block's
  in the 7B's, is rounded to `e4m3fn` (3 mantissa bits), with no scale, down to the biases,
  norms, modulation tables, input and output layers and RoPE's frequencies. The Q4_K_M file
  quantises only the 288 large matrices, with a scale every 32 weights, and keeps the other 840
  tensors in fp16: the likely reason it measures closer to the source than fp8.
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
- **Files of our own,** the small tensors and RoPE's frequencies kept in 16 bits, as in the
  Q4_K_M file:
  - fp8 with scales, its matrices multiplied in fp8 on the GPUs that can (Ada and later)
  - NVFP4 for Blackwell: 4-bit weights and activations, a scale every 16 values
  - GGUF with each tensor's bit width chosen by how much the output suffers from it, as
    Unsloth's dynamic method chooses from a calibration set
  - The weights need no data: their scales come from their own values. Activations quantized
    to 4 bits are where quality can go, and on video DiTs mostly because their ranges drift
    across the denoising steps; SeedVR2 runs a single step. Should a calibration set be needed,
    the full-reference clips of each kind of source can serve.
  - Kernels: comfy-kitchen (Comfy-Org, Apache-2.0) quantizes and multiplies FP8, NVFP4 and
    MXFP8, its activations quantized on the fly, but converts no model. ComfyUI's own code is
    GPL-3.0, never copied.
  - The gain is mostly memory. ComfyUI reports about 2× over fp8 or bf16 on Blackwell, but the
    DiT is about a fifth of a 1080p run here: a DiT twice as fast shortens a job by about 10%.
- **One Hugging Face repo** holds every file seedvr2x makes, its own fp16 included, all
  converted from the masters by one documented script: each file's origin can be checked, and
  nothing depends on another party's repo staying as it is. Apache-2.0 allows it, with the
  licence and a notice of the changes.
  - Our fp16 changes no bit of v1's output. numz's 7B fp16 DiT and fp16 VAE are the masters
    rounded to the nearest fp16, ties to even: every element of 1,128 and 250 tensors, under
    the same names. Truncation would differ on half the elements, a cast through bf16 on 87%.
    The 3B and the sharp 7B weren't checked: their masters aren't on the GPU box.
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
- **Guiding users:** the docs, per kind of source (anime, dark, live action…), each figure read
  against the spread between seeds; `--plan` showing what each model gets on the user's own
  card (window length, BlockSwap, tiles, time); whether the planner may pick a smaller file
  when the user allows it; and whether the sharp 7B becomes the default.

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
2. **A directory of segments** (sptenc's split, or any other splitter), for material already
   split. The detector runs over all its frames as over a file's: inside the segments too,
   since a split that merges short scenes (sptenc's, below 5 s) hides real cuts there, and at
   the joins, where one that isn't a real cut is stitched like a long shot. The output mirrors
   the input's segments (same frame ranges and names), so sptenc encodes them as it would its
   own split.
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
`--color-correction {lab,none}`, `lab` by default, is a setting, so a resume compares it.

What numz's `lab` does, per batch slice (numz `4490bd1`):
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

Ours:
- **Licences.** The wavelet functions are StableSR's (non-commercial), so the split is
  rewritten from its method, clean room: another agent read StableSR's code and passed on
  only the method. The CIELAB conversions and the matching are numz's own (Apache-2.0),
  ported and attributed in NOTICE and in the code.
- **Pooled per shot.** One mapping per shot, which is numz with one batch per shot, as the
  stitching study advised: no batch boundary is left for the correction to make.
  - The decode makes two passes. The VAE decodes once into a temporary bf16 buffer in the
    shot's directory: about 12 MB per 1080p frame, so 18 GB per minute of 1080p shot, and
    four times that at 4K.
  - The first pass gathers the histograms; the second maps and writes. A shot's first frames
    come out after its whole decode.
  - Decoding twice instead would add about 54% to the run.
- **Matching.** numz's mapping, each content quantile to the reference's, computed through
  integer histograms pooled over the shot: 2^16 bins per channel, linear within a bin.
  - numz's exact sort doesn't scale to a shot: a pooled GPU sort takes about 104 B per output
    pixel-frame, and torch's CUDA sort stops at 2^31 elements (about 1,035 frames at 1080p).
  - The histograms are 2^16 bins per channel over [−128, 128), 1/256 unit each. That range
    holds the CIELAB value of every RGB in [0, 1] (L* 0–100, a* −86.2 to 98.2, b* −107.9 to
    94.5). A value outside it, or NaN, is refused.
  - The histograms are deterministic. Each value comes out within one bin of the values
    numz's sort gives to the values of its bin, and tied values come out alike. Per pixel,
    the two differ by a median 0.001 units.
  - numz's sort also orders the values inside a bin, and it gives tied pixels different
    reference values according to their position in memory: up to 4 a* units apart on a
    flat test area.
- **Numerics.**
  - The split runs in float32. numz's bf16 is 0.10 level off on average, 1.17 at most.
  - The low band is moved by adding the difference of the low bands: content +
    (low(reference) − low(content)). That is the same sum as high(content) + low(reference),
    without rounding a high band on its own. A float32 high band added back to its low band
    misses the image at 0.4–2.2% of values on test frames, whereas content moved onto itself
    comes back bit for bit.
  - In the pipeline, `lab` runs on the GPU. On the CPU, torch's `pow` makes the conversions'
    last bits depend on the call's size and thread count (258 of 1.56M values), and likely on
    the CPU's vector unit. The resume environment records neither, while the GPU model is
    compared. The CPU path serves the tests. Each shot is converted in the same calls in both
    passes.
  - The reference is the encoder's exact input tensor, rebuilt from the input copy with the
    same transform, as ByteDance does.
  - The corrected frames stay float32 for the writer. The low band and the mapping are
    computed in float32, and a bf16 cast would cut them back to 8 significant bits.
- **Input copy.** An FFV1 `gbrp16le` copy of the 16-bit frames the encode reads, at input
  resolution (about 70 MiB per second of 1080p input). It is written to
  `resume/shot_<start>/input.mkv` while the encode reads the frames, recorded with the shot's
  latent, and kept until its segment is finished. It's exact, sequential, and independent of
  seeking.
  - It is derived data. It's a lossless decode of an input whose content is checked, so it can
    be remade bit for bit with the same ffmpeg and conversions.
    - After an accepted change of ffmpeg or its conversions, a remade copy holds the new
      decode's frames. `lab`'s reference may then differ from the encoder's input by what the
      change changed. Only bit-identity is given up, as with any accepted change.
  - **Integrity, checked end to end.**
    - Each frame gets a CRC-32 over its 16-bit planes as written, kept in
      `resume/shot_<start>/input.crc32`. That file is written whole with the copy, before the
      latent is recorded, and a missing checksum file counts as a missing copy.
    - Each frame read back is checked before `lab` uses it. That costs about 2 ms per 1080p
      frame per pass. The checksums guard the file, not the decode, so a remade copy gets new
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
  - The decode's second pass writes a shot's last frames only once its copy has been read
    whole and checked, since they may finish the output segment, which is then recorded.
  - For `-o x.mkv`, the same files live in `<output>.work`, made and locked at the start as an
    output directory is, so a second run to the same file is refused. It is removed when the
    run ends, however it ends. What a killed run left there (only seedvr2x's files) is emptied
    at the next start; anything else is refused, never deleted.
- With `none`, nothing is copied or buffered, and the decode streams.

Validated against numz's `lab` on the metrics, not bit for bit (milestone 5).

## Output

seedvr2x delivers the upscale losslessly and owns no encoder flags. How the output gets
compressed is the user's choice: afterwards, from the master, or during the run with
`--segment-cmd` (below).

- **FFV1 masters**, every frame a keyframe, per-slice CRCs, exact frame rate, from the float
  frames (no 8-bit step, bugs 09/19):
  - `gbrp16le`: research and archive master, closest to the model
  - `yuv420p10le`, BT.709, limited range, explicit conversion (zscale): the master handed to
    sptenc. sptenc takes such a file as it is, without any conversion, which its MANUAL
    recommends for tools that can write YUV. So the RGB→YUV conversion happens once, in the
    tool that requires zscale, and exactly (white at 940).
    - This is sptenc's intended input, not a workaround. sptenc converts RGB sources with
      swscale (16-bit white at 943 instead of 940), a documented trade-off: zscale isn't in
      every ffmpeg build, and sptenc ships for any build.
    - The master shares the pixel format and tags of `sptenc master`, chroma sited left
      included. The conversion tests (see [Input](#input)) check that zscale sites the chroma
      left, as tagged.
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
  - **Layout, by sptenc's rule.** The detector picks the cuts, then the minimum segment length
    (5 s by default) merges each too-short segment into its shorter neighbour, on the frame
    grid. With a directory of segments as input, the output mirrors it instead.
    - `sptenc encode` takes any segments, so they needn't match sptenc's own split. They would
      with scdet run as sptenc runs it, at the same threshold and minimum.
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
        `.mkv` seedvr2x gives it, a PNG directory's name as it is. So mirrored PNG segments
        `A` and `A.mkv` (from inputs `A.mkv` and `A.mkv.mkv`) keep theirs apart.
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
  - **with sptenc:** the directory as it is, for `sptenc encode <dir> <out> -f <source>` (or
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
- **Choices, in order:** window length (the biggest quality lever: fewer boundaries),
  BlockSwap blocks, VAE tile sizes, then what's left goes to speed: `compile_dit` (−26 to −32%
  DiT time, +0.1 to +1.4 GiB), and `compile_vae` only when its memory fits (it about doubles
  VAE activation memory, for −16 to −19% VAE time).
- `--plan` prints the plan and the time estimate without running.
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
      the writers' chains. The value is the same on two CPUs with one ffmpeg build, and it
      catches a zimg upgrade that ffmpeg's version string doesn't show.
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
    frames of skipped shots are decoded and dropped (decode-and-count), until frame-exact
    seeking is settled.
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

## Documentation for users

seedvr2x's README and manual explain what would otherwise surprise a newcomer, with the reason
and the measurement behind each point.

They open with the [Terms](#terms) in plain words. A window, for instance: "a stretch of a shot
that the model upscales in one go. A long shot is done in several windows that overlap a
little, blended where they overlap before the frames are rebuilt, so the joins don't show.
seedvr2x sizes the windows from your GPU's memory; you can cap them." The model's internal
attention windows stay out of the user docs; they matter only to developers.

**Colour correction, first and in full:**
- **The model itself shifts colours.** Raw SeedVR2 output is more saturated and bluer than its
  input. On clip A, with colour correction off, the saturation spread is +34% on a* and +28%
  on b*, and the low-frequency colour error is ΔE 4.44
  ([quality.md](../research/docs/quality.md#colour-correction)).
- **ByteDance's own pipeline corrects it, with a fix it can't ship.** Its inference script
  runs StableSR's wavelet colour fix after the model (`projects/inference_seedvr2_7b.py:300`),
  but the file is under a non-commercial licence, so the readme asks users to download it
  themselves (`readme.md:142`). Without it, the script prints "Color fix is not avaliable" and
  writes the drifted output.
- **numz bundles that code** (uncredited) as `wavelet`, and adds `lab` on top, its default:
  the wavelet step, then CIELAB colour matching. The wavelet fix does most of the work (ΔE
  1.04); `lab` brings the saturation closer still (ΔE 1.01).
- **seedvr2x:** `lab` by default, with the wavelet split rewritten from its method (no
  non-commercial code) and numz's LAB matching (Apache-2.0). `none` gives the raw model's
  colours, for comparison or for users who grade themselves.

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
- the two workflows: a file in and a finished file out, or sptenc's pre-split directories
- the model: v1 runs the 7B fp16 alone, the reference every check is made against, and phase 2
  brings the others
- lossless delivery: why there are no encoder options, and how `--segment-cmd` compresses with
  the user's own command
- disk use: master sizes per hour, the decode buffer and input copies during a run
- GPU memory, two points:
  - nvidia-smi can show the card nearly full while a phase needs much less: the allocator keeps
    freed memory reserved, and trims it only when an allocation would fail
  - which phase sets the peak: the DiT through the window length, the VAE through the frame
    size, which is why 4K needs tiled decoding
- why zscale is required, and how to get an ffmpeg build that has it
- the refused sources (VFR, interlaced, telecined, rotated, cropped, unusual pixel formats or
  matrices), and what to do with each
- colour and shape: what the output is tagged with and why (BT.709 at HD, primaries and
  transfer kept, square pixels)
- resume: the same command resumes; what refuses a resume and why; `--accept-env-change`
- integrity: the per-frame checksums kept with the output, and `seedvr2x verify` to check a
  job's masters before trusting them
- seeds and reproducibility: the same settings give the same output, bit for bit

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
   drift. Then, as a small step of its own, the model check of [Weights](#weights): today a 3B
   or fp8 file is accepted by its name, then fails later or runs unchecked.
3. The planner, BlockSwap and tiling (milestone 3), then `--until`. On the 96 GB card at
   1080p, windows and the streamed decode already bound memory, and the planner's inputs (4K
   limits, the margin) come from the measurement campaign. Consumer cards need it to run 1080p
   at all.
4. The shot detector, as soon as the scene-detection brief is in (it waits for the user's
   labels), ahead of what is left of 3: both workflows start from it. Until then, the cuts come
   from a cut list.
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
     | a*/b* spread | closer to the input's | 1% |
     | Y shift (signed) | closer to 0 | 0.1 level |
     | Hold and low-frequency boundary steps | lower | 0.02 level |

   - PSNR between ours and numz's is reported for information.
   - A GPU test, `test_lab.py`, holds the result on milestone 1's input, with thresholds set
     from the first runs.
6. **Assembly (standalone):** the finished file's video timestamps equal the source's, frame
   for frame, and every other stream is copied.
7. **Visual review** of long runs by the user.

## Open questions

### Input
- **Own scene detection**, the biggest quality lever outside the model, and what both
  workflows start from (see [Two kinds of users](#two-kinds-of-users-both-first-class)).
  - A missed cut is the costly error. On the first three cuts measured, in one run across the
    cut, the next shot's first frame lost 2.4–9.6 dB of PSNR-Y on two of them, the loss lasted
    up to 13 frames, and up to 5% of the previous shot showed in it
    ([cuts.md](../research/docs/cuts.md)).
  - A false cut costs no fidelity, at most a low-frequency step on a continuous shot, but a
    burst of them chops a shot into many small pieces. scdet's first passes, not labelled yet:
    on action anime it fires in bursts on new drawings after held frames and on effects (at
    threshold 10, half the shots of a dark anime episode last under 0.5 s), and it misses some
    dark cuts ([PROGRESS.md](../research/PROGRESS.md)).
  - So the detector must catch every real cut first, then not fire in bursts. Candidates:
    - ffmpeg's scdet, as sptenc runs it: no new dependency
    - PySceneDetect's adaptive and content detectors: they detect fades, but add a dependency
    - TransNetV2, a neural network trained to find shot boundaries (MIT licence, open weights)
    - a detector on our own decoded frames
  - Chosen on the user's labels of the measurement campaign's review sheets, which must first
    hold every candidate's detections. The scene-detection brief also settles the threshold and
    whether shots need a minimum length of their own: cuts.md finds a short shot better run
    alone than merged into its neighbour, from 1 frame on.

  Output segments keep sptenc's minimum length (see [Output](#output)).
- **Frame-exact access into long-GOP sources**, to resume a shot and to read its input frames
  again for colour correction. Three ways:
  - ffmpeg's accurate seek: fast, but trusts timestamps
  - decoding from the start and counting: exact, but slow on a film. Resume uses it for now.
  - a lossless intermediate: exact and fast, but large

  Plan: measure accurate seek against decode-and-count on real long-GOP files.

### To measure
- **Numerics:** numz's or ByteDance's, change by change, once it's confirmed which ones really
  differ on the 7B fp16 path (see [Vendored model code](#vendored-model-code)):
  - the RoPE angle precision
  - the fp32 → fp16 → bf16 cast of input frames
  - the fp16 weights cast to bf16
  - VAE mode vs sample
  - the VAE decode's output precision. bf16 keeps 8 significant bits, ≈ 8-bit steps near
    white, so a 16-bit master can't hold more there than the decode gives, and smooth bright
    gradients could band. fp16 keeps 11 bits but VAE activations may overflow it; fp32 costs
    memory and time.
  - the chroma kernels: Catmull-Rom upsampling at decode, bilinear downsampling for the
    `yuv420p10le` master
  - the input preparation: the resize kernel and its `antialias` flag, and multiples of 16
    reached by padding (numz) or cropping (ByteDance)
  - a shot padded to 4n + 1 frames by mirroring its end (numz) or repeating its last frame
    (ByteDance)
- **Decode resume granularity:** whether sub-segment decoding with warm-up latents is
  bit-identical.
- **A shot's first frame.** The causal VAE encodes it alone, so it comes out less restored,
  closer to the input ([quality.md](../research/docs/quality.md#--prepend_frames)). numz has
  such a frame at every batch; seedvr2x only at each shot's start, right after a cut, where
  the eye is least likely to notice. To measure: whether prepending mirrored frames to each
  shot (numz's `--prepend_frames`, kept inside the shot) is worth its cost of about one latent
  per shot.
- **4K and long windows:** the planner's limits on large outputs, where the DiT window is the
  constraint.

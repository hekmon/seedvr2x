# seedvr2x: the manual

The [README](README.md) says what seedvr2x is, what it needs and how to install it. This manual
explains what would otherwise surprise you, with the reason and the measurement behind each
point. [DESIGN.md](DESIGN.md) holds every decision in full, and
[../research/docs/](../research/docs/) the measurements, linked where a figure comes from.

> **Not built yet.** seedvr2x is being built, and this manual describes version 1 as
> [DESIGN.md](DESIGN.md) specifies it. Not built today: the planner that fits a job to the
> card's memory, with its swapped blocks and its tiles; `--until`; the assembly of one finished
> file, and `--segment-cmd`. Wherever this manual speaks of them, it describes the design; the
> sections that are theirs are marked *not built yet*, and the README's
> [Status](README.md#status) has the list.

What surprises a newcomer most:

- **It takes one video file, and nothing else:** no directory of segments or of frames. Every
  cut is seedvr2x's own, or yours through a cut list
  ([the two workflows](#the-two-workflows)).
- **It has no batch size,** and none of the options that go with one in other SeedVR2 tools
  ([the pipeline](#the-pipeline-for-seedvr2-users-coming-from-numz)).
- **It reads the whole source before it upscales anything.** That first pass counts the
  frames, checks the frame rate against them and finds the cuts, in minutes, up to an hour for
  a long 4K film ([how shots are found](#how-shots-are-found)).
- **It is slow, and made to be stopped.** The model took about 4.4 s per 1080p frame on the
  card it was measured on, 40 GPU hours for a 24-minute episode
  ([stitching.md](../research/docs/stitching.md#cost-model)). That card's clock was capped by a
  fault (about 580 MHz under load, where it runs 2.6 GHz free): a healthy one should take 2 to
  3 times less, which isn't measured yet. Either way a job lasts nights, hence
  [resume](#resume) and [`--until`](#stopping-at-a-set-time).
- **Its output is big:** about 100–135 GiB per hour of 1080p, lossless
  ([disk use](#disk-use)). Compressing it is your choice and your encoder's
  ([lossless delivery](#lossless-delivery)).
- **Its output is redrawn,** not a larger copy of the source: pixel metrics score it below a
  plain upscale, while eyes prefer its redraw of lines and text
  ([what the model does to a picture](#what-the-model-does-to-a-picture)).
- **It refuses rather than degrades.** A source it can't read right, a size that doesn't fit
  the card, a tool that would be inexact: each is refused with the reason and the way out
  ([refused sources](#refused-sources)).
- **nvidia-smi can show the card nearly full** while the job needs much less
  ([GPU memory](#gpu-memory)).

## Contents

- [The terms, in plain words](#the-terms-in-plain-words)
- [Colour correction](#colour-correction)
- [The pipeline, for SeedVR2 users coming from numz](#the-pipeline-for-seedvr2-users-coming-from-numz)
- [The two workflows](#the-two-workflows)
- [How shots are found](#how-shots-are-found)
- [The model](#the-model)
- [What the model does to a picture](#what-the-model-does-to-a-picture)
- [Lossless delivery](#lossless-delivery)
- [Disk use](#disk-use)
- [GPU memory](#gpu-memory)
- [Small cards](#small-cards)
- [Why zscale is required](#why-zscale-is-required)
- [Refused sources](#refused-sources)
- [Colour and shape of the output](#colour-and-shape-of-the-output)
- [Resume](#resume)
- [Stopping at a set time](#stopping-at-a-set-time)
- [Integrity](#integrity)
- [Seeds and reproducibility](#seeds-and-reproducibility)
- [Options](#options)
- [Exit codes](#exit-codes)
- [The files of a job](#the-files-of-a-job)
- [Environment variables](#environment-variables)

## The terms, in plain words

- **Frame:** one picture of the video.
- **Shot:** the frames from one cut to the next. seedvr2x upscales each shot on its own:
  nothing crosses a cut.
- **The model's two parts.** The **VAE** turns frames into a compressed form the model works
  on (the encode), and that form back into frames (the decode). The **DiT** is the part that
  restores the picture, in between. At 1080p the VAE takes about four fifths of a job's time.
- **Latent:** the VAE's compressed form of frames. A shot's first latent holds 1 frame and
  every next one holds 4, so 4n + 1 frames make n + 1 latents. Windows are counted in latents:
  13 latents are 49 frames.
- **Window:** a stretch of a shot that the model upscales in one go. A long shot is done in
  several windows that overlap a little, blended where they overlap before the frames are
  rebuilt, so the joins don't show. seedvr2x sizes the windows from your GPU's memory; you can
  cap them.
- **Tile:** a square piece of the picture. When a whole frame doesn't fit the GPU's memory, the
  VAE works tile by tile; the tiles overlap and are blended.
- **Swapped block:** the DiT is a stack of 36 blocks. A swapped block waits in system memory
  and goes to the GPU for its turn only, which leaves the GPU's memory to the window. This is
  BlockSwap.
- **First pass:** the decode of the whole source that comes before the upscale. It counts the
  frames, checks their timestamps, looks for combed frames and runs the shot detector.
- **Master:** a lossless video file: every frame as the upscale made it, to be kept, or
  compressed by an encoder.
- **Output segment:** an output file. It holds one or more whole shots, and lasts 5 seconds at
  least by default. A job's output is its segments, in a directory, or joined into one file.
- **Unit:** a piece of saved work: a shot's encode, a window, a segment's decode. A stop loses
  only the unit in progress.
- **Plan:** what seedvr2x decides for a job on your card before it starts: the longest window,
  how many blocks are swapped, the tiles.

## Colour correction

Colour correction comes first, and in full: the model changes the colours, and every tool
around it corrects them its own way.

### The model itself shifts colours

Raw SeedVR2 output is more saturated than its input, and its hue moves with the content.

- On a dark anime clip, with colour correction off, the colours spread 34% wider on the
  green-red axis (a\*) and 28% on the blue-yellow one (b\*), towards blue, and the coarse
  colour is ΔE 4.44 from the input's
  ([quality.md](../research/docs/quality.md#colour-correction)). ΔE is a distance between two
  colours: lower is closer.
- Over 10 clips that have an original to compare with, the output is more saturated on 9, and
  shifts towards yellow on 7. "Towards blue" held on the dark anime clips alone
  ([colour.md](../research/docs/colour.md)).
- Tiles add their own: a tiled VAE shifts each tile's level and colour uniformly, by up to 2
  8-bit levels in the decode and 4 in the encode with 512-pixel tiles
  ([vram.md](../research/docs/vram.md#tiling)).

### How the others correct it

- **ByteDance's own pipeline corrects it, with a fix it can't ship.** Its inference script runs
  StableSR's wavelet colour fix after the model, but that file is under a non-commercial
  licence, so ByteDance's readme asks users to download it themselves. Without it, the script
  prints "Color fix is not avaliable" and writes the drifted output.
- **numz bundles that code,** uncredited, as `wavelet`, and adds `lab` on top, its default: the
  wavelet step, then colour matching in CIELAB. The wavelet fix does most of the work (ΔE 1.04
  from the input on that dark anime clip); `lab` brings the saturation closer still (ΔE 1.01).
- **What `lab` leaves.** Against the original, on four animated clips, it brings the coarse
  colour error down by 37–55%, to a ΔE00 of 1.07–1.58, where the input itself, plainly
  upscaled, is at 0.6–0.7 ([numerics.md](../research/docs/numerics.md)). Its wavelet step
  replaces only what is coarser than about 30–60 pixels, while the model's extra saturation
  and drift sit finer ([quality.md](../research/docs/quality.md#colour-correction)). Its
  matching step then corrects how the colours are distributed over a batch, not where they
  are, and that is what moves `lab`'s colours.

### What seedvr2x does: `split`

`split` is seedvr2x's own correction, and its default. Frame by frame, it takes from the
source what the source holds, and keeps from the model what the model adds:

- **the source's coarse colour,** below about 1.6 source pixels. That is the source's own
  colour resolution: a 4:2:0 video has one colour sample every 2 pixels.
- **the source's coarse brightness,** below about 6.5 pixels of the output;
- **the model's detail,** everything finer;
- **no matching of colour statistics,** the step that moves `lab`'s colours.

Why these scales ([colour.md](../research/docs/colour.md)):

- Colour taken finer than the source's own colour resolution costs perceptual quality (LPIPS
  and DISTS); taken coarser, it leaves colour fringes at edges. The scale follows the upscale
  factor, to stay near 1.6 source pixels: checked at ×1.5, ×2, ×3 and ×4. Between those
  factors the rule interpolates; below ×1.5 and above ×4 it is untested.
- Brightness at 6.5 pixels works at every factor measured. Coarser, at numz's 13 pixels, the
  model's own brightness flickers more than `lab`'s; finer, at 3.2, fine texture starts to go.
- Each frame needs only itself and its source frame: nothing is pooled over a shot, so the
  frames stream out as they are decoded, with no second pass.

Against numz's `lab`, as seedvr2x first built it, measured with the regular 7B on shots of 45
frames. Each line counts the clips where `split` is better / worse / within the spread
between seeds:

| Material | Colour error (ΔE00) | Other scores |
|---|---|---|
| 1080p, ×2 from a mildly degraded source (8 clips, 3 seeds) | −0.30 at full detail, −0.46 after a 4-pixel blur, −0.29 after 16 (8 / 0 / 0 each) | PSNR-Y +0.63 dB (4 / 0 / 4); LPIPS −0.0096 (7 / 0 / 1); DISTS −0.0023 (2 / 2 / 4); flicker −0.44 (6 / 0 / 2), its coarse part −0.76 (8 / 0 / 0); colour fringes −0.46 (7 / 0 / 1); no clip loses detail |
| The same, from a heavily degraded source (7 clips) | −0.27 after a 4-pixel blur (7 / 0 / 0) | PSNR-Y +0.65 dB (5 / 0 / 2); LPIPS −0.0044 (4 / 1 / 2); coarse flicker −0.70 (7 / 0 / 0); DISTS +0.0011 (1 / 4 / 2): 0.003 to 0.005 worse on four compressed inputs |
| Through the default master, 10-bit 4:2:0 | | VMAF +3.7 (7 / 0 / 1), no banding added |
| ×4 (5 clips) | −0.18 after a 4-pixel blur (5 / 0 / 0) | PSNR-Y +0.55 dB (3 / 0 / 2); LPIPS −0.0020 (4 / 0 / 1); DISTS +0.0009 (2 / 2 / 1); coarse flicker −0.58 (5 / 0 / 0) |
| 4K, ×2 from 1080p (13 shots: clean live action, animation made in 4K, a grainy scan) | −0.33 on the coarse colour (13 / 0 / 0) | PSNR-Y +0.57 dB, VMAF +5.3, coarse flicker −0.58 (13 / 0 / 0 each); LPIPS −0.0086 (10 / 3 / 0); DISTS −0.0012 (6 / 6 / 1) |

- **To the eye,** on crops beside the original: never worse than `lab`, and better where
  `lab`'s colour noise shows, a cartoon in fast motion and a compressed live-action shot.
- **Window joins:** the coarse step at a long shot's joins is 40% lower (0.046 to 0.028 of an
  8-bit level, on one clip).
- **Tiles:** after `split`, the worst 60-pixel block of a frame moves by 0.04–0.50 8-bit levels
  at 1080p and 0.14–0.49 at 4K, with decode tiles down to 512 pixels, and by up to 0.77 with
  512-pixel encode tiles: about half of what `lab` leaves. So colour sets no lower limit on
  the tiles, down to 512 pixels. It holds on flat content at 4K too, a smooth sky or painted
  animation, where the worst block moves by up to 0.77, in painted texture
  ([colour.md](../research/docs/colour.md#4k-tiles-on-flat-content)).
- **Cost:** 22.8 ms and 0.56 GiB of GPU memory per 4K frame, against `lab`'s 104.6 ms and
  0.84 GiB.
- **Licences:** the wavelet split is StableSR's method, whose code is under a non-commercial
  licence. seedvr2x's is written from the method, without that code, and holds no numz code
  either.

What `split` doesn't do: it converts no gamut and grades nothing. The primaries and transfer
of the source are the output's ([colour and shape](#colour-and-shape-of-the-output)).

### `--color-correction none`

`none` gives the raw model's colours, which drift: for comparison, or if you grade the result
yourself. The choice is one of the job's settings: a job started with one isn't resumed with
the other.

### The input copies

To correct a frame, seedvr2x needs the source frame the model was given. It keeps a lossless
copy of each shot's source frames, at the source's resolution, written while the shot is
encoded and removed when the shot's output segment is finished: about 32 MiB per second of
1080p source (1.38 MB per frame on one clip). See [disk use](#disk-use).

- Each copied frame has a checksum, checked when it is read back, and ffmpeg's own messages
  are read too. Both are needed: ffmpeg decodes through a damaged part of a frame with nothing
  but a line among its messages, and some damage it doesn't report at all
  ([integrity](#integrity)).
- A copy that is missing or damaged stops the run, naming it; the next run makes it again from
  the source and keeps the shot's finished work. A lost temporary file never ends a job of
  many hours.
- With `none`, nothing is copied.

## The pipeline, for SeedVR2 users coming from numz

If you come from numz's ComfyUI-SeedVR2_VideoUpscaler, you will look for `--batch_size`,
`--temporal_overlap` or `--chunk_size`, and not find them. They aren't missing: the work they
tune is done another way. This section follows numz's pipeline with each option where it acts,
then seedvr2x's, then says why the options have nothing left to do. The
[table](#numzs-options-one-by-one) at the end is the reference.

### numz: phases over fixed batches

- It reads the whole clip into RAM, or each `--chunk_size` chunk of it.
- It cuts the clip into batches of `--batch_size` frames that ignore the cuts, each padded to
  4n + 1 frames.
- It then runs four phases, each over every batch: encode every batch, upscale every batch,
  decode every batch, colour-correct every batch.
- The latents wait in RAM between phases (`--tensor_offload_device cpu`), and the decoded
  frames gather in RAM before they are written. Without the cache options, the DiT is loaded
  for each file or chunk ([cli-flags.md](../research/docs/cli-flags.md)).

So RAM grows with the clip, every batch boundary is a visible jump, and a stop loses
everything. Most options patch what this design creates:

- `--temporal_overlap` cross-fades the batch boundaries in pixels, after decoding. Its weights
  only blend with odd values of 3 or more
  ([bug 06](../research/bugs/06-temporal-overlap-blend-weights.md)): 1, 2 and 4 are a hard
  switch that still costs compute. Even corrected, a pixel cross-fade removes 67% of the jump,
  softens the mixed frames by about 20%, and costs 16% more.
- `--chunk_size` bounds the RAM, and leaves a hard seam at every chunk boundary.
- `--cache_dit` and `--cache_vae` keep the models between files or chunks, since each chunk is
  a full run.
- `--prepend_frames` makes the clip's first frame not a batch's first. The VAE encodes a
  batch's first frame alone, and it comes out less restored
  ([quality.md](../research/docs/quality.md#--prepend_frames)). On one GPU, numz doesn't remove
  the prepended frames ([bug 05](../research/bugs/05-prepend-frames-not-removed.md)).
- `--uniform_batch_size` pads a short last batch to full size.
- The offload devices, `--blocks_to_swap` and the tile sizes fit the GPU, by hand.

### seedvr2x: per shot, from cut to cut

There is no batch. A shot of any length is padded once to 4n + 1 frames, because the VAE's
first latent holds 1 frame and every next one holds 4, and its windows are as long as the
memory allows.

- **Encode.** Frames stream from ffmpeg's decoding of the source into the VAE at the VAE's own
  pace: 5 frames, then 4 per latent. The shot never has to sit in memory.
- **Upscale.** A shot that fits the memory is one window. A longer one runs as windows of
  balanced lengths that share 2 latents (8 frames) with each neighbour; the shared latents are
  mixed before the decode, so no boundary is left inside a shot.
- **Decode.** One streamed pass over the shot, then the colour correction, and the frames are
  written into the output segments as they come out.
- **Memory.** Only the latents are ever whole in memory, so RAM stays flat whatever the length:
  2.3–2.4 GiB over a 377-frame job of 6 shots, run at 540p.
- **Tiled, it differs** *(not built yet)*. When the VAE works in tiles, as the decode does
  below 48 GB at 1080p and on every card at 4K, each tile runs over the whole shot, so the
  shot's frames wait in a scratch until the last tile: system memory up to an allowance, then
  a file beside the saved work. It takes 12.7 MB per 1080p frame and 50.5 MB per 4K frame.
- **The models stay loaded** for the whole job.
- **Every unit is saved,** so a job stops and resumes, bit for bit.

### Why the options go

- **Nothing to cross-fade.** No batch boundary is left inside a shot. Windows share whole
  latents, so there is no odd-or-even quirk, and mixing them removes 80% of a window
  boundary's jump, with no softening, for about 9% more compute on the clip measured, and an
  estimated 12% on long shots ([stitching.md](../research/docs/stitching.md)).
- **No chunks, no caches.** Streaming keeps RAM flat, and one process runs the whole job.
- **No prepended frames.** Only a shot's first frame is encoded alone, right after its cut,
  where it is least visible; numz has one at every batch. Prepending frames to it was measured
  and not kept: it costs 2–8% more GPU time on animation, and takes 0.6–2.8 dB off that frame's
  closeness to the original on 7 shots of 11
  ([cuts.md](../research/docs/cuts.md#a-shots-first-frame-prepending-mirrored-frames)).
- **No short last batch.** Windows are balanced: 21 latents under a cap of 6 run as windows of
  6, 6, 6, 6 and 5.
- **No memory options to tune.** The planner fits the GPU from its free memory: the window,
  the swapped blocks, the tiles. They stay yours to fix ([GPU memory](#gpu-memory)).

### numz's options, one by one

| numz option | seedvr2x |
|---|---|
| `--batch_size`, `--temporal_overlap`, `--prepend_frames`, `--uniform_batch_size` | gone: shots, windows and the mixing of shared latents replace them, sized by the planner; `--window` caps the windows |
| `--chunk_size`, `--cache_dit`, `--cache_vae` | gone: one process for the whole job, streaming |
| `--blocks_to_swap`, the tile sizes, the offload devices | *Not built yet:* chosen by the planner; `--swap-blocks`, `--encode-tile` and `--decode-tile` fix them, exactly or refused; no offload device to name |
| `--compile_dit`, `--compile_vae` | not in seedvr2x: `torch.compile` is an open question, to be measured on seedvr2x itself. Compiled, the DiT would compile again at every new window length, and numz's compiled VAE takes twice its memory |
| `--attention_mode` | gone: FlashAttention 2 when installed, else PyTorch's SDPA, logged |
| `--color_correction` | `--color-correction`: `split` by default, or `none`; numz's `lab`, `wavelet`, `wavelet_adaptive`, `hsv` and `adain` are gone |
| `--input_noise_scale` | gone: numz's own addition, which only degrades |
| `--latent_noise_scale` | gone from version 1: ByteDance uses 0 |
| `--seed` | `--seed`, kept |
| `--resolution`, `--max_resolution` | `--resolution`, the output's short side; nothing stands for `--max_resolution` today |
| `--output_format`, `--video_backend`, `--10bit` | `--format`: FFV1 masters (`yuv420p10le` by default, `gbrp16le`) or 16-bit PNG |
| `--dit_model`, `--model_dir` | `--dit-model` and `--model-dir`, the sharp 7B by default; the files come from seedvr2x's own Hugging Face repository, pinned; no model file is ever deleted |
| `--cuda_device` | one device per process |

## The two workflows

Both take the source, one video file: any codec ffmpeg decodes, as it is, with no intermediate
master to make first. They differ only at the end.

- **A file in, a finished file out** *(not built yet: today `-o x.mkv` writes one FFV1 master,
  video only)*. `-o film.1080p.mkv` gives one file: the upscaled video, joined from its
  segments by stream copy, with the source's audio, subtitles, chapters and attachments. Each
  segment's duration comes from its frame count and every timestamp sits on the frame grid, so
  the video's timestamps are the source's, frame for frame. Joined carelessly, the video
  drifts: 46 ms behind the audio over a 163-segment episode, in sptenc's measurements.
- **The source in, a directory of segments out.** Any `-o` that isn't a `.mkv` path is a
  directory: the output segments, their checksums and the job's record
  ([the files of a job](#the-files-of-a-job)). It is what `sptenc encode <dir> <out> -f <source>`
  takes, or `sptenc concat` for one master. The default master, 10-bit 4:2:0, is the format
  sptenc takes as it is, with no conversion. Another video suffix (`.mp4`, `.mov`…) is refused.

**Output segments follow sptenc's rule.** The cuts give the segments, then each segment
shorter than the minimum (5 seconds by default, `--min-segment`) is merged into its shorter
neighbour. The rule counts in frames: 5 seconds are 120 frames at 24000/1001.

- The merge costs no quality. A segment can hold several shots, and the model still starts
  afresh at each cut inside it: the shots are the model's units, the segments the encoder's.
- It saves what short segments cost an encoder: a forced keyframe each, and too few frames to
  make up for it.
- `--min-segment 0` keeps a segment per shot.

**A directory of segments is no input.** Segments cut by another tool carry that tool's joins,
and those aren't all cuts: of an earlier split's 411 joins, 63 scored under the threshold
ffmpeg's scene detector takes for a cut
([scene-detection.md](../research/docs/scene-detection.md)). A
directory is refused, with the way to join segments split losslessly back into one file. The
join must keep the frame grid, as `sptenc concat` does: ffmpeg's concat demuxer gives every
frame back, but each join's timestamps 1 ms early, and such a file is refused once it has
drifted half a frame from its rate, after 21 joins at 24 fps *(not built yet: today the drift
is a warning, and the refusal advises the concat demuxer)*.

## How shots are found

Finding the cuts is the biggest lever on quality outside the model itself.

### Why cuts matter

- Each latent packs 4 frames, so a cut inside a group of 4 mixes both scenes in one latent.
- The VAE carries what it has seen from frame to frame, and windows are mixed where they
  overlap: across a cut, both would blend two scenes.
- Measured ([cuts.md](../research/docs/cuts.md)): a missed cut costs the next shot's first
  frames, 5–31 dB·frames of PSNR-Y on four cuts of six. A false cut costs nothing measured:
  at most a small step in coarse brightness on a shot that was continuous.

So a missed cut costs more than a false one, and seedvr2x errs on the side of cutting.

### The detector

The detector is [TransNetV2](https://github.com/soCzech/TransNetV2), the official model. It
runs on the GPU during the first pass, on every frame, and gives each frame a probability of
being a shot's last. A cut is drawn after the peak of each run of frames that reaches the
threshold, 0.3 by default.

- **Every detection starts a shot, however short.** There is no minimum shot length: real cuts
  come 1–3 frames apart in action anime, and a short shot is better run alone than merged into
  its neighbour, from 1 frame on.
- **Chosen on labelled cuts,** 130 candidates in two rounds, drawn where detectors disagree
  and judged blind
  ([scene-detection.md](../research/docs/scene-detection.md#rounds-1-and-2-pooled)).
  Against ffmpeg's scdet as usually run (threshold 10):

  | | TransNetV2 at 0.3 | scdet at 10 |
  |---|---|---|
  | Live action: cuts found | 98% | 25% |
  | Live action: detections that are cuts | 84% | 89% |
  | Animation: cuts found | 90% | 84% |
  | Animation: detections that are cuts | 82% | 62% |

  scdet takes every new drawing after a held one for a cut, so it fires in bursts: 351 per
  hour of animation, against TransNetV2's 19.
- **Why 0.3.** Against 0.5 it misses an estimated 92 cuts per hour of animation instead of
  122, and 19 per hour of live action instead of 70, for more false cuts (82% of animation's
  detections are cuts, against 89%). Below 0.3 the trade turns: at 0.2 or 0.15 each cut found
  costs 6 to 14 false cuts on animation and 14 to 21 on live action, and most of the misses
  score under 0.1, out of any threshold's reach.
- **The first pass's cost.** It decodes every frame once, in software, and the detector scores
  each. The decode alone runs at 730–2,360 frames per second on 1080p H.264 and 171 on an HEVC
  master ([seeking.md](../research/docs/seeking.md)); the detector scores 810 frames per
  second on the capped card, 107 s per hour of source, and is then the slower of the two: the
  pass ran at 723–808 frames per second on five films. A 4-hour 4K film's pass stays about an
  hour. Against days of upscale, that is nothing.
- **Kept with the job.** The probabilities are saved with the first pass's record, so another
  threshold gives its cuts with no second decode, and a resumed job keeps its cuts.

### When it gets it wrong

- **Fast action anime** is its blind spot: it misses about 60–70 cuts per hour there (25–180
  at the ends of the estimate), cuts it scores under 0.1.
- **Fast camera work,** found in animation made natively in 4K: in a fast camera flight every
  frame differs from the last by as much as a cut does. Two real cuts there scored 0.207 and
  0.155, while a continuous shot in the same flight peaked at 0.232: no threshold separates
  them
  ([scene-detection.md](../research/docs/scene-detection.md#native-4k-animation-cuts-inside-fast-camera-motion)).
- **Still pictures** can give false cuts. They cost no fidelity, but a step may show on a held
  drawing.

What you can do, from the lightest:

1. **`--cut-threshold P`**: lower where cuts go missing, higher where false ones show. Erring
   lower is the cheaper mistake. It is one of the job's settings.
2. **The possible cuts.** `--plan FILE` writes the cuts found to FILE and, commented, the
   possible ones: each run of frames peaking from 0.1 up to the threshold. Check them in a
   player, and uncomment the real ones. Expect about 250 possible cuts per hour of animation
   and 100 of live action, one real cut in 7 to 18 of them.
3. **A cut list of your own,** `--cuts FILE`, which replaces the detection: the planned list
   edited, or any list in the same format. `--cut-threshold` is refused with it.

### The cut list

`--plan FILE` stops before the upscale and writes the list. A 4.4-minute short made in 4K has
61 cuts at 0.3 and 39 possible cuts; planned again at 0.2, with no decode, 76 and 24. The
list looks like this, after a few comment lines that say what it is:

```
1234 0.993 0:00:51.469
# 1300 0.212 0:00:54.221
1391 0.871 0:00:58.017
```

- A line per cut: the frame the new shot starts at, counted from 0; the detector's peak
  probability, which it gives the frame before; and the cut frame's time from the file's
  start, rounded up to the millisecond (here at 24000/1001 frames per second).
- A possible cut is a commented line. Uncomment it to keep it; delete a line to drop a cut;
  add a line to cut somewhere else.
- seedvr2x reads the frame number alone: the first frame of each shot except the first, one
  per line, `#` for comments, anything after the number ignored. A list from another tool
  needs nothing more. Times aren't read: only a frame number is exact.

Then run with `--cuts FILE`. With an output directory, `--plan` also leaves the job's record
and frame index there: the same command without `--plan` runs the plan with no second first
pass, and another `--cut-threshold` or a `--cuts` list plans it again from the saved
probabilities.

### Checking a cut in a player

The time on each line is there for this. Measured on VLC 3.0.24:

- VLC's Go to Time (Ctrl+T) takes whole seconds. Type the hours, minutes and seconds of the
  cut's time, then step a frame per press with E: the cut comes within the second.
- VLC doesn't step back (Shift+Left goes 3 seconds back).
- On a `.ts` file its seeks are rough: go a few seconds earlier, and play.
- *Not built yet:* `--plan FILE` also writes a playlist beside the list, an entry per possible
  cut starting a second before it, so that VLC goes from one to the next with no time to type.

mpv lands on the cut frame from the list's time itself.

## The model

Version 1 runs ByteDance's SeedVR2 7B in float16, with its VAE in float16. ByteDance
publishes two 7B models of the same architecture, memory and time:

- **The sharp 7B is the default.** Judged by eye on 75 crops at 4K and 1080p, it was preferred
  or no different on 67 (preferred on 27, the regular 7B on 8). At 4K it invents less of the
  fine texture the eyes rejected with the regular 7B, whose skin came out "almost reptilian"
  on a grainy film. No measured guard gets worse: the same banding and colour, fewer colour
  fringes, less flicker. At ×2 to 1080p the metrics barely tell the two apart
  ([colour.md](../research/docs/colour.md#step-6-the-sharp-7b-for-the-eyes)). Despite its
  name, it adds no more fine texture than the regular one there, on average.
- **The regular 7B is the reference** every check of seedvr2x is made against, numz's own
  runs among them. `--dit-model seedvr2x_ema_7b_fp16.safetensors` runs it.
- **At other upscale factors** than ×2, the sharp 7B's lead isn't settled. The metrics side
  with it at ×1.5 (3 clips of 3) and mostly at ×3, and are mixed at ×4, where one clip is
  3.5 dB further from the original; no eyes have looked yet.

The files, where they come from and how they are checked are in the README's
[Model files](README.md#model-files), with the table of every format. A later phase brings
the smaller files of both 7Bs, each with its measured quality. The VAE stays in float16: its
weights take 0.47 GiB, so a smaller file would save nothing, and its memory is the pictures it
computes, which tiling bounds.

## What the model does to a picture

SeedVR2 restores: it redraws the picture rather than enlarging it. That is what it is for,
and it changes what a measurement means. A metric that compares the output with an original
rewards a copy of that original's own blur and grain. seedvr2x uses such metrics as guards
against damage (colour drift, flicker, banding, artefacts), and eyes for the look.

### On live action

Measured ([numerics.md](../research/docs/numerics.md)):

- The perceptual metrics side with the model over a plain upscale, and the pixel ones with
  the plain upscale.
- It redraws grain and detail of the right kind at the wrong place.
- Its grain follows its input's. From an input that kept a quarter of a film's grain, it gave
  back about half; from one that kept most of it, a little more than the original's.

### At 4K

×2 from 1080p, it redraws rather than enlarges
([colour.md](../research/docs/colour.md#step-5-4k-on-clean-and-grainy-sources-and-a-detail-strength)):

- Metrics against an original put it behind a plain upscale, on clean sources as on grainy
  ones: on PSNR-Y and VMAF on every one of 17 shots (PSNR-Y 4–9 dB lower), on LPIPS and DISTS
  on most. It adds less banding than the plain upscale on 16 shots of 17, and the eyes prefer
  its redraw of lines and text.
- The regular 7B invents fine texture the eyes rejected. The sharp 7B, the default, invents
  less of it and keeps the redraw: 2.6–5.7 times the original's fine detail per kind of source
  on 13 shots, against the regular 7B's 3.4–7.3.
- A more compressed input changes neither.
- Clouds and smooth gradients from compressed sources can still show light artefacts.

There is no strength or detail option in version 1. What remains of the over-rendering goes
both ways with the content, and the one global blend measured washes lines out to the eye.

## Lossless delivery

seedvr2x delivers the upscale, losslessly, and has no encoder options: no codec, no quality
setting, no preset. How the output gets compressed is your choice, with your encoder and its
flags: afterwards, from the master, or during the run with `--segment-cmd`.

### The formats

`--format` chooses among three, all written from the model's float frames, with no 8-bit
step:

- **`yuv420p10le`, the default:** an FFV1 master in 10-bit YUV 4:2:0, limited range, BT.709 at
  HD sizes. It is what encoders take. The conversion from the model's RGB happens once, here,
  exactly (white at 940). Left to whatever reads an RGB master, it can come out a little off:
  ffmpeg's default converter puts white at 943.
- **`gbrp16le`:** an FFV1 master in 16-bit RGB, the closest to the model. It is five times as
  big (measured with `lab`, the earlier correction), and meant for precision work and
  measurements on short samples, not for whole films.
- **`png`:** 16-bit PNG files, a directory per segment, frames numbered from `000000.png`.

The FFV1 masters have every frame a keyframe, a checksum per slice, and the source's exact
frame rate. The 4:2:0 master's colour is reduced with a bilinear filter, measured against
four sharper ones: the lowest colour error on every one of 8 clips
([numerics.md](../research/docs/numerics.md#the-masters-chroma-420-kernels-and-zscales-slices)).

### `--segment-cmd`: your encoder, during the run

*Not built yet.*

A lossless film is too big for most disks, and one pipe into one encoder can't be paused. So
seedvr2x runs your command once per output segment:

```
uv run seedvr2x film.mkv -o film.1080p.mkv --segment-cmd 'ffmpeg -i - -c:v libx265 -crf 16 {out}'
```

- The command reads the segment on its standard input, lossless and tagged, in the master's
  format: `yuv420p10le` by default, so your encoder converts nothing.
- It writes the file seedvr2x names, `{out}`.
- seedvr2x never reads into your command, and only checks the result's frame count.
- Disk use is then the compressed size, and a stop keeps every finished segment.
- The frames in those files are your encoder's: they get no checksum from seedvr2x
  ([integrity](#integrity)).

This is why output segments last 5 seconds at least: an encoder pays for each short segment
with a keyframe it can't make up for.

## Disk use

**The output.** Per hour of 1080p at 24000/1001 frames per second, and four times that at 4K:

| Format | Per hour of 1080p |
|---|---|
| `yuv420p10le`, the default | about 100–135 GiB |
| `gbrp16le`, with colour correction (measured with `lab`, the earlier one) | 540–690 GiB |
| with `--segment-cmd` | what your encoder makes of it |

The 16-bit RGB master is that big because the corrected output fills its 16 bits: tens of
thousands of distinct values in a frame.

**During a run,** beside the output:

- **The input copies** the colour correction reads: about 32 MiB per second of 1080p source,
  for the shots of the segment being made, removed when it is finished
  ([the input copies](#the-input-copies)).
- **The saved units:** each shot's latent, removed once its windows are done, and each
  window's output, removed once the shot's segment is finished.
- **A tiled shot's scratch** *(not built yet)*: 12.7 MB per 1080p frame and 50.5 MB per 4K
  frame, in system memory up to an allowance, then in a file, for the one shot being encoded
  or decoded.

**Once:** 17 GB of model files in Hugging Face's cache ([README](README.md#model-files)).

## GPU memory

*Not built yet: the planner, BlockSwap and tiling. Today a shot runs in one window unless
`--window` caps them, nothing leaves the GPU and nothing is tiled.*

### What nvidia-smi shows

nvidia-smi can show the card nearly full while the job needs much less. PyTorch's allocator
keeps the memory it has freed reserved for itself, and trims it only when an allocation would
fail. The figures to read are the peaks seedvr2x logs for each shot, and those `--plan`
predicts. The allocator is `cudaMallocAsync`, which trims its pool on a full card where
PyTorch's native one fragments and fails: seedvr2x sets it through `PYTORCH_CUDA_ALLOC_CONF`
unless you have set that variable.

### Which phase sets the peak

A shot goes through three phases, and each has its own peak
([vram.md](../research/docs/vram.md)):

- **The DiT's peak follows the window's length:** 16 GiB of weights, plus about 1 GiB per
  latent at 1080p and four times that per latent at 4K. This is where a card's memory turns
  into quality: a longer window means fewer joins in a long shot.
- **The VAE's peak follows the frame's size, not the shot's length,** since it works through
  the frames 4 at a time: about 19 GiB to encode and 35 GiB to decode at 1080p, and four times
  as much at 4K, about 134 GiB for the decode. No card decodes a 4K frame whole, which is why 4K
  needs tiles: about 18 GiB with tiles of 1024 pixels.

### The plan

seedvr2x makes one plan per job, after the first pass and before the models load.

- **From the memory free at the start,** as the driver reports it, less a margin of 0.6 GiB.
  What counts is what is free, not the card's size: a card that drives a desktop is judged by
  what it has left, and a 96 GB card half taken by another program is a 48 GB one to seedvr2x.
  The plan is made once: what another program takes later isn't planned for.
- **The window first,** the biggest lever on quality: the longest shot whole when it fits,
  else the longest window that fits, with as many blocks swapped as it takes and the VAE off
  the GPU meanwhile. A card swaps blocks before it shortens a window.
- **Then the fewest swapped blocks** that hold that window.
- **Then the largest tiles,** for the decode, then for the encode, and none where the whole
  frame fits. The DiT leaves the GPU for the VAE's phases before a tile shrinks. Tiles are
  squares of 512 pixels or more, in steps of 64, overlapping by 64. A frame takes the same
  time whatever the tile.

The windows this gives at the two usual sizes, in latents (and frames), by the memory model.
They are estimates until measured on seedvr2x itself:

| Card | 1080p, no block swapped | 1080p, all 36 swapped | 4K, no block swapped | 4K, all 36 swapped |
|---|---|---|---|---|
| 16 GB | the weights don't fit | about 11 (41) | | 2: refused |
| 24 GB | 5 (17) | 19 (73) | | 4 by the fit, under the 5 a job takes: at the limit, see below |
| 32 GB | 13 (49) | 27 (105) | 3: swaps instead | 6 (21) |
| 48 GB | 29 (113) | | 7 (25) | 10 (37) |
| 96 GB | 76 (301) | | 19 (73) | 22 (85) |

- **Nothing caps a large card:** the memory a small card spends on swapped blocks and tiles
  goes, on a large one, to longer windows.
- **A size that doesn't fit is refused, never degraded.** No window cap goes under 5 latents:
  a window in the middle of a shot shares 2 with each neighbour and keeps at least one of its
  own. When a cap of 5 doesn't fit with every block swapped, the job is refused before it
  starts. The message gives the free memory that size takes, what the driver reports free,
  and the largest output size that fits. At 4K that is 22 to 23 GiB free: a 16 GB card never
  has it, and a 24 GB card has about that with nothing else on it.

### Fixing a choice yourself

`--window N`, `--swap-blocks N`, `--encode-tile PX` and `--decode-tile PX` (0 for no tiles)
each fix one choice, and the planner makes the others around it. What you fix is taken exactly
or refused, never adjusted: the refusal names what fits, the longest window, the fewest blocks
or the largest tile. `--window N` is a cap, of 5 or more: no window is longer than N latents,
and a shorter shot is still one window of its own length.

Of these, `--window` exists today.

### A plan shapes the output a little

Its window lengths set where a long shot's joins fall, and its tiles where the picture is cut.
A larger card gives fewer joins and larger tiles.

- Windows change the output about as much as a change of seed does: 41.9 dB from the
  one-window run on one clip, where two seeds are 37–44 dB apart
  ([models.md](../research/docs/models.md)).
- Tiles drift in colour, each as a whole, and more as they get smaller. The colour correction
  removes most of it ([colour correction](#what-seedvr2x-does-split)).

So the plan is part of the job: its window cap and its tiles are recorded, and a resumed job
keeps them ([resume](#resume)). Two runs with the same settings and the same plan give the
same output, bit for bit; swapping blocks changes no bit.

### What `--plan` prints

`--plan FILE` runs the first pass, makes the plan and stops. It fetches, checks and loads the
model files as a run does, to time a short calibration. It prints:

- the card, its free memory and what the plan may use;
- the padded size of the frames;
- the shots and the longest one, the window cap, how many shots run in one window, the joins;
- the swapped blocks and the system memory they pin;
- the tiles and their grid, and the scratch of the longest tiled shot;
- each phase's predicted peak, and which one bounds the plan;
- the time per phase, the total, and the longest unit, which is how early `--until` may stop;
- what you fixed.

It also writes the cut list to FILE ([the cut list](#the-cut-list)), which is the part built
today.

## Small cards

*Not built yet, as the planner.*

seedvr2x is developed on a 96 GB card and made for cards of 16 to 32 GB too. There, the plan
decides whether a job runs at all, and it asks more of the rest of the machine.

**What a small card asks of the host:**

- **Pinned system memory for the swapped blocks:** 0.42 GiB per block, 15.19 GiB with all 36.
  A swapped block is copied to the GPU from pinned memory, which the system can't page out:
  57 GB/s, against 18–25 from ordinary memory. Nothing is copied back, since the weights never
  change.
- **A tiled shot's scratch,** up to its allowance in system memory: 12.7 MB per 1080p frame,
  50.5 MB per 4K frame.
- **The loading of the weights:** the process peaks at 16.5 GiB of system memory while they
  load, and stays flat afterwards.

A job these don't fit, or whose copies the host won't pin, is refused before the run, naming
the memory it needs.

**What a small card can't run:** a size whose smallest window doesn't fit, counted from what
is free at the start, not from the card's size. 4K takes 22 to 23 GiB free.

**What to expect:**

- A 16 GB card can't hold the model's 15.35 GiB of weights: it swaps every block.
- Below 48 GB, a 1080p decode only fits tiled; at 4K every card tiles it, the 96 GB one
  included.
- These cards also compute more slowly: an RTX 5080 has 84 compute units (SMs) where the
  96 GB card has 188. A job lasts longer there, and [resume](#resume) and
  [`--until`](#stopping-at-a-set-time) matter most.
- A later phase brings smaller model files, which lighten the swapping
  ([README](README.md#which-file-to-use)).

## Why zscale is required

ffmpeg does every colour conversion, in and out, and seedvr2x gives it every parameter: the
matrix, the range, the chroma's position and filter. It does them with zscale, ffmpeg's filter
over the zimg library, and refuses to run without it.

- **ffmpeg's default converter is inexact,** even on the bit depth alone. zscale expands 8-bit
  values to 16 bits exactly, white at 65535. swscale, its accuracy flags included, puts white
  at 65283, 0.4% low. An error on every frame the model sees breaks the rule that the upscale
  looks like its source.
- **So swscale is no fallback:** without zscale, seedvr2x refuses to run.
- **Getting a build that has it.** zscale comes with libzimg (ffmpeg configured with
  `--enable-libzimg`). The builds of [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds)
  have it, and so does Ubuntu's ffmpeg package from 25.04 on; 24.04's ffmpeg is 6.1, too old.
  `ffmpeg -hide_banner -filters | grep zscale` tells whether yours does.
- **Why 7.1.** The masters' tags need an option of ffmpeg's `setparams` filter that came with
  7.1 (`chroma_location`). seedvr2x checks for the option itself, and for every filter,
  encoder, decoder and muxer it uses, at each start: a build lacking one is refused at once,
  saying which, rather than hours in.
- **The same bytes on every machine.** By default ffmpeg splits zscale's work in slices, one
  per CPU, and a master then changes with the machine: up to 1.7% of its colour samples, on the
  rows at the slices' edges. seedvr2x runs every zscale on one slice, for 2.3–2.6 ms per 1080p
  frame
  ([numerics.md](../research/docs/numerics.md#the-masters-chroma-420-kernels-and-zscales-slices)).

`seedvr2x verify` needs only ffmpeg and ffprobe: it decodes the frames as they are stored.

## Refused sources

Version 1 is for SDR, progressive sources at a constant frame rate, and it reads what a file
says about itself strictly. A source it can't read right is refused with a message, before
any GPU work for what the file declares, and after the first pass for what only its frames
show. Interlacing, HDR, contradicted colour tags, a rotation and a crop are refused in one
message, numbered when there are several; a pixel format, a matrix or an RGB range comes in a
message of its own.

| The source | Why it is refused | What to do |
|---|---|---|
| **HDR:** a PQ (`smpte2084`) or HLG (`arib-std-b67`) transfer | The model was trained on SDR video; what it makes of HDR-coded pixels is unmeasured, and HDR10's metadata isn't carried. HDR comes in a later version. | Tone-map the source to SDR first, with your own tools: a grading choice, as a gamut conversion is. |
| **Dolby Vision** whose base layer isn't SDR, profile 5's included | The same. A base layer that is SDR is read, as a player without Dolby Vision shows it: the output has no Dolby Vision. | The same. |
| **Dolby Vision metadata on the frames, with no record saying what the base layer is** | A base layer is never guessed. | Read the file it came from, if that has the record (ffmpeg keeps it when it remuxes to Matroska). Or, if you know the base layer is SDR, drop the metadata: `-c copy -bsf:v dovi_rpu=strip=1`. |
| **Interlaced,** declared by the stream | Deinterlacing is a workflow of its own, with specialised tools and many ways to do it. Refused in every version. | Deinterlace it first, with a tool made for it. |
| **A variable frame rate:** frames whose durations differ by more than 1 ms | The output is written at one exact rate, frame for frame. | Make it constant first, with your own tools. If the frames follow one rate and the file is merely wrong about it, see [below](#a-frame-rate-its-timestamps-contradict). |
| **Timestamps that jump:** a step back of more than 0.1 s, or forward of more than 10 s | A discontinuity, as in joined MPEG-TS recordings. seedvr2x keeps the source's timestamps, which its frame index needs, where ffmpeg would smooth the jump over. | For the formats ffmpeg corrects (MPEG-TS and MPEG-PS among them), a remux takes it out, every frame kept: the message gives the command, `ffmpeg -i SOURCE -map 0 -map -0:d -c copy fixed.mkv`. |
| **Colour tags that contradict each other:** the container says one thing, the first frame another, with different meanings | A bad file: either reading could be wrong, and seedvr2x doesn't guess. Two names of one meaning (`bt470bg` and `smpte170m`, one matrix) are no contradiction. | If the frames are right, correct the container's tags with a remux, or `mkvpropedit` on Matroska. If the container is right, correct the bitstream's: the message gives ffmpeg's command when the codec has a filter for it (H.264, HEVC, AV1, VP9, MPEG-2, ProRes, APV). For a YUV source's matrix, `--input-matrix` settles it. |
| **A rotation or a flip** in the display matrix, as phone and camera files have | Not read in version 1; a later version may process the frames as stored and carry the rotation over. | Apply the rotation with your own tools first. |
| **Cropping declared by the container** (MP4 `clap`, Matroska `PixelCrop`) | Not read in version 1. | Apply the crop with your own tools first. |
| **A pixel format outside the tested list:** an alpha channel, palette, grey, XYZ, Bayer, float, 5-6-5 RGB | The conversions are tested format by format. Dropping an alpha channel would change the picture wherever it isn't opaque. | Convert it to a format of the list below first. |
| **A matrix other than BT.709, BT.601 or BT.2020 non-constant luminance** | The others need a conversion seedvr2x never makes, or are too rare to be tested. | Convert it with your own tools. |
| **RGB tagged limited range** | Rare, and most likely a wrong tag: either reading could be wrong. | Correct the tag, or convert the file. |
| **A directory** | The source is the only input ([the two workflows](#the-two-workflows)). | Give the file the segments came from, or join them back into one. |

The pixel formats read: planar YUV 4:2:0, 4:2:2 and 4:4:4 at 8 to 16 bits; 4:1:1, 4:1:0 and
4:4:0 at 8 bits; the full-range `yuvj` 4:2:0, 4:2:2 and 4:4:4; planar RGB at 8 to 16 bits; and
the usual packed or semi-planar formats (`nv12`, `p010le`, `yuyv422`, `rgb24`, `rgb48le` and
their kin), repacked exactly.

A file with no video stream, or whose codec this ffmpeg build doesn't decode, is refused too.

Two limits to know:

- **HDR is looked for at the start only:** in the stream's tags and on the first decoded
  frame. A stream that turns to HDR later passes.
- **A file with several video streams** is read by its first.

### A frame rate its timestamps contradict

Some files declare one rate while their frames follow another. The case measured: a Blu-ray
remux that declares 24 frames per second, whose frames step at 24 with one in 500 held half as
long again, which keeps them on the timeline of 24000/1001, its audio's rate
([seeking.md](../research/docs/seeking.md#failure-mechanisms), mechanism 7). Read at 24, the
upscale would end
0.1% short of its audio, 3.6 seconds an hour.

seedvr2x doesn't reinterpret such a file on its own: it is a bad file, and it is refused like
any other. But the message does the work. When the timestamps follow another rate, it names
that rate, how far the frames stray from it, and two fixes:

- **Remake the file** at that rate: a lossless FFV1 master with the other streams copied, a
  clean file for every tool. The message gives the ffmpeg command, which keeps every frame,
  none dropped or doubled. It costs a full encode and the disk space of a lossless file: about
  80 GB for that film, whose source took 14.5 GB.
- **Or give `--frame-rate R`**: the frames are taken at R, without the intermediate file. It
  is accepted only if every frame lies within half a frame of R's timeline, so that none is
  dropped or doubled, and R becomes one of the job's settings. Write R exactly, as a fraction
  or a whole number: `24000/1001`, not 23.976.

*Not built yet:* a file is also refused when its frames keep a constant duration but lie half
a frame or more from their place on the declared rate's timeline, as 48 frames per second
declared 50 would. When the timestamps follow no rate exactly, as in a file joined by ffmpeg's
concat demuxer, `--frame-rate` can't hold: the message says how far the frames wander and
from which frame, and the ways out are a join that keeps the frame grid, or the remake at the
declared rate. Today both are a warning, which says how many frames stray, from which and how
far, and the file is read at the rate it declares.

### Missing frames

*Not built yet: today a missing frame makes its neighbour last twice as long, and the file is
refused as a variable frame rate.*

Some files have frames on a regular grid with places left empty: a decoder that drops the
pictures before a keyframe, an encoder that dropped repeated frames. Those have one reading,
the one a player shows: the frame before the hole, held. seedvr2x repeats that frame in each
empty place, says so (the frame, the hole's length, how many holes), and records it. The
output keeps the source's length and stays with its audio.

### What is read with a warning

- **Combed frames in a source that doesn't declare interlacing,** as hard-telecined sources
  often are. ffmpeg's `idet` looks at every frame during the first pass, and seedvr2x warns
  when it finds 10% of them or more combed. The run goes on: the frames are read as they are,
  combing and all. For a clean upscale, inverse-telecine or deinterlace the source first.
  Measured: at most 2.1% of frames on 35 real progressive sources, against 88% on a telecined
  DVD.
- **Missing tags,** guessed:
  - the matrix: BT.709 when the width is 1280 or more or the height above 576, else BT.601.
    That is the rule the player mpv applies. `--input-matrix` sets it.
  - the range: limited for YUV;
  - the chroma's position: left, the default of H.264, HEVC and MPEG-2;
  - no sample aspect: square pixels, said in the log without a warning. `--input-sar` sets
    it.
- **Frames ffmpeg decodes with an error.** The run goes on. Such a frame doesn't decode the
  same way twice, so the output of a damaged source isn't guaranteed bit-identical from one
  run to the next.

## Colour and shape of the output

One rule decides all of it: the upscale must look like its source, in any player.

- **Matrix.** The default master is BT.709 at HD sizes and above, whatever the source's.
  Players read the tag, or assume BT.709 at HD sizes, so this is what shows the same colours
  everywhere. Below HD (a width under 1280 and a height of 576 or less), it is BT.601, tagged
  as the source tags it when the source says `bt470bg` or `smpte170m`, else `smpte170m`.
- **Range and chroma.** Limited range, white at 940 and black at 64 in 10 bits, the chroma
  sited left: what `sptenc master` writes too.
- **Primaries and transfer.** Copied as the source declares them, never converted; untagged
  stays untagged. Converting the gamut, an SD source's primaries to BT.709's, changes the RGB
  values themselves, and most players ignore the primaries tag: they would show the upscale
  differently from its source. A gamut conversion is a grading choice, for your own tools.
- **Shape.** The sample aspect the source declares is honoured, and the output has square
  pixels at the source's display aspect: a 720×480 DVD at 16:9 gives 1920×1080. numz's CLI
  ignores the sample aspect, so its upscale of a 16:9 DVD comes out 16% too narrow (NTSC) or
  30% (PAL). `--input-sar` corrects a wrong or missing tag.
- **Size.** `--resolution` is the output's short side, 1080 by default. The model works on
  frames padded to multiples of 16, and the padding is trimmed: the output is the size asked.
  seedvr2x pads with rows mirrored from the picture, then black ones, where numz pads with
  black alone, which costs the bottom 16 rows of its output 3–10 dB
  ([numerics.md](../research/docs/numerics.md#reflectblack16)).
- **PNG** carries the primaries and transfer too, as the chunks ffmpeg's PNG encoder writes
  from the tags, and none for an untagged source.

## Resume

Work is saved in units, each written whole before it is recorded. A stop loses only the unit
in progress.

| Stage | What is saved | What a stop loses |
|---|---|---|
| VAE encode | the shot's latent | the shot's encode |
| DiT | each window's output | the window in progress |
| VAE decode | the output segment | the segment's decode and write, which start again whole |

A segment's shots are all encoded and upscaled before the segment is decoded and written, so
a job's first frames come out later than its first log lines suggest.

**To resume, run the same command again,** with the same `-o` directory. There is no resume
option. Finished segments and saved units are skipped, and the log says what was found:
"resuming film.segments: 3 of 12 segments finished, …". A finished job run again says
"finished already" and does nothing.

**A directory holds one job.** The same command is that job; anything else is another job,
and is refused with the list of what differs, each line giving the recorded value and the
new one. The one exception is a plan, which takes other cuts as long as it has made no unit
([the cut list](#the-cut-list)). What is compared:

- **the settings:** every option that shapes the output, the model files by their SHA-256,
  and seedvr2x's own code. Any change to the code refuses a resume: update seedvr2x between
  jobs, not during one.
- **the source,** by its content: its size and SHA-256. A source that was moved or re-dated
  is the same source.
- **the environment:** the GPU, PyTorch, CUDA and cuDNN, the attention backend, ffmpeg and a
  fingerprint of its conversions, Python and its packages' versions. The NVIDIA driver is
  recorded but not compared.

It is that strict because a resume promises the uninterrupted run's output, bit for bit, and
each of these can change bits.

**`--accept-env-change`.** When the environment is all that differs, the refusal says so and
names this option. Another GPU, another PyTorch or another ffmpeg only breaks the bit-identity
across the resume, not the upscale's correctness, and refusing outright would throw away tens
of GPU hours over an upgrade. The change is written in the job's record: what changed, when,
and how much was made before it.

**What a resume doesn't do again:**

- **The first pass.** Its record is trusted when the source, the settings and ffmpeg are
  unchanged, which saves a full decode per resume: about 17 minutes on a 2-hour HEVC master.
  The job's cuts stay its own.
- **Decoding from the start.** A resumed job reads the source from its first unfinished shot,
  through the frame index the first pass built, and checks every frame it reads against that
  index, by its timestamp and its checksum. A read takes 0.15–0.27 s on Blu-ray and web H.264,
  1.24 s on an HEVC master with long gaps between keyframes, 16 s at worst. The checks aren't
  zeal: seeking by time alone missed on 6 real sources of 10, by up to 142 frames
  ([seeking.md](../research/docs/seeking.md)).

**The directory is the job's.** It must hold what the job's record names and nothing else: a
file from anywhere else is refused, never deleted. One seedvr2x at a time writes a directory.
What a stopped run left half-written is discarded.

**The plan across a resume** *(not built yet)*. A resumed job keeps what shapes its output,
the window cap and the tiles, and plans the rest again from the memory free that day: the
swapped blocks, what stays on the GPU. If what it recorded no longer fits, it stops and names
the phase and the shortfall.

**It was checked the hard way:** a job stopped by a kill during an encode, during a window
and during a segment's decode, by Ctrl-C once and twice, and resumed after each. Every segment
equalled the uninterrupted run's, bit for bit. This holds for a source that decodes the same
each time; a damaged one may not.

**One file can't resume today:** `-o x.mkv` keeps nothing of a stopped run, until assembly is
built. Use a directory for anything long.

### Stopping

- **Ctrl-C, once,** finishes the unit in progress, then stops. seedvr2x says which: "stopping
  after shot 3/62's window 1/2; Ctrl-C again stops at once". The longest unit is a segment's
  decode.
- **Ctrl-C twice, or SIGTERM,** stops at once. Only the unit in progress is lost.
- Either way the GPU is freed entirely, and the same command resumes.
- **A value that isn't a number** (NaN, infinity) coming out of the model stops the run, with
  a message naming the stage, the shot and the window or frames. Such a unit is never saved,
  and never written to a master.

## Stopping at a set time

*Not built yet.*

`--until HH:MM` stops the run cleanly before a unit that wouldn't finish in time, to give the
computer back in the morning.

```
uv run seedvr2x film.mkv -o film.segments --until 07:30
```

- **The time** is the next time the local clock shows HH:MM. The log gives it as a full date.
- **What it stops before:** before each unit, the run stops if now plus 1.15 times the unit's
  estimated time passes it. Units are never reordered to fill the time left.
- **How early it may stop** is set by the longest unit, a segment's decode. `--plan` prints
  it.
- **Its exit code is 75,** so that a script tells a job stopped by the clock, to be run again,
  from a finished one (0) and from a Ctrl-C (130):

  ```sh
  uv run seedvr2x film.mkv -o film.segments --until 07:30
  case $? in
    0)  echo "finished" ;;
    75) echo "stopped by the clock: the same command goes on tonight" ;;
    *)  echo "stopped or refused: read the log" ;;
  esac
  ```

**The estimates are measured, never constants.** A short calibration at the job's size, 13
frames encoded and decoded and one window of 2 latents, sets them; `--plan` runs it, and so
does a run given `--until`. After every unit they are corrected by the last units' real
times, which follows a card whose speed drifts: on one card the DiT's time per token moved
between 0.23 and 0.45 ms.

## Integrity

A master is the product, and a lossless file can be damaged without anything saying so.

- **FFV1 damage can be silent.** ffmpeg decodes through a slice that fails its own checksum
  and exits with success, the slice hidden under the previous frame's; its only sign is a line
  among its messages. A damaged size field can make it skip slices without a word: a frame
  came back with 2,880 of its 3,072 pixels from another frame.
- **So every frame gets a checksum,** a CRC-32 computed as the frame is written, the only
  cheap moment (about 2 ms per 1080p frame). The checksums are kept with the output and
  outlive the job: `checksums/` in an output directory, a file per segment, and
  `<output>.crc32` beside a one-file output.
- **`seedvr2x verify OUTPUT`** decodes every frame and checks it, on demand: before an encoder
  or an archive trusts the output. OUTPUT is an output directory, one of its segments, or a
  one-file output. Anything ffmpeg reports while decoding fails the check too.
- **It fails on anything short of whole:** a frame that isn't as written, checksums that are
  missing, a segment that is unfinished. So this never hands sptenc an incomplete directory:

  ```
  uv run seedvr2x verify film.segments && sptenc encode film.segments film.1080p.mkv -f film.mkv
  ```

  Its last line says "12 checked, every frame as written", or which were not: "of 12, 1 not as
  written and 1 unfinished".
- **A resume** checks each finished segment's size against the job's record: cheap, and it
  catches a file cut short.
- **`--segment-cmd` outputs** *(not built yet)* only get their frame count checked: their
  frames are your encoder's.

The same care goes to what the job reads: the model files are checked by size and SHA-256
before every run ([README](README.md#model-files)), and the input copies frame by frame
([the input copies](#the-input-copies)).

## Seeds and reproducibility

`--seed` sets the seed of the model's noise, 42 by default.

- **The same settings and the same plan give the same output, bit for bit,** in the same
  environment. The plan is the window cap and, once tiling is built, the tiles: on another
  card, fix them to get the same output ([GPU memory](#fixing-a-choice-yourself)).
- **Each shot has its own noise,** drawn from the seed and the number of the shot's first
  frame in the source. So a shot's output depends only on the seed, its frames and its place:
  a resumed job, or a job run again with other cuts elsewhere, gives the untouched shots
  again, exactly.
- **A long shot's windows share one draw,** made once for the shot: a latent shared by two
  windows gets the same noise in both, which brings the result 1.2–2.0 dB closer to the
  one-window run than a draw per window.
- **The largest seed depends on the source's length:** the seed plus the last shot's first
  frame plus 1,000,000 must stay under 2^32. A seed out of range is refused before the model
  loads, with the bound for that source.
- **What changes the output's bits:** another seed, another plan, another environment (the
  GPU, PyTorch, ffmpeg, the attention backend). **What doesn't:** stopping and resuming, and
  how many blocks are swapped *(not built yet)*.

How much is a seed? Two seeds of the same model give outputs 37–44 dB apart
([models.md](../research/docs/models.md)). That spread is the yardstick every figure of this
manual is read against: a change that moves the output less than another seed does is within
it.

## Options

`uv run seedvr2x --help` is the reference for what the installed version takes.

**Input and output**

| Option | What it does |
|---|---|
| `input` | The source, one video file. |
| `-o`, `--output OUTPUT` | A `.mkv` path: one file. Another video suffix (`.mp4`, `.mov`…) is refused. Anything else: a directory of output segments, new or empty, or an unfinished job's, which the command resumes ([the two workflows](#the-two-workflows)). |
| `--format {yuv420p10le,gbrp16le,png}` | The output's format, `yuv420p10le` by default ([the formats](#the-formats)). |
| `--resolution N` | The output's short side in pixels, 1080 by default. |
| `--min-segment SECONDS` | Output segments last this long at least, 5 by default; 0 keeps a segment per shot. |
| `--segment-cmd CMD` | *Not built yet.* Your encoder's command, run once per output segment ([`--segment-cmd`](#--segment-cmd-your-encoder-during-the-run)). |

**Shots**

| Option | What it does |
|---|---|
| `--cut-threshold P` | The shot detector's threshold, above 0 and at most 1; 0.3 by default. Refused with `--cuts`. |
| `--cuts FILE` | A cut list, in place of the detector ([the cut list](#the-cut-list)). |
| `--plan FILE` | Plan the job and stop before the upscale: the cut list written to FILE, and the plan printed ([what `--plan` prints](#what---plan-prints); the plan itself *not built yet*). |

**The model**

| Option | What it does |
|---|---|
| `--dit-model FILE` | The DiT's file: `seedvr2x_ema_7b_sharp_fp16.safetensors` by default, or `seedvr2x_ema_7b_fp16.safetensors` ([the model](#the-model)). |
| `--vae-model FILE` | The VAE's file, `seedvr2x_ema_vae_fp16.safetensors` by default. |
| `--model-dir DIR` | Read the model files in DIR, by name, and download nothing ([README](README.md#model-files)). |
| `--seed SEED` | The seed, 42 by default ([seeds](#seeds-and-reproducibility)). |
| `--color-correction {split,none}` | `split` by default ([colour correction](#colour-correction)). |

**What the source says of itself, when it is missing or wrong**

| Option | What it does |
|---|---|
| `--input-matrix {bt709,bt470bg,smpte170m,bt2020nc}` | The matrix of a YUV source. |
| `--input-sar N:D` | The source's sample aspect ratio. |
| `--frame-rate N/D` | The source's frame rate, when its timestamps contradict the one it declares ([a frame rate its timestamps contradict](#a-frame-rate-its-timestamps-contradict)). |

**Memory**

| Option | What it does |
|---|---|
| `--window LATENTS` | A cap on the windows, in latents, 5 at least ([GPU memory](#fixing-a-choice-yourself)). |
| `--swap-blocks N` | *Not built yet.* The number of DiT blocks swapped. |
| `--encode-tile PX`, `--decode-tile PX` | *Not built yet.* The VAE's tiles, 0 for none. |

**Running**

| Option | What it does |
|---|---|
| `--until HH:MM` | *Not built yet.* Stop cleanly before that time ([stopping at a set time](#stopping-at-a-set-time)). |
| `--accept-env-change` | Resume a job whose environment changed ([resume](#resume)). |
| `--dump-frames DIR` | Also save every output frame as a float32 `frame_NNNNNN.npy` file, for measurements. |
| `-v`, `--verbose` | Debug logging. |
| `--version`, `-h`, `--help` | The version; the help. |

`uv run seedvr2x verify OUTPUT` checks an output ([integrity](#integrity)).

seedvr2x runs on the first CUDA device its process sees. On a machine with several cards,
CUDA's own `CUDA_VISIBLE_DEVICES` chooses which, and separate jobs run on separate cards.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | The job is finished; also a job that was finished already, and a plan written by `--plan`. |
| 1 | Refused or failed: the last log line says why. For `verify`: a frame not as written, checksums missing, or a segment unfinished. |
| 2 | An unknown option, a missing argument, or a value of the wrong form. |
| 75 | Stopped by `--until`, to be run again *(not built yet)*. |
| 130 | Stopped by Ctrl-C. |
| 143 | Stopped by SIGTERM. |

## The files of a job

An output directory:

```
film.segments/
  seg_000000.mkv           the output segments, in order
  seg_000001.mkv
  checksums/
    seg_000000.crc32       a CRC-32 per frame; kept with the output
    seg_000001.crc32
  manifest.json            the job's record: settings, environment, source, shots, segments
  frame_index.bin          the first pass's record: each frame's timestamp, checksum and
                           probability of a cut
  resume/                  saved work, removed as the segments finish
    shot_000123/           a shot, named by its first frame
      latent.pt
      window_0000.pt
      input.mkv            the input copy, and its checksums
      input.crc32
```

- With `--format png`, each segment is a directory, `seg_000000/`, of frames numbered from
  `000000.png`.
- Every file is written under a `.partial` name and renamed when whole.
- When the job is finished, the segments, `checksums/`, `manifest.json` and `frame_index.bin`
  remain. sptenc reads the `.mkv` files alone.

A one-file output, `film.mkv`, has `film.mkv.crc32` beside it, and during a run with the
colour correction on, a work directory for the input copies, `film.mkv.work`, removed when
the run ends.

## Environment variables

| Variable | Effect |
|---|---|
| `HF_HOME`, `HF_HUB_CACHE` | Where Hugging Face's cache is, `~/.cache/huggingface` by default: the model files go there. |
| `HF_HUB_OFFLINE=1` | Keeps the network out: the model files are taken from the cache. |
| `HF_HUB_DISABLE_TELEMETRY` | Set to `1` by seedvr2x unless you have set it. |
| `PYTORCH_CUDA_ALLOC_CONF` | Set to `backend:cudaMallocAsync` by seedvr2x unless you have set it ([GPU memory](#what-nvidia-smi-shows)). |
| `CUDA_VISIBLE_DEVICES` | CUDA's own: which card the job sees. |

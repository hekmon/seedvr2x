# seedvr2x's ComfyUI nodes

[ComfyUI](https://github.com/Comfy-Org/ComfyUI) nodes that run
[SeedVR2](https://github.com/ByteDance-Seed/SeedVR) with seedvr2x's own model files, loaders and
memory planner: a picture, a batch of pictures or the frames of a short clip upscaled shot by shot
on the card you have, with the plan said in ComfyUI's console and a refusal, where one is due,
before any model loads.

> **Not published yet.** The nodes are built and tested against ComfyUI v0.44.0 and are installed by
> hand, as below; they are published with version 0.2.0 ([DESIGN.md](../../DESIGN.md)). Whole
> films, resume and 16-bit output are [the command line's](../seedvr2x/README.md): the
> `apps/seedvr2x` folder of this repository, beside these nodes.

## The nodes

- **seedvr2x Models** chooses the model file by name, with its size and its error per weight, as
  the [model card](../../models/hf/README.md) gives it (how far its stored weights are from
  ByteDance's original, the median over the 7B's 288 large matrices): `Sharp 7B, float16
  (16.5 GB, 0.02%)` (the default), `Sharp 7B, int8 (8.3 GB, 0.86%)`, `Sharp 7B, dynamic GGUF
  (4.8 GB, 7.55%)`, and the regular 7B in each. It downloads what Hugging Face's cache lacks of the
  model file and the VAE when it runs, once ([Model files](#model-files)), and says on the node
  the file chosen and whether it is kept in RAM. It also chooses whether the model stays in RAM
  between runs: `auto`, `on` or `off` ([Memory](#memory)). Its optional input `model folder`,
  unconnected by default, reads the files from a folder you give instead of the cache. Nothing is
  loaded here: the first upscale run loads.
- **seedvr2x Cuts** finds where a clip changes scene ([Cuts and shots](#cuts-and-shots)): it takes
  the clip's frames, and gives the cut list that Upscale Video needs; the cuts are said in
  ComfyUI's console.
- **seedvr2x Upscale Image** upscales each picture of a batch on its own, to the output's short
  side given by `resolution` (1080 by default).
- **seedvr2x Upscale Video** upscales a clip's frames shot by shot, each shot in sliding windows
  of the model, no window crossing a cut.
- **seedvr2x Tile**, **seedvr2x Blocks Swapped** and **seedvr2x Window Length**, in the node
  library's `seedvr2x/plan settings`, set one of the plan's numbers yourself
  ([Setting a value yourself](#setting-a-value-yourself)). The other three are in `seedvr2x`.

Each upscale node takes `images` first (Upscale Video then takes `cuts`), then `models`, and gives
`images`. Its settings are widgets, each at its default:

| Widget | Default | Notes |
| --- | --- | --- |
| resolution | `1080` | The output's short side, in pixels. |
| quality (Upscale Video only) | `standard` | How many times the model runs over each shot: `standard` once, `enhanced` twice, `maximum` four times, each run starting a frame or two further into the shot, each frame kept from the run that rebuilds it best. Sharper lines and less flicker where there is motion, little change on a still shot; 2 or 4 times as long as `standard`, and no more memory. A picture is one frame, so Upscale Image has no quality. |
| attention | `auto` | FlashAttention 2 where ComfyUI's environment has it, else PyTorch's SDPA. |
| colour correction | on (`split`) | On, `split`, keeps the input's overall colours and brightness with the model's fine detail; off, `none`, leaves the model's own colours, which can drift. |
| seed | `42` | ComfyUI's seed with its control after generate, `fixed`. |
| plan only | off | Make the plan, say it in the console, and stop: no model is loaded and the nodes after the images output do not run. |

The numbers the plan makes are optional inputs: [Setting a value yourself](#setting-a-value-yourself).

Each upscale node makes its plan at run time, where the input's size and the card are known, and
says it whole in ComfyUI's console; the node itself shows two short lines, where the run is and
its time ([While it runs](#while-it-runs)). The nodes have no text output: the plan and the cuts are read in
the console. The plan is made for the card as it would be without ComfyUI's own models.

## Cuts and shots

The model works on a stretch of frames at once, in windows that overlap, and a window that holds a
cut blends the two scenes. So Upscale Video takes the clip's **cut list**, a required input that
only seedvr2x Cuts gives, and upscales each shot of it on its own: no window crosses a cut. The
list is a fact of the clip (the same images go to Cuts and to Upscale Video), not a setting, and the
plan is made once for the clip, the longest shot's windows deciding it. Each shot is seeded as the
command line seeds a shot: the `seed` plus the shot's first frame, counted from 0. The plan names
the shots one by one: where each is, and its windows.

seedvr2x Cuts has two modes:

- `detect` (the default) runs seedvr2x's shot detector, TransNetV2 (a file of about 30 MB,
  downloaded once from Hugging Face into its cache, like the model files), on the clip's frames,
  shrunk to its small input in torch, with no ffmpeg. It runs on the card ComfyUI uses while the node
  runs, and on the processor where there is none, and frees its memory after. `threshold` is how
  sure the detector must be that a frame starts a new shot: 0.3 by default, the command line's.
  A cut found for nothing only gives a shot its own windows, so the default errs toward cutting.
  Its optional input `model folder` is where to read the detector's file from instead of the
  cache, as in seedvr2x Models.
- `one shot` runs no detector: the whole clip is one shot, for a clip you know has no cut.

The console lists each cut by the frame its shot starts at, counted from 0, with the detector's
probability, and the **possible cuts**: scenes it found less sure than the threshold, which are not
in the list. Look at them in the clip; a threshold under a possible cut's probability makes it a
cut. The node's own line says how many: `2 shots · 1 possible cut below 0.30`.

## Setting a value yourself

The numbers the plan makes are optional inputs of the upscale nodes, each a socket of its own
type: `encode tile` and `decode tile`, `blocks swapped`, and Upscale Video's `window length`.
Unconnected, the plan chooses, and the plan in ComfyUI's console says what it chose
(`encode tile: auto -> 1024 px`).
To set one, drag a link out of the socket and release it on the canvas: ComfyUI offers the one
node that gives that type, first in its search box, where Enter adds it already connected, and
among the suggestions of its link menu, which a release with Shift opens instead (ComfyUI's
setting `Comfy.LinkRelease.Action` swaps the two). The node library lists them under
`seedvr2x/plan settings`. Each value node carries its range, its step and a description:

| Node | Widgets | Gives |
| --- | --- | --- |
| seedvr2x Tile | tile size (px): 512 to 2048 in steps of 64, 1024 by default; whole image at once: off by default | a tile for encode tile or decode tile, or no tiles where whole image at once is on |
| seedvr2x Blocks Swapped | blocks swapped: 0 to 36, 0 by default | the model's blocks that wait in RAM and pass through the card in turn |
| seedvr2x Window Length | window length (frames): 17 to 509 in steps of 4, 41 by default | the longest stretch of a clip the model works on at once |

A value connected is taken exactly and the plan is built around it, or the run is refused before
anything loads. Each widget keeps ComfyUI's input socket, so an `Int` node, or any node that gives
an integer, can drive the value through its value node; a number off the node's grid is then
refused when the value node runs.

## While it runs

The node shows how far it has got, in what ComfyUI gives a node:

- **The bar** is the time the upscale should take, not a count of steps: every tile of the encode,
  every window of the model and every tile of the decode, of every shot or picture, each by its
  size. It starts from the speed seedvr2x measured on an RTX 5090, and as soon as the node has made
  a unit of one kind (an encode tile, a window, a decode tile) it uses your computer's own speed
  for that kind and spreads the rest of the bar over what is left; it never goes back.
- **The text on the node** is two short lines at most, a progress hint and never a log
  (ComfyUI's box holds two lines, and a longer text has to be scrolled). The first says where the
  run is and what it does, whole: widen the node to read a long one. The second is the time, at
  most 40 characters: the time so far until the first window is done, then the time left. The
  text moves when a tile, a window or a few frames are done, and its time each second between
  them, so that a long window's text does not stand still:

  | State | First line | Second line |
  | --- | --- | --- |
  | the node starts | `planning…` | |
  | downloading, reading the files | `downloading 3.1/8.3 GB`, `reading 2.0/8.8 GB` | the time left at the bytes' speed |
  | a clip, encoding | `shot 2/4 · encoding`, `shot 2/4 · tile 3/6 · encoding` | `6 s elapsed · estimating time left` |
  | a clip, the model | `shot 2/4 · window 2/3 · upscaling` (one window: `shot 2/4 · upscaling`) | `~38 min left` |
  | a clip, decoding | `shot 2/4 · frames 60/118 · decoding`, `shot 2/4 · tile 2/4 · decoding` | `~12 min left` |
  | enhanced or maximum | `run 3/4 · window 2/3 · upscaling`, and with several shots `shot 2/4 · run 3/4 · window 2/3 · upscaling` | `~3 min left` |
  | a picture of a batch | `picture 3/8 · tile 2/4 · encoding`, `picture 3/8 · upscaling`, `picture 3/8 · tile 1/4 · decoding` | `~45 s left` |
  | the GPU does not answer | the line it was on | `GPU not answering for 3 min` |
  | plan only | `plan: window 41 frames · decode 1088 px · encode 1024 px`, `plan: encode 1024 px · decode 1088 px` | `whole plan in the console` |
  | finished | `480 frames in 4 min 32 s`, `8 pictures in 41 s` | |
  | seedvr2x Cuts | `scanning 64/120 frames`, `2 shots · 1 possible cut below 0.30`, `one shot: 120 frames` | |

  A clip of one shot has no shot part, and a run part is there only at enhanced and maximum. The
  time is rough: the time so far in seconds under a minute, then in whole minutes (whole hours
  past 1,000 minutes, whole days past 1,000 hours); the time left in seconds under a minute,
  whole minutes under an hour, then hours and minutes. A download's or a reading's time left is
  at the speed its bytes have come since their first.
- **ComfyUI's console** has the rest, each of seedvr2x's lines beginning with `[seedvr2x]`, the
  library's included, so that they read apart from other nodes' lines, which are left as they are:
  the plan whole before a run (with `plan only`, the same), the cuts with their probabilities and
  the possible cuts, each shot's frames, the time a run took ("480 frames upscaled in 4 min
  32 s"), where each model file is, and the full sentence of everything the node's line only
  points at.
- **A refusal** shows `refused: see the error` on the node. Its text is in ComfyUI's error and,
  once, in the console, with the way out in the nodes' terms.
- **A wait that lasts.** Once the node has made one unit of a kind, it knows how long the next
  should take. If the card has not answered for more than four times that (and at least a
  minute), the node's second line says `GPU not answering for 3 min` (the minutes it has waited),
  under the line it was on, and ComfyUI's
  console says it with the card's reading and the way out. The wait is not given up on its own: a
  slow card (shared with a game) answers in the end, and the line goes back to normal. Cancel leaves
  the wait. If the card has stopped answering, though, the work queued on it can't be undone inside
  ComfyUI. A run left that way gives nothing back to the card on its way out, as that would wait on
  it again. At the next run's start the node asks the card, without waiting, whether the work has
  finished: if so it gives the models back and goes on; if not, the run is refused, saying that the
  card is still busy, and you can wait and run again or restart ComfyUI.
- **The card's states** have a line each, and the full sentence in the console:

  | State | Line |
  | --- | --- |
  | the models are given back after a Cancel or an error | `freeing the GPU… Cancel again to leave` (a second Cancel leaves it, run again once the card has finished, or restart ComfyUI) |
  | the card has not finished its last work, after that, or a run refused for it | `GPU busy: run again or restart ComfyUI` |
  | a CUDA error failed the card, now or in an earlier run | `GPU failed: restart ComfyUI` |

- **The preview** is the newest frame decoded, as ComfyUI's samplers show their steps, once for each
  few frames of a clip and once for each picture of a batch, at ComfyUI's preview size
  (`--preview-size`, 512 by default).
- **Before the upscale**, the downloads are seedvr2x Models': a bar of its own, in bytes, and on
  the node the bytes ("downloading 3.1/8.3 GB") over the time left at their speed, then one line,
  the file chosen and whether it is kept ("Sharp 7B int8 · kept in RAM"); ComfyUI keeps the node's output, so this happens once for as
  long as the node's inputs stay as they are. The upscale node's bar starts with the reading of
  the files into memory, in bytes ("reading 2.0/8.8 GB"), and again at 0 for the upscale; if a
  file has gone from the cache since seedvr2x Models ran, the upscale node says so in ComfyUI's
  console and downloads it again on that bar. seedvr2x Cuts shows the frames it has scanned, and
  downloads its detector's file on its own bar.
- **Cancel** stops the run after the tile or the window being made, or the piece of a file being
  read, and frees the card. Hugging Face's fast downloader (Xet) reports from a thread of its own,
  where a Cancel cannot stop it: the node leaves the pressed Cancel as it is and honours it as soon
  as that file has finished downloading.

## Install

Nothing is published, so install by hand. Link or copy this folder, `apps/comfyui`, into ComfyUI's
`custom_nodes` folder, under any name. Then, in ComfyUI's own Python environment:

```
pip install --no-deps /path/to/seedvr2x/core
pip install "diffusers>=0.42" omegaconf rotary-embedding-torch
```

- **Python 3.13** is what the library requires today ([core/pyproject.toml](../../core/pyproject.toml)
  holds its dependencies, with their ranges). torch is ComfyUI's own; so are the numpy, Pillow,
  safetensors, torchvision, einops and huggingface_hub it already has. The library is installed with
  `--no-deps` so that pip leaves them alone.
- **diffusers 0.42 or later**: 0.40 pins huggingface_hub below 2, which ComfyUI's environment may
  have passed. The library's tests pass with the Pillow 13 and huggingface_hub 2 that ComfyUI's
  installs hold; diffusers past the command line's 0.40 is not yet compared with it bit for bit.
- The two example workflows, `workflows/picture.json` and `workflows/clip.json`, load by dragging
  them into ComfyUI. Each has seedvr2x Models connected, every widget at its default and every
  optional input unconnected; the clip's has seedvr2x Cuts between Load Video and Upscale Video.

## Model files

The files are seedvr2x's own conversions, from [hekmon/seedvr2x](https://huggingface.co/hekmon/seedvr2x)
at a revision pinned in the code, each checked by its pinned size and SHA-256 before it is used.
They are no files ComfyUI's loaders take, so they are not in its `models/` folders (its own SeedVR2
files sit there and are not seedvr2x's): they stay in Hugging Face's cache, where one copy serves
every seedvr2x app, the command line and the image server included ([the command line's
README](../seedvr2x/README.md#model-files)).

- **What they take.** The model file chosen: 16.5 GB for float16, 8.3 GB for int8, 4.8 GB for the
  dynamic GGUF; the VAE, 0.5 GB; seedvr2x Cuts' shot detector, 30.5 MB.
- **Where they are.** On Windows `%USERPROFILE%\.cache\huggingface\hub`, on Linux
  `~/.cache/huggingface/hub`; or `hub` under the directory `HF_HOME` names, or the directory
  `HF_HUB_CACHE` names. ComfyUI's console says where each file is when seedvr2x Models runs
  ("Sharp 7B int8, 8.3 GB, in Hugging Face's cache: ...").
- **Downloaded when missing.** seedvr2x Models downloads what the cache lacks of the model file
  and the VAE when it runs, with its bar and its line, and again only if its inputs change or the
  cache loses a file. With `HF_HUB_OFFLINE` set nothing is downloaded and the node says so.
- **Moving the cache.** Set `HF_HOME` before ComfyUI starts, to a folder on a disk with room for
  the files. For ComfyUI Desktop on Windows, in the user environment variables (Settings, System,
  About, Advanced system settings, Environment Variables; or `setx HF_HOME D:\hf` in a terminal),
  then restart ComfyUI. For the portable build, a line `set HF_HOME=D:\hf` before the line that
  starts python in `run_nvidia_gpu.bat`. On Linux, in the shell that starts ComfyUI
  (`HF_HOME=/data/hf python main.py`). Files already in the old cache are not moved: copy the
  `models--hekmon--seedvr2x` folder of its `hub` into the new one, or let the node download them
  again.
- **Using a folder instead.** Connect a string node (ComfyUI's String node, for one) holding the
  folder's path to seedvr2x Models' `model folder` input; quotes around the path (Windows' "Copy as
  path" adds them) and spaces at its ends are taken off, and an empty value is the same as
  nothing connected. The files are read there by name, nothing downloaded: the VAE,
  `seedvr2x_ema_vae_fp16.safetensors`, and the model file chosen, under its name in the repository
  (`seedvr2x_ema_7b_sharp_fp16.safetensors` for the default), the same files and names the command
  line reads from a folder. A path that is no folder, or a folder lacking one of the files, is
  refused by the node, saying which.

**On Windows.** Without Windows' Developer Mode, huggingface_hub can't make the links its cache is
built of: it copies the files instead, and says so once in ComfyUI's console, in a warning that
begins "`huggingface_hub` cache-system uses symlinks by default". Setting the environment variable
`HF_HUB_DISABLE_SYMLINKS_WARNING` silences it. For seedvr2x's files nothing is duplicated: each is
downloaded once, and only a change of the version seedvr2x pins keeps the earlier copies beside
the new ones. Developer Mode turns the links on. A cache written under WSL can't be read from
Windows: its links are WSL's own, and Windows fails on them with "The system cannot access the
file", which the nodes' refusal shows. Give each side its own cache (`HF_HOME`).

## Limits

- **A clip is kept whole in RAM**, as ComfyUI keeps any video: about 100 MB a 4K frame in float32,
  the input's frames and the output's. ComfyUI has the input in memory before the node starts, so
  the node counts the output and its own work against the RAM that is available, and says the input
  beside them (the plan's `RAM` line); a clip that does not fit is refused before any model loads,
  and the refusal says so. For a whole film, use [the command line](../seedvr2x/README.md): it
  reads a film shot by shot and never keeps it whole, picks up again after a stop, and writes
  lossless files (the `apps/seedvr2x` folder of this repository; ComfyUI's console says so once,
  when the nodes load).
- **Nothing resumes.** A run that stops starts again.
- **A card that stopped answering may need ComfyUI restarted.** When a second Cancel leaves the
  wait for the card, the next run asks the card, without waiting, whether it has finished its last
  work: if it has, the nodes give their models back and the run goes on; if not, the run is
  refused ("the graphics card is still busy"), and only a new ComfyUI is certain to have a card
  that answers.
- **A graphics card that failed needs ComfyUI restarted too.** A CUDA error other than running out
  of memory (an illegal memory access, a "CUDA error: out of memory" that a kernel met) leaves the
  card's state untrusted, and under ComfyUI's cudaMallocAsync freeing memory there can end ComfyUI's
  process at once, with no message. So the nodes free nothing on that card: the node shows `GPU
  failed: restart ComfyUI`, ComfyUI's console and error say it with CUDA's own words, and every
  later run is refused at its start until ComfyUI is restarted. Running out of memory (torch's "out
  of memory" error) is not a failure: the card is freed as after any error.
- **ComfyUI's image.** The output is its `IMAGE`, float32, and its savers write 8 bits or lossy
  video. The command line writes 16-bit PNG and lossless masters, with seedvr2x's record in each
  output; the nodes write no file and carry no record.
- **8-bit YUV sources.** From an 8-bit YUV video (most files: H.264, H.265, VP9, AV1) the nodes
  upscale what ComfyUI's own decode gives, 8 bits a channel, where the command line converts the
  video to 16 bits a channel first. The two outputs differ by a fraction of a level: 0.26 of one
  level (of 255) at most on the clip measured. From an RGB source (a lossless RGB
  video) the nodes give the command line's frames bit for bit where both run the same torch,
  its CUDA build and the same attention kernel. Across versions and platforms the frames
  differ, as any two environments' do: ComfyUI's torch 2.12.1 under Windows against the
  command line's 2.14.1 under WSL2, both on SDPA, gave 38.7 and 43.7 dB of PSNR on a clip's
  two shots, each side repeating its own frames bit for bit.

## Memory

- **In RAM between runs.** `keep in RAM between runs` on the Models node is `auto`, `on` or
  `off`. The weights are kept in pinned system memory after a run, up to 15.81 GiB for the float16
  file and the VAE, and the next run does not read them again. `auto` (the default) keeps them
  where they fit under ComfyUI's own cap on pinned memory (its `MAX_PINNED_MEMORY`: 40% of the
  RAM on Windows, up to 90% of it on Linux, none where pinned memory is off); `on` keeps them
  whatever the cap; `off` reads the files for each run and drops them after it. The plan says
  which: `keep in RAM between runs: auto -> kept, 8.23 GiB (ComfyUI allows 12.80 GiB)`, or
  `auto -> not kept: 15.81 GiB is over the 12.80 GiB ComfyUI allows`. Under ComfyUI's
  `--disable-smart-memory`, which unloads every model after each prompt, nothing is kept.
  ComfyUI's `Unload models` frees them, and ComfyUI may drop them itself when its own models need
  the room: the next run then reads the files again, and says so in the console. Choosing another
  file frees the first one's memory before the second is pinned.
- **On the card, only while a node runs.** The plan counts on the card as it would be without
  ComfyUI's models and less the room left to the rest of the card, and ComfyUI is asked to free
  only the room the plan needs, with 0.25 GiB over it for the buffer its partial unloads keep: its
  models stay where both fit. That room is one figure, the larger of ComfyUI's own reserve for the
  system (its `--reserve-vram`, 0.4 to 0.7 GiB by default) and the least seedvr2x leaves on a
  driver that lends system memory (0.5 GiB), not the two added; the plan says which set it, and
  raising `--reserve-vram` raises the room. The card is freed
  when the node ends, on Cancel and on an error too.
- **A driver that lends system memory** (Windows, WSL2) borrows your computer's RAM where the card
  is full, which Windows shows as "shared GPU memory", whatever NVIDIA's "CUDA - Sysmem Fallback
  Policy" is set to. It slows a run, and ComfyUI's allocator decides how well the run is kept to
  the card. Under its default, cudaMallocAsync (on its CUDA 13 builds), nothing slows and no bit
  of a picture moves, but it holds more of the card than it uses, which the plan counts: on an 8 GB
  card a picture fits either way, and a 1080p clip only under torch's own allocator with its
  expandable segments. That is ComfyUI's `--disable-cuda-malloc` launch option together with the
  environment variable `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, set before ComfyUI
  starts (ComfyUI sets no segments of its own; without them torch's own allocator plans with
  4.5 GiB of the card kept free beside the model's work, and a step short of room fails where the
  segments slow it down and run it). On Windows itself, torch has no expandable segments, and the
  driver is taken for one that lends: keep ComfyUI's default, cudaMallocAsync, for which the plan
  leaves each step its room (torch's own allocator there keeps 4.5 GiB of the card free and still
  fails where a step is short). The nodes say which applies once, in the console, where the
  driver lends system memory. The nodes set no limit on torch's memory: ComfyUI's process is
  shared with its other nodes, and a limit set there cannot be taken back.
- **A card too small** is refused before any model loads, with the way out in the nodes' own
  words: a smaller file in seedvr2x Models, a tile or the blocks swapped connected to the upscale
  node.

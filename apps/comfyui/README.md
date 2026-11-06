# seedvr2x's ComfyUI nodes

[ComfyUI](https://github.com/Comfy-Org/ComfyUI) nodes that run
[SeedVR2](https://github.com/ByteDance-Seed/SeedVR) with seedvr2x's own model files, loaders and
memory planner: a picture, a batch of pictures or the frames of a short clip upscaled shot by shot
on the card you have, with the plan said on the node and a refusal, where one is due, before any
model loads.

> **Not published yet.** The nodes are built and tested against ComfyUI v0.44.0 and are installed by
> hand, as below; they are published with version 0.2.0 ([DESIGN.md](../../DESIGN.md)). Whole
> films, resume and 16-bit output are [the command line's](../seedvr2x/README.md): the
> `apps/seedvr2x` folder of this repository, beside these nodes.

## The nodes

- **seedvr2x Models** chooses the model file by name, with its size and its error per weight, as
  the [model card](../../models/hf/README.md) gives it (how far its stored weights are from
  ByteDance's original, the median over the 7B's 288 large matrices): `Sharp 7B, float16
  (16.5 GB, 0.02%)` (the default), `Sharp 7B, int8 (8.3 GB, 0.86%)`, `Sharp 7B, dynamic GGUF
  (4.8 GB, 7.55%)`, and the regular 7B in each. It also chooses where the files are read from
  (`model folder`: empty for Hugging Face's cache, or a folder) and whether the model stays in
  RAM between runs: `auto`, `on` or `off` ([Memory](#memory)). Nothing is loaded here: the first
  upscale run loads.
- **seedvr2x Cuts** finds where a clip changes scene ([Cuts and shots](#cuts-and-shots)): it takes
  the clip's frames, and gives the cut list that Upscale Video needs and a text of the cuts.
- **seedvr2x Upscale Image** upscales each picture of a batch on its own, to the output's short
  side given by `resolution` (1080 by default).
- **seedvr2x Upscale Video** upscales a clip's frames shot by shot, each shot in sliding windows
  of the model, no window crossing a cut.
- **seedvr2x Tile**, **seedvr2x Blocks Swapped** and **seedvr2x Window Length**, in the node
  library's `seedvr2x/plan settings`, set one of the plan's numbers yourself
  ([Setting a value yourself](#setting-a-value-yourself)). The other three are in `seedvr2x`.

Each upscale node takes `images` first (Upscale Video then takes `cuts`), then `models`, and gives
`images` and `plan`. Its settings are widgets, each at its default:

| Widget | Default | Notes |
| --- | --- | --- |
| resolution | `1080` | The output's short side, in pixels. |
| quality (Upscale Video only) | `standard` | How many times the model runs over each shot: `standard` once, `enhanced` twice, `maximum` four times, each run starting a frame or two further into the shot, each frame kept from the run that rebuilds it best. Sharper lines and less flicker where there is motion, little change on a still shot; 2 or 4 times as long as `standard`, and no more memory. A picture is one frame, so Upscale Image has no quality. |
| attention | `auto` | FlashAttention 2 where ComfyUI's environment has it, else PyTorch's SDPA. |
| colour correction | on (`split`) | On, `split`, keeps the input's overall colours and brightness with the model's fine detail; off, `none`, leaves the model's own colours, which can drift. |
| seed | `42` | ComfyUI's seed with its control after generate, `fixed`. |
| plan only | off | Make the plan, say it and give it as text, and stop: no model is loaded and the nodes after the images output do not run. |

The numbers the plan makes are optional inputs: [Setting a value yourself](#setting-a-value-yourself).

Each upscale node makes its plan at run time, where the input's size and the card are known, and
says it three ways: on the node, in ComfyUI's console, and as its `plan` output, text that
`Preview as Text` shows. The plan is made for the card as it would be without ComfyUI's own models.

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
  `model folder` is where to read the detector's file from instead of the cache, as in
  seedvr2x Models.
- `one shot` runs no detector: the whole clip is one shot, for a clip you know has no cut.

Its `text` output, also shown on the node, lists each cut by the frame its shot starts at, counted
from 0, with the detector's probability, and the **possible cuts**: scenes it found less sure than
the threshold, which are not in the list. Look at them in the clip; a threshold under a possible
cut's probability makes it a cut.

## Setting a value yourself

The numbers the plan makes are optional inputs of the upscale nodes, each a socket of its own
type: `encode tile` and `decode tile`, `blocks swapped`, and Upscale Video's `window length`.
Unconnected, the plan chooses, and its text says what it chose (`encode tile: auto -> 1024 px`).
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
at a revision pinned in the code, each checked by its pinned size and SHA-256 before it is used. The
first run downloads the model file chosen and the VAE (0.5 GB) into Hugging Face's cache, once:
`~/.cache/huggingface/hub`, or `$HF_HOME/hub`, or the directory `HF_HUB_CACHE` names. Set `HF_HOME`
before starting ComfyUI to put the cache on another disk. The command line finds the same files
there ([its README](../seedvr2x/README.md#model-files)).

Not ComfyUI's `models/` folders: its own SeedVR2 files sit there and are not seedvr2x's. To read
the files from a folder instead, give it in `model folder`: they are read there by name, nothing
downloaded: the VAE, `seedvr2x_ema_vae_fp16.safetensors`, and the model file chosen, under its name
in the repository (`seedvr2x_ema_7b_sharp_fp16.safetensors` for the default), the same files and
names the command line reads from a folder.

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
- **ComfyUI's image.** The output is its `IMAGE`, float32, and its savers write 8 bits or lossy
  video. The command line writes 16-bit PNG and lossless masters, with seedvr2x's record in each
  output; the nodes write no file and carry no record.

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
  the room: the next run then reads the files again, and says so on the node. Choosing another
  file frees the first one's memory before the second is pinned.
- **On the card, only while a node runs.** The plan counts on the card as it would be without
  ComfyUI's models, and ComfyUI is asked to free only the room the plan needs, with 0.25 GiB over
  it for the buffer its partial unloads keep: its models stay where both fit. The card is freed
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
  segments slow it down and run it). The nodes say which applies once, in the console, where the
  driver lends system memory. The nodes set no limit on torch's memory: ComfyUI's process is
  shared with its other nodes, and a limit set there cannot be taken back.
- **A card too small** is refused before any model loads, with the way out in the nodes' own
  words: a smaller file in seedvr2x Models, a tile or the blocks swapped connected to the upscale
  node.

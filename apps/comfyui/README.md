# seedvr2x's ComfyUI nodes

Three [ComfyUI](https://github.com/Comfy-Org/ComfyUI) nodes that run
[SeedVR2](https://github.com/ByteDance-Seed/SeedVR) with seedvr2x's own model files, loaders and
memory planner: a picture, a batch of pictures or the frames of a short clip upscaled on the card
you have, with the plan said on the node and a refusal, where one is due, before any model loads.

> **Not published yet.** The nodes are built and tested against ComfyUI v0.44.0 and are installed by
> hand, as below; they are published with version 0.2.0 ([DESIGN.md](../../DESIGN.md)). Long
> videos, shots, resume and 16-bit output are [the command line's](../seedvr2x/README.md).

## The nodes

- **seedvr2x Models** chooses the DiT file by name, with its size and its error per weight, as
  the [model card](../../models/hf/README.md) gives it (how far its stored weights are from
  ByteDance's original, the median over the 7B's 288 large matrices): `Sharp 7B, float16
  (16.5 GB, 0.02%)` (the default), `Sharp 7B, int8 (8.3 GB, 0.86%)`, `Sharp 7B, dynamic GGUF
  (4.8 GB, 7.55%)`, and the regular 7B in each. It also chooses where the files are read from
  (`model folder`: empty for Hugging Face's cache, or a folder) and whether the model stays in
  RAM between runs: `auto`, `on` or `off` ([Memory](#memory)). Nothing is loaded here: the first
  upscale run loads.
- **seedvr2x Upscale Image** upscales each picture of a batch on its own, to the output's short
  side given by `resolution` (1080 by default).
- **seedvr2x Upscale Video** upscales the frames it is given as one shot, in sliding windows of the
  model.

Each upscale node takes `images` first, then `models`, and gives `images` and `plan`. Its settings
are widgets, each at its default:

| Widget | Default | Notes |
| --- | --- | --- |
| resolution | `1080` | The output's short side, in pixels. |
| attention | `auto` | FlashAttention 2 where ComfyUI's environment has it, else PyTorch's SDPA. |
| colour correction | on (`split`) | On, `split`, keeps the input's overall colours and brightness with the model's fine detail; off, `none`, leaves the model's own colours, which can drift. |
| seed | `42` | ComfyUI's seed with its control after generate, `fixed`. |
| plan only | off | Make the plan, say it and give it as text, and stop: no model is loaded and the nodes after the images output do not run. |

The numbers the plan makes are optional inputs, sockets with no widget. Unconnected, the plan
chooses, and its text says what it chose (`encode tile: auto -> 1024 px`); a number connected is
taken exactly and the plan is built around it, or the run is refused before anything loads.

| Input | Values |
| --- | --- |
| encode tile, decode tile | The VAE's tiles, squares of this many output pixels a side: 512 to 2048 in steps of 64, or `2112` for no tiles, the whole image at once. |
| blocks swapped | The model's blocks that wait in RAM and pass through the card in turn, from `0` (none) to 36. |
| window length (Upscale Video) | The longest stretch of frames the model works on at once, in groups of 4 frames, from 5 to 128. |

Connect an `Int` node to one to set it. ComfyUI's `Primitive` node, which a double click on such
an input makes, sends it no value in ComfyUI v0.44.0 (its frontend 1.57.6): the plan's text then
says `auto`. No value node takes the input's range from it, so a value outside it is refused when
the node starts, saying the range.

Each upscale node makes its plan at run time, where the input's size and the card are known, and
says it three ways: on the node, in ComfyUI's console, and as its `plan` output, text that
`Preview as Text` shows. The plan is made for the card as it would be without ComfyUI's own models.

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
  optional input unconnected.

## Model files

The files are seedvr2x's own conversions, from [hekmon/seedvr2x](https://huggingface.co/hekmon/seedvr2x)
at a revision pinned in the code, each checked by its pinned size and SHA-256 before it is used. The
first run downloads the DiT file chosen and the VAE (0.5 GB) into Hugging Face's cache, once:
`~/.cache/huggingface/hub`, or `$HF_HOME/hub`, or the directory `HF_HUB_CACHE` names. Set `HF_HOME`
before starting ComfyUI to put the cache on another disk. The command line finds the same files
there ([its README](../seedvr2x/README.md#model-files)).

Not ComfyUI's `models/` folders: its own SeedVR2 files sit there and are not seedvr2x's. To read
the files from a folder instead, give it in `model folder`: they are read there by name, nothing
downloaded: the VAE, `seedvr2x_ema_vae_fp16.safetensors`, and the DiT file chosen, under its name
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

- **One shot.** Upscale Video has no cut detection: a cut inside a window blends two scenes.
  Split the clip at its cuts, or use the command line.
- **Nothing resumes.** A run that stops starts again.
- **In memory.** The clip and its output are held in memory, as ComfyUI holds any video: about
  100 MB a 4K frame in float32.
- **ComfyUI's image.** The output is its `IMAGE`, float32, and its savers write 8 bits or lossy
  video. The command line writes 16-bit PNG and lossless masters, with seedvr2x's record in each
  output; the nodes write no file and carry no record.

## Memory

- **In RAM between runs.** `keep in RAM between runs` on the Models node is `auto`, `on` or
  `off`. The weights are kept in pinned system memory after a run, up to 15.8 GiB for the float16
  file and the VAE, and the next run does not read them again. `auto` (the default) keeps them
  where they fit under ComfyUI's own cap on pinned memory (its `MAX_PINNED_MEMORY`: 40% of the
  RAM on Windows, up to 90% of it on Linux, none where pinned memory is off); `on` keeps them
  whatever the cap; `off` reads the files for each run and drops them after it. The plan says
  which: `keep in RAM between runs: auto -> kept, 7.90 GiB (ComfyUI allows 12.80 GiB)`, or
  `auto -> not kept: 15.80 GiB is over the 12.80 GiB ComfyUI allows`. Under ComfyUI's
  `--disable-smart-memory`, which unloads every model after each prompt, nothing is kept.
  ComfyUI's `Unload models` frees them, and ComfyUI may drop them itself when its own models need
  the room: the next run then reads the files again, and says so on the node. Choosing another
  file frees the first one's memory before the second is pinned.
- **On the card, only while a node runs.** The plan counts on the card as it would be without
  ComfyUI's models, and ComfyUI is asked to free only the room the plan needs, with 0.25 GiB over
  it for the buffer its partial unloads keep: its models stay where both fit. The card is freed
  when the node ends, on Cancel and on an error too.
- **A driver that lends system memory** (Windows, WSL2) does so where the card is full, which
  slows a run. cudaMallocAsync, ComfyUI's default allocator on its CUDA 13 builds, holds a run
  to the card less well than torch's own: start ComfyUI with `--disable-cuda-malloc` to give it.
  Torch's own allocator, without expandable segments (nothing in `PYTORCH_CUDA_ALLOC_CONF`),
  fails a step that is short of room where they slow it down and run it: set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before starting ComfyUI. The nodes say
  which applies once, in the console, where the driver lends system memory.
  The nodes set no limit on torch's memory: ComfyUI's process is shared with its other nodes,
  and a limit set there cannot be taken back.
- **A card too small** is refused before any model loads, with the way out in the nodes' own
  words: a smaller file in seedvr2x Models, a tile or the blocks swapped connected to the upscale
  node.

# seedvr2x-imgserver

seedvr2x's image server: pictures upscaled over HTTP by [SeedVR2](https://github.com/ByteDance-Seed/SeedVR),
the models loaded once and kept on the GPU. A client posts a picture's bytes and gets the upscaled
picture back, as a PNG or a lossless WebP. The server runs the same code as the
[command line's](../seedvr2x/README.md) still-picture path, so a picture comes out the same for
the same settings and seed.

> **Version 0.1.0.** [DESIGN.md](../../DESIGN.md) (The image server) is the specification. What
> has not been measured yet is marked as such below.

It is a production server of pictures, not the command line over HTTP: a request sets the output's
size and, if it wants, a seed. Everything else is fixed when the server starts.

- **Models loaded once,** after a warm-up on the largest picture the server takes.
- **The card's memory is reserved at the start** and held. A start the card can't hold is
  refused, naming what would fit.
- **A queue:** pictures wait for the GPU in order, up to a limit; past it, 503.
- **Lossless output:** PNG or lossless WebP, at the input's depth (8 or 16 bits), upright, with
  the source's colour profile and seedvr2x's record ([below](#the-record-in-each-output)).
- **`/health` and `/metrics`** for a container's probes and for Prometheus.

## Requirements

- **Linux**, x86_64.
- **An NVIDIA GPU that computes in bfloat16**, and its driver. PyTorch's CUDA 13.0 build needs a
  driver of the 580 series or later (580.65.06 on Linux). The server refuses to start without a
  CUDA device or on a card that does not compute in bfloat16.
- **GPU memory** for the models, the pictures and a margin. The start reads the card, plans, and
  refuses when the reservation does not fit, saying the most pictures at the size asked, and the
  largest size at that count, that would. One picture at a time may swap some of the DiT's blocks
  to system memory; more than one at a time needs the whole DiT on the card. The server has been
  developed on a 96 GB card and checked on an RTX 5090; its plan on other sizes of card is
  provisional until it is measured ([DESIGN.md](../../DESIGN.md#still-images), Still images).
- **Disk space:** about 17 GB for the models, downloaded once into Hugging Face's cache at the
  first start (or read from a directory you give).
- **For the image: Docker, and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)**
  on the host, which gives the container the driver. The image holds no driver and no CUDA
  toolkit; PyTorch's wheels bring CUDA's libraries.
- **From the repository:** [uv](https://docs.astral.sh/uv/), which installs Python 3.13 and the
  locked dependencies.
- No ffmpeg: the server reads and writes pictures with Pillow and seedvr2x's own writers.
- FlashAttention 2 is optional and not in the image: the server uses it when installed, else
  PyTorch's SDPA, and its log says which runs.

## Running it

### From the repository

```
git clone https://github.com/hekmon/seedvr2x.git
cd seedvr2x/apps/imgserver
uv sync
SEEDVR2X_API_KEY=a-long-random-string uv run seedvr2x-imgserver
```

It listens on 127.0.0.1:8390. The first start pulls seedvr2x's model files from Hugging Face, at
the revision this version pins, into Hugging Face's cache; `--model-dir` reads them from a
directory instead.

### The image

```
docker run --rm --gpus all \
  --stop-timeout 120 \
  -p 127.0.0.1:8390:8390 \
  -v hf-cache:/data/huggingface \
  -e SEEDVR2X_API_KEY=a-long-random-string \
  ghcr.io/hekmon/seedvr2x-imgserver
```

The image is published for linux/amd64 when a release is tagged: `ghcr.io/hekmon/seedvr2x-imgserver`,
tagged with the version (`0.1.0`), its major and minor (`0.1`) and `latest`. Options go after the
image's name.

- **The server listens on 0.0.0.0:8390 inside the container,** which port publishing needs.
  Publish it on 127.0.0.1 as above unless a reverse proxy is in front ([Before you expose
  it](#before-you-expose-it)).
- **The models stay outside the image.** The volume `hf-cache` holds Hugging Face's cache
  (`HF_HOME=/data/huggingface`): the first start fills it, the next ones read it. A bind mount
  must be writable by the image's user (UID 10001).
- **A model directory instead,** mounted read-only, holding the DiT and the VAE files:

  ```
  docker run --rm --gpus all -p 127.0.0.1:8390:8390 \
    -v /path/to/models:/models:ro \
    -e SEEDVR2X_API_KEY=a-long-random-string \
    ghcr.io/hekmon/seedvr2x-imgserver --model-dir /models
  ```

- **Closed networks:** `-e HF_HUB_OFFLINE=1` forbids any download; the models are then in the
  cache volume already, or in `--model-dir`.
- **The user is not root** (UID 10001), and owns its home and the cache directory only.
- **The health check** asks `/health` on port 8390 every 30 seconds, with a start period of 10
  minutes for the start's model check and loading. A container given another port in `--listen`
  needs its own check.
- **Stopping:** `docker stop` sends SIGTERM, which stops the server cleanly ([below](#stopping)),
  but Docker kills the container after 10 seconds by default, and a 4K picture takes 14 s on the
  RTX 5090 (DESIGN.md, Still images): give the picture running time to finish with
  `docker run --stop-timeout 120` (in Compose, `stop_grace_period: 2m`). 120 s is a choice: the
  running pictures' time at the card's speed, and the queue's pictures are answered 503 at once.
- **After a CUDA error the container exits,** so that a restart policy acts. A CUDA error other
  than out of memory leaves the GPU's state untrusted: the picture that failed is answered 500,
  the waiting ones 503, `/health` says `failed` and no new picture is taken while the pictures
  still running are answered, then the server exits with code 3 ([exit codes](#exit-codes)).
  Docker restarts it with `docker run --restart on-failure` or `--restart unless-stopped` (in
  Compose, `restart: on-failure` or `unless-stopped`), and Kubernetes restarts the container
  under any `restartPolicy` but `Never`. Without a policy the container stays stopped. The
  restart repeats the start, model check and warm-up included.

To build the image from a checkout, from the repository's root:
`docker build -f apps/imgserver/Dockerfile -t seedvr2x-imgserver .`. BuildKit reads
`apps/imgserver/Dockerfile.dockerignore`, which lets only the server and the library into the
build; the classic builder ignores that file and sends the whole repository as the build context,
so build with `docker buildx build` (or `DOCKER_BUILDKIT=1 docker build`, with buildx installed).

## Options

`seedvr2x-imgserver --help` lists them.

| Option | Default | Meaning |
|---|---|---|
| `--listen HOST:PORT` | `127.0.0.1:8390` | Where it listens. The image sets `0.0.0.0:8390`. |
| `--max-size WxH` | `3840x2160` | The largest output, in either orientation (even sides of 32 px at least). The reservation holds `--max-concurrent` pictures of it. |
| `--max-input PIXELS` | `50,000,000` | The largest picture taken, in pixels as stored. It is copied and resized on the GPU, so the reservation holds it. A body larger than this size allows is answered 413 before it is read. |
| `--max-concurrent N` | `1` | The pictures upscaled at once. More than one needs the whole DiT on the card ([several pictures at once](#several-pictures-at-once)). |
| `--queue N` | `8` | The pictures that may wait for the GPU. Past them, 503. |
| `--model-dir DIR` | none | Read the model files from DIR, nothing downloaded. By default seedvr2x's own files come from Hugging Face's cache, pulled at the pinned revision where missing. |
| `--dit-model FILE` | `seedvr2x_ema_7b_sharp_fp16.safetensors` | The DiT. |
| `--vae-model FILE` | `seedvr2x_ema_vae_fp16.safetensors` | The VAE. |
| `--encode-tile PX` | planned | Pin the encode's tiles (0: untiled; else a multiple of 64 from 512). One picture at a time only. |
| `--decode-tile PX` | planned | Pin the decode's tiles, as above. |
| `--swap-blocks N` | planned | Pin the DiT's blocks swapped to system memory. |
| `--api-key KEY` | none | Require `Authorization: Bearer KEY` on `/upscale`. `SEEDVR2X_API_KEY` sets it too, and `--api-key` wins. An empty key, one with a blank at either end and one that isn't printable ASCII with no space are refused. |
| `-v`, `--verbose` | off | Debug lines. |

The three pins fix one picture at a time's plan: with `--max-concurrent` above 1 the server
refuses them.

## The start

The server listens at once, so that a probe or a client finds it. `/health` says `starting` (503)
until the warm-up has passed. Its log then says, in order:

1. the API key's source (never the key), or a warning that there is none;
2. the model files checked by their SHA-256, pulled at the pinned revision where missing;
3. the card and the plan for pictures within `--max-size`, in the CLI's words;
4. the models loaded, and the attention that runs (`flash_attn_2` or `sdpa`); then the
   reservation, what it holds (the models, the pictures at once and each one's share), the CUDA
   context beside it, and what fits in it: the pictures within `--max-size` at once;
5. the warm-up, `--max-concurrent` pictures at `--max-size` at once, which proves the plan on the
   card before the first request, and the peak of memory it reached;
6. `ready: /health says ready`.

**A refused start** exits with code 1 and says why: no CUDA device, a card without bfloat16, an
empty API key, a port already taken, a model file missing or wrong, or a reservation the card
can't free, which names the most pictures at the size asked, and the largest size at that count,
that would fit. Lower `--max-size`, `--max-input` or `--max-concurrent`.

The start's duration has not been timed. Its two long parts, measured for one picture on an RTX
5090 (DESIGN.md, Still images), are the check of the DiT file (about 84 s) and the loading (77 s).

## The API

### `POST /upscale`

The body is the picture's bytes, with a `Content-Length` (a chunked body is refused). The format is
told by the picture's content, never by `Content-Type`: PNG, JPEG, WebP, TIFF, BMP and a GIF of one
frame, with the rules of the [command line's pictures](../seedvr2x/README.md). An animated picture,
a camera's raw file, an HDR picture, a 16-bit TIFF or a picture with transparency is refused (422),
the answer saying the way out.

| Request header | |
|---|---|
| `Seedvr2x-Resolution` | **Required.** The output's short side, in pixels. The output is at most `--max-size`, in either orientation. |
| `Seedvr2x-Seed` | Optional, a whole number from 0 to 4,293,967,295. A random one where absent. The seed used is always returned. |
| `Accept` | `image/png` by default; `image/webp` for lossless WebP. |
| `Authorization` | `Bearer KEY`, where the server has a key. |

Any other `Seedvr2x-` header is refused with 400, never ignored: everything else is fixed at the
server's start.

**The format.** PNG is the default. Lossless WebP holds 8 bits and an ICC profile: a picture it
can't hold (16 bits, or cICP, cHRM or gAMA without a profile) is answered as a PNG where the
client's `Accept` takes PNG too, else 406. The output keeps the input's depth, 8 bits for 8 and 16
for 16: a grainy 3840x2880 picture is 10 MB at 8 bits and 46 MB at 16 (DESIGN.md, Still images).
The picture is written upright whatever its orientation tag, and carries none.

**The answer** is the picture, with these headers:

| Header | |
|---|---|
| `Seedvr2x-Seed` | The seed used: send it back to get the same picture again. |
| `Seedvr2x-Source-Size` | The source's size as stored, `WIDTHxHEIGHT`. |
| `Seedvr2x-Output-Size` | The output's size, upright. |
| `Server-Timing` | `queue` (the wait for the GPU) and `upscale`, in milliseconds. |

A 4K picture from 1080p took 14 s to upscale on an RTX 5090 with the desktop live (DESIGN.md,
Still images). Throughput under load has not been measured yet.

### Statuses

Each error carries seedvr2x's message as text.

| Status | When |
|---|---|
| 200 | The picture. |
| 400 | A header wrong, missing or unknown (`Seedvr2x-Resolution` missing, a seed out of range, an unknown `Seedvr2x-` header, a setting given twice, an `Accept` weight not from 0 to 1); a `Content-Length` that is not one whole number. |
| 401 | The key missing or wrong, with `WWW-Authenticate: Bearer`. Checked before any body is read. |
| 404, 405, 501 | Another path; GET on `/upscale` (`Allow: POST`); another method. |
| 406 | An `Accept` the server can't meet, or a picture WebP can't hold where PNG is not taken either. |
| 411 | No `Content-Length`, or a chunked body. |
| 413 | A body over the limit, which follows `--max-input`. The body is not read. |
| 422 | The body is empty; the picture refused (not a picture, damaged, raw, HDR, animated, transparent, over `--max-input`, too small for the size asked); or the output over `--max-size`. The message names the way out. |
| 503 | The queue is full, or the server is starting or stopping, with `Retry-After` (15 s). After a CUDA error the server has failed and answers 503 without `Retry-After`: it takes no picture until it is restarted. |
| 500 | A picture failed (out of memory, an output not finite); the server goes on. A CUDA error other than out of memory answers 500 too, then the server exits ([exit codes](#exit-codes)). |

### `GET /health`

Plain text, no key needed.

| State | Status | |
|---|---|---|
| `starting` | 503 | The models are loading, or the warm-up has not passed. |
| `ready` | 200 | The server takes pictures. |
| `stopping` | 503 | It was asked to stop: no new picture. |
| `failed` | 503 | A CUDA error other than out of memory left the card's state untrusted; the server takes no picture, answers the pictures running, then exits with code 3 ([exit codes](#exit-codes)). |

### `GET /metrics`

Prometheus' text format, no key needed. The names:

| Metric | Type | |
|---|---|---|
| `seedvr2x_imgserver_requests_total{status}` | counter | Requests answered, by HTTP status. |
| `seedvr2x_imgserver_upscale_seconds` | histogram | A picture's upscale on the GPU. |
| `seedvr2x_imgserver_wait_seconds` | histogram | A picture's wait in the queue. |
| `seedvr2x_imgserver_output_pixels_total` | counter | The pixels of the pictures made. |
| `seedvr2x_imgserver_pictures_running` | gauge | The pictures upscaling now. |
| `seedvr2x_imgserver_pictures_waiting` | gauge | The pictures in the queue. |
| `seedvr2x_imgserver_vram_reserved_bytes` | gauge | torch's memory reserved on the GPU now. |
| `seedvr2x_imgserver_vram_reserved_peak_bytes` | gauge | The same, at its peak. |
| `seedvr2x_imgserver_vram_reservation_bytes` | gauge | The reservation torch is held to. |
| `seedvr2x_imgserver_ready` | gauge | 1 once the warm-up has passed, until the server stops or fails. |
| `seedvr2x_imgserver_info{...}` | gauge | The version, the model files, the attention kernel. |

### Examples

```
curl --data-binary @photo.jpg \
  -H 'Seedvr2x-Resolution: 2160' \
  -H "Authorization: Bearer $SEEDVR2X_API_KEY" \
  -D - -o out.png \
  http://127.0.0.1:8390/upscale
```

`-D -` prints the answer's headers, the seed among them. The same request as lossless WebP, with
the seed fixed:

```
curl --data-binary @photo.jpg \
  -H 'Seedvr2x-Resolution: 2160' -H 'Seedvr2x-Seed: 12345' \
  -H 'Accept: image/webp' \
  -H "Authorization: Bearer $SEEDVR2X_API_KEY" \
  -o out.webp http://127.0.0.1:8390/upscale
```

`--data-binary` sends the file as it is; `-d` would strip its line feeds. `curl -s
http://127.0.0.1:8390/health` and `/metrics` show the server's state.

## Several pictures at once

`--max-concurrent` is 1 by default. What running several at once gains is not known yet: it is to
be measured at 1, 2 and 4 at once on small, medium and large pictures before a default is set
(DESIGN.md, The image server), and no figure for it is in this document.

More than one picture at a time needs the whole DiT on the card: a card that has to swap the DiT's
blocks runs one at a time, and the start refuses more. A picture starts on the GPU when its
predicted peak fits what is left of the reservation, so smaller pictures may run more at once than
the count; the others wait in the queue, then get 503.

## The record in each output

Each picture carries seedvr2x's record as one XMP packet (PNG's `iTXt` chunk, WebP's XMP chunk):
the app and its version in `xmp:CreatorTool`, the source's and the output's sizes, the short side
asked, the model files' names with their SHA-256, the seed, the colour correction, the tiles and
the attention that ran. It holds nothing of the source's but its size, and no date, so the bytes
are the same every time for the same picture and settings.

It also marks the picture as made with a generative model, in IPTC's field for it
(`Iptc4xmpExt:DigitalSourceType`), which platforms read to label a picture so. DESIGN.md, The
record in each output, says what it holds and why.

## Stopping

SIGTERM or Ctrl-C: no new picture is taken, the pictures waiting are answered 503, the pictures
running finish and are answered, then the server exits with code 0. A second signal ends it at
once.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Stopped by SIGTERM or Ctrl-C, the pictures running finished and answered. |
| 1 | A refused start ([The start](#the-start)): the log says why. Restarting does not cure it. |
| 2 | A wrong option, as `--help` says. |
| 3 | Failed by a CUDA error other than out of memory, in a request or in the warm-up: the pictures running were answered first. A restart policy starts it again. |
| 128+N | A second signal N while stopping (143 for SIGTERM): the process ended at once. |

## Before you expose it

- **Put TLS in a reverse proxy in front.** Over plain HTTP the API key crosses the network in
  clear. On a LAN too, the key makes sure that only the proxy reaches the server.
- **Keep the server behind the proxy.** It is Python's `http.server` with a thread a connection,
  which Python's own documentation calls not hardened for production. The server limits a body's
  size before reading it, the time a client has for its request's head and for its body, the
  connections served at once, and the pictures taken in at once (past the count running and the
  queue, a picture is answered 503 before its body is read); the proxy
  also limits connections and request rates, and sets its own body limit to match `--max-input`
  (a 50 MP picture's body can be hundreds of megabytes).
- **Publish the container's port on 127.0.0.1** (`-p 127.0.0.1:8390:8390`) unless a proxy is in
  front: a bare `-p 8390:8390` publishes it on every interface.
- **Prefer `SEEDVR2X_API_KEY` to `--api-key`:** an option shows in the host's process list.
- `/health` and `/metrics` take no key, so that probes and Prometheus need none: keep them off the
  proxy's public side.
- Without a key, any client that reaches the server can use the GPU. The log warns of it at the
  start.

## Licence

The server is Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)); the image also carries the
library's MIT-licensed shot detector model, hence its label "Apache-2.0 AND MIT".

# Benchmarking SeedVR2 runs

[`scripts/bench.py`](../scripts/bench.py) wraps one run of SeedVR2's `inference_cli.py` and turns
it into a structured record: time and memory per phase, sub-timings, OOM/retry events. It is
standard library only, so any Python ≥ 3.9 can run it; the CLI itself runs with the SeedVR2
venv's python.

## Usage

```bash
# one measured run: everything after "--" goes to inference_cli.py untouched
python3 scripts/bench.py run 7b-1080-bs5 --seedvr2-dir /path/to/seedvr2 --runs-dir runs \
  --env 'PATH=/path/to/ffmpeg/bin:$PATH' \
  -- input.mkv --output out/ --output_format mp4 --video_backend ffmpeg \
     --model_dir /path/to/models --dit_model seedvr2_ema_7b_fp16.safetensors \
     --resolution 1080 --batch_size 5 --load_cap 21 --attention_mode sdpa

python3 scripts/bench.py table --runs-dir runs            # Markdown summary, latest record per name
python3 scripts/bench.py parse runs/*.log --json          # re-parse logs
python3 scripts/bench.py parse runs/x.log --append --runs-dir runs   # store a re-parsed record
```

| Subcommand | What it does |
|---|---|
| `run <name> [options] -- <CLI args>` | Refuses to start if a compute process holds the GPU (`--force` overrides). Records the GPU state, adds `--debug` if missing, runs the CLI with its output shown and teed to `<runs-dir>/<name>.log`, samples device memory every `--interval` s (default 0.1), writes the samples to `<name>.nvml.csv` and appends the record to `<runs-dir>/results.jsonl` |
| `parse <log>...` | Parses existing logs. If `<stem>.nvml.csv` is next to the log, the NVML figures are recomputed from it. If the results file holds the `run` record of that log, its run-only fields (command, env, exit status, wall time, RSS, GPU state) are carried over, so re-parsing after a parser fix loses nothing. Records get `source: parse` or `reparse` |
| `table [names...]` | One Markdown row per run (`--all`: every record instead of the latest per name) |

Other `run` options: `--env K=V` (repeatable, values go through `os.path.expandvars`),
`--python` (default `<seedvr2-dir>/.venv/bin/python`), `--gpu` (NVML index), `--overwrite`,
`--wrap SCRIPT` (runs `python SCRIPT inference_cli.py ARGS`, e.g.
[`attn_probe.py`](../scripts/attn_probe.py), [`vae_probe.py`](../scripts/vae_probe.py),
[`swap_probe.py`](../scripts/swap_probe.py) or [`vram_cap.py`](../scripts/vram_cap.py)). The CLI
gets `BENCH_LOG` and `BENCH_RUN_NAME` in its environment, so a wrapper can write its own output
next to the log.
`SEEDVR2_DIR` and `BENCH_RUNS_DIR` set the defaults for `--seedvr2-dir` and `--runs-dir`.

The CLI only needs `ffmpeg` on `PATH` for `--video_backend ffmpeg`: pass it through `--env PATH=...`.

## The record

One JSON object per line in `results.jsonl`. Memory values are in GiB: SeedVR2 divides by
1024³ and prints "GB".

| Field | Content |
|---|---|
| `name`, `timestamp`, `source`, `log` | Run name, start time (ISO, with offset), `run` / `parse` / `reparse`, log path |
| `command`, `cli_args`, `wrap` | Exact command line, the CLI arguments alone, the `--wrap` script if any |
| `args` | The `🔧 Arguments:` block of the log: every option's effective value, defaults included |
| `extra_env`, `alloc_conf`, `alloc_conf_source` | Environment added with `--env`. Effective `PYTORCH_CUDA_ALLOC_CONF`: inherited or set, or `backend:cudaMallocAsync` with source `cli-default`, because the CLI `setdefault`s it |
| `seedvr2_git` | `rev`, `describe`, `dirty` of the SeedVR2 checkout |
| `platform` | From the log header: SeedVR2 version, OS, GPU, Python, torch, CUDA, cuDNN, attention libraries, Conv3d workaround, initial free memory |
| `input` | Source video: frames, width, height, fps |
| `generation` | Frames processed, input/padded/output resolution, batch size, seed, first latents shape |
| `batches`, `chunks` | Number of batches per phase, and of streaming chunks (`--chunk_size`) |
| `phases."1"…"4"` | Aggregated per phase (VAE encode, DiT, VAE decode, post-processing), see below |
| `phase_runs` | Each phase occurrence (several when streaming): start/end log time, time, raw `└─` timings, NVML peak |
| `nvml` | Sampler (`nvml_v2`, or `nvidia-smi` as a fallback), sample count, baseline before the run, overall peak and its time, peak before phase 1 (CUDA context, model structures) |
| `events` | Log lines matching OOM / out of memory / retry / Traceback / error / ⚠️ / ❌, with time, phase and kind |
| `oom_events`, `retries`, `alloc_retries` | Counts. SeedVR2's VAE retries once after an OOM (`retry_on_oom`: "OOM during …", "Clearing memory and retrying"). `alloc_retries`: allocation failures torch's allocator recovered from by itself, without an exception ("[cudaMallocAsync] recovered from an allocation failure by trimming the pool and retrying", "expandable_segments: memory mapping failed"); they are neither OOM events nor retries ([vram.md](vram.md#the-allocator)) |
| `vram_cap` | With [`vram_cap.py`](../scripts/vram_cap.py): the emulated card (size, capacity, usable memory, context, room left for torch, mode, allocator, ballast). The ballast is subtracted from every NVML figure of the record (`nvml.ballast_gib`) |
| `total_s`, `avg_fps` | The CLI's own figures (its timer starts after the imports) |
| `wall_s`, `startup_s` | Wall time measured by bench.py, and wall − `total_s` (interpreter start + imports) |
| `max_rss_gib`, `cpu_user_s`, `cpu_sys_s` | `getrusage(RUSAGE_CHILDREN)` after the CLI exits |
| `exit_status`, `status`, `gpu_before`, `gpu_after` | Exit code. `ok`, `ok-after-oom` (completed after an OOM event; `alloc_retries` don't count), `oom`, `failed` or `incomplete`. GPU state from `nvidia-smi`: memory, P-state, temperature, power, clocks, compute processes |

Per phase (`phases."N"`):

| Field | Meaning |
|---|---|
| `time_s` | The CLI's "Phase N … complete" time (summed over chunks) |
| `torch_peak_alloc_gib` | Highest `Peak:` among the phase's memory snapshots. That is `max_memory_allocated`, which SeedVR2 resets after each snapshot, so the max over the phase's snapshots is the phase's peak |
| `torch_reserved_end_gib`, `torch_alloc_end_gib` | `reserved` / `allocated` in the "After phase N" snapshot |
| `torch_reserved_max_snapshot_gib` | Highest `reserved` among the phase's snapshots (still snapshots, not a peak) |
| `ram_process_end_gib`, `ram_process_max_snapshot_gib` | Process RSS from the snapshots |
| `nvml_peak_gib`, `nvml_start_gib`, `nvml_end_gib`, `nvml_growth_gib` | Device memory from NVML between the phase banner and its "After phase N" snapshot: peak, value at both ends, and peak − start |
| `snapshots` | Every memory snapshot in the phase ("After VAE loading for encoding", "After DiT loading for upscaling", …), each with the NVML value at that time |
| `timings` | The `└─` sub-timings grouped by label, with batch numbers replaced by `#` ("DiT inference #"): `n`, `total_s`, `first_s` (batch 1), `min_s`, `max_s`, `mean_rest_s` (mean without batch 1) |

## Caveats

- **torch "Peak" is peak *allocated*:** live tensors only. It excludes the allocator's cache,
  fragmentation, the CUDA context and library workspaces.
- **"reserved" is a snapshot at the end of a phase, not a peak.** SeedVR2 doesn't print
  `max_memory_reserved`.
- **NVML is what the driver really holds**: the cudaMallocAsync pool, the CUDA context,
  cuBLAS/cuDNN workspaces. It's the number that decides whether a run fits on a smaller GPU.
  With the default `backend:cudaMallocAsync`, **the pool never shrinks during a run**: reserved
  and NVML only grow, and a phase's NVML peak includes memory kept from earlier phases. Read
  `nvml_growth_gib` and the torch peak to see what a phase itself needs.
- NVML is sampled (100 ms by default). An allocation spike shorter than that can be missed, but
  pool growth persists, so it shows up in the next sample. The log has a 1 ms resolution, and
  phase windows are aligned on the host clock (the CLI logs local time without a date; bench.py
  anchors it to the run's start date and handles midnight).
- NVML counts the whole device. bench.py refuses to start when another compute process is
  present, but a process that starts mid-run would be counted.
- **First call warms up**: the first batch of each phase includes kernel selection and lazy
  initialization (cuDNN autotuning, Triton JIT…). `first_s` and `mean_rest_s` separate it.
  A single-batch run is all warmup.
- SeedVR2 prints the `└─` breakdown **sorted by duration**, not in batch order: bench.py puts it
  back in batch order. Some grandchild timers ("VAE decode") reuse one name for every batch, so
  the log shows the last batch's value under each batch. They are counted once and flagged with
  a `note`.
- **The GPU is power-limited, so consecutive runs drift.** VAE and DiT phases hit the card's
  600 W cap (`nvidia-smi -q -d PERFORMANCE`: "SW Power Cap: Active"), and clocks then depend on
  temperature. Back-to-back runs of the same configuration slowed by up to 9% over a series.
  Between sessions the gap is much larger: the same 1080p batch-81 DiT took 39.6 s in one
  session and 53.6 s in another, and the emulated-card runs of [vram.md](vram.md#recipe-per-card-size-validated)
  ran 40–60% slower than earlier ones (clocks down to 577 MHz under the cap). **Absolute times
  are only comparable between runs made back to back.** To compare an option that only affects
  one phase, normalize by a phase it can't affect (e.g. DiT time / VAE encode time when
  comparing attention backends), or interleave and repeat runs.
- **CUDA runs asynchronously, so the time between two log lines is not the GPU time of what
  lies between them.** The work is paid at the next synchronization: a copy to the CPU, or the
  next batch. Per-batch totals ("Encoded/Decoded batch N") are reliable; the split inside a batch
  is not.

## Reference: two runs on the reference stack

RTX PRO 6000 (96 GB), 7B fp16, 1080p anime input, no offload, no tiling. `smoke` is the run
from [environment.md](environment.md), parsed afterwards (no NVML).

| Run | Frames / batches | Encode | DiT (inference) | Decode | Post | Total | NVML peak |
|---|---|---|---|---|---|---|---|
| `bs 9, sageattn_3` (smoke) | 9 / 1 | 8.80 s · 19.7 / 22.7 / – | 16.97 s (12.20) · 19.6 / 22.7 / – | 15.53 s · 34.9 / 40.7 / – | 5.05 s · 1.8 / 40.7 / – | 47.2 s | – |
| `bs 5, sdpa` | 21 / 5 | 16.26 s · 14.6 / 17.6 / 18.4 | 27.83 s (22.95) · 18.1 / 18.6 / 19.4 | 36.76 s · 21.7 / 24.6 / 25.3 | 0.51 s · 1.0 / 24.6 / 25.3 | 82.3 s | 25.3 |

Cells: time · torch peak allocated / torch reserved at phase end / NVML peak (GiB).

- NVML sits a constant **0.78 GiB above torch reserved** at every phase end: the CUDA context and
  library workspaces. NVML reads 0.69 GiB before the first tensor is allocated.
- Under cudaMallocAsync, reserved never goes down. The DiT's 15.9 GiB of weights went into the
  17.6 GiB pool left by VAE encoding: reserved grew by only 1 GiB in phase 2. The run's NVML peak
  is set by VAE decode, and phase 4 keeps it although it allocates 1 GiB at most.
- Warmup: DiT inference takes 6.09 s for batch 1 against 4.85 s for the following full batches.
  The VAE batches show no visible warmup: about 3.8 s per 5-frame encode and 8.6 s per decode.
- Asynchrony in the log: for decode batch 1, 8.4 s pass before "Trimming spatial padding" and
  0.2 s for the copy to the CPU. For batch 2 it's 6.5 s and 2.1 s, with the same 8.6 s total
  (see caveats).

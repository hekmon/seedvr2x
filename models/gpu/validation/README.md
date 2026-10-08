# models/gpu/validation

The scripts that ran and scored the GPU validation of phase 2's files, and made the pages behind
[VALIDATION.md](../../VALIDATION.md): a queue of numz runs on one GPU, a CPU pool that scores each
run with colour.md's tools as it lands, the summaries that pair every file with its float16 model
and apply the guards, then the pairings, the crops for the eyes and a few counts. They are the
files that ran on the GPU box on 2026-10-07 and 08, with every path read from the environment
instead of written in (`glue.env`, below), and otherwise unchanged (Provenance, below).

## How they chain

```text
make_jobs_a.sh / make_jobs_b.sh LABEL  >>  $VAL_STATE/gpu/jobs.tsv        one line per run (bench.py run ...)
runner.sh: the first line without a marker, under flock $VAL_GPU_LOCK      -> gpu/done/<id> or gpu/failed/<id>
ms_pool2.sh, every 20 s: ms_jobs2.sh turns each done marker into CPU jobs   -> score/done/<id> (ms_job.sh)
    sc  colour_eval.py score           the scores against the GT, none and split:ycc:4:3
    vm  colour_eval.py render, vmaf    16-bit masters, VMAF v1 + CAMBI (ms_vmafcheck.py), ffv1_out.py --diff
    bd  colour_bands.py scan           the finest band's energy
    fr  fr_metrics.py                  DISTS and LPIPS on every frame (while score/FR_ON, FR4_ON exist)
    r7*, rsh*                          the float16 models' own references and seeds (their bands)
after any job ends: ms_sum.py, ms_sum4k.py, ms_sum4ksh.py                   -> $VAL_STATE/sum/*.md
by hand: pair4g.py, pair3b.py, floor_report.py, remake_all.sh, the crops, the counts
```

The runs themselves are numz's CLI at `4490bd1` through `bench.py`, wrapped by colour's
`colour_dump.py` (the latents and the decode) and, for our safetensors files, by
[`../ck_patch.py`](../ck_patch.py) (comfy-kitchen's layers); GGUF files through numz's own loader.

## The scripts

| Script | Does | Run |
|---|---|---|
| `runner.sh` | the GPU queue: `jobs.tsv`'s lines in file order, read again before each job, one job per hold of the GPU lock; inside the hold, no other compute process may run and `$VAL_DISK` must have 60 GB free, and the GPU's clock is logged; then a done or failed marker, `gpu-jobs.csv`; `STOP` between jobs; no start past `CUTOFF` without `GO_PAST_0845`; never exits on an empty queue | `bash $VAL_GLUE/runner.sh 2>&1 \| tee -a $VAL_STATE/85-runner.log`, in a terminal of its own |
| `status.sh` | the queue at a glance: counts, the running job, the last clock, an ETA | `bash $VAL_GLUE/status.sh` |
| `make_jobs_a.sh` | tier 1's lines (1080p, the 8 d1 clips): without an argument slice A's 52, else each label's; the float16 reference runs' flags, our wrapper and its mode or numz's GGUF loader, the dump | `bash $VAL_GLUE/make_jobs_a.sh LABEL >> $VAL_STATE/gpu/jobs.tsv`; `--list` |
| `make_jobs_b.sh` | slice B's lines (4K): the 7B's labels on its 6 shots, the sharp's on its 5, each shot's own flags, VAE decode in tiles | `bash $VAL_GLUE/make_jobs_b.sh LABEL >> $VAL_STATE/gpu/jobs.tsv` |
| `smoke_one.sh`, `smoke_jobs.txt` | the smoke test: one run per line of `smoke_jobs.txt` (tag, file, mode, numz's extra arguments) on anime-clean, in ck_patch's check mode; `DRY=1`: the command line and guards, no GPU | per line: `mkdir -p ${VAL_GPU_LOCK%/*} && flock $VAL_GPU_LOCK bash -c 'bash $VAL_GLUE/smoke_one.sh TAG FILE MODE [ARGS]'` |
| `smoke_check.py` | the smoke's checks: `report` (each run's backends, operations and per-layer figures from its `ck_patch.json`, its decode against the 7B float16's, BlockSwap's bit for bit); `equal DIR DIR` | `$METRICS_PY $VAL_GLUE/smoke_check.py report` |
| `reuse_smoke.sh` | when tier 1's fp8 W8A16 run of anime-clean equals the smoke's (check mode on) bit for bit, the smoke's other decodes stand for tier 1's on that clip: copied with a `SOURCE.txt`, their done markers written | `bash $VAL_GLUE/reuse_smoke.sh 2>&1 \| tee $VAL_STATE/87-reuse.log` |
| `check_run.sh` | a run's quick check by job id: ck_patch's report (inputs quantized in row chunks), NaN warnings and tracebacks in numz's log, wall times | `bash $VAL_GLUE/check_run.sh JOBID...` |
| `gpustats.py` | the GPU over the runs: clocks and clock-event reasons under load from the sampler, durations per label, failures | `$METRICS_PY $VAL_GLUE/gpustats.py` |
| `ms_env.sh`, `ms_env2.sh` | the pool's settings, sourced: the paths from glue.env, colour's per-clip bars and band rows, the shots, threads | |
| `ms_pool2.sh` | the CPU pool: the job list every 20 s, ready jobs started in order within `NMAX` slots (a 4K job takes 2) and 60 GB of memory after the running jobs' expected peaks; the summaries re-made after any job ends; `STOP` ends it once its jobs are done | `bash $VAL_GLUE/ms_pool2.sh 2>&1 \| tee -a $VAL_STATE/95-pool.log`, in a terminal of its own |
| `ms_jobs2.sh` | the job list (prints only): each done marker's sc, vm, bd, fr jobs with colour.md's commands, the float16 models' reference jobs, our Q4_K against numz's Q4_K_M, the 3B's inputs and page; each job's state (done, failed, running, ready, wait, blocked) | `bash $VAL_GLUE/ms_jobs2.sh`; `MS_DRY_GPUDONE=DIR` lists from other markers |
| `ms_job.sh` | one pool job, in a session of its own, on cores 0-31 at nice 19: its log, its peak memory, its marker, a line in `times.tsv` | started by the pool |
| `ms_status2.sh` | the pool at a glance: jobs by state, running, failed, times by kind, memory, the pages | `bash $VAL_GLUE/ms_status2.sh -q` |
| `ms_vmafcheck.py` | a VMAF JSON is whole: every frame finite, VMAF v1's 4K model from 2160 rows or 3840 columns up, else its 1080p one | called by the vm jobs |
| `ms_inputs.sh` | a run's encoder input and reference against colour's 7B s42 ones, by MD5 (the 3B's anime-clean runs); a report, never a block | called by the pool |
| `ms_sum.py` | 1080p: each file's frames paired with its float16 model's at seed 42 (colour_eval.py's series and bootstrap), the guards per kind of source, the strict and calibrated rules with the floor: `<label>.md`, `00-overview.md`, `7b-band.md`, `sharp-band.md`, `validation.md` | the pool; `$METRICS_PY $VAL_GLUE/ms_sum.py [--out DIR]` |
| `ms_sum4k.py`, `ms_sum4ksh.py` | the same at 4K, the 7B's files and the sharp's, against two-seed bands with the floor: `4k-*.md`, `4k-sh-*.md` | the pool |
| `ms_floor.py` | the floor: a minimum spread per metric, one unit of the resolution it is printed at, where the seeds agree closer than that; at 4K since 2026-10-08 (S16), at 1080p too since S23 (one rule for both resolutions), in both rules and every multiple, a multiple of the floor marked † (imported by the summaries and the pairings) | |
| `pair4g.py` | the 4 GB files against our Q4_K (or `--base`), 1080p: cell by cell against the 7B fp16's 3-seed spread or its floor, detail and band energy, the distance between the files | `$METRICS_PY $VAL_GLUE/pair4g.py`; `--base q4ki --labels dyn --name pair-dyn-q4ki.md` |
| `pair3b.py` | the 3B (ours from the current weights, numz's from the first) against the sharp 7B's dynamic GGUF, the sharp and the 7B, against the GT, their differences against the sharp's 3-seed spread or its floor; `--test`: stand-in labels | the pool, once both 3B labels are scored |
| `floor_report.py` | which verdicts and rows the floor changes: at 4K (the 7B's, the sharp's), at 1080p | `$METRICS_PY $VAL_GLUE/floor_report.py 4k\|4ksh\|1080` |
| `remake_all.sh` | every page again, after a pool summary in progress | `bash $VAL_GLUE/remake_all.sh [all]` |
| `ms_crops.py`, `ms_crops4k.py` | strips for the eyes (GT, bicubic, float16, each file) at colour's windows, cut by colour_crops.py: `$VAL_DATA/gpu/crops/<name>/`; `ms_crops.py --refs` names the float16 panels itself (`7B fp16`, `sharp fp16`), and captions the 3B's labels in words (3B current, 3B first) | `$METRICS_PY $VAL_GLUE/ms_crops.py --labels ... --name ... [--refs ...]`; the 3B's: `--refs "sharp fp16" --labels sh-dyn,3b-cur,3b-first --name crops-3b` |
| `count4k.py` | the cells past the strict and the calibrated line in 4K pages | `$METRICS_PY $VAL_GLUE/count4k.py $VAL_STATE/sum/4k-sh-*.md` |
| `dynmap.py` | the dynamic GGUF's type per matrix, both models, by layer kind and block range | `$METRICS_PY $VAL_GLUE/dynmap.py` |
| `mc_rules.py`, `mc_rules2.py` | the Monte Carlo of the rules: how often a further float16 seed fails the strict rule (29% of cells with 3 seeds, 50% with 2), and the calibrated multiple K (2.7, 10.9) | `python3 $VAL_GLUE/mc_rules.py` |
| `glue_env.py` | `env(NAME)` for the Python scripts | |

Each script's header says more: its inputs, outputs and rules.

## glue.env

Every path these scripts use is an environment variable, and `glue.env.example` names them all,
each with what it holds: this folder and the repository's scripts, numz's checkout and models,
the clips, colour's outputs, the metrics environment, ffmpeg, our model directory and
comfy-kitchen, and two roots for what the runs write: `VAL_DATA` (big files) and `VAL_STATE`
(markers, logs, pages), laid out as on the box. Copy it, fill it in, source it; a script refuses
to run, naming the first variable it needs that is unset.

```bash
cp models/gpu/validation/glue.env.example ~/glue.env    # then set each value
source ~/glue.env
mkdir -p $VAL_STATE/gpu $VAL_STATE/score $VAL_STATE/sum $VAL_DATA/gpu
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks_event_reasons.active,power.draw,temperature.gpu,memory.used,utilization.gpu \
  --format=csv -l 5 | tee -a $VAL_STATE/gpu/gpu-samples.csv           # the sampler status.sh and gpustats.py read
bash $VAL_GLUE/make_jobs_a.sh > $VAL_STATE/gpu/jobs.tsv               # slice A; more labels later with >>
bash $VAL_GLUE/runner.sh 2>&1 | tee -a $VAL_STATE/85-runner.log
touch $VAL_STATE/score/FR_ON $VAL_STATE/score/FR4_ON                  # fr_metrics.py's jobs, 1080p and 4K
bash $VAL_GLUE/ms_pool2.sh 2>&1 | tee -a $VAL_STATE/95-pool.log
```

Switches: the runner stops between jobs on `$VAL_STATE/gpu/STOP`, starts nothing from `CUTOFF`
(an environment variable, UTC) without `$VAL_STATE/gpu/GO_PAST_0845`, and takes a failed job again
once its marker `gpu/failed/<id>` is gone. The pool stops on `$VAL_STATE/score/STOP`, reads its
slots from `score/NMAX` (6 by default), its fr_metrics.py switches from `score/FR_ON` and
`score/FR4_ON`, test labels from `score/labels-extra.tsv` and `score/labels-extra-4k.tsv` (label,
decode path with `{cd}` for `<clip>-d1`, optional clips: the pipeline's validation scored colour's
sharp decodes as label `valsharp`, `$COLOUR_OUT/dumps/{cd}/sh42/decode.pt` written out), and
runs a failed job again once `score/failed/<id>` is gone.

## Provenance

Ported on 2026-10-08 from the box's copies, the ones that made VALIDATION.md's figures, under
the same names (but `smoke_one.sh` and `smoke_jobs.txt`, the smoke test's `one.sh` and
`jobs.txt`; the Monte Carlo's two scripts were run outside the box). Each file is the box's with
its paths read from glue.env: substituting the box's values back gives the box's file byte for
byte, all 30; then the Python files were formatted by ruff (100 columns), their syntax trees
unchanged, but for an unused import dropped from `floor_report.py`. `ms_crops.py` is the box's
later copy, ported the same way: `--refs` and the 3B's captions, which cut the 3B's strips (on
the earlier crop sets it writes the same files as the first copy). On the box, with its own
values: the generators print every label's lines byte for byte as the box's did (90 runs); the
summaries make the pool's 38 live pages, and from empty caches the box versions' pages; the
pairings, the floor report, the job list, the smoke test's report and the other tools give the
box versions' outputs; all but each page's `Generated <time> by <script>` line. The Monte Carlo
prints VALIDATION.md's figures (29% and 50% of cells, 94% and 98% of runs; K 2.70 and 10.93).
Then the floor went to 1080p too (S23, 2026-10-08): `ms_sum.py`, `ms_floor.py`, `pair4g.py`,
`pair3b.py` and `floor_report.py` took the same edit here and on the box, proved the same way
(the box's values substituted back give the box's new files byte for byte, the syntax trees
unchanged by ruff; with the box's values the summaries, the pairings and the floor report give
the box versions' pages, all but the stamped line).
Left on the box: the first pool (`ms_jobs.sh`, `ms_pool.sh`, `ms_status.sh`; `ms_jobs2.sh` keeps
their lines word for word), tests and one-off helpers (estimates, reorders, crops loops).

## Credits

Our code. It calls this repository's [research/scripts](../../../research/scripts):
`colour_dump.py`, `colour_eval.py`, `colour_bands.py`, `colour_crops.py`
([colour.md](../../../research/docs/colour.md#reproduce)), `bench.py`, `fr_metrics.py`,
`ffv1_out.py`; it runs numz's SeedVR2 CLI (numz/SeedVR2_comfyUI at `4490bd1`, Apache-2.0),
unchanged, through `bench.py` and our wrapper `../ck_patch.py`, which uses comfy-kitchen
(Apache-2.0); ffmpeg with libvmaf scores. No third-party code is copied here.

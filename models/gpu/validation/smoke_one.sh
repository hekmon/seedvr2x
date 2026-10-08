#!/bin/bash
# One GPU smoke run of models/gpu/ck_patch.py: anime-clean d1, 1080p, 45 frames, seed 42, colour's
# dump (decode.pt + latents.pt), ck_patch's check mode for the comfy-kitchen formats. Run INSIDE the lock:
#   mkdir -p ${VAL_GPU_LOCK%/*} && flock $VAL_GPU_LOCK bash -c 'bash $VAL_GLUE/smoke_one.sh TAG FILE MODE|- [numz args...]'
# DRY=1 bash smoke_one.sh TAG FILE MODE|- [...]: no GPU (CUDA hidden), outside the lock; numz stops once the DiT's
# and the VAE's files are resolved (CK_PATCH_DRYRUN=1): checks the command line, the paths and the guards.
set -u
for v in VAL_DATA VAL_STATE VAL_MODELS VAL_MODELDIR VAL_PYLIB NUMZ_DIR NUMZ_PY MEAS_CLIPS MEAS_SCRIPTS COLOUR_SCRIPTS FFMPEG_BIN; do [ -n "${!v:-}" ] || { echo "${0##*/}: $v is unset: source your glue.env (models/gpu/validation/glue.env.example)" >&2; exit 2; }; done
TAG=$1 FILE=$2 MODE=$3
shift 3
M=$VAL_DATA
S=$VAL_STATE/smoke
RUNS=$S/runs
EXTRA=()
if [ "${DRY:-0}" = 1 ]; then
  export CUDA_VISIBLE_DEVICES=
  RUNS=$S/dry
  TAG=dry-$TAG
  EXTRA=(--force --overwrite --env CK_PATCH_DRYRUN=1)
else
  apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
  if [ -n "$apps" ]; then echo "ABORT: a compute process runs outside the lock: $apps"; exit 9; fi
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks_event_reasons.active,power.draw,temperature.gpu,memory.used --format=csv
fi
CK=(--env "CK_PATCH_LOG=$RUNS/model-smoke-$TAG.ck_patch.json")
if [ "$MODE" != - ]; then
  CK+=(--env "CK_PATCH_MODE=$MODE" --env CK_PATCH_EXPECT=288 --env CK_PATCH_CHECK=1
       --env "CK_PATCH_FP16=$VAL_MODELS/dist/seedvr2x_ema_7b_fp16.safetensors")
fi
$NUMZ_PY $MEAS_SCRIPTS/bench.py run "model-smoke-$TAG" "${EXTRA[@]}" \
  --seedvr2-dir $NUMZ_DIR --runs-dir "$RUNS" \
  --wrap $COLOUR_SCRIPTS/colour_dump.py --wrap "$VAL_MODELS/gpu/ck_patch.py" \
  --env "PATH=$FFMPEG_BIN:$PATH" --env "PYTHONPATH=$VAL_PYLIB" --env PYTHONDONTWRITEBYTECODE=1 \
  --env "COLOUR_DUMP=$M/dumps/smoke/anime-clean-d1/$TAG" --env COLOUR_DUMP_INPUTS=0 "${CK[@]}" \
  -- $MEAS_CLIPS/anime-clean.d1.lr.mkv --output "$M/out/smoke/$TAG/" \
  --model_dir "$VAL_MODELDIR" --dit_model "$FILE" --resolution 1080 --attention_mode flash_attn_2 \
  --batch_size 45 --load_cap 45 --color_correction none --seed 42 "$@"
status=$?
[ "${DRY:-0}" = 1 ] || nvidia-smi --query-gpu=timestamp,clocks.sm,clocks_event_reasons.active,power.draw,temperature.gpu,memory.used --format=csv
exit $status

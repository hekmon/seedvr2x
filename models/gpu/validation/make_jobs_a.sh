#!/bin/bash
# Slice A's GPU jobs (model conversation, 2026-10-07), "id<TAB>command" lines for runner.sh, in run order:
#   bash $VAL_GLUE/make_jobs_a.sh > $VAL_STATE/gpu/jobs-a.tsv          (slice A's 52 lines)
#   bash $VAL_GLUE/make_jobs_a.sh LABEL... >> $VAL_STATE/gpu/jobs.tsv  (those labels' lines)
#   bash $VAL_GLUE/make_jobs_a.sh --list                                       (the label table)
# By file (design's order), each on the 8 d1 clips; then numz's own Q4_K_M on the 4 clips it has no run on.
# Labels added after slice A (S6, 2026-10-07; the same lines as slice A's of the same mode, but for the label,
# the file, the mode and the dump dir): nv4a16 (the NVFP4 file in CK_PATCH_MODE=w4a16), dyn and q4ki (S5's
# dynamic GGUF seedvr2x_ema_7b_dyn.gguf and its control seedvr2x_ema_7b_Q4_K_imatrix.gguf, our GGUF like q4k).
# S7 (2026-10-07, slice C's sharp 7B tier 1): sh-fp8a16 sh-fp8a8 sh-q4k sh-q80 sh-int8 sh-nv4a4 sh-nv4a16, the sharp
# 7B's phase-2 files (seedvr2x_ema_7b_sharp_*), as make_jobs_b.sh defines them: the same lines as the 7B's label of the
# same mode, but for the label (in the id), the file and the dump dir ($M/gpu/dumps/<clip>-d1/sh-<label>-s42/;
# COLOUR_DUMP_INPUTS=0: the sharp's encoder input and reference = the 7B s42's, md5-checked by colour). numz's flags
# are the sharp fp16's own s42 runs' too (colour's dumps/<clip>-d1/sh42/meta.json: the same flags, in the same order).
# S8 (2026-10-07): sh-dyn and sh-q4ki, the sharp's dynamic GGUF seedvr2x_ema_7b_sharp_dyn.gguf and its control
# seedvr2x_ema_7b_sharp_Q4_K_imatrix.gguf (seedvr2_gguf_dyn.py --model sharp, with the sharp fp16's own imatrix), our
# GGUF like sh-q4k (make_jobs_b.sh's rows).
# S15 (2026-10-08, the 3B check): 3b-cur, our seedvr2x_ema_3b_fp16.safetensors (models/seedvr2_fp16_3b.py: the 3B's
# current master, ByteDance-Seed/SeedVR2-3B at 37255ff, rounded to float16), through ck_patch.py with no CK mode as a
# GGUF label (no marker: numz's own loader, ck_patch.py for the model directory only); 3b-first, numz's own
# seedvr2_ema_3b_fp16.safetensors (the 3B's first weights), numz's own file as nq4km and the sharp's seeds 43/1234
# (resolved from $NUMZ_MODELS, where numz's validation cache already holds it). numz takes the 3B's
# configuration when "7b" is not in the name (src/core/model_configuration.py:719). An optional fourth column names
# the clip whose run also dumps the encoder input and the reference (COLOUR_DUMP_INPUTS=1): anime-clean for both, for
# the scorer to check them against colour's 7B s42 ones; every other run COLOUR_DUMP_INPUTS=0, as before.
# All or nothing: no line is printed (exit 1) when a label's file or a clip's input is missing (dyn, q4ki
# before their build); exit 2 on an unknown label. A new label = one more line in TABLE.
# numz's flags = the 7B fp16 s42 reference run's (colour's dumps/<clip>-d1/s42/meta.json: --resolution 1080
# --attention_mode flash_attn_2 --batch_size 45 --load_cap 45 --color_correction none --debug --seed 42);
# only --dit_model, --model_dir, --output, the dump dir and our wrapper (ck_patch.py, its environment) change.
# Each command removes the dump's files first and checks decode.pt and latents.pt after the run.
set -u
for v in VAL_DATA VAL_STATE VAL_MODELS VAL_MODELDIR VAL_PYLIB NUMZ_DIR NUMZ_PY NUMZ_MODELS MEAS_CLIPS MEAS_SCRIPTS COLOUR_SCRIPTS FFMPEG_BIN; do [ -n "${!v:-}" ] || { echo "${0##*/}: $v is unset: source your glue.env (models/gpu/validation/glue.env.example)" >&2; exit 2; }; done
M=$VAL_DATA
RUNS=$VAL_STATE/runs
CL=$MEAS_CLIPS
CLIPS="anime-clean anime-grain anime-dark cartoon-bright anime-sky anime-bright live-vfx live-slow"
NQ4KM_CLIPS="anime-sky anime-bright live-vfx live-slow"
FLAGS="--resolution 1080 --attention_mode flash_attn_2 --batch_size 45 --load_cap 45 --color_correction none --debug --seed 42"
# label  file  mode  [inputs clip]   (mode: a CK_PATCH_MODE; - no CK mode: our GGUF or float16 file, numz's own loaders;
# numz: numz's own file, no ck_patch; inputs clip: the one clip run with COLOUR_DUMP_INPUTS=1)
TABLE="fp8a16 seedvr2x_ema_7b_fp8_scaled.safetensors w8a16
fp8a8 seedvr2x_ema_7b_fp8_scaled.safetensors w8a8
q4k seedvr2x_ema_7b_Q4_K.gguf -
q80 seedvr2x_ema_7b_Q8_0.gguf -
int8 seedvr2x_ema_7b_int8_convrot.safetensors int8
nv4a4 seedvr2x_ema_7b_nvfp4.safetensors w4a4
nq4km seedvr2_ema_7b-Q4_K_M.gguf numz
nv4a16 seedvr2x_ema_7b_nvfp4.safetensors w4a16
dyn seedvr2x_ema_7b_dyn.gguf -
q4ki seedvr2x_ema_7b_Q4_K_imatrix.gguf -
sh-fp8a16 seedvr2x_ema_7b_sharp_fp8_scaled.safetensors w8a16
sh-fp8a8 seedvr2x_ema_7b_sharp_fp8_scaled.safetensors w8a8
sh-q4k seedvr2x_ema_7b_sharp_Q4_K.gguf -
sh-q80 seedvr2x_ema_7b_sharp_Q8_0.gguf -
sh-int8 seedvr2x_ema_7b_sharp_int8_convrot.safetensors int8
sh-nv4a4 seedvr2x_ema_7b_sharp_nvfp4.safetensors w4a4
sh-nv4a16 seedvr2x_ema_7b_sharp_nvfp4.safetensors w4a16
sh-dyn seedvr2x_ema_7b_sharp_dyn.gguf -
sh-q4ki seedvr2x_ema_7b_sharp_Q4_K_imatrix.gguf -
3b-cur seedvr2x_ema_3b_fp16.safetensors - anime-clean
3b-first seedvr2_ema_3b_fp16.safetensors numz anime-clean"
SLICE_A="fp8a16 fp8a8 q4k q80 int8 nv4a4 nq4km"
line() {  # LABEL CLIP FILE MODE [INPUTS_CLIP]   (MODE: a CK_PATCH_MODE; - our GGUF or float16 file; numz numz's own file)
  local L=$1 c=$2 f=$3 mode=$4 inputs=0
  [ "$c" = "${5:-}" ] && inputs=1
  local id=a-$c-d1-$L D=$M/gpu/dumps/$c-d1/$L-s42 w="" env="" md=$VAL_MODELDIR
  if [ "$mode" = numz ]; then
    md=$NUMZ_MODELS
  else
    w=" --wrap $VAL_MODELS/gpu/ck_patch.py"
    env=" --env PYTHONPATH=$VAL_PYLIB --env CK_PATCH_LOG=$RUNS/model-$id.ck_patch.json"
    [ "$mode" = - ] || env+=" --env CK_PATCH_MODE=$mode --env CK_PATCH_EXPECT=288"
  fi
  printf '%s\t%s\n' "$id" "rm -f $D/decode.pt $D/latents.pt $D/meta.json && $NUMZ_PY $MEAS_SCRIPTS/bench.py run model-$id --overwrite --seedvr2-dir $NUMZ_DIR --runs-dir $RUNS --wrap $COLOUR_SCRIPTS/colour_dump.py$w --env 'PATH=$FFMPEG_BIN:\$PATH' --env PYTHONDONTWRITEBYTECODE=1$env --env COLOUR_DUMP=$D --env COLOUR_DUMP_INPUTS=$inputs -- $CL/$c.d1.lr.mkv --output $M/gpu/out/$id/ --model_dir $md --dit_model $f $FLAGS && test -s $D/decode.pt && test -s $D/latents.pt"
}
if [ "${1:-}" = --list ]; then echo "$TABLE"; exit 0; fi
[ $# -gt 0 ] || set -- $SLICE_A
out=""
for L in "$@"; do
  row=$(awk -v l="$L" '$1 == l' <<< "$TABLE")
  [ -n "$row" ] || { echo "make_jobs_a: unknown label '$L' (see --list)" >&2; exit 2; }
  read -r _ f mode inp <<< "$row"
  md=$VAL_MODELDIR
  [ "$mode" = numz ] && md=$NUMZ_MODELS
  [ -f "$md/$f" ] || { echo "make_jobs_a: $md/$f is missing: no line for $L" >&2; exit 1; }
  cl=$CLIPS
  [ "$L" = nq4km ] && cl=$NQ4KM_CLIPS
  for c in $cl; do
    [ -f "$CL/$c.d1.lr.mkv" ] || { echo "make_jobs_a: $CL/$c.d1.lr.mkv is missing: no line for $L" >&2; exit 1; }
    out+=$(line "$L" "$c" "$f" "$mode" "$inp")$'\n'
  done
done
printf '%s' "$out"

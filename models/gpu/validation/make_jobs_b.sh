#!/bin/bash
# Slice B's GPU jobs (model conversation, 2026-10-07): the 6 "id<TAB>command" lines of ONE label for runner.sh,
# on the 6 batch-A 4K shots that have two 7B fp16 seeds (colour's dumps/<shot>-d1/s42 and s43), in this order:
# sollevante-painted, sollevante-line, sollevante-action, sollevante-dark, digital-sunrise, digital-space.
#   bash $VAL_GLUE/make_jobs_b.sh LABEL >> $VAL_STATE/gpu/jobs.tsv   (the orchestrator appends)
#   bash $VAL_GLUE/make_jobs_b.sh --list                                       (the label table)
# Each line is make_jobs_a.sh's line for that label (our wrapper ck_patch.py and its environment, CK_PATCH_MODE,
# CK_PATCH_EXPECT=288, the model directory, PYTHONPATH, bench.py run model-<id> --overwrite, the dump checks),
# with the shot's 4K input and numz's flags = the 7B fp16 s42 reference run's of that shot, in its order (colour's
# dumps/<shot>-d1/s42/meta.json): --resolution 2160 (sollevante-*) or 2016 (digital-*) --attention_mode
# flash_attn_2 --batch_size N --load_cap N (N = the shot's frames: 45, sollevante-dark 41) --color_correction none
# --debug --seed 42 --vae_decode_tiled --vae_decode_tile_size 2048 --vae_decode_tile_overlap 64 (encode untiled);
# only --dit_model, --model_dir, --output, the dump dir and our wrapper change.
# Ids b-<shot>-d1-<label>; dumps $M/gpu/dumps/<shot>-d1/<label>-s42/ (decode.pt, latents.pt, meta.json;
# COLOUR_DUMP_INPUTS=0: the inputs are the 7B s42's enc_bf16/ref_f32); numz's outputs $M/gpu/out/<id>/.
# Each command removes the dump's files first and checks decode.pt and latents.pt after the run.
# All or nothing: no line is printed (exit 1) when the label's file or a shot's input is missing (dyn, sh-dyn
# before their build); exit 2 on an unknown label. A new label = one more line in TABLE.
# S6 (2026-10-07): q4ki (seedvr2x_ema_7b_Q4_K_imatrix.gguf, S5's control of dyn: every matrix Q4_K with the same
# importance) and sh-q4ki (the sharp's) added, our GGUF like q4k and dyn.
# S9 (2026-10-07): the sharp 7B's files (labels sh-*) print 5 lines, on the 5 4K shots that have two SHARP fp16 seeds
# (colour's dumps/<shot>-d1/sh42 and sh43), in this order: digital-cockpit, ouatia-face, digital-space,
# sollevante-painted, cel4k-detail; numz's flags = the sharp 7B fp16 sh42 run's of that shot, in its order (colour's
# dumps/<shot>-d1/sh42/meta.json: --resolution 2016 (digital-*), 2160 (ouatia-face, sollevante-painted) or 2048
# (cel4k-detail), 45 frames = batch = load_cap, the same decode tiles, encode untiled); the rest of each line as the
# 7B's labels'. The 7B's labels' lines are unchanged (the 6 shots above, byte for byte).
set -u
for v in VAL_DATA VAL_STATE VAL_MODELS VAL_MODELDIR VAL_PYLIB NUMZ_DIR NUMZ_PY NUMZ_MODELS MEAS_SHOTS MEAS_SCRIPTS COLOUR_SCRIPTS FFMPEG_BIN; do [ -n "${!v:-}" ] || { echo "${0##*/}: $v is unset: source your glue.env (models/gpu/validation/glue.env.example)" >&2; exit 2; }; done
M=$VAL_DATA
RUNS=$VAL_STATE/runs
CL=$MEAS_SHOTS
SHOTS="sollevante-painted:2160:45 sollevante-line:2160:45 sollevante-action:2160:45 sollevante-dark:2160:41
digital-sunrise:2016:45 digital-space:2016:45"
SHOTS_SH="digital-cockpit:2016:45 ouatia-face:2160:45 digital-space:2016:45 sollevante-painted:2160:45
cel4k-detail:2048:45"
TILES="--vae_decode_tiled --vae_decode_tile_size 2048 --vae_decode_tile_overlap 64"
# label  file  mode   (mode: a CK_PATCH_MODE; - our GGUF, numz's own GGUF loader; numz: numz's own file, no ck_patch)
TABLE="fp8a16 seedvr2x_ema_7b_fp8_scaled.safetensors w8a16
fp8a8 seedvr2x_ema_7b_fp8_scaled.safetensors w8a8
q4k seedvr2x_ema_7b_Q4_K.gguf -
q80 seedvr2x_ema_7b_Q8_0.gguf -
int8 seedvr2x_ema_7b_int8_convrot.safetensors int8
nv4a4 seedvr2x_ema_7b_nvfp4.safetensors w4a4
nv4a16 seedvr2x_ema_7b_nvfp4.safetensors w4a16
nq4km seedvr2_ema_7b-Q4_K_M.gguf numz
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
sh-q4ki seedvr2x_ema_7b_sharp_Q4_K_imatrix.gguf -"
if [ "${1:-}" = --list ]; then echo "$TABLE"; exit 0; fi
[ $# = 1 ] || { echo "usage: $0 LABEL | --list" >&2; exit 2; }
L=$1
row=$(awk -v l="$L" '$1 == l' <<< "$TABLE")
[ -n "$row" ] || { echo "make_jobs_b: unknown label '$L' (see --list)" >&2; exit 2; }
read -r _ f mode <<< "$row"
md=$VAL_MODELDIR
[ "$mode" = numz ] && md=$NUMZ_MODELS
[ -f "$md/$f" ] || { echo "make_jobs_b: $md/$f is missing: no line for $L" >&2; exit 1; }
out=""
shots=$SHOTS
case $L in sh-*) shots=$SHOTS_SH ;; esac  # S9: the sharp's files on the sharp's 5 two-seed shots
for s in $shots; do
  IFS=: read -r c res n <<< "$s"
  [ -f "$CL/$c.d1.lr.mkv" ] || { echo "make_jobs_b: $CL/$c.d1.lr.mkv is missing: no line for $L" >&2; exit 1; }
  FLAGS="--resolution $res --attention_mode flash_attn_2 --batch_size $n --load_cap $n --color_correction none --debug --seed 42 $TILES"
  id=b-$c-d1-$L D=$M/gpu/dumps/$c-d1/$L-s42 w="" env=""
  if [ "$mode" != numz ]; then
    w=" --wrap $VAL_MODELS/gpu/ck_patch.py"
    env=" --env PYTHONPATH=$VAL_PYLIB --env CK_PATCH_LOG=$RUNS/model-$id.ck_patch.json"
    [ "$mode" = - ] || env+=" --env CK_PATCH_MODE=$mode --env CK_PATCH_EXPECT=288"
  fi
  out+=$(printf '%s\t%s' "$id" "rm -f $D/decode.pt $D/latents.pt $D/meta.json && $NUMZ_PY $MEAS_SCRIPTS/bench.py run model-$id --overwrite --seedvr2-dir $NUMZ_DIR --runs-dir $RUNS --wrap $COLOUR_SCRIPTS/colour_dump.py$w --env 'PATH=$FFMPEG_BIN:\$PATH' --env PYTHONDONTWRITEBYTECODE=1$env --env COLOUR_DUMP=$D --env COLOUR_DUMP_INPUTS=0 -- $CL/$c.d1.lr.mkv --output $M/gpu/out/$id/ --model_dir $md --dit_model $f $FLAGS && test -s $D/decode.pt && test -s $D/latents.pt")$'\n'
done
printf '%s' "$out"

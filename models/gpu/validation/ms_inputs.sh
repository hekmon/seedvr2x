#!/bin/bash
# Model conversation, S16 (2026-10-08): a run's encoder input and reference (its COLOUR_DUMP_INPUTS=1 dumps,
# enc_bf16.pt and ref_f32.pt) against colour's 7B fp16 s42 ones of the same clip, by md5 (same VAE, same input:
# expected equal). A report, never a block: prints and writes $VAL_STATE/sum/inputs-<label>-<cd>.txt, exits 0
# whatever it finds (2 only on missing arguments). The pool runs it for the 3B's anime-clean runs (ms_jobs2.sh's
# section 6, job x-inputs-anime-clean-d1-<label>, kind df).
#   bash $VAL_GLUE/ms_inputs.sh LABEL CD       e.g. 3b-cur anime-clean-d1
set -u
for v in VAL_GLUE VAL_STATE; do [ -n "${!v:-}" ] || { echo "${0##*/}: $v is unset: source your glue.env (models/gpu/validation/glue.env.example)" >&2; exit 2; }; done
L=${1:-} cd=${2:-}
if [ -z "$L" ] || [ -z "$cd" ]; then echo "usage: ms_inputs.sh LABEL CLIP-d1" >&2; exit 2; fi
source "$VAL_GLUE/ms_env2.sh"
out=$VAL_STATE/sum/inputs-$L-$cd.txt
mkdir -p "${out%/*}"
{
  echo "$(date -u +%FT%TZ) inputs of $L on $cd: the run's dumps $GD/$cd/$L-s42/ against colour's 7B fp16 s42 $CD/$cd/s42/"
  res=equal
  for f in enc_bf16.pt ref_f32.pt; do
    a=$GD/$cd/$L-s42/$f b=$CD/$cd/s42/$f
    if [ ! -e "$a" ] || [ ! -e "$b" ]; then
      echo "$f: MISSING ($( [ -e "$a" ] || echo "the run's $a" ) $( [ -e "$b" ] || echo "colour's $b" ))"
      [ "$res" = DIFFERENT ] || res=MISSING
      continue
    fi
    ma=$(md5sum < "$a" | cut -d' ' -f1) mb=$(md5sum < "$b" | cut -d' ' -f1)
    sa=$(stat -L -c %s "$a") sb=$(stat -L -c %s "$b")
    if [ "$ma" = "$mb" ]; then
      echo "$f: equal (md5 $ma, $sa B)"
    else
      echo "$f: DIFFERENT (the run's md5 $ma, $sa B; colour's $mb, $sb B)"
      res=DIFFERENT
    fi
  done
  echo "result: $res"
} | tee "$out.tmp"
mv -f "$out.tmp" "$out"
exit 0

#!/bin/bash
# Step 5 of S2's prompt (model conversation, 2026-10-07): the smoke runs reused for slice A on anime-clean d1.
# When a-anime-clean-d1-fp8a16 (no check mode) equals the smoke's fp8-w8a16 (check mode) bit for bit (decode.pt
# and latents.pt: the check mode is neutral and the runs deterministic), the smoke's decodes of fp8-w8a8, int8,
# nvfp4-w4a4, q4k and nvfp4-w4a16 (decode.pt, latents.pt, meta.json) are copied into the slice's dump dirs
# (<label>-s42, with SOURCE.txt) and their done markers written (the runner skips them, the pool scores them).
#   bash $VAL_GLUE/reuse_smoke.sh 2>&1 | tee $VAL_STATE/87-reuse.log
set -u
for v in VAL_DATA VAL_STATE VAL_GLUE METRICS_PY COLOUR_OUT; do [ -n "${!v:-}" ] || { echo "${0##*/}: $v is unset: source your glue.env (models/gpu/validation/glue.env.example)" >&2; exit 2; }; done
S=$VAL_DATA/dumps/smoke/anime-clean-d1
GD=$VAL_DATA/gpu/dumps/anime-clean-d1
G=$VAL_STATE/gpu
ts() { date -u +%FT%TZ; }
[ -e "$G/done/a-anime-clean-d1-fp8a16" ] || { echo "a-anime-clean-d1-fp8a16 is not done"; exit 1; }
out=$(CUDA_VISIBLE_DEVICES= nice -n 19 taskset -c 0-31 $METRICS_PY "$VAL_GLUE/smoke_check.py" equal "$S/fp8-w8a16" "$GD/fp8a16-s42")
echo "$(ts) smoke fp8-w8a16 (check mode) vs a-anime-clean-d1-fp8a16 (none): $out"
if [ "$out" != '{"decode.pt": true, "latents.pt": true}' ]; then echo "$(ts) NOT identical: no reuse, every job runs"; exit 2; fi
for pair in fp8-w8a8:fp8a8 int8:int8 nvfp4-w4a4:nv4a4 q4k:q4k nvfp4-w4a16:nv4a16; do
  tag=${pair%%:*} L=${pair#*:}
  id=a-anime-clean-d1-$L D=$GD/$L-s42
  if [ -e "$G/done/$id" ] || [ -e "$G/failed/$id" ]; then echo "$(ts) $id has a marker already: left alone"; continue; fi
  if [ -s "$G/RUNNING" ] && [ "$(cut -f1 "$G/RUNNING")" = "$id" ]; then echo "$(ts) $id is running: left alone"; continue; fi
  mkdir -p "$D"
  ok=1
  for f in decode.pt latents.pt meta.json; do
    cp "$S/$tag/$f" "$D/$f.tmp" && mv -f "$D/$f.tmp" "$D/$f" && cmp -s "$S/$tag/$f" "$D/$f" || ok=0
  done
  if [ $ok != 1 ]; then echo "$(ts) $id: copy FAILED, no marker"; continue; fi
  if [ "$tag" = q4k ]; then how="no check mode: a GGUF file, numz's own loader"; else how="CK_PATCH_CHECK=1"; fi
  {
    echo "decode.pt, latents.pt and meta.json copied from the GPU smoke run model-smoke-$tag ($(ts)):"
    echo "  dump $S/$tag/ ($how)"
    echo "  log $VAL_STATE/smoke/runs/model-smoke-$tag.log, report $VAL_STATE/smoke/runs/model-smoke-$tag.ck_patch.json"
    echo "reused for $id: a-anime-clean-d1-fp8a16 (no check mode) equals the smoke's fp8-w8a16 (check mode) bit for bit"
    echo "(decode.pt and latents.pt), so the check mode is neutral and the runs deterministic."
    echo "md5:"; (cd "$D" && md5sum decode.pt latents.pt meta.json | sed 's/^/  /')
  } > "$D/SOURCE.txt"
  printf "exit 0\nreused model-smoke-%s (see %s/SOURCE.txt)\nend %s\n" "$tag" "$D" "$(ts)" > "$G/.marker.reuse"
  mv -f "$G/.marker.reuse" "$G/done/$id"
  echo "$(ts) $id <- model-smoke-$tag: copied, done marker written"
done

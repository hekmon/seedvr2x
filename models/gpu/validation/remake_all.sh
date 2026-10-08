#!/bin/bash
# Model conversation, S16 (2026-10-08): re-make every summary page in one go, into $VAL_STATE/sum/:
#   ms_sum.py     1080p: 00-overview.md, <label>.md (3b-* included: informative, no band of their own), 7b-band.md,
#                 sharp-band.md, validation.md
#   ms_sum4k.py   the 7B's 4K (with the 4K floor): 4k-00-overview.md, 4k-<label>.md, 4k-7b-band.md, 4k-validation.md
#   ms_sum4ksh.py the sharp's 4K (with the 4K floor): 4k-sh-00-overview.md, 4k-sh-*.md, 4k-sharp-band.md,
#                 4k-sh-validation.md
#   pair3b.py     the 3B against the sharp's 4 GB pick: 3b-00-overview.md + .csv (once a 3B label is scored)
#   with "all":   pair4g.py (the 7B's 4 GB files against q4k, 1080p: $VAL_STATE/s6/sum/pair4g.md; nothing new lands for it)
# The pool (ms_pool2.sh) already re-makes the first three after every job that ends; the pool's own job x-pair3b
# makes the 3B page once, when both 3B labels are scored. Waits while a pool summary runs (same pages).
#   bash $VAL_GLUE/remake_all.sh [all] 2>&1 | tee $VAL_STATE/<NN>-remake.log
set -u
for v in VAL_GLUE VAL_DATA METRICS_PY FFMPEG_BIN; do [ -n "${!v:-}" ] || { echo "${0##*/}: $v is unset: source your glue.env (models/gpu/validation/glue.env.example)" >&2; exit 2; }; done
PY=$METRICS_PY
SC=$VAL_GLUE
export PATH=$FFMPEG_BIN:$PATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
st=0
run() {
  local t0=$SECONDS
  while pgrep -f "$PY $SC/ms_sum" > /dev/null; do sleep 5; done  # a pool summary writes the same pages
  echo "== $(date -u +%FT%TZ) $*"
  local out rc
  out=$(taskset -c 0-31 nice -n 19 "$@" 2>&1 < /dev/null); rc=$?
  tail -2 <<< "$out" | sed 's/^/   /'
  echo "   exit $rc, $(( SECONDS - t0 )) s"
  (( rc == 0 )) || st=1
}
run $PY $SC/ms_sum.py
run $PY $SC/ms_sum4k.py
run $PY $SC/ms_sum4ksh.py
if compgen -G "$VAL_DATA/gpu/eval/3b-*/*-d1" > /dev/null; then
  run $PY $SC/pair3b.py
else
  echo "== pair3b.py: no 3B scores yet (gpu/eval/3b-*), skipped"
fi
[ "${1:-}" = all ] && run $PY $SC/pair4g.py
echo "== $(date -u +%FT%TZ) done (Paris $(TZ=Europe/Paris date +%H:%M)), status $st"
exit $st

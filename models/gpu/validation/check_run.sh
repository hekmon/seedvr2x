#!/bin/bash
# Model conversation, S16 (2026-10-08): a GPU run's quick check, by job id (the runner's: a-<clip>-d1-<label>,
# b-<shot>-d1-<label>, ...). Read only. Per id:
#  - ck_patch's report $VAL_STATE/runs/model-<id>.ck_patch.json (runs through models/gpu/ck_patch.py only:
#    numz's own loader for GGUF and float16 files writes none): DiT, mode, the activation inputs quantized in row
#    chunks (x_chunked: calls = chunked inputs, chunks, largest input in elements; comfy-kitchen's 32-bit kernels),
#    refused dtype casts, aborted;
#  - numz's log $VAL_STATE/runs/model-<id>.log: "invalid value encountered in cast" (numpy's NaN cast
#    warning), a word "nan" (any case), "Traceback": counts and the first lines;
#  - wall time: the runner's (gpu/gpu-jobs.csv, the id's last line: load + run + dump), numz's own "completed
#    successfully in", ck_patch's started -> ended.
# Exit 0 when every id is clean (log found, no NaN warning, no Traceback, not aborted), else 1.
#   bash $VAL_GLUE/check_run.sh JOBID [JOBID ...]
set -u
for v in VAL_STATE METRICS_PY; do [ -n "${!v:-}" ] || { echo "${0##*/}: $v is unset: source your glue.env (models/gpu/validation/glue.env.example)" >&2; exit 2; }; done
R=$VAL_STATE/runs
GJ=$VAL_STATE/gpu/gpu-jobs.csv
PYJ=$(command -v python3 || echo $METRICS_PY)
[ $# -ge 1 ] || { echo "usage: check_run.sh JOBID [JOBID ...]" >&2; exit 2; }
bad=0
for id in "$@"; do
  ck=$R/model-$id.ck_patch.json log=$R/model-$id.log
  echo "== $id"
  if [ -f "$ck" ]; then
    CUDA_VISIBLE_DEVICES= "$PYJ" - "$ck" <<'EOF' || bad=1
import json, sys
from datetime import datetime
j = json.load(open(sys.argv[1], encoding="utf-8"))
x = j.get("x_chunked")
if isinstance(x, dict):
    xs = f"{x.get('calls', 0)} chunked inputs ({x.get('chunks', 0)} chunks, largest {x.get('largest_elements', 0):,} elements)"
elif x is None:
    xs = "0 chunked inputs (no x_chunked record)"
else:
    xs = f"x_chunked = {x!r}"
dur = ""
try:
    t0 = datetime.strptime(j["started"], "%Y-%m-%dT%H:%M:%S%z")
    t1 = datetime.strptime(j["ended"], "%Y-%m-%dT%H:%M:%S%z")
    dur = f"{(t1 - t0).total_seconds():.0f} s ({j['started']} -> {j['ended']})"
except (KeyError, TypeError, ValueError):
    dur = f"started {j.get('started')}, ended {j.get('ended')}"
dit = (j.get("dit") or {}).get("name", "?")
print(f"ck_patch: DiT {dit}, mode {j.get('mode')}, {xs}; refused dtype casts {j.get('refused_dtype_casts')}; "
      f"aborted {j.get('aborted')}; ck_patch wall {dur}")
sys.exit(1 if j.get("aborted") else 0)
EOF
  else
    echo "ck_patch: no report ($ck): not a ck_patch run (numz's own loader: GGUF, float16), or never started"
  fi
  if [ -f "$log" ]; then
    nc_=$(grep -c 'invalid value encountered in cast' "$log")
    nn=$(grep -ciwE 'nan' "$log")
    nt=$(grep -c 'Traceback' "$log")
    echo "numz log: \"invalid value encountered in cast\" x$nc_, word nan x$nn, Traceback x$nt ($log)"
    if (( nc_ + nn + nt )); then
      grep -nE 'invalid value encountered in cast|Traceback' "$log" | head -3 | cut -c1-200 | sed 's/^/   /'
      grep -niwE 'nan' "$log" | grep -v 'invalid value encountered in cast' | head -3 | cut -c1-200 | sed 's/^/   /'
      (( nc_ + nt )) && bad=1
    fi
    w=$(grep -oE 'completed successfully in [0-9.]+s' "$log" | tail -1)
    echo "numz: ${w:-no \"completed successfully\" line}"
  else
    echo "numz log: MISSING ($log)"; bad=1
  fi
  g=$(grep "^$id," "$GJ" 2>/dev/null | tail -1)
  n=$(grep -c "^$id," "$GJ" 2>/dev/null)
  if [ -n "$g" ]; then
    IFS=, read -r _ st t0 t1 sec <<< "$g"
    echo "runner: exit $st, $sec s wall ($t0 -> $t1)$( (( n > 1 )) && echo "; the last of $n runs of this id")"
  else
    echo "runner: no line in $GJ"
  fi
done
exit $bad

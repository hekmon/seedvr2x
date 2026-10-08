#!/bin/bash
# The model conversation's GPU queue at a glance (runner.sh's files):  bash $VAL_GLUE/status.sh
# Jobs of jobs.tsv done / failed / running / pending, the running job and its minutes, the runner's state, the
# last run-start GPU line, and an ETA (mean of the jobs run to success so far, Paris time).
for v in VAL_STATE VAL_GLUE; do [ -n "${!v:-}" ] || { echo "${0##*/}: $v is unset: source your glue.env (models/gpu/validation/glue.env.example)" >&2; exit 2; }; done
G=$VAL_STATE/gpu
now=$(date +%s)
paris() { TZ=Europe/Paris date -d "@$1" +%H:%M; }
echo "== $(date -u +%FT%TZ) (Paris $(paris $now))"
run=0 rid="" rs0=0
if [ -s "$G/RUNNING" ]; then
  IFS=$'\t' read -r rid rt0 rs0 rpid < "$G/RUNNING"
  kill -0 "$rpid" 2>/dev/null && run=1
fi
total=0 nd=0 nf=0 np=0 nbad=0 first="" fl=""
if [ -f "$G/jobs.tsv" ]; then
  while IFS=$'\t' read -r id cmd; do
    [[ -z "${id// /}" || "$id" == \#* ]] && continue
    if [[ ! "$id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ || -z "${cmd// /}" ]]; then nbad=$(( nbad + 1 )); continue; fi
    total=$(( total + 1 ))
    if [ -e "$G/done/$id" ]; then nd=$(( nd + 1 ))
    elif [ -e "$G/failed/$id" ]; then nf=$(( nf + 1 )); fl+=" $id (exit $(sed -n 's/^exit //p' "$G/failed/$id"))"
    elif (( run )) && [ "$id" = "$rid" ]; then :
    else np=$(( np + 1 )); [ -n "$first" ] || first=$id; fi
  done < "$G/jobs.tsv"
fi
extra=$(comm -23 <(ls "$G/done" 2>/dev/null | sort) <(cut -f1 "$G/jobs.tsv" 2>/dev/null | sort) | wc -l)
echo "-- jobs.tsv: $total jobs: done $nd, failed $nf, running $run, pending $np${first:+ (next: $first)}; malformed lines $nbad; done markers not in jobs.tsv: $extra"
[ -z "$fl" ] || echo "-- failed:$fl"
if (( run )); then echo "-- running: $rid since $rt0 (Paris $(paris $rs0)), $(( (now - rs0) / 60 )) min"
elif [ -n "$rid" ]; then echo "-- RUNNING is stale: $rid since $rt0 (its hold, pid $rpid, is gone)"
else echo "-- running: nothing"; fi
pids=$(pgrep -f "^bash $VAL_GLUE/runner.sh" | tr '\n' ' ')
echo "-- runner: ${pids:-NOT RUNNING}; STOP $([ -e "$G/STOP" ] && echo PRESENT || echo absent); GO_PAST_0845 $([ -e "$G/GO_PAST_0845" ] && echo PRESENT || echo absent)"
[ -s "$G/gpu-runstart.csv" ] && echo "-- last run start: $(tail -n 1 "$G/gpu-runstart.csv")"
[ -s "$G/gpu-samples.csv" ] && echo "-- GPU now: $(tail -n 1 "$G/gpu-samples.csv")"
if [ -s "$G/gpu-jobs.csv" ]; then
  read -r n mean < <(awk -F, 'NR > 1 && $2 == 0 { n++; s += $5 } END { printf "%d %d\n", n, n ? s / n : 0 }' "$G/gpu-jobs.csv")
  if (( n > 0 )); then
    left_s=0
    if (( run )); then left_s=$(( mean - (now - rs0) )); (( left_s < 0 )) && left_s=0; fi
    eta=$(( now + left_s + np * (mean + 10) ))
    echo "-- mean of $n jobs run to success: $(( mean / 60 )) min $(( mean % 60 )) s; ETA of the running + $np pending: Paris $(paris $eta) (the 08:45 stop not counted)"
  fi
  echo "-- last jobs:"; tail -n 3 "$G/gpu-jobs.csv" | sed 's/^/   /'
fi

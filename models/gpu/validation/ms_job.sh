#!/bin/bash
# Model conversation, slice A scoring: one job of the CPU pool (as colour's b2_cpu_job.sh, our paths and cores).
# ms_pool.sh starts it with setsid: its own session, so the pool's exit (Ctrl-C included) never reaches it, and a
# later pool adopts it (running/<id> with a live pid). Writes running/<id> ("pid kind start"), runs the job on
# cores 0-31 at nice 19 into logs/<id>.log, samples its session's RSS every 5 s (peak MB in peak/<id>), then
# done/<id> or failed/<id>, a line in times.tsv (id kind start end seconds peak-MB status), removes running/<id>.
# Usage: ms_job.sh ID KIND COMMAND
for v in VAL_STATE; do [ -n "${!v:-}" ] || { echo "${0##*/}: $v is unset: source your glue.env (models/gpu/validation/glue.env.example)" >&2; exit 2; }; done
id=$1 kind=$2 cmd=$3
SC=$VAL_STATE/score
s0=$(date +%s)
echo "$$ $kind $s0" > "$SC/running/$id"
rm -f "$SC/peak/$id"
{ echo "$(date -u +%FT%TZ) start $id (kind $kind, pid $$)"; echo "\$ $cmd"; } > "$SC/logs/$id.log"
( pk=0
  while :; do
    r=$(ps -s $$ -o rss= 2>/dev/null | awk '{s += $1} END {printf "%d", s / 1024}')
    if (( r > pk )); then pk=$r; echo "$pk" > "$SC/peak/$id"; fi
    sleep 5
  done ) &
smp=$!
if taskset -c 0-31 nice -n 19 bash -c "set -o pipefail; $cmd" >> "$SC/logs/$id.log" 2>&1; then
  st=done; touch "$SC/done/$id"; rm -f "$SC/failed/$id"
else
  st="FAILED (status $?)"; touch "$SC/failed/$id"
fi
kill "$smp" 2>/dev/null; wait "$smp" 2>/dev/null
s1=$(date +%s)
pk=$(cat "$SC/peak/$id" 2>/dev/null || echo 0)
echo "$(date -u +%FT%TZ) $st $id in $(( s1 - s0 )) s, peak RSS ~${pk} MB (session, sampled every 5 s)" >> "$SC/logs/$id.log"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$kind" "$s0" "$s1" "$(( s1 - s0 ))" "$pk" "$st" >> "$SC/times.tsv"
rm -f "$SC/running/$id"

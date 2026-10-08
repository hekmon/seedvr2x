#!/bin/bash
# Model conversation, slices A (1080p) and B (4K) scoring: the CPU pool's job list (ms_pool2.sh runs it at every
# turn; ms_status2.sh too). Slice A's lines are ms_jobs.sh's, word for word (same ids, kinds, commands, order);
# slice B's are added. One line per job, highest priority first:   id <TAB> kind <TAB> state <TAB> why <TAB> command
# state: done / failed (markers $SC/done/<id>, $SC/failed/<id>), running ($SC/running/<id>), ready, wait (why: what
# it waits for), blocked (its GPU run failed). kind: sc (score), vm (render + VMAF/CAMBI + distance), bd
# (colour_bands.py scan), fr (fr_metrics.py's all-frame DISTS etc., lowest priority), df (a distance between two
# kept masters); 4K jobs: sc4, vm4, bd4, fr4 (ms_pool2.sh weighs them 2 slots and expects their own peaks).
# Labels: every GPU marker $GPUDONE/a-<clip>-d1-<label> (1080p) and b-<shot>-d1-<label> (4K) (the runner writes it
# after the run's last write), and the lines of $SC/labels-extra.tsv (1080p) and $SC/labels-extra-4k.tsv (4K)
# ("label <TAB> decode path with {cd} for <clip>-d1 [<TAB> clips]", e.g. valsharp, val4k = tests), which wait for
# their decode file.
# Scoring = colour's commands, word for word but the paths and the ref tag:
#  - sc: colour_eval.py score --clip <clip>-d1 --gt GT --bars B --ref <label>=<the 7B s42 ref_f32.pt>
#    --content s42=<decode> --variants none,split:ycc:4:3 --out $EVD/<label>/<clip>-d1 --lpips --dists-every 9 --threads N
#  - vm: render none + split (gbrp16le) -> $MD/<label>, colour_eval.py vmaf (VMAF v1 + CAMBI; the 2160 model from
#    3840 columns) -> $VMD/<label>/<clip>-d1, checked (ms_vmafcheck.py, the shot's frame count), ffv1_out.py --diff
#    of the none master against the 7B s42's -> $DFD/<label>/<clip>-d1.s42.txt. The none master is kept (later
#    distances); the split master too while the fr switch of its resolution exists (the fr job deletes it).
#  - fr: fr_metrics.py GT --clip <clip>-d1 --out none@<label> 42 M --out split:ycc:4:3@<label> 42 M
#    --json-dir $FRD/<label>/<clip>-d1 --no-vmaf --threads N; 1080p while $SC/FR_ON exists, 4K while $SC/FR4_ON does
#  - bd: colour_bands.py scan ... --bicubic <clip>.d1.bicubic.mkv [--rows R] --every 3 --threads N -> $BD/<label>/<clip>-d1.json
# Slice A (1080p, ms_jobs.sh's): GT and bicubic from $CLIPS, colour's 1080p --bars and rows, N = 4 (T), 45 frames.
# Slice B (4K, the 6 shots $SHOTS_B, GPU ids b-<shot>-d1-<label>, dumps $GD/<shot>-d1/<label>-s42/decode.pt):
# colour's batch-A and baton-2 4K settings: GT and bicubic from $C4, --bars 0:0, whole frames, N = 8 (T4), the
# shot's frames (sollevante-dark 41) in the VMAF check.
# The 7B fp16's own (label 7b, ref tag f32 = colour's): r7m-<cd> renders its s42 masters (kept: every distance
# is to its none master) and VMAFs them into $VMD/7b-check (to compare with colour's); 1080p: seeds 43 and 1234
# scored, VMAF'd, distance to s42, scanned as a file's run (r7s-, r7v-, r7b-, r7f-): its 3-seed spread; its s42
# scores are colour's (eval-b2, vmaf-b2, bands-b2). 4K: its s42 and s43 scores are colour's (eval-b1), its s42 VMAF
# too (vmaf-b1 split, vmaf-b2 none), its s42 bands (bands/<shot>-d1.json), its s43 split VMAF (vmaf-b1); ours:
# r7v-<shot>-d1-s43 (s43 none + split VMAF + CAMBI into $VMD/7b, distance to s42), r7b-<shot>-d1-s43 (bands): its
# 2-seed spread. Read by ms_sum.py (1080p) and ms_sum4k.py (4K).
# The sharp 7B's files (labels sh-*): the same jobs, but their distance is to the sharp 7B fp16's s42 none master
# (rshm-<cd> renders it once from colour's sh42 decode, rshf- its fr_metrics), and the summaries pair them with the
# sharp's s42 scores (colour's eval-b2 ~sharp, at 1080p and 4K).
# The sharp 7B fp16's own seeds 43 and 1234 at 1080p (S8, 2026-10-07): our GPU runs a-<clip>-d1-sharp-s<seed> (dumps
# $GD/<clip>-d1/sharp-s<seed>/decode.pt; not a file's run: kept out of section 2), scored, VMAF'd, distance to the
# sharp's s42 none master, scanned, as the 7B's seeds (rshs-, rshv-, rshb-, rshf-<cd>-s<seed>; label sharp, ref tag
# sharp; their none masters deleted by fr as the 7B's seeds'): with colour's sh42, the sharp's own 3-seed spread,
# ms_sum.py's band for the sh-* labels once scored.
# S9 (2026-10-07): slice B of the sharp's files: the sh-* labels' 4K runs land on the sharp's 5 two-seed shots
# $SHOTS_SH (GPU ids b-<shot>-d1-sh-<file>), scored with colour's baton-2 4K commands (the first film: --bars 42:42,
# bands rows 64:2096: ms_env2.sh); section 3d: the sharp 7B fp16's own 4K references on those shots: its s42 masters
# (rshm-<cd>, as section 2 makes them for the first sh-* run of a shot), its seed 43 none + split VMAF + CAMBI
# into $VMD/sharp/<cd> and its distance to s42 (rshv-<cd>-s43), fr_metrics of both seeds (rshf-<cd>-s42, -s43):
# with colour's sh42 + sh43 scores (eval-b2 ~sharp), VMAF (vmaf-b2: s42 none + split, s43 split) and bands
# (bands-b2 <cd>-sharp.json, <cd>-sharp-s43.json), the sharp's own 2-seed spread at 4K (ms_sum4ksh.py's band).
# S16 (2026-10-08): the 3B's runs (labels 3b-cur, 3b-first; GPU ids a-<clip>-d1-3b-*) need nothing new: section 2
# scores them as any file's run (sc, vm, bd, fr; against the GT; split reference = colour's 7B s42 ref_f32.pt; their
# distance to the 7B fp16's s42 none master). Added: section 6 (their anime-clean run's encoder input and reference
# against colour's: ms_inputs.sh, a report, never a block) and section 7 (x-pair3b: pair3b.py ->
# sum/3b-00-overview.md once both 3B labels are scored on the 8 clips). A dry run lists the jobs from other GPU
# marker directories: MS_DRY_GPUDONE=DIR MS_DRY_GPUFAIL=DIR bash ms_jobs2.sh (nothing is run or written).
set -u
for v in VAL_GLUE; do [ -n "${!v:-}" ] || { echo "${0##*/}: $v is unset: source your glue.env (models/gpu/validation/glue.env.example)" >&2; exit 2; }; done
source "$VAL_GLUE/ms_env2.sh"
if [ -n "${MS_DRY_GPUDONE:-}" ]; then GPUDONE=$MS_DRY_GPUDONE GPUFAIL=${MS_DRY_GPUFAIL:-$MS_DRY_GPUDONE/../failed}; fi
FR=0; [ -e "$SC/FR_ON" ] && FR=1
FR4=0; [ -e "$SC/FR4_ON" ] && FR4=1
CE="$PY $CS/colour_eval.py"
BAND="$PY $CS/colour_bands.py"
CHK="$PY $VAL_GLUE/ms_vmafcheck.py"
FFD="$PY $MEAS/ffv1_out.py --diff"
FRM="$PY $MEAS/fr_metrics.py"
ctx1() { GTD=$CLIPS TH=$T NF=45 K4="" FRX=$FR; }               # slice A's settings
ctx4() { GTD=$C4 TH=$T4 NF=${NFR[$1]:-45} K4=4 FRX=$FR4; }     # slice B's settings for shot $1

job() {  # id kind needs; the command in CMD; needs: g:<gpu id> d:<our job id> f:<path>
  local id=$1 kind=$2 needs=$3 s why="" n
  if [ -e "$SC/done/$id" ]; then s=done
  elif [ -e "$SC/failed/$id" ]; then s=failed
  elif [ -e "$SC/running/$id" ]; then s=running
  else
    s=ready
    for n in $needs; do
      case $n in
        g:*) if [ -e "$GPUDONE/${n#g:}" ]; then :
             elif [ -e "$GPUFAIL/${n#g:}" ]; then s=blocked; why="GPU ${n#g:} failed"; break
             else s=wait; why="${why:+$why, }GPU ${n#g:}"; fi ;;
        d:*) [ -e "$SC/done/${n#d:}" ] || { s=wait; why="${why:+$why, }${n#d:}"; } ;;
        f:*) [ -e "${n#f:}" ] || { s=wait; why="${why:+$why, }${n##*/}"; } ;;
      esac
    done
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$id" "$kind" "$s" "${why:--}" "$CMD"
}

score_cmd() {  # CD LABEL RT CT DEC
  local c=${1%-d1}
  CMD="$CE score --clip $1 --gt $GTD/$c.gt.mkv --bars ${BARS[$c]:-0:0} --ref $3=$CD/$1/s42/ref_f32.pt --content $4=$5 --variants none,$SP --out $EVD/$2/$1 --lpips --dists-every 9 --threads $TH"
}
masters() {  # CD LABEL RT CT -> MN MS (none and split masters), M42 (the reference's s42 none master: the
  # sharp 7B fp16's for the sharp's files sh-*, else the 7B fp16's), REF=1 for a reference render (7b or sharp, s42)
  MN=$MD/$2/$1.$4.none~$3.gbrp16le.mkv MS=$MD/$2/$1.$4.split_ycc_4_3~$3.gbrp16le.mkv M42=$MD/7b/$1.s42.none~f32.gbrp16le.mkv
  case $2 in sh-*|sharp) M42=$MD/sharp/$1.s42.none~sharp.gbrp16le.mkv ;; esac
  REF=0; case $2/$4 in 7b/s42|sharp/s42) REF=1 ;; esac
}
vm_cmd() {  # CD LABEL RT CT DEC VMAF_OUT
  local cd=$1 L=$2 rt=$3 ct=$4 dec=$5 vout=$6 g=$GTD/${1%-d1}.gt.mkv
  masters $cd $L $rt $ct
  CMD="mkdir -p $MD/$L $DFD/$L && $CE render --clip $cd --gt $g --ref $rt=$CD/$cd/s42/ref_f32.pt --content $ct=$dec --variants none,$SP --out $MD/$L --pix-fmt gbrp16le --threads $TH"
  CMD+=" && $CE vmaf --clip $cd --gt $g --master 'none@$rt=$MN' --master '$SP@$rt=$MS' --master-content $ct --out $vout --threads $TH"
  CMD+=" && $CHK '$vout/$cd.$ct.none~$rt~.json' $NF && $CHK '$vout/$cd.$ct.split_ycc_4_3~$rt~.json' $NF"
  if [ $REF = 0 ]; then
    CMD+=" && $FFD '$MN' '$M42' > '$DFD/$L/$cd.$ct.txt.tmp' && mv -f '$DFD/$L/$cd.$ct.txt.tmp' '$DFD/$L/$cd.$ct.txt' && cat '$DFD/$L/$cd.$ct.txt'"
  fi
  if [ $FRX = 0 ] && [ $REF = 0 ]; then
    CMD+=" && rm -f '$MS'"
    case $L in 7b|sharp) CMD+=" && rm -f '$MN'" ;; esac
  fi
  true
}
fr_cmd() {  # CD LABEL RT CT: fr_metrics on the two masters, then the masters vm kept for it deleted
  local cd=$1 L=$2 rt=$3 ct=$4 g=$GTD/${1%-d1}.gt.mkv
  masters $cd $L $rt $ct
  CMD="mkdir -p $FRD/$L/$cd && $FRM $g --clip $cd --out none@$rt ${ct#s} '$MN' --out $SP@$rt ${ct#s} '$MS' --json-dir $FRD/$L/$cd --no-vmaf --threads $TH"
  if [ $REF = 0 ]; then
    CMD+=" && rm -f '$MS'"
    case $L in 7b|sharp) CMD+=" && rm -f '$MN'" ;; esac
  fi
  true
}
bd_cmd() {  # CD DEC OUT
  local c=${1%-d1}
  CMD="mkdir -p ${3%/*} && $BAND scan --clip $1 --gt $GTD/$c.gt.mkv --ref $CD/$1/s42/ref_f32.pt --content $2 --bicubic $GTD/$c.d1.bicubic.mkv${ROWS[$c]:+ --rows ${ROWS[$c]}} --every 3 --threads $TH --out $3"
}

FRJOBS=""  # fr jobs last: "id|cd|label|rt|ct|needs|res" lines (res 1: 1080p, 4: 4K)
# 1. the 7B fp16's 1080p s42 masters (every distance needs its none master; the 4K ones come in 2b)
ctx1
for c in $CLIPS_A; do
  cd=$c-d1
  vm_cmd $cd 7b f32 s42 $CD/$cd/s42/decode.pt $VMD/7b-check/$cd; job r7m-$cd vm ""
  FRJOBS+="r7f-$cd-s42|$cd|7b|f32|s42|d:r7m-$cd|1"$'\n'
done
# 2. the files' runs, in GPU landing order (marker time), the extra labels (tests) first: 1080p (slice A), then
# 2b the 7B fp16's 4K s42 masters, then 4K (slice B)
units=""
for xf in labels-extra.tsv:1 labels-extra-4k.tsv:4; do
  res=${xf##*:} xf=$SC/${xf%:*}
  [ -f "$xf" ] || continue
  while IFS=$'\t' read -r L tpl clips; do  # an optional third column: the clips (space list), else all of them
    [[ -z "${L:-}" || "$L" == \#* ]] && continue
    if [ "$res" = 4 ]; then all=$SHOTS_B; else all=$CLIPS_A; fi
    for c in ${clips:-$all}; do units+="0 x $L $c ${tpl//\{cd\}/$c-d1} $res"$'\n'; done
  done < "$xf"
done
for f in "$GPUDONE"/a-* "$GPUFAIL"/a-*; do
  [ -e "$f" ] || continue
  n=${f##*/} t=$(stat -c %Y "$f")
  case $n in a-*-d1-sharp-s[0-9]*) continue ;; esac  # the sharp fp16's own seeds: section 3c
  for c in $CLIPS_A; do
    case $n in a-$c-d1-*) L=${n#a-$c-d1-}; units+="$t g $L $c $GD/$c-d1/$L-s42/decode.pt 1"$'\n'; break ;; esac
  done
done
for f in "$GPUDONE"/b-* "$GPUFAIL"/b-*; do
  [ -e "$f" ] || continue
  n=${f##*/} t=$(stat -c %Y "$f")
  for c in $SHOTS_B $SHOTS_SH; do  # S9: the sharp's B shots too
    case $n in b-$c-d1-*) L=${n#b-$c-d1-}; units+="$t g $L $c $GD/$c-d1/$L-s42/decode.pt 4"$'\n'; break ;; esac
  done
done
declare -A SEEN=()
unit() {  # src L c dec res
  local src=$1 L=$2 c=$3 dec=$4 res=$5 cd=$3-d1 gp need refjob
  [ -n "${c:-}" ] || return 0
  [ -z "${SEEN[$L/$c]:-}" ] && SEEN[$L/$c]=1 || return 0
  if [ "$res" = 4 ]; then ctx4 $c; gp=b; else ctx1; gp=a; fi
  if [ "$src" = g ]; then need="g:$gp-$cd-$L f:$dec"; else need="f:$dec"; fi
  refjob=r7m-$cd
  case $L in sh-*)  # the sharp's files: their reference = the sharp 7B fp16 s42 (colour's sh42 decode), rendered once
    refjob=rshm-$cd
    if [ -z "${SEEN[sharp/$c]:-}" ]; then
      SEEN[sharp/$c]=1
      vm_cmd $cd sharp sharp s42 $CD/$cd/sh42/decode.pt $VMD/sharp-check/$cd; job rshm-$cd vm$K4 "f:$CD/$cd/sh42/decode.pt"
      FRJOBS+="rshf-$cd-s42|$cd|sharp|sharp|s42|d:rshm-$cd|$res"$'\n'
    fi ;;
  esac
  score_cmd $cd $L $L s42 $dec; job $L-sc-$cd sc$K4 "$need"
  vm_cmd $cd $L $L s42 $dec $VMD/$L/$cd; job $L-vm-$cd vm$K4 "$need d:$refjob"
  bd_cmd $cd $dec $BD/$L/$cd.json; job $L-bd-$cd bd$K4 "$need"
  [ "$src" = x ] && [ "$res" = 4 ] && return 0  # 4K tests (labels-extra-4k.tsv): no fr_metrics
  FRJOBS+="$L-fr-$cd|$cd|$L|$L|s42|d:$L-vm-$cd|$res"$'\n'
}
while read -r _ src L c dec res; do unit "$src" "$L" "$c" "$dec" "$res"; done < <(grep ' 1$' <<< "$units" | sort -n -k1,1 -s)
for c in $SHOTS_B; do
  ctx4 $c; cd=$c-d1
  vm_cmd $cd 7b f32 s42 $CD/$cd/s42/decode.pt $VMD/7b-check/$cd; job r7m-$cd vm4 ""
  FRJOBS+="r7f-$cd-s42|$cd|7b|f32|s42|d:r7m-$cd|4"$'\n'
done
while read -r _ src L c dec res; do unit "$src" "$L" "$c" "$dec" "$res"; done < <(grep ' 4$' <<< "$units" | sort -n -k1,1 -s)
# 3. the 7B fp16's other seeds: 1080p 43 and 1234 (its 3-seed spread), then 4K 43 (its 2-seed spread: the score
# is colour's eval-b1, the split VMAF colour's vmaf-b1; the none VMAF, the distance and the bands are ours)
ctx1
for s in 43 1234; do
  for c in $CLIPS_A; do
    cd=$c-d1 dec=$CD/$c-d1/s$s/decode.pt
    score_cmd $cd 7b f32 s$s $dec; job r7s-$cd-s$s sc "f:$dec"
    vm_cmd $cd 7b f32 s$s $dec $VMD/7b/$cd; job r7v-$cd-s$s vm "f:$dec d:r7m-$cd"
    bd_cmd $cd $dec $BD/7b/$cd-s$s.json; job r7b-$cd-s$s bd "f:$dec"
    FRJOBS+="r7f-$cd-s$s|$cd|7b|f32|s$s|d:r7v-$cd-s$s|1"$'\n'
  done
done
for c in $SHOTS_B; do
  ctx4 $c; cd=$c-d1 dec=$CD/$c-d1/s43/decode.pt
  vm_cmd $cd 7b f32 s43 $dec $VMD/7b/$cd; job r7v-$cd-s43 vm4 "f:$dec d:r7m-$cd"
  bd_cmd $cd $dec $BD/7b/$cd-s43.json; job r7b-$cd-s43 bd4 "f:$dec"
  FRJOBS+="r7f-$cd-s43|$cd|7b|f32|s43|d:r7v-$cd-s43|4"$'\n'
done
# 3c. the sharp 7B fp16's own seeds at 1080p, 43 and 1234 (S8): scored, VMAF'd, distance to the sharp's s42, scanned,
# as the 7B's (its s42 master rendered here when no sh-* run of the clip has landed yet: the same job as section 2's)
ctx1
for s in 43 1234; do
  for c in $CLIPS_A; do
    cd=$c-d1 dec=$GD/$c-d1/sharp-s$s/decode.pt gid=a-$c-d1-sharp-s$s
    if [ -z "${SEEN[sharp/$c]:-}" ]; then
      SEEN[sharp/$c]=1
      vm_cmd $cd sharp sharp s42 $CD/$cd/sh42/decode.pt $VMD/sharp-check/$cd; job rshm-$cd vm "f:$CD/$cd/sh42/decode.pt"
      FRJOBS+="rshf-$cd-s42|$cd|sharp|sharp|s42|d:rshm-$cd|1"$'\n'
    fi
    score_cmd $cd sharp sharp s$s $dec; job rshs-$cd-s$s sc "g:$gid f:$dec"
    vm_cmd $cd sharp sharp s$s $dec $VMD/sharp/$cd; job rshv-$cd-s$s vm "g:$gid f:$dec d:rshm-$cd"
    bd_cmd $cd $dec $BD/sharp/$cd-s$s.json; job rshb-$cd-s$s bd "g:$gid f:$dec"
    FRJOBS+="rshf-$cd-s$s|$cd|sharp|sharp|s$s|d:rshv-$cd-s$s|1"$'\n'
  done
done
# 3d. the sharp 7B fp16's own 4K references (S9) on its 5 two-seed shots: its s42 masters (rshm-<cd>, here when no
# sh-* run or test of the shot has made it in section 2) and seed 43 (colour's sh43 decode): none + split VMAF + CAMBI
# into $VMD/sharp/<cd>, distance to its s42 (rshv-<cd>-s43); fr_metrics of both seeds (rshf-<cd>-s42, rshf-<cd>-s43;
# the s43 masters deleted by it). Its s42 and s43 scores and bands are colour's (eval-b2 ~sharp, bands-b2).
for c in $SHOTS_SH; do
  ctx4 $c; cd=$c-d1 dec=$CD/$c-d1/sh43/decode.pt
  if [ -z "${SEEN[sharp/$c]:-}" ]; then
    SEEN[sharp/$c]=1
    vm_cmd $cd sharp sharp s42 $CD/$cd/sh42/decode.pt $VMD/sharp-check/$cd; job rshm-$cd vm4 "f:$CD/$cd/sh42/decode.pt"
    FRJOBS+="rshf-$cd-s42|$cd|sharp|sharp|s42|d:rshm-$cd|4"$'\n'
  fi
  vm_cmd $cd sharp sharp s43 $dec $VMD/sharp/$cd; job rshv-$cd-s43 vm4 "f:$dec d:rshm-$cd"
  FRJOBS+="rshf-$cd-s43|$cd|sharp|sharp|s43|d:rshv-$cd-s43|4"$'\n'
done
# 4. our Q4_K against numz's Q4_K_M: the distance between their none masters (1080p)
for c in $NQ4KM_CLIPS; do
  cd=$c-d1
  CMD="mkdir -p $DFD/x && $FFD '$MD/q4k/$cd.s42.none~q4k.gbrp16le.mkv' '$MD/nq4km/$cd.s42.none~nq4km.gbrp16le.mkv' > '$DFD/x/$cd.q4k-nq4km.txt.tmp' && mv -f '$DFD/x/$cd.q4k-nq4km.txt.tmp' '$DFD/x/$cd.q4k-nq4km.txt' && cat '$DFD/x/$cd.q4k-nq4km.txt'"
  job x-q4k-nq4km-$cd df "d:q4k-vm-$cd d:nq4km-vm-$cd"
done
# 5. fr_metrics (lowest priority; 1080p only while $SC/FR_ON exists, 4K only while $SC/FR4_ON does): 1080p, then
# 4K; in each, the 7B's first, then the files' in landing order
if [ $FR = 1 ] || [ $FR4 = 1 ]; then
  while IFS='|' read -r id cd L rt ct needs res; do
    [ -n "${id:-}" ] || continue
    if [ "$res" = 4 ]; then [ $FR4 = 1 ] || continue; ctx4 ${cd%-d1}; else [ $FR = 1 ] || continue; ctx1; fi
    fr_cmd $cd $L $rt $ct; job $id fr$K4 "$needs"
  done < <(for r in 1 4; do grep "^r7f-.*|$r\$" <<< "$FRJOBS"; grep -v '^r7f-' <<< "$FRJOBS" | grep "|$r\$"; done)
fi
# 6. S16: the 3B's anime-clean runs also dump the encoder input and the reference (COLOUR_DUMP_INPUTS=1): their md5
# against colour's 7B fp16 s42 ones of the clip (same VAE, same input: expected equal); ms_inputs.sh prints and writes
# $VAL_STATE/sum/inputs-<label>-<cd>.txt and exits 0 whatever it finds (a report, never a block)
for f in "$GPUDONE"/a-anime-clean-d1-3b-*; do
  [ -e "$f" ] || continue
  L=${f##*/a-anime-clean-d1-}
  CMD="bash $VAL_GLUE/ms_inputs.sh $L anime-clean-d1"
  job x-inputs-anime-clean-d1-$L df "g:a-anime-clean-d1-$L"
done
# 7. S16: the 3B's page (pair3b.py: the 3B against the sharp 7B's 4 GB pick, the sharp and the 7B float16, at
# seed 42 on the 8 d1 clips -> sum/3b-00-overview.md + .csv) once every job of both 3B labels is done (re-made by hand
# afterwards: remake_all.sh)
if compgen -G "$GPUDONE/a-*-d1-3b-cur" > /dev/null || compgen -G "$GPUDONE/a-*-d1-3b-first" > /dev/null; then
  need3b=""
  for L in 3b-cur 3b-first; do
    for c in $CLIPS_A; do
      for k in sc vm bd; do need3b+=" d:$L-$k-$c-d1"; done
      [ $FR = 1 ] && need3b+=" d:$L-fr-$c-d1"
    done
  done
  need3b+=" d:x-inputs-anime-clean-d1-3b-cur d:x-inputs-anime-clean-d1-3b-first"
  CMD="$PY $VAL_GLUE/pair3b.py"
  job x-pair3b df "$need3b"
fi

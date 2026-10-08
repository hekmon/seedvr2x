# Model conversation, slice A scoring (night of 2026-10-07): shared settings, sourced by ms_jobs.sh, ms_pool.sh,
# ms_status.sh. Colour's tools and the 7B fp16's dumps and scores are read only; every output of ours goes to
# $VAL_DATA/gpu/{eval,vmaf,masters,bands,diff,fr}, markers and logs to $VAL_STATE/score/.
for v in VAL_GLUE VAL_STATE VAL_DATA COLOUR_OUT COLOUR_SCRIPTS COLOUR_BASELINE MEAS_CLIPS MEAS_SCRIPTS METRICS_PY FFMPEG_BIN; do [ -n "${!v:-}" ] || { echo "${0##*/}: $v is unset: source your glue.env (models/gpu/validation/glue.env.example)" >&2; exit 2; }; done
O=$COLOUR_OUT                             # colour's (read only): scripts/, dumps/, eval-b2/, vmaf-b2/, bands-b2/
CS=$COLOUR_SCRIPTS                         # colour_eval.py, colour_bands.py, colour_crops.py (called as they are)
CD=$O/dumps                                # the 7B fp16 dumps: <clip>-d1/s{42,43,1234}/decode.pt, s42/ref_f32.pt
CLIPS=$MEAS_CLIPS
SC=$VAL_STATE/score                        # our job markers (done/ failed/ running/ peak/), logs/
G=$VAL_DATA/gpu                            # big outputs
GD=$G/dumps                                # our runs' dumps (the GPU runner's): <clip>-d1/<label>-s42/decode.pt
GPUDONE=$VAL_STATE/gpu/done                # the GPU runner's markers a-<clip>-d1-<label>
GPUFAIL=$VAL_STATE/gpu/failed
EVD=$G/eval VMD=$G/vmaf MD=$G/masters BD=$G/bands DFD=$G/diff FRD=$G/fr
PY=$METRICS_PY
MEAS=$MEAS_SCRIPTS                         # fr_metrics.py, ffv1_out.py (measurement's, read only)
# colour's pool environment (b2_cpu.sh), identical, plus no GPU
export COLOUR_BASELINE MEAS_SCRIPTS=$MEAS PATH=$FFMPEG_BIN:$PATH
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TMPDIR=$G/tmp CUDA_VISIBLE_DEVICES=
CLIPS_A="anime-clean anime-grain anime-sky anime-bright cartoon-bright live-vfx live-slow anime-dark"
NQ4KM_CLIPS="anime-sky anime-bright live-vfx live-slow"
declare -A BARS=([anime-grain]=30:29 [live-slow]=130:132 [live-vfx]=140:140)   # colour's --bars (else 0:0)
declare -A ROWS=([anime-grain]=32:1048 [live-slow]=136:944 [live-vfx]=144:936) # colour's bands rows (else whole)
SP=split:ycc:4:3
T=4   # threads per job: colour's 1080p jobs' (T1=4)

#!/usr/bin/env bash
# Reproducible SeedVR2 Python environment for NVIDIA Blackwell (sm_120), with
# every attention backend SeedVR2 can use built from source as a real wheel.
# Rationale, pitfalls and verification: docs/environment.md
#
# Usage: setup_env.sh <phase>...
#   venv     create the venv (uv), install torch + SeedVR2 requirements + build deps
#   src      clone the kernel sources at pinned refs
#   sage2    build the SageAttention 2 wheel
#   sage3    build the SageAttention 3 (Blackwell FP4) wheel
#   flash2   build the FlashAttention 2 wheel
#   install  install every built wheel into the venv
#   verify   run scripts/probe_env.py against the venv
#   all      all of the above, in order
#
# Every knob below can be overridden from the environment.
# No root needed: only uv, git, a CUDA toolkit and a host compiler.
set -euo pipefail

SEEDVR2_DIR=${SEEDVR2_DIR:?set SEEDVR2_DIR to the ComfyUI-SeedVR2_VideoUpscaler checkout}
BUILD_DIR=${BUILD_DIR:-$(dirname "$SEEDVR2_DIR")/build}
VENV=${VENV:-$SEEDVR2_DIR/.venv}
PYTHON_VERSION=${PYTHON_VERSION:-3.13}

TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}
TORCH_VERSION=${TORCH_VERSION:-2.14.1}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.29.1}

CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}   # must match torch's CUDA major (cu130 -> 13.x)
ARCH=${ARCH:-12.0}                              # compute capability to build kernels for
MAX_JOBS=${MAX_JOBS:-16}                        # parallel compile jobs (each nvcc job can use several GB of RAM)
NVCC_THREADS=${NVCC_THREADS:-4}

SAGE_REPO=${SAGE_REPO:-https://github.com/thu-ml/SageAttention.git}
SAGE_REF=${SAGE_REF:-d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5}
CUTLASS_REPO=${CUTLASS_REPO:-https://github.com/NVIDIA/cutlass.git}
CUTLASS_REF=${CUTLASS_REF:-v4.8.0}              # SA3 clones cutlass HEAD if absent: pin it instead
FA_REPO=${FA_REPO:-https://github.com/Dao-AILab/flash-attention.git}
FA_REF=${FA_REF:-v2.8.3.post1}

SRC=$BUILD_DIR/src
WHEELS=$BUILD_DIR/wheels
PY=$VENV/bin/python
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# $VENV/bin on PATH matters: torch.utils.cpp_extension only uses ninja (parallel builds,
# honoring MAX_JOBS) when a `ninja` binary is on PATH, else it silently compiles serially.
export CUDA_HOME PATH="$VENV/bin:$CUDA_HOME/bin:$PATH" MAX_JOBS NVCC_THREADS
export UV_LINK_MODE=${UV_LINK_MODE:-copy}  # uv cache and venv often sit on different filesystems

log() { printf '\n=== [%s] %s\n' "$(date +%T)" "$*"; }

phase_venv() {
	log "venv: $VENV (python $PYTHON_VERSION)"
	[[ -e $VENV ]] && { echo "refusing to overwrite existing $VENV (move it away first)"; exit 1; }
	uv venv --seed --python "$PYTHON_VERSION" "$VENV"
	log "torch $TORCH_VERSION / torchvision $TORCHVISION_VERSION from $TORCH_INDEX"
	uv pip install --python "$PY" --index-url "$TORCH_INDEX" \
		"torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"
	log "SeedVR2 requirements (torch already satisfied, kept as is)"
	uv pip install --python "$PY" -r "$SEEDVR2_DIR/requirements.txt"
	log "build dependencies for --no-build-isolation wheels"
	uv pip install --python "$PY" setuptools wheel ninja packaging psutil
	"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.get_device_name(0))"
}

phase_src() {
	mkdir -p "$SRC"
	log "SageAttention @ $SAGE_REF"
	[[ -d $SRC/SageAttention ]] || git clone -q "$SAGE_REPO" "$SRC/SageAttention"
	git -C "$SRC/SageAttention" checkout -q "$SAGE_REF"
	log "cutlass @ $CUTLASS_REF (for SageAttention 3)"
	local cutlass=$SRC/SageAttention/sageattention3_blackwell/csrc/cutlass
	[[ -d $cutlass ]] || git clone -q --depth 1 --branch "$CUTLASS_REF" "$CUTLASS_REPO" "$cutlass"
	log "flash-attention @ $FA_REF"
	[[ -d $SRC/flash-attention ]] || git clone -q --depth 1 --branch "$FA_REF" "$FA_REPO" "$SRC/flash-attention"
	git -C "$SRC/flash-attention" submodule update --init --depth 1 csrc/cutlass
}

# pip wheel without build isolation: the extension must compile against the venv's torch.
# Recent torch (2.14 here) headers need C++20 and torch.utils.cpp_extension compiles with
# -std=c++20, but only when the extension does not pass its own -std. These setup.py files
# hard-code -std=c++17, which would win, so bump it (idempotent).
build_wheel() {
	mkdir -p "$WHEELS"
	sed -i 's/-std=c++17/-std=c++20/g' "$1/setup.py"
	(cd "$1" && rm -rf build ./*.egg-info && "$PY" -m pip wheel --no-build-isolation --no-deps -v -w "$WHEELS" .)
}

phase_sage2() {
	log "SageAttention 2 for sm_$ARCH"
	TORCH_CUDA_ARCH_LIST=$ARCH EXT_PARALLEL=4 build_wheel "$SRC/SageAttention"
}

phase_sage3() {
	log "SageAttention 3 (Blackwell FP4); targets the GPU visible at build time"
	build_wheel "$SRC/SageAttention/sageattention3_blackwell"
}

phase_flash2() {
	log "FlashAttention 2 for sm_${ARCH/./}"
	FLASH_ATTN_CUDA_ARCHS=${ARCH/./} FLASH_ATTENTION_FORCE_BUILD=TRUE build_wheel "$SRC/flash-attention"
}

phase_install() {
	log "installing wheels from $WHEELS"
	ls -1 "$WHEELS"
	uv pip install --python "$PY" --no-deps "$WHEELS"/*.whl
}

phase_verify() {
	log "probe"
	"$PY" "$SCRIPT_DIR/probe_env.py" "$SEEDVR2_DIR"
}

[[ $# -gt 0 ]] || { sed -n '2,16p' "$0"; exit 1; }
for p in "$@"; do
	case $p in
		all) for q in venv src sage2 sage3 flash2 install verify; do "phase_$q"; done ;;
		venv|src|sage2|sage3|flash2|install|verify) "phase_$p" ;;
		*) echo "unknown phase: $p"; exit 1 ;;
	esac
done

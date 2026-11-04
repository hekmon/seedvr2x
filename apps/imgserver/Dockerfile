# seedvr2x's image server (DESIGN.md, The image server, The Docker image).
#
# Build from the repository's root, which keeps the layout the server's lock expects (it installs
# core/ editable from ../../core):
#
#   docker build -f apps/imgserver/Dockerfile -t seedvr2x-imgserver .
#
# apps/imgserver/Dockerfile.dockerignore says what the build context holds.

# python:3.13-slim and uv 0.12.21, each by digest (taken from the registry, 2026-11-03): nothing
# CUDA in the base, the host gives the NVIDIA driver and the Container Toolkit, and PyTorch's
# wheels bring CUDA's libraries. Bump both on purpose.
FROM ghcr.io/astral-sh/uv:0.12.21@sha256:a7aed3216253ee804de3e2d8afa5073baa1a177335345d43845cd4165e43b711 AS uv

FROM python:3.13-slim@sha256:ae14d4ce8771606477a81446e7ef429edd6762305fc50c84507c1041fa8285d1

COPY --from=uv /uv /usr/local/bin/uv

# uv uses this image's Python and never downloads one; the packages are copied, not linked (the
# cache is not kept in the image); their bytecode is compiled at build, the user below can't
# write it.
ENV UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON_PREFERENCE=only-system \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1

WORKDIR /app/apps/imgserver

# The dependencies first, from the lock alone: this layer (torch's CUDA wheels, several GB) is
# rebuilt only when the lock changes, not when a module of the server or of the library does.
# seedvr2x-core is a path dependency, installed with the sources below.
COPY apps/imgserver/pyproject.toml apps/imgserver/uv.lock apps/imgserver/.python-version ./
RUN uv sync --frozen --no-dev --no-install-project --no-install-package seedvr2x-core

# The library, then the server, installed editable from where they sit (the lock's path source is
# ../../core). The files in the build context are the ignore file's.
COPY core /app/core
COPY apps/imgserver/LICENSE apps/imgserver/NOTICE apps/imgserver/README.md ./
COPY apps/imgserver/*.py ./
RUN uv sync --frozen --no-dev

# A user other than root, owning its home and the cache directory alone. The models stay outside
# the image: Hugging Face's cache is a volume (the models are pulled into it at the first start,
# at the pinned revision), or a model directory is mounted read-only and given with --model-dir.
# HF_HUB_OFFLINE=1 forbids the pull, for a closed network.
RUN useradd --uid 10001 --user-group --create-home --home-dir /home/seedvr2x --shell /usr/sbin/nologin seedvr2x \
    && mkdir -p /data/huggingface \
    && chown seedvr2x:seedvr2x /data/huggingface
USER seedvr2x

ENV PATH="/app/apps/imgserver/.venv/bin:$PATH" \
    HOME=/home/seedvr2x \
    HF_HOME=/data/huggingface \
    PYTHONUNBUFFERED=1 \
    SEEDVR2X_LISTEN=0.0.0.0:8390

# The caches the server writes, where any user can: a container run with --user (the host's
# Hugging Face cache mounted writable, the README) is not the owner of HOME, and torch's kernel
# cache under it could not be created, the caching off (an RTX 5090, 2026-11-04, --user
# 1000:1000: "Specified kernel cache directory could not be created!",
# /home/seedvr2x/.cache/torch/kernels). torch takes XDG_CACHE_HOME before HOME
# (aten/src/ATen/native/cuda/jit_utils.cpp) and makes torch/kernels under it, but not the
# directory itself: /tmp/cache, missing, failed the same way for every user (the same night), so
# it is /tmp, which the base image has (mode 1777), a tmpfs mounted there too. Triton reads its
# own variable, not XDG's.
ENV XDG_CACHE_HOME=/tmp \
    TRITON_CACHE_DIR=/tmp/triton

VOLUME /data/huggingface
EXPOSE 8390

# /health answers 200 once the warm-up has passed and 503 before (starting) and after (stopping,
# failed). The start takes minutes: the model files' check reads the DiT's 16.5 GB (about 84 s on
# an RTX 5090's disk) and loading took 77 s there (DESIGN.md, Still images), so the start period
# is 10 minutes. Python's urllib: the image has no curl. The port is the one SEEDVR2X_LISTEN
# gives (the part after its last colon, as --listen reads it), on the container's own 127.0.0.1,
# which a server listening on 0.0.0.0 answers: a container given another port stays healthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; port = os.environ.get('SEEDVR2X_LISTEN', ':8390').rpartition(':')[2]; urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=4)"]

# Listening on all interfaces, which port publishing needs, through SEEDVR2X_LISTEN above; publish
# the port on 127.0.0.1 unless a reverse proxy is in front (the README). The ENTRYPOINT is the bare
# command and CMD empty: options given to `docker run IMAGE --model-dir /models` append to the
# ENTRYPOINT. An address in CMD would be dropped by any option given, since options replace CMD,
# and the server would then listen on the container's own 127.0.0.1, which port publishing can't
# reach. Another address: -e SEEDVR2X_LISTEN=0.0.0.0:9000 (or --listen after the image's name,
# which wins over the variable).
ENTRYPOINT ["seedvr2x-imgserver"]
CMD []

LABEL org.opencontainers.image.title="seedvr2x-imgserver" \
      org.opencontainers.image.description="seedvr2x's image server: SeedVR2 picture upscaling over HTTP, the models loaded once and kept on the GPU" \
      org.opencontainers.image.source="https://github.com/hekmon/seedvr2x" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT"

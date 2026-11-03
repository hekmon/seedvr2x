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
    PYTHONUNBUFFERED=1

VOLUME /data/huggingface
EXPOSE 8390

# /health answers 200 once the warm-up has passed and 503 before (starting) and after (stopping,
# failed). The start takes minutes: the model files' check reads the DiT's 16.5 GB (about 84 s on
# an RTX 5090's disk) and loading took 77 s there (DESIGN.md, Still images), so the start period
# is 10 minutes. Python's urllib: the image has no curl. A server given another port with
# --listen needs its own check.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8390/health', timeout=4)"]

# Listening on all interfaces, which port publishing needs; publish the port on 127.0.0.1 unless a
# reverse proxy is in front (the README). Options append: docker run IMAGE --model-dir /models.
ENTRYPOINT ["seedvr2x-imgserver", "--listen", "0.0.0.0:8390"]
CMD []

LABEL org.opencontainers.image.title="seedvr2x-imgserver" \
      org.opencontainers.image.description="seedvr2x's image server: SeedVR2 picture upscaling over HTTP, the models loaded once and kept on the GPU" \
      org.opencontainers.image.source="https://github.com/hekmon/seedvr2x" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT"

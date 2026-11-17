# NorthRL Multi-Stage Dockerfile
# Based on NVIDIA NGC PyTorch image

# Stage 1: Builder - Build wheel package
ARG NGC_IMAGE_TAG=25.02-py3

FROM nvcr.io/nvidia/pytorch:${NGC_IMAGE_TAG} AS builder

ARG MAX_JOBS=16

LABEL stage="builder"
LABEL description="NorthRL Builder Stage - Build wheel package"

WORKDIR /build

# Set MAX_JOBS for parallel compilation
ENV MAX_JOBS=${MAX_JOBS}

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel build

# Copy project files
COPY pyproject.toml /build/
COPY README.md /build/
COPY northrl/ /build/northrl/

# Build wheel package
RUN python -m build --wheel --outdir /build/dist

# Verify wheel package
RUN ls -lh /build/dist/*.whl

# Stage 2: Runtime - Production Image
ARG NGC_IMAGE_TAG=25.02-py3

FROM nvcr.io/nvidia/pytorch:${NGC_IMAGE_TAG} AS runtime

# Pass build arguments to runtime environment variables
ARG NGC_IMAGE_TAG
ARG MAX_JOBS=16
ARG NORTHRL_EXTRAS=docker
ARG SGLANG_VERSION=0.5.8
ARG NORTHRL_VERSION=dev
ENV NGC_IMAGE_TAG=${NGC_IMAGE_TAG}
ENV MAX_JOBS=${MAX_JOBS}
ENV NORTHRL_VERSION=${NORTHRL_VERSION}

# Metadata
LABEL maintainer="NexAGI Team"
LABEL description="NorthRL Universal Image - Supports Controller, Workers, and Train Router"
LABEL version="${NORTHRL_VERSION}"

# Set working directory
WORKDIR /app

# Install additional system dependencies (NGC image already includes most tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    tmux \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and basic tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Pre-remove NGC-shipped torch family before installing the northrl wheel.
# northrl pins torch==2.7.1 while NGC ships torch 2.7.0a0+*.nv*; pip would
# otherwise uninstall the NGC torch during conflict resolution, which
# copies ~6GB of CUDA .so files to /tmp/pip-uninstall-* and overflows the
# runner disk on GitHub-hosted runners. Removing them in their own layer
# avoids the temp-copy and lowers peak disk during the install step.
RUN pip uninstall -y torch torchvision torchaudio || true

# Copy wheel package from builder stage
COPY --from=builder /build/dist/*.whl /tmp/

# Install NorthRL wheel package with configurable optional dependencies
# Default extra is [docker]; sglang is installed separately to avoid conflicting dependencies.
RUN WHEEL_FILE=$(ls /tmp/*.whl) && \
    pip install --no-cache-dir "${WHEEL_FILE}[${NORTHRL_EXTRAS}]" && \
    rm -rf /tmp/*.whl

# Prepare IFEval tokenizer data in the image for all runtime users.
ARG NLTK_DATA_BASE_URL=
ENV NLTK_DATA=/usr/local/share/nltk_data
COPY docker/prepare_nltk_data.sh /tmp/prepare_nltk_data.sh
RUN --mount=type=cache,target=/root/.cache/northrl-nltk,sharing=locked \
    NLTK_DATA_BASE_URL="$NLTK_DATA_BASE_URL" bash /tmp/prepare_nltk_data.sh && \
    rm /tmp/prepare_nltk_data.sh

# Install sglang without dependencies to bypass openai/torch/grpcio version pin conflicts.
# Then install only the lightweight utility packages that sglang's tool/reasoning parsers
# actually need at runtime.
RUN pip install --no-cache-dir --no-deps "sglang==${SGLANG_VERSION}" && \
    pip install --no-cache-dir pybase64 partial-json-parser orjson


RUN pip uninstall -y flash-attn && \
pip install --no-cache-dir flash-attn --no-build-isolation

# Set Python environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

RUN python -m northrl.main --version

# Create necessary directories
RUN mkdir -p /app/data /app/checkpoints /app/logs

# Use tini as the init system to handle signals properly
ENTRYPOINT ["/usr/bin/tini", "--"]

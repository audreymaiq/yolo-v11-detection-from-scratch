# Training / inference image with the exact dependency versions of uv.lock.
#
#   docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t yolo11 .
#   docker run --gpus all --shm-size=8g -v "$PWD/datasets:/app/datasets" -v "$PWD/runs:/app/runs" \
#       yolo11 train --config configs/training.yaml
#
# Linux torch wheels from PyPI bundle the CUDA runtime, so the host needs only the NVIDIA driver and the NVIDIA
# Container Toolkit (without --gpus the image runs on CPU). --shm-size: dataloader workers share batches through
# /dev/shm, which Docker limits to 64 MB by default.

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

# OpenCV (opencv-python) needs these shared libraries at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

WORKDIR /app

# Extras to install, e.g. --build-arg EXTRAS="--extra export --extra tracking"
ARG EXTRAS="--extra export"

# Dependencies first: this layer is rebuilt only when the lock changes, not on every source edit.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project ${EXTRAS}

COPY src ./src
COPY configs ./configs
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable ${EXTRAS}

# There is no .git in the image: provenance.json records this commit instead (see yolo11/utils/provenance.py).
ARG GIT_COMMIT=""
ENV YOLO11_GIT_COMMIT=${GIT_COMMIT}

ENTRYPOINT ["yolo11"]
CMD ["--help"]

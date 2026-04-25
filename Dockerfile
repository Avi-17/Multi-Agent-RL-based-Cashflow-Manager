# Use a specific version of the base image for reproducibility
ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

# 1. Install uv binary from official image (much faster than curl)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app/env

# 2. Install system dependencies (only if needed for building)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# 3. Optimized dependency installation
# Use mounts to avoid copying files until necessary and utilize persistent cache
# --no-install-project avoids building the project itself (which fails without source code)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-install-project --no-dev

# 4. Copy the rest of the application
COPY . /app/env

# 5. Final sync to install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# =========================
# Runtime Stage
# =========================
FROM ${BASE_IMAGE}

WORKDIR /app/env

# Copy the virtual environment and application code from the builder
COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env

# Set environment variables in a single layer
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/env:$PYTHONPATH" \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME="0.0.0.0"

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s \
  CMD curl -f http://localhost:7860 || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]

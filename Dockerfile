# # =========================
# # Base Image
# # =========================
# ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
# FROM ${BASE_IMAGE} AS builder

# WORKDIR /app

# # Install git (needed for dependencies)
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends git curl && \
#     rm -rf /var/lib/apt/lists/*

# # Copy project
# COPY . /app/env
# WORKDIR /app/env

# # Ensure uv exists
# RUN if ! command -v uv >/dev/null 2>&1; then \
#         curl -LsSf https://astral.sh/uv/install.sh | sh && \
#         mv /root/.local/bin/uv /usr/local/bin/uv && \
#         mv /root/.local/bin/uvx /usr/local/bin/uvx; \
#     fi

# # Install dependencies
# RUN --mount=type=cache,target=/root/.cache/uv \
#     if [ -f uv.lock ]; then \
#         uv sync --frozen --no-editable; \
#     else \
#         uv sync --no-editable; \
#     fi


# # =========================
# # Runtime
# # =========================
# FROM ${BASE_IMAGE}

# WORKDIR /app/env

# # Copy virtual env
# COPY --from=builder /app/env/.venv /app/.venv

# # Copy code
# COPY --from=builder /app/env /app/env

# # Activate venv
# ENV PATH="/app/.venv/bin:$PATH"
# ENV PYTHONPATH="/app/env:$PYTHONPATH"

# # IMPORTANT: Gradio port
# EXPOSE 7860

# # Optional lightweight healthcheck
# HEALTHCHECK --interval=30s --timeout=5s \
#   CMD curl -f http://localhost:7860 || exit 1

# # Run Gradio app
# # CMD ["python", "server/app.py"]
# CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]


# =========================
# Base Image
# =========================
ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv once
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# 🔥 COPY ONLY dependency files first (cache optimization)
COPY pyproject.toml uv.lock* /app/env/
WORKDIR /app/env

# Install deps (cached unless deps change)
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

# 🔥 NOW copy rest of code (doesn't break cache)
COPY . /app/env


# =========================
# Runtime
# =========================
FROM ${BASE_IMAGE}

WORKDIR /app/env

COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s \
  CMD curl -f http://localhost:7860 || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
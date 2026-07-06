FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/app/models

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv git build-essential cmake curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

COPY configs ./configs
COPY scripts ./scripts

ENTRYPOINT ["uv", "run", "ft"]
CMD ["--help"]

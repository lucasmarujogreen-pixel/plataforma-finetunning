#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-runs/2026-07-04_13-47-25_qwen3-0.6b_qwen3-0.6b-condicoes-v6}"
MODEL="${2:-claude-haiku-4-5-20251001}"
ENV_FILE="/home/lucas/projetos/motor-ia/.env"

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  ANTHROPIC_API_KEY="$(grep '^AnthropicConfiguration__ApiKey__Agent=' "$ENV_FILE" | cut -d= -f2- | tr -d '"' | tr -d "'")"
  export ANTHROPIC_API_KEY
fi

if [[ -z "$ANTHROPIC_API_KEY" ]]; then
  echo "ANTHROPIC_API_KEY nao encontrada (env ou $ENV_FILE)" >&2
  exit 1
fi

exec uv run --with anthropic python scripts/evaluate_poc_api.py \
  --run "$RUN_DIR" \
  --limit 200 \
  --model "$MODEL"

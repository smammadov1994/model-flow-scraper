#!/usr/bin/env bash
# One-command setup: Python env, Qdrant, then download + enrich + index (~30-45 min the first time).
set -euo pipefail
cd "$(dirname "$0")"

command -v uv >/dev/null || { echo "Install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
command -v docker >/dev/null || { echo "Install Docker Desktop first."; exit 1; }

[ -d .venv ] || uv venv -q --python 3.12
uv pip install -q -e '.[dev]'

docker compose up -d qdrant
for i in $(seq 1 30); do curl -sf http://localhost:6333 >/dev/null && break; sleep 1; done

if command -v ollama >/dev/null; then
  ollama list | grep -q '^qwen3:4b' || ollama pull qwen3:4b
else
  echo "Ollama not found: questions will be understood with keyword rules only (install Ollama for better parsing)."
fi

.venv/bin/model-finder setup "$@"
echo
echo "Done. Ask:   .venv/bin/model-finder ask \"voice model on my phone\""
echo "Web page:    .venv/bin/model-finder serve   ->  http://localhost:8765"

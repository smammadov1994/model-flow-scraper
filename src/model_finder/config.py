"""Paths, thresholds and service settings. Everything can be overridden with MF_* env vars."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("MF_HOME", Path(__file__).resolve().parents[2]))


def _load_env_file(path: Path) -> None:
    """Read KEY=VALUE lines from .env (gitignored) without overriding variables already set."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip().removeprefix("export ").strip(), v.strip().strip("'\""))


_load_env_file(ROOT / ".env")
DATA = ROOT / "data"
RAW = DATA / "raw"
PARQUET = RAW / "models.parquet"
CATALOG_DB = DATA / "catalog.db"

HUB_STATS_URL = "https://huggingface.co/datasets/cfahlgren1/hub-stats/resolve/main/models.parquet"

QDRANT_URL = os.environ.get("MF_QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("MF_COLLECTION", "models")
# Qwen3-Embedding-0.6B won a 7-model benchmark on our own data (research/embedding-bench). Runs on the Mac GPU.
# MF_DENSE_MODEL=BAAI/bge-small-en-v1.5 is ~25x faster to build (CPU, fastembed) and somewhat less accurate.
DENSE_MODEL = os.environ.get("MF_DENSE_MODEL", "Qwen/Qwen3-Embedding-0.6B")
DENSE_DEVICE = os.environ.get("MF_DENSE_DEVICE")  # mps / cuda / cpu; auto when unset
BM25_MODEL = "qdrant/bm25"  # computed inside Qdrant (server-side inference)
DENSE_ALL = os.environ.get("MF_DENSE_ALL", "0") == "1"  # 1 = semantic vectors for every model on first build

# Which models are worth indexing. A model is kept if it is popular OR recent-and-noticed.
MIN_DOWNLOADS = int(os.environ.get("MF_MIN_DOWNLOADS", 250))  # downloads in the last 30 days
MIN_LIKES = int(os.environ.get("MF_MIN_LIKES", 10))
RECENT_DAYS = int(os.environ.get("MF_RECENT_DAYS", 90))
RECENT_MIN_DOWNLOADS = int(os.environ.get("MF_RECENT_MIN_DOWNLOADS", 25))
RECENT_MIN_LIKES = int(os.environ.get("MF_RECENT_MIN_LIKES", 2))
NOTABLE_AUTHOR_DOWNLOADS = 50_000  # authors with any model this popular get their brand-new models indexed

# LLM used for query understanding and card extraction. "auto" picks:
# ANTHROPIC_API_KEY -> anthropic, else a running Ollama, else rule-based only.
LLM = os.environ.get("MF_LLM", "auto")
OLLAMA_URL = os.environ.get("MF_OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODELS = [m for m in os.environ.get("MF_OLLAMA_MODEL", "qwen3:4b,qwen3.5:2b,llama3.2:3b,gemma4:12b").split(",") if m]
ANTHROPIC_MODEL = os.environ.get("MF_ANTHROPIC_MODEL", "claude-haiku-5-5")

# Question understanding + re-ranking: "auto" uses TypeSafe Jev when TYPESAFE_API_KEY is set, else the LLM above.
# "jev" / "llm" / "rules" force one.
PARSER = os.environ.get("MF_PARSER", "auto")
JEV_MODEL = os.environ.get("MF_JEV_MODEL", "jev-latest")
JEV_TIMEOUT = float(os.environ.get("MF_JEV_TIMEOUT", 10))
RERANK_N = int(os.environ.get("MF_RERANK_N", 30))  # shortlist size Jev re-scores (one parallel question each)

# Model Studio planner: "auto" = Anthropic API if a key is set, else the Claude Code CLI if logged in, else templates.
STUDIO_PLANNER = os.environ.get("MF_STUDIO_PLANNER", "auto")
STUDIO_MODEL = os.environ.get("MF_STUDIO_MODEL", "claude-opus-5-5")
STUDIO_CLI_MODEL = os.environ.get("MF_STUDIO_CLI_MODEL", "opus")

WEB_PORT = int(os.environ.get("MF_PORT", 8765))

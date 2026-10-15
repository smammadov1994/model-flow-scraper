"""Tiny LLM wrapper with three backends and a no-LLM fallback.

MF_LLM=auto (default) -> anthropic if ANTHROPIC_API_KEY is set, else Ollama if running, else none.
MF_LLM=ollama:qwen3:4b | anthropic:claude-haiku-5-5 | claude-cli:haiku | none
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from functools import lru_cache

import httpx

from . import config


class LLMUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def backend() -> tuple[str, str | None]:
    spec = config.LLM
    if spec != "auto":
        kind, _, model = spec.partition(":")
        return kind, model or None
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", config.ANTHROPIC_MODEL
    try:
        tags = httpx.get(f"{config.OLLAMA_URL}/api/tags", timeout=1.5).json().get("models", [])
        have = {m["name"] for m in tags}
        for m in config.OLLAMA_MODELS:
            if m in have:
                return "ollama", m
    except (httpx.HTTPError, ValueError):
        pass
    return "none", None


def name() -> str:
    kind, model = backend()
    return f"{kind}:{model}" if model else kind


def available() -> bool:
    return backend()[0] != "none"


def _extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError(f"no JSON in LLM output: {text[:200]!r}")
    return json.loads(m.group(0))


def complete(system: str, user: str, schema: dict | None = None, max_tokens: int = 600, timeout: float = 60) -> str:
    kind, model = backend()
    if kind == "ollama":
        body = {
            "model": model, "stream": False, "think": False, "keep_alive": "30m",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": 8192},
        }
        if schema:
            body["format"] = schema
        r = httpx.post(f"{config.OLLAMA_URL}/api/chat", json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()["message"]["content"]
    if kind == "anthropic":
        base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
        sys_prompt = system + ("\nReply with one JSON object only, matching this JSON schema:\n" + json.dumps(schema) if schema else "")
        r = httpx.post(
            f"{base}/v1/messages",
            headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
            json={"model": model, "max_tokens": max_tokens, "temperature": 0, "system": sys_prompt,
                  "messages": [{"role": "user", "content": user}]},
            timeout=timeout,
        )
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"])
    if kind == "claude-cli":
        if not shutil.which("claude"):
            raise LLMUnavailable("claude CLI not found")
        prompt = system + ("\nReply with one JSON object only, matching this JSON schema:\n" + json.dumps(schema) if schema else "") + "\n\n" + user
        out = subprocess.run(["claude", "-p", "--model", model or "haiku", "--output-format", "json", prompt],
                             capture_output=True, text=True, timeout=timeout * 3)
        return json.loads(out.stdout).get("result", "")
    raise LLMUnavailable("no LLM backend configured (install Ollama, or set ANTHROPIC_API_KEY)")


def complete_json(system: str, user: str, schema: dict, **kw) -> dict:
    return _extract_json(complete(system, user, schema=schema, **kw))

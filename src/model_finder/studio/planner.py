"""Turns a conversation into a pipeline *plan*: steps (with HF task ids), data-flow edges, device.

Claude does the planning (it is good at decomposing "a voice assistant on my iPhone" into listen → think →
speak and at revising a plan from follow-up messages). Backends, first available wins:
  1. Anthropic API via the official SDK (ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN, e.g. in .env)
  2. the Claude Code CLI (`claude -p`, your subscription; needs `claude /login`)
  3. no Claude: Jev picks one of a few ready-made pipeline shapes (templates.py)
Model choice is NOT done here — resolve.py does that with Model Finder + Jev.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

from .. import config
from ..summarize import TASK_PLAIN

DEVICE_KINDS = ["phone", "mac", "laptop", "gpu", "server", "any"]
NODE_TYPES = ["input", "model", "logic", "output"]

PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["reply", "title", "device", "steps", "edges"],
    "properties": {
        "reply": {"type": "string", "description": "2-4 plain sentences to the user: what you built/changed and why. No markdown."},
        "title": {"type": "string", "description": "Short name for the pipeline (max 6 words)."},
        "device": {
            "type": "object", "additionalProperties": False, "required": ["kind", "label", "memory_gb"],
            "properties": {
                "kind": {"type": "string", "enum": DEVICE_KINDS},
                "label": {"type": "string", "description": "e.g. 'iPhone', 'MacBook Pro', 'RTX 4060 (8 GB)', 'any'"},
                "memory_gb": {"type": ["number", "null"], "description": "RAM/VRAM the user stated, else null"},
            },
        },
        "steps": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["key", "type", "title", "step", "task", "inputs", "outputs", "search_hint", "constraints"],
                "properties": {
                    "key": {"type": "string", "description": "stable short id, e.g. 'stt'. Keep keys unchanged when revising."},
                    "type": {"type": "string", "enum": NODE_TYPES},
                    "title": {"type": "string", "description": "2-4 words, e.g. 'Speech to text'"},
                    "step": {"type": "string", "description": "one plain sentence: what this step does in this pipeline"},
                    "task": {"type": ["string", "null"], "enum": sorted(TASK_PLAIN) + [None],
                             "description": "Hugging Face task id, only for type=model"},
                    "inputs": {"type": "array", "items": {"type": "string"}},
                    "outputs": {"type": "array", "items": {"type": "string"}},
                    "search_hint": {"type": "string", "description": "what an ideal model for this step does, in plain words, for search"},
                    "constraints": {
                        "type": "object", "additionalProperties": False,
                        "required": ["max_params_b", "languages", "license_open", "prefer", "pin_model"],
                        "properties": {
                            "max_params_b": {"type": ["number", "null"]},
                            "languages": {"type": "array", "items": {"type": "string"}},
                            "license_open": {"type": ["boolean", "null"]},
                            "prefer": {"type": ["string", "null"], "enum": ["smallest", "fastest", "best_quality", "most_popular", None]},
                            "pin_model": {"type": ["string", "null"], "description": "exact HF repo id if the user asked for a specific model"},
                        },
                    },
                },
            },
        },
        "edges": {
            "type": "array",
            "items": {"type": "object", "additionalProperties": False, "required": ["from", "to", "label"],
                      "properties": {"from": {"type": "string"}, "to": {"type": "string"}, "label": {"type": "string"}}},
        },
    },
}

SYSTEM = """You design AI pipelines made of open models from Hugging Face, for a visual workflow editor.
Given what the user wants to build (and the current pipeline, if any), return the full revised plan:
- Steps are nodes. Use type "input" for sources (microphone, camera, file, text box), "model" for steps an AI
  model performs (give its Hugging Face task id), "logic" for plain code (chunking, routing, a vector index,
  tool calls) and "output" for sinks (speaker, screen, file, API response).
- Keep it minimal: the fewest model steps that do the job well. One model per model step.
- Edges carry data between steps; label them with the data type (audio, text, image, embeddings, json).
- Device: where it must run. "phone" = iPhone/Android, "mac" = Mac/MacBook, "laptop" = other laptop/PC,
  "gpu" = desktop graphics card. Record memory only if the user states a number.
- When revising, keep existing step keys for steps that stay, change only what the user asked for, and put a
  model id in pin_model only if the user names a specific model.
- reply: talk to a non-technical person in 2-4 short sentences: what the pipeline does and what changed.
  Do not name specific models in the reply (they are chosen afterwards by search)."""


def _anthropic_ready() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def backend() -> str:
    if config.STUDIO_PLANNER in ("anthropic", "cli", "templates"):
        return config.STUDIO_PLANNER
    if _anthropic_ready():
        return "anthropic"
    if shutil.which("claude") and _cli_logged_in():
        return "cli"
    return "templates"


_cli_ok: bool | None = None


def _cli_logged_in() -> bool:
    """`claude auth status` is cheap; cache the answer for the process."""
    global _cli_ok
    if _cli_ok is None:
        try:
            out = subprocess.run(["claude", "auth", "status"], capture_output=True, text=True, timeout=15)
            _cli_ok = out.returncode == 0 and bool(json.loads(out.stdout or "{}").get("loggedIn"))
        except (OSError, subprocess.TimeoutExpired, ValueError):
            _cli_ok = False
    return _cli_ok


def _prompt(history: list[dict], message: str, pipeline: dict | None) -> str:
    convo = "\n".join(f"{m['role'].upper()}: {m['text']}" for m in history[-12:])
    current = _pipeline_brief(pipeline) if pipeline else "(none yet)"
    return (f"Conversation so far:\n{convo or '(new conversation)'}\n\nCurrent pipeline:\n{current}\n\n"
            f"New user message:\n{message}\n\nReturn the full revised plan as JSON.")


def _pipeline_brief(p: dict) -> str:
    lines = [f"title: {p.get('title')}", f"device: {json.dumps(p.get('device'))}"]
    for n in p.get("nodes", []):
        m = (n.get("model") or {}).get("id")
        lines.append(f"- key={n['id']} type={n['type']} title={n['title']!r} task={n.get('task')} model={m}")
    for e in p.get("edges", []):
        lines.append(f"  edge {e['source']} -> {e['target']} ({e.get('label')})")
    return "\n".join(lines)


def _plan_anthropic(prompt: str) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    resp = client.beta.messages.create(
        model=config.STUDIO_MODEL,
        max_tokens=16000,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": PLAN_SCHEMA}},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",  # re-run on a fallback model if a safety classifier declines
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("Claude declined to plan this request.")
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text)


def _plan_cli(prompt: str) -> dict:
    out = subprocess.run(
        ["claude", "-p", "--model", config.STUDIO_CLI_MODEL, "--output-format", "json", "--no-session-persistence",
         "--tools", "", "--system-prompt", SYSTEM, "--json-schema", json.dumps(PLAN_SCHEMA), prompt],
        capture_output=True, text=True, timeout=300,
    )
    data = json.loads(out.stdout or "{}")
    if data.get("is_error") or not data.get("structured_output"):
        raise RuntimeError(f"claude CLI failed: {data.get('result') or out.stderr[:300]}")
    return data["structured_output"]


def plan(history: list[dict], message: str, pipeline: dict | None) -> tuple[dict, str]:
    """Returns (plan, planner_name). Falls back down the chain on errors."""
    kind = backend()
    prompt = _prompt(history, message, pipeline)
    errors = []
    if kind == "anthropic":
        try:
            return _plan_anthropic(prompt), f"claude-api:{config.STUDIO_MODEL}"
        except Exception as e:  # noqa: BLE001 — any failure falls through to the next planner
            errors.append(f"API: {e}")
    if kind in ("anthropic", "cli") and shutil.which("claude") and _cli_logged_in():
        try:
            return _plan_cli(prompt), f"claude-cli:{config.STUDIO_CLI_MODEL}"
        except Exception as e:  # noqa: BLE001
            errors.append(f"CLI: {e}")
    from . import templates

    p = templates.plan(message, pipeline)
    if errors:
        p["reply"] += " (Claude wasn't reachable, so I used a ready-made layout: " + "; ".join(errors)[:200] + ")"
    return p, "templates+jev"

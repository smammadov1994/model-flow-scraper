"""Pipeline -> n8n workflow JSON (importable via n8n's "Import from file").

Format per docs/n8n-study.md: top-level {name, nodes, connections, settings, pinData, meta}; nodes have a uuid
`id`, a unique `name`, `type`, `typeVersion`, `position: [x, y]` and `parameters`; `connections` is keyed by the
*source node name*. Each pipeline step becomes one n8n node:
  - a chat LLM available as GGUF -> HTTP Request to a local Ollama server (`hf.co/<repo>:<quant>`), runnable as is
  - every other step -> a Code node that documents the chosen model and passes data through (a stub to fill in)
plus a sticky note per model with how to run it, and a Manual Trigger feeding the inputs.
"""
from __future__ import annotations

import json
import uuid

from .. import config
from ..summarize import task_label

TRIGGER = ("n8n-nodes-base.manualTrigger", 1)
CODE = ("n8n-nodes-base.code", 2)
HTTP = ("n8n-nodes-base.httpRequest", 4.2)
STICKY = ("n8n-nodes-base.stickyNote", 1)
NS = uuid.UUID("0f9b8e2a-4c1d-4e57-9a51-7d2c3b6e8f10")


def _uid(*parts: str) -> str:
    return str(uuid.uuid5(NS, "/".join(parts)))


def _unique(name: str, used: set[str]) -> str:
    base, i = name[:60], 2
    while name in used:
        name = f"{base} {i}"
        i += 1
    used.add(name)
    return name


def _code_for(n: dict) -> str:
    m = n.get("model") or {}
    lines = [f"// {n['title']}: {n.get('step') or ''}".rstrip(": ")]
    if m:
        lines += [f"// Model: {m['id']} ({m.get('url')})",
                  f"// Build: {m.get('variant_repo') or m['id']}" + (f" [{m.get('variant')}]" if m.get("variant") else ""),
                  f"// Memory: ~{m.get('ram_gb')} GB" if m.get("ram_gb") else "// Memory: unknown",
                  f"// How to run: {m.get('how_to_run') or 'see the model card'}",
                  "// TODO: call your local runtime for this model here (HTTP request, CLI, or a sidecar service)."]
    elif n["type"] == "input":
        lines.append("// Replace this sample with your real input (webhook, file, form, ...).")
        return "\n".join(lines) + f"\nreturn [{{ json: {{ input: {json.dumps('sample ' + (n['outputs'] or ['data'])[0])} }} }}];"
    lines.append("// Passes data through unchanged for now.")
    return "\n".join(lines) + "\nreturn $input.all();"


def _is_hosted_chat(n: dict) -> bool:
    m = n.get("model") or {}
    return m.get("hosting") == "hosted_api" and n.get("task") in ("text-generation", "image-text-to-text")


def _is_ollama_llm(n: dict) -> bool:
    m = n.get("model") or {}
    return n.get("task") in ("text-generation", "image-text-to-text") and m.get("format") == "gguf" and bool(m.get("variant"))


def export(pipeline: dict) -> dict:
    used: set[str] = set()
    names: dict[str, str] = {}
    nodes: list[dict] = []
    pid = pipeline.get("id") or "pipeline"

    trigger_name = _unique("When clicking 'Test workflow'", used)
    nodes.append({"id": _uid(pid, "trigger"), "name": trigger_name, "type": TRIGGER[0], "typeVersion": TRIGGER[1],
                  "position": [-300, 200], "parameters": {}})
    for n in pipeline["nodes"]:
        pos = n.get("position") or {"x": 0, "y": 0}
        xy = [int(pos["x"]), int(pos["y"])]
        label = n["title"] if n["type"] != "model" else f"{n['title']} ({(n.get('model') or {}).get('id', 'no model')})"
        name = _unique(label, used)
        names[n["id"]] = name
        if _is_hosted_chat(n):
            m = n["model"]
            # OpenAI-compatible chat completions through Hugging Face Inference Providers (auto-routes to a provider)
            body = (f"={{{{ JSON.stringify({{ model: {json.dumps(m['id'])}, messages: [{{ role: 'user', "
                    "content: $json.text || $json.input || '' }] }) }}")
            nodes.append({"id": _uid(pid, n["id"]), "name": name, "type": HTTP[0], "typeVersion": HTTP[1], "position": xy,
                          "parameters": {"method": "POST", "url": "https://router.huggingface.co/v1/chat/completions",
                                         "sendHeaders": True,
                                         "headerParameters": {"parameters": [
                                             {"name": "Authorization", "value": "Bearer PASTE_YOUR_HF_TOKEN"}]},
                                         "sendBody": True, "contentType": "json", "specifyBody": "json",
                                         "jsonBody": body, "options": {}}})
        elif _is_ollama_llm(n):
            m = n["model"]
            repo = m.get("variant_repo") or m["id"]
            # the whole body must be one expression (a parameter string starting with "=" is evaluated);
            # {{ }} inside a plain JSON string would be sent literally
            ollama_model = json.dumps(f"hf.co/{repo}:{m['variant']}")
            body = f"={{{{ JSON.stringify({{ model: {ollama_model}, prompt: $json.text || $json.input || '', stream: false }}) }}}}"
            nodes.append({"id": _uid(pid, n["id"]), "name": name, "type": HTTP[0], "typeVersion": HTTP[1], "position": xy,
                          "parameters": {"method": "POST", "url": f"{config.N8N_OLLAMA_URL.rstrip('/')}/api/generate", "sendBody": True,
                                         "contentType": "json", "specifyBody": "json", "jsonBody": body, "options": {}}})
        else:
            nodes.append({"id": _uid(pid, n["id"]), "name": name, "type": CODE[0], "typeVersion": CODE[1], "position": xy,
                          "parameters": {"mode": "runOnceForAllItems", "language": "javaScript", "jsCode": _code_for(n)}})
        if n.get("model"):
            m = n["model"]
            pull = ("\n\nBefore running: replace PASTE_YOUR_HF_TOKEN in the node's Authorization header with a "
                    "Hugging Face token (better: an n8n Header Auth credential)." if _is_hosted_chat(n) else
                    f"\n\nBefore running: `ollama pull hf.co/{m.get('variant_repo') or m['id']}:{m['variant']}`"
                    if _is_ollama_llm(n) else "")
            note = (f"## {task_label(n.get('task'))}\n**[{m['id']}]({m.get('url')})**\n\n"
                    + (f"~{m['ram_gb']} GB · " if m.get("ram_gb") else "") + f"{m.get('license') or 'license: see card'}\n\n"
                    + (m.get("how_to_run") or "") + pull)
            nodes.append({"id": _uid(pid, n["id"], "note"), "name": _unique(f"Note: {n['title']}", used),
                          "type": STICKY[0], "typeVersion": STICKY[1], "position": [xy[0] - 20, xy[1] + 140],
                          "parameters": {"content": note, "width": 300, "height": 220, "color": 4}})

    connections: dict[str, dict] = {}

    def link(src: str, dst: str) -> None:
        connections.setdefault(src, {"main": [[]]})["main"][0].append({"node": dst, "type": "main", "index": 0})

    has_pred = {e["target"] for e in pipeline["edges"]}
    for n in pipeline["nodes"]:
        if n["id"] not in has_pred:
            link(trigger_name, names[n["id"]])
    for e in pipeline["edges"]:
        if e["source"] in names and e["target"] in names:
            link(names[e["source"]], names[e["target"]])

    wf = {"name": pipeline.get("title") or "Model Studio pipeline", "nodes": nodes, "connections": connections,
          "settings": {"executionOrder": "v1"}, "pinData": {}, "active": False,
          "meta": {"generatedBy": "model-finder studio", "goal": pipeline.get("goal")}}
    validate(wf)
    return wf


VALID_VERSIONS = {TRIGGER[0]: {1}, CODE[0]: {1, 2}, HTTP[0]: {1, 2, 3, 4, 4.1, 4.2, 4.3, 4.4}, STICKY[0]: {1}}


def validate(wf: dict) -> None:
    """n8n's importer accepts broken files silently (docs/n8n-study.md §5), so check our own output."""
    names = [n["name"] for n in wf["nodes"]]
    ids = [n["id"] for n in wf["nodes"]]
    problems = []
    if len(set(names)) != len(names):
        problems.append("duplicate node names")
    if len(set(ids)) != len(ids):
        problems.append("duplicate node ids")
    for n in wf["nodes"]:
        if n["typeVersion"] not in VALID_VERSIONS.get(n["type"], set()):
            problems.append(f"{n['name']}: unsupported {n['type']} v{n['typeVersion']}")
        pos = n.get("position")
        if not (isinstance(pos, list) and len(pos) == 2 and all(isinstance(v, (int, float)) for v in pos)):
            problems.append(f"{n['name']}: position must be [x, y]")
        if "credentials" in n:
            problems.append(f"{n['name']}: must not carry credentials")
    known = set(names)
    for src, outs in wf["connections"].items():
        if src not in known:
            problems.append(f"connection from unknown node {src!r}")
        for branch in outs.get("main", []):
            for c in branch:
                if c["node"] not in known:
                    problems.append(f"connection to unknown node {c['node']!r}")
    sticky = {n["name"] for n in wf["nodes"] if n["type"] == STICKY[0]}
    if sticky & set(wf["connections"]):
        problems.append("sticky notes cannot have connections")
    if problems:
        raise ValueError("invalid n8n export: " + "; ".join(problems))

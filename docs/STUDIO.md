# Model Studio — contract (v1)

Chat on the left, n8n-style canvas on the right. You say what you want to build ("a voice assistant that runs
on my iPhone"); the planner splits it into steps, Model Finder finds models per step under the device's memory
budget, Jev picks the best fit, and the result is laid out as connected nodes. v1 *designs* pipelines (no
execution). Export to an n8n workflow file.

Our own UI and code — n8n is studied (REA + public source/docs, see `docs/n8n-study.md`) for UX ideas and its
workflow file format only; no n8n code, styles, icons or branding are copied.

## Layout

```
src/model_finder/studio/      backend: planner (Claude), resolver (model-finder + Jev), budget, n8n export
src/model_finder/web.py       mounts /studio (static build) and /api/studio/*
studio/                       frontend source (Vite + React + TypeScript + @xyflow/react)
src/model_finder/static/studio/   frontend build output (committed, served at /studio/)
```

## Pipeline JSON (the one shape everything shares)

```jsonc
{
  "id": "p_1a2b3c",
  "title": "Offline voice assistant",
  "goal": "I want a voice assistant that runs on my iPhone",
  "device": {"kind": "phone", "label": "iPhone", "ram_cap_gb": 4.0},   // kind: phone|mac|laptop|gpu|server|any
  "nodes": [
    {
      "id": "n1",
      "type": "input",              // input | model | logic | output
      "title": "Microphone",
      "step": "Captures the user's voice",
      "task": null,                 // HF task id for model nodes, e.g. "automatic-speech-recognition"
      "inputs": [], "outputs": ["audio"],
      "position": {"x": 0, "y": 120},
      "model": null,                // set for type=model, see Model below
      "alternatives": [],
      "status": "ok",               // ok | no_match | over_budget | pending
      "notes": null
    },
    {
      "id": "n2", "type": "model", "title": "Speech to text", "step": "Transcribes the user's speech",
      "task": "automatic-speech-recognition", "inputs": ["audio"], "outputs": ["text"],
      "position": {"x": 320, "y": 120},
      "model": {
        "id": "openai/whisper-large-v3-turbo", "url": "https://huggingface.co/openai/whisper-large-v3-turbo",
        "params_h": "809M", "ram_gb": 0.6, "variant": "q5_0", "variant_repo": "ggerganov/whisper.cpp",
        "format": "ggml", "license": "mit", "open_license": true, "downloads": 5100000,
        "how_to_run": "whisper.cpp runs this on iOS/Android; use the q5/q8 .bin.",
        "why": "Speech to text, 809M parameters, ~0.6 GB ...", "fit": 0.91
      },
      "alternatives": [ /* up to 4 more Model objects */ ],
      "status": "ok", "notes": null
    }
  ],
  "edges": [{"id": "e1", "source": "n1", "target": "n2", "label": "audio"}],
  "budget": {"ram_cap_gb": 4.0, "total_ram_gb": 2.7, "fits": true,
             "note": "All three models loaded at once need ~2.7 GB of the phone's ~4 GB budget."},
  "updated_at": "2026-10-15T20:00:00Z"
}
```

## API (same origin as the Model Finder web server, port 8765)

| method | path | body | returns |
|---|---|---|---|
| POST | `/api/studio/chat` | `{"session_id": str\|null, "message": str}` | `{"session_id", "reply": str, "pipeline": Pipeline, "timing_ms": {...}, "planner": "claude-cli:opus" }` |
| GET | `/api/studio/sessions/{id}` | – | `{"session_id", "messages": [{"role","text"}], "pipeline"}` |
| POST | `/api/studio/swap` | `{"session_id", "node_id", "model_id"}` | `{"pipeline"}` (picks an alternative; budget recomputed) |
| POST | `/api/studio/move` | `{"session_id", "positions": {"n1": {"x","y"}}}` | `{"ok": true}` (persist manual layout) |
| GET | `/api/studio/export/n8n/{session_id}` | – | n8n workflow JSON (`Content-Disposition: attachment`) |
| GET | `/api/studio/export/pipeline/{session_id}` | – | Pipeline JSON download |

Errors: HTTP 4xx/5xx with `{"error": "plain-English message"}`.
Chat replies take ~5–20 s (Claude plans the steps; model search + Jev are ~1 s), so the UI shows progress.

## v1.1 — where it runs (local hardware vs cloud)

The harness asks where the pipeline should run before planning, unless the message or the UI target already says.

### New / changed fields

`POST /api/studio/chat` body adds optional `"deployment": "local" | "phone" | "cloud" | null`
(set by the target switch or an `ask` option button). Response adds:

```jsonc
"ask": null | {                       // present when the harness needs an answer before planning
  "question": "Where should this run?",
  "options": [
    {"label": "This computer", "value": "local", "detail": "Apple M1 Pro · 16 GB — only models that fit here"},
    {"label": "My phone", "value": "phone", "detail": "on-device models under ~4 GB"},
    {"label": "Cloud", "value": "cloud", "detail": "hosted APIs or a rented GPU — bigger models"}
  ]
}
```
When `ask` is set, `pipeline` may be `null` (nothing planned yet). Clicking an option sends
`{"session_id", "message": option.label, "deployment": option.value}`.

`pipeline.deployment`:
```jsonc
{"mode": "local" | "phone" | "cloud" | "other",
 "label": "This Mac · Apple M1 Pro · 16 GB",            // shown in the top bar instead of device.label
 "hardware": {"chip": "Apple M1 Pro", "ram_gb": 16, "gpu": "Apple M1 Pro (16-core GPU)", "vram_gb": null, "unified": true} | null}
```

Model object adds:
```jsonc
"hosting": "local" | "hosted_api" | "gpu_server",
"providers": [{"name": "together", "price_in": 0.2, "price_out": 0.6, "tokens_per_s": 95.0, "latency_ms": 420}]  // live hosted APIs, cheapest first; [] if none
```
In cloud mode `budget` gets `"mode": "cloud"` and `note` describes hosted APIs / GPU size instead of a memory cap
(`ram_cap_gb` may be null; `total_ram_gb` counts only `gpu_server` models).

`GET /api/studio/hardware` → `{"label": "Apple M1 Pro · 16 GB", "kind": "mac", "chip", "ram_gb", "gpu", "vram_gb", "unified", "budget_gb": 11.2}`

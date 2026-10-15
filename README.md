# Model Flow Scraper

A personal "model radar": it keeps track of new AI models as they come out on **Hugging Face**, **Replicate**
and **X (Twitter)**, works out what each one does, how big it is and what hardware it runs on, and lets you
ask in plain English, e.g. *"I am looking to make a voice model on my phone"*, to get a short ranked list
of models that actually fit.

It has three parts that work together:

| Part | Folder | What it does |
|---|---|---|
| **Model Finder** (search app) | `src/`, `setup.sh` | Local catalog of ~160k Hugging Face models with size, memory, formats, license and device fit, indexed in Qdrant (semantic + keyword search). Answers questions from the command line or a small web page. |
| **Ingestion pipeline** (n8n) | `n8n/` | Runs every 6 hours. Pulls the newest and trending Hugging Face models, quantized copies, all public Replicate models and the saved X posts, cleans them and joins them into one record per model: `data/joined/models_joined.json`. Model Finder merges that file in. |
| **X watch** | `automation/`, `data/x/` | A Claude desktop scheduled task that reads the signed-in X feed and searches 3× a day (read-only), and saves model posts, links, likes and "what people say" to `data/x/x_signals.jsonl`. |

Plus `research/embedding-bench/`: a benchmark of 7 small embedding models on this project's own data
(Qwen3-Embedding-0.6B won and is now the default).

## Where we left off (2026-10-15)

**Built and working**
- Model Finder: seeded from the `cfahlgren1/hub-stats` daily snapshot (~160k useful models), enrichment
  (task, size, formats/quants, memory estimate, device tier, license), copies linked to their originals,
  hybrid search in Qdrant with an LLM that turns questions into filters, CLI + web page, tests.
- Embeddings were switched from `bge-small-en-v1.5` to `Qwen/Qwen3-Embedding-0.6B` after the benchmark
  (P@10 0.667 vs 0.560 on our questions). A full index rebuild with Qwen3 was **still running** on the
  first computer when this was pushed, so the new machine has to build its own index anyway (see below).
- n8n pipeline: active, every 6 hours, about 2 minutes per run. Last run: 2,614 joined models
  (Hugging Face newest 1,000 + trending 500 + ~430 quantized copies, 1,811 Replicate models, X posts).
  Field-by-field schema in `n8n/SCHEMA.md`.
- X watch: scheduled for 9am / 2pm / 8pm. The read-and-extract steps were tested by hand and 9 real posts
  are in `data/x/x_signals.jsonl`, but **the scheduled task had not completed a run yet**.

**Open items / next steps**
1. Confirm the X watch actually runs on schedule (check the task's run history in the Claude app).
   Its prompt is saved in `automation/x-model-watch.md`.
2. Re-check search quality after the Qwen3 rebuild, e.g. `model-finder ask "I am looking to make a voice
   model on my phone"`; the before-switch answers are saved in `research/embedding-switch/`.
3. Replicate: 562 Replicate-only models have no task label (vague names, not in a task collection), and
   popular Replicate models whose Hugging Face repo is not newest/trending stay "API only" in the n8n file.
   Model Finder fetches missing Hugging Face repos itself, which closes part of this.
4. Memory estimates are formulas, not measurements; mixed-precision repos (e.g. FP8 parts) can come out high.
5. "On my phone"-style questions depend on the device filters, not on the embeddings: every embedder
   tested ranked them poorly on its own.
6. The Replicate API token used on the first computer was pasted into a chat. Make a new one, delete the
   old one, and use the new one below.

## Pick up on another computer

You need: Docker Desktop, [uv](https://docs.astral.sh/uv/), and optionally [Ollama](https://ollama.com)
(for better question parsing; otherwise keyword rules are used).

```bash
git clone https://github.com/smammadov1994/model-flow-scraper.git
cd model-flow-scraper
./setup.sh          # Python env, Qdrant, Hugging Face snapshot, enrich, index (about an hour the first time)
.venv/bin/model-finder ask "I am looking to make a voice model on my phone"
```

n8n ingestion pipeline:

```bash
cd n8n
cp .env.example .env          # then paste your Replicate token after REPLICATE_API_TOKEN=
docker compose up -d          # editor at http://localhost:5679 (create the local owner account on first visit)
python3 build_workflow.py
docker exec model-finder-n8n n8n import:workflow --input=/workflows/workflow.json
docker exec model-finder-n8n n8n publish:workflow --id=modelFinderIngest
docker restart model-finder-n8n
```

The pipeline writes `data/joined/models_joined.json`; `model-finder update` (or `model-finder joined`) merges it.

X watch: in the Claude desktop app, sign in to X in Claude's built-in browser, then create a scheduled task
(9am, 2pm, 8pm daily) using the prompt in `automation/x-model-watch.md`. Change the
`~/Desktop/Projects/model-finder` path in it if the repo lives somewhere else.

Not in git (rebuilt or re-created locally): the Hugging Face snapshot (`data/raw`, 1.5 GB), the catalog
database, the Qdrant index, the n8n database, `data/joined/`, and `.env` secrets.

---

# Model Finder (search app)

Ask in plain English — "I want to make a voice model on my phone" — and get a short, ranked list of
Hugging Face models that actually fit, with why each one fits and how to run it on that device.

Everything runs on your machine: a local copy of Hugging Face's model catalog, a local search database
(Qdrant in Docker), and a small local LLM (Ollama) that turns your question into search filters.

## Quick start

```bash
./setup.sh
```

That installs the Python environment, starts Qdrant, pulls the `qwen3:4b` model into Ollama (if Ollama is
installed), downloads today's Hugging Face snapshot (1.6 GB), keeps the ~160k models that are actually used,
works out their size/memory/formats, and builds the search index. First run takes roughly an hour on an
M1 Pro, mostly turning model descriptions into search vectors on the Mac's GPU.

Then ask:

```bash
.venv/bin/model-finder ask "I am looking to make a voice model on my phone"
.venv/bin/model-finder ask "small coding model for my MacBook"
.venv/bin/model-finder ask "image generation under 8GB VRAM"
```

Or open the web page:

```bash
.venv/bin/model-finder serve
```

and go to http://localhost:8765 (`?demo=1` shows a canned answer without the backend).

If `model-finder serve` is running, `ask` sends the question to it (the model is already loaded there), which is much faster than loading it fresh (~15 s).

Useful flags for `ask`: `--device phone|mac|laptop|gpu|any` (overrides what's guessed from the question),
`--limit 10`, `--json`, `--no-llm` (keyword rules only), `--no-answer` (skip the short LLM overview).

## Keeping it fresh

| When | Command | What it does |
|---|---|---|
| Daily / hourly | `model-finder update` | Pulls models created since the last run (newest first, 1,000 per request, with file lists) plus the current top‑1,000 trending, merges the n8n joined file if present, re-links copies to originals, re-indexes only what changed. |
| Weekly | `model-finder reseed` | Re-downloads the daily snapshot and refreshes download/like counts for everything. Only changed models are re-embedded. |
| Any time | `model-finder add org/model …` | Index specific repos right now. |
| Any time | `model-finder add-from-text posts.txt` | Index every Hugging Face link found in pasted text (the X/Twitter hook, see below). |
| Any time | `model-finder joined` | Merge `data/joined/models_joined.json` from the n8n workflow (HF + Replicate + X). |
| Any time | `model-finder cards --limit 200` | Let the LLM read more model cards to fill missing task/size/license (answers are kept only if the card text backs them up). |
| Any time | `model-finder sizes --limit 3000` | Estimate size from weight-file sizes for models that publish no parameter count. |
| Overnight | `model-finder index --backfill` | Add semantic vectors to the long tail (first build gives them to the ~35k most-used/newest models; the rest use keyword search until then). About 2 hours for all. |
| — | `model-finder stats` | Counts and last refresh times. |

To schedule: e.g. `crontab -e` → `0 * * * * cd ~/Desktop/Projects/model-finder && .venv/bin/model-finder update >> logs/update.log 2>&1`.
Set `HF_TOKEN` for higher Hugging Face rate limits (anonymous is 500 requests / 5 min — plenty for hourly updates).

## How it works

```
Hugging Face daily snapshot ─ seed ─┐
Hub API (newest + trending) ─ update ┼─> enrich ─> catalog (SQLite) ─> families ─> Qdrant index ─> ask / web page
n8n joined file (HF+Replicate+X) ────┘
```

1. **Seed** — `cfahlgren1/hub-stats` (`models.parquet`, ~3.15M models, refreshed daily, includes each repo's
   file list). We keep a model if it had ≥250 downloads in the last 30 days or ≥10 likes, or if it is
   from the last 90 days and has ≥25 downloads / ≥2 likes / comes from an org that has a 50k‑download model.
   (Tune with `MF_MIN_DOWNLOADS`, `MF_MIN_LIKES`, `MF_RECENT_DAYS`, …) Result: ~160k models.
2. **Enrich** (`enrich.py`) — per model, from metadata and file names only:
   - **Task**: the Hub's `pipeline_tag`, else transformers info, card, tags, the base model's task, library,
     architecture, and finally keywords in the name. About half the Hub has no task tag, so this matters.
   - **Size**: safetensors parameter count, else the GGUF header, else the size in the name (`7B`, `82M`).
     Packed 4‑bit repos (MLX/GPTQ/AWQ) under-report, so their name wins.
   - **Formats** from file names: GGUF (with quant names like `Q4_K_M`), whisper.cpp GGML, MLX, ONNX
     (fp16/int8/q4 variants), Core ML, ExecuTorch `.pte`, TFLite/LiteRT, OpenVINO, MLC, safetensors,
     PyTorch, diffusers, AWQ/GPTQ/EXL2/bitsandbytes.
   - **License, languages, base model, release date.**
   - **Memory estimate** per variant and a **device tier** (formula below).
   - **Weight-file sizes** (`sizes.py`): for popular models with no parameter count anywhere (CLIP, XTTS,
     wav2vec2 …), one API call fetches file sizes; one copy of the weights ÷ bytes per parameter.
   - **LLM card reading** (`card_llm.py`): only for models still missing task or size, most-downloaded first;
     an answer is kept only if the card text supports it (the size number appears in the card, the task's
     key words appear) — small local LLMs otherwise guess. Results cached so each card is read once.
3. **Families** (`families.py`) — links converted copies to the original: e.g. `hexgrad/Kokoro-82M` gets
   its ONNX and MLX builds attached, so "Kokoro on a phone" is answered by the original plus the file you
   actually download. Uses the Hub's "quantized" relation, but only if the copy has about the same
   parameter count (the Hub labels many distilled models "quantized"), and name matching for converter
   orgs (mlx-community, onnx-community, bartowski, unsloth, …) whose metadata is often wrong.
4. **Index** (`index.py`) — one point per model family in Qdrant: a semantic vector of a generated "what this
   model does" paragraph (`Qwen/Qwen3-Embedding-0.6B`, 1024 dims, on the Mac GPU in fp16; queries get its
   "query" instruction, documents none), plus BM25 keyword vectors computed inside Qdrant, plus filterable
   fields (task, modality, formats, memory, params, license, languages, date). Semantic vectors are cached by
   text hash in the catalog, so updates only embed what changed. To save first-build time, they go to the
   ~35k most-downloaded/liked/new models; `index --backfill` does the rest. Qwen3 won a 7-model benchmark on
   this data (`research/embedding-bench/`); `MF_DENSE_MODEL=BAAI/bge-small-en-v1.5` is ~25× faster to
   build on CPU if you want speed over accuracy (needs `index --recreate`, since vector sizes differ).
5. **Ask** (`query.py`) — the LLM turns the question into a search phrase and filters (each field is
   declared to it with a type and description); keyword rules run alongside and win on obvious things.
   Device → memory budget and formats is plain code, not the LLM. Search = dense + BM25 fused with
   reciprocal rank fusion, filtered. If fewer than 5 models survive, filters are loosened step by step
   (date/language → license → size unknown or 1.5× budget → related tasks → no hardware filter) and the
   answer says what was loosened. Ranking = 50% search relevance, 25% popularity, 12% recency, plus small
   boosts (good format for the device, exact task, code-specialised for coding questions, X buzz) and
   penalties (test/toy repos). Copies collapse into one card per model.

### Memory formula

```
weights_gb = parameters × bits_per_weight ÷ 8 ÷ 1e9
ram_gb     = weights_gb × 1.20                        # runtime overhead
           + 0.17 × √(parameters in billions)          # chat/LLM models only: 4k-token context cache
           + 1.0                                       # image/video generators only: latents + decoding
```

Bits per weight come from the actual variant: GGUF `Q4_K_M` = 4.85, `Q8_0` = 8.5, MLX 4‑bit = 4.5,
ONNX q4 = 4.5 / int8 = 8.5, fp16/bf16 = 16 (full table in `enrich.BPW`). The native size uses the
repo's real dtype (never assumes fp32). Example: an 8B model at Q4_K_M ≈ 4.85 GB × 1.2 + 0.48 ≈ 6.3 GB.
1–2‑bit quants are listed but never used as the "minimum", because quality drops too far.

### Device tiers

| Tier | Memory budget | Must have a format that runs there |
|---|---|---|
| phone | ≤ 4 GB | GGUF, GGML, ONNX, Core ML, ExecuTorch, TFLite, MLX, MLC |
| laptop / Mac | ≤ 12 GB (16 GB machine) | the above + safetensors/PyTorch/diffusers/OpenVINO |
| GPU box | ≤ 24 GB | anything |
| server | more | anything |

If you say how much memory you have, that replaces the default: "under 8GB VRAM" → 8 GB; "32GB MacBook"
→ 70% of 32; a phone gets half its RAM.

## LLM options

`MF_LLM=auto` (default) picks: Anthropic API if `ANTHROPIC_API_KEY` is set (`MF_ANTHROPIC_MODEL`,
default `claude-haiku-5-5`), else a running Ollama (`qwen3:4b`, then `qwen3.5:2b`, `llama3.2:3b`,
`gemma4:12b`), else keyword rules only. Force one with `MF_LLM=ollama:qwen3:4b`, `MF_LLM=anthropic:claude-haiku-5-5`,
`MF_LLM=claude-cli:haiku`, or `MF_LLM=none`.

## X / Twitter hook

Not built in v1. What exists: `sources/x.py` pulls `huggingface.co/<org>/<model>` links out of any text, and
`model-finder add-from-text` indexes them; `XSource.discover()` is the stub to implement with the X API.
Separately, the n8n workflow in `n8n/` (owned by another pipeline) writes `data/joined/models_joined.json`
with X posts joined to models. `model-finder joined` (also run by `setup` and `update`) merges it: Hugging
Face models get "what people say on X" lines and a small ranking boost for recent buzz, and models only
announced on X or only on Replicate become searchable (they're skipped by hardware filters since there are
no files to measure).

## Layout

```
src/model_finder/
  config.py      paths, thresholds, env vars (MF_*)
  seed.py        snapshot download + selection (DuckDB over the parquet)
  hf_poll.py     incremental updates from the live Hub, `add`
  ingest.py      any source row -> normalized raw -> enriched record -> catalog
  enrich.py      formats, quants, task, params, RAM, tiers, license, languages (pure functions)
  card_llm.py    LLM pass over model cards for missing fields (cached, evidence-checked)
  sizes.py       parameter estimates from weight-file sizes
  families.py    original <-> converted copies
  summarize.py   the "what this model does" text that gets embedded
  index.py       Qdrant collection, embedding, sync
  query.py       question -> plan -> search -> relax -> rank -> explain
  howto.py       one-line "how to run it on this device"
  llm.py         Ollama / Anthropic / claude CLI / none
  cli.py, web.py, static/index.html
  sources/x.py, sources/joined.py
docs/SPEC.md     field-by-field contract
```

Tests: `.venv/bin/pytest -q` (no network needed).

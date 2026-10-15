# models_joined.json — schema v1.0

One JSON array, one object per model, sorted by `popularity.score` (high first).
Produced by the n8n workflow "Model Finder - data ingestion (HF + Replicate + X)".
This is the cleaned, joined data **before** embedding. The `search_text` field is what should be embedded;
everything else is metadata for filters and display.

| Field | Type | What it means |
|---|---|---|
| `id` | string | Stable key. `hf:<org>/<repo>` for Hugging Face models, `replicate:<owner>/<name>` for Replicate-only models, `x:<normalized name>` for models only seen on X so far. |
| `name` | string | Short display name. |
| `status` | enum | `released` (weights on HF) · `api_only` (Replicate, no matching HF repo) · `announced` (X says weights are coming) · `seen_on_x` (talked about on X, not matched yet). |
| `summary` | string | One plain-English line: task, size, where it runs, RAM. |
| `task.primary` | string | HF-style task id, e.g. `text-to-speech`, `automatic-speech-recognition`, `text-generation`. |
| `task.label` / `task.group` | string | Friendly label and broad group (`text`, `speech-in`, `speech-out`, `vision`, `image-generation`, `video`, `multimodal`, `embedding`, `other`). |
| `task.all` | string[] | Every task the model is tagged with. |
| `task.inputs` / `task.outputs` | string[] | e.g. `["text"]` → `["audio"]`. |
| `task.capabilities` | string[] | Search words: `voice cloning`, `coding`, `reasoning`, `tool calling`, `dictation`… |
| `task.source` | string | Where the task came from: `pipeline_tag`, `card`, `tags`, `library`, `name`, `replicate_collection`, `x_post`, `keywords`, `base_model`. |
| `size.params` | int | Total parameters. |
| `size.params_active` | int | Active parameters per token for mixture-of-experts models. |
| `size.params_source` | string | `safetensors` / `gguf_header` (exact), `name` / `filename` / `base_model` / `x_post` (inferred). |
| `size.context_length`, `size.architecture`, `size.native_dtypes` | | From GGUF/safetensors headers when available. |
| `hardware.est_min_ram_gb` | number | Estimated RAM for this repo's smallest practical variant. Formula: params × bits ÷ 8 × 1.2, plus a context allowance for LLMs and +1 GB for image/video generators. Same rule as `src/model_finder/enrich.py`. |
| `hardware.smallest_variant` | string | e.g. `Q4_K_M`, `4bit`, `native`. |
| `hardware.variants[]` | object[] | Every variant in this repo: `{variant, bits_per_weight, est_ram_gb}`. |
| `hardware.runs_on` | string[] | Tiers this repo fits: `phone` (≤4 GB and a phone-ready format), `laptop` (≤12 GB), `gpu` (≤24 GB), `server`. |
| `hardware.community_variants[]` | object[] | Quantized copies by others (unsloth, mlx-community, bartowski…): `{repo, formats, smallest_variant, est_ram_gb, runs_on, downloads_30d}`. |
| `hardware.best_option` | object | The smallest practical way to run it, own repo or a copy: `{repo, variant, est_ram_gb, runs_on}`. |
| `hardware.runs_on_any_variant` | string[] | Tiers reachable through the model itself **or** any community copy. Use this for "on my phone" filters. |
| `hardware.on_device_ready` | string[] | Phone-friendly formats present: `gguf`, `mlx`, `coreml`, `onnx`, `executorch`, `tflite`, `mlc`. |
| `hardware.notes` | string[] | Plain-English caveats. |
| `formats` | string[] | `safetensors`, `gguf`, `mlx`, `onnx`, `coreml`, `executorch`, `tflite`, `pytorch`, `diffusers`, `awq`, `gptq`, `exl2`, `bnb`, `nemo`. |
| `quantizations` | string[] | GGUF quant names found in filenames, smallest first. |
| `license` | object | `{id, name, url, commercial_use: yes/no/check_terms/unknown, inherited_from?, hint_from_x?}`. |
| `languages` | string[] | ISO codes from the model card. |
| `lineage` | object | `{relation: finetune/quantized/adapter/merge, base_models: [...], variant_of?}`. `variant_of` is set on real quantized copies (same size as the base), so search results can be grouped under the original. |
| `availability.huggingface` | object | `{repo, author, library, gated, lists: [trending/newest/variant_lookup], file_count, inference_providers}`. |
| `availability.replicate` | object | `{model, url, run_count, visibility, is_official, collections, inputs, latest_version, cover_image_url, hf_repos_mentioned}`. |
| `popularity` | object | `hf_downloads_30d`, `hf_likes`, `hf_trending_score`, `replicate_runs`, `x_mentions`, `x_engagement` (likes + 3×reposts), `x_buzz` (engagement decayed with a 7-day time constant), `score` (0–100 percentile of a log-weighted mix). |
| `social` | object | X data: `x_mentions`, `takes[]` (one-line "what people think"), `top_posts[]` `{url, author, posted_at, likes, reposts, views, take, excerpt}`, `kinds`, `size_hints`, `license_hints`, `on_device_mentions`, `first_seen`. `null` if never seen on X. |
| `links` | object | `huggingface`, `replicate`, `github`, `paper`, `x_posts[]`. |
| `dates` | object | `created_at`, `last_modified`. |
| `description` | string | Replicate description when available (HF card text is not fetched yet). |
| `search_text` | string | Ready-to-embed paragraph combining summary, task, formats, license, languages, lineage, description, X takes and tags. |
| `quality` | object | `completeness` (0–1), `missing[]`, `inferred[]`, `needs_enrichment`, `needs_hf_fetch`. |
| `provenance` | object | `sources[]` (`huggingface`, `replicate`, `x`), `fetched_at`, `schema_version`. |

## How sources are joined

1. **Hugging Face** newest (1,000) + trending (500) lists are the base. Brand-new uploads with no downloads, likes, trend score, notable author, X mention or Replicate match are dropped (counted in `run_report.json`).
2. **Quantized copies**: for the top 60 trending originals without GGUF/MLX, the workflow asks HF for their most-downloaded `quantized` children. Children inherit size/task/license from the base, and each base gets `hardware.community_variants` + `best_option`.
3. **Replicate** models attach to an HF entity when their description/GitHub/paper/weights URL links a HF repo we have, or when the normalized name matches. Otherwise they become `replicate:` entities with `status: api_only`. Collection slugs (e.g. `text-to-speech`) fill in the task.
4. **X posts** (`data/x/x_signals.jsonl`, latest record per URL, last 14 days) attach by HF repo link first, then by normalized model name. Unmatched names become `x:` entities (`announced` / `seen_on_x`) so new releases are searchable before they hit HF.

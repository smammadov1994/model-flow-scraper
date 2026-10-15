# model-finder — internal spec (v1)

Goal: answer plain-English questions like "voice model on my phone" with a ranked,
explained list of Hugging Face models, from a local index.

Python 3.12 project, `src/model_finder/`, venv at `.venv` (`uv pip install -e '.[dev]'`),
tests in `tests/` run with `.venv/bin/pytest -q`. No network in unit tests.

## Pipeline

```
hub-stats parquet ──seed.py──┐
HF list_models ──hf_poll.py──┼─> raw row dict ─> enrich.build_record() ─> catalog.db (SQLite)
X/Twitter (hook) ─sources/───┘                                            │
                                     families.py (base <-> derivatives) ──┘
catalog.db ─> index.py (Qdrant: dense + server BM25, payload) ─> query.py ─> cli.py / web.py
```

## Raw row (input to enrichment)

A plain dict with the same keys as the hub-stats `models` split / HF `list_models` expand output:

| key | type | notes |
|---|---|---|
| `id` | str | `org/name` |
| `author` | str\|None | |
| `pipeline_tag` | str\|None | missing for ~half the Hub |
| `library_name` | str\|None | |
| `tags` | list[str] | includes `license:x`, `base_model:...`, `dataset:...`, language codes (`en`), `arxiv:...`, `region:us`, `gguf`, `mlx`, `onnx`, task names, etc. |
| `siblings` | list[str] | repo-relative filenames, e.g. `onnx/model_q4.onnx` (already flattened to strings) |
| `card` | dict | parsed `cardData` (YAML front-matter of README): may hold `license`, `language` (str or list), `base_model` (str or list), `pipeline_tag`, `library_name`, `tags` |
| `config` | dict | parsed `config` (may have `model_type`, `architectures`, `diffusers: {_class_name}`) |
| `transformers_info` | dict\|None | may hold `pipeline_tag`, `auto_model` |
| `safetensors` | dict\|None | `{"parameters": {"BF16": n, "F32": n, ...}, "total": n}` |
| `gguf` | dict\|None | `{"total": n_params, "architecture": str, "context_length": int, ...}` |
| `base_models` | dict\|None | `{"relation": "quantized"|"finetune"|"adapter"|"merge", "models": [{"id": ...}]}` |
| `downloads`, `likes` | int | `downloads` = last 30 days |
| `trending` | float | |
| `created_at`, `last_modified` | ISO str | |
| `base_task` | str\|None | pipeline_tag of the base model (looked up by the caller) — use as a fallback for task |

## Record (output of enrichment = Qdrant payload)

`enrich.build_record(raw: dict) -> dict` returns:

| field | type | meaning |
|---|---|---|
| `id`, `author`, `name` (part after `/`), `url` (`https://huggingface.co/{id}`) | str | |
| `task` | str\|None | HF pipeline tag; see "Task inference" |
| `task_source` | str | `pipeline_tag` / `transformers_info` / `card` / `tags` / `base_model` / `library` / `config` / `name` / `llm` / `unknown` |
| `modality` | str | coarse group from TASK_GROUPS: `text`, `code`, `vision`, `image-generation`, `video`, `speech-in` (ASR), `speech-out` (TTS/voice), `audio`, `multimodal`, `embedding`, `other` |
| `library` | str\|None | |
| `params` | int\|None | parameter count |
| `params_source` | str | `safetensors` / `gguf` / `name` / `file_size` / `llm` / `unknown` |
| `native_dtype` | str\|None | dominant dtype from safetensors parameters (`BF16`, `F16`, `F32`, `F8_E4M3`, ...) |
| `formats` | list[str] | formats present in THIS repo (see "Formats") |
| `gguf_quants` | list[str] | quant names found in filenames, upper-case, e.g. `["Q4_K_M","Q8_0","F16"]` |
| `onnx_variants` | list[str] | e.g. `["fp32","fp16","q4","int8"]` |
| `quant_bits` | float\|None | if the repo itself is a fixed quant (mlx `4bit` name/tag, AWQ/GPTQ = 4, `8-bit` tag), the bits per weight |
| `license` | str\|None | lower-case SPDX-ish (`apache-2.0`, `mit`, `llama3.1`, `other`, ...) |
| `open_license` | bool | license in PERMISSIVE set (apache-2.0, mit, bsd-*, cc-by-4.0, cc0-1.0, openrail*, bigscience-openrail-m, creativeml-openrail-m, mpl-2.0, unlicense, cc-by-sa-4.0, gpl-*, lgpl-*, agpl-3.0, afl-3.0, artistic-2.0, isc, wtfpl, ecl-2.0, zlib) |
| `languages` | list[str] | ISO codes from card `language` + bare 2-3 letter language tags |
| `base_model` | str\|None | first base model id |
| `relation` | str\|None | `quantized`/`finetune`/`adapter`/`merge` |
| `is_adapter` | bool | LoRA/adapter repo (relation adapter, `peft` library, or only `adapter_model.*` weights) |
| `created_at` | str ISO | |
| `created_ts` | int | unix seconds (for range filters) |
| `last_modified` | str ISO | |
| `downloads`, `likes` | int | |
| `trending` | float | |
| `ram_gb` | dict[str, float] | estimated RAM per available variant, e.g. `{"Q4_K_M": 4.9, "Q8_0": 8.3, "fp16": 15.2}` (see "RAM") |
| `ram_gb_min` | float\|None | lowest RAM among *practical* variants that exist in this repo (exclude 1–2 bit and IQ1/IQ2 quants: too lossy) |
| `ram_gb_native` | float\|None | RAM at native precision |
| `device_tier` | str | smallest tier that fits `ram_gb_min` *and* has a runnable format: `phone` < `laptop` < `gpu` < `server`; `unknown` if no size |
| `tags_clean` | list[str] | tags minus noise (`license:`, `region:`, `endpoints_compatible`, `arxiv:`, `base_model:`, `dataset:`, `deploy:`, `autotrain_compatible`, `text-generation-inference`, `has_space`, language codes, `model-index`, `generated_from_trainer`) |
| `datasets` | list[str] | from `dataset:` tags |
| `context_length` | int\|None | from gguf |
| `architecture` | str\|None | gguf architecture or config model_type |

Family fields are added later by `families.py` (not by enrich): `family_formats`, `derivatives`, `family_ram_gb_min`, `family_device_tier`.

## Formats

Detect from filenames (case-insensitive), tags, library, author:

| format | rule |
|---|---|
| `gguf` | any `*.gguf` |
| `ggml` | `ggml-*.bin` or `*.ggml` (whisper.cpp / old llama.cpp) |
| `mlx` | tag `mlx` or library `mlx`/`mlx-lm`/`mlx-audio`, or author `mlx-community` |
| `onnx` | any `*.onnx` |
| `coreml` | path contains `.mlpackage/` or `.mlmodelc` or ends `.mlmodel` (also zipped `.mlmodelc.zip`) |
| `executorch` | `*.pte` |
| `tflite` | `*.tflite`, `*.litertlm`, or MediaPipe `*.task` |
| `openvino` | `openvino_model.xml` or tag `openvino` |
| `mlc` | `mlc-chat-config.json` or tag `mlc-llm` |
| `safetensors` | `*.safetensors` |
| `pytorch` | `*.bin` (not ggml-), `*.pt`, `*.pth`, `*.ckpt` |
| `diffusers` | `model_index.json` or library `diffusers` |
| `awq` / `gptq` / `exl2` / `bnb` | tag or name contains awq / gptq / exl2 / (`bnb` or `bitsandbytes`) |
| `nemo` | `*.nemo` |

GGUF quant names: regex on `.gguf` filenames for (in priority order)
`IQ[1-4]_(XXS|XS|S|M|NL)`, `Q[2-8]_K(_[SML]|_XL)?`, `Q[4-8]_[01]`, `TQ[12]_0`, `MXFP4`, `BF16`, `F16`, `F32`.
Treat `-UD-Q4_K_XL` style (unsloth dynamic) as `Q4_K_XL`. Files without a match: ignore.

ONNX variants: from `*.onnx` basenames: `fp16`, `q4f16`, `q4`, `int8`/`quantized`/`uint8`/`q8` → `int8`, `bnb4`, otherwise `fp32`.

## Bits per weight (for RAM estimates)

```
BPW = {
 "IQ1_S":1.56,"IQ1_M":1.75,"IQ2_XXS":2.06,"IQ2_XS":2.31,"IQ2_S":2.5,"IQ2_M":2.7,"Q2_K":2.63,
 "IQ3_XXS":3.06,"IQ3_XS":3.3,"IQ3_S":3.44,"IQ3_M":3.66,"Q3_K_S":3.5,"Q3_K_M":3.91,"Q3_K_L":4.27,"Q3_K_XL":4.3,
 "IQ4_XS":4.25,"IQ4_NL":4.5,"Q4_0":4.55,"Q4_1":5.0,"Q4_K_S":4.58,"Q4_K_M":4.85,"Q4_K_XL":5.0,"MXFP4":4.25,
 "Q5_0":5.54,"Q5_1":6.0,"Q5_K_S":5.54,"Q5_K_M":5.69,"Q5_K_XL":5.8,"Q6_K":6.59,"Q6_K_XL":6.8,"Q8_0":8.5,"Q8_K_XL":9.0,
 "TQ1_0":1.69,"TQ2_0":2.06,"F16":16,"BF16":16,"F32":32,
 "int4":4.5,"q4":4.5,"q4f16":4.5,"bnb4":4.5,"int8":8.5,"fp16":16,"bf16":16,"fp32":32,"fp8":8.5,
}
```
Native dtype bits: BF16/F16 = 16, F32 = 32, F8_* = 8, I8/U8 = 8, Q4/F4 = 4; unknown native → 16 for repos with safetensors, else infer.

## RAM formula (document this in README too)

```
weights_gb = params * bpw / 8 / 1e9
ram_gb     = weights_gb * 1.20            # runtime overhead (buffers, activations, runtime itself)
           + context_gb                    # only for text-generation / image-text-to-text / LLM-ish tasks
           + gen_extra_gb                  # only for image/video generation: 1.0 (latents/VAE decode at default res)
context_gb = 0.17 * sqrt(params_in_billions)   # fp16 KV cache for a 4k-token context, GQA-typical
```
Rounded to 2 decimals. Example: 8B at Q4_K_M → 8e9*4.85/8/1e9 = 4.85 GB *1.2 = 5.82 + 0.48 = 6.30 GB.

Variants to estimate for a repo: every `gguf_quants` entry; every `onnx_variants` entry;
if `quant_bits` set → one variant named `f"{quant_bits:g}bit"`; native (`safetensors`/pytorch) → `"native"`
using native dtype bits. If the repo has no params, `ram_gb` = {} and `device_tier` = `unknown`.

Size fallback when params unknown: `enrich.params_from_size(total_bytes, dtype_hint)` → params = bytes / bytes_per_param,
where dtype_hint comes from the file name (`fp16`/`f16`/`half` → 2, `int8`/`q8` → 1, `q4`/`int4` → 0.5,
`bf16` → 2) else `config.torch_dtype` else 2 for safetensors / 4 for legacy `.bin/.pth/.ckpt`. Mark `params_source="file_size"`.
(Used by a separate optional enrichment step that fetches file sizes; enrich.py only provides the function.)

## Device tiers

Memory budget the model may use (comfortable, leaves room for OS/apps):

| tier | ram budget | runnable formats |
|---|---|---|
| `phone` | ≤ 4.0 GB | gguf, ggml, onnx, coreml, executorch, tflite, mlx, mlc |
| `laptop` | ≤ 12.0 GB (16 GB MacBook / laptop) | above + safetensors, pytorch, diffusers, openvino |
| `gpu` | ≤ 24.0 GB (one consumer GPU, e.g. 24 GB) | anything |
| `server` | > 24 GB | anything |

`device_tier(ram_by_variant: dict[str,float], formats: list[str]) -> str`: smallest tier where some
variant's RAM ≤ budget AND the repo has a runnable format for that tier. A variant counts for phone only
if it comes from a phone format (gguf quants → gguf, onnx variants → onnx, `Nbit` from mlx → mlx, native
only if formats include coreml/executorch/tflite/mlc/ggml).

## Task inference (`enrich.infer_task(raw) -> (task, source)`)

Order: `pipeline_tag` → `transformers_info.pipeline_tag` → card `pipeline_tag` → first tag that is a known
pipeline tag (KNOWN_TASKS list of HF pipeline tags) → `base_task` → library map (`sentence-transformers` →
`sentence-similarity`, `diffusers` → `text-to-image` unless config class name contains Video/Img2Img/Inpaint,
`open_clip` → `zero-shot-image-classification`, `pyannote-audio` → `voice-activity-detection`,
`speechbrain` → `automatic-speech-recognition`, `nemo` → None, `peft` → None, `mlx-audio`→`text-to-speech`, `timm`→`image-classification`,
`ultralytics`→`object-detection`, `paddleocr`→`image-to-text`, `f5-tts`/`coqui`/`kokoro`/`chatterbox`→`text-to-speech`) → config architectures
(`*ForCausalLM` → `text-generation`, `*ForConditionalGeneration` with whisper → ASR, `*ForSequenceClassification` →
`text-classification`, `*ForTokenClassification` → `token-classification`, `diffusers._class_name` contains `Pipeline` → `text-to-image`/`text-to-video` by name) →
name keywords on the repo name (lower-case): `tts|text-to-speech|kokoro|xtts|f5|piper|parler|bark|vits|voice|cosyvoice|chatterbox|orpheus|sesame|csm` → `text-to-speech`;
`whisper|asr|stt|speech-to-text|wav2vec|parakeet|canary|moonshine` → `automatic-speech-recognition`; `embed|e5|bge|gte|minilm` → `feature-extraction`;
`sdxl|stable-diffusion|flux|diffusion|text-to-image` → `text-to-image`; `instruct|chat|coder|llm|-it$|gpt` → `text-generation` →
`(None, "unknown")`.

TASK_GROUPS (task → modality): text-generation, text2text-generation, summarization, translation, question-answering,
fill-mask, text-classification, token-classification, zero-shot-classification → `text`;
(code is decided by tags/name containing `code|coder|starcoder|codestral` with a text task → `code`);
image-classification, object-detection, image-segmentation, depth-estimation, zero-shot-image-classification,
zero-shot-object-detection, image-feature-extraction, mask-generation, keypoint-detection, video-classification → `vision`;
text-to-image, image-to-image, unconditional-image-generation, image-to-3d, text-to-3d → `image-generation`;
text-to-video, image-to-video, video-to-video → `video`;
automatic-speech-recognition, audio-classification, voice-activity-detection → `speech-in`;
text-to-speech, text-to-audio, audio-to-audio → `speech-out` (text-to-audio = music/sfx is `audio`; tag/name with `music` → `audio`);
image-text-to-text, visual-question-answering, image-to-text, any-to-any, video-text-to-text, audio-text-to-text, document-question-answering → `multimodal`;
feature-extraction, sentence-similarity, text-ranking → `embedding`; else `other`.

## How-to-run (`howto.run_hint(record, device) -> str`)

One short, concrete line per (best format for device, task). Examples:
- phone + gguf + text-gen: "Download the Q4_K_M .gguf and open it in an on-device llama.cpp app (PocketPal AI, LLM Farm on iOS / ChatterUI on Android)."
- phone + onnx + TTS/ASR: "Use the ONNX files with ONNX Runtime Mobile or sherpa-onnx (Android/iOS); pick the q4/int8 variant to save memory."
- phone + ggml + ASR: "whisper.cpp runs this on iOS/Android; use the q5/q8 .bin."
- phone + coreml: "Add the Core ML package to an iOS app (Xcode) — runs on the Neural Engine."
- phone + executorch: "Load the .pte file with the ExecuTorch runtime in your Android/iOS app."
- phone + tflite: "Run with Google LiteRT / MediaPipe (e.g. AI Edge Gallery on Android)."
- phone + mlx: "Runs in iOS apps built on mlx-swift (e.g. the MLX examples)."
- laptop(mac) + mlx + text-gen: "`pip install mlx-lm` then `mlx_lm.generate --model {id} --prompt 'hi'`."
- laptop + gguf: "`ollama run hf.co/{id}:{quant}` or open it in LM Studio / llama.cpp."
- laptop/gpu + diffusers: "`pip install diffusers` and load with DiffusionPipeline.from_pretrained('{id}', torch_dtype=float16)."
- gpu + safetensors text-gen: "Serve with vLLM (`vllm serve {id}`) or transformers."
- mlx + TTS: "`pip install mlx-audio` and run `mlx_audio.tts.generate --model {id} --text 'hi'`."
- fallback: "Load with the `{library}` library (see the model card)."
`device` is one of `phone`, `laptop`, `mac`, `gpu`, `server`, `any`. The quant chosen is the best one that fits the device budget (prefer Q4_K_M, then Q4_K_S/Q4_0/IQ4_XS, then Q5_K_M, Q8_0).

## Query layer

LLM rewrites the question into `{semantic_query, tasks[], modality, device, memory_gb, needs_formats[],
license_open, max_params_b, min_params_b, created_after_days, languages[], keywords[]}`; code maps
device → RAM cap + formats. Search = Qdrant prefetch(dense, bm25) + RRF with payload filters;
relax filters step by step if < 5 results; re-rank by fused relevance, then recency + popularity;
collapse derivatives into their family; explain why + how to run.

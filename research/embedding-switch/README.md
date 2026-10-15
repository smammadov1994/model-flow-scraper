# Embedding switch: bge-small → Qwen3-Embedding-0.6B (2026-10-15)

Question: "I am looking to make a voice model on my phone" (end-to-end `model-finder ask`, top 10).

| # | Before: bge-small (index of the 2,560 most-popular models only — the first build was interrupted) | After: Qwen3-Embedding-0.6B (full index, 118,685 models) |
|---|---|---|
| 1 | nari-labs/Dia-1.6B (1.6B, ~1.03 GB) | hexgrad/Kokoro-82M (82M, ~0.06 GB) |
| 2 | Supertone/supertonic-3 (None, ~0.24 GB) | Qwen/Qwen3-TTS-12Hz-1.7B-Base (1.9B, ~1.26 GB) |
| 3 | FunAudioLLM/Fun-CosyVoice3-0.5B-2512 (500M, ~0.63 GB) | nari-labs/Dia-1.6B (1.6B, ~1.03 GB) |
| 4 | ResembleAI/chatterbox (None, ~0.12 GB) | Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice (1.9B, ~1.15 GB) |
| 5 | SWivid/F5-TTS (None, ~0.84 GB) | openbmb/VoxCPM2 (2.3B, ~1.61 GB) |
| 6 | microsoft/VibeVoice-Realtime-0.5B (1B, ~1.3 GB) | pnnbao-ump/VieNeu-TTS-v3-Turbo (131M, ~0.17 GB) |
| 7 | fishaudio/s2-pro (4.6B, ~0.51 GB) | neuphonic/neutts-air (748M, ~0.51 GB) |
| 8 | coqui/XTTS-v2 (None, ~0.28 GB) | BreezeBlue/Breeze-TTS-2 (3.5B, ~2.33 GB) |
| 9 | Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice (1.9B, ~1.15 GB) | Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice (906M, ~1.01 GB) |
| 10 | openbmb/VoxCPM2 (2.3B, ~1.61 GB) | microsoft/VibeVoice-1.5B (2.7B, ~1.97 GB) |

Not an apples-to-apples test: the "before" index held only the 2,560 most-downloaded models, which made popular
picks easy. On the full index the first Qwen3 run (before ranking fixes) surfaced obscure TTS repos and missed
Kokoro-82M entirely. That was not the embedding: with hundreds of phone-ready TTS models, every one "turns text
into speech", so the top-150 text matches were close to random among them. Fixes made alongside the switch:

- Candidate lists now include "most-downloaded among the top 1,000 semantic matches" and "… among the top 1,000
  keyword matches" (Qdrant nested prefetch + order_by), fused with RRF.
- Device/size words ("on my phone", "small", "MacBook", "8GB VRAM") are stripped from the text used for
  matching — filters already enforce them, and left in they pull in models that merely say "small"/"edge".
- Converted copies (MLX/ONNX/Core ML repos, ComfyUI repackages) are linked to their original more reliably, and
  copies whose size doesn't match the original no longer set the family's minimum memory.
- Draft/speculative-decoding heads (DFlash, DSpark, EAGLE, MTP) and barely-used old repos are penalized.

Dense-only top 10 (no filters) for the same question: see before_dense_only.txt / after_dense_only.txt.
All three acceptance questions after the change: three_questions_after.txt.

Cost: Qwen3 embeds ~15–20 descriptions/s on the M1 Pro GPU (bge-small: ~60/s on CPU when idle). First build gives
semantic vectors to the ~33k most-used/new models (~30 min); `model-finder index --backfill` does the remaining
~86k (~1.5 h). Vectors are cached by text hash, so updates only embed new/changed descriptions. A fresh CLI
process spends ~10–15 s loading the model on the first question; the web server loads it once at startup.

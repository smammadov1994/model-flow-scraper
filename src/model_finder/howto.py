"""Pick the best variant/format for a device and say, in one line, how to run it. Pure functions."""

from __future__ import annotations

from typing import Any

from . import enrich

# device -> RAM budget in GB (None = no cap)
DEVICE_BUDGET: dict[str, float | None] = {
    "phone": enrich.TIER_BUDGET_GB["phone"],
    "laptop": enrich.TIER_BUDGET_GB["laptop"],
    "mac": enrich.TIER_BUDGET_GB["laptop"],
    "gpu": enrich.TIER_BUDGET_GB["gpu"],
    "server": None,
    "any": None,
}
_ALIASES = {
    "iphone": "phone", "android": "phone", "mobile": "phone", "ios": "phone",
    "macbook": "mac", "macos": "mac", "apple": "mac",
    "desktop": "laptop", "pc": "laptop", "cpu": "laptop",
    "cuda": "gpu", "nvidia": "gpu",
}

GGUF_PREF = [
    "Q4_K_M", "Q4_K_S", "Q4_0", "IQ4_XS", "Q4_K_XL", "IQ4_NL", "Q4_1",
    "Q5_K_M", "Q5_K_S", "Q5_K_XL", "Q5_0", "Q5_1", "Q6_K", "Q6_K_XL", "Q8_0", "Q8_K_XL",
    "Q3_K_L", "Q3_K_M", "Q3_K_XL", "Q3_K_S", "IQ3_M", "IQ3_S", "IQ3_XS", "IQ3_XXS", "MXFP4",
    "BF16", "F16", "F32",
]
_ONNX_PHONE = ["q4", "int8", "q4f16", "fp16", "bnb4", "fp32"]
_ONNX_OTHER = ["fp16", "fp32", "int8", "q4f16", "q4", "bnb4"]

_ALL_ORDER = ["diffusers", "safetensors", "pytorch", "awq", "gptq", "exl2", "bnb", "nemo", "gguf", "onnx",
              "openvino", "mlx", "ggml", "coreml", "executorch", "tflite", "mlc"]
_ORDERS: dict[tuple[str, str], list[str]] = {
    ("phone", "llm"): ["gguf", "mlc", "executorch", "tflite", "mlx", "onnx", "coreml", "ggml"],
    ("phone", "asr"): ["ggml", "onnx", "coreml", "gguf", "tflite", "executorch", "mlx", "mlc"],
    ("phone", "gen"): ["coreml", "onnx", "tflite", "executorch", "mlc", "gguf", "mlx", "ggml"],
    ("phone", "other"): ["onnx", "gguf", "ggml", "coreml", "tflite", "executorch", "mlx", "mlc"],
    ("laptop", "llm"): ["gguf", "onnx", "safetensors", "pytorch", "openvino", "mlx", "coreml"],
    ("laptop", "asr"): ["ggml", "onnx", "gguf", "safetensors", "pytorch", "openvino", "mlx", "coreml"],
    ("laptop", "gen"): ["diffusers", "safetensors", "pytorch", "gguf", "onnx", "openvino", "coreml", "mlx"],
    ("laptop", "other"): ["onnx", "gguf", "safetensors", "pytorch", "openvino", "mlx", "coreml"],
    ("mac", "llm"): ["mlx", "gguf", "coreml", "onnx", "safetensors", "pytorch", "openvino"],
    ("mac", "asr"): ["mlx", "ggml", "coreml", "onnx", "gguf", "safetensors", "pytorch", "openvino"],
    ("mac", "gen"): ["diffusers", "coreml", "mlx", "gguf", "onnx", "safetensors", "pytorch", "openvino"],
    ("mac", "other"): ["mlx", "onnx", "coreml", "gguf", "safetensors", "pytorch", "openvino"],
}


def _norm_device(device: str | None) -> str:
    d = (device or "any").strip().lower()
    d = _ALIASES.get(d, d)
    return d if d in DEVICE_BUDGET else "any"


def _cat(task: str | None) -> str:
    if task in enrich.LLM_TASKS:
        return "llm"
    if task == "automatic-speech-recognition":
        return "asr"
    if enrich.TASK_GROUPS.get(task or "") in ("image-generation", "video", "vision"):
        return "gen"
    return "other"


def _format_order(dev: str, cat: str) -> list[str]:
    order = _ORDERS.get((dev, cat)) or _ALL_ORDER
    return order + [f for f in _ALL_ORDER if f not in order]


def _allowed(dev: str, fmt: str) -> bool:
    if dev == "phone":
        return fmt in enrich.PHONE_FORMATS
    if dev in ("laptop", "mac"):
        return fmt in enrich.TIER_FORMATS["laptop"]
    return fmt in enrich.ALL_FORMATS


def _variant_rank(fmt: str, variant: str, dev: str) -> int:
    if fmt == "gguf":
        if variant in GGUF_PREF:
            return GGUF_PREF.index(variant)
        return 100 if not enrich.practical(variant) else 50
    if fmt == "onnx":
        order = _ONNX_PHONE if dev == "phone" else _ONNX_OTHER
        return order.index(variant) if variant in order else 50
    return 0


def _candidates(record: dict) -> list[tuple[str, str, float | None]]:
    """(variant, format, ram_gb or None if unknown) for every way to run this repo."""
    ram = record.get("ram_gb")
    if not isinstance(ram, dict):
        ram = enrich.ram_variants(record)
    fmts = record.get("formats") or []
    out = []
    for v, f in enrich.variant_formats(record).items():
        r = ram.get(v)
        if f == "native":
            for nf in enrich.native_formats(fmts) or ["native"]:
                out.append((v, nf, r))
        else:
            out.append((v, f, r))
    return out


def _best(record: dict, dev: str) -> tuple[str | None, str | None]:
    budget = DEVICE_BUDGET[dev]
    order = _format_order(dev, _cat(record.get("task")))
    best = None
    for v, f, r in _candidates(record):
        if not _allowed(dev, f):
            continue
        if budget is not None and r is not None and r > budget:
            continue
        rank = (order.index(f) if f in order else 99, _variant_rank(f, v, dev), r or 0.0)
        if best is None or rank < best[0]:
            best = (rank, v, f)
    return (best[1], best[2]) if best else (None, None)


def best_variant(record: dict, device: str) -> tuple[str | None, str | None]:
    """(variant, format) preferred for `device` that fits its memory budget, else (None, None)."""
    return _best(record if isinstance(record, dict) else {}, _norm_device(device))


# ----------------------------------------------------------------------------- hints

_LABEL = {
    "gguf": "GGUF", "ggml": "ggml", "onnx": "ONNX", "coreml": "Core ML", "executorch": "ExecuTorch",
    "tflite": "LiteRT/TFLite", "mlx": "MLX", "mlc": "MLC", "safetensors": "safetensors",
    "pytorch": "PyTorch", "diffusers": "diffusers", "openvino": "OpenVINO", "awq": "AWQ", "gptq": "GPTQ",
    "exl2": "EXL2", "bnb": "bitsandbytes", "nemo": "NeMo",
}


def _hint(fmt: str, variant: str | None, dev: str, task: str | None, rid: str, library: str | None,
          derived: bool) -> str | None:
    label = _LABEL.get(fmt, fmt)
    llm = task in enrich.LLM_TASKS
    lead = f"Use the {label} build {rid}" if derived else f"Use the {label} files"
    phone = dev == "phone"

    if fmt == "gguf":
        quant = variant if variant and variant != "native" else "Q4_K_M"
        group = enrich.TASK_GROUPS.get(task or "")
        src = f" from {rid}" if derived else ""
        if group in ("image-generation", "video"):
            return (f"Download the {quant} .gguf{src} and load it in ComfyUI (ComfyUI-GGUF node) or "
                    "stable-diffusion.cpp — GGUF image/video models don't run in llama.cpp or Ollama.")
        if not llm and task not in (None, "feature-extraction", "sentence-similarity", "text-ranking"):
            return (f"Download the {quant} .gguf{src} and run it with the runtime named on the model card "
                    "(speech/audio GGUFs need their own llama.cpp-based tool, not chat apps).")
        if phone:
            dl = f"Download the {quant} .gguf" + (f" from {rid}" if derived else "")
            what = "open it in" if llm else "run it with llama.cpp or"
            return (f"{dl} and {what} an on-device llama.cpp app "
                    "(PocketPal AI, LLM Farm on iOS / ChatterUI on Android).")
        if dev in ("gpu", "server", "any"):
            return f"{lead}: `llama-server -hf {rid}:{quant}` (llama.cpp), or `ollama run hf.co/{rid}:{quant}`."
        return f"`ollama run hf.co/{rid}:{quant}` or open it in LM Studio / llama.cpp."
    if fmt == "onnx":
        if phone:
            tail = (f"use the {variant} variant to save memory" if variant in ("q4", "q4f16", "int8", "bnb4")
                    else "pick the q4/int8 variant to save memory")
            return f"{lead} with ONNX Runtime Mobile or sherpa-onnx (Android/iOS); {tail}."
        tail = f"; the {variant} variant is a good default" if variant else ""
        return (f"{lead} with ONNX Runtime (`pip install onnxruntime`), Transformers.js or "
                f"Optimum{tail}.")
    if fmt == "ggml":
        if task == "automatic-speech-recognition":
            if phone:
                return "whisper.cpp runs this on iOS/Android; use the q5/q8 .bin."
            return "Run it with whisper.cpp (`whisper-cli -m model.bin -f audio.wav`); use the q5/q8 .bin."
        return f"{lead} with its matching C/C++ runtime (whisper.cpp for Whisper models; see the model card)."
    if fmt == "coreml":
        if phone:
            return "Add the Core ML package to an iOS app (Xcode) — runs on the Neural Engine."
        return "Open the Core ML package in Xcode, or load it with coremltools on a Mac."
    if fmt == "executorch":
        return "Load the .pte file with the ExecuTorch runtime in your Android/iOS app."
    if fmt == "tflite":
        return "Run with Google LiteRT / MediaPipe (e.g. AI Edge Gallery on Android)."
    if fmt == "mlc":
        return "Run it with MLC LLM (the MLC Chat app on Android/iOS, or `mlc_llm chat`)."
    if fmt == "mlx":
        if phone:
            return "Runs in iOS apps built on mlx-swift (e.g. the MLX examples)."
        if task == "text-to-speech":
            return f"`pip install mlx-audio` and run `mlx_audio.tts.generate --model {rid} --text 'hi'`."
        if task == "automatic-speech-recognition":
            return f"`pip install mlx-audio` and run `mlx_audio.stt.generate --model {rid} --audio audio.wav`."
        if task in ("image-text-to-text", "visual-question-answering"):
            return (f"`pip install mlx-vlm` then `python -m mlx_vlm.generate --model {rid} "
                    "--prompt 'describe this' --image photo.jpg`.")
        if llm:
            return f"`pip install mlx-lm` then `mlx_lm.generate --model {rid} --prompt 'hi'`."
        return None
    if fmt == "openvino":
        return f"{lead} with OpenVINO (`pip install optimum[openvino]`, then `OVModelForCausalLM` or the Optimum pipelines)."
    if fmt in ("awq", "gptq"):
        return f"Serve with vLLM (`vllm serve {rid}`); it picks up the {label} quantization automatically."
    if fmt == "exl2":
        return "Load the EXL2 weights with ExLlamaV2 or TabbyAPI on an NVIDIA GPU."
    if fmt == "bnb":
        return "`pip install transformers bitsandbytes` and load with `from_pretrained(..., load_in_4bit=True)` on an NVIDIA GPU."
    if fmt == "nemo":
        return f"`pip install nemo_toolkit[asr]` and load with `ASRModel.from_pretrained('{rid}')`."
    if fmt == "diffusers" or (fmt in ("safetensors", "pytorch") and (library or "").lower() == "diffusers"):
        return (f"`pip install diffusers` and load with "
                f"`DiffusionPipeline.from_pretrained('{rid}', torch_dtype=torch.float16)`.")
    if fmt in ("safetensors", "pytorch"):
        lib = (library or "").lower()
        if lib == "sentence-transformers":
            return f"`pip install sentence-transformers` then `SentenceTransformer('{rid}')`."
        if dev in ("gpu", "server", "any") and llm:
            return f"Serve with vLLM (`vllm serve {rid}`) or transformers."
        if task and lib in ("", "transformers"):
            return f"`pip install transformers torch` then `pipeline('{task}', model='{rid}')`."
    return None


def _fallback(record: dict) -> str:
    lib = record.get("library")
    if isinstance(lib, str) and lib.strip():
        return f"Load with the `{lib.strip()}` library (see the model card)."
    return "See the model card for how to run it."


def _derivative_record(base: dict, d: dict) -> dict:
    ram = d.get("ram_gb")
    pseudo = {
        "id": d.get("id"), "task": base.get("task"), "params": base.get("params"),
        "formats": d.get("formats") or [], "gguf_quants": d.get("gguf_quants") or [],
        "onnx_variants": d.get("onnx_variants") or [], "quant_bits": d.get("quant_bits"),
        "native_dtype": None,
    }
    if isinstance(ram, dict):
        pseudo["ram_gb"] = ram
    else:
        est = enrich.ram_variants(pseudo)
        if isinstance(ram, (int, float)) and not isinstance(ram, bool) and est:
            # a single number = the derivative's smallest RAM; use it to gate every variant
            est = {k: float(ram) for k in est}
        pseudo["ram_gb"] = est
    return pseudo


def run_hint(record: dict, device: str) -> str:
    """One line saying how to run this model on `device`; may point to a derivative repo."""
    record = record if isinstance(record, dict) else {}
    dev = _norm_device(device)
    task = record.get("task")
    rid = record.get("id") or ""
    variant, fmt = _best(record, dev)
    if fmt:
        h = _hint(fmt, variant, dev, task, rid, record.get("library"), derived=False)
        if h:
            return h

    derivs = record.get("derivatives")
    best = None
    order = _format_order(dev, _cat(task))
    for d in derivs if isinstance(derivs, list) else []:
        if not isinstance(d, dict) or not d.get("id"):
            continue
        v, f = _best(_derivative_record(record, d), dev)
        if not f:
            continue
        rank = (order.index(f) if f in order else 99, _variant_rank(f, v or "", dev))
        h = _hint(f, v, dev, task, d["id"], d.get("library") or record.get("library"), derived=True)
        if h and (best is None or rank < best[0]):
            best = (rank, h)
    if best:
        return best[1]
    return _fallback(record)


def hint_for(record: dict, fmt: str, variant: str | None, device: str, repo: str | None = None) -> str | None:
    """Hint for an already-chosen (format, variant, repo) — keeps the instructions consistent with the pick."""
    rid = repo or record.get("id") or ""
    return _hint(fmt, variant, _norm_device(device), record.get("task"), rid, record.get("library"),
                 derived=bool(repo and repo != record.get("id")))

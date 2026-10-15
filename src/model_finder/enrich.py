"""Raw Hugging Face row -> enriched Record. Pure functions, no network.

See docs/SPEC.md. Public entry point: ``build_record(raw)``.

RAM formula (see SPEC):

    weights_gb = params * bpw / 8 / 1e9
    ram_gb     = weights_gb * 1.20            # runtime overhead
               + 0.17 * sqrt(params_in_B)     # fp16 KV cache for ~4k ctx, LLM-ish tasks only
               + 1.0                          # latents/VAE decode, image/video generation only
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Iterable

# --------------------------------------------------------------------------- constants

BPW: dict[str, float] = {
    "IQ1_S": 1.56, "IQ1_M": 1.75, "IQ2_XXS": 2.06, "IQ2_XS": 2.31, "IQ2_S": 2.5, "IQ2_M": 2.7, "Q2_K": 2.63,
    "IQ3_XXS": 3.06, "IQ3_XS": 3.3, "IQ3_S": 3.44, "IQ3_M": 3.66, "Q3_K_S": 3.5, "Q3_K_M": 3.91,
    "Q3_K_L": 4.27, "Q3_K_XL": 4.3,
    "IQ4_XS": 4.25, "IQ4_NL": 4.5, "Q4_0": 4.55, "Q4_1": 5.0, "Q4_K_S": 4.58, "Q4_K_M": 4.85,
    "Q4_K_XL": 5.0, "MXFP4": 4.25,
    "Q5_0": 5.54, "Q5_1": 6.0, "Q5_K_S": 5.54, "Q5_K_M": 5.69, "Q5_K_XL": 5.8, "Q6_K": 6.59,
    "Q6_K_XL": 6.8, "Q8_0": 8.5, "Q8_K_XL": 9.0,
    "TQ1_0": 1.69, "TQ2_0": 2.06, "F16": 16, "BF16": 16, "F32": 32,
    "int4": 4.5, "q4": 4.5, "q4f16": 4.5, "bnb4": 4.5, "int8": 8.5, "fp16": 16, "bf16": 16, "fp32": 32,
    "fp8": 8.5,
}

KNOWN_TASKS: set[str] = {
    "text-generation", "text2text-generation", "fill-mask", "text-classification", "token-classification",
    "question-answering", "summarization", "translation", "zero-shot-classification", "sentence-similarity",
    "feature-extraction", "text-ranking", "table-question-answering", "image-classification",
    "object-detection", "image-segmentation", "depth-estimation", "image-to-image", "text-to-image",
    "unconditional-image-generation", "image-to-text", "image-text-to-text", "visual-question-answering",
    "document-question-answering", "zero-shot-image-classification", "zero-shot-object-detection",
    "mask-generation", "image-feature-extraction", "keypoint-detection", "video-classification",
    "text-to-video", "image-to-video", "video-to-video", "video-text-to-text", "text-to-3d", "image-to-3d",
    "automatic-speech-recognition", "audio-classification", "text-to-speech", "text-to-audio",
    "audio-to-audio", "voice-activity-detection", "audio-text-to-text", "any-to-any",
    "reinforcement-learning", "robotics", "tabular-classification", "tabular-regression",
    "time-series-forecasting", "graph-ml", "other",
}


def _groups(modality: str, tasks: str) -> dict[str, str]:
    return {t: modality for t in tasks.split()}


TASK_GROUPS: dict[str, str] = {
    **_groups("text", "text-generation text2text-generation summarization translation question-answering "
                      "fill-mask text-classification token-classification zero-shot-classification "
                      "table-question-answering"),
    **_groups("vision", "image-classification object-detection image-segmentation depth-estimation "
                        "zero-shot-image-classification zero-shot-object-detection image-feature-extraction "
                        "mask-generation keypoint-detection video-classification"),
    **_groups("image-generation", "text-to-image image-to-image unconditional-image-generation image-to-3d "
                                  "text-to-3d"),
    **_groups("video", "text-to-video image-to-video video-to-video"),
    **_groups("speech-in", "automatic-speech-recognition audio-classification voice-activity-detection"),
    **_groups("speech-out", "text-to-speech text-to-audio audio-to-audio"),
    **_groups("multimodal", "image-text-to-text visual-question-answering image-to-text any-to-any "
                            "video-text-to-text audio-text-to-text document-question-answering"),
    **_groups("embedding", "feature-extraction sentence-similarity text-ranking"),
    **_groups("other", "reinforcement-learning robotics tabular-classification tabular-regression "
                       "time-series-forecasting graph-ml other"),
}

TIER_BUDGET_GB: dict[str, float] = {"phone": 4.0, "laptop": 12.0, "gpu": 24.0}
TIER_ORDER = ("phone", "laptop", "gpu", "server")

PHONE_FORMATS: frozenset[str] = frozenset(
    {"gguf", "ggml", "onnx", "coreml", "executorch", "tflite", "mlx", "mlc"})
_LAPTOP_FORMATS = PHONE_FORMATS | {"safetensors", "pytorch", "diffusers", "openvino"}
ALL_FORMATS: frozenset[str] = frozenset(
    _LAPTOP_FORMATS | {"awq", "gptq", "exl2", "bnb", "nemo"})
TIER_FORMATS: dict[str, frozenset[str]] = {
    "phone": PHONE_FORMATS,
    "laptop": frozenset(_LAPTOP_FORMATS),
    "gpu": ALL_FORMATS,
    "server": ALL_FORMATS,
}
# Formats that make an un-quantized ("native") variant runnable on a phone.
PHONE_NATIVE_FORMATS: frozenset[str] = frozenset({"coreml", "executorch", "tflite", "mlc", "ggml", "mlx"})
# Formats whose weights are described by a specific variant rather than "native".
_VARIANT_ONLY_FORMATS = frozenset({"gguf", "onnx", "awq", "gptq", "exl2", "bnb"})

PERMISSIVE_LICENSES: frozenset[str] = frozenset({
    "apache-2.0", "mit", "cc-by-4.0", "cc0-1.0", "bigscience-openrail-m", "creativeml-openrail-m",
    "mpl-2.0", "unlicense", "cc-by-sa-4.0", "agpl-3.0", "afl-3.0", "artistic-2.0", "isc", "wtfpl",
    "ecl-2.0", "zlib",
})
_PERMISSIVE_PREFIXES = ("bsd-", "openrail", "gpl-", "lgpl-", "agpl-")

LLM_TASKS = frozenset({"text-generation", "text2text-generation", "image-text-to-text", "any-to-any",
                       "video-text-to-text", "audio-text-to-text", "visual-question-answering"})
GEN_TASKS = frozenset({"text-to-image", "image-to-image", "text-to-video", "image-to-video",
                       "video-to-video", "unconditional-image-generation"})

OVERHEAD = 1.20
GEN_EXTRA_GB = 1.0

# --------------------------------------------------------------------------- small coercion helpers


def _dict(x: Any) -> dict:
    return x if isinstance(x, dict) else {}


def _str(x: Any) -> str | None:
    if isinstance(x, str):
        x = x.strip()
        return x or None
    return None


def _str_list(x: Any) -> list[str]:
    """str -> [str]; list -> its non-empty str items; anything else -> []."""
    if isinstance(x, str):
        return [x.strip()] if x.strip() else []
    if isinstance(x, (list, tuple, set)):
        return [i.strip() for i in x if isinstance(i, str) and i.strip()]
    return []


def _num(x: Any, default: float = 0) -> float:
    try:
        if x is None or isinstance(x, bool):
            return default
        v = float(x)
        return default if math.isnan(v) or math.isinf(v) else v
    except (TypeError, ValueError):
        return default


def _int(x: Any, default: int = 0) -> int:
    return int(_num(x, default))


def _files(siblings: Any) -> list[str]:
    out: list[str] = []
    if not isinstance(siblings, (list, tuple)):
        return out
    for s in siblings:
        if isinstance(s, str):
            out.append(s)
        elif isinstance(s, dict) and isinstance(s.get("rfilename"), str):
            out.append(s["rfilename"])
    return out


def _base(path: str) -> str:
    return path.rsplit("/", 1)[-1]


# --------------------------------------------------------------------------- formats

_QUANT_PATTERNS = [
    re.compile(r"(?<![A-Z0-9])(IQ[1-4]_(?:XXS|XS|S|M|NL))(?![A-Z0-9])"),
    re.compile(r"(?<![A-Z0-9])(Q[2-8]_K(?:_(?:XL|[SML]))?)(?![A-Z0-9])"),
    re.compile(r"(?<![A-Z0-9])(Q[4-8]_[01])(?![A-Z0-9])"),
    re.compile(r"(?<![A-Z0-9])(TQ[12]_0)(?![A-Z0-9])"),
    re.compile(r"(?<![A-Z0-9])(MXFP4)(?![A-Z0-9])"),
    re.compile(r"(?<![A-Z0-9])(BF16)(?![A-Z0-9])"),
    re.compile(r"(?<![A-Z0-9])(F16)(?![A-Z0-9])"),
    re.compile(r"(?<![A-Z0-9])(F32)(?![A-Z0-9])"),
]
_NONWEIGHT_BINS = {"training_args.bin", "optimizer.bin", "scheduler.bin", "rng_state.bin"}


def _is_coreml_path(p: str) -> bool:
    return ".mlpackage/" in p or ".mlmodelc" in p or p.endswith(".mlmodel") or ".mlpackage" in p


def detect_formats(files: list[str], tags: list[str], library: str | None,
                   author: str | None, repo_id: str) -> list[str]:
    """Formats present in this repo (sorted, unique)."""
    out: set[str] = set()
    low = [f.lower() for f in _str_list(files)]
    tagl = {t.lower() for t in _str_list(tags)}
    lib = (library or "").lower()
    auth = (author or "").lower()
    name = (repo_id or "").lower()

    for p in low:
        b = _base(p)
        if b.endswith(".gguf"):
            out.add("gguf")
        if (b.startswith("ggml-") and b.endswith(".bin")) or b.endswith(".ggml"):
            out.add("ggml")
        if b.endswith(".onnx"):
            out.add("onnx")
        if _is_coreml_path(p):
            out.add("coreml")
        if b.endswith(".pte"):
            out.add("executorch")
        if b.endswith((".tflite", ".litertlm", ".task")):
            out.add("tflite")
        if b == "openvino_model.xml":
            out.add("openvino")
        if b == "mlc-chat-config.json":
            out.add("mlc")
        if b.endswith(".safetensors"):
            out.add("safetensors")
        if b.endswith((".pt", ".pth", ".ckpt")) and not _is_coreml_path(p):
            out.add("pytorch")
        if (b.endswith(".bin") and not b.startswith("ggml-") and b not in _NONWEIGHT_BINS
                and not _is_coreml_path(p)):
            out.add("pytorch")
        if b == "model_index.json":
            out.add("diffusers")
        if b.endswith(".nemo"):
            out.add("nemo")

    if "mlx" in tagl or lib in ("mlx", "mlx-lm", "mlx-audio", "mlx-vlm") or auth == "mlx-community":
        out.add("mlx")
    if "openvino" in tagl:
        out.add("openvino")
    if "mlc-llm" in tagl:
        out.add("mlc")
    if lib == "diffusers":
        out.add("diffusers")
    if lib == "nemo":
        out.add("nemo")

    for kind in ("awq", "gptq", "exl2"):
        if re.search(r"(?<![a-z0-9])" + kind, name) or any(kind in t for t in tagl):
            out.add(kind)
    if any("bnb" in t or "bitsandbytes" in t for t in tagl) or re.search(r"(?<![a-z0-9])(bnb|bitsandbytes)", name):
        out.add("bnb")
    return sorted(out)


def _quant_rank(q: str) -> tuple[float, str]:
    return (BPW.get(q, 99.0), q)


def gguf_quants(files: list[str]) -> list[str]:
    """Quant names from ``*.gguf`` basenames, upper-case, ordered by bits per weight."""
    found: set[str] = set()
    for f in _str_list(files):
        b = _base(f)
        if not b.lower().endswith(".gguf") or "mmproj" in b.lower():
            continue
        up = b.upper()
        for pat in _QUANT_PATTERNS:
            m = pat.search(up)
            if m:
                found.add(m.group(1))
                break
    return sorted(found, key=_quant_rank)


_ONNX_ORDER = ["fp32", "fp16", "int8", "q4f16", "q4", "bnb4"]


def _onnx_variant(basename: str) -> str:
    stem = basename.lower()[:-len(".onnx")]
    toks = set(re.split(r"[^a-z0-9]+", stem))
    if "q4f16" in toks:
        return "q4f16"
    if toks & {"fp16", "f16", "half"}:
        return "fp16"
    if toks & {"q4", "int4", "uint4"}:
        return "q4"
    if "bnb4" in toks:
        return "bnb4"
    if toks & {"int8", "quantized", "uint8", "q8"}:
        return "int8"
    return "fp32"


def onnx_variants(files: list[str]) -> list[str]:
    found = {_onnx_variant(_base(f)) for f in _str_list(files) if f.lower().endswith(".onnx")}
    return [v for v in _ONNX_ORDER if v in found]


_NAME_BITS_RE = re.compile(r"(?<![a-z0-9])(\d+(?:\.\d+)?)-?bit(?![a-z0-9])")
_NAME_FP_RE = re.compile(r"(?<![a-z0-9])(?:nv)?(?:mx)?(fp|int)([48])(?![0-9])")


def quant_bits(repo_id: str, tags: list[str], formats: list[str]) -> float | None:
    """Bits per weight if the repo itself is a fixed quant, else None."""
    name = (repo_id or "").lower().split("/")[-1]
    fmts = set(formats or [])
    m = _NAME_BITS_RE.search(name)
    if m:
        return float(m.group(1))
    m = _NAME_FP_RE.search(name)
    if m:
        return float(m.group(2))
    tagl = {t.lower() for t in _str_list(tags)}
    if "4-bit" in tagl or "4bit" in tagl:
        return 4.0
    if "8-bit" in tagl or "8bit" in tagl:
        return 8.0
    if fmts & {"awq", "gptq"}:
        return 4.0
    if "mlx" in fmts and re.search(r"(?<![a-z0-9])bf16(?![a-z0-9])", name):
        return 16.0
    return None


def native_formats(formats: Iterable[str] | None) -> list[str]:
    """Formats whose weights are un-quantized ("native" precision) variants."""
    return sorted(f for f in (formats or []) if f not in _VARIANT_ONLY_FORMATS)


# --------------------------------------------------------------------------- task / modality


def _tokens(s: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", (s or "").lower()) if t]


_LIBRARY_TASK: dict[str, str | None] = {
    "sentence-transformers": "sentence-similarity",
    "open_clip": "zero-shot-image-classification", "open-clip": "zero-shot-image-classification",
    "pyannote-audio": "voice-activity-detection",
    "speechbrain": "automatic-speech-recognition",
    "nemo": None, "peft": None,
    "mlx-audio": "text-to-speech",
    "timm": "image-classification",
    "ultralytics": "object-detection",
    "paddleocr": "image-to-text",
    "f5-tts": "text-to-speech", "coqui": "text-to-speech", "kokoro": "text-to-speech",
    "chatterbox": "text-to-speech",
}

_TTS_EXACT = {"tts", "bark", "csm"}
_TTS_PREFIX = ("kokoro", "xtts", "f5", "piper", "parler", "vits", "voice", "cosyvoice", "chatterbox",
               "orpheus", "sesame")
_ASR_EXACT = {"asr", "stt"}
_ASR_PREFIX = ("whisper", "wav2vec", "parakeet", "canary", "moonshine")
_EMB_EXACT = {"e5", "bge", "gte"}
_EMB_PREFIX = ("embed", "minilm")
_IMG_PREFIX = ("sdxl", "flux", "diffusion")
_TXT_PREFIX = ("instruct", "chat", "coder", "gpt")


def _tok_match(toks: list[str], exact: set[str], prefix: tuple[str, ...]) -> bool:
    return any(t in exact or t.startswith(prefix) for t in toks)


def _task_from_name(name: str) -> str | None:
    n = name.lower().replace("_", "-")
    toks = _tokens(n)
    if "text-to-speech" in n or _tok_match(toks, _TTS_EXACT, _TTS_PREFIX):
        return "text-to-speech"
    if "speech-to-text" in n or _tok_match(toks, _ASR_EXACT, _ASR_PREFIX):
        return "automatic-speech-recognition"
    if _tok_match(toks, _EMB_EXACT, _EMB_PREFIX):
        return "feature-extraction"
    if "text-to-image" in n or _tok_match(toks, set(), _IMG_PREFIX):
        return "text-to-image"
    if _tok_match(toks, {"llm"}, _TXT_PREFIX) or (toks and toks[-1] == "it"):
        return "text-generation"
    return None


def _task_from_arch(cfg: dict) -> str | None:
    archs = _str_list(cfg.get("architectures"))
    mt = (_str(cfg.get("model_type")) or "").lower()
    for a in archs:
        al = a.lower()
        if al.endswith("forcausallm") or al.endswith("lmheadmodel"):
            return "text-generation"
        if al.endswith("forconditionalgeneration"):
            if "whisper" in al or mt == "whisper":
                return "automatic-speech-recognition"
            if any(k in al for k in ("t5", "bart", "pegasus", "marian")):
                return "text2text-generation"
            continue
        if al.endswith(("forspeechseq2seq", "forctc")):
            return "automatic-speech-recognition"
        if al.endswith("forsequenceclassification"):
            return "text-classification"
        if al.endswith("fortokenclassification"):
            return "token-classification"
        if al.endswith("forquestionanswering"):
            return "question-answering"
        if al.endswith("formaskedlm"):
            return "fill-mask"
        if al.endswith("forimageclassification"):
            return "image-classification"
        if al.endswith("forobjectdetection"):
            return "object-detection"
    return None


def _diffusers_task(class_name: str) -> str:
    c = class_name.lower()
    if "video" in c:
        return "text-to-video"
    if "img2img" in c or "inpaint" in c:
        return "image-to-image"
    return "text-to-image"


def infer_task(raw: dict) -> tuple[str | None, str]:
    raw = _dict(raw)
    card = _dict(raw.get("card"))
    cfg = _dict(raw.get("config"))
    tags = _str_list(raw.get("tags")) + _str_list(card.get("tags"))

    pt = _str(raw.get("pipeline_tag"))
    if pt:
        return pt, "pipeline_tag"
    pt = _str(_dict(raw.get("transformers_info")).get("pipeline_tag"))
    if pt:
        return pt, "transformers_info"
    pt = _str(card.get("pipeline_tag"))
    if pt:
        return pt, "card"
    for t in tags:
        if t in KNOWN_TASKS and t != "other":
            return t, "tags"
    bt = _str(raw.get("base_task"))
    if bt:
        return bt, "base_model"

    lib = (_str(raw.get("library_name")) or _str(card.get("library_name")) or "").lower()
    diff_cls = _str(_dict(cfg.get("diffusers")).get("_class_name")) or ""
    if lib == "diffusers":
        return _diffusers_task(diff_cls), "library"
    if _LIBRARY_TASK.get(lib):
        return _LIBRARY_TASK[lib], "library"

    t = _task_from_arch(cfg)
    if t:
        return t, "config"
    if diff_cls and "pipeline" in diff_cls.lower():
        return _diffusers_task(diff_cls), "config"

    rid = _str(raw.get("id")) or ""
    t = _task_from_name(rid.split("/")[-1])
    if t:
        return t, "name"
    return None, "unknown"


_CODE_EXACT = {"code", "coder", "coding"}
_CODE_PREFIX = ("starcoder", "codestral", "codellama", "codegen", "codegemma", "codeqwen", "codet5",
                "codebert")
_MUSIC_TOKENS = {"music", "musicgen", "sfx", "audiogen", "audioldm", "sound"}


def _has_code(tags: list[str], name: str) -> bool:
    cand = [name or ""] + [t for t in tags if not t.lower().startswith(("dataset:", "arxiv:"))]
    for s in cand:
        if any(t in _CODE_EXACT or t.startswith(_CODE_PREFIX) for t in _tokens(s)):
            return True
    return False


def _has_music(tags: list[str], name: str) -> bool:
    cand = [name or ""] + [t for t in tags if not t.lower().startswith(("dataset:", "arxiv:"))]
    return any(t in _MUSIC_TOKENS for s in cand for t in _tokens(s))


def modality(task: str | None, tags: list[str], name: str) -> str:
    tags = _str_list(tags)
    g = TASK_GROUPS.get(task or "", "other")
    if g == "text" and _has_code(tags, name or ""):
        return "code"
    if task in ("text-to-audio", "audio-to-audio", "audio-classification") and _has_music(tags, name or ""):
        return "audio"
    return g


# --------------------------------------------------------------------------- params

_NAME_PARAMS_RE = re.compile(
    r"(?<![A-Za-z0-9])(?<!\d\.)(?:(\d+)[xX])?(\d+(?:\.\d+)?)([BbMm])(?![A-Za-z0-9])")


def params_from_name(name: str) -> int | None:
    """Parse ``7B``/``1.5b``/``82M``/``8x7B`` tokens delimited by - _ . or string boundaries."""
    if not isinstance(name, str):
        return None
    name = name.split("/")[-1]
    for m in _NAME_PARAMS_RE.finditer(name):
        mult = int(m.group(1)) if m.group(1) else 1
        val = float(m.group(2))
        unit = 1e9 if m.group(3) in "Bb" else 1e6
        n = int(round(mult * val * unit))
        if 1_000_000 <= n <= 10_000_000_000_000:
            return n
    return None


_HINT_BYTES = [  # (token set, bytes per param)
    ({"fp32", "f32", "float32"}, 4.0),
    ({"bf16", "bfloat16", "fp16", "f16", "float16", "half"}, 2.0),
    ({"fp8", "float8", "int8", "q8", "uint8", "i8", "u8"}, 1.0),
    ({"q4", "int4", "fp4", "bnb4", "nf4", "uint4"}, 0.5),
]


def _hint_bytes(text: str | None) -> float | None:
    if not text:
        return None
    toks = set(_tokens(text.replace("torch.", "")))
    for names, b in _HINT_BYTES:
        if toks & names:
            return b
    return None


def params_from_size(total_bytes: float, dtype_hint: str | None, filename: str | None = None) -> int:
    """Params from file size: file-name hint, then dtype hint, then 2 B (safetensors) / 4 B (legacy)."""
    b = _hint_bytes(filename)
    if b is None:
        b = _hint_bytes(dtype_hint)
    if b is None:
        fn = (filename or "").lower()
        b = 4.0 if fn.endswith((".bin", ".pth", ".pt", ".ckpt")) else 2.0
    return int(_num(total_bytes) / b)


def dtype_bits(dtype: str | None) -> float | None:
    if not dtype:
        return None
    d = dtype.upper()
    if d.startswith(("F8", "FP8", "E4M3", "E5M2")):
        return 8.0
    if d in ("BF16", "F16", "FP16", "HALF"):
        return 16.0
    if d in ("F32", "FP32", "FLOAT"):
        return 32.0
    if d in ("F64", "FP64"):
        return 64.0
    if d in ("I8", "U8", "INT8", "UINT8"):
        return 8.0
    if d in ("Q4", "F4", "FP4", "I4", "U4"):
        return 4.0
    return None


_PACKED_DTYPES = {"U32", "I32", "U8"}


def infer_params(raw: dict) -> tuple[int | None, str, str | None]:
    """(params, source, native_dtype): safetensors.total -> gguf.total -> name -> None."""
    raw = _dict(raw)
    st = _dict(raw.get("safetensors"))
    native: str | None = None
    by_dtype = {k: _num(v) for k, v in _dict(st.get("parameters")).items() if isinstance(k, str)}
    if by_dtype:
        native = max(by_dtype, key=lambda k: by_dtype[k])
    name_params = params_from_name(_str(raw.get("id")) or "")

    total = _int(st.get("total")) or int(sum(by_dtype.values()))
    if total > 0:
        # Packed low-bit repos (mlx/gptq/awq/bnb) report packed element counts, far below the real count.
        if native in _PACKED_DTYPES and name_params:
            return name_params, "name", native
        return total, "safetensors", native
    g = _int(_dict(raw.get("gguf")).get("total"))
    if g > 0:
        # The Hub's gguf summary describes one file; in repos that also ship a draft/MTP head or projector it
        # can be that small file. Trust the size in the name when the header is wildly smaller.
        if name_params and g < 0.25 * name_params:
            return name_params, "name", native
        return g, "gguf", native
    if name_params:
        return name_params, "name", native
    return None, "unknown", native


# --------------------------------------------------------------------------- RAM / tiers


def estimate_ram_gb(params: float | None, bpw: float, task: str | None) -> float:
    if not params or params <= 0:
        return 0.0
    weights = params * bpw / 8 / 1e9
    ram = weights * OVERHEAD
    if task in LLM_TASKS:
        ram += 0.17 * math.sqrt(params / 1e9)
    if task in GEN_TASKS:
        ram += GEN_EXTRA_GB
    return round(ram, 2)


def practical(variant: str) -> bool:
    """False for lossy 1-2 bit variants (IQ1*/IQ2*/Q2*/TQ*/1-2bit)."""
    v = (variant or "").upper()
    if v.startswith(("IQ1", "IQ2", "Q2", "TQ")):
        return False
    if re.fullmatch(r"[12](\.\d+)?BIT", v):
        return False
    return True


def _quant_variant_format(formats: set[str]) -> str:
    if "mlx" in formats:
        return "mlx"
    for f in ("awq", "gptq", "exl2", "bnb"):
        if f in formats:
            return f
    return "native"


def variant_formats(rec: dict) -> dict[str, str]:
    """RAM variant name -> the format it comes from (gguf/onnx/mlx/awq/.../native)."""
    rec = _dict(rec)
    formats = set(_str_list(rec.get("formats")))
    out: dict[str, str] = {}
    for q in _str_list(rec.get("gguf_quants")):
        out[q] = "gguf"
    for o in _str_list(rec.get("onnx_variants")):
        out[o] = "onnx"
    qb = rec.get("quant_bits")
    if isinstance(qb, (int, float)) and not isinstance(qb, bool):
        out[f"{qb:g}bit"] = _quant_variant_format(formats)
    elif native_formats(formats) or (not formats and not out):
        out["native"] = "native"
    return out


def native_bits(native_dtype: str | None, formats: Iterable[str] | None) -> float:
    b = dtype_bits(native_dtype)
    if b:
        return b
    fm = set(formats or [])
    return 32.0 if ("pytorch" in fm and "safetensors" not in fm) else 16.0


def _variant_bpw(variant: str) -> float | None:
    for key in (variant, variant.upper(), variant.lower()):
        if key in BPW:
            return BPW[key]
    m = re.fullmatch(r"(\d+(?:\.\d+)?)bit", variant)
    if m:
        bits = float(m.group(1))
        return bits if bits >= 16 else bits + 0.5  # group scales/biases overhead
    return None


def ram_variants(rec: dict) -> dict[str, float]:
    """Estimated RAM (GB) per variant. Works on a partial or full record. {} when params unknown."""
    rec = _dict(rec)
    params = rec.get("params")
    if not isinstance(params, (int, float)) or params <= 0:
        return {}
    task = rec.get("task")
    out: dict[str, float] = {}
    for v in variant_formats(rec):
        if v == "native":
            bpw = native_bits(rec.get("native_dtype"), rec.get("formats"))
        else:
            bpw = _variant_bpw(v)
        if bpw is None:
            continue
        out[v] = estimate_ram_gb(params, bpw, task)
    return out


def _default_variant_format(variant: str, formats: set[str]) -> str:
    if variant == "native":
        return "native"
    if variant.endswith("bit"):
        return _quant_variant_format(formats)
    if variant in ("fp32", "fp16", "q4", "q4f16", "int8", "bnb4"):
        return "onnx"
    return "gguf"


def _runnable(tier: str, fmt: str, formats: set[str]) -> bool:
    if tier in ("gpu", "server"):
        return True
    if fmt == "native":
        nat = set(native_formats(formats))
        if not formats:
            return tier != "phone"
        allowed = PHONE_NATIVE_FORMATS if tier == "phone" else TIER_FORMATS[tier]
        return bool(nat & allowed)
    return fmt in TIER_FORMATS[tier]


def device_tier(ram_by_variant: dict[str, float], formats: list[str],
                variant_formats: dict[str, str] | None = None) -> str:
    """Smallest tier where some practical variant fits the budget and has a runnable format."""
    if not ram_by_variant:
        return "unknown"
    fm = set(_str_list(formats))
    vf = variant_formats or {}
    items = [(v, r) for v, r in ram_by_variant.items() if practical(v)] or list(ram_by_variant.items())
    for tier in ("phone", "laptop", "gpu"):
        budget = TIER_BUDGET_GB[tier]
        for v, r in items:
            fmt = vf.get(v) or _default_variant_format(v, fm)
            if r <= budget and _runnable(tier, fmt, fm):
                return tier
    return "server"


# --------------------------------------------------------------------------- license / languages / tags

def normalize_license(raw: dict) -> str | None:
    raw = _dict(raw)
    lic = _dict(raw.get("card")).get("license")
    cands = _str_list(lic)
    if cands:
        return cands[0].lower()
    for t in _str_list(raw.get("tags")):
        if t.lower().startswith("license:"):
            v = t[len("license:"):].strip().lower()
            if v:
                return v
    return None


def is_open_license(lic: str | None) -> bool:
    if not lic:
        return False
    lic = lic.lower()
    return lic in PERMISSIVE_LICENSES or lic.startswith(_PERMISSIVE_PREFIXES)


_ISO1 = frozenset("""aa ab ae af ak am an ar as av ay az ba be bg bh bi bm bn bo br bs ca ce ch co cr cs cu cv
cy da de dv dz ee el en eo es et eu fa ff fi fj fo fr fy ga gd gl gn gu gv ha he hi ho hr ht hu hy hz ia id ie
ig ii ik io is it iu ja jv ka kg ki kj kk kl km kn ko kr ks ku kv kw ky la lb lg li ln lo lt lu lv mg mh mi mk
ml mn mr ms mt my na nb nd ne ng nl nn no nr nv ny oc oj om or os pa pi pl ps pt qu rm rn ro ru rw sa sc sd se
sg si sk sl sm sn so sq sr ss st su sv sw ta te tg th ti tk tl tn to tr ts tt tw ty ug uk ur uz ve vi vo wa wo
xh yi yo za zh zu""".split())
_ISO3 = frozenset("""fil yue ceb haw war ilo hmn mai bho mag sat kok doi mni brx gom awa lus tpi bem ace ban
bug min mad ast arz azb pnb skr ckb kmr tzm nso lad lmo scn vec""".split())
_LANG_RE = re.compile(r"^[a-z]{2,3}(-[a-z0-9]{2,8})*$")


def languages(raw: dict) -> list[str]:
    raw = _dict(raw)
    out: set[str] = set()
    for v in _str_list(_dict(raw.get("card")).get("language")):
        v = v.lower()
        if _LANG_RE.match(v):
            out.add(v)
    for t in _str_list(raw.get("tags")):
        tl = t.lower()
        if tl in _ISO1 or tl in _ISO3:
            out.add(tl)
    return sorted(out)


_NOISE_PREFIXES = ("license:", "region:", "arxiv:", "base_model:", "dataset:", "deploy:", "doi:")
_NOISE_TAGS = {"endpoints_compatible", "autotrain_compatible", "text-generation-inference", "has_space",
               "model-index", "generated_from_trainer"}


def clean_tags(tags: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for t in _str_list(tags):
        tl = t.lower()
        if tl.startswith(_NOISE_PREFIXES) or tl in _NOISE_TAGS or tl in _ISO1 or tl in _ISO3:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# --------------------------------------------------------------------------- timestamps


def _parse_dt(v: Any) -> datetime | None:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, datetime):
        dt = v
    elif isinstance(v, (int, float)):
        ts = float(v)
        if ts > 1e11:
            ts /= 1000.0
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    elif isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        if s.endswith(("Z", "z")):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
            if not m:
                return None
            dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _iso_str(v: Any) -> str | None:
    if isinstance(v, str):
        return v.strip() or None
    dt = _parse_dt(v)
    return dt.isoformat() if dt else None


def _ts(v: Any) -> int:
    dt = _parse_dt(v)
    return int(dt.timestamp()) if dt else 0


# --------------------------------------------------------------------------- base model / adapter

_RELATIONS = ("quantized", "finetune", "adapter", "merge")
_WEIGHT_EXTS = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".onnx", ".npz", ".pte", ".tflite")


def _base_model_info(raw: dict, tags: list[str]) -> tuple[str | None, str | None]:
    bm = _dict(raw.get("base_models"))
    rel = _str(bm.get("relation"))
    rel = rel if rel in _RELATIONS else None
    base: str | None = None
    models = bm.get("models")
    if isinstance(models, (list, tuple)):
        for m in models:
            mid = _str(m.get("id")) if isinstance(m, dict) else _str(m)
            if mid:
                base = mid
                break
    tag_plain = tag_rel = tag_rel_id = None
    for t in tags:
        if not t.startswith("base_model:"):
            continue
        rest = t[len("base_model:"):]
        for r in _RELATIONS:
            if rest.startswith(r + ":"):
                if tag_rel is None:
                    tag_rel, tag_rel_id = r, rest[len(r) + 1:]
                break
        else:
            if tag_plain is None and rest:
                tag_plain = rest
    if base is None:
        cb = _str_list(_dict(raw.get("card")).get("base_model"))
        base = cb[0] if cb else (tag_rel_id or tag_plain)
    if rel is None:
        rel = tag_rel
    return base, rel


def _only_adapter_weights(files: list[str]) -> bool:
    weights = [_base(f).lower() for f in files if f.lower().endswith(_WEIGHT_EXTS)]
    return bool(weights) and all(w.startswith("adapter_model") for w in weights)


# --------------------------------------------------------------------------- build_record


def build_record(raw: dict) -> dict:
    """Raw HF row -> Record (everything except family fields). Never raises on odd input."""
    raw = _dict(raw)
    card = _dict(raw.get("card"))
    cfg = _dict(raw.get("config"))
    gguf = _dict(raw.get("gguf"))

    rid = _str(raw.get("id")) or ""
    author = _str(raw.get("author")) or (rid.split("/")[0] if "/" in rid else None)
    name = rid.split("/", 1)[-1]
    tags: list[str] = []
    for t in _str_list(raw.get("tags")) + _str_list(card.get("tags")):
        if t not in tags:
            tags.append(t)
    files = _files(raw.get("siblings"))
    library = _str(raw.get("library_name")) or _str(card.get("library_name"))

    task, task_source = infer_task({**raw, "tags": tags})
    params, params_source, native_dtype = infer_params(raw)
    formats = detect_formats(files, tags, library, author, rid)
    quants = gguf_quants(files)
    onnx_vars = onnx_variants(files)
    qbits = quant_bits(rid, tags, formats)
    lic = normalize_license({**raw, "tags": tags})
    base_model, relation = _base_model_info(raw, tags)
    is_adapter = relation == "adapter" or (library or "").lower() == "peft" or _only_adapter_weights(files)

    rec: dict[str, Any] = {
        "id": rid, "author": author, "name": name,
        "url": f"https://huggingface.co/{rid}",
        "task": task, "task_source": task_source,
        "modality": modality(task, tags, name),
        "library": library,
        "params": params, "params_source": params_source, "native_dtype": native_dtype,
        "formats": formats, "gguf_quants": quants, "onnx_variants": onnx_vars, "quant_bits": qbits,
        "license": lic, "open_license": is_open_license(lic),
        "languages": languages({**raw, "tags": tags}),
        "base_model": base_model, "relation": relation, "is_adapter": bool(is_adapter),
        "created_at": _iso_str(raw.get("created_at")),
        "created_ts": _ts(raw.get("created_at")),
        "last_modified": _iso_str(raw.get("last_modified")),
        "downloads": _int(raw.get("downloads")),
        "likes": _int(raw.get("likes")),
        "trending": float(_num(raw.get("trending"))),
    }

    ram = ram_variants(rec)
    rec["ram_gb"] = ram
    practical_vals = [r for v, r in ram.items() if practical(v)] or list(ram.values())
    rec["ram_gb_min"] = min(practical_vals) if practical_vals else None
    if params:
        if qbits is not None:
            native_ram = ram.get(f"{qbits:g}bit")
        else:
            native_ram = ram.get("native")
            if native_ram is None:
                native_ram = estimate_ram_gb(params, native_bits(native_dtype, formats), task)
    else:
        native_ram = None
    rec["ram_gb_native"] = native_ram
    rec["device_tier"] = device_tier(ram, formats, variant_formats(rec))

    rec["tags_clean"] = clean_tags(tags)
    rec["datasets"] = [t[len("dataset:"):] for t in tags if t.startswith("dataset:") and len(t) > 8]
    ctx = _int(gguf.get("context_length"))
    rec["context_length"] = ctx or None
    rec["architecture"] = _str(gguf.get("architecture")) or _str(cfg.get("model_type"))
    return rec

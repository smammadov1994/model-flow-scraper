import pytest

from model_finder import enrich as e

KOKORO = {
    "id": "hexgrad/Kokoro-82M", "author": "hexgrad", "pipeline_tag": "text-to-speech", "library_name": None,
    "tags": ["text-to-speech", "en", "license:apache-2.0", "arxiv:2306.07691", "region:us"],
    "siblings": ["kokoro-v1_0.pth", "voices/af_heart.pt", "config.json"],
    "card": {"license": "apache-2.0", "language": ["en"], "base_model": "yl4579/StyleTTS2-LJSpeech"},
    "safetensors": None, "gguf": None,
    "downloads": 5, "likes": 2, "trending": 1.5,
    "created_at": "2024-12-26T00:00:00.000Z", "last_modified": "2025-06-01 10:00:00",
}
KOKORO_ONNX = {
    "id": "onnx-community/Kokoro-82M-ONNX", "author": "onnx-community", "pipeline_tag": "text-to-speech",
    "library_name": "transformers.js",
    "tags": ["transformers.js", "onnx", "text-to-speech", "base_model:hexgrad/Kokoro-82M",
             "base_model:quantized:hexgrad/Kokoro-82M", "license:apache-2.0"],
    "siblings": ["onnx/model.onnx", "onnx/model_fp16.onnx", "onnx/model_q4.onnx", "onnx/model_q4f16.onnx",
                 "onnx/model_quantized.onnx", "onnx/model_uint8.onnx"] ,
    "card": {}, "base_models": {"relation": "quantized", "models": [{"id": "hexgrad/Kokoro-82M"}]},
}
WHISPER_CPP_FILES = ["ggml-base.en-q5_1.bin", "ggml-large-v3-turbo-encoder.mlmodelc.zip", "README.md"]
QWEN_GGUF = {
    "id": "unsloth/Qwen3-8B-GGUF", "author": "unsloth", "pipeline_tag": "text-generation",
    "library_name": "transformers",
    "tags": ["transformers", "gguf", "qwen3", "text-generation", "base_model:Qwen/Qwen3-8B",
             "base_model:quantized:Qwen/Qwen3-8B", "license:apache-2.0", "en"],
    "siblings": ["Qwen3-8B-Q4_K_M.gguf", "Qwen3-8B-UD-Q4_K_XL.gguf", "Qwen3-8B-Q8_0.gguf",
                 "Qwen3-8B-BF16.gguf", "README.md"],
    "gguf": {"total": 8_190_735_360, "architecture": "qwen3", "context_length": 40960},
    "card": {"license": "apache-2.0"},
}
MLX = {
    "id": "mlx-community/Llama-3.2-3B-Instruct-4bit", "author": "mlx-community", "pipeline_tag": "text-generation",
    "library_name": "mlx", "tags": ["mlx", "safetensors", "llama", "text-generation", "license:llama3.2", "en",
                                     "base_model:meta-llama/Llama-3.2-3B-Instruct"],
    "siblings": ["model.safetensors", "config.json", "tokenizer.json"],
    "safetensors": {"parameters": {"U32": 401_000_000, "F16": 100_000_000}, "total": 501_000_000},
}
SDXL = {
    "id": "stabilityai/stable-diffusion-xl-base-1.0", "author": "stabilityai", "pipeline_tag": None,
    "library_name": "diffusers", "tags": ["diffusers", "safetensors", "stable-diffusion-xl", "license:openrail++"],
    "siblings": ["model_index.json", "unet/diffusion_pytorch_model.fp16.safetensors",
                 "vae/diffusion_pytorch_model.fp16.safetensors"],
    "config": {"diffusers": {"_class_name": "StableDiffusionXLPipeline"}},
    "safetensors": {"parameters": {"BF16": 3_400_000_000, "F32": 100_000_000}, "total": 3_500_000_000},
}
COREML_SD = {
    "id": "apple/coreml-stable-diffusion-v1-5", "author": "apple", "pipeline_tag": "text-to-image",
    "library_name": "coreml", "tags": ["coreml", "text-to-image"],
    "siblings": ["original/compiled/Unet.mlmodelc/weights/weight.bin", "original/compiled/Unet.mlmodelc/model.mil",
                 "split_einsum/packages/Unet.mlpackage/Data/com.apple.CoreML/weights/weight.bin"],
}


# ---------------------------------------------------------------- formats
def test_kokoro_formats_and_params():
    r = e.build_record(KOKORO)
    assert r["formats"] == ["pytorch"]
    assert r["params"] == 82_000_000 and r["params_source"] == "name"
    assert r["task"] == "text-to-speech" and r["task_source"] == "pipeline_tag"
    assert r["modality"] == "speech-out"
    assert r["license"] == "apache-2.0" and r["open_license"] is True
    assert r["languages"] == ["en"]
    assert r["base_model"] == "yl4579/StyleTTS2-LJSpeech"
    assert r["url"] == "https://huggingface.co/hexgrad/Kokoro-82M"
    assert r["created_ts"] == 1735171200
    assert r["last_modified"] == "2025-06-01 10:00:00"
    assert r["device_tier"] == "laptop"  # pytorch only: not a phone format


def test_kokoro_onnx():
    r = e.build_record(KOKORO_ONNX)
    assert "onnx" in r["formats"]
    assert set(r["onnx_variants"]) == {"fp32", "fp16", "q4", "q4f16", "int8"}
    assert r["relation"] == "quantized" and r["base_model"] == "hexgrad/Kokoro-82M"
    assert r["device_tier"] == "phone"
    assert r["ram_gb_min"] < 1


def test_whisper_cpp_formats():
    f = e.detect_formats(WHISPER_CPP_FILES, [], None, "ggerganov", "ggerganov/whisper.cpp")
    assert "ggml" in f and "coreml" in f and "pytorch" not in f
    r = e.build_record({"id": "ggerganov/whisper.cpp", "siblings": WHISPER_CPP_FILES, "tags": []})
    assert r["task"] == "automatic-speech-recognition" and r["task_source"] == "name"
    assert r["device_tier"] == "unknown" and r["ram_gb"] == {}


def test_qwen_gguf():
    r = e.build_record(QWEN_GGUF)
    assert r["formats"] == ["gguf"]
    assert r["gguf_quants"] == ["Q4_K_M", "Q4_K_XL", "Q8_0", "BF16"]
    assert r["params"] == 8_190_735_360 and r["params_source"] == "gguf"
    assert 6.4 <= r["ram_gb"]["Q4_K_M"] <= 6.5
    assert r["ram_gb_min"] == r["ram_gb"]["Q4_K_M"]
    assert r["device_tier"] == "laptop"
    assert r["context_length"] == 40960 and r["architecture"] == "qwen3"
    assert "native" not in r["ram_gb"]
    assert r["modality"] == "text"


def test_spec_ram_example():
    assert e.estimate_ram_gb(8_000_000_000, 4.85, "text-generation") == 6.30


def test_mlx_quant_phone():
    r = e.build_record(MLX)
    assert r["quant_bits"] == 4.0
    assert "mlx" in r["formats"]
    assert r["params"] == 3_000_000_000 and r["params_source"] == "name"  # packed U32 total is not the real count
    assert r["ram_gb"].keys() == {"4bit"}
    assert r["ram_gb"]["4bit"] < 4
    assert r["device_tier"] == "phone"


def test_coreml_sd():
    r = e.build_record(COREML_SD)
    assert r["formats"] == ["coreml"]


def test_sdxl_task_via_library():
    r = e.build_record(SDXL)
    assert r["task"] == "text-to-image" and r["task_source"] == "library"
    assert r["modality"] == "image-generation"
    assert set(r["formats"]) == {"diffusers", "safetensors"}
    assert r["native_dtype"] == "BF16"
    assert r["ram_gb"]["native"] == 9.4  # 3.5e9*2*1.2 + 1.0 generation extra, no KV cache
    assert r["device_tier"] == "laptop"


def test_diffusers_class_variants():
    base = {"id": "x/y", "library_name": "diffusers"}
    assert e.infer_task({**base, "config": {"diffusers": {"_class_name": "WanVideoPipeline"}}})[0] == "text-to-video"
    assert e.infer_task({**base, "config": {"diffusers": {"_class_name": "StableDiffusionImg2ImgPipeline"}}})[0] == "image-to-image"


# ---------------------------------------------------------------- quants / variants
def test_gguf_quants_misc():
    files = ["a/model-IQ4_XS.gguf", "model.Q5_K_S.gguf", "x-q6_k.gguf", "mmproj-F16.gguf", "model-f16.gguf",
             "readme.md", "model-MXFP4.gguf", "m-IQ2_XXS.gguf", "m-Q4_0.gguf", "weird.gguf",
             "m-Q3_K_XL.gguf"]
    assert set(e.gguf_quants(files)) == {"IQ4_XS", "Q5_K_S", "Q6_K", "F16", "MXFP4", "IQ2_XXS", "Q4_0", "Q3_K_XL"}


def test_onnx_variants():
    assert e.onnx_variants(["onnx/decoder_model_merged_q4.onnx", "encoder.onnx", "x/m_bnb4.onnx",
                            "m_int8.onnx", "model.onnx_data"]) == ["fp32", "int8", "q4", "bnb4"]


@pytest.mark.parametrize("rid,tags,fmts,expect", [
    ("mlx-community/Qwen3-8B-4bit", ["mlx"], ["mlx"], 4.0),
    ("mlx-community/Qwen3-8B-8bit", [], ["mlx"], 8.0),
    ("mlx-community/Qwen3-8B-3bit", [], ["mlx"], 3.0),
    ("mlx-community/Qwen3-8B-6bit", [], ["mlx"], 6.0),
    ("mlx-community/Qwen3-8B-bf16", [], ["mlx"], 16.0),
    ("TheBloke/Llama-2-7B-AWQ", ["awq"], ["awq"], 4.0),
    ("x/y-GPTQ", [], ["gptq"], 4.0),
    ("x/y", ["8-bit"], [], 8.0),
    ("x/y", ["4-bit"], [], 4.0),
    ("x/y", [], ["safetensors"], None),
    ("x/Qwen3-8B-BF16", [], ["gguf"], None),
])
def test_quant_bits(rid, tags, fmts, expect):
    assert e.quant_bits(rid, tags, fmts) == expect


def test_quant_format_detection_by_name_and_tags():
    f = e.detect_formats(["model.safetensors"], ["gptq"], None, "x", "x/y-GPTQ-Int4")
    assert {"gptq", "safetensors"} <= set(f)
    assert "awq" in e.detect_formats([], [], None, "x", "TheBloke/foo-AWQ")
    assert "bnb" in e.detect_formats([], ["bitsandbytes"], None, "x", "x/y")
    assert "bnb" in e.detect_formats([], [], None, "unsloth", "unsloth/y-bnb-4bit")
    assert "mlx" in e.detect_formats([], [], None, "mlx-community", "mlx-community/y")
    assert "tflite" in e.detect_formats(["gemma.task", "a.litertlm"], [], None, "x", "x/y")
    assert "executorch" in e.detect_formats(["m.pte"], [], None, "x", "x/y")
    assert "openvino" in e.detect_formats(["openvino_model.xml"], [], None, "x", "x/y")
    assert "mlc" in e.detect_formats(["mlc-chat-config.json"], [], None, "x", "x/y")
    assert "nemo" in e.detect_formats(["m.nemo"], [], None, "x", "x/y")
    assert "pytorch" in e.detect_formats(["pytorch_model.bin"], [], None, "x", "x/y")


# ---------------------------------------------------------------- params
@pytest.mark.parametrize("name,expect", [
    ("Llama-3.1-8B-Instruct", 8_000_000_000),
    ("Qwen2.5-1.5B-Instruct", 1_500_000_000),
    ("Qwen3-0.6B", 600_000_000),
    ("Kokoro-82M", 82_000_000),
    ("org/Kokoro-82M-ONNX", 82_000_000),
    ("Mixtral-8x7B-Instruct-v0.1", 56_000_000_000),
    ("Qwen3-30B-A3B", 30_000_000_000),
    ("Qwen3-235B-A22B-Instruct-2507", 235_000_000_000),
    ("Llama-3.2-3B-Instruct-4bit", 3_000_000_000),
    ("model_7b_chat", 7_000_000_000),
    ("model-70b.gguf", 70_000_000_000),
    ("gemma-3-27b-it", 27_000_000_000),
    ("Qwen3-A3B", None),
    ("gemma-3n-E4B-it", None),
    ("gpt2", None),
    ("multilingual-e5-large", None),
    ("bge-m3", None),
    ("Phi-3-mini-4k-instruct", None),
    ("model-2024-10-14", None),
    ("whisper-large-v3-turbo", None),
    ("foo-v2", None),
    ("foo-512", None),
    ("model-8bit", None),
    ("v1.5b", None),
    (None, None),
])
def test_params_from_name(name, expect):
    assert e.params_from_name(name) == expect


def test_params_from_size():
    assert e.params_from_size(2_000_000_000, "float16") == 1_000_000_000
    assert e.params_from_size(4_000_000_000, "torch.bfloat16") == 2_000_000_000
    assert e.params_from_size(4_000_000_000, None, "model.fp16.safetensors") == 2_000_000_000
    assert e.params_from_size(4_000_000_000, "float32", "model-fp16.safetensors") == 2_000_000_000
    assert e.params_from_size(1_000_000_000, None, "model_int8.onnx") == 1_000_000_000
    assert e.params_from_size(1_000_000_000, None, "model_q4.onnx") == 2_000_000_000
    assert e.params_from_size(4_000_000_000, None, "model.safetensors") == 2_000_000_000
    assert e.params_from_size(4_000_000_000, None, "pytorch_model.bin") == 1_000_000_000
    assert e.params_from_size(4_000_000_000, "float32") == 1_000_000_000


def test_infer_params_priority_and_dtype():
    raw = {"id": "a/b-7B", "safetensors": {"parameters": {"BF16": 5, "F32": 1_000_000_000, "I8": 3},
                                            "total": 1_000_000_008},
           "gguf": {"total": 42}}
    assert e.infer_params(raw) == (1_000_000_008, "safetensors", "F32")
    assert e.infer_params({"id": "a/b-7B", "gguf": {"total": 6_900_000_000}}) == (6_900_000_000, "gguf", None)
    # header far smaller than the name says = the summary describes a draft/MTP file; trust the name
    assert e.infer_params({"id": "a/b-7B", "gguf": {"total": 42}}) == (7_000_000_000, "name", None)
    assert e.infer_params({"id": "a/b-7B"}) == (7_000_000_000, "name", None)
    assert e.infer_params({"id": "a/b"}) == (None, "unknown", None)


# ---------------------------------------------------------------- task / modality
def test_task_order():
    assert e.infer_task({"pipeline_tag": "fill-mask", "transformers_info": {"pipeline_tag": "x"}}) == ("fill-mask", "pipeline_tag")
    assert e.infer_task({"transformers_info": {"pipeline_tag": "summarization"}}) == ("summarization", "transformers_info")
    assert e.infer_task({"card": {"pipeline_tag": "translation"}}) == ("translation", "card")
    assert e.infer_task({"tags": ["en", "text-classification"]}) == ("text-classification", "tags")
    assert e.infer_task({"base_task": "text-generation", "id": "a/b"}) == ("text-generation", "base_model")
    assert e.infer_task({"library_name": "sentence-transformers"}) == ("sentence-similarity", "library")
    assert e.infer_task({"library_name": "peft", "id": "a/zzz"}) == (None, "unknown")
    assert e.infer_task({"library_name": "timm"})[0] == "image-classification"
    assert e.infer_task({"config": {"architectures": ["LlamaForCausalLM"]}}) == ("text-generation", "config")
    assert e.infer_task({"config": {"architectures": ["WhisperForConditionalGeneration"]}})[0] == "automatic-speech-recognition"
    assert e.infer_task({"config": {"architectures": ["BertForSequenceClassification"]}})[0] == "text-classification"
    assert e.infer_task({"config": {"architectures": ["BertForTokenClassification"]}})[0] == "token-classification"
    assert e.infer_task({"config": {"diffusers": {"_class_name": "FluxPipeline"}}}) == ("text-to-image", "config")
    assert e.infer_task({"id": "a/b", "tags": ["other"]}) == (None, "unknown")


@pytest.mark.parametrize("rid,expect", [
    ("hexgrad/Kokoro-82M", "text-to-speech"), ("a/xtts-v2", "text-to-speech"), ("a/F5-TTS", "text-to-speech"),
    ("openai/whisper-small", "automatic-speech-recognition"), ("nvidia/parakeet-tdt", "automatic-speech-recognition"),
    ("intfloat/multilingual-e5-large", "feature-extraction"), ("BAAI/bge-m3", "feature-extraction"),
    ("black-forest-labs/FLUX.1-dev", "text-to-image"), ("a/sdxl-turbo", "text-to-image"),
    ("a/foo-Instruct", "text-generation"), ("a/gemma-2b-it", "text-generation"), ("openai/gpt-oss-20b", "text-generation"),
    ("a/zzz", None),
])
def test_task_from_name(rid, expect):
    t, src = e.infer_task({"id": rid})
    assert t == expect
    assert src == ("name" if expect else "unknown")


def test_modality_rules():
    assert e.modality("text-generation", [], "Qwen2.5-Coder-7B") == "code"
    assert e.modality("text-generation", ["code"], "foo") == "code"
    assert e.modality("text-generation", [], "bert-decoder") == "text"
    assert e.modality("text-generation", ["dataset:bigcode/the-stack"], "foo") == "text"
    assert e.modality("text-to-audio", ["music"], "musicgen-small") == "audio"
    assert e.modality("text-to-audio", [], "foo") == "speech-out"
    assert e.modality("automatic-speech-recognition", [], "x") == "speech-in"
    assert e.modality("feature-extraction", [], "x") == "embedding"
    assert e.modality("image-text-to-text", [], "x") == "multimodal"
    assert e.modality("text-to-video", [], "x") == "video"
    assert e.modality("image-segmentation", [], "x") == "vision"
    assert e.modality("tabular-regression", [], "x") == "other"
    assert e.modality(None, [], "x") == "other"
    assert set(e.TASK_GROUPS) <= e.KNOWN_TASKS


# ---------------------------------------------------------------- ram / tiers
def test_ram_extras():
    assert e.estimate_ram_gb(1e9, 16, "text-to-image") == round(2.4 + 1.0, 2)
    assert e.estimate_ram_gb(1e9, 16, "feature-extraction") == 2.4
    assert e.estimate_ram_gb(4e9, 16, "image-text-to-text") == round(4 * 2 * 1.2 + 0.34, 2)
    assert e.estimate_ram_gb(None, 16, None) == 0.0


def test_practical():
    for v in ("IQ1_S", "IQ2_XXS", "Q2_K", "TQ1_0", "2bit"):
        assert not e.practical(v)
    for v in ("Q4_K_M", "IQ3_XS", "Q3_K_S", "4bit", "native", "q4", "F16"):
        assert e.practical(v)


def test_ram_min_excludes_impractical():
    raw = {**QWEN_GGUF, "siblings": ["m-IQ1_S.gguf", "m-Q4_K_M.gguf"]}
    r = e.build_record(raw)
    assert r["ram_gb_min"] == r["ram_gb"]["Q4_K_M"] > r["ram_gb"]["IQ1_S"]


def test_ram_variants_and_variant_formats_on_full_record():
    r = e.build_record(QWEN_GGUF)
    assert e.ram_variants(r) == r["ram_gb"]
    vf = e.variant_formats(r)
    assert vf == {"Q4_K_M": "gguf", "Q4_K_XL": "gguf", "Q8_0": "gguf", "BF16": "gguf"}
    assert e.device_tier(r["ram_gb"], r["formats"], vf) == r["device_tier"]
    m = e.build_record(MLX)
    assert e.variant_formats(m) == {"4bit": "mlx"}
    o = e.build_record(KOKORO_ONNX)
    assert set(e.variant_formats(o).values()) == {"onnx"}
    k = e.build_record(KOKORO)
    assert e.variant_formats(k) == {"native": "native"}
    assert k["ram_gb_native"] == k["ram_gb"]["native"]


def test_device_tier_rules():
    # same RAM, different formats
    assert e.device_tier({"Q4_K_M": 3.0}, ["gguf"], {"Q4_K_M": "gguf"}) == "phone"
    assert e.device_tier({"native": 3.0}, ["safetensors"], {"native": "native"}) == "laptop"
    assert e.device_tier({"native": 3.0}, ["coreml"], {"native": "native"}) == "phone"
    assert e.device_tier({"4bit": 3.0}, ["awq"], {"4bit": "awq"}) == "gpu"
    assert e.device_tier({"Q4_K_M": 11.9}, ["gguf"]) == "laptop"
    assert e.device_tier({"Q4_K_M": 20}, ["gguf"]) == "gpu"
    assert e.device_tier({"Q4_K_M": 40}, ["gguf"]) == "server"
    assert e.device_tier({}, ["gguf"]) == "unknown"
    # impractical variant alone does not make a phone model if a practical one is bigger
    assert e.device_tier({"IQ1_S": 2.0, "Q4_K_M": 6.0}, ["gguf"], {"IQ1_S": "gguf", "Q4_K_M": "gguf"}) == "laptop"
    # no file list at all: assume safetensors-like weights
    assert e.device_tier({"native": 5.0}, []) == "laptop"
    # defaults when variant_formats omitted
    assert e.device_tier({"q4": 2.0}, ["onnx"]) == "phone"
    assert e.device_tier({"Q4_K_M": 2.0}, ["gguf"]) == "phone"


# ---------------------------------------------------------------- license / langs / tags
def test_license_and_open():
    assert e.normalize_license({"card": {"license": "Apache-2.0"}}) == "apache-2.0"
    assert e.normalize_license({"tags": ["x", "license:MIT"]}) == "mit"
    assert e.normalize_license({"card": {"license": ["cc-by-4.0", "mit"]}, "tags": ["license:mit"]}) == "cc-by-4.0"
    assert e.normalize_license({"card": {"license": None}}) is None
    assert e.is_open_license("bsd-3-clause") and e.is_open_license("openrail++") and e.is_open_license("gpl-3.0")
    assert not e.is_open_license("llama3.1") and not e.is_open_license("other") and not e.is_open_license(None)


def test_languages_and_clean_tags():
    raw = {"card": {"language": "fr"}, "tags": ["en", "de", "mlx", "awq", "gguf", "pt", "zh", "multilingual"]}
    assert e.languages(raw) == ["de", "en", "fr", "pt", "zh"]
    assert e.languages({"card": {"language": ["en", "es", "multilingual", 3, None]}}) == ["en", "es"]
    assert e.languages({"card": {"language": None}, "tags": None}) == []
    tags = ["text-generation", "license:mit", "region:us", "endpoints_compatible", "arxiv:1234", "base_model:a/b",
            "dataset:foo/bar", "deploy:azure", "autotrain_compatible", "text-generation-inference", "has_space",
            "en", "model-index", "generated_from_trainer", "qwen3", "qwen3", "gguf"]
    assert e.clean_tags(tags) == ["text-generation", "qwen3", "gguf"]


# ---------------------------------------------------------------- robustness / misc fields
def test_build_record_never_raises_on_junk():
    junk = [
        {}, {"id": None}, {"id": "a/b", "tags": None, "siblings": None, "card": None, "config": None,
                           "safetensors": None, "gguf": None, "base_models": None, "downloads": None},
        {"id": "a/b", "card": {"language": 5, "base_model": 7, "license": 3}, "safetensors": {"total": "x"},
         "tags": [1, None, "en"], "siblings": [None, 3, {"rfilename": "m.gguf"}], "created_at": 12},
        {"id": "a/b", "safetensors": {"parameters": None, "total": None}, "gguf": {"total": "abc"},
         "downloads": "7", "likes": float("nan"), "created_at": object(), "base_models": {"models": [None, 5]}},
        {"id": "a/b", "created_at": "not a date", "transformers_info": [], "library_name": 3},
    ]
    for raw in junk:
        r = e.build_record(raw)
        assert r["device_tier"] in {"unknown", "phone", "laptop", "gpu", "server"}
        assert not any(k.startswith("_") for k in r)
    r = e.build_record({"id": "a/b", "siblings": [None, 3, {"rfilename": "m.gguf"}]})
    assert r["formats"] == ["gguf"]
    assert e.build_record({})["name"] == ""


def test_card_list_and_str_fields():
    r = e.build_record({"id": "a/b", "card": {"language": ["en", "fr"], "base_model": ["x/y", "z/w"],
                                              "license": "mit"}})
    assert r["languages"] == ["en", "fr"] and r["base_model"] == "x/y" and r["license"] == "mit"
    r = e.build_record({"id": "a/b", "card": {"language": "de", "base_model": "x/y"}})
    assert r["languages"] == ["de"] and r["base_model"] == "x/y"
    r = e.build_record({"id": "a/b", "tags": ["base_model:finetune:p/q", "dataset:d/e"]})
    assert r["base_model"] == "p/q" and r["relation"] == "finetune" and r["datasets"] == ["d/e"]


@pytest.mark.parametrize("val", [
    "2026-10-14T13:13:49.000Z", "2026-10-14 13:13:49", "2026-10-14T13:13:49+00:00",
])
def test_created_ts_formats(val):
    assert e.build_record({"id": "a/b", "created_at": val})["created_ts"] == 1_791_983_629 - 0 * 1


def test_created_ts_datetime_and_numbers():
    from datetime import datetime, timezone
    dt = datetime(2026, 10, 14, 13, 13, 49, tzinfo=timezone.utc)
    r = e.build_record({"id": "a/b", "created_at": dt, "last_modified": dt})
    assert r["created_ts"] == int(dt.timestamp()) and r["created_at"] == dt.isoformat()
    assert e.build_record({"id": "a/b", "created_at": datetime(2026, 10, 14, 13, 13, 49)})["created_ts"] == int(dt.timestamp())
    assert e.build_record({"id": "a/b"})["created_ts"] == 0


def test_is_adapter():
    assert e.build_record({"id": "a/b", "siblings": ["adapter_model.safetensors", "adapter_config.json"]})["is_adapter"]
    assert e.build_record({"id": "a/b", "library_name": "peft"})["is_adapter"]
    assert e.build_record({"id": "a/b", "base_models": {"relation": "adapter", "models": [{"id": "x/y"}]}})["is_adapter"]
    assert not e.build_record({"id": "a/b", "siblings": ["adapter_model.safetensors", "model.safetensors"]})["is_adapter"]
    assert not e.build_record({"id": "a/b", "siblings": ["model.safetensors"]})["is_adapter"]

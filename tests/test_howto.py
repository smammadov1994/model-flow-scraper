import pytest

from model_finder import enrich as e
from model_finder import howto as h
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

KOKORO_R = e.build_record(KOKORO)
ONNX_R = e.build_record(KOKORO_ONNX)
GGUF_R = e.build_record(QWEN_GGUF)
MLX_R = e.build_record(MLX)
SDXL_R = e.build_record(SDXL)
WHISPER_R = e.build_record({"id": "ggerganov/whisper.cpp", "siblings": WHISPER_CPP_FILES, "tags": []})


def test_gguf_variant_choice_prefers_q4_k_m():
    assert h.best_variant(GGUF_R, "laptop") == ("Q4_K_M", "gguf")
    assert h.best_variant(GGUF_R, "mac") == ("Q4_K_M", "gguf")
    assert h.best_variant(GGUF_R, "any") == ("Q4_K_M", "gguf")


def test_gguf_phone_does_not_fit_8b():
    assert h.best_variant(GGUF_R, "phone") == (None, None)


def test_gguf_fallback_when_q4_k_m_missing():
    rec = e.build_record({**QWEN_GGUF, "siblings": ["m-Q8_0.gguf", "m-Q4_0.gguf", "m-Q5_K_M.gguf"]})
    assert h.best_variant(rec, "laptop") == ("Q4_0", "gguf")
    rec = e.build_record({**QWEN_GGUF, "siblings": ["m-Q8_0.gguf", "m-Q5_K_M.gguf", "m-F16.gguf"]})
    assert h.best_variant(rec, "laptop") == ("Q5_K_M", "gguf")
    # an impractical quant is a last resort
    rec = e.build_record({**QWEN_GGUF, "siblings": ["m-IQ1_S.gguf", "m-F16.gguf"]})
    assert h.best_variant(rec, "laptop")[0] == "IQ1_S"  # F16 (~19 GB) does not fit a laptop


def test_budget_respected():
    # only Q8_0 and BF16 present: 8B BF16 is too big for a laptop, Q8_0 fits
    rec = e.build_record({**QWEN_GGUF, "siblings": ["m-Q8_0.gguf", "m-BF16.gguf"]})
    assert h.best_variant(rec, "laptop") == ("Q8_0", "gguf")  # 8.19*8.5/8*1.2 + .49 = 10.9
    assert h.best_variant(rec, "phone") == (None, None)


def test_onnx_phone():
    v, f = h.best_variant(ONNX_R, "phone")
    assert f == "onnx" and v == "q4"
    hint = h.run_hint(ONNX_R, "phone")
    assert "ONNX Runtime Mobile" in hint and "sherpa-onnx" in hint and "q4" in hint


def test_kokoro_pth_on_phone_points_to_onnx_derivative():
    rec = {**KOKORO_R, "derivatives": [
        {"id": "onnx-community/Kokoro-82M-ONNX", "formats": ["onnx"], "gguf_quants": [],
         "onnx_variants": ["fp32", "fp16", "q4", "int8"], "quant_bits": None, "ram_gb": ONNX_R["ram_gb"]}]}
    assert h.best_variant(rec, "phone") == (None, None)
    hint = h.run_hint(rec, "phone")
    assert hint.startswith("Use the ONNX build onnx-community/Kokoro-82M-ONNX with ")
    assert "ONNX Runtime Mobile" in hint


def test_derivative_without_ram_uses_base_params():
    rec = {"id": "o/base", "task": "text-generation", "params": 3_000_000_000, "formats": ["safetensors"],
           "gguf_quants": [], "onnx_variants": [], "quant_bits": None, "native_dtype": "BF16", "library": "transformers",
           "derivatives": [{"id": "u/base-GGUF", "formats": ["gguf"], "gguf_quants": ["Q8_0", "Q4_K_M"],
                            "onnx_variants": [], "quant_bits": None}]}
    hint = h.run_hint(rec, "phone")
    assert "Q4_K_M" in hint and "u/base-GGUF" in hint and "llama.cpp" in hint
    assert "ollama run hf.co/u/base-GGUF:Q4_K_M" in h.run_hint({**rec, "formats": [], "params": 8e9}, "laptop")


def test_derivatives_too_big_are_skipped():
    rec = {"id": "o/big", "task": "text-generation", "params": 70_000_000_000, "formats": ["safetensors"],
           "derivatives": [
               {"id": "u/big-4bit", "formats": ["mlx"], "gguf_quants": [], "onnx_variants": [], "quant_bits": 4.0,
                "ram_gb": 45.0},
               {"id": "u/big-GGUF", "formats": ["gguf"], "gguf_quants": ["Q4_K_M"], "onnx_variants": [],
                "quant_bits": None, "ram_gb": {"Q4_K_M": 47.0}}]}
    assert h.run_hint(rec, "mac") == "See the model card for how to run it."
    assert h.run_hint(rec, "phone") == "See the model card for how to run it."
    # the base repo itself is runnable with no budget cap, so no derivative is needed
    assert h.run_hint(rec, "server") == "Serve with vLLM (`vllm serve o/big`) or transformers."


def test_best_derivative_wins_by_device_format_order():
    rec = {"id": "o/m", "task": "text-generation", "params": 3e9, "formats": ["pytorch"],
           "derivatives": [
               {"id": "u/m-onnx", "formats": ["onnx"], "onnx_variants": ["q4"]},
               {"id": "u/m-mlx", "formats": ["mlx"], "quant_bits": 4.0},
               {"id": "u/m-gguf", "formats": ["gguf"], "gguf_quants": ["Q4_K_M"]}]}
    assert "u/m-gguf" in h.run_hint(rec, "phone")
    assert "u/m-mlx" in h.run_hint(rec, "mac")


def test_mlx_hints():
    assert h.best_variant(MLX_R, "phone") == ("4bit", "mlx")
    assert "mlx-swift" in h.run_hint(MLX_R, "phone")
    assert h.best_variant(MLX_R, "mac") == ("4bit", "mlx")
    assert h.run_hint(MLX_R, "mac") == (
        "`pip install mlx-lm` then `mlx_lm.generate --model mlx-community/Llama-3.2-3B-Instruct-4bit --prompt 'hi'`.")


def test_mlx_tts_hint():
    rec = e.build_record({"id": "mlx-community/Kokoro-82M-bf16", "author": "mlx-community", "pipeline_tag": "text-to-speech",
                          "library_name": "mlx-audio", "tags": ["mlx"], "siblings": ["model.safetensors"]})
    hint = h.run_hint(rec, "mac")
    assert "pip install mlx-audio" in hint and "mlx_audio.tts.generate --model mlx-community/Kokoro-82M-bf16" in hint


def test_gguf_hints():
    assert h.run_hint(GGUF_R, "laptop") == "`ollama run hf.co/unsloth/Qwen3-8B-GGUF:Q4_K_M` or open it in LM Studio / llama.cpp."
    small = e.build_record({**QWEN_GGUF, "id": "unsloth/Qwen3-1B-GGUF", "gguf": {"total": 1_000_000_000}})
    hint = h.run_hint(small, "phone")
    assert "Q4_K_M .gguf" in hint and "PocketPal AI" in hint


def test_whisper_cpp_hints():
    assert h.best_variant(WHISPER_R, "phone")[1] == "ggml"
    assert h.run_hint(WHISPER_R, "phone") == "whisper.cpp runs this on iOS/Android; use the q5/q8 .bin."
    assert "whisper.cpp" in h.run_hint(WHISPER_R, "mac")


def test_coreml_hint_on_phone():
    rec = e.build_record(COREML_SD)
    assert h.best_variant(rec, "phone") == ("native", "coreml")
    assert "Core ML package" in h.run_hint(rec, "phone")


def test_executorch_tflite():
    base = {"id": "o/m", "task": "text-generation", "params": 1e9, "gguf_quants": [], "onnx_variants": [],
            "quant_bits": None}
    assert "ExecuTorch" in h.run_hint({**base, "formats": ["executorch"]}, "phone")
    assert "LiteRT" in h.run_hint({**base, "formats": ["tflite"]}, "phone")


def test_diffusers():
    assert h.best_variant(SDXL_R, "laptop") == ("native", "diffusers")
    hint = h.run_hint(SDXL_R, "gpu")
    assert "pip install diffusers" in hint and "DiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0'" in hint
    assert h.best_variant(SDXL_R, "phone") == (None, None)


def test_gpu_safetensors_llm_vllm():
    rec = e.build_record({"id": "o/Llama-8B", "pipeline_tag": "text-generation", "library_name": "transformers",
                          "siblings": ["model.safetensors"], "safetensors": {"total": 8_000_000_000, "parameters": {"BF16": 8_000_000_000}}})
    assert h.best_variant(rec, "gpu") == ("native", "safetensors")
    assert h.best_variant(rec, "laptop") == (None, None)  # 8B bf16 ~ 20 GB
    assert h.run_hint(rec, "gpu") == "Serve with vLLM (`vllm serve o/Llama-8B`) or transformers."


def test_awq_hint():
    rec = e.build_record({"id": "o/Llama-8B-AWQ", "pipeline_tag": "text-generation", "tags": ["awq", "4-bit"],
                          "siblings": ["model.safetensors"], "safetensors": {"total": 8_000_000_000}})
    assert h.best_variant(rec, "gpu") == ("4bit", "awq")
    assert h.best_variant(rec, "laptop") == (None, None)
    assert "vllm serve o/Llama-8B-AWQ" in h.run_hint(rec, "gpu")


def test_fallback_hints():
    assert h.run_hint({"id": "a/b", "library": "timm", "formats": []}, "phone") == "Load with the `timm` library (see the model card)."
    assert h.run_hint({}, "phone") == "See the model card for how to run it."
    assert h.run_hint(None, "weird") == "See the model card for how to run it."
    assert h.best_variant({}, "phone") == (None, None) or h.best_variant({}, "phone")[0] == "native"


def test_device_aliases_and_any():
    assert h.best_variant(GGUF_R, "MacBook") == ("Q4_K_M", "gguf")
    assert h.best_variant(GGUF_R, "") == ("Q4_K_M", "gguf")
    assert h.best_variant(GGUF_R, None) == ("Q4_K_M", "gguf")


@pytest.mark.parametrize("dev", ["phone", "laptop", "mac", "gpu", "server", "any"])
@pytest.mark.parametrize("rec", [KOKORO_R, ONNX_R, GGUF_R, MLX_R, SDXL_R, WHISPER_R, e.build_record(COREML_SD)])
def test_run_hint_always_one_line(rec, dev):
    hint = h.run_hint(rec, dev)
    assert isinstance(hint, str) and hint and "\n" not in hint

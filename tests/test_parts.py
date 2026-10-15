"""Tests for the pieces that don't need Qdrant or the network: question rules, families, X hook, ingest."""
from model_finder import families, query
from model_finder.sources.x import extract_hf_ids


def plan(q, device=None):
    return query.make_plan(q, device, use_llm=False)


def test_phone_voice():
    p = plan("I am looking to make a voice model on my phone")
    assert "text-to-speech" in p.tasks
    assert p.device == "phone" and p.ram_cap_gb == 4.0 and "onnx" in p.formats


def test_macbook_coding():
    p = plan("small coding model for my MacBook")
    assert p.tasks == ["text-generation"] and p.device == "mac" and p.ram_cap_gb == 12.0
    assert "code" in p.keywords


def test_vram():
    p = plan("image generation under 8GB VRAM")
    assert p.tasks == ["text-to-image"] and p.device == "gpu" and p.ram_cap_gb == 8.0


def test_memory_stated_mac():
    p = plan("chat model for my 32GB MacBook Pro")
    assert p.device == "mac" and p.ram_cap_gb == 22.4


def test_device_override_and_license():
    p = plan("transcribe meetings, commercial use", device="laptop")
    assert p.tasks == ["automatic-speech-recognition"] and p.device == "laptop" and p.license_open is True


def test_filter_builds_for_every_relax_level():
    p = plan("voice model on my phone")
    for _, opts in query.LADDER:
        query.build_filter(p, opts)


def test_stem():
    assert families.stem("mlx-community/Kokoro-82M-4bit") == "kokoro-82m"
    assert families.stem("onnx-community/Kokoro-82M-ONNX") == "kokoro-82m"
    assert families.stem("bartowski/Qwen2.5-Coder-7B-Instruct-GGUF") == "qwen2.5-coder-7b-instruct"
    assert families.stem("Qwen/Qwen2.5-Coder-7B-Instruct-AWQ") == "qwen2.5-coder-7b-instruct"


def _m(i, author, fmts, dl, base=None, rel=None, tier="unknown", ram=None):
    return {"id": i, "author": author, "base_model": base, "relation": rel, "formats": fmts, "downloads": dl,
            "gguf_quants": [], "onnx_variants": [], "quant_bits": None, "ram_gb": {}, "ram_gb_min": ram,
            "device_tier": tier}


def test_families_link_conversions():
    ms = [
        _m("hexgrad/Kokoro-82M", "hexgrad", ["pytorch"], 2_000_000),
        _m("onnx-community/Kokoro-82M-ONNX", "onnx-community", ["onnx"], 90_000, "hexgrad/Kokoro-82M", "quantized", "phone", 0.2),
        # mlx copy points at the wrong base with relation finetune: linked by name instead
        _m("mlx-community/Kokoro-82M-4bit", "mlx-community", ["mlx", "safetensors"], 5_000, "yl4579/StyleTTS2-LJSpeech", "finetune", "phone", 0.1),
        _m("someone/my-kokoro-finetune", "someone", ["safetensors"], 300, "hexgrad/Kokoro-82M", "finetune"),
    ]
    f = families.compute(ms)
    root = f["hexgrad/Kokoro-82M"]
    assert root["family_root"] == "hexgrad/Kokoro-82M"
    assert {"onnx", "mlx", "pytorch"} <= set(root["family_formats"])
    assert root["family_device_tier"] == "phone" and root["family_ram_gb_min"] == 0.1
    assert [d["id"] for d in root["derivatives"]] == ["onnx-community/Kokoro-82M-ONNX", "mlx-community/Kokoro-82M-4bit"]
    assert f["mlx-community/Kokoro-82M-4bit"]["family_root"] == "hexgrad/Kokoro-82M"
    assert f["someone/my-kokoro-finetune"]["family_root"] == "someone/my-kokoro-finetune"  # finetunes are their own model


def test_extract_hf_ids():
    t = "New drop: https://huggingface.co/Qwen/Qwen3-TTS-1B and hf.co/onnx-community/Kokoro-82M-ONNX. Data: huggingface.co/datasets/x/y"
    assert extract_hf_ids(t) == ["Qwen/Qwen3-TTS-1B", "onnx-community/Kokoro-82M-ONNX"]


def test_distill_labelled_quantized_is_not_folded():
    big = {**_m("org/Big-12L", "org", ["safetensors"], 1000), "params": 33_000_000}
    small = {**_m("org/Small-6L", "org", ["safetensors"], 900, "org/Big-12L", "quantized"), "params": 22_000_000}
    real = {**_m("org/Big-12L-int8", "org", ["onnx"], 10, "org/Big-12L", "quantized"), "params": 33_000_000}
    for m in (big, small, real):
        m.setdefault("params", None)
    f = families.compute([big, small, real])
    assert f["org/Small-6L"]["family_root"] == "org/Small-6L"
    assert f["org/Big-12L-int8"]["family_root"] == "org/Big-12L"


def test_card_extraction_needs_evidence():
    from model_finder.card_llm import verified

    card = "Contriever is an unsupervised dense retriever. Use it to compute embeddings for retrieval."
    assert verified({"task": "text-to-speech", "params_b": 0.082, "license": "mit"}, card) == {
        "task": None, "params_b": None, "license": None}
    assert verified({"task": "feature-extraction", "params_b": 0.11}, card + " It has 110M parameters.") == {
        "task": "feature-extraction", "params_b": 0.11}

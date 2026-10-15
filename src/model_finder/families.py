"""Links converted/quantized copies (GGUF, ONNX, MLX, Core ML ...) back to the original model.

Why: "Kokoro-82M on my phone" is answered by hexgrad/Kokoro-82M (the model people know) *plus*
onnx-community/Kokoro-82M-ONNX (the file you actually put on the phone). The original gets
`family_formats` / `family_ram_gb_min` / `family_device_tier` / `derivatives`, and every copy gets
`family_root` so search results can be collapsed to one card per model.
"""
from __future__ import annotations

import json
import re

TIER_ORDER = ["phone", "laptop", "gpu", "server", "unknown"]

# Orgs whose repos are mostly format conversions of someone else's model.
CONVERTER_ORGS = {
    "mlx-community", "onnx-community", "lmstudio-community", "bartowski", "unsloth", "thebloke", "quantfactory",
    "mradermacher", "second-state", "ggml-org", "xenova", "maziyarpanahi", "qwen-gguf", "gaianet", "nexaai",
    "litert-community", "executorch-community", "coreml-community", "apple", "intel", "openvino", "cortexso",
    "ggerganov", "devquasar", "tensorblock", "noctrex", "lmstudio-ai", "mlc-ai", "pytorch", "redhatai",
    "neuralmagic", "nm-testing", "turboderp", "unsloth-community", "bnb-community", "csukuangfj", "k2-fsa",
    "comfy-org", "kijai", "city96", "gguf-org", "calcuis",
}
DEPLOY_FORMATS = {"gguf", "ggml", "onnx", "mlx", "coreml", "executorch", "tflite", "mlc", "openvino", "awq", "gptq", "exl2", "bnb"}
_SUFFIX = re.compile(
    r"([-_.](gguf|ggml|onnx|mlx|coreml|executorch|tflite|litert|litertlm|openvino|ov|awq|gptq|exl2|bnb|mlc|"
    r"[2-8]bit|[2-8]-bit|bf16|fp16|fp8|f16|f32|int4|int8|q[2-8](_[0-9a-z]+)*|q4f16(_1)?|i1|imatrix|quantized|quant|"
    r"dwq|mxfp4|nvfp4|w4a16|w8a8|nf4|fp8-dynamic|dynamic|ud|hf|web|timestamped))+$"
)


def stem(model_id: str) -> str:
    name = model_id.split("/")[-1].lower()
    prev = None
    while prev != name:
        prev = name
        name = _SUFFIX.sub("", name)
    return name


def _load(con) -> list[dict]:
    q = """select id,
      json_extract(record,'$.author'), json_extract(record,'$.base_model'), json_extract(record,'$.relation'),
      json_extract(record,'$.formats'), json_extract(record,'$.downloads'), json_extract(record,'$.gguf_quants'),
      json_extract(record,'$.onnx_variants'), json_extract(record,'$.quant_bits'), json_extract(record,'$.ram_gb'),
      json_extract(record,'$.ram_gb_min'), json_extract(record,'$.device_tier'),
      json_extract(record,'$.family_root'), json_extract(record,'$.family_formats'),
      json_extract(record,'$.family_ram_gb_min'), json_extract(record,'$.family_device_tier'),
      json_extract(record,'$.derivatives'), json_extract(record,'$.params')
      from models"""
    keys = ["id", "author", "base_model", "relation", "formats", "downloads", "gguf_quants", "onnx_variants",
            "quant_bits", "ram_gb", "ram_gb_min", "device_tier", "old_root", "old_ff", "old_fram", "old_ftier", "old_der", "params"]
    out = []
    for row in con.execute(q):
        d = dict(zip(keys, row))
        for k in ("formats", "gguf_quants", "onnx_variants", "ram_gb", "old_ff", "old_der"):
            d[k] = json.loads(d[k]) if d[k] else ([] if k != "ram_gb" else {})
        out.append(d)
    return out


def _is_conversion(m: dict) -> bool:
    fm = set(m["formats"])
    if (m["author"] or "").lower() in CONVERTER_ORGS:
        return True
    if not fm & DEPLOY_FORMATS:
        return False
    name = m["id"].split("/")[-1].lower()
    # MLX repos ship safetensors too; a format suffix in the name ("-mlx", "-GGUF", "-onnx", "-4bit") means a copy
    return "safetensors" not in fm or "mlx" in fm or stem(m["id"]) != name


def _same_size(a: dict, b: dict) -> bool:
    """A real quantized copy has the base's parameter count. The Hub auto-labels many distills/pruned models as
    'quantized'; those are different models and must not be folded into the base."""
    pa, pb = a.get("params"), b.get("params")
    if not pa or not pb:
        return True
    return 0.8 <= pa / pb <= 1.25


def compute(models: list[dict]) -> dict[str, dict]:
    """Returns {id: family fields} for every model."""
    by_id = {m["id"]: m for m in models}
    # best "original" per name stem: not a conversion, most downloads
    originals: dict[str, dict] = {}
    for m in models:
        if _is_conversion(m) and m["relation"] == "quantized":
            continue
        s = stem(m["id"])
        if (m["author"] or "").lower() in CONVERTER_ORGS:
            continue
        if s not in originals or (m["downloads"] or 0) > (originals[s]["downloads"] or 0):
            originals[s] = m

    root_of: dict[str, str] = {}
    for m in models:
        root = None
        if m["relation"] == "quantized" and m["base_model"] in by_id and _same_size(m, by_id[m["base_model"]]):
            root = m["base_model"]
        elif _is_conversion(m):
            o = originals.get(stem(m["id"]))
            if o and o["id"] != m["id"]:
                root = o["id"]
            elif m["base_model"] in by_id and stem(m["base_model"]) == stem(m["id"]):
                root = m["base_model"]
        if root and root != m["id"]:
            # follow one more hop (copy of a copy) but never loop
            r2 = root_of.get(root)
            root_of[m["id"]] = r2 if r2 and r2 != m["id"] else root

    members: dict[str, list[dict]] = {}
    for mid, root in root_of.items():
        members.setdefault(root, []).append(by_id[mid])

    fields: dict[str, dict] = {}
    for m in models:
        mid = m["id"]
        # copies whose own size estimate disagrees with the original are mislabeled; don't let them set the minimum
        ders = sorted((d for d in members.get(mid, []) if _same_size(d, m)), key=lambda d: -(d["downloads"] or 0))
        group = [m] + ders
        ff = sorted({f for g in group for f in g["formats"]})
        rams = [g["ram_gb_min"] for g in group if g["ram_gb_min"]]
        tiers = [g["device_tier"] or "unknown" for g in group]
        fields[mid] = {
            "family_root": root_of.get(mid, mid),
            "family_formats": ff,
            "family_ram_gb_min": min(rams) if rams else None,
            "family_device_tier": min(tiers, key=TIER_ORDER.index),
            "derivatives": [
                {k: d[k] for k in ("id", "formats", "gguf_quants", "onnx_variants", "quant_bits", "ram_gb",
                                   "ram_gb_min", "device_tier", "downloads")}
                for d in ders[:8]
            ],
            "n_derivatives": len(ders),
        }
    return fields


def rebuild(con) -> int:
    """Recompute family fields for the whole catalog; only rows whose fields changed are re-indexed."""
    models = _load(con)
    fields = compute(models)
    changed = []
    for m in models:
        f = fields[m["id"]]
        old = {"family_root": m["old_root"], "family_formats": m["old_ff"], "family_ram_gb_min": m["old_fram"],
               "family_device_tier": m["old_ftier"], "derivatives": m["old_der"]}
        if any(old[k] != f[k] for k in old):
            changed.append((json.dumps(f), m["id"]))
    con.executemany(
        """update models set record = json_patch(record, ?), indexed_at = null where id = ?""", changed
    )
    con.commit()
    return len(changed)

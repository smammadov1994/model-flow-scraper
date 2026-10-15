"""Turns rows from any source (hub-stats parquet, HF API, X hook) into enriched catalog records."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Iterable

from . import catalog, enrich

CARD_KEYS = ("license", "license_name", "language", "base_model", "pipeline_tag", "library_name", "tags", "datasets")
GGUF_KEYS = ("total", "architecture", "context_length")


def _json(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return {}
    return v if v is not None else {}


def _iso(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.strftime("%Y-%m-%dT%H:%M:%SZ")
    return str(v)


def _strip_none(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: _strip_none(v) for k, v in d.items() if v is not None}
    return d


def normalize(row: dict) -> dict:
    """Accepts a hub-stats row or an HF /api/models item; returns the raw dict described in docs/SPEC.md."""
    card = _json(row.get("cardData")) or {}
    card = {k: card[k] for k in CARD_KEYS if k in card} if isinstance(card, dict) else {}
    cfg = _json(row.get("config")) or {}
    cfg = {k: cfg[k] for k in ("model_type", "architectures", "diffusers", "torch_dtype") if isinstance(cfg, dict) and k in cfg}
    gguf = _json(row.get("gguf")) or None
    if isinstance(gguf, dict):
        gguf = {k: gguf[k] for k in GGUF_KEYS if k in gguf} or None
    st = _strip_none(row.get("safetensors")) or None
    sib = row.get("siblings") or []
    files = [s["rfilename"] if isinstance(s, dict) else str(s) for s in sib]
    bm = row.get("baseModels")
    if isinstance(bm, list):  # API shape is sometimes a list of {relation, models}
        bm = bm[0] if bm else None
    if isinstance(bm, dict):
        bm = {"relation": bm.get("relation"), "models": [{"id": m.get("id")} for m in (bm.get("models") or []) if m]}
    ti = _strip_none(row.get("transformersInfo")) or None
    return {
        "id": row["id"],
        "author": row.get("author") or row["id"].split("/")[0],
        "pipeline_tag": row.get("pipeline_tag"),
        "library_name": row.get("library_name"),
        "tags": list(row.get("tags") or []),
        "siblings": files,
        "card": card,
        "config": cfg,
        "transformers_info": ti,
        "safetensors": st,
        "gguf": gguf,
        "base_models": bm,
        "downloads": int(row.get("downloads") or 0),
        "likes": int(row.get("likes") or 0),
        "trending": float(row.get("trendingScore") or 0.0),
        "created_at": _iso(row.get("createdAt")),
        "last_modified": _iso(row.get("lastModified")),
        "base_task": row.get("base_task"),
    }


def base_id(raw: dict) -> str | None:
    bm = raw.get("base_models") or {}
    models = bm.get("models") or []
    if models and models[0].get("id"):
        return models[0]["id"]
    b = (raw.get("card") or {}).get("base_model")
    if isinstance(b, list):
        b = b[0] if b else None
    return b if isinstance(b, str) else None


def build(raw: dict, con=None) -> dict:
    """Enrich one raw row. Applies cached LLM-extracted fields when present."""
    rec = enrich.build_record(raw)
    if con is not None:
        extra = catalog.cache_get(con, rec["id"], "card")
        if extra:
            apply_card_extraction(rec, extra)
        size = catalog.cache_get(con, rec["id"], "size")  # weight-file sizes (sizes.py) for repos with no param count
        if size:
            from .sizes import apply_size

            apply_size(rec, size, raw)
        social = catalog.cache_get(con, rec["id"], "social")  # X/Replicate overlay from the n8n joined file
        if social:
            from .sources.joined import apply_overlay

            apply_overlay(rec, social)
    return rec


def apply_card_extraction(rec: dict, extra: dict) -> None:
    """Fill gaps (never overwrite metadata the Hub already gave us) with LLM card-extraction output."""
    if not rec.get("task") and extra.get("task") in enrich.KNOWN_TASKS:
        rec["task"], rec["task_source"] = extra["task"], "llm"
        rec["modality"] = enrich.modality(rec["task"], rec.get("tags_clean") or [], rec.get("name") or "")
    if not rec.get("params") and isinstance(extra.get("params_b"), (int, float)) and extra["params_b"] > 0:
        rec["params"], rec["params_source"] = int(extra["params_b"] * 1e9), "llm"
        recompute_ram(rec)
    if not rec.get("license") and extra.get("license"):
        rec["license"] = str(extra["license"]).lower()
        rec["open_license"] = rec["license"] in enrich.PERMISSIVE_LICENSES
    if not rec.get("languages") and extra.get("languages"):
        rec["languages"] = [str(x).lower() for x in extra["languages"]][:20]
    if extra.get("summary"):
        rec["card_summary"] = str(extra["summary"])[:400]


def recompute_ram(rec: dict) -> None:
    rec["ram_gb"] = enrich.ram_variants(rec)
    variant_formats = enrich.variant_formats(rec)
    practical = {k: v for k, v in rec["ram_gb"].items() if enrich.practical(k)}
    rec["ram_gb_min"] = min(practical.values()) if practical else None
    rec["device_tier"] = enrich.device_tier(rec["ram_gb"], rec.get("formats") or [], variant_formats)


def ingest(con, rows: Iterable[dict], source: str) -> int:
    """Normalize + enrich + store. Returns count."""
    batch, n = [], 0
    for row in rows:
        raw = normalize(row)
        try:
            rec = build(raw, con)
        except Exception as e:  # one weird repo must never kill a 150k-row seed
            print(f"  ! skipped {raw['id']}: {e!r}")
            continue
        batch.append((rec, raw, source))
        if "inferenceProviderMapping" in row and source != "seed":  # live API rows: refresh hosted-API info
            from . import providers

            mapping = row["inferenceProviderMapping"]
            providers.store(con, rec["id"], mapping if isinstance(mapping, (list, dict)) else [])
        if len(batch) >= 2000:
            n += catalog.upsert(con, batch)
            con.commit()
            batch = []
    if batch:
        n += catalog.upsert(con, batch)
        con.commit()
    return n

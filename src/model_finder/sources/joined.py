"""Extra source: data/joined/models_joined.json written by the n8n workflow (HF + Replicate + X, see n8n/SCHEMA.md).

What we take from it:
  * `hf:` entries -> a "social" overlay (X buzz, one-line takes, post links, release status) merged into the
    catalog record for that repo. Stored in llm_cache(kind="social") so a reseed/poll never erases it.
    Repos we don't have yet are fetched from the Hub and indexed normally.
  * `x:` and `replicate:` entries -> their own records, so models announced on X (or API-only on Replicate)
    are searchable before they exist on Hugging Face. They have no files, so hardware filters skip them.
The n8n workflow owns that file; we only read it.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from .. import catalog, config, enrich, families, ingest

JOINED = config.DATA / "joined" / "models_joined.json"


def social_overlay(e: dict) -> dict | None:
    soc = e.get("social") or {}
    pop = e.get("popularity") or {}
    if not soc and not pop.get("replicate_runs"):
        return None
    posts = soc.get("top_posts") or []
    return {
        "status": e.get("status"),
        "x_buzz": round(float(soc.get("x_buzz") or pop.get("x_buzz") or 0), 1),
        "x_mentions": int(soc.get("x_mentions") or 0),
        "x_takes": [t for t in (soc.get("takes") or []) if t][:3],
        "x_posts": [p["url"] for p in posts if p.get("url")][:3],
        "x_first_seen": soc.get("first_seen"),
        "replicate_runs": int(pop.get("replicate_runs") or 0),
        "replicate_url": ((e.get("availability") or {}).get("replicate") or {}).get("url"),
    }


def apply_overlay(rec: dict, ov: dict) -> None:
    for k, v in ov.items():
        if v not in (None, [], 0, ""):
            rec[k] = v


def _ts(s: str | None) -> int:
    if not s:
        return 0
    try:
        return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return 0


def standalone_record(e: dict) -> dict:
    """Record for an X-only / Replicate-only entity (no Hugging Face repo yet)."""
    task = (e.get("task") or {}).get("primary")
    task = task if task in enrich.KNOWN_TASKS else None
    size = e.get("size") or {}
    hw = e.get("hardware") or {}
    lic = e.get("license") or {}
    links = e.get("links") or {}
    soc = e.get("social") or {}
    dates = e.get("dates") or {}
    created = dates.get("created_at") or soc.get("first_seen")
    url = links.get("replicate") or links.get("huggingface") or (links.get("x_posts") or [None])[0]
    tier = "unknown"
    for t in ("phone", "laptop", "gpu", "server"):
        if t in (hw.get("runs_on_any_variant") or []):
            tier = t
            break
    rec = {
        "id": e["id"], "author": None, "name": e.get("name") or e["id"].split(":", 1)[-1], "url": url,
        "task": task, "task_source": (e.get("task") or {}).get("source") or "unknown",
        "modality": enrich.modality(task, [], e.get("name") or "") if task else "other",
        "library": None, "params": size.get("params"), "params_source": size.get("params_source") or "unknown",
        "native_dtype": None, "formats": e.get("formats") or [], "gguf_quants": [], "onnx_variants": [],
        "quant_bits": None, "license": lic.get("id"), "open_license": lic.get("commercial_use") == "yes",
        "languages": e.get("languages") or [], "base_model": None, "relation": None, "is_adapter": False,
        "created_at": created, "created_ts": _ts(created), "last_modified": dates.get("last_modified"),
        "downloads": 0, "likes": 0, "trending": 0.0,
        "ram_gb": {}, "ram_gb_min": hw.get("est_min_ram_gb"), "ram_gb_native": None, "device_tier": tier,
        "tags_clean": (e.get("task") or {}).get("capabilities") or [], "datasets": [],
        "context_length": size.get("context_length"), "architecture": size.get("architecture"),
        "card_summary": e.get("description") or e.get("summary"),
        "source_kind": e["id"].split(":", 1)[0],
    }
    ov = social_overlay(e)
    if ov:
        apply_overlay(rec, ov)
    rec["status"] = e.get("status")
    return rec


def run(path: Path = JOINED, fetch_missing: int = 300) -> dict:
    if not path.exists():
        print(f"No joined file at {path} — run the n8n workflow first.")
        return {}
    data = json.loads(path.read_text())
    overlays = standalone = 0
    missing: list[str] = []
    with catalog.session() as con:
        known = {i for (i,) in con.execute("select id from models")}
        rows = []
        for e in data:
            kind, _, ident = e["id"].partition(":")
            if kind == "hf":
                ov = social_overlay(e)
                if ov:
                    catalog.cache_set(con, ident, "social", ov)
                    overlays += 1
                    if ident in known:
                        rec = catalog.get(con, ident)
                        apply_overlay(rec, ov)
                        rows.append((rec, None, "seed"))
                if ident not in known:
                    missing.append(ident)
            elif kind in ("x", "replicate"):
                rows.append((standalone_record(e), None, "n8n"))
                standalone += 1
        catalog.upsert(con, rows)
        catalog.kv_set(con, "last_joined", {"at": time.time(), "file_mtime": path.stat().st_mtime})
        con.commit()
    added = 0
    if missing and fetch_missing:
        from .. import hf_poll

        print(f"  fetching {min(len(missing), fetch_missing)} Hugging Face repos from the joined file that we don't have yet ...")
        added = hf_poll.add(missing[:fetch_missing], source="n8n")
    else:
        with catalog.session() as con:
            families.rebuild(con)
    print(f"Joined file: {overlays} X/Replicate overlays, {standalone} X/Replicate-only models, {added} new HF repos")
    return {"overlays": overlays, "standalone": standalone, "added": added}

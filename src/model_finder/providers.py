"""Hosted-API availability per model (Hugging Face Inference Providers: Together, Fireworks, hf-inference, ...).

Comes from the snapshot's `inferenceProviderMapping` (and the live API's expand of the same name): provider,
status, task, price per 1M tokens, measured speed/latency. Kept in its own catalog table so the search index
doesn't need rebuilding when it changes.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import catalog, config

SCHEMA = "create table if not exists providers (id text primary key, data text not null)"


def _compact(mapping) -> list[dict]:
    if isinstance(mapping, dict):  # the live API sometimes returns {provider: {...}}
        mapping = [{"provider": k, **(v or {})} for k, v in mapping.items()]
    out = []
    for m in mapping or []:
        if not m or m.get("status") != "live":
            continue
        det = m.get("providerDetails") or {}
        price = det.get("pricing") or {}
        perf = m.get("performance") or {}
        feats = m.get("features") or {}
        out.append({"name": m.get("provider"), "task": m.get("task"), "provider_id": m.get("providerId"),
                    "price_in": price.get("input"), "price_out": price.get("output"),
                    "tokens_per_s": perf.get("tokensPerSecond"), "latency_ms": perf.get("requestLatencyMs"),
                    "context": det.get("context_length"), "tools": bool(feats.get("toolCalling"))})
    out.sort(key=lambda p: (p["price_in"] is None, (p["price_in"] or 0) + (p["price_out"] or 0)))
    return out


def store(con, model_id: str, mapping) -> None:
    con.execute(SCHEMA)
    live = _compact(mapping)
    if live:
        con.execute("insert or replace into providers(id, data) values (?,?)", (model_id, json.dumps(live)))
    else:
        con.execute("delete from providers where id=?", (model_id,))


def refresh_from_parquet(path: Path = config.PARQUET) -> int:
    import duckdb

    rows = duckdb.connect().execute(
        "select id, inferenceProviderMapping from read_parquet(?) where len(inferenceProviderMapping) > 0", [str(path)]
    ).fetchall()
    n = 0
    with catalog.session() as con:
        con.execute(SCHEMA)
        con.execute("delete from providers")
        for mid, mapping in rows:
            live = _compact(mapping)
            if live:
                con.execute("insert into providers(id, data) values (?,?)", (mid, json.dumps(live)))
                n += 1
    print(f"Hosted-API availability: {n:,} models have a live inference provider")
    return n


def get_many(ids: list[str]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    with catalog.session() as con:
        con.execute(SCHEMA)
        for i in range(0, len(ids), 500):
            chunk = ids[i : i + 500]
            q = f"select id, data from providers where id in ({','.join('?' * len(chunk))})"
            out.update({k: json.loads(v) for k, v in con.execute(q, chunk)})
    return out

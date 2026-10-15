"""Incremental updates from the live Hub: newest models (by createdAt) and what's trending right now.

Uses the same /api/models endpoint as huggingface_hub.list_models, with `expand` so one request
returns 1,000 models *with* their file lists — no per-model calls. Anonymous limit is 500 requests / 5 min;
set HF_TOKEN for more.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import httpx

from . import catalog, config, families, ingest

API = "https://huggingface.co/api/models"
EXPAND = ["author", "baseModels", "cardData", "config", "createdAt", "downloads", "gguf", "lastModified",
          "library_name", "likes", "pipeline_tag", "safetensors", "siblings", "tags", "transformersInfo", "trendingScore"]


def _client() -> httpx.Client:
    tok = os.environ.get("HF_TOKEN")
    return httpx.Client(headers={"authorization": f"Bearer {tok}"} if tok else {}, timeout=60, follow_redirects=True)


def _pages(c: httpx.Client, params: dict, max_pages: int):
    url, q = API, {**params, "expand[]": EXPAND, "limit": 1000}
    for _ in range(max_pages):
        r = c.get(url, params=q)
        if r.status_code == 429:
            wait = int(r.headers.get("retry-after", "30"))
            print(f"  rate limited, sleeping {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        yield r.json()
        nxt = r.links.get("next", {}).get("url")
        if not nxt:
            return
        url, q = nxt, None


def fetch_one(model_id: str, c: httpx.Client | None = None) -> dict:
    c = c or _client()
    r = c.get(f"{API}/{model_id}", params={"expand[]": EXPAND})
    r.raise_for_status()
    return r.json()


def _ts(s: str | None) -> float:
    if not s:
        return 0.0
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def _worth_keeping(row: dict, notable: set[str], known: set[str]) -> bool:
    """New models have ~0 downloads, so 'popular' can't be the bar. Keep if anyone noticed it, if a
    well-known org made it, or if it's a converted copy of a model we already index."""
    if (row.get("downloads") or 0) >= config.RECENT_MIN_DOWNLOADS or (row.get("likes") or 0) >= config.RECENT_MIN_LIKES:
        return True
    if (row.get("author") or row["id"].split("/")[0]) in notable:
        return True
    sibs = row.get("siblings") or []
    if len(sibs) <= 2:
        return False
    bm = row.get("baseModels") or {}
    bm = bm[0] if isinstance(bm, list) and bm else bm
    base = ((bm or {}).get("models") or [{}])[0].get("id") if isinstance(bm, dict) else None
    return bool(base and base in known and (bm.get("relation") == "quantized"))


def run(max_new_pages: int = 30, trending_pages: int = 1) -> dict:
    with catalog.session() as con, _client() as c:
        since = catalog.kv_get(con, "poll_since")
        if since is None:  # first run: start from the newest model in the catalog (= the snapshot date)
            row = con.execute("select max(json_extract(record,'$.created_ts')) from models").fetchone()
            since = row[0] or (time.time() - 3 * 86400)
        notable = {a for (a,) in con.execute(
            "select json_extract(record,'$.author') a from models group by a having max(json_extract(record,'$.downloads')) >= ?",
            (config.NOTABLE_AUTHOR_DOWNLOADS,))}
        known = {i for (i,) in con.execute("select id from models")}
        print(f"Polling new models created after {datetime.fromtimestamp(since, timezone.utc):%Y-%m-%d %H:%M} UTC ...")
        seen = kept = 0
        newest = since
        rows: list[dict] = []
        done = False
        for page in _pages(c, {"sort": "createdAt", "direction": -1}, max_new_pages):
            for row in page:
                ts = _ts(row.get("createdAt"))
                if ts <= since:
                    done = True
                    break
                seen += 1
                newest = max(newest, ts)
                if _worth_keeping(row, notable, known):
                    rows.append(row)
            if done:
                break
        kept = ingest.ingest(con, rows, "poll")
        print(f"  {seen:,} new models on the Hub, {kept:,} worth indexing")
        trend_rows = [r for page in _pages(c, {"sort": "trendingScore", "direction": -1}, trending_pages) for r in page]
        trend = ingest.ingest(con, [r for r in trend_rows if len(r.get("siblings") or []) > 2], "trending")
        print(f"  refreshed {trend:,} trending models")
        catalog.kv_set(con, "poll_since", newest)
        catalog.kv_set(con, "last_update", time.time())
        changed = families.rebuild(con)
        print(f"  family links updated for {changed:,} models")
    return {"new_seen": seen, "new_kept": kept, "trending": trend}


def add(ids: list[str], source: str = "manual") -> int:
    """Index specific repos right now (used by `model-finder add` and by external hooks like X monitoring)."""
    rows = []
    with _client() as c:
        for i in ids:
            try:
                rows.append(fetch_one(i, c))
            except httpx.HTTPStatusError as e:
                print(f"  ! {i}: {e.response.status_code}")
    with catalog.session() as con:
        n = ingest.ingest(con, rows, source)
        families.rebuild(con)
    return n

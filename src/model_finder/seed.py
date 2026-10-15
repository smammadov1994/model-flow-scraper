"""Seed the catalog from the daily cfahlgren1/hub-stats snapshot (one 1.6 GB parquet, ~3M models)."""
from __future__ import annotations

import time
from pathlib import Path

import duckdb
import httpx

from . import catalog, config, ingest


def download(force: bool = False, url: str = config.HUB_STATS_URL, dest: Path = config.PARQUET) -> Path:
    """Download the snapshot unless we already have one from today (it refreshes daily)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force and time.time() - dest.stat().st_mtime < 20 * 3600:
        print(f"Using cached snapshot {dest} ({dest.stat().st_size / 1e9:.2f} GB)")
        return dest
    tmp = dest.with_suffix(".part")
    print(f"Downloading {url} ...")
    with httpx.stream("GET", url, follow_redirects=True, timeout=None) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        done, last = 0, 0.0
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if time.time() - last > 5:
                    last = time.time()
                    print(f"  {done / 1e9:.2f}/{total / 1e9:.2f} GB")
    tmp.rename(dest)
    return dest


SELECT_SQL = """
with m as (select * from read_parquet(?)),
notable as (
  select author from m where author is not null group by author having max(downloads) >= {notable}
),
kept as (
  select * from m
  where len(siblings) > 2
    and (downloads >= {min_dl} or likes >= {min_likes}
         or (createdAt >= now() - interval {recent} day
             and (downloads >= {r_dl} or likes >= {r_likes} or author in (select author from notable))))
)
select k.id, k.author, k.cardData, k.lastModified, k.likes, k.trendingScore, k.config, k.downloads,
       k.safetensors, k.tags, k.pipeline_tag, k.library_name, k.transformersInfo, k.siblings, k.createdAt,
       k.baseModels, k.gguf, b.pipeline_tag as base_task
from kept k
left join m b on b.id = k.baseModels.models[1].id
"""


def select_rows(parquet: Path = config.PARQUET, limit: int | None = None):
    sql = SELECT_SQL.format(
        notable=config.NOTABLE_AUTHOR_DOWNLOADS, min_dl=config.MIN_DOWNLOADS, min_likes=config.MIN_LIKES,
        recent=config.RECENT_DAYS, r_dl=config.RECENT_MIN_DOWNLOADS, r_likes=config.RECENT_MIN_LIKES,
    )
    if limit:
        sql += f" order by k.downloads desc limit {int(limit)}"
    con = duckdb.connect()
    con.execute("set memory_limit='4GB'")
    rel = con.execute(sql, [str(parquet)])
    cols = [d[0] for d in rel.description]
    while True:
        chunk = rel.fetchmany(5000)
        if not chunk:
            break
        for row in chunk:
            yield dict(zip(cols, row))


def run(force_download: bool = False, limit: int | None = None) -> int:
    path = download(force_download)
    t0 = time.time()
    with catalog.session() as con:
        print("Selecting usable models and enriching ...")
        n = 0
        rows = select_rows(path, limit)

        def counted():
            nonlocal n
            for r in rows:
                n += 1
                if n % 20000 == 0:
                    print(f"  {n:,} rows ({time.time() - t0:.0f}s)")
                yield r

        stored = ingest.ingest(con, counted(), "seed")
    from . import providers

    providers.refresh_from_parquet(path)
    with catalog.session() as con:
        catalog.kv_set(con, "last_seed", {"at": time.time(), "snapshot_mtime": path.stat().st_mtime, "rows": stored})
    print(f"Seeded {stored:,} models in {time.time() - t0:.0f}s")
    return stored

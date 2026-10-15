"""SQLite catalog: the source of truth for enriched model records, plus small caches and state.

Qdrant is rebuilt from here, so the index can always be thrown away and recreated.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from typing import Iterable, Iterator

from . import config

SCHEMA = """
create table if not exists models (
  id text primary key,
  record text not null,          -- enriched record (json), see docs/SPEC.md
  raw text,                      -- compact raw row kept for re-enrichment (json)
  source text,                   -- seed | poll | manual | x
  updated_at real not null,
  indexed_at real                -- null = needs (re)indexing
);
create index if not exists models_indexed on models(indexed_at);
create table if not exists kv (key text primary key, value text);
create table if not exists embed_cache (model text, hash text, vec blob, primary key (model, hash));
create table if not exists llm_cache (id text, kind text, value text, created_at real, primary key (id, kind));
"""


def connect(path=None) -> sqlite3.Connection:
    path = path or config.CATALOG_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=60)
    con.execute("pragma journal_mode=wal")
    con.execute("pragma synchronous=normal")
    con.executescript(SCHEMA)
    return con


@contextmanager
def session(path=None) -> Iterator[sqlite3.Connection]:
    con = connect(path)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def upsert(con, rows: Iterable[tuple[dict, dict | None, str]]) -> int:
    """rows: (record, raw, source). Marks each as needing re-indexing."""
    now = time.time()
    data = [(r["id"], json.dumps(r), json.dumps(raw) if raw is not None else None, src, now) for r, raw, src in rows]
    con.executemany(
        """insert into models(id, record, raw, source, updated_at, indexed_at) values (?,?,?,?,?,null)
           on conflict(id) do update set record=excluded.record, raw=coalesce(excluded.raw, models.raw),
             source=excluded.source, updated_at=excluded.updated_at, indexed_at=null""",
        data,
    )
    return len(data)


def get(con, model_id: str) -> dict | None:
    row = con.execute("select record from models where id=?", (model_id,)).fetchone()
    return json.loads(row[0]) if row else None


def get_many(con, ids: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for i in range(0, len(ids), 500):
        chunk = ids[i : i + 500]
        q = f"select id, record from models where id in ({','.join('?' * len(chunk))})"
        out.update({k: json.loads(v) for k, v in con.execute(q, chunk)})
    return out


def iter_records(con, only_unindexed=False, with_raw=False, batch=2000):
    where = "where indexed_at is null" if only_unindexed else ""
    cols = "record, raw" if with_raw else "record"
    cur = con.execute(f"select {cols} from models {where}")
    while True:
        rows = cur.fetchmany(batch)
        if not rows:
            return
        for row in rows:
            if with_raw:
                yield json.loads(row[0]), (json.loads(row[1]) if row[1] else None)
            else:
                yield json.loads(row[0])


def mark_indexed(con, ids: list[str]) -> None:
    now = time.time()
    con.executemany("update models set indexed_at=? where id=?", [(now, i) for i in ids])


def count(con) -> int:
    return con.execute("select count(*) from models").fetchone()[0]


def kv_get(con, key: str, default=None):
    row = con.execute("select value from kv where key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else default


def kv_set(con, key: str, value) -> None:
    con.execute("insert into kv(key, value) values (?,?) on conflict(key) do update set value=excluded.value",
                (key, json.dumps(value)))


def cache_get(con, model_id: str, kind: str):
    row = con.execute("select value from llm_cache where id=? and kind=?", (model_id, kind)).fetchone()
    return json.loads(row[0]) if row else None


def cache_set(con, model_id: str, kind: str, value) -> None:
    con.execute("insert or replace into llm_cache(id, kind, value, created_at) values (?,?,?,?)",
                (model_id, kind, json.dumps(value), time.time()))

"""Studio sessions (chat history + current pipeline) in a small SQLite file next to the catalog."""
from __future__ import annotations

import json
import secrets
import sqlite3
import time

from .. import config

PATH = config.DATA / "studio.db"


def _con() -> sqlite3.Connection:
    PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(PATH, timeout=30)
    con.execute("create table if not exists sessions (id text primary key, messages text, pipeline text, updated_at real)")
    cols = {r[1] for r in con.execute("pragma table_info(sessions)")}
    if "state" not in cols:  # v1.1: deployment choice + a goal waiting for the "where should it run?" answer
        con.execute("alter table sessions add column state text")
    return con


def new_id() -> str:
    return "s_" + secrets.token_hex(6)


def load(sid: str) -> dict | None:
    with _con() as con:
        row = con.execute("select messages, pipeline, state from sessions where id=?", (sid,)).fetchone()
    if not row:
        return None
    return {"session_id": sid, "messages": json.loads(row[0] or "[]"), "pipeline": json.loads(row[1]) if row[1] else None,
            "state": json.loads(row[2]) if row[2] else {}}


def save(sid: str, messages: list[dict], pipeline: dict | None, state: dict | None = None) -> None:
    with _con() as con:
        con.execute("insert or replace into sessions(id, messages, pipeline, updated_at, state) values (?,?,?,?,?)",
                    (sid, json.dumps(messages), json.dumps(pipeline) if pipeline else None, time.time(),
                     json.dumps(state or {})))

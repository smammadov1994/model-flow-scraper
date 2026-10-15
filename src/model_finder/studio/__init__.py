"""Model Studio: chat + canvas pipeline designer on top of Model Finder (see docs/STUDIO.md)."""
from __future__ import annotations

import time

from . import planner, resolve, store


def chat(session_id: str | None, message: str) -> dict:
    t0 = time.time()
    sess = store.load(session_id) if session_id else None
    sid = sess["session_id"] if sess else store.new_id()
    history = sess["messages"] if sess else []
    previous = sess["pipeline"] if sess else None
    plan, planner_name = planner.plan(history, message, previous)
    t1 = time.time()
    goal = (previous or {}).get("goal") or message
    pipeline = resolve.build(plan, goal, previous)
    pipeline["id"] = sid
    history = history + [{"role": "user", "text": message}, {"role": "assistant", "text": plan["reply"]}]
    store.save(sid, history, pipeline)
    return {"session_id": sid, "reply": plan["reply"], "pipeline": pipeline, "planner": planner_name,
            "timing_ms": {"plan": int((t1 - t0) * 1000), "models": pipeline.pop("resolve_ms", 0),
                          "total": int((time.time() - t0) * 1000)}}


def swap(session_id: str, node_id: str, model_id: str) -> dict:
    sess = _need(session_id)
    pipeline = resolve.swap(sess["pipeline"], node_id, model_id)
    store.save(session_id, sess["messages"], pipeline)
    return {"pipeline": pipeline}


def move(session_id: str, positions: dict) -> None:
    sess = _need(session_id)
    for n in sess["pipeline"]["nodes"]:
        if n["id"] in positions:
            pos = positions[n["id"]]
            n["position"] = {"x": float(pos["x"]), "y": float(pos["y"])}
    store.save(session_id, sess["messages"], sess["pipeline"])


def _need(session_id: str) -> dict:
    sess = store.load(session_id)
    if not sess or not sess.get("pipeline"):
        raise KeyError("Unknown session — start a new chat.")
    return sess

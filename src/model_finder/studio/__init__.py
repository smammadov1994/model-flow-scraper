"""Model Studio: chat + canvas pipeline designer on top of Model Finder (see docs/STUDIO.md)."""
from __future__ import annotations

import re
import time

from .. import jev
from . import hardware, planner, resolve, store

MODES = ("local", "phone", "cloud")
_LOCAL = r"\b(this (computer|mac|machine|laptop|pc)|my (mac|macbook|computer|laptop|pc|desktop)|locally|local(ly)?|offline|on[- ]prem|on my machine|macbook|imac|mac mini)\b"
_PHONE = r"\b(i?phone|android|mobile|smartphone|on[- ]device|ipad|tablet)\b"
_CLOUD = r"\b(cloud|hosted|api|apis|server(less)?|aws|gcp|azure|saas|scale|production|deploy(ed)? online|web app|website|rent(ed)? gpu)\b"


def ask_where() -> dict:
    hw = hardware.detect()
    return {
        "question": "Where should this run?",
        "options": [
            {"label": "This computer", "value": "local",
             "detail": f"{hw['label']} — only models that fit here (~{hw['budget_gb']:g} GB for models)"},
            {"label": "My phone", "value": "phone", "detail": "on-device models under ~4 GB"},
            {"label": "Cloud", "value": "cloud", "detail": "hosted APIs or a rented GPU — bigger models, pay per use"},
        ],
    }


def infer_mode(message: str) -> str | None:
    """Where the user wants it to run, if the message says so; None if unclear (then we ask)."""
    m = message.lower()
    hits = [mode for mode, pat in (("phone", _PHONE), ("local", _LOCAL), ("cloud", _CLOUD)) if re.search(pat, m)]
    if len(hits) == 1:
        return hits[0]
    if jev.available():
        from typesafe_sdk import Choice

        try:
            r = jev.client().system_one(state={"request": message}, questions={"where": Choice(
                instructions="Where does `request` say the AI pipeline must run?",
                criteria={"local": "on the user's own computer (laptop, Mac, desktop, offline, privately)",
                          "phone": "on a phone or tablet",
                          "cloud": "in the cloud, on a server, as a hosted API or web service",
                          "unclear": "the request does not say where it runs"})})
            a = r.answers["where"]
            if a.choice != "unclear" and a.confidence >= 0.6:
                return a.choice
        except Exception:  # noqa: BLE001
            pass
    return hits[0] if hits else None


def deployment_for(mode: str) -> dict:
    """mode -> deployment info + the device constraints the resolver uses (code decides, not the planner)."""
    if mode == "local":
        hw = hardware.detect()
        name = "This Mac" if hw["kind"] == "mac" else "This computer"
        return {"mode": "local", "label": f"{name} · {hw['label']}", "hardware": hw,
                "device": {"kind": hw["kind"], "label": f"{name} ({hw['chip']})", "memory_gb": hw["memory_gb"]}}
    if mode == "phone":
        return {"mode": "phone", "label": "Phone (on-device)", "hardware": None,
                "device": {"kind": "phone", "label": "phone", "memory_gb": None}}
    return {"mode": "cloud", "label": "Cloud · hosted APIs or GPU server", "hardware": None,
            "device": {"kind": "server", "label": "cloud", "memory_gb": None}}


def chat(session_id: str | None, message: str, deployment: str | None = None) -> dict:
    t0 = time.time()
    sess = store.load(session_id) if session_id else None
    sid = sess["session_id"] if sess else store.new_id()
    history = sess["messages"] if sess else []
    previous = sess["pipeline"] if sess else None
    state = dict((sess or {}).get("state") or {})
    history = history + [{"role": "user", "text": message}]

    mode = deployment if deployment in MODES else None
    if mode is None and state.get("awaiting_where"):
        mode = infer_mode(message)  # free-text answer to our question ("on my mac", "cloud please")
    if mode is None and not state.get("mode"):
        mode = infer_mode(message)
    if mode is None and not state.get("mode"):
        # nothing planned yet and we don't know the target: ask before spending a planning call
        state["awaiting_where"] = True
        state.setdefault("goal", message)
        ask = ask_where()
        history.append({"role": "assistant", "text": ask["question"]})
        store.save(sid, history, previous, state)
        return {"session_id": sid, "reply": ask["question"], "ask": ask, "pipeline": previous, "planner": None,
                "timing_ms": {"total": int((time.time() - t0) * 1000)}}

    if mode:
        state["mode"] = mode
    dep = deployment_for(state["mode"])
    goal = state.get("goal") or (previous or {}).get("goal") or message
    # the planner sees the original request (not just "Cloud") plus the decided target
    request = goal if state.pop("awaiting_where", False) else message
    target = {"local": f"run locally on {dep['label']}", "phone": "run on a phone, on-device",
              "cloud": "run in the cloud (hosted APIs or a rented GPU; no local hardware limits)"}[state["mode"]]
    plan, planner_name = planner.plan(history[:-1], f"{request}\n(Target: {target}.)", previous)
    plan["device"] = dep["device"]  # hardware comes from detection / the user's choice, never from the planner
    t1 = time.time()
    state["goal"] = goal
    pipeline = resolve.build(plan, goal, previous)
    pipeline["id"] = sid
    pipeline["deployment"] = {k: dep[k] for k in ("mode", "label", "hardware")}
    history.append({"role": "assistant", "text": plan["reply"]})
    store.save(sid, history, pipeline, state)
    return {"session_id": sid, "reply": plan["reply"], "ask": None, "pipeline": pipeline, "planner": planner_name,
            "timing_ms": {"plan": int((t1 - t0) * 1000), "models": pipeline.pop("resolve_ms", 0),
                          "total": int((time.time() - t0) * 1000)}}


def swap(session_id: str, node_id: str, model_id: str) -> dict:
    sess = _need(session_id)
    pipeline = resolve.swap(sess["pipeline"], node_id, model_id)
    store.save(session_id, sess["messages"], pipeline, sess.get("state"))
    return {"pipeline": pipeline}


def move(session_id: str, positions: dict) -> None:
    sess = _need(session_id)
    for n in sess["pipeline"]["nodes"]:
        if n["id"] in positions:
            pos = positions[n["id"]]
            n["position"] = {"x": float(pos["x"]), "y": float(pos["y"])}
    store.save(session_id, sess["messages"], sess["pipeline"], sess.get("state"))


def _need(session_id: str) -> dict:
    sess = store.load(session_id)
    if not sess or not sess.get("pipeline"):
        raise KeyError("Unknown session — start a new chat.")
    return sess

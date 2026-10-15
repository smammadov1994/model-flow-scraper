"""Local web page + JSON API (same origin)."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import catalog, index, jev, llm, query

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="Model Finder")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class Ask(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    device: str = "auto"
    limit: int = Field(default=8, ge=1, le=20)


@app.on_event("startup")
def warm():
    index.embedder()  # load the embedding model once, not on the first question


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/sample_response.json")
def sample():
    return FileResponse(STATIC / "sample_response.json")


@app.post("/api/ask")
async def ask(body: Ask):
    try:
        return await run_in_threadpool(query.ask, body.question, body.device, body.limit)
    except Exception as e:  # surface a readable error to the page
        return JSONResponse({"error": f"{e.__class__.__name__}: {e}"}, status_code=500)


def _iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if ts else None


@app.get("/api/stats")
def stats():
    with catalog.session() as con:
        seed = catalog.kv_get(con, "last_seed") or {}
        return {
            "models_indexed": index.count(),
            "last_seed": _iso(seed.get("snapshot_mtime")),
            "last_update": _iso(catalog.kv_get(con, "last_update")),
            "llm_backend": jev.name() if jev.available() else llm.name(),
        }


# ---------------------------------------------------------------- Model Studio (docs/STUDIO.md)

from fastapi.responses import RedirectResponse  # noqa: E402

from . import studio  # noqa: E402
from .studio import n8n as studio_n8n  # noqa: E402
from .studio import store as studio_store  # noqa: E402

STUDIO_DIR = STATIC / "studio"


class StudioChat(BaseModel):
    session_id: str | None = None
    message: str = Field(min_length=2, max_length=2000)


class StudioSwap(BaseModel):
    session_id: str
    node_id: str
    model_id: str


class StudioMove(BaseModel):
    session_id: str
    positions: dict[str, dict[str, float]]


def _err(e: Exception, status: int = 500) -> JSONResponse:
    return JSONResponse({"error": f"{e}" if isinstance(e, KeyError) else f"{e.__class__.__name__}: {e}"}, status_code=status)


@app.post("/api/studio/chat")
async def studio_chat(body: StudioChat):
    try:
        return await run_in_threadpool(studio.chat, body.session_id, body.message)
    except Exception as e:  # noqa: BLE001
        return _err(e)


@app.get("/api/studio/sessions/{sid}")
def studio_session(sid: str):
    sess = studio_store.load(sid)
    return sess if sess else _err(KeyError("Unknown session"), 404)


@app.post("/api/studio/swap")
def studio_swap(body: StudioSwap):
    try:
        return studio.swap(body.session_id, body.node_id, body.model_id)
    except KeyError as e:
        return _err(e, 404)


@app.post("/api/studio/move")
def studio_move(body: StudioMove):
    try:
        studio.move(body.session_id, body.positions)
        return {"ok": True}
    except KeyError as e:
        return _err(e, 404)


def _download(data: dict, filename: str) -> JSONResponse:
    return JSONResponse(data, headers={"Content-Disposition": f'attachment; filename="{filename}"'})


def _slug(s: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "-", (s or "pipeline").lower()).strip("-")[:40] or "pipeline"


@app.get("/api/studio/export/n8n/{sid}")
def studio_export_n8n(sid: str):
    sess = studio_store.load(sid)
    if not sess or not sess.get("pipeline"):
        return _err(KeyError("Unknown session"), 404)
    return _download(studio_n8n.export(sess["pipeline"]), f"{_slug(sess['pipeline']['title'])}.n8n.json")


@app.get("/api/studio/export/pipeline/{sid}")
def studio_export_pipeline(sid: str):
    sess = studio_store.load(sid)
    if not sess or not sess.get("pipeline"):
        return _err(KeyError("Unknown session"), 404)
    return _download(sess["pipeline"], f"{_slug(sess['pipeline']['title'])}.pipeline.json")


@app.get("/studio")
def studio_root():
    return RedirectResponse("/studio/")


if STUDIO_DIR.exists():
    app.mount("/studio", StaticFiles(directory=STUDIO_DIR, html=True), name="studio")

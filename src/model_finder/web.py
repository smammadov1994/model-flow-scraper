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

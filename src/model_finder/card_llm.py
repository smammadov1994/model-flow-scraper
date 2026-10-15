"""LLM pass over model cards (README) — only for fields still missing after rule-based enrichment.

Results are cached in catalog.llm_cache (kind="card"), so each card is read at most once.
"""
from __future__ import annotations

import json
import re
import time

import httpx

from . import catalog, families, ingest, llm
from .summarize import TASK_PLAIN

SCHEMA = {
    "type": "object",
    "properties": {
        "task": {"type": ["string", "null"], "enum": sorted(TASK_PLAIN) + [None]},
        "params_b": {"type": ["number", "null"]},
        "license": {"type": ["string", "null"]},
        "languages": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["task", "params_b", "license", "languages", "summary"],
}
SYSTEM = (
    "Extract facts about an AI model from its model card. Use null when the card does not say; never guess.\n"
    "- task: the Hugging Face task id of what the model does\n"
    "- params_b: number of parameters in billions, only if the card states it, else null\n"
    "- license: SPDX-like id, lower case (apache-2.0, mit, cc-by-nc-4.0, other)\n"
    "- languages: ISO 639-1 codes it supports\n"
    "- summary: one plain sentence (max 30 words) on what it does and what is special about it"
)


# Words that must appear in the card before we believe an LLM-assigned task (small local LLMs guess wildly).
TASK_EVIDENCE = {
    "text-to-speech": r"text[- ]to[- ]speech|\btts\b|speech synthesis|voice clon|synthesi[sz]e speech",
    "automatic-speech-recognition": r"speech recognition|\basr\b|transcri|speech[- ]to[- ]text",
    "text-generation": r"language model|\bllm\b|chat|instruct|text generation|causal",
    "text-to-image": r"text[- ]to[- ]image|image generation|diffusion|generate images",
    "image-to-video": r"image[- ]to[- ]video|animate",
    "text-to-video": r"text[- ]to[- ]video|video generation",
    "feature-extraction": r"embedding|retriev|semantic search|dense",
    "sentence-similarity": r"embedding|similarity|semantic search",
    "audio-classification": r"classif|speaker|emotion",
    "voice-activity-detection": r"voice activity|diariz|segmentation",
    "object-detection": r"detect",
    "image-classification": r"classif",
    "translation": r"translat",
}
_SIZE = re.compile(r"(\d+(?:\.\d+)?)\s*(m|b|million|billion)\b", re.I)


def verified(extra: dict, text: str) -> dict:
    """Keep only fields the card text actually supports."""
    out = dict(extra)
    low = text.lower()
    task = out.get("task")
    if task:
        pat = TASK_EVIDENCE.get(task, re.escape(task.replace("-", " ")) + "|" + re.escape(task))
        if not re.search(pat, low):
            out["task"] = None
    lic = out.get("license")
    if lic and str(lic).lower().replace("-", " ") not in low.replace("-", " "):
        out["license"] = None
    pb = out.get("params_b")
    if isinstance(pb, (int, float)) and pb > 0:
        sizes = [float(n) * (1e-3 if u.lower() in ("m", "million") else 1.0) for n, u in _SIZE.findall(text)]
        if not any(abs(sz - pb) <= 0.1 * pb for sz in sizes):
            out["params_b"] = None
    return out


def _readme(c: httpx.Client, model_id: str) -> str | None:
    r = c.get(f"https://huggingface.co/{model_id}/raw/main/README.md")
    if r.status_code != 200:
        return None
    text = re.sub(r"^---\n.*?\n---\n", "", r.text, count=1, flags=re.S)  # drop YAML front-matter (already have it)
    text = re.sub(r"<[^>]+>|!\[[^\]]*\]\([^)]*\)", " ", text)  # html + images
    return re.sub(r"\n{3,}", "\n\n", text).strip()[:6000]


def candidates(con, limit: int, include_summaries: bool = False) -> list[str]:
    cond = "json_extract(record,'$.task') is null or json_extract(record,'$.params') is null"
    if include_summaries:
        cond = "1=1"
    q = f"""select id from models
            where ({cond}) and id not in (select id from llm_cache where kind='card')
              and json_extract(record,'$.family_root') = id
            order by json_extract(record,'$.downloads') desc limit ?"""
    return [i for (i,) in con.execute(q, (limit,))]


def run(limit: int = 100, include_summaries: bool = False) -> int:
    if not llm.available():
        print("No LLM available (start Ollama or set ANTHROPIC_API_KEY) — skipping card extraction.")
        return 0
    done = 0
    t0 = time.time()
    with catalog.session() as con, httpx.Client(timeout=30, follow_redirects=True) as c:
        ids = candidates(con, limit, include_summaries)
        print(f"Reading {len(ids)} model cards with {llm.name()} ...")
        for mid in ids:
            try:
                text = _readme(c, mid)
                extra = {} if not text else verified(
                    llm.complete_json(SYSTEM, f"Model: {mid}\n\nCard:\n{text}", SCHEMA, max_tokens=400), text)
            except Exception as e:
                print(f"  ! {mid}: {e!r}")
                continue
            catalog.cache_set(con, mid, "card", extra)
            row = con.execute("select raw from models where id=?", (mid,)).fetchone()
            if row and row[0]:
                rec = ingest.build(json.loads(row[0]), con)
                catalog.upsert(con, [(rec, None, "seed")])
            done += 1
            if done % 10 == 0:
                con.commit()
                print(f"  {done}/{len(ids)} ({(time.time() - t0) / done:.1f}s each)")
        families.rebuild(con)
    print(f"Card extraction done for {done} models.")
    return done

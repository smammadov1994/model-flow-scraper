"""Fill missing parameter counts from weight-file sizes (one Hub API call per model, most-downloaded first).

~15% of indexed models (often popular older ones: CLIP, XTTS, wav2vec2) publish no safetensors/GGUF parameter
count and no size in the name. Their weight files still tell us roughly how big they are.
Results are cached in llm_cache(kind="size"), so reseeds keep them.
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from . import catalog, enrich, families, ingest
from .hf_poll import _client

WEIGHT_EXT = (".safetensors", ".bin", ".pth", ".pt", ".ckpt", ".onnx", ".gguf", ".nemo", ".msgpack", ".h5")
SKIP = re.compile(r"(optimizer|scheduler|rng_state|training_args|onnx_data_)|(^|/)(samples?|examples?|voices?)/", re.I)
FORMAT_PREF = [".safetensors", ".bin", ".pth", ".pt", ".ckpt", ".nemo", ".onnx", ".gguf", ".msgpack", ".h5"]


def weight_bytes(siblings: list[dict]) -> tuple[int, str | None]:
    """Total bytes of ONE copy of the weights: pick a single format and skip duplicate precision variants."""
    files = [(s["rfilename"], s.get("size") or 0) for s in siblings
             if s["rfilename"].lower().endswith(WEIGHT_EXT) and not SKIP.search(s["rfilename"])]
    if not files:
        return 0, None
    for ext in FORMAT_PREF:
        group = [(f, b) for f, b in files if f.lower().endswith(ext)]
        if not group:
            continue
        names = {f for f, _ in group}
        # if both model.safetensors and model.fp16.safetensors exist, count only the full-precision one
        dedup = [(f, b) for f, b in group
                 if not (re.search(r"[._-](fp16|f16|bf16|half|int8|q\d\w*|quantized)\b", f.lower())
                         and re.sub(r"[._-](fp16|f16|bf16|half|int8|q\d\w*|quantized)", "", f) in names)]
        if ext == ".gguf":  # many quants of the same model: the smallest whole file is closest to one copy
            dedup = [min(dedup, key=lambda x: x[1])]
        if ext == ".onnx":
            dedup = [max(dedup, key=lambda x: x[1])]
        return sum(b for _, b in dedup), dedup[0][0]
    return 0, None


def _fetch(c: httpx.Client, mid: str) -> tuple[str, dict | None]:
    for _ in range(3):
        r = c.get(f"https://huggingface.co/api/models/{mid}", params={"blobs": "true"})
        if r.status_code == 429:
            time.sleep(int(r.headers.get("retry-after", "30")))
            continue
        if r.status_code != 200:
            return mid, None
        total, sample = weight_bytes(r.json().get("siblings") or [])
        return mid, {"total_bytes": total, "sample_file": sample}
    return mid, None


def apply_size(rec: dict, size: dict, raw: dict | None) -> None:
    if rec.get("params") or not size or not size.get("total_bytes"):
        return
    dtype_hint = ((raw or {}).get("config") or {}).get("torch_dtype")
    rec["params"] = enrich.params_from_size(size["total_bytes"], dtype_hint, size.get("sample_file"))
    rec["params_source"] = "file_size"
    ingest.recompute_ram(rec)


def run(limit: int = 1500, workers: int = 4) -> int:
    t0 = time.time()
    with catalog.session() as con:
        ids = [i for (i,) in con.execute(
            """select id from models where json_extract(record,'$.params') is null
                 and coalesce(json_extract(record,'$.family_root'), id) = id and instr(id, ':') = 0
                 and id not in (select id from llm_cache where kind='size')
               order by json_extract(record,'$.downloads') desc limit ?""", (limit,))]
        print(f"Fetching weight-file sizes for {len(ids)} models without a parameter count ...", flush=True)
        done = 0
        with _client() as c, ThreadPoolExecutor(workers) as ex:
            for mid, size in ex.map(lambda m: _fetch(c, m), ids):
                if size is None:
                    continue
                catalog.cache_set(con, mid, "size", size)
                row = con.execute("select raw from models where id=?", (mid,)).fetchone()
                raw = json.loads(row[0]) if row and row[0] else None
                rec = ingest.build(raw, con) if raw else catalog.get(con, mid)
                catalog.upsert(con, [(rec, None, "seed")])
                done += 1
                if done % 200 == 0:
                    con.commit()
                    print(f"  {done}/{len(ids)} ({time.time() - t0:.0f}s)", flush=True)
        con.commit()
        print(f"  sizes added for {done} models; re-linking families ... {families.rebuild(con):,} changed", flush=True)
    return done

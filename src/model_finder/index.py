"""Qdrant index: dense embedding of the summary + server-side BM25 on the same text, plus payload filters."""
from __future__ import annotations

import hashlib
import time
import uuid
from functools import lru_cache

import numpy as np
from qdrant_client import QdrantClient, models

from . import catalog, config
from .summarize import build_summary

NS = uuid.UUID("6f1d2c4e-8a51-4d0b-9a0e-3c2b7f9d1e11")

KEYWORD_FIELDS = ["task", "modality", "family_formats", "formats", "family_device_tier", "license", "languages",
                  "family_root", "author", "library"]
FLOAT_FIELDS = ["family_ram_gb_min", "ram_gb_min"]
INT_FIELDS = ["params", "created_ts", "downloads", "likes"]
BOOL_FIELDS = ["open_license", "is_adapter", "has_dense"]


def point_id(model_id: str) -> str:
    return str(uuid.uuid5(NS, model_id))


@lru_cache(maxsize=1)
def client() -> QdrantClient:
    # cloud_inference=True: Document(...) objects are sent as text and embedded *by the Qdrant server*
    # (we only use that for BM25; dense vectors are computed here, see embedder()).
    return QdrantClient(url=config.QDRANT_URL, cloud_inference=True, timeout=120)


class _FastEmbed:
    """Small ONNX models on CPU (e.g. BAAI/bge-small-en-v1.5)."""

    def __init__(self, name: str):
        from fastembed import TextEmbedding

        self.m = TextEmbedding(name)
        self.dim = len(next(iter(self.m.embed(["x"]))))

    def docs(self, texts):
        return [v.tolist() for v in self.m.embed(texts, batch_size=128)]

    def query(self, text):
        return next(iter(self.m.query_embed(text))).tolist()


class _SentenceTransformer:
    """Larger models via sentence-transformers on the Mac GPU (MPS) in fp16, e.g. Qwen/Qwen3-Embedding-0.6B.
    Queries get the model's own "query" instruction prompt; documents get none (Qwen3's recommended setup)."""

    def __init__(self, name: str):
        import torch
        from sentence_transformers import SentenceTransformer

        dev = config.DENSE_DEVICE or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        kw = {"torch_dtype": torch.float16} if dev != "cpu" else {}
        self.m = SentenceTransformer(name, device=dev, model_kwargs=kw)
        dim_fn = getattr(self.m, "get_embedding_dimension", None) or self.m.get_sentence_embedding_dimension
        self.dim = dim_fn()

    def docs(self, texts):
        return self.m.encode_document(texts, batch_size=32, normalize_embeddings=True).tolist()

    def query(self, text):
        return self.m.encode_query([text], normalize_embeddings=True)[0].tolist()


FASTEMBED_MODELS = {"BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"}


@lru_cache(maxsize=1)
def embedder():
    name = config.DENSE_MODEL
    return _FastEmbed(name) if name in FASTEMBED_MODELS else _SentenceTransformer(name)


def dense_dim() -> int:
    return embedder().dim


def embed(texts: list[str]) -> list[list[float]]:
    """Document vectors, cached by (model, text hash) in the catalog so re-indexing only embeds changed text."""
    if not texts:
        return []
    keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
    out: dict[str, list[float]] = {}
    with catalog.session() as con:
        for i in range(0, len(keys), 500):
            chunk = keys[i : i + 500]
            q = f"select hash, vec from embed_cache where model=? and hash in ({','.join('?' * len(chunk))})"
            for h, blob in con.execute(q, [config.DENSE_MODEL, *chunk]):
                out[h] = np.frombuffer(blob, dtype=np.float16).astype(np.float32).tolist()
        todo = sorted({k: t for k, t in zip(keys, texts) if k not in out}.items(), key=lambda kv: len(kv[1]))
        if todo:  # length-sorted batches waste less padding
            vecs = embedder().docs([t for _, t in todo])
            con.executemany("insert or replace into embed_cache(model, hash, vec) values (?,?,?)",
                            [(config.DENSE_MODEL, k, np.asarray(v, dtype=np.float16).tobytes()) for (k, _), v in zip(todo, vecs)])
            out.update({k: v for (k, _), v in zip(todo, vecs)})
    return [out[k] for k in keys]


def embed_query(text: str) -> list[float]:
    return embedder().query(text)


def ensure_collection(recreate: bool = False) -> None:
    c = client()
    if recreate and c.collection_exists(config.COLLECTION):
        c.delete_collection(config.COLLECTION)
    if c.collection_exists(config.COLLECTION):
        have = c.get_collection(config.COLLECTION).config.params.vectors["dense"].size
        if have != dense_dim():
            raise SystemExit(f"Collection '{config.COLLECTION}' has {have}-dim vectors but {config.DENSE_MODEL} makes "
                             f"{dense_dim()}-dim ones. Rebuild with:  model-finder index --recreate")
        return
    c.create_collection(
        config.COLLECTION,
        vectors_config={"dense": models.VectorParams(size=dense_dim(), distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    schema = {
        **{f: models.PayloadSchemaType.KEYWORD for f in KEYWORD_FIELDS},
        **{f: models.PayloadSchemaType.FLOAT for f in FLOAT_FIELDS},
        **{f: models.PayloadSchemaType.INTEGER for f in INT_FIELDS},
        **{f: models.PayloadSchemaType.BOOL for f in BOOL_FIELDS},
    }
    for field, typ in schema.items():
        c.create_payload_index(config.COLLECTION, field, field_schema=typ)


def wants_dense(r: dict) -> bool:
    """Semantic (dense) vectors cost ~50 ms each on a laptop CPU, so the first build gives them to the models
    people actually use or that are new and noticed. Everything still gets keyword (BM25) search, which Qdrant
    computes itself in microseconds. `model-finder index --backfill` adds dense vectors to the rest later."""
    if config.DENSE_ALL:
        return True
    recent = (r.get("created_ts") or 0) > time.time() - 30 * 86400
    return bool(
        (r.get("downloads") or 0) >= 2000 or (r.get("likes") or 0) >= 30 or (r.get("n_derivatives") or 0) >= 2
        or (recent and ((r.get("downloads") or 0) >= 100 or (r.get("likes") or 0) >= 5))
        or (r.get("x_buzz") or 0) > 0
    )


def _points(records: list[dict], force_dense: bool = False) -> list[models.PointStruct]:
    texts = [build_summary(r, r.get("card_summary")) for r in records]
    dense_idx = [i for i, r in enumerate(records) if force_dense or wants_dense(r)]
    dense = dict(zip(dense_idx, embed([texts[i] for i in dense_idx]))) if dense_idx else {}
    pts = []
    for i, (r, t) in enumerate(zip(records, texts)):
        payload = dict(r)
        payload["summary"] = t
        payload["has_dense"] = i in dense
        vec = {"bm25": models.Document(text=t, model=config.BM25_MODEL)}
        if i in dense:
            vec["dense"] = dense[i]
        pts.append(models.PointStruct(id=point_id(r["id"]), vector=vec, payload=payload))
    return pts


def _priority_sql() -> str:
    # most-used first, so a half-finished build is already useful
    return ("coalesce(json_extract(record,'$.downloads'),0) + 50 * coalesce(json_extract(record,'$.likes'),0) "
            "+ 1000 * coalesce(json_extract(record,'$.x_buzz'),0)")


def sync(recreate: bool = False, batch: int = 256) -> int:
    """Index every catalog row that changed since it was last indexed.

    Only one point per model family is embedded: the original model (its converted copies ride along in
    `derivatives` / `family_formats`), plus copies whose original we don't have.
    """
    ensure_collection(recreate)
    c = client()
    n, nd, t0 = 0, 0, time.time()
    with catalog.session() as con:
        if recreate:
            con.execute("update models set indexed_at = null")
        rows = con.execute(
            "select id, coalesce(json_extract(record,'$.family_root'), id) = id from models where indexed_at is null "
            f"order by {_priority_sql()} desc"
        ).fetchall()
        ids = [i for i, is_root in rows if is_root]
        copies = [i for i, is_root in rows if not is_root]
        if copies and not recreate:
            for i in range(0, len(copies), 1000):
                delete(copies[i : i + 1000])  # in case it was indexed before it got linked to an original
        catalog.mark_indexed(con, copies)
        con.commit()
        todo = len(ids)
        print(f"Indexing {todo:,} models into Qdrant ({config.QDRANT_URL}, collection '{config.COLLECTION}'); "
              f"{len(copies):,} converted copies are folded into their originals ...", flush=True)
        for i in range(0, len(ids), batch):
            recs = catalog.get_many(con, ids[i : i + batch])
            recs = [recs[k] for k in ids[i : i + batch] if k in recs]
            pts = _points(recs)
            nd += sum(1 for p in pts if p.payload["has_dense"])
            c.upsert(config.COLLECTION, points=pts, wait=False)
            catalog.mark_indexed(con, [r["id"] for r in recs])
            con.commit()
            n += len(recs)
            if n % (batch * 10) == 0 or n == len(ids):
                rate = n / max(time.time() - t0, 1e-6)
                print(f"  {n:,}/{todo:,} ({nd:,} with semantic vectors; {rate:.0f}/s, "
                      f"~{(todo - n) / max(rate, 1e-6) / 60:.1f} min left)", flush=True)
        catalog.kv_set(con, "last_index", time.time())
    return n


def backfill_dense(limit: int | None = None, batch: int = 128) -> int:
    """Add semantic vectors to points that only have keyword vectors (most popular first)."""
    c = client()
    flt = models.Filter(must=[models.FieldCondition(key="has_dense", match=models.MatchValue(value=False))])
    todo = c.count(config.COLLECTION, count_filter=flt, exact=True).count
    limit = min(limit or todo, todo)
    print(f"Adding semantic vectors to {limit:,} of {todo:,} keyword-only models ...", flush=True)
    done, t0 = 0, time.time()
    while done < limit:
        pts, _ = c.scroll(config.COLLECTION, scroll_filter=flt, limit=min(batch, limit - done),
                          with_payload=["summary", "id"], order_by=models.OrderBy(key="downloads", direction="desc"))
        if not pts:
            break
        vecs = embed([p.payload["summary"] for p in pts])
        c.update_vectors(config.COLLECTION, points=[models.PointVectors(id=p.id, vector={"dense": v}) for p, v in zip(pts, vecs)])
        c.set_payload(config.COLLECTION, payload={"has_dense": True}, points=[p.id for p in pts])
        done += len(pts)
        if done % (batch * 10) == 0 or done >= limit:
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  {done:,}/{limit:,} ({rate:.0f}/s, ~{(limit - done) / max(rate, 1e-6) / 60:.1f} min left)", flush=True)
    return done


def delete(ids: list[str]) -> None:
    client().delete(config.COLLECTION, points_selector=models.PointIdsList(points=[point_id(i) for i in ids]))


def count() -> int:
    c = client()
    if not c.collection_exists(config.COLLECTION):
        return 0
    return c.count(config.COLLECTION, exact=False).count

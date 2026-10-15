"""Dense-retrieval benchmark of small embedding models on model-finder's own joined data.

Corpus: search_text of every original model in data/joined/models_joined.json (copies skipped).
Queries: plain-English requests; relevance comes from the record metadata (task / capability / device),
so no hand labelling. Metric: precision@10 (share of top-10 hits that fit) and MRR@10.
"""
import gc
import json
import sys
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

JOINED = sys.argv[1]
ONLY = sys.argv[2].split(",") if len(sys.argv) > 2 else None

recs = [r for r in json.load(open(JOINED)) if not ((r.get("lineage") or {}).get("variant_of"))]
docs = [r["search_text"] for r in recs]


def T(*tasks):
    return lambda r: bool(set(r["task"]["all"] or []) & set(tasks)) or r["task"]["primary"] in tasks


def cap(word):
    return lambda r: word in (r["task"]["capabilities"] or [])


def dev(tier):
    return lambda r: tier in (r["hardware"].get("runs_on_any_variant") or [])


def both(*fs):
    return lambda r: all(f(r) for f in fs)


def any_(*fs):
    return lambda r: any(f(r) for f in fs)


ASR = T("automatic-speech-recognition")
TTS = T("text-to-speech")
QUERIES = [
    ("I am looking to make a voice model on my phone", both(any_(TTS, ASR, T("audio-to-audio")), dev("phone"))),
    ("make my app talk out loud with a natural voice", TTS),
    ("clone my own voice from a short recording", both(TTS, cap("voice cloning"))),
    ("turn my podcast recordings into written transcripts", ASR),
    ("offline dictation on an iPhone", both(ASR, dev("phone"))),
    ("transcribe a meeting and tell me who said what", any_(ASR, T("voice-activity-detection"))),
    ("a chatbot I can run on my MacBook", both(T("text-generation", "image-text-to-text", "any-to-any"), dev("laptop"))),
    ("help me write and debug python code", both(T("text-generation"), cap("coding"))),
    ("a model that is good at step by step math reasoning", both(T("text-generation", "image-text-to-text"), cap("reasoning"))),
    ("assistant that can call tools and APIs", both(T("text-generation", "image-text-to-text"), cap("tool calling"))),
    ("generate pictures from a text prompt", T("text-to-image")),
    ("edit a photo by describing the change", T("image-to-image")),
    ("upscale and restore old blurry photos", T("image-to-image")),
    ("cut out the background of product photos", T("image-segmentation", "mask-generation", "image-to-image")),
    ("make a short video clip from a prompt", T("text-to-video")),
    ("animate a still image into a video", T("image-to-video")),
    ("compose a song with vocals", T("text-to-audio")),
    ("read the text off scanned receipts", T("image-text-to-text", "image-to-text", "document-question-answering")),
    ("describe what is in a photo", T("image-text-to-text", "image-to-text", "visual-question-answering")),
    ("count people in security camera footage", T("object-detection", "zero-shot-object-detection")),
    ("estimate how far away things are in a picture", T("depth-estimation")),
    ("semantic search over my notes for a RAG app", T("feature-extraction", "sentence-similarity")),
    ("rerank search results to improve relevance", T("text-ranking")),
    ("tag customer emails as positive or negative", T("text-classification", "zero-shot-classification")),
    ("find names and phone numbers in documents", T("token-classification")),
    ("turn a photo into a 3D model", T("image-to-3d", "text-to-3d")),
    ("forecast next month's sales numbers", T("time-series-forecasting")),
    ("a voice assistant I can talk to in real time", any_(T("any-to-any", "audio-text-to-text"), ASR, TTS)),
    ("small model to run on a phone for chatting", both(T("text-generation", "image-text-to-text", "any-to-any"), dev("phone"))),
]

MODELS = {
    # name: (repo, kwargs, query_prompt, doc_prompt) ; prompts given as literal prefixes or ("name", ...)
    "potion-base-8M (static, 8M)": ("minishlab/potion-base-8M", {}, "", ""),
    "all-MiniLM-L6-v2 (23M)": ("sentence-transformers/all-MiniLM-L6-v2", {}, "", ""),
    "bge-small-en-v1.5 (33M, current)": ("BAAI/bge-small-en-v1.5", {},
                                         "Represent this sentence for searching relevant passages: ", ""),
    "granite-embedding-small-english-r2 (48M)": ("ibm-granite/granite-embedding-small-english-r2", {}, "", ""),
    "UltraBERT-Embed-150M": ("dmis-lab/UltraBERT-Embed-150M", {}, None, None),
    "jina-embeddings-v5-text-nano (212M, non-commercial)": ("jinaai/jina-embeddings-v5-text-nano",
                                                             {"trust_remote_code": True}, ("query",), ("document",),
                                                             {"task": "retrieval"}),
    "EmbeddingGemma 2 text-only (270M)": ("google/embeddinggemma-2",
                                          {"config_kwargs": {"vision_config": None, "audio_config": None}},
                                          ("SearchQuery",), ("Document",)),
    "snowflake-arctic-embed-m-v2.0 (305M)": ("Snowflake/snowflake-arctic-embed-m-v2.0",
                                             {"trust_remote_code": True, "config_kwargs": {
                                                 "use_memory_efficient_attention": False, "unpad_inputs": False}},
                                             ("query",), ""),
    "nomic-embed-text-v2-moe (475M)": ("nomic-ai/nomic-embed-text-v2-moe", {"trust_remote_code": True},
                                       ("query",), ("passage",)),
    "Qwen3-Embedding-0.6B (596M)": ("Qwen/Qwen3-Embedding-0.6B", {}, ("query",), ""),
}


def encode(model, texts, prompt, bs, ekw=None):
    kw = {**(ekw or {}), "batch_size": bs, "normalize_embeddings": True, "convert_to_numpy": True, "show_progress_bar": False}
    if prompt is None:  # let the model's own query/document prompts apply
        return None
    if isinstance(prompt, tuple):
        return model.encode(texts, prompt_name=prompt[0], **kw)
    return model.encode([prompt + t for t in texts] if prompt else texts, **kw)


N_REL = [sum(1 for r in recs if rel(r)) for _, rel in QUERIES]
results = []
device = "mps" if torch.backends.mps.is_available() else "cpu"
for name, (repo, kwargs, qp, dp, *extra) in MODELS.items():
    ekw = extra[0] if extra else {}
    if ONLY and not any(o.lower() in name.lower() for o in ONLY):
        continue
    try:
        t0 = time.time()
        model = SentenceTransformer(repo, device=device, **kwargs)
        load_s = time.time() - t0
        bs = 64
        t0 = time.time()
        D = encode(model, docs, dp, bs, ekw)
        if D is None:
            D = model.encode_document(docs, batch_size=bs, normalize_embeddings=True, show_progress_bar=False)
        doc_s = time.time() - t0
        qs = [q for q, _ in QUERIES]
        t0 = time.time()
        Q = encode(model, qs, qp, 32, ekw)
        if Q is None:
            Q = model.encode_query(qs, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        q_ms = (time.time() - t0) / len(qs) * 1000
        S = np.asarray(Q) @ np.asarray(D).T
        p10, mrr, per = [], [], []
        for (q, rel), row, n_rel in zip(QUERIES, S, N_REL):
            top = np.argsort(-row)[:10]
            hits = [bool(rel(recs[i])) for i in top]
            p10.append(sum(hits[:min(10, n_rel)]) / min(10, n_rel))  # fair when <10 relevant exist
            mrr.append(next((1 / (k + 1) for k, h in enumerate(hits) if h), 0.0))
            per.append(sum(hits))
        params = sum(p.numel() for p in model.parameters())
        res = {"model": name, "params_M": round(params / 1e6), "dim": int(np.asarray(D).shape[1]),
               "p@10": round(float(np.mean(p10)), 3), "mrr@10": round(float(np.mean(mrr)), 3),
               "index_s": round(doc_s, 1), "query_ms": round(q_ms, 1), "load_s": round(load_s, 1),
               "per_query_hits": per}
        print(json.dumps(res), flush=True)
        results.append(res)
        del model
    except Exception as e:  # keep going if one model will not load
        print(json.dumps({"model": name, "error": f"{type(e).__name__}: {str(e)[:300]}"}), flush=True)
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

print(f"corpus={len(docs)} queries={len(QUERIES)} device={device}")

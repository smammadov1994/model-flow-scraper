# Embedding model benchmark (2026-10-15)

Dense-only retrieval on model-finder's own data: 2,173 original models (`search_text` from
`data/joined/models_joined.json`), 29 plain-English requests. Relevance comes from each record's
metadata (task, capability, device tier). Run on an M1 Pro (MPS).

| Model | Params | Dim | P@10 | MRR@10 | Index 2,173 docs | Query | License |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3-Embedding-0.6B | 596M | 1024 | **0.667** | **0.730** | 241 s | 35 ms | Apache-2.0 |
| ibm-granite/granite-embedding-small-english-r2 | 48M | 384 | 0.618 | 0.683 | 9 s | 7 ms | Apache-2.0 |
| jinaai/jina-embeddings-v5-text-nano | 239M | 768 | 0.615 | 0.679 | 22 s | 13 ms | CC-BY-NC-4.0 |
| google/embeddinggemma-2 (text-only) | 271M | 768 | 0.611 | 0.630 | 70 s | 10 ms | Apache-2.0 |
| BAAI/bge-small-en-v1.5 (current default) | 33M | 384 | 0.560 | 0.730 | 6 s | 5 ms | MIT |
| minishlab/potion-base-8M (static) | 8M | 256 | 0.537 | 0.671 | 1 s | 0.1 ms | MIT |
| sentence-transformers/all-MiniLM-L6-v2 | 23M | 384 | 0.532 | 0.621 | 4 s | 4 ms | Apache-2.0 |

Not run: nomic-embed-text-v2-moe and snowflake-arctic-embed-m-v2.0 (their remote code breaks on
transformers 5.x), dmis-lab/UltraBERT-Embed-150M (needs untrusted remote code), microsoft/bitnet-embedding
(needs a BitNet runtime).

Caveats: 29 queries, so gaps under ~0.03 are noise. All models do badly on device-constrained
queries ("on my phone"); those need payload filters, not better embeddings. BM25 + filters
in the real pipeline will narrow the gaps.

Re-run: a venv with `sentence-transformers>=6.1 transformers>=5.19 torch torchvision pillow peft model2vec`, then
`python -I bench.py ../../data/joined/models_joined.json [name-filter,...]`.

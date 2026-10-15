"""model-finder command line.

  model-finder setup            # download snapshot, enrich, link families, read top cards, build index
  model-finder ask "voice model on my phone" [--device phone] [--json]
  model-finder update           # pull models newer than the snapshot + trending, re-index changes
  model-finder serve            # local web page on http://localhost:8765
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time

from . import config


def _qdrant_ok() -> bool:
    import httpx

    try:
        return httpx.get(config.QDRANT_URL, timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


def _need_qdrant():
    if not _qdrant_ok():
        sys.exit(f"Qdrant is not reachable at {config.QDRANT_URL}. Start it with:  docker compose up -d")


def _joined():
    from .sources import joined

    if joined.JOINED.exists():
        joined.run()


def cmd_joined(a):
    from . import index
    from .sources import joined

    _need_qdrant()
    joined.run(fetch_missing=a.fetch)
    index.sync()


def cmd_setup(a):
    from . import card_llm, catalog, families, index, seed, sizes

    _need_qdrant()
    t0 = time.time()
    seed.run(force_download=a.refresh, limit=a.limit)
    with catalog.session() as con:
        print(f"Linking converted copies to their originals ... {families.rebuild(con):,} updated")
    if a.sizes:
        sizes.run(limit=a.sizes)
    if a.cards:
        card_llm.run(limit=a.cards)
    _joined()
    index.sync(recreate=a.recreate)
    print(f"Setup finished in {(time.time() - t0) / 60:.1f} min. Try:  model-finder ask \"voice model on my phone\"")


def cmd_update(a):
    from . import card_llm, hf_poll, index

    _need_qdrant()
    hf_poll.run(max_new_pages=a.pages)
    _joined()
    if a.cards:
        card_llm.run(limit=a.cards)
    index.sync()


def cmd_reseed(a):
    a.limit, a.recreate, a.sizes = None, False, 300
    a.refresh = True
    cmd_setup(a)


def cmd_index(a):
    from . import index

    _need_qdrant()
    if a.backfill is not None:
        index.backfill_dense(limit=a.backfill or None)
    else:
        index.sync(recreate=a.recreate)


def cmd_sizes(a):
    from . import index, sizes

    sizes.run(limit=a.limit)
    index.sync()


def cmd_cards(a):
    from . import card_llm, index

    card_llm.run(limit=a.limit, include_summaries=a.summaries)
    index.sync()


def cmd_add(a):
    from . import hf_poll, index

    n = hf_poll.add(a.ids)
    index.sync()
    print(f"Indexed {n} model(s).")


def cmd_add_from_text(a):
    from . import hf_poll, index
    from .sources.x import extract_hf_ids

    text = sys.stdin.read() if a.file == "-" else open(a.file).read()
    ids = extract_hf_ids(text)
    print(f"Found {len(ids)} Hugging Face links: {', '.join(ids[:10])}{' ...' if len(ids) > 10 else ''}")
    if ids:
        n = hf_poll.add(ids, source="x")
        index.sync()
        print(f"Indexed {n}.")


def _ask_server(a) -> dict | None:
    """If `model-finder serve` is running it already has the embedding model loaded — ask it (seconds, not
    the ~15 s a fresh process spends loading the model)."""
    import httpx

    if a.no_llm or a.no_answer:
        return None
    try:
        r = httpx.post(f"http://127.0.0.1:{config.WEB_PORT}/api/ask",
                       json={"question": a.question, "device": a.device, "limit": a.limit}, timeout=180)
        return r.json() if r.status_code == 200 else None
    except httpx.HTTPError:
        return None


def cmd_ask(a):
    from . import query

    _need_qdrant()
    res = _ask_server(a) or query.ask(a.question, device=a.device, limit=a.limit, use_llm=not a.no_llm,
                                      answer=not a.no_answer)
    if a.json:
        print(json.dumps(res, indent=2))
        return
    p = res["plan"]
    looking = ", ".join(p["tasks"]) or "any task"
    dev = p["device"] + (f" (≤{p['ram_cap_gb']:g} GB)" if p["ram_cap_gb"] else "")
    print(f"\nLooking for: {looking} · device: {dev}  [parsed by {p['parser']}]")
    if res["relaxed"]:
        print("Loosened: " + "; ".join(res["relaxed"]))
    if res["answer"]:
        print("\n" + textwrap.fill(res["answer"], 100))
    for i, r in enumerate(res["results"], 1):
        fit = f"  fit {r['fit']:.2f}" if r.get("fit") is not None else ""
        print(f"\n{i}. {r['id']}  [{r['device_tier']}]{fit}  {r['url']}")
        print(textwrap.fill(r["why"], 100, initial_indent="   ", subsequent_indent="   "))
        print(textwrap.fill("Run: " + r["how_to_run"], 100, initial_indent="   ", subsequent_indent="        "))
    t = res["timing_ms"]
    print(f"\n({t['total']} ms: " + ", ".join(f"{k} {v}" for k, v in t.items() if k != "total") + ")")


def cmd_serve(a):
    import httpx
    import uvicorn

    _need_qdrant()
    try:
        if httpx.get(f"http://127.0.0.1:{a.port}/api/stats", timeout=3).status_code == 200:
            print(f"Model Finder is already running at http://localhost:{a.port} — nothing to do.")
            print("(To restart it: pkill -f 'model-finder serve', then run this again.)")
            return
    except httpx.HTTPError:
        pass
    print(f"Open http://localhost:{a.port}")
    uvicorn.run("model_finder.web:app", host="127.0.0.1", port=a.port, log_level="warning")


def cmd_stats(a):
    from . import catalog, index, llm

    with catalog.session() as con:
        print(json.dumps({
            "catalog_models": catalog.count(con),
            "indexed": index.count() if _qdrant_ok() else "qdrant down",
            "last_seed": catalog.kv_get(con, "last_seed"),
            "last_update": catalog.kv_get(con, "last_update"),
            "llm": llm.name(),
        }, indent=2, default=str))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="model-finder", description="Find the right Hugging Face model for a need.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("setup", help="one-time build: snapshot -> catalog -> index")
    s.add_argument("--refresh", action="store_true", help="re-download the snapshot even if today's is cached")
    s.add_argument("--limit", type=int, help="only the N most-downloaded models (quick test)")
    s.add_argument("--sizes", type=int, default=1500, help="fetch weight-file sizes for this many models lacking a param count")
    s.add_argument("--cards", type=int, default=60, help="read this many model cards with the LLM (0 = skip)")
    s.add_argument("--recreate", action="store_true", help="drop and rebuild the Qdrant collection")
    s.set_defaults(fn=cmd_setup)

    s = sub.add_parser("update", help="incremental: new + trending models from the live Hub")
    s.add_argument("--pages", type=int, default=30, help="max pages of 1,000 new models to scan")
    s.add_argument("--cards", type=int, default=20)
    s.set_defaults(fn=cmd_update)

    s = sub.add_parser("reseed", help="full refresh from today's snapshot (re-index only what changed)")
    s.add_argument("--cards", type=int, default=0)
    s.set_defaults(fn=cmd_reseed)

    s = sub.add_parser("index", help="push catalog changes into Qdrant")
    s.add_argument("--recreate", action="store_true")
    s.add_argument("--backfill", type=int, nargs="?", const=0, metavar="N",
                   help="add semantic vectors to keyword-only models (N most popular; all if no N)")
    s.set_defaults(fn=cmd_index)

    s = sub.add_parser("cards", help="LLM-read model cards to fill missing task/size/license")
    s.add_argument("--limit", type=int, default=100)
    s.add_argument("--summaries", action="store_true", help="also do popular models that aren't missing anything")
    s.set_defaults(fn=cmd_cards)

    s = sub.add_parser("sizes", help="estimate size from weight files for models with no param count")
    s.add_argument("--limit", type=int, default=1500)
    s.set_defaults(fn=cmd_sizes)

    s = sub.add_parser("add", help="index specific repos now")
    s.add_argument("ids", nargs="+")
    s.set_defaults(fn=cmd_add)

    s = sub.add_parser("add-from-text", help="index every Hugging Face link found in a text file (X hook)")
    s.add_argument("file")
    s.set_defaults(fn=cmd_add_from_text)

    s = sub.add_parser("joined", help="merge the n8n joined file (HF + Replicate + X) into the index")
    s.add_argument("--fetch", type=int, default=300, help="max HF repos from that file to fetch if missing")
    s.set_defaults(fn=cmd_joined)

    s = sub.add_parser("ask", help="ask a question")
    s.add_argument("question")
    s.add_argument("--device", choices=["auto", "phone", "mac", "laptop", "gpu", "server", "any"], default="auto")
    s.add_argument("--limit", type=int, default=6)
    s.add_argument("--no-llm", action="store_true", help="rule-based understanding only")
    s.add_argument("--no-answer", action="store_true", help="skip the LLM overview paragraph")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("serve", help="local web page")
    s.add_argument("--port", type=int, default=config.WEB_PORT)
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("stats")
    s.set_defaults(fn=cmd_stats)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()

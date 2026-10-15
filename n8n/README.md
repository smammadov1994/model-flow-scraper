# n8n ingestion pipeline

Pulls models from Hugging Face, Replicate and the X sweep, cleans them, joins them into one record
per model, and writes:

- `../data/joined/models_joined.json`: the joined data (see `SCHEMA.md`)
- `../data/joined/run_report.json`: counts per source, joins made, rows dropped, completeness

This stage stops before embedding and RAG. `search_text` in each record is the text to embed.

## Run it

```bash
cd ~/Desktop/Projects/model-finder/n8n
docker compose up -d                  # n8n UI at http://localhost:5679
```

The first time you open the UI, n8n asks you to create a local owner account (it stays on this Mac).
The workflow is called **Model Finder - data ingestion (HF + Replicate + X)**. Press *Execute workflow*,
or turn it on to run every 6 hours.

Run it from the terminal without the UI:

```bash
docker exec -e N8N_RUNNERS_BROKER_PORT=5690 model-finder-n8n n8n execute --id=modelFinderIngest
```

## Replicate

Put your token in `.env` (`REPLICATE_API_TOKEN=r8_...`, from https://replicate.com/account/api-tokens),
then run `docker compose up -d` again so n8n picks it up. Without a token, the Replicate nodes fail
softly and the rest of the pipeline still runs.

## Editing

The JavaScript for the Code nodes lives in `code/`. After changing it:

```bash
python3 build_workflow.py
docker exec model-finder-n8n n8n import:workflow --input=/workflows/workflow.json
docker exec model-finder-n8n n8n publish:workflow --id=modelFinderIngest   # keep the 6-hour schedule on
docker restart model-finder-n8n
```

The workflow is published (active), so it runs every 6 hours while the container is up.
`restart: unless-stopped` brings it back after a Docker or Mac restart.

Re-importing replaces the workflow, so make changes here rather than in the n8n editor
(or export from the editor first).

## Sources

Each source is a Code node that fetches with `this.helpers.httpRequest` and keeps only the fields the
join needs. Plain HTTP Request nodes passed tens of MB per run to the editor and froze it.

| Node | Endpoint | Notes |
|---|---|---|
| HF newest models | `GET huggingface.co/api/models?sort=createdAt&limit=1000&expand[]=…` | no token needed |
| HF trending models | `…?sort=trendingScore&limit=500` | |
| HF quantized variants | `…?filter=base_model:quantized:<id>&sort=downloads&limit=6` | top 100 trending originals |
| Replicate models | `GET api.replicate.com/v1/models?sort_by=model_created_at` | up to 40 pages, 3 tries per page |
| Replicate collections | `GET /v1/collections` then `/v1/collections/{slug}` | gives task labels like `text-to-speech` |
| Read X signals | `/data/x/x_signals.jsonl` | written by the `x-model-watch` scheduled task |

Run history is pruned automatically (last 30 runs / 7 days). `backups/` holds a copy of the
n8n database from before the history was cleared on 2026-10-14.

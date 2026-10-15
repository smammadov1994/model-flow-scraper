"""Generate workflow.json (n8n) from the JS files in ./code.

Run: python3 build_workflow.py   -> writes workflow.json next to this file.

Each source is a Code node that fetches with this.helpers.httpRequest and keeps only the
fields the join needs. Plain HTTP Request nodes would pass the full API responses (tens of MB)
to the editor on every manual run, which freezes it.
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
CODE = HERE / "code"

HF_EXPAND = [
    "author", "baseModels", "cardData", "createdAt", "downloads", "gated", "gguf",
    "inferenceProviderMapping", "lastModified", "library_name", "likes", "pipeline_tag",
    "safetensors", "siblings", "tags", "trendingScore",
]
HF_QS = "&".join(f"expand[]={e}" for e in HF_EXPAND)
SLIM_HF = (CODE / "_slim_hf.js").read_text()
# Keep going when a source fails so one outage never blocks the whole run.
SOFT = {"onError": "continueRegularOutput", "alwaysOutputData": True}


def node(name, ntype, version, pos, params, **extra):
    return {"id": name.lower().replace(" ", "-"), "name": name, "type": ntype, "typeVersion": version,
            "position": pos, "parameters": params, **extra}


def code(name, pos, file, soft=False, **subs):
    js = (CODE / file).read_text().replace("//__SLIM_HF__", SLIM_HF).replace("__HF_QS__", HF_QS)
    for k, v in subs.items():
        js = js.replace(f"__{k}__", v)
    return node(name, "n8n-nodes-base.code", 2, pos, {"jsCode": js}, **(SOFT if soft else {}))


def to_json_file(name, pos, filename):
    return node(name, "n8n-nodes-base.convertToFile", 1.1, pos, {
        "operation": "toJson", "mode": "once",
        "options": {"format": True, "fileName": filename}})


def write_file(name, pos, path):
    return node(name, "n8n-nodes-base.readWriteFile", 1, pos, {
        "operation": "write", "fileName": path, "dataPropertyName": "data", "options": {}})


hf = "https://huggingface.co/api/models"
nodes = [
    node("Run manually", "n8n-nodes-base.manualTrigger", 1, [0, 300], {}),
    node("Every 6 hours", "n8n-nodes-base.scheduleTrigger", 1.2, [0, 500],
         {"rule": {"interval": [{"field": "hours", "hoursInterval": 6}]}}),

    code("HF newest models", [300, 0], "fetch_hf_list.js", soft=True,
         URL=f"{hf}?sort=createdAt&direction=-1&limit=1000&{HF_QS}"),
    code("HF trending models", [300, 200], "fetch_hf_list.js", soft=True,
         URL=f"{hf}?sort=trendingScore&direction=-1&limit=500&{HF_QS}"),
    code("Pick bases for variants", [520, 200], "pick_bases.js", soft=True),
    code("HF quantized variants", [740, 200], "fetch_hf_variants.js", soft=True),
    code("Replicate models", [300, 400], "fetch_replicate_models.js", soft=True),
    code("Replicate collections", [300, 600], "fetch_replicate_collections.js", soft=True),
    node("Read X signals", "n8n-nodes-base.readWriteFile", 1, [300, 800],
         {"fileSelector": "/data/x/x_signals.jsonl", "options": {}}, **SOFT),
    node("Parse X file", "n8n-nodes-base.extractFromFile", 1.1, [520, 800],
         {"operation": "text", "binaryPropertyName": "data", "destinationKey": "data", "options": {}}, **SOFT),

    node("Wait for all sources", "n8n-nodes-base.merge", 3.2, [980, 400],
         {"mode": "append", "numberInputs": 6}),
    code("Clean & join", [1200, 400], "join.js"),

    node("One item per model", "n8n-nodes-base.splitOut", 1, [1420, 300],
         {"fieldToSplitOut": "models", "include": "noOtherFields", "options": {}}),
    to_json_file("Models to JSON", [1640, 300], "models_joined.json"),
    write_file("Save models_joined.json", [1860, 300], "/data/joined/models_joined.json"),

    code("Run report", [1420, 520], "report.js"),
    to_json_file("Report to JSON", [1640, 520], "run_report.json"),
    write_file("Save run_report.json", [1860, 520], "/data/joined/run_report.json"),
]

MERGE = "Wait for all sources"


def link(*targets):
    return {"main": [[{"node": t, "type": "main", "index": i if t == MERGE else 0} for t, i in targets]]}


fan_out = [("HF newest models", 0), ("HF trending models", 0), ("Replicate models", 0),
           ("Replicate collections", 0), ("Read X signals", 0)]
connections = {
    "Run manually": link(*fan_out),
    "Every 6 hours": link(*fan_out),
    "HF newest models": link((MERGE, 0)),
    "HF trending models": link((MERGE, 1), ("Pick bases for variants", 0)),
    "Pick bases for variants": link(("HF quantized variants", 0)),
    "HF quantized variants": link((MERGE, 5)),
    "Replicate models": link((MERGE, 2)),
    "Replicate collections": link((MERGE, 3)),
    "Read X signals": link(("Parse X file", 0)),
    "Parse X file": link((MERGE, 4)),
    MERGE: link(("Clean & join", 0)),
    "Clean & join": link(("One item per model", 0), ("Run report", 0)),
    "One item per model": link(("Models to JSON", 0)),
    "Models to JSON": link(("Save models_joined.json", 0)),
    "Run report": link(("Report to JSON", 0)),
    "Report to JSON": link(("Save run_report.json", 0)),
}

workflow = {
    "id": "modelFinderIngest",
    "name": "Model Finder - data ingestion (HF + Replicate + X)",
    "active": False,
    "nodes": nodes,
    "connections": connections,
    "settings": {"executionOrder": "v1", "saveManualExecutions": True},
    "pinData": {},
}
(HERE / "workflow.json").write_text(json.dumps(workflow, indent=2))
print(f"wrote workflow.json with {len(nodes)} nodes")

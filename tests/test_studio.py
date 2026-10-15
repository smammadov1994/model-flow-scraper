"""Studio pieces that need no network: budget swapping, layout, n8n export, template plans."""
from model_finder.studio import n8n, resolve, templates


def M(id_, ram, fit=0.8, fmt="gguf", variant="Q4_K_M"):
    return {"id": id_, "url": f"https://huggingface.co/{id_}", "ram_gb": ram, "fit": fit, "format": fmt,
            "variant": variant, "variant_repo": id_ + "-GGUF", "how_to_run": "run it", "license": "mit"}


def node(i, typ, task=None, model=None, alts=()):
    return {"id": i, "type": typ, "title": i.upper(), "step": "", "task": task, "inputs": [], "outputs": ["text"],
            "position": None, "model": model, "alternatives": list(alts), "status": "ok", "notes": None}


def test_fit_budget_swaps_biggest_for_smaller_alternative():
    nodes = [node("stt", "model", "automatic-speech-recognition", M("a/stt", 1.0)),
             node("llm", "model", "text-generation", M("a/big", 5.0), [M("a/small", 1.5, 0.7), M("a/bad", 0.5, 0.1)])]
    resolve._fit_budget(nodes, 4.0)
    assert nodes[1]["model"]["id"] == "a/small"  # low-fit alternative is not used
    assert any(a["id"] == "a/big" for a in nodes[1]["alternatives"])
    b = resolve._budget(nodes, 4.0)
    assert b["fits"] and b["total_ram_gb"] == 2.5


def test_over_budget_marked_when_nothing_fits():
    nodes = [node("llm", "model", "text-generation", M("a/big", 9.0))]
    resolve._fit_budget(nodes, 4.0)
    assert nodes[0]["status"] == "over_budget" and not resolve._budget(nodes, 4.0)["fits"]


def test_layout_layers_and_keeps_user_positions():
    nodes = [node("a", "input"), node("b", "model"), node("c", "output"), node("q", "input")]
    edges = [{"source": "a", "target": "b"}, {"source": "b", "target": "c"}, {"source": "q", "target": "b"}]
    resolve._layout(nodes, edges, {"c": {"x": 5, "y": 6}})
    pos = {n["id"]: n["position"] for n in nodes}
    assert pos["a"]["x"] == pos["q"]["x"] == 0 and pos["b"]["x"] == resolve.X_GAP and pos["c"] == {"x": 5, "y": 6}
    assert pos["a"]["y"] != pos["q"]["y"]


def test_n8n_export_shape():
    p = {"id": "s_1", "title": "Voice", "goal": "g", "device": {}, "budget": {},
         "nodes": [dict(node("mic", "input"), position={"x": 0, "y": 0}),
                   dict(node("llm", "model", "text-generation", M("Qwen/Qwen3-4B", 3.0)), position={"x": 380, "y": 0}),
                   dict(node("tts", "model", "text-to-speech", M("hexgrad/Kokoro-82M", 0.1, fmt="onnx", variant="q4")),
                        position={"x": 760, "y": 0})],
         "edges": [{"id": "e1", "source": "mic", "target": "llm", "label": "text"},
                   {"id": "e2", "source": "llm", "target": "tts", "label": "text"}]}
    wf = n8n.export(p)
    names = [x["name"] for x in wf["nodes"]]
    assert len(names) == len(set(names))  # n8n requires unique names
    types = {x["name"]: x["type"] for x in wf["nodes"]}
    llm = next(n for n in names if n.startswith("LLM"))
    assert types[llm] == "n8n-nodes-base.httpRequest"  # GGUF chat model -> local Ollama call
    body = next(x for x in wf["nodes"] if x["name"] == llm)["parameters"]["jsonBody"]
    assert body.startswith("={{") and '"hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M"' in body  # one expression, evaluated by n8n
    trig = names[0]
    assert wf["connections"][trig]["main"][0][0]["node"] == "MIC"
    assert wf["connections"][llm]["main"][0][0]["node"].startswith("TTS")
    assert all(isinstance(x["position"], list) and len(x["position"]) == 2 for x in wf["nodes"])
    assert sum(x["type"].endswith("stickyNote") for x in wf["nodes"]) == 2


def test_templates_are_valid_graphs():
    for key, (_, _, build) in templates.TEMPLATES.items():
        steps, edges = build()
        keys = {s["key"] for s in steps}
        assert len(keys) == len(steps), key
        assert all(e["from"] in keys and e["to"] in keys for e in edges), key
        assert all(s["task"] for s in steps if s["type"] == "model"), key


def test_n8n_validate_rejects_broken_files():
    import pytest

    good = {"nodes": [{"id": "1", "name": "A", "type": n8n.CODE[0], "typeVersion": 2, "position": [0, 0], "parameters": {}}],
            "connections": {"A": {"main": [[{"node": "Missing", "type": "main", "index": 0}]]}}}
    with pytest.raises(ValueError, match="unknown node"):
        n8n.validate(good)
    good["connections"] = {}
    good["nodes"][0]["position"] = {"x": 0, "y": 0}
    with pytest.raises(ValueError, match=r"\[x, y\]"):
        n8n.validate(good)

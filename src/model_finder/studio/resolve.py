"""Plan -> Pipeline: pick a model for every model step (Model Finder search + Jev fit), enforce the device's
memory budget across all steps, and lay the nodes out left-to-right. Shape documented in docs/STUDIO.md."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from .. import catalog, jev, query
from ..summarize import TASK_PLAIN

ALTERNATIVES = 4
X_GAP, Y_GAP = 380, 190


def _model_obj(res: dict) -> dict:
    return {
        "id": res["id"], "url": res["url"], "params_h": res.get("params_h"), "ram_gb": res.get("ram_gb_min"),
        "variant": res.get("best_variant"), "variant_repo": res.get("best_variant_repo"),
        "format": res.get("best_variant_format"), "license": res.get("license"),
        "open_license": res.get("open_license"), "downloads": res.get("downloads"),
        "how_to_run": res.get("how_to_run"), "why": res.get("why"), "fit": res.get("fit"),
    }


def _step_plan(step: dict, device: dict) -> query.Plan:
    hint = step.get("search_hint") or step.get("step") or step["title"]
    c = step.get("constraints") or {}
    p = query.Plan(question=hint, semantic_query=hint, tasks=[step["task"]] if step.get("task") else [],
                   device=device.get("kind") or "any", memory_gb=device.get("memory_gb"),
                   license_open=c.get("license_open") or None, max_params_b=c.get("max_params_b"),
                   languages=[l.lower()[:3] for l in c.get("languages") or []], parser="studio")
    return query.apply_device(p)


QUALITY_TASKS = {"text-generation", "image-text-to-text", "text-to-image", "text-to-video", "image-to-video",
                 "any-to-any", "audio-text-to-text"}  # bigger (within budget) is noticeably better


def _candidates(step: dict, device: dict, goal: str, share_gb: float | None = None) -> tuple[list[dict], list[str]]:
    """Ranked candidate models (contract Model objects) for one step. `share_gb` = this step's fair share of
    the device memory; for quality-sensitive tasks the biggest model within it gets a boost."""
    p = _step_plan(step, device)
    ranked, relaxed = query.search_ranked(p, query.search_text(p))
    pool = ranked[:20]
    with catalog.session() as con:
        roots = catalog.get_many(con, list({r.get("family_root") or r["id"] for _, r in pool}))
        pin = (step.get("constraints") or {}).get("pin_model")
        pinned = catalog.get(con, pin) if pin else None
    pool = [(s, roots.get(r.get("family_root") or r["id"]) or r) for s, r in pool]
    fit: dict[str, float] = {}
    if jev.available() and pool:
        try:
            fit = jev.fit_scores(f"{goal}\nThis step: {step.get('step') or step['title']}", query.plan_text(p),
                                 [r for _, r in pool])
            pool = sorted(((s + 0.35 * fit.get(r["id"], 0.5), r) for s, r in pool), key=lambda x: -x[0])
        except Exception as e:  # noqa: BLE001
            print(f"  (Jev fit failed for step {step['key']}: {e!r})")
    prefer = (step.get("constraints") or {}).get("prefer")
    if share_gb and step.get("task") in QUALITY_TASKS and prefer not in ("smallest", "fastest"):
        def use_budget(item):
            s, r = item
            ram = r.get("family_ram_gb_min") or 0
            if not 0 < ram <= share_gb:
                return s
            # use the memory you have: a 9B model beats a 0.6B one when both fit; tiny ones get a nudge down
            return s + 0.4 * ram / share_gb - (0.15 if ram < 0.1 * share_gb else 0)
        pool = sorted(pool[:12], key=lambda x: -use_budget(x)) + pool[12:]
    if prefer in ("smallest", "fastest"):
        head = sorted(pool[:8], key=lambda x: (x[1].get("family_ram_gb_min") or 1e9))
        pool = head + pool[8:]
    elif prefer == "most_popular":
        head = sorted(pool[:8], key=lambda x: -(x[1].get("downloads") or 0))
        pool = head + pool[8:]
    out = []
    if pinned:
        out.append(_model_obj({**query.to_result(pinned, 1.0, p), "fit": None}))
    for s, rec in pool:
        if pinned and rec["id"] == pinned["id"]:
            continue
        res = query.to_result(rec, s, p)
        res["fit"] = round(fit[rec["id"]], 3) if rec["id"] in fit else None
        out.append(_model_obj(res))
    return out[: 1 + ALTERNATIVES], relaxed


def _layout(nodes: list[dict], edges: list[dict], old_pos: dict[str, dict]) -> None:
    """Longest-path layering, left to right; keeps positions the user already arranged."""
    preds: dict[str, list[str]] = {n["id"]: [] for n in nodes}
    for e in edges:
        if e["target"] in preds and e["source"] in preds:
            preds[e["target"]].append(e["source"])
    layer: dict[str, int] = {}

    def depth(nid: str, seen: frozenset = frozenset()) -> int:
        if nid in layer:
            return layer[nid]
        if nid in seen:  # cycle guard
            return 0
        d = 0 if not preds[nid] else 1 + max(depth(x, seen | {nid}) for x in preds[nid])
        layer[nid] = d
        return d

    rows: dict[int, int] = {}
    counts: dict[int, int] = {}
    for n in nodes:
        counts[depth(n["id"])] = counts.get(depth(n["id"]), 0) + 1
    for n in nodes:
        L = layer[n["id"]]
        i = rows.get(L, 0)
        rows[L] = i + 1
        default = {"x": L * X_GAP, "y": (i - (counts[L] - 1) / 2) * Y_GAP + 200}
        n["position"] = old_pos.get(n["id"]) or default


def _budget(nodes: list[dict], cap: float | None) -> dict:
    models = [n for n in nodes if n.get("model")]
    total = round(sum((n["model"].get("ram_gb") or 0) for n in models), 2)
    unknown = [n["title"] for n in models if not n["model"].get("ram_gb")]
    fits = cap is None or total <= cap
    if cap is None:
        note = f"All models together need about {total:g} GB of memory."
    elif fits:
        note = f"All {len(models)} models loaded at once need about {total:g} GB of the ~{cap:g} GB budget."
    else:
        note = f"Loaded together the models need about {total:g} GB — more than the ~{cap:g} GB budget. Run them one at a time or pick smaller ones."
    if unknown:
        note += " Unknown size: " + ", ".join(unknown) + "."
    return {"ram_cap_gb": cap, "total_ram_gb": total, "fits": fits, "note": note}


def _fit_budget(nodes: list[dict], cap: float | None) -> None:
    """If the chosen models don't fit together, swap the biggest for a smaller decent alternative, repeatedly."""
    if cap is None:
        return
    for _ in range(12):
        models = [n for n in nodes if n.get("model") and n["model"].get("ram_gb")]
        if sum(n["model"]["ram_gb"] for n in models) <= cap:
            return
        swapped = False
        for n in sorted(models, key=lambda n: -n["model"]["ram_gb"]):
            cur = n["model"]["ram_gb"]
            smaller = [a for a in n["alternatives"] if a.get("ram_gb") and a["ram_gb"] < cur * 0.8
                       and (a.get("fit") is None or a["fit"] >= 0.4)]
            if smaller:
                best = max(smaller, key=lambda a: (a.get("fit") or 0.5) - a["ram_gb"] / (4 * cap))
                n["alternatives"] = [a for a in n["alternatives"] if a is not best] + [n["model"]]
                n["model"] = best
                n["notes"] = f"Swapped to a smaller model so everything fits in ~{cap:g} GB."
                swapped = True
                break
        if not swapped:
            break
    total = sum((n["model"].get("ram_gb") or 0) for n in nodes if n.get("model"))
    if total > cap:
        for n in nodes:
            if n.get("model") and (n["model"].get("ram_gb") or 0) > cap / 2:
                n["status"] = "over_budget"


def build(plan: dict, goal: str, previous: dict | None = None) -> dict:
    t0 = time.time()
    device = dict(plan.get("device") or {"kind": "any", "label": "any", "memory_gb": None})
    dev_plan = query.apply_device(query.Plan(question="", semantic_query="", device=device.get("kind") or "any",
                                             memory_gb=device.get("memory_gb")))
    cap = dev_plan.ram_cap_gb
    prev_nodes = {n["id"]: n for n in (previous or {}).get("nodes", [])}
    nodes = []
    for s in plan["steps"]:
        nodes.append({"id": s["key"], "type": s["type"], "title": s["title"], "step": s.get("step"),
                      "task": s.get("task") if s["type"] == "model" else None,
                      "inputs": s.get("inputs") or [], "outputs": s.get("outputs") or [],
                      "position": None, "model": None, "alternatives": [], "status": "ok", "notes": None})
    model_steps = [s for s in plan["steps"] if s["type"] == "model"]
    # fair memory share per step: quality-sensitive steps (LLM, image/video gen) get 3x the weight of others
    weights = {s["key"]: (3 if s.get("task") in QUALITY_TASKS else 1) for s in model_steps}
    share = {k: (cap * w / sum(weights.values())) * 0.9 for k, w in weights.items()} if cap and weights else {}

    def work(step):
        prev = prev_nodes.get(step["key"])
        # unchanged step with a model already chosen (maybe swapped by the user): keep it
        if (prev and prev.get("model") and prev.get("task") == step.get("task")
                and not (step.get("constraints") or {}).get("pin_model")
                and (previous or {}).get("device") == plan.get("device")):
            return step["key"], [prev["model"]] + prev.get("alternatives", []), []
        return (step["key"], *_candidates(step, device, goal, share.get(step["key"])))

    with ThreadPoolExecutor(max_workers=4) as ex:
        results = {k: (c, r) for k, c, r in ex.map(work, model_steps)}
    for n in nodes:
        if n["id"] in results:
            cands, relaxed = results[n["id"]]
            if cands:
                n["model"], n["alternatives"] = cands[0], cands[1:]
                if relaxed:
                    n["notes"] = "Few exact matches; loosened: " + "; ".join(relaxed)
            else:
                n["status"] = "no_match"
                n["notes"] = f"No {TASK_PLAIN.get(n['task'] or '', (n['task'] or 'model',))[0].lower()} model fits this device."
    _fit_budget(nodes, cap)
    keys = {n["id"] for n in nodes}
    edges = [{"id": f"e{i + 1}", "source": e["from"], "target": e["to"], "label": e.get("label") or ""}
             for i, e in enumerate(plan.get("edges") or []) if e["from"] in keys and e["to"] in keys]
    _layout(nodes, edges, {k: v.get("position") for k, v in prev_nodes.items() if v.get("position")})
    device["ram_cap_gb"] = cap
    return {
        "id": (previous or {}).get("id"), "title": plan.get("title") or "Untitled pipeline", "goal": goal,
        "device": device, "nodes": nodes, "edges": edges, "budget": _budget(nodes, cap),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "resolve_ms": int((time.time() - t0) * 1000),
    }


def swap(pipeline: dict, node_id: str, model_id: str) -> dict:
    for n in pipeline["nodes"]:
        if n["id"] == node_id and n.get("model"):
            alt = next((a for a in n["alternatives"] if a["id"] == model_id), None)
            if alt is None:
                raise KeyError(f"{model_id} is not an alternative for {node_id}")
            n["alternatives"] = [a for a in n["alternatives"] if a["id"] != model_id] + [n["model"]]
            n["model"], n["status"], n["notes"] = alt, "ok", "Chosen by you."
            break
    else:
        raise KeyError(f"no model node {node_id}")
    pipeline["budget"] = _budget(pipeline["nodes"], pipeline["device"].get("ram_cap_gb"))
    if not pipeline["budget"]["fits"]:
        for n in pipeline["nodes"]:
            if n["id"] == node_id:
                n["status"] = "over_budget"
    return pipeline

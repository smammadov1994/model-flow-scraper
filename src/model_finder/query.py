"""Question -> plan (LLM self-query + rule backup) -> filtered hybrid search -> relax -> rank -> explain."""
from __future__ import annotations

import math
import re
import time
from dataclasses import asdict, dataclass, field

from qdrant_client import models

from . import catalog, config, enrich, howto, index, llm
from .summarize import TASK_PLAIN, human_params, task_label

DEVICES = ["phone", "mac", "laptop", "gpu", "server", "any"]
PHONE_FORMATS = ["gguf", "ggml", "onnx", "coreml", "executorch", "tflite", "mlx", "mlc"]
# Formats that make a model nicer on a device (ranking boost, not a filter except for phone).
PREFERRED_FORMATS = {
    "phone": {"onnx", "coreml", "executorch", "tflite", "gguf", "ggml", "mlx", "mlc"},
    "mac": {"mlx", "gguf", "coreml", "ggml"},
    "laptop": {"gguf", "onnx", "openvino", "ggml"},
    "gpu": {"safetensors", "diffusers", "gguf", "awq", "gptq", "exl2"},
}
DEFAULT_CAP_GB = {"phone": 4.0, "mac": 12.0, "laptop": 12.0, "gpu": 24.0}
MIN_RESULTS = 5

# ---------------------------------------------------------------- plan


@dataclass
class Plan:
    question: str
    semantic_query: str
    tasks: list[str] = field(default_factory=list)
    device: str = "any"
    memory_gb: float | None = None  # memory the user said they have (RAM, or VRAM for gpu)
    ram_cap_gb: float | None = None  # what we filter on (estimated RAM need <= this)
    formats: list[str] = field(default_factory=list)  # hard format requirement (phone only)
    license_open: bool | None = None
    max_params_b: float | None = None
    min_params_b: float | None = None
    created_after_days: int | None = None
    languages: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    parser: str = "rules"


FIELDS = {  # declared to the LLM: name -> (json schema, description)
    "semantic_query": ({"type": "string"}, "a short description of the ideal model in plain words (what it does, key capabilities); used for semantic search"),
    "tasks": ({"type": "array", "items": {"type": "string", "enum": sorted(TASK_PLAIN)}},
              "1-3 Hugging Face task ids the model must perform. 'voice model'/'talk'/'read aloud'/'clone my voice' = text-to-speech (add audio-to-audio for voice conversion); 'transcribe'/'dictation' = automatic-speech-recognition; chat/coding/writing LLM = text-generation; 'see images and chat' = image-text-to-text; embeddings/RAG = feature-extraction + sentence-similarity"),
    "device": ({"type": "string", "enum": DEVICES}, "where it must run: phone (iPhone/Android/mobile), mac (MacBook/Mac/Apple Silicon), laptop (non-Mac laptop or CPU PC), gpu (desktop/NVIDIA GPU, VRAM mentioned), server, any (not stated)"),
    "memory_gb": ({"type": ["number", "null"]}, "memory the user says they have in GB (e.g. 'under 8GB VRAM' -> 8, '16GB MacBook' -> 16), else null"),
    "license_open": ({"type": ["boolean", "null"]}, "true only if the user needs commercial use / open source / permissive license, else null"),
    "max_params_b": ({"type": ["number", "null"]}, "upper bound on parameters in billions if the user states one ('under 3B' -> 3; 'small' alone -> null), else null"),
    "min_params_b": ({"type": ["number", "null"]}, "lower bound on parameters in billions if stated, else null"),
    "created_after_days": ({"type": ["integer", "null"]}, "only models released in the last N days if the user asks for new/latest/recent (latest -> 180, this month -> 31), else null"),
    "languages": ({"type": "array", "items": {"type": "string"}}, "ISO 639-1 codes of languages the model must support if stated (e.g. 'Spanish' -> 'es'), else []"),
    "keywords": ({"type": "array", "items": {"type": "string"}}, "specific must-have words like model family names or features (e.g. 'voice cloning', 'whisper', 'qwen'), else []"),
}

SYSTEM = (
    "You turn a person's request for an AI model into search filters for a Hugging Face model index. "
    "Fill every field; use null/[] when the request does not say. Do not invent constraints.\nFields:\n"
    + "\n".join(f"- {k}: {d}" for k, (_, d) in FIELDS.items())
)
SCHEMA = {"type": "object", "properties": {k: s for k, (s, _) in FIELDS.items()}, "required": list(FIELDS)}

# Rule-based parser: always runs; fills gaps the LLM leaves, and is the whole parser when no LLM exists.
TASK_RULES = [
    (r"\b(voice clon\w*|clone (my|a) voice)\b", ["text-to-speech", "audio-to-audio"]),
    (r"\b(tts|text[- ]to[- ]speech|voice model|voice|speak\w*|read (it )?aloud|narrat\w*|talking)\b", ["text-to-speech"]),
    (r"\b(transcri\w*|speech[- ]to[- ]text|stt|asr|dictat\w*|subtitle\w*|speech recognition)\b", ["automatic-speech-recognition"]),
    (r"\b(music|song|sound effects?)\b", ["text-to-audio"]),
    (r"\b(image generation|generate images?|text[- ]to[- ]image|ai art|diffusion|draw\w*|pictures?)\b", ["text-to-image"]),
    (r"\b(video generation|generate videos?|text[- ]to[- ]video)\b", ["text-to-video"]),
    (r"\b(embedding\w*|semantic search|rag|retrieval)\b", ["feature-extraction", "sentence-similarity"]),
    (r"\b(rerank\w*)\b", ["text-ranking"]),
    (r"\b(ocr|read documents?|screenshots?|vision language|vlm|describe images?|look at (images?|photos?))\b", ["image-text-to-text"]),
    (r"\b(object detection|detect objects?)\b", ["object-detection"]),
    (r"\b(translat\w*)\b", ["translation", "text-generation"]),
    (r"\b(cod(e|ing|er)|programming|chat\w*|llm|assistant|reasoning|writ(e|ing))\b", ["text-generation"]),
]
DEVICE_RULES = [
    (r"\b(phone|iphone|android|mobile|on[- ]device|smartphone|pixel)\b", "phone"),
    (r"\b(mac ?book|mac|macos|apple silicon|m[1-5]( pro| max| ultra)?)\b", "mac"),
    (r"\b(gpu|vram|rtx|nvidia|cuda|graphics card|[34]0[5-9]0)\b", "gpu"),
    (r"\b(laptop|notebook|cpu only|pc)\b", "laptop"),
    (r"\b(server|cluster|h100|a100)\b", "server"),
]
LANG_WORDS = {"english": "en", "spanish": "es", "french": "fr", "german": "de", "chinese": "zh", "japanese": "ja",
              "korean": "ko", "russian": "ru", "arabic": "ar", "hindi": "hi", "portuguese": "pt", "italian": "it",
              "turkish": "tr", "azerbaijani": "az", "vietnamese": "vi", "dutch": "nl", "polish": "pl", "persian": "fa"}


def parse_rules(q: str) -> dict:
    s = q.lower()
    out: dict = {"tasks": [], "device": "any", "memory_gb": None, "max_params_b": None, "license_open": None,
                 "created_after_days": None, "languages": []}
    for pat, tasks in TASK_RULES:
        if re.search(pat, s):
            out["tasks"] = tasks
            break
    if out["tasks"] == ["text-generation"] and re.search(r"\b(cod(e|ing|er)|programming)\b", s):
        out["keywords"] = ["code"]
    for pat, dev in DEVICE_RULES:
        if re.search(pat, s):
            out["device"] = dev
            break
    m = re.search(r"(\d+(?:\.\d+)?)\s*gb", s)
    if m:
        out["memory_gb"] = float(m.group(1))
    m = re.search(r"(under|below|less than|max|<)\s*(\d+(?:\.\d+)?)\s*b\b", s)
    if m:
        out["max_params_b"] = float(m.group(2))
    if re.search(r"\b(commercial|open[- ]source|permissive|apache|mit license)\b", s):
        out["license_open"] = True
    if re.search(r"\b(latest|newest|new|recent|this year)\b", s):
        out["created_after_days"] = 180
    out["languages"] = [code for w, code in LANG_WORDS.items() if w in s]
    return out


def make_plan(question: str, device: str | None = None, use_llm: bool = True) -> Plan:
    rules = parse_rules(question)
    parsed, parser = {}, "rules"
    if use_llm and llm.available():
        try:
            parsed = llm.complete_json(SYSTEM, f"Request: {question}", SCHEMA, max_tokens=400, timeout=45)
            parser = llm.name()
        except Exception as e:  # LLM hiccup -> rules only
            print(f"  (LLM parse failed, using rules: {e!r})")
    p = Plan(question=question, semantic_query=(parsed.get("semantic_query") or question).strip(), parser=parser)
    tasks = [t for t in (parsed.get("tasks") or []) if t in TASK_PLAIN]
    # rules win when they are specific and the LLM disagrees completely (small local models drift)
    if rules["tasks"] and not set(tasks) & set(rules["tasks"]):
        tasks = rules["tasks"] + tasks[:1]
    p.tasks = list(dict.fromkeys(tasks or rules["tasks"]))[:3]
    dev = device if device and device != "auto" else None
    llm_dev = parsed.get("device") if parsed.get("device") in DEVICES else "any"
    p.device = dev or (rules["device"] if rules["device"] != "any" else llm_dev)
    # only believe a memory figure the user actually typed (small local LLMs like to invent "16 GB")
    p.memory_gb = rules["memory_gb"] or (_num(parsed.get("memory_gb")) if re.search(r"\d", question) else None)
    p.max_params_b = _num(parsed.get("max_params_b")) or rules["max_params_b"]
    p.min_params_b = _num(parsed.get("min_params_b"))
    p.license_open = parsed.get("license_open") if isinstance(parsed.get("license_open"), bool) else rules["license_open"]
    if p.license_open is False:
        p.license_open = None
    cad = parsed.get("created_after_days")
    p.created_after_days = int(cad) if isinstance(cad, (int, float)) and cad > 0 else rules["created_after_days"]
    p.languages = [l.lower()[:3] for l in (parsed.get("languages") or rules["languages"]) if isinstance(l, str)][:4]
    p.keywords = [k for k in (parsed.get("keywords") or []) if isinstance(k, str)][:5] or rules.get("keywords", [])
    # device -> memory cap + hard formats (deterministic, never left to the LLM)
    if p.device in DEFAULT_CAP_GB:
        if p.memory_gb:
            share = 1.0 if p.device == "gpu" else (0.5 if p.device == "phone" else 0.7)
            p.ram_cap_gb = round(p.memory_gb * share, 1)
        else:
            p.ram_cap_gb = DEFAULT_CAP_GB[p.device]
    elif p.memory_gb:
        p.ram_cap_gb = p.memory_gb
    if p.device == "phone":
        p.formats = PHONE_FORMATS
    return p


def _num(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and v > 0 else None


# ---------------------------------------------------------------- filters + relaxation

LADDER = [
    ("all filters", {}),
    ("dropped release-date and language filters", {"date": False, "lang": False}),
    ("dropped license filter", {"date": False, "lang": False, "license": False}),
    ("allowed models of unknown size and up to 1.5x the memory budget",
     {"date": False, "lang": False, "license": False, "size": "loose"}),
    ("searched across related tasks", {"date": False, "lang": False, "license": False, "size": "loose", "task": "modality"}),
    ("dropped hardware filters", {"date": False, "lang": False, "license": False, "size": False, "task": "modality"}),
]


def build_filter(p: Plan, opts: dict) -> models.Filter:
    must: list = []
    must_not: list = [models.FieldCondition(key="is_adapter", match=models.MatchValue(value=True))]
    task_mode = opts.get("task", True)
    if p.tasks and task_mode is True:
        must.append(models.FieldCondition(key="task", match=models.MatchAny(any=p.tasks)))
    elif p.tasks and task_mode == "modality":
        mods = sorted({enrich.TASK_GROUPS.get(t, "other") for t in p.tasks} | ({"code"} if "text-generation" in p.tasks else set()))
        must.append(models.Filter(should=[
            models.FieldCondition(key="modality", match=models.MatchAny(any=mods)),
            models.FieldCondition(key="task", match=models.MatchAny(any=p.tasks)),
        ]))
    size = opts.get("size", True)
    if size and p.ram_cap_gb:
        if size is True and p.device == "phone" and not p.memory_gb:
            must.append(models.FieldCondition(key="family_device_tier", match=models.MatchValue(value="phone")))
        else:
            cap = p.ram_cap_gb * (1.5 if size == "loose" else 1.0)
            fits = models.FieldCondition(key="family_ram_gb_min", range=models.Range(lte=cap))
            if size == "loose":
                must.append(models.Filter(should=[fits, models.IsEmptyCondition(is_empty=models.PayloadField(key="family_ram_gb_min"))]))
            else:
                must.append(fits)
    if size and p.formats:
        must.append(models.FieldCondition(key="family_formats", match=models.MatchAny(any=p.formats)))
    if size and p.max_params_b:
        must.append(models.FieldCondition(key="params", range=models.Range(lte=int(p.max_params_b * 1e9 * 1.05))))
    if size and p.min_params_b:
        must.append(models.FieldCondition(key="params", range=models.Range(gte=int(p.min_params_b * 1e9 * 0.95))))
    if p.license_open and opts.get("license", True):
        must.append(models.FieldCondition(key="open_license", match=models.MatchValue(value=True)))
    if p.created_after_days and opts.get("date", True):
        must.append(models.FieldCondition(key="created_ts", range=models.Range(gte=int(time.time() - p.created_after_days * 86400))))
    if p.languages and opts.get("lang", True):
        must.append(models.FieldCondition(key="languages", match=models.MatchAny(any=p.languages)))
    return models.Filter(must=must, must_not=must_not)


def hybrid_search(text: str, flt: models.Filter, limit: int = 150) -> list[models.ScoredPoint]:
    dense = index.embed_query(text)
    return index.client().query_points(
        config.COLLECTION,
        prefetch=[
            models.Prefetch(query=dense, using="dense", filter=flt, limit=limit),
            models.Prefetch(query=models.Document(text=text, model=config.BM25_MODEL), using="bm25", filter=flt, limit=limit),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=models.PayloadSelectorExclude(exclude=["summary", "tags_clean", "datasets"]),
    ).points


# ---------------------------------------------------------------- ranking


def _score(pt: models.ScoredPoint, top: float, p: Plan) -> float:
    r = pt.payload
    rel = pt.score / top if top else 0.0
    pop = min(1.0, (math.log10((r.get("downloads") or 0) + 1) + 2 * math.log10((r.get("likes") or 0) + 1)) / 14)
    age_days = max(0.0, (time.time() - (r.get("created_ts") or 0)) / 86400)
    rec = math.exp(-age_days / 365)
    fmt = 1.0 if set(r.get("family_formats") or []) & PREFERRED_FORMATS.get(p.device, set()) else 0.0
    task = 1.0 if r.get("task") in p.tasks else 0.0
    kw = 1.0 if any(k.lower() in (r.get("id") or "").lower() for k in p.keywords if len(k) > 2) else 0.0
    penalty = 0.0
    rid = (r.get("id") or "").lower()
    if re.search(r"(^|[-_/])(tests?|testing|tiny-random|random|dummy|debug|toy)([-_/]|$)", rid):
        penalty += 0.3  # CI fixtures and toy checkpoints, not real models
    if r.get("task") in enrich.LLM_TASKS and (r.get("params") or 1e10) < 3e8:
        penalty += 0.15  # sub-300M "chat" models are rarely what anyone wants
    wants_code = any(k.lower().startswith(("code", "coding", "program")) for k in p.keywords) or bool(
        re.search(r"\b(cod(e|ing|er)|programming)\b", p.question.lower()))
    code = 0.0
    if wants_code:
        if re.search(r"cod(e|er|ing)|starcoder|codestral|devstral|codellama|codegemma", rid.split("/")[-1]):
            code = 1.0  # a code-specialised model
        elif r.get("modality") == "code":
            code = 0.3  # general model that lists coding among its skills
    buzz = min(1.0, math.log10((r.get("x_buzz") or 0) + 1) / 4)  # recent X attention (n8n joined file)
    return 0.50 * rel + 0.25 * pop + 0.12 * rec + 0.05 * fmt + 0.05 * task + 0.03 * kw + 0.03 * buzz + 0.12 * code - penalty


def collapse(points: list[models.ScoredPoint], p: Plan) -> list[tuple[float, dict]]:
    """One entry per model family; the family's score is its best member's score."""
    if not points:
        return []
    top = max(pt.score for pt in points)
    best: dict[str, tuple[float, dict]] = {}
    for pt in points:
        s = _score(pt, top, p)
        root = pt.payload.get("family_root") or pt.payload["id"]
        if root not in best or s > best[root][0]:
            best[root] = (s, pt.payload)
    return sorted(best.values(), key=lambda x: -x[0])


# ---------------------------------------------------------------- explanation

VARIANT_PREF = ["Q4_K_M", "Q4_K_XL", "Q4_K_S", "IQ4_XS", "IQ4_NL", "Q4_0", "MXFP4", "q4", "q4f16", "4bit", "4.5bit",
                "Q5_K_M", "Q5_K_S", "5bit", "6bit", "Q6_K", "int8", "Q8_0", "8bit", "fp16", "F16", "BF16", "native", "fp32", "F32"]


def pick_variant(root: dict, device: str, cap: float | None) -> tuple[str | None, str | None, float | None, str | None]:
    """Best (variant, repo, ram_gb, format) across the model and its converted copies that fits the device."""
    members = [root] + list(root.get("derivatives") or [])
    cands = []
    for m in members:
        vf = enrich.variant_formats(m)
        for v, ram in (m.get("ram_gb") or {}).items():
            fmt = vf.get(v, "native")
            if not enrich.practical(v):
                continue
            if device == "phone" and fmt not in PHONE_FORMATS and not (fmt == "native" and set(m.get("formats") or []) & set(PHONE_FORMATS)):
                continue
            fits = cap is None or ram <= cap
            pref = VARIANT_PREF.index(v) if v in VARIANT_PREF else len(VARIANT_PREF)
            dev_bonus = 0 if fmt in PREFERRED_FORMATS.get(device, set()) else 1
            cands.append((not fits, dev_bonus, pref, ram, v, m["id"], fmt))
    if not cands:
        return None, None, None, None
    cands.sort()
    _, _, _, ram, v, repo, fmt = cands[0]
    return v, repo, ram, fmt


def why(rec: dict, p: Plan, variant: str | None, ram: float | None, repo: str | None) -> str:
    bits = [task_label(rec.get("task"))]
    ph = human_params(rec.get("params"))
    if ph:
        bits.append(f"{ph} parameters")
    if ram:
        where = f"~{ram:g} GB of memory" + (f" as {variant}" if variant and variant != "native" else "")
        if p.ram_cap_gb:
            verdict = "fits" if ram <= p.ram_cap_gb else "is a tight fit for"
            dev = {"phone": "a phone", "mac": "your Mac", "laptop": "a laptop", "gpu": "your GPU"}.get(p.device, "your device")
            where += f" — {verdict} {dev} (budget ~{p.ram_cap_gb:g} GB)"
        bits.append(where)
    else:
        bits.append("memory need unknown — check the card")
    s = ", ".join(bits) + "."
    if repo and repo != rec["id"]:
        s += f" Ready-made build: {repo}."
    lic = rec.get("license")
    if lic:
        s += f" License {lic}" + (" (commercial use OK)." if rec.get("open_license") else ".")
    dl, likes = rec.get("downloads") or 0, rec.get("likes") or 0
    if rec.get("status") in ("announced", "seen_on_x", "api_only"):
        s += {"announced": " Announced on X — weights not on Hugging Face yet.",
              "seen_on_x": " Only seen on X so far — no downloadable weights found.",
              "api_only": " Available through the Replicate API (no open weights found)."}[rec["status"]]
    else:
        s += f" {_fmt_n(dl)} downloads last month, {_fmt_n(likes)} likes"
        if rec.get("created_at"):
            s += f", released {rec['created_at'][:10]}"
        s += "."
    if rec.get("x_takes"):
        s += f" On X: {rec['x_takes'][0]}"
    return s.rstrip(".") + "."


def _fmt_n(n: int) -> str:
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}K" if n >= 1e3 else str(n)


def to_result(rec: dict, score: float, p: Plan) -> dict:
    variant, repo, ram, fmt = pick_variant(rec, p.device, p.ram_cap_gb)
    standalone = ":" in rec["id"]  # x:... / replicate:... entities from the n8n joined file
    if rec.get("status") == "api_only" or (standalone and rec.get("replicate_url")):
        hint = f"Call it through the Replicate API: {rec.get('replicate_url') or rec.get('url')}"
    elif standalone:
        hint = "Not downloadable yet — follow the X post for the release: " + (rec.get("url") or "")
    else:
        try:
            hint = None
            if fmt and fmt != "native" and p.device != "any":
                hint = howto.hint_for(rec, fmt, variant, p.device, repo)
            hint = hint or howto.run_hint(rec, p.device if p.device != "any" else "any")
        except Exception as e:  # never let a hint bug kill an answer
            hint = f"See the model card ({e.__class__.__name__})."
    return {
        "id": rec["id"], "url": rec.get("url") if standalone else f"https://huggingface.co/{rec['id']}",
        "status": rec.get("status") or "released", "x_takes": rec.get("x_takes") or [], "x_posts": rec.get("x_posts") or [],
        "task": rec.get("task"), "modality": rec.get("modality"),
        "params": rec.get("params"), "params_h": human_params(rec.get("params")),
        "ram_gb_min": ram if ram is not None else rec.get("family_ram_gb_min"),
        "device_tier": rec.get("family_device_tier") or rec.get("device_tier"),
        "best_variant": variant, "best_variant_repo": repo, "best_variant_format": fmt,
        "formats": rec.get("formats") or [], "family_formats": rec.get("family_formats") or rec.get("formats") or [],
        "license": rec.get("license"), "open_license": bool(rec.get("open_license")),
        "downloads": rec.get("downloads") or 0, "likes": rec.get("likes") or 0, "created_at": rec.get("created_at"),
        "why": why(rec, p, variant, ram, repo), "how_to_run": hint,
        "derivatives": [{"id": d["id"], "formats": d.get("formats") or [], "downloads": d.get("downloads") or 0}
                        for d in (rec.get("derivatives") or [])[:6]],
        "score": round(score, 4),
    }


ANSWER_SYSTEM = ("You help a non-expert pick an AI model. Using ONLY the candidate facts given, write 2-4 short plain "
                 "sentences: which one to try first and why, and one alternative. No markdown, no lists, no jargon.")


def overview(question: str, results: list[dict]) -> str | None:
    if not results or not llm.available():
        return None
    facts = "\n".join(f"{i + 1}. {r['id']}: {r['why']} How to run: {r['how_to_run']}" for i, r in enumerate(results[:5]))
    try:
        return llm.complete(ANSWER_SYSTEM, f"Request: {question}\nCandidates:\n{facts}", max_tokens=220, timeout=60).strip()
    except Exception:
        return None


# ---------------------------------------------------------------- entry point


def ask(question: str, device: str | None = None, limit: int = 8, use_llm: bool = True, answer: bool = True) -> dict:
    t0 = time.time()
    p = make_plan(question, device, use_llm=use_llm)
    t1 = time.time()
    text = p.semantic_query
    if p.keywords:
        text += " " + " ".join(p.keywords)
    relaxed: list[str] = []
    ranked: list[tuple[float, dict]] = []
    for i, (label, opts) in enumerate(LADDER):
        pts = hybrid_search(text, build_filter(p, opts))
        ranked = collapse(pts, p)
        if i:
            relaxed.append(label)
        if len(ranked) >= MIN_RESULTS:
            break
    t2 = time.time()
    # show the original model (root) with all its builds, not a random copy
    with catalog.session() as con:
        roots = catalog.get_many(con, list({r.get("family_root") or r["id"] for _, r in ranked[:limit]}))
    results = []
    for s, r in ranked[:limit]:
        rec = roots.get(r.get("family_root") or r["id"]) or r
        results.append(to_result(rec, s, p))
    t3 = time.time()
    ans = overview(question, results) if answer and use_llm else None
    plan = asdict(p)
    plan["parser"] = p.parser
    return {
        "question": question, "plan": plan, "relaxed": relaxed, "answer": ans, "results": results,
        "timing_ms": {"parse": int((t1 - t0) * 1000), "search": int((t2 - t1) * 1000),
                      "explain": int((t3 - t2) * 1000), "total": int((time.time() - t0) * 1000)},
    }

"""TypeSafe Jev (a "System One" decision model) for the two judgment steps in a query:

1. understand the question  -> one call, every field asked in parallel as typed Choice/Noul questions
2. re-rank the shortlist    -> one call, one "is this a good fit?" Noul per candidate model

Jev returns calibrated probabilities in ~0.1-0.5 s instead of generating text, so both steps are much faster
than a local LLM. Numbers ("8GB", "under 3B") stay with plain code: Jev's docs say it is weak at arithmetic.
Enabled when TYPESAFE_API_KEY is set (put it in .env); any failure falls back to the LLM/rule path.
Docs: https://docs.typesafe.ai
"""
from __future__ import annotations

import os
from functools import lru_cache

from . import config
from .summarize import TASK_PLAIN, human_params, task_label

DEVICE_CRITERIA = {
    "phone": "must run on a phone or tablet: iPhone, Android, mobile, on-device",
    "mac": "must run on a Mac: MacBook, iMac, Mac mini, Apple Silicon (M1-M5)",
    "laptop": "must run on a non-Mac laptop or ordinary PC, CPU only",
    "gpu": "must run on a desktop graphics card: NVIDIA, RTX, CUDA, VRAM mentioned",
    "server": "runs on a server or cloud machine with large GPUs",
    "any": "no device or hardware is mentioned",
}
LANGUAGES = {"en": "English", "es": "Spanish", "fr": "French", "de": "German", "zh": "Chinese", "ja": "Japanese",
             "ko": "Korean", "ru": "Russian", "ar": "Arabic", "hi": "Hindi", "pt": "Portuguese", "it": "Italian",
             "tr": "Turkish", "az": "Azerbaijani", "vi": "Vietnamese", "nl": "Dutch", "pl": "Polish", "fa": "Persian"}


def available() -> bool:
    return bool(os.environ.get("TYPESAFE_API_KEY")) and config.PARSER in ("auto", "jev")


@lru_cache(maxsize=1)
def client():
    from typesafe_sdk import TypeSafeClient

    # one long-lived client: keeps the HTTPS connection warm between questions (matters at ~100 ms per call)
    return TypeSafeClient(model=config.JEV_MODEL, timeout=config.JEV_TIMEOUT)


def name() -> str:
    return f"jev:{config.JEV_MODEL}"


def _task_criteria() -> dict[str, str]:
    crit = {t: f"{label}: {plain}" for t, (label, plain) in TASK_PLAIN.items()}
    crit["unclear"] = "the request does not say what the model should do"
    return crit


def parse(question: str) -> dict:
    """Question -> the same fields the LLM parser returns (minus numbers, which the rule parser extracts)."""
    from typesafe_sdk import Choice, Noul

    lang_crit = {code: f"must handle {name}" for code, name in LANGUAGES.items()}
    lang_crit["unspecified"] = "no particular language is required"
    r = client().system_one(
        state={"request": question},
        questions={
            "task": Choice(instructions="What must the AI model in `request` do? Pick the closest task.",
                           criteria=_task_criteria()),
            "device": Choice(instructions="Where must the model in `request` run?", criteria=DEVICE_CRITERIA),
            "language": Choice(instructions="Which spoken/written language must the model in `request` support?",
                               criteria=lang_crit),
            "license_open": Noul(instructions="Does `request` require commercial use, open source, or a permissive license?",
                                 criteria={"true": "commercial use, open-source or license terms are asked for",
                                           "false": "licensing is not mentioned"}),
            "newest": Noul(instructions="Does `request` ask for new, latest, or recently released models?"),
            "code": Noul(instructions="Is `request` about writing, completing, or understanding programming code?"),
            "voice_clone": Noul(instructions="Does `request` involve cloning or imitating a specific person's voice?"),
        },
    )
    a = r.answers
    probs = a["task"].probabilities
    ranked = [t for t, p in sorted(probs.items(), key=lambda kv: -kv[1]) if t != "unclear"]
    tasks = [ranked[0]] + [t for t in ranked[1:3] if probs[t] >= 0.2]
    if a["task"].choice == "unclear" and probs.get("unclear", 0) > 0.6:
        tasks = []
    if a["voice_clone"].noul > 0.6 and "text-to-speech" in tasks and "audio-to-audio" not in tasks:
        tasks.append("audio-to-audio")
    keywords = []
    if a["code"].noul > 0.6:
        keywords.append("code")
    if a["voice_clone"].noul > 0.6:
        keywords.append("voice cloning")
    lang = a["language"].choice
    return {
        "semantic_query": question,
        "tasks": tasks,
        "device": a["device"].choice,
        "license_open": True if a["license_open"].noul > 0.7 else None,
        "created_after_days": 180 if a["newest"].noul > 0.7 else None,
        "languages": [] if lang == "unspecified" or a["language"].confidence < 0.5 else [lang],
        "keywords": keywords,
        "_jev": {k: round(v.noul, 3) if hasattr(v, "noul") else {"choice": v.choice, "confidence": round(v.confidence, 3)}
                 for k, v in a.items()},
    }


def describe(rec: dict) -> str:
    """Short, factual candidate description for the fit question (keep per-question context small)."""
    bits = [task_label(rec.get("task"))]
    if rec.get("params"):
        bits.append(f"{human_params(rec['params'])} parameters")
    ram = rec.get("family_ram_gb_min") or rec.get("ram_gb_min")
    if ram:
        bits.append(f"needs about {ram:g} GB memory")
    fmts = rec.get("family_formats") or rec.get("formats") or []
    if fmts:
        bits.append("formats " + ", ".join(fmts[:8]))
    if rec.get("languages"):
        bits.append("languages " + ", ".join(rec["languages"][:6]))
    if rec.get("license"):
        bits.append(f"license {rec['license']}")
    tags = [t for t in (rec.get("tags_clean") or []) if t not in fmts][:10]
    if tags:
        bits.append("tags " + ", ".join(tags))
    if rec.get("card_summary"):
        bits.append(str(rec["card_summary"])[:200])
    return f"{rec['id']}: " + "; ".join(bits)


def fit_scores(question: str, plan_text: str, recs: list[dict]) -> dict[str, float]:
    """One call: P(good fit) for each candidate, asked as independent parallel yes/no questions."""
    from typesafe_sdk import Noul

    qs = {
        f"c{i}": Noul(
            instructions=f"Is this model a good choice for `request`? Model: {describe(r)}",
            criteria={"true": "it does what the request asks, on the hardware it names",
                      "false": "it does a different job, is a test/toy/helper model, or would not run where asked"},
        )
        for i, r in enumerate(recs)
    }
    resp = client().system_one(state={"request": question, "requirements": plan_text}, questions=qs)
    return {r["id"]: resp.answers[f"c{i}"].noul for i, r in enumerate(recs)}

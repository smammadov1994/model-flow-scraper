"""No-Claude fallback planner: Jev picks the closest ready-made pipeline shape, then Model Finder fills it.

Covers the common shapes; anything else becomes a single model step for the task Jev detects.
"""
from __future__ import annotations

from .. import jev, query


def _s(key, typ, title, step, task=None, inputs=(), outputs=(), hint=""):
    return {"key": key, "type": typ, "title": title, "step": step, "task": task, "inputs": list(inputs),
            "outputs": list(outputs), "search_hint": hint or step,
            "constraints": {"max_params_b": None, "languages": [], "license_open": None, "prefer": None, "pin_model": None}}


def _chain(*steps):
    edges = []
    for a, b in zip(steps, steps[1:]):
        edges.append({"from": a["key"], "to": b["key"], "label": (a["outputs"] or ["data"])[0]})
    return list(steps), edges


TEMPLATES = {
    "voice_assistant": ("Voice assistant", "talk to an assistant by voice and hear it answer", lambda: _chain(
        _s("mic", "input", "Microphone", "Captures what the user says", outputs=["audio"]),
        _s("stt", "model", "Speech to text", "Turns the user's speech into text", "automatic-speech-recognition",
           ["audio"], ["text"], "fast accurate speech recognition"),
        _s("llm", "model", "Assistant brain", "Understands the request and writes a reply", "text-generation",
           ["text"], ["text"], "small instruction-tuned chat assistant"),
        _s("tts", "model", "Text to speech", "Speaks the reply out loud", "text-to-speech", ["text"], ["audio"],
           "natural sounding text to speech voice"),
        _s("speaker", "output", "Speaker", "Plays the spoken reply", inputs=["audio"]))),
    "transcribe_summarize": ("Transcribe & summarize", "turn recordings or meetings into transcripts and summaries", lambda: _chain(
        _s("audio", "input", "Recording", "An audio or video recording", outputs=["audio"]),
        _s("stt", "model", "Speech to text", "Transcribes the recording", "automatic-speech-recognition", ["audio"], ["text"],
           "accurate long-form speech transcription"),
        _s("llm", "model", "Summarizer", "Writes a short summary and action items", "text-generation", ["text"], ["text"],
           "instruction-tuned model good at summarization"),
        _s("doc", "output", "Summary", "The transcript and summary", inputs=["text"]))),
    "image_generation": ("Image generator", "create pictures or product photos from a text prompt", lambda: _chain(
        _s("prompt", "input", "Prompt", "Describes the picture to make", outputs=["text"]),
        _s("t2i", "model", "Image generation", "Creates the image from the prompt", "text-to-image", ["text"], ["image"],
           "high quality text to image diffusion model"),
        _s("img", "output", "Image", "The generated picture", inputs=["image"]))),
    "doc_qa": ("Document Q&A", "ask questions about your own documents (RAG)", lambda: (
        [_s("docs", "input", "Documents", "Your files (PDF, notes, pages)", outputs=["text"]),
         _s("embed", "model", "Embeddings", "Turns document chunks into search vectors", "feature-extraction", ["text"],
            ["embeddings"], "text embedding model for semantic search and RAG"),
         _s("index", "logic", "Vector index", "Stores chunks and finds the relevant ones", inputs=["embeddings"], outputs=["text"]),
         _s("question", "input", "Question", "What the user asks", outputs=["text"]),
         _s("llm", "model", "Answer writer", "Answers using the retrieved passages", "text-generation", ["text"], ["text"],
            "instruction-tuned chat model good at grounded answers"),
         _s("answer", "output", "Answer", "The answer with sources", inputs=["text"])],
        [{"from": "docs", "to": "embed", "label": "text"}, {"from": "embed", "to": "index", "label": "embeddings"},
         {"from": "question", "to": "index", "label": "text"}, {"from": "index", "to": "llm", "label": "passages"},
         {"from": "question", "to": "llm", "label": "text"}, {"from": "llm", "to": "answer", "label": "text"}])),
    "chatbot": ("Chat assistant", "a text chatbot or coding assistant", lambda: _chain(
        _s("msg", "input", "Message", "What the user types", outputs=["text"]),
        _s("llm", "model", "Chat model", "Writes the reply", "text-generation", ["text"], ["text"], "instruction-tuned chat assistant"),
        _s("reply", "output", "Reply", "Shown to the user", inputs=["text"]))),
    "speech_translation": ("Speech translator", "translate what someone says into another language, spoken", lambda: _chain(
        _s("mic", "input", "Microphone", "Captures speech", outputs=["audio"]),
        _s("stt", "model", "Speech to text", "Transcribes the speech", "automatic-speech-recognition", ["audio"], ["text"],
           "multilingual speech recognition"),
        _s("mt", "model", "Translator", "Translates the text", "translation", ["text"], ["text"], "multilingual translation"),
        _s("tts", "model", "Text to speech", "Speaks the translation", "text-to-speech", ["text"], ["audio"], "multilingual text to speech"),
        _s("speaker", "output", "Speaker", "Plays the translation", inputs=["audio"]))),
    "vision_assistant": ("Vision assistant", "look at photos, screenshots or documents and answer questions", lambda: _chain(
        _s("cam", "input", "Image", "A photo, screenshot or scanned page", outputs=["image"]),
        _s("vlm", "model", "Vision + chat", "Looks at the image and answers", "image-text-to-text", ["image"], ["text"],
           "vision language model for image understanding and OCR"),
        _s("out", "output", "Answer", "The description or extracted text", inputs=["text"]))),
    "video_generation": ("Video generator", "make short video clips from text or an image", lambda: _chain(
        _s("prompt", "input", "Prompt", "Describes the clip", outputs=["text"]),
        _s("t2v", "model", "Video generation", "Creates the video clip", "text-to-video", ["text"], ["video"], "text to video model"),
        _s("vid", "output", "Video", "The generated clip", inputs=["video"]))),
}


def plan(message: str, pipeline: dict | None) -> dict:
    """Plan without Claude: Jev (or keyword rules) picks a template; device from the same parse."""
    parsed = {}
    shape = None
    if jev.available():
        from typesafe_sdk import Choice

        try:
            parsed = jev.parse(message)
            crit = {k: f"{title}: {desc}" for k, (title, desc, _) in TEMPLATES.items()}
            crit["single_model"] = "just one AI model doing one job, none of the above"
            r = jev.client().system_one(state={"request": message}, questions={
                "shape": Choice(instructions="Which kind of AI pipeline does `request` want to build?", criteria=crit)})
            shape = r.answers["shape"].choice
        except Exception:  # noqa: BLE001
            shape = None
    p = query.make_plan(message, use_llm=False) if not parsed else None
    tasks = parsed.get("tasks") or (p.tasks if p else [])
    device = parsed.get("device") or (p.device if p else "any")
    if shape is None:  # keyword fallback
        m = message.lower()
        shape = ("voice_assistant" if "assistant" in m and ("voice" in m or "talk" in m) else
                 "transcribe_summarize" if "transcri" in m or "meeting" in m else
                 "doc_qa" if any(w in m for w in ("document", "pdf", "rag", "my notes", "my files")) else
                 "speech_translation" if "translat" in m and ("speech" in m or "voice" in m) else
                 "image_generation" if "image" in m or "photo" in m or "picture" in m else
                 "single_model")
    if shape in TEMPLATES:
        title, _, build = TEMPLATES[shape]
        steps, edges = build()
    else:
        task = tasks[0] if tasks else "text-generation"
        title = "Single model"
        steps, edges = _chain(
            _s("in", "input", "Input", "What goes in", outputs=["data"]),
            _s("model", "model", "Model", message[:120], task, ["data"], ["data"], message),
            _s("out", "output", "Output", "The result", inputs=["data"]))
    import re

    mem = re.search(r"(\d+(?:\.\d+)?)\s*gb", message.lower())
    return {"reply": f"I laid out a {title.lower()} pipeline and picked models for each step.",
            "title": title, "device": {"kind": device, "label": device, "memory_gb": float(mem.group(1)) if mem else None},
            "steps": steps, "edges": edges}

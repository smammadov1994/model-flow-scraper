"""Builds the plain-English "what this model does" text that gets embedded and keyword-indexed."""
from __future__ import annotations

TASK_PLAIN: dict[str, tuple[str, str]] = {
    # task: (short label, what it does in plain words incl. common synonyms people search for)
    "text-to-speech": ("Text to speech", "turns written text into natural spoken audio: voice synthesis, TTS, voice generation, voice cloning, speaking assistant voice"),
    "text-to-audio": ("Text to audio", "generates audio, music or sound effects from a text description"),
    "audio-to-audio": ("Audio to audio", "transforms audio: voice conversion, voice changing, speech enhancement, noise removal, source separation"),
    "automatic-speech-recognition": ("Speech to text", "transcribes spoken audio into text: speech recognition, dictation, subtitles, voice input"),
    "audio-classification": ("Audio classification", "labels sounds or speech: audio tagging, emotion, speaker or language identification"),
    "voice-activity-detection": ("Voice activity detection", "detects when someone is speaking; speaker diarization"),
    "audio-text-to-text": ("Audio + text chat", "listens to audio and answers in text: audio understanding, spoken assistant"),
    "text-generation": ("Chat / text", "large language model that writes and answers text: chatbot, assistant, reasoning, writing, coding help"),
    "text2text-generation": ("Text to text", "rewrites or transforms text: translation, summarization, paraphrasing"),
    "summarization": ("Summarization", "shortens long text into a summary"),
    "translation": ("Translation", "translates text between languages"),
    "question-answering": ("Question answering", "finds answers to questions inside a passage of text"),
    "fill-mask": ("Fill mask", "language encoder for understanding text (BERT-style), base for classifiers"),
    "text-classification": ("Text classification", "labels text: sentiment, topic, moderation, intent"),
    "token-classification": ("Named entities", "tags words in text: named entity recognition, PII detection, part of speech"),
    "zero-shot-classification": ("Zero-shot classification", "labels text with any categories you name"),
    "feature-extraction": ("Embeddings", "turns text into vectors for semantic search, RAG, clustering and similarity"),
    "sentence-similarity": ("Embeddings", "turns sentences into vectors for semantic search, RAG and similarity"),
    "text-ranking": ("Reranker", "reranks search results by relevance for RAG and search"),
    "image-classification": ("Image classification", "labels what is in a photo"),
    "object-detection": ("Object detection", "finds and boxes objects in images and video frames"),
    "image-segmentation": ("Image segmentation", "cuts out objects or regions in an image, background removal"),
    "depth-estimation": ("Depth estimation", "estimates distance/depth from a single image"),
    "zero-shot-image-classification": ("Image-text matching", "CLIP-style model that matches images with text labels; image search"),
    "zero-shot-object-detection": ("Open-vocabulary detection", "finds any object you describe in words"),
    "image-feature-extraction": ("Image embeddings", "turns images into vectors for image search and similarity"),
    "mask-generation": ("Mask generation", "segments anything in an image from clicks or boxes"),
    "keypoint-detection": ("Keypoint detection", "finds body pose or face keypoints"),
    "text-to-image": ("Image generation", "creates images from a text prompt: AI art, diffusion, picture generation"),
    "image-to-image": ("Image editing", "edits or restyles an image: inpainting, upscaling, style transfer"),
    "unconditional-image-generation": ("Image generation", "generates images without a prompt"),
    "text-to-video": ("Video generation", "creates video clips from a text prompt"),
    "image-to-video": ("Image to video", "animates a still image into a video clip"),
    "video-to-video": ("Video editing", "edits or restyles video"),
    "text-to-3d": ("3D generation", "creates 3D objects from text"),
    "image-to-3d": ("3D generation", "creates 3D objects from an image"),
    "image-to-text": ("Image captioning / OCR", "describes images or reads text in them (OCR)"),
    "image-text-to-text": ("Vision + chat", "multimodal model that looks at images and answers in text: visual assistant, document and screenshot understanding, OCR"),
    "visual-question-answering": ("Visual Q&A", "answers questions about an image"),
    "document-question-answering": ("Document Q&A", "answers questions about scanned documents and forms"),
    "video-text-to-text": ("Video understanding", "watches video and answers in text"),
    "video-classification": ("Video classification", "labels what happens in a video"),
    "any-to-any": ("Omni model", "multimodal model that takes and produces several of text, image, audio, speech"),
    "time-series-forecasting": ("Forecasting", "predicts future values of time series"),
    "reinforcement-learning": ("Reinforcement learning", "trained agent/policy for games or control"),
    "robotics": ("Robotics", "robot control policy, vision-language-action"),
}

TIER_PLAIN = {
    "phone": "small enough to run on a phone",
    "laptop": "runs on a laptop or MacBook",
    "gpu": "needs a desktop GPU (up to ~24 GB)",
    "server": "needs a multi-GPU server",
    "unknown": "size unknown",
}


def task_label(task: str | None) -> str:
    if not task:
        return "Unknown task"
    return TASK_PLAIN.get(task, (task.replace("-", " ").capitalize(), ""))[0]


def human_params(n: int | None) -> str | None:
    if not n:
        return None
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            v = n / div
            return f"{v:.0f}{unit}" if v >= 10 else f"{v:.1f}".rstrip("0").rstrip(".") + unit
    return str(n)


def build_summary(r: dict, card_summary: str | None = None) -> str:
    """Deterministic description used for dense + BM25 search. Keep the important words early."""
    task = r.get("task")
    label, plain = TASK_PLAIN.get(task or "", (task_label(task), ""))
    name = r.get("name") or r["id"]
    parts = [f"{name.replace('-', ' ').replace('_', ' ')} ({r['id']})."]
    parts.append(f"{label}: {plain}." if plain else f"{label}.")
    if r.get("modality") == "code":
        parts.append("Code model for programming, code completion and coding assistants.")
    if card_summary:
        parts.append(card_summary.strip())
    p = human_params(r.get("params"))
    if p:
        parts.append(f"{p} parameters.")
    tier = r.get("family_device_tier") or r.get("device_tier") or "unknown"
    ram = r.get("family_ram_gb_min") or r.get("ram_gb_min")
    if tier != "unknown":
        parts.append(f"Hardware: {TIER_PLAIN[tier]}" + (f", about {ram:g} GB of memory." if ram else "."))
    fmts = r.get("family_formats") or r.get("formats") or []
    if fmts:
        parts.append("Formats: " + ", ".join(fmts) + ".")
    if r.get("languages"):
        parts.append("Languages: " + ", ".join(r["languages"][:12]) + ".")
    if r.get("base_model"):
        parts.append(f"Based on {r['base_model']}" + (f" ({r['relation']})." if r.get("relation") else "."))
    if r.get("license"):
        parts.append(f"License {r['license']}.")
    tags = [t for t in r.get("tags_clean") or [] if t not in fmts][:25]
    if tags:
        parts.append("Tags: " + ", ".join(tags) + ".")
    if r.get("library"):
        parts.append(f"Library: {r['library']}.")
    if r.get("x_takes"):
        parts.append("What people say on X: " + " ".join(r["x_takes"][:3]))
    if r.get("status") in ("announced", "seen_on_x"):
        parts.append("Announced, weights not on Hugging Face yet.")
    elif r.get("status") == "api_only":
        parts.append("Available through the Replicate API.")
    return " ".join(parts)

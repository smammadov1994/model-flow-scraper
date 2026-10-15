"""Hook for X/Twitter monitoring (out of scope for v1).

The idea: watch accounts/lists/searches that announce models (lab accounts, @_akhaliq, @huggingface, ...),
pull Hugging Face links out of posts, and index those repos immediately — often hours before they trend.

What exists today:
  * `extract_hf_ids(text)` — finds `huggingface.co/<org>/<model>` links (and `hf.co/...`) in any text.
  * `model-finder add-from-text <file or ->` — runs it over pasted posts / an export and indexes the hits.

To finish later: implement `XSource.discover()` (X API v2 search/recent with a bearer token, or an RSS bridge),
then call `hf_poll.add(ids, source="x")` on a schedule next to `model-finder update`.
"""
from __future__ import annotations

import re

_LINK = re.compile(r"(?:https?://)?(?:www\.)?(?:huggingface\.co|hf\.co)/(?!datasets/|spaces/|papers/|blog/|docs/)"
                   r"([A-Za-z0-9][\w.-]{0,95})/([A-Za-z0-9][\w.-]{0,95})")


def extract_hf_ids(text: str) -> list[str]:
    ids = []
    for org, name in _LINK.findall(text):
        name = name.rstrip(".")
        if org.lower() in {"models", "settings", "join", "login", "organizations"}:
            continue
        ids.append(f"{org}/{name}")
    return list(dict.fromkeys(ids))


class XSource:
    """Placeholder. discover() should return HF repo ids seen in recent posts."""

    def __init__(self, bearer_token: str | None = None, queries: list[str] | None = None):
        self.bearer_token = bearer_token
        self.queries = queries or ['"huggingface.co/" (model OR weights OR release)']

    def discover(self) -> list[str]:
        raise NotImplementedError("X monitoring is not built yet — see module docstring")

"""Shared tokenization and stop-word handling for retrieval."""

from __future__ import annotations

import re

TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]+")

# Common English stop words plus query-scaffolding verbs that add little retrieval value.
STOP_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "describe",
    "described",
    "document",
    "documents",
    "explain",
    "explained",
    "followed",
    "following",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "mention",
    "mentions",
    "of",
    "on",
    "or",
    "regarding",
    "report",
    "reported",
    "reporting",
    "reports",
    "says",
    "section",
    "sections",
    "show",
    "shows",
    "that",
    "the",
    "to",
    "use",
    "used",
    "using",
    "what",
    "with",
}


def tokenize_text(text: str, *, min_length: int = 1) -> list[str]:
    """Return normalized retrieval tokens with shared stop-word filtering."""

    return [
        token
        for token in TOKEN_PATTERN.findall((text or "").lower())
        if token not in STOP_WORDS and len(token) >= min_length
    ]

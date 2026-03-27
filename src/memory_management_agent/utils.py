from __future__ import annotations

import re
from typing import Iterable, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_text(text: str) -> str:
    tokens = _TOKEN_RE.findall(text.lower())
    return " ".join(tokens)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def token_count(text: str) -> int:
    return max(1, len(tokenize(text)))


def jaccard_similarity(left: str, right: str) -> float:
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens and not right_tokens:
        return 1.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def contains_any(text: str, terms: Sequence[str] | Iterable[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)

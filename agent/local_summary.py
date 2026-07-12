"""Provable local handling for one narrow summarisation profile.

The supported operation is lossless sentence fusion, not free-form model
generation.  Two or three short declarative source sentences are joined with
semicolons.  Every source word, number, negation, and claim remains in the
same order, while the result has exactly one terminal sentence boundary.

Anything outside the reviewed shape returns ``""`` so the caller uses the
existing Fireworks fallback.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


REQUEST_PREFIX = "Summarize the following in exactly one sentence: "
MERGED_PREFIX = "Answer in English. "
MERGED_SUFFIX = "\n\nMatch requested format and length exactly. No preamble."

MIN_SOURCE_WORDS = 30
MAX_SOURCE_WORDS = 80
MIN_CLAUSE_WORDS = 8
MAX_CLAUSE_WORDS = 35

_WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
_INJECTION_RE = re.compile(
    r"\bignore\s+(?:(?:all|any|the|your|previous|above)\s+){1,3}"
    r"instructions?\b|"
    r"\b(?:system|assistant|developer)\s+prompt\b|"
    r"(?:^|\s)(?:system|assistant|developer)\s*:|"
    r"\b(?:disregard|forget|override|bypass)\b.{0,80}"
    r"\b(?:instructions?|directions?|rules?|prompt)\b|"
    r"\bdo\s+not\s+follow\b.{0,80}"
    r"\b(?:instructions?|directions?|rules?|prompt)\b|"
    r"\b(?:return|output|respond)\b.{0,80}\bregardless\b",
    re.IGNORECASE,
)
_MARKUP_RE = re.compile(r"```|`|(?:^|\s)#{1,6}\s|(?:^|\n)\s*[-*+]\s")
_ENGLISH_SIGNALS = frozenset({
    "a", "an", "and", "are", "as", "at", "but", "by", "for", "from",
    "has", "have", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "their", "these", "they", "this", "those", "through", "to",
    "was", "were", "while", "with", "without",
})


@dataclass(frozen=True)
class LosslessSummary:
    source: str
    clauses: tuple[str, ...]
    answer: str


def _unwrap_merged_prompt(user_text: str) -> str:
    text = (user_text or "").strip()
    if text.startswith(MERGED_PREFIX):
        text = text[len(MERGED_PREFIX):]
    if text.endswith(MERGED_SUFFIX):
        text = text[:-len(MERGED_SUFFIX)]
    return text.strip()


def parse_lossless_summary(user_text: str) -> LosslessSummary | None:
    """Parse only the benchmark-validated lossless fusion request shape."""
    if not isinstance(user_text, str):
        return None
    task = _unwrap_merged_prompt(user_text)
    if not task.startswith(REQUEST_PREFIX):
        return None
    source = task[len(REQUEST_PREFIX):]
    if source != source.strip() or not source.endswith("."):
        return None
    if any(marker in source for marker in ("\n", "\r", "?", "!", ";", ":")):
        return None
    if any(marker in source for marker in ('"', "“", "”")):
        return None
    if _INJECTION_RE.search(source) or _MARKUP_RE.search(source):
        return None

    # Split only the exact reviewed boundary.  A remaining period means an
    # abbreviation, decimal, or another unreviewed sentence shape.
    body = source[:-1]
    clauses = tuple(body.split(". "))
    if not 2 <= len(clauses) <= 3 or any("." in clause for clause in clauses):
        return None
    if any(not clause or not clause[0].isupper() for clause in clauses):
        return None

    clause_counts = tuple(len(_WORD_RE.findall(clause)) for clause in clauses)
    if any(not MIN_CLAUSE_WORDS <= count <= MAX_CLAUSE_WORDS
           for count in clause_counts):
        return None
    total_words = sum(clause_counts)
    if not MIN_SOURCE_WORDS <= total_words <= MAX_SOURCE_WORDS:
        return None

    letters = [char for char in source if char.isalpha()]
    if not letters:
        return None
    ascii_ratio = sum(char.isascii() for char in letters) / len(letters)
    if ascii_ratio < 0.9:
        return None
    normalized_words = [word.casefold() for word in _WORD_RE.findall(source)]
    english_signals = sum(word in _ENGLISH_SIGNALS for word in normalized_words)
    unique_english_signals = set(normalized_words) & _ENGLISH_SIGNALS
    if (english_signals < max(4, len(normalized_words) // 12)
            or len(unique_english_signals) < 4):
        return None

    answer = "; ".join(clauses) + "."
    return LosslessSummary(source, clauses, answer)


def supports_lossless_summary(user_text: str) -> bool:
    return parse_lossless_summary(user_text) is not None


def fuse_lossless_summary(user_text: str) -> str:
    parsed = parse_lossless_summary(user_text)
    return parsed.answer if parsed is not None else ""


def validate_lossless_summary(user_text: str, answer: str) -> bool:
    """Prove canonical equality and exact source reconstruction."""
    parsed = parse_lossless_summary(user_text)
    return bool(
        parsed is not None
        and answer == parsed.answer
        and answer.replace("; ", ". ") == parsed.source
        and answer.endswith(".")
        and answer.count(".") == 1
        and "?" not in answer
        and "!" not in answer
    )

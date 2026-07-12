"""Narrow deterministic tools for exactly provable arithmetic answers.

The public entry point intentionally returns ``""`` for every unsupported,
ambiguous, or invalid prompt.  Callers can therefore use a non-empty answer as
proof that the reviewed grammar matched and otherwise fall back to Fireworks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction


_ENGLISH_PREFIX = "Answer in English. "
_MATH_SUFFIX = "\n\nAnswer first; show only essential working."
_MAX_PROMPT_LENGTH = 1000
_MAX_COUNT = 10**15

# Counts are integers because the supported questions ask about discrete
# objects.  Comma grouping is accepted only when it is syntactically valid.
_COUNT = r"(?:0|[1-9]\d{0,15}|[1-9]\d{0,2}(?:,\d{3}){1,5})"
_PERCENT = r"(?:0|[1-9]\d{0,2})(?:\.\d{1,6})?"
_SAFE_WORDS = r"[A-Za-z]+(?:[ '-][A-Za-z]+){0,7}"
_WEEKDAY = r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
_DAY_PART = r"(?:morning|afternoon|evening|night)"
_TIME_POINT = rf"(?:noon|midday|midnight|{_DAY_PART})"
_CONTEXT = (
    rf"(?:on\s+{_WEEKDAY}|"
    rf"(?:in|during)\s+(?:the\s+)?{_DAY_PART}|"
    rf"(?:by|before|after)\s+(?:the\s+)?{_TIME_POINT})"
)
_UNSAFE_INSTRUCTION_RE = re.compile(
    r"\b(?:ignore|disregard|override)\b|"
    r"\b(?:system|developer|assistant)\s+prompt\b|"
    r"\b(?:previous|above)\s+instructions?\b",
    re.IGNORECASE,
)

_TASK_RE = re.compile(
    rf"^(?:A|An|The)\s+(?P<holder>{_SAFE_WORDS})\s+has\s+"
    rf"(?P<initial>{_COUNT})\s+(?P<unit>{_SAFE_WORDS})\.\s*"
    # The event parser below is fully anchored and allowlisted.  This outer
    # capture stays broad enough for a decimal percentage such as 12.5%.
    rf"(?P<events>.{{1,240}}?)\.\s*"
    rf"How\s+many\s+(?P<query_unit>{_SAFE_WORDS})\s+"
    rf"(?:remain|are\s+left)\?$",
    re.IGNORECASE,
)

# The fixed second removal inherits the same explicit removal verb.  Requiring
# a time/context preposition after "more" prevents accepting text such as
# "and 20 more arrive" as another removal.
_ACTOR_REMOVAL_RE = re.compile(
    rf"^It\s+(?:sells?|sold|ships?|shipped|checks?\s+out|checked\s+out|"
    rf"removes?|removed|uses?|used|loses?|lost|discards?|discarded|"
    rf"donates?|donated|gives?\s+away|gave\s+away)\s+"
    rf"(?P<percent>{_PERCENT})%\s+{_CONTEXT}\s+and\s+"
    rf"(?P<fixed>{_COUNT})\s+more\s+{_CONTEXT}$",
    re.IGNORECASE,
)

# Some inventories describe the removed group rather than an actor performing
# an action ("20% are absent ... and 15 more leave ...").  Both predicates are
# allowlisted removal states/actions and must be explicit.
_GROUP_REMOVAL_RE = re.compile(
    rf"^(?P<percent>{_PERCENT})%\s+"
    rf"(?:are\s+absent|leave|depart|are\s+removed|are\s+sold|are\s+shipped)\s+"
    rf"(?:(?:{_CONTEXT})|early|late)\s+and\s+"
    rf"(?P<fixed>{_COUNT})\s+more\s+"
    rf"(?:leave|depart|are\s+removed|are\s+sold|are\s+shipped)\s+"
    rf"(?:(?:{_CONTEXT})|early|late)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class InventoryRemainderProblem:
    """Exact inputs for the reviewed percent-then-fixed-removal profile."""

    initial: int
    percent_removed: Fraction
    fixed_removed: int


def _strip_merged_wrapper(user_text: object) -> str:
    if not isinstance(user_text, str):
        return ""
    text = user_text.strip()
    if text.startswith(_ENGLISH_PREFIX):
        text = text[len(_ENGLISH_PREFIX):]
    if text.endswith(_MATH_SUFFIX):
        text = text[:-len(_MATH_SUFFIX)]
    return text.strip()


def _parse_count(raw: str) -> int | None:
    try:
        value = int(raw.replace(",", ""))
    except (TypeError, ValueError):
        return None
    return value if 0 <= value <= _MAX_COUNT else None


def parse_inventory_remainder(user_text: str) -> InventoryRemainderProblem | None:
    """Parse the reviewed inventory grammar, or return ``None`` fail-closed."""
    task = _strip_merged_wrapper(user_text)
    if not task or len(task) > _MAX_PROMPT_LENGTH:
        return None
    if _UNSAFE_INSTRUCTION_RE.search(task):
        return None

    match = _TASK_RE.fullmatch(task)
    if not match:
        return None
    if match.group("unit").casefold() != match.group("query_unit").casefold():
        return None

    events = match.group("events")
    removal = _ACTOR_REMOVAL_RE.fullmatch(events)
    if removal is None:
        removal = _GROUP_REMOVAL_RE.fullmatch(events)
    if removal is None:
        return None

    initial = _parse_count(match.group("initial"))
    fixed = _parse_count(removal.group("fixed"))
    if initial is None or fixed is None or initial == 0:
        return None
    try:
        percent = Fraction(removal.group("percent"))
    except (ValueError, ZeroDivisionError):
        return None
    if not 0 <= percent <= 100:
        return None

    percentage_count = initial * percent / 100
    # Discrete inventories cannot safely infer a rounding convention.
    if percentage_count.denominator != 1:
        return None
    remaining = initial - percentage_count - fixed
    if remaining.denominator != 1 or remaining < 0:
        return None
    return InventoryRemainderProblem(initial, percent, fixed)


def solve_math(user_text: str) -> str:
    """Return an exact numeric answer, or ``""`` for unsupported input."""
    problem = parse_inventory_remainder(user_text)
    if problem is None:
        return ""
    percentage_count = problem.initial * problem.percent_removed / 100
    remaining = problem.initial - percentage_count - problem.fixed_removed
    if remaining.denominator != 1 or remaining < 0:
        return ""
    return str(remaining.numerator)

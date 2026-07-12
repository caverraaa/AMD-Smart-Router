"""Pure, conservative validation helpers for locally generated answers.

The local lane must fail closed: an answer that does not satisfy a cheap,
deterministic structural check is returned as an empty string by LocalModel,
which lets the existing caller fall back to Fireworks.
"""
import re
from dataclasses import dataclass

try:
    from agent.local_tools import solve_assignment_logic
except ImportError:  # executed with /app/agent on sys.path
    from local_tools import solve_assignment_logic


NER_TYPES = ("PERSON", "ORG", "LOCATION", "DATE")
NER_FIRST_SUFFIX = (
    "Use exactly one pair per line: exact source text — PERSON, ORG, LOCATION, "
    "or DATE. Preserve relative-date words."
)

_NER_REQUEST_RE = re.compile(
    r"\bextract\b.{0,80}\b(?:named\s+)?entit(?:y|ies)\b|"
    r"\blist\b.{0,80}\b(?:person|organi[sz]ation|location|date)s?\b",
    re.IGNORECASE | re.DOTALL,
)
_UNSUPPORTED_FORMAT_RE = re.compile(
    r"\b(?:json|csv|xml|ya?ml|table|markdown|dictionary|dict|array)\b",
    re.IGNORECASE,
)
_UNSUPPORTED_TYPE_RE = re.compile(
    r"\b(?:PRODUCT|EVENT|MONEY|PERCENT|QUANTITY|TIME|LANGUAGE|LAW|WORK_OF_ART)\b",
    re.IGNORECASE,
)
_NER_LINE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?[\"']?(?P<entity>.+?)[\"']?\s*"
    r"(?:(?:—|–|-|:)\s*)?"
    r"(?P<type>PERSON|PER|ORG|ORGANIZATION|ORGANISATION|LOCATION|LOC|DATE)"
    r"\s*[.;]?\s*$",
    re.IGNORECASE,
)
_TYPE_ALIASES = {
    "PERSON": "PERSON",
    "PER": "PERSON",
    "ORG": "ORG",
    "ORGANIZATION": "ORG",
    "ORGANISATION": "ORG",
    "LOCATION": "LOCATION",
    "LOC": "LOCATION",
    "DATE": "DATE",
}

_MONTH = (
    r"January|February|March|April|May|June|July|August|September|"
    r"October|November|December"
)
_WEEKDAY = r"Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"
_SEASON = r"spring|summer|autumn|fall|winter"
_RELATIVE_PERIOD = rf"{_WEEKDAY}|{_SEASON}|{_MONTH}|day|week|month|year"
_DATE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        rf"\b(?:last|next|this)\s+(?:{_RELATIVE_PERIOD})\b",
        r"\b(?:yesterday|today|tomorrow|tonight)\b",
        rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTH})(?:\s+\d{{4}})?\b",
        rf"\b(?:{_MONTH})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?\b",
        rf"\b(?:{_MONTH})\s+\d{{4}}\b",
        rf"\b(?:{_MONTH})\b",
    )
)


@dataclass(frozen=True)
class NerEntity:
    text: str
    type: str


@dataclass(frozen=True)
class NerValidation:
    valid: bool
    answer: str
    issues: tuple[str, ...]


@dataclass(frozen=True)
class LogicValidation:
    valid: bool
    answer: str
    issues: tuple[str, ...]


def validate_logic_answer(user_text, answer):
    """Prove and canonicalize a supported assignment-puzzle answer.

    The expected answer is derived from all parsed constraints, not from the
    model output.  Unsupported and ambiguous puzzles fail closed.
    """
    expected = solve_assignment_logic(user_text)
    if not expected:
        return LogicValidation(False, "", ("unsupported or ambiguous logic",))
    if _normalized(answer).strip(" .") != _normalized(expected).strip(" ."):
        return LogicValidation(False, "", ("answer does not satisfy constraints",))
    return LogicValidation(True, expected, ())


def _normalized(text):
    return " ".join(text.casefold().split())


def is_ner_request(user_text):
    """Return True only for extraction requests, not task classification."""
    return bool(_NER_REQUEST_RE.search(user_text or ""))


def supports_local_ner_request(user_text):
    """Accept only the locally validated schema and line-oriented format."""
    text = (user_text or "").strip().split("\n\n", 1)[0]
    instruction = re.split(r"\bfrom\s*:", text, maxsplit=1,
                           flags=re.IGNORECASE)[0]
    return bool(
        is_ner_request(text)
        and not _UNSUPPORTED_FORMAT_RE.search(instruction)
        and not _UNSUPPORTED_TYPE_RE.search(instruction)
    )


def extract_ner_source(user_text):
    """Isolate the source sentence from main.py's merged local message.

    This intentionally drops the appended category constraint; otherwise its
    example relative date (currently ``next month``) would look like an entity
    that the answer must contain.
    """
    task = (user_text or "").strip().split("\n\n", 1)[0].strip()
    task = re.sub(r"^Answer in English\.\s*", "", task, flags=re.IGNORECASE)
    match = re.search(r"\bfrom\s*:\s*(.+)$", task, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"\b(?:text|sentence)\s*:\s*(.+)$", task,
                      flags=re.IGNORECASE | re.DOTALL)
    return (match.group(1) if match else task).strip()


def build_ner_prompt(user_text, previous_answer=None, issues=()):
    """Build Gemma's category-specific extraction or one-shot repair prompt."""
    source = extract_ner_source(user_text)
    if previous_answer is None:
        # Preserve the original, judge-tested merged prompt. This one narrow
        # format nudge fixes Gemma's occasional alternating-line output and
        # dropped relative-date prefix without changing entity semantics.
        return f"{user_text.rstrip()}\n{NER_FIRST_SUFFIX}"
    rules = (
        "Copy every entity span exactly from the source, including words like "
        "last, next, or this in relative dates. Do not shorten, normalize, or "
        "translate spans. Output only `exact entity — TYPE`, one per line. "
        "TYPE must be PERSON, ORG, LOCATION, or DATE."
    )
    feedback = "; ".join(issues) if issues else "invalid structure"
    return (
        f"Repair the NER answer. {rules}\nSource: {source}\n"
        f"Previous answer:\n{previous_answer}\nProblems: {feedback}\nCorrected answer:"
    )


def parse_ner_answer(answer):
    """Parse common terse NER formats, returning None on any extra text."""
    lines = [line.strip() for line in (answer or "").splitlines() if line.strip()]
    if not lines:
        return None
    entities = []
    for line in lines:
        match = _NER_LINE_RE.fullmatch(line)
        if not match:
            return None
        entity = match.group("entity").strip().strip("\"'")
        if not entity:
            return None
        entities.append(NerEntity(entity, _TYPE_ALIASES[match.group("type").upper()]))
    return tuple(entities)


def find_date_mentions(source):
    """Return non-overlapping date spans, preferring the longest exact span."""
    spans = []
    for pattern in _DATE_PATTERNS:
        for match in pattern.finditer(source or ""):
            start, end = match.span()
            if any(start < old_end and end > old_start for old_start, old_end, _ in spans):
                continue
            spans.append((start, end, match.group(0)))
    return tuple(text for _, _, text in sorted(spans))


def validate_ner_answer(user_text, answer):
    """Validate and canonicalize a local NER answer.

    The check is deliberately conservative where it can be certain: every
    entity must be copied from the source, every recognisable source date must
    be present with its complete span, and no untyped prose is accepted.
    Semantic completeness for non-date entities remains a golden-set gate.
    """
    source = extract_ner_source(user_text)
    parsed = parse_ner_answer(answer)
    if parsed is None:
        return NerValidation(False, "", ("use one entity and type per line",))

    issues = []
    source_normalized = _normalized(source)
    seen = set()
    for item in parsed:
        key = (_normalized(item.text), item.type)
        if key in seen:
            issues.append(f"remove duplicate `{item.text}`")
        seen.add(key)
        if _normalized(item.text) not in source_normalized:
            issues.append(f"`{item.text}` is not an exact source span")

    date_entities = {
        _normalized(item.text) for item in parsed if item.type == "DATE"
    }
    expected_dates = {_normalized(mention) for mention in find_date_mentions(source)}
    for mention in find_date_mentions(source):
        if _normalized(mention) not in date_entities:
            issues.append(f"copy date span `{mention}` exactly")
    for item in parsed:
        if item.type == "DATE" and expected_dates and _normalized(item.text) not in expected_dates:
            issues.append(f"remove or expand incomplete date `{item.text}`")

    if issues:
        return NerValidation(False, "", tuple(dict.fromkeys(issues)))
    canonical = "\n".join(f"{item.text} — {item.type}" for item in parsed)
    return NerValidation(True, canonical, ())


def repair_shortened_date_spans(user_text, answer):
    """Safely restore a dropped relative-date prefix before LLM repair.

    Gemma occasionally copies ``March`` instead of ``last March``. This only
    rewrites a DATE when exactly one source date ends with that emitted span;
    ambiguity is left untouched so validation can fail closed.
    """
    parsed = parse_ner_answer(answer)
    if parsed is None:
        return answer
    expected = tuple(find_date_mentions(extract_ner_source(user_text)))
    present = {_normalized(item.text) for item in parsed if item.type == "DATE"}
    replacements = {}
    for item in parsed:
        if item.type != "DATE" or _normalized(item.text) in {
                _normalized(span) for span in expected}:
            continue
        short = _normalized(item.text)
        matches = [
            span for span in expected
            if _normalized(span) not in present
            and _normalized(span).endswith(" " + short)
        ]
        if len(matches) == 1:
            replacements[(item.text, item.type)] = matches[0]
    return "\n".join(
        f"{replacements.get((item.text, item.type), item.text)} — {item.type}"
        for item in parsed
    )


def repair_trailing_descriptors(answer):
    """Trim lowercase prose accidentally attached after a named span.

    A named PERSON/ORG/LOCATION span normally ends in a token containing an
    uppercase character (connectors such as ``of`` or ``de`` may occur inside
    it). Thus ``Tesla factory`` safely becomes ``Tesla`` while ``Museum of
    Modern Art`` stays unchanged. All-lowercase names are left untouched.
    """
    parsed = parse_ner_answer(answer)
    if parsed is None:
        return answer
    repaired = []
    for item in parsed:
        text = item.text
        if item.type != "DATE":
            tokens = text.split()
            named_indexes = [
                index for index, token in enumerate(tokens)
                if any(char.isupper() for char in token)
            ]
            if named_indexes and named_indexes[-1] < len(tokens) - 1:
                text = " ".join(tokens[:named_indexes[-1] + 1])
        repaired.append(f"{text} — {item.type}")
    return "\n".join(repaired)

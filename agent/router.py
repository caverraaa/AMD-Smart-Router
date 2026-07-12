"""Near-free prompt categorizer + per-category output constraints.

First regex match wins; anything unmatched is "unknown" (which always
routes to Fireworks — misroutes may only ever cost tokens, not accuracy).
"""
import re

CATEGORIES = (
    "sentiment", "ner", "summarisation", "code_debug",
    "code_gen", "math", "logic", "factual", "unknown",
)

_RULES = [
    ("sentiment", r"(classify|what is|determine|label)\b.{0,40}\bsentiment\b|\bsentiment of\b|classify\b.{0,40}\b(review|feedback)\b(?!.{0,60}\binto\b)"),
    ("code_gen", r"\bwrite\b.*\b(function|class|script|program|method)\b|\bimplement\b.*\b(function|class|method)\b"),
    ("ner", r"named entit|\bNER\b|extract\b.*\b(entit\w+|person|organi[sz]ation|location)"),
    ("summarisation", r"\bsummar(y|ise|ize|ising|izing)|\btl;?dr\b|\bcondense\b"),
    ("code_debug", r"(bug|debug|fix|fails|incorrect|error)\b.*\b(code|function|def |snippet)|(code|function|def |snippet).*\b(bug|debug|has a bug|fails|fix)"),
    ("math", r"\d+\s*%|\bpercent|\bcalculate\b|how (much|many)|\bremain\b|\bkm/h\b|\bmph\b|\brevenue\b|\bprofit\b|\bsum of\b"),
    ("logic", r"who (owns|has|is|does)|\bdeduce\b|logic puzzle|each own|\bseated\b|\bsits\b|\bconstraints? must\b"),
    ("factual", r"^what (is|are|was|were)\b|\bcapital of\b|\bexplain\b|\bdefine\b|\bdescribe\b|how does .* work"),
]
_COMPILED = [(cat, re.compile(pattern, re.IGNORECASE | re.DOTALL)) for cat, pattern in _RULES]

# Explanatory/code-writing intents never belong to the classification lanes
# (sentiment/ner): "explain how sentiment analysis works" is a factual ask,
# not a request to classify sentiment; a code_gen ask that happens to mention
# entities/locations shouldn't be misrouted to ner either.
_CLASSIFICATION_GUARD = re.compile(
    r"\b(explain|how does|write (a|an)?\s*(function|class|script|program)|implement)\b",
    re.IGNORECASE)

CONSTRAINTS = {
    "sentiment": "Label positive, negative, neutral, or mixed; add one brief reason. Use mixed for clear pros and cons.",
    "ner": "Entity — PERSON/ORG/LOCATION/DATE, one per line; include relative dates. No extra text.",
    "summarisation": "Match requested format and length exactly. No preamble.",
    "math": "Answer first; show only essential working.",
    "code_debug": "Output only the corrected code.",
    "code_gen": "Output only the code.",
    "logic": "Answer first; briefly verify every constraint.",
    "factual": "Direct answer; max two sentences.",
}


def categorize(prompt):
    skip_classification_lanes = bool(_CLASSIFICATION_GUARD.search(prompt))
    for cat, rx in _COMPILED:
        if skip_classification_lanes and cat in ("sentiment", "ner"):
            continue
        if rx.search(prompt):
            return cat
    return "unknown"


def build_user_message(prompt, category):
    constraint = CONSTRAINTS.get(category)
    return f"{prompt}\n\n{constraint}" if constraint else prompt

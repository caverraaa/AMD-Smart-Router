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
    ("sentiment", r"\bsentiment\b|classify\b.*\b(review|feedback|tone)"),
    ("ner", r"named entit|\bNER\b|extract\b.*\b(entit\w+|person|organi[sz]ation|location)"),
    ("summarisation", r"\bsummar(y|ise|ize|ising|izing)|\btl;?dr\b|\bcondense\b"),
    ("code_debug", r"(bug|debug|fix|fails|incorrect|error)\b.*\b(code|function|def |snippet)|(code|function|def |snippet).*\b(bug|debug|has a bug|fails|fix)"),
    ("code_gen", r"\bwrite\b.*\b(function|class|script|program|method)\b|\bimplement\b.*\b(function|class|method)\b"),
    ("math", r"\d+\s*%|\bpercent|\bcalculate\b|how (much|many)|\bremain\b|\bkm/h\b|\bmph\b|\brevenue\b|\bprofit\b|\bsum of\b"),
    ("logic", r"who (owns|has|is|does)|\bdeduce\b|logic puzzle|each own|\bseated\b|\bsits\b|\bconstraints? must\b"),
    ("factual", r"^what (is|are|was|were)\b|\bcapital of\b|\bexplain\b|\bdefine\b|\bdescribe\b|how does .* work"),
]
_COMPILED = [(cat, re.compile(pattern, re.IGNORECASE | re.DOTALL)) for cat, pattern in _RULES]

CONSTRAINTS = {
    "sentiment": "Answer with the sentiment label (positive, negative, neutral, or mixed) plus a one-sentence justification. If the text contains both clearly positive and clearly negative aspects, label it mixed.",
    "ner": "List each entity with its type (PERSON, ORG, LOCATION, DATE), one per line, including relative dates such as 'next month'. No extra text.",
    "summarisation": "Obey the stated format and length exactly. No preamble.",
    "math": "Give the final answer first, then at most two sentences of working.",
    "code_debug": "Output only the corrected code.",
    "code_gen": "Output only the code.",
    "logic": "State the answer first, then verify each constraint in one short line each.",
    "factual": "Answer in at most two sentences.",
}


def categorize(prompt):
    for cat, rx in _COMPILED:
        if rx.search(prompt):
            return cat
    return "unknown"


def build_user_message(prompt, category):
    constraint = CONSTRAINTS.get(category)
    return f"{prompt}\n\n{constraint}" if constraint else prompt

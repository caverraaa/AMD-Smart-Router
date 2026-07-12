"""Small deterministic tools for answers that can be proved locally.

The functions in this module are deliberately narrow.  Unsupported or
ambiguous input returns an empty string, which keeps the cloud fallback in
control of accuracy.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass


_COUNT_WORDS = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
}
_SAFE_ITEM_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9]*(?:[ '\-][A-Za-z0-9]+)*$"
)
_INTRO_RE = re.compile(
    r"^(?P<count>two|three|four|five|six|seven|eight|[2-8])\s+"
    r"(?P<group>[^,:.!?]{1,40}),\s*"
    r"(?P<actors>.+?),\s*each\s+(?P<relation>[^:.!?]{1,160}):\s*"
    r"(?P<values>[^.!?]{1,240})\.\s*(?P<body>.+)$",
    re.IGNORECASE | re.DOTALL,
)
_SAFE_INTRO_RELATION_RE = re.compile(
    r"^(?P<verb>[A-Za-z]+)(?:\s+in)?\s+(?:a|an)\s+"
    r"(?:[A-Za-z]+\s+of\s+(?:a|an)\s+)?"
    r"(?:different|distinct|unique)\s+"
    r"[A-Za-z]+(?:\s+of\s+[A-Za-z]+)?$",
    re.IGNORECASE,
)
_UNSAFE_UNIQUENESS_QUALIFIER_RE = re.compile(
    r"\b(?:not|no|never|without|possibly|perhaps|maybe|may|might|"
    r"necessarily|required|optional|if|unless|when|only|except|provided|"
    r"assuming|depending|sometimes|or|either|neither|nor)\b",
    re.IGNORECASE,
)
_QUERY_RE = re.compile(r"^Who\s+(?P<predicate>[^.!?]{1,120})\?$", re.IGNORECASE)
_NEGATIVE_RELATION_RE = re.compile(
    r"^(?:does|do|did)\s+not\s+(?P<relation>.+)$", re.IGNORECASE
)

# One reviewed generated-task paraphrase uses "specialize in a dish" in the
# setup and "make the dish" in its constraints.  All other predicate changes
# fail closed instead of assuming that two unrelated relations are equivalent.
_INTRO_QUERY_EQUIVALENCES = frozenset({("specialize", "make")})


@dataclass(frozen=True)
class AssignmentPuzzle:
    actors: tuple[str, ...]
    values: tuple[str, ...]
    constraints: tuple[tuple[int, int, bool], ...]
    target_value: int
    query_predicate: str


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def _strip_merged_wrapper(user_text: str) -> str:
    text = (user_text or "").strip()
    prefix = "Answer in English. "
    if text.startswith(prefix):
        text = text[len(prefix):]
    # main.py appends the category constraint after a blank line.  Rejecting
    # arbitrary suffixes inside the task itself remains important, so peel at
    # most this exact wrapper.
    suffix = "\n\nAnswer first; briefly verify every constraint."
    if text.endswith(suffix):
        text = text[:-len(suffix)]
    return text.strip()


def _split_coordinated_list(text: str) -> tuple[str, ...]:
    normalized = re.sub(
        r",?\s+(?:and|or)\s+", ",", text.strip(), count=1,
        flags=re.IGNORECASE,
    )
    items = tuple(item.strip() for item in normalized.split(","))
    if not items or any(not item or not _SAFE_ITEM_RE.fullmatch(item)
                        for item in items):
        return ()
    if len({_normalized(item) for item in items}) != len(items):
        return ()
    return items


def _count_value(raw: str) -> int | None:
    lowered = raw.casefold()
    return int(lowered) if lowered.isdigit() else _COUNT_WORDS.get(lowered)


def _split_trailing_value(text: str, values: tuple[str, ...]):
    for value_index, value in sorted(
            enumerate(values), key=lambda item: len(item[1]), reverse=True):
        match = re.search(
            rf"(?:^|\s)(?:the\s+)?{re.escape(value)}$",
            text.strip(), re.IGNORECASE,
        )
        if match:
            relation = text.strip()[:match.start()].strip()
            return relation, value_index
    return None


def _verb_key(relation: str) -> str | None:
    words = re.findall(r"[A-Za-z]+", relation.casefold())
    words = [word for word in words if word not in ("a", "an", "the")]
    if len(words) != 1:
        return None
    verb = words[0]
    irregular = {"has": "have", "had": "have"}
    if verb in irregular:
        return irregular[verb]
    if verb.endswith("ied") and len(verb) > 4:
        return verb[:-3] + "y"
    if verb.endswith("ies") and len(verb) > 4:
        return verb[:-3] + "y"
    if verb.endswith("ed") and len(verb) > 3:
        return verb[:-2]
    if verb.endswith("s") and not verb.endswith("ss") and len(verb) > 2:
        return verb[:-1]
    return verb


def _actor_and_tail(sentence: str, actors: tuple[str, ...]):
    for actor_index, actor in sorted(
            enumerate(actors), key=lambda item: len(item[1]), reverse=True):
        match = re.fullmatch(
            rf"{re.escape(actor)}\s+(?P<tail>.+)", sentence,
            re.IGNORECASE,
        )
        if match:
            return actor_index, match.group("tail").strip()
    return None


def parse_assignment_logic(user_text: str) -> AssignmentPuzzle | None:
    """Parse a small, explicit all-different assignment puzzle.

    Every sentence must match the supported grammar.  This avoids turning a
    broad logic category into an unsafe local lane.
    """
    task = _strip_merged_wrapper(user_text)
    if not task or len(task) > 2000:
        return None
    intro = _INTRO_RE.fullmatch(task)
    if not intro:
        return None
    intro_relation = intro.group("relation").strip()
    intro_relation_match = _SAFE_INTRO_RELATION_RE.fullmatch(intro_relation)
    if (
        not intro_relation_match
        or _UNSAFE_UNIQUENESS_QUALIFIER_RE.search(intro_relation)
    ):
        return None

    count = _count_value(intro.group("count"))
    actors = _split_coordinated_list(intro.group("actors"))
    values = _split_coordinated_list(intro.group("values"))
    if count is None or not 2 <= count <= 8:
        return None
    if len(actors) != count or len(values) != count:
        return None

    body = intro.group("body").strip()
    query_match = re.search(r"Who\s+[^.!?]{1,120}\?\s*$", body, re.IGNORECASE)
    if not query_match:
        return None
    query_text = query_match.group(0).strip()
    facts_text = body[:query_match.start()].strip()
    # A question must be the final sentence.  Conditional/disjunctive or
    # prompt-injection suffixes are therefore rejected instead of ignored.
    if not facts_text or "?" in facts_text or "!" in facts_text:
        return None
    query = _QUERY_RE.fullmatch(query_text)
    if not query:
        return None
    query_predicate = query.group("predicate").strip()
    query_parts = _split_trailing_value(query_predicate, values)
    if not query_parts:
        return None
    query_relation, target_value = query_parts
    query_verb = _verb_key(query_relation)
    if query_verb is None:
        return None
    intro_verb = _verb_key(intro_relation_match.group("verb"))
    if (
        intro_verb is None
        or (
            intro_verb != query_verb
            and (intro_verb, query_verb) not in _INTRO_QUERY_EQUIVALENCES
        )
    ):
        return None

    fact_sentences = tuple(
        sentence.strip() for sentence in facts_text.split(".") if sentence.strip()
    )
    # Reject missing punctuation and empty/extra sentence fragments.
    if not fact_sentences or not facts_text.endswith("."):
        return None
    constraints = []
    has_positive = False
    has_negative = False
    for sentence in fact_sentences:
        actor_tail = _actor_and_tail(sentence, actors)
        if not actor_tail:
            return None
        actor_index, tail = actor_tail
        value_parts = _split_trailing_value(tail, values)
        if not value_parts:
            return None
        relation, value_index = value_parts
        negative = _NEGATIVE_RELATION_RE.fullmatch(relation)
        if negative:
            relation = negative.group("relation").strip()
            equals = False
            has_negative = True
        else:
            if re.search(r"\b(?:not|never|neither|nor|or|if)\b", relation,
                         re.IGNORECASE):
                return None
            equals = True
            has_positive = True
        if _verb_key(relation) != query_verb:
            return None
        constraints.append((actor_index, value_index, equals))

    if not has_positive or not has_negative:
        return None
    return AssignmentPuzzle(
        actors=actors,
        values=values,
        constraints=tuple(constraints),
        target_value=target_value,
        query_predicate=query_predicate,
    )


def solve_assignment_logic(user_text: str) -> str:
    """Return the proved answer, or ``""`` for unsupported/ambiguous input."""
    puzzle = parse_assignment_logic(user_text)
    if puzzle is None:
        return ""
    possible_actors = set()
    solution_count = 0
    for assignment in itertools.permutations(range(len(puzzle.values))):
        if any((assignment[actor] == value) != equals
               for actor, value, equals in puzzle.constraints):
            continue
        solution_count += 1
        possible_actors.update(
            actor for actor, value in enumerate(assignment)
            if value == puzzle.target_value
        )
        if len(possible_actors) > 1:
            return ""
    if solution_count == 0 or len(possible_actors) != 1:
        return ""
    actor = puzzle.actors[next(iter(possible_actors))]
    return f"{actor} {puzzle.query_predicate}."

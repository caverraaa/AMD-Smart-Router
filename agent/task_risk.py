"""Deterministic task-risk gate for conservative local execution.

Task category, task complexity, and answer verifiability are deliberately
separate signals.  Complexity describes the shape and cost of a prompt; it
does not make an otherwise unverifiable answer safe.  A short factual question
therefore stays in the cloud, while a supported assignment puzzle can run
locally because its answer is proved by a deterministic solver.

This module performs no model or API calls.  Experimental profiles are off by
default.  Passing an experimental profile name is an explicit assertion by
the caller that its local generator and validator passed the required gate.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

try:
    from agent.code_tools import solve_code_task
    from agent.local_summary import supports_lossless_summary
    from agent.local_tools import solve_assignment_logic
    from agent.local_validators import supports_local_ner_request
    from agent.math_tools import solve_math
except ImportError:  # executed with /app/agent on sys.path
    from code_tools import solve_code_task
    from local_summary import supports_lossless_summary
    from local_tools import solve_assignment_logic
    from local_validators import supports_local_ner_request
    from math_tools import solve_math


LANE_CLOUD = "cloud"
LANE_DETERMINISTIC_LOCAL = "deterministic_local"
LANE_VALIDATED_LOCAL = "validated_local"

VERIFIABILITY_PROVED = "proved"
VERIFIABILITY_VALIDATED = "validated"
VERIFIABILITY_UNVERIFIED = "unverified"

COMPLEXITY_LOW = "low"
COMPLEXITY_MEDIUM = "medium"
COMPLEXITY_HIGH = "high"
COMPLEXITY_UNKNOWN = "unknown"

PROFILE_LOGIC_ASSIGNMENT_V1 = "logic-assignment-v1"
PROFILE_NER_EXACT_SPAN_V1 = "ner-exact-span-v1"
PROFILE_SENTIMENT_LABEL_REASON_V1 = "sentiment-label-reason-v1"
PROFILE_SUMMARY_LOSSLESS_FUSION_V1 = "summary-lossless-fusion-v1"
PROFILE_MATH_INVENTORY_REMAINDER_V1 = "math-inventory-remainder-v1"
PROFILE_CODE_DEBUG_VERIFIED_V1 = "code-debug-verified-v1"
PROFILE_CODE_GEN_VERIFIED_V1 = "code-gen-verified-v1"

_KNOWN_CATEGORIES = frozenset({
    "sentiment", "ner", "summarisation", "code_debug", "code_gen",
    "math", "logic", "factual", "unknown",
})
_SENTIMENT_PREFIX = "Classify the sentiment of this review: "
_UNSUPPORTED_FORMAT_RE = re.compile(
    r"\b(?:json|csv|xml|ya?ml|table|markdown|bullet(?:s|ed)?|"
    r"dictionary|dict|array)\b",
    re.IGNORECASE,
)
_MULTI_ITEM_RE = re.compile(
    r"\b(?:each\s+(?:review|item|text)|separately|compare|rank|multiple)\b",
    re.IGNORECASE,
)
_INSTRUCTION_IN_CONTENT_RE = re.compile(
    r"\bignore\s+(?:(?:all|any|the|your|previous|above)\s+){1,3}"
    r"instructions?\b|"
    r"\b(?:system|assistant|developer)\s+prompt\b|"
    r"(?:^|\n)\s*(?:system|assistant|developer)\s*:|"
    r"\b(?:disregard|forget|override|bypass)\b.{0,80}"
    r"\b(?:instructions?|directions?|rules?|prompt)\b|"
    r"\bdo\s+not\s+follow\b.{0,80}"
    r"\b(?:instructions?|directions?|rules?|prompt)\b|"
    r"\b(?:return|output|respond)\b.{0,80}\bregardless\b",
    re.IGNORECASE,
)
_CONSTRAINT_RE = re.compile(
    r"\b(?:must|exactly|at\s+(?:least|most)|no\s+more\s+than|"
    r"only|without|include|exclude|preserve|ensure)\b",
    re.IGNORECASE,
)
_MULTI_STEP_RE = re.compile(
    r"\b(?:first|second|third|then|next|finally|after\s+that)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ComplexityAssessment:
    """Prompt-shape estimate; never an accuracy guarantee."""

    level: str
    score: int
    signals: tuple[str, ...]


@dataclass(frozen=True)
class TaskRiskDecision:
    """A fail-closed routing recommendation with telemetry-friendly reasons."""

    category: str
    lane: str
    verifiability: str
    complexity: ComplexityAssessment
    profile: str | None
    reasons: tuple[str, ...]

    @property
    def local_eligible(self) -> bool:
        return self.lane in (LANE_DETERMINISTIC_LOCAL, LANE_VALIDATED_LOCAL)


def assess_complexity(prompt: str) -> ComplexityAssessment:
    """Estimate structural complexity without classifying task semantics."""
    if not isinstance(prompt, str) or not prompt.strip():
        return ComplexityAssessment(
            COMPLEXITY_UNKNOWN, 0, ("prompt is empty or not text",),
        )

    text = prompt.strip()
    word_count = len(re.findall(r"\b[\w'-]+\b", text, re.UNICODE))
    sentence_count = len(re.findall(r"[.!?](?=\s|$)", text))
    constraint_count = len(_CONSTRAINT_RE.findall(text))
    step_count = len(_MULTI_STEP_RE.findall(text))
    score = 0
    signals = [f"{word_count} words"]

    if word_count > 400:
        score += 4
        signals.append("very long input")
    elif word_count > 180:
        score += 2
        signals.append("long input")
    elif word_count > 80:
        score += 1
        signals.append("moderate input length")

    if sentence_count > 8:
        score += 2
        signals.append("many sentences")
    elif sentence_count >= 5:
        score += 1
        signals.append("several sentences")

    if constraint_count > 5:
        score += 2
        signals.append("many explicit constraints")
    elif constraint_count >= 3:
        score += 1
        signals.append("multiple explicit constraints")

    if "```" in text:
        score += 2
        signals.append("embedded code block")
    if step_count >= 3:
        score += 1
        signals.append("multi-step request")

    if score == 0:
        level = COMPLEXITY_LOW
    elif score < 4:
        level = COMPLEXITY_MEDIUM
    else:
        level = COMPLEXITY_HIGH
    return ComplexityAssessment(level, score, tuple(signals))


def assess_task_risk(
    prompt: str,
    category: str,
    *,
    enabled_profiles: Iterable[str] = (),
) -> TaskRiskDecision:
    """Return a conservative local/cloud decision without executing a task.

    Existing reviewed lanes are recognized automatically.  Experimental
    summarisation remains cloud-bound unless its exact profile name is passed
    in ``enabled_profiles``.  Unknown inputs, unsupported shapes, and checker
    errors all fail closed to ``cloud``.
    """
    complexity = assess_complexity(prompt)
    normalized_category = (
        category.strip().lower() if isinstance(category, str) else "unknown"
    )
    if normalized_category not in _KNOWN_CATEGORIES:
        return _cloud(
            "unknown", complexity, "category has no reviewed local profile",
        )
    if not isinstance(prompt, str) or not prompt.strip():
        return _cloud(
            normalized_category, complexity, "prompt is empty or not text",
        )

    text = prompt.strip()
    if _INSTRUCTION_IN_CONTENT_RE.search(text):
        return _cloud(
            normalized_category,
            complexity,
            "instruction-like content is outside reviewed local profiles",
        )

    if normalized_category == "logic":
        try:
            answer = solve_assignment_logic(text)
        except Exception:  # checker failure must only increase token cost
            answer = ""
        if answer:
            return TaskRiskDecision(
                normalized_category,
                LANE_DETERMINISTIC_LOCAL,
                VERIFIABILITY_PROVED,
                complexity,
                PROFILE_LOGIC_ASSIGNMENT_V1,
                ("unique answer proved by deterministic assignment solver",),
            )
        return _cloud(
            normalized_category,
            complexity,
            "logic shape is unsupported, ambiguous, or contradictory",
        )

    if normalized_category == "ner":
        try:
            supported = supports_local_ner_request(text)
        except Exception:  # checker failure must only increase token cost
            supported = False
        if not supported:
            return _cloud(
                normalized_category,
                complexity,
                "NER request is outside the exact-span validator profile",
            )
        if complexity.level == COMPLEXITY_HIGH:
            return _cloud(
                normalized_category,
                complexity,
                "NER task exceeds the reviewed local complexity envelope",
            )
        return TaskRiskDecision(
            normalized_category,
            LANE_VALIDATED_LOCAL,
            VERIFIABILITY_VALIDATED,
            complexity,
            PROFILE_NER_EXACT_SPAN_V1,
            ("answer is guarded by exact-span and output-schema validation",),
        )

    if normalized_category == "sentiment":
        if not _supports_sentiment_profile(text):
            return _cloud(
                normalized_category,
                complexity,
                "sentiment request is outside the reviewed label-and-reason profile",
            )
        if complexity.level == COMPLEXITY_HIGH:
            return _cloud(
                normalized_category,
                complexity,
                "sentiment task exceeds the reviewed local complexity envelope",
            )
        return TaskRiskDecision(
            normalized_category,
            LANE_VALIDATED_LOCAL,
            VERIFIABILITY_VALIDATED,
            complexity,
            PROFILE_SENTIMENT_LABEL_REASON_V1,
            ("task matches the benchmark-validated label-and-reason profile",),
        )

    if normalized_category == "summarisation":
        profiles = _profile_set(enabled_profiles)
        if PROFILE_SUMMARY_LOSSLESS_FUSION_V1 not in profiles:
            return _cloud(
                normalized_category,
                complexity,
                "summarisation profile is disabled pending accuracy validation",
            )
        if not supports_lossless_summary(text):
            return _cloud(
                normalized_category,
                complexity,
                "summarisation request is outside the enabled profile",
            )
        if complexity.level == COMPLEXITY_HIGH:
            return _cloud(
                normalized_category,
                complexity,
                "summarisation task exceeds the enabled profile complexity envelope",
            )
        return TaskRiskDecision(
            normalized_category,
            LANE_DETERMINISTIC_LOCAL,
            VERIFIABILITY_VALIDATED,
            complexity,
            PROFILE_SUMMARY_LOSSLESS_FUSION_V1,
            ("lossless fusion is structurally proved and rubric-validated",),
        )

    if normalized_category == "math":
        try:
            answer = solve_math(text)
        except Exception:  # exact solver failure must only increase API cost
            answer = ""
        if answer:
            return TaskRiskDecision(
                normalized_category,
                LANE_DETERMINISTIC_LOCAL,
                VERIFIABILITY_PROVED,
                complexity,
                PROFILE_MATH_INVENTORY_REMAINDER_V1,
                ("answer recomputed with exact rational inventory arithmetic",),
            )
        return _cloud(
            normalized_category,
            complexity,
            "no reviewed deterministic math profile matched",
        )

    if normalized_category in ("code_debug", "code_gen"):
        try:
            source = solve_code_task(text, normalized_category)
        except Exception:  # validator failure must only increase API cost
            source = ""
        if source:
            profile = (
                PROFILE_CODE_DEBUG_VERIFIED_V1
                if normalized_category == "code_debug"
                else PROFILE_CODE_GEN_VERIFIED_V1
            )
            return TaskRiskDecision(
                normalized_category,
                LANE_DETERMINISTIC_LOCAL,
                VERIFIABILITY_VALIDATED,
                complexity,
                profile,
                ("canonical code passed AST allowlist and property tests",),
            )
        return _cloud(
            normalized_category,
            complexity,
            "no execution-validated deterministic code profile matched",
        )

    cloud_reason = {
        "factual": "factual correctness is not locally verifiable",
        "unknown": "task intent has no reviewed local profile",
    }.get(normalized_category, "category has no reviewed local profile")
    return _cloud(normalized_category, complexity, cloud_reason)


def _cloud(
    category: str,
    complexity: ComplexityAssessment,
    reason: str,
) -> TaskRiskDecision:
    return TaskRiskDecision(
        category,
        LANE_CLOUD,
        VERIFIABILITY_UNVERIFIED,
        complexity,
        None,
        (reason,),
    )


def _supports_sentiment_profile(text: str) -> bool:
    if not text.startswith(_SENTIMENT_PREFIX):
        return False
    source = text[len(_SENTIMENT_PREFIX):]
    word_count = len(re.findall(r"\b[\w'-]+\b", source, re.UNICODE))
    return bool(
        source == source.strip()
        and 3 <= word_count <= 40
        and source.endswith(".")
        and source.count(".") == 1
        and not any(marker in source for marker in ("\n", "\r", "?", "!", ";", ":"))
        and not _UNSUPPORTED_FORMAT_RE.search(text)
        and not _MULTI_ITEM_RE.search(text)
    )


def _profile_set(profiles: Iterable[str]) -> frozenset[str]:
    if isinstance(profiles, str):
        return frozenset((profiles,))
    try:
        return frozenset(
            profile for profile in profiles if isinstance(profile, str)
        )
    except Exception:  # malformed/lazy profile inputs must not enable a lane
        return frozenset()

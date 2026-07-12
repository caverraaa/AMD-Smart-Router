"""Conservative batching helpers for cloud-bound factual tasks in groups 3-4.

The scoring path keeps batching disabled unless ``ENABLE_BATCHING=1``.  The
parser intentionally fails closed: structural ambiguity (malformed JSON,
duplicate keys, or unexpected task IDs) causes the entire batch to fall back
to the existing one-task-per-call path.  Missing or invalid values only fall
back for the affected tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Callable, Iterable, Mapping, Sequence


BATCHABLE_CATEGORIES = frozenset({"factual"})
MIN_BATCH_SIZE = 3
MAX_BATCH_SIZE = 4
MAX_BATCH_TOKENS = 4096

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


class _JSONObject(list):
    """Marker that distinguishes a decoded JSON object from a JSON array."""


@dataclass(frozen=True)
class ParsedBatch:
    """Validated answers and task IDs that must use the individual fallback."""

    answers: dict[str, str]
    fallback_ids: tuple[str, ...]
    error: str | None = None


def batching_enabled(value: str | None = None) -> bool:
    """Return true only for an explicit, conventional truthy flag value."""

    if value is None:
        value = os.environ.get("ENABLE_BATCHING", "")
    return value.strip().lower() in _TRUE_VALUES


def plan_batches(
    tasks: Sequence[dict],
    routing: Mapping[str, str] | None = None,
) -> tuple[list[list[dict]], list[dict]]:
    """Partition tasks into safe same-category batches and individual work.

    Only tasks already bound for Fireworks are eligible.  Local-routed tasks
    remain individual so a local failure can use the established cloud
    fallback without first spending a batch call.
    """

    routing = routing or {}
    eligible: dict[str, list[dict]] = {category: [] for category in BATCHABLE_CATEGORIES}
    singles: list[dict] = []
    for task in tasks:
        category = task.get("category", "unknown")
        if category in BATCHABLE_CATEGORIES and routing.get(category) != "local":
            eligible[category].append(task)
        else:
            singles.append(task)

    batches: list[list[dict]] = []
    for category in sorted(BATCHABLE_CATEGORIES):
        candidates = eligible[category]
        count = len(candidates)
        options = []
        for triples in range(count // MIN_BATCH_SIZE + 1):
            for quads in range(count // MAX_BATCH_SIZE + 1):
                covered = triples * MIN_BATCH_SIZE + quads * MAX_BATCH_SIZE
                if covered <= count:
                    options.append((count - covered, triples + quads,
                                    -quads, triples, quads))
        _, _, _, triples, quads = min(options)
        offset = 0
        for size in [MAX_BATCH_SIZE] * quads + [MIN_BATCH_SIZE] * triples:
            batches.append(candidates[offset:offset + size])
            offset += size
        singles.extend(candidates[offset:])
    return batches, singles


def build_batch_messages(
    tasks: Sequence[dict],
    system_prompt: str,
    user_message_builder: Callable[[str, str], str],
) -> list[dict[str, str]]:
    """Build an injection-resistant JSON envelope for one factual batch."""

    payload = {
        "tasks": [
            {
                "task_id": str(task["task_id"]),
                "prompt": user_message_builder(
                    task["prompt"], task.get("category", "unknown")
                ),
            }
            for task in tasks
        ]
    }
    batch_instruction = (
        "Return only one JSON object mapping each supplied task_id to its "
        "complete English answer string. Include every supplied task_id "
        "exactly once and no other keys. Answer each task independently."
    )
    return [
        {"role": "system", "content": f"{system_prompt} {batch_instruction}"},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        },
    ]


def batch_token_cap(per_task_cap: int, task_count: int) -> int:
    """Scale the existing category cap while respecting the API ceiling."""

    if per_task_cap <= 0 or task_count <= 0:
        raise ValueError("per_task_cap and task_count must be positive")
    return min(MAX_BATCH_TOKENS, per_task_cap * task_count)


def parse_batch_answers(content: str, expected_ids: Iterable[str]) -> ParsedBatch:
    """Strictly validate a JSON object keyed by exactly the requested IDs.

    Missing IDs and non-string/blank values are isolated to per-task fallback.
    Duplicate or unexpected keys make the whole object untrustworthy and
    therefore fall back every task.
    """

    expected = tuple(str(task_id) for task_id in expected_ids)
    expected_set = set(expected)
    if len(expected_set) != len(expected):
        return ParsedBatch({}, expected, "duplicate expected task_id")
    if not isinstance(content, str) or not content.strip():
        return ParsedBatch({}, expected, "empty batch content")

    try:
        decoded = json.loads(content, object_pairs_hook=_JSONObject)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return ParsedBatch({}, expected, f"malformed batch JSON: {exc}")
    if not isinstance(decoded, _JSONObject):
        return ParsedBatch({}, expected, "batch response root is not a JSON object")

    seen: set[str] = set()
    duplicate: set[str] = set()
    unexpected: set[str] = set()
    values: dict[str, object] = {}
    for raw_key, value in decoded:
        key = str(raw_key)
        if key in seen:
            duplicate.add(key)
        seen.add(key)
        values[key] = value
        if key not in expected_set:
            unexpected.add(key)

    if duplicate:
        names = ", ".join(sorted(duplicate))
        return ParsedBatch({}, expected, f"duplicate task_id(s): {names}")
    if unexpected:
        names = ", ".join(sorted(unexpected))
        return ParsedBatch({}, expected, f"unexpected task_id(s): {names}")

    answers: dict[str, str] = {}
    fallback: list[str] = []
    for task_id in expected:
        value = values.get(task_id)
        if isinstance(value, str) and value.strip():
            answers[task_id] = value.strip()
        else:
            fallback.append(task_id)
    error = "missing or invalid answer value" if fallback else None
    return ParsedBatch(answers, tuple(fallback), error)

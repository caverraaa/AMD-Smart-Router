"""Offline benchmark for the local factual and logic lanes.

This script never creates a network client.  It loads the bundled GGUF and
checks local answers against the human-reviewed ``expected_intent`` fields.
Run it in a fresh Python process for each independent trial.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.local_model import LocalModel  # noqa: E402
from agent.router import build_user_message  # noqa: E402


def _words(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", text.casefold()))


def _contains_words(answer: str, expected: str) -> bool:
    haystack = _words(answer)
    needle = _words(expected)
    if not needle:
        return False
    width = len(needle)
    return any(haystack[index:index + width] == needle
               for index in range(len(haystack) - width + 1))


def strict_match(category: str, answer: str, expected_intent: str) -> bool:
    """Deterministic benchmark check for the reviewed factual/logic rubrics."""
    if not answer.strip():
        return False
    if category == "factual":
        claims = [claim.strip(" .") for claim in expected_intent.split(";")]
        return all(_contains_words(answer, claim) for claim in claims if claim)
    if category == "logic":
        return _contains_words(answer, expected_intent.strip(" ."))
    raise ValueError(f"unsupported benchmark category: {category}")


def _load_tasks(path: Path, categories: set[str]) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        tasks = json.load(handle)
    return [task for task in tasks if task.get("category") in categories]


def _answer(local: LocalModel, task: dict) -> dict:
    prompt = "Answer in English. " + build_user_message(
        task["prompt"], task["category"])
    started = time.monotonic()
    answer = local.answer(prompt, category=task["category"])
    elapsed = time.monotonic() - started
    return {
        "task_id": task["task_id"],
        "category": task["category"],
        "answer": answer,
        "expected_intent": task["expected_intent"],
        "passed": strict_match(task["category"], answer,
                               task["expected_intent"]),
        "elapsed_seconds": round(elapsed, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path,
                        default=ROOT / "eval" / "golden_tasks.json")
    parser.add_argument("--model", type=Path,
                        default=ROOT / "models" / "gemma-2-2b-it-Q4_K_M.gguf")
    parser.add_argument("--categories", nargs="+", choices=("factual", "logic"),
                        default=("factual", "logic"))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    tasks = _load_tasks(args.golden, set(args.categories))
    local = LocalModel(path=str(args.model))
    started = time.monotonic()
    if args.workers == 1:
        rows = [_answer(local, task) for task in tasks]
    else:
        rows = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_answer, local, task) for task in tasks]
            for future in as_completed(futures):
                rows.append(future.result())
        order = {task["task_id"]: index for index, task in enumerate(tasks)}
        rows.sort(key=lambda row: order[row["task_id"]])

    by_category = {}
    for category in args.categories:
        category_rows = [row for row in rows if row["category"] == category]
        by_category[category] = {
            "passed": sum(row["passed"] for row in category_rows),
            "total": len(category_rows),
            "worst_seconds": max(
                (row["elapsed_seconds"] for row in category_rows), default=0),
        }
    report = {
        "workers": args.workers,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "summary": by_category,
        "rows": rows,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all(value["passed"] == value["total"]
                    for value in by_category.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

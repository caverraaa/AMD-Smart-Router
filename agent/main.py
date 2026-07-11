"""Phase 1 baseline agent: every task goes to the cheapest allowed Fireworks model.

Spec: docs/superpowers/specs/2026-07-11-phase1-baseline-design.md
"""
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

SYSTEM_PROMPT = "Answer in English. Be correct and complete, but concise."
MAX_TOKENS = 1024
FIRST_TIMEOUT_SECONDS = 20.0
RETRY_TIMEOUT_SECONDS = 25.0
RETRY_BACKOFF_SECONDS = 2.0
SOFT_BUDGET_SECONDS = 540.0  # 9 min of the 10-min limit; the rest is startup/write/exit margin
MAX_WORKERS = 4

_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)b(?![a-z0-9])")


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def parse_model_size(model_id):
    """Billions of params parsed from the ID string; inf when the ID reveals nothing."""
    m = _SIZE_RE.search(model_id.lower())
    return float(m.group(1)) if m else math.inf


def pick_cheapest_model(allowed_models):
    override = os.environ.get("CHEAP_MODEL")  # local-dev knob; the harness never sets it
    if override in allowed_models:
        return override
    return min(allowed_models, key=parse_model_size)  # ties keep list order


def load_config():
    required = ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        log(f"FATAL: missing required environment variables: {', '.join(missing)}")
        raise SystemExit(1)
    models = [m.strip() for m in os.environ["ALLOWED_MODELS"].split(",") if m.strip()]
    if not models:
        log("FATAL: ALLOWED_MODELS contains no model IDs")
        raise SystemExit(1)
    return {
        "api_key": os.environ["FIREWORKS_API_KEY"],
        "base_url": os.environ["FIREWORKS_BASE_URL"],
        "allowed_models": models,
    }


def load_tasks(path):
    """Returns (usable task_ids in input order, answerable task dicts).

    A task with a usable task_id but no usable prompt stays in task_ids —
    it must still appear in results (answer: "") — but is not answerable.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("tasks.json top level must be a JSON list")
    task_ids, answerable = [], []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict) or not isinstance(entry.get("task_id"), str) or not entry["task_id"]:
            log(f"WARN: entry {i} has no usable task_id, skipping")
            continue
        task_ids.append(entry["task_id"])
        if isinstance(entry.get("prompt"), str) and entry["prompt"]:
            answerable.append({"task_id": entry["task_id"], "prompt": entry["prompt"]})
        else:
            log(f"WARN: task {entry['task_id']} has no usable prompt; will emit empty answer")
    return task_ids, answerable


def write_snapshot(task_ids, answers, path):
    """Atomically replace `path` with the full, schema-valid results list.

    Called after every completed task, so the file on disk is always
    complete valid JSON no matter when the container dies.
    """
    results = [
        {"task_id": str(tid), "answer": str(answers.get(tid) or "")}
        for tid in task_ids
    ]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    os.replace(tmp, path)


def answer_task(client, model, task, deadline):
    """One Fireworks call with one retry. Never raises; failures return answer ''."""
    result = {"task_id": task["task_id"], "answer": "",
              "prompt_tokens": 0, "completion_tokens": 0, "error": None}
    for attempt, timeout in enumerate((FIRST_TIMEOUT_SECONDS, RETRY_TIMEOUT_SECONDS)):
        if time.monotonic() >= deadline:
            result["error"] = "soft budget exhausted before dispatch"
            return result
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": task["prompt"]},
                ],
                max_tokens=MAX_TOKENS,
                timeout=timeout,
            )
            result["answer"] = (resp.choices[0].message.content or "").strip()
            usage = getattr(resp, "usage", None)
            if usage is not None:
                result["prompt_tokens"] = usage.prompt_tokens or 0
                result["completion_tokens"] = usage.completion_tokens or 0
            result["error"] = None
            return result
        except Exception as exc:  # noqa: BLE001 — one task must never kill the run
            result["error"] = f"{type(exc).__name__}: {exc}"
            if attempt == 0:
                time.sleep(RETRY_BACKOFF_SECONDS)
    return result

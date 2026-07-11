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

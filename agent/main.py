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

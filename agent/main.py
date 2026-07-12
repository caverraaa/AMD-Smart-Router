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

try:
    from agent.batching import (
        BATCHABLE_CATEGORIES,
        MAX_BATCH_SIZE,
        MIN_BATCH_SIZE,
        batch_token_cap,
        batching_enabled,
        build_batch_messages,
        parse_batch_answers,
        plan_batches,
    )
    from agent.router import build_user_message, categorize
except ImportError:  # executed as a script (python /app/agent/main.py)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agent.batching import (
        BATCHABLE_CATEGORIES,
        MAX_BATCH_SIZE,
        MIN_BATCH_SIZE,
        batch_token_cap,
        batching_enabled,
        build_batch_messages,
        parse_batch_answers,
        plan_batches,
    )
    from agent.router import build_user_message, categorize

SYSTEM_PROMPT = "Answer in English. Be accurate and brief."
# Caps include hidden reasoning tokens. Short-answer categories get tight
# budgets; code and unknown tasks retain room for complete implementations.
CATEGORY_TOKEN_CAPS = {
    "sentiment": (512, 2048),
    "ner": (512, 2048),
    "factual": (512, 2048),
    "summarisation": (768, 2048),
    "math": (768, 2048),
    "logic": (2048, 4096),
    "code_debug": (2048, 4096),
    "code_gen": (2048, 4096),
    "unknown": (2048, 4096),
}
# Backwards-compatible aliases: unknown is the safe default category.
MAX_TOKENS, RETRY_MAX_TOKENS = CATEGORY_TOKEN_CAPS["unknown"]
FIRST_TIMEOUT_SECONDS = 20.0
RETRY_TIMEOUT_SECONDS = 25.0
RETRY_BACKOFF_SECONDS = 2.0
SOFT_BUDGET_SECONDS = 540.0  # 9 min of the 10-min limit; the rest is startup/write/exit margin
MAX_WORKERS = 4
FULL_EFFORT_CATEGORIES = ("logic",)  # low reasoning effort measurably breaks deduction puzzles
# Offline-validated preference order. Selection always intersects this policy
# with ALLOWED_MODELS. gpt-oss-120b + low effort passed practice 8/8.
OFFLINE_MODEL_POLICY = (
    ("accounts/fireworks/models/gpt-oss-120b", {"reasoning_effort": "low"}),
)
# Worst-case lock wait + generation reserve for the local lane. Mirrors
# agent.local_model.LOCAL_WORST_SECONDS — that copy bounds LocalModel's own
# internal budget; this copy gates whether the lane is attempted at all.
LOCAL_WORST_SECONDS = 30.0

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


def _empty_usage_stats():
    return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}


def pick_working_model(client, allowed_models, probe_stats=None):
    """Select deterministically without making a paid API call.

    ``client`` and ``probe_stats`` remain for Phase 2 call-site compatibility.
    The result is always allowlisted, and probe telemetry remains zero.
    """
    del client, probe_stats
    override = os.environ.get("CHEAP_MODEL")
    if override in allowed_models:
        selected = override
    else:
        selected = next(
            (model for model, _ in OFFLINE_MODEL_POLICY if model in allowed_models),
            None,
        ) or pick_cheapest_model(allowed_models)
    extra = next(
        (dict(body) for model, body in OFFLINE_MODEL_POLICY if model == selected),
        None,
    )
    return selected, extra


def token_caps_for(category):
    """Return ``(first, retry)`` caps; unrecognized categories use defaults."""
    return CATEGORY_TOKEN_CAPS.get(category, CATEGORY_TOKEN_CAPS["unknown"])


def _is_retryable_exception(exc):
    """Conservative retry classification for OpenAI-compatible failures."""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is not None:
        return status in (408, 409, 425, 429) or status >= 500
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    error_text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in error_text for marker in (
        "timeout", "connection", "rate limit", "ratelimit", "temporar",
        "internal server", "service unavailable",
    ))


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


_ID_KEYS = ("task_id", "id", "taskId")
_PROMPT_KEYS = ("prompt", "question", "input", "task", "text", "query", "instruction")
_WRAPPER_KEYS = ("tasks", "data", "items")


def load_tasks(path):
    """Returns (usable task_ids in input order, answerable task dicts).

    Deliberately tolerant: the eval set's exact schema is unseen, and a task
    silently dropped here scores zero with no trace. Integer ids are coerced,
    the prompt may live under several names, and a {"tasks": [...]} wrapper
    is unwrapped. A task with a usable id but no usable prompt stays in
    task_ids — it must still appear in results (answer: "").
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        for key in _WRAPPER_KEYS:
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    if not isinstance(raw, list):
        raise ValueError("tasks.json top level must be a JSON list")
    task_ids, answerable, seen = [], [], set()
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            log(f"WARN: entry {i} is not an object, skipping")
            continue
        tid = next((entry[k] for k in _ID_KEYS if entry.get(k) not in (None, "")), None)
        if tid is None:
            log(f"WARN: entry {i} has no usable task_id, skipping")
            continue
        tid = str(tid)
        if tid in seen:
            log(f"WARN: duplicate task_id {tid}, keeping first occurrence")
            continue
        seen.add(tid)
        task_ids.append(tid)
        prompt = next(
            (entry[k] for k in _PROMPT_KEYS if isinstance(entry.get(k), str) and entry[k].strip()),
            None,
        )
        if prompt is None:  # last resort: any non-id string field
            prompt = next(
                (v for k, v in entry.items()
                 if k not in _ID_KEYS and isinstance(v, str) and v.strip()),
                None,
            )
        if prompt:
            answerable.append({"task_id": tid, "prompt": prompt})
        else:
            log(f"WARN: task {tid} has no usable prompt; will emit empty answer")
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


def load_routing_table(path):
    """category -> "local"|"fireworks". Missing/invalid file means all-cloud."""
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return {k: v for k, v in raw.items() if v in ("local", "fireworks")}
    except Exception:  # noqa: BLE001
        return {}


def answer_task(client, model, task, deadline, extra_body=None, local=None, routing=None):
    """One Fireworks call with one retry. Never raises; failures return answer ''."""
    result = {"task_id": task["task_id"], "answer": "",
              "prompt_tokens": 0, "completion_tokens": 0, "error": None,
              "category": task.get("category", "unknown"), "lane": "fireworks",
              "fireworks_calls": 0, "retry_calls": 0,
              "retry_prompt_tokens": 0, "retry_completion_tokens": 0}
    routing = routing or {}
    if (local is not None and routing.get(result["category"]) == "local"
            and time.monotonic() + LOCAL_WORST_SECONDS < deadline):
        try:
            local_prompt = (
                "Answer in English. "
                + build_user_message(task["prompt"], result["category"])
            )
            if hasattr(local, "answer"):
                text = local.answer(
                    local_prompt, category=result["category"], deadline=deadline)
            else:  # compatibility with minimal test/dev local adapters
                text = local.generate(local_prompt, deadline=deadline)
            if text:
                result["answer"] = text
                result["lane"] = "local"
                return result  # zero Fireworks tokens
        except Exception as exc:  # noqa: BLE001 — local failure falls back to cloud
            log(f"WARN: local lane failed for {task['task_id']}: {type(exc).__name__}: {exc}")
    effective_extra = None if result["category"] in FULL_EFFORT_CATEGORIES else extra_body
    first_cap, retry_cap = token_caps_for(result["category"])
    attempts = ((FIRST_TIMEOUT_SECONDS, first_cap), (RETRY_TIMEOUT_SECONDS, retry_cap))
    for attempt, (timeout, max_tokens) in enumerate(attempts):
        if time.monotonic() >= deadline:
            result["error"] = "soft budget exhausted before dispatch"
            return result
        try:
            result["fireworks_calls"] += 1
            if attempt:
                result["retry_calls"] += 1
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_message(task["prompt"], result["category"])},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
                timeout=timeout,
                **({"extra_body": effective_extra} if effective_extra else {}),
            )
            usage = getattr(resp, "usage", None)
            if usage is not None:  # tokens are billed per attempt — accumulate
                attempt_prompt_tokens = usage.prompt_tokens or 0
                attempt_completion_tokens = usage.completion_tokens or 0
                result["prompt_tokens"] += attempt_prompt_tokens
                result["completion_tokens"] += attempt_completion_tokens
                if attempt:
                    result["retry_prompt_tokens"] += attempt_prompt_tokens
                    result["retry_completion_tokens"] += attempt_completion_tokens
            choice = resp.choices[0]
            content = (choice.message.content or "").strip()
            finish = getattr(choice, "finish_reason", None)
            if finish == "length":
                # A visible prefix is still incomplete (especially for code).
                result["error"] = "truncated content (finish_reason=length)"
            elif not content:
                # reasoning models can burn the whole cap on hidden reasoning
                # and return empty content; retry with more headroom
                result["error"] = f"empty content (finish_reason={finish})"
            else:
                result["answer"] = content
                result["error"] = None
                return result
        except Exception as exc:  # noqa: BLE001 — one task must never kill the run
            result["error"] = f"{type(exc).__name__}: {exc}"
            if not _is_retryable_exception(exc):
                return result
        if attempt == 0:
            time.sleep(RETRY_BACKOFF_SECONDS)
    return result


def answer_batch(client, model, tasks, deadline, extra_body=None):
    """Attempt one strict same-category cloud batch, without a batch retry.

    Any task not validated in the response is returned in ``fallback_tasks``
    for the existing individual ``answer_task`` path.  This function never
    raises, and all usage from the batch attempt is retained even when parsing
    fails and every task falls back.
    """
    tasks = list(tasks)
    outcome = {
        "answers": {}, "fallback_tasks": tasks,
        "prompt_tokens": 0, "completion_tokens": 0,
        "fireworks_calls": 0, "error": None,
        "category": tasks[0].get("category", "unknown") if tasks else "unknown",
    }
    categories = {task.get("category", "unknown") for task in tasks}
    if (
        not MIN_BATCH_SIZE <= len(tasks) <= MAX_BATCH_SIZE
        or len(categories) != 1
        or outcome["category"] not in BATCHABLE_CATEGORIES
    ):
        outcome["error"] = "ineligible or mixed-category batch"
        return outcome
    now = time.monotonic()
    if now >= deadline:
        outcome["error"] = "soft budget exhausted before batch dispatch"
        return outcome

    category = outcome["category"]
    effective_extra = None if category in FULL_EFFORT_CATEGORIES else extra_body
    first_cap, _ = token_caps_for(category)
    try:
        outcome["fireworks_calls"] = 1
        resp = client.chat.completions.create(
            model=model,
            messages=build_batch_messages(tasks, SYSTEM_PROMPT, build_user_message),
            max_tokens=batch_token_cap(first_cap, len(tasks)),
            temperature=0.0,
            timeout=min(FIRST_TIMEOUT_SECONDS, max(0.1, deadline - now)),
            **({"extra_body": effective_extra} if effective_extra else {}),
        )
        usage = getattr(resp, "usage", None)
        if usage is not None:
            outcome["prompt_tokens"] = usage.prompt_tokens or 0
            outcome["completion_tokens"] = usage.completion_tokens or 0
        choice = resp.choices[0]
        content = (choice.message.content or "").strip()
        finish = getattr(choice, "finish_reason", None)
        if finish == "length":
            outcome["error"] = "truncated batch content (finish_reason=length)"
            return outcome
        parsed = parse_batch_answers(content, (task["task_id"] for task in tasks))
        outcome["answers"] = parsed.answers
        fallback = set(parsed.fallback_ids)
        outcome["fallback_tasks"] = [
            task for task in tasks if str(task["task_id"]) in fallback
        ]
        outcome["error"] = parsed.error
    except Exception as exc:  # noqa: BLE001 - every task falls back individually
        outcome["error"] = f"{type(exc).__name__}: {exc}"
    return outcome


def main():
    start = time.monotonic()
    deadline = start + SOFT_BUDGET_SECONDS
    input_path = os.environ.get("AGENT_INPUT", "/input/tasks.json")
    output_path = os.environ.get("AGENT_OUTPUT", "/output/results.json")

    cfg = load_config()

    try:
        task_ids, answerable = load_tasks(input_path)
    except Exception as exc:  # unreadable input: still leave valid JSON behind
        log(f"FATAL: cannot read tasks: {type(exc).__name__}: {exc}")
        write_snapshot([], {}, output_path)
        return 1

    answers = {tid: "" for tid in task_ids}
    write_snapshot(task_ids, answers, output_path)  # valid output exists from t=0

    task_prompt_tokens = task_completion_tokens = failed = 0
    task_calls = retry_calls = retry_prompt_tokens = retry_completion_tokens = 0
    batch_calls = batch_prompt_tokens = batch_completion_tokens = 0
    batch_accepted_tasks = batch_fallback_tasks = batch_failures = 0
    probe_stats = _empty_usage_stats()
    model = "(none)"
    try:
        client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"], max_retries=0)
        model, extra_body = pick_working_model(
            client, cfg["allowed_models"], probe_stats=probe_stats)
        log(f"model: {model} (from {len(cfg['allowed_models'])} allowed)")
        routing = load_routing_table(os.environ.get(
            "ROUTING_TABLE",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "routing_table.json")))
        local = None
        if any(v == "local" for v in routing.values()):
            try:
                from agent.local_model import LocalModel
                local = LocalModel()
                log("local model loaded")
            except Exception as exc:  # noqa: BLE001 — pure-cloud mode is always safe
                log(f"WARN: local model unavailable ({type(exc).__name__}: {exc}); pure-cloud mode")
                routing = {}
        for t in answerable:
            t["category"] = categorize(t["prompt"])

        batch_groups = []
        individual_tasks = list(answerable)
        if batching_enabled():
            batch_groups, individual_tasks = plan_batches(answerable, routing)
            log(
                f"batching: enabled categories={','.join(sorted(BATCHABLE_CATEGORIES))} "
                f"batches={len(batch_groups)} "
                f"batched_tasks={sum(len(group) for group in batch_groups)}"
            )

        # Batch requests run first. Any missing, malformed, or ambiguous item
        # is appended to the established individual path below.
        if batch_groups:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                futures = [
                    pool.submit(answer_batch, client, model, group, deadline, extra_body)
                    for group in batch_groups
                ]
                for fut in as_completed(futures):
                    outcome = fut.result()  # answer_batch never raises
                    accepted = len(outcome["answers"])
                    fallback_count = len(outcome["fallback_tasks"])
                    batch_size = accepted + fallback_count
                    batch_calls += outcome["fireworks_calls"]
                    batch_prompt_tokens += outcome["prompt_tokens"]
                    batch_completion_tokens += outcome["completion_tokens"]
                    batch_accepted_tasks += accepted
                    batch_fallback_tasks += fallback_count
                    batch_failures += int(bool(outcome["error"]))
                    task_calls += outcome["fireworks_calls"]
                    task_prompt_tokens += outcome["prompt_tokens"]
                    task_completion_tokens += outcome["completion_tokens"]
                    log(
                        f"batch cat={outcome['category']} size={batch_size} "
                        f"accepted={accepted} fallback={fallback_count} "
                        f"pt={outcome['prompt_tokens']} ct={outcome['completion_tokens']} "
                        f"calls={outcome['fireworks_calls']}"
                    )
                    if outcome["error"]:
                        log(f"WARN: batch {outcome['category']}: {outcome['error']}")
                    for task_id, answer in outcome["answers"].items():
                        answers[task_id] = answer
                        log(
                            f"task={task_id} cat={outcome['category']} pt=0 ct=0 "
                            "calls=0 retries=0 lane=fireworks_batch"
                        )
                    individual_tasks.extend(outcome["fallback_tasks"])
                    write_snapshot(task_ids, answers, output_path)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [pool.submit(answer_task, client, model, t, deadline,
                                   extra_body, local, routing) for t in individual_tasks]
            for fut in as_completed(futures):
                r = fut.result()  # answer_task never raises
                answers[r["task_id"]] = r["answer"]
                log(f"task={r['task_id']} cat={r['category']} "
                    f"pt={r['prompt_tokens']} ct={r['completion_tokens']} "
                    f"calls={r['fireworks_calls']} retries={r['retry_calls']} lane={r['lane']}")
                task_prompt_tokens += r["prompt_tokens"]
                task_completion_tokens += r["completion_tokens"]
                task_calls += r["fireworks_calls"]
                retry_calls += r["retry_calls"]
                retry_prompt_tokens += r["retry_prompt_tokens"]
                retry_completion_tokens += r["retry_completion_tokens"]
                if r["error"]:
                    failed += 1
                    log(f"WARN: {r['task_id']}: {r['error']}")
                write_snapshot(task_ids, answers, output_path)
    except Exception as exc:  # noqa: BLE001 — a valid snapshot already exists; don't fail the run
        log(f"WARN: run aborted early: {type(exc).__name__}: {exc}")
    finally:
        write_snapshot(task_ids, answers, output_path)
        answered = sum(1 for a in answers.values() if a)
        prompt_tokens = probe_stats["prompt_tokens"] + task_prompt_tokens
        completion_tokens = probe_stats["completion_tokens"] + task_completion_tokens
        fireworks_calls = probe_stats["calls"] + task_calls
        log(
            f"stats: tasks={len(task_ids)} answered={answered} failed={failed} "
            f"fireworks_calls={fireworks_calls} probe_calls={probe_stats['calls']} "
            f"task_calls={task_calls} retry_calls={retry_calls} "
            f"batch_calls={batch_calls} batch_failures={batch_failures} "
            f"batch_accepted_tasks={batch_accepted_tasks} "
            f"batch_fallback_tasks={batch_fallback_tasks} "
            f"probe_prompt_tokens={probe_stats['prompt_tokens']} "
            f"probe_completion_tokens={probe_stats['completion_tokens']} "
            f"batch_prompt_tokens={batch_prompt_tokens} "
            f"batch_completion_tokens={batch_completion_tokens} "
            f"task_prompt_tokens={task_prompt_tokens} "
            f"task_completion_tokens={task_completion_tokens} "
            f"retry_prompt_tokens={retry_prompt_tokens} "
            f"retry_completion_tokens={retry_completion_tokens} "
            f"prompt_tokens={prompt_tokens} completion_tokens={completion_tokens} "
            f"total_tokens={prompt_tokens + completion_tokens} "
            f"elapsed={time.monotonic() - start:.1f}s model={model}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Phase 1 Baseline Routing Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A submittable batch-job Docker container that answers every task in `/input/tasks.json` via the cheapest `ALLOWED_MODELS` Fireworks model and always writes valid `/output/results.json`.

**Architecture:** Single entrypoint `agent/main.py`: fail-fast env loading → task validation → cheapest-model pick (size-regex + env override) → `ThreadPoolExecutor(4)` with 20 s/25 s request timeouts and one retry → atomic full-file snapshot of results after every completed task → stats to stderr, exit 0. Spec: `docs/superpowers/specs/2026-07-11-phase1-baseline-design.md`.

**Tech Stack:** Python 3.11, `openai` client (only runtime dep), pytest (dev only), Docker `python:3.11-slim` on linux/amd64.

## Global Constraints

- Grading VM: 4 GB RAM, 2 vCPU, CPU-only — smoke test must run with `--memory=4g --cpus=2`
- Total runtime 10 min; container ready within 60 s; each response under 30 s
- Image ≤ 10 GB compressed, must include a `linux/amd64` manifest
- All Fireworks calls through `FIREWORKS_BASE_URL`; models only from runtime `ALLOWED_MODELS`; never hardcode model IDs
- `.env` must never enter the image or git history (`.gitignore` already covers it; `.dockerignore` in Task 1)
- No answer hardcoding/caching keyed on prompt text
- All answers in English (system prompt enforces)
- `/output/results.json`: JSON list of `{"task_id": str, "answer": str}` — always present, always valid
- Runtime constants (from spec, use these exact values): `MAX_TOKENS=1024`, `FIRST_TIMEOUT_SECONDS=20.0`, `RETRY_TIMEOUT_SECONDS=25.0`, `RETRY_BACKOFF_SECONDS=2.0`, `SOFT_BUDGET_SECONDS=540.0`, `MAX_WORKERS=4`, system prompt `"Answer in English. Be correct and complete, but concise."`

---

### Task 1: Repo skeleton, dependencies, dev environment

**Files:**
- Create: `requirements.txt`, `requirements-dev.txt`, `.dockerignore`, `practice_tasks.json`, `agent/__init__.py`, `tests/__init__.py`
- Test: manual verification (no code yet)

**Interfaces:**
- Consumes: nothing
- Produces: `.venv` with `openai` + `pytest` installed; `practice_tasks.json` used by Task 8's smoke test; `agent` and `tests` importable as packages from the repo root

- [ ] **Step 1: Write `requirements.txt`**

```
openai>=1.60,<2
```

- [ ] **Step 2: Write `requirements-dev.txt`**

```
-r requirements.txt
pytest>=8
```

- [ ] **Step 3: Write `.dockerignore`**

The Dockerfile (Task 7) only COPYs `requirements.txt` and `agent/`, so this is defense in depth — it guarantees `.env` and junk can never enter the build context even if the Dockerfile changes later.

```
.env
.env.example
.git
.gitignore
.venv
.cursorrules
*.pdf
*.txt.bak
guide.txt
hackaton_description.txt
claude.md
docs
tests
input
output
__pycache__
*.pyc
run_local.sh
practice_tasks.json
requirements-dev.txt
README.md
```

- [ ] **Step 4: Write `practice_tasks.json`** (the 8 practice tasks from the participant guide; practice-04's placeholder paragraph replaced with a real one as the guide instructs)

```json
[
  { "task_id": "practice-01", "prompt": "What is the capital of Australia, and what body of water is it near?" },
  { "task_id": "practice-02", "prompt": "A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. How many items remain?" },
  { "task_id": "practice-03", "prompt": "Classify the sentiment of this review: The battery life is great, but the screen scratches too easily." },
  { "task_id": "practice-04", "prompt": "Summarize the following in exactly one sentence: Remote work has reshaped how companies think about office space. Many firms have downsized their headquarters, adopted hot-desking, or moved to fully distributed models. At the same time, employees report both greater flexibility and a blurring of boundaries between work and home life, prompting some organizations to introduce meeting-free days and stricter norms around after-hours communication." },
  { "task_id": "practice-05", "prompt": "Extract all named entities and their types from: Maria Sanchez joined Fireworks AI in Berlin last March." },
  { "task_id": "practice-06", "prompt": "This function should return the max of a list but has a bug: def get_max(nums): return nums[0]. Find and fix it." },
  { "task_id": "practice-07", "prompt": "Three friends, Sam, Jo, and Lee, each own a different pet: cat, dog, bird. Sam does not own the bird. Jo owns the dog. Who owns the cat?" },
  { "task_id": "practice-08", "prompt": "Write a Python function that returns the second-largest number in a list, handling duplicates correctly." }
]
```

- [ ] **Step 5: Create empty package markers**

Create `agent/__init__.py` and `tests/__init__.py`, both empty (they make `from agent import main` work under pytest from the repo root).

- [ ] **Step 6: Create the dev venv and install**

Run:
```bash
cd /home/caverraaa/workspace/github.com/caverraaa/agent-router
python3 -m venv .venv && .venv/bin/pip install --quiet -r requirements-dev.txt
.venv/bin/python -c "import openai, pytest, json; json.load(open('practice_tasks.json')); print('OK')"
```
Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add requirements.txt requirements-dev.txt .dockerignore practice_tasks.json agent/__init__.py tests/__init__.py
git commit -m "chore: repo skeleton, deps, practice tasks"
```

---

### Task 2: `pick_cheapest_model` — size-regex + env override

**Files:**
- Create: `agent/main.py`
- Test: `tests/test_pick_model.py`

**Interfaces:**
- Consumes: nothing (module bootstrap)
- Produces: `parse_model_size(model_id: str) -> float` (billions of params, `math.inf` if unparseable); `pick_cheapest_model(allowed_models: list[str]) -> str` (always returns a member of `allowed_models`); module constants listed in Global Constraints; `log(msg: str) -> None` (stderr, flushed)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pick_model.py`:

```python
import math

from agent.main import parse_model_size, pick_cheapest_model


def test_parse_size_plain_billions():
    assert parse_model_size("accounts/fireworks/models/llama-v3p1-8b-instruct") == 8.0


def test_parse_size_decimal():
    assert parse_model_size("accounts/fireworks/models/qwen3-0.6b") == 0.6


def test_parse_size_case_insensitive():
    assert parse_model_size("accounts/fireworks/models/Gemma-2-9B-it") == 9.0


def test_parse_size_unparseable_is_inf():
    assert parse_model_size("accounts/fireworks/models/mixtral-moe-instruct") == math.inf


def test_parse_size_does_not_match_word_prefixes():
    # 'b' followed by more letters is a word, not a size suffix
    assert parse_model_size("accounts/fireworks/models/model-3base") == math.inf


def test_picks_smallest(monkeypatch):
    monkeypatch.delenv("CHEAP_MODEL", raising=False)
    allowed = [
        "accounts/fireworks/models/llama-v3p1-70b-instruct",
        "accounts/fireworks/models/gemma-2-2b-it",
        "accounts/fireworks/models/llama-v3p1-8b-instruct",
    ]
    assert pick_cheapest_model(allowed) == "accounts/fireworks/models/gemma-2-2b-it"


def test_all_unparseable_falls_back_to_first(monkeypatch):
    monkeypatch.delenv("CHEAP_MODEL", raising=False)
    allowed = ["accounts/x/alpha-instruct", "accounts/x/beta-instruct"]
    assert pick_cheapest_model(allowed) == "accounts/x/alpha-instruct"


def test_env_override_wins(monkeypatch):
    allowed = ["accounts/x/small-2b", "accounts/x/big-70b"]
    monkeypatch.setenv("CHEAP_MODEL", "accounts/x/big-70b")
    assert pick_cheapest_model(allowed) == "accounts/x/big-70b"


def test_env_override_outside_list_is_ignored(monkeypatch):
    allowed = ["accounts/x/small-2b", "accounts/x/big-70b"]
    monkeypatch.setenv("CHEAP_MODEL", "accounts/evil/not-allowed")
    assert pick_cheapest_model(allowed) == "accounts/x/small-2b"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_pick_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.main'` (collection error is fine)

- [ ] **Step 3: Write the implementation**

Create `agent/main.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_pick_model.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add agent/main.py tests/test_pick_model.py
git commit -m "feat: cheapest-model selection (size regex + CHEAP_MODEL override)"
```

---

### Task 3: Env config and task loading/validation

**Files:**
- Modify: `agent/main.py` (append below `pick_cheapest_model`)
- Test: `tests/test_config_and_tasks.py`

**Interfaces:**
- Consumes: `log` from Task 2
- Produces: `load_config() -> dict` with keys `api_key: str`, `base_url: str`, `allowed_models: list[str]` (raises `SystemExit(1)` naming missing vars); `load_tasks(path: str) -> tuple[list[str], list[dict]]` — (all usable task_ids in input order, answerable `{"task_id","prompt"}` dicts)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_config_and_tasks.py`:

```python
import json

import pytest

from agent.main import load_config, load_tasks

GOOD_ENV = {
    "FIREWORKS_API_KEY": "fk-test",
    "FIREWORKS_BASE_URL": "https://proxy.example/v1",
    "ALLOWED_MODELS": "accounts/x/small-2b, accounts/x/big-70b,,",
}


def set_env(monkeypatch, overrides=None):
    env = {**GOOD_ENV, **(overrides or {})}
    for k in GOOD_ENV:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        if v is not None:
            monkeypatch.setenv(k, v)


def test_load_config_happy_path(monkeypatch):
    set_env(monkeypatch)
    cfg = load_config()
    assert cfg["api_key"] == "fk-test"
    assert cfg["base_url"] == "https://proxy.example/v1"
    assert cfg["allowed_models"] == ["accounts/x/small-2b", "accounts/x/big-70b"]


@pytest.mark.parametrize("missing", ["FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS"])
def test_load_config_missing_var_exits_1(monkeypatch, capsys, missing):
    set_env(monkeypatch, {missing: None})
    with pytest.raises(SystemExit) as e:
        load_config()
    assert e.value.code == 1
    assert missing in capsys.readouterr().err


def test_load_config_empty_model_list_exits_1(monkeypatch):
    set_env(monkeypatch, {"ALLOWED_MODELS": " , ,"})
    with pytest.raises(SystemExit) as e:
        load_config()
    assert e.value.code == 1


def write_tasks(tmp_path, data):
    p = tmp_path / "tasks.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


def test_load_tasks_happy_path(tmp_path):
    path = write_tasks(tmp_path, [
        {"task_id": "t1", "prompt": "hello"},
        {"task_id": "t2", "prompt": "world"},
    ])
    task_ids, answerable = load_tasks(path)
    assert task_ids == ["t1", "t2"]
    assert answerable == [{"task_id": "t1", "prompt": "hello"}, {"task_id": "t2", "prompt": "world"}]


def test_load_tasks_entry_without_prompt_kept_in_ids_only(tmp_path):
    path = write_tasks(tmp_path, [{"task_id": "t1"}, {"task_id": "t2", "prompt": "ok"}])
    task_ids, answerable = load_tasks(path)
    assert task_ids == ["t1", "t2"]
    assert [t["task_id"] for t in answerable] == ["t2"]


def test_load_tasks_entry_without_task_id_skipped(tmp_path):
    path = write_tasks(tmp_path, [{"prompt": "orphan"}, {"task_id": "t2", "prompt": "ok"}, "junk"])
    task_ids, answerable = load_tasks(path)
    assert task_ids == ["t2"]
    assert [t["task_id"] for t in answerable] == ["t2"]


def test_load_tasks_non_list_raises(tmp_path):
    path = write_tasks(tmp_path, {"task_id": "t1"})
    with pytest.raises(ValueError):
        load_tasks(path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_config_and_tasks.py -v`
Expected: FAIL — `ImportError: cannot import name 'load_config'`

- [ ] **Step 3: Write the implementation** (append to `agent/main.py` after `pick_cheapest_model`)

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_config_and_tasks.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add agent/main.py tests/test_config_and_tasks.py
git commit -m "feat: fail-fast env config and tolerant task loading"
```

---

### Task 4: Atomic results snapshot with schema coercion

**Files:**
- Modify: `agent/main.py` (append below `load_tasks`)
- Test: `tests/test_snapshot.py`

**Interfaces:**
- Consumes: nothing new
- Produces: `write_snapshot(task_ids: list[str], answers: dict[str, str], path: str) -> None` — writes the FULL results list (one entry per task_id, input order) as valid JSON via tmp-file + `os.replace`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_snapshot.py`:

```python
import json
import os

from agent.main import write_snapshot


def read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_writes_all_ids_in_order(tmp_path):
    path = str(tmp_path / "results.json")
    write_snapshot(["t1", "t2", "t3"], {"t2": "two"}, path)
    assert read(path) == [
        {"task_id": "t1", "answer": ""},
        {"task_id": "t2", "answer": "two"},
        {"task_id": "t3", "answer": ""},
    ]


def test_coerces_none_answer_to_empty_string(tmp_path):
    path = str(tmp_path / "results.json")
    write_snapshot(["t1"], {"t1": None}, path)
    assert read(path) == [{"task_id": "t1", "answer": ""}]


def test_no_tmp_file_left_behind(tmp_path):
    path = str(tmp_path / "results.json")
    write_snapshot(["t1"], {"t1": "x"}, path)
    assert os.listdir(tmp_path) == ["results.json"]


def test_overwrite_keeps_file_valid(tmp_path):
    path = str(tmp_path / "results.json")
    write_snapshot(["t1"], {}, path)
    write_snapshot(["t1"], {"t1": "answered"}, path)
    assert read(path) == [{"task_id": "t1", "answer": "answered"}]


def test_empty_task_list_writes_empty_json_list(tmp_path):
    path = str(tmp_path / "results.json")
    write_snapshot([], {}, path)
    assert read(path) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_snapshot.py -v`
Expected: FAIL — `ImportError: cannot import name 'write_snapshot'`

- [ ] **Step 3: Write the implementation** (append to `agent/main.py`)

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_snapshot.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add agent/main.py tests/test_snapshot.py
git commit -m "feat: atomic schema-coerced results snapshot"
```

---

### Task 5: `answer_task` — one call, one retry, deadline-aware

**Files:**
- Modify: `agent/main.py` (append below `write_snapshot`)
- Test: `tests/test_answer_task.py`

**Interfaces:**
- Consumes: constants from Task 2 (`SYSTEM_PROMPT`, `MAX_TOKENS`, `FIRST_TIMEOUT_SECONDS`, `RETRY_TIMEOUT_SECONDS`, `RETRY_BACKOFF_SECONDS`)
- Produces: `answer_task(client, model: str, task: dict, deadline: float) -> dict` with keys `task_id: str`, `answer: str`, `prompt_tokens: int`, `completion_tokens: int`, `error: str | None`. Never raises. `deadline` is a `time.monotonic()` value; past it, no API call is made.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_answer_task.py`:

```python
import time
from types import SimpleNamespace

import agent.main as m
from agent.main import answer_task


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, outcomes):
        self.chat = SimpleNamespace(completions=FakeCompletions(outcomes))


def fake_response(text, prompt_tokens=10, completion_tokens=5):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


TASK = {"task_id": "t1", "prompt": "What is 2+2?"}
FUTURE = time.monotonic() + 3600


def test_success(monkeypatch):
    client = FakeClient([fake_response("4")])
    r = answer_task(client, "m-2b", TASK, FUTURE)
    assert r == {"task_id": "t1", "answer": "4", "prompt_tokens": 10,
                 "completion_tokens": 5, "error": None}
    call = client.chat.completions.calls[0]
    assert call["model"] == "m-2b"
    assert call["max_tokens"] == m.MAX_TOKENS
    assert call["timeout"] == m.FIRST_TIMEOUT_SECONDS
    assert call["messages"][0] == {"role": "system", "content": m.SYSTEM_PROMPT}
    assert call["messages"][1] == {"role": "user", "content": "What is 2+2?"}


def test_retry_succeeds_with_longer_timeout(monkeypatch):
    monkeypatch.setattr(m, "RETRY_BACKOFF_SECONDS", 0)
    client = FakeClient([RuntimeError("boom"), fake_response("4")])
    r = answer_task(client, "m-2b", TASK, FUTURE)
    assert r["answer"] == "4"
    assert r["error"] is None
    assert client.chat.completions.calls[1]["timeout"] == m.RETRY_TIMEOUT_SECONDS


def test_both_attempts_fail_returns_empty(monkeypatch):
    monkeypatch.setattr(m, "RETRY_BACKOFF_SECONDS", 0)
    client = FakeClient([RuntimeError("a"), RuntimeError("b")])
    r = answer_task(client, "m-2b", TASK, FUTURE)
    assert r["answer"] == ""
    assert "RuntimeError" in r["error"]
    assert len(client.chat.completions.calls) == 2


def test_past_deadline_makes_no_api_call():
    client = FakeClient([fake_response("never")])
    r = answer_task(client, "m-2b", TASK, time.monotonic() - 1)
    assert r["answer"] == ""
    assert "budget" in r["error"]
    assert client.chat.completions.calls == []


def test_none_content_and_missing_usage_handled():
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
        usage=None,
    )
    client = FakeClient([resp])
    r = answer_task(client, "m-2b", TASK, FUTURE)
    assert r["answer"] == ""
    assert r["prompt_tokens"] == 0 and r["completion_tokens"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_answer_task.py -v`
Expected: FAIL — `ImportError: cannot import name 'answer_task'`

- [ ] **Step 3: Write the implementation** (append to `agent/main.py`)

Note: read `RETRY_BACKOFF_SECONDS` via the module global at call time (not captured in a default arg) so tests can monkeypatch it.

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_answer_task.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add agent/main.py tests/test_answer_task.py
git commit -m "feat: deadline-aware answer_task with single retry"
```

---

### Task 6: `main()` orchestration — pool, budget, stats, exit codes

**Files:**
- Modify: `agent/main.py` (append below `answer_task`)
- Test: `tests/test_main_flow.py`

**Interfaces:**
- Consumes: everything from Tasks 2–5; `OpenAI` from the `openai` package (patchable as `agent.main.OpenAI`)
- Produces: `main() -> int`; `__main__` guard `sys.exit(main())`. Dev-only env overrides `AGENT_INPUT` (default `/input/tasks.json`) and `AGENT_OUTPUT` (default `/output/results.json`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_main_flow.py`:

```python
import json
from types import SimpleNamespace

import agent.main as m

from tests.test_answer_task import FakeClient, fake_response


def setup_env(monkeypatch, tmp_path, tasks):
    input_path = tmp_path / "tasks.json"
    output_path = tmp_path / "results.json"
    input_path.write_text(json.dumps(tasks), encoding="utf-8")
    monkeypatch.setenv("FIREWORKS_API_KEY", "fk-test")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://proxy.example/v1")
    monkeypatch.setenv("ALLOWED_MODELS", "accounts/x/small-2b,accounts/x/big-70b")
    monkeypatch.delenv("CHEAP_MODEL", raising=False)
    monkeypatch.setenv("AGENT_INPUT", str(input_path))
    monkeypatch.setenv("AGENT_OUTPUT", str(output_path))
    return output_path


def patch_client(monkeypatch, outcomes):
    created = {}

    def fake_openai(base_url, api_key):
        created["base_url"] = base_url
        created["api_key"] = api_key
        return FakeClient(outcomes)

    monkeypatch.setattr(m, "OpenAI", fake_openai)
    return created


def test_happy_path(monkeypatch, tmp_path, capsys):
    # 1 worker: with 4, either task could claim either fake outcome (flaky)
    monkeypatch.setattr(m, "MAX_WORKERS", 1)
    out = setup_env(monkeypatch, tmp_path, [
        {"task_id": "t1", "prompt": "a"},
        {"task_id": "t2", "prompt": "b"},
    ])
    created = patch_client(monkeypatch, [fake_response("A"), fake_response("B")])
    assert m.main() == 0
    results = {r["task_id"]: r["answer"] for r in json.loads(out.read_text())}
    assert results == {"t1": "A", "t2": "B"}
    assert created["base_url"] == "https://proxy.example/v1"
    assert created["api_key"] == "fk-test"
    err = capsys.readouterr().err
    assert "stats:" in err and "total_tokens=30" in err


def test_one_task_failure_does_not_kill_run(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "RETRY_BACKOFF_SECONDS", 0)
    monkeypatch.setattr(m, "MAX_WORKERS", 1)  # deterministic outcome ordering
    out = setup_env(monkeypatch, tmp_path, [
        {"task_id": "t1", "prompt": "a"},
        {"task_id": "t2", "prompt": "b"},
    ])
    patch_client(monkeypatch, [RuntimeError("x"), RuntimeError("y"), fake_response("B")])
    assert m.main() == 0
    results = {r["task_id"]: r["answer"] for r in json.loads(out.read_text())}
    assert results == {"t1": "", "t2": "B"}


def test_unanswerable_task_still_in_output(monkeypatch, tmp_path):
    out = setup_env(monkeypatch, tmp_path, [
        {"task_id": "t1"},
        {"task_id": "t2", "prompt": "b"},
    ])
    patch_client(monkeypatch, [fake_response("B")])
    assert m.main() == 0
    results = json.loads(out.read_text())
    assert results == [{"task_id": "t1", "answer": ""}, {"task_id": "t2", "answer": "B"}]


def test_unreadable_tasks_writes_empty_list_and_exits_1(monkeypatch, tmp_path):
    out = setup_env(monkeypatch, tmp_path, [])
    (tmp_path / "tasks.json").write_text("{not json", encoding="utf-8")
    patch_client(monkeypatch, [])
    assert m.main() == 1
    assert json.loads(out.read_text()) == []


def test_exhausted_budget_degrades_to_empty_answers(monkeypatch, tmp_path):
    out = setup_env(monkeypatch, tmp_path, [{"task_id": "t1", "prompt": "a"}])
    client_outcomes = [fake_response("never")]
    patch_client(monkeypatch, client_outcomes)
    monkeypatch.setattr(m, "SOFT_BUDGET_SECONDS", -1.0)  # deadline already passed
    assert m.main() == 0
    assert json.loads(out.read_text()) == [{"task_id": "t1", "answer": ""}]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_main_flow.py -v`
Expected: FAIL — `AttributeError: module 'agent.main' has no attribute 'main'`

- [ ] **Step 3: Write the implementation** (append to `agent/main.py`)

```python
def main():
    start = time.monotonic()
    deadline = start + SOFT_BUDGET_SECONDS
    input_path = os.environ.get("AGENT_INPUT", "/input/tasks.json")
    output_path = os.environ.get("AGENT_OUTPUT", "/output/results.json")

    cfg = load_config()
    model = pick_cheapest_model(cfg["allowed_models"])
    log(f"model: {model} (from {len(cfg['allowed_models'])} allowed)")

    try:
        task_ids, answerable = load_tasks(input_path)
    except Exception as exc:  # unreadable input: still leave valid JSON behind
        log(f"FATAL: cannot read tasks: {type(exc).__name__}: {exc}")
        write_snapshot([], {}, output_path)
        return 1

    answers = {tid: "" for tid in task_ids}
    write_snapshot(task_ids, answers, output_path)  # valid output exists from t=0

    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
    prompt_tokens = completion_tokens = failed = 0
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [pool.submit(answer_task, client, model, t, deadline) for t in answerable]
            for fut in as_completed(futures):
                r = fut.result()  # answer_task never raises
                answers[r["task_id"]] = r["answer"]
                prompt_tokens += r["prompt_tokens"]
                completion_tokens += r["completion_tokens"]
                if r["error"]:
                    failed += 1
                    log(f"WARN: {r['task_id']}: {r['error']}")
                write_snapshot(task_ids, answers, output_path)
    finally:
        write_snapshot(task_ids, answers, output_path)
        answered = sum(1 for a in answers.values() if a)
        log(
            f"stats: tasks={len(task_ids)} answered={answered} failed={failed} "
            f"prompt_tokens={prompt_tokens} completion_tokens={completion_tokens} "
            f"total_tokens={prompt_tokens + completion_tokens} "
            f"elapsed={time.monotonic() - start:.1f}s model={model}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Implementation note: `main()` must read `SOFT_BUDGET_SECONDS`, `MAX_WORKERS`, and `RETRY_BACKOFF_SECONDS` as module globals at call time (the code above does), so the monkeypatched tests work.

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/pytest -v`
Expected: all tests pass (Tasks 2–6: 33 tests)

- [ ] **Step 5: Commit**

```bash
git add agent/main.py tests/test_main_flow.py
git commit -m "feat: main orchestration with worker pool, soft budget, stats"
```

---

### Task 7: Dockerfile + linux/amd64 build

**Files:**
- Create: `Dockerfile`
- Test: docker build + a no-network container run

**Interfaces:**
- Consumes: `requirements.txt`, `agent/` from earlier tasks
- Produces: local image `routing-agent` used by Task 8's smoke test

- [ ] **Step 1: Write `Dockerfile`**

```dockerfile
FROM python:3.11-slim

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY agent/ /app/agent/

CMD ["python", "/app/agent/main.py"]
```

- [ ] **Step 2: Build for linux/amd64**

Run:
```bash
docker buildx build --platform linux/amd64 -t routing-agent . 2>&1 | tail -5
```
Expected: build succeeds. If `docker`/`buildx` is unavailable in this WSL2 session, STOP and ask the user to enable Docker Desktop WSL integration — everything before this task is still valid.

- [ ] **Step 3: Verify image size and fail-fast behavior**

Run:
```bash
docker image ls routing-agent --format '{{.Size}}'
docker run --rm routing-agent; echo "exit=$?"
```
Expected: size well under 1 GB (≈200 MB); run prints `FATAL: missing required environment variables: FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS` and `exit=1`

- [ ] **Step 4: Commit**

```bash
git add Dockerfile
git commit -m "feat: slim linux/amd64 Dockerfile"
```

---

### Task 8: Smoke test script, .env.example, README, live practice run

**Files:**
- Create: `run_local.sh`, `.env.example`, `README.md`
- Test: the script itself run end-to-end against real Fireworks

**Interfaces:**
- Consumes: `practice_tasks.json` (Task 1), image build (Task 7), user-provided `.env`
- Produces: the pre-submission validation ritual: `./run_local.sh` = build + grading-VM-limits run + schema assert + printed answers for the human accuracy eyeball

- [ ] **Step 1: Write `.env.example`** (committed; real `.env` stays gitignored)

```
# Local development only — the grading harness injects the real values.
FIREWORKS_API_KEY=fw_your_key_here
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
# Comma-separated Fireworks model IDs. Real list publishes on launch day.
ALLOWED_MODELS=accounts/fireworks/models/llama-v3p1-8b-instruct
# Optional: force the model pick (must be a member of ALLOWED_MODELS)
# CHEAP_MODEL=
```

- [ ] **Step 2: Write `run_local.sh`**

```bash
#!/usr/bin/env bash
# Pre-submission smoke test: build + run under grading-VM limits + schema assert.
set -euo pipefail
cd "$(dirname "$0")"

[ -f .env ] || { echo "FATAL: .env missing (copy .env.example and fill in your key)"; exit 1; }

mkdir -p input output
cp practice_tasks.json input/tasks.json

docker buildx build --platform linux/amd64 -t routing-agent .

# set -e makes a non-zero container exit abort the script — that IS the exit-code assertion
docker run --rm --memory=4g --cpus=2 \
  --env-file .env \
  -v "$PWD/input:/input" -v "$PWD/output:/output" \
  routing-agent

python3 - <<'EOF'
import json

expected = {t["task_id"] for t in json.load(open("input/tasks.json"))}
results = json.load(open("output/results.json"))
assert isinstance(results, list), "results must be a JSON list"
seen = set()
for entry in results:
    assert isinstance(entry, dict), f"entry not an object: {entry!r}"
    assert isinstance(entry.get("task_id"), str), f"bad task_id: {entry!r}"
    assert isinstance(entry.get("answer"), str), f"bad answer: {entry!r}"
    seen.add(entry["task_id"])
assert seen == expected, f"task_id mismatch: missing={expected - seen} extra={seen - expected}"
print(f"SCHEMA OK — {len(results)} results\n")
print("Eyeball these answers BEFORE submitting (10 submissions/hour limit):")
for entry in results:
    print(f"\n=== {entry['task_id']} ===\n{entry['answer']}")
EOF
```

Then: `chmod +x run_local.sh`

- [ ] **Step 3: Write `README.md`** (submission requires setup/usage instructions)

```markdown
# Token-Efficient Routing Agent — AMD Hackathon ACT II, Track 1

Batch-job container: reads `/input/tasks.json`, answers each task via the
cheapest model in `ALLOWED_MODELS` through `FIREWORKS_BASE_URL`, writes
`/output/results.json`, exits 0.

## Setup

    cp .env.example .env    # fill in your Fireworks key (local dev only;
                            # the grading harness injects real values)

## Run the smoke test (build + grading-VM limits + schema check)

    ./run_local.sh

## Run unit tests

    python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
    .venv/bin/pytest

## Build & push for submission

    docker buildx build --platform linux/amd64 -t <registry>/<user>/routing-agent:latest --push .

The image must be publicly pullable and include a linux/amd64 manifest.

## Design

See `docs/superpowers/specs/2026-07-11-phase1-baseline-design.md` — every
design decision maps to a grading failure status it protects against.
```

- [ ] **Step 4: Run the live smoke test**

Precondition: `.env` filled in with the user's real Fireworks key and a valid model ID in `ALLOWED_MODELS`.

Run: `./run_local.sh`
Expected: build succeeds; container exits 0 in well under 10 minutes; `SCHEMA OK — 8 results`; 8 non-empty answers printed. **Human step: read all 8 answers and confirm each is plausibly correct before any real submission.**

- [ ] **Step 5: Check stderr stats line**

From the container output in Step 4, confirm a final line like:
`stats: tasks=8 answered=8 failed=0 prompt_tokens=… completion_tokens=… total_tokens=… elapsed=…s model=…`

- [ ] **Step 6: Commit**

```bash
git add run_local.sh .env.example README.md
git commit -m "feat: smoke-test script, env example, README"
```

---

## Not in this plan (deliberately)

- Pushing to a public registry and submitting — user-driven, after eyeballing practice answers.
- Local model / router / token tuning — Phases 2–3, separate plans.
- Scripted LLM judge — deferred per the approved design decision.

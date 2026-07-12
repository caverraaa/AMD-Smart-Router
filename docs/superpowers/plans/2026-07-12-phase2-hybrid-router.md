# Phase 2 Token Diet + Judged Hybrid Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut Fireworks tokens far below the scored 8,421 while never risking the 94.7% accuracy baseline — cheap cuts shipped early as v3, a judged Gemma-2-2B-it local lane shipped as v4 only behind our own ≥95% judge gate.

**Architecture:** Tiered and independently shippable. Tier 1 adds per-task telemetry and a reasoning-tax-aware model probe. Tier 2 adds a regex categorizer with per-category output constraints (v3). Tier 3 builds the golden set + binary-verdict LLM judge. Tier 4 adds `agent/local_model.py` (llama-cpp, single-threaded behind a lock) and a routing-table lane branch inside `answer_task()` (v4, judge-gated). Phase 1 machinery (pool, budget, snapshots, tolerant parsing, probe fallback, empty-content retry) is untouched.

**Tech Stack:** Python 3.11, `openai`, `llama-cpp-python` (CPU wheel, Tier 4 image only), Gemma-2-2B-it Q4_K_M GGUF, pytest.

## Global Constraints

- LIVE competition, ~16 h left; scoring queue backlogged. **The submission pointer only moves to an image that beats ≥95% on our own judge.** v3 ships EARLY; v4 only behind the judge gate.
- Hard limits: 4 GB RAM / 2 vCPU CPU-only, 10-min runtime, 60 s container-ready, 30 s per response, ≤10 GB compressed linux/amd64, English answers.
- All Fireworks traffic through `FIREWORKS_BASE_URL`; models only from runtime `ALLOWED_MODELS`; no answer caching/hardcoding; `.env` never in image or git.
- Phase 1 invariants preserved: `max_retries=0`, tolerant `load_tasks`, probe-with-fallback, empty-content retry, atomic snapshots, exit 0 when results written.
- Local model: **Gemma-2-2B-it Q4_K_M** (~1.7 GB), `llama-cpp-python`, `n_threads=2`, no system role (Gemma has no system turn — merge into the user message).
- Routing gate (from spec, count-based): a category routes LOCAL iff Gemma ≥10/12 on golden AND cloud − Gemma ≤ 1 task; `unknown` and unparseable classifications → fireworks.
- Local answers bounded by construction: `max_tokens=160` local cap (time-bounds generation on CPU); deadline checked before every local call.
- All test runs: `.venv/bin/pytest` from the repo root (51 tests passing at plan time, commit a04fe08).
- Docker needs sudo on this machine → every build/run/push step is marked **USER ACTION** with the exact command for the user to run via `!` prefix.

---

### Task 1: Regex categorizer (`agent/router.py`)

**Files:**
- Create: `agent/router.py`
- Test: `tests/test_router.py`

**Interfaces:**
- Consumes: nothing
- Produces: `CATEGORIES: tuple[str,...]` = `("sentiment","ner","summarisation","code_debug","code_gen","math","logic","factual","unknown")`; `categorize(prompt: str) -> str` (always returns a member of CATEGORIES); `CONSTRAINTS: dict[str,str]`; `build_user_message(prompt: str, category: str) -> str`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_router.py`:

```python
import pytest

from agent.router import CATEGORIES, CONSTRAINTS, build_user_message, categorize

PRACTICE = [
    ("What is the capital of Australia, and what body of water is it near?", "factual"),
    ("A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. How many items remain?", "math"),
    ("Classify the sentiment of this review: The battery life is great, but the screen scratches too easily.", "sentiment"),
    ("Summarize the following in exactly one sentence: Remote work has reshaped offices.", "summarisation"),
    ("Extract all named entities and their types from: Maria Sanchez joined Fireworks AI in Berlin last March.", "ner"),
    ("This function should return the max of a list but has a bug: def get_max(nums): return nums[0]. Find and fix it.", "code_debug"),
    ("Three friends, Sam, Jo, and Lee, each own a different pet: cat, dog, bird. Sam does not own the bird. Jo owns the dog. Who owns the cat?", "logic"),
    ("Write a Python function that returns the second-largest number in a list, handling duplicates correctly.", "code_gen"),
]


@pytest.mark.parametrize("prompt,expected", PRACTICE)
def test_categorize_practice_set(prompt, expected):
    assert categorize(prompt) == expected


def test_categorize_eval_grade_variants():
    assert categorize("A train leaves at 09:14 travelling 87 km/h. At what clock time does it arrive?") == "math"
    assert categorize("On 14 February 2025, Dr. Amara Okafor of the European Space Agency presented... Extract all named entities and their types.") == "ner"
    assert categorize("Explain how a hash table achieves average O(1) lookup.") == "factual"


def test_unmatched_prompt_is_unknown():
    assert categorize("zorble the frumious bandersnatch") == "unknown"


def test_categorize_always_returns_member():
    for p in ["", "hello", "do the thing with the stuff"]:
        assert categorize(p) in CATEGORIES


def test_build_user_message_appends_constraint():
    msg = build_user_message("Classify the sentiment: great phone.", "sentiment")
    assert msg.startswith("Classify the sentiment: great phone.")
    assert CONSTRAINTS["sentiment"] in msg


def test_build_user_message_unknown_passthrough():
    assert build_user_message("mystery task", "unknown") == "mystery task"


def test_sentiment_constraint_demands_justification():
    assert "justification" in CONSTRAINTS["sentiment"].lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_router.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'agent.router'`

- [ ] **Step 3: Write the implementation**

Create `agent/router.py`:

```python
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
    "sentiment": "Answer with the sentiment label plus a one-sentence justification.",
    "ner": "List each entity with its type, one per line. No extra text.",
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_router.py -v`
Expected: 16 passed. If a practice-prompt test fails, adjust the failing regex (rule order is part of the contract: sentiment → ner → summarisation → code_debug → code_gen → math → logic → factual).

- [ ] **Step 5: Run full suite, commit**

Run: `.venv/bin/pytest -q` → 67 passed
```bash
git add agent/router.py tests/test_router.py
git commit -m "feat(p2): regex categorizer with per-category output constraints"
```

---

### Task 2: Per-task telemetry (Tier 1)

**Files:**
- Modify: `agent/main.py` (imports, `answer_task` result dict, `main()` dispatch + collector)
- Test: `tests/test_telemetry.py`

**Interfaces:**
- Consumes: `categorize` from Task 1
- Produces: `answer_task` result dict gains `"category": str` and `"lane": str` (always `"fireworks"` until Task 9); collector logs one stderr line per task: `task=<id> cat=<category> pt=<n> ct=<n> lane=<lane>`; `main()` stamps `task["category"]` before dispatch

- [ ] **Step 1: Write the failing tests**

Create `tests/test_telemetry.py`:

```python
import json
import re
import time

import agent.main as m
from agent.main import answer_task

from tests.test_answer_task import FakeClient, fake_response
from tests.test_main_flow import patch_client, setup_env

FUTURE = time.monotonic() + 3600


def test_answer_task_carries_category_and_lane():
    client = FakeClient([fake_response("4")])
    task = {"task_id": "t1", "prompt": "What is 2+2?", "category": "math"}
    r = answer_task(client, "m-x", task, FUTURE)
    assert r["category"] == "math"
    assert r["lane"] == "fireworks"


def test_answer_task_defaults_unknown_category():
    client = FakeClient([fake_response("4")])
    r = answer_task(client, "m-x", {"task_id": "t1", "prompt": "hm"}, FUTURE)
    assert r["category"] == "unknown"


def test_collector_logs_per_task_line(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(m, "MAX_WORKERS", 1)
    setup_env(monkeypatch, tmp_path, [
        {"task_id": "t1", "prompt": "Classify the sentiment of this review: great."},
    ])
    patch_client(monkeypatch, [fake_response("OK"), fake_response("Positive")])
    assert m.main() == 0
    err = capsys.readouterr().err
    line = next(l for l in err.splitlines() if l.startswith("task=t1"))
    assert re.fullmatch(r"task=t1 cat=sentiment pt=\d+ ct=\d+ lane=fireworks", line)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_telemetry.py -v`
Expected: FAIL — `KeyError: 'category'` / missing telemetry line

- [ ] **Step 3: Implement**

In `agent/main.py`, below the stdlib imports and `from openai import OpenAI`, add the router import with a script-execution fallback (the container runs `python /app/agent/main.py` directly, so `agent` isn't on `sys.path` there):

```python
try:
    from agent.router import build_user_message, categorize
except ImportError:  # executed as a script (python /app/agent/main.py)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agent.router import build_user_message, categorize
```

In `answer_task`, change the result-dict initialization to:

```python
    result = {"task_id": task["task_id"], "answer": "",
              "prompt_tokens": 0, "completion_tokens": 0, "error": None,
              "category": task.get("category", "unknown"), "lane": "fireworks"}
```

In `main()`, stamp categories before dispatch (immediately before the `futures = [...]` line):

```python
            for t in answerable:
                t["category"] = categorize(t["prompt"])
```

And in the collector loop, right after `answers[r["task_id"]] = r["answer"]`, add:

```python
                log(f"task={r['task_id']} cat={r['category']} "
                    f"pt={r['prompt_tokens']} ct={r['completion_tokens']} lane={r['lane']}")
```

(`build_user_message` is imported now but first used in Task 4.)

- [ ] **Step 4: Run full suite**

Run: `.venv/bin/pytest -q`
Expected: 70 passed (67 + 3 new)

- [ ] **Step 5: Commit**

```bash
git add agent/main.py tests/test_telemetry.py
git commit -m "feat(p2): per-task category/token/lane telemetry to stderr"
```

---

### Task 3: Reasoning-tax-aware model selection (Tier 1)

**Files:**
- Modify: `agent/main.py` (`PROBE_MAX_TOKENS`, new `_probe`, rewrite `pick_working_model`, `answer_task` gains `extra_body`, `main()` unpacks tuple)
- Test: `tests/test_hardening.py` (update probe tests), new tests in same file

**Interfaces:**
- Consumes: existing `parse_model_size`, constants
- Produces: `pick_working_model(client, allowed_models) -> tuple[str, dict | None]` (model id, request `extra_body` or None); `answer_task(client, model, task, deadline, extra_body=None)`; new constants `PROBE_MAX_TOKENS = 200`, `LOW_OVERHEAD_TOKENS = 30`, `PROBE_PROMPT = "What is 2+2? Answer with just the number."`

- [ ] **Step 1: Update existing probe tests and add new ones**

In `tests/test_hardening.py`, replace the four probe tests with:

```python
def probe_response(text, completion_tokens):
    return fake_response(text, prompt_tokens=15, completion_tokens=completion_tokens)


def test_probe_picks_cheapest_when_lean(monkeypatch):
    monkeypatch.delenv("CHEAP_MODEL", raising=False)
    client = FakeClient([probe_response("4", 5)])  # overhead 4 <= 30 -> stop
    allowed = ["accounts/x/big-70b", "accounts/x/small-2b"]
    assert pick_working_model(client, allowed) == ("accounts/x/small-2b", None)
    assert len(client.chat.completions.calls) == 1


def test_probe_falls_through_to_next_model(monkeypatch):
    monkeypatch.delenv("CHEAP_MODEL", raising=False)
    client = FakeClient([RuntimeError("not a chat model"), probe_response("4", 5)])
    allowed = ["accounts/x/image-model", "accounts/x/chat-model"]
    assert pick_working_model(client, allowed) == ("accounts/x/chat-model", None)


def test_all_probes_fail_falls_back_to_cheapest(monkeypatch):
    monkeypatch.delenv("CHEAP_MODEL", raising=False)
    client = FakeClient([RuntimeError("a"), RuntimeError("b"), RuntimeError("c"), RuntimeError("d")])
    allowed = ["accounts/x/big-70b", "accounts/x/small-2b"]
    # 2 default probes + 0 low-effort probes (they only run after a HIGH overhead success)
    assert pick_working_model(client, allowed) == ("accounts/x/small-2b", None)
    assert len(client.chat.completions.calls) == 2


def test_probe_respects_cheap_model_override(monkeypatch):
    monkeypatch.setenv("CHEAP_MODEL", "accounts/x/big-70b")
    client = FakeClient([probe_response("4", 5)])
    allowed = ["accounts/x/small-2b", "accounts/x/big-70b"]
    model, extra = pick_working_model(client, allowed)
    assert model == "accounts/x/big-70b"
    assert client.chat.completions.calls[0]["model"] == "accounts/x/big-70b"


def test_reasoning_model_gets_low_effort_knob(monkeypatch):
    monkeypatch.delenv("CHEAP_MODEL", raising=False)
    # default probe: 150 billed for "4" (overhead 149) -> tries reasoning_effort low
    # low-effort probe: 20 billed (overhead 19 <= 30) -> selected with the knob
    client = FakeClient([probe_response("4", 150), probe_response("4", 20)])
    model, extra = pick_working_model(client, ["accounts/x/reasoner-8b"])
    assert model == "accounts/x/reasoner-8b"
    assert extra == {"reasoning_effort": "low"}
    assert client.chat.completions.calls[1]["extra_body"] == {"reasoning_effort": "low"}


def test_low_effort_rejected_keeps_default(monkeypatch):
    monkeypatch.delenv("CHEAP_MODEL", raising=False)
    # model A: high overhead, low-effort knob rejected -> stays candidate at 149
    # model B: lean -> wins
    client = FakeClient([
        probe_response("4", 150), RuntimeError("unknown param reasoning_effort"),
        probe_response("4", 10),
    ])
    model, extra = pick_working_model(client, ["accounts/x/reasoner-2b", "accounts/x/plain-8b"])
    assert model == "accounts/x/plain-8b"
    assert extra is None


def test_answer_task_passes_extra_body():
    client = FakeClient([fake_response("4")])
    task = {"task_id": "t1", "prompt": "2+2?"}
    answer_task(client, "m-x", task, FUTURE, extra_body={"reasoning_effort": "low"})
    assert client.chat.completions.calls[0]["extra_body"] == {"reasoning_effort": "low"}
```

- [ ] **Step 2: Run to verify failures**

Run: `.venv/bin/pytest tests/test_hardening.py -v`
Expected: the updated/new tests FAIL (tuple return, extra_body, overhead logic all missing)

- [ ] **Step 3: Implement in `agent/main.py`**

Replace `PROBE_MAX_TOKENS = 16` with:

```python
PROBE_MAX_TOKENS = 200  # headroom so hidden reasoning shows up in the overhead measurement
LOW_OVERHEAD_TOKENS = 30  # a model this lean is good enough; stop spending probe tokens
PROBE_PROMPT = "What is 2+2? Answer with just the number."
```

Replace the whole `pick_working_model` with:

```python
def _probe(client, model, extra_body=None):
    """One tiny call; returns billed-minus-visible completion-token overhead.

    Reasoning models bill hidden thinking as completion tokens — on the
    scored eval the overhead was ~2/3 of the whole bill, so it outranks
    parameter count when choosing the cheapest usable model.
    """
    kwargs = {"extra_body": extra_body} if extra_body else {}
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROBE_PROMPT}],
        max_tokens=PROBE_MAX_TOKENS,
        timeout=PROBE_TIMEOUT_SECONDS,
        **kwargs,
    )
    content = (resp.choices[0].message.content or "").strip()
    usage = getattr(resp, "usage", None)
    billed = (usage.completion_tokens or 0) if usage else 0
    return billed - max(1, len(content) // 4)


def pick_working_model(client, allowed_models):
    """Returns (model_id, extra_body|None): cheapest usable model by measured
    token overhead; never a model that can't actually answer a chat call."""
    ranked = sorted(allowed_models, key=parse_model_size)  # stable: ties keep list order
    override = os.environ.get("CHEAP_MODEL")
    if override in allowed_models:
        ranked = [override] + [x for x in ranked if x != override]
    candidates = []  # (overhead, rank_index, model, extra_body)
    for i, model in enumerate(ranked):
        try:
            overhead = _probe(client, model)
        except Exception as exc:  # noqa: BLE001 — failed probe just means next model
            log(f"WARN: model {model} failed probe: {type(exc).__name__}: {exc}")
            continue
        extra = None
        if overhead > LOW_OVERHEAD_TOKENS:
            try:  # reasoning model: does it accept a low-effort knob?
                low = _probe(client, model, extra_body={"reasoning_effort": "low"})
                if low < overhead:
                    overhead, extra = low, {"reasoning_effort": "low"}
            except Exception:  # noqa: BLE001 — knob rejected; keep default behavior
                pass
        log(f"probe: {model} overhead={overhead} extra={extra}")
        candidates.append((overhead, i, model, extra))
        if overhead <= LOW_OVERHEAD_TOKENS:
            break  # lean enough; stop spending probe tokens
    if not candidates:
        log("WARN: no model passed the probe; using cheapest by name anyway")
        return ranked[0], None
    _, _, model, extra = min(candidates)
    return model, extra
```

In `answer_task`, change the signature to `def answer_task(client, model, task, deadline, extra_body=None):` and the create call to include the extras:

```python
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": task["prompt"]},
                ],
                max_tokens=max_tokens,
                timeout=timeout,
                **({"extra_body": extra_body} if extra_body else {}),
            )
```

In `main()`, change the pick line to:

```python
        model, extra_body = pick_working_model(client, cfg["allowed_models"])
```

and the submit line to:

```python
            futures = [pool.submit(answer_task, client, model, t, deadline, extra_body) for t in answerable]
```

- [ ] **Step 4: Run full suite**

Run: `.venv/bin/pytest -q`
Expected: 73 passed (all main-flow tests still pass: their probe outcome `fake_response("OK", 10, 5)` has overhead 5−1=4 ≤ 30 so exactly one probe is consumed, as before)

- [ ] **Step 5: Live sanity check (our key, costs pennies)**

Run:
```bash
set -a && source .env && set +a && ALLOWED_MODELS="accounts/fireworks/models/gpt-oss-120b,accounts/fireworks/models/kimi-k2p5" .venv/bin/python - <<'EOF'
import os
from openai import OpenAI
from agent.main import pick_working_model
client = OpenAI(base_url=os.environ["FIREWORKS_BASE_URL"], api_key=os.environ["FIREWORKS_API_KEY"], max_retries=0)
print(pick_working_model(client, os.environ["ALLOWED_MODELS"].split(",")))
EOF
```
Expected: prints a `probe: ... overhead=N` line per model tried and a `(model, extra)` tuple; record which model wins and its overhead in the commit message.

- [ ] **Step 6: Commit**

```bash
git add agent/main.py tests/test_hardening.py
git commit -m "feat(p2): reasoning-tax-aware model selection with low-effort knob"
```

---

### Task 4: Token diet wired into answer_task → ship v3 (Tier 2)

**Files:**
- Modify: `agent/main.py` (`SYSTEM_PROMPT`, `answer_task` user message)
- Test: `tests/test_telemetry.py` (add constraint-wiring test)

**Interfaces:**
- Consumes: `build_user_message` (Task 1), category stamping (Task 2)
- Produces: every Fireworks call sends `build_user_message(prompt, category)`; new `SYSTEM_PROMPT = "Answer in English. Be accurate and brief."`

- [ ] **Step 1: Write the failing test** (append to `tests/test_telemetry.py`)

```python
def test_fireworks_call_includes_category_constraint():
    from agent.router import CONSTRAINTS
    client = FakeClient([fake_response("Positive — praises battery.")])
    task = {"task_id": "t1", "prompt": "Classify the sentiment: great battery.",
            "category": "sentiment"}
    answer_task(client, "m-x", task, FUTURE)
    sent = client.chat.completions.calls[0]["messages"][1]["content"]
    assert sent.startswith("Classify the sentiment: great battery.")
    assert CONSTRAINTS["sentiment"] in sent


def test_system_prompt_is_short():
    assert len(m.SYSTEM_PROMPT) <= 60
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_telemetry.py -v`
Expected: 2 new FAIL

- [ ] **Step 3: Implement**

In `agent/main.py` replace the SYSTEM_PROMPT line with:

```python
SYSTEM_PROMPT = "Answer in English. Be accurate and brief."
```

In `answer_task`, change the user message line to:

```python
                    {"role": "user", "content": build_user_message(task["prompt"], result["category"])},
```

- [ ] **Step 4: Full suite**

Run: `.venv/bin/pytest -q` → 75 passed

- [ ] **Step 5: Live validation on practice + evalsim**

```bash
set -a && source .env && set +a && mkdir -p /tmp/p2t2 && cp practice_tasks.json /tmp/p2t2/tasks.json && AGENT_INPUT=/tmp/p2t2/tasks.json AGENT_OUTPUT=/tmp/p2t2/results.json .venv/bin/python agent/main.py
```
Expected: stats line with total_tokens meaningfully below the previous ~2,111 practice-run bill; read all 8 answers (`cat /tmp/p2t2/results.json | .venv/bin/python -m json.tool`); the sentiment answer must now carry a justification. Record the token number.

- [ ] **Step 6: Commit, then USER ACTION — build, verify, push v3, re-save submission**

```bash
git add agent/main.py tests/test_telemetry.py
git commit -m "feat(p2): token diet — short system prompt + per-category constraints (v3)"
```
**USER ACTION** (run each with `!`):
```
! sudo docker buildx build --platform linux/amd64 --load -t routing-agent -t caverraaa/routing-agent:v3-tokendiet .
! sudo docker run --rm --memory=4g --cpus=2 --env-file .env -v $PWD/input:/input -v $PWD/output:/output routing-agent
! sudo docker push caverraaa/routing-agent:v3-tokendiet
```
Then re-save the lablab submission to `v3-tokendiet` NOW (early = better odds of being scored). Do not wait for later tiers.

---

### Task 5: Golden set (Tier 3)

**Files:**
- Create: `eval/make_golden.py`, `eval/golden_tasks.json` (generated then human-verified)
- Test: verification is human + a schema check inside `make_golden.py --check`

**Interfaces:**
- Consumes: `.env` (our own key), `agent/router.py` categories
- Produces: `eval/golden_tasks.json`: list of `{"task_id": "g-<cat>-<n>", "category": <cat>, "prompt": str, "expected_intent": str}` — 12 each for sentiment/ner/factual/logic, 6 each for math/summarisation/code_debug/code_gen (72 total)

- [ ] **Step 1: Write `eval/make_golden.py`**

```python
"""Draft golden tasks with a strong model; a human verifies before commit.

Usage:
  python eval/make_golden.py            # generate eval/golden_tasks.json draft
  python eval/make_golden.py --check    # schema-validate the (edited) file
"""
import json
import os
import sys

COUNTS = {"sentiment": 12, "ner": 12, "factual": 12, "logic": 12,
          "math": 6, "summarisation": 6, "code_debug": 6, "code_gen": 6}

SEEDS = {
    "sentiment": ("Classify the sentiment of this review: The battery life is great, "
                  "but the screen scratches too easily.",
                  "Label 'mixed' (or equivalent) plus a justification naming the positive and negative aspects."),
    "ner": ("Extract all named entities and their types from: Maria Sanchez joined "
            "Fireworks AI in Berlin last March.",
            "Identifies Maria Sanchez=PERSON, Fireworks AI=ORG, Berlin=LOCATION, last March=DATE."),
    "factual": ("What is the capital of Australia, and what body of water is it near?",
                "Canberra; Lake Burley Griffin."),
    "logic": ("Three friends, Sam, Jo, and Lee, each own a different pet: cat, dog, bird. "
              "Sam does not own the bird. Jo owns the dog. Who owns the cat?",
              "Sam owns the cat."),
    "math": ("A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. "
             "How many items remain?", "144."),
    "summarisation": ("Summarize the following in exactly one sentence: <paragraph>",
                      "Exactly one sentence covering the main points."),
    "code_debug": ("This function should return the max of a list but has a bug: "
                   "def get_max(nums): return nums[0]. Find and fix it.",
                   "Corrected function that iterates and returns the true maximum."),
    "code_gen": ("Write a Python function that returns the second-largest number in a "
                 "list, handling duplicates correctly.",
                 "Working function returning second-largest distinct value, handles duplicates."),
}

GEN_PROMPT = """Generate {n} new tasks of the category "{cat}" for evaluating an AI assistant.
Model them on this example task (same difficulty and style, different content):
Task: {seed_prompt}
Expected: {seed_intent}

Return ONLY a JSON array of objects: {{"prompt": "...", "expected_intent": "..."}}.
expected_intent must state the objectively correct answer or the rubric a judge can verify."""


def generate():
    from openai import OpenAI
    client = OpenAI(base_url=os.environ["FIREWORKS_BASE_URL"],
                    api_key=os.environ["FIREWORKS_API_KEY"], max_retries=0)
    judge_model = os.environ.get("JUDGE_MODEL", "accounts/fireworks/models/kimi-k2p5")
    out = []
    for cat, n in COUNTS.items():
        seed_prompt, seed_intent = SEEDS[cat]
        out.append({"task_id": f"g-{cat}-0", "category": cat,
                    "prompt": seed_prompt, "expected_intent": seed_intent})
        resp = client.chat.completions.create(
            model=judge_model, max_tokens=4096, timeout=120,
            messages=[{"role": "user", "content": GEN_PROMPT.format(
                n=n - 1, cat=cat, seed_prompt=seed_prompt, seed_intent=seed_intent)}])
        text = resp.choices[0].message.content
        drafted = json.loads(text[text.index("["):text.rindex("]") + 1])
        for i, d in enumerate(drafted[: n - 1], start=1):
            out.append({"task_id": f"g-{cat}-{i}", "category": cat,
                        "prompt": d["prompt"], "expected_intent": d["expected_intent"]})
    with open("eval/golden_tasks.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(out)} tasks — HUMAN-VERIFY every expected_intent before committing")


def check():
    tasks = json.load(open("eval/golden_tasks.json", encoding="utf-8"))
    from collections import Counter
    counts = Counter(t["category"] for t in tasks)
    assert counts == Counter(COUNTS), f"count mismatch: {counts}"
    for t in tasks:
        assert t["task_id"] and t["prompt"] and t["expected_intent"], t
    ids = [t["task_id"] for t in tasks]
    assert len(ids) == len(set(ids)), "duplicate task_ids"
    print(f"OK: {len(tasks)} tasks, counts {dict(counts)}")


if __name__ == "__main__":
    check() if "--check" in sys.argv else generate()
```

- [ ] **Step 2: Generate the draft**

```bash
set -a && source .env && set +a && .venv/bin/python eval/make_golden.py
```
Expected: `wrote 72 tasks — HUMAN-VERIFY...`

- [ ] **Step 3: USER ACTION — verify expected answers**

Human reviews `eval/golden_tasks.json`, fixing any wrong `expected_intent` (especially math answers — recompute each one). Timebox: 30 min, prioritize the four candidate-local categories.

- [ ] **Step 4: Schema check + commit**

```bash
.venv/bin/python eval/make_golden.py --check
git add eval/make_golden.py eval/golden_tasks.json
git commit -m "feat(p2): 72-task golden set (drafted by model, human-verified)"
```

---

### Task 6: Binary-verdict judge (Tier 3)

**Files:**
- Create: `eval/local_judge.py`
- Test: `tests/test_judge.py`

**Interfaces:**
- Consumes: `eval/golden_tasks.json` (Task 5), `.env`
- Produces: `score_results(golden: list, results: list, verdicts: dict[str,bool]) -> dict` pure aggregation; `judge_prompt(task, answer) -> str`; `parse_verdict(text: str) -> bool`; CLI `python eval/local_judge.py <results.json> [--threshold 95]` printing per-category counts + global %, exit 1 below threshold

- [ ] **Step 1: Write the failing tests**

Create `tests/test_judge.py`:

```python
import sys

sys.path.insert(0, "eval")
from local_judge import judge_prompt, parse_verdict, score_results  # noqa: E402

GOLDEN = [
    {"task_id": "g-math-0", "category": "math", "prompt": "2+2?", "expected_intent": "4."},
    {"task_id": "g-math-1", "category": "math", "prompt": "3+3?", "expected_intent": "6."},
    {"task_id": "g-ner-0", "category": "ner", "prompt": "x", "expected_intent": "y"},
]


def test_parse_verdict_yes_no():
    assert parse_verdict("YES") is True
    assert parse_verdict("The answer satisfies the rubric. Verdict: YES.") is True
    assert parse_verdict("NO — the label is missing justification") is False
    assert parse_verdict("") is False  # unparseable counts as NO (conservative)


def test_judge_prompt_contains_rubric_and_answer():
    p = judge_prompt(GOLDEN[0], "it is 4")
    assert "2+2?" in p and "4." in p and "it is 4" in p and "YES or NO" in p


def test_score_results_aggregates_per_category():
    results = [{"task_id": "g-math-0", "answer": "4"},
               {"task_id": "g-math-1", "answer": "7"},
               {"task_id": "g-ner-0", "answer": "y"}]
    verdicts = {"g-math-0": True, "g-math-1": False, "g-ner-0": True}
    s = score_results(GOLDEN, results, verdicts)
    assert s["per_category"]["math"] == (1, 2)
    assert s["per_category"]["ner"] == (1, 1)
    assert round(s["global_pct"], 1) == 66.7


def test_missing_answer_counts_as_no():
    s = score_results(GOLDEN, [], {})
    assert s["global_pct"] == 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_judge.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'local_judge'`

- [ ] **Step 3: Write `eval/local_judge.py`**

```python
"""Binary-verdict LLM judge: scores any results.json against the golden set.

Usage: python eval/local_judge.py <results.json> [--golden eval/golden_tasks.json]
                                  [--threshold 95]
Uses OUR OWN Fireworks key (.env) — local judging costs nothing on the leaderboard.
"""
import argparse
import json
import os
import re
import sys

_VERDICT_RE = re.compile(r"\b(YES|NO)\b", re.IGNORECASE)


def judge_prompt(task, answer):
    return (f"Task given to an AI assistant: {task['prompt']}\n"
            f"Expected (rubric): {task['expected_intent']}\n"
            f"Assistant's answer: {answer}\n\n"
            "Does the answer satisfy the rubric? Reply YES or NO.")


def parse_verdict(text):
    m = _VERDICT_RE.search(text or "")
    return bool(m and m.group(1).upper() == "YES")


def score_results(golden, results, verdicts):
    answers = {r["task_id"]: r.get("answer", "") for r in results}
    per_category = {}
    yes_total = 0
    for t in golden:
        cat = t["category"]
        ok = bool(answers.get(t["task_id"])) and verdicts.get(t["task_id"], False)
        y, n = per_category.get(cat, (0, 0))
        per_category[cat] = (y + (1 if ok else 0), n + 1)
        yes_total += 1 if ok else 0
    return {"per_category": per_category,
            "global_pct": 100.0 * yes_total / len(golden) if golden else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--golden", default="eval/golden_tasks.json")
    ap.add_argument("--threshold", type=float, default=95.0)
    args = ap.parse_args()

    golden = json.load(open(args.golden, encoding="utf-8"))
    results = json.load(open(args.results, encoding="utf-8"))
    answers = {r["task_id"]: r.get("answer", "") for r in results}

    from openai import OpenAI
    client = OpenAI(base_url=os.environ["FIREWORKS_BASE_URL"],
                    api_key=os.environ["FIREWORKS_API_KEY"], max_retries=0)
    judge_model = os.environ.get("JUDGE_MODEL", "accounts/fireworks/models/kimi-k2p5")

    verdicts = {}
    for t in golden:
        answer = answers.get(t["task_id"], "")
        if not answer:
            verdicts[t["task_id"]] = False
            continue
        resp = client.chat.completions.create(
            model=judge_model, max_tokens=512, timeout=30,
            messages=[{"role": "user", "content": judge_prompt(t, answer)}])
        verdicts[t["task_id"]] = parse_verdict(resp.choices[0].message.content)

    s = score_results(golden, results, verdicts)
    for cat, (y, n) in sorted(s["per_category"].items()):
        print(f"{cat:>14}: {y}/{n}")
    print(f"{'GLOBAL':>14}: {s['global_pct']:.1f}%  (threshold {args.threshold}%)")
    sys.exit(0 if s["global_pct"] >= args.threshold else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests, then full suite**

Run: `.venv/bin/pytest tests/test_judge.py -v` → 4 passed; `.venv/bin/pytest -q` → 79 passed

- [ ] **Step 5: Live baseline — judge v3 (cloud-only) on the golden set**

```bash
set -a && source .env && set +a && mkdir -p /tmp/p2golden && cp eval/golden_tasks.json /tmp/p2golden/tasks.json && ALLOWED_MODELS="accounts/fireworks/models/gpt-oss-120b" AGENT_INPUT=/tmp/p2golden/tasks.json AGENT_OUTPUT=/tmp/p2golden/results_cloud.json .venv/bin/python agent/main.py && .venv/bin/python eval/local_judge.py /tmp/p2golden/results_cloud.json
```
Expected: per-category counts + global % (this is the **cloud baseline column** of the routing table). If global < 95%, STOP and fix answer quality before any Tier 4 work — v3 must be re-verified too.

- [ ] **Step 6: Commit**

```bash
git add eval/local_judge.py tests/test_judge.py
git commit -m "feat(p2): binary-verdict judge harness (router-table generator + submission gate)"
```

---

### Task 7: Speed spike — go/no-go (Tier 4 gate)

**Files:**
- Create: `eval/spike_local.py`
- Test: manual run under 2-core limit

**Interfaces:**
- Consumes: nothing from the repo
- Produces: GO/NO-GO decision recorded in the ledger; `models/gemma-2-2b-it-Q4_K_M.gguf` downloaded locally (`models/` gitignored)

- [ ] **Step 1: gitignore + dev dependency**

Append `models/` to `.gitignore`. Append `llama-cpp-python` to `requirements-dev.txt` (dev venv only; the v4 image installs its own).
```bash
.venv/bin/pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```
(If the CPU wheel index fails, build from source: `sudo apt-get install -y build-essential cmake` then plain `pip install llama-cpp-python`.)

- [ ] **Step 2: Write `eval/spike_local.py`**

```python
"""Go/no-go: Gemma-2-2B-it Q4_K_M on 2 CPU threads. Run under taskset -c 0,1.

GO requires: load <= 40s AND slowest answer <= 25s.
"""
import time

MODEL = "models/gemma-2-2b-it-Q4_K_M.gguf"
PROMPTS = [
    ("sentiment", "Classify the sentiment of this review and justify in one sentence: "
                  "The checkout was seamless but the product arrived cracked."),
    ("ner", "Extract all named entities and their types, one per line: On 14 February 2025, "
            "Dr. Amara Okafor of the European Space Agency presented at MIT in Cambridge."),
]

t0 = time.monotonic()
from llama_cpp import Llama  # noqa: E402
llm = Llama(model_path=MODEL, n_ctx=2048, n_threads=2, verbose=False)
load_s = time.monotonic() - t0
print(f"load: {load_s:.1f}s")

worst = 0.0
for name, prompt in PROMPTS:
    t = time.monotonic()
    out = llm.create_chat_completion(messages=[{"role": "user", "content": prompt}],
                                     max_tokens=160, temperature=0.0)
    dt = time.monotonic() - t
    worst = max(worst, dt)
    text = out["choices"][0]["message"]["content"].strip()
    print(f"{name}: {dt:.1f}s, {len(text)} chars -> {text[:100]!r}")

print(f"VERDICT: {'GO' if load_s <= 40 and worst <= 25 else 'NO-GO'} "
      f"(load {load_s:.1f}s, worst answer {worst:.1f}s)")
```

- [ ] **Step 3: Download the model and run the spike**

```bash
mkdir -p models && curl -L -o models/gemma-2-2b-it-Q4_K_M.gguf "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf"
taskset -c 0,1 .venv/bin/python eval/spike_local.py
```
Expected: `VERDICT: GO (...)`. **If NO-GO: Tier 4 is cancelled — Phase 2 ships as v3; skip to Task 11.**

- [ ] **Step 4: Commit**

```bash
git add .gitignore requirements-dev.txt eval/spike_local.py
git commit -m "feat(p2): local-model speed spike (go/no-go gate)"
```

---

### Task 8: `agent/local_model.py` (Tier 4)

**Files:**
- Create: `agent/local_model.py`
- Test: `tests/test_local_model.py`

**Interfaces:**
- Consumes: nothing from the repo (llama_cpp imported lazily inside `__init__`)
- Produces: `LocalModel(path=..., llama_factory=None, n_threads=2)`; `generate(user_text: str, max_tokens: int = 160) -> str` (thread-safe via lock; Gemma has NO system role — callers pass one merged user message); `classify(prompt: str, categories: tuple) -> str | None` (single-word category or None); module constants `LOCAL_MAX_TOKENS = 160`, `CLASSIFY_MAX_TOKENS = 8`, `DEFAULT_MODEL_PATH` (env `LOCAL_MODEL_PATH`, default `/app/models/gemma-2-2b-it-Q4_K_M.gguf`)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_local_model.py`:

```python
import threading

from agent.local_model import CLASSIFY_MAX_TOKENS, LOCAL_MAX_TOKENS, LocalModel


class FakeLlama:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def create_chat_completion(self, **kwargs):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        with self._lock:
            self.active -= 1
        return {"choices": [{"message": {"content": reply}}]}


def make(replies):
    fake = FakeLlama(replies)
    lm = LocalModel(path="unused.gguf", llama_factory=lambda **kw: fake)
    return lm, fake


def test_generate_returns_stripped_text():
    lm, fake = make(["  Positive — praises battery.  "])
    assert lm.generate("classify this") == "Positive — praises battery."
    call = fake.calls[0]
    assert call["max_tokens"] == LOCAL_MAX_TOKENS
    assert call["messages"] == [{"role": "user", "content": "classify this"}]  # no system role


def test_generate_custom_cap():
    lm, fake = make(["x"])
    lm.generate("p", max_tokens=64)
    assert fake.calls[0]["max_tokens"] == 64


def test_classify_valid_word():
    lm, _ = make(["sentiment"])
    assert lm.classify("is this good or bad?", ("sentiment", "ner")) == "sentiment"


def test_classify_normalizes_punctuation_and_case():
    lm, _ = make([" Sentiment. "])
    assert lm.classify("x", ("sentiment", "ner")) == "sentiment"


def test_classify_garbage_returns_none():
    lm, _ = make(["I think this could be several things"])
    assert lm.classify("x", ("sentiment", "ner")) is None


def test_classify_uses_small_token_cap():
    lm, fake = make(["sentiment"])
    lm.classify("x", ("sentiment",))
    assert fake.calls[0]["max_tokens"] == CLASSIFY_MAX_TOKENS


def test_concurrent_calls_serialize():
    lm, fake = make(["a"] * 8)
    threads = [threading.Thread(target=lm.generate, args=("p",)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert fake.max_active == 1  # the lock never admits two generations at once
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_local_model.py -v`
Expected: `ModuleNotFoundError: No module named 'agent.local_model'`

- [ ] **Step 3: Write `agent/local_model.py`**

```python
"""Gemma-2-2B-it via llama-cpp, CPU-only, serialized behind a lock.

llama.cpp contexts are not thread-safe: exactly one generation runs at a
time; the Fireworks worker pool is unaffected. Gemma has no system role,
so callers pass a single merged user message. Generation time is bounded
by construction via small max_tokens caps (no kill-timer exists in
llama-cpp) — the speed spike validates the worst case fits 30 s/response.
"""
import os
import threading

DEFAULT_MODEL_PATH = os.environ.get(
    "LOCAL_MODEL_PATH", "/app/models/gemma-2-2b-it-Q4_K_M.gguf")
LOCAL_MAX_TOKENS = 160
CLASSIFY_MAX_TOKENS = 8


class LocalModel:
    def __init__(self, path=DEFAULT_MODEL_PATH, llama_factory=None, n_threads=2):
        if llama_factory is None:
            from llama_cpp import Llama  # lazy: not installed in the v3 image
            llama_factory = Llama
        self._lock = threading.Lock()
        self._llm = llama_factory(model_path=path, n_ctx=2048,
                                  n_threads=n_threads, verbose=False)

    def generate(self, user_text, max_tokens=LOCAL_MAX_TOKENS):
        with self._lock:
            out = self._llm.create_chat_completion(
                messages=[{"role": "user", "content": user_text}],
                max_tokens=max_tokens, temperature=0.0)
        return (out["choices"][0]["message"]["content"] or "").strip()

    def classify(self, prompt, categories):
        instruction = ("Classify this task. Answer with exactly one word from: "
                       + ", ".join(categories) + ".\nTask: " + prompt[:500]
                       + "\nCategory:")
        word = self.generate(instruction, max_tokens=CLASSIFY_MAX_TOKENS)
        word = word.lower().strip(" .:\n\"'")
        return word if word in categories else None
```

- [ ] **Step 4: Run tests, full suite**

Run: `.venv/bin/pytest tests/test_local_model.py -v` → 7 passed; `.venv/bin/pytest -q` → 86 passed

- [ ] **Step 5: Commit**

```bash
git add agent/local_model.py tests/test_local_model.py
git commit -m "feat(p2): serialized llama-cpp local model wrapper with classify()"
```

---

### Task 9: Routing table + lane branch in `answer_task` (Tier 4)

**Files:**
- Modify: `agent/main.py` (`load_routing_table`, `answer_task` local lane, `main()` local init)
- Create: `agent/routing_table.json` (initial: all fireworks)
- Test: `tests/test_routing.py`

**Interfaces:**
- Consumes: `LocalModel` (Task 8), `build_user_message`/`categorize`/`CATEGORIES` (Task 1)
- Produces: `load_routing_table(path) -> dict[str,str]` (only `"local"|"fireworks"` values survive; missing/broken file → `{}`); `answer_task(client, model, task, deadline, extra_body=None, local=None, routing=None)` — LOCAL lane when `routing.get(category) == "local"` and `local` is not None; local failure/empty falls through to the Fireworks path; local success → `lane="local"`, zero tokens; `unknown` category with `local` present → one `classify()` attempt, result still routed via table (default fireworks)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_routing.py`:

```python
import json
import time

import agent.main as m
from agent.main import answer_task, load_routing_table

from tests.test_answer_task import FakeClient, fake_response

FUTURE = time.monotonic() + 3600


class FakeLocal:
    def __init__(self, reply="", classify_reply=None):
        self.reply = reply
        self.classify_reply = classify_reply
        self.generate_calls = []

    def generate(self, user_text, max_tokens=160):
        self.generate_calls.append(user_text)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply

    def classify(self, prompt, categories):
        return self.classify_reply


def task(cat="sentiment"):
    return {"task_id": "t1", "prompt": "Classify: great!", "category": cat}


def test_load_routing_table(tmp_path):
    p = tmp_path / "rt.json"
    p.write_text(json.dumps({"sentiment": "local", "math": "fireworks", "bad": "nope"}))
    assert load_routing_table(str(p)) == {"sentiment": "local", "math": "fireworks"}
    assert load_routing_table(str(tmp_path / "missing.json")) == {}


def test_local_lane_success_zero_tokens():
    client = FakeClient([])  # any API call would crash on empty outcomes
    local = FakeLocal(reply="Positive — enthusiastic tone.")
    r = answer_task(client, "m-x", task(), FUTURE,
                    local=local, routing={"sentiment": "local"})
    assert r["answer"] == "Positive — enthusiastic tone."
    assert r["lane"] == "local"
    assert r["prompt_tokens"] == 0 and r["completion_tokens"] == 0
    assert client.chat.completions.calls == []


def test_local_empty_falls_back_to_fireworks():
    client = FakeClient([fake_response("Positive.")])
    local = FakeLocal(reply="")
    r = answer_task(client, "m-x", task(), FUTURE,
                    local=local, routing={"sentiment": "local"})
    assert r["answer"] == "Positive."
    assert r["lane"] == "fireworks"


def test_local_exception_falls_back_to_fireworks():
    client = FakeClient([fake_response("Positive.")])
    local = FakeLocal(reply=RuntimeError("llama crashed"))
    r = answer_task(client, "m-x", task(), FUTURE,
                    local=local, routing={"sentiment": "local"})
    assert r["answer"] == "Positive."
    assert r["lane"] == "fireworks"


def test_category_routed_to_fireworks_never_touches_local():
    client = FakeClient([fake_response("42")])
    local = FakeLocal(reply="should not be used")
    r = answer_task(client, "m-x", task("math"), FUTURE,
                    local=local, routing={"sentiment": "local"})
    assert r["lane"] == "fireworks"
    assert local.generate_calls == []


def test_unknown_category_classified_then_routed():
    client = FakeClient([])
    local = FakeLocal(reply="Positive — nice.", classify_reply="sentiment")
    r = answer_task(client, "m-x", task("unknown"), FUTURE,
                    local=local, routing={"sentiment": "local"})
    assert r["lane"] == "local"
    assert r["category"] == "sentiment"


def test_unknown_unparseable_classification_goes_cloud():
    client = FakeClient([fake_response("cloud answer")])
    local = FakeLocal(classify_reply=None)
    r = answer_task(client, "m-x", task("unknown"), FUTURE,
                    local=local, routing={"sentiment": "local"})
    assert r["lane"] == "fireworks"
    assert r["answer"] == "cloud answer"


def test_no_local_model_behaves_as_before():
    client = FakeClient([fake_response("A")])
    r = answer_task(client, "m-x", task(), FUTURE)
    assert r["answer"] == "A" and r["lane"] == "fireworks"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_routing.py -v`
Expected: FAIL — `ImportError: cannot import name 'load_routing_table'` etc.

- [ ] **Step 3: Implement in `agent/main.py`**

Add below `write_snapshot`:

```python
def load_routing_table(path):
    """category -> "local"|"fireworks". Missing/invalid file means all-cloud."""
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return {k: v for k, v in raw.items() if v in ("local", "fireworks")}
    except Exception:  # noqa: BLE001
        return {}
```

Change `answer_task`'s signature to
`def answer_task(client, model, task, deadline, extra_body=None, local=None, routing=None):`
and insert the lane branch immediately after the `result = {...}` initialization (before the Fireworks attempts loop):

```python
    routing = routing or {}
    if local is not None and result["category"] == "unknown":
        try:
            guessed = local.classify(task["prompt"], tuple(routing))
            if guessed:
                result["category"] = guessed
        except Exception:  # noqa: BLE001 — classification is best-effort only
            pass
    if (local is not None and routing.get(result["category"]) == "local"
            and time.monotonic() < deadline):
        try:
            text = local.generate(build_user_message(task["prompt"], result["category"]))
            if text:
                result["answer"] = text
                result["lane"] = "local"
                return result  # zero Fireworks tokens
        except Exception as exc:  # noqa: BLE001 — local failure falls back to cloud
            log(f"WARN: local lane failed for {task['task_id']}: {type(exc).__name__}: {exc}")
```

In `main()`, after the `model, extra_body = pick_working_model(...)` line, add local init:

```python
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
```

and change the submit line to:

```python
            futures = [pool.submit(answer_task, client, model, t, deadline,
                                   extra_body, local, routing) for t in answerable]
```

Create `agent/routing_table.json` (all-cloud until Task 10's measurements say otherwise):

```json
{"sentiment": "fireworks", "ner": "fireworks", "factual": "fireworks", "logic": "fireworks", "math": "fireworks", "summarisation": "fireworks", "code_debug": "fireworks", "code_gen": "fireworks"}
```

- [ ] **Step 4: Full suite**

Run: `.venv/bin/pytest -q` → 94 passed

- [ ] **Step 5: Commit**

```bash
git add agent/main.py agent/routing_table.json tests/test_routing.py
git commit -m "feat(p2): judge-gated local lane in answer_task with cloud fallback"
```

---

### Task 10: Measure, fill routing table, v4 image, judge gate (Tier 4 ship)

**Files:**
- Modify: `agent/routing_table.json` (from measured data), `Dockerfile` (v4)
- Create: none

**Interfaces:**
- Consumes: everything above
- Produces: pushed `caverraaa/routing-agent:v4-hybrid` IFF full-container judge ≥95%

- [ ] **Step 1: Local-lane golden run (dev machine, 2 cores)**

```bash
set -a && source .env && set +a && printf '%s' '{"sentiment": "local", "ner": "local", "factual": "local", "logic": "local", "math": "fireworks", "summarisation": "fireworks", "code_debug": "fireworks", "code_gen": "fireworks"}' > /tmp/rt_all_candidates.json && LOCAL_MODEL_PATH=models/gemma-2-2b-it-Q4_K_M.gguf ROUTING_TABLE=/tmp/rt_all_candidates.json AGENT_INPUT=/tmp/p2golden/tasks.json AGENT_OUTPUT=/tmp/p2golden/results_local.json taskset -c 0,1 .venv/bin/python agent/main.py && .venv/bin/python eval/local_judge.py /tmp/p2golden/results_local.json --threshold 0
```
Expected: per-category counts for the local lanes + per-task telemetry with wall-clock in the stats. Record the four candidate-local categories' counts.

- [ ] **Step 2: Apply the count-based combined gate**

For each candidate category: LOCAL iff local ≥10/12 AND (cloud − local) ≤ 1, where cloud counts come from Task 6 Step 5. Also require: slowest local answer in that category ≤25 s (from the telemetry timestamps). Edit `agent/routing_table.json` accordingly; categories that fail stay `"fireworks"`. Commit the table with the measured counts in the commit message:

```bash
git add agent/routing_table.json
git commit -m "feat(p2): routing table from judged data — local: <list>, counts: <numbers>"
```

- [ ] **Step 3: v4 Dockerfile**

Replace `Dockerfile` content with:

```dockerfile
FROM python:3.11-slim

RUN pip install --no-cache-dir llama-cpp-python \
      --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

ADD https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf /app/models/gemma-2-2b-it-Q4_K_M.gguf

COPY agent/ /app/agent/

CMD ["python", "/app/agent/main.py"]
```

```bash
git add Dockerfile
git commit -m "feat(p2): v4 image with baked Gemma-2-2B-it GGUF + llama-cpp CPU wheel"
```

- [ ] **Step 4: USER ACTION — build and full-container validation under grading limits**

```
! sudo docker buildx build --platform linux/amd64 --load -t routing-agent-v4 -t caverraaa/routing-agent:v4-hybrid .
! sudo docker run --rm --memory=4g --cpus=2 --env-file .env -v /tmp/p2golden:/input -v /tmp/p2golden/out:/output routing-agent-v4
```
(first copy golden as input: `mkdir -p /tmp/p2golden/out && cp eval/golden_tasks.json /tmp/p2golden/tasks.json`)
Watch stderr: `local model loaded` must appear well inside 60 s; per-task lines show `lane=local` for routed categories; elapsed comfortably under 9 min.

- [ ] **Step 5: Judge gate**

```bash
.venv/bin/python eval/local_judge.py /tmp/p2golden/out/results.json
```
Expected: `GLOBAL: >= 95%`, exit 0, AND total Fireworks tokens (stats line) meaningfully below the v3 golden-run bill. **If either fails: do NOT push v4 — the submission pointer stays on v3. Stop here and go to Task 11.**

- [ ] **Step 6: USER ACTION — ship v4**

```
! sudo docker push caverraaa/routing-agent:v4-hybrid
```
Re-save the lablab submission to `v4-hybrid`.

---

### Task 11: Freeze ritual (Tier 5 — hard stop T−2.5 h)

**Files:** none (checklist)

- [ ] **Step 1:** `.venv/bin/pytest -q` → all green; `git status` clean; ledger updated.
- [ ] **Step 2: USER ACTION —** final `./run_local.sh` (practice tasks, grading limits) on whichever tag the submission points at; eyeball all 8 answers.
- [ ] **Step 3: USER ACTION —** anonymous-pull verification:
```
! sudo docker pull caverraaa/routing-agent:<final-tag> && sudo docker run --rm --entrypoint python caverraaa/routing-agent:<final-tag> -c "import sys; sys.path.insert(0,'/app'); import agent.main as m; print('fingerprint', m.MAX_TOKENS, hasattr(m,'load_routing_table'))"
```
- [ ] **Step 4:** Confirm the lablab submission page points at the intended tag. **No pointer moves after this check** — a blind swap can only lose the scored fallback.
- [ ] **Step 5:** Write the wrap-up: final routing table, token telemetry from the last golden run, judge scores — this is the demo/README material.

---

## Self-review notes

- Spec coverage: Tier 0 is a user action (Discord), not a code task — called out in Global Constraints and Task 11; Tiers 1–5 map to Tasks 1–11.
- Cut line if time runs short: after Task 4 (v3 shipped) every subsequent task is optional; after Task 6 the judge protects everything; Task 7 NO-GO short-circuits 8–10.
- Test-count checkpoints: 67 → 70 → 73 → 75 → 79 → 86 → 94.

# Phase 1 Design — Baseline Routing Agent

**Project:** AMD Developer Hackathon ACT II, Track 1 — Hybrid Token-Efficient Routing Agent
**Date:** 2026-07-11 · **Deadline:** 2026-07-12 00:00 local
**Status:** Approved by user 2026-07-11

## Goal

A submittable-today baseline: a batch-job Docker container that reads `/input/tasks.json`, sends every prompt to the cheapest model in `ALLOWED_MODELS` through an OpenAI-compatible client pointed at `FIREWORKS_BASE_URL`, writes `/output/results.json`, and exits 0. Boring and bulletproof; Phase 2 (local-model hybrid router) builds on it without restructuring.

## Decisions made (with rationale)

1. **Concurrency: `ThreadPoolExecutor(max_workers=4)`.** Sequential execution blows the 10-minute budget if the hidden set has ~30+ tasks at 15–25 s/response (40 × 20 s ≈ 13 min). Four workers give ~4× headroom (~3.5 min for the same load). The `openai` client is thread-safe.
2. **`pick_cheapest_model`: size-regex + env override.** Optional `CHEAP_MODEL` env var (validated against `ALLOWED_MODELS`, local use only — the harness won't set it) → else smallest parameter count parsed from the model ID string (`8b`, `1.7b`, …) → else first list entry. Never returns an ID outside `ALLOWED_MODELS`. No hardcoded model IDs anywhere.
3. **Output length: `max_tokens=1024`** with system prompt `"Answer in English. Be correct and complete, but concise."` Generous enough for code generation and multi-step math (accuracy gate is priority #1); caps runaway responses that would break the 30 s/response rule. Aggressive token tuning is Phase 3.
4. **Local eval today: schema check + human eyeball.** The smoke test asserts the output schema and prints the 8 practice answers for manual review before any submission (10/hour rate limit). A scripted LLM judge is deferred to Phase 2.

## Repo skeleton

```
agent/main.py            # single entrypoint, stdlib + openai only
Dockerfile               # python:3.11-slim, linux/amd64
requirements.txt         # openai (pinned)
practice_tasks.json      # the 8 practice tasks, verbatim from the participant guide
run_local.sh             # smoke test: build, run with grading-VM limits, assert schema
.dockerignore            # .env, .git, *.pdf, guide files, output/
.gitignore               # .env, input/, output/, __pycache__
docs/superpowers/specs/  # this document
```

The directory is `git init`-ed today; submission requires a public GitHub repo.

## main.py flow

```
load env (fail fast, name the missing var on stderr, exit 1)
read + validate /input/tasks.json
model = pick_cheapest_model(ALLOWED_MODELS)
ThreadPoolExecutor(4) over tasks:
  each task: 1 API call, timeout=20s; on failure 1 retry (2s backoff, timeout=25s)
  per-task try/except → a failed task emits {task_id, answer: ""}
  after each completed task: atomic snapshot write of the FULL results list
wall-clock guard: soft budget 9m00s; tasks not yet dispatched at the
  deadline are emitted with answer:"" (graceful degradation)
finally: final snapshot write + per-run stats to stderr
exit 0 whenever results.json was written, even with some "" answers
```

### Component details

- **Env loading.** `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, `ALLOWED_MODELS` — missing any → exact missing name printed to stderr, exit 1. `ALLOWED_MODELS` split on comma, entries stripped, empties dropped.
- **Task validation.** Input must be a JSON list; each entry needs string `task_id` and `prompt`. A malformed entry is logged to stderr and skipped, but if it has a usable `task_id` it still appears in results with `answer: ""`. One bad entry never kills the run.
- **Timeout math.** Per-response rule is <30 s, so no request may run longer: 20 s first attempt, 25 s retry. Worst case per worker slot-cycle ≈ 47 s; 4 workers × 9-minute soft budget ≈ 45 slot-cycles ≈ capacity for >100 tasks. The 9 m soft budget leaves 60 s margin for startup, final write, exit.
- **Snapshot writes.** Serialize the full current results list to `/output/results.json.tmp`, then `os.replace` onto `/output/results.json`. Writes happen only from the main collector thread (no locking races). The file on disk is always complete, valid JSON — from the first completed task onward.
- **Schema enforcement.** Before every write: coerce `task_id` to str, `answer` to str (never None/null). Exactly one output entry per input task.
- **API client.** `OpenAI(base_url=FIREWORKS_BASE_URL, api_key=FIREWORKS_API_KEY)` is the only HTTP client in the codebase. No answer caching keyed on prompt text anywhere.
- **Stats to stderr** (stdout stays clean): tasks total/answered/failed, prompt+completion token sums from `response.usage`, elapsed wall-clock, model used.

## Dockerfile

```dockerfile
FROM python:3.11-slim
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY agent/ /app/agent/
CMD ["python", "/app/agent/main.py"]
```

- Built with `docker buildx build --platform linux/amd64`.
- No `COPY . .` — `.env` can never reach the image; `.dockerignore` backs this up.
- No model download, no heavy deps: image ~150 MB, cold start ~2–3 s (60 s ready-rule trivially met).

## run_local.sh smoke test

1. `mkdir -p input output`; copy `practice_tasks.json` → `input/tasks.json`
2. `docker buildx build --platform linux/amd64 -t routing-agent .`
3. `docker run --rm --memory=4g --cpus=2 --env-file .env -v $PWD/input:/input -v $PWD/output:/output routing-agent` (mirrors the grading VM)
4. Assert: exit code 0; `output/results.json` exists, parses as JSON, is a list of 8 objects, each with string `task_id` and `answer`; every input `task_id` present
5. Print all 8 answers for the human eyeball accuracy check before any submission

## Failure-status map

| Status | Designed out by |
|---|---|
| PULL_ERROR | buildx `--platform linux/amd64`; public registry push |
| RUNTIME_ERROR | fail-fast env check; per-task try/except; top-level try/finally; exit-0 path |
| TIMEOUT | 20/25 s request timeouts; max 1 retry; 4-worker pool; 9-min soft budget with degradation |
| OUTPUT_MISSING | atomic snapshot after every task + finally-guarded final write |
| INVALID_RESULTS_SCHEMA | schema coercion before every write; smoke test asserts it |
| MODEL_VIOLATION | model chosen only from runtime `ALLOWED_MODELS`; single client via `FIREWORKS_BASE_URL` |
| IMAGE_TOO_LARGE | slim base, no local model in Phase 1 (~150 MB) |
| ACCURACY_GATE_FAILED | 1024 max_tokens (no truncation); practice-task eyeball check before submitting |

## Hard-constraint checklist (verify at every step)

- 4 GB RAM / 2 vCPU / CPU-only: Phase 1 uses ~100 MB RAM; smoke test runs with `--memory=4g --cpus=2`.
- 10-min runtime / 60 s startup / 30 s per response: soft budget 9 m; cold start seconds; request timeouts 20/25 s.
- ≤10 GB compressed, linux/amd64 manifest: slim image, buildx platform flag.
- All Fireworks traffic through `FIREWORKS_BASE_URL`; models only from runtime `ALLOWED_MODELS`.
- No answer hardcoding/caching (eval uses unseen variants).
- English-only answers: enforced in the system prompt.

## Out of scope for Phase 1

Local model, category router, token tuning below the 1024 cap, scripted LLM judge, prompt trimming. Phase 2 swaps only the internals of `answer_task()`; the pool, snapshot, and budget machinery are unchanged.

## Residual risk (accepted)

If the hidden set exceeds ~150 tasks, the pool degrades the tail to `answer: ""`. Nothing in the guide suggests a set that large; adding more workers risks proxy rate-limiting.

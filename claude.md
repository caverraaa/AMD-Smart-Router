# CLAUDE.md — AMD Hackathon ACT II, Track 1: Hybrid Token-Efficient Routing Agent

## Deadline mode
Hard deadline: **July 12, 00:00 local (~17h)**. Prefer the dumb-but-working solution. One change at a time, small testable increments. Flag anything that risks the deadline. Explain non-obvious decisions briefly — the human is backend/data lead and must understand every piece of code.

## What we are building
A **batch-job Docker container** (NOT a web app, NOT an API server):
1. On startup, read `/input/tasks.json`: `[{ "task_id": "t1", "prompt": "..." }, ...]`
2. Answer every task with the cheapest sufficient model.
3. Write `/output/results.json` before exiting: `[{ "task_id": "t1", "answer": "..." }, ...]`
4. Exit 0 on success, non-zero on failure.

## Scoring (what we optimize)
1. **Accuracy gate** (LLM-Judge). Below threshold → excluded from leaderboard. Accuracy is non-negotiable.
2. Among passers: rank ascending by **total Fireworks tokens** (input + output, recorded by the judging proxy). Fewer = better.
3. **Local model inference = ZERO tokens** but full accuracy credit. Zero Fireworks calls is a valid winning strategy (`flagged: ZERO_API_CALLS` is not a failure).

Implication: every prompt sent to Fireworks costs score. Every prompt answered locally is free. Route aggressively local; use Fireworks only where the local model is unreliable.

## Hard runtime constraints (design against these, always)
| Constraint | Value |
|---|---|
| Grading VM | **4 GB RAM, 2 vCPU, CPU-only** (no GPU at eval time) |
| Total runtime | **10 minutes** |
| Container ready | within **60 seconds** of start |
| Per-response | under **30 seconds** |
| Image size | ≤ **10 GB compressed**, must include **linux/amd64** manifest |
| Language | all answers in **English** |
| Local model sizing | 2B–3B 4-bit quant safe; 7B 4-bit fills entire RAM budget — do not use |

## Environment variables (harness-injected — NEVER hardcode, NEVER bundle .env in image)
```python
import os
api_key  = os.environ["FIREWORKS_API_KEY"]           # harness key, not ours
base_url = os.environ["FIREWORKS_BASE_URL"]          # ALL Fireworks calls MUST go through this
models   = os.environ["ALLOWED_MODELS"].split(",")   # exact permitted IDs, read at runtime
```
- Calls bypassing `FIREWORKS_BASE_URL` → not recorded → wasted tokens toward accuracy but risk of scoring anomalies. Never bypass.
- Model outside `ALLOWED_MODELS` → `MODEL_VIOLATION`, submission invalidated.
- Local dev may use a `.env` file, but it must never end up in the image. Add `.env` to `.dockerignore` and `.gitignore`.

## Task categories (must handle all 8)
1. Factual knowledge  2. Mathematical reasoning  3. Sentiment classification  4. Text summarisation (format/length constraints)  5. Named entity recognition  6. Code debugging  7. Logical/deductive reasoning  8. Code generation

No hardcoding or caching of answers — evaluation uses unseen prompt variants.

## Build phases (in this order — do not skip ahead)
**Phase 1 — Baseline (submit first):** read tasks → send every prompt to the cheapest `ALLOWED_MODELS` model via OpenAI-compatible client with `base_url` override → write results. Robust JSON I/O, per-task try/except (one failed task must not kill the run), request timeouts, minimal retries.

**Phase 2 — Hybrid router:** bake a local 2–3B 4-bit model into the image (llama.cpp / llama-cpp-python; prefer **Gemma-2-2B-it** where quality is comparable — $1,000 partner prize for best use of Gemma). Route by category:
- Local (free): sentiment, NER, simple factual, simple logic
- Fireworks: math, code debugging, code generation, anything local is unreliable on
- Router itself must be near-free: keyword/regex heuristics or one tiny local classification pass.

**Phase 3 — Optimizations (only if time remains):** output-length tuning (short answers, but never below what the judge needs), prompt trimming, local-first with Fireworks fallback on low confidence. Log per-run stats (local vs API routing counts, token totals) — needed for demo video/slides.

## Failure modes checklist (design each one out)
- `PULL_ERROR`: image must be public with linux/amd64 manifest (`docker buildx build --platform linux/amd64`)
- `RUNTIME_ERROR`: per-task exception handling; top-level try/finally
- `TIMEOUT`: hard per-request timeouts, capped retries, no infinite loops; budget total wall-clock
- `OUTPUT_MISSING`: write `/output/results.json` even on partial failure (write incrementally or in finally block)
- `INVALID_RESULTS_SCHEMA`: every entry must have both `task_id` and `answer` (strings); validate before writing
- `MODEL_VIOLATION`: only models from `ALLOWED_MODELS`, read at runtime
- `IMAGE_TOO_LARGE`: keep under 10 GB compressed; slim base image, no stray layers
- `ACCURACY_GATE_FAILED`: run local eval on practice tasks before every submission

Submissions rate-limited to **10/hour/team** → always validate locally on the 8 practice tasks before submitting.

## Code conventions
- Python 3.11+, single main entrypoint (`agent/main.py`), stdlib + `openai` client + (Phase 2) `llama-cpp-python`
- Every Fireworks call: explicit timeout, one retry max with backoff, token usage logged
- When proposing code, state: which phase it belongs to, and which constraint it protects (RAM / time / tokens / accuracy)
- No answer caching keyed on prompt text. Routing heuristics on prompt *structure* are fine; memorized answers are not.

## Local dev commands
```bash
# run against practice tasks
mkdir -p input output && cp practice_tasks.json input/tasks.json
docker build --platform linux/amd64 -t routing-agent .
docker run --rm --memory=4g --cpus=2 \
  -v $PWD/input:/input -v $PWD/output:/output \
  --env-file .env routing-agent
# then: python eval/local_judge.py output/results.json
```
Always test with `--memory=4g --cpus=2` to mirror the grading VM.
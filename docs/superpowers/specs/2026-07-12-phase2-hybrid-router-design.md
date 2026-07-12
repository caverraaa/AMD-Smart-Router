# Phase 2 Design — Token Diet + Judged Hybrid Router

**Project:** AMD Developer Hackathon ACT II, Track 1
**Date:** 2026-07-12 · **Competition ends:** ~16 h from design time (evening 2026-07-12 local)
**Status:** Approved by user 2026-07-12
**Baseline:** v2-hardened scored **94.7% accuracy / 8,421 tokens / rank 55**. Everything below is upside only if the accuracy gate is never lost.

## Strategic context (governs all decisions)

- The scoring queue is multi-day backlogged: a resubmission tonight may never be scored before the deadline. We are flying blind on the gate for anything shipped now.
- Pessimistic defaults until Discord answers otherwise: the submission pointer's **last** target is what counts, even unscored, even if it would fail. Therefore: **the pointer only moves to an image that beats ≥95% on our own judge**, and cheap-safe improvements ship EARLY (better odds of being scored) while risky ones ship only behind the judge gate.
- Tiers are ordered by value-per-minute, cheapest first. Each tier is independently shippable; a hard stop after any tier still leaves a valid submission.

## Decisions locked during brainstorming

1. **Local model: Gemma-2-2B-it**, Q4_K_M GGUF (~1.7 GB), via `llama-cpp-python`, CPU-only. Gemma-prize aligned; 2B 4-bit fits 4 GB RAM with room for the agent.
2. **Router: hybrid** — regex/keyword rules first; on no-match, one local-model classification pass (single-word category output); unparseable classification → Fireworks. Misroutes may only ever cost tokens, never accuracy.
3. **Local-trust validation: golden set + LLM judge, count-based combined gate.** A category routes LOCAL iff Gemma scores **≥10/12** on that category **and** cloud − Gemma **≤ 1 task**. (Counts, not percentages — at 12 tasks/category one task = 8.3 points, so point-thresholds are fake precision. Absolute floor + relative gate cover each other's blind spots.)
4. **Judge design:** binary YES/NO per task against a stored `expected_intent` rubric ("Expected: label + justification. Does this answer satisfy that?"), scored by a strong Fireworks model on our own key (local dev cost doesn't count toward the leaderboard). No 0–100 scores.

## Tier 0 — Facts (parallel, timeboxed 20 min)

Discord: (a) do queued-at-deadline submissions count; (b) does a new pointer replace scored v2 if unscored/failing; (c) does "Best Use of Gemma via Fireworks" accept a baked local GGUF (no Gemma exists in the observed model catalog, so local is the only Gemma play regardless). Proceed on pessimistic defaults if unanswered.

## Tier 1 — Telemetry + reasoning-tax probe (~45 min, near-zero risk)

- Per-task stderr line in the collector loop: `task=<id> cat=<category> pt=<prompt_tokens> ct=<completion_tokens> lane=<local|fireworks>`.
- **Reasoning tax:** gpt-oss-120b billed ~5,117 completion tokens for ~1,500 tokens of visible text on the 10-task sim — hidden reasoning is billed. `pick_working_model` gains a second criterion: probe each candidate with one fixed short task, record `completion_tokens − visible_token_estimate`; among probe-passers, prefer lowest overhead, ties → cheapest by size, then list order.
- If the endpoint accepts `reasoning_effort: "low"` (gpt-oss family): probe once with it; keep on success, drop on any 4xx. Never assume.

## Tier 2 — Prompt-level token cuts → ship as v3 EARLY (~1 h, low risk)

All inside `answer_task()` + a new `agent/router.py` (categorizer is shared with Tier 4):
- Shorter system prompt (billed once per task).
- Regex category detection (8 categories + `unknown`) attaches per-category output constraints appended to the user message:
  - sentiment → "Answer with the sentiment label plus a one-sentence justification." (fixes the known bare-"mixed" weakness)
  - code debugging → "Output only the corrected code."
  - code generation → "Output only the code."
  - summarisation → obey the task's own format/length constraint; no preamble.
  - math → "Give the final answer first, then at most two sentences of working."
  - factual/NER/logic → concise-answer nudges.
- Optional stop sequences where safe.
- Validate on practice + evalsim, eyeball all answers, push `v3-tokendiet`, re-save submission immediately.

## Tier 3 — Golden set + judge harness (~90 min, timeboxed; gate for Tier 4)

- `eval/golden_tasks.json`: 12 tasks × 4 candidate-local categories (sentiment, NER, simple factual, simple logic) + 6 × 4 cloud-bound categories = 72 tasks. Strong-model-drafted variants, human-verified expected answers, each task carries `category` and `expected_intent`.
- `eval/local_judge.py <results.json>`: scores ANY results file against the golden set (dual use: router-table generator AND pre-submission confidence gate for every image tonight). Output: per-category YES counts + global %. Uses our own Fireworks key, binary verdicts.
- Rule of thumb: nothing ships if global judge score < ~95% (real gate margin unknown).
- While the golden run executes, log per-answer wall-clock per category (accuracy AND latency both gate local routing).

## Tier 4 — Hybrid local runtime (~3–5 h, gated on Tier 3)

- **Go/no-go speed spike first (30 min):** Gemma-2-2B-it Q4_K_M under `docker run --cpus=2 --memory=4g`: measure tok/s on one sentiment + one NER prompt and model load time. Requirements: slowest candidate category ≤25 s/answer, load ≤40 s of the 60 s ready budget. Fail → Phase 2 ends at v3.
- **Bake:** GGUF downloaded at Docker **build** time into its own layer (no runtime downloads). Image ≈2 GB compressed — far under 10 GB.
- **Concurrency:** llama.cpp context is not thread-safe → one dedicated local-inference thread owning the model behind a mutex/queue; the existing 4-worker Fireworks pool is untouched. Router assigns each task a lane before dispatch.
- **Local lane timer:** hard 20 s per local answer; on expiry or empty/garbage output → fall back to Fireworks for that task (costs tokens, protects accuracy and the 30 s/response rule).
- **Routing table:** built from Tier 3 data with the count-based combined gate; embedded as a constant dict `{category: "local"|"fireworks"}` plus `unknown → fireworks`. The local classify-on-no-match asks Gemma for a single-word category from the fixed set; anything unparseable → fireworks.
- Ship as `v4-hybrid` ONLY if the full container beats ≥95% on the judge; otherwise the pointer stays on v3.

## Tier 5 — Freeze ritual (hard stop T−2.5 h before deadline)

Final `run_local.sh` under grading limits; judge ≥95%; anonymous `docker pull` of the pushed digest; confirm the lablab submission points at the intended tag; NO blind last-minute pointer moves.

## What must not change (Phase 1 invariants)

Pool/deadline-budget/snapshot/stats machinery; tolerant task parsing; model probe with fallback; empty-content retry; `max_retries=0`; exit-0-when-results-written; all Fireworks traffic through `FIREWORKS_BASE_URL`; models only from runtime `ALLOWED_MODELS`; no answer caching; English answers. Hard limits: 4 GB RAM / 2 vCPU CPU-only, 10-min runtime, 60 s ready, 30 s/response, ≤10 GB compressed linux/amd64.

## File plan

```
agent/router.py        # regex rules, category constants, per-category output constraints
agent/local_model.py   # llama-cpp wrapper: load, single-thread queue, timed generate, classify()
agent/main.py          # answer_task() grows lane branch; probe gains token-overhead ranking
eval/golden_tasks.json # 72 tasks with category + expected_intent
eval/local_judge.py    # binary-verdict judge, works on any results.json
Dockerfile             # + GGUF download layer + llama-cpp-python (v4 only; v3 image stays slim)
```

## Failure-status map (Phase 2 additions)

| Risk | Designed out by |
|---|---|
| TIMEOUT (local lane slow) | speed spike go/no-go; 20 s local timer with Fireworks fallback; load ≤40 s measured |
| ACCURACY_GATE_FAILED (misroute) | unknown→cloud default; count-based combined gate from judged data; ≥95% judge floor before any pointer move |
| IMAGE_TOO_LARGE | 2 GB total; GGUF in one build layer |
| RUNTIME_ERROR (llama.cpp) | local lane wrapped in try/except → Fireworks fallback; model-load failure → pure-cloud mode (v3 behavior) |
| Wasted submission pointer | v3 shipped early; v4 only behind judge gate; freeze ritual |

## Out of scope

Local math/summarisation lanes, fine-tuning, embedding-based routing, output-length tuning beyond Tier 2 constraints — only if every tier lands green with time to spare.

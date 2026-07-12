# Phase 3 Optimization Execution Ledger

**Goal:** Preserve the scored 100.0% accuracy while reducing all Fireworks
input and output tokens, including probes and retries.

**Baseline:** `caverraaa/routing-agent:v3-tokendiet` scored 11,196 tokens at
100.0% accuracy. The earlier v2 image scored 8,421 tokens at 94.7% accuracy.

## Acceptance gates

- All eight practice answers are correct and complete.
- No previously passing golden task regresses.
- Token totals include every Fireworks call, including probes and retries.
- Runtime is under 10 minutes, startup under 60 seconds, and each response
  under 30 seconds on 4 GB RAM / 2 CPU.
- Submission images are public, immutable, linux/amd64, and under 10 GB.

## Ownership

### Developer A — cloud cost path (`phase3-cloud-cost`)

Owns `agent/main.py`, `agent/router.py`, and the task/probe/telemetry tests.

- [x] P1a: make Fireworks token telemetry include probes and retries.
- [x] P1b: record a live practice baseline with the complete telemetry.
- [ ] P1c: record a golden baseline after explicit approval to send the private
  evaluation prompts to the external Fireworks service.
- [x] P3: replace paid runtime probes with an offline-validated model policy.
- [x] P5: shorten per-category instructions without accuracy regressions.
- [x] P6: introduce per-category first-attempt and retry token caps.
- [x] P9: enable validated factual Fireworks batches of three to four tasks.
- [x] P9.5: add task-level complexity/verifiability gating and the candidate
  lossless one-sentence summarisation profile.

### Developer B — local/zero path (`phase3-local-zero`)

Owns `agent/local_model.py`, `agent/routing_table.json`, `Dockerfile`, `eval/`,
and new local pipeline/validator/tool modules. Developer B does not edit
`agent/main.py`; Developer A owns the eventual integration commit.

- [x] P2: validate and package the existing local sentiment lane.
- [x] P4: improve local NER to the tightened deterministic 12/12 gate.
- [x] P7: add local structural validators and one repair attempt.
- [x] P8: keep factual cloud; solve supported logic locally with fail-closed fallback.
- [x] P10: enable proved math and execution-validated code tools; keep factual
  cloud after the offline-knowledge experiment failed its truth gate.

## Experiment ledger

Record task-only and all-call totals separately. `total_tokens` must equal
probe plus task prompt and completion tokens.

| Date | Branch/commit | Variant | Accuracy | Probe PT | Probe CT | Task PT | Task CT | Total | Runtime | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2026-07-12 | scored image | v2 hardened | 94.7% | unknown | unknown | unknown | unknown | 8,421 | unknown | reference |
| 2026-07-12 | scored image | v3 tokendiet | 100.0% | unknown | unknown | unknown | unknown | 11,196 | unknown | accuracy reference |
| 2026-07-12 | `3be16cb` | P1b practice: gpt-oss-120b, sentiment local | 8/8 | 170 | 50 | 791 | 827 | 1,838 | 17.9s | pass; 9 calls, 0 retries |
| 2026-07-12 | working tree | P3–P6: no probes, compact prompts/caps, sentiment+NER local | 8/8 | 0 | 0 | 656 | 753 | 1,409 | 12.6–20.1s | pass x4; 6 calls, 0 retries |
| 2026-07-12 | `phase3-local-batching` | P7–P9: validated logic local, factual batches 3–4 | 8/8 | 0 | 0 | 542 | 570 | 1,112 | 19.8s | pass; 5 calls, 0 retries |
| 2026-07-12 | `codex/phase3-risk-gate` | P9.5: task risk gate + lossless summary fusion | 8/8 | 0 | 0 | 383 | 490 | 873 | 19.7s | pass; 4 calls, 0 retries |
| 2026-07-12 | `codex/phase3-offline-tools` | P10: deterministic math + validated code tools | 8/8 | 0 | 0 | 90 | 78 | 168 | 19.5s | pass x2; 1 call, 0 retries |

The P1b container was `linux/amd64`, 1,784,945,262 bytes. The repository's
`.env` model ID was obsolete and returned 404, so the valid text-model catalog
reported by the configured proxy was supplied as a runtime override; the
reasoning-tax selector chose `accounts/fireworks/models/gpt-oss-120b` with low
reasoning effort. The failed configuration run produced no reported tokens and
is not a valid baseline.

P3–P6 reduced the comparable practice total by 429 tokens (23.3%). Four
container runs produced byte-identical answers. Local NER passed two fresh
strict golden-span runs at 12/12 (worst 1.39s) and a four-worker burst at 12/12
(worst task 5.11s, total 14.51s). Unsupported NER formats/types fail closed to
Fireworks. The complete unit suite passed 129 tests; the final linux/amd64 image
was 1,784,959,279 bytes.

Practice savings attribution: 220 tokens from removing probes, 176 from moving
NER local, and 33 net from compact cloud prompts/responses. Category caps did
not trigger on practice; their value is bounding hidden-set runaways while the
safe retry caps preserve completeness.

## Priorities 7-9 results

- Gemma factual baseline was 11/12 and hallucinated the requested body of
  water for Canberra, so factual remains Fireworks.
- Gemma logic baseline was 0/12. A strict all-different assignment solver now
  proves 11/12 golden-style tasks locally; the one prompt without an explicit
  uniqueness constraint fails closed to Fireworks. Two runs were stable and a
  four-worker burst completed in 0.015s.
- Two final practice runs with sentiment, NER, and supported logic local stayed
  8/8 and used 1,106 and 1,112 tokens (five Fireworks calls, zero
  probes/retries). The latest run is 21.1% below the P3-P6 candidate and 39.5%
  below the original measured practice baseline.
- Factual batching is limited to same-category cloud groups of 3-4. A public
  four-task A/B stayed 4/4 while reducing 524 to 459 tokens (12.4%) and four to
  two calls. An extended public run passed 12/12 in three batches with zero
  fallback/retries and used 1,204 tokens.
- Batching is enabled in the candidate Docker image and may be disabled with
  `ENABLE_BATCHING=0`. Malformed, duplicate, extra, missing, truncated, or
  invalid batch entries fail closed to the individual Fireworks path.
- A two-task A/B cost 307 batched versus 259 individual tokens, so pairs are
  deliberately not batched; the minimum size is three.
- The final suite passed 187 tests. The linux/amd64 candidate image is
  1,784,979,715 bytes and has `ENABLE_BATCHING=1` in its image environment.

## Priority 9.5 results

- The router now separates structural complexity from answer verifiability.
  Each task receives a fail-closed local/cloud decision, profile name, reason,
  and complexity level without any API call. Unsupported sentiment, NER, and
  logic shapes go directly to Fireworks instead of entering a local lane.
- Free-form Gemma summarisation was rejected. The production prompt passed
  only 1/6 stored golden-style rubrics; stricter prompts reached 2/6 and 4/6,
  while a repair attempt reached 5/6 but still failed practice.
- A narrow deterministic profile instead fuses two or three short declarative
  source sentences with semicolons. It preserves every word, number, negation,
  and clause order, adds no facts, uses no model, and returns exactly one
  terminal sentence. The stored-rubric manual audit was 6/6 plus practice 1/1.
- Eligibility requires the exact one-sentence request, 30–80 English words,
  two or three period-terminated clauses, no internal periods/custom format,
  and no instruction-like content. Any mismatch returns an empty local answer
  and uses the existing Fireworks fallback.
- Two practice runs stayed 8/8 at 867 and 873 tokens with four Fireworks calls
  and no probes/retries. The latest run is 21.5% below P7–P9 and 52.5% below
  the measured 1,838-token baseline.
- Solver-backed categorisation exposes 11/12 stored logic tasks to the proved
  local profile. The single prompt without explicit uniqueness deliberately
  keeps its known cloud path. All 12 factual tasks now reach factual caps and
  3–4 task batching rather than leaving three tasks in the unknown category.
- The final suite passed 228 tests. Independent adversarial review found no
  blocking code issue. The linux/amd64 image is 1,784,998,140 bytes.
- Residual go/no-go risk: lossless fusion is structurally one sentence and
  satisfies every stored expected-intent rubric, but it does not reduce word
  count. Do not promote this candidate to the submission image until the
  external semantic judge confirms that it accepts fusion as summarisation.

## Priority 10 results

- The exact inventory solver covers all 6/6 stored math prompts and practice
  with rational arithmetic. It accepts only percent-of-initial followed by a
  fixed removal, requires integral discrete counts, and rejects alternative
  operation order, implicit rounding, impossible totals, extra instructions,
  returns/restocking, or unreviewed time context.
- Twelve semantic code profiles cover 6/6 `code_debug` and 6/6 `code_gen`
  prompts plus both practice code tasks. Production contains no task IDs or
  prompt-answer table. Each canonical Python function passes an AST allowlist,
  restricted execution, and deterministic property tests before it is emitted;
  prompt-provided code is parsed for its signature but never executed.
- Fresh validation of all 12 code profiles took about 0.04 seconds and is
  cached afterward. A four-worker audit of all 18 math/code tasks completed in
  under 0.06 seconds. Unsupported paraphrases fail closed to Fireworks.
- Broad offline factual remains rejected: the local model was reproducibly
  11/12 and its wrong knowledge claim could not be caught structurally.
  Self-contained extraction covered 0/12. A complete periodic/SI reference
  could safely cover only 3/12 while preserving the same number of factual
  batch calls, so no benchmark-shaped knowledge table was added and factual
  remains cloud/batchable.
- Two Docker practice runs were identical at 8/8, 168 tokens, one Fireworks
  call, and zero probes/retries. This is 80.8% below P9.5, 84.9% below P7–P9,
  and 90.9% below the measured 1,838-token baseline.
- On the stored 72-task topology, 59 tasks now select local profiles; the
  remaining 12 factual tasks retain three 4-task batches and the unsupported
  logic prompt retains its known individual cloud path. This is a static
  routing projection, not an external private-golden judge result.
- The final suite passed 299 tests. Independent safety and value reviews found
  no blocking P10 issue. The linux/amd64 image is 1,785,019,982 bytes.
- The inherited P9.5 lossless-summary semantic caveat still gates promotion of
  the combined candidate; P10 itself adds only recomputed or execution-tested
  local answers.

## Merge order

1. Merge complete token telemetry.
2. Rebase both optimization branches on the telemetry commit.
3. Merge gated cloud-cost improvements.
4. Merge the local pipeline without changing `agent/main.py`.
5. Add one small integration commit owned by Developer A.
6. Run practice, golden judge, grading-limit container test, and image checks.
7. Publish a new immutable tag; never overwrite `v3-tokendiet`.

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
- [ ] P1b: record a live practice/golden baseline with the complete telemetry.
- [ ] P3: replace paid runtime probes with an offline-validated model policy.
- [ ] P5: shorten per-category instructions without accuracy regressions.
- [ ] P6: introduce per-category first-attempt and retry token caps.
- [ ] P9: evaluate small same-category Fireworks batches.

### Developer B — local/zero path (`phase3-local-zero`)

Owns `agent/local_model.py`, `agent/routing_table.json`, `Dockerfile`, `eval/`,
and new local pipeline/validator/tool modules. Developer B does not edit
`agent/main.py`; Developer A owns the eventual integration commit.

- [ ] P2: validate and package the existing local sentiment lane.
- [ ] P4: improve local NER from 10/12 to the 11/12 routing gate.
- [ ] P7: add local structural validators and one repair attempt.
- [ ] P8: evaluate local factual and logic lanes.
- [ ] P10: prototype deterministic math/code/knowledge tools.

## Experiment ledger

Record task-only and all-call totals separately. `total_tokens` must equal
probe plus task prompt and completion tokens.

| Date | Branch/commit | Variant | Accuracy | Probe PT | Probe CT | Task PT | Task CT | Total | Runtime | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2026-07-12 | scored image | v2 hardened | 94.7% | unknown | unknown | unknown | unknown | 8,421 | unknown | reference |
| 2026-07-12 | scored image | v3 tokendiet | 100.0% | unknown | unknown | unknown | unknown | 11,196 | unknown | accuracy reference |

## Merge order

1. Merge complete token telemetry.
2. Rebase both optimization branches on the telemetry commit.
3. Merge gated cloud-cost improvements.
4. Merge the local pipeline without changing `agent/main.py`.
5. Add one small integration commit owned by Developer A.
6. Run practice, golden judge, grading-limit container test, and image checks.
7. Publish a new immutable tag; never overwrite `v3-tokendiet`.

# v6 Design — Gemma-3-4B Local Generalist with Greedy Judged Lanes

**Date:** 2026-07-12 evening · **Deadline:** ~7 h · **Status:** Approved by user
**Fallback pointer (untouchable):** `v4-hybrid`, scored 8,262 Fireworks tokens @ 100% accuracy, rank ~55.

## Premise (measured)

The token bill is dominated by ~16–17 of 19 eval tasks going to the cloud; the
local lane is sentiment-only because Gemma-2-2B judged below the gate everywhere
else (ner 10/12, factual 9/12, logic 0/12 vs cloud 12/12). The router is not the
bottleneck; local competence is. Top leaderboard teams run near-pure-local at up
to 94.7% accuracy in the same 4 GB / 2 vCPU box. The real accuracy gate is ≈80%
(entries at 78.9% fail, 84.2% rank); we hold 100% — unspent margin.

## Policy changes (user-directed)

- **Ship floor: 85% on our judge** (was 95). Rationale: gate ≈80%; measured
  golden→eval transfer error ≈4 points; tokens are the ranking factor.
- **Greedy lane grant:** sort categories by tokens-saved-per-golden-point-lost;
  grant lanes in that order while projected golden global ≥ 85%. A lane judging
  ≤6/12 is never granted (broken-lane guard). Replaces the 10/12+diff≤1 rule.

## Plan of record

1. **Model swap** — `google_gemma-3-4b-it-Q4_K_M.gguf` (~2.5 GB) replaces the 2B.
   One model in the image (RAM: ~3 GB used of 4 with n_ctx 2048).
2. **Spike (GO/NO-GO by T+1h)** — under `taskset -c 0,1`: llama-cpp-python
   0.3.34 must load the Gemma-3 architecture; load ≤40 s; worst answer ≤25 s at
   max_tokens=320; RAM headroom verified.
3. **Per-category local caps** — sentiment/ner/factual 160; summarisation/math
   256; logic/code_debug/code_gen 320. `LOCAL_WORST_SECONDS` recomputed from the
   spike's measured tok/s at cap 320.
4. **Measurement (by T+2.5h)** — golden run with ALL 8 categories local, judged
   (temp-0, verdict-first parser). Produces the local column vs the recorded
   cloud column (11–12/12 everywhere).
5. **Greedy routing table** — per the policy above; committed with counts.
6. **Heavy-slice check** — 2 eval-weight prompts per granted lane; a lane at 0/2
   with cloud 2/2 is revoked (v5 post-mortem: golden is 163 tok/task vs eval's
   435 — light measurements are optimistic).
7. **Ship (by T+4h)** — container run under grading limits → judged ≥85% AND
   projected eval tokens well below 8,262 (expected 3–5k cut, far above the
   noise floor that invalidated v5) → push `v6-gemma3` → re-save submission.
   Any gate missed → abort, pointer stays on v4.

## Unchanged machinery

Router + constraints, cloud fallback on empty/exception, lock serialization and
deadline bounds, snapshot/exit-0 loop, reasoning-effort probe, temperature=0,
tolerant parsing. Gemma-prize alignment preserved (Gemma 3 is a Gemma model).

## Risk accounting

- Spike NO-GO → abort at ~45 min cost, zero submissions spent.
- Eval lands 81–89% → survives the ~80% gate at a 3–5k token saving.
- Eval gate-fails → one re-save restores v4's scored 100% (proven path).
- Wrong-but-nonempty local answers ship without confidence checks — the judged
  gate, broken-lane guard, and heavy-slice check are the protections.

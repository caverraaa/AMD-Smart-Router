# v7 Design — Qwen3-1.7B Local Generalist (delta from v6)

**Date:** 2026-07-12 late · **Status:** Approved by user · **Supersedes:** the
aborted v6 (Gemma-3-4B spike NO-GO: 7–9.5 tok/s on 2 cores excluded all
long-form lanes; residual pot ~700–950 tokens was inside the measurement noise
band). **Fallback pointer (untouchable):** `v4-hybrid`, scored 8,262 @ 100%.

## Deltas from the approved v6 design (everything else carries over verbatim)

1. **Model:** `Qwen_Qwen3-1.7B-Q4_K_M.gguf` (bartowski, ~1.1 GB file, ~1.5 GB
   loaded). Rationale: strongest ≤2B-class model; estimated 15–25 tok/s on 2
   pinned cores makes the 256-cap lanes (summarisation, math) time-feasible
   where the 4B was excluded — pot grows to ~1.5–2.2k eval tokens (above the
   ~1k noise band), projected landing ~6.1–6.7k total ≈ rank ~44–47.
2. **Non-thinking enforcement (both layers):** append Qwen3's `/no_think`
   soft-switch to every local prompt; strip `<think>…</think>` blocks in
   `LocalModel.generate` as defense-in-depth. Spike verifies: qwen3 arch loads
   under llama-cpp-python 0.3.34, no think-blocks in output, latency matches
   estimates.
3. **Gemma-prize alignment dropped** (user decision — one-lane Gemma usage
   judged uncompetitive for the prize; model choice unconstrained).

## Carried over from v6 unchanged

85% judged ship floor; greedy lane grant by tokens-saved-per-golden-point-lost
while projected global ≥85%, never granting a lane ≤6/12; per-category local
caps (sentiment/ner/factual 160, summarisation/math 256, logic/code 320);
measurement of ALL 8 categories; heavy-slice check (2 eval-weight prompts per
granted lane, revoke on 0/2); one model in the image; container judged gate;
timeline gates (spike T+1h, golden T+2.5h, ship T+4h, else abort to v4);
pointer discipline. The v6 Task-1 code (per-category caps, commit a6c39aa) is
model-agnostic and reused; only the model path and the no-think layers change.

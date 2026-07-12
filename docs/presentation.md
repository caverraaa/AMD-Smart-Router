# Slide deck — AMD Smart Router (Track 1) · 5 slides

Deck language: English. Diagram: redraw in Excalidraw from the spec on slide 3.

---

## Slide 1 — Title

**AMD Smart Router**
Hybrid Token-Efficient Routing Agent — Track 1

Team: **Verso**

- Docker: `caverraaa/routing-agent:v4-hybrid` (Docker Hub, public)
- Code: github.com/caverraaa/AMD-Smart-Router
- Result: **100.0% accuracy · 8,262 Fireworks tokens** (hidden eval)

---

## Slide 2 — The task & our result

> Answer 8 categories of NL tasks with the fewest Fireworks tokens that still
> clear the accuracy gate — local inference is free, wrong answers are fatal.

**Our submission: 100.0% accuracy at 8,262 tokens** — the top accuracy tier,
26% fewer tokens than our own accuracy-first baseline (11,196), achieved by
measuring where tokens actually go instead of guessing.

---

## Slide 3 — Architecture (one flow, redraw in Excalidraw)

```
/input/tasks.json
      │  tolerant parser (int ids, alt keys, wrappers)
      ▼
regex categorizer ──► category + output constraint
      │
      ├── sentiment ──► LOCAL: Gemma-2-2B-it Q4 (llama.cpp, CPU)
      │                 0 Fireworks tokens · lock-serialized · deadline-bounded
      │                 └─ empty/failure? → falls through to cloud ↓
      │
      └── everything else ──► FIREWORKS via proxy
                              startup probe picks model + reasoning_effort=low
                              (logic exempt) · temperature=0 · retry on empty
      ▼
atomic snapshot after EVERY task → /output/results.json (always valid, exit 0)
```

Notes for the drawing: 3 columns (input → router → two lanes) merging into the
output box; the local→cloud fallback arrow is the detail worth showing.

---

## Slide 4 — Where the tokens went (measured, not guessed)

| Mechanism | Effect |
|---|---|
| **Hidden-reasoning probe** — measures billed-vs-visible tokens per model at startup, applies `reasoning_effort=low` when accepted | **−~3,000** (reasoning was ~40% of the naive bill) |
| Token diet — 44-char system prompt + per-category output constraints | −~500, and raised accuracy 94.7% → 100% |
| **Gemma-2-2B serves the local lane, admitted per-category by a count-based judge gate** — sentiment earned it 12/12 vs cloud 12/12; logic was denied at 0/12 | sentiment at **zero tokens** |
| `temperature=0` everywhere | deterministic answers & measurements |

Every routing decision is data: a 72-task golden set + binary-verdict LLM
judge gate every change before the submission pointer moves.

---

## Slide 5 — Results & what we proved

| Version | Tokens | Accuracy | What changed |
|---|---|---|---|
| v2 baseline | 8,421 | 94.7% | cheapest model, hardened I/O |
| v3 | 11,196 | **100%** | bought accuracy with constraints |
| **v4 (submitted)** | **8,262** | **100%** | reasoning-tax cut + Gemma lane |

Measured on the way (all in the repo's engineering log):
- The hidden eval = 19 tasks; the accuracy gate ≈ 80% — accuracy above the
  gate is worthless, so we converted margin into token cuts **only when the
  measured saving exceeded our measurement noise** (one experiment that
  didn't — v5 — degraded and was reverted within a single cycle).
- The local-model frontier in 4 GB / 2 vCPU: ≤2B models lack accuracy beyond
  sentiment; ≥3B models lack speed for long answers. The 0-token leaders paid
  5–37 accuracy points for that trade — our judge gate is why we didn't.

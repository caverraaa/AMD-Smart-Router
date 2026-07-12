# AMD Smart Router — Hybrid Token-Efficient Routing Agent

AMD Developer Hackathon ACT II, Track 1. Batch-job container: reads
`/input/tasks.json`, answers 8 categories of NL tasks with the fewest
Fireworks tokens that still clear the accuracy gate, writes
`/output/results.json`, exits 0.

**Scored result: 100% accuracy, 8,262 Fireworks tokens** on the hidden
evaluation set (`v4-hybrid`).

## How it works

1. **Reasoning-tax-aware model selection.** A tiny startup probe measures
   each allowed model's hidden-reasoning overhead (billed-minus-visible
   completion tokens) and applies `reasoning_effort: "low"` when the
   endpoint accepts it — hidden reasoning was ~40% of the naive bill.
   Logic tasks are exempted (low effort measurably breaks deduction).
2. **Category router.** Regex rules classify each prompt (sentiment, NER,
   summarisation, code debug/gen, math, logic, factual) and attach a
   per-category output constraint; unknown always goes to the cloud.
3. **Local lane.** Gemma-2-2B-it (Q4_K_M, llama-cpp, CPU) is baked into
   the image and answers **sentiment** tasks at zero Fireworks tokens —
   the only category where its judged accuracy matched the cloud model
   (12/12 vs 12/12 on our 72-task golden set). Any local failure falls
   back to the cloud; the lane is deadline-bounded and serialized behind
   a lock (llama.cpp contexts are not thread-safe).
4. **Hardened batch loop.** Tolerant input parsing (int ids, alternate
   field names, wrappers), 4-worker pool, per-request timeouts + one
   retry (also fired by empty content from reasoning burn-out), atomic
   `results.json` snapshot after every task, `temperature=0` everywhere,
   wall-clock soft budget with graceful degradation, exit 0 whenever
   results exist.

Routing decisions are data, not vibes: `eval/local_judge.py` scores any
results file against `eval/golden_tasks.json` (72 tasks, binary LLM-judge
verdicts) — a category routes local only if the local model is within one
task of the cloud model and above an absolute floor.

## Setup

```bash
cp .env.example .env          # your Fireworks key (local dev only; the
                              # grading harness injects real values)
# the local model is not in git — fetch it before building:
mkdir -p models && curl -L -o models/gemma-2-2b-it-Q4_K_M.gguf \
  "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf"
```

## Run the smoke test (build + grading-VM limits + schema check)

```bash
./run_local.sh
```

## Unit tests (103) and judged evaluation

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest
# judge any results file against the golden set (uses your own key):
.venv/bin/python eval/local_judge.py output/results.json
```

## Build & push for submission

```bash
docker buildx build --platform linux/amd64 -t <registry>/<user>/routing-agent:tag --push .
```

The image must be publicly pullable with a linux/amd64 manifest.

## Design docs & engineering log

- Phase 1 spec: `docs/superpowers/specs/2026-07-11-phase1-baseline-design.md`
- Phase 2 spec: `docs/superpowers/specs/2026-07-12-phase2-hybrid-router-design.md`
- Highlights of the journey: a 0.0%→94.7% forensic debugging of silent
  input-schema failure; discovery that hidden reasoning tokens were ~40%
  of the bill; a judged routing table where Gemma earned its lane with
  data (and logic was denied at 0/12); and a reverted v5 that taught us
  terse output constraints on reasoning models *relocate* tokens into
  unbounded hidden reasoning rather than deleting them.

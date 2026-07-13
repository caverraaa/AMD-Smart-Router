# Token-Efficient Routing Agent — AMD Hackathon ACT II, Track 1

Batch-job container: reads `/input/tasks.json`, answers each task via a
validated preferred model (or the largest identifiable fallback) in
`ALLOWED_MODELS` through `FIREWORKS_BASE_URL`, writes `/output/results.json`,
and exits 0.

The checked-in submission profile is deliberately **accuracy-first**: all
eight categories use Fireworks and factual batching is disabled.  The Phase 3
local tools remain available for further evaluation, but are not enabled in
the Docker image because the external submission returned
`ACCURACY_GATE_FAILED`.  If the preferred validated model is unavailable, the
agent now falls back to the largest identifiable model in `ALLOWED_MODELS`
instead of the smallest one.

## Setup

    cp .env.example .env    # fill in your Fireworks key (local dev only;
                            # the grading harness injects real values)

## Run the smoke test (build + grading-VM limits + schema check)

    ./run_local.sh

## Run unit tests

    python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
    .venv/bin/pytest

## Build & push for submission

    docker buildx build --platform linux/amd64 -t <registry>/<user>/routing-agent:latest --push .

The image must be publicly pullable and include a linux/amd64 manifest.

## Design

See `docs/superpowers/specs/2026-07-11-phase1-baseline-design.md` — every
design decision maps to a grading failure status it protects against.

Before re-enabling a local category or batching, run the full golden semantic
judge and require at least 95% global accuracy.  Practice-task unit tests alone
do not validate hidden-task generalisation.

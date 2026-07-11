# Token-Efficient Routing Agent — AMD Hackathon ACT II, Track 1

Batch-job container: reads `/input/tasks.json`, answers each task via the
cheapest model in `ALLOWED_MODELS` through `FIREWORKS_BASE_URL`, writes
`/output/results.json`, exits 0.

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

#!/usr/bin/env bash
# Pre-submission smoke test: build + run under grading-VM limits + schema assert.
set -euo pipefail
cd "$(dirname "$0")"

[ -f .env ] || { echo "FATAL: .env missing (copy .env.example and fill in your key)"; exit 1; }

mkdir -p input output
cp practice_tasks.json input/tasks.json

docker buildx build --platform linux/amd64 -t routing-agent .

# set -e makes a non-zero container exit abort the script — that IS the exit-code assertion
docker run --rm --memory=4g --cpus=2 \
  --env-file .env \
  -v "$PWD/input:/input" -v "$PWD/output:/output" \
  routing-agent

python3 - <<'EOF'
import json

expected = {t["task_id"] for t in json.load(open("input/tasks.json"))}
results = json.load(open("output/results.json"))
assert isinstance(results, list), "results must be a JSON list"
seen = set()
for entry in results:
    assert isinstance(entry, dict), f"entry not an object: {entry!r}"
    assert isinstance(entry.get("task_id"), str), f"bad task_id: {entry!r}"
    assert isinstance(entry.get("answer"), str), f"bad answer: {entry!r}"
    seen.add(entry["task_id"])
assert seen == expected, f"task_id mismatch: missing={expected - seen} extra={seen - expected}"
print(f"SCHEMA OK — {len(results)} results\n")
print("Eyeball these answers BEFORE submitting (10 submissions/hour limit):")
for entry in results:
    print(f"\n=== {entry['task_id']} ===\n{entry['answer']}")
EOF

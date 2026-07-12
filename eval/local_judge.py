"""Binary-verdict LLM judge: scores any results.json against the golden set.

Usage: python eval/local_judge.py <results.json> [--golden eval/golden_tasks.json]
                                  [--threshold 95]
Uses OUR OWN Fireworks key (.env) — local judging costs nothing on the leaderboard.
"""
import argparse
import json
import os
import re
import sys

def judge_prompt(task, answer):
    return (f"Task given to an AI assistant: {task['prompt']}\n"
            f"Expected (rubric): {task['expected_intent']}\n"
            f"Assistant's answer: {answer}\n\n"
            "Does the answer satisfy the rubric? Start your reply with exactly one word, "
            "YES or NO, then one short sentence of justification.")


def parse_verdict(text):
    m = re.match(r"\s*\**\s*(YES|NO)\b", text or "", re.IGNORECASE)
    if m:
        return m.group(1).upper() == "YES"
    matches = re.findall(r"\b(YES|NO)\b", text or "")  # uppercase only: a trailing verdict
    return bool(matches) and matches[-1] == "YES"


def score_results(golden, results, verdicts):
    answers = {r["task_id"]: r.get("answer", "") for r in results}
    per_category = {}
    yes_total = 0
    for t in golden:
        cat = t["category"]
        ok = bool(answers.get(t["task_id"])) and verdicts.get(t["task_id"], False)
        y, n = per_category.get(cat, (0, 0))
        per_category[cat] = (y + (1 if ok else 0), n + 1)
        yes_total += 1 if ok else 0
    return {"per_category": per_category,
            "global_pct": 100.0 * yes_total / len(golden) if golden else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--golden", default="eval/golden_tasks.json")
    ap.add_argument("--threshold", type=float, default=95.0)
    args = ap.parse_args()

    golden = json.load(open(args.golden, encoding="utf-8"))
    results = json.load(open(args.results, encoding="utf-8"))
    answers = {r["task_id"]: r.get("answer", "") for r in results}

    from openai import OpenAI
    client = OpenAI(base_url=os.environ["FIREWORKS_BASE_URL"],
                    api_key=os.environ["FIREWORKS_API_KEY"], max_retries=0)
    judge_model = os.environ.get("JUDGE_MODEL", "accounts/fireworks/models/kimi-k2p6")

    verdicts = {}
    for t in golden:
        answer = answers.get(t["task_id"], "")
        if not answer:
            verdicts[t["task_id"]] = False
            continue
        resp = client.chat.completions.create(
            model=judge_model, max_tokens=2048, timeout=30,
            messages=[{"role": "user", "content": judge_prompt(t, answer)}])
        verdicts[t["task_id"]] = parse_verdict(resp.choices[0].message.content)

    s = score_results(golden, results, verdicts)
    for cat, (y, n) in sorted(s["per_category"].items()):
        print(f"{cat:>14}: {y}/{n}")
    print(f"{'GLOBAL':>14}: {s['global_pct']:.1f}%  (threshold {args.threshold}%)")
    sys.exit(0 if s["global_pct"] >= args.threshold else 1)


if __name__ == "__main__":
    main()

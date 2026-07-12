"""Draft golden tasks with a strong model; a human verifies before commit.

Usage:
  python eval/make_golden.py            # generate eval/golden_tasks.json draft
  python eval/make_golden.py --check    # schema-validate the (edited) file
"""
import json
import os
import sys

COUNTS = {"sentiment": 12, "ner": 12, "factual": 12, "logic": 12,
          "math": 6, "summarisation": 6, "code_debug": 6, "code_gen": 6}

SEEDS = {
    "sentiment": ("Classify the sentiment of this review: The battery life is great, "
                  "but the screen scratches too easily.",
                  "Label 'mixed' (or equivalent) plus a justification naming the positive and negative aspects."),
    "ner": ("Extract all named entities and their types from: Maria Sanchez joined "
            "Fireworks AI in Berlin last March.",
            "Identifies Maria Sanchez=PERSON, Fireworks AI=ORG, Berlin=LOCATION, last March=DATE."),
    "factual": ("What is the capital of Australia, and what body of water is it near?",
                "Canberra; Lake Burley Griffin."),
    "logic": ("Three friends, Sam, Jo, and Lee, each own a different pet: cat, dog, bird. "
              "Sam does not own the bird. Jo owns the dog. Who owns the cat?",
              "Sam owns the cat."),
    "math": ("A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. "
             "How many items remain?", "144."),
    "summarisation": ("Summarize the following in exactly one sentence: <paragraph>",
                      "Exactly one sentence covering the main points."),
    "code_debug": ("This function should return the max of a list but has a bug: "
                   "def get_max(nums): return nums[0]. Find and fix it.",
                   "Corrected function that iterates and returns the true maximum."),
    "code_gen": ("Write a Python function that returns the second-largest number in a "
                 "list, handling duplicates correctly.",
                 "Working function returning second-largest distinct value, handles duplicates."),
}

GEN_PROMPT = """Generate {n} new tasks of the category "{cat}" for evaluating an AI assistant.
Model them on this example task (same difficulty and style, different content):
Task: {seed_prompt}
Expected: {seed_intent}

Return ONLY a JSON array of objects: {{"prompt": "...", "expected_intent": "..."}}.
expected_intent must state the objectively correct answer or the rubric a judge can verify."""


def generate():
    from openai import OpenAI
    client = OpenAI(base_url=os.environ["FIREWORKS_BASE_URL"],
                    api_key=os.environ["FIREWORKS_API_KEY"], max_retries=0)
    judge_model = os.environ.get("JUDGE_MODEL", "accounts/fireworks/models/kimi-k2p5")
    out = []
    for cat, n in COUNTS.items():
        seed_prompt, seed_intent = SEEDS[cat]
        out.append({"task_id": f"g-{cat}-0", "category": cat,
                    "prompt": seed_prompt, "expected_intent": seed_intent})
        resp = client.chat.completions.create(
            model=judge_model, max_tokens=4096, timeout=120,
            messages=[{"role": "user", "content": GEN_PROMPT.format(
                n=n - 1, cat=cat, seed_prompt=seed_prompt, seed_intent=seed_intent)}])
        text = resp.choices[0].message.content
        drafted = json.loads(text[text.index("["):text.rindex("]") + 1])
        for i, d in enumerate(drafted[: n - 1], start=1):
            out.append({"task_id": f"g-{cat}-{i}", "category": cat,
                        "prompt": d["prompt"], "expected_intent": d["expected_intent"]})
    with open("eval/golden_tasks.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(out)} tasks — HUMAN-VERIFY every expected_intent before committing")


def check():
    tasks = json.load(open("eval/golden_tasks.json", encoding="utf-8"))
    from collections import Counter
    counts = Counter(t["category"] for t in tasks)
    assert counts == Counter(COUNTS), f"count mismatch: {counts}"
    for t in tasks:
        assert t["task_id"] and t["prompt"] and t["expected_intent"], t
    ids = [t["task_id"] for t in tasks]
    assert len(ids) == len(set(ids)), "duplicate task_ids"
    print(f"OK: {len(tasks)} tasks, counts {dict(counts)}")


if __name__ == "__main__":
    check() if "--check" in sys.argv else generate()

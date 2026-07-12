import sys

sys.path.insert(0, "eval")
from local_judge import judge_prompt, parse_verdict, score_results  # noqa: E402

GOLDEN = [
    {"task_id": "g-math-0", "category": "math", "prompt": "2+2?", "expected_intent": "4."},
    {"task_id": "g-math-1", "category": "math", "prompt": "3+3?", "expected_intent": "6."},
    {"task_id": "g-ner-0", "category": "ner", "prompt": "x", "expected_intent": "y"},
]


def test_parse_verdict_yes_no():
    assert parse_verdict("YES") is True
    assert parse_verdict("The answer satisfies the rubric. Verdict: YES.") is True
    assert parse_verdict("NO — the label is missing justification") is False
    assert parse_verdict("") is False  # unparseable counts as NO (conservative)


def test_judge_prompt_contains_rubric_and_answer():
    p = judge_prompt(GOLDEN[0], "it is 4")
    assert "2+2?" in p and "4." in p and "it is 4" in p and "YES or NO" in p


def test_score_results_aggregates_per_category():
    results = [{"task_id": "g-math-0", "answer": "4"},
               {"task_id": "g-math-1", "answer": "7"},
               {"task_id": "g-ner-0", "answer": "y"}]
    verdicts = {"g-math-0": True, "g-math-1": False, "g-ner-0": True}
    s = score_results(GOLDEN, results, verdicts)
    assert s["per_category"]["math"] == (1, 2)
    assert s["per_category"]["ner"] == (1, 1)
    assert round(s["global_pct"], 1) == 66.7


def test_missing_answer_counts_as_no():
    s = score_results(GOLDEN, [], {})
    assert s["global_pct"] == 0.0

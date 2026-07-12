import pytest

from agent.router import CATEGORIES, CONSTRAINTS, build_user_message, categorize

PRACTICE = [
    ("What is the capital of Australia, and what body of water is it near?", "factual"),
    ("A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. How many items remain?", "math"),
    ("Classify the sentiment of this review: The battery life is great, but the screen scratches too easily.", "sentiment"),
    ("Summarize the following in exactly one sentence: Remote work has reshaped offices.", "summarisation"),
    ("Extract all named entities and their types from: Maria Sanchez joined Fireworks AI in Berlin last March.", "ner"),
    ("This function should return the max of a list but has a bug: def get_max(nums): return nums[0]. Find and fix it.", "code_debug"),
    ("Three friends, Sam, Jo, and Lee, each own a different pet: cat, dog, bird. Sam does not own the bird. Jo owns the dog. Who owns the cat?", "logic"),
    ("Write a Python function that returns the second-largest number in a list, handling duplicates correctly.", "code_gen"),
]


@pytest.mark.parametrize("prompt,expected", PRACTICE)
def test_categorize_practice_set(prompt, expected):
    assert categorize(prompt) == expected


def test_categorize_eval_grade_variants():
    assert categorize("A train leaves at 09:14 travelling 87 km/h. At what clock time does it arrive?") == "math"
    assert categorize("On 14 February 2025, Dr. Amara Okafor of the European Space Agency presented... Extract all named entities and their types.") == "ner"
    assert categorize("Explain how a hash table achieves average O(1) lookup.") == "factual"


def test_proved_assignment_variants_route_logic_without_broad_keyword_match():
    assert categorize(
        "Three athletes, Pat, Quinn, and Ray, each play a different sport: "
        "soccer, tennis, or basketball. Pat does not play basketball. "
        "Quinn plays soccer. Who plays tennis?"
    ) == "logic"
    assert categorize(
        "Three students, Ann, Bob, and Cal, each study a different language: "
        "Spanish, French, or German. Ann does not study German. "
        "Bob studies Spanish. Who studies French?"
    ) == "logic"
    assert categorize("For each different era, explain who ruled Rome.") == "factual"


def test_router_exposes_all_proved_golden_logic_tasks_to_local_gate():
    import json
    import pathlib

    golden = json.loads(pathlib.Path("eval/golden_tasks.json").read_text(
        encoding="utf-8"))
    logic = [task for task in golden if task["category"] == "logic"]
    assert sum(categorize(task["prompt"]) == "logic" for task in logic) == 11


def test_all_golden_factual_tasks_reach_factual_caps_and_batching():
    import json
    import pathlib

    golden = json.loads(pathlib.Path("eval/golden_tasks.json").read_text(
        encoding="utf-8"))
    factual = [task for task in golden if task["category"] == "factual"]
    assert len(factual) == 12
    assert all(categorize(task["prompt"]) == "factual" for task in factual)


def test_unmatched_prompt_is_unknown():
    assert categorize("zorble the frumious bandersnatch") == "unknown"


def test_categorize_always_returns_member():
    for p in ["", "hello", "do the thing with the stuff"]:
        assert categorize(p) in CATEGORIES


def test_build_user_message_appends_constraint():
    msg = build_user_message("Classify the sentiment: great phone.", "sentiment")
    assert msg.startswith("Classify the sentiment: great phone.")
    assert CONSTRAINTS["sentiment"] in msg


def test_build_user_message_unknown_passthrough():
    assert build_user_message("mystery task", "unknown") == "mystery task"


def test_sentiment_constraint_preserves_label_reason_and_mixed_rule():
    text = CONSTRAINTS["sentiment"].lower()
    assert all(word in text for word in ("positive", "negative", "neutral", "mixed", "reason"))


def test_compact_constraints_preserve_completion_requirements():
    assert "relative dates" in CONSTRAINTS["ner"].lower()
    assert "format and length exactly" in CONSTRAINTS["summarisation"].lower()
    assert "working" in CONSTRAINTS["math"].lower()
    assert "corrected code" in CONSTRAINTS["code_debug"].lower()
    assert CONSTRAINTS["code_gen"] == "Output only the code."
    assert "every constraint" in CONSTRAINTS["logic"].lower()
    assert "two sentences" in CONSTRAINTS["factual"].lower()
    assert sum(map(len, CONSTRAINTS.values())) <= 420


def test_explain_sentiment_is_not_classification():
    assert categorize("Explain how sentiment analysis works in NLP.") != "sentiment"


def test_classify_into_categories_is_not_sentiment():
    assert categorize(
        "Classify each customer feedback message into billing, shipping, "
        "or product categories."
    ) != "sentiment"


def test_classify_tone_is_not_sentiment():
    assert categorize("Classify the tone of this email as formal or informal.") != "sentiment"


def test_write_function_over_locations_is_code_gen():
    assert categorize(
        "Write a Python function to extract all locations from a list of address strings."
    ) == "code_gen"


def test_constraints_do_not_leak_golden_strings():
    import json
    import pathlib

    golden = json.loads(pathlib.Path("eval/golden_tasks.json").read_text(encoding="utf-8"))
    blob = " ".join(t["prompt"] + " " + t["expected_intent"] for t in golden).lower()
    for text in CONSTRAINTS.values():
        for phrase in ("'yesterday'", "'last week'", "'next month'"):
            if phrase in text:
                assert phrase.strip("'") not in blob, f"constraint example {phrase} appears in golden set"

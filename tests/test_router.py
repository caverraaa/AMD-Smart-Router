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


def test_sentiment_constraint_demands_justification():
    assert "justification" in CONSTRAINTS["sentiment"].lower()

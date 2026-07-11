import math

from agent.main import parse_model_size, pick_cheapest_model


def test_parse_size_plain_billions():
    assert parse_model_size("accounts/fireworks/models/llama-v3p1-8b-instruct") == 8.0


def test_parse_size_decimal():
    assert parse_model_size("accounts/fireworks/models/qwen3-0.6b") == 0.6


def test_parse_size_case_insensitive():
    assert parse_model_size("accounts/fireworks/models/Gemma-2-9B-it") == 9.0


def test_parse_size_unparseable_is_inf():
    assert parse_model_size("accounts/fireworks/models/mixtral-moe-instruct") == math.inf


def test_parse_size_does_not_match_word_prefixes():
    # 'b' followed by more letters is a word, not a size suffix
    assert parse_model_size("accounts/fireworks/models/model-3base") == math.inf


def test_picks_smallest(monkeypatch):
    monkeypatch.delenv("CHEAP_MODEL", raising=False)
    allowed = [
        "accounts/fireworks/models/llama-v3p1-70b-instruct",
        "accounts/fireworks/models/gemma-2-2b-it",
        "accounts/fireworks/models/llama-v3p1-8b-instruct",
    ]
    assert pick_cheapest_model(allowed) == "accounts/fireworks/models/gemma-2-2b-it"


def test_all_unparseable_falls_back_to_first(monkeypatch):
    monkeypatch.delenv("CHEAP_MODEL", raising=False)
    allowed = ["accounts/x/alpha-instruct", "accounts/x/beta-instruct"]
    assert pick_cheapest_model(allowed) == "accounts/x/alpha-instruct"


def test_env_override_wins(monkeypatch):
    allowed = ["accounts/x/small-2b", "accounts/x/big-70b"]
    monkeypatch.setenv("CHEAP_MODEL", "accounts/x/big-70b")
    assert pick_cheapest_model(allowed) == "accounts/x/big-70b"


def test_env_override_outside_list_is_ignored(monkeypatch):
    allowed = ["accounts/x/small-2b", "accounts/x/big-70b"]
    monkeypatch.setenv("CHEAP_MODEL", "accounts/evil/not-allowed")
    assert pick_cheapest_model(allowed) == "accounts/x/small-2b"

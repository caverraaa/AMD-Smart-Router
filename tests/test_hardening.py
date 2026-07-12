"""Tests for tolerant input, offline model selection, and bounded retries."""
import json
import time

import pytest

import agent.main as m
from agent.main import load_tasks, pick_working_model, answer_task

from tests.test_answer_task import FakeClient, fake_response

FUTURE = time.monotonic() + 3600


def write_tasks(tmp_path, data):
    p = tmp_path / "tasks.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


# --- Fix 1: tolerant task parsing -------------------------------------------

def test_integer_task_ids_are_coerced(tmp_path):
    path = write_tasks(tmp_path, [{"task_id": 1, "prompt": "a"}, {"task_id": 2, "prompt": "b"}])
    task_ids, answerable = load_tasks(path)
    assert task_ids == ["1", "2"]
    assert [t["task_id"] for t in answerable] == ["1", "2"]


def test_alternate_id_key(tmp_path):
    path = write_tasks(tmp_path, [{"id": "t1", "prompt": "a"}])
    task_ids, answerable = load_tasks(path)
    assert task_ids == ["t1"]
    assert answerable == [{"task_id": "t1", "prompt": "a"}]


@pytest.mark.parametrize("key", ["question", "input", "task", "text", "query", "instruction"])
def test_alternate_prompt_keys(tmp_path, key):
    path = write_tasks(tmp_path, [{"task_id": "t1", key: "what is 2+2?"}])
    task_ids, answerable = load_tasks(path)
    assert answerable == [{"task_id": "t1", "prompt": "what is 2+2?"}]


def test_unknown_string_field_used_as_last_resort(tmp_path):
    path = write_tasks(tmp_path, [{"task_id": "t1", "weird_field": "the actual prompt"}])
    _, answerable = load_tasks(path)
    assert answerable == [{"task_id": "t1", "prompt": "the actual prompt"}]


def test_dict_wrapper_with_tasks_key(tmp_path):
    path = write_tasks(tmp_path, {"tasks": [{"task_id": "t1", "prompt": "a"}]})
    task_ids, answerable = load_tasks(path)
    assert task_ids == ["t1"]


def test_duplicate_task_ids_keep_first(tmp_path):
    path = write_tasks(tmp_path, [
        {"task_id": "t1", "prompt": "first"},
        {"task_id": "t1", "prompt": "second"},
    ])
    task_ids, answerable = load_tasks(path)
    assert task_ids == ["t1"]
    assert answerable == [{"task_id": "t1", "prompt": "first"}]


# --- Fix 2: deterministic offline model policy -------------------------------

VALIDATED_MODEL = "accounts/fireworks/models/gpt-oss-120b"


def test_offline_policy_prefers_validated_model_without_api_calls(monkeypatch):
    monkeypatch.delenv("CHEAP_MODEL", raising=False)
    client = FakeClient([])
    allowed = ["accounts/x/small-2b", VALIDATED_MODEL]
    assert pick_working_model(client, allowed) == (
        VALIDATED_MODEL, {"reasoning_effort": "low"})
    assert client.chat.completions.calls == []


def test_offline_policy_unknown_catalog_falls_back_to_smallest(monkeypatch):
    monkeypatch.delenv("CHEAP_MODEL", raising=False)
    client = FakeClient([])
    allowed = ["accounts/x/big-70b", "accounts/x/small-2b"]
    assert pick_working_model(client, allowed) == ("accounts/x/small-2b", None)
    assert client.chat.completions.calls == []


def test_offline_policy_respects_allowlisted_override(monkeypatch):
    monkeypatch.setenv("CHEAP_MODEL", "accounts/x/big-70b")
    client = FakeClient([])
    allowed = [VALIDATED_MODEL, "accounts/x/big-70b"]
    assert pick_working_model(client, allowed) == ("accounts/x/big-70b", None)
    assert client.chat.completions.calls == []


def test_offline_policy_ignores_non_allowlisted_override(monkeypatch):
    monkeypatch.setenv("CHEAP_MODEL", "accounts/evil/not-allowed")
    client = FakeClient([])
    allowed = ["accounts/x/small-2b", VALIDATED_MODEL]
    model, _ = pick_working_model(client, allowed)
    assert model == VALIDATED_MODEL
    assert model in allowed


def test_offline_policy_leaves_probe_stats_zero(monkeypatch):
    monkeypatch.delenv("CHEAP_MODEL", raising=False)
    stats = m._empty_usage_stats()
    pick_working_model(FakeClient([]), [VALIDATED_MODEL], probe_stats=stats)
    assert stats == {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}


def test_answer_task_passes_extra_body():
    client = FakeClient([fake_response("4")])
    task = {"task_id": "t1", "prompt": "2+2?"}
    answer_task(client, "m-x", task, FUTURE, extra_body={"reasoning_effort": "low"})
    assert client.chat.completions.calls[0]["extra_body"] == {"reasoning_effort": "low"}


def test_logic_category_suppresses_low_effort_knob():
    client = FakeClient([fake_response("Sam owns the cat.")])
    task = {"task_id": "t1", "prompt": "who owns the cat?", "category": "logic"}
    answer_task(client, "m-x", task, FUTURE, extra_body={"reasoning_effort": "low"})
    assert "extra_body" not in client.chat.completions.calls[0]


# --- Fix 3: empty-content retry (reasoning exhaustion) -----------------------

TASK = {"task_id": "t1", "prompt": "hard question"}


def empty_length_response(completion_tokens=2048):
    r = fake_response("", prompt_tokens=50, completion_tokens=completion_tokens)
    r.choices[0].finish_reason = "length"
    return r


def test_empty_content_triggers_retry_with_bigger_cap(monkeypatch):
    monkeypatch.setattr(m, "RETRY_BACKOFF_SECONDS", 0)
    client = FakeClient([empty_length_response(), fake_response("42", 50, 900)])
    r = answer_task(client, "m-x", TASK, FUTURE)
    assert r["answer"] == "42"
    assert r["error"] is None
    calls = client.chat.completions.calls
    assert calls[0]["max_tokens"] == m.CATEGORY_TOKEN_CAPS["unknown"][0]
    assert calls[1]["max_tokens"] == m.CATEGORY_TOKEN_CAPS["unknown"][1]
    # tokens from BOTH attempts count (the proxy billed both)
    assert r["completion_tokens"] == 2048 + 900
    assert r["retry_calls"] == 1
    assert r["retry_prompt_tokens"] == 50
    assert r["retry_completion_tokens"] == 900


def test_empty_content_twice_returns_empty_with_error(monkeypatch):
    monkeypatch.setattr(m, "RETRY_BACKOFF_SECONDS", 0)
    client = FakeClient([empty_length_response(), empty_length_response()])
    r = answer_task(client, "m-x", TASK, FUTURE)
    assert r["answer"] == ""
    assert "truncated content" in r["error"]
    assert len(client.chat.completions.calls) == 2


def test_visible_truncation_is_retried_not_returned(monkeypatch):
    monkeypatch.setattr(m, "RETRY_BACKOFF_SECONDS", 0)
    truncated = fake_response("def unfinished(", prompt_tokens=20, completion_tokens=1024)
    truncated.choices[0].finish_reason = "length"
    client = FakeClient([truncated, fake_response("def complete():\n    return 1")])
    task = {"task_id": "t1", "prompt": "write a function", "category": "code_gen"}
    r = answer_task(client, "m-x", task, FUTURE)
    assert r["answer"] == "def complete():\n    return 1"
    assert r["retry_calls"] == 1
    assert client.chat.completions.calls[0]["max_tokens"] == 2048
    assert client.chat.completions.calls[1]["max_tokens"] == 4096


def test_category_caps_are_positive_and_retries_have_more_room():
    assert "unknown" in m.CATEGORY_TOKEN_CAPS
    for first, retry in m.CATEGORY_TOKEN_CAPS.values():
        assert 0 < first < retry <= 4096

"""Tests for the 0.0%-accuracy hardening: tolerant input schema, model
probing with fallback, and empty-content (reasoning exhaustion) retry."""
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


# --- Fix 2: model probing with fallback --------------------------------------

def test_probe_picks_cheapest_when_it_works(monkeypatch):
    monkeypatch.delenv("CHEAP_MODEL", raising=False)
    client = FakeClient([fake_response("OK")])
    allowed = ["accounts/x/big-70b", "accounts/x/small-2b"]
    assert pick_working_model(client, allowed) == "accounts/x/small-2b"
    assert len(client.chat.completions.calls) == 1  # one probe only


def test_probe_falls_through_to_next_model(monkeypatch):
    monkeypatch.delenv("CHEAP_MODEL", raising=False)
    client = FakeClient([RuntimeError("not a chat model"), fake_response("OK")])
    allowed = ["accounts/x/image-model", "accounts/x/chat-model"]
    # both parse inf -> list order; first fails probe, second succeeds
    assert pick_working_model(client, allowed) == "accounts/x/chat-model"


def test_all_probes_fail_falls_back_to_cheapest(monkeypatch):
    monkeypatch.delenv("CHEAP_MODEL", raising=False)
    client = FakeClient([RuntimeError("a"), RuntimeError("b")])
    allowed = ["accounts/x/big-70b", "accounts/x/small-2b"]
    assert pick_working_model(client, allowed) == "accounts/x/small-2b"


def test_probe_respects_cheap_model_override(monkeypatch):
    monkeypatch.setenv("CHEAP_MODEL", "accounts/x/big-70b")
    client = FakeClient([fake_response("OK")])
    allowed = ["accounts/x/small-2b", "accounts/x/big-70b"]
    assert pick_working_model(client, allowed) == "accounts/x/big-70b"
    assert client.chat.completions.calls[0]["model"] == "accounts/x/big-70b"


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
    assert calls[0]["max_tokens"] == m.MAX_TOKENS
    assert calls[1]["max_tokens"] == m.RETRY_MAX_TOKENS
    # tokens from BOTH attempts count (the proxy billed both)
    assert r["completion_tokens"] == 2048 + 900


def test_empty_content_twice_returns_empty_with_error(monkeypatch):
    monkeypatch.setattr(m, "RETRY_BACKOFF_SECONDS", 0)
    client = FakeClient([empty_length_response(), empty_length_response()])
    r = answer_task(client, "m-x", TASK, FUTURE)
    assert r["answer"] == ""
    assert "empty content" in r["error"]
    assert len(client.chat.completions.calls) == 2

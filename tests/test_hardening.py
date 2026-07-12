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


# --- Fix 2: model probing with fallback (reasoning-tax-aware) ----------------

def probe_response(text, completion_tokens):
    return fake_response(text, prompt_tokens=15, completion_tokens=completion_tokens)


def test_probe_picks_cheapest_when_lean(monkeypatch):
    monkeypatch.delenv("CHEAP_MODEL", raising=False)
    client = FakeClient([probe_response("4", 5)])  # overhead 4 <= 30 -> stop
    allowed = ["accounts/x/big-70b", "accounts/x/small-2b"]
    assert pick_working_model(client, allowed) == ("accounts/x/small-2b", None)
    assert len(client.chat.completions.calls) == 1


def test_probe_falls_through_to_next_model(monkeypatch):
    monkeypatch.delenv("CHEAP_MODEL", raising=False)
    client = FakeClient([RuntimeError("not a chat model"), probe_response("4", 5)])
    allowed = ["accounts/x/image-model", "accounts/x/chat-model"]
    assert pick_working_model(client, allowed) == ("accounts/x/chat-model", None)


def test_all_probes_fail_falls_back_to_cheapest(monkeypatch):
    monkeypatch.delenv("CHEAP_MODEL", raising=False)
    client = FakeClient([RuntimeError("a"), RuntimeError("b"), RuntimeError("c"), RuntimeError("d")])
    allowed = ["accounts/x/big-70b", "accounts/x/small-2b"]
    # 2 default probes + 0 low-effort probes (they only run after a HIGH overhead success)
    assert pick_working_model(client, allowed) == ("accounts/x/small-2b", None)
    assert len(client.chat.completions.calls) == 2


def test_probe_respects_cheap_model_override(monkeypatch):
    monkeypatch.setenv("CHEAP_MODEL", "accounts/x/big-70b")
    client = FakeClient([probe_response("4", 5)])
    allowed = ["accounts/x/small-2b", "accounts/x/big-70b"]
    model, extra = pick_working_model(client, allowed)
    assert model == "accounts/x/big-70b"
    assert client.chat.completions.calls[0]["model"] == "accounts/x/big-70b"


def test_reasoning_model_gets_low_effort_knob(monkeypatch):
    monkeypatch.delenv("CHEAP_MODEL", raising=False)
    # default probe: 150 billed for "4" (overhead 149) -> tries reasoning_effort low
    # low-effort probe: 20 billed (overhead 19 <= 30) -> selected with the knob
    client = FakeClient([probe_response("4", 150), probe_response("4", 20)])
    model, extra = pick_working_model(client, ["accounts/x/reasoner-8b"])
    assert model == "accounts/x/reasoner-8b"
    assert extra == {"reasoning_effort": "low"}
    assert client.chat.completions.calls[1]["extra_body"] == {"reasoning_effort": "low"}


def test_low_effort_rejected_keeps_default(monkeypatch):
    monkeypatch.delenv("CHEAP_MODEL", raising=False)
    # model A: high overhead, low-effort knob rejected -> stays candidate at 149
    # model B: lean enough to skip the knob (overhead 3 <= 5) -> wins
    client = FakeClient([
        probe_response("4", 150), RuntimeError("unknown param reasoning_effort"),
        probe_response("4", 4),
    ])
    model, extra = pick_working_model(client, ["accounts/x/reasoner-2b", "accounts/x/plain-8b"])
    assert model == "accounts/x/plain-8b"
    assert extra is None


def test_moderate_overhead_still_gets_knob_attempt(monkeypatch):
    monkeypatch.delenv("CHEAP_MODEL", raising=False)
    # live bug: default probe measured overhead 30 (== old threshold) and the
    # knob was skipped; any overhead above estimate noise must try the knob
    client = FakeClient([probe_response("4", 31), probe_response("4", 16)])
    model, extra = pick_working_model(client, ["accounts/x/reasoner-120b"])
    assert extra == {"reasoning_effort": "low"}
    assert len(client.chat.completions.calls) == 2


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

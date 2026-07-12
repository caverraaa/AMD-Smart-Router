import time
from types import SimpleNamespace

import pytest

import agent.main as m
from agent.main import answer_task


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, outcomes):
        self.chat = SimpleNamespace(completions=FakeCompletions(outcomes))


def fake_response(text, prompt_tokens=10, completion_tokens=5):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


TASK = {"task_id": "t1", "prompt": "What is 2+2?"}
FUTURE = time.monotonic() + 3600


def test_success(monkeypatch):
    client = FakeClient([fake_response("4")])
    r = answer_task(client, "m-2b", TASK, FUTURE)
    assert r == {"task_id": "t1", "answer": "4", "prompt_tokens": 10,
                 "completion_tokens": 5, "error": None, "category": "unknown",
                 "lane": "fireworks", "fireworks_calls": 1, "retry_calls": 0,
                 "retry_prompt_tokens": 0, "retry_completion_tokens": 0,
                 "complexity": "low", "verifiability": "unverified",
                 "risk_profile": None,
                 "risk_reason": "task intent has no reviewed local profile"}
    call = client.chat.completions.calls[0]
    assert call["model"] == "m-2b"
    assert call["max_tokens"] == m.CATEGORY_TOKEN_CAPS["unknown"][0]
    assert call["temperature"] == 0.0
    assert call["timeout"] == m.FIRST_TIMEOUT_SECONDS
    assert call["messages"][0] == {"role": "system", "content": m.SYSTEM_PROMPT}
    assert call["messages"][1] == {"role": "user", "content": "What is 2+2?"}


def test_retry_succeeds_with_longer_timeout(monkeypatch):
    monkeypatch.setattr(m, "RETRY_BACKOFF_SECONDS", 0)
    client = FakeClient([TimeoutError("boom"), fake_response("4")])
    r = answer_task(client, "m-2b", TASK, FUTURE)
    assert r["answer"] == "4"
    assert r["error"] is None
    assert r["fireworks_calls"] == 2 and r["retry_calls"] == 1
    assert r["retry_prompt_tokens"] == 10
    assert r["retry_completion_tokens"] == 5
    assert client.chat.completions.calls[1]["timeout"] == m.RETRY_TIMEOUT_SECONDS
    assert client.chat.completions.calls[1]["max_tokens"] == m.CATEGORY_TOKEN_CAPS["unknown"][1]


def test_both_attempts_fail_returns_empty(monkeypatch):
    monkeypatch.setattr(m, "RETRY_BACKOFF_SECONDS", 0)
    client = FakeClient([TimeoutError("a"), TimeoutError("b")])
    r = answer_task(client, "m-2b", TASK, FUTURE)
    assert r["answer"] == ""
    assert "TimeoutError" in r["error"]
    assert r["fireworks_calls"] == 2 and r["retry_calls"] == 1
    assert len(client.chat.completions.calls) == 2


def test_non_retryable_client_error_stops_after_one_call(monkeypatch):
    class BadRequestError(Exception):
        status_code = 400

    monkeypatch.setattr(m, "RETRY_BACKOFF_SECONDS", 0)
    client = FakeClient([BadRequestError("invalid request")])
    r = answer_task(client, "m-2b", TASK, FUTURE)
    assert r["answer"] == ""
    assert "BadRequestError" in r["error"]
    assert r["fireworks_calls"] == 1 and r["retry_calls"] == 0
    assert len(client.chat.completions.calls) == 1


@pytest.mark.parametrize("category,caps", sorted(m.CATEGORY_TOKEN_CAPS.items()))
def test_first_attempt_uses_category_cap(category, caps):
    client = FakeClient([fake_response("ok")])
    task = {"task_id": "t1", "prompt": "do it", "category": category}
    answer_task(client, "m-2b", task, FUTURE)
    assert client.chat.completions.calls[0]["max_tokens"] == caps[0]


def test_past_deadline_makes_no_api_call():
    client = FakeClient([fake_response("never")])
    r = answer_task(client, "m-2b", TASK, time.monotonic() - 1)
    assert r["answer"] == ""
    assert "budget" in r["error"]
    assert client.chat.completions.calls == []


def test_none_content_and_missing_usage_handled(monkeypatch):
    monkeypatch.setattr(m, "RETRY_BACKOFF_SECONDS", 0)

    def none_resp():
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
            usage=None,
        )

    # empty content counts as a failed attempt and is retried once
    client = FakeClient([none_resp(), none_resp()])
    r = answer_task(client, "m-2b", TASK, FUTURE)
    assert r["answer"] == ""
    assert "empty content" in r["error"]
    assert r["prompt_tokens"] == 0 and r["completion_tokens"] == 0
    assert r["fireworks_calls"] == 2 and r["retry_calls"] == 1
    assert len(client.chat.completions.calls) == 2

import json
from types import SimpleNamespace

import agent.main as m

from tests.test_answer_task import FakeClient, fake_response


def setup_env(monkeypatch, tmp_path, tasks):
    input_path = tmp_path / "tasks.json"
    output_path = tmp_path / "results.json"
    input_path.write_text(json.dumps(tasks), encoding="utf-8")
    monkeypatch.setenv("FIREWORKS_API_KEY", "fk-test")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://proxy.example/v1")
    monkeypatch.setenv("ALLOWED_MODELS", "accounts/x/small-2b,accounts/x/big-70b")
    monkeypatch.delenv("CHEAP_MODEL", raising=False)
    monkeypatch.setenv("AGENT_INPUT", str(input_path))
    monkeypatch.setenv("AGENT_OUTPUT", str(output_path))
    return output_path


def patch_client(monkeypatch, outcomes):
    created = {}

    def fake_openai(base_url, api_key, max_retries):
        created["base_url"] = base_url
        created["api_key"] = api_key
        created["max_retries"] = max_retries
        client = FakeClient(outcomes)
        created["client"] = client
        return client

    monkeypatch.setattr(m, "OpenAI", fake_openai)
    return created


def test_happy_path(monkeypatch, tmp_path, capsys):
    # 1 worker: with 4, either task could claim either fake outcome (flaky)
    monkeypatch.setattr(m, "MAX_WORKERS", 1)
    out = setup_env(monkeypatch, tmp_path, [
        {"task_id": "t1", "prompt": "a"},
        {"task_id": "t2", "prompt": "b"},
    ])
    created = patch_client(monkeypatch, [fake_response("A"), fake_response("B")])
    assert m.main() == 0
    results = {r["task_id"]: r["answer"] for r in json.loads(out.read_text())}
    assert results == {"t1": "A", "t2": "B"}
    assert created["base_url"] == "https://proxy.example/v1"
    assert created["api_key"] == "fk-test"
    assert created["max_retries"] == 0
    err = capsys.readouterr().err
    assert "stats:" in err and "total_tokens=30" in err
    assert "fireworks_calls=2" in err
    assert "probe_calls=0" in err
    assert "probe_prompt_tokens=0" in err
    assert "probe_completion_tokens=0" in err
    assert "task_prompt_tokens=20" in err
    assert "task_completion_tokens=10" in err


def test_one_task_failure_does_not_kill_run(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "RETRY_BACKOFF_SECONDS", 0)
    monkeypatch.setattr(m, "MAX_WORKERS", 1)  # deterministic outcome ordering
    out = setup_env(monkeypatch, tmp_path, [
        {"task_id": "t1", "prompt": "a"},
        {"task_id": "t2", "prompt": "b"},
    ])
    # t1 fails twice with retryable timeouts, then t2 succeeds
    patch_client(monkeypatch, [TimeoutError("x"), TimeoutError("y"), fake_response("B")])
    assert m.main() == 0
    results = {r["task_id"]: r["answer"] for r in json.loads(out.read_text())}
    assert results == {"t1": "", "t2": "B"}


def test_unanswerable_task_still_in_output(monkeypatch, tmp_path):
    out = setup_env(monkeypatch, tmp_path, [
        {"task_id": "t1"},
        {"task_id": "t2", "prompt": "b"},
    ])
    patch_client(monkeypatch, [fake_response("B")])
    assert m.main() == 0
    results = json.loads(out.read_text())
    assert results == [{"task_id": "t1", "answer": ""}, {"task_id": "t2", "answer": "B"}]


def test_unreadable_tasks_writes_empty_list_and_exits_1(monkeypatch, tmp_path):
    out = setup_env(monkeypatch, tmp_path, [])
    (tmp_path / "tasks.json").write_text("{not json", encoding="utf-8")
    patch_client(monkeypatch, [])
    assert m.main() == 1
    assert json.loads(out.read_text()) == []


def test_exhausted_budget_degrades_to_empty_answers(monkeypatch, tmp_path):
    out = setup_env(monkeypatch, tmp_path, [{"task_id": "t1", "prompt": "a"}])
    client_outcomes = [fake_response("never")]
    patch_client(monkeypatch, client_outcomes)
    monkeypatch.setattr(m, "SOFT_BUDGET_SECONDS", -1.0)  # deadline already passed
    assert m.main() == 0
    assert json.loads(out.read_text()) == [{"task_id": "t1", "answer": ""}]


def test_client_construction_failure_still_writes_results(monkeypatch, tmp_path):
    out = setup_env(monkeypatch, tmp_path, [{"task_id": "t1", "prompt": "a"}])
    def exploding_openai(**kwargs):
        raise RuntimeError("no client")
    monkeypatch.setattr(m, "OpenAI", exploding_openai)
    assert m.main() == 0
    assert json.loads(out.read_text()) == [{"task_id": "t1", "answer": ""}]


def test_local_lane_wired_through_main(monkeypatch, tmp_path, capsys):
    class StubLocal:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, user_text, max_tokens=160, deadline=None):
            return "local answer"

    monkeypatch.setattr("agent.local_model.LocalModel", StubLocal)
    routing_path = tmp_path / "routing.json"
    routing_path.write_text(json.dumps({"sentiment": "local"}), encoding="utf-8")
    monkeypatch.setenv("ROUTING_TABLE", str(routing_path))
    out = setup_env(monkeypatch, tmp_path, [
        {"task_id": "t1", "prompt": "Classify the sentiment of this review: great!"},
    ])
    # No outcomes are provided: selection and a successful local task must
    # make zero Fireworks calls.
    created = patch_client(monkeypatch, [])
    assert m.main() == 0
    results = {r["task_id"]: r["answer"] for r in json.loads(out.read_text())}
    assert results == {"t1": "local answer"}
    err = capsys.readouterr().err
    assert "local model loaded" in err
    assert created["client"].chat.completions.calls == []
    assert "probe_calls=0" in err and "fireworks_calls=0" in err

import json
import time

import pytest

import agent.main as m
from agent.batching import (
    MAX_BATCH_SIZE,
    batch_token_cap,
    batching_enabled,
    build_batch_messages,
    parse_batch_answers,
    plan_batches,
)
from tests.test_answer_task import FakeClient, fake_response
from tests.test_main_flow import patch_client, setup_env


FUTURE = time.monotonic() + 3600


def factual(task_id, prompt="What is the capital of France?"):
    return {"task_id": task_id, "prompt": prompt, "category": "factual"}


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_batching_requires_explicit_truthy_flag(value):
    assert batching_enabled(value)


@pytest.mark.parametrize("value", ["", "0", "false", "disabled", "2"])
def test_batching_is_off_for_other_values(value):
    assert not batching_enabled(value)


def test_plan_batches_only_same_category_cloud_tasks():
    tasks = [factual(f"f{i}") for i in range(6)] + [
        {"task_id": "m1", "prompt": "Calculate 2+2", "category": "math"},
        factual("local-f"),
    ]
    # Category routing is the unit of routing, so mark the separate local task
    # as a synthetic local-only category rather than mixing lanes in factual.
    tasks[-1]["category"] = "sentiment"
    batches, singles = plan_batches(tasks, {"sentiment": "local"})
    assert [len(batch) for batch in batches] == [3, 3]
    assert all({task["category"] for task in batch} == {"factual"} for batch in batches)
    assert {task["task_id"] for task in singles} == {"m1", "local-f"}


def test_plan_batches_does_not_batch_a_local_routed_category():
    tasks = [factual("f1"), factual("f2")]
    batches, singles = plan_batches(tasks, {"factual": "local"})
    assert batches == []
    assert singles == tasks


def test_single_eligible_task_stays_individual():
    task = factual("f1")
    batches, singles = plan_batches([task], {})
    assert batches == [] and singles == [task]


def test_two_tasks_stay_individual_but_three_form_a_batch():
    pair = [factual("f1"), factual("f2")]
    batches, singles = plan_batches(pair, {})
    assert batches == [] and singles == pair

    triple = pair + [factual("f3")]
    batches, singles = plan_batches(triple, {})
    assert [[task["task_id"] for task in batch] for batch in batches] == [
        ["f1", "f2", "f3"]
    ]
    assert singles == []


@pytest.mark.parametrize(
    "count,expected_sizes,expected_singles",
    [
        (2, [], 2),
        (3, [3], 0),
        (4, [4], 0),
        (5, [4], 1),
        (6, [3, 3], 0),
        (7, [4, 3], 0),
        (8, [4, 4], 0),
        (9, [3, 3, 3], 0),
        (10, [4, 3, 3], 0),
        (11, [4, 4, 3], 0),
        (12, [4, 4, 4], 0),
    ],
)
def test_planner_optimizes_three_and_four_task_groups(
    count, expected_sizes, expected_singles
):
    tasks = [factual(f"f{i}") for i in range(count)]
    batches, singles = plan_batches(tasks, {})
    assert [len(batch) for batch in batches] == expected_sizes
    assert len(singles) == expected_singles


def test_batch_prompt_is_strict_json_and_preserves_encoded_prompts():
    tasks = [factual("f1", 'What is "JSON"?'), factual("f2", "Define API.")]
    messages = build_batch_messages(tasks, "Answer in English.", lambda p, c: f"{p}\n{c}")
    assert messages[0]["role"] == "system"
    assert "exactly once and no other keys" in messages[0]["content"]
    payload = json.loads(messages[1]["content"])
    assert payload["tasks"][0] == {
        "task_id": "f1", "prompt": 'What is "JSON"?\nfactual'
    }


def test_parse_complete_batch_success():
    parsed = parse_batch_answers('{"f2":" B ","f1":"A"}', ["f1", "f2"])
    assert parsed.answers == {"f1": "A", "f2": "B"}
    assert parsed.fallback_ids == () and parsed.error is None


@pytest.mark.parametrize("content", ["not json", "[]", '"text"', "", "```json\n{}\n```"])
def test_malformed_or_non_object_response_falls_back_every_task(content):
    parsed = parse_batch_answers(content, ["f1", "f2"])
    assert parsed.answers == {}
    assert parsed.fallback_ids == ("f1", "f2")
    assert parsed.error


def test_missing_and_invalid_values_fall_back_individually():
    parsed = parse_batch_answers('{"f1":"Paris","f2":17}', ["f1", "f2", "f3"])
    assert parsed.answers == {"f1": "Paris"}
    assert parsed.fallback_ids == ("f2", "f3")


@pytest.mark.parametrize(
    "content,error_fragment",
    [
        ('{"f1":"A","f1":"B","f2":"C"}', "duplicate"),
        ('{"f1":"A","f2":"B","extra":"C"}', "unexpected"),
    ],
)
def test_duplicate_or_extra_ids_fail_closed_for_whole_batch(content, error_fragment):
    parsed = parse_batch_answers(content, ["f1", "f2"])
    assert parsed.answers == {}
    assert parsed.fallback_ids == ("f1", "f2")
    assert error_fragment in parsed.error


def test_batch_cap_scales_and_is_bounded():
    assert batch_token_cap(512, 2) == 1024
    assert batch_token_cap(2048, 4) == 4096


def test_answer_batch_success_is_one_call_with_usage_and_policy():
    client = FakeClient([
        fake_response('{"f1":"Paris","f2":"HTTP","f3":"Rome"}', 30, 12)
    ])
    outcome = m.answer_batch(
        client, "m-x", [factual("f1"), factual("f2"), factual("f3")], FUTURE,
        extra_body={"reasoning_effort": "low"},
    )
    assert outcome["answers"] == {"f1": "Paris", "f2": "HTTP", "f3": "Rome"}
    assert outcome["fallback_tasks"] == []
    assert outcome["prompt_tokens"] == 30 and outcome["completion_tokens"] == 12
    assert outcome["fireworks_calls"] == 1 and outcome["error"] is None
    call = client.chat.completions.calls[0]
    assert call["max_tokens"] == 1536
    assert call["timeout"] == m.FIRST_TIMEOUT_SECONDS
    assert call["extra_body"] == {"reasoning_effort": "low"}


def test_answer_batch_timeout_is_not_retried_and_falls_back_all():
    client = FakeClient([TimeoutError("slow")])
    tasks = [factual("f1"), factual("f2"), factual("f3")]
    outcome = m.answer_batch(client, "m-x", tasks, FUTURE)
    assert outcome["answers"] == {}
    assert outcome["fallback_tasks"] == tasks
    assert outcome["fireworks_calls"] == 1
    assert "TimeoutError" in outcome["error"]
    assert len(client.chat.completions.calls) == 1


def test_answer_batch_truncation_keeps_billed_usage_and_falls_back():
    response = fake_response('{"f1":"partial"', 40, 512)
    response.choices[0].finish_reason = "length"
    tasks = [factual("f1"), factual("f2"), factual("f3")]
    outcome = m.answer_batch(FakeClient([response]), "m-x", tasks, FUTURE)
    assert outcome["answers"] == {} and outcome["fallback_tasks"] == tasks
    assert outcome["prompt_tokens"] == 40 and outcome["completion_tokens"] == 512
    assert "truncated" in outcome["error"]


def test_answer_batch_rejects_experimentally_unprofitable_pair():
    client = FakeClient([])
    tasks = [factual("f1"), factual("f2")]
    outcome = m.answer_batch(client, "m-x", tasks, FUTURE)
    assert outcome["answers"] == {} and outcome["fallback_tasks"] == tasks
    assert outcome["fireworks_calls"] == 0
    assert "ineligible" in outcome["error"]
    assert client.chat.completions.calls == []


def _cloud_routing(monkeypatch, tmp_path):
    path = tmp_path / "routing.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("ROUTING_TABLE", str(path))


def test_main_batch_success_counts_shared_call_once(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(m, "MAX_WORKERS", 1)
    out = setup_env(monkeypatch, tmp_path, [
        {"task_id": "f1", "prompt": "What is the capital of France?"},
        {"task_id": "f2", "prompt": "Define HTTP."},
        {"task_id": "f3", "prompt": "What is the capital of Italy?"},
    ])
    monkeypatch.setenv("ENABLE_BATCHING", "1")
    _cloud_routing(monkeypatch, tmp_path)
    created = patch_client(monkeypatch, [
        fake_response(
            '{"f1":"Paris.","f2":"A web transfer protocol.","f3":"Rome."}',
            30, 12,
        )
    ])
    assert m.main() == 0
    assert json.loads(out.read_text()) == [
        {"task_id": "f1", "answer": "Paris."},
        {"task_id": "f2", "answer": "A web transfer protocol."},
        {"task_id": "f3", "answer": "Rome."},
    ]
    assert len(created["client"].chat.completions.calls) == 1
    err = capsys.readouterr().err
    assert "fireworks_calls=1" in err and "task_calls=1" in err
    assert "batch_calls=1" in err and "batch_accepted_tasks=3" in err
    assert "batch_fallback_tasks=0" in err and "total_tokens=42" in err


def test_main_partial_batch_falls_back_only_missing_task_and_counts_all_usage(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(m, "MAX_WORKERS", 1)
    out = setup_env(monkeypatch, tmp_path, [
        {"task_id": "f1", "prompt": "What is the capital of France?"},
        {"task_id": "f2", "prompt": "Define HTTP."},
        {"task_id": "f3", "prompt": "What is the capital of Italy?"},
    ])
    monkeypatch.setenv("ENABLE_BATCHING", "1")
    _cloud_routing(monkeypatch, tmp_path)
    created = patch_client(monkeypatch, [
        fake_response('{"f1":"Paris.","f3":"Rome."}', 20, 10),
        fake_response("A web transfer protocol.", 10, 5),
    ])
    assert m.main() == 0
    results = {item["task_id"]: item["answer"] for item in json.loads(out.read_text())}
    assert results == {
        "f1": "Paris.",
        "f2": "A web transfer protocol.",
        "f3": "Rome.",
    }
    assert len(created["client"].chat.completions.calls) == 2
    err = capsys.readouterr().err
    assert "fireworks_calls=2" in err and "task_calls=2" in err
    assert "batch_calls=1" in err and "batch_fallback_tasks=1" in err
    assert "task_prompt_tokens=30" in err
    assert "task_completion_tokens=15" in err and "total_tokens=45" in err
    assert "answered=3 failed=0" in err


def test_flag_off_keeps_individual_path_unchanged(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(m, "MAX_WORKERS", 1)
    out = setup_env(monkeypatch, tmp_path, [
        {"task_id": "f1", "prompt": "What is the capital of France?"},
        {"task_id": "f2", "prompt": "Define HTTP."},
    ])
    _cloud_routing(monkeypatch, tmp_path)
    created = patch_client(monkeypatch, [fake_response("Paris."), fake_response("HTTP.")])
    assert m.main() == 0
    assert len(created["client"].chat.completions.calls) == 2
    assert [item["answer"] for item in json.loads(out.read_text())] == ["Paris.", "HTTP."]
    err = capsys.readouterr().err
    assert "batching: enabled" not in err
    assert "batch_calls=0" in err and "batch_fallback_tasks=0" in err

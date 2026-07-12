import json
import re
import time

import agent.main as m
from agent.main import answer_task

from tests.test_answer_task import FakeClient, fake_response
from tests.test_main_flow import patch_client, setup_env

FUTURE = time.monotonic() + 3600


def test_answer_task_carries_category_and_lane():
    client = FakeClient([fake_response("4")])
    task = {"task_id": "t1", "prompt": "What is 2+2?", "category": "math"}
    r = answer_task(client, "m-x", task, FUTURE)
    assert r["category"] == "math"
    assert r["lane"] == "fireworks"


def test_answer_task_defaults_unknown_category():
    client = FakeClient([fake_response("4")])
    r = answer_task(client, "m-x", {"task_id": "t1", "prompt": "hm"}, FUTURE)
    assert r["category"] == "unknown"


def test_collector_logs_per_task_line(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(m, "MAX_WORKERS", 1)
    setup_env(monkeypatch, tmp_path, [
        {"task_id": "t1", "prompt": "Classify the sentiment of this review: great."},
    ])
    patch_client(monkeypatch, [fake_response("OK"), fake_response("Positive")])
    assert m.main() == 0
    err = capsys.readouterr().err
    line = next(l for l in err.splitlines() if l.startswith("task=t1"))
    assert re.fullmatch(
        r"task=t1 cat=sentiment pt=\d+ ct=\d+ calls=1 retries=0 lane=fireworks",
        line)


def test_fireworks_call_includes_category_constraint():
    from agent.router import CONSTRAINTS
    client = FakeClient([fake_response("Positive — praises battery.")])
    task = {"task_id": "t1", "prompt": "Classify the sentiment: great battery.",
            "category": "sentiment"}
    answer_task(client, "m-x", task, FUTURE)
    sent = client.chat.completions.calls[0]["messages"][1]["content"]
    assert sent.startswith("Classify the sentiment: great battery.")
    assert CONSTRAINTS["sentiment"] in sent


def test_system_prompt_is_short():
    assert len(m.SYSTEM_PROMPT) <= 60

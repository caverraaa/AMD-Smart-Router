import json
import time

from agent.main import answer_task, load_routing_table

from tests.test_answer_task import FakeClient, fake_response

FUTURE = time.monotonic() + 3600


class FakeLocal:
    def __init__(self, reply="", classify_reply=None):
        self.reply = reply
        self.classify_reply = classify_reply
        self.generate_calls = []

    def generate(self, user_text, max_tokens=160, deadline=None):
        self.generate_calls.append(user_text)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply

    def classify(self, prompt, categories):
        return self.classify_reply


def task(cat="sentiment"):
    return {
        "task_id": "t1",
        "prompt": "Classify the sentiment of this review: The product is great.",
        "category": cat,
    }


def test_load_routing_table(tmp_path):
    p = tmp_path / "rt.json"
    p.write_text(json.dumps({"sentiment": "local", "math": "fireworks", "bad": "nope"}))
    assert load_routing_table(str(p)) == {"sentiment": "local", "math": "fireworks"}
    assert load_routing_table(str(tmp_path / "missing.json")) == {}


def test_local_lane_success_zero_tokens():
    client = FakeClient([])  # any API call would crash on empty outcomes
    local = FakeLocal(reply="Positive — enthusiastic tone.")
    r = answer_task(client, "m-x", task(), FUTURE,
                    local=local, routing={"sentiment": "local"})
    assert r["answer"] == "Positive — enthusiastic tone."
    assert r["lane"] == "local"
    assert r["prompt_tokens"] == 0 and r["completion_tokens"] == 0
    assert client.chat.completions.calls == []


def test_local_empty_falls_back_to_fireworks():
    client = FakeClient([fake_response("Positive.")])
    local = FakeLocal(reply="")
    r = answer_task(client, "m-x", task(), FUTURE,
                    local=local, routing={"sentiment": "local"})
    assert r["answer"] == "Positive."
    assert r["lane"] == "fireworks"


def test_local_exception_falls_back_to_fireworks():
    client = FakeClient([fake_response("Positive.")])
    local = FakeLocal(reply=RuntimeError("llama crashed"))
    r = answer_task(client, "m-x", task(), FUTURE,
                    local=local, routing={"sentiment": "local"})
    assert r["answer"] == "Positive."
    assert r["lane"] == "fireworks"


def test_category_routed_to_fireworks_never_touches_local():
    client = FakeClient([fake_response("42")])
    local = FakeLocal(reply="should not be used")
    r = answer_task(client, "m-x", task("math"), FUTURE,
                    local=local, routing={"sentiment": "local"})
    assert r["lane"] == "fireworks"
    assert local.generate_calls == []


def test_unknown_category_always_goes_cloud():
    client = FakeClient([fake_response("cloud answer")])
    local = FakeLocal(reply="should not be used", classify_reply="sentiment")
    r = answer_task(client, "m-x", task("unknown"), FUTURE,
                    local=local, routing={"sentiment": "local"})
    assert r["lane"] == "fireworks"
    assert r["answer"] == "cloud answer"
    assert local.generate_calls == []


def test_no_local_model_behaves_as_before():
    client = FakeClient([fake_response("A")])
    r = answer_task(client, "m-x", task(), FUTURE)
    assert r["answer"] == "A" and r["lane"] == "fireworks"


def test_near_deadline_skips_local_lane_goes_cloud():
    client_with_one_response = FakeClient([fake_response("cloud answer")])
    local = FakeLocal(reply="x")
    r = answer_task(client_with_one_response, "m-x", task("sentiment"),
                    time.monotonic() + 5.0, local=local, routing={"sentiment": "local"})
    assert r["lane"] == "fireworks"
    assert local.generate_calls == []


def test_risk_gate_skips_unsupported_task_even_when_category_is_local():
    client = FakeClient([fake_response("Positive and negative.")])
    local = FakeLocal(reply="unsafe local answer")
    unsupported = {
        "task_id": "t1",
        "prompt": "Classify each review separately as JSON: Great. Terrible.",
        "category": "sentiment",
    }

    result = answer_task(
        client, "m-x", unsupported, FUTURE,
        local=local, routing={"sentiment": "local"},
    )

    assert result["lane"] == "fireworks"
    assert result["risk_profile"] is None
    assert local.generate_calls == []


def test_deterministic_summary_uses_short_reserve_near_deadline():
    source = (
        "Remote work changed office planning and reduced commuting for many staff. "
        "Companies adopted flexible schedules and hired from a wider geographic area. "
        "Employees gained autonomy but reported weaker boundaries between work and home."
    )
    summary_task = {
        "task_id": "s1",
        "prompt": "Summarize the following in exactly one sentence: " + source,
        "category": "summarisation",
    }
    local = FakeLocal(reply=source.replace(". ", "; "))
    result = answer_task(
        FakeClient([]), "m-x", summary_task, time.monotonic() + 2.0,
        local=local, routing={"summarisation": "local"},
    )
    assert result["lane"] == "local"
    assert result["verifiability"] == "validated"
    assert len(local.generate_calls) == 1

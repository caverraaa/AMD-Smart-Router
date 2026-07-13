import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_routes_are_accuracy_first():
    routes = json.loads((ROOT / "agent" / "routing_table.json").read_text())
    assert set(routes) == {
        "sentiment", "ner", "factual", "logic", "math",
        "summarisation", "code_debug", "code_gen",
    }
    assert set(routes.values()) == {"fireworks"}


def test_submission_image_is_reproducible_and_disables_batching():
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "ENV ENABLE_BATCHING=0" in dockerfile
    assert "llama-cpp-python" not in dockerfile
    assert "COPY models/" not in dockerfile

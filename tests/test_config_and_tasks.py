import json

import pytest

from agent.main import load_config, load_tasks

GOOD_ENV = {
    "FIREWORKS_API_KEY": "fk-test",
    "FIREWORKS_BASE_URL": "https://proxy.example/v1",
    "ALLOWED_MODELS": "accounts/x/small-2b, accounts/x/big-70b,,",
}


def set_env(monkeypatch, overrides=None):
    env = {**GOOD_ENV, **(overrides or {})}
    for k in GOOD_ENV:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        if v is not None:
            monkeypatch.setenv(k, v)


def test_load_config_happy_path(monkeypatch):
    set_env(monkeypatch)
    cfg = load_config()
    assert cfg["api_key"] == "fk-test"
    assert cfg["base_url"] == "https://proxy.example/v1"
    assert cfg["allowed_models"] == ["accounts/x/small-2b", "accounts/x/big-70b"]


@pytest.mark.parametrize("missing", ["FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS"])
def test_load_config_missing_var_exits_1(monkeypatch, capsys, missing):
    set_env(monkeypatch, {missing: None})
    with pytest.raises(SystemExit) as e:
        load_config()
    assert e.value.code == 1
    assert missing in capsys.readouterr().err


def test_load_config_empty_model_list_exits_1(monkeypatch):
    set_env(monkeypatch, {"ALLOWED_MODELS": " , ,"})
    with pytest.raises(SystemExit) as e:
        load_config()
    assert e.value.code == 1


def write_tasks(tmp_path, data):
    p = tmp_path / "tasks.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


def test_load_tasks_happy_path(tmp_path):
    path = write_tasks(tmp_path, [
        {"task_id": "t1", "prompt": "hello"},
        {"task_id": "t2", "prompt": "world"},
    ])
    task_ids, answerable = load_tasks(path)
    assert task_ids == ["t1", "t2"]
    assert answerable == [{"task_id": "t1", "prompt": "hello"}, {"task_id": "t2", "prompt": "world"}]


def test_load_tasks_entry_without_prompt_kept_in_ids_only(tmp_path):
    path = write_tasks(tmp_path, [{"task_id": "t1"}, {"task_id": "t2", "prompt": "ok"}])
    task_ids, answerable = load_tasks(path)
    assert task_ids == ["t1", "t2"]
    assert [t["task_id"] for t in answerable] == ["t2"]


def test_load_tasks_entry_without_task_id_skipped(tmp_path):
    path = write_tasks(tmp_path, [{"prompt": "orphan"}, {"task_id": "t2", "prompt": "ok"}, "junk"])
    task_ids, answerable = load_tasks(path)
    assert task_ids == ["t2"]
    assert [t["task_id"] for t in answerable] == ["t2"]


def test_load_tasks_non_list_raises(tmp_path):
    path = write_tasks(tmp_path, {"task_id": "t1"})
    with pytest.raises(ValueError):
        load_tasks(path)

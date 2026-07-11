import json
import os

from agent.main import write_snapshot


def read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_writes_all_ids_in_order(tmp_path):
    path = str(tmp_path / "results.json")
    write_snapshot(["t1", "t2", "t3"], {"t2": "two"}, path)
    assert read(path) == [
        {"task_id": "t1", "answer": ""},
        {"task_id": "t2", "answer": "two"},
        {"task_id": "t3", "answer": ""},
    ]


def test_coerces_none_answer_to_empty_string(tmp_path):
    path = str(tmp_path / "results.json")
    write_snapshot(["t1"], {"t1": None}, path)
    assert read(path) == [{"task_id": "t1", "answer": ""}]


def test_no_tmp_file_left_behind(tmp_path):
    path = str(tmp_path / "results.json")
    write_snapshot(["t1"], {"t1": "x"}, path)
    assert os.listdir(tmp_path) == ["results.json"]


def test_overwrite_keeps_file_valid(tmp_path):
    path = str(tmp_path / "results.json")
    write_snapshot(["t1"], {}, path)
    write_snapshot(["t1"], {"t1": "answered"}, path)
    assert read(path) == [{"task_id": "t1", "answer": "answered"}]


def test_empty_task_list_writes_empty_json_list(tmp_path):
    path = str(tmp_path / "results.json")
    write_snapshot([], {}, path)
    assert read(path) == []

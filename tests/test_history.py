import json

from src import history


def test_history_round_trip_and_limit(tmp_path, monkeypatch):
    target = tmp_path / "history.json"
    monkeypatch.setattr(history, "HISTORY_PATH", target)
    history.save_history({"run": 1})
    history.save_history({"run": 2})
    assert history.load_history() == [{"run": 1}, {"run": 2}]
    history.save_history({"run": 3}, limit=2)
    assert history.load_history() == [{"run": 2}, {"run": 3}]
    assert json.loads(target.read_text(encoding="utf-8"))[-1]["run"] == 3

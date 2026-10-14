"""Small local JSON history store; no trained models or user data are persisted."""
from __future__ import annotations

import json
from pathlib import Path

HISTORY_PATH = Path("results") / "history.json"


def load_history():
    try:
        if not HISTORY_PATH.exists():
            return []
        value = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        return value if isinstance(value, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def save_history(record, limit=100):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    history = load_history()
    history.append(record)
    history = history[-limit:]
    temp_path = HISTORY_PATH.with_suffix(".tmp")
    temp_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(HISTORY_PATH)

"""Append-only CSV event log for state-change events (empty / present / moving).

Writes one row per debounced state transition; exposes a helper to read back
the most recent rows for display in the UI.
"""

import csv
import datetime
from pathlib import Path

LOG_PATH = Path(__file__).parent.parent / "events.csv"
_HEADER = ["timestamp", "state", "room", "confidence"]


def log_event(state: str, room: str | None, confidence: float):
    """Append one state-transition row to events.csv, writing the header on first use."""
    first_write = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if first_write:
            writer.writerow(_HEADER)
        writer.writerow([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            state,
            room or "",
            f"{confidence:.2f}",
        ])


def load_recent(n: int = 8) -> list[dict]:
    """Return the last n rows as dicts, newest first."""
    if not LOG_PATH.exists():
        return []
    try:
        with LOG_PATH.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return list(reversed(rows[-n:]))
    except Exception:
        return []

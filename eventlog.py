import csv
import datetime
from pathlib import Path

LOG_PATH = Path(__file__).parent / "events.csv"
_HEADER = ["timestamp", "state", "room", "confidence"]


def log_event(state: str, room: str | None, confidence: float):
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

"""KNN room fingerprinter — maps a live RSSI snapshot to a named location.

Each room accumulates one or more 15-second recording sessions.  Classification
uses penalised Euclidean distance over the common network subset so rooms
recorded with different visible APs still compare meaningfully.  State is
persisted to rooms.json and auto-migrated from the legacy centroid-only format.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

RECORD_SECONDS = 15
SAVE_PATH = Path(__file__).parent / "rooms.json"

K = 5                    # neighbours to vote on
MIN_CONFIDENCE = 0.45    # below this the classifier abstains ("Unknown")


@dataclass
class RoomFingerprint:
    name: str
    sessions: int                          # number of recording passes
    samples: list[dict[str, float]] = field(default_factory=list)


def _distance(a: dict[str, float], b: dict[str, float]) -> float | None:
    """Penalised Euclidean distance on the common network subset.
    Returns None when the two dicts share no networks."""
    common = set(a) & set(b)
    if not common:
        return None
    dist = math.sqrt(sum((a[s] - b[s]) ** 2 for s in common))
    coverage = len(common) / max(len(a), len(b))
    return dist / max(coverage, 0.01)


class Fingerprinter:
    def __init__(self):
        self.rooms: dict[str, RoomFingerprint] = {}
        self._load()

    # ── recording ─────────────────────────────────────────────────────────────

    def record(self, name: str, samples: list[dict[str, float]]):
        """Append a new recording session to a room (creates room if new)."""
        if name in self.rooms:
            self.rooms[name].samples.extend(samples)
            self.rooms[name].sessions += 1
        else:
            self.rooms[name] = RoomFingerprint(name=name, sessions=1, samples=list(samples))
        self._save()

    def delete(self, name: str):
        self.rooms.pop(name, None)
        self._save()

    # ── classification ────────────────────────────────────────────────────────

    def classify(self, current: dict[str, float]) -> tuple[str, float] | None:
        """
        K-nearest-neighbour classification across all stored samples.

        Returns (room_name, confidence) where confidence is votes/K (0–1),
        or ("Unknown", confidence) when below MIN_CONFIDENCE,
        or None when no rooms are trained.
        """
        if not self.rooms:
            return None

        # Collect (distance, room_name) for every stored sample
        neighbours: list[tuple[float, str]] = []
        for room in self.rooms.values():
            for sample in room.samples:
                d = _distance(current, sample)
                if d is not None:
                    neighbours.append((d, room.name))

        if not neighbours:
            return None

        neighbours.sort(key=lambda x: x[0])
        k_nearest = neighbours[:K]

        # Inverse-distance weighted vote
        weights: dict[str, float] = {}
        for dist, room_name in k_nearest:
            weights[room_name] = weights.get(room_name, 0.0) + 1.0 / max(dist, 0.01)

        winner = max(weights, key=weights.__getitem__)
        confidence = weights[winner] / sum(weights.values())

        if confidence < MIN_CONFIDENCE:
            return "Unknown", confidence

        return winner, confidence

    # ── persistence ───────────────────────────────────────────────────────────

    def _save(self):
        """Serialise all rooms (with their raw sample lists) to rooms.json."""
        data = {
            name: {"sessions": fp.sessions, "samples": fp.samples}
            for name, fp in self.rooms.items()
        }
        SAVE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load(self):
        """Deserialise rooms.json, migrating the old centroid-only format if needed."""
        if not SAVE_PATH.exists():
            return
        try:
            data = json.loads(SAVE_PATH.read_text(encoding="utf-8"))
            loaded = {}
            for name, val in data.items():
                # Handle both old format (plain dict of means) and new format
                if isinstance(val, dict) and "samples" in val:
                    loaded[name] = RoomFingerprint(
                        name=name,
                        sessions=val.get("sessions", 1),
                        samples=val["samples"],
                    )
                else:
                    # Migrate old centroid-only format: treat the mean as one sample
                    loaded[name] = RoomFingerprint(name=name, sessions=1, samples=[val])
            self.rooms = loaded
        except Exception:
            self.rooms = {}

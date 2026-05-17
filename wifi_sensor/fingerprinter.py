"""KNN room fingerprinter — maps a live RSSI snapshot to a named location.

Each room accumulates one or more 15-second recording sessions.  Classification
uses penalised Euclidean distance over the common network subset so rooms
recorded with different visible APs still compare meaningfully.  State is
persisted to rooms.json and auto-migrated from the legacy centroid-only format.
"""

import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

RECORD_SECONDS = 15
SAVE_PATH = Path(__file__).parent.parent / "rooms.json"

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
    def __init__(
        self,
        save_path: Path | None = None,
        k: int | None = None,
        min_confidence: float | None = None,
        max_samples: int = 500,
    ):
        self.save_path      = SAVE_PATH      if save_path      is None else save_path
        self.k              = K              if k              is None else k
        self.min_confidence = MIN_CONFIDENCE if min_confidence is None else min_confidence
        self.max_samples    = max_samples
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
        fp = self.rooms[name]
        if len(fp.samples) > self.max_samples:
            fp.samples = fp.samples[-self.max_samples:]
        logger.info("recorded room %r  sessions=%d  total_samples=%d", name, fp.sessions, len(fp.samples))
        self._save()

    def delete(self, name: str):
        if name not in self.rooms:
            return
        self.rooms.pop(name)
        logger.info("deleted room %r", name)
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
        k_nearest = neighbours[:self.k]

        # Inverse-distance weighted vote
        weights: dict[str, float] = {}
        for dist, room_name in k_nearest:
            weights[room_name] = weights.get(room_name, 0.0) + 1.0 / max(dist, 0.01)

        winner = max(weights, key=weights.__getitem__)
        confidence = weights[winner] / sum(weights.values())

        if confidence < self.min_confidence:
            return "Unknown", confidence

        return winner, confidence

    # ── persistence ───────────────────────────────────────────────────────────

    def _save(self):
        """Serialise all rooms to save_path atomically via a temp file + rename."""
        data = {
            name: {"sessions": fp.sessions, "samples": fp.samples}
            for name, fp in self.rooms.items()
        }
        fd, tmp = tempfile.mkstemp(dir=self.save_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self.save_path)
        except Exception:
            os.unlink(tmp)
            raise

    def _load(self):
        """Deserialise save_path, migrating the old centroid-only format if needed."""
        if not self.save_path.exists():
            return
        try:
            data = json.loads(self.save_path.read_text(encoding="utf-8"))
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
            logger.exception("failed to load %s — starting with empty rooms", self.save_path)
            self.rooms = {}

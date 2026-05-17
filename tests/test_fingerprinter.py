"""Tests for fingerprinter.py — KNN room classification and persistence."""
import json
import tempfile
import unittest
from pathlib import Path

from wifi_sensor.fingerprinter import Fingerprinter, RoomFingerprint, _distance


SAMPLE_A = {"net1": -50.0, "net2": -70.0}
SAMPLE_B = {"net1": -51.0, "net2": -71.0}   # near A
SAMPLE_C = {"net1": -90.0, "net2": -30.0}   # far from A


class TestDistance(unittest.TestCase):
    def test_identical_dicts_zero_distance(self):
        self.assertAlmostEqual(_distance(SAMPLE_A, SAMPLE_A), 0.0)

    def test_nearby_dicts_small_distance(self):
        d = _distance(SAMPLE_A, SAMPLE_B)
        self.assertIsNotNone(d)
        self.assertLess(d, 5.0)

    def test_far_dicts_large_distance(self):
        d = _distance(SAMPLE_A, SAMPLE_C)
        near = _distance(SAMPLE_A, SAMPLE_B)
        self.assertGreater(d, near)

    def test_no_common_keys_returns_none(self):
        self.assertIsNone(_distance({"a": -50}, {"b": -50}))


class TestRecord(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mktemp(suffix=".json"))
        self.fp = Fingerprinter(save_path=self._tmp)

    def tearDown(self):
        self._tmp.unlink(missing_ok=True)

    def test_record_creates_room(self):
        self.fp.record("living", [SAMPLE_A] * 5)
        self.assertIn("living", self.fp.rooms)

    def test_record_appends_to_existing_room(self):
        self.fp.record("living", [SAMPLE_A] * 5)
        self.fp.record("living", [SAMPLE_B] * 5)
        self.assertEqual(self.fp.rooms["living"].sessions, 2)
        self.assertEqual(len(self.fp.rooms["living"].samples), 10)

    def test_record_caps_samples_at_max(self):
        fp = Fingerprinter(save_path=self._tmp, max_samples=8)
        fp.record("living", [SAMPLE_A] * 5)
        fp.record("living", [SAMPLE_B] * 5)
        self.assertLessEqual(len(fp.rooms["living"].samples), 8)

    def test_delete_removes_room(self):
        self.fp.record("living", [SAMPLE_A] * 5)
        self.fp.delete("living")
        self.assertNotIn("living", self.fp.rooms)

    def test_delete_nonexistent_no_error(self):
        self.fp.delete("nonexistent")


class TestClassify(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mktemp(suffix=".json"))
        self.fp = Fingerprinter(save_path=self._tmp)
        self.fp.record("room_a", [SAMPLE_A] * 15)
        self.fp.record("room_c", [SAMPLE_C] * 15)

    def tearDown(self):
        self._tmp.unlink(missing_ok=True)

    def test_classify_returns_none_when_no_rooms(self):
        empty = Fingerprinter(save_path=Path(tempfile.mktemp(suffix=".json")))
        self.assertIsNone(empty.classify(SAMPLE_A))

    def test_classify_near_a_returns_room_a(self):
        result = self.fp.classify(SAMPLE_B)
        self.assertIsNotNone(result)
        room, conf = result
        self.assertEqual(room, "room_a")
        self.assertGreater(conf, 0.5)

    def test_confidence_between_0_and_1(self):
        result = self.fp.classify(SAMPLE_A)
        if result:
            _, conf = result
            self.assertGreaterEqual(conf, 0.0)
            self.assertLessEqual(conf, 1.0)


class TestPersistence(unittest.TestCase):
    def test_save_and_load(self):
        tmp = Path(tempfile.mktemp(suffix=".json"))
        try:
            fp = Fingerprinter(save_path=tmp)
            fp.record("kitchen", [SAMPLE_A] * 5)
            fp2 = Fingerprinter(save_path=tmp)
            self.assertIn("kitchen", fp2.rooms)
            self.assertEqual(len(fp2.rooms["kitchen"].samples), 5)
        finally:
            tmp.unlink(missing_ok=True)

    def test_migrate_old_format(self):
        tmp = Path(tempfile.mktemp(suffix=".json"))
        try:
            tmp.write_text(json.dumps({"hall": {"net1": -60.0, "net2": -70.0}}))
            fp = Fingerprinter(save_path=tmp)
            self.assertIn("hall", fp.rooms)
            self.assertEqual(fp.rooms["hall"].sessions, 1)
            self.assertEqual(len(fp.rooms["hall"].samples), 1)
        finally:
            tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

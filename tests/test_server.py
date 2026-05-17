"""Tests for server.py — Flask API endpoints and dashboard HTML correctness."""
import json
import re
import unittest
from unittest.mock import patch

import server


class TestDashboardHTML(unittest.TestCase):
    """Verify the served HTML is structurally correct and JS-safe."""

    def setUp(self):
        server.app.testing = True
        self.client = server.app.test_client()

    def test_index_returns_200(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)

    def test_content_type_html_utf8(self):
        r = self.client.get("/")
        self.assertIn("text/html", r.content_type)
        self.assertIn("utf-8", r.content_type.lower())

    def test_no_cache_header(self):
        r = self.client.get("/")
        cc = r.headers.get("Cache-Control", "")
        self.assertIn("no-cache", cc)

    def test_no_broken_backslash_regex(self):
        # The old code had /\\/g which Python mangled to /\/g — invalid in JS
        html = self.client.get("/").data.decode("utf-8")
        self.assertNotIn(r"replace(/\/g,", html, "Broken backslash regex found in HTML")

    def test_uses_data_ssid_attribute(self):
        html = self.client.get("/").data.decode("utf-8")
        self.assertIn("data-ssid", html)
        self.assertIn("this.dataset.ssid", html)

    def test_offline_overlay_present(self):
        html = self.client.get("/").data.decode("utf-8")
        self.assertIn('id="offline"', html)

    def test_setinterval_refresh_present(self):
        html = self.client.get("/").data.decode("utf-8")
        self.assertIn("setInterval(refresh", html)

    def test_chart_js_async(self):
        html = self.client.get("/").data.decode("utf-8")
        # Chart.js script tag should have async attribute
        self.assertRegex(html, r"chart\.js.*async|async.*chart\.js")

    def test_viewport_meta_present(self):
        html = self.client.get("/").data.decode("utf-8")
        self.assertIn('name="viewport"', html)


class TestAPIStatus(unittest.TestCase):
    def setUp(self):
        server.app.testing = True
        self.client = server.app.test_client()

    def test_status_returns_200(self):
        r = self.client.get("/api/status")
        self.assertEqual(r.status_code, 200)

    def test_status_json_shape(self):
        r = self.client.get("/api/status")
        d = r.get_json()
        for field in ("state", "networks", "rooms", "threshold", "focused_ssid",
                      "score_history", "events", "bands", "recording"):
            self.assertIn(field, d, msg=f"Missing field: {field}")

    def test_threshold_is_float(self):
        d = self.client.get("/api/status").get_json()
        self.assertIsInstance(d["threshold"], float)

    def test_focused_ssid_initially_null(self):
        server._focused_ssid = None
        d = self.client.get("/api/status").get_json()
        self.assertIsNone(d["focused_ssid"])


class TestAPIFocus(unittest.TestCase):
    def setUp(self):
        server.app.testing = True
        self.client = server.app.test_client()
        server._focused_ssid = None

    def tearDown(self):
        server._focused_ssid = None

    def test_set_focus(self):
        r = self.client.post("/api/focus",
                             data=json.dumps({"ssid": "TestNet [aa:bb:cc:dd:ee:ff]"}),
                             content_type="application/json")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(server._focused_ssid, "TestNet [aa:bb:cc:dd:ee:ff]")

    def test_clear_focus_with_null(self):
        server._focused_ssid = "SomeNet"
        self.client.post("/api/focus",
                         data=json.dumps({"ssid": None}),
                         content_type="application/json")
        self.assertIsNone(server._focused_ssid)

    def test_clear_focus_with_empty_string(self):
        server._focused_ssid = "SomeNet"
        self.client.post("/api/focus",
                         data=json.dumps({"ssid": ""}),
                         content_type="application/json")
        self.assertIsNone(server._focused_ssid)


class TestAPIThreshold(unittest.TestCase):
    def setUp(self):
        server.app.testing = True
        self.client = server.app.test_client()
        self._orig = server._threshold

    def tearDown(self):
        server._threshold = self._orig

    def test_set_threshold(self):
        r = self.client.post("/api/threshold",
                             data=json.dumps({"value": 3.0}),
                             content_type="application/json")
        self.assertEqual(r.status_code, 200)
        self.assertAlmostEqual(server._threshold, 3.0)

    def test_threshold_clamped_to_min(self):
        self.client.post("/api/threshold",
                         data=json.dumps({"value": 0.0}),
                         content_type="application/json")
        self.assertGreaterEqual(server._threshold, 0.5)

    def test_threshold_clamped_to_max(self):
        self.client.post("/api/threshold",
                         data=json.dumps({"value": 99.0}),
                         content_type="application/json")
        self.assertLessEqual(server._threshold, 5.0)

    def test_missing_value_returns_400(self):
        r = self.client.post("/api/threshold",
                             data=json.dumps({}),
                             content_type="application/json")
        self.assertEqual(r.status_code, 400)


class TestAPIRecalibrate(unittest.TestCase):
    def setUp(self):
        server.app.testing = True
        self.client = server.app.test_client()

    def test_recalibrate_returns_ok(self):
        r = self.client.post("/api/recalibrate")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.get_json()["ok"])

    def test_recalibrate_clears_score_history(self):
        server._score_history.append(1.5)
        self.client.post("/api/recalibrate")
        self.assertEqual(len(server._score_history), 0)


class TestAPIRecord(unittest.TestCase):
    def setUp(self):
        server.app.testing = True
        self.client = server.app.test_client()
        server._recording = False

    def tearDown(self):
        server._recording = False

    def test_missing_name_returns_400(self):
        r = self.client.post("/api/record",
                             data=json.dumps({}),
                             content_type="application/json")
        self.assertEqual(r.status_code, 400)

    def test_start_recording(self):
        r = self.client.post("/api/record",
                             data=json.dumps({"name": "kitchen"}),
                             content_type="application/json")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(server._recording)

    def test_already_recording_returns_409(self):
        server._recording = True
        r = self.client.post("/api/record",
                             data=json.dumps({"name": "hall"}),
                             content_type="application/json")
        self.assertEqual(r.status_code, 409)


if __name__ == "__main__":
    unittest.main()

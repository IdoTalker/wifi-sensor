"""Tests for scanner.py — netsh output parsing and RSSI conversion."""
import unittest
from unittest.mock import patch, MagicMock

from wifi_sensor.scanner import scan_networks


SAMPLE_NETSH_OUTPUT = b"""
SSID 1 : HomeNetwork
 Network type            : Infrastructure
 Authentication          : WPA2-Personal
 Encryption              : CCMP
 BSSID 1                 : aa:bb:cc:dd:ee:ff
      Signal             : 80%
      Channel            : 6
SSID 2 : OfficeNet
 Network type            : Infrastructure
 Authentication          : WPA3-Personal
 Encryption              : CCMP
 BSSID 1                 : 11:22:33:44:55:66
      Signal             : 50%
      Channel            : 11
 BSSID 2                 : 11:22:33:44:55:77
      Signal             : 20%
      Channel            : 11
"""


class TestScanNetworks(unittest.TestCase):
    def _mock_run(self, stdout=SAMPLE_NETSH_OUTPUT, returncode=0):
        mock = MagicMock()
        mock.stdout = stdout
        mock.returncode = returncode
        return mock

    def test_returns_dict(self):
        with patch("wifi_sensor.scanner.subprocess.run", return_value=self._mock_run()):
            result = scan_networks()
        self.assertIsInstance(result, dict)

    def test_parses_all_bssids(self):
        with patch("wifi_sensor.scanner.subprocess.run", return_value=self._mock_run()):
            result = scan_networks()
        self.assertEqual(len(result), 3)

    def test_key_format_ssid_bssid(self):
        with patch("wifi_sensor.scanner.subprocess.run", return_value=self._mock_run()):
            result = scan_networks()
        for key in result:
            self.assertRegex(key, r".+ \[.+\]", msg=f"Key '{key}' is not 'SSID [bssid]' format")

    def test_rssi_conversion_80pct(self):
        with patch("wifi_sensor.scanner.subprocess.run", return_value=self._mock_run()):
            result = scan_networks()
        key = "HomeNetwork [aa:bb:cc:dd:ee:ff]"
        self.assertIn(key, result)
        self.assertAlmostEqual(result[key], (80 / 2) - 100)  # = -60.0

    def test_rssi_conversion_50pct(self):
        with patch("wifi_sensor.scanner.subprocess.run", return_value=self._mock_run()):
            result = scan_networks()
        key = "OfficeNet [11:22:33:44:55:66]"
        self.assertAlmostEqual(result[key], (50 / 2) - 100)  # = -75.0

    def test_rssi_in_valid_range(self):
        with patch("wifi_sensor.scanner.subprocess.run", return_value=self._mock_run()):
            result = scan_networks()
        for key, rssi in result.items():
            self.assertGreaterEqual(rssi, -100.0, msg=f"{key}: {rssi} below -100")
            self.assertLessEqual(rssi, 0.0, msg=f"{key}: {rssi} above 0")

    def test_empty_output_returns_empty_dict(self):
        with patch("wifi_sensor.scanner.subprocess.run", return_value=self._mock_run(stdout=b"")):
            result = scan_networks()
        self.assertEqual(result, {})

    def test_timeout_returns_empty_dict(self):
        import subprocess
        with patch("wifi_sensor.scanner.subprocess.run", side_effect=subprocess.TimeoutExpired("netsh", 5)):
            result = scan_networks()
        self.assertEqual(result, {})

    def test_not_found_returns_empty_dict(self):
        with patch("wifi_sensor.scanner.subprocess.run", side_effect=FileNotFoundError):
            result = scan_networks()
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()

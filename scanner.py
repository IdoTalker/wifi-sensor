"""Windows Wi-Fi scanner — wraps `netsh wlan` to return a {ssid: rssi_dbm} snapshot."""

import subprocess
import re


def scan_networks() -> dict[str, float]:
    """Return {ssid: rssi_dbm} for all visible networks."""
    try:
        result = subprocess.run(
            ["netsh", "wlan", "show", "networks", "mode=bssid"],
            capture_output=True, timeout=5
        )
        output = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {}

    networks: dict[str, float] = {}
    current_ssid = None

    for line in output.splitlines():
        ssid_match = re.match(r"^\s*SSID\s+\d+\s*:\s*(.+)", line)
        signal_match = re.match(r"^\s*Signal\s*:\s*(\d+)%", line)

        if ssid_match:
            current_ssid = ssid_match.group(1).strip()
        elif signal_match and current_ssid:
            pct = int(signal_match.group(1))
            dbm = (pct / 2) - 100  # netsh reports 0–100 %; Microsoft's documented conversion to dBm
            # Keep best signal if SSID appears multiple times (multiple BSSIDs)
            if current_ssid not in networks or dbm > networks[current_ssid]:
                networks[current_ssid] = dbm

    return networks

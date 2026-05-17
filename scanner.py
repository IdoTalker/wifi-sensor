"""Windows Wi-Fi scanner — wraps `netsh wlan` to return a {ssid: rssi_dbm} snapshot."""

import subprocess
import re


def scan_networks() -> dict[str, float]:
    """Return {"SSID [bssid]": rssi_dbm} for every visible BSSID."""
    try:
        result = subprocess.run(
            ["netsh", "wlan", "show", "networks", "mode=bssid"],
            capture_output=True, timeout=5
        )
        output = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {}

    networks: dict[str, float] = {}
    current_ssid: str | None = None
    current_bssid: str | None = None

    for line in output.splitlines():
        ssid_match   = re.match(r"^\s*SSID\s+\d+\s*:\s*(.+)", line)
        bssid_match  = re.match(r"^\s*BSSID\s+\d+\s*:\s*(.+)", line)
        signal_match = re.match(r"^\s*Signal\s*:\s*(\d+)%", line)

        if ssid_match:
            current_ssid  = ssid_match.group(1).strip()
            current_bssid = None
        elif bssid_match:
            current_bssid = bssid_match.group(1).strip()
        elif signal_match and current_ssid and current_bssid:
            pct = int(signal_match.group(1))
            dbm = (pct / 2) - 100  # netsh 0–100 % → dBm (Microsoft documented conversion)
            networks[f"{current_ssid} [{current_bssid}]"] = dbm

    return networks

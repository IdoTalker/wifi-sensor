# Wi-Fi Presence Detector — Agent Handoff

## What this is
A desktop + web app that uses Wi-Fi RSSI signal fluctuations to detect presence and motion in the house, without any special hardware. Runs on Windows, reads from standard home router via `netsh`.

## Current state: feature-complete on RSSI
All meaningful software improvements have been implemented. The next step is a hardware upgrade (ESP32 CSI) which the user has ordered.

---

## File map

| File | Role |
|---|---|
| `scanner.py` | Polls `netsh wlan show networks`, returns `{ssid: rssi_dbm}` |
| `detector.py` | Per-network `MotionDetector` — mean-shift + variance Z-score, adaptive baseline |
| `classifier.py` | FFT on fused score history → 3-state: `empty / present / moving` |
| `fingerprinter.py` | KNN room fingerprinting — multi-session, saves to `rooms.json` |
| `eventlog.py` | Appends state-change events to `events.csv`, loads recent rows |
| `main.py` | Tkinter desktop GUI — run with `python main.py` |
| `server.py` | Flask web dashboard — run with `python server.py`, open http://localhost:5000 |
| `requirements.txt` | `matplotlib`, `numpy`, `flask` |

---

## How to run

```
cd J:\Ido\claude\wifi-sensor
pip install -r requirements.txt

python main.py      # desktop GUI (Tkinter)
python server.py    # web dashboard, any device on network can open it
```

---

## Architecture

```
scan_networks()  →  MotionDetector (per network)  →  fused max score
                                                          ↓
                                              fft_classify(scores, threshold)
                                                          ↓
                                              empty / present / moving
                                                          ↓
                                              Fingerprinter.classify()  →  room name
                                                          ↓
                                              log_event() → events.csv
```

### Detection pipeline
1. `scanner.py` polls `netsh` every 1 s → `{ssid: rssi_dbm}`
2. Each SSID has its own `MotionDetector` that calibrates for 30 s, then scores anomalies
3. Fused score = max across all calibrated detectors
4. `classifier.py` runs FFT on last 120 scores → 3-state classification
5. `fingerprinter.py` does KNN on current RSSI vector → room name + confidence

### Key constants (detector.py)
- `CALIBRATION_SAMPLES = 30` — quiet seconds needed before detection starts
- `WINDOW_SIZE = 60` — rolling window for history
- `VAR_WINDOW = 8` — short window for variance computation
- `HYSTERESIS = 3` — consecutive anomalous samples to flip state
- `ADAPT_RATE = 0.015` — EMA rate for baseline drift during clear periods

### Key constants (classifier.py)
- `BREATHING_BAND = (0.05, 0.30)` Hz — "present but still" signature
- `MOTION_BAND = (0.30, 0.50)` Hz — faster changes (walking aliases into this at 1 Hz)
- `MIN_SAMPLES = 20` — seconds of post-calibration data before FFT is reliable
- `EMPTY_FACTOR = 0.4` — mean score < threshold × this → classified as empty

### Key constants (fingerprinter.py)
- `K = 3` — KNN neighbours
- `MIN_CONFIDENCE = 0.45` — below this returns `("Unknown", conf)`
- `RECORD_SECONDS = 15` — recording duration per session

---

## Known issues / limitations
- 1 Hz sampling rate (netsh is slow) → Nyquist limit 0.5 Hz, walking frequency aliases
- Detection quality is moderate (~70% room accuracy) — fundamentally limited by RSSI data
- `netsh` output can contain non-ASCII SSIDs; handled with `errors='replace'` in scanner.py
- Rooms panel chips need `fp.sessions` attribute (new format); old `rooms.json` auto-migrates

---

## Next step: ESP32 CSI upgrade

The user has ordered **2× ESP32-WROOM-32 DevKit V1** boards from AliExpress (₪14.90 each, ships to Israel, ~3 weeks delivery). These are the **original ESP32 dual-core** chip (confirmed: Bluetooth 4.2, 802.11 B/G/N, has DAC) — correct for CSI support.

### What needs to be built when boards arrive

**1. ESP32 firmware** (`esp32_csi/esp32_csi.ino`)
- Arduino sketch for ESP32-WROOM-32
- Connects to home Wi-Fi
- Enables CSI via `esp_wifi_set_csi_config()` + `esp_wifi_set_csi_rx_cb()`
- Streams CSI packets over UDP to PC (port 5001)
- Rate-limited to ~20 Hz

**Packet format (binary):**
```
Bytes 0–7:   device_id (char[8], null-padded, e.g. "ESP32-0\0")
Byte  8:     rssi (int8)
Byte  9:     channel (uint8)
Bytes 10–11: csi_len (uint16 LE) — number of int8 values following
Bytes 12+:   CSI buffer (int8 pairs: imag0, real0, imag1, real1, ...)
```

**2. `scanner_csi.py`** — drop-in replacement for `scanner.py`
- Listens on UDP port 5001
- Parses binary packets above
- Computes per-subcarrier amplitude: `sqrt(imag² + real²)`
- Returns two channels per device:
  - `"ESP32-0"` → RSSI (int8, familiar scale)
  - `"ESP32-0 (CSI)"` → spatial variance of 52 subcarrier amplitudes (more motion-sensitive)
- Same return type as `scanner.py`: `dict[str, float]`
- Background thread receives UDP; `scan_networks()` reads latest with staleness check (5 s)

**3. To switch:** replace `from scanner import scan_networks` with `from scanner_csi import scan_networks` in `main.py` and `server.py`. Recalibrate after switching (baselines will be different scale).

### Arduino libraries needed
- ESP32 board package for Arduino IDE: `https://dl.espressif.com/dl/package_esp32_index.json`
- No extra libraries — uses built-in `esp_wifi.h` and `WiFiUdp.h`

### Flash settings for Arduino IDE
- Board: "ESP32 Dev Module"
- Upload Speed: 921600
- CPU Frequency: 240MHz
- Flash Size: 4MB

---

## User context
- University student, Windows 11, Python 3.14
- Working directory: `J:\Ido\claude\wifi-sensor\`
- Has 6 Wi-Fi networks visible: Snunit74, BeSpot479C_5.0, BeSpot479C_2.4 + 3 others
- Strongest network: BeSpot479C_2.4 at ~-61 dBm
- Preferred style: concise responses, no emoji, dark Catppuccin theme in UI

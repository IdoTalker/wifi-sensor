# Wi-Fi Presence Detector

Detects whether someone is home — and where — using only Wi-Fi RSSI signal fluctuations. No special hardware required; runs on any Windows machine with a Wi-Fi adapter.

## How it works

Every second, `netsh` polls all nearby Wi-Fi networks. Each access point gets its own anomaly detector that builds a baseline of the "quiet" RSSI and flags deviations. The fused anomaly score is passed through an FFT classifier that separates breathing-rhythm signals (~0.1–0.3 Hz) from motion signals (~0.3–0.5 Hz), producing a 3-state output: **empty / present / moving**. A KNN fingerprinter maps the current RSSI vector to a named room.

```
netsh (1 Hz)  →  MotionDetector × N  →  fused score
                                               ↓
                                     FFT classifier (empty / present / moving)
                                               ↓
                                     KNN fingerprinter  →  room name
                                               ↓
                                     events.csv  +  web dashboard
```

## Features

- 3-state activity detection: empty, someone present, someone moving
- Per-room fingerprinting via KNN — record 15 s of scans per room to train
- Live web dashboard (Flask) accessible from any device on the LAN
- House map editor — draw rooms on a canvas, place your router, rooms colour-code live by signal quality
- Network focus mode — click any network to restrict detection to a single AP
- Adjustable sensitivity slider with live recalibration
- Anomaly score chart (5-minute rolling window)
- Background log scanner — pattern rules + Claude Haiku AI analysis, fires every 100 KB of new log content
- Tkinter desktop GUI as an alternative to the web dashboard

## Requirements

- Windows 10/11 with Wi-Fi adapter
- Python 3.10+
- 2+ visible Wi-Fi access points (more = better accuracy)

```
pip install -r requirements.txt
```

Optional: add an `ANTHROPIC_API_KEY` to `.env` to enable AI log analysis.

## Running

```bash
# Web dashboard — open http://localhost:5000 or http://<LAN-IP>:5000
python server.py

# Desktop GUI (Tkinter)
python main.py

# Tests
python -m unittest discover tests
```

## Project structure

```
wifi-sensor/
├── wifi_sensor/          # core library package
│   ├── scanner.py        # netsh polling → {ssid: rssi_dbm}
│   ├── detector.py       # per-AP anomaly detection (EMA + Z-score)
│   ├── classifier.py     # FFT 3-state classifier
│   ├── fingerprinter.py  # KNN room fingerprinting
│   ├── eventlog.py       # state-change event log (events.csv)
│   └── log_scanner.py    # background log analysis
├── main.py               # Tkinter desktop GUI
├── server.py             # Flask web dashboard + API
├── tests/                # 67 unit tests
└── requirements.txt
```

## Detection accuracy

- Room-level accuracy: ~70% with 3+ access points visible
- Fundamentally limited by 1 Hz netsh sampling (Nyquist = 0.5 Hz)
- Planned upgrade: ESP32 CSI for 20 Hz sub-carrier amplitude data

## License

MIT

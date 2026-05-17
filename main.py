"""Tkinter desktop GUI for the Wi-Fi presence detector.

Runs a background scan thread (1 Hz) and refreshes the UI every 500 ms.
Displays per-network anomaly scores, a live score chart, FFT band indicators,
room-fingerprint chips, a 3-state status bar, and a recent-event log.
"""

import threading
import time
import tkinter as tk
from tkinter import ttk, simpledialog
from collections import deque

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from scanner import scan_networks
from detector import MotionDetector, CALIBRATION_SAMPLES, WINDOW_SIZE
from fingerprinter import Fingerprinter, RECORD_SECONDS
from classifier import classify as fft_classify, MIN_SAMPLES as FFT_MIN_SAMPLES
from eventlog import log_event, load_recent

POLL_INTERVAL = 1.0
SCORE_HISTORY = 120  # 2-minute buffer gives FFT resolution of ~0.008 Hz

BG     = "#1e1e2e"
BG2    = "#181825"
PANEL  = "#313244"
FG     = "#cdd6f4"
ACCENT = "#89b4fa"
GREEN  = "#a6e3a1"
RED    = "#f38ba8"
ORANGE = "#fab387"
PURPLE = "#cba6f7"
MUTED  = "#585b70"


class App(tk.Tk):
    """Main application window.

    Owns a background daemon thread (_scan_loop) that polls Wi-Fi every second
    and updates shared state under _lock.  The main thread drives the UI at
    500 ms via self.after() calls and never touches shared state without
    acquiring _lock first.
    """

    def __init__(self):
        super().__init__()
        self.title("Wi-Fi Presence Detector")
        self.configure(bg=BG)
        self.resizable(True, False)

        self._threshold_var = tk.DoubleVar(value=2.0)
        self._running = True
        self._lock = threading.Lock()

        # Per-network motion detection
        self._detectors: dict[str, MotionDetector] = {}
        self._networks: dict[str, float] = {}
        self._net_scores: dict[str, float] = {}
        self._net_motion: dict[str, bool] = {}
        self._score_history: deque[float] = deque(maxlen=SCORE_HISTORY)

        # Room fingerprinting
        self._fingerprinter = Fingerprinter()
        self._recording = False
        self._record_name = ""
        self._record_buf: list[dict[str, float]] = []
        self._record_remaining = 0
        self._rooms_dirty = True  # triggers panel rebuild on next UI tick
        self._location: tuple[str, float] | None = None  # (room, confidence)

        # Event log + state debounce
        self._prev_state = "unknown"
        self._pending_state = "unknown"
        self._pending_count = 0
        self._recent_events: list[dict] = load_recent(8)
        self._events_dirty = True

        self._build_ui()
        self._refresh_networks()

        self._worker = threading.Thread(target=self._scan_loop, daemon=True)
        self._worker.start()
        self.after(200, self._update_ui)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        PAD = 10

        # ── top bar ──
        top = tk.Frame(self, bg=BG, pady=PAD)
        top.pack(fill="x", padx=PAD)

        tk.Button(top, text="Refresh Networks", command=self._refresh_networks,
                  bg=PANEL, fg=FG, relief="flat", padx=8).pack(side="left", padx=(0, 6))
        tk.Button(top, text="Recalibrate All", command=self._recalibrate_all,
                  bg=PANEL, fg=FG, relief="flat", padx=8).pack(side="left", padx=(0, 20))

        tk.Label(top, text="Sensitivity:", bg=BG, fg=FG, font=("Segoe UI", 9)).pack(side="left")
        tk.Label(top, text="High", bg=BG, fg=FG, font=("Segoe UI", 8)).pack(side="left", padx=(4, 0))
        ttk.Scale(top, from_=0.5, to=5.0, orient="horizontal", length=140,
                  variable=self._threshold_var,
                  command=self._on_threshold_change).pack(side="left", padx=2)
        tk.Label(top, text="Low", bg=BG, fg=FG, font=("Segoe UI", 8)).pack(side="left")
        self._thresh_label = tk.Label(top, text="2.0", bg=BG, fg=ACCENT,
                                      font=("Segoe UI", 9), width=4)
        self._thresh_label.pack(side="left", padx=(4, 0))

        # ── network status panel ──
        self._net_frame = tk.Frame(self, bg=PANEL, pady=6)
        self._net_frame.pack(fill="x", padx=PAD, pady=(0, 4))
        self._net_rows: dict[str, dict] = {}

        # ── rooms panel ──
        rooms_outer = tk.Frame(self, bg=PANEL, pady=6)
        rooms_outer.pack(fill="x", padx=PAD, pady=(0, PAD))

        rooms_header = tk.Frame(rooms_outer, bg=PANEL)
        rooms_header.pack(fill="x", padx=8)
        tk.Label(rooms_header, text="Rooms", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 8, "bold")).pack(side="left")
        self._record_btn = tk.Button(rooms_header, text="+ Record Room",
                                     command=self._start_recording,
                                     bg=PANEL, fg=ACCENT, relief="flat", padx=6,
                                     font=("Segoe UI", 8))
        self._record_btn.pack(side="right")

        self._chips_frame = tk.Frame(rooms_outer, bg=PANEL)
        self._chips_frame.pack(fill="x", padx=8, pady=(4, 0))

        # ── chart ──
        self._fig = Figure(figsize=(7, 2.6), dpi=100, facecolor=BG)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor(BG2)
        self._ax.tick_params(colors=FG, labelsize=8)
        for spine in self._ax.spines.values():
            spine.set_edgecolor("#45475a")
        self._ax.set_xlabel("seconds ago", color=FG, fontsize=8)
        self._ax.set_ylabel("anomaly score", color=FG, fontsize=8)
        self._score_line, = self._ax.plot([], [], color=ACCENT, linewidth=1.5)
        self._thresh_line = self._ax.axhline(y=self._threshold_var.get(),
                                              color=RED, linestyle="--", linewidth=1, alpha=0.7)
        self._fig.tight_layout(pad=1.5)

        canvas = FigureCanvasTkAgg(self._fig, master=self)
        canvas.get_tk_widget().pack(fill="both", padx=PAD)
        self._canvas = canvas

        # ── frequency band strip ──
        freq_frame = tk.Frame(self, bg=BG, pady=3)
        freq_frame.pack(fill="x", padx=PAD)

        tk.Label(freq_frame, text="Breathing (0.05–0.3 Hz):", bg=BG, fg=MUTED,
                 font=("Segoe UI", 8), anchor="w", width=22).pack(side="left")
        self._breath_bar_lbl = tk.Label(freq_frame, text="—", bg=BG, fg=GREEN,
                                        font=("Segoe UI", 8), anchor="w", width=18)
        self._breath_bar_lbl.pack(side="left")

        tk.Label(freq_frame, text="  Motion (0.3–0.5 Hz):", bg=BG, fg=MUTED,
                 font=("Segoe UI", 8), anchor="w", width=20).pack(side="left")
        self._motion_bar_lbl = tk.Label(freq_frame, text="—", bg=BG, fg=RED,
                                        font=("Segoe UI", 8), anchor="w", width=18)
        self._motion_bar_lbl.pack(side="left")

        # ── location label ──
        self._location_var = tk.StringVar(value="")
        self._location_label = tk.Label(self, textvariable=self._location_var,
                                        font=("Segoe UI", 10), bg=BG, fg=PURPLE, pady=2)
        self._location_label.pack(fill="x", padx=PAD)

        # ── status bar ──
        self._status_var = tk.StringVar(value="Refreshing networks…")
        self._status_label = tk.Label(self, textvariable=self._status_var,
                                      font=("Segoe UI", 16, "bold"), bg=PANEL, fg=FG, pady=14)
        self._status_label.pack(fill="x", padx=PAD, pady=(2, 4))

        # ── event log panel ──
        log_outer = tk.Frame(self, bg=PANEL, pady=4)
        log_outer.pack(fill="x", padx=PAD, pady=(0, PAD))
        tk.Label(log_outer, text="Recent Events", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=8)
        self._event_rows: list[tk.Label] = []
        for _ in range(8):
            lbl = tk.Label(log_outer, text="", bg=PANEL, fg=MUTED,
                           font=("Segoe UI", 8), anchor="w")
            lbl.pack(fill="x", padx=12)
            self._event_rows.append(lbl)

    def _ensure_net_rows(self, ssids: list[str]):
        """Add or remove per-network rows in the network panel to match ssids."""
        existing = set(self._net_rows.keys())
        current = set(ssids)
        for ssid in existing - current:
            for w in self._net_rows[ssid].values():
                w.destroy()
            del self._net_rows[ssid]
        for ssid in current - existing:
            row = tk.Frame(self._net_frame, bg=PANEL)
            row.pack(fill="x", padx=8, pady=1)
            dot = tk.Label(row, text="●", bg=PANEL, fg=MUTED, font=("Segoe UI", 10))
            dot.pack(side="left")
            tk.Label(row, text=ssid, bg=PANEL, fg=FG,
                     font=("Segoe UI", 9), width=28, anchor="w").pack(side="left", padx=(4, 0))
            score_lbl = tk.Label(row, text="score: —", bg=PANEL, fg=MUTED,
                                 font=("Segoe UI", 9), width=12, anchor="w")
            score_lbl.pack(side="left", padx=(4, 0))
            status_lbl = tk.Label(row, text="calibrating", bg=PANEL, fg=ORANGE,
                                  font=("Segoe UI", 9, "bold"), width=14, anchor="w")
            status_lbl.pack(side="left")
            self._net_rows[ssid] = {"dot": dot, "score": score_lbl, "status": status_lbl}

    def _rebuild_chips(self, rooms: dict):
        """Rebuild the room chips strip from the current Fingerprinter.rooms dict."""
        for widget in self._chips_frame.winfo_children():
            widget.destroy()
        if not rooms:
            tk.Label(self._chips_frame, text="No rooms recorded yet.",
                     bg=PANEL, fg=MUTED, font=("Segoe UI", 8)).pack(side="left")
            return
        for name, fp in sorted(rooms.items()):
            sessions = fp.sessions
            chip = tk.Frame(self._chips_frame, bg="#45475a", padx=4, pady=2)
            chip.pack(side="left", padx=(0, 6))
            label = f"{name} ({sessions})" if sessions > 1 else name
            color = FG if sessions >= 2 else ORANGE  # orange hint → record again
            tk.Label(chip, text=label, bg="#45475a", fg=color,
                     font=("Segoe UI", 9)).pack(side="left")
            tk.Button(chip, text="×", bg="#45475a", fg=MUTED, relief="flat",
                      font=("Segoe UI", 9), padx=2,
                      command=lambda n=name: self._delete_room(n)).pack(side="left")

    # ── actions ──────────────────────────────────────────────────────────────

    def _on_threshold_change(self, _=None):
        val = round(self._threshold_var.get(), 1)
        self._thresh_label.configure(text=f"{val:.1f}")
        self._thresh_line.set_ydata([val, val])
        with self._lock:
            for det in self._detectors.values():
                det.threshold = val

    def _refresh_networks(self):
        nets = scan_networks()
        with self._lock:
            self._networks = nets
            for ssid in nets:
                if ssid not in self._detectors:
                    self._detectors[ssid] = MotionDetector(threshold=self._threshold_var.get())
        self._ensure_net_rows(sorted(nets.keys()))

    def _recalibrate_all(self):
        with self._lock:
            for det in self._detectors.values():
                det.reset()
            self._score_history.clear()
        self._status_var.set("Calibrating…")
        self._status_label.configure(bg=PANEL, fg=FG)

    def _start_recording(self):
        with self._lock:
            if self._recording:
                return
        name = simpledialog.askstring("Record Room", "Enter a name for this location:",
                                      parent=self)
        if not name or not name.strip():
            return
        name = name.strip()
        with self._lock:
            self._recording = True
            self._record_name = name
            self._record_buf = []
            self._record_remaining = RECORD_SECONDS
        self._record_btn.configure(state="disabled")

    def _delete_room(self, name: str):
        self._fingerprinter.delete(name)
        self._rooms_dirty = True

    def _on_close(self):
        self._running = False
        self.destroy()

    # ── background scan loop ─────────────────────────────────────────────────

    def _scan_loop(self):
        """Background thread: scan networks, update detectors, handle recording/classification."""
        while self._running:
            start = time.monotonic()
            nets = scan_networks()

            with self._lock:
                self._networks = nets
                for ssid in nets:
                    if ssid not in self._detectors:
                        self._detectors[ssid] = MotionDetector(threshold=self._threshold_var.get())

                # Update motion detectors
                scores: list[float] = []
                for ssid, rssi in nets.items():
                    det = self._detectors[ssid]
                    det.update(rssi)
                    self._net_scores[ssid] = det.score()
                    self._net_motion[ssid] = det.is_motion()
                    if det.calibrated:
                        scores.append(det.score())
                if scores:
                    self._score_history.append(max(scores))

                # Recording or classifying
                if self._recording:
                    self._record_buf.append(dict(nets))
                    self._record_remaining -= 1
                    if self._record_remaining <= 0:
                        self._fingerprinter.record(self._record_name, self._record_buf)
                        self._recording = False
                        self._rooms_dirty = True
                else:
                    self._location = self._fingerprinter.classify(nets)

            elapsed = time.monotonic() - start
            time.sleep(max(0.0, POLL_INTERVAL - elapsed))

    # ── UI refresh (main thread) ──────────────────────────────────────────────

    def _update_ui(self):
        """UI refresh callback (main thread, 500 ms cadence).

        Snapshots shared state under _lock, then updates all widgets.
        Re-schedules itself via self.after() so it never blocks the event loop.
        """
        if not self._running:
            return

        with self._lock:
            detectors = dict(self._detectors)
            net_scores = dict(self._net_scores)
            net_motion = dict(self._net_motion)
            score_history = list(self._score_history)
            recording = self._recording
            record_remaining = self._record_remaining
            rooms_dirty = self._rooms_dirty
            location = self._location
            if rooms_dirty:
                self._rooms_dirty = False

        # ── rooms panel ──
        if rooms_dirty:
            self._rebuild_chips(self._fingerprinter.rooms)

        if recording:
            self._record_btn.configure(
                text=f"Recording… {record_remaining}s", state="disabled", fg=ORANGE)
        else:
            self._record_btn.configure(text="+ Record Room", state="normal", fg=ACCENT)

        # ── location display ──
        if location:
            room, conf = location
            pct = int(conf * 100)
            bar_filled = int(conf * 10)
            bar = "█" * bar_filled + "░" * (10 - bar_filled)
            if room == "Unknown":
                self._location_var.set(f"Location: Unknown  {bar}  {pct}%  (low confidence)")
                self._location_label.configure(fg=MUTED)
            else:
                self._location_var.set(f"Location: {room}  {bar}  {pct}%")
                self._location_label.configure(fg=PURPLE)
        elif self._fingerprinter.rooms:
            self._location_var.set("Location: scanning…")
            self._location_label.configure(fg=MUTED)
        else:
            self._location_var.set("")

        # ── FFT classification ──
        threshold = self._threshold_var.get()
        act_state, b_frac, m_frac = fft_classify(score_history, threshold)

        def _bar(frac: float, width: int = 12) -> str:
            filled = round(frac * width)
            return "█" * filled + "░" * (width - filled) + f"  {int(frac * 100)}%"

        if act_state == "unknown":
            self._breath_bar_lbl.configure(text="—")
            self._motion_bar_lbl.configure(text="—")
        else:
            self._breath_bar_lbl.configure(text=_bar(b_frac))
            self._motion_bar_lbl.configure(text=_bar(m_frac))

        # ── debounce + event log ──
        DEBOUNCE_TICKS = 6  # 6 × 500 ms = 3 s stable before committing
        if act_state == self._pending_state:
            self._pending_count += 1
        else:
            self._pending_state = act_state
            self._pending_count = 1

        if self._pending_count == DEBOUNCE_TICKS and act_state != self._prev_state:
            room_name = location[0] if location and location[0] != "Unknown" else None
            conf = location[1] if location else 0.0
            log_event(act_state, room_name, conf)
            self._prev_state = act_state
            self._recent_events = load_recent(8)
            self._events_dirty = True

        # ── event log panel ──
        if self._events_dirty:
            self._events_dirty = False
            STATE_COLORS = {"empty": GREEN, "present": "#f9e2af", "moving": RED}
            for i, lbl in enumerate(self._event_rows):
                if i < len(self._recent_events):
                    ev = self._recent_events[i]
                    ts = ev.get("timestamp", "")[-8:]   # HH:MM:SS
                    st = ev.get("state", "")
                    rm = ev.get("room", "")
                    room_part = f"  —  {rm}" if rm else ""
                    conf_part = f"  ({float(ev.get('confidence', 0)):.0%})" if ev.get("confidence") else ""
                    color = STATE_COLORS.get(st, MUTED)
                    lbl.configure(text=f"{ts}  {st}{room_part}{conf_part}", fg=color)
                else:
                    lbl.configure(text="")

        # ── network rows ──
        all_calibrated = bool(detectors) and all(d.calibrated for d in detectors.values())

        for ssid, widgets in self._net_rows.items():
            det = detectors.get(ssid)
            if det is None:
                continue
            score = net_scores.get(ssid, 0.0)
            motion = net_motion.get(ssid, False)
            if not det.calibrated:
                prog = det.calibration_progress
                widgets["dot"].configure(fg=MUTED)
                widgets["score"].configure(text=f"cal {prog}/{CALIBRATION_SAMPLES}", fg=MUTED)
                widgets["status"].configure(text="calibrating", fg=ORANGE)
            elif motion:
                widgets["dot"].configure(fg=RED)
                widgets["score"].configure(text=f"score: {score:.2f}", fg=FG)
                widgets["status"].configure(text="MOTION", fg=RED)
            else:
                widgets["dot"].configure(fg=GREEN)
                widgets["score"].configure(text=f"score: {score:.2f}", fg=FG)
                widgets["status"].configure(text="clear", fg=GREEN)

        # ── status bar (3-state classifier) ──
        AMBER = "#f9e2af"
        if not detectors:
            self._status_var.set("No networks found — click Refresh")
            self._status_label.configure(bg=PANEL, fg=FG)
        elif not all_calibrated:
            remaining = max(
                CALIBRATION_SAMPLES - d.calibration_progress
                for d in detectors.values() if not d.calibrated
            )
            self._status_var.set(f"Calibrating… ({remaining}s, stay still)")
            self._status_label.configure(bg=PANEL, fg=FG)
        elif act_state == "unknown":
            need = FFT_MIN_SAMPLES - len(score_history)
            self._status_var.set(f"Analyzing… ({need}s more)")
            self._status_label.configure(bg=PANEL, fg=FG)
        elif act_state == "empty":
            self._status_var.set("Empty")
            self._status_label.configure(bg=GREEN, fg="#1e1e2e")
        elif act_state == "present":
            self._status_var.set("Someone Present")
            self._status_label.configure(bg=AMBER, fg="#1e1e2e")
        else:  # moving
            count = sum(1 for m in net_motion.values() if m)
            total = len(net_motion)
            suffix = f"  ({count}/{total} networks)" if total else ""
            self._status_var.set(f"Someone Moving{suffix}")
            self._status_label.configure(bg=RED, fg="#1e1e2e")

        # ── chart ──
        if score_history:
            n = len(score_history)
            xs = list(range(-n + 1, 1))
            self._score_line.set_data(xs, score_history)
            self._ax.set_xlim(-(SCORE_HISTORY - 1), 0)
            ymax = max(max(score_history) * 1.2, self._threshold_var.get() * 1.5, 4.0)
            self._ax.set_ylim(0, ymax)
            self._thresh_line.set_ydata([self._threshold_var.get()] * 2)
            self._canvas.draw_idle()

        self.after(500, self._update_ui)


if __name__ == "__main__":
    app = App()
    app.mainloop()

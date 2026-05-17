"""Log scanner: pattern-matching + Claude AI analysis of wifi_sensor.log.

Called by the log-watcher background thread in server.py whenever the log
grows by LOG_SCAN_THRESHOLD_BYTES since the previous scan.  Results are
appended to scan_suggestions.json (capped at 20 entries).
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

LOG_SCAN_THRESHOLD_BYTES = 100 * 1024   # 100 KB growth triggers a scan
SUGGESTIONS_PATH = Path(__file__).parent.parent / "scan_suggestions.json"
_MAX_AI_LINES = 300   # cap sent to Claude to limit token use

# ── pattern rules ─────────────────────────────────────────────────────────────
# Each entry: (compiled regex, severity, type, suggestion text)

_PATTERNS: list[tuple[re.Pattern, str, str, str]] = [
    (
        re.compile(r"stuck-motion recal triggered", re.I),
        "high", "fix",
        "Stuck-motion recalibration fired: the detector stayed in MOTION for 5 minutes straight "
        "then force-reset. This usually means persistent RSSI jitter (microwave, rogue device). "
        "Consider lowering EMA_ALPHA in detector.py for heavier smoothing, or raising "
        "STUCK_MOTION_TICKS if the environment is genuinely active that long.",
    ),
    (
        re.compile(r"netsh timed out|timed out.*returning empty scan", re.I),
        "medium", "fix",
        "netsh scan timed out and returned empty results. Repeated timeouts degrade detection "
        "quality. Try increasing the subprocess timeout in scanner.py, or check whether a "
        "background Windows process (Update, driver reconnect) is monopolising the adapter.",
    ),
    (
        re.compile(r"netsh not found", re.I),
        "high", "fix",
        "'netsh' was not found — the Wi-Fi scanner cannot run. "
        "Ensure you are running on a Windows host with the Wi-Fi adapter enabled and that "
        "System32 is on PATH.",
    ),
    (
        re.compile(r"unhandled error in scan loop", re.I),
        "high", "fix",
        "The main scan loop crashed and self-recovered. Repeated crashes mean missed detections. "
        "Check the traceback immediately following this line in the log for the root cause.",
    ),
    (
        re.compile(r"pruning stale uncalibrated detector", re.I),
        "low", "upgrade",
        "Uncalibrated detectors are being pruned. Normal for transient networks, but if it "
        "happens for your target AP, consider raising STALE_ABSENT_TICKS in server.py so the "
        "detector survives brief scan gaps during calibration.",
    ),
    (
        re.compile(r"\[ERROR\]|\[CRITICAL\]|Traceback \(most recent call last\)", re.I),
        "high", "fix",
        "ERROR or CRITICAL entries detected. These indicate unexpected exceptions that may "
        "affect detection reliability — review the full traceback in the log file.",
    ),
    (
        re.compile(r"manual recalibrate triggered", re.I),
        "low", "upgrade",
        "Manual recalibration was triggered. If you find yourself recalibrating often, consider "
        "adding a scheduled nightly recalibration (e.g. via the /api/recalibrate endpoint from "
        "a cron job) to keep the baseline fresh.",
    ),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _interesting_lines(text: str) -> str:
    """Return up to _MAX_AI_LINES lines: all WARNING/ERROR lines + recent tail."""
    lines = text.splitlines()
    important = [l for l in lines if re.search(r"\[(WARNING|ERROR|CRITICAL)\]", l, re.I)]
    tail = lines[-100:]
    combined = important[-((_MAX_AI_LINES // 2)):] + tail
    seen: set[str] = set()
    unique: list[str] = []
    for l in combined:
        if l not in seen:
            seen.add(l)
            unique.append(l)
    return "\n".join(unique[-_MAX_AI_LINES:])


def _run_patterns(text: str) -> list[dict]:
    results: list[dict] = []
    for pat, severity, kind, message in _PATTERNS:
        matches = pat.findall(text)
        if not matches:
            continue
        n = len(matches)
        results.append({
            "type": kind,
            "source": "pattern",
            "severity": severity,
            "count": n,
            "text": f"[×{n}] {message}" if n > 1 else message,
        })
    return results


def _run_claude(excerpt: str) -> list[dict]:
    """Send the log excerpt to Claude and return a list of suggestion dicts."""
    try:
        import anthropic
    except ImportError:
        logger.debug("anthropic package not installed — skipping AI log analysis")
        return []

    try:
        client = anthropic.Anthropic()
        prompt = (
            "You are a diagnostic assistant for a Wi-Fi-based presence detection system.\n"
            "The system scans nearby Wi-Fi networks at 1 Hz and uses per-BSSID RSSI anomaly\n"
            "detection (Z-score + rolling variance + EMA smoothing) to determine if someone\n"
            "is home. It runs on Windows via netsh.\n\n"
            "Analyse the log excerpt below and return a JSON array of suggestion objects.\n"
            "Each object must have exactly these fields:\n"
            '  "type":     "fix" | "upgrade"\n'
            '  "severity": "high" | "medium" | "low"\n'
            '  "text":     a concise, actionable suggestion (1-3 sentences)\n\n'
            "Rules:\n"
            "- Focus only on real problems visible in the log.\n"
            "- If the log looks healthy, return [].\n"
            "- Do NOT include markdown or code fences — return raw JSON only.\n\n"
            "--- LOG EXCERPT ---\n"
            f"{excerpt}\n"
            "--- END ---"
        )
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.M)
        raw = re.sub(r"\s*```$", "", raw, flags=re.M)
        items: list[dict] = json.loads(raw)
        for item in items:
            item["source"] = "claude"
        logger.info("log_scanner: Claude returned %d suggestion(s)", len(items))
        return items
    except Exception:
        logger.warning("log_scanner: Claude API call failed", exc_info=True)
        return []


# ── public API ────────────────────────────────────────────────────────────────

def scan(log_path: Path, log_size_bytes: int) -> dict:
    """Run a full scan of *log_path* and append the result to SUGGESTIONS_PATH.

    Returns the scan record dict (empty dict on read failure).
    """
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        logger.warning("log_scanner: could not read %s", log_path)
        return {}

    pattern_hits = _run_patterns(text)
    ai_hits = _run_claude(_interesting_lines(text))

    record: dict = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "log_size_kb": round(log_size_bytes / 1024, 1),
        "pattern_hits": len(pattern_hits),
        "ai_hits": len(ai_hits),
        "suggestions": pattern_hits + ai_hits,
    }

    existing: list[dict] = []
    if SUGGESTIONS_PATH.exists():
        try:
            existing = json.loads(SUGGESTIONS_PATH.read_text(encoding="utf-8"))
        except Exception:
            existing = []
    existing.append(record)
    SUGGESTIONS_PATH.write_text(
        json.dumps(existing[-20:], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info(
        "log_scanner: scanned %.1f KB — %d pattern hits, %d AI suggestions",
        log_size_bytes / 1024, len(pattern_hits), len(ai_hits),
    )
    return record

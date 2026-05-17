import threading
import time
from collections import deque

from flask import Flask, jsonify, request

from scanner import scan_networks
from detector import MotionDetector, CALIBRATION_SAMPLES
from fingerprinter import Fingerprinter, RECORD_SECONDS
from classifier import classify as fft_classify, MIN_SAMPLES as FFT_MIN_SAMPLES
from eventlog import log_event, load_recent

POLL_INTERVAL = 1.0
SCORE_HISTORY  = 120
DEBOUNCE_TICKS = 3   # seconds at 1 Hz

# ── shared state ──────────────────────────────────────────────────────────────

_lock = threading.Lock()

_detectors:    dict[str, MotionDetector] = {}
_current_nets: dict[str, float]          = {}
_net_scores:   dict[str, float]          = {}
_net_motion:   dict[str, bool]           = {}
_score_history: deque[float]             = deque(maxlen=SCORE_HISTORY)

_fingerprinter = Fingerprinter()
_location: tuple[str, float] | None = None

_recording        = False
_record_name      = ""
_record_buf:  list[dict[str, float]] = []
_record_remaining = 0

_act_state    = "unknown"
_b_frac       = 0.0
_m_frac       = 0.0
_threshold    = 2.0

_prev_state    = "unknown"
_pending_state = "unknown"
_pending_count = 0

# ── background scan loop ──────────────────────────────────────────────────────

def _scan_loop():
    global _location, _recording, _record_name, _record_buf, _record_remaining
    global _act_state, _b_frac, _m_frac
    global _prev_state, _pending_state, _pending_count, _current_nets

    while True:
        start = time.monotonic()
        nets  = scan_networks()

        with _lock:
            _current_nets = nets

            for ssid in nets:
                if ssid not in _detectors:
                    _detectors[ssid] = MotionDetector(threshold=_threshold)

            scores: list[float] = []
            for ssid, rssi in nets.items():
                det = _detectors[ssid]
                det.update(rssi)
                _net_scores[ssid] = det.score()
                _net_motion[ssid] = det.is_motion()
                if det.calibrated:
                    scores.append(det.score())
            if scores:
                _score_history.append(max(scores))

            if _recording:
                _record_buf.append(dict(nets))
                _record_remaining -= 1
                if _record_remaining <= 0:
                    _fingerprinter.record(_record_name, _record_buf)
                    _recording = False
            else:
                _location = _fingerprinter.classify(nets)

            # Determine activity state
            history = list(_score_history)
            all_cal = bool(_detectors) and all(d.calibrated for d in _detectors.values())

            if not _detectors:
                _act_state = "unknown"
            elif not all_cal:
                _act_state = "calibrating"
            elif len(history) < FFT_MIN_SAMPLES:
                _act_state = "analyzing"
            else:
                _act_state, _b_frac, _m_frac = fft_classify(history, _threshold)

            # Debounce + event log
            if _act_state == _pending_state:
                _pending_count += 1
            else:
                _pending_state = _act_state
                _pending_count = 1

            if _pending_count == DEBOUNCE_TICKS and _act_state != _prev_state:
                room_name = _location[0] if _location and _location[0] != "Unknown" else None
                conf      = _location[1] if _location else 0.0
                log_event(_act_state, room_name, conf)
                _prev_state = _act_state

        elapsed = time.monotonic() - start
        time.sleep(max(0.0, POLL_INTERVAL - elapsed))

# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)

@app.route("/")
def index():
    return DASHBOARD_HTML, 200, {"Content-Type": "text/html; charset=utf-8"}

@app.route("/api/status")
def api_status():
    with _lock:
        all_cal = bool(_detectors) and all(d.calibrated for d in _detectors.values())

        # State detail string
        detail = ""
        if _act_state == "calibrating" and _detectors:
            remaining = max(
                CALIBRATION_SAMPLES - d.calibration_progress
                for d in _detectors.values() if not d.calibrated
            )
            detail = f"({remaining}s, stay still)"
        elif _act_state == "analyzing":
            detail = f"({FFT_MIN_SAMPLES - len(_score_history)}s more)"
        elif _act_state == "moving":
            count = sum(1 for m in _net_motion.values() if m)
            total = len(_net_motion)
            if total:
                detail = f"({count}/{total} networks)"

        networks = sorted([
            {
                "ssid":         ssid,
                "rssi":         round(_current_nets.get(ssid, 0.0), 1),
                "score":        round(_net_scores.get(ssid, 0.0), 2),
                "motion":       _net_motion.get(ssid, False),
                "calibrated":   det.calibrated,
                "cal_progress": det.calibration_progress,
            }
            for ssid, det in _detectors.items()
        ], key=lambda x: x["ssid"])

        loc = None
        if _location:
            room, conf = _location
            loc = {"room": room, "confidence": round(conf, 2)}

        rooms = sorted(
            [{"name": n, "sessions": fp.sessions} for n, fp in _fingerprinter.rooms.items()],
            key=lambda x: x["name"],
        )

        return jsonify({
            "state":            _act_state,
            "state_detail":     detail,
            "location":         loc,
            "networks":         networks,
            "bands":            {"breathing": round(_b_frac, 3), "motion": round(_m_frac, 3)},
            "recording":        _recording,
            "record_remaining": _record_remaining,
            "rooms":            rooms,
            "threshold":        _threshold,
            "events":           load_recent(10),
        })

@app.route("/api/record", methods=["POST"])
def api_record():
    global _recording, _record_name, _record_buf, _record_remaining
    name = (request.json or {}).get("name", "").strip()
    if not name:
        return jsonify({"error": "name required"}), 400
    with _lock:
        if _recording:
            return jsonify({"error": "already recording"}), 409
        _recording        = True
        _record_name      = name
        _record_buf       = []
        _record_remaining = RECORD_SECONDS
    return jsonify({"ok": True})

@app.route("/api/rooms/<name>", methods=["DELETE"])
def api_delete_room(name):
    _fingerprinter.delete(name)
    return jsonify({"ok": True})

@app.route("/api/threshold", methods=["POST"])
def api_threshold():
    global _threshold
    val = (request.json or {}).get("value")
    if val is None:
        return jsonify({"error": "value required"}), 400
    _threshold = max(0.5, min(5.0, float(val)))
    with _lock:
        for det in _detectors.values():
            det.threshold = _threshold
    return jsonify({"ok": True, "threshold": _threshold})

@app.route("/api/recalibrate", methods=["POST"])
def api_recalibrate():
    with _lock:
        for det in _detectors.values():
            det.reset()
        _score_history.clear()
    return jsonify({"ok": True})

# ── Dashboard HTML ────────────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Wi-Fi Presence Detector</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#1e1e2e;color:#cdd6f4;font-family:'Segoe UI',system-ui,sans-serif;padding:16px;max-width:900px;margin:0 auto}
header{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px}
header h1{font-size:1.05em;color:#89b4fa;font-weight:600}
.live{display:flex;align-items:center;gap:6px;font-size:.8em;color:#585b70}
.dot-live{width:8px;height:8px;background:#a6e3a1;border-radius:50%;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
#status-box{padding:20px;text-align:center;font-size:1.9em;font-weight:700;border-radius:8px;margin-bottom:12px;background:#313244;transition:background .4s,color .4s}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px}
@media(max-width:560px){.grid2{grid-template-columns:1fr}}
.card{background:#313244;border-radius:8px;padding:12px}
.card h3{font-size:.68em;color:#585b70;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px}
.net-row{display:flex;align-items:center;gap:8px;padding:2px 0;font-size:.84em}
.net-ssid{flex:1;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
.bar-row{display:flex;align-items:center;gap:8px;margin:5px 0;font-size:.82em}
.bar-label{width:80px;color:#585b70}
.bar-track{flex:1;background:#181825;border-radius:4px;height:6px}
.bar-fill{height:6px;border-radius:4px;transition:width .4s}
.bar-pct{width:36px;text-align:right}
.chip{display:inline-flex;align-items:center;gap:4px;background:#45475a;border-radius:4px;padding:2px 8px;margin:2px;font-size:.82em}
.chip-del{background:none;border:none;color:#585b70;cursor:pointer;font-size:.95em;padding:0 2px}
.chip-del:hover{color:#f38ba8}
.record-row{display:flex;gap:8px;margin-top:8px}
.record-row input{flex:1;background:#1e1e2e;border:1px solid #45475a;color:#cdd6f4;padding:4px 8px;border-radius:4px;font-size:.84em;outline:none}
.record-row input:focus{border-color:#89b4fa}
.btn{background:#1e1e2e;border:1px solid #45475a;color:#89b4fa;padding:4px 12px;border-radius:4px;cursor:pointer;font-size:.84em}
.btn:hover{background:#45475a}
.btn:disabled{opacity:.4;cursor:default}
#rec-status{font-size:.78em;color:#fab387;margin-top:4px;min-height:1.2em}
.ev-row{padding:3px 0;font-size:.81em;border-bottom:1px solid #45475a}
.ev-row:last-child{border-bottom:none}
.muted{color:#585b70;font-size:.84em}
.loc-name{font-size:1.15em;margin-bottom:4px}
.loc-bar{font-size:.82em;color:#cba6f7;letter-spacing:-1px}
.sens-row{display:flex;align-items:center;gap:8px;margin-top:8px;font-size:.82em}
.sens-row label{color:#585b70}
.sens-row input[type=range]{flex:1;accent-color:#89b4fa}
.action-row{display:flex;gap:8px;margin-top:8px}
</style>
</head>
<body>
<header>
  <h1>Wi-Fi Presence Detector</h1>
  <div class="live"><div class="dot-live"></div> live</div>
</header>

<div id="status-box">—</div>

<div class="grid2">
  <div class="card">
    <h3>Location</h3>
    <div id="loc-name" class="loc-name muted">—</div>
    <div id="loc-bar" class="loc-bar"></div>
  </div>
  <div class="card">
    <h3>Networks</h3>
    <div id="networks"></div>
  </div>
</div>

<div class="grid2">
  <div class="card">
    <h3>Frequency Bands</h3>
    <div class="bar-row">
      <span class="bar-label">Breathing</span>
      <div class="bar-track"><div class="bar-fill" id="bar-b" style="background:#a6e3a1;width:0"></div></div>
      <span class="bar-pct" id="pct-b" style="color:#a6e3a1">—</span>
    </div>
    <div class="bar-row">
      <span class="bar-label">Motion</span>
      <div class="bar-track"><div class="bar-fill" id="bar-m" style="background:#f38ba8;width:0"></div></div>
      <span class="bar-pct" id="pct-m" style="color:#f38ba8">—</span>
    </div>
    <div class="sens-row">
      <label>Sensitivity</label>
      <span class="muted">High</span>
      <input type="range" id="threshold" min="0.5" max="5" step="0.1" value="2.0" oninput="setThreshold(this.value)">
      <span class="muted">Low</span>
      <span id="thresh-val" style="color:#89b4fa;width:28px">2.0</span>
    </div>
    <div class="action-row">
      <button class="btn" onclick="recalibrate()">Recalibrate</button>
    </div>
  </div>
  <div class="card">
    <h3>Rooms</h3>
    <div id="rooms"></div>
    <div class="record-row">
      <input id="room-input" placeholder="Room name…" onkeydown="if(event.key==='Enter')startRecord()">
      <button class="btn" id="rec-btn" onclick="startRecord()">Record (15s)</button>
    </div>
    <div id="rec-status"></div>
  </div>
</div>

<div class="card">
  <h3>Recent Events</h3>
  <div id="events"></div>
</div>

<script>
const STATE_COLOR = {empty:'#a6e3a1',present:'#f9e2af',moving:'#f38ba8'};
const STATE_LABEL = {
  empty:'Empty', present:'Someone Present', moving:'Someone Moving',
  calibrating:'Calibrating…', analyzing:'Analyzing…', unknown:'—'
};

let thresholdPending = null;

async function refresh(){
  try{
    const d = await (await fetch('/api/status')).json();

    // Status
    const box = document.getElementById('status-box');
    const label = (STATE_LABEL[d.state]||d.state) + (d.state_detail?' '+d.state_detail:'');
    box.textContent = label;
    const col = STATE_COLOR[d.state];
    box.style.background = col||'#313244';
    box.style.color = col?'#1e1e2e':'#cdd6f4';

    // Location
    const locN = document.getElementById('loc-name');
    const locB = document.getElementById('loc-bar');
    if(d.location){
      const pct = Math.round(d.location.confidence*100);
      const f = Math.round(d.location.confidence*10);
      const bar = '█'.repeat(f)+'░'.repeat(10-f);
      locN.textContent = d.location.room;
      locN.style.color = d.location.room==='Unknown'?'#585b70':'#cba6f7';
      locB.textContent = bar+' '+pct+'%';
    } else {
      locN.textContent = d.rooms.length?'scanning…':'—';
      locN.style.color = '#585b70';
      locB.textContent = '';
    }

    // Networks
    document.getElementById('networks').innerHTML = d.networks.map(n=>{
      const dc = n.calibrated?(n.motion?'#f38ba8':'#a6e3a1'):'#585b70';
      const sc = n.calibrated?(n.motion?'#f38ba8':'#a6e3a1'):'#585b70';
      const st = n.calibrated?(n.motion?'MOTION':'clear'):'cal '+n.cal_progress+'/30';
      return `<div class="net-row">
        <span style="color:${dc}">●</span>
        <span class="net-ssid">${n.ssid}</span>
        <span style="color:#585b70">${n.score.toFixed(2)}</span>
        <span style="color:${sc};font-weight:bold;width:80px;text-align:right">${st}</span>
      </div>`;
    }).join('')||'<span class="muted">No networks found.</span>';

    // Bands
    if(d.state!=='unknown'&&d.state!=='calibrating'){
      const b=Math.round(d.bands.breathing*100), m=Math.round(d.bands.motion*100);
      document.getElementById('bar-b').style.width=b+'%';
      document.getElementById('pct-b').textContent=b+'%';
      document.getElementById('bar-m').style.width=m+'%';
      document.getElementById('pct-m').textContent=m+'%';
    }

    // Threshold slider sync (don't override if user is dragging)
    if(thresholdPending===null){
      document.getElementById('threshold').value = d.threshold;
      document.getElementById('thresh-val').textContent = d.threshold.toFixed(1);
    }

    // Rooms
    document.getElementById('rooms').innerHTML = d.rooms.map(r=>{
      const label = r.sessions>1?`${r.name} (${r.sessions})`:r.name;
      const col = r.sessions>=2?'#cdd6f4':'#fab387';
      return `<span class="chip"><span style="color:${col}">${label}</span><button class="chip-del" onclick="deleteRoom('${r.name}')">×</button></span>`;
    }).join('')||'<span class="muted">No rooms recorded yet.</span>';

    // Record button
    const btn = document.getElementById('rec-btn');
    const rs  = document.getElementById('rec-status');
    if(d.recording){
      btn.disabled=true; rs.textContent='Recording… '+d.record_remaining+'s';
    } else {
      btn.disabled=false; rs.textContent='';
    }

    // Events
    const EC={empty:'#a6e3a1',present:'#f9e2af',moving:'#f38ba8'};
    document.getElementById('events').innerHTML = d.events.map(e=>{
      const ts=e.timestamp.slice(11);
      const rm=e.room?' — '+e.room:'';
      const cf=e.confidence?' ('+Math.round(e.confidence*100)+'%)':'';
      return `<div class="ev-row" style="color:${EC[e.state]||'#585b70'}">${ts}  ${e.state}${rm}${cf}</div>`;
    }).join('')||'<span class="muted">No events yet.</span>';

  }catch(e){/* server restarting */}
}

async function startRecord(){
  const name=document.getElementById('room-input').value.trim();
  if(!name)return;
  await fetch('/api/record',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})});
  document.getElementById('room-input').value='';
}

async function deleteRoom(name){
  await fetch('/api/rooms/'+encodeURIComponent(name),{method:'DELETE'});
}

function setThreshold(val){
  document.getElementById('thresh-val').textContent=parseFloat(val).toFixed(1);
  clearTimeout(thresholdPending);
  thresholdPending=setTimeout(async()=>{
    await fetch('/api/threshold',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({value:parseFloat(val)})});
    thresholdPending=null;
  },400);
}

async function recalibrate(){
  await fetch('/api/recalibrate',{method:'POST'});
}

setInterval(refresh,1000);
refresh();
</script>
</body>
</html>"""

# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import socket
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        ip = "your-ip"

    t = threading.Thread(target=_scan_loop, daemon=True)
    t.start()

    print(f"\n  Dashboard → http://localhost:5000")
    print(f"  On network → http://{ip}:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

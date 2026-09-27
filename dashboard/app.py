"""
dashboard/app.py — Flask + SocketIO real-time web dashboard.
Pushes live alerts and stats to the browser every second.
"""

import threading
import time
from collections import Counter
from flask import Flask, render_template, jsonify, request, send_file
from flask_socketio import SocketIO, emit


app     = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*",
                    async_mode="threading")

# Global references — set by main.py before app.run()
_alert_manager = None
_detector_ref  = None
_capture_ref   = None
_flow_tracker_ref = None
_trainer_ref   = None
_start_time    = time.time()


def init_dashboard(alert_manager, detector=None,
                   capture=None, trainer=None, flow_tracker=None):
    """Called by main.py to inject live references."""
    global _alert_manager, _detector_ref, _capture_ref, _flow_tracker_ref
    global _trainer_ref, _start_time
    _alert_manager = alert_manager
    _detector_ref  = detector
    _capture_ref   = capture
    _flow_tracker_ref = flow_tracker
    _trainer_ref   = trainer
    _start_time    = time.time()


# ─── HTTP Routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stats")
def api_stats():
    if _alert_manager is None:
        return jsonify({"error": "not initialised"})
    stats = _dashboard_stats()
    stats["uptime"] = _fmt_uptime(time.time() - _start_time)
    return jsonify(stats)


@app.route("/api/report")
def api_report():
    if _alert_manager is None:
        return jsonify({"error": "No active session data is available yet."}), 404

    session_data = _build_session_data()
    if not session_data["stats"].get("total_packets") and not session_data["stats"].get("total_flows"):
        return jsonify({"error": "No active session data is available yet."}), 404

    from reporting.report_generator import NIDSReportGenerator

    report = NIDSReportGenerator(session_data).generate_bytes()
    return send_file(report, as_attachment=True,
                     download_name="nids_report.pdf",
                     mimetype="application/pdf")


@app.route("/api/alerts")
def api_alerts():
    n = int(request.args.get("n", 50))
    if _alert_manager is None:
        return jsonify([])
    return jsonify(_alert_manager.recent_alerts(n))


@app.route("/api/explain/<alert_id>")
def api_explain(alert_id):
    """Return explanation for a specific alert by ID."""
    if _alert_manager is None:
        return jsonify({})
    for a in _alert_manager.recent_alerts(500):
        if a["id"] == alert_id:
            return jsonify(a.get("explanation", {}))
    return jsonify({"error": "alert not found"}), 404


@app.route("/api/model_info")
def api_model_info():
    if _trainer_ref is None:
        return jsonify({"status": "no model"})
    return jsonify({
        "status":   "loaded",
        "classes":  list(_trainer_ref.label_enc.classes_)
                    if _trainer_ref.label_enc else [],
        "features": 24,
        "models":   ["Random Forest (n=150)", "Isolation Forest (n=150)"],
    })


# ─── SocketIO Events ──────────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    emit("connected", {"msg": "NIDS dashboard connected"})
    _push_full_state()


@socketio.on("request_state")
def on_request_state():
    _push_full_state()


def _push_full_state():
    if _alert_manager is None:
        return
    stats  = _dashboard_stats()
    stats["uptime"] = _fmt_uptime(time.time() - _start_time)
    emit("stats_update",  stats)
    emit("alerts_update", _alert_manager.recent_alerts(50))


# ─── Background Broadcaster ───────────────────────────────────────────────────

def _broadcast_loop():
    """Push live updates to all connected clients every second."""
    while True:
        time.sleep(1)
        if _alert_manager is None:
            continue
        try:
            stats = _dashboard_stats()
            stats["uptime"] = _fmt_uptime(time.time() - _start_time)
            socketio.emit("stats_update",  stats)
            socketio.emit("alerts_update", _alert_manager.recent_alerts(50))
        except Exception:
            pass


def start_broadcaster():
    t = threading.Thread(target=_broadcast_loop, daemon=True)
    t.start()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _fmt_uptime(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _threat_level(stats: dict) -> str:
    if int(stats.get("critical", 0)) > 0:
        return "CRITICAL"
    if int(stats.get("high", 0)) > 0:
        return "HIGH"
    if int(stats.get("medium", 0)) > 0:
        return "MEDIUM"
    return "SECURE"


def _dashboard_stats() -> dict:
    stats = dict(_alert_manager.dashboard_stats())
    alerts = _alert_manager.recent_alerts(500)

    capture_stats = getattr(_capture_ref, "stats", {}) or {}
    stats["total_packets"] = int(capture_stats.get("total", 0))
    stats["active_flows"] = (
        _flow_tracker_ref.active_count() if _flow_tracker_ref is not None else 0
    )
    stats["completed_flows"] = int(stats.get("total_flows", 0))
    stats["normal_flows"] = int(stats.get("clean", 0))
    stats["suspicious_flows"] = max(
        stats["completed_flows"] - stats["normal_flows"], 0
    )
    stats["uncertain"] = int(stats.get("uncertain", sum(
        1 for alert in alerts
        if str(alert.get("severity", "")).upper() == "UNCERTAIN"
    )))
    stats["confirmed_attack_types"] = dict(stats.get("attack_types", {}))
    stats["threat_level"] = _threat_level(stats)
    stats["total_alerts"] = int(stats.get("total_alerts", len(alerts)))
    stats["capture_running"] = bool(
        _capture_ref is not None
        and getattr(_capture_ref, "is_running", False)
        and getattr(_capture_ref, "error", None) is None
    )

    confidence_values = []
    destination_counts = Counter()
    for alert in alerts:
        raw_confidence = str(alert.get("confidence", "")).rstrip("%")
        try:
            confidence_values.append(float(raw_confidence))
        except ValueError:
            pass
        destination = str(alert.get("dst", ""))
        if destination:
            destination_counts[destination.rsplit(":", 1)[0]] += 1
    stats["confidence_avg"] = round(
        sum(confidence_values) / len(confidence_values), 1
    ) if confidence_values else 0.0
    stats["top_destinations"] = [
        {"ip": ip, "count": count}
        for ip, count in destination_counts.most_common(5)
    ]
    return stats


def _build_session_data() -> dict:
    stats = _dashboard_stats()
    capture_mode = getattr(_capture_ref, "mode", "unknown")
    return {
        "meta": {
            "session_start": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_start_time)),
            "session_end": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hostname": "dashboard session",
            "mode": str(capture_mode).capitalize(),
        },
        "stats": stats,
        "alerts": _alert_manager.all_alerts_for_report(),
        "top_sources": stats.get("top_sources", []),
    }


def run(host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
    start_broadcaster()
    socketio.run(app, host=host, port=port,
                 debug=debug, use_reloader=False, log_output=False)

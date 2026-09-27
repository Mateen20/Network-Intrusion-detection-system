from types import SimpleNamespace
from pathlib import Path

import pytest

import dashboard.app as dashboard


ALERTS = [
    {
        "id": "a1",
        "severity": "HIGH",
        "src": "10.0.0.4:50000",
        "dst": "10.0.0.8:443",
        "label": "WEB_ATTACK",
        "confidence": "91.0%",
        "timestamp": "12:00:00",
        "explanation": {},
        "mitre": {},
        "mitre_badge": {},
    },
    {
        "id": "a2",
        "severity": "UNCERTAIN",
        "src": "10.0.0.5:50001",
        "dst": "10.0.0.8:22",
        "label": "EXPLOIT",
        "confidence": "42.0%",
        "timestamp": "12:00:01",
        "explanation": {},
        "mitre": {},
        "mitre_badge": {},
    },
]


class FakeAlertManager:
    def dashboard_stats(self):
        return {
            "packets_per_sec": 4,
            "total_flows": 3,
            "clean": 1,
            "critical": 0,
            "high": 1,
            "medium": 0,
            "top_sources": [{"ip": "10.0.0.4", "count": 1}],
            "attack_types": {"web_attack": 1},
            "uncertain": 1,
            "total_alerts": 2,
        }

    def recent_alerts(self, n=50):
        return ALERTS[:n]

    def all_alerts_for_report(self):
        return ALERTS


def setup_dashboard(capture_running=True):
    capture = SimpleNamespace(
        stats={"total": 42},
        is_running=capture_running,
        error=None,
        mode="simulate",
    )
    flow_tracker = SimpleNamespace(active_count=lambda: 2)
    dashboard.init_dashboard(
        FakeAlertManager(), capture=capture, flow_tracker=flow_tracker
    )


def test_dashboard_title_and_pdf_button():
    setup_dashboard()

    html = dashboard.app.test_client().get("/").get_data(as_text=True)

    assert "<h1>Network Intrusion Detection</h1>" in html
    assert "<h1>NIDS" not in html
    assert '>PDF</button>' in html


def test_dashboard_stats_keep_packet_flow_and_alert_counts_distinct():
    setup_dashboard()

    response = dashboard.app.test_client().get("/api/stats")
    stats = response.get_json()

    assert stats["total_packets"] == 42
    assert stats["active_flows"] == 2
    assert stats["completed_flows"] == 3
    assert stats["normal_flows"] == 1
    assert stats["suspicious_flows"] == 2
    assert stats["uncertain"] == 1
    assert stats["confirmed_attack_types"] == {"web_attack": 1}
    assert stats["threat_level"] == "HIGH"
    assert stats["confidence_avg"] == 66.5
    assert stats["top_destinations"] == [{"ip": "10.0.0.8", "count": 2}]
    assert stats["capture_running"] is True


def test_attack_distribution_has_separate_confirmed_uncertain_and_normal_categories():
    setup_dashboard()

    html = dashboard.app.test_client().get("/").get_data(as_text=True)

    assert "'Uncertain','Normal'" in html
    assert "s.uncertain ?? 0, s.normal_flows ?? s.clean ?? 0" in html
    assert "HIGH RISK" in html
    assert "MEDIUM RISK" in html


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ({}, "SECURE"),
        ({"uncertain": 5}, "SECURE"),
        ({"medium": 1}, "MEDIUM"),
        ({"high": 1}, "HIGH"),
        ({"critical": 1}, "CRITICAL"),
        ({"medium": 1, "high": 1}, "HIGH"),
        ({"high": 1, "critical": 1}, "CRITICAL"),
    ],
)
def test_threat_level_uses_confirmed_severity_only(counts, expected):
    assert dashboard._threat_level(counts) == expected


def test_dashboard_pdf_download_uses_current_session():
    setup_dashboard()
    reports_dir = Path(dashboard.__file__).resolve().parents[1] / "reports"
    reports_dir.mkdir(exist_ok=True)
    files_before = set(reports_dir.iterdir())

    response = dashboard.app.test_client().get("/api/report")

    assert response.status_code == 200
    assert response.mimetype == "application/pdf"
    assert response.data.startswith(b"%PDF-")
    assert response.headers["Content-Disposition"] == "attachment; filename=nids_report.pdf"
    assert set(reports_dir.iterdir()) == files_before


def test_dashboard_pdf_without_session_returns_message():
    dashboard.init_dashboard(None)

    response = dashboard.app.test_client().get("/api/report")

    assert response.status_code == 404
    assert "No active session data" in response.get_json()["error"]
"""Bounded diagnostics for confirmed live detections only."""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from ml.trainer import FEATURE_NAMES


CONFIRMED_SEVERITIES = frozenset({"CONFIRMED", "MEDIUM", "HIGH", "CRITICAL"})
MAX_DIAGNOSTIC_RECORDS = 1000
DEFAULT_DIAGNOSTIC_PATH = (
    Path(__file__).resolve().parents[1] / "debug" / "live_confirmed_alerts.jsonl"
)


class LiveConfirmedDiagnosticWriter:
    """Persist a bounded history of completed flows already marked confirmed."""

    def __init__(self, path: str | Path = DEFAULT_DIAGNOSTIC_PATH,
                 max_records: int = MAX_DIAGNOSTIC_RECORDS):
        if max_records < 1:
            raise ValueError("max_records must be positive")
        self.path = Path(path)
        self.max_records = max_records
        self._lock = threading.Lock()

    def record(self, flow, features: dict, detection: dict, alert=None) -> bool:
        severity = str(detection.get("severity", "")).upper()
        if severity not in CONFIRMED_SEVERITIES:
            return False

        missing_features = [name for name in FEATURE_NAMES if name not in features]
        if missing_features:
            raise ValueError(f"Confirmed flow is missing canonical features: {missing_features}")

        source = str(detection.get("source", ""))
        random_forest = {
            "available": False,
            "reason": "The final detector result does not expose the Random Forest stage output",
        }
        modern_detector = {
            "available": False,
            "reason": "The final detector result does not expose a modern-stage prediction",
        }
        if source == "classic_model":
            random_forest = {
                "available": True,
                "predicted_class": detection.get("label"),
                "confidence": detection.get("confidence"),
                "probabilities": detection.get("probabilities"),
            }
        elif source == "modern_threat_detector":
            modern_detector = {
                "available": True,
                "predicted_class": detection.get("label"),
                "severity": detection.get("severity"),
                "confidence": detection.get("confidence"),
                "probabilities": detection.get("probabilities"),
                "anomaly": detection.get("anomaly"),
                "anomaly_score": detection.get("anomaly_score"),
            }

        mitre_id = self._mitre_id(alert, detection.get("label"))
        protocol = int(flow.protocol)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": {
                "ip": flow.src_ip,
                "port": int(flow.src_port),
            },
            "destination": {
                "ip": flow.dst_ip,
                "port": int(flow.dst_port),
            },
            "protocol": {
                "number": protocol,
                "name": ("TCP", "UDP", "ICMP")[min(max(protocol, 0), 2)],
            },
            "flow": {
                "duration_seconds": float(flow.duration),
                "packet_count": int(flow.pkt_count),
                "source_packet_count": int(flow.src_pkt_count),
                "destination_packet_count": int(flow.pkt_count - flow.src_pkt_count),
                "source_bytes": int(flow.src_bytes),
                "destination_bytes": int(flow.dst_bytes),
                "syn_count": int(flow.syn_count),
                "syn_ack_count": int(flow.syn_ack_count),
                "fin_count": int(flow.fin_count),
                "rst_count": int(flow.rst_count),
                "urgent_count": int(flow.urg_count),
                "wrong_fragment_count": int(flow.wrong_frags),
            },
            "feature_order": list(FEATURE_NAMES),
            "features": [
                {"name": name, "value": float(features[name])}
                for name in FEATURE_NAMES
            ],
            "random_forest": random_forest,
            "isolation_forest": {
                "anomaly": detection.get("anomaly"),
                "score": detection.get("anomaly_score"),
            },
            "modern_detector": modern_detector,
            "final_severity": severity,
            "final_attack_class": detection.get("label"),
            "mitre_id": mitre_id,
        }

        self._append_bounded(record)
        return True

    def _append_bounded(self, record: dict) -> None:
        with self._lock:
            records = []
            if self.path.exists():
                with self.path.open("r", encoding="utf-8") as diagnostic_file:
                    for line in diagnostic_file:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

            records = (records + [record])[-self.max_records:]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self.path.with_name(self.path.name + ".tmp")
            with temporary_path.open("w", encoding="utf-8") as diagnostic_file:
                for item in records:
                    diagnostic_file.write(json.dumps(item, allow_nan=False) + "\n")
            os.replace(temporary_path, self.path)

    @staticmethod
    def _mitre_id(alert, attack_class) -> str | None:
        mapping = getattr(alert, "mitre", None) if alert is not None else None
        if not mapping:
            try:
                from alerts.alert_manager import _get_mapper
                mapper = _get_mapper()
                mapping = mapper.map(attack_class) if mapper else None
            except Exception:
                mapping = None
        primary = mapping.get("primary_technique") if mapping else None
        return primary.get("id") if primary else None
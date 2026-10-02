import json
from types import SimpleNamespace

from core.live_diagnostics import LiveConfirmedDiagnosticWriter
from ml.trainer import FEATURE_NAMES


def make_flow():
    return SimpleNamespace(
        src_ip="192.0.2.10",
        dst_ip="198.51.100.20",
        src_port=51000,
        dst_port=443,
        protocol=0,
        duration=0.25,
        pkt_count=4,
        src_pkt_count=2,
        src_bytes=1200,
        dst_bytes=400,
        syn_count=1,
        syn_ack_count=0,
        fin_count=0,
        rst_count=0,
        urg_count=0,
        wrong_frags=0,
    )


def make_features():
    return {name: float(index) for index, name in enumerate(FEATURE_NAMES)}


def make_detection(severity="CRITICAL"):
    return {
        "label": "dos",
        "severity": severity,
        "confidence": 0.97,
        "anomaly": True,
        "anomaly_score": -0.42,
        "probabilities": {"dos": 0.97, "normal": 0.03},
        "source": "classic_model",
    }


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_confirmed_flow_persists_complete_ordered_diagnostic(tmp_path):
    path = tmp_path / "debug" / "live_confirmed_alerts.jsonl"
    writer = LiveConfirmedDiagnosticWriter(path)
    alert = SimpleNamespace(mitre={"primary_technique": {"id": "T1498"}})

    assert writer.record(make_flow(), make_features(), make_detection(), alert)

    record = read_jsonl(path)[0]
    assert record["source"] == {"ip": "192.0.2.10", "port": 51000}
    assert record["destination"] == {"ip": "198.51.100.20", "port": 443}
    assert record["protocol"] == {"number": 0, "name": "TCP"}
    assert record["flow"]["packet_count"] == 4
    assert record["flow"]["source_packet_count"] == 2
    assert record["flow"]["destination_packet_count"] == 2
    assert record["flow"]["source_bytes"] == 1200
    assert record["flow"]["destination_bytes"] == 400
    assert record["feature_order"] == FEATURE_NAMES
    assert [item["name"] for item in record["features"]] == FEATURE_NAMES
    assert [item["value"] for item in record["features"]] == list(make_features().values())
    assert record["random_forest"]["available"] is True
    assert record["random_forest"]["predicted_class"] == "dos"
    assert record["random_forest"]["probabilities"]["dos"] == 0.97
    assert record["isolation_forest"] == {"anomaly": True, "score": -0.42}
    assert record["modern_detector"]["available"] is False
    assert record["final_severity"] == "CRITICAL"
    assert record["final_attack_class"] == "dos"
    assert record["mitre_id"] == "T1498"
    assert record["timestamp"]


def test_normal_flow_is_not_recorded(tmp_path):
    path = tmp_path / "live_confirmed_alerts.jsonl"
    writer = LiveConfirmedDiagnosticWriter(path)
    detection = make_detection("CLEAN")
    detection.update(label="normal", source="classic_model")

    assert writer.record(make_flow(), make_features(), detection) is False
    assert path.exists() is False


def test_diagnostic_write_does_not_change_detection_result(tmp_path):
    writer = LiveConfirmedDiagnosticWriter(tmp_path / "live_confirmed_alerts.jsonl")
    detection = make_detection()
    original_detection = json.loads(json.dumps(detection))

    writer.record(make_flow(), make_features(), detection)

    assert detection == original_detection


def test_modern_final_result_is_recorded_without_claiming_rf_output(tmp_path):
    writer = LiveConfirmedDiagnosticWriter(tmp_path / "live_confirmed_alerts.jsonl")
    detection = make_detection()
    detection.update(
        label="encrypted_c2",
        source="modern_threat_detector",
        probabilities={"encrypted_c2": 0.97, "normal": 0.03},
    )

    writer.record(make_flow(), make_features(), detection)

    record = read_jsonl(writer.path)[0]
    assert record["random_forest"]["available"] is False
    assert record["modern_detector"]["available"] is True
    assert record["modern_detector"]["predicted_class"] == "encrypted_c2"
    assert record["modern_detector"]["probabilities"]["encrypted_c2"] == 0.97


def test_diagnostic_file_retains_only_most_recent_records(tmp_path):
    path = tmp_path / "live_confirmed_alerts.jsonl"
    writer = LiveConfirmedDiagnosticWriter(path, max_records=2)

    for index in range(3):
        detection = make_detection()
        detection["confidence"] = index / 10
        writer.record(make_flow(), make_features(), detection)

    records = read_jsonl(path)
    assert len(records) == 2
    assert [record["random_forest"]["confidence"] for record in records] == [0.1, 0.2]
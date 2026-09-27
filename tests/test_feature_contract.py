import math

from core.flow_tracker import FLOW_TIMEOUT, FlowTracker
from ml.trainer import (
    FEATURE_NAMES,
    NIDSTrainer,
    auto_load_dataset,
    generate_synthetic_dataset,
)


def test_live_flow_features_match_training_schema():
    tracker = FlowTracker()
    pkt = {
        "src_ip": "192.168.1.10",
        "dst_ip": "192.168.1.1",
        "src_port": 50000,
        "dst_port": 443,
        "protocol": 0,
        "size": 120,
        "flags": {"SYN": True, "ACK": True, "FIN": False, "RST": False, "URG": False},
        "ts": 0.0,
    }

    tracker.process_packet(pkt)
    assert tracker.active_count() == 1

    # A newly observed flow must remain active until its inactivity timeout.
    tracker._flows[next(iter(tracker._flows))].last_seen -= FLOW_TIMEOUT + 1
    expired = tracker.collect_expired()
    assert expired, "expected the inactive flow to be collected"

    _, features = expired[0]
    assert list(features.keys()) == FEATURE_NAMES
    assert len(features) == len(FEATURE_NAMES)
    assert all(isinstance(value, float) for value in features.values())
    assert all(math.isfinite(value) for value in features.values())

    # Basic sanity: protocol should be one of the training model codes
    assert features["protocol_type"] in (0.0, 1.0, 2.0)
    assert features["port_number"] == 443.0
    assert features["duration"] > 0.0

    # Missing features must not silently become a misleading all-zero vector
    assert any(value != 0.0 for value in [features["src_bytes"], features["dst_bytes"], features["packet_rate"]])


def test_feature_names_are_in_order_and_valid_for_scaler():
    feature_names = FEATURE_NAMES
    assert feature_names[0] == "duration"
    assert feature_names[-1] == "is_well_known_port"
    assert len(feature_names) == 24
    assert len(set(feature_names)) == len(feature_names)
    assert all(name and name.strip() for name in feature_names)

    # ensure the names are the canonical schema expected by the model
    assert feature_names == [
        "duration", "protocol_type", "src_bytes", "dst_bytes",
        "wrong_fragment", "urgent", "count", "srv_count",
        "serror_rate", "rerror_rate", "same_srv_rate", "diff_srv_rate",
        "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate", "dst_host_serror_rate",
        "packet_rate", "byte_rate",
        "flag_syn_ratio", "flag_fin_ratio", "flag_rst_ratio",
        "port_number", "is_well_known_port",
    ]


def test_nsl_kdd_loader_projects_original_fields_to_live_contract(tmp_path):
    values = [0] * 41
    values[0] = 2
    values[1] = "tcp"
    values[2] = "http"
    values[3] = "SF"
    values[4] = 100
    values[5] = 200
    values[7] = 1
    values[8] = 2
    values[22] = 10
    values[23] = 8
    values[24] = 0.1
    values[26] = 0.2
    values[28] = 0.75
    values[29] = 0.25
    values[31] = 12
    values[32] = 9
    values[33] = 0.75
    values[34] = 0.25
    values[39] = 0.1
    dataset = tmp_path / "KDDTrain+.txt"
    dataset.write_text(",".join(map(str, values + ["normal", 17])), encoding="utf-8")

    features, labels = auto_load_dataset(str(dataset))

    assert list(features.columns) == FEATURE_NAMES
    assert labels.tolist() == ["normal"]
    assert features.loc[0, "protocol_type"] == 0.0
    assert features.loc[0, "src_bytes"] == 100.0
    assert features.loc[0, "dst_bytes"] == 200.0
    assert features.loc[0, "wrong_fragment"] == 1.0
    assert features.loc[0, "urgent"] == 2.0
    assert features.loc[0, "port_number"] == 80.0
    assert features.loc[0, "packet_rate"] == 5.0
    assert features.loc[0, "dst_host_same_srv_rate"] == 0.75


def test_legacy_model_artifacts_without_feature_metadata_are_not_loaded(tmp_path):
    for name in ("rf.pkl", "iso.pkl", "scaler.pkl", "label_enc.pkl"):
        (tmp_path / name).touch()

    trainer = NIDSTrainer(model_dir=str(tmp_path))

    assert trainer.models_exist() is False
    assert trainer.load() is False


def test_normal_training_duration_covers_observed_live_short_flows():
    features, labels = generate_synthetic_dataset(1000)
    normal_duration = features.loc[labels == "normal", "duration"]

    assert 0.2 <= normal_duration.median() <= 0.45
    assert normal_duration.quantile(0.95) >= 1.0
    assert normal_duration.quantile(0.99) <= 10.0


def test_normal_training_includes_legitimate_udp_discovery_ports():
    features, labels = generate_synthetic_dataset(1000)
    normal = features.loc[labels == "normal"]
    discovery = normal.loc[normal["port_number"].isin([1900.0, 5353.0, 5355.0])]

    assert len(discovery) >= 150
    assert (discovery["protocol_type"] == 1.0).all()
    assert (discovery["is_well_known_port"] == 0.0).all()
    assert (discovery["wrong_fragment"] == 0.0).all()
    assert (discovery["urgent"] == 0.0).all()

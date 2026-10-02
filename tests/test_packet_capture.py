import time
from collections import Counter

import pytest

from alerts.alert_manager import AlertManager
import core.packet_capture as packet_capture
from core.flow_tracker import FLOW_TIMEOUT, FlowTracker
from ml.detector import NIDSDetector
from ml.trainer import (
    ATTACK_CLASSES,
    FEATURE_NAMES,
    NIDSTrainer,
    generate_synthetic_dataset,
)
from core.packet_capture import (
    SIMULATION_PROFILES,
    PacketCapture,
    PacketCaptureError,
    select_interface,
)


INTERFACES = [
    {
        "identifier": r"\Device\NPF_{WIFI}",
        "name": "Wi-Fi",
        "description": "Intel Wi-Fi Adapter",
        "guid": "{WIFI}",
        "ip": "192.168.1.10",
    },
    {
        "identifier": r"\Device\NPF_{ETH}",
        "name": "Ethernet",
        "description": "Ethernet Adapter",
        "guid": "{ETH}",
        "ip": "192.168.1.11",
    },
]


def test_auto_selects_usable_wifi_interface():
    selected = select_interface("auto", INTERFACES)

    assert selected["name"] == "Wi-Fi"
    assert selected["identifier"] == r"\Device\NPF_{WIFI}"


def test_invalid_interface_lists_scapy_options():
    with pytest.raises(PacketCaptureError, match="Available interfaces") as error:
        select_interface("2", INTERFACES)

    assert "Wi-Fi" in str(error.value)
    assert r"\Device\NPF_{WIFI}" in str(error.value)


def test_live_capture_uses_resolved_scapy_identifier(monkeypatch):
    monkeypatch.setattr(packet_capture, "discover_interfaces", lambda: INTERFACES)

    capture = PacketCapture(mode="live", interface="Wi-Fi")
    capture._live_capture = lambda: setattr(capture, "_running", False)
    capture.start()
    capture.stop()

    assert capture.interface == r"\Device\NPF_{WIFI}"
    assert capture.interface_name == "Wi-Fi"


def test_stop_cleanly_joins_live_capture_worker(monkeypatch):
    monkeypatch.setattr(packet_capture, "discover_interfaces", lambda: INTERFACES)

    capture = PacketCapture(mode="live", interface="auto")

    def worker():
        while capture.is_running:
            time.sleep(0.01)

    capture._live_capture = worker
    capture.start()
    capture.stop()

    assert not capture.is_running
    assert capture._thread is None
    assert capture.error is None


def test_simulation_scenarios_produce_distinct_flow_features():
    features_by_scenario = {}

    for scenario in SIMULATION_PROFILES:
        packets = PacketCapture._make_scenario_packets(scenario)
        assert packets
        assert all("label" not in packet and "_attack" not in packet for packet in packets)

        tracker = FlowTracker()
        for packet in packets:
            tracker.process_packet(packet)
        for flow in tracker._flows.values():
            flow.last_seen -= FLOW_TIMEOUT + 1

        completed = tracker.collect_expired()
        assert completed
        features = [flow_features for _, flow_features in completed]
        feature_names = (
            "src_bytes", "dst_bytes", "count", "srv_count", "serror_rate",
            "rerror_rate", "same_srv_rate", "diff_srv_rate", "packet_rate",
            "byte_rate", "flag_syn_ratio", "flag_rst_ratio", "port_number",
        )
        features_by_scenario[scenario] = tuple(
            round(sum(row[name] for row in features) / len(features), 3)
            for name in feature_names
        )

    assert len(set(features_by_scenario.values())) == len(SIMULATION_PROFILES)
    assert all(
        packets[0]["src_port"] >= 1024
        for packets in (
            PacketCapture._make_scenario_packets(scenario)
            for scenario in SIMULATION_PROFILES
        )
    )


def test_attack_scenario_emits_one_unlabeled_behavior_batch(monkeypatch):
    packets = []
    capture = PacketCapture(mode="simulate", callback=packets.append)
    capture._running = True
    monkeypatch.setattr(packet_capture.time, "sleep", lambda _: None)

    worker = capture._inject_attack("dos")
    worker.join(timeout=2)

    expected = (
        SIMULATION_PROFILES["dos"]["connections"]
        * SIMULATION_PROFILES["dos"]["packets"]
    )
    assert not worker.is_alive()
    assert len(packets) == expected
    assert capture.stats["attacks_injected"] == expected
    assert all(
        not {"label", "attack", "attack_type", "_attack"}.intersection(packet)
        for packet in packets
    )


def test_simulation_uses_real_detector_pipeline_without_injected_labels(tmp_path):
    import numpy as np

    training_features, training_labels = generate_synthetic_dataset(300)
    normal_training = training_features[training_labels == "normal"]
    normal_bounds = {
        name: (
            float(normal_training[name].quantile(0.001)),
            float(normal_training[name].quantile(0.999)),
        )
        for name in FEATURE_NAMES
    }
    random_state = np.random.get_state()
    np.random.seed(42)
    try:
        trainer = NIDSTrainer(model_dir=str(tmp_path))
        trainer.train(n_per_class=1500)
    finally:
        np.random.set_state(random_state)

    assert trainer.models_exist()
    assert trainer.scaler.feature_names_in_.tolist() == FEATURE_NAMES
    detector = NIDSDetector(trainer)

    captured_features = dict(zip(FEATURE_NAMES, [
        0.001, 0.0, 66.0, 0.0, 0.0, 0.0, 4.0, 4.0,
        1.0, 0.0, 1.0, 0.0, 60.0, 57.0, 0.95, 0.05,
        1.0, 2.0, 66.0, 1.0, 0.0, 0.0, 53.0, 1.0,
    ]))
    corrected_capture_features = dict(captured_features)
    corrected_capture_features["dst_host_serror_rate"] = 1.0 / 60.0
    corrected_detection = detector.predict(corrected_capture_features)
    assert list(corrected_capture_features) == FEATURE_NAMES
    assert corrected_detection["severity"] in {"CLEAN", "UNCERTAIN"}
    assert corrected_detection["severity"] not in {"MEDIUM", "HIGH", "CRITICAL"}
    if corrected_detection["severity"] == "CLEAN":
        assert corrected_detection["label"] == "normal"

    def dns_detection(protocol):
        tracker = FlowTracker()
        timestamp = time.time()
        client_ip = "10.0.0.25"
        resolver_ip = "203.0.113.53"
        client_port = 53500 + protocol
        if protocol == 0:
            packets = [
                (client_ip, resolver_ip, client_port, 53, 60, {"SYN": True}, 0.0),
                (resolver_ip, client_ip, 53, client_port, 60, {"SYN": True, "ACK": True}, 0.01),
                (client_ip, resolver_ip, client_port, 53, 52, {"ACK": True}, 0.02),
                (client_ip, resolver_ip, client_port, 53, 72, {"ACK": True}, 0.03),
                (resolver_ip, client_ip, 53, client_port, 104, {"ACK": True}, 0.04),
            ]
        else:
            packets = [
                (client_ip, resolver_ip, client_port, 53, 71, {}, 0.0),
                (resolver_ip, client_ip, 53, client_port, 125, {}, 0.02),
            ]

        for src_ip, dst_ip, src_port, dst_port, size, flags, offset in packets:
            tracker.process_packet({
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "src_port": src_port,
                "dst_port": dst_port,
                "protocol": protocol,
                "size": size,
                "flags": flags,
                "ts": timestamp + offset,
            })

        flow = next(iter(tracker._flows.values()))
        duration = flow.duration
        flow.last_seen = time.time() - FLOW_TIMEOUT - 1
        flow.start_time = flow.last_seen - duration
        _, features = tracker.collect_expired()[0]
        assert list(features) == FEATURE_NAMES
        assert features["port_number"] == 53.0
        assert features["protocol_type"] == float(protocol)
        return detector.predict(features)

    for dns_result in (dns_detection(0), dns_detection(1)):
        assert (
            dns_result["label"] == "normal" and dns_result["severity"] == "CLEAN"
        ) or dns_result["severity"] == "UNCERTAIN"

    response_first_tracker = FlowTracker()
    timestamp = time.time()
    response_first_packets = [
        {
            "src_ip": "93.184.216.34", "dst_ip": "192.168.1.10",
            "src_port": 443, "dst_port": 52000, "protocol": 0,
            "size": 7553, "flags": {"ACK": True},
            "ts": timestamp,
        },
        {
            "src_ip": "192.168.1.10", "dst_ip": "93.184.216.34",
            "src_port": 52000, "dst_port": 443, "protocol": 0,
            "size": 1908, "flags": {"ACK": True},
            "ts": timestamp + 0.922,
        },
    ]
    for packet in response_first_packets:
        response_first_tracker.process_packet(packet)
    response_first_flow = next(iter(response_first_tracker._flows.values()))
    response_first_duration = response_first_flow.duration
    response_first_flow.last_seen = time.time() - FLOW_TIMEOUT - 1
    response_first_flow.start_time = response_first_flow.last_seen - response_first_duration
    _, response_first_features = response_first_tracker.collect_expired()[0]
    response_first_detection = detector.predict(response_first_features)
    assert response_first_features["port_number"] == 443.0
    assert response_first_features["src_bytes"] == 1908.0
    assert response_first_features["dst_bytes"] == 7553.0
    assert (response_first_detection["label"], response_first_detection["severity"]) == (
        "normal", "CLEAN"
    )

    discovery_flows = [
        [
            ("10.66.239.176", "224.0.0.252", 55906, 5355, 1, 71, 0.0),
            ("10.66.239.176", "224.0.0.252", 55906, 5355, 1, 71, 0.4),
        ],
        [
            ("10.66.239.176", "239.255.255.250", 62119, 1900, 1, 800, 0.0),
            ("10.66.239.176", "239.255.255.250", 62119, 1900, 1, 811, 23.9),
        ],
    ]
    for packets in discovery_flows:
        tracker = FlowTracker()
        base = time.time()
        for src_ip, dst_ip, src_port, dst_port, protocol, size, offset in packets:
            tracker.process_packet({
                "src_ip": src_ip, "dst_ip": dst_ip,
                "src_port": src_port, "dst_port": dst_port,
                "protocol": protocol, "size": size, "flags": {},
                "ts": base + offset,
            })
        flow = next(iter(tracker._flows.values()))
        duration = flow.duration
        flow.last_seen = time.time() - FLOW_TIMEOUT - 1
        flow.start_time = flow.last_seen - duration
        _, features = tracker.collect_expired()[0]
        detection = detector.predict(features)
        assert features["port_number"] in {1900.0, 5355.0}
        assert features["protocol_type"] == 1.0
        assert (detection["label"], detection["severity"]) == ("normal", "CLEAN")

    alert_manager = AlertManager()
    predictions = Counter()
    capture = PacketCapture(mode="simulate")
    processed_flows = 0

    for scenario, profile in SIMULATION_PROFILES.items():
        tracker = FlowTracker()
        packets = capture._make_scenario_packets(scenario)
        timestamp = time.time()
        assert packets
        assert all(
            not {"label", "attack", "attack_type", "_attack"}.intersection(packet)
            for packet in packets
        )

        capture.set_callback(tracker.process_packet)
        for index, packet in enumerate(packets):
            packet["ts"] = timestamp + index * profile["interval"]
            capture.callback(packet)

        for flow in tracker._flows.values():
            duration = flow.duration
            flow.last_seen = time.time() - FLOW_TIMEOUT - 1
            flow.start_time = flow.last_seen - duration

        completed = tracker.collect_expired()
        assert completed
        processed_flows += len(completed)
        sample_indexes = {0, len(completed) // 2, len(completed) - 1}
        for index in sorted(sample_indexes):
            flow, features = completed[index]
            assert list(features) == FEATURE_NAMES
            if scenario == "normal":
                assert 0.0 < features["duration"] < 1.0
                assert 0.5 <= features["packet_rate"] <= 25.0
                assert 0.0 <= features["serror_rate"] <= 0.1
                assert 0.0 <= features["rerror_rate"] <= 0.1
                assert 0.7 <= features["same_srv_rate"] <= 1.0
                assert 0.0 <= features["diff_srv_rate"] <= 0.3
                assert 1.0 <= features["count"] <= 50.0
                assert 1.0 <= features["srv_count"] <= 30.0
                assert 0.0 <= features["flag_syn_ratio"] <= 0.3
                assert 0.0 <= features["flag_fin_ratio"] <= 0.3
                assert 0.0 <= features["flag_rst_ratio"] <= 0.1
                assert features["port_number"] == 443.0
                assert features["is_well_known_port"] == 1.0
                for name in (
                    "duration", "src_bytes", "dst_bytes", "count", "srv_count",
                    "dst_host_count", "dst_host_srv_count", "packet_rate",
                    "byte_rate", "flag_syn_ratio", "flag_fin_ratio", "flag_rst_ratio",
                ):
                    lower, upper = normal_bounds[name]
                    if name.endswith("_ratio") or name == "duration":
                        boundary_tolerance = 0.01
                    elif name in {"count", "srv_count", "dst_host_count", "dst_host_srv_count"}:
                        boundary_tolerance = 1.0
                    else:
                        boundary_tolerance = 0.0
                    assert lower - boundary_tolerance <= features[name] <= upper + boundary_tolerance, name
            detection = detector.predict(features)
            assert detection["label"] in ATTACK_CLASSES
            assert detection["severity"] in {
                "CLEAN", "UNCERTAIN", "MEDIUM", "HIGH", "CRITICAL"
            }
            predictions[(detection["label"], detection["severity"])] += 1
            alert_manager.add(flow, detection, {})

    stats = alert_manager.dashboard_stats()
    confirmed = Counter()
    for (label, severity), count in predictions.items():
        if severity in {"MEDIUM", "HIGH", "CRITICAL"}:
            confirmed[label] += count

    assert processed_flows > len(predictions)
    assert predictions[("normal", "CLEAN")] > 0
    assert sum(count for (_, severity), count in predictions.items() if severity == "UNCERTAIN") > 0
    assert len(confirmed) >= 2
    assert stats["attack_types"] == dict(confirmed)
    assert stats["uncertain"] == sum(
        count for (_, severity), count in predictions.items() if severity == "UNCERTAIN"
    )
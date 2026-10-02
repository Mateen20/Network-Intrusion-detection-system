import time

from core.flow_tracker import FLOW_TIMEOUT, FlowTracker
from ml.trainer import FEATURE_NAMES


def packet(size=120, flags=None):
    return {
        "src_ip": "192.168.1.10",
        "dst_ip": "192.168.1.1",
        "src_port": 50000,
        "dst_port": 443,
        "protocol": 0,
        "size": size,
        "flags": flags or {"SYN": False, "ACK": True, "FIN": False, "RST": False, "URG": False},
    }


def age_flow(tracker):
    flow = next(iter(tracker._flows.values()))
    flow.last_seen -= FLOW_TIMEOUT + 1


def test_flow_accumulates_multiple_packets_before_collection():
    tracker = FlowTracker()
    tracker.process_packet(packet(120, {"SYN": True, "ACK": True, "FIN": False, "RST": False, "URG": False}))
    tracker.process_packet(packet(80))

    assert tracker.active_count() == 1
    flow = next(iter(tracker._flows.values()))
    assert flow.pkt_count == 2
    assert flow.src_bytes == 200


def test_flow_remains_active_before_timeout():
    tracker = FlowTracker()
    tracker.process_packet(packet())

    assert tracker.collect_expired() == []
    assert tracker.active_count() == 1


def test_flow_expires_after_inactivity():
    tracker = FlowTracker()
    tracker.process_packet(packet())
    age_flow(tracker)

    expired = tracker.collect_expired()

    assert len(expired) == 1
    assert tracker.active_count() == 0


def test_expired_flow_is_collected_once():
    tracker = FlowTracker()
    tracker.process_packet(packet())
    age_flow(tracker)

    assert len(tracker.collect_expired()) == 1
    assert tracker.collect_expired() == []


def test_reappearing_tuple_completes_old_flow_before_starting_new_flow():
    tracker = FlowTracker()
    tracker.process_packet(packet(120))
    old_flow = next(iter(tracker._flows.values()))
    old_flow.last_seen -= FLOW_TIMEOUT + 1

    returned = tracker.process_packet(packet(80))

    assert returned is old_flow
    assert tracker.active_count() == 1
    completed = tracker.collect_expired()
    assert len(completed) == 1
    assert completed[0][0] is old_flow
    assert completed[0][1]["src_bytes"] == 120.0
    assert tracker.collect_expired() == []


def test_response_first_flow_uses_initiator_to_service_feature_direction():
    tracker = FlowTracker()
    timestamp = time.time()
    tracker.process_packet({
        "src_ip": "93.184.216.34",
        "dst_ip": "192.168.1.10",
        "src_port": 443,
        "dst_port": 52000,
        "protocol": 0,
        "size": 1200,
        "flags": {"ACK": True},
        "ts": timestamp,
    })
    tracker.process_packet({
        "src_ip": "192.168.1.10",
        "dst_ip": "93.184.216.34",
        "src_port": 52000,
        "dst_port": 443,
        "protocol": 0,
        "size": 400,
        "flags": {"ACK": True},
        "ts": timestamp + 0.1,
    })

    flow = next(iter(tracker._flows.values()))
    assert flow.src_ip == "192.168.1.10"
    assert flow.src_port == 52000
    assert flow.dst_ip == "93.184.216.34"
    assert flow.dst_port == 443

    flow.last_seen -= FLOW_TIMEOUT + 1
    _, features = tracker.collect_expired()[0]
    assert features["src_bytes"] == 400.0
    assert features["dst_bytes"] == 1200.0
    assert features["port_number"] == 443.0
    assert features["is_well_known_port"] == 1.0


def test_collected_flow_has_canonical_features():
    tracker = FlowTracker()
    tracker.process_packet(packet(120, {"SYN": True, "ACK": True, "FIN": False, "RST": False, "URG": False}))
    tracker.process_packet(packet(80))
    age_flow(tracker)

    _, features = tracker.collect_expired()[0]

    assert list(features) == FEATURE_NAMES
    assert len(features) == 24
    assert features["src_bytes"] == 200.0
    assert features["port_number"] == 443.0


def _collect_host_error_features(connection_count, failed_count):
    tracker = FlowTracker()
    timestamp = time.time()
    for index in range(connection_count):
        flags = (
            {"SYN": True, "ACK": False, "FIN": False, "RST": False, "URG": False}
            if index < failed_count
            else {"SYN": False, "ACK": True, "FIN": False, "RST": False, "URG": False}
        )
        tracker.process_packet({
            "src_ip": "192.168.1.10",
            "dst_ip": "192.168.1.1",
            "src_port": 50000 + index,
            "dst_port": 443,
            "protocol": 0,
            "size": 66,
            "flags": flags,
            "ts": timestamp,
        })

    for flow in tracker._flows.values():
        duration = flow.duration
        flow.last_seen = time.time() - FLOW_TIMEOUT - 1
        flow.start_time = flow.last_seen - duration

    return tracker.collect_expired()


def test_single_incomplete_syn_does_not_mark_host_aggregate_fully_failed():
    completed = _collect_host_error_features(connection_count=60, failed_count=1)

    assert len(completed) == 60
    assert all(features["serror_rate"] == 1.0 for flow, features in completed if flow.syn_count)
    assert all(features["dst_host_serror_rate"] < 1.0 for _, features in completed)


def test_host_serror_rate_is_failed_connections_over_host_connections():
    completed = _collect_host_error_features(connection_count=60, failed_count=1)

    assert all(
        features["dst_host_serror_rate"] == 1 / 60
        for _, features in completed
    )
    assert all(list(features) == FEATURE_NAMES for _, features in completed)
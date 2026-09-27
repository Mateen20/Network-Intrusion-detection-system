"""
core/flow_tracker.py — Aggregates raw packet events into network flows
and extracts ML-ready features from each completed flow.
"""

import time
import threading
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field

FLOW_TIMEOUT = 30.0   # seconds of inactivity before flow is finalised


def _canonical_feature_names() -> list[str]:
    from ml.trainer import FEATURE_NAMES
    return list(FEATURE_NAMES)


@dataclass
class Flow:
    src_ip:    str
    dst_ip:    str
    src_port:  int
    dst_port:  int
    protocol:  int      # 0=TCP 1=UDP 2=ICMP

    # Counters updated as packets arrive
    start_time:    float = field(default_factory=time.time)
    last_seen:     float = field(default_factory=time.time)
    pkt_count:     int   = 0
    src_pkt_count: int   = 0
    src_bytes:     int   = 0
    dst_bytes:     int   = 0
    syn_count:     int   = 0
    syn_ack_count: int   = 0
    fin_count:     int   = 0
    rst_count:     int   = 0
    urg_count:     int   = 0
    wrong_frags:   int   = 0
    same_srv:      int   = 0
    diff_srv:      int   = 0
    connection_stats: dict = field(default_factory=dict)
    connection_record: dict | None = None

    def update(self, pkt_size: int, is_src: bool, flags: dict, timestamp: float):
        self.pkt_count  += 1
        self.last_seen   = max(self.last_seen, timestamp)
        if is_src:
            self.src_pkt_count += 1
            self.src_bytes += pkt_size
        else:
            self.dst_bytes += pkt_size
        self.syn_count  += int(flags.get("SYN", False) and not flags.get("ACK", False))
        self.syn_ack_count += int(flags.get("SYN", False) and flags.get("ACK", False))
        self.fin_count  += int(flags.get("FIN", False))
        self.rst_count  += int(flags.get("RST", False))
        self.urg_count  += int(flags.get("URG", False))

    @property
    def duration(self) -> float:
        return self.last_seen - self.start_time

    def is_expired(self) -> bool:
        return (time.time() - self.last_seen) > FLOW_TIMEOUT

    def to_features(self, global_stats: dict) -> dict:
        """Convert a flow into the exact named feature vector expected by the trained model."""
        duration  = max(self.duration, 0.001)
        rate_duration = max(duration, 1.0)
        pkt_rate  = float(global_stats.get("conn_rate", 0.0))
        byte_rate = (self.src_bytes + self.dst_bytes) / rate_duration
        pkt_cnt   = max(self.pkt_count, 1)

        same_total = max(global_stats.get("same_srv_total", 1), 1)
        diff_total = max(global_stats.get("diff_srv_total", 0), 0)
        same_srv_rate = float(global_stats.get("same_srv_rate", same_total / max(same_total + diff_total, 1)))
        diff_srv_rate = float(global_stats.get("diff_srv_rate", diff_total / max(same_total + diff_total, 1)))

        feature_map = {
            "duration":               float(duration),
            "protocol_type":          float(self.protocol),
            "src_bytes":              float(self.src_bytes),
            "dst_bytes":              float(self.dst_bytes),
            "wrong_fragment":         float(self.wrong_frags),
            "urgent":                 float(self.urg_count),
            "count":                  float(global_stats.get("conn_2s", 1)),
            "srv_count":              float(global_stats.get("srv_2s", 1)),
            "serror_rate":            float(self.syn_count > 0 and self.syn_ack_count == 0),
            "rerror_rate":            float(self.rst_count / pkt_cnt),
            "same_srv_rate":          same_srv_rate,
            "diff_srv_rate":          diff_srv_rate,
            "dst_host_count":         float(global_stats.get("dst_host_count", 1)),
            "dst_host_srv_count":     float(global_stats.get("dst_host_srv_count", 1)),
            "dst_host_same_srv_rate": float(global_stats.get("dst_host_same_srv_rate", 1.0)),
            "dst_host_diff_srv_rate": float(global_stats.get("dst_host_diff_srv_rate", 0.0)),
            "dst_host_serror_rate":   float(global_stats.get("dst_host_serror_rate", 0.0)),
            "packet_rate":            float(pkt_rate),
            "byte_rate":              float(byte_rate),
            "flag_syn_ratio":         float(self.syn_count / max(self.src_pkt_count, 1)),
            "flag_fin_ratio":         float(self.fin_count / pkt_cnt),
            "flag_rst_ratio":         float(self.rst_count / pkt_cnt),
            "port_number":            float(self.dst_port),
            "is_well_known_port":     float(1.0 if 0 < self.dst_port < 1024 else 0.0),
        }

        feature_names = _canonical_feature_names()
        ordered = {name: float(feature_map[name]) for name in feature_names}
        return ordered


class FlowTracker:
    """Tracks all active network flows and emits completed flows."""

    def __init__(self):
        self._flows:  dict[tuple, Flow] = {}
        self._lock = threading.RLock()
        self._history: list[dict]       = []   # recent completed flows
        self._conn_window: list[dict]   = []   # connection starts in 2-sec window
        self._connection_history: deque[dict] = deque(maxlen=512)
        self._completed_flows: list[Flow] = []
        self.last_feature_vector: dict | None = None
        self.last_feature_debug: dict | None = None

    # ─── Public API ───────────────────────────────────────────────────────────

    def process_packet(self, pkt_info: dict) -> Flow | None:
        """
        Feed one packet. Returns a completed Flow if one just expired,
        else None.
        """
        with self._lock:
            return self._process_packet(pkt_info)

    def _process_packet(self, pkt_info: dict) -> Flow | None:
        key     = self._flow_key(pkt_info)
        rev_key = self._flow_key(pkt_info, reverse=True)

        timestamp = self._packet_timestamp(pkt_info)
        self._conn_window = [
            connection for connection in self._conn_window
            if 0.0 <= timestamp - connection["timestamp"] < 2.0
        ]
        completed_flow = self._queue_expired_flows()

        # Look up existing flow
        if key in self._flows:
            flow = self._flows[key]
            is_src = True
            new_connection = False
        elif rev_key in self._flows:
            flow   = self._flows[rev_key]
            key    = rev_key
            is_src = False
            new_connection = False
        else:
            # New flow
            src_port = pkt_info.get("src_port", 0)
            dst_port = pkt_info.get("dst_port", 0)
            reverse_initial_packet = 0 < src_port < 1024 and dst_port >= 1024
            flow = Flow(
                src_ip   = pkt_info.get("dst_ip" if reverse_initial_packet else "src_ip", "0.0.0.0"),
                dst_ip   = pkt_info.get("src_ip" if reverse_initial_packet else "dst_ip", "0.0.0.0"),
                src_port = dst_port if reverse_initial_packet else src_port,
                dst_port = src_port if reverse_initial_packet else dst_port,
                protocol = pkt_info.get("protocol", 0),
                start_time = timestamp,
                last_seen = timestamp,
            )
            key = self._flow_key(pkt_info, reverse=reverse_initial_packet)
            self._flows[key] = flow
            is_src = not reverse_initial_packet
            new_connection = True

        if new_connection:
            record = {
                "timestamp": timestamp,
                "dst_ip": flow.dst_ip,
                "dst_port": flow.dst_port,
                "protocol": flow.protocol,
                "failed": False,
            }
            flow.connection_record = record
            self._conn_window.append(record)
            self._connection_history.append(record)

        flow.update(
            pkt_size = pkt_info.get("size", 0),
            is_src   = is_src,
            flags    = pkt_info.get("flags", {}),
            timestamp = timestamp,
        )
        if flow.connection_record is not None:
            flow.connection_record["failed"] = bool(
                flow.syn_count and flow.rst_count and not flow.syn_ack_count
            )
        flow.connection_stats = self._global_stats(flow, timestamp)

        return completed_flow

    def collect_expired(self) -> list[tuple[Flow, dict]]:
        """Return and remove all expired flows as (flow, features) pairs."""
        with self._lock:
            self._queue_expired_flows()
            results = []
            completed = self._completed_flows
            self._completed_flows = []
            for flow in completed:
                if flow.connection_record is not None and flow.syn_count and not flow.syn_ack_count:
                    flow.connection_record["failed"] = True
                stats = dict(flow.connection_stats)
                if flow.connection_record is not None and flow.connection_record["failed"]:
                    stats["dst_host_serror_rate"] = max(
                        stats.get("dst_host_serror_rate", 0.0), 1.0
                    )
                features = flow.to_features(stats)
                self.last_feature_vector = features
                self.last_feature_debug = {"flow": flow.__dict__.copy(), "stats": stats}
                results.append((flow, features))
            return results

    def active_count(self) -> int:
        return len(self._flows)

    def get_last_feature_debug(self) -> dict | None:
        return self.last_feature_debug

    def get_last_feature_vector(self) -> dict | None:
        return self.last_feature_vector

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _queue_expired_flows(self) -> Flow | None:
        first_completed = None
        expired_keys = [key for key, flow in self._flows.items() if flow.is_expired()]
        for key in expired_keys:
            flow = self._flows.pop(key, None)
            if flow is None or flow.pkt_count == 0:
                continue
            self._completed_flows.append(flow)
            if first_completed is None:
                first_completed = flow
        return first_completed

    def _global_stats(self, flow: Flow, timestamp: float) -> dict:
        recent_connections = [
            connection for connection in self._connection_history
            if 0.0 <= timestamp - connection["timestamp"] < 2.0
        ]
        host_connections = [
            connection for connection in self._connection_history
            if connection["dst_ip"] == flow.dst_ip
        ]
        same_service = [
            connection for connection in host_connections
            if connection["dst_port"] == flow.dst_port
            and connection["protocol"] == flow.protocol
        ]
        same_recent_service = [
            connection for connection in recent_connections
            if connection["dst_ip"] == flow.dst_ip
            and connection["dst_port"] == flow.dst_port
            and connection["protocol"] == flow.protocol
        ]
        recent_host = [
            connection for connection in recent_connections
            if connection["dst_ip"] == flow.dst_ip
        ]
        same_total = max(len(recent_host), 1)

        return {
            "conn_2s":               len(recent_host),
            "srv_2s":                max(len(same_recent_service), 1),
            "dst_host_count":        min(len(host_connections), 255),
            "dst_host_srv_count":    min(len(same_service), 255),
            "dst_host_same_srv_rate": len(same_service) / max(len(host_connections), 1),
            "dst_host_diff_srv_rate": max(len(host_connections) - len(same_service), 0) / max(len(host_connections), 1),
            "dst_host_serror_rate":  sum(c["failed"] for c in host_connections) / max(len(host_connections), 1),
            "same_srv_total":        len(same_recent_service),
            "diff_srv_total":        max(len(recent_host) - len(same_recent_service), 0),
            "same_srv_rate":         len(same_recent_service) / same_total,
            "diff_srv_rate":         max(len(recent_host) - len(same_recent_service), 0) / same_total,
            "conn_rate":             len(recent_host) / 2.0,
        }

    @staticmethod
    def _packet_timestamp(pkt_info: dict) -> float:
        try:
            timestamp = float(pkt_info.get("ts", time.time()))
        except (TypeError, ValueError):
            return time.time()
        return timestamp if math.isfinite(timestamp) and timestamp > 0 else time.time()

    @staticmethod
    def _flow_key(pkt: dict, reverse: bool = False) -> tuple:
        if reverse:
            return (pkt.get("dst_ip"), pkt.get("src_ip"),
                    pkt.get("dst_port"), pkt.get("src_port"),
                    pkt.get("protocol"))
        return (pkt.get("src_ip"), pkt.get("dst_ip"),
                pkt.get("src_port"), pkt.get("dst_port"),
                pkt.get("protocol"))

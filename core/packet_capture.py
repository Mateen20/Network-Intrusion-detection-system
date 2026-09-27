"""
core/packet_capture.py — Packet capture with two modes:
  LIVE      : Scapy sniff on real interface (requires root/admin)
  SIMULATE  : Realistic synthetic traffic generator (works everywhere)
"""

import random
import threading
import time
from typing import Callable


class PacketCaptureError(RuntimeError):
    """Raised when live capture cannot be configured or started."""


def discover_interfaces() -> list[dict]:
    """Return Scapy's Windows/Npcap interfaces with friendly metadata."""
    try:
        from scapy.all import conf, get_if_list
    except ImportError as exc:
        raise PacketCaptureError(
            "Scapy is not installed. Install the project requirements before "
            "using --mode live."
        ) from exc

    interfaces = []
    for identifier in get_if_list():
        details = conf.ifaces.get(identifier)
        interfaces.append({
            "identifier": identifier,
            "name": getattr(details, "name", "") or identifier,
            "description": getattr(details, "description", "") or "",
            "guid": getattr(details, "guid", "") or "",
            "ip": getattr(details, "ip", "") or "",
        })
    return interfaces


def select_interface(requested: str, interfaces: list[dict]) -> dict:
    """Resolve an explicit Scapy identifier/name or automatically find Wi-Fi."""
    value = (requested or "auto").strip()
    lowered = value.casefold()

    if lowered in {"auto", "wifi", "wi-fi", "wireless"}:
        candidates = [
            iface for iface in interfaces
            if "wi-fi" in iface["name"].casefold()
            or "wifi" in iface["name"].casefold()
            or "wireless" in iface["description"].casefold()
            or "wi-fi" in iface["description"].casefold()
        ]
        usable = [
            iface for iface in candidates
            if iface["ip"] and not iface["ip"].startswith(("127.", "169.254."))
        ]
        if usable:
            return usable[0]
        if candidates:
            return candidates[0]

    for iface in interfaces:
        aliases = (
            iface["identifier"], iface["name"],
            iface["description"], iface["guid"],
        )
        if any(value.casefold() == alias.casefold() for alias in aliases if alias):
            return iface

    available = "\n".join(
        f"  {iface['name']} -> {iface['identifier']}"
        + (f" ({iface['description']})" if iface["description"] else "")
        for iface in interfaces
    ) or "  No Scapy/Npcap interfaces were discovered."
    raise PacketCaptureError(
        f"Invalid live interface {requested!r}. Use a Scapy interface name or "
        f"identifier, not a Windows adapter index. Available interfaces:\n{available}"
    )

# ─── Simulation Config ────────────────────────────────────────────────────────
NORMAL_HOSTS = [f"192.168.1.{i}" for i in range(2, 30)]
SERVER_IP    = "192.168.1.1"

SIMULATION_PROFILES = {
    "normal": {
        "sources": 1, "connections": 28, "packets": 8,
        "ports": (443,), "protocols": (0,),
        "request_size": (80, 1200), "response_size": (1200, 3500),
        "reply_every": 2, "interval": 0.005,
        "request_flags": {"ACK": True}, "response_flags": {"ACK": True},
    },
    "dos": {
        "sources": 1, "connections": 128, "packets": 1,
        "ports": (80,), "protocols": (0,),
        "request_size": (100, 200), "reply_every": 0, "interval": 0.015,
        "request_flags": {"SYN": True},
    },
    "ddos": {
        "sources": 16, "connections": 220, "packets": 1,
        "ports": (80,), "protocols": (0,),
        "request_size": (40, 70), "reply_every": 0, "interval": 0.008,
        "request_flags": {"SYN": True},
    },
    "port_scan": {
        "sources": 1, "connections": 18, "packets": 2,
        "ports": (21, 22, 23, 25, 53, 80, 110, 139, 443, 445, 3306, 3389,
                  5432, 5900, 8080, 8443, 27017, 31337),
        "protocols": (0,), "request_size": (40, 60), "response_size": (40, 60),
        "reply_every": 2, "interval": 0.04,
        "request_flags": {"SYN": True}, "response_flags": {"RST": True},
    },
    "vuln_scan": {
        "sources": 1, "connections": 48, "packets": 4,
        "ports": (21, 22, 23, 25, 80, 443, 445, 3306), "protocols": (0,),
        "request_size": (40, 180), "response_size": (80, 220),
        "reply_every": 2, "interval": 0.025,
        "request_flags": {"ACK": True}, "response_flags": {"RST": True},
    },
    "brute_force": {
        "sources": 1, "connections": 160, "packets": 5,
        "ports": (22,), "protocols": (0,),
        "request_size": (250, 1200), "response_size": (40, 300),
        "reply_every": 2, "interval": 0.002,
        "request_flags": {"ACK": True}, "response_flags": {"RST": True},
        "final_flags": {"FIN": True},
    },
    "exploit": {
        "sources": 1, "connections": 1, "packets": 6,
        "ports": (445,), "protocols": (0,),
        "request_size": (5000, 25000), "response_size": (2000, 6000),
        "reply_every": 3, "interval": 0.4,
        "request_flags": {"ACK": True, "URG": True},
        "response_flags": {"ACK": True},
        "final_flags": {"FIN": True},
    },
    "web_attack": {
        "sources": 1, "connections": 40, "packets": 4,
        "ports": (80,), "protocols": (0,),
        "request_size": (300, 1000), "response_size": (800, 2500),
        "reply_every": 2, "interval": 0.008,
        "request_flags": {"ACK": True}, "response_flags": {"ACK": True},
        "final_flags": {"FIN": True},
    },
    "infiltration": {
        "sources": 1, "connections": 1, "packets": 12,
        "ports": (31337,), "protocols": (0,),
        "request_size": (40, 120), "response_size": (40, 100),
        "reply_every": 4, "interval": 0.18,
        "request_flags": {"ACK": True}, "response_flags": {"ACK": True},
    },
    "exfiltration": {
        "sources": 1, "connections": 1, "packets": 8,
        "ports": (443,), "protocols": (0,),
        "destination": "198.51.100.20",
        "request_size": (50000, 200000), "response_size": (1000, 5000),
        "reply_every": 4, "interval": 0.08,
        "request_flags": {"ACK": True}, "response_flags": {"ACK": True},
    },
}


class PacketCapture:
    """Unified interface for live capture and traffic simulation."""

    def __init__(self, mode: str = "simulate",
                 interface: str = "auto",
                 callback: Callable | None = None):
        self.mode       = mode          # "live" | "simulate"
        self.interface  = interface
        self.callback   = callback      # fn(pkt_info: dict)
        self._running   = False
        self._thread    = None
        self._error     = None
        self.interface_name = interface
        self.stats      = {"total": 0, "attacks_injected": 0}

    # ─── Public API ───────────────────────────────────────────────────────────

    def start(self):
        if self.mode == "live":
            self._configure_live_interface()
        self._running = True
        if self.mode == "live":
            self._thread = threading.Thread(
                target=self._live_capture, daemon=True)
        else:
            self._thread = threading.Thread(
                target=self._simulate, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def error(self) -> PacketCaptureError | None:
        return self._error

    def set_callback(self, fn: Callable):
        self.callback = fn

    # ─── Live Capture (Scapy) ─────────────────────────────────────────────────

    def _configure_live_interface(self):
        interfaces = discover_interfaces()
        selected = select_interface(self.interface, interfaces)
        self.interface = selected["identifier"]
        self.interface_name = selected["name"]

    def _live_capture(self):
        try:
            from scapy.all import sniff, IP, TCP, UDP, ICMP
        except ImportError:
            self._error = PacketCaptureError(
                "Scapy is not installed. Install the project requirements before "
                "using --mode live."
            )
            self._running = False
            return

        def _process(pkt):
            if not self._running:
                return
            info = self._parse_scapy_pkt(pkt)
            if info:
                self.stats["total"] += 1
                if self.callback:
                    self.callback(info)

        try:
            while self._running:
                sniff(iface=self.interface, prn=_process,
                      store=False, timeout=1)
        except PermissionError as exc:
            self._error = PacketCaptureError(
                "Npcap denied live capture. Run the terminal with the privileges "
                "required by your Npcap installation."
            )
            print(f"[!] {self._error}")
            self._running = False
        except Exception as e:
            self._error = PacketCaptureError(
                f"Npcap/Scapy live capture failed on {self.interface!r}: {e}"
            )
            print(f"[!] {self._error}")
            self._running = False

    @staticmethod
    def _parse_scapy_pkt(pkt) -> dict | None:
        from scapy.all import IP, TCP, UDP, ICMP
        if not pkt.haslayer(IP):
            return None
        ip = pkt[IP]
        proto_map = {"tcp": 0, "udp": 1, "icmp": 2}
        proto = 0

        flags = {}
        sport, dport = 0, 0

        if pkt.haslayer(TCP):
            proto  = 0
            tcp    = pkt[TCP]
            sport  = tcp.sport
            dport  = tcp.dport
            f      = tcp.flags
            flags  = {
                "SYN": bool(f & 0x02),
                "ACK": bool(f & 0x10),
                "FIN": bool(f & 0x01),
                "RST": bool(f & 0x04),
                "URG": bool(f & 0x20),
            }
        elif pkt.haslayer(UDP):
            proto = 1
            udp   = pkt[UDP]
            sport = udp.sport
            dport = udp.dport
        elif pkt.haslayer(ICMP):
            proto = 2

        return {
            "src_ip":   ip.src,
            "dst_ip":   ip.dst,
            "src_port": sport,
            "dst_port": dport,
            "protocol": proto,
            "size":     len(pkt),
            "flags":    flags,
            "ts":       time.time(),
        }

    # ─── Traffic Simulation ───────────────────────────────────────────────────

    def _simulate(self):
        """
        Generates normal sessions and cycles through distinct attack behaviors.
        """
        next_attack = time.time() + random.uniform(4, 8)
        attack_types = [name for name in SIMULATION_PROFILES if name != "normal"]
        random.shuffle(attack_types)
        attack_index = 0

        while self._running:
            now = time.time()

            if now >= next_attack:
                attack_type = attack_types[attack_index]
                attack_index = (attack_index + 1) % len(attack_types)
                self._inject_attack(attack_type)
                next_attack = now + random.uniform(8, 15)

            for pkt in self._make_scenario_packets("normal"):
                if not self._running:
                    break
                pkt["ts"] = time.time()
                self.stats["total"] += 1
                if self.callback:
                    self.callback(pkt)
                time.sleep(SIMULATION_PROFILES["normal"]["interval"])

    def _inject_attack(self, attack_type: str):
        profile = SIMULATION_PROFILES[attack_type]

        def _burst():
            for pkt in self._make_scenario_packets(attack_type):
                if not self._running:
                    break
                pkt["ts"] = time.time()
                self.stats["total"] += 1
                self.stats["attacks_injected"] += 1
                if self.callback:
                    self.callback(pkt)
                time.sleep(profile["interval"])

        t = threading.Thread(target=_burst, daemon=True)
        t.start()
        return t

    @staticmethod
    def _make_scenario_packets(scenario: str) -> list[dict]:
        profile = SIMULATION_PROFILES[scenario]
        source_ips = set()
        while len(source_ips) < profile["sources"]:
            source_ips.add(
                random.choice(NORMAL_HOSTS)
                if scenario == "normal"
                else f"10.0.{random.randint(0, 255)}.{random.randint(1, 254)}"
            )
        source_ips = list(source_ips)
        packets = []

        for connection_index in range(profile["connections"]):
            source_ip = source_ips[connection_index % len(source_ips)]
            destination_ip = profile.get("destination", SERVER_IP)
            source_port = random.randint(1024, 65535)
            destination_port = profile["ports"][connection_index % len(profile["ports"])]
            protocol = random.choice(profile["protocols"])

            for packet_index in range(profile["packets"]):
                if scenario == "normal":
                    is_reply = packet_index == 1 or packet_index % 2 == 1
                    handshake_flags = (
                        {"SYN": True},
                        {"SYN": True, "ACK": True},
                        {"ACK": True},
                    )
                    flags = (
                        handshake_flags[packet_index]
                        if packet_index < len(handshake_flags)
                        else {"ACK": True}
                    )
                else:
                    is_reply = (
                        profile.get("reply_every", 0) > 0
                        and (packet_index + 1) % profile["reply_every"] == 0
                    )
                    flag_key = "response_flags" if is_reply else "request_flags"
                    flags = profile.get(flag_key, {}).copy()
                    if packet_index == profile["packets"] - 1:
                        flags.update(profile.get("final_flags", {}))
                src_ip, dst_ip = (destination_ip, source_ip) if is_reply else (source_ip, destination_ip)
                src_port, dst_port = (destination_port, source_port) if is_reply else (source_port, destination_port)
                size_range = profile.get("response_size") if is_reply else profile["request_size"]
                packets.append({
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "src_port": src_port,
                    "dst_port": dst_port,
                    "protocol": protocol,
                    "size": random.randint(*size_range),
                    "flags": flags,
                    "ts": time.time(),
                })

        return packets

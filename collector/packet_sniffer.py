"""Scapy-based live packet sniffer for network telemetry.

Captures packets on a configured network interface and extracts
structured telemetry records into an asyncio queue for downstream
processing.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional

from loguru import logger

from config.settings import get_settings

try:
    from scapy.all import AsyncSniffer, IP, TCP, UDP, ICMP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    logger.warning("scapy not available — PacketSniffer will not function")


class PacketSniffer:
    """Captures live network packets and extracts telemetry records.

    Each captured packet is parsed into a structured dictionary and placed
    onto an asyncio queue for consumption by the feature pipeline or Kafka
    producer.

    Attributes:
        settings: Application settings instance.
        queue: Async queue holding parsed packet records.
    """

    def __init__(self, queue: Optional[asyncio.Queue] = None) -> None:
        """Initialize the packet sniffer.

        Args:
            queue: Optional asyncio queue to push records to.
                   A new queue is created if not provided.
        """
        self.settings = get_settings()
        self.queue: asyncio.Queue = queue or asyncio.Queue(maxsize=10_000)
        self._sniffer: Optional[Any] = None
        self._running = False

    def _process_packet(self, packet: Any) -> None:
        """Parse a scapy packet into a telemetry record.

        Args:
            packet: Raw scapy packet object.
        """
        if not SCAPY_AVAILABLE or not packet.haslayer(IP):
            return

        ip_layer = packet[IP]

        record: Dict[str, Any] = {
            "timestamp": time.time(),
            "src_ip": ip_layer.src,
            "dst_ip": ip_layer.dst,
            "protocol": "UNKNOWN",
            "length": len(packet),
            "ttl": ip_layer.ttl,
            "tcp_flags": None,
            "src_port": None,
            "dst_port": None,
        }

        if packet.haslayer(TCP):
            tcp_layer = packet[TCP]
            record["protocol"] = "TCP"
            record["tcp_flags"] = str(tcp_layer.flags)
            record["src_port"] = tcp_layer.sport
            record["dst_port"] = tcp_layer.dport
        elif packet.haslayer(UDP):
            udp_layer = packet[UDP]
            record["protocol"] = "UDP"
            record["src_port"] = udp_layer.sport
            record["dst_port"] = udp_layer.dport
        elif packet.haslayer(ICMP):
            record["protocol"] = "ICMP"

        try:
            self.queue.put_nowait(record)
        except asyncio.QueueFull:
            logger.warning("Packet queue full — dropping oldest record")
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(record)
            except asyncio.QueueEmpty:
                pass

    def start(self) -> None:
        """Begin packet sniffing in a background thread.

        Raises:
            RuntimeError: If scapy is not available.
        """
        if not SCAPY_AVAILABLE:
            raise RuntimeError(
                "scapy is not installed — cannot start packet sniffer"
            )

        if self._running:
            logger.warning("Sniffer is already running")
            return

        logger.info(
            "Starting packet sniffer on interface={} filter='{}'",
            self.settings.NETWORK_INTERFACE,
            self.settings.CAPTURE_FILTER,
        )

        self._sniffer = AsyncSniffer(
            iface=self.settings.NETWORK_INTERFACE,
            filter=self.settings.CAPTURE_FILTER,
            prn=self._process_packet,
            store=False,
        )

        try:
            self._sniffer.start()
            self._running = True
            logger.info("Packet sniffer started")
        except OSError as exc:
            logger.error("Failed to start sniffer: {}", exc)
            self._running = False
            raise

    def stop(self) -> None:
        """Gracefully stop the packet sniffer."""
        if self._sniffer is not None and self._running:
            try:
                self._sniffer.stop()
                logger.info("Packet sniffer stopped")
            except Exception as exc:
                logger.error("Error stopping sniffer: {}", exc)
            finally:
                self._running = False
                self._sniffer = None

    @property
    def is_running(self) -> bool:
        """Return whether the sniffer is currently active."""
        return self._running

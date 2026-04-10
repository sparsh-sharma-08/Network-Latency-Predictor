"""ICMP echo prober for measuring network round-trip time.

Uses raw sockets (no scapy) to send ICMP echo requests and measure RTT.
Supports concurrent probing of multiple hosts with retry logic.
"""

from __future__ import annotations

import asyncio
import os
import socket
import struct
import time
from typing import Any, Dict, List, Optional

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import get_settings

# ICMP protocol constants
ICMP_ECHO_REQUEST = 8
ICMP_ECHO_REPLY = 0
ICMP_HEADER_FMT = "!BBHHH"
ICMP_HEADER_SIZE = struct.calcsize(ICMP_HEADER_FMT)
ICMP_PAYLOAD = b"NetworkLatencyPredictor" + b"\x00" * 9  # 32 bytes total


def _compute_checksum(data: bytes) -> int:
    """Compute the Internet checksum (RFC 1071) for ICMP packets.

    Args:
        data: Raw bytes to checksum.

    Returns:
        16-bit checksum value.
    """
    if len(data) % 2:
        data += b"\x00"

    total = 0
    for i in range(0, len(data), 2):
        word = (data[i] << 8) + data[i + 1]
        total += word

    total = (total >> 16) + (total & 0xFFFF)
    total += total >> 16
    return ~total & 0xFFFF


def _build_icmp_packet(identifier: int, sequence: int) -> bytes:
    """Build an ICMP echo request packet with correct checksum.

    Args:
        identifier: 16-bit identifier for the ICMP request.
        sequence: 16-bit sequence number.

    Returns:
        Complete ICMP packet bytes ready to send.
    """
    # Build header with zero checksum first
    header = struct.pack(
        ICMP_HEADER_FMT, ICMP_ECHO_REQUEST, 0, 0, identifier, sequence
    )
    # Compute checksum over header + payload
    checksum = _compute_checksum(header + ICMP_PAYLOAD)
    # Rebuild header with correct checksum
    header = struct.pack(
        ICMP_HEADER_FMT, ICMP_ECHO_REQUEST, 0, checksum, identifier, sequence
    )
    return header + ICMP_PAYLOAD


class ICMPProber:
    """Sends ICMP echo requests to target hosts and measures RTT.

    Attributes:
        settings: Application settings instance.
        timeout: Socket timeout in seconds.
    """

    def __init__(self, timeout: float = 2.0) -> None:
        """Initialize the ICMP prober.

        Args:
            timeout: Maximum seconds to wait for a reply.
        """
        self.settings = get_settings()
        self.timeout = timeout
        self._sequence = 0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        reraise=True,
    )
    def probe(self, host: str) -> Dict[str, Any]:
        """Send an ICMP echo request and measure the round-trip time.

        Args:
            host: Target host IP address or hostname.

        Returns:
            Dictionary with keys: timestamp, host, rtt_ms, ttl, packet_loss.
        """
        self._sequence = (self._sequence + 1) % 65536
        identifier = os.getpid() & 0xFFFF

        result: Dict[str, Any] = {
            "timestamp": time.time(),
            "host": host,
            "rtt_ms": None,
            "ttl": None,
            "packet_loss": True,
        }

        try:
            sock = socket.socket(
                socket.AF_INET, socket.SOCK_RAW, socket.getprotobyname("icmp")
            )
        except PermissionError:
            # Fall back to DGRAM ICMP (macOS / unprivileged)
            try:
                sock = socket.socket(
                    socket.AF_INET, socket.SOCK_DGRAM, socket.getprotobyname("icmp")
                )
            except OSError as exc:
                logger.warning(
                    "Cannot create ICMP socket (needs root/CAP_NET_RAW): {}", exc
                )
                return result

        try:
            sock.settimeout(self.timeout)
            packet = _build_icmp_packet(identifier, self._sequence)

            send_time = time.perf_counter()
            sock.sendto(packet, (host, 0))

            # Wait for reply
            while True:
                data, addr = sock.recvfrom(1024)
                recv_time = time.perf_counter()

                # Parse IP header to get TTL (byte index 8)
                if len(data) >= 20:
                    ttl = data[8]
                else:
                    ttl = None

                # ICMP reply starts after IP header (usually 20 bytes)
                icmp_offset = 20 if len(data) > ICMP_HEADER_SIZE + 20 else 0
                icmp_header = data[icmp_offset : icmp_offset + ICMP_HEADER_SIZE]
                if len(icmp_header) < ICMP_HEADER_SIZE:
                    continue

                icmp_type, _, _, recv_id, recv_seq = struct.unpack(
                    ICMP_HEADER_FMT, icmp_header
                )

                if icmp_type == ICMP_ECHO_REPLY and recv_seq == self._sequence:
                    rtt = (recv_time - send_time) * 1000.0  # ms
                    result["rtt_ms"] = round(rtt, 3)
                    result["ttl"] = ttl
                    result["packet_loss"] = False
                    break

        except socket.timeout:
            logger.debug("ICMP probe to {} timed out", host)
        except OSError as exc:
            logger.warning("ICMP probe to {} failed: {}", host, exc)
        finally:
            sock.close()

        result["timestamp"] = time.time()
        return result

    async def probe_async(self, host: str) -> Dict[str, Any]:
        """Run a probe in an executor for async compatibility.

        Args:
            host: Target host IP address.

        Returns:
            Probe result dictionary.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.probe, host)

    async def probe_all(self) -> List[Dict[str, Any]]:
        """Probe all configured hosts concurrently.

        Returns:
            List of probe result dictionaries, one per host.
        """
        tasks = [self.probe_async(host) for host in self.settings.PROBE_HOSTS]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output: List[Dict[str, Any]] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "Probe to {} raised exception: {}",
                    self.settings.PROBE_HOSTS[i],
                    result,
                )
                output.append(
                    {
                        "timestamp": time.time(),
                        "host": self.settings.PROBE_HOSTS[i],
                        "rtt_ms": None,
                        "ttl": None,
                        "packet_loss": True,
                    }
                )
            else:
                output.append(result)

        return output

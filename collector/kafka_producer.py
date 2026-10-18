"""Kafka producer for publishing network telemetry records.

Wraps kafka-python's KafkaProducer with JSON serialization, automatic
retries on transient failures, and context-manager support for graceful
shutdown.
"""

from __future__ import annotations

import json
import time
from types import TracebackType
from typing import Any, Dict, Optional, Type

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import get_settings

try:
    from kafka import KafkaProducer as _KafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    logger.warning("kafka-python not available — TelemetryProducer disabled")


class TelemetryProducer:
    """Publishes network telemetry and feature records to Kafka topics.

    Intended to be used as a context manager:

        async with TelemetryProducer() as producer:
            producer.publish_raw(record)

    Attributes:
        settings: Application settings.
    """

    def __init__(self) -> None:
        """Initialize the Kafka producer (connects lazily)."""
        self.settings = get_settings()
        self._producer: Optional[Any] = None

    def _ensure_producer(self) -> None:
        """Create the underlying KafkaProducer if not already connected."""
        if self._producer is not None:
            return

        if not KAFKA_AVAILABLE:
            raise RuntimeError("kafka-python is not installed")

        try:
            self._producer = _KafkaProducer(
                bootstrap_servers=self.settings.KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks="all",
                retries=3,
                max_in_flight_requests_per_connection=1,
            )
            logger.info(
                "Kafka producer connected to {}",
                self.settings.KAFKA_BOOTSTRAP_SERVERS,
            )
        except Exception as exc:
            logger.error("Failed to create Kafka producer: {}", exc)
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def publish_raw(self, record: Dict[str, Any]) -> None:
        """Publish a raw telemetry record to the raw topic.

        Args:
            record: Telemetry dictionary (timestamp, rtt_ms, etc.).
        """
        self._ensure_producer()
        try:
            self._producer.send(
                self.settings.KAFKA_TOPIC_RAW,
                key=str(record.get("timestamp", time.time())),
                value=record,
            )
            logger.debug("Published raw record to {}", self.settings.KAFKA_TOPIC_RAW)
        except Exception as exc:
            logger.error("Failed to publish raw record: {}", exc)
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def publish_features(self, features: Dict[str, Any]) -> None:
        """Publish enriched feature records to the features topic.

        Args:
            features: Feature dictionary from the feature pipeline.
        """
        self._ensure_producer()
        try:
            self._producer.send(
                self.settings.KAFKA_TOPIC_FEATURES,
                key=str(features.get("timestamp", time.time())),
                value=features,
            )
            logger.debug(
                "Published features to {}", self.settings.KAFKA_TOPIC_FEATURES
            )
        except Exception as exc:
            logger.error("Failed to publish features: {}", exc)
            raise

    def flush(self, timeout: float = 10.0) -> None:
        """Flush all buffered messages to Kafka.

        Args:
            timeout: Maximum seconds to wait for flush.
        """
        if self._producer is not None:
            self._producer.flush(timeout=timeout)

    def close(self) -> None:
        """Close the Kafka producer and release resources."""
        if self._producer is not None:
            try:
                self._producer.flush(timeout=5)
                self._producer.close(timeout=5)
                logger.info("Kafka producer closed")
            except Exception as exc:
                logger.error("Error closing Kafka producer: {}", exc)
            finally:
                self._producer = None

    def __enter__(self) -> "TelemetryProducer":
        """Enter context manager."""
        self._ensure_producer()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Exit context manager, closing the producer."""
        self.close()

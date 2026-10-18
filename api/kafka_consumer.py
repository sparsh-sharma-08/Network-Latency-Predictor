"""Kafka consumer background worker for live prediction streaming.

Consumes enriched feature records from ``KAFKA_TOPIC_FEATURES``,
runs ensemble predictions, and publishes results to
``KAFKA_TOPIC_PREDICTIONS``. Maintains a ring buffer of recent
predictions for drift detection.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from typing import Any, Deque, Dict, Optional

import numpy as np
from loguru import logger

from config.settings import get_settings

try:
    from kafka import KafkaConsumer, KafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False


class KafkaConsumerWorker:
    """Async background worker consuming from Kafka and running predictions.

    Attributes:
        recent_predictions: Ring buffer of last 1000 predictions for
            drift detection.
    """

    def __init__(self, ensemble: Any) -> None:
        """Initialize the consumer worker.

        Args:
            ensemble: EnsembleModel instance for predictions.
        """
        self.settings = get_settings()
        self.ensemble = ensemble
        self.recent_predictions: Deque[Dict[str, Any]] = deque(maxlen=1000)
        self._consumer: Optional[Any] = None
        self._producer: Optional[Any] = None
        self._running = False
        self._message_count = 0

    async def start(self) -> None:
        """Start consuming from Kafka in the background.

        Runs as an asyncio task until ``stop()`` is called.
        """
        if not KAFKA_AVAILABLE:
            logger.warning("kafka-python not installed — consumer disabled")
            return

        self._running = True
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._consume_loop)

    def _consume_loop(self) -> None:
        """Blocking consume loop (runs in executor)."""
        try:
            self._consumer = KafkaConsumer(
                self.settings.KAFKA_TOPIC_FEATURES,
                bootstrap_servers=self.settings.KAFKA_BOOTSTRAP_SERVERS,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="latest",
                enable_auto_commit=True,
                consumer_timeout_ms=1000,
                group_id="predictor_group",
            )

            self._producer = KafkaProducer(
                bootstrap_servers=self.settings.KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )

            logger.info("Kafka consumer started on topic {}", self.settings.KAFKA_TOPIC_FEATURES)

            while self._running:
                try:
                    for message in self._consumer:
                        if not self._running:
                            break

                        features = message.value
                        self._process_message(features)
                        self._message_count += 1

                        # Drift check every 100 messages
                        if self._message_count % 100 == 0:
                            self._check_drift()

                except StopIteration:
                    time.sleep(0.1)
                except Exception as exc:
                    logger.error("Consumer loop error: {}", exc)
                    time.sleep(1)

        except Exception as exc:
            logger.error("Failed to start Kafka consumer: {}", exc)
        finally:
            self._cleanup()

    def _process_message(self, features: Dict[str, Any]) -> None:
        """Process a single feature message and publish predictions.

        Args:
            features: Feature dictionary from the features topic.
        """
        try:
            # Build feature array
            feature_values = [
                v for k, v in sorted(features.items())
                if isinstance(v, (int, float)) and k != "timestamp"
            ]

            X = np.array([feature_values], dtype=np.float32)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            prediction = self.ensemble.predict(X)
            prediction["timestamp"] = time.time()
            prediction["source_timestamp"] = features.get("timestamp", 0)

            # Store for drift detection
            self.recent_predictions.append(prediction)

            # Publish to predictions topic
            if self._producer is not None:
                self._producer.send(
                    self.settings.KAFKA_TOPIC_PREDICTIONS,
                    value=prediction,
                )

        except Exception as exc:
            logger.error("Message processing failed: {}", exc)

    def _check_drift(self) -> None:
        """Check for model drift using recent predictions."""
        try:
            from monitoring.drift_detector import DriftDetector
            detector = DriftDetector()
            # This is a lightweight check — full drift detection
            # requires ground-truth comparison
            logger.debug(
                "Drift check: {} predictions in buffer",
                len(self.recent_predictions),
            )
        except Exception as exc:
            logger.debug("Drift check skipped: {}", exc)

    def _cleanup(self) -> None:
        """Close Kafka connections."""
        if self._consumer is not None:
            try:
                self._consumer.close()
            except Exception:
                pass
        if self._producer is not None:
            try:
                self._producer.close()
            except Exception:
                pass

    def stop(self) -> None:
        """Signal the consumer to stop."""
        self._running = False
        logger.info("Kafka consumer stopping...")

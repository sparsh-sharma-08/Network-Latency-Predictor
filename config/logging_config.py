"""Structured JSON logging configuration using loguru.

Call ``setup_logging()`` once at process start-up to configure loguru
with structured JSON output, rotation, and appropriate log levels.
"""

from __future__ import annotations

import sys

from loguru import logger


def setup_logging(level: str = "INFO", json_output: bool = True) -> None:
    """Configure loguru for structured logging.

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_output: If True, emit logs as JSON objects for machine parsing.
    """
    logger.remove()  # Remove default handler

    if json_output:
        logger.add(
            sys.stderr,
            level=level,
            serialize=True,  # JSON output
            backtrace=True,
            diagnose=False,  # Avoid leaking local vars in production
        )
    else:
        logger.add(
            sys.stderr,
            level=level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
                "<level>{message}</level>"
            ),
            backtrace=True,
            diagnose=True,
        )

    # File sink with rotation
    logger.add(
        "logs/network_predictor_{time}.log",
        level=level,
        rotation="50 MB",
        retention="7 days",
        compression="gz",
        serialize=json_output,
    )

    logger.info("Logging configured", level=level, json_output=json_output)

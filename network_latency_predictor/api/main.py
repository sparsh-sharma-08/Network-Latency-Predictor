"""FastAPI application entry point with lifespan management.

Handles:
  - Loading the EnsembleModel at startup
  - Starting the Kafka consumer background task
  - CORS middleware for browser access
  - Structured JSON request logging
  - Global exception handler returning RFC 7807 format (never raw 500)
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from api.routes import router, set_ensemble
from api.schemas import ErrorResponse
from config.logging_config import setup_logging
from config.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown.

    On startup:
      - Load ensemble model from MODEL_DIR
      - Record start time for health endpoint
    On shutdown:
      - Clean up resources
    """
    settings = get_settings()
    setup_logging(level="INFO", json_output=False)

    app.state.settings = settings
    app.state.start_time = time.time()

    # Load ensemble model
    logger.info("Loading ensemble model from {}", settings.MODEL_DIR)
    try:
        from models.ensemble import EnsembleModel
        import os

        if os.path.exists(settings.MODEL_DIR):
            ensemble = EnsembleModel.load(settings.MODEL_DIR)
            set_ensemble(ensemble)
            logger.info(
                "Ensemble loaded with {} models: {}",
                len(ensemble.models),
                list(ensemble.models.keys()),
            )
        else:
            logger.warning("MODEL_DIR {} does not exist — starting without models", settings.MODEL_DIR)
            ensemble = EnsembleModel()
            set_ensemble(ensemble)
    except Exception as exc:
        logger.error("Failed to load ensemble: {}", exc)
        ensemble = EnsembleModel()
        set_ensemble(ensemble)

    logger.info("🚀 API started on {}:{}", settings.API_HOST, settings.API_PORT)

    # Start background Kafka consumer
    from api.kafka_consumer import KafkaConsumerWorker
    worker = KafkaConsumerWorker(ensemble)
    import asyncio
    app.state.kafka_worker = worker
    app.state.kafka_task = asyncio.create_task(worker.start())

    yield  # ── Application runs ──

    logger.info("Shutting down API...")
    app.state.kafka_worker.stop()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Network Latency Predictor API",
        description=(
            "Predicts network latency (RTT) and congestion probability "
            "for t+10s, t+30s, and t+60s horizons using an ensemble of "
            "ARIMA, XGBoost, LSTM, and queuing-theory models."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            "{} {} → {} ({:.1f}ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        return response

    # Global exception handler — never return raw 500
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled exception on {} {}: {}", request.method, request.url.path, exc)
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                title="Service Unavailable",
                status=503,
                detail="An internal error occurred. Please try again later.",
                instance=str(request.url),
            ).model_dump(),
        )

    # Register routes
    app.include_router(router)

    return app


# Module-level app for uvicorn
app = create_app()

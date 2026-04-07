"""Health check endpoint."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Request
from qdrant_client import QdrantClient

from src.api.models import HealthResponse
from src.config import settings

router = APIRouter(tags=["Health"])
logger = structlog.get_logger(__name__)


@router.get("/health", response_model=HealthResponse, summary="Service health check")
async def health(request: Request) -> HealthResponse:
    """Check connectivity to all downstream services.

    Returns an overall status of ``"healthy"`` when all services are reachable,
    or ``"degraded"`` when at least one service is unavailable.
    """
    services: dict[str, str] = {}

    # Qdrant
    try:
        qdrant: QdrantClient = request.app.state.qdrant_client
        qdrant.get_collections()
        services["qdrant"] = "ok"
    except Exception as exc:
        services["qdrant"] = f"error: {exc}"
        logger.warning("health_qdrant_fail", error=str(exc))

    # Redis
    try:
        redis = request.app.state.redis_client
        redis.ping()
        services["redis"] = "ok"
    except Exception as exc:
        services["redis"] = f"error: {exc}"
        logger.warning("health_redis_fail", error=str(exc))

    overall = "healthy" if all(v == "ok" for v in services.values()) else "degraded"
    return HealthResponse(
        status=overall,
        version=settings.APP_VERSION,
        services=services,
    )

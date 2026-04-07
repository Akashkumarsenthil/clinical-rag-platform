"""Prometheus metrics exposition endpoint."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter(tags=["Monitoring"])


@router.get(
    "/metrics",
    response_class=PlainTextResponse,
    summary="Prometheus metrics",
    include_in_schema=False,
)
async def metrics() -> PlainTextResponse:
    """Expose all registered Prometheus metrics in text/plain format.

    The response follows the Prometheus exposition format and is intended to
    be scraped by a Prometheus server.
    """
    data = generate_latest()
    return PlainTextResponse(content=data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)

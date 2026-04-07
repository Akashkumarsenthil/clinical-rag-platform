"""Custom FastAPI middleware: request ID, latency logging, and rate limiting."""

from __future__ import annotations

import time
import uuid
from collections import defaultdict

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

logger = structlog.get_logger(__name__)

RATE_LIMIT_MAX = 100  # requests per window
RATE_LIMIT_WINDOW = 60  # seconds


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attaches a unique X-Request-ID header to every request and response.

    If the incoming request already carries an X-Request-ID header the value
    is preserved; otherwise a new UUID-4 is generated.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class LatencyMiddleware(BaseHTTPMiddleware):
    """Records wall-clock latency for every request and logs it with structlog."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        request_id = getattr(request.state, "request_id", "-")
        logger.info(
            "request_complete",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=round(elapsed_ms, 2),
            request_id=request_id,
        )
        response.headers["X-Latency-Ms"] = str(round(elapsed_ms, 2))
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token-bucket rate limiter: max 100 requests per minute per client IP.

    Uses an in-process counter dictionary.  In multi-worker deployments use
    Redis (see the Redis-backed variant in the integration tests).
    """

    def __init__(self, app, max_requests: int = RATE_LIMIT_MAX, window_seconds: int = RATE_LIMIT_WINDOW) -> None:
        super().__init__(app)
        self._max = max_requests
        self._window = window_seconds
        # {ip: (window_start, count)}
        self._counters: dict[str, tuple[float, int]] = defaultdict(lambda: (0.0, 0))

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start, count = self._counters[ip]

        if now - window_start > self._window:
            # Reset window
            self._counters[ip] = (now, 1)
        else:
            count += 1
            self._counters[ip] = (window_start, count)
            if count > self._max:
                logger.warning("rate_limit_exceeded", ip=ip, count=count)
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Too many requests. Please retry after 60 seconds."},
                )

        return await call_next(request)

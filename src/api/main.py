"""FastAPI application entry point with lifespan context manager."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import redis as redis_lib
import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient

from src.api.middleware import LatencyMiddleware, RateLimitMiddleware, RequestIDMiddleware
from src.api.routes import chat_ui, documents, health, ingest, metrics, query, search
from src.config import settings
from src.monitoring.tracer import setup_tracing
from src.retrieval.reranker import CrossEncoderReranker

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: initialise and tear down shared resources."""
    logger.info("startup_begin", environment=settings.ENVIRONMENT)

    # PostgreSQL — create tables if they don't exist
    from src.db.engine import engine
    from src.db.models import Base
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("postgres_tables_ready")
    except Exception as exc:
        logger.warning("postgres_init_failed", error=str(exc))

    # Qdrant
    qdrant_client = QdrantClient(url=settings.QDRANT_URL)
    app.state.qdrant_client = qdrant_client
    logger.info("qdrant_connected", url=settings.QDRANT_URL)

    # Qdrant payload indexes for filtered retrieval
    try:
        collection = settings.QDRANT_COLLECTION_NAME
        existing = {c.name for c in qdrant_client.get_collections().collections}
        if collection in existing:
            qdrant_client.create_payload_index(
                collection_name=collection,
                field_name="doc_id",
                field_schema="keyword",
            )
            qdrant_client.create_payload_index(
                collection_name=collection,
                field_name="file_hash",
                field_schema="keyword",
            )
            logger.info("qdrant_payload_indexes_ensured")
    except Exception as exc:
        logger.warning("qdrant_index_setup_failed", error=str(exc))

    # PDF upload directory
    from pathlib import Path
    Path(settings.PDF_STORAGE_DIR).mkdir(parents=True, exist_ok=True)

    # Redis
    redis_client = redis_lib.from_url(settings.REDIS_URL, decode_responses=True)
    app.state.redis_client = redis_client
    logger.info("redis_connected", url=settings.REDIS_URL)

    # Pre-load the CrossEncoder model so the first request is not slow
    reranker = CrossEncoderReranker()
    reranker._get_model()  # triggers model download/cache
    app.state.reranker = reranker
    logger.info("reranker_loaded")

    # OpenTelemetry (best-effort; don't fail startup if collector is absent)
    try:
        setup_tracing(service_name="clinical-rag-platform")
    except Exception as exc:
        logger.warning("tracing_setup_failed", error=str(exc))

    logger.info("startup_complete")
    yield

    # Shutdown
    logger.info("shutdown_begin")
    try:
        qdrant_client.close()
    except Exception:
        pass
    try:
        redis_client.close()
    except Exception:
        pass
    logger.info("shutdown_complete")


def create_app() -> FastAPI:
    """Factory function that creates and configures the FastAPI application."""
    app = FastAPI(
        title="Clinical RAG Platform",
        description=(
            "Production-grade retrieval-augmented generation for clinical documents. "
            "Combines dense + sparse hybrid retrieval with cross-encoder reranking "
            "and a LangGraph multi-step agent."
        ),
        version=settings.APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
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

    # Custom middleware (outermost first)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(LatencyMiddleware)
    app.add_middleware(RequestIDMiddleware)

    # Routers
    app.include_router(chat_ui.router)
    app.include_router(health.router)
    app.include_router(query.router, prefix="/api/v1")
    app.include_router(ingest.router, prefix="/api/v1")
    app.include_router(documents.router, prefix="/api/v1")
    app.include_router(search.router, prefix="/api/v1")
    app.include_router(metrics.router)

    return app


app = create_app()

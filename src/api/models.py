"""Pydantic request/response models for the Clinical RAG API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for the POST /query endpoint."""

    question: str = Field(..., min_length=1, description="The clinical question to answer.")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of source chunks to return.")
    session_id: str = Field(..., description="Unique identifier for the conversation session.")


class SourceDoc(BaseModel):
    """A single source document/chunk returned in a query response."""

    content: str = Field(..., description="Text content of the chunk.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Source metadata.")
    score: float = Field(..., description="Relevance score from the retriever/reranker.")


class QueryResponse(BaseModel):
    """Response body for the POST /query endpoint."""

    answer: str = Field(..., description="Generated answer.")
    sources: list[SourceDoc] = Field(default_factory=list, description="Supporting sources.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Estimated answer confidence.")
    latency_ms: float = Field(..., description="Total end-to-end latency in milliseconds.")
    request_id: str = Field(..., description="Unique request identifier (from X-Request-ID).")


class IngestRequest(BaseModel):
    """Request body for the POST /ingest endpoint (URL-based ingestion)."""

    url: Optional[str] = Field(default=None, description="Publicly accessible PDF URL to ingest.")
    chunk_strategy: str = Field(
        default="recursive",
        description="Chunking strategy: 'recursive', 'semantic', or 'sliding_window'.",
    )


class IngestResponse(BaseModel):
    """Response body for the POST /ingest endpoint."""

    doc_id: str = Field(..., description="Unique document identifier.")
    chunks_created: int = Field(..., description="Number of chunks inserted into the vector store.")
    status: str = Field(..., description="'success', 'skipped', or 'error'.")


class HealthResponse(BaseModel):
    """Response body for the GET /health endpoint."""

    status: str = Field(..., description="'healthy' or 'degraded'.")
    version: str = Field(..., description="Application version string.")
    services: dict[str, str] = Field(
        default_factory=dict,
        description="Per-service health status ('ok' or error message).",
    )

"""Pydantic request/response models for Phases 4–5 (document management + doc-scoped chat)."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    """Response returned after a document upload is accepted."""

    doc_id: str = Field(..., description="Deterministic document identifier (UUID-5 of file hash).")
    filename: str = Field(..., description="Original uploaded filename.")
    status: str = Field(..., description="Initial processing status (always 'QUEUED').")


class DocumentListItem(BaseModel):
    """Compact document summary used in list views."""

    doc_id: str
    filename: str
    status: str
    chunk_count: int
    uploaded_at: datetime


class DocumentDetail(BaseModel):
    """Full document details including extracted metadata and summary."""

    doc_id: str
    filename: str
    status: str
    chunk_count: int
    uploaded_at: datetime
    metadata: dict[str, Any] | None = None
    summary: str | None = None


class DocumentStatus(BaseModel):
    """Real-time processing progress for a document pipeline run."""

    doc_id: str
    status: str
    stage: str = Field(..., description="Current pipeline stage name.")
    stage_num: int = Field(..., ge=0, description="Current stage number (1-indexed, 0 if unknown).")
    total_stages: int = Field(default=3, description="Total pipeline stages.")
    chunks_embedded: int = Field(default=0, ge=0)
    total_chunks: int = Field(default=0, ge=0)
    percent: float = Field(..., ge=0.0, le=100.0, description="Overall completion percentage.")


class VectorInfo(BaseModel):
    """Summary of a single Qdrant vector point."""

    point_id: str
    chunk_index: int
    vector_dim: int
    vector_preview: list[float] = Field(
        ..., description="First 5 values of the embedding vector."
    )


class DocumentChatRequest(BaseModel):
    """Request body for doc-scoped chat."""

    question: str = Field(..., min_length=1, description="Question to ask about the document.")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of source chunks to consider.")


class DocumentChatResponse(BaseModel):
    """Response from doc-scoped chat."""

    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list, description="Source chunks with page numbers.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    latency_ms: float


class SearchResult(BaseModel):
    """A single document match from the metadata search endpoint."""

    doc_id: str
    filename: str
    status: str
    chunk_count: int
    patient_name: Optional[str] = None
    mrn: Optional[str] = None
    dob: Optional[date] = None
    document_type: Optional[str] = None

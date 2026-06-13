"""Document management and doc-scoped chat endpoints (Phase 4–5)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse
from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.agents.graph import build_graph
from src.agents.state import AgentState
from src.api.models_v2 import (
    DocumentChatRequest,
    DocumentChatResponse,
    DocumentDetail,
    DocumentListItem,
    DocumentStatus,
    DocumentUploadResponse,
    VectorInfo,
)
from src.config import settings
from src.db.engine import get_session
from src.db.models import (
    DocumentMetadata,
    DocumentRecord,
    DocumentStatus as DBDocumentStatus,
    DocumentSummary,
)  # noqa: F401 — DBDocumentStatus used in upload_document
from src.ingestion.pdf_loader import PDFLoader
from src.services.document_pipeline import DocumentPipeline

router = APIRouter(tags=["Documents"])
logger = structlog.get_logger(__name__)

_GRAPH = build_graph()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _run_document_pipeline(doc_id: str, storage_path: str) -> None:
    """Background task: run the 3-stage pipeline (metadata → summary → embed)."""
    pipeline = DocumentPipeline()
    pipeline.run(doc_id, storage_path)


def _compute_percent(stage: str, stage_num: int, chunks_embedded: int, total_chunks: int) -> float:
    """Derive an overall completion percentage from stage progress.

    Stage 1 complete = 33%, Stage 2 complete = 66%, Stage 3 progress is
    66 + (chunks_embedded / total_chunks * 34).
    """
    if stage_num <= 0:
        return 0.0
    if stage_num == 1:
        return 33.0
    if stage_num == 2:
        return 66.0
    # Stage 3 (embedding)
    if total_chunks <= 0:
        return 66.0
    return 66.0 + (min(chunks_embedded, total_chunks) / total_chunks) * 34.0


# ── POST /api/v1/documents ────────────────────────────────────────────────────


@router.post(
    "/documents",
    response_model=DocumentUploadResponse,
    status_code=202,
    summary="Upload a clinical PDF document",
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> DocumentUploadResponse:
    """Accept a multipart PDF upload, persist it, and queue pipeline processing.

    The document is identified by a deterministic UUID-5 derived from the file
    content hash, making the upload idempotent.  Re-uploading identical bytes
    resets the status to QUEUED and re-runs the pipeline.

    Args:
        background_tasks: FastAPI background task manager.
        file: Uploaded PDF file.

    Returns:
        DocumentUploadResponse with doc_id, filename, and initial status.

    Raises:
        HTTPException 400: If the uploaded file is empty.
        HTTPException 500: If the database or file-system write fails.
    """
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    doc_id, file_hash = PDFLoader.doc_id_from_bytes(data)
    filename = file.filename or "upload.pdf"

    storage_dir = Path(settings.PDF_STORAGE_DIR)
    storage_dir.mkdir(parents=True, exist_ok=True)
    storage_path = storage_dir / f"{doc_id}.pdf"
    storage_path.write_bytes(data)

    logger.info(
        "document_upload",
        doc_id=doc_id,
        filename=filename,
        bytes=len(data),
        file_hash=file_hash,
    )

    session = get_session()
    try:
        existing = session.get(DocumentRecord, doc_id)
        if existing is not None:
            existing.status = DBDocumentStatus.QUEUED
            existing.filename = filename
            existing.file_hash = file_hash
            existing.storage_path = str(storage_path)
            existing.error_msg = None
        else:
            record = DocumentRecord(
                doc_id=doc_id,
                filename=filename,
                storage_path=str(storage_path),
                file_hash=file_hash,
                status=DBDocumentStatus.QUEUED,
            )
            session.add(record)
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.error("document_upload_db_error", doc_id=doc_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Database error: {exc}") from exc
    finally:
        session.close()

    background_tasks.add_task(_run_document_pipeline, doc_id, str(storage_path))

    return DocumentUploadResponse(doc_id=doc_id, filename=filename, status="QUEUED")


# ── GET /api/v1/documents ─────────────────────────────────────────────────────


@router.get(
    "/documents",
    response_model=list[DocumentListItem],
    summary="List all documents",
)
async def list_documents() -> list[DocumentListItem]:
    """Return a summary list of all ingested documents.

    Includes each document's processing status, chunk count, and upload time.
    """
    session = get_session()
    try:
        records = session.query(DocumentRecord).order_by(DocumentRecord.uploaded_at.desc()).all()
        return [
            DocumentListItem(
                doc_id=r.doc_id,
                filename=r.filename,
                status=r.status.value,
                chunk_count=r.chunk_count,
                uploaded_at=r.uploaded_at,
            )
            for r in records
        ]
    finally:
        session.close()


# ── GET /api/v1/documents/{doc_id} ────────────────────────────────────────────


@router.get(
    "/documents/{doc_id}",
    response_model=DocumentDetail,
    summary="Get document details",
)
async def get_document(doc_id: str) -> DocumentDetail:
    """Return full details for a single document, including metadata and summary.

    Args:
        doc_id: The document's unique identifier.

    Raises:
        HTTPException 404: If no document with the given ID exists.
    """
    session = get_session()
    try:
        record = session.get(DocumentRecord, doc_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found.")

        meta_row = (
            session.query(DocumentMetadata)
            .filter(DocumentMetadata.doc_id == doc_id)
            .first()
        )
        summary_row = (
            session.query(DocumentSummary)
            .filter(DocumentSummary.doc_id == doc_id)
            .first()
        )

        metadata: dict[str, Any] | None = None
        if meta_row is not None:
            metadata = {
                "patient_name": meta_row.patient_name,
                "first_name": meta_row.first_name,
                "last_name": meta_row.last_name,
                "dob": str(meta_row.dob) if meta_row.dob else None,
                "age": meta_row.age,
                "sex": meta_row.sex,
                "mrn": meta_row.mrn,
                "encounter_date": str(meta_row.encounter_date) if meta_row.encounter_date else None,
                "provider": meta_row.provider,
                "document_type": meta_row.document_type,
            }

        return DocumentDetail(
            doc_id=record.doc_id,
            filename=record.filename,
            status=record.status.value,
            chunk_count=record.chunk_count,
            uploaded_at=record.uploaded_at,
            metadata=metadata,
            summary=summary_row.summary if summary_row else None,
        )
    finally:
        session.close()


# ── GET /api/v1/documents/{doc_id}/status ─────────────────────────────────────


@router.get(
    "/documents/{doc_id}/status",
    response_model=DocumentStatus,
    summary="Get document processing status",
)
async def get_document_status(request: Request, doc_id: str) -> DocumentStatus:
    """Read real-time processing progress from Redis.

    The pipeline publishes progress updates to ``doc_progress:{doc_id}`` as a
    JSON hash.  Percent is derived from stage completion.

    Args:
        request: FastAPI request (used to access Redis from app state).
        doc_id: The document's unique identifier.

    Raises:
        HTTPException 404: If no progress data or document record exists.
    """
    redis = request.app.state.redis_client
    key = f"doc_progress:{doc_id}"

    raw = redis.get(key)
    if raw is not None:
        progress = json.loads(raw)
        stage = progress.get("stage", "unknown")
        stage_num = int(progress.get("stage_num", 0))
        total_stages = int(progress.get("total_stages", 3))
        chunks_embedded = int(progress.get("chunks_embedded", 0))
        total_chunks = int(progress.get("total_chunks", 0))
        status = progress.get("status", "processing")

        percent = _compute_percent(stage, stage_num, chunks_embedded, total_chunks)

        return DocumentStatus(
            doc_id=doc_id,
            status=status,
            stage=stage,
            stage_num=stage_num,
            total_stages=total_stages,
            chunks_embedded=chunks_embedded,
            total_chunks=total_chunks,
            percent=round(percent, 1),
        )

    # Fallback: read status from Postgres
    session = get_session()
    try:
        record = session.get(DocumentRecord, doc_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found.")

        status_val = record.status.value
        stage = status_val
        stage_num = {"queued": 0, "extracting": 1, "summarizing": 2, "embedding": 3, "completed": 3, "failed": 0}.get(
            status_val, 0
        )
        percent = {"queued": 0.0, "extracting": 0.0, "summarizing": 33.0, "embedding": 66.0, "completed": 100.0, "failed": 0.0}.get(
            status_val, 0.0
        )

        return DocumentStatus(
            doc_id=doc_id,
            status=status_val,
            stage=stage,
            stage_num=stage_num,
            total_stages=3,
            chunks_embedded=record.chunk_count if status_val == "completed" else 0,
            total_chunks=record.chunk_count if status_val == "completed" else 0,
            percent=percent,
        )
    finally:
        session.close()


# ── GET /api/v1/documents/{doc_id}/file ───────────────────────────────────────


@router.get(
    "/documents/{doc_id}/file",
    summary="Download the original PDF",
    responses={200: {"content": {"application/pdf": {}}}},
)
async def get_document_file(doc_id: str) -> FileResponse:
    """Stream the original uploaded PDF file.

    Args:
        doc_id: The document's unique identifier.

    Raises:
        HTTPException 404: If the document or its file is not found on disk.
    """
    storage_path = Path(settings.PDF_STORAGE_DIR) / f"{doc_id}.pdf"

    if not storage_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF file for document {doc_id} not found.")

    return FileResponse(
        path=str(storage_path),
        media_type="application/pdf",
        filename=f"{doc_id}.pdf",
    )


# ── GET /api/v1/documents/{doc_id}/vectors ────────────────────────────────────


@router.get(
    "/documents/{doc_id}/vectors",
    response_model=list[VectorInfo],
    summary="List Qdrant vectors for a document",
)
async def get_document_vectors(request: Request, doc_id: str) -> list[VectorInfo]:
    """Scroll all Qdrant points belonging to a document.

    Returns a summary of each vector point including its dimension and a
    preview of the first 5 embedding values.

    Args:
        request: FastAPI request (used to access Qdrant client from app state).
        doc_id: The document's unique identifier.
    """
    qdrant = request.app.state.qdrant_client
    collection = settings.QDRANT_COLLECTION_NAME

    scroll_filter = Filter(
        must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
    )

    vectors: list[VectorInfo] = []
    offset = None

    while True:
        results, next_offset = qdrant.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=100,
            offset=offset,
            with_vectors=True,
            with_payload=True,
        )

        for point in results:
            vec = point.vector if isinstance(point.vector, list) else []
            vectors.append(
                VectorInfo(
                    point_id=str(point.id),
                    chunk_index=point.payload.get("chunk_index", 0) if point.payload else 0,
                    vector_dim=len(vec),
                    vector_preview=vec[:5],
                )
            )

        if next_offset is None:
            break
        offset = next_offset

    logger.info("document_vectors_fetched", doc_id=doc_id, count=len(vectors))
    return vectors


# ── POST /api/v1/documents/{doc_id}/chat ──────────────────────────────────────


@router.post(
    "/documents/{doc_id}/chat",
    response_model=DocumentChatResponse,
    summary="Chat with a specific document",
)
async def chat_with_document(doc_id: str, body: DocumentChatRequest) -> DocumentChatResponse:
    """Run the RAG agent scoped to a single document.

    The agent's retrieval step is filtered to only return chunks belonging to
    the specified document, enabling focused Q&A.

    Args:
        doc_id: The document to scope retrieval to.
        body: Chat request with question and optional top_k.

    Returns:
        DocumentChatResponse with answer, sources, confidence, and latency.

    Raises:
        HTTPException 404: If the document does not exist.
        HTTPException 500: If the agent raises an unhandled exception.
    """
    session = get_session()
    try:
        record = session.get(DocumentRecord, doc_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found.")
    finally:
        session.close()

    logger.info("document_chat_start", doc_id=doc_id, question=body.question[:80])
    start = time.perf_counter()

    try:
        initial_state: AgentState = {
            "question": body.question,
            "documents": [],
            "generation": None,
            "num_retries": 0,
            "confidence_score": 0.0,
            "session_id": f"doc-chat-{doc_id}",
            "query_rewritten": False,
            "doc_id_filter": doc_id,
        }

        final_state: AgentState = await _GRAPH.ainvoke(initial_state)

        elapsed_ms = (time.perf_counter() - start) * 1000

        sources = []
        for chunk in final_state["documents"][: body.top_k]:
            chunk_doc_id = chunk.metadata.get("doc_id")
            if chunk_doc_id != doc_id:
                logger.warning(
                    "doc_scoped_source_leak",
                    expected=doc_id,
                    got=chunk_doc_id,
                )
            sources.append(
                {
                    "content": chunk.content[:300],
                    "page_number": chunk.metadata.get("page_number"),
                    "score": round(chunk.score, 4),
                    "source": chunk.metadata.get("source"),
                    "doc_id": chunk_doc_id,
                }
            )

        confidence = min(max(final_state["confidence_score"], 0.0), 1.0)

        logger.info(
            "document_chat_done",
            doc_id=doc_id,
            latency_ms=round(elapsed_ms, 2),
            sources=len(sources),
            confidence=round(confidence, 3),
        )

        return DocumentChatResponse(
            answer=final_state["generation"] or "",
            sources=sources,
            confidence=confidence,
            latency_ms=round(elapsed_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("document_chat_error", doc_id=doc_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {exc}") from exc

"""Ingest endpoint: file upload or URL-based document ingestion."""

from __future__ import annotations

import tempfile
from pathlib import Path

import httpx
import structlog
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi import status as http_status

from src.api.models import IngestResponse
from src.ingestion.pipeline import IngestionPipeline, PipelineError
from src.monitoring.metrics import ingestion_counter

router = APIRouter(tags=["Ingestion"])
logger = structlog.get_logger(__name__)


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=http_status.HTTP_202_ACCEPTED,
    summary="Ingest a clinical PDF document",
)
async def ingest(
    request: Request,
    file: UploadFile | None = File(default=None),
    url: str | None = Form(default=None),
    chunk_strategy: str = Form(default="recursive"),
) -> IngestResponse:
    """Ingest a PDF document from a multipart file upload or a public URL.

    At least one of *file* or *url* must be provided.

    The pipeline is idempotent: re-ingesting the same file returns status
    ``"skipped"`` without duplicating data in the vector store.

    Args:
        request: FastAPI request (unused here; available for middleware).
        file: Uploaded PDF file (multipart/form-data).
        url: Publicly accessible URL of a PDF to download and ingest.
        chunk_strategy: Chunking strategy name (default ``"recursive"``).

    Returns:
        IngestResponse with doc_id, chunks_created, and status.

    Raises:
        HTTPException 400: If neither file nor url is provided, or the URL fetch fails.
        HTTPException 422: If the uploaded content is not a valid PDF.
        HTTPException 500: If the ingestion pipeline fails unexpectedly.
    """
    request_id = getattr(request.state, "request_id", "unknown")

    if file is None and url is None:
        raise HTTPException(
            status_code=400, detail="Provide either a file upload or a 'url' form field."
        )

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            if file is not None:
                tmp_path = Path(tmpdir) / (file.filename or "upload.pdf")
                content = await file.read()
                tmp_path.write_bytes(content)
                logger.info(
                    "ingest_file_upload",
                    filename=file.filename,
                    bytes=len(content),
                    request_id=request_id,
                )
            else:
                assert url is not None  # guarded above
                logger.info("ingest_url_fetch", url=url, request_id=request_id)
                async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
                    response = await client.get(url)
                    if response.status_code != 200:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Failed to download URL (HTTP {response.status_code}): {url}",
                        )
                filename = url.split("/")[-1].split("?")[0] or "download.pdf"
                tmp_path = Path(tmpdir) / filename
                tmp_path.write_bytes(response.content)

            pipeline = IngestionPipeline(chunk_strategy=chunk_strategy)
            result = pipeline.run(str(tmp_path))

        ingestion_counter.labels(status=result.status).inc()
        return IngestResponse(
            doc_id=result.doc_id,
            chunks_created=result.chunks_created,
            status=result.status,
        )

    except HTTPException:
        raise
    except PipelineError as exc:
        ingestion_counter.labels(status="error").inc()
        logger.error("ingest_pipeline_error", error=str(exc), request_id=request_id)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        ingestion_counter.labels(status="error").inc()
        logger.error("ingest_unexpected_error", error=str(exc), request_id=request_id)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

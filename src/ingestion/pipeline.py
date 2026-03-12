"""End-to-end ingestion pipeline: load → chunk → embed → upsert."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.config import settings
from src.ingestion.chunker import get_chunker
from src.ingestion.embedder import Embedder
from src.ingestion.pdf_loader import PDFLoader

logger = structlog.get_logger(__name__)

VECTOR_DIM = 384  # all-MiniLM-L6-v2 output dimension


class PipelineError(Exception):
    """Raised when the ingestion pipeline encounters an unrecoverable error."""


@dataclass
class IngestionResult:
    """Summary of a single pipeline run."""

    doc_id: str
    chunks_created: int
    skipped: bool
    status: str


class IngestionPipeline:
    """Orchestrates the full document ingestion workflow.

    The pipeline is **idempotent**: it computes the file's SHA-256 hash and
    checks whether a point with that payload hash already exists in Qdrant.
    If so, the pipeline returns immediately without re-processing.

    Parameters
    ----------
    qdrant_client:
        Pre-configured QdrantClient instance (injected for testability).
    embedder:
        Embedder instance; defaults to a new Embedder using settings.
    chunk_strategy:
        Name of the chunking strategy to use (default from settings).
    """

    def __init__(
        self,
        qdrant_client: QdrantClient | None = None,
        embedder: Embedder | None = None,
        chunk_strategy: str | None = None,
    ) -> None:
        self._qdrant = qdrant_client or QdrantClient(url=settings.QDRANT_URL)
        self._embedder = embedder or Embedder()
        self._strategy = chunk_strategy or settings.CHUNK_STRATEGY
        self._loader = PDFLoader()
        self._collection = settings.QDRANT_COLLECTION_NAME
        self._ensure_collection()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, source: str | Path) -> IngestionResult:
        """Run the full ingestion pipeline for a single PDF file.

        Args:
            source: Path to the PDF file on disk.

        Returns:
            IngestionResult with doc_id, chunks_created, and status.

        Raises:
            PipelineError: On unrecoverable errors.
        """
        source_path = Path(source).resolve()
        logger.info("pipeline_start", source=str(source_path), strategy=self._strategy)

        try:
            # Step 1: Load
            pages: list[Document] = self._loader.load(str(source_path))
        except Exception as exc:
            raise PipelineError(f"PDF loading failed for '{source_path}': {exc}") from exc

        if not pages:
            raise PipelineError(f"No content extracted from '{source_path}'")

        doc_id: str = pages[0].metadata["doc_id"]
        file_hash: str = pages[0].metadata["file_hash"]

        # Step 2: Idempotency check
        if self._already_ingested(file_hash):
            logger.info("pipeline_skip", doc_id=doc_id, reason="already_ingested")
            return IngestionResult(
                doc_id=doc_id,
                chunks_created=0,
                skipped=True,
                status="skipped",
            )

        # Step 3: Chunk
        try:
            chunker = get_chunker(self._strategy)
            chunks: list[Document] = chunker.chunk(pages)
        except Exception as exc:
            raise PipelineError(f"Chunking failed: {exc}") from exc

        if not chunks:
            raise PipelineError("Chunking produced zero chunks")

        # Step 4: Embed
        try:
            texts = [c.page_content for c in chunks]
            vectors = self._embedder.embed_documents(texts)
        except Exception as exc:
            raise PipelineError(f"Embedding failed: {exc}") from exc

        # Step 5: Upsert to Qdrant
        try:
            self._upsert(chunks, vectors, doc_id, file_hash)
        except Exception as exc:
            raise PipelineError(f"Qdrant upsert failed: {exc}") from exc

        logger.info(
            "pipeline_complete",
            doc_id=doc_id,
            chunks_created=len(chunks),
        )
        return IngestionResult(
            doc_id=doc_id,
            chunks_created=len(chunks),
            skipped=False,
            status="success",
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not already exist."""
        existing = {c.name for c in self._qdrant.get_collections().collections}
        if self._collection not in existing:
            self._qdrant.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )
            logger.info("qdrant_collection_created", collection=self._collection)

    def _already_ingested(self, file_hash: str) -> bool:
        """Return True if a point with this file_hash payload already exists."""
        try:
            results, _ = self._qdrant.scroll(
                collection_name=self._collection,
                scroll_filter={
                    "must": [
                        {"key": "file_hash", "match": {"value": file_hash}}
                    ]
                },
                limit=1,
            )
            return len(results) > 0
        except Exception:
            return False

    def _upsert(
        self,
        chunks: list[Document],
        vectors: list[list[float]],
        doc_id: str,
        file_hash: str,
    ) -> None:
        """Batch-upsert chunk vectors and payloads into Qdrant."""
        import hashlib
        import uuid

        points: list[PointStruct] = []
        for chunk, vector in zip(chunks, vectors):
            payload: dict[str, Any] = dict(chunk.metadata)
            payload["text"] = chunk.page_content
            payload["file_hash"] = file_hash
            payload["doc_id"] = doc_id

            # Deterministic point ID from doc_id + chunk_index
            point_id_str = f"{doc_id}-{chunk.metadata.get('chunk_index', 0)}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, point_id_str))

            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        self._qdrant.upsert(collection_name=self._collection, points=points)
        logger.info("qdrant_upsert", collection=self._collection, num_points=len(points))

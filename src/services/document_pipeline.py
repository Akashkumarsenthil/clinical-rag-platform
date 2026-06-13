"""Three-stage background pipeline: metadata extraction, summary, embedding.

Orchestrates per-document processing with Redis progress tracking and
Postgres status updates. The embedding stage is idempotent: it deletes
all existing Qdrant points for the doc_id before upserting fresh ones.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import structlog
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct

from src.config import settings
from src.db.engine import get_session
from src.db.models import (
    DocumentMetadata,
    DocumentRecord,
    DocumentStatus,
    DocumentSummary,
)
from src.ingestion.chunker import get_chunker
from src.ingestion.embedder import Embedder
from src.ingestion.pdf_loader import PDFLoader
from src.services.metadata_extractor import MetadataExtractor
from src.services.summary_generator import SummaryGenerator

logger = structlog.get_logger(__name__)


class DocumentPipeline:
    """Runs the 3-stage ingestion pipeline for a single document.

    Stages:
        1. Metadata extraction (LLM + regex)
        2. Summary generation (LLM)
        3. Embedding (chunk → embed → delete old points → upsert)
    """

    def __init__(self) -> None:
        self._extractor = MetadataExtractor()
        self._summarizer = SummaryGenerator()
        self._embedder = Embedder()
        self._loader = PDFLoader()
        self._qdrant = QdrantClient(url=settings.QDRANT_URL)
        self._collection = settings.QDRANT_COLLECTION_NAME

    def run(self, doc_id: str, storage_path: str) -> None:
        """Execute all three pipeline stages for a document.

        Updates Postgres status and Redis progress after each stage.
        On any failure, sets status=FAILED and records the error.
        """
        log = logger.bind(doc_id=doc_id)
        log.info("pipeline_3stage_start")

        pages = self._loader.load(storage_path)
        full_text = "\n\n".join(p.page_content for p in pages)

        try:
            # Stage 1: Metadata extraction
            self._update_status(doc_id, DocumentStatus.EXTRACTING)
            self._update_progress(doc_id, "extracting_metadata", 1)
            metadata = self._extractor.extract(full_text)
            self._save_metadata(doc_id, metadata)
            log.info("stage1_complete", fields=sum(1 for v in metadata.values() if v))

            # Stage 2: Summary generation
            self._update_status(doc_id, DocumentStatus.SUMMARIZING)
            self._update_progress(doc_id, "generating_summary", 2)
            summary = self._summarizer.generate(full_text)
            self._save_summary(doc_id, summary)
            log.info("stage2_complete", summary_len=len(summary))

            # Stage 3: Embedding (idempotent)
            self._update_status(doc_id, DocumentStatus.EMBEDDING)
            self._update_progress(doc_id, "embedding_chunks", 3)
            chunk_count = self._embed_document(doc_id, pages, log)

            # Done
            self._update_status(doc_id, DocumentStatus.COMPLETED, chunk_count=chunk_count)
            self._update_progress(doc_id, "completed", 3, done=True,
                                  chunks_embedded=chunk_count, total_chunks=chunk_count)
            log.info("pipeline_3stage_complete", chunks=chunk_count)

        except Exception as exc:
            log.error("pipeline_3stage_failed", error=str(exc))
            self._update_status(doc_id, DocumentStatus.FAILED, error_msg=str(exc)[:500])
            self._update_progress(doc_id, "failed", 0, error=str(exc)[:200])

    def _embed_document(self, doc_id: str, pages: list[Document], log: Any) -> int:
        """Delete old points, chunk, embed, and upsert. Returns chunk count."""
        # Delete existing points for idempotency
        self._qdrant.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
        )
        log.info("old_points_deleted", doc_id=doc_id)

        chunker = get_chunker(settings.CHUNK_STRATEGY)
        chunks: list[Document] = chunker.chunk(pages)
        total = len(chunks)

        if total == 0:
            return 0

        self._update_progress(doc_id, "embedding_chunks", 3,
                              chunks_embedded=0, total_chunks=total)

        texts = [c.page_content for c in chunks]
        vectors = self._embedder.embed_documents(texts)

        file_hash = pages[0].metadata.get("file_hash", "")

        points: list[PointStruct] = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            payload: dict[str, Any] = dict(chunk.metadata)
            payload["text"] = chunk.page_content
            payload["file_hash"] = file_hash
            payload["doc_id"] = doc_id

            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}-{chunk.metadata.get('chunk_index', i)}"))
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

            if (i + 1) % 50 == 0:
                self._update_progress(doc_id, "embedding_chunks", 3,
                                      chunks_embedded=i + 1, total_chunks=total)

        self._qdrant.upsert(collection_name=self._collection, points=points)
        log.info("qdrant_upsert_complete", points=len(points))

        from src.retrieval.sparse_index_manager import invalidate_sparse_corpus_index
        invalidate_sparse_corpus_index()

        return total

    # ── Status + progress helpers ──────────────────────────────────────

    @staticmethod
    def _update_status(
        doc_id: str,
        status: DocumentStatus,
        chunk_count: int | None = None,
        error_msg: str | None = None,
    ) -> None:
        session = get_session()
        try:
            record = session.get(DocumentRecord, doc_id)
            if record:
                record.status = status
                if chunk_count is not None:
                    record.chunk_count = chunk_count
                if error_msg is not None:
                    record.error_msg = error_msg
                session.commit()
        finally:
            session.close()

    @staticmethod
    def _update_progress(
        doc_id: str,
        stage: str,
        stage_num: int,
        done: bool = False,
        chunks_embedded: int = 0,
        total_chunks: int = 0,
        error: str | None = None,
    ) -> None:
        import redis as redis_lib
        try:
            r = redis_lib.from_url(settings.REDIS_URL, decode_responses=True)
            data = {
                "stage": stage,
                "stage_num": stage_num,
                "total_stages": 3,
                "chunks_embedded": chunks_embedded,
                "total_chunks": total_chunks,
                "status": "completed" if done else ("failed" if error else "processing"),
                "error": error,
            }
            r.setex(f"doc_progress:{doc_id}", 3600, json.dumps(data))
        except Exception:
            pass

    @staticmethod
    def _save_metadata(doc_id: str, metadata: dict[str, Any]) -> None:
        from dateutil.parser import parse as parse_date
        session = get_session()
        try:
            existing = session.query(DocumentMetadata).filter_by(doc_id=doc_id).first()
            if existing:
                session.delete(existing)
                session.flush()

            dob = None
            if metadata.get("dob"):
                try:
                    dob = parse_date(str(metadata["dob"])).date()
                except Exception:
                    pass

            encounter_date = None
            if metadata.get("encounter_date"):
                try:
                    encounter_date = parse_date(str(metadata["encounter_date"])).date()
                except Exception:
                    pass

            meta = DocumentMetadata(
                doc_id=doc_id,
                patient_name=metadata.get("patient_name"),
                first_name=metadata.get("first_name"),
                last_name=metadata.get("last_name"),
                dob=dob,
                age=metadata.get("age"),
                sex=metadata.get("sex"),
                mrn=metadata.get("mrn"),
                encounter_date=encounter_date,
                provider=metadata.get("provider"),
                document_type=metadata.get("document_type"),
                raw_json=metadata,
            )
            session.add(meta)
            session.commit()
        finally:
            session.close()

    @staticmethod
    def _save_summary(doc_id: str, summary: str) -> None:
        session = get_session()
        try:
            existing = session.query(DocumentSummary).filter_by(doc_id=doc_id).first()
            if existing:
                session.delete(existing)
                session.flush()

            s = DocumentSummary(
                doc_id=doc_id,
                summary=summary,
                model=settings.CHAT_MODEL,
            )
            session.add(s)
            session.commit()
        finally:
            session.close()

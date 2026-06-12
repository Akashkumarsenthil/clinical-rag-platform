"""Global BM25 index lifecycle: lazy build from Qdrant, invalidate on ingest."""

from __future__ import annotations

import structlog
from qdrant_client import QdrantClient

from src.config import settings
from src.retrieval.corpus_loader import load_chunks_from_qdrant
from src.retrieval.sparse_retriever import SparseRetriever

logger = structlog.get_logger(__name__)


class SparseIndexManager:
    """Manages the full-corpus BM25 index backed by Qdrant payloads."""

    def __init__(self) -> None:
        self._sparse = SparseRetriever()
        self._stale = True
        self._qdrant: QdrantClient | None = None

    @property
    def sparse(self) -> SparseRetriever:
        return self._sparse

    @property
    def is_built(self) -> bool:
        return not self._stale and self._sparse.is_indexed

    def invalidate(self) -> None:
        """Mark the global index stale so it is rebuilt on next retrieval."""
        self._stale = True
        logger.info("sparse_global_index_invalidated")

    def ensure_global_index(
        self,
        qdrant_client: QdrantClient | None = None,
    ) -> SparseRetriever:
        """Build the full-corpus BM25 index from Qdrant if stale."""
        if not self._stale and self._sparse.is_indexed:
            return self._sparse

        client = qdrant_client or self._qdrant or QdrantClient(url=settings.QDRANT_URL)
        self._qdrant = client

        chunks = load_chunks_from_qdrant(
            qdrant_client=client,
            collection_name=settings.QDRANT_COLLECTION_NAME,
        )
        self._sparse.build_index(chunks)
        self._stale = False
        logger.info("sparse_global_index_built", corpus_size=len(chunks))
        return self._sparse

    def build_doc_scoped_index(self, doc_id: str) -> SparseRetriever:
        """Build a BM25 index from only one document's Qdrant points."""
        client = self._qdrant or QdrantClient(url=settings.QDRANT_URL)
        chunks = load_chunks_from_qdrant(
            qdrant_client=client,
            collection_name=settings.QDRANT_COLLECTION_NAME,
            doc_id=doc_id,
        )
        scoped = SparseRetriever()
        scoped.build_index(chunks)
        logger.info("sparse_doc_index_built", doc_id=doc_id, corpus_size=len(chunks))
        return scoped


# Process-wide singleton
_index_manager = SparseIndexManager()


def get_index_manager() -> SparseIndexManager:
    return _index_manager


def invalidate_sparse_corpus_index() -> None:
    """Public hook: call after any Qdrant upsert/delete that changes the corpus."""
    _index_manager.invalidate()

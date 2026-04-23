"""Dense vector retrieval against Qdrant using cosine similarity."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog
from qdrant_client import QdrantClient

from src.config import settings
from src.ingestion.embedder import Embedder

logger = structlog.get_logger(__name__)


class RetrievalError(Exception):
    """Raised when a retrieval operation fails."""


@dataclass
class ScoredChunk:
    """A retrieved text chunk with its relevance score and source metadata."""

    content: str
    metadata: dict[str, Any]
    score: float


class DenseRetriever:
    """Retrieves documents from Qdrant using dense vector similarity search.

    Parameters
    ----------
    qdrant_client:
        Pre-configured QdrantClient (injected for testability).
    embedder:
        Embedder used to vectorise queries.
    collection_name:
        Qdrant collection to search (defaults to settings value).
    """

    def __init__(
        self,
        qdrant_client: QdrantClient | None = None,
        embedder: Embedder | None = None,
        collection_name: str | None = None,
    ) -> None:
        self._qdrant = qdrant_client or QdrantClient(url=settings.QDRANT_URL)
        self._embedder = embedder or Embedder()
        self._collection = collection_name or settings.QDRANT_COLLECTION_NAME

    def retrieve(self, query: str, top_k: int = 20, query_filter=None) -> list[ScoredChunk]:
        """Embed the query and perform cosine-similarity search in Qdrant.

        Args:
            query: The natural-language question or search string.
            top_k: Maximum number of results to return.
            query_filter: Optional Qdrant Filter to scope results (e.g. by doc_id).

        Returns:
            List of ScoredChunk objects sorted by descending similarity score.

        Raises:
            RetrievalError: If the Qdrant search fails.
        """
        logger.info("dense_retrieve", query=query[:80], top_k=top_k)

        try:
            vectors = self._embedder.embed_documents([query])
            query_vector = vectors[0]
        except Exception as exc:
            raise RetrievalError(f"Failed to embed query: {exc}") from exc

        try:
            results = self._qdrant.search(
                collection_name=self._collection,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
                query_filter=query_filter,
            )
        except Exception as exc:
            raise RetrievalError(f"Qdrant search failed: {exc}") from exc

        chunks = [
            ScoredChunk(
                content=hit.payload.get("text", ""),
                metadata={k: v for k, v in hit.payload.items() if k != "text"},
                score=hit.score,
            )
            for hit in results
        ]

        logger.info("dense_retrieve_done", results=len(chunks))
        return chunks

"""Load retrieval corpus chunks from Qdrant for BM25 indexing."""

from __future__ import annotations

from typing import Any

import structlog
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.config import settings
from src.retrieval.dense_retriever import ScoredChunk

logger = structlog.get_logger(__name__)


def load_chunks_from_qdrant(
    qdrant_client: QdrantClient | None = None,
    collection_name: str | None = None,
    doc_id: str | None = None,
) -> list[ScoredChunk]:
    """Scroll Qdrant and convert point payloads into ScoredChunk objects.

    Args:
        qdrant_client: Optional pre-configured client (defaults to settings URL).
        collection_name: Qdrant collection to read from.
        doc_id: When set, only load points belonging to this document.

    Returns:
        List of ScoredChunk objects with ``content`` from payload ``text`` and
        remaining payload fields in ``metadata`` (including ``doc_id``).
    """
    client = qdrant_client or QdrantClient(url=settings.QDRANT_URL)
    collection = collection_name or settings.QDRANT_COLLECTION_NAME

    scroll_filter: Filter | None = None
    if doc_id is not None:
        scroll_filter = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        )

    chunks: list[ScoredChunk] = []
    offset = None

    while True:
        results, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in results:
            payload: dict[str, Any] = dict(point.payload or {})
            text = payload.pop("text", "")
            if not text:
                continue
            chunks.append(
                ScoredChunk(
                    content=text,
                    metadata=payload,
                    score=0.0,
                )
            )

        if next_offset is None:
            break
        offset = next_offset

    logger.info(
        "corpus_loaded_from_qdrant",
        collection=collection,
        doc_id=doc_id,
        chunks=len(chunks),
    )
    return chunks

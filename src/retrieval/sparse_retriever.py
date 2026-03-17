"""BM25-based sparse retrieval over an in-memory corpus."""

from __future__ import annotations

import re
import structlog
from rank_bm25 import BM25Okapi

from src.retrieval.dense_retriever import RetrievalError, ScoredChunk

logger = structlog.get_logger(__name__)

STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "not", "no", "nor",
        "so", "yet", "both", "either", "neither", "as", "if", "then", "than",
        "that", "this", "these", "those", "it", "its", "we", "our", "you",
        "your", "he", "his", "she", "her", "they", "their", "up", "out",
        "about", "into", "through", "during", "before", "after", "above",
    }
)


class SparseRetriever:
    """Retrieves documents using BM25 over an in-memory corpus.

    The corpus must be loaded via :meth:`build_index` before any call to
    :meth:`retrieve`.  The index is rebuilt in O(n) time and kept in memory;
    suitable for corpora up to ~100 k chunks.
    """

    def __init__(self) -> None:
        self._corpus_chunks: list[ScoredChunk] = []
        self._bm25: BM25Okapi | None = None

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def build_index(self, chunks: list[ScoredChunk]) -> None:
        """Build (or rebuild) the BM25 index from a list of ScoredChunk objects.

        Args:
            chunks: All available chunks; used as the BM25 corpus.
        """
        if not chunks:
            logger.warning("sparse_build_index_empty")
            self._corpus_chunks = []
            self._bm25 = None
            return

        self._corpus_chunks = chunks
        tokenized = [self._tokenize(c.content) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("sparse_index_built", corpus_size=len(chunks))

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 20) -> list[ScoredChunk]:
        """Score all corpus documents against the query and return top-k.

        Args:
            query: The natural-language query string.
            top_k: Number of top-scoring chunks to return.

        Returns:
            List of ScoredChunk objects, sorted by descending BM25 score.

        Raises:
            RetrievalError: If the index has not been built yet.
        """
        if self._bm25 is None or not self._corpus_chunks:
            raise RetrievalError(
                "SparseRetriever index is empty. Call build_index() before retrieve()."
            )

        logger.info("sparse_retrieve", query=query[:80], top_k=top_k)

        tokens = self._tokenize(query)
        scores: list[float] = self._bm25.get_scores(tokens).tolist()

        # Pair scores with chunks and sort descending
        ranked = sorted(
            zip(scores, self._corpus_chunks), key=lambda x: x[0], reverse=True
        )
        top = ranked[:top_k]

        results = [
            ScoredChunk(
                content=chunk.content,
                metadata=dict(chunk.metadata),
                score=score,
            )
            for score, chunk in top
        ]

        logger.info("sparse_retrieve_done", results=len(results))
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase, strip punctuation, remove stopwords, and split on whitespace."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = [t for t in text.split() if t and t not in STOPWORDS]
        return tokens

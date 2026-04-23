"""Hybrid retrieval combining dense and sparse results via Reciprocal Rank Fusion."""

from __future__ import annotations

import structlog

from src.retrieval.dense_retriever import DenseRetriever, RetrievalError, ScoredChunk
from src.retrieval.sparse_retriever import SparseRetriever

logger = structlog.get_logger(__name__)

RRF_K = 60  # standard RRF constant


class HybridRetriever:
    """Fuses dense and sparse retrieval results using Reciprocal Rank Fusion (RRF).

    RRF score for document *d*:

        score(d) = Σ  1 / (k + rank(d, list_i))

    where the sum is over all ranked lists (dense + sparse) that contain *d*,
    and *k* = 60 is the standard smoothing constant.

    Parameters
    ----------
    dense_retriever:
        DenseRetriever instance.
    sparse_retriever:
        SparseRetriever instance (must have index pre-built).
    candidate_k:
        Number of candidates to fetch from each sub-retriever (default 20).
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever | None = None,
        sparse_retriever: SparseRetriever | None = None,
        candidate_k: int = 20,
    ) -> None:
        self._dense = dense_retriever or DenseRetriever()
        self._sparse = sparse_retriever or SparseRetriever()
        self._candidate_k = candidate_k

    def retrieve(self, query: str, top_k: int = 10, query_filter=None) -> list[ScoredChunk]:
        """Retrieve and fuse results from dense and sparse retrievers.

        Args:
            query: Natural-language question or search string.
            top_k: Number of final fused results to return.
            query_filter: Optional Qdrant Filter to scope dense retrieval (e.g. by doc_id).

        Returns:
            List of ScoredChunk objects sorted by descending RRF score.

        Raises:
            RetrievalError: If both sub-retrievers fail simultaneously.
        """
        logger.info("hybrid_retrieve", query=query[:80], top_k=top_k)

        dense_results: list[ScoredChunk] = []
        sparse_results: list[ScoredChunk] = []

        try:
            dense_results = self._dense.retrieve(query, top_k=self._candidate_k, query_filter=query_filter)
        except Exception as exc:
            logger.warning("dense_retrieval_failed", error=str(exc))

        try:
            sparse_results = self._sparse.retrieve(query, top_k=self._candidate_k)
        except Exception as exc:
            logger.warning("sparse_retrieval_failed", error=str(exc))

        if not dense_results and not sparse_results:
            raise RetrievalError("Both dense and sparse retrievers returned no results.")

        fused = self._rrf_fuse(dense_results, sparse_results, top_k=top_k)
        logger.info("hybrid_retrieve_done", fused_results=len(fused))
        return fused

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rrf_fuse(
        dense: list[ScoredChunk],
        sparse: list[ScoredChunk],
        top_k: int,
    ) -> list[ScoredChunk]:
        """Apply RRF over two ranked lists and return the top-k fused chunks."""
        # Use chunk content as the deduplication key
        rrf_scores: dict[str, float] = {}
        # Map content key → representative ScoredChunk (from dense if available)
        chunk_map: dict[str, ScoredChunk] = {}

        for rank, chunk in enumerate(dense, start=1):
            key = chunk.content
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            chunk_map[key] = chunk

        for rank, chunk in enumerate(sparse, start=1):
            key = chunk.content
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            if key not in chunk_map:
                chunk_map[key] = chunk

        sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
        results: list[ScoredChunk] = []
        for key in sorted_keys[:top_k]:
            original = chunk_map[key]
            results.append(
                ScoredChunk(
                    content=original.content,
                    metadata=dict(original.metadata),
                    score=rrf_scores[key],
                )
            )
        return results

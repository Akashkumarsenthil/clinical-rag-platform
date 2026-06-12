"""Hybrid retrieval combining dense and sparse results via Reciprocal Rank Fusion."""

from __future__ import annotations

import structlog

from src.retrieval.dense_retriever import DenseRetriever, RetrievalError, ScoredChunk
from src.retrieval.sparse_index_manager import get_index_manager
from src.retrieval.sparse_retriever import SparseRetriever

logger = structlog.get_logger(__name__)

RRF_K = 60  # standard RRF constant


class HybridRetriever:
    """Fuses dense and sparse retrieval results using Reciprocal Rank Fusion (RRF).

    RRF score for document *d*:

        score(d) = Σ  1 / (k + rank(d, list_i))

    where the sum is over all ranked lists (dense + sparse) that contain *d*,
    and *k* = 60 is the standard smoothing constant.

    The sparse (BM25) corpus is loaded from Qdrant:
    - Global ``/query``: full collection, built lazily and refreshed after ingest.
    - Doc-scoped chat: BM25 index built from only the target doc_id's points.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever | None = None,
        sparse_retriever: SparseRetriever | None = None,
        candidate_k: int = 20,
    ) -> None:
        self._dense = dense_retriever or DenseRetriever()
        # Optional injected sparse retriever (used in tests); otherwise use index manager.
        self._sparse_override = sparse_retriever
        self._candidate_k = candidate_k
        self._index_manager = get_index_manager()

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        query_filter=None,
        doc_id_filter: str | None = None,
    ) -> list[ScoredChunk]:
        """Retrieve and fuse results from dense and sparse retrievers.

        Args:
            query: Natural-language question or search string.
            top_k: Number of final fused results to return.
            query_filter: Optional Qdrant Filter to scope dense retrieval (e.g. by doc_id).
            doc_id_filter: When set, BM25 is indexed from only this document's chunks.

        Returns:
            List of ScoredChunk objects sorted by descending RRF score.

        Raises:
            RetrievalError: If both sub-retrievers fail simultaneously.
        """
        logger.info(
            "hybrid_retrieve",
            query=query[:80],
            top_k=top_k,
            doc_id_filter=doc_id_filter,
        )

        dense_results: list[ScoredChunk] = []
        sparse_results: list[ScoredChunk] = []

        try:
            dense_results = self._dense.retrieve(
                query,
                top_k=self._candidate_k,
                query_filter=query_filter,
            )
        except Exception as exc:
            logger.warning("dense_retrieval_failed", error=str(exc))

        sparse_retriever = self._resolve_sparse_retriever(doc_id_filter)

        try:
            sparse_results = sparse_retriever.retrieve(query, top_k=self._candidate_k)
            logger.info(
                "sparse_contribution",
                doc_id_filter=doc_id_filter,
                results=len(sparse_results),
            )
        except Exception as exc:
            logger.warning("sparse_retrieval_failed", error=str(exc))

        if doc_id_filter:
            dense_results = self._filter_by_doc_id(dense_results, doc_id_filter)
            sparse_results = self._filter_by_doc_id(sparse_results, doc_id_filter)

        if not dense_results and not sparse_results:
            raise RetrievalError("Both dense and sparse retrievers returned no results.")

        fused = self._rrf_fuse(dense_results, sparse_results, top_k=top_k)
        logger.info(
            "hybrid_retrieve_done",
            fused_results=len(fused),
            dense_candidates=len(dense_results),
            sparse_candidates=len(sparse_results),
        )
        return fused

    def _resolve_sparse_retriever(self, doc_id_filter: str | None) -> SparseRetriever:
        """Return the BM25 retriever for this query (global or doc-scoped)."""
        if self._sparse_override is not None:
            return self._sparse_override

        if doc_id_filter:
            return self._index_manager.build_doc_scoped_index(doc_id_filter)

        return self._index_manager.ensure_global_index(
            qdrant_client=self._dense._qdrant,  # noqa: SLF001 — reuse connected client
        )

    @staticmethod
    def _filter_by_doc_id(
        chunks: list[ScoredChunk],
        doc_id: str,
    ) -> list[ScoredChunk]:
        """Safety filter: keep only chunks belonging to the requested document."""
        return [c for c in chunks if c.metadata.get("doc_id") == doc_id]

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
        rrf_scores: dict[str, float] = {}
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

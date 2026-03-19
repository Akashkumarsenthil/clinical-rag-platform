"""Cross-encoder reranker using a MiniLM model from sentence-transformers."""

from __future__ import annotations

from typing import Any

import structlog

from src.retrieval.dense_retriever import RetrievalError, ScoredChunk

logger = structlog.get_logger(__name__)

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankError(Exception):
    """Raised when reranking fails."""


class CrossEncoderReranker:
    """Reranks retrieved chunks using a CrossEncoder model.

    The model scores each (query, chunk) pair independently and returns the
    top-k chunks sorted by descending cross-encoder score.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier (defaults to ms-marco-MiniLM-L-6-v2).
    """

    def __init__(self, model_name: str = MODEL_NAME, model: Any = None) -> None:
        self._model_name = model_name
        self._model = model  # injected for testing; loaded lazily otherwise

    def _get_model(self) -> Any:
        """Lazily load the CrossEncoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415

            logger.info("loading_reranker_model", model=self._model_name)
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[ScoredChunk],
        top_k: int = 5,
    ) -> list[ScoredChunk]:
        """Score all (query, chunk) pairs and return top_k by cross-encoder score.

        Args:
            query: The question/search string.
            chunks: Candidate chunks to rerank (typically 10–20).
            top_k: Number of final results to return.

        Returns:
            A list of ScoredChunk objects with updated scores, sorted descending.

        Raises:
            RerankError: If the model fails to score the pairs.
        """
        if not chunks:
            return []

        top_k = min(top_k, len(chunks))
        model = self._get_model()

        logger.info("reranking", query=query[:80], candidates=len(chunks), top_k=top_k)

        pairs = [[query, chunk.content] for chunk in chunks]

        try:
            scores: list[float] = model.predict(pairs).tolist()
        except Exception as exc:
            raise RerankError(f"CrossEncoder prediction failed: {exc}") from exc

        ranked = sorted(
            zip(scores, chunks), key=lambda x: x[0], reverse=True
        )

        results = [
            ScoredChunk(
                content=chunk.content,
                metadata=dict(chunk.metadata),
                score=score,
            )
            for score, chunk in ranked[:top_k]
        ]

        logger.info("reranking_done", returned=len(results))
        return results

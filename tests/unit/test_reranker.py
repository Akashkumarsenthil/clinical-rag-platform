"""Unit tests for CrossEncoderReranker."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.retrieval.dense_retriever import ScoredChunk
from src.retrieval.reranker import CrossEncoderReranker, RerankError


def make_chunk(content: str, score: float = 0.5) -> ScoredChunk:
    return ScoredChunk(content=content, metadata={"source": "test.pdf"}, score=score)


def make_mock_model(scores: list[float]) -> MagicMock:
    mock = MagicMock()
    mock.predict.return_value = np.array(scores)
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCrossEncoderReranker:
    def test_output_sorted_by_score_descending(self):
        scores = [0.3, 0.9, 0.1, 0.7, 0.5]
        chunks = [make_chunk(f"chunk_{i}") for i in range(5)]
        reranker = CrossEncoderReranker(model=make_mock_model(scores))

        results = reranker.rerank("test query", chunks, top_k=5)

        assert len(results) == 5
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_top_k_limits_output(self):
        scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        chunks = [make_chunk(f"chunk_{i}") for i in range(5)]
        reranker = CrossEncoderReranker(model=make_mock_model(scores))

        results = reranker.rerank("query", chunks, top_k=3)
        assert len(results) == 3

    def test_top_k_larger_than_chunks_returns_all(self):
        scores = [0.4, 0.6]
        chunks = [make_chunk("a"), make_chunk("b")]
        reranker = CrossEncoderReranker(model=make_mock_model(scores))

        results = reranker.rerank("query", chunks, top_k=100)
        assert len(results) == 2

    def test_single_chunk_returns_single_result(self):
        reranker = CrossEncoderReranker(model=make_mock_model([0.77]))
        chunks = [make_chunk("Only one chunk here.")]

        results = reranker.rerank("query", chunks, top_k=5)
        assert len(results) == 1
        assert abs(results[0].score - 0.77) < 1e-6

    def test_empty_chunks_returns_empty(self):
        reranker = CrossEncoderReranker(model=MagicMock())
        results = reranker.rerank("query", [], top_k=5)
        assert results == []

    def test_model_prediction_error_raises_rerank_error(self):
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("CUDA OOM")
        reranker = CrossEncoderReranker(model=mock_model)
        chunks = [make_chunk("some text")]

        with pytest.raises(RerankError):
            reranker.rerank("query", chunks, top_k=1)

    def test_scores_are_updated_in_output(self):
        """Output ScoredChunk.score should reflect the cross-encoder score, not the original."""
        original_score = 0.1
        cross_encoder_score = 0.95
        chunks = [make_chunk("doc", score=original_score)]
        reranker = CrossEncoderReranker(model=make_mock_model([cross_encoder_score]))

        results = reranker.rerank("query", chunks, top_k=1)
        assert abs(results[0].score - cross_encoder_score) < 1e-6

    def test_correct_pairs_sent_to_model(self):
        mock_model = make_mock_model([0.5, 0.6])
        reranker = CrossEncoderReranker(model=mock_model)
        query = "What is the treatment for hypertension?"
        chunks = [make_chunk("Lisinopril"), make_chunk("Amlodipine")]

        reranker.rerank(query, chunks, top_k=2)

        called_pairs = mock_model.predict.call_args[0][0]
        assert called_pairs[0] == [query, "Lisinopril"]
        assert called_pairs[1] == [query, "Amlodipine"]

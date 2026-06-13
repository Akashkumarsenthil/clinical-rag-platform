"""Unit tests for HybridRetriever RRF fusion logic."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.retrieval.dense_retriever import RetrievalError, ScoredChunk
from src.retrieval.hybrid_retriever import HybridRetriever, RRF_K
from src.retrieval.sparse_retriever import SparseRetriever


def make_chunk(content: str, score: float = 0.5) -> ScoredChunk:
    return ScoredChunk(content=content, metadata={"source": "test.pdf"}, score=score)


def make_mock_retriever(chunks: list[ScoredChunk]) -> MagicMock:
    mock = MagicMock()
    mock.retrieve.return_value = chunks
    return mock


# ---------------------------------------------------------------------------
# RRF fusion tests
# ---------------------------------------------------------------------------


class TestRRFFusion:
    def test_rrf_score_formula(self):
        """Items ranked higher should receive higher RRF scores."""
        dense_chunks = [make_chunk(f"doc_{i}", score=1.0 - i * 0.1) for i in range(5)]
        sparse_chunks = []

        dense_mock = make_mock_retriever(dense_chunks)
        sparse_mock = make_mock_retriever(sparse_chunks)

        retriever = HybridRetriever(
            dense_retriever=dense_mock, sparse_retriever=sparse_mock
        )
        results = retriever.retrieve("test query", top_k=5)

        # doc_0 ranked #1 in dense should appear first
        assert results[0].content == "doc_0"
        # Scores should be strictly decreasing
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_rrf_fuse_equal_rank_identical_content(self):
        """A document appearing in both lists at the same rank gets a doubled RRF score."""
        shared_chunk = make_chunk("shared_document")
        dense_chunks = [shared_chunk]
        sparse_chunks = [make_chunk("shared_document")]  # same content, different object

        dense_mock = make_mock_retriever(dense_chunks)
        sparse_mock = make_mock_retriever(sparse_chunks)

        retriever = HybridRetriever(
            dense_retriever=dense_mock, sparse_retriever=sparse_mock
        )
        results = retriever.retrieve("test", top_k=5)

        assert len(results) == 1
        # Score should be 2 * (1 / (RRF_K + 1))
        expected = 2.0 / (RRF_K + 1)
        assert abs(results[0].score - expected) < 1e-9

    def test_empty_dense_returns_sparse_results(self):
        sparse_chunks = [make_chunk("sparse_only_doc", score=0.8)]
        dense_mock = make_mock_retriever([])
        sparse_mock = make_mock_retriever(sparse_chunks)

        retriever = HybridRetriever(
            dense_retriever=dense_mock, sparse_retriever=sparse_mock
        )
        results = retriever.retrieve("query", top_k=5)
        assert len(results) == 1
        assert results[0].content == "sparse_only_doc"

    def test_empty_sparse_returns_dense_results(self):
        dense_chunks = [make_chunk("dense_only_doc", score=0.9)]
        dense_mock = make_mock_retriever(dense_chunks)
        sparse_mock = make_mock_retriever([])

        retriever = HybridRetriever(
            dense_retriever=dense_mock, sparse_retriever=sparse_mock
        )
        results = retriever.retrieve("query", top_k=5)
        assert len(results) == 1
        assert results[0].content == "dense_only_doc"

    def test_both_empty_raises_retrieval_error(self):
        dense_mock = make_mock_retriever([])
        sparse_mock = make_mock_retriever([])

        retriever = HybridRetriever(
            dense_retriever=dense_mock, sparse_retriever=sparse_mock
        )
        with pytest.raises(RetrievalError):
            retriever.retrieve("query")

    def test_top_k_limits_results(self):
        dense_chunks = [make_chunk(f"doc_{i}") for i in range(15)]
        sparse_chunks = [make_chunk(f"sparse_{i}") for i in range(10)]

        dense_mock = make_mock_retriever(dense_chunks)
        sparse_mock = make_mock_retriever(sparse_chunks)

        retriever = HybridRetriever(
            dense_retriever=dense_mock, sparse_retriever=sparse_mock
        )
        results = retriever.retrieve("query", top_k=5)
        assert len(results) == 5

    def test_dense_retriever_failure_falls_back_to_sparse(self):
        sparse_chunks = [make_chunk("sparse_fallback")]
        dense_mock = MagicMock()
        dense_mock.retrieve.side_effect = RetrievalError("Dense failed")
        sparse_mock = make_mock_retriever(sparse_chunks)

        retriever = HybridRetriever(
            dense_retriever=dense_mock, sparse_retriever=sparse_mock
        )
        results = retriever.retrieve("query", top_k=5)
        assert len(results) == 1
        assert results[0].content == "sparse_fallback"

    def test_unique_documents_from_both_lists(self):
        """Items unique to each list should both appear in fused results."""
        dense_chunks = [make_chunk("dense_unique")]
        sparse_chunks = [make_chunk("sparse_unique")]

        dense_mock = make_mock_retriever(dense_chunks)
        sparse_mock = make_mock_retriever(sparse_chunks)

        retriever = HybridRetriever(
            dense_retriever=dense_mock, sparse_retriever=sparse_mock
        )
        results = retriever.retrieve("query", top_k=10)
        contents = {r.content for r in results}
        assert "dense_unique" in contents
        assert "sparse_unique" in contents

    def test_sparse_contribution_in_fusion(self):
        """Sparse-only unique content must appear when BM25 index is populated."""
        dense_chunks = [make_chunk("shared sepsis protocol document")]
        sparse_corpus = [
            make_chunk("shared sepsis protocol document"),
            make_chunk("sparse unique bacteremia antibiotic guideline"),
        ]
        sparse = SparseRetriever()
        sparse.build_index(sparse_corpus)

        dense_mock = make_mock_retriever(dense_chunks)
        retriever = HybridRetriever(
            dense_retriever=dense_mock, sparse_retriever=sparse
        )
        results = retriever.retrieve("bacteremia antibiotic", top_k=5)
        contents = {r.content for r in results}
        assert "sparse unique bacteremia antibiotic guideline" in contents


class TestDocScopedHybrid:
    """Doc-scoped retrieval must never fuse chunks from other documents."""

    def test_doc_scoped_retrieve_excludes_other_documents(self):
        doc_a = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        doc_b = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

        chunks_a = [
            ScoredChunk(
                content="patient alpha hypertension treatment plan",
                metadata={"doc_id": doc_a},
                score=0.9,
            ),
            ScoredChunk(
                content="alpha blood pressure medication dosage",
                metadata={"doc_id": doc_a},
                score=0.8,
            ),
        ]
        chunks_b = [
            ScoredChunk(
                content="patient beta diabetes insulin regimen",
                metadata={"doc_id": doc_b},
                score=0.9,
            ),
        ]

        dense_mock = MagicMock()
        # Simulate dense returning cross-doc hits before safety filter
        dense_mock.retrieve.return_value = chunks_a + chunks_b

        scoped_sparse = SparseRetriever()
        scoped_sparse.build_index(chunks_a)

        retriever = HybridRetriever(
            dense_retriever=dense_mock, sparse_retriever=scoped_sparse
        )
        results = retriever.retrieve(
            "hypertension blood pressure",
            top_k=5,
            doc_id_filter=doc_a,
        )

        assert results, "expected fused results for doc-scoped query"
        for chunk in results:
            assert chunk.metadata.get("doc_id") == doc_a

        contents = {r.content for r in results}
        assert "patient beta diabetes insulin regimen" not in contents

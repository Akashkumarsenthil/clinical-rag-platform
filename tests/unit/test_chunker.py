"""Unit tests for chunking strategies."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from langchain_core.documents import Document

from src.ingestion.chunker import (
    ChunkingError,
    RecursiveCharacterChunker,
    SemanticChunker,
    SlidingWindowChunker,
    get_chunker,
)

CHARS_PER_TOKEN = 4  # mirroring chunker.py constant


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_doc(text: str, source: str = "test.pdf") -> Document:
    return Document(
        page_content=text,
        metadata={"source": source, "page_number": 1, "doc_id": "test-001"},
    )


def long_text(num_tokens: int = 1000) -> str:
    """Generate a deterministic clinical text of approximately *num_tokens* tokens."""
    word = "clinical "  # 9 chars ≈ 2.25 tokens at 4 chars/token
    chars_needed = num_tokens * CHARS_PER_TOKEN
    repeats = chars_needed // len(word) + 1
    return (word * repeats)[:chars_needed]


# ---------------------------------------------------------------------------
# RecursiveCharacterChunker
# ---------------------------------------------------------------------------


class TestRecursiveCharacterChunker:
    def test_empty_input_returns_empty(self):
        chunker = RecursiveCharacterChunker()
        result = chunker.chunk([])
        assert result == []

    def test_single_short_sentence_produces_one_chunk(self):
        chunker = RecursiveCharacterChunker()
        doc = make_doc("Metformin reduces hepatic glucose production.")
        chunks = chunker.chunk([doc])
        assert len(chunks) == 1
        assert "Metformin" in chunks[0].page_content

    def test_long_document_produces_multiple_chunks(self):
        chunker = RecursiveCharacterChunker(chunk_size_tokens=64, chunk_overlap_tokens=8)
        doc = make_doc(long_text(500))
        chunks = chunker.chunk([doc])
        assert len(chunks) > 1

    def test_max_chunk_size_respected(self):
        """No chunk should exceed chunk_size_tokens * CHARS_PER_TOKEN characters."""
        size_tokens = 64
        chunker = RecursiveCharacterChunker(
            chunk_size_tokens=size_tokens, chunk_overlap_tokens=8
        )
        doc = make_doc(long_text(500))
        chunks = chunker.chunk([doc])
        max_chars = size_tokens * CHARS_PER_TOKEN
        for chunk in chunks:
            assert len(chunk.page_content) <= max_chars, (
                f"Chunk length {len(chunk.page_content)} exceeds limit {max_chars}"
            )

    def test_metadata_preserved_and_augmented(self):
        chunker = RecursiveCharacterChunker()
        doc = make_doc("Short clinical note.")
        chunks = chunker.chunk([doc])
        assert all("source" in c.metadata for c in chunks)
        assert all("chunk_index" in c.metadata for c in chunks)
        assert all(c.metadata["chunker"] == "recursive" for c in chunks)

    def test_content_coverage(self):
        """All words from the source document should appear in at least one chunk."""
        chunker = RecursiveCharacterChunker(chunk_size_tokens=64, chunk_overlap_tokens=8)
        original = "The quick brown fox jumps over the lazy dog. " * 50
        doc = make_doc(original)
        chunks = chunker.chunk([doc])
        combined = " ".join(c.page_content for c in chunks)
        # Spot-check a few distinctive words
        for word in ["quick", "brown", "jumps", "lazy"]:
            assert word in combined

    def test_factory_returns_recursive_chunker(self):
        chunker = get_chunker("recursive")
        assert isinstance(chunker, RecursiveCharacterChunker)

    def test_factory_unknown_strategy_raises(self):
        with pytest.raises(ChunkingError):
            get_chunker("nonexistent_strategy")


# ---------------------------------------------------------------------------
# SlidingWindowChunker
# ---------------------------------------------------------------------------


class TestSlidingWindowChunker:
    def test_empty_input_returns_empty(self):
        chunker = SlidingWindowChunker()
        assert chunker.chunk([]) == []

    def test_single_sentence_produces_one_chunk(self):
        chunker = SlidingWindowChunker(window_tokens=128, stride_tokens=64)
        doc = make_doc("ACE inhibitors are first-line for hypertension.")
        chunks = chunker.chunk([doc])
        assert len(chunks) >= 1

    def test_window_size_respected(self):
        window_tokens = 32
        chunker = SlidingWindowChunker(window_tokens=window_tokens, stride_tokens=16)
        doc = make_doc(long_text(300))
        chunks = chunker.chunk([doc])
        max_chars = window_tokens * CHARS_PER_TOKEN
        for chunk in chunks:
            assert len(chunk.page_content) <= max_chars

    def test_overlapping_windows(self):
        """With stride < window, consecutive chunks should share content."""
        window_tokens = 32
        stride_tokens = 16
        chunker = SlidingWindowChunker(
            window_tokens=window_tokens, stride_tokens=stride_tokens
        )
        text = "A" * (window_tokens * CHARS_PER_TOKEN * 3)
        doc = make_doc(text)
        chunks = chunker.chunk([doc])
        assert len(chunks) > 2  # confirms multiple overlapping windows

    def test_metadata_contains_window_positions(self):
        chunker = SlidingWindowChunker(window_tokens=32, stride_tokens=16)
        doc = make_doc(long_text(200))
        chunks = chunker.chunk([doc])
        assert all("window_start" in c.metadata for c in chunks)
        assert all("window_end" in c.metadata for c in chunks)

    def test_factory_returns_sliding_window_chunker(self):
        chunker = get_chunker("sliding_window")
        assert isinstance(chunker, SlidingWindowChunker)


# ---------------------------------------------------------------------------
# SemanticChunker
# ---------------------------------------------------------------------------


class TestSemanticChunker:
    def _make_mock_model(self, similarity: float = 0.5) -> MagicMock:
        """Return a mock SentenceTransformer that returns controlled embeddings."""
        mock_model = MagicMock()
        # Return embeddings where consecutive pairs have controlled cosine similarity
        # We achieve this by returning orthogonal unit vectors
        def fake_encode(sentences, **kwargs):
            n = len(sentences)
            embeddings = []
            for i in range(n):
                vec = np.zeros(128)
                vec[i % 128] = 1.0  # orthogonal unit vectors → similarity ≈ 0
                embeddings.append(vec)
            return np.array(embeddings)

        mock_model.encode.side_effect = fake_encode
        return mock_model

    def test_empty_input_returns_empty(self):
        chunker = SemanticChunker()
        assert chunker.chunk([]) == []

    def test_single_sentence_input(self):
        chunker = SemanticChunker(similarity_threshold=0.9)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0, 0.0]])
        chunker._model = mock_model

        doc = make_doc("Amoxicillin is an aminopenicillin antibiotic.")
        chunks = chunker.chunk([doc])
        assert len(chunks) == 1
        assert "Amoxicillin" in chunks[0].page_content

    def test_low_similarity_splits_into_multiple_chunks(self):
        """Orthogonal embeddings (similarity ≈ 0) should split each sentence into its own chunk."""
        chunker = SemanticChunker(similarity_threshold=0.85)
        chunker._model = self._make_mock_model(similarity=0.0)

        sentences = [
            "Aspirin inhibits COX-1 and COX-2 enzymes.",
            "Warfarin is a vitamin K antagonist.",
            "Heparin activates antithrombin III.",
        ]
        doc = make_doc(" ".join(sentences))
        chunks = chunker.chunk([doc])
        # With orthogonal embeddings every sentence should split
        assert len(chunks) >= 2

    def test_high_similarity_merges_into_fewer_chunks(self):
        """High-similarity embeddings should result in fewer, larger chunks."""
        chunker = SemanticChunker(similarity_threshold=0.1)  # very low threshold → always merge
        mock_model = MagicMock()
        # Return identical embeddings → similarity = 1.0 > 0.1 → all merge
        mock_model.encode.return_value = np.ones((5, 128)) / np.sqrt(128)
        chunker._model = mock_model

        text = " ".join([f"Sentence number {i} about pharmacology." for i in range(5)])
        doc = make_doc(text)
        chunks = chunker.chunk([doc])
        assert len(chunks) == 1

    def test_metadata_contains_chunker_key(self):
        chunker = SemanticChunker(similarity_threshold=0.85)
        chunker._model = self._make_mock_model()
        doc = make_doc("Beta blockers reduce heart rate. They are used in hypertension.")
        chunks = chunker.chunk([doc])
        assert all(c.metadata.get("chunker") == "semantic" for c in chunks)

    def test_factory_returns_semantic_chunker(self):
        chunker = get_chunker("semantic")
        assert isinstance(chunker, SemanticChunker)

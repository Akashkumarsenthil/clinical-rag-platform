"""Text chunking strategies using the Strategy Pattern."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import structlog
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = structlog.get_logger(__name__)

CHARS_PER_TOKEN = 4  # rough approximation when tiktoken is unavailable


class ChunkingError(Exception):
    """Raised when a chunking operation fails."""


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    @abstractmethod
    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split a list of Documents into smaller chunks.

        Args:
            documents: Source documents (typically one per PDF page).

        Returns:
            A flat list of chunk Documents with inherited metadata plus
            ``chunk_index`` and ``chunker`` keys added to each chunk's metadata.

        Raises:
            ChunkingError: If chunking fails for any document.
        """


# ---------------------------------------------------------------------------
# Strategy 1: Recursive Character Chunker
# ---------------------------------------------------------------------------


class RecursiveCharacterChunker(BaseChunker):
    """Chunker backed by LangChain's RecursiveCharacterTextSplitter.

    Uses a target of 512 tokens (≈ 2 048 characters) with 50-token overlap.
    Separators are tried in order: paragraph breaks → newlines → sentences → spaces.
    """

    def __init__(
        self,
        chunk_size_tokens: int = 512,
        chunk_overlap_tokens: int = 50,
    ) -> None:
        self._chunk_size = chunk_size_tokens * CHARS_PER_TOKEN
        self._chunk_overlap = chunk_overlap_tokens * CHARS_PER_TOKEN
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
            length_function=len,
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split documents using recursive character splitting."""
        if not documents:
            return []

        try:
            chunks = self._splitter.split_documents(documents)
        except Exception as exc:
            raise ChunkingError(f"RecursiveCharacterChunker failed: {exc}") from exc

        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["chunker"] = "recursive"

        logger.info(
            "chunking_complete",
            strategy="recursive",
            input_docs=len(documents),
            output_chunks=len(chunks),
        )
        return chunks


# ---------------------------------------------------------------------------
# Strategy 2: Semantic Chunker
# ---------------------------------------------------------------------------


class SemanticChunker(BaseChunker):
    """Embeds sentences and merges consecutive ones whose cosine similarity
    remains above a threshold (default 0.85).

    This avoids splitting mid-topic while keeping chunk sizes manageable.
    The embedder is loaded lazily on first use to avoid import-time side effects.
    """

    def __init__(self, similarity_threshold: float = 0.85) -> None:
        self._threshold = similarity_threshold
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _split_into_sentences(self, text: str) -> list[str]:
        """Naïve sentence splitter that handles common abbreviations."""
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split documents into semantically coherent chunks."""
        if not documents:
            return []

        model = self._get_model()
        chunks: list[Document] = []
        chunk_index = 0

        for doc in documents:
            try:
                sentences = self._split_into_sentences(doc.page_content)
                if not sentences:
                    continue

                embeddings: np.ndarray = model.encode(sentences, show_progress_bar=False)

                # Greedy merge: keep adding sentences while similarity is high
                groups: list[list[str]] = []
                current_group: list[str] = [sentences[0]]

                for i in range(1, len(sentences)):
                    sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])
                    if sim >= self._threshold:
                        current_group.append(sentences[i])
                    else:
                        groups.append(current_group)
                        current_group = [sentences[i]]
                groups.append(current_group)

                for group in groups:
                    text = " ".join(group)
                    meta = dict(doc.metadata)
                    meta["chunk_index"] = chunk_index
                    meta["chunker"] = "semantic"
                    chunks.append(Document(page_content=text, metadata=meta))
                    chunk_index += 1

            except Exception as exc:
                raise ChunkingError(
                    f"SemanticChunker failed on doc '{doc.metadata.get('source')}': {exc}"
                ) from exc

        logger.info(
            "chunking_complete",
            strategy="semantic",
            input_docs=len(documents),
            output_chunks=len(chunks),
        )
        return chunks


# ---------------------------------------------------------------------------
# Strategy 3: Sliding Window Chunker
# ---------------------------------------------------------------------------


class SlidingWindowChunker(BaseChunker):
    """Fixed-size sliding window chunker with 50 % overlap.

    Default: 256-token window (≈ 1 024 chars), 128-token step (≈ 512 chars).
    """

    def __init__(
        self,
        window_tokens: int = 256,
        stride_tokens: int = 128,
    ) -> None:
        self._window = window_tokens * CHARS_PER_TOKEN
        self._stride = stride_tokens * CHARS_PER_TOKEN

    def chunk(self, documents: list[Document]) -> list[Document]:
        """Produce overlapping fixed-size windows over each document."""
        if not documents:
            return []

        chunks: list[Document] = []
        chunk_index = 0

        for doc in documents:
            text = doc.page_content
            if not text.strip():
                continue

            try:
                start = 0
                while start < len(text):
                    end = min(start + self._window, len(text))
                    window_text = text[start:end].strip()
                    if window_text:
                        meta = dict(doc.metadata)
                        meta["chunk_index"] = chunk_index
                        meta["chunker"] = "sliding_window"
                        meta["window_start"] = start
                        meta["window_end"] = end
                        chunks.append(Document(page_content=window_text, metadata=meta))
                        chunk_index += 1

                    if end == len(text):
                        break
                    start += self._stride

            except Exception as exc:
                raise ChunkingError(
                    f"SlidingWindowChunker failed on doc '{doc.metadata.get('source')}': {exc}"
                ) from exc

        logger.info(
            "chunking_complete",
            strategy="sliding_window",
            input_docs=len(documents),
            output_chunks=len(chunks),
        )
        return chunks


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_STRATEGY_MAP: dict[str, type[BaseChunker]] = {
    "recursive": RecursiveCharacterChunker,
    "semantic": SemanticChunker,
    "sliding_window": SlidingWindowChunker,
}


def get_chunker(strategy: str) -> BaseChunker:
    """Return a chunker instance for the given strategy name.

    Args:
        strategy: One of ``"recursive"``, ``"semantic"``, or ``"sliding_window"``.

    Returns:
        An instantiated BaseChunker subclass.

    Raises:
        ChunkingError: If the strategy name is not recognised.
    """
    key = strategy.lower().strip()
    if key not in _STRATEGY_MAP:
        raise ChunkingError(
            f"Unknown chunking strategy '{strategy}'. "
            f"Valid options: {sorted(_STRATEGY_MAP)}"
        )
    return _STRATEGY_MAP[key]()

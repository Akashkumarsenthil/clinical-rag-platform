"""Local sentence-transformers embedder — no API key, no cost."""

from __future__ import annotations

import math
from typing import Any

import structlog
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings

logger = structlog.get_logger(__name__)

_MODEL_CACHE: dict[str, SentenceTransformer] = {}


class EmbeddingError(Exception):
    """Raised when an embedding request fails after all retries."""


class Embedder:
    """Local embedding client using sentence-transformers.

    Uses ``all-MiniLM-L6-v2`` by default (~80 MB, 384-dim vectors).
    No API key required. Runs on CPU, CUDA, or Apple MPS.

    Parameters
    ----------
    model:
        HuggingFace model name (default from settings).
    batch_size:
        Texts per forward pass (default from settings).
    device:
        Torch device string — "cpu", "cuda", or "mps" (default from settings).
    """

    def __init__(
        self,
        model: str | None = None,
        batch_size: int | None = None,
        device: str | None = None,
        _override_model: Any | None = None,  # for unit-test injection
    ) -> None:
        self._model_name = model or settings.EMBEDDING_MODEL
        self._batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        self._device = device or settings.EMBEDDING_DEVICE

        if _override_model is not None:
            self._st_model = _override_model
        else:
            if self._model_name not in _MODEL_CACHE:
                logger.info("loading_embedding_model", model=self._model_name, device=self._device)
                _MODEL_CACHE[self._model_name] = SentenceTransformer(
                    self._model_name, device=self._device
                )
            self._st_model = _MODEL_CACHE[self._model_name]

    @property
    def embedding_dim(self) -> int:
        """Return the vector dimensionality of the loaded model."""
        return self._st_model.get_sentence_embedding_dimension()  # type: ignore[return-value]

    def count_tokens(self, text: str) -> int:
        """Approximate token count using the model's tokenizer."""
        tokenizer = self._st_model.tokenizer
        return len(tokenizer.encode(text, add_special_tokens=False))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts in batches.

        Args:
            texts: Plain-text strings to embed.

        Returns:
            A list of float vectors (one per input text, same order).

        Raises:
            EmbeddingError: If embedding fails after all retry attempts.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        num_batches = math.ceil(len(texts) / self._batch_size)

        for batch_idx in range(num_batches):
            start = batch_idx * self._batch_size
            end = min(start + self._batch_size, len(texts))
            batch = texts[start:end]

            logger.info(
                "embedding_batch",
                batch=batch_idx + 1,
                total_batches=num_batches,
                texts_in_batch=len(batch),
            )
            batch_embeddings = self._embed_with_retry(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        """Run a single batch through sentence-transformers with retry on failure."""
        try:
            vectors = self._st_model.encode(
                texts,
                batch_size=self._batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return [v.tolist() for v in vectors]
        except Exception as exc:
            logger.warning("embedding_retry", error=str(exc))
            raise EmbeddingError(f"Local embedding failed: {exc}") from exc

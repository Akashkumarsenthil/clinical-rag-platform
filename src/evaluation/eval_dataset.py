"""QA pair generation for evaluation using Groq or Ollama."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import structlog
from groq import Groq
from qdrant_client import QdrantClient

from src.config import settings

logger = structlog.get_logger(__name__)


class DatasetGenerationError(Exception):
    """Raised when QA dataset generation fails."""


@dataclass
class QAPair:
    """A single question-answer pair with supporting contexts."""

    question: str
    answer: str
    contexts: list[str] = field(default_factory=list)


_GENERATION_PROMPT = """\
You are a clinical education specialist creating evaluation questions.

Given the following clinical text excerpt, generate {num_pairs} diverse
question-answer pairs that test understanding of the content.

Rules:
- Questions should be specific and answerable from the context.
- Answers should be concise (1-3 sentences).
- Cover different aspects: diagnosis, treatment, mechanisms, dosing.
- Output valid JSON only: a list of objects with keys "question" and "answer".

Clinical text:
{context}

JSON output:"""


class EvalDatasetGenerator:
    """Generates question-answer pairs from the Qdrant vector store.

    Uses Groq (free) or Ollama (local) — no OpenAI required.

    Parameters
    ----------
    qdrant_client:
        Optional pre-configured QdrantClient.
    groq_client:
        Optional pre-configured Groq client (used when LLM_BACKEND="groq").
    """

    def __init__(
        self,
        qdrant_client: QdrantClient | None = None,
        groq_client: Groq | None = None,
    ) -> None:
        self._qdrant = qdrant_client or QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY or None,
        )
        self._groq = groq_client or Groq(api_key=settings.GROQ_API_KEY)
        self._collection = settings.QDRANT_COLLECTION_NAME

    def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM backend and return the text response."""
        backend = settings.LLM_BACKEND.lower()

        if backend == "ollama":
            import httpx

            resp = httpx.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate",
                json={"model": settings.OLLAMA_MODEL, "prompt": prompt, "stream": False},
                timeout=60.0,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")

        # Default: Groq
        response = self._groq.chat.completions.create(
            model=settings.CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
        )
        return response.choices[0].message.content or "[]"

    def generate(self, num_pairs: int = 100) -> list[QAPair]:
        """Generate QA pairs by sampling chunks from Qdrant and prompting the LLM.

        Args:
            num_pairs: Total number of QA pairs to generate.

        Returns:
            List of QAPair objects with question, answer, and source contexts.

        Raises:
            DatasetGenerationError: If chunk sampling or LLM generation fails.
        """
        logger.info("eval_dataset_generate_start", target_pairs=num_pairs)

        try:
            points, _ = self._qdrant.scroll(
                collection_name=self._collection,
                limit=200,
                with_payload=True,
            )
        except Exception as exc:
            raise DatasetGenerationError(
                f"Failed to fetch chunks from Qdrant: {exc}"
            ) from exc

        if not points:
            raise DatasetGenerationError(
                "No documents found in Qdrant. Run ingest first."
            )

        chunks = [
            p.payload.get("text", "")
            for p in points
            if p.payload and p.payload.get("text")
        ]
        pairs_per_chunk = max(1, num_pairs // max(len(chunks), 1))
        all_pairs: list[QAPair] = []

        for chunk_text in chunks:
            if len(all_pairs) >= num_pairs:
                break
            if len(chunk_text.strip()) < 100:
                continue

            to_generate = min(pairs_per_chunk, num_pairs - len(all_pairs))
            prompt = _GENERATION_PROMPT.format(
                num_pairs=to_generate,
                context=chunk_text[:2000],
            )

            try:
                raw = self._call_llm(prompt)
                raw = (
                    raw.strip()
                    .removeprefix("```json")
                    .removeprefix("```")
                    .removesuffix("```")
                    .strip()
                )
                generated: list[dict] = json.loads(raw)

                for item in generated:
                    if "question" in item and "answer" in item:
                        all_pairs.append(
                            QAPair(
                                question=item["question"],
                                answer=item["answer"],
                                contexts=[chunk_text],
                            )
                        )
            except Exception as exc:
                logger.warning("qa_generation_chunk_failed", error=str(exc))
                continue

        logger.info("eval_dataset_generate_done", pairs_generated=len(all_pairs))
        return all_pairs[:num_pairs]

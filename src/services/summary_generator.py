"""Clinical document summary generation using LLM."""

from __future__ import annotations

import structlog

from src.agents.nodes import _get_llm

logger = structlog.get_logger(__name__)

_SUMMARY_PROMPT = """You are a clinical documentation specialist.
Generate a concise clinical summary (maximum 300 words) of the following
medical document. Focus on: chief complaint, key findings, diagnoses,
treatment plan, and follow-up instructions.

Document text (first 5000 chars):
{text}

Clinical Summary:"""


class SummaryGenerator:
    """Generates bounded-length clinical summaries of documents."""

    def generate(self, text: str) -> str:
        """Generate a clinical summary from document text."""
        logger.info("summary_generation_start", text_len=len(text))

        llm = _get_llm()
        prompt = _SUMMARY_PROMPT.format(text=text[:5000])
        response = llm.invoke(prompt)
        summary = response.content.strip()  # type: ignore[union-attr]

        logger.info("summary_generation_done", summary_len=len(summary))
        return summary

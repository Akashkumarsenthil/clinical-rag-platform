"""LangGraph agent state definition."""

from __future__ import annotations

from typing import Optional

from typing_extensions import NotRequired, TypedDict

from src.retrieval.dense_retriever import ScoredChunk


class AgentState(TypedDict):
    """Shared mutable state passed between LangGraph nodes.

    Attributes
    ----------
    question:
        The user's original (or rewritten) question.
    documents:
        Retrieved and optionally filtered ScoredChunk objects.
    generation:
        The LLM's generated answer (None until generate_node runs).
    num_retries:
        Number of retrieve→grade→rewrite iterations completed.
    confidence_score:
        Estimated confidence of the generated answer (0.0–1.0).
    session_id:
        Caller-provided identifier for the conversation session.
    query_rewritten:
        Whether the query has already been rewritten at least once.
    doc_id_filter:
        When set, retrieval is scoped to this document ID only.
    """

    question: str
    documents: list[ScoredChunk]
    generation: Optional[str]
    num_retries: int
    confidence_score: float
    session_id: str
    query_rewritten: bool
    doc_id_filter: NotRequired[Optional[str]]

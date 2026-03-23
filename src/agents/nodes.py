"""LangGraph node functions for the clinical RAG agent."""

from __future__ import annotations

import math
import structlog
from langchain_core.language_models.chat_models import BaseChatModel

from src.agents.state import AgentState
from src.config import settings
from src.retrieval.dense_retriever import ScoredChunk
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker

logger = structlog.get_logger(__name__)

# Shared singletons — loaded once per process
_hybrid_retriever: HybridRetriever | None = None
_reranker: CrossEncoderReranker | None = None
_llm: BaseChatModel | None = None


def _get_hybrid_retriever() -> HybridRetriever:
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever()
    return _hybrid_retriever


def _get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker


def _get_llm() -> BaseChatModel:
    """Return the configured LLM — Groq or Ollama, chosen by LLM_BACKEND setting."""
    global _llm
    if _llm is not None:
        return _llm

    backend = settings.LLM_BACKEND.lower()

    if backend == "ollama":
        from langchain_community.chat_models import ChatOllama  # type: ignore[import-untyped]

        logger.info("llm_backend", backend="ollama", model=settings.OLLAMA_MODEL)
        _llm = ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            temperature=0.0,
        )
    else:
        from langchain_groq import ChatGroq

        if not settings.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
                "or set LLM_BACKEND=ollama to use a local model."
            )
        logger.info("llm_backend", backend="groq", model=settings.CHAT_MODEL)
        _llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.CHAT_MODEL,
            temperature=0.0,
        )

    return _llm


# ── Node: retrieve ────────────────────────────────────────────────────────────

def retrieve_node(state: AgentState) -> AgentState:
    """Retrieve candidate chunks using HybridRetriever then rerank with CrossEncoder.

    Updates state['documents'] with the top-5 reranked chunks.
    """
    question = state["question"]
    logger.info("node_retrieve", question=question[:80])

    retriever = _get_hybrid_retriever()
    reranker = _get_reranker()

    candidates: list[ScoredChunk] = retriever.retrieve(question, top_k=10)
    reranked: list[ScoredChunk] = reranker.rerank(question, candidates, top_k=5)

    logger.info("node_retrieve_done", docs=len(reranked))
    return {**state, "documents": reranked}


# ── Node: grade_docs ──────────────────────────────────────────────────────────

_GRADE_PROMPT = """You are a clinical information grader.
Given a question and a document excerpt, decide if the excerpt contains information
that is relevant to answering the question.

Question: {question}

Document excerpt:
{content}

Reply with exactly one word: "yes" if relevant, "no" if not relevant."""


def grade_docs_node(state: AgentState) -> AgentState:
    """Filter documents to only those graded as relevant by the LLM.

    Each document is scored independently; irrelevant chunks are dropped.
    """
    question = state["question"]
    documents = state["documents"]
    logger.info("node_grade_docs", candidates=len(documents))

    llm = _get_llm()
    relevant: list[ScoredChunk] = []

    for chunk in documents:
        prompt = _GRADE_PROMPT.format(question=question, content=chunk.content[:800])
        try:
            response = llm.invoke(prompt)
            verdict = response.content.strip().lower()  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning("grading_failed", error=str(exc))
            verdict = "yes"  # fail-open

        if verdict.startswith("yes"):
            relevant.append(chunk)

    logger.info("node_grade_docs_done", relevant=len(relevant))
    return {**state, "documents": relevant}


# ── Node: generate ────────────────────────────────────────────────────────────

_GENERATE_PROMPT = """You are a clinical knowledge assistant.
Using only the provided context excerpts, answer the clinician's question concisely
and accurately. If the context does not contain sufficient information, say so.

Context:
{context}

Question: {question}

Answer:"""


def generate_node(state: AgentState) -> AgentState:
    """Generate a RAG answer from graded context documents.

    Sets state['generation'] and state['confidence_score'].
    Confidence is the sigmoid-normalised mean of the reranker scores.
    """
    question = state["question"]
    documents = state["documents"]
    logger.info("node_generate", docs=len(documents))

    context = "\n\n---\n\n".join(
        f"[Source: {c.metadata.get('source', 'unknown')}, "
        f"page {c.metadata.get('page_number', '?')}]\n{c.content}"
        for c in documents
    )

    prompt = _GENERATE_PROMPT.format(context=context, question=question)
    llm = _get_llm()

    try:
        response = llm.invoke(prompt)
        generation = response.content.strip()  # type: ignore[union-attr]
    except Exception as exc:
        logger.error("generation_failed", error=str(exc))
        generation = "An error occurred while generating the answer."

    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    confidence = (
        float(sum(_sigmoid(c.score) for c in documents) / len(documents))
        if documents
        else 0.0
    )

    logger.info("node_generate_done", confidence=round(confidence, 3))
    return {**state, "generation": generation, "confidence_score": confidence}


# ── Node: fallback ────────────────────────────────────────────────────────────

FALLBACK_RESPONSE = (
    "I was unable to find sufficient information in the clinical knowledge base "
    "to answer your question accurately. Please consult the primary literature "
    "or a clinical specialist."
)


def fallback_node(state: AgentState) -> AgentState:
    """Return a canned 'insufficient information' response after max retries."""
    logger.info("node_fallback", question=state["question"][:80])
    return {**state, "generation": FALLBACK_RESPONSE, "confidence_score": 0.0}


# ── Node: rewrite_query ───────────────────────────────────────────────────────

_REWRITE_PROMPT = """You are a clinical search query optimizer.
The following question did not yield relevant results from a clinical knowledge base.
Rewrite it to be more specific, using clinical terminology where appropriate.

Original question: {question}

Rewritten question (output only the question, no explanation):"""


def rewrite_query_node(state: AgentState) -> AgentState:
    """Rewrite the query to improve retrieval on the next iteration."""
    question = state["question"]
    logger.info("node_rewrite_query", original=question[:80])

    llm = _get_llm()
    prompt = _REWRITE_PROMPT.format(question=question)

    try:
        response = llm.invoke(prompt)
        rewritten = response.content.strip()  # type: ignore[union-attr]
    except Exception as exc:
        logger.warning("rewrite_failed", error=str(exc))
        rewritten = question  # keep original if rewrite fails

    logger.info("node_rewrite_query_done", rewritten=rewritten[:80])
    return {
        **state,
        "question": rewritten,
        "num_retries": state["num_retries"] + 1,
        "query_rewritten": True,
    }

"""LangChain tool wrappers for use inside the agent."""

from __future__ import annotations

import structlog
from langchain_core.tools import tool

from src.retrieval.hybrid_retriever import HybridRetriever

logger = structlog.get_logger(__name__)

_retriever: HybridRetriever | None = None


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


@tool
def search_documents(query: str, top_k: int = 5) -> str:
    """Search the clinical knowledge base and return the top relevant text chunks.

    Args:
        query: Natural-language clinical question or search string.
        top_k: Number of chunks to retrieve (default 5).

    Returns:
        A formatted string of the top text chunks with source metadata.
    """
    logger.info("tool_search_documents", query=query[:80], top_k=top_k)
    retriever = _get_retriever()
    chunks = retriever.retrieve(query, top_k=top_k)

    if not chunks:
        return "No relevant documents found."

    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page_number", "?")
        score = round(chunk.score, 4)
        parts.append(
            f"[{i}] Source: {source}, Page: {page}, Score: {score}\n{chunk.content}"
        )

    return "\n\n---\n\n".join(parts)


@tool
def get_patient_context(session_id: str) -> str:
    """Retrieve synthetic patient context for the given session.

    This is a mock tool that returns a plausible patient summary.
    In production this would query the EHR system.

    Args:
        session_id: The caller's session identifier.

    Returns:
        A synthetic patient context string.
    """
    logger.info("tool_get_patient_context", session_id=session_id)

    # Deterministic mock based on session_id hash for reproducibility
    seed = sum(ord(c) for c in session_id) % 4

    MOCK_PATIENTS = [
        (
            "Patient: 67-year-old male. "
            "Diagnoses: Type 2 diabetes mellitus (HbA1c 8.2%), hypertension, CKD stage 3. "
            "Current medications: Metformin 1000 mg BID, Lisinopril 10 mg QD, Atorvastatin 40 mg QD. "
            "Allergies: Penicillin (rash). "
            "Recent labs: eGFR 42 mL/min/1.73m², K+ 4.8 mEq/L."
        ),
        (
            "Patient: 45-year-old female. "
            "Diagnoses: Rheumatoid arthritis, GERD. "
            "Current medications: Methotrexate 15 mg weekly, Folic acid 1 mg QD, Omeprazole 20 mg QD. "
            "Allergies: Sulfonamides. "
            "Recent labs: LFTs within normal limits, CBC normal."
        ),
        (
            "Patient: 55-year-old male. "
            "Diagnoses: COPD (GOLD stage 2), depression. "
            "Current medications: Tiotropium 18 mcg QD, Albuterol PRN, Sertraline 50 mg QD. "
            "Allergies: NKDA. "
            "Recent PFTs: FEV1/FVC 0.62, FEV1 65% predicted."
        ),
        (
            "Patient: 32-year-old female. "
            "Diagnoses: Asthma (moderate persistent), anxiety disorder. "
            "Current medications: Fluticasone/salmeterol 250/50 mcg BID, Albuterol PRN, Escitalopram 10 mg QD. "
            "Allergies: Aspirin (bronchospasm). "
            "Recent peak flow: 380 L/min (personal best 420)."
        ),
    ]

    return MOCK_PATIENTS[seed]

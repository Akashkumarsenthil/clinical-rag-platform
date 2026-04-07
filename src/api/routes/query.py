"""Query endpoint: runs the LangGraph agent and returns an answer."""

from __future__ import annotations

import time

import structlog
from fastapi import APIRouter, HTTPException, Request

from src.agents.graph import build_graph
from src.agents.state import AgentState
from src.api.models import QueryRequest, QueryResponse, SourceDoc
from src.monitoring.metrics import active_requests, query_counter, query_latency

router = APIRouter(tags=["Query"])
logger = structlog.get_logger(__name__)

# Build the compiled graph once at import time
_GRAPH = build_graph()


@router.post("/query", response_model=QueryResponse, summary="Answer a clinical question")
async def query(request: Request, body: QueryRequest) -> QueryResponse:
    """Run the multi-step RAG agent and return a sourced answer.

    The agent executes: retrieve → grade → (generate | rewrite → retrieve loop | fallback).

    Args:
        request: FastAPI request object (used to extract request_id from state).
        body: QueryRequest with question, top_k, and session_id.

    Returns:
        QueryResponse with answer, source chunks, confidence, and latency.

    Raises:
        HTTPException 500: If the agent raises an unhandled exception.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    active_requests.labels(endpoint="/query").inc()
    start = time.perf_counter()

    logger.info(
        "query_start",
        question=body.question[:80],
        session_id=body.session_id,
        request_id=request_id,
    )

    try:
        initial_state: AgentState = {
            "question": body.question,
            "documents": [],
            "generation": None,
            "num_retries": 0,
            "confidence_score": 0.0,
            "session_id": body.session_id,
            "query_rewritten": False,
        }

        final_state: AgentState = await _GRAPH.ainvoke(initial_state)

        elapsed_ms = (time.perf_counter() - start) * 1000

        sources = [
            SourceDoc(
                content=chunk.content,
                metadata=chunk.metadata,
                score=chunk.score,
            )
            for chunk in final_state["documents"][: body.top_k]
        ]

        query_counter.labels(endpoint="/query", status="success").inc()
        query_latency.labels(endpoint="/query").observe(elapsed_ms / 1000)

        logger.info(
            "query_done",
            request_id=request_id,
            latency_ms=round(elapsed_ms, 2),
            sources=len(sources),
            confidence=round(final_state["confidence_score"], 3),
        )

        return QueryResponse(
            answer=final_state["generation"] or "",
            sources=sources,
            confidence=min(max(final_state["confidence_score"], 0.0), 1.0),
            latency_ms=round(elapsed_ms, 2),
            request_id=request_id,
        )

    except Exception as exc:
        query_counter.labels(endpoint="/query", status="error").inc()
        logger.error("query_error", error=str(exc), request_id=request_id)
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {exc}") from exc
    finally:
        active_requests.labels(endpoint="/query").dec()

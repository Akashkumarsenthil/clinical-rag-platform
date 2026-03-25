"""LangGraph StateGraph definition for the clinical RAG agent."""

from __future__ import annotations

from typing import Literal

import structlog
from langgraph.graph import END, START, StateGraph

from src.agents.nodes import (
    fallback_node,
    generate_node,
    grade_docs_node,
    retrieve_node,
    rewrite_query_node,
)
from src.agents.state import AgentState

logger = structlog.get_logger(__name__)

MAX_RETRIES = 2


def _route_after_grading(
    state: AgentState,
) -> Literal["generate", "rewrite_query", "fallback"]:
    """Conditional edge: decide next step after document grading.

    - If relevant docs found → generate.
    - If no relevant docs and retries < MAX_RETRIES → rewrite then retrieve again.
    - If exhausted retries → fallback.
    """
    has_docs = len(state["documents"]) > 0
    retries = state["num_retries"]

    if has_docs:
        return "generate"
    if retries < MAX_RETRIES:
        return "rewrite_query"
    return "fallback"


def build_graph() -> StateGraph:
    """Construct and compile the LangGraph StateGraph.

    Graph topology::

        START → retrieve → grade_docs ──┬── generate → END
                     ↑                  ├── rewrite_query → retrieve (loop)
                     │                  └── fallback → END

    Returns:
        A compiled LangGraph application ready for invocation.
    """
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_docs", grade_docs_node)
    graph.add_node("generate", generate_node)
    graph.add_node("fallback", fallback_node)
    graph.add_node("rewrite_query", rewrite_query_node)

    # Edges
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_docs")
    graph.add_conditional_edges(
        "grade_docs",
        _route_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "fallback": "fallback",
        },
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)

    compiled = graph.compile()
    logger.info("agent_graph_built")
    return compiled

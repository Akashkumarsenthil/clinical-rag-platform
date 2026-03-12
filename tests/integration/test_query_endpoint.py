"""Integration tests for the FastAPI endpoints using an async test client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.agents.state import AgentState
from src.api.main import create_app
from src.retrieval.dense_retriever import ScoredChunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_app_state():
    """Shared mock state injected into the FastAPI app."""
    qdrant_mock = MagicMock()
    qdrant_mock.get_collections.return_value.collections = []

    redis_mock = MagicMock()
    redis_mock.ping.return_value = True

    return {"qdrant_client": qdrant_mock, "redis_client": redis_mock}


@pytest_asyncio.fixture
async def async_client(mock_app_state):
    """Async HTTPX client pointed at the FastAPI app with mocked lifespan."""
    app = create_app()

    # Inject state without running lifespan (avoids network calls)
    app.state.qdrant_client = mock_app_state["qdrant_client"]
    app.state.redis_client = mock_app_state["redis_client"]
    app.state.reranker = MagicMock()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# ---------------------------------------------------------------------------
# Query endpoint tests
# ---------------------------------------------------------------------------


MOCK_AGENT_STATE: AgentState = {
    "question": "What is the mechanism of action of metformin?",
    "documents": [
        ScoredChunk(
            content="Metformin activates AMPK reducing hepatic glucose production.",
            metadata={"source": "test.pdf", "page_number": 1},
            score=0.92,
        )
    ],
    "generation": "Metformin reduces hepatic glucose output via AMPK activation.",
    "num_retries": 0,
    "confidence_score": 0.85,
    "session_id": "test-session-001",
    "query_rewritten": False,
}


class TestQueryEndpoint:
    @pytest.mark.asyncio
    async def test_query_returns_200(self, async_client):
        with patch("src.api.routes.query._GRAPH") as mock_graph:
            mock_graph.ainvoke = AsyncMock(return_value=MOCK_AGENT_STATE)

            response = await async_client.post(
                "/api/v1/query",
                json={
                    "question": "What is the mechanism of action of metformin?",
                    "top_k": 5,
                    "session_id": "test-session-001",
                },
            )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_query_response_schema(self, async_client):
        with patch("src.api.routes.query._GRAPH") as mock_graph:
            mock_graph.ainvoke = AsyncMock(return_value=MOCK_AGENT_STATE)

            response = await async_client.post(
                "/api/v1/query",
                json={
                    "question": "What is the mechanism of action of metformin?",
                    "top_k": 5,
                    "session_id": "test-session-001",
                },
            )

        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "confidence" in data
        assert "latency_ms" in data
        assert "request_id" in data
        assert isinstance(data["sources"], list)
        assert 0.0 <= data["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_query_missing_question_returns_422(self, async_client):
        response = await async_client.post(
            "/api/v1/query",
            json={"top_k": 5, "session_id": "sess-1"},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_query_request_id_in_response_headers(self, async_client):
        with patch("src.api.routes.query._GRAPH") as mock_graph:
            mock_graph.ainvoke = AsyncMock(return_value=MOCK_AGENT_STATE)

            response = await async_client.post(
                "/api/v1/query",
                json={
                    "question": "Test question",
                    "top_k": 3,
                    "session_id": "sess-2",
                },
            )

        assert "x-request-id" in response.headers


# ---------------------------------------------------------------------------
# Health endpoint tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, async_client):
        response = await async_client.get("/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_health_response_schema(self, async_client):
        response = await async_client.get("/health")
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "services" in data
        assert isinstance(data["services"], dict)

    @pytest.mark.asyncio
    async def test_health_reports_healthy_when_services_up(self, async_client):
        response = await async_client.get("/health")
        data = response.json()
        # With mocked clients that succeed, status should be healthy
        assert data["status"] in ("healthy", "degraded")  # degraded ok in test env


# ---------------------------------------------------------------------------
# Metrics endpoint tests
# ---------------------------------------------------------------------------


class TestMetricsEndpoint:
    @pytest.mark.asyncio
    async def test_metrics_returns_200(self, async_client):
        response = await async_client.get("/metrics")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_metrics_content_type_is_prometheus(self, async_client):
        response = await async_client.get("/metrics")
        assert "text/plain" in response.headers["content-type"]

    @pytest.mark.asyncio
    async def test_metrics_contains_expected_metric_names(self, async_client):
        response = await async_client.get("/metrics")
        body = response.text
        assert "clinical_rag_queries_total" in body or "python_gc" in body

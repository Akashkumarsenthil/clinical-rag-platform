"""Prometheus metrics definitions for the Clinical RAG Platform."""

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# Query metrics
# ---------------------------------------------------------------------------

query_counter = Counter(
    name="clinical_rag_queries_total",
    documentation="Total number of query requests processed.",
    labelnames=["endpoint", "status"],
)

query_latency = Histogram(
    name="clinical_rag_query_duration_seconds",
    documentation="End-to-end query latency in seconds.",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
    labelnames=["endpoint"],
)

# ---------------------------------------------------------------------------
# Ingestion metrics
# ---------------------------------------------------------------------------

ingestion_counter = Counter(
    name="clinical_rag_ingestions_total",
    documentation="Total number of document ingestion requests.",
    labelnames=["status"],
)

# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

retrieval_hit_rate = Gauge(
    name="clinical_rag_retrieval_hit_rate",
    documentation="Fraction of queries that returned at least one relevant document.",
)

# ---------------------------------------------------------------------------
# Concurrency metrics
# ---------------------------------------------------------------------------

active_requests = Gauge(
    name="clinical_rag_active_requests",
    documentation="Number of requests currently being processed.",
    labelnames=["endpoint"],
)

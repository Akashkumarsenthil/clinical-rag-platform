# System Architecture

## Overview

The Clinical RAG Platform is a production-grade retrieval-augmented generation system
designed for clinical document Q&A.  It combines dense vector search, BM25 sparse
retrieval, cross-encoder reranking, and a multi-step LangGraph agent to deliver
high-quality, sourced answers from clinical literature.

---

## Full Data Flow

### Query Path

```mermaid
flowchart TD
    Client([Client]) -->|POST /api/v1/query| API[FastAPI]
    API --> MW[Middleware\nRequestID · Latency · RateLimit]
    MW --> QR[/query route/]
    QR --> AG[LangGraph Agent]

    AG --> RN[retrieve_node]
    RN --> HR[HybridRetriever]
    HR --> DR[DenseRetriever\nQdrant cosine search]
    HR --> SR[SparseRetriever\nBM25 index]
    DR --> RRF[RRF Fusion]
    SR --> RRF
    RRF --> RE[CrossEncoderReranker\nms-marco-MiniLM-L-6-v2]
    RE --> GD[grade_docs_node\nLLM relevance filter]

    GD -->|relevant docs found| GN[generate_node\nGPT-4o RAG answer]
    GD -->|no relevant docs,\nretries left| RW[rewrite_query_node\nGPT-4o query expansion]
    GD -->|retries exhausted| FB[fallback_node\ncanned response]

    RW --> RN

    GN --> RS[QueryResponse\nanswer · sources · confidence]
    FB --> RS
    RS --> Client
```

### Ingestion Path

```mermaid
flowchart LR
    SRC([PDF file / URL]) --> IP[IngestionPipeline]
    IP --> PL[PDFLoader\nPyMuPDF]
    PL --> CK[Chunker\nrecursive / semantic / sliding_window]
    CK --> EM[Embedder\ntext-embedding-3-small\nbatch=32 · retry 3x]
    EM --> QD[(Qdrant\nvector store)]
    IP -->|idempotency check\nSHA-256 hash| QD
```

---

## Component Descriptions

| Component | Technology | Purpose |
|---|---|---|
| API Gateway | FastAPI 0.111 | REST endpoints, middleware, lifespan management |
| Agent Orchestration | LangGraph 0.1.19 | Multi-step retrieve→grade→generate loop |
| Dense Retrieval | Qdrant 1.9.2 + OpenAI embeddings | ANN cosine similarity search |
| Sparse Retrieval | rank-bm25 | BM25 keyword-based retrieval |
| Hybrid Fusion | Reciprocal Rank Fusion (k=60) | Combines dense + sparse ranked lists |
| Reranking | CrossEncoder ms-marco-MiniLM-L-6-v2 | Fine-grained query-passage scoring |
| PDF Parsing | PyMuPDF 1.24 | Fast PDF text extraction with section detection |
| Embedding | OpenAI text-embedding-3-small | 1536-dim dense vectors |
| Generation | GPT-4o | Grounded answer generation |
| Evaluation | RAGAS 0.1.10 | Faithfulness, relevancy, precision, recall |
| Metrics | Prometheus + Grafana | Latency, throughput, hit-rate observability |
| Tracing | OpenTelemetry (OTLP) | Distributed trace correlation |
| Caching/Rate-limit | Redis 7.2 | Per-IP rate limiting, future response caching |

---

## Scalability Notes

- **Horizontal scaling**: The FastAPI app is stateless; scale with multiple Uvicorn workers or Kubernetes replicas.
- **Qdrant**: Supports distributed mode (sharding + replication) for large corpora.
- **BM25 index**: Currently in-memory; for multi-worker setups, serialise with `pickle` and load from shared storage on startup.
- **Rate limiting**: The in-process `RateLimitMiddleware` should be replaced with a Redis-backed implementation in multi-worker deployments.

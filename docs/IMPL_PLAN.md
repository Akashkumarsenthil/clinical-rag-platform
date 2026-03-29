# Implementation Plan — Clinical Document Workspace

## 1. Source Tree Map

```
src/
├── __init__.py                         # Package root; exports __version__
├── config.py                           # Pydantic Settings (env/.env), singleton `settings`
│
├── agents/
│   ├── __init__.py                     # Re-exports build_graph, AgentState
│   ├── state.py                        # AgentState TypedDict (question, documents, generation, …)
│   ├── nodes.py                        # LangGraph nodes: retrieve, grade_docs, generate, rewrite_query, fallback
│   ├── tools.py                        # LangChain @tool wrappers: search_documents, get_patient_context
│   └── graph.py                        # Builds/compiles the LangGraph StateGraph (START→retrieve→grade→…→END)
│
├── api/
│   ├── __init__.py                     # Package marker
│   ├── main.py                         # FastAPI app factory + lifespan (Qdrant, Redis, reranker init)
│   ├── models.py                       # Pydantic request/response schemas (QueryRequest, IngestRequest, etc.)
│   ├── middleware.py                    # RequestID, Latency, RateLimit middleware
│   └── routes/
│       ├── __init__.py                 # Package marker
│       ├── chat_ui.py                  # GET / → embedded HTML/JS chat SPA
│       ├── health.py                   # GET /health → Qdrant + Redis connectivity
│       ├── ingest.py                   # POST /api/v1/ingest → file upload or URL → IngestionPipeline
│       ├── query.py                    # POST /api/v1/query → LangGraph agent → answer + sources
│       └── metrics.py                  # GET /metrics → Prometheus exposition
│
├── ingestion/
│   ├── __init__.py                     # Re-exports PDFLoader, Embedder, IngestionPipeline, etc.
│   ├── pdf_loader.py                   # PyMuPDF loader; 1 Document per page; SHA-256 hash, UUID5 doc_id
│   ├── chunker.py                      # Strategy pattern: Recursive / Semantic / SlidingWindow chunkers
│   ├── embedder.py                     # Local sentence-transformers (all-MiniLM-L6-v2, 384-dim), batched
│   └── pipeline.py                     # Orchestrator: load → idempotency check → chunk → embed → Qdrant upsert
│
├── retrieval/
│   ├── __init__.py                     # Re-exports all retrievers + ScoredChunk
│   ├── dense_retriever.py              # Qdrant cosine search; ScoredChunk dataclass
│   ├── sparse_retriever.py             # BM25Okapi in-memory retriever
│   ├── hybrid_retriever.py             # RRF fusion of dense + sparse
│   └── reranker.py                     # CrossEncoder (ms-marco-MiniLM-L-6-v2) re-scoring
│
├── evaluation/
│   ├── __init__.py                     # Re-exports evaluators
│   ├── ragas_eval.py                   # RAGAS metrics (faithfulness, relevancy, precision, recall)
│   ├── eval_dataset.py                 # Generates Q&A pairs from Qdrant chunks via LLM
│   └── benchmark_runner.py             # Compares chunking strategies via RAGAS
│
└── monitoring/
    ├── __init__.py                     # Re-exports metrics + setup_tracing
    ├── metrics.py                      # Prometheus counters, histograms, gauges
    └── tracer.py                       # OpenTelemetry tracing (graceful degradation)
```

---

## 2. Ingestion Path (end-to-end trace)

```
POST /api/v1/ingest  (file upload or URL)
  └─ src/api/routes/ingest.py::ingest()
       ├─ Save uploaded bytes or fetch URL → tmp file on disk
       └─ IngestionPipeline(chunk_strategy).run(tmp_path)
            └─ src/ingestion/pipeline.py::IngestionPipeline.run()
                 │
                 ├─ Step 1: PDFLoader.load(path)
                 │    └─ src/ingestion/pdf_loader.py
                 │    └─ Returns: list[Document], one per page
                 │    └─ doc_id = uuid5(NAMESPACE_URL, resolved_path)
                 │    └─ file_hash = sha256(file_bytes)
                 │
                 ├─ Step 2: Idempotency check
                 │    └─ Qdrant scroll filter: payload.file_hash == file_hash
                 │    └─ If exists → return IngestionResult(skipped=True)
                 │
                 ├─ Step 3: Chunk
                 │    └─ get_chunker(strategy).chunk(pages)
                 │    └─ Adds chunk_index, chunker to each chunk's metadata
                 │
                 ├─ Step 4: Embed
                 │    └─ Embedder.embed_documents(texts)
                 │    └─ Returns list[list[float]], 384-dim vectors
                 │
                 └─ Step 5: Qdrant upsert
                      └─ _upsert(chunks, vectors, doc_id, file_hash)
```

### Qdrant Collection Details

| Property              | Value                                           |
|-----------------------|-------------------------------------------------|
| **Collection name**   | `clinical_docs` (from `settings.QDRANT_COLLECTION_NAME`) |
| **Vector dimension**  | 384 (`VECTOR_DIM` constant in pipeline.py)      |
| **Distance metric**   | Cosine (`Distance.COSINE`)                      |

### Qdrant Point Structure (payload schema per chunk)

Created in `pipeline.py::_upsert()` (lines 174-199):

```python
point_id  = uuid5(NAMESPACE_URL, f"{doc_id}-{chunk_index}")   # deterministic
vector    = [float] * 384                                      # all-MiniLM-L6-v2
payload   = {
    # ── From PDFLoader page metadata ──
    "source":         str,    # filesystem path to PDF
    "page_number":    int,    # 1-indexed
    "total_pages":    int,
    "section_header": str,    # inferred from bold spans or regex
    "doc_id":         str,    # uuid5 of resolved path
    "file_hash":      str,    # sha256 of file
    # ── Added by chunker ──
    "chunk_index":    int,    # 0-indexed within the document
    "chunker":        str,    # "recursive" | "semantic" | "sliding_window"
    # ── Added by pipeline._upsert() ──
    "text":           str,    # the actual chunk text content
    "file_hash":      str,    # (duplicated from page metadata)
    "doc_id":         str,    # (duplicated from page metadata)
}
```

**Key finding: `doc_id` is ALREADY stored in every Qdrant point payload.** No schema migration needed — we can filter on it immediately.

---

## 3. Query Path (end-to-end trace)

```
POST /api/v1/query  { question, top_k, session_id }
  └─ src/api/routes/query.py::query()
       ├─ Build AgentState { question, documents=[], generation=None, ... }
       └─ _GRAPH.ainvoke(initial_state)
            └─ src/agents/graph.py::build_graph()  (compiled once at import)
                 │
                 ├─ Node: retrieve_node()  [src/agents/nodes.py:79]
                 │    ├─ HybridRetriever.retrieve(question, top_k=10)
                 │    │    ├─ DenseRetriever.retrieve(query, top_k=20)
                 │    │    │    └─ Qdrant.search(collection, query_vector, limit=20)
                 │    │    │       ⚠ NO query_filter param → searches ALL docs
                 │    │    ├─ SparseRetriever.retrieve(query, top_k=20)
                 │    │    └─ RRF fusion → top 10
                 │    └─ CrossEncoderReranker.rerank(question, candidates, top_k=5)
                 │
                 ├─ Node: grade_docs_node()  [nodes.py:103]
                 │    └─ LLM grades each chunk as "yes"/"no" relevant
                 │    └─ Drops irrelevant chunks from state["documents"]
                 │
                 ├─ Conditional edge: _route_after_grading()  [graph.py:20]
                 │    ├─ has relevant docs → "generate"
                 │    ├─ no docs + retries < 2 → "rewrite_query" → loop back to retrieve
                 │    └─ no docs + exhausted → "fallback"
                 │
                 ├─ Node: generate_node()  [nodes.py:130]
                 │    └─ LLM synthesizes answer from context chunks
                 │    └─ Confidence = mean(sigmoid(reranker_score))
                 │
                 └─ Node: fallback_node()  [nodes.py:168]
                      └─ Returns canned "insufficient information" response
```

### Doc-ID Filter Injection Point

**Primary injection: `DenseRetriever.retrieve()` in `src/retrieval/dense_retriever.py` line 75:**

```python
# CURRENT (no filter):
results = self._qdrant.search(
    collection_name=self._collection,
    query_vector=query_vector,
    limit=top_k,
    with_payload=True,
)

# NEEDED — add optional query_filter parameter:
results = self._qdrant.search(
    collection_name=self._collection,
    query_vector=query_vector,
    query_filter=query_filter,   # ← inject Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
    limit=top_k,
    with_payload=True,
)
```

**Propagation chain:**
1. `DenseRetriever.retrieve(query, top_k, query_filter=None)` — add optional param
2. `HybridRetriever.retrieve(query, top_k, query_filter=None)` — pass through to dense
3. `retrieve_node(state)` — read `state.get("doc_id_filter")` and build Qdrant filter
4. New `/api/v1/documents/{doc_id}/chat` endpoint — set `doc_id_filter` in AgentState

The SparseRetriever (BM25) operates on an in-memory corpus and has no Qdrant filter concept. For doc-scoped chat, we can either:
- (a) Skip sparse and use dense-only with filter (simplest, recommended)
- (b) Build a per-doc BM25 index on the fly from Qdrant scroll results

**Recommendation: Option (a)** — use dense-only retrieval for doc-scoped chat since the corpus is small (one document), and the dense retriever with Qdrant filter is sufficient.

---

## 4. Configuration & Environment Variables

### Settings class: `src/config.py` (Pydantic Settings)

Loading: `BaseSettings` with `SettingsConfigDict(env_file=".env")`. All vars can be set via env or `.env`.

| Variable                  | Default                        | Usage                            |
|---------------------------|-------------------------------|----------------------------------|
| `LLM_BACKEND`            | `"groq"`                       | `"groq"` or `"ollama"`          |
| `GROQ_API_KEY`           | `""`                           | Groq API authentication          |
| `CHAT_MODEL`             | `"llama-3.3-70b-versatile"`   | Groq model name                  |
| `OLLAMA_BASE_URL`        | `"http://localhost:11434"`     | Ollama server URL                |
| `OLLAMA_MODEL`           | `"llama3.2"`                   | Ollama model name                |
| `EMBEDDING_MODEL`        | `"all-MiniLM-L6-v2"`          | sentence-transformers model      |
| `EMBEDDING_BATCH_SIZE`   | `32`                           | Embedding batch size             |
| `EMBEDDING_DEVICE`       | `"cpu"`                        | `"cpu"` / `"cuda"` / `"mps"`   |
| `QDRANT_URL`             | `"http://localhost:6333"`      | Qdrant connection URL            |
| `QDRANT_COLLECTION_NAME` | `"clinical_docs"`              | Qdrant collection name           |
| `QDRANT_API_KEY`         | `""`                           | Qdrant Cloud auth (optional)     |
| `REDIS_URL`              | `"redis://localhost:6379"`     | Redis connection URL             |
| `LOG_LEVEL`              | `"INFO"`                       | Logging level                    |
| `ENVIRONMENT`            | `"development"`                | Runtime environment tag          |
| `MAX_RETRIES`            | `3`                            | Tenacity retry count             |
| `CHUNK_STRATEGY`         | `"recursive"`                  | Default chunking strategy        |
| `APP_VERSION`            | `"0.1.0"`                      | Reported in /health              |

### Config files

| File                              | Purpose                                    |
|-----------------------------------|--------------------------------------------|
| `.env.example`                    | Template with all vars + comments          |
| `configs/prometheus.yml`          | Prometheus scrape targets (app, qdrant)    |
| `configs/qdrant_config.yaml`      | Qdrant storage, WAL, service, cluster cfg  |

### New vars needed

| Variable        | Default                                              | Purpose                    |
|-----------------|------------------------------------------------------|----------------------------|
| `DATABASE_URL`  | `postgresql://clinical:clinical@postgres:5432/clinical_rag` | SQLAlchemy connection      |
| `PDF_STORAGE_DIR`| `/app/uploads`                                      | Persistent PDF storage     |

---

## 5. Test & Lint Layout

### Test structure
```
tests/
├── conftest.py                          # Shared fixtures: sample_documents, sample_scored_chunks, mocks
├── unit/
│   ├── __init__.py
│   ├── test_chunker.py                  # Tests for all 3 chunking strategies + factory
│   ├── test_hybrid_retriever.py         # RRF fusion, fallback, top_k
│   └── test_reranker.py                 # Cross-encoder sorting, top_k, error handling
└── integration/
    ├── __init__.py
    ├── test_ingest_pipeline.py          # Pipeline with mocked Qdrant/embedder
    └── test_query_endpoint.py           # FastAPI TestClient for /query, /health, /metrics
```

### Conventions
- **Linter**: `ruff check src/ tests/ scripts/`
- **Type-checker**: `mypy src/ --ignore-missing-imports --no-strict-optional`
- **Test runner**: `pytest tests/unit/ -v --cov=src`
- **CI**: GitHub Actions runs ruff → mypy → pytest (unit only) on push/PR to main
- **Fixtures**: mock Qdrant/Redis/LLM; never call real services in unit tests

---

## 6. Docker Compose Wiring

### Current services

| Service      | Image                    | Container Name       | Ports       | Health Check              | Network       |
|--------------|--------------------------|----------------------|-------------|---------------------------|---------------|
| `app`        | `clinical-rag-platform`  | `clinical-rag-app`   | 8000        | `curl /health`            | clinical-net  |
| `qdrant`     | `qdrant/qdrant:v1.9.2`   | `clinical-qdrant`    | 6333, 6334  | `bash /dev/tcp/6333`      | clinical-net  |
| `redis`      | `redis:7.2-alpine`       | `clinical-redis`     | 6379        | `redis-cli ping`          | clinical-net  |
| `prometheus` | `prom/prometheus:v2.52.0` | `clinical-prometheus` | 9090        | `wget /-/healthy`         | clinical-net  |
| `grafana`    | `grafana/grafana:10.4.3` | `clinical-grafana`   | 3000        | `wget /api/health`        | clinical-net  |
| `ollama`     | `ollama/ollama:latest`   | `clinical-ollama`    | 11434       | profile: ollama (opt-in) | clinical-net  |

### Services to add

| Service      | Image                    | Container Name       | Ports       | Depends On | Notes                                    |
|--------------|--------------------------|----------------------|-------------|------------|-------------------------------------------|
| `postgres`   | `postgres:16-alpine`     | `clinical-postgres`  | 5432        | —          | `DATABASE_URL` env var; persistent volume |
| `frontend`   | Build from `frontend/`   | `clinical-frontend`  | 5173        | app        | Vite dev server; nginx for prod           |

### App depends_on additions
- `postgres: condition: service_healthy`

### Volumes to add
- `postgres_data:` for persistent DB storage
- `pdf_uploads:` mounted to app at `/app/uploads`

---

## 7. Implementation Phases

### Phase 1: Backend (PostgreSQL + services + endpoints)
1. Add `sqlalchemy`, `alembic`, `asyncpg` to requirements
2. Add `DATABASE_URL`, `PDF_STORAGE_DIR` to Settings
3. Create SQLAlchemy models: `documents`, `document_metadata`, `document_summaries`
4. Create Alembic migration
5. Add `postgres` to docker-compose
6. Implement metadata extraction service (Groq JSON-schema prompt + regex fallback)
7. Implement summary service (Groq, bounded-length)
8. Implement 3-stage pipeline with Redis progress tracking
9. Add `query_filter` param to `DenseRetriever.retrieve()` and propagate
10. Add new API endpoints (documents CRUD, search, doc-scoped chat, vectors)

### Phase 2: Frontend (React + Vite + shadcn/ui)
1. Scaffold `frontend/` with Vite + React + TypeScript
2. Install shadcn/ui + Tailwind + TanStack Query + react-pdf
3. Build Upload surface (left-rail uploader + doc list + progress + vector inspector)
4. Build Search surface (metadata search + results table)
5. Build Document Workspace (PDF viewer + metadata panel + summary + scoped chat)

### Phase 3: Integration & Polish
1. Add frontend to docker-compose
2. Seed script using new 3-stage pipeline
3. Alembic migrations committed
4. Tests for new endpoints + metadata parser
5. README update with new architecture

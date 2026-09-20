# Clinical RAG Platform

Hybrid RAG for clinical documents — dense + BM25 retrieval, cross-encoder reranking, LangGraph agent, document workspace UI.

Uses Groq (Llama 3.3 70B) for LLM and local sentence-transformers for embeddings. All patient data in demos is synthetic.

**Public engineering showcase:** [akashkumarsenthil.github.io/clinical-rag-platform](https://akashkumarsenthil.github.io/clinical-rag-platform/)

The public page is a zero-cost, browser-only retrieval demonstration. It uses a
committed synthetic corpus and makes no API, model, database, or cloud-service
requests. BM25 and Reciprocal Rank Fusion run directly in the browser. The page
clearly labels its transparent semantic and reranking proxies rather than
presenting them as live MiniLM or cross-encoder inference.

---

## Quick start

```bash
git clone https://github.com/Akashkumarsenthil/clinical-rag-platform.git
cd clinical-rag-platform
cp .env.example .env   # add GROQ_API_KEY
docker compose up -d --build
```

- **UI:** http://localhost:5173
- **API:** http://localhost:8000/docs
- **Health:** http://localhost:8000/health

Seed sample docs (optional):

```bash
docker exec clinical-rag-app python scripts/seed_documents.py
```

---

## What it does

1. **Upload** — PDF → metadata extraction → summary → embed to Qdrant (3-stage pipeline)
2. **Search** — filter by patient metadata in Postgres
3. **Workspace** — PDF viewer + doc-scoped hybrid RAG chat

The browser showcase also documents the implementation, failure handling,
limitations, and a five-question synthetic retrieval regression set. It is a
document-retrieval engineering demonstration, not a validated clinical
decision-support system and not medical advice.

---

## API

```bash
# Global RAG query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Med-PaLM?", "top_k": 5, "session_id": "demo"}'

# Upload a document
curl -X POST http://localhost:8000/api/v1/documents \
  -F "file=@report.pdf"
```

---

## Environment

| Variable | Default | Notes |
|---|---|---|
| `GROQ_API_KEY` | — | console.groq.com (free) |
| `LLM_BACKEND` | `groq` | or `ollama` |
| `DATABASE_URL` | postgres local | set in docker-compose |
| `QDRANT_URL` | `http://localhost:6333` | or Qdrant Cloud |
| `REDIS_URL` | `redis://localhost:6379` | progress + rate limit |

---

## Deploy (free tier)

**Render** — connect repo, uses `render.yaml`. Add `GROQ_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY` as secrets.

**HuggingFace Spaces** — Docker SDK, use `Dockerfile.hf`, port 7860.

**Qdrant Cloud** — https://cloud.qdrant.io (1GB free) for persistent vectors.

---

## Dev

```bash
pip install -r requirements.txt
pytest tests/unit/ -v
ruff check src/ tests/
node --test tests/static_demo.test.mjs
```

To preview the static showcase locally:

```bash
python -m http.server 4173 --directory docs
```

Install commit hook (blocks unwanted co-author trailers):

```bash
bash scripts/install-hooks.sh
```

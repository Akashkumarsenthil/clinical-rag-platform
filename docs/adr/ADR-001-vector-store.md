# ADR-001: Vector Store Selection

**Status**: Accepted  
**Date**: 2024-06-01  
**Deciders**: Platform team

---

## Context

The Clinical RAG Platform requires a vector store that can:
- Handle millions of 1 536-dimensional embeddings with sub-second ANN search.
- Support rich metadata filtering (e.g., filter by `doc_id`, `section_header`).
- Be self-hosted for HIPAA / data-residency compliance.
- Offer sparse vector support for future BM25-in-vector hybrid search.

## Decision

**Use Qdrant** (v1.9.2, self-hosted via Docker).

## Alternatives Considered

| Criterion | Qdrant | Pinecone | Chroma |
|---|---|---|---|
| Self-hosted | ✅ Yes | ❌ SaaS only | ✅ Yes |
| Payload filtering | ✅ Rich JSON filters | ✅ Metadata filters | ⚠️ Limited |
| Sparse vector support | ✅ Native (v1.7+) | ✅ Serverless plan | ❌ No |
| Language | Rust (fast, low memory) | Go | Python (slower) |
| Persistence | ✅ Disk + WAL | ✅ Managed | ✅ SQLite |
| Horizontal scaling | ✅ Distributed mode | ✅ Managed | ⚠️ Single node |
| Python SDK maturity | ✅ Excellent | ✅ Excellent | ✅ Good |
| License | Apache 2.0 | Proprietary | Apache 2.0 |

## Rationale

1. **Self-hosted for data sovereignty**: Clinical data cannot leave a controlled
   environment.  Pinecone's SaaS model is disqualified.
2. **Payload filtering**: Qdrant supports arbitrary JSON payload conditions,
   enabling per-patient, per-document, or per-section retrieval without
   post-filter re-ranking.
3. **Sparse vector support**: Qdrant's native sparse vector index (v1.7+) opens a
   path to replace the in-memory BM25 index with a fully integrated hybrid search
   without a separate retrieval service.
4. **Rust performance**: Lower memory footprint and higher QPS than Python-based
   stores at equivalent hardware.
5. **WAL durability**: Write-ahead log guarantees no data loss on ungraceful
   shutdown.

## Consequences

- Team must operate Qdrant as part of the infrastructure (Docker, Kubernetes).
- Qdrant's distributed mode (replication) requires a paid or community setup;
  single-node is sufficient for ≤50 M vectors.
- Future migration path to sparse+dense hybrid is clear without swapping stores.
